"""可插拔脱敏策略（策略模式）。

D21 提供了字段级正则脱敏引擎 :class:`FieldMasker`。本模块在其之上引入
**策略层**，使得「使用哪一套规则、作用于哪些字段」可以按租户（tenant）灵活
配置并热插拔：

* :class:`MaskPolicy` 是对单套脱敏策略的不可变描述（规则集 + 字段白名单 +
  ``enabled`` 开关），可借此组合出不同租户的差异策略；
* :class:`PolicyMasker` 是按租户登记的策略注册表与分发器：给定 ``tenant_id``
  即解析出对应的 :class:`FieldMasker`，未显式登记的租户回退到默认策略。

所有脱敏仍是纯正则规则改写，**不含任何分词、实体识别或语义解析逻辑**。

典型用法::

    # 某租户只关心手机号脱敏
    phone_only = MaskPolicy(name="tenant-a", rules=(PHONE_RULE,))
    mgr = PolicyMasker().register("hospital_a", phone_only)
    masked = mgr.mask_message("hospital_a", message)

    # 组合两套策略（规则并集 + 字段并集）
    combined = MaskPolicy.combine(phone_only, MaskPolicy(name="b", fields=("content", "note")))
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.privacy.masker import (
    BUILTIN_RULES,
    DEFAULT_MASKED_FIELDS,
    FieldMasker,
    MaskRule,
)

#: 内置规则按名称索引，供「配置里只写规则名」的配置驱动策略构建使用。
BUILTIN_RULES_BY_NAME: dict[str, MaskRule] = {rule.name: rule for rule in BUILTIN_RULES}


class MaskPolicy(BaseModel):
    """单套脱敏策略的不可变描述。

    策略决定「用哪些规则 + 脱敏哪些字段」。它只是配置载体，真正的改写由
    :meth:`build_masker` 产出的 :class:`FieldMasker` 完成。多个租户可持有
    不同 ``MaskPolicy`` 实例，从而实现按租户差异化的脱敏策略。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    rules: tuple[MaskRule, ...] = Field(default_factory=lambda: BUILTIN_RULES)
    fields: tuple[str, ...] = Field(default_factory=lambda: DEFAULT_MASKED_FIELDS)
    enabled: bool = True

    @field_validator("rules")
    @classmethod
    def _non_empty_rules(cls, v: tuple[MaskRule, ...]) -> tuple[MaskRule, ...]:
        """规则集不可为空。"""
        if not v:
            raise ValueError("policy requires at least one mask rule")
        return v

    @field_validator("fields")
    @classmethod
    def _non_empty_fields(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        """字段白名单不可为空，且字段名必须非空。"""
        if not v:
            raise ValueError("policy requires at least one target field")
        if any(not field for field in v):
            raise ValueError("policy target field name must be non-empty")
        return v

    def build_masker(self) -> FieldMasker:
        """根据策略配置构建一个 :class:`FieldMasker`。

        Returns:
            绑定了本策略规则集与字段白名单的脱敏引擎。

        Raises:
            ValidationError: 规则名重复（由 :class:`FieldMasker` 校验抛出）。
        """
        return FieldMasker(rules=list(self.rules), fields=list(self.fields))

    @classmethod
    def from_rule_names(
        cls,
        name: str,
        rule_names: Iterable[str],
        *,
        fields: Iterable[str] | None = None,
        enabled: bool = True,
    ) -> MaskPolicy:
        """按内置规则名构建策略（配置驱动入口）。

        外部配置（JSON/YAML/环境变量）中只需写规则名字符串，由本方法解析为
        真正的 :class:`MaskRule`，避免把正则写进配置文件。重复规则名自动去重。

        Args:
            name: 策略名称。
            rule_names: 内置规则名，取值见 :data:`BUILTIN_RULES_BY_NAME`。
            fields: 纳管字段名，``None`` 时用 :data:`DEFAULT_MASKED_FIELDS`。
            enabled: 是否启用该策略。

        Returns:
            解析后的策略实例。

        Raises:
            ValidationError: 规则名未知，或解析后规则集为空时。
        """
        selected: list[MaskRule] = []
        for rule_name in rule_names:
            rule = BUILTIN_RULES_BY_NAME.get(rule_name)
            if rule is None:
                available = ", ".join(sorted(BUILTIN_RULES_BY_NAME))
                raise ValidationError(
                    f"unknown builtin mask rule: {rule_name!r}; available: {available}"
                )
            if rule not in selected:
                selected.append(rule)
        if not selected:
            raise ValidationError("policy requires at least one mask rule")
        return cls(
            name=name,
            rules=tuple(selected),
            fields=tuple(fields) if fields is not None else DEFAULT_MASKED_FIELDS,
            enabled=enabled,
        )

    @classmethod
    def combine(cls, *policies: MaskPolicy, name: str = "combined") -> MaskPolicy:
        """组合多套策略，规则取并集、字段取并集。

        规则以 ``name`` 去重（后出现的覆盖先出现的），字段保序去重。
        组合后的 ``enabled`` 为各策略的逻辑或——任一策略启用即视为启用。

        Args:
            *policies: 待组合的策略，至少 1 个。
            name: 组合策略的名称。

        Returns:
            合并后的新策略。

        Raises:
            ValidationError: 未提供任何策略时。
        """
        if not policies:
            raise ValidationError("combine requires at least one policy")
        merged_rules: dict[str, MaskRule] = {}
        merged_fields: list[str] = []
        enabled = False
        for policy in policies:
            enabled = enabled or policy.enabled
            for rule in policy.rules:
                merged_rules[rule.name] = rule
            for field in policy.fields:
                if field not in merged_fields:
                    merged_fields.append(field)
        return cls(
            name=name,
            rules=tuple(merged_rules.values()),
            fields=tuple(merged_fields),
            enabled=enabled,
        )


def default_policy() -> MaskPolicy:
    """返回使用全部内置规则、作用于 ``content`` 字段的默认策略（进程级共享）。"""
    return MaskPolicy(
        name="default", rules=BUILTIN_RULES, fields=DEFAULT_MASKED_FIELDS, enabled=True
    )


class PolicyConfig(BaseModel):
    """单个租户的脱敏策略配置（纯数据，可直接由 JSON/YAML 反序列化）。

    与 :class:`MaskPolicy` 的区别：本模型的 ``rules`` 是**规则名字符串**，正则
    留在代码内置规则里，配置方只做取舍不写正则。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    rules: tuple[str, ...] = Field(default_factory=lambda: tuple(BUILTIN_RULES_BY_NAME))
    fields: tuple[str, ...] = Field(default_factory=lambda: DEFAULT_MASKED_FIELDS)
    enabled: bool = True

    def to_policy(self, name: str) -> MaskPolicy:
        """把配置解析为可执行策略。

        Args:
            name: 生成策略的名称（通常取租户 ID）。

        Returns:
            解析后的 :class:`MaskPolicy`。

        Raises:
            ValidationError: 配置中出现未知规则名，或规则集为空时。
        """
        return MaskPolicy.from_rule_names(
            name, self.rules, fields=self.fields, enabled=self.enabled
        )


class PolicyMasker:
    """按租户登记的脱敏策略注册表与分发器（策略模式上下文）。

    未显式登记的租户一律回退到 :meth:`default_policy`；策略可按租户热插拔，
    解析结果带缓存（策略对象不可变，可安全复用）。
    """

    def __init__(self, default: MaskPolicy | None = None) -> None:
        """构造策略分发器。

        Args:
            default: 默认策略，``None`` 时使用 :func:`default_policy`。
        """
        self._default = default if default is not None else default_policy()
        self._policies: dict[str, MaskPolicy] = {}
        # 策略对象不可变且可哈希，按策略实例缓存脱敏引擎。
        self._cache: dict[MaskPolicy, FieldMasker] = {}

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, PolicyConfig | Mapping[str, object]],
        *,
        default: MaskPolicy | None = None,
    ) -> PolicyMasker:
        """从「租户 ID → 策略配置」映射一次性装配分发器。

        配置项既可是 :class:`PolicyConfig` 实例，也可是等价的普通字典（会经
        pydantic 校验），例如::

            PolicyMasker.from_config({
                "hospital_a": {"rules": ["phone"]},
                "hospital_b": {"rules": ["phone", "id_card"], "enabled": False},
            })

        Args:
            config: 租户 ID 到策略配置的映射；空映射表示所有租户走默认策略。
            default: 未登记租户使用的默认策略，``None`` 时用 :func:`default_policy`。

        Returns:
            已按配置完成登记的分发器。

        Raises:
            ValidationError: 租户 ID 为空、配置结构非法或含未知规则名时。
        """
        masker = cls(default=default)
        for tenant_id, entry in config.items():
            # 先校验租户 ID：它同时用作策略名，为空需按本层语义报错而非 pydantic 报错。
            if not tenant_id:
                raise ValidationError("tenant_id must be non-empty")
            parsed = (
                entry if isinstance(entry, PolicyConfig) else PolicyConfig.model_validate(entry)
            )
            masker.register(tenant_id, parsed.to_policy(tenant_id))
        return masker

    @property
    def default(self) -> MaskPolicy:
        """返回当前默认策略。"""
        return self._default

    def register(self, tenant_id: str, policy: MaskPolicy) -> PolicyMasker:
        """为某个租户登记脱敏策略（同名租户覆盖旧策略）。

        Args:
            tenant_id: 租户 ID，必须非空。
            policy: 该租户对应的脱敏策略。

        Returns:
            自身，便于链式调用。

        Raises:
            ValidationError: ``tenant_id`` 为空时。
        """
        if not tenant_id:
            raise ValidationError("tenant_id must be non-empty")
        self._policies[tenant_id] = policy
        self._cache.clear()
        return self

    def unregister(self, tenant_id: str) -> bool:
        """移除某租户的策略登记。

        Args:
            tenant_id: 待移除的租户 ID。

        Returns:
            移除成功（该租户此前已登记）返回 ``True``，否则 ``False``。
        """
        removed = self._policies.pop(tenant_id, None) is not None
        if removed:
            self._cache.clear()
        return removed

    def is_registered(self, tenant_id: str) -> bool:
        """判断某租户是否已显式登记策略。"""
        return tenant_id in self._policies

    def get_policy(self, tenant_id: str) -> MaskPolicy:
        """返回某租户对应的策略，未登记时回退默认策略。"""
        return self._policies.get(tenant_id, self._default)

    def is_enabled(self, tenant_id: str) -> bool:
        """返回某租户策略的启用状态（含默认策略）。"""
        return self.get_policy(tenant_id).enabled

    def _masker(self, policy: MaskPolicy) -> FieldMasker:
        """解析策略对应的脱敏引擎（带缓存）。"""
        cached = self._cache.get(policy)
        if cached is None:
            cached = policy.build_masker()
            self._cache[policy] = cached
        return cached

    def resolve(self, tenant_id: str) -> FieldMasker:
        """按租户解析出对应的 :class:`FieldMasker`。

        Args:
            tenant_id: 租户 ID；未登记时返回默认策略引擎。

        Returns:
            绑定该租户（或默认）策略的脱敏引擎。
        """
        return self._masker(self.get_policy(tenant_id))

    def mask_message(self, tenant_id: str, message: MedMessage) -> MedMessage:
        """按租户策略脱敏单条消息。

        策略 ``enabled=False`` 时原样返回（不脱敏、不打 ``masked`` 标记）。

        Args:
            tenant_id: 租户 ID。
            message: 原始消息。

        Returns:
            脱敏后的消息副本；策略禁用时返回入参本身。
        """
        policy = self.get_policy(tenant_id)
        if not policy.enabled:
            return message
        return self._masker(policy).mask_message(message)

    def mask_messages(self, tenant_id: str, messages: Iterable[MedMessage]) -> list[MedMessage]:
        """按租户策略批量脱敏消息，保持原有顺序。"""
        return [self.mask_message(tenant_id, message) for message in messages]

    def mask_text(self, tenant_id: str, text: str) -> str:
        """按租户策略脱敏一段自由文本。

        策略 ``enabled=False`` 时原样返回。

        Args:
            tenant_id: 租户 ID。
            text: 待脱敏文本。

        Returns:
            脱敏后的文本；策略禁用时返回入参本身。
        """
        policy = self.get_policy(tenant_id)
        if not policy.enabled:
            return text
        return self._masker(policy).mask_text(text).text

    def snapshot(self) -> dict[str, MaskPolicy]:
        """返回当前租户策略登记的只读副本（键为 tenant_id）。"""
        return dict(self._policies)
