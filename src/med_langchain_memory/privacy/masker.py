"""字段级正则脱敏引擎。

医疗会话中的隐私信息（手机号、身份证号、病历号、床号）通过 **纯正则规则**
在字段粒度上脱敏：:class:`MaskRule` 描述单条规则，:class:`FieldMasker` 负责
按「纳管字段白名单」派发规则并输出脱敏结果。

设计约束（合规硬要求）：
    * 只做正则匹配与字符替换，**不含任何分词、实体识别、语义解析逻辑**；
    * 规则可完整替换匹配片段，也可只脱敏其中一个捕获组（保留 ``床号:`` 等标签）；
    * 未纳管字段一律原样透传，避免误伤业务字段。

典型用法::

    masker = FieldMasker()
    masker.mask_text("患者电话 13812345678").text     # -> "患者电话 138****5678"
    masker.mask_message(message)                      # -> 脱敏副本，masked=True
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import ValidationError


@lru_cache(maxsize=256)
def _compile(pattern: str) -> re.Pattern[str]:
    """编译并缓存正则表达式（规则实例不可变，可安全共享编译结果）。"""
    return re.compile(pattern)


class MaskRule(BaseModel):
    """单条字段级正则脱敏规则。

    命中片段的改写方式二选一：给定 ``replacement`` 时整段替换为固定文本；
    否则保留 ``keep_prefix`` 个首字符与 ``keep_suffix`` 个尾字符、中间以
    ``mask_char`` 等长填充（保留长度不足时整段掩码）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, max_length=64)
    pattern: str = Field(min_length=1)
    group: int = Field(default=0, ge=0)
    keep_prefix: int = Field(default=0, ge=0)
    keep_suffix: int = Field(default=0, ge=0)
    mask_char: str = Field(default="*", min_length=1, max_length=1)
    replacement: str | None = None

    @field_validator("pattern")
    @classmethod
    def _validate_pattern(cls, v: str) -> str:
        """校验 pattern 为合法正则表达式。"""
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern: {exc}") from exc
        return v

    @model_validator(mode="after")
    def _validate_group(self) -> MaskRule:
        """校验 ``group`` 未超出 pattern 实际捕获组数量。"""
        groups = _compile(self.pattern).groups
        if self.group > groups:
            raise ValueError(f"group {self.group} out of range: pattern has {groups} group(s)")
        return self

    @property
    def regex(self) -> re.Pattern[str]:
        """返回该规则编译后的正则对象。"""
        return _compile(self.pattern)

    def substitute(self, matched: str) -> str:
        """计算单个命中片段的脱敏结果。

        Args:
            matched: 正则命中的原始片段。

        Returns:
            脱敏后的片段文本。
        """
        if self.replacement is not None:
            return self.replacement
        if self.keep_prefix + self.keep_suffix >= len(matched):
            return self.mask_char * len(matched)
        head = matched[: self.keep_prefix]
        tail = matched[len(matched) - self.keep_suffix :] if self.keep_suffix else ""
        body = self.mask_char * (len(matched) - self.keep_prefix - self.keep_suffix)
        return head + body + tail

    def apply(self, value: str) -> tuple[str, int]:
        """对文本应用本规则。

        Args:
            value: 待脱敏文本。

        Returns:
            ``(脱敏后文本, 命中次数)``；未命中时文本原样返回、次数为 0。
        """
        hits = 0

        def _replace(match: re.Match[str]) -> str:
            nonlocal hits
            whole = match.group(0)
            target = match.group(self.group)
            if not target:
                return whole
            hits += 1
            masked = self.substitute(target)
            if self.group == 0:
                return masked
            start, end = match.span(self.group)
            offset = match.start()
            return whole[: start - offset] + masked + whole[end - offset :]

        return self.regex.sub(_replace, value), hits


#: 中国大陆手机号：保留前 3 位与后 4 位（``138****5678``）。
PHONE_RULE = MaskRule(
    name="phone",
    pattern=r"(?<!\d)1[3-9]\d{9}(?!\d)",
    keep_prefix=3,
    keep_suffix=4,
)

#: 18 位居民身份证号：保留前 6 位地址码与后 4 位，掩码中间 8 位出生日期。
ID_CARD_RULE = MaskRule(
    name="id_card",
    pattern=(
        r"(?<![0-9A-Za-z])\d{6}(?:18|19|20)\d{2}(?:0[1-9]|1[0-2])"
        r"(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?![0-9A-Za-z])"
    ),
    keep_prefix=6,
    keep_suffix=4,
)

#: 带标签的病历号/住院号/门诊号：仅脱敏号码本体，保留标签，末 4 位可读。
MEDICAL_RECORD_RULE = MaskRule(
    name="medical_record_no",
    pattern=r"(?:病历号|病案号|住院号|门诊号)\s*[:：]?\s*([A-Za-z]{0,3}\d{6,12})",
    group=1,
    keep_suffix=4,
)

#: 带标签的床号（``床号: 12-3``）：仅脱敏号码本体，保留标签。
BED_NO_RULE = MaskRule(
    name="bed_no",
    pattern=r"(?:床号|床位)\s*[:：]?\s*(\d{1,4}(?:-\d{1,4})?)",
    group=1,
)

#: 后缀式床号（``12床``）：仅脱敏号码本体，保留「床」字。
BED_NO_SUFFIX_RULE = MaskRule(
    name="bed_no_suffix",
    pattern=r"(?<!\d)(\d{1,4}(?:-\d{1,4})?)床",
    group=1,
)

#: 内置规则集，按「具体优先」顺序应用（身份证先于手机号，避免长号被截断匹配）。
BUILTIN_RULES: tuple[MaskRule, ...] = (
    ID_CARD_RULE,
    PHONE_RULE,
    MEDICAL_RECORD_RULE,
    BED_NO_RULE,
    BED_NO_SUFFIX_RULE,
)

#: 默认纳管脱敏的字段名（消息正文）。
DEFAULT_MASKED_FIELDS: tuple[str, ...] = ("content",)


class MaskResult(BaseModel):
    """一次脱敏调用的结果。"""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    hits: dict[str, int] = Field(default_factory=dict)

    @property
    def changed(self) -> bool:
        """是否有任意规则命中。"""
        return bool(self.hits)

    @property
    def total_hits(self) -> int:
        """全部规则的命中次数之和。"""
        return sum(self.hits.values())


class FieldMasker:
    """字段级正则脱敏引擎。

    引擎持有一组 :class:`MaskRule` 与一份纳管字段白名单：只有白名单内的字段
    才会被扫描改写，其余字段原样透传。规则按给定顺序串行应用。
    """

    def __init__(
        self,
        rules: Sequence[MaskRule] | None = None,
        *,
        fields: Sequence[str] | None = None,
    ) -> None:
        """构造脱敏引擎。

        Args:
            rules: 规则集，默认使用 :data:`BUILTIN_RULES`。
            fields: 纳管字段名，默认 :data:`DEFAULT_MASKED_FIELDS`。

        Raises:
            ValidationError: 规则集或字段集为空、规则名重复、字段名为空时。
        """
        selected = tuple(rules) if rules is not None else BUILTIN_RULES
        if not selected:
            raise ValidationError("masker requires at least one mask rule")
        names = [rule.name for rule in selected]
        duplicated = sorted({name for name in names if names.count(name) > 1})
        if duplicated:
            raise ValidationError(f"duplicated mask rule names: {', '.join(duplicated)}")
        targets = tuple(fields) if fields is not None else DEFAULT_MASKED_FIELDS
        if not targets:
            raise ValidationError("masker requires at least one target field")
        if any(not field for field in targets):
            raise ValidationError("mask target field name must be non-empty")
        self._rules = selected
        self._fields = targets
        self._field_set = frozenset(targets)

    @property
    def rules(self) -> tuple[MaskRule, ...]:
        """返回引擎持有的规则集。"""
        return self._rules

    @property
    def fields(self) -> tuple[str, ...]:
        """返回纳管脱敏的字段名。"""
        return self._fields

    def mask_text(self, value: str) -> MaskResult:
        """对任意文本按序应用全部规则（不做字段白名单判断）。

        Args:
            value: 待脱敏文本。

        Returns:
            脱敏结果，``hits`` 仅包含实际命中的规则名。
        """
        text = value
        hits: dict[str, int] = {}
        for rule in self._rules:
            text, count = rule.apply(text)
            if count:
                hits[rule.name] = hits.get(rule.name, 0) + count
        return MaskResult(text=text, hits=hits)

    def mask_field(self, field: str, value: str) -> MaskResult:
        """按字段白名单脱敏单个字段值。

        Args:
            field: 字段名。
            value: 字段值。

        Returns:
            脱敏结果；字段未纳管时原样返回且 ``hits`` 为空。
        """
        if field not in self._field_set:
            return MaskResult(text=value)
        return self.mask_text(value)

    def mask_fields(self, data: Mapping[str, str]) -> dict[str, str]:
        """批量脱敏映射结构，仅改写纳管字段的值。

        Args:
            data: 字段名到字段值的映射。

        Returns:
            新字典，键顺序与入参一致。
        """
        return {key: self.mask_field(key, value).text for key, value in data.items()}

    def mask_message(self, message: MedMessage) -> MedMessage:
        """返回脱敏后的消息副本。

        正文按 ``content`` 字段规则脱敏，``metadata`` 逐键按白名单脱敏，
        并将 ``masked`` 置为 ``True`` 标记「已过脱敏引擎」。

        Args:
            message: 原始消息。

        Returns:
            脱敏后的新消息；入参 ``masked`` 已为 ``True`` 时原样返回（幂等）。
        """
        if message.masked:
            return message
        result = self.mask_field("content", message.content)
        return message.model_copy(
            update={
                "content": result.text,
                "metadata": self.mask_fields(message.metadata),
                "masked": True,
            }
        )

    def mask_messages(self, messages: Iterable[MedMessage]) -> list[MedMessage]:
        """批量脱敏消息，保持原有顺序。"""
        return [self.mask_message(message) for message in messages]


@lru_cache(maxsize=1)
def default_masker() -> FieldMasker:
    """返回使用全部内置规则的进程级共享脱敏引擎（惰性单例）。"""
    return FieldMasker()


def mask_text(value: str) -> str:
    """用内置规则脱敏一段文本的便捷函数。

    Args:
        value: 待脱敏文本。

    Returns:
        脱敏后的文本。
    """
    return default_masker().mask_text(value).text
