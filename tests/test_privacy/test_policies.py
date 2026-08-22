"""``privacy/policies.py`` 的单元测试：可插拔脱敏策略。

覆盖：MaskPolicy 构造与校验、build_masker、策略组合（combine）、PolicyMasker
的租户登记/解析/分发/回退/禁用短路（每个公开方法均含正向 + 边界/异常用例）。
外部依赖全部 mock/fake，无需真实中间件即可跑通。
"""

from __future__ import annotations

import pydantic
import pytest

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.privacy import (
    BUILTIN_RULES,
    BUILTIN_RULES_BY_NAME,
    DEFAULT_MASKED_FIELDS,
    ID_CARD_RULE,
    PHONE_RULE,
    FieldMasker,
    MaskPolicy,
    PolicyConfig,
    PolicyMasker,
    default_policy,
)


def make_message(content: str, **overrides: object) -> MedMessage:
    """构造测试用消息，默认字段均合法。"""
    payload: dict[str, object] = {
        "session_id": "s-1",
        "tenant_id": "t-1",
        "dept_id": "d-1",
        "patient_id": "p-1",
        "role": MessageRole.PATIENT,
        "content": content,
    }
    payload.update(overrides)
    return MedMessage(**payload)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# MaskPolicy 模型与校验
# --------------------------------------------------------------------------- #
class TestMaskPolicyModel:
    def test_construct_uses_builtins_by_default(self) -> None:
        """正向：不传规则/字段时使用内置全集与默认字段，且默认启用。"""
        policy = MaskPolicy(name="default-ish")
        assert policy.rules == BUILTIN_RULES
        assert policy.fields == DEFAULT_MASKED_FIELDS
        assert policy.enabled is True

    def test_construct_custom_rules_fields_disabled(self) -> None:
        """正向：自定义规则集、字段与禁用开关。"""
        policy = MaskPolicy(
            name="custom", rules=(PHONE_RULE,), fields=("content", "note"), enabled=False
        )
        assert policy.rules == (PHONE_RULE,)
        assert policy.fields == ("content", "note")
        assert policy.enabled is False

    def test_empty_rules_rejected(self) -> None:
        """异常：规则集不可为空。"""
        with pytest.raises(pydantic.ValidationError, match="at least one mask rule"):
            MaskPolicy(name="x", rules=())

    def test_empty_fields_rejected(self) -> None:
        """异常：字段白名单不可为空。"""
        with pytest.raises(pydantic.ValidationError, match="at least one target field"):
            MaskPolicy(name="x", fields=())

    def test_empty_field_name_rejected(self) -> None:
        """边界：字段名不可含空字符串。"""
        with pytest.raises(pydantic.ValidationError, match="non-empty"):
            MaskPolicy(name="x", fields=("content", ""))

    def test_build_masker_carries_rules_and_fields(self) -> None:
        """正向：build_masker 产出的引擎持有本策略的规则与字段。"""
        policy = MaskPolicy(name="p", rules=(PHONE_RULE, ID_CARD_RULE), fields=("content",))
        masker = policy.build_masker()
        assert isinstance(masker, FieldMasker)
        assert set(masker.rules) == {PHONE_RULE, ID_CARD_RULE}
        assert masker.fields == ("content",)

    def test_build_masker_rejects_duplicate_rule_names(self) -> None:
        """边界：策略内规则名重复时，build_masker 抛出校验错误。"""
        policy = MaskPolicy(name="dup", rules=(PHONE_RULE, PHONE_RULE))
        with pytest.raises(ValidationError, match="duplicated mask rule names"):
            policy.build_masker()


# --------------------------------------------------------------------------- #
# 按规则名的配置驱动构建（from_rule_names）
# --------------------------------------------------------------------------- #
class TestPolicyFromRuleNames:
    def test_builtin_index_covers_all_rules(self) -> None:
        """正向：名称索引与内置规则集一一对应。"""
        assert set(BUILTIN_RULES_BY_NAME) == {rule.name for rule in BUILTIN_RULES}
        assert BUILTIN_RULES_BY_NAME["phone"] is PHONE_RULE

    def test_select_rules_by_name(self) -> None:
        """正向：按名称挑选规则，顺序与入参一致。"""
        policy = MaskPolicy.from_rule_names("cfg", ["phone", "id_card"])
        assert policy.rules == (PHONE_RULE, ID_CARD_RULE)
        assert policy.fields == DEFAULT_MASKED_FIELDS
        assert policy.enabled is True

    def test_custom_fields_and_disabled(self) -> None:
        """正向：字段与启用开关可透传。"""
        policy = MaskPolicy.from_rule_names(
            "cfg", ["phone"], fields=["content", "note"], enabled=False
        )
        assert policy.fields == ("content", "note")
        assert policy.enabled is False

    def test_duplicate_names_are_deduped(self) -> None:
        """边界：重复规则名自动去重，不会触发 build_masker 重名报错。"""
        policy = MaskPolicy.from_rule_names("cfg", ["phone", "phone"])
        assert policy.rules == (PHONE_RULE,)
        assert isinstance(policy.build_masker(), FieldMasker)

    def test_unknown_rule_name_rejected(self) -> None:
        """异常：未知规则名被拒绝，并提示可用规则。"""
        with pytest.raises(ValidationError, match="unknown builtin mask rule"):
            MaskPolicy.from_rule_names("cfg", ["phone", "no_such_rule"])

    def test_empty_rule_names_rejected(self) -> None:
        """异常：规则名为空集合时拒绝。"""
        with pytest.raises(ValidationError, match="at least one mask rule"):
            MaskPolicy.from_rule_names("cfg", [])


# --------------------------------------------------------------------------- #
# 租户策略配置（PolicyConfig）
# --------------------------------------------------------------------------- #
class TestPolicyConfig:
    def test_defaults_cover_all_builtin_rules(self) -> None:
        """正向：默认配置覆盖全部内置规则与默认字段。"""
        cfg = PolicyConfig()
        assert set(cfg.rules) == set(BUILTIN_RULES_BY_NAME)
        assert cfg.fields == DEFAULT_MASKED_FIELDS
        assert cfg.enabled is True

    def test_to_policy_resolves_rules(self) -> None:
        """正向：配置解析为可执行策略，名称取入参。"""
        policy = PolicyConfig(rules=("phone",)).to_policy("hospital_a")
        assert policy.name == "hospital_a"
        assert policy.rules == (PHONE_RULE,)

    def test_to_policy_unknown_rule_rejected(self) -> None:
        """异常：配置含未知规则名时解析失败。"""
        with pytest.raises(ValidationError, match="unknown builtin mask rule"):
            PolicyConfig(rules=("ghost_rule",)).to_policy("t")

    def test_extra_key_rejected(self) -> None:
        """边界：配置含未知键时被 pydantic 拒绝。"""
        with pytest.raises(pydantic.ValidationError):
            PolicyConfig.model_validate({"rules": ["phone"], "typo": 1})


# --------------------------------------------------------------------------- #
# 配置驱动装配（PolicyMasker.from_config）
# --------------------------------------------------------------------------- #
class TestPolicyMaskerFromConfig:
    def test_from_config_with_plain_dicts(self) -> None:
        """正向：普通字典配置逐租户生效，策略互不干扰。"""
        mgr = PolicyMasker.from_config(
            {
                "hospital_a": {"rules": ["phone"]},
                "hospital_b": {"rules": ["id_card"]},
            }
        )
        text = "电话13812345678 身份证11010119900307123X"
        out_a = mgr.mask_text("hospital_a", text)
        out_b = mgr.mask_text("hospital_b", text)
        assert "138****5678" in out_a and "11010119900307123X" in out_a
        assert "110101********123X" in out_b and "13812345678" in out_b

    def test_from_config_with_policy_config_instances(self) -> None:
        """正向：直接传 PolicyConfig 实例同样生效。"""
        mgr = PolicyMasker.from_config({"hospital_a": PolicyConfig(rules=("phone",))})
        assert mgr.get_policy("hospital_a").rules == (PHONE_RULE,)

    def test_from_config_honours_enabled_flag(self) -> None:
        """正向：配置中的 enabled=False 使该租户短路不脱敏。"""
        mgr = PolicyMasker.from_config({"t": {"rules": ["phone"], "enabled": False}})
        assert mgr.is_enabled("t") is False
        assert mgr.mask_text("t", "13812345678") == "13812345678"

    def test_from_config_empty_falls_back_to_default(self) -> None:
        """边界：空配置时所有租户走默认策略。"""
        mgr = PolicyMasker.from_config({})
        assert mgr.snapshot() == {}
        assert mgr.mask_text("anyone", "13812345678") == "138****5678"

    def test_from_config_custom_default(self) -> None:
        """正向：可指定未登记租户的默认策略。"""
        fallback = MaskPolicy(name="fallback", rules=(ID_CARD_RULE,))
        mgr = PolicyMasker.from_config({"t": {"rules": ["phone"]}}, default=fallback)
        assert mgr.get_policy("other") is fallback
        assert mgr.mask_text("other", "13812345678") == "13812345678"

    def test_from_config_unknown_rule_rejected(self) -> None:
        """异常：配置含未知规则名时装配失败。"""
        with pytest.raises(ValidationError, match="unknown builtin mask rule"):
            PolicyMasker.from_config({"t": {"rules": ["nope"]}})

    def test_from_config_empty_tenant_id_rejected(self) -> None:
        """异常：空租户 ID 被拒绝。"""
        with pytest.raises(ValidationError, match="non-empty"):
            PolicyMasker.from_config({"": {"rules": ["phone"]}})

    def test_from_config_invalid_entry_rejected(self) -> None:
        """边界：配置结构非法（含未知键）时被 pydantic 拒绝。"""
        with pytest.raises(pydantic.ValidationError):
            PolicyMasker.from_config({"t": {"rules": ["phone"], "unexpected": True}})


# --------------------------------------------------------------------------- #
# 策略组合（combine）
# --------------------------------------------------------------------------- #
class TestPolicyCombine:
    def test_combine_unions_rules_and_fields(self) -> None:
        """正向：组合取规则并集与字段并集。"""
        a = MaskPolicy(name="a", rules=(PHONE_RULE,), fields=("content",))
        b = MaskPolicy(name="b", rules=(ID_CARD_RULE,), fields=("note",))
        combined = MaskPolicy.combine(a, b, name="ab")
        assert set(combined.rules) == {PHONE_RULE, ID_CARD_RULE}
        assert combined.fields == ("content", "note")
        assert combined.name == "ab"

    def test_combine_dedups_rules_by_name(self) -> None:
        """边界：规则按名称去重，不重复计数。"""
        a = MaskPolicy(name="a", rules=(PHONE_RULE, ID_CARD_RULE))
        b = MaskPolicy(name="b", rules=(PHONE_RULE,))
        combined = MaskPolicy.combine(a, b)
        assert len(combined.rules) == 2

    def test_combine_enabled_is_logical_or(self) -> None:
        """边界：组合策略任一启用即视为启用（逻辑或）。"""
        off = MaskPolicy(name="off", rules=(PHONE_RULE,), enabled=False)
        on = MaskPolicy(name="on", rules=(ID_CARD_RULE,))
        assert MaskPolicy.combine(off, on).enabled is True
        assert MaskPolicy.combine(on, on).enabled is True

    def test_combine_all_disabled(self) -> None:
        """边界：全部禁用时组合策略亦禁用。"""
        a = MaskPolicy(name="a", rules=(PHONE_RULE,), enabled=False)
        b = MaskPolicy(name="b", rules=(ID_CARD_RULE,), enabled=False)
        assert MaskPolicy.combine(a, b).enabled is False

    def test_combine_preserves_field_order(self) -> None:
        """正向：字段并集保序（先出现的在前）。"""
        a = MaskPolicy(name="a", rules=(PHONE_RULE,), fields=("content",))
        b = MaskPolicy(name="b", rules=(ID_CARD_RULE,), fields=("note", "content"))
        combined = MaskPolicy.combine(a, b)
        assert combined.fields == ("content", "note")

    def test_combine_requires_at_least_one(self) -> None:
        """异常：未提供任何策略时拒绝。"""
        with pytest.raises(ValidationError, match="at least one policy"):
            MaskPolicy.combine()


# --------------------------------------------------------------------------- #
# PolicyMasker 租户登记与分发
# --------------------------------------------------------------------------- #
class TestPolicyMaskerManager:
    def test_register_and_get_policy(self) -> None:
        """正向：登记后按租户取回对应策略。"""
        policy = MaskPolicy(name="tenant-a", rules=(PHONE_RULE,))
        mgr = PolicyMasker().register("hospital_a", policy)
        assert mgr.get_policy("hospital_a") is policy

    def test_register_returns_self_for_chaining(self) -> None:
        """边界：register 返回自身以支持链式调用。"""
        mgr = PolicyMasker()
        assert mgr.register("t", MaskPolicy(name="t")) is mgr

    def test_unregister_returns_true_when_registered(self) -> None:
        """正向：移除已登记租户返回 True。"""
        mgr = PolicyMasker().register("t", MaskPolicy(name="t"))
        assert mgr.unregister("t") is True
        assert mgr.is_registered("t") is False

    def test_unregister_returns_false_when_absent(self) -> None:
        """边界：移除不存在的租户返回 False。"""
        assert PolicyMasker().unregister("nope") is False

    def test_unregistered_tenant_falls_back_to_default(self) -> None:
        """正向：未登记租户解析到默认策略。"""
        mgr = PolicyMasker()
        assert mgr.get_policy("unknown") is mgr.default
        masker = mgr.resolve("unknown")
        assert isinstance(masker, FieldMasker)
        # 默认策略引擎对手机号脱敏生效。
        assert masker.mask_text("13812345678").text == "138****5678"

    def test_is_registered_false_for_unknown(self) -> None:
        """边界：未知租户未登记。"""
        assert PolicyMasker().is_registered("ghost") is False

    def test_register_empty_tenant_rejected(self) -> None:
        """异常：空租户 ID 登记被拒绝。"""
        with pytest.raises(ValidationError, match="non-empty"):
            PolicyMasker().register("", MaskPolicy(name="x"))

    def test_resolve_returns_field_masker(self) -> None:
        """正向：resolve 返回 FieldMasker 实例。"""
        mgr = PolicyMasker().register("hospital_a", MaskPolicy(name="a", rules=(PHONE_RULE,)))
        assert isinstance(mgr.resolve("hospital_a"), FieldMasker)

    def test_mask_message_applies_tenant_policy(self) -> None:
        """正向：租户策略仅对 phone 生效，id_card 不受影响。"""
        mgr = PolicyMasker().register("hospital_a", MaskPolicy(name="a", rules=(PHONE_RULE,)))
        msg = make_message("电话13812345678 身份证11010119900307123X")
        masked = mgr.mask_message("hospital_a", msg)
        assert "138****5678" in masked.content
        assert "11010119900307123X" in masked.content  # id_card 规则未启用
        assert masked.masked is True

    def test_mask_message_unregistered_tenant_masks_all(self) -> None:
        """正向：未登记租户走默认策略，覆盖全部内置规则。"""
        mgr = PolicyMasker()
        msg = make_message("电话13812345678 身份证11010119900307123X")
        masked = mgr.mask_message("unknown", msg)
        assert "138****5678" in masked.content
        assert "110101********123X" in masked.content

    def test_mask_text_applies_tenant_policy(self) -> None:
        """正向：mask_text 同样按租户策略派发。"""
        mgr = PolicyMasker().register("hospital_a", MaskPolicy(name="a", rules=(ID_CARD_RULE,)))
        out = mgr.mask_text("hospital_a", "身份证11010119900307123X 电话13812345678")
        assert "110101********123X" in out
        assert "13812345678" in out  # 手机号规则未启用

    def test_mask_messages_batch(self) -> None:
        """正向：批量脱敏保持顺序与逐条策略。"""
        mgr = PolicyMasker().register("hospital_a", MaskPolicy(name="a", rules=(PHONE_RULE,)))
        msgs = [
            make_message("电话13812345678"),
            make_message("电话13900000000"),
        ]
        out = mgr.mask_messages("hospital_a", msgs)
        assert out[0].content == "电话138****5678"
        assert out[1].content == "电话139****0000"

    def test_disabled_policy_short_circuits_mask_message(self) -> None:
        """边界：策略禁用时 mask_message 原样返回（不脱敏、不打标）。"""
        mgr = PolicyMasker().register(
            "hospital_a", MaskPolicy(name="a", rules=(PHONE_RULE,), enabled=False)
        )
        msg = make_message("电话13812345678")
        result = mgr.mask_message("hospital_a", msg)
        assert result is msg
        assert result.masked is False

    def test_disabled_policy_short_circuits_mask_text(self) -> None:
        """边界：策略禁用时 mask_text 原样返回。"""
        mgr = PolicyMasker().register(
            "hospital_a", MaskPolicy(name="a", rules=(PHONE_RULE,), enabled=False)
        )
        assert mgr.mask_text("hospital_a", "13812345678") == "13812345678"

    def test_is_enabled_reflects_policy(self) -> None:
        """正向：is_enabled 反映租户策略开关（含默认）。"""
        mgr = PolicyMasker().register("on", MaskPolicy(name="on", rules=(PHONE_RULE,)))
        mgr.register("off", MaskPolicy(name="off", rules=(PHONE_RULE,), enabled=False))
        assert mgr.is_enabled("on") is True
        assert mgr.is_enabled("off") is False
        assert mgr.is_enabled("unknown") is True  # 默认启用

    def test_disabled_default_policy_short_circuits(self) -> None:
        """边界：默认策略禁用时，所有租户均不脱敏。"""
        mgr = PolicyMasker(default=MaskPolicy(name="off", enabled=False))
        msg = make_message("电话13812345678")
        assert mgr.mask_message("any", msg) is msg
        assert mgr.mask_text("any", "13812345678") == "13812345678"

    def test_default_property(self) -> None:
        """正向：default 属性返回构造时设定的默认策略。"""
        custom = MaskPolicy(name="custom-default", rules=(PHONE_RULE,))
        mgr = PolicyMasker(default=custom)
        assert mgr.default is custom

    def test_snapshot_returns_copy(self) -> None:
        """边界：snapshot 返回登记副本，外部修改不影响内部。"""
        mgr = PolicyMasker().register("t", MaskPolicy(name="t"))
        snap = mgr.snapshot()
        snap["t"] = MaskPolicy(name="other")
        assert mgr.get_policy("t").name == "t"

    def test_default_policy_helper_enabled(self) -> None:
        """正向：default_policy() 便捷函数返回启用状态的内置策略。"""
        policy = default_policy()
        assert policy.enabled is True
        assert policy.rules == BUILTIN_RULES
        assert policy.build_masker().mask_text("13812345678").text == "138****5678"
