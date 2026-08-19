"""``privacy/masker.py`` 的单元测试：字段级正则脱敏引擎。

覆盖：规则模型构造与校验、单片段替换策略、捕获组定向脱敏、五条内置医疗规则、
脱敏结果模型、引擎构造校验、字段白名单隔离、消息脱敏幂等性与便捷函数
（每个公开方法均含正向 + 边界/异常用例）。
"""

from __future__ import annotations

import pydantic
import pytest

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.exceptions import MedMemoryError, ValidationError
from med_langchain_memory.privacy import (
    BED_NO_RULE,
    BED_NO_SUFFIX_RULE,
    BUILTIN_RULES,
    DEFAULT_MASKED_FIELDS,
    ID_CARD_RULE,
    MEDICAL_RECORD_RULE,
    PHONE_RULE,
    FieldMasker,
    MaskResult,
    MaskRule,
    default_masker,
    mask_text,
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
# MaskRule 模型与校验
# --------------------------------------------------------------------------- #
class TestMaskRuleModel:
    def test_construct_with_defaults(self) -> None:
        """正向：最小字段构造，默认值正确。"""
        rule = MaskRule(name="digits", pattern=r"\d+")
        assert rule.name == "digits"
        assert rule.group == 0
        assert rule.keep_prefix == 0
        assert rule.keep_suffix == 0
        assert rule.mask_char == "*"
        assert rule.replacement is None

    def test_frozen_model_rejects_mutation(self) -> None:
        """边界：规则不可变，赋值被拒绝。"""
        rule = MaskRule(name="digits", pattern=r"\d+")
        with pytest.raises(pydantic.ValidationError):
            rule.name = "other"  # type: ignore[misc]

    def test_invalid_regex_rejected(self) -> None:
        """异常：非法正则表达式在构造期被拒绝。"""
        with pytest.raises(pydantic.ValidationError, match="invalid regex pattern"):
            MaskRule(name="bad", pattern=r"(unclosed")

    def test_group_out_of_range_rejected(self) -> None:
        """异常：group 超出实际捕获组数量被拒绝。"""
        with pytest.raises(pydantic.ValidationError, match="out of range"):
            MaskRule(name="bad-group", pattern=r"\d+", group=1)

    def test_group_within_range_accepted(self) -> None:
        """正向：group 等于捕获组数量时合法。"""
        rule = MaskRule(name="ok-group", pattern=r"x(\d+)", group=1)
        assert rule.group == 1

    def test_empty_name_rejected(self) -> None:
        """边界：规则名不可为空。"""
        with pytest.raises(pydantic.ValidationError):
            MaskRule(name="", pattern=r"\d+")

    def test_multi_char_mask_char_rejected(self) -> None:
        """边界：掩码字符必须为单字符。"""
        with pytest.raises(pydantic.ValidationError):
            MaskRule(name="x", pattern=r"\d+", mask_char="**")

    def test_negative_keep_prefix_rejected(self) -> None:
        """边界：保留位数不可为负。"""
        with pytest.raises(pydantic.ValidationError):
            MaskRule(name="x", pattern=r"\d+", keep_prefix=-1)

    def test_extra_field_rejected(self) -> None:
        """边界：未定义字段被拒绝（extra=forbid）。"""
        with pytest.raises(pydantic.ValidationError):
            MaskRule(name="x", pattern=r"\d+", unknown="y")  # type: ignore[call-arg]

    def test_regex_property_is_cached(self) -> None:
        """正向：同一 pattern 复用编译结果。"""
        first = MaskRule(name="a", pattern=r"\d{4}")
        second = MaskRule(name="b", pattern=r"\d{4}")
        assert first.regex is second.regex


# --------------------------------------------------------------------------- #
# MaskRule.substitute
# --------------------------------------------------------------------------- #
class TestMaskRuleSubstitute:
    def test_keep_prefix_and_suffix(self) -> None:
        """正向：保留首尾并等长填充中间。"""
        rule = MaskRule(name="r", pattern=r"\d+", keep_prefix=2, keep_suffix=2)
        assert rule.substitute("123456") == "12**56"

    def test_keep_prefix_only(self) -> None:
        """正向：仅保留前缀。"""
        rule = MaskRule(name="r", pattern=r"\d+", keep_prefix=3)
        assert rule.substitute("123456") == "123***"

    def test_keep_longer_than_value_masks_all(self) -> None:
        """边界：保留位数不小于片段长度时整段掩码，不泄漏原文。"""
        rule = MaskRule(name="r", pattern=r"\d+", keep_prefix=5, keep_suffix=5)
        assert rule.substitute("1234") == "****"

    def test_replacement_takes_precedence(self) -> None:
        """正向：给定 replacement 时整段替换，忽略保留位数。"""
        rule = MaskRule(name="r", pattern=r"\d+", keep_prefix=2, replacement="[REDACTED]")
        assert rule.substitute("123456") == "[REDACTED]"

    def test_custom_mask_char(self) -> None:
        """正向：自定义掩码字符。"""
        rule = MaskRule(name="r", pattern=r"\d+", mask_char="#")
        assert rule.substitute("12") == "##"

    def test_empty_input_returns_empty(self) -> None:
        """边界：空片段返回空串。"""
        rule = MaskRule(name="r", pattern=r"\d*")
        assert rule.substitute("") == ""


# --------------------------------------------------------------------------- #
# MaskRule.apply
# --------------------------------------------------------------------------- #
class TestMaskRuleApply:
    def test_single_match(self) -> None:
        """正向：命中一次并返回命中计数。"""
        rule = MaskRule(name="r", pattern=r"\d{4}")
        text, hits = rule.apply("code 1234 end")
        assert text == "code **** end"
        assert hits == 1

    def test_multiple_matches_counted(self) -> None:
        """正向：多次命中全部替换并累计计数。"""
        rule = MaskRule(name="r", pattern=r"\d{2}")
        text, hits = rule.apply("11 22 33")
        assert text == "** ** **"
        assert hits == 3

    def test_no_match_returns_original(self) -> None:
        """边界：未命中时原样返回、计数为 0。"""
        rule = MaskRule(name="r", pattern=r"\d+")
        text, hits = rule.apply("no digits here")
        assert text == "no digits here"
        assert hits == 0

    def test_group_masking_preserves_surrounding_text(self) -> None:
        """正向：group 定向脱敏，保留标签等上下文。"""
        rule = MaskRule(name="r", pattern=r"(?:编号)[:：]?(\d+)", group=1)
        text, hits = rule.apply("编号:98765 已登记")
        assert text == "编号:***** 已登记"
        assert hits == 1

    def test_group_masking_with_multiple_groups(self) -> None:
        """正向：多捕获组时只改写目标组。"""
        rule = MaskRule(name="r", pattern=r"(\w+)-(\d+)", group=2)
        text, hits = rule.apply("bed-1024")
        assert text == "bed-****"
        assert hits == 1

    def test_unmatched_optional_group_skipped(self) -> None:
        """边界：可选捕获组未参与匹配时不计命中、不改写。"""
        rule = MaskRule(name="r", pattern=r"A(\d+)?", group=1)
        text, hits = rule.apply("A")
        assert text == "A"
        assert hits == 0

    def test_empty_text(self) -> None:
        """边界：空文本安全返回。"""
        rule = MaskRule(name="r", pattern=r"\d+")
        assert rule.apply("") == ("", 0)


# --------------------------------------------------------------------------- #
# 内置医疗规则
# --------------------------------------------------------------------------- #
class TestBuiltinRules:
    def test_rule_names_are_unique(self) -> None:
        """正向：内置规则名唯一。"""
        names = [rule.name for rule in BUILTIN_RULES]
        assert len(names) == len(set(names)) == 5

    def test_phone_masked(self) -> None:
        """正向：手机号保留前 3 后 4。"""
        assert PHONE_RULE.apply("13812345678") == ("138****5678", 1)

    def test_phone_ignores_non_mobile_prefix(self) -> None:
        """边界：非 1[3-9] 开头的 11 位数字不命中。"""
        assert PHONE_RULE.apply("12812345678") == ("12812345678", 0)

    def test_phone_ignores_longer_digit_run(self) -> None:
        """边界：更长数字串（如身份证）不被误判为手机号。"""
        text, hits = PHONE_RULE.apply("11010119900307123X")
        assert hits == 0
        assert text == "11010119900307123X"

    def test_id_card_masked(self) -> None:
        """正向：身份证保留前 6 后 4，掩码中间出生日期。"""
        assert ID_CARD_RULE.apply("11010119900307123X") == ("110101********123X", 1)

    def test_id_card_rejects_invalid_month(self) -> None:
        """边界：非法月份（13 月）不命中。"""
        assert ID_CARD_RULE.apply("110101199013071234")[1] == 0

    def test_medical_record_keeps_label(self) -> None:
        """正向：病历号保留标签与末 4 位。"""
        text, hits = MEDICAL_RECORD_RULE.apply("病历号：ZY20240001")
        assert text == "病历号：******0001"
        assert hits == 1

    def test_medical_record_requires_label(self) -> None:
        """边界：无标签的裸数字不命中（字段级规则不做上下文猜测）。"""
        assert MEDICAL_RECORD_RULE.apply("ZY20240001")[1] == 0

    def test_bed_no_labeled_masked(self) -> None:
        """正向：带标签床号仅脱敏号码本体。"""
        assert BED_NO_RULE.apply("床号: 12-3") == ("床号: ****", 1)

    def test_bed_no_requires_label(self) -> None:
        """边界：无床号标签时不命中。"""
        assert BED_NO_RULE.apply("12-3")[1] == 0

    def test_bed_no_suffix_masked(self) -> None:
        """正向：后缀式床号保留「床」字。"""
        assert BED_NO_SUFFIX_RULE.apply("12床患者") == ("**床患者", 1)

    def test_bed_no_suffix_ignores_plain_digits(self) -> None:
        """边界：无「床」后缀时不命中。"""
        assert BED_NO_SUFFIX_RULE.apply("12 患者")[1] == 0


# --------------------------------------------------------------------------- #
# MaskResult
# --------------------------------------------------------------------------- #
class TestMaskResult:
    def test_hits_aggregation(self) -> None:
        """正向：changed 与 total_hits 反映命中情况。"""
        result = MaskResult(text="x", hits={"phone": 2, "id_card": 1})
        assert result.changed is True
        assert result.total_hits == 3

    def test_empty_hits(self) -> None:
        """边界：无命中时 changed 为 False、总数为 0。"""
        result = MaskResult(text="x")
        assert result.changed is False
        assert result.total_hits == 0

    def test_extra_field_rejected(self) -> None:
        """边界：未定义字段被拒绝。"""
        with pytest.raises(pydantic.ValidationError):
            MaskResult(text="x", unknown=1)  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# FieldMasker 构造
# --------------------------------------------------------------------------- #
class TestFieldMaskerInit:
    def test_defaults(self) -> None:
        """正向：默认使用内置规则与默认纳管字段。"""
        masker = FieldMasker()
        assert masker.rules == BUILTIN_RULES
        assert masker.fields == DEFAULT_MASKED_FIELDS

    def test_custom_rules_and_fields(self) -> None:
        """正向：自定义规则与字段生效。"""
        rule = MaskRule(name="only", pattern=r"\d+")
        masker = FieldMasker([rule], fields=["note", "content"])
        assert masker.rules == (rule,)
        assert masker.fields == ("note", "content")

    def test_empty_rules_rejected(self) -> None:
        """异常：空规则集被拒绝。"""
        with pytest.raises(ValidationError, match="at least one mask rule"):
            FieldMasker([])

    def test_duplicated_rule_names_rejected(self) -> None:
        """异常：规则名重复被拒绝。"""
        rules = [
            MaskRule(name="dup", pattern=r"\d+"),
            MaskRule(name="dup", pattern=r"[a-z]+"),
        ]
        with pytest.raises(ValidationError, match="duplicated mask rule names: dup"):
            FieldMasker(rules)

    def test_empty_fields_rejected(self) -> None:
        """异常：空字段集被拒绝。"""
        with pytest.raises(ValidationError, match="at least one target field"):
            FieldMasker(fields=[])

    def test_empty_field_name_rejected(self) -> None:
        """异常：字段名为空串被拒绝。"""
        with pytest.raises(ValidationError, match="must be non-empty"):
            FieldMasker(fields=["content", ""])

    def test_error_is_med_memory_error(self) -> None:
        """边界：校验异常归属统一异常体系。"""
        with pytest.raises(MedMemoryError):
            FieldMasker([])


# --------------------------------------------------------------------------- #
# FieldMasker 文本与字段脱敏
# --------------------------------------------------------------------------- #
class TestFieldMaskerText:
    def test_mask_text_applies_all_rules(self) -> None:
        """正向：多类隐私信息在一次调用中全部脱敏。"""
        masker = FieldMasker()
        result = masker.mask_text("患者 13812345678，身份证 11010119900307123X，床号: 7")
        assert result.text == "患者 138****5678，身份证 110101********123X，床号: *"
        assert result.hits == {"id_card": 1, "phone": 1, "bed_no": 1}
        assert result.changed is True

    def test_mask_text_without_pii(self) -> None:
        """边界：无隐私信息时原样返回、无命中。"""
        masker = FieldMasker()
        result = masker.mask_text("主诉：咳嗽三天")
        assert result.text == "主诉：咳嗽三天"
        assert result.hits == {}

    def test_mask_text_empty_string(self) -> None:
        """边界：空文本安全返回。"""
        assert FieldMasker().mask_text("").text == ""

    def test_mask_text_labeled_id_double_masking_is_safe(self) -> None:
        """边界：多规则重叠命中只会掩码更多、不会泄漏原文。"""
        result = FieldMasker().mask_text("住院号：110101199003071234")
        assert "1990" not in result.text
        assert result.total_hits >= 1

    def test_mask_field_whitelisted(self) -> None:
        """正向：纳管字段被脱敏。"""
        masker = FieldMasker()
        assert masker.mask_field("content", "电话 13812345678").text == "电话 138****5678"

    def test_mask_field_not_whitelisted(self) -> None:
        """边界：未纳管字段原样透传且无命中。"""
        masker = FieldMasker()
        result = masker.mask_field("remark", "电话 13812345678")
        assert result.text == "电话 13812345678"
        assert result.hits == {}

    def test_mask_fields_only_targets(self) -> None:
        """正向：批量脱敏仅改写纳管字段，其余字段与键顺序保持不变。"""
        masker = FieldMasker(fields=["content", "note"])
        data = {"content": "13812345678", "note": "床号: 9", "remark": "13812345678"}
        assert masker.mask_fields(data) == {
            "content": "138****5678",
            "note": "床号: *",
            "remark": "13812345678",
        }

    def test_mask_fields_empty_mapping(self) -> None:
        """边界：空映射返回空字典。"""
        assert FieldMasker().mask_fields({}) == {}


# --------------------------------------------------------------------------- #
# FieldMasker 消息脱敏
# --------------------------------------------------------------------------- #
class TestFieldMaskerMessage:
    def test_mask_message_content_and_flag(self) -> None:
        """正向：正文脱敏并置 masked 标记。"""
        message = make_message("请联系 13812345678")
        masked = FieldMasker().mask_message(message)
        assert masked.content == "请联系 138****5678"
        assert masked.masked is True
        assert masked.message_id == message.message_id

    def test_mask_message_does_not_mutate_original(self) -> None:
        """边界：返回副本，原消息保持不变。"""
        message = make_message("请联系 13812345678")
        FieldMasker().mask_message(message)
        assert message.content == "请联系 13812345678"
        assert message.masked is False

    def test_mask_message_is_idempotent(self) -> None:
        """边界：已脱敏消息原样返回（同一对象）。"""
        masker = FieldMasker()
        once = masker.mask_message(make_message("电话 13812345678"))
        twice = masker.mask_message(once)
        assert twice is once
        assert twice.content == "电话 138****5678"

    def test_mask_message_without_pii_still_flagged(self) -> None:
        """边界：无隐私信息时正文不变，但标记已过脱敏引擎。"""
        masked = FieldMasker().mask_message(make_message("主诉：头痛"))
        assert masked.content == "主诉：头痛"
        assert masked.masked is True

    def test_mask_message_masks_whitelisted_metadata(self) -> None:
        """正向：纳管的 metadata 键被脱敏，未纳管键透传。"""
        masker = FieldMasker(fields=["content", "contact"])
        message = make_message(
            "主诉：发热",
            metadata={"contact": "13812345678", "source": "13812345678"},
        )
        masked = masker.mask_message(message)
        assert masked.metadata == {"contact": "138****5678", "source": "13812345678"}

    def test_mask_messages_preserves_order(self) -> None:
        """正向：批量脱敏保持顺序。"""
        messages = [make_message("13812345678"), make_message("12床")]
        result = FieldMasker().mask_messages(messages)
        assert [m.content for m in result] == ["138****5678", "**床"]

    def test_mask_messages_empty(self) -> None:
        """边界：空序列返回空列表。"""
        assert FieldMasker().mask_messages([]) == []


# --------------------------------------------------------------------------- #
# 便捷入口
# --------------------------------------------------------------------------- #
class TestConvenienceHelpers:
    def test_default_masker_is_cached_singleton(self) -> None:
        """正向：默认引擎为惰性单例。"""
        assert default_masker() is default_masker()
        assert default_masker().rules == BUILTIN_RULES

    def test_mask_text_helper(self) -> None:
        """正向：便捷函数直接返回脱敏文本。"""
        assert mask_text("电话 13812345678") == "电话 138****5678"

    def test_mask_text_helper_without_pii(self) -> None:
        """边界：无隐私信息时原样返回。"""
        assert mask_text("普通描述") == "普通描述"
