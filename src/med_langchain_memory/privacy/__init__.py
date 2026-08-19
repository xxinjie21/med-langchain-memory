"""隐私合规层：字段级正则脱敏。

本层以 :class:`MaskRule` + :class:`FieldMasker` 实现纯规则脱敏，覆盖手机号、
身份证号、病历号与床号等结构化隐私字段。**不含任何分词、实体识别或语义解析
逻辑**，全部改写均由正则匹配与字符替换完成。
"""

from __future__ import annotations

from .masker import (
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

__all__ = [
    "MaskRule",
    "MaskResult",
    "FieldMasker",
    "BUILTIN_RULES",
    "DEFAULT_MASKED_FIELDS",
    "PHONE_RULE",
    "ID_CARD_RULE",
    "MEDICAL_RECORD_RULE",
    "BED_NO_RULE",
    "BED_NO_SUFFIX_RULE",
    "default_masker",
    "mask_text",
]
