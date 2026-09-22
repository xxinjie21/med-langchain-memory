"""长会话 LLM 摘要压缩单元测试（D27）。

覆盖：摘要策略校验、摘要消息标记读取、摘要链返回值提取、确定性摘要链、
摘要提示词模板、压缩触发与规划（保护位 / 尾部保留 / 区间标记）、摘要链注入与
异常传播、以及 ``MedRunnableWithMessageHistory`` 的「时序窗口 → 摘要压缩 →
Token 预算」三级流水线集成。

全部用例零外部依赖、零网络、零真实 LLM：固定字符计数器 + 摘要链替身 + 内存后端
即可跑通。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.runnable import (
    COMPRESSED_COUNT_KEY,
    COMPRESSED_RANGE_KEY,
    DEFAULT_SUMMARY_HEADER,
    SUMMARY_FLAG_KEY,
    SUMMARY_INSTRUCTION,
    ExtractiveSummaryChain,
    MedRunnableWithMessageHistory,
    SummaryCompressor,
    SummaryPolicy,
    SummaryReport,
    TokenBudgetPolicy,
    TokenCounter,
    build_summary_prompt,
    compressed_count_of,
    compressed_range_of,
    extract_summary_text,
    is_summary_message,
)
from med_langchain_memory.runnable.trimmer import ContextWindowPolicy
from med_langchain_memory.stores import StoreFactory

# --------------------------------------------------------------------- #
# 测试替身与夹具
# --------------------------------------------------------------------- #

#: 固定基准时间戳（epoch millis），保证时序用例可复现。
BASE_MS = 1_700_000_000_000

TENANT_ID = "h-a"
DEPT_ID = "cardiology"
PATIENT_ID = "p-1"

#: 替身摘要链固定返回的摘要正文。
SUMMARY_TEXT = "SUMMARY"


class _CharCounter(TokenCounter):
    """按字符数计数的确定性替身（``"abcd"`` -> 4 token）。"""

    def count(self, text: str) -> int:
        """返回文本字符数。"""
        return len(text)


class _FixedChain:
    """返回固定文本的摘要链替身，并记录每次调用的入参。"""

    def __init__(self, text: str = SUMMARY_TEXT) -> None:
        """初始化替身链。

        Args:
            text: ``invoke`` 固定返回的摘要正文。
        """
        self._text = text
        self.calls: list[Any] = []

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        """记录入参并返回固定摘要正文。"""
        self.calls.append(input)
        return self._text


class _RaisingChain:
    """``invoke`` 直接抛异常的摘要链替身。"""

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> str:
        """抛出运行时错误，用于验证异常透传。"""
        raise RuntimeError("chain exploded")


def _ai(content: str, **extra: Any) -> AIMessage:
    """构造助手消息（非保护位）。"""
    return AIMessage(content=content, additional_kwargs=dict(extra))


def _human(content: str, **extra: Any) -> HumanMessage:
    """构造患者消息（首条即主诉，受保护）。"""
    return HumanMessage(content=content, additional_kwargs=dict(extra))


def _sys(content: str, **extra: Any) -> SystemMessage:
    """构造系统槽位消息（受保护）。"""
    return SystemMessage(content=content, additional_kwargs=dict(extra))


def _thread(*contents: str) -> list[BaseMessage]:
    """构造 ``[系统槽位, 患者主诉, 后续交替消息...]`` 的消息序列。

    Args:
        *contents: 对话正文，第一条为患者主诉，后续按「患者 / 助手」交替。

    Returns:
        以 ``SystemMessage`` 开头、第二条为患者主诉的消息列表。
    """
    messages: list[BaseMessage] = [_sys("sys")]
    for index, content in enumerate(contents):
        messages.append(_human(content) if index % 2 == 0 else _ai(content))
    return messages


def _contents(messages: list[BaseMessage]) -> list[str]:
    """提取消息正文，便于断言压缩结果。"""
    return [str(message.content) for message in messages]


def _compressor(chain: Any = None, **policy_kwargs: Any) -> SummaryCompressor:
    """构造使用固定字符计数器的摘要压缩器。"""
    return SummaryCompressor(
        SummaryPolicy(**policy_kwargs),
        chain=_FixedChain() if chain is None else chain,
        counter=_CharCounter(),
    )


def _echo_runnable() -> RunnableLambda:
    """构造无需外部 LLM 的下游 Runnable。"""
    return RunnableLambda(lambda payload: {"echo": payload})


# --------------------------------------------------------------------- #
# SummaryPolicy 校验
# --------------------------------------------------------------------- #
def test_policy_defaults() -> None:
    """策略默认值：仅需一个触发阈值，尾部保留 6 条。"""
    policy = SummaryPolicy(max_messages=20)
    assert policy.max_tokens is None
    assert policy.keep_recent_messages == 6
    assert policy.keep_system_messages is True
    assert policy.keep_chief_complaint is True
    assert policy.prefer_stored_token_count is True
    assert policy.summary_header == DEFAULT_SUMMARY_HEADER


def test_policy_requires_at_least_one_trigger() -> None:
    """两个触发阈值都未配置时策略非法（边界：避免静默失效）。"""
    with pytest.raises(PydanticValidationError, match="at least one"):
        SummaryPolicy()


def test_policy_rejects_non_positive_max_messages() -> None:
    """``max_messages`` 必须为正整数。"""
    with pytest.raises(PydanticValidationError):
        SummaryPolicy(max_messages=0)


def test_policy_rejects_non_positive_max_tokens() -> None:
    """``max_tokens`` 必须为正整数。"""
    with pytest.raises(PydanticValidationError):
        SummaryPolicy(max_tokens=0)


def test_policy_allows_zero_keep_recent() -> None:
    """``keep_recent_messages`` 允许为 0（全部折叠）。"""
    assert SummaryPolicy(max_messages=1, keep_recent_messages=0).keep_recent_messages == 0


def test_policy_rejects_negative_keep_recent() -> None:
    """``keep_recent_messages`` 不允许为负。"""
    with pytest.raises(PydanticValidationError):
        SummaryPolicy(max_messages=1, keep_recent_messages=-1)


def test_policy_rejects_empty_header() -> None:
    """摘要抬头不允许为空串。"""
    with pytest.raises(PydanticValidationError):
        SummaryPolicy(max_messages=1, summary_header="")


def test_policy_rejects_unknown_field() -> None:
    """未知字段被拒绝（extra=forbid）。"""
    with pytest.raises(PydanticValidationError):
        SummaryPolicy(max_messages=1, unknown_field=1)  # type: ignore[call-arg]


def test_policy_is_frozen() -> None:
    """策略不可变。"""
    policy = SummaryPolicy(max_messages=1)
    with pytest.raises(PydanticValidationError):
        policy.max_messages = 5  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 摘要消息标记
# --------------------------------------------------------------------- #
def test_is_summary_message_true_for_marked_message() -> None:
    """带标记的消息被识别为摘要。"""
    assert is_summary_message(_sys("x", **{SUMMARY_FLAG_KEY: True})) is True


def test_is_summary_message_false_for_plain_message() -> None:
    """未标记的消息不是摘要。"""
    assert is_summary_message(_ai("x")) is False


def test_compressed_range_of_reads_valid_range() -> None:
    """合法区间被正确读出。"""
    message = _sys("x", **{COMPRESSED_RANGE_KEY: [2, 7]})
    assert compressed_range_of(message) == (2, 7)


def test_compressed_range_of_rejects_invalid_payloads() -> None:
    """区间缺失或格式非法时返回 None（边界：多分支防御）。"""
    assert compressed_range_of(_ai("x")) is None
    assert compressed_range_of(_sys("x", **{COMPRESSED_RANGE_KEY: [1]})) is None
    assert compressed_range_of(_sys("x", **{COMPRESSED_RANGE_KEY: "1-2"})) is None
    assert compressed_range_of(_sys("x", **{COMPRESSED_RANGE_KEY: [True, 2]})) is None
    assert compressed_range_of(_sys("x", **{COMPRESSED_RANGE_KEY: ["1", 2]})) is None
    assert compressed_range_of(_sys("x", **{COMPRESSED_RANGE_KEY: [-1, 2]})) is None
    assert compressed_range_of(_sys("x", **{COMPRESSED_RANGE_KEY: [5, 2]})) is None


def test_compressed_count_of_reads_valid_count() -> None:
    """条数标记被正确读出。"""
    assert compressed_count_of(_sys("x", **{COMPRESSED_COUNT_KEY: 3})) == 3


def test_compressed_count_of_falls_back_to_zero() -> None:
    """条数标记缺失或非法时返回 0。"""
    assert compressed_count_of(_ai("x")) == 0
    assert compressed_count_of(_sys("x", **{COMPRESSED_COUNT_KEY: -1})) == 0
    assert compressed_count_of(_sys("x", **{COMPRESSED_COUNT_KEY: "3"})) == 0
    assert compressed_count_of(_sys("x", **{COMPRESSED_COUNT_KEY: True})) == 0


# --------------------------------------------------------------------- #
# 摘要链返回值提取
# --------------------------------------------------------------------- #
def test_extract_summary_text_from_string() -> None:
    """字符串返回值去除首尾空白。"""
    assert extract_summary_text("  hello  ") == "hello"


def test_extract_summary_text_from_message() -> None:
    """消息返回值按正文提取。"""
    assert extract_summary_text(_ai("from message")) == "from message"


def test_extract_summary_text_from_mapping_text() -> None:
    """映射返回值取 ``text`` 字段。"""
    assert extract_summary_text({"text": "mapped"}) == "mapped"


def test_extract_summary_text_from_mapping_content() -> None:
    """映射返回值缺 ``text`` 时回退 ``content``。"""
    assert extract_summary_text({"content": "mapped content"}) == "mapped content"


def test_extract_summary_text_rejects_mapping_without_text() -> None:
    """映射中无字符串正文时抛 TypeError（边界）。"""
    with pytest.raises(TypeError, match="must contain a string"):
        extract_summary_text({"text": 123})


def test_extract_summary_text_rejects_unsupported_type() -> None:
    """不支持的类型抛 TypeError（边界）。"""
    with pytest.raises(TypeError, match="unsupported summary chain output"):
        extract_summary_text(123)


def test_extract_summary_text_rejects_empty_text() -> None:
    """空正文视为摘要失败（边界：不写入空摘要）。"""
    with pytest.raises(ValueError, match="empty text"):
        extract_summary_text("   ")


# --------------------------------------------------------------------- #
# 确定性摘要链
# --------------------------------------------------------------------- #
def test_extractive_chain_invokes_with_dict_payload() -> None:
    """按 ``{"messages": [...]}`` 载荷生成带角色标签的摘要。"""
    chain = ExtractiveSummaryChain()
    text = chain.invoke({"messages": [_human("cough"), _ai("noted")]})
    assert text == "[patient] cough\n[assistant] noted"


def test_extractive_chain_accepts_raw_sequence() -> None:
    """也接受裸消息序列（便利重载）。"""
    chain = ExtractiveSummaryChain()
    assert chain.invoke([_human("fever")]) == "[patient] fever"


def test_extractive_chain_accepts_single_message() -> None:
    """单条消息自动包装为列表。"""
    chain = ExtractiveSummaryChain()
    assert chain.invoke(_ai("ok")) == "[assistant] ok"


def test_extractive_chain_uses_med_role_label() -> None:
    """带 ``med_role`` 扩展字段时优先用医疗角色标签。"""
    chain = ExtractiveSummaryChain()
    message = _ai("bp 120/80", med_role="doctor")
    assert chain.invoke({"messages": [message]}) == "[doctor] bp 120/80"


def test_extractive_chain_truncates_long_message() -> None:
    """单条消息超长时截断并追加省略号。"""
    chain = ExtractiveSummaryChain(max_chars_per_message=5)
    assert chain.invoke({"messages": [_ai("0123456789")]}) == "[assistant] 01234..."


def test_extractive_chain_truncates_total_length() -> None:
    """摘要总长度超限时整体截断。"""
    chain = ExtractiveSummaryChain(max_chars_per_message=50, max_chars=12)
    assert chain.invoke({"messages": [_ai("0123456789")]}) == "[assistant] ..."


def test_extractive_chain_rejects_empty_input() -> None:
    """空消息列表不生成空摘要（边界）。"""
    with pytest.raises(ValueError, match="no messages to summarize"):
        ExtractiveSummaryChain().invoke({"messages": []})


def test_extractive_chain_rejects_missing_messages_key() -> None:
    """载荷缺 ``messages`` 字段（值为 None）时同样视为无内容（边界）。"""
    with pytest.raises(ValueError, match="no messages to summarize"):
        ExtractiveSummaryChain().invoke({"messages": None})


def test_extractive_chain_rejects_non_message_element() -> None:
    """载荷中混入非消息元素时抛 TypeError（边界）。"""
    with pytest.raises(TypeError, match="must contain messages"):
        ExtractiveSummaryChain().invoke({"messages": ["not-a-message"]})


def test_extractive_chain_rejects_non_sequence_payload() -> None:
    """载荷类型不受支持时抛 TypeError（边界）。"""
    with pytest.raises(TypeError, match="unsupported summary chain input"):
        ExtractiveSummaryChain().invoke({"messages": 42})


def test_extractive_chain_rejects_non_positive_limits() -> None:
    """长度上限必须为正（边界）。"""
    with pytest.raises(ValueError, match="must be positive"):
        ExtractiveSummaryChain(max_chars=0)
    with pytest.raises(ValueError, match="must be positive"):
        ExtractiveSummaryChain(max_chars_per_message=0)


def test_extractive_chain_exposes_limits() -> None:
    """属性暴露配置的长度上限。"""
    chain = ExtractiveSummaryChain(max_chars_per_message=7, max_chars=9)
    assert chain.max_chars_per_message == 7
    assert chain.max_chars == 9


# --------------------------------------------------------------------- #
# 摘要提示词模板
# --------------------------------------------------------------------- #
def test_build_summary_prompt_returns_chat_template() -> None:
    """返回聊天提示词模板，系统指令 + 消息占位符齐备。"""
    prompt = build_summary_prompt()
    assert isinstance(prompt, ChatPromptTemplate)
    formatted = prompt.format_messages(messages=[_human("cough")])
    assert formatted[0].content == SUMMARY_INSTRUCTION
    assert formatted[1].content == "cough"


def test_build_summary_prompt_accepts_custom_instruction() -> None:
    """可自定义系统指令。"""
    prompt = build_summary_prompt("custom instruction")
    formatted = prompt.format_messages(messages=[])
    assert [m.content for m in formatted] == ["custom instruction"]


# --------------------------------------------------------------------- #
# 触发与规划
# --------------------------------------------------------------------- #
def test_should_compress_false_below_threshold() -> None:
    """未命中阈值时不触发。"""
    assert _compressor(max_messages=10).should_compress(_thread("c1", "c2", "c3")) is False


def test_should_compress_true_when_over_max_messages() -> None:
    """可压缩条数超阈值时触发。"""
    assert _compressor(max_messages=2, keep_recent_messages=1).should_compress(
        _thread("c1", "c2", "c3", "c4", "c5")
    )


def test_should_compress_false_when_nothing_to_compress() -> None:
    """阈值命中但尾部保留已覆盖全部候选时不触发（边界）。"""
    compressor = _compressor(max_messages=1, keep_recent_messages=5)
    assert compressor.should_compress(_thread("c1", "c2", "c3")) is False


def test_should_compress_false_for_empty_input() -> None:
    """空序列不触发（边界）。"""
    assert _compressor(max_messages=1).should_compress([]) is False


def test_compress_skips_when_not_triggered() -> None:
    """未命中阈值时原样返回，报告标记未生效。"""
    messages = _thread("c1", "c2", "c3")
    result = _compressor(max_messages=10).compress(messages)
    assert result.messages == messages
    assert result.summary_message is None
    assert result.report.applied is False
    assert result.report.trigger is None
    assert result.report.kept_messages == len(messages)


def test_compress_warns_when_nothing_to_compress() -> None:
    """阈值命中但无内容可折叠时给出说明性告警（边界）。"""
    result = _compressor(max_messages=1, keep_recent_messages=5).compress(_thread("c1", "c2", "c3"))
    assert result.report.applied is False
    assert result.report.trigger == "max_messages"
    assert any("nothing to compress" in w for w in result.report.warnings)


def test_compress_warns_on_empty_input() -> None:
    """空输入给出说明性告警（边界）。"""
    result = _compressor(max_messages=1).compress([])
    assert result.messages == []
    assert result.report.applied is False
    assert result.report.warnings == ["empty message list"]


def test_compress_folds_middle_slice_into_summary() -> None:
    """中间段被折叠，摘要插入被折叠区间处，尾部原文保留。"""
    chain = _FixedChain()
    compressor = _compressor(chain, max_messages=3, keep_recent_messages=2)
    messages = _thread("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")

    result = compressor.compress(messages)

    assert _contents(result.messages) == [
        "sys",
        "c1",
        f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}",
        "c7",
        "c8",
    ]
    assert result.report.applied is True
    assert result.report.trigger == "max_messages"
    assert result.report.compressed_count == 5
    assert result.report.compressed_range == (2, 6)
    assert result.report.kept_messages == 5


def test_compress_passes_only_compressed_slice_to_chain() -> None:
    """摘要链只收到被折叠区间的消息。"""
    chain = _FixedChain()
    compressor = _compressor(chain, max_messages=3, keep_recent_messages=2)
    compressor.compress(_thread("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"))

    assert len(chain.calls) == 1
    payload = chain.calls[0]
    assert isinstance(payload, dict)
    assert _contents(payload["messages"]) == ["c2", "c3", "c4", "c5", "c6"]


def test_compress_marks_summary_message() -> None:
    """摘要消息是 system 消息并携带区间与条数标记。"""
    result = _compressor(max_messages=3, keep_recent_messages=2).compress(
        _thread("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
    )
    summary = result.summary_message
    assert summary is not None
    assert isinstance(summary, SystemMessage)
    assert is_summary_message(summary) is True
    assert compressed_range_of(summary) == (2, 6)
    assert compressed_count_of(summary) == 5
    assert str(summary.content).startswith(DEFAULT_SUMMARY_HEADER)
    assert result.messages[2] is summary


def test_compress_counts_summary_tokens() -> None:
    """摘要消息的 token 占用按本压缩器计数口径统计。"""
    result = _compressor(max_messages=3, keep_recent_messages=2).compress(
        _thread("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
    )
    expected = len(f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}")
    assert result.report.summary_tokens == expected


def test_compress_triggers_on_token_budget() -> None:
    """token 阈值同样可触发压缩。"""
    compressor = _compressor(max_tokens=15, keep_recent_messages=1, keep_chief_complaint=False)
    result = compressor.compress(_thread("a" * 10, "b" * 10, "c" * 10))

    assert result.report.applied is True
    assert result.report.trigger == "max_tokens"
    assert result.report.compressed_count == 2
    assert result.report.compressed_range == (1, 2)
    assert _contents(result.messages) == [
        "sys",
        f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}",
        "c" * 10,
    ]


def test_compress_prefers_message_level_trigger() -> None:
    """两个阈值同时命中时以 ``max_messages`` 为先。"""
    compressor = _compressor(
        max_messages=1, max_tokens=1, keep_recent_messages=0, keep_chief_complaint=False
    )
    result = compressor.compress(_thread("c1", "c2"))
    assert result.report.trigger == "max_messages"


def test_compress_reuses_stored_token_count_for_trigger() -> None:
    """已存 token 数被优先复用于触发判定。"""
    compressor = _compressor(max_tokens=50, keep_recent_messages=0)
    messages: list[BaseMessage] = [_sys("s"), _ai("x", token_count=100)]
    result = compressor.compress(messages)
    assert result.report.applied is True
    assert result.report.compressed_count == 1


def test_compress_ignores_stored_token_count_when_disabled() -> None:
    """关闭复用后按计数器实际计数判定（边界）。"""
    compressor = _compressor(max_tokens=50, keep_recent_messages=0, prefer_stored_token_count=False)
    messages: list[BaseMessage] = [_sys("s"), _ai("x", token_count=100)]
    assert compressor.compress(messages).report.applied is False


def test_compress_keeps_recent_tail_verbatim() -> None:
    """尾部 ``keep_recent_messages`` 条始终保留原文。"""
    result = _compressor(max_messages=2, keep_recent_messages=3).compress(
        _thread("c1", "c2", "c3", "c4", "c5", "c6")
    )
    assert _contents(result.messages)[-3:] == ["c4", "c5", "c6"]


def test_compress_with_zero_keep_recent_folds_all_candidates() -> None:
    """``keep_recent_messages=0`` 时全部候选折叠进摘要（边界）。"""
    result = _compressor(max_messages=1, keep_recent_messages=0).compress(_thread("c1", "c2", "c3"))
    assert _contents(result.messages) == ["sys", "c1", f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}"]
    assert result.report.compressed_range == (2, 3)


def test_compress_can_release_system_messages() -> None:
    """关闭系统槽位保护后系统消息也参与压缩。"""
    result = _compressor(
        max_messages=1, keep_recent_messages=1, keep_system_messages=False
    ).compress(_thread("c1", "c2", "c3", "c4"))

    assert _contents(result.messages) == [
        f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}",
        "c1",
        "c4",
    ]
    assert result.report.compressed_count == 3


def test_compress_can_release_chief_complaint() -> None:
    """关闭主诉保护后首条患者消息也参与压缩。"""
    result = _compressor(
        max_messages=1, keep_recent_messages=1, keep_chief_complaint=False
    ).compress(_thread("c1", "c2", "c3"))

    assert _contents(result.messages) == ["sys", f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}", "c3"]
    assert result.report.compressed_range == (1, 2)


def test_compress_retains_protected_message_inside_range() -> None:
    """区间内被保护的消息（如诊疗规范系统消息）原样保留。"""
    messages: list[BaseMessage] = [
        _sys("sys"),
        _human("c1"),
        _ai("c2"),
        _sys("guideline"),
        _human("c3"),
        _ai("c4"),
        _human("c5"),
        _ai("c6"),
    ]
    result = _compressor(max_messages=2, keep_recent_messages=1).compress(messages)

    assert _contents(result.messages) == [
        "sys",
        "c1",
        f"{DEFAULT_SUMMARY_HEADER}\n{SUMMARY_TEXT}",
        "guideline",
        "c6",
    ]
    assert result.report.compressed_range == (2, 6)
    assert result.report.compressed_count == 4


def test_compress_is_idempotent_on_its_own_output() -> None:
    """对自身输出再压缩不再生效（摘要消息受保护，边界）。"""
    compressor = _compressor(max_messages=3, keep_recent_messages=2)
    first = compressor.compress(_thread("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"))
    second = compressor.compress(first.messages)

    assert first.report.applied is True
    assert second.report.applied is False
    assert second.messages == first.messages


def test_compress_does_not_mutate_input() -> None:
    """压缩不修改入参序列。"""
    messages = _thread("c1", "c2", "c3", "c4", "c5")
    snapshot = list(messages)
    _compressor(max_messages=2, keep_recent_messages=1).compress(messages)
    assert messages == snapshot


def test_compress_propagates_chain_failure() -> None:
    """摘要链异常向上透传（边界）。"""
    with pytest.raises(RuntimeError, match="chain exploded"):
        _compressor(_RaisingChain(), max_messages=2, keep_recent_messages=1).compress(
            _thread("c1", "c2", "c3", "c4")
        )


def test_compress_rejects_empty_summary_text() -> None:
    """摘要链返回空文本时抛 ValueError（边界）。"""
    with pytest.raises(ValueError, match="empty text"):
        _compressor(_FixedChain("   "), max_messages=2, keep_recent_messages=1).compress(
            _thread("c1", "c2", "c3", "c4")
        )


def test_compress_rejects_unsupported_chain_output() -> None:
    """摘要链返回不支持类型时抛 TypeError（边界）。"""
    chain = RunnableLambda(lambda payload: 42)
    with pytest.raises(TypeError, match="unsupported summary chain output"):
        _compressor(chain, max_messages=2, keep_recent_messages=1).compress(
            _thread("c1", "c2", "c3", "c4")
        )


def test_compress_accepts_lcel_chain() -> None:
    """可直接注入 LCEL 链（``RunnableLambda``）。"""
    chain = RunnableLambda(lambda payload: f"{len(payload['messages'])}-folded")
    result = _compressor(chain, max_messages=2, keep_recent_messages=1).compress(
        _thread("c1", "c2", "c3", "c4")
    )
    assert result.report.applied is True
    assert str(result.summary_message.content).endswith("2-folded")


def test_compress_uses_extractive_chain_by_default() -> None:
    """未注入摘要链时使用零依赖的确定性摘要链。"""
    compressor = SummaryCompressor(
        SummaryPolicy(max_messages=1, keep_recent_messages=0),
        counter=_CharCounter(),
    )
    assert isinstance(compressor.chain, ExtractiveSummaryChain)
    result = compressor.compress([_sys("sys"), _ai("cough"), _ai("fever")])
    assert "[assistant] cough" in str(result.summary_message.content)


def test_compressor_exposes_configuration() -> None:
    """属性暴露策略、摘要链与计数器。"""
    policy = SummaryPolicy(max_messages=5)
    chain = _FixedChain()
    counter = _CharCounter()
    compressor = SummaryCompressor(policy, chain=chain, counter=counter)
    assert compressor.policy is policy
    assert compressor.chain is chain
    assert compressor.counter is counter


def test_count_uses_same_cost_policy() -> None:
    """``count`` 与触发判定使用同一计数口径。"""
    compressor = _compressor(max_messages=100)
    assert compressor.count([_ai("x", token_count=5), _ai("abcd")]) == 9


def test_compress_history_reads_store_messages() -> None:
    """可直接压缩存储句柄中的全量消息。"""
    history = StoreFactory.create(
        "memory",
        session_id="s-summary-hist",
        tenant_id=TENANT_ID,
        dept_id=DEPT_ID,
        patient_id=PATIENT_ID,
    )
    history.clear()
    history.add_messages(_thread("c1", "c2", "c3", "c4", "c5"))

    result = _compressor(max_messages=2, keep_recent_messages=1).compress_history(history)
    assert result.report.applied is True
    assert result.report.compressed_count == 3


def test_summary_report_defaults() -> None:
    """报告默认值表示未生效。"""
    report = SummaryReport()
    assert report.applied is False
    assert report.trigger is None
    assert report.compressed_range is None
    assert report.compressed_count == 0
    assert report.warnings == []


# --------------------------------------------------------------------- #
# 与 MedRunnableWithMessageHistory 的三级流水线集成
# --------------------------------------------------------------------- #
def test_runnable_without_summarizer_returns_none_report() -> None:
    """未注入摘要策略时 ``summary`` 报告为 None，消息原样返回。"""
    rwh = MedRunnableWithMessageHistory(_echo_runnable(), backend="memory")
    messages = _thread("c1", "c2")
    result = rwh.build_context(messages)

    assert result.messages == messages
    assert result.summary is None
    assert rwh.summarizer is None
    assert rwh.summary_chain is None


def test_runnable_summarize_context_without_policy_is_noop() -> None:
    """未注入策略时 ``summarize_context`` 原样返回且标记未生效。"""
    rwh = MedRunnableWithMessageHistory(_echo_runnable(), backend="memory")
    messages = _thread("c1", "c2")
    result = rwh.summarize_context(messages)

    assert result.messages == messages
    assert result.summary_message is None
    assert result.report.applied is False


def test_runnable_summarize_context_applies_policy() -> None:
    """注入策略后 ``summarize_context`` 折叠中间段。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        summarizer=SummaryPolicy(max_messages=2, keep_recent_messages=1),
        summary_chain=_FixedChain(),
        token_counter=_CharCounter(),
    )
    result = rwh.summarize_context(_thread("c1", "c2", "c3", "c4"))

    assert result.report.applied is True
    assert result.report.compressed_count == 2
    assert is_summary_message(result.messages[2])


def test_runnable_exposes_summary_configuration() -> None:
    """属性暴露注入的策略与摘要链。"""
    policy = SummaryPolicy(max_messages=5)
    chain = _FixedChain()
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        summarizer=policy,
        summary_chain=chain,
    )
    assert rwh.summarizer is policy
    assert rwh.summary_chain is chain


def test_runnable_three_stage_pipeline() -> None:
    """时序窗口先裁、摘要压缩次之、Token 预算兜底，三级叠加产出最终上下文。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        trim_policy=ContextWindowPolicy(max_messages=4),
        summarizer=SummaryPolicy(max_messages=2, keep_recent_messages=1),
        summary_chain=_FixedChain(),
        token_budget=TokenBudgetPolicy(max_tokens=200),
        token_counter=_CharCounter(),
    )
    messages = _thread("c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8")
    result = rwh.build_context(messages, now_ms=BASE_MS)

    assert _contents(result.messages)[0:2] == ["sys", "c1"]
    assert is_summary_message(result.messages[2])
    assert _contents(result.messages)[3] == "c8"
    assert result.summary is not None
    assert result.summary.applied is True
    assert result.summary.compressed_count == 3
    assert result.summary.compressed_range == (2, 4)
    assert result.report.applied is True
    assert result.report.dropped_messages == 0


def test_runnable_budget_keeps_summary_message() -> None:
    """预算极紧时摘要消息作为系统消息受保护，仅尾部原文被裁掉。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        summarizer=SummaryPolicy(max_messages=2, keep_recent_messages=1),
        summary_chain=_FixedChain(),
        token_budget=TokenBudgetPolicy(max_tokens=20),
        token_counter=_CharCounter(),
    )
    result = rwh.build_context(_thread("c1", "c2", "c3", "c4"))

    assert any(is_summary_message(m) for m in result.messages)
    assert result.report.dropped_messages == 1
    assert result.report.over_budget is True
    assert result.summary is not None
    assert result.summary.applied is True


def test_runnable_build_context_does_not_mutate_input() -> None:
    """三级流水线不修改入参序列。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        summarizer=SummaryPolicy(max_messages=2, keep_recent_messages=1),
        summary_chain=_FixedChain(),
        token_budget=TokenBudgetPolicy(max_tokens=200),
        token_counter=_CharCounter(),
    )
    messages = _thread("c1", "c2", "c3", "c4")
    snapshot = list(messages)
    rwh.build_context(messages)
    assert messages == snapshot
