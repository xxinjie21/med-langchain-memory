"""Token 预算裁剪单元测试（D26）。

覆盖：消息文本提取、启发式 / tiktoken 计数器（含降级路径）、计数器名称解析、
预算策略校验、预算内贪心保留的各类边界（保护位、连续性、预留额度、告警阈值、
已存 token 数复用）、与 ``MedRunnableWithMessageHistory`` 的两级上下文流水线集成。

全部用例零外部依赖、零网络：固定字符计数器 + 内存后端即可跑通；真实 tiktoken
用例在缺包或编码表不可达时自动跳过。
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.runnable import (
    DEFAULT_TIKTOKEN_ENCODING,
    HeuristicTokenCounter,
    MedRunnableWithMessageHistory,
    TiktokenTokenCounter,
    TokenBudgetPolicy,
    TokenBudgetTrimmer,
    TokenCounter,
    message_text,
    resolve_token_counter,
    stored_token_count,
)
from med_langchain_memory.runnable.trimmer import ContextWindowPolicy
from med_langchain_memory.stores import StoreFactory
from med_langchain_memory.stores.base import MED_ROLE_KEY

# --------------------------------------------------------------------- #
# 测试替身与夹具
# --------------------------------------------------------------------- #

#: 固定基准时间戳（epoch millis），保证时序用例可复现。
BASE_MS = 1_700_000_000_000

TENANT_ID = "h-a"
DEPT_ID = "cardiology"
PATIENT_ID = "p-1"


class _CharCounter(TokenCounter):
    """按字符数计数的确定性替身（``"abcd"`` -> 4 token）。"""

    def count(self, text: str) -> int:
        """返回文本字符数。"""
        return len(text)


def _ai(content: str, *, created_at: int = BASE_MS) -> AIMessage:
    """构造助手消息（医疗角色 ASSISTANT，非保护位）。"""
    return AIMessage(content=content, additional_kwargs={"created_at": created_at})


def _sys(content: str, *, created_at: int = BASE_MS) -> SystemMessage:
    """构造系统槽位消息（受保护）。"""
    return SystemMessage(content=content, additional_kwargs={"created_at": created_at})


def _human(content: str, *, created_at: int = BASE_MS) -> HumanMessage:
    """构造患者消息（首条即主诉，受保护）。"""
    return HumanMessage(content=content, additional_kwargs={"created_at": created_at})


def _with_stored(content: str, token_count: int) -> AIMessage:
    """构造携带已算好 token 数的助手消息。"""
    return AIMessage(content=content, additional_kwargs={"token_count": token_count})


def _contents(messages: list[BaseMessage]) -> list[str]:
    """提取消息正文，便于断言裁剪结果。"""
    return [str(message.content) for message in messages]


def _trimmer(**policy_kwargs: Any) -> TokenBudgetTrimmer:
    """构造使用固定字符计数器的预算裁剪器。"""
    return TokenBudgetTrimmer(TokenBudgetPolicy(**policy_kwargs), counter=_CharCounter())


def _echo_runnable() -> RunnableLambda:
    """构造无需外部 LLM 的下游 Runnable。"""
    return RunnableLambda(lambda payload: {"echo": payload})


# --------------------------------------------------------------------- #
# 消息文本提取
# --------------------------------------------------------------------- #
def test_message_text_plain_string() -> None:
    """纯字符串正文原样返回。"""
    assert message_text(_ai("你好 world")) == "你好 world"


def test_message_text_structured_blocks() -> None:
    """结构化内容块按 text 字段拼接。"""
    message = AIMessage(
        content=[{"type": "text", "text": "abc"}, {"type": "text", "text": "de"}]  # type: ignore[arg-type]
    )
    assert message_text(message) == "abcde"


def test_message_text_structured_blocks_with_unknown_block() -> None:
    """未知类型的结构化块退化为空串，不抛异常。"""
    message = AIMessage(content=[{"type": "image_url", "image_url": "x"}] * 1)  # type: ignore[arg-type]
    assert message_text(message) == ""


def test_message_text_structured_blocks_with_plain_string_entry() -> None:
    """结构化内容列表中混入的纯字符串条目按原样计入。"""
    message = AIMessage(content=["ab", {"type": "text", "text": "cd"}])  # type: ignore[arg-type]
    assert message_text(message) == "abcd"


def test_message_text_empty_content() -> None:
    """空正文返回空串。"""
    assert message_text(AIMessage(content="")) == ""


# --------------------------------------------------------------------- #
# 启发式计数器
# --------------------------------------------------------------------- #
def test_heuristic_count_empty_returns_zero() -> None:
    """空串计 0 token。"""
    assert HeuristicTokenCounter().count("") == 0


def test_heuristic_count_ascii_uses_four_chars_per_token() -> None:
    """ASCII 文本按 4 字符 ≈ 1 token 向上取整。"""
    counter = HeuristicTokenCounter()
    assert counter.count("abcd") == 1
    assert counter.count("abcde") == 2
    assert counter.count("a" * 100) == 25


def test_heuristic_count_cjk_uses_one_token_per_char() -> None:
    """非 ASCII（中文）字符按 1 token/字计。"""
    assert HeuristicTokenCounter().count("你好") == 2


def test_heuristic_count_mixed_text() -> None:
    """中英混排分别计数后相加。"""
    assert HeuristicTokenCounter().count("你好abc") == 3


def test_counter_counts_message_and_sequence() -> None:
    """消息级与序列级计数按同一口径求和。"""
    counter = _CharCounter()
    messages = [_ai("aaaa"), _ai("bb"), _ai("")]
    assert counter.count_message(messages[0]) == 4
    assert counter.count_messages(messages) == 6
    assert counter.count_messages([]) == 0


# --------------------------------------------------------------------- #
# tiktoken 计数器（懒加载 + 降级）
# --------------------------------------------------------------------- #
def test_tiktoken_counter_not_degraded_before_use() -> None:
    """构造时不触发加载，故尚未降级。"""
    counter = TiktokenTokenCounter()
    assert counter.degraded is False
    assert counter.encoding_name == DEFAULT_TIKTOKEN_ENCODING


def test_tiktoken_counter_degrades_when_import_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺少 tiktoken 时降级为启发式计数，不抛异常。"""
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    counter = TiktokenTokenCounter()
    assert counter.count("abcd") == 1
    assert counter.degraded is True


def test_tiktoken_counter_degrades_when_encoding_load_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """编码表加载失败（如离线拉取 BPE 失败）时降级，不中断会话链路。"""
    fake = types.ModuleType("tiktoken")

    def _boom(name: str) -> Any:
        raise RuntimeError(f"cannot load encoding {name!r}")

    fake.get_encoding = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tiktoken", fake)

    counter = TiktokenTokenCounter()
    assert counter.count("你好") == 2
    assert counter.degraded is True


def test_tiktoken_counter_uses_custom_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """可注入自定义降级计数器。"""
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    counter = TiktokenTokenCounter(fallback=_CharCounter())
    assert counter.count("abcdef") == 6


def test_tiktoken_counter_with_real_encoding() -> None:
    """真实 tiktoken 编码表可用时返回精确计数（离线不可达时跳过）。"""
    pytest.importorskip("tiktoken")
    counter = TiktokenTokenCounter()
    assert counter.count("hello") > 0
    if counter.degraded:
        pytest.skip("tiktoken encoding data unavailable offline")
    assert counter.count("hello") == 1
    assert counter.count("") == 0


# --------------------------------------------------------------------- #
# 计数器名称解析
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("spec", [None, "heuristic", "char", "default"])
def test_resolve_counter_defaults_to_heuristic(spec: str | None) -> None:
    """空值与启发式别名均解析为启发式计数器。"""
    assert isinstance(resolve_token_counter(spec), HeuristicTokenCounter)


def test_resolve_counter_tiktoken_alias_uses_default_encoding() -> None:
    """``"tiktoken"`` 使用默认编码表。"""
    counter = resolve_token_counter("tiktoken")
    assert isinstance(counter, TiktokenTokenCounter)
    assert counter.encoding_name == DEFAULT_TIKTOKEN_ENCODING


def test_resolve_counter_explicit_encoding_name() -> None:
    """白名单内的编码表名被接受。"""
    counter = resolve_token_counter("o200k_base")
    assert isinstance(counter, TiktokenTokenCounter)
    assert counter.encoding_name == "o200k_base"


def test_resolve_counter_passthrough_instance() -> None:
    """已是计数器实例时原样返回。"""
    counter = _CharCounter()
    assert resolve_token_counter(counter) is counter


def test_resolve_counter_unknown_name_raises() -> None:
    """未知名称直接报错，避免拼写错误被静默降级。"""
    with pytest.raises(ValueError, match="unknown token counter"):
        resolve_token_counter("tiktokn")


# --------------------------------------------------------------------- #
# 预算策略校验
# --------------------------------------------------------------------- #
def test_policy_defaults_and_effective_budget() -> None:
    """默认值符合预期，有效预算扣除预留额度。"""
    policy = TokenBudgetPolicy(max_tokens=1000, reserve_tokens=200)
    assert policy.effective_budget == 800
    assert policy.keep_system_messages is True
    assert policy.keep_chief_complaint is True
    assert policy.warn_ratio == 0.9
    assert policy.prefer_stored_token_count is True


@pytest.mark.parametrize("max_tokens", [0, -1])
def test_policy_rejects_non_positive_budget(max_tokens: int) -> None:
    """总预算必须为正。"""
    with pytest.raises(PydanticValidationError):
        TokenBudgetPolicy(max_tokens=max_tokens)


def test_policy_rejects_negative_reserve() -> None:
    """预留额度不能为负。"""
    with pytest.raises(PydanticValidationError):
        TokenBudgetPolicy(max_tokens=10, reserve_tokens=-1)


@pytest.mark.parametrize("reserve", [10, 11])
def test_policy_rejects_reserve_not_smaller_than_budget(reserve: int) -> None:
    """预留额度必须小于总预算，保证有效预算为正。"""
    with pytest.raises(PydanticValidationError):
        TokenBudgetPolicy(max_tokens=10, reserve_tokens=reserve)


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_policy_rejects_out_of_range_warn_ratio(ratio: float) -> None:
    """告警阈值必须落在 (0, 1]。"""
    with pytest.raises(PydanticValidationError):
        TokenBudgetPolicy(max_tokens=10, warn_ratio=ratio)


def test_policy_forbids_unknown_fields() -> None:
    """策略不可扩展未知字段。"""
    with pytest.raises(PydanticValidationError):
        TokenBudgetPolicy(max_tokens=10, unknown=1)  # type: ignore[call-arg]


def test_policy_is_frozen() -> None:
    """策略不可变。"""
    policy = TokenBudgetPolicy(max_tokens=10)
    with pytest.raises(PydanticValidationError):
        policy.max_tokens = 20  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 预算裁剪：基础行为
# --------------------------------------------------------------------- #
def test_trimmer_defaults_to_heuristic_counter() -> None:
    """未注入计数器时使用零依赖的启发式计数器。"""
    trimmer = TokenBudgetTrimmer(TokenBudgetPolicy(max_tokens=10))
    assert isinstance(trimmer.counter, HeuristicTokenCounter)
    assert trimmer.policy.max_tokens == 10


def test_trim_empty_sequence() -> None:
    """空输入返回空结果与零值报告。"""
    result = _trimmer(max_tokens=100).trim([])
    assert result.messages == []
    assert result.report.applied is True
    assert result.report.budget_tokens == 100
    assert result.report.effective_budget == 100
    assert result.report.used_tokens == 0
    assert result.report.warnings == []


def test_trim_keeps_all_when_within_budget() -> None:
    """预算充足时不裁剪、不告警。"""
    messages = [_ai("aaaa"), _ai("bbbb"), _ai("cccc")]
    result = _trimmer(max_tokens=100).trim(messages)
    assert _contents(result.messages) == ["aaaa", "bbbb", "cccc"]
    assert result.report.kept_messages == 3
    assert result.report.dropped_messages == 0
    assert result.report.used_tokens == 12
    assert result.report.over_budget is False
    assert result.report.warnings == []


def test_trim_drops_oldest_to_fit_budget() -> None:
    """预算不足时从最旧开始丢弃，保留最近的连续后缀。"""
    messages = [_ai("aaaa"), _ai("bbbb"), _ai("cccc")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["bbbb", "cccc"]
    assert result.report.kept_messages == 2
    assert result.report.dropped_messages == 1
    assert result.report.used_tokens == 8
    assert result.report.dropped_tokens == 4
    assert any("dropped 1 message(s)" in w for w in result.report.warnings)


def test_trim_does_not_skip_oversized_middle_message() -> None:
    """放不下的消息直接截断，不跳过它去取更旧的消息（保对话连续）。"""
    messages = [_ai("aaaa"), _ai("b" * 100), _ai("cccc")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["cccc"]
    assert result.report.dropped_messages == 2


def test_trim_preserves_original_order() -> None:
    """输出保持原始时序升序。"""
    messages = [_sys("ss"), _ai("aaaa"), _ai("bbbb")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["ss", "bbbb"]


def test_trim_skips_protected_message_during_reverse_walk() -> None:
    """反向贪心遇到被保护消息时跳过（不计入连续后缀），继续向前累加。"""
    messages = [_ai("aaaa"), _sys("ss"), _ai("bbbb")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["ss", "bbbb"]
    assert result.report.used_tokens == 6
    assert result.report.dropped_messages == 1


def test_trim_does_not_mutate_input() -> None:
    """裁剪不修改入参序列。"""
    messages = [_ai("aaaa"), _ai("bbbb"), _ai("cccc")]
    snapshot = list(messages)
    _trimmer(max_tokens=4).trim(messages)
    assert messages == snapshot


# --------------------------------------------------------------------- #
# 预算裁剪：保护位
# --------------------------------------------------------------------- #
def test_system_message_is_protected_and_counted() -> None:
    """系统槽位消息始终保留，且占用计入预算。"""
    messages = [_sys("ss"), _ai("aaaa"), _ai("bbbb")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["ss", "bbbb"]
    assert result.report.used_tokens == 6
    assert result.report.dropped_messages == 1


def test_chief_complaint_is_protected() -> None:
    """首条患者主诉消息始终保留。"""
    messages = [_human("aaaa"), _ai("bbbb"), _ai("cccc")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["aaaa", "cccc"]


def test_chief_complaint_protection_can_be_disabled() -> None:
    """关闭主诉保护后，主诉消息按普通消息参与裁剪。"""
    messages = [_human("aaaa"), _ai("bbbb"), _ai("cccc")]
    result = _trimmer(max_tokens=8, keep_chief_complaint=False).trim(messages)
    assert _contents(result.messages) == ["bbbb", "cccc"]


def test_chief_complaint_detected_via_med_role() -> None:
    """医疗角色扩展字段标记为 patient 的消息同样视为主诉。"""
    patient = AIMessage(content="aaaa", additional_kwargs={MED_ROLE_KEY: "patient"})
    messages: list[BaseMessage] = [patient, _ai("bbbb"), _ai("cccc")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["aaaa", "cccc"]


def test_system_protection_can_be_disabled() -> None:
    """关闭系统保护后，系统消息可被裁掉。"""
    messages = [_sys("s" * 20), _ai("aaaa")]
    result = _trimmer(max_tokens=8, keep_system_messages=False).trim(messages)
    assert _contents(result.messages) == ["aaaa"]
    assert result.report.over_budget is False


def test_protected_message_over_budget_reports_over_budget() -> None:
    """被保护消息自身超预算时产出超限告警，但依旧不丢弃。"""
    messages = [_sys("s" * 20), _ai("aaaa")]
    result = _trimmer(max_tokens=8).trim(messages)
    assert _contents(result.messages) == ["s" * 20]
    assert result.report.over_budget is True
    assert result.report.used_tokens == 20
    assert result.report.warnings[0].startswith("token budget exceeded")


# --------------------------------------------------------------------- #
# 预算裁剪：预留额度、告警阈值、已存 token 数
# --------------------------------------------------------------------- #
def test_reserve_tokens_shrinks_effective_budget() -> None:
    """预留输出额度后，上下文可用预算相应缩小。"""
    messages = [_ai("aaaa"), _ai("bbbb"), _ai("cccc")]
    result = _trimmer(max_tokens=10, reserve_tokens=4).trim(messages)
    assert result.report.budget_tokens == 10
    assert result.report.effective_budget == 6
    assert _contents(result.messages) == ["cccc"]


def test_near_budget_emits_warning() -> None:
    """使用率超过告警阈值时产出「预算将尽」告警。"""
    messages = [_ai("aaaa"), _ai("bbbb")]
    result = _trimmer(max_tokens=10, warn_ratio=0.5).trim(messages)
    assert result.report.used_tokens == 8
    assert any("nearly exhausted" in w for w in result.report.warnings)


def test_warn_ratio_one_only_warns_when_exceeded() -> None:
    """告警阈值取 1.0 时，用满但未超出不告警。"""
    messages = [_ai("aaaa"), _ai("bbbb")]
    result = _trimmer(max_tokens=8, warn_ratio=1.0).trim(messages)
    assert result.report.used_tokens == 8
    assert result.report.over_budget is False
    assert result.report.warnings == []


def test_prefers_stored_token_count() -> None:
    """默认复用消息上已算好的 token 数，避免重复编码。"""
    messages = [_with_stored("a" * 100, 5)]
    result = _trimmer(max_tokens=5).trim(messages)
    assert _contents(result.messages) == ["a" * 100]
    assert result.report.used_tokens == 5


def test_stored_token_count_can_be_ignored() -> None:
    """关闭复用后按计数器实时计算，超预算消息被裁掉。"""
    messages = [_with_stored("a" * 100, 5)]
    result = _trimmer(max_tokens=5, prefer_stored_token_count=False).trim(messages)
    assert result.messages == []
    assert result.report.used_tokens == 0
    assert result.report.dropped_messages == 1


def test_stored_token_count_reader() -> None:
    """``token_count`` 为 0 / 非法类型时视为未计算。"""
    assert stored_token_count(_with_stored("x", 7)) == 7
    assert stored_token_count(_ai("x")) is None
    assert stored_token_count(AIMessage(content="x", additional_kwargs={"token_count": 0})) is None
    assert (
        stored_token_count(AIMessage(content="x", additional_kwargs={"token_count": "7"})) is None
    )


# --------------------------------------------------------------------- #
# 预算裁剪：计数与存储句柄
# --------------------------------------------------------------------- #
def test_count_uses_same_cost_policy() -> None:
    """``count`` 与裁剪使用同一预算占用口径。"""
    trimmer = _trimmer(max_tokens=100)
    messages = [_with_stored("a" * 100, 5), _ai("aaaa")]
    assert trimmer.count(messages) == 9


def test_trim_history_reads_store_messages() -> None:
    """可直接裁剪存储句柄中的全量消息。"""
    history = StoreFactory.create(
        "memory",
        session_id="s-token-hist",
        tenant_id=TENANT_ID,
        dept_id=DEPT_ID,
        patient_id=PATIENT_ID,
    )
    history.clear()
    history.add_messages([_ai("aaaa"), _ai("bbbb"), _ai("cccc")])

    result = _trimmer(max_tokens=8).trim_history(history)
    assert _contents(result.messages) == ["bbbb", "cccc"]
    assert result.report.kept_messages == 2


# --------------------------------------------------------------------- #
# 与 MedRunnableWithMessageHistory 的集成
# --------------------------------------------------------------------- #
def test_runnable_without_budget_returns_unapplied_report() -> None:
    """未注入预算策略时报告标记未生效，消息原样返回。"""
    rwh = MedRunnableWithMessageHistory(_echo_runnable(), backend="memory")
    messages = [_ai("aaaa"), _ai("bbbb")]
    result = rwh.build_context(messages)
    assert result.messages == messages
    assert result.report.applied is False
    assert result.report.warnings == []
    assert rwh.token_budget is None


def test_runnable_token_pipeline_applies_both_stages() -> None:
    """时序窗口先裁、Token 预算后裁，两级叠加产出最终上下文。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        trim_policy=ContextWindowPolicy(max_messages=1),
        token_budget=TokenBudgetPolicy(max_tokens=4),
        token_counter=_CharCounter(),
    )
    messages: list[BaseMessage] = [
        _human("aaaa", created_at=BASE_MS),
        _ai("bbbb", created_at=BASE_MS + 1000),
        _ai("cccc", created_at=BASE_MS + 2000),
    ]
    result = rwh.build_context(messages, now_ms=BASE_MS + 2000)

    # 时序窗口保留主诉 + 最近 1 条对话，随后预算层再裁掉超预算的主诉之后那条。
    assert _contents(result.messages) == ["aaaa"]
    assert result.report.applied is True
    assert result.report.dropped_messages == 1
    assert result.report.used_tokens == 4
    assert any("nearly exhausted" in w for w in result.report.warnings)


def test_runnable_exposes_policy_and_counter() -> None:
    """属性暴露注入的策略与解析后的计数器。"""
    policy = TokenBudgetPolicy(max_tokens=64)
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        token_budget=policy,
        token_counter="tiktoken",
    )
    assert rwh.token_budget is policy
    assert isinstance(rwh.token_counter, TiktokenTokenCounter)


def test_runnable_rejects_unknown_counter_name() -> None:
    """未知计数器名称在构造时即报错。"""
    with pytest.raises(ValueError, match="unknown token counter"):
        MedRunnableWithMessageHistory(
            _echo_runnable(),
            backend="memory",
            token_counter="nope",
        )


def test_runnable_build_context_does_not_mutate_input() -> None:
    """两级流水线不修改入参序列。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        trim_policy=ContextWindowPolicy(max_messages=1),
        token_budget=TokenBudgetPolicy(max_tokens=4),
        token_counter=_CharCounter(),
    )
    messages: list[BaseMessage] = [_human("aaaa"), _ai("bbbb"), _ai("cccc")]
    snapshot = list(messages)
    rwh.build_context(messages)
    assert messages == snapshot
