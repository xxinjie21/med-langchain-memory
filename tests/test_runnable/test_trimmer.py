"""时序上下文裁剪单元测试（D25）。

覆盖：裁剪策略校验、结构化字段解析（``created_at`` / ``med_role``）、主诉与系统
槽位定位、按条数 / 按时间窗 / 组合裁剪的边界行为，以及与
:class:`MedRunnableWithMessageHistory`、存储句柄的集成。

全部用例零外部依赖：内存后端 + 固定时间戳即可跑通。
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError as PydanticValidationError

import med_langchain_memory
from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.runnable import (
    ContextWindowPolicy,
    MedRunnableWithMessageHistory,
    TimeWindowTrimmer,
    extract_created_at,
    find_chief_complaint,
    is_system_message,
    med_role_of,
)
from med_langchain_memory.runnable.trimmer import CREATED_AT_KEY
from med_langchain_memory.stores import InMemoryMedHistory
from med_langchain_memory.stores.base import MED_ROLE_KEY, to_langchain_message

# --------------------------------------------------------------------- #
# 测试夹具与替身
# --------------------------------------------------------------------- #

#: 固定基准时间戳（epoch millis），保证时间窗用例可复现。
BASE_MS = 1_700_000_000_000

TENANT_ID = "h-a"
DEPT_ID = "cardiology"
PATIENT_ID = "p-1"
SESSION_ID = "s-1"

_NAMESPACE: dict[str, str] = {
    "session_id": SESSION_ID,
    "tenant_id": TENANT_ID,
    "dept_id": DEPT_ID,
    "patient_id": PATIENT_ID,
}


def _med(role: MessageRole, content: str, created_at: int, session_id: str = SESSION_ID) -> MedMessage:
    """构造一条固定命名空间的医疗消息。"""
    return MedMessage(
        role=role,
        content=content,
        created_at=created_at,
        metadata={"seq": content},
        session_id=session_id,
        tenant_id=TENANT_ID,
        dept_id=DEPT_ID,
        patient_id=PATIENT_ID,
    )


def _msg(role: MessageRole, content: str, created_at: int) -> BaseMessage:
    """构造携带医疗扩展字段的 LangChain 消息。"""
    return to_langchain_message(_med(role, content, created_at))


def _conversation(count: int, *, start: int = 0, step_ms: int = 1000) -> list[BaseMessage]:
    """构造交替的患者/助手对话，内容编号与时间戳自 ``start`` 起依次递增。"""
    messages: list[BaseMessage] = []
    for offset in range(count):
        seq = start + offset
        role = MessageRole.PATIENT if offset % 2 == 0 else MessageRole.ASSISTANT
        messages.append(_msg(role, f"m{seq}", BASE_MS + seq * step_ms))
    return messages


def _contents(messages: list[BaseMessage]) -> list[str]:
    """提取消息正文，便于断言裁剪结果。"""
    return [str(m.content) for m in messages]


def _echo() -> RunnableLambda:
    """构造无需外部 LLM 的下游 Runnable。"""
    return RunnableLambda(lambda payload: {"echo": payload})


# --------------------------------------------------------------------- #
# ContextWindowPolicy
# --------------------------------------------------------------------- #


def test_policy_defaults_disable_all_trimming() -> None:
    """默认策略不做任何裁剪，且两种保护位均开启。"""
    policy = ContextWindowPolicy()

    assert policy.max_messages is None
    assert policy.max_age_seconds is None
    assert policy.keep_system_messages is True
    assert policy.keep_chief_complaint is True


@pytest.mark.parametrize(
    "payload",
    [
        {"max_messages": 0},
        {"max_messages": -1},
        {"max_age_seconds": 0},
        {"max_age_seconds": -5},
    ],
)
def test_policy_rejects_non_positive_limits(payload: dict[str, Any]) -> None:
    """条数上限必须 ≥1、时间窗必须 >0，否则构造失败。"""
    with pytest.raises(PydanticValidationError):
        ContextWindowPolicy(**payload)


def test_policy_is_immutable() -> None:
    """策略为不可变值对象，禁止就地改写。"""
    policy = ContextWindowPolicy(max_messages=3)

    with pytest.raises(PydanticValidationError):
        policy.max_messages = 5  # type: ignore[misc]


def test_policy_rejects_unknown_field() -> None:
    """策略禁止传入未声明字段，避免配置拼写错误被静默忽略。"""
    with pytest.raises(PydanticValidationError):
        ContextWindowPolicy(max_token=10)  # type: ignore[call-arg]


# --------------------------------------------------------------------- #
# 结构化字段解析
# --------------------------------------------------------------------- #


def test_extract_created_at_reads_millis() -> None:
    """能正确读取扩展字段中的 epoch 毫秒时间戳。"""
    message = _msg(MessageRole.PATIENT, "头疼", BASE_MS)

    assert extract_created_at(message) == BASE_MS


@pytest.mark.parametrize("raw", [None, "1700000000000", True, 0, -1])
def test_extract_created_at_returns_none_for_invalid(raw: Any) -> None:
    """缺失或非法时间戳返回 ``None``（对应消息在裁剪时保守保留）。"""
    message = AIMessage(content="x", additional_kwargs={CREATED_AT_KEY: raw})

    assert extract_created_at(message) is None


def test_med_role_prefers_extension_field() -> None:
    """优先采用 ``med_role`` 扩展字段，而非 LangChain 类型推断。"""
    message = AIMessage(content="x", additional_kwargs={MED_ROLE_KEY: MessageRole.DOCTOR.value})

    assert med_role_of(message) is MessageRole.DOCTOR


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (AIMessage(content="x"), MessageRole.ASSISTANT),
        (SystemMessage(content="x"), MessageRole.SYSTEM),
        (ToolMessage(content="x", tool_call_id="t-1"), None),
    ],
)
def test_med_role_fallback_and_unsupported(
    message: BaseMessage, expected: MessageRole | None
) -> None:
    """无扩展字段时按类型回退；不支持的消息类型返回 ``None``。"""
    assert med_role_of(message) is expected


def test_med_role_returns_none_for_unknown_value() -> None:
    """未知角色值不抛异常，返回 ``None``。"""
    message = AIMessage(content="x", additional_kwargs={MED_ROLE_KEY: "nurse"})

    assert med_role_of(message) is None


def test_is_system_message_detects_system_slot() -> None:
    """系统消息可被识别（无论是否携带扩展字段）。"""
    assert is_system_message(SystemMessage(content="规范")) is True
    assert is_system_message(_msg(MessageRole.SYSTEM, "规范", BASE_MS)) is True


def test_is_system_message_rejects_conversation_message() -> None:
    """普通对话消息不被当作系统槽位。"""
    assert is_system_message(_msg(MessageRole.PATIENT, "头疼", BASE_MS)) is False
    assert is_system_message(AIMessage(content="建议复诊")) is False


def test_find_chief_complaint_returns_first_patient_message() -> None:
    """主诉为首条患者消息。"""
    messages = [
        _msg(MessageRole.SYSTEM, "规范", BASE_MS),
        _msg(MessageRole.PATIENT, "主诉", BASE_MS + 1),
        _msg(MessageRole.PATIENT, "补充", BASE_MS + 2),
    ]

    assert find_chief_complaint(messages) == 1


def test_find_chief_complaint_returns_none_without_patient() -> None:
    """会话中不存在患者消息时返回 ``None``。"""
    messages = [_msg(MessageRole.DOCTOR, "问诊", BASE_MS), _msg(MessageRole.ASSISTANT, "建议", BASE_MS)]

    assert find_chief_complaint(messages) is None


def test_find_chief_complaint_skips_doctor_opening() -> None:
    """医生开场白不算主诉，主诉取后续首条患者消息。"""
    messages = [
        _msg(MessageRole.DOCTOR, "请问哪里不舒服", BASE_MS),
        _msg(MessageRole.PATIENT, "胸闷", BASE_MS + 1),
    ]

    assert find_chief_complaint(messages) == 1


# --------------------------------------------------------------------- #
# TimeWindowTrimmer：基础与条数裁剪
# --------------------------------------------------------------------- #


def test_trimmer_exposes_injected_policy() -> None:
    """裁剪器通过 ``policy`` 属性暴露注入的策略实例。"""
    policy = ContextWindowPolicy(max_messages=3, max_age_seconds=60)
    trimmer = TimeWindowTrimmer(policy)

    assert trimmer.policy is policy
    assert trimmer.policy.max_messages == 3
    assert trimmer.policy.max_age_seconds == 60


def test_trimmer_default_policy_disables_trimming() -> None:
    """策略缺省构造时裁剪器不产生任何裁剪。"""
    trimmer = TimeWindowTrimmer(ContextWindowPolicy())
    messages = _conversation(4)

    assert trimmer.policy.max_messages is None
    assert _contents(trimmer.trim(messages, now_ms=BASE_MS)) == ["m0", "m1", "m2", "m3"]


def test_trim_empty_sequence() -> None:
    """空会话裁剪后仍为空。"""
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2))

    assert trimmer.trim([]) == []


def test_trim_without_limits_returns_copy() -> None:
    """未配置任何上限时原样返回等价副本。"""
    messages = _conversation(3)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy())

    result = trimmer.trim(messages)

    assert _contents(result) == ["m0", "m1", "m2"]
    assert result is not messages


def test_trim_by_count_keeps_latest_messages_in_order() -> None:
    """按条数裁剪保留会话尾部消息，且保持时序升序。"""
    messages = _conversation(5)
    trimmer = TimeWindowTrimmer(
        ContextWindowPolicy(max_messages=2, keep_chief_complaint=False)
    )

    assert _contents(trimmer.trim(messages)) == ["m3", "m4"]


@pytest.mark.parametrize("limit", [5, 6])
def test_trim_by_count_noop_when_limit_covers_all(limit: int) -> None:
    """上限不小于消息数时不做裁剪。"""
    messages = _conversation(5)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=limit))

    assert _contents(trimmer.trim(messages)) == ["m0", "m1", "m2", "m3", "m4"]


def test_trim_by_count_keeps_system_message_outside_budget() -> None:
    """系统槽位消息被保护，不占用条数预算。"""
    messages = [SystemMessage(content="规范"), *_conversation(4, start=1)]
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2, keep_chief_complaint=False))

    assert _contents(trimmer.trim(messages)) == ["规范", "m3", "m4"]


def test_trim_by_count_keeps_chief_complaint_outside_budget() -> None:
    """主诉消息被保护，即便已滑出尾部窗口也保留在最前。"""
    messages = _conversation(5)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2))

    assert _contents(trimmer.trim(messages)) == ["m0", "m3", "m4"]


def test_trim_drops_system_when_protection_disabled() -> None:
    """关闭系统消息保护后，系统消息与普通消息一样参与裁剪。"""
    messages = [SystemMessage(content="规范"), *_conversation(3, start=1)]
    policy = ContextWindowPolicy(
        max_messages=2, keep_system_messages=False, keep_chief_complaint=False
    )

    assert _contents(TimeWindowTrimmer(policy).trim(messages)) == ["m2", "m3"]


def test_trim_drops_chief_complaint_when_protection_disabled() -> None:
    """关闭主诉保护后，主诉不再被强制保留。"""
    messages = _conversation(4)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2, keep_chief_complaint=False))

    assert _contents(trimmer.trim(messages)) == ["m2", "m3"]


def test_trim_does_not_mutate_input() -> None:
    """裁剪为纯函数，不修改入参序列。"""
    messages = _conversation(4)
    before = _contents(messages)

    TimeWindowTrimmer(ContextWindowPolicy(max_messages=1)).trim(messages)

    assert _contents(messages) == before


def test_trim_accepts_arbitrary_sequence() -> None:
    """支持任意序列类型（如元组）作为输入。"""
    messages = tuple(_conversation(3))
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=1, keep_chief_complaint=False))

    assert _contents(trimmer.trim(messages)) == ["m2"]


# --------------------------------------------------------------------- #
# TimeWindowTrimmer：时间窗裁剪
# --------------------------------------------------------------------- #


def test_trim_by_time_window_drops_stale_messages() -> None:
    """早于时间窗的消息被丢弃，窗内消息保留。"""
    messages = _conversation(4)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_age_seconds=2, keep_chief_complaint=False))
    now_ms = BASE_MS + 3_000

    assert _contents(trimmer.trim(messages, now_ms=now_ms)) == ["m1", "m2", "m3"]


def test_trim_by_time_window_keeps_cutoff_boundary() -> None:
    """恰好落在窗口边界（``created_at == cutoff``）的消息保留。"""
    messages = _conversation(3)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_age_seconds=2))
    now_ms = BASE_MS + 2_000

    assert _contents(trimmer.trim(messages, now_ms=now_ms)) == ["m0", "m1", "m2"]


def test_trim_by_time_window_keeps_messages_without_timestamp() -> None:
    """缺失时间戳的消息无法判定时序，保守保留。"""
    messages = [
        AIMessage(content="无时间戳"),
        _msg(MessageRole.PATIENT, "近期", BASE_MS + 5_000),
    ]
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_age_seconds=1))

    assert _contents(trimmer.trim(messages, now_ms=BASE_MS + 5_000)) == ["无时间戳", "近期"]


def test_trim_by_time_window_protects_system_and_chief_complaint() -> None:
    """时间窗裁剪同样不淘汰系统槽位与主诉。"""
    messages = [
        SystemMessage(content="规范"),
        _msg(MessageRole.PATIENT, "主诉", BASE_MS),
        _msg(MessageRole.ASSISTANT, "建议", BASE_MS),
    ]
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_age_seconds=1))

    assert _contents(trimmer.trim(messages, now_ms=BASE_MS + 60_000)) == ["规范", "主诉"]


def test_trim_combines_time_window_and_count() -> None:
    """时间窗与条数上限叠加生效：先过滤时间，再截取尾部。"""
    messages = _conversation(6)
    now_ms = BASE_MS + 4_000

    plain = TimeWindowTrimmer(
        ContextWindowPolicy(max_messages=2, max_age_seconds=3, keep_chief_complaint=False)
    )
    assert _contents(plain.trim(messages, now_ms=now_ms)) == ["m4", "m5"]

    guarded = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2, max_age_seconds=3))
    assert _contents(guarded.trim(messages, now_ms=now_ms)) == ["m0", "m4", "m5"]


def test_trim_by_time_window_defaults_to_system_clock() -> None:
    """未显式传入 ``now_ms`` 时使用系统时间，陈旧历史只剩受保护位。"""
    messages = _conversation(3)
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_age_seconds=1))

    assert _contents(trimmer.trim(messages)) == ["m0"]


# --------------------------------------------------------------------- #
# 与存储句柄 / Runnable 的集成
# --------------------------------------------------------------------- #


def test_trim_history_reads_from_store() -> None:
    """可直接裁剪存储句柄中的全量消息。"""
    history = InMemoryMedHistory(**{**_NAMESPACE, "session_id": "s-trim-1"})
    history.add_med_messages(
        [_med(MessageRole.PATIENT, f"m{i}", BASE_MS + i, session_id="s-trim-1") for i in range(4)]
    )
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2))

    assert _contents(trimmer.trim_history(history)) == ["m0", "m2", "m3"]


def test_trim_history_honours_explicit_now_ms() -> None:
    """``trim_history`` 支持显式基准时间，陈旧历史只保留受保护位。"""
    history = InMemoryMedHistory(**{**_NAMESPACE, "session_id": "s-trim-stale"})
    history.add_med_messages(
        [_med(MessageRole.PATIENT, f"m{i}", BASE_MS + i, session_id="s-trim-stale") for i in range(3)]
    )
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_age_seconds=1))

    assert _contents(trimmer.trim_history(history, now_ms=BASE_MS + 600_000)) == ["m0"]


def test_trim_history_on_empty_store() -> None:
    """空会话句柄裁剪结果为空（内存后端按会话键共享，需使用独立会话 ID）。"""
    history = InMemoryMedHistory(**{**_NAMESPACE, "session_id": "s-trim-empty"})
    trimmer = TimeWindowTrimmer(ContextWindowPolicy(max_messages=2))

    assert trimmer.trim_history(history) == []


def test_runnable_trim_policy_defaults_to_none() -> None:
    """未注入裁剪策略时 Runnable 的 ``trim_policy`` 为 ``None``。"""
    runnable = MedRunnableWithMessageHistory(_echo(), backend="memory")

    assert runnable.trim_policy is None


def test_runnable_trim_context_applies_policy() -> None:
    """注入策略后 Runnable 可裁剪上下文，并保留主诉。"""
    runnable = MedRunnableWithMessageHistory(
        _echo(), backend="memory", trim_policy=ContextWindowPolicy(max_messages=2)
    )

    assert _contents(runnable.trim_context(_conversation(5))) == ["m0", "m3", "m4"]


def test_runnable_trim_context_without_policy_returns_copy() -> None:
    """未注入策略时 ``trim_context`` 返回等价副本。"""
    runnable = MedRunnableWithMessageHistory(_echo(), backend="memory")
    messages = _conversation(2)

    result = runnable.trim_context(messages)

    assert _contents(result) == ["m0", "m1"]
    assert result is not messages


def test_trimmer_symbols_are_exported() -> None:
    """裁剪相关符号由 ``runnable`` 包统一导出。"""
    assert med_langchain_memory.runnable.TimeWindowTrimmer is TimeWindowTrimmer
    assert med_langchain_memory.runnable.ContextWindowPolicy is ContextWindowPolicy
