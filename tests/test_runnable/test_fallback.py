"""读写降级兜底单元测试（D29）。

覆盖：熔断策略校验与状态机（闭合计数 → 打开 → 恢复窗口半开探测 → 闭合 / 重新打开）、
熔断器快照与重置、降级策略链校验、降级报告派生属性、``FallbackHistory`` 的读 / 写 /
清理降级与熔断跳过、``FallbackHistoryResolver`` 的句柄组装与熔断器共享，以及
``MedRunnableWithMessageHistory`` 注入 ``fallback_policy`` 后的端到端降级。

全部用例零真实中间件：用进程内单实例存储替身注入读写故障，用假时钟驱动恢复窗口。
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.exceptions import (
    FallbackExhaustedError,
    StorageError,
    ValidationError,
)
from med_langchain_memory.runnable import (
    DEFAULT_FAILURE_THRESHOLD,
    DEFAULT_HALF_OPEN_MAX_CALLS,
    DEFAULT_RECOVERY_TIMEOUT_MS,
    DEFAULT_SUCCESS_THRESHOLD,
    CircuitBreaker,
    CircuitBreakerPolicy,
    CircuitState,
    FallbackAttempt,
    FallbackHistory,
    FallbackHistoryResolver,
    FallbackPolicy,
    FallbackReport,
    MedRunnableWithMessageHistory,
)
from med_langchain_memory.stores import MedChatMessageHistory

# --------------------------------------------------------------------- #
# 测试替身与夹具
# --------------------------------------------------------------------- #

TENANT = "h-a"
DEPT = "cardio"
PATIENT = "p-1"


class _FakeHistory(MedChatMessageHistory):
    """进程内单实例存储替身：每个实例持有独立消息列表，并可注入读写故障。

    与 ``InMemoryMedHistory`` 不同，本替身**不做跨实例共享**，因此主后端与备后端
    可分别断言各自的落库内容；故障按次数消耗（``fail_*`` 为剩余失败配额）。
    """

    def __init__(
        self,
        *,
        session_id: str,
        tenant_id: str = TENANT,
        dept_id: str = DEPT,
        patient_id: str = PATIENT,
        ttl_seconds: int | None = None,
        fail_reads: int = 0,
        fail_writes: int = 0,
        fail_clears: int = 0,
    ) -> None:
        super().__init__(
            session_id=session_id,
            tenant_id=tenant_id,
            dept_id=dept_id,
            patient_id=patient_id,
            ttl_seconds=ttl_seconds,
        )
        self._messages: list[MedMessage] = []
        self.fail_reads = fail_reads
        self.fail_writes = fail_writes
        self.fail_clears = fail_clears
        self.read_calls = 0
        self.write_calls = 0
        self.clear_calls = 0

    def _append(self, messages: list[MedMessage]) -> None:
        """追加消息；失败配额未耗尽时抛 :class:`StorageError`。"""
        self.write_calls += 1
        if self.fail_writes > 0:
            self.fail_writes -= 1
            raise StorageError("write unavailable")
        self._messages.extend(messages)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """读取消息；失败配额未耗尽时抛 :class:`StorageError`。"""
        self.read_calls += 1
        if self.fail_reads > 0:
            self.fail_reads -= 1
            raise StorageError("read unavailable")
        return list(self._messages) if limit is None else list(self._messages[-limit:])

    def clear(self) -> None:
        """清空消息；失败配额未耗尽时抛 :class:`StorageError`。"""
        self.clear_calls += 1
        if self.fail_clears > 0:
            self.fail_clears -= 1
            raise StorageError("clear unavailable")
        self._messages.clear()

    @property
    def stored(self) -> list[MedMessage]:
        """本实例已落库的消息（供断言）。"""
        return list(self._messages)


class _FakeClock:
    """可手动推进的毫秒假时钟。"""

    def __init__(self, now_ms: int = 1_000_000) -> None:
        self.now_ms = now_ms

    def __call__(self) -> int:
        """返回当前假时刻（epoch 毫秒）。"""
        return self.now_ms

    def advance(self, delta_ms: int) -> None:
        """推进假时钟。"""
        self.now_ms += delta_ms


class _FakeFactory:
    """按后端名返回预设句柄的工厂替身，并记录 ``create`` 调用参数。"""

    def __init__(self, handles: dict[str, MedChatMessageHistory]) -> None:
        self.handles = handles
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def create(self, backend: str, **kwargs: Any) -> MedChatMessageHistory:
        """记录调用并返回预设句柄。"""
        self.calls.append((backend, kwargs))
        return self.handles[backend]


@pytest.fixture
def session_id() -> str:
    """每次调用生成互不相同的会话 ID，避免用例间存储串扰。"""
    return f"s-{uuid.uuid4().hex[:8]}"


def _handle(session_id: str, **kwargs: Any) -> _FakeHistory:
    """构造带故障注入能力的主/备句柄替身。"""
    return _FakeHistory(session_id=session_id, **kwargs)


def _message(session_id: str, content: str) -> MedMessage:
    """构造归属于 ``session_id`` 的医疗消息。"""
    return MedMessage(
        session_id=session_id,
        tenant_id=TENANT,
        dept_id=DEPT,
        patient_id=PATIENT,
        role=MessageRole.PATIENT,
        content=content,
    )


def _history(
    session_id: str,
    *,
    primary: _FakeHistory,
    fallback: _FakeHistory,
    policy: FallbackPolicy | None = None,
    breakers: dict[str, CircuitBreaker] | None = None,
) -> FallbackHistory:
    """组装主/备两后端的 :class:`FallbackHistory`。"""
    return FallbackHistory(
        policy=policy or FallbackPolicy(primary="memory", fallbacks=("memory_fb",)),
        handles={"memory": primary, "memory_fb": fallback},
        breakers=breakers,
    )


# --------------------------------------------------------------------- #
# CircuitState / CircuitBreakerPolicy
# --------------------------------------------------------------------- #


def test_circuit_state_values() -> None:
    """三种熔断状态的字符串取值稳定（供报告与监控消费）。"""
    assert CircuitState.CLOSED.value == "closed"
    assert CircuitState.OPEN.value == "open"
    assert CircuitState.HALF_OPEN.value == "half_open"


def test_circuit_policy_defaults() -> None:
    """熔断策略默认值与模块常量一致。"""
    policy = CircuitBreakerPolicy()
    assert policy.failure_threshold == DEFAULT_FAILURE_THRESHOLD
    assert policy.recovery_timeout_ms == DEFAULT_RECOVERY_TIMEOUT_MS
    assert policy.half_open_max_calls == DEFAULT_HALF_OPEN_MAX_CALLS
    assert policy.success_threshold == DEFAULT_SUCCESS_THRESHOLD


@pytest.mark.parametrize(
    "field,value",
    [
        ("failure_threshold", 0),
        ("failure_threshold", -1),
        ("recovery_timeout_ms", -1),
        ("half_open_max_calls", 0),
        ("success_threshold", 0),
    ],
)
def test_circuit_policy_rejects_invalid_values(field: str, value: int) -> None:
    """阈值必须为正数、恢复窗口不得为负，否则 pydantic 校验失败。"""
    with pytest.raises(PydanticValidationError):
        CircuitBreakerPolicy(**{field: value})


def test_circuit_policy_is_frozen_and_forbids_extra() -> None:
    """熔断策略不可变且拒绝未知字段。"""
    policy = CircuitBreakerPolicy()
    with pytest.raises(PydanticValidationError):
        policy.failure_threshold = 9  # type: ignore[misc]
    with pytest.raises(PydanticValidationError):
        CircuitBreakerPolicy(unknown=1)  # type: ignore[call-arg]


# --------------------------------------------------------------------- #
# CircuitBreaker
# --------------------------------------------------------------------- #


def test_breaker_starts_closed_and_allows() -> None:
    """新建熔断器处于闭合状态并放行调用。"""
    breaker = CircuitBreaker("memory")
    assert breaker.name == "memory"
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow() is True
    assert breaker.snapshot().failure_count == 0


def test_breaker_empty_name_rejected() -> None:
    """熔断器名不得为空。"""
    with pytest.raises(ValidationError):
        CircuitBreaker("")


def test_breaker_opens_after_threshold_failures() -> None:
    """连续失败达到阈值即打开，随后拒绝放行并累计拒绝次数。"""
    breaker = CircuitBreaker("memory", policy=CircuitBreakerPolicy(failure_threshold=2))
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.allow() is False
    assert breaker.allow() is False
    assert breaker.snapshot().rejected_count == 2
    assert breaker.snapshot().failure_total == 2


def test_breaker_closed_success_resets_failure_count() -> None:
    """闭合状态下成功一次即清零连续失败计数。"""
    breaker = CircuitBreaker("memory", policy=CircuitBreakerPolicy(failure_threshold=3))
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    assert breaker.snapshot().failure_count == 0
    assert breaker.state is CircuitState.CLOSED
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state is CircuitState.CLOSED


def test_breaker_rejects_until_recovery_window_elapsed() -> None:
    """打开后未到恢复窗口一律拒绝，且打开时刻被记录。"""
    clock = _FakeClock()
    breaker = CircuitBreaker(
        "memory",
        policy=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_ms=500),
        clock=clock,
    )
    breaker.record_failure()
    assert breaker.snapshot().opened_at_ms == clock.now_ms
    clock.advance(499)
    assert breaker.allow() is False
    assert breaker.state is CircuitState.OPEN


def test_breaker_enters_half_open_after_recovery_window() -> None:
    """恢复窗口到期后转入半开并放行探测请求。"""
    clock = _FakeClock()
    breaker = CircuitBreaker(
        "memory",
        policy=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_ms=100),
        clock=clock,
    )
    breaker.record_failure()
    clock.advance(100)
    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN


def test_breaker_recovery_timeout_zero_probes_immediately() -> None:
    """恢复窗口为 0 时下一次调用即进入半开探测。"""
    breaker = CircuitBreaker(
        "memory",
        policy=CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_ms=0),
    )
    breaker.record_failure()
    assert breaker.allow() is True
    assert breaker.state is CircuitState.HALF_OPEN


def test_breaker_closes_after_half_open_success_threshold() -> None:
    """半开状态下连续成功达到闭合阈值即恢复闭合并清空计数。"""
    clock = _FakeClock()
    policy = CircuitBreakerPolicy(
        failure_threshold=1,
        recovery_timeout_ms=10,
        half_open_max_calls=2,
        success_threshold=2,
    )
    breaker = CircuitBreaker("memory", policy=policy, clock=clock)
    breaker.record_failure()
    clock.advance(10)
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.state is CircuitState.HALF_OPEN
    assert breaker.allow() is True
    breaker.record_success()
    assert breaker.state is CircuitState.CLOSED
    snapshot = breaker.snapshot()
    assert snapshot.failure_count == 0
    assert snapshot.opened_at_ms is None
    assert snapshot.success_total == 2


def test_breaker_reopens_immediately_on_half_open_failure() -> None:
    """半开探测失败立即重新打开（不等失败阈值）。"""
    clock = _FakeClock()
    breaker = CircuitBreaker(
        "memory",
        policy=CircuitBreakerPolicy(failure_threshold=5, recovery_timeout_ms=10),
        clock=clock,
    )
    for _ in range(5):
        breaker.record_failure()
    clock.advance(10)
    assert breaker.allow() is True
    clock.advance(7)
    breaker.record_failure()
    assert breaker.state is CircuitState.OPEN
    assert breaker.snapshot().opened_at_ms == clock.now_ms
    assert breaker.allow() is False


def test_breaker_limits_half_open_probes() -> None:
    """半开状态下放行的探测请求数受 ``half_open_max_calls`` 限制。"""
    clock = _FakeClock()
    breaker = CircuitBreaker(
        "memory",
        policy=CircuitBreakerPolicy(
            failure_threshold=1, recovery_timeout_ms=0, half_open_max_calls=1
        ),
        clock=clock,
    )
    breaker.record_failure()
    assert breaker.allow() is True
    assert breaker.allow() is False
    assert breaker.state is CircuitState.HALF_OPEN


def test_breaker_success_while_open_keeps_circuit_open() -> None:
    """打开状态下登记成功不会自行闭合（须经半开探测路径）。"""
    breaker = CircuitBreaker("memory", policy=CircuitBreakerPolicy(failure_threshold=1))
    breaker.record_failure()
    breaker.record_success()
    assert breaker.state is CircuitState.OPEN
    assert breaker.snapshot().success_total == 1


def test_breaker_reset_clears_state_and_counters() -> None:
    """``reset`` 恢复到初始闭合状态并清空全部计数。"""
    breaker = CircuitBreaker("memory", policy=CircuitBreakerPolicy(failure_threshold=1))
    breaker.record_failure()
    assert breaker.allow() is False
    breaker.reset()
    snapshot = breaker.snapshot()
    assert snapshot.state is CircuitState.CLOSED
    assert snapshot.failure_count == 0
    assert snapshot.rejected_count == 0
    assert snapshot.failure_total == 0
    assert snapshot.success_total == 0
    assert snapshot.opened_at_ms is None
    assert breaker.allow() is True


def test_breaker_snapshot_reports_open_flag() -> None:
    """快照 ``open`` 属性反映熔断是否处于打开状态。"""
    breaker = CircuitBreaker("memory", policy=CircuitBreakerPolicy(failure_threshold=1))
    assert breaker.snapshot().open is False
    breaker.record_failure()
    snapshot = breaker.snapshot()
    assert snapshot.open is True
    assert snapshot.name == "memory"
    assert snapshot.failure_count == 1


# --------------------------------------------------------------------- #
# FallbackAttempt / FallbackReport
# --------------------------------------------------------------------- #


def test_attempt_defaults() -> None:
    """尝试记录默认字段（成功、非跳过、无错误）。"""
    attempt = FallbackAttempt(backend="memory", ok=True)
    assert attempt.skipped is False
    assert attempt.error is None


def test_report_primary_served_is_not_degraded() -> None:
    """由主后端服务时不算降级。"""
    report = FallbackReport(
        operation="read",
        primary_backend="memory",
        served_backend="memory",
        attempts=(FallbackAttempt(backend="memory", ok=True),),
    )
    assert report.served is True
    assert report.degraded is False
    assert report.failed_backends == ()
    assert report.skipped_backends == ()


def test_report_degraded_to_fallback() -> None:
    """由备后端服务时标记降级。"""
    report = FallbackReport(
        operation="write",
        primary_backend="memory",
        served_backend="memory_fb",
        attempts=(
            FallbackAttempt(backend="memory", ok=False, error="StorageError: boom"),
            FallbackAttempt(backend="memory_fb", ok=True),
        ),
    )
    assert report.served is True
    assert report.degraded is True
    assert report.failed_backends == ("memory",)
    assert report.skipped_backends == ()


def test_report_splits_failed_and_skipped_backends() -> None:
    """报告可分别列出实际失败与因熔断跳过的后端。"""
    report = FallbackReport(
        operation="read",
        primary_backend="memory",
        served_backend="es",
        attempts=(
            FallbackAttempt(backend="memory", ok=False, skipped=True, error="circuit open: open"),
            FallbackAttempt(backend="redis", ok=False, error="StorageError: down"),
            FallbackAttempt(backend="es", ok=True),
        ),
    )
    assert report.failed_backends == ("redis",)
    assert report.skipped_backends == ("memory",)
    assert report.degraded is True


def test_report_without_served_backend() -> None:
    """空操作或全部失败时 ``served`` 为假且不算降级。"""
    report = FallbackReport(operation="write", primary_backend="memory")
    assert report.served is False
    assert report.degraded is False
    assert report.attempts == ()


# --------------------------------------------------------------------- #
# FallbackPolicy
# --------------------------------------------------------------------- #


def test_policy_backends_chain() -> None:
    """``backends`` 返回主后端在前的完整链。"""
    policy = FallbackPolicy(primary="redis", fallbacks=("memory", "file"))
    assert policy.backends == ("redis", "memory", "file")
    assert policy.breaker.failure_threshold == DEFAULT_FAILURE_THRESHOLD


def test_policy_strips_whitespace() -> None:
    """主/备后端名两端空白被去除。"""
    policy = FallbackPolicy(primary="  redis  ", fallbacks=(" memory ",))
    assert policy.backends == ("redis", "memory")


def test_policy_defaults_to_primary_only() -> None:
    """未给出备后端时链路只有主后端。"""
    policy = FallbackPolicy(primary="memory")
    assert policy.backends == ("memory",)
    assert policy.fallbacks == ()


def test_policy_rejects_string_fallbacks() -> None:
    """备后端列表不得误传字符串（会被逐字符拆分）。"""
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="memory", fallbacks="redis")  # type: ignore[arg-type]


def test_policy_accepts_none_fallbacks() -> None:
    """备后端为 ``None`` 时归一化为空链（等价于只配置主后端）。"""
    policy = FallbackPolicy(primary="memory", fallbacks=None)  # type: ignore[arg-type]
    assert policy.backends == ("memory",)


def test_policy_rejects_empty_primary() -> None:
    """主后端名不得为空串。"""
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="")
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="   ")


def test_policy_rejects_empty_fallback_name() -> None:
    """备后端名不得为空串。"""
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="memory", fallbacks=("redis", "  "))


def test_policy_rejects_duplicate_backend() -> None:
    """后端链中不得出现重复后端。"""
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="memory", fallbacks=("memory",))
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="memory", fallbacks=("redis", "redis"))


def test_policy_is_frozen_and_forbids_extra() -> None:
    """降级策略不可变且拒绝未知字段。"""
    policy = FallbackPolicy(primary="memory")
    with pytest.raises(PydanticValidationError):
        policy.primary = "redis"  # type: ignore[misc]
    with pytest.raises(PydanticValidationError):
        FallbackPolicy(primary="memory", unknown=1)  # type: ignore[call-arg]


# --------------------------------------------------------------------- #
# FallbackHistory：读
# --------------------------------------------------------------------- #


def test_history_read_served_by_primary(session_id: str) -> None:
    """主后端健康时读取由主后端服务，不算降级。"""
    primary, fallback = _handle(session_id), _handle(session_id)
    primary._append([_message(session_id, "hi")])
    history = _history(session_id, primary=primary, fallback=fallback)

    messages, report = history.read_med_messages()
    assert [m.content for m in messages] == ["hi"]
    assert report.served_backend == "memory"
    assert report.degraded is False
    assert report.operation == "read"
    assert fallback.read_calls == 0
    assert history.last_report is report


def test_history_read_degrades_to_fallback(session_id: str) -> None:
    """主后端读取失败时降级到备后端，报告保留失败明细。"""
    primary = _handle(session_id, fail_reads=1)
    fallback = _handle(session_id)
    fallback._append([_message(session_id, "archived")])
    history = _history(session_id, primary=primary, fallback=fallback)

    messages, report = history.read_med_messages()
    assert [m.content for m in messages] == ["archived"]
    assert report.degraded is True
    assert report.served_backend == "memory_fb"
    assert report.failed_backends == ("memory",)
    assert report.attempts[0].error is not None
    assert "StorageError" in report.attempts[0].error


def test_history_read_raises_when_all_backends_fail(session_id: str) -> None:
    """主备全部失败时抛 ``FallbackExhaustedError`` 并携带完整报告。"""
    primary = _handle(session_id, fail_reads=1)
    fallback = _handle(session_id, fail_reads=1)
    history = _history(session_id, primary=primary, fallback=fallback)

    with pytest.raises(FallbackExhaustedError) as excinfo:
        history.read_med_messages()
    report = excinfo.value.report
    assert isinstance(report, FallbackReport)
    assert report.served is False
    assert report.failed_backends == ("memory", "memory_fb")
    assert "all backends failed for read" in str(excinfo.value)


def test_history_read_skips_open_backend(session_id: str) -> None:
    """熔断已打开的后端被直接跳过，不再发起调用。"""
    primary = _handle(session_id)
    fallback = _handle(session_id)
    fallback._append([_message(session_id, "x")])
    breaker = CircuitBreaker("memory", policy=CircuitBreakerPolicy(failure_threshold=1))
    breaker.record_failure()
    history = _history(session_id, primary=primary, fallback=fallback, breakers={"memory": breaker})

    _, report = history.read_med_messages()
    assert report.skipped_backends == ("memory",)
    assert report.served_backend == "memory_fb"
    assert primary.read_calls == 0
    assert report.attempts[0].error == "circuit open: open"


def test_history_read_limit_returns_recent_messages(session_id: str) -> None:
    """``limit`` 只返回最近 N 条。"""
    primary = _handle(session_id)
    primary._append([_message(session_id, f"m{i}") for i in range(3)])
    history = _history(session_id, primary=primary, fallback=_handle(session_id))

    messages, _ = history.read_med_messages(limit=2)
    assert [m.content for m in messages] == ["m1", "m2"]


@pytest.mark.parametrize("limit", [0, -1])
def test_history_read_rejects_non_positive_limit(session_id: str, limit: int) -> None:
    """``limit`` 非正数时拒绝读取，且不触发任何后端调用。"""
    primary = _handle(session_id)
    history = _history(session_id, primary=primary, fallback=_handle(session_id))
    with pytest.raises(ValidationError):
        history.read_med_messages(limit=limit)
    assert primary.read_calls == 0


# --------------------------------------------------------------------- #
# FallbackHistory：写
# --------------------------------------------------------------------- #


def test_history_write_lands_in_primary(session_id: str) -> None:
    """主后端健康时写入落在主后端。"""
    primary, fallback = _handle(session_id), _handle(session_id)
    history = _history(session_id, primary=primary, fallback=fallback)

    report = history.add_med_messages([_message(session_id, "hello")])
    assert report.operation == "write"
    assert report.served_backend == "memory"
    assert [m.content for m in primary.stored] == ["hello"]
    assert fallback.stored == []


def test_history_write_degrades_to_fallback_store(session_id: str) -> None:
    """主后端写入失败时数据落到备存储，并标记降级。"""
    primary = _handle(session_id, fail_writes=1)
    fallback = _handle(session_id)
    history = _history(session_id, primary=primary, fallback=fallback)

    report = history.add_med_messages([_message(session_id, "degraded")])
    assert report.degraded is True
    assert report.served_backend == "memory_fb"
    assert primary.stored == []
    assert [m.content for m in fallback.stored] == ["degraded"]


def test_history_write_empty_sequence_is_noop(session_id: str) -> None:
    """空消息序列不触发任何后端调用，返回未服务的空报告。"""
    primary = _handle(session_id)
    history = _history(session_id, primary=primary, fallback=_handle(session_id))

    report = history.add_med_messages([])
    assert report.served is False
    assert report.attempts == ()
    assert primary.write_calls == 0
    assert history.last_report is report


def test_history_write_raises_when_all_backends_fail(session_id: str) -> None:
    """主备全部写入失败时抛异常。"""
    history = _history(
        session_id,
        primary=_handle(session_id, fail_writes=1),
        fallback=_handle(session_id, fail_writes=1),
    )
    with pytest.raises(FallbackExhaustedError):
        history.add_med_messages([_message(session_id, "x")])


def test_history_breaker_accumulates_across_calls(session_id: str) -> None:
    """熔断状态跨多次调用累积：连续失败达阈值后主后端被跳过。"""
    primary = _handle(session_id, fail_reads=10)
    fallback = _handle(session_id)
    history = _history(
        session_id,
        primary=primary,
        fallback=fallback,
        policy=FallbackPolicy(
            primary="memory",
            fallbacks=("memory_fb",),
            breaker=CircuitBreakerPolicy(failure_threshold=3),
        ),
    )

    for _ in range(3):
        _, report = history.read_med_messages()
        assert report.degraded is True
    assert history.breaker_for("memory").state is CircuitState.OPEN

    _, report = history.read_med_messages()
    assert report.skipped_backends == ("memory",)
    assert primary.read_calls == 3


def test_history_recovers_to_primary_after_half_open_success(session_id: str) -> None:
    """恢复窗口到期后经半开探测成功，主后端重新接管。"""
    clock = _FakeClock()
    primary = _handle(session_id, fail_reads=1)
    fallback = _handle(session_id)
    policy = CircuitBreakerPolicy(failure_threshold=1, recovery_timeout_ms=100)
    history = _history(
        session_id,
        primary=primary,
        fallback=fallback,
        policy=FallbackPolicy(primary="memory", fallbacks=("memory_fb",), breaker=policy),
        breakers={
            "memory": CircuitBreaker("memory", policy=policy, clock=clock),
            "memory_fb": CircuitBreaker("memory_fb", policy=policy, clock=clock),
        },
    )

    _, degraded = history.read_med_messages()
    assert degraded.degraded is True
    assert history.breaker_for("memory").state is CircuitState.OPEN

    _, skipped = history.read_med_messages()
    assert skipped.skipped_backends == ("memory",)

    clock.advance(100)
    _, recovered = history.read_med_messages()
    assert recovered.served_backend == "memory"
    assert recovered.degraded is False
    assert history.breaker_for("memory").state is CircuitState.CLOSED


# --------------------------------------------------------------------- #
# FallbackHistory：LangChain 契约与元数据
# --------------------------------------------------------------------- #


def test_history_messages_returns_langchain_messages(session_id: str) -> None:
    """``messages`` 按 LangChain 契约返回消息对象列表。"""
    primary = _handle(session_id)
    primary._append([_message(session_id, "chief complaint")])
    history = _history(session_id, primary=primary, fallback=_handle(session_id))

    messages: list[BaseMessage] = history.messages
    assert len(messages) == 1
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "chief complaint"


def test_history_add_messages_roundtrip(session_id: str) -> None:
    """``add_messages`` 写入后可由 ``messages`` 读回同内容。"""
    primary = _handle(session_id)
    history = _history(session_id, primary=primary, fallback=_handle(session_id))

    history.add_messages([HumanMessage(content="fever for 3 days")])
    assert [m.content for m in history.messages] == ["fever for 3 days"]
    assert primary.storage_key == "med:chat:h-a:cardio:" + session_id


def test_history_add_messages_raises_when_all_backends_fail(session_id: str) -> None:
    """``add_messages`` 在全部后端不可用时同样抛异常。"""
    history = _history(
        session_id,
        primary=_handle(session_id, fail_writes=1),
        fallback=_handle(session_id, fail_writes=1),
    )
    with pytest.raises(FallbackExhaustedError):
        history.add_messages([HumanMessage(content="x")])


def test_history_clear_clears_all_handles(session_id: str) -> None:
    """``clear`` 成功清理主后端；报告记为清理操作。"""
    primary = _handle(session_id)
    primary._append([_message(session_id, "x")])
    history = _history(session_id, primary=primary, fallback=_handle(session_id))

    history.clear()
    assert primary.stored == []
    assert history.last_report is not None
    assert history.last_report.operation == "clear"


def test_history_clear_degrades_then_raises(session_id: str) -> None:
    """``clear`` 主后端失败时降级到备后端，全失败则抛异常。"""
    fallback = _handle(session_id)
    history = _history(session_id, primary=_handle(session_id, fail_clears=1), fallback=fallback)
    history.clear()
    assert fallback.clear_calls == 1
    assert history.last_report is not None
    assert history.last_report.degraded is True

    broken = _history(
        session_id,
        primary=_handle(session_id, fail_clears=1),
        fallback=_handle(session_id, fail_clears=1),
    )
    with pytest.raises(FallbackExhaustedError):
        broken.clear()


def test_history_namespace_properties_come_from_primary(session_id: str) -> None:
    """命名空间类属性取自策略中的主后端句柄。"""
    primary = _handle(session_id)
    history = _history(session_id, primary=primary, fallback=_handle(session_id))
    assert history.session_id == session_id
    assert history.tenant_id == TENANT
    assert history.dept_id == DEPT
    assert history.patient_id == PATIENT
    assert history.storage_key == primary.storage_key
    assert history.backends == ("memory", "memory_fb")
    assert history.policy.primary == "memory"
    assert history.last_report is None


def test_history_requires_handles_for_every_backend(session_id: str) -> None:
    """``handles`` 未覆盖策略中的全部后端时拒绝构造。"""
    policy = FallbackPolicy(primary="memory", fallbacks=("redis",))
    with pytest.raises(ValidationError) as excinfo:
        FallbackHistory(policy=policy, handles={"memory": _handle(session_id)})
    assert "redis" in str(excinfo.value)


def test_history_handle_and_breaker_lookup(session_id: str) -> None:
    """按后端名可取回句柄与熔断器；未知后端名拒绝。"""
    primary, fallback = _handle(session_id), _handle(session_id)
    history = _history(session_id, primary=primary, fallback=fallback)
    assert history.handle_for("memory") is primary
    assert history.handle_for("memory_fb") is fallback
    assert history.breaker_for("memory") is history.breaker_for("memory")
    with pytest.raises(ValidationError):
        history.handle_for("es")
    with pytest.raises(ValidationError):
        history.breaker_for("es")


def test_history_autocreates_breakers_when_not_injected(session_id: str) -> None:
    """未注入熔断器时自动按策略创建，且不同实例互不共享。"""
    first = _history(session_id, primary=_handle(session_id), fallback=_handle(session_id))
    second = _history(session_id, primary=_handle(session_id), fallback=_handle(session_id))
    assert first.breaker_for("memory") is not second.breaker_for("memory")
    assert first.breaker_for("memory").policy.failure_threshold == DEFAULT_FAILURE_THRESHOLD


# --------------------------------------------------------------------- #
# FallbackHistoryResolver
# --------------------------------------------------------------------- #


def test_resolver_requires_policy() -> None:
    """未给出降级策略时拒绝构造。"""
    with pytest.raises(ValidationError):
        FallbackHistoryResolver()


def test_resolver_builds_handles_for_all_backends(session_id: str) -> None:
    """``resolve`` 按后端链顺序创建句柄并透传命名空间与 TTL。"""
    handles = {"memory": _handle(session_id), "file": _handle(session_id)}
    factory = _FakeFactory(handles)
    resolver = FallbackHistoryResolver(
        policy=FallbackPolicy(primary="memory", fallbacks=("file",)),
        store_factory=factory,
        ttl_seconds=3600,
    )

    history = resolver.resolve(
        session_id=session_id, tenant_id=TENANT, dept_id=DEPT, patient_id=PATIENT
    )
    assert [name for name, _ in factory.calls] == ["memory", "file"]
    assert factory.calls[0][1] == {
        "session_id": session_id,
        "tenant_id": TENANT,
        "dept_id": DEPT,
        "patient_id": PATIENT,
        "ttl_seconds": 3600,
    }
    assert history.handle_for("file") is handles["file"]


def test_resolver_options_prefer_per_backend_then_default(session_id: str) -> None:
    """后端专属参数优先于默认参数。"""
    handles = {"memory": _handle(session_id), "file": _handle(session_id)}
    factory = _FakeFactory(handles)
    resolver = FallbackHistoryResolver(
        policy=FallbackPolicy(primary="memory", fallbacks=("file",)),
        store_factory=factory,
        store_options={"file": {"path": "/tmp/med.jsonl"}},
        default_options={"encoding": "utf-8"},
    )
    resolver.resolve(session_id=session_id, tenant_id=TENANT, dept_id=DEPT, patient_id=PATIENT)

    assert factory.calls[0][1]["encoding"] == "utf-8"
    assert factory.calls[1][1]["path"] == "/tmp/med.jsonl"
    assert factory.calls[1][1]["encoding"] == "utf-8"
    assert resolver.options_for("memory") == {"encoding": "utf-8"}


def test_resolver_shares_breakers_across_resolves(session_id: str) -> None:
    """同一后端名在多次 ``resolve`` 之间共享同一个熔断器（状态累积）。"""
    handles = {"memory": _handle(session_id), "file": _handle(session_id)}
    factory = _FakeFactory(handles)
    resolver = FallbackHistoryResolver(
        policy=FallbackPolicy(
            primary="memory",
            fallbacks=("file",),
            breaker=CircuitBreakerPolicy(failure_threshold=1),
        ),
        store_factory=factory,
    )

    first = resolver.resolve(
        session_id=session_id, tenant_id=TENANT, dept_id=DEPT, patient_id=PATIENT
    )
    first.breaker_for("memory").record_failure()
    second = resolver.resolve(
        session_id=session_id, tenant_id=TENANT, dept_id=DEPT, patient_id=PATIENT
    )

    assert second.breaker_for("memory").state is CircuitState.OPEN
    assert set(resolver.breakers) == {"memory", "file"}
    assert resolver.policy.primary == "memory"


# --------------------------------------------------------------------- #
# MedRunnableWithMessageHistory 集成
# --------------------------------------------------------------------- #


def _echo_runnable() -> RunnableLambda:
    """构造无需外部 LLM 的下游 Runnable：回显历史条数与用户输入。"""

    def _model(inp: dict[str, Any]) -> str:
        history: list[BaseMessage] = inp.get("history", [])
        return f"history={len(history)} input={inp['input']}"

    return RunnableLambda(_model)


def test_runnable_without_fallback_policy_unchanged() -> None:
    """未注入降级策略时行为与既有实现一致（无解析器、无降级包装）。"""
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        input_messages_key="input",
        history_messages_key="history",
    )
    assert rwh.fallback_policy is None
    assert rwh.fallback_resolver is None


def test_runnable_rejects_fallback_options_without_policy() -> None:
    """只给后端专属参数而不给降级策略时立即报错。"""
    with pytest.raises(ValueError):
        MedRunnableWithMessageHistory(
            _echo_runnable(),
            backend="memory",
            fallback_store_options={"memory": {"foo": 1}},
        )


def test_runnable_invokes_with_healthy_primary() -> None:
    """注入降级策略且主后端健康时，读写均由主后端服务。"""
    handles = {"memory": _handle("s-d29-ok"), "memory_fb": _handle("s-d29-ok")}
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        store_factory=_FakeFactory(handles),  # type: ignore[arg-type]
        input_messages_key="input",
        history_messages_key="history",
        fallback_policy=FallbackPolicy(primary="memory", fallbacks=("memory_fb",)),
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-d29-ok", TENANT, DEPT, PATIENT)

    assert rwh.fallback_policy is not None
    assert rwh.fallback_resolver is not None
    assert rwh.invoke({"input": "hello"}, config=cfg) == "history=0 input=hello"
    assert len(handles["memory"].stored) == 2
    assert handles["memory_fb"].stored == []
    assert rwh.invoke({"input": "again"}, config=cfg) == "history=2 input=again"


def test_runnable_degrades_write_to_fallback_store() -> None:
    """主后端写入故障时问诊调用仍然成功，数据落到备存储。"""
    handles = {
        "memory": _handle("s-d29-degrade", fail_writes=1),
        "memory_fb": _handle("s-d29-degrade"),
    }
    rwh = MedRunnableWithMessageHistory(
        _echo_runnable(),
        backend="memory",
        store_factory=_FakeFactory(handles),  # type: ignore[arg-type]
        input_messages_key="input",
        history_messages_key="history",
        fallback_policy=FallbackPolicy(
            primary="memory",
            fallbacks=("memory_fb",),
            breaker=CircuitBreakerPolicy(failure_threshold=1),
        ),
    )
    cfg = MedRunnableWithMessageHistory.build_config("s-d29-degrade", TENANT, DEPT, PATIENT)

    assert rwh.invoke({"input": "fever"}, config=cfg) == "history=0 input=fever"
    assert handles["memory"].stored == []
    assert [m.content for m in handles["memory_fb"].stored] == ["fever", "history=0 input=fever"]

    # 熔断已打开：第二轮直接跳过主后端，仍能正常服务。
    assert rwh.invoke({"input": "cough"}, config=cfg) == "history=2 input=cough"
    assert len(handles["memory_fb"].stored) == 4
    resolver = rwh.fallback_resolver
    assert resolver is not None
    assert resolver.breaker_for("memory").state is CircuitState.OPEN


def test_history_read_limit_zero_boundary_documented(session_id: str) -> None:
    """边界：``limit=None`` 与 ``limit=1`` 均可正常读取。"""
    primary = _handle(session_id)
    primary._append([_message(session_id, "a"), _message(session_id, "b")])
    history = _history(session_id, primary=primary, fallback=_handle(session_id))

    all_messages, _ = history.read_med_messages(limit=None)
    latest, _ = history.read_med_messages(limit=1)
    assert len(all_messages) == 2
    assert [m.content for m in latest] == ["b"]
