"""读写降级兜底：主存储故障 → 备存储兜底 + 简易熔断器。

医院问诊高峰期，主存储（Redis 集群 / MySQL 分表）可能因网络抖动、实例故障或连接池
耗尽而短时不可用。若直接向上抛出 :class:`StorageError`，整条问诊链路都会失败；本模块
提供**会话读写降级**能力：

* :class:`CircuitBreakerPolicy` —— 熔断策略（失败阈值、恢复窗口、半开探测配额、闭合阈值）；
* :class:`CircuitState` —— 熔断状态枚举 ``closed`` / ``open`` / ``half_open``；
* :class:`CircuitBreaker` —— 每个后端一个的简易熔断器：连续失败达到阈值即打开，恢复窗口
  到期后进入半开并放行少量探测请求，探测成功达标则闭合、探测失败立即重新打开。只做
  「计数 + 状态机」，不做滑动窗口统计、不做并发限流、不做超时控制；
* :class:`FallbackPolicy` —— 主后端 + 有序备后端链 + 熔断策略；
* :class:`FallbackAttempt` / :class:`FallbackReport` —— 单次调用的尝试明细与汇总报告；
* :class:`FallbackHistory` —— 与 ``MedChatMessageHistory`` 同契约（``messages`` /
  ``add_messages`` / ``clear``）的会话句柄包装，读/写失败时按策略顺序切换后端，
  熔断已打开的后端直接跳过；
* :class:`FallbackHistoryResolver` —— 按命名空间组装主/备句柄，并按后端名共享熔断器
  （熔断状态跨调用累积，否则降级毫无意义）。

设计取舍：降级是**尽力而为**——所有后端都失败时抛出
:class:`~med_langchain_memory.exceptions.FallbackExhaustedError`，异常携带完整报告，
绝不静默吞掉失败；写降级会把数据落到备存储，调用方可通过
:attr:`FallbackReport.degraded` 感知并做后续补偿。本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from med_langchain_memory.domain.message import MedMessage, now_millis
from med_langchain_memory.exceptions import (
    FallbackExhaustedError,
    ValidationError,
)
from med_langchain_memory.stores import (
    MedChatMessageHistory,
    StoreFactory,
    from_langchain_message,
    to_langchain_message,
)

#: 默认连续失败阈值：达到即打开熔断。
DEFAULT_FAILURE_THRESHOLD = 3

#: 默认恢复窗口（毫秒）：打开后经过该时长进入半开。
DEFAULT_RECOVERY_TIMEOUT_MS = 30_000

#: 默认半开探测配额：半开状态下最多同时放行的探测请求数。
DEFAULT_HALF_OPEN_MAX_CALLS = 1

#: 默认闭合阈值：半开状态下连续成功次数达到该值即闭合熔断。
DEFAULT_SUCCESS_THRESHOLD = 1


class CircuitState(StrEnum):
    """熔断器状态枚举。"""

    CLOSED = "closed"
    """闭合：正常放行，只做失败计数。"""

    OPEN = "open"
    """打开：直接拒绝放行，等待恢复窗口到期。"""

    HALF_OPEN = "half_open"
    """半开：放行少量探测请求，据其结果决定闭合或重新打开。"""


class CircuitBreakerPolicy(BaseModel):
    """熔断策略（不可变）。

    Attributes:
        failure_threshold: 连续失败次数达到该值即打开熔断。
        recovery_timeout_ms: 打开后经过该时长进入半开；``0`` 表示下次调用即探测。
        half_open_max_calls: 半开状态下最多同时放行的探测请求数。
        success_threshold: 半开状态下连续成功次数达到该值即闭合熔断。

    Raises:
        pydantic.ValidationError: 任一阈值非正数，或恢复窗口为负数时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    failure_threshold: int = Field(default=DEFAULT_FAILURE_THRESHOLD, ge=1)
    recovery_timeout_ms: int = Field(default=DEFAULT_RECOVERY_TIMEOUT_MS, ge=0)
    half_open_max_calls: int = Field(default=DEFAULT_HALF_OPEN_MAX_CALLS, ge=1)
    success_threshold: int = Field(default=DEFAULT_SUCCESS_THRESHOLD, ge=1)


class CircuitSnapshot(BaseModel):
    """熔断器状态快照（不可变），供监控与测试断言。

    Attributes:
        name: 熔断器名（通常为存储后端名）。
        state: 当前状态。
        failure_count: 当前连续失败次数（闭合状态下生效）。
        success_count: 半开状态下已累计的成功次数。
        opened_at_ms: 最近一次打开的时刻（epoch 毫秒），未打开时为 ``None``。
        rejected_count: 累计被拒绝放行的调用次数。
        failure_total: 累计失败调用次数（自上次 :meth:`CircuitBreaker.reset` 起）。
        success_total: 累计成功调用次数（自上次 :meth:`CircuitBreaker.reset` 起）。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    state: CircuitState
    failure_count: int = 0
    success_count: int = 0
    opened_at_ms: int | None = None
    rejected_count: int = 0
    failure_total: int = 0
    success_total: int = 0

    @property
    def open(self) -> bool:
        """熔断是否处于打开（拒绝放行）状态。"""
        return self.state is CircuitState.OPEN


class CircuitBreaker:
    """单后端简易熔断器（失败计数 + 恢复窗口 + 半开探测）。

    线程安全：所有状态变更在同一把互斥锁内完成，可被多线程共享使用。

    Args:
        name: 熔断器名（通常为存储后端名），仅用于报告与调试。
        policy: 熔断策略；``None`` 表示使用默认策略。
        clock: 毫秒时钟（返回 epoch 毫秒的可调用对象）；``None`` 表示使用系统时间，
            测试可注入假时钟以获得确定性。

    Raises:
        ValidationError: ``name`` 为空字符串时。
    """

    def __init__(
        self,
        name: str,
        *,
        policy: CircuitBreakerPolicy | None = None,
        clock: Callable[[], int] | None = None,
    ) -> None:
        if not name:
            raise ValidationError("circuit breaker name must not be empty")
        self._name = name
        self._policy = policy or CircuitBreakerPolicy()
        self._clock = clock or now_millis
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._half_open_calls = 0
        self._opened_at_ms: int | None = None
        self._rejected = 0
        self._failure_total = 0
        self._success_total = 0

    @property
    def name(self) -> str:
        """熔断器名。"""
        return self._name

    @property
    def policy(self) -> CircuitBreakerPolicy:
        """本熔断器使用的策略。"""
        return self._policy

    @property
    def state(self) -> CircuitState:
        """当前熔断状态（只读，状态推进只发生在 :meth:`allow` / ``record_*``）。"""
        with self._lock:
            return self._state

    def allow(self) -> bool:
        """判断本次调用是否放行，必要时推进状态机。

        ``open`` 状态下：未到恢复窗口直接拒绝并累计 ``rejected_count``；已到恢复窗口
        则转入 ``half_open`` 并放行。``half_open`` 状态下最多放行
        ``policy.half_open_max_calls`` 个探测请求。

        Returns:
            ``True`` 表示放行；``False`` 表示熔断打开，调用方应跳过该后端。
        """
        with self._lock:
            if self._state is CircuitState.OPEN:
                if not self._recovery_elapsed():
                    self._rejected += 1
                    return False
                self._enter_half_open()
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls >= self._policy.half_open_max_calls:
                    self._rejected += 1
                    return False
                self._half_open_calls += 1
            return True

    def record_success(self) -> None:
        """登记一次成功调用。

        闭合状态下清零连续失败计数；半开状态下累计成功次数，达到
        ``policy.success_threshold`` 即闭合熔断。
        """
        with self._lock:
            self._success_total += 1
            if self._state is CircuitState.HALF_OPEN:
                self._half_open_calls = max(self._half_open_calls - 1, 0)
                self._successes += 1
                if self._successes >= self._policy.success_threshold:
                    self._close()
            elif self._state is CircuitState.CLOSED:
                self._failures = 0

    def record_failure(self) -> None:
        """登记一次失败调用。

        半开状态下探测失败**立即重新打开**（不等阈值）；闭合状态下连续失败达到
        ``policy.failure_threshold`` 即打开熔断。
        """
        with self._lock:
            self._failure_total += 1
            self._failures += 1
            self._successes = 0
            self._half_open_calls = max(self._half_open_calls - 1, 0)
            if (
                self._state is CircuitState.HALF_OPEN
                or self._failures >= self._policy.failure_threshold
            ):
                self._open()

    def reset(self) -> None:
        """恢复到初始闭合状态并清空全部计数（含累计统计）。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._successes = 0
            self._half_open_calls = 0
            self._opened_at_ms = None
            self._rejected = 0
            self._failure_total = 0
            self._success_total = 0

    def snapshot(self) -> CircuitSnapshot:
        """返回当前状态快照（不可变）。"""
        with self._lock:
            return CircuitSnapshot(
                name=self._name,
                state=self._state,
                failure_count=self._failures,
                success_count=self._successes,
                opened_at_ms=self._opened_at_ms,
                rejected_count=self._rejected,
                failure_total=self._failure_total,
                success_total=self._success_total,
            )

    # ------------------------------------------------------------------ #
    # 内部状态流转（调用方需已持锁）
    # ------------------------------------------------------------------ #
    def _open(self) -> None:
        """打开熔断并记录打开时刻。"""
        self._state = CircuitState.OPEN
        self._opened_at_ms = self._clock()
        self._successes = 0
        self._half_open_calls = 0

    def _close(self) -> None:
        """闭合熔断并清空计数。"""
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._successes = 0
        self._half_open_calls = 0
        self._opened_at_ms = None

    def _enter_half_open(self) -> None:
        """转入半开状态并重置探测配额。"""
        self._state = CircuitState.HALF_OPEN
        self._successes = 0
        self._half_open_calls = 0

    def _recovery_elapsed(self) -> bool:
        """恢复窗口是否已到期（``open`` 状态必然已记录打开时刻）。"""
        opened_at = self._opened_at_ms or 0
        return self._clock() - opened_at >= self._policy.recovery_timeout_ms


class FallbackAttempt(BaseModel):
    """单个后端的降级尝试记录（不可变）。

    Attributes:
        backend: 后端名。
        ok: 本次尝试是否成功。
        skipped: 是否因熔断打开而被跳过（此时 ``ok`` 恒为 ``False``）。
        error: 失败原因摘要（``异常类名: 异常信息``），成功时为 ``None``。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    backend: str
    ok: bool
    skipped: bool = False
    error: str | None = None


class FallbackReport(BaseModel):
    """一次读写降级调用的汇总报告（不可变）。

    Attributes:
        operation: 操作名（``read`` / ``write`` / ``clear``）。
        primary_backend: 策略中的主后端名。
        served_backend: 实际服务本次操作的后端名；全部失败或空操作时为 ``None``。
        attempts: 按尝试顺序记录的后端明细。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: str
    primary_backend: str
    served_backend: str | None = None
    attempts: tuple[FallbackAttempt, ...] = ()

    @property
    def served(self) -> bool:
        """是否有后端成功服务本次操作。"""
        return self.served_backend is not None

    @property
    def degraded(self) -> bool:
        """是否发生了降级（由备后端而非主后端服务）。"""
        return self.served_backend is not None and self.served_backend != self.primary_backend

    @property
    def failed_backends(self) -> tuple[str, ...]:
        """本次实际失败（非跳过）的后端名，按尝试顺序。"""
        return tuple(a.backend for a in self.attempts if not a.ok and not a.skipped)

    @property
    def skipped_backends(self) -> tuple[str, ...]:
        """因熔断打开而被跳过的后端名，按尝试顺序。"""
        return tuple(a.backend for a in self.attempts if a.skipped)


class FallbackPolicy(BaseModel):
    """主备存储降级策略（不可变）。

    Attributes:
        primary: 主存储后端名（须已注册到 ``StoreFactory``）。
        fallbacks: 有序备存储后端名列表，主后端失败时按序尝试；不得与主后端重复。
        breaker: 每个后端各一份的熔断策略。

    Raises:
        pydantic.ValidationError: 主后端为空、备后端名为空串或与链中已有后端重复时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    primary: str = Field(min_length=1)
    fallbacks: tuple[str, ...] = ()
    breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)

    @field_validator("primary", mode="before")
    @classmethod
    def _strip_primary(cls, value: Any) -> Any:
        """去除主后端名两端空白。"""
        return value.strip() if isinstance(value, str) else value

    @field_validator("fallbacks", mode="before")
    @classmethod
    def _strip_fallbacks(cls, value: Any) -> Any:
        """把备后端列表归一化为去空白后的元组，并拒绝误传字符串。"""
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("fallbacks must be a sequence of backend names, not a string")
        return tuple(item.strip() if isinstance(item, str) else item for item in value)

    @model_validator(mode="after")
    def _validate_chain(self) -> FallbackPolicy:
        """校验后端链：非空且无重复。"""
        seen = {self.primary}
        for name in self.fallbacks:
            if not name:
                raise ValueError("fallback backend names must not be empty")
            if name in seen:
                raise ValueError(f"duplicate backend in fallback chain: {name!r}")
            seen.add(name)
        return self

    @property
    def backends(self) -> tuple[str, ...]:
        """完整后端链 ``(primary, *fallbacks)``。"""
        return (self.primary, *self.fallbacks)


class FallbackHistory(BaseChatMessageHistory):
    """主备存储会话句柄：读写失败时按策略降级到备后端。

    与 :class:`~med_langchain_memory.stores.MedChatMessageHistory` 同契约
    （``messages`` / ``add_messages`` / ``clear``），可直接交给
    ``RunnableWithMessageHistory`` 使用；另提供返回结构化报告的
    :meth:`read_med_messages` 与 :meth:`add_med_messages`。

    Args:
        policy: 主备降级策略。
        handles: 后端名 → 会话句柄映射，须覆盖 ``policy.backends`` 中的全部后端。
        breakers: 后端名 → 熔断器映射；缺省的后端会自动新建一个（不跨实例共享状态）。

    Raises:
        ValidationError: ``handles`` 未覆盖策略中的全部后端时。
    """

    def __init__(
        self,
        *,
        policy: FallbackPolicy,
        handles: Mapping[str, MedChatMessageHistory],
        breakers: Mapping[str, CircuitBreaker] | None = None,
    ) -> None:
        missing = [name for name in policy.backends if name not in handles]
        if missing:
            raise ValidationError(f"missing store handles for backends: {missing}")
        provided = dict(breakers or {})
        self._policy = policy
        self._handles: dict[str, MedChatMessageHistory] = {
            name: handles[name] for name in policy.backends
        }
        self._breakers: dict[str, CircuitBreaker] = {}
        for name in policy.backends:
            existing = provided.get(name)
            self._breakers[name] = (
                existing if existing is not None else CircuitBreaker(name, policy=policy.breaker)
            )
        self._primary = self._handles[policy.primary]
        self._last_report: FallbackReport | None = None

    # ------------------------------------------------------------------ #
    # 属性
    # ------------------------------------------------------------------ #
    @property
    def policy(self) -> FallbackPolicy:
        """本句柄使用的降级策略。"""
        return self._policy

    @property
    def backends(self) -> tuple[str, ...]:
        """完整后端链 ``(primary, *fallbacks)``。"""
        return self._policy.backends

    @property
    def last_report(self) -> FallbackReport | None:
        """最近一次读/写/清理的降级报告；尚未执行过任何操作时为 ``None``。"""
        return self._last_report

    @property
    def session_id(self) -> str:
        """当前会话 ID（取自主后端句柄）。"""
        return self._primary.session_id

    @property
    def tenant_id(self) -> str:
        """当前租户 ID（取自主后端句柄）。"""
        return self._primary.tenant_id

    @property
    def dept_id(self) -> str:
        """当前科室 ID（取自主后端句柄）。"""
        return self._primary.dept_id

    @property
    def patient_id(self) -> str:
        """当前患者 ID（取自主后端句柄）。"""
        return self._primary.patient_id

    @property
    def storage_key(self) -> str:
        """统一存储键（取自主后端句柄）。"""
        return self._primary.storage_key

    def handle_for(self, backend: str) -> MedChatMessageHistory:
        """按后端名取回会话句柄。

        Raises:
            ValidationError: 该后端不在策略链中时。
        """
        handle = self._handles.get(backend)
        if handle is None:
            raise ValidationError(f"unknown backend in fallback chain: {backend!r}")
        return handle

    def breaker_for(self, backend: str) -> CircuitBreaker:
        """按后端名取回熔断器。

        Raises:
            ValidationError: 该后端不在策略链中时。
        """
        breaker = self._breakers.get(backend)
        if breaker is None:
            raise ValidationError(f"unknown backend in fallback chain: {backend!r}")
        return breaker

    # ------------------------------------------------------------------ #
    # 降级读写
    # ------------------------------------------------------------------ #
    def read_med_messages(
        self, limit: int | None = None
    ) -> tuple[list[MedMessage], FallbackReport]:
        """按时序读取医疗消息，主后端失败时自动降级到备后端。

        Args:
            limit: 仅返回最近 ``limit`` 条；``None`` 表示全部。

        Returns:
            ``(消息列表, 降级报告)``。

        Raises:
            ValidationError: ``limit`` 为非正数时。
            FallbackExhaustedError: 全部后端读取失败或被熔断跳过时（异常携带报告）。
        """
        if limit is not None and limit <= 0:
            raise ValidationError("limit must be a positive integer or None")
        result, report = self._dispatch("read", lambda handle: handle.get_med_messages(limit))
        return list(result), report

    def add_med_messages(self, messages: Sequence[MedMessage]) -> FallbackReport:
        """追加医疗消息，主后端写入失败时自动降级到备后端。

        Args:
            messages: 待写入消息；空序列为空操作，不触发任何后端调用。

        Returns:
            降级报告；空序列时返回 ``served_backend`` 为 ``None`` 的空报告。

        Raises:
            FallbackExhaustedError: 全部后端写入失败或被熔断跳过时（异常携带报告）。
        """
        if not messages:
            return self._remember(
                FallbackReport(operation="write", primary_backend=self._policy.primary)
            )
        payload = list(messages)
        _, report = self._dispatch("write", lambda handle: handle.add_med_messages(payload))
        return report

    # ------------------------------------------------------------------ #
    # LangChain BaseChatMessageHistory 契约
    # ------------------------------------------------------------------ #
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """按时序返回全部消息（LangChain 格式），读取失败时自动降级。

        Raises:
            FallbackExhaustedError: 全部后端读取失败时。
        """
        med_messages, _ = self.read_med_messages()
        return [to_langchain_message(message) for message in med_messages]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """批量追加 LangChain 消息（先转为医疗消息再走降级写入）。

        Args:
            messages: LangChain 消息序列。

        Raises:
            ValidationError: 任一消息无法转换为医疗消息时。
            FallbackExhaustedError: 全部后端写入失败时。
        """
        self.add_med_messages([self._coerce(message) for message in messages])

    def clear(self) -> None:
        """清除本会话在全部后端中的消息（主后端失败时降级到备后端）。

        Raises:
            FallbackExhaustedError: 全部后端清理失败时。
        """
        self._dispatch("clear", lambda handle: handle.clear())

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _dispatch(
        self,
        operation: str,
        action: Callable[[MedChatMessageHistory], Any],
    ) -> tuple[Any, FallbackReport]:
        """按后端链依次尝试执行 ``action``。

        Args:
            operation: 操作名，写入报告。
            action: 接收单个后端句柄的调用动作。

        Returns:
            ``(成功后端返回的结果, 降级报告)``。

        Raises:
            FallbackExhaustedError: 全部后端失败或被跳过时（异常携带报告）。
        """
        attempts: list[FallbackAttempt] = []
        served: str | None = None
        result: Any = None
        for name in self._policy.backends:
            breaker = self._breakers[name]
            if not breaker.allow():
                attempts.append(
                    FallbackAttempt(
                        backend=name,
                        ok=False,
                        skipped=True,
                        error=f"circuit open: {breaker.state.value}",
                    )
                )
                continue
            try:
                result = action(self._handles[name])
            except Exception as exc:  # noqa: BLE001 - 降级需覆盖任意存储层异常
                breaker.record_failure()
                attempts.append(
                    FallbackAttempt(
                        backend=name,
                        ok=False,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue
            breaker.record_success()
            served = name
            attempts.append(FallbackAttempt(backend=name, ok=True))
            break

        report = self._remember(
            FallbackReport(
                operation=operation,
                primary_backend=self._policy.primary,
                served_backend=served,
                attempts=tuple(attempts),
            )
        )
        if served is None:
            detail = "; ".join(
                f"{item.backend}={item.error}" for item in attempts if item.error is not None
            )
            raise FallbackExhaustedError(
                f"all backends failed for {operation}: {detail or 'no backend available'}",
                report=report,
            )
        return result, report

    def _remember(self, report: FallbackReport) -> FallbackReport:
        """记录并返回报告。"""
        self._last_report = report
        return report

    def _coerce(self, message: BaseMessage) -> MedMessage:
        """用本会话命名空间把 LangChain 消息转换为医疗消息。"""
        return from_langchain_message(
            message,
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            dept_id=self.dept_id,
            patient_id=self.patient_id,
        )


class FallbackHistoryResolver:
    """按命名空间组装主/备会话句柄，并按后端名共享熔断器。

    Args:
        policy: 主备降级策略；``None`` 视为非法配置。
        store_factory: 存储工厂（类或兼容 ``create`` 签名的对象）。
        ttl_seconds: 会话级 TTL（秒），``None`` 表示永不过期。
        store_options: 后端名 → 该后端专属构造参数（在 ``default_options`` 之上覆盖）。
        default_options: 未单独配置的后端所使用的构造参数。

    Raises:
        ValidationError: ``policy`` 为 ``None`` 时。
    """

    def __init__(
        self,
        *,
        policy: FallbackPolicy | None = None,
        store_factory: Any = StoreFactory,
        ttl_seconds: int | None = None,
        store_options: Mapping[str, Mapping[str, Any]] | None = None,
        default_options: Mapping[str, Any] | None = None,
    ) -> None:
        if policy is None:
            raise ValidationError("fallback policy must not be None")
        self._policy = policy
        self._factory = store_factory
        self._ttl_seconds = ttl_seconds
        self._options: dict[str, dict[str, Any]] = {
            name: dict(options) for name, options in (store_options or {}).items()
        }
        self._default_options = dict(default_options or {})
        self._breakers: dict[str, CircuitBreaker] = {}
        self._guard = threading.Lock()

    @property
    def policy(self) -> FallbackPolicy:
        """本解析器使用的降级策略。"""
        return self._policy

    @property
    def breakers(self) -> dict[str, CircuitBreaker]:
        """已创建的熔断器副本（后端名 → 熔断器）。"""
        with self._guard:
            return dict(self._breakers)

    def breaker_for(self, backend: str) -> CircuitBreaker:
        """取得（必要时创建）指定后端的熔断器。

        Args:
            backend: 后端名。

        Returns:
            该后端的熔断器；同一后端名在本解析器内始终返回同一实例，
            因此熔断状态可跨多次 :meth:`resolve` 累积。
        """
        with self._guard:
            breaker = self._breakers.get(backend)
            if breaker is None:
                breaker = CircuitBreaker(backend, policy=self._policy.breaker)
                self._breakers[backend] = breaker
            return breaker

    def options_for(self, backend: str) -> dict[str, Any]:
        """返回指定后端实际生效的构造参数。

        Args:
            backend: 后端名。

        Returns:
            以 ``default_options`` 为底、该后端专属参数覆盖后的字典；
            后端未单独配置时即为默认参数的副本。
        """
        merged = dict(self._default_options)
        merged.update(self._options.get(backend, {}))
        return merged

    def resolve(
        self,
        *,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
    ) -> FallbackHistory:
        """为指定会话组装主备句柄并返回降级会话句柄。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。

        Returns:
            覆盖策略全部后端的 :class:`FallbackHistory`。

        Raises:
            StoreNotFoundError: 链中某后端未注册时。
            StorageError: 链中某后端构造函数不接受给定参数时。
        """
        handles: dict[str, MedChatMessageHistory] = {}
        for name in self._policy.backends:
            handles[name] = self._factory.create(
                name,
                session_id=session_id,
                tenant_id=tenant_id,
                dept_id=dept_id,
                patient_id=patient_id,
                ttl_seconds=self._ttl_seconds,
                **self.options_for(name),
            )
        return FallbackHistory(
            policy=self._policy,
            handles=handles,
            breakers={name: self.breaker_for(name) for name in self._policy.backends},
        )
