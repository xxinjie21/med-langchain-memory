"""并发会话锁单元测试（D28）。

覆盖：锁策略校验与看门狗周期推导、本地线程锁（获取 / 等待 / 超时 / 释放 / 续期 /
上下文管理器 / 进程内同键互斥）、Redis 分布式锁（``SET NX PX`` 抢占、令牌校验释放
与续期、看门狗自动续期、租约丢失检测、客户端异常降级）、锁管理器选型与命名空间键
构造，以及 ``MedRunnableWithMessageHistory`` 的 ``invoke`` 自动加锁与 ``session_lock``
临界区集成。

全部用例零真实中间件：Redis 用 fakeredis 替身，进程内锁直接用标准库实现。
"""

from __future__ import annotations

import itertools
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import fakeredis
import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.exceptions import (
    LockAcquisitionError,
    LockError,
    ValidationError,
)
from med_langchain_memory.runnable import (
    DEFAULT_ACQUIRE_TIMEOUT_MS,
    DEFAULT_LOCK_TTL_MS,
    DEFAULT_RETRY_INTERVAL_MS,
    LOCK_KEY_PREFIX,
    WATCHDOG_TTL_DIVISOR,
    LocalSessionLock,
    LockPolicy,
    MedRunnableWithMessageHistory,
    RedisSessionLock,
    SessionLock,
    SessionLockManager,
)
from med_langchain_memory.runnable import lock as lock_module

# --------------------------------------------------------------------- #
# 测试替身与夹具
# --------------------------------------------------------------------- #


@pytest.fixture(autouse=True)
def _reset_local_lock_registry() -> Iterator[None]:
    """每个用例前清空进程内锁注册表，避免固定锁键在用例之间互相串扰。"""
    with lock_module._LOCAL_LOCKS_GUARD:  # noqa: SLF001 - 测试需要隔离进程级注册表
        lock_module._LOCAL_LOCKS.clear()
    yield


@pytest.fixture
def unique_key() -> Any:
    """返回锁键工厂：每次调用生成互不相同的键，避免本地锁注册表跨用例串扰。"""
    counter = itertools.count()

    def _make(prefix: str = LOCK_KEY_PREFIX) -> str:
        return f"{prefix}:pytest:{next(counter)}:{uuid.uuid4().hex[:8]}"

    return _make


@pytest.fixture
def client() -> Any:
    """fakeredis 客户端替身（无真实 Redis 依赖）。"""
    return fakeredis.FakeRedis()


class BrokenSetClient:
    """``set`` 直接抛异常的客户端替身（模拟连接失败）。"""

    def set(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("connection refused")

    def pipeline(self) -> Any:
        raise RuntimeError("connection refused")


class BrokenPipelineClient:
    """``set`` 正常但事务不可用的客户端替身（模拟 pipeline / WATCH 异常）。"""

    def __init__(self) -> None:
        self.store: dict[str, Any] = {}

    def set(self, key: str, value: Any, nx: bool = False, px: int | None = None) -> Any:
        if nx and key in self.store:
            return None
        self.store[key] = value
        return True

    def pipeline(self) -> Any:
        raise RuntimeError("pipeline unavailable")


class RecordingManager:
    """记录 ``hold`` 调用的锁管理器替身（供 Runnable 集成用例断言）。"""

    def __init__(self, policy: LockPolicy | None = None) -> None:
        self.delegate = SessionLockManager(policy=policy or LockPolicy())
        self.held: list[tuple[str, str, str]] = []
        self.released = 0

    @property
    def policy(self) -> LockPolicy:
        """委托给内部真实管理器的策略。"""
        return self.delegate.policy

    def build_key(self, tenant_id: str, dept_id: str, session_id: str) -> str:
        """委托给内部真实管理器的键构造。"""
        return self.delegate.build_key(tenant_id, dept_id, session_id)

    @contextmanager
    def hold(
        self,
        tenant_id: str,
        dept_id: str,
        session_id: str,
        *,
        timeout_ms: int | None = None,
    ) -> Iterator[Any]:
        """记录一次临界区进入 / 退出，实际加锁仍走本地锁。"""
        with self.delegate.hold(tenant_id, dept_id, session_id, timeout_ms=timeout_ms) as lock:
            self.held.append((tenant_id, dept_id, session_id))
            try:
                yield lock
            finally:
                self.released += 1


def _build_runnable(**kwargs: Any) -> MedRunnableWithMessageHistory:
    """构造一个包裹恒等链的医疗 Runnable（内存后端，零外部依赖）。"""
    chain = RunnableLambda(lambda payload: {"answer": "ok"})
    return MedRunnableWithMessageHistory(chain, backend="memory", **kwargs)


def _config(
    session_id: str = "s-1",
    tenant_id: str = "h-a",
    dept_id: str = "cardio",
    patient_id: str = "p-1",
) -> dict[str, Any]:
    """构造完整四段式调用配置。"""
    return MedRunnableWithMessageHistory.build_config(session_id, tenant_id, dept_id, patient_id)


# --------------------------------------------------------------------- #
# 锁策略
# --------------------------------------------------------------------- #


class TestLockPolicy:
    """``LockPolicy`` 默认值、看门狗周期推导与校验。"""

    def test_defaults(self) -> None:
        policy = LockPolicy()
        assert policy.ttl_ms == DEFAULT_LOCK_TTL_MS
        assert policy.acquire_timeout_ms == DEFAULT_ACQUIRE_TIMEOUT_MS
        assert policy.retry_interval_ms == DEFAULT_RETRY_INTERVAL_MS
        assert policy.watchdog_enabled is True
        assert policy.watchdog_interval_ms is None

    def test_watchdog_period_derives_from_ttl(self) -> None:
        policy = LockPolicy(ttl_ms=900)
        assert policy.watchdog_period_ms == 900 // WATCHDOG_TTL_DIVISOR

    def test_watchdog_period_explicit_wins(self) -> None:
        policy = LockPolicy(ttl_ms=3000, watchdog_interval_ms=200)
        assert policy.watchdog_period_ms == 200

    def test_watchdog_period_never_zero(self) -> None:
        policy = LockPolicy(ttl_ms=2)
        assert policy.watchdog_period_ms == 1

    def test_watchdog_period_must_be_smaller_than_ttl(self) -> None:
        with pytest.raises(PydanticValidationError):
            LockPolicy(ttl_ms=100, watchdog_interval_ms=100)

    def test_ttl_must_be_positive(self) -> None:
        with pytest.raises(PydanticValidationError):
            LockPolicy(ttl_ms=0)

    def test_acquire_timeout_may_be_zero(self) -> None:
        assert LockPolicy(acquire_timeout_ms=0).acquire_timeout_ms == 0

    def test_acquire_timeout_must_not_be_negative(self) -> None:
        with pytest.raises(PydanticValidationError):
            LockPolicy(acquire_timeout_ms=-1)

    def test_retry_interval_must_be_positive(self) -> None:
        with pytest.raises(PydanticValidationError):
            LockPolicy(retry_interval_ms=0)

    def test_extra_field_forbidden(self) -> None:
        with pytest.raises(PydanticValidationError):
            LockPolicy(unknown_field=1)  # type: ignore[call-arg]

    def test_policy_is_frozen(self) -> None:
        policy = LockPolicy()
        with pytest.raises(PydanticValidationError):
            policy.ttl_ms = 1  # type: ignore[misc]


# --------------------------------------------------------------------- #
# 本地线程锁
# --------------------------------------------------------------------- #


class TestLocalSessionLock:
    """本地降级锁的获取、等待、释放、续期与上下文管理器。"""

    def test_acquire_and_release(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        assert lock.locked is False
        assert lock.acquire(0) is True
        assert lock.locked is True
        assert lock.release() is True
        assert lock.locked is False

    def test_key_and_token(self, unique_key: Any) -> None:
        key = unique_key()
        first = LocalSessionLock(key)
        second = LocalSessionLock(key)
        assert first.key == key
        assert first.token and second.token
        assert first.token != second.token

    def test_explicit_token_respected(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key(), token="fixed-token")
        assert lock.token == "fixed-token"

    def test_empty_key_rejected(self) -> None:
        with pytest.raises(ValidationError):
            LocalSessionLock("")

    def test_default_policy_used(self, unique_key: Any) -> None:
        assert LocalSessionLock(unique_key()).policy.acquire_timeout_ms == (
            DEFAULT_ACQUIRE_TIMEOUT_MS
        )

    def test_custom_policy_exposed(self, unique_key: Any) -> None:
        policy = LockPolicy(acquire_timeout_ms=7)
        assert LocalSessionLock(unique_key(), policy=policy).policy is policy

    def test_reacquire_while_held_is_idempotent(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        assert lock.acquire(0) is True
        assert lock.acquire(0) is True
        assert lock.release() is True

    def test_same_key_instances_are_mutually_exclusive(self, unique_key: Any) -> None:
        key = unique_key()
        holder = LocalSessionLock(key)
        other = LocalSessionLock(key)
        assert holder.acquire(0) is True
        assert other.acquire(0) is False
        assert holder.release() is True
        assert other.acquire(0) is True
        other.release()

    def test_acquire_negative_timeout_rejected(self, unique_key: Any) -> None:
        with pytest.raises(ValidationError):
            LocalSessionLock(unique_key()).acquire(-1)

    def test_acquire_waits_until_release(self, unique_key: Any) -> None:
        key = unique_key()
        holder = LocalSessionLock(key)
        waiter = LocalSessionLock(key, policy=LockPolicy(retry_interval_ms=5))
        assert holder.acquire(0) is True

        def release_later() -> None:
            time.sleep(0.1)
            holder.release()

        thread = threading.Thread(target=release_later)
        thread.start()
        try:
            assert waiter.acquire(2000) is True
        finally:
            thread.join(timeout=2)
            waiter.release()

    def test_acquire_returns_false_on_timeout(self, unique_key: Any) -> None:
        key = unique_key()
        holder = LocalSessionLock(key)
        assert holder.acquire(0) is True
        try:
            assert LocalSessionLock(key).acquire(20) is False
        finally:
            holder.release()

    def test_release_without_hold_returns_false(self, unique_key: Any) -> None:
        assert LocalSessionLock(unique_key()).release() is False

    def test_second_release_returns_false(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        lock.acquire(0)
        assert lock.release() is True
        assert lock.release() is False

    def test_extend_tracks_hold_state(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        assert lock.extend() is False
        lock.acquire(0)
        assert lock.extend() is True
        lock.release()
        assert lock.extend() is False

    def test_lost_is_always_false(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        lock.acquire(0)
        assert lock.lost is False
        lock.release()

    def test_context_manager_acquires_and_releases(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        with lock as held:
            assert held is lock
            assert lock.locked is True
        assert lock.locked is False

    def test_context_manager_releases_on_exception(self, unique_key: Any) -> None:
        lock = LocalSessionLock(unique_key())
        with pytest.raises(RuntimeError), lock:
            raise RuntimeError("boom")
        assert lock.locked is False

    def test_context_manager_raises_when_busy(self, unique_key: Any) -> None:
        key = unique_key()
        holder = LocalSessionLock(key)
        assert holder.acquire(0) is True
        try:
            with (
                pytest.raises(LockAcquisitionError),
                LocalSessionLock(key, policy=LockPolicy(acquire_timeout_ms=0)),
            ):
                pytest.fail("should not enter critical section")
        finally:
            holder.release()

    def test_abstract_base_not_instantiable(self) -> None:
        with pytest.raises(TypeError):
            SessionLock()  # type: ignore[abstract]


# --------------------------------------------------------------------- #
# Redis 分布式锁
# --------------------------------------------------------------------- #


class TestRedisSessionLock:
    """Redis 分布式锁的抢占、令牌校验、续期与看门狗。"""

    def test_acquire_sets_key_with_ttl(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key, client=client, policy=LockPolicy(ttl_ms=2000, watchdog_enabled=False)
        )
        assert lock.acquire(0) is True
        assert lock.key == key
        assert client.get(key) == lock.token.encode()
        assert 0 < client.pttl(key) <= 2000
        assert lock.locked is True

    def test_empty_key_rejected(self, client: Any) -> None:
        with pytest.raises(ValidationError):
            RedisSessionLock("", client=client)

    def test_client_required(self, unique_key: Any) -> None:
        with pytest.raises(ValidationError):
            RedisSessionLock(unique_key(), client=None)

    def test_token_is_unique_per_instance(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        first = RedisSessionLock(key, client=client, token="t-1")
        second = RedisSessionLock(key, client=client)
        assert first.token == "t-1"
        assert second.token != "t-1"

    def test_policy_exposed(self, client: Any, unique_key: Any) -> None:
        policy = LockPolicy(ttl_ms=1500, watchdog_enabled=False)
        assert RedisSessionLock(unique_key(), client=client, policy=policy).policy is policy

    def test_reacquire_while_held_is_idempotent(self, client: Any, unique_key: Any) -> None:
        lock = RedisSessionLock(
            unique_key(), client=client, policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        )
        assert lock.acquire(0) is True
        assert lock.acquire(0) is True
        assert lock.release() is True

    def test_second_holder_blocked_until_release(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        policy = LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        holder = RedisSessionLock(key, client=client, policy=policy)
        other = RedisSessionLock(key, client=client, policy=policy)
        assert holder.acquire(0) is True
        assert other.acquire(0) is False
        assert holder.release() is True
        assert other.acquire(0) is True
        other.release()

    def test_acquire_returns_false_on_timeout(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        policy = LockPolicy(ttl_ms=5000, retry_interval_ms=5, watchdog_enabled=False)
        holder = RedisSessionLock(key, client=client, policy=policy)
        other = RedisSessionLock(key, client=client, policy=policy)
        assert holder.acquire(0) is True
        try:
            assert other.acquire(30) is False
        finally:
            holder.release()

    def test_acquire_waits_and_succeeds_after_release(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        policy = LockPolicy(ttl_ms=5000, retry_interval_ms=5, watchdog_enabled=False)
        holder = RedisSessionLock(key, client=client, policy=policy)
        waiter = RedisSessionLock(key, client=client, policy=policy)
        assert holder.acquire(0) is True

        def release_later() -> None:
            time.sleep(0.1)
            holder.release()

        thread = threading.Thread(target=release_later)
        thread.start()
        try:
            assert waiter.acquire(2000) is True
        finally:
            thread.join(timeout=2)
            waiter.release()

    def test_acquire_negative_timeout_rejected(self, client: Any, unique_key: Any) -> None:
        lock = RedisSessionLock(unique_key(), client=client)
        with pytest.raises(ValidationError):
            lock.acquire(-5)

    def test_release_deletes_key(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key, client=client, policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        )
        lock.acquire(0)
        assert lock.release() is True
        assert client.exists(key) == 0
        assert lock.locked is False
        assert lock.release() is False

    def test_release_by_non_owner_marks_lost(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        policy = LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        owner = RedisSessionLock(key, client=client, policy=policy, token="owner")
        owner.acquire(0)
        client.set(key, "someone-else", px=5000)
        assert owner.release() is False
        assert owner.lost is True
        assert owner.locked is False

    def test_extend_refreshes_ttl(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key, client=client, policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        )
        lock.acquire(0)
        client.pexpire(key, 100)
        assert lock.extend() is True
        assert client.pttl(key) > 100
        assert lock.lost is False
        lock.release()

    def test_extend_without_hold_returns_false(self, client: Any, unique_key: Any) -> None:
        lock = RedisSessionLock(unique_key(), client=client)
        assert lock.extend() is False
        assert lock.lost is False

    def test_extend_when_key_gone_marks_lost(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key, client=client, policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        )
        lock.acquire(0)
        client.delete(key)
        assert lock.extend() is False
        assert lock.lost is True
        assert lock.locked is False

    def test_watchdog_renews_lease(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key,
            client=client,
            policy=LockPolicy(ttl_ms=400, watchdog_interval_ms=100),
        )
        assert lock.acquire(0) is True
        try:
            time.sleep(0.7)
            assert client.exists(key) == 1
            assert lock.locked is True
            assert lock.lost is False
        finally:
            lock.release()
        assert client.exists(key) == 0

    def test_watchdog_detects_lost_lease(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key,
            client=client,
            policy=LockPolicy(ttl_ms=400, watchdog_interval_ms=100),
        )
        assert lock.acquire(0) is True
        client.delete(key)
        time.sleep(0.35)
        assert lock.lost is True
        assert lock.locked is False
        assert lock.release() is False

    def test_watchdog_disabled_lets_lease_expire(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key,
            client=client,
            policy=LockPolicy(ttl_ms=200, watchdog_enabled=False),
        )
        assert lock.acquire(0) is True
        time.sleep(0.35)
        assert client.exists(key) == 0
        assert lock.lost is False
        assert lock.release() is False
        assert lock.lost is True

    def test_context_manager_releases(self, client: Any, unique_key: Any) -> None:
        key = unique_key()
        lock = RedisSessionLock(
            key, client=client, policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False)
        )
        with lock:
            assert client.exists(key) == 1
        assert client.exists(key) == 0

    def test_acquire_error_wrapped_as_lock_error(self, unique_key: Any) -> None:
        lock = RedisSessionLock(unique_key(), client=BrokenSetClient())
        with pytest.raises(LockError, match="acquire"):
            lock.acquire(0)

    def test_release_error_marks_lost(self, unique_key: Any) -> None:
        lock = RedisSessionLock(
            unique_key(),
            client=BrokenPipelineClient(),
            policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False),
        )
        assert lock.acquire(0) is True
        assert lock.release() is False
        assert lock.lost is True

    def test_extend_error_marks_lost(self, unique_key: Any) -> None:
        lock = RedisSessionLock(
            unique_key(),
            client=BrokenPipelineClient(),
            policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False),
        )
        assert lock.acquire(0) is True
        assert lock.extend() is False
        assert lock.lost is True
        assert lock.locked is False

    def test_works_with_decoded_responses_client(self, unique_key: Any) -> None:
        decoded = fakeredis.FakeRedis(decode_responses=True)
        lock = RedisSessionLock(
            unique_key(),
            client=decoded,
            policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False),
        )
        assert lock.acquire(0) is True
        assert lock.extend() is True
        assert lock.release() is True


# --------------------------------------------------------------------- #
# 锁管理器
# --------------------------------------------------------------------- #


class TestSessionLockManager:
    """锁管理器的选型、键构造与两个对外入口。"""

    def test_local_mode_by_default(self) -> None:
        manager = SessionLockManager()
        assert manager.distributed is False
        assert manager.client is None
        assert isinstance(manager.create("h-a", "cardio", "s-1"), LocalSessionLock)

    def test_distributed_mode_with_client(self, client: Any) -> None:
        manager = SessionLockManager(client=client)
        assert manager.distributed is True
        assert manager.client is client
        assert isinstance(manager.create("h-a", "cardio", "s-1"), RedisSessionLock)

    def test_default_policy(self) -> None:
        assert SessionLockManager().policy.ttl_ms == DEFAULT_LOCK_TTL_MS

    def test_custom_policy_exposed(self) -> None:
        policy = LockPolicy(ttl_ms=1234)
        assert SessionLockManager(policy=policy).policy is policy

    def test_build_key_format(self) -> None:
        manager = SessionLockManager()
        assert manager.build_key("h-a", "cardio", "s-1") == "med:lock:h-a:cardio:s-1"

    def test_build_key_rejects_empty_part(self) -> None:
        with pytest.raises(ValidationError):
            SessionLockManager().build_key("h-a", "cardio", "")

    def test_empty_prefix_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SessionLockManager(key_prefix="")

    def test_prefix_trailing_colon_stripped(self) -> None:
        manager = SessionLockManager(key_prefix="med:lock:test:")
        assert manager.build_key("h-a", "cardio", "s-1") == "med:lock:test:h-a:cardio:s-1"

    def test_acquire_returns_held_lock(self) -> None:
        manager = SessionLockManager()
        lock = manager.acquire("h-a", "cardio", "s-1")
        try:
            assert lock.locked is True
        finally:
            lock.release()

    def test_acquire_raises_when_busy(self) -> None:
        manager = SessionLockManager(policy=LockPolicy(acquire_timeout_ms=0))
        holder = manager.acquire("h-a", "cardio", "s-1")
        try:
            with pytest.raises(LockAcquisitionError, match="session lock busy"):
                manager.acquire("h-a", "cardio", "s-1")
        finally:
            holder.release()

    def test_hold_releases_after_exit(self) -> None:
        manager = SessionLockManager()
        with manager.hold("h-a", "cardio", "s-1") as lock:
            assert lock.locked is True
        assert lock.locked is False
        again = manager.create("h-a", "cardio", "s-1")
        assert again.acquire(0) is True
        again.release()

    def test_hold_releases_on_exception(self) -> None:
        manager = SessionLockManager()
        with pytest.raises(RuntimeError), manager.hold("h-a", "cardio", "s-1"):
            raise RuntimeError("boom")
        again = manager.create("h-a", "cardio", "s-1")
        assert again.acquire(0) is True
        again.release()

    def test_hold_raises_when_busy(self) -> None:
        manager = SessionLockManager(policy=LockPolicy(acquire_timeout_ms=0))
        holder = manager.acquire("h-a", "cardio", "s-1")
        try:
            with (
                pytest.raises(LockAcquisitionError),
                manager.hold("h-a", "cardio", "s-1"),
            ):
                pytest.fail("should not enter critical section")
        finally:
            holder.release()

    def test_namespace_isolation(self) -> None:
        manager = SessionLockManager(policy=LockPolicy(acquire_timeout_ms=0))
        first = manager.acquire("h-a", "cardio", "s-1")
        second = manager.acquire("h-a", "neuro", "s-1")
        third = manager.acquire("h-b", "cardio", "s-1")
        try:
            assert first.key != second.key != third.key
        finally:
            first.release()
            second.release()
            third.release()

    def test_distributed_hold_creates_and_removes_key(self, client: Any) -> None:
        manager = SessionLockManager(
            client=client,
            policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False),
        )
        key = manager.build_key("h-a", "cardio", "s-1")
        with manager.hold("h-a", "cardio", "s-1"):
            assert client.exists(key) == 1
        assert client.exists(key) == 0


# --------------------------------------------------------------------- #
# Runnable 集成
# --------------------------------------------------------------------- #


class TestMedRunnableLockIntegration:
    """``MedRunnableWithMessageHistory`` 的自动加锁与临界区入口。"""

    def test_locking_disabled_by_default(self) -> None:
        runnable = _build_runnable()
        assert runnable.lock_manager is None
        assert runnable.invoke({"q": "hi"}, _config()) == {"answer": "ok"}

    def test_manager_built_from_policy_without_client(self) -> None:
        runnable = _build_runnable(lock_policy=LockPolicy(ttl_ms=1000))
        manager = runnable.lock_manager
        assert manager is not None
        assert manager.distributed is False
        assert manager.policy.ttl_ms == 1000

    def test_manager_built_from_policy_with_client(self, client: Any) -> None:
        runnable = _build_runnable(lock_policy=LockPolicy(ttl_ms=1000), lock_client=client)
        manager = runnable.lock_manager
        assert manager is not None
        assert manager.distributed is True
        assert manager.client is client

    def test_explicit_manager_wins(self) -> None:
        manager = SessionLockManager()
        runnable = _build_runnable(
            lock_policy=LockPolicy(ttl_ms=9999),
            lock_manager=manager,
        )
        assert runnable.lock_manager is manager
        assert manager.policy.ttl_ms == DEFAULT_LOCK_TTL_MS

    def test_invoke_locks_resolved_namespace(self) -> None:
        manager = RecordingManager()
        runnable = _build_runnable(lock_manager=manager)
        assert runnable.invoke({"q": "hi"}, _config()) == {"answer": "ok"}
        assert manager.held == [("h-a", "cardio", "s-1")]
        assert manager.released == 1

    def test_invoke_falls_back_to_default_namespace(self) -> None:
        manager = RecordingManager()
        runnable = _build_runnable(
            lock_manager=manager,
            default_namespace={"tenant_id": "h-def", "dept_id": "general"},
        )
        runnable.invoke({"q": "hi"}, _config(tenant_id="", dept_id=""))
        assert manager.held == [("h-def", "general", "s-1")]

    def test_invoke_uses_tenant_context_fallback(self) -> None:
        from med_langchain_memory.runnable import TenantContext

        manager = RecordingManager()
        runnable = _build_runnable(
            lock_manager=manager,
            tenant_context=TenantContext(tenant_id="h-ctx", dept_id="ctx-dept"),
        )
        runnable.invoke({"q": "hi"}, _config(tenant_id="", dept_id=""))
        assert manager.held == [("h-ctx", "ctx-dept", "s-1")]

    def test_invoke_skips_lock_without_session_id(self) -> None:
        manager = RecordingManager()
        runnable = _build_runnable(lock_manager=manager)
        with pytest.raises(ValueError):
            runnable.invoke(
                {"q": "hi"},
                {"configurable": {"tenant_id": "h-a", "dept_id": "cardio", "patient_id": "p-1"}},
            )
        assert manager.held == []

    def test_invoke_skips_lock_without_config(self) -> None:
        manager = RecordingManager()
        runnable = _build_runnable(lock_manager=manager)
        with pytest.raises(ValueError):
            runnable.invoke({"q": "hi"})
        assert manager.held == []

    def test_invoke_releases_lock_on_downstream_error(self) -> None:
        def boom(payload: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("llm down")

        manager = SessionLockManager(policy=LockPolicy(acquire_timeout_ms=0))
        runnable = MedRunnableWithMessageHistory(
            RunnableLambda(boom), backend="memory", lock_manager=manager
        )
        with pytest.raises(RuntimeError):
            runnable.invoke({"q": "hi"}, _config())
        again = manager.create("h-a", "cardio", "s-1")
        assert again.acquire(0) is True
        again.release()

    def test_invoke_with_distributed_lock(self, client: Any) -> None:
        runnable = _build_runnable(
            lock_policy=LockPolicy(ttl_ms=5000, watchdog_enabled=False),
            lock_client=client,
        )
        assert runnable.invoke({"q": "hi"}, _config()) == {"answer": "ok"}
        assert client.keys("med:lock:*") == []

    def test_invoke_serializes_concurrent_calls(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def slow(payload: dict[str, Any]) -> dict[str, Any]:
            entered.set()
            release.wait(2)
            return {"answer": "ok"}

        manager = SessionLockManager(policy=LockPolicy(acquire_timeout_ms=0))
        runnable = MedRunnableWithMessageHistory(
            RunnableLambda(slow), backend="memory", lock_manager=manager
        )
        config = _config()
        errors: list[BaseException] = []

        def run() -> None:
            try:
                runnable.invoke({"q": "hi"}, config)
            except BaseException as exc:  # pragma: no cover - 仅用于失败诊断
                errors.append(exc)

        thread = threading.Thread(target=run)
        thread.start()
        try:
            assert entered.wait(2) is True
            assert manager.create("h-a", "cardio", "s-1").acquire(0) is False
            with pytest.raises(LockAcquisitionError):
                runnable.invoke({"q": "hi"}, config)
        finally:
            release.set()
            thread.join(timeout=2)
        assert errors == []
        again = manager.create("h-a", "cardio", "s-1")
        assert again.acquire(0) is True
        again.release()

    def test_session_lock_requires_enabled_locking(self) -> None:
        runnable = _build_runnable()
        with (
            pytest.raises(LockError, match="not enabled"),
            runnable.session_lock("s-1", tenant_id="h-a", dept_id="cardio"),
        ):
            pytest.fail("should not enter critical section")

    def test_session_lock_holds_and_releases(self) -> None:
        runnable = _build_runnable(lock_policy=LockPolicy(ttl_ms=5000))
        with runnable.session_lock("s-1", tenant_id="h-a", dept_id="cardio") as lock:
            assert lock.locked is True
            assert lock.key == "med:lock:h-a:cardio:s-1"
        assert lock.locked is False

    def test_session_lock_uses_default_namespace(self) -> None:
        runnable = _build_runnable(
            lock_policy=LockPolicy(ttl_ms=5000),
            default_namespace={"tenant_id": "h-def", "dept_id": "general"},
        )
        with runnable.session_lock("s-9") as lock:
            assert lock.key == "med:lock:h-def:general:s-9"

    def test_session_lock_raises_when_busy(self) -> None:
        manager = SessionLockManager(policy=LockPolicy(acquire_timeout_ms=0))
        runnable = _build_runnable(lock_manager=manager)
        holder = manager.acquire("h-a", "cardio", "s-1")
        try:
            with (
                pytest.raises(LockAcquisitionError),
                runnable.session_lock("s-1", tenant_id="h-a", dept_id="cardio"),
            ):
                pytest.fail("should not enter critical section")
        finally:
            holder.release()

    def test_resolve_namespace_variants(self) -> None:
        runnable = _build_runnable()
        assert runnable.resolve_namespace(_config()) == ("h-a", "cardio", "s-1")
        assert runnable.resolve_namespace(None) is None
        assert runnable.resolve_namespace("not-a-mapping") is None
        assert runnable.resolve_namespace({}) is None
        assert runnable.resolve_namespace({"configurable": "not-a-mapping"}) is None
        assert runnable.resolve_namespace({"configurable": {"session_id": ""}}) is None
        assert runnable.resolve_namespace({"configurable": {"session_id": 42}}) is None

    def test_resolve_namespace_returns_empty_tenant_and_dept(self) -> None:
        runnable = _build_runnable()
        namespace = runnable.resolve_namespace({"configurable": {"session_id": "s-1"}})
        assert namespace == ("", "", "s-1")
