"""并发会话锁：Redis SETNX 分布式锁 + 本地线程锁降级。

同一患者会话被并发问诊（多端接入、请求重试、批处理补写）时，「读历史 → 调模型 →
写历史」这一段临界区必须串行，否则会出现上下文错乱与消息重复追加。本模块提供一把
**会话级**锁，命名空间与存储键对齐（``med:lock:{tenant_id}:{dept_id}:{session_id}``）：

* :class:`LockPolicy` —— 租约时长、获取等待、重试间隔与看门狗周期（不可变策略）；
* :class:`SessionLock` —— 锁抽象：``acquire`` / ``release`` / ``extend`` + 上下文管理器；
* :class:`LocalSessionLock` —— 进程内 ``threading.Lock`` 降级实现（按锁键共享互斥量，
  无租约、零外部依赖），未配置 Redis 客户端时自动使用；
* :class:`RedisSessionLock` —— ``SET key token NX PX ttl`` 抢占 + 唯一令牌校验释放：
  续期与释放都走 ``WATCH``/``MULTI`` 事务做 compare-and-pexpire / compare-and-delete，
  **不依赖 Lua 脚本**（fakeredis 等测试替身同样可用）；持有期间由守护线程按看门狗
  周期续期，续期失败即标记租约丢失；
* :class:`SessionLockManager` —— 按命名空间创建锁、按是否注入客户端选型（有客户端走
  Redis，无客户端走本地），对外只暴露 ``acquire`` 与 ``hold`` 两个入口。

设计取舍：只做「互斥 + 租约」，不做排队公平性、不做可重入计数、不做跨进程文件锁；
续期与释放均以持有者令牌为准，避免误删他人锁。本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from med_langchain_memory.exceptions import (
    LockAcquisitionError,
    LockError,
    ValidationError,
)

from .tenant import build_namespace_key

#: 会话锁键前缀（与存储键前缀 ``med:chat`` 区分）。
LOCK_KEY_PREFIX = "med:lock"

#: 默认租约时长（毫秒）：超过该时长未续期即视为持有者失联，锁自动释放。
DEFAULT_LOCK_TTL_MS = 30_000

#: 默认获取锁的最长等待时长（毫秒）。
DEFAULT_ACQUIRE_TIMEOUT_MS = 5_000

#: 默认重试间隔（毫秒）。
DEFAULT_RETRY_INTERVAL_MS = 50

#: 未显式配置看门狗周期时的除数：周期 = 租约时长 / 该值。
WATCHDOG_TTL_DIVISOR = 3

#: 进程内本地锁注册表：锁键 → 互斥量（保证同名锁的不同实例互斥）。
_LOCAL_LOCKS: dict[str, Any] = {}
_LOCAL_LOCKS_GUARD = threading.Lock()


def _local_lock_for(key: str) -> Any:
    """取得（必要时创建）进程内与 ``key`` 一一对应的互斥量。

    Args:
        key: 会话锁键。

    Returns:
        ``threading.Lock`` 实例；同一进程内相同键始终返回同一个对象，
        因此本地降级锁也能真正互斥（注册表按会话数增长，不做回收）。
    """
    with _LOCAL_LOCKS_GUARD:
        lock = _LOCAL_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCAL_LOCKS[key] = lock
        return lock


def _wrap_lock_error(action: str, key: str, exc: Exception) -> LockError:
    """把锁客户端 / 事务异常包装为统一的 :class:`LockError`。

    Args:
        action: 出错的锁操作名（``acquire`` / ``extend`` / ``release``）。
        key: 会话锁键。
        exc: 原始异常。

    Returns:
        带上下文的 :class:`LockError`，调用方 ``raise ... from exc`` 抛出。
    """
    return LockError(f"redis lock {action} failed for {key}: {exc}")


class LockPolicy(BaseModel):
    """并发会话锁策略（不可变）。

    Attributes:
        ttl_ms: 租约时长（毫秒）；持有者需在到期前续期，否则锁自动释放。
        acquire_timeout_ms: 获取锁的最长等待时长（毫秒），``0`` 表示只试一次。
        retry_interval_ms: 抢占失败后的重试间隔（毫秒）。
        watchdog_enabled: 是否启用看门狗自动续期（仅分布式锁生效）。
        watchdog_interval_ms: 看门狗续期周期（毫秒）；``None`` 表示取 ``ttl_ms`` 的三分之一。

    Raises:
        pydantic.ValidationError: 时长非正数、等待时长为负数，或看门狗周期
            不小于租约时长（续期将毫无意义）时。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    ttl_ms: int = Field(default=DEFAULT_LOCK_TTL_MS, ge=1)
    acquire_timeout_ms: int = Field(default=DEFAULT_ACQUIRE_TIMEOUT_MS, ge=0)
    retry_interval_ms: int = Field(default=DEFAULT_RETRY_INTERVAL_MS, ge=1)
    watchdog_enabled: bool = True
    watchdog_interval_ms: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _validate_watchdog(self) -> LockPolicy:
        """校验看门狗周期严格小于租约时长，避免续期永远追不上过期。"""
        if self.watchdog_interval_ms is not None and self.watchdog_interval_ms >= self.ttl_ms:
            raise ValueError("watchdog_interval_ms must be smaller than ttl_ms")
        return self

    @property
    def watchdog_period_ms(self) -> int:
        """实际生效的看门狗续期周期（毫秒）。"""
        if self.watchdog_interval_ms is not None:
            return self.watchdog_interval_ms
        return max(self.ttl_ms // WATCHDOG_TTL_DIVISOR, 1)


class SessionLock(ABC):
    """会话锁抽象。

    实现需保证：``acquire`` 返回 ``True`` 即表示本实例成为持有者；``release`` 与
    ``extend`` 只能作用于本实例持有的锁（令牌校验），避免误删他人锁。
    """

    @property
    @abstractmethod
    def key(self) -> str:
        """本锁对应的会话锁键。"""
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @property
    @abstractmethod
    def token(self) -> str:
        """本次持有的唯一令牌（用于校验释放 / 续期归属）。"""
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @property
    @abstractmethod
    def locked(self) -> bool:
        """本实例当前是否持有锁。"""
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @property
    def lost(self) -> bool:
        """持有期间租约是否已丢失（过期被他人抢占、续期失败等）。

        默认实现返回 ``False``；无租约概念的本地锁始终为 ``False``。
        """
        return False

    @abstractmethod
    def acquire(self, timeout_ms: int | None = None) -> bool:
        """尝试获取锁。

        Args:
            timeout_ms: 最长等待时长（毫秒）；``None`` 表示取策略默认值。

        Returns:
            获取成功返回 ``True``；等待超时返回 ``False``。

        Raises:
            ValidationError: ``timeout_ms`` 为负数时。
            LockError: 锁基础设施（Redis 客户端）调用失败时。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def release(self) -> bool:
        """释放本实例持有的锁。

        Returns:
            成功释放返回 ``True``；本实例未持有或令牌已不匹配（租约已被他人接管）
            返回 ``False``。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def extend(self) -> bool:
        """续期本实例持有的锁。

        Returns:
            续期成功返回 ``True``；未持有、令牌不匹配或客户端异常返回 ``False``。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    def __enter__(self) -> SessionLock:
        """进入临界区：获取锁，失败抛 :class:`LockAcquisitionError`。

        Raises:
            LockAcquisitionError: 在策略等待时间内未获取到锁时。
        """
        if not self.acquire():
            raise LockAcquisitionError(f"session lock busy: {self.key}")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """退出临界区：释放锁（无论临界区内是否抛异常）。"""
        self.release()


class LocalSessionLock(SessionLock):
    """进程内会话锁（无 Redis 时的降级实现）。

    底层互斥量按锁键在进程内共享，因此不同实例对同一会话同样互斥；没有租约概念，
    故 :meth:`extend` 仅在持有期间返回 ``True``，看门狗不生效（无需续期）。

    Args:
        key: 会话锁键。
        policy: 锁策略，``None`` 表示使用默认策略。
        token: 持有者令牌，``None`` 时自动生成。
    """

    def __init__(
        self,
        key: str,
        *,
        policy: LockPolicy | None = None,
        token: str | None = None,
    ) -> None:
        if not key:
            raise ValidationError("lock key must not be empty")
        self._policy = policy or LockPolicy()
        self._key = key
        self._token = token or uuid.uuid4().hex
        self._mutex = _local_lock_for(key)
        self._held = False

    @property
    def key(self) -> str:
        """本锁对应的会话锁键。"""
        return self._key

    @property
    def token(self) -> str:
        """本次持有的唯一令牌。"""
        return self._token

    @property
    def locked(self) -> bool:
        """本实例当前是否持有锁。"""
        return self._held

    @property
    def policy(self) -> LockPolicy:
        """本锁使用的策略。"""
        return self._policy

    def acquire(self, timeout_ms: int | None = None) -> bool:
        """尝试获取进程内互斥量。

        Args:
            timeout_ms: 最长等待时长（毫秒）；``None`` 表示取策略默认值。

        Returns:
            获取成功返回 ``True``；超时返回 ``False``。本实例已持有时直接返回
            ``True``（同实例重复获取幂等，不叠加计数）。

        Raises:
            ValidationError: ``timeout_ms`` 为负数时。
        """
        if self._held:
            return True
        wait_ms = self._policy.acquire_timeout_ms if timeout_ms is None else timeout_ms
        if wait_ms < 0:
            raise ValidationError("timeout_ms must not be negative")
        acquired = self._mutex.acquire(timeout=wait_ms / 1000)
        self._held = bool(acquired)
        return self._held

    def release(self) -> bool:
        """释放进程内互斥量。

        Returns:
            释放成功返回 ``True``；本实例未持有时返回 ``False``。
        """
        if not self._held:
            return False
        self._held = False
        self._mutex.release()
        return True

    def extend(self) -> bool:
        """续期（本地锁无租约，等价于查询持有状态）。

        Returns:
            持有期间返回 ``True``，否则 ``False``。
        """
        return self._held


class RedisSessionLock(SessionLock):
    """Redis 分布式会话锁。

    抢占用 ``SET key token NX PX ttl``（单命令原子）；释放与续期用
    ``WATCH`` + ``MULTI`` 事务做令牌校验，防止租约过期后误删他人锁。
    持有期间由守护线程按看门狗周期自动续期。

    Args:
        key: 会话锁键。
        client: Redis 客户端（``redis.Redis`` 或兼容的测试替身）。
        policy: 锁策略，``None`` 表示使用默认策略。
        token: 持有者令牌，``None`` 时自动生成。
    """

    def __init__(
        self,
        key: str,
        *,
        client: Any,
        policy: LockPolicy | None = None,
        token: str | None = None,
    ) -> None:
        if not key:
            raise ValidationError("lock key must not be empty")
        if client is None:
            raise ValidationError("redis lock requires a client")
        self._policy = policy or LockPolicy()
        self._key = key
        self._client = client
        self._token = token or uuid.uuid4().hex
        self._held = False
        self._lost = False
        self._stop = threading.Event()
        self._watchdog: threading.Thread | None = None

    @property
    def key(self) -> str:
        """本锁对应的会话锁键。"""
        return self._key

    @property
    def token(self) -> str:
        """本次持有的唯一令牌。"""
        return self._token

    @property
    def locked(self) -> bool:
        """本实例当前是否持有锁。"""
        return self._held

    @property
    def lost(self) -> bool:
        """持有期间租约是否已丢失。"""
        return self._lost

    @property
    def policy(self) -> LockPolicy:
        """本锁使用的策略。"""
        return self._policy

    def acquire(self, timeout_ms: int | None = None) -> bool:
        """以 ``SET NX PX`` 抢占锁，必要时按重试间隔轮询至超时。

        Args:
            timeout_ms: 最长等待时长（毫秒）；``None`` 表示取策略默认值。

        Returns:
            抢占成功返回 ``True`` 并启动看门狗；超时返回 ``False``。

        Raises:
            ValidationError: ``timeout_ms`` 为负数时。
            LockError: Redis 客户端调用失败时。
        """
        if self._held:
            return True
        wait_ms = self._policy.acquire_timeout_ms if timeout_ms is None else timeout_ms
        if wait_ms < 0:
            raise ValidationError("timeout_ms must not be negative")
        deadline = time.monotonic() + wait_ms / 1000
        while True:
            if self._try_set():
                self._held = True
                self._lost = False
                self._start_watchdog()
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(self._policy.retry_interval_ms / 1000)

    def release(self) -> bool:
        """停止看门狗并按令牌校验删除锁键。

        Returns:
            成功删除返回 ``True``；本实例未持有、令牌不匹配或客户端异常返回 ``False``
            （后两者同时标记 :attr:`lost`）。
        """
        self._stop_watchdog()
        if not self._held:
            return False
        try:
            removed = bool(self._compare_and_delete())
        except LockError:
            self._lose()
            return False
        if not removed:
            self._lose()
            return False
        self._held = False
        return True

    def extend(self) -> bool:
        """按令牌校验续期租约。

        Returns:
            续期成功返回 ``True``；未持有、令牌不匹配或客户端异常返回 ``False``
            （后两者同时标记 :attr:`lost` 并放弃持有状态）。
        """
        if not self._held:
            return False
        try:
            renewed = bool(self._compare_and_expire())
        except LockError:
            self._lose()
            return False
        if not renewed:
            self._lose()
            return False
        return True

    def _lose(self) -> None:
        """标记租约丢失：清空持有状态并置 ``lost``（看门狗检测到失效时调用）。"""
        self._held = False
        self._lost = True

    # ------------------------------------------------------------------ #
    # 内部实现
    # ------------------------------------------------------------------ #
    def _try_set(self) -> bool:
        """执行一次 ``SET key token NX PX ttl`` 抢占。

        Raises:
            LockError: Redis 客户端调用失败时。
        """
        try:
            result = self._client.set(self._key, self._token, nx=True, px=self._policy.ttl_ms)
        except Exception as exc:  # noqa: BLE001 - 客户端异常类型随版本变化，统一包装
            raise _wrap_lock_error("acquire", self._key, exc) from exc
        return bool(result)

    def _current_token(self, pipe: Any) -> str | None:
        """在 ``WATCH`` 事务中读取当前锁持有者令牌。"""
        raw = pipe.get(self._key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            return raw.decode()
        return str(raw)

    def _compare_and_delete(self) -> bool:
        """令牌匹配才删除锁键（``WATCH`` + ``MULTI`` 事务）。

        Raises:
            LockError: 客户端调用或事务冲突（``WatchError``）时。
        """
        try:
            with self._client.pipeline() as pipe:
                pipe.watch(self._key)
                if self._current_token(pipe) != self._token:
                    pipe.unwatch()
                    pipe.reset()
                    return False
                pipe.multi()
                pipe.delete(self._key)
                results = pipe.execute()
        except Exception as exc:  # noqa: BLE001 - 事务冲突与网络异常统一降级
            raise _wrap_lock_error("release", self._key, exc) from exc
        return bool(results and results[0])

    def _compare_and_expire(self) -> bool:
        """令牌匹配才把租约重置为 ``ttl_ms``（``WATCH`` + ``MULTI`` 事务）。

        Raises:
            LockError: 客户端调用或事务冲突（``WatchError``）时。
        """
        try:
            with self._client.pipeline() as pipe:
                pipe.watch(self._key)
                if self._current_token(pipe) != self._token:
                    pipe.unwatch()
                    pipe.reset()
                    return False
                pipe.multi()
                pipe.pexpire(self._key, self._policy.ttl_ms)
                results = pipe.execute()
        except Exception as exc:  # noqa: BLE001 - 事务冲突与网络异常统一降级
            raise _wrap_lock_error("extend", self._key, exc) from exc
        return bool(results and results[0])

    def _start_watchdog(self) -> None:
        """按策略启动看门狗线程（已启动或未启用时为空操作）。"""
        if not self._policy.watchdog_enabled or self._watchdog is not None:
            return
        self._stop.clear()
        thread = threading.Thread(
            target=self._watchdog_loop,
            args=(self._policy.watchdog_period_ms,),
            name=f"med-lock-watchdog:{self._key}",
            daemon=True,
        )
        self._watchdog = thread
        thread.start()

    def _stop_watchdog(self) -> None:
        """通知并等待看门狗线程退出（幂等）。"""
        thread = self._watchdog
        self._watchdog = None
        self._stop.set()
        if thread is not None:
            thread.join(timeout=2.0)

    def _watchdog_loop(self, period_ms: int) -> None:
        """看门狗循环：按周期续期，续期失败即停止并保留 ``lost`` 标记。"""
        while not self._stop.wait(period_ms / 1000):
            if not self.extend():
                return


class SessionLockManager:
    """会话锁管理器：按命名空间创建锁，并按是否注入客户端选型。

    Args:
        client: Redis 客户端；``None`` 表示降级为进程内本地锁。
        policy: 锁策略；``None`` 表示使用默认策略。
        key_prefix: 锁键前缀，末尾冒号会被自动去除。

    Raises:
        ValidationError: ``key_prefix`` 为空字符串时。
    """

    def __init__(
        self,
        *,
        client: Any | None = None,
        policy: LockPolicy | None = None,
        key_prefix: str = LOCK_KEY_PREFIX,
    ) -> None:
        if not key_prefix:
            raise ValidationError("key_prefix must not be empty")
        self._client = client
        self._policy = policy or LockPolicy()
        self._key_prefix = key_prefix.rstrip(":")

    @property
    def policy(self) -> LockPolicy:
        """本管理器使用的锁策略。"""
        return self._policy

    @property
    def client(self) -> Any | None:
        """注入的 Redis 客户端；``None`` 表示本地降级模式。"""
        return self._client

    @property
    def distributed(self) -> bool:
        """是否处于分布式（Redis）模式。"""
        return self._client is not None

    def build_key(self, tenant_id: str, dept_id: str, session_id: str) -> str:
        """构造会话锁键 ``{prefix}:{tenant_id}:{dept_id}:{session_id}``。

        Args:
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            session_id: 会话 ID。

        Returns:
            会话锁键。

        Raises:
            ValidationError: 任一段为空时。
        """
        return f"{self._key_prefix}:{build_namespace_key(tenant_id, dept_id, session_id)}"

    def create(self, tenant_id: str, dept_id: str, session_id: str) -> SessionLock:
        """为指定会话创建一把锁（尚未获取）。

        Args:
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            session_id: 会话 ID。

        Returns:
            分布式模式下返回 :class:`RedisSessionLock`，否则返回 :class:`LocalSessionLock`。
        """
        key = self.build_key(tenant_id, dept_id, session_id)
        if self._client is None:
            return LocalSessionLock(key, policy=self._policy)
        return RedisSessionLock(key, client=self._client, policy=self._policy)

    def acquire(
        self,
        tenant_id: str,
        dept_id: str,
        session_id: str,
        *,
        timeout_ms: int | None = None,
    ) -> SessionLock:
        """获取会话锁，失败即抛异常。

        Args:
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            session_id: 会话 ID。
            timeout_ms: 最长等待时长（毫秒）；``None`` 表示取策略默认值。

        Returns:
            已持有的锁实例。

        Raises:
            LockAcquisitionError: 等待超时仍未获取到锁时。
        """
        lock = self.create(tenant_id, dept_id, session_id)
        if not lock.acquire(timeout_ms):
            raise LockAcquisitionError(f"session lock busy: {lock.key}")
        return lock

    @contextmanager
    def hold(
        self,
        tenant_id: str,
        dept_id: str,
        session_id: str,
        *,
        timeout_ms: int | None = None,
    ) -> Iterator[SessionLock]:
        """以上下文管理器方式持有会话锁，退出时自动释放。

        Args:
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            session_id: 会话 ID。
            timeout_ms: 最长等待时长（毫秒）；``None`` 表示取策略默认值。

        Yields:
            已持有的锁实例。

        Raises:
            LockAcquisitionError: 等待超时仍未获取到锁时。
        """
        lock = self.acquire(tenant_id, dept_id, session_id, timeout_ms=timeout_ms)
        try:
            yield lock
        finally:
            lock.release()
