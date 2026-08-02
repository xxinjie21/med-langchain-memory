"""进程内内存存储适配器 :class:`InMemoryMedHistory`。

定位：单元测试、本地开发与降级兜底场景下的零依赖后端，同时作为其余存储实现
（文件 / Redis / MySQL / ES）的行为基准——所有后端都应与本实现语义一致。

设计要点：

* **进程内共享**：数据挂在类级字典上，按 ``storage_key`` 索引。
  因此同一会话的多个实例句柄看到同一份数据，与真实远端存储语义一致；
* **线程安全**：所有读写在同一把可重入锁内完成；
* **逻辑 TTL**：无原生过期能力，用 ``expires_at`` 时间戳 + 惰性淘汰模拟，
  写入时滑动续期，读取时发现过期即整会话丢弃。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import ClassVar

from med_langchain_memory.domain.message import MedMessage, now_millis

from .base import MedChatMessageHistory
from .factory import StoreFactory


@dataclass
class _SessionEntry:
    """单个会话在内存中的存储条目。

    Attributes:
        messages: 时序升序的消息列表。
        expires_at: 逻辑过期时间（epoch 毫秒），``None`` 表示永不过期。
    """

    messages: list[MedMessage] = field(default_factory=list)
    expires_at: int | None = None


@StoreFactory.register("memory")
class InMemoryMedHistory(MedChatMessageHistory):
    """进程内内存会话历史，注册名 ``memory``。

    Example:
        >>> history = InMemoryMedHistory(
        ...     session_id="s-1", tenant_id="hosp-a", dept_id="cardio", patient_id="p-1"
        ... )
        >>> history.get_med_messages()
        []
    """

    #: 以 ``expires_at`` 时间戳 + 惰性淘汰的方式支持会话级 TTL。
    supports_ttl: ClassVar[bool] = True

    #: 进程内全局会话表：``storage_key -> _SessionEntry``。
    _store: ClassVar[dict[str, _SessionEntry]] = {}

    #: 保护 :attr:`_store` 的可重入锁。
    _lock: ClassVar[threading.RLock] = threading.RLock()

    # ------------------------------------------------------------------ #
    # 存储原语
    # ------------------------------------------------------------------ #
    def _append(self, messages: list[MedMessage]) -> None:
        """追加消息并按 ``created_at`` 稳定排序（同毫秒保持写入顺序）。"""
        with self._lock:
            self._evict_if_expired()
            entry = InMemoryMedHistory._store.setdefault(self.storage_key, _SessionEntry())
            entry.messages.extend(messages)
            entry.messages.sort(key=lambda message: message.created_at)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """读取时序升序消息；会话不存在或已过期时返回空列表。"""
        with self._lock:
            self._evict_if_expired()
            entry = InMemoryMedHistory._store.get(self.storage_key)
            if entry is None:
                return []
            return list(entry.messages) if limit is None else list(entry.messages[-limit:])

    def clear(self) -> None:
        """删除本会话的全部消息；若已设置 TTL 则重新开始计时。"""
        with self._lock:
            InMemoryMedHistory._store.pop(self.storage_key, None)
            self.refresh_ttl()

    def _apply_ttl(self, ttl_seconds: int) -> None:
        """将过期时刻写入会话条目（滑动续期：每次调用都重新计时）。"""
        with self._lock:
            entry = InMemoryMedHistory._store.setdefault(self.storage_key, _SessionEntry())
            entry.expires_at = now_millis() + ttl_seconds * 1000

    # ------------------------------------------------------------------ #
    # 内存后端专有能力
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        """当前会话已存储的消息条数（会话已过期时为 ``0``）。"""
        return len(self._read())

    @property
    def expires_at(self) -> int | None:
        """本会话的逻辑过期时刻（epoch 毫秒），未设置 TTL 时为 ``None``。"""
        with self._lock:
            entry = InMemoryMedHistory._store.get(self.storage_key)
            return None if entry is None else entry.expires_at

    @classmethod
    def session_keys(cls) -> list[str]:
        """返回进程内全部会话存储键（字典序，含仅设置了 TTL 的空会话）。"""
        with cls._lock:
            return sorted(InMemoryMedHistory._store)

    @classmethod
    def reset(cls) -> None:
        """清空进程内全部会话数据，供测试隔离与进程级重置使用。"""
        with cls._lock:
            InMemoryMedHistory._store.clear()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _evict_if_expired(self) -> None:
        """惰性淘汰：会话已越过 ``expires_at`` 时整条移除（需在锁内调用）。"""
        entry = InMemoryMedHistory._store.get(self.storage_key)
        if entry is None or entry.expires_at is None:
            return
        if now_millis() >= entry.expires_at:
            del InMemoryMedHistory._store[self.storage_key]
