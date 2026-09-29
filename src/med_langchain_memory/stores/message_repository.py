"""消息仓储：按会话维度追加与读取消息。

``MedChatMessageHistory`` 面向「LangChain 上下文装配」，句柄与租户/科室/患者强绑定，
不适合直接作为 HTTP 接口的后端。本模块提供更薄的仓储抽象：

* :class:`MessageRepository` —— 只定义 4 个原语（append / read / count / clear），
  入参为命名空间坐标 + 会话 ID，出参为领域消息；
* :class:`InMemoryMessageRepository` —— 进程内实现，作为 API 层的默认后端与测试替身，
  由调用方通过 ``app.state.message_repository`` 替换为 Redis/MySQL 实现。

排序契约：``read`` 恒按 ``(created_at, message_id)`` 升序返回，与消息游标分页的
定位键一致。本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import StorageError, ValidationError

from .session_repository import SessionScope


class MessageRepository(ABC):
    """会话消息仓储抽象。

    实现需保证：同一 :class:`SessionScope` + ``session_id`` 下的消息按时序稳定有序；
    不同命名空间之间完全隔离，互不可见。
    """

    @abstractmethod
    def append(
        self,
        scope: SessionScope,
        session_id: str,
        messages: Sequence[MedMessage],
    ) -> list[MedMessage]:
        """追加消息并返回本次写入的消息（按时序升序）。

        Args:
            scope: 命名空间坐标。
            session_id: 会话 ID。
            messages: 待写入消息，不可为空。

        Returns:
            本次追加的消息，按时序升序排列。

        Raises:
            ValidationError: ``messages`` 为空时。
            StorageError: 存在不属于该会话命名空间的消息时。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def read(self, scope: SessionScope, session_id: str) -> list[MedMessage]:
        """读取会话全部消息（按时序升序）。

        Args:
            scope: 命名空间坐标。
            session_id: 会话 ID。

        Returns:
            时序升序消息列表；会话不存在或为空时返回空列表。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def count(self, scope: SessionScope, session_id: str) -> int:
        """返回会话已存储的消息条数。

        Args:
            scope: 命名空间坐标。
            session_id: 会话 ID。

        Returns:
            消息条数；会话不存在时为 ``0``。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def clear(self, scope: SessionScope, session_id: str) -> None:
        """清空会话的全部消息（幂等）。

        Args:
            scope: 命名空间坐标。
            session_id: 会话 ID。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现


class InMemoryMessageRepository(MessageRepository):
    """基于进程内字典的消息仓储。

    以统一存储键为字典键、消息列表为值，所有操作由一把互斥锁保护，
    可安全用于多线程 ASGI worker（同一进程内）。进程重启即丢失，仅适用于开发与测试。

    Example:
        >>> from med_langchain_memory.stores.session_repository import SessionScope
        >>> scope = SessionScope(tenant_id="hosp-a", dept_id="cardio")
        >>> InMemoryMessageRepository().read(scope, "s-1")
        []
    """

    def __init__(self) -> None:
        """初始化空仓储。"""
        self._lock = threading.Lock()
        self._messages: dict[str, list[MedMessage]] = {}

    def append(
        self,
        scope: SessionScope,
        session_id: str,
        messages: Sequence[MedMessage],
    ) -> list[MedMessage]:
        """追加消息；命名空间不匹配时抛 :class:`StorageError`。"""
        if not messages:
            raise ValidationError("messages must not be empty")
        key = scope.storage_key(session_id)
        for message in messages:
            if message.storage_key != key:
                raise StorageError(
                    f"message {message.message_id} belongs to {message.storage_key}, not {key}"
                )
        batch = sorted(messages, key=_sort_key)
        with self._lock:
            self._messages.setdefault(key, []).extend(batch)
        return batch

    def read(self, scope: SessionScope, session_id: str) -> list[MedMessage]:
        """读取会话全部消息（时序升序）；不存在时返回空列表。"""
        with self._lock:
            stored = self._messages.get(scope.storage_key(session_id), [])
            return sorted(stored, key=_sort_key)

    def count(self, scope: SessionScope, session_id: str) -> int:
        """返回会话消息条数；不存在时为 ``0``。"""
        with self._lock:
            return len(self._messages.get(scope.storage_key(session_id), []))

    def clear(self, scope: SessionScope, session_id: str) -> None:
        """清空会话消息（幂等：不存在时为空操作）。"""
        with self._lock:
            self._messages.pop(scope.storage_key(session_id), None)


def _sort_key(message: MedMessage) -> tuple[int, str]:
    """消息时序排序键 ``(created_at, message_id)``。"""
    return (message.created_at, message.message_id)
