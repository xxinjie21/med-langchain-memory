"""会话历史句柄解析器：按后端名构造 :class:`MedChatMessageHistory`。

管理端点（跨存储迁移 / 快照导出）需要在同一会话命名空间下取得**两个不同后端**
的历史句柄，但 HTTP 层不应直接依赖 :class:`StoreFactory` 的全局注册表——那是
进程级单例状态，既不便注入也难以在测试中隔离。

本模块把「按后端名 + 命名空间构造历史句柄」抽成 :class:`HistoryResolver` 抽象，
默认实现 :class:`StoreFactoryHistoryResolver` 委托 :class:`StoreFactory`；
测试或私有部署可实现自己的解析器（如按连接池复用客户端），通过
``app.state.history_resolver`` 注入。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from .base import MedChatMessageHistory
from .factory import StoreFactory
from .session_repository import SessionScope


class HistoryResolver(ABC):
    """会话历史句柄解析器抽象。"""

    @abstractmethod
    def resolve(
        self,
        backend: str,
        *,
        scope: SessionScope,
        session_id: str,
        patient_id: str,
        ttl_seconds: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MedChatMessageHistory:
        """构造指定后端 + 命名空间下的会话历史句柄。

        Args:
            backend: 已注册的后端名（如 ``memory`` / ``redis``）。
            scope: 会话命名空间坐标（租户 + 科室）。
            session_id: 会话 ID。
            patient_id: 患者 ID（通常取自会话元数据）。
            ttl_seconds: 会话级 TTL（秒）；``None`` 表示永不过期。
            options: 透传给具体存储实现的额外构造参数。

        Returns:
            对应后端的会话历史句柄。

        Raises:
            StoreNotFoundError: 后端未注册时。
            StorageError: 实现类不接受给定参数（构造签名不匹配）时。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现


class StoreFactoryHistoryResolver(HistoryResolver):
    """基于 :class:`StoreFactory` 全局注册表的默认解析器。

    Example:
        >>> from med_langchain_memory.stores.session_repository import SessionScope
        >>> scope = SessionScope(tenant_id="hosp-a", dept_id="cardio")
        >>> history = StoreFactoryHistoryResolver().resolve(
        ...     "memory", scope=scope, session_id="s-1", patient_id="p-1"
        ... )
        >>> history.storage_key
        'med:chat:hosp-a:cardio:s-1'
    """

    def resolve(
        self,
        backend: str,
        *,
        scope: SessionScope,
        session_id: str,
        patient_id: str,
        ttl_seconds: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MedChatMessageHistory:
        """委托 :meth:`StoreFactory.create` 实例化历史句柄。"""
        return StoreFactory.create(
            backend,
            session_id=session_id,
            tenant_id=scope.tenant_id,
            dept_id=scope.dept_id,
            patient_id=patient_id,
            ttl_seconds=ttl_seconds,
            **dict(options or {}),
        )
