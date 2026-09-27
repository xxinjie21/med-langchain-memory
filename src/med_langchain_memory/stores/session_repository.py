"""会话索引仓储：会话元数据的新增、查询、更新与分页列表。

``MedChatMessageHistory`` 只承载「某个已知会话的消息读写」，无法回答「某租户某科室
有哪些会话」这类索引型问题。本模块补齐这一层：

* :class:`SessionScope` —— 会话命名空间坐标（``tenant_id`` + ``dept_id``），
  既是仓储的查询条件，也负责推导统一存储键；
* :class:`SessionRepository` —— 仓储抽象，只定义 4 个原语（add/get/update/list）；
* :class:`InMemorySessionRepository` —— 进程内字典实现，作为 API 层的默认后端与测试替身，
  由调用方通过 ``app.state.session_repository`` 替换为 Redis/MySQL 实现。

设计取舍：不做二级索引、不做全文检索、不缓存；会话状态流转由领域层
:class:`~med_langchain_memory.domain.session.SessionMeta` 负责，本层只做持久化。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod

from pydantic import BaseModel, ConfigDict

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import (
    IntegrityError,
    SessionNotFoundError,
    ValidationError,
)

#: ``list`` 单页默认条数。
DEFAULT_PAGE_SIZE = 50

#: ``list`` 单页最大条数（防止一次拉取过多会话）。
MAX_PAGE_SIZE = 200


class SessionScope(BaseModel):
    """会话命名空间坐标：租户 + 科室。

    Attributes:
        tenant_id: 医院/机构租户 ID。
        dept_id: 科室 ID。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tenant_id: IdStr
    dept_id: IdStr

    def storage_key(self, session_id: str) -> str:
        """返回指定会话在本命名空间下的统一存储键。

        Args:
            session_id: 会话 ID。

        Returns:
            ``med:chat:{tenant_id}:{dept_id}:{session_id}``。
        """
        return f"med:chat:{self.tenant_id}:{self.dept_id}:{session_id}"

    def matches(self, meta: SessionMeta) -> bool:
        """判断会话元数据是否属于本命名空间。

        Args:
            meta: 会话元数据。

        Returns:
            租户与科室同时匹配时为 ``True``。
        """
        return meta.tenant_id == self.tenant_id and meta.dept_id == self.dept_id


def validate_pagination(*, limit: int, offset: int) -> None:
    """校验分页参数合法性。

    Args:
        limit: 单页条数。
        offset: 起始偏移量。

    Raises:
        ValidationError: ``limit`` 不在 ``1..MAX_PAGE_SIZE`` 或 ``offset`` 为负数时。
    """
    if limit < 1 or limit > MAX_PAGE_SIZE:
        raise ValidationError(f"limit must be within 1..{MAX_PAGE_SIZE}, got {limit}")
    if offset < 0:
        raise ValidationError(f"offset must be >= 0, got {offset}")


class SessionRepository(ABC):
    """会话元数据仓储抽象。

    实现需保证：同一 :class:`SessionScope` 下 ``session_id`` 唯一；
    ``list`` 结果按 ``(created_at, session_id)`` 升序稳定排序。
    """

    @abstractmethod
    def add(self, meta: SessionMeta) -> SessionMeta:
        """新增会话。

        Args:
            meta: 会话元数据。

        Returns:
            已写入的会话元数据。

        Raises:
            IntegrityError: 同一命名空间下会话已存在时。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def get(self, scope: SessionScope, session_id: str) -> SessionMeta | None:
        """按命名空间查询会话。

        Args:
            scope: 命名空间坐标。
            session_id: 会话 ID。

        Returns:
            会话元数据；不存在返回 ``None``。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def update(self, meta: SessionMeta) -> SessionMeta:
        """整体覆盖已有会话。

        Args:
            meta: 会话元数据（以 ``storage_key`` 定位）。

        Returns:
            更新后的会话元数据。

        Raises:
            SessionNotFoundError: 目标会话不存在时。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现

    @abstractmethod
    def list(
        self,
        scope: SessionScope,
        *,
        status: SessionStatus | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[list[SessionMeta], int]:
        """按命名空间分页列出会话。

        Args:
            scope: 命名空间坐标。
            status: 状态过滤；``None`` 表示不过滤。
            limit: 单页条数。
            offset: 起始偏移量。

        Returns:
            ``(当前页会话列表, 命中总数)``。

        Raises:
            ValidationError: 分页参数非法时。
        """
        raise NotImplementedError  # pragma: no cover - 抽象方法由子类实现


class InMemorySessionRepository(SessionRepository):
    """基于进程内字典的会话仓储。

    以统一存储键为字典键，所有操作由一把互斥锁保护，可安全用于多线程
    ASGI worker（同一进程内）。进程重启即丢失，仅适用于开发与测试。

    Example:
        >>> repo = InMemorySessionRepository()
        >>> scope = SessionScope(tenant_id="hosp-a", dept_id="cardio")
        >>> repo.add(SessionMeta(
        ...     session_id="s-1", tenant_id="hosp-a", dept_id="cardio", patient_id="p-1"
        ... )).session_id
        's-1'
    """

    def __init__(self) -> None:
        """初始化空仓储。"""
        self._lock = threading.Lock()
        self._sessions: dict[str, SessionMeta] = {}

    def add(self, meta: SessionMeta) -> SessionMeta:
        """新增会话；键冲突时抛 :class:`IntegrityError`。"""
        key = meta.storage_key
        with self._lock:
            if key in self._sessions:
                raise IntegrityError(f"session already exists: {key}")
            self._sessions[key] = meta
        return meta

    def get(self, scope: SessionScope, session_id: str) -> SessionMeta | None:
        """按命名空间查询会话；不存在返回 ``None``。"""
        with self._lock:
            return self._sessions.get(scope.storage_key(session_id))

    def update(self, meta: SessionMeta) -> SessionMeta:
        """整体覆盖已有会话；不存在时抛 :class:`SessionNotFoundError`。"""
        key = meta.storage_key
        with self._lock:
            if key not in self._sessions:
                raise SessionNotFoundError(f"session not found: {key}")
            self._sessions[key] = meta
        return meta

    def list(
        self,
        scope: SessionScope,
        *,
        status: SessionStatus | None = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> tuple[list[SessionMeta], int]:
        """按命名空间分页列出会话，返回 ``(当前页, 总数)``。"""
        validate_pagination(limit=limit, offset=offset)
        with self._lock:
            matched = [
                meta
                for meta in self._sessions.values()
                if scope.matches(meta) and (status is None or meta.status == status)
            ]
        matched.sort(key=lambda meta: (meta.created_at, meta.session_id))
        return matched[offset : offset + limit], len(matched)
