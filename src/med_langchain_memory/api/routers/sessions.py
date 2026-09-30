"""会话管理端点：创建 / 查询 / 列表 / 关闭 / 归档 / 软删除。

命名空间（``tenant_id`` + ``dept_id``）统一由查询参数显式声明，
会话 ID 走路径参数，因此路由本身无状态、无隐式上下文。

端点一览：

======================================  ======  ==================================
方法 + 路径                              状态码  说明
======================================  ======  ==================================
``POST   /sessions``                     201     创建会话（``session_id`` 可省略，服务端生成）
``GET    /sessions``                     200     分页列表（可按 ``status`` 过滤）
``GET    /sessions/{session_id}``        200     查询单个会话
``POST   /sessions/{session_id}/close``  200     ``ACTIVE`` → ``CLOSED``
``POST   /sessions/{session_id}/archive``200     关闭/活跃 → ``ARCHIVED``
``DELETE /sessions/{session_id}``        200     软删除标记（仅 ``ARCHIVED`` → ``DELETED``）
======================================  ======  ==================================

状态流转非法返回 409（:class:`~med_langchain_memory.exceptions.StateTransitionError`），
会话不存在返回 404（:class:`~med_langchain_memory.exceptions.SessionNotFoundError`），
二者均由 :mod:`med_langchain_memory.api.errors` 统一映射为 ``ErrorResponse``。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import APIRouter, Query, status

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.stores.session_repository import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    SessionRepository,
    SessionScope,
)

from ..deps import SessionRepositoryDep, SessionScopeDep
from ..schemas import SessionCreateRequest, SessionListResponse, SessionResponse
from ..session_guards import require_session


def _transition(
    repository: SessionRepository,
    scope: SessionScope,
    session_id: str,
    target: SessionStatus,
) -> SessionMeta:
    """加载会话 → 状态流转 → 回写仓储。

    Args:
        repository: 会话仓储。
        scope: 命名空间坐标。
        session_id: 会话 ID。
        target: 目标状态。

    Returns:
        流转并持久化后的会话元数据。

    Raises:
        SessionNotFoundError: 会话不存在时。
        StateTransitionError: 当前状态不允许流转到 ``target`` 时。
    """
    meta = require_session(repository, scope, session_id)
    return repository.update(meta.transition_to(target))


def build_sessions_router() -> APIRouter:
    """构建会话管理路由。

    Returns:
        已注册全部会话端点的 ``APIRouter``（前缀 ``/sessions``）。
    """
    router = APIRouter(prefix="/sessions", tags=["sessions"])

    @router.post(
        "",
        response_model=SessionResponse,
        status_code=status.HTTP_201_CREATED,
        summary="创建会话",
        responses={status.HTTP_409_CONFLICT: {"description": "会话已存在"}},
    )
    async def create_session(
        payload: SessionCreateRequest,
        scope: SessionScopeDep,
        repository: SessionRepositoryDep,
    ) -> SessionResponse:
        """创建会话：``session_id`` 缺省时由服务端生成 UUID。"""
        meta = SessionMeta(
            session_id=payload.session_id or str(uuid.uuid4()),
            tenant_id=scope.tenant_id,
            dept_id=scope.dept_id,
            patient_id=payload.patient_id,
            metadata=payload.metadata,
        )
        return SessionResponse.from_meta(repository.add(meta))

    @router.get("", response_model=SessionListResponse, summary="分页查询会话")
    async def list_sessions(
        scope: SessionScopeDep,
        repository: SessionRepositoryDep,
        session_status: Annotated[
            SessionStatus | None, Query(alias="status", description="按状态过滤")
        ] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_PAGE_SIZE)] = DEFAULT_PAGE_SIZE,
        offset: Annotated[int, Query(ge=0)] = 0,
    ) -> SessionListResponse:
        """分页列出本租户本科室的会话，按创建时间升序。"""
        items, total = repository.list(scope, status=session_status, limit=limit, offset=offset)
        return SessionListResponse(
            total=total,
            limit=limit,
            offset=offset,
            items=[SessionResponse.from_meta(meta) for meta in items],
        )

    @router.get(
        "/{session_id}",
        response_model=SessionResponse,
        summary="查询单个会话",
        responses={status.HTTP_404_NOT_FOUND: {"description": "会话不存在"}},
    )
    async def get_session(
        session_id: IdStr,
        scope: SessionScopeDep,
        repository: SessionRepositoryDep,
    ) -> SessionResponse:
        """按会话 ID 查询详情。"""
        return SessionResponse.from_meta(require_session(repository, scope, session_id))

    @router.post(
        "/{session_id}/close",
        response_model=SessionResponse,
        summary="关闭会话",
        responses={
            status.HTTP_404_NOT_FOUND: {"description": "会话不存在"},
            status.HTTP_409_CONFLICT: {"description": "状态流转非法"},
        },
    )
    async def close_session(
        session_id: IdStr,
        scope: SessionScopeDep,
        repository: SessionRepositoryDep,
    ) -> SessionResponse:
        """关闭会话（``ACTIVE`` → ``CLOSED``）。"""
        meta = _transition(repository, scope, session_id, SessionStatus.CLOSED)
        return SessionResponse.from_meta(meta)

    @router.post(
        "/{session_id}/archive",
        response_model=SessionResponse,
        summary="归档会话",
        responses={
            status.HTTP_404_NOT_FOUND: {"description": "会话不存在"},
            status.HTTP_409_CONFLICT: {"description": "状态流转非法"},
        },
    )
    async def archive_session(
        session_id: IdStr,
        scope: SessionScopeDep,
        repository: SessionRepositoryDep,
    ) -> SessionResponse:
        """归档会话（``ACTIVE`` / ``CLOSED`` → ``ARCHIVED``）。"""
        meta = _transition(repository, scope, session_id, SessionStatus.ARCHIVED)
        return SessionResponse.from_meta(meta)

    @router.delete(
        "/{session_id}",
        response_model=SessionResponse,
        summary="软删除会话",
        responses={
            status.HTTP_404_NOT_FOUND: {"description": "会话不存在"},
            status.HTTP_409_CONFLICT: {"description": "仅归档会话可软删除"},
        },
    )
    async def delete_session(
        session_id: IdStr,
        scope: SessionScopeDep,
        repository: SessionRepositoryDep,
    ) -> SessionResponse:
        """软删除标记（仅 ``ARCHIVED`` → ``DELETED``），消息数据由保留期策略清理。"""
        meta = _transition(repository, scope, session_id, SessionStatus.DELETED)
        return SessionResponse.from_meta(meta)

    return router
