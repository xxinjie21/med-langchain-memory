"""消息端点：批量追加与游标分页查询（含脱敏开关）。

端点一览：

========================================================  ======  ===================================
方法 + 路径                                                状态码  说明
========================================================  ======  ===================================
``POST /sessions/{session_id}/messages``                   201     批量追加消息（1..100 条）
``GET  /sessions/{session_id}/messages``                   200     游标分页查询（``cursor`` / ``limit`` / ``mask``）
========================================================  ======  ===================================

设计要点：

* **命名空间由查询参数显式声明**，``patient_id`` 一律取自会话元数据，调用方无法伪造；
* **写入要求会话处于 ``ACTIVE``**，其余状态返回 409（``SessionNotActiveError`` 继承
  ``StateTransitionError``，复用 409 映射）；会话不存在返回 404；
* **读取时脱敏**：正文按原文落库，``mask``（默认 ``true``）为真时按租户策略做字段级
  正则脱敏，响应中的 ``masked`` 标记与请求参数一致；``mask=false`` 返回原文，
  供已授权的医生端回显使用；
* **游标分页**：翻页期间新追加的消息不会造成重复或漏读，见
  :mod:`med_langchain_memory.api.cursor`。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status

from med_langchain_memory.api.cursor import (
    DEFAULT_MESSAGE_PAGE_SIZE,
    MAX_MESSAGE_PAGE_SIZE,
    decode_cursor,
    encode_cursor,
    paginate,
)
from med_langchain_memory.domain.message import IdStr, MedMessage
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import SessionNotActiveError, SessionNotFoundError
from med_langchain_memory.stores.session_repository import SessionRepository, SessionScope

from ..deps import MessageMaskerDep, MessageRepositoryDep, SessionRepositoryDep, SessionScopeDep
from ..schemas import (
    MessageAppendRequest,
    MessageAppendResponse,
    MessageListResponse,
    MessageResponse,
)

_NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "会话不存在"}
}


def _require(repository: SessionRepository, scope: SessionScope, session_id: str) -> SessionMeta:
    """读取会话元数据，不存在时抛出 404 对应的领域异常。

    Args:
        repository: 会话仓储。
        scope: 命名空间坐标。
        session_id: 会话 ID。

    Returns:
        命中的会话元数据。

    Raises:
        SessionNotFoundError: 指定命名空间下不存在该会话。
    """
    meta = repository.get(scope, session_id)
    if meta is None:
        raise SessionNotFoundError(f"session not found: {scope.storage_key(session_id)}")
    return meta


def _require_visible(
    repository: SessionRepository, scope: SessionScope, session_id: str
) -> SessionMeta:
    """读取会话，要求其对普通查询可见（已软删除的会话按 404 处理）。

    Args:
        repository: 会话仓储。
        scope: 命名空间坐标。
        session_id: 会话 ID。

    Returns:
        命中的会话元数据。

    Raises:
        SessionNotFoundError: 会话不存在或已处于 ``DELETED`` 状态时。
    """
    meta = _require(repository, scope, session_id)
    if meta.status is SessionStatus.DELETED:
        raise SessionNotFoundError(f"session not found: {scope.storage_key(session_id)}")
    return meta


def _require_active(
    repository: SessionRepository, scope: SessionScope, session_id: str
) -> SessionMeta:
    """读取会话，要求其处于 ``ACTIVE`` 状态（否则拒绝写入）。

    Args:
        repository: 会话仓储。
        scope: 命名空间坐标。
        session_id: 会话 ID。

    Returns:
        命中的会话元数据。

    Raises:
        SessionNotFoundError: 会话不存在时。
        SessionNotActiveError: 会话不处于 ``ACTIVE`` 状态时。
    """
    meta = _require(repository, scope, session_id)
    if meta.status is not SessionStatus.ACTIVE:
        raise SessionNotActiveError(
            f"cannot append messages to {scope.storage_key(session_id)}: "
            f"status is {meta.status.value}, expected {SessionStatus.ACTIVE.value}"
        )
    return meta


def _build_messages(
    payload: MessageAppendRequest,
    *,
    scope: SessionScope,
    session_id: str,
    patient_id: str,
) -> list[MedMessage]:
    """把追加请求体转换为领域消息。

    命名空间与患者 ID 由会话元数据补齐，请求体中的字段只决定内容与时序。

    Args:
        payload: 追加请求体。
        scope: 命名空间坐标。
        session_id: 会话 ID。
        patient_id: 患者 ID（取自会话元数据）。

    Returns:
        领域消息列表，顺序与请求体一致。
    """
    built: list[MedMessage] = []
    for item in payload.messages:
        fields: dict[str, Any] = {
            "session_id": session_id,
            "tenant_id": scope.tenant_id,
            "dept_id": scope.dept_id,
            "patient_id": patient_id,
            "role": item.role,
            "content": item.content,
            "token_count": item.token_count,
            "metadata": dict(item.metadata),
        }
        if item.message_id is not None:
            fields["message_id"] = item.message_id
        if item.created_at is not None:
            fields["created_at"] = item.created_at
        built.append(MedMessage(**fields))
    return built


def build_messages_router() -> APIRouter:
    """构建消息路由。

    Returns:
        已注册追加与查询端点的 ``APIRouter``（前缀 ``/sessions``）。
    """
    router = APIRouter(prefix="/sessions", tags=["messages"])

    @router.post(
        "/{session_id}/messages",
        response_model=MessageAppendResponse,
        status_code=status.HTTP_201_CREATED,
        summary="批量追加消息",
        responses={
            **_NOT_FOUND_RESPONSE,
            status.HTTP_409_CONFLICT: {"description": "会话不处于 ACTIVE 状态"},
        },
    )
    async def append_messages(
        session_id: IdStr,
        payload: MessageAppendRequest,
        scope: SessionScopeDep,
        sessions: SessionRepositoryDep,
        repository: MessageRepositoryDep,
    ) -> MessageAppendResponse:
        """向会话追加消息并同步刷新会话的消息计数。"""
        meta = _require_active(sessions, scope, session_id)
        batch = _build_messages(
            payload, scope=scope, session_id=session_id, patient_id=meta.patient_id
        )
        stored = repository.append(scope, session_id, batch)
        updated = sessions.update(meta.touch(len(stored)))
        return MessageAppendResponse(
            session_id=session_id,
            appended=len(stored),
            message_count=updated.message_count,
            items=[MessageResponse.from_med(message) for message in stored],
        )

    @router.get(
        "/{session_id}/messages",
        response_model=MessageListResponse,
        summary="游标分页查询消息",
        responses=_NOT_FOUND_RESPONSE,
    )
    async def list_messages(
        session_id: IdStr,
        scope: SessionScopeDep,
        sessions: SessionRepositoryDep,
        repository: MessageRepositoryDep,
        masker: MessageMaskerDep,
        cursor: Annotated[str | None, Query(description="上一页返回的 next_cursor")] = None,
        limit: Annotated[int, Query(ge=1, le=MAX_MESSAGE_PAGE_SIZE)] = DEFAULT_MESSAGE_PAGE_SIZE,
        mask: Annotated[bool, Query(description="是否对返回内容做字段级脱敏")] = True,
    ) -> MessageListResponse:
        """按时序升序游标分页返回消息，可按开关执行字段级脱敏。"""
        _require_visible(sessions, scope, session_id)
        page, next_cursor = paginate(
            repository.read(scope, session_id),
            limit=limit,
            cursor=decode_cursor(cursor) if cursor is not None else None,
        )
        visible = masker.mask_messages(scope.tenant_id, page) if mask else page
        return MessageListResponse(
            session_id=session_id,
            limit=limit,
            masked=mask,
            next_cursor=encode_cursor(next_cursor) if next_cursor is not None else None,
            items=[MessageResponse.from_med(message) for message in visible],
        )

    return router
