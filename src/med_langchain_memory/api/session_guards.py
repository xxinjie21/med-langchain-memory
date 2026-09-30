"""会话读取守卫：把「取会话 + 可见性 / 状态校验」抽成可复用的纯函数。

三个守卫共用同一套 404 / 409 语义，供会话、消息与管理路由复用，避免各路由
各自复制一份 ``_require`` 实现：

* :func:`require_session` —— 会话存在即可（管理端点需要处理已软删除的会话，
  例如合规导出与迁移）；
* :func:`require_visible_session` —— 已软删除（``DELETED``）的会话按 404 处理，
  用于普通查询；
* :func:`require_active_session` —— 仅 ``ACTIVE`` 允许写入，其余状态抛 409
  （:class:`~med_langchain_memory.exceptions.SessionNotActiveError`）。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import SessionNotActiveError, SessionNotFoundError
from med_langchain_memory.stores.session_repository import SessionRepository, SessionScope


def require_session(
    repository: SessionRepository, scope: SessionScope, session_id: str
) -> SessionMeta:
    """读取会话元数据，不存在时抛出 404 对应的领域异常。

    Args:
        repository: 会话仓储。
        scope: 命名空间坐标。
        session_id: 会话 ID。

    Returns:
        命中的会话元数据（可能处于任意状态，含 ``DELETED``）。

    Raises:
        SessionNotFoundError: 指定命名空间下不存在该会话。
    """
    meta = repository.get(scope, session_id)
    if meta is None:
        raise SessionNotFoundError(f"session not found: {scope.storage_key(session_id)}")
    return meta


def require_visible_session(
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
    meta = require_session(repository, scope, session_id)
    if meta.status is SessionStatus.DELETED:
        raise SessionNotFoundError(f"session not found: {scope.storage_key(session_id)}")
    return meta


def require_active_session(
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
    meta = require_session(repository, scope, session_id)
    if meta.status is not SessionStatus.ACTIVE:
        raise SessionNotActiveError(
            f"cannot append messages to {scope.storage_key(session_id)}: "
            f"status is {meta.status.value}, expected {SessionStatus.ACTIVE.value}"
        )
    return meta
