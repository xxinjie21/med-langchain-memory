"""API 依赖注入：会话命名空间解析、会话仓储与消息仓储/脱敏引擎获取。

* :func:`resolve_session_scope` —— 从查询参数 ``tenant_id`` / ``dept_id`` 构造
  :class:`SessionScope`，所有会话端点都要求显式声明命名空间，杜绝跨租户误查；
  后续迭代接入 API Key 后，只需替换本函数即可把命名空间来源改为密钥 scope。
* :func:`get_session_repository` / :func:`get_message_repository` —— 从 ``app.state``
  取仓储实例，由应用工厂注入，便于测试替换为隔离实例或真实后端。
* :func:`get_message_masker` —— 从 ``app.state.masker`` 取按租户配置的脱敏策略分发器，
  供消息查询端点按 ``mask`` 开关执行字段级正则脱敏。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query, Request

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.privacy.policies import PolicyMasker
from med_langchain_memory.stores.message_repository import MessageRepository
from med_langchain_memory.stores.session_repository import (
    SessionRepository,
    SessionScope,
)


def resolve_session_scope(
    tenant_id: Annotated[IdStr, Query(description="医院/机构租户 ID")],
    dept_id: Annotated[IdStr, Query(description="科室 ID")],
) -> SessionScope:
    """由查询参数构造会话命名空间。

    Args:
        tenant_id: 医院/机构租户 ID。
        dept_id: 科室 ID。

    Returns:
        会话命名空间坐标 :class:`SessionScope`。
    """
    return SessionScope(tenant_id=tenant_id, dept_id=dept_id)


def get_session_repository(request: Request) -> SessionRepository:
    """从应用状态读取会话仓储。

    Args:
        request: 当前请求。

    Returns:
        应用工厂注入的 :class:`SessionRepository`。

    Raises:
        RuntimeError: 应用未注入会话仓储（``app.state.session_repository`` 缺失或类型不符）。
    """
    repository: object = getattr(request.app.state, "session_repository", None)
    if not isinstance(repository, SessionRepository):
        raise RuntimeError("session repository is not configured on app.state")
    return repository


#: 会话命名空间依赖别名（供路由签名直接使用）。
SessionScopeDep = Annotated[SessionScope, Depends(resolve_session_scope)]

#: 会话仓储依赖别名。
SessionRepositoryDep = Annotated[SessionRepository, Depends(get_session_repository)]


def get_message_repository(request: Request) -> MessageRepository:
    """从应用状态读取消息仓储。

    Args:
        request: 当前请求。

    Returns:
        应用工厂注入的 :class:`MessageRepository`。

    Raises:
        RuntimeError: 应用未注入消息仓储（``app.state.message_repository`` 缺失或类型不符）。
    """
    repository: object = getattr(request.app.state, "message_repository", None)
    if not isinstance(repository, MessageRepository):
        raise RuntimeError("message repository is not configured on app.state")
    return repository


def get_message_masker(request: Request) -> PolicyMasker:
    """从应用状态读取按租户配置的脱敏策略分发器。

    Args:
        request: 当前请求。

    Returns:
        应用工厂注入的 :class:`PolicyMasker`。

    Raises:
        RuntimeError: 应用未注入脱敏分发器（``app.state.masker`` 缺失或类型不符）。
    """
    masker: object = getattr(request.app.state, "masker", None)
    if not isinstance(masker, PolicyMasker):
        raise RuntimeError("masker is not configured on app.state")
    return masker


#: 消息仓储依赖别名。
MessageRepositoryDep = Annotated[MessageRepository, Depends(get_message_repository)]

#: 脱敏策略分发器依赖别名。
MessageMaskerDep = Annotated[PolicyMasker, Depends(get_message_masker)]
