"""API 依赖注入：会话命名空间解析（含 API Key 鉴权）、仓储与脱敏引擎获取。

* :func:`resolve_session_scope` —— 从查询参数 ``tenant_id`` / ``dept_id`` 构造
  :class:`SessionScope`。**鉴权开关由应用状态决定**：``app.state.authenticator``
  为 ``None`` 时（未配置密钥，如本地开发）沿用「查询参数即命名空间」的历史行为；
  配置了认证器时则强制要求请求头携带 API Key，认证通过后由
  :func:`~med_langchain_memory.api.auth.authorize_scope` 校验租户与科室白名单，
  失败分别返回 401 / 403。
* :func:`authenticator_of` —— 从 ``app.state`` 取认证器；未配置或类型不符时返回 ``None``。
* :func:`get_session_repository` / :func:`get_message_repository` —— 从 ``app.state``
  取仓储实例，由应用工厂注入，便于测试替换为隔离实例或真实后端。
* :func:`get_message_masker` —— 从 ``app.state.masker`` 取按租户配置的脱敏策略分发器，
  供消息查询端点按 ``mask`` 开关执行字段级正则脱敏。
* :func:`get_history_resolver` —— 从 ``app.state.history_resolver`` 取会话历史解析器，
  供管理端点（跨存储迁移 / 快照导出）按后端名构造存储句柄。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query, Request

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.privacy.policies import PolicyMasker
from med_langchain_memory.stores.history_resolver import HistoryResolver
from med_langchain_memory.stores.message_repository import MessageRepository
from med_langchain_memory.stores.session_repository import (
    SessionRepository,
    SessionScope,
)

from .auth import ApiKeyAuthenticator, authorize_scope


def authenticator_of(request: Request) -> ApiKeyAuthenticator | None:
    """读取应用状态中的 API Key 认证器。

    Args:
        request: 当前请求。

    Returns:
        已配置的 :class:`ApiKeyAuthenticator`；未配置或类型不符时返回 ``None``
        （表示该应用未启用鉴权）。
    """
    authenticator: object = getattr(request.app.state, "authenticator", None)
    return authenticator if isinstance(authenticator, ApiKeyAuthenticator) else None


def resolve_session_scope(
    request: Request,
    tenant_id: Annotated[IdStr, Query(description="医院/机构租户 ID")],
    dept_id: Annotated[IdStr, Query(description="科室 ID")],
) -> SessionScope:
    """构造并鉴权会话命名空间。

    Args:
        request: 当前请求（用于读取应用状态中的认证器与请求头）。
        tenant_id: 医院/机构租户 ID。
        dept_id: 科室 ID。

    Returns:
        会话命名空间坐标 :class:`SessionScope`。

    Raises:
        AuthenticationError: 应用已启用鉴权但请求未携带有效 API Key 时（401）。
        AuthorizationError: 密钥与请求声明的租户不一致，或科室不在白名单内时（403）。
    """
    authenticator = authenticator_of(request)
    if authenticator is None:
        return SessionScope(tenant_id=tenant_id, dept_id=dept_id)
    principal = authenticator.authenticate(request.headers.get(authenticator.header_name))
    return authorize_scope(principal, tenant_id=tenant_id, dept_id=dept_id)


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


def get_history_resolver(request: Request) -> HistoryResolver:
    """从应用状态读取会话历史解析器。

    Args:
        request: 当前请求。

    Returns:
        应用工厂注入的 :class:`HistoryResolver`。

    Raises:
        RuntimeError: 应用未注入解析器（``app.state.history_resolver`` 缺失或类型不符）。
    """
    resolver: object = getattr(request.app.state, "history_resolver", None)
    if not isinstance(resolver, HistoryResolver):
        raise RuntimeError("history resolver is not configured on app.state")
    return resolver


#: 会话历史解析器依赖别名。
HistoryResolverDep = Annotated[HistoryResolver, Depends(get_history_resolver)]
