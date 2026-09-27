"""API 依赖注入：会话命名空间解析与会话仓储获取。

* :func:`resolve_session_scope` —— 从查询参数 ``tenant_id`` / ``dept_id`` 构造
  :class:`SessionScope`，所有会话端点都要求显式声明命名空间，杜绝跨租户误查；
  后续迭代接入 API Key 后，只需替换本函数即可把命名空间来源改为密钥 scope。
* :func:`get_session_repository` —— 从 ``app.state.session_repository`` 取仓储实例，
  由应用工厂注入，便于测试替换为隔离实例或真实后端。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Query, Request

from med_langchain_memory.domain.message import IdStr
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
