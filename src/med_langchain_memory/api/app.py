"""FastAPI 应用工厂。

只暴露一个 :func:`create_app` 工厂，不在模块导入时创建应用实例（避免 import 副作用，
也便于测试构造多个互相隔离的实例）。本地启动：

.. code-block:: bash

    uvicorn med_langchain_memory.api.app:create_app --factory

装配顺序（自外向内）：请求日志中间件 → 统一异常处理 → 健康检查路由 → 会话路由 → 消息路由
→ 管理路由。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from fastapi import FastAPI

from med_langchain_memory.config import PACKAGE_LOGGER_NAME, MedMemorySettings
from med_langchain_memory.privacy.policies import PolicyMasker
from med_langchain_memory.stores.history_resolver import (
    HistoryResolver,
    StoreFactoryHistoryResolver,
)
from med_langchain_memory.stores.message_repository import (
    InMemoryMessageRepository,
    MessageRepository,
)
from med_langchain_memory.stores.session_repository import (
    InMemorySessionRepository,
    SessionRepository,
)

from .errors import register_exception_handlers
from .middleware import RequestLoggingMiddleware
from .routers.admin import build_admin_router
from .routers.health import HealthProbe, build_health_router
from .routers.messages import build_messages_router
from .routers.sessions import build_sessions_router


def configure_logging(level: str) -> logging.Logger:
    """设置包级 logger 的日志级别。

    库不自行添加 handler，输出目标交由应用侧（``logging.basicConfig`` / dictConfig）
    决定，符合「库不配置全局日志」的惯例。

    Args:
        level: 标准库日志级别名（``DEBUG`` / ``INFO`` / ``WARNING`` / ``ERROR`` / ``CRITICAL``）。

    Returns:
        已设置级别的包级 logger。
    """
    logger = logging.getLogger(PACKAGE_LOGGER_NAME)
    logger.setLevel(level)
    return logger


def create_app(
    settings: MedMemorySettings | None = None,
    *,
    health_probes: Mapping[str, HealthProbe] | None = None,
    session_repository: SessionRepository | None = None,
    message_repository: MessageRepository | None = None,
    masker: PolicyMasker | None = None,
    history_resolver: HistoryResolver | None = None,
) -> FastAPI:
    """构建 FastAPI 应用。

    Args:
        settings: 服务配置；缺省时从环境变量构造（前缀 ``MED_MEMORY_``）。
        health_probes: 就绪探针映射（名称 → 无参可调用），供后续迭代接入存储连通性检查。
        session_repository: 会话仓储实现；缺省使用进程内实现
            :class:`~med_langchain_memory.stores.session_repository.InMemorySessionRepository`。
        message_repository: 消息仓储实现；缺省使用进程内实现
            :class:`~med_langchain_memory.stores.message_repository.InMemoryMessageRepository`。
        masker: 按租户配置的脱敏策略分发器；缺省使用内置规则作用于 ``content`` 字段的
            :class:`~med_langchain_memory.privacy.policies.PolicyMasker`。
        history_resolver: 会话历史解析器；缺省使用委托存储工厂的
            :class:`~med_langchain_memory.stores.history_resolver.StoreFactoryHistoryResolver`，
            供管理端点按后端名构造存储句柄。

    Returns:
        已挂载请求日志中间件、统一异常处理、健康检查、会话管理、消息与管理路由的
        ``FastAPI`` 实例；构造后的配置可从 ``app.state.settings`` 读取，仓储、脱敏分发器与
        历史解析器分别从 ``app.state.session_repository`` / ``app.state.message_repository`` /
        ``app.state.masker`` / ``app.state.history_resolver`` 读取。
    """
    resolved = settings if settings is not None else MedMemorySettings()
    configure_logging(resolved.log_level)

    app = FastAPI(
        title=resolved.app_name,
        version=resolved.app_version,
        docs_url=resolved.docs_url,
        redoc_url=resolved.redoc_url,
        openapi_url=resolved.openapi_url,
    )
    app.state.settings = resolved
    app.state.session_repository = (
        session_repository if session_repository is not None else InMemorySessionRepository()
    )
    app.state.message_repository = (
        message_repository if message_repository is not None else InMemoryMessageRepository()
    )
    app.state.masker = masker if masker is not None else PolicyMasker()
    app.state.history_resolver = (
        history_resolver if history_resolver is not None else StoreFactoryHistoryResolver()
    )

    app.add_middleware(RequestLoggingMiddleware, header_name=resolved.request_id_header)
    register_exception_handlers(app)
    app.include_router(build_health_router(resolved, health_probes))
    app.include_router(build_sessions_router())
    app.include_router(build_messages_router())
    app.include_router(build_admin_router())
    return app
