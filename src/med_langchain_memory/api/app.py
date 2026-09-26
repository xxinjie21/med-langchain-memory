"""FastAPI 应用工厂。

只暴露一个 :func:`create_app` 工厂，不在模块导入时创建应用实例（避免 import 副作用，
也便于测试构造多个互相隔离的实例）。本地启动：

.. code-block:: bash

    uvicorn med_langchain_memory.api.app:create_app --factory

装配顺序（自外向内）：请求日志中间件 → 统一异常处理 → 健康检查路由。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

from fastapi import FastAPI

from med_langchain_memory.config import PACKAGE_LOGGER_NAME, MedMemorySettings

from .errors import register_exception_handlers
from .middleware import RequestLoggingMiddleware
from .routers.health import HealthProbe, build_health_router


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
) -> FastAPI:
    """构建 FastAPI 应用。

    Args:
        settings: 服务配置；缺省时从环境变量构造（前缀 ``MED_MEMORY_``）。
        health_probes: 就绪探针映射（名称 → 无参可调用），供后续迭代接入存储连通性检查。

    Returns:
        已挂载请求日志中间件、统一异常处理与健康检查路由的 ``FastAPI`` 实例；
        构造后的配置可从 ``app.state.settings`` 读取。
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

    app.add_middleware(RequestLoggingMiddleware, header_name=resolved.request_id_header)
    register_exception_handlers(app)
    app.include_router(build_health_router(resolved, health_probes))
    return app
