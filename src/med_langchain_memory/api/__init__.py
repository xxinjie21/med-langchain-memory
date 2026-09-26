"""FastAPI 接口层。

面向医院侧调用方的 HTTP 接入层：应用工厂、请求日志中间件、统一异常处理与健康检查。
本层是**可选依赖**（``pip install med-langchain-memory[api]``），因此不在包顶层
``med_langchain_memory/__init__.py`` 中导入，避免未装 FastAPI 时整包不可用。
"""

from __future__ import annotations

from .app import configure_logging, create_app
from .errors import (
    DEFAULT_ERROR_STATUS,
    STATUS_MAP,
    ErrorResponse,
    error_code_for,
    register_exception_handlers,
    request_id_of,
    status_for,
)
from .middleware import (
    DEFAULT_REQUEST_ID_HEADER,
    PROCESS_TIME_HEADER,
    REQUEST_ID_STATE_KEY,
    RequestLoggingMiddleware,
)
from .routers import HealthProbe, HealthResponse, build_health_router, run_probe

__all__ = [
    "DEFAULT_ERROR_STATUS",
    "DEFAULT_REQUEST_ID_HEADER",
    "PROCESS_TIME_HEADER",
    "REQUEST_ID_STATE_KEY",
    "STATUS_MAP",
    "ErrorResponse",
    "HealthProbe",
    "HealthResponse",
    "RequestLoggingMiddleware",
    "build_health_router",
    "configure_logging",
    "create_app",
    "error_code_for",
    "register_exception_handlers",
    "request_id_of",
    "run_probe",
    "status_for",
]
