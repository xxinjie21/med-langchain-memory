"""FastAPI 接口层。

面向医院侧调用方的 HTTP 接入层：应用工厂、请求日志中间件、统一异常处理、健康检查、
会话管理、消息与管理端点（归档统计 / 跨存储迁移 / 快照导出）。
本层是**可选依赖**（``pip install med-langchain-memory[api]``），因此不在包顶层
``med_langchain_memory/__init__.py`` 中导入，避免未装 FastAPI 时整包不可用。
"""

from __future__ import annotations

from .app import configure_logging, create_app
from .cursor import (
    DEFAULT_MESSAGE_PAGE_SIZE,
    MAX_MESSAGE_PAGE_SIZE,
    MessageCursor,
    decode_cursor,
    encode_cursor,
    paginate,
)
from .deps import (
    HistoryResolverDep,
    MessageMaskerDep,
    MessageRepositoryDep,
    SessionRepositoryDep,
    SessionScopeDep,
    get_history_resolver,
    get_message_masker,
    get_message_repository,
    get_session_repository,
    resolve_session_scope,
)
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
from .routers import (
    HealthProbe,
    HealthResponse,
    build_admin_router,
    build_health_router,
    build_messages_router,
    build_sessions_router,
    run_probe,
)
from .schemas import (
    MAX_APPEND_BATCH,
    MAX_MIGRATION_BATCH_SIZE,
    AdminStatsResponse,
    MessageAppendRequest,
    MessageAppendResponse,
    MessageCreate,
    MessageListResponse,
    MessageResponse,
    MigrationRequest,
    MigrationResponse,
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    SnapshotRequest,
    SnapshotResponse,
)

__all__ = [
    "DEFAULT_ERROR_STATUS",
    "DEFAULT_MESSAGE_PAGE_SIZE",
    "DEFAULT_REQUEST_ID_HEADER",
    "MAX_APPEND_BATCH",
    "MAX_MESSAGE_PAGE_SIZE",
    "MAX_MIGRATION_BATCH_SIZE",
    "PROCESS_TIME_HEADER",
    "REQUEST_ID_STATE_KEY",
    "STATUS_MAP",
    "AdminStatsResponse",
    "ErrorResponse",
    "HealthProbe",
    "HealthResponse",
    "HistoryResolverDep",
    "MessageAppendRequest",
    "MessageAppendResponse",
    "MessageCreate",
    "MessageCursor",
    "MessageListResponse",
    "MessageMaskerDep",
    "MessageRepositoryDep",
    "MessageResponse",
    "MigrationRequest",
    "MigrationResponse",
    "RequestLoggingMiddleware",
    "SessionCreateRequest",
    "SessionListResponse",
    "SessionRepositoryDep",
    "SessionResponse",
    "SessionScopeDep",
    "SnapshotRequest",
    "SnapshotResponse",
    "build_admin_router",
    "build_health_router",
    "build_messages_router",
    "build_sessions_router",
    "configure_logging",
    "create_app",
    "decode_cursor",
    "encode_cursor",
    "error_code_for",
    "get_history_resolver",
    "get_message_masker",
    "get_message_repository",
    "get_session_repository",
    "paginate",
    "register_exception_handlers",
    "request_id_of",
    "resolve_session_scope",
    "run_probe",
    "status_for",
]
