"""统一异常处理：把领域异常与校验失败映射为标准 HTTP 错误响应。

响应体统一为 :class:`ErrorResponse`（``error`` / ``message`` / ``request_id`` /
``detail``），便于前端与下游服务按 ``error`` 字段做稳定分支：

* 领域异常（:class:`~med_langchain_memory.exceptions.MedMemoryError` 及其子类）
  → 按 :data:`STATUS_MAP` 映射状态码，未登记的类回落到 500；
* ``RequestValidationError``（路径/查询/请求体参数校验失败）→ 422，
  并在 ``detail.fields`` 中给出出错字段路径；
* ``StarletteHTTPException``（404 / 405 等框架级异常）→ 保留原状态码，
  错误码形如 ``http_404``；
* 其他未捕获异常 → 500，错误码固定 ``internal_server_error``，
  响应体**不泄漏**内部异常信息（仅记录到日志）。

设计取舍：只做「异常 → 响应」的映射，不做重试、不做错误上报通道。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Final

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from med_langchain_memory.config import PACKAGE_LOGGER_NAME
from med_langchain_memory.exceptions import (
    AuditSinkError,
    FallbackExhaustedError,
    IntegrityError,
    LockAcquisitionError,
    LockError,
    MedMemoryError,
    StateTransitionError,
    StorageError,
    StoreNotFoundError,
    StoreRegistrationError,
    TenantIsolationError,
    ValidationError,
)

_LOGGER = logging.getLogger(f"{PACKAGE_LOGGER_NAME}.api.errors")

#: 未登记异常的兜底状态码。
DEFAULT_ERROR_STATUS: Final = 500

#: 领域异常 → HTTP 状态码映射。
#:
#: **顺序敏感**：子类必须排在基类之前（如 ``LockAcquisitionError`` 先于
#: ``LockError``、``FallbackExhaustedError`` 先于 ``StorageError``）。
STATUS_MAP: Final[tuple[tuple[type[MedMemoryError], int], ...]] = (
    (ValidationError, 400),
    (TenantIsolationError, 403),
    (StoreNotFoundError, 404),
    (StateTransitionError, 409),
    (StoreRegistrationError, 409),
    (IntegrityError, 409),
    (LockAcquisitionError, 409),
    (AuditSinkError, 500),
    (LockError, 503),
    (FallbackExhaustedError, 503),
    (StorageError, 503),
)

#: ``CamelCase`` → ``snake_case`` 的边界切分。
_CAMEL_BOUNDARY: Final = re.compile(r"(?<!^)(?=[A-Z])")


class ErrorResponse(BaseModel):
    """统一错误响应体。

    Attributes:
        error: 机器可读错误码，如 ``validation_error``、``http_404``。
        message: 人类可读的错误描述。
        request_id: 请求追踪 ID（由请求日志中间件注入，缺失时为 ``None``）。
        detail: 附加结构化信息（如参数校验失败的字段路径），无则为 ``None``。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    error: str = Field(min_length=1)
    message: str
    request_id: str | None = None
    detail: dict[str, Any] | None = None


def error_code_for(exc: BaseException) -> str:
    """由异常类名推导机器可读错误码（``StorageError`` → ``storage_error``）。

    Args:
        exc: 任意异常实例。

    Returns:
        小写下划线形式的错误码。
    """
    return _CAMEL_BOUNDARY.sub("_", type(exc).__name__).lower()


def status_for(exc: BaseException) -> int:
    """返回领域异常对应的 HTTP 状态码。

    Args:
        exc: 任意异常实例。

    Returns:
        :data:`STATUS_MAP` 中首个匹配类型的状态码；无匹配时返回 500。
    """
    for exc_type, status_code in STATUS_MAP:
        if isinstance(exc, exc_type):
            return status_code
    return DEFAULT_ERROR_STATUS


def request_id_of(request: Request) -> str | None:
    """读取中间件写入 ``request.state`` 的请求 ID。

    Args:
        request: 当前请求。

    Returns:
        请求 ID 字符串；未注入或类型异常时返回 ``None``。
    """
    value: object = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else None


def register_exception_handlers(app: FastAPI) -> None:
    """为应用注册统一异常处理器（领域异常 / 参数校验 / HTTP 异常 / 未捕获异常）。

    Args:
        app: 目标 FastAPI 应用（原地修改）。
    """

    @app.exception_handler(MedMemoryError)
    async def _handle_med_memory_error(request: Request, exc: Exception) -> JSONResponse:
        """领域异常 → 按 :data:`STATUS_MAP` 映射状态码的统一错误响应。"""
        return JSONResponse(
            status_code=status_for(exc),
            content=ErrorResponse(
                error=error_code_for(exc),
                message=str(exc) or type(exc).__name__,
                request_id=request_id_of(request),
            ).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """请求参数校验失败 → 422，并回传出错字段路径。"""
        fields = sorted({".".join(str(part) for part in error["loc"]) for error in exc.errors()})
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="request_validation_error",
                message="请求参数校验失败",
                request_id=request_id_of(request),
                detail={"fields": fields},
            ).model_dump(),
        )

    @app.exception_handler(StarletteHTTPException)
    async def _handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """框架级 HTTP 异常（404 / 405 等）→ 保留原状态码的统一错误响应。"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error=f"http_{exc.status_code}",
                message=str(exc.detail),
                request_id=request_id_of(request),
            ).model_dump(),
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        """未捕获异常 → 500（不泄漏内部细节，仅落日志）。"""
        _LOGGER.exception("unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=DEFAULT_ERROR_STATUS,
            content=ErrorResponse(
                error="internal_server_error",
                message="服务器内部错误",
                request_id=request_id_of(request),
            ).model_dump(),
        )
