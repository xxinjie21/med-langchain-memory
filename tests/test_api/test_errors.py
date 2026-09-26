"""统一异常处理单元测试（D30）。

覆盖：领域异常 → 状态码映射表全量分支与未登记类型兜底、错误码推导、请求 ID 提取
（已注入 / 未注入 / 类型异常）、四类异常处理器（领域异常 / 参数校验 / 框架 HTTP 异常 /
未捕获异常）的端到端响应。
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from starlette.requests import Request as StarletteRequest

from med_langchain_memory.api import (
    DEFAULT_ERROR_STATUS,
    STATUS_MAP,
    create_app,
    error_code_for,
    request_id_of,
    status_for,
)
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


class UnmappedError(MedMemoryError):
    """未登记到 :data:`STATUS_MAP` 的自定义领域异常（用于验证兜底分支）。"""


def _client_raising(exc: BaseException, *, raise_server_exceptions: bool = True) -> TestClient:
    """构造一个 ``/boom`` 端点会抛出 ``exc`` 的测试客户端。"""
    app = create_app()

    async def _boom() -> None:
        raise exc

    app.add_api_route("/boom", _boom, methods=["GET"])
    return TestClient(app, raise_server_exceptions=raise_server_exceptions)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (ValidationError("bad payload"), 400),
        (TenantIsolationError("cross tenant"), 403),
        (StoreNotFoundError("unknown store"), 404),
        (StateTransitionError("closed -> active"), 409),
        (StoreRegistrationError("duplicated name"), 409),
        (IntegrityError("checksum mismatch"), 409),
        (LockAcquisitionError("lock busy"), 409),
        (AuditSinkError("disk full"), 500),
        (LockError("redis unavailable"), 503),
        (FallbackExhaustedError("all stores down"), 503),
        (StorageError("io error"), 503),
    ],
)
def test_status_for_maps_every_registered_exception(exc: MedMemoryError, expected: int) -> None:
    """映射表中的每一条领域异常都命中预期状态码。"""
    assert status_for(exc) == expected


def test_status_for_falls_back_for_unmapped_domain_error() -> None:
    """未登记的领域异常（边界）回落到 500。"""
    assert status_for(UnmappedError("whatever")) == DEFAULT_ERROR_STATUS


def test_status_for_falls_back_for_foreign_exception() -> None:
    """与领域无关的异常（边界）同样回落到 500。"""
    assert status_for(RuntimeError("nope")) == DEFAULT_ERROR_STATUS


def test_status_map_orders_subclasses_before_bases() -> None:
    """映射表顺序敏感：子类必须先于其基类出现。"""
    types = [exc_type for exc_type, _ in STATUS_MAP]
    assert types.index(LockAcquisitionError) < types.index(LockError)
    assert types.index(FallbackExhaustedError) < types.index(StorageError)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (StorageError("x"), "storage_error"),
        (FallbackExhaustedError("x"), "fallback_exhausted_error"),
        (TenantIsolationError("x"), "tenant_isolation_error"),
        (RuntimeError("x"), "runtime_error"),
    ],
)
def test_error_code_for_derives_snake_case(exc: BaseException, expected: str) -> None:
    """类名按大驼峰边界切成小写下划线错误码。"""
    assert error_code_for(exc) == expected


def test_request_id_of_reads_injected_value() -> None:
    """请求 ID 由中间件注入后可被异常处理器读到，且与响应头一致。"""
    app = create_app()

    async def _rid(request: Request) -> dict[str, str | None]:
        return {"rid": request_id_of(request)}

    app.add_api_route("/rid", _rid, methods=["GET"])
    response = TestClient(app).get("/rid")
    assert response.json()["rid"] == response.headers["X-Request-ID"]


def test_request_id_of_returns_none_without_middleware() -> None:
    """未经过中间件的请求（边界）没有请求 ID。"""
    request = StarletteRequest({"type": "http", "method": "GET", "path": "/", "headers": []})
    assert request_id_of(request) is None


def test_request_id_of_returns_none_for_non_string_state() -> None:
    """``state.request_id`` 非字符串（边界）时返回 ``None``。"""
    app = create_app()

    async def _bad(request: Request) -> dict[str, Any]:
        request.state.request_id = 12345
        return {"rid": request_id_of(request)}

    app.add_api_route("/rid-bad", _bad, methods=["GET"])
    assert TestClient(app).get("/rid-bad").json()["rid"] is None


def test_domain_error_handler_returns_mapped_response() -> None:
    """领域异常经统一处理器返回映射状态码、错误码与请求 ID。"""
    response = _client_raising(StorageError("redis 不可用")).get("/boom")
    assert response.status_code == 503
    body = response.json()
    assert body["error"] == "storage_error"
    assert body["message"] == "redis 不可用"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["detail"] is None


def test_domain_error_handler_uses_class_name_for_empty_message() -> None:
    """领域异常无消息时（边界）回落到类名，避免空 message。"""
    response = _client_raising(MedMemoryError()).get("/boom")
    assert response.status_code == DEFAULT_ERROR_STATUS
    assert response.json()["message"] == "MedMemoryError"


def test_unmapped_domain_error_returns_500() -> None:
    """未登记领域异常经处理器返回 500。"""
    response = _client_raising(UnmappedError("custom")).get("/boom")
    assert response.status_code == 500
    assert response.json()["error"] == "unmapped_error"


def test_validation_error_handler_returns_422_with_field_paths() -> None:
    """参数校验失败返回 422 并给出出错字段路径。"""
    app = create_app()

    async def _items(limit: int = 1) -> dict[str, int]:
        return {"limit": limit}

    app.add_api_route("/items", _items, methods=["GET"])
    response = TestClient(app).get("/items", params={"limit": "abc"})
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "request_validation_error"
    assert body["message"] == "请求参数校验失败"
    assert body["detail"]["fields"] == ["query.limit"]


def test_http_exception_handler_returns_unified_404() -> None:
    """框架级 404 使用统一错误体。"""
    response = TestClient(create_app()).get("/not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "http_404"
    assert body["message"] == "Not Found"
    assert isinstance(body["request_id"], str)


def test_http_exception_handler_preserves_headers() -> None:
    """框架级 405 保留 ``Allow`` 响应头（边界：异常自带响应头）。"""
    response = TestClient(create_app()).post("/health")
    assert response.status_code == 405
    assert response.headers["allow"] == "GET"
    assert response.json()["error"] == "http_405"


def test_unexpected_error_handler_returns_generic_500(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """未捕获异常返回 500 且不泄漏内部细节，同时落日志。"""
    with caplog.at_level(logging.ERROR):
        response = _client_raising(
            RuntimeError("secret detail"), raise_server_exceptions=False
        ).get("/boom")
    assert response.status_code == 500
    body = response.json()
    assert body["error"] == "internal_server_error"
    assert body["message"] == "服务器内部错误"
    assert "secret detail" not in response.text
    assert any("unhandled error on GET /boom" in record.getMessage() for record in caplog.records)
