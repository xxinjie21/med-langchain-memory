"""请求日志中间件单元测试（D30）。

覆盖：请求 ID 生成 / 透传 / 空白纠正、响应头回写（请求 ID + 耗时）、访问日志级别
（2xx → INFO、5xx → ERROR、下游异常 → ERROR）、非 HTTP 流量透传，以及中间件属性。
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Mapping

import pytest
from fastapi.testclient import TestClient
from starlette.responses import Response
from starlette.types import Message, Receive, Scope, Send

from med_langchain_memory.api import (
    DEFAULT_REQUEST_ID_HEADER,
    PROCESS_TIME_HEADER,
    RequestLoggingMiddleware,
    create_app,
)
from med_langchain_memory.config import PACKAGE_LOGGER_NAME, MedMemorySettings

_HEX32 = re.compile(r"\A[0-9a-f]{32}\Z")


async def _receive() -> Message:
    """空请求体接收器。"""
    return {"type": "http.request"}


async def _discard(message: Message) -> None:
    """丢弃响应消息的发送器。"""
    return None


def _http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> Scope:
    """构造一个最小 HTTP scope。"""
    return {"type": "http", "method": "GET", "path": "/probe", "headers": headers or []}


def test_generates_request_id_when_header_absent() -> None:
    """未携带请求 ID 时自动生成 32 位十六进制 ID，并回写响应头与耗时头。"""
    response = TestClient(create_app()).get("/health")
    assert _HEX32.match(response.headers[DEFAULT_REQUEST_ID_HEADER])
    assert float(response.headers[PROCESS_TIME_HEADER]) >= 0.0


def test_passes_through_incoming_request_id() -> None:
    """调用方传入的请求 ID 原样透传。"""
    response = TestClient(create_app()).get(
        "/health", headers={DEFAULT_REQUEST_ID_HEADER: "trace-42"}
    )
    assert response.headers[DEFAULT_REQUEST_ID_HEADER] == "trace-42"


def test_blank_request_id_is_regenerated() -> None:
    """纯空白请求 ID（边界）被视为缺失并重新生成。"""
    response = TestClient(create_app()).get("/health", headers={DEFAULT_REQUEST_ID_HEADER: "   "})
    assert _HEX32.match(response.headers[DEFAULT_REQUEST_ID_HEADER])


def test_custom_header_name_is_used() -> None:
    """自定义请求 ID 头名生效，默认头名不再出现。"""
    app = create_app(MedMemorySettings(request_id_header="X-Trace-ID"))
    response = TestClient(app).get("/health", headers={"X-Trace-ID": "abc"})
    assert response.headers["X-Trace-ID"] == "abc"
    assert DEFAULT_REQUEST_ID_HEADER not in response.headers


def test_middleware_exposes_configured_attributes() -> None:
    """中间件公开属性与构造参数一致（含默认值分支）。"""
    logger = logging.getLogger("tests.middleware.attrs")
    custom = RequestLoggingMiddleware(_discard, header_name="X-Trace-ID", logger=logger)
    assert custom.header_name == "X-Trace-ID"
    assert custom.logger is logger

    default = RequestLoggingMiddleware(_discard)
    assert default.header_name == DEFAULT_REQUEST_ID_HEADER
    assert default.logger.name == f"{PACKAGE_LOGGER_NAME}.api.access"


def test_non_http_scope_is_passed_through_untouched() -> None:
    """非 HTTP 流量（边界：lifespan）原样透传，不注入任何状态。"""
    seen: list[object] = []

    async def _downstream(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope.get("state"))

    middleware = RequestLoggingMiddleware(_downstream)
    asyncio.run(middleware({"type": "lifespan"}, _receive, _discard))
    assert seen == [None]


def test_access_log_is_emitted_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """2xx 请求记一条 INFO 访问日志，含方法 / 路径 / 状态码 / 请求 ID。"""
    client = TestClient(create_app())
    with caplog.at_level(logging.INFO, logger=PACKAGE_LOGGER_NAME):
        response = client.get("/health", headers={DEFAULT_REQUEST_ID_HEADER: "trace-log"})
    assert response.status_code == 200
    records = [r for r in caplog.records if "GET /health -> 200" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO
    assert "[trace-log]" in records[0].getMessage()


def test_server_error_response_is_logged_at_error(caplog: pytest.LogCaptureFixture) -> None:
    """5xx 响应（边界）记 ERROR 级别。"""
    app = create_app()

    async def _teapot() -> Response:
        return Response(status_code=500, content="boom")

    app.add_api_route("/teapot", _teapot, methods=["GET"])
    client = TestClient(app, raise_server_exceptions=False)
    with caplog.at_level(logging.INFO, logger=PACKAGE_LOGGER_NAME):
        response = client.get("/teapot")
    assert response.status_code == 500
    records = [r for r in caplog.records if "GET /teapot -> 500" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR


def test_downstream_exception_is_logged_and_reraised(caplog: pytest.LogCaptureFixture) -> None:
    """下游异常记 ERROR（状态码按 500 兜底）后原样抛出，不吞异常。"""

    async def _boom(scope: Scope, receive: Receive, send: Send) -> None:
        raise RuntimeError("downstream exploded")

    logger_name = "tests.middleware.boom"
    middleware = RequestLoggingMiddleware(_boom, logger=logging.getLogger(logger_name))
    with (
        caplog.at_level(logging.INFO, logger=logger_name),
        pytest.raises(RuntimeError, match="downstream exploded"),
    ):
        asyncio.run(middleware(_http_scope(), _receive, _discard))
    records = [r for r in caplog.records if "GET /probe -> 500" in r.getMessage()]
    assert len(records) == 1
    assert records[0].levelno == logging.ERROR


def test_response_body_messages_are_forwarded_unchanged() -> None:
    """非 ``http.response.start`` 消息（边界）不加头、直接转发。"""
    sent: list[Mapping[str, object]] = []

    async def _downstream(scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    async def _capture(message: Message) -> None:
        sent.append(dict(message))

    middleware = RequestLoggingMiddleware(_downstream)
    asyncio.run(middleware(_http_scope(), _receive, _capture))
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"]
    assert sent[1] == {"type": "http.response.body", "body": b""}
