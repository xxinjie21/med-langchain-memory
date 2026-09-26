"""请求日志中间件（纯 ASGI 实现）。

每个 HTTP 请求都会：

1. 解析请求追踪 ID —— 优先透传调用方传入的 ``X-Request-ID``（去除首尾空白后非空），
   否则生成一个 ``uuid4().hex``；
2. 把请求 ID 写入 ``scope["state"]["request_id"]``，供异常处理器统一回填到错误响应；
3. 记录一条访问日志（方法 / 路径 / 状态码 / 耗时 / 请求 ID），5xx 记 ``ERROR``，
   其余记 ``INFO``；下游抛异常时同样记 ``ERROR`` 并把异常继续抛出（不吞异常）；
4. 在响应头回写请求 ID 与 ``X-Process-Time-Ms`` 耗时。

设计取舍：采用纯 ASGI 中间件而非 ``BaseHTTPMiddleware``，后者基于 anyio 任务组
转发，会带来上下文变量隔离与流式响应包装的额外副作用；本实现只包一层 ``send``。
非 HTTP 流量（``lifespan`` / ``websocket``）原样透传，不做任何处理。
"""

from __future__ import annotations

import logging
import time
import uuid

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from med_langchain_memory.config import ACCESS_LOGGER_NAME

#: 默认请求 ID 头名。
DEFAULT_REQUEST_ID_HEADER = "X-Request-ID"

#: 响应耗时头名（毫秒，保留三位小数）。
PROCESS_TIME_HEADER = "X-Process-Time-Ms"

#: ``scope["state"]`` 中存放请求 ID 的键。
REQUEST_ID_STATE_KEY = "request_id"


class RequestLoggingMiddleware:
    """注入请求追踪 ID 并记录访问日志的纯 ASGI 中间件。

    Args:
        app: 下游 ASGI 应用。
        header_name: 请求 ID 头名，默认 ``X-Request-ID``。
        logger: 访问日志 logger，默认 ``med_langchain_memory.api.access``。
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        header_name: str = DEFAULT_REQUEST_ID_HEADER,
        logger: logging.Logger | None = None,
    ) -> None:
        self.app = app
        self.header_name = header_name
        self.logger = logger if logger is not None else logging.getLogger(ACCESS_LOGGER_NAME)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI 入口：非 HTTP 流量直接透传，HTTP 流量包裹 ``send`` 以附加响应头。"""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = self._request_id_of(scope)
        scope.setdefault("state", {})[REQUEST_ID_STATE_KEY] = request_id
        started = time.perf_counter()
        status_code = 500

        async def send_with_headers(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = message["status"]
                headers = MutableHeaders(scope=message)
                headers[self.header_name] = request_id
                headers[PROCESS_TIME_HEADER] = f"{self._elapsed_ms(started):.3f}"
            await send(message)

        try:
            await self.app(scope, receive, send_with_headers)
        except Exception:
            self._log(scope, status_code, started, request_id, failed=True)
            raise
        else:
            self._log(scope, status_code, started, request_id)

    def _request_id_of(self, scope: Scope) -> str:
        """解析请求 ID：透传非空入参，否则生成新的 ``uuid4().hex``。"""
        incoming = (Headers(scope=scope).get(self.header_name) or "").strip()
        return incoming or uuid.uuid4().hex

    def _log(
        self,
        scope: Scope,
        status_code: int,
        started: float,
        request_id: str,
        *,
        failed: bool = False,
    ) -> None:
        """输出一条访问日志；``failed`` 为真或状态码 ≥500 时记 ``ERROR``。"""
        message = "%s %s -> %s (%.3f ms) [%s]"
        args = (
            scope.get("method", "-"),
            scope.get("path", "-"),
            status_code,
            self._elapsed_ms(started),
            request_id,
        )
        if failed:
            self.logger.error(message, *args)
        else:
            level = logging.ERROR if status_code >= 500 else logging.INFO
            self.logger.log(level, message, *args)

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        """自 ``started`` 起经过的毫秒数。"""
        return (time.perf_counter() - started) * 1000.0
