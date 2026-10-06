"""真实中间件集成测试的支撑工具（纯标准库，不参与生产链路）。

定位：本模块回答「可选的真实中间件集成测试**要不要跑、能不能跑**」两个问题，
让 ``tests/test_integration/`` 下的用例做到**默认离线全绿**：

* **要不要跑**——由环境变量 :data:`ENV_PREFIX`（``MED_MEMORY_IT``）显式开关决定；
  未开启时集成用例以 ``skip`` 收尾，普通 ``pytest`` 依旧只依赖替身中间件；
* **能不能跑**——用 TCP 探针确认 Redis / MySQL / Elasticsearch 是否真的在监听，
  不可达时给出「如何用 docker compose 起服务」的可执行提示，
  而不是让用例抛出一堆连接错误。

设计取舍：

* 只用标准库 ``socket`` / ``urllib.parse`` 做「端口可连」判定，
  **不导入任何中间件客户端**——否则测试支撑模块会反向依赖可选后端，
  缺包时连「跳过」这一步都做不到；
* 只做连接可达性判断，不做任何协议握手与文本内容处理。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import os
import socket
import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

from med_langchain_memory.exceptions import ValidationError

#: 集成测试总开关的环境变量名。
ENV_PREFIX = "MED_MEMORY_IT"

#: 视为「已开启」的取值（大小写不敏感）。
_TRUTHY = frozenset({"1", "true", "yes", "on"})

#: 单次 TCP 探针超时（秒）。
DEFAULT_PROBE_TIMEOUT_SECONDS = 0.5

#: :func:`check_service` 轮询等待服务就绪的默认上限（秒）。
DEFAULT_WAIT_SECONDS = 30.0

#: 轮询间隔（秒）。
DEFAULT_WAIT_INTERVAL_SECONDS = 0.5

#: 环境变量覆盖的连接串省略端口时，按 URL scheme 兜底的服务端口。
_SCHEME_PORTS = {"redis": 6379, "mysql": 3306, "http": 9200, "https": 9200}

#: 启动真实中间件的命令，写进跳过原因里便于直接复制执行。
COMPOSE_HINT = "docker compose -f docker-compose.integration.yml up -d"


@dataclass(frozen=True)
class IntegrationService:
    """一个集成测试依赖的外部服务（名称 + 地址 + 覆盖用环境变量）。"""

    #: 服务名，如 ``redis`` / ``mysql`` / ``elasticsearch``。
    name: str

    #: 生效的连接串（环境变量覆盖后的最终值）。
    url: str

    #: TCP 探针使用的主机。
    host: str

    #: TCP 探针使用的端口。
    port: int

    #: 可覆盖该服务地址的环境变量名，如 ``MED_MEMORY_IT_REDIS_URL``。
    env_var: str

    def describe(self) -> str:
        """返回「服务名 @ host:port（可用环境变量覆盖）」的单行描述。

        Returns:
            形如 ``redis @ localhost:6379 (override via MED_MEMORY_IT_REDIS_URL)`` 的文本。
        """
        return f"{self.name} @ {self.host}:{self.port} (override via {self.env_var})"


@dataclass(frozen=True)
class ServiceStatus:
    """:func:`check_service` 的判定结果。"""

    #: 被检查的服务。
    service: IntegrationService

    #: 集成测试开关是否已打开。
    enabled: bool

    #: 服务端口是否可连。
    reachable: bool

    #: 不可用原因（可直接作为 ``pytest.skip`` 的消息）；可用时为 ``None``。
    reason: str | None

    @property
    def ready(self) -> bool:
        """集成测试是否可以针对该服务真实执行（开关已打开且服务可达）。"""
        return self.enabled and self.reachable


def _split_url(name: str, url: str) -> tuple[str, int]:
    """从连接串中解析出探针用的主机与端口。

    Args:
        name: 服务名，仅用于错误信息。
        url: 连接串，如 ``redis://localhost:6379/15``。

    Returns:
        ``(host, port)``；连接串省略端口时按 scheme 兜底（``redis`` → 6379 等）。

    Raises:
        ValidationError: 连接串缺少主机，或端口非法（非数字 / 越界）时。
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValidationError(f"invalid url for {name}: {url!r} ({exc})") from exc
    if not parts.hostname:
        raise ValidationError(f"invalid url for {name}: {url!r} (missing host)")
    if port is None:
        port = _SCHEME_PORTS.get(parts.scheme.split("+")[0], 0)
    return parts.hostname, port


def _default_service(name: str, url: str) -> IntegrationService:
    """按内置连接串构造默认服务条目。"""
    host, port = _split_url(name, url)
    return IntegrationService(name, url, host, port, f"{ENV_PREFIX}_{name.upper()}_URL")


#: 集成测试依赖的三个服务及其默认地址（与 docker-compose.integration.yml 一一对应）。
#:
#: 宿主端口刻意**避开标准端口**（16379 / 13306 / 19200 而非 6379 / 3306 / 9200）：
#: 开发机上常常已经跑着标准端口的中间件，集成栈用独立端口号隔离，互不干扰。
DEFAULT_SERVICES: tuple[IntegrationService, ...] = (
    _default_service("redis", "redis://localhost:16379/15"),
    _default_service("mysql", "mysql+pymysql://root:med@localhost:13306/med_memory"),
    _default_service("elasticsearch", "http://localhost:19200"),
)

_DEFAULTS_BY_NAME = {service.name: service for service in DEFAULT_SERVICES}


def integration_enabled(env: Mapping[str, str] | None = None) -> bool:
    """判断集成测试总开关是否打开。

    Args:
        env: 环境变量映射；``None`` 时读取 ``os.environ``。

    Returns:
        ``MED_MEMORY_IT`` 取值为 ``1`` / ``true`` / ``yes`` / ``on``（大小写不敏感）时为 ``True``。
    """
    environ = os.environ if env is None else env
    return environ.get(ENV_PREFIX, "").strip().lower() in _TRUTHY


def resolve_service(name: str, env: Mapping[str, str] | None = None) -> IntegrationService:
    """解析服务地址：优先取环境变量覆盖，否则用内置默认值。

    Args:
        name: 服务名，取值 ``redis`` / ``mysql`` / ``elasticsearch``。
        env: 环境变量映射；``None`` 时读取 ``os.environ``。

    Returns:
        解析后的服务条目；覆盖变量为空串时退回内置默认值。

    Raises:
        ValidationError: ``name`` 未知，或覆盖用的连接串缺少主机 / 端口非法时。
    """
    base = _DEFAULTS_BY_NAME.get(name)
    if base is None:
        known = ", ".join(sorted(_DEFAULTS_BY_NAME))
        raise ValidationError(f"unknown integration service {name!r}; known: {known}")
    environ = os.environ if env is None else env
    raw = environ.get(base.env_var, "").strip()
    if not raw:
        return base
    host, port = _split_url(name, raw)
    return IntegrationService(base.name, raw, host, port, base.env_var)


def probe_tcp(host: str, port: int, *, timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS) -> bool:
    """探测 ``host:port`` 是否能建立 TCP 连接。

    Args:
        host: 主机名或 IP。
        port: 端口号；``<= 0`` 直接判定为不可达。
        timeout: 连接超时（秒）。

    Returns:
        能建立连接时为 ``True``，否则为 ``False``（任何 ``OSError`` 都视为不可达）。
    """
    if port <= 0:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_service(
    service: IntegrationService,
    *,
    timeout: float = DEFAULT_WAIT_SECONDS,
    interval: float = DEFAULT_WAIT_INTERVAL_SECONDS,
) -> bool:
    """轮询等待服务开始监听（容器冷启动时使用）。

    Args:
        service: 目标服务。
        timeout: 最长等待秒数；``<= 0`` 时只探测一次。
        interval: 两次探测之间的间隔秒数。

    Returns:
        在 ``timeout`` 内探测到端口可连时为 ``True``。
    """
    deadline = time.monotonic() + max(timeout, 0.0)
    while True:
        if probe_tcp(service.host, service.port):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval)


def check_service(
    service: IntegrationService,
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = DEFAULT_PROBE_TIMEOUT_SECONDS,
    wait_seconds: float | None = None,
) -> ServiceStatus:
    """综合判定服务是否可用于集成测试，并给出可直接展示的跳过原因。

    Args:
        service: 目标服务。
        env: 环境变量映射；``None`` 时读取 ``os.environ``。
        timeout: 单次 TCP 探针超时（秒）。
        wait_seconds: 非 ``None`` 时先按该上限轮询等待服务就绪（容器冷启动场景）。

    Returns:
        :class:`ServiceStatus`；``ready`` 为 ``False`` 时 ``reason`` 一定非空。
    """
    if not integration_enabled(env):
        return ServiceStatus(
            service, False, False, f"{ENV_PREFIX}=1 not set; skipping {service.describe()}"
        )
    reachable = (
        wait_for_service(service, timeout=wait_seconds)
        if wait_seconds is not None
        else probe_tcp(service.host, service.port, timeout=timeout)
    )
    if not reachable:
        return ServiceStatus(
            service,
            True,
            False,
            f"{service.describe()} is not reachable; start it with: {COMPOSE_HINT}",
        )
    return ServiceStatus(service, True, True, None)
