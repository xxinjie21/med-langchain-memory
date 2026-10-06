"""``med_langchain_memory.testing.services`` 单元测试。

被测模块是集成测试的「开关 + 探针」，因此这里**不需要任何中间件**：
正向用例用回环地址上的临时监听端口，异常用例用刚关闭的空闲端口，
边界用例覆盖空环境变量、非法 URL、非正端口等分支。

每个公开方法/属性均含正向与边界用例：
``integration_enabled`` / ``resolve_service`` / ``probe_tcp`` /
``wait_for_service`` / ``check_service`` / ``IntegrationService.describe`` /
``ServiceStatus.ready`` / ``DEFAULT_SERVICES``。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.testing import (
    COMPOSE_HINT,
    DEFAULT_SERVICES,
    ENV_PREFIX,
    IntegrationService,
    check_service,
    integration_enabled,
    probe_tcp,
    resolve_service,
    wait_for_service,
)

HOST = "127.0.0.1"


@contextmanager
def listening_port() -> Iterator[int]:
    """在回环地址上临时监听一个随机端口，产出该端口号，退出时关闭。"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((HOST, 0))
    sock.listen(1)
    try:
        yield int(sock.getsockname()[1])
    finally:
        sock.close()


def free_port() -> int:
    """返回一个刚被释放、当前无人监听的端口号。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def make_service(port: int, name: str = "redis") -> IntegrationService:
    """按给定端口构造一个服务条目（连接串形式与内置默认值保持一致）。"""
    return IntegrationService(
        name=name,
        url=f"redis://{HOST}:{port}",
        host=HOST,
        port=port,
        env_var=f"{ENV_PREFIX}_{name.upper()}_URL",
    )


class TestIntegrationEnabled:
    """总开关解析：只认显式的真值。"""

    @pytest.mark.parametrize("raw", ["1", "true", "TRUE", "yes", "On", " 1 "])
    def test_truthy_values_enable(self, raw: str) -> None:
        """正向：常见真值写法（含大小写与空白）均视为开启。"""
        assert integration_enabled({ENV_PREFIX: raw}) is True

    @pytest.mark.parametrize("raw", ["", "0", "no", "off", "2", "  "])
    def test_falsy_values_disable(self, raw: str) -> None:
        """边界：空串、假值与未知取值一律视为关闭。"""
        assert integration_enabled({ENV_PREFIX: raw}) is False

    def test_missing_variable_defaults_to_disabled(self) -> None:
        """边界：环境变量缺失时默认关闭（保证普通 pytest 离线可跑）。"""
        assert integration_enabled({}) is False


class TestResolveService:
    """服务地址解析：环境变量覆盖优先，否则退回内置默认值。"""

    @pytest.mark.parametrize(
        ("name", "url", "port"),
        [
            ("redis", "redis://localhost:16379/15", 16379),
            ("mysql", "mysql+pymysql://root:med@localhost:13306/med_memory", 13306),
            ("elasticsearch", "http://localhost:19200", 19200),
        ],
    )
    def test_defaults_match_documented_addresses(self, name: str, url: str, port: int) -> None:
        """正向：三个服务的内置默认地址与文档 / compose 编排一致。"""
        service = resolve_service(name, {})
        assert (service.url, service.host, service.port) == (url, "localhost", port)
        assert service.env_var == f"{ENV_PREFIX}_{name.upper()}_URL"

    @pytest.mark.parametrize("name", ["redis", "mysql", "elasticsearch"])
    def test_default_ports_avoid_standard_middleware_ports(self, name: str) -> None:
        """边界：默认宿主端口刻意避开标准端口，避免与本机既有中间件抢占。"""
        standard = {"redis": 6379, "mysql": 3306, "elasticsearch": 9200}
        assert resolve_service(name, {}).port != standard[name]

    def test_env_override_replaces_host_and_port(self) -> None:
        """正向：环境变量覆盖后主机、端口、连接串同步更新。"""
        service = resolve_service("redis", {f"{ENV_PREFIX}_REDIS_URL": "redis://10.0.0.7:6390/3"})
        assert (service.url, service.host, service.port) == (
            "redis://10.0.0.7:6390/3",
            "10.0.0.7",
            6390,
        )

    def test_env_override_without_port_falls_back_to_scheme_default(self) -> None:
        """边界：覆盖串省略端口时按 scheme 兜底（redis → 6379）。"""
        service = resolve_service("redis", {f"{ENV_PREFIX}_REDIS_URL": "redis://cache.internal"})
        assert (service.host, service.port) == ("cache.internal", 6379)

    def test_blank_env_value_falls_back_to_default(self) -> None:
        """边界：覆盖变量为空串 / 全空白时退回默认值。"""
        for raw in ("", "   "):
            service = resolve_service("mysql", {f"{ENV_PREFIX}_MYSQL_URL": raw})
            assert service.url == "mysql+pymysql://root:med@localhost:13306/med_memory"

    def test_unknown_service_rejected(self) -> None:
        """异常：未知服务名直接拒绝，并在消息里列出可用服务。"""
        with pytest.raises(ValidationError, match="unknown integration service"):
            resolve_service("kafka", {})

    @pytest.mark.parametrize(
        "url",
        ["redis://localhost:notaport", "redis://localhost:99999999", "redis://", "redis://:6379"],
    )
    def test_invalid_url_rejected(self, url: str) -> None:
        """异常：端口非法 / 缺少主机 / 空主机的覆盖串一律拒绝。"""
        with pytest.raises(ValidationError, match="invalid url for redis"):
            resolve_service("redis", {f"{ENV_PREFIX}_REDIS_URL": url})


class TestProbeTcp:
    """TCP 探针：只看端口能否建立连接。"""

    def test_returns_true_for_listening_port(self) -> None:
        """正向：回环地址上的监听端口判定为可达。"""
        with listening_port() as port:
            assert probe_tcp(HOST, port) is True

    def test_returns_false_for_closed_port(self) -> None:
        """边界：无人监听的端口判定为不可达（连接被拒）。"""
        assert probe_tcp(HOST, free_port(), timeout=0.5) is False

    @pytest.mark.parametrize("port", [0, -1])
    def test_returns_false_for_non_positive_port(self, port: int) -> None:
        """边界：非正端口不发起连接，直接判定不可达。"""
        assert probe_tcp(HOST, port) is False


class TestWaitForService:
    """轮询等待：容器冷启动场景。"""

    def test_returns_true_when_service_listens(self) -> None:
        """正向：服务已在监听时立即返回 ``True``。"""
        with listening_port() as port:
            assert wait_for_service(make_service(port), timeout=5.0) is True

    def test_returns_false_immediately_on_zero_timeout(self) -> None:
        """边界：``timeout<=0`` 时只探测一次即返回 ``False``，不做无谓等待。"""
        assert wait_for_service(make_service(free_port()), timeout=0.0) is False

    def test_retries_until_deadline_then_gives_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """边界：未就绪时按间隔重试（覆盖 sleep 分支），超时后返回 ``False``。

        探针被替换为「恒不可达」，避免依赖真实端口拒绝连接的耗时，
        使「重试一次以上」这条断言确定成立。
        """
        probes: list[int] = []

        def never_ready(host: str, port: int, **kwargs: object) -> bool:
            probes.append(port)
            return False

        monkeypatch.setattr("med_langchain_memory.testing.services.probe_tcp", never_ready)
        assert wait_for_service(make_service(6379), timeout=0.05, interval=0.01) is False
        assert len(probes) > 1


class TestCheckService:
    """综合判定：开关 + 可达性 + 可直接展示的跳过原因。"""

    def test_disabled_reports_switch_reason(self) -> None:
        """边界：开关未打开时不做探测，原因里点明环境变量名。"""
        with listening_port() as port:
            status = check_service(make_service(port), env={})
        assert (status.enabled, status.reachable, status.ready) == (False, False, False)
        assert status.reason is not None
        assert ENV_PREFIX in status.reason
        assert "skipping" in status.reason

    def test_enabled_but_unreachable_hints_compose(self) -> None:
        """异常：开关已开但服务不可达时，原因里给出可复制的启动命令。"""
        status = check_service(make_service(free_port()), env={ENV_PREFIX: "1"}, timeout=0.5)
        assert (status.enabled, status.reachable, status.ready) == (True, False, False)
        assert status.reason is not None
        assert COMPOSE_HINT in status.reason

    def test_enabled_and_reachable_is_ready(self) -> None:
        """正向：开关已开且端口可连时判定就绪，且不产生跳过原因。"""
        with listening_port() as port:
            status = check_service(make_service(port), env={ENV_PREFIX: "1"})
        assert (status.enabled, status.reachable, status.ready) == (True, True, True)
        assert status.reason is None

    def test_wait_seconds_polls_before_probing(self) -> None:
        """正向：给定 ``wait_seconds`` 时走轮询路径，可达即就绪。"""
        with listening_port() as port:
            status = check_service(make_service(port), env={ENV_PREFIX: "1"}, wait_seconds=5.0)
        assert status.ready is True


class TestServiceDescriptors:
    """默认服务表与描述文本。"""

    def test_default_services_cover_three_backends(self) -> None:
        """正向：默认表覆盖 Redis / MySQL / Elasticsearch 三个服务。"""
        assert {service.name for service in DEFAULT_SERVICES} == {"redis", "mysql", "elasticsearch"}

    def test_default_env_var_naming_convention(self) -> None:
        """正向：覆盖变量名统一为 ``MED_MEMORY_IT_<NAME>_URL``。"""
        for service in DEFAULT_SERVICES:
            assert service.env_var == f"{ENV_PREFIX}_{service.name.upper()}_URL"

    def test_describe_mentions_host_port_and_env_var(self) -> None:
        """正向：描述文本同时给出地址与覆盖变量，便于贴进跳过原因。"""
        described = make_service(6379).describe()
        assert f"{HOST}:6379" in described
        assert f"{ENV_PREFIX}_REDIS_URL" in described
