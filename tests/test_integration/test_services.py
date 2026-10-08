"""``med_langchain_memory.testing.services`` 单元测试。

被测模块是集成测试的「开关 + 探针」，因此这里**不需要任何中间件**：
正向用例用回环地址上的临时监听端口，异常用例用刚关闭的空闲端口，
边界用例覆盖空环境变量、非法 URL、非正端口等分支。

每个公开方法/属性均含正向与边界用例：
``integration_enabled`` / ``resolve_service`` / ``probe_tcp`` /
``wait_for_service`` / ``check_service`` / ``cluster_startup_nodes`` /
``detect_host_address`` / ``IntegrationService.describe`` /
``ServiceStatus.ready`` / ``DEFAULT_SERVICES``。
"""

from __future__ import annotations

import socket
from collections.abc import Iterator
from contextlib import contextmanager

import pytest

from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.testing import (
    CLUSTER_ANNOUNCE_ENV,
    COMPOSE_HINT,
    DEFAULT_SERVICES,
    ENV_PREFIX,
    REDIS_CLUSTER_NODE_PORTS,
    IntegrationService,
    check_service,
    cluster_startup_nodes,
    detect_host_address,
    integration_enabled,
    probe_tcp,
    resolve_service,
    wait_for_service,
)

HOST = "127.0.0.1"


def expected_env_var(name: str) -> str:
    """服务名 → 覆盖用环境变量名（连字符转下划线）。"""
    return f"{ENV_PREFIX}_{name.upper().replace('-', '_')}_URL"


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
            ("redis-cluster", "redis://localhost:17001/0", 17001),
        ],
    )
    def test_defaults_match_documented_addresses(self, name: str, url: str, port: int) -> None:
        """正向：各服务的内置默认地址与文档 / compose 编排一致。"""
        service = resolve_service(name, {})
        assert (service.url, service.host, service.port) == (url, "localhost", port)
        assert service.env_var == expected_env_var(name)

    @pytest.mark.parametrize("name", ["redis", "mysql", "elasticsearch", "redis-cluster"])
    def test_default_ports_avoid_standard_middleware_ports(self, name: str) -> None:
        """边界：默认宿主端口刻意避开标准端口，避免与本机既有中间件抢占。"""
        standard = {"redis": 6379, "mysql": 3306, "elasticsearch": 9200, "redis-cluster": 6379}
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

    def test_default_services_cover_all_backends(self) -> None:
        """正向：默认表覆盖 Redis / MySQL / Elasticsearch / Redis Cluster 四个服务。"""
        assert {service.name for service in DEFAULT_SERVICES} == {
            "redis",
            "mysql",
            "elasticsearch",
            "redis-cluster",
        }

    def test_default_env_var_naming_convention(self) -> None:
        """正向：覆盖变量名统一为 ``MED_MEMORY_IT_<NAME>_URL``（连字符转下划线）。"""
        for service in DEFAULT_SERVICES:
            assert service.env_var == expected_env_var(service.name)

    def test_cluster_env_var_has_no_hyphen(self) -> None:
        """边界：``redis-cluster`` 的覆盖变量名必须是合法环境变量名。"""
        cluster = resolve_service("redis-cluster", {})
        assert cluster.env_var == "MED_MEMORY_IT_REDIS_CLUSTER_URL"
        assert "-" not in cluster.env_var

    def test_describe_mentions_host_port_and_env_var(self) -> None:
        """正向：描述文本同时给出地址与覆盖变量，便于贴进跳过原因。"""
        described = make_service(6379).describe()
        assert f"{HOST}:6379" in described
        assert f"{ENV_PREFIX}_REDIS_URL" in described


class TestClusterStartupNodes:
    """集群启动节点展开。"""

    def test_expands_all_six_nodes_for_default_seed(self) -> None:
        """正向：默认种子端口展开为 6 个节点，主机取服务地址。"""
        nodes = cluster_startup_nodes(resolve_service("redis-cluster", {}))
        assert nodes == [{"host": "localhost", "port": port} for port in REDIS_CLUSTER_NODE_PORTS]
        assert len(nodes) == 6

    def test_env_override_keeps_host_and_expands_ports(self) -> None:
        """正向：只覆盖主机时端口仍按内置 6 节点展开。"""
        service = resolve_service(
            "redis-cluster", {f"{ENV_PREFIX}_REDIS_CLUSTER_URL": "redis://10.1.2.3:17001"}
        )
        nodes = cluster_startup_nodes(service)
        assert {node["host"] for node in nodes} == {"10.1.2.3"}
        assert [node["port"] for node in nodes] == list(REDIS_CLUSTER_NODE_PORTS)

    def test_custom_seed_port_falls_back_to_single_node(self) -> None:
        """边界：对接自建集群（非内置端口）时退回单种子节点，由客户端自行发现拓扑。"""
        service = resolve_service(
            "redis-cluster", {f"{ENV_PREFIX}_REDIS_CLUSTER_URL": "redis://cluster.local:7000"}
        )
        assert cluster_startup_nodes(service) == [{"host": "cluster.local", "port": 7000}]

    def test_rejects_non_cluster_service(self) -> None:
        """异常：传入非集群服务时直接拒绝，避免静默给出错误节点表。"""
        with pytest.raises(ValidationError, match="expects the 'redis-cluster' service"):
            cluster_startup_nodes(resolve_service("redis", {}))

    def test_node_ports_are_unique_and_off_standard_range(self) -> None:
        """边界：6 个节点端口互不重复，且都不是标准 Redis 端口。"""
        assert len(set(REDIS_CLUSTER_NODE_PORTS)) == 6
        assert 6379 not in REDIS_CLUSTER_NODE_PORTS
        assert tuple(sorted(REDIS_CLUSTER_NODE_PORTS)) == REDIS_CLUSTER_NODE_PORTS


class TestDetectHostAddress:
    """宿主机地址探测（Docker Desktop 场景下的集群广播地址）。"""

    def test_returns_dotted_quad_or_none(self) -> None:
        """正向：有默认路由时返回点分四段 IPv4 地址。"""
        address = detect_host_address()
        if address is None:
            pytest.skip("no default route in this environment")
        octets = address.split(".")
        assert len(octets) == 4
        assert all(octet.isdigit() for octet in octets)

    def test_returns_none_when_routing_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """异常：底层 socket 报错时返回 ``None``，不向上抛异常。"""
        import socket as socket_module

        class _FailingSocket:
            def settimeout(self, timeout: float) -> None:
                pass

            def connect(self, address: tuple[str, int]) -> None:
                raise OSError("network is unreachable")

            def close(self) -> None:
                pass

        monkeypatch.setattr(socket_module, "socket", lambda *args, **kwargs: _FailingSocket())
        assert detect_host_address() is None

    def test_announce_env_name_is_documented(self) -> None:
        """边界：广播地址环境变量名符合 ``MED_MEMORY_IT_*`` 前缀约定。"""
        assert CLUSTER_ANNOUNCE_ENV == "MED_MEMORY_IT_CLUSTER_IP"
        assert CLUSTER_ANNOUNCE_ENV.startswith(ENV_PREFIX)
