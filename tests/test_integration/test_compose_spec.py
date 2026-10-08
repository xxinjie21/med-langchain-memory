"""``docker-compose.integration.yml`` 的规范测试（纯标准库正则解析）。

编排文件是集成测试的「唯一环境来源」，一旦与
:data:`med_langchain_memory.testing.services.DEFAULT_SERVICES` 的默认地址脱节，
用例就会在起好容器后仍然被跳过。因此这里把「编排里写的」与「代码里默认的」做机器校验：

* 三个核心服务与 Redis Cluster 的 6 个节点 + 1 个初始化容器都必须显式声明；
* 每个常驻服务都必须有健康检查（否则用例会在容器未就绪时误判）；
* 发布端口必须与默认服务地址的端口逐一对应；
* Elasticsearch 必须单节点且关闭安全插件（本地集成测试前提）；
* MySQL 库名必须与默认连接串一致；
* 集群节点必须开集群模式、广播地址可控、总线端口成对映射，
  且初始化容器按 3 主 3 从组装集群。
"""

from __future__ import annotations

import re
from pathlib import Path

from med_langchain_memory.testing import (
    CLUSTER_ANNOUNCE_ENV,
    DEFAULT_SERVICES,
    REDIS_CLUSTER_NODE_PORTS,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.integration.yml"

#: 编排中必须出现的核心服务名。
EXPECTED_SERVICES = ("redis", "mysql", "elasticsearch")

#: Redis Cluster 的 6 个节点服务名（3 主 3 从）。
CLUSTER_NODE_SERVICES = tuple(f"redis-cluster-{index}" for index in range(1, 7))

#: 一次性初始化集群的服务名。
CLUSTER_INIT_SERVICE = "redis-cluster-init"

#: 集群专用网络名。
CLUSTER_NETWORK = "cluster"

#: 集群节点广播地址的默认值（固定子网的网关）。
DEFAULT_CLUSTER_ANNOUNCE_IP = "172.31.240.1"

#: 集群总线端口相对客户端端口的偏移。
CLUSTER_BUS_PORT_OFFSET = 10000

#: 形如 ``image: redis:7-alpine`` 的镜像声明。
IMAGE_PATTERN = re.compile(r"^\s*image:\s*(\S+)\s*$", re.MULTILINE)

#: 形如 ``"16379:6379"`` 的端口映射。
PORT_MAPPING_PATTERN = re.compile(r'"(\d+):(\d+)"')


def read_compose() -> str:
    """读取编排文件全文。

    Raises:
        AssertionError: 文件缺失或为空时。
    """
    assert COMPOSE_PATH.is_file(), f"missing compose file: {COMPOSE_PATH}"
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert text.strip(), "compose file must not be empty"
    return text


def port_mappings() -> list[tuple[int, int]]:
    """解析全部 ``"宿主端口:容器端口"`` 映射为整数二元组列表。"""
    return [
        (int(host), int(container))
        for host, container in PORT_MAPPING_PATTERN.findall(read_compose())
    ]


class TestComposeFile:
    """文件形态与服务声明。"""

    def test_file_exists_and_readable(self) -> None:
        """正向：编排文件存在且内容非空。"""
        assert read_compose().startswith("#")

    def test_no_tab_indentation(self) -> None:
        """边界：YAML 不得使用制表符缩进。"""
        text = read_compose()
        offending = [index + 1 for index, line in enumerate(text.splitlines()) if "\t" in line]
        assert offending == [], f"tab characters found on lines: {offending}"

    def test_declares_all_expected_services(self) -> None:
        """正向：核心三服务与集群 6 节点 + 1 初始化容器都已声明。"""
        text = read_compose()
        declared = [*EXPECTED_SERVICES, *CLUSTER_NODE_SERVICES, CLUSTER_INIT_SERVICE]
        for name in declared:
            assert re.search(rf"^\s{{2}}{name}:\s*$", text, re.MULTILINE), f"service {name} missing"

    def test_images_are_pinned_to_a_tag(self) -> None:
        """边界：镜像必须带显式标签（禁止隐式 latest）。"""
        images = IMAGE_PATTERN.findall(read_compose())
        expected = len(EXPECTED_SERVICES) + len(CLUSTER_NODE_SERVICES) + 1
        assert len(images) == expected
        for image in images:
            assert re.search(r":[\w.-]+$", image), f"image {image} is not pinned to a tag"

    def test_scanner_detects_unpinned_image(self) -> None:
        """边界：镜像标签扫描器本身有效（正对照，防止空扫描假通过）。"""
        assert IMAGE_PATTERN.findall("    image: redis:7-alpine\n") == ["redis:7-alpine"]
        unpinned = IMAGE_PATTERN.findall("    image: redis\n")[0]
        assert re.search(r":[\w.-]+$", unpinned) is None


class TestComposeServiceSettings:
    """各服务的配置要点。"""

    def test_every_long_running_service_has_a_healthcheck(self) -> None:
        """正向：每个常驻服务都声明健康检查（用例靠它判断容器就绪）。

        初始化容器是一次性任务（``restart: "no"``），不参与健康检查。
        """
        expected = len(EXPECTED_SERVICES) + len(CLUSTER_NODE_SERVICES)
        assert read_compose().count("healthcheck:") == expected

    def test_published_ports_match_default_service_ports(self) -> None:
        """正向：宿主发布端口与 ``DEFAULT_SERVICES`` 的默认端口逐一对应。"""
        text = read_compose()
        for service in DEFAULT_SERVICES:
            assert f'"{service.port}:' in text, f"missing host port {service.port} ({service.name})"

    def test_host_ports_avoid_standard_middleware_ports(self) -> None:
        """正向：宿主端口刻意避开标准端口，且容器内仍用标准端口。

        开发机上常常已跑着 6379 / 3306 / 9200 的中间件，直接映射标准端口会让
        ``docker compose up`` 因端口占用失败，因此集成栈用独立宿主端口隔离。
        """
        mappings = port_mappings()
        core = dict(zip((16379, 13306, 19200), (6379, 3306, 9200), strict=True))
        assert core.items() <= set(mappings)
        for host, container in mappings:
            assert host != container, f"host port {host} equals container port {container}"
            assert host not in {6379, 3306, 9200}, f"host port {host} collides with a standard port"

    def test_elasticsearch_is_single_node_without_security(self) -> None:
        """正向：ES 以单节点、关闭安全插件的方式启动（本地集成前提）。"""
        text = read_compose()
        assert "discovery.type: single-node" in text
        assert 'xpack.security.enabled: "false"' in text

    def test_mysql_database_matches_default_url(self) -> None:
        """正向：MySQL 库名与默认连接串中的库名一致。"""
        match = re.search(r"MYSQL_DATABASE:\s*(\S+)", read_compose())
        assert match is not None, "MYSQL_DATABASE is not declared"
        database = match.group(1)
        mysql_service = next(item for item in DEFAULT_SERVICES if item.name == "mysql")
        assert mysql_service.url.endswith(f"/{database}")

    def test_redis_uses_a_dedicated_database_index(self) -> None:
        """正向：Redis 默认地址使用独立 DB（15），避免污染开发者本机数据。"""
        redis_service = next(item for item in DEFAULT_SERVICES if item.name == "redis")
        assert redis_service.url.endswith("/15")


class TestRedisClusterTopology:
    """Redis Cluster 三主三从编排的规范。"""

    def test_declares_six_nodes_and_one_initializer(self) -> None:
        """正向：6 个节点 + 1 个初始化容器都已声明。"""
        text = read_compose()
        for name in CLUSTER_NODE_SERVICES:
            assert re.search(rf"^\s{{2}}{name}:\s*$", text, re.MULTILINE), f"{name} missing"
        assert re.search(rf"^\s{{2}}{CLUSTER_INIT_SERVICE}:\s*$", text, re.MULTILINE)

    def test_every_node_enables_cluster_mode(self) -> None:
        """正向：每个节点都开启集群模式并声明节点超时。"""
        text = read_compose()
        assert text.count("- --cluster-enabled") == len(CLUSTER_NODE_SERVICES)
        assert text.count("- --cluster-node-timeout") == len(CLUSTER_NODE_SERVICES)

    def test_announce_ip_is_configurable(self) -> None:
        """正向：广播地址用 ``MED_MEMORY_IT_CLUSTER_IP`` 控制，默认取固定子网网关。

        Docker Desktop 场景下容器 IP 与子网网关都不可从宿主机访问，
        必须允许外部覆盖；Linux（含 CI runner）用默认值即可。
        """
        text = read_compose()
        expected = "${" + CLUSTER_ANNOUNCE_ENV + ":-" + DEFAULT_CLUSTER_ANNOUNCE_IP + "}"
        assert text.count(expected) == len(CLUSTER_NODE_SERVICES) + 6  # 6 节点 + init 的 6 次引用
        assert "subnet: 172.31.240.0/24" in text, "cluster network must pin its subnet"

    def test_nodes_publish_client_and_bus_ports(self) -> None:
        """正向：每个节点成对发布客户端端口与集群总线端口。"""
        mappings = set(port_mappings())
        for port in REDIS_CLUSTER_NODE_PORTS:
            assert (port, 6379) in mappings, f"missing client mapping for {port}"
            assert (port + CLUSTER_BUS_PORT_OFFSET, 16379) in mappings, (
                f"missing bus mapping {port}"
            )
        assert len(REDIS_CLUSTER_NODE_PORTS) == 6

    def test_initializer_builds_three_masters_three_replicas(self) -> None:
        """正向：初始化容器按 3 主 3 从组装集群，并等待全部节点健康。"""
        text = read_compose()
        assert "--cluster-replicas 1" in text
        assert "--cluster-yes" in text
        assert text.count("condition: service_healthy") == len(CLUSTER_NODE_SERVICES)
        assert 'restart: "no"' in text, "initializer must not restart after it finishes"

    def test_initializer_targets_announced_addresses(self) -> None:
        """正向：初始化连接的是广播地址（与节点 gossip 地址一致），不是服务名。

        广播地址与容器内服务名不一致时，用服务名建集群会让节点间 gossip
        指向不可达地址，集群永远停在 ``cluster_state:fail``。
        """
        text = read_compose()
        initializer = text.split(f"{CLUSTER_INIT_SERVICE}:", 1)[1]
        for port in REDIS_CLUSTER_NODE_PORTS:
            assert f":{port}" in initializer, f"initializer does not target port {port}"

    def test_cluster_nodes_join_the_dedicated_network(self) -> None:
        """正向：集群节点都挂在专用网络上（固定子网保证网关地址稳定）。"""
        text = read_compose()
        assert re.search(rf"^\s{{2}}{CLUSTER_NETWORK}:\s*$", text, re.MULTILINE)
        assert text.count(f"- {CLUSTER_NETWORK}") == len(CLUSTER_NODE_SERVICES) + 1
