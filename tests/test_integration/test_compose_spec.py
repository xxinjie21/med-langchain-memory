"""``docker-compose.integration.yml`` 的规范测试（纯标准库正则解析）。

编排文件是集成测试的「唯一环境来源」，一旦与
:data:`med_langchain_memory.testing.services.DEFAULT_SERVICES` 的默认地址脱节，
用例就会在起好容器后仍然被跳过。因此这里把「编排里写的」与「代码里默认的」做机器校验：

* 三个服务与镜像标签必须显式声明；
* 每个服务都必须有健康检查（否则用例会在容器未就绪时误判）；
* 发布端口必须与默认服务地址的端口逐一对应；
* Elasticsearch 必须单节点且关闭安全插件（本地集成测试前提）；
* MySQL 库名必须与默认连接串一致。
"""

from __future__ import annotations

import re
from pathlib import Path

from med_langchain_memory.testing import DEFAULT_SERVICES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.integration.yml"

#: 编排中必须出现的服务名。
EXPECTED_SERVICES = ("redis", "mysql", "elasticsearch")

#: 形如 ``image: redis:7-alpine`` 的镜像声明。
IMAGE_PATTERN = re.compile(r"^\s*image:\s*(\S+)\s*$", re.MULTILINE)


def read_compose() -> str:
    """读取编排文件全文。

    Raises:
        AssertionError: 文件缺失或为空时。
    """
    assert COMPOSE_PATH.is_file(), f"missing compose file: {COMPOSE_PATH}"
    text = COMPOSE_PATH.read_text(encoding="utf-8")
    assert text.strip(), "compose file must not be empty"
    return text


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
        """正向：Redis / MySQL / Elasticsearch 三个服务都已声明。"""
        text = read_compose()
        for name in EXPECTED_SERVICES:
            assert re.search(rf"^\s{{2}}{name}:\s*$", text, re.MULTILINE), f"service {name} missing"

    def test_images_are_pinned_to_a_tag(self) -> None:
        """边界：镜像必须带显式标签（禁止隐式 latest）。"""
        images = IMAGE_PATTERN.findall(read_compose())
        assert len(images) == len(EXPECTED_SERVICES)
        for image in images:
            assert re.search(r":[\w.-]+$", image), f"image {image} is not pinned to a tag"

    def test_scanner_detects_unpinned_image(self) -> None:
        """边界：镜像标签扫描器本身有效（正对照，防止空扫描假通过）。"""
        assert IMAGE_PATTERN.findall("    image: redis:7-alpine\n") == ["redis:7-alpine"]
        unpinned = IMAGE_PATTERN.findall("    image: redis\n")[0]
        assert re.search(r":[\w.-]+$", unpinned) is None


class TestComposeServiceSettings:
    """各服务的配置要点。"""

    def test_every_service_has_a_healthcheck(self) -> None:
        """正向：每个服务都声明健康检查（用例靠它判断容器就绪）。"""
        assert read_compose().count("healthcheck:") == len(EXPECTED_SERVICES)

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
        mappings = re.findall(r'"(\d+):(\d+)"', read_compose())
        assert len(mappings) == len(EXPECTED_SERVICES)
        host_ports = [int(host) for host, _ in mappings]
        container_ports = [int(container) for _, container in mappings]
        assert host_ports == [16379, 13306, 19200]
        assert container_ports == [6379, 3306, 9200]
        assert all(host != container for host, container in mappings)

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
