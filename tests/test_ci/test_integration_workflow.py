"""Specification tests for the optional real-middleware integration workflow.

These tests parse ``.github/workflows/integration.yml`` with stdlib-only tooling
(regular expressions) and cross-check it against the project metadata:

* the workflow must stay **opt-in** (``workflow_dispatch`` + weekly schedule)
  and must never be merged into the mandatory ``ci.yml`` gate;
* it must select the integration suite explicitly (``-m integration``) and
  switch the harness on via ``MED_MEMORY_IT=1``;
* the Redis Cluster must be started through ``docker-compose.integration.yml``
  (GitHub Actions ``services:`` cannot express a 6-node cluster), with the
  announce IP pinned to the compose subnet gateway and a teardown step;
* the ``integration`` marker must be registered in ``pyproject.toml``
  (``--strict-markers`` is enabled, so an unregistered marker fails the run);
* every ``uses:`` reference must stay pinned to a major version tag.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from med_langchain_memory.testing import DEFAULT_SERVICES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "integration.yml"
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
README_PATH = PROJECT_ROOT / "README.md"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.integration.yml"

#: 形如 ``uses: actions/checkout@v4`` 的动作引用。
USES_PATTERN = re.compile(r"uses:\s*(\S+)")

#: 允许的动作引用形态：``owner/repo@v<major>``。
PINNED_ACTION_PATTERN = re.compile(r"[\w.-]+/[\w.-]+@v\d+$")


def read_workflow() -> str:
    """Return the raw text of the integration workflow file.

    Raises:
        AssertionError: If the workflow file does not exist or is empty.
    """
    assert WORKFLOW_PATH.is_file(), f"missing workflow file: {WORKFLOW_PATH}"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert text.strip(), "workflow file must not be empty"
    return text


class TestWorkflowFile:
    def test_workflow_exists_and_readable(self) -> None:
        """正向：工作流文件存在且以 ``name:`` 开头。"""
        assert read_workflow().startswith("name:")

    def test_no_tab_indentation(self) -> None:
        """边界：YAML 不得使用制表符缩进。"""
        offending = [
            index + 1 for index, line in enumerate(read_workflow().splitlines()) if "\t" in line
        ]
        assert offending == [], f"tab characters found on lines: {offending}"


class TestTriggers:
    def test_is_opt_in_manual_and_scheduled(self) -> None:
        """正向：仅手动触发 + 每周定时，不挂在 push / pull_request 上。"""
        text = read_workflow()
        assert re.search(r"^on:", text, re.MULTILINE)
        assert "workflow_dispatch:" in text
        assert re.search(r'schedule:\s*\n\s*-\s*cron:\s*"[^"]+"', text)
        assert "pull_request:" not in text
        assert re.search(r"^\s{2}push:", text, re.MULTILINE) is None

    def test_concurrency_cancels_in_progress_runs(self) -> None:
        """正向：并发组避免同一分支重复占资源。"""
        text = read_workflow()
        assert "concurrency:" in text
        assert "cancel-in-progress: true" in text


class TestJob:
    def test_job_selects_integration_suite_and_enables_harness(self) -> None:
        """正向：显式选中 integration 用例并打开 ``MED_MEMORY_IT`` 开关。"""
        text = read_workflow()
        assert re.search(r"^\s{2}integration:", text, re.MULTILINE)
        assert re.search(r"run:\s*pytest -m integration\s*$", text, re.MULTILINE)
        assert 'MED_MEMORY_IT: "1"' in text

    def test_job_declares_service_containers(self) -> None:
        """正向：Redis 与 Elasticsearch 服务容器带健康检查启动。"""
        text = read_workflow()
        assert re.search(r"^\s+redis:\s*$", text, re.MULTILINE)
        assert re.search(r"^\s+elasticsearch:\s*$", text, re.MULTILINE)
        assert text.count("--health-cmd") == 2

    def test_job_installs_dev_extras(self) -> None:
        """正向：测试依赖走 ``.[dev]``，与 ci.yml 保持一致。"""
        assert 'pip install -e ".[dev]"' in read_workflow()


class TestRedisClusterJob:
    """Redis Cluster 必须由 compose 拉起（``services:`` 只支持单容器）。"""

    def test_starts_all_six_cluster_nodes_via_compose(self) -> None:
        """正向：6 个集群节点全部通过编排文件启动。"""
        text = read_workflow()
        assert COMPOSE_PATH.name in text
        for index in range(1, 7):
            assert f"redis-cluster-{index}" in text, f"redis-cluster-{index} is not started"

    def test_runs_the_one_shot_initializer(self) -> None:
        """正向：初始化容器以 ``run --rm`` 方式执行（一次性组集群）。"""
        text = read_workflow()
        assert re.search(r"run --rm redis-cluster-init", text)

    def test_sets_cluster_announce_ip_for_linux_runner(self) -> None:
        """正向：显式声明广播地址（Linux runner 用固定子网网关）。"""
        text = read_workflow()
        assert "MED_MEMORY_IT_CLUSTER_IP" in text
        assert "172.31.240.1" in text

    def test_passes_cluster_url_to_tests(self) -> None:
        """正向：用例侧拿到集群种子地址（与 ``DEFAULT_SERVICES`` 一致）。"""
        text = read_workflow()
        cluster = next(service for service in DEFAULT_SERVICES if service.name == "redis-cluster")
        assert f"MED_MEMORY_IT_REDIS_CLUSTER_URL: {cluster.url}" in text

    def test_tears_the_cluster_down_afterwards(self) -> None:
        """边界：无论成败都要清理容器，避免 runner 上残留 6 个节点。"""
        text = read_workflow()
        assert "down -v" in text
        assert re.search(r"if:\s*always\(\)", text)


class TestActionPinning:
    def test_all_actions_pin_a_major_version(self) -> None:
        """边界：所有 ``uses:`` 引用都必须至少锁定大版本。"""
        uses = USES_PATTERN.findall(read_workflow())
        assert uses, "workflow must reference at least one action"
        unpinned = [ref for ref in uses if PINNED_ACTION_PATTERN.search(ref) is None]
        assert unpinned == [], f"unpinned action references: {unpinned}"

    def test_pinning_scanner_rejects_floating_refs(self) -> None:
        """边界：锁版本扫描器本身有效（正对照，防止空扫描假通过）。"""
        assert PINNED_ACTION_PATTERN.search("actions/checkout@v4") is not None
        assert PINNED_ACTION_PATTERN.search("actions/checkout@main") is None


class TestMarkerRegistration:
    def test_integration_marker_registered_in_pyproject(self) -> None:
        """正向：``integration`` 标记已注册（``--strict-markers`` 下必需）。"""
        config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        markers = config["tool"]["pytest"]["ini_options"]["markers"]
        assert any(marker.startswith("integration:") for marker in markers)

    def test_strict_markers_still_enabled(self) -> None:
        """边界：严格标记模式不得被关掉，否则拼错的标记会静默通过。"""
        config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
        assert "--strict-markers" in addopts

    def test_default_addopts_do_not_deselect_integration(self) -> None:
        """边界：默认 ``addopts`` 不得内置 ``-m``，否则手工 ``-m integration`` 会互相覆盖。"""
        config = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
        addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
        assert " -m " not in f" {addopts} "


class TestDocumentation:
    def test_readme_documents_opt_in_workflow(self) -> None:
        """正向：README 必须给出 compose 启动、开关变量与标记筛选三要素。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert COMPOSE_PATH.name in readme
        assert "MED_MEMORY_IT" in readme
        assert "-m integration" in readme

    def test_readme_documents_cluster_announce_override(self) -> None:
        """正向：README 必须说明 Docker Desktop 下要覆盖集群广播地址。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert "MED_MEMORY_IT_CLUSTER_IP" in readme
        assert "redis-cluster" in readme

    @pytest.mark.parametrize("path", [WORKFLOW_PATH, COMPOSE_PATH])
    def test_deliverables_reference_each_other(self, path: Path) -> None:
        """正向：工作流与编排文件互相引用，避免只改一处。"""
        text = path.read_text(encoding="utf-8")
        assert COMPOSE_PATH.name in text or WORKFLOW_PATH.name in text
