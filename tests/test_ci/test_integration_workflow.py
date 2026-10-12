"""Specification tests for the optional real-middleware integration workflow.

These tests parse ``.github/workflows/integration.yml`` with stdlib-only tooling
(regular expressions) and cross-check it against the project metadata:

* the workflow must stay **opt-in** (``workflow_dispatch`` + nightly schedule)
  and must never be merged into the mandatory ``ci.yml`` gate;
* it must run as a **shard matrix** whose entries are exactly the shards the real
  ``med_langchain_memory.testing.shards`` module declares, with ``fail-fast: false``
  so one red shard does not hide the others;
* every shard must start **only its own** containers through
  ``docker-compose.integration.yml`` (``services:`` cannot be sharded per matrix
  entry), and every compose service name it asks for must exist in that file;
* the workflow must not hardcode a single service address or shard name: the
  ``shards`` CLI emits the env vars (``--format env``), the pytest paths
  (``--format paths``), the compose services (``--format compose``) and the
  one-shot initializer (``--format init``), and every one of those invocations
  must be parseable by the real argument parser;
* the preflight guard (``med_langchain_memory.testing.preflight``) must run **before**
  the suite, and its per-shard required list must cover the shard's services, so a
  broken environment fails the run instead of silently skipping every case;
* the Redis Cluster announce IP must be pinned to the compose subnet gateway
  (Linux runners) and every container must be torn down afterwards;
* each shard must emit its own JUnit report, render a Markdown trend summary into
  ``$GITHUB_STEP_SUMMARY`` and roll the report forward through a **per-shard** cache;
* the ``integration`` marker must be registered in ``pyproject.toml``
  (``--strict-markers`` is enabled, so an unregistered marker fails the run);
* every ``uses:`` reference must stay pinned to a major version tag.
"""

from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import pytest

from med_langchain_memory.testing import (
    DEFAULT_SERVICES,
    ENV_PREFIX,
    ENV_REQUIRED,
    render_env,
    render_paths,
    render_required,
    resolve_shard,
    select_shards,
    shard_names,
)
from med_langchain_memory.testing.preflight import build_parser as build_preflight_parser
from med_langchain_memory.testing.shards import build_parser as build_shards_parser
from med_langchain_memory.testing.trend import (
    BASELINE_REPORT_NAME,
    JUNIT_REPORT_NAME,
    TREND_SUMMARY_NAME,
    build_parser,
)

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

#: 渲染趋势摘要的模块入口（与 ``testing/trend.py`` 的 CLI 对齐）。
TREND_MODULE = "med_langchain_memory.testing.trend"

#: 前置守卫的模块入口（与 ``testing/preflight.py`` 的 CLI 对齐）。
PREFLIGHT_MODULE = "med_langchain_memory.testing.preflight"

#: 分片解析的模块入口（与 ``testing/shards.py`` 的 CLI 对齐）。
SHARDS_MODULE = "med_langchain_memory.testing.shards"

#: 工作流里的矩阵表达式（断言前统一替换成字面分片名）。
SHARD_EXPRESSION = "${{ matrix.shard }}"

#: 替换矩阵表达式时使用的字面分片名。
SAMPLE_SHARD = "redis"

#: 每个分片自己的报告目录（矩阵表达式保留，便于断言目录结构）。
SHARD_REPORT_DIR = f"reports/{SHARD_EXPRESSION}"

#: ``python -m <module> ...`` 调用整段（含续行），用于交叉校验 CLI 选项。
TREND_INVOCATION_PATTERN = re.compile(
    rf"python -m {re.escape(TREND_MODULE)}(?P<args>(?:[^\n]*\\\n)*[^\n]*)"
)

#: ``python -m ...shards`` 调用到第一个 shell 元字符为止的参数段。
SHARDS_INVOCATION_PATTERN = re.compile(
    rf"python -m {re.escape(SHARDS_MODULE)}(?P<args>[^)&|>$\n]*)"
)


def read_workflow() -> str:
    """Return the raw text of the integration workflow file.

    Raises:
        AssertionError: If the workflow file does not exist or is empty.
    """
    assert WORKFLOW_PATH.is_file(), f"missing workflow file: {WORKFLOW_PATH}"
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert text.strip(), "workflow file must not be empty"
    return text


def rendered_workflow(shard: str = SAMPLE_SHARD) -> str:
    """Return the workflow text with the matrix expression replaced by a literal shard."""
    return read_workflow().replace(SHARD_EXPRESSION, shard)


def trend_arguments() -> list[str]:
    """Return the CLI tokens passed to the trend module inside the workflow.

    The workflow spreads the invocation over several ``\\``-continued lines and
    quotes the shard-scoped paths, so both the continuations and the quoting are
    normalised away with ``shlex`` before the real parser sees the tokens.
    """
    match = TREND_INVOCATION_PATTERN.search(rendered_workflow())
    assert match is not None, f"{TREND_MODULE} is never invoked by the workflow"
    return shlex.split(match.group("args").replace("\\\n", " "))


def shards_invocations() -> list[str]:
    """Return the argument strings of every ``python -m ...shards`` invocation."""
    matches = SHARDS_INVOCATION_PATTERN.findall(rendered_workflow())
    assert matches, f"{SHARDS_MODULE} is never invoked by the workflow"
    return [match.strip() for match in matches]


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
        """正向：仅手动触发 + 定时，不挂在 push / pull_request 上。"""
        text = read_workflow()
        assert re.search(r"^on:", text, re.MULTILINE)
        assert "workflow_dispatch:" in text
        assert re.search(r'schedule:\s*\n(?:\s*#[^\n]*\n)*\s*-\s*cron:\s*"[^"]+"', text)
        assert "pull_request:" not in text
        assert re.search(r"^\s{2}push:", text, re.MULTILINE) is None

    def test_runs_nightly_every_day(self) -> None:
        """正向：定时任务必须是**每日夜间**跑一次（而不是每周）。"""
        crons = re.findall(r'cron:\s*"([^"]+)"', read_workflow())
        assert crons == ["0 18 * * *"], f"expected a single nightly cron, got {crons}"

    def test_concurrency_cancels_in_progress_runs(self) -> None:
        """正向：并发组避免同一分支重复占资源。"""
        text = read_workflow()
        assert "concurrency:" in text
        assert "cancel-in-progress: true" in text


class TestShardMatrix:
    """分片矩阵：作业必须按真实分片表切分，且一个分片红不掩盖其它分片。"""

    def test_matrix_matches_the_real_shard_table(self) -> None:
        """正向：矩阵取值与 ``shard_names()`` 逐字一致（改一处必须改另一处）。"""
        match = re.search(r"shard:\s*\[(?P<names>[^\]]+)\]", read_workflow())
        assert match is not None, "the shard matrix is missing"
        declared = tuple(name.strip() for name in match.group("names").split(","))
        assert declared == shard_names()

    def test_matrix_does_not_use_fail_fast(self) -> None:
        """边界：一个分片失败不得取消其它分片（四个服务的结论都要拿到）。"""
        text = read_workflow()
        assert "strategy:" in text
        assert "fail-fast: false" in text
        assert "fail-fast: true" not in text

    def test_job_name_includes_the_shard(self) -> None:
        """正向：作业名带分片名，Job Summary 里能分辨是哪个分片。"""
        assert re.search(r"^ {4}name:\s*\S.*matrix\.shard", read_workflow(), re.MULTILINE)


class TestJob:
    def test_job_selects_integration_suite_and_enables_harness(self) -> None:
        """正向：显式选中 integration 用例，并由 shards 模块打开 ``MED_MEMORY_IT`` 开关。"""
        text = read_workflow()
        assert re.search(r"^\s{2}integration:", text, re.MULTILINE)
        assert re.search(r"pytest -m integration\b", text)
        # 开关不再硬编码在工作流里，而是由分片环境步骤生成。
        assert f'{ENV_PREFIX}: "1"' not in text
        assert '--format env >> "$GITHUB_ENV"' in text

    def test_job_does_not_declare_service_containers(self) -> None:
        """边界：不得使用 ``services:``——它无法按矩阵分片，会让每个分片都起全部容器。"""
        text = read_workflow()
        assert re.search(r"^\s{4}services:", text, re.MULTILINE) is None
        assert "--health-cmd" not in text
        assert "docker compose -f docker-compose.integration.yml" in text

    def test_workflow_hardcodes_no_service_address(self) -> None:
        """边界：任何服务地址变量都不得出现在工作流里（单一真值来源在 shards 模块）。"""
        text = read_workflow()
        for service in DEFAULT_SERVICES:
            assert service.env_var not in text, f"{service.env_var} is hardcoded in the workflow"
        assert ENV_REQUIRED not in text

    def test_shards_cli_emits_every_service_address(self) -> None:
        """正向：被移出工作流的地址确实由模块生成（正对照，防止两边都漏）。"""
        assert render_required(select_shards()) == ",".join(
            service.name for service in DEFAULT_SERVICES
        )
        env_text = render_env(select_shards())
        for service in DEFAULT_SERVICES:
            assert f"{service.env_var}=" in env_text

    def test_job_installs_dev_extras(self) -> None:
        """正向：测试依赖走 ``.[dev]``，与 ci.yml 保持一致（含 MySQL 驱动 pymysql）。"""
        text = read_workflow()
        assert 'pip install -e ".[dev]"' in text
        assert "pymysql" in PYPROJECT_PATH.read_text(encoding="utf-8")

    def test_all_shards_cli_invocations_use_real_options(self) -> None:
        """边界：工作流传给 shards CLI 的每个选项都必须能被真实解析器解析通过。"""
        parsed = [
            build_shards_parser().parse_args(shlex.split(args)) for args in shards_invocations()
        ]
        assert {item.format for item in parsed} == {"env", "compose", "init", "paths"}
        for item in parsed:
            assert item.shard == [SAMPLE_SHARD]

    def test_shard_services_are_started_through_compose(self) -> None:
        """正向：只拉起本分片需要的容器，并等健康检查通过。"""
        text = read_workflow()
        assert "up -d --wait" in text
        assert "--format compose" in text
        assert 'run --rm "$initializer"' in text
        assert "--format init" in text

    def test_requested_compose_services_exist_in_the_compose_file(self) -> None:
        """边界：分片表要的每个 compose 服务名都必须在编排文件里真实存在。"""
        compose = COMPOSE_PATH.read_text(encoding="utf-8")
        for name in shard_names():
            for service in resolve_shard(name).compose_services:
                assert re.search(rf"^  {re.escape(service)}:$", compose, re.MULTILINE), (
                    f"{service} is requested by shard {name} but missing from the compose file"
                )

    def test_pytest_step_passes_the_shard_paths(self) -> None:
        """正向：pytest 目标路径由模块给出，与分片表逐字一致。"""
        assert render_paths([resolve_shard(SAMPLE_SHARD)]) == (
            "tests/test_integration/test_redis_live.py"
        )
        assert "--format paths" in read_workflow()


class TestPreflightGuard:
    """前置守卫：声明的服务必须真的可用，禁止「集体跳过」伪装成绿灯。"""

    def test_per_shard_required_list_covers_every_shard(self) -> None:
        """正向：每个分片的必需服务清单就是它自己声明的服务。"""
        for name in shard_names():
            shard = resolve_shard(name)
            assert render_required([shard]) == shard.required
            assert set(shard.required.split(",")) <= {s.name for s in DEFAULT_SERVICES}

    def test_workflow_runs_the_guard_module(self) -> None:
        """正向：守卫以模块路径执行（包根 ``main`` 属于 trend CLI，不可混用）。"""
        assert f"python -m {PREFLIGHT_MODULE}" in read_workflow()

    def test_guard_runs_after_env_and_before_the_suite(self) -> None:
        """边界：守卫必须在环境注入之后、pytest 之前，否则失败信息会被一堆 skip 淹没。"""
        text = read_workflow()
        env_step = text.index("--format env >>")
        guard = text.index(PREFLIGHT_MODULE)
        pytest_step = text.index("pytest -m integration")
        assert env_step < guard < pytest_step, "preflight guard must sit between env and pytest"

    def test_guard_uses_only_documented_cli_options(self) -> None:
        """边界：工作流传给守卫的命令行必须能被真实解析器解析通过。"""
        match = re.search(rf"python -m {PREFLIGHT_MODULE}(?P<args>[^\n]*)", read_workflow())
        assert match is not None
        parsed = build_preflight_parser().parse_args(match.group("args").split())
        assert parsed.services == []


class TestRedisClusterShard:
    """Redis Cluster 由 compose 拉起（``services:`` 只支持单容器，且无法分片）。"""

    def test_cluster_shard_starts_all_six_nodes(self) -> None:
        """正向：集群分片声明 6 个节点，工作流通过 ``--format compose`` 交给 compose。"""
        cluster = resolve_shard("redis-cluster")
        assert len(cluster.compose_services) == 6
        for index in range(1, 7):
            assert f"redis-cluster-{index}" in cluster.compose_services

    def test_cluster_shard_runs_the_one_shot_initializer(self) -> None:
        """正向：初始化容器由 ``--format init`` 给出，工作流不硬编码容器名。"""
        assert resolve_shard("redis-cluster").initializer == "redis-cluster-init"
        assert "--format init" in read_workflow()
        assert "redis-cluster-init" not in read_workflow()

    def test_non_cluster_shards_have_no_initializer(self) -> None:
        """边界：非集群分片不声明初始化容器（``--format init`` 输出空串即跳过）。"""
        for name in shard_names():
            if name != "redis-cluster":
                assert resolve_shard(name).initializer is None

    def test_sets_cluster_announce_ip_for_linux_runner(self) -> None:
        """正向：显式声明广播地址（Linux runner 用固定子网网关）。"""
        text = read_workflow()
        assert "MED_MEMORY_IT_CLUSTER_IP" in text
        assert "172.31.240.1" in text

    def test_tears_the_cluster_down_afterwards(self) -> None:
        """边界：无论成败都要清理容器，避免 runner 上残留 6 个节点。"""
        text = read_workflow()
        assert "down -v" in text
        assert re.search(r"if:\s*always\(\)", text)


class TestTrendReporting:
    """每个分片独立产出可读摘要，并把报告滚动给同一分片的下一次运行做对比。"""

    def test_pytest_emits_junit_report_per_shard(self) -> None:
        """正向：pytest 把结果写进分片自己的 JUnit XML（摘要与缓存的输入）。"""
        assert f'--junit-xml="{SHARD_REPORT_DIR}/{JUNIT_REPORT_NAME}"' in read_workflow()

    def test_renders_summary_into_step_summary(self) -> None:
        """正向：摘要渲染成 Markdown 并追加进 ``$GITHUB_STEP_SUMMARY``。"""
        text = read_workflow()
        assert f'--output "{SHARD_REPORT_DIR}/{TREND_SUMMARY_NAME}"' in text
        assert f'cat "{SHARD_REPORT_DIR}/{TREND_SUMMARY_NAME}" >> "$GITHUB_STEP_SUMMARY"' in text

    def test_summary_step_runs_even_when_tests_fail(self) -> None:
        """边界：用例失败时仍要出摘要（失败信息正是最需要被看到的）。"""
        step = re.search(
            r"- name: Publish trend summary\n(?P<body>.*?)(?=\n\s*- name:|\Z)",
            read_workflow(),
            re.DOTALL,
        )
        assert step is not None, "Publish trend summary step is missing"
        assert re.search(r"if:\s*always\(\)", step.group("body"))

    def test_rolls_report_forward_through_a_per_shard_cache(self) -> None:
        """正向：缓存按分片隔离，恢复出来的报告被改名为基线。"""
        text = read_workflow()
        assert "actions/cache@v4" in text
        assert f"path: {SHARD_REPORT_DIR}" in text
        assert f"key: integration-report-{SHARD_EXPRESSION}-${{{{ github.run_id }}}}" in text
        assert re.search(r"restore-keys:\s*\|", text)
        assert f'mv "{SHARD_REPORT_DIR}/{JUNIT_REPORT_NAME}"' in text
        assert f'"{SHARD_REPORT_DIR}/{BASELINE_REPORT_NAME}"' in text

    def test_uploads_report_as_artifact_per_shard(self) -> None:
        """正向：报告目录按分片作为 artifact 上传，便于事后排查。"""
        text = read_workflow()
        assert "actions/upload-artifact@v4" in text
        assert f"name: integration-report-{SHARD_EXPRESSION}" in text
        assert re.search(rf"path:\s*{re.escape(SHARD_REPORT_DIR)}/\s*$", text, re.MULTILINE)

    def test_trend_cli_flags_exist_in_the_real_parser(self) -> None:
        """边界：工作流传给 trend CLI 的每个长选项都必须真实存在。"""
        known = {option for action in build_parser()._actions for option in action.option_strings}
        flags = [token for token in trend_arguments() if token.startswith("--")]
        assert flags, "workflow must pass options to the trend module"
        unknown = [flag for flag in flags if flag not in known]
        assert unknown == [], f"workflow references unknown trend options: {unknown}"

    def test_trend_arguments_are_parseable(self) -> None:
        """边界：工作流里那段命令行必须能被真实解析器解析通过。"""
        parsed = build_parser().parse_args(trend_arguments())
        assert parsed.report == Path("reports") / SAMPLE_SHARD / JUNIT_REPORT_NAME
        assert parsed.baseline == Path("reports") / SAMPLE_SHARD / BASELINE_REPORT_NAME
        assert parsed.output == Path("reports") / SAMPLE_SHARD / TREND_SUMMARY_NAME


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

    def test_readme_documents_the_preflight_guard(self) -> None:
        """正向：README 必须说明「必需服务不可用即失败」的守卫与其变量名。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert ENV_REQUIRED in readme
        assert PREFLIGHT_MODULE in readme

    def test_readme_documents_the_shard_cli(self) -> None:
        """正向：README 必须说明分片并行与分片解析入口（否则改分片表没人知道）。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert SHARDS_MODULE in readme
        assert "分片" in readme
        for name in shard_names():
            assert name in readme

    @pytest.mark.parametrize("path", [WORKFLOW_PATH, COMPOSE_PATH])
    def test_deliverables_reference_each_other(self, path: Path) -> None:
        """正向：工作流与编排文件互相引用，避免只改一处。"""
        text = path.read_text(encoding="utf-8")
        assert COMPOSE_PATH.name in text or WORKFLOW_PATH.name in text
