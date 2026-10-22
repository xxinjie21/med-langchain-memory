"""Specification tests for the optional real-middleware integration workflow.

These tests parse ``.github/workflows/integration.yml`` with stdlib-only tooling
(regular expressions) and cross-check it against the project metadata:

* the workflow must stay **opt-in** (``workflow_dispatch`` + nightly schedule)
  and must never be merged into the mandatory ``ci.yml`` gate;
* it must run as a **shard matrix** whose entries are exactly the shards the real
  ``med_langchain_memory.testing.shards`` module declares, with ``fail-fast: false``
  so one red shard does not hide the others; the matrix itself (shard names plus the
  per-shard timeout budgets) must be rendered by that module inside a ``plan`` job
  and consumed by the matrix job through ``fromJSON``;
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
* each shard must emit its own JUnit report and upload it as an artifact; since
  D47 the per-shard trend summary and the per-shard baseline cache are gone —
  the ``aggregate`` job renders both the ledger comparison and the shard-level
  trend (new failures / fixed per shard) from the **single** rolling ledger
  baseline, so the whole nightly run keeps exactly one cached baseline;
* the ``aggregate`` job must depend on the matrix job, run ``if: always()``, download
  every shard report through the real ``ARTIFACT_PREFIX``, aggregate with the real CLI
  options and publish the combined summary even when the aggregation step itself fails;
* the ``aggregate`` job must also roll a **ledger baseline** forward (``actions/cache``
  keyed by run id, restored before aggregation) and pass ``--baseline-dir`` /
  ``--snapshot-dir`` so the combined summary can be diffed against the previous run;
* since D48 the ``aggregate`` job additionally uploads a **ledger artifact**
  (this run's shard reports plus the restored previous baseline) **before**
  aggregating — the snapshot step overwrites ``baseline/`` with the current run,
  so uploading afterwards would leave the dashboard comparing a run against itself;
* the ``dashboard`` job must depend on ``aggregate``, run ``if: always()``, install
  runtime dependencies only, start no containers, render the page with the real
  dashboard CLI options and publish it to GitHub Pages through a version-pinned
  action, with ``contents: write`` scoped to that job alone;
* since D49 the ``aggregate`` job additionally posts the **same verdict** as a
  GitHub commit status (``integration/nightly``) through the real notifier CLI,
  with ``statuses: write`` scoped to that job alone, the token passed through the
  environment rather than the command line, and ``continue-on-error`` so a failed
  badge update never turns a green run red;
* the ``integration`` marker must be registered in ``pyproject.toml``
  (``--strict-markers`` is enabled, so an unregistered marker fails the run);
* every ``uses:`` reference must stay pinned to a major version tag.
"""

from __future__ import annotations

import importlib
import json
import re
import shlex
import tomllib
from pathlib import Path

import pytest

from med_langchain_memory.testing import (
    CLUSTER_INITIALIZER,
    DEFAULT_SERVICES,
    ENV_PREFIX,
    ENV_REQUIRED,
    MATRIX_SHARD_KEY,
    MATRIX_TIMEOUT_KEY,
    SHARDS,
    render_env,
    render_matrix,
    render_paths,
    render_required,
    resolve_shard,
    select_shards,
    shard_names,
)
from med_langchain_memory.testing.aggregate import (
    AGGREGATE_SUMMARY_NAME,
    ARTIFACT_PREFIX,
    BASELINE_LEDGER_DIR,
    LEDGER_ARTIFACT_NAME,
    collect_shard_reports,
)
from med_langchain_memory.testing.aggregate import (
    build_parser as build_aggregate_parser,
)
from med_langchain_memory.testing.dashboard import (
    DASHBOARD_INDEX_NAME,
    DEFAULT_SITE_DIR,
)
from med_langchain_memory.testing.dashboard import (
    build_parser as build_dashboard_parser,
)
from med_langchain_memory.testing.notifier import STATUS_CONTEXT
from med_langchain_memory.testing.notifier import (
    build_parser as build_notifier_parser,
)
from med_langchain_memory.testing.preflight import build_parser as build_preflight_parser
from med_langchain_memory.testing.shards import build_parser as build_shards_parser
from med_langchain_memory.testing.trend import JUNIT_REPORT_NAME

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "integration.yml"
CI_WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT_PATH = PROJECT_ROOT / "pyproject.toml"
README_PATH = PROJECT_ROOT / "README.md"
COMPOSE_PATH = PROJECT_ROOT / "docker-compose.integration.yml"

#: 形如 ``uses: actions/checkout@v4`` 的动作引用（锚定行首，避免误匹配
#: ``permissions:`` 里的 ``statuses: write`` 这类同尾字串）。
USES_PATTERN = re.compile(r"^\s*uses:\s*(\S+)", re.MULTILINE)

#: 允许的动作引用形态：``owner/repo@v<major>``。
PINNED_ACTION_PATTERN = re.compile(r"[\w.-]+/[\w.-]+@v\d+$")

#: 分片趋势渲染模块（D47 起工作流不再调用它；本地单报告场景仍可用）。
TREND_MODULE = "med_langchain_memory.testing.trend"

#: 前置守卫的模块入口（与 ``testing/preflight.py`` 的 CLI 对齐）。
PREFLIGHT_MODULE = "med_langchain_memory.testing.preflight"

#: 分片解析的模块入口（与 ``testing/shards.py`` 的 CLI 对齐）。
SHARDS_MODULE = "med_langchain_memory.testing.shards"

#: 分片结果聚合的模块入口（与 ``testing/aggregate.py`` 的 CLI 对齐）。
AGGREGATE_MODULE = "med_langchain_memory.testing.aggregate"

#: 夜间看板渲染的模块入口（与 ``testing/dashboard.py`` 的 CLI 对齐）。
DASHBOARD_MODULE = "med_langchain_memory.testing.dashboard"

#: 状态发布（D49）的模块入口（与 ``testing/notifier.py`` 的 CLI 对齐）。
NOTIFIER_MODULE = "med_langchain_memory.testing.notifier"

#: 发布 commit status 的步骤名（工作流里这一步的名字）。
NOTIFIER_STEP_NAME = "Post nightly integration status"

#: GitHub Pages 发布动作（D48 看板作业使用，必须锁大版本）。
PAGES_ACTION = "peaceiris/actions-gh-pages"

#: 看板作业下载账本 artifact 的目标目录。
LEDGER_DOWNLOAD_DIR = "ledger"

#: 看板 CLI 的调用整段（工作流里同样用 ``\`` 续行）。
DASHBOARD_INVOCATION_PATTERN = re.compile(
    rf"python -m {re.escape(DASHBOARD_MODULE)}(?P<args>(?:[^\n]*\\\n)*[^\n]*)"
)

#: notifier CLI 的调用整段（工作流里同样用 ``\`` 续行）。
NOTIFIER_INVOCATION_PATTERN = re.compile(
    rf"python -m {re.escape(NOTIFIER_MODULE)}(?P<args>(?:[^\n]*\\\n)*[^\n]*)"
)

#: shell 命令替换（如 ``$(date -u ...)``）：含空格，交给 shlex 前先替换成占位符。
COMMAND_SUBSTITUTION_PATTERN = re.compile(r"\$\([^)]*\)")

#: 工作流里的矩阵表达式（断言前统一替换成字面分片名）。
SHARD_EXPRESSION = "${{ matrix.shard }}"

#: 替换矩阵表达式时使用的字面分片名。
SAMPLE_SHARD = "redis"

#: 每个分片自己的报告目录（矩阵表达式保留，便于断言目录结构）。
SHARD_REPORT_DIR = f"reports/{SHARD_EXPRESSION}"

#: 汇总作业下载 artifact 的目标目录（聚合模块的 ``--reports-dir``）。
AGGREGATE_REPORTS_DIR = "reports"

#: 汇总作业滚动总账基线的缓存键前缀（与工作流里的 ``key:`` / ``restore-keys:`` 一致）。
BASELINE_CACHE_PREFIX = "integration-baseline-"

#: 聚合 CLI 的调用整段（工作流里同样用 ``\`` 续行）。
AGGREGATE_INVOCATION_PATTERN = re.compile(
    rf"python -m {re.escape(AGGREGATE_MODULE)}(?P<args>(?:[^\n]*\\\n)*[^\n]*)"
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


def shards_invocations() -> list[str]:
    """Return the argument strings of every ``python -m ...shards`` invocation."""
    matches = SHARDS_INVOCATION_PATTERN.findall(rendered_workflow())
    assert matches, f"{SHARDS_MODULE} is never invoked by the workflow"
    return [match.strip() for match in matches]


def aggregate_arguments() -> list[str]:
    """Return the CLI tokens passed to the aggregate module inside the workflow."""
    match = AGGREGATE_INVOCATION_PATTERN.search(read_workflow())
    assert match is not None, f"{AGGREGATE_MODULE} is never invoked by the workflow"
    return shlex.split(match.group("args").replace("\\\n", " "))


def dashboard_arguments() -> list[str]:
    """Return the CLI tokens passed to the dashboard module inside the workflow."""
    match = DASHBOARD_INVOCATION_PATTERN.search(read_workflow())
    assert match is not None, f"{DASHBOARD_MODULE} is never invoked by the workflow"
    raw = match.group("args").replace("\\\n", " ")
    # ``$(date ...)`` 这类命令替换含空格，先换成占位符再交给 shlex 与真实解析器。
    return shlex.split(COMMAND_SUBSTITUTION_PATTERN.sub("PLACEHOLDER", raw))


def notifier_arguments() -> list[str]:
    """Return the CLI tokens passed to the notifier module inside the workflow."""
    match = NOTIFIER_INVOCATION_PATTERN.search(read_workflow())
    assert match is not None, f"{NOTIFIER_MODULE} is never invoked by the workflow"
    raw = match.group("args").replace("\\\n", " ")
    return shlex.split(COMMAND_SUBSTITUTION_PATTERN.sub("PLACEHOLDER", raw))


def step_body(name: str) -> str:
    """Return the YAML body of a workflow step (truncated at the next step)."""
    match = re.search(
        rf"- name: {re.escape(name)}\n(?P<body>.*?)(?=\n\s*- name:|\Z)",
        read_workflow(),
        re.DOTALL,
    )
    assert match is not None, f"step {name!r} is missing from the workflow"
    return match.group("body")


def aggregate_job_body() -> str:
    """Return the YAML body of the ``aggregate`` job (truncated at the next job)."""
    match = re.search(
        r"^ {2}aggregate:\n(?P<body>.*?)(?=\n {2}\S|\Z)", read_workflow(), re.MULTILINE | re.DOTALL
    )
    assert match is not None, "the aggregate job is missing"
    return match.group("body")


def job_body(job: str) -> str:
    """Return the YAML body of a top-level job (truncated at the next job)."""
    match = re.search(
        rf"^ {{2}}{re.escape(job)}:\n(?P<body>.*?)(?=\n {{2}}\S|\Z)",
        read_workflow(),
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"job {job!r} is missing from the workflow"
    return match.group("body")


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
        """正向：矩阵取值与超时预算和 ``shard_names()`` / 分片表逐字一致（改一处必须改另一处）。"""
        text = read_workflow()
        assert "matrix: ${{ fromJSON(needs.plan.outputs.matrix) }}" in text
        assert "timeout-minutes: ${{ matrix.timeout_minutes }}" in text
        payload = json.loads(render_matrix(select_shards()))
        assert payload[MATRIX_SHARD_KEY] == list(shard_names())
        assert payload[MATRIX_TIMEOUT_KEY] == [shard.timeout_minutes for shard in SHARDS]

    def test_matrix_does_not_use_fail_fast(self) -> None:
        """边界：一个分片失败不得取消其它分片（每个服务的结论都要拿到）。"""
        text = read_workflow()
        assert "strategy:" in text
        assert "fail-fast: false" in text
        assert "fail-fast: true" not in text

    def test_job_name_includes_the_shard(self) -> None:
        """正向：作业名带分片名，Job Summary 里能分辨是哪个分片。"""
        assert re.search(r"^ {4}name:\s*\S.*matrix\.shard", read_workflow(), re.MULTILINE)


class TestPlanJob:
    """矩阵真值：``plan`` 作业把分片表渲染成 JSON，``integration`` 作业原样消费。"""

    def test_plan_job_exposes_the_matrix_output(self) -> None:
        """正向：plan 作业存在，并把矩阵声明为作业级输出。"""
        body = job_body("plan")
        assert re.search(r"^ {4}outputs:\n^ {6}matrix:", body, re.MULTILINE)

    def test_matrix_is_emitted_by_the_real_shards_cli(self) -> None:
        """边界：工作流渲染矩阵的命令必须能被真实解析器解析，且输出单行 JSON。"""
        body = job_body("plan")
        assert f"python -m {SHARDS_MODULE} --format matrix" in body
        assert '>> "$GITHUB_OUTPUT"' in body

    def test_integration_job_consumes_the_plan_output(self) -> None:
        """正向：integration 作业 ``needs: plan`` 并把输出喂给 ``fromJSON``。"""
        body = job_body("integration")
        assert re.search(r"^ {4}needs:\s*plan\b", body, re.MULTILINE)
        assert "matrix: ${{ fromJSON(needs.plan.outputs.matrix) }}" in body

    def test_plan_job_installs_runtime_dependencies_only(self) -> None:
        """边界：plan 作业只 import 分片表，装运行期依赖即可（不拉 dev extras）。"""
        body = job_body("plan")
        assert "pip install -e ." in body
        assert ".[dev]" not in body

    def test_plan_job_starts_no_containers(self) -> None:
        """边界：plan 作业只算矩阵，不碰任何中间件。"""
        assert "docker compose" not in job_body("plan")

    def test_plan_runs_before_integration_and_aggregate(self) -> None:
        """边界：作业顺序必须是 plan → integration → aggregate。"""
        text = read_workflow()
        assert text.index("\n  plan:\n") < text.index("\n  integration:\n")
        assert text.index("\n  integration:\n") < text.index("\n  aggregate:\n")


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
        assert {item.format for item in parsed} == {
            "env",
            "compose",
            "init",
            "paths",
            "matrix",
        }
        matrix_invocations = [item for item in parsed if item.format == "matrix"]
        assert len(matrix_invocations) == 1, "the plan job must render the matrix exactly once"
        assert matrix_invocations[0].shard == [], "matrix rendering always covers every shard"
        for item in parsed:
            if item.format != "matrix":
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

    @pytest.mark.parametrize("name", ["redis-cluster", "redis-cluster-failover"])
    def test_cluster_shards_start_all_six_nodes(self, name: str) -> None:
        """正向：两个集群分片各自声明 6 个节点，工作流通过 ``--format compose`` 交给 compose。"""
        cluster = resolve_shard(name)
        assert len(cluster.compose_services) == 6
        for index in range(1, 7):
            assert f"redis-cluster-{index}" in cluster.compose_services

    def test_cluster_shards_run_the_one_shot_initializer(self) -> None:
        """正向：初始化容器由 ``--format init`` 给出，工作流不硬编码容器名。"""
        for name in ("redis-cluster", "redis-cluster-failover"):
            assert resolve_shard(name).initializer == CLUSTER_INITIALIZER
        assert "--format init" in read_workflow()
        assert "redis-cluster-init" not in read_workflow()

    def test_non_cluster_shards_have_no_initializer(self) -> None:
        """边界：非集群分片不声明初始化容器（``--format init`` 输出空串即跳过）。"""
        for name in shard_names():
            if not name.startswith("redis-cluster"):
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
    """分片报告的上传与趋势上收（D47：趋势与基线上收汇总作业，只留一套滚动基线）。"""

    def test_pytest_emits_junit_report_per_shard(self) -> None:
        """正向：pytest 把结果写进分片自己的 JUnit XML（汇总与逐分片趋势的唯一输入）。"""
        assert f'--junit-xml="{SHARD_REPORT_DIR}/{JUNIT_REPORT_NAME}"' in read_workflow()

    def test_uploads_report_as_artifact_per_shard(self) -> None:
        """正向：报告目录按分片作为 artifact 上传，供汇总作业下载。"""
        text = read_workflow()
        assert "actions/upload-artifact@v4" in text
        assert f"name: integration-report-{SHARD_EXPRESSION}" in text
        assert re.search(rf"path:\s*{re.escape(SHARD_REPORT_DIR)}/\s*$", text, re.MULTILINE)

    def test_upload_step_runs_even_when_tests_fail(self) -> None:
        """边界：用例失败也要上传报告（失败信息正是汇总作业最需要的输入）。"""
        body = step_body("Upload integration report")
        assert re.search(r"if:\s*always\(\)", body)

    def test_shard_job_no_longer_renders_a_trend_summary(self) -> None:
        """正向（D47）：分片作业不再调用 trend CLI，也不再渲染分片级 Markdown 摘要。"""
        text = read_workflow()
        assert TREND_MODULE not in text
        assert "integration-trend.md" not in text

    def test_shard_job_keeps_no_baseline_of_its_own(self) -> None:
        """边界（D47）：分片作业不再滚动自己的报告缓存，也没有 baseline 改名步骤。"""
        integration = job_body("integration")
        assert "key:" not in integration
        assert "actions/cache" not in integration
        assert "previous.xml" not in read_workflow()

    def test_only_one_rolling_cache_remains_for_the_whole_run(self) -> None:
        """边界（D47）：整套夜间报告只保留汇总作业的一份基线缓存（两套基线合一）。"""
        text = read_workflow()
        assert text.count("uses: actions/cache@") == 1
        aggregate = aggregate_job_body()
        assert "uses: actions/cache@v4" in aggregate
        assert f"path: {BASELINE_LEDGER_DIR}" in aggregate

    def test_trend_cli_module_still_importable_for_local_use(self) -> None:
        """正对照（D47）：trend 模块与 CLI 仍保留（本地单报告场景可用，只是工作流不调用）。"""
        module = importlib.import_module(TREND_MODULE)
        assert callable(module.build_trend)
        assert callable(module.main)

    def test_readme_documents_the_shard_level_trend(self) -> None:
        """正向（D47）：README 必须说明逐分片趋势由汇总作业渲染（否则没人知道去哪看）。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert "逐分片趋势" in readme
        assert AGGREGATE_MODULE in readme


class TestAggregateJob:
    """汇总作业：把五个分片的报告合成一份总账，并给出整轮唯一的结论退出码。"""

    def test_job_waits_for_the_matrix_and_always_runs(self) -> None:
        """正向：汇总作业依赖矩阵作业，且无论矩阵成败都要出总账。"""
        body = aggregate_job_body()
        assert re.search(r"needs:\s*integration\b", body)
        assert re.search(r"if:\s*always\(\)", body)

    def test_aggregate_runs_after_the_matrix_job(self) -> None:
        """边界：作业顺序必须是矩阵在前、汇总在后（``needs`` 的实际语义）。"""
        text = read_workflow()
        assert text.index("\n  integration:\n") < text.index("\n  aggregate:\n")

    def test_downloads_every_shard_artifact_by_prefix(self) -> None:
        """正向：按 artifact 名前缀一次性下载五个分片的报告目录。"""
        text = read_workflow()
        assert "actions/download-artifact@v4" in text
        assert f"pattern: {ARTIFACT_PREFIX}*" in text
        assert f"path: {AGGREGATE_REPORTS_DIR}" in text

    def test_aggregate_cli_is_invoked_with_real_options(self) -> None:
        """边界：工作流传给聚合 CLI 的参数必须能被真实解析器解析通过。"""
        parsed = build_aggregate_parser().parse_args(aggregate_arguments())
        assert parsed.reports_dir == Path(AGGREGATE_REPORTS_DIR)
        assert parsed.output == Path(AGGREGATE_REPORTS_DIR) / AGGREGATE_SUMMARY_NAME
        # 不限定分片：期望清单由分片表给出，与矩阵取值同源。
        assert parsed.shard == []

    def test_publishes_the_summary_even_when_aggregation_fails(self) -> None:
        """边界：聚合步骤失败（缺报告 / 有失败用例）时总账仍要贴进 Job Summary。"""
        body = step_body("Publish aggregate summary")
        assert re.search(r"if:\s*always\(\)", body)
        assert f'cat "{AGGREGATE_REPORTS_DIR}/{AGGREGATE_SUMMARY_NAME}"' in body
        assert '"$GITHUB_STEP_SUMMARY"' in body

    def test_expected_shards_come_from_the_shard_table(self) -> None:
        """正向：汇总默认期望的分片就是分片表的全部条目（与矩阵逐字同源）。"""
        statuses = collect_shard_reports(Path("no-such-reports"))
        assert tuple(status.name for status in statuses) == shard_names()
        assert all(not status.present for status in statuses)

    def test_aggregate_job_starts_no_services(self) -> None:
        """边界：汇总作业只读 artifact，不得再拉起任何中间件容器。"""
        assert "docker compose" not in aggregate_job_body()

    def test_readme_documents_the_aggregate_job(self) -> None:
        """正向：README 必须说明汇总作业与聚合入口（否则改聚合口径没人知道）。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert AGGREGATE_MODULE in readme
        assert AGGREGATE_SUMMARY_NAME in readme


class TestAggregateBaseline:
    """总账基线：汇总作业自己把上一轮报告滚动起来，好让总账能和上一轮对比。"""

    def test_restores_the_ledger_baseline_from_a_rolling_cache(self) -> None:
        """正向：基线与分片报告一样走 ``actions/cache`` 滚动（key 带 run_id + 前缀恢复）。"""
        body = step_body("Restore previous ledger baseline")
        assert "actions/cache@v4" in body
        assert f"path: {BASELINE_LEDGER_DIR}" in body
        assert f"key: {BASELINE_CACHE_PREFIX}${{{{ github.run_id }}}}" in body
        assert re.search(r"restore-keys:\s*\|", body)

    def test_baseline_is_restored_before_aggregation(self) -> None:
        """边界：基线必须在聚合之前恢复，否则对比永远拿不到上一轮数据。"""
        body = aggregate_job_body()
        restore = body.index("- name: Restore previous ledger baseline")
        assert restore < body.index("- name: Aggregate shard reports")

    def test_aggregate_cli_uses_the_ledger_baseline_dir(self) -> None:
        """正向：聚合 CLI 的 ``--baseline-dir`` / ``--snapshot-dir`` 都指向基线目录。"""
        parsed = build_aggregate_parser().parse_args(aggregate_arguments())
        assert parsed.baseline_dir == Path(BASELINE_LEDGER_DIR)
        assert parsed.snapshot_dir == Path(BASELINE_LEDGER_DIR)

    def test_aggregate_options_exist_in_the_real_parser(self) -> None:
        """边界：工作流传给聚合 CLI 的每个长选项都必须真实存在。"""
        known = {
            option
            for action in build_aggregate_parser()._actions
            for option in action.option_strings
        }
        flags = [token for token in aggregate_arguments() if token.startswith("--")]
        assert flags, "workflow must pass options to the aggregate module"
        unknown = [flag for flag in flags if flag not in known]
        assert unknown == [], f"workflow references unknown aggregate options: {unknown}"

    def test_baseline_dir_does_not_collide_with_the_reports_dir(self) -> None:
        """边界：基线目录必须与 artifact 下载目录分开，否则会被下一轮的报告覆盖。"""
        assert BASELINE_LEDGER_DIR != AGGREGATE_REPORTS_DIR
        assert f"path: {BASELINE_LEDGER_DIR}" in read_workflow()

    def test_baseline_dir_is_gitignored(self) -> None:
        """边界：本地跑 ``--snapshot-dir`` 会在仓库根建目录，必须不进版本库。"""
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        assert f"{BASELINE_LEDGER_DIR}/" in ignored

    def test_readme_documents_the_ledger_comparison(self) -> None:
        """正向：README 必须说明总账与上一轮的对比（否则改口径没人知道）。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert BASELINE_LEDGER_DIR in readme
        assert "--baseline-dir" in readme
        assert "--snapshot-dir" in readme
        assert "与上一轮对比" in readme


class TestDashboardJob:
    """看板作业（D48）：同一份总账渲染成自包含 HTML，发布到 GitHub Pages。"""

    def test_dashboard_job_waits_for_aggregate_and_always_runs(self) -> None:
        """正向：看板作业依赖汇总作业，且无论汇总成败都要发布（最需要看板时反而要有看板）。"""
        body = job_body("dashboard")
        assert re.search(r"needs:\s*aggregate\b", body)
        assert re.search(r"if:\s*always\(\)", body)

    def test_dashboard_runs_after_aggregate(self) -> None:
        """边界：作业顺序必须是矩阵在前、汇总居中、看板收尾。"""
        text = read_workflow()
        assert text.index("\n  aggregate:\n") < text.index("\n  dashboard:\n")

    def test_dashboard_downloads_the_ledger_artifact_by_name(self) -> None:
        """正向：看板按汇总作业上传的账本 artifact 名精确下载（名字由聚合模块给出）。"""
        body = job_body("dashboard")
        assert "actions/download-artifact@v4" in body
        assert f"name: {LEDGER_ARTIFACT_NAME}" in body
        assert f"path: {LEDGER_DOWNLOAD_DIR}" in body

    def test_ledger_artifact_upload_runs_even_when_tests_fail(self) -> None:
        """边界：矩阵或汇总失败时账本 artifact 仍要上传（看板要能展示失败那一轮）。"""
        body = step_body("Upload ledger artifact")
        assert re.search(r"if:\s*always\(\)", body)
        assert f"name: {LEDGER_ARTIFACT_NAME}" in body
        assert "actions/upload-artifact@v4" in body
        assert re.search(r"path:\s*\|", body)
        assert "reports" in body
        assert "baseline" in body

    def test_ledger_artifact_is_uploaded_before_aggregation(self) -> None:
        """边界：账本必须在聚合之前上传——``--snapshot-dir`` 会把 baseline/ 覆盖成本轮快照，
        之后再上传就只剩「本轮 vs 本轮」，对比小节永远显示持平。"""
        body = aggregate_job_body()
        upload = body.index(f"name: {LEDGER_ARTIFACT_NAME}")
        assert upload < body.index("- name: Aggregate shard reports")
        restore = body.index("- name: Restore previous ledger baseline")
        assert restore < upload

    def test_baseline_directory_is_prepared_before_upload(self) -> None:
        """边界：缓存未命中时 baseline/ 不存在，先建目录再上传（首轮运行也要有账本）。"""
        body = aggregate_job_body()
        assert body.index("- name: Ensure baseline directory exists") < body.index(
            f"name: {LEDGER_ARTIFACT_NAME}"
        )

    def test_dashboard_cli_is_invoked_with_real_options(self) -> None:
        """边界：工作流传给看板 CLI 的参数必须能被真实解析器解析，且指向账本目录。"""
        parsed = build_dashboard_parser().parse_args(dashboard_arguments())
        assert parsed.reports_dir == Path(LEDGER_DOWNLOAD_DIR) / "reports"
        assert parsed.baseline_dir == Path(LEDGER_DOWNLOAD_DIR) / BASELINE_LEDGER_DIR
        assert parsed.output == Path(DEFAULT_SITE_DIR) / DASHBOARD_INDEX_NAME
        assert parsed.shard == []
        assert "github.run_id" in parsed.run_url

    def test_dashboard_cli_options_exist_in_the_real_parser(self) -> None:
        """边界：工作流传给看板 CLI 的每个长选项都必须真实存在。"""
        known = {
            option
            for action in build_dashboard_parser()._actions
            for option in action.option_strings
        }
        flags = [token for token in dashboard_arguments() if token.startswith("--")]
        assert flags, "workflow must pass options to the dashboard module"
        unknown = [flag for flag in flags if flag not in known]
        assert unknown == [], f"workflow references unknown dashboard options: {unknown}"

    def test_dashboard_publishes_to_github_pages(self) -> None:
        """正向：页面通过锁版本的 Pages 动作发布，publish_dir 与 CLI 默认输出一致。"""
        body = job_body("dashboard")
        assert f"uses: {PAGES_ACTION}@v4" in body
        assert f"publish_dir: ./{DEFAULT_SITE_DIR}" in body
        assert "keep_files: true" in body

    def test_publish_permissions_are_scoped_to_the_dashboard_job(self) -> None:
        """边界：``contents: write`` 只给看板作业（其余作业保持只读，最小权限）。"""
        assert "contents: write" in job_body("dashboard")
        for other in ("plan", "integration", "aggregate"):
            assert "contents: write" not in job_body(other)

    def test_dashboard_job_installs_runtime_dependencies_only(self) -> None:
        """边界：渲染看板是纯标准库逻辑，只装运行期依赖，不拉 dev extras。"""
        body = job_body("dashboard")
        assert "pip install -e ." in body
        assert ".[dev]" not in body

    def test_dashboard_job_starts_no_containers(self) -> None:
        """边界：看板作业只读 artifact，不得再拉起任何中间件容器。"""
        assert "docker compose" not in job_body("dashboard")

    def test_dashboard_job_adds_no_second_baseline_cache(self) -> None:
        """边界（D47 延续）：看板不引入第二份滚动缓存——整套夜间报告仍只有一份基线。"""
        assert read_workflow().count("uses: actions/cache@") == 1
        assert "actions/cache" not in job_body("dashboard")

    def test_readme_documents_the_dashboard(self) -> None:
        """正向：README 必须说明看板入口与账本 artifact（否则没人知道去哪看页面）。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert DASHBOARD_MODULE in readme
        assert LEDGER_ARTIFACT_NAME in readme
        assert "GitHub Pages" in readme


class TestStatusNotification:
    """状态徽章（D49）：汇总作业把整轮结论回写成 commit status。"""

    def test_status_is_posted_after_the_summary_is_published(self) -> None:
        """正向：先贴总账再广播状态（顺序即「先总结、后通知」）。"""
        body = aggregate_job_body()
        publish = body.index("- name: Publish aggregate summary")
        assert publish < body.index(f"- name: {NOTIFIER_STEP_NAME}")

    def test_status_step_always_runs_and_never_fails_the_job(self) -> None:
        """边界：步骤 ``if: always()`` 且 ``continue-on-error``——徽章没更新不染红测试结论。"""
        body = step_body(NOTIFIER_STEP_NAME)
        assert re.search(r"if:\s*always\(\)", body)
        assert "continue-on-error: true" in body

    def test_notifier_cli_is_invoked_with_real_options(self) -> None:
        """边界：工作流传给 notifier CLI 的参数必须能被真实解析器解析。"""
        parsed = build_notifier_parser().parse_args(notifier_arguments())
        assert parsed.repo == "${{ github.repository }}"
        assert parsed.sha == "${{ github.sha }}"
        assert parsed.reports_dir == Path(AGGREGATE_REPORTS_DIR)
        assert "github.run_id" in parsed.target_url

    def test_notifier_cli_options_exist_in_the_real_parser(self) -> None:
        """边界：工作流传给 notifier CLI 的每个长选项都必须真实存在。"""
        known = {
            option
            for action in build_notifier_parser()._actions
            for option in action.option_strings
        }
        flags = [token for token in notifier_arguments() if token.startswith("--")]
        assert flags, "workflow must pass options to the notifier module"
        unknown = [flag for flag in flags if flag not in known]
        assert unknown == [], f"workflow references unknown notifier options: {unknown}"

    def test_token_goes_through_the_environment_not_the_command_line(self) -> None:
        """边界：token 只经 ``GITHUB_TOKEN`` 传入，命令行不出现 secret（避免进日志）。"""
        body = step_body(NOTIFIER_STEP_NAME)
        assert "GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}" in body
        arguments = notifier_arguments()
        assert "--token" not in arguments
        assert "secrets.GITHUB_TOKEN" not in " ".join(arguments)

    def test_status_context_comes_from_the_module_not_the_workflow(self) -> None:
        """边界：context 名只在模块里定义，工作流不重复声明字面量（口径不漂移）。"""
        assert STATUS_CONTEXT == "integration/nightly"
        assert f'context: "{STATUS_CONTEXT}"' not in read_workflow()

    def test_status_write_permission_is_scoped_to_the_aggregate_job(self) -> None:
        """边界：``statuses: write`` 只给汇总作业（最小权限；其余作业保持默认只读）。"""
        assert re.search(r"permissions:\s*\n\s*statuses:\s*write", aggregate_job_body())
        for other in ("plan", "integration", "dashboard"):
            assert "statuses: write" not in job_body(other)

    def test_status_step_starts_no_containers_and_adds_no_cache(self) -> None:
        """边界：发布步骤不拉容器、不引入第二份缓存（整套夜间报告仍只有一份基线）。"""
        body = step_body(NOTIFIER_STEP_NAME)
        assert "docker compose" not in body
        assert "actions/cache" not in body
        assert read_workflow().count("uses: actions/cache@") == 1

    def test_readme_documents_the_status_badge(self) -> None:
        """正向：README 必须带夜间集成徽章并说明状态发布入口（否则没人知道去哪看）。"""
        readme = README_PATH.read_text(encoding="utf-8")
        assert NOTIFIER_MODULE in readme
        assert "img.shields.io/github/actions/workflow/status" in readme
        assert STATUS_CONTEXT in readme


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
