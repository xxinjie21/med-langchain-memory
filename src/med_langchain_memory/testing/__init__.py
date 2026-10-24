"""测试支撑层：可选真实中间件集成测试的服务探测与开关。

本包**不参与生产链路**（``med_langchain_memory/__init__.py`` 不导入它），
仅被 ``tests/test_integration/`` 使用，且只依赖标准库。

公开符号见 :mod:`med_langchain_memory.testing.services`（服务探测与开关）、
:mod:`med_langchain_memory.testing.cluster`（集群拓扑解析与故障转移演练）、
:mod:`med_langchain_memory.testing.trend`（JUnit 报告汇总与趋势对比）、
:mod:`med_langchain_memory.testing.preflight`（CI 前置守卫：必需服务不可用即失败）、
:mod:`med_langchain_memory.testing.shards`（集成套件按服务分片：并行作业的单一真值来源）、
:mod:`med_langchain_memory.testing.aggregate`（分片结果聚合：分片报告合成一份总账、
与上一轮基线对比（含逐分片用例级趋势 :class:`ShardTrend`）、给出单一退出码）与
:mod:`med_langchain_memory.testing.dashboard`（把同一份总账渲染成自包含 HTML
看板页，由工作流发布到 GitHub Pages）与
:mod:`med_langchain_memory.testing.notifier`（把同一份总账的结论回写成 GitHub
commit status，并给出 README 徽章片段）与
:mod:`med_langchain_memory.testing.commitlint`（Conventional Commits 提交信息守门：
pre-commit commit-msg 钩子与 ci.yml PR 作业共用同一套字段级规则）。
"""

from __future__ import annotations

from .aggregate import (
    AGGREGATE_SUMMARY_NAME,
    ARTIFACT_PREFIX,
    BASELINE_LEDGER_DIR,
    EXIT_FAILURES,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
    LEDGER_ARTIFACT_NAME,
    AggregateReport,
    AggregateTrend,
    ShardDelta,
    ShardDeltaState,
    ShardStatus,
    ShardTrend,
    build_aggregate,
    build_aggregate_trend,
    collect_shard_reports,
    shard_report_path,
    snapshot_reports,
    verdict_exit_code,
)

# 注意：aggregate 的 CLI 入口（build_parser / main）同样**不**在此导出——
# 包根这两个名字属于 trend 的 CLI。聚合入口按模块路径调用：
# ``python -m med_langchain_memory.testing.aggregate``。
from .cluster import (
    COMPOSE_PROJECT_NAME,
    DEFAULT_DOCKER_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    FAILOVER_WAIT_SECONDS,
    RECOVERY_WAIT_SECONDS,
    ClusterNodeState,
    DockerContainerController,
    FailoverPlan,
    cluster_container_name,
    cluster_node_index,
    find_by_node_id,
    parse_cluster_nodes,
    parse_slots,
    plan_failover,
    run_docker,
    wait_until,
)

# 注意：preflight 的 CLI 入口（build_parser / main）**不**在此导出——
# 包根上的这两个名字已经属于 trend 的 CLI，导出会造成静默覆盖。
# 守卫的入口按模块路径调用：``python -m med_langchain_memory.testing.preflight``。
# 注意：commitlint 的 CLI 入口（build_parser / main）同样**不**在此导出——
# 包根这两个名字属于 trend 的 CLI。守门入口按模块路径调用：
# ``python -m med_langchain_memory.testing.commitlint``。
from .commitlint import (
    ALLOWED_SCOPES,
    ALLOWED_TYPES,
    EXIT_VIOLATIONS,
    GIT_LOG_FORMAT,
    MAX_SUBJECT_LENGTH,
    LintFinding,
    LintReport,
    iter_commit_messages,
    lint_commits,
    lint_message,
    parse_git_log_messages,
    render_report,
)
from .dashboard import (
    DASHBOARD_INDEX_NAME,
    DASHBOARD_TITLE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SITE_DIR,
    render_dashboard,
)

# 注意：notifier 的 CLI 入口（build_parser / main）同样**不**在此导出——
# 包根这两个名字属于 trend 的 CLI。状态发布入口按模块路径调用：
# ``python -m med_langchain_memory.testing.notifier``。
from .notifier import (
    EXIT_BAD_REPO,
    EXIT_POST_FAILED,
    EXIT_POSTED,
    INTEGRATION_WORKFLOW_FILE,
    MAX_DESCRIPTION_LENGTH,
    STATUS_CONTEXT,
    StatusState,
    badge_markdown,
    badge_url,
    build_status_payload,
    post_commit_status,
    status_description,
    status_state_for_exit_code,
    statuses_url,
    workflow_url,
)
from .preflight import (
    ENV_REQUIRED,
    PreflightReport,
    parse_required,
    preflight,
)
from .services import (
    CLUSTER_ANNOUNCE_ENV,
    COMPOSE_HINT,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    DEFAULT_SERVICES,
    DEFAULT_WAIT_INTERVAL_SECONDS,
    DEFAULT_WAIT_SECONDS,
    ENV_PREFIX,
    REDIS_CLUSTER_NODE_PORTS,
    IntegrationService,
    ServiceStatus,
    check_service,
    cluster_startup_nodes,
    detect_host_address,
    integration_enabled,
    probe_tcp,
    resolve_service,
    wait_for_service,
)

# 注意：shards 的 CLI 入口（build_parser / main）同样**不**在此导出——
# 包根这两个名字属于 trend 的 CLI。分片入口按模块路径调用：
# ``python -m med_langchain_memory.testing.shards``。
from .shards import (
    ALL_SHARDS,
    CLUSTER_INITIALIZER,
    DEFAULT_TIMEOUT_MINUTES,
    FORMATS,
    INTEGRATION_TEST_DIR,
    MATRIX_SHARD_KEY,
    MATRIX_TIMEOUT_KEY,
    SHARDS,
    Shard,
    render_compose,
    render_env,
    render_init,
    render_matrix,
    render_paths,
    render_required,
    render_table,
    required_services,
    resolve_shard,
    select_shards,
    shard_names,
)
from .trend import (
    BASELINE_REPORT_NAME,
    DEFAULT_REPORTS_DIR,
    DEFAULT_SLOWEST_LIMIT,
    JUNIT_REPORT_NAME,
    TREND_SUMMARY_NAME,
    CaseResult,
    SuiteReport,
    TrendReport,
    build_parser,
    build_trend,
    main,
    parse_junit_report,
    parse_junit_xml,
    render_markdown,
)

__all__ = [
    "AGGREGATE_SUMMARY_NAME",
    "ALL_SHARDS",
    "ALLOWED_SCOPES",
    "ALLOWED_TYPES",
    "ARTIFACT_PREFIX",
    "BASELINE_LEDGER_DIR",
    "BASELINE_REPORT_NAME",
    "CLUSTER_ANNOUNCE_ENV",
    "CLUSTER_INITIALIZER",
    "COMPOSE_HINT",
    "COMPOSE_PROJECT_NAME",
    "DASHBOARD_INDEX_NAME",
    "DASHBOARD_TITLE",
    "DEFAULT_DOCKER_TIMEOUT_SECONDS",
    "DEFAULT_OUTPUT_PATH",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_REPORTS_DIR",
    "DEFAULT_SERVICES",
    "DEFAULT_SITE_DIR",
    "DEFAULT_SLOWEST_LIMIT",
    "DEFAULT_TIMEOUT_MINUTES",
    "DEFAULT_WAIT_INTERVAL_SECONDS",
    "DEFAULT_WAIT_SECONDS",
    "ENV_PREFIX",
    "ENV_REQUIRED",
    "EXIT_BAD_REPO",
    "EXIT_FAILURES",
    "EXIT_INCOMPLETE",
    "EXIT_OK",
    "EXIT_POST_FAILED",
    "EXIT_POSTED",
    "EXIT_USAGE",
    "EXIT_VIOLATIONS",
    "FAILOVER_WAIT_SECONDS",
    "FORMATS",
    "GIT_LOG_FORMAT",
    "INTEGRATION_TEST_DIR",
    "INTEGRATION_WORKFLOW_FILE",
    "JUNIT_REPORT_NAME",
    "LEDGER_ARTIFACT_NAME",
    "MATRIX_SHARD_KEY",
    "MATRIX_TIMEOUT_KEY",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_SUBJECT_LENGTH",
    "RECOVERY_WAIT_SECONDS",
    "REDIS_CLUSTER_NODE_PORTS",
    "SHARDS",
    "STATUS_CONTEXT",
    "TREND_SUMMARY_NAME",
    "AggregateReport",
    "AggregateTrend",
    "CaseResult",
    "ClusterNodeState",
    "DockerContainerController",
    "FailoverPlan",
    "IntegrationService",
    "LintFinding",
    "LintReport",
    "PreflightReport",
    "ServiceStatus",
    "Shard",
    "ShardDelta",
    "ShardDeltaState",
    "ShardStatus",
    "ShardTrend",
    "StatusState",
    "SuiteReport",
    "TrendReport",
    "badge_markdown",
    "badge_url",
    "build_aggregate",
    "build_aggregate_trend",
    "build_parser",
    "build_status_payload",
    "build_trend",
    "check_service",
    "cluster_container_name",
    "cluster_node_index",
    "cluster_startup_nodes",
    "collect_shard_reports",
    "detect_host_address",
    "find_by_node_id",
    "integration_enabled",
    "iter_commit_messages",
    "lint_commits",
    "lint_message",
    "main",
    "parse_cluster_nodes",
    "parse_git_log_messages",
    "parse_junit_report",
    "parse_junit_xml",
    "parse_required",
    "parse_slots",
    "plan_failover",
    "post_commit_status",
    "preflight",
    "probe_tcp",
    "render_compose",
    "render_dashboard",
    "render_env",
    "render_init",
    "render_markdown",
    "render_matrix",
    "render_paths",
    "render_report",
    "render_required",
    "render_table",
    "required_services",
    "resolve_service",
    "resolve_shard",
    "run_docker",
    "select_shards",
    "shard_names",
    "shard_report_path",
    "snapshot_reports",
    "status_description",
    "status_state_for_exit_code",
    "statuses_url",
    "wait_for_service",
    "verdict_exit_code",
    "wait_until",
    "workflow_url",
]
