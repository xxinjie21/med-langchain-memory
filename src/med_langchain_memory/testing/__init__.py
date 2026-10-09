"""测试支撑层：可选真实中间件集成测试的服务探测与开关。

本包**不参与生产链路**（``med_langchain_memory/__init__.py`` 不导入它），
仅被 ``tests/test_integration/`` 使用，且只依赖标准库。

公开符号见 :mod:`med_langchain_memory.testing.services`（服务探测与开关）与
:mod:`med_langchain_memory.testing.cluster`（集群拓扑解析与故障转移演练）。
"""

from __future__ import annotations

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

__all__ = [
    "CLUSTER_ANNOUNCE_ENV",
    "COMPOSE_HINT",
    "COMPOSE_PROJECT_NAME",
    "DEFAULT_DOCKER_TIMEOUT_SECONDS",
    "DEFAULT_POLL_INTERVAL_SECONDS",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_SERVICES",
    "DEFAULT_WAIT_INTERVAL_SECONDS",
    "DEFAULT_WAIT_SECONDS",
    "ENV_PREFIX",
    "FAILOVER_WAIT_SECONDS",
    "RECOVERY_WAIT_SECONDS",
    "REDIS_CLUSTER_NODE_PORTS",
    "ClusterNodeState",
    "DockerContainerController",
    "FailoverPlan",
    "IntegrationService",
    "ServiceStatus",
    "check_service",
    "cluster_container_name",
    "cluster_node_index",
    "cluster_startup_nodes",
    "detect_host_address",
    "find_by_node_id",
    "integration_enabled",
    "parse_cluster_nodes",
    "parse_slots",
    "plan_failover",
    "probe_tcp",
    "resolve_service",
    "run_docker",
    "wait_for_service",
    "wait_until",
]
