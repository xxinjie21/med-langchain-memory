"""真实 Redis Cluster 故障转移演练（默认跳过，见 ``conftest.py``）。

这是本项目**唯一会主动破坏环境**的用例：它会 ``docker stop`` 掉一个主节点，
验证集群自动把它的从节点晋升为新主，并在验证读写仍可用之后把节点重新拉起。
单机替身与离线断言都无法覆盖「主节点消失后拓扑自动收敛」这层契约，
因此这条演练必须跑在真实三主三从集群上。

演练链路（一次用例内完成，无论成败都会尝试恢复原状）：

1. 从存活节点读 ``CLUSTER NODES`` 原始文本 → :func:`plan_failover` 选出「主 + 其从」；
2. ``docker stop`` 主节点容器 → 轮询拓扑，直到从节点晋升为 ``master`` 且持有槽位；
3. 用**存活节点**重建集群客户端，验证读写在晋升后的新主上依旧可用；
4. ``docker start`` 原主节点容器 → 轮询直到它作为从节点重新入列、
   集群回到「3 个主节点 + 16384 槽全覆盖」；
5. 刷新会话级集群客户端的槽位映射，避免后续用例读到过期拓扑。

运行方式::

    export MED_MEMORY_IT_CLUSTER_IP=<宿主机局域网 IP>   # 仅 Docker Desktop 需要
    docker compose -f docker-compose.integration.yml up -d
    MED_MEMORY_IT=1 pytest -m integration tests/test_integration/test_redis_cluster_failover.py
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Any

import pytest

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.stores.redis_cluster_store import (
    RedisClusterMedHistory,
    build_cluster_client,
)
from med_langchain_memory.testing import (
    FAILOVER_WAIT_SECONDS,
    RECOVERY_WAIT_SECONDS,
    REDIS_CLUSTER_NODE_PORTS,
    ClusterNodeState,
    DockerContainerController,
    FailoverPlan,
    find_by_node_id,
    parse_cluster_nodes,
    plan_failover,
    wait_until,
)

NAMESPACE: dict[str, str] = {
    "session_id": "s-cluster-failover",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-2048",
}

#: 集群槽位总数（3 主全覆盖才算恢复）。
TOTAL_SLOTS = 16384

#: 拓扑轮询间隔（秒）。
POLL_INTERVAL_SECONDS = 2.0


def make_message(content: str = "chest pain") -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    return MedMessage(**NAMESPACE, role=MessageRole.PATIENT, content=content)


def surviving_ports(excluding: int) -> list[int]:
    """返回除 ``excluding`` 之外的集群节点端口（演练期间的可达节点）。"""
    return [port for port in REDIS_CLUSTER_NODE_PORTS if port != excluding]


def read_topology(host: str, ports: Sequence[int]) -> tuple[ClusterNodeState, ...]:
    """从任一存活节点读取并解析 ``CLUSTER NODES`` 原始文本。

    刻意用 ``execute_command("CLUSTER", "NODES")`` 而不是 ``cluster_nodes()``：
    redis-py 只为字面量 ``"CLUSTER NODES"`` 注册了解析回调，
    按首参数 ``"CLUSTER"`` 查找时拿到的才是**原始文本**，
    从而与 :func:`parse_cluster_nodes` 的解析口径完全一致（含端口）。

    Args:
        host: 集群节点宿主地址（与 compose 发布端口同一主机）。
        ports: 候选节点端口，按顺序尝试。

    Returns:
        解析后的节点元组。

    Raises:
        AssertionError: 所有候选节点都无法应答时。
    """
    redis_module = pytest.importorskip("redis", reason="redis is an optional dependency")
    errors: list[str] = []
    for port in ports:
        probe = redis_module.Redis(
            host=host,
            port=port,
            socket_timeout=5.0,
            socket_connect_timeout=5.0,
        )
        try:
            return parse_cluster_nodes(probe.execute_command("CLUSTER", "NODES"))
        except redis_module.exceptions.RedisError as exc:
            errors.append(f"{port}: {exc}")
        finally:
            probe.close()
    raise AssertionError(f"no cluster node answered CLUSTER NODES ({'; '.join(errors)})")


def is_promoted(host: str, plan: FailoverPlan) -> bool:
    """判断计划中的从节点是否已晋升为新主并接管槽位。"""
    nodes = read_topology(host, surviving_ports(plan.primary.port))
    promoted = find_by_node_id(nodes, plan.replica.node_id)
    return promoted is not None and promoted.is_primary and promoted.owns_slots


def is_recovered(host: str, plan: FailoverPlan) -> bool:
    """判断集群是否已回到「3 主 + 16384 槽全覆盖」且旧主已降级重新入列。"""
    nodes = read_topology(host, surviving_ports(plan.primary.port))
    former_primary = find_by_node_id(nodes, plan.primary.node_id)
    if former_primary is None or former_primary.owns_slots or former_primary.is_failed:
        return False
    owners = [node for node in nodes if node.is_primary and node.owns_slots]
    return len(owners) == 3 and sum(node.slot_count for node in owners) == TOTAL_SLOTS


@pytest.fixture(scope="module")
def docker_controller() -> DockerContainerController:
    """docker CLI 控制器；宿主机没有 ``docker`` 时跳过整个模块。"""
    if shutil.which("docker") is None:
        pytest.skip("docker CLI is not available; cannot drill a cluster failover")
    return DockerContainerController()


@pytest.mark.integration
class TestClusterFailoverDrill:
    """主节点下线 → 从节点晋升 → 读写可用 → 原主恢复。"""

    def test_primary_failure_promotes_replica_and_keeps_read_write(
        self,
        redis_cluster_service: Any,
        redis_cluster_live: Any,
        docker_controller: DockerContainerController,
    ) -> None:
        """端到端演练：停掉一个主节点后，集群仍能读写且能恢复。

        恢复动作放在 ``finally`` 里：即使断言失败也要把节点拉回来，
        避免把演练残留的降级集群留给后续用例。
        """
        host = redis_cluster_service.host
        plan = plan_failover(read_topology(host, REDIS_CLUSTER_NODE_PORTS))
        if plan is None:
            pytest.skip("cluster topology has no managed primary/replica pair to drill")

        try:
            docker_controller.stop(plan.primary_container)

            assert wait_until(
                lambda: is_promoted(host, plan),
                timeout=FAILOVER_WAIT_SECONDS,
                interval=POLL_INTERVAL_SECONDS,
            ), (
                f"{plan.replica.address} did not take over slot ownership within "
                f"{FAILOVER_WAIT_SECONDS:.0f}s after stopping {plan.primary.address}"
            )

            client = build_cluster_client(
                [{"host": host, "port": port} for port in surviving_ports(plan.primary.port)]
            )
            try:
                history = RedisClusterMedHistory(**NAMESPACE, client=client, ttl_seconds=120)
                history.add_med_messages([make_message("after failover")])
                assert [message.content for message in history.get_med_messages()] == [
                    "after failover"
                ]
                assert history.exists() is True
                history.clear()
                assert history.exists() is False
            finally:
                client.close()
        finally:
            docker_controller.start(plan.primary_container)
            assert wait_until(
                lambda: is_recovered(host, plan),
                timeout=RECOVERY_WAIT_SECONDS,
                interval=POLL_INTERVAL_SECONDS,
            ), (
                f"cluster did not converge back to 3 primaries after restarting "
                f"{plan.primary_container} (former primary {plan.primary.address}, "
                f"promoted replica container {plan.replica_container})"
            )
            # 会话级客户端缓存的是演练前的槽位映射，这里强制重新发现拓扑，
            # 否则后续集群用例会先撞上一连串 MOVED 才自愈。
            redis_cluster_live.nodes_manager.initialize()
