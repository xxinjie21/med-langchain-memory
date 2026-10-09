"""Redis Cluster 故障转移演练的支撑工具（纯标准库，不参与生产链路）。

定位：把「读拓扑 → 选演练对象 → 停/起容器 → 轮询等待收敛」这条链路里
**与 Redis 客户端无关**的部分抽成可离线单测的纯函数；真机用例
（``tests/test_integration/test_redis_cluster_failover.py``）只负责
「取原始拓扑文本 → 调本模块 → 断言读写仍可用」：

* :func:`parse_cluster_nodes` —— 解析 ``CLUSTER NODES`` 原始文本；
* :func:`plan_failover` —— 从拓扑里确定性地挑出「一个主节点 + 它的一个从节点」；
* :func:`cluster_node_index` / :func:`cluster_container_name` —— 端口 ↔ 容器名映射；
* :func:`wait_until` —— 可注入时钟的通用轮询；
* :class:`DockerContainerController` —— 用 ``docker`` CLI 停/起容器（执行器可注入）。

设计取舍：

* **只依赖标准库**：本模块不导入任何中间件客户端，因此缺包时「跳过」这一步永远可用；
* **不封装 Redis 命令**：读取拓扑由用例侧用 ``execute_command("CLUSTER", "NODES")``
  完成（该写法只按首个参数查响应回调，因此拿到的是原始文本而非 redis-py 的解析结果），
  本模块只做纯文本解析，不与客户端版本耦合。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from med_langchain_memory.exceptions import StorageError, ValidationError

from .services import DEFAULT_WAIT_SECONDS, REDIS_CLUSTER_NODE_PORTS

#: 集成栈的 compose 工程名（``docker-compose.integration.yml`` 的 ``name:``）。
#:
#: 集群节点在编排里显式声明了 ``container_name``，因此容器名与工程名固定对应，
#: 不随「谁在哪个目录下执行 compose」而漂移。
COMPOSE_PROJECT_NAME = "med-memory-integration"

#: 轮询间隔（秒）。
DEFAULT_POLL_INTERVAL_SECONDS = 1.0

#: 单次 ``docker`` 调用的超时（秒）。
DEFAULT_DOCKER_TIMEOUT_SECONDS = 60.0

#: 主节点被停掉后，等待从节点晋升的上限（秒）。
#:
#: 节点 ``--cluster-node-timeout`` 为 5000ms，晋升通常在 5–15 秒内完成；
#: 留足余量以容忍容器冷启动与 gossip 收敛。
FAILOVER_WAIT_SECONDS = 90.0

#: 旧主节点重新入列（降级为从节点）的上限（秒）。
RECOVERY_WAIT_SECONDS = 120.0


@dataclass(frozen=True)
class ClusterNodeState:
    """``CLUSTER NODES`` 单行解析结果。"""

    #: 40 位十六进制节点 ID。
    node_id: str

    #: 节点广播给客户端的主机（``--cluster-announce-ip``）。
    host: str

    #: 节点广播给客户端的客户端端口（``--cluster-announce-port``）。
    port: int

    #: 逗号分隔的标志位（``myself`` / ``master`` / ``slave`` / ``fail?`` / ``fail`` …）。
    flags: tuple[str, ...]

    #: 该从节点所属主节点的 ID；主节点自身为 ``None``（原始值为 ``-``）。
    master_id: str | None

    #: 负责的槽位闭区间（``((0, 5460), ...)``），从节点通常为空。
    slots: tuple[tuple[int, int], ...]

    #: 链路状态，``connected`` 或 ``disconnected``。
    link_state: str

    @property
    def address(self) -> str:
        """返回 ``host:port`` 形式的可读地址。"""
        return f"{self.host}:{self.port}"

    @property
    def is_primary(self) -> bool:
        """是否为已晋升的主节点（带 ``master`` 标志）。"""
        return "master" in self.flags

    @property
    def is_replica(self) -> bool:
        """是否为从节点（带 ``slave`` 标志）。"""
        return "slave" in self.flags

    @property
    def is_failed(self) -> bool:
        """是否已被集群标记为故障（``fail`` 标志，注意区别于待定故障 ``fail?``）。"""
        return "fail" in self.flags

    @property
    def owns_slots(self) -> bool:
        """是否持有至少一个槽位。"""
        return bool(self.slots)

    @property
    def slot_count(self) -> int:
        """持有槽位的总个数（闭区间求和）。"""
        return sum(end - start + 1 for start, end in self.slots)


@dataclass(frozen=True)
class FailoverPlan:
    """一次故障转移演练的目标组合：停掉的主节点 + 预期晋升的从节点。"""

    #: 将被停掉的主节点。
    primary: ClusterNodeState

    #: 预期接管的从节点。
    replica: ClusterNodeState

    #: 主节点对应的容器名。
    primary_container: str

    #: 从节点对应的容器名。
    replica_container: str


def _parse_address(address: str) -> tuple[str, int]:
    """从 ``ip:port@cport[,hostname]`` 中取出主机与客户端端口。"""
    host_port = address.split("@", 1)[0]
    host, _, raw_port = host_port.rpartition(":")
    if not host or not raw_port.isdigit():
        raise ValidationError(f"malformed node address in CLUSTER NODES: {address!r}")
    return host, int(raw_port)


def parse_slots(tokens: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """解析 ``CLUSTER NODES`` 行尾的槽位标记为闭区间元组。

    迁移中的标记（形如 ``[4096->-<node-id>]`` / ``[4096-<-<node-id>]``）
    表示槽位正在搬运，不属于稳定归属，直接跳过。

    Args:
        tokens: 第 9 个字段起的槽位标记序列，如 ``["0-5460", "5461"]``。

    Returns:
        闭区间元组，如 ``((0, 5460), (5461, 5461))``。

    Raises:
        ValidationError: 出现无法解析的槽位标记时。
    """
    ranges: list[tuple[int, int]] = []
    for token in tokens:
        if token.startswith("["):
            continue
        start, _, end = token.partition("-")
        if not start.isdigit() or (end and not end.isdigit()):
            raise ValidationError(f"malformed slot token in CLUSTER NODES: {token!r}")
        ranges.append((int(start), int(end) if end else int(start)))
    return tuple(ranges)


def _parse_node_line(line: str) -> ClusterNodeState:
    """解析 ``CLUSTER NODES`` 的单行输出。"""
    parts = line.split()
    if len(parts) < 8:
        raise ValidationError(f"malformed CLUSTER NODES line: {line!r}")
    host, port = _parse_address(parts[1])
    master_id = parts[3]
    return ClusterNodeState(
        node_id=parts[0],
        host=host,
        port=port,
        flags=tuple(parts[2].split(",")),
        master_id=None if master_id == "-" else master_id,
        slots=parse_slots(parts[8:]),
        link_state=parts[7],
    )


def parse_cluster_nodes(payload: str | bytes) -> tuple[ClusterNodeState, ...]:
    """把 ``CLUSTER NODES`` 原始输出解析为节点列表。

    Args:
        payload: 命令原始回复，``str`` 或 ``bytes``（未开 ``decode_responses`` 时是后者）。
            空行会被忽略，便于直接喂整段输出。

    Returns:
        按输出顺序排列的节点元组；无内容时为空元组。

    Raises:
        ValidationError: 某一行字段不足 8 个，或地址 / 槽位标记无法解析时。
    """
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8", errors="replace")
    return tuple(
        _parse_node_line(stripped)
        for stripped in (line.strip() for line in payload.splitlines())
        if stripped
    )


def find_by_node_id(nodes: Sequence[ClusterNodeState], node_id: str) -> ClusterNodeState | None:
    """按节点 ID 查找节点，不存在时返回 ``None``。"""
    return next((node for node in nodes if node.node_id == node_id), None)


def cluster_node_index(port: int) -> int:
    """把集群节点端口映射为 1 起的节点序号（1–6）。

    Args:
        port: 节点广播的客户端端口，取值必须属于 ``REDIS_CLUSTER_NODE_PORTS``。

    Returns:
        节点序号，``17001`` → ``1`` … ``17006`` → ``6``。

    Raises:
        ValidationError: 端口不属于集成栈的 6 个节点（对接自建集群时无法定位容器）。
    """
    try:
        return REDIS_CLUSTER_NODE_PORTS.index(port) + 1
    except ValueError as exc:
        raise ValidationError(
            f"port {port} is not one of the integration cluster node ports "
            f"{list(REDIS_CLUSTER_NODE_PORTS)}"
        ) from exc


def cluster_container_name(index: int) -> str:
    """返回第 ``index`` 个集群节点的容器名。

    名字必须与 ``docker-compose.integration.yml`` 里的 ``container_name`` 一致，
    否则 ``docker stop`` 会打在错误的目标上（或直接报容器不存在）。

    Args:
        index: 1 起的节点序号。

    Returns:
        形如 ``med-memory-integration-redis-cluster-3`` 的容器名。

    Raises:
        ValidationError: 序号越界（非 1–6）时。
    """
    if not 1 <= index <= len(REDIS_CLUSTER_NODE_PORTS):
        raise ValidationError(
            f"cluster node index {index} out of range 1..{len(REDIS_CLUSTER_NODE_PORTS)}"
        )
    return f"{COMPOSE_PROJECT_NAME}-redis-cluster-{index}"


def plan_failover(nodes: Sequence[ClusterNodeState]) -> FailoverPlan | None:
    """从集群拓扑中确定性地挑出一组可演练的「主节点 + 从节点」。

    选择规则（确定性，便于复现与单测）：

    1. 候选主节点：带 ``master`` 标志、持有槽位、未被标记 ``fail``，
       且端口属于集成栈的 6 个节点；
    2. 其从节点：``master_id`` 指向该主节点、链路 ``connected``、端口同样属于 6 个节点；
    3. 主节点与从节点都按端口升序扫描，取第一个满足条件的组合。

    Args:
        nodes: :func:`parse_cluster_nodes` 的解析结果。

    Returns:
        :class:`FailoverPlan`；没有可用组合（如集群已处于降级拓扑、
        或对接的是自建集群）时返回 ``None``。
    """
    managed = {port: index for index, port in enumerate(REDIS_CLUSTER_NODE_PORTS, start=1)}
    ordered = sorted(nodes, key=lambda node: node.port)
    for primary in ordered:
        if not (primary.is_primary and primary.owns_slots and not primary.is_failed):
            continue
        index = managed.get(primary.port)
        if index is None:
            continue
        replica = next(
            (
                node
                for node in ordered
                if node.master_id == primary.node_id
                and node.link_state == "connected"
                and node.port in managed
            ),
            None,
        )
        if replica is None:
            continue
        return FailoverPlan(
            primary=primary,
            replica=replica,
            primary_container=cluster_container_name(index),
            replica_container=cluster_container_name(managed[replica.port]),
        )
    return None


def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout: float = DEFAULT_WAIT_SECONDS,
    interval: float = DEFAULT_POLL_INTERVAL_SECONDS,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """轮询 ``predicate`` 直到为真或超时。

    Args:
        predicate: 每次轮询调用的判定函数；其抛出的异常不在此处吞掉，
            由调用方决定「异常算失败还是继续等」。
        timeout: 最长等待秒数；``<= 0`` 时只判定一次。
        interval: 两次判定之间的休眠秒数（``<= 0`` 时不休眠）。
        clock: 单调时钟函数，默认 ``time.monotonic``（单测注入假时钟）。
        sleep: 休眠函数，默认 ``time.sleep``（单测注入计数器）。

    Returns:
        超时前 ``predicate`` 返回过 ``True`` 时为 ``True``，否则 ``False``。
    """
    deadline = clock() + max(timeout, 0.0)
    while True:
        if predicate():
            return True
        if clock() >= deadline:
            return False
        sleep(max(interval, 0.0))


def run_docker(args: Sequence[str], *, timeout: float = DEFAULT_DOCKER_TIMEOUT_SECONDS) -> str:
    """执行一次 ``docker`` 子命令并返回标准输出。

    Args:
        args: 子命令与参数，如 ``["stop", "med-memory-integration-redis-cluster-1"]``。
        timeout: 命令超时（秒）。

    Returns:
        命令的标准输出（文本模式）。

    Raises:
        StorageError: ``docker`` 可执行文件缺失、调用超时，或退出码非 0 时。
    """
    command = ["docker", *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise StorageError(f"failed to run {' '.join(command)}: {exc}") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise StorageError(
            f"docker {' '.join(args)} failed (exit {completed.returncode}): {detail}"
        )
    return completed.stdout


class DockerContainerController:
    """用 ``docker`` CLI 停 / 起集成栈容器（故障注入与恢复）。

    执行器可注入，因此「停主节点 → 起主节点」这条编排逻辑可以在没有 Docker 的
    环境里用假执行器完整单测。
    """

    def __init__(self, runner: Callable[[Sequence[str]], str] = run_docker) -> None:
        """初始化控制器。

        Args:
            runner: 接收 ``docker`` 子命令序列的执行器，默认走真实 CLI。
        """
        self._runner = runner

    def stop(self, container: str) -> None:
        """停止容器（等价 ``docker stop <container>``）。

        Args:
            container: 容器名，通常来自 :func:`cluster_container_name`。

        Raises:
            StorageError: 底层执行器失败时。
        """
        self._runner(["stop", container])

    def start(self, container: str) -> None:
        """启动容器（等价 ``docker start <container>``）。

        Args:
            container: 容器名，通常来自 :func:`cluster_container_name`。

        Raises:
            StorageError: 底层执行器失败时。
        """
        self._runner(["start", container])
