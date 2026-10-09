"""``med_langchain_memory.testing.cluster`` 单元测试（离线，不需要 Docker / Redis）。

被测模块是故障转移演练的「拓扑解析 + 目标选择 + 容器控制 + 轮询」四件套，
全部为纯函数或可注入依赖的类，因此这里用**合成拓扑文本 + 假执行器 + 假时钟**
覆盖每条分支，不依赖任何真实中间件：

* :class:`ClusterNodeState` 的 6 个派生属性；
* :func:`parse_slots` / :func:`parse_cluster_nodes` 的正向与畸形输入；
* :func:`find_by_node_id` / :func:`cluster_node_index` / :func:`cluster_container_name`；
* :func:`plan_failover` 的选择规则与「无可用组合」边界；
* :func:`wait_until` 的立即成功 / 超时 / 单次探测 / 非正间隔；
* :func:`run_docker` 与 :class:`DockerContainerController`。
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

import pytest

from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.testing import (
    COMPOSE_PROJECT_NAME,
    REDIS_CLUSTER_NODE_PORTS,
    ClusterNodeState,
    DockerContainerController,
    cluster_container_name,
    cluster_node_index,
    find_by_node_id,
    parse_cluster_nodes,
    parse_slots,
    plan_failover,
    run_docker,
    wait_until,
)

HOST = "172.31.240.1"

MASTER_1 = "0" * 39 + "1"
MASTER_2 = "0" * 39 + "2"
MASTER_3 = "0" * 39 + "3"
REPLICA_1 = "0" * 39 + "4"
REPLICA_2 = "0" * 39 + "5"
REPLICA_3 = "0" * 39 + "6"

#: 三主三从的槽位划分（0–5460 / 5461–10922 / 10923–16383，16384 槽全覆盖）。
SLOT_LAYOUT = ("0-5460", "5461-10922", "10923-16383")


def node_line(
    node_id: str,
    port: int,
    flags: str,
    *,
    master: str = "-",
    slots: tuple[str, ...] = (),
    link_state: str = "connected",
    host: str = HOST,
) -> str:
    """拼一行 ``CLUSTER NODES`` 输出。"""
    fields = [
        node_id,
        f"{host}:{port}@{port}",
        flags,
        master,
        "0",
        "1700000000000",
        "1",
        link_state,
    ]
    return " ".join([*fields, *slots])


def topology(*, primary_1_flags: str = "myself,master") -> str:
    """构造三主三从的合成拓扑文本。"""
    return "\n".join(
        [
            node_line(MASTER_1, 17001, primary_1_flags, slots=(SLOT_LAYOUT[0],)),
            node_line(MASTER_2, 17002, "master", slots=(SLOT_LAYOUT[1],)),
            node_line(MASTER_3, 17003, "master", slots=(SLOT_LAYOUT[2],)),
            node_line(REPLICA_1, 17004, "slave", master=MASTER_1),
            node_line(REPLICA_2, 17005, "slave", master=MASTER_2),
            node_line(REPLICA_3, 17006, "slave", master=MASTER_3),
        ]
    )


class TestClusterNodeState:
    """节点状态的派生属性。"""

    def test_primary_flags_and_slots(self) -> None:
        """正向：主节点识别 + 槽位求和 + 地址拼接。"""
        node = parse_cluster_nodes(topology())[0]
        assert node.address == f"{HOST}:17001"
        assert node.is_primary is True
        assert node.is_replica is False
        assert node.is_failed is False
        assert node.owns_slots is True
        assert node.slot_count == 5461
        assert node.master_id is None

    def test_replica_flags_and_master_pointer(self) -> None:
        """正向：从节点识别 + ``master_id`` 指向所属主节点。"""
        node = parse_cluster_nodes(topology())[3]
        assert node.is_primary is False
        assert node.is_replica is True
        assert node.master_id == MASTER_1
        assert node.owns_slots is False
        assert node.slot_count == 0

    def test_fail_flag_distinguishes_pfail(self) -> None:
        """边界：``fail?``（待定故障）不算 ``fail``，避免误判可用节点。"""
        pending = ClusterNodeState(
            node_id=MASTER_1,
            host=HOST,
            port=17001,
            flags=("myself", "master", "fail?"),
            master_id=None,
            slots=((0, 5460),),
            link_state="connected",
        )
        failed = ClusterNodeState(
            node_id=MASTER_1,
            host=HOST,
            port=17001,
            flags=("master", "fail"),
            master_id=None,
            slots=((0, 5460),),
            link_state="disconnected",
        )
        assert pending.is_failed is False
        assert failed.is_failed is True


class TestParseSlots:
    """槽位标记解析。"""

    def test_parses_ranges_and_single_slots(self) -> None:
        """正向：区间与单槽位都解析为闭区间。"""
        assert parse_slots(["0-5460", "5461"]) == ((0, 5460), (5461, 5461))

    def test_empty_tokens_yield_empty_tuple(self) -> None:
        """边界：从节点没有槽位标记时返回空元组。"""
        assert parse_slots([]) == ()

    def test_skips_migration_markers(self) -> None:
        """边界：迁移中的 ``[slot->-id]`` / ``[slot-<-id]`` 标记被跳过，不计入稳定归属。"""
        assert parse_slots(["[4096->-abc]", "[4097-<-def]", "0-5460"]) == ((0, 5460),)

    @pytest.mark.parametrize("token", ["abc", "0-xyz", "-"])
    def test_rejects_malformed_token(self, token: str) -> None:
        """异常：无法解析的槽位标记直接拒绝。"""
        with pytest.raises(ValidationError, match="malformed slot token"):
            parse_slots([token])


class TestParseClusterNodes:
    """整段拓扑文本解析。"""

    def test_parses_all_nodes_in_order(self) -> None:
        """正向：6 个节点全部解析，顺序与输入一致。"""
        nodes = parse_cluster_nodes(topology())
        assert [node.port for node in nodes] == list(REDIS_CLUSTER_NODE_PORTS)
        assert [node.node_id for node in nodes] == [
            MASTER_1,
            MASTER_2,
            MASTER_3,
            REPLICA_1,
            REPLICA_2,
            REPLICA_3,
        ]

    def test_accepts_bytes_payload(self) -> None:
        """正向：未开 ``decode_responses`` 时的 ``bytes`` 回复同样可解析。"""
        nodes = parse_cluster_nodes(topology().encode("utf-8"))
        assert len(nodes) == 6

    def test_ignores_blank_lines(self) -> None:
        """边界：首尾空行与行内空白被忽略，不产生空节点。"""
        payload = "\n\n" + topology().replace("\n", "\n\n") + "\n\n"
        assert len(parse_cluster_nodes(payload)) == 6

    def test_empty_payload_yields_empty_tuple(self) -> None:
        """边界：空输出解析为空元组（集群尚未收敛时的真实回复）。"""
        assert parse_cluster_nodes("") == ()
        assert parse_cluster_nodes(b"") == ()

    def test_rejects_line_with_too_few_fields(self) -> None:
        """异常：字段不足 8 个的行直接拒绝，避免产出半残节点。"""
        with pytest.raises(ValidationError, match="malformed CLUSTER NODES line"):
            parse_cluster_nodes("abc def ghi")

    def test_rejects_malformed_address(self) -> None:
        """异常：地址缺少端口时拒绝（避免把空主机当成有效节点）。"""
        line = node_line(MASTER_1, 17001, "master", slots=(SLOT_LAYOUT[0],))
        broken = line.replace(f"{HOST}:17001@17001", f"{HOST}@17001")
        with pytest.raises(ValidationError, match="malformed node address"):
            parse_cluster_nodes(broken)


class TestFindByNodeId:
    """按节点 ID 查找。"""

    def test_returns_node_when_present(self) -> None:
        """正向：命中时返回对应节点。"""
        nodes = parse_cluster_nodes(topology())
        found = find_by_node_id(nodes, REPLICA_3)
        assert found is not None
        assert found.port == 17006

    def test_returns_none_when_absent(self) -> None:
        """边界：未命中时返回 ``None``，由调用方决定跳过还是失败。"""
        assert find_by_node_id(parse_cluster_nodes(topology()), "deadbeef") is None


class TestClusterNodeIndex:
    """端口 → 节点序号。"""

    @pytest.mark.parametrize(("port", "index"), [(17001, 1), (17003, 3), (17006, 6)])
    def test_maps_managed_ports(self, port: int, index: int) -> None:
        """正向：内置 6 个端口映射为 1–6 的序号。"""
        assert cluster_node_index(port) == index

    def test_rejects_unmanaged_port(self) -> None:
        """异常：自建集群的端口无法映射到容器，必须显式拒绝。"""
        with pytest.raises(ValidationError, match="not one of the integration cluster node ports"):
            cluster_node_index(6379)


class TestClusterContainerName:
    """序号 → 容器名。"""

    def test_names_follow_compose_convention(self) -> None:
        """正向：容器名 = 工程名 + 服务名，与编排里的 ``container_name`` 一致。"""
        assert cluster_container_name(1) == f"{COMPOSE_PROJECT_NAME}-redis-cluster-1"
        assert cluster_container_name(6) == f"{COMPOSE_PROJECT_NAME}-redis-cluster-6"

    @pytest.mark.parametrize("index", [0, 7, -1])
    def test_rejects_out_of_range_index(self, index: int) -> None:
        """异常：越界序号拒绝，避免拼出指向不存在容器的名字。"""
        with pytest.raises(ValidationError, match="out of range"):
            cluster_container_name(index)


class TestPlanFailover:
    """演练目标选择。"""

    def test_picks_lowest_port_primary_with_its_replica(self) -> None:
        """正向：确定性选出端口最小的主节点与其从节点，并解析出容器名。"""
        plan = plan_failover(parse_cluster_nodes(topology()))
        assert plan is not None
        assert plan.primary.port == 17001
        assert plan.replica.port == 17004
        assert plan.primary_container == cluster_container_name(1)
        assert plan.replica_container == cluster_container_name(4)

    def test_skips_failed_primary(self) -> None:
        """边界：已被标记 ``fail`` 的主节点不参与演练，顺延到下一个主节点。"""
        plan = plan_failover(parse_cluster_nodes(topology(primary_1_flags="master,fail")))
        assert plan is not None
        assert plan.primary.port == 17002
        assert plan.replica.port == 17005

    def test_skips_disconnected_replica(self) -> None:
        """边界：链路断开的从节点不能作为接管目标，顺延到下一个主节点。"""
        payload = "\n".join(
            [
                node_line(MASTER_1, 17001, "myself,master", slots=(SLOT_LAYOUT[0],)),
                node_line(REPLICA_1, 17004, "slave", master=MASTER_1, link_state="disconnected"),
                node_line(MASTER_2, 17002, "master", slots=(SLOT_LAYOUT[1],)),
                node_line(REPLICA_2, 17005, "slave", master=MASTER_2),
            ]
        )
        plan = plan_failover(parse_cluster_nodes(payload))
        assert plan is not None
        assert plan.primary.port == 17002
        assert plan.replica.port == 17005

    def test_ignores_nodes_outside_the_managed_port_set(self) -> None:
        """边界：自建集群的节点无法定位容器，整体返回 ``None``。"""
        payload = "\n".join(
            [
                node_line(MASTER_1, 7001, "myself,master", slots=(SLOT_LAYOUT[0],)),
                node_line(REPLICA_1, 7004, "slave", master=MASTER_1),
            ]
        )
        assert plan_failover(parse_cluster_nodes(payload)) is None

    def test_returns_none_when_no_replica_available(self) -> None:
        """边界：只有主节点（从节点已下线）时返回 ``None``，用例据此跳过。"""
        payload = "\n".join(
            [
                node_line(MASTER_1, 17001, "myself,master", slots=(SLOT_LAYOUT[0],)),
                node_line(MASTER_2, 17002, "master", slots=(SLOT_LAYOUT[1],)),
            ]
        )
        assert plan_failover(parse_cluster_nodes(payload)) is None

    def test_returns_none_for_empty_topology(self) -> None:
        """边界：空拓扑（尚未收敛）返回 ``None``，不抛异常。"""
        assert plan_failover(()) is None


class TestWaitUntil:
    """通用轮询（注入假时钟，避免真实等待）。"""

    def test_returns_true_immediately_when_predicate_holds(self) -> None:
        """正向：首次判定即为真时不休眠。"""
        slept: list[float] = []
        assert wait_until(lambda: True, clock=lambda: 0.0, sleep=slept.append) is True
        assert slept == []

    def test_retries_until_predicate_becomes_true(self) -> None:
        """正向：判定失败时按间隔重试，直到为真。"""
        state = {"now": 0.0}
        calls: list[float] = []

        def predicate() -> bool:
            calls.append(state["now"])
            return len(calls) >= 3

        assert (
            wait_until(
                predicate,
                timeout=10.0,
                interval=1.0,
                clock=lambda: state["now"],
                sleep=lambda seconds: state.__setitem__("now", state["now"] + seconds),
            )
            is True
        )
        assert len(calls) == 3

    def test_returns_false_when_deadline_passes(self) -> None:
        """边界：超时前始终为假则返回 ``False``，且不越过截止时间。"""
        state = {"now": 0.0}
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            state["now"] += seconds

        assert (
            wait_until(
                lambda: False,
                timeout=3.0,
                interval=1.0,
                clock=lambda: state["now"],
                sleep=sleep,
            )
            is False
        )
        assert slept == [1.0, 1.0, 1.0]

    def test_non_positive_timeout_probes_once(self) -> None:
        """边界：``timeout<=0`` 只判定一次，绝不进入等待循环。"""
        calls: list[int] = []
        assert (
            wait_until(
                lambda: bool(calls.append(1)) and False,
                timeout=0.0,
                clock=lambda: 0.0,
                sleep=lambda seconds: None,
            )
            is False
        )
        assert len(calls) == 1

    def test_non_positive_interval_never_sleeps(self) -> None:
        """边界：``interval<=0`` 时以 0 秒休眠（不死等），仍按截止时间退出。"""
        state = {"now": 0.0}
        slept: list[float] = []

        def sleep(seconds: float) -> None:
            slept.append(seconds)
            state["now"] += 1.0

        assert (
            wait_until(
                lambda: False,
                timeout=2.0,
                interval=0.0,
                clock=lambda: state["now"],
                sleep=sleep,
            )
            is False
        )
        assert slept == [0.0, 0.0]


class TestRunDocker:
    """``docker`` CLI 调用封装。"""

    def test_returns_stdout_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正向：成功时返回标准输出，并以 ``check=False`` 自行判定退出码。"""
        recorded: dict[str, object] = {}

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            recorded["command"] = command
            recorded["kwargs"] = kwargs
            return subprocess.CompletedProcess(command, 0, stdout="c1\n", stderr="")

        monkeypatch.setattr("med_langchain_memory.testing.cluster.subprocess.run", fake_run)
        assert run_docker(["stop", "c1"]) == "c1\n"
        assert recorded["command"] == ["docker", "stop", "c1"]
        assert recorded["kwargs"] == {
            "capture_output": True,
            "text": True,
            "timeout": 60.0,
            "check": False,
        }

    def test_raises_on_non_zero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """异常：退出码非 0 时抛 ``StorageError``，消息里带 stderr 明细。"""
        monkeypatch.setattr(
            "med_langchain_memory.testing.cluster.subprocess.run",
            lambda command, **kwargs: subprocess.CompletedProcess(
                command, 1, stdout="", stderr="No such container"
            ),
        )
        with pytest.raises(StorageError, match="exit 1.*No such container"):
            run_docker(["stop", "c1"])

    def test_falls_back_to_stdout_when_stderr_is_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """边界：stderr 为空时用 stdout 作为明细，避免错误信息为空。"""
        monkeypatch.setattr(
            "med_langchain_memory.testing.cluster.subprocess.run",
            lambda command, **kwargs: subprocess.CompletedProcess(
                command, 125, stdout="daemon not running", stderr=""
            ),
        )
        with pytest.raises(StorageError, match="daemon not running"):
            run_docker(["stop", "c1"])

    def test_wraps_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """异常：``docker`` 可执行文件缺失（``OSError``）时包装为 ``StorageError``。"""

        def missing_docker(command: list[str], **kwargs: object) -> None:
            raise FileNotFoundError("docker")

        monkeypatch.setattr("med_langchain_memory.testing.cluster.subprocess.run", missing_docker)
        with pytest.raises(StorageError, match="failed to run docker stop c1"):
            run_docker(["stop", "c1"])

    def test_wraps_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """异常：命令超时（``SubprocessError``）时包装为 ``StorageError``。"""

        def slow_docker(command: list[str], **kwargs: object) -> None:
            raise subprocess.TimeoutExpired(command, timeout=1.0)

        monkeypatch.setattr("med_langchain_memory.testing.cluster.subprocess.run", slow_docker)
        with pytest.raises(StorageError, match="failed to run docker start c2"):
            run_docker(["start", "c2"])


class TestDockerContainerController:
    """容器停 / 起编排。"""

    def test_issues_stop_and_start_subcommands(self) -> None:
        """正向：``stop`` / ``start`` 分别拼出正确的子命令序列。"""
        calls: list[list[str]] = []

        def runner(args: Sequence[str]) -> str:
            calls.append(list(args))
            return "ok"

        controller = DockerContainerController(runner=runner)
        controller.stop("c1")
        controller.start("c2")
        assert calls == [["stop", "c1"], ["start", "c2"]]

    def test_propagates_runner_failure(self) -> None:
        """异常：执行器抛错时原样上抛，便于用例感知演练环境异常。"""

        def failing(args: Sequence[str]) -> str:
            raise StorageError("docker is not reachable")

        with pytest.raises(StorageError, match="docker is not reachable"):
            DockerContainerController(runner=failing).stop("c1")

    def test_default_runner_uses_the_docker_cli(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正向：不注入执行器时走真实 CLI 封装（此处替换 ``subprocess.run`` 拦截）。"""
        recorded: list[list[str]] = []

        def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            recorded.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("med_langchain_memory.testing.cluster.subprocess.run", fake_run)
        DockerContainerController().stop(cluster_container_name(2))
        assert recorded == [["docker", "stop", cluster_container_name(2)]]
