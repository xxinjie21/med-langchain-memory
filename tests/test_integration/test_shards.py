"""``med_langchain_memory.testing.shards`` 单元测试。

分片表只依赖标准库，因此**不需要任何中间件**、也不需要 Docker：
正向用例直接校验分片表自身的不变量（用例路径存在、服务名合法、
集群分片覆盖 6 个节点与初始化容器），边界用例覆盖「未选中任何分片」
「分片名拼错」「服务名与服务表脱节」「环境变量被覆盖」等分支。

每个公开成员均含正向与边界用例：``Shard.required`` / ``Shard.timeout_minutes`` / ``SHARDS`` /
``shard_names`` / ``resolve_shard`` / ``select_shards`` / ``required_services`` /
``render_paths`` / ``render_required`` / ``render_compose`` / ``render_init`` /
``render_env`` / ``render_table`` / ``render_matrix`` / ``build_parser`` / ``main``。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.testing import (
    ALL_SHARDS,
    CLUSTER_INITIALIZER,
    DEFAULT_TIMEOUT_MINUTES,
    ENV_PREFIX,
    ENV_REQUIRED,
    FORMATS,
    INTEGRATION_TEST_DIR,
    MATRIX_SHARD_KEY,
    MATRIX_TIMEOUT_KEY,
    REDIS_CLUSTER_NODE_PORTS,
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
from med_langchain_memory.testing.shards import build_parser, main

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 分片的模块入口（``python -m`` 调用形态）。
SHARDS_MODULE = "med_langchain_memory.testing.shards"

#: 期望的分片集合（与工作流矩阵一致，改这里就等于改契约）。
EXPECTED_SHARD_NAMES = (
    "redis",
    "mysql",
    "elasticsearch",
    "redis-cluster",
    "redis-cluster-failover",
)

#: 使用 Redis Cluster 的分片（各自拉起一套 6 节点集群）。
CLUSTER_SHARD_NAMES = ("redis-cluster", "redis-cluster-failover")

#: 服务表里的服务名（分片只能引用它们）。
KNOWN_SERVICES = ("redis", "mysql", "elasticsearch", "redis-cluster")


class TestShardTable:
    def test_shard_names_are_unique_and_complete(self) -> None:
        """正向：分片名不重复，且覆盖全部真机分片。"""
        names = shard_names()
        assert names == EXPECTED_SHARD_NAMES
        assert len(set(names)) == len(names)

    def test_every_shard_declares_paths_services_and_compose_services(self) -> None:
        """正向：每个分片都声明了用例路径、必需服务与 compose 服务名。"""
        for shard in SHARDS:
            assert shard.title, f"{shard.name} has no title"
            assert shard.pytest_paths, f"{shard.name} declares no pytest path"
            assert shard.services, f"{shard.name} declares no service"
            assert shard.compose_services, f"{shard.name} declares no compose service"
            assert set(shard.services) <= set(KNOWN_SERVICES)

    def test_every_pytest_path_exists_on_disk(self) -> None:
        """边界：分片表里的路径必须真实存在，否则 CI 会「收集到 0 个用例」而假绿。"""
        for shard in SHARDS:
            for relative in shard.pytest_paths:
                assert relative.startswith(f"{INTEGRATION_TEST_DIR}/")
                assert (PROJECT_ROOT / relative).is_file(), f"missing test file: {relative}"

    def test_required_property_is_comma_joined(self) -> None:
        """正向：``Shard.required`` 是逗号分隔的服务清单。"""
        assert resolve_shard("redis").required == "redis"
        assert resolve_shard("redis-cluster").required == "redis-cluster"

    def test_cluster_shard_covers_every_node_and_initializer(self) -> None:
        """正向：集群分片拉起全部 6 个节点，并带一次性初始化容器。"""
        cluster = resolve_shard("redis-cluster")
        expected = tuple(f"redis-cluster-{index}" for index in range(1, 7))
        assert cluster.compose_services == expected
        assert len(cluster.compose_services) == len(REDIS_CLUSTER_NODE_PORTS)
        assert cluster.initializer == CLUSTER_INITIALIZER

    def test_only_the_cluster_shards_need_an_initializer(self) -> None:
        """边界：只有集群分片声明初始化容器，其余分片不声明（工作流会少跑一步无用命令）。"""
        for shard in SHARDS:
            if shard.name in CLUSTER_SHARD_NAMES:
                assert shard.initializer == CLUSTER_INITIALIZER, shard.name
            else:
                assert shard.initializer is None, shard.name
        assert not {shard.name for shard in SHARDS if shard.initializer} - set(CLUSTER_SHARD_NAMES)

    def test_cluster_live_shard_runs_only_the_live_suite(self) -> None:
        """正向：只读集群分片只跑真机套件，停主节点的演练不属于它。"""
        paths = resolve_shard("redis-cluster").pytest_paths
        assert f"{INTEGRATION_TEST_DIR}/test_redis_cluster_live.py" in paths
        assert f"{INTEGRATION_TEST_DIR}/test_redis_cluster_failover.py" not in paths

    def test_failover_drill_is_its_own_shard(self) -> None:
        """正向：破坏性演练独立成片，不污染同批次的只读套件。"""
        drill = resolve_shard("redis-cluster-failover")
        assert drill.pytest_paths == (f"{INTEGRATION_TEST_DIR}/test_redis_cluster_failover.py",)
        assert drill.services == ("redis-cluster",)
        assert drill.compose_services == tuple(f"redis-cluster-{index}" for index in range(1, 7))

    def test_cluster_shards_together_cover_both_suites(self) -> None:
        """边界：两个集群分片的并集仍覆盖真机套件与演练，不重不漏。"""
        covered = [
            relative
            for name in CLUSTER_SHARD_NAMES
            for relative in resolve_shard(name).pytest_paths
        ]
        assert sorted(covered) == sorted(
            [
                f"{INTEGRATION_TEST_DIR}/test_redis_cluster_live.py",
                f"{INTEGRATION_TEST_DIR}/test_redis_cluster_failover.py",
            ]
        )


class TestResolveShard:
    def test_resolves_known_shard(self) -> None:
        """正向：按名字取到对应分片。"""
        assert resolve_shard("mysql").name == "mysql"

    def test_unknown_shard_raises_with_known_names(self) -> None:
        """异常：分片名拼错时抛 ``ValidationError``，且错误信息列出全部合法取值。"""
        with pytest.raises(ValidationError) as excinfo:
            resolve_shard("redys")
        message = str(excinfo.value)
        assert "redys" in message
        for name in EXPECTED_SHARD_NAMES:
            assert name in message


class TestSelectShards:
    def test_none_selects_every_shard(self) -> None:
        """正向：未指定分片时选中全部（保持分片表顺序）。"""
        assert select_shards() == SHARDS

    @pytest.mark.parametrize("names", [[], ["", "  "], [ALL_SHARDS]])
    def test_blank_or_all_selects_every_shard(self, names: list[str]) -> None:
        """边界：空序列 / 纯空白 / ``all`` 都等于「全部分片」。"""
        assert select_shards(names) == SHARDS

    def test_single_name_selects_one_shard(self) -> None:
        """正向：单个分片名只选中它自己。"""
        assert [shard.name for shard in select_shards(["elasticsearch"])] == ["elasticsearch"]

    def test_result_follows_table_order_not_argument_order(self) -> None:
        """边界：选中顺序跟随分片表，而不是命令行书写顺序（缓存键/表格才稳定）。"""
        selected = select_shards(["redis-cluster", "redis"])
        assert [shard.name for shard in selected] == ["redis", "redis-cluster"]

    def test_duplicate_and_padded_names_collapse(self) -> None:
        """边界：重复名与带空白的名字被归一化，不产生重复分片。"""
        assert [shard.name for shard in select_shards([" mysql ", "mysql"])] == ["mysql"]

    def test_unknown_name_raises(self) -> None:
        """异常：含未知分片名时抛 ``ValidationError``。"""
        with pytest.raises(ValidationError):
            select_shards(["redis", "oracle"])


class TestRequiredServices:
    def test_single_shard_lists_its_own_service(self) -> None:
        """正向：单分片的必需服务就是它声明的服务。"""
        assert required_services(select_shards(["mysql"])) == ("mysql",)

    def test_union_follows_service_table_order(self) -> None:
        """正向：合并多个分片时按服务表顺序输出，与手写清单逐字一致。"""
        assert required_services(SHARDS) == KNOWN_SERVICES

    def test_union_ignores_argument_order(self) -> None:
        """边界：即使传入顺序被打乱，输出顺序依旧稳定。"""
        reversed_shards = tuple(reversed(SHARDS))
        assert required_services(reversed_shards) == KNOWN_SERVICES

    def test_empty_selection_returns_empty_tuple(self) -> None:
        """边界：未选中任何分片时返回空元组。"""
        assert required_services(()) == ()

    def test_unknown_service_reference_raises(self) -> None:
        """异常：分片引用了服务表里没有的服务名时必须显式报错，而不是静默丢服务。"""
        bogus = Shard(
            name="oracle",
            title="bogus",
            pytest_paths=("tests/test_integration/test_redis_live.py",),
            services=("oracle",),
            compose_services=("oracle",),
        )
        with pytest.raises(ValidationError) as excinfo:
            required_services((bogus,))
        assert "oracle" in str(excinfo.value)


class TestRenderPlan:
    def test_paths_for_a_single_shard(self) -> None:
        """正向：单分片渲染出它的用例路径。"""
        assert render_paths(select_shards(["redis"])) == (
            f"{INTEGRATION_TEST_DIR}/test_redis_live.py"
        )

    def test_paths_are_deduplicated_across_shards(self) -> None:
        """边界：合并全部分片时路径不重复（去重保序）。"""
        paths = render_paths(SHARDS).split()
        assert len(paths) == len(set(paths))
        assert len(paths) == sum(len(shard.pytest_paths) for shard in SHARDS)

    def test_required_matches_the_service_table_order(self) -> None:
        """正向：必需服务清单与 D42 时代手写的 ``redis,mysql,elasticsearch,redis-cluster`` 相同。"""
        assert render_required(SHARDS) == "redis,mysql,elasticsearch,redis-cluster"

    def test_required_is_empty_without_shards(self) -> None:
        """边界：未选中分片时清单为空串。"""
        assert render_required(()) == ""

    def test_compose_lists_every_node_for_the_cluster_shard(self) -> None:
        """正向：集群分片渲染出 6 个节点服务名。"""
        assert render_compose(select_shards(["redis-cluster"])).split() == [
            f"redis-cluster-{index}" for index in range(1, 7)
        ]

    def test_compose_union_keeps_order_without_duplicates(self) -> None:
        """边界：合并全部分片时 compose 服务名去重保序。"""
        names = render_compose(SHARDS).split()
        assert names[:3] == ["redis", "mysql", "elasticsearch"]
        assert len(names) == len(set(names)) == 9

    def test_init_is_empty_for_non_cluster_shards(self) -> None:
        """边界：非集群分片没有初始化容器。"""
        assert render_init(select_shards(["redis", "mysql", "elasticsearch"])) == ""

    def test_init_is_the_cluster_container_for_the_cluster_shard(self) -> None:
        """正向：集群分片渲染出一次性初始化容器名。"""
        assert render_init(select_shards(["redis-cluster"])) == CLUSTER_INITIALIZER

    def test_init_deduplicates_when_the_cluster_shard_repeats(self) -> None:
        """边界：同一个集群分片重复出现时容器名只输出一次。"""
        cluster = resolve_shard("redis-cluster")
        assert render_init((cluster, cluster)) == CLUSTER_INITIALIZER

    def test_table_lists_every_shard_with_services_and_init(self) -> None:
        """正向：表格输出每个分片一行，含服务清单与初始化容器。"""
        table = render_table(SHARDS)
        lines = table.splitlines()
        assert len(lines) == len(SHARDS)
        for shard in SHARDS:
            assert any(line.startswith(shard.name) for line in lines)
        assert f"services=redis-cluster  init={CLUSTER_INITIALIZER}" in table
        assert "init=-" in table

    def test_table_shows_the_timeout_budget_per_shard(self) -> None:
        """正向：表格带超时预算列，与分片表声明逐字一致。"""
        table = render_table(SHARDS)
        for shard in SHARDS:
            assert f"timeout={shard.timeout_minutes}m" in table

    def test_table_without_shards_returns_a_placeholder(self) -> None:
        """边界：未选中分片时给出可读提示而不是抛异常。"""
        assert render_table(()) == "no shards selected"


class TestRenderEnv:
    def test_env_declares_switch_required_and_addresses(self) -> None:
        """正向：环境输出含总开关、必需服务清单与地址变量。"""
        lines = render_env(select_shards(["redis"])).splitlines()
        assert lines[0] == f"{ENV_PREFIX}=1"
        assert lines[1] == f"{ENV_REQUIRED}=redis"
        assert lines[2] == "MED_MEMORY_IT_REDIS_URL=redis://localhost:16379/15"

    def test_env_only_lists_the_selected_shard_services(self) -> None:
        """边界：只选中 MySQL 时不得出现其它服务的地址变量。"""
        text = render_env(select_shards(["mysql"]))
        assert "MED_MEMORY_IT_MYSQL_URL=" in text
        assert "MED_MEMORY_IT_REDIS_URL" not in text
        assert "MED_MEMORY_IT_ELASTICSEARCH_URL" not in text

    def test_env_union_covers_every_service(self) -> None:
        """正向：合并全部分片时四个地址变量齐全，且必需清单为四项。"""
        lines = render_env(SHARDS).splitlines()
        keys = [line.split("=", 1)[0] for line in lines]
        assert keys == [
            ENV_PREFIX,
            ENV_REQUIRED,
            "MED_MEMORY_IT_REDIS_URL",
            "MED_MEMORY_IT_MYSQL_URL",
            "MED_MEMORY_IT_ELASTICSEARCH_URL",
            "MED_MEMORY_IT_REDIS_CLUSTER_URL",
        ]
        assert lines[1] == f"{ENV_REQUIRED}=redis,mysql,elasticsearch,redis-cluster"

    def test_env_passes_local_overrides_through(self) -> None:
        """边界：本机覆盖了服务地址时原样透传（本地与 CI 行为一致）。"""
        env = {"MED_MEMORY_IT_REDIS_URL": "redis://example.internal:6390/3"}
        lines = render_env(select_shards(["redis"]), env).splitlines()
        assert "MED_MEMORY_IT_REDIS_URL=redis://example.internal:6390/3" in lines

    def test_env_lines_are_github_env_compatible(self) -> None:
        """边界：每行都是 ``KEY=VALUE`` 且键里无空白，可直接写进 ``$GITHUB_ENV``。"""
        for line in render_env(SHARDS).splitlines():
            key, separator, value = line.partition("=")
            assert separator == "="
            assert key and " " not in key
            assert value


class TestTimeoutBudget:
    """墙钟预算：每个分片声明自己的 ``timeout_minutes``，默认值统一兜底。"""

    def test_every_shard_declares_a_positive_timeout(self) -> None:
        """正向：每个分片的超时预算都是正整数，演练与 ES 分片比普通分片宽。"""
        for shard in SHARDS:
            assert isinstance(shard.timeout_minutes, int)
            assert shard.timeout_minutes > 0, shard.name
        assert (
            resolve_shard("elasticsearch").timeout_minutes > resolve_shard("redis").timeout_minutes
        )
        assert (
            resolve_shard("redis-cluster-failover").timeout_minutes
            > resolve_shard("redis").timeout_minutes
        )

    def test_default_timeout_used_when_not_declared(self) -> None:
        """边界：未显式声明时分片落到模块级默认值（构造替身即可验证）。"""
        bare = Shard(
            name="bare",
            title="bare",
            pytest_paths=("tests/test_integration/test_redis_live.py",),
            services=("redis",),
            compose_services=("redis",),
        )
        assert bare.timeout_minutes == DEFAULT_TIMEOUT_MINUTES
        assert DEFAULT_TIMEOUT_MINUTES > 0


class TestRenderMatrix:
    """``--format matrix``：把分片表渲染成 GitHub Actions 的矩阵 JSON。"""

    def test_matrix_pairs_shard_names_with_timeouts(self) -> None:
        """正向：两个数组等长、按索引配对，取值与分片表逐字一致。"""
        payload = json.loads(render_matrix(SHARDS))
        assert payload[MATRIX_SHARD_KEY] == list(shard_names())
        assert payload[MATRIX_TIMEOUT_KEY] == [shard.timeout_minutes for shard in SHARDS]
        assert len(payload[MATRIX_SHARD_KEY]) == len(payload[MATRIX_TIMEOUT_KEY]) == len(SHARDS)

    def test_matrix_is_a_single_line_of_json(self) -> None:
        """边界：单行输出才能安全嵌进 ``$GITHUB_OUTPUT`` 的 ``$(...)`` 替换。"""
        text = render_matrix(SHARDS)
        assert "\n" not in text
        assert text.startswith("{") and text.endswith("}")

    def test_matrix_of_a_single_shard(self) -> None:
        """正向：只选中一个分片时矩阵只有一个条目（本地调试形态）。"""
        payload = json.loads(render_matrix(select_shards(["redis"])))
        assert payload == {MATRIX_SHARD_KEY: ["redis"], MATRIX_TIMEOUT_KEY: [15]}

    def test_matrix_without_shards_has_empty_arrays(self) -> None:
        """边界：未选中分片时输出空数组而不是抛异常。"""
        assert json.loads(render_matrix(())) == {MATRIX_SHARD_KEY: [], MATRIX_TIMEOUT_KEY: []}


class TestCli:
    def test_default_format_is_table(self) -> None:
        """正向：未指定 ``--format`` 时默认输出表格。"""
        args = build_parser().parse_args([])
        assert args.format == "table"
        assert args.shard == []

    def test_all_documented_formats_are_accepted(self) -> None:
        """正向：``FORMATS`` 里的每一种都能被真实解析器接受。"""
        for name in FORMATS:
            assert build_parser().parse_args(["--format", name]).format == name

    def test_unknown_format_is_rejected(self) -> None:
        """异常：未知格式由 argparse 拒绝（退出码 2）。"""
        with pytest.raises(SystemExit) as excinfo:
            build_parser().parse_args(["--format", "yaml"])
        assert excinfo.value.code == 2

    @pytest.mark.parametrize("fmt", ["paths", "required", "compose", "init", "env", "matrix"])
    def test_main_prints_the_requested_format(
        self, fmt: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：``main`` 按格式打印并返回 0。"""
        assert main(["--shard", "redis-cluster", "--format", fmt]) == 0
        assert capsys.readouterr().out.strip()

    def test_main_table_lists_every_shard(self, capsys: pytest.CaptureFixture[str]) -> None:
        """正向：默认表格输出覆盖全部分片。"""
        assert main([]) == 0
        out = capsys.readouterr().out
        for name in EXPECTED_SHARD_NAMES:
            assert name in out

    def test_main_returns_two_for_an_unknown_shard(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """异常：分片名非法时返回用法错误码 2，错误写进 stderr。"""
        assert main(["--shard", "oracle"]) == 2
        captured = capsys.readouterr()
        assert "oracle" in captured.err
        assert captured.out == ""

    def test_main_defaults_argv_to_sys_argv(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``argv`` 为 ``None`` 时读取 ``sys.argv``（真实 CLI 调用路径）。"""
        monkeypatch.setattr(sys, "argv", ["med-memory-shards", "--format", "required"])
        assert main() == 0
        assert capsys.readouterr().out.strip() == "redis,mysql,elasticsearch,redis-cluster"

    def test_main_env_output_is_parseable_as_key_values(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``--format env`` 的输出可被 ``$GITHUB_ENV`` 直接消费。"""
        assert main(["--shard", "mysql", "--format", "env"]) == 0
        parsed = dict(line.split("=", 1) for line in capsys.readouterr().out.strip().splitlines())
        assert parsed[ENV_PREFIX] == "1"
        assert parsed[ENV_REQUIRED] == "mysql"
        assert parsed["MED_MEMORY_IT_MYSQL_URL"].startswith("mysql+pymysql://")

    @pytest.mark.parametrize(
        ("args", "expected"),
        [(["--shard", "redis", "--format", "required"], 0), (["--shard", "oracle"], 2)],
    )
    def test_module_entry_point_really_runs(self, args: list[str], expected: int) -> None:
        """正向 / 异常：``python -m ...shards`` 必须真的执行。

        这是**子进程级**的端到端校验：模块若漏写 ``if __name__ == "__main__"`` 守卫，
        ``-m`` 调用会「静默成功、什么都不做」——单元测试直接调 ``main()`` 永远发现不了。
        """
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        for key in list(env):
            if key.startswith(ENV_PREFIX):
                del env[key]
        result = subprocess.run(
            [sys.executable, "-m", SHARDS_MODULE, *args],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == expected, result.stderr
        if expected == 0:
            assert result.stdout.strip() == "redis"

    def test_module_entry_point_emits_matrix_json(self) -> None:
        """正向：子进程级校验 ``--format matrix`` 输出可被 ``json.loads`` 消费。"""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        for key in list(env):
            if key.startswith(ENV_PREFIX):
                del env[key]
        result = subprocess.run(
            [sys.executable, "-m", SHARDS_MODULE, "--format", "matrix"],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        assert payload[MATRIX_SHARD_KEY] == list(EXPECTED_SHARD_NAMES)
