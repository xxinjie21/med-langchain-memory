"""集成测试分片：把真机套件按服务切成互不依赖的并行作业。

动机：D37–D42 的集成流水线是**单个作业串行跑全部真机套件**，墙钟时间等于
四个服务的启动时间与用例耗时之和——Redis 秒级、MySQL 数十秒、Elasticsearch
冷启动最慢、6 节点 Redis Cluster 还要额外组集群。本模块把「每个分片跑哪些用例、
需要哪些服务、拉起哪些容器、注入哪些环境变量」收敛成**纯标准库的数据 + 命令行**，
工作流只按分片矩阵调用它，于是四个分片并行执行、每个分片只拉起自己需要的容器，
墙钟时间由「最慢的那个分片」决定，而不是「全部服务串行之和」。

设计取舍：

* **单一真值来源**：用例路径、必需服务、compose 服务名、服务地址环境变量、矩阵取值与
  超时预算全部由本模块生成（服务地址复用 :mod:`med_langchain_memory.testing.services`
  的默认值），工作流里**不再出现任何 ``MED_MEMORY_IT_*_URL`` 字面量、也不再出现分片名清单**，
  杜绝两处漂移；
* **只依赖标准库**（外加同样只依赖标准库的 ``testing.services`` / ``testing.preflight``），
  不做协议握手、不做任何文本处理；
* 集群分片的一次性初始化容器（``redis-cluster-init``）也由分片表描述，
  工作流不硬编码分片名——新增分片只需改 :data:`SHARDS`。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from med_langchain_memory.exceptions import ValidationError

from .preflight import ENV_REQUIRED
from .services import DEFAULT_SERVICES, ENV_PREFIX, REDIS_CLUSTER_NODE_PORTS, resolve_service

#: 集成用例所在目录（相对仓库根），分片表的路径都以此为前缀。
INTEGRATION_TEST_DIR = "tests/test_integration"

#: ``--shard`` 的快捷取值：选中全部分片。
ALL_SHARDS = "all"

#: ``--format`` 支持的全部取值。
FORMATS: tuple[str, ...] = ("table", "paths", "required", "compose", "init", "env", "matrix")

#: 集群分片的一次性初始化容器名（与 ``docker-compose.integration.yml`` 一致）。
CLUSTER_INITIALIZER = "redis-cluster-init"

#: 分片未显式声明超时预算时使用的默认值（分钟）。
DEFAULT_TIMEOUT_MINUTES = 20

#: 矩阵 JSON 里的分片名列表键（工作流 ``matrix.shard``）。
MATRIX_SHARD_KEY = "shard"

#: 矩阵 JSON 里的超时预算列表键（工作流 ``matrix.timeout_minutes``）。
MATRIX_TIMEOUT_KEY = "timeout_minutes"


def _cluster_compose_services() -> tuple[str, ...]:
    """返回集群分片需要拉起的 compose 服务名。

    Returns:
        形如 ``("redis-cluster-1", ..., "redis-cluster-6")`` 的元组，
        节点数量与 :data:`~med_langchain_memory.testing.services.REDIS_CLUSTER_NODE_PORTS` 对齐。
    """
    return tuple(f"redis-cluster-{index}" for index in range(1, len(REDIS_CLUSTER_NODE_PORTS) + 1))


@dataclass(frozen=True)
class Shard:
    """一个集成测试分片：一组用例 + 它需要的中间件服务。"""

    #: 分片名（同时用作矩阵键、缓存键前缀与 artifact 名后缀）。
    name: str

    #: 人类可读的分片标题。
    title: str

    #: 该分片要执行的 pytest 目标路径（相对仓库根）。
    pytest_paths: tuple[str, ...]

    #: 该分片依赖的中间件服务名（即 ``MED_MEMORY_IT_REQUIRED`` 的取值）。
    services: tuple[str, ...]

    #: 该分片需要 ``docker compose up`` 拉起的服务名。
    compose_services: tuple[str, ...]

    #: 该分片的墙钟预算（分钟），直接映射到工作流的 ``timeout-minutes``。
    timeout_minutes: int = DEFAULT_TIMEOUT_MINUTES

    #: 拉起后需要执行的一次性初始化容器名；无需初始化时为 ``None``。
    initializer: str | None = None

    @property
    def required(self) -> str:
        """该分片的必需服务清单（逗号分隔，可直接喂给 ``MED_MEMORY_IT_REQUIRED``）。"""
        return ",".join(self.services)


#: 分片表：**单一真值来源**。新增分片只需在这里加一行。
SHARDS: tuple[Shard, ...] = (
    Shard(
        name="redis",
        title="Redis single node",
        pytest_paths=(f"{INTEGRATION_TEST_DIR}/test_redis_live.py",),
        services=("redis",),
        compose_services=("redis",),
        timeout_minutes=15,
    ),
    Shard(
        name="mysql",
        title="MySQL sharded store",
        pytest_paths=(f"{INTEGRATION_TEST_DIR}/test_mysql_live.py",),
        services=("mysql",),
        compose_services=("mysql",),
        timeout_minutes=20,
    ),
    Shard(
        name="elasticsearch",
        title="Elasticsearch archive",
        pytest_paths=(f"{INTEGRATION_TEST_DIR}/test_es_live.py",),
        services=("elasticsearch",),
        compose_services=("elasticsearch",),
        timeout_minutes=30,
    ),
    Shard(
        name="redis-cluster",
        title="Redis Cluster live suite",
        pytest_paths=(f"{INTEGRATION_TEST_DIR}/test_redis_cluster_live.py",),
        services=("redis-cluster",),
        compose_services=_cluster_compose_services(),
        timeout_minutes=30,
        initializer=CLUSTER_INITIALIZER,
    ),
    # 破坏性演练（停主节点）单独成片：它自己拉起一套集群，因此不会污染同批次的
    # 只读套件；也因为它最慢，独立成片后不再拖住别的分片。
    Shard(
        name="redis-cluster-failover",
        title="Redis Cluster failover drill",
        pytest_paths=(f"{INTEGRATION_TEST_DIR}/test_redis_cluster_failover.py",),
        services=("redis-cluster",),
        compose_services=_cluster_compose_services(),
        timeout_minutes=25,
        initializer=CLUSTER_INITIALIZER,
    ),
)

_SHARDS_BY_NAME = {shard.name: shard for shard in SHARDS}


def shard_names() -> tuple[str, ...]:
    """返回全部分片名。

    Returns:
        分片名元组，顺序与 :data:`SHARDS` 一致。
    """
    return tuple(shard.name for shard in SHARDS)


def resolve_shard(name: str) -> Shard:
    """按名字取分片。

    Args:
        name: 分片名，取值见 :func:`shard_names`。

    Returns:
        对应的 :class:`Shard`。

    Raises:
        ValidationError: 分片名未知时（错误信息里列出全部合法取值）。
    """
    shard = _SHARDS_BY_NAME.get(name)
    if shard is None:
        known = ", ".join(shard_names())
        raise ValidationError(f"unknown integration shard {name!r}; known: {known}")
    return shard


def select_shards(names: Sequence[str] | None = None) -> tuple[Shard, ...]:
    """把命令行给定的分片名解析成分片序列。

    Args:
        names: 分片名序列；``None`` / 空序列 / 含 :data:`ALL_SHARDS` 时返回全部分片。

    Returns:
        分片元组，顺序与 :data:`SHARDS` 一致（重复名自动去重）。

    Raises:
        ValidationError: 含未知分片名时。
    """
    requested = [name.strip() for name in (names or ()) if name.strip()]
    if not requested or ALL_SHARDS in requested:
        return SHARDS
    selected = {resolve_shard(name).name for name in requested}
    return tuple(shard for shard in SHARDS if shard.name in selected)


def required_services(shards: Sequence[Shard]) -> tuple[str, ...]:
    """汇总分片依赖的中间件服务名。

    Args:
        shards: 分片序列。

    Returns:
        去重后的服务名元组，顺序与
        :data:`~med_langchain_memory.testing.services.DEFAULT_SERVICES` 一致
        （因此 ``--format required`` 的输出与 D42 时代手写的清单逐字相同）。

    Raises:
        ValidationError: 分片引用了 :data:`DEFAULT_SERVICES` 里没有的服务名时
            （说明分片表与服务表脱节，必须显式报错而不是静默丢服务）。
    """
    seen: dict[str, None] = {}
    for shard in shards:
        for name in shard.services:
            seen.setdefault(name, None)
    known = [service.name for service in DEFAULT_SERVICES]
    unknown = [name for name in seen if name not in known]
    if unknown:
        raise ValidationError(f"shards reference unknown services: {', '.join(unknown)}")
    return tuple(name for name in known if name in seen)


def _join_unique(values: Iterable[str]) -> str:
    """把字符串序列去重后以空格连接（保序）。"""
    return " ".join(dict.fromkeys(values))


def render_paths(shards: Sequence[Shard]) -> str:
    """渲染成空格分隔的 pytest 目标路径。

    Args:
        shards: 分片序列。

    Returns:
        去重保序的路径串，可直接拼在 ``pytest -m integration`` 之后。
    """
    return _join_unique(path for shard in shards for path in shard.pytest_paths)


def render_required(shards: Sequence[Shard]) -> str:
    """渲染成逗号分隔的必需服务清单（``MED_MEMORY_IT_REQUIRED`` 的取值）。

    Args:
        shards: 分片序列。

    Returns:
        形如 ``redis,mysql`` 的清单；未选中任何分片时为空串。
    """
    return ",".join(required_services(shards))


def render_compose(shards: Sequence[Shard]) -> str:
    """渲染成空格分隔的 compose 服务名。

    Args:
        shards: 分片序列。

    Returns:
        去重保序的服务名串，可直接拼在 ``docker compose up -d --wait`` 之后。
    """
    return _join_unique(name for shard in shards for name in shard.compose_services)


def render_init(shards: Sequence[Shard]) -> str:
    """渲染一次性初始化容器名。

    Args:
        shards: 分片序列。

    Returns:
        去重保序的容器名串；无初始化需求时为空串（工作流据此判断是否要执行 ``run --rm``）。
    """
    return _join_unique(shard.initializer for shard in shards if shard.initializer is not None)


def render_env(shards: Sequence[Shard], env: Mapping[str, str] | None = None) -> str:
    """渲染该分片需要注入的环境变量（``KEY=VALUE`` 多行文本）。

    Args:
        shards: 分片序列。
        env: 环境变量映射；``None`` 时读取 ``os.environ``。本机若覆盖了某个
            服务地址，这里会原样透传，保证本地与 CI 行为一致。

    Returns:
        依次为总开关、必需服务清单、以及每个服务的地址变量，
        可直接 ``>> "$GITHUB_ENV"``。
    """
    lines: list[tuple[str, str]] = [
        (ENV_PREFIX, "1"),
        (ENV_REQUIRED, render_required(shards)),
    ]
    for name in required_services(shards):
        service = resolve_service(name, env)
        lines.append((service.env_var, service.url))
    return "\n".join(f"{key}={value}" for key, value in lines)


def render_matrix(shards: Sequence[Shard]) -> str:
    """渲染成 GitHub Actions 的 ``strategy.matrix`` JSON（单行）。

    Args:
        shards: 分片序列。

    Returns:
        形如 ``{"shard":["redis",...],"timeout_minutes":[15,...]}`` 的单行 JSON，
        两个数组等长且按索引配对，可直接喂给 ``matrix: ${{ fromJSON(...) }}``。
    """
    payload = {
        MATRIX_SHARD_KEY: [shard.name for shard in shards],
        MATRIX_TIMEOUT_KEY: [shard.timeout_minutes for shard in shards],
    }
    return json.dumps(payload, separators=(",", ":"))


def render_table(shards: Sequence[Shard]) -> str:
    """渲染人可读的分片清单（``--format table``）。

    Args:
        shards: 分片序列。

    Returns:
        每行一个分片：``名称  服务  init  超时  用例路径``；未选中任何分片时返回提示行。
    """
    if not shards:
        return "no shards selected"
    width = max(len(shard.name) for shard in shards)
    return "\n".join(
        f"{shard.name.ljust(width)}  services={shard.required}"
        f"  init={shard.initializer or '-'}  timeout={shard.timeout_minutes}m"
        f"  paths={' '.join(shard.pytest_paths)}"
        for shard in shards
    )


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（供 CI 与本地直接调用）。"""
    parser = argparse.ArgumentParser(
        prog="med-memory-shards",
        description="Resolve integration test shards into pytest paths, services and env vars.",
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        metavar="NAME",
        help=(
            f"shard to select; repeatable; omit or pass '{ALL_SHARDS}' for every shard "
            f"(known: {', '.join(shard_names())})"
        ),
    )
    parser.add_argument(
        "--format",
        choices=FORMATS,
        default="table",
        help="output format (default: table)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：按格式打印分片计划并返回退出码。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示正常输出；``2`` 表示分片名非法（用法错误）。
    """
    args = build_parser().parse_args(argv)
    try:
        shards = select_shards(args.shard)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    renderers: dict[str, Callable[[Sequence[Shard]], str]] = {
        "table": render_table,
        "paths": render_paths,
        "required": render_required,
        "compose": render_compose,
        "init": render_init,
        "env": render_env,
        "matrix": render_matrix,
    }
    print(renderers[args.format](shards))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
