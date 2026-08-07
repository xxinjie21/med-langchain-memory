"""Redis 集群存储适配器 :class:`RedisClusterMedHistory`。

与单机版 :class:`RedisMedHistory` 的唯一差异在**键布局**：用 ``{}`` 把
``session_id`` 包成 Redis Cluster 的 hash tag，使得属于同一会话的两条键
（消息 List 与元数据 Hash）都落在同一个 slot，从而满足集群事务 pipeline
的单节点约束。存储逻辑（pipeline 批量写、元数据哈希累加、会话级 TTL 等）全部复用父类；
同 slot 保证也让 ``EXPIRE`` 两条键可以放在同一个事务 pipeline 里下发。

``client`` 必须是已建好的 :class:`redis.RedisCluster`（连接池由其构造函数配置，
见 :func:`build_cluster_client`）；本类构造期**不发起任何网络请求**。
该客户端**不得**开启 ``decode_responses``，否则 protobuf 二进制会被破坏。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from typing import Any, ClassVar, cast

from redis import Redis, RedisCluster
from redis.cluster import ClusterNode

from med_langchain_memory.exceptions import StorageError

from .factory import StoreFactory
from .redis_store import MESSAGES_SUFFIX, META_SUFFIX, RedisMedHistory

#: 启动节点描述：``{"host": str, "port": int}``。
StartupNode = dict[str, Any]


def build_cluster_client(
    startup_nodes: list[StartupNode],
    *,
    max_connections: int = 16,
    decode_responses: bool = False,
) -> RedisCluster:
    """构造一个连接池可配的 Redis 集群客户端。

    Args:
        startup_nodes: 启动节点列表，每项形如 ``{"host": "127.0.0.1", "port": 6379}``。
        max_connections: 单节点连接池的最大连接数（连接池配置入口）。
        decode_responses: 必须为 ``False``，消息体为 protobuf 二进制。

    Returns:
        配置好连接池的 :class:`redis.RedisCluster` 实例（惰性建连，首次命令才触达节点）。

    Raises:
        StorageError: 启用了 ``decode_responses`` 时。
    """
    if decode_responses:
        raise StorageError("redis cluster client must not enable decode_responses")
    nodes = [
        ClusterNode(str(node["host"]), int(node["port"]))  # type: ignore[no-untyped-call]
        for node in startup_nodes
    ]
    return RedisCluster(startup_nodes=nodes, max_connections=max_connections)  # type: ignore[no-any-return]


@StoreFactory.register("redis-cluster")
class RedisClusterMedHistory(RedisMedHistory):
    """Redis 集群会话历史，注册名 ``redis-cluster``。

    示例：
        >>> import fakeredis
        >>> history = RedisClusterMedHistory(
        ...     session_id="s-1", tenant_id="hosp-a", dept_id="cardio",
        ...     patient_id="p-1", client=fakeredis.FakeRedis(),
        ... )
        >>> history.get_med_messages()
        []
    """

    #: 与单机版一致：hash tag 保证两条键同 slot，可直接复用父类的原生 TTL 实现。
    supports_ttl: ClassVar[bool] = True

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        client: RedisCluster,
        ttl_seconds: int | None = None,
        renew_on_read: bool = False,
    ) -> None:
        """初始化 Redis 集群会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            client: 已建好且连接池已配置的 Redis 集群客户端。
            ttl_seconds: 会话级 TTL（秒）；``None`` 表示不设过期。
            renew_on_read: 读取是否也参与滑动续期；默认仅写入续期。

        Raises:
            ValidationError: ID 不合法或 ``ttl_seconds`` 非正数时。
            StorageError: 未传入 ``client``，或客户端开启了 ``decode_responses`` 时。
        """
        if client is None:
            raise StorageError(
                "redis-cluster backend requires an explicit `client` "
                "(a configured redis.RedisCluster); build one via build_cluster_client(...)"
            )
        # TTL 延后到重写键名之后再下发，否则 EXPIRE 会打在未加 tag 的旧键上。
        super().__init__(
            session_id,
            tenant_id,
            dept_id,
            patient_id,
            client=cast("Redis", client),
            renew_on_read=renew_on_read,
        )
        # 用 hash tag 包裹 session_id，确保 messages 与 meta 键落到同一 slot。
        tagged = f"med:chat:{self.tenant_id}:{self.dept_id}:{{{self.session_id}}}"
        self._messages_key = f"{tagged}{MESSAGES_SUFFIX}"
        self._meta_key = f"{tagged}{META_SUFFIX}"
        if ttl_seconds is not None:
            self.set_ttl(ttl_seconds)
