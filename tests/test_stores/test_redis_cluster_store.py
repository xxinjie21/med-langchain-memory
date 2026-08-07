"""RedisClusterMedHistory 单元测试。

分四部分：

* :class:`TestClusterStoreBehavior` 复用跨后端共享行为套件，校验通用存储契约；
* :class:`TestHashTagKeys` 校验集群键布局：用 ``{session_id}`` 作 hash tag，
  使会话内 messages 与 meta 同 slot，而 tenant/dept 不被包进 tag（避免整租户塌缩到单 slot）；
* :class:`TestBuildClusterClient` 校验连接池配置入口（max_connections 透传、拒绝 decode_responses）；
* :class:`TestConstructor` / :class:`TestFactoryIntegration` 校验构造约束与工厂注册。

全部用例基于 ``fakeredis`` 内存替身，无需真实集群即可运行。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import fakeredis
import pytest
from behavior import MedHistoryBehaviorSuite
from redis.exceptions import RedisError

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.exceptions import StorageError
from med_langchain_memory.stores import StoreFactory
from med_langchain_memory.stores.redis_cluster_store import (
    RedisClusterMedHistory,
    build_cluster_client,
)

NAMESPACE = {
    "session_id": "s-cluster",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-cluster"
TAGGED_BASE = "med:chat:hospital_a:cardiology:{s-cluster}"


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


@pytest.fixture
def client() -> Iterator[fakeredis.FakeRedis]:
    """独立的内存 redis 替身（二进制模式）。"""
    fake = fakeredis.FakeRedis()
    yield fake
    fake.flushall()


@pytest.fixture
def history(client: fakeredis.FakeRedis) -> RedisClusterMedHistory:
    """默认命名空间下的 Redis 集群会话历史。"""
    return RedisClusterMedHistory(**NAMESPACE, client=client)


class _BoomClient:
    """所有命令都抛 ``RedisError`` 的假客户端，用于校验异常包装。"""

    def pipeline(self, transaction: bool = True) -> Any:
        raise RedisError("connection lost")

    def lrange(self, *args: Any, **kwargs: Any) -> Any:
        raise RedisError("connection lost")

    def llen(self, *args: Any, **kwargs: Any) -> Any:
        raise RedisError("connection lost")

    def hgetall(self, *args: Any, **kwargs: Any) -> Any:
        raise RedisError("connection lost")


@pytest.fixture
def broken_history() -> RedisClusterMedHistory:
    """注入了故障客户端的集群会话历史。"""
    return RedisClusterMedHistory(**NAMESPACE, client=_BoomClient())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 共享行为契约
# --------------------------------------------------------------------------- #
class TestClusterStoreBehavior(MedHistoryBehaviorSuite):
    """Redis 集群后端必须满足全部通用存储行为契约。"""

    backend_name = "redis-cluster"
    shared_across_handles = True

    @pytest.fixture(autouse=True)
    def _isolated_client(self) -> Iterator[None]:
        """每个用例使用独立的内存 redis 替身。"""
        self.client = fakeredis.FakeRedis()
        yield
        self.client.flushall()

    def make_history(self, **overrides: Any) -> RedisClusterMedHistory:
        """构造 Redis 集群存储实例，复用同一个替身客户端。"""
        kwargs: dict[str, Any] = {**self.NAMESPACE, "client": self.client}
        kwargs.update(overrides)
        return RedisClusterMedHistory(**kwargs)


# --------------------------------------------------------------------------- #
# 集群键布局（hash tag）
# --------------------------------------------------------------------------- #
class TestHashTagKeys:
    """校验 hash tag 保证会话内同 slot、租户/科室不被包入 tag。"""

    def test_key_layout_matches_storage_spec(self, history: RedisClusterMedHistory) -> None:
        assert history.storage_key == STORAGE_KEY  # 命名空间校验用的键不带 tag
        assert history.messages_key == f"{TAGGED_BASE}:messages"
        assert history.meta_key == f"{TAGGED_BASE}:meta"

    def test_messages_and_meta_share_same_slot_tag(self, history: RedisClusterMedHistory) -> None:
        tag = "{" + NAMESPACE["session_id"] + "}"
        assert tag in history.messages_key
        assert tag in history.meta_key
        # 去掉后缀后两键的 tagged 前缀完全一致 -> 必然同 slot
        assert (
            history.messages_key.rsplit(":messages", 1)[0] == history.meta_key.rsplit(":meta", 1)[0]
        )

    def test_hash_tag_wraps_only_session_id(self, history: RedisClusterMedHistory) -> None:
        # tenant/dept 不能出现在 {} 内，否则整个租户会被压到单一 slot，丧失集群分片能力
        assert "{" + NAMESPACE["tenant_id"] + "}" not in history.messages_key
        assert "{" + NAMESPACE["dept_id"] + "}" not in history.messages_key

    def test_different_sessions_get_different_tags(self, client: fakeredis.FakeRedis) -> None:
        a = RedisClusterMedHistory(**{**NAMESPACE, "session_id": "s-a"}, client=client)
        b = RedisClusterMedHistory(**{**NAMESPACE, "session_id": "s-b"}, client=client)

        assert "{s-a}" in a.messages_key
        assert "{s-b}" in b.messages_key
        assert a.messages_key != b.messages_key


# --------------------------------------------------------------------------- #
# 集群 TTL
# --------------------------------------------------------------------------- #
class TestClusterTtl:
    """校验 TTL 打在带 hash tag 的键上（同 slot 才能走事务 pipeline）。"""

    def test_supports_native_ttl(self) -> None:
        assert RedisClusterMedHistory.supports_ttl is True

    def test_constructor_ttl_expires_tagged_keys(self, client: fakeredis.FakeRedis) -> None:
        history = RedisClusterMedHistory(**NAMESPACE, client=client, ttl_seconds=50)

        history.add_med_messages([make_message()])

        assert history.messages_key == f"{TAGGED_BASE}:messages"
        assert client.ttl(history.messages_key) == 50
        assert client.ttl(history.meta_key) == 50
        assert client.ttl(f"{STORAGE_KEY}:messages") == -2  # 未加 tag 的旧键不应被写入

    def test_set_ttl_none_persists_tagged_keys(self, history: RedisClusterMedHistory) -> None:
        history.set_ttl(60)
        history.add_med_messages([make_message()])

        history.set_ttl(None)

        assert history.ttl_remaining() is None
        assert history.exists() is True


# --------------------------------------------------------------------------- #
# 连接池配置入口
# --------------------------------------------------------------------------- #
class TestBuildClusterClient:
    """校验 build_cluster_client 的池配置透传与非法参数拒绝。"""

    def test_forwards_startup_nodes_and_pool_config(self) -> None:
        nodes = [
            {"host": "10.0.0.1", "port": 6379},
            {"host": "10.0.0.2", "port": 6379},
        ]
        with patch("med_langchain_memory.stores.redis_cluster_store.RedisCluster") as mock_cluster:
            mock_cluster.return_value = object()  # 仅校验调用入参，不真正建连
            build_cluster_client(nodes, max_connections=32)

        assert mock_cluster.call_count == 1
        _, kwargs = mock_cluster.call_args
        assert kwargs["max_connections"] == 32
        sent = [(node.host, node.port) for node in kwargs["startup_nodes"]]
        assert sent == [("10.0.0.1", 6379), ("10.0.0.2", 6379)]

    def test_rejects_decode_responses(self) -> None:
        with pytest.raises(StorageError, match="must not enable decode_responses"):
            build_cluster_client([{"host": "h", "port": 6379}], decode_responses=True)


# --------------------------------------------------------------------------- #
# 构造约束
# --------------------------------------------------------------------------- #
class TestConstructor:
    """校验集群后端必须注入 client，且拒绝破坏二进制的客户端。"""

    def test_requires_explicit_client(self) -> None:
        with pytest.raises(StorageError, match="requires an explicit"):
            RedisClusterMedHistory(**NAMESPACE, client=None)  # type: ignore[arg-type]

    def test_decode_responses_client_is_rejected(self, client: fakeredis.FakeRedis) -> None:
        client.connection_pool.connection_kwargs["decode_responses"] = True

        with pytest.raises(StorageError, match="must not enable decode_responses"):
            RedisClusterMedHistory(**NAMESPACE, client=client)

    def test_append_wraps_redis_error(self, broken_history: RedisClusterMedHistory) -> None:
        with pytest.raises(StorageError, match="redis append failed"):
            broken_history.add_med_messages([make_message()])

    def test_read_wraps_redis_error(self, broken_history: RedisClusterMedHistory) -> None:
        with pytest.raises(StorageError, match="redis read failed"):
            broken_history.get_med_messages()


# --------------------------------------------------------------------------- #
# 工厂集成
# --------------------------------------------------------------------------- #
class TestFactoryIntegration:
    """校验集群后端在工厂注册表中的可用性。"""

    def test_backend_is_registered(self) -> None:
        assert StoreFactory.is_registered("redis-cluster")
        assert StoreFactory.get("redis-cluster") is RedisClusterMedHistory
        assert "redis-cluster" in StoreFactory.available()

    def test_create_through_factory(self, client: fakeredis.FakeRedis) -> None:
        built = StoreFactory.create("redis-cluster", **NAMESPACE, client=client)

        assert isinstance(built, RedisClusterMedHistory)
        assert built.client is client

    def test_create_from_config(self, client: fakeredis.FakeRedis) -> None:
        from med_langchain_memory.stores import StoreConfig

        config = StoreConfig(backend="redis-cluster", options={"client": client})

        built = StoreFactory.create_from_config(config, **NAMESPACE)

        assert isinstance(built, RedisClusterMedHistory)
        assert built.size == 0

    def test_create_without_client_raises(self) -> None:
        with pytest.raises(StorageError, match="cannot build RedisClusterMedHistory"):
            StoreFactory.create("redis-cluster", **NAMESPACE)
