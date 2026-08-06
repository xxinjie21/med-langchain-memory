"""RedisMedHistory 单元测试。

分四部分：

* :class:`TestRedisStoreBehavior` 复用跨后端共享行为套件，校验通用存储契约；
* 存储结构用例：List + Hash 键布局、protobuf 载荷、pipeline 单次往返、元数据累加；
* 健壮性用例：损坏载荷、损坏元数据、``decode_responses`` 客户端、redis 异常包装；
* 集成用例：工厂注册与配置驱动实例化。

全部用例基于 ``fakeredis`` 内存替身，无需真实 Redis 实例即可运行。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from unittest.mock import patch

import fakeredis
import pytest
from behavior import MedHistoryBehaviorSuite
from redis import Redis
from redis.exceptions import RedisError

from med_langchain_memory.domain import MedMessage, MessageRole, SessionStatus
from med_langchain_memory.exceptions import StorageError
from med_langchain_memory.serde import ProtobufSerializer
from med_langchain_memory.stores import StoreConfig, StoreFactory
from med_langchain_memory.stores.redis_store import (
    DEFAULT_REDIS_URL,
    MESSAGES_SUFFIX,
    META_SUFFIX,
    RedisMedHistory,
)

NAMESPACE = {
    "session_id": "s-redis",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-redis"


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
def history(client: fakeredis.FakeRedis) -> RedisMedHistory:
    """默认命名空间下的 Redis 会话历史。"""
    return RedisMedHistory(**NAMESPACE, client=client)


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
def broken_history() -> RedisMedHistory:
    """注入了故障客户端的会话历史。"""
    return RedisMedHistory(**NAMESPACE, client=_BoomClient())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# 共享行为契约
# --------------------------------------------------------------------------- #
class TestRedisStoreBehavior(MedHistoryBehaviorSuite):
    """Redis 后端必须满足全部通用存储行为契约。"""

    backend_name = "redis"
    shared_across_handles = True

    @pytest.fixture(autouse=True)
    def _isolated_client(self) -> Iterator[None]:
        """每个用例使用独立的内存 redis 替身。"""
        self.client = fakeredis.FakeRedis()
        yield
        self.client.flushall()

    def make_history(self, **overrides: Any) -> RedisMedHistory:
        """构造 Redis 存储实例，复用同一个替身客户端。"""
        kwargs: dict[str, Any] = {**self.NAMESPACE, "client": self.client}
        kwargs.update(overrides)
        return RedisMedHistory(**kwargs)


# --------------------------------------------------------------------------- #
# 存储结构
# --------------------------------------------------------------------------- #
class TestStorageLayout:
    """校验 List + Hash 的键布局与落库内容。"""

    def test_key_names_follow_storage_spec(self, history: RedisMedHistory) -> None:
        assert history.storage_key == STORAGE_KEY
        assert history.messages_key == f"{STORAGE_KEY}{MESSAGES_SUFFIX}"
        assert history.meta_key == f"{STORAGE_KEY}{META_SUFFIX}"

    def test_append_writes_protobuf_payload_into_list(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        message = make_message("fever 38.5")

        history.add_med_messages([message])

        raw = client.lrange(history.messages_key, 0, -1)
        assert len(raw) == 1
        decoded = ProtobufSerializer().deserialize_message(raw[0])
        assert decoded.content == "fever 38.5"
        assert decoded.message_id == message.message_id

    def test_append_keeps_rpush_order(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        serializer = ProtobufSerializer()
        history.add_med_messages(
            [
                make_message("first", created_at=1_700_000_000_000),
                make_message("second", created_at=1_700_000_001_000),
            ]
        )

        raw = client.lrange(history.messages_key, 0, -1)
        assert [serializer.deserialize_message(item).content for item in raw] == [
            "first",
            "second",
        ]

    def test_append_writes_meta_hash(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        history.add_med_messages([make_message(), make_message("second")])

        meta = {k.decode(): v.decode() for k, v in client.hgetall(history.meta_key).items()}
        assert meta["session_id"] == NAMESPACE["session_id"]
        assert meta["tenant_id"] == NAMESPACE["tenant_id"]
        assert meta["dept_id"] == NAMESPACE["dept_id"]
        assert meta["patient_id"] == NAMESPACE["patient_id"]
        assert meta["status"] == SessionStatus.ACTIVE.value
        assert meta["message_count"] == "2"
        assert int(meta["updated_at"]) >= int(meta["created_at"])

    def test_append_batches_commands_into_single_pipeline(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        messages = [make_message(f"msg-{index}") for index in range(3)]

        with patch.object(client, "pipeline", wraps=client.pipeline) as spy:
            history.add_med_messages(messages)

        assert spy.call_count == 1
        assert history.size == 3

    def test_second_append_accumulates_count_and_keeps_created_at(
        self, history: RedisMedHistory
    ) -> None:
        history.add_med_messages([make_message("first")])
        first_meta = history.fetch_session_meta()
        assert first_meta is not None

        history.add_med_messages([make_message("second")])
        second_meta = history.fetch_session_meta()

        assert second_meta is not None
        assert second_meta.message_count == 2
        assert second_meta.created_at == first_meta.created_at
        assert second_meta.updated_at >= first_meta.updated_at

    def test_clear_removes_both_keys(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        history.add_med_messages([make_message()])

        history.clear()

        assert client.exists(history.messages_key) == 0
        assert client.exists(history.meta_key) == 0
        assert history.get_med_messages() == []

    def test_sessions_use_separate_keys_on_same_client(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        other = RedisMedHistory(**{**NAMESPACE, "session_id": "s-other"}, client=client)

        history.add_med_messages([make_message("mine")])

        assert other.size == 0
        assert other.messages_key != history.messages_key


# --------------------------------------------------------------------------- #
# 专有能力
# --------------------------------------------------------------------------- #
class TestRedisSpecificApi:
    """校验 Redis 后端专有属性与方法。"""

    def test_size_counts_stored_messages(self, history: RedisMedHistory) -> None:
        history.add_med_messages([make_message(), make_message("second")])

        assert history.size == 2

    def test_size_of_empty_session_is_zero(self, history: RedisMedHistory) -> None:
        assert history.size == 0

    def test_client_property_returns_injected_client(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        assert history.client is client

    def test_client_is_built_from_url_when_not_injected(self) -> None:
        built = RedisMedHistory(**NAMESPACE, url=DEFAULT_REDIS_URL)

        assert isinstance(built.client, Redis)
        assert built.client.connection_pool.connection_kwargs["port"] == 6379

    def test_fetch_session_meta_returns_none_for_new_session(
        self, history: RedisMedHistory
    ) -> None:
        assert history.fetch_session_meta() is None

    def test_fetch_session_meta_returns_persisted_meta(self, history: RedisMedHistory) -> None:
        history.add_med_messages([make_message()])

        meta = history.fetch_session_meta()

        assert meta is not None
        assert meta.storage_key == STORAGE_KEY
        assert meta.status is SessionStatus.ACTIVE
        assert meta.message_count == 1

    def test_fetch_session_meta_is_shared_across_handles(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        history.add_med_messages([make_message()])

        reopened = RedisMedHistory(**NAMESPACE, client=client)
        meta = reopened.fetch_session_meta()

        assert meta is not None
        assert meta.message_count == 1

    def test_ttl_is_not_supported_yet(self, history: RedisMedHistory) -> None:
        assert RedisMedHistory.supports_ttl is False
        with pytest.raises(StorageError, match="does not support native ttl"):
            history.set_ttl(60)


# --------------------------------------------------------------------------- #
# 健壮性
# --------------------------------------------------------------------------- #
class TestRobustness:
    """校验损坏数据与连接故障下的错误语义。"""

    def test_read_corrupted_payload_raises(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        client.rpush(history.messages_key, b"\xff\xff\xff\xff")

        with pytest.raises(StorageError, match="corrupted message at index 0"):
            history.get_med_messages()

    def test_read_payload_failing_validation_raises(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        client.rpush(history.messages_key, b"")

        with pytest.raises(StorageError, match="corrupted message at index 0"):
            history.get_med_messages()

    def test_fetch_session_meta_with_missing_fields_raises(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        client.hset(history.meta_key, mapping={"session_id": "s-redis"})

        with pytest.raises(StorageError, match="corrupted session meta"):
            history.fetch_session_meta()

    def test_fetch_session_meta_with_non_numeric_count_raises(
        self, history: RedisMedHistory, client: fakeredis.FakeRedis
    ) -> None:
        history.add_med_messages([make_message()])
        client.hset(history.meta_key, "message_count", "many")

        with pytest.raises(StorageError, match="corrupted session meta"):
            history.fetch_session_meta()

    def test_decode_responses_client_is_rejected(self) -> None:
        decoding = fakeredis.FakeRedis(decode_responses=True)

        with pytest.raises(StorageError, match="must not enable decode_responses"):
            RedisMedHistory(**NAMESPACE, client=decoding)

    def test_append_wraps_redis_error(self, broken_history: RedisMedHistory) -> None:
        with pytest.raises(StorageError, match="redis append failed"):
            broken_history.add_med_messages([make_message()])

    def test_read_wraps_redis_error(self, broken_history: RedisMedHistory) -> None:
        with pytest.raises(StorageError, match="redis read failed"):
            broken_history.get_med_messages()

    def test_clear_wraps_redis_error(self, broken_history: RedisMedHistory) -> None:
        with pytest.raises(StorageError, match="redis clear failed"):
            broken_history.clear()

    def test_size_wraps_redis_error(self, broken_history: RedisMedHistory) -> None:
        with pytest.raises(StorageError, match="redis llen failed"):
            _ = broken_history.size

    def test_fetch_session_meta_wraps_redis_error(self, broken_history: RedisMedHistory) -> None:
        with pytest.raises(StorageError, match="redis hgetall failed"):
            broken_history.fetch_session_meta()


# --------------------------------------------------------------------------- #
# 工厂集成
# --------------------------------------------------------------------------- #
class TestFactoryIntegration:
    """校验 Redis 后端在工厂注册表中的可用性。"""

    def test_backend_is_registered(self) -> None:
        assert StoreFactory.is_registered("redis")
        assert StoreFactory.get("redis") is RedisMedHistory
        assert "redis" in StoreFactory.available()

    def test_create_through_factory(self, client: fakeredis.FakeRedis) -> None:
        built = StoreFactory.create("redis", **NAMESPACE, client=client)

        assert isinstance(built, RedisMedHistory)
        assert built.client is client

    def test_create_from_config(self, client: fakeredis.FakeRedis) -> None:
        config = StoreConfig(backend="redis", options={"client": client})

        built = StoreFactory.create_from_config(config, **NAMESPACE)

        assert isinstance(built, RedisMedHistory)
        assert built.size == 0

    def test_create_with_unknown_option_raises(self, client: fakeredis.FakeRedis) -> None:
        with pytest.raises(StorageError, match="cannot build RedisMedHistory"):
            StoreFactory.create("redis", **NAMESPACE, client=client, unknown_option=1)
