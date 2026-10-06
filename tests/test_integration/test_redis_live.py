"""真实 Redis 集成测试（默认跳过，见 ``conftest.py``）。

覆盖两层：

* :class:`TestRedisLiveBehavior` —— 复用 ``tests/test_stores/behavior.py`` 的跨后端行为基准套件，
  与 ``fakeredis`` 单测共享同一份语义契约，但跑在**真实 Redis 服务端**上；
* :class:`TestRedisLiveStorageLayout` —— 替身无法证伪的服务端事实：
  键类型（List / Hash）、原生 TTL 可见性、protobuf 载荷字节级往返。

运行方式::

    docker compose -f docker-compose.integration.yml up -d redis
    MED_MEMORY_IT=1 pytest -m integration tests/test_integration/test_redis_live.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from behavior import MedHistoryBehaviorSuite

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.serde import ProtobufSerializer
from med_langchain_memory.stores import MedChatMessageHistory
from med_langchain_memory.stores.redis_store import MESSAGES_SUFFIX, META_SUFFIX, RedisMedHistory

NAMESPACE: dict[str, str] = {
    "session_id": "s-redis-live",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-redis-live"


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


@pytest.mark.integration
class TestRedisLiveBehavior(MedHistoryBehaviorSuite):
    """真实 Redis 上的跨后端行为契约。"""

    backend_name = "redis"

    @pytest.fixture(autouse=True)
    def _live_client(self, redis_live: Any) -> Iterator[None]:
        """每个用例前清空集成测试专用 DB，保证用例之间互不干扰。"""
        self._client = redis_live
        redis_live.flushdb()
        yield
        redis_live.flushdb()

    def make_history(self, **overrides: Any) -> MedChatMessageHistory:
        """构造指向真实 Redis 的会话历史。"""
        return RedisMedHistory(**{**self.NAMESPACE, **overrides}, client=self._client)


@pytest.mark.integration
class TestRedisLiveStorageLayout:
    """服务端键布局、原生 TTL 与 protobuf 载荷。"""

    @pytest.fixture
    def history(self, redis_live: Any) -> Iterator[RedisMedHistory]:
        """带 120 秒 TTL 的真实 Redis 会话历史。"""
        redis_live.flushdb()
        yield RedisMedHistory(**NAMESPACE, client=redis_live, ttl_seconds=120)
        redis_live.flushdb()

    def test_writes_land_on_documented_keys(
        self, history: RedisMedHistory, redis_live: Any
    ) -> None:
        """正向：消息落 ``:messages`` List、元数据落 ``:meta`` Hash。"""
        history.add_med_messages([make_message()])
        assert redis_live.type(f"{STORAGE_KEY}{MESSAGES_SUFFIX}") == b"list"
        assert redis_live.type(f"{STORAGE_KEY}{META_SUFFIX}") == b"hash"

    def test_native_ttl_is_visible_to_redis(
        self, history: RedisMedHistory, redis_live: Any
    ) -> None:
        """正向：TTL 由 Redis 原生 ``EXPIRE`` 生效，服务端可读出剩余秒数。"""
        history.add_med_messages([make_message()])
        assert 0 < redis_live.ttl(f"{STORAGE_KEY}{MESSAGES_SUFFIX}") <= 120
        assert history.ttl_remaining() is not None

    def test_payload_is_byte_exact_protobuf(
        self, history: RedisMedHistory, redis_live: Any
    ) -> None:
        """正向：列表中的载荷是 protobuf 二进制，反序列化与写入对象等价。"""
        message = make_message(content="allergy to penicillin")
        history.add_med_messages([message])
        raw = redis_live.lrange(f"{STORAGE_KEY}{MESSAGES_SUFFIX}", 0, -1)
        assert len(raw) == 1
        assert ProtobufSerializer().deserialize_message(raw[0]) == message

    def test_clear_removes_both_keys(self, history: RedisMedHistory, redis_live: Any) -> None:
        """边界：``clear()`` 后两条键都不存在，``exists()`` 转为假。"""
        history.add_med_messages([make_message()])
        assert history.exists() is True
        history.clear()
        assert redis_live.exists(f"{STORAGE_KEY}{MESSAGES_SUFFIX}") == 0
        assert redis_live.exists(f"{STORAGE_KEY}{META_SUFFIX}") == 0
        assert history.exists() is False
