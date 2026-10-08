"""真实 Redis Cluster 集成测试（默认跳过，见 ``conftest.py``）。

覆盖三层：

* :class:`TestRedisClusterLiveBehavior` —— 复用 ``tests/test_stores/behavior.py``
  的跨后端行为基准套件，跑在**真实三主三从集群**上；
* :class:`TestClusterSlotAffinity` —— hash tag 的实际效果：同一会话两条键
  落在同一 slot（服务端 ``CLUSTER KEYSLOT`` 计算，替身算不出来）；
* :class:`TestClusterPipelineSemantics` —— **真机回归**：redis-py 集群客户端
  弃用了 ``MULTI`` 事务（``pipeline(transaction=True)`` 直接抛
  ``RedisClusterException``），而单机实现默认走事务 pipeline。
  该缺陷只有真实集群能暴露，是引入本套件最主要的原因。

运行方式::

    export MED_MEMORY_IT_CLUSTER_IP=<宿主机局域网 IP>   # 仅 Docker Desktop 需要
    docker compose -f docker-compose.integration.yml up -d
    MED_MEMORY_IT=1 pytest -m integration tests/test_integration/test_redis_cluster_live.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from behavior import MedHistoryBehaviorSuite

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.serde import ProtobufSerializer
from med_langchain_memory.stores import MedChatMessageHistory
from med_langchain_memory.stores.redis_cluster_store import RedisClusterMedHistory
from med_langchain_memory.stores.redis_store import MESSAGES_SUFFIX, META_SUFFIX

NAMESPACE: dict[str, str] = {
    "session_id": "s-cluster-live",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
TAGGED_BASE = "med:chat:hospital_a:cardiology:{s-cluster-live}"
MESSAGES_KEY = f"{TAGGED_BASE}{MESSAGES_SUFFIX}"
META_KEY = f"{TAGGED_BASE}{META_SUFFIX}"


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


def flush_cluster(client: Any) -> None:
    """清空集群全部主节点上的数据。

    ``FLUSHDB`` 在集群模式下必须显式指定目标节点（默认只作用于随机一个节点），
    因此统一走 ``target_nodes=PRIMARIES``。

    刻意**不**放进 ``conftest.py`` 供导入：``conftest`` 是通用模块名，
    全量收集时会被其它测试目录的同名模块抢占（``tests/test_lifecycle/conftest.py``）。
    """
    client.flushdb(target_nodes=client.PRIMARIES)


@pytest.mark.integration
class TestRedisClusterLiveBehavior(MedHistoryBehaviorSuite):
    """真实 Redis Cluster 上的跨后端行为契约。"""

    backend_name = "redis-cluster"

    @pytest.fixture(autouse=True)
    def _live_cluster(self, redis_cluster_live: Any) -> Iterator[None]:
        """每个用例前后清空集群全部主节点，保证用例之间互不干扰。"""
        self._client = redis_cluster_live
        flush_cluster(redis_cluster_live)
        yield
        flush_cluster(redis_cluster_live)

    def make_history(self, **overrides: Any) -> MedChatMessageHistory:
        """构造指向真实集群的会话历史。"""
        return RedisClusterMedHistory(**{**self.NAMESPACE, **overrides}, client=self._client)


@pytest.mark.integration
class TestClusterSlotAffinity:
    """hash tag 的服务端效果：会话内两条键同 slot，不同会话可分散。"""

    @pytest.fixture
    def history(self, redis_cluster_live: Any) -> Iterator[RedisClusterMedHistory]:
        """指向真实集群、带 120 秒 TTL 的会话历史。"""
        flush_cluster(redis_cluster_live)
        yield RedisClusterMedHistory(**NAMESPACE, client=redis_cluster_live, ttl_seconds=120)
        flush_cluster(redis_cluster_live)

    def test_messages_and_meta_share_one_slot(
        self, history: RedisClusterMedHistory, redis_cluster_live: Any
    ) -> None:
        """正向：``CLUSTER KEYSLOT`` 对两条键给出同一 slot（pipeline 才能走单节点）。"""
        assert redis_cluster_live.cluster_keyslot(history.messages_key) == (
            redis_cluster_live.cluster_keyslot(history.meta_key)
        )

    def test_tagged_keys_are_readable_through_redirects(
        self, history: RedisClusterMedHistory, redis_cluster_live: Any
    ) -> None:
        """正向：经 MOVED 重定向读写带 tag 的键，服务端能看到落盘数据。"""
        history.add_med_messages([make_message(content="follow up in two weeks")])

        assert redis_cluster_live.type(history.messages_key) == b"list"
        assert redis_cluster_live.type(history.meta_key) == b"hash"
        assert redis_cluster_live.llen(history.messages_key) == 1

    def test_native_ttl_is_visible_on_tagged_keys(
        self, history: RedisClusterMedHistory, redis_cluster_live: Any
    ) -> None:
        """正向：TTL 由集群节点原生 ``EXPIRE`` 生效，服务端可读出剩余秒数。"""
        history.add_med_messages([make_message()])

        assert 0 < redis_cluster_live.ttl(history.messages_key) <= 120
        assert 0 < redis_cluster_live.ttl(history.meta_key) <= 120
        assert history.ttl_remaining() is not None

    def test_payload_is_byte_exact_protobuf(
        self, history: RedisClusterMedHistory, redis_cluster_live: Any
    ) -> None:
        """正向：集群上存的仍是 protobuf 二进制，反序列化与写入对象等价。"""
        message = make_message(content="allergy to penicillin")
        history.add_med_messages([message])

        raw = redis_cluster_live.lrange(history.messages_key, 0, -1)
        assert len(raw) == 1
        assert ProtobufSerializer().deserialize_message(raw[0]) == message

    def test_sessions_spread_across_slots(self, redis_cluster_live: Any) -> None:
        """边界：不同会话的键可以落在不同 slot（集群分片能力未被 tag 削弱）。"""
        slots = {
            redis_cluster_live.cluster_keyslot(
                f"med:chat:hospital_a:cardiology:{{s-{index}}}:messages"
            )
            for index in range(20)
        }
        assert len(slots) > 1

    def test_clear_removes_both_keys(self, history: RedisClusterMedHistory) -> None:
        """边界：``clear()`` 后集群上两条键都不存在。"""
        history.add_med_messages([make_message()])
        assert history.exists() is True

        history.clear()

        assert history.exists() is False


@pytest.mark.integration
class TestClusterPipelineSemantics:
    """真机回归：集群客户端不接受事务 pipeline。"""

    def test_cluster_rejects_transactional_pipeline(self, redis_cluster_live: Any) -> None:
        """边界：确认本集群的 redis-py 客户端确实拒绝 ``transaction=True``。

        这条断言是本套件存在的理由：一旦 redis-py 改了行为，
        「集群必须用非事务 pipeline」的适配就不再是必需的，需要重新评估。
        """
        from redis.exceptions import RedisClusterException

        with pytest.raises(RedisClusterException, match="transaction is deprecated"):
            redis_cluster_live.pipeline(transaction=True)

    def test_write_clear_and_ttl_work_on_real_cluster(self, redis_cluster_live: Any) -> None:
        """正向：写入 / 清空 / TTL 下发与取消在真实集群上全部可用。"""
        flush_cluster(redis_cluster_live)
        history = RedisClusterMedHistory(**NAMESPACE, client=redis_cluster_live)

        history.add_med_messages([make_message("real cluster write")])
        assert [m.content for m in history.get_med_messages()] == ["real cluster write"]

        history.set_ttl(120)
        assert history.ttl_remaining() is not None

        history.set_ttl(None)
        assert history.ttl_remaining() is None

        history.clear()
        assert history.exists() is False
