"""真实 Elasticsearch 集成测试（默认跳过，见 ``conftest.py``）。

覆盖两层：

* :class:`TestEsLiveBehavior` —— 复用 ``tests/test_stores/behavior.py`` 的跨后端行为基准套件，
  与 fake ES 单测共享同一份语义契约，但跑在**真实 ES 集群**上；
* :class:`TestEsLiveArchive` —— 替身无法证伪的服务端事实：
  索引模板幂等注册、按月滚动索引真实落盘、``search_archive`` 跨会话检索。

集成测试使用独立索引前缀 ``med-chat-it``（与生产默认前缀 ``med-chat-archive`` 隔离），
每个用例前后删除该前缀的索引与归档模板，保证可重复执行。

运行方式::

    docker compose -f docker-compose.integration.yml up -d elasticsearch
    MED_MEMORY_IT=1 pytest -m integration tests/test_integration/test_es_live.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from behavior import MedHistoryBehaviorSuite

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.stores import MedChatMessageHistory
from med_langchain_memory.stores.es_store import (
    ARCHIVE_TEMPLATE_NAME,
    EsArchiveMedHistory,
    monthly_index,
    search_archive,
)

#: 集成测试专用归档索引前缀（生产默认为 ``med-chat-archive``）。
LIVE_PREFIX = "med-chat-it"

#: 集成测试专用索引通配符。
LIVE_INDEX_PATTERN = f"{LIVE_PREFIX}-*"

NAMESPACE: dict[str, str] = {
    "session_id": "s-es-live",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}

#: 固定时间戳（2023-11-14T22:13:20Z），用于断言按创建时间路由到固定月份索引。
FIXED_MS = 1_700_000_000_000


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


def live_history(client: Any, **overrides: Any) -> EsArchiveMedHistory:
    """构造指向真实 ES 的归档历史（使用集成测试专用索引前缀）。"""
    return EsArchiveMedHistory(
        **{**NAMESPACE, **overrides}, client=client, prefix=LIVE_PREFIX, refresh=True
    )


def ensure_live_template(client: Any) -> None:
    """幂等注册集成测试专用归档索引模板（前缀级，与会话命名空间无关）。

    模板必须先注册：它把 ``tenant_id`` / ``dept_id`` / ``session_id`` 等命名空间字段
    映射为 ``keyword``。若缺模板，ES 会按动态映射把这些字段建成 ``text``，
    分词后 ``term`` 精确过滤失效（``session_id="s-1"`` 会被切成 ``s`` 与 ``1``），
    于是写入成功却读不出来——这是替身测试无法暴露的真实集群行为。
    """
    EsArchiveMedHistory(
        session_id="s-template-probe",
        tenant_id="hospital_a",
        dept_id="cardiology",
        patient_id="p-1",
        client=client,
        prefix=LIVE_PREFIX,
    ).ensure_index_template()


def drop_live_indices(client: Any) -> None:
    """删除集成测试专用索引与归档索引模板（不存在时忽略）。

    ES 8 默认 ``action.destructive_requires_name=true``，**禁止按通配符删除索引**，
    因此先按通配符查出具体索引名，再逐个具名删除。
    """
    options = client.options(ignore_status=404)
    existing = client.indices.get(
        index=LIVE_INDEX_PATTERN, ignore_unavailable=True, allow_no_indices=True
    )
    names = sorted(existing)
    if names:
        options.indices.delete(index=",".join(names), ignore_unavailable=True)
    options.indices.delete_index_template(name=ARCHIVE_TEMPLATE_NAME)


@pytest.fixture(autouse=True)
def _clean_archive(es_live: Any) -> Iterator[None]:
    """每个用例前后清理集成测试专用索引与模板，并在用例前重新注册索引模板。"""
    drop_live_indices(es_live)
    ensure_live_template(es_live)
    yield
    drop_live_indices(es_live)


@pytest.mark.integration
class TestEsLiveBehavior(MedHistoryBehaviorSuite):
    """真实 ES 上的跨后端行为契约。"""

    backend_name = "elasticsearch"

    @pytest.fixture(autouse=True)
    def _live_client(self, es_live: Any) -> Iterator[None]:
        """把真实 ES 客户端注入行为套件。"""
        self._client = es_live
        yield

    def make_history(self, **overrides: Any) -> MedChatMessageHistory:
        """构造指向真实 ES 的归档历史（沿用行为套件的命名空间）。"""
        return EsArchiveMedHistory(
            **{**self.NAMESPACE, **overrides},
            client=self._client,
            prefix=LIVE_PREFIX,
            refresh=True,
        )


@pytest.mark.integration
class TestEsLiveArchive:
    """索引模板、按月滚动索引与跨会话检索。"""

    def test_index_template_registration_is_idempotent(self, es_live: Any) -> None:
        """正向：模板已由 ``live_history`` 注册，重复调用不重复写入，``force=True`` 覆盖。"""
        history = live_history(es_live)
        assert history.ensure_index_template() is False
        assert history.ensure_index_template(force=True) is True
        assert history.ensure_index_template() is False

    def test_messages_are_routed_to_monthly_index(self, es_live: Any) -> None:
        """正向：消息按创建时间落到 ``{prefix}-{yyyy.MM}`` 索引，``count`` 与之一致。"""
        history = live_history(es_live)
        history.add_med_messages([make_message(created_at=FIXED_MS)])
        assert bool(es_live.indices.exists(index=monthly_index(FIXED_MS, LIVE_PREFIX))) is True
        assert history.count() == 1

    def test_search_archive_filters_by_tenant(self, es_live: Any) -> None:
        """正向：跨会话检索按租户强制隔离，其它租户查不到任何数据。"""
        history = live_history(es_live)
        history.add_med_messages([make_message("a"), make_message("b")])
        found = search_archive(
            es_live,
            tenant_id=NAMESPACE["tenant_id"],
            session_id=NAMESPACE["session_id"],
            prefix=LIVE_PREFIX,
        )
        assert [message.content for message in found] == ["a", "b"]
        assert search_archive(es_live, tenant_id="hospital_other", prefix=LIVE_PREFIX) == []

    def test_clear_removes_archived_documents(self, es_live: Any) -> None:
        """边界：``clear()`` 后归档文档被删除，``count()`` 归零。"""
        history = live_history(es_live)
        history.add_med_messages([make_message()])
        assert history.count() == 1
        history.clear()
        assert history.count() == 0
        assert history.get_med_messages() == []
