"""EsArchiveMedHistory 单元测试。

分六部分：

* :class:`TestEsArchiveBehavior` 复用跨后端共享行为套件，校验通用存储契约；
* 索引规划用例：按月滚动索引名、通配符检索模式、索引模板结构与幂等注册；
* 文档映射用例：protobuf 载荷双写、payload 优先还原、字段回退、损坏数据报错；
* 写入策略用例：跨月分发、bulk 分块、``_id`` 幂等覆盖、refresh 透传、逐条错误；
* 归档能力用例：``archive_from`` 热存储沉降、``count`` 统计、``search_archive`` 跨会话检索；
* 健壮性与集成用例：构造校验、ES 异常包装、工厂注册与配置驱动实例化。

全部用例基于内存 :class:`fake_es.FakeElasticsearch` 替身，无需真实 ES 集群即可运行。
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from behavior import MedHistoryBehaviorSuite
from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import Elasticsearch
from fake_es import FakeElasticsearch

from med_langchain_memory.domain import MedMessage, MessageRole, SessionStatus
from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.stores import InMemoryMedHistory, StoreConfig, StoreFactory
from med_langchain_memory.stores.es_store import (
    ARCHIVE_INDEX_PREFIX,
    ARCHIVE_TEMPLATE_NAME,
    MAX_SEARCH_SIZE,
    EsArchiveMedHistory,
    build_index_template,
    from_document,
    index_pattern,
    monthly_index,
    search_archive,
    to_document,
)

NAMESPACE = {
    "session_id": "s-es",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-es"

#: 2026-08-10T00:00:00Z
AUG_2026 = 1_786_060_800_000
#: 2026-09-01T00:00:00Z
SEP_2026 = 1_788_307_200_000


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


@pytest.fixture
def client() -> FakeElasticsearch:
    """独立的内存 ES 替身。"""
    return FakeElasticsearch()


@pytest.fixture
def history(client: FakeElasticsearch) -> EsArchiveMedHistory:
    """默认命名空间下的 ES 归档会话历史。"""
    return EsArchiveMedHistory(**NAMESPACE, client=cast("Elasticsearch", client))


# --------------------------------------------------------------------------- #
# 一、跨后端共享行为契约
# --------------------------------------------------------------------------- #
class TestEsArchiveBehavior(MedHistoryBehaviorSuite):
    """ES 归档后端必须满足全部通用存储行为契约。"""

    backend_name = "elasticsearch"
    shared_across_handles = True

    @pytest.fixture(autouse=True)
    def _isolated_client(self) -> None:
        """每个用例使用独立的内存 ES 替身。"""
        self.client = FakeElasticsearch()

    def make_history(self, **overrides: Any) -> EsArchiveMedHistory:
        """构造 ES 归档存储实例，复用同一个替身客户端。"""
        kwargs: dict[str, Any] = {**self.NAMESPACE, "client": self.client}
        kwargs.update(overrides)
        return EsArchiveMedHistory(**kwargs)


# --------------------------------------------------------------------------- #
# 二、索引规划
# --------------------------------------------------------------------------- #
class TestIndexPlanning:
    """按月滚动索引名、检索模式与索引模板。"""

    def test_monthly_index_uses_utc_year_month(self) -> None:
        assert monthly_index(AUG_2026) == "med-chat-archive-2026.08"

    def test_monthly_index_rolls_over_between_months(self) -> None:
        assert monthly_index(SEP_2026) == "med-chat-archive-2026.09"
        assert monthly_index(AUG_2026) != monthly_index(SEP_2026)

    def test_monthly_index_honours_custom_prefix(self) -> None:
        assert monthly_index(AUG_2026, prefix="archive") == "archive-2026.08"

    @pytest.mark.parametrize("created_at", [0, -1])
    def test_monthly_index_rejects_non_positive_timestamp(self, created_at: int) -> None:
        with pytest.raises(ValidationError, match="positive epoch millis"):
            monthly_index(created_at)

    def test_index_pattern_covers_all_months(self) -> None:
        assert index_pattern() == "med-chat-archive-*"
        assert index_pattern("archive") == "archive-*"

    def test_template_matches_archive_pattern(self) -> None:
        template = build_index_template()

        assert template["index_patterns"] == ["med-chat-archive-*"]
        assert template["priority"] == 200
        assert template["meta"]["owner"] == "med-langchain-memory"

    def test_template_mappings_disable_text_analysis(self) -> None:
        properties = build_index_template()["template"]["mappings"]["properties"]

        assert properties["content"]["index"] is False, "归档层不得对正文做任何文本分析"
        assert properties["metadata"]["enabled"] is False
        assert properties["payload"]["type"] == "binary"
        assert properties["session_id"]["type"] == "keyword"
        assert properties["created_at"]["type"] == "long"
        assert properties["ordinal"]["type"] == "integer"

    def test_template_settings_are_overridable(self) -> None:
        settings = build_index_template(shards=3, replicas=2)["template"]["settings"]

        assert settings["number_of_shards"] == 3
        assert settings["number_of_replicas"] == 2
        assert settings["codec"] == "best_compression"

    def test_ensure_index_template_is_idempotent(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        assert history.ensure_index_template() is True
        assert history.ensure_index_template() is False
        assert ARCHIVE_TEMPLATE_NAME in client.templates
        assert client.templates[ARCHIVE_TEMPLATE_NAME]["index_patterns"] == ["med-chat-archive-*"]

    def test_ensure_index_template_force_overwrites(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.ensure_index_template()
        client.templates[ARCHIVE_TEMPLATE_NAME] = {"stale": True}

        assert history.ensure_index_template(force=True) is True
        assert "stale" not in client.templates[ARCHIVE_TEMPLATE_NAME]

    def test_ensure_index_template_wraps_transport_error(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        client.fail_on("put_index_template", EsConnectionError("cluster down"))

        with pytest.raises(StorageError, match="elasticsearch index_template failed"):
            history.ensure_index_template()


# --------------------------------------------------------------------------- #
# 三、文档映射
# --------------------------------------------------------------------------- #
class TestDocumentMapping:
    """检索字段 + protobuf 载荷的双写与还原。"""

    def test_document_carries_searchable_fields_and_payload(self) -> None:
        message = make_message("fever", created_at=AUG_2026, token_count=7)

        document = to_document(message, archived_at=AUG_2026 + 1000, ordinal=3)

        assert document["session_id"] == "s-es"
        assert document["storage_key"] == STORAGE_KEY
        assert document["role"] == "patient"
        assert document["token_count"] == 7
        assert document["archived_at"] == AUG_2026 + 1000
        assert document["ordinal"] == 3
        assert isinstance(document["payload"], str) and document["payload"]

    def test_document_roundtrip_is_lossless(self) -> None:
        message = make_message("cough", role=MessageRole.DOCTOR, metadata={"ward": "3a"})

        assert from_document(to_document(message)) == message

    def test_archived_at_defaults_to_now(self) -> None:
        document = to_document(make_message())

        assert document["archived_at"] > 0

    def test_from_document_falls_back_to_flat_fields(self) -> None:
        message = make_message("no payload", created_at=AUG_2026)
        document = to_document(message)
        document.pop("payload")

        restored = from_document(document)

        assert restored.content == "no payload"
        assert restored.message_id == message.message_id
        assert restored.created_at == AUG_2026

    def test_from_document_rejects_corrupted_payload(self) -> None:
        document = to_document(make_message())
        document["payload"] = "bm90LXByb3RvYnVm" * 4

        with pytest.raises(StorageError, match="corrupted archived payload"):
            from_document(document)

    def test_from_document_rejects_missing_fields(self) -> None:
        document = to_document(make_message())
        document.pop("payload")
        document.pop("session_id")

        with pytest.raises(StorageError, match="corrupted archived document"):
            from_document(document)


# --------------------------------------------------------------------------- #
# 四、写入策略
# --------------------------------------------------------------------------- #
class TestBulkWrites:
    """跨月分发、分块、幂等与错误处理。"""

    def test_messages_are_routed_to_monthly_indices(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages(
            [
                make_message("august", created_at=AUG_2026),
                make_message("september", created_at=SEP_2026),
            ]
        )

        assert sorted(client.documents) == ["med-chat-archive-2026.08", "med-chat-archive-2026.09"]
        assert len(client.documents["med-chat-archive-2026.08"]) == 1
        assert len(client.documents["med-chat-archive-2026.09"]) == 1

    def test_bulk_is_split_into_chunks(self, client: FakeElasticsearch) -> None:
        history = EsArchiveMedHistory(
            **NAMESPACE, client=cast("Elasticsearch", client), chunk_size=2
        )

        history.add_med_messages(
            [make_message(f"m-{i}", created_at=AUG_2026 + i) for i in range(5)]
        )

        assert client.bulk_batches == [2, 2, 1]
        assert history.count() == 5

    def test_single_bulk_request_for_small_batch(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages([make_message("a"), make_message("b")])

        assert client.bulk_batches == [2]

    def test_replaying_same_messages_is_idempotent(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        messages = [make_message("once", created_at=AUG_2026)]

        history.add_med_messages(messages)
        history.add_med_messages(messages)

        assert history.count() == 1
        assert len(client.all_sources()) == 1

    def test_refresh_flag_is_forwarded(self, client: FakeElasticsearch) -> None:
        history = EsArchiveMedHistory(
            **NAMESPACE, client=cast("Elasticsearch", client), refresh=False
        )

        history.add_med_messages([make_message()])

        assert client.refresh_flags == [False]

    def test_bulk_item_errors_raise_storage_error(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        message = make_message("rejected")
        client.rejected_ids.add(message.message_id)

        with pytest.raises(StorageError, match="bulk archive failed"):
            history.add_med_messages([message])

    def test_limit_query_uses_descending_search(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages(
            [make_message(f"m-{i}", created_at=AUG_2026 + i) for i in range(3)]
        )

        assert [m.content for m in history.get_med_messages(limit=2)] == ["m-1", "m-2"]
        assert client.last_search["size"] == 2
        assert client.last_search["sort"][0]["created_at"]["order"] == "desc"

    def test_full_read_uses_ascending_search_with_max_window(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages([make_message()])
        history.get_med_messages()

        assert client.last_search["size"] == MAX_SEARCH_SIZE
        assert client.last_search["sort"][0]["created_at"]["order"] == "asc"

    def test_same_millisecond_messages_keep_write_order(self, history: EsArchiveMedHistory) -> None:
        batch = [make_message(f"m-{i}", created_at=AUG_2026) for i in range(6)]

        history.add_med_messages(batch)

        assert [m.content for m in history.get_med_messages()] == [m.content for m in batch]
        assert [m.content for m in history.get_med_messages(limit=3)] == ["m-3", "m-4", "m-5"]

    def test_read_rejects_corrupted_hit(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages([make_message(created_at=AUG_2026)])
        index = "med-chat-archive-2026.08"
        doc_id = next(iter(client.documents[index]))
        client.documents[index][doc_id]["payload"] = "!!!not-base64!!!"

        with pytest.raises(StorageError, match="corrupted archived payload"):
            history.get_med_messages()


# --------------------------------------------------------------------------- #
# 五、归档能力
# --------------------------------------------------------------------------- #
class TestArchiveCapabilities:
    """热存储沉降、统计与跨会话检索。"""

    @pytest.fixture(autouse=True)
    def _clean_hot_store(self) -> None:
        """内存热存储是进程级共享的，逐用例清空避免相互污染。"""
        InMemoryMedHistory.reset()

    def test_archive_from_hot_store(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        hot = InMemoryMedHistory(**NAMESPACE)
        hot.add_med_messages([make_message("first"), make_message("second")])

        archived = history.archive_from(hot)

        assert archived == 2
        assert hot.session_meta.status is SessionStatus.ARCHIVED
        assert [m.content for m in history.get_med_messages()] == ["first", "second"]
        assert hot.get_med_messages() != [], "沉降不清理热存储，由调度方决定"

    def test_archive_from_empty_session(self, history: EsArchiveMedHistory) -> None:
        hot = InMemoryMedHistory(**NAMESPACE)

        assert history.archive_from(hot) == 0
        assert history.count() == 0

    def test_archive_from_rejects_namespace_mismatch(self, history: EsArchiveMedHistory) -> None:
        foreign = InMemoryMedHistory(**{**NAMESPACE, "session_id": "s-other"})

        with pytest.raises(StorageError, match="namespace mismatch"):
            history.archive_from(foreign)

    def test_count_of_empty_session_is_zero(self, history: EsArchiveMedHistory) -> None:
        assert history.count() == 0

    def test_search_archive_filters_by_tenant(self, client: FakeElasticsearch) -> None:
        mine = EsArchiveMedHistory(**NAMESPACE, client=cast("Elasticsearch", client))
        other_ns = {**NAMESPACE, "tenant_id": "hospital_b"}
        theirs = EsArchiveMedHistory(**other_ns, client=cast("Elasticsearch", client))
        mine.add_med_messages([make_message("mine", created_at=AUG_2026)])
        theirs.add_med_messages([make_message("theirs", created_at=AUG_2026, **other_ns)])

        found = search_archive(cast("Elasticsearch", client), tenant_id="hospital_a")

        assert [m.content for m in found] == ["mine"]

    def test_search_archive_applies_optional_filters(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages(
            [
                make_message("early", created_at=AUG_2026),
                make_message("late", created_at=SEP_2026),
            ]
        )

        window = search_archive(
            cast("Elasticsearch", client),
            tenant_id="hospital_a",
            dept_id="cardiology",
            patient_id="p-1024",
            session_id="s-es",
            start_ms=AUG_2026,
            end_ms=AUG_2026 + 1,
        )

        assert [m.content for m in window] == ["early"]

    def test_search_archive_respects_limit_and_order(
        self, history: EsArchiveMedHistory, client: FakeElasticsearch
    ) -> None:
        history.add_med_messages(
            [make_message(f"m-{i}", created_at=AUG_2026 + i) for i in range(4)]
        )

        found = search_archive(cast("Elasticsearch", client), tenant_id="hospital_a", limit=2)

        assert [m.content for m in found] == ["m-0", "m-1"]

    def test_search_archive_on_empty_cluster(self, client: FakeElasticsearch) -> None:
        assert search_archive(cast("Elasticsearch", client), tenant_id="hospital_a") == []

    def test_search_archive_requires_tenant(self, client: FakeElasticsearch) -> None:
        with pytest.raises(ValidationError, match="tenant_id is required"):
            search_archive(cast("Elasticsearch", client), tenant_id="")

    @pytest.mark.parametrize("limit", [0, -1, MAX_SEARCH_SIZE + 1])
    def test_search_archive_rejects_invalid_limit(
        self, client: FakeElasticsearch, limit: int
    ) -> None:
        with pytest.raises(ValidationError, match="limit must be within"):
            search_archive(cast("Elasticsearch", client), tenant_id="hospital_a", limit=limit)

    def test_search_archive_rejects_inverted_range(self, client: FakeElasticsearch) -> None:
        with pytest.raises(ValidationError, match="invalid time range"):
            search_archive(
                cast("Elasticsearch", client),
                tenant_id="hospital_a",
                start_ms=SEP_2026,
                end_ms=AUG_2026,
            )

    def test_search_archive_wraps_transport_error(self, client: FakeElasticsearch) -> None:
        client.fail_on("search", EsConnectionError("cluster down"))

        with pytest.raises(StorageError, match="elasticsearch search failed"):
            search_archive(cast("Elasticsearch", client), tenant_id="hospital_a")


# --------------------------------------------------------------------------- #
# 六、健壮性与工厂集成
# --------------------------------------------------------------------------- #
class TestRobustnessAndFactory:
    """构造校验、异常包装与工厂注册。"""

    @pytest.mark.parametrize("prefix", ["", "MED-ARCHIVE", "med archive", "med-*"])
    def test_invalid_prefix_is_rejected(self, client: FakeElasticsearch, prefix: str) -> None:
        with pytest.raises(ValidationError, match="invalid archive index prefix"):
            EsArchiveMedHistory(**NAMESPACE, client=cast("Elasticsearch", client), prefix=prefix)

    @pytest.mark.parametrize("chunk_size", [0, -3])
    def test_invalid_chunk_size_is_rejected(
        self, client: FakeElasticsearch, chunk_size: int
    ) -> None:
        with pytest.raises(ValidationError, match="chunk_size must be"):
            EsArchiveMedHistory(
                **NAMESPACE, client=cast("Elasticsearch", client), chunk_size=chunk_size
            )

    def test_custom_prefix_changes_index_layout(self, client: FakeElasticsearch) -> None:
        history = EsArchiveMedHistory(
            **NAMESPACE, client=cast("Elasticsearch", client), prefix="med-archive"
        )

        history.add_med_messages([make_message(created_at=AUG_2026)])

        assert history.index_pattern == "med-archive-*"
        assert history.prefix == "med-archive"
        assert list(client.documents) == ["med-archive-2026.08"]

    def test_default_client_is_lazily_constructed(self) -> None:
        history = EsArchiveMedHistory(**NAMESPACE)

        assert isinstance(history.client, Elasticsearch)
        assert history.chunk_size > 0

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ("bulk", "elasticsearch bulk failed"),
            ("search", "elasticsearch search failed"),
            ("count", "elasticsearch count failed"),
            ("delete_by_query", "elasticsearch delete_by_query failed"),
        ],
    )
    def test_transport_errors_are_wrapped(
        self,
        history: EsArchiveMedHistory,
        client: FakeElasticsearch,
        action: str,
        expected: str,
    ) -> None:
        client.fail_on(action, EsConnectionError("cluster down"))
        operations = {
            "bulk": lambda: history.add_med_messages([make_message()]),
            "search": history.get_med_messages,
            "count": history.count,
            "delete_by_query": history.clear,
        }

        with pytest.raises(StorageError, match=expected):
            operations[action]()

    def test_backend_registered_under_elasticsearch(self) -> None:
        assert StoreFactory.is_registered("elasticsearch")
        assert StoreFactory.get("elasticsearch") is EsArchiveMedHistory
        assert ARCHIVE_INDEX_PREFIX in EsArchiveMedHistory(**NAMESPACE).index_pattern

    def test_factory_creates_store_from_config(self, client: FakeElasticsearch) -> None:
        config = StoreConfig(backend="elasticsearch", options={"client": client})

        history = StoreFactory.create_from_config(config, **NAMESPACE)

        assert isinstance(history, EsArchiveMedHistory)
        assert history.storage_key == STORAGE_KEY
