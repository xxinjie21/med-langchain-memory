"""``stores.mysql_shard_router`` 单元测试。

覆盖：

* 路由一致性：同一 ``session_id`` 始终映射到同一分片；
* 取值范围：分片编号始终在 ``[0, shard_count)`` 内；
* 表名/表对象一致性：与 ``mysql_schema`` 定义完全对齐；
* 分布均匀性：大量随机 session_id 覆盖全部分片；
* 批量扇出：``group_by_shard`` / ``group_messages`` 归组与去重；
* 端到端集成：SQLite 内存库上按路由写入并读回消息（含批量扇出读写）；
* 边界与异常：空 session_id、非法 shard_count。

全部用例跑在 SQLite 内存库上，无需真实 MySQL。
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from typing import Any

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="SQLAlchemy is an optional dependency")

from sqlalchemy import Engine, create_engine, insert, select  # noqa: E402

from med_langchain_memory.domain.message import MedMessage, MessageRole  # noqa: E402
from med_langchain_memory.exceptions import ValidationError  # noqa: E402
from med_langchain_memory.stores.mysql_schema import (  # noqa: E402
    SHARD_COUNT,
    create_all,
    message_table,
    message_table_name,
    message_to_row,
)
from med_langchain_memory.stores.mysql_shard_router import (  # noqa: E402
    DEFAULT_ROUTER,
    ShardRouter,
)

TENANT = "hosp-001"
DEPT = "cardio"
PATIENT = "pat-9527"


def _make_message(session_id: str, **overrides: Any) -> MedMessage:
    """构造一条属于指定会话的测试用医疗消息。"""
    fields: dict[str, Any] = {
        "session_id": session_id,
        "tenant_id": TENANT,
        "dept_id": DEPT,
        "patient_id": PATIENT,
        "role": MessageRole.PATIENT,
        "content": "chest pain",
        "token_count": 5,
        "metadata": {"channel": "app"},
    }
    fields.update(overrides)
    return MedMessage(**fields)


@pytest.fixture
def engine() -> Iterator[Engine]:
    """提供一个已建表的 SQLite 内存库引擎。"""
    eng = create_engine("sqlite+pysqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


# --------------------------------------------------------------------------- #
# 构造与校验
# --------------------------------------------------------------------------- #
class TestConstruction:
    """路由器构造与参数校验。"""

    def test_default_shard_count_is_16(self) -> None:
        router = ShardRouter()
        assert router.shard_count == SHARD_COUNT == 16

    def test_custom_shard_count(self) -> None:
        router = ShardRouter(shard_count=8)
        assert router.shard_count == 8

    def test_shard_count_of_one(self) -> None:
        router = ShardRouter(shard_count=1)
        assert router.shard_count == 1

    @pytest.mark.parametrize("shard_count", [0, -1, 17, 100])
    def test_rejects_shard_count_out_of_range(self, shard_count: int) -> None:
        with pytest.raises(ValidationError, match="shard_count must be in"):
            ShardRouter(shard_count=shard_count)

    def test_all_shards_returns_full_range(self) -> None:
        router = ShardRouter()
        assert router.all_shards() == list(range(16))

    def test_all_shards_respects_custom_count(self) -> None:
        router = ShardRouter(shard_count=4)
        assert router.all_shards() == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# 路由一致性
# --------------------------------------------------------------------------- #
class TestRoutingConsistency:
    """同一 session_id 始终映射到同一分片。"""

    def test_same_session_returns_same_shard(self) -> None:
        router = ShardRouter()
        assert router.shard_of("sess-0001") == router.shard_of("sess-0001")

    @pytest.mark.parametrize(
        "session_id",
        ["sess-0001", "patient-visit-abc", "a", "session.with.dots", "UPPER-CASE-123"],
    )
    def test_shard_is_within_valid_range(self, session_id: str) -> None:
        router = ShardRouter()
        shard = router.shard_of(session_id)
        assert 0 <= shard < SHARD_COUNT

    def test_shard_matches_crc32_formula(self) -> None:
        """验证路由公式 ``crc32(session_id) % 16`` 与规范一致。"""
        router = ShardRouter()
        for sid in ["sess-001", "sess-002", "sess-003"]:
            expected = zlib.crc32(sid.encode("utf-8")) % SHARD_COUNT
            assert router.shard_of(sid) == expected

    def test_different_sessions_can_map_to_different_shards(self) -> None:
        router = ShardRouter()
        shards = {router.shard_of(f"session-{i:04d}") for i in range(200)}
        assert len(shards) > 1

    def test_shard_count_one_forces_all_to_shard_zero(self) -> None:
        router = ShardRouter(shard_count=1)
        for i in range(50):
            assert router.shard_of(f"session-{i}") == 0


# --------------------------------------------------------------------------- #
# 表名与表对象
# --------------------------------------------------------------------------- #
class TestTableResolution:
    """分表名与表对象查找。"""

    def test_table_name_of_matches_shard(self) -> None:
        router = ShardRouter()
        sid = "sess-0001"
        shard = router.shard_of(sid)
        assert router.table_name_of(sid) == message_table_name(shard)

    def test_table_name_format(self) -> None:
        router = ShardRouter()
        name = router.table_name_of("sess-0001")
        assert name.startswith("med_message_")
        suffix = int(name.split("_")[-1])
        assert 0 <= suffix < SHARD_COUNT

    def test_table_of_returns_correct_table_object(self) -> None:
        router = ShardRouter()
        sid = "sess-0001"
        shard = router.shard_of(sid)
        assert router.table_of(sid) is message_table(shard)

    def test_table_of_name_matches_table_name_of(self) -> None:
        router = ShardRouter()
        sid = "patient-visit-xyz"
        assert router.table_of(sid).name == router.table_name_of(sid)

    def test_same_session_same_table_across_calls(self) -> None:
        router = ShardRouter()
        first = router.table_of("sess-0001")
        second = router.table_of("sess-0001")
        assert first is second


# --------------------------------------------------------------------------- #
# 边界与异常
# --------------------------------------------------------------------------- #
class TestEdgeCases:
    """空值与异常输入。"""

    def test_empty_session_id_raises(self) -> None:
        router = ShardRouter()
        with pytest.raises(ValidationError, match="session_id must not be empty"):
            router.shard_of("")

    def test_empty_session_id_raises_for_table_name(self) -> None:
        router = ShardRouter()
        with pytest.raises(ValidationError, match="session_id must not be empty"):
            router.table_name_of("")

    def test_empty_session_id_raises_for_table(self) -> None:
        router = ShardRouter()
        with pytest.raises(ValidationError, match="session_id must not be empty"):
            router.table_of("")


# --------------------------------------------------------------------------- #
# 分布均匀性
# --------------------------------------------------------------------------- #
class TestDistribution:
    """大量 session_id 的分片分布应覆盖全部 16 张表。"""

    def test_all_shards_are_covered(self) -> None:
        router = ShardRouter()
        shards = {router.shard_of(f"session-{i:04d}") for i in range(2000)}
        assert shards == set(range(SHARD_COUNT))

    def test_distribution_is_reasonably_uniform(self) -> None:
        """每张分表至少承接 2000 个会话的 3%。"""
        router = ShardRouter()
        counts: dict[int, int] = {s: 0 for s in range(SHARD_COUNT)}
        for i in range(2000):
            counts[router.shard_of(f"session-{i:04d}")] += 1
        threshold = 2000 * 0.03
        for shard, count in counts.items():
            assert count >= threshold, f"shard {shard} underfilled: {count}"


# --------------------------------------------------------------------------- #
# 端到端集成：SQLite 内存库
# --------------------------------------------------------------------------- #
class TestEndToEndWithSqlite:
    """按路由写入消息分表并读回，验证路由与 schema 的端到端一致性。"""

    def test_write_and_read_through_router(self, engine: Engine) -> None:
        router = ShardRouter()
        sid = "sess-e2e-001"
        message = _make_message(sid, content="headache for 3 days")

        table = router.table_of(sid)
        with engine.begin() as conn:
            conn.execute(insert(table).values(message_to_row(message)))
        with engine.connect() as conn:
            row = conn.execute(select(table)).mappings().one()

        assert row["message_id"] == message.message_id
        assert row["session_id"] == sid
        assert row["content"] == "headache for 3 days"

    def test_same_session_messages_land_in_same_table(self, engine: Engine) -> None:
        router = ShardRouter()
        sid = "sess-e2e-002"
        msgs = [_make_message(sid, content=f"msg-{i}") for i in range(5)]

        table = router.table_of(sid)
        with engine.begin() as conn:
            conn.execute(insert(table), [message_to_row(m) for m in msgs])

        with engine.connect() as conn:
            rows = conn.execute(select(table).order_by(table.c.created_at)).mappings().all()
        assert len(rows) == 5
        assert [r["content"] for r in rows] == [f"msg-{i}" for i in range(5)]

    def test_different_sessions_use_correct_tables(self, engine: Engine) -> None:
        """两条不同会话的消息各写各的分表，互不干扰。"""
        router = ShardRouter()
        sid_a = "sess-e2e-003"
        sid_b = "sess-e2e-004"
        msg_a = _make_message(sid_a, content="from session A")
        msg_b = _make_message(sid_b, content="from session B")

        table_a = router.table_of(sid_a)
        table_b = router.table_of(sid_b)

        with engine.begin() as conn:
            conn.execute(insert(table_a).values(message_to_row(msg_a)))
            conn.execute(insert(table_b).values(message_to_row(msg_b)))

        with engine.connect() as conn:
            rows_a = conn.execute(select(table_a)).mappings().all()
            rows_b = conn.execute(select(table_b)).mappings().all()

        if table_a is table_b:
            assert len(rows_a) == len(rows_b) == 2
        else:
            assert len(rows_a) == 1
            assert rows_a[0]["content"] == "from session A"
            assert len(rows_b) == 1
            assert rows_b[0]["content"] == "from session B"

    def test_router_table_name_matches_actual_table_in_db(self, engine: Engine) -> None:
        """路由给出的表名在已建表的引擎中真实存在。"""
        router = ShardRouter()
        for sid in [f"session-{i:04d}" for i in range(100)]:
            name = router.table_name_of(sid)
            with engine.connect() as conn:
                assert engine.dialect.has_table(conn, name), f"missing table {name}"


# --------------------------------------------------------------------------- #
# 分表清单
# --------------------------------------------------------------------------- #
class TestAllTables:
    """``all_tables`` 返回路由范围内的全部分表。"""

    def test_returns_all_16_tables_in_order(self) -> None:
        tables = ShardRouter().all_tables()
        assert len(tables) == SHARD_COUNT
        assert [t.name for t in tables] == [message_table_name(i) for i in range(SHARD_COUNT)]

    def test_returns_same_table_objects_as_schema(self) -> None:
        tables = ShardRouter().all_tables()
        assert all(tables[i] is message_table(i) for i in range(SHARD_COUNT))

    def test_respects_custom_shard_count(self) -> None:
        tables = ShardRouter(shard_count=3).all_tables()
        assert [t.name for t in tables] == ["med_message_00", "med_message_01", "med_message_02"]

    def test_single_shard_router_returns_one_table(self) -> None:
        assert [t.name for t in ShardRouter(shard_count=1).all_tables()] == ["med_message_00"]


# --------------------------------------------------------------------------- #
# 批量扇出：会话 ID 归组
# --------------------------------------------------------------------------- #
class TestGroupByShard:
    """``group_by_shard`` 把会话 ID 按分片归组，供批量查询扇出。"""

    def test_each_session_lands_in_its_own_shard_group(self) -> None:
        router = ShardRouter()
        sids = [f"session-{i:04d}" for i in range(50)]
        grouped = router.group_by_shard(sids)
        for shard, group in grouped.items():
            assert all(router.shard_of(sid) == shard for sid in group)

    def test_no_session_is_lost(self) -> None:
        router = ShardRouter()
        sids = [f"session-{i:04d}" for i in range(50)]
        grouped = router.group_by_shard(sids)
        assert sorted(sid for group in grouped.values() for sid in group) == sorted(sids)

    def test_deduplicates_repeated_session_ids(self) -> None:
        router = ShardRouter()
        grouped = router.group_by_shard(["sess-a", "sess-a", "sess-a"])
        assert list(grouped.values()) == [["sess-a"]]

    def test_preserves_first_seen_order_within_group(self) -> None:
        """单分片路由下，组内顺序应与首次出现顺序一致。"""
        router = ShardRouter(shard_count=1)
        grouped = router.group_by_shard(["c", "a", "b", "a"])
        assert grouped == {0: ["c", "a", "b"]}

    def test_keys_are_sorted_and_groups_non_empty(self) -> None:
        router = ShardRouter()
        grouped = router.group_by_shard([f"session-{i:04d}" for i in range(200)])
        assert list(grouped) == sorted(grouped)
        assert all(group for group in grouped.values())

    def test_empty_input_returns_empty_mapping(self) -> None:
        assert ShardRouter().group_by_shard([]) == {}

    def test_accepts_generator_input(self) -> None:
        router = ShardRouter(shard_count=1)
        grouped = router.group_by_shard(f"sess-{i}" for i in range(3))
        assert grouped == {0: ["sess-0", "sess-1", "sess-2"]}

    def test_rejects_empty_session_id(self) -> None:
        with pytest.raises(ValidationError, match="session_id must not be empty"):
            ShardRouter().group_by_shard(["sess-a", ""])


# --------------------------------------------------------------------------- #
# 批量扇出：消息归组
# --------------------------------------------------------------------------- #
class TestGroupMessages:
    """``group_messages`` 把待写消息按分片归组，供批量插入。"""

    def test_messages_are_grouped_by_session_shard(self) -> None:
        router = ShardRouter()
        messages = [_make_message(f"session-{i:04d}") for i in range(30)]
        grouped = router.group_messages(messages)
        for shard, group in grouped.items():
            assert all(router.shard_of(m.session_id) == shard for m in group)

    def test_no_message_is_lost(self) -> None:
        router = ShardRouter()
        messages = [_make_message(f"session-{i:04d}") for i in range(30)]
        grouped = router.group_messages(messages)
        assert sum(len(group) for group in grouped.values()) == 30

    def test_same_session_messages_stay_in_one_group_in_order(self) -> None:
        router = ShardRouter()
        messages = [_make_message("sess-same", content=f"msg-{i}") for i in range(4)]
        grouped = router.group_messages(messages)
        assert len(grouped) == 1
        (group,) = grouped.values()
        assert [m.content for m in group] == [f"msg-{i}" for i in range(4)]

    def test_distinct_sessions_may_merge_into_one_shard(self) -> None:
        router = ShardRouter(shard_count=1)
        grouped = router.group_messages([_make_message("sess-a"), _make_message("sess-b")])
        assert list(grouped) == [0]
        assert [m.session_id for m in grouped[0]] == ["sess-a", "sess-b"]

    def test_keys_are_sorted(self) -> None:
        router = ShardRouter()
        grouped = router.group_messages([_make_message(f"session-{i:04d}") for i in range(100)])
        assert list(grouped) == sorted(grouped)

    def test_empty_input_returns_empty_mapping(self) -> None:
        assert ShardRouter().group_messages([]) == {}

    def test_rejects_message_with_empty_session_id(self) -> None:
        broken = _make_message("sess-a").model_copy(update={"session_id": ""})
        with pytest.raises(ValidationError, match="session_id must not be empty"):
            ShardRouter().group_messages([broken])


# --------------------------------------------------------------------------- #
# 进程内默认路由器
# --------------------------------------------------------------------------- #
class TestDefaultRouter:
    """``DEFAULT_ROUTER`` 为生产默认的 16 分表路由器。"""

    def test_is_a_router_with_full_shard_count(self) -> None:
        assert isinstance(DEFAULT_ROUTER, ShardRouter)
        assert DEFAULT_ROUTER.shard_count == SHARD_COUNT

    def test_routes_identically_to_fresh_router(self) -> None:
        fresh = ShardRouter()
        for i in range(50):
            sid = f"session-{i:04d}"
            assert DEFAULT_ROUTER.shard_of(sid) == fresh.shard_of(sid)

    def test_is_exported_from_stores_package(self) -> None:
        from med_langchain_memory import stores

        assert stores.DEFAULT_ROUTER is DEFAULT_ROUTER
        assert stores.ShardRouter is ShardRouter

    def test_rejects_empty_session_id(self) -> None:
        with pytest.raises(ValidationError, match="session_id must not be empty"):
            DEFAULT_ROUTER.shard_of("")


# --------------------------------------------------------------------------- #
# 端到端：批量扇出写入 / 读取
# --------------------------------------------------------------------------- #
class TestFanOutWithSqlite:
    """按分片归组批量写入多张分表，再扇出读回，验证零丢失。"""

    def test_batch_write_and_fan_out_read(self, engine: Engine) -> None:
        router = ShardRouter()
        sessions = [f"fanout-{i:03d}" for i in range(40)]
        messages = [_make_message(sid, content=f"note-{sid}") for sid in sessions]

        grouped = router.group_messages(messages)
        with engine.begin() as conn:
            for shard, group in grouped.items():
                conn.execute(insert(message_table(shard)), [message_to_row(m) for m in group])

        recovered: list[str] = []
        with engine.connect() as conn:
            for shard, sids in router.group_by_shard(sessions).items():
                table = message_table(shard)
                stmt = select(table.c.session_id).where(table.c.session_id.in_(sids))
                recovered.extend(row[0] for row in conn.execute(stmt))

        assert sorted(recovered) == sorted(sessions)

    def test_fan_out_query_touches_only_matching_shards(self, engine: Engine) -> None:
        """一条会话的消息只出现在它自己的分表中，其余分表为空。"""
        router = ShardRouter()
        sid = "fanout-single"
        with engine.begin() as conn:
            conn.execute(insert(router.table_of(sid)).values(message_to_row(_make_message(sid))))

        with engine.connect() as conn:
            hits = [
                table.name
                for table in router.all_tables()
                if conn.execute(select(table.c.message_id).where(table.c.session_id == sid)).first()
                is not None
            ]
        assert hits == [router.table_name_of(sid)]
