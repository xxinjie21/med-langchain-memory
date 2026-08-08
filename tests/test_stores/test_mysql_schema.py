"""``stores.mysql_schema`` 单元测试。

全部用例跑在 SQLite 内存库上，无需真实 MySQL：
建表语句由同一份 :data:`METADATA` 渲染，能在 SQLite 建成并读写，即验证了结构自洽性；
MySQL 方言特有的产物（InnoDB / utf8mb4 / JSON 列）通过 :func:`render_ddl` 的文本断言覆盖。
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from typing import Any

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy", reason="SQLAlchemy is an optional dependency")

from sqlalchemy import Engine, create_engine, insert, select  # noqa: E402

from med_langchain_memory.domain.message import MedMessage, MessageRole  # noqa: E402
from med_langchain_memory.domain.session import SessionMeta, SessionStatus  # noqa: E402
from med_langchain_memory.exceptions import StorageError, ValidationError  # noqa: E402
from med_langchain_memory.stores.mysql_schema import (  # noqa: E402
    BASELINE_REVISION,
    MESSAGE_TABLE_PREFIX,
    MESSAGE_TABLES,
    METADATA,
    SCHEMA_VERSION_TABLE,
    SESSION_TABLE,
    SHARD_COUNT,
    all_tables,
    create_all,
    current_revision,
    message_from_row,
    message_table,
    message_table_name,
    message_to_row,
    render_ddl,
    session_from_row,
    session_to_row,
)

TENANT = "hosp-001"
DEPT = "cardio"
SESSION = "sess-0001"
PATIENT = "pat-9527"


@pytest.fixture
def engine() -> Iterator[Engine]:
    """提供一个已建表的 SQLite 内存库引擎。"""
    eng = create_engine("sqlite+pysqlite:///:memory:")
    create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def blank_engine() -> Iterator[Engine]:
    """提供一个未初始化模式的 SQLite 内存库引擎。"""
    eng = create_engine("sqlite+pysqlite:///:memory:")
    yield eng
    eng.dispose()


def _make_message(**overrides: Any) -> MedMessage:
    """构造一条测试用医疗消息。"""
    fields: dict[str, Any] = {
        "session_id": SESSION,
        "tenant_id": TENANT,
        "dept_id": DEPT,
        "patient_id": PATIENT,
        "role": MessageRole.PATIENT,
        "content": "持续胸闷三天",
        "token_count": 7,
        "masked": True,
        "metadata": {"channel": "app"},
    }
    fields.update(overrides)
    return MedMessage(**fields)


def _make_session(**overrides: Any) -> SessionMeta:
    """构造一份测试用会话元数据。"""
    fields: dict[str, Any] = {
        "session_id": SESSION,
        "tenant_id": TENANT,
        "dept_id": DEPT,
        "patient_id": PATIENT,
        "message_count": 3,
        "metadata": {"visit": "outpatient"},
    }
    fields.update(overrides)
    return SessionMeta(**fields)


class TestTableDefinitions:
    """表结构定义。"""

    def test_shard_tables_are_complete_and_named(self) -> None:
        assert len(MESSAGE_TABLES) == SHARD_COUNT
        assert set(MESSAGE_TABLES) == set(range(SHARD_COUNT))
        assert MESSAGE_TABLES[0].name == f"{MESSAGE_TABLE_PREFIX}_00"
        assert MESSAGE_TABLES[SHARD_COUNT - 1].name == f"{MESSAGE_TABLE_PREFIX}_15"

    def test_all_shard_tables_share_identical_columns(self) -> None:
        expected = {
            "message_id",
            "session_id",
            "tenant_id",
            "dept_id",
            "patient_id",
            "role",
            "content",
            "token_count",
            "masked",
            "created_at",
            "metadata",
        }
        for table in MESSAGE_TABLES.values():
            assert {c.name for c in table.columns} == expected
            assert [c.name for c in table.primary_key] == ["message_id"]

    def test_session_table_columns(self) -> None:
        assert {c.name for c in SESSION_TABLE.columns} == {
            "session_id",
            "tenant_id",
            "dept_id",
            "patient_id",
            "status",
            "message_count",
            "created_at",
            "updated_at",
            "metadata",
        }
        assert [c.name for c in SESSION_TABLE.primary_key] == ["session_id"]

    def test_index_names_are_globally_unique(self) -> None:
        names = [idx.name for table in all_tables() for idx in table.indexes]
        assert len(names) == len(set(names))
        assert all(name is not None and name.startswith("ix_") for name in names)

    def test_index_names_fit_mysql_identifier_limit(self) -> None:
        for table in all_tables():
            for index in table.indexes:
                assert index.name is not None
                assert len(index.name) <= 64

    def test_all_tables_ordering_and_membership(self) -> None:
        tables = all_tables()
        assert tables[0] is SCHEMA_VERSION_TABLE
        assert tables[1] is SESSION_TABLE
        assert [t.name for t in tables[2:]] == [message_table_name(i) for i in range(SHARD_COUNT)]
        assert {t.name for t in tables} == set(METADATA.tables)

    def test_message_table_uses_innodb_utf8mb4(self) -> None:
        table = message_table(0)
        assert table.kwargs["mysql_engine"] == "InnoDB"
        assert table.kwargs["mysql_charset"] == "utf8mb4"


class TestShardLookup:
    """分表名与分表对象查询。"""

    @pytest.mark.parametrize(
        ("shard", "expected"),
        [(0, "med_message_00"), (7, "med_message_07"), (15, "med_message_15")],
    )
    def test_message_table_name_pads_index(self, shard: int, expected: str) -> None:
        assert message_table_name(shard) == expected

    @pytest.mark.parametrize("shard", [-1, SHARD_COUNT, 99])
    def test_message_table_name_rejects_out_of_range(self, shard: int) -> None:
        with pytest.raises(ValidationError, match="shard must be in"):
            message_table_name(shard)

    def test_message_table_returns_registered_table(self) -> None:
        assert message_table(3).name == "med_message_03"
        assert message_table(3) is MESSAGE_TABLES[3]

    @pytest.mark.parametrize("shard", [-1, SHARD_COUNT])
    def test_message_table_rejects_out_of_range(self, shard: int) -> None:
        with pytest.raises(ValidationError, match="shard must be in"):
            message_table(shard)


class TestMessageRowMapping:
    """消息领域模型与数据行互转。"""

    def test_message_to_row_maps_every_column(self) -> None:
        message = _make_message()
        row = message_to_row(message)
        assert set(row) == {c.name for c in message_table(0).columns}
        assert row["role"] == "patient"
        assert row["masked"] is True
        assert row["metadata"] == {"channel": "app"}

    def test_message_roundtrip_through_sqlite(self, engine: Engine) -> None:
        message = _make_message()
        table = message_table(5)
        with engine.begin() as conn:
            conn.execute(insert(table).values(message_to_row(message)))
        with engine.connect() as conn:
            row = conn.execute(select(table)).mappings().one()
        assert message_from_row(row) == message

    def test_message_roundtrip_on_every_shard(self, engine: Engine) -> None:
        for shard in range(SHARD_COUNT):
            table = message_table(shard)
            message = _make_message(content=f"shard {shard}")
            with engine.begin() as conn:
                conn.execute(insert(table).values(message_to_row(message)))
            with engine.connect() as conn:
                row = conn.execute(select(table)).mappings().one()
            assert message_from_row(row).content == f"shard {shard}"

    def test_message_from_row_accepts_missing_metadata(self) -> None:
        row = message_to_row(_make_message())
        row["metadata"] = None
        assert message_from_row(row).metadata == {}

    def test_message_from_row_rejects_unknown_role(self) -> None:
        row = message_to_row(_make_message())
        row["role"] = "nurse"
        with pytest.raises(StorageError, match="corrupted message row"):
            message_from_row(row)

    def test_message_from_row_rejects_missing_column(self) -> None:
        row = message_to_row(_make_message())
        del row["content"]
        with pytest.raises(StorageError, match="corrupted message row"):
            message_from_row(row)

    def test_message_from_row_rejects_non_mapping_metadata(self) -> None:
        row = message_to_row(_make_message())
        row["metadata"] = "not-a-map"
        with pytest.raises(StorageError, match="corrupted message row"):
            message_from_row(row)

    def test_message_from_row_rejects_invalid_field_value(self) -> None:
        row = message_to_row(_make_message())
        row["token_count"] = -1
        with pytest.raises(StorageError, match="corrupted message row"):
            message_from_row(row)


class TestSessionRowMapping:
    """会话元数据与数据行互转。"""

    def test_session_to_row_maps_every_column(self) -> None:
        row = session_to_row(_make_session())
        assert set(row) == {c.name for c in SESSION_TABLE.columns}
        assert row["status"] == "active"
        assert row["message_count"] == 3

    def test_session_roundtrip_through_sqlite(self, engine: Engine) -> None:
        meta = _make_session()
        with engine.begin() as conn:
            conn.execute(insert(SESSION_TABLE).values(session_to_row(meta)))
        with engine.connect() as conn:
            row = conn.execute(select(SESSION_TABLE)).mappings().one()
        assert session_from_row(row) == meta

    def test_session_roundtrip_preserves_archived_status(self, engine: Engine) -> None:
        meta = _make_session(status=SessionStatus.ARCHIVED)
        with engine.begin() as conn:
            conn.execute(insert(SESSION_TABLE).values(session_to_row(meta)))
        with engine.connect() as conn:
            row = conn.execute(select(SESSION_TABLE)).mappings().one()
        assert session_from_row(row).status is SessionStatus.ARCHIVED

    def test_session_from_row_rejects_unknown_status(self) -> None:
        row = session_to_row(_make_session())
        row["status"] = "paused"
        with pytest.raises(StorageError, match="corrupted session row"):
            session_from_row(row)

    def test_session_from_row_rejects_missing_column(self) -> None:
        row = session_to_row(_make_session())
        del row["tenant_id"]
        with pytest.raises(StorageError, match="corrupted session row"):
            session_from_row(row)

    def test_session_from_row_rejects_updated_before_created(self) -> None:
        row = session_to_row(_make_session())
        row["updated_at"] = row["created_at"] - 1000
        with pytest.raises(StorageError, match="corrupted session row"):
            session_from_row(row)


class TestRenderDdl:
    """DDL 渲染。"""

    def test_mysql_ddl_contains_all_tables(self) -> None:
        ddl = render_ddl("mysql")
        assert "CREATE TABLE med_session" in ddl
        assert "CREATE TABLE med_schema_version" in ddl
        for shard in range(SHARD_COUNT):
            assert f"CREATE TABLE {message_table_name(shard)}" in ddl

    def test_mysql_ddl_uses_innodb_and_utf8mb4(self) -> None:
        ddl = render_ddl("mysql")
        assert ddl.count("ENGINE=InnoDB") == SHARD_COUNT + 2
        assert "CHARSET=utf8mb4" in ddl
        assert "BIGINT" in ddl
        assert "JSON" in ddl

    def test_mysql_ddl_creates_every_index(self) -> None:
        ddl = render_ddl("mysql")
        expected = sum(len(table.indexes) for table in all_tables())
        assert len(re.findall(r"CREATE INDEX ", ddl)) == expected

    def test_mysql_ddl_statements_are_terminated(self) -> None:
        statements = [s for s in render_ddl("mysql").split("\n\n") if s.strip()]
        assert statements
        assert all(s.strip().endswith(";") for s in statements)

    def test_sqlite_dialect_is_supported(self) -> None:
        ddl = render_ddl("sqlite")
        assert "CREATE TABLE med_session" in ddl
        assert "ENGINE=InnoDB" not in ddl

    def test_render_ddl_rejects_unknown_dialect(self) -> None:
        with pytest.raises(ValidationError, match="unsupported dialect"):
            render_ddl("oracle")


class TestBaseline:
    """建表与迁移基线登记。"""

    def test_create_all_creates_tables_and_records_revision(self, blank_engine: Engine) -> None:
        assert create_all(blank_engine) == BASELINE_REVISION
        assert current_revision(blank_engine) == BASELINE_REVISION
        with blank_engine.connect() as conn:
            for table in all_tables():
                assert blank_engine.dialect.has_table(conn, table.name)

    def test_create_all_is_idempotent(self, blank_engine: Engine) -> None:
        create_all(blank_engine)
        create_all(blank_engine)
        with blank_engine.connect() as conn:
            rows = conn.execute(select(SCHEMA_VERSION_TABLE)).all()
        assert len(rows) == 1
        assert rows[0][0] == BASELINE_REVISION

    def test_current_revision_returns_none_before_init(self, blank_engine: Engine) -> None:
        assert current_revision(blank_engine) is None

    def test_baseline_revision_format(self) -> None:
        assert re.fullmatch(r"\d{4}_[a-z_]+", BASELINE_REVISION)
