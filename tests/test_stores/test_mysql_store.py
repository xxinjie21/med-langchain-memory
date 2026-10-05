"""MySQLMedHistory 单元测试。

分五部分：

* :class:`TestMySQLStoreBehavior` 复用跨后端共享行为套件，校验通用存储契约；
* 分表路由用例：``crc32(session_id) % 16`` 落表位置、同会话不跨分片、自定义路由器；
* 保序用例：``ordinal`` 同毫秒保序、跨批次单调递增、``created_at`` 优先于 ``ordinal``；
* 会话元数据用例：``med_session`` 行 upsert、归档/软删除状态回写、``count`` 与 ``fetch_session_meta``；
* 构造与健壮性用例：TTL 拒绝、惰性建引擎、``ensure_schema`` 幂等、工厂注册与异常包装。

全部用例跑在 **SQLite 内存库**（``StaticPool`` 共享同一内存实例）上，
建表复用生产同一份 :data:`~med_langchain_memory.stores.mysql_schema.METADATA`，
无需真实 MySQL 即可运行。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from behavior import MedHistoryBehaviorSuite

pytest.importorskip("sqlalchemy", reason="SQLAlchemy is an optional dependency")

from sqlalchemy import Engine, create_engine, insert, select  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from med_langchain_memory.domain import MedMessage, MessageRole, SessionStatus  # noqa: E402
from med_langchain_memory.exceptions import StorageError, ValidationError  # noqa: E402
from med_langchain_memory.stores import StoreConfig, StoreFactory  # noqa: E402
from med_langchain_memory.stores.mysql_schema import (  # noqa: E402
    BASELINE_REVISION,
    MESSAGE_TABLES,
    SESSION_TABLE,
    create_all,
    message_table,
)
from med_langchain_memory.stores.mysql_shard_router import DEFAULT_ROUTER, ShardRouter  # noqa: E402
from med_langchain_memory.stores.mysql_store import (  # noqa: E402
    DEFAULT_MYSQL_URL,
    MySQLMedHistory,
)

NAMESPACE = {
    "session_id": "s-mysql",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}
STORAGE_KEY = "med:chat:hospital_a:cardiology:s-mysql"

#: 固定时间戳（2023-11-14T22:13:20Z），便于构造同毫秒消息。
FIXED_MS = 1_700_000_000_000


def make_engine(with_schema: bool = True) -> Engine:
    """创建独立的 SQLite 内存库引擎。

    ``StaticPool`` 保证所有连接复用同一个内存数据库实例，
    否则每次 ``engine.connect()`` 都会拿到一个空库。
    """
    engine = create_engine(
        "sqlite+pysqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    if with_schema:
        create_all(engine)
    return engine


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


def raw_rows(engine: Engine, session_id: str) -> list[dict[str, Any]]:
    """直接读取指定会话所在分表的原始行（按 ordinal 升序）。"""
    table = message_table(DEFAULT_ROUTER.shard_of(session_id))
    statement = select(table).where(table.c.session_id == session_id).order_by(table.c.ordinal)
    with engine.connect() as conn:
        return [dict(row) for row in conn.execute(statement).mappings().all()]


@pytest.fixture
def engine() -> Iterator[Engine]:
    """已建表的独立 SQLite 内存库引擎。"""
    eng = make_engine()
    yield eng
    eng.dispose()


@pytest.fixture
def history(engine: Engine) -> MySQLMedHistory:
    """默认命名空间下的 MySQL 会话历史。"""
    return MySQLMedHistory(**NAMESPACE, engine=engine)


# --------------------------------------------------------------------------- #
# 一、跨后端共享行为契约
# --------------------------------------------------------------------------- #
class TestMySQLStoreBehavior(MedHistoryBehaviorSuite):
    """MySQL 分表后端必须满足全部通用存储行为契约。"""

    backend_name = "mysql"
    shared_across_handles = True

    @pytest.fixture(autouse=True)
    def _isolated_engine(self) -> Iterator[None]:
        """每个用例使用独立的 SQLite 内存库。"""
        self.engine = make_engine()
        yield
        self.engine.dispose()

    def make_history(self, **overrides: Any) -> MySQLMedHistory:
        """构造 MySQL 会话历史，复用当前用例的引擎。"""
        kwargs: dict[str, Any] = {**self.NAMESPACE, "engine": self.engine}
        kwargs.update(overrides)
        return MySQLMedHistory(**kwargs)


# --------------------------------------------------------------------------- #
# 二、分表路由
# --------------------------------------------------------------------------- #
class TestShardRouting:
    """消息按 ``crc32(session_id) % 16`` 落到唯一分表。"""

    def test_shard_and_table_name_match_default_router(self, history: MySQLMedHistory) -> None:
        assert history.shard == DEFAULT_ROUTER.shard_of(NAMESPACE["session_id"])
        assert history.table_name == DEFAULT_ROUTER.table_name_of(NAMESPACE["session_id"])
        assert history.table_name.startswith("med_message_")

    def test_router_property_exposes_injected_router(self, engine: Engine) -> None:
        router = ShardRouter(shard_count=4)
        history = MySQLMedHistory(**NAMESPACE, engine=engine, router=router)

        assert history.router is router
        assert history.shard < 4
        assert history.table_name == router.table_name_of(NAMESPACE["session_id"])

    def test_writes_land_only_in_the_routed_shard(self, engine: Engine) -> None:
        history = MySQLMedHistory(**NAMESPACE, engine=engine)
        history.add_med_messages([make_message("a"), make_message("b")])

        shard = history.shard
        for number, table in MESSAGE_TABLES.items():
            with engine.connect() as conn:
                count = len(conn.execute(select(table)).all())
            assert count == (2 if number == shard else 0), f"unexpected rows in {table.name}"

    def test_two_appends_keep_one_session_in_one_shard(self, engine: Engine) -> None:
        history = MySQLMedHistory(**NAMESPACE, engine=engine)
        history.add_med_messages([make_message("first")])
        history.add_med_messages([make_message("second")])

        assert len(raw_rows(engine, NAMESPACE["session_id"])) == 2

    def test_different_sessions_may_share_a_shard(self, engine: Engine) -> None:
        """同一分表内的不同会话互不可见（``session_id`` 过滤生效）。"""
        first = MySQLMedHistory(**NAMESPACE, engine=engine)
        second = MySQLMedHistory(**{**NAMESPACE, "session_id": "s-other"}, engine=engine)
        first.add_med_messages([make_message("mine")])

        assert [m.content for m in first.get_med_messages()] == ["mine"]
        assert second.get_med_messages() == []


# --------------------------------------------------------------------------- #
# 三、写入顺序（ordinal）
# --------------------------------------------------------------------------- #
class TestOrdinalOrdering:
    """同毫秒消息依靠存储层分配的 ``ordinal`` 保序。"""

    def test_same_millisecond_messages_keep_insertion_order(self, history: MySQLMedHistory) -> None:
        messages = [make_message(f"m{index}", created_at=FIXED_MS) for index in range(4)]

        history.add_med_messages(messages)

        assert [m.content for m in history.get_med_messages()] == ["m0", "m1", "m2", "m3"]

    def test_ordinal_starts_at_one_and_increments_within_batch(
        self, engine: Engine, history: MySQLMedHistory
    ) -> None:
        history.add_med_messages([make_message("a"), make_message("b")])

        assert [row["ordinal"] for row in raw_rows(engine, NAMESPACE["session_id"])] == [1, 2]

    def test_ordinal_is_monotonic_across_appends(
        self, engine: Engine, history: MySQLMedHistory
    ) -> None:
        history.add_med_messages([make_message("a"), make_message("b")])
        history.add_med_messages([make_message("c")])

        assert [row["ordinal"] for row in raw_rows(engine, NAMESPACE["session_id"])] == [1, 2, 3]

    def test_created_at_takes_precedence_over_ordinal(
        self, engine: Engine, history: MySQLMedHistory
    ) -> None:
        later = make_message("later", created_at=FIXED_MS + 9_000)
        earlier = make_message("earlier", created_at=FIXED_MS + 1_000)

        history.add_med_messages([later, earlier])

        assert [m.content for m in history.get_med_messages()] == ["earlier", "later"]
        assert [row["ordinal"] for row in raw_rows(engine, NAMESPACE["session_id"])] == [1, 2]

    def test_clear_then_append_restarts_ordinal(
        self, engine: Engine, history: MySQLMedHistory
    ) -> None:
        history.add_med_messages([make_message("a"), make_message("b")])

        history.clear()
        history.add_med_messages([make_message("c")])

        assert [row["ordinal"] for row in raw_rows(engine, NAMESPACE["session_id"])] == [1]


# --------------------------------------------------------------------------- #
# 四、会话元数据行
# --------------------------------------------------------------------------- #
class TestSessionRow:
    """``med_session`` 会话行的 upsert、状态回写与统计。"""

    def test_fetch_session_meta_returns_none_before_first_write(
        self, history: MySQLMedHistory
    ) -> None:
        assert history.fetch_session_meta() is None

    def test_append_creates_session_row(self, engine: Engine, history: MySQLMedHistory) -> None:
        history.add_med_messages([make_message("a"), make_message("b")])

        meta = history.fetch_session_meta()
        assert meta is not None
        assert meta.session_id == NAMESPACE["session_id"]
        assert meta.message_count == 2
        assert meta.status is SessionStatus.ACTIVE

    def test_second_append_updates_existing_row(
        self, engine: Engine, history: MySQLMedHistory
    ) -> None:
        history.add_med_messages([make_message("a")])
        history.add_med_messages([make_message("b"), make_message("c")])

        meta = history.fetch_session_meta()
        assert meta is not None
        assert meta.message_count == 3
        with engine.connect() as conn:
            assert len(conn.execute(select(SESSION_TABLE)).all()) == 1

    def test_session_row_tracks_updated_at(self, engine: Engine, history: MySQLMedHistory) -> None:
        history.add_med_messages([make_message("a")])

        meta = history.fetch_session_meta()
        assert meta is not None
        assert meta.updated_at >= meta.created_at

    def test_archive_persists_archived_status(self, history: MySQLMedHistory) -> None:
        history.add_med_messages([make_message("a")])

        history.archive()

        meta = history.fetch_session_meta()
        assert meta is not None
        assert meta.status is SessionStatus.ARCHIVED
        assert meta.message_count == 1

    def test_delete_persists_deleted_status(self, history: MySQLMedHistory) -> None:
        history.add_med_messages([make_message("a")])
        history.archive()

        history.delete()

        meta = history.fetch_session_meta()
        assert meta is not None
        assert meta.status is SessionStatus.DELETED
        assert history.get_med_messages() != [], "软删除不得清理消息数据"

    def test_count_tracks_written_messages(self, history: MySQLMedHistory) -> None:
        assert history.count() == 0

        history.add_med_messages([make_message("a"), make_message("b")])

        assert history.count() == 2

    def test_clear_removes_messages_and_session_row(
        self, engine: Engine, history: MySQLMedHistory
    ) -> None:
        history.add_med_messages([make_message("a")])

        history.clear()

        assert history.count() == 0
        assert history.fetch_session_meta() is None
        with engine.connect() as conn:
            assert conn.execute(select(SESSION_TABLE)).all() == []

    def test_fetch_session_meta_rejects_corrupted_row(self, engine: Engine) -> None:
        history = MySQLMedHistory(**NAMESPACE, engine=engine)
        row = {
            "session_id": NAMESPACE["session_id"],
            "tenant_id": NAMESPACE["tenant_id"],
            "dept_id": NAMESPACE["dept_id"],
            "patient_id": NAMESPACE["patient_id"],
            "status": "paused",
            "message_count": 0,
            "created_at": FIXED_MS,
            "updated_at": FIXED_MS,
            "metadata": {},
        }
        with engine.begin() as conn:
            conn.execute(insert(SESSION_TABLE).values(row))

        with pytest.raises(StorageError, match="corrupted session row"):
            history.fetch_session_meta()


# --------------------------------------------------------------------------- #
# 五、构造、工厂与异常包装
# --------------------------------------------------------------------------- #
class TestConstructorAndRegistration:
    """构造参数校验、惰性建引擎与工厂注册。"""

    def test_default_url_targets_mysql(self) -> None:
        assert DEFAULT_MYSQL_URL.startswith("mysql+")
        assert "localhost" in DEFAULT_MYSQL_URL

    def test_storage_key_matches_namespace(self, history: MySQLMedHistory) -> None:
        assert history.storage_key == STORAGE_KEY

    def test_ttl_is_rejected(self, engine: Engine) -> None:
        with pytest.raises(StorageError, match="does not support native ttl"):
            MySQLMedHistory(**NAMESPACE, engine=engine, ttl_seconds=60)

    def test_ensure_schema_flag_builds_tables(self) -> None:
        engine = make_engine(with_schema=False)
        try:
            history = MySQLMedHistory(**NAMESPACE, engine=engine, ensure_schema=True)
            assert history.count() == 0
        finally:
            engine.dispose()

    def test_ensure_schema_is_idempotent(self, engine: Engine) -> None:
        history = MySQLMedHistory(**NAMESPACE, engine=engine)

        assert history.ensure_schema() == BASELINE_REVISION
        assert history.ensure_schema() == BASELINE_REVISION

    def test_engine_is_created_lazily_from_url(self, tmp_path: Path) -> None:
        url = f"sqlite+pysqlite:///{(tmp_path / 'med.db').as_posix()}"
        history = MySQLMedHistory(**NAMESPACE, url=url)

        assert history.ensure_schema() == BASELINE_REVISION
        assert history.count() == 0
        history.engine.dispose()

    def test_unusable_url_is_wrapped_as_storage_error(self) -> None:
        history = MySQLMedHistory(**NAMESPACE, url="nosuchdialect://localhost/db")

        with pytest.raises(StorageError, match="mysql engine failed"):
            history.count()

    def test_backend_is_registered(self) -> None:
        assert StoreFactory.is_registered("mysql")
        assert StoreFactory.get("mysql") is MySQLMedHistory
        assert "mysql" in StoreFactory.available()

    def test_factory_create_with_engine_option(self, engine: Engine) -> None:
        history = StoreFactory.create("mysql", **NAMESPACE, engine=engine)

        assert isinstance(history, MySQLMedHistory)
        assert history.get_med_messages() == []

    def test_factory_create_from_config(self, engine: Engine) -> None:
        config = StoreConfig(backend="mysql", options={"engine": engine})

        history = StoreFactory.create_from_config(config, **NAMESPACE)

        assert isinstance(history, MySQLMedHistory)

    def test_factory_create_rejects_unknown_option(self) -> None:
        with pytest.raises(StorageError, match="cannot build MySQLMedHistory"):
            StoreFactory.create("mysql", **NAMESPACE, unknown_option=1)


class TestErrorWrapping:
    """底层 SQLAlchemy 异常统一包装为 :class:`StorageError`。"""

    @pytest.fixture
    def broken(self) -> Iterator[MySQLMedHistory]:
        """指向未建表数据库的历史句柄。"""
        engine = make_engine(with_schema=False)
        yield MySQLMedHistory(**NAMESPACE, engine=engine)
        engine.dispose()

    def test_read_without_schema_raises_storage_error(self, broken: MySQLMedHistory) -> None:
        with pytest.raises(StorageError, match="mysql read failed"):
            broken.get_med_messages()

    def test_append_without_schema_raises_storage_error(self, broken: MySQLMedHistory) -> None:
        with pytest.raises(StorageError, match="mysql append failed"):
            broken.add_med_messages([make_message("a")])

    def test_clear_without_schema_raises_storage_error(self, broken: MySQLMedHistory) -> None:
        with pytest.raises(StorageError, match="mysql clear failed"):
            broken.clear()

    def test_count_without_schema_raises_storage_error(self, broken: MySQLMedHistory) -> None:
        with pytest.raises(StorageError, match="mysql count failed"):
            broken.count()

    def test_fetch_session_meta_without_schema_raises_storage_error(
        self, broken: MySQLMedHistory
    ) -> None:
        with pytest.raises(StorageError, match="mysql fetch meta failed"):
            broken.fetch_session_meta()

    def test_persist_status_failure_is_wrapped(self, engine: Engine) -> None:
        """消息表正常但会话表缺失时，归档回写失败必须报错而非静默。"""
        history = MySQLMedHistory(**NAMESPACE, engine=engine)
        history.add_med_messages([make_message("a")])
        SESSION_TABLE.drop(engine)

        with pytest.raises(StorageError, match="mysql persist status failed"):
            history.archive()

    def test_validation_error_is_not_wrapped(self, history: MySQLMedHistory) -> None:
        """非 SQLAlchemy 异常（参数校验）原样抛出，不被误包装。"""
        with pytest.raises(ValidationError, match="limit must be"):
            history.get_med_messages(limit=0)
