"""真实 MySQL 集成测试（默认跳过，见 ``conftest.py``）。

覆盖两层：

* :class:`TestMySQLLiveBehavior` —— 复用 ``tests/test_stores/behavior.py`` 的跨后端行为基准套件，
  与 SQLite 内存库单测共享同一份语义契约，但跑在**真实 MySQL 服务端**上；
* :class:`TestMySQLLiveSharding` —— 替身无法证伪的服务端事实：
  ``crc32`` 分表落表位置、``ordinal`` 跨句柄同毫秒保序、``med_session`` 会话行 upsert。

运行方式（需宿主侧自备 DBAPI 驱动，如 ``pip install pymysql``）::

    docker compose -f docker-compose.integration.yml up -d mysql
    MED_MEMORY_IT=1 pytest -m integration tests/test_integration/test_mysql_live.py
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from behavior import MedHistoryBehaviorSuite
from sqlalchemy import delete, select

from med_langchain_memory.domain import MedMessage, MessageRole
from med_langchain_memory.stores import MedChatMessageHistory
from med_langchain_memory.stores.mysql_schema import (
    MESSAGE_TABLES,
    SESSION_TABLE,
    message_table,
)
from med_langchain_memory.stores.mysql_store import MySQLMedHistory

NAMESPACE: dict[str, str] = {
    "session_id": "s-mysql-live",
    "tenant_id": "hospital_a",
    "dept_id": "cardiology",
    "patient_id": "p-1024",
}

#: 固定时间戳（2023-11-14T22:13:20Z），用于构造同毫秒消息。
FIXED_MS = 1_700_000_000_000


def make_message(content: str = "chest pain", **overrides: Any) -> MedMessage:
    """构造一条属于 :data:`NAMESPACE` 的合法医疗消息。"""
    kwargs: dict[str, Any] = {**NAMESPACE, "role": MessageRole.PATIENT, "content": content}
    kwargs.update(overrides)
    return MedMessage(**kwargs)


def truncate_all(engine: Any) -> None:
    """清空全部分表与会话表（集成测试库专用，逐表 ``DELETE``）。"""
    with engine.begin() as conn:
        conn.execute(delete(SESSION_TABLE))
        for table in MESSAGE_TABLES.values():
            conn.execute(delete(table))


@pytest.mark.integration
class TestMySQLLiveBehavior(MedHistoryBehaviorSuite):
    """真实 MySQL 上的跨后端行为契约。"""

    backend_name = "mysql"

    @pytest.fixture(autouse=True)
    def _live_engine(self, mysql_live: Any) -> Iterator[None]:
        """每个用例前后清空数据表，保证用例之间互不干扰。"""
        self._engine = mysql_live
        truncate_all(mysql_live)
        yield
        truncate_all(mysql_live)

    def make_history(self, **overrides: Any) -> MedChatMessageHistory:
        """构造指向真实 MySQL 的会话历史。"""
        return MySQLMedHistory(**{**self.NAMESPACE, **overrides}, engine=self._engine)


@pytest.mark.integration
class TestMySQLLiveSharding:
    """分表路由、跨句柄保序与会话行 upsert。"""

    @pytest.fixture
    def history(self, mysql_live: Any) -> Iterator[MySQLMedHistory]:
        """指向真实 MySQL 的空会话历史。"""
        truncate_all(mysql_live)
        yield MySQLMedHistory(**NAMESPACE, engine=mysql_live)
        truncate_all(mysql_live)

    def test_messages_land_in_hash_selected_shard(
        self, history: MySQLMedHistory, mysql_live: Any
    ) -> None:
        """正向：消息只落在 ``crc32(session_id) % 16`` 选中的那张分表里。"""
        history.add_med_messages([make_message("a"), make_message("b")])
        table = message_table(history.shard)
        assert history.table_name == f"med_message_{history.shard:02d}"
        with mysql_live.connect() as conn:
            rows = (
                conn.execute(select(table).where(table.c.session_id == NAMESPACE["session_id"]))
                .mappings()
                .all()
            )
        assert len(rows) == 2
        assert [int(row["ordinal"]) for row in rows] == [1, 2]

    def test_ordinal_keeps_same_millisecond_order_across_handles(self, mysql_live: Any) -> None:
        """正向：同毫秒消息跨句柄写入后，读取顺序与写入顺序一致。"""
        first = MySQLMedHistory(**NAMESPACE, engine=mysql_live)
        second = MySQLMedHistory(**NAMESPACE, engine=mysql_live)
        first.add_med_messages([make_message("first", created_at=FIXED_MS)])
        second.add_med_messages([make_message("second", created_at=FIXED_MS)])
        assert [message.content for message in first.get_med_messages()] == ["first", "second"]
        assert second.count() == 2

    def test_session_row_is_upserted(self, history: MySQLMedHistory) -> None:
        """正向：写入后 ``med_session`` 行存在且条数与库内真值一致。"""
        history.add_med_messages([make_message()])
        meta = history.fetch_session_meta()
        assert meta is not None
        assert (meta.session_id, meta.tenant_id, meta.message_count) == (
            NAMESPACE["session_id"],
            NAMESPACE["tenant_id"],
            1,
        )

    def test_clear_removes_rows_and_session_row(
        self, history: MySQLMedHistory, mysql_live: Any
    ) -> None:
        """边界：``clear()`` 后分表与会话表都不再留有本会话数据。"""
        history.add_med_messages([make_message()])
        history.clear()
        assert history.count() == 0
        assert history.fetch_session_meta() is None
