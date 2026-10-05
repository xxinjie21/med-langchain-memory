"""MySQL 存储适配器 :class:`MySQLMedHistory`。

定位：**持久化主存储**——问诊会话的消息与会话元数据落到 MySQL，
消息按 ``crc32(session_id) % 16`` 一致性 hash 路由到 16 张同构分表
（表结构见 :mod:`med_langchain_memory.stores.mysql_schema`）。

写入语义：

* 一次 ``add_med_messages`` 的同一会话消息按序写入**同一张分表**
  （``executemany`` 单次往返），并同步 upsert ``med_session`` 会话行
  （状态、消息条数、最后活跃时间），供租户/科室维度的会话列表与 TTL 扫描；
* 每条消息写入前由存储层分配**会话内单调递增的 ``ordinal``**，
  读取按 ``(created_at, ordinal)`` 排序，保证「同毫秒消息保持写入顺序」——
  ``message_id`` 为 UUIDv7，同毫秒内随机，不可单独用作 tiebreaker；
  排序语义与内存 / 文件 / Redis / ES 后端完全一致；
* ``archive()`` / ``delete()`` 除完成状态流转外，会把新状态回写 ``med_session``。

会话级 TTL 由 lifecycle 层调度（本后端 ``supports_ttl = False``），
存储层提供基类的 ``is_expired`` 逻辑判定与 ``med_session.updated_at`` 索引供 TTL 扫描。

同一会话的并发写入由上层会话锁（``runnable/lock.py``）串行化——``ordinal`` 的
「读最大值 + 批量插入」不是原子操作，跨进程并发写同一会话时需由锁保证互斥。

本后端为可选依赖（``pip install med-langchain-memory[mysql]``），未安装 ``SQLAlchemy``
时导入本模块会抛 ``ImportError``，:class:`StoreFactory` 中也不会出现 ``mysql``。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar

from sqlalchemy import (
    Connection,
    Engine,
    create_engine,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.exc import SQLAlchemyError

from med_langchain_memory.domain.message import MedMessage, now_millis
from med_langchain_memory.domain.session import SessionMeta
from med_langchain_memory.exceptions import StorageError

from .base import MedChatMessageHistory
from .factory import StoreFactory
from .mysql_schema import (
    SESSION_TABLE,
    create_all,
    message_from_row,
    message_to_row,
    session_from_row,
    session_to_row,
)
from .mysql_shard_router import DEFAULT_ROUTER, ShardRouter

#: 未显式注入引擎时使用的默认连接串。
#: 注意需自行安装 DBAPI 驱动（如 ``pymysql``），引擎在首次访问存储时才惰性创建。
DEFAULT_MYSQL_URL = "mysql+pymysql://root:@localhost:3306/med_memory"


@StoreFactory.register("mysql")
class MySQLMedHistory(MedChatMessageHistory):
    """MySQL 分表会话历史，注册名 ``mysql``。

    Args 中 ``engine`` 与 ``url`` 二选一：显式注入引擎便于测试与连接池复用；
    只给 ``url`` 时由本类在**首次访问存储时**惰性建连（构造阶段不导入驱动）。

    Example:
        >>> from sqlalchemy import create_engine
        >>> from med_langchain_memory.stores.mysql_schema import create_all
        >>> engine = create_engine("sqlite+pysqlite:///:memory:")
        >>> create_all(engine)
        '0001_baseline'
        >>> history = MySQLMedHistory(
        ...     session_id="s-1",
        ...     tenant_id="hosp-a",
        ...     dept_id="cardio",
        ...     patient_id="p-1",
        ...     engine=engine,
        ... )
        >>> history.get_med_messages()
        []
    """

    #: MySQL 无原生过期能力，TTL 由上层 lifecycle 调度器负责。
    supports_ttl: ClassVar[bool] = False

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        engine: Engine | None = None,
        url: str = DEFAULT_MYSQL_URL,
        router: ShardRouter | None = None,
        ensure_schema: bool = False,
        ttl_seconds: int | None = None,
    ) -> None:
        """初始化 MySQL 会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            engine: 已建好的 SQLAlchemy 引擎；为 ``None`` 时按 ``url`` 惰性创建。
            url: 连接串，仅在 ``engine`` 为 ``None`` 时生效。
            router: 分表路由器，缺省复用进程内 16 张分表的
                :data:`~med_langchain_memory.stores.mysql_shard_router.DEFAULT_ROUTER`。
            ensure_schema: 构造时是否幂等建表（生产建议由 DDL/迁移管理，测试可置 ``True``）。
            ttl_seconds: 必须为 ``None``，本后端不支持原生 TTL。

        Raises:
            ValidationError: ID 不合法或 ``ttl_seconds`` 非正数时。
            StorageError: 传入了 ``ttl_seconds``（本后端不支持原生 TTL），
                或 ``ensure_schema=True`` 但建表失败时。
        """
        super().__init__(session_id, tenant_id, dept_id, patient_id, ttl_seconds=ttl_seconds)
        self._engine = engine
        self._url = url
        self._router = DEFAULT_ROUTER if router is None else router
        self._table = self._router.table_of(self.session_id)
        if ensure_schema:
            self.ensure_schema()

    # ------------------------------------------------------------------ #
    # 存储原语
    # ------------------------------------------------------------------ #
    def _append(self, messages: list[MedMessage]) -> None:
        """把消息按序写入本会话所属分表，并同步 upsert ``med_session`` 会话行。

        ``ordinal`` 从该会话在当前分表内的最大序号续接，因此跨多次 ``add_med_messages``
        仍然单调递增，读取时的排序结果与写入顺序一致。
        """
        table = self._table
        with self._guard("append"), self.engine.begin() as conn:
            base = self._max_ordinal(conn)
            rows = [
                message_to_row(message, ordinal=base + offset)
                for offset, message in enumerate(messages, start=1)
            ]
            conn.execute(insert(table), rows)
            self._write_session_row(conn, self._meta, self._count_in(conn))

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """按时序读取本会话消息；``limit`` 表示最近 N 条。

        Raises:
            StorageError: 查询失败（表不存在、连接中断等）时。
        """
        table = self._table
        statement = (
            select(table)
            .where(table.c.session_id == self.session_id)
            .order_by(table.c.created_at, table.c.ordinal)
        )
        with self._guard("read"), self.engine.connect() as conn:
            rows = conn.execute(statement).mappings().all()
        messages = [message_from_row(dict(row)) for row in rows]
        return messages if limit is None else messages[-limit:]

    def clear(self) -> None:
        """删除本会话的全部分表消息与会话元数据行（本会话不存在时为空操作）。

        Raises:
            StorageError: 删除失败时。
        """
        table = self._table
        with self._guard("clear"), self.engine.begin() as conn:
            conn.execute(delete(table).where(table.c.session_id == self.session_id))
            conn.execute(delete(SESSION_TABLE).where(SESSION_TABLE.c.session_id == self.session_id))

    # ------------------------------------------------------------------ #
    # 归档 / 软删除：状态流转后回写 med_session
    # ------------------------------------------------------------------ #
    def archive(self) -> tuple[SessionMeta, list[MedMessage]]:
        """流转至 ``ARCHIVED`` 并把新状态回写 ``med_session``。"""
        meta, messages = super().archive()
        self._persist_status(meta)
        return meta, messages

    def delete(self) -> tuple[SessionMeta, list[MedMessage]]:
        """流转至 ``DELETED``（软删除）并把新状态回写 ``med_session``。"""
        meta, messages = super().delete()
        self._persist_status(meta)
        return meta, messages

    # ------------------------------------------------------------------ #
    # MySQL 后端专有能力
    # ------------------------------------------------------------------ #
    @property
    def engine(self) -> Engine:
        """底层 SQLAlchemy 引擎（``url`` 模式下首次访问时惰性创建）。

        Raises:
            StorageError: 连接串方言不可用或 DBAPI 驱动缺失时。
        """
        if self._engine is None:
            with self._guard("engine"):
                engine = create_engine(self._url)
            self._engine = engine
        return self._engine

    @property
    def router(self) -> ShardRouter:
        """本实例使用的分表路由器。"""
        return self._router

    @property
    def shard(self) -> int:
        """本会话所属分片编号（``crc32(session_id) % shard_count``）。"""
        return self._router.shard_of(self.session_id)

    @property
    def table_name(self) -> str:
        """本会话消息落库的分表名（如 ``med_message_07``）。"""
        return self._table.name

    def count(self) -> int:
        """统计本会话在当前分表中的消息条数（``COUNT(*)``，不解码消息体）。

        Raises:
            StorageError: 查询失败时。
        """
        with self._guard("count"), self.engine.connect() as conn:
            return self._count_in(conn)

    def fetch_session_meta(self) -> SessionMeta | None:
        """从 ``med_session`` 读回会话元数据（跨实例句柄共享的权威版本）。

        Returns:
            会话元数据；本会话从未写入过消息时返回 ``None``。

        Raises:
            StorageError: 查询失败，或会话行字段缺失/非法时。
        """
        statement = select(SESSION_TABLE).where(SESSION_TABLE.c.session_id == self.session_id)
        with self._guard("fetch meta"), self.engine.connect() as conn:
            row = conn.execute(statement).mappings().first()
        return None if row is None else session_from_row(dict(row))

    def ensure_schema(self) -> str:
        """在底层引擎上幂等创建全部表并登记迁移基线版本。

        Returns:
            本次生效的基线版本号。

        Raises:
            StorageError: 建表失败时。
        """
        with self._guard("ensure schema"):
            return create_all(self.engine)

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    @contextmanager
    def _guard(self, operation: str) -> Iterator[None]:
        """把 SQLAlchemy 异常统一包装为 :class:`StorageError`。"""
        try:
            yield
        except SQLAlchemyError as exc:
            raise StorageError(f"mysql {operation} failed: {exc}") from exc

    def _max_ordinal(self, conn: Connection) -> int:
        """读取本会话在当前分表内的最大 ``ordinal``；无数据时返回 ``0``。"""
        statement = select(func.max(self._table.c.ordinal)).where(
            self._table.c.session_id == self.session_id
        )
        value = conn.execute(statement).scalar()
        return 0 if value is None else int(value)

    def _count_in(self, conn: Connection) -> int:
        """在给定连接上统计本会话的消息条数。"""
        statement = (
            select(func.count())
            .select_from(self._table)
            .where(self._table.c.session_id == self.session_id)
        )
        return int(conn.execute(statement).scalar_one())

    def _write_session_row(self, conn: Connection, meta: SessionMeta, count: int) -> None:
        """按「存在则更新、不存在则插入」写 ``med_session`` 行（需在事务连接内调用）。"""
        row = session_to_row(meta)
        row["message_count"] = count
        row["updated_at"] = max(now_millis(), row["created_at"])
        exists = conn.execute(
            select(SESSION_TABLE.c.session_id).where(SESSION_TABLE.c.session_id == self.session_id)
        ).first()
        if exists is None:
            conn.execute(insert(SESSION_TABLE).values(row))
        else:
            conn.execute(
                update(SESSION_TABLE)
                .where(SESSION_TABLE.c.session_id == self.session_id)
                .values(row)
            )

    def _persist_status(self, meta: SessionMeta) -> None:
        """把状态流转后的会话元数据回写 ``med_session``（消息条数取库内真值）。

        Raises:
            StorageError: 回写失败时。
        """
        with self._guard("persist status"), self.engine.begin() as conn:
            self._write_session_row(conn, meta, self._count_in(conn))
