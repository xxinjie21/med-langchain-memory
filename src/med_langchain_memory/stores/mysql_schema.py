"""MySQL 存储模式定义：表结构、DDL 渲染与迁移基线。

本模块用 SQLAlchemy 2.0 Core 声明医疗会话中间件的 MySQL 物理模型，是 D14 分表路由与
``mysql_store`` 的数据契约来源：

* :data:`SESSION_TABLE` —— 会话元数据表 ``med_session``（一会话一行）；
* :data:`MESSAGE_TABLES` —— 消息分表 ``med_message_00`` ~ ``med_message_15``，共
  :data:`SHARD_COUNT` 张同构表，字段对齐 ROADMAP《对外存储规范》与 protobuf ``MedMessage``；
* :data:`SCHEMA_VERSION_TABLE` —— 迁移基线版本表 ``med_schema_version``，
  用极简的"版本号 + 应用时间"两列记录已落地的模式版本（不引入额外迁移框架依赖）。

16 张分表在此仅做**结构定义**，``session_id -> shard`` 的一致性 hash 路由属于 D14。
所有表均可用 :func:`render_ddl` 渲染为可直接执行的 SQL 脚本，或用 :func:`create_all`
在给定 ``Engine`` 上建表（测试用 SQLite 内存库亦可）。

本模块只做结构映射与字段搬运，不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    Engine,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    insert,
    select,
    text,
)
from sqlalchemy.dialects.mysql.base import MySQLDialect
from sqlalchemy.dialects.sqlite.base import SQLiteDialect
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.schema import CreateIndex, CreateTable

from med_langchain_memory.domain.message import MedMessage, MessageRole, now_millis
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import StorageError, ValidationError

#: 消息分表数量，与 ROADMAP 规范 ``med_message_{crc32(session_id) % 16}`` 一致。
SHARD_COUNT = 16

#: 消息分表名前缀。
MESSAGE_TABLE_PREFIX = "med_message"

#: 迁移基线版本号，:func:`create_all` 建表后写入 :data:`SCHEMA_VERSION_TABLE`。
BASELINE_REVISION = "0001_baseline"

#: 业务 ID 列宽，与 :data:`med_langchain_memory.domain.message.IdStr` 的 max_length 对齐。
_ID_LEN = 64

#: 统一索引/主键命名规则，保证 16 张分表的索引名互不冲突（SQLite 要求全局唯一）。
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}

#: 本库全部 MySQL 表所属的元数据容器。
METADATA = MetaData(naming_convention=NAMING_CONVENTION)

_DIALECTS: dict[str, Dialect] = {
    "mysql": MySQLDialect(),
    "sqlite": SQLiteDialect(),
}


def message_table_name(shard: int) -> str:
    """返回指定分片编号对应的消息分表名。

    Args:
        shard: 分片编号，取值范围 ``[0, SHARD_COUNT)``。

    Returns:
        形如 ``med_message_07`` 的表名（编号补零到两位）。

    Raises:
        ValidationError: 分片编号越界时。
    """
    if not 0 <= shard < SHARD_COUNT:
        raise ValidationError(f"shard must be in [0, {SHARD_COUNT}), got {shard}")
    return f"{MESSAGE_TABLE_PREFIX}_{shard:02d}"


def _build_message_table(shard: int) -> Table:
    """构建一张消息分表（结构对齐 protobuf ``MedMessage``）。"""
    return Table(
        message_table_name(shard),
        METADATA,
        Column("message_id", String(36), primary_key=True),
        Column("session_id", String(_ID_LEN), nullable=False),
        Column("tenant_id", String(_ID_LEN), nullable=False),
        Column("dept_id", String(_ID_LEN), nullable=False),
        Column("patient_id", String(_ID_LEN), nullable=False),
        Column("role", String(16), nullable=False),
        Column("content", Text, nullable=False),
        Column("token_count", Integer, nullable=False, server_default=text("0")),
        Column("masked", Boolean, nullable=False, server_default=text("0")),
        Column("created_at", BigInteger, nullable=False),
        Column("metadata", JSON, nullable=False),
        Index(None, "session_id", "created_at"),
        Index(None, "tenant_id", "dept_id"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
        mysql_collate="utf8mb4_unicode_ci",
    )


#: 会话元数据表：一次问诊会话一行，承载状态机与消息计数。
SESSION_TABLE = Table(
    "med_session",
    METADATA,
    Column("session_id", String(_ID_LEN), primary_key=True),
    Column("tenant_id", String(_ID_LEN), nullable=False),
    Column("dept_id", String(_ID_LEN), nullable=False),
    Column("patient_id", String(_ID_LEN), nullable=False),
    Column("status", String(16), nullable=False, server_default=text("'active'")),
    Column("message_count", Integer, nullable=False, server_default=text("0")),
    Column("created_at", BigInteger, nullable=False),
    Column("updated_at", BigInteger, nullable=False),
    Column("metadata", JSON, nullable=False),
    Index(None, "tenant_id", "dept_id", "status"),
    Index(None, "updated_at"),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_unicode_ci",
)

#: 迁移基线版本表：记录已应用的模式版本，避免重复初始化。
SCHEMA_VERSION_TABLE = Table(
    "med_schema_version",
    METADATA,
    Column("revision", String(32), primary_key=True),
    Column("applied_at", BigInteger, nullable=False),
    mysql_engine="InnoDB",
    mysql_charset="utf8mb4",
    mysql_collate="utf8mb4_unicode_ci",
)

#: 分片编号到消息分表的映射，进程内单例。
MESSAGE_TABLES: dict[int, Table] = {
    shard: _build_message_table(shard) for shard in range(SHARD_COUNT)
}


def message_table(shard: int) -> Table:
    """返回指定分片编号的消息分表对象。

    Args:
        shard: 分片编号，取值范围 ``[0, SHARD_COUNT)``。

    Returns:
        对应的 SQLAlchemy ``Table``。

    Raises:
        ValidationError: 分片编号越界时。
    """
    if shard not in MESSAGE_TABLES:
        raise ValidationError(f"shard must be in [0, {SHARD_COUNT}), got {shard}")
    return MESSAGE_TABLES[shard]


def all_tables() -> list[Table]:
    """返回本库全部表，按建表顺序（元数据表在前，消息分表按编号升序）。"""
    return [SCHEMA_VERSION_TABLE, SESSION_TABLE, *(MESSAGE_TABLES[i] for i in range(SHARD_COUNT))]


# ---------------------------------------------------------------------- #
# 领域模型 <-> 数据行
# ---------------------------------------------------------------------- #
def message_to_row(message: MedMessage) -> dict[str, Any]:
    """将医疗消息转换为可直接 ``insert()`` 的行字典。

    Args:
        message: 待落库的医疗消息。

    Returns:
        键与消息分表列名一一对应的字典。
    """
    return {
        "message_id": message.message_id,
        "session_id": message.session_id,
        "tenant_id": message.tenant_id,
        "dept_id": message.dept_id,
        "patient_id": message.patient_id,
        "role": message.role.value,
        "content": message.content,
        "token_count": message.token_count,
        "masked": message.masked,
        "created_at": message.created_at,
        "metadata": dict(message.metadata),
    }


def message_from_row(row: Mapping[str, Any]) -> MedMessage:
    """将消息分表的一行还原为 :class:`MedMessage`。

    Args:
        row: 查询结果行（``Row._mapping`` 或等价字典）。

    Returns:
        还原后的医疗消息。

    Raises:
        StorageError: 行缺少必需列、角色非法或字段校验失败时。
    """
    try:
        return MedMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            dept_id=row["dept_id"],
            patient_id=row["patient_id"],
            role=MessageRole(row["role"]),
            content=row["content"],
            token_count=row["token_count"],
            masked=bool(row["masked"]),
            created_at=row["created_at"],
            metadata=_as_str_map(row["metadata"]),
        )
    except (KeyError, TypeError, ValueError, PydanticValidationError) as exc:
        raise StorageError(f"corrupted message row: {exc}") from exc


def session_to_row(meta: SessionMeta) -> dict[str, Any]:
    """将会话元数据转换为可直接 ``insert()`` 的行字典。

    Args:
        meta: 会话元数据。

    Returns:
        键与 ``med_session`` 列名一一对应的字典。
    """
    return {
        "session_id": meta.session_id,
        "tenant_id": meta.tenant_id,
        "dept_id": meta.dept_id,
        "patient_id": meta.patient_id,
        "status": meta.status.value,
        "message_count": meta.message_count,
        "created_at": meta.created_at,
        "updated_at": meta.updated_at,
        "metadata": dict(meta.metadata),
    }


def session_from_row(row: Mapping[str, Any]) -> SessionMeta:
    """将 ``med_session`` 的一行还原为 :class:`SessionMeta`。

    Args:
        row: 查询结果行（``Row._mapping`` 或等价字典）。

    Returns:
        还原后的会话元数据。

    Raises:
        StorageError: 行缺少必需列、状态非法或字段校验失败时。
    """
    try:
        return SessionMeta(
            session_id=row["session_id"],
            tenant_id=row["tenant_id"],
            dept_id=row["dept_id"],
            patient_id=row["patient_id"],
            status=SessionStatus(row["status"]),
            message_count=row["message_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=_as_str_map(row["metadata"]),
        )
    except (KeyError, TypeError, ValueError, PydanticValidationError) as exc:
        raise StorageError(f"corrupted session row: {exc}") from exc


def _as_str_map(value: object) -> dict[str, str]:
    """将 JSON 列的值归一化为 ``dict[str, str]``（``None`` 视为空表）。"""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(f"metadata must be a mapping, got {type(value).__name__}")
    return {str(k): str(v) for k, v in value.items()}


# ---------------------------------------------------------------------- #
# DDL 渲染与基线初始化
# ---------------------------------------------------------------------- #
def render_ddl(dialect: str = "mysql") -> str:
    """将全部表渲染为可执行的 ``CREATE TABLE`` / ``CREATE INDEX`` 脚本。

    Args:
        dialect: 目标方言，支持 ``"mysql"``（默认）与 ``"sqlite"``。

    Returns:
        以分号分隔、可直接导入数据库的 SQL 文本。

    Raises:
        ValidationError: 方言不受支持时。
    """
    engine_dialect = _DIALECTS.get(dialect)
    if engine_dialect is None:
        supported = ", ".join(sorted(_DIALECTS))
        raise ValidationError(f"unsupported dialect {dialect!r}, expected one of: {supported}")

    statements: list[str] = []
    for table in all_tables():
        statements.append(str(CreateTable(table).compile(dialect=engine_dialect)).strip() + ";")
        for index in sorted(table.indexes, key=lambda idx: idx.name or ""):
            statements.append(str(CreateIndex(index).compile(dialect=engine_dialect)).strip() + ";")
    return "\n\n".join(statements) + "\n"


def create_all(engine: Engine) -> str:
    """在给定引擎上创建全部表并登记迁移基线版本。

    重复调用是幂等的：已存在的表不会重建，基线版本也只登记一次。

    Args:
        engine: 目标数据库引擎（生产为 MySQL，测试可用 SQLite 内存库）。

    Returns:
        本次生效的基线版本号 :data:`BASELINE_REVISION`。

    Raises:
        StorageError: 建表或写入版本记录失败时。
    """
    try:
        METADATA.create_all(engine)
        with engine.begin() as conn:
            stmt = select(SCHEMA_VERSION_TABLE.c.revision).where(
                SCHEMA_VERSION_TABLE.c.revision == BASELINE_REVISION
            )
            if conn.execute(stmt).first() is None:
                conn.execute(
                    insert(SCHEMA_VERSION_TABLE).values(
                        revision=BASELINE_REVISION, applied_at=now_millis()
                    )
                )
    except Exception as exc:  # pragma: no cover - 依赖具体驱动错误
        raise StorageError(f"failed to initialize mysql schema: {exc}") from exc
    return BASELINE_REVISION


def current_revision(engine: Engine) -> str | None:
    """查询数据库当前已应用的最新模式版本。

    Args:
        engine: 目标数据库引擎。

    Returns:
        最近一次应用的版本号；模式尚未初始化时返回 ``None``。
    """
    stmt = select(SCHEMA_VERSION_TABLE.c.revision).order_by(
        SCHEMA_VERSION_TABLE.c.applied_at.desc(), SCHEMA_VERSION_TABLE.c.revision.desc()
    )
    with engine.connect() as conn:
        if not engine.dialect.has_table(conn, SCHEMA_VERSION_TABLE.name):
            return None
        row = conn.execute(stmt).first()
    return None if row is None else str(row[0])
