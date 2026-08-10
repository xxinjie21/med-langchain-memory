"""Elasticsearch 归档存储适配器 :class:`EsArchiveMedHistory`。

定位：**冷归档层**——问诊结束或热存储（Redis/MySQL）会话超期后，
消息整体沉降到 ES，供后续按租户/科室/患者/时间范围检索与统计。

索引规划（按月滚动）：

* 索引名 ``med-chat-archive-{yyyy.MM}``，按**消息创建时间（UTC）**路由，
  同一次 bulk 中跨月消息会自动落到各自月份索引；
* 检索一律走通配符 ``med-chat-archive-*``，无需调用方感知月份切分；
* 索引模板 :func:`build_index_template` 统一定义 settings 与 mappings，
  由 :meth:`EsArchiveMedHistory.ensure_index_template` 幂等注册，新月份索引自动继承。

文档结构遵循「**可检索字段 + protobuf 权威载荷**」双写：

* 命名空间、角色、时间戳等结构化字段以 ``keyword`` / ``long`` 建索引，供归档查询；
* ``payload`` 为 protobuf 二进制的 base64（ES ``binary`` 类型，不索引），
  读取时优先由它无损还原 :class:`MedMessage`，保证与其余后端的跨语言字节级一致。

``content`` 字段显式 ``index: false``——归档层只做存储与元数据检索，
不启用任何分析器/分词器，不做文本内容理解（项目硬约束）。

写入使用 ``_bulk`` 批量接口并按 :data:`DEFAULT_CHUNK_SIZE` 分块，
文档 ``_id`` 取 ``message_id``，因此重放同一批归档数据是幂等的（覆盖写）。

本后端为可选依赖（``pip install med-langchain-memory[es]``），
未安装 ``elasticsearch`` 时导入本模块会抛 ``ImportError``，
:class:`StoreFactory` 中也不会出现 ``elasticsearch``。
"""

from __future__ import annotations

import base64
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, ClassVar, cast

from elasticsearch import ApiError, Elasticsearch, TransportError

from med_langchain_memory.domain.message import MedMessage, now_millis
from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.serde.base import SerializationError
from med_langchain_memory.serde.protobuf_serializer import ProtobufSerializer

from .base import MedChatMessageHistory
from .factory import StoreFactory

#: 归档索引名前缀，实际索引为 ``{prefix}-{yyyy.MM}``。
ARCHIVE_INDEX_PREFIX = "med-chat-archive"

#: 索引模板名称。
ARCHIVE_TEMPLATE_NAME = "med-chat-archive"

#: 未显式注入客户端时使用的默认连接地址。
DEFAULT_ES_URL = "http://localhost:9200"

#: 单次 bulk 请求携带的最大文档数。
DEFAULT_CHUNK_SIZE = 500

#: 单次 search 允许返回的最大命中数（ES ``index.max_result_window`` 默认值）。
MAX_SEARCH_SIZE = 10_000

#: 归档检索的排序键：创建时间为主，同毫秒时回落到归档时间与批内序号，
#: 保证与热存储完全一致的写入顺序语义（``message_id`` 同毫秒内随机，不可用作 tiebreaker）。
SORT_FIELDS = ("created_at", "archived_at", "ordinal")

_SERIALIZER = ProtobufSerializer()


def monthly_index(created_at_ms: int, prefix: str = ARCHIVE_INDEX_PREFIX) -> str:
    """按消息创建时间计算所属的按月滚动归档索引名。

    Args:
        created_at_ms: 消息创建时间（epoch 毫秒）。
        prefix: 索引前缀。

    Returns:
        形如 ``med-chat-archive-2026.08`` 的索引名（月份按 UTC 计算）。

    Raises:
        ValidationError: ``created_at_ms`` 非正数时。
    """
    if created_at_ms <= 0:
        raise ValidationError(f"created_at must be a positive epoch millis, got {created_at_ms}")
    moment = datetime.fromtimestamp(created_at_ms / 1000, tz=UTC)
    return f"{prefix}-{moment:%Y.%m}"


def index_pattern(prefix: str = ARCHIVE_INDEX_PREFIX) -> str:
    """返回覆盖全部月份归档索引的通配符检索模式。"""
    return f"{prefix}-*"


def build_index_template(
    prefix: str = ARCHIVE_INDEX_PREFIX,
    *,
    shards: int = 1,
    replicas: int = 1,
) -> dict[str, Any]:
    """构造归档索引模板请求体（``PUT _index_template`` 的关键字参数）。

    Args:
        prefix: 索引前缀，决定模板匹配的 ``index_patterns``。
        shards: 主分片数。
        replicas: 副本数。

    Returns:
        含 ``index_patterns`` / ``priority`` / ``template`` / ``meta`` 的字典，
        可直接以 ``**`` 展开传给 ``indices.put_index_template``。
    """
    return {
        "index_patterns": [index_pattern(prefix)],
        "priority": 200,
        "template": {
            "settings": {
                "number_of_shards": shards,
                "number_of_replicas": replicas,
                "refresh_interval": "5s",
                "codec": "best_compression",
            },
            "mappings": {
                "dynamic": "strict",
                "properties": {
                    "message_id": {"type": "keyword"},
                    "session_id": {"type": "keyword"},
                    "tenant_id": {"type": "keyword"},
                    "dept_id": {"type": "keyword"},
                    "patient_id": {"type": "keyword"},
                    "storage_key": {"type": "keyword"},
                    "role": {"type": "keyword"},
                    # 只存不索引：归档层不做任何文本分析。
                    "content": {"type": "text", "index": False},
                    "token_count": {"type": "integer"},
                    "masked": {"type": "boolean"},
                    "created_at": {"type": "long"},
                    "archived_at": {"type": "long"},
                    "ordinal": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": False},
                    "payload": {"type": "binary"},
                },
            },
        },
        "meta": {"owner": "med-langchain-memory", "revision": 1},
    }


def to_document(
    message: MedMessage, *, archived_at: int | None = None, ordinal: int = 0
) -> dict[str, Any]:
    """将医疗消息转换为归档文档（检索字段 + protobuf 载荷）。

    Args:
        message: 待归档消息。
        archived_at: 归档时间（epoch 毫秒），缺省取当前时间。
        ordinal: 同一次归档写入内的位置序号，用于 ``created_at`` 相同（同毫秒）时
            的稳定排序——``message_id`` 是 UUIDv7，同毫秒内随机位不保证写入顺序。

    Returns:
        可直接写入 ES 的文档字典。
    """
    payload = _SERIALIZER.serialize_message(message)
    return {
        "message_id": message.message_id,
        "session_id": message.session_id,
        "tenant_id": message.tenant_id,
        "dept_id": message.dept_id,
        "patient_id": message.patient_id,
        "storage_key": message.storage_key,
        "role": message.role.value,
        "content": message.content,
        "token_count": message.token_count,
        "masked": message.masked,
        "created_at": message.created_at,
        "archived_at": now_millis() if archived_at is None else archived_at,
        "ordinal": ordinal,
        "metadata": dict(message.metadata),
        "payload": base64.b64encode(payload).decode("ascii"),
    }


def from_document(source: Mapping[str, Any]) -> MedMessage:
    """将归档文档还原为医疗消息。

    优先解码 ``payload`` 中的 protobuf 权威载荷；缺失时按检索字段重建。

    Args:
        source: ES 命中文档的 ``_source``。

    Returns:
        还原后的医疗消息。

    Raises:
        StorageError: 载荷损坏或字段缺失/非法，无法还原时。
    """
    payload = source.get("payload")
    if isinstance(payload, str) and payload:
        try:
            return _SERIALIZER.deserialize_message(base64.b64decode(payload, validate=True))
        except (SerializationError, ValueError) as exc:
            raise StorageError(f"corrupted archived payload: {exc}") from exc
    try:
        return MedMessage(
            message_id=source["message_id"],
            session_id=source["session_id"],
            tenant_id=source["tenant_id"],
            dept_id=source["dept_id"],
            patient_id=source["patient_id"],
            role=source["role"],
            content=source["content"],
            token_count=source.get("token_count", 0),
            masked=source.get("masked", False),
            created_at=source["created_at"],
            metadata=dict(source.get("metadata", {})),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError(f"corrupted archived document: {exc}") from exc


def search_archive(
    client: Elasticsearch,
    *,
    tenant_id: str,
    dept_id: str | None = None,
    patient_id: str | None = None,
    session_id: str | None = None,
    start_ms: int | None = None,
    end_ms: int | None = None,
    limit: int = 100,
    prefix: str = ARCHIVE_INDEX_PREFIX,
) -> list[MedMessage]:
    """跨会话检索归档消息（按租户强制隔离，其余条件可选叠加）。

    Args:
        client: elasticsearch 客户端。
        tenant_id: 必填的租户 ID，保证不同医院数据不互相可见。
        dept_id: 科室 ID 过滤。
        patient_id: 患者 ID 过滤。
        session_id: 会话 ID 过滤。
        start_ms: 创建时间下界（含）。
        end_ms: 创建时间上界（含）。
        limit: 最多返回条数，取值范围 ``1..10000``。
        prefix: 归档索引前缀。

    Returns:
        按 ``created_at`` 升序排列的消息列表。

    Raises:
        ValidationError: ``tenant_id`` 为空、``limit`` 越界或时间区间反转时。
        StorageError: ES 查询失败或命中文档损坏时。
    """
    if not tenant_id:
        raise ValidationError("tenant_id is required for archive search")
    if not 1 <= limit <= MAX_SEARCH_SIZE:
        raise ValidationError(f"limit must be within 1..{MAX_SEARCH_SIZE}, got {limit}")
    if start_ms is not None and end_ms is not None and start_ms > end_ms:
        raise ValidationError(f"invalid time range: {start_ms} > {end_ms}")

    filters: list[dict[str, Any]] = [{"term": {"tenant_id": tenant_id}}]
    optional = {"dept_id": dept_id, "patient_id": patient_id, "session_id": session_id}
    filters.extend({"term": {field: value}} for field, value in optional.items() if value)
    bounds: dict[str, int] = {}
    if start_ms is not None:
        bounds["gte"] = start_ms
    if end_ms is not None:
        bounds["lte"] = end_ms
    if bounds:
        filters.append({"range": {"created_at": bounds}})

    hits = _run_search(
        client,
        index=index_pattern(prefix),
        query={"bool": {"filter": filters}},
        size=limit,
        order="asc",
        target=f"archive:{tenant_id}",
    )
    return [from_document(hit) for hit in hits]


@StoreFactory.register("elasticsearch")
class EsArchiveMedHistory(MedChatMessageHistory):
    """Elasticsearch 归档会话历史，注册名 ``elasticsearch``。

    与热存储后端语义一致（同样满足跨后端行为契约），差异在于：
    写入走 bulk 批量接口、按月滚动索引，且不支持原生会话 TTL
    （归档层的留存期由 lifecycle 层的保留策略负责，而非键过期）。

    Example:
        >>> from unittest.mock import MagicMock
        >>> history = EsArchiveMedHistory(
        ...     session_id="s-1",
        ...     tenant_id="hosp-a",
        ...     dept_id="cardio",
        ...     patient_id="p-1",
        ...     client=MagicMock(),
        ... )
        >>> history.index_pattern
        'med-chat-archive-*'
    """

    #: 归档层不使用键级 TTL，留存期由合规保留策略控制。
    supports_ttl: ClassVar[bool] = False

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        client: Elasticsearch | None = None,
        hosts: str = DEFAULT_ES_URL,
        prefix: str = ARCHIVE_INDEX_PREFIX,
        refresh: bool = True,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        ttl_seconds: int | None = None,
    ) -> None:
        """初始化 ES 归档会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            client: 已建好的 elasticsearch 客户端；为 ``None`` 时按 ``hosts`` 创建。
            hosts: ES 地址，仅在 ``client`` 为 ``None`` 时生效。
            prefix: 归档索引前缀。
            refresh: 写入/删除后是否立即刷新，保证读己写一致（归档流量低，默认开启）。
            chunk_size: 单次 bulk 携带的最大文档数。
            ttl_seconds: 必须为 ``None``，归档层不支持键级 TTL。

        Raises:
            ValidationError: ID 不合法、``prefix`` 非法或 ``chunk_size`` 非正数时。
            StorageError: 传入了 ``ttl_seconds``（本后端不支持原生 TTL）时。
        """
        super().__init__(session_id, tenant_id, dept_id, patient_id, ttl_seconds=ttl_seconds)
        if not prefix or any(ch in prefix for ch in ' *?,"<>|') or prefix != prefix.lower():
            raise ValidationError(f"invalid archive index prefix: {prefix!r}")
        if chunk_size <= 0:
            raise ValidationError(f"chunk_size must be a positive integer, got {chunk_size}")
        self._client: Elasticsearch = Elasticsearch(hosts) if client is None else client
        self._prefix = prefix
        self._refresh = refresh
        self._chunk_size = chunk_size

    # ------------------------------------------------------------------ #
    # 存储原语
    # ------------------------------------------------------------------ #
    def _append(self, messages: list[MedMessage]) -> None:
        """按月份索引分发并 bulk 写入；``_id`` 取 ``message_id``，重放幂等。

        Raises:
            StorageError: ES 请求失败或返回逐条错误时。
        """
        archived_at = now_millis()
        for offset, chunk in _chunks(messages, self._chunk_size):
            operations: list[dict[str, Any]] = []
            for position, message in enumerate(chunk, start=offset):
                operations.append(
                    {
                        "index": {
                            "_index": monthly_index(message.created_at, self._prefix),
                            "_id": message.message_id,
                        }
                    }
                )
                operations.append(to_document(message, archived_at=archived_at, ordinal=position))
            with self._guard("bulk"):
                response = self._client.bulk(operations=operations, refresh=self._refresh)
            self._raise_on_bulk_errors(response)

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """检索本会话全部归档消息，按 ``created_at`` 升序返回。

        ``limit`` 语义为"最近 N 条"：查询侧倒序取 N 条后再翻转，避免全量拉取。

        Raises:
            StorageError: ES 查询失败或命中文档损坏时。
        """
        order = "asc" if limit is None else "desc"
        size = MAX_SEARCH_SIZE if limit is None else min(limit, MAX_SEARCH_SIZE)
        hits = _run_search(
            self._client,
            index=self.index_pattern,
            query=self._session_query(),
            size=size,
            order=order,
            target=self.storage_key,
        )
        if order == "desc":
            hits.reverse()
        return [from_document(hit) for hit in hits]

    def clear(self) -> None:
        """按会话条件删除全部归档文档（无匹配时为空操作）。

        Raises:
            StorageError: ES 请求失败时。
        """
        with self._guard("delete_by_query"):
            self._client.delete_by_query(
                index=self.index_pattern,
                query=self._session_query(),
                refresh=self._refresh,
                conflicts="proceed",
                ignore_unavailable=True,
            )

    # ------------------------------------------------------------------ #
    # 归档层专有能力
    # ------------------------------------------------------------------ #
    def ensure_index_template(self, *, force: bool = False) -> bool:
        """幂等注册按月滚动归档索引模板。

        Args:
            force: 为 ``True`` 时跳过存在性检查，强制覆盖已有模板。

        Returns:
            本次是否实际写入了模板（已存在且未强制覆盖时为 ``False``）。

        Raises:
            StorageError: ES 请求失败时。
        """
        with self._guard("index_template"):
            if not force and bool(
                self._client.indices.exists_index_template(name=ARCHIVE_TEMPLATE_NAME)
            ):
                return False
            self._client.indices.put_index_template(
                name=ARCHIVE_TEMPLATE_NAME,
                **build_index_template(self._prefix),
            )
        return True

    def archive_from(self, source: MedChatMessageHistory) -> int:
        """把热存储会话整体沉降到归档层：导出消息 + 状态流转 + 批量写入。

        源会话数据不会被清理，是否清理由调用方（lifecycle 归档调度）决定。

        Args:
            source: 同一命名空间下的热存储会话历史。

        Returns:
            实际归档的消息条数。

        Raises:
            StorageError: 源会话与本实例命名空间不一致，或写入失败时。
            StateTransitionError: 源会话当前状态不允许归档时。
        """
        if source.storage_key != self.storage_key:
            raise StorageError(
                f"cannot archive {source.storage_key} into {self.storage_key}: namespace mismatch"
            )
        _meta, messages = source.archive()
        self.add_med_messages(messages)
        return len(messages)

    def count(self) -> int:
        """统计本会话已归档的消息条数。

        Raises:
            StorageError: ES 请求失败时。
        """
        with self._guard("count"):
            response = self._client.count(
                index=self.index_pattern,
                query=self._session_query(),
                ignore_unavailable=True,
            )
        return int(cast("Mapping[str, Any]", response).get("count", 0))

    @property
    def client(self) -> Elasticsearch:
        """底层 elasticsearch 客户端。"""
        return self._client

    @property
    def index_pattern(self) -> str:
        """本实例检索归档数据使用的索引通配符。"""
        return index_pattern(self._prefix)

    @property
    def prefix(self) -> str:
        """归档索引前缀。"""
        return self._prefix

    @property
    def chunk_size(self) -> int:
        """单次 bulk 携带的最大文档数。"""
        return self._chunk_size

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _session_query(self) -> dict[str, Any]:
        """构造本会话（含租户/科室隔离）的过滤查询。"""
        return {
            "bool": {
                "filter": [
                    {"term": {"tenant_id": self.tenant_id}},
                    {"term": {"dept_id": self.dept_id}},
                    {"term": {"session_id": self.session_id}},
                ]
            }
        }

    def _raise_on_bulk_errors(self, response: object) -> None:
        """检查 bulk 响应的逐条结果，存在失败项时抛出 :class:`StorageError`。"""
        body = cast("Mapping[str, Any]", response)
        if not body.get("errors"):
            return
        reasons: list[str] = []
        for item in body.get("items", []):
            # bulk 响应形如 {"index": {"_id": ..., "error": {...}}}，取唯一的动作结果。
            result: Mapping[str, Any] = next(iter(cast("Mapping[str, Any]", item).values()), {})
            error = result.get("error")
            if error:
                reasons.append(str(error))
        raise StorageError(f"bulk archive failed for {self.storage_key}: {'; '.join(reasons)}")

    @contextmanager
    def _guard(self, action: str) -> Iterator[None]:
        """把 elasticsearch 客户端异常统一包装成 :class:`StorageError`。"""
        with _guard(action, self.storage_key):
            yield


def _chunks(messages: Sequence[MedMessage], size: int) -> Iterator[tuple[int, list[MedMessage]]]:
    """按固定大小切分消息序列，同时给出每块在原序列中的起始下标。"""
    for start in range(0, len(messages), size):
        yield start, list(messages[start : start + size])


@contextmanager
def _guard(action: str, target: str) -> Iterator[None]:
    """把 elasticsearch 客户端异常统一包装成 :class:`StorageError`。"""
    try:
        yield
    except (ApiError, TransportError) as exc:
        raise StorageError(f"elasticsearch {action} failed for {target}: {exc}") from exc


def _run_search(
    client: Elasticsearch,
    *,
    index: str,
    query: dict[str, Any],
    size: int,
    order: str,
    target: str,
) -> list[dict[str, Any]]:
    """执行一次归档检索并返回命中文档的 ``_source`` 列表。"""
    with _guard("search", target):
        response = client.search(
            index=index,
            query=query,
            size=size,
            sort=[{field: {"order": order}} for field in SORT_FIELDS],
            ignore_unavailable=True,
        )
    hits = cast("Mapping[str, Any]", response).get("hits", {}).get("hits", [])
    return [cast("dict[str, Any]", hit.get("_source", {})) for hit in hits]
