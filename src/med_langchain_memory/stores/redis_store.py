"""Redis 单机存储适配器 :class:`RedisMedHistory`。

定位：生产环境的**热会话存储**——正在进行中的医患问诊消息读写走 Redis，
超期会话再由 lifecycle 层迁移到 MySQL / ES 冷存储。

存储结构（每个会话两个键，均以统一存储键为前缀）：

* ``med:chat:{tenant}:{dept}:{session}:messages`` —— **List**，
  按写入顺序 ``RPUSH`` protobuf 二进制消息体，天然保序、追加 O(1)；
* ``med:chat:{tenant}:{dept}:{session}:meta`` —— **Hash**，
  存会话命名空间、状态、消息条数与时间戳，条数用 ``HINCRBY`` 原子累加。

一次 ``add_med_messages`` 的全部写命令（列表追加 + 元数据更新）打包进单个
pipeline 事务提交，批量写只有一次网络往返。

注意：消息体是 protobuf 二进制，客户端**不得**开启 ``decode_responses``，
构造时会显式校验。本后端为可选依赖（``pip install med-langchain-memory[redis]``），
未安装 ``redis`` 时导入本模块会抛 ``ImportError``，:class:`StoreFactory` 中也不会出现 ``redis``。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import ClassVar, cast

from redis import Redis
from redis.exceptions import RedisError

from med_langchain_memory.domain.message import MedMessage, now_millis
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import StorageError
from med_langchain_memory.serde.base import SerializationError
from med_langchain_memory.serde.protobuf_serializer import ProtobufSerializer

from .base import MedChatMessageHistory
from .factory import StoreFactory

#: 消息列表键后缀。
MESSAGES_SUFFIX = ":messages"

#: 会话元数据哈希键后缀。
META_SUFFIX = ":meta"

#: 未显式注入客户端时使用的默认连接串。
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


@StoreFactory.register("redis")
class RedisMedHistory(MedChatMessageHistory):
    """Redis 单机会话历史，注册名 ``redis``。

    Args 中 ``client`` 与 ``url`` 二选一：显式注入客户端便于测试与连接池复用，
    只给 ``url`` 时由本类惰性建连（构造阶段不发起网络请求）。

    Example:
        >>> import fakeredis
        >>> history = RedisMedHistory(
        ...     session_id="s-1",
        ...     tenant_id="hosp-a",
        ...     dept_id="cardio",
        ...     patient_id="p-1",
        ...     client=fakeredis.FakeRedis(),
        ... )
        >>> history.get_med_messages()
        []
    """

    #: 会话级原生 TTL 由 D12 迭代接入，当前迭代只做数据结构与批量写。
    supports_ttl: ClassVar[bool] = False

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        client: Redis | None = None,
        url: str = DEFAULT_REDIS_URL,
        ttl_seconds: int | None = None,
    ) -> None:
        """初始化 Redis 会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            client: 已建好的 redis 客户端；为 ``None`` 时按 ``url`` 创建。
            url: redis 连接串，仅在 ``client`` 为 ``None`` 时生效。
            ttl_seconds: 必须为 ``None``，本迭代尚未接入原生 TTL。

        Raises:
            ValidationError: ID 不合法时。
            StorageError: 传入了 ``ttl_seconds``，或客户端开启了 ``decode_responses`` 时。
        """
        super().__init__(session_id, tenant_id, dept_id, patient_id, ttl_seconds=ttl_seconds)
        self._client: Redis = Redis.from_url(url) if client is None else client
        self._assert_binary_client()
        self._serializer = ProtobufSerializer()
        self._messages_key = f"{self.storage_key}{MESSAGES_SUFFIX}"
        self._meta_key = f"{self.storage_key}{META_SUFFIX}"

    # ------------------------------------------------------------------ #
    # 存储原语
    # ------------------------------------------------------------------ #
    def _append(self, messages: list[MedMessage]) -> None:
        """在单个 pipeline 事务内追加消息体并更新会话元数据哈希。"""
        payloads = [self._serializer.serialize_message(message) for message in messages]
        now = now_millis()
        with self._guard("append"), self._client.pipeline(transaction=True) as pipe:
            pipe.rpush(self._messages_key, *payloads)
            pipe.hsetnx(self._meta_key, "created_at", str(now))
            pipe.hset(self._meta_key, mapping=self._meta_mapping(now))
            pipe.hincrby(self._meta_key, "message_count", len(messages))
            pipe.execute()

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """读取整个列表并按 ``created_at`` 稳定排序；键不存在时返回空列表。

        Redis List 本身保序，此处仍统一排序，是为了与其余后端语义完全一致
        （允许调用方乱序补写历史消息）。

        Raises:
            StorageError: redis 命令失败或存在无法解码的消息体时。
        """
        with self._guard("read"):
            raw = cast("list[bytes]", self._client.lrange(self._messages_key, 0, -1))
        messages = [self._decode(index, blob) for index, blob in enumerate(raw)]
        messages.sort(key=lambda message: message.created_at)
        return messages if limit is None else messages[-limit:]

    def clear(self) -> None:
        """删除本会话的消息列表与元数据哈希（键不存在时为空操作）。

        Raises:
            StorageError: redis 命令失败时。
        """
        with self._guard("clear"), self._client.pipeline(transaction=True) as pipe:
            pipe.delete(self._messages_key)
            pipe.delete(self._meta_key)
            pipe.execute()

    # ------------------------------------------------------------------ #
    # Redis 后端专有能力
    # ------------------------------------------------------------------ #
    @property
    def client(self) -> Redis:
        """底层 redis 客户端。"""
        return self._client

    @property
    def messages_key(self) -> str:
        """本会话消息列表的完整键名。"""
        return self._messages_key

    @property
    def meta_key(self) -> str:
        """本会话元数据哈希的完整键名。"""
        return self._meta_key

    @property
    def size(self) -> int:
        """当前会话已存储的消息条数（``LLEN``，不解码消息体）。

        Raises:
            StorageError: redis 命令失败时。
        """
        with self._guard("llen"):
            return int(cast(int, self._client.llen(self._messages_key)))

    def fetch_session_meta(self) -> SessionMeta | None:
        """从 Redis 哈希读回会话元数据（跨实例句柄共享的权威版本）。

        Returns:
            会话元数据；本会话从未写入过任何消息时返回 ``None``。

        Raises:
            StorageError: redis 命令失败，或哈希字段缺失/非法时。
        """
        with self._guard("hgetall"):
            raw = cast("dict[bytes, bytes]", self._client.hgetall(self._meta_key))
        if not raw:
            return None
        try:
            # UnicodeDecodeError 与 pydantic ValidationError 均为 ValueError 子类。
            fields = {key.decode(): value.decode() for key, value in raw.items()}
            return SessionMeta(
                session_id=fields["session_id"],
                tenant_id=fields["tenant_id"],
                dept_id=fields["dept_id"],
                patient_id=fields["patient_id"],
                status=SessionStatus(fields["status"]),
                message_count=int(fields["message_count"]),
                created_at=int(fields["created_at"]),
                updated_at=int(fields["updated_at"]),
            )
        except (KeyError, ValueError) as exc:
            raise StorageError(f"corrupted session meta at {self._meta_key}: {exc}") from exc

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _meta_mapping(self, now_ms: int) -> dict[str, str]:
        """构造每次写入都会覆盖的元数据字段（``created_at`` 与条数除外）。"""
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "dept_id": self.dept_id,
            "patient_id": self.patient_id,
            "status": self.session_meta.status.value,
            "updated_at": str(now_ms),
        }

    def _decode(self, index: int, blob: bytes) -> MedMessage:
        """解码列表中的单条 protobuf 消息体。"""
        try:
            return self._serializer.deserialize_message(blob)
        except (SerializationError, ValueError) as exc:
            raise StorageError(
                f"corrupted message at index {index} of {self._messages_key}: {exc}"
            ) from exc

    def _assert_binary_client(self) -> None:
        """拒绝开启了 ``decode_responses`` 的客户端（会破坏 protobuf 二进制）。"""
        pool = getattr(self._client, "connection_pool", None)
        options = getattr(pool, "connection_kwargs", None)
        if isinstance(options, dict) and options.get("decode_responses"):
            raise StorageError(
                "redis client must not enable decode_responses: "
                "message payloads are protobuf binary"
            )

    @contextmanager
    def _guard(self, action: str) -> Iterator[None]:
        """把 redis 客户端异常统一包装成 :class:`StorageError`。"""
        try:
            yield
        except RedisError as exc:
            raise StorageError(f"redis {action} failed for {self.storage_key}: {exc}") from exc
