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

会话级 TTL 走 Redis 原生 ``EXPIRE``（两条键在同一 pipeline 事务内一起续期）：

* **滑动续期**：每次写入后由基类 ``refresh_ttl()`` 重新计时；
  构造时传 ``renew_on_read=True`` 可让读取也参与续期（活跃会话不会因只读而过期）；
* **取消过期**：``set_ttl(None)`` 下发 ``PERSIST``，键恢复为永不过期；
* **过期回调**：键过期由服务端完成、客户端无法被动感知，
  故提供 :meth:`RedisMedHistory.on_expired` 注册回调 +
  :meth:`RedisMedHistory.check_expiry` 主动探测，供 lifecycle 层归档调度轮询。

注意：消息体是 protobuf 二进制，客户端**不得**开启 ``decode_responses``，
构造时会显式校验。本后端为可选依赖（``pip install med-langchain-memory[redis]``），
未安装 ``redis`` 时导入本模块会抛 ``ImportError``，:class:`StoreFactory` 中也不会出现 ``redis``。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
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

#: 会话过期回调签名：入参为探测到过期的历史实例。
ExpiryCallback = Callable[["RedisMedHistory"], None]


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

    #: 依托 Redis 原生 ``EXPIRE`` 支持会话级 TTL。
    supports_ttl: ClassVar[bool] = True

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
        renew_on_read: bool = False,
    ) -> None:
        """初始化 Redis 会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            client: 已建好的 redis 客户端；为 ``None`` 时按 ``url`` 创建。
            url: redis 连接串，仅在 ``client`` 为 ``None`` 时生效。
            ttl_seconds: 会话级 TTL（秒）；``None`` 表示不设过期。
                非 ``None`` 时构造阶段即下发一次 ``EXPIRE``（键不存在则为空操作）。
            renew_on_read: 读取是否也参与滑动续期；默认仅写入续期。

        Raises:
            ValidationError: ID 不合法或 ``ttl_seconds`` 非正数时。
            StorageError: 客户端开启了 ``decode_responses``，或下发 TTL 失败时。
        """
        super().__init__(session_id, tenant_id, dept_id, patient_id)
        self._client: Redis = Redis.from_url(url) if client is None else client
        self._assert_binary_client()
        self._serializer = ProtobufSerializer()
        self._messages_key = f"{self.storage_key}{MESSAGES_SUFFIX}"
        self._meta_key = f"{self.storage_key}{META_SUFFIX}"
        self._renew_on_read = renew_on_read
        self._expiry_callbacks: list[ExpiryCallback] = []
        self._observed_present = False
        self._expiry_fired = False
        if ttl_seconds is not None:
            self.set_ttl(ttl_seconds)

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
        self._observed_present = True
        self._expiry_fired = False

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """读取整个列表并按 ``created_at`` 稳定排序；键不存在时返回空列表。

        Redis List 本身保序，此处仍统一排序，是为了与其余后端语义完全一致
        （允许调用方乱序补写历史消息）。

        Raises:
            StorageError: redis 命令失败或存在无法解码的消息体时。
        """
        with self._guard("read"):
            raw = cast("list[bytes]", self._client.lrange(self._messages_key, 0, -1))
        if self._renew_on_read and raw:
            self.refresh_ttl()
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
        # 显式清理不是"过期"，复位观察状态避免 check_expiry 误报。
        self._observed_present = False
        self._expiry_fired = False

    # ------------------------------------------------------------------ #
    # 会话级 TTL
    # ------------------------------------------------------------------ #
    def set_ttl(self, ttl_seconds: int | None) -> None:
        """设置会话 TTL 并立即下发；``None`` 表示取消过期（下发 ``PERSIST``）。

        Args:
            ttl_seconds: 过期秒数；``None`` 取消已设置的过期时间。

        Raises:
            ValidationError: ``ttl_seconds`` 为非正数时。
            StorageError: redis 命令失败时。
        """
        super().set_ttl(ttl_seconds)
        if ttl_seconds is None:
            self._persist()

    def _apply_ttl(self, ttl_seconds: int) -> None:
        """在单个 pipeline 事务内为消息列表与元数据哈希重新计时。

        对不存在的键 ``EXPIRE`` 是空操作，因此空会话上调用同样安全。
        """
        with self._guard("expire"), self._client.pipeline(transaction=True) as pipe:
            pipe.expire(self._messages_key, ttl_seconds)
            pipe.expire(self._meta_key, ttl_seconds)
            pipe.execute()

    def ttl_remaining(self) -> int | None:
        """读取消息列表键在 Redis 中的剩余存活秒数。

        Returns:
            剩余秒数；键不存在或未设置过期时返回 ``None``。

        Raises:
            StorageError: redis 命令失败时。
        """
        with self._guard("ttl"):
            remaining = int(cast(int, self._client.ttl(self._messages_key)))
        return remaining if remaining >= 0 else None

    def exists(self) -> bool:
        """判断本会话在 Redis 中是否仍有数据（消息列表或元数据任一存在）。

        Raises:
            StorageError: redis 命令失败时。
        """
        with self._guard("exists"):
            return bool(self._client.exists(self._messages_key, self._meta_key))

    def on_expired(self, callback: ExpiryCallback) -> None:
        """注册会话过期回调，由 :meth:`check_expiry` 按注册顺序触发。

        Args:
            callback: 入参为本历史实例的可调用对象；同一实例可注册多个。
        """
        self._expiry_callbacks.append(callback)

    def check_expiry(self) -> bool:
        """主动探测会话是否已被 Redis 淘汰，首次探测到过期时触发回调。

        键过期发生在服务端，客户端收不到通知，故由调用方（如 lifecycle 层的
        归档调度器）定期轮询本方法。仅当本实例**曾观察到会话存在**（写入过消息，
        或此前探测时键仍在）才可能判定为过期，避免把从未写入的新会话误判为过期。
        会话被重新写入后过期状态自动复位，可再次触发回调。

        Returns:
            会话是否已过期；未设置 TTL 时恒为 ``False``。

        Raises:
            StorageError: redis 命令失败时；回调自身抛出的异常原样向上传播。
        """
        if self.ttl_seconds is None:
            return False
        if self.exists():
            self._observed_present = True
            self._expiry_fired = False
            return False
        if not self._observed_present:
            return False
        if not self._expiry_fired:
            self._expiry_fired = True
            for callback in self._expiry_callbacks:
                callback(self)
        return True

    def _persist(self) -> None:
        """移除两条键上的过期时间（键不存在时为空操作）。"""
        with self._guard("persist"), self._client.pipeline(transaction=True) as pipe:
            pipe.persist(self._messages_key)
            pipe.persist(self._meta_key)
            pipe.execute()

    # ------------------------------------------------------------------ #
    # Redis 后端专有能力
    # ------------------------------------------------------------------ #
    @property
    def renew_on_read(self) -> bool:
        """读取是否参与滑动续期。"""
        return self._renew_on_read

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
