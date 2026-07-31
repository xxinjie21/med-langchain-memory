"""存储适配层抽象：:class:`MedChatMessageHistory`。

在 LangChain ``BaseChatMessageHistory`` 契约之上补充医疗中间件所需的三类钩子：

* **租户钩子**：``tenant_id`` / ``dept_id`` / ``patient_id`` 命名空间与统一存储键，越权访问拒绝；
* **TTL 钩子**：会话级过期时间设置、写入滑动续期、无原生 TTL 存储的逻辑过期判定；
* **归档钩子**：会话状态流转至 ``ARCHIVED`` 并导出全量消息，供归档层写入。

子类只需实现三个存储原语 ``_append`` / ``_read`` / ``clear``，
其余 LangChain 契约方法（``messages`` / ``add_message`` / ``add_messages`` 及其异步版本）由本基类提供。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, ClassVar

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.domain.message import MedMessage, MessageRole, now_millis
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import (
    StorageError,
    TenantIsolationError,
    ValidationError,
)

#: 医疗角色写入 LangChain ``additional_kwargs`` 时使用的键名。
MED_ROLE_KEY = "med_role"

_ROLE_TO_LC: dict[MessageRole, type[BaseMessage]] = {
    MessageRole.PATIENT: HumanMessage,
    MessageRole.DOCTOR: HumanMessage,
    MessageRole.ASSISTANT: AIMessage,
    MessageRole.SYSTEM: SystemMessage,
}

#: 缺少 ``med_role`` 扩展字段时，按 LangChain 消息类型回退推断角色。
_LC_TYPE_TO_ROLE: dict[str, MessageRole] = {
    "human": MessageRole.PATIENT,
    "ai": MessageRole.ASSISTANT,
    "system": MessageRole.SYSTEM,
}


def _is_uuid(value: object) -> bool:
    """判断给定值是否为合法 UUID 字符串。"""
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except ValueError:
        return False
    return True


def to_langchain_message(message: MedMessage) -> BaseMessage:
    """将 :class:`MedMessage` 转换为 LangChain ``BaseMessage``。

    医疗扩展字段（角色、租户命名空间、token 数、脱敏标记、时间戳、自定义标签）
    写入 ``additional_kwargs``，保证可无损还原。

    Args:
        message: 待转换的医疗消息。

    Returns:
        对应的 LangChain 消息实例（patient/doctor -> Human，assistant -> AI，system -> System）。
    """
    message_cls = _ROLE_TO_LC[message.role]
    return message_cls(
        id=message.message_id,
        content=message.content,
        additional_kwargs={
            MED_ROLE_KEY: message.role.value,
            "session_id": message.session_id,
            "tenant_id": message.tenant_id,
            "dept_id": message.dept_id,
            "patient_id": message.patient_id,
            "token_count": message.token_count,
            "masked": message.masked,
            "created_at": message.created_at,
            "metadata": dict(message.metadata),
        },
    )


def from_langchain_message(
    message: BaseMessage,
    *,
    session_id: str,
    tenant_id: str,
    dept_id: str,
    patient_id: str,
) -> MedMessage:
    """将 LangChain ``BaseMessage`` 还原为 :class:`MedMessage`。

    优先取 ``additional_kwargs`` 中由 :func:`to_langchain_message` 写入的医疗扩展字段；
    字段缺失时按消息类型推断角色，并用调用方给出的命名空间补齐 ID。

    Args:
        message: LangChain 消息。
        session_id: 归属会话 ID（扩展字段缺失时使用）。
        tenant_id: 归属租户 ID（扩展字段缺失时使用）。
        dept_id: 归属科室 ID（扩展字段缺失时使用）。
        patient_id: 归属患者 ID（扩展字段缺失时使用）。

    Returns:
        转换后的医疗消息。

    Raises:
        ValidationError: 消息类型不受支持、角色非法、内容非纯文本或字段校验失败时。
    """
    extra: dict[str, Any] = dict(message.additional_kwargs)
    role = _resolve_role(message, extra)

    if not isinstance(message.content, str):
        raise ValidationError("only plain text message content is supported")

    fields: dict[str, Any] = {
        "session_id": extra.get("session_id", session_id),
        "tenant_id": extra.get("tenant_id", tenant_id),
        "dept_id": extra.get("dept_id", dept_id),
        "patient_id": extra.get("patient_id", patient_id),
        "role": role,
        "content": message.content,
        "token_count": extra.get("token_count", 0),
        "masked": extra.get("masked", False),
        "metadata": {str(k): str(v) for k, v in dict(extra.get("metadata", {})).items()},
    }
    if _is_uuid(message.id):
        fields["message_id"] = message.id
    created_at = extra.get("created_at")
    if isinstance(created_at, int) and created_at > 0:
        fields["created_at"] = created_at

    try:
        return MedMessage(**fields)
    except PydanticValidationError as exc:
        raise ValidationError(f"cannot convert langchain message: {exc}") from exc


def _resolve_role(message: BaseMessage, extra: dict[str, Any]) -> MessageRole:
    """从扩展字段或消息类型解析医疗角色。"""
    raw_role = extra.get(MED_ROLE_KEY)
    if raw_role is not None:
        try:
            return MessageRole(raw_role)
        except ValueError as exc:
            raise ValidationError(f"unknown {MED_ROLE_KEY}: {raw_role!r}") from exc
    role = _LC_TYPE_TO_ROLE.get(message.type)
    if role is None:
        raise ValidationError(f"unsupported langchain message type: {message.type!r}")
    return role


class MedChatMessageHistory(BaseChatMessageHistory, ABC):
    """医疗会话消息历史抽象基类，所有存储适配器的统一父类。

    子类必须实现 :meth:`_append`、:meth:`_read` 与 :meth:`clear`；
    支持原生 TTL 的子类需将 :attr:`supports_ttl` 置为 ``True`` 并覆写 :meth:`_apply_ttl`。
    """

    #: 底层存储是否支持原生 TTL（Redis 为 True，内存/文件/MySQL 为 False）。
    supports_ttl: ClassVar[bool] = False

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """初始化会话历史。

        Args:
            session_id: 会话 ID。
            tenant_id: 医院/机构租户 ID。
            dept_id: 科室 ID。
            patient_id: 患者 ID。
            ttl_seconds: 会话级 TTL（秒），``None`` 表示永不过期。

        Raises:
            ValidationError: 任一 ID 不符合命名规范时。
            StorageError: 指定了 TTL 但底层存储不支持时。
        """
        try:
            self._meta = SessionMeta(
                session_id=session_id,
                tenant_id=tenant_id,
                dept_id=dept_id,
                patient_id=patient_id,
            )
        except PydanticValidationError as exc:
            raise ValidationError(f"invalid session namespace: {exc}") from exc
        self._ttl_seconds: int | None = None
        if ttl_seconds is not None:
            self.set_ttl(ttl_seconds)

    # ------------------------------------------------------------------ #
    # 租户钩子
    # ------------------------------------------------------------------ #
    @property
    def session_id(self) -> str:
        """当前会话 ID。"""
        return self._meta.session_id

    @property
    def tenant_id(self) -> str:
        """当前租户 ID。"""
        return self._meta.tenant_id

    @property
    def dept_id(self) -> str:
        """当前科室 ID。"""
        return self._meta.dept_id

    @property
    def patient_id(self) -> str:
        """当前患者 ID。"""
        return self._meta.patient_id

    @property
    def storage_key(self) -> str:
        """统一存储键 ``med:chat:{tenant_id}:{dept_id}:{session_id}``。"""
        return self._meta.storage_key

    @property
    def session_meta(self) -> SessionMeta:
        """当前会话元数据快照（不可变对象）。"""
        return self._meta

    def belongs_to(self, tenant_id: str, dept_id: str | None = None) -> bool:
        """判断本会话是否属于给定租户（可选科室）。"""
        if tenant_id != self.tenant_id:
            return False
        return dept_id is None or dept_id == self.dept_id

    def assert_tenant(self, tenant_id: str, dept_id: str | None = None) -> None:
        """校验调用方租户/科室归属，越权时拒绝访问。

        Raises:
            TenantIsolationError: 调用方租户或科室与本会话不匹配时。
        """
        if not self.belongs_to(tenant_id, dept_id):
            raise TenantIsolationError(
                f"access denied: {tenant_id}:{dept_id} cannot access {self.storage_key}"
            )

    # ------------------------------------------------------------------ #
    # LangChain BaseChatMessageHistory 契约
    # ------------------------------------------------------------------ #
    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        """按时序返回本会话全部消息（LangChain 格式）。"""
        return [to_langchain_message(m) for m in self._read()]

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """批量追加 LangChain 消息（``add_message`` 亦复用本实现）。

        Raises:
            ValidationError: 任一消息无法转换为 :class:`MedMessage` 时。
        """
        self.add_med_messages([self._coerce(m) for m in messages])

    @abstractmethod
    def clear(self) -> None:
        """清除本会话在底层存储中的全部消息。"""

    # ------------------------------------------------------------------ #
    # 医疗消息读写
    # ------------------------------------------------------------------ #
    def add_med_messages(self, messages: Sequence[MedMessage]) -> None:
        """追加医疗消息：校验归属 -> 落库 -> 刷新元数据与 TTL。

        Args:
            messages: 待写入消息，空序列为空操作。

        Raises:
            StorageError: 存在不属于本会话命名空间的消息时。
        """
        if not messages:
            return
        for message in messages:
            if message.storage_key != self.storage_key:
                raise StorageError(
                    f"message {message.message_id} belongs to {message.storage_key}, "
                    f"not {self.storage_key}"
                )
        self._append(list(messages))
        self._meta = self._meta.touch(len(messages))
        self.refresh_ttl()

    def get_med_messages(self, limit: int | None = None) -> list[MedMessage]:
        """按时序读取医疗消息。

        Args:
            limit: 仅返回最近 ``limit`` 条；``None`` 表示全部。

        Returns:
            时序升序的消息列表。

        Raises:
            ValidationError: ``limit`` 为非正数时。
        """
        if limit is not None and limit <= 0:
            raise ValidationError("limit must be a positive integer or None")
        return self._read(limit)

    # ------------------------------------------------------------------ #
    # TTL 钩子
    # ------------------------------------------------------------------ #
    @property
    def ttl_seconds(self) -> int | None:
        """当前会话 TTL（秒），``None`` 表示永不过期。"""
        return self._ttl_seconds

    def set_ttl(self, ttl_seconds: int | None) -> None:
        """设置会话级 TTL 并立即下发到底层存储。

        Args:
            ttl_seconds: 过期秒数；``None`` 表示取消过期。

        Raises:
            ValidationError: ``ttl_seconds`` 为非正数时。
            StorageError: 底层存储不支持原生 TTL 却设置了过期时间时。
        """
        if ttl_seconds is not None:
            if ttl_seconds <= 0:
                raise ValidationError("ttl_seconds must be a positive integer or None")
            if not self.supports_ttl:
                raise StorageError(f"{type(self).__name__} does not support native ttl")
        self._ttl_seconds = ttl_seconds
        self.refresh_ttl()

    def refresh_ttl(self) -> bool:
        """滑动续期：已设置 TTL 且存储支持时重新计时。

        Returns:
            是否实际执行了续期。
        """
        if not self.supports_ttl or self._ttl_seconds is None:
            return False
        self._apply_ttl(self._ttl_seconds)
        return True

    def _apply_ttl(self, ttl_seconds: int) -> None:
        """将 TTL 下发到底层存储；``supports_ttl`` 为真的子类必须覆写。"""
        raise NotImplementedError(f"{type(self).__name__} must override _apply_ttl")

    def is_expired(self, now_ms: int | None = None) -> bool:
        """基于 ``updated_at`` + TTL 判断会话是否逻辑过期。

        供内存/文件等无原生 TTL 的存储做惰性淘汰。

        Args:
            now_ms: 当前 epoch 毫秒，缺省取系统时间。

        Returns:
            未设置 TTL 时恒为 ``False``。
        """
        if self._ttl_seconds is None:
            return False
        now = now_millis() if now_ms is None else now_ms
        return now >= self._meta.updated_at + self._ttl_seconds * 1000

    # ------------------------------------------------------------------ #
    # 归档钩子
    # ------------------------------------------------------------------ #
    def archive(self) -> tuple[SessionMeta, list[MedMessage]]:
        """将会话流转至 ``ARCHIVED`` 并导出全量消息。

        仅负责状态流转与数据导出，不清理热存储数据（由归档调度方决定）。

        Returns:
            ``(归档后的会话元数据, 全量消息)``。

        Raises:
            StateTransitionError: 当前状态不允许归档时（如已 DELETED）。
        """
        messages = self._read()
        self._meta = self._meta.transition_to(SessionStatus.ARCHIVED)
        return self._meta, messages

    # ------------------------------------------------------------------ #
    # 子类需实现的存储原语
    # ------------------------------------------------------------------ #
    @abstractmethod
    def _append(self, messages: list[MedMessage]) -> None:
        """将消息按时序追加写入底层存储。"""

    @abstractmethod
    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """从底层存储按时序读取消息；``limit`` 为最近 N 条。"""

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _coerce(self, message: BaseMessage) -> MedMessage:
        """用本会话命名空间将 LangChain 消息转换为医疗消息。"""
        return from_langchain_message(
            message,
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            dept_id=self.dept_id,
            patient_id=self.patient_id,
        )
