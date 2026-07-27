"""会话元数据模型 ``SessionMeta`` 与状态机 ``SessionStatus``。"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from med_langchain_memory.exceptions import StateTransitionError

from .message import IdStr, now_millis


class SessionStatus(StrEnum):
    """会话生命周期状态枚举。

    合法流转：
    ``ACTIVE -> CLOSED | ARCHIVED``，
    ``CLOSED -> ACTIVE(复诊重开) | ARCHIVED``，
    ``ARCHIVED -> DELETED(软删除)``，
    ``DELETED`` 为终态。
    """

    ACTIVE = "active"
    CLOSED = "closed"
    ARCHIVED = "archived"
    DELETED = "deleted"


_ALLOWED_TRANSITIONS: dict[SessionStatus, frozenset[SessionStatus]] = {
    SessionStatus.ACTIVE: frozenset({SessionStatus.CLOSED, SessionStatus.ARCHIVED}),
    SessionStatus.CLOSED: frozenset({SessionStatus.ACTIVE, SessionStatus.ARCHIVED}),
    SessionStatus.ARCHIVED: frozenset({SessionStatus.DELETED}),
    SessionStatus.DELETED: frozenset(),
}


class SessionMeta(BaseModel):
    """一次医患问诊会话的元数据。"""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    session_id: IdStr
    tenant_id: IdStr
    dept_id: IdStr
    patient_id: IdStr
    status: SessionStatus = SessionStatus.ACTIVE
    message_count: int = Field(default=0, ge=0)
    created_at: int = Field(default_factory=now_millis, gt=0)
    updated_at: int = Field(default=0, ge=0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_updated_at(self) -> SessionMeta:
        """updated_at 缺省时对齐 created_at，且不得早于 created_at。"""
        if self.updated_at == 0:
            object.__setattr__(self, "updated_at", self.created_at)
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at")
        return self

    @property
    def storage_key(self) -> str:
        """返回统一存储键 ``med:chat:{tenant_id}:{dept_id}:{session_id}``。"""
        return f"med:chat:{self.tenant_id}:{self.dept_id}:{self.session_id}"

    def can_transition_to(self, target: SessionStatus) -> bool:
        """判断当前状态能否流转到 ``target``。"""
        return target in _ALLOWED_TRANSITIONS[self.status]

    def transition_to(self, target: SessionStatus) -> SessionMeta:
        """返回流转到 ``target`` 后的新 ``SessionMeta``（不可变更新）。

        Raises:
            StateTransitionError: 当流转不被状态机允许时。
        """
        if not self.can_transition_to(target):
            raise StateTransitionError(
                f"illegal session transition: {self.status.value} -> {target.value}"
            )
        return self.model_copy(update={"status": target, "updated_at": now_millis()})

    def touch(self, added_messages: int = 1) -> SessionMeta:
        """记录新增消息：累加计数并刷新 ``updated_at``，返回新实例。

        Raises:
            ValueError: 当 ``added_messages`` 为负数时。
        """
        if added_messages < 0:
            raise ValueError("added_messages must be >= 0")
        return self.model_copy(
            update={
                "message_count": self.message_count + added_messages,
                "updated_at": now_millis(),
            }
        )
