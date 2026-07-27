"""医患对话消息模型 ``MedMessage``。

字段对齐 ROADMAP《对外存储规范》中的跨语言 protobuf 字段定义，
本模块只做结构与字段校验，不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import os
import time
import uuid
from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

#: 各类业务 ID（租户/科室/会话/患者/消息）的合法字符约束。
#: 禁止 ``:`` 与 ``{}``，避免破坏 Redis 键规范 ``med:chat:{tenant}:{dept}:{session}``。
IdStr = Annotated[str, StringConstraints(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")]


class MessageRole(StrEnum):
    """消息发送方角色枚举（与 protobuf enum 对齐）。"""

    PATIENT = "patient"
    DOCTOR = "doctor"
    ASSISTANT = "assistant"
    SYSTEM = "system"


def now_millis() -> int:
    """返回当前 epoch 毫秒时间戳。"""
    return time.time_ns() // 1_000_000


def new_message_id() -> str:
    """生成 UUIDv7 风格的时序有序消息 ID。

    布局遵循 RFC 9562：前 48 bit 为 epoch 毫秒时间戳，
    version=7，variant=RFC 4122，其余位为随机数。
    保证按生成时间字典序单调递增（同毫秒内按随机位排序）。
    """
    ts = now_millis() & 0xFFFF_FFFF_FFFF
    rand = int.from_bytes(os.urandom(10), "big")
    value = ts << 80
    value |= 0x7 << 76  # version 7
    value |= ((rand >> 62) & 0xFFF) << 64  # rand_a (12 bit)
    value |= 0b10 << 62  # variant
    value |= rand & 0x3FFF_FFFF_FFFF_FFFF  # rand_b (62 bit)
    return str(uuid.UUID(int=value))


class MedMessage(BaseModel):
    """医疗会话中的单条消息。

    对齐跨语言存储规范：所有下游服务落库/读取的消息均使用该字段集。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_id: str = Field(default_factory=new_message_id)
    session_id: IdStr
    tenant_id: IdStr
    dept_id: IdStr
    patient_id: IdStr
    role: MessageRole
    content: str = Field(min_length=1)
    token_count: int = Field(default=0, ge=0)
    masked: bool = False
    created_at: int = Field(default_factory=now_millis, gt=0)
    metadata: dict[str, str] = Field(default_factory=dict)

    @field_validator("message_id")
    @classmethod
    def _validate_message_id(cls, v: str) -> str:
        """校验 message_id 必须为合法 UUID 字符串。"""
        uuid.UUID(v)
        return v

    @property
    def storage_key(self) -> str:
        """返回该消息所属会话的统一存储键。

        格式：``med:chat:{tenant_id}:{dept_id}:{session_id}``。
        """
        return f"med:chat:{self.tenant_id}:{self.dept_id}:{self.session_id}"
