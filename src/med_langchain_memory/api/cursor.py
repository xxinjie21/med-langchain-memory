"""消息列表的游标分页工具（不透明游标编解码 + 时序切片）。

医患会话的消息量大且**持续追加**，因此消息列表用游标分页而非 ``offset`` 分页：
游标承载「上一条消息的时序定位键」，翻页期间新写入的消息不会造成重复或漏读。

游标线格式：``base64url("<created_at>:<message_id>")``（去掉 ``=`` 填充）。
定位键为 ``(created_at, message_id)`` 二元组，与 :class:`MedMessage` 的时序排序
键一致；``message_id`` 为 UUIDv7，同毫秒内仍可稳定定序。

设计取舍：本模块只做「编解码 + 切片」，不读写存储。存储侧只需返回按时序升序的
全量消息即可，把范围查询下推到后端是后续存储实现的优化项，不影响对外契约。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import base64
import binascii
import uuid
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict, Field

from med_langchain_memory.domain.message import MedMessage
from med_langchain_memory.exceptions import ValidationError

#: 消息列表默认单页条数。
DEFAULT_MESSAGE_PAGE_SIZE = 50

#: 消息列表单页最大条数（防止一次拉取过多消息）。
MAX_MESSAGE_PAGE_SIZE = 200

#: 游标编解码使用的编码方式（URL 安全，可直接放进查询参数）。
_CURSOR_ENCODING = "utf-8"


class MessageCursor(BaseModel):
    """消息游标定位键：``(created_at, message_id)``。

    Attributes:
        created_at: 定位消息的创建时间（epoch 毫秒）。
        message_id: 定位消息的 ID。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    created_at: int = Field(gt=0)
    message_id: str = Field(min_length=1)

    @property
    def sort_key(self) -> tuple[int, str]:
        """返回用于时序比较的排序键 ``(created_at, message_id)``。"""
        return (self.created_at, self.message_id)

    @classmethod
    def of(cls, message: MedMessage) -> MessageCursor:
        """由消息构造游标。

        Args:
            message: 定位消息（通常是当前页的最后一条）。

        Returns:
            指向该消息位置的游标。
        """
        return cls(created_at=message.created_at, message_id=message.message_id)


def encode_cursor(cursor: MessageCursor) -> str:
    """把游标编码为对调用方不透明的字符串。

    Args:
        cursor: 游标定位键。

    Returns:
        形如 ``MTc2MjU2MDAwMDAwMDoxOWFh...`` 的 URL 安全字符串（无 ``=`` 填充）。
    """
    raw = f"{cursor.created_at}:{cursor.message_id}"
    return base64.urlsafe_b64encode(raw.encode(_CURSOR_ENCODING)).decode("ascii").rstrip("=")


def decode_cursor(token: str) -> MessageCursor:
    """把游标字符串解码为定位键。

    Args:
        token: :func:`encode_cursor` 产出的字符串。

    Returns:
        解析后的游标定位键。

    Raises:
        ValidationError: 游标为空、非 base64url、缺少分隔符、时间戳非法或
            ``message_id`` 不是合法 UUID 时。
    """
    if not token:
        raise ValidationError("cursor must be a non-empty string")
    padded = token + "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(padded.encode("ascii")).decode(_CURSOR_ENCODING)
    except (binascii.Error, UnicodeDecodeError, UnicodeEncodeError, ValueError) as exc:
        raise ValidationError(f"malformed cursor: {token!r}") from exc

    created_at_text, separator, message_id = raw.partition(":")
    if not separator:
        raise ValidationError(f"malformed cursor: {token!r}")
    try:
        created_at = int(created_at_text)
    except ValueError as exc:
        raise ValidationError(f"malformed cursor: {token!r}") from exc
    try:
        uuid.UUID(message_id)
    except ValueError as exc:
        raise ValidationError(f"malformed cursor: {token!r}") from exc
    if created_at <= 0:
        raise ValidationError(f"malformed cursor: {token!r}")
    return MessageCursor(created_at=created_at, message_id=message_id)


def paginate(
    messages: Sequence[MedMessage],
    *,
    limit: int,
    cursor: MessageCursor | None = None,
) -> tuple[list[MedMessage], MessageCursor | None]:
    """按 ``(created_at, message_id)`` 升序切片，返回当前页与下一页游标。

    严格晚于 ``cursor`` 定位键的消息才进入结果集，因此同一游标可安全重放。
    当且仅当「还有剩余消息」时返回非 ``None`` 的 ``next_cursor``。

    Args:
        messages: 候选消息（顺序不限，内部会按时序升序排序）。
        limit: 单页最大条数，必须为正整数。
        cursor: 上一页返回的游标；``None`` 表示从头开始。

    Returns:
        ``(当前页消息, 下一页游标或 None)``。

    Raises:
        ValidationError: ``limit`` 不是正整数时。
    """
    if limit < 1:
        raise ValidationError(f"limit must be a positive integer, got {limit}")
    ordered = sorted(messages, key=lambda message: (message.created_at, message.message_id))
    if cursor is not None:
        boundary = cursor.sort_key
        ordered = [
            message for message in ordered if (message.created_at, message.message_id) > boundary
        ]
    page = ordered[:limit]
    if len(ordered) <= limit:
        return page, None
    return page, MessageCursor.of(page[-1])
