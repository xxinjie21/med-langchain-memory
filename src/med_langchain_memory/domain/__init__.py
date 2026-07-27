"""领域模型层：消息、会话与审计的核心数据结构。"""

from __future__ import annotations

from .message import MedMessage, MessageRole, new_message_id
from .session import SessionMeta, SessionStatus

__all__ = [
    "MedMessage",
    "MessageRole",
    "new_message_id",
    "SessionMeta",
    "SessionStatus",
]
