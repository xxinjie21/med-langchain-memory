"""存储适配层：统一的医疗会话历史抽象、工厂注册器与各存储引擎实现。"""

from __future__ import annotations

from .base import (
    MED_ROLE_KEY,
    MedChatMessageHistory,
    from_langchain_message,
    to_langchain_message,
)
from .factory import StoreConfig, StoreFactory

__all__ = [
    "MED_ROLE_KEY",
    "MedChatMessageHistory",
    "StoreConfig",
    "StoreFactory",
    "from_langchain_message",
    "to_langchain_message",
]
