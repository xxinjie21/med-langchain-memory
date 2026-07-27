"""统一异常体系。

所有 med-langchain-memory 抛出的业务异常均继承自 :class:`MedMemoryError`，
便于上层（API / runnable）做统一捕获与错误响应。
"""

from __future__ import annotations


class MedMemoryError(Exception):
    """库中所有业务异常的基类。"""


class ValidationError(MedMemoryError):
    """领域模型或请求参数校验失败。"""


class StateTransitionError(MedMemoryError):
    """会话状态非法流转时抛出。"""


class StorageError(MedMemoryError):
    """存储适配器读写失败。"""
