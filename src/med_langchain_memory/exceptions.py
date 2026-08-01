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


class TenantIsolationError(MedMemoryError):
    """跨租户 / 跨科室越权访问会话数据时抛出。"""


class StoreRegistrationError(MedMemoryError):
    """存储适配器注册失败（名称非法、重复注册或类型不合法）。"""


class StoreNotFoundError(MedMemoryError):
    """按名称查找存储适配器失败（未注册的后端）。"""
