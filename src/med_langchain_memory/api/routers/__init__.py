"""API 路由子包。

当前提供健康检查路由（存活 / 就绪）、会话管理路由（创建 / 查询 / 关闭 / 归档 / 软删除）、
消息路由（批量追加 / 游标分页查询）与管理路由（归档统计 / 跨存储迁移 / 快照导出）。
"""

from __future__ import annotations

from .admin import build_admin_router
from .health import HealthProbe, HealthResponse, build_health_router, run_probe
from .messages import build_messages_router
from .sessions import build_sessions_router

__all__ = [
    "HealthProbe",
    "HealthResponse",
    "build_admin_router",
    "build_health_router",
    "build_messages_router",
    "build_sessions_router",
    "run_probe",
]
