"""API 路由子包。

当前提供健康检查路由（存活 / 就绪）与会话管理路由（创建 / 查询 / 关闭 / 归档 / 软删除）；
后续迭代在此叠加消息端点与管理端点。
"""

from __future__ import annotations

from .health import HealthProbe, HealthResponse, build_health_router, run_probe
from .sessions import build_sessions_router

__all__ = [
    "HealthProbe",
    "HealthResponse",
    "build_health_router",
    "build_sessions_router",
    "run_probe",
]
