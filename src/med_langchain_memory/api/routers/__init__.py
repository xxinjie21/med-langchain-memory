"""API 路由子包。

当前提供健康检查路由（存活 / 就绪）；后续迭代在此叠加会话、消息与管理端点。
"""

from __future__ import annotations

from .health import HealthProbe, HealthResponse, build_health_router, run_probe

__all__ = ["HealthProbe", "HealthResponse", "build_health_router", "run_probe"]
