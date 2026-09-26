"""健康检查端点。

* ``GET /health`` —— 存活检查（liveness）：进程能响应即为健康，不探测任何外部依赖；
* ``GET /health/ready`` —— 就绪检查（readiness）：逐个执行注入的探针，
  全部为真返回 200，任一为假返回 503（``status`` 为 ``degraded``）。

探针是一个无参、返回 ``bool`` 的可调用对象（典型实现：``lambda: redis_client.ping()``）。
探针自身抛异常时视为该项不健康，异常不会冒泡打断整体检查。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, ConfigDict, Field

from med_langchain_memory.config import MedMemorySettings

#: 就绪探针类型：无参调用，返回 ``True`` 表示该依赖健康。
HealthProbe = Callable[[], bool]


class HealthResponse(BaseModel):
    """健康检查响应体。

    Attributes:
        status: ``ok`` 表示全部探针通过，``degraded`` 表示存在不健康依赖。
        app: 服务名。
        version: 服务版本。
        checks: 各探针名称 → 是否健康；存活检查下为空字典。
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["ok", "degraded"]
    app: str
    version: str
    checks: dict[str, bool] = Field(default_factory=dict)


def run_probe(probe: HealthProbe) -> bool:
    """执行单个探针并把结果规整为 ``bool``。

    Args:
        probe: 无参可调用对象。

    Returns:
        探针返回值转成的布尔值；探针抛异常时返回 ``False``。
    """
    try:
        return bool(probe())
    except Exception:
        return False


def build_health_router(
    settings: MedMemorySettings,
    probes: Mapping[str, HealthProbe] | None = None,
) -> APIRouter:
    """构建健康检查路由。

    Args:
        settings: 服务配置，用于回显服务名与版本。
        probes: 就绪探针映射（名称 → 无参可调用），为空时 ``/health/ready`` 恒为健康。

    Returns:
        已注册 ``/health`` 与 ``/health/ready`` 的 ``APIRouter``。
    """
    router = APIRouter(tags=["health"])
    registered: dict[str, HealthProbe] = dict(probes or {})

    def _body(state: Literal["ok", "degraded"], checks: dict[str, bool]) -> HealthResponse:
        return HealthResponse(
            status=state,
            app=settings.app_name,
            version=settings.app_version,
            checks=checks,
        )

    @router.get("/health", response_model=HealthResponse, summary="存活检查")
    async def liveness() -> HealthResponse:
        """存活检查：进程可响应即健康。"""
        return _body("ok", {})

    @router.get(
        "/health/ready",
        response_model=HealthResponse,
        summary="就绪检查",
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
    )
    async def readiness(response: Response) -> HealthResponse:
        """就绪检查：全部探针通过返回 200，否则返回 503 且 ``status`` 为 ``degraded``。"""
        checks = {name: run_probe(probe) for name, probe in registered.items()}
        healthy = all(checks.values())
        if not healthy:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return _body("ok" if healthy else "degraded", checks)

    return router
