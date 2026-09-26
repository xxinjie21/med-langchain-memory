"""健康检查端点单元测试（D30）。

覆盖：存活检查、就绪检查在「无探针 / 全部通过 / 存在失败 / 探针抛异常」四种情形下的
状态码与响应体，以及 :func:`run_probe` 的结果规整与异常吞并行为。
"""

from __future__ import annotations

from collections.abc import Mapping

from fastapi.testclient import TestClient

import med_langchain_memory
from med_langchain_memory.api import HealthProbe, create_app, run_probe
from med_langchain_memory.config import MedMemorySettings


def _client(probes: Mapping[str, HealthProbe] | None = None) -> TestClient:
    """构造仅挂载健康检查路由的测试客户端。"""
    return TestClient(create_app(MedMemorySettings(), health_probes=probes))


def test_liveness_returns_ok() -> None:
    """存活检查恒返回 200 且不携带任何探针结果。"""
    response = _client().get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "app": "med-langchain-memory",
        "version": "0.1.0",
        "checks": {},
    }


def test_readiness_without_probes_is_healthy() -> None:
    """未配置探针时就绪检查视为健康（边界：空探针集合）。"""
    response = _client().get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["checks"] == {}


def test_readiness_all_probes_pass() -> None:
    """全部探针通过 → 200 且逐项回显 ``True``。"""
    response = _client({"redis": lambda: True, "mysql": lambda: True}).get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["checks"] == {"redis": True, "mysql": True}


def test_readiness_reports_degraded_when_a_probe_fails() -> None:
    """任一探针失败 → 503 且 ``status`` 为 ``degraded``（边界：部分失败）。"""
    response = _client({"redis": lambda: True, "mysql": lambda: False}).get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "degraded"
    assert body["checks"] == {"redis": True, "mysql": False}


def test_readiness_reports_degraded_when_probe_raises() -> None:
    """探针抛异常视为不健康，异常不冒泡（边界：探针实现缺陷）。"""

    def _boom() -> bool:
        raise RuntimeError("connection reset")

    response = _client({"es": _boom}).get("/health/ready")
    assert response.status_code == 503
    assert response.json() == {
        "status": "degraded",
        "app": "med-langchain-memory",
        "version": med_langchain_memory.__version__,
        "checks": {"es": False},
    }


def test_readiness_probes_are_evaluated_per_request() -> None:
    """每次请求都重新执行探针，不缓存结果。"""
    calls: list[int] = []

    def _counting() -> bool:
        calls.append(1)
        return True

    client = _client({"redis": _counting})
    client.get("/health/ready")
    client.get("/health/ready")
    assert len(calls) == 2


def test_readiness_reflects_custom_settings() -> None:
    """就绪响应回显自定义服务名与版本。"""
    settings = MedMemorySettings(app_name="gateway", app_version="2.0.0")
    response = TestClient(create_app(settings, health_probes={"redis": lambda: True})).get(
        "/health/ready"
    )
    assert response.status_code == 200
    assert response.json()["app"] == "gateway"
    assert response.json()["version"] == "2.0.0"


def test_run_probe_coerces_truthy_and_falsy_values() -> None:
    """探针返回值被规整为 ``bool``（边界：非严格布尔返回）。"""
    assert run_probe(lambda: 1) is True
    assert run_probe(lambda: 0) is False


def test_run_probe_swallows_exceptions() -> None:
    """探针抛异常时返回 ``False`` 而非向上抛出。"""

    def _boom() -> bool:
        raise ValueError("bad probe")

    assert run_probe(_boom) is False
