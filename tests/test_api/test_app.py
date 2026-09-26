"""应用工厂单元测试（D30）。

覆盖：默认与自定义配置装配、文档路由开关、``app.state.settings`` 暴露、
请求日志中间件接线、健康检查路由挂载与就绪探针透传、环境变量驱动的配置，
以及 :func:`configure_logging` 的级别设置。
"""

from __future__ import annotations

import logging

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from med_langchain_memory.api import (
    DEFAULT_REQUEST_ID_HEADER,
    configure_logging,
    create_app,
)
from med_langchain_memory.config import PACKAGE_LOGGER_NAME, MedMemorySettings


def test_create_app_uses_default_settings() -> None:
    """无参构造时使用默认配置，并挂载健康检查路由。"""
    app = create_app()
    assert isinstance(app, FastAPI)
    assert app.title == "med-langchain-memory"
    client = TestClient(app)
    assert client.get("/health").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_create_app_uses_injected_settings() -> None:
    """显式传入的配置被完整采用，并可从 ``app.state.settings`` 读回。"""
    settings = MedMemorySettings(app_name="gateway", app_version="2.3.4", log_level="WARNING")
    app = create_app(settings)
    assert app.title == "gateway"
    assert app.version == "2.3.4"
    assert app.state.settings is settings


def test_create_app_reads_settings_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺省配置从 ``MED_MEMORY_`` 环境变量构造（边界：无显式参数）。"""
    monkeypatch.setenv("MED_MEMORY_APP_NAME", "env-gateway")
    app = create_app()
    assert app.title == "env-gateway"
    assert app.state.settings.app_name == "env-gateway"


def test_docs_routes_are_mounted_when_enabled() -> None:
    """``docs_enabled=True`` 时 Swagger / ReDoc / OpenAPI 均可访问。"""
    client = TestClient(create_app(MedMemorySettings(docs_enabled=True)))
    assert client.get("/docs").status_code == 200
    assert client.get("/redoc").status_code == 200
    assert client.get("/openapi.json").status_code == 200


def test_docs_routes_are_absent_when_disabled() -> None:
    """``docs_enabled=False`` 时三个文档路由全部 404（边界）。"""
    client = TestClient(create_app(MedMemorySettings(docs_enabled=False)))
    for path in ("/docs", "/redoc", "/openapi.json"):
        response = client.get(path)
        assert response.status_code == 404
        assert response.json()["error"] == "http_404"


def test_openapi_schema_exposes_health_paths() -> None:
    """OpenAPI 文档包含健康检查端点。"""
    schema = TestClient(create_app()).get("/openapi.json").json()
    assert "/health" in schema["paths"]
    assert "/health/ready" in schema["paths"]


def test_request_logging_middleware_is_wired() -> None:
    """请求日志中间件已生效：响应携带请求 ID 头。"""
    response = TestClient(create_app()).get("/health")
    assert DEFAULT_REQUEST_ID_HEADER in response.headers


def test_health_probes_are_forwarded_to_readiness() -> None:
    """``create_app`` 的探针映射透传到就绪检查。"""
    app = create_app(health_probes={"redis": lambda: False})
    response = TestClient(app).get("/health/ready")
    assert response.status_code == 503
    assert response.json()["checks"] == {"redis": False}


def test_configure_logging_sets_package_logger_level() -> None:
    """``configure_logging`` 设置包级 logger 级别并返回该 logger。"""
    logger = configure_logging("DEBUG")
    assert logger.name == PACKAGE_LOGGER_NAME
    assert logger.level == logging.DEBUG
    configure_logging("INFO")
    assert logging.getLogger(PACKAGE_LOGGER_NAME).level == logging.INFO
