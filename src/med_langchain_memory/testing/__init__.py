"""测试支撑层：可选真实中间件集成测试的服务探测与开关。

本包**不参与生产链路**（``med_langchain_memory/__init__.py`` 不导入它），
仅被 ``tests/test_integration/`` 使用，且只依赖标准库。

公开符号见 :mod:`med_langchain_memory.testing.services`。
"""

from __future__ import annotations

from .services import (
    COMPOSE_HINT,
    DEFAULT_PROBE_TIMEOUT_SECONDS,
    DEFAULT_SERVICES,
    DEFAULT_WAIT_INTERVAL_SECONDS,
    DEFAULT_WAIT_SECONDS,
    ENV_PREFIX,
    IntegrationService,
    ServiceStatus,
    check_service,
    integration_enabled,
    probe_tcp,
    resolve_service,
    wait_for_service,
)

__all__ = [
    "COMPOSE_HINT",
    "DEFAULT_PROBE_TIMEOUT_SECONDS",
    "DEFAULT_SERVICES",
    "DEFAULT_WAIT_INTERVAL_SECONDS",
    "DEFAULT_WAIT_SECONDS",
    "ENV_PREFIX",
    "IntegrationService",
    "ServiceStatus",
    "check_service",
    "integration_enabled",
    "probe_tcp",
    "resolve_service",
    "wait_for_service",
]
