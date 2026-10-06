"""真实中间件集成测试的公共夹具。

默认行为：**全部跳过**。集成用例需要同时满足两个条件才会真实执行：

1. 环境变量 ``MED_MEMORY_IT=1``（显式开关，避免普通 ``pytest`` 依赖真实中间件）；
2. 目标服务端口可连（本地 ``docker compose -f docker-compose.integration.yml up -d``）。

任一条件不满足时，夹具调用 ``pytest.skip(原因)``，原因文本里带可复制的启动命令。
这样「默认离线全绿」与「起容器即真跑」由同一套用例覆盖，无需维护两份断言。

用例本身复用 ``tests/test_stores/behavior.py`` 的跨后端行为基准套件，
与 fakeredis / SQLite 内存库 / fake ES 的单测共享同一份语义契约。
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# tests/test_stores/behavior.py 是跨后端共享的行为基准套件（文件名不以 test_ 开头，
# pytest 不会直接收集），各后端单测按顶层模块名 ``behavior`` 导入。
# 集成测试要复用同一份契约，故把该目录加入 sys.path。
_STORES_TESTS_DIR = Path(__file__).resolve().parents[1] / "test_stores"
if str(_STORES_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_STORES_TESTS_DIR))

from med_langchain_memory.testing import (  # noqa: E402
    IntegrationService,
    check_service,
    resolve_service,
)

#: 等待容器冷启动就绪的上限（秒）：Redis 秒级，MySQL / ES 需数十秒。
WAIT_SECONDS = 120.0

#: MySQL / ES 客户端握手重试间隔（秒）。
CONNECT_RETRY_INTERVAL_SECONDS = 1.0


def require_service(name: str) -> IntegrationService:
    """校验服务可用于集成测试，不可用时跳过当前用例。

    Args:
        name: 服务名（``redis`` / ``mysql`` / ``elasticsearch``）。

    Returns:
        解析后的服务地址（含环境变量覆盖）。

    Raises:
        pytest.skip.Exception: 开关未打开或服务端口不可达时。
    """
    status = check_service(resolve_service(name), wait_seconds=WAIT_SECONDS)
    if not status.ready:
        pytest.skip(status.reason)
    return status.service


@pytest.fixture(scope="session")
def redis_service() -> IntegrationService:
    """真实 Redis 服务地址（不可用时跳过）。"""
    return require_service("redis")


@pytest.fixture(scope="session")
def mysql_service() -> IntegrationService:
    """真实 MySQL 服务地址（不可用时跳过）。"""
    return require_service("mysql")


@pytest.fixture(scope="session")
def es_service() -> IntegrationService:
    """真实 Elasticsearch 服务地址（不可用时跳过）。"""
    return require_service("elasticsearch")


@pytest.fixture(scope="session")
def redis_live(redis_service: IntegrationService) -> Iterator[Any]:
    """真实 Redis 客户端（指向集成测试专用 DB，收尾清空）。"""
    redis_module = pytest.importorskip("redis", reason="redis is an optional dependency")
    client = redis_module.Redis.from_url(redis_service.url)
    try:
        client.ping()
    except redis_module.exceptions.RedisError as exc:
        pytest.skip(f"{redis_service.describe()} is not a usable redis server: {exc}")
    yield client
    client.flushdb()
    client.close()


@pytest.fixture(scope="session")
def mysql_live(mysql_service: IntegrationService) -> Iterator[Any]:
    """真实 MySQL 引擎（自动建表，收尾释放连接池）。

    需要宿主侧自备 DBAPI 驱动（如 ``pymysql``），缺失时本夹具跳过。
    """
    pytest.importorskip("sqlalchemy", reason="SQLAlchemy is an optional dependency")
    pytest.importorskip("pymysql", reason="a MySQL DBAPI driver (e.g. pymysql) is required")

    from sqlalchemy import create_engine, text
    from sqlalchemy.exc import SQLAlchemyError

    from med_langchain_memory.stores.mysql_schema import create_all

    engine = create_engine(mysql_service.url, pool_pre_ping=True)
    deadline = time.monotonic() + WAIT_SECONDS
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            last_error = None
            break
        except SQLAlchemyError as exc:  # 端口已开但实例仍在初始化
            last_error = exc
            time.sleep(CONNECT_RETRY_INTERVAL_SECONDS)
    if last_error is not None:
        engine.dispose()
        pytest.skip(f"{mysql_service.describe()} is not ready for queries: {last_error}")
    create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def es_live(es_service: IntegrationService) -> Iterator[Any]:
    """真实 Elasticsearch 客户端（等待集群就绪，收尾关闭连接）。"""
    es_module = pytest.importorskip(
        "elasticsearch", reason="elasticsearch is an optional dependency"
    )
    client = es_module.Elasticsearch(es_service.url)
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            if client.ping():
                break
        except es_module.TransportError:  # 端口已开但集群仍在启动
            pass
        time.sleep(CONNECT_RETRY_INTERVAL_SECONDS)
    else:
        client.close()
        pytest.skip(f"{es_service.describe()} did not answer ping within {WAIT_SECONDS:.0f}s")
    yield client
    client.close()
