"""管理接口单元测试（D33）。

覆盖三个端点的正向与边界 / 异常路径：

* ``GET  /admin/stats`` —— 空命名空间、按状态分组计数、跨租户隔离、超单页上限翻页；
* ``POST /admin/sessions/{id}/migrate`` —— 迁移成功、幂等重跑、多批次、跳过校验、
  已软删除会话仍可迁移、会话 / 后端不存在 404、请求体校验 422；
* ``POST /admin/sessions/{id}/snapshot`` —— 文件包可解码并恢复、SHA-256 与字节数自洽、
  空会话、自定义 schema 版本、篡改后校验和不匹配、异常路径；
* 管理 DTO 契约与 :func:`get_history_resolver` 依赖注入行为。

外部依赖全部使用替身：会话仓储用进程内实现，存储句柄用测试内的
:class:`_IsolatedHistory`（实例级列表，避免类级共享导致「源 / 目标」互相污染）。
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError
from starlette.requests import Request

from med_langchain_memory.api import (
    MAX_MIGRATION_BATCH_SIZE,
    AdminStatsResponse,
    MigrationRequest,
    MigrationResponse,
    SnapshotResponse,
    create_app,
    get_history_resolver,
)
from med_langchain_memory.config import MedMemorySettings
from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import IntegrityError, StoreNotFoundError
from med_langchain_memory.lifecycle import (
    MigrationResult,
    SessionSnapshotPackage,
    SessionSnapshotter,
)
from med_langchain_memory.stores import (
    HistoryResolver,
    InMemorySessionRepository,
    SessionScope,
    StoreFactoryHistoryResolver,
)
from med_langchain_memory.stores.base import MedChatMessageHistory

#: 测试用命名空间坐标与查询参数。
SCOPE = SessionScope(tenant_id="hosp-a", dept_id="cardio")
SCOPE_PARAMS: dict[str, str] = {"tenant_id": "hosp-a", "dept_id": "cardio"}

#: 端点路径。
STATS_URL = "/admin/stats"
MIGRATE_URL = "/admin/sessions/s-1/migrate"
SNAPSHOT_URL = "/admin/sessions/s-1/snapshot"

#: 测试替身认可的「已注册后端」。
KNOWN_BACKENDS: tuple[str, ...] = ("memory", "redis")


class _IsolatedHistory(MedChatMessageHistory):
    """实例级内存历史。

    :class:`~med_langchain_memory.stores.memory_store.InMemoryMedHistory` 按类级字典
    共享数据，用它同时充当「源 / 目标」会让两侧看到同一份消息，无法断言迁移真的
    写到了目标后端，因此这里改用实例级列表。
    """

    def __init__(
        self,
        session_id: str,
        tenant_id: str,
        dept_id: str,
        patient_id: str,
        *,
        ttl_seconds: int | None = None,
    ) -> None:
        """初始化空历史（不共享任何类级状态）。"""
        super().__init__(session_id, tenant_id, dept_id, patient_id, ttl_seconds=ttl_seconds)
        self._items: list[MedMessage] = []

    def _append(self, messages: list[MedMessage]) -> None:
        """按时序插入消息。"""
        self._items.extend(messages)
        self._items.sort(key=lambda message: (message.created_at, message.message_id))

    def _read(self, limit: int | None = None) -> list[MedMessage]:
        """按时序读取消息；``limit`` 为最近 N 条。"""
        if limit is None:
            return list(self._items)
        return list(self._items[-limit:])

    def clear(self) -> None:
        """清空实例级消息列表。"""
        self._items.clear()


class _FakeResolver(HistoryResolver):
    """按 ``(后端名, 存储键)`` 缓存实例级历史的测试替身。

    未登记的后端名与 :class:`~med_langchain_memory.stores.factory.StoreFactory`
    行为一致，抛 :class:`StoreNotFoundError`。
    """

    def __init__(self, known: tuple[str, ...] = KNOWN_BACKENDS) -> None:
        """初始化替身，``known`` 为认可的已注册后端名。"""
        self._known = known
        self.calls: list[tuple[str, str]] = []
        self.histories: dict[tuple[str, str], _IsolatedHistory] = {}

    def resolve(
        self,
        backend: str,
        *,
        scope: SessionScope,
        session_id: str,
        patient_id: str,
        ttl_seconds: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> MedChatMessageHistory:
        """返回（必要时新建）指定后端 + 存储键的历史句柄。"""
        if backend not in self._known:
            raise StoreNotFoundError(f"unknown store backend: {backend!r}")
        key = (backend, scope.storage_key(session_id))
        self.calls.append(key)
        if key not in self.histories:
            self.histories[key] = _IsolatedHistory(
                session_id=session_id,
                tenant_id=scope.tenant_id,
                dept_id=scope.dept_id,
                patient_id=patient_id,
                ttl_seconds=ttl_seconds,
            )
        return self.histories[key]

    def seed(self, backend: str, session_id: str, count: int) -> _IsolatedHistory:
        """向指定后端预置 ``count`` 条时序递增的消息。"""
        history = self.resolve(backend, scope=SCOPE, session_id=session_id, patient_id="p-1")
        history.add_med_messages(
            [
                MedMessage(
                    session_id=session_id,
                    tenant_id=SCOPE.tenant_id,
                    dept_id=SCOPE.dept_id,
                    patient_id="p-1",
                    role=MessageRole.PATIENT,
                    content=f"msg-{index}",
                    created_at=1000 + index,
                )
                for index in range(count)
            ]
        )
        return history


def _client(
    resolver: HistoryResolver,
    repository: InMemorySessionRepository | None = None,
) -> TestClient:
    """构造注入了会话仓储与历史解析器的测试客户端。"""
    return TestClient(
        create_app(
            MedMemorySettings(),
            session_repository=repository
            if repository is not None
            else InMemorySessionRepository(),
            history_resolver=resolver,
        )
    )


def _meta(
    session_id: str = "s-1",
    *,
    status: SessionStatus = SessionStatus.ACTIVE,
    message_count: int = 0,
    tenant_id: str = "hosp-a",
    dept_id: str = "cardio",
    created_at: int = 1000,
) -> SessionMeta:
    """构造可控状态与计数的会话元数据。"""
    return SessionMeta(
        session_id=session_id,
        tenant_id=tenant_id,
        dept_id=dept_id,
        patient_id="p-1",
        status=status,
        message_count=message_count,
        created_at=created_at,
    )


def _repo_with(*metas: SessionMeta) -> InMemorySessionRepository:
    """构造并填充会话仓储。"""
    repository = InMemorySessionRepository()
    for meta in metas:
        repository.add(meta)
    return repository


def _migrate_body(**overrides: Any) -> dict[str, Any]:
    """构造迁移请求体（缺省为 memory → redis 全量迁移）。"""
    body: dict[str, Any] = {"source_backend": "memory", "target_backend": "redis"}
    body.update(overrides)
    return body


# --------------------------------------------------------------------------- #
# GET /admin/stats
# --------------------------------------------------------------------------- #
def test_stats_empty_namespace_returns_zeroed_breakdown() -> None:
    """无会话时四种状态计数均为 0（边界：空命名空间）。"""
    response = _client(_FakeResolver()).get(STATS_URL, params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json() == {
        "tenant_id": "hosp-a",
        "dept_id": "cardio",
        "total_sessions": 0,
        "total_messages": 0,
        "by_status": {"active": 0, "closed": 0, "archived": 0, "deleted": 0},
    }


def test_stats_counts_sessions_and_messages_by_status() -> None:
    """按状态分组计数、累计消息数，且不含其他租户会话。"""
    repository = _repo_with(
        _meta("s-1", message_count=3, created_at=1000),
        _meta("s-2", status=SessionStatus.CLOSED, message_count=2, created_at=2000),
        _meta("s-3", status=SessionStatus.ARCHIVED, created_at=3000),
        _meta("s-4", status=SessionStatus.DELETED, created_at=4000),
        _meta("s-9", tenant_id="hosp-b", message_count=99, created_at=500),
    )
    body = _client(_FakeResolver(), repository).get(STATS_URL, params=SCOPE_PARAMS).json()
    assert body["total_sessions"] == 4
    assert body["total_messages"] == 5
    assert body["by_status"] == {"active": 1, "closed": 1, "archived": 1, "deleted": 1}


def test_stats_pages_through_beyond_single_page_limit() -> None:
    """会话数超过仓储单页上限（200）时统计不被截断（边界：翻页）。"""
    repository = _repo_with(
        *[_meta(f"s-{index:03d}", created_at=1000 + index) for index in range(205)]
    )
    body = _client(_FakeResolver(), repository).get(STATS_URL, params=SCOPE_PARAMS).json()
    assert body["total_sessions"] == 205
    assert body["by_status"]["active"] == 205


@pytest.mark.parametrize(
    "params",
    [
        {"dept_id": "cardio"},
        {"tenant_id": "hosp-a"},
        {"tenant_id": "hosp:a", "dept_id": "cardio"},
        {"tenant_id": "", "dept_id": "cardio"},
    ],
)
def test_stats_rejects_invalid_scope(params: dict[str, str]) -> None:
    """命名空间查询参数缺失或非法 → 422（边界：越权入口被前置拦截）。"""
    response = _client(_FakeResolver()).get(STATS_URL, params=params)
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_error"


# --------------------------------------------------------------------------- #
# POST /admin/sessions/{id}/migrate
# --------------------------------------------------------------------------- #
def test_migrate_copies_messages_to_target_backend() -> None:
    """迁移把源后端消息写入目标后端，且校验通过、结果字段自洽。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 2)
    response = _client(resolver, _repo_with(_meta("s-1"))).post(
        MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body()
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == "s-1"
    assert body["session_key"] == "med:chat:hosp-a:cardio:s-1"
    assert body["source_backend"] == "memory"
    assert body["target_backend"] == "redis"
    assert (body["source_total"], body["target_total"]) == (2, 2)
    assert (body["newly_written"], body["skipped"]) == (2, 0)
    assert body["verified"] is True
    assert body["finished"] is True
    assert body["errors"] == []
    target = resolver.histories[("redis", "med:chat:hosp-a:cardio:s-1")]
    assert [message.content for message in target.get_med_messages()] == ["msg-0", "msg-1"]


def test_migrate_is_idempotent_on_rerun() -> None:
    """重复迁移按消息 ID 去重，不产生重复数据（边界：幂等重跑）。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 2)
    client = _client(resolver, _repo_with(_meta("s-1")))
    client.post(MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body())
    body = client.post(MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body()).json()
    assert (body["newly_written"], body["skipped"]) == (0, 2)
    assert body["target_total"] == 2
    assert body["finished"] is True


def test_migrate_writes_all_batches() -> None:
    """``batch_size`` 小于消息数时全部批次均被写入。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 5)
    body = (
        _client(resolver, _repo_with(_meta("s-1")))
        .post(MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body(batch_size=2))
        .json()
    )
    assert body["newly_written"] == 5
    assert body["target_total"] == 5
    assert body["verified"] is True


def test_migrate_without_verify_reports_none() -> None:
    """``verify=false`` 时不做一致性校验，``verified`` 为 ``None``。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 1)
    body = (
        _client(resolver, _repo_with(_meta("s-1")))
        .post(MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body(verify=False))
        .json()
    )
    assert body["verified"] is None
    assert body["newly_written"] == 1


def test_migrate_soft_deleted_session_is_allowed() -> None:
    """已软删除会话仍可迁移（合规导出场景，管理端点不受可见性限制）。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 1)
    response = _client(resolver, _repo_with(_meta("s-1", status=SessionStatus.DELETED))).post(
        MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body()
    )
    assert response.status_code == 200
    assert response.json()["newly_written"] == 1


def test_migrate_missing_session_returns_404() -> None:
    """会话不存在 → 404（边界：不存在的会话 ID）。"""
    response = _client(_FakeResolver()).post(MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body())
    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found_error"


def test_migrate_is_isolated_per_namespace() -> None:
    """其他租户无法迁移本租户会话（越权防护）。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 1)
    response = _client(resolver, _repo_with(_meta("s-1"))).post(
        MIGRATE_URL,
        params={"tenant_id": "hosp-b", "dept_id": "cardio"},
        json=_migrate_body(),
    )
    assert response.status_code == 404


def test_migrate_unknown_backend_returns_404() -> None:
    """未注册的后端名 → 404 ``store_not_found_error``（边界：后端名拼错）。"""
    response = _client(_FakeResolver(), _repo_with(_meta("s-1"))).post(
        MIGRATE_URL, params=SCOPE_PARAMS, json=_migrate_body(target_backend="nope")
    )
    assert response.status_code == 404
    assert response.json()["error"] == "store_not_found_error"


def test_admin_error_response_carries_request_id() -> None:
    """管理端点的错误响应同样透传请求追踪 ID。"""
    response = _client(_FakeResolver()).post(
        MIGRATE_URL,
        params=SCOPE_PARAMS,
        json=_migrate_body(),
        headers={"X-Request-ID": "rid-admin"},
    )
    assert response.status_code == 404
    assert response.json()["request_id"] == "rid-admin"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"source_backend": "memory"},
        {"target_backend": "redis"},
        {"source_backend": "", "target_backend": "redis"},
        {"source_backend": "memory", "target_backend": "redis", "batch_size": 0},
        {
            "source_backend": "memory",
            "target_backend": "redis",
            "batch_size": MAX_MIGRATION_BATCH_SIZE + 1,
        },
        {"source_backend": "memory", "target_backend": "redis", "unknown": 1},
    ],
)
def test_migrate_rejects_invalid_body(payload: dict[str, Any]) -> None:
    """请求体缺字段、后端名为空、批次越界或含未声明字段 → 422。"""
    response = _client(_FakeResolver()).post(MIGRATE_URL, params=SCOPE_PARAMS, json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_error"


# --------------------------------------------------------------------------- #
# POST /admin/sessions/{id}/snapshot
# --------------------------------------------------------------------------- #
def test_snapshot_returns_verifiable_restorable_package(tmp_path: Path) -> None:
    """导出的文件包可解码、校验和自洽，并能被生命周期层恢复回新句柄。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 2)
    response = _client(resolver, _repo_with(_meta("s-1"))).post(
        SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "memory"}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["session_id"] == "s-1"
    assert body["session_key"] == "med:chat:hosp-a:cardio:s-1"
    assert body["backend"] == "memory"
    assert body["schema_version"] == "1"
    assert body["message_count"] == 2

    raw = base64.b64decode(body["payload_base64"])
    assert body["size_bytes"] == len(raw)
    assert body["sha256"] == hashlib.sha256(raw).hexdigest()

    package = SessionSnapshotPackage.from_bytes(raw)
    assert package.schema_version == "1"

    path = tmp_path / "s-1.medsnap"
    path.write_bytes(raw)
    restored = _IsolatedHistory(
        session_id="s-1", tenant_id="hosp-a", dept_id="cardio", patient_id="p-1"
    )
    summary = SessionSnapshotter().import_session(path, restored)
    assert summary.verified is True
    assert [message.content for message in restored.get_med_messages()] == ["msg-0", "msg-1"]


def test_snapshot_detects_tampered_package() -> None:
    """篡改文件包任一字节后校验和不匹配（边界：文件损坏 / 被改写）。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 1)
    body = (
        _client(resolver, _repo_with(_meta("s-1")))
        .post(SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "memory"})
        .json()
    )
    tampered = bytearray(base64.b64decode(body["payload_base64"]))
    tampered[-1] ^= 0xFF
    with pytest.raises(IntegrityError):
        SessionSnapshotPackage.from_bytes(bytes(tampered))


def test_snapshot_empty_session_returns_minimal_package() -> None:
    """无消息的会话也能导出（边界：空会话），字节数仍大于 0。"""
    body = (
        _client(_FakeResolver(), _repo_with(_meta("s-1")))
        .post(SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "memory"})
        .json()
    )
    assert body["message_count"] == 0
    assert body["size_bytes"] > 0


def test_snapshot_honours_schema_version() -> None:
    """``schema_version`` 由请求指定并回显。"""
    body = (
        _client(_FakeResolver(), _repo_with(_meta("s-1")))
        .post(SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "memory", "schema_version": "9"})
        .json()
    )
    assert body["schema_version"] == "9"
    assert (
        SessionSnapshotPackage.from_bytes(base64.b64decode(body["payload_base64"])).schema_version
        == "9"
    )


def test_snapshot_soft_deleted_session_is_allowed() -> None:
    """已软删除会话仍可导出（合规留存场景）。"""
    resolver = _FakeResolver()
    resolver.seed("memory", "s-1", 1)
    response = _client(resolver, _repo_with(_meta("s-1", status=SessionStatus.DELETED))).post(
        SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "memory"}
    )
    assert response.status_code == 200
    assert response.json()["message_count"] == 1


def test_snapshot_missing_session_returns_404() -> None:
    """会话不存在 → 404。"""
    response = _client(_FakeResolver()).post(
        SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "memory"}
    )
    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found_error"


def test_snapshot_unknown_backend_returns_404() -> None:
    """未注册的后端名 → 404。"""
    response = _client(_FakeResolver(), _repo_with(_meta("s-1"))).post(
        SNAPSHOT_URL, params=SCOPE_PARAMS, json={"backend": "nope"}
    )
    assert response.status_code == 404
    assert response.json()["error"] == "store_not_found_error"


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"backend": ""},
        {"backend": "memory", "schema_version": ""},
        {"backend": "memory", "unknown": 1},
    ],
)
def test_snapshot_rejects_invalid_body(payload: dict[str, Any]) -> None:
    """请求体缺字段、后端名 / 版本为空或含未声明字段 → 422。"""
    response = _client(_FakeResolver()).post(SNAPSHOT_URL, params=SCOPE_PARAMS, json=payload)
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# DTO 契约
# --------------------------------------------------------------------------- #
def test_migration_response_from_result_maps_fields() -> None:
    """迁移结果 → 响应 DTO 字段完整映射（含错误明细）。"""
    result = MigrationResult(
        session_key="med:chat:hosp-a:cardio:s-1",
        source_total=3,
        target_total=3,
        newly_written=2,
        skipped=1,
        verified=True,
        finished=True,
        errors=["batch write failed: boom"],
    )
    dto = MigrationResponse.from_result(
        result, session_id="s-1", source_backend="memory", target_backend="redis"
    )
    assert dto.session_id == "s-1"
    assert dto.session_key == "med:chat:hosp-a:cardio:s-1"
    assert (dto.source_total, dto.target_total) == (3, 3)
    assert (dto.newly_written, dto.skipped) == (2, 1)
    assert dto.verified is True
    assert dto.finished is True
    assert dto.errors == ["batch write failed: boom"]


def test_migration_request_defaults() -> None:
    """迁移请求体缺省批次 100、开启校验、无额外构造选项。"""
    request = MigrationRequest(source_backend="memory", target_backend="redis")
    assert request.batch_size == 100
    assert request.verify is True
    assert request.source_options == {}
    assert request.target_options == {}


def test_migration_response_is_frozen() -> None:
    """迁移响应 DTO 不可变（边界：赋值被拒绝）。"""
    dto = MigrationResponse.from_result(
        MigrationResult(session_key="k", finished=True),
        session_id="s-1",
        source_backend="memory",
        target_backend="redis",
    )
    with pytest.raises(PydanticValidationError):
        dto.newly_written = 1  # type: ignore[misc]


def test_snapshot_response_rejects_short_sha256() -> None:
    """快照响应 DTO 拒绝非法长度的 SHA-256（边界：摘要被截断）。"""
    with pytest.raises(PydanticValidationError):
        SnapshotResponse(
            session_id="s-1",
            session_key="k",
            backend="memory",
            schema_version="1",
            message_count=0,
            size_bytes=1,
            sha256="abc",
            payload_base64="AA==",
        )


def test_admin_stats_response_rejects_negative_total() -> None:
    """统计响应 DTO 拒绝负数计数（边界：非法统计值）。"""
    with pytest.raises(PydanticValidationError):
        AdminStatsResponse(
            tenant_id="hosp-a", dept_id="cardio", total_sessions=-1, total_messages=0
        )


# --------------------------------------------------------------------------- #
# 依赖注入与对外契约
# --------------------------------------------------------------------------- #
def _request_with_app(app: object) -> Request:
    """构造仅带 ``app`` 的最小 ASGI 请求对象。"""
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": app})


def test_get_history_resolver_returns_injected_instance() -> None:
    """``app.state.history_resolver`` 中的解析器被原样返回。"""
    resolver = _FakeResolver()
    app = SimpleNamespace(state=SimpleNamespace(history_resolver=resolver))
    assert get_history_resolver(_request_with_app(app)) is resolver


def test_get_history_resolver_without_injection_raises() -> None:
    """未注入解析器（或类型不符）时抛 :class:`RuntimeError`（边界：装配缺失）。"""
    app = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="not configured"):
        get_history_resolver(_request_with_app(app))


def test_create_app_defaults_to_store_factory_resolver() -> None:
    """应用工厂缺省注入基于存储工厂的解析器，管理端点开箱可用。"""
    app = create_app(MedMemorySettings())
    assert isinstance(app.state.history_resolver, StoreFactoryHistoryResolver)


def test_openapi_exposes_admin_endpoints() -> None:
    """OpenAPI 文档包含全部管理端点（对外契约可发现）。"""
    schema = _client(_FakeResolver()).get("/openapi.json").json()
    assert set(schema["paths"]["/admin/stats"]) == {"get"}
    assert set(schema["paths"]["/admin/sessions/{session_id}/migrate"]) == {"post"}
    assert set(schema["paths"]["/admin/sessions/{session_id}/snapshot"]) == {"post"}
