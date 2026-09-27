"""会话接口单元测试（D31）。

覆盖六个端点的正向路径与边界/异常路径，以及 DTO 契约与依赖注入行为：

* ``POST   /sessions`` —— 生成 ID / 显式 ID / 重复创建 409 / 字段校验 422；
* ``GET    /sessions`` —— 空列表、排序、状态过滤、分页、非法分页 422；
* ``GET    /sessions/{id}`` —— 命中、404、跨租户不可见；
* ``POST   /sessions/{id}/close|archive``、``DELETE /sessions/{id}`` —— 合法流转与非法流转 409；
* :class:`SessionScope` / :class:`SessionResponse` 的 DTO 契约与 :func:`get_session_repository` 的兜底分支。
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError
from starlette.requests import Request

from med_langchain_memory.api import (
    SessionCreateRequest,
    SessionListResponse,
    SessionResponse,
    create_app,
    get_session_repository,
)
from med_langchain_memory.config import MedMemorySettings
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.stores.session_repository import (
    InMemorySessionRepository,
    SessionRepository,
)

#: 默认命名空间查询参数。
SCOPE_PARAMS: dict[str, str] = {"tenant_id": "hosp-a", "dept_id": "cardio"}


def _client(repository: SessionRepository | None = None) -> TestClient:
    """构造注入指定仓储（缺省为新建内存仓储）的测试客户端。"""
    return TestClient(create_app(MedMemorySettings(), session_repository=repository))


def _create(client: TestClient, **overrides: Any) -> dict[str, Any]:
    """调用创建端点并返回响应体，便于后续断言。"""
    payload: dict[str, Any] = {"patient_id": "p-1"}
    payload.update(overrides)
    response = client.post("/sessions", params=SCOPE_PARAMS, json=payload)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _meta(
    session_id: str = "s-1",
    *,
    tenant_id: str = "hosp-a",
    dept_id: str = "cardio",
    status: SessionStatus = SessionStatus.ACTIVE,
    created_at: int = 1000,
) -> SessionMeta:
    """构造可控时间戳的会话元数据。"""
    return SessionMeta(
        session_id=session_id,
        tenant_id=tenant_id,
        dept_id=dept_id,
        patient_id="p-1",
        status=status,
        created_at=created_at,
    )


# --------------------------------------------------------------------------- #
# POST /sessions
# --------------------------------------------------------------------------- #
def test_create_generates_session_id() -> None:
    """未指定 session_id 时由服务端生成合法 UUID，初始状态为 ACTIVE。"""
    body = _create(_client())
    uuid.UUID(body["session_id"])
    assert body["status"] == "active"
    assert body["message_count"] == 0
    assert body["tenant_id"] == "hosp-a"
    assert body["dept_id"] == "cardio"
    assert body["created_at"] > 0
    assert body["updated_at"] >= body["created_at"]


def test_create_with_explicit_id_and_metadata() -> None:
    """显式 session_id 与 metadata 被原样保留。"""
    body = _create(_client(), session_id="s-2026", metadata={"visit_type": "first"})
    assert body["session_id"] == "s-2026"
    assert body["metadata"] == {"visit_type": "first"}


def test_create_duplicate_returns_409() -> None:
    """同一命名空间重复创建 → 409 ``integrity_error``（边界：主键冲突）。"""
    client = _client()
    _create(client, session_id="s-dup")
    response = client.post(
        "/sessions", params=SCOPE_PARAMS, json={"patient_id": "p-1", "session_id": "s-dup"}
    )
    assert response.status_code == 409
    assert response.json()["error"] == "integrity_error"


def test_create_same_id_in_other_tenant_is_allowed() -> None:
    """不同租户下同名 session_id 互不冲突（命名空间隔离）。"""
    client = _client()
    _create(client, session_id="s-1")
    response = client.post(
        "/sessions",
        params={"tenant_id": "hosp-b", "dept_id": "cardio"},
        json={"patient_id": "p-2", "session_id": "s-1"},
    )
    assert response.status_code == 201


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"patient_id": ""},
        {"patient_id": "p:1"},
        {"patient_id": "p-1", "session_id": "bad id"},
        {"patient_id": "p-1", "unknown_field": "x"},
    ],
)
def test_create_rejects_invalid_body(payload: dict[str, Any]) -> None:
    """请求体字段缺失、非法字符或含未声明字段 → 422。"""
    response = _client().post("/sessions", params=SCOPE_PARAMS, json=payload)
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_error"


@pytest.mark.parametrize(
    "params",
    [
        {"dept_id": "cardio"},
        {"tenant_id": "hosp-a"},
        {"tenant_id": "", "dept_id": "cardio"},
        {"tenant_id": "hosp:a", "dept_id": "cardio"},
    ],
)
def test_create_requires_valid_scope_params(params: dict[str, str]) -> None:
    """缺少或非法的命名空间查询参数 → 422（边界：越权入口被前置拦截）。"""
    response = _client().post("/sessions", params=params, json={"patient_id": "p-1"})
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_error"


# --------------------------------------------------------------------------- #
# GET /sessions
# --------------------------------------------------------------------------- #
def test_list_empty_returns_zero_total() -> None:
    """无会话时返回空列表与 total 0（边界：空仓储）。"""
    response = _client().get("/sessions", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json() == {"total": 0, "limit": 50, "offset": 0, "items": []}


def test_list_returns_sessions_in_creation_order() -> None:
    """列表按创建时间升序返回本命名空间会话。"""
    repository = InMemorySessionRepository()
    repository.add(_meta("s-2", created_at=2000))
    repository.add(_meta("s-1", created_at=1000))
    repository.add(_meta("s-9", tenant_id="hosp-b", created_at=500))
    body = _client(repository).get("/sessions", params=SCOPE_PARAMS).json()
    assert body["total"] == 2
    assert [item["session_id"] for item in body["items"]] == ["s-1", "s-2"]


def test_list_filters_by_status() -> None:
    """``status`` 查询参数生效。"""
    repository = InMemorySessionRepository()
    repository.add(_meta("s-1", status=SessionStatus.ACTIVE))
    repository.add(_meta("s-2", status=SessionStatus.ARCHIVED, created_at=2000))
    body = (
        _client(repository).get("/sessions", params={**SCOPE_PARAMS, "status": "archived"}).json()
    )
    assert body["total"] == 1
    assert body["items"][0]["session_id"] == "s-2"


def test_list_pagination() -> None:
    """``limit`` / ``offset`` 生效且 ``total`` 为命中总数。"""
    repository = InMemorySessionRepository()
    for index in range(3):
        repository.add(_meta(f"s-{index}", created_at=1000 + index))
    body = (
        _client(repository)
        .get("/sessions", params={**SCOPE_PARAMS, "limit": 1, "offset": 1})
        .json()
    )
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert body["total"] == 3
    assert [item["session_id"] for item in body["items"]] == ["s-1"]


@pytest.mark.parametrize(
    "params", [{"limit": 0}, {"limit": 201}, {"offset": -1}, {"status": "nope"}]
)
def test_list_rejects_invalid_query(params: dict[str, Any]) -> None:
    """非法分页或未知状态值 → 422。"""
    response = _client().get("/sessions", params={**SCOPE_PARAMS, **params})
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# GET /sessions/{session_id}
# --------------------------------------------------------------------------- #
def test_get_session_returns_detail() -> None:
    """按 ID 查询返回会话详情。"""
    client = _client()
    created = _create(client, session_id="s-1")
    response = client.get("/sessions/s-1", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json() == created


def test_get_session_missing_returns_404() -> None:
    """会话不存在 → 404 ``session_not_found_error``（边界：不存在的 ID）。"""
    response = _client().get("/sessions/s-none", params=SCOPE_PARAMS)
    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found_error"


def test_get_session_hidden_from_other_tenant() -> None:
    """跨租户查询同一 session_id 返回 404 而非泄漏数据（越权防护）。"""
    client = _client()
    _create(client, session_id="s-1")
    response = client.get("/sessions/s-1", params={"tenant_id": "hosp-b", "dept_id": "cardio"})
    assert response.status_code == 404


def test_get_session_rejects_malformed_id() -> None:
    """路径参数含非法字符 → 422（边界：脏路径）。"""
    response = _client().get("/sessions/bad:id", params=SCOPE_PARAMS)
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# 状态流转端点
# --------------------------------------------------------------------------- #
def test_close_active_session() -> None:
    """ACTIVE → CLOSED 返回 200 且状态更新。"""
    client = _client()
    _create(client, session_id="s-1")
    response = client.post("/sessions/s-1/close", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json()["status"] == "closed"


def test_close_twice_returns_409() -> None:
    """重复关闭 → 409 ``state_transition_error``（边界：非法流转）。"""
    client = _client()
    _create(client, session_id="s-1")
    client.post("/sessions/s-1/close", params=SCOPE_PARAMS)
    response = client.post("/sessions/s-1/close", params=SCOPE_PARAMS)
    assert response.status_code == 409
    assert response.json()["error"] == "state_transition_error"


def test_close_missing_session_returns_404() -> None:
    """对不存在的会话执行流转 → 404。"""
    response = _client().post("/sessions/s-none/close", params=SCOPE_PARAMS)
    assert response.status_code == 404


def test_archive_active_session() -> None:
    """ACTIVE → ARCHIVED 允许直接归档。"""
    client = _client()
    _create(client, session_id="s-1")
    response = client.post("/sessions/s-1/archive", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json()["status"] == "archived"


def test_archive_closed_session() -> None:
    """CLOSED → ARCHIVED 为合法路径。"""
    client = _client()
    _create(client, session_id="s-1")
    client.post("/sessions/s-1/close", params=SCOPE_PARAMS)
    response = client.post("/sessions/s-1/archive", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json()["status"] == "archived"


def test_archive_twice_returns_409() -> None:
    """已归档会话再次归档 → 409。"""
    client = _client()
    _create(client, session_id="s-1")
    client.post("/sessions/s-1/archive", params=SCOPE_PARAMS)
    response = client.post("/sessions/s-1/archive", params=SCOPE_PARAMS)
    assert response.status_code == 409


def test_delete_archived_session_soft_deletes() -> None:
    """ARCHIVED → DELETED 软删除成功。"""
    client = _client()
    _create(client, session_id="s-1")
    client.post("/sessions/s-1/archive", params=SCOPE_PARAMS)
    response = client.delete("/sessions/s-1", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"


def test_delete_active_session_returns_409() -> None:
    """未归档的会话不允许直接软删除（边界：跳过归档阶段）。"""
    client = _client()
    _create(client, session_id="s-1")
    response = client.delete("/sessions/s-1", params=SCOPE_PARAMS)
    assert response.status_code == 409
    assert response.json()["error"] == "state_transition_error"


def test_transitions_are_isolated_per_namespace() -> None:
    """其他租户无法流转本租户会话（越权防护）。"""
    client = _client()
    _create(client, session_id="s-1")
    response = client.post(
        "/sessions/s-1/close", params={"tenant_id": "hosp-b", "dept_id": "cardio"}
    )
    assert response.status_code == 404


def test_error_response_carries_request_id() -> None:
    """错误响应透传请求追踪 ID（跨中间件与异常处理器）。"""
    response = _client().get(
        "/sessions/s-none", params=SCOPE_PARAMS, headers={"X-Request-ID": "rid-123"}
    )
    assert response.status_code == 404
    assert response.json()["request_id"] == "rid-123"
    assert response.headers["X-Request-ID"] == "rid-123"


# --------------------------------------------------------------------------- #
# DTO 契约
# --------------------------------------------------------------------------- #
def test_session_response_from_meta() -> None:
    """领域模型 → 响应 DTO 字段完整映射。"""
    dto = SessionResponse.from_meta(_meta("s-1"))
    assert dto.session_id == "s-1"
    assert dto.tenant_id == "hosp-a"
    assert dto.status is SessionStatus.ACTIVE
    assert dto.message_count == 0


def test_session_response_is_frozen() -> None:
    """响应 DTO 不可变（边界：赋值被拒绝）。"""
    dto = SessionResponse.from_meta(_meta())
    with pytest.raises(PydanticValidationError):
        dto.session_id = "s-2"  # type: ignore[misc]


def test_session_response_rejects_unknown_field() -> None:
    """响应 DTO 拒绝未声明字段。"""
    with pytest.raises(PydanticValidationError):
        SessionResponse(**{**SessionResponse.from_meta(_meta()).model_dump(), "extra": 1})


def test_session_create_request_requires_patient_id() -> None:
    """创建请求体缺少 patient_id 即校验失败（边界：必填字段缺失）。"""
    with pytest.raises(PydanticValidationError):
        SessionCreateRequest.model_validate({})


def test_session_list_response_rejects_negative_total() -> None:
    """列表 DTO 拒绝负数 total（边界：非法计数）。"""
    with pytest.raises(PydanticValidationError):
        SessionListResponse(total=-1, limit=50, offset=0, items=[])


# --------------------------------------------------------------------------- #
# 依赖注入
# --------------------------------------------------------------------------- #
def _request_with_app(app: object) -> Request:
    """构造仅带 ``app`` 的最小 ASGI 请求对象。"""
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": app})


def test_get_session_repository_returns_injected_instance() -> None:
    """``app.state.session_repository`` 中的仓储被原样返回。"""
    repository = InMemorySessionRepository()
    app = SimpleNamespace(state=SimpleNamespace(session_repository=repository))
    assert get_session_repository(_request_with_app(app)) is repository


def test_get_session_repository_without_injection_raises() -> None:
    """未注入仓储（或类型不符）时抛 :class:`RuntimeError`（边界：装配缺失）。"""
    app = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="not configured"):
        get_session_repository(_request_with_app(app))


def test_create_app_defaults_to_in_memory_repository() -> None:
    """应用工厂缺省注入进程内仓储，会话端点开箱可用。"""
    app = create_app(MedMemorySettings())
    assert isinstance(app.state.session_repository, InMemorySessionRepository)


def test_openapi_exposes_session_endpoints() -> None:
    """OpenAPI 文档包含全部会话端点（对外契约可发现）。"""
    schema = _client().get("/openapi.json").json()
    assert "/sessions" in schema["paths"]
    assert set(schema["paths"]["/sessions"]) == {"get", "post"}
    assert set(schema["paths"]["/sessions/{session_id}"]) == {"get", "delete"}
    assert "/sessions/{session_id}/close" in schema["paths"]
    assert "/sessions/{session_id}/archive" in schema["paths"]
