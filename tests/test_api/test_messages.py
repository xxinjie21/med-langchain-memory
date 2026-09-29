"""消息接口单元测试（D32）。

覆盖两个端点的正向路径与边界/异常路径，以及脱敏开关、游标分页与依赖注入行为：

* ``POST /sessions/{id}/messages`` —— 单条/批量追加、时序保序、消息计数同步、
  显式 ID 与时间戳、会话不存在 404、非 ACTIVE 状态 409、批量条数/字段校验 422、越权 404；
* ``GET  /sessions/{id}/messages`` —— 空列表、默认脱敏、``mask=false`` 取原文、
  按租户策略差异化脱敏、游标翻页与重放、翻页期间新增消息不重复、非法游标 400、
  非法分页 422、不存在/已软删除会话 404、越权 404；
* DTO 契约与 :func:`get_message_repository` / :func:`get_message_masker` 的兜底分支。
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
    MAX_APPEND_BATCH,
    MessageAppendRequest,
    MessageCreate,
    MessageListResponse,
    MessageResponse,
    create_app,
    get_message_masker,
    get_message_repository,
)
from med_langchain_memory.config import MedMemorySettings
from med_langchain_memory.domain.message import MedMessage, MessageRole
from med_langchain_memory.privacy.policies import PolicyMasker
from med_langchain_memory.stores.message_repository import InMemoryMessageRepository
from med_langchain_memory.stores.session_repository import InMemorySessionRepository

#: 默认命名空间查询参数。
SCOPE_PARAMS: dict[str, str] = {"tenant_id": "hosp-a", "dept_id": "cardio"}

#: 其他租户命名空间查询参数（用于越权用例）。
OTHER_PARAMS: dict[str, str] = {"tenant_id": "hosp-b", "dept_id": "cardio"}

#: 含手机号与身份证号的正文（用于脱敏断言）。
SENSITIVE_TEXT = "患者电话 13812345678 身份证 110101199003071234"


def _client(
    *,
    sessions: InMemorySessionRepository | None = None,
    messages: InMemoryMessageRepository | None = None,
    masker: PolicyMasker | None = None,
) -> TestClient:
    """构造注入了指定后端（缺省为新建内存实现）的测试客户端。"""
    return TestClient(
        create_app(
            MedMemorySettings(),
            session_repository=sessions,
            message_repository=messages,
            masker=masker,
        )
    )


def _open_session(client: TestClient, session_id: str = "s-1", **params: str) -> None:
    """创建一个 ACTIVE 会话（缺省 ID ``s-1``）。"""
    response = client.post(
        "/sessions",
        params={**SCOPE_PARAMS, **params},
        json={"patient_id": "p-1", "session_id": session_id},
    )
    assert response.status_code == 201, response.text


def _append(
    client: TestClient,
    messages: list[dict[str, Any]],
    *,
    session_id: str = "s-1",
    params: dict[str, str] | None = None,
) -> dict[str, Any]:
    """调用追加端点并断言成功，返回响应体。"""
    response = client.post(
        f"/sessions/{session_id}/messages",
        params=params if params is not None else SCOPE_PARAMS,
        json={"messages": messages},
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _message_body(index: int, *, created_at: int | None = None, **extra: Any) -> dict[str, Any]:
    """构造一条追加请求体。"""
    payload: dict[str, Any] = {"role": "patient", "content": f"msg-{index}"}
    if created_at is not None:
        payload["created_at"] = created_at
    payload.update(extra)
    return payload


def _med_message(index: int, *, created_at: int | None = None) -> MedMessage:
    """构造领域消息（用于 DTO 契约用例）。"""
    return MedMessage(
        session_id="s-1",
        tenant_id="hosp-a",
        dept_id="cardio",
        patient_id="p-1",
        role=MessageRole.DOCTOR,
        content=f"msg-{index}",
        token_count=3,
        created_at=created_at if created_at is not None else 1000 + index,
    )


# --------------------------------------------------------------------------- #
# POST /sessions/{id}/messages
# --------------------------------------------------------------------------- #
def test_append_single_message() -> None:
    """追加单条消息返回 201，回显内容与角色，初始 ``masked`` 为假。"""
    client = _client()
    _open_session(client)
    body = _append(client, [_message_body(0)])
    assert body["session_id"] == "s-1"
    assert body["appended"] == 1
    assert body["message_count"] == 1
    assert body["items"][0]["content"] == "msg-0"
    assert body["items"][0]["role"] == "patient"
    assert body["items"][0]["masked"] is False
    assert body["items"][0]["patient_id"] == "p-1"


def test_append_batch_returns_messages_in_time_order() -> None:
    """批量追加按 ``(created_at, message_id)`` 升序回显。"""
    client = _client()
    _open_session(client)
    body = _append(
        client,
        [
            _message_body(2, created_at=3000),
            _message_body(0, created_at=1000),
            _message_body(1, created_at=2000),
        ],
    )
    assert body["appended"] == 3
    assert [item["content"] for item in body["items"]] == ["msg-0", "msg-1", "msg-2"]


def test_append_syncs_session_message_count() -> None:
    """追加后会话详情中的 ``message_count`` 同步累加。"""
    client = _client()
    _open_session(client)
    _append(client, [_message_body(0), _message_body(1)])
    _append(client, [_message_body(2)])
    detail = client.get("/sessions/s-1", params=SCOPE_PARAMS).json()
    assert detail["message_count"] == 3


def test_append_keeps_explicit_id_and_timestamp() -> None:
    """调用方显式给出的 ``message_id`` 与 ``created_at`` 被原样保留（导入/回放场景）。"""
    client = _client()
    _open_session(client)
    message_id = str(uuid.uuid4())
    body = _append(client, [_message_body(0, created_at=777, message_id=message_id)])
    assert body["items"][0]["message_id"] == message_id
    assert body["items"][0]["created_at"] == 777


def test_append_with_explicit_null_message_id_generates_id() -> None:
    """显式传 ``message_id: null`` 等同于缺省，由服务端生成 UUID（边界：显式 null）。"""
    client = _client()
    _open_session(client)
    body = _append(client, [{"role": "patient", "content": "hi", "message_id": None}])
    uuid.UUID(body["items"][0]["message_id"])


def test_append_preserves_metadata_and_token_count() -> None:
    """``metadata`` 与 ``token_count`` 被持久化并回显。"""
    client = _client()
    _open_session(client)
    body = _append(client, [_message_body(0, metadata={"stage": "triage"}, token_count=12)])
    assert body["items"][0]["metadata"] == {"stage": "triage"}
    assert body["items"][0]["token_count"] == 12


def test_append_ignores_caller_supplied_patient_id() -> None:
    """患者 ID 取自会话元数据，调用方无法在消息体里伪造（越权防护）。"""
    client = _client()
    _open_session(client)
    response = client.post(
        "/sessions/s-1/messages",
        params=SCOPE_PARAMS,
        json={"messages": [{"role": "patient", "content": "hi", "patient_id": "p-evil"}]},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_error"


def test_append_to_missing_session_returns_404() -> None:
    """向不存在的会话追加 → 404 ``session_not_found_error``。"""
    response = _client().post(
        "/sessions/s-none/messages", params=SCOPE_PARAMS, json={"messages": [_message_body(0)]}
    )
    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found_error"


@pytest.mark.parametrize("transition", ["close", "archive"])
def test_append_to_inactive_session_returns_409(transition: str) -> None:
    """会话非 ACTIVE（已关闭/已归档）时拒绝写入 → 409（边界：状态守卫）。"""
    client = _client()
    _open_session(client)
    assert client.post(f"/sessions/s-1/{transition}", params=SCOPE_PARAMS).status_code == 200
    response = client.post(
        "/sessions/s-1/messages", params=SCOPE_PARAMS, json={"messages": [_message_body(0)]}
    )
    assert response.status_code == 409
    assert response.json()["error"] == "session_not_active_error"


def test_append_from_other_tenant_returns_404() -> None:
    """其他租户无法向本租户会话追加消息（越权防护）。"""
    client = _client()
    _open_session(client)
    response = client.post(
        "/sessions/s-1/messages", params=OTHER_PARAMS, json={"messages": [_message_body(0)]}
    )
    assert response.status_code == 404


def test_append_rejects_empty_batch() -> None:
    """空批次 → 422（边界：空写入）。"""
    client = _client()
    _open_session(client)
    response = client.post("/sessions/s-1/messages", params=SCOPE_PARAMS, json={"messages": []})
    assert response.status_code == 422


def test_append_rejects_oversized_batch() -> None:
    """超过单次批量上限 → 422（边界：批次过大）。"""
    client = _client()
    _open_session(client)
    response = client.post(
        "/sessions/s-1/messages",
        params=SCOPE_PARAMS,
        json={"messages": [_message_body(index) for index in range(MAX_APPEND_BATCH + 1)]},
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "message",
    [
        {"role": "nurse", "content": "hi"},
        {"role": "patient", "content": ""},
        {"role": "patient"},
        {"content": "hi"},
        {"role": "patient", "content": "hi", "message_id": "not-a-uuid"},
        {"role": "patient", "content": "hi", "created_at": 0},
        {"role": "patient", "content": "hi", "token_count": -1},
    ],
)
def test_append_rejects_invalid_message(message: dict[str, Any]) -> None:
    """角色非法、正文为空、字段缺失、ID 非 UUID、时间戳非正、token 负数 → 422。"""
    client = _client()
    _open_session(client)
    response = client.post(
        "/sessions/s-1/messages", params=SCOPE_PARAMS, json={"messages": [message]}
    )
    assert response.status_code == 422
    assert response.json()["error"] == "request_validation_error"


def test_append_rejects_unknown_top_level_field() -> None:
    """请求体含未声明字段 → 422（边界：拼写错误被显式拒绝）。"""
    client = _client()
    _open_session(client)
    response = client.post(
        "/sessions/s-1/messages",
        params=SCOPE_PARAMS,
        json={"messages": [_message_body(0)], "tenant_id": "hosp-b"},
    )
    assert response.status_code == 422


def test_append_rejects_malformed_session_id() -> None:
    """路径参数含非法字符 → 422（边界：脏路径）。"""
    response = _client().post(
        "/sessions/bad:id/messages", params=SCOPE_PARAMS, json={"messages": [_message_body(0)]}
    )
    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# GET /sessions/{id}/messages
# --------------------------------------------------------------------------- #
def test_list_empty_session_returns_empty_page() -> None:
    """无消息时返回空页、无下一页游标、默认开启脱敏（边界：空会话）。"""
    client = _client()
    _open_session(client)
    response = client.get("/sessions/s-1/messages", params=SCOPE_PARAMS)
    assert response.status_code == 200
    assert response.json() == {
        "session_id": "s-1",
        "limit": 50,
        "masked": True,
        "next_cursor": None,
        "items": [],
    }


def test_list_masks_sensitive_content_by_default() -> None:
    """默认 ``mask=true``：手机号与身份证号被字段级正则脱敏，``masked`` 标记为真。"""
    client = _client()
    _open_session(client)
    _append(client, [{"role": "patient", "content": SENSITIVE_TEXT}])
    body = client.get("/sessions/s-1/messages", params=SCOPE_PARAMS).json()
    item = body["items"][0]
    assert body["masked"] is True
    assert item["masked"] is True
    assert "13812345678" not in item["content"]
    assert "138****5678" in item["content"]
    assert "110101199003071234" not in item["content"]


def test_list_with_mask_disabled_returns_raw_content() -> None:
    """``mask=false`` 返回原文，供已授权调用方回显（脱敏开关参数）。"""
    client = _client()
    _open_session(client)
    _append(client, [{"role": "patient", "content": SENSITIVE_TEXT}])
    body = client.get("/sessions/s-1/messages", params={**SCOPE_PARAMS, "mask": "false"}).json()
    item = body["items"][0]
    assert body["masked"] is False
    assert item["masked"] is False
    assert item["content"] == SENSITIVE_TEXT


def test_list_applies_per_tenant_mask_policy() -> None:
    """脱敏按租户策略差异化生效（本租户只登记手机号规则，身份证保持原文）。"""
    masker = PolicyMasker.from_config({"hosp-a": {"rules": ["phone"]}})
    client = _client(masker=masker)
    _open_session(client)
    _append(client, [{"role": "patient", "content": SENSITIVE_TEXT}])
    item = client.get("/sessions/s-1/messages", params=SCOPE_PARAMS).json()["items"][0]
    assert "138****5678" in item["content"]
    assert "110101199003071234" in item["content"]


def test_list_paginates_with_cursor() -> None:
    """游标分页：首页返回 ``next_cursor``，次页返回剩余消息且无游标。"""
    client = _client()
    _open_session(client)
    _append(client, [_message_body(index, created_at=1000 + index) for index in range(3)])
    first = client.get("/sessions/s-1/messages", params={**SCOPE_PARAMS, "limit": 2}).json()
    assert [item["content"] for item in first["items"]] == ["msg-0", "msg-1"]
    assert first["next_cursor"] is not None

    second = client.get(
        "/sessions/s-1/messages",
        params={**SCOPE_PARAMS, "limit": 2, "cursor": first["next_cursor"]},
    ).json()
    assert [item["content"] for item in second["items"]] == ["msg-2"]
    assert second["next_cursor"] is None


def test_list_cursor_replay_is_stable() -> None:
    """同一游标重复请求返回同一页（边界：幂等重放）。"""
    client = _client()
    _open_session(client)
    _append(client, [_message_body(index, created_at=1000 + index) for index in range(3)])
    cursor = client.get("/sessions/s-1/messages", params={**SCOPE_PARAMS, "limit": 1}).json()[
        "next_cursor"
    ]
    params = {**SCOPE_PARAMS, "limit": 1, "cursor": cursor}
    first = client.get("/sessions/s-1/messages", params=params).json()
    second = client.get("/sessions/s-1/messages", params=params).json()
    assert first == second


def test_list_cursor_ignores_messages_inserted_before_boundary() -> None:
    """翻页期间插入更早的消息不会导致重复（游标基于时序定位键）。"""
    client = _client()
    _open_session(client)
    _append(client, [_message_body(index, created_at=1000 + index * 1000) for index in range(3)])
    first = client.get("/sessions/s-1/messages", params={**SCOPE_PARAMS, "limit": 2}).json()
    _append(client, [_message_body(9, created_at=1500)])
    second = client.get(
        "/sessions/s-1/messages",
        params={**SCOPE_PARAMS, "limit": 2, "cursor": first["next_cursor"]},
    ).json()
    assert [item["content"] for item in second["items"]] == ["msg-2"]


def test_list_rejects_malformed_cursor() -> None:
    """非法游标 → 400 ``validation_error``（边界：脏游标）。"""
    client = _client()
    _open_session(client)
    response = client.get("/sessions/s-1/messages", params={**SCOPE_PARAMS, "cursor": "%%%"})
    assert response.status_code == 400
    assert response.json()["error"] == "validation_error"


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 201}, {"mask": "maybe"}])
def test_list_rejects_invalid_query(params: dict[str, Any]) -> None:
    """非法分页或非布尔脱敏开关 → 422。"""
    client = _client()
    _open_session(client)
    response = client.get("/sessions/s-1/messages", params={**SCOPE_PARAMS, **params})
    assert response.status_code == 422


def test_list_missing_session_returns_404() -> None:
    """会话不存在 → 404（边界：不存在的 ID）。"""
    response = _client().get("/sessions/s-none/messages", params=SCOPE_PARAMS)
    assert response.status_code == 404
    assert response.json()["error"] == "session_not_found_error"


def test_list_soft_deleted_session_returns_404() -> None:
    """已软删除会话的消息不再对外可见 → 404（合规：软删除后不可查询）。"""
    client = _client()
    _open_session(client)
    _append(client, [_message_body(0)])
    client.post("/sessions/s-1/archive", params=SCOPE_PARAMS)
    client.delete("/sessions/s-1", params=SCOPE_PARAMS)
    response = client.get("/sessions/s-1/messages", params=SCOPE_PARAMS)
    assert response.status_code == 404


def test_list_from_other_tenant_returns_404() -> None:
    """其他租户查询同一 session_id 返回 404 而非泄漏数据（越权防护）。"""
    client = _client()
    _open_session(client)
    _append(client, [_message_body(0)])
    assert client.get("/sessions/s-1/messages", params=OTHER_PARAMS).status_code == 404


def test_list_isolates_messages_between_sessions() -> None:
    """同命名空间下不同会话的消息互不串台。"""
    client = _client()
    _open_session(client, "s-1")
    _open_session(client, "s-2")
    _append(client, [_message_body(0)], session_id="s-1")
    _append(client, [_message_body(1)], session_id="s-2")
    body = client.get("/sessions/s-2/messages", params=SCOPE_PARAMS).json()
    assert [item["content"] for item in body["items"]] == ["msg-1"]


def test_error_response_carries_request_id() -> None:
    """错误响应透传请求追踪 ID（跨中间件与异常处理器）。"""
    response = _client().get(
        "/sessions/s-none/messages", params=SCOPE_PARAMS, headers={"X-Request-ID": "rid-msg"}
    )
    assert response.status_code == 404
    assert response.json()["request_id"] == "rid-msg"


# --------------------------------------------------------------------------- #
# DTO 契约
# --------------------------------------------------------------------------- #
def test_message_response_from_med_maps_all_fields() -> None:
    """领域模型 → 响应 DTO 字段完整映射。"""
    message = _med_message(0, created_at=1234)
    dto = MessageResponse.from_med(message)
    assert dto.message_id == message.message_id
    assert dto.session_id == "s-1"
    assert dto.role is MessageRole.DOCTOR
    assert dto.token_count == 3
    assert dto.masked is False
    assert dto.created_at == 1234


def test_message_response_is_frozen() -> None:
    """响应 DTO 不可变（边界：赋值被拒绝）。"""
    dto = MessageResponse.from_med(_med_message(0))
    with pytest.raises(PydanticValidationError):
        dto.content = "tampered"  # type: ignore[misc]


def test_message_create_requires_content_and_role() -> None:
    """创建 DTO 缺少必填字段即校验失败（边界：必填字段缺失）。"""
    with pytest.raises(PydanticValidationError):
        MessageCreate.model_validate({"content": "hi"})


def test_message_create_rejects_unknown_field() -> None:
    """创建 DTO 拒绝未声明字段（如伪造 tenant_id）。"""
    with pytest.raises(PydanticValidationError):
        MessageCreate.model_validate({"role": "patient", "content": "hi", "tenant_id": "hosp-b"})


def test_message_create_accepts_explicit_null_message_id() -> None:
    """``message_id`` 显式传 ``None`` 视为缺省，由服务端生成 ID（边界：显式 null）。"""
    dto = MessageCreate.model_validate({"role": "patient", "content": "hi", "message_id": None})
    assert dto.message_id is None


def test_message_append_request_requires_at_least_one_message() -> None:
    """追加请求体至少 1 条消息（边界：空数组）。"""
    with pytest.raises(PydanticValidationError):
        MessageAppendRequest.model_validate({"messages": []})


def test_message_list_response_rejects_zero_limit() -> None:
    """列表 DTO 拒绝非正 ``limit``（边界：非法分页）。"""
    with pytest.raises(PydanticValidationError):
        MessageListResponse(session_id="s-1", limit=0, masked=True, items=[])


# --------------------------------------------------------------------------- #
# 依赖注入
# --------------------------------------------------------------------------- #
def _request_with_app(app: object) -> Request:
    """构造仅带 ``app`` 的最小 ASGI 请求对象。"""
    return Request({"type": "http", "method": "GET", "path": "/", "headers": [], "app": app})


def test_get_message_repository_returns_injected_instance() -> None:
    """``app.state.message_repository`` 中的仓储被原样返回。"""
    repository = InMemoryMessageRepository()
    app = SimpleNamespace(state=SimpleNamespace(message_repository=repository))
    assert get_message_repository(_request_with_app(app)) is repository


def test_get_message_repository_without_injection_raises() -> None:
    """未注入消息仓储时抛 :class:`RuntimeError`（边界：装配缺失）。"""
    app = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="message repository is not configured"):
        get_message_repository(_request_with_app(app))


def test_get_message_masker_returns_injected_instance() -> None:
    """``app.state.masker`` 中的脱敏分发器被原样返回。"""
    masker = PolicyMasker()
    app = SimpleNamespace(state=SimpleNamespace(masker=masker))
    assert get_message_masker(_request_with_app(app)) is masker


def test_get_message_masker_without_injection_raises() -> None:
    """未注入脱敏分发器时抛 :class:`RuntimeError`（边界：装配缺失）。"""
    app = SimpleNamespace(state=SimpleNamespace())
    with pytest.raises(RuntimeError, match="masker is not configured"):
        get_message_masker(_request_with_app(app))


def test_create_app_defaults_to_in_memory_backends() -> None:
    """应用工厂缺省注入进程内仓储与内置规则脱敏分发器，消息端点开箱可用。"""
    app = create_app(MedMemorySettings())
    assert isinstance(app.state.message_repository, InMemoryMessageRepository)
    assert isinstance(app.state.session_repository, InMemorySessionRepository)
    assert isinstance(app.state.masker, PolicyMasker)


def test_openapi_exposes_message_endpoints() -> None:
    """OpenAPI 文档包含消息端点与脱敏/游标查询参数（对外契约可发现）。"""
    schema = _client().get("/openapi.json").json()
    path = "/sessions/{session_id}/messages"
    assert set(schema["paths"][path]) == {"get", "post"}
    params = {param["name"] for param in schema["paths"][path]["get"]["parameters"]}
    assert {"cursor", "limit", "mask", "tenant_id", "dept_id"} <= params
