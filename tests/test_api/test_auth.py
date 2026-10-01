"""API Key 鉴权与科室 scope 授权单元测试（D34）。

覆盖四个层次：

* 纯函数层 —— :func:`hash_api_key` 的确定性与 :func:`dept_allowed` 的精确/通配匹配；
* 模型层 —— :class:`ApiKeyRecord` / :class:`AuthPrincipal` 的字段校验（空科室集合、
  非法科室标识、额外字段、不可变）与 ``allows_dept``；
* 认证器层 —— :class:`ApiKeyAuthenticator` 的注册（空密钥、重复 key_id、重复密钥）、
  认证（缺失/空白/未知/停用/大小写敏感/前后空白）与 :func:`authorize_scope` 的
  租户不符、科室越权分支；
* 接口层 —— 端到端 401 / 403 / 201 行为、通配符密钥、自定义请求头名、鉴权关闭时的
  向后兼容路径，以及健康检查不受鉴权影响。
"""

from __future__ import annotations

from collections.abc import Iterable
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError as PydanticValidationError
from starlette.requests import Request

from med_langchain_memory.api import (
    DEFAULT_API_KEY_HEADER,
    WILDCARD_DEPT,
    ApiKeyAuthenticator,
    ApiKeyRecord,
    AuthPrincipal,
    authenticator_of,
    authorize_scope,
    create_app,
    dept_allowed,
    hash_api_key,
    resolve_session_scope,
)
from med_langchain_memory.config import MedMemorySettings
from med_langchain_memory.exceptions import (
    AuthenticationError,
    AuthorizationError,
    IntegrityError,
    ValidationError,
)
from med_langchain_memory.stores.session_repository import SessionScope

#: 测试用明文密钥 / 标识 / 命名空间。
RAW_KEY = "k-1-secret"
KEY_ID = "k-1"
TENANT = "hosp-a"
DEPT = "cardio"

#: 默认命名空间查询参数与默认请求头。
SCOPE_PARAMS: dict[str, str] = {"tenant_id": TENANT, "dept_id": DEPT}
AUTH_HEADER: dict[str, str] = {DEFAULT_API_KEY_HEADER: RAW_KEY}


def _record(
    *,
    key_id: str = KEY_ID,
    tenant_id: str = TENANT,
    dept_ids: frozenset[str] = frozenset({DEPT}),
    enabled: bool = True,
    label: str | None = None,
) -> ApiKeyRecord:
    """构造授权档案，仅覆盖需要变化的字段。"""
    return ApiKeyRecord(
        key_id=key_id,
        tenant_id=tenant_id,
        dept_ids=dept_ids,
        enabled=enabled,
        label=label,
    )


def _authenticator(
    records: dict[str, ApiKeyRecord] | None = None,
    *,
    header_name: str = DEFAULT_API_KEY_HEADER,
) -> ApiKeyAuthenticator:
    """构造装载单把默认密钥的认证器（便于覆盖主要分支）。"""
    return ApiKeyAuthenticator.from_records(
        records if records is not None else {RAW_KEY: _record()},
        header_name=header_name,
    )


def _client(
    authenticator: ApiKeyAuthenticator | None = None,
    *,
    settings: MedMemorySettings | None = None,
) -> TestClient:
    """构造注入指定认证器的测试客户端；``None`` 表示不启用鉴权。"""
    return TestClient(
        create_app(
            settings if settings is not None else MedMemorySettings(), authenticator=authenticator
        )
    )


def _request(app: object, *, headers: Iterable[tuple[bytes, bytes]] = ()) -> Request:
    """构造仅带 ``app`` 与请求头的最小 ASGI 请求对象。"""
    return Request(
        {"type": "http", "method": "GET", "path": "/", "headers": list(headers), "app": app}
    )


# --------------------------------------------------------------------------- #
# hash_api_key / dept_allowed
# --------------------------------------------------------------------------- #
def test_hash_api_key_is_deterministic_sha256_hex() -> None:
    """摘要稳定且为 64 位小写十六进制（正向）。"""
    digest = hash_api_key(RAW_KEY)
    assert digest == hash_api_key(RAW_KEY)
    assert len(digest) == 64
    assert digest == digest.lower()
    assert all(char in "0123456789abcdef" for char in digest)


def test_hash_api_key_differs_for_different_inputs() -> None:
    """不同输入（含空串）得到不同摘要（边界）。"""
    digests = {hash_api_key(value) for value in ("", "a", "a ", "A", RAW_KEY)}
    assert len(digests) == 5


def test_dept_allowed_exact_and_wildcard() -> None:
    """精确匹配命中、未命中为假，通配符覆盖全部科室（正向 + 边界）。"""
    assert dept_allowed(frozenset({DEPT}), DEPT) is True
    assert dept_allowed(frozenset({DEPT}), "neuro") is False
    assert dept_allowed(frozenset({DEPT}), "cardio-oncology") is False
    assert dept_allowed(frozenset({WILDCARD_DEPT}), "any-dept") is True


# --------------------------------------------------------------------------- #
# ApiKeyRecord
# --------------------------------------------------------------------------- #
def test_api_key_record_defaults_and_allows_dept() -> None:
    """缺省 ``label``/``enabled`` 合法，``allows_dept`` 按白名单判定（正向）。"""
    record = _record()
    assert record.label is None
    assert record.enabled is True
    assert record.allows_dept(DEPT) is True
    assert record.allows_dept("neuro") is False


def test_api_key_record_accepts_wildcard_dept() -> None:
    """``"*"`` 是合法科室标识并覆盖全部科室（正向边界）。"""
    record = _record(dept_ids=frozenset({WILDCARD_DEPT}))
    assert record.allows_dept("neuro") is True


def test_api_key_record_rejects_empty_dept_ids() -> None:
    """科室集合为空时校验失败（异常）。"""
    with pytest.raises(PydanticValidationError):
        _record(dept_ids=frozenset())


@pytest.mark.parametrize("dept_id", ["科室", "cardio/../x", "", "a" * 65])
def test_api_key_record_rejects_invalid_dept_id(dept_id: str) -> None:
    """非法科室标识（中文、路径穿越、空串、超长）一律校验失败（异常）。"""
    with pytest.raises(PydanticValidationError):
        _record(dept_ids=frozenset({dept_id}))


def test_api_key_record_forbids_extra_fields() -> None:
    """额外字段被拒绝，避免配置拼写错误被静默忽略（异常）。"""
    with pytest.raises(PydanticValidationError):
        ApiKeyRecord(
            key_id=KEY_ID,
            tenant_id=TENANT,
            dept_ids=frozenset({DEPT}),
            unknown="x",  # type: ignore[call-arg]
        )


def test_api_key_record_is_frozen() -> None:
    """档案不可变，防止运行期被意外改写授权范围（异常）。"""
    record = _record()
    with pytest.raises(PydanticValidationError):
        record.enabled = False  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# AuthPrincipal
# --------------------------------------------------------------------------- #
def test_auth_principal_allows_dept() -> None:
    """主体按科室集合判定访问权限（正向 + 边界）。"""
    principal = AuthPrincipal(key_id=KEY_ID, tenant_id=TENANT, dept_ids=frozenset({DEPT}))
    assert principal.allows_dept(DEPT) is True
    assert principal.allows_dept("neuro") is False


def test_auth_principal_rejects_empty_key_id() -> None:
    """``key_id`` 为空串时校验失败（异常）。"""
    with pytest.raises(PydanticValidationError):
        AuthPrincipal(key_id="", tenant_id=TENANT, dept_ids=frozenset({DEPT}))


def test_auth_principal_is_frozen() -> None:
    """主体不可变（异常）。"""
    principal = AuthPrincipal(key_id=KEY_ID, tenant_id=TENANT, dept_ids=frozenset({DEPT}))
    with pytest.raises(PydanticValidationError):
        principal.tenant_id = "hosp-b"  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# ApiKeyAuthenticator · 构造与注册
# --------------------------------------------------------------------------- #
def test_authenticator_default_header_name() -> None:
    """缺省使用 ``X-API-Key`` 头（正向）。"""
    assert ApiKeyAuthenticator().header_name == DEFAULT_API_KEY_HEADER


def test_authenticator_custom_header_name() -> None:
    """支持自定义请求头名（正向）。"""
    assert ApiKeyAuthenticator(header_name="X-Hospital-Key").header_name == "X-Hospital-Key"


def test_authenticator_rejects_empty_header_name() -> None:
    """请求头名为空串时拒绝构造（异常）。"""
    with pytest.raises(ValidationError):
        ApiKeyAuthenticator(header_name="")


def test_add_key_returns_record_and_enables_authentication() -> None:
    """注册后返回原档案且可立即认证（正向）。"""
    authenticator = ApiKeyAuthenticator()
    record = _record()
    assert authenticator.add_key(RAW_KEY, record) is record
    assert authenticator.authenticate(RAW_KEY).key_id == KEY_ID


def test_add_key_rejects_empty_raw_key() -> None:
    """空密钥不可注册（异常）。"""
    authenticator = ApiKeyAuthenticator()
    with pytest.raises(ValidationError):
        authenticator.add_key("", _record())


def test_add_key_rejects_duplicate_key_id() -> None:
    """同一 ``key_id`` 重复注册被拒绝（异常）。"""
    authenticator = _authenticator()
    with pytest.raises(IntegrityError):
        authenticator.add_key("another-secret", _record())


def test_add_key_rejects_duplicate_raw_key() -> None:
    """同一明文密钥重复注册被拒绝（异常）。"""
    authenticator = _authenticator()
    with pytest.raises(IntegrityError):
        authenticator.add_key(RAW_KEY, _record(key_id="k-2"))


def test_from_records_loads_multiple_keys() -> None:
    """批量构造装载多把密钥，各自映射到自己的主体（正向）。"""
    authenticator = _authenticator(
        {
            RAW_KEY: _record(),
            "k-2-secret": _record(key_id="k-2", dept_ids=frozenset({WILDCARD_DEPT})),
        }
    )
    assert authenticator.authenticate(RAW_KEY).key_id == KEY_ID
    wildcard = authenticator.authenticate("k-2-secret")
    assert wildcard.key_id == "k-2"
    assert wildcard.allows_dept("neuro") is True


def test_from_records_with_empty_mapping_rejects_all_keys() -> None:
    """空映射构造的认证器拒绝任何密钥（边界）。"""
    authenticator = _authenticator({})
    with pytest.raises(AuthenticationError):
        authenticator.authenticate(RAW_KEY)


# --------------------------------------------------------------------------- #
# ApiKeyAuthenticator · 认证
# --------------------------------------------------------------------------- #
def test_authenticate_returns_projected_principal() -> None:
    """认证成功返回主体，且不携带 label/enabled 等管理字段（正向）。"""
    principal = _authenticator({RAW_KEY: _record(label="门诊医生站")}).authenticate(RAW_KEY)
    assert principal.key_id == KEY_ID
    assert principal.tenant_id == TENANT
    assert principal.dept_ids == frozenset({DEPT})
    assert set(principal.model_dump()) == {"key_id", "tenant_id", "dept_ids"}


def test_authenticate_strips_surrounding_whitespace() -> None:
    """密钥前后空白被忽略（边界）。"""
    assert _authenticator().authenticate(f"  {RAW_KEY}\n").key_id == KEY_ID


@pytest.mark.parametrize("raw_key", [None, "", "   "])
def test_authenticate_rejects_missing_key(raw_key: str | None) -> None:
    """未携带密钥（None / 空串 / 纯空白）返回 401 语义（异常）。"""
    with pytest.raises(AuthenticationError):
        _authenticator().authenticate(raw_key)


def test_authenticate_rejects_unknown_key() -> None:
    """未注册的密钥被拒绝（异常）。"""
    with pytest.raises(AuthenticationError):
        _authenticator().authenticate("not-registered")


def test_authenticate_is_case_sensitive() -> None:
    """密钥比较大小写敏感（边界）。"""
    with pytest.raises(AuthenticationError):
        _authenticator().authenticate(RAW_KEY.upper())


def test_authenticate_rejects_disabled_key() -> None:
    """已停用的密钥认证失败，错误信息含 key_id 便于排查（异常）。"""
    authenticator = _authenticator({RAW_KEY: _record(enabled=False)})
    with pytest.raises(AuthenticationError, match=KEY_ID):
        authenticator.authenticate(RAW_KEY)


# --------------------------------------------------------------------------- #
# authorize_scope
# --------------------------------------------------------------------------- #
def test_authorize_scope_returns_scope() -> None:
    """租户与科室均命中时返回命名空间坐标（正向）。"""
    principal = AuthPrincipal(key_id=KEY_ID, tenant_id=TENANT, dept_ids=frozenset({DEPT}))
    scope = authorize_scope(principal, tenant_id=TENANT, dept_id=DEPT)
    assert isinstance(scope, SessionScope)
    assert scope.storage_key("s-1") == f"med:chat:{TENANT}:{DEPT}:s-1"


def test_authorize_scope_allows_wildcard_dept() -> None:
    """通配符主体可访问任意科室（正向边界）。"""
    principal = AuthPrincipal(key_id=KEY_ID, tenant_id=TENANT, dept_ids=frozenset({WILDCARD_DEPT}))
    assert authorize_scope(principal, tenant_id=TENANT, dept_id="neuro").dept_id == "neuro"


def test_authorize_scope_rejects_tenant_mismatch() -> None:
    """跨租户访问被拒绝（异常）。"""
    principal = AuthPrincipal(key_id=KEY_ID, tenant_id=TENANT, dept_ids=frozenset({DEPT}))
    with pytest.raises(AuthorizationError, match="hosp-b"):
        authorize_scope(principal, tenant_id="hosp-b", dept_id=DEPT)


def test_authorize_scope_rejects_dept_outside_allowlist() -> None:
    """科室不在白名单时被拒绝（异常）。"""
    principal = AuthPrincipal(key_id=KEY_ID, tenant_id=TENANT, dept_ids=frozenset({DEPT}))
    with pytest.raises(AuthorizationError, match="neuro"):
        authorize_scope(principal, tenant_id=TENANT, dept_id="neuro")


# --------------------------------------------------------------------------- #
# authenticator_of / resolve_session_scope
# --------------------------------------------------------------------------- #
def test_authenticator_of_reads_app_state() -> None:
    """已注入认证器时原样返回（正向）。"""
    authenticator = _authenticator()
    app = SimpleNamespace(state=SimpleNamespace(authenticator=authenticator))
    assert authenticator_of(_request(app)) is authenticator


def test_authenticator_of_without_injection_returns_none() -> None:
    """未注入时返回 ``None``，表示未启用鉴权（边界）。"""
    app = SimpleNamespace(state=SimpleNamespace())
    assert authenticator_of(_request(app)) is None


def test_authenticator_of_with_wrong_type_returns_none() -> None:
    """注入对象类型不符时返回 ``None``，避免误判为已启用鉴权（异常）。"""
    app = SimpleNamespace(state=SimpleNamespace(authenticator="not-an-authenticator"))
    assert authenticator_of(_request(app)) is None


def test_resolve_session_scope_without_authenticator_uses_query_params() -> None:
    """未启用鉴权时直接采用查询参数构造命名空间（向后兼容路径）。"""
    app = SimpleNamespace(state=SimpleNamespace())
    scope = resolve_session_scope(_request(app), TENANT, DEPT)
    assert scope == SessionScope(tenant_id=TENANT, dept_id=DEPT)


def test_resolve_session_scope_with_authenticator_authorizes() -> None:
    """启用鉴权且请求头合法时按密钥授权结果返回命名空间（正向）。"""
    app = SimpleNamespace(state=SimpleNamespace(authenticator=_authenticator()))
    request = _request(app, headers=[(b"x-api-key", RAW_KEY.encode())])
    assert resolve_session_scope(request, TENANT, DEPT) == SessionScope(
        tenant_id=TENANT, dept_id=DEPT
    )


def test_resolve_session_scope_with_authenticator_requires_key() -> None:
    """启用鉴权但未携带请求头时抛认证异常（异常）。"""
    app = SimpleNamespace(state=SimpleNamespace(authenticator=_authenticator()))
    with pytest.raises(AuthenticationError):
        resolve_session_scope(_request(app), TENANT, DEPT)


# --------------------------------------------------------------------------- #
# 端到端：401 / 403 / 201
# --------------------------------------------------------------------------- #
def test_endpoint_without_key_returns_401() -> None:
    """启用鉴权后未带密钥的会话请求返回 401，响应体为统一错误结构（正向异常路径）。"""
    response = _client(_authenticator()).post(
        "/sessions", params=SCOPE_PARAMS, json={"patient_id": "p-1"}
    )
    assert response.status_code == 401
    body = response.json()
    assert body["error"] == "authentication_error"
    assert body["request_id"]
    assert response.headers.get("X-Request-ID") == body["request_id"]


@pytest.mark.parametrize("api_key", ["wrong-key", "K-1-SECRET", " "])
def test_endpoint_with_invalid_key_returns_401(api_key: str) -> None:
    """错误 / 大小写不符 / 纯空白密钥均返回 401（异常）。"""
    response = _client(_authenticator()).get(
        "/sessions", params=SCOPE_PARAMS, headers={DEFAULT_API_KEY_HEADER: api_key}
    )
    assert response.status_code == 401
    assert response.json()["error"] == "authentication_error"


def test_endpoint_with_disabled_key_returns_401() -> None:
    """停用密钥返回 401（异常）。"""
    client = _client(_authenticator({RAW_KEY: _record(enabled=False)}))
    response = client.get("/sessions", params=SCOPE_PARAMS, headers=AUTH_HEADER)
    assert response.status_code == 401


def test_endpoint_with_valid_key_allows_full_session_flow() -> None:
    """合法密钥下创建 + 查询 + 列表全部可用（正向）。"""
    client = _client(_authenticator())
    created = client.post(
        "/sessions", params=SCOPE_PARAMS, json={"patient_id": "p-1"}, headers=AUTH_HEADER
    )
    assert created.status_code == 201, created.text
    session_id = created.json()["session_id"]

    detail = client.get(f"/sessions/{session_id}", params=SCOPE_PARAMS, headers=AUTH_HEADER)
    assert detail.status_code == 200
    listed = client.get("/sessions", params=SCOPE_PARAMS, headers=AUTH_HEADER)
    assert listed.status_code == 200
    assert listed.json()["total"] == 1


def test_endpoint_with_tenant_mismatch_returns_403() -> None:
    """密钥绑定租户与请求租户不一致返回 403（异常）。"""
    client = _client(_authenticator())
    response = client.post(
        "/sessions",
        params={"tenant_id": "hosp-b", "dept_id": DEPT},
        json={"patient_id": "p-1"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 403
    assert response.json()["error"] == "authorization_error"


def test_endpoint_with_dept_outside_allowlist_returns_403() -> None:
    """科室不在白名单返回 403（异常）。"""
    client = _client(_authenticator())
    response = client.get(
        "/sessions", params={"tenant_id": TENANT, "dept_id": "neuro"}, headers=AUTH_HEADER
    )
    assert response.status_code == 403
    assert response.json()["error"] == "authorization_error"


def test_endpoint_with_wildcard_key_allows_any_dept() -> None:
    """通配符密钥可访问该租户下任意科室（正向边界）。"""
    client = _client(_authenticator({RAW_KEY: _record(dept_ids=frozenset({WILDCARD_DEPT}))}))
    response = client.post(
        "/sessions",
        params={"tenant_id": TENANT, "dept_id": "neuro"},
        json={"patient_id": "p-1"},
        headers=AUTH_HEADER,
    )
    assert response.status_code == 201, response.text


def test_endpoint_uses_custom_header_name() -> None:
    """自定义请求头名生效：默认头被忽略（401），自定义头通过（边界）。"""
    header_name = "X-Hospital-Key"
    client = _client(_authenticator(header_name=header_name))

    ignored = client.get("/sessions", params=SCOPE_PARAMS, headers=AUTH_HEADER)
    assert ignored.status_code == 401

    accepted = client.get("/sessions", params=SCOPE_PARAMS, headers={header_name: RAW_KEY})
    assert accepted.status_code == 200


def test_endpoint_without_authenticator_stays_open() -> None:
    """未配置认证器时保持免鉴权（向后兼容 D31–D33 行为）。"""
    response = _client().post("/sessions", params=SCOPE_PARAMS, json={"patient_id": "p-1"})
    assert response.status_code == 201


def test_health_endpoints_are_not_authenticated() -> None:
    """健康检查不依赖命名空间依赖，鉴权开启时仍免密钥可访问（边界）。"""
    client = _client(_authenticator())
    assert client.get("/health").status_code == 200
    assert client.get("/health/ready").status_code == 200


def test_admin_endpoint_requires_key() -> None:
    """管理端点同样受保护：无密钥 401，带密钥 200（正向 + 异常）。"""
    client = _client(_authenticator())
    assert client.get("/admin/stats", params=SCOPE_PARAMS).status_code == 401
    assert client.get("/admin/stats", params=SCOPE_PARAMS, headers=AUTH_HEADER).status_code == 200


def test_message_endpoint_requires_key() -> None:
    """消息端点同样受保护：无密钥 401，带密钥可追加（正向 + 异常）。"""
    client = _client(_authenticator())
    created = client.post(
        "/sessions", params=SCOPE_PARAMS, json={"patient_id": "p-1"}, headers=AUTH_HEADER
    )
    session_id = created.json()["session_id"]
    payload = {"messages": [{"role": "patient", "content": "头痛三天"}]}

    assert (
        client.post(
            f"/sessions/{session_id}/messages", params=SCOPE_PARAMS, json=payload
        ).status_code
        == 401
    )
    appended = client.post(
        f"/sessions/{session_id}/messages",
        params=SCOPE_PARAMS,
        json=payload,
        headers=AUTH_HEADER,
    )
    assert appended.status_code == 201, appended.text


def test_auth_is_enforced_before_business_lookup() -> None:
    """鉴权先于业务查询：不存在的会话在无密钥时返回 401 而非 404（边界）。"""
    client = _client(_authenticator())
    assert client.get("/sessions/s-missing", params=SCOPE_PARAMS).status_code == 401
    assert (
        client.get("/sessions/s-missing", params=SCOPE_PARAMS, headers=AUTH_HEADER).status_code
        == 404
    )


def test_app_state_exposes_authenticator() -> None:
    """应用工厂把认证器写入 ``app.state``，供依赖与运维自检读取（正向）。"""
    authenticator = _authenticator()
    app: FastAPI = create_app(MedMemorySettings(), authenticator=authenticator)
    assert app.state.authenticator is authenticator
