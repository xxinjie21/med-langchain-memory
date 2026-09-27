"""会话索引仓储单元测试（D31）。

覆盖：:class:`SessionScope` 的存储键推导与命名空间匹配、分页参数校验、
:class:`InMemorySessionRepository` 的 add/get/update/list 正向路径与
「重复创建 / 目标缺失 / 跨租户不可见 / 分页越界」等边界与异常路径。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError as PydanticValidationError

from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.exceptions import IntegrityError, SessionNotFoundError, ValidationError
from med_langchain_memory.stores.session_repository import (
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE,
    InMemorySessionRepository,
    SessionScope,
    validate_pagination,
)

SCOPE = SessionScope(tenant_id="hosp-a", dept_id="cardio")
OTHER_TENANT = SessionScope(tenant_id="hosp-b", dept_id="cardio")
OTHER_DEPT = SessionScope(tenant_id="hosp-a", dept_id="neuro")


def _meta(
    session_id: str = "s-1",
    *,
    tenant_id: str = "hosp-a",
    dept_id: str = "cardio",
    patient_id: str = "p-1",
    status: SessionStatus = SessionStatus.ACTIVE,
    created_at: int = 1000,
    message_count: int = 0,
) -> SessionMeta:
    """构造可控时间戳的会话元数据。"""
    return SessionMeta(
        session_id=session_id,
        tenant_id=tenant_id,
        dept_id=dept_id,
        patient_id=patient_id,
        status=status,
        created_at=created_at,
        message_count=message_count,
    )


# --------------------------------------------------------------------------- #
# SessionScope
# --------------------------------------------------------------------------- #
def test_scope_storage_key_follows_spec() -> None:
    """存储键严格遵循 ``med:chat:{tenant}:{dept}:{session}`` 规范。"""
    assert SCOPE.storage_key("s-1") == "med:chat:hosp-a:cardio:s-1"


def test_scope_matches_same_namespace() -> None:
    """同租户同科室的会话元数据匹配成功。"""
    assert SCOPE.matches(_meta()) is True


@pytest.mark.parametrize(
    "meta",
    [
        _meta(tenant_id="hosp-b"),
        _meta(dept_id="neuro"),
        _meta(tenant_id="hosp-b", dept_id="neuro"),
    ],
)
def test_scope_rejects_other_namespace(meta: SessionMeta) -> None:
    """租户或科室任一不符即不匹配（边界：跨租户/跨科室）。"""
    assert SCOPE.matches(meta) is False


def test_scope_is_frozen() -> None:
    """命名空间坐标不可变（边界：赋值被拒绝）。"""
    with pytest.raises(PydanticValidationError):
        SCOPE.tenant_id = "hosp-b"  # type: ignore[misc]


def test_scope_rejects_invalid_id_chars() -> None:
    """ID 含 ``:`` 会破坏存储键规范，构造即失败（边界：非法字符）。"""
    with pytest.raises(PydanticValidationError):
        SessionScope(tenant_id="hosp:a", dept_id="cardio")


def test_scope_rejects_unknown_field() -> None:
    """未声明字段被拒绝（``extra="forbid"``）。"""
    with pytest.raises(PydanticValidationError):
        SessionScope(tenant_id="hosp-a", dept_id="cardio", ward_id="w-1")  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# validate_pagination
# --------------------------------------------------------------------------- #
def test_validate_pagination_accepts_bounds() -> None:
    """边界值 1 与 MAX_PAGE_SIZE 均合法。"""
    validate_pagination(limit=1, offset=0)
    validate_pagination(limit=MAX_PAGE_SIZE, offset=10_000)


@pytest.mark.parametrize(("limit", "offset"), [(0, 0), (-1, 0), (MAX_PAGE_SIZE + 1, 0), (10, -1)])
def test_validate_pagination_rejects_out_of_range(limit: int, offset: int) -> None:
    """limit 越界或 offset 为负时抛 :class:`ValidationError`。"""
    with pytest.raises(ValidationError):
        validate_pagination(limit=limit, offset=offset)


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #
def test_add_returns_and_stores_meta() -> None:
    """新增会话返回同一元数据且可被检索到。"""
    repository = InMemorySessionRepository()
    meta = _meta()
    assert repository.add(meta) is meta
    assert repository.get(SCOPE, "s-1") == meta


def test_add_duplicate_raises_integrity_error() -> None:
    """同一命名空间重复创建同一会话被拒绝（边界：重复键）。"""
    repository = InMemorySessionRepository()
    repository.add(_meta())
    with pytest.raises(IntegrityError):
        repository.add(_meta(patient_id="p-2"))


def test_add_same_session_id_in_other_namespace_is_allowed() -> None:
    """不同租户/科室下同名 session_id 互不冲突（命名空间隔离）。"""
    repository = InMemorySessionRepository()
    repository.add(_meta(tenant_id="hosp-a"))
    repository.add(_meta(tenant_id="hosp-b"))
    assert repository.get(SCOPE, "s-1") is not None
    assert repository.get(OTHER_TENANT, "s-1") is not None


# --------------------------------------------------------------------------- #
# get
# --------------------------------------------------------------------------- #
def test_get_missing_returns_none() -> None:
    """查询不存在的会话返回 ``None``（边界：空仓储）。"""
    assert InMemorySessionRepository().get(SCOPE, "s-1") is None


def test_get_with_wrong_scope_returns_none() -> None:
    """跨科室查询看不到他人会话（边界：越权查询）。"""
    repository = InMemorySessionRepository()
    repository.add(_meta())
    assert repository.get(OTHER_DEPT, "s-1") is None


# --------------------------------------------------------------------------- #
# update
# --------------------------------------------------------------------------- #
def test_update_replaces_existing_meta() -> None:
    """更新后读回的是新元数据（状态已流转）。"""
    repository = InMemorySessionRepository()
    repository.add(_meta())
    closed = _meta(status=SessionStatus.CLOSED)
    assert repository.update(closed) == closed
    stored = repository.get(SCOPE, "s-1")
    assert stored is not None
    assert stored.status is SessionStatus.CLOSED


def test_update_missing_raises_session_not_found() -> None:
    """更新不存在的会话抛 :class:`SessionNotFoundError`（边界：目标缺失）。"""
    with pytest.raises(SessionNotFoundError):
        InMemorySessionRepository().update(_meta())


# --------------------------------------------------------------------------- #
# list
# --------------------------------------------------------------------------- #
def test_list_empty_repository() -> None:
    """空仓储返回空页与总数 0（边界：无数据）。"""
    assert InMemorySessionRepository().list(SCOPE) == ([], 0)


def test_list_filters_by_scope_and_sorts_by_created_at() -> None:
    """仅返回本命名空间会话，且按 ``(created_at, session_id)`` 升序。"""
    repository = InMemorySessionRepository()
    repository.add(_meta("s-3", created_at=3000))
    repository.add(_meta("s-1", created_at=1000))
    repository.add(_meta("s-2", created_at=2000))
    repository.add(_meta("s-9", tenant_id="hosp-b", created_at=500))
    items, total = repository.list(SCOPE)
    assert [meta.session_id for meta in items] == ["s-1", "s-2", "s-3"]
    assert total == 3


def test_list_sorts_ties_by_session_id() -> None:
    """``created_at`` 相同时按 ``session_id`` 字典序保证稳定（边界：同毫秒创建）。"""
    repository = InMemorySessionRepository()
    repository.add(_meta("s-b", created_at=1000))
    repository.add(_meta("s-a", created_at=1000))
    items, _ = repository.list(SCOPE)
    assert [meta.session_id for meta in items] == ["s-a", "s-b"]


def test_list_filters_by_status() -> None:
    """``status`` 过滤只命中指定状态，``total`` 同步反映过滤后数量。"""
    repository = InMemorySessionRepository()
    repository.add(_meta("s-1", status=SessionStatus.ACTIVE))
    repository.add(_meta("s-2", status=SessionStatus.CLOSED))
    repository.add(_meta("s-3", status=SessionStatus.ARCHIVED))
    items, total = repository.list(SCOPE, status=SessionStatus.CLOSED)
    assert [meta.session_id for meta in items] == ["s-2"]
    assert total == 1


def test_list_pagination_returns_page_and_full_total() -> None:
    """分页只截取当前页，``total`` 始终是命中总数。"""
    repository = InMemorySessionRepository()
    for index in range(5):
        repository.add(_meta(f"s-{index}", created_at=1000 + index))
    items, total = repository.list(SCOPE, limit=2, offset=1)
    assert [meta.session_id for meta in items] == ["s-1", "s-2"]
    assert total == 5


def test_list_offset_beyond_total_returns_empty_page() -> None:
    """偏移量超出总数时返回空页但总数不变（边界：越界翻页）。"""
    repository = InMemorySessionRepository()
    repository.add(_meta())
    items, total = repository.list(SCOPE, limit=10, offset=5)
    assert items == []
    assert total == 1


def test_list_default_page_size_is_used() -> None:
    """默认单页条数为 :data:`DEFAULT_PAGE_SIZE`。"""
    repository = InMemorySessionRepository()
    for index in range(DEFAULT_PAGE_SIZE + 3):
        repository.add(_meta(f"s-{index:03d}", created_at=1000 + index))
    items, total = repository.list(SCOPE)
    assert len(items) == DEFAULT_PAGE_SIZE
    assert total == DEFAULT_PAGE_SIZE + 3


@pytest.mark.parametrize(("limit", "offset"), [(0, 0), (MAX_PAGE_SIZE + 1, 0), (10, -1)])
def test_list_rejects_invalid_pagination(limit: int, offset: int) -> None:
    """非法分页参数直接抛 :class:`ValidationError`，不静默截断。"""
    with pytest.raises(ValidationError):
        InMemorySessionRepository().list(SCOPE, limit=limit, offset=offset)
