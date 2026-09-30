"""管理端点：归档统计、跨存储迁移触发、会话快照导出。

端点一览：

======================================================  ======  ==========================================
方法 + 路径                                              状态码  说明
======================================================  ======  ==========================================
``GET  /admin/stats``                                    200     按状态汇总会话数与消息数
``POST /admin/sessions/{session_id}/migrate``            200     触发跨存储迁移（幂等可重跑）
``POST /admin/sessions/{session_id}/snapshot``           200     导出快照文件包（base64 内联返回）
======================================================  ======  ==========================================

设计要点：

* **管理端点看得见已软删除的会话**：合规导出与迁移必须能覆盖 ``DELETED`` 状态，
  因此统一走 :func:`~med_langchain_memory.api.session_guards.require_session`
  （仅要求存在），而普通查询走 ``require_visible_session``；
* **存储句柄由 :class:`HistoryResolver` 解析**，HTTP 层不直接触碰
  :class:`~med_langchain_memory.stores.factory.StoreFactory` 的全局注册表，
  便于测试注入替身与私有部署替换；
* **快照不落盘**：文件包以 base64 内联返回并附 SHA-256，调用方自行决定存储位置，
  避免管理接口在服务端任意写文件；
* **迁移与快照都基于存储层**（``MedChatMessageHistory``），与会话/消息仓储
  （索引层）互不影响，二者可在不同后端上独立演进。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, status

from med_langchain_memory.domain.message import IdStr
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.lifecycle.migrator import Migrator
from med_langchain_memory.lifecycle.snapshot import SessionSnapshotter
from med_langchain_memory.stores.session_repository import (
    MAX_PAGE_SIZE,
    SessionRepository,
    SessionScope,
)

from ..deps import HistoryResolverDep, SessionRepositoryDep, SessionScopeDep
from ..schemas import (
    AdminStatsResponse,
    MigrationRequest,
    MigrationResponse,
    SnapshotRequest,
    SnapshotResponse,
)
from ..session_guards import require_session

_NOT_FOUND_RESPONSE: dict[int | str, dict[str, Any]] = {
    status.HTTP_404_NOT_FOUND: {"description": "会话或存储后端不存在"}
}


def _iter_sessions(repository: SessionRepository, scope: SessionScope) -> Iterator[SessionMeta]:
    """按分页遍历命名空间下的全部会话（含已软删除会话）。

    仓储 ``list`` 单页上限为 ``MAX_PAGE_SIZE``，此处循环翻页，避免大租户下
    统计结果被静默截断。

    Args:
        repository: 会话仓储。
        scope: 命名空间坐标。

    Yields:
        该命名空间下的会话元数据。
    """
    offset = 0
    while True:
        page, total = repository.list(scope, limit=MAX_PAGE_SIZE, offset=offset)
        yield from page
        offset += len(page)
        if not page or offset >= total:
            return


def build_admin_router() -> APIRouter:
    """构建管理路由。

    Returns:
        已注册统计、迁移与快照端点的 ``APIRouter``（前缀 ``/admin``）。
    """
    router = APIRouter(prefix="/admin", tags=["admin"])

    @router.get(
        "/stats",
        response_model=AdminStatsResponse,
        summary="按状态汇总会话与消息统计",
    )
    async def admin_stats(
        scope: SessionScopeDep,
        sessions: SessionRepositoryDep,
    ) -> AdminStatsResponse:
        """统计本租户本科室的会话数（按状态分组）与累计消息条数。"""
        by_status: dict[str, int] = {item.value: 0 for item in SessionStatus}
        total_sessions = 0
        total_messages = 0
        for meta in _iter_sessions(sessions, scope):
            by_status[meta.status.value] += 1
            total_messages += meta.message_count
            total_sessions += 1
        return AdminStatsResponse(
            tenant_id=scope.tenant_id,
            dept_id=scope.dept_id,
            total_sessions=total_sessions,
            total_messages=total_messages,
            by_status=by_status,
        )

    @router.post(
        "/sessions/{session_id}/migrate",
        response_model=MigrationResponse,
        summary="触发会话跨存储迁移",
        responses={
            **_NOT_FOUND_RESPONSE,
            status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "源/目标存储不可用"},
        },
    )
    async def migrate_session(
        session_id: IdStr,
        payload: MigrationRequest,
        scope: SessionScopeDep,
        sessions: SessionRepositoryDep,
        resolver: HistoryResolverDep,
    ) -> MigrationResponse:
        """把会话消息从源后端迁移到目标后端（按消息 ID 幂等，可安全重跑）。"""
        meta = require_session(sessions, scope, session_id)
        source = resolver.resolve(
            payload.source_backend,
            scope=scope,
            session_id=session_id,
            patient_id=meta.patient_id,
            options=payload.source_options,
        )
        target = resolver.resolve(
            payload.target_backend,
            scope=scope,
            session_id=session_id,
            patient_id=meta.patient_id,
            options=payload.target_options,
        )
        result = Migrator(
            source,
            target,
            batch_size=payload.batch_size,
            verify=payload.verify,
        ).migrate()
        return MigrationResponse.from_result(
            result,
            session_id=session_id,
            source_backend=payload.source_backend,
            target_backend=payload.target_backend,
        )

    @router.post(
        "/sessions/{session_id}/snapshot",
        response_model=SnapshotResponse,
        summary="导出会话快照文件包",
        responses=_NOT_FOUND_RESPONSE,
    )
    async def export_snapshot(
        session_id: IdStr,
        payload: SnapshotRequest,
        scope: SessionScopeDep,
        sessions: SessionRepositoryDep,
        resolver: HistoryResolverDep,
    ) -> SnapshotResponse:
        """导出会话元数据 + 全量消息为带 SHA-256 校验和的快照文件包。"""
        meta = require_session(sessions, scope, session_id)
        history = resolver.resolve(
            payload.backend,
            scope=scope,
            session_id=session_id,
            patient_id=meta.patient_id,
            options=payload.options,
        )
        prepared = SessionSnapshotter().prepare_snapshot(
            history, schema_version=payload.schema_version
        )
        return SnapshotResponse(
            session_id=session_id,
            session_key=meta.storage_key,
            backend=payload.backend,
            schema_version=prepared.schema_version,
            message_count=prepared.message_count,
            size_bytes=prepared.size_bytes,
            sha256=prepared.sha256,
            payload_base64=base64.b64encode(prepared.to_bytes()).decode("ascii"),
        )

    return router
