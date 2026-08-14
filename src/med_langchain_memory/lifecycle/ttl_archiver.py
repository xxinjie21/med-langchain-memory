"""TTL 驱动的会话自动归档调度器。

将「活跃会话超期 → 自动迁移至冷归档层」这一后台职责收敛到单一模块：

* :class:`TtlArchiver` 接收一个会话枚举器、热存储构造器与归档层构造器，
  周期性调用 :meth:`TtlArchiver.run` 扫描候选会话；
* 仅依赖 ``MedChatMessageHistory`` 公共接口，因此热存储与归档层可为任意已注册后端
  （内存 / 文件 / Redis / MySQL / ES）；典型的归档构造器返回 ``EsArchiveMedHistory``；
* 单会话归档 :func:`archive_expired_session` 在被测会话 ``is_expired`` 为 ``True``
  且状态为 ``ACTIVE`` / ``CLOSED`` 时，先把全量消息写入归档层，
  **写入成功后才**将热会话状态流转为 ``ARCHIVED``，避免「状态已归档但数据丢失」；
* 归档层以 ``message_id`` 为幂等键，重跑不会产生重复数据；
* 调度器对单会话异常容忍：某会话失败不影响其余会话，错误汇总进 :class:`ArchiveReport`。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.stores.base import MedChatMessageHistory

#: 允许被归档（流转至 ``ARCHIVED``）的会话状态集合。
_ARCHIVABLE = frozenset({SessionStatus.ACTIVE, SessionStatus.CLOSED})


@dataclass
class ArchiveResult:
    """单会话归档结果。

    Attributes:
        session_key: 会话统一存储键。
        archived: 本次是否实际执行了归档写入 + 状态流转。
        message_count: 本次实际写入归档层的消息条数。
        reason: 未归档时的原因标记（``not_expired`` / ``status:xxx`` / 空串）。
        error: 归档过程中抛出的异常描述；``None`` 表示无错误。
    """

    session_key: str
    archived: bool = False
    message_count: int = 0
    reason: str = ""
    error: str | None = None


@dataclass
class ArchiveReport:
    """一次调度执行的总体报告。

    Attributes:
        scanned: 枚举到的候选会话总数。
        archived: 本次实际归档的会话数。
        skipped: 因未过期或已归档而跳过的会话数。
        failed: 因异常未能完成的会话数。
        message_total: 全部归档会话写入的消息条数合计。
        errors: 失败会话的 ``{session_key}: {异常}`` 描述列表。
        results: 逐会话结果明细，顺序与枚举顺序一致。
    """

    scanned: int = 0
    archived: int = 0
    skipped: int = 0
    failed: int = 0
    message_total: int = 0
    errors: list[str] = field(default_factory=list)
    results: list[ArchiveResult] = field(default_factory=list)


def _archive_one(
    hot: MedChatMessageHistory,
    archive: MedChatMessageHistory,
    delete_after_archive: bool,
    now_ms: int | None,
) -> ArchiveResult:
    """把单个过期会话迁移到归档层（核心单会话逻辑，供调度器与便捷函数复用）。

    先判定过期与可归档状态，再读取并写入全量消息，最后才流转热会话状态；
    仅当写入全部成功时状态才变为 ``ARCHIVED``，从而保证失败可安全重跑。

    Args:
        hot: 热存储会话历史（需支持 TTL 才能被判定过期）。
        archive: 归档层会话历史，命名空间须与 ``hot`` 一致。
        delete_after_archive: 归档成功后是否清除热存储数据。
        now_ms: 当前 epoch 毫秒；``None`` 表示取系统时间。

    Returns:
        单会话归档结果 :class:`ArchiveResult`。
    """
    key = hot.storage_key
    if not hot.is_expired(now_ms):
        return ArchiveResult(session_key=key, reason="not_expired")
    if hot.session_meta.status not in _ARCHIVABLE:
        return ArchiveResult(session_key=key, reason=f"status:{hot.session_meta.status.value}")

    messages = hot.get_med_messages()
    if messages:
        # 归档层以 message_id 为幂等键（如 ES _id），重跑不重复。
        archive.add_med_messages(messages)
    # 写入成功后才流转状态，避免「状态已归档但数据丢失」。
    hot.archive()
    if delete_after_archive:
        hot.clear()
    return ArchiveResult(session_key=key, archived=True, message_count=len(messages))


def archive_expired_session(
    hot: MedChatMessageHistory,
    archive: MedChatMessageHistory,
    *,
    delete_after_archive: bool = False,
    now_ms: int | None = None,
) -> ArchiveResult:
    """便捷函数：将单个过期会话迁移到归档层。

    Args:
        hot: 热存储会话历史。
        archive: 归档层会话历史（命名空间须与 ``hot`` 一致）。
        delete_after_archive: 归档成功后是否清除热存储数据。
        now_ms: 当前 epoch 毫秒；``None`` 表示取系统时间。

    Returns:
        单会话归档结果。
    """
    return _archive_one(hot, archive, delete_after_archive, now_ms)


class TtlArchiver:
    """TTL 驱动的会话自动归档调度器（后台运行）。

    仅依赖 ``MedChatMessageHistory`` 公共接口，与具体存储后端解耦；
    典型用法是周期性调用 :meth:`run` 扫描候选会话并沉降到 ES 归档层。

    Example:
        >>> from med_langchain_memory.stores.es_store import EsArchiveMedHistory
        >>> archiver = TtlArchiver(
        ...     list_sessions=lambda: ["med:chat:hosp-a:cardio:s-1"],
        ...     make_hot=lambda key: ...,  # 构造热存储句柄
        ...     make_archive=lambda meta: EsArchiveMedHistory(
        ...         session_id=meta.session_id, tenant_id=meta.tenant_id,
        ...         dept_id=meta.dept_id, patient_id=meta.patient_id, client=es_client,
        ...     ),
        ... )
        >>> report = archiver.run()
    """

    def __init__(
        self,
        *,
        list_sessions: Callable[[], list[str]],
        make_hot: Callable[[str], MedChatMessageHistory],
        make_archive: Callable[[SessionMeta], MedChatMessageHistory],
        delete_after_archive: bool = False,
    ) -> None:
        """初始化调度器。

        Args:
            list_sessions: 枚举候选会话统一存储键的回调（如扫描 Redis 键空间）。
            make_hot: 按存储键构造热存储句柄的回调。
            make_archive: 按会话元数据构造归档层句柄的回调（典型返回 ``EsArchiveMedHistory``）。
            delete_after_archive: 归档成功后是否清除热存储数据，默认 ``False``。

        Raises:
            ValueError: 任一回调为非可调用对象时。
        """
        if not callable(list_sessions) or not callable(make_hot) or not callable(make_archive):
            raise ValueError("list_sessions, make_hot and make_archive must be callable")
        self._list_sessions = list_sessions
        self._make_hot = make_hot
        self._make_archive = make_archive
        self._delete_after_archive = delete_after_archive

    def archive_session(
        self,
        hot: MedChatMessageHistory,
        archive: MedChatMessageHistory,
        *,
        now_ms: int | None = None,
    ) -> ArchiveResult:
        """归档单个过期会话（公开封装，供外部按需调用）。

        Args:
            hot: 热存储会话历史。
            archive: 归档层会话历史（命名空间须与 ``hot`` 一致）。
            now_ms: 当前 epoch 毫秒；``None`` 表示取系统时间。

        Returns:
            单会话归档结果。
        """
        return _archive_one(hot, archive, self._delete_after_archive, now_ms)

    def run(self, *, now_ms: int | None = None) -> ArchiveReport:
        """扫描全部候选会话，将过期者可归档者沉降到归档层。

        对单会话异常（构造失败 / 写入失败等）容忍：记录进报告并继续处理后续会话。

        Args:
            now_ms: 判过期使用的当前 epoch 毫秒；``None`` 表示取系统时间。

        Returns:
            调度报告 :class:`ArchiveReport`，含扫描数、归档数、跳过数、失败数及逐会话明细。
        """
        report = ArchiveReport()
        keys = self._list_sessions()
        report.scanned = len(keys)
        for key in keys:
            try:
                hot = self._make_hot(key)
                archive = self._make_archive(hot.session_meta)
                result = _archive_one(hot, archive, self._delete_after_archive, now_ms)
            except Exception as exc:  # 容忍单会话失败，保证调度整体推进
                report.failed += 1
                report.errors.append(f"{key}: {exc}")
                report.results.append(ArchiveResult(session_key=key, error=str(exc)))
                continue
            report.results.append(result)
            if result.archived:
                report.archived += 1
                report.message_total += result.message_count
            else:
                report.skipped += 1
        return report
