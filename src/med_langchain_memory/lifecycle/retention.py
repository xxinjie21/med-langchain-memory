"""合规保留期：软删除标记与到期物理清理。

将「归档会话在保留期满后软删除、软删除宽限期后物理清理」这一合规职责收敛到单一模块：

* :class:`RetentionPolicy` 声明保留期策略：``retention_days`` 为 ``ARCHIVED`` 会话保留天数，
  超过则标记为软删除（``DELETED``）；``grace_days`` 为软删除后的宽限天数，超过则物理清理消息数据；
* :func:`soft_delete_session` 将一个 ``ARCHIVED`` 会话流转为 ``DELETED`` 软删除标记；
* :func:`purge_expired_session` 将宽限期已满的 ``DELETED`` 会话消息数据物理清除；
* :class:`RetentionManager` 调度器按枚举的候选会话键，自动完成「到期软删除」与「到期物理清理」
  两阶段，对单会话异常容忍并汇总为 :class:`RetentionReport`。

软删除只能从 ``ARCHIVED`` 状态流转（与领域层状态机一致），因此保留期策略面向已经完成
TTL 归档（见 :mod:`med_langchain_memory.lifecycle.ttl_archiver`）的会话。
本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from med_langchain_memory.domain.message import now_millis
from med_langchain_memory.domain.session import SessionMeta, SessionStatus
from med_langchain_memory.stores.base import MedChatMessageHistory

_MS_PER_DAY = 86_400_000


def _days_to_ms(days: int) -> int:
    """将天数换算为毫秒。"""
    return days * _MS_PER_DAY


@dataclass
class RetentionPolicy:
    """合规保留期策略。

    Attributes:
        retention_days: ``ARCHIVED`` 会话保留天数，超过则软删除标记。
        grace_days: 软删除后宽限天数，超过则物理清理消息数据（撤销窗口）。
    """

    retention_days: int
    grace_days: int = 0

    def __post_init__(self) -> None:
        """校验策略参数合法性。

        Raises:
            ValueError: ``retention_days`` 非正或 ``grace_days`` 为负时。
        """
        if self.retention_days <= 0:
            raise ValueError("retention_days must be a positive integer")
        if self.grace_days < 0:
            raise ValueError("grace_days must be >= 0")

    def is_due_for_soft_delete(self, meta: SessionMeta, now_ms: int) -> bool:
        """判断 ``ARCHIVED`` 会话是否已过保留期，应软删除标记。

        Args:
            meta: 会话元数据。
            now_ms: 当前 epoch 毫秒。

        Returns:
            仅当状态为 ``ARCHIVED`` 且距 ``updated_at`` 已超过 ``retention_days`` 时为 ``True``。
        """
        return (
            meta.status is SessionStatus.ARCHIVED
            and now_ms - meta.updated_at >= _days_to_ms(self.retention_days)
        )

    def is_due_for_purge(self, meta: SessionMeta, now_ms: int) -> bool:
        """判断 ``DELETED`` 会话是否已过宽限期，应物理清理。

        Args:
            meta: 会话元数据。
            now_ms: 当前 epoch 毫秒。

        Returns:
            仅当状态为 ``DELETED`` 且距 ``updated_at`` 已超过 ``grace_days`` 时为 ``True``。
        """
        return (
            meta.status is SessionStatus.DELETED
            and now_ms - meta.updated_at >= _days_to_ms(self.grace_days)
        )


@dataclass
class SoftDeleteResult:
    """单会话软删除结果。

    Attributes:
        session_key: 会话统一存储键。
        deleted: 本次是否实际执行了软删除流转。
        reason: 未删除时的原因标记（``not_archived`` / 空串）。
        error: 过程中的异常描述；``None`` 表示无错误。
    """

    session_key: str
    deleted: bool = False
    reason: str = ""
    error: str | None = None


@dataclass
class PurgeResult:
    """单会话物理清理结果。

    Attributes:
        session_key: 会话统一存储键。
        purged: 本次是否实际清除了消息数据。
        purged_count: 本次物理清除的消息条数。
        reason: 未清理时的原因标记（``not_deleted`` / ``not_due`` / ``already_purged`` / 空串）。
        error: 过程中的异常描述；``None`` 表示无错误。
    """

    session_key: str
    purged: bool = False
    purged_count: int = 0
    reason: str = ""
    error: str | None = None


@dataclass
class SessionRetentionResult:
    """调度器逐会话汇总结果。

    Attributes:
        session_key: 会话统一存储键。
        soft_deleted: 本次是否执行了软删除标记。
        purged: 本次是否执行了物理清理。
        purged_count: 物理清除的消息条数。
        reason: 跳过/未处理原因标记。
        error: 过程中的异常描述；``None`` 表示无错误。
    """

    session_key: str
    soft_deleted: bool = False
    purged: bool = False
    purged_count: int = 0
    reason: str = ""
    error: str | None = None


@dataclass
class RetentionReport:
    """一次保留期调度执行的总体报告。

    Attributes:
        scanned: 枚举到的候选会话总数。
        soft_deleted: 本次实际软删除的会话数。
        purged: 本次实际物理清理的会话数。
        skipped: 因未到期或状态不符而跳过的会话数。
        failed: 因异常未能完成的会话数。
        errors: 失败会话的 ``{session_key}: {异常}`` 描述列表。
        results: 逐会话结果明细，顺序与枚举顺序一致。
    """

    scanned: int = 0
    soft_deleted: int = 0
    purged: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    results: list[SessionRetentionResult] = field(default_factory=list)


def soft_delete_session(history: MedChatMessageHistory) -> SoftDeleteResult:
    """便捷函数：将单个 ``ARCHIVED`` 会话软删除标记（流转为 ``DELETED``）。

    Args:
        history: 会话历史句柄，命名空间须为 ``ARCHIVED`` 状态。

    Returns:
        软删除结果 :class:`SoftDeleteResult`。
    """
    key = history.storage_key
    if history.session_meta.status is not SessionStatus.ARCHIVED:
        return SoftDeleteResult(session_key=key, reason="not_archived")
    history.delete()
    return SoftDeleteResult(session_key=key, deleted=True)


def purge_expired_session(
    history: MedChatMessageHistory,
    policy: RetentionPolicy,
    *,
    now_ms: int | None = None,
) -> PurgeResult:
    """便捷函数：将宽限期已满的 ``DELETED`` 会话消息数据物理清除。

    清除前先校验状态与宽限期，已无消息的会话视为已清理并跳过，保证幂等重跑。

    Args:
        history: 会话历史句柄，命名空间须为 ``DELETED`` 状态。
        policy: 保留期策略，提供宽限期判定。
        now_ms: 当前 epoch 毫秒；``None`` 表示取系统时间。

    Returns:
        物理清理结果 :class:`PurgeResult`。
    """
    key = history.storage_key
    if history.session_meta.status is not SessionStatus.DELETED:
        return PurgeResult(session_key=key, reason="not_deleted")
    now = now_millis() if now_ms is None else now_ms
    if not policy.is_due_for_purge(history.session_meta, now):
        return PurgeResult(session_key=key, reason="not_due")
    messages = history.get_med_messages()
    if not messages:
        return PurgeResult(session_key=key, reason="already_purged")
    history.clear()
    return PurgeResult(session_key=key, purged=True, purged_count=len(messages))


class RetentionManager:
    """合规保留期调度器：自动完成到期软删除与到期物理清理。

    仅依赖 ``MedChatMessageHistory`` 公共接口，与具体存储后端解耦；
    典型用法是周期性调用 :meth:`run` 扫描候选会话并执行两阶段保留期处理。

    Example:
        >>> from med_langchain_memory.stores.memory_store import InMemoryMedHistory
        >>> manager = RetentionManager(
        ...     policy=RetentionPolicy(retention_days=365, grace_days=7),
        ...     list_sessions=lambda: ["med:chat:hosp-a:cardio:s-1"],
        ...     make_history=lambda key: InMemoryMedHistory(
        ...         session_id="s-1", tenant_id="hosp-a", dept_id="cardio", patient_id="p-1",
        ...     ),
        ... )
        >>> report = manager.run()
    """

    def __init__(
        self,
        *,
        policy: RetentionPolicy,
        list_sessions: Callable[[], list[str]],
        make_history: Callable[[str], MedChatMessageHistory],
        after_purge: Callable[[str], None] | None = None,
    ) -> None:
        """初始化调度器。

        Args:
            policy: 保留期策略。
            list_sessions: 枚举候选会话统一存储键的回调（建议仅枚举 ARCHIVED/DELETED 会话）。
            make_history: 按存储键构造会话历史句柄的回调。
            after_purge: 物理清理成功后的钩子（如清理归档索引），``None`` 表示无。

        Raises:
            ValueError: ``policy`` 非 :class:`RetentionPolicy` 实例或任一回调非可调用时。
        """
        if not isinstance(policy, RetentionPolicy):
            raise ValueError("policy must be a RetentionPolicy instance")
        if not callable(list_sessions) or not callable(make_history):
            raise ValueError("list_sessions and make_history must be callable")
        self._policy = policy
        self._list_sessions = list_sessions
        self._make_history = make_history
        self._after_purge = after_purge

    def soft_delete(self, history: MedChatMessageHistory) -> SoftDeleteResult:
        """对单个 ``ARCHIVED`` 会话执行软删除标记（公开封装，供外部按需调用）。

        Args:
            history: 会话历史句柄。

        Returns:
            软删除结果。
        """
        return soft_delete_session(history)

    def purge(
        self,
        history: MedChatMessageHistory,
        *,
        now_ms: int | None = None,
    ) -> PurgeResult:
        """对单个 ``DELETED`` 会话执行物理清理（公开封装，供外部按需调用）。

        Args:
            history: 会话历史句柄。
            now_ms: 当前 epoch 毫秒；``None`` 表示取系统时间。

        Returns:
            物理清理结果。
        """
        return purge_expired_session(history, self._policy, now_ms=now_ms)

    def run(self, *, now_ms: int | None = None) -> RetentionReport:
        """扫描候选会话，自动完成到期软删除与到期物理清理两阶段。

        对单会话异常（构造失败 / 状态流转失败等）容忍：记录进报告并继续处理后续会话。

        Args:
            now_ms: 判定期限使用的当前 epoch 毫秒；``None`` 表示取系统时间。

        Returns:
            调度报告 :class:`RetentionReport`，含扫描数、软删除数、清理数、跳过数、
            失败数及逐会话明细。
        """
        now = now_millis() if now_ms is None else now_ms
        report = RetentionReport()
        keys = self._list_sessions()
        report.scanned = len(keys)
        for key in keys:
            try:
                history = self._make_history(key)
                meta = history.session_meta
                if meta.status is SessionStatus.ARCHIVED:
                    if self._policy.is_due_for_soft_delete(meta, now):
                        res = self.soft_delete(history)
                        item = SessionRetentionResult(
                            session_key=res.session_key,
                            soft_deleted=res.deleted,
                            reason=res.reason,
                        )
                        if res.deleted:
                            report.soft_deleted += 1
                        else:
                            report.skipped += 1
                    else:
                        item = SessionRetentionResult(session_key=key, reason="retention_not_elapsed")
                        report.skipped += 1
                elif meta.status is SessionStatus.DELETED:
                    if self._policy.is_due_for_purge(meta, now):
                        res = self.purge(history, now_ms=now)
                        item = SessionRetentionResult(
                            session_key=res.session_key,
                            purged=res.purged,
                            purged_count=res.purged_count,
                            reason=res.reason,
                        )
                        if res.purged:
                            report.purged += 1
                            if self._after_purge is not None:
                                self._after_purge(key)
                        else:
                            report.skipped += 1
                    else:
                        item = SessionRetentionResult(session_key=key, reason="grace_not_elapsed")
                        report.skipped += 1
                else:
                    item = SessionRetentionResult(session_key=key, reason=f"status:{meta.status.value}")
                    report.skipped += 1
            except Exception as exc:  # 容忍单会话失败，保证调度整体推进
                report.failed += 1
                report.errors.append(f"{key}: {exc}")
                item = SessionRetentionResult(session_key=key, error=str(exc))
            report.results.append(item)
        return report
