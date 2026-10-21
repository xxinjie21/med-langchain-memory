"""集成测试分片结果聚合：把四个分片的报告合成一份「一眼看全貌」的结论。

动机：D43 把真机套件切成四个矩阵分片后，Job Summary、artifact 与缓存也都变成四份——
想知道「今晚到底绿没绿、是哪个后端挂了」得点开四个作业分别看。本模块把四个分片的
JUnit 报告聚合成一份汇总（分片明细表 + 合计行 + 缺失报告清单 + 失败用例清单 +
最慢用例），并给出**单一退出码**作为整轮集成测试的结论。

设计取舍：

* 只依赖标准库（外加同层只依赖标准库的 :mod:`med_langchain_memory.testing.shards` /
  :mod:`med_langchain_memory.testing.trend`）：用例明细的解析口径完全复用
  :func:`med_langchain_memory.testing.trend.parse_junit_report`，本模块不重复实现
  一遍 XML 解析，也不信任 ``<testsuite>`` 上的 ``tests=`` / ``failures=`` 属性；
* 期望看到哪些分片**由分片表决定**（``testing.shards.shard_names()``），
  与工作流矩阵取值同源，工作流不额外声明一份分片清单；
* 兼容两种报告目录布局：本地跑出来的 ``reports/<分片>/integration.xml``，
  以及 ``actions/download-artifact`` 解出来的
  ``reports/integration-report-<分片>/integration.xml``（目录名即 artifact 名）；
* **总账也能和上一轮比**：D41 的趋势对比只到「单个分片 vs 该分片上一次」这一粒度，
  要回答「今晚整轮比昨晚好了还是差了、是哪一片退化了」还得自己把四份摘要拼起来。
  因此本模块在聚合之上再加一层 :class:`AggregateTrend`：拿 ``--baseline-dir`` 里的
  上一轮分片报告算出**整轮合计的变化**（用例数 / 失败数 / 通过率 / 耗时）、
  **逐分片变化**（失败数与通过率的变化 + 状态标记）以及跨分片的
  **新增失败 / 已修复**清单；``--snapshot-dir`` 把本轮报告按规范布局存一份，
  工作流用滚动缓存把它喂给下一轮，于是对比不需要任何外部存储；
* **两套基线合一**（D47）：分片粒度的用例级趋势原先由各分片作业用
  :mod:`med_langchain_memory.testing.trend` 的 CLI 各自维护——每个分片各滚一份
  ``actions/cache`` 基线，缓存与口径都有两份。现在趋势整体上收汇总作业：
  本模块的 :attr:`AggregateTrend.shard_trends` 用**同一份** ``baseline/`` 快照
  逐分片构建 :class:`~med_langchain_memory.testing.trend.TrendReport`
  （口径与 ``trend`` 逐字一致），于是整轮总账、逐分片状态与逐分片用例级趋势
  共享一套滚动基线；``trend`` 的 CLI 保留给本地单报告场景，工作流不再按分片
  缓存报告，也不再调用趋势 CLI；
* 只做计数、排序与聚合，不读取、不处理任何测试输出文本内容。

退出码（汇总作业的整轮结论）：

===========  ==========================================================
退出码        含义
===========  ==========================================================
``0``        所有分片报告齐备且没有失败 / 错误用例
``1``        有分片报告缺失或无法解析（CI 基础设施问题，禁止静默绿灯）
``2``        报告齐备但存在失败 / 错误用例
``3``        ``--shard`` 传了未知分片名（用法错误）
===========  ==========================================================

无论退出码是多少，只要命令行能正常解析，汇总 Markdown 都会被写出来——
汇总步骤与退出码解耦，失败信息才不会因为「步骤中断」而丢失。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from med_langchain_memory.exceptions import ValidationError

from .shards import resolve_shard, select_shards
from .trend import (
    DEFAULT_REPORTS_DIR,
    DEFAULT_SLOWEST_LIMIT,
    JUNIT_REPORT_NAME,
    CaseResult,
    SuiteReport,
    TrendReport,
    build_trend,
    parse_junit_report,
)

#: artifact 下载后的目录名前缀（与工作流 ``upload-artifact`` 的 ``name:`` 一致）。
ARTIFACT_PREFIX = "integration-report-"

#: 汇总作业上传的「账本」artifact 名：一个包含本轮分片报告与上一轮基线报告的包，
#: 供 ``dashboard`` 作业渲染看板页（D48）。名字由本模块给出，工作流与测试共用。
LEDGER_ARTIFACT_NAME = "integration-ledger"

#: 聚合摘要文件名（汇总作业写进 Job Summary 的那份）。
AGGREGATE_SUMMARY_NAME = "integration-aggregate.md"

#: 上一轮总账基线的默认目录（汇总作业从滚动缓存恢复到这里，并在这里写本轮快照）。
BASELINE_LEDGER_DIR = "baseline"

#: 单个分片相对上一轮基线的状态。
ShardDeltaState = Literal["absent", "added", "removed", "regressed", "improved", "unchanged"]

#: 退出码：报告齐备且无失败用例。
EXIT_OK = 0

#: 退出码：有分片报告缺失或无法解析。
EXIT_INCOMPLETE = 1

#: 退出码：报告齐备但存在失败 / 错误用例。
EXIT_FAILURES = 2

#: 退出码：``--shard`` 传了未知分片名。
EXIT_USAGE = 3


@dataclass(frozen=True)
class ShardStatus:
    """单个分片的报告状态（报告可用或不可用都算一种状态）。"""

    #: 分片名（见 :func:`med_langchain_memory.testing.shards.shard_names`）。
    name: str

    #: 人类可读的分片标题。
    title: str

    #: 探测到的报告路径；报告不存在时是规范位置 ``<reports>/<分片>/integration.xml``。
    path: Path

    #: 解析出的报告；缺失或解析失败时为 ``None``。
    report: SuiteReport | None = None

    #: 不可用原因（缺失 / XML 非法）；报告可用时为 ``None``。
    error: str | None = None

    @property
    def present(self) -> bool:
        """报告是否已成功解析。"""
        return self.report is not None

    @property
    def reason(self) -> str:
        """不可用原因；报告可用时返回空串。"""
        if self.report is not None:
            return ""
        return self.error or f"report not found: {self.path}"


def shard_report_path(reports_dir: Path, name: str) -> Path:
    """按两种目录布局探测某个分片的 JUnit 报告路径。

    Args:
        reports_dir: 报告根目录（工作流里是 ``reports``）。
        name: 分片名。

    Returns:
        存在的报告路径；两种布局都不存在时返回规范位置
        ``reports_dir / name / integration.xml``（便于错误信息直接指路）。
    """
    canonical = reports_dir / name / JUNIT_REPORT_NAME
    downloaded = reports_dir / f"{ARTIFACT_PREFIX}{name}" / JUNIT_REPORT_NAME
    for candidate in (canonical, downloaded):
        if candidate.is_file():
            return candidate
    return canonical


def collect_shard_reports(
    reports_dir: Path = Path(DEFAULT_REPORTS_DIR),
    shards: Sequence[str] | None = None,
) -> tuple[ShardStatus, ...]:
    """读取每个分片的 JUnit 报告（缺失与解析失败都不抛异常，记进状态里）。

    Args:
        reports_dir: 报告根目录。
        shards: 分片名序列；``None`` / 空序列 / 含 ``all`` 时取全部分片。

    Returns:
        与 :func:`~med_langchain_memory.testing.shards.shard_names` 同序的状态元组。

    Raises:
        ValidationError: ``shards`` 含未知分片名时。
    """
    statuses: list[ShardStatus] = []
    for name in (shard.name for shard in select_shards(shards)):
        shard = resolve_shard(name)
        path = shard_report_path(reports_dir, name)
        report: SuiteReport | None = None
        error: str | None = None
        if path.is_file():
            try:
                report = parse_junit_report(path)
            except ValidationError as exc:
                error = str(exc)
        else:
            error = f"report not found: {path}"
        statuses.append(
            ShardStatus(name=name, title=shard.title, path=path, report=report, error=error)
        )
    return tuple(statuses)


def snapshot_reports(
    reports_dir: Path,
    target_dir: Path,
    shards: Sequence[str] | None = None,
) -> tuple[Path, ...]:
    """把本轮各分片的报告按规范布局复制一份到 ``target_dir``（供下一轮当基线）。

    与 :func:`collect_shard_reports` 一样兼容两种来源布局，但**只按规范布局写出**
    ``<target_dir>/<分片>/integration.xml``——基线目录因此始终是单一定型结构，
    读取方（``--baseline-dir``）不必再猜布局。

    Args:
        reports_dir: 本轮报告根目录。
        target_dir: 目标目录（工作流里是 :data:`BASELINE_LEDGER_DIR`）。
        shards: 分片名序列；``None`` / 空序列 / 含 ``all`` 时取全部分片。

    Returns:
        实际写出的文件路径元组；只包含成功解析的报告，顺序与分片表一致。

    Raises:
        ValidationError: ``shards`` 含未知分片名时。
    """
    written: list[Path] = []
    for status in collect_shard_reports(reports_dir, shards):
        if not status.present:
            continue
        target = target_dir / status.name / JUNIT_REPORT_NAME
        if status.path.resolve() == target.resolve():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(status.path, target)
        written.append(target)
    return tuple(written)


@dataclass(frozen=True)
class AggregateReport:
    """全部分片的聚合结果。"""

    #: 各分片状态（顺序即分片表顺序）。
    statuses: tuple[ShardStatus, ...]

    @property
    def reports(self) -> tuple[SuiteReport, ...]:
        """已成功解析的报告（按分片表顺序）。"""
        return tuple(status.report for status in self.statuses if status.report is not None)

    @property
    def missing(self) -> tuple[ShardStatus, ...]:
        """缺失或无法解析的分片状态。"""
        return tuple(status for status in self.statuses if not status.present)

    @property
    def complete(self) -> bool:
        """是否每个分片都有可用报告。"""
        return not self.missing

    @property
    def total(self) -> int:
        """用例总数（含跳过；只统计已解析的报告）。"""
        return sum(report.total for report in self.reports)

    @property
    def passed(self) -> int:
        """通过用例数。"""
        return sum(report.passed for report in self.reports)

    @property
    def failed(self) -> int:
        """失败 + 错误用例数。"""
        return sum(report.failed for report in self.reports)

    @property
    def skipped(self) -> int:
        """跳过用例数。"""
        return sum(report.skipped for report in self.reports)

    @property
    def duration(self) -> float:
        """全部用例耗时之和（秒，保留 3 位）。"""
        return round(sum(report.duration for report in self.reports), 3)

    @property
    def pass_rate(self) -> float:
        """通过率 = 通过 / （总数 - 跳过）；没有可执行用例时返回 ``0.0``。"""
        effective = self.total - self.skipped
        return round(self.passed / effective, 4) if effective > 0 else 0.0

    @property
    def has_failures(self) -> bool:
        """是否存在失败或错误用例。"""
        return any(report.has_failures for report in self.reports)

    def failures(self) -> tuple[tuple[str, CaseResult], ...]:
        """返回全部失败 / 错误用例（``(分片名, 用例)`` 二元组）。"""
        return tuple(
            (status.name, case)
            for status in self.statuses
            if status.report is not None
            for case in status.report.failures()
        )

    def slowest(self, limit: int = DEFAULT_SLOWEST_LIMIT) -> tuple[tuple[str, CaseResult], ...]:
        """返回耗时最长的前 ``limit`` 个用例（``(分片名, 用例)`` 二元组，耗时降序）。

        Args:
            limit: 展示条数；``<= 0`` 时返回空元组。

        Returns:
            二元组元组；耗时相同时保持分片表与报告内的原始顺序（``sorted`` 稳定）。
        """
        pairs = [
            (status.name, case)
            for status in self.statuses
            if status.report is not None
            for case in status.report.cases
        ]
        ordered = sorted(pairs, key=lambda pair: pair[1].duration, reverse=True)
        return tuple(ordered[: max(limit, 0)])


def build_aggregate(statuses: Sequence[ShardStatus]) -> AggregateReport:
    """组装聚合报告。

    Args:
        statuses: 各分片的报告状态。

    Returns:
        :class:`AggregateReport`。
    """
    return AggregateReport(statuses=tuple(statuses))


@dataclass(frozen=True)
class ShardDelta:
    """单个分片在本轮与上一轮基线之间的差异。"""

    #: 分片名。
    name: str

    #: 人类可读的分片标题。
    title: str

    #: 本轮该分片的报告；本轮缺报告时为 ``None``。
    current: SuiteReport | None = None

    #: 基线里该分片的报告；基线缺该片时为 ``None``。
    baseline: SuiteReport | None = None

    @property
    def comparable(self) -> bool:
        """本轮与基线都有该分片的报告时才可比较。"""
        return self.current is not None and self.baseline is not None

    @property
    def state(self) -> ShardDeltaState:
        """分片状态：新增 / 缺报告 / 回退 / 改善 / 持平。

        判定顺序：先看两侧是否都有报告，再比失败（错误）用例数，
        失败数相同则比通过率（用例增删导致的通过率变化也算变化）。
        """
        current = self.current
        baseline = self.baseline
        if current is None and baseline is None:
            return "absent"
        if baseline is None:
            return "added"
        if current is None:
            return "removed"
        if current.failed != baseline.failed:
            return "regressed" if current.failed > baseline.failed else "improved"
        if current.pass_rate != baseline.pass_rate:
            return "improved" if current.pass_rate > baseline.pass_rate else "regressed"
        return "unchanged"

    @property
    def failed_delta(self) -> int | None:
        """失败 + 错误用例数的变化；不可比较时为 ``None``。"""
        if self.current is None or self.baseline is None:
            return None
        return self.current.failed - self.baseline.failed

    @property
    def total_delta(self) -> int | None:
        """用例总数的变化；不可比较时为 ``None``。"""
        if self.current is None or self.baseline is None:
            return None
        return self.current.total - self.baseline.total

    @property
    def pass_rate_delta(self) -> float | None:
        """通过率的变化（本轮 - 基线，保留 4 位）；不可比较时为 ``None``。"""
        if self.current is None or self.baseline is None:
            return None
        return round(self.current.pass_rate - self.baseline.pass_rate, 4)


@dataclass(frozen=True)
class ShardTrend:
    """单个分片相对其上一轮基线的用例级趋势。

    口径与 :mod:`med_langchain_memory.testing.trend` 的 :class:`TrendReport`
    完全一致（由 :func:`~med_langchain_memory.testing.trend.build_trend` 构建）：
    D47 之前，分片粒度的用例级趋势由各分片作业用 ``trend`` CLI 各自维护
    （每个分片各滚一份 ``actions/cache`` 基线）；现在两套基线合一——汇总作业
    用同一份 ``baseline/`` 快照逐分片重建趋势，缓存与口径都只有一处。
    """

    #: 分片名。
    name: str

    #: 人类可读的分片标题。
    title: str

    #: 该分片「本轮 vs 基线」的趋势（两侧报告均由调用方保证存在）。
    trend: TrendReport

    @property
    def new_failures(self) -> tuple[CaseResult, ...]:
        """本轮失败、基线未失败的用例。"""
        return self.trend.new_failures

    @property
    def fixed(self) -> tuple[CaseResult, ...]:
        """基线失败、本轮不再失败的用例。"""
        return self.trend.fixed

    @property
    def pass_rate_delta(self) -> float:
        """通过率的变化（本轮 - 基线，保留 4 位）。"""
        return self.trend.pass_rate_delta


@dataclass(frozen=True)
class AggregateTrend:
    """整轮总账相对上一轮基线的对比结果。

    基线的粒度就是「上一轮的四个分片报告」：既给出整轮合计的变化，也给出逐分片的变化，
    因此「今晚比昨晚差了没有、是哪一片退化了」一眼可见（单分片粒度的趋势见
    :mod:`med_langchain_memory.testing.trend`）。

    基线不齐（上一轮有分片在 pytest 之前就挂了，没产出 JUnit）时对比**仍然可用**，
    只是只覆盖有报告的那些分片；此时 :attr:`baseline_complete` 为 ``False``，
    渲染出的对比小节会显式列出缺哪几片，避免把「少了一片」误读成「用例变少了」。

    分片粒度的**用例级**趋势（新增失败 / 已修复 / 通过率变化）见
    :attr:`shard_trends`：它与 :attr:`shards` 的状态表、跨分片清单共用同一份基线，
    口径与 :mod:`med_langchain_memory.testing.trend` 逐字一致。
    """

    #: 本轮总账。
    current: AggregateReport

    #: 上一轮总账；``None`` 表示首次运行，无对比基线。
    baseline: AggregateReport | None = None

    @property
    def available(self) -> bool:
        """是否存在可用基线（至少解析出一个基线分片报告）。"""
        baseline = self.baseline
        return baseline is not None and bool(baseline.reports)

    @property
    def baseline_complete(self) -> bool:
        """基线是否每个分片的报告都齐备（无基线时为 ``False``）。"""
        baseline = self.baseline
        return baseline is not None and baseline.complete

    @property
    def shards(self) -> tuple[ShardDelta, ...]:
        """逐分片差异（顺序即分片表顺序）；无可用基线时为空元组。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return ()
        known = {status.name: status.report for status in baseline.statuses}
        return tuple(
            ShardDelta(
                name=status.name,
                title=status.title,
                current=status.report,
                baseline=known.get(status.name),
            )
            for status in self.current.statuses
        )

    @property
    def shard_trends(self) -> tuple[ShardTrend, ...]:
        """逐分片用例级趋势（口径复用 :func:`trend.build_trend`）。

        只覆盖**本轮与基线都有报告**的分片；任一侧缺报告时该片不出现在这里
        （缺报告的事实由 :attr:`shards` 的状态表表达）。无可用基线时为空元组。
        """
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return ()
        known = {status.name: status.report for status in baseline.statuses}
        trends: list[ShardTrend] = []
        for status in self.current.statuses:
            report = status.report
            base = known.get(status.name)
            if report is None or base is None:
                continue
            trends.append(
                ShardTrend(
                    name=status.name,
                    title=status.title,
                    trend=build_trend(report, base),
                )
            )
        return tuple(trends)

    @property
    def new_failures(self) -> tuple[tuple[str, CaseResult], ...]:
        """本轮失败、但基线未失败的用例（``(分片名, 用例)``）；无基线时为空。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return ()
        known = {(shard, case.node_id) for shard, case in baseline.failures()}
        return tuple(
            pair for pair in self.current.failures() if (pair[0], pair[1].node_id) not in known
        )

    @property
    def fixed(self) -> tuple[tuple[str, CaseResult], ...]:
        """基线失败、本轮不再失败的用例（``(分片名, 用例)``）；无基线时为空。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return ()
        failing_now = {(shard, case.node_id) for shard, case in self.current.failures()}
        return tuple(
            pair for pair in baseline.failures() if (pair[0], pair[1].node_id) not in failing_now
        )

    @property
    def total_delta(self) -> int | None:
        """整轮用例总数的变化；无基线时为 ``None``。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return None
        return self.current.total - baseline.total

    @property
    def failed_delta(self) -> int | None:
        """整轮失败 + 错误用例数的变化；无基线时为 ``None``。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return None
        return self.current.failed - baseline.failed

    @property
    def pass_rate_delta(self) -> float | None:
        """整轮通过率的变化（本轮 - 基线，保留 4 位）；无基线时为 ``None``。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return None
        return round(self.current.pass_rate - baseline.pass_rate, 4)

    @property
    def duration_delta(self) -> float | None:
        """整轮耗时的变化（秒，保留 3 位）；无基线时为 ``None``。"""
        baseline = self.baseline
        if baseline is None or not baseline.reports:
            return None
        return round(self.current.duration - baseline.duration, 3)


def build_aggregate_trend(
    current: AggregateReport, baseline: AggregateReport | None = None
) -> AggregateTrend:
    """组装总账对比。

    Args:
        current: 本轮总账。
        baseline: 上一轮总账；``None`` 表示首次运行。

    Returns:
        :class:`AggregateTrend`。
    """
    return AggregateTrend(current=current, baseline=baseline)


def _section(title: str, bullets: Sequence[str], *, empty: str) -> list[str]:
    """把一组条目渲染成 Markdown 小节（为空时输出 ``empty`` 占位句）。"""
    lines = [f"### {title}", ""]
    lines.extend(bullets if bullets else [empty])
    lines.append("")
    return lines


def _shard_row(status: ShardStatus) -> str:
    """渲染分片明细表的一行；无报告时整行标红提示。"""
    report = status.report
    if report is None:
        return f"| `{status.name}` | ⚠️ 无报告 | — | — | — | — | — |"
    return (
        f"| `{status.name}` | {report.total} | {report.passed} | {report.failed} "
        f"| {report.skipped} | {report.pass_rate:.2%} | {report.duration:.3f}s |"
    )


def _case_lines(pairs: Sequence[tuple[str, CaseResult]], *, with_duration: bool) -> list[str]:
    """渲染 ``(分片名, 用例)`` 列表（带分片前缀，便于一眼定位是哪个后端）。"""
    lines = []
    for shard, case in pairs:
        suffix = f" — {case.duration:.3f}s" if with_duration else ""
        lines.append(f"- `{shard}` · `{case.node_id}`{suffix}")
    return lines


#: 分片状态在对比表里的中文标记。
_STATE_LABELS: dict[ShardDeltaState, str] = {
    "absent": "—",
    "added": "新增",
    "removed": "⚠️ 缺报告",
    "regressed": "⚠️ 回退",
    "improved": "✅ 改善",
    "unchanged": "持平",
}


def _delta_text(value: int | float | None, *, percent: bool = False, unit: str = "") -> str:
    """渲染带符号的变化值；``None``（不可比较）渲染成 ``—``。

    Args:
        value: 变化量；``None`` 表示两侧不可比（缺报告）。
        percent: 按百分比渲染（保留 2 位）。
        unit: 非空时按 3 位小数加单位渲染（如 ``s``）。
    """
    if value is None:
        return "—"
    if percent:
        return f"{value:+.2%}"
    if unit:
        return f"{value:+.3f}{unit}"
    return f"{value:+d}"


def _progress(report: SuiteReport) -> str:
    """渲染「通过 / 可执行」进度（分母排除跳过用例）。"""
    return f"{report.passed}/{report.total - report.skipped}"


def _delta_row(delta: ShardDelta) -> str:
    """渲染逐分片对比表的一行（任一侧缺报告时变化列渲染成 ``—``，不伪造 0）。"""
    current = delta.current
    baseline = delta.baseline
    current_cell = "—" if current is None else _progress(current)
    baseline_cell = "—" if baseline is None else _progress(baseline)
    failed = _delta_text(delta.failed_delta)
    rate = _delta_text(delta.pass_rate_delta, percent=True)
    return (
        f"| `{delta.name}` | {current_cell} | {baseline_cell} | {failed} | {rate} "
        f"| {_STATE_LABELS[delta.state]} |"
    )


def _shard_trend_section(shard_trends: Sequence[ShardTrend]) -> list[str]:
    """渲染逐分片用例级趋势小节；无可比较分片时返回空列表（小节整体不出现）。"""
    if not shard_trends:
        return []
    lines = ["### 逐分片趋势（与上一轮的自己相比）", ""]
    for shard in shard_trends:
        lines.append(
            f"- `{shard.name}` · 通过率 {shard.pass_rate_delta:+.2%} · "
            f"新增失败 {len(shard.new_failures)} · 已修复 {len(shard.fixed)}"
        )
        lines.extend(f"  - 新增：`{case.node_id}`" for case in shard.new_failures)
        lines.extend(f"  - 修复：`{case.node_id}`" for case in shard.fixed)
    lines.append("")
    return lines


def _comparison_section(trend: AggregateTrend) -> list[str]:
    """渲染「与上一轮对比」小节；无可用基线时返回空列表（首次运行不出现该小节）。"""
    baseline = trend.baseline
    if baseline is None or not baseline.reports:
        return []
    current = trend.current
    lines = [
        "### 与上一轮对比",
        "",
        "| 指标 | 本次 | 基线 | 变化 |",
        "|---|---|---|---|",
        f"| 用例总数 | {current.total} | {baseline.total} | {_delta_text(trend.total_delta)} |",
        f"| 通过 | {current.passed} | {baseline.passed} | "
        f"{_delta_text(current.passed - baseline.passed)} |",
        f"| 失败/错误 | {current.failed} | {baseline.failed} | {_delta_text(trend.failed_delta)} |",
        f"| 跳过 | {current.skipped} | {baseline.skipped} | "
        f"{_delta_text(current.skipped - baseline.skipped)} |",
        f"| 通过率 | {current.pass_rate:.2%} | {baseline.pass_rate:.2%} | "
        f"{_delta_text(trend.pass_rate_delta, percent=True)} |",
        f"| 总耗时 | {current.duration:.3f}s | {baseline.duration:.3f}s | "
        f"{_delta_text(trend.duration_delta, unit='s')} |",
        "",
    ]
    if trend.baseline_complete:
        lines += [
            f"基线报告齐备（{len(baseline.reports)}/{len(baseline.statuses)} 个分片），"
            f"通过率变化 **{_delta_text(trend.pass_rate_delta, percent=True)}**",
            "",
        ]
    else:
        missing = ", ".join(status.name for status in baseline.missing)
        lines += [
            f"⚠️ 基线只有 {len(baseline.reports)}/{len(baseline.statuses)} 个分片有报告"
            f"（缺：{missing}），对比只覆盖有报告的分片",
            "",
        ]
    lines += [
        "| 分片 | 本次 通过/可执行 | 基线 通过/可执行 | 失败变化 | 通过率变化 | 状态 |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_delta_row(delta) for delta in trend.shards)
    lines.append("")
    lines += _shard_trend_section(trend.shard_trends)
    lines += _section(
        "本轮新增失败",
        _case_lines(trend.new_failures, with_duration=False),
        empty="无（与基线相比没有新增失败用例）",
    )
    lines += _section(
        "已修复", _case_lines(trend.fixed, with_duration=False), empty="无（基线里没有失败用例）"
    )
    return lines


def render_markdown(
    report: AggregateReport,
    *,
    slowest_limit: int = DEFAULT_SLOWEST_LIMIT,
    baseline: AggregateReport | None = None,
) -> str:
    """把聚合报告渲染成 Markdown（可直接追加进 ``$GITHUB_STEP_SUMMARY``）。

    Args:
        report: 本轮聚合报告。
        slowest_limit: 「耗时最长的用例」小节展示条数。
        baseline: 上一轮聚合报告；``None`` 或其中一份报告都没解析出来时
            不渲染「与上一轮对比」小节（首次运行与缓存未命中都是这种情形）。

    Returns:
        以换行结尾的 Markdown 文本。
    """
    total_shards = len(report.statuses)
    present = len(report.reports)
    if not report.complete:
        headline = f"报告不齐（{len(report.missing)}/{total_shards} 个分片缺报告）"
    elif report.has_failures:
        headline = "存在失败"
    else:
        headline = "全部通过"
    lines = [
        "# 集成测试分片聚合",
        "",
        f"结果：**{headline}**"
        f"（{report.passed}/{report.total - report.skipped} 通过，{report.skipped} 跳过，"
        f"{present}/{total_shards} 个分片有报告）",
        "",
        "| 分片 | 用例 | 通过 | 失败/错误 | 跳过 | 通过率 | 耗时 |",
        "|---|---|---|---|---|---|---|",
    ]
    lines.extend(_shard_row(status) for status in report.statuses)
    lines.append(
        f"| **合计** | {report.total} | {report.passed} | {report.failed} "
        f"| {report.skipped} | {report.pass_rate:.2%} | {report.duration:.3f}s |"
    )
    lines.append("")
    lines += _comparison_section(build_aggregate_trend(report, baseline))
    lines += _section(
        "缺失或无法解析的报告",
        [f"- `{status.name}`：{status.reason}" for status in report.missing],
        empty="无（每个分片的报告都齐备）",
    )
    lines += _section(
        "失败用例",
        _case_lines(report.failures(), with_duration=False),
        empty="无（已解析的报告里没有失败用例）",
    )
    lines += _section(
        "耗时最长的用例",
        _case_lines(report.slowest(slowest_limit), with_duration=True),
        empty="无用例明细",
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（供 CI 与本地直接调用）。

    Returns:
        配置好 ``--reports-dir`` / ``--shard`` / ``--output`` / ``--slowest``
        / ``--baseline-dir`` / ``--snapshot-dir`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="med-memory-aggregate",
        description="Aggregate per-shard JUnit reports into a single integration verdict.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(DEFAULT_REPORTS_DIR),
        help=f"root directory holding per-shard reports (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--shard",
        action="append",
        default=[],
        metavar="NAME",
        help="shard to include; repeatable; omit or pass 'all' for every shard",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="write the Markdown summary to this path instead of stdout",
    )
    parser.add_argument(
        "--slowest",
        type=int,
        default=DEFAULT_SLOWEST_LIMIT,
        help=f"how many slow cases to list (default: {DEFAULT_SLOWEST_LIMIT})",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=None,
        help="directory holding the previous run's shard reports; enables the ledger comparison",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=None,
        help="copy this run's shard reports here so the next run can diff against them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：读分片报告 → 渲染汇总 → 写文件或标准输出 → 快照本轮 → 返回整轮结论。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        见模块文档的退出码表。汇总 Markdown **先写后判**，因此即使退出码非 0，
        ``--output`` 指定的文件也已就绪，供工作流的 ``if: always()`` 步骤贴进 Job Summary；
        ``--snapshot-dir`` 的快照同样在返回退出码之前完成，保证「报告缺失」这类
        基础设施故障不会连带把基线链断掉。
    """
    args = build_parser().parse_args(argv)
    try:
        names = tuple(shard.name for shard in select_shards(args.shard))
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = build_aggregate(collect_shard_reports(args.reports_dir, names))
    baseline = (
        build_aggregate(collect_shard_reports(args.baseline_dir, names))
        if args.baseline_dir is not None
        else None
    )
    markdown = render_markdown(report, slowest_limit=args.slowest, baseline=baseline)
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if args.snapshot_dir is not None:
        snapshot_reports(args.reports_dir, args.snapshot_dir, names)
    if not report.complete:
        return EXIT_INCOMPLETE
    return EXIT_FAILURES if report.has_failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
