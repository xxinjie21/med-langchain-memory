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
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from med_langchain_memory.exceptions import ValidationError

from .shards import resolve_shard, select_shards
from .trend import (
    DEFAULT_REPORTS_DIR,
    DEFAULT_SLOWEST_LIMIT,
    JUNIT_REPORT_NAME,
    CaseResult,
    SuiteReport,
    parse_junit_report,
)

#: artifact 下载后的目录名前缀（与工作流 ``upload-artifact`` 的 ``name:`` 一致）。
ARTIFACT_PREFIX = "integration-report-"

#: 聚合摘要文件名（汇总作业写进 Job Summary 的那份）。
AGGREGATE_SUMMARY_NAME = "integration-aggregate.md"

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


def render_markdown(report: AggregateReport, *, slowest_limit: int = DEFAULT_SLOWEST_LIMIT) -> str:
    """把聚合报告渲染成 Markdown（可直接追加进 ``$GITHUB_STEP_SUMMARY``）。

    Args:
        report: 聚合报告。
        slowest_limit: 「耗时最长的用例」小节展示条数。

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
        配置好 ``--reports-dir`` / ``--shard`` / ``--output`` / ``--slowest`` 的解析器。
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：读分片报告 → 渲染汇总 → 写文件或标准输出 → 返回整轮结论。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        见模块文档的退出码表。汇总 Markdown **先写后判**，因此即使退出码非 0，
        ``--output`` 指定的文件也已就绪，供工作流的 ``if: always()`` 步骤贴进 Job Summary。
    """
    args = build_parser().parse_args(argv)
    try:
        names = tuple(shard.name for shard in select_shards(args.shard))
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    report = build_aggregate(collect_shard_reports(args.reports_dir, names))
    markdown = render_markdown(report, slowest_limit=args.slowest)
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    if not report.complete:
        return EXIT_INCOMPLETE
    return EXIT_FAILURES if report.has_failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
