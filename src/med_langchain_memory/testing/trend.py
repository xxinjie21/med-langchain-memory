"""集成测试夜间 CI 的 JUnit 报告汇总与趋势对比（纯标准库，不参与生产链路）。

定位：夜间工作流 ``.github/workflows/integration.yml`` 每次运行都会产出
``pytest --junit-xml`` 报告；本模块把它解析成结构化摘要、渲染成 Markdown
写进 GitHub Actions 的 Job Summary，并在拿到上一次运行的报告时给出
「新增失败 / 已修复 / 通过率变化」的趋势对比。

设计取舍：

* 只依赖标准库 ``xml.etree.ElementTree``，**不引入任何第三方报告解析库**；
* 汇总口径（用例数 / 通过 / 失败 / 跳过 / 通过率）全部**从 ``<testcase>`` 子节点推导**，
  不信任 ``<testsuite>`` 上的 ``tests=`` / ``failures=`` 属性——两者不一致时以用例明细为准；
* 只做计数与排序，不读取、不处理任何测试输出文本内容。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from med_langchain_memory.exceptions import ValidationError

#: JUnit 报告文件名（工作流里 ``pytest --junit-xml`` 的目标）。
JUNIT_REPORT_NAME = "integration.xml"

#: 上一次运行的报告在本次运行中的固定路径（工作流从缓存恢复后改名而来）。
BASELINE_REPORT_NAME = "previous.xml"

#: 渲染出的 Markdown 摘要文件名。
TREND_SUMMARY_NAME = "integration-trend.md"

#: 报告与摘要的默认输出目录（工作流里与缓存路径一致）。
DEFAULT_REPORTS_DIR = "reports"

#: 摘要里「耗时最长的用例」默认展示条数。
DEFAULT_SLOWEST_LIMIT = 5

#: 单个测试用例归一化后的结果。
Outcome = Literal["passed", "failed", "error", "skipped"]

#: 视为「不通过」的结果（``skipped`` 不算失败）。
_FAILING: frozenset[Outcome] = frozenset({"failed", "error"})


@dataclass(frozen=True)
class CaseResult:
    """单个测试用例的结果。"""

    #: 用例所属类 / 模块（JUnit ``classname`` 属性）。
    classname: str

    #: 用例名（JUnit ``name`` 属性）。
    name: str

    #: 归一化结果。
    outcome: Outcome

    #: 耗时（秒）。
    duration: float

    @property
    def node_id(self) -> str:
        """返回 ``classname::name`` 形式的节点标识（无 classname 时只返回 name）。"""
        return f"{self.classname}::{self.name}" if self.classname else self.name

    @property
    def failing(self) -> bool:
        """是否为失败或错误（``skipped`` 不算）。"""
        return self.outcome in _FAILING


@dataclass(frozen=True)
class SuiteReport:
    """一次 pytest 运行的汇总（对应一个 JUnit ``<testsuite>``）。"""

    #: 报告名，用于标题展示。
    name: str

    #: 全部用例明细。
    cases: tuple[CaseResult, ...]

    @property
    def total(self) -> int:
        """用例总数（含跳过）。"""
        return len(self.cases)

    def count(self, outcome: Outcome) -> int:
        """返回指定结果的用例数。

        Args:
            outcome: 目标结果。

        Returns:
            结果等于 ``outcome`` 的用例条数。
        """
        return sum(1 for case in self.cases if case.outcome == outcome)

    @property
    def passed(self) -> int:
        """通过用例数。"""
        return self.count("passed")

    @property
    def failed(self) -> int:
        """失败 + 错误用例数。"""
        return self.count("failed") + self.count("error")

    @property
    def skipped(self) -> int:
        """跳过用例数。"""
        return self.count("skipped")

    @property
    def duration(self) -> float:
        """全部用例耗时之和（秒，保留 3 位）。"""
        return round(sum(case.duration for case in self.cases), 3)

    @property
    def pass_rate(self) -> float:
        """通过率 = 通过 / （总数 - 跳过）；没有可执行用例时返回 ``0.0``。"""
        effective = self.total - self.skipped
        return round(self.passed / effective, 4) if effective > 0 else 0.0

    @property
    def has_failures(self) -> bool:
        """是否存在失败或错误用例。"""
        return any(case.failing for case in self.cases)

    def failures(self) -> tuple[CaseResult, ...]:
        """返回全部失败 / 错误用例。"""
        return tuple(case for case in self.cases if case.failing)


def _case_outcome(case: ET.Element) -> Outcome:
    """按子节点判定单个用例的结果。"""
    if case.find("failure") is not None:
        return "failed"
    if case.find("error") is not None:
        return "error"
    if case.find("skipped") is not None:
        return "skipped"
    return "passed"


def _case_duration(case: ET.Element) -> float:
    """读取用例耗时；缺失或非数字时按 ``0.0`` 处理。"""
    try:
        return float(case.get("time", "0"))
    except ValueError:
        return 0.0


def parse_junit_xml(text: str, *, name: str = "integration") -> SuiteReport:
    """解析 JUnit XML 文本为 :class:`SuiteReport`。

    Args:
        text: JUnit XML 全文（根节点为 ``<testsuite>`` 或 ``<testsuites>``）。
        name: 报告名，仅用于展示。

    Returns:
        汇总报告；用例明细从 ``<testcase>`` 子节点逐条推导。

    Raises:
        ValidationError: XML 无法解析，或根节点既不是 ``<testsuite>`` 也不是
            ``<testsuites>`` 时。
    """
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise ValidationError(f"invalid junit xml: {exc}") from exc
    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = list(root.findall("testsuite"))
    else:
        raise ValidationError(f"unexpected junit root element: {root.tag!r}")
    cases = tuple(
        CaseResult(
            classname=case.get("classname", ""),
            name=case.get("name", ""),
            outcome=_case_outcome(case),
            duration=_case_duration(case),
        )
        for suite in suites
        for case in suite.findall("testcase")
    )
    return SuiteReport(name=name, cases=cases)


def parse_junit_report(path: Path) -> SuiteReport:
    """从文件解析 JUnit 报告。

    Args:
        path: JUnit XML 文件路径；文件名（去扩展名）作为报告名。

    Returns:
        解析后的汇总报告。

    Raises:
        ValidationError: 文件不存在，或内容不是合法 JUnit XML 时。
    """
    if not path.is_file():
        raise ValidationError(f"junit report not found: {path}")
    return parse_junit_xml(path.read_text(encoding="utf-8"), name=path.stem)


@dataclass(frozen=True)
class TrendReport:
    """当前运行相对基线的对比结果。"""

    #: 本次运行。
    current: SuiteReport

    #: 基线（上一次运行）；缺失时为 ``None``，所有对比项退化为空。
    baseline: SuiteReport | None = None

    @property
    def new_failures(self) -> tuple[CaseResult, ...]:
        """本次失败、但基线未失败的用例（无基线时为空）。"""
        if self.baseline is None:
            return ()
        known = {case.node_id for case in self.baseline.failures()}
        return tuple(case for case in self.current.failures() if case.node_id not in known)

    @property
    def fixed(self) -> tuple[CaseResult, ...]:
        """基线失败、本次不再失败的用例（无基线时为空）。"""
        if self.baseline is None:
            return ()
        failing_now = {case.node_id for case in self.current.failures()}
        return tuple(case for case in self.baseline.failures() if case.node_id not in failing_now)

    @property
    def pass_rate_delta(self) -> float:
        """通过率变化（本次 - 基线，保留 4 位；无基线时为 ``0.0``）。"""
        if self.baseline is None:
            return 0.0
        return round(self.current.pass_rate - self.baseline.pass_rate, 4)

    def slowest(self, limit: int = DEFAULT_SLOWEST_LIMIT) -> tuple[CaseResult, ...]:
        """返回本次耗时最长的前 ``limit`` 个用例（按耗时降序）。

        Args:
            limit: 展示条数；``<= 0`` 时返回空元组。

        Returns:
            用例元组；耗时相同时保持原始顺序（``sorted`` 稳定）。
        """
        ordered = sorted(self.current.cases, key=lambda case: case.duration, reverse=True)
        return tuple(ordered[: max(limit, 0)])


def build_trend(current: SuiteReport, baseline: SuiteReport | None = None) -> TrendReport:
    """组装趋势报告。

    Args:
        current: 本次运行的报告。
        baseline: 上一次运行的报告；``None`` 表示首次运行，无对比基线。

    Returns:
        :class:`TrendReport`。
    """
    return TrendReport(current=current, baseline=baseline)


def _case_section(
    title: str,
    cases: Sequence[CaseResult],
    *,
    empty: str,
    with_duration: bool = False,
) -> list[str]:
    """把一组用例渲染成 Markdown 小节（为空时输出 ``empty`` 占位句）。"""
    lines = [f"### {title}", ""]
    if cases:
        for case in cases:
            suffix = f" — {case.duration:.3f}s" if with_duration else ""
            lines.append(f"- `{case.node_id}`{suffix}")
    else:
        lines.append(empty)
    lines.append("")
    return lines


def render_markdown(trend: TrendReport, *, slowest_limit: int = DEFAULT_SLOWEST_LIMIT) -> str:
    """把趋势报告渲染成 Markdown（可直接追加进 ``$GITHUB_STEP_SUMMARY``）。

    Args:
        trend: 趋势报告。
        slowest_limit: 「耗时最长的用例」小节展示条数。

    Returns:
        以换行结尾的 Markdown 文本。
    """
    current = trend.current
    baseline = trend.baseline
    missing = "—"
    lines = [
        f"# 集成测试趋势 · {current.name}",
        "",
        f"结果：**{'存在失败' if current.has_failures else '全部通过'}**"
        f"（{current.passed}/{current.total - current.skipped} 通过，{current.skipped} 跳过）",
        "",
        "| 指标 | 本次 | 基线 |",
        "|---|---|---|",
        f"| 用例总数 | {current.total} | {missing if baseline is None else baseline.total} |",
        f"| 通过 | {current.passed} | {missing if baseline is None else baseline.passed} |",
        f"| 失败/错误 | {current.failed} | {missing if baseline is None else baseline.failed} |",
        f"| 跳过 | {current.skipped} | {missing if baseline is None else baseline.skipped} |",
        f"| 通过率 | {current.pass_rate:.2%} | "
        f"{missing if baseline is None else format(baseline.pass_rate, '.2%')} |",
        f"| 总耗时 | {current.duration:.3f}s | "
        f"{missing if baseline is None else format(baseline.duration, '.3f') + 's'} |",
        "",
    ]
    if baseline is not None:
        lines += [f"通过率变化：**{trend.pass_rate_delta:+.2%}**", ""]
    lines += _case_section(
        "本次新增失败", trend.new_failures, empty="无（与基线相比没有新增失败用例）"
    )
    lines += _case_section("已修复", trend.fixed, empty="无（基线中没有失败用例）")
    lines += _case_section(
        "耗时最长的用例",
        trend.slowest(slowest_limit),
        empty="无用例明细",
        with_duration=True,
    )
    return "\n".join(lines).rstrip("\n") + "\n"


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器。

    Returns:
        配置好 ``report`` 位置参数与 ``--baseline`` / ``--output`` / ``--slowest``
        选项的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="med-memory-trend",
        description="Summarise a pytest JUnit report and compare it with a baseline run.",
    )
    parser.add_argument("report", type=Path, help="path to the current JUnit XML report")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="path to the previous JUnit XML report (ignored when the file is missing)",
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
    """命令行入口：解析报告 → 渲染 Markdown → 写文件或标准输出。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示摘要已产出；``1`` 表示报告缺失或非法（错误信息写 stderr）。
        是否存在失败用例由报告内容体现，不作为退出码，避免夜间工作流的
        摘要步骤掩盖 pytest 本身的失败状态。
    """
    args = build_parser().parse_args(argv)
    baseline_path: Path | None = args.baseline
    try:
        current = parse_junit_report(args.report)
        baseline = (
            parse_junit_report(baseline_path)
            if baseline_path is not None and baseline_path.is_file()
            else None
        )
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    markdown = render_markdown(build_trend(current, baseline), slowest_limit=args.slowest)
    if args.output is None:
        print(markdown, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
