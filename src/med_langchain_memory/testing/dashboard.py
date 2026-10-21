"""夜间集成报告的外部看板：把总账渲染成自包含 HTML 页面，发布到 GitHub Pages。

动机（D48）：D41–D47 把夜间集成报告越做越细——分片矩阵并行、总账聚合、与上一轮
对比、逐分片用例级趋势——但这些结论只活在 **GitHub Actions 的 Job Summary** 里：
要么登进 GitHub 点开当次运行，要么在 artifact 里翻 JUnit XML。「今晚绿没绿、
哪一片退化了、新增了哪些失败用例」没有一个日常可见的入口。本模块把同一份总账
渲染成一份**自包含**（内联 CSS、零外部资源、零 JS）的 HTML 看板页，由工作流的
``dashboard`` 作业发布到仓库 ``gh-pages`` 分支（GitHub Pages），于是：

* 任何人打开一个固定 URL 就能看到最近一轮的完整结论；
* 页面数据与 Job Summary **同源同口径**——直接消费
  :mod:`med_langchain_memory.testing.aggregate` 的 :class:`AggregateReport` /
  :class:`AggregateTrend`，不重新解析一遍 XML，也不重新定义一套通过率 /
  状态判定；
* 渲染是纯函数、纯标准库，离线可单测；发布失败不影响任何测试结论
  （看板作业 ``if: always()``，且不持有任何门禁状态）。

设计取舍：

* **看板不是门禁**：它只消费汇总作业已经得出的结论。退出码与聚合 CLI 的
  0/1/2 口径一致，仅供本地排查，不在 CI 里卡关；
* **先写页面、后返回退出码**：报告缺失或有用例失败时页面仍必须落盘——
  「看不到看板」不该连带丢掉「为什么失败」的信息；
* **所有动态文本一律 ``html.escape``**：用例名、分片名、运行链接都是不可信
  输入（节点 ID 理论上可包含任意字符），渲染层统一转义，杜绝 HTML 注入；
* 浅色主题、内联样式、无外部请求：GitHub Pages 默认域名下直接打开，不依赖 CDN。

退出码（与 :mod:`med_langchain_memory.testing.aggregate` 一致）：

===========  ==========================================================
退出码        含义
===========  ==========================================================
``0``        所有分片报告齐备且没有失败 / 错误用例
``1``        有分片报告缺失或无法解析
``2``        报告齐备但存在失败 / 错误用例
``3``        ``--shard`` 传了未知分片名（用法错误）
===========  ==========================================================

无论退出码是多少，只要命令行能正常解析，看板页面都会被写出来。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import argparse
import html
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from med_langchain_memory.exceptions import ValidationError

from .aggregate import (
    EXIT_FAILURES,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
    AggregateReport,
    AggregateTrend,
    ShardDelta,
    build_aggregate,
    build_aggregate_trend,
    collect_shard_reports,
)
from .shards import select_shards
from .trend import DEFAULT_REPORTS_DIR, DEFAULT_SLOWEST_LIMIT, CaseResult, SuiteReport

#: 看板页面标题（``<title>`` 与页首大标题）。
DASHBOARD_TITLE = "med-langchain-memory 集成测试看板"

#: 看板入口文件名（GitHub Pages 默认文档）。
DASHBOARD_INDEX_NAME = "index.html"

#: 看板站点目录（``dashboard`` 作业的 ``publish_dir``）。
DEFAULT_SITE_DIR = "site"

#: 默认输出路径：``site/index.html``。
DEFAULT_OUTPUT_PATH = Path(DEFAULT_SITE_DIR) / DASHBOARD_INDEX_NAME

#: 内联样式（浅色主题；页面不引用任何外部资源）。
_STYLE = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; padding: 24px; background: #f6f8fa; color: #1f2328;
       font-family: -apple-system, "Segoe UI", "Noto Sans SC", Arial, sans-serif; }
main { max-width: 960px; margin: 0 auto; }
header { margin-bottom: 8px; }
h1 { font-size: 20px; margin: 0 0 8px; }
h2 { font-size: 15px; margin: 24px 0 8px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 999px;
         font-size: 13px; font-weight: 600; }
.badge.ok { background: #dafbe1; color: #1a7f37; }
.badge.warn { background: #fff8c5; color: #9a6700; }
.badge.fail { background: #ffebe9; color: #cf222e; }
.detail { color: #1f2328; font-size: 13px; margin-left: 8px; }
.meta { color: #59636e; font-size: 12px; margin: 4px 0 0; }
.meta a { color: #0969da; }
table { border-collapse: collapse; width: 100%; background: #ffffff;
        font-size: 13px; }
th, td { border: 1px solid #d1d9e0; padding: 6px 8px; text-align: left; }
th { background: #f6f8fa; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
ul { background: #ffffff; border: 1px solid #d1d9e0; border-radius: 6px;
     padding: 12px 12px 12px 28px; margin: 0; font-size: 13px; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: 12px; }
.empty { color: #59636e; font-size: 13px; margin: 0; }
footer { margin-top: 24px; color: #59636e; font-size: 12px; }
"""

#: 结论等级：``ok`` 全绿 / ``warn`` 报告不齐 / ``fail`` 存在失败。
VerdictLevel = Literal["ok", "warn", "fail"]

#: 分片状态在看板里的标记（与聚合模块的状态判定同源，标签独立维护）。
_STATE_LABELS: dict[str, str] = {
    "absent": "—",
    "added": "新增",
    "removed": "⚠️ 缺报告",
    "regressed": "⚠️ 回退",
    "improved": "✅ 改善",
    "unchanged": "持平",
}


@dataclass(frozen=True)
class _Verdict:
    """整轮看板结论（等级 + 人读标签 + 一行细节）。"""

    #: 结论等级（对应徽章样式）。
    level: VerdictLevel

    #: 人读标签（徽章文本）。
    label: str

    #: 一行细节（通过数 / 缺报告数等）。
    detail: str


def _verdict(report: AggregateReport) -> _Verdict:
    """按「报告齐备性 → 失败数」的优先级给出整轮结论。"""
    total_shards = len(report.statuses)
    present = len(report.reports)
    executed = report.total - report.skipped
    if not report.complete:
        return _Verdict("warn", "报告不齐", f"{present}/{total_shards} 个分片有报告")
    if report.has_failures:
        return _Verdict(
            "fail", "存在失败", f"{report.passed}/{executed} 通过，{report.failed} 个失败"
        )
    return _Verdict("ok", "全部通过", f"{report.passed}/{executed} 通过")


def _esc(value: object) -> str:
    """HTML 转义（含引号，可安全用于文本节点与属性值）。"""
    return html.escape(str(value), quote=True)


def _delta(value: int | float | None, *, percent: bool = False, unit: str = "") -> str:
    """渲染带符号的变化值；``None``（任一侧缺报告、不可比较）渲染成 ``—``。

    Args:
        value: 变化量；``None`` 表示两侧不可比。
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
    passed = report.passed
    total = report.total
    skipped = report.skipped
    return f"{passed}/{total - skipped}"


def _page(*, title: str, body: Sequence[str]) -> str:
    """包上 DOCTYPE / head / 内联样式，产出完整 HTML 文档。"""
    return "\n".join(
        [
            "<!DOCTYPE html>",
            '<html lang="zh-CN">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_esc(title)}</title>",
            f"<style>{_STYLE}</style>",
            "</head>",
            "<body>",
            "<main>",
            *body,
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )


def _header(
    report: AggregateReport,
    verdict: _Verdict,
    *,
    run_url: str,
    generated_at: str,
) -> list[str]:
    """渲染页首：标题、结论徽章、一行元信息（生成时间 / 分片覆盖 / 运行链接）。"""
    meta: list[str] = []
    if generated_at:
        meta.append(f"生成时间 {_esc(generated_at)}")
    meta.append(f"{len(report.reports)}/{len(report.statuses)} 个分片有报告")
    if run_url:
        meta.append(f'<a href="{_esc(run_url)}">查看本次运行</a>')
    return [
        "<header>",
        f"<h1>{_esc(DASHBOARD_TITLE)}</h1>",
        f'<p><span class="badge {verdict.level}">{_esc(verdict.label)}</span>'
        f'<span class="detail">{_esc(verdict.detail)}</span></p>',
        f'<p class="meta">{" · ".join(meta)}</p>',
        "</header>",
    ]


def _shard_table(report: AggregateReport, trend: AggregateTrend) -> list[str]:
    """渲染分片明细表（缺报告的分片整行标注，不伪造 0 数据）。"""
    states = {delta.name: delta.state for delta in trend.shards}
    lines = [
        "<section>",
        "<h2>分片明细</h2>",
        "<table>",
        '<thead><tr><th>分片</th><th class="num">用例</th><th class="num">通过</th>'
        '<th class="num">失败/错误</th><th class="num">跳过</th>'
        '<th class="num">通过率</th><th class="num">耗时</th><th>较上轮</th></tr></thead>',
        "<tbody>",
    ]
    for status in report.statuses:
        state = _STATE_LABELS[states.get(status.name, "absent")]
        shard = status.report
        if shard is None:
            lines.append(
                f"<tr><td><code>{_esc(status.name)}</code></td>"
                '<td colspan="6">⚠️ 无报告</td>'
                f"<td>{state}</td></tr>"
            )
            continue
        lines.append(
            f"<tr><td><code>{_esc(status.name)}</code></td>"
            f'<td class="num">{shard.total}</td><td class="num">{shard.passed}</td>'
            f'<td class="num">{shard.failed}</td><td class="num">{shard.skipped}</td>'
            f'<td class="num">{shard.pass_rate:.2%}</td>'
            f'<td class="num">{shard.duration:.3f}s</td>'
            f"<td>{state}</td></tr>"
        )
    lines += ["</tbody>", "</table>", "</section>"]
    return lines


def _comparison_section(trend: AggregateTrend) -> list[str]:
    """渲染「与上一轮对比」；无可用基线时返回空列表（小节整体不出现）。"""
    baseline = trend.baseline
    if baseline is None or not baseline.reports:
        return []
    current = trend.current
    rows = [
        ("用例总数", str(current.total), str(baseline.total), _delta(trend.total_delta)),
        (
            "通过",
            str(current.passed),
            str(baseline.passed),
            _delta(current.passed - baseline.passed),
        ),
        (
            "失败/错误",
            str(current.failed),
            str(baseline.failed),
            _delta(trend.failed_delta),
        ),
        (
            "跳过",
            str(current.skipped),
            str(baseline.skipped),
            _delta(current.skipped - baseline.skipped),
        ),
        (
            "通过率",
            f"{current.pass_rate:.2%}",
            f"{baseline.pass_rate:.2%}",
            _delta(trend.pass_rate_delta, percent=True),
        ),
        (
            "总耗时",
            f"{current.duration:.3f}s",
            f"{baseline.duration:.3f}s",
            _delta(trend.duration_delta, unit="s"),
        ),
    ]
    lines = [
        "<section>",
        "<h2>与上一轮对比</h2>",
        "<table>",
        '<thead><tr><th>指标</th><th class="num">本次</th>'
        '<th class="num">上轮</th><th class="num">变化</th></tr></thead>',
        "<tbody>",
    ]
    for name, current_cell, baseline_cell, delta_cell in rows:
        lines.append(
            f"<tr><td>{name}</td>"
            f'<td class="num">{current_cell}</td><td class="num">{baseline_cell}</td>'
            f'<td class="num">{delta_cell}</td></tr>'
        )
    lines += [
        "</tbody>",
        "</table>",
        "<table>",
        '<thead><tr><th>分片</th><th class="num">本次 通过/可执行</th>'
        '<th class="num">上轮 通过/可执行</th><th class="num">失败变化</th>'
        '<th class="num">通过率变化</th><th>状态</th></tr></thead>',
        "<tbody>",
    ]
    lines.extend(_delta_row(delta) for delta in trend.shards)
    lines += ["</tbody>", "</table>", "</section>"]
    return lines


def _delta_row(delta: ShardDelta) -> str:
    """渲染逐分片对比表的一行（任一侧缺报告时变化列渲染成 ``—``，不伪造 0）。"""
    current_cell = "—" if delta.current is None else _progress(delta.current)
    baseline_cell = "—" if delta.baseline is None else _progress(delta.baseline)
    return (
        f"<tr><td><code>{_esc(delta.name)}</code></td>"
        f'<td class="num">{current_cell}</td><td class="num">{baseline_cell}</td>'
        f'<td class="num">{_delta(delta.failed_delta)}</td>'
        f'<td class="num">{_delta(delta.pass_rate_delta, percent=True)}</td>'
        f"<td>{_STATE_LABELS[delta.state]}</td></tr>"
    )


def _case_list(
    pairs: Sequence[tuple[str, CaseResult]],
    *,
    with_duration: bool,
    empty: str,
) -> list[str]:
    """渲染 ``(分片名, 用例)`` 列表；为空时输出占位句而不是空章节。"""
    if not pairs:
        return [f'<p class="empty">{empty}</p>']
    items = []
    for shard, case in pairs:
        suffix = f" — {case.duration:.3f}s" if with_duration else ""
        items.append(
            f"<li><code>{_esc(shard)}</code> · <code>{_esc(case.node_id)}</code>{suffix}</li>"
        )
    return ["<ul>", *items, "</ul>"]


def _list_section(
    title: str,
    pairs: Sequence[tuple[str, CaseResult]],
    *,
    with_duration: bool,
    empty: str,
) -> list[str]:
    """渲染一个「标题 + 用例列表」小节。"""
    return [f"<h2>{title}</h2>", *_case_list(pairs, with_duration=with_duration, empty=empty)]


def _footer() -> list[str]:
    """渲染页脚（数据来源与发布位置，便于读者追溯口径）。"""
    return [
        "<footer>",
        "由 <code>med_langchain_memory.testing.dashboard</code> 渲染 · "
        "数据口径与 <code>med_langchain_memory.testing.aggregate</code> 一致 · "
        "由 <code>dashboard</code> 作业发布到 GitHub Pages（gh-pages 分支）",
        "</footer>",
    ]


def render_dashboard(
    report: AggregateReport,
    *,
    baseline: AggregateReport | None = None,
    run_url: str = "",
    generated_at: str = "",
    slowest_limit: int = DEFAULT_SLOWEST_LIMIT,
) -> str:
    """把聚合报告渲染成自包含 HTML 看板页。

    Args:
        report: 本轮聚合报告（:func:`~med_langchain_memory.testing.aggregate.build_aggregate`
            的产物）。
        baseline: 上一轮聚合报告；``None`` 或其中一份报告都没解析出来时不渲染
            「与上一轮对比」小节（首次运行与缓存未命中都是这种情形）。
        run_url: 本次运行的链接；空串时不渲染链接。
        generated_at: 生成时间（ISO 8601 字符串）；空串时不渲染该元信息。
        slowest_limit: 「耗时最长的用例」小节展示条数；``<= 0`` 时列表为空。

    Returns:
        以换行结尾的完整 HTML 文档文本。
    """
    trend = build_aggregate_trend(report, baseline)
    body: list[str] = []
    body += _header(report, _verdict(report), run_url=run_url, generated_at=generated_at)
    body += _shard_table(report, trend)
    body += _comparison_section(trend)
    body += _list_section(
        "失败用例",
        report.failures(),
        with_duration=False,
        empty="无（已解析的报告里没有失败用例）",
    )
    body += _list_section(
        "耗时最长的用例",
        report.slowest(slowest_limit),
        with_duration=True,
        empty="无用例明细",
    )
    body += _footer()
    return _page(title=DASHBOARD_TITLE, body=body)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（供 CI 与本地直接调用）。

    Returns:
        配置好 ``--reports-dir`` / ``--shard`` / ``--baseline-dir`` / ``--output``
        / ``--run-url`` / ``--generated-at`` / ``--slowest`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="med-memory-dashboard",
        description="Render the aggregated integration ledger as a self-contained HTML page.",
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
        "--baseline-dir",
        type=Path,
        default=None,
        help="directory holding the previous run's shard reports; enables the comparison section",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"HTML output path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--run-url",
        default="",
        help="link to the workflow run; omitted from the page when empty",
    )
    parser.add_argument(
        "--generated-at",
        default="",
        help="generation timestamp shown in the page header; omitted when empty",
    )
    parser.add_argument(
        "--slowest",
        type=int,
        default=DEFAULT_SLOWEST_LIMIT,
        help=f"how many slow cases to list (default: {DEFAULT_SLOWEST_LIMIT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：读分片报告 → 渲染看板页 → 写文件 → 返回退出码。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        见模块文档的退出码表。看板页面**先写后判**，因此即使退出码非 0，
        ``--output`` 指定的文件也已就绪——「发布失败」不该丢掉失败详情。
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
    page = render_dashboard(
        report,
        baseline=baseline,
        run_url=args.run_url,
        generated_at=args.generated_at,
        slowest_limit=args.slowest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    if not report.complete:
        return EXIT_INCOMPLETE
    return EXIT_FAILURES if report.has_failures else EXIT_OK


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
