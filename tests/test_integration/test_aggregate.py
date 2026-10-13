"""``med_langchain_memory.testing.aggregate`` 的离线单测（D44 分片结果聚合）。

全部用例只依赖标准库与合成 JUnit XML，**不需要任何真实中间件、Docker 或网络**，
因此这些用例在默认 ``pytest`` 下就会执行（不带 ``integration`` 标记）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.testing.aggregate import (
    AGGREGATE_SUMMARY_NAME,
    ARTIFACT_PREFIX,
    EXIT_FAILURES,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
    AggregateReport,
    ShardStatus,
    build_aggregate,
    build_parser,
    collect_shard_reports,
    main,
    render_markdown,
    shard_report_path,
)
from med_langchain_memory.testing.shards import shard_names
from med_langchain_memory.testing.trend import JUNIT_REPORT_NAME, parse_junit_xml

#: 聚合模块的模块路径（子进程用例用它与真实 CLI 对齐）。
AGGREGATE_MODULE = "med_langchain_memory.testing.aggregate"

#: 仓库根目录（子进程用例的 cwd 与 PYTHONPATH 基准）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

JUNIT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="{tests}" failures="{failures}" errors="{errors}"
             skipped="{skipped}" time="{time}">
{cases}
  </testsuite>
</testsuites>
"""


def make_case(
    classname: str,
    name: str,
    *,
    time: str = "0.100",
    kind: str | None = None,
) -> str:
    """构造一个 ``<testcase>`` 片段；``kind`` 取 ``failure`` / ``error`` / ``skipped``。"""
    body = "" if kind is None else f'<{kind} message="boom" />'
    return f'<testcase classname="{classname}" name="{name}" time="{time}">{body}</testcase>'


def make_junit(cases: list[str]) -> str:
    """按模板拼一份 JUnit XML（``tests`` 属性故意写成错值，聚合口径只信用例明细）。"""
    return JUNIT_TEMPLATE.format(
        tests=999,
        failures=0,
        errors=0,
        skipped=0,
        time="1.000",
        cases="\n".join(f"    {case}" for case in cases),
    )


def write_report(
    reports_dir: Path,
    shard: str,
    cases: list[str],
    *,
    layout: str = "direct",
    text: str | None = None,
) -> Path:
    """写入一个分片的报告，返回文件路径。

    Args:
        reports_dir: 报告根目录。
        shard: 分片名。
        cases: ``<testcase>`` 片段列表（``text`` 给定时忽略）。
        layout: ``direct`` → ``reports/<分片>/``；``artifact`` →
            ``reports/integration-report-<分片>/``（artifact 下载后的布局）。
        text: 直接写入的原始文本（用于构造非法 XML）。
    """
    directory = reports_dir / (shard if layout == "direct" else f"{ARTIFACT_PREFIX}{shard}")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / JUNIT_REPORT_NAME
    path.write_text(make_junit(cases) if text is None else text, encoding="utf-8")
    return path


def passing_case(shard: str, *, time: str = "0.100") -> str:
    """构造一个通过用例（类名带分片名，便于断言归属）。"""
    return make_case(f"tests.test_integration.test_{shard}", "test_ok", time=time)


@pytest.fixture
def all_green_reports(tmp_path: Path) -> Path:
    """四个分片都齐备、且全部通过的报告目录。"""
    root = tmp_path / "reports"
    for name in shard_names():
        write_report(root, name, [passing_case(name)])
    return root


class TestShardReportPath:
    """报告路径探测：兼容本地布局与 artifact 下载布局。"""

    def test_finds_canonical_layout(self, tmp_path: Path) -> None:
        """正向：本地 ``reports/<分片>/integration.xml`` 能被找到。"""
        expected = write_report(tmp_path, "redis", [passing_case("redis")])
        assert shard_report_path(tmp_path, "redis") == expected

    def test_finds_downloaded_artifact_layout(self, tmp_path: Path) -> None:
        """正向：``actions/download-artifact`` 解出来的目录名（artifact 名）也能被找到。"""
        expected = write_report(tmp_path, "mysql", [passing_case("mysql")], layout="artifact")
        assert expected.parent.name == f"{ARTIFACT_PREFIX}mysql"
        assert shard_report_path(tmp_path, "mysql") == expected

    def test_prefers_canonical_layout_when_both_exist(self, tmp_path: Path) -> None:
        """边界：两种布局同时存在时以规范布局为准（本地跑的结果优先于下载物）。"""
        canonical = write_report(tmp_path, "redis", [passing_case("redis")])
        write_report(tmp_path, "redis", [passing_case("redis")], layout="artifact")
        assert shard_report_path(tmp_path, "redis") == canonical

    def test_falls_back_to_canonical_path_when_missing(self, tmp_path: Path) -> None:
        """边界：两种布局都不存在时返回规范位置，便于错误信息直接指路。"""
        assert shard_report_path(tmp_path, "redis") == tmp_path / "redis" / JUNIT_REPORT_NAME


class TestCollectShardReports:
    """逐分片读取报告：缺失与解析失败都不抛异常，而是记进状态。"""

    def test_collects_every_shard_in_table_order(self, all_green_reports: Path) -> None:
        """正向：默认覆盖全部分片，顺序与分片表一致。"""
        statuses = collect_shard_reports(all_green_reports)
        assert tuple(status.name for status in statuses) == shard_names()
        assert all(status.present for status in statuses)
        assert all(status.reason == "" for status in statuses)

    def test_empty_reports_dir_reports_all_missing(self, tmp_path: Path) -> None:
        """边界：报告目录为空时每个分片都被标记为缺失（而不是抛异常）。"""
        statuses = collect_shard_reports(tmp_path / "nope")
        assert len(statuses) == len(shard_names())
        assert not any(status.present for status in statuses)
        assert all("report not found" in status.reason for status in statuses)

    def test_missing_report_points_at_the_canonical_path(self, tmp_path: Path) -> None:
        """边界：缺失原因里带上规范路径，避免「到底找的哪个目录」说不清。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        statuses = {status.name: status for status in collect_shard_reports(tmp_path)}
        missing = statuses["mysql"]
        assert not missing.present
        assert str(tmp_path / "mysql" / JUNIT_REPORT_NAME) in missing.reason
        assert statuses["redis"].present

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("<not-junit>", "invalid junit xml"),
            ("<foo />", "unexpected junit root element"),
        ],
        ids=["malformed-xml", "wrong-root"],
    )
    def test_invalid_report_is_recorded_as_error(
        self, tmp_path: Path, text: str, expected: str
    ) -> None:
        """异常：报告无法解析时记为不可用，并保留解析失败的原因。"""
        write_report(tmp_path, "redis", [], text=text)
        status = collect_shard_reports(tmp_path, ["redis"])[0]
        assert not status.present
        assert status.error is not None and expected in status.error
        assert status.reason == status.error

    def test_partial_selection_limits_the_statuses(self, all_green_reports: Path) -> None:
        """正向：``--shard`` 语义下只收集被选中的分片。"""
        statuses = collect_shard_reports(all_green_reports, ["mysql", "redis"])
        assert tuple(status.name for status in statuses) == ("redis", "mysql")

    def test_all_selector_returns_every_shard(self, all_green_reports: Path) -> None:
        """边界：显式传 ``all`` 与省略参数等价。"""
        assert collect_shard_reports(all_green_reports, ["all"]) == collect_shard_reports(
            all_green_reports
        )

    def test_unknown_shard_name_is_rejected(self, tmp_path: Path) -> None:
        """异常：未知分片名必须报错，而不是静默返回空结果。"""
        with pytest.raises(ValidationError):
            collect_shard_reports(tmp_path, ["mongo"])

    def test_status_carries_the_shard_title(self, all_green_reports: Path) -> None:
        """正向：状态里带上分片标题（Job Summary 里可直接展示）。"""
        statuses = {status.name: status for status in collect_shard_reports(all_green_reports)}
        assert statuses["redis"].title == "Redis single node"


def make_status(
    name: str,
    cases: list[tuple[str, str]],
) -> ShardStatus:
    """构造一个 ``ShardStatus``；``cases`` 为 ``(用例名, kind)`` 列表，``kind`` 可为空串。"""
    fragments = [
        make_case(f"tests.test_integration.test_{name}", case, kind=kind or None)
        for case, kind in cases
    ]
    return ShardStatus(
        name=name,
        title=name,
        path=Path(name) / JUNIT_REPORT_NAME,
        report=parse_junit_xml(make_junit(fragments), name=name),
    )


class TestAggregateReport:
    """聚合口径：合计、通过率、缺失清单、失败清单与最慢用例。"""

    def test_totals_sum_across_shards(self) -> None:
        """正向：用例数 / 通过 / 失败 / 跳过 / 耗时都按分片求和。"""
        report = build_aggregate(
            [
                make_status("redis", [("a", ""), ("b", "failure")]),
                make_status("mysql", [("c", ""), ("d", "skipped"), ("e", "error")]),
            ]
        )
        assert report.total == 5
        assert report.passed == 2
        assert report.failed == 2
        assert report.skipped == 1
        assert report.duration == pytest.approx(0.5, abs=0.01)

    def test_pass_rate_excludes_skipped(self) -> None:
        """正向：通过率分母排除跳过用例。"""
        report = build_aggregate([make_status("redis", [("a", ""), ("b", "skipped")])])
        assert report.pass_rate == 1.0

    def test_pass_rate_is_zero_without_runnable_cases(self) -> None:
        """边界：全部跳过（没有可执行用例）时通过率为 0，而不是除零报错。"""
        report = build_aggregate([make_status("redis", [("a", "skipped")])])
        assert report.pass_rate == 0.0
        assert report.total == 1
        assert report.skipped == 1

    def test_complete_and_missing_track_unavailable_shards(self) -> None:
        """正向：缺失分片既进 ``missing`` 也让 ``complete`` 为假。"""
        statuses = collect_shard_reports(Path("no-such-reports"))
        report = build_aggregate(statuses)
        assert not report.complete
        assert tuple(status.name for status in report.missing) == shard_names()
        assert report.reports == ()

    def test_has_failures_is_false_for_a_missing_only_report(self) -> None:
        """边界：只有缺失、没有解析出的报告时不算「有失败用例」（缺失另有退出码）。"""
        report = build_aggregate(collect_shard_reports(Path("no-such-reports"), ["redis"]))
        assert not report.has_failures
        assert not report.complete

    def test_failures_are_tagged_with_the_shard(self) -> None:
        """正向：失败清单带上分片名，跨分片同名用例也不会混淆。"""
        report = build_aggregate(
            [
                make_status("redis", [("test_x", "failure")]),
                make_status("mysql", [("test_x", "error")]),
            ]
        )
        assert [(shard, case.name) for shard, case in report.failures()] == [
            ("redis", "test_x"),
            ("mysql", "test_x"),
        ]

    def test_slowest_is_sorted_across_shards_and_limited(self) -> None:
        """正向：最慢用例跨分片排序，并受 ``limit`` 截断。"""
        report = build_aggregate(
            [
                make_status("redis", [("a", "")]),
                make_status("mysql", [("b", "")]),
            ]
        )
        assert len(report.slowest(1)) == 1
        assert len(report.slowest()) == 2
        durations = [case.duration for _, case in report.slowest(2)]
        assert durations == sorted(durations, reverse=True)

    def test_slowest_with_non_positive_limit_returns_empty(self) -> None:
        """边界：``limit <= 0`` 返回空元组（用于 ``--slowest 0`` 关掉该小节）。"""
        report = build_aggregate([make_status("redis", [("a", "")])])
        assert report.slowest(0) == ()
        assert report.slowest(-3) == ()

    def test_empty_aggregate_is_safe(self) -> None:
        """边界：没有任何分片时所有指标退化为 0 且不报错。"""
        report = AggregateReport(statuses=())
        assert (report.total, report.passed, report.failed, report.skipped) == (0, 0, 0, 0)
        assert report.pass_rate == 0.0
        assert report.complete and not report.has_failures
        assert report.failures() == () and report.slowest() == ()


class TestRenderMarkdown:
    """Markdown 渲染：标题结论、分片表、合计行与三个小节。"""

    def test_headline_reports_all_green(self, all_green_reports: Path) -> None:
        """正向：全绿时结论行给出通过数与分片覆盖数。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        assert "结果：**全部通过**" in markdown
        assert f"{len(shard_names())}/{len(shard_names())} 个分片有报告" in markdown

    def test_headline_flags_missing_reports(self, tmp_path: Path) -> None:
        """正向：缺报告时结论行直接说明缺了几片（而不是显示「全部通过」）。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        markdown = render_markdown(build_aggregate(collect_shard_reports(tmp_path)))
        assert "结果：**报告不齐（3/4 个分片缺报告）**" in markdown
        assert "1/4 个分片有报告" in markdown

    def test_headline_flags_failures(self) -> None:
        """正向：有失败用例时结论行标红。"""
        report = build_aggregate([make_status("redis", [("a", "failure")])])
        assert "结果：**存在失败**" in render_markdown(report)

    def test_shard_table_lists_every_shard_with_metrics(self, all_green_reports: Path) -> None:
        """正向：分片明细表逐片一行，含用例数 / 通过 / 通过率 / 耗时。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        for name in shard_names():
            assert f"| `{name}` | 1 | 1 | 0 | 0 | 100.00% | 0.100s |" in markdown

    def test_missing_shard_row_is_marked(self, tmp_path: Path) -> None:
        """边界：缺失分片的表格行显示 ``⚠️ 无报告``，而不是伪造 0 数据。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        markdown = render_markdown(build_aggregate(collect_shard_reports(tmp_path)))
        assert "| `mysql` | ⚠️ 无报告 | — | — | — | — | — |" in markdown

    def test_total_row_sums_present_reports(self) -> None:
        """正向：合计行只统计已解析的报告。"""
        report = build_aggregate(
            [
                make_status("redis", [("a", ""), ("b", "failure")]),
                make_status("mysql", [("c", "skipped")]),
            ]
        )
        assert "| **合计** | 3 | 1 | 1 | 1 | 50.00% |" in render_markdown(report)

    def test_missing_section_lists_the_reason(self, tmp_path: Path) -> None:
        """正向：缺失清单逐片给出原因（含路径），便于直接排查。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        markdown = render_markdown(build_aggregate(collect_shard_reports(tmp_path)))
        assert "### 缺失或无法解析的报告" in markdown
        assert f"- `mysql`：report not found: {tmp_path / 'mysql' / JUNIT_REPORT_NAME}" in markdown

    def test_missing_section_shows_placeholder_when_complete(self, all_green_reports: Path) -> None:
        """边界：报告齐备时缺失小节给出占位句而不是空标题。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        assert "无（每个分片的报告都齐备）" in markdown

    def test_failure_section_lists_shard_and_node_id(self) -> None:
        """正向：失败清单带分片前缀与 ``classname::name`` 节点标识。"""
        report = build_aggregate([make_status("mysql", [("test_x", "failure")])])
        markdown = render_markdown(report)
        assert "- `mysql` · `tests.test_integration.test_mysql::test_x`" in markdown

    def test_failure_section_placeholder(self, all_green_reports: Path) -> None:
        """边界：无失败用例时给出占位句。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        assert "无（已解析的报告里没有失败用例）" in markdown

    def test_slowest_section_renders_durations(self) -> None:
        """正向：最慢用例小节带耗时。"""
        report = build_aggregate([make_status("redis", [("a", "")])])
        markdown = render_markdown(report, slowest_limit=1)
        assert "- `redis` · `tests.test_integration.test_redis::a` — 0.100s" in markdown

    def test_slowest_section_can_be_disabled(self) -> None:
        """边界：``slowest_limit=0`` 时该小节退化为占位句。"""
        report = build_aggregate([make_status("redis", [("a", "")])])
        assert "无用例明细" in render_markdown(report, slowest_limit=0)

    def test_markdown_ends_with_a_single_newline(self, all_green_reports: Path) -> None:
        """边界：渲染结果以单个换行结尾（可直接 ``cat >> $GITHUB_STEP_SUMMARY``）。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        assert markdown.endswith("\n") and not markdown.endswith("\n\n")


class TestCli:
    """命令行入口：默认值、输出文件与四种退出码。"""

    def test_parser_defaults(self) -> None:
        """正向：默认从 ``reports`` 取报告、打印到标准输出、展示 5 条最慢用例。"""
        args = build_parser().parse_args([])
        assert args.reports_dir == Path("reports")
        assert args.output is None
        assert args.slowest == 5
        assert args.shard == []

    def test_ok_exit_code_writes_the_summary(self, all_green_reports: Path, tmp_path: Path) -> None:
        """正向：全绿时退出码 0，且汇总文件已写出。"""
        output = tmp_path / "out" / AGGREGATE_SUMMARY_NAME
        assert main(["--reports-dir", str(all_green_reports), "--output", str(output)]) == EXIT_OK
        assert "结果：**全部通过**" in output.read_text(encoding="utf-8")

    def test_incomplete_exit_code_still_writes_the_summary(self, tmp_path: Path) -> None:
        """边界：缺报告时退出码 1，但汇总文件仍先写好（供 ``if: always()`` 步骤贴出）。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        output = tmp_path / AGGREGATE_SUMMARY_NAME
        code = main(["--reports-dir", str(tmp_path), "--output", str(output)])
        assert code == EXIT_INCOMPLETE
        assert "报告不齐" in output.read_text(encoding="utf-8")

    def test_failures_exit_code(self, tmp_path: Path) -> None:
        """正向：报告齐备但有失败用例时退出码 2。"""
        for name in shard_names():
            cases = [passing_case(name)]
            if name == "mysql":
                cases.append(
                    make_case("tests.test_integration.test_mysql", "test_bad", kind="failure")
                )
            write_report(tmp_path, name, cases)
        assert main(["--reports-dir", str(tmp_path)]) == EXIT_FAILURES

    def test_missing_reports_win_over_failures(self, tmp_path: Path) -> None:
        """边界：既缺报告又有失败时按「缺报告」优先（基础设施问题更根本）。"""
        write_report(
            tmp_path,
            "redis",
            [make_case("tests.test_integration.test_redis", "test_bad", kind="failure")],
        )
        assert main(["--reports-dir", str(tmp_path)]) == EXIT_INCOMPLETE

    def test_unknown_shard_is_a_usage_error(self, capsys: pytest.CaptureFixture[str]) -> None:
        """异常：未知分片名退出码 3，并把合法取值写进 stderr。"""
        assert main(["--shard", "mongo"]) == EXIT_USAGE
        captured = capsys.readouterr()
        assert "unknown integration shard 'mongo'" in captured.err
        assert captured.out == ""

    def test_selected_shard_only_needs_its_own_report(self, tmp_path: Path) -> None:
        """正向：``--shard`` 限定后，其余分片缺报告不影响退出码。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        assert main(["--reports-dir", str(tmp_path), "--shard", "redis"]) == EXIT_OK

    def test_prints_to_stdout_without_output_option(
        self, all_green_reports: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：未给 ``--output`` 时汇总打到标准输出。"""
        assert main(["--reports-dir", str(all_green_reports), "--slowest", "0"]) == EXIT_OK
        captured = capsys.readouterr()
        assert captured.out.startswith("# 集成测试分片聚合")
        assert "无用例明细" in captured.out


class TestModuleEntryPoint:
    """子进程级 CLI 用例：``python -m`` 真的会执行（漏写 ``__main__`` 会静默无事发生）。"""

    @pytest.mark.parametrize(
        ("shards_with_reports", "expected"),
        [
            (("redis", "mysql", "elasticsearch", "redis-cluster"), EXIT_OK),
            (("redis",), EXIT_INCOMPLETE),
        ],
        ids=["all-green", "missing-reports"],
    )
    def test_module_entry_point_returns_expected_code(
        self, tmp_path: Path, shards_with_reports: tuple[str, ...], expected: int
    ) -> None:
        """正向/边界：真跑 ``python -m`` 时退出码与预期一致，并打印汇总。"""
        for name in shards_with_reports:
            write_report(tmp_path, name, [passing_case(name)])
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        result = subprocess.run(
            [sys.executable, "-m", AGGREGATE_MODULE, "--reports-dir", str(tmp_path)],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == expected, result.stderr
        assert "# 集成测试分片聚合" in result.stdout
