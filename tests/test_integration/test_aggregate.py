"""``med_langchain_memory.testing.aggregate`` 的离线单测（D44 聚合 + D45 总账基线 + D47 逐分片趋势）。

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
    BASELINE_LEDGER_DIR,
    EXIT_FAILURES,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
    AggregateReport,
    AggregateTrend,
    ShardDelta,
    ShardStatus,
    ShardTrend,
    build_aggregate,
    build_aggregate_trend,
    build_parser,
    collect_shard_reports,
    main,
    render_markdown,
    shard_report_path,
    snapshot_reports,
    verdict_exit_code,
)
from med_langchain_memory.testing.shards import shard_names
from med_langchain_memory.testing.trend import JUNIT_REPORT_NAME, build_trend, parse_junit_xml

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
    """全部分片都齐备、且全部通过的报告目录。"""
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
        missing = len(shard_names()) - 1
        assert f"结果：**报告不齐（{missing}/{len(shard_names())} 个分片缺报告）**" in markdown
        assert f"1/{len(shard_names())} 个分片有报告" in markdown

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


class TestVerdictExitCode:
    """整轮结论的退出码口径（D49 起与 notifier 的状态发布共用的单一真值来源）。"""

    def test_positive_all_green(self, all_green_reports: Path) -> None:
        """正向：齐备且无失败用例 → ``EXIT_OK``。"""
        report = build_aggregate(collect_shard_reports(all_green_reports))
        assert verdict_exit_code(report) == EXIT_OK

    def test_boundary_incomplete_wins_over_failures(self, tmp_path: Path) -> None:
        """边界：缺报告（即使另有失败用例）→ ``EXIT_INCOMPLETE``（基础设施问题优先）。"""
        write_report(
            tmp_path,
            "redis",
            [make_case("tests.test_integration.test_redis", "test_bad", kind="failure")],
        )
        report = build_aggregate(collect_shard_reports(tmp_path))
        assert verdict_exit_code(report) == EXIT_INCOMPLETE

    def test_boundary_failures_when_complete(self, tmp_path: Path) -> None:
        """边界：报告齐备但有失败用例 → ``EXIT_FAILURES``。"""
        for name in shard_names():
            cases = [passing_case(name)]
            if name == "mysql":
                cases.append(
                    make_case("tests.test_integration.test_mysql", "test_bad", kind="failure")
                )
            write_report(tmp_path, name, cases)
        report = build_aggregate(collect_shard_reports(tmp_path))
        assert verdict_exit_code(report) == EXIT_FAILURES


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
            (shard_names(), EXIT_OK),
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


def aggregate_of(spec: dict[str, list[tuple[str, str]]]) -> AggregateReport:
    """按 ``{分片名: [(用例名, kind)]}`` 构造聚合报告（``kind`` 为空串表示通过）。"""
    return build_aggregate([make_status(name, cases) for name, cases in spec.items()])


def reports_of(root: Path, spec: dict[str, list[str]]) -> Path:
    """把 ``{分片名: [<testcase> 片段]}`` 写进 ``root``，返回 ``root``。"""
    for name, cases in spec.items():
        write_report(root, name, cases)
    return root


class TestSnapshotReports:
    """本轮报告快照：把报告按规范布局存一份，供下一轮当基线。"""

    def test_copies_direct_layout_into_the_canonical_target(self, tmp_path: Path) -> None:
        """正向：``reports/<分片>/integration.xml`` 被复制到 ``baseline/<分片>/integration.xml``。"""
        source = write_report(tmp_path / "reports", "redis", [passing_case("redis")])
        target = tmp_path / BASELINE_LEDGER_DIR
        written = snapshot_reports(tmp_path / "reports", target, ["redis"])
        assert written == (target / "redis" / JUNIT_REPORT_NAME,)
        assert written[0].read_text(encoding="utf-8") == source.read_text(encoding="utf-8")

    def test_normalises_the_artifact_layout_too(self, tmp_path: Path) -> None:
        """正向：artifact 下载布局也能被规范成 ``baseline/<分片>/``（基线只有一种结构）。"""
        write_report(tmp_path / "reports", "mysql", [passing_case("mysql")], layout="artifact")
        target = tmp_path / BASELINE_LEDGER_DIR
        written = snapshot_reports(tmp_path / "reports", target, ["mysql"])
        assert written == (target / "mysql" / JUNIT_REPORT_NAME,)
        assert not (target / f"{ARTIFACT_PREFIX}mysql").exists()

    def test_written_paths_follow_the_shard_table_order(self, all_green_reports: Path) -> None:
        """正向：返回路径按分片表顺序排列（与聚合口径同源）。"""
        target = all_green_reports.parent / BASELINE_LEDGER_DIR
        written = snapshot_reports(all_green_reports, target)
        assert tuple(path.parent.name for path in written) == shard_names()

    def test_missing_shards_are_skipped(self, tmp_path: Path) -> None:
        """边界：本轮缺报告的分片不写快照（基线里保持缺，才能被对比显式标出）。"""
        write_report(tmp_path / "reports", "redis", [passing_case("redis")])
        written = snapshot_reports(tmp_path / "reports", tmp_path / BASELINE_LEDGER_DIR)
        assert tuple(path.parent.name for path in written) == ("redis",)
        assert not (tmp_path / BASELINE_LEDGER_DIR / "mysql").exists()

    def test_unknown_shard_is_rejected(self, tmp_path: Path) -> None:
        """异常：未知分片名必须报错，而不是静默少存一片。"""
        with pytest.raises(ValidationError):
            snapshot_reports(tmp_path, tmp_path / BASELINE_LEDGER_DIR, ["mongo"])

    def test_partial_selection_snapshots_only_the_selected_shard(self, tmp_path: Path) -> None:
        """正向：``--shard`` 语义下只快照被选中的分片。"""
        for name in ("redis", "mysql"):
            write_report(tmp_path / "reports", name, [passing_case(name)])
        written = snapshot_reports(tmp_path / "reports", tmp_path / BASELINE_LEDGER_DIR, ["redis"])
        assert tuple(path.parent.name for path in written) == ("redis",)

    def test_skips_when_the_source_is_the_target(self, tmp_path: Path) -> None:
        """边界：源与目标同一路径时不做自拷贝（``copyfile`` 会因同文件报错）。"""
        write_report(tmp_path, "redis", [passing_case("redis")])
        assert snapshot_reports(tmp_path, tmp_path, ["redis"]) == ()
        assert collect_shard_reports(tmp_path, ["redis"])[0].present

    def test_snapshot_round_trips_into_a_readable_baseline(self, all_green_reports: Path) -> None:
        """正向：快照出来的目录能被 ``--baseline-dir`` 那一侧原样读回（工作流的实际契约）。"""
        target = all_green_reports.parent / BASELINE_LEDGER_DIR
        snapshot_reports(all_green_reports, target)
        statuses = collect_shard_reports(target)
        assert all(status.present for status in statuses)
        assert tuple(status.name for status in statuses) == shard_names()


class TestShardDelta:
    """逐分片差异：状态判定与三项变化量。"""

    def test_unchanged_when_both_sides_match(self) -> None:
        """正向：两侧一致时为「持平」，变化量都是 0。"""
        delta = ShardDelta(
            name="redis",
            title="Redis single node",
            current=parse_junit_xml(make_junit([passing_case("redis")])),
            baseline=parse_junit_xml(make_junit([passing_case("redis")])),
        )
        assert delta.comparable
        assert delta.state == "unchanged"
        assert (delta.failed_delta, delta.total_delta, delta.pass_rate_delta) == (0, 0, 0.0)

    def test_regressed_when_failures_grow(self) -> None:
        """正向：失败用例变多即「回退」。"""
        delta = ShardDelta(
            name="mysql",
            title="MySQL",
            current=parse_junit_xml(
                make_junit(
                    [passing_case("mysql"), make_case("tests.test_integration", "x", kind="error")]
                )
            ),
            baseline=parse_junit_xml(make_junit([passing_case("mysql")])),
        )
        assert delta.state == "regressed"
        assert delta.failed_delta == 1
        assert delta.total_delta == 1

    def test_improved_when_failures_shrink(self) -> None:
        """正向：失败用例变少即「改善」。"""
        delta = ShardDelta(
            name="redis",
            title="Redis",
            current=parse_junit_xml(make_junit([passing_case("redis")])),
            baseline=parse_junit_xml(
                make_junit(
                    [
                        passing_case("redis"),
                        make_case("tests.test_integration", "x", kind="failure"),
                    ]
                )
            ),
        )
        assert delta.state == "improved"
        assert delta.failed_delta == -1
        assert delta.pass_rate_delta > 0

    def test_regressed_when_failures_flat_but_pass_rate_drops(self) -> None:
        """边界：失败数不变、通过率下降（少了一条通过用例）也算回退。"""
        delta = ShardDelta(
            name="redis",
            title="Redis",
            current=parse_junit_xml(
                make_junit(
                    [
                        make_case("tests.test_integration", "a"),
                        make_case("tests.test_integration", "b", kind="failure"),
                    ]
                )
            ),
            baseline=parse_junit_xml(
                make_junit(
                    [
                        make_case("tests.test_integration", "a"),
                        make_case("tests.test_integration", "b"),
                        make_case("tests.test_integration", "c", kind="failure"),
                    ]
                )
            ),
        )
        assert delta.failed_delta == 0
        assert delta.state == "regressed"
        assert delta.pass_rate_delta < 0

    def test_added_when_only_the_current_run_has_a_report(self) -> None:
        """边界：基线没有该分片（上一轮它没跑出报告）时标记为「新增」。"""
        delta = ShardDelta(
            name="mysql",
            title="MySQL",
            current=parse_junit_xml(make_junit([passing_case("mysql")])),
        )
        assert not delta.comparable
        assert delta.state == "added"
        assert delta.failed_delta is None

    def test_removed_when_only_the_baseline_has_a_report(self) -> None:
        """边界：本轮缺该分片报告时标记为「缺报告」，不当作「用例归零」。"""
        delta = ShardDelta(
            name="elasticsearch",
            title="Elasticsearch",
            baseline=parse_junit_xml(make_junit([passing_case("elasticsearch")])),
        )
        assert not delta.comparable
        assert delta.state == "removed"
        assert delta.total_delta is None

    def test_absent_when_neither_side_has_a_report(self) -> None:
        """边界：两侧都没有报告时状态为「—」，三项变化量都是 ``None``。"""
        delta = ShardDelta(name="redis", title="Redis")
        assert delta.state == "absent"
        assert (delta.failed_delta, delta.total_delta, delta.pass_rate_delta) == (None, None, None)


class TestAggregateTrend:
    """整轮总账对比：可用性判定、逐分片差异、跨分片新增失败 / 已修复与合计变化量。"""

    def test_without_a_baseline_every_comparison_degrades_to_empty(self) -> None:
        """边界：首次运行（无基线）时不报错，所有对比项退化为空 / ``None``。"""
        trend = build_aggregate_trend(aggregate_of({"redis": [("a", "")]}))
        assert isinstance(trend, AggregateTrend)
        assert not trend.available
        assert not trend.baseline_complete
        assert trend.shards == ()
        assert (trend.new_failures, trend.fixed) == ((), ())
        assert (trend.total_delta, trend.failed_delta, trend.pass_rate_delta) == (None, None, None)
        assert trend.duration_delta is None

    def test_baseline_without_any_report_is_not_available(self, tmp_path: Path) -> None:
        """边界：``--baseline-dir`` 指向空目录（缓存未命中）时同样退化为无基线。"""
        empty = build_aggregate(collect_shard_reports(tmp_path / "nope"))
        trend = build_aggregate_trend(aggregate_of({"redis": [("a", "")]}), empty)
        assert not trend.available
        assert trend.shards == ()
        assert trend.total_delta is None

    def test_partial_baseline_is_available_but_not_complete(self, tmp_path: Path) -> None:
        """边界：基线只覆盖部分分片时对比仍可用，但必须显式标出「不齐」。

        真实场景是「上一轮有分片在 pytest 之前就挂了」：基线目录里只有 redis 的报告，
        而期望的分片清单仍是完整的五片。
        """
        reports = all_green_reports_of(tmp_path)
        write_report(tmp_path / "old", "redis", [passing_case("redis")])
        baseline = build_aggregate(collect_shard_reports(tmp_path / "old"))
        trend = build_aggregate_trend(build_aggregate(collect_shard_reports(reports)), baseline)
        assert trend.available
        assert not trend.baseline_complete
        assert tuple(delta.name for delta in trend.shards) == shard_names()
        assert {delta.state for delta in trend.shards if delta.baseline is None} == {"added"}

    def test_shards_cover_the_shard_table_order(self, all_green_reports: Path) -> None:
        """正向：逐分片差异与分片表同序，且两侧报告都被带上。"""
        baseline = build_aggregate(collect_shard_reports(all_green_reports))
        trend = build_aggregate_trend(
            build_aggregate(collect_shard_reports(all_green_reports)), baseline
        )
        assert tuple(delta.name for delta in trend.shards) == shard_names()
        assert all(delta.comparable for delta in trend.shards)
        assert trend.baseline_complete

    def test_aggregate_level_deltas_are_signed(self) -> None:
        """正向：整轮合计的变化量按「本轮 - 基线」给出。"""
        current = aggregate_of({"redis": [("a", ""), ("b", "")], "mysql": [("c", "")]})
        baseline = aggregate_of(
            {"redis": [("a", ""), ("b", "failure")], "mysql": [("c", "skipped")]}
        )
        trend = build_aggregate_trend(current, baseline)
        assert trend.total_delta == 0
        assert trend.failed_delta == -1
        assert trend.pass_rate_delta > 0

    def test_duration_delta_is_rounded_to_milliseconds(self) -> None:
        """边界：耗时变化保留 3 位小数（浮点减法后不留长尾）。"""
        current = build_aggregate(
            [make_status("redis", [("a", "")])],
        )
        baseline = build_aggregate(
            [make_status("redis", [("a", "")])],
        )
        # 直接改耗时：make_status 默认 0.100s，这里显式构造一份 0.300s 的报告。
        slower = build_aggregate(
            [
                ShardStatus(
                    name="redis",
                    title="Redis single node",
                    path=Path("redis") / JUNIT_REPORT_NAME,
                    report=parse_junit_xml(
                        make_junit([make_case("tests.test_integration", "a", time="0.300")])
                    ),
                )
            ]
        )
        assert build_aggregate_trend(current, baseline).duration_delta == 0.0
        assert build_aggregate_trend(slower, baseline).duration_delta == 0.2

    def test_new_failures_are_scoped_by_shard(self) -> None:
        """正向：同名用例出现在不同分片时不算「已修复 / 新增」（分片名参与比对）。"""
        current = aggregate_of({"redis": [("test_x", "")], "mysql": [("test_x", "failure")]})
        baseline = aggregate_of({"redis": [("test_x", "failure")], "mysql": [("test_x", "")]})
        trend = build_aggregate_trend(current, baseline)
        assert [(shard, case.name) for shard, case in trend.new_failures] == [("mysql", "test_x")]
        assert [(shard, case.name) for shard, case in trend.fixed] == [("redis", "test_x")]

    def test_new_failures_and_fixed_are_empty_for_identical_runs(self) -> None:
        """边界：两轮完全相同时两个清单都为空。"""
        report = aggregate_of({"redis": [("a", ""), ("b", "failure")]})
        trend = build_aggregate_trend(report, report)
        assert trend.new_failures == ()
        assert trend.fixed == ()


class TestShardTrend:
    """分片趋势外壳：转发 trend 口径的用例级清单（D47 两套基线合一）。"""

    @staticmethod
    def make_shard_trend() -> ShardTrend:
        """构造一个 redis 分片趋势：本轮 b 失败、基线 a 失败（互相置换）。"""
        current = parse_junit_xml(
            make_junit(
                [
                    make_case("c", "a"),
                    make_case("c", "b", kind="failure"),
                ]
            )
        )
        baseline = parse_junit_xml(
            make_junit(
                [
                    make_case("c", "a", kind="failure"),
                    make_case("c", "b"),
                ]
            )
        )
        return ShardTrend(
            name="redis",
            title="Redis single node",
            trend=build_trend(current, baseline),
        )

    def test_forwards_new_failures_and_fixed(self) -> None:
        """正向：新增失败与已修复按 trend 口径原样转发。"""
        shard = self.make_shard_trend()
        assert tuple(case.node_id for case in shard.new_failures) == ("c::b",)
        assert tuple(case.node_id for case in shard.fixed) == ("c::a",)

    def test_forwards_pass_rate_delta(self) -> None:
        """正向：通过率变化直接转发 trend 的结论。"""
        shard = self.make_shard_trend()
        assert shard.pass_rate_delta == 0.0


class TestShardTrends:
    """AggregateTrend.shard_trends：同一份基线同时服务整轮与分片粒度。"""

    def test_covers_every_comparable_shard_in_table_order(self, all_green_reports: Path) -> None:
        """正向：本轮与基线都齐备的分片各出一条趋势，顺序与标题跟随分片表。"""
        current = build_aggregate(collect_shard_reports(all_green_reports))
        baseline = build_aggregate(collect_shard_reports(all_green_reports))
        trends = build_aggregate_trend(current, baseline).shard_trends
        assert tuple(shard.name for shard in trends) == shard_names()
        assert tuple(shard.title for shard in trends) == tuple(
            status.title for status in current.statuses
        )

    def test_matches_the_trend_module_result_case_by_case(self) -> None:
        """正对照：分片趋势的用例级口径与 ``trend.build_trend`` 逐字一致（不重新发明）。"""
        current = aggregate_of({"redis": [("a", ""), ("b", "failure")]})
        baseline = aggregate_of({"redis": [("a", "failure"), ("b", "")]})
        trends = build_aggregate_trend(current, baseline).shard_trends
        assert len(trends) == 1
        assert tuple(case.node_id for case in trends[0].new_failures) == (
            "tests.test_integration.test_redis::b",
        )
        assert tuple(case.node_id for case in trends[0].fixed) == (
            "tests.test_integration.test_redis::a",
        )

    def test_skips_shards_missing_on_either_side(self, tmp_path: Path) -> None:
        """边界：任一侧缺报告的分片不出现在趋势里（缺报告由状态表表达）。"""
        reports = all_green_reports_of(tmp_path / "current")
        write_report(tmp_path / "previous", "redis", [passing_case("redis")])
        trend = build_aggregate_trend(
            build_aggregate(collect_shard_reports(reports)),
            build_aggregate(collect_shard_reports(tmp_path / "previous")),
        )
        assert tuple(shard.name for shard in trend.shard_trends) == ("redis",)

    def test_empty_without_a_baseline(self) -> None:
        """边界：无基线时逐分片趋势为空（首次运行该小节不出现）。"""
        report = aggregate_of({"redis": [("a", "")]})
        assert build_aggregate_trend(report).shard_trends == ()

    def test_empty_when_baseline_has_no_reports(self) -> None:
        """边界：基线目录一片报告都没解析出来时同样为空（缓存未命中）。"""
        report = aggregate_of({"redis": [("a", "")]})
        empty = build_aggregate(collect_shard_reports(Path("no-such-reports")))
        assert build_aggregate_trend(report, empty).shard_trends == ()


class TestComparisonRendering:
    """对比小节的渲染：指标表、逐分片状态表、逐分片趋势、新增失败 / 已修复清单与基线不齐警告。"""

    def test_section_is_absent_without_a_baseline(self, all_green_reports: Path) -> None:
        """边界：无基线时不出现对比小节（首次运行的输出与 D44 保持一致）。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        assert "与上一轮对比" not in markdown

    def test_section_renders_metric_table_with_signed_deltas(self) -> None:
        """正向：指标表给出本次 / 基线 / 变化三列，变化带符号。"""
        current = aggregate_of({"redis": [("a", ""), ("b", "")]})
        baseline = aggregate_of({"redis": [("a", ""), ("b", "failure")]})
        markdown = render_markdown(current, baseline=baseline)
        assert "### 与上一轮对比" in markdown
        assert "| 用例总数 | 2 | 2 | +0 |" in markdown
        assert "| 通过 | 2 | 1 | +1 |" in markdown
        assert "| 失败/错误 | 0 | 1 | -1 |" in markdown
        assert "| 总耗时 | 0.200s | 0.200s | +0.000s |" in markdown
        assert "基线报告齐备（1/1 个分片），通过率变化 **+50.00%**" in markdown

    def test_shard_delta_table_marks_regressions(self) -> None:
        """正向：逐分片状态表标出回退（含失败变化与通过率变化）。"""
        current = aggregate_of({"redis": [("a", "failure")]})
        baseline = aggregate_of({"redis": [("a", "")]})
        markdown = render_markdown(current, baseline=baseline)
        assert "| `redis` | 0/1 | 1/1 | +1 | -100.00% | ⚠️ 回退 |" in markdown

    def test_shard_delta_table_marks_added_shards(self, tmp_path: Path) -> None:
        """边界：基线缺某片时该行变化列留空并标「新增」，不伪造 0 数据。"""
        reports = all_green_reports_of(tmp_path)
        write_report(tmp_path / "old", "redis", [passing_case("redis")])
        current = build_aggregate(collect_shard_reports(reports))
        baseline = build_aggregate(collect_shard_reports(tmp_path / "old"))
        markdown = render_markdown(current, baseline=baseline)
        assert "| `mysql` | 1/1 | — | — | — | 新增 |" in markdown
        assert (
            "⚠️ 基线只有 1/5 个分片有报告"
            "（缺：mysql, elasticsearch, redis-cluster, redis-cluster-failover）" in markdown
        )

    def test_new_failures_and_fixed_are_listed_with_shard_prefix(self) -> None:
        """正向：新增失败与已修复都带分片前缀，可直接定位到后端。"""
        current = aggregate_of({"redis": [("a", "")], "mysql": [("b", "failure")]})
        baseline = aggregate_of({"redis": [("a", "failure")], "mysql": [("b", "")]})
        markdown = render_markdown(current, baseline=baseline)
        assert "### 本轮新增失败" in markdown
        assert "- `mysql` · `tests.test_integration.test_mysql::b`" in markdown
        assert "### 已修复" in markdown
        assert "- `redis` · `tests.test_integration.test_redis::a`" in markdown

    def test_placeholders_when_nothing_changed(self) -> None:
        """边界：无新增失败、无已修复时两个小节都给出占位句。"""
        report = aggregate_of({"redis": [("a", "")]})
        markdown = render_markdown(report, baseline=report)
        assert "无（与基线相比没有新增失败用例）" in markdown
        assert "无（基线里没有失败用例）" in markdown

    def test_shard_trend_section_lists_new_failures_and_fixed(self) -> None:
        """正向：逐分片趋势小节逐片列出新增失败与已修复（带 node id）。"""
        current = aggregate_of({"redis": [("a", ""), ("b", "failure")], "mysql": [("c", "")]})
        baseline = aggregate_of({"redis": [("a", "failure"), ("b", "")], "mysql": [("c", "")]})
        markdown = render_markdown(current, baseline=baseline)
        assert "### 逐分片趋势（与上一轮的自己相比）" in markdown
        assert "- `redis` · 通过率 +0.00% · 新增失败 1 · 已修复 1" in markdown
        assert "  - 新增：`tests.test_integration.test_redis::b`" in markdown
        assert "  - 修复：`tests.test_integration.test_redis::a`" in markdown
        assert "- `mysql` · 通过率 +0.00% · 新增失败 0 · 已修复 0" in markdown

    def test_shard_trend_section_is_absent_without_a_baseline(
        self, all_green_reports: Path
    ) -> None:
        """边界：无基线时该小节整体不出现（首次运行的输出保持 D45 形态）。"""
        markdown = render_markdown(build_aggregate(collect_shard_reports(all_green_reports)))
        assert "逐分片趋势" not in markdown

    def test_shard_trend_section_is_absent_when_no_shard_is_comparable(
        self, tmp_path: Path
    ) -> None:
        """边界：基线有报告但与本轮无一片能对上时小节也不出现，对比小节本身仍在。"""
        write_report(tmp_path / "current", "redis", [passing_case("redis")])
        write_report(tmp_path / "previous", "mysql", [passing_case("mysql")])
        markdown = render_markdown(
            build_aggregate(collect_shard_reports(tmp_path / "current")),
            baseline=build_aggregate(collect_shard_reports(tmp_path / "previous")),
        )
        assert "### 与上一轮对比" in markdown
        assert "逐分片趋势" not in markdown

    def test_shard_trend_section_skips_shards_without_baseline(self, tmp_path: Path) -> None:
        """边界：基线只覆盖部分分片时，小节只列两侧齐备的分片，不伪造其它片。"""
        reports = all_green_reports_of(tmp_path / "current")
        write_report(tmp_path / "previous", "redis", [passing_case("redis")])
        markdown = render_markdown(
            build_aggregate(collect_shard_reports(reports)),
            baseline=build_aggregate(collect_shard_reports(tmp_path / "previous")),
        )
        assert "### 逐分片趋势（与上一轮的自己相比）" in markdown
        section = markdown.split("### 逐分片趋势（与上一轮的自己相比）")[1].split("###")[0]
        assert "- `redis` ·" in section
        assert "- `mysql` ·" not in section

    def test_shard_trend_section_sits_between_delta_table_and_cross_shard_lists(self) -> None:
        """正向：小节位置在逐分片状态表之后、跨分片清单之前（阅读顺序稳定）。"""
        current = aggregate_of({"redis": [("a", ""), ("b", "failure")]})
        baseline = aggregate_of({"redis": [("a", ""), ("b", "failure")]})
        markdown = render_markdown(current, baseline=baseline)
        delta_at = markdown.index("| `redis` | 1/2 |")
        trend_at = markdown.index("### 逐分片趋势")
        cross_at = markdown.index("### 本轮新增失败")
        assert delta_at < trend_at < cross_at

    def test_existing_sections_and_newline_contract_are_preserved(self) -> None:
        """边界：加了对比小节后，原有小节与「单个换行结尾」的契约不变。"""
        report = aggregate_of({"redis": [("a", "")]})
        markdown = render_markdown(report, baseline=report)
        assert markdown.startswith("# 集成测试分片聚合")
        assert "| **合计** | 1 | 1 | 0 | 0 | 100.00% | 0.100s |" in markdown
        assert "### 缺失或无法解析的报告" in markdown
        assert "### 失败用例" in markdown
        assert "### 耗时最长的用例" in markdown
        assert markdown.endswith("\n") and not markdown.endswith("\n\n")


def all_green_reports_of(root: Path) -> Path:
    """在 ``root`` 下造一份「五个分片各一条通过用例」的报告目录。"""
    reports = root / "reports"
    for name in shard_names():
        write_report(reports, name, [passing_case(name)])
    return reports


class TestBaselineCli:
    """命令行入口：基线目录、快照目录与两者的组合行为。"""

    def test_parser_defaults_disable_baseline_and_snapshot(self) -> None:
        """正向：默认不带基线与快照（保持 D44 的默认行为不变）。"""
        args = build_parser().parse_args([])
        assert args.baseline_dir is None
        assert args.snapshot_dir is None

    def test_baseline_dir_enables_the_comparison(
        self, all_green_reports: Path, tmp_path: Path
    ) -> None:
        """正向：给了 ``--baseline-dir`` 且目录里有报告时输出对比小节。"""
        output = tmp_path / AGGREGATE_SUMMARY_NAME
        code = main(
            [
                "--reports-dir",
                str(all_green_reports),
                "--baseline-dir",
                str(all_green_reports),
                "--output",
                str(output),
            ]
        )
        assert code == EXIT_OK
        assert "### 与上一轮对比" in output.read_text(encoding="utf-8")

    def test_absent_baseline_dir_renders_no_comparison(
        self, all_green_reports: Path, tmp_path: Path
    ) -> None:
        """边界：基线目录不存在（缓存未命中）时静默退化为无对比，退出码不受影响。"""
        code = main(
            [
                "--reports-dir",
                str(all_green_reports),
                "--baseline-dir",
                str(tmp_path / "nope"),
                "--slowest",
                "0",
            ]
        )
        assert code == EXIT_OK

    def test_absent_baseline_dir_prints_no_section(
        self, all_green_reports: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：同上，标准输出里也确实没有对比小节。"""
        assert (
            main(
                ["--reports-dir", str(all_green_reports), "--baseline-dir", str(tmp_path / "nope")]
            )
            == EXIT_OK
        )
        assert "与上一轮对比" not in capsys.readouterr().out

    def test_snapshot_dir_writes_the_canonical_baseline(
        self, all_green_reports: Path, tmp_path: Path
    ) -> None:
        """正向：``--snapshot-dir`` 把五个分片的报告写进 ``baseline/<分片>/``。"""
        target = tmp_path / BASELINE_LEDGER_DIR
        code = main(
            [
                "--reports-dir",
                str(all_green_reports),
                "--snapshot-dir",
                str(target),
                "--slowest",
                "0",
            ]
        )
        assert code == EXIT_OK
        for name in shard_names():
            assert (target / name / JUNIT_REPORT_NAME).is_file()

    def test_snapshot_is_written_even_when_reports_are_missing(self, tmp_path: Path) -> None:
        """边界：本轮缺报告（退出码 1）时快照仍要写，否则基线链会被一次基础设施故障断掉。"""
        write_report(tmp_path / "reports", "redis", [passing_case("redis")])
        target = tmp_path / BASELINE_LEDGER_DIR
        code = main(["--reports-dir", str(tmp_path / "reports"), "--snapshot-dir", str(target)])
        assert code == EXIT_INCOMPLETE
        assert (target / "redis" / JUNIT_REPORT_NAME).is_file()

    def test_snapshot_then_baseline_round_trip(
        self, all_green_reports: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：先快照再拿它当基线，两轮相同 → 对比小节全为 0 变化。"""
        target = tmp_path / BASELINE_LEDGER_DIR
        assert (
            main(["--reports-dir", str(all_green_reports), "--snapshot-dir", str(target)])
            == EXIT_OK
        )
        assert (
            main(
                [
                    "--reports-dir",
                    str(all_green_reports),
                    "--baseline-dir",
                    str(target),
                    "--slowest",
                    "0",
                ]
            )
            == EXIT_OK
        )
        out = capsys.readouterr().out
        assert (
            f"基线报告齐备（{len(shard_names())}/{len(shard_names())} 个分片），通过率变化 **+0.00%**"
            in out
        )
        assert "无（与基线相比没有新增失败用例）" in out

    def test_shard_selection_applies_to_the_baseline(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：``--shard`` 同时约束本轮与基线的分片集合（只比这一片）。"""
        reports = all_green_reports_of(tmp_path)
        baseline = tmp_path / BASELINE_LEDGER_DIR
        for name in ("redis", "mysql"):
            write_report(baseline, name, [passing_case(name)])
        code = main(
            [
                "--reports-dir",
                str(reports),
                "--baseline-dir",
                str(baseline),
                "--shard",
                "redis",
                "--slowest",
                "0",
            ]
        )
        assert code == EXIT_OK
        assert "基线报告齐备（1/1 个分片）" in capsys.readouterr().out

    def test_module_entry_point_accepts_baseline_and_snapshot(self, tmp_path: Path) -> None:
        """正向：真跑 ``python -m`` 时两个新选项都存在（argparse 不认会退成 2）。"""
        reports = all_green_reports_of(tmp_path)
        baseline = tmp_path / BASELINE_LEDGER_DIR
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                AGGREGATE_MODULE,
                "--reports-dir",
                str(reports),
                "--baseline-dir",
                str(baseline),
                "--snapshot-dir",
                str(baseline),
            ],
            capture_output=True,
            text=True,
            env=env,
            cwd=PROJECT_ROOT,
            check=False,
        )
        assert result.returncode == EXIT_OK, result.stderr
        assert (baseline / "redis" / JUNIT_REPORT_NAME).is_file()
