"""``med_langchain_memory.testing.trend`` 的离线单测（D41 夜间 CI 趋势上报）。

全部用例只依赖标准库与合成 JUnit XML，**不需要任何真实中间件、Docker 或网络**，
因此这些用例在默认 ``pytest`` 下就会执行（不带 ``integration`` 标记）。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.testing.trend import (
    DEFAULT_SLOWEST_LIMIT,
    CaseResult,
    SuiteReport,
    TrendReport,
    build_parser,
    build_trend,
    main,
    parse_junit_report,
    parse_junit_xml,
    render_markdown,
)

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
    message: str = "boom",
) -> str:
    """构造一个 ``<testcase>`` 片段；``kind`` 取 ``failure`` / ``error`` / ``skipped``。"""
    body = "" if kind is None else f'<{kind} message="{message}" />'
    return f'<testcase classname="{classname}" name="{name}" time="{time}">{body}</testcase>'


def make_junit(
    cases: list[str],
    *,
    tests: int | None = None,
    failures: int = 0,
    errors: int = 0,
    skipped: int = 0,
    time: str = "1.000",
) -> str:
    """按模板拼一份 JUnit XML；``tests`` 缺省时按实际用例数填充。"""
    return JUNIT_TEMPLATE.format(
        tests=len(cases) if tests is None else tests,
        failures=failures,
        errors=errors,
        skipped=skipped,
        time=time,
        cases="\n".join(f"    {case}" for case in cases),
    )


MIXED_REPORT = make_junit(
    [
        make_case("tests.test_api.test_health", "test_ok", time="0.010"),
        make_case("tests.test_api.test_health", "test_bad", time="0.900", kind="failure"),
        make_case("tests.test_api.test_health", "test_boom", time="0.050", kind="error"),
        make_case("tests.test_api.test_health", "test_skip", kind="skipped"),
    ]
)


class TestCaseResult:
    def test_node_id_joins_classname_and_name(self) -> None:
        """正向：有 classname 时节点标识为 ``classname::name``。"""
        case = CaseResult("tests.test_a", "test_x", "passed", 0.1)
        assert case.node_id == "tests.test_a::test_x"

    def test_node_id_falls_back_to_name(self) -> None:
        """边界：classname 缺失时只用用例名，不产生前导 ``::``。"""
        assert CaseResult("", "test_x", "passed", 0.1).node_id == "test_x"

    @pytest.mark.parametrize(
        ("outcome", "expected"),
        [("passed", False), ("skipped", False), ("failed", True), ("error", True)],
    )
    def test_failing_excludes_skipped(self, outcome: str, expected: bool) -> None:
        """正向 / 边界：只有失败与错误算「不通过」，跳过不算。"""
        case = CaseResult("c", "n", outcome, 0.0)  # type: ignore[arg-type]
        assert case.failing is expected


class TestParseJunitXml:
    def test_parses_all_outcomes(self) -> None:
        """正向：四种结果都能从子节点正确归一化。"""
        report = parse_junit_xml(MIXED_REPORT, name="nightly")
        assert report.name == "nightly"
        assert report.total == 4
        assert report.passed == 1
        assert report.failed == 2
        assert report.skipped == 1
        assert report.has_failures is True

    def test_accepts_bare_testsuite_root(self) -> None:
        """边界：根节点直接是 ``<testsuite>`` 时同样可解析。"""
        text = f'<testsuite name="pytest">{make_case("c", "test_a")}</testsuite>'
        report = parse_junit_xml(text)
        assert report.total == 1
        assert report.passed == 1

    def test_empty_testsuites_yields_zero_pass_rate(self) -> None:
        """边界：``<testsuites>`` 下没有套件时为空报告，通过率退化为 0.0。"""
        report = parse_junit_xml("<testsuites></testsuites>")
        assert report.total == 0
        assert report.pass_rate == 0.0
        assert report.has_failures is False

    def test_counts_are_derived_from_cases_not_attributes(self) -> None:
        """边界：``<testsuite>`` 的统计属性与用例明细冲突时，以明细为准。"""
        text = make_junit([make_case("c", "test_only")], tests=99, failures=42)
        report = parse_junit_xml(text)
        assert report.total == 1
        assert report.failed == 0

    def test_missing_duration_defaults_to_zero(self) -> None:
        """边界：``time`` 属性缺失按 0.0 计。"""
        text = '<testsuite><testcase classname="c" name="test_a" /></testsuite>'
        report = parse_junit_xml(text)
        assert report.cases[0].duration == 0.0

    def test_unparsable_duration_defaults_to_zero(self) -> None:
        """边界：``time`` 非数字时不抛错，按 0.0 计。"""
        report = parse_junit_xml(make_junit([make_case("c", "test_a", time="n/a")]))
        assert report.cases[0].duration == 0.0

    def test_unexpected_root_element_is_rejected(self) -> None:
        """异常：根节点不是 testsuite/testsuites 时抛 ``ValidationError``。"""
        with pytest.raises(ValidationError, match="unexpected junit root element"):
            parse_junit_xml("<report><testcase /></report>")

    def test_malformed_xml_is_rejected(self) -> None:
        """异常：XML 语法错误时抛 ``ValidationError``。"""
        with pytest.raises(ValidationError, match="invalid junit xml"):
            parse_junit_xml("<testsuites><oops>")


class TestParseJunitReport:
    def test_reads_file_and_uses_stem_as_name(self, tmp_path: Path) -> None:
        """正向：报告名取文件名（去扩展名）。"""
        path = tmp_path / "nightly.xml"
        path.write_text(MIXED_REPORT, encoding="utf-8")
        report = parse_junit_report(path)
        assert report.name == "nightly"
        assert report.total == 4

    def test_missing_file_is_rejected(self, tmp_path: Path) -> None:
        """异常：文件不存在时抛 ``ValidationError``。"""
        with pytest.raises(ValidationError, match="junit report not found"):
            parse_junit_report(tmp_path / "absent.xml")

    def test_directory_is_rejected(self, tmp_path: Path) -> None:
        """边界：路径指向目录（非普通文件）时同样视为缺失。"""
        with pytest.raises(ValidationError, match="junit report not found"):
            parse_junit_report(tmp_path)


class TestSuiteReportMetrics:
    def test_pass_rate_excludes_skipped(self) -> None:
        """正向：通过率的分母排除跳过用例。"""
        report = parse_junit_xml(MIXED_REPORT)
        assert report.pass_rate == pytest.approx(1 / 3, abs=1e-4)

    def test_pass_rate_is_zero_when_everything_skipped(self) -> None:
        """边界：全部跳过时分母为 0，返回 0.0 而不是除零异常。"""
        report = parse_junit_xml(make_junit([make_case("c", "test_a", kind="skipped")], skipped=1))
        assert report.pass_rate == 0.0

    def test_duration_sums_and_rounds(self) -> None:
        """正向：总耗时按用例求和并保留 3 位。"""
        report = parse_junit_xml(MIXED_REPORT)
        assert report.duration == pytest.approx(1.06, abs=1e-3)

    def test_failures_lists_only_failing_cases(self) -> None:
        """正向：``failures()`` 只返回失败与错误用例。"""
        report = parse_junit_xml(MIXED_REPORT)
        names = [case.name for case in report.failures()]
        assert names == ["test_bad", "test_boom"]

    def test_count_returns_zero_for_absent_outcome(self) -> None:
        """边界：没有该结果的用例时计数为 0。"""
        report = parse_junit_xml(make_junit([make_case("c", "test_a")]))
        assert report.count("error") == 0


class TestTrendReport:
    def test_without_baseline_all_comparisons_are_empty(self) -> None:
        """边界：首次运行（无基线）时对比项退化为空 / 零。"""
        trend = build_trend(parse_junit_xml(MIXED_REPORT))
        assert trend.baseline is None
        assert trend.new_failures == ()
        assert trend.fixed == ()
        assert trend.pass_rate_delta == 0.0

    def test_detects_new_failure_and_fixed_case(self) -> None:
        """正向：能同时识别「新增失败」与「已修复」。"""
        baseline = parse_junit_xml(
            make_junit(
                [
                    make_case("c", "test_ok"),
                    make_case("c", "test_old_fail", kind="failure"),
                ]
            )
        )
        current = parse_junit_xml(
            make_junit(
                [
                    make_case("c", "test_ok"),
                    make_case("c", "test_old_fail"),
                    make_case("c", "test_new_fail", kind="error"),
                ]
            )
        )
        trend = build_trend(current, baseline)
        assert [case.name for case in trend.new_failures] == ["test_new_fail"]
        assert [case.name for case in trend.fixed] == ["test_old_fail"]
        assert trend.pass_rate_delta == pytest.approx(1 / 6, abs=1e-4)

    def test_identical_runs_have_no_drift(self) -> None:
        """边界：两次运行完全相同 → 无新增失败、无修复、通过率变化为 0。"""
        report = parse_junit_xml(MIXED_REPORT)
        trend = build_trend(report, report)
        assert trend.new_failures == ()
        assert trend.fixed == ()
        assert trend.pass_rate_delta == 0.0

    def test_slowest_orders_by_duration(self) -> None:
        """正向：``slowest`` 按耗时降序返回前 N 条。"""
        trend = build_trend(parse_junit_xml(MIXED_REPORT))
        assert [case.name for case in trend.slowest(2)] == ["test_bad", "test_skip"]

    @pytest.mark.parametrize("limit", [0, -3])
    def test_slowest_with_non_positive_limit_is_empty(self, limit: int) -> None:
        """边界：``limit <= 0`` 时返回空元组，不触发负索引切片。"""
        trend = build_trend(parse_junit_xml(MIXED_REPORT))
        assert trend.slowest(limit) == ()

    def test_slowest_limit_larger_than_total(self) -> None:
        """边界：``limit`` 超过用例总数时返回全部用例。"""
        trend = build_trend(parse_junit_xml(MIXED_REPORT))
        assert len(trend.slowest(99)) == 4


class TestRenderMarkdown:
    def test_without_baseline_renders_placeholder_column(self) -> None:
        """正向：无基线时基线列填 ``—``，且不出现通过率变化行。"""
        markdown = render_markdown(build_trend(parse_junit_xml(MIXED_REPORT, name="nightly")))
        assert markdown.startswith("# 集成测试趋势 · nightly")
        assert "| 用例总数 | 4 | — |" in markdown
        assert "通过率变化" not in markdown
        assert "存在失败" in markdown
        assert markdown.endswith("\n")

    def test_with_baseline_renders_delta_and_sections(self) -> None:
        """正向：有基线时输出通过率变化与「已修复」小节内容。"""
        baseline = parse_junit_xml(make_junit([make_case("c", "test_old", kind="failure")]))
        current = parse_junit_xml(make_junit([make_case("c", "test_old")]))
        markdown = render_markdown(build_trend(current, baseline))
        assert "通过率变化：**+100.00%**" in markdown
        assert "`c::test_old`" in markdown
        assert "全部通过" in markdown

    def test_empty_sections_render_placeholders(self) -> None:
        """边界：无失败 / 无修复时给出占位句，而不是空小节。"""
        markdown = render_markdown(build_trend(parse_junit_xml(MIXED_REPORT)))
        assert "无（与基线相比没有新增失败用例）" in markdown
        assert "无（基线中没有失败用例）" in markdown

    def test_slowest_section_shows_duration_suffix(self) -> None:
        """正向：耗时小节带 ``s`` 后缀，条数受 ``slowest_limit`` 限制。"""
        markdown = render_markdown(build_trend(parse_junit_xml(MIXED_REPORT)), slowest_limit=1)
        assert "`tests.test_api.test_health::test_bad` — 0.900s" in markdown

    def test_report_without_cases_renders_empty_placeholder(self) -> None:
        """边界：空报告（无任何用例）时耗时小节给出「无用例明细」。"""
        markdown = render_markdown(build_trend(parse_junit_xml("<testsuites></testsuites>")))
        assert "无用例明细" in markdown
        assert "| 通过率 | 0.00% | — |" in markdown


class TestCli:
    def test_prints_markdown_to_stdout(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：不给 ``--output`` 时摘要写到标准输出。"""
        report = tmp_path / "integration.xml"
        report.write_text(MIXED_REPORT, encoding="utf-8")
        assert main([str(report)]) == 0
        out = capsys.readouterr().out
        assert out.startswith("# 集成测试趋势 · integration")
        assert out.endswith("\n")

    def test_writes_markdown_to_output_file(self, tmp_path: Path) -> None:
        """正向：``--output`` 会创建父目录并落盘。"""
        report = tmp_path / "integration.xml"
        report.write_text(MIXED_REPORT, encoding="utf-8")
        target = tmp_path / "reports" / "integration-trend.md"
        assert main([str(report), "--output", str(target)]) == 0
        assert target.read_text(encoding="utf-8").startswith("# 集成测试趋势 · integration")

    def test_missing_report_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """异常：报告缺失时退出码 1，错误写 stderr。"""
        assert main([str(tmp_path / "absent.xml")]) == 1
        assert "junit report not found" in capsys.readouterr().err

    def test_malformed_report_returns_one(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """异常：报告内容非法时退出码 1，错误写 stderr。"""
        broken = tmp_path / "broken.xml"
        broken.write_text("<testsuites><oops>", encoding="utf-8")
        assert main([str(broken)]) == 1
        assert "invalid junit xml" in capsys.readouterr().err

    def test_missing_baseline_falls_back_to_first_run(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``--baseline`` 指向不存在的文件时按首次运行处理，不报错。"""
        report = tmp_path / "integration.xml"
        report.write_text(MIXED_REPORT, encoding="utf-8")
        assert main([str(report), "--baseline", str(tmp_path / "previous.xml")]) == 0
        assert "通过率变化" not in capsys.readouterr().out

    def test_existing_baseline_is_compared(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """正向：基线存在时摘要里带通过率变化行。"""
        report = tmp_path / "integration.xml"
        report.write_text(MIXED_REPORT, encoding="utf-8")
        baseline = tmp_path / "previous.xml"
        baseline.write_text(
            make_junit([make_case("c", "test_old", kind="failure")]), encoding="utf-8"
        )
        assert main([str(report), "--baseline", str(baseline), "--slowest", "1"]) == 0
        out = capsys.readouterr().out
        assert "通过率变化：" in out
        assert out.count(" — ") == 1  # --slowest 1：耗时小节只列一条

    def test_does_not_fail_on_failing_cases(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：存在失败用例时仍返回 0（失败由 pytest 步骤自身负责暴露）。"""
        report = tmp_path / "integration.xml"
        report.write_text(
            make_junit([make_case("c", "test_bad", kind="failure")]), encoding="utf-8"
        )
        assert main([str(report)]) == 0
        assert "存在失败" in capsys.readouterr().out


class TestParser:
    def test_defaults(self) -> None:
        """正向：仅给位置参数时其余选项取默认值。"""
        args = build_parser().parse_args(["reports/integration.xml"])
        assert args.report == Path("reports/integration.xml")
        assert args.baseline is None
        assert args.output is None
        assert args.slowest == DEFAULT_SLOWEST_LIMIT

    def test_option_strings_are_stable(self) -> None:
        """边界：工作流里引用的三个长选项必须真实存在（防止 YAML 与 CLI 漂移）。"""
        options = {option for action in build_parser()._actions for option in action.option_strings}
        assert {"--baseline", "--output", "--slowest"} <= options


class TestSuiteReportConstruction:
    def test_trend_report_is_frozen(self) -> None:
        """边界：``TrendReport`` 为不可变数据类，避免下游误改对比结果。"""
        report = parse_junit_xml(MIXED_REPORT)
        trend = TrendReport(current=report)
        with pytest.raises(Exception, match="cannot assign to field"):
            trend.current = report  # type: ignore[misc]

    def test_suite_report_counts_each_outcome_once(self) -> None:
        """边界：``count`` 只统计完全匹配的结果。"""
        report = SuiteReport(
            name="synthetic",
            cases=(
                CaseResult("c", "a", "passed", 0.1),
                CaseResult("c", "b", "failed", 0.2),
            ),
        )
        assert report.count("passed") == 1
        assert report.count("skipped") == 0
