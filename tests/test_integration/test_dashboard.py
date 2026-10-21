"""``med_langchain_memory.testing.dashboard`` 的离线单测（D48 夜间看板）。

全部用例只依赖标准库与合成 JUnit XML，**不需要任何真实中间件、Docker 或网络**，
因此这些用例在默认 ``pytest`` 下就会执行（不带 ``integration`` 标记）。
CLI 用例以进程内调用覆盖全部分支，并保留子进程用例（``python -m ...``）验证
真实入口——与工作流里的调用方式一致。
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from med_langchain_memory.testing.aggregate import (
    EXIT_FAILURES,
    EXIT_INCOMPLETE,
    EXIT_OK,
    EXIT_USAGE,
    build_aggregate,
    collect_shard_reports,
)
from med_langchain_memory.testing.dashboard import (
    DASHBOARD_INDEX_NAME,
    DASHBOARD_TITLE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_SITE_DIR,
    build_parser,
    main,
    render_dashboard,
)
from med_langchain_memory.testing.shards import shard_names
from med_langchain_memory.testing.trend import DEFAULT_SLOWEST_LIMIT, JUNIT_REPORT_NAME

#: 看板模块的模块路径（子进程用例用它与真实 CLI 对齐）。
DASHBOARD_MODULE = "med_langchain_memory.testing.dashboard"

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


def make_case(classname: str, name: str, *, time: str = "0.100", kind: str | None = None) -> str:
    """构造一个 ``<testcase>`` 片段；``kind`` 取 ``failure`` / ``error`` / ``skipped``。"""
    body = "" if kind is None else f'<{kind} message="boom" />'
    return f'<testcase classname="{classname}" name="{name}" time="{time}">{body}</testcase>'


def make_junit(cases: list[str]) -> str:
    """按模板拼一份 JUnit XML（``tests`` 属性故意写成错值，口径只信用例明细）。"""
    return JUNIT_TEMPLATE.format(
        tests=999,
        failures=0,
        errors=0,
        skipped=0,
        time="1.000",
        cases="\n".join(f"    {case}" for case in cases),
    )


def write_report(reports_dir: Path, shard: str, cases: list[str]) -> Path:
    """写入一个分片的报告（规范布局 ``reports/<分片>/integration.xml``）。"""
    directory = reports_dir / shard
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / JUNIT_REPORT_NAME
    path.write_text(make_junit(cases), encoding="utf-8")
    return path


def passing_case(shard: str, *, time: str = "0.100") -> str:
    """构造一个通过用例（类名带分片名，便于断言归属）。"""
    return make_case(f"tests.test_integration.test_{shard}", "test_ok", time=time)


def failing_case(shard: str) -> str:
    """构造一个失败用例。"""
    return make_case(f"tests.test_integration.test_{shard}", "test_broken", kind="failure")


def build_reports(root: Path, spec: dict[str, list[str]]) -> Path:
    """按 ``{分片: 用例片段}`` 规格写出一棵报告目录。"""
    reports_dir = root / "reports"
    for shard, cases in spec.items():
        write_report(reports_dir, shard, cases)
    return reports_dir


@pytest.fixture
def green_reports(tmp_path: Path) -> Path:
    """全部分片都齐备、且全部通过的报告目录。"""
    return build_reports(tmp_path, {name: [passing_case(name)] for name in shard_names()})


@pytest.fixture
def failing_reports(tmp_path: Path) -> Path:
    """全部分片齐备，但 redis 分片有一个失败用例。"""
    return build_reports(
        tmp_path,
        {
            name: [passing_case(name)] + ([failing_case(name)] if name == "redis" else [])
            for name in shard_names()
        },
    )


@pytest.fixture
def partial_reports(tmp_path: Path) -> Path:
    """缺一个分片报告的报告目录（模拟矩阵作业在 pytest 之前就挂了）。"""
    return build_reports(
        tmp_path, {name: [passing_case(name)] for name in shard_names() if name != "elasticsearch"}
    )


def render(reports_dir: Path, **kwargs: object) -> str:
    """便捷入口：收集报告目录并渲染看板页。"""
    return render_dashboard(build_aggregate(collect_shard_reports(reports_dir)), **kwargs)  # type: ignore[arg-type]


class TestRenderDashboard:
    """渲染层：结构、结论、转义、对比小节。"""

    def test_page_is_a_self_contained_document(self, green_reports: Path) -> None:
        """正向：完整 HTML 文档，内联样式，零外部资源（无 script / link / src）。"""
        page = render(green_reports)
        assert page.startswith("<!DOCTYPE html>")
        assert '<html lang="zh-CN">' in page
        assert "<style>" in page
        assert page.rstrip().endswith("</html>")
        assert "<script" not in page
        assert "<link" not in page
        assert "src=" not in page

    def test_all_green_renders_ok_verdict_with_every_shard(self, green_reports: Path) -> None:
        """正向：全绿时结论为「全部通过」，每个分片都出现在明细表里。"""
        page = render(green_reports)
        assert '<span class="badge ok">全部通过</span>' in page
        assert DASHBOARD_TITLE in page
        for name in shard_names():
            assert f"<code>{name}</code>" in page
        assert "失败用例" in page
        assert "无（已解析的报告里没有失败用例）" in page

    def test_failures_are_listed_with_shard_prefix(self, failing_reports: Path) -> None:
        """正向：失败用例带分片前缀列出，结论徽章变为 fail。"""
        page = render(failing_reports)
        assert '<span class="badge fail">存在失败</span>' in page
        assert "<code>redis</code> · <code>" in page
        assert "test_broken" in page

    def test_missing_report_is_marked_without_fabricated_numbers(
        self, partial_reports: Path
    ) -> None:
        """边界：缺报告的分片整行标注「无报告」，不伪造 0 数据，结论降级为 warn。"""
        page = render(partial_reports)
        assert '<span class="badge warn">报告不齐</span>' in page
        assert "⚠️ 无报告" in page
        assert "elasticsearch" in page

    def test_run_url_is_rendered_as_link(self, green_reports: Path) -> None:
        """正向：给定运行链接时渲染成锚点。"""
        page = render(green_reports, run_url="https://github.com/o/r/actions/runs/1")
        assert '<a href="https://github.com/o/r/actions/runs/1">查看本次运行</a>' in page

    def test_empty_run_url_renders_no_link(self, green_reports: Path) -> None:
        """边界：运行链接为空串时不渲染锚点（本地渲染没有链接可给）。"""
        page = render(green_reports)
        assert "<a href" not in page

    def test_generated_at_is_rendered_when_present(self, green_reports: Path) -> None:
        """正向：给定生成时间时出现在元信息行。"""
        page = render(green_reports, generated_at="2026-10-21T02:00:00Z")
        assert "生成时间 2026-10-21T02:00:00Z" in page

    def test_generated_at_is_omitted_when_empty(self, green_reports: Path) -> None:
        """边界：生成时间为空串时不出现「生成时间」元信息。"""
        page = render(green_reports)
        assert "生成时间" not in page

    def test_dynamic_text_is_escaped(self, tmp_path: Path) -> None:
        """边界：用例名与运行链接中的 HTML 元字符必须被转义（防注入）。"""
        case = (
            '<testcase classname="tests.test_x" '
            'name="test_&lt;script&gt;&amp;&quot;boom&quot;" time="0.100">'
            '<failure message="boom" /></testcase>'
        )
        reports_dir = build_reports(tmp_path, {"redis": [case]})
        page = render(
            reports_dir,
            run_url='https://example.com/?q="><script>alert(1)</script>',
        )
        assert "<script>" not in page
        assert "&lt;script&gt;" in page
        assert "&quot;&gt;&lt;script&gt;" in page

    def test_comparison_section_rendered_with_baseline(self, tmp_path: Path) -> None:
        """正向：给定上一轮基线时渲染对比小节（合计表 + 逐分片表）。"""
        current = build_reports(
            tmp_path, {name: [passing_case(name), failing_case(name)] for name in shard_names()}
        )
        baseline = build_reports(
            tmp_path / "baseline", {name: [passing_case(name)] for name in shard_names()}
        )
        report = build_aggregate(collect_shard_reports(current))
        base = build_aggregate(collect_shard_reports(baseline))
        page = render_dashboard(report, baseline=base)
        assert "与上一轮对比" in page
        assert "用例总数" in page
        assert '<span class="badge fail">存在失败</span>' in page

    def test_comparison_section_absent_without_baseline(self, green_reports: Path) -> None:
        """边界：无基线（首次运行）时对比小节整体不出现。"""
        page = render(green_reports)
        assert "与上一轮对比" not in page

    def test_comparison_section_absent_when_baseline_has_no_reports(
        self, green_reports: Path, tmp_path: Path
    ) -> None:
        """边界：基线目录存在但一份报告都解析不出来时，同样不渲染对比小节。"""
        empty = tmp_path / "empty-baseline"
        empty.mkdir()
        report = build_aggregate(collect_shard_reports(green_reports))
        base = build_aggregate(collect_shard_reports(empty))
        page = render_dashboard(report, baseline=base)
        assert "与上一轮对比" not in page

    @pytest.mark.parametrize(
        ("state", "expected_label"),
        [
            ("absent", "—"),
            ("added", "新增"),
            ("removed", "⚠️ 缺报告"),
            ("regressed", "⚠️ 回退"),
            ("improved", "✅ 改善"),
            ("unchanged", "持平"),
        ],
    )
    def test_per_shard_state_labels(self, tmp_path: Path, state: str, expected_label: str) -> None:
        """正向：逐分片状态标记覆盖全部六种取值（缺报告不伪造数据）。"""
        if state == "absent":
            current_spec = {n: [passing_case(n)] for n in shard_names() if n != "elasticsearch"}
            baseline_spec = dict(current_spec)
        elif state == "added":
            current_spec = {n: [passing_case(n)] for n in shard_names()}
            baseline_spec = {n: [passing_case(n)] for n in shard_names() if n != "elasticsearch"}
        elif state == "removed":
            current_spec = {n: [passing_case(n)] for n in shard_names() if n != "elasticsearch"}
            baseline_spec = {n: [passing_case(n)] for n in shard_names()}
        elif state == "regressed":
            current_spec = {n: [passing_case(n)] + [failing_case(n)] for n in shard_names()}
            baseline_spec = {n: [passing_case(n)] for n in shard_names()}
        elif state == "improved":
            current_spec = {n: [passing_case(n)] for n in shard_names()}
            baseline_spec = {n: [passing_case(n)] + [failing_case(n)] for n in shard_names()}
        else:
            current_spec = {n: [passing_case(n)] for n in shard_names()}
            baseline_spec = {n: [passing_case(n)] for n in shard_names()}
        current = build_reports(tmp_path / "current", current_spec)
        baseline = build_reports(tmp_path / "baseline", baseline_spec)
        report = build_aggregate(collect_shard_reports(current))
        base = build_aggregate(collect_shard_reports(baseline))
        page = render_dashboard(report, baseline=base)
        assert f"<td>{expected_label}</td>" in page

    def test_one_sided_comparison_renders_dash(self, tmp_path: Path) -> None:
        """边界：任一侧缺报告的分片，变化列渲染成 ``—`` 而不是伪造 0。"""
        current = build_reports(tmp_path, {name: [passing_case(name)] for name in shard_names()})
        baseline = build_reports(
            tmp_path / "baseline",
            {name: [passing_case(name)] for name in shard_names() if name != "redis"},
        )
        report = build_aggregate(collect_shard_reports(current))
        base = build_aggregate(collect_shard_reports(baseline))
        page = render_dashboard(report, baseline=base)
        assert '<td class="num">—</td>' in page

    def test_slowest_limit_zero_lists_nothing(self, green_reports: Path) -> None:
        """边界：``slowest_limit <= 0`` 时最慢用例列表为空（占位句而非空章节）。"""
        page = render(green_reports, slowest_limit=0)
        assert "无用例明细" in page

    def test_slowest_cases_show_duration(self, green_reports: Path) -> None:
        """正向：最慢用例带耗时（秒，3 位小数）。"""
        page = render(green_reports)
        assert "耗时最长的用例" in page
        assert "0.100s" in page

    def test_parser_defaults(self) -> None:
        """正向：默认参数 = 全部分片 + ``site/index.html`` + 无基线。"""
        args = build_parser().parse_args([])
        assert args.shard == []
        assert args.reports_dir == Path("reports")
        assert args.baseline_dir is None
        assert args.output == DEFAULT_OUTPUT_PATH
        assert Path(DEFAULT_SITE_DIR) / DASHBOARD_INDEX_NAME == DEFAULT_OUTPUT_PATH
        assert args.run_url == ""
        assert args.generated_at == ""
        assert args.slowest == DEFAULT_SLOWEST_LIMIT


class TestDashboardCli:
    """命令行入口：进程内调用覆盖全部分支，子进程验证真实 ``python -m`` 入口。"""

    def run_cli(self, cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
        """在 ``cwd`` 下执行看板 CLI（PYTHONPATH 指向 src，与 pytest 的 pythonpath 一致）。"""
        env = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}
        return subprocess.run(
            [sys.executable, "-m", DASHBOARD_MODULE, *args],
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
        )

    def test_writes_default_output_path(self, green_reports: Path, tmp_path: Path) -> None:
        """正向：不传 ``--output`` 时写入 ``site/index.html``（发布作业的 publish_dir）。"""
        result = self.run_cli(tmp_path, "--reports-dir", str(green_reports))
        assert result.returncode == EXIT_OK, result.stderr
        assert (tmp_path / DEFAULT_SITE_DIR / DASHBOARD_INDEX_NAME).is_file()

    def test_failures_exit_two_but_page_still_written(
        self, failing_reports: Path, tmp_path: Path
    ) -> None:
        """边界：有失败用例时退出码为 2，但页面已先落盘（先写后判）。"""
        output = tmp_path / "site" / DASHBOARD_INDEX_NAME
        code = main(["--reports-dir", str(failing_reports), "--output", str(output)])
        assert code == EXIT_FAILURES
        assert output.is_file()
        assert "存在失败" in output.read_text(encoding="utf-8")

    def test_missing_reports_exit_one_but_page_still_written(
        self, partial_reports: Path, tmp_path: Path
    ) -> None:
        """边界：报告缺失时退出码为 1，页面同样落盘（缺报告信息必须可见）。"""
        output = tmp_path / "site" / DASHBOARD_INDEX_NAME
        code = main(["--reports-dir", str(partial_reports), "--output", str(output)])
        assert code == EXIT_INCOMPLETE
        assert output.is_file()
        assert "报告不齐" in output.read_text(encoding="utf-8")

    def test_unknown_shard_is_a_usage_error(
        self, green_reports: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """边界：``--shard`` 传未知分片名时退出码为 3，错误信息进 stderr，且不写页面。"""
        code = main(["--reports-dir", str(green_reports), "--shard", "nope"])
        assert code == EXIT_USAGE
        assert "nope" in capsys.readouterr().err
        assert not (tmp_path / DEFAULT_SITE_DIR).exists()

    def test_unknown_shard_is_rejected_through_the_real_entry(
        self, green_reports: Path, tmp_path: Path
    ) -> None:
        """正对照：子进程走真实 ``python -m`` 入口（漏写 ``__main__`` 会静默无事发生）。"""
        result = self.run_cli(tmp_path, "--reports-dir", str(green_reports), "--shard", "nope")
        assert result.returncode == EXIT_USAGE
        assert "nope" in result.stderr

    def test_run_url_and_generated_at_reach_the_page(
        self, green_reports: Path, tmp_path: Path
    ) -> None:
        """正向：运行链接与生成时间经 CLI 传达到页面。"""
        output = tmp_path / "site" / DASHBOARD_INDEX_NAME
        result = self.run_cli(
            tmp_path,
            "--reports-dir",
            str(green_reports),
            "--output",
            str(output),
            "--run-url",
            "https://github.com/o/r/actions/runs/42",
            "--generated-at",
            "2026-10-21T02:00:00Z",
        )
        assert result.returncode == EXIT_OK, result.stderr
        page = output.read_text(encoding="utf-8")
        assert "https://github.com/o/r/actions/runs/42" in page
        assert "2026-10-21T02:00:00Z" in page

    def test_baseline_dir_enables_the_comparison_section(self, tmp_path: Path) -> None:
        """正向：``--baseline-dir`` 指向上一轮报告时页面渲染对比小节。"""
        current = build_reports(
            tmp_path, {name: [passing_case(name), failing_case(name)] for name in shard_names()}
        )
        baseline = build_reports(
            tmp_path / "baseline", {name: [passing_case(name)] for name in shard_names()}
        )
        output = tmp_path / "site" / DASHBOARD_INDEX_NAME
        result = self.run_cli(
            tmp_path,
            "--reports-dir",
            str(current),
            "--baseline-dir",
            str(baseline),
            "--output",
            str(output),
        )
        assert result.returncode == EXIT_FAILURES
        page = output.read_text(encoding="utf-8")
        assert "与上一轮对比" in page
        assert "⚠️ 回退" in page

    def test_slowest_option_is_forwarded(self, green_reports: Path, tmp_path: Path) -> None:
        """边界：``--slowest 0`` 经 CLI 生效（最慢用例列表为空）。"""
        output = tmp_path / "site" / DASHBOARD_INDEX_NAME
        result = self.run_cli(
            tmp_path,
            "--reports-dir",
            str(green_reports),
            "--output",
            str(output),
            "--slowest",
            "0",
        )
        assert result.returncode == EXIT_OK, result.stderr
        assert "无用例明细" in output.read_text(encoding="utf-8")

    def test_shard_option_limits_the_page(self, green_reports: Path, tmp_path: Path) -> None:
        """正向：``--shard`` 可重复，只渲染指定分片（本地排查单后端时用）。"""
        output = tmp_path / "site" / DASHBOARD_INDEX_NAME
        result = self.run_cli(
            tmp_path,
            "--reports-dir",
            str(green_reports),
            "--output",
            str(output),
            "--shard",
            "redis",
        )
        assert result.returncode == EXIT_OK, result.stderr
        page = output.read_text(encoding="utf-8")
        assert "<code>redis</code>" in page
        assert "<code>mysql</code>" not in page
