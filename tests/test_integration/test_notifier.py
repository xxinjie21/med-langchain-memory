"""``med_langchain_memory.testing.notifier`` 的离线单测（D49 commit status 发布）。

全部用例只依赖标准库与合成 JUnit XML：HTTP 调用通过可注入的假 opener 覆盖，
**不需要网络、GitHub token 或任何真实中间件**，因此这些用例在默认 ``pytest``
下就会执行（不带 ``integration`` 标记）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from med_langchain_memory.exceptions import ValidationError
from med_langchain_memory.testing import notifier
from med_langchain_memory.testing.aggregate import build_aggregate, collect_shard_reports
from med_langchain_memory.testing.notifier import (
    EXIT_BAD_REPO,
    EXIT_POST_FAILED,
    EXIT_POSTED,
    INTEGRATION_WORKFLOW_FILE,
    MAX_DESCRIPTION_LENGTH,
    STATUS_CONTEXT,
    badge_markdown,
    badge_url,
    build_status_payload,
    main,
    post_commit_status,
    status_description,
    status_state_for_exit_code,
    statuses_url,
    workflow_url,
)
from med_langchain_memory.testing.shards import shard_names
from med_langchain_memory.testing.trend import JUNIT_REPORT_NAME

#: notifier 模块的模块路径（子进程用例用它与真实 CLI 对齐）。
NOTIFIER_MODULE = "med_langchain_memory.testing.notifier"

#: 仓库根目录（子进程用例的 cwd 与 PYTHONPATH 基准）。
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: 工作流文件路径（徽章链接必须指向真实存在的工作流）。
WORKFLOW_PATH = PROJECT_ROOT / ".github" / "workflows" / INTEGRATION_WORKFLOW_FILE

JUNIT_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="{tests}" failures="{failures}" errors="{errors}"
             skipped="{skipped}" time="{time}">
{cases}
  </testsuite>
</testsuites>
"""


def make_case(classname: str, name: str, *, kind: str | None = None) -> str:
    """构造一个 ``<testcase>`` 片段；``kind`` 取 ``failure`` / ``error`` / ``skipped``。"""
    body = "" if kind is None else f'<{kind} message="boom" />'
    return f'<testcase classname="{classname}" name="{name}" time="0.100">{body}</testcase>'


def make_junit(cases: list[str]) -> str:
    """按模板拼一份 JUnit XML（属性故意写成错值，口径只信用例明细）。"""
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


class _FakeResponse:
    """假 HTTP 响应：只提供 ``status`` 与上下文管理器协议（urlopen 的用法）。"""

    def __init__(self, status: int) -> None:
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


class _RecordingOpener:
    """记录请求并返回固定状态码的假 opener（网络替身）。"""

    def __init__(self, status: int = 201) -> None:
        self.requests: list[Request] = []
        self._status = status

    def __call__(self, request: Request) -> _FakeResponse:
        self.requests.append(request)
        return _FakeResponse(self._status)


@pytest.fixture
def all_green_reports(tmp_path: Path) -> Path:
    """全部分片都齐备、且全部通过的报告目录。"""
    root = tmp_path / "reports"
    for name in shard_names():
        write_report(root, name, [make_case(f"tests.test_integration.test_{name}", "test_ok")])
    return root


@pytest.fixture
def failing_reports(tmp_path: Path) -> Path:
    """报告齐备、但有一个失败用例的报告目录。"""
    root = tmp_path / "reports"
    for name in shard_names():
        write_report(root, name, [make_case(f"tests.test_integration.test_{name}", "test_ok")])
    write_report(root, shard_names()[0], [make_case("tests.test_x", "test_bad", kind="failure")])
    return root


@pytest.fixture
def incomplete_reports(tmp_path: Path) -> Path:
    """缺一个分片报告的报告目录（模拟 CI 基础设施故障）。"""
    root = tmp_path / "reports"
    for name in shard_names()[:-1]:
        write_report(root, name, [make_case(f"tests.test_integration.test_{name}", "test_ok")])
    return root


class TestStatusesUrl:
    """statuses API 地址构造与 repo 形式校验。"""

    def test_positive_format(self) -> None:
        """正向：按 owner/name 与 SHA 拼出 GitHub statuses API 地址。"""
        assert statuses_url("owner/name", "abc123") == (
            "https://api.github.com/repos/owner/name/statuses/abc123"
        )

    @pytest.mark.parametrize("repo", ["", "owner", "owner/", "/name", "owner/name/extra", "a/b/c"])
    def test_boundary_invalid_repo_raises(self, repo: str) -> None:
        """边界：不是「恰好一段 owner、一段 name」时抛 ValidationError。"""
        with pytest.raises(ValidationError):
            statuses_url(repo, "abc123")


class TestStatusStateForExitCode:
    """整轮退出码 → commit status state 的映射。"""

    @pytest.mark.parametrize(
        ("exit_code", "state"),
        [(0, "success"), (2, "failure"), (1, "error"), (3, "error"), (7, "error")],
    )
    def test_mapping(self, exit_code: int, state: str) -> None:
        """正向+边界：只有全绿是 success、有失败是 failure，其余一律 error。"""
        assert status_state_for_exit_code(exit_code) == state


class TestStatusDescription:
    """一句话结论（status description）。"""

    def test_positive_all_green(self, all_green_reports: Path) -> None:
        """正向：全绿时报「通过数/可执行数 across N shards」。"""
        report = build_aggregate(collect_shard_reports(all_green_reports))
        executed = report.total - report.skipped
        assert status_description(report) == (
            f"{report.passed}/{executed} passed across {len(shard_names())} shards"
        )

    def test_boundary_with_failures(self, failing_reports: Path) -> None:
        """边界：有失败用例时报失败数与可执行数。"""
        report = build_aggregate(collect_shard_reports(failing_reports))
        description = status_description(report)
        assert "failing of" in description
        assert description.startswith("1 failing")

    def test_boundary_incomplete(self, incomplete_reports: Path) -> None:
        """边界：缺报告时报缺几片（基础设施故障，不伪装成绿灯）。"""
        report = build_aggregate(collect_shard_reports(incomplete_reports))
        description = status_description(report)
        assert "shards missing reports" in description
        assert description.startswith("1/")


class TestBuildStatusPayload:
    """commit status 请求体构造。"""

    def test_positive_fields(self) -> None:
        """正向：state / context / description 三键齐备，context 用模块常量。"""
        payload = build_status_payload(0, description="12/12 passed across 5 shards")
        assert payload == {
            "state": "success",
            "context": STATUS_CONTEXT,
            "description": "12/12 passed across 5 shards",
        }

    def test_boundary_target_url_optional(self) -> None:
        """边界：target_url 为空时不下发该字段，非空时下发。"""
        assert "target_url" not in build_status_payload(0, description="x")
        payload = build_status_payload(2, description="x", target_url="https://run")
        assert payload["target_url"] == "https://run"
        assert payload["state"] == "failure"

    def test_boundary_long_description_truncated(self) -> None:
        """边界：description 超长时按 GitHub 的 140 字符上限截断。"""
        payload = build_status_payload(0, description="x" * 500)
        assert len(payload["description"]) == MAX_DESCRIPTION_LENGTH


class TestBadge:
    """README 徽章片段。"""

    def test_positive_badge_url(self) -> None:
        """正向：徽章指向 shields.io 的 workflow 状态，带分支与标签参数。"""
        url = badge_url("owner/name")
        assert url.startswith("https://img.shields.io/github/actions/workflow/status/")
        assert "owner/name/integration.yml" in url
        assert "branch=main" in url
        assert "label=integration%20nightly" in url

    def test_positive_badge_markdown_wraps_workflow_link(self) -> None:
        """正向：Markdown 由「徽章图片 + 工作流页链接」两层组成。"""
        markdown = badge_markdown("owner/name")
        assert markdown.startswith("[![integration nightly](")
        assert markdown.endswith(f"]({workflow_url('owner/name')})")

    def test_boundary_branch_with_space_is_encoded(self) -> None:
        """边界：分支名含空格时按 URL 编码（%20），不产出坏链接。"""
        assert "branch=release%20candidate" in badge_url("owner/name", branch="release candidate")

    def test_badge_points_at_the_real_workflow_file(self) -> None:
        """正向（交叉校验）：徽章链接里的工作流文件名必须真实存在。"""
        assert WORKFLOW_PATH.is_file()

    @pytest.mark.parametrize("repo", ["owner", "owner/name/extra"])
    def test_boundary_invalid_repo_raises(self, repo: str) -> None:
        """边界：repo 形式非法时徽章函数同样拒绝（不产出坏徽章）。"""
        with pytest.raises(ValidationError):
            badge_url(repo)
        with pytest.raises(ValidationError):
            badge_markdown(repo)


class TestPostCommitStatus:
    """HTTP 发布（假 opener 替身，不碰网络）。"""

    def test_positive_posts_and_returns_status(self) -> None:
        """正向：POST 到 statuses 地址，带鉴权与 API 版本头，返回 HTTP 状态码。"""
        opener = _RecordingOpener(201)
        payload = build_status_payload(0, description="ok", target_url="https://run")
        code = post_commit_status(
            repo="owner/name",
            sha="abc123",
            payload=payload,
            token="secret-token",
            opener=opener,
        )
        assert code == 201
        (request,) = opener.requests
        assert request.full_url == "https://api.github.com/repos/owner/name/statuses/abc123"
        assert request.get_method() == "POST"
        assert request.get_header("Authorization") == "Bearer secret-token"
        assert request.get_header("Accept") == "application/vnd.github+json"
        assert request.get_header("X-github-api-version") == "2022-11-28"
        assert json.loads(request.data.decode("utf-8")) == payload

    def test_boundary_http_error_propagates(self) -> None:
        """边界：API 返回非 2xx 时异常向上传（由 CLI 决定记为发布失败）。"""

        def failing_opener(request: Request) -> _FakeResponse:
            raise HTTPError(request.full_url, 401, "Unauthorized", {}, None)

        with pytest.raises(URLError):
            post_commit_status(
                repo="owner/name",
                sha="abc123",
                payload={"state": "success"},
                token="bad",
                opener=failing_opener,
            )

    def test_boundary_invalid_repo_never_sends(self) -> None:
        """边界：repo 非法时在校验阶段就失败，不发任何请求。"""
        opener = _RecordingOpener()
        with pytest.raises(ValidationError):
            post_commit_status(
                repo="owner", sha="abc123", payload={"state": "success"}, token="t", opener=opener
            )
        assert opener.requests == []


class TestBuildParser:
    """CLI 参数解析器。"""

    def test_positive_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """正向：token 默认取 ``$GITHUB_TOKEN``，其余默认值与聚合 CLI 的口径一致。"""
        monkeypatch.setenv("GITHUB_TOKEN", "env-token")
        args = notifier.build_parser().parse_args(
            ["--repo", "owner/name", "--sha", "abc", "--reports-dir", "reports"]
        )
        assert args.repo == "owner/name"
        assert args.sha == "abc"
        assert args.reports_dir == Path("reports")
        assert args.token == "env-token"
        assert args.target_url == ""
        assert args.print_badge is False

    def test_boundary_missing_required_option(self) -> None:
        """边界：``--repo`` / ``--sha`` 缺失时 argparse 直接报用法错误。"""
        with pytest.raises(SystemExit):
            notifier.build_parser().parse_args(["--sha", "abc"])


class TestMain:
    """CLI 入口（进程内调用，opener 经 monkeypatch 替换）。"""

    def test_positive_all_green_posts_success(
        self,
        all_green_reports: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """正向：全绿报告 → success 状态 → HTTP 201 → 退出码 0。"""
        opener = _RecordingOpener(201)
        monkeypatch.setattr(notifier, "urlopen", opener)
        code = main(
            [
                "--repo",
                "owner/name",
                "--sha",
                "abc123",
                "--reports-dir",
                str(all_green_reports),
                "--token",
                "t",
                "--target-url",
                "https://run",
            ]
        )
        assert code == EXIT_POSTED
        (request,) = opener.requests
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["state"] == "success"
        assert payload["context"] == STATUS_CONTEXT
        assert payload["target_url"] == "https://run"
        assert "posted success status" in capsys.readouterr().out

    def test_boundary_failures_post_failure_state(
        self, failing_reports: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """边界：有失败用例 → failure 状态（代码/环境问题，不是基础设施故障）。"""
        opener = _RecordingOpener(201)
        monkeypatch.setattr(notifier, "urlopen", opener)
        code = main(
            ["--repo", "o/n", "--sha", "s", "--reports-dir", str(failing_reports), "--token", "t"]
        )
        assert code == EXIT_POSTED
        (request,) = opener.requests
        assert json.loads(request.data.decode("utf-8"))["state"] == "failure"

    def test_boundary_incomplete_posts_error_state(
        self, incomplete_reports: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """边界：缺报告 → error 状态（CI 基础设施问题，禁止静默绿灯）。"""
        opener = _RecordingOpener(201)
        monkeypatch.setattr(notifier, "urlopen", opener)
        code = main(
            [
                "--repo",
                "o/n",
                "--sha",
                "s",
                "--reports-dir",
                str(incomplete_reports),
                "--token",
                "t",
            ]
        )
        assert code == EXIT_POSTED
        (request,) = opener.requests
        assert json.loads(request.data.decode("utf-8"))["state"] == "error"

    def test_boundary_missing_token_fails_without_posting(
        self,
        all_green_reports: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """边界：没有 token 时记发布失败，且不发出任何请求。"""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        opener = _RecordingOpener()
        monkeypatch.setattr(notifier, "urlopen", opener)
        code = main(["--repo", "o/n", "--sha", "s", "--reports-dir", str(all_green_reports)])
        assert code == EXIT_POST_FAILED
        assert opener.requests == []
        assert "no GitHub token" in capsys.readouterr().err

    def test_boundary_api_error_returns_post_failed(
        self,
        all_green_reports: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """边界：网络/API 错误 → 退出码 1 且诊断信息进标准错误。"""

        def failing_opener(request: Request) -> _FakeResponse:
            raise HTTPError(request.full_url, 503, "Service Unavailable", {}, None)

        monkeypatch.setattr(notifier, "urlopen", failing_opener)
        code = main(
            [
                "--repo",
                "o/n",
                "--sha",
                "s",
                "--reports-dir",
                str(all_green_reports),
                "--token",
                "t",
            ]
        )
        assert code == EXIT_POST_FAILED
        assert "failed to post status" in capsys.readouterr().err

    def test_boundary_invalid_repo_returns_usage(self, capsys: pytest.CaptureFixture[str]) -> None:
        """边界：repo 形式非法 → 退出码 3（用法错误），不读报告也不发请求。"""
        code = main(["--repo", "owner", "--sha", "s", "--token", "t"])
        assert code == EXIT_BAD_REPO
        assert "owner/name" in capsys.readouterr().err

    def test_positive_print_badge_skips_posting(self, capsys: pytest.CaptureFixture[str]) -> None:
        """正向：``--print-badge`` 只打印 README 徽章，不发布任何状态。"""
        code = main(["--repo", "owner/name", "--sha", "s", "--print-badge"])
        assert code == EXIT_POSTED
        out = capsys.readouterr().out
        assert out.startswith("[![integration nightly](")
        assert "img.shields.io" in out


def test_subprocess_entry_point_prints_badge() -> None:
    """正向（子进程）：``python -m ...notifier`` 真实入口可用（D42 的 __main__ 教训）。"""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            NOTIFIER_MODULE,
            "--repo",
            "owner/name",
            "--sha",
            "abc",
            "--print-badge",
        ],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == EXIT_POSTED, result.stderr
    assert "img.shields.io" in result.stdout


def test_subprocess_entry_point_rejects_bad_repo() -> None:
    """边界（子进程）：repo 非法时真实入口返回退出码 3。"""
    result = subprocess.run(
        [sys.executable, "-m", NOTIFIER_MODULE, "--repo", "owner", "--sha", "abc", "--token", "t"],
        cwd=PROJECT_ROOT,
        env={**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == EXIT_BAD_REPO
    assert "owner/name" in result.stderr
