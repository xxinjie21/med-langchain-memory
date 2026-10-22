"""夜间集成结论回写 commit status：给仓库一个「最近一轮绿没绿」的一眼入口。

动机（D49）：D41–D48 把夜间集成报告做成了 Job Summary 总账、滚动基线对比、
逐分片趋势与 GitHub Pages 看板——但这些都是「登进 GitHub 才能看」的视图。
候选方向里剩下的「状态徽章」由本模块落地：把整轮结论以 **commit status**
（context: :data:`STATUS_CONTEXT`）回写到触发运行的那个提交上，于是：

* README 顶部徽章（ shields.io 的 workflow 状态徽章）常亮显示夜间集成结论；
* 提交页 / 检查列表里每条提交都带 ``integration/nightly`` 状态，
  点进去就是当次运行的链接（``target_url``）。

设计取舍：

* **结论现算，不传递退出码**：发布步骤直接读汇总作业同一份 ``reports/``，
  用 :func:`~med_langchain_memory.testing.aggregate.verdict_exit_code`
  （聚合 CLI 自己那条退出码口径）判状态——不依赖上一步的退出码传递，
  上一步挂了（缺报告）照样能发布 ``error`` 状态，而不是静默跳过；
* **状态映射保守**：只有「齐备且全绿」是 ``success``；「齐备但有失败用例」是
  ``failure``（代码/环境问题）；「报告缺失」与一切其它退出码都是 ``error``
  （CI 基础设施问题，禁止把基础设施故障显示成绿灯）；
* **只做 PR 评论之外的那一半**：本仓库的集成工作流只挂 ``workflow_dispatch``
  与夜间 cron——重型 docker 套件不该随每个 PR 运行，因此没有 PR 可评论，
  commit status 是同一信息在「提交维度」的正确载体；
* **发布失败不冒充测试结论**：网络或 GitHub API 故障时返回退出码 1，
  工作流步骤带 ``continue-on-error``——「徽章没更新」不该把作业染红；
* 依赖仅标准库（``urllib.request`` + ``json``）；HTTP 调用通过可注入的
  ``opener`` 完成，离线单测不碰网络。

退出码（CLI）：

===========  ==========================================================
退出码        含义
===========  ==========================================================
``0``        status 已发布（HTTP 201）；或按请求只打印了 README 徽章
``1``        发布失败：缺少 token、网络错误或 GitHub API 非 2xx
``3``        ``--repo`` 不是 ``owner/name`` 形式（用法错误）
===========  ==========================================================

无论退出码是多少，只要参数能解析，诊断信息都会打印到标准错误——
发布失败的步骤里要能看见「为什么没发出去」。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from med_langchain_memory.exceptions import ValidationError

from .aggregate import (
    EXIT_FAILURES,
    EXIT_OK,
    AggregateReport,
    build_aggregate,
    collect_shard_reports,
    verdict_exit_code,
)
from .trend import DEFAULT_REPORTS_DIR

#: commit status 的 context 名（提交页与检查列表里显示的这一项）。
STATUS_CONTEXT = "integration/nightly"

#: 集成工作流文件名（徽章链接与「徽章链接指向真实文件」的断言共用）。
INTEGRATION_WORKFLOW_FILE = "integration.yml"

#: GitHub status description 的长度上限（API 限制 140 字符，超出截断）。
MAX_DESCRIPTION_LENGTH = 140

#: 退出码：status 已发布（或只打印了徽章）。
EXIT_POSTED = 0

#: 退出码：发布失败（缺 token / 网络或 API 错误）。
EXIT_POST_FAILED = 1

#: 退出码：``--repo`` 形式非法。
EXIT_BAD_REPO = 3

#: commit status 的 state 取值（GitHub API 枚举）。
StatusState = Literal["success", "failure", "error"]


def _validate_repo(repo: str) -> None:
    """校验 ``owner/name`` 形式。

    Args:
        repo: GitHub 仓库标识。

    Raises:
        ValidationError: 不是「恰好一段 owner、一段 name」的形式。
    """
    owner, slash, name = repo.partition("/")
    if not owner or not slash or not name or "/" in name:
        raise ValidationError(f"repo must be 'owner/name', got: {repo!r}")


def statuses_url(repo: str, sha: str) -> str:
    """构造 commit statuses API 地址。

    Args:
        repo: ``owner/name`` 形式的仓库标识。
        sha: 要挂状态的提交 SHA。

    Returns:
        ``https://api.github.com/repos/<repo>/statuses/<sha>``。

    Raises:
        ValidationError: ``repo`` 形式非法。
    """
    _validate_repo(repo)
    return f"https://api.github.com/repos/{repo}/statuses/{sha}"


def status_state_for_exit_code(exit_code: int) -> StatusState:
    """把聚合口径的整轮退出码映射成 commit status 的 state。

    Args:
        exit_code: :func:`~med_langchain_memory.testing.aggregate.verdict_exit_code`
            的返回值（或任何整数；未登记的取值一律按基础设施故障处理）。

    Returns:
        ``0`` → ``success``；``2`` → ``failure``；其余 → ``error``。
    """
    if exit_code == EXIT_OK:
        return "success"
    if exit_code == EXIT_FAILURES:
        return "failure"
    return "error"


def status_description(report: AggregateReport) -> str:
    """用聚合报告生成一句话结论（GitHub status 的 description）。

    Args:
        report: 本轮聚合报告。

    Returns:
        不超过 :data:`MAX_DESCRIPTION_LENGTH` 字符的一句话：
        缺报告时报缺几片，有失败时报失败数，全绿时报通过数。
    """
    total_shards = len(report.statuses)
    present = len(report.reports)
    if not report.complete:
        return f"{total_shards - present}/{total_shards} shards missing reports"
    executed = report.total - report.skipped
    if report.has_failures:
        return f"{report.failed} failing of {executed} across {total_shards} shards"
    return f"{report.passed}/{executed} passed across {total_shards} shards"


def build_status_payload(
    exit_code: int,
    *,
    description: str,
    target_url: str = "",
) -> dict[str, str]:
    """构造 commit status API 的请求体。

    Args:
        exit_code: 整轮退出码（决定 ``state``）。
        description: 一句话结论；超长时按 :data:`MAX_DESCRIPTION_LENGTH` 截断。
        target_url: 当次运行链接；空串时不下发该字段（API 视为可选）。

    Returns:
        含 ``state`` / ``context`` / ``description``（可选 ``target_url``）的字典。
    """
    payload: dict[str, str] = {
        "state": status_state_for_exit_code(exit_code),
        "context": STATUS_CONTEXT,
        "description": description[:MAX_DESCRIPTION_LENGTH],
    }
    if target_url:
        payload["target_url"] = target_url
    return payload


def workflow_url(repo: str) -> str:
    """构造集成工作流页面地址（徽章点击跳转处）。

    Args:
        repo: ``owner/name`` 形式的仓库标识。

    Returns:
        ``https://github.com/<repo>/actions/workflows/<integration.yml>``。

    Raises:
        ValidationError: ``repo`` 形式非法。
    """
    _validate_repo(repo)
    return f"https://github.com/{repo}/actions/workflows/{INTEGRATION_WORKFLOW_FILE}"


def badge_url(repo: str, *, branch: str = "main") -> str:
    """构造 shields.io 的集成工作流状态徽章地址。

    Args:
        repo: ``owner/name`` 形式的仓库标识。
        branch: 徽章关注的分支（集成工作流只在定时/手动触发时跑默认分支）。

    Returns:
        shields.io workflow status 徽章 URL。

    Raises:
        ValidationError: ``repo`` 形式非法。
    """
    _validate_repo(repo)
    quoted_branch = branch.replace(" ", "%20")
    return (
        f"https://img.shields.io/github/actions/workflow/status/"
        f"{repo}/{INTEGRATION_WORKFLOW_FILE}?branch={quoted_branch}"
        f"&label=integration%20nightly"
    )


def badge_markdown(repo: str, *, branch: str = "main") -> str:
    """构造可直接粘进 README 的徽章 Markdown（图片链到徽章、外层链到工作流页）。

    Args:
        repo: ``owner/name`` 形式的仓库标识。
        branch: 徽章关注的分支。

    Returns:
        形如 ``[![integration nightly](<badge>)](<workflow>)`` 的 Markdown 片段。

    Raises:
        ValidationError: ``repo`` 形式非法。
    """
    return f"[![integration nightly]({badge_url(repo, branch=branch)})]({workflow_url(repo)})"


def post_commit_status(
    *,
    repo: str,
    sha: str,
    payload: Mapping[str, str],
    token: str,
    opener: Callable[[Request], Any] | None = None,
) -> int:
    """把 commit status POST 到 GitHub API。

    Args:
        repo: ``owner/name`` 形式的仓库标识。
        sha: 目标提交 SHA。
        payload: :func:`build_status_payload` 的产物。
        token: GitHub token（需要 ``statuses: write`` 权限）。
        opener: 发送请求的可调用对象；``None`` 时用 :func:`urllib.request.urlopen`。
            测试注入假 opener 即可离线覆盖，无需网络。

    Returns:
        HTTP 状态码（GitHub 成功时是 ``201``）。

    Raises:
        ValidationError: ``repo`` 形式非法。
        URLError: 网络错误或 API 返回非 2xx（``HTTPError`` 是其子类）。
    """
    request = Request(statuses_url(repo, sha), data=json.dumps(payload).encode("utf-8"))
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    request.add_header("Content-Type", "application/json")
    send = opener if opener is not None else urlopen
    with send(request) as response:
        return int(response.status)


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数解析器（供 CI 与本地直接调用）。

    Returns:
        配置好 ``--repo`` / ``--sha`` / ``--reports-dir`` / ``--token`` /
        ``--target-url`` / ``--print-badge`` 的解析器。
    """
    parser = argparse.ArgumentParser(
        prog="med-memory-notifier",
        description="Post the nightly integration verdict as a GitHub commit status.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        metavar="OWNER/NAME",
        help="GitHub repository as owner/name",
    )
    parser.add_argument(
        "--sha",
        required=True,
        metavar="SHA",
        help="commit SHA to attach the status to",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path(DEFAULT_REPORTS_DIR),
        help=f"root directory holding per-shard reports (default: {DEFAULT_REPORTS_DIR})",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN", ""),
        help="GitHub token with statuses:write (default: $GITHUB_TOKEN)",
    )
    parser.add_argument(
        "--target-url",
        default="",
        help="link attached to the status (the workflow run URL); omitted when empty",
    )
    parser.add_argument(
        "--print-badge",
        action="store_true",
        help="print the README badge markdown for this repo instead of posting",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """命令行入口：读分片报告 → 现算结论 → 发布 commit status（或只打印徽章）。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        见模块文档的退出码表。结论口径与聚合 CLI 逐字一致
        （:func:`~med_langchain_memory.testing.aggregate.verdict_exit_code`），
        发布步骤因此不依赖上一步的退出码传递。
    """
    args = build_parser().parse_args(argv)
    try:
        _validate_repo(args.repo)
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_BAD_REPO
    if args.print_badge:
        print(badge_markdown(args.repo))
        return EXIT_POSTED
    if not args.token:
        print("error: no GitHub token (set GITHUB_TOKEN or pass --token)", file=sys.stderr)
        return EXIT_POST_FAILED
    report = build_aggregate(collect_shard_reports(args.reports_dir))
    payload = build_status_payload(
        verdict_exit_code(report),
        description=status_description(report),
        target_url=args.target_url,
    )
    try:
        code = post_commit_status(repo=args.repo, sha=args.sha, payload=payload, token=args.token)
    except URLError as exc:
        print(f"error: failed to post status: {exc}", file=sys.stderr)
        return EXIT_POST_FAILED
    print(f"posted {payload['state']} status to {args.repo}@{args.sha} (HTTP {code})")
    return EXIT_POSTED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
