"""Conventional Commits 提交信息守门（纯标准库）。

把提交信息按 `Conventional Commits`_ 规范做**字段级规则校验**（主题行的
type/scope/描述、正文分隔），由 pre-commit 的 ``commit-msg`` 钩子与
``.github/workflows/ci.yml`` 的 PR 作业调用，保证 ``changelog.py``（D50）的
发布说明输入始终结构良好。

设计约束：

* **只用标准库**（argparse / re / subprocess / dataclasses / pathlib），零第三方依赖，
  不参与运行时链路（与 ``testing/`` 其余模块同级的 CI 支撑工具）；
* 不调用任何中文文本预处理或语义解析——只对「提交主题行」这一结构化字段做正则匹配；
* 输入是单条消息文本（``--message`` / ``--message-file``）或 git log 文本
  （位置参数 REV_RANGE），单测全程离线（git 调用可被 monkeypatch 覆盖）。

命令行用法::

    python -m med_langchain_memory.testing.commitlint --message "feat: add x"
    python -m med_langchain_memory.testing.commitlint --message-file .git/COMMIT_EDITMSG
    python -m med_langchain_memory.testing.commitlint origin/main..HEAD

退出码：``0`` 全部合规；``1`` 存在违规；``2`` 用法或 git 调用错误。

.. _Conventional Commits: https://www.conventionalcommits.org/
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

#: ``git log --format`` 记录格式：完整哈希、主题、正文，记录以 ``\\x1e`` 结尾。
GIT_LOG_FORMAT = "%H%x1f%s%x1f%b%x1e"

#: Conventional Commits 主题行：``type(scope)!: description``（冒号后须恰好一个空格）。
SUBJECT_PATTERN = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[a-z0-9][a-z0-9._-]*)\))?(?P<breaking>!)?: (?P<description>\S.*)$"
)

#: 完整（或缩写）提交哈希：十六进制，7–64 位（与 ``changelog.py`` 同口径）。
HASH_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")

#: 允许的 type 集合（与 ROADMAP 提交规范的类型集合一致）。
ALLOWED_TYPES: tuple[str, ...] = (
    "feat",
    "fix",
    "perf",
    "refactor",
    "test",
    "docs",
    "ci",
    "chore",
)

#: 允许的 scope 集合（与本仓库历史使用过的范围一致；无 scope 的消息同样合法）。
ALLOWED_SCOPES: tuple[str, ...] = (
    "api",
    "ci",
    "deps",
    "domain",
    "lifecycle",
    "privacy",
    "proto",
    "runnable",
    "serde",
    "stores",
)

#: 主题行最大长度（ Conventional Commits 建议 ≤ 72 字符）。
MAX_SUBJECT_LENGTH = 72

#: 短哈希长度（与 ``git log --oneline`` 一致）。
SHORT_HASH_LENGTH = 7

#: CLI 退出码：全部合规。
EXIT_OK = 0

#: CLI 退出码：存在违规提交信息。
EXIT_VIOLATIONS = 1

#: CLI 退出码：用法错误或 git 调用失败。
EXIT_USAGE = 2


@dataclass(frozen=True)
class LintFinding:
    """一条违规发现。"""

    code: str
    message: str


@dataclass(frozen=True)
class LintReport:
    """单条提交信息的校验结果。"""

    commit: str | None
    subject: str
    findings: tuple[LintFinding, ...]

    @property
    def ok(self) -> bool:
        """无任何违规发现时为 ``True``。"""
        return not self.findings


def lint_message(message: str, *, commit: str | None = None) -> LintReport:
    """按 Conventional Commits 规范校验单条提交信息。

    校验规则（按出现顺序）：

    * ``empty-message``：去掉注释行（``#`` 开头）与首尾空行后内容为空；
    * ``subject-format``：主题行不匹配 ``type(scope)!: description``；
    * ``type-invalid``：type 不在 :data:`ALLOWED_TYPES` 中；
    * ``scope-invalid``：scope 不在 :data:`ALLOWED_SCOPES` 中；
    * ``description-trailing-whitespace``：描述尾部带空白；
    * ``description-trailing-period``：描述以句点结尾；
    * ``subject-too-long``：主题行超过 :data:`MAX_SUBJECT_LENGTH` 字符；
    * ``body-separator``：正文存在但未与主题行以空行分隔。

    Args:
        message: 完整提交信息（可含 git 注释行与 CRLF/CR 换行）。
        commit: 可选提交标识（短哈希），仅用于报告展示。

    Returns:
        校验结果；``findings`` 为空表示合规。
    """
    normalized = message.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line for line in normalized.split("\n") if not line.lstrip().startswith("#")]
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return LintReport(commit, "", (LintFinding("empty-message", "commit message is empty"),))
    subject = lines[0]
    body = lines[1:]
    findings: list[LintFinding] = []
    match = SUBJECT_PATTERN.match(subject)
    if match is None:
        findings.append(
            LintFinding("subject-format", 'subject must match "type(scope)!: description"')
        )
    else:
        commit_type = match.group("type")
        if commit_type not in ALLOWED_TYPES:
            findings.append(
                LintFinding(
                    "type-invalid",
                    f"unknown type {commit_type!r}; allowed: {', '.join(ALLOWED_TYPES)}",
                )
            )
        scope = match.group("scope")
        if scope is not None and scope not in ALLOWED_SCOPES:
            findings.append(
                LintFinding(
                    "scope-invalid",
                    f"unknown scope {scope!r}; allowed: {', '.join(ALLOWED_SCOPES)}",
                )
            )
        description = match.group("description")
        if description != description.rstrip():
            findings.append(
                LintFinding(
                    "description-trailing-whitespace", "description has trailing whitespace"
                )
            )
        if description.rstrip().endswith("."):
            findings.append(
                LintFinding("description-trailing-period", "description must not end with a period")
            )
        if len(subject) > MAX_SUBJECT_LENGTH:
            findings.append(
                LintFinding(
                    "subject-too-long",
                    f"subject is {len(subject)} chars; max is {MAX_SUBJECT_LENGTH}",
                )
            )
    if body and body[0].strip():
        findings.append(
            LintFinding("body-separator", "body must be separated from the subject by a blank line")
        )
    return LintReport(commit, subject, tuple(findings))


def parse_git_log_messages(text: str) -> list[tuple[str, str]]:
    """解析 ``git log`` 全文为 ``(完整哈希, 完整消息)`` 列表。

    Args:
        text: :data:`GIT_LOG_FORMAT` 格式的多条记录文本。

    Returns:
        保持 git 输出顺序（新→旧）；字段数不符、哈希非法、主题为空的记录被跳过。

    Note:
        先 ``rstrip("\\x1e")`` 再按字段分隔符切分——``\\x1e`` / ``\\x1f`` 会被
        ``str.strip()`` 当作空白字符，直接 ``strip()`` 会把正文为空时的字段分隔符
        一起吃掉，导致字段数误判（与 ``changelog.py`` 同一坑位）。
    """
    messages: list[tuple[str, str]] = []
    for record in text.split("\x1e"):
        fields = record.rstrip("\x1e").split("\x1f")
        if len(fields) != 3:
            continue
        commit_hash, subject, body = (field.strip() for field in fields)
        if HASH_PATTERN.fullmatch(commit_hash) is None or not subject:
            continue
        message = subject if not body else f"{subject}\n\n{body}"
        messages.append((commit_hash, message))
    return messages


def iter_commit_messages(
    rev_range: str | None = None, *, cwd: Path | None = None
) -> list[tuple[str, str]]:
    """调用 ``git log`` 收集 ``(完整哈希, 完整消息)``（本模块唯一触及外部进程的地方）。

    Args:
        rev_range: git 版本范围（如 ``origin/main..HEAD``）；缺省取全部历史。
        cwd: 执行 git 的工作目录；缺省为当前目录。

    Returns:
        ``parse_git_log_messages`` 的结果（新→旧）。

    Raises:
        subprocess.CalledProcessError: git 退出码非零（如范围非法、不是 git 仓库）。
        OSError: 找不到 git 可执行文件。
    """
    command = ["git", "log", f"--format={GIT_LOG_FORMAT}"]
    if rev_range:
        command.append(rev_range)
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=True)
    return parse_git_log_messages(completed.stdout)


def lint_commits(rev_range: str | None = None, *, cwd: Path | None = None) -> list[LintReport]:
    """校验一个版本范围内的全部提交信息。

    Args:
        rev_range: git 版本范围；缺省取全部历史。
        cwd: 执行 git 的工作目录；缺省为当前目录。

    Returns:
        逐条提交的 :class:`LintReport`（新→旧），``commit`` 为 7 位短哈希。

    Raises:
        subprocess.CalledProcessError: git 退出码非零。
        OSError: 找不到 git 可执行文件。
    """
    return [
        lint_message(message, commit=commit_hash[:SHORT_HASH_LENGTH])
        for commit_hash, message in iter_commit_messages(rev_range, cwd=cwd)
    ]


def render_report(reports: Sequence[LintReport]) -> str:
    """把校验结果渲染为人可读的文本报告。

    Args:
        reports: 校验结果列表。

    Returns:
        多行文本：逐条列出违规（``FAIL <commit> <subject>`` + 缩进的违规明细），
        末尾一行汇总（合规 / 违规计数）；无违规时只保留汇总行。
    """
    lines: list[str] = []
    for report in reports:
        if report.ok:
            continue
        label = report.commit or "<message>"
        lines.append(f"FAIL {label} {report.subject}")
        lines.extend(f"  - {finding.code}: {finding.message}" for finding in report.findings)
    bad = sum(1 for report in reports if not report.ok)
    total = len(reports)
    if bad:
        lines.append(f"{bad}/{total} commit message(s) violate Conventional Commits")
    else:
        lines.append(f"{total} commit message(s) conform to Conventional Commits")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器（选项供工作流静态测试交叉校验）。"""
    parser = argparse.ArgumentParser(
        prog="python -m med_langchain_memory.testing.commitlint",
        description="Lint commit messages against the Conventional Commits spec.",
    )
    parser.add_argument(
        "range",
        nargs="?",
        help='git revision range to lint, e.g. "origin/main..HEAD"',
    )
    parser.add_argument("--message", help="lint a single commit message given on the command line")
    parser.add_argument(
        "--message-file",
        type=Path,
        help="read the commit message from a file ('-' for stdin); the pre-commit commit-msg hook passes the message file here",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口。

    Args:
        argv: 命令行参数；缺省取 ``sys.argv``。

    Returns:
        ``0`` 全部合规；``1`` 存在违规；``2`` 用法错误（未指定或多指定输入源）、
        消息文件读不了或 git 调用失败，原因打印到 stderr。
    """
    args = build_parser().parse_args(argv)
    sources = [args.message is not None, args.message_file is not None, args.range is not None]
    if sum(sources) != 1:
        print(
            "commitlint: pass exactly one of --message, --message-file or a revision range",
            file=sys.stderr,
        )
        return EXIT_USAGE
    if args.message is not None:
        reports = [lint_message(args.message)]
    elif args.message_file is not None:
        try:
            if str(args.message_file) == "-":
                raw = sys.stdin.read()
            else:
                raw = args.message_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"commitlint: cannot read message file: {exc}", file=sys.stderr)
            return EXIT_USAGE
        reports = [lint_message(raw)]
    else:
        try:
            reports = lint_commits(args.range)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"commitlint: git log failed: {exc}", file=sys.stderr)
            return EXIT_USAGE
    print(render_report(reports))
    return EXIT_OK if all(report.ok for report in reports) else EXIT_VIOLATIONS


if __name__ == "__main__":  # pragma: no cover - 仅 `python -m` 入口
    raise SystemExit(main())
