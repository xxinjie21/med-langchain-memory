"""发布守卫（release guard）。

`.github/workflows/release.yml` 在 ``v*`` tag 推送时调用本模块，把三方版本
对齐起来，并校验 ``python -m build`` 的产物是否完整：

1. ``git tag``（如 ``v0.1.0``）
2. ``pyproject.toml`` 的 ``[project].version``
3. 包内 ``med_langchain_memory.__version__``

三方不一致是发布事故的常见来源（改了代码忘了改版本、tag 打错、dist 目录里躺着
上一次构建的旧 wheel），因此在构建前、发布前各卡一道。

本模块**只用标准库**（``argparse`` / ``re`` / ``tomllib`` / ``zipfile`` / ``tarfile``），
不引入任何新依赖，也不参与运行时链路。

命令行用法::

    python -m med_langchain_memory.release --tag v0.1.0
    python -m med_langchain_memory.release --dist dist --expected-version v0.1.0
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import tomllib
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from . import __version__ as PACKAGE_VERSION

#: 发布 tag 的固定前缀：``v0.1.0``。
TAG_PREFIX = "v"

#: 严格语义化版本（本库不使用预发布 / 构建元数据后缀）。
SEMVER_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")

#: 带前缀的发布 tag。
TAG_PATTERN = re.compile(rf"^{TAG_PREFIX}(\d+\.\d+\.\d+)$")

#: wheel ``*.dist-info/METADATA`` 中的版本行。
METADATA_VERSION_PATTERN = re.compile(r"^Version:\s*(\S+)\s*$", re.MULTILINE)

#: wheel 中必须存在的成员（缺任一即视为打包配置被破坏）。
REQUIRED_WHEEL_MEMBERS: tuple[str, ...] = (
    "med_langchain_memory/__init__.py",
    "med_langchain_memory/release.py",
    "med_langchain_memory/py.typed",
    "med_langchain_memory/serde/med_session_pb2.py",
    "med_langchain_memory/serde/protobuf_serializer.py",
)

#: sdist 中必须存在的成员后缀。
REQUIRED_SDIST_MEMBERS: tuple[str, ...] = ("pyproject.toml", "protos/med_session.proto")


def default_pyproject_path() -> Path:
    """返回仓库根的 ``pyproject.toml`` 路径（基于源码布局推断）。

    仅适用于「源码 / editable 安装」场景；已安装的 wheel 里没有 pyproject，
    因此调用方可用 ``--pyproject`` 显式指定。
    """
    return Path(__file__).resolve().parents[2] / "pyproject.toml"


def is_semver(value: str) -> bool:
    """判断 ``value`` 是否为 ``major.minor.patch`` 形态。

    Args:
        value: 待判定的版本字符串。

    Returns:
        是严格三段式数字版本时返回 ``True``。
    """
    return SEMVER_PATTERN.match(value) is not None


def normalize_version(value: str) -> str:
    """把 ``v0.1.0`` 或 ``0.1.0`` 统一成 ``0.1.0``。

    Args:
        value: 带或不带 ``v`` 前缀的版本字符串。

    Returns:
        去前缀后的语义化版本。

    Raises:
        ValueError: 去掉前缀后仍不是语义化版本。
    """
    candidate = value.strip()
    if candidate.startswith(TAG_PREFIX):
        candidate = candidate[len(TAG_PREFIX) :]
    if not is_semver(candidate):
        raise ValueError(f"invalid version: {value!r} (expected '<major>.<minor>.<patch>')")
    return candidate


def parse_tag(tag: str) -> str:
    """把发布 tag 归一化为版本号。

    Args:
        tag: git tag 名，必须形如 ``v0.1.0``。

    Returns:
        去前缀后的版本号。

    Raises:
        ValueError: tag 缺少 ``v`` 前缀或不是三段式版本。
    """
    match = TAG_PATTERN.match(tag.strip())
    if match is None:
        raise ValueError(
            f"invalid release tag: {tag!r} (expected '{TAG_PREFIX}<major>.<minor>.<patch>')"
        )
    return match.group(1)


def read_project_version(pyproject_path: Path) -> str:
    """读取 ``pyproject.toml`` 中声明的版本。

    Args:
        pyproject_path: ``pyproject.toml`` 路径。

    Returns:
        ``[project].version`` 的值。

    Raises:
        FileNotFoundError: 路径不存在。
        ValueError: 文件缺少 ``[project].version`` 或类型不是字符串。
    """
    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    version = data.get("project", {}).get("version")
    if not isinstance(version, str):
        raise ValueError(f"missing [project].version in {pyproject_path}")
    return version


@dataclass(frozen=True)
class ReleaseCheck:
    """tag / pyproject / 包内版本的三方一致性结论。

    Attributes:
        tag: 原始 tag 文本。
        version: 归一化后的版本号。
        project_version: ``pyproject.toml`` 声明的版本。
        package_version: 包内 ``__version__``。
        errors: 不一致项描述；为空即通过。
    """

    tag: str
    version: str
    project_version: str
    package_version: str
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """三方版本是否完全一致。"""
        return not self.errors

    def summary(self) -> str:
        """返回可读的多行报告，供 CI 日志直接打印。"""
        head = "release check OK" if self.ok else "release check FAILED"
        lines = [
            f"{head}: tag={self.tag} version={self.version} "
            f"pyproject={self.project_version} package={self.package_version}"
        ]
        lines.extend(f"  ! {error}" for error in self.errors)
        return "\n".join(lines)


def check_release(
    tag: str,
    *,
    pyproject_path: Path | None = None,
    package_version: str = PACKAGE_VERSION,
) -> ReleaseCheck:
    """校验 tag 与 ``pyproject.toml`` / 包内版本一致。

    Args:
        tag: git tag 名，必须形如 ``v0.1.0``。
        pyproject_path: ``pyproject.toml`` 路径；``None`` 时用仓库根默认值。
        package_version: 包内 ``__version__``，默认取当前包的真实值。

    Returns:
        校验结论（不一致时 ``ok`` 为 ``False``，``errors`` 列出全部差异）。

    Raises:
        ValueError: tag 形态非法。
        FileNotFoundError: ``pyproject.toml`` 不存在。
    """
    version = parse_tag(tag)
    project_version = read_project_version(pyproject_path or default_pyproject_path())

    errors: list[str] = []
    if project_version != version:
        errors.append(f"pyproject version {project_version!r} != tag version {version!r}")
    if package_version != version:
        errors.append(f"package __version__ {package_version!r} != tag version {version!r}")

    return ReleaseCheck(
        tag=tag,
        version=version,
        project_version=project_version,
        package_version=package_version,
        errors=tuple(errors),
    )


@dataclass(frozen=True)
class DistCheck:
    """``python -m build`` 产物的完整性结论。

    Attributes:
        dist_dir: 被检查的 dist 目录。
        artifacts: 目录下实际存在的文件名（排序后）。
        metadata_version: wheel ``METADATA`` 中声明的版本；读不到时为 ``None``。
        errors: 缺失 / 不一致项描述；为空即通过。
    """

    dist_dir: Path
    artifacts: tuple[str, ...]
    metadata_version: str | None
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """产物是否齐全且版本正确。"""
        return not self.errors

    def summary(self) -> str:
        """返回可读的多行报告，供 CI 日志直接打印。"""
        head = "dist check OK" if self.ok else "dist check FAILED"
        lines = [
            f"{head}: {self.dist_dir} "
            f"({len(self.artifacts)} artifact(s), metadata={self.metadata_version})"
        ]
        lines.extend(f"  - {name}" for name in self.artifacts)
        lines.extend(f"  ! {error}" for error in self.errors)
        return "\n".join(lines)


def _read_metadata_version(archive: zipfile.ZipFile, names: Sequence[str]) -> str | None:
    """从 wheel 的 ``*.dist-info/METADATA`` 中解析版本号。"""
    for name in names:
        if name.endswith(".dist-info/METADATA"):
            match = METADATA_VERSION_PATTERN.search(archive.read(name).decode("utf-8"))
            if match is not None:
                return match.group(1)
    return None


def _check_wheel(wheel: Path, required_members: Sequence[str]) -> tuple[str | None, list[str]]:
    """检查单个 wheel 的必需成员与元数据版本。"""
    errors: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        errors.extend(
            f"{wheel.name} missing {member}" for member in required_members if member not in names
        )
        metadata_version = _read_metadata_version(archive, names)
    return metadata_version, errors


def _check_sdist(sdist: Path, required_members: Sequence[str]) -> list[str]:
    """检查 sdist 中是否包含必需成员（按路径后缀匹配）。"""
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    return [
        f"{sdist.name} missing {member}"
        for member in required_members
        if not any(name.endswith(member) for name in names)
    ]


def inspect_dist(
    dist_dir: Path,
    *,
    expected_version: str | None = None,
    required_wheel_members: Sequence[str] = REQUIRED_WHEEL_MEMBERS,
    required_sdist_members: Sequence[str] = REQUIRED_SDIST_MEMBERS,
) -> DistCheck:
    """校验 dist 目录下的 sdist / wheel 是否齐全且内容正确。

    Args:
        dist_dir: ``python -m build`` 的输出目录。
        expected_version: 期望版本（可带 ``v`` 前缀）；给定时与 wheel 元数据比对。
        required_wheel_members: wheel 中必须存在的成员路径。
        required_sdist_members: sdist 中必须存在的成员路径后缀。

    Returns:
        校验结论。

    Raises:
        ValueError: ``expected_version`` 不是合法版本号。
    """
    expected = None if expected_version is None else normalize_version(expected_version)
    if not dist_dir.is_dir():
        return DistCheck(dist_dir, (), None, (f"dist directory not found: {dist_dir}",))

    artifacts = tuple(sorted(p.name for p in dist_dir.iterdir() if p.is_file()))
    wheels = sorted(dist_dir.glob("*.whl"))
    sdists = sorted(dist_dir.glob("*.tar.gz"))

    errors: list[str] = []
    if not wheels:
        errors.append("no wheel (*.whl) found")
    if not sdists:
        errors.append("no sdist (*.tar.gz) found")
    if len(wheels) > 1:
        errors.append(f"multiple wheels found: {[p.name for p in wheels]}")

    metadata_version: str | None = None
    if wheels:
        metadata_version, wheel_errors = _check_wheel(wheels[0], required_wheel_members)
        errors.extend(wheel_errors)
    if sdists:
        errors.extend(_check_sdist(sdists[0], required_sdist_members))

    if expected is not None:
        if metadata_version is None:
            errors.append("wheel METADATA Version not found")
        elif metadata_version != expected:
            errors.append(f"wheel metadata version {metadata_version!r} != expected {expected!r}")

    return DistCheck(dist_dir, artifacts, metadata_version, tuple(errors))


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（供 ``main`` 与测试复用）。"""
    parser = argparse.ArgumentParser(
        prog="python -m med_langchain_memory.release",
        description="Release guard: keep tag / pyproject / package version in sync "
        "and verify build artifacts.",
    )
    parser.add_argument("--tag", help="git tag to validate, e.g. v0.1.0")
    parser.add_argument("--dist", type=Path, help="dist directory produced by 'python -m build'")
    parser.add_argument(
        "--expected-version",
        help="version the wheel metadata must declare (accepts 'v0.1.0' or '0.1.0')",
    )
    parser.add_argument(
        "--pyproject",
        type=Path,
        help="path to pyproject.toml (defaults to the repository root)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口。

    Args:
        argv: 参数列表；``None`` 时取 ``sys.argv[1:]``。

    Returns:
        全部检查通过返回 ``0``，任一项失败返回 ``1``；
        参数非法由 argparse 直接以退出码 ``2`` 结束。
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.tag is None and args.dist is None:
        parser.error("at least one of --tag / --dist is required")

    exit_code = 0
    if args.tag is not None:
        try:
            release_check = check_release(args.tag, pyproject_path=args.pyproject)
        except (ValueError, FileNotFoundError) as exc:
            print(f"release check FAILED: {exc}", file=sys.stderr)
            return 1
        print(release_check.summary())
        exit_code |= 0 if release_check.ok else 1

    if args.dist is not None:
        try:
            dist_check = inspect_dist(args.dist, expected_version=args.expected_version)
        except (ValueError, OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
            print(f"dist check FAILED: {exc}", file=sys.stderr)
            return 1
        print(dist_check.summary())
        exit_code |= 0 if dist_check.ok else 1

    return exit_code


if __name__ == "__main__":  # pragma: no cover - 仅 `python -m` 入口
    raise SystemExit(main())
