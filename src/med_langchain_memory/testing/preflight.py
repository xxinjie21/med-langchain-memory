"""集成测试前置守卫：把「静默跳过」变成「显式失败」。

问题：集成用例在服务不可达时会 ``pytest.skip``（这是本地开发的正确行为），
但在 CI 上表现为**全绿却一条真机断言都没跑**的假绿灯——服务名拼错、
容器没起来、端口映射写错，都只会安静地变成一堆 ``skipped``。

本模块在 ``pytest`` 之前运行：按 ``MED_MEMORY_IT_REQUIRED`` 声明的服务清单
逐个探针，任一不可用即打印原因并以退出码 1 结束，让假绿灯变成红灯。

设计取舍：

* 只做 TCP 可达性判断（复用 :mod:`med_langchain_memory.testing.services`），
  **不导入任何中间件客户端**，也不做协议握手——「集群是否已收敛」这类判断
  交给用例自己的夹具（如 ``CLUSTER INFO`` 轮询）；
* 未声明任何必需服务时视为通过（本地手工跑 ``pytest`` 不受影响）；
* 服务名非法不抛异常，而是记进报告的 ``errors`` —— 守卫的价值在于把问题**列全**。

本模块不含任何文本预处理 / 语义解析逻辑。
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from med_langchain_memory.exceptions import ValidationError

from .services import ServiceStatus, check_service, resolve_service

#: 声明「必须可用」的服务清单的环境变量名（逗号分隔）。
#:
#: 例如 ``MED_MEMORY_IT_REQUIRED=redis,mysql,elasticsearch,redis-cluster``。
ENV_REQUIRED = "MED_MEMORY_IT_REQUIRED"


@dataclass(frozen=True)
class PreflightReport:
    """一次前置检查的结果汇总。"""

    #: 逐个服务的检查结果（顺序与声明顺序一致）。
    statuses: tuple[ServiceStatus, ...] = ()

    #: 无法解析的服务名对应的错误信息（如拼写错误）。
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """全部声明的服务是否都可用于集成测试。"""
        return not self.errors and all(status.ready for status in self.statuses)

    @property
    def failures(self) -> tuple[str, ...]:
        """所有问题的可读描述（服务名非法 + 服务未就绪），可用于日志与退出信息。"""
        missing = tuple(
            status.reason if status.reason is not None else status.service.describe()
            for status in self.statuses
            if not status.ready
        )
        return (*self.errors, *missing)

    def render(self) -> str:
        """渲染成多行文本（逐服务一行，末行为总判定）。

        Returns:
            可直接 ``print`` 的报告文本。
        """
        lines = [f"preflight: {len(self.statuses)} required service(s)"]
        for status in self.statuses:
            if status.ready:
                lines.append(f"  [ok]   {status.service.describe()}")
            else:
                # 未就绪原因由 check_service 生成，本身已含服务地址与覆盖变量名，
                # 这里不再重复前缀（否则输出里同一串会连写两遍）。
                lines.append(f"  [fail] {status.reason}")
        for error in self.errors:
            lines.append(f"  [fail] {error}")
        if self.ok:
            lines.append("preflight passed")
        else:
            lines.append(f"preflight failed: {len(self.failures)} problem(s)")
        return "\n".join(lines)


def parse_required(env: Mapping[str, str] | None = None) -> tuple[str, ...]:
    """解析必需服务清单。

    Args:
        env: 环境变量映射；``None`` 时读取 ``os.environ``。

    Returns:
        逗号分隔解析出的服务名元组：去除首尾空白、丢弃空项、去重且保持声明顺序；
        变量缺失或全为空项时返回空元组。
    """
    environ = os.environ if env is None else env
    unique: dict[str, None] = {}
    for chunk in environ.get(ENV_REQUIRED, "").split(","):
        name = chunk.strip()
        if name:
            unique.setdefault(name, None)
    return tuple(unique)


def preflight(
    names: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    wait_seconds: float | None = None,
) -> PreflightReport:
    """逐个检查声明的服务是否可用于集成测试。

    Args:
        names: 服务名序列（``redis`` / ``mysql`` / ``elasticsearch`` / ``redis-cluster``）。
        env: 环境变量映射；``None`` 时读取 ``os.environ``。
        wait_seconds: 非 ``None`` 时按该上限轮询等待服务就绪（容器冷启动场景）；
            ``None`` 时每个服务只探测一次。

    Returns:
        :class:`PreflightReport`；服务名非法时对应条目落在 ``errors``，
        不中断对其余服务的检查。
    """
    statuses: list[ServiceStatus] = []
    errors: list[str] = []
    for name in names:
        try:
            service = resolve_service(name, env)
        except ValidationError as exc:
            errors.append(str(exc))
            continue
        statuses.append(check_service(service, env=env, wait_seconds=wait_seconds))
    return PreflightReport(tuple(statuses), tuple(errors))


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（供 CI 与本地直接调用）。"""
    parser = argparse.ArgumentParser(
        prog="med-memory-preflight",
        description="Fail fast when declared integration services are not reachable.",
    )
    parser.add_argument(
        "services",
        nargs="*",
        metavar="SERVICE",
        help=f"required services; defaults to ${ENV_REQUIRED} (comma separated)",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=None,
        metavar="SECONDS",
        help="poll each service for up to SECONDS before giving up (default: probe once)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """命令行入口：打印报告并返回退出码。

    Args:
        argv: 参数列表；``None`` 时读取 ``sys.argv[1:]``。

    Returns:
        ``0`` 表示全部声明服务就绪（或未声明任何服务），``1`` 表示存在未就绪 / 非法服务。
    """
    args = build_parser().parse_args(argv)
    positional: list[str] = list(args.services)
    names: tuple[str, ...] = tuple(positional) or parse_required()
    wait: float | None = args.wait
    report = preflight(names, wait_seconds=wait)
    print(report.render())
    return 0 if report.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
