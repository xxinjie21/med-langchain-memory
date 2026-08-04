"""跨进程独占文件锁 :class:`FileLock`。

文件存储适配器在多进程（如多 worker 的 API 服务）下追加写同一会话文件时，
需要内核级互斥才能保证记录不交错。本模块只用标准库实现：

* **Windows**：``msvcrt.locking`` 对锁文件首字节加非阻塞独占锁；
* **POSIX**：``fcntl.flock`` 加非阻塞独占锁。

同一进程内针对同一路径的多个 :class:`FileLock` 实例共享一份锁状态，
因此可重入且线程安全；跨进程则完全依赖内核文件锁。

本模块不含任何文本内容解析逻辑。
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import ClassVar

from med_langchain_memory.exceptions import StorageError, ValidationError

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


def lock_fd(fd: int) -> None:
    """对文件描述符加非阻塞独占锁。

    Args:
        fd: 已打开的锁文件描述符（文件位置须在 0）。

    Raises:
        OSError: 锁已被其他描述符/进程持有时立即失败。
    """
    if sys.platform == "win32":
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def unlock_fd(fd: int) -> None:
    """释放文件描述符上的独占锁。

    Args:
        fd: 此前由 :func:`lock_fd` 加锁的描述符。
    """
    if sys.platform == "win32":
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)


@dataclass
class _LockState:
    """同一进程内按锁文件路径共享的锁状态。

    Attributes:
        guard: 进程内互斥用的可重入锁。
        fd: 持有内核锁的文件描述符，未持锁时为 ``None``。
        depth: 重入层数，``0`` 表示当前未持锁。
    """

    guard: threading.RLock = field(default_factory=threading.RLock)
    fd: int | None = None
    depth: int = 0


class FileLock:
    """基于锁文件的跨进程独占锁，可作为上下文管理器使用。

    Example:
        >>> import tempfile, pathlib
        >>> path = pathlib.Path(tempfile.mkdtemp()) / "session.lock"
        >>> with FileLock(path) as lock:
        ...     lock.is_held
        True
    """

    #: 进程内共享锁状态表：规范化锁文件路径 -> :class:`_LockState`。
    _states: ClassVar[dict[str, _LockState]] = {}

    #: 保护 :attr:`_states` 自身的互斥锁。
    _states_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        path: str | Path,
        *,
        timeout: float = 10.0,
        poll_interval: float = 0.02,
    ) -> None:
        """初始化文件锁。

        Args:
            path: 锁文件路径，父目录会在加锁时自动创建。
            timeout: 获取锁的最长等待秒数，超时抛 :class:`StorageError`。
            poll_interval: 内核锁重试轮询间隔（秒）。

        Raises:
            ValidationError: ``timeout`` 为负数或 ``poll_interval`` 非正数时。
        """
        if timeout < 0:
            raise ValidationError("timeout must be a non-negative number")
        if poll_interval <= 0:
            raise ValidationError("poll_interval must be a positive number")
        self._path = Path(path)
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._state = self._state_for(self._path)

    @property
    def path(self) -> Path:
        """锁文件路径。"""
        return self._path

    @property
    def is_held(self) -> bool:
        """当前进程是否正持有该锁。"""
        return self._state.depth > 0

    def acquire(self) -> None:
        """获取独占锁；同线程重入只增加计数，不重复加内核锁。

        Raises:
            StorageError: 等待超过 ``timeout`` 仍未获得锁时。
        """
        if not self._state.guard.acquire(timeout=self._timeout):
            raise StorageError(f"timeout acquiring in-process lock for {self._path}")
        if self._state.depth > 0:
            self._state.depth += 1
            return
        try:
            self._state.fd = self._open_locked_fd()
        except BaseException:
            self._state.guard.release()
            raise
        self._state.depth = 1

    def release(self) -> None:
        """释放一层锁；计数归零时才真正释放内核锁并关闭描述符。

        Raises:
            StorageError: 当前未持有该锁时。
        """
        if self._state.depth == 0:
            raise StorageError(f"cannot release unheld lock {self._path}")
        self._state.depth -= 1
        if self._state.depth == 0 and self._state.fd is not None:
            fd = self._state.fd
            self._state.fd = None
            try:
                unlock_fd(fd)
            finally:
                os.close(fd)
        self._state.guard.release()

    def __enter__(self) -> FileLock:
        """进入上下文即加锁。"""
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """离开上下文即解锁（异常也保证释放）。"""
        self.release()

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _open_locked_fd(self) -> int:
        """打开锁文件并在超时窗口内重试加内核锁，返回已加锁的描述符。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o644)
        deadline = time.monotonic() + self._timeout
        while True:
            try:
                lock_fd(fd)
                return fd
            except OSError as exc:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    raise StorageError(f"timeout acquiring file lock {self._path}") from exc
                time.sleep(self._poll_interval)

    @classmethod
    def _state_for(cls, path: Path) -> _LockState:
        """取得（或创建）给定路径在本进程内的共享锁状态。"""
        key = os.path.normcase(os.path.abspath(path))
        with cls._states_guard:
            return cls._states.setdefault(key, _LockState())
