"""FileLock 跨进程文件锁单元测试。

覆盖三个层面：

* 上下文管理与重入语义（同线程重入、释放计数、未持锁释放）；
* 进程内互斥：第二个线程在超时窗口内拿不到锁时抛 :class:`StorageError`；
* 内核锁语义：直接对两个文件描述符加锁，验证真正的操作系统级互斥。

全部用例零外部依赖，仅使用 ``tmp_path`` 与标准库线程。
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from med_langchain_memory.exceptions import StorageError, ValidationError
from med_langchain_memory.stores import file_lock as file_lock_module
from med_langchain_memory.stores.file_lock import FileLock, lock_fd, unlock_fd


@pytest.fixture
def lock_path(tmp_path: Path) -> Path:
    """位于临时目录下、尚未创建的锁文件路径。"""
    return tmp_path / "nested" / "session.lock"


# --------------------------------------------------------------------------- #
# 基本加解锁
# --------------------------------------------------------------------------- #
def test_acquire_creates_lock_file_and_marks_held(lock_path: Path) -> None:
    lock = FileLock(lock_path)
    assert lock.is_held is False

    lock.acquire()
    try:
        assert lock.is_held is True
        assert lock_path.is_file(), "锁文件及其父目录应被自动创建"
    finally:
        lock.release()

    assert lock.is_held is False


def test_context_manager_acquires_and_releases(lock_path: Path) -> None:
    lock = FileLock(lock_path)

    with lock as entered:
        assert entered is lock
        assert lock.is_held is True

    assert lock.is_held is False


def test_context_manager_releases_on_exception(lock_path: Path) -> None:
    lock = FileLock(lock_path)

    with pytest.raises(RuntimeError), lock:
        raise RuntimeError("boom")

    assert lock.is_held is False


def test_reentrant_acquire_in_same_thread(lock_path: Path) -> None:
    lock = FileLock(lock_path)

    with lock, FileLock(lock_path):
        assert lock.is_held is True

    assert lock.is_held is False


def test_release_without_acquire_raises(lock_path: Path) -> None:
    lock = FileLock(lock_path)

    with pytest.raises(StorageError, match="cannot release unheld lock"):
        lock.release()


def test_path_property_exposes_lock_file(lock_path: Path) -> None:
    assert FileLock(lock_path).path == lock_path


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #
def test_negative_timeout_rejected(lock_path: Path) -> None:
    with pytest.raises(ValidationError, match="timeout must be"):
        FileLock(lock_path, timeout=-1)


@pytest.mark.parametrize("poll_interval", [0, -0.5])
def test_non_positive_poll_interval_rejected(lock_path: Path, poll_interval: float) -> None:
    with pytest.raises(ValidationError, match="poll_interval must be"):
        FileLock(lock_path, poll_interval=poll_interval)


# --------------------------------------------------------------------------- #
# 进程内互斥
# --------------------------------------------------------------------------- #
def test_instances_on_same_path_share_state(lock_path: Path) -> None:
    first = FileLock(lock_path)
    second = FileLock(lock_path)

    with first:
        assert second.is_held is True, "同一路径的实例共享进程内锁状态"

    assert second.is_held is False


def test_second_thread_times_out_while_locked(lock_path: Path) -> None:
    holder = FileLock(lock_path)
    failures: list[Exception] = []

    def contend() -> None:
        try:
            with FileLock(lock_path, timeout=0.05):
                pass
        except Exception as exc:
            failures.append(exc)

    holder.acquire()
    try:
        worker = threading.Thread(target=contend)
        worker.start()
        worker.join(timeout=5)
        assert not worker.is_alive()
    finally:
        holder.release()

    assert len(failures) == 1
    assert isinstance(failures[0], StorageError)
    assert "timeout acquiring in-process lock" in str(failures[0])


def test_lock_is_reusable_after_release(lock_path: Path) -> None:
    lock = FileLock(lock_path)

    with lock:
        pass
    with lock:
        assert lock.is_held is True

    assert lock.is_held is False


# --------------------------------------------------------------------------- #
# 内核级互斥（绕过进程内 guard，直接操作描述符）
# --------------------------------------------------------------------------- #
def test_kernel_lock_timeout_releases_in_process_guard(
    lock_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def always_busy(fd: int) -> None:
        raise OSError("locked by another process")

    monkeypatch.setattr(file_lock_module, "lock_fd", always_busy)
    lock = FileLock(lock_path, timeout=0.05, poll_interval=0.01)

    with pytest.raises(StorageError, match="timeout acquiring file lock"):
        lock.acquire()

    assert lock.is_held is False
    monkeypatch.undo()
    with FileLock(lock_path, timeout=0.5):
        pass


def test_kernel_lock_blocks_other_descriptor(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    first = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    second = os.open(lock_path, os.O_RDWR | os.O_CREAT)
    try:
        lock_fd(first)

        with pytest.raises(OSError):
            lock_fd(second)

        unlock_fd(first)
        lock_fd(second)
        unlock_fd(second)
    finally:
        os.close(first)
        os.close(second)
