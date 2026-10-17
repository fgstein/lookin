"""Writing files that other Lookin sessions read and write as well.

Several Lookin sessions routinely share files: every browser tab of
`lookin web` is a session, `lookin ui` and `lookin web` run side by side on
one machine, and two people point theirs at one library on a network mount.
Two rules follow, and this module is where both live.

**Atomic writes** (`atomic_write`, `atomic_write_text`). A file is written
to a temporary name in the same directory, then renamed over the target
with os.replace, so a reader -- or a crash, or a full disk -- never sees
half a file. The temporary name carries a random token besides the pid and
the thread: two machines on one mount can easily share a pid and a thread
id, and two writers sharing one temporary file is exactly the torn write
this exists to prevent. A failed write removes its temporary file instead
of leaving it beside the target forever.

**Read-modify-write under a lock** (`locked`). Atomic writes do not stop
two sessions from both reading a file, each changing it, and the second
write erasing the first. A lock file next to the target does, across
processes and machines: it is created with O_CREAT | O_EXCL, which is
atomic on local disks and on NFSv3+ / SMB alike, and needs nothing that
is not in the standard library. Threads of one process wait on an
in-process lock first, so they never spin on the file. Locks nest within a
thread (a mutator that re-reads through a locked loader does not deadlock
on itself).

The lock is advisory and deliberately forgiving: a lock file older than
LOCK_STALE_S is a crashed session's and is broken, and a session that
cannot get the lock within LOCK_TIMEOUT_S -- or cannot create a lock file
at all, on a read-only mount -- goes ahead without it and says so in the
log. A lost update is bad; a UI frozen forever by somebody else's crash is
worse.
"""
from __future__ import annotations

import os
import secrets
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

LOCK_TIMEOUT_S = 10.0
# Far longer than any write this app holds a lock for (a few hundred ms for
# the largest review store), so a live holder is never mistaken for a dead
# one; short enough that a crashed session does not block the next save for
# long.
LOCK_STALE_S = 30.0
_POLL_S = 0.02


def _log(msg: str) -> None:
    print(f"[lookin] {msg}", flush=True)


def temp_path(path) -> Path:
    """A temporary name beside `path` that no other writer will pick."""
    path = Path(path)
    return path.with_name(f".{path.name}.tmp{os.getpid()}.{threading.get_ident()}.{secrets.token_hex(4)}")


def atomic_write(path, write: Callable[[Path], None]) -> None:
    """Call `write(tmp)` to produce the file, then move it over `path`.
    Raises what the write raised, after removing the temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = temp_path(path)
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_text(path, text: str) -> None:
    atomic_write(path, lambda tmp: tmp.write_text(text, encoding="utf-8"))


def lock_path(path) -> Path:
    path = Path(path)
    return path.with_name(f".{path.name}.lock")


_registry_lock = threading.Lock()
_thread_locks: dict = {}          # lock file path -> threading.RLock
_held = threading.local()         # .depth: {lock file path: nesting depth}


def _thread_lock(key: str) -> threading.RLock:
    with _registry_lock:
        lock = _thread_locks.get(key)
        if lock is None:
            lock = _thread_locks[key] = threading.RLock()
        return lock


def _try_break_stale(lock_file: Path, stale_after: float) -> None:
    try:
        age = time.time() - lock_file.stat().st_mtime
    except OSError:
        return
    if age > stale_after:
        try:
            lock_file.unlink()
            _log(f"broke a stale lock {lock_file} ({age:.0f}s old)")
        except OSError:
            pass


def _acquire_file(lock_file: Path, timeout: float, stale_after: float) -> bool:
    """True once the lock file is ours; False when going ahead without it."""
    deadline = time.monotonic() + timeout
    if not lock_file.parent.is_dir():
        # Never created here: that would resurrect a workspace directory
        # deleted meanwhile. A directory that does not exist yet (a first
        # run's config directory) has nobody to race with.
        return False
    while True:
        try:
            fd = os.open(str(lock_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            _try_break_stale(lock_file, stale_after)
        except OSError as ex:
            # Read-only directory, missing directory, a filesystem refusing
            # O_EXCL: nobody else can take the lock here either, and the
            # write that follows will report its own failure.
            _log(f"could not lock {lock_file} ({ex}); going ahead without it")
            return False
        else:
            try:
                os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode("utf-8", "replace"))
            finally:
                os.close(fd)
            return True
        if time.monotonic() >= deadline:
            _log(f"{lock_file} still held after {timeout:.0f}s; going ahead without it")
            return False
        time.sleep(_POLL_S)


@contextmanager
def locked(path, timeout: float = LOCK_TIMEOUT_S, stale_after: float = LOCK_STALE_S):
    """Hold the lock on `path` (any file, existing or not) for the block."""
    lock_file = lock_path(path)
    key = os.path.abspath(str(lock_file))
    depth = getattr(_held, "depth", None)
    if depth is None:
        depth = _held.depth = {}
    if depth.get(key):
        depth[key] += 1
        try:
            yield
        finally:
            depth[key] -= 1
        return
    thread_lock = _thread_lock(key)
    with thread_lock:
        owned = _acquire_file(lock_file, timeout, stale_after)
        depth[key] = 1
        try:
            yield
        finally:
            depth[key] = 0
            if owned:
                try:
                    lock_file.unlink()
                except OSError:
                    pass
