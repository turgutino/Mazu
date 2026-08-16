"""Real concurrency/crash-safety tests for the checkpoint index lock (mazu/checkpoint/
lock.py) and the atomicity hardening in mazu/checkpoint/store.py + manager.py.

These are deliberately heavier than typical unit tests -- they spawn real OS
subprocesses and real threads, because the bug class being guarded against (two
processes racing on index.json, a process crashing mid-lock) cannot be proven with
single-threaded, single-process assertions alone.
"""

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from mazu.checkpoint.lock import LockTimeoutError, ReentrantFileLock
from mazu.checkpoint.manager import CheckpointManager
from mazu.checkpoint.store import CheckpointIndex


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / ".gitignore").write_text(".mazu/\n", encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# ReentrantFileLock -- direct unit tests
# ---------------------------------------------------------------------------


def test_lock_is_reentrant_within_same_thread(tmp_path: Path):
    lock = ReentrantFileLock(tmp_path / ".index.lock")
    with lock.acquire():
        with lock.acquire():  # would deadlock against itself if not reentrant
            with lock.acquire():
                pass  # reaching here at all proves reentrancy works


def test_lock_serializes_two_threads(tmp_path: Path):
    """Two threads sharing ONE ReentrantFileLock instance must never run their
    critical sections concurrently -- proven by an overlap counter that would go
    above 1 if the lock let both in at once.
    """
    lock = ReentrantFileLock(tmp_path / ".index.lock")
    concurrent_count = 0
    max_concurrent = 0
    counter_guard = threading.Lock()  # protects the plain Python counter itself

    def worker():
        nonlocal concurrent_count, max_concurrent
        for _ in range(20):
            with lock.acquire():
                with counter_guard:
                    concurrent_count += 1
                    max_concurrent = max(max_concurrent, concurrent_count)
                time.sleep(0.005)
                with counter_guard:
                    concurrent_count -= 1

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "worker thread hung -- possible deadlock"

    assert max_concurrent == 1


def test_lock_released_after_holder_process_is_killed(tmp_path: Path):
    """The core justification for using an OS-level advisory lock (flock/
    msvcrt.locking) instead of a lock-file-existence scheme: the OS releases the
    lock automatically when the holding process dies for ANY reason, including a
    hard kill. A lock-file-existence scheme would instead leave a permanently
    stuck lock file behind, wedging every future checkpoint operation until a
    human manually deletes it -- this test proves that failure mode can't happen.
    """
    lock_path = tmp_path / ".index.lock"
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parent.parent)!r})
        from pathlib import Path as _Path
        from mazu.checkpoint.lock import ReentrantFileLock
        import time
        lock = ReentrantFileLock(_Path({str(lock_path)!r}))
        with lock.acquire():
            print("LOCKED", flush=True)
            time.sleep(60)
    """)
    proc = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        # Wait for confirmation the child actually holds the lock before killing it
        # -- otherwise this test could pass for the wrong reason (child never
        # started in time, so there was never real contention).
        first_line = proc.stdout.readline()
        assert first_line.strip() == "LOCKED"

        proc.kill()  # hard kill, no chance for the child to release cleanly
        proc.wait(timeout=10)

        # A second acquisition attempt, from this (the test) process, must succeed
        # quickly -- if it hangs until LockTimeoutError, the lock leaked.
        lock = ReentrantFileLock(lock_path, timeout=10.0)
        start = time.monotonic()
        with lock.acquire():
            pass
        elapsed = time.monotonic() - start
        assert elapsed < 5.0, f"lock took {elapsed}s to acquire after holder was killed"
    finally:
        if proc.poll() is None:
            proc.kill()


def test_lock_timeout_raises_when_never_released(tmp_path: Path):
    """A genuinely stuck holder (still alive, never releasing) must surface as a
    clear, bounded-time error, not an infinite hang -- confirms the timeout path
    itself actually fires rather than only ever being reached in theory.
    """
    lock_path = tmp_path / ".index.lock"
    holder = ReentrantFileLock(lock_path)

    with holder.acquire():
        waiter = ReentrantFileLock(lock_path, timeout=0.3)
        with pytest.raises(LockTimeoutError):
            with waiter.acquire():
                pass


# ---------------------------------------------------------------------------
# CheckpointIndex -- atomic save() correctness
# ---------------------------------------------------------------------------


def test_index_save_leaves_no_stray_temp_files(tmp_path: Path):
    index = CheckpointIndex(tmp_path)
    index.append({"id": "cp_000001", "step": 1})
    index.append({"id": "cp_000002", "step": 2})

    leftover = list(tmp_path.glob("*.tmp"))
    assert leftover == []
    assert index.load() == [{"id": "cp_000001", "step": 1}, {"id": "cp_000002", "step": 2}]


def test_index_save_failure_does_not_corrupt_existing_file(tmp_path: Path, monkeypatch):
    """If json.dumps (or the write itself) raises partway through save(), the REAL
    index.json on disk must be left exactly as it was before the failed call --
    proving the temp-file-then-os.replace technique actually protects against a
    torn/corrupted write, not just against the happy path.
    """
    index = CheckpointIndex(tmp_path)
    index.append({"id": "cp_000001", "step": 1})
    before = index.index_path.read_text(encoding="utf-8")

    import mazu.checkpoint.store as store_module

    def _boom(*a, **k):
        raise OSError("simulated disk failure mid-write")

    monkeypatch.setattr(store_module.os, "fsync", _boom)

    with pytest.raises(OSError):
        index.save([{"id": "cp_000001", "step": 1}, {"id": "cp_000002", "step": 2}])

    after = index.index_path.read_text(encoding="utf-8")
    assert after == before  # untouched -- the failed write never replaced the real file
    assert list(tmp_path.glob("*.tmp")) == []  # temp file cleaned up, not left orphaned


# ---------------------------------------------------------------------------
# CheckpointManager.snapshot() -- orphan cleanup on failure
# ---------------------------------------------------------------------------


def test_snapshot_failure_cleans_up_orphaned_checkpoint_dir(project: Path, monkeypatch):
    """If something fails partway through writing a checkpoint's files (here:
    copying the skills directory), the partially-written checkpoint_dir must not
    be left behind on disk, and no index entry should exist for it either.
    """
    (project / "a.py").write_text("print('a')")
    manager = CheckpointManager(project)
    manager.skills_dir.mkdir(parents=True)
    (manager.skills_dir / "a_skill.py").write_text("def run(args): return 'x'")

    import shutil as shutil_module

    real_copytree = shutil_module.copytree

    def _boom(*a, **k):
        raise OSError("simulated failure mid-skills-copy")

    monkeypatch.setattr("mazu.checkpoint.manager.shutil.copytree", _boom)

    with pytest.raises(OSError):
        manager.snapshot(messages=[], trigger="manual")

    assert manager.list_checkpoints() == []  # no half-written checkpoint indexed
    assert not (manager.checkpoints_dir / "cp_000001").exists()  # orphaned dir cleaned up


# ---------------------------------------------------------------------------
# Real multi-process race: the actual bug class this whole change targets
# ---------------------------------------------------------------------------

_SNAPSHOT_WORKER_SCRIPT = """
import sys
sys.path.insert(0, {repo_root!r})
from pathlib import Path
from mazu.checkpoint.manager import CheckpointManager

manager = CheckpointManager(Path({project!r}))
for i in range(10):
    manager.snapshot(messages=[{{"role": "user", "content": str(i)}}], trigger="manual")
print("DONE")
"""


def test_two_concurrent_processes_never_collide_on_checkpoint_ids(project: Path):
    """The actual scenario the critique raised: two real `mazu` processes (two
    terminal windows) checkpointing the same project at the same time. Before the
    lock, this could produce two checkpoints computing the same next_num from a
    stale read, colliding on disk (one silently overwriting the other's snapshot
    directory) and/or one process's index.append() clobbering the other's.

    Runs two REAL OS processes (not threads -- threads share the GIL and Python
    object state in ways that could mask a real cross-process race), each creating
    10 checkpoints, and asserts the final index has all 20, with unique ids/steps
    and no on-disk directory collisions.
    """
    repo_root = str(Path(__file__).resolve().parent.parent)
    script = _SNAPSHOT_WORKER_SCRIPT.format(repo_root=repo_root, project=str(project))

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]

    outputs = []
    for proc in procs:
        stdout, stderr = proc.communicate(timeout=120)
        outputs.append((proc.returncode, stdout, stderr))

    for returncode, stdout, stderr in outputs:
        assert returncode == 0, f"worker process failed:\\n{stderr}"
        assert "DONE" in stdout

    manager = CheckpointManager(project)
    entries = manager.list_checkpoints()

    assert len(entries) == 20, f"expected 20 checkpoints, got {len(entries)} -- lost writes"

    ids = [e["id"] for e in entries]
    steps = [e["step"] for e in entries]
    assert len(set(ids)) == 20, f"duplicate checkpoint ids: {ids}"
    assert len(set(steps)) == 20, f"duplicate step numbers: {steps}"

    # Every entry's own on-disk snapshot directory must exist and contain its own
    # conversation.json -- proving no two checkpoints wrote into the same directory
    # and clobbered each other's content.
    for entry in entries:
        conv_path = manager.checkpoints_dir / entry["id"] / "conversation.json"
        assert conv_path.exists(), f"missing snapshot dir/content for {entry['id']}"
