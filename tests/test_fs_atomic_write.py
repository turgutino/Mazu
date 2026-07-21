"""Tests for Addendum 6's crash-safe (atomic) writes: write_file/edit_file must never
leave a truncated or partially-written file at the real path if something goes wrong
mid-write, and must never leave a stray temp file behind on success.
"""

import os
from pathlib import Path

import pytest

from mazu.tools.fs import _atomic_write_text, make_fs_tools


@pytest.fixture
def tools(tmp_path: Path) -> dict:
    return {t.name: t for t in make_fs_tools(tmp_path)}


def test_atomic_write_creates_correct_content(tmp_path: Path):
    target = tmp_path / "a.py"
    _atomic_write_text(target, "print('hi')")
    assert target.read_text(encoding="utf-8") == "print('hi')"


def test_atomic_write_leaves_no_stray_temp_file(tmp_path: Path):
    target = tmp_path / "a.py"
    _atomic_write_text(target, "content")
    assert os.listdir(tmp_path) == ["a.py"]


def test_atomic_write_overwrite_leaves_no_stray_temp_file(tmp_path: Path):
    target = tmp_path / "a.py"
    _atomic_write_text(target, "first")
    _atomic_write_text(target, "second")
    assert target.read_text(encoding="utf-8") == "second"
    assert os.listdir(tmp_path) == ["a.py"]


def test_atomic_write_failure_leaves_original_file_untouched(tmp_path: Path, monkeypatch):
    """Simulates a crash mid-write (the fsync/write step raises) -- the pre-existing
    file at the real path must be completely unaffected, and no temp file left behind.
    """
    target = tmp_path / "a.py"
    target.write_text("original content", encoding="utf-8")

    import mazu.tools.fs as fs_module

    def _boom(*args, **kwargs):
        raise OSError("simulated crash mid-write")

    monkeypatch.setattr(fs_module.os, "fsync", _boom)

    with pytest.raises(OSError):
        _atomic_write_text(target, "new content that should never land")

    assert target.read_text(encoding="utf-8") == "original content"
    # No orphaned temp file left in the directory.
    assert os.listdir(tmp_path) == ["a.py"]


def test_atomic_write_failure_when_file_never_existed_leaves_nothing_behind(tmp_path: Path, monkeypatch):
    target = tmp_path / "new.py"

    import mazu.tools.fs as fs_module

    monkeypatch.setattr(fs_module.os, "fsync", lambda *a, **k: (_ for _ in ()).throw(OSError("boom")))

    with pytest.raises(OSError):
        _atomic_write_text(target, "content")

    assert not target.exists()
    assert os.listdir(tmp_path) == []


def test_write_file_tool_uses_atomic_write(tools, tmp_path: Path):
    result = tools["write_file"].handler({"path": "a.py", "content": "print('hi')"})
    assert not result.is_error
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "print('hi')"
    assert os.listdir(tmp_path) == ["a.py"]


def test_edit_file_tool_uses_atomic_write(tools, tmp_path: Path):
    tools["write_file"].handler({"path": "a.py", "content": "print('old')"})
    result = tools["edit_file"].handler({"path": "a.py", "old_str": "old", "new_str": "new"})
    assert not result.is_error
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "print('new')"
    assert os.listdir(tmp_path) == ["a.py"]


# ---------------------------------------------------------------------------
# Regression: existing fs.py behavior (dry-run, path-escape checks) is unchanged
# ---------------------------------------------------------------------------


def test_dry_run_write_file_still_never_touches_disk(tmp_path: Path):
    dry_tools = {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}
    result = dry_tools["write_file"].handler({"path": "a.py", "content": "x"})
    assert not result.is_error
    assert not (tmp_path / "a.py").exists()
