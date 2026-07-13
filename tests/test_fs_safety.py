import os
from pathlib import Path

import pytest

from mazu.tools.fs import make_fs_tools


@pytest.fixture
def tools(tmp_path: Path) -> dict:
    return {t.name: t for t in make_fs_tools(tmp_path)}


def test_read_write_roundtrip(tools, tmp_path):
    tools["write_file"].handler({"path": "a.py", "content": "print('hi')"})
    result = tools["read_file"].handler({"path": "a.py"})
    assert "print('hi')" in result.content
    assert not result.is_error


def test_write_file_creates_intermediate_dirs(tools, tmp_path):
    result = tools["write_file"].handler({"path": "sub/dir/a.py", "content": "x"})
    assert not result.is_error
    assert (tmp_path / "sub" / "dir" / "a.py").exists()


def test_read_file_path_traversal_blocked(tools):
    result = tools["read_file"].handler({"path": "../../etc/passwd"})
    assert result.is_error


def test_write_file_path_traversal_blocked(tools, tmp_path):
    result = tools["write_file"].handler({"path": "../outside.py", "content": "x"})
    assert result.is_error
    assert not (tmp_path.parent / "outside.py").exists()


def test_write_file_absolute_path_outside_root_blocked(tools, tmp_path):
    # An absolute path elsewhere on disk, not just a relative "../" traversal --
    # exercises the same _safe_path boundary check via a different input shape.
    outside_root = str(tmp_path.parent / "some_other_dir" / "evil.py")
    result = tools["write_file"].handler({"path": outside_root, "content": "x"})
    assert result.is_error


def test_edit_file_requires_unique_match(tools):
    tools["write_file"].handler({"path": "a.py", "content": "x = 1\nx = 1\n"})
    result = tools["edit_file"].handler({"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})
    assert result.is_error
    assert "matches 2 times" in result.content


def test_edit_file_replaces_unique_match(tools):
    tools["write_file"].handler({"path": "a.py", "content": "x = 1\ny = 2\n"})
    result = tools["edit_file"].handler({"path": "a.py", "old_str": "x = 1", "new_str": "x = 99"})
    assert not result.is_error
    read_back = tools["read_file"].handler({"path": "a.py"})
    assert "x = 99" in read_back.content


def test_glob_files_symlink_escape_blocked(tools, tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (outside / "secret.txt").write_text("secret")

    link = tmp_path / "escape_link"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted in this environment")

    result = tools["glob_files"].handler({"pattern": "escape_link/*"})
    assert "secret.txt" not in result.content


def test_list_dir_defaults_to_root(tools, tmp_path):
    tools["write_file"].handler({"path": "a.py", "content": "x"})
    result = tools["list_dir"].handler({})
    assert "a.py" in result.content


# ---------------------------------------------------------------------------
# dry-run mode (Phase E)
# ---------------------------------------------------------------------------


@pytest.fixture
def dry_run_tools(tmp_path: Path) -> dict:
    return {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}


def test_dry_run_write_file_does_not_touch_disk(dry_run_tools, tmp_path):
    result = dry_run_tools["write_file"].handler({"path": "a.py", "content": "print('hi')"})
    assert not result.is_error
    assert "[dry-run]" in result.content
    assert "Would write" in result.content
    assert not (tmp_path / "a.py").exists()


def test_dry_run_write_file_reports_correct_byte_count(dry_run_tools):
    result = dry_run_tools["write_file"].handler({"path": "a.py", "content": "hello"})
    assert "5 bytes" in result.content


def test_dry_run_edit_file_does_not_modify_the_real_file(tmp_path):
    # Seed a real file first using the non-dry-run tools, then switch to dry-run
    # tools pointed at the same directory to edit it.
    real_tools = {t.name: t for t in make_fs_tools(tmp_path)}
    real_tools["write_file"].handler({"path": "a.py", "content": "x = 1\n"})

    dry_tools = {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}
    result = dry_tools["edit_file"].handler({"path": "a.py", "old_str": "x = 1", "new_str": "x = 99"})

    assert not result.is_error
    assert "[dry-run]" in result.content
    assert "Would edit" in result.content
    assert real_tools["read_file"].handler({"path": "a.py"}).content.split("\t", 1)[1] == "x = 1"


def test_dry_run_edit_file_still_validates_old_str_uniqueness(tmp_path):
    real_tools = {t.name: t for t in make_fs_tools(tmp_path)}
    real_tools["write_file"].handler({"path": "a.py", "content": "x = 1\nx = 1\n"})

    dry_tools = {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}
    result = dry_tools["edit_file"].handler({"path": "a.py", "old_str": "x = 1", "new_str": "x = 2"})

    # A dry-run report must be trustworthy: a plan that would fail for real (an
    # ambiguous old_str match) must fail in the dry-run preview too, not silently
    # claim success.
    assert result.is_error
    assert "matches 2 times" in result.content


def test_dry_run_edit_file_reports_error_when_old_str_missing(tmp_path):
    real_tools = {t.name: t for t in make_fs_tools(tmp_path)}
    real_tools["write_file"].handler({"path": "a.py", "content": "x = 1\n"})

    dry_tools = {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}
    result = dry_tools["edit_file"].handler({"path": "a.py", "old_str": "not there", "new_str": "x"})
    assert result.is_error


def test_dry_run_read_file_still_reads_for_real(tmp_path):
    real_tools = {t.name: t for t in make_fs_tools(tmp_path)}
    real_tools["write_file"].handler({"path": "a.py", "content": "real content"})

    dry_tools = {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}
    result = dry_tools["read_file"].handler({"path": "a.py"})
    assert "real content" in result.content


def test_dry_run_list_dir_and_glob_still_work_for_real(tmp_path):
    real_tools = {t.name: t for t in make_fs_tools(tmp_path)}
    real_tools["write_file"].handler({"path": "a.py", "content": "x"})

    dry_tools = {t.name: t for t in make_fs_tools(tmp_path, dry_run=True)}
    assert "a.py" in dry_tools["list_dir"].handler({}).content
    assert "a.py" in dry_tools["glob_files"].handler({"pattern": "*.py"}).content
