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
