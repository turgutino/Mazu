"""Tests for mazu/runs/detect_test_command.py -- pre-fills Explore's optional
"Test command" field. Real filesystem fixtures (tmp_path), no mocking: this
module only does file-system reads, so the real thing is exactly as fast and
simple as a fake.
"""

import platform

from mazu.runs.detect_test_command import detect_test_command

_IS_WINDOWS = platform.system() == "Windows"


def test_no_markers_returns_none(tmp_path):
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_pytest_ini_detected_at_root(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == "python -m pytest -q"


def test_tests_dir_with_test_files_detected(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_foo.py").write_text("def test_x(): pass\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == "python -m pytest -q"


def test_pyproject_toml_with_pytest_section_detected(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[tool.pytest.ini_options]\naddopts = '-q'\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == "python -m pytest -q"


def test_pyproject_toml_without_pytest_section_not_detected(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_package_json_with_test_script_detected(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"test": "jest"}}', encoding="utf-8")
    assert detect_test_command(tmp_path) == "npm test"


def test_package_json_without_test_script_not_detected(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"build": "webpack"}}', encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_malformed_package_json_does_not_crash(tmp_path):
    (tmp_path / "package.json").write_text("{not valid json", encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_go_mod_detected(tmp_path):
    (tmp_path / "go.mod").write_text("module example.com/foo\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == "go test ./..."


def test_cargo_toml_detected(tmp_path):
    (tmp_path / "Cargo.toml").write_text("[package]\nname = 'foo'\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == "cargo test"


def test_pom_xml_detected(tmp_path):
    (tmp_path / "pom.xml").write_text("<project></project>", encoding="utf-8")
    assert detect_test_command(tmp_path) == "mvn test"


def test_root_marker_wins_over_nested_marker(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "go.mod").write_text("module x\n", encoding="utf-8")
    assert detect_test_command(tmp_path) == "python -m pytest -q"


def test_one_level_nested_marker_gets_cd_prefix(tmp_path):
    sub = tmp_path / "backend"
    sub.mkdir()
    (sub / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")

    result = detect_test_command(tmp_path)
    assert result == "cd backend && python -m pytest -q"


def test_deeply_nested_marker_beyond_max_depth_not_found(tmp_path):
    # 3 levels down -- outside the 2-level search depth.
    deep = tmp_path / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_ignored_directories_are_never_descended_into(tmp_path):
    noise = tmp_path / "node_modules" / "some-package"
    noise.mkdir(parents=True)
    (noise / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    assert detect_test_command(tmp_path) is None


def test_venv_python_used_when_present(tmp_path):
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    bin_dir = tmp_path / "venv" / ("Scripts" if _IS_WINDOWS else "bin")
    bin_dir.mkdir(parents=True)
    exe = bin_dir / ("python.exe" if _IS_WINDOWS else "python")
    exe.write_text("", encoding="utf-8")

    result = detect_test_command(tmp_path)
    expected_exe = "venv\\Scripts\\python.exe" if _IS_WINDOWS else "venv/bin/python"
    assert result == f"{expected_exe} -m pytest -q"


def test_real_project_shape_backend_subdir_with_venv(tmp_path):
    # Mirrors the exact shape found live in this session's own testing: a
    # Flask app one level down in a "magaza"-style subdirectory, with its own
    # venv and tests/ folder -- not pytest.ini/pyproject.toml at all.
    app_dir = tmp_path / "magaza"
    tests_dir = app_dir / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test_magaza.py").write_text("def test_x(): pass\n", encoding="utf-8")
    bin_dir = app_dir / "venv" / ("Scripts" if _IS_WINDOWS else "bin")
    bin_dir.mkdir(parents=True)
    (bin_dir / ("python.exe" if _IS_WINDOWS else "python")).write_text("", encoding="utf-8")

    result = detect_test_command(tmp_path)
    if _IS_WINDOWS:
        assert result == "cd magaza && venv\\Scripts\\python.exe -m pytest -q"
    else:
        assert result == "cd magaza && venv/bin/python -m pytest -q"
