"""Tests for mazu/cli.py's new Faza 4 additions: `mazu --version` and
`mazu checkpoint list`. Uses Click's CliRunner against an isolated filesystem, so
these exercise the real command wiring, not just the underlying functions.
"""

import subprocess
from importlib.metadata import version as installed_version

import pytest
from click.testing import CliRunner

import mazu
from mazu.cli import main


@pytest.fixture(autouse=True)
def _git_identity(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    subprocess.run(["git", "config", "--global", "user.email", "test@example.com"])
    subprocess.run(["git", "config", "--global", "user.name", "Test"])


def test_version_flag_reports_the_real_installed_version():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])

    assert result.exit_code == 0
    assert "mazu" in result.output.lower()
    assert installed_version("mazu") in result.output


def test_mazu_dunder_version_matches_installed_package():
    # Regression test for a real bug: mazu/__init__.py used to hardcode
    # __version__ = "0.1.0" as a plain string that nothing ever kept in sync with
    # pyproject.toml's actual version, so it silently drifted out of date on every
    # release after the first. It must now be derived from package metadata.
    assert mazu.__version__ == installed_version("mazu")


def test_checkpoint_list_empty_project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(main, ["checkpoint", "list"])

    assert result.exit_code == 0
    assert "No checkpoints yet." in result.output


def test_checkpoint_list_shows_created_checkpoints(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()

    result = runner.invoke(main, ["checkpoint"])
    assert result.exit_code == 0, result.output

    result = runner.invoke(main, ["checkpoint", "list"])
    assert result.exit_code == 0, result.output
    assert "cp_000001" in result.output
