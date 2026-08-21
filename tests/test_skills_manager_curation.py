from pathlib import Path

import pytest

from mazu.skills.manager import SkillManager

CODE_A = "def run(args):\n    return 'a'\n"
CODE_B = "def run(args):\n    return 'b'\n"


def test_save_rejects_code_that_already_includes_the_main_guard(tmp_path: Path):
    """Regression test for a real bug found via live testing: a caller (a human,
    or an LLM tool call like Curator's write_skill) that pastes a complete
    standalone script -- including its own `if __name__ == "__main__":` guard --
    used to get it silently wrapped a SECOND time by SKILL_TEMPLATE, producing a
    file where the guard appears twice. Both copies run when the file executes:
    the first correctly reads stdin and returns the right answer, but the second
    then re-reads (now-empty) stdin, gets `{}`, and crashes -- so a skill with a
    100% correct logical fix was still broken on every real invocation. This must
    now fail loudly and immediately instead of silently shipping that.
    """
    manager = SkillManager(tmp_path)
    full_script = (
        "import json\nimport sys\n\ndef run(args):\n    return str(args['n'] * 2)\n\n"
        'if __name__ == "__main__":\n    raw = sys.stdin.read()\n'
        "    args = json.loads(raw) if raw.strip() else {}\n    print(run(args))\n"
    )
    with pytest.raises(ValueError, match="ONLY the"):
        manager.save("s1", "desc", full_script)


def test_save_accepts_plain_function_body(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)  # must not raise
    assert manager.exists("s1")


def test_save_on_a_new_skill_starts_at_zero_usage(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)
    meta = manager.get_meta("s1")
    assert meta["usage_count"] == 0
    assert meta["archived"] is False


def test_resaving_an_existing_skill_preserves_usage_count_and_created_at(tmp_path: Path):
    """Regression test: save() used to reset usage_count/created_at to 0/now on
    every call, which would destroy a skill's own evidence the moment its code was
    rewritten to fix a bug -- exactly what Curator needs to do."""
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)
    manager.run("s1", {})
    manager.run("s1", {})
    before = manager.get_meta("s1")
    assert before["usage_count"] == 2

    manager.save("s1", "improved desc", CODE_B)

    after = manager.get_meta("s1")
    assert after["usage_count"] == 2
    assert after["created_at"] == before["created_at"]
    assert after["description"] == "improved desc"
    assert "return 'b'" in manager.read_code("s1")


def test_archive_excludes_from_list_but_keeps_the_file_on_disk(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)

    ok = manager.archive("s1", reason="failing too often")
    assert ok is True

    assert manager.list() == []
    assert manager.exists("s1") is True
    assert manager.read_code("s1") is not None
    meta = manager.get_meta("s1")
    assert meta["archived"] is True
    assert meta["archived_reason"] == "failing too often"


def test_list_include_archived_shows_it(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)
    manager.archive("s1")
    names = {m["name"] for m in manager.list(include_archived=True)}
    assert names == {"s1"}


def test_unarchive_restores_to_list(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)
    manager.archive("s1")
    ok = manager.unarchive("s1")
    assert ok is True
    names = {m["name"] for m in manager.list()}
    assert names == {"s1"}


def test_archive_missing_skill_returns_false(tmp_path: Path):
    manager = SkillManager(tmp_path)
    assert manager.archive("nope") is False
    assert manager.unarchive("nope") is False


def test_supersede_archives_old_and_records_replacement(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("old_skill", "desc", CODE_A)
    manager.save("new_skill", "better desc", CODE_B)

    ok = manager.supersede("old_skill", "new_skill")
    assert ok is True

    names = {m["name"] for m in manager.list()}
    assert names == {"new_skill"}
    old_meta = manager.get_meta("old_skill")
    assert old_meta["archived"] is True
    assert old_meta["superseded_by"] == "new_skill"


def test_supersede_requires_both_skills_to_exist(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("old_skill", "desc", CODE_A)
    assert manager.supersede("old_skill", "does_not_exist") is False
    assert manager.supersede("does_not_exist", "old_skill") is False


def test_update_meta_merges_fields(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("s1", "desc", CODE_A)
    ok = manager.update_meta("s1", success_count=3, failure_count=1, last_outcome="ok")
    assert ok is True
    meta = manager.get_meta("s1")
    assert meta["success_count"] == 3
    assert meta["failure_count"] == 1
    assert meta["last_outcome"] == "ok"
    assert meta["description"] == "desc"  # untouched fields survive


def test_build_context_block_excludes_archived(tmp_path: Path):
    manager = SkillManager(tmp_path)
    manager.save("active_one", "desc", CODE_A)
    manager.save("archived_one", "desc", CODE_B)
    manager.archive("archived_one")

    block = manager.build_context_block()
    assert "active_one" in block
    assert "archived_one" not in block
