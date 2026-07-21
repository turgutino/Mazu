from pathlib import Path

from mazu.skills.manager import SkillManager


def test_run_handles_non_ascii_output_without_crashing(tmp_path: Path):
    """Regression test for a real bug found via live testing: a skill script that
    prints non-ASCII text (observed with a Turkish/Azerbaijani letter) used to crash
    the skill's own subprocess on Windows with UnicodeEncodeError, since its stdout
    defaulted to the console's legacy codepage rather than UTF-8. PYTHONIOENCODING is
    now set for every skill subprocess to prevent this at the source.
    """
    manager = SkillManager(tmp_path)
    code = (
        "def run(args):\n"
        f"    print(chr({0x1F600}))\n"
        f"    print(chr({305}))\n"
        "    return 'ok'\n"
    )
    manager.save("non_ascii_skill", "prints non-ASCII text", code)

    output, is_error = manager.run("non_ascii_skill", {})

    assert not is_error
    assert chr(0x1F600) in output
    assert chr(305) in output
