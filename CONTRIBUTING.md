# Contributing to Mazu

Thanks for considering a contribution. Mazu is a small, early-stage project maintained by one person. Reading [ARCHITECTURE.md](ARCHITECTURE.md) first gives you a map of the code and of the places where its guarantees are weakest.

## Setup

```bash
git clone https://github.com/turgutino/Mazu.git
cd Mazu
pip install -e ".[dev,openai,gemini,ui,documents]"
```

`dev` pulls in `pytest`. `openai` is needed to exercise the `openai:`, `deepseek:`, and `local:` provider paths (they share one implementation, since DeepSeek's API is OpenAI-compatible). `gemini`, `ui`, and `documents` are the other optional extras; CI installs all of them.

## Running the tests

```bash
pytest
```

The suite (about 1,100 tests) runs with **zero API cost**: every model call is mocked. If you add behavior that talks to a provider, mock the SDK call the way the existing tests do (see `tests/test_provider_routing.py` and `tests/test_streaming.py`) instead of requiring a real key.

CI (`.github/workflows/ci.yml`) runs the suite on Linux, Windows, and macOS across Python 3.11–3.13 for every push and pull request.

**Do not point tests at your real home directory.** Tests that run `git config --global` or write under `~/.mazu` must redirect `HOME` and `USERPROFILE` to a temporary directory first, as the existing fixtures do.

## Before opening a PR

- **Add or update tests for anything you change.** A PR that changes behavior without a corresponding test is unlikely to be merged as-is.
- **Run `pytest` locally** and make sure it is green before pushing.
- **Keep changes focused.** One bug fix or one feature per PR.
- **Be precise about what is verified.** If you touch a provider (`mazu/llm/providers/`), say what you checked against the installed SDK's real types and what you assumed from documentation; the project has been wrong before by trusting docs alone.
- **Back claims about the agent with a measurement.** If you change how the agent behaves (prompts, loop logic, memory, context handling) and say it improves results or cost, show it with `mazu bench`: same suite, same model, several repeats, and report the interval. Single before/after runs are noise at this scale; the report tells you when intervals overlap.
- **Update the docs you affect:** the README's command tables, `CHANGELOG.md` (an entry under the unreleased heading), and `ARCHITECTURE.md` if you add a module or change where something is gated.

## Measuring a change

```bash
python bench/mazu_history/build_suite.py --out ./bench-work/mazu-history --check   # free, no model calls
mazu bench run --suite bench-work/mazu-history/suite.json --configs configs.json --repeats 3 --out results.jsonl
mazu bench report results.jsonl
```

`mazu bench run` makes real, billed model calls (the README has a configs example). Keep result files and `bench-work/` out of commits; both are gitignored.

## Reporting bugs

Open a GitHub issue with what you ran, what you expected, what happened, and your OS, Python version, Mazu version, and provider. `mazu doctor` output helps. For security issues, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Scope

Mazu favors a readable core over broad feature coverage. For something substantial (a new subsystem, dependency, or provider), open an issue to discuss the approach before investing in an implementation. [ROADMAP.md](ROADMAP.md) lists what is being worked on and the known gaps, which are good places to start.
