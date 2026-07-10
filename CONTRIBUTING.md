# Contributing to Mazu

Thanks for considering a contribution. Mazu is a small, early-stage project — the codebase is intentionally kept simple (~3000 lines), so it's realistic to read most of it before your first change.

## Setup

```bash
git clone https://github.com/turgutino/Mazu.git
cd Mazu
pip install -e ".[dev,openai]"
```

`dev` pulls in `pytest`; `openai` pulls in the `openai` package needed to exercise the `openai:`/`deepseek:` provider code paths (they share one implementation, since DeepSeek's API is OpenAI-compatible).

## Running the tests

```bash
pytest
```

The whole suite runs with **zero API cost** — every model call is mocked. If you're adding behavior that talks to a provider (a new provider, a change to request/response handling), mock the SDK call the same way the existing tests do (see `tests/test_provider_routing.py` and `tests/test_streaming.py` for the established patterns) rather than requiring a real API key to run CI.

CI (`.github/workflows/ci.yml`) runs the suite across Python 3.11–3.13 on Linux, Windows, and macOS on every push and pull request.

## Before opening a PR

- **Add or update tests for anything you change.** This project only recently got a test suite at all (see the commit history) after several real bugs shipped that a test would have caught — please don't let that regress. A PR that changes behavior without a corresponding test is unlikely to be merged as-is.
- **Run `pytest` locally and make sure it's green** before pushing.
- Keep changes focused — a PR that fixes one bug or adds one feature is much easier to review than one that bundles several unrelated things.
- If you're touching a provider (`mazu/llm/providers/`), be precise about what's actually verified vs. assumed — this project has been burned before by claiming something works based on reading the SDK docs alone rather than checking the SDK's actual shapes/behavior. Where possible, verify against the installed SDK's real types (see how `tests/test_streaming.py` checks its fakes against the real `openai`/`anthropic` package shapes) rather than guessing.

## Reporting bugs

Open a GitHub issue with: what you ran, what you expected, what actually happened, and your OS/Python version/provider. If it's a security issue, see [SECURITY.md](SECURITY.md) instead of opening a public issue.

## Project structure

See the [README](README.md#how-it-fits-together) for a map of the codebase before diving in.

## Scope

Mazu is deliberately minimal — it favors a small, readable core over broad feature coverage. If you're proposing something substantial (a new subsystem, a new dependency, a new provider), it's worth opening an issue to discuss the approach before investing time in a full implementation.
