# Demo: a mistake, found and undone

This walks through the scenario Mazu's checkpoint/memory system exists for: an agent makes a change that looks fine at the time, breaks something, and you find and undo exactly that change — without losing the good work that came before or after it, and without the same mistake happening again next time.

**Every command and every line of output below is real** — run against an actual project, actual git history, and actual SQLite stores, not written from memory or invented for illustration. The one thing that *is* scripted (not live): the model's own responses, fed in as fixed, deterministic text instead of a real API call, so this walkthrough reproduces identically every time instead of depending on what a live model happens to say on a given day. Every Mazu mechanic driven by those responses — the git commits, the checkpoints, the SQLite writes, the test failures and passes — is genuinely real, not mocked.

## The setup

A two-file project: `calc.py` with a `divide()` function that deliberately converts `ZeroDivisionError` into a clearer `ValueError`, and `test_calc.py` covering both the normal case and that error-message behavior.

```python
# calc.py
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")
```

```python
# test_calc.py
def test_divide():
    assert divide(10, 2) == 5

def test_divide_by_zero_raises_clean_error():
    with pytest.raises(ValueError, match="Cannot divide by zero"):
        divide(10, 0)
```

`mazu init`, commit, baseline tests pass.

## Session 1: a good change

Task: *"Add a multiply(a, b) function to calc.py, and a test for it."*

```
model: anthropic:claude-sonnet-5
run — task: Add a multiply(a, b) function to calc.py, and a test for it.
max-steps=5 checkpoint-every=1 allow-shell=False

[usage] step: 0 in, 0 out | running total: 0 in, 0 out | ~$0.0000 so far
[checkpoint cp_000001 @ c0f5e5e3]
[usage] step: 0 in, 0 out | running total: 0 in, 0 out | ~$0.0000 so far
[checkpoint cp_000002 @ e419500d]
[usage] step: 0 in, 0 out | running total: 0 in, 0 out | ~$0.0000 so far
[step 3] Added a multiply() function to calc.py, with a test for it.

Task complete (model signaled end_turn).

=== Run report ===
Session: demo-session-1
Stop reason: end_turn
Steps: 3/5
Files changed: calc.py, test_calc.py
Checkpoints created: 2
Memories saved: 0
Tool errors: 0
```

```
$ pytest -q
...                                                                      [100%]
3 passed in 0.05s
```

Two checkpoints (`cp_000001`, `cp_000002`), one per tool-use round, both clean.

## Session 2: the mistake

Task: *"Simplify calc.py — remove any code that looks unnecessary."*

The model decides the `try`/`except` in `divide()` is redundant — Python already raises `ZeroDivisionError` on its own — and removes it. Reasonable-sounding, and wrong: `test_divide_by_zero_raises_clean_error` depends on the specific `ValueError` and message, not the raw exception.

```
model: anthropic:claude-sonnet-5
run — task: Simplify calc.py -- remove any code that looks unnecessary.
max-steps=5 checkpoint-every=1 allow-shell=False

[usage] step: 0 in, 0 out | running total: 0 in, 0 out | ~$0.0000 so far
[checkpoint cp_000003 @ 74eccf22]
[usage] step: 0 in, 0 out | running total: 0 in, 0 out | ~$0.0000 so far
[step 2] Simplified divide() by removing the try/except block -- Python already
raises ZeroDivisionError on its own, so the wrapper seemed redundant.

Task complete (model signaled end_turn).

=== Run report ===
Session: demo-session-2
Stop reason: end_turn
Steps: 2/5
Files changed: calc.py
Checkpoints created: 1
Memories saved: 0
Tool errors: 0
```

```
$ pytest -q
.F.                                                                      [100%]
================================== FAILURES ===================================
___________________ test_divide_by_zero_raises_clean_error ____________________

    def test_divide_by_zero_raises_clean_error():
        with pytest.raises(ValueError, match="Cannot divide by zero"):
>           divide(10, 0)

a = 10, b = 0

    def divide(a, b):
>       return a / b
               ^^^^^
E       ZeroDivisionError: division by zero

calc.py:2: ZeroDivisionError
=========================== short test summary info ===========================
FAILED test_calc.py::test_divide_by_zero_raises_clean_error - ZeroDivisionErr...
1 failed, 2 passed in 0.31s
```

A real, reproducible failure — not a narrated one.

## Finding it: `mazu timeline`

```
$ mazu timeline
cp_000001  2026-07-14T09:44:30.040617+00:00  (auto_after_tool_round) step 1
    commit: c0f5e5e3
    files:  (first checkpoint — nothing to compare against)
    snapshot: memory, no skills

cp_000002  2026-07-14T09:44:30.268345+00:00  (auto_after_tool_round) step 2
    commit: e419500d
    files:  test_calc.py
    snapshot: memory, no skills

cp_000003  2026-07-14T09:45:08.935267+00:00  (auto_after_tool_round) step 1
    commit: 74eccf22
    files:  calc.py
    snapshot: memory, no skills
```

`cp_000003` stands out immediately: it's the only checkpoint touching `calc.py` again after session 1 already finished it, from a task that was explicitly about "removing" things. That's the one to look at.

## Seeing the damage: checkpoint diff

```
$ mazu checkpoint compare cp_000002 cp_000003
Diff from cp_000002 (2026-07-14T09:44:30.268345+00:00) to cp_000003 (2026-07-14T09:45:08.935267+00:00):

 calc.py | 5 +----
 1 file changed, 1 insertion(+), 4 deletions(-)
```

`mazu checkpoint diff`/`compare` currently report a `--stat` summary (files touched, lines added/removed) rather than full line-by-line content — enough to confirm *which* file and roughly how much changed, not enough to read the actual removed code in the terminal. For that, the underlying git commits (shown by `mazu checkpoint show <id>`) are real commits — a plain `git diff <commit_a> <commit_b>` reads the full patch:

```diff
$ git diff e419500d 74eccf22 -- calc.py
diff --git a/calc.py b/calc.py
index 0ca2ee1..1fe090b 100644
--- a/calc.py
+++ b/calc.py
@@ -1,8 +1,5 @@
 def divide(a, b):
-    try:
-        return a / b
-    except ZeroDivisionError:
-        raise ValueError("Cannot divide by zero")
+    return a / b
 
 
 def multiply(a, b):
```

Now the mistake is fully legible: the error-translation wrapper is gone, not just "something in calc.py changed."

You can also see exactly which tool call did it, with its full input, via the action log:

```
$ mazu log show demo-session-2
[2026-07-14T09:45:08.708109+00:00] write_file — ok
    input: {"path": "calc.py", "content": "def divide(a, b):\n    return a / b\n\n\ndef multiply(a, b):\n    return a * b\n"}
    output: Wrote 74 bytes to calc.py
    changed: calc.py
```

## Undoing it: `mazu rollback`

```
$ mazu rollback cp_000002
Rolling back to cp_000002 (2026-07-14T09:44:30.268345+00:00): step 2
 calc.py | 5 +----
 1 file changed, 1 insertion(+), 4 deletions(-)

This discards code and memory changes since then. Proceed? [y/N]: y
Restored to cp_000002. (Conversation state only restores inside a live `mazu chat` session.)
```

```
$ cat calc.py
def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        raise ValueError("Cannot divide by zero")


def multiply(a, b):
    return a * b
```

```
$ pytest -q
...                                                                      [100%]
3 passed in 0.02s
```

Back to session 1's known-good state — the `multiply()` work is intact, only the mistake is gone. (If you wanted to keep exploring the "simplified" direction without losing this good state, `mazu branch-from cp_000003 simplify-attempt` would branch from the mistake instead of discarding it — rollback is for "get me back to good," branch-from is for "let me poke at that dead end a bit more, separately.")

## Making sure it isn't repeated: memory

Telling the agent about the mistake explicitly:

```
> Remember: do not remove the try/except in divide() -- it is intentional, a test depends on it.
```

```
$ mazu memory list
[   1]  (mistake, explicit) Do not remove the try/except in divide()
       divide() in calc.py wraps ZeroDivisionError and re-raises it as ValueError("Cannot divide by zero") on purpose -- test_calc.py::test_divide_by_zero_raises_clean_error depends on that exact type and message. A prior session removed it as "unnecessary" and broke that test (see cp_000003, rolled back).
       tags: calc.py,divide,error-handling
```

The important part isn't that it's saved — it's that `mistake`-category memories are always surfaced, regardless of whether the current task's wording happens to overlap with the memory's own wording:

```
$ mazu memory why "simplify calc.py, remove anything unnecessary"
[x] [   1] (mistake) Do not remove the try/except in divide() — recent mistake
```

`[x]` means included; `recent mistake` (not a relevance score) means it didn't need to rank well against the query to get there — the three most recent `mistake`-category memories are always injected into context, on top of whatever query-relevance ranking surfaces. A future session asking to "simplify" or "clean up" this file sees the warning before it can repeat the mistake, not after.

## What this demonstrates

- **`mazu timeline`** turns "something broke, when?" into a short, chronological list instead of a raw git log.
- **`mazu checkpoint diff`/`compare`** narrow "which checkpoint" down to file-level confidence quickly; full line-level detail is one `git diff <commit_a> <commit_b>` away using the real commit hashes checkpoints already are.
- **`mazu log show`** pinpoints the exact tool call, with its exact input, when a file-level diff alone isn't enough context.
- **`mazu rollback`** undoes precisely the bad checkpoint's state — code and memory together, so what the agent remembers never drifts out of sync with what's actually on disk.
- **`mazu memory`** (the `mistake` category specifically) is what turns "we fixed it this time" into "it won't happen again" — a floor that doesn't depend on the next task happening to phrase things the same way the fix did.
