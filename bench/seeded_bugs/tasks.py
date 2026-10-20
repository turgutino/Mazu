"""Eight small, self-contained bug-fix / feature tasks for `mazu bench`.

Each task = a buggy mini project (files), the agent-facing prompt (symptom only, never
the fix), a HIDDEN test the agent never sees (kept outside the repo and run after the
agent exits), and a reference fix so build_suite.py --check can prove the task is
solvable and that the untouched repo really fails. Standard library only.
"""

TASKS = {}


def task(id, prompt, files, hidden_test, fixed):
    TASKS[id] = dict(id=id, prompt=prompt, files=files, hidden_test=hidden_test, fixed=fixed)


# 1 ---------------------------------------------------------------- pagination
task(
    "paginate-off-by-one",
    "Users report that the last page of results from `paginate` in pager.py is missing "
    "items, and that asking for a page past the end should return an empty list. "
    "Fix pager.py. Run the existing tests to see current behaviour.",
    {
        "pager.py": '''def paginate(items, page, per_page):
    """Return the items on 1-based `page`."""
    start = page * per_page
    end = start + per_page - 1
    return list(items)[start:end]


def page_count(total, per_page):
    return total // per_page
''',
        "test_pager.py": '''import unittest
from pager import paginate


class T(unittest.TestCase):
    def test_first_page(self):
        self.assertEqual(paginate(list(range(10)), 1, 3), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
''',
    },
    '''import unittest
from pager import paginate, page_count


class H(unittest.TestCase):
    def test_pages(self):
        d = list(range(10))
        self.assertEqual(paginate(d, 1, 3), [0, 1, 2])
        self.assertEqual(paginate(d, 2, 3), [3, 4, 5])
        self.assertEqual(paginate(d, 4, 3), [9])
        self.assertEqual(paginate(d, 5, 3), [])

    def test_count(self):
        self.assertEqual(page_count(10, 3), 4)
        self.assertEqual(page_count(9, 3), 3)
        self.assertEqual(page_count(0, 3), 0)
''',
    {"pager.py": '''def paginate(items, page, per_page):
    """Return the items on 1-based `page`."""
    start = (page - 1) * per_page
    return list(items)[start:start + per_page]


def page_count(total, per_page):
    return -(-total // per_page)
'''},
)

# 2 ---------------------------------------------------------------- range overlap
task(
    "range-overlap-inclusive",
    "`overlaps` in ranges.py should say whether two inclusive integer ranges (start, end) "
    "share at least one value, but ranges that only touch at an endpoint, and ranges "
    "where one contains the other, give wrong answers. Fix it.",
    {
        "ranges.py": '''def overlaps(a, b):
    """a and b are (start, end) tuples, both ends inclusive."""
    return a[0] < b[1] and b[0] < a[1]


def merge(ranges):
    """Merge overlapping inclusive ranges into a sorted list."""
    out = []
    for r in sorted(ranges):
        if out and overlaps(out[-1], r):
            out[-1] = (out[-1][0], max(out[-1][1], r[1]))
        else:
            out.append(r)
    return out
''',
    },
    '''import unittest
from ranges import overlaps, merge


class H(unittest.TestCase):
    def test_overlaps(self):
        self.assertTrue(overlaps((1, 5), (5, 9)))
        self.assertTrue(overlaps((1, 10), (3, 4)))
        self.assertTrue(overlaps((3, 4), (1, 10)))
        self.assertTrue(overlaps((2, 2), (2, 2)))
        self.assertFalse(overlaps((1, 4), (5, 9)))
        self.assertFalse(overlaps((5, 9), (1, 4)))

    def test_merge(self):
        self.assertEqual(merge([(5, 9), (1, 5), (12, 14)]), [(1, 9), (12, 14)])
        self.assertEqual(merge([(1, 10), (2, 3)]), [(1, 10)])
        self.assertEqual(merge([]), [])
''',
    {"ranges.py": '''def overlaps(a, b):
    """a and b are (start, end) tuples, both ends inclusive."""
    return a[0] <= b[1] and b[0] <= a[1]


def merge(ranges):
    """Merge overlapping inclusive ranges into a sorted list."""
    out = []
    for r in sorted(ranges):
        if out and overlaps(out[-1], r):
            out[-1] = (out[-1][0], max(out[-1][1], r[1]))
        else:
            out.append(r)
    return out
'''},
)

# 3 ---------------------------------------------------------------- word counts
task(
    "wordfreq-tokenizer",
    "`word_counts` in wordfreq.py miscounts: 'Don't' and 'don't' should be the same word "
    "(one token, keeping the apostrophe), and punctuation like commas or periods must "
    "not stick to words. Fix the tokenizer so `top_words` returns the right ranking.",
    {
        "wordfreq.py": '''def tokenize(text):
    return text.split(" ")


def word_counts(text):
    counts = {}
    for w in tokenize(text):
        if w:
            counts[w] = counts.get(w, 0) + 1
    return counts


def top_words(text, n):
    items = sorted(word_counts(text).items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:n]
''',
    },
    '''import unittest
from wordfreq import word_counts, top_words


class H(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(word_counts("Don't stop. don't STOP, stop!"),
                         {"don't": 2, "stop": 3})

    def test_top(self):
        text = "the cat and the dog. The end, and fin"
        self.assertEqual(top_words(text, 2), [("the", 3), ("and", 2)])

    def test_empty(self):
        self.assertEqual(word_counts(""), {})
        self.assertEqual(word_counts("  ...  "), {})
''',
    {"wordfreq.py": '''import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*", text.lower())


def word_counts(text):
    counts = {}
    for w in tokenize(text):
        counts[w] = counts.get(w, 0) + 1
    return counts


def top_words(text, n):
    items = sorted(word_counts(text).items(), key=lambda kv: (-kv[1], kv[0]))
    return items[:n]
'''},
)

# 4 ---------------------------------------------------------------- LRU cache
task(
    "lru-cache-recency",
    "The LRUCache in lru.py evicts the wrong entry: reading a key with `get` is supposed "
    "to make it the most recently used, and overwriting an existing key must not grow "
    "the cache past its capacity. Fix lru.py.",
    {
        "lru.py": '''from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def put(self, key, value):
        self._data[key] = value
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data
''',
    },
    '''import unittest
from lru import LRUCache


class H(unittest.TestCase):
    def test_get_refreshes(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(c.get("a"), 1)
        c.put("c", 3)
        self.assertIn("a", c)
        self.assertNotIn("b", c)
        self.assertIn("c", c)

    def test_overwrite_refreshes_and_keeps_size(self):
        c = LRUCache(2)
        c.put("a", 1)
        c.put("b", 2)
        c.put("a", 10)
        self.assertEqual(len(c), 2)
        c.put("c", 3)
        self.assertEqual(c.get("a"), 10)
        self.assertNotIn("b", c)

    def test_missing(self):
        c = LRUCache(1)
        self.assertEqual(c.get("x", "d"), "d")
        c.put("a", 1)
        c.put("b", 2)
        self.assertEqual(len(c), 1)
        self.assertNotIn("a", c)
''',
    {"lru.py": '''from collections import OrderedDict


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self._data = OrderedDict()

    def get(self, key, default=None):
        if key not in self._data:
            return default
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data
'''},
)

# 5 ---------------------------------------------------------------- csv parser
task(
    "csv-quoted-fields",
    "`parse_line` in minicsv.py splits on every comma, so quoted fields that contain "
    "commas are broken apart, and a doubled quote inside a quoted field (\"\") should "
    "become a single quote character. Fix the parser; `parse` must keep working.",
    {
        "minicsv.py": '''def parse_line(line):
    return [f.strip() for f in line.split(",")]


def parse(text):
    return [parse_line(l) for l in text.splitlines() if l.strip()]
''',
    },
    '''import unittest
from minicsv import parse_line, parse


class H(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_line("a, b ,c"), ["a", "b", "c"])

    def test_quoted_comma(self):
        self.assertEqual(parse_line('x,"y,z",w'), ["x", "y,z", "w"])

    def test_escaped_quote(self):
        self.assertEqual(parse_line('"say ""hi""",2'), ['say "hi"', "2"])

    def test_empty_fields(self):
        self.assertEqual(parse_line("a,,c"), ["a", "", "c"])
        self.assertEqual(parse_line(""), [""])

    def test_parse(self):
        self.assertEqual(parse('a,"b,c"\\n\\n1,2\\n'), [["a", "b,c"], ["1", "2"]])
''',
    {"minicsv.py": '''def parse_line(line):
    fields, cur, in_q, i = [], [], False, 0
    while i < len(line):
        ch = line[i]
        if in_q:
            if ch == '"':
                if i + 1 < len(line) and line[i + 1] == '"':
                    cur.append('"')
                    i += 1
                else:
                    in_q = False
            else:
                cur.append(ch)
        elif ch == '"':
            in_q = True
        elif ch == ",":
            fields.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    fields.append("".join(cur).strip())
    return fields


def parse(text):
    return [parse_line(l) for l in text.splitlines() if l.strip()]
'''},
)

# 6 ---------------------------------------------------------------- rate limiter
task(
    "token-bucket-refill",
    "The TokenBucket in bucket.py lets too few requests through when calls are close "
    "together: partial tokens earned between calls are thrown away. It should refill "
    "continuously (fractional tokens count) and never exceed capacity. Fix bucket.py "
    "without making it depend on the real clock -- it takes `now` as an argument.",
    {
        "bucket.py": '''class TokenBucket:
    def __init__(self, capacity, rate_per_sec, now=0.0):
        self.capacity = capacity
        self.rate = rate_per_sec
        self.tokens = capacity
        self.last = now

    def allow(self, now, cost=1):
        elapsed = int(now - self.last)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False
''',
    },
    '''import unittest
from bucket import TokenBucket


class H(unittest.TestCase):
    def test_fractional_refill(self):
        b = TokenBucket(2, 1.0, now=0.0)
        self.assertTrue(b.allow(0.0))
        self.assertTrue(b.allow(0.0))
        self.assertFalse(b.allow(0.0))
        self.assertFalse(b.allow(0.5))
        self.assertTrue(b.allow(1.0))

    def test_many_small_steps(self):
        b = TokenBucket(1, 1.0, now=0.0)
        self.assertTrue(b.allow(0.0))
        got = [b.allow(t / 10) for t in range(1, 21)]
        self.assertEqual(sum(got), 2)

    def test_capacity_cap(self):
        b = TokenBucket(3, 10.0, now=0.0)
        self.assertTrue(b.allow(100.0, cost=3))
        self.assertFalse(b.allow(100.0))

    def test_cost_over_capacity_never_allowed(self):
        b = TokenBucket(2, 1.0, now=0.0)
        self.assertFalse(b.allow(1000.0, cost=3))
''',
    {"bucket.py": '''class TokenBucket:
    def __init__(self, capacity, rate_per_sec, now=0.0):
        self.capacity = capacity
        self.rate = rate_per_sec
        self.tokens = capacity
        self.last = now

    def allow(self, now, cost=1):
        elapsed = max(0.0, now - self.last)
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last = now
        if self.tokens >= cost - 1e-9:
            self.tokens -= cost
            return True
        return False
'''},
)

# 7 ---------------------------------------------------------------- CLI feature
task(
    "cli-json-flag",
    "Add a `--json` flag to the `stats` command in cli.py. With it, print the result of "
    "`summarize` as a single JSON object (keys count, total, mean, min, max) instead of "
    "the text lines. Without the flag, behaviour must not change. Do not change "
    "calc.py's function signatures.",
    {
        "calc.py": '''def summarize(values):
    values = list(values)
    if not values:
        raise ValueError("no values")
    return {
        "count": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }
''',
        "cli.py": '''import argparse
import sys

from calc import summarize


def build_parser():
    p = argparse.ArgumentParser(prog="calc")
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stats", help="summary statistics of numbers")
    st.add_argument("numbers", nargs="+", type=float)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "stats":
        s = summarize(args.numbers)
        for k in ("count", "total", "mean", "min", "max"):
            print(f"{k}: {s[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''',
    },
    '''import io
import json
import unittest
from contextlib import redirect_stdout

import cli


def run(argv):
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


class H(unittest.TestCase):
    def test_text_unchanged(self):
        code, out = run(["stats", "1", "2", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(out.splitlines(),
                         ["count: 3", "total: 6.0", "mean: 2.0", "min: 1.0", "max: 3.0"])

    def test_json(self):
        code, out = run(["stats", "--json", "1", "2", "3"])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(out),
                         {"count": 3, "total": 6.0, "mean": 2.0, "min": 1.0, "max": 3.0})
        self.assertEqual(len(out.strip().splitlines()), 1)

    def test_json_flag_after_numbers(self):
        code, out = run(["stats", "4", "--json"])
        self.assertEqual(json.loads(out)["count"], 1)
''',
    {"cli.py": '''import argparse
import json
import sys

from calc import summarize


def build_parser():
    p = argparse.ArgumentParser(prog="calc")
    sub = p.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("stats", help="summary statistics of numbers")
    st.add_argument("numbers", nargs="+", type=float)
    st.add_argument("--json", action="store_true", help="print JSON")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.cmd == "stats":
        s = summarize(args.numbers)
        if args.json:
            print(json.dumps(s))
        else:
            for k in ("count", "total", "mean", "min", "max"):
                print(f"{k}: {s[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''},
)

# 8 ---------------------------------------------------------------- shared default
task(
    "config-shared-default",
    "Settings leak between callers: after `load_settings({'db': {'port': 1}})` is called, "
    "a later `load_settings({})` returns port 1 as well, and mutating a returned dict "
    "changes the defaults. Find the root cause (it spans config.py and defaults.py) "
    "and fix it so every call returns an independent, fully merged result.",
    {
        "defaults.py": '''DEFAULTS = {
    "debug": False,
    "db": {"host": "localhost", "port": 5432, "options": ["ssl"]},
    "workers": 4,
}


def get_defaults():
    return DEFAULTS
''',
        "config.py": '''from defaults import get_defaults


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(overrides=None):
    return _merge(get_defaults(), overrides or {})
''',
    },
    '''import unittest

import config
import defaults


class H(unittest.TestCase):
    def test_no_leak_between_calls(self):
        a = config.load_settings({"db": {"port": 1}})
        self.assertEqual(a["db"]["port"], 1)
        self.assertEqual(a["db"]["host"], "localhost")
        b = config.load_settings({})
        self.assertEqual(b["db"]["port"], 5432)

    def test_mutation_does_not_touch_defaults(self):
        a = config.load_settings()
        a["db"]["options"].append("x")
        a["workers"] = 99
        b = config.load_settings()
        self.assertEqual(b["db"]["options"], ["ssl"])
        self.assertEqual(b["workers"], 4)

    def test_overrides_not_mutated(self):
        o = {"db": {"port": 7}}
        r = config.load_settings(o)
        r["db"]["port"] = 8
        self.assertEqual(o, {"db": {"port": 7}})

    def test_defaults_unchanged(self):
        config.load_settings({"workers": 1})
        self.assertEqual(defaults.get_defaults()["workers"], 4)
''',
    {"defaults.py": '''import copy

DEFAULTS = {
    "debug": False,
    "db": {"host": "localhost", "port": 5432, "options": ["ssl"]},
    "workers": 4,
}


def get_defaults():
    return copy.deepcopy(DEFAULTS)
''',
     "config.py": '''import copy

from defaults import get_defaults


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def load_settings(overrides=None):
    return _merge(get_defaults(), overrides or {})
'''},
)
