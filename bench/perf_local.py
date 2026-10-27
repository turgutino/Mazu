"""Local, free performance and robustness measurements for Mazu: memory retrieval vs size, checkpoint cost
vs project size, restore() interrupted at every step, and CLI startup. No network, no model calls;
everything runs in a throwaway directory with an isolated HOME. Memory text is synthetic.
Run: python bench/perf_local.py"""
import json, os, random, shutil, statistics, subprocess, sys, tempfile, time, types
from pathlib import Path
from unittest import mock

ROOT = Path(tempfile.mkdtemp(prefix="mazu_perf_"))
os.environ["HOME"] = str(ROOT / "home"); os.environ["USERPROFILE"] = str(ROOT / "home")
(ROOT / "home").mkdir()
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazu.memory.store import MemoryStore
from mazu.memory.retrieval import build_context_block
from mazu.checkpoint.manager import CheckpointManager
import mazu.checkpoint.manager as mgr
from mazu.skills.manager import SkillManager

def git(cwd, *a):
    return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)

def new_repo(name):
    p = ROOT / name; p.mkdir()
    git(p, "init", "-q"); git(p, "config", "user.email", "t@t.invalid"); git(p, "config", "user.name", "t")
    return p

def med(xs): return statistics.median(xs)

# ---------------------------------------------------------------- E1 memory retrieval vs size
print("== E1. Memory: context-block build time and duplicate check vs number of memories ==")
rnd = random.Random(7)
vocab = [f"{a}{b}" for a in ("auth","db","cache","queue","test","build","deploy","api","user","log","cfg","net","file","parse","index")
         for b in ("", "er", "ing", "ed", "s", "_v2", "_old", "x")]
def sentence(n): return " ".join(rnd.choice(vocab) for _ in range(n))
cats = ["decision", "convention", "mistake", "task_outcome", "fact"]
mp = ROOT / "mem"; mp.mkdir()
store = MemoryStore(mp / "memory.db")
rows = []
inserted = 0
for target in (100, 1000, 5000, 10000):
    while inserted < target:
        store.add(category=rnd.choice(cats), title=sentence(6), body=sentence(40), tags=sentence(3),
                  source="auto_extracted", session_id="s")
        inserted += 1
    q = "fix the authentication cache bug in the parser"
    build = []
    for _ in range(5):
        t = time.perf_counter(); blk = build_context_block(store, query=q); build.append(time.perf_counter() - t)
    dup = []
    for _ in range(5):
        t = time.perf_counter(); store.find_duplicate("fact", sentence(6), sentence(40)); dup.append(time.perf_counter() - t)
    size = (mp / "memory.db").stat().st_size / 1e6
    rows.append((target, med(build) * 1000, med(dup) * 1000, size, len(blk)))
print(f"{'memories':>9} {'context build (ms)':>20} {'duplicate check (ms)':>22} {'db size (MB)':>13} {'block chars':>12}")
for r in rows: print(f"{r[0]:>9} {r[1]:>20.1f} {r[2]:>22.1f} {r[3]:>13.2f} {r[4]:>12}")
store.close()

# ---------------------------------------------------------------- E2 checkpoint cost vs repo size
print("\n== E2. Checkpoint snapshot time and disk use vs project size ==")
print(f"{'files':>6} {'repo MB':>8} {'snapshot (ms)':>14} {'restore (ms)':>13} {'per-checkpoint MB':>18}")
for nfiles in (50, 500, 3000):
    p = new_repo(f"proj{nfiles}")
    for i in range(nfiles):
        d = p / f"d{i % 20}"; d.mkdir(exist_ok=True)
        (d / f"f{i}.py").write_text("x = %d\n" % i * 50)
    git(p, "add", "-A"); git(p, "commit", "-qm", "init")
    ms = MemoryStore(p / ".mazu" / "memory.db")
    for i in range(300): ms.add(category="fact", title=f"t{i}", body=sentence(40), source="explicit", session_id="s")
    ms.close()
    cm = CheckpointManager(p)
    times = []
    for k in range(5):
        (p / "d0" / "f0.py").write_text(f"x = {k}\n")
        t = time.perf_counter(); cm.snapshot([{"role": "user", "content": "hi"}], "bench"); times.append(time.perf_counter() - t)
    cps = cm.list_checkpoints()
    cp_dir = p / ".mazu" / "checkpoints" / cps[-1]["id"]
    cp_mb = sum(f.stat().st_size for f in cp_dir.rglob("*") if f.is_file()) / 1e6
    repo_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".git" not in f.parts and ".mazu" not in f.parts) / 1e6
    t = time.perf_counter(); cm.restore(cps[0]["id"]); rest = time.perf_counter() - t
    print(f"{nfiles:>6} {repo_mb:>8.2f} {med(times) * 1000:>14.0f} {rest * 1000:>13.0f} {cp_mb:>18.2f}")

# ---------------------------------------------------------------- E3 restore fault injection
print("\n== E3. restore() interrupted at each step: is the state consistent, and does a retry repair it? ==")
STEPS = ["git reset --hard", "git clean -fd", "memory.db restore", "skills delete", "skills copy", "read conversation", "truncate index"]

def build_state():
    p = new_repo("fi_%d" % random.randrange(10**9))
    (p / "a.py").write_text("A\n"); git(p, "add", "-A"); git(p, "commit", "-qm", "init")
    ms = MemoryStore(p / ".mazu" / "memory.db"); ms.add(category="fact", title="m1", body="first", source="explicit", session_id="s"); ms.close()
    sm = SkillManager(p); sm.save("s1", "d", "def run(args):\n    return 'x'\n")
    cm = CheckpointManager(p)
    cp1 = cm.snapshot([{"role": "user", "content": "state A"}], "A")["id"]
    (p / "a.py").write_text("B\n")
    ms = MemoryStore(p / ".mazu" / "memory.db"); ms.add(category="fact", title="m2", body="second", source="explicit", session_id="s"); ms.close()
    sm.save("s2", "d", "def run(args):\n    return 'y'\n")
    cm.snapshot([{"role": "user", "content": "state B"}], "B")
    return p, cm, cp1

def observe(p, cm):
    code = (p / "a.py").read_text().strip()
    ms = MemoryStore(p / ".mazu" / "memory.db"); mem = sorted(r["title"] for r in ms.all_active()); ms.close()
    skills = sorted(d.name for d in (p / ".mazu" / "skills").iterdir()) if (p / ".mazu" / "skills").exists() else []
    idx = [e["id"] for e in cm.list_checkpoints()]
    return code, mem, skills, idx

def era(obs):
    code, mem, skills, _ = obs
    marks = {"A" if code == "A" else "B", "A" if mem == ["m1"] else "B", "A" if skills == ["s1"] else "B"}
    return "consistent " + marks.pop() if len(marks) == 1 else "MIXED (code=%s memory=%s skills=%s)" % (code, ",".join(mem), ",".join(skills))

class Boom(RuntimeError): pass

def run_with_crash(target_step):
    p, cm, cp1 = build_state()
    counter = {"n": 0}
    def tick(real):
        def wrapped(*a, **k):
            counter["n"] += 1
            if counter["n"] == target_step: raise Boom(STEPS[target_step - 1])
            return real(*a, **k)
        return wrapped
    real_git = mgr._git
    proxy_shutil = types.SimpleNamespace(**{n: getattr(shutil, n) for n in dir(shutil) if not n.startswith("__")})
    proxy_shutil.rmtree = tick(shutil.rmtree); proxy_shutil.copytree = tick(shutil.copytree)
    proxy_json = types.SimpleNamespace(**{n: getattr(json, n) for n in dir(json) if not n.startswith("__")})
    proxy_json.loads = tick(json.loads)
    with mock.patch.object(mgr, "_git", tick(real_git)), mock.patch.object(mgr, "_backup_sqlite", tick(mgr._backup_sqlite)), \
         mock.patch.object(mgr, "shutil", proxy_shutil), mock.patch.object(mgr, "json", proxy_json), \
         mock.patch.object(type(cm.index), "truncate_after", tick(type(cm.index).truncate_after)):
        try: cm.restore(cp1); crashed = False
        except Boom: crashed = True
    after_crash = observe(p, cm)
    try: cm.restore(cp1); retry = "retry OK"
    except Exception as e: retry = "retry FAILED: %s" % e
    after_retry = observe(p, cm)
    return crashed, after_crash, retry, after_retry

print(f"{'interrupted before step':<26} {'state right after the crash':<62} {'after retrying restore'}")
mixed = 0; repaired = 0
for i, name in enumerate(STEPS, 1):
    crashed, ac, retry, ar = run_with_crash(i)
    e1 = era(ac); e2 = era(ar)
    ok_retry = e2 == "consistent A" and ar[3] == ac[3][:1] + [] or e2 == "consistent A"
    mixed += e1.startswith("MIXED"); repaired += (e1.startswith("MIXED") and e2 == "consistent A")
    print(f"{i}. {name:<22} {e1:<62} {e2}  index={ar[3]}")
print(f"\ncrash points leaving a MIXED state: {mixed} of {len(STEPS)}; of those, repaired by simply running restore again: {repaired}")

# ---------------------------------------------------------------- E4 startup
print("\n== E4. CLI startup ==")
ts = []
for _ in range(5):
    t = time.perf_counter(); subprocess.run([sys.executable, "-m", "mazu.cli", "--version"], capture_output=True, cwd=ROOT); ts.append(time.perf_counter() - t)
print(f"`mazu --version`: median {med(ts) * 1000:.0f} ms")
shutil.rmtree(ROOT, ignore_errors=True)
