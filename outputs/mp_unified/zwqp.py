"""WQP: U2s's pool pick, chosen again under WQ (docs/WQP_POOL_PICK_UNDER_WQ.md).

webqsp's select carve (M3B's; 305 questions with a gold in the graph) goes through the committed chain that filed
webqsp's s1eval stats for U2s, step 4d to U2s: twelve stages, each imported unchanged and run in its own process under
this file's patches (section 1). Before them, stages 1-5 run again on webqsp's s1eval carve through the same patches
and must give the filed arrays (section 2). Then U2s's choice (its section 6) on WQ's selection set: the four passage
sets' s1sel carves and webqsp's select carve, metaqa out (section 3).

    python outputs/mp_unified/zwqp.py chain [--threads 6] [--retry]   (the check, stages 1-12, then the verdict;
                                                                       resumes after the stages the ledger holds)
    python outputs/mp_unified/zwqp.py stage NAME [--threads 6] [--check]
    python outputs/mp_unified/zwqp.py compare-check
    python outputs/mp_unified/zwqp.py reproduce                       (U2s's verdict from the six filed tables)
    python outputs/mp_unified/zwqp.py verdict
    python outputs/mp_unified/zwqp.py --selftest
"""
import os
import sys

BLAS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
for _v in BLAS:
    os.environ.setdefault(_v, "2")              # every chain module's own setting, here before numpy loads
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import copy  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
import types  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zwq as WQ  # noqa: E402  (patch, same_arrays and the small helpers; stdlib only at import)

DECLARED = "docs/WQP_POOL_PICK_UNDER_WQ.md"
STAGE = "WQP"
DS, CARVE, KEY = "webqsp", "select", "mselect"          # step 4c's CARVES_KEY: select -> mselect
SELECT_N = 305
SELECT_SHA = "d3c3cc4ba8d4d03983e3713500b585b1fabd0e597a8e317cf267c286591d4a96"   # the select look's carve_ids_sha256
CHECK_CARVE, CHECK_N = "s1eval", 1503
CHECK_SHA = "faa1d372294cc62f116549bdce087bb0d3f8ce533dce8bfd814c56e1c42e7a2e"    # carves.json's webqsp s1eval
FREEZE = "58958f33a3af74c21fa625ffd79da16c573cb5d2ba7e9745da0056b568591ab0"
OUT = "outputs/wqp"
CHECK_OUT = "outputs/wqp/check"
LEDGER = "outputs/wqp/ledger.json"
CHECK = "outputs/wqp/check.json"
VERDICT = "outputs/wqp/verdict"
STAGES = ("s4d", "s4e", "p1d", "p1e", "p1f", "p1g", "u2c", "u2e", "u2m", "u2m_b", "u2m_a", "u2s")
CHECKED = STAGES[:5]
SUB = {"s4d": "step4d/pools", "s4e": "step4e/pools", "p1d": "p1d", "p1e": "p1e", "p1f": "p1f", "p1g": "p1g",
       "u2c": "u2c", "u2e": "u2e", "u2m": "u2m", "u2m_b": "u2m/b", "u2m_a": "u2m/a", "u2s": "u2s"}
FILED = {"s4d": "outputs/step4d/pools/webqsp__s1eval.npz", "s4e": "outputs/step4e/pools/webqsp__s1eval.npz",
         "p1d": "outputs/p1d/pos_webqsp__s1eval.npz", "p1e": "outputs/p1e/pos_webqsp__s1eval.npz",
         "p1f": "outputs/p1f/stats_webqsp__s1eval.npz"}
PLACED = ("u2m/verdict_ce.json", "u2m/b/verdict_b.json", "u2m/a/verdict_a.json")     # under outputs/ and OUT
WATCH = ("step1", "step4b", "step4d", "step4e", "p1c2", "p1d", "p1e", "p1f", "p1g", "p1g2", "u2b", "u2c", "u2e",
         "u2m", "u2p", "u2q", "u2s", "m3b", "d5", "mp_unified/look")                # the chain's filed roots
PASSAGE_SEL = ("squad", "musique", "hotpotqa", "2wiki")
SEL = PASSAGE_SEL + (DS,)
KEYS = ("metaqa",) + PASSAGE_SEL + ("webqsp_s1eval", DS)                            # the tables: S and beside
LOO = {"WQ-musique": "musique", "WQ-hotpotqa": "hotpotqa"}
CHOSEN = "g:f3+cut_cos_7000@0.9"                                                    # U2c's chosen pool
U2S_CAP = 1953.5
U2S_VERDICT = "outputs/u2s/verdict.json"
U2C_VERDICT = "outputs/u2c/verdict.json"
P1C2_WEBQSP = "outputs/p1c2/pools/webqsp/pools.json"
THREADS = 6
NO_LEVEL = 0.94                         # an incumbent that meets no level: below every level (0.950 - 2 x 0.005)
PATCHED = ("S4", "C4", "D5", "E", "D4", "D", "K", "F", "PG", "U2C", "U2E", "M", "MB", "A", "S", "P")
for _t in (STAGES, tuple(SUB), tuple(FILED), PLACED, WATCH, SEL, KEYS, PATCHED):
    assert len(set(_t)) == len(_t), _t
assert tuple(SUB) == STAGES and tuple(FILED) == CHECKED
assert set(LOO) == set(WQ.SPLITS) and set(SEL) == set(WQ.SPLITS["WQ-musique"]) | set(WQ.SPLITS["WQ-hotpotqa"])
assert all(LOO[sp] in WQ.ZERO_SHOT[sp] and set(SEL) - {LOO[sp]} == set(WQ.SPLITS[sp]) for sp in LOO)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zwqp: {msg}", flush=True)


def rel(p, base=ROOT):
    return WQ.rel(p, base)


# ── the chain's modules and the patches (section 1) ──────────────────────────


def load():
    """The chain's modules, each one object (a patch then reaches every stage that reads it)."""
    import p1e_anchor_kind as K                     # P1e's walks; through it P1d, step 4e, step 4c and step 4
    import p1f_pool_size as F
    import p1g_pool_beam as PG
    import p1g2_gated as PG2
    import d5_gold_reach as D5
    import step4b_pool_reach as S4B
    import zu2c as U2C
    import zu2e as U2E
    import zu2m as M
    import zu2m_b as MB
    import zu2m_a as A
    import zu2m_a_types as AT
    import zu2p as P
    import zu2q as Q
    import zu2s as S
    E = K.E
    ns = types.SimpleNamespace(K=K, D=K.D, G=K.G, E=E, S4=K.S4, C4=E.C4, D4=E.D4, F=F, PG=PG, PG2=PG2, D5=D5,
                               S4B=S4B, U2C=U2C, U2E=U2E, M=M, MB=MB, A=A, AT=AT, P=P, Q=Q, S=S)
    by_file = {Path(getattr(ns, k).__file__).resolve(): getattr(ns, k) for k in PATCHED}
    if len(by_file) != len(PATCHED):
        raise SystemExit("zwqp: two of the patched names are one module")
    seen = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).resolve() in by_file:
            seen.setdefault(Path(f).resolve(), set()).add(id(m))
    for k in vars(ns):
        for v in vars(getattr(ns, k)).values():
            if isinstance(v, types.ModuleType) and getattr(v, "__file__", None):
                t = by_file.get(Path(v.__file__).resolve())
                if t is not None and v is not t:
                    raise SystemExit(f"zwqp: {k} holds a second copy of {Path(v.__file__).name}")
    if any(len(s) != 1 for s in seen.values()) or len(seen) != len(by_file):
        raise SystemExit("zwqp: a patched module is loaded twice, or not from sys.modules")
    return ns


def with_select(base):
    """base (step 4's Opened) with webqsp's select carve beside carves.json's (section 1: M3B's select population, as
    zwq's with_fit adds the fit carve): step 4c's populations then build and check it as any other select carve."""

    class OpenedWQP(base):
        def __init__(self):
            super().__init__()
            pd = self.carves["per_dataset"][DS]
            ent = {"name": CARVE, "n": SELECT_N, "sha256": SELECT_SHA}
            if pd.get(KEY, ent) != ent:
                raise SystemExit(f"zwqp: carves.json holds a webqsp {KEY} carve {pd[KEY]}, not M3B's select")
            pd[KEY] = ent

    OpenedWQP.__name__ = OpenedWQP.__qualname__ = f"WQP{base.__name__}"
    return OpenedWQP


def read_ledger(base=ROOT):
    p = base / LEDGER
    if not p.is_file():
        return {"declared_in": DECLARED, "stage": STAGE, "stages": {}, "earlier": []}
    return WQ.read_json(p)


def file_entry(key, entry, base=ROOT):
    led = read_ledger(base)
    if key in led["stages"]:
        led.setdefault("earlier", []).append({"stage": key, **led["stages"][key]})
    led["stages"][key] = entry
    led["utc"] = WQ.utc()
    WQ.write_json(base / LEDGER, led)


def entry_ok(led, key, base=ROOT):
    """Whether the ledger's entry for key is OK and its record and arrays are still the bytes it filed."""
    e = led["stages"].get(key)
    if not e or e.get("status") != "OK" or not e.get("record") or not e.get("file"):
        return False
    rp, fp = base / e["record"], base / e["file"]
    return (rp.is_file() and fp.is_file() and WQ.sha_file(rp) == e["record_sha256"]
            and WQ.sha_file(fp) == e["file_sha256"])


def ledger_file(stage, base=ROOT):
    """The sha256 of the select arrays WQP's ledger filed for stage, its record on disk checked to be the one filed."""
    e = read_ledger(base)["stages"].get(stage)
    if not e or e.get("status") != "OK" or e.get("carve") != CARVE:
        raise SystemExit(f"zwqp: the ledger holds no OK select entry for {stage}")
    rp = base / e["record"]
    if not rp.is_file() or WQ.sha_file(rp) != e["record_sha256"]:
        raise SystemExit(f"zwqp: {e['record']} is not the record the ledger filed for {stage}")
    if WQ.read_json(rp).get("file_sha256") != e["file_sha256"]:
        raise SystemExit(f"zwqp: {e['record']} does not name the arrays the ledger filed for {stage}")
    return e["file_sha256"]


def repointed(orig, at, prev, base=ROOT):
    """orig (a carried() whose result's at-th item is a filed verdict) with the verdict's webqsp entry naming the select
    arrays WQP's ledger filed for the stage before (section 1's three re-pointed checks); the filed verdict's object
    is never changed, and every other dataset's entry stays the filed one."""

    def carried(*a, **kw):
        res = list(orig(*a, **kw))
        v = copy.deepcopy(res[at])
        if not isinstance(v, dict) or DS not in v.get("stats", {}):
            raise SystemExit(f"zwqp: {getattr(orig, '__name__', orig)}'s verdict has no {DS} entry to re-point")
        v["stats"][DS]["file_sha256"] = ledger_file(prev, base)
        res[at] = v
        return tuple(res)

    carried.__name__ = getattr(orig, "__name__", "carried")
    carried.__wrapped__ = orig
    return carried


@contextlib.contextmanager
def on_wqp(ns, check=False, base=ROOT):
    """Section 1's patches for the block, every one restored after: each stage's root under outputs/wqp (with check,
    outputs/wqp/check), webqsp's select entry beside carves.json's; without check also the carve switches (select for
    webqsp) and the three re-pointed checks."""
    out = base / (CHECK_OUT if check else OUT)
    r = {s: out / SUB[s] for s in STAGES}
    with contextlib.ExitStack() as st:
        def put(obj, **kw):
            st.enter_context(WQ.patch(obj, **kw))
        put(ns.S4, Opened=with_select(ns.S4.Opened))
        put(ns.D4, POOLS=r["s4d"], MANIFEST=r["s4d"] / "pools.json")
        put(ns.E, POOLS=r["s4e"], MANIFEST=r["s4e"] / "pools.json")
        put(ns.D, OUT=r["p1d"])
        put(ns.K, OUT=r["p1e"])
        put(ns.F, OUT=r["p1f"], P1E=r["p1e"])
        put(ns.PG, OUT=r["p1g"])
        put(ns.U2C, OUT=r["u2c"])
        put(ns.U2E, OUT=r["u2e"])
        put(ns.M, OUT=r["u2m"])
        put(ns.MB, OUT=r["u2m_b"])
        put(ns.A, OUT=r["u2m_a"])
        put(ns.S, OUT=r["u2s"])
        if not check:
            of_c4, of_d5, rc = ns.C4.carves_of, ns.D5.carves_of, ns.E.run_carves

            def run_carves(op, name, wanted, *a, **kw):
                return rc(op, name, {CARVE} if name == DS else wanted, *a, **kw)
            put(ns.C4, carves_of=lambda name: (CARVE,) if name == DS else of_c4(name))
            put(ns.D5, carves_of=lambda name: (CARVE,) if name == DS else of_d5(name))
            put(ns.E, run_carves=run_carves)
            put(ns.MB, carried=repointed(ns.MB.carried, 0, "u2m", base))
            put(ns.A, carried_family=repointed(ns.A.carried_family, 3, "u2m_b", base))
            put(ns.P, carried=repointed(ns.P.carried, 4, "u2m_a", base))
        yield ns


def place_verdicts(base=ROOT):
    """(c) and (e)'s, (b)'s and (a)'s filed verdicts copied under outputs/wqp byte for byte: every sha256 a stage
    checks or records is then the filed one. A file already there must be those bytes; nothing is overwritten."""
    out = {}
    for p in PLACED:
        src, dst = base / "outputs" / p, base / OUT / p
        sha = WQ.sha_file(src)
        if dst.exists():
            if WQ.sha_file(dst) != sha:
                raise SystemExit(f"zwqp: {rel(dst, base)} is present and is not {rel(src, base)}'s bytes")
            out[p] = "present"
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.name}.{os.getpid()}.tmp")
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        if WQ.sha_file(dst) != sha:
            raise SystemExit(f"zwqp: {rel(dst, base)} was copied, but its sha256 is not {sha[:12]}")
        out[p] = "copied"
    return out


# ── one stage, its checks and its ledger entry (section 5) ───────────────────


def run_stage(ns, name, threads):
    return {"s4d": lambda: ns.D4.pools_stage([DS], threads),
            "s4e": lambda: ns.E.file_stage([DS], threads),
            "p1d": lambda: ns.D.pools_stage(DS, threads),
            "p1e": lambda: ns.K.pools_stage(DS, threads),
            "p1f": lambda: ns.F.stats_stage(DS, threads),
            "p1g": lambda: ns.PG.stats_stage(DS, 0, threads),
            "u2c": lambda: ns.U2C.stats_stage(DS, "today", threads),
            "u2e": lambda: ns.U2E.stats_stage(DS, threads),
            "u2m": lambda: ns.M.stats_stage(DS, threads),
            "u2m_b": lambda: ns.MB.stats_stage(DS, threads),
            "u2m_a": lambda: ns.A.stats_stage(DS, threads),
            "u2s": lambda: ns.S.stats_stage(DS, threads)}[name]()


def record_of(stage, root, carve):
    """(the record's path, its webqsp/carve entry or None, the arrays' path or None, the sha256 the entry gives the
    arrays, the record's head) of one stage's filing under root (outputs/, outputs/wqp or outputs/wqp/check)."""
    key = f"{DS}__{carve}"
    d = Path(root) / SUB[stage]
    if stage in ("s4d", "s4e"):
        rp = d / "pools.json"
    elif stage in ("p1d", "p1e"):
        rp = d / f"pools_{DS}.json"
    elif stage in ("p1f", "p1g"):
        rp = d / f"stats_{DS}.json"
    else:
        rp = d / "stats" / ("today" if stage == "u2c" else "") / f"{key}.json"
    head = WQ.read_json(rp) if rp.is_file() else None
    if head is None:
        return rp, None, None, None, None
    if stage in ("s4d", "s4e"):
        ent = head.get("files", {}).get(key)
        return rp, ent, None if ent is None else d / str(ent.get("file")), None if ent is None else ent.get("sha256"), head
    if stage in ("p1d", "p1e"):
        ent = head.get("carves", {}).get(key)
        return rp, ent, d / f"pos_{key}.npz", None if ent is None else ent.get("positions_npz_sha256"), head
    if stage in ("p1f", "p1g"):
        ent = head.get("carves", {}).get(key)
        return rp, ent, d / f"stats_{key}.npz", None if ent is None else ent.get("file_sha256"), head
    return rp, head, rp.with_suffix(".npz"), head.get("file_sha256"), head


def problems(stage, root, carve, n, digest, base=ROOT):
    """Why a stage's filing of webqsp/carve under root is not the one section 5 requires, or []."""
    rp, ent, fp, sha, head = record_of(stage, root, carve)
    if ent is None:
        return [f"{stage}: no {DS}/{carve} entry in {rel(rp, base)}"]
    out = []
    if int(ent.get("questions", -1)) != n:
        out.append(f"{stage}: {ent.get('questions')} questions, not {n}")
    for k in ("carve_ids_sha256", "carve_sha256"):
        if k in ent and ent[k] != digest:
            out.append(f"{stage}: {k} {str(ent[k])[:12]}, not {digest[:12]}")
    for k, want in (("dataset", DS), ("carve", carve)):
        if k in ent and ent[k] != want:
            out.append(f"{stage}: {k} {ent[k]}, not {want}")
    fz = head.get("freeze_RECORD_SHA256")
    if fz is not None and fz != FREEZE:
        out.append(f"{stage}: freeze {str(fz)[:12]}, not {FREEZE[:12]}")
    if head.get("limit") not in (None, 0):
        out.append(f"{stage}: a --limit run ({head.get('limit')})")
    if fp is None or not fp.is_file() or WQ.sha_file(fp) != sha:
        out.append(f"{stage}: {rel(fp, base) if fp else '(no file)'} is missing or not its record's sha256")
    chk = ent.get("checks")
    if isinstance(chk, dict):
        bad = sorted(c for c, v in chk.items() if c != "rows_served_verified" and not v)
        if bad:
            out.append(f"{stage}: checks {bad} did not hold")
    return out


def loader_check(ns, stage, carve):
    """The stage's own loader on what it filed (its script, its file's sha256, its checks and its summary); it stops
    on a problem. Step 4d, step 4e, P1d and P1e have none."""
    def need(x):
        if x is None:
            raise SystemExit(f"{stage}: its loader finds no {DS}/{carve} arrays")
    if stage == "p1f":
        ns.F.load_stats(DS, carve)
    elif stage == "p1g":
        ns.PG2.load_arrays(DS, carve)
    elif stage == "u2c":
        need(ns.U2C.load_stats(DS, "today"))
    elif stage == "u2e":
        need(ns.U2E.load_u2e(DS))
    elif stage == "u2m":
        need(ns.M.load_u2m(DS))
    elif stage == "u2m_b":
        need(ns.MB.load_b(DS, ns.MB.carried()[3]))
    elif stage == "u2m_a":
        _v, names, g, _vb = ns.A.carried_family()
        need(ns.A.load_a(DS, names, g))
    elif stage == "u2s":
        need(ns.S.load_u2s(DS))


def snapshot(base=ROOT):
    """(size, mtime) of every file under the chain's filed roots (WATCH), __pycache__ left out."""
    out = {}
    for w in WATCH:
        for r, dirs, files in os.walk(base / "outputs" / w):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            for f in files:
                p = Path(r) / f
                try:
                    s = p.stat()
                except FileNotFoundError:
                    continue
                out[p.relative_to(base).as_posix()] = (s.st_size, s.st_mtime_ns)
    return out


def changed(before, after):
    return sorted(k for k in set(before) | set(after) if before.get(k) != after.get(k))


def stage(name, threads=THREADS, check=False, base=ROOT):
    """One stage in this process under the patches, then section 5's checks; its ledger entry either way."""
    if name not in STAGES or (check and name not in CHECKED):
        raise SystemExit(f"zwqp: no stage {name}{' in the check' if check else ''}")
    carve, n, digest = (CHECK_CARVE, CHECK_N, CHECK_SHA) if check else (CARVE, SELECT_N, SELECT_SHA)
    key = f"check:{name}" if check else name
    root = base / (CHECK_OUT if check else OUT)
    probs, placed = [], None
    ns = load()
    if not check:
        placed = place_verdicts(base)
    before = snapshot(base)
    log(f"{key}: {DS}/{carve}, numba {threads} threads, BLAS {[os.environ.get(v) for v in BLAS]}")
    t0 = time.time()
    try:
        with on_wqp(ns, check, base):
            rc = run_stage(ns, name, threads)
            if rc not in (0, None):
                probs.append(f"{name}: returned {rc}")
            probs += problems(name, root, carve, n, digest, base)
            if not probs:
                loader_check(ns, name, carve)
    except SystemExit as e:
        traceback.print_exc()
        probs.append(f"{name}: stopped: {WQ.clean(e)}")
    except Exception as e:                      # noqa: BLE001  (the ledger names it; the chain stops)
        traceback.print_exc()
        probs.append(f"{name}: {type(e).__name__}: {WQ.clean(e)}")
    secs = round(time.time() - t0, 1)
    ch = changed(before, snapshot(base))
    if ch:
        probs.append(f"{name}: {len(ch)} files changed outside outputs/wqp: {ch[:8]}")
    rp, ent, fp, sha, _h = record_of(name, root, carve)
    entry = {"status": "FAILED" if probs else "OK", "problems": probs, "carve": carve,
             "questions": None if ent is None else ent.get("questions"),
             "record": rel(rp, base), "record_sha256": WQ.sha_file(rp) if rp.is_file() else None,
             "file": None if fp is None else rel(fp, base),
             "file_sha256": WQ.sha_file(fp) if fp is not None and fp.is_file() else None,
             "seconds": secs, "threads": threads, "blas": {v: os.environ.get(v) for v in BLAS},
             "placed": placed, "script_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    file_entry(key, entry, base)
    log(f"{key}: {entry['status']} ({secs:.0f}s){'; ' + '; '.join(probs) if probs else ''}")
    return 1 if probs else 0


# ── the check (section 2) ────────────────────────────────────────────────────


def compare_check(base=ROOT):
    """Stages 1-5 on webqsp's s1eval carve (outputs/wqp/check) against the filed arrays: REPRODUCED needs every key,
    dtype, shape and byte of all five; the files' sha256 and the entries' other fields are reported beside."""
    led = read_ledger(base)
    missing = [s for s in CHECKED if not entry_ok(led, f"check:{s}", base)]
    if missing:
        raise SystemExit(f"zwqp: the check's stages {missing} are not OK in the ledger")
    rows = {}
    for s in CHECKED:
        _rp, ent, fp, _sha, _h = record_of(s, base / CHECK_OUT, CHECK_CARVE)
        _rp0, ent0, fp0, _sha0, _h0 = record_of(s, base / "outputs", CHECK_CARVE)
        if fp0 is None or rel(fp0, base) != FILED[s] or not fp0.is_file():
            raise SystemExit(f"zwqp: the filed {s} arrays are not {FILED[s]}")
        cmp = WQ.same_arrays(fp, fp0)
        rows[s] = {"new": rel(fp, base), "filed": FILED[s], "arrays": cmp,
                   "file_sha256": {"new": WQ.sha_file(fp), "filed": WQ.sha_file(fp0)},
                   "entry_fields_differ": sorted(k for k in set(ent) | set(ent0) if k not in ("seconds", "utc")
                                                 and ent.get(k) != ent0.get(k))}
    v = "REPRODUCED" if all(rows[s]["arrays"]["equal"] for s in CHECKED) else "NOT_REPRODUCED"
    rec = {"declared_in": DECLARED, "stage": STAGE, "check": "section 2", "verdict": v, "carve": CHECK_CARVE,
           "questions": CHECK_N, "stages": rows, "seconds": {s: led["stages"][f"check:{s}"]["seconds"] for s in CHECKED},
           "threads": {s: led["stages"][f"check:{s}"]["threads"] for s in CHECKED},
           "script_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    WQ.write_json(base / CHECK, rec)
    log(f"check: {v} ({', '.join(f'{s} {rows[s]['arrays']['equal']}' for s in CHECKED)})")
    return 0 if v == "REPRODUCED" else 2


# ── the chain ────────────────────────────────────────────────────────────────


def chain(threads=THREADS, retry=False, base=ROOT):
    """The check (stages 1-5 on s1eval), its comparison, stages 1-12 on select, then the verdict: each in its own
    process, one after another; a stage the ledger holds as OK (its bytes unchanged) is not run again, and a FAILED
    one stops the chain unless --retry."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    me = str(Path(__file__).resolve())

    def run(args):
        log(f"run {' '.join(args)}")
        return subprocess.run([sys.executable, me] + args, cwd=str(base), env=env).returncode

    def step(key, args):
        led = read_ledger(base)
        if entry_ok(led, key, base):
            log(f"{key}: OK in the ledger")
            return
        e = led["stages"].get(key)
        if e is not None and e.get("status") == "FAILED" and not retry:
            raise SystemExit(f"zwqp: {key} FAILED ({'; '.join(e.get('problems', []))}); --retry runs it again")
        rc = run(args)
        led = read_ledger(base)
        if rc != 0 and (led["stages"].get(key) or {}).get("status") != "FAILED":
            file_entry(key, {"status": "FAILED", "problems": [f"{key}: the process ended {rc} without its entry"],
                             "carve": CHECK_CARVE if key.startswith("check:") else CARVE, "record": None,
                             "file": None, "utc": WQ.utc()}, base)
        if rc != 0 or not entry_ok(read_ledger(base), key, base):
            raise SystemExit(f"zwqp: {key} did not pass; the chain stops")

    for s in CHECKED:
        step(f"check:{s}", ["stage", s, "--threads", str(threads), "--check"])
    if run(["compare-check"]) != 0:                 # always: the check's entries may have been run again
        raise SystemExit("zwqp: the check is NOT_REPRODUCED; select's stages do not run (section 2)")
    for s in STAGES:
        step(s, ["stage", s, "--threads", str(threads)])
    return run(["verdict"])


# ── the choice (section 3) ───────────────────────────────────────────────────


def choose(S, F, sel, w, cap, inc, tab_u, tab_o, gr, orr):
    """U2s section 6 on the datasets sel (weights w, the cap): tab_u and tab_o the universal and own tables ({d:
    (ALL, rows)} per candidate), gr and orr their recipes, inc the incumbent's own index. The incumbent's level and cost
    are read on sel; one that meets no level counts as NO_LEVEL, below every level."""
    import numpy as np
    A_ = np.stack([np.asarray(tab_u[d][0], np.float64) for d in sel], axis=1)
    RW = np.stack([np.asarray(tab_u[d][1], np.float64) for d in sel], axis=1)
    costs = np.asarray([F.cost(dict(zip(sel, r)), w) for r in RW.tolist()], np.float64)
    cost_r = np.asarray([round(float(c), 1) for c in costs])
    A_o = np.stack([np.asarray(tab_o[d][0], np.float64) for d in sel], axis=1)
    RW_o = np.stack([np.asarray(tab_o[d][1], np.float64) for d in sel], axis=1)
    c_o = np.asarray([F.cost(dict(zip(sel, r)), w) for r in RW_o.tolist()], np.float64)
    cr_o = np.asarray([round(float(c), 1) for c in c_o])
    ceil = A_.max(axis=0)
    l_inc = S.level_of(A_o[inc], ceil)
    c_inc = float(cr_o[inc])
    tf = S.three_frontiers(A_, costs, cost_r, np.asarray(gr), A_o, c_o, cr_o, np.asarray(orr), ceil, cap)
    lv = {k: tf[k][1] for k in tf}
    ix = {k: tf[k][2] for k in tf}
    cost_o = None if ix["O"] is None else float(cr_o[ix["O"]])
    vc, ve, vo, car = S.decide(lv["C"], lv["E"], lv["O"], cost_o, NO_LEVEL if l_inc is None else l_inc, c_inc)
    return {"sel": tuple(sel), "w": dict(w), "cap": float(cap), "ceil": ceil, "cost_r": cost_r, "cr_o": cr_o,
            "tf": tf, "level": lv, "index": ix, "inc": int(inc), "l_inc": l_inc, "c_inc": c_inc,
            "verdicts": (vc, ve, vo), "carried": car}


def carried_key(car):
    """zu2s's decide says 'U2p' when nothing beats its incumbent; here the incumbent is U2s's pick."""
    return "U2s" if car == "U2p" else car


def filed_six(ns, base=ROOT):
    """U2s's and U2c's filed verdicts, U2p's carried walk, U2c's cap under U2s's set and the six filed U2s stats
    (webqsp's s1eval keyed webqsp_s1eval), each checked as section 5 says; nothing here reads select."""
    import numpy as np
    S, M, P, Q, F, U2C = ns.S, ns.M, ns.P, ns.Q, ns.F, ns.U2C
    vs = WQ.read_json(base / U2S_VERDICT)
    if vs.get("script_sha256") != S.script_sha256():
        raise SystemExit(f"{U2S_VERDICT} is not the committed zu2s.py's verdict")
    if vs.get("carried") != "O" or vs.get("carried_pick") != vs["pick_O"]["name"]:
        raise SystemExit(f"{U2S_VERDICT} does not carry pick O, section 3's incumbent")
    vc_ = WQ.read_json(base / U2C_VERDICT)
    if vc_.get("script_sha256") != U2C.script_sha256() or vc_.get("chosen") != CHOSEN:
        raise SystemExit(f"{U2C_VERDICT} is not the committed zu2c.py's verdict choosing {CHOSEN}")
    cw = Q.carried_walk()
    vp = cw["verdict_p"]
    fc = vc_["graphs"]["chosen"]["chosen_s1sel"]
    w5 = F.weights()
    cap5 = round(F.cost({d: float(fc[d]["rows"]) for d in S.TRAIN5}, w5), 1)
    if not cap5 == U2S_CAP == float(vp["cap"]) == float(vc_["graphs"]["chosen"]["cost"]):
        raise SystemExit(f"U2c's chosen pool costs {cap5} under U2s's set; U2p's cap is {vp['cap']}, not {U2S_CAP}")
    am, rm = {}, {}
    for d in S.SIX:
        k = "webqsp_s1eval" if d == DS else d
        x = S.load_u2s(d)
        if x is None:
            raise SystemExit(f"{d}: no filed U2s stats")
        am[k], rm[k] = x
        if rm[k]["file_sha256"] != vs["stats"][d]["file_sha256"]:
            raise SystemExit(f"{d}: U2s's stats are not the file U2s's verdict read")
        if not np.array_equal(am[k]["ptr"], M.load_u2m(d)[0]["ptr"]):
            raise SystemExit(f"{d}: U2s's questions and golds are not U2m's")
    return types.SimpleNamespace(vs=vs, vc=vc_, cw=cw, vp=vp, vp_sha=WQ.sha_file(P.OUT / "verdict.json"),
                                 walk=(cw["pick"], list(cw["names"]), cw["G"], cw["beta"]), fc=fc, w5=w5,
                                 cap5=cap5, am=am, rm=rm)


def on_walk(fs, keys):
    """Every key's U2s stats on the walk U2p's filed verdict carries (pick, points, G, beta, its verdict's sha256)."""
    for k in keys:
        r = fs.rm[k]
        if (r["pick"], r["points"], r["G"], r["beta"]) != fs.walk or r["verdict_p_sha256"] != fs.vp_sha:
            raise SystemExit(f"{k}: U2s's stats are not on the walk U2p's filed verdict carries")


def tables(ns, fs, keys):
    """zu2s's tables on each key's arrays: every universal and own candidate's (ALL, rows), each candidate's recipe and
    budget form, the universal names (by index) and every own name with its index."""
    import numpy as np
    S, M = ns.S, ns.M
    tabs = {k: S.table_arrays(fs.am[k], range(len(S.GRID)), range(S.R), fs.rm[k]["cell_base"]) for k in keys}
    own = {k: S.table_arrays(fs.am[k], [fs.rm[k]["own_cell"]], range(S.R), fs.rm[k]["cell_base"]) for k in keys}
    nu, no_ = len(S.GRID) * S.R * S.NO * S.NB * S.NT, S.R * S.NO * S.NB * S.NT
    if any(tabs[k][0].size != nu or own[k][0].size != no_ for k in keys):
        raise SystemExit("a table is not zu2s's size")
    bud = S.budgets()
    _gk, gr, _go, gb, _gt = np.unravel_index(np.arange(nu), (len(S.GRID), S.R, S.NO, S.NB, S.NT))
    _ok, orr, _oo, ob, _ot = np.unravel_index(np.arange(no_), (1, S.R, S.NO, S.NB, S.NT))
    form_of_b = np.asarray([S.FORMS.index(x[1]) for x in bud])

    def name_u(i):
        k, r, o, b, t = S.decode(i, len(S.GRID), S.R)
        return M.candidate_name(S.GRID[k][0], r, o, bud[b][0], S.TAUS_DESC[t])

    names_o = []
    for i in range(no_):
        _k, r, o, b, t = S.decode(i, 1, S.R)
        names_o.append(M.candidate_name("own", r, o, bud[b][0], S.TAUS_DESC[t]))
    pos_o = {nm: i for i, nm in enumerate(names_o)}
    if len(pos_o) != no_:
        raise SystemExit("two own candidates share a name")
    return types.SimpleNamespace(tabs=tabs, own=own, nu=nu, no=no_, gr=gr, orr=orr, gf=form_of_b[gb],
                                 of=form_of_b[ob], name_u=name_u, names_o=names_o, pos_o=pos_o)


def pick_of(S, tb, res, key):
    """zu2s's record of frontier key's pick in a choice: level, name, cost and budget form."""
    i = res["index"][key]
    if i is None:
        return {"level": None, "name": None, "cost": None, "form": None}
    if key == "O":
        return {"level": res["level"][key], "name": tb.names_o[i], "cost": float(res["cr_o"][i]),
                "form": S.FORMS[tb.of[i]]}
    return {"level": res["level"][key], "name": tb.name_u(i), "cost": float(res["cost_r"][i]),
            "form": S.FORMS[tb.gf[i]]}


def reproduce(ns, fs, tb):
    """Section 4: U2s's verdict from these tables, under its own set (the five s1sel carves), P1f's weights, U2p's cap
    and U2p's incumbent. It stops unless every verdict, pick, level, cost and ceiling is U2s's filed one."""
    import numpy as np
    S, F = ns.S, ns.F
    cw, vs = fs.cw, fs.vs
    inc_point = cw["names"][cw["pi"]]
    i_p = tb.pos_o[inc_point]
    _cell, r_, o_, b_, t_ = ns.MB.parse(inc_point)
    if int(np.ravel_multi_index((0, r_, o_, b_, S.TAUS_DESC.index(t_)), (1, S.R, S.NO, S.NB, S.NT))) != i_p:
        raise SystemExit(f"{inc_point}: zu2s's index and this file's differ")
    rep = choose(S, F, S.TRAIN5, fs.w5, float(fs.vp["cap"]), i_p, tb.tabs, tb.own, tb.gr, tb.orr)
    u = vs["u2p_pick"]
    if rep["l_inc"] is None or abs(rep["l_inc"] - u["level"]) > 1e-9 or abs(rep["c_inc"] - u["cost"]) > 1e-9:
        raise SystemExit(f"U2p's pick reads {rep['l_inc']} at {rep['c_inc']}, not U2s's {u['level']} at {u['cost']}")
    got = {"verdict_c": rep["verdicts"][0], "verdict_e": rep["verdicts"][1], "verdict_o": rep["verdicts"][2],
           "carried": rep["carried"], "pick_C": pick_of(S, tb, rep, "C"), "pick_E": pick_of(S, tb, rep, "E"),
           "pick_O": pick_of(S, tb, rep, "O"), "ceilings": dict(zip(S.TRAIN5, rep["ceil"].tolist()))}
    got["carried_pick"] = (cw["pick"].rsplit("|", 3)[0] if rep["carried"] == "U2p"
                           else got[f"pick_{rep['carried']}"]["name"])
    diff = sorted(k for k in got if got[k] != vs[k])
    if diff:
        raise SystemExit(f"U2s's verdict is not reproduced from these tables: {diff}")
    return got


def reproduce_only(base=ROOT):
    """Section 8's last check, run before select's stages: the six filed tables give U2s's filed verdict."""
    t0 = time.time()
    ns = load()
    fs = filed_six(ns, base)
    six = tuple("webqsp_s1eval" if d == DS else d for d in ns.S.SIX)
    on_walk(fs, six)
    tb = tables(ns, fs, six)
    log(f"reproduce: tables of {tb.nu} universal and {tb.no} own candidates on {len(six)} carves "
        f"({time.time() - t0:.0f}s)")
    got = reproduce(ns, fs, tb)
    log(f"reproduce: U2s's verdict reproduced ({got['verdict_c']}, {got['verdict_e']}, {got['verdict_o']}; carried "
        f"{got['carried']}: {got['carried_pick']}), cap {fs.cap5} ({time.time() - t0:.0f}s)")
    return 0


def verdict(base=ROOT):
    import numpy as np
    t0 = time.time()
    ns = load()
    S, M, F, U2C = ns.S, ns.M, ns.F, ns.U2C
    led = read_ledger(base)
    bad = [s for s in STAGES if not entry_ok(led, s, base)]
    if bad:
        raise SystemExit(f"INCOMPLETE: the ledger holds no OK entry for {bad}")
    ck = WQ.read_json(base / CHECK)
    if ck.get("verdict") != "REPRODUCED":
        raise SystemExit("zwqp: the check is not REPRODUCED")
    fs = filed_six(ns, base)
    vs, am, rm, vp_sha, cap5 = fs.vs, fs.am, fs.rm, fs.vp_sha, fs.cap5
    with on_wqp(ns, base=base):
        x = S.load_u2s(DS)
        if x is None:
            raise SystemExit(f"{DS}/{CARVE}: no U2s stats under {OUT}")
        am[DS], rm[DS] = x
        if rm[DS]["file_sha256"] != led["stages"]["u2s"]["file_sha256"] or rm[DS]["carve"] != CARVE:
            raise SystemExit(f"{DS}/{CARVE}: U2s's stats are not the file WQP's ledger filed")
        if not np.array_equal(am[DS]["ptr"], M.load_u2m(DS)[0]["ptr"]):
            raise SystemExit(f"{DS}/{CARVE}: U2s's questions and golds are not U2m's")
        xc = U2C.load_stats(DS, "today")
        if xc is None or xc[1]["file_sha256"] != led["stages"]["u2c"]["file_sha256"]:
            raise SystemExit(f"{DS}/{CARVE}: U2c's stats are not the file WQP's ledger filed")
    on_walk(fs, KEYS)

    # the cap: U2c's chosen pool's cost, restated on S (section 3); under U2s's set it is 1,953.5 (filed_six)
    by = {g[0]: g for g in U2C.grid_candidates()}
    chosen_sel = U2C.table_of({DS: xc[0]}, [by[CHOSEN]])[CHOSEN][DS]
    m = WQ.read_json(base / P1C2_WEBQSP)["files"]
    w = {d: int(fs.w5[d]) for d in PASSAGE_SEL}
    w[DS] = WQ.FIT_N + int(m[f"{DS}__s1eval"]["questions"])
    chosen_rows = {d: float(fs.fc[d]["rows"]) for d in PASSAGE_SEL}
    chosen_rows[DS] = float(chosen_sel["rows"])
    cap = round(F.cost({d: chosen_rows[d] for d in SEL}, w), 1)
    log(f"WQP verdict: cap {cap} on S (U2s's {cap5}); webqsp weight {w[DS]}; chosen pool on select {chosen_sel} "
        f"({time.time() - t0:.0f}s)")

    tb = tables(ns, fs, KEYS)
    tabs, own, gr, orr, names_o, pos_o, name_u = tb.tabs, tb.own, tb.gr, tb.orr, tb.names_o, tb.pos_o, tb.name_u
    nu, no_ = tb.nu, tb.no
    log(f"WQP verdict: tables of {nu} universal and {no_} own candidates on {len(KEYS)} carves "
        f"({time.time() - t0:.0f}s)")

    def pick_rec(res, key):
        return pick_of(S, tb, res, key)

    def read_at(i, own_):
        tb_ = own if own_ else tabs
        return {k: {"ALL": float(tb_[k][0][i]), "rows": float(tb_[k][1][i])} for k in KEYS}

    # section 4: U2s's verdict reproduced from these tables, under its set, weights, cap and incumbent
    got = reproduce(ns, fs, tb)
    log(f"WQP verdict: U2s's verdict reproduced ({vs['verdict_c']}, {vs['verdict_e']}, {vs['verdict_o']}; "
        f"{vs['carried_pick']}) ({time.time() - t0:.0f}s)")

    # section 3: the choice on S
    inc_name = vs["carried_pick"]
    i_inc = pos_o[inc_name]
    res = choose(S, F, SEL, w, cap, i_inc, tabs, own, gr, orr)
    vc, ve, vo = res["verdicts"]
    car = carried_key(res["carried"])
    picks = {k: pick_rec(res, k) for k in ("C", "E", "O")}
    carried_name = inc_name if car == "U2s" else picks[car]["name"]
    car_i, car_own = (i_inc, True) if car == "U2s" else (res["index"][car], car == "O")

    def points(r_, key):
        own_ = key == "O"
        cr = r_["cr_o"] if own_ else r_["cost_r"]
        out = []
        for t, tg, i in r_["tf"][key][0]:
            p = {"level": t, "targets": dict(zip(r_["sel"], tg.tolist()))}
            if i is None:
                out.append({**p, "pick": None})
            else:
                out.append({**p, "pick": names_o[i] if own_ else name_u(i), "cost": float(cr[i]),
                            "within_cap": bool(cr[i] <= r_["cap"] + 1e-9), "read": read_at(i, own_)})
        return out

    # section 4, beside: leave one out, each d's ceiling and fewest rows, the price
    loo = {}
    for sp, d_out in LOO.items():
        sel_l = tuple(d for d in SEL if d != d_out)
        w_l = {d: w[d] for d in sel_l}
        cap_l = round(F.cost({d: chosen_rows[d] for d in sel_l}, w_l), 1)
        r_l = choose(S, F, sel_l, w_l, cap_l, i_inc, tabs, own, gr, orr)
        c_l = carried_key(r_l["carried"])
        p_l = {k: pick_rec(r_l, k) for k in ("C", "E", "O")}
        loo[sp] = {"left_out": d_out, "selection": list(sel_l), "weights": w_l, "cap": cap_l,
                   "ceilings": dict(zip(sel_l, r_l["ceil"].tolist())),
                   "incumbent": {"level": r_l["l_inc"], "cost": r_l["c_inc"]},
                   "verdict_c": r_l["verdicts"][0], "verdict_e": r_l["verdicts"][1], "verdict_o": r_l["verdicts"][2],
                   "carried": c_l, "carried_pick": inc_name if c_l == "U2s" else p_l[c_l]["name"],
                   "pick_C": p_l["C"], "pick_E": p_l["E"], "pick_O": p_l["O"]}
    tol = ns.PG.TOL
    per_dataset = {}
    for k in KEYS:
        al, rw = tabs[k]
        cd = float(al.max())
        rows_at = {}
        for t in S.LEVELS:
            tg = round(min(t, cd - tol), 4)
            i = np.flatnonzero(al >= tg - 1e-9)
            k_ = int(i[np.argmin(rw[i])])
            rows_at[str(t)] = {"target": tg, "rows": float(rw[k_]), "ALL": float(al[k_]), "candidate": name_u(k_)}
        per_dataset[k] = {"ceiling": cd, "fewest_rows": rows_at}
    nxt = ("U2n's zrckpt refit on " + ("U2s's pick" if car == "U2s" else "the pick WQP carries") +
           ", under WQ, in its own file; " +
           ("the upstream choices stand; under WQ, U2s section 10's other next forms: RRF5T's agreement as the "
            "budget, and P1h's cut over the walk's mass" if car == "U2s" else
            "the upstream choices chosen again under WQ, each in its own file: U2m (c) and (e), (b), (a), U2p's beta"))
    rec = {"declared_in": DECLARED, "stage": STAGE, "script_sha256": WQ.sha_src(__file__),
           "zu2s_sha256": S.script_sha256(), "zu2c_sha256": U2C.script_sha256(),
           "u2s_verdict_sha256": WQ.sha_file(base / U2S_VERDICT), "u2c_verdict_sha256": WQ.sha_file(base / U2C_VERDICT),
           "verdict_p_sha256": vp_sha, "ledger_sha256": WQ.sha_file(base / LEDGER),
           "check_sha256": WQ.sha_file(base / CHECK), "check": ck["verdict"],
           "verdict_c": vc, "verdict_e": ve, "verdict_o": vo, "carried": car, "carried_pick": carried_name,
           "next": nxt, "selection": list(SEL), "weights": w,
           "cap": {"value": cap, "chosen": CHOSEN, "rows": chosen_rows, "u2s_cap": cap5},
           "incumbent": {"name": inc_name, "level": res["l_inc"], "level_used": NO_LEVEL if res["l_inc"] is None
                         else res["l_inc"], "cost": res["c_inc"], "within_cap": bool(res["c_inc"] <= cap + 1e-9)},
           "pick_C": picks["C"], "pick_E": picks["E"], "pick_O": picks["O"],
           "ceilings": dict(zip(SEL, res["ceil"].tolist())), "levels": list(S.LEVELS),
           "candidates": {"universal": nu, "own": no_, "budgets": S.NB},
           "frontier_C": points(res, "C"), "frontier_E": points(res, "E"), "frontier_O": points(res, "O"),
           "u2s_reproduced": {k: got[k] for k in ("verdict_c", "verdict_e", "verdict_o", "carried", "carried_pick")},
           "picks_read": {k: read_at(res["index"][k], k == "O") for k in ("C", "E", "O")
                          if res["index"][k] is not None},
           "price": {"carried": read_at(car_i, car_own), "incumbent": read_at(i_inc, True)},
           "leave_one_out": loo, "per_dataset": per_dataset,
           "stats": {k: {"file_sha256": rm[k]["file_sha256"], "carve": rm[k]["carve"], "seconds": rm[k]["seconds"]}
                     for k in KEYS},
           "ledger_seconds": {s: led["stages"][s]["seconds"] for s in STAGES},
           "seconds": round(time.time() - t0, 1), "utc": WQ.utc()}
    WQ.write(base / VERDICT, rec, report_md(rec))
    log(f"WQP: {vc}, {ve}, {vo}; carried {car}: {carried_name} ({time.time() - t0:.0f}s)")
    return 0


def report_md(rec):
    def pk(p):
        if p["name"] is None:
            return "none within the cap"
        return f"`{p['name']}` ({p['form']}) at level {p['level']}, cost {p['cost']}"
    inc = rec["incumbent"]
    L = [f"# WQP: {rec['verdict_c']}, {rec['verdict_e']}, {rec['verdict_o']}", "",
         f"Declared in {rec['declared_in']}. Carried forward: **{rec['carried']}**, `{rec['carried_pick']}`.", "",
         f"- Selection set S: {', '.join(rec['selection'])} (webqsp on its select carve, the others on s1sel); "
         f"weights {rec['weights']}.",
         f"- Cap: {rec['cap']['value']} (U2c's chosen pool, `{rec['cap']['chosen']}`, on S; "
         f"{rec['cap']['u2s_cap']} under U2s's set).",
         f"- Incumbent, U2s's pick: `{inc['name']}` at level {inc['level']} on S, cost {inc['cost']} "
         f"({'within' if inc['within_cap'] else 'over'} the cap).",
         f"- Pick C (kind_t, universal): {pk(rec['pick_C'])}.",
         f"- Pick E (kind-free, universal): {pk(rec['pick_E'])}.",
         f"- Pick O (the own construction, kind_t): {pk(rec['pick_O'])}.",
         f"- U2s's verdict reproduced from the same tables: {rec['u2s_reproduced']['verdict_c']}, "
         f"{rec['u2s_reproduced']['verdict_e']}, {rec['u2s_reproduced']['verdict_o']}; "
         f"`{rec['u2s_reproduced']['carried_pick']}`. The check (section 2): {rec['check']}.",
         f"- Next: {rec['next']}.", "",
         "## Leave one out (each split's own training sets)", "",
         "| split | left out | cap | (c) | (e) | (o) | carried |", "|---|---|---|---|---|---|---|"]
    for sp, e in rec["leave_one_out"].items():
        L.append(f"| {sp} | {e['left_out']} | {e['cap']} | {e['verdict_c']} | {e['verdict_e']} | {e['verdict_o']} | "
                 f"{e['carried']}: `{e['carried_pick']}` |")
    lv = [str(t) for t in rec["levels"]]
    L += ["", "## Each carve's ceiling and fewest rows by level", "",
          "| carve | ceiling | " + " | ".join(lv) + " |", "|---|---|" + "---|" * len(lv)]
    for k, e in rec["per_dataset"].items():
        L.append(f"| {k} | {e['ceiling']} | " + " | ".join(f"{e['fewest_rows'][t]['rows']}" for t in lv) + " |")
    L += ["", "## The price: the carried pick against the incumbent (ALL, rows)", "",
          "| carve | carried | incumbent |", "|---|---|---|"]
    for k in rec["price"]["carried"]:
        c, i = rec["price"]["carried"][k], rec["price"]["incumbent"][k]
        L.append(f"| {k} | {c['ALL']}, {c['rows']} | {i['ALL']}, {i['rows']} |")
    return L


# ── self-test (section 8) ────────────────────────────────────────────────────


def _toy_ns(calls, shared):
    SN = types.SimpleNamespace

    class Op:
        def __init__(self):
            self.carves = {"per_dataset": {DS: {"eval": {"name": "s1eval"}}, "squad": {}}}

    def rc(op, name, wanted, arms, threads, sink):
        calls.append((name, set(wanted)))
    return SN(S4=SN(Opened=Op), D4=SN(POOLS="d4", MANIFEST="d4m"), E=SN(POOLS="e", MANIFEST="em", run_carves=rc),
              D=SN(OUT="d"), K=SN(OUT="k"), F=SN(OUT="f", P1E="fp"), PG=SN(OUT="pg"), U2C=SN(OUT="c"),
              U2E=SN(OUT="e2"), M=SN(OUT="m"),
              MB=SN(OUT="mb", carried=lambda: (shared["ce"], "C", [], ["p"])),
              A=SN(OUT="a", carried_family=lambda *a, **k: (shared["ce"], ["p"], 0, shared["b"])),
              P=SN(carried=lambda: (shared["ce"], ["p"], 0, shared["b"], shared["a"], 0)), S=SN(OUT="s"),
              C4=SN(carves_of=lambda n: ("s1eval",) if n == DS else ("fit", "select", "s1sel", "s1eval")),
              D5=SN(carves_of=lambda n: ("s1eval",) if n == DS else ("s1sel",)))


def _state(ns):
    return {k: dict(vars(getattr(ns, k))) for k in vars(ns)}


def _selftest_patches(base):
    calls, shared = [], {"ce": {"stats": {DS: {"file_sha256": "old"}, "squad": {"file_sha256": "sq"}}},
                         "b": {"stats": {DS: {"file_sha256": "oldb"}}}, "a": {"stats": {DS: {"file_sha256": "olda"}}}}
    ns = _toy_ns(calls, shared)
    before = _state(ns)
    with on_wqp(ns, base=base):
        o = base / OUT
        assert (ns.D4.POOLS, ns.D4.MANIFEST) == (o / "step4d/pools", o / "step4d/pools/pools.json")
        assert (ns.E.POOLS, ns.E.MANIFEST) == (o / "step4e/pools", o / "step4e/pools/pools.json")
        assert (ns.D.OUT, ns.K.OUT, ns.F.OUT, ns.F.P1E, ns.PG.OUT) == (o / "p1d", o / "p1e", o / "p1f", o / "p1e",
                                                                       o / "p1g")
        assert (ns.U2C.OUT, ns.U2E.OUT, ns.M.OUT, ns.MB.OUT, ns.A.OUT, ns.S.OUT) == (
            o / "u2c", o / "u2e", o / "u2m", o / "u2m/b", o / "u2m/a", o / "u2s")
        assert ns.C4.carves_of(DS) == ("select",) and ns.C4.carves_of("squad") == ("fit", "select", "s1sel", "s1eval")
        assert ns.D5.carves_of(DS) == ("select",) and ns.D5.carves_of("squad") == ("s1sel",)
        ns.E.run_carves(None, DS, {"s1eval"}, (), 2, None)
        ns.E.run_carves(None, "squad", {"s1eval", "s1sel"}, (), 2, None)
        assert calls == [(DS, {"select"}), ("squad", {"s1eval", "s1sel"})], calls
        op = ns.S4.Opened()
        assert op.carves["per_dataset"][DS][KEY] == {"name": "select", "n": SELECT_N, "sha256": SELECT_SHA}
        assert KEY not in op.carves["per_dataset"]["squad"]
        assert ns.MB.carried is not before["MB"]["carried"] and ns.P.carried is not before["P"]["carried"]
    assert _state(ns) == before, "a switch or root was not restored"
    try:
        with on_wqp(ns, base=base):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert _state(ns) == before, "a switch or root was not restored after a failure"
    with on_wqp(ns, check=True, base=base):
        c = base / CHECK_OUT
        assert ns.D.OUT == c / "p1d" and ns.S.OUT == c / "u2s" and ns.F.P1E == c / "p1e"
        for k, a in (("C4", "carves_of"), ("D5", "carves_of"), ("E", "run_carves"), ("MB", "carried"),
                     ("A", "carried_family"), ("P", "carried")):
            assert getattr(getattr(ns, k), a) is before[k][a], f"check mode switched {k}.{a}"
        assert ns.S4.Opened().carves["per_dataset"][DS][KEY]["name"] == "select"
    assert _state(ns) == before

    class Op2(before["S4"]["Opened"]):
        def __init__(self):
            super().__init__()
            self.carves["per_dataset"][DS][KEY] = {"name": "select", "n": 1, "sha256": "0" * 64}
    assert "not M3B's select" in WQ.must_stop(with_select(Op2))
    log("selftest: the patches set every switch and root inside the block, restore them after (on a failure "
        "too), switch no carve in check mode, and refuse another select entry")
    return ns, shared


def _selftest_repoint(base, ns, shared):
    import numpy as np
    sd = base / OUT / "u2m" / "stats"
    sd.mkdir(parents=True, exist_ok=True)
    npz = sd / f"{DS}__{CARVE}.npz"
    np.savez(npz, x=np.arange(3))
    fsha = WQ.sha_file(npz)
    rp = sd / f"{DS}__{CARVE}.json"
    WQ.write_json(rp, {"dataset": DS, "carve": CARVE, "file_sha256": fsha})
    file_entry("u2m", {"status": "OK", "carve": CARVE, "record": rel(rp, base), "record_sha256": WQ.sha_file(rp),
                       "file": rel(npz, base), "file_sha256": fsha}, base)
    with on_wqp(ns, base=base):
        r = ns.MB.carried()
        assert r[0]["stats"][DS]["file_sha256"] == fsha and r[0]["stats"]["squad"]["file_sha256"] == "sq"
        assert r[1:] == ("C", [], ["p"])
        assert shared["ce"]["stats"][DS]["file_sha256"] == "old", "the filed verdict's object was changed"
        assert "no OK select entry for u2m_b" in WQ.must_stop(ns.A.carried_family)
        assert "no OK select entry for u2m_a" in WQ.must_stop(ns.P.carried)
        WQ.write_json(rp, {"dataset": DS, "carve": CARVE, "file_sha256": "f" * 64})
        assert "is not the record the ledger filed" in WQ.must_stop(ns.MB.carried)
    WQ.write_json(rp, {"dataset": DS, "carve": CARVE, "file_sha256": fsha})
    file_entry("u2m", {"status": "OK", "carve": CARVE, "record": rel(rp, base), "record_sha256": WQ.sha_file(rp),
                       "file": rel(npz, base), "file_sha256": fsha}, base)
    assert entry_ok(read_ledger(base), "u2m", base) and len(read_ledger(base)["earlier"]) == 1
    log("selftest: the re-pointed checks change webqsp's file sha256 only, take it from the ledger, and stop on a "
        "record that is not the ledger's or a stage the ledger does not hold")


def _selftest_checks(base):
    import numpy as np
    root = base / OUT
    sd = root / "u2e" / "stats"
    sd.mkdir(parents=True, exist_ok=True)
    npz = sd / f"{DS}__{CARVE}.npz"
    np.savez(npz, x=np.arange(4))
    good = {"dataset": DS, "carve": CARVE, "questions": SELECT_N, "file_sha256": WQ.sha_file(npz),
            "freeze_RECORD_SHA256": FREEZE, "checks": {"a": True, "rows_served_verified": False}}
    rp = sd / f"{DS}__{CARVE}.json"

    def probs_with(**kw):
        WQ.write_json(rp, {**good, **kw})
        return problems("u2e", root, CARVE, SELECT_N, SELECT_SHA, base)
    assert probs_with() == [], probs_with()
    assert any("questions" in p for p in probs_with(questions=SELECT_N - 1))
    assert any("freeze" in p for p in probs_with(freeze_RECORD_SHA256="0" * 64))
    assert any("checks ['a']" in p for p in probs_with(checks={"a": False, "rows_served_verified": False}))
    assert any("not its record's sha256" in p for p in probs_with(file_sha256="0" * 64))
    assert any("carve" in p for p in probs_with(carve="s1eval"))
    pd_ = root / "p1d"
    pd_.mkdir(parents=True, exist_ok=True)
    pos = pd_ / f"pos_{DS}__{CARVE}.npz"
    np.savez(pos, x=np.arange(5))
    part = {"freeze_RECORD_SHA256": FREEZE, "limit": 0,
            "carves": {f"{DS}__{CARVE}": {"questions": SELECT_N, "carve_sha256": SELECT_SHA, "checks": "a string",
                                          "positions_npz_sha256": WQ.sha_file(pos)}}}
    WQ.write_json(pd_ / f"pools_{DS}.json", part)
    assert problems("p1d", root, CARVE, SELECT_N, SELECT_SHA, base) == []
    part["carves"][f"{DS}__{CARVE}"]["carve_sha256"] = CHECK_SHA
    WQ.write_json(pd_ / f"pools_{DS}.json", part)
    assert any("carve_sha256" in p for p in problems("p1d", root, CARVE, SELECT_N, SELECT_SHA, base))
    part["carves"][f"{DS}__{CARVE}"]["carve_sha256"], part["limit"] = SELECT_SHA, 5
    WQ.write_json(pd_ / f"pools_{DS}.json", part)
    assert any("--limit" in p for p in problems("p1d", root, CARVE, SELECT_N, SELECT_SHA, base))
    assert problems("p1g", root, CARVE, SELECT_N, SELECT_SHA, base)[0].startswith("p1g: no webqsp/select entry")
    (base / "outputs" / "u2c").mkdir(parents=True, exist_ok=True)
    (base / "outputs" / "u2c" / "x.json").write_text("{}", encoding="utf-8")
    s0 = snapshot(base)
    (root / "new.json").write_text("{}", encoding="utf-8")
    (base / "outputs" / "u2c" / "__pycache__").mkdir()
    (base / "outputs" / "u2c" / "__pycache__" / "y.pyc").write_bytes(b"0")
    assert changed(s0, snapshot(base)) == [], "a write under outputs/wqp or __pycache__ was flagged"
    (base / "outputs" / "u2c" / "x.json").write_text('{"a": 1}', encoding="utf-8")
    (base / "outputs" / "d5" / "z").mkdir(parents=True)
    (base / "outputs" / "d5" / "z" / "n.json").write_text("{}", encoding="utf-8")
    assert changed(s0, snapshot(base)) == ["outputs/d5/z/n.json", "outputs/u2c/x.json"], changed(s0, snapshot(base))
    for p in PLACED:
        (base / "outputs" / p).parent.mkdir(parents=True, exist_ok=True)
        (base / "outputs" / p).write_text(f'{{"v": "{p}"}}', encoding="utf-8")
    assert set(place_verdicts(base).values()) == {"copied"} and set(place_verdicts(base).values()) == {"present"}
    (base / OUT / PLACED[0]).write_text("{}", encoding="utf-8")
    assert "is present and is not" in WQ.must_stop(place_verdicts, base)
    log("selftest: the after-stage checks refuse a wrong question count, digest, carve, freeze, file, failed check "
        "or --limit run, and a write outside outputs/wqp; the placed verdicts are the filed bytes")


def _selftest_choice():
    import numpy as np
    import p1f_pool_size as F
    import zu2s as S
    kf = S.KIND_FREE[0]
    rows_u = np.array([100.0, 300.0, 1000.0])
    tab_u = {"x": (np.array([0.96, 0.96, 0.99]), rows_u), "y": (np.array([0.96, 0.97, 0.99]), rows_u)}
    gr = np.array([0, kf, 0])
    rows_o = np.array([50.0, 80.0, 10.0])
    tab_o = {"x": (np.array([0.955, 0.96, 0.90]), rows_o), "y": (np.array([0.955, 0.96, 0.90]), rows_o)}
    orr = np.array([0, 0, 0])
    w = {"x": 1, "y": 3}

    def run(cap, inc, sel=("x", "y")):
        r = choose(S, F, sel, {d: w[d] for d in sel}, cap, inc, tab_u, tab_o, gr, orr)
        return r, r["verdicts"] + (carried_key(r["carried"]),)
    r, v = run(500.0, 1)
    assert list(r["ceil"]) == [0.99, 0.99] and (r["l_inc"], r["c_inc"]) == (0.96, 80.0)
    assert (r["level"], r["index"]) == ({"C": 0.96, "E": 0.96, "O": 0.96}, {"C": 0, "E": 1, "O": 1}), r["level"]
    tg = dict((t, list(g)) for t, g, _i in r["tf"]["C"][0])
    assert tg[0.96] == [0.96, 0.96] and tg[0.99] == [0.985, 0.985] and tg[1.0] == [0.985, 0.985]
    assert v == ("UNIFIED_HOLDS", "KIND_FREE_NO_PRICE", "NO_GAIN", "E"), v
    assert run(500.0, 0)[1] == ("UNIFIED_RAISED", "KIND_FREE_NO_PRICE", "OWN_RAISED", "E")
    assert run(150.0, 1)[1] == ("UNIFIED_HOLDS", "KIND_FREE_TOO_DEAR", "NO_GAIN", "C")
    assert run(90.0, 1)[1] == ("NOT_UNIFIED", "KIND_FREE_TOO_DEAR", "NO_GAIN", "U2s")
    assert run(90.0, 0)[1] == ("NOT_UNIFIED", "KIND_FREE_TOO_DEAR", "OWN_RAISED", "O")
    r, v = run(60.0, 1)                         # the incumbent over the cap stays the bar
    assert r["c_inc"] > r["cap"] and r["level"]["O"] == 0.955 and v[2:] == ("NO_GAIN", "U2s"), v
    r, v = run(90.0, 2)                         # an incumbent that meets no level: every pick with a level beats it
    assert r["l_inc"] is None and v[2:] == ("OWN_RAISED", "O"), v
    assert round(F.cost({"x": 100.0, "y": 200.0}, w), 1) == 175.0 and round(F.cost({"y": 200.0}, {"y": 3}), 1) == 200.0
    r_l, v_l = run(500.0, 1, sel=("y",))        # leave one out: x out, E's pick rises to 0.97
    assert list(r_l["ceil"]) == [0.99] and r_l["level"]["E"] == 0.97 and v_l == ("UNIFIED_HOLDS",
                                                                                   "KIND_FREE_NO_PRICE", "NO_GAIN", "E")
    assert NO_LEVEL < min(S.LEVELS) - S.PG.TOL
    log("selftest: the choice gives the levels, targets, verdicts and carried pick in each case, keeps an incumbent "
        "over the cap as the bar, beats one with no level, restates the cap, and leaves one out")


def selftest():
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        ns, shared = _selftest_patches(base)
        _selftest_repoint(base, ns, shared)
        _selftest_checks(base)
    _selftest_choice()
    ns = load()
    assert ns.E.C4 is ns.C4 and ns.E.S4 is ns.S4 and ns.S4B.S4 is ns.S4 and ns.K.D is ns.D
    for s in CHECKED:
        assert (ROOT / FILED[s]).is_file() and rel(record_of(s, ROOT / "outputs", CHECK_CARVE)[2]) == FILED[s]
    for p in PLACED:
        assert (ROOT / "outputs" / p).is_file()
    assert WQ.read_json(ROOT / U2S_VERDICT)["carried"] == "O" and WQ.read_json(ROOT / U2C_VERDICT)["chosen"] == CHOSEN
    log("selftest: the chain's modules load once each; the filed check arrays, verdicts and picks are where "
        "section 1 names them")
    log("selftest passed")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(prog="zwqp.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("chain", "stage", "compare-check", "reproduce", "verdict"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--threads", type=int, default=THREADS)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--retry", action="store_true")
    a = ap.parse_args(argv)
    os.chdir(ROOT)
    if a.cmd == "stage":
        if a.name not in STAGES:
            ap.error(f"stage takes one of {STAGES}")
        return stage(a.name, a.threads, a.check)
    if a.cmd == "chain":
        return chain(a.threads, a.retry)
    if a.cmd == "compare-check":
        return compare_check()
    if a.cmd == "reproduce":
        return reproduce_only()
    return verdict()


if __name__ == "__main__":
    sys.exit(main())
