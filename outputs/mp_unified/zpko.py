"""PKO (docs/PKO_PICK_O_POOLS_UNDER_WQ.md): U2s's carried pick O built under WQ on the eleven carves the WQ fits read,
and the chain those fits need: the look, the lean cache, rmatch and zrc on the chained carves, the links, the typed
arrays on the chained carves and the KT input. Pools only: nothing trains, nothing is fit, nothing reads a gold, and no
recall, R@5 or ceiling of the new pools is computed or printed. Committed modules are imported unchanged; what differs
is patched in process.

    python outputs/mp_unified/zpko.py plan [--carves D/C,...] [--sq S] [--look-rows N] [--part-rows N]
    python outputs/mp_unified/zpko.py check DATASET [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zpko.py check-merge DATASET --parts K
    python outputs/mp_unified/zpko.py pools DATASET --carve C [--check] [--threads 2] [--limit Q] [--shard I/K]
    python outputs/mp_unified/zpko.py merge DATASET --carve C --parts K
    python outputs/mp_unified/zpko.py check-squad --carve C [--dry]
    python outputs/mp_unified/zpko.py copy-bases
    python outputs/mp_unified/zpko.py look DATASET --carve C --shard I/K
    python outputs/mp_unified/zpko.py cache DATASET --carve C --part I/K
    python outputs/mp_unified/zpko.py build {rmatch,zrc,links} DATASET --carve C
    python outputs/mp_unified/zpko.py typed DATASET --carve C
    python outputs/mp_unified/zpko.py kt DATASET --carve C
    python outputs/mp_unified/zpko.py record DATASET --carve C
    python outputs/mp_unified/zpko.py prune --carve D/C
    python outputs/mp_unified/zpko.py gate
    python outputs/mp_unified/zpko.py chain [--carves D/C,...] [--lanes N] [--prune] [--retry] [--threads 2]
                                            [--look-rows N] [--part-rows N] [--ram-wait S] [--no-preflight]
                                            [--as-given] [--checks-filed] [--disk-reserve-gib G]
    python outputs/mp_unified/zpko.py --selftest

plan         Per carve: questions, rows per question (U2s's filed cost at pick O on the dataset's U2s carve, read without
             its gold arrays; exact once the carve is merged), bytes and hours per step (section 9).
check        Section 5 (a) on U2s's carve of DATASET (s1sel; webqsp's s1eval): every question's |B|, |I|, |I \\ B|, fill,
             the 14 order lengths, U2e's union sizes, n1, r0, the second link, the walk-mass counts, k_q and the kept
             rows against U2s's filed arrays. --limit Q: the first Q questions, nothing written, seconds per question.
pools        Section 2's rule on one WQ carve's shard (--check on webqsp's s1eval: section 5 (a) on every question).
merge        The shards joined: outputs/pko/pools/D__C.npz, its KT rows and its manifest.
check-squad  Section 5 (b); --dry reads U2d new's squad records only.
look, cache, build, typed, kt, record   Section 4's steps; prune and chain: section 6; gate: section 5 (c).
"""
from __future__ import annotations

import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    if __name__ == "__main__":
        os.environ[_v] = "2"                             # section 4: BLAS at 2 in every PKO process
    else:
        os.environ.setdefault(_v, "2")
if __name__ == "__main__":
    os.environ.setdefault("NUMBA_NUM_THREADS", "2")      # numba 2 per lane (the chain sets it for every step)
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

try:
    import psutil  # noqa: E402  (A1: the peak record and the RAM preflight; nothing else uses it)
except ImportError:
    psutil = None

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zwq as WQ  # noqa: E402  (WQ's carves, patch, place and webqsp's fit opening, unchanged)
import zu2d_pools as ZP  # noqa: E402  (U2d's carve opening, KT orders, saves, cache order, step-1 check, unchanged)
import zu2s as S  # noqa: E402  (U2s's budgets, walk-mass counts and script sha, unchanged)
import p1gb_pools as PGB  # noqa: E402  (the pools file's arrays, unchanged)
import p1c3_kt as KT3  # noqa: E402  (P1c3's KT columns and the cache's rows, unchanged)

M, MB, A, AT, P, Q = S.M, S.MB, S.A, S.AT, S.P, S.Q
U2E, F, P1H, D5, D5B = S.U2E, S.F, S.P1H, S.D5, S.D5B
utc, sha_file, read_json, write_json = WQ.utc, WQ.sha_file, WQ.read_json, WQ.write_json


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zpko: {msg}", flush=True)

DECLARED = "docs/PKO_PICK_O_POOLS_UNDER_WQ.md"
STAGE = "PKO"
PICK = "own/kind_t/kt_cos:ada_f16_k0@0.85"
CONSTRUCTION = "PKO:" + PICK
PICK_PARSED = ("own", "kind_t", "kt_cos", "ada_f16_k0", "ADA", (16, 0), 0.85)
OUT = ROOT / "outputs" / "pko"
U2S_OUT = ROOT / "outputs" / "u2s"
U2D_NEW = ROOT / "outputs" / "u2d" / "new"
STEP1 = ROOT / "outputs" / "step1" / "cache"
STEP1_FIT = ROOT / "outputs" / "wqk" / "s1cache"            # WQK's step-1 cache, webqsp's fit carve
FIT = ("webqsp", "fit")


def _pair(x):
    d, c = x.split("/") if isinstance(x, str) else x
    return str(d), str(c)


CARVES = tuple(_pair(x) for x in WQ.SCREEN)
CHAINED = tuple(_pair(x) for x in WQ.CHAINED)
QUESTIONS = {("webqsp", "fit"): 1202, ("squad", "fit"): 5856, ("musique", "s1fit"): 4601, ("hotpotqa", "fit"): 5930,
             ("2wiki", "fit"): 5928, ("metaqa", "s1eval"): 9785, ("squad", "s1eval"): 11873,
             ("musique", "s1eval"): 2417, ("hotpotqa", "s1eval"): 7405, ("2wiki", "s1eval"): 12576,
             ("webqsp", "s1eval"): 1503}
U2S_CARVE = {"webqsp": "s1eval"}                             # U2s's carve of each dataset: s1sel, webqsp's s1eval
CHECKS = ("webqsp", "hotpotqa")                              # section 5 (a)
SQUAD = tuple(c for c in CARVES if c[0] == "squad")          # section 5 (b)
U2S_KEYS = ("bs", "inc", "n_i", "fill", "lens", "n_u", "n_n1", "n_r0", "n_n2", "taken", "hubs", "mq", "nw")
U2S_GOLD = ("pos", "in_I", "one_link", "ptr")                # U2s's gold arrays: never loaded
POOL_GOLD = ("gold_ptr", "held", "n_gold", "n_held", "golds", "pos", "in_I", "one_link")   # never written
FLAT = {"bs_nodes": "bs_ptr", "nodes": "ptr"}                # the pools file's flat rows and their pointers
SAME = ("declared_in", "stage", "dataset", "carve", "pick", "construction", "graph_detail", "walk", "threads", "u2s",
        "types", "scripts")
SHARD_Q = {"webqsp": 1000, "metaqa": 1000}   # questions per pools shard: a KB carve opens in ~2 min
SHARD_Q_PASSAGE = 3000                       # a passage carve opens in > 9 min (hotpotqa, timed)
CHUNK_NODES = 24000                                          # nodes per look chunk (mp_approx_l0's; look_x_six.py:260)
LOOK_ROWS = 1_000_000                                        # rows per look shard (WQK's peak 6.04 GiB at ~0.85M)
PART_ROWS = 1_250_000                                        # rows per cache part, A0's one value (WQK's peak 6.5 GiB at ~1.27M)
PART_ROWS_BY = {"webqsp": 500_000, "metaqa": 1_200_000, "musique": 950_000, "hotpotqa": 400_000, "2wiki": 450_000,
                "squad": PART_ROWS}                          # A1: rows per cache part by dataset, for a peak of ~3 GiB
RESERVE = 15 * 2 ** 30                                       # section 6: the disk kept free
ZPKO_KEY = "outputs/mp_unified/zpko.py"                      # A1: how the records name this file among the scripts
ZPKO_SAME = ("8629c86949b4d1fd01f0dc980c58af0419b63617426a6c521899f409beffd16b",   # its sha before A1 (A0): the same pools pass
             "5306512eaa3b8d0fd1b08551528e9216082e4978030ad202359236f83113d185")   # A1's; A2 changes only check (b)
RULE_B = "A2"                                                # check (b)'s rule (docs/PKO_A2_AMENDMENT.md); a record carries it
GIB = 2 ** 30
MARGIN_GIB = 1.0                                             # A1: kept free above a step's measured peak
LANE_GIB = {"squad": 5.0, "metaqa": 5.0, "webqsp": 7.0, "2wiki": 7.0, "hotpotqa": 7.0, "musique": 9.5}   # pools, check, look
CACHE_FLOOR_GIB = 1.6                                        # a cache part: floor + GiB per million rows (laptop fits)
CACHE_GIB_MROW = {"2wiki": 4.5, "hotpotqa": 5.8, "webqsp": 2.9, "metaqa": 1.1, "musique": 1.5, "squad": 1.5}
STEP_GIB = {"rmatch": 4.0, "zrc": 4.0, "links": 3.0, "typed": 3.0, "kt": 4.0, "record": 3.0, "check-merge": 1.0,
            "check-squad": 2.0, "copy-bases": 0.5, "gate": 1.0, "prune": 0.5}   # whole figures, margin in; GUESS until A1's ledger holds a peak
MERGE_BPR, MERGE_FLOOR_GIB = 114, 0.3                        # merge: every shard's arrays and KT loaded, then joined
RAM_WAIT_S = 7200.0                                          # A1: how long a step waits for its RAM before the chain stops
ROWS_Q = 2000.0                                              # rows per question when U2s's filed cost is not read
POOLS_BPR = 40                                               # bytes per row, pools parts and the merged file
BPR = {  # bytes per pool row per root, from WQK's and U2d's built carves (conservative)
    "webqsp": {"look": 360, "cache": 550, "rmatch": 510, "zrc": 510, "links": 70, "kt": 17, "typed": 24},
    "metaqa": {"look": 360, "cache": 270, "rmatch": 32, "zrc": 32, "links": 75, "kt": 17, "typed": 25},
    "squad": {"look": 470, "cache": 360, "links": 170, "kt": 17},
    "musique": {"look": 690, "cache": 295, "links": 390, "kt": 17},
    "hotpotqa": {"look": 371, "cache": 700, "links": 71, "kt": 17},
    "2wiki": {"look": 375, "cache": 580, "links": 75, "kt": 17},
}
RATES_US = {"look": 1170.0, "cache": 115.0, "rmatch": 162.0, "zrc": 211.0, "links": 8.0}   # WQK's laptop build
LEDGER_LOCK = threading.Lock()


# ── roots, names and shas ────────────────────────────────────────────────────

class Roots(SimpleNamespace):
    """PKO's roots, read by attribute or by key."""

    def __getitem__(self, k):
        return getattr(self, k)


def roots(base=None):
    b = Path(OUT if base is None else base)
    return Roots(base=b, out=b, pools=b / "pools", parts=b / "pools" / "parts", check=b / "check",
                 check_parts=b / "check" / "parts", look=b / "look", cache=b / "cache", rmatch=b / "chains" / "rmatch",
                 zrc=b / "chains" / "zrc", links=b / "links", typed=b / "typed", kt=b / "kt", records=b / "records",
                 logs=b / "logs", ledger=b / "ledger.json", gate=b / "gate.json")


def relr(p):
    """A path as records name it: relative to the repository when under it."""
    p = Path(p).resolve()
    try:
        return p.relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def relb(R, p):
    """A path relative to PKO's base, as the ledger and the records' inputs name it."""
    return Path(os.path.relpath(Path(p).resolve(), Path(R.base).resolve())).as_posix()


def stem(name, carve, shard=None):
    return f"{name}__{carve}" if shard is None else f"{name}__{carve}__{shard[0]}of{shard[1]}"


def by_index(paths):
    """Part paths in their index order (part_10of13 after part_9of13)."""
    def key(p):
        p = Path(p)
        m = re.search(r"(\d+)of(\d+)", p.parent.name if p.name == "record.json" else p.name)
        return (int(m.group(2)), int(m.group(1))) if m else (0, 0)
    return sorted((Path(p) for p in paths), key=key)


def src_sha(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def scripts():
    """The sha256 (LF-normalised, as zu2s names its own) of every script PKO's outputs depend on."""
    mods = (S, M, MB, A, AT, P, Q, U2E, F, P1H, D5, D5B, ZP, WQ, PGB, KT3)
    paths = {Path(m.__file__).resolve() for m in mods}
    paths |= {HERE / f for f in ("zpko.py", "look_u2d.py", "lean_cache.py", "zp1c2.py", "zu2n.py", "zwqk.py")}
    return {relr(p): src_sha(p) for p in sorted(paths) if p.exists()}


def head_key(h):
    """A shard's head as the merge compares it: this file's sha counts as one value when it is the current file's or a
    declared earlier one (A1: the pools pass is the same code; docs/PKO_A1_AMENDMENT.md section 5). Any other script's
    sha, or any other zpko.py, still makes two shards differ."""
    s = dict(h["scripts"])
    if s.get(ZPKO_KEY) in ZPKO_SAME + (src_sha(Path(__file__)),):
        s[ZPKO_KEY] = "zpko.py"
    return dict(h, scripts=s)


def u2s_carve(name):
    return U2S_CARVE.get(name, "s1sel")


def u2s_verdict():
    """U2s's verdict, checked to carry pick O as section 1 names it."""
    v = read_json(U2S_OUT / "verdict.json")
    if v.get("script_sha256") != S.script_sha256():
        raise SystemExit("PKO: U2s's verdict was not written by the zu2s.py on disk")
    if v.get("carried") != "O" or v.get("carried_pick") != PICK or v.get("pick_O", {}).get("name") != PICK:
        raise SystemExit("PKO: U2s's verdict does not carry pick O as PKO declares it")
    return v


def u2s_record(name, v=None):
    """(record, npz path) of U2s's filed arrays on the dataset's U2s carve, checked against the verdict."""
    v = u2s_verdict() if v is None else v
    base = U2S_OUT / "stats" / stem(name, u2s_carve(name))
    rec = read_json(base.with_name(base.name + ".json"))
    if rec.get("script_sha256") != S.script_sha256() or rec.get("dataset") != name or rec.get("carve") != u2s_carve(name):
        raise SystemExit(f"{name}: U2s's record is not zu2s.py's on {name}/{u2s_carve(name)}")
    if rec.get("file_sha256") != v["stats"][name]["file_sha256"]:
        raise SystemExit(f"{name}: U2s's record is not the one its verdict read")
    return rec, base.with_name(base.name + ".npz")


def u2s_arrays(name, rec, npz, keys=U2S_KEYS):
    """U2s's walk-free per-question arrays (its gold arrays are never loaded)."""
    if set(keys) & set(U2S_GOLD):
        raise SystemExit("PKO never loads U2s's gold arrays")
    if sha_file(npz) != rec["file_sha256"]:
        raise SystemExit(f"{name}: U2s's arrays are not the ones its record names")
    with np.load(npz, allow_pickle=False) as z:
        a = {k: np.asarray(z[k]) for k in keys}
    if int(a["bs"].shape[0]) != int(rec["questions"]):
        raise SystemExit(f"{name}: U2s's arrays do not hold its record's questions")
    return a


def pick_plan():
    """Pick O parsed by U2m's own parser: (cell, recipe, order, budget, tau), checked as section 2 states it."""
    cell, r, o, b, tau = MB.parse(PICK)
    bname, kind, prm = S.budgets()[b]
    got = (cell, M.RECIPES[r], M.NAMES[o], bname, kind, tuple(prm), float(tau))
    if got != PICK_PARSED:
        raise SystemExit(f"PKO: pick O parses as {got}, not as declared")
    return SimpleNamespace(r=int(r), o=int(o), b=int(b), kind=kind, prm=tuple(prm), tau=float(tau))


def kept_counts(kt_len, fill, bs_size, plan):
    """(k_q, k, kept) per question: ADA's k_q from today's fill, k = min(|kt_cos|, k_q), and P1h's cut at tau on |B|
    and k (section 2.5)."""
    kq = np.asarray(U2E.budget_k(plan.kind, plan.prm, np.asarray(fill, np.int64)), np.int64)
    k = np.minimum(np.asarray(kt_len, np.int64), kq)
    keep = np.asarray(P1H.keep_counts(np.asarray(bs_size, np.int64), k, plan.tau), np.int64)
    if not ((keep >= 0) & (keep <= k)).all():
        raise SystemExit("PKO: P1h's cut kept more than k rows")
    return kq, k, keep


# ── opening a carve and the rule (section 2) ─────────────────────────────────

@contextlib.contextmanager
def fit_ctx(name, carve):
    """WQK's opening of webqsp's fit carve (zwqk.on_wqk_build with zwq.on_wq_pools); nothing on the other ten."""
    if (name, carve) != FIT:
        yield
        return
    import zwqk as WQK
    with WQK.on_wqk_build(), WQ.on_wq_pools(False, False):
        yield


def _field(obj, k):
    try:
        return obj[k]
    except (KeyError, TypeError, IndexError, ValueError):
        return getattr(obj, k, None)


def kb_relations(seen):
    """The KB structural store's typed relations as zu2s reads them (the store's typed_rel when it has typed arrays),
    or None (section 2.1); the store is the one zu2d_pools.open_carve's open_unions gave."""
    out = seen.get("out")
    stores = out[0] if isinstance(out, tuple) and out and isinstance(out[0], dict) else None
    if stores is None or "structural" not in stores:
        raise SystemExit("PKO: open_unions did not give the KB's stores")
    st = stores["structural"]
    if getattr(st, "typed_indptr", None) is None:
        return None
    return np.asarray(st.typed_rel)


def open_pko(name, carve, threads=2, v=None):
    """(cx, ctx): the carve on U2s's graph and carried walk, every per-dataset input checked against U2s's record of
    the dataset (cells, own cell and base, graph, EM and score types, walk, alpha, eps, nmax)."""
    v = u2s_verdict() if v is None else v
    rec, npz = u2s_record(name, v)
    plan = pick_plan()
    cw = Q.carried_walk()
    if not (cw["G"] == 0 == rec["G"] and cw["pick"] == rec["pick"] and float(cw["beta"]) == float(rec["beta"])):
        raise SystemExit(f"{name}: the carried walk is not the one U2s's record names")
    passage = name in P.PASSAGE
    import p1e_anchor_kind as K           # the module zu2d_pools.open_carve and zu2s.stats_stage take G from
    seen = {}
    G = K.G
    orig = G.open_unions

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.update(out=out)
        return out

    with fit_ctx(name, carve), WQ.patch(G, open_unions=spy):
        cx = ZP.open_carve("new", name, carve, threads)
    construction = cx.op.construction[name]
    cells, own_k = M.cells_of(construction)
    n_cells = rec["cells"] if isinstance(rec["cells"], int) else len(rec["cells"])
    if len(cells) != n_cells or int(own_k) != int(rec["own_cell"]):
        raise SystemExit(f"{name}: U2m's cells or the own cell differ from U2s's record")
    if not isinstance(rec["cells"], int) and json.loads(json.dumps(cells, default=str)) != rec["cells"]:
        raise SystemExit(f"{name}: U2m's cells differ from U2s's record")
    own_b = [bp for _bs, bp in M.BASES].index(construction["base_pool"])
    if int(rec["cell_base"][own_k]) != own_b:
        raise SystemExit(f"{name}: the own construction's base is not U2s's")
    if json.loads(json.dumps(cx.desc, default=str)) != rec["graph_detail"]:
        raise SystemExit(f"{name}/{carve}: the graph is not U2s's")
    tarr, arec = AT.load_types(name, cx.desc)
    if arec["file_sha256"] != rec["em_types_sha256"]:
        raise SystemExit(f"{name}: the EM types are not U2s's")
    mu = np.asarray(tarr["mu"], np.float64)
    k_a = int(arec["null"])
    entries = int(np.asarray(cx.adj[0][0])[-1])
    if mu.shape[0] != k_a or not (int(arec["entries"]) == entries == int(np.asarray(tarr["cls"]).size)):
        raise SystemExit(f"{name}: the EM types do not fit the graph")
    beta = float(cw["beta"])
    srec = None
    if passage:
        cls_s, zbar, srec = P.load_types(name, cx.desc)
        if srec["file_sha256"] != rec["score_types_sha256"] or cls_s.size != np.asarray(tarr["cls"]).size:
            raise SystemExit(f"{name}: the score types are not U2s's")
        ty = (np.ascontiguousarray(P.pair_types(tarr["cls"], cls_s, k_a), np.int16),) + A.no_types()[1:]
        ws = P.weights(zbar, beta)
        walk = f"{cw['a_cfg']} x LW@{beta:g}"
        del cls_s
    else:
        rel = kb_relations(seen)
        if bool(arec["typed"]) != (rel is not None):
            raise SystemExit(f"{name}: the typed relations are not as U2s's types record them")
        ty = (np.ascontiguousarray(tarr["cls"], np.int8), np.ascontiguousarray(tarr["bst"], np.int64),
              np.ascontiguousarray(tarr["bln"], np.int32), np.ascontiguousarray(rel, np.int16)) \
            if rel is not None else (np.ascontiguousarray(tarr["cls"], np.int8),) + A.no_types()[1:]
        ws = None
        walk = cw["a_cfg"]
    seen.clear()
    if walk != rec["walk"]:
        raise SystemExit(f"{name}: the walk {walk} is not U2s's {rec['walk']}")
    alpha, eps, bmax = float(rec["alpha"]), float(rec["eps"]), int(rec["nmax"])
    if float(K.E.ALPHA) != alpha or float(cx.S4.EPS) != eps or int(cx.D.NMAX) != bmax:
        raise SystemExit(f"{name}: alpha, eps or nmax is not U2s's")
    base_own = cx.op.m3b_compile.base_rows({"base_pool": construction["base_pool"]}, cx.prep.dense_ids,
                                           cx.prep.splade_ids, cx.op.m3a, cx.op.m3b_contract, cx.constant)
    rjson = npz.with_name(npz.stem + ".json")
    ctx = SimpleNamespace(
        name=name, carve=carve, rec=rec, npz=npz, plan=plan, cw=cw, own_k=int(own_k), own_b=int(own_b),
        passage=passage, mu=mu, ty=ty, ws=ws, alpha=alpha, eps=eps, bmax=bmax, walk=walk, base_own=base_own,
        desc=cx.desc, u2s=lambda: u2s_arrays(name, rec, npz),
        u2s_names={"verdict": relr(U2S_OUT / "verdict.json"), "verdict_sha256": sha_file(U2S_OUT / "verdict.json"),
                   "record": relr(rjson), "record_sha256": sha_file(rjson), "file_sha256": rec["file_sha256"],
                   "freeze_RECORD_SHA256": rec.get("freeze_RECORD_SHA256")},
        types={"em_types_sha256": arec["file_sha256"], "score_types_sha256": None if srec is None else srec["file_sha256"]})
    return cx, ctx


def question_ids(cx):
    """The carve's question ids in its question order (its population's, as zu2d_pools.pools_stage names them)."""
    ids = [str(q) for q in cx.pop.ids]
    if len(ids) != len(cx.bs) or len(set(ids)) != len(ids):
        raise SystemExit("PKO: the population's ids are not one per question")
    return ids


def rule_batch(cx, ctx, js, verify=False):
    """Section 2's rule on a batch of questions: zu2s.stats_stage's calls, in its order, on the own construction and
    kind_t only (every callee is U2s's, unchanged), plus the KT input on P1e's walks."""
    K, D, prep, constant, threads = cx.K, cx.D, cx.prep, cx.constant, cx.threads
    js = list(js)
    srcs = D.sources_batch(prep, js, cx.linked, cx.unions, constant, threads)
    steered = K.steered_batch(prep, js, cx.linked, cx.unions, srcs, constant, threads)
    k0, a1 = K.restarts_batch(prep, js, cx.linked, cx.unions, srcs, constant)
    stz = [K.steer(prep, j, constant) for j in js]
    qs = [D5B.unit(np.asarray(prep.qemb[j], np.float64)) for j in js]
    wa = [A.omega(q, ctx.mu, Q.A_BETA, True)[0] for q in qs]
    wqs = [P.pair_weights(x, ctx.ws) for x in wa] if ctx.passage else wa
    wal = {w: A.rwalk(cx.unions[Q.WALK_UNION[w]], a1 if w.startswith("A1") else k0, stz if w.startswith("KT") else None,
                      wqs, 1, ctx.ty, ctx.alpha, ctx.eps, ctx.bmax, threads) for w in Q.WALKS}
    out = []
    for i, j in enumerate(js):
        src, _nn, (L, Ls_) = srcs[i]
        route = D.routes(L, Ls_)
        knd = K.kind(Ls_, prep.seeds[j], cx.nonly)
        sc = (A.scaled(wal["A1st"][i]), A.scaled(wal["A1all"][i]), A.scaled(wal["Kst"][i]), src[3], src[4])
        stc = dict(steered[i])
        stc["KTst"], stc["KTall"] = wal["KTst"][i][0], wal["KTall"][i][0]
        o1, p1 = K.p1e_orders(sc, stc, route, knd, cx.n_st, constant)
        kt = np.asarray(o1["KIND_T"], np.int64)
        if not np.array_equal(kt, np.asarray(p1["ROUTEst"] if knd == "entity" else o1["RRF5T"], np.int64)):
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: KIND_T is not its kind's order")
        seeds = np.asarray(prep.seeds[j], np.int64)
        B = np.unique(np.union1d(np.asarray(ctx.base_own[j], np.int64), seeds))
        I = np.asarray(cx.inc[j], np.int64)
        if not (np.array_equal(B, np.unique(np.asarray(cx.bs[j], np.int64))) and np.isin(seeds, B).all()
                and np.isin(B, I).all()):
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: B is not step 4d's, or not inside I")
        n1 = D5B.n1_table(cx.adj, B, D5.weights_of(stz[i], B))[0]
        rows = M.Rows(cx.nodes, verify=verify)
        rows.add(n1)
        cos1 = (D5B.unit(rows.read(n1)) @ qs[i]) if n1.size else np.zeros(0, np.float32)
        cr = D5B.ranks_desc(cos1, n1) if n1.size else S.EMPTY
        n2, first, taken, hubs = U2E.second_link(cx.adj, n1, cr, B)
        r0src = np.asarray(src[4][0], np.int64)
        rows.add(np.concatenate([kt, r0src, I, np.asarray(n2, np.int64)]))
        full = F.outside(kt, B)
        fill = int(M.fill_of(full, I.size, B.size))
        r0 = F.outside(r0src, B)
        iout = F.outside(I, B)
        ords, sizes, U, mem = U2E.question_orders(full, n1, cr, cos1, r0, iout, n2, first, rows.read, qs[i], constant,
                                                  D.rrf)
        if not U2E.orders_exact(ords, B, full, n1, U, mem):
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: an order is not exact")
        kt_cos = np.asarray(ords[M.NAMES[ctx.plan.o]], np.int64)
        kq, k, keep = (int(x[0]) for x in kept_counts([kt_cos.size], [fill], [B.size], ctx.plan))
        exp = kt_cos[:keep]
        pool = np.union1d(B, exp)
        if pool.size != B.size + exp.size or np.unique(exp).size != exp.size:
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: the expansion meets B or repeats a row")
        mass = [S.mass_counts(wal[w][i][0], wal[w][i][1], B) for w in S.MASS_WALKS]
        oe, pe = K.p1e_orders(src, steered[i], bool(route), knd, cx.n_st, constant)
        ktc = KT3.columns(ZP.kt_orders(oe, pe), pool, exp, knd == "entity").astype(ZP.KT_DT)
        if ktc.shape[0] != pool.size:
            raise SystemExit(f"{ctx.name}/{ctx.carve} question {j}: one KT row per pool row is not what P1c3 gave")
        fa = np.asarray(first, np.int64)
        out.append({"B": B, "I": I, "exp": exp, "fill": fill, "kq": kq, "k": k, "keep": keep, "kt_len": int(kt_cos.size),
                    "today_inc_size": int(cx.today_inc[j]), "entity": knd == "entity",
                    "route": bool(route), "lens": [int(np.asarray(ords[n]).size) for n in M.NAMES],
                    "n_u": [int(sizes[m]) for m in M.MS], "n_i": int(iout.size), "n_n1": int(n1.size),
                    "n_r0": int(r0.size), "n_n2": [int((fa < m).sum()) for m in M.M_GRID], "taken": int(taken),
                    "hubs": int(hubs), "mq": [int(x) for mq_, _c in mass for x in mq_],
                    "nw": [int(c) for _mq, c in mass], "kt": ktc})
    return out


# ── section 5 (a): U2s's fields ──────────────────────────────────────────────

def u2s_fields(f):
    """One question's fields as U2s files them for the own construction and kind_t, from PKO's rule."""
    return {"bs": [f["B"].size], "inc": [f["I"].size], "n_i": [f["n_i"]], "fill": [f["fill"]], "lens": f["lens"],
            "n_u": f["n_u"], "n_n1": [f["n_n1"]], "n_r0": [f["n_r0"]], "n_n2": f["n_n2"], "taken": [f["taken"]],
            "hubs": [f["hubs"]], "mq": f["mq"], "nw": f["nw"], "kq": [f["kq"]], "kept": [f["B"].size + f["keep"]]}


def u2s_filed(a, j, own_k, own_b, plan):
    """The same fields from U2s's filed arrays (zu2s's columns: cell, recipe and order; base, walk and tau_w)."""
    r, nm, ns = plan.r, len(M.MS), len(M.M_GRID)
    c, c0 = own_k * M.R + r, M.column(own_k, r, 0)
    fill, bs = int(a["fill"][j, c]), int(a["bs"][j, own_k])
    lens = [int(x) for x in a["lens"][j, c0:c0 + M.NO]]
    kq, _k, keep = (int(x[0]) for x in kept_counts([lens[plan.o]], [fill], [bs], plan))
    return {"bs": [bs], "inc": [int(a["inc"][j, own_k])], "n_i": [int(a["n_i"][j, own_k])], "fill": [fill],
            "lens": lens, "n_u": [int(x) for x in a["n_u"][j, c * nm:(c + 1) * nm]],
            "n_n1": [int(a["n_n1"][j, own_b])], "n_r0": [int(a["n_r0"][j, own_b])],
            "n_n2": [int(x) for x in a["n_n2"][j, own_b * ns:(own_b + 1) * ns]], "taken": [int(a["taken"][j, own_b])],
            "hubs": [int(a["hubs"][j, own_b])],
            "mq": [int(x) for x in a["mq"][j, S.mq_col(own_b, 0, 0):S.mq_col(own_b + 1, 0, 0)]],
            "nw": [int(x) for x in a["nw"][j, own_b * S.NW:(own_b + 1) * S.NW]], "kq": [kq], "kept": [bs + keep]}


def differ(mine, theirs):
    """The fields on which two field sets differ."""
    return [k for k in theirs if [int(x) for x in mine[k]] != [int(x) for x in theirs[k]]]


def filed_rows_q(names=None):
    """Rows per question at pick O on each dataset's U2s carve, from U2s's filed |B|, fill and kt_cos length (the cost
    U2s files; no gold array is loaded)."""
    v, plan, out = u2s_verdict(), pick_plan(), {}
    for name in names or sorted({c[0] for c in CARVES}):
        rec, npz = u2s_record(name, v)
        a = u2s_arrays(name, rec, npz, ("bs", "fill", "lens"))
        own_k = int(rec["own_cell"])
        bs = np.asarray(a["bs"][:, own_k], np.int64)
        fill = np.asarray(a["fill"][:, own_k * M.R + plan.r], np.int64)
        ktl = np.asarray(a["lens"][:, M.column(own_k, plan.r, plan.o)], np.int64)
        out[name] = float((bs + kept_counts(ktl, fill, bs, plan)[2]).mean())
    return out


# ── the pools pass, the merge and the pools file (sections 4 and 5) ─────────

def head(name, carve, ctx, threads):
    return {"declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": carve, "pick": PICK,
            "construction": CONSTRUCTION, "graph_detail": ctx.desc, "walk": ctx.walk, "threads": int(threads),
            "u2s": ctx.u2s_names, "types": ctx.types, "scripts": scripts()}


def part_arrays(ids, facts):
    """One shard's pools-file arrays (p1gb's, with today_count = fill, fill_want = k_q, fill_k = kept, x_count = 0)
    and PKO's per-question extras; no gold field."""
    n = len(facts)

    def col(key, dt):
        return np.asarray([f[key] for f in facts], dt).reshape(n)

    arrays = dict(PGB.pool_arrays(list(ids), [f["B"] for f in facts], [f["I"] for f in facts],
                                  [f["exp"] for f in facts], col("fill", np.int64), col("kq", np.int64),
                                  col("keep", np.int64), np.zeros(n, np.int64)))
    arrays.update(today_inc_size=col("today_inc_size", np.int32), budget=col("k", np.int32),
                  kt_len=col("kt_len", np.int32), entity=col("entity", bool), route=col("route", bool),
                  changed=np.asarray([not np.array_equal(np.union1d(f["B"], f["exp"]), np.unique(f["I"]))
                                      for f in facts], bool).reshape(n))
    if set(arrays) & set(POOL_GOLD):
        raise SystemExit("PKO: a gold field reached the pools file")
    return arrays


def merge_arrays(parts):
    """One array set from shards in question order: per-question arrays joined, pointers shifted, flat rows joined."""
    keys = list(parts[0])
    if any(set(p) != set(keys) for p in parts):
        raise SystemExit("PKO: the shards' arrays differ in their keys")
    out = {}
    for key in keys:
        xs = [np.asarray(p[key]) for p in parts]
        if xs[0].ndim == 0:
            if any(x.tobytes() != xs[0].tobytes() for x in xs):
                raise SystemExit(f"PKO: the shards' {key} differ")
            out[key] = xs[0]
        elif key in FLAT.values():
            ptrs, off = [], 0
            for n_, x in enumerate(xs):
                if x.size == 0 or int(x[0]) != 0:
                    raise SystemExit(f"PKO: a shard's {key} does not start at 0")
                ptrs.append(x[(0 if n_ == 0 else 1):].astype(np.int64) + off)
                off += int(x[-1])
            out[key] = np.concatenate(ptrs).astype(xs[0].dtype)
        else:
            out[key] = np.concatenate(xs)
    return out


def pool_sizes(filed):
    return np.asarray(filed["bs_size"], np.int64) + np.asarray(filed["count"], np.int64)


def pools_of(filed):
    """Each question's pool (B and the kept expansion), sorted."""
    bp, bn = np.asarray(filed["bs_ptr"], np.int64), np.asarray(filed["bs_nodes"], np.int64)
    ep, en = np.asarray(filed["ptr"], np.int64), np.asarray(filed["nodes"], np.int64)
    return [np.union1d(bn[bp[q]:bp[q + 1]], en[ep[q]:ep[q + 1]]) for q in range(bp.size - 1)]


def check_filed(filed, kt=None):
    """The pools file's own invariants (as p1gb's replace_pools reads them): pointers, B and the expansion disjoint and
    each unique, count = fill_k + x_count; with kt, one KT row per pool row."""
    bp, bn = np.asarray(filed["bs_ptr"], np.int64), np.asarray(filed["bs_nodes"], np.int64)
    ep, en = np.asarray(filed["ptr"], np.int64), np.asarray(filed["nodes"], np.int64)
    n = len(filed["ids"])
    ok = (bp.size == ep.size == n + 1 and int(bp[-1]) == bn.size and int(ep[-1]) == en.size
          and np.array_equal(np.diff(bp), np.asarray(filed["bs_size"], np.int64))
          and np.array_equal(np.diff(ep), np.asarray(filed["count"], np.int64))
          and np.array_equal(np.asarray(filed["count"], np.int64),
                             np.asarray(filed["fill_k"], np.int64) + np.asarray(filed["x_count"], np.int64)))
    for q in range(n if ok else 0):
        b, e = bn[bp[q]:bp[q + 1]], en[ep[q]:ep[q + 1]]
        if np.unique(b).size != b.size or np.unique(e).size != e.size or np.isin(e, b).any():
            ok = False
            break
    if ok and kt is not None:
        ok = int(np.asarray(kt).shape[0]) == int(pool_sizes(filed).sum())
    if not ok:
        raise SystemExit("PKO: the pools file breaks its own invariants")


def beside(filed):
    """Section 7's counts of a carve's pools (costs only; nothing recall-like)."""
    n = max(1, len(filed["ids"]))
    rows = int(pool_sizes(filed).sum())
    return {"questions": len(filed["ids"]), "rows": rows, "rows_per_question": rows / n,
            "changed": int(np.asarray(filed["changed"]).sum()),
            "short": int((np.asarray(filed["fill_k"]) < np.asarray(filed["fill_want"])).sum()),
            "budget_mean": float(np.asarray(filed["budget"], np.float64).mean()) if len(filed["ids"]) else 0.0,
            "fill_want_mean": float(np.asarray(filed["fill_want"], np.float64).mean()) if len(filed["ids"]) else 0.0,
            "today_inc_mean": float(np.asarray(filed["today_inc_size"], np.float64).mean()) if len(filed["ids"]) else 0.0}


def pools_pass(name, carve, threads=2, limit=0, shard=None, check=False, pools=True, R=None, opener=None, rule=None):
    """Section 2's rule on one shard of a carve; with check, section 5 (a) on every question it builds. --limit: the
    first Q questions of the shard, nothing written, seconds per question logged."""
    R = roots() if R is None else R
    opener = open_pko if opener is None else opener
    rule = rule_batch if rule is None else rule
    i, k = (0, 1) if shard is None else shard
    st = stem(name, carve, (i, k))
    if check and carve != u2s_carve(name):
        raise SystemExit(f"{name}/{carve} is not U2s's carve of {name}")
    outs = ([R.parts / f"{st}.json"] if pools else []) + ([R.check_parts / f"{st}.json"] if check else [])
    for p in outs if not limit else []:
        if p.exists():
            raise SystemExit(f"{relr(p)} exists: PKO never rebuilds a filed part")
    t0 = time.time()
    cx, ctx = opener(name, carve, threads)
    t_open = time.time() - t0
    nq = len(cx.bs)
    lo, hi = nq * i // k, nq * (i + 1) // k
    if limit:
        hi = min(hi, lo + int(limit))
    js = list(range(lo, hi))
    ref = ctx.u2s() if check else None
    if ref is not None and int(ref["bs"].shape[0]) != nq:
        raise SystemExit(f"{name}/{carve}: U2s's arrays do not hold the carve's questions")
    facts, bad, t1 = [], {}, time.time()
    for b0 in range(0, len(js), Q.BATCH):
        jb = js[b0:b0 + Q.BATCH]
        for j, f in zip(jb, rule(cx, ctx, jb, verify=False)):
            if ref is not None:
                for key in differ(u2s_fields(f), u2s_filed(ref, j, ctx.own_k, ctx.own_b, ctx.plan)):
                    bad.setdefault(key, []).append(j)
            facts.append(f)
        log(f"{st}: {len(facts)}/{len(js)} questions, {time.time() - t1:.1f} s")
    sec = time.time() - t1
    chk = None if not check else {
        "against": ctx.u2s_names, "questions": len(js), "fields": sorted(u2s_fields(facts[0])) if facts else [],
        "mismatches": {key: len(v) for key, v in sorted(bad.items())},
        "first": {key: v[:5] for key, v in sorted(bad.items())}, "verdict": "IDENTICAL" if not bad else "DIFFERENT"}
    if limit:
        log(f"{name}/{carve} {len(js)} questions: {sec:.1f} s, {sec / max(1, len(js)):.3f} s per question "
            f"(open {t_open:.1f} s; threads {threads}); nothing written")
        if check:
            log(f"section 5 (a) on {len(js)} questions: {chk['verdict']} {chk['mismatches'] or ''}")
            if bad:
                raise SystemExit(1)
        return {"seconds": sec, "open": t_open, "questions": len(js), "check": chk}
    hd = dict(head(name, carve, ctx, threads), shard=[i, k], range=[lo, hi], nq=nq, questions=len(js),
              seconds={"open": round(t_open, 1), "pass": round(sec, 1)}, utc=utc())
    if check:
        write_json(R.check_parts / f"{st}.json", dict(hd, check=chk))
        if bad:
            raise SystemExit(f"{st}: section 5 (a) DIFFERENT on {chk['mismatches']}")
    if not pools:
        return {"seconds": sec, "questions": len(js), "check": chk}
    arrays = part_arrays(question_ids(cx)[lo:hi], facts)
    kt = np.concatenate([f["kt"] for f in facts]) if facts else np.zeros((0, KT3.NKT), ZP.KT_DT)
    check_filed(arrays, kt)
    R.parts.mkdir(parents=True, exist_ok=True)
    sha = ZP.save_npz(R.parts / f"{st}.npz", arrays)
    sha_kt = ZP.save_npy(R.parts / f"{st}.kt.npy", kt)
    write_json(R.parts / f"{st}.json", dict(hd, file=f"{st}.npz", file_sha256=sha, rows=int(kt.shape[0]),
                                            kt={"file": f"{st}.kt.npy", "sha256": sha_kt, "shape": list(kt.shape)},
                                            check=chk))
    log(f"{st}: {len(js)} questions, {int(kt.shape[0])} rows, {sec:.1f} s")
    return {"seconds": sec, "questions": len(js), "check": chk}


def merge_carve(name, carve, k, R=None):
    """The K shards of a carve joined into pools/D__C.npz, its KT rows (pools/D__C.kt.npy) and its manifest
    (pools/D__C.json, written last; the merged files are rewritten only while it is absent)."""
    R = roots() if R is None else R
    st = stem(name, carve)
    man = R.pools / f"{st}.json"
    if man.exists():
        raise SystemExit(f"{relr(man)} exists: PKO never re-merges a filed carve")
    arrays, kts, heads, shas, at, nq, secs = [], [], [], {}, 0, None, 0.0
    for i in range(int(k)):
        pj = R.parts / f"{stem(name, carve, (i, k))}.json"
        if not pj.exists():
            raise SystemExit(f"{relr(pj)} is missing")
        part = read_json(pj)
        nq = part["nq"] if nq is None else nq
        if part["range"][0] != at or part["nq"] != nq or part["shard"] != [i, int(k)]:
            raise SystemExit(f"{relr(pj)}: the shards are not contiguous ranges of one carve")
        at = part["range"][1]
        if sha_file(R.parts / part["file"]) != part["file_sha256"] or \
                sha_file(R.parts / part["kt"]["file"]) != part["kt"]["sha256"]:
            raise SystemExit(f"{relr(pj)}: its arrays are not the ones it names")
        with np.load(R.parts / part["file"], allow_pickle=False) as z:
            arrays.append({key: z[key] for key in z.files})
        kts.append(np.load(R.parts / part["kt"]["file"], allow_pickle=False))
        heads.append({key: part[key] for key in SAME})
        shas[pj.name] = sha_file(pj)
        secs += float(part["seconds"]["pass"])
    if at != nq:
        raise SystemExit(f"{name}/{carve}: the shards cover {at} of {nq} questions")
    if any(head_key(h) != head_key(heads[0]) for h in heads):
        raise SystemExit(f"{name}/{carve}: the shards were not built by one script on one input")
    filed = merge_arrays(arrays)
    kt = np.concatenate(kts)
    check_filed(filed, kt)
    if len(filed["ids"]) != nq or len(set(str(x) for x in filed["ids"])) != nq:
        raise SystemExit(f"{name}/{carve}: the question ids are not the carve's")
    sha = ZP.save_npz(R.pools / f"{st}.npz", filed)
    sha_kt = ZP.save_npy(R.pools / f"{st}.kt.npy", kt)
    bes = beside(filed)
    ent = dict(heads[0], file=relr(R.pools / f"{st}.npz"), sha256=sha, questions=int(nq), rows=bes["rows"],
               changed=bes["changed"], short=bes["short"], beside=bes, parts=shas, pass_seconds=round(secs, 1),
               kt={"file": relr(R.pools / f"{st}.kt.npy"), "sha256": sha_kt, "shape": list(kt.shape)}, utc=utc())
    write_json(man, ent)
    log(f"{name}/{carve}: merged {k} shards, {nq} questions, {bes['rows']} rows")
    return ent


def resolve(f):
    p = Path(f)
    return p if p.is_absolute() else ROOT / p


def load_filed(name, carve, R=None):
    """(filed, ent, manifest sha) of a merged PKO carve, as zu2d_pools.load_filed gives U2d's."""
    R = roots() if R is None else R
    man = R.pools / f"{stem(name, carve)}.json"
    if not man.exists():
        raise SystemExit(f"{relr(man)} is missing: merge the carve first")
    ent = read_json(man)
    if sha_file(resolve(ent["file"])) != ent["sha256"]:
        raise SystemExit(f"{name}/{carve}: the pools file is not the one its manifest names")
    with np.load(resolve(ent["file"]), allow_pickle=False) as z:
        filed = {key: z[key] for key in z.files}
    return filed, ent, sha_file(man)


def check_merge(name, k, R=None):
    """Section 5 (a)'s check parts of a dataset joined: check/D.json, PASS when they cover U2s's carve, IDENTICAL."""
    R = roots() if R is None else R
    carve, out = u2s_carve(name), R.check / f"{name}.json"
    if out.exists():
        raise SystemExit(f"{relr(out)} exists")
    at, nq, parts, bad, against = 0, None, {}, {}, None
    for i in range(int(k)):
        pj = R.check_parts / f"{stem(name, carve, (i, k))}.json"
        if not pj.exists():
            raise SystemExit(f"{relr(pj)} is missing")
        c = read_json(pj)
        nq = c["nq"] if nq is None else nq
        if c["range"][0] != at or c["nq"] != nq:
            raise SystemExit(f"{relr(pj)}: the check parts are not contiguous")
        at = c["range"][1]
        parts[pj.name] = sha_file(pj)
        against = c["check"]["against"]
        for key, n in c["check"]["mismatches"].items():
            bad[key] = bad.get(key, 0) + int(n)
    verdict = "PASS" if at == nq and not bad else "FAIL"
    write_json(out, {"declared_in": DECLARED, "stage": STAGE, "check": "5 (a)", "dataset": name, "carve": carve,
                     "questions": int(nq), "covered": int(at), "parts": parts, "against": against,
                     "mismatches": bad, "verdict": verdict, "utc": utc()})
    log(f"section 5 (a) {name}/{carve}: {verdict}")
    if verdict != "PASS":
        raise SystemExit(1)
    return verdict


# ── section 5 (b): squad ─────────────────────────────────────────────────────

def rec_sha(rec, key):
    """An array's sha256 in a cache part record."""
    for c in (_field(rec.get("arrays", {}), key), rec.get(key), _field(rec.get("files", {}), key)):
        if isinstance(c, dict) and isinstance(c.get("sha256"), str):
            return c["sha256"]
        if isinstance(c, str) and len(c) == 64:
            return c
    s = rec.get(f"{key}_sha256")
    if isinstance(s, str):
        return s
    raise SystemExit(f"PKO: no sha256 for {key!r} in the cache record")


def u2d_squad(carve, u2d=U2D_NEW):
    """U2d new's squad part records of a carve and its manifest entry (only the keys section 5 (b) names)."""
    recs = by_index((Path(u2d) / "cache" / "squad" / carve).glob("part_*/record.json"))
    if not recs:
        raise SystemExit(f"U2d new has no squad/{carve} cache records")
    man = read_json(Path(u2d) / "pools" / "squad" / "pools.json")
    ent = man.get("files", {}).get(f"squad__{carve}")
    if not isinstance(ent, dict):
        raise SystemExit(f"U2d new's squad manifest has no squad__{carve} entry")
    keys = ("questions", "changed", "short", "budget_mean", "fill_want_mean", "today_inc_mean", "graph_detail",
            "pool_mean")
    out = {k: ent[k] for k in keys}                              # never I, pool, caps or shares (section 5 (b))
    out["B_q"] = {k: ent["B_q"][k] for k in ("max", "mean")}
    out["rows"] = {k: ent["rows"][k] for k in ("max", "mean")}
    out["parts_rows"] = int(sum(int(x["rows"]) for x in ent["parts"]))
    return recs, out


def close(a, b):
    """Two manifest values equal: numbers to 1e-9 relative, dicts and lists entry by entry, the rest exactly."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return bool(np.isclose(a, b, rtol=1e-9, atol=0.0))
    return a == b


def squad_summary(filed, ent):
    """What section 5 (b) compares with U2d new's manifest entry, written the way U2d writes its own
    (zu2d_pools.carve_summary: means at one decimal; B_q = pool size - base-set size, the fill; PKO A2)."""
    sz = pool_sizes(filed)
    fill = np.asarray(filed["count"], np.int64)
    r1 = lambda x: round(float(x), 1)                              # noqa: E731 -- U2d files its means at one decimal
    return {"questions": int(len(filed["ids"])), "changed": ent["changed"], "short": ent["short"],
            "budget_mean": r1(ent["beside"]["budget_mean"]), "fill_want_mean": r1(ent["beside"]["fill_want_mean"]),
            "today_inc_mean": r1(ent["beside"]["today_inc_mean"]), "graph_detail": ent["graph_detail"],
            "pool_mean": r1(sz.mean()), "B_q": {"max": int(fill.max()), "mean": r1(fill.mean())},
            "rows": {"max": int(sz.max()), "mean": r1(sz.mean())}, "parts_rows": int(sz.sum())}


def check_squad(carve, R=None, u2d=U2D_NEW, dry=False):
    """Section 5 (b): PKO's squad pools against U2d new's (k_q = 0, the pool is B)."""
    R = roots() if R is None else R
    out = R.check / f"squad__{carve}.json"
    recs, uman = u2d_squad(carve, u2d)
    rs = [read_json(p) for p in recs]
    ids = [str(x) for r in rs for x in r["ids"]]
    want = {"n": [rec_sha(r, "n") for r in rs], "node": [rec_sha(r, "node") for r in rs]}
    if dry:
        log(f"U2d new squad/{carve}: {len(recs)} part records, {len(ids)} ids, manifest keys {sorted(uman)}")
        return None
    if out.exists():
        aside = out.with_name(f"squad__{carve}.a1-fail.json")      # A2: a FAIL filed under A1's rule is kept, aside, and rerun
        old = read_json(out)
        if old.get("verdict") != "FAIL" or old.get("rule") == RULE_B or aside.exists():
            raise SystemExit(f"{relr(out)} exists")
        os.replace(out, aside)
        log(f"{relr(out)}: a FAIL filed before A2 is kept as {relr(aside)}; the check runs again under A2")
    filed, ent, msha = load_filed("squad", carve, R)
    pos = {str(x): q for q, x in enumerate(filed["ids"])}
    if sorted(pos) != sorted(ids):
        raise SystemExit(f"squad/{carve}: the questions are not U2d new's")
    pools = pools_of(filed)
    got_n, got_node, at = [], [], 0
    for r in rs:
        qs = [pos[str(x)] for x in r["ids"]]
        got_n.append(ZP.npy_sha(np.asarray([pools[q].size for q in qs], np.int32)))
        got_node.append(ZP.npy_sha(np.concatenate([pools[q] for q in qs]).astype(np.int32) if qs
                                   else np.zeros(0, np.int32)))
        at += len(qs)
    mine = squad_summary(filed, ent)
    diffs = [k for k in uman if not close(mine.get(k), uman[k])]
    zero = bool((np.asarray(filed["fill_want"]) == 0).all() and (np.asarray(filed["count"]) == 0).all())
    same = got_n == want["n"] and got_node == want["node"] and not diffs and zero
    link = squad_link(carve, R, recs, rs, u2d) if same else None
    verdict = "PASS" if same else "FAIL"
    write_json(out, {"declared_in": DECLARED, "stage": STAGE, "check": "5 (b)", "rule": RULE_B, "dataset": "squad", "carve": carve,
                     "u2d_records": {relr(p): sha_file(p) for p in recs}, "pools_manifest_sha256": msha,
                     "n_sha256": got_n == want["n"], "node_sha256": got_node == want["node"], "k_q_zero": zero,
                     "manifest_differs": diffs, "link": link, "verdict": verdict, "utc": utc()})
    log(f"section 5 (b) squad/{carve}: {verdict}" + (f" (manifest differs on {diffs})" if diffs else ""))
    if verdict != "PASS":
        raise SystemExit(1)
    return verdict


def squad_link(carve, R, recs, rs, u2d):
    """U2d new's squad cache arrays placed in PKO's cache root when every one is present with its sha256; otherwise
    'REBUILD' and squad's chain is built."""
    files = []
    for p, r in zip(recs, rs):
        arr = r.get("arrays", {})
        for key, a in (arr.items() if isinstance(arr, dict) else []):
            src = p.parent / (a.get("file") or f"{key}.npy") if isinstance(a, dict) else None
            if src is None or not src.exists() or sha_file(src) != a.get("sha256"):
                return {"state": "REBUILD", "why": "U2d new's squad cache arrays are not all on this machine"}
            files.append((src, p.parent.name))
        files.append((p, p.parent.name))
    if not any(f[0].name != "record.json" for f in files):
        return {"state": "REBUILD", "why": "U2d new's squad cache records name no arrays"}
    placed = []
    for src, part in files:
        dst = R.cache / "squad" / carve / part / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        WQ.place(src, dst, sha_file(src))
        placed.append(relb(R, dst))
    write_json(R.cache / "squad" / carve / "linked.json", {"declared_in": DECLARED, "from": relr(Path(u2d)),
                                                           "files": placed, "utc": utc()})
    return {"state": "LINKED", "files": len(placed)}


# ── the chain's steps (section 4) ────────────────────────────────────────────

def checks_state(R):
    names = [f"{n}.json" for n in CHECKS] + [f"squad__{c}.json" for _d, c in SQUAD]
    return {n: (read_json(R.check / n).get("verdict") if (R.check / n).exists() else None) for n in names}


def need_checks(R):
    st = checks_state(R)
    if any(v != "PASS" for v in st.values()):
        raise SystemExit(f"PKO: section 5's checks have not all passed ({st}); nothing is built before they do")


def copy_bases(R=None, src=STEP1):
    """lean_cache's two bases placed in PKO's cache root, each checked against its json's npz_sha256."""
    R = roots() if R is None else R
    placed = []
    for b in ("2wiki", "hotpotqa"):
        js, npz = Path(src) / f"basis_{b}.json", Path(src) / f"basis_{b}.npz"
        if sha_file(npz) != read_json(js)["npz_sha256"]:
            raise SystemExit(f"{relr(npz)} is not the basis its json names")
        for s_ in (npz, js):
            d_ = R.cache / s_.name
            d_.parent.mkdir(parents=True, exist_ok=True)
            WQ.place(s_, d_, sha_file(s_))
            placed.append(d_)
    return placed


def zp_roots(R):
    return Roots(base=R.base, out=R.base, pools=R.pools, parts=R.parts, look=R.look, cache=R.cache, kt=R.kt,
                 links=R.links, rmatch=R.rmatch, zrc=R.zrc, typed=R.typed, records=R.records)


def look_shard(name, carve, shard, R=None):
    """look_u2d on PKO's pools: zu2d_pools.load_filed and roots, and look_u2d.DECLARED, patched in process."""
    R = roots() if R is None else R
    need_checks(R)
    import look_u2d as LU
    filed, ent, msha = load_filed(name, carve, R)
    zr = zp_roots(R)

    def lf(*a, **k):
        got = [x for x in list(a) + list(k.values()) if isinstance(x, str)]
        if name not in got or carve not in got:
            raise SystemExit(f"look_u2d asked for {got}; PKO patched {name}/{carve} only")
        return filed, ent, msha

    def rt(*_a, **_k):
        return zr

    pl = {"DECLARED": DECLARED}
    pl.update({k: v for k, v in (("load_filed", lf), ("roots", rt)) if hasattr(LU, k)})
    argv = ["--build", "new", "--dataset", name, "--carve", carve, "--full", "--shard", f"{shard[0]}/{shard[1]}"]
    with fit_ctx(name, carve), WQ.patch(ZP, load_filed=lf, roots=rt), WQ.patch(LU, **pl):
        LU.main(argv)
    outs = look_outs(R, name, carve, shard)
    if not all(p.exists() for p in outs):
        raise SystemExit(f"look {name}/{carve} {shard}: look_u2d did not write {[relr(p) for p in outs]}")


def look_outs(R, name, carve, shard):
    d = R.look / name / carve
    return [d / f"record_{shard[0]}of{shard[1]}.json", d / f"pools_{shard[0]}of{shard[1]}.json"]


def cache_part(name, carve, part, R=None):
    """lean_cache.py on PKO's look and cache roots, in process (webqsp's fit opened as WQK opened it)."""
    R = roots() if R is None else R
    need_checks(R)
    out = R.cache / name / carve / f"part_{part[0]}of{part[1]}" / "record.json"
    linked = R.cache / name / carve / "linked.json"
    if linked.exists() and relb(R, out) in read_json(linked)["files"]:
        log(f"{relr(out)}: linked from U2d new (section 5 (b)); nothing to build")
        return
    argv = [str(HERE / "lean_cache.py"), "--dataset", name, "--carve", carve, "--part", f"{part[0]}/{part[1]}",
            "--look-root", relr(R.look), "--out-root", relr(R.cache)]
    with fit_ctx(name, carve), WQ.patch(sys, argv=argv):
        try:
            runpy.run_path(str(HERE / "lean_cache.py"), run_name="__main__")
        except SystemExit as e:
            if e.code not in (None, 0):
                raise
    if not out.exists():
        raise SystemExit(f"cache {name}/{carve} {part}: lean_cache did not write {relr(out)}")


def chain_build(what, name, carve, R=None):
    """zp1c2's rmatch, zrc or links on PKO's roots."""
    R = roots() if R is None else R
    need_checks(R)
    import zp1c2 as Z2
    with fit_ctx(name, carve), WQ.patch(Z2, LOOK=R.look, CACHE=R.cache, CH_RM=R.rmatch, CH_ZRC=R.zrc, LINKS=R.links):
        Z2.build(what, name, carve, False)


def typed_carve(name, carve, R=None):
    """zu2n's typed arrays on PKO's roots (webqsp's fit through zu2n.extra_carves)."""
    R = roots() if R is None else R
    need_checks(R)
    import zu2n as U2N
    known = tuple(tuple(c) for c in U2N.TYPED_CARVES)
    extra = contextlib.nullcontext() if (name, carve) in known else U2N.extra_carves((name, carve))
    with fit_ctx(name, carve), extra:
        U2N.build_typed(name, carve, chunk_root=R.look, out_root=R.typed, look_root=R.look, cache_root=R.cache,
                        links_root=R.links, ch_root=R.rmatch)


def kt_stage(name, carve, R=None, cache_rows=None):
    """The pools' KT rows put in the cache's row order (zu2d_pools.to_cache_order), with the record zp1c3.load_kt
    reads (its stage patches zp1c3.DECLARED to PKO's, section 8)."""
    R = roots() if R is None else R
    cache_rows = KT3.cache_rows if cache_rows is None else cache_rows
    d = R.kt / name / carve
    if (d / "record.json").exists():
        raise SystemExit(f"{relr(d / 'record.json')} exists")
    t0 = time.time()
    filed, ent, msha = load_filed(name, carve, R)
    if sha_file(resolve(ent["kt"]["file"])) != ent["kt"]["sha256"]:
        raise SystemExit(f"{name}/{carve}: the KT rows are not the ones the manifest names")
    flat = np.load(resolve(ent["kt"]["file"]), allow_pickle=False)
    c_ids, c_n, c_node, part_shas = cache_rows(R.cache, name, carve)
    kt = np.asarray(ZP.to_cache_order(filed, flat, c_ids, c_n, c_node), ZP.KT_DT)
    if kt.shape != flat.shape:
        raise SystemExit(f"{name}/{carve}: the cache's rows are not the pools' rows")
    d.mkdir(parents=True, exist_ok=True)
    sha_kt = ZP.save_npy(d / "kt.npy", kt)
    sha_n = ZP.save_npy(d / "n.npy", np.asarray(c_n, np.int32))
    write_json(d / "ids.json", [str(x) for x in c_ids])
    rec = {"declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": carve, "construction": CONSTRUCTION,
           "columns_declared_in": KT3.DECLARED, "columns": list(KT3.COLS), "nmax": int(KT3.NMAX), "dtype": "float16",
           "walks": "P1e's (P1c3's definition, as U2d new's)",
           "arrays": {"kt": {"file": "kt.npy", "sha256": sha_kt, "shape": list(kt.shape)},
                      "n": {"file": "n.npy", "sha256": sha_n, "shape": [int(np.asarray(c_n).size)]}},
           "cache_root": relr(R.cache), "cache_parts": part_shas, "questions": len(c_ids), "rows": int(kt.shape[0]),
           "ids_sha256": sha_file(d / "ids.json"), "pools_manifest_sha256": msha, "pools_file_sha256": ent["sha256"],
           "pools_kt_sha256": ent["kt"]["sha256"], "scripts": scripts(),
           "checks": {"cache_rows_are_pool_rows": "IDENTICAL (zu2d_pools.to_cache_order)"},
           "seconds": round(time.time() - t0, 1), "utc": utc()}
    write_json(d / "record.json", rec)
    log(f"kt {name}/{carve}: {len(c_ids)} questions, {int(kt.shape[0])} rows")
    return rec


def step1_check(name, carve, filed):
    """zu2d_pools.step1_check on the carve's step-1 cache (WQK's for webqsp's fit)."""
    root = STEP1_FIT if (name, carve) == FIT else STEP1
    recs = [read_json(p) for p in by_index((root / name / carve).glob("part_*/record.json"))]
    if not recs:
        raise SystemExit(f"no step-1 cache records for {name}/{carve}")
    return ZP.step1_check(filed, recs)


def record_carve(name, carve, R=None, step1=None):
    """Section 5 (c): the carve's record, naming the sha256 of every input it was built from."""
    R = roots() if R is None else R
    step1 = step1_check if step1 is None else step1
    out = R.records / f"{stem(name, carve)}.json"
    if out.exists():
        raise SystemExit(f"{relr(out)} exists")
    filed, ent, msha = load_filed(name, carve, R)
    chained = (name, carve) in CHAINED
    inputs, problems = {}, []

    def add(paths, what):
        paths = list(paths)
        if not paths:
            problems.append(f"no {what}")
        for p in paths:
            if Path(p).exists():
                inputs[relb(R, p)] = sha_file(p)
            else:
                problems.append(f"{relb(R, p)} is missing")

    def d(root):
        return Path(root) / name / carve

    st = stem(name, carve)
    add([R.pools / f"{st}.json", resolve(ent["file"]), resolve(ent["kt"]["file"])], "pools")
    add(by_index(d(R.look).glob("record_*.json")), "look records")
    add(by_index(d(R.look).glob("pools_*.json")), "look pools records")
    add(by_index(d(R.cache).glob("part_*/record.json")), "cache records")
    add(by_index(d(R.links).glob("part_*/record.json")), "links records")
    if chained:
        add(by_index(d(R.rmatch).glob("part_*/record.json")), "rmatch records")
        add(by_index(d(R.zrc).glob("part_*/record.json")), "zrc records")
        add([d(R.typed) / "record.json"], "typed record")
    add([d(R.kt) / f for f in ("record.json", "kt.npy", "n.npy", "ids.json")], "KT")
    ktr = read_json(d(R.kt) / "record.json") if (d(R.kt) / "record.json").exists() else {}
    if ktr and (ktr.get("pools_manifest_sha256") != msha or ktr.get("pools_file_sha256") != ent["sha256"]):
        problems.append("the KT record was not built from this pools file")
    s1 = step1(name, carve, filed)
    rec = {"declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": carve, "pick": PICK,
           "construction": CONSTRUCTION, "chained": chained, "graph_detail": ent["graph_detail"], "walk": ent["walk"],
           "u2s": ent["u2s"], "types": ent["types"], "inputs": inputs, "pools_manifest_sha256": msha,
           "step1_check": json.loads(json.dumps(s1, default=str)), "beside": ent["beside"], "scripts": scripts(),
           "problems": problems, "utc": utc()}
    write_json(out, rec)
    log(f"record {name}/{carve}: {len(inputs)} inputs" + (f", problems {problems}" if problems else ""))
    if problems:
        raise SystemExit(1)
    return rec


def gate(R=None, carves=CARVES):
    """Every carve's record present and every input it names re-hashed; section 5's checks passed."""
    R = roots() if R is None else R
    problems, recs = [], {}
    st = checks_state(R)
    problems += [f"check {n}: {v}" for n, v in st.items() if v != "PASS"]
    for name, carve in carves:
        p = R.records / f"{stem(name, carve)}.json"
        if not p.exists():
            problems.append(f"{relb(R, p)} is missing")
            continue
        rec = read_json(p)
        for f, sha in rec["inputs"].items():
            q = R.base / f
            if not q.exists() or sha_file(q) != sha:
                problems.append(f"{f} is not the file {p.name} names")
        recs[p.name] = sha_file(p)
    verdict = "READY" if not problems else "NOT_READY"
    write_json(R.gate, {"declared_in": DECLARED, "stage": STAGE, "carves": recs, "checks": st, "verdict": verdict,
                        "problems": problems, "scripts": scripts(), "utc": utc()})
    log(f"gate: {verdict}" + (f" ({len(problems)} problems)" if problems else ""))
    if problems:
        raise SystemExit(1)
    return verdict


def prune_carve(name, carve, R=None):
    """Section 6: the carve's look chunks deleted (regenerable), only once every step that reads the look is filed OK;
    its records and pools JSON are kept."""
    R = roots() if R is None else R
    led = read_ledger(R)["entries"]
    pre = f"cache:{name}/{carve}:"
    ns = {k.rsplit(":", 1)[1].split("/")[1] for k in led if k.startswith(pre)}
    want = [f"links:{name}/{carve}"]
    if (name, carve) in CHAINED:
        want += [f"rmatch:{name}/{carve}", f"zrc:{name}/{carve}", f"typed:{name}/{carve}"]
    if len(ns) == 1:
        n = int(next(iter(ns)))
        want += [f"{pre}{i}/{n}" for i in range(n)]
    not_ok = [k for k in want if not entry_ok(R, led.get(k))] + ([] if len(ns) == 1 else [f"{pre}*"])
    if not_ok:
        raise SystemExit(f"prune {name}/{carve}: refused, the look still has readers not filed OK: {not_ok}")
    d = R.look / name / carve
    if (d / "pruned.json").exists():
        raise SystemExit(f"{relr(d / 'pruned.json')} exists")
    chunks = sorted((d / "chunks").glob("c*.npz"))
    gone = sum(p.stat().st_size for p in chunks)
    for p in chunks:
        p.unlink()
    write_json(d / "pruned.json", {"declared_in": DECLARED, "stage": STAGE, "dataset": name, "carve": carve,
                                   "removed": len(chunks), "bytes": int(gone), "readers": sorted(want), "utc": utc()})
    log(f"prune {name}/{carve}: {len(chunks)} look chunks, {gone / 2 ** 30:.2f} GiB")
    return len(chunks)


# ── ledger, disk guard and lanes (section 6) ─────────────────────────────────

def read_ledger(R):
    return read_json(R.ledger) if Path(R.ledger).exists() else {"declared_in": DECLARED, "stage": STAGE, "entries": {}}


def entry_ok(R, e):
    """An OK entry whose files are still the ones it names."""
    if not e or e.get("status") != "OK":
        return False
    return all((R.base / f).exists() and sha_file(R.base / f) == sha for f, sha in e.get("files", {}).items())


def file_entry(R, key, status, files, seconds, rc=0, peak=None, need=None):
    """One ledger entry. A1 adds peak_rss_bytes (the step's process tree) and need_bytes (the RAM it declared) when
    known; an entry without them (A0's) is read exactly as before."""
    with LEDGER_LOCK:
        led = read_ledger(R)
        led["entries"][key] = {"status": status, "files": {relb(R, p): sha_file(p) for p in files},
                               "seconds": round(float(seconds), 1), "rc": int(rc), "utc": utc()}
        for k_, v_ in (("peak_rss_bytes", peak), ("need_bytes", need)):
            if v_ is not None:
                led["entries"][key][k_] = int(v_)
        write_json(R.ledger, led)


# ── A1: peak working set of a step, and the RAM preflight ────────────────────

class RamRefused(Exception):
    """A step waited RAM_WAIT_S (or --ram-wait) for the RAM it declares and did not get it; nothing is filed."""


def _tree(pid):
    p = psutil.Process(pid)
    return [p] + p.children(recursive=True)


def tree_rss(pid):
    """Resident bytes of a process and its children now (0 when psutil is missing or the process has gone)."""
    if psutil is None or not pid:
        return 0
    try:
        return sum(q.memory_info().rss for q in _tree(pid))
    except psutil.Error:
        return 0


class PeakWatch:
    """Samples a process tree every `every` seconds; peak is the larger of the best sampled tree sum and the largest
    single-process peak the OS keeps (peak_wset on Windows, VmHWM on Linux), so a spike between two samples counts."""

    def __init__(self, pid, every=0.5):
        self.pid, self.every, self.peak = pid, float(every), 0
        self.stop_ev, self.t = threading.Event(), None
        if psutil is not None:
            self.sample()
            self.t = threading.Thread(target=self.run, daemon=True)
            self.t.start()

    def sample(self):
        try:
            tot, best = 0, 0
            for q in _tree(self.pid):
                try:
                    mi = q.memory_info()
                except psutil.Error:
                    continue
                tot += mi.rss
                best = max(best, int(getattr(mi, "peak_wset", 0) or 0), _hwm(q.pid))
            self.peak = max(self.peak, tot, best)
        except psutil.Error:
            pass

    def run(self):
        while not self.stop_ev.wait(self.every):
            self.sample()

    def stop(self):
        """The peak in bytes, or None when psutil is missing."""
        if self.t is None:
            return None
        self.stop_ev.set()
        self.t.join(5)
        return int(self.peak)


def _hwm(pid):
    """VmHWM of a process in bytes on Linux (0 elsewhere or when it cannot be read)."""
    try:
        with open(f"/proc/{pid}/status", "rb") as fh:
            for line in fh:
                if line.startswith(b"VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _kind(key):
    return key.split(":")[0]


def need_bytes(kind, name, rows=None, part_rows=None):
    """A step's declared RAM in bytes: its measured (or guessed) peak plus MARGIN_GIB (docs/PKO_A1_AMENDMENT.md
    section 4). pools, check and look hold a carve's graph and names: a lane's figure by dataset; a cache part grows
    with its rows; a merge loads every shard; the rest are small."""
    if kind in ("pools", "check", "look"):
        gib = LANE_GIB[name]
    elif kind == "cache":
        gib = CACHE_FLOOR_GIB + CACHE_GIB_MROW[name] * float(part_rows or PART_ROWS_BY.get(name, PART_ROWS)) / 1e6 \
            + MARGIN_GIB
    elif kind == "merge":
        gib = MERGE_FLOOR_GIB + MERGE_BPR * float(rows or 0) / GIB + MARGIN_GIB
    else:
        gib = STEP_GIB[kind]
    return int(gib * GIB)


class RamGuard:
    """A1's preflight: a step starts only when the RAM it declares is available now, less what the steps already running
    have yet to grow into. It waits up to `wait` seconds (polling every `poll`), then raises RamRefused: the chain
    stops and nothing is filed FAILED, so the same command resumes. Without psutil it cannot measure and admits."""

    def __init__(self, wait=RAM_WAIT_S, avail=None, rss=None, poll=30.0):
        self.wait, self.poll = float(wait), float(poll)
        self.avail = avail or ((lambda: psutil.virtual_memory().available) if psutil is not None else (lambda: None))
        self.rss = rss or (lambda st: tree_rss(getattr(st, "pid", None)))
        self.cv, self.running, self.refused = threading.Condition(), {}, None

    def pending(self):
        return sum(max(0, int(getattr(st, "need", 0)) - self.rss(st)) for st in self.running.values())

    def admit(self, st):
        need, t0, said = int(getattr(st, "need", 0) or 0), time.time(), False
        with self.cv:
            while True:
                have = self.avail()
                if have is None or need <= have - self.pending():
                    self.running[st.key] = st
                    return
                left = self.wait - (time.time() - t0)
                if left <= 0:
                    self.refused = (st.key, need, int(have))
                    log(f"ram: {st.key} needs {need / GIB:.1f} GiB, {have / GIB:.1f} GiB is available (less what "
                        f"the running steps still take); the chain stops, nothing is filed FAILED")
                    raise RamRefused(st.key)
                if not said:
                    log(f"ram: {st.key} waits for {need / GIB:.1f} GiB ({have / GIB:.1f} GiB available now, up to "
                        f"{self.wait:.0f} s)")
                    said = True
                self.cv.wait(timeout=min(self.poll, left))

    def done(self, st):
        with self.cv:
            self.running.pop(st.key, None)
            self.cv.notify_all()


def run_step(R, st, threads):
    """One step as its own process (numba and BLAS at 2), its output appended to logs/<key>.log. A1: its pid and peak
    working set are left on st (st.pid while it runs, st.peak_rss_bytes after)."""
    R.logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", NUMBA_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS="2",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    base = [] if Path(R.base).resolve() == OUT.resolve() else ["--base", str(R.base)]
    with open(R.logs / (re.sub(r"[^A-Za-z0-9_.-]+", "_", st.key) + ".log"), "ab") as fh:
        fh.write(f"\n== {utc()} {st.key}\n".encode())
        fh.flush()
        proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + base + list(st.argv), cwd=str(ROOT),
                                env=env, stdout=fh, stderr=subprocess.STDOUT)
        st.pid = proc.pid
        watch = PeakWatch(proc.pid)
        rc = proc.wait()
        st.peak_rss_bytes = watch.stop()
        st.pid = None
        return rc


def do_step(R, st, retry, runner, ram=None):
    """Skip a step filed OK whose files still match; refuse a FAILED one without --retry; else (A1: once its declared
    RAM is available) run and file it, with its peak working set when the runner measured one."""
    with LEDGER_LOCK:
        e = read_ledger(R)["entries"].get(st.key)
    if entry_ok(R, e):
        return True
    if e is not None and e.get("status") == "FAILED" and not retry:
        log(f"{st.key}: FAILED in the ledger (--retry runs it again)")
        return False
    if ram is not None:
        ram.admit(st)
    t = time.time()
    try:
        rc = runner(st)
    finally:
        if ram is not None:
            ram.done(st)
    have = [Path(p) for p in st.outs if Path(p).exists()]
    ok = rc == 0 and len(have) == len(st.outs)
    peak = getattr(st, "peak_rss_bytes", None)
    file_entry(R, st.key, "OK" if ok else "FAILED", have, time.time() - t, rc, peak, getattr(st, "need", None) or None)
    log(f"{st.key}: {'OK' if ok else 'FAILED'} ({time.time() - t:.1f} s"
        f"{'' if peak is None else f', peak {peak / GIB:.2f} GiB'})")
    return ok


def du(paths):
    n = 0
    for p in paths:
        p = Path(p)
        if p.is_file():
            n += p.stat().st_size
        elif p.is_dir():
            n += sum(q.stat().st_size for q in p.rglob("*") if q.is_file())
    return n


def du_carve(R, name, carve):
    st = stem(name, carve)
    return du([Path(R[r]) / name / carve for r in ("look", "cache", "rmatch", "zrc", "links", "typed", "kt")]
              + list(Path(R.parts).glob(f"{st}__*")) + list(Path(R.pools).glob(f"{st}.*")))


class Disk:
    """Section 6's guard: a carve starts only when its estimated bytes fit in the free disk less 15 GiB and less what
    the running carves are still expected to write; it waits while another carve runs, and stops otherwise."""

    def __init__(self, R, reserve=RESERVE, free=None, used=None, poll=60.0):
        self.reserve, self.poll = int(reserve), float(poll)
        here = Path(R.base) if Path(R.base).exists() else ROOT
        self.free = free or (lambda: shutil.disk_usage(here).free)
        self.used = used or (lambda job: du_carve(R, job.name, job.carve))
        self.cv = threading.Condition()
        self.running = {}

    def pending(self):
        return sum(max(0, est - (self.used(job) - u0)) for job, est, u0 in self.running.values())

    def admit(self, job, est):
        with self.cv:
            said = False
            while True:
                room = self.free() - self.reserve - self.pending()
                if est <= room:
                    self.running[job.key] = (job, int(est), self.used(job))
                    return True
                if not self.running:
                    log(f"disk: {job.key} needs {est / 2 ** 30:.1f} GiB, {room / 2 ** 30:.1f} GiB is free above the "
                        f"reserve; the chain stops")
                    return False
                if not said:
                    log(f"disk: {job.key} waits ({est / 2 ** 30:.1f} GiB needed, {room / 2 ** 30:.1f} GiB free above "
                        f"the reserve and the running carves)")
                    said = True
                self.cv.wait(timeout=self.poll)

    def done(self, job):
        with self.cv:
            self.running.pop(job.key, None)
            self.cv.notify_all()


def run_lanes(R, jobs, lanes, retry, runner, disk, stop, ram=None):
    """Jobs (one carve each) on N lanes, each lane running its carve's steps in order. A FAILED step, the disk guard or
    (A1) the RAM preflight stops the chain: the lanes finish the step they are in and start nothing new. A1: an error
    in a job's steps (a refused count change, a missing manifest) stops the chain too, where it used to end its lane
    silently."""
    queue, qlock = list(jobs), threading.Lock()

    def lane():
        while not stop.is_set():
            with qlock:
                if not queue:
                    return
                job = queue.pop(0)
            if not disk.admit(job, job.est()):
                stop.set()
                return
            try:
                for st in job.steps():
                    if stop.is_set() or not do_step(R, st, retry, runner, ram):
                        stop.set()
                        return
            except RamRefused:
                stop.set()
                return
            except (SystemExit, Exception) as e:       # noqa: BLE001
                log(f"{job.key}: stopped: {e}")
                stop.set()
                return
            finally:
                disk.done(job)

    ts = [threading.Thread(target=lane, daemon=True) for _ in range(max(1, int(lanes)))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return not stop.is_set()


def step(key, argv, outs, need=0):
    return SimpleNamespace(key=key, argv=[str(x) for x in argv], outs=[Path(p) for p in outs], need=int(need),
                           pid=None, peak_rss_bytes=None)


def shard_q(name):
    return SHARD_Q.get(name, SHARD_Q_PASSAGE)


def shards(n, per):
    return max(1, int(math.ceil(int(n) / float(per))))


def chunk_count(nq, rows):
    """The look's chunks for a carve of nq questions and `rows` pool rows, as look_x_six cuts them (line 260):
    max(1, CHUNK_NODES // mean pool) questions to a chunk. A0 used squad's 479 for every carve (docs/PKO_A1_AMENDMENT.md
    section 2)."""
    per = max(1, int(CHUNK_NODES // max(float(rows) / max(1, int(nq)), 1.0)))
    return shards(nq, per)


def look_parts(name, nq, rows, look_rows=None, part_rows=None):
    """(look shards, cache parts) of a carve: rows / the target, never more than the carve's chunks. A1's targets are
    LOOK_ROWS and PART_ROWS_BY[name] unless the flags give one."""
    n_chunks = chunk_count(nq, rows)
    return (min(n_chunks, shards(rows, look_rows or LOOK_ROWS)),
            min(n_chunks, shards(rows, part_rows or PART_ROWS_BY.get(name, PART_ROWS))))


def other_counts(R, name, carve, nl, nc):
    """Look records and cache, rmatch, zrc and links parts of this carve filed under another shard or part count: the
    look and lean_cache read all of a carve's files, so a carve's counts are fixed by its first file."""
    bad = []
    for pat in ("record_*of*.json", "pools_*of*.json"):
        for p in (Path(R.look) / name / carve).glob(pat):
            m = re.search(r"of(\d+)\.json$", p.name)
            if m and int(m.group(1)) != nl:
                bad.append(p)
    for root in ("cache", "rmatch", "zrc", "links"):
        for p in (Path(R[root]) / name / carve).glob("part_*of*"):
            m = re.search(r"of(\d+)$", p.name)
            if m and int(m.group(1)) != nc:
                bad.append(p)
    return bad


def chained_roots(name, carve):
    return ("look", "cache", "links", "kt") + (("rmatch", "zrc", "typed") if (name, carve) in CHAINED else ())


def est_bytes(name, carve, rows):
    return int(rows * (POOLS_BPR + sum(BPR[name].get(r, 0) for r in chained_roots(name, carve))))


def est_seconds(name, carve, rows, nq, sq=None):
    us = sum(RATES_US.get(r, 0.0) for r in chained_roots(name, carve))
    return rows * us / 1e6 + (nq * sq if sq else 0.0)


def carve_rows(R, name, carve, rq):
    """(rows, exact): the merged manifest's rows, or the questions times U2s's filed rows per question."""
    man = R.pools / f"{stem(name, carve)}.json"
    if man.exists():
        return int(read_json(man)["rows"]), True
    return int(math.ceil(QUESTIONS[(name, carve)] * rq.get(name, ROWS_Q))), False


def pre_steps(R, name, carve, threads, rows=None):
    """The pools shards, the merge and the carve's section 5 check. `rows` (the carve's rows, estimated before the
    merge) sizes the merge's declared RAM only; no key, argument or output depends on it."""
    k = shards(QUESTIONS[(name, carve)], shard_q(name))
    chk = carve == u2s_carve(name)
    out = []
    for i in range(k):
        st = stem(name, carve, (i, k))
        out.append(step(f"pools:{name}/{carve}:{i}/{k}",
                        ["pools", name, "--carve", carve, "--shard", f"{i}/{k}", "--threads", threads]
                        + (["--check"] if chk else []),
                        [R.parts / f"{st}.json"] + ([R.check_parts / f"{st}.json"] if chk else []),
                        need_bytes("pools", name)))
    out.append(step(f"merge:{name}/{carve}", ["merge", name, "--carve", carve, "--parts", k],
                    [R.pools / f"{stem(name, carve)}.json"], need_bytes("merge", name, rows)))
    if chk:
        out.append(step(f"check-merge:{name}", ["check-merge", name, "--parts", k], [R.check / f"{name}.json"],
                        need_bytes("check-merge", name)))
    if name == "squad":
        out.append(step(f"check-squad:{carve}", ["check-squad", "--carve", carve], [R.check / f"squad__{carve}.json"],
                        need_bytes("check-squad", name)))
    return out


def post_steps(R, name, carve, rows, nq, prune_looks, look_rows=None, part_rows=None):
    """Section 4's build after the pools: look shards, cache parts, rmatch and zrc, links, typed, KT, record, prune.
    A1: the shard and part counts come from the carve's rows and its real chunk count (look_parts), not from squad's
    479 questions to a chunk; each step carries the RAM it declares."""
    nl, nc = look_parts(name, nq, rows, look_rows, part_rows)
    c, d = f"{name}/{carve}", (lambda root: Path(root) / name / carve)
    chained = (name, carve) in CHAINED
    out = [step(f"look:{c}:{i}/{nl}", ["look", name, "--carve", carve, "--shard", f"{i}/{nl}"],
                look_outs(R, name, carve, (i, nl)), need_bytes("look", name)) for i in range(nl)]
    out += [step(f"cache:{c}:{i}/{nc}", ["cache", name, "--carve", carve, "--part", f"{i}/{nc}"],
                 [d(R.cache) / f"part_{i}of{nc}" / "record.json"], need_bytes("cache", name, None, rows / nc))
            for i in range(nc)]
    for what in (("rmatch", "zrc") if chained else ()) + ("links",):
        out.append(step(f"{what}:{c}", ["build", what, name, "--carve", carve],
                        [d(R[what]) / f"part_{i}of{nc}" / "record.json" for i in range(nc)], need_bytes(what, name)))
    if chained:
        out.append(step(f"typed:{c}", ["typed", name, "--carve", carve], [d(R.typed) / "record.json"],
                        need_bytes("typed", name)))
    out.append(step(f"kt:{c}", ["kt", name, "--carve", carve], [d(R.kt) / "record.json"], need_bytes("kt", name)))
    out.append(step(f"record:{c}", ["record", name, "--carve", carve], [R.records / f"{stem(name, carve)}.json"],
                    need_bytes("record", name)))
    if prune_looks:
        out.append(step(f"prune:{c}", ["prune", "--carve", c], [d(R.look) / "pruned.json"], need_bytes("prune", name)))
    return out


def carve_job(R, name, carve, rq, prune_looks, threads, pools_only=False, look_rows=None, part_rows=None):
    def steps():
        yield from pre_steps(R, name, carve, threads, carve_rows(R, name, carve, rq)[0])
        if pools_only:
            return
        ent = read_json(R.pools / f"{stem(name, carve)}.json")
        nq, rows = int(ent["questions"]), int(ent["rows"])
        nl, nc = look_parts(name, nq, rows, look_rows, part_rows)
        bad = other_counts(R, name, carve, nl, nc)
        if bad:
            raise SystemExit(f"{name}/{carve} needs {nl} look shards and {nc} cache parts, but {[relr(p) for p in bad]} "
                             f"are filed under other counts: the look and lean_cache read all of a carve's files. "
                             f"Give --look-rows and --part-rows the values they had, or clear the carve's look and "
                             f"cache after reading the logs")
        yield from post_steps(R, name, carve, rows, nq, prune_looks, look_rows, part_rows)

    def est():
        rows, _exact = carve_rows(R, name, carve, rq)
        return rows * POOLS_BPR if pools_only else est_bytes(name, carve, rows)

    return SimpleNamespace(key=f"{name}/{carve}", name=name, carve=carve, steps=steps, est=est)


def check_job(R, name, threads):
    """Section 5 (a) on a dataset whose U2s carve is not a WQ carve (hotpotqa's s1sel)."""
    carve = u2s_carve(name)
    k = shards(int(u2s_record(name)[0]["questions"]), shard_q(name))
    sts = [step(f"check:{name}:{i}/{k}", ["check", name, "--shard", f"{i}/{k}", "--threads", threads],
                [R.check_parts / f"{stem(name, carve, (i, k))}.json"], need_bytes("check", name)) for i in range(k)]
    sts.append(step(f"check-merge:{name}", ["check-merge", name, "--parts", k], [R.check / f"{name}.json"],
                    need_bytes("check-merge", name)))
    return SimpleNamespace(key=f"{name}/{carve}", name=name, carve=carve, steps=lambda: iter(sts), est=lambda: 0)


def small_first(R, carves, rq):
    """The carves in the order of their rows (exact once merged, else the plan's estimate), the smallest first; ties
    keep the order given."""
    return sorted(carves, key=lambda c: carve_rows(R, *c, rq)[0])


def chain(carves=None, lanes=1, retry=False, prune_looks=False, threads=2, R=None, runner=None, disk=None,
          look_rows=None, part_rows=None, ram=None, as_given=False, checks_filed=False):
    """Section 6: copy-bases; section 5's checks on N lanes; each carve's steps on N lanes; the gate. A1: carves run
    smallest first unless as_given; a step starts only when its declared RAM is free (ram=None builds the RAM guard,
    ram=False runs without one); look_rows and part_rows override the shard and part targets; checks_filed skips the
    section 5 jobs when check/ already holds their PASS (a run on a base the checks were filed in, e.g. per carve on
    Modal). Returns 0 done, 1 stopped by a failure or a guard, 2 stopped for RAM (nothing filed FAILED)."""
    R = roots() if R is None else R
    carves = tuple(carves or CARVES)
    bad = [c for c in carves if c not in CARVES]
    if bad:
        raise SystemExit(f"not WQ carves: {bad}")
    runner = runner or (lambda st: run_step(R, st, threads))
    disk = disk or Disk(R)
    ram = RamGuard() if ram is None else (ram or None)
    rq = filed_rows_q()
    if not as_given:
        carves = tuple(small_first(R, carves, rq))
    stop = threading.Event()
    bases = step("copy-bases", ["copy-bases"], [R.cache / f"basis_{b}.{x}" for b in ("2wiki", "hotpotqa")
                                                for x in ("npz", "json")], need_bytes("copy-bases", "squad"))
    if not do_step(R, bases, retry, runner):
        return 1
    if not checks_filed:
        first = []
        for name in CHECKS:
            c = (name, u2s_carve(name))
            first.append(carve_job(R, *c, rq, prune_looks, threads, pools_only=True) if c in CARVES
                         else check_job(R, name, threads))
        first += [carve_job(R, *c, rq, prune_looks, threads, pools_only=True) for c in SQUAD]
        if not run_lanes(R, first, lanes, retry, runner, disk, stop, ram):
            return 2 if ram is not None and ram.refused else 1
    if any(v != "PASS" for v in checks_state(R).values()):
        log(f"section 5's checks: {checks_state(R)}; nothing is built")
        return 1
    jobs = [carve_job(R, *c, rq, prune_looks, threads, look_rows=look_rows, part_rows=part_rows) for c in carves]
    if not run_lanes(R, jobs, lanes, retry, runner, disk, stop, ram):
        return 2 if ram is not None and ram.refused else 1
    if set(carves) == set(CARVES) and not do_step(R, step("gate", ["gate"], [R.gate], need_bytes("gate", "squad")),
                                                  retry, runner):
        return 1
    log("chain: done")
    return 0


def plan_table(carves=None, sq=None, R=None, look_rows=None, part_rows=None):
    """Per carve: questions, rows per question, rows, bytes and hours (section 9's estimates); A1: its look chunks,
    look shards, cache parts and the order the chain runs it in."""
    R = roots() if R is None else R
    rq = filed_rows_q()
    cs = tuple(carves or CARVES)
    rank = {c: i + 1 for i, c in enumerate(small_first(R, cs, rq))}
    tot, cache_need = {"q": 0, "rows": 0, "bytes": 0, "look": 0, "sec": 0.0}, {}
    log(f"{'carve':<16} {'questions':>9} {'rows/q':>8} {'rows':>11} {'GiB':>7} {'look GiB':>8} {'hours':>7} {'run':>3} "
        f"{'chunks':>6} {'look':>4} {'cache':>5} {'part rows':>9}  rows")
    for name, carve in cs:
        nq = QUESTIONS[(name, carve)]
        rows, exact = carve_rows(R, name, carve, rq)
        b, sec = est_bytes(name, carve, rows), est_seconds(name, carve, rows, nq, sq)
        lk = rows * BPR[name]["look"]
        nl, nc = look_parts(name, nq, rows, look_rows, part_rows)
        cache_need[name] = max(cache_need.get(name, 0), need_bytes("cache", name, None, rows / nc))
        log(f"{name + '/' + carve:<16} {nq:>9} {rows / nq:>8.1f} {rows:>11} {b / 2 ** 30:>7.1f} {lk / 2 ** 30:>8.1f} "
            f"{sec / 3600:>7.2f} {rank[(name, carve)]:>3} {chunk_count(nq, rows):>6} {nl:>4} {nc:>5} {rows // nc:>9}  "
            f"{'exact' if exact else 'U2s filed rate'}")
        for k_, v_ in (("q", nq), ("rows", rows), ("bytes", b), ("look", lk), ("sec", sec)):
            tot[k_] += v_
    log(f"{'all':<16} {tot['q']:>9} {tot['rows'] / tot['q']:>8.1f} {tot['rows']:>11} {tot['bytes'] / 2 ** 30:>7.1f} "
        f"{tot['look'] / 2 ** 30:>8.1f} {tot['sec'] / 3600:>7.2f}  (hours on one lane"
        f"{'' if sq else ', without the pools pass'})")
    log("rows per question at pick O on U2s's carves (U2s's filed cost): "
        + ", ".join(f"{k}: {v:.1f}" for k, v in sorted(rq.items())))
    log("declared RAM (GiB, peak plus 1): " + ", ".join(
        f"{n}: pools/look {need_bytes('look', n) / GIB:.1f}, largest cache part {cache_need[n] / GIB:.1f}"
        for n in sorted(cache_need)))
    return tot


# ── self-test (toy data in temp folders only) ────────────────────────────────

def _toy_fact(j, plan, n_nodes=400):
    rng = np.random.default_rng(1000 + j)
    nb, ni = int(rng.integers(3, 12)), int(rng.integers(0, 15))
    perm = rng.permutation(n_nodes).astype(np.int64)
    B, rest = np.sort(perm[:nb]), perm[nb:]
    I = np.sort(np.concatenate([B, rest[:ni]]))
    kt_len = int(rng.integers(0, 40))
    kt_cos = rest[5:5 + kt_len]
    fill = int(min(kt_len, ni))
    kq, k, keep = (int(x[0]) for x in kept_counts([kt_len], [fill], [nb], plan))
    exp = kt_cos[:keep]
    lens = [int(x) for x in rng.integers(0, 50, M.NO)]
    lens[plan.o] = kt_len
    return {"B": B, "I": I, "exp": exp, "fill": fill, "kq": kq, "k": k, "keep": keep, "kt_len": kt_len,
            "today_inc_size": int(I.size + rng.integers(0, 3)), "entity": bool(j % 2), "route": bool(j % 3 == 0),
            "lens": lens, "n_u": [int(x) for x in rng.integers(0, 99, len(M.MS))], "n_i": ni,
            "n_n1": int(rng.integers(0, 9)), "n_r0": int(rng.integers(0, 9)),
            "n_n2": [int(x) for x in rng.integers(0, 9, len(M.M_GRID))], "taken": int(rng.integers(0, 9)),
            "hubs": int(rng.integers(0, 9)), "mq": [int(x) for x in rng.integers(0, 9, S.NW * S.NTW)],
            "nw": [int(x) for x in rng.integers(0, 9, S.NW)],
            "kt": rng.standard_normal((np.union1d(B, exp).size, KT3.NKT)).astype(np.float16)}


def _toy_u2s(facts, own_k, own_b, plan, ncells=13):
    """U2s's walk-free arrays laid out as zu2s files them, from toy facts."""
    nq, nm, ns, nb = len(facts), len(M.MS), len(M.M_GRID), len(M.BASES)
    a = {"bs": np.zeros((nq, ncells), np.int64), "inc": np.zeros((nq, ncells), np.int64),
         "n_i": np.zeros((nq, ncells), np.int64), "fill": np.zeros((nq, ncells * M.R), np.int64),
         "lens": np.zeros((nq, ncells * M.R * M.NO), np.int32), "n_u": np.zeros((nq, ncells * M.R * nm), np.int64),
         "n_n1": np.zeros((nq, nb), np.int64), "n_r0": np.zeros((nq, nb), np.int64),
         "n_n2": np.zeros((nq, nb * ns), np.int64), "taken": np.zeros((nq, nb), np.int64),
         "hubs": np.zeros((nq, nb), np.int64), "mq": np.zeros((nq, S.NMQ), np.int64),
         "nw": np.zeros((nq, nb * S.NW), np.int64)}
    for j, f in enumerate(facts):
        c, c0 = own_k * M.R + plan.r, M.column(own_k, plan.r, 0)
        a["bs"][j, own_k], a["inc"][j, own_k], a["n_i"][j, own_k] = f["B"].size, f["I"].size, f["n_i"]
        a["fill"][j, c] = f["fill"]
        a["lens"][j, c0:c0 + M.NO] = f["lens"]
        a["n_u"][j, c * nm:(c + 1) * nm] = f["n_u"]
        a["n_n1"][j, own_b], a["n_r0"][j, own_b] = f["n_n1"], f["n_r0"]
        a["n_n2"][j, own_b * ns:(own_b + 1) * ns] = f["n_n2"]
        a["taken"][j, own_b], a["hubs"][j, own_b] = f["taken"], f["hubs"]
        a["mq"][j, S.mq_col(own_b, 0, 0):S.mq_col(own_b + 1, 0, 0)] = f["mq"]
        a["nw"][j, own_b * S.NW:(own_b + 1) * S.NW] = f["nw"]
    return a


def _toy_open(facts, ref):
    cx = SimpleNamespace(bs=[f["B"] for f in facts], pop=SimpleNamespace(ids=[f"q{j:03d}" for j in range(len(facts))]))
    ctx = SimpleNamespace(own_k=12, own_b=2, plan=pick_plan(), desc={"graph": "toy"}, walk="toy", u2s=lambda: ref,
                          u2s_names={"toy": True}, types={"toy": True})
    return (lambda name, carve, threads: (cx, ctx)), (lambda cx_, ctx_, js, verify=False: [facts[j] for j in js])


def _raises(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return True
    return False


def same_arrays(a, b):
    return set(a) == set(b) and all(
        np.asarray(a[k]).dtype == np.asarray(b[k]).dtype and np.asarray(a[k]).shape == np.asarray(b[k]).shape
        and np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes() for k in a)


def selftest_a1(tmp, plan, facts, op, rule):
    """PKO A1 (docs/PKO_A1_AMENDMENT.md): rows-based chunking and part sizes, the peak record, the RAM preflight, the
    carve order, the counts guard, and that nothing A0 filed or keyed changes. Toy folders only."""
    n, GiB = 0, GIB
    # (a) the chunk rule is look_x_six's: squad's 479 questions to a chunk stay 13 chunks; the KB carves have hundreds
    assert chunk_count(5856, 292_994) == shards(5856, 479) == 13
    assert chunk_count(1202, 5_773_701) == math.ceil(1202 / int(24000 // (5_773_701 / 1202))) == 301
    assert chunk_count(1202, 5_773_701) > 100 > shards(1202, 479)             # A0 said 3
    assert chunk_count(1, 10) == 1 and chunk_count(3, 3_000_000) == 3          # a pool above 24,000 nodes: one a chunk
    rows = {("webqsp", "fit"): 5_773_701, ("squad", "fit"): 293_027, ("musique", "s1fit"): 25_407_562,
            ("hotpotqa", "fit"): 4_228_755, ("2wiki", "fit"): 6_345_639, ("metaqa", "s1eval"): 42_759_516,
            ("squad", "s1eval"): 594_110, ("musique", "s1eval"): 13_347_116, ("hotpotqa", "s1eval"): 5_280_595,
            ("2wiki", "s1eval"): 13_462_003, ("webqsp", "s1eval"): 7_219_527}
    assert set(rows) == set(CARVES)
    for (nm, cv), r in rows.items():
        nl, nc = look_parts(nm, QUESTIONS[(nm, cv)], r)
        assert 1 <= nl <= chunk_count(QUESTIONS[(nm, cv)], r) and 1 <= nc <= chunk_count(QUESTIONS[(nm, cv)], r)
        assert r / nl <= LOOK_ROWS and r / nc <= PART_ROWS_BY[nm]              # the intended size, per carve
    assert look_parts("metaqa", 9785, 42_759_516) == (43, 36) and look_parts("webqsp", 1202, 5_773_701) == (6, 12)
    assert look_parts("webqsp", 1202, 5_773_701, look_rows=2_000_000, part_rows=300_000) == (3, 20)   # the flags
    assert look_parts("webqsp", 3, 3_000_000) == (3, 3)                        # never more than the carve's chunks
    n += 1
    # (d) A0's keys, arguments and outputs for the carve whose shard 0/2 is filed do not move; rows size only the RAM
    R = roots(tmp / "pko")
    pre = pre_steps(R, "webqsp", "s1eval", 2)
    assert [s_.key for s_ in pre] == ["pools:webqsp/s1eval:0/2", "pools:webqsp/s1eval:1/2", "merge:webqsp/s1eval",
                                      "check-merge:webqsp"]
    assert pre[0].argv == ["pools", "webqsp", "--carve", "s1eval", "--shard", "0/2", "--threads", "2", "--check"]
    assert pre[0].outs == [R.parts / "webqsp__s1eval__0of2.json", R.check_parts / "webqsp__s1eval__0of2.json"]
    assert pre[2].argv == ["merge", "webqsp", "--carve", "s1eval", "--parts", "2"]
    big = pre_steps(R, "webqsp", "s1eval", 2, rows=7_219_527)
    assert [(s_.key, s_.argv, s_.outs) for s_ in big] == [(s_.key, s_.argv, s_.outs) for s_ in pre]
    assert big[2].need > pre[2].need > 0
    # an entry A0 filed (no peak, no need) is still OK and is skipped; one with them is read the same; a changed file is not
    RA = roots(tmp / "a0ledger")
    f0 = RA.base / "toy" / "x.txt"
    f0.parent.mkdir(parents=True)
    f0.write_text("x\n", encoding="utf-8")
    write_json(RA.ledger, {"declared_in": DECLARED, "stage": STAGE, "entries": {"k": {
        "status": "OK", "files": {relb(RA, f0): sha_file(f0)}, "seconds": 1315.3, "rc": 0, "utc": "2026-10-29T16:17:09Z"}}})
    boom = lambda st: (_ for _ in ()).throw(AssertionError("an OK entry was run again"))        # noqa: E731
    assert entry_ok(RA, read_ledger(RA)["entries"]["k"]) and do_step(RA, step("k", [], [f0], 5 * GiB), False, boom)
    file_entry(RA, "k2", "OK", [f0], 2.0, 0, 123, 456)
    e2 = read_ledger(RA)["entries"]["k2"]
    assert e2["peak_rss_bytes"] == 123 and e2["need_bytes"] == 456 and entry_ok(RA, e2)
    file_entry(RA, "k3", "OK", [f0], 2.0)
    assert "peak_rss_bytes" not in read_ledger(RA)["entries"]["k3"]
    f0.write_text("changed\n", encoding="utf-8")
    assert not entry_ok(RA, read_ledger(RA)["entries"]["k"])
    f0.write_text("x\n", encoding="utf-8")
    n += 1
    # (b) the peak: a real child holding ~160 MiB is seen; do_step files what the runner leaves on the step
    if psutil is not None:
        child = subprocess.Popen([sys.executable, "-I", "-c", "x = bytearray(160 << 20)\n"
                                  "for i in range(0, len(x), 4096): x[i] = 1\nimport time; time.sleep(1.5)"])
        w = PeakWatch(child.pid, every=0.1)
        child.wait()
        peak = w.stop()
        assert 150 * 2 ** 20 <= peak < 2 * GiB
        st = step("help", ["-h"], [])
        assert run_step(RA, st, 2) == 0 and st.pid is None and st.peak_rss_bytes > 20 * 2 ** 20
        assert (RA.logs / "help.log").exists()
    seen = []

    def runner_peak(st):
        st.peak_rss_bytes = 7 * GiB
        seen.append(st.key)
        st.outs[0].write_text("y\n", encoding="utf-8")
        return 0

    st = step("peak", [], [RA.base / "toy" / "y.txt"], 3 * GiB)
    assert do_step(RA, st, False, runner_peak) and seen == ["peak"]
    ep = read_ledger(RA)["entries"]["peak"]
    assert ep["peak_rss_bytes"] == 7 * GiB and ep["need_bytes"] == 3 * GiB and entry_ok(RA, ep)
    n += 1
    # (c) the preflight: declared RAM must be available less what running steps have yet to grow into
    box = {"avail": 10 * GiB, "rss": 0}
    g = RamGuard(wait=0.3, avail=lambda: box["avail"], rss=lambda st_: box["rss"], poll=0.05)
    A, B = step("A", [], [], 6 * GiB), step("B", [], [], 6 * GiB)
    g.admit(A)
    t0 = time.time()
    assert _raises_any(g.admit, B, exc=RamRefused) and 0.25 <= time.time() - t0 < 3
    assert g.refused == ("B", 6 * GiB, 10 * GiB)
    box["rss"] = 5 * GiB                                                  # A has grown into 5 of its 6 GiB
    g.admit(B)
    g.done(A)
    g.done(B)
    g2 = RamGuard(wait=5, avail=lambda: box["avail"], rss=lambda st_: 0, poll=0.05)
    g2.admit(A)
    got = []
    th = threading.Thread(target=lambda: (g2.admit(B), got.append("in")))
    th.start()
    time.sleep(0.3)
    assert not got
    g2.done(A)                                                            # A's RAM is back: B starts
    th.join(5)
    assert got == ["in"]
    RamGuard(wait=0, avail=lambda: None).admit(step("C", [], [], 99 * GiB))        # it cannot measure: it admits
    # a lane: a step that does not fit stops the chain, runs nothing and files nothing; with room it runs
    RR = roots(tmp / "ram")
    calls = []

    def runner_ok(st_):
        calls.append(st_.key)
        for p in st_.outs:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(f"{st_.key}\n", encoding="utf-8")
        return 0

    def jobr(key, need):
        return SimpleNamespace(key=key, name=key, carve="c", est=lambda: 0,
                               steps=lambda: iter([step(key + "1", [], [RR.base / "toy" / f"{key}1.txt"], need)]))
    roomy = Disk(RR, reserve=0, free=lambda: 10 ** 15, used=lambda j: 0, poll=0.01)
    tight_ram = RamGuard(wait=0.1, avail=lambda: 2 * GiB, rss=lambda st_: 0, poll=0.02)
    assert not run_lanes(RR, [jobr("big", 6 * GiB)], 1, False, runner_ok, roomy, threading.Event(), tight_ram)
    assert not calls and tight_ram.refused[0] == "big1" and not read_ledger(RR)["entries"]
    assert run_lanes(RR, [jobr("fit", GiB)], 1, False, runner_ok, roomy, threading.Event(), tight_ram)
    assert calls == ["fit1"]
    n += 1

    # a job whose steps raise stops the chain (A0's lane ended silently and run_lanes said True)
    def bad_steps():
        yield step("ok1", [], [RR.base / "toy" / "ok1.txt"])
        raise SystemExit("counts")
    jb = SimpleNamespace(key="bad", name="bad", carve="c", est=lambda: 0, steps=bad_steps)
    assert not run_lanes(RR, [jb, jobr("after", 0)], 1, False, runner_ok, roomy, threading.Event())
    assert "after1" not in calls
    n += 1
    # the carve order: smallest rows first, a merged manifest's rows over the estimate
    rq = {"webqsp": 4803.0, "squad": 50.0, "musique": 5522.0, "hotpotqa": 713.0, "2wiki": 1071.0, "metaqa": 4370.0}
    order = small_first(R, CARVES, rq)
    assert [f"{a}/{b}" for a, b in order] == [
        "squad/fit", "squad/s1eval", "hotpotqa/fit", "hotpotqa/s1eval", "webqsp/fit", "2wiki/fit", "webqsp/s1eval",
        "musique/s1eval", "2wiki/s1eval", "musique/s1fit", "metaqa/s1eval"]
    write_json(R.pools / "squad__fit.json", {"rows": 10 ** 9})
    assert small_first(R, CARVES, rq)[-1] == ("squad", "fit")
    (R.pools / "squad__fit.json").unlink()
    n += 1
    # a carve's counts are fixed by its first file: a rerun with other counts is refused, the same counts are not
    RC = roots(tmp / "counts")
    write_json(RC.pools / "squad__fit.json", {"questions": 5856, "rows": 292_994})
    for p in (RC.look / "squad" / "fit" / "record_0of2.json", RC.look / "squad" / "fit" / "pools_1of2.json"):
        write_json(p, {"toy": 1})
    (RC.cache / "squad" / "fit" / "part_0of3").mkdir(parents=True)
    assert other_counts(RC, "squad", "fit", 2, 3) == [] and len(other_counts(RC, "squad", "fit", 1, 3)) == 2
    assert [p.name for p in other_counts(RC, "squad", "fit", 2, 4)] == ["part_0of3"]
    jc = carve_job(RC, "squad", "fit", {"squad": 50.0}, False, 2)
    assert _raises(lambda: list(jc.steps()))                                 # nl = 1, nc = 1: both filed under other counts
    jc2 = carve_job(RC, "squad", "fit", {"squad": 50.0}, False, 2, look_rows=150_000, part_rows=100_000)
    got = [s_.key for s_ in jc2.steps()]
    assert "look:squad/fit:1/2" in got and "cache:squad/fit:2/3" in got and "look:squad/fit:0/1" not in got
    n += 1

    # the merge: shards built by this file and by A0's (its declared sha) join; any other sha does not
    def two_shards(name_):
        RM = roots(tmp / name_)
        for i in range(2):
            pools_pass("toyds", "toyc", shard=(i, 2), R=RM, opener=op, rule=rule)
        return RM

    def set_script(RM, i, key, val):
        pj = RM.parts / f"toyds__toyc__{i}of2.json"
        d_ = read_json(pj)
        d_["scripts"][key] = val
        write_json(pj, d_)

    RM = two_shards("mrg0")
    assert ZPKO_KEY in read_json(RM.parts / "toyds__toyc__0of2.json")["scripts"]
    set_script(RM, 0, ZPKO_KEY, ZPKO_SAME[0])
    assert merge_carve("toyds", "toyc", 2, RM)["questions"] == len(facts)
    RM = two_shards("mrg1")
    set_script(RM, 0, ZPKO_KEY, "1" * 64)
    assert _raises(merge_carve, "toyds", "toyc", 2, RM)
    RM = two_shards("mrg2")
    other = next(k_ for k_ in read_json(RM.parts / "toyds__toyc__0of2.json")["scripts"] if k_ != ZPKO_KEY)
    set_script(RM, 0, other, "2" * 64)
    assert _raises(merge_carve, "toyds", "toyc", 2, RM)
    n += 1
    # the declared RAM: lane figures by dataset, a cache part with its rows, a merge with its shards
    assert need_bytes("look", "musique") == int(9.5 * GiB) and need_bytes("pools", "squad") == 5 * GiB
    assert abs(need_bytes("cache", "webqsp", None, 500_000) - int((1.6 + 2.9 * 0.5 + 1.0) * GiB)) <= 2
    assert abs(need_bytes("merge", "metaqa", 42_759_516) / GiB - (0.3 + 114 * 42_759_516 / GiB + 1.0)) < 1e-6
    ps = post_steps(R, "metaqa", "s1eval", 42_759_516, 9785, True)
    assert all(s_.need > 0 for s_ in ps) and {_kind(s_.key) for s_ in ps} == {
        "look", "cache", "rmatch", "zrc", "links", "typed", "kt", "record", "prune"}
    assert sum(_kind(s_.key) == "cache" for s_ in ps) == 36 and sum(_kind(s_.key) == "look" for s_ in ps) == 43
    n += 1
    return n


def _raises_any(fn, *a, exc=Exception, **k):
    try:
        fn(*a, **k)
    except exc:
        return True
    return False


def selftest():
    plan, n = pick_plan(), 0
    # the cut: k = min(|kt_cos|, k_q), kept <= k, nothing when today's fill is 0 (squad)
    kq, k, keep = kept_counts([0, 100, 5, 1000, 30], [0, 3, 10, 40, 0], [50, 20, 100, 7, 9], plan)
    assert (k == np.minimum([0, 100, 5, 1000, 30], kq)).all() and (keep <= k).all() and kq[0] == 0 == kq[4]
    n += 1
    facts = [_toy_fact(j, plan) for j in range(23)]
    ref = _toy_u2s(facts, 12, 2, plan)
    # U2s's columns read back each question's fields; a changed filed entry is caught
    assert all(not differ(u2s_fields(f), u2s_filed(ref, j, 12, 2, plan)) for j, f in enumerate(facts))
    bad = {k_: v.copy() for k_, v in ref.items()}
    bad["n_n2"][4, 2 * len(M.M_GRID) + 1] += 1
    bad["lens"][7, M.column(12, plan.r, plan.o)] += 3
    assert differ(u2s_fields(facts[4]), u2s_filed(bad, 4, 12, 2, plan)) == ["n_n2"]
    assert "lens" in differ(u2s_fields(facts[7]), u2s_filed(bad, 7, 12, 2, plan))
    n += 1
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        R = roots(tmp / "pko")
        op, rule = _toy_open(facts, ref)
        # the pools pass on two shards, the merge, the file's invariants, one-shot equality, no rebuild
        for i in range(2):
            pools_pass("toyds", "toyc", shard=(i, 2), R=R, opener=op, rule=rule)
        assert _raises(pools_pass, "toyds", "toyc", shard=(0, 2), R=R, opener=op, rule=rule)
        ent = merge_carve("toyds", "toyc", 2, R)
        filed, ent2, msha = load_filed("toyds", "toyc", R)
        one = part_arrays([f"q{j:03d}" for j in range(len(facts))], facts)
        assert same_arrays(filed, one) and ent2 == ent and ent["questions"] == len(facts)
        assert ent["rows"] == sum(np.union1d(f["B"], f["exp"]).size for f in facts)
        assert all(np.array_equal(p, np.union1d(f["B"], f["exp"])) for p, f in zip(pools_of(filed), facts))
        assert _raises(merge_carve, "toyds", "toyc", 2, R)
        broken = dict(filed, nodes=np.asarray(filed["nodes"]).copy())
        q = int(np.flatnonzero(np.asarray(filed["count"]) > 0)[0])
        broken["nodes"][int(filed["ptr"][q])] = int(filed["bs_nodes"][int(filed["bs_ptr"][q])])
        assert _raises(check_filed, broken)
        n += 1
        # section 5 (a) on U2s's carve: IDENTICAL then PASS; a changed filed entry: DIFFERENT, no PASS
        for i in range(2):
            pools_pass("toyds", "s1sel", shard=(i, 2), check=True, pools=False, R=R, opener=op, rule=rule)
        assert check_merge("toyds", 2, R) == "PASS" and read_json(R.check / "toyds.json")["verdict"] == "PASS"
        R2 = roots(tmp / "pko2")
        op2, rule2 = _toy_open(facts, bad)
        assert _raises(pools_pass, "toyds", "s1sel", shard=(0, 1), check=True, pools=False, R=R2, opener=op2,
                       rule=rule2)
        # question 7's longer kt_cos also moves its kept rows (k = min(|kt_cos|, k_q) = 15, not 12)
        assert read_json(R2.check_parts / "toyds__s1sel__0of1.json")["check"]["mismatches"] == \
            {"kept": 1, "lens": 1, "n_n2": 1}
        assert pools_pass("toyds", "s1sel", limit=5, check=True, R=R2, opener=op, rule=rule)["check"]["verdict"] \
            == "IDENTICAL" and not (R2.parts / "toyds__s1sel__0of1.json").exists()
        n += 1
        # KT in the cache's order (zu2d_pools.to_cache_order), then the record chain and the gate
        order = list(range(len(facts)))[::-1]
        pools = [np.union1d(f["B"], f["exp"]) for f in facts]
        cr = R.cache / "toyds" / "toyc" / "part_0of1" / "record.json"
        write_json(cr, {"toy": "cache"})

        def fake_rows(root, name, carve):
            return ([f"q{j:03d}" for j in order], np.asarray([pools[j].size for j in order], np.int32),
                    np.concatenate([pools[j] for j in order]).astype(np.int32), {"part_0of1": sha_file(cr)})

        kt_stage("toyds", "toyc", R, cache_rows=fake_rows)
        kt = np.load(R.kt / "toyds" / "toyc" / "kt.npy")
        assert kt.dtype == np.float16 and np.array_equal(kt, np.concatenate([facts[j]["kt"] for j in order]))
        assert _raises(kt_stage, "toyds", "toyc", R, cache_rows=fake_rows)
        for p in (R.look / "toyds" / "toyc" / "record_0of1.json", R.look / "toyds" / "toyc" / "pools_0of1.json",
                  R.links / "toyds" / "toyc" / "part_0of1" / "record.json"):
            write_json(p, {"toy": p.name})
        rec = record_carve("toyds", "toyc", R, step1=lambda name, carve, f: "toy")
        assert not rec["problems"] and len(rec["inputs"]) == 11     # pools 3, look 2, cache 1, links 1, KT 4
        for nm in checks_state(R):
            write_json(R.check / nm, {"verdict": "PASS"})
        assert gate(R, carves=(("toyds", "toyc"),)) == "READY"
        (R.look / "toyds" / "toyc" / "record_0of1.json").write_text("{}\n", encoding="utf-8")
        assert _raises(gate, R, carves=(("toyds", "toyc"),))
        assert read_json(R.gate)["verdict"] == "NOT_READY"
        n += 1
        # the ledger: a FAILED step stops the chain, a rerun skips what is OK, --retry resumes, changed files rerun
        RL = roots(tmp / "ledger")
        calls, fail = {}, {"a2"}

        def runner(st):
            calls[st.key] = calls.get(st.key, 0) + 1
            if st.key in fail:
                return 1
            for p in st.outs:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(f"{st.key} {calls[st.key]}\n", encoding="utf-8")
            return 0

        def job(key, keys, est=0):
            return SimpleNamespace(key=key, name=key, carve="c", est=lambda: est,
                                   steps=lambda: iter([step(k_, [], [RL.base / "toy" / f"{k_}.txt"]) for k_ in keys]))

        roomy = Disk(RL, reserve=0, free=lambda: 10 ** 15, used=lambda j: 0, poll=0.01)
        jobs = lambda: [job("A", ["a1", "a2", "a3"]), job("B", ["b1", "b2"])]  # noqa: E731
        assert not run_lanes(RL, jobs(), 1, False, runner, roomy, threading.Event()) and calls == {"a1": 1, "a2": 1}
        assert read_ledger(RL)["entries"]["a2"]["status"] == "FAILED"
        assert not run_lanes(RL, jobs(), 1, False, runner, roomy, threading.Event()) and calls == {"a1": 1, "a2": 1}
        fail.clear()
        assert run_lanes(RL, jobs(), 1, True, runner, roomy, threading.Event())
        assert calls == {"a1": 1, "a2": 2, "a3": 1, "b1": 1, "b2": 1}
        assert run_lanes(RL, jobs(), 2, False, runner, roomy, threading.Event()) and sum(calls.values()) == 6
        (RL.base / "toy" / "a1.txt").write_text("changed\n", encoding="utf-8")
        assert run_lanes(RL, jobs(), 2, False, runner, roomy, threading.Event()) and calls["a1"] == 2
        assert run_lanes(RL, [job("C", ["c1"]), job("D", ["d1"]), job("E", ["e1"])], 3, False, runner, roomy,
                         threading.Event()) and all(calls[k_] == 1 for k_ in ("c1", "d1", "e1"))
        n += 1
        # the disk guard: admits what fits, waits while a carve runs, stops when none does
        box, used = {"free": 100}, {}
        d = Disk(RL, reserve=10, free=lambda: box["free"], used=lambda j: used.get(j.key, 0), poll=0.01)
        X, Y, Z = (SimpleNamespace(key=k_) for k_ in "XYZ")
        assert d.admit(X, 50)
        got = []
        t = threading.Thread(target=lambda: got.append(d.admit(Y, 50)))
        t.start()
        time.sleep(0.2)
        assert not got
        used["X"], box["free"] = 30, 70
        time.sleep(0.2)
        assert not got
        d.done(X)
        t.join(5)
        assert got == [True]
        d.done(Y)
        assert not d.admit(Z, 1000)
        tight = Disk(RL, reserve=0, free=lambda: 10, used=lambda j: 0, poll=0.01)
        assert not run_lanes(RL, [job("F", ["f1"], est=11)], 1, False, runner, tight, threading.Event())
        assert "f1" not in calls
        n += 1
        # prune: refused while a reader of the look is not filed OK; then only the chunks go
        RP = roots(tmp / "prune")
        dl = RP.look / "toyds" / "toyc"
        (dl / "chunks").mkdir(parents=True)
        for i in range(3):
            np.savez(dl / "chunks" / f"c{i:04d}.npz", x=np.arange(5))
        write_json(dl / "record_0of1.json", {"toy": 1})
        crec = RP.cache / "toyds" / "toyc" / "part_0of1" / "record.json"
        lrec = RP.links / "toyds" / "toyc" / "part_0of1" / "record.json"
        write_json(crec, {"toy": 2})
        write_json(lrec, {"toy": 3})
        assert _raises(prune_carve, "toyds", "toyc", RP)
        file_entry(RP, "cache:toyds/toyc:0/1", "OK", [crec], 1.0)
        file_entry(RP, "links:toyds/toyc", "FAILED", [], 1.0, 1)
        assert _raises(prune_carve, "toyds", "toyc", RP) and len(list((dl / "chunks").glob("c*.npz"))) == 3
        file_entry(RP, "links:toyds/toyc", "OK", [lrec], 1.0)
        assert prune_carve("toyds", "toyc", RP) == 3 and not list((dl / "chunks").glob("c*.npz"))
        assert (dl / "record_0of1.json").exists() and (dl / "pruned.json").exists() and crec.exists()
        assert _raises(prune_carve, "toyds", "toyc", RP)
        n += 1
        # the bases: placed with their sha256; a basis that is not its json's is refused
        s1 = tmp / "s1"
        s1.mkdir()
        for b in ("2wiki", "hotpotqa"):
            np.savez(s1 / f"basis_{b}.npz", x=np.arange(4))
            write_json(s1 / f"basis_{b}.json", {"npz_sha256": sha_file(s1 / f"basis_{b}.npz")})
        placed = copy_bases(roots(tmp / "b1"), src=s1)
        assert len(placed) == 4 and all(sha_file(p) == sha_file(s1 / p.name) for p in placed)
        write_json(s1 / "basis_2wiki.json", {"npz_sha256": "0" * 64})
        assert _raises(copy_bases, roots(tmp / "b2"), src=s1)
        n += 1
        # squad against U2d new's records (k_q = 0, the pool is B): PASS on its sizes and nodes, FAIL on one node moved
        sq = []
        for j in range(17):
            f = _toy_fact(500 + j, plan)
            f.update(exp=np.zeros(0, np.int64), I=f["B"], fill=0, kq=0, k=0, keep=0, kt_len=0, n_i=0,
                     today_inc_size=int(f["B"].size), kt=f["kt"][:f["B"].size])
            sq.append(f)
        RS = roots(tmp / "sq")
        ops, rules = _toy_open(sq, None)
        pools_pass("squad", "fit", shard=(0, 1), R=RS, opener=ops, rule=rules)
        merge_carve("squad", "fit", 1, RS)
        sfiled, sent, _m = load_filed("squad", "fit", RS)
        ids_u = [f"q{j:03d}" for j in range(len(sq))][::-1]                    # U2d new's record names its own order
        u2d = tmp / "u2d_new"
        sizes = np.asarray([sq[int(q[1:])]["B"].size for q in ids_u], np.int32)
        nodes = np.concatenate([sq[int(q[1:])]["B"] for q in ids_u]).astype(np.int32)
        write_json(u2d / "cache" / "squad" / "fit" / "part_0of1" / "record.json",
                   {"ids": ids_u, "arrays": {"n": {"dtype": "int32", "shape": [sizes.size], "sha256": ZP.npy_sha(sizes)},
                                             "node": {"dtype": "int32", "shape": [nodes.size],
                                                      "sha256": ZP.npy_sha(nodes)},
                                             "xc": {"dtype": "float16", "shape": [1], "sha256": "0" * 64}}})
        bsz = np.asarray([f["B"].size for f in sq], np.float64)
        assert round(float(bsz.mean()), 1) != float(bsz.mean())               # A2: the toy mean needs the rounding to match

        def u2d_manifest(**over):                                             # written as U2d writes one (A2): means at one
            e = {"questions": len(sq), "changed": 0, "short": 0, "budget_mean": 0.0, "fill_want_mean": 0.0,   # decimal,
                 "today_inc_mean": round(float(bsz.mean()), 1), "graph_detail": sent["graph_detail"],         # B_q = the fill
                 "pool_mean": round(float(bsz.mean()), 1), "B_q": {"max": 0, "mean": 0.0, "p90": 0},
                 "rows": {"max": int(bsz.max()), "mean": round(float(bsz.mean()), 1), "p90": 1}, "I": "never read",
                 "caps": "never read", "parts": [{"rows": int(bsz.sum())}]}
            e.update(over)
            write_json(u2d / "pools" / "squad" / "pools.json", {"files": {"squad__fit": e}})
        u2d_manifest()
        assert check_squad("fit", RS, u2d=u2d, dry=True) is None
        assert check_squad("fit", RS, u2d=u2d) == "PASS"
        out = read_json(RS.check / "squad__fit.json")
        assert out["link"]["state"] == "REBUILD" and out["k_q_zero"] and not out["manifest_differs"]
        RS2 = roots(tmp / "sq2")
        pools_pass("squad", "fit", shard=(0, 1), R=RS2, opener=ops, rule=rules)
        merge_carve("squad", "fit", 1, RS2)
        u2d_manifest(changed=3)
        assert _raises(check_squad, "fit", RS2, u2d=u2d)
        assert read_json(RS2.check / "squad__fit.json")["manifest_differs"] == ["changed"]
        assert read_json(RS2.check / "squad__fit.json")["rule"] == RULE_B and _raises(check_squad, "fit", RS2, u2d=u2d)   # final
        u2d_manifest()                                                        # A2: a FAIL filed under A1's rule has no `rule`:
        f_ = read_json(RS2.check / "squad__fit.json")                         # it is kept aside and the check runs again
        f_.pop("rule")
        write_json(RS2.check / "squad__fit.json", f_)
        assert check_squad("fit", RS2, u2d=u2d) == "PASS"
        assert read_json(RS2.check / "squad__fit.json")["rule"] == RULE_B
        assert read_json(RS2.check / "squad__fit.a1-fail.json") == f_
        write_json(RS2.check / "squad__fit.json", dict(f_, verdict="FAIL"))   # a second move over the aside file stops
        assert _raises(check_squad, "fit", RS2, u2d=u2d)
        # real shape: squad/fit's totals (5,856 questions, 292,994 rows, maximum 54, no fill) summarised against the numbers
        # U2d filed for squad__fit (outputs/u2d/new/pools/squad/pools.json): exact means do not match them, A2's do
        szr = np.full(5856, 50, np.int64)
        szr[:48] = 54
        szr[48] = 52
        assert int(szr.sum()) == 292_994 and int(szr.max()) == 54
        real = squad_summary({"ids": np.arange(5856), "bs_size": szr, "count": np.zeros(5856, np.int64)},
                             {"changed": 0, "short": 0, "graph_detail": "g",
                              "beside": {"budget_mean": 0.0, "fill_want_mean": 0.0, "today_inc_mean": 292_994 / 5856}})
        u_real = {"today_inc_mean": 50.0, "pool_mean": 50.0, "B_q": {"max": 0, "mean": 0.0},
                  "rows": {"max": 54, "mean": 50.0}, "parts_rows": 292_994}
        assert all(close(real[k], v) for k, v in u_real.items()) and not close(292_994 / 5856, 50.0)
        n += 1
        # the chain's steps: shards, look shards and cache parts within the carve's chunks, prune last
        sts = pre_steps(R, "webqsp", "s1eval", 2) + post_steps(R, "webqsp", "s1eval", 3_000_000, 1503, True)
        keys = [s_.key for s_ in sts]
        assert keys[:2] == ["pools:webqsp/s1eval:0/2", "pools:webqsp/s1eval:1/2"] and "--check" in sts[0].argv
        assert "check-merge:webqsp" in keys and "typed:webqsp/s1eval" in keys and keys[-1] == "prune:webqsp/s1eval"
        # A1: 3M rows is 3 look shards of 1M and 6 cache parts of 0.5M (webqsp's part target), far under its 126 chunks
        assert sum(k_.startswith("look:") for k_ in keys) == 3 and sum(k_.startswith("cache:") for k_ in keys) == 6
        sq = post_steps(R, "squad", "fit", 292_994, 5856, False)
        assert not any(s_.key.startswith(("rmatch", "zrc", "typed")) for s_ in sq)
        n += 1
        n += selftest_a1(tmp, plan, facts, op, rule)
    log(f"zpko selftest: {n} checks pass")


# ── command line ─────────────────────────────────────────────────────────────

def shard_arg(s):
    i, k = (int(x) for x in s.split("/"))
    if not 0 <= i < k:
        raise argparse.ArgumentTypeError(f"bad shard {s}")
    return i, k


def carves_arg(s):
    return tuple(_pair(x.strip()) for x in s.split(",") if x.strip())


def main(argv=None):
    ap = argparse.ArgumentParser(description="PKO: U2s's pick O pools under WQ (docs/PKO_PICK_O_POOLS_UNDER_WQ.md)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--base", default=None, help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("plan")
    p.add_argument("--carves", type=carves_arg, default=None)
    p.add_argument("--sq", type=float, default=None, help="pools-pass seconds per question, for the hours column")
    p.add_argument("--look-rows", type=int, default=None, help="A1: rows per look shard (default LOOK_ROWS)")
    p.add_argument("--part-rows", type=int, default=None, help="A1: rows per cache part (default PART_ROWS_BY)")
    for c in ("check", "pools"):
        p = sub.add_parser(c)
        p.add_argument("dataset")
        p.add_argument("--threads", type=int, default=2)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--shard", type=shard_arg, default=None)
        if c == "pools":
            p.add_argument("--carve", required=True)
            p.add_argument("--check", action="store_true")
    p = sub.add_parser("check-merge")
    p.add_argument("dataset")
    p.add_argument("--parts", type=int, required=True)
    p = sub.add_parser("merge")
    p.add_argument("dataset")
    p.add_argument("--carve", required=True)
    p.add_argument("--parts", type=int, required=True)
    p = sub.add_parser("check-squad")
    p.add_argument("--carve", required=True)
    p.add_argument("--dry", action="store_true")
    sub.add_parser("copy-bases")
    for c, opt in (("look", "--shard"), ("cache", "--part")):
        p = sub.add_parser(c)
        p.add_argument("dataset")
        p.add_argument("--carve", required=True)
        p.add_argument(opt, type=shard_arg, required=True)
    p = sub.add_parser("build")
    p.add_argument("what", choices=("rmatch", "zrc", "links"))
    p.add_argument("dataset")
    p.add_argument("--carve", required=True)
    for c in ("typed", "kt", "record"):
        p = sub.add_parser(c)
        p.add_argument("dataset")
        p.add_argument("--carve", required=True)
    p = sub.add_parser("prune")
    p.add_argument("--carve", type=_pair, required=True)
    sub.add_parser("gate")
    p = sub.add_parser("chain")
    p.add_argument("--carves", type=carves_arg, default=None)
    p.add_argument("--lanes", type=int, default=1)
    p.add_argument("--prune", action="store_true")
    p.add_argument("--retry", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--look-rows", type=int, default=None, help="A1: rows per look shard (default LOOK_ROWS)")
    p.add_argument("--part-rows", type=int, default=None, help="A1: rows per cache part (default PART_ROWS_BY)")
    p.add_argument("--ram-wait", type=float, default=RAM_WAIT_S,
                   help="A1: seconds a step waits for its declared RAM before the chain stops (0: stop at once)")
    p.add_argument("--no-preflight", action="store_true", help="A1: start steps without the free-RAM check")
    p.add_argument("--as-given", action="store_true", help="A1: run the carves in the order given, not smallest first")
    p.add_argument("--checks-filed", action="store_true",
                   help="A1: section 5's checks are already PASS in this base; queue only the carves")
    p.add_argument("--disk-reserve-gib", type=float, default=RESERVE / GIB, help="free disk kept (default 15)")
    a = ap.parse_args(argv)
    if a.selftest:
        selftest()
        return 0
    R = roots(a.base)
    if a.cmd == "plan":
        plan_table(a.carves, a.sq, R, a.look_rows, a.part_rows)
    elif a.cmd == "check":
        pools_pass(a.dataset, u2s_carve(a.dataset), a.threads, a.limit, a.shard, check=True, pools=False, R=R)
    elif a.cmd == "pools":
        if (a.dataset, a.carve) not in CARVES:
            raise SystemExit(f"{a.dataset}/{a.carve} is not a WQ carve")
        pools_pass(a.dataset, a.carve, a.threads, a.limit, a.shard, check=a.check, pools=True, R=R)
    elif a.cmd == "check-merge":
        check_merge(a.dataset, a.parts, R)
    elif a.cmd == "merge":
        merge_carve(a.dataset, a.carve, a.parts, R)
    elif a.cmd == "check-squad":
        check_squad(a.carve, R, dry=a.dry)
    elif a.cmd == "copy-bases":
        copy_bases(R)
    elif a.cmd == "look":
        look_shard(a.dataset, a.carve, a.shard, R)
    elif a.cmd == "cache":
        cache_part(a.dataset, a.carve, a.part, R)
    elif a.cmd == "build":
        chain_build(a.what, a.dataset, a.carve, R)
    elif a.cmd == "typed":
        typed_carve(a.dataset, a.carve, R)
    elif a.cmd == "kt":
        kt_stage(a.dataset, a.carve, R)
    elif a.cmd == "record":
        record_carve(a.dataset, a.carve, R)
    elif a.cmd == "prune":
        prune_carve(*a.carve, R)
    elif a.cmd == "gate":
        gate(R)
    elif a.cmd == "chain":
        return chain(a.carves, a.lanes, a.retry, a.prune, a.threads, R, disk=Disk(R, reserve=a.disk_reserve_gib * GIB),
                     look_rows=a.look_rows, part_rows=a.part_rows,
                     ram=False if a.no_preflight else RamGuard(wait=a.ram_wait), as_given=a.as_given,
                     checks_filed=a.checks_filed)
    else:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
