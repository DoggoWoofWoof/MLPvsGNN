"""ABL (docs/ABL_LINK_FAMILY_ABLATION.md): what each link family carries, with the models of record fixed.

Three families, the same on every dataset: S (the original structural edges the store ships with), R (our rules as one
family: the rule graph W2_S together with the NER store) and K (nearest-neighbour links). An arm is a subset of {S, R, K}:
what fills the `structural` slot (S's graph, R's rule graph, their union, or nothing; zabl_g.py builds each and its types),
and whether the `ner` store (R) and the `knn` store (K) are in. A family that is out is an empty store (same node count,
no edge) in every process that opens it. The first stage, B, the seeds, the rule and its budgets are the record's. ML1's
pair-1 fits (zg6r, zrckp; seeds 0-2; both WQ splits) are forked and read on the arm's pools; nothing is refitted. No LLM,
no test split.

    python outputs/mp_unified/zabl.py --selftest
    python outputs/mp_unified/zabl.py pools --arm A --dataset D [--shard i/k] [--limit Q] [--host]
    python outputs/mp_unified/zabl.py step KIND --arm A --dataset D [--index i/n] [--shard i/k] [--host]
    python outputs/mp_unified/zabl.py chain --arm A --dataset D [--lanes N] [--host]      # the carve's whole chain
    python outputs/mp_unified/zabl.py gate --arm A [--datasets d1,d2] [--host]
    python outputs/mp_unified/zabl.py fork --arm A
    python outputs/mp_unified/zabl.py read --arm A --name FIT [--datasets d1,d2]           # the host's GPU
    python outputs/mp_unified/zabl.py prune --arm A                                         # look and cache go, reads stay
    python outputs/mp_unified/zabl.py check --name FIT                                      # check 5 (arm REC, musique)
    python outputs/mp_unified/zabl.py report
    python outputs/mp_unified/zabl.py items --arm A [--lanes N]                             # feeder rows, printed
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"                                  # BLAS at 2, as in every PKO process (zpko section 4)
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # ML1's reads (zml1s.where_ok)

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse
import concurrent.futures as cf
import contextlib
import copy
import json
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace

import numpy as np
import zabl_g as G
import zb1p as B

DECLARED = "docs/ABL_LINK_FAMILY_ABLATION.md"
STAGE = "ABL"
OUT = ROOT / "outputs" / "abl"
CARVE = "s1eval"
# the arm = what fills the structural slot ("" nothing, S, R, SR = their union), and whether the ner store (R's other
# half) and the knn store (K) are in. REC is the record on the passage datasets (R + K), for check 5 only.
ARM = {"SRK": ("SR", 1, 1), "SR": ("SR", 1, 0), "SK": ("S", 0, 1), "RK": ("R", 1, 1), "S": ("S", 0, 0),
       "R": ("R", 1, 0), "K": ("", 0, 1), "0": ("", 0, 0)}
CHECK_ARM = "REC"
# SWP (docs/SWP_S_TO_R_SWAP.md, 8 Nov): two arms with the ner and knn stores both in and the slot S or empty, so that the swap S -> R is RK
# against SNK with N and K fixed. They are not among the eight subsets of {S, R, K}, so the grid, its report and its self-test stay as filed.
ARM_SWP = {"SNK": ("S", 1, 1), "NK": ("", 1, 1)}
ARM_ALL = {**ARM, **ARM_SWP}
BUILD_ORDER = ("SRK", "SK", "SR", "S", "RK", "R", "K", "0")              # the user's asks first; RK on passage is ML1's
BASE_ARM = "SRK"
DATASETS = ("musique", "metaqa", "hotpotqa", "squad", "2wiki", "webqsp")   # the order an arm runs them (section 6)
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")
KB = ("webqsp", "metaqa")
SHARDS = {"webqsp": 2, "musique": 3, "hotpotqa": 3, "metaqa": 6, "squad": 3, "2wiki": 5}   # pools shards per carve
ARMS_ML = B.ARMS                                               # ("zg6r", "zrckp")
SEEDS, SPLITS, SUF, TRAINS = (0,), B.SPLITS, B.SUF, B.TRAINS       # amendment A3 (7 Nov): seed 0 only, B has 0-2
FITS = tuple(f"scr-ml1-{arm}-s{seed}{SUF[split]}" for arm in ARMS_ML for seed in SEEDS for split in SPLITS)   # 4 of B's 12
SCREEN_FITS = B.SCREEN_FITS
PYLIB_RUN = B.PYLIB_RUN
STEPS = ("pools", "merge", "look", "bases", "cache", "rmatch", "zrc", "links", "typed", "kt", "record")
NO_STEP1 = {"state": "not applicable: an ABL arm has no step-1 cache (docs/ABL_LINK_FAMILY_ABLATION.md)"}
BOOT = 2000
CHECK_FITS = ("scr-ml1-zg6r-s0-hp", "scr-ml1-zrckp-s0-hp")      # check 5: musique's in-domain fits of seed 0

log, utc, sha_file, sha_src, rel = B.log, B.utc, B.sha_file, B.sha_src, B.rel
read_json, write_json, Patches = B.read_json, B.write_json, B.Patches


def arm_dir(arm):
    return OUT / arm


def applies(arm):
    """The datasets an arm is built on: all six, except RK on the passage datasets, which is the record (ML1's reads)."""
    if arm not in ARM_ALL:
        raise SystemExit(f"ABL: arm {arm}: one of {sorted(ARM_ALL)}")
    return tuple(d for d in DATASETS if not (arm == "RK" and d in PASSAGE))


def dropped(arm):
    """The stores that are out among ner and knn."""
    _slot, ner, knn = ARM_ALL[arm]
    return tuple(f for f, k in (("ner", ner), ("knn", knn)) if not k)


def in_domain_splits(ds):
    return tuple(sp for sp in SPLITS if any(d == ds for d, _c in TRAINS[sp]))


def roots_of(M, arm):
    return M.PK.roots(arm_dir(arm))


# ── a family that is out, and the slot's content ─────────────────────────────

def empty_store(family, n):
    """A ner or knn store of n nodes with no edge."""
    from mp_retrieval.m3b_pools import FamilyStore
    return FamilyStore(family, int(n), np.zeros(n + 1, np.int64), np.zeros(0, np.int32),
                       weight=np.zeros(0, np.float16), edges_stored=0)


def slot_store(M, name, slot, sl, st0):
    """The structural store of a slot. A passage dataset: U2c's store of the slot graph (one relation). A KB dataset: the
    KB store's layout with the slot's collapsed rows and typed rows (S's typed rows where the slot holds S, none
    otherwise; the relation counts are S's, or zero without S)."""
    n = sl.ip.size - 1
    if name in PASSAGE:
        return M.U2C.csr_store(n, sl.ip, sl.col)
    if slot == "S":
        return st0
    from mp_retrieval.m3b_pools import FamilyStore
    has_s = sl.typed[1].size > 0
    return FamilyStore("structural", int(n), np.ascontiguousarray(sl.ip, np.int64), np.ascontiguousarray(sl.col, np.int32),
                       typed_indptr=sl.typed[0], typed_col=sl.typed[1], typed_rel=sl.typed[2], typed_dir=sl.typed[3],
                       rel_count=(st0.rel_count if has_s else np.zeros_like(st0.rel_count)),
                       edges_stored=int(sl.col.size) // 2)


@contextlib.contextmanager
def on_arm(M, name, arm):
    """The package's dataset opened with the arm's slot and stores, for zpko.pools_pass, look_shard, cache, links and KT
    (every patch put back on exit). A family that is out is asserted to hold no edge wherever it is loaded; the slot's
    graph, its types and U2s's record (the graph-dependent parts) are the arm's."""
    slot, _ner, _knn = ARM_ALL[arm]
    drop = dropped(arm)
    PK, S4, U2C, E, ZP = M.PK, M.S4, M.U2C, M.E, M.ZP
    passage = name in PASSAGE
    P_ = Patches()
    seen = {"empty": [], "slot": slot}
    cache = {}
    try:
        mp_mod = S4.m3b_pools
        orig_store = mp_mod.load_or_build_store
        orig_gc = U2C.graph_csr

        def ensure(n_):
            """A passage dataset's slot before any store is opened (the pools ask for the graph first)."""
            if "sl" not in cache:
                base = G.base_of(name, int(n_), gc=orig_gc)
                cache.update(sl=G.load_slot(base, slot, orig_at=orig_at), st0=None, base=base)
            return cache["sl"]

        def the_slot(ds_, cache_dir):
            if "sl" not in cache:
                n = int(ds_.n_nodes)
                st0 = None if passage else orig_store(ds_, "structural", cache_dir)
                base = G.base_of(name, n, store=st0, gc=orig_gc)
                sl = G.load_slot(base, slot, orig_at=orig_at)
                if slot == "" and sl.col.size:
                    raise SystemExit(f"ABL {arm}/{name}: the empty slot holds edges")
                cache.update(sl=sl, st0=st0, base=base)
            return cache["sl"]

        def load_store(ds_, family, cache_dir):
            if ds_.name != name:
                return orig_store(ds_, family, cache_dir)
            if family in drop:
                st = empty_store(family, int(ds_.n_nodes))
                if int(np.asarray(st.col).size):
                    raise SystemExit(f"ABL {arm}/{name}: the {family} store that is out holds edges")
                seen["empty"].append(family)
                return st
            if family != "structural":
                return orig_store(ds_, family, cache_dir)
            sl = the_slot(ds_, cache_dir)
            if "store" not in cache:
                cache["store"] = slot_store(M, name, slot, sl, cache["st0"])
            return cache["store"]

        P_.set(mp_mod, load_or_build_store=load_store)

        def union_pools(name_, carve_, pop, prep, bases):
            """Step 4e's union_pools without its frozen-pool check (an arm's pool differs from step 4d's by design):
            base and seeds are checked against step 4d's filed carve, question by question (ABL check 1)."""
            if name_ != name:
                raise SystemExit(f"ABL: union_pools({name_}); ABL opens {name} only")
            path = E.D4.POOLS / f"{name_}__{carve_}.npz"
            ent = json.loads(E.D4.MANIFEST.read_text(encoding="utf-8"))["files"][f"{name_}__{carve_}"]
            if E.sha_file(path) != ent["sha256"]:
                raise SystemExit(f"{path}: not step 4d's manifest's file")
            with np.load(path) as z:
                f = {k: z[k] for k in z.files}
            n = len(pop.ids)
            if f["ids"].tolist()[:n] != list(pop.ids):
                raise SystemExit(f"{name_}/{carve_}: step 4d's first {n} ids are not the population's")
            bs_all = []
            for j in range(n):
                bs = np.union1d(np.asarray(bases[j], np.int64), np.asarray(prep.seeds[j], np.int64))
                if not np.array_equal(bs, f["bs_nodes"][f["bs_ptr"][j]:f["bs_ptr"][j + 1]].astype(np.int64)):
                    raise SystemExit(f"{pop.ids[j]}: base and seeds are not step 4d's")
                bs_all.append(bs)
            seen["base_seeds_checked"] = seen.get("base_seeds_checked", 0) + n
            return None, None, bs_all, [np.asarray(p, np.int64) for p in prep.pools], None

        P_.set(E, union_pools=union_pools)
        if passage:
            def graph_csr(name_, graph, n_):
                if name_ != name or graph != "chosen":
                    return orig_gc(name_, graph, n_)
                sl = ensure(int(n_))
                return sl.ip.copy(), sl.col.copy(), copy.deepcopy(sl.desc)

            orig_u2e = ZP.u2e_graph

            def u2e_graph(name_):
                if name_ != name:
                    return orig_u2e(name_)
                return copy.deepcopy(cache["sl"].desc if "sl" in cache else G.slot_meta(name, slot)["desc"])

            P_.set(U2C, graph_csr=graph_csr)
            P_.set(ZP, u2e_graph=u2e_graph)
        orig_rec = PK.u2s_record

        def u2s_record(name_, v_=None):
            rec, npz = orig_rec(name_, v_)
            if name_ != name:
                return rec, npz
            meta = G.slot_meta(name, slot)
            out = dict(rec, em_types_sha256=meta["em_file_sha256"])
            if passage:
                out.update(graph_detail=copy.deepcopy(meta["desc"]), score_types_sha256=meta["score_file_sha256"])
            return out, npz

        P_.set(PK, u2s_record=u2s_record)
        orig_at, orig_p = PK.AT.load_types, PK.P.load_types

        def at_load(name_, desc=None):
            if name_ != name:
                return orig_at(name_, desc)
            if "sl" not in cache:
                raise SystemExit(f"ABL {arm}/{name}: the types asked for before the slot")
            if desc is not None and desc != (cache["sl"].desc if passage else cache["base"].rdesc):
                raise SystemExit(f"ABL {arm}/{name}: the types asked for on {desc}, not the slot's graph")
            return cache["sl"].at_load(name_, None)

        def p_load(d, desc=None):
            if d != name:
                return orig_p(d, desc)
            if "sl" not in cache or (desc is not None and desc != cache["sl"].desc):
                raise SystemExit(f"ABL {arm}/{name}: the score types asked for on {desc}, not the slot's graph")
            return cache["sl"].p_load(d, None)

        P_.set(PK.AT, load_types=at_load)
        P_.set(PK.P, load_types=p_load)
        yield SimpleNamespace(drop=drop, seen=seen, cache=cache)
    finally:
        P_.restore()


# ── the steps ─────────────────────────────────────────────────────────────────

def step_record(R, arm, name, kind, extra):
    write_json(Path(R.base) / "steps" / f"{name}__{CARVE}__{kind}.json",
               {"declared_in": DECLARED, "stage": STAGE, "arm": arm, "dataset": name, "carve": CARVE, "step": kind,
                "zabl_sha256": sha_src(__file__), **extra, "utc": utc()})


def pools(arm, name, shard=(0, 1), limit=0, host=False):
    M = B.mods(host)
    R = roots_of(M, arm)
    with B.checks(M), on_arm(M, name, arm) as A:
        res = M.PK.pools_pass(name, CARVE, threads=2, limit=int(limit or 0), shard=shard, R=R)
    log(f"pools {arm}/{name} {shard}: {res}; empty stores {A.seen['empty']}")
    return 0


def clean_look(R, name, nl):
    """A look shard that stopped before its record leaves chunk files no record lists; they are removed (ABL's own
    regenerable files) so the shard runs again on a clean folder."""
    import re
    d = Path(R.look) / name / CARVE
    if not (d / "chunks").exists():
        return
    base = Path(R.base).resolve()
    if not base.is_relative_to(OUT.resolve()) and not str(base).startswith(tempfile.gettempdir()):
        raise SystemExit(f"ABL: {rel(d)} is not under ABL's roots; nothing removed")
    listed = set()
    for r_ in d.glob(f"record_*of{nl}.json"):
        listed |= set(read_json(r_)["chunks"])
    stray = []
    for p in (d / "chunks").glob("c*.npz"):
        m = re.fullmatch(r"c(\d+)(\.tmp)?", p.stem)
        if m and (m.group(2) or int(m.group(1)) not in listed):
            stray.append(p)
    for p in stray:
        p.unlink()
    if stray:
        log(f"look {name}/{CARVE}: {len(stray)} chunk files of an unfinished shard removed")


def run_step(kind, arm, name, index=None, shard=None, host=False):
    M = B.mods(host)
    PK = M.PK
    R = roots_of(M, arm)
    t = time.time()
    if kind == "pools":
        return pools(arm, name, shard or (0, 1), 0, host)
    if kind == "merge":
        ent = PK.merge_carve(name, CARVE, SHARDS[name], R)
        step_record(R, arm, name, "merge", {"questions": ent["questions"], "rows": ent["rows"],
                                            "changed": ent["changed"], "short": ent["short"]})
    elif kind == "look":
        with B.checks(M), on_arm(M, name, arm):
            PK.look_shard(name, CARVE, index, R)
    elif kind == "bases":
        PK.copy_bases(R)
    elif kind == "cache":
        with B.checks(M), on_arm(M, name, arm):
            PK.cache_part(name, CARVE, index, R)
    elif kind in ("rmatch", "zrc", "links"):
        with B.checks(M), on_arm(M, name, arm):
            PK.chain_build(kind, name, CARVE, R)
    elif kind == "typed":
        with B.checks(M), on_arm(M, name, arm):
            PK.typed_carve(name, CARVE, R)
    elif kind == "kt":
        PK.kt_stage(name, CARVE, R)
    elif kind == "record":
        rec = PK.record_carve(name, CARVE, R, step1=lambda *_a: dict(NO_STEP1))
        step_record(R, arm, name, "record", {"inputs": len(rec["inputs"]), "problems": rec["problems"]})
    else:
        raise SystemExit(f"ABL: step {kind}: one of {STEPS}")
    log(f"step {kind} {arm}/{name} {index or shard or ''}: {time.time() - t:.0f} s")
    return 0


def drop_failed_record(R, st):
    """PKO's record step writes the record and then exits 1 when it names problems, and the chain would take that file
    for a filed record. A record that names problems is ABL's own file under ABL's roots: it is removed (its problems
    are logged) so the record step runs again. Returns the problems removed, or None."""
    p = Path(R.records) / f"{st}.json"
    if not p.exists():
        return None
    problems = read_json(p).get("problems")
    if not problems:
        return None
    base = Path(R.base).resolve()
    if not base.is_relative_to(OUT.resolve()) and not str(base).startswith(tempfile.gettempdir()):
        raise SystemExit(f"ABL: {rel(p)} is not under ABL's roots; nothing removed")
    p.unlink()
    log(f"record {st}: a failed attempt's record named problems {problems}; removed, the step runs again")
    return problems


def lower_priority():
    """Windows: BELOW_NORMAL for this process; every step and shard it starts inherits the class, so a GPU fit on the same host
    gets the cores first (7 Nov: a fit's epochs ran 400 s alone and 688 s beside a look step; 499-564 s once the look step was
    lowered). Scheduling only: no input, call or result changes."""
    if os.name != "nt":
        return False
    try:
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        return bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x4000))
    except Exception:
        return False


def chain(arm, name, lanes=1, host=False):
    """The carve's whole chain, each step its own process; the independent shards, look shards and cache parts of a
    stage run `lanes` at a time. A step already filed is skipped."""
    if host:
        log(f"chain {arm}/{name}: below-normal priority {'set' if lower_priority() else 'not set'}")
    M = B.mods(host)
    PK = M.PK
    R = roots_of(M, arm)
    base = ([sys.executable, str(PYLIB_RUN)] if host else [sys.executable]) + [str(Path(__file__).resolve())]
    tail = ["--host"] if host else []
    env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
               MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")

    def one(kind, flag=None):
        cmd = base + ["step", kind, "--arm", arm, "--dataset", name] + (flag or []) + tail
        t = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env, check=False).returncode
        if rc:
            raise SystemExit(f"ABL chain {arm}/{name}: {kind} {flag or ''} exited {rc}")
        log(f"chain {arm}/{name}: {kind} {flag or ''} done in {time.time() - t:.0f} s")

    def many(kind, flags):
        if not flags:
            return
        with cf.ThreadPoolExecutor(max_workers=max(1, int(lanes))) as ex:
            for fut in [ex.submit(one, kind, f) for f in flags]:
                fut.result()

    st = PK.stem(name, CARVE)
    man = R.pools / f"{st}.json"
    k = SHARDS[name]
    if not man.exists():
        many("pools", [["--shard", f"{i}/{k}"] for i in range(k)
                       if not (R.parts / f"{PK.stem(name, CARVE, (i, k))}.json").exists()])
        one("merge")
    ent = read_json(man)
    nl, nc = PK.look_parts(name, int(ent["questions"]), int(ent["rows"]))
    bad = PK.other_counts(R, name, CARVE, nl, nc)
    if bad:
        raise SystemExit(f"ABL chain {arm}/{name}: files of other shard or part counts: {[rel(p) for p in bad[:4]]}")
    todo = [i for i in range(nl) if not all(p.exists() for p in PK.look_outs(R, name, CARVE, (i, nl)))]
    if todo:
        clean_look(R, name, nl)
    many("look", [["--index", f"{i}/{nl}"] for i in todo])
    if not all((Path(R.cache) / f"basis_{b}.{x}").exists() for b in ("2wiki", "hotpotqa") for x in ("npz", "json")):
        one("bases")
    many("cache", [["--index", f"{i}/{nc}"] for i in range(nc)
                   if not (Path(R.cache) / name / CARVE / f"part_{i}of{nc}" / "record.json").exists()])
    chained = (name, CARVE) in PK.CHAINED

    def parts_filed(root):
        return all((Path(root) / name / CARVE / f"part_{i}of{nc}" / "record.json").exists() for i in range(nc))

    for what in (("rmatch", "zrc") if chained else ()) + ("links",):
        if not parts_filed(getattr(R, what)):
            one(what)
    if chained and not (Path(R.typed) / name / CARVE / "record.json").exists():
        one("typed")
    if not (Path(R.kt) / name / CARVE / "record.json").exists():
        one("kt")
    drop_failed_record(R, st)
    if not (Path(R.records) / f"{st}.json").exists():
        one("record")
    log(f"chain {arm}/{name}: filed ({ent['questions']} questions, {ent['rows']} rows, {nl} look shards, {nc} parts)")
    return 0


def built(arm):
    """The datasets whose carve record is filed under the arm."""
    return tuple(ds for ds in DATASETS if (arm_dir(arm) / "records" / f"{ds}__{CARVE}.json").exists())


def gate(arm, datasets=None, host=False):
    M = B.mods(host)
    R = roots_of(M, arm)
    datasets = tuple(datasets or built(arm))
    if not datasets:
        raise SystemExit(f"ABL {arm}: no carve is filed")
    with B.checks(M):
        verdict = M.PK.gate(R, carves=[(ds, CARVE) for ds in datasets])
    write_json(arm_dir(arm) / f"abl_gate_{'-'.join(datasets)}.json", {"declared_in": DECLARED, "stage": STAGE, "arm": arm,
                                                "pko_gate": rel(R.gate), "pko_gate_sha256": sha_file(R.gate),
                                                "verdict": verdict, "carves": list(datasets),
                                                "zabl_sha256": sha_src(__file__), "utc": utc()})
    return 0


# ── the fits and their reads ──────────────────────────────────────────────────

def fork(arm):
    for fit in FITS:
        src, dst = SCREEN_FITS / fit, OUT / "fits" / arm / fit
        want = sha_file(src / "models.pt")
        if (dst / "fork.json").exists():
            if read_json(dst / "fork.json")["models_sha256"] != want or sha_file(dst / "models.pt") != want:
                raise SystemExit(f"ABL: {rel(dst)} holds other models than {rel(src)}")
            continue
        tmp = dst.with_name(dst.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(src, tmp, ignore=lambda d, names, s=src: [x for x in names if Path(d) == s and x in ("reads", "read.json")])
        if sha_file(tmp / "models.pt") != want:
            raise SystemExit(f"ABL: {rel(tmp)}'s models.pt is not the source's")
        os.replace(tmp, dst)
        write_json(dst / "fork.json", {"declared_in": DECLARED, "fit": fit, "arm": arm, "source": rel(src),
                                       "models_sha256": want, "zabl_sha256": sha_src(__file__), "utc": utc()})
        log(f"fork {rel(src)} -> {rel(dst)}")
    return 0


def arm_roots(ML, arm):
    """ML1's roots and gate for an arm; REC (check 5) reads on the record's own."""
    if arm == CHECK_ARM:
        return dict(ML.ROOTS), ML.GATE
    b = f"outputs/abl/{arm}"
    return ({**ML.ROOTS, "pools": f"{b}/pools", "plook": f"{b}/look", "cache": f"{b}/cache", "rmatch": f"{b}/chains/rmatch",
             "zrc": f"{b}/chains/zrc", "links": f"{b}/links", "kt": f"{b}/kt"}, f"{b}/gate.json")


def read(arm, fit, datasets=None):
    """ML1's read of a fork on the arm's roots and gate (zml1s's code, no stamp); the reads land in the fork."""
    import zml1s as ML
    import zp1c3 as Z3
    import zwq as WQ
    if fit not in FITS or ML.ALL_FITS.get(fit) is None:
        raise SystemExit(f"ABL: {fit} is not one of ML1's zg6r or zrckp fits")
    mlarm, _seed, _split = ML.ALL_FITS[fit]
    datasets = tuple(datasets or (built(arm) if arm != CHECK_ARM else ("musique",)))
    if not datasets:
        raise SystemExit(f"ABL {arm}: no carve is filed")
    fdir = OUT / "fits" / arm / fit
    if not (fdir / "fork.json").exists():
        raise SystemExit(f"ABL: {rel(fdir)} is not forked (zabl.py fork --arm {arm})")
    outs = [fdir / "reads" / f"{ds}__{CARVE}.npz" for ds in datasets]
    if all(p.exists() for p in outs):
        log(f"read {arm}/{fit}: filed")
        return 0
    env = ML.device_env()
    ML.where_ok(env, "read")
    _rec, k = ML.fit_stamp(fit)
    if ML.device_of(env) != k.get("device"):
        raise SystemExit(f"ABL: {fit} was fitted on {k.get('device')}; this read runs on {ML.device_of(env)}")
    roots, gate_path = arm_roots(ML, arm)
    flags = ["--read", ",".join(f"{ds}={CARVE}" for ds in datasets), "--out-root", rel(OUT / "fits" / arm)]
    P_ = Patches()
    rm = Z3.RM
    orig_read = rm.read

    def rm_read(argv):
        argv = list(argv)
        for i in (0, 2):
            if flags[i] not in argv:
                argv += flags[i:i + 2]
        if argv[argv.index("--out-root") + 1] != flags[3] or argv[argv.index("--read") + 1] != flags[1]:
            raise SystemExit(f"ABL: the read's flags are not ABL's: {argv}")
        return orig_read(argv)

    t = time.time()
    try:
        P_.set(ML, ROOTS=roots, GATE=gate_path)
        if arm != CHECK_ARM:
            P_.set(ML, TYPED=f"outputs/abl/{arm}/typed")      # the chained carves' typed links are the arm's own
        P_.set(Z3, eval_carves=lambda: [(ds, CARVE) for ds in datasets])
        P_.set(rm, read=rm_read)
        if mlarm in ML.BASE_OF:
            rc, extra = ML.run_arm(fit, env, "read")
        else:
            argv = ["--name", fit, "--device", "cuda", "--host", "--cache-root", roots["cache"], "--rows-cap",
                    str(ML.ROWS_CAP[mlarm])] + flags
            ML._ACTIVE[0] = fit
            try:
                with ML.on_ml1():
                    WQ.cache_root_ok(roots["cache"])
                    WQ.gate_passed()
                    why = WQ.wq_why(fit)
                    if why:
                        raise SystemExit(f"ABL: {why}")
                    with WQ.on_wq() as W:
                        rc = W.Z3P.read(argv)
            finally:
                ML._ACTIVE[0] = None
            extra = {}
    finally:
        P_.restore()
    if rc:
        return rc
    miss = [rel(p) for p in outs if not p.exists()]
    if miss:
        raise SystemExit(f"ABL: the read wrote no {miss}")
    write_json(fdir / f"abl_read_{'-'.join(datasets)}.json",
               {"declared_in": DECLARED, "stage": STAGE, "fit": fit, "arm": arm, "ml_arm": mlarm, "carve": CARVE,
                "reads": {rel(p): sha_file(p) for p in outs}, "device": ML.device_of(env), "gate": gate_path,
                "extra": json.loads(json.dumps(extra, default=str)), "seconds": round(time.time() - t, 1),
                "zabl_sha256": sha_src(__file__), "utc": utc()})
    log(f"read {arm}/{fit}: {time.time() - t:.0f} s")
    return 0


def prune(arm, datasets=None):
    """A dataset's look and cache go once its four reads (seed 0; amendment A3) are filed (they cost lane-hours to regenerate, so only reads
    and records are required first); the pools, links, kt, records and reads stay. One dataset at a time keeps an arm's
    peak on disk to one dataset's footprint."""
    datasets = tuple(datasets or built(arm))
    missing = [f"{fit}/{ds}" for fit in FITS for ds in datasets
               if not (OUT / "fits" / arm / fit / "reads" / f"{ds}__{CARVE}.npz").exists()]
    if missing or not datasets:
        raise SystemExit(f"ABL: {arm} has {len(missing)} reads missing, e.g. {missing[:3]}; nothing pruned")
    freed = 0
    for ds in datasets:
        for sub in ("look", "cache"):
            d = arm_dir(arm) / sub / ds
            if d.exists():
                freed += sum(p.stat().st_size for p in d.rglob("*") if p.is_file())
                shutil.rmtree(d)
    log(f"prune {arm}/{','.join(datasets)}: {freed / 2**30:.1f} GiB of look and cache removed")
    return 0


def check_rec(fit, ds="musique", load_read=None, out=None):
    """ABL check 5: the passage record re-read through ABL's own read path (arm REC, forked fit, the record's roots and
    gate) equals ML1's filed read on the same carve, per question, to 1e-4 (ids, gold totals and every measure)."""
    import p3c_claim1 as P3
    load_read = load_read or P3.load_read
    mine = load_read(ROOT, f"outputs/abl/fits/{CHECK_ARM}/{fit}", ds)
    ml1 = load_read(ROOT, f"outputs/screen/fits/{fit}", ds)
    ok = (np.array_equal(mine["ids"], ml1["ids"]) and np.array_equal(mine["gt"], ml1["gt"])
          and mine["m"].shape == ml1["m"].shape)
    worst = float(np.max(np.abs(mine["m"] - ml1["m"]))) if ok else None
    ok = bool(ok and worst <= 1e-4)
    rec = {"declared_in": DECLARED, "stage": STAGE, "check": "REC re-read equals ML1's read (section 4.5)", "fit": fit,
           "dataset": ds, "carve": CARVE, "abl_read_sha256": mine["sha256"], "ml1_read_sha256": ml1["sha256"],
           "questions": int(mine["ids"].size), "max_abs_difference": worst, "equal_to_1e-4": ok,
           "zabl_sha256": sha_src(__file__), "utc": utc()}
    if out is None:
        write_json(arm_dir(CHECK_ARM) / f"abl_check_{fit}__{ds}.json", rec)
    log(f"check REC {fit}/{ds}: max |difference| {worst}; {'EQUAL' if ok else 'NOT EQUAL'}")
    return (0 if ok else 1) if out is None else rec


# ── the report ────────────────────────────────────────────────────────────────

def per_q(arm, fit, ds):
    """One fit's per-question R@5 on a carve, from the arm's fork (RK on a passage dataset is the record: ML1's filed read)."""
    import p3c_claim1 as P3
    fdir = f"outputs/screen/fits/{fit}" if (arm == "RK" and ds in PASSAGE) else f"outputs/abl/fits/{arm}/{fit}"
    if not (ROOT / fdir / "reads" / f"{ds}__{CARVE}.npz").exists():
        return None
    r = P3.load_read(ROOT, fdir, ds)
    return r["ids"], r["gt"], r["m"][:, 0], r["sha256"]


def cell(arm, ds, mlarm, regime):
    """Per-question R@5 of the seed mean over the model's fits in the regime (in-domain or held out)."""
    splits = [sp for sp in SPLITS if (sp in in_domain_splits(ds)) == (regime == "in-domain")]
    fits = [f"scr-ml1-{mlarm}-s{sd}{SUF[sp]}" for sp in splits for sd in SEEDS]
    got = [per_q(arm, f, ds) for f in fits]
    if not fits or any(g is None for g in got):
        return None
    ids = got[0][0]
    if any(not np.array_equal(g[0], ids) for g in got):
        raise SystemExit(f"ABL: {arm}/{ds}: the fits' questions differ")
    return {"ids": ids, "gt": got[0][1], "per_fit": [g[2] for g in got], "sha256": [g[3] for g in got], "fits": fits}


def compare(a, b):
    """b minus a on the questions with a gold, over the fits' mean: point, 95% interval (paired bootstrap over
    questions), the call, and in how many of the fits b is below a."""
    import p3c_claim1 as P3
    ok = a["gt"] > 0
    if not np.array_equal(a["ids"], b["ids"]):
        raise SystemExit("ABL: two arms' questions differ")
    ma, mb = np.mean(np.stack(a["per_fit"]), 0)[ok], np.mean(np.stack(b["per_fit"]), 0)[ok]
    d = mb - ma
    boots = np.concatenate([d[idx].mean(keepdims=True) for idx in P3.resamples(d.size, 20260706, BOOT)])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    call = "BELOW" if hi < 0 else "ABOVE" if lo > 0 else "LEVEL"
    below = sum(int(np.mean((fb - fa)[ok]) < 0) for fa, fb in zip(a["per_fit"], b["per_fit"]))
    return {"a": float(ma.mean()), "b": float(mb.mean()), "diff": float(d.mean()), "ci": [float(lo), float(hi)],
            "call": call, "fits_below": below, "fits": len(a["per_fit"]), "questions": int(ok.sum())}


ADDS = (("S adds (SRK - RK)", "RK", "SRK"), ("R adds (SRK - SK)", "SK", "SRK"), ("K adds (SRK - SR)", "SR", "SRK"),
        ("S alone (S - 0)", "0", "S"), ("R alone (R - 0)", "0", "R"), ("K alone (K - 0)", "0", "K"))


def fmt(c):
    return (f"{100 * c['diff']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}) | {c['call']} | "
            f"{c['fits_below']}/{c['fits']}")


def report():
    rec = {"declared_in": DECLARED, "stage": STAGE, "boot": BOOT, "base": BASE_ARM, "arms": {}, "adds": {}, "gnn_minus_mlp": {},
           "missing": []}
    cap = ("R@5 x100 on each development carve (s1eval), the pair-1 fits of ML1 fixed, seed 0 only (amendment A3; both splits where a "
           "dataset is read under both); differences in points with a 95% interval from 2,000 paired bootstrap resamples "
           "over questions; call = BELOW / LEVEL / ABOVE; 'below' = fits (of those read) where the first is below the second.")
    lines = [f"# ABL report (declared in {DECLARED})", "", cap, "", f"## Arm minus {BASE_ARM} (all three families)", "",
             f"| arm | dataset | regime | model | {BASE_ARM} | arm | diff (CI) | call | below |",
             "|---|---|---|---|---:|---:|---|---|---|"]
    for arm in BUILD_ORDER[1:]:
        for ds in DATASETS:
            for regime in ("in-domain", "held out"):
                for mlarm in ARMS_ML:
                    a, b = cell(BASE_ARM, ds, mlarm, regime), cell(arm, ds, mlarm, regime)
                    if a is None or b is None:
                        if a is not None and ds in applies(arm) and (OUT / "fits" / arm).exists():
                            rec["missing"].append(f"{arm}/{ds}/{mlarm}/{regime}")
                        continue
                    c = compare(a, b)
                    rec["arms"].setdefault(arm, {}).setdefault(ds, {}).setdefault(regime, {})[mlarm] = c
                    lines.append(f"| {arm} | {ds} | {regime} | {mlarm} | {100 * c['a']:.1f} | {100 * c['b']:.1f} | {fmt(c)} |")
    lines += ["", "## What each family adds (the second arm minus the first)", "",
              "| effect | dataset | regime | model | first | second | diff (CI) | call | below |",
              "|---|---|---|---|---:|---:|---|---|---|"]
    for label, x, y in ADDS:
        for ds in DATASETS:
            for regime in ("in-domain", "held out"):
                for mlarm in ARMS_ML:
                    a, b = cell(x, ds, mlarm, regime), cell(y, ds, mlarm, regime)
                    if a is None or b is None:
                        continue
                    c = compare(a, b)
                    rec["adds"].setdefault(label, {}).setdefault(ds, {}).setdefault(regime, {})[mlarm] = c
                    lines.append(f"| {label} | {ds} | {regime} | {mlarm} | {100 * c['a']:.1f} | {100 * c['b']:.1f} | {fmt(c)} |")
    lines += ["", "## GNN minus MLP, per arm (zg6r minus zrckp)", "",
              "| arm | dataset | regime | zrckp | zg6r | diff (CI) | call | below |", "|---|---|---|---:|---:|---|---|---|"]
    for arm in BUILD_ORDER:
        for ds in DATASETS:
            for regime in ("in-domain", "held out"):
                m, g_ = cell(arm, ds, "zrckp", regime), cell(arm, ds, "zg6r", regime)
                if m is None or g_ is None:
                    continue
                c = compare(m, g_)
                rec["gnn_minus_mlp"].setdefault(arm, {}).setdefault(ds, {})[regime] = c
                lines.append(f"| {arm} | {ds} | {regime} | {100 * c['a']:.1f} | {100 * c['b']:.1f} | {fmt(c)} |")
    rec["zabl_sha256"] = sha_src(__file__)
    rec["utc"] = utc()
    write_json(OUT / "report.json", rec)
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"report: {rel(OUT / 'report.md')}; {len(rec['missing'])} cells missing")
    return 0


# ── the feeder rows ───────────────────────────────────────────────────────────

# the code the host needs (B1P's rows' list, which holds the closure of what zb1p and zpko import) and ABL's own files
CODE = [
    "outputs/mp_unified/chaincov.py",
    "outputs/mp_unified/lean_cache.py",
    "outputs/mp_unified/lean_gpu.py",
    "outputs/mp_unified/lean_host.py",
    "outputs/mp_unified/lean_mlp.py",
    "outputs/mp_unified/lean_mlp2.py",
    "outputs/mp_unified/lean_mlp3.py",
    "outputs/mp_unified/lean_mlp4.py",
    "outputs/mp_unified/lean_mlp5.py",
    "outputs/mp_unified/lean_mlp7.py",
    "outputs/mp_unified/lean_mlp7g.py",
    "outputs/mp_unified/lean_mlp8.py",
    "outputs/mp_unified/lean_score.py",
    "outputs/mp_unified/lean_screen.py",
    "outputs/mp_unified/lean_screen2.py",
    "outputs/mp_unified/lean_screen3.py",
    "outputs/mp_unified/lean_screen4.py",
    "outputs/mp_unified/lean_screen5.py",
    "outputs/mp_unified/look_p1c.py",
    "outputs/mp_unified/look_p1c2.py",
    "outputs/mp_unified/look_step1.py",
    "outputs/mp_unified/look_x_six.py",
    "outputs/mp_unified/relcols.py",
    "outputs/mp_unified/relz.py",
    "outputs/mp_unified/rmatch.py",
    "outputs/mp_unified/screen_gate.py",
    "outputs/mp_unified/screen_pair.py",
    "outputs/mp_unified/screen_recall.py",
    "outputs/mp_unified/u1b.py",
    "outputs/mp_unified/u1d.py",
    "outputs/mp_unified/zg2.py",
    "outputs/mp_unified/zg6.py",
    "outputs/mp_unified/zg6b.py",
    "outputs/mp_unified/zg6b2.py",
    "outputs/mp_unified/zg6r.py",
    "outputs/mp_unified/zg6rw.py",
    "outputs/mp_unified/zgs.py",
    "outputs/mp_unified/zkg.py",
    "outputs/mp_unified/zlink.py",
    "outputs/mp_unified/zml1.py",
    "outputs/mp_unified/zml1s.py",
    "outputs/mp_unified/zp1c2.py",
    "outputs/mp_unified/zp1c3.py",
    "outputs/mp_unified/zp1c3p.py",
    "outputs/mp_unified/zp1h.py",
    "outputs/mp_unified/zprop.py",
    "outputs/mp_unified/zrc.py",
    "outputs/mp_unified/zrm.py",
    "outputs/mp_unified/zu2b.py",
    "outputs/mp_unified/zu2c.py",
    "outputs/mp_unified/zu2d.py",
    "outputs/mp_unified/zu2d_pools.py",
    "outputs/mp_unified/zu2e.py",
    "outputs/mp_unified/zu2n.py",
    "outputs/mp_unified/zwq.py",
    "outputs/mp_unified/zwqk.py",
    "outputs/host_ops/retry_cmd.py",
    "outputs/host_ops/cuda_alloc.py",
    "outputs/mp_unified/ziso.py",
    "outputs/mp_unified/zdep.py",
    "docs/DEP_GNN_DEPTH.md",
    "outputs/mp_unified/verdict_gate.py",
    "outputs/mp_unified/zrtr.py",
    "docs/RTR_GO_BACK_ROUNDS.md",
    "outputs/mp_unified/ltwdiag.py",
    "docs/LTW_LEARNED_NEXT_HOP.md",
    "outputs/mp_unified/chndiag.py",
    "docs/CHN_CHAIN_COMPLEMENT.md",
    "src/mp_retrieval/m3b_features.py",
    "scripts/m3b_compile.py",
    "outputs/mp_unified/zu2s.py",
    "outputs/mp_unified/look_u2d.py",
    "scripts/p1c3_kt.py",
    "scripts/p1e_anchor_kind.py",
    "scripts/p1gb_pools.py",
    "outputs/mp_unified/zdob.py",
    "outputs/mp_unified/ltmcols.py",
    "outputs/mp_unified/zltm.py",
    "outputs/mp_unified/zltg.py",
    "outputs/host_ops/pylib_run.py",
    "outputs/host_ops/pylib/numba067_llvmlite049_cp313_win_amd64.zip",
    "outputs/pkoh/inputs.json",
    "outputs/mp_unified/look_b1.py",
    "outputs/mp_unified/look_b1u.py",
    "outputs/mp_unified/b1f.py",
    "outputs/mp_unified/b1b.py",
    "outputs/mp_unified/u1a.py",
    "outputs/mp_unified/zabl.py",
    "scripts/p3c_claim1.py",
    "outputs/mp_unified/zpko.py",
    "outputs/mp_unified/zb1p.py",
    "outputs/mp_unified/zpkoh.py",
    "outputs/mp_unified/zabl_g.py",
    "outputs/mp_unified/zabl_u.py",
    "outputs/mp_unified/zu2m_a_types.py",
    "outputs/mp_unified/zu2m_a.py",
    "outputs/mp_unified/zu2p.py",
    "outputs/mp_unified/zu2k.py",
]
READ = ("python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.27 "
        "--gc 0.9 -- outputs/mp_unified/zabl.py")
SMALL = ("webqsp", "musique")                          # carves whose look and cache are a few GB (1.5k and 2.4k questions)
LANE_GB = {"webqsp": 7.0, "musique": 9.5, "hotpotqa": 9.5, "metaqa": 9.5, "squad": 7.0, "2wiki": 9.5}
G_GB = {"webqsp": 40, "metaqa": 20, "musique": 20, "squad": 20, "hotpotqa": 40, "2wiki": 50}   # the slot graphs and types


def g_items():
    """The slot graphs and their types, once per dataset (zabl_g.py all): the KB datasets wait on ABL-U's build."""
    ins = ";".join(CODE)
    rows = []
    for ds in DATASETS:
        dep = f"abl-u-{ds}" if ds in KB else ""
        rows.append(f"line|x|{dep}|abl-g-{ds}|2|{G_GB[ds]}|{ins}|python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- "
                    f"python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl_g.py all {ds} --host")
    return rows


def rec_items():
    """Check 5's rows: the passage record forked and re-read on musique/s1eval (seed 0, in-domain, both models) through
    ABL's read path, then compared to ML1's filed read."""
    ins = ";".join(CODE)
    rows = [f"line|x||abl-REC-fork|1|2|{ins}|python outputs/mp_unified/zabl.py fork --arm {CHECK_ARM}"]
    names = []
    for f in CHECK_FITS:
        n = f"abl-REC-read-{f.replace('scr-ml1-', '')}"
        names.append(n)
        rows.append(f"line|x|abl-REC-fork|{n}|1.1|6|{ins}|{READ} read --arm {CHECK_ARM} --name {f} --datasets musique|gpu:0.29")
    for f, n in zip(CHECK_FITS, names):
        rows.append(f"line|x|{n}|abl-REC-check-{f.replace('scr-ml1-', '')}|1|2|{ins}|python outputs/mp_unified/zabl.py check --name {f}")
    return rows


def items(arm, lanes=6):
    """Feeder rows (outputs/host_ops/hostfeed_items.txt format) for one arm. The arm's datasets run one after another
    (DATASETS' order) so that its disk peak is one dataset's: chain, then the gate on that dataset, then the four
    reads (seed 0; amendment A3) on the GPU, then the prune of its look and cache; the next dataset's chain waits on that prune. A chain also
    waits on the dataset's slot graphs (abl-g-<D>) when the arm's slot is not empty. Arms are independent of one another
    and run side by side as the host's lanes allow."""
    if arm == CHECK_ARM:
        return rec_items()
    dss = applies(arm)
    slot = ARM_ALL[arm][0]
    py = "python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py"
    retry = "python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --"
    ins = ";".join(CODE)
    rows = [f"line|x||abl-{arm}-fork|1|2|{ins}|python outputs/mp_unified/zabl.py fork --arm {arm}"]
    prev = ""
    for ds in dss:
        n = f"abl-{arm}-{ds}"
        deps = ",".join(x for x in (prev, f"abl-g-{ds}" if slot else "", f"abl-u-{ds}" if ds in KB and "R" in slot else "") if x)
        rows.append(f"line|x|{deps}|{n}|{2 * lanes}|{lanes * LANE_GB[ds]:g}|{ins}|{retry} {py} chain --arm {arm} --dataset {ds} --lanes {lanes} --host")
        g = f"abl-{arm}-gate-{ds}"
        rows.append(f"line|x|{n}|{g}|2|8|{ins}|{py} gate --arm {arm} --datasets {ds} --host")
        reads = []
        for fit in FITS:
            r = f"abl-{arm}-read-{ds}-{fit.replace('scr-ml1-', '')}"
            reads.append(r)
            rows.append(f"line|x|{g},abl-{arm}-fork|{r}|1.1|6|{ins}|{READ} read --arm {arm} --name {fit} --datasets {ds}|gpu:0.29")
        pr = f"abl-{arm}-prune-{ds}"
        rows.append(f"line|x|{','.join(reads)}|{pr}|1|2|{ins}|python outputs/mp_unified/zabl.py prune --arm {arm} --datasets {ds}")
        prev = g if ds in SMALL else pr            # a small carve's files are a few GB: the next chain need not wait for the card
    return rows


def selftest():
    assert set(ARM_SWP) == {"SNK", "NK"} and not set(ARM_SWP) & set(ARM) and dropped("SNK") == () and dropped("NK") == ()
    assert ARM_ALL["SNK"] == ("S", 1, 1) and ARM_ALL["NK"] == ("", 1, 1) and applies("SNK") == DATASETS == applies("NK")
    assert len(ARM) == 8 and len(set(ARM.values())) == 8 and set(BUILD_ORDER) == set(ARM) and len(BUILD_ORDER) == 8
    for arm, (slot, ner, knn) in ARM.items():          # the eight subsets of {S, R, K}, each once, named by its letters
        letters = ({"S"} if "S" in slot else set()) | ({"R"} if "R" in slot else set()) | ({"K"} if knn else set())
        assert ner == int("R" in slot), arm              # the ner store is R's other half
        assert letters == (set() if arm == "0" else set(arm)), arm
    assert len({frozenset(set(a) - {"0"}) for a in ARM}) == 8
    assert dropped("SRK") == () and dropped("SR") == ("knn",) and dropped("SK") == ("ner",) and dropped("RK") == ()
    assert dropped("S") == ("ner", "knn") and dropped("R") == ("knn",) and dropped("K") == ("ner",) and dropped("0") == ("ner", "knn")
    assert len(set(DATASETS)) == 6 and set(PASSAGE) | set(KB) == set(DATASETS) and not set(PASSAGE) & set(KB)
    assert applies("SRK") == DATASETS and applies("RK") == ("metaqa", "webqsp") and applies("0") == DATASETS
    assert in_domain_splits("musique") == ("WQ-hotpotqa",) and in_domain_splits("hotpotqa") == ("WQ-musique",)
    assert in_domain_splits("metaqa") == () and set(in_domain_splits("2wiki")) == set(SPLITS)
    assert set(in_domain_splits("webqsp")) == set(SPLITS) and set(in_domain_splits("squad")) == set(SPLITS)
    assert len(FITS) == 4 and set(FITS) <= set(B.FITS) and SEEDS == (0,) and all(k in SHARDS for k in DATASETS) and set(CHECK_FITS) <= set(FITS)
    # an empty ner or knn store has no edge and the right size
    st = empty_store("ner", 5)
    assert st.indptr.shape == (6,) and st.col.size == 0 and st.weight.size == 0
    rows = items("SRK", 6)
    assert len(rows) == 1 + 6 * (1 + 1 + 4 + 1) and all(r.count("|") >= 7 for r in rows)
    assert sum("gpu:" in r for r in rows) == 24 and all(r.startswith("line|x|") for r in rows)
    names = [r.split("|")[3] for r in rows]
    assert len(names) == len(set(names)), "item names are unique"
    by = {r.split("|")[3]: r.split("|")[2].split(",") for r in rows}
    # the datasets run one after another: each chain waits on the previous dataset's prune (or gate), and on its slot graphs
    assert by["abl-SRK-musique"] == ["abl-g-musique"] and by["abl-SRK-metaqa"] == ["abl-SRK-gate-musique", "abl-g-metaqa", "abl-u-metaqa"]
    assert by["abl-SRK-webqsp"] == ["abl-SRK-prune-2wiki", "abl-g-webqsp", "abl-u-webqsp"]
    assert all(d in names[:names.index(n)] or d.startswith(("abl-g-", "abl-u-"))
               for n, ds_ in by.items() for d in ds_ if d)
    k = items("K", 6)
    assert not any("abl-g-" in r.split("|")[2] for r in k), "the empty slot needs no slot graph"
    assert len(items("RK", 6)) == 1 + 2 * 7 and len(items("0", 6)) == 1 + 6 * 7
    g = g_items()
    assert [r.split("|")[3] for r in g] == [f"abl-g-{d}" for d in DATASETS] and g[-1].split("|")[2] == "abl-u-webqsp" and g[1].split("|")[2] == "abl-u-metaqa"
    r5 = items(CHECK_ARM)
    assert len(r5) == 1 + 2 + 2 and sum("gpu:" in r for r in r5) == 2 and all(r.startswith("line|x|") for r in r5)
    # check 5 on fake reads: equal reads pass, a 1e-3 change fails
    def fake(delta=0.0):
        def lr(_root, fit, _ds):
            m = np.full((4, 3), 0.5) + (delta if "abl" in fit else 0.0)
            return {"ids": np.array(list("abcd")), "gt": np.ones(4, np.int64), "m": m, "sha256": "x"}
        return lr
    assert check_rec("f", "d", fake(), out=True)["equal_to_1e-4"]
    assert not check_rec("f", "d", fake(1e-3), out=True)["equal_to_1e-4"]
    # cell(): which fits make the in-domain and held-out reads of a dataset (fit names are ML1's), with per_q faked
    seen = []

    def fake_per_q(arm_, fit_, ds_):
        seen.append(fit_)
        return np.array(list("abc")), np.ones(3, np.int64), np.full(3, 0.5), "sha"
    real_pq, g_ = per_q, globals()
    g_["per_q"] = fake_per_q
    try:
        mi = cell("SK", "musique", "zg6r", "in-domain")
        mh = cell("SK", "musique", "zg6r", "held out")
        wi = cell("SK", "webqsp", "zrckp", "in-domain")
        wh = cell("SK", "webqsp", "zrckp", "held out")
        ki = cell("SK", "metaqa", "zg6r", "in-domain")
        kh = cell("SK", "metaqa", "zg6r", "held out")
    finally:
        g_["per_q"] = real_pq
    assert mi and mh and wi and kh and wh is None and ki is None, "webqsp has no held-out fits; metaqa has no in-domain fits"
    assert len(mi["fits"]) == 1 and len(mh["fits"]) == 1 and len(wi["fits"]) == 2 and len(kh["fits"]) == 2
    assert set(mi["fits"]) | set(mh["fits"]) <= set(FITS) and not set(mi["fits"]) & set(mh["fits"])
    assert all(f.endswith("-hp") for f in mi["fits"]) != all(f.endswith("-hp") for f in mh["fits"])
    # the paired comparison: identical cells are LEVEL with difference 0; a cell 0.1 lower in every question is BELOW
    ids = np.array([str(i) for i in range(200)])
    rng = np.random.default_rng(0)
    base = rng.random(200)
    a = {"ids": ids, "gt": np.ones(200, np.int64), "per_fit": [base, base]}
    b = {"ids": ids, "gt": np.ones(200, np.int64), "per_fit": [base - 0.1, base - 0.1]}
    c0, c1 = compare(a, a), compare(a, b)
    assert c0["call"] == "LEVEL" and abs(c0["diff"]) < 1e-12 and c1["call"] == "BELOW" and c1["fits_below"] == 2
    assert abs(c1["diff"] + 0.1) < 1e-9
    # the chained carves (webqsp, metaqa) have PKO's rmatch, zrc and typed steps in ABL's chain, in PKO's order; a record a
    # failed attempt left (it names problems) is removed so the record step runs again, a clean one is kept
    assert STEPS.index("cache") < STEPS.index("rmatch") < STEPS.index("zrc") < STEPS.index("links") < STEPS.index("typed") \
        < STEPS.index("kt") < STEPS.index("record") and len(set(STEPS)) == len(STEPS)
    with tempfile.TemporaryDirectory() as td:
        R_ = SimpleNamespace(base=Path(td), records=Path(td) / "records")
        R_.records.mkdir()
        assert drop_failed_record(R_, "metaqa__s1eval") is None                         # no record: nothing to do
        write_json(R_.records / "metaqa__s1eval.json", {"problems": []})
        assert drop_failed_record(R_, "metaqa__s1eval") is None and (R_.records / "metaqa__s1eval.json").exists()
        write_json(R_.records / "metaqa__s1eval.json", {"problems": ["no rmatch records"]})
        assert drop_failed_record(R_, "metaqa__s1eval") == ["no rmatch records"] and not (R_.records / "metaqa__s1eval.json").exists()
    print("zabl self-test: OK")
    return 0


def parse_pair(s):
    if s is None:
        return None
    i, k = (int(x) for x in s.split("/"))
    if not 0 <= i < k:
        raise SystemExit("--index/--shard i/n with 0 <= i < n")
    return (i, k)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("kind", nargs="?")
    ap.add_argument("--arm", choices=sorted(ARM_ALL) + [CHECK_ARM])
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--datasets")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--shard")
    ap.add_argument("--lanes", type=int, default=6)
    ap.add_argument("--name")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return selftest()
    need = {"pools": ("arm", "dataset"), "chain": ("arm", "dataset"), "step": ("arm", "dataset"), "gate": ("arm",),
            "fork": ("arm",), "read": ("arm", "name"), "prune": ("arm",), "items": ("arm",), "check": ("name",)}
    for k in need.get(a.cmd, ()):
        if getattr(a, k) is None:
            raise SystemExit(f"{a.cmd}: --{k} is required")
    ds_list = tuple(a.datasets.split(",")) if a.datasets else None
    if a.cmd == "pools":
        return pools(a.arm, a.dataset, parse_pair(a.shard) or (0, 1), a.limit, a.host)
    if a.cmd == "chain":
        return chain(a.arm, a.dataset, a.lanes, a.host)
    if a.cmd == "step":
        return run_step(a.kind, a.arm, a.dataset, parse_pair(a.index), parse_pair(a.shard), a.host)
    if a.cmd == "gate":
        return gate(a.arm, ds_list, a.host)
    if a.cmd == "fork":
        return fork(a.arm)
    if a.cmd == "read":
        return read(a.arm, a.name, ds_list)
    if a.cmd == "prune":
        return prune(a.arm, ds_list)
    if a.cmd == "check":
        return check_rec(a.name, (ds_list or ("musique",))[0])
    if a.cmd == "report":
        return report()
    if a.cmd == "items":
        print("\n".join(g_items() + items(a.arm, a.lanes) if a.kind == "with-g" else items(a.arm, a.lanes)))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
