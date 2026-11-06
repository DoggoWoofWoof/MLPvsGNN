"""ABL (docs/ABL_LINK_FAMILY_ABLATION.md): what each link family carries, with the models of record fixed.

The record holds three stores in the pool rule: the graph slot G (`structural`: N, U2b's W2_S, on the four passage
datasets; S, the KB, on MetaQA and WebQSP), O (`ner`, the old NER store) and K (`knn`). An arm is a subset of {G, O, K}:
a family that is out is an empty store (same node count, no edges) in every process that opens it. The first stage, B,
the seeds, the rule and its budgets are the record's. ML1's pair-1 fits (zg6r, zrckp; seeds 0-2; both WQ splits) are
forked and read on the arm's pools; nothing is refitted. No LLM, no corpus hyperlink, no test split.

    python outputs/mp_unified/zabl.py --selftest
    python outputs/mp_unified/zabl.py pools --arm A --dataset D [--shard i/k] [--limit Q] [--host]
    python outputs/mp_unified/zabl.py step KIND --arm A --dataset D [--index i/n] [--shard i/k] [--host]
    python outputs/mp_unified/zabl.py chain --arm A --dataset D [--lanes N] [--host]      # the carve's whole chain
    python outputs/mp_unified/zabl.py gate --arm A [--datasets d1,d2] [--host]
    python outputs/mp_unified/zabl.py fork --arm A
    python outputs/mp_unified/zabl.py read --arm A --name FIT [--datasets d1,d2]           # the host's GPU
    python outputs/mp_unified/zabl.py prune --arm A                                         # look and cache go, reads stay
    python outputs/mp_unified/zabl.py report
    python outputs/mp_unified/zabl.py items --arm A [--lanes N]                             # feeder rows, printed
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"                                  # BLAS at 2, as in every PKO process (zpko section 4)
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # ML1's reads (zml1s.where_ok)

from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

import zb1p as B  # noqa: E402  (helpers, module loader and PKO's checks; B1P's own stage is not run from here)

DECLARED = "docs/ABL_LINK_FAMILY_ABLATION.md"
STAGE = "ABL"
OUT = ROOT / "outputs" / "abl"
CARVE = "s1eval"
# the arm = which of G (graph slot), O (old ner store), K (knn) are in
FLAGS = {"GOK": (1, 1, 1), "GK": (1, 0, 1), "G": (1, 0, 0), "OK": (0, 1, 1), "GO": (1, 1, 0), "K": (0, 0, 1),
         "O": (0, 1, 0), "0": (0, 0, 0), "Kb": (0, 0, 1), "0b": (0, 0, 0)}   # Kb, 0b: K and 0 built on the KB datasets
BUILD_ORDER = ("GO", "GK", "G", "K", "0", "Kb", "0b", "OK", "O")          # the record, GOK, is ML1's reads: nothing is built
FAMILY = ("structural", "ner", "knn")
DATASETS = ("webqsp", "musique", "hotpotqa", "metaqa", "squad", "2wiki")        # smallest development carve first
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")
KB = ("webqsp", "metaqa")
# where an arm is built (docs/ABL_LINK_FAMILY_ABLATION.md section 2): R (slot + ner) moves as one on the passage
# datasets; on the KB datasets S is the slot and R the ner store; removing S there is for ablation only (sout)
APPLIES = {"GO": DATASETS, "GK": KB, "G": KB, "K": PASSAGE, "0": PASSAGE}
APPLIES_SOUT = {"Kb": KB, "0b": KB, "OK": KB, "O": KB}
SHARDS = {"webqsp": 2, "musique": 3, "hotpotqa": 3, "metaqa": 6, "squad": 3, "2wiki": 5}   # pools shards per carve
ARMS_ML = B.ARMS                                               # ("zg6r", "zrckp")
SEEDS, SPLITS, SUF, FITS, TRAINS = B.SEEDS, B.SPLITS, B.SUF, B.FITS, B.TRAINS
SCREEN_FITS = B.SCREEN_FITS
PYLIB_RUN = B.PYLIB_RUN
STEPS = ("pools", "merge", "look", "bases", "cache", "links", "kt", "record")
NO_STEP1 = {"state": "not applicable: an ABL arm has no step-1 cache (docs/ABL_LINK_FAMILY_ABLATION.md)"}
BOOT = 2000

log, utc, sha_file, sha_src, rel = B.log, B.utc, B.sha_file, B.sha_src, B.rel
read_json, write_json, Patches = B.read_json, B.write_json, B.Patches


def arm_dir(arm):
    return OUT / arm


def dropped(arm):
    if arm not in FLAGS:
        raise SystemExit(f"ABL: arm {arm}: one of {sorted(FLAGS)}")
    return tuple(f for f, k in zip(FAMILY, FLAGS[arm]) if not k)


def in_domain_splits(ds):
    return tuple(sp for sp in SPLITS if any(d == ds for d, _c in TRAINS[sp]))


def roots_of(M, arm):
    return M.PK.roots(arm_dir(arm))


# ── a family that is out ─────────────────────────────────────────────────────

def empty_store(M, family, n):
    """A family store of n nodes with no edge, of the class the walks read (zu2c.csr_store for the structural family;
    FamilyStore with empty weights for ner and knn)."""
    if family == "structural":
        return M.U2C.csr_store(n, np.zeros(n + 1, np.int64), np.zeros(0, np.int32))
    from mp_retrieval.m3b_pools import FamilyStore
    return FamilyStore(family, int(n), np.zeros(n + 1, np.int64), np.zeros(0, np.int32),
                       weight=np.zeros(0, np.float16), edges_stored=0)


@contextlib.contextmanager
def on_arm(M, name, arm):
    """The package's dataset opened with the arm's families out, for zpko.pools_pass, look_shard, cache, links and KT
    (every patch put back on exit). A family that is out is asserted to hold no edge wherever it is loaded."""
    drop = dropped(arm)
    PK, S4, U2C = M.PK, M.S4, M.U2C
    P_ = Patches()
    seen = {"empty": []}
    try:
        mp_mod = S4.m3b_pools
        orig_store = mp_mod.load_or_build_store

        def load_store(ds_, family, cache_dir):
            if ds_.name != name or family not in drop:
                return orig_store(ds_, family, cache_dir)
            st = empty_store(M, family, int(ds_.n_nodes))
            if int(np.asarray(st.col).size):
                raise SystemExit(f"ABL {arm}/{name}: the {family} store that is out holds edges")
            seen["empty"].append(family)
            return st

        P_.set(mp_mod, load_or_build_store=load_store)
        E = M.E

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
        if "structural" in drop:
            orig_gc = U2C.graph_csr

            def graph_csr(name_, graph, n_):
                g = orig_gc(name_, graph, n_)
                if g is None or name_ != name:
                    return g
                ip, _col, desc = g                      # the record's description kept: U2e's check still holds
                return np.zeros(np.asarray(ip).size, np.int64), np.zeros(0, np.int32), desc

            orig_at, orig_p = PK.AT.load_types, PK.P.load_types

            def at_load(name_, desc=None):
                arr, rec = orig_at(name_, desc)
                if name_ != name:
                    return arr, rec
                arr = dict(arr)
                arr["cls"] = np.zeros(0, np.asarray(arr["cls"]).dtype)
                return arr, dict(rec, entries=0, abl={"arm": arm, "per_entry_classes": "empty: no slot entries"})

            def p_load(d, desc=None):
                cls, zbar, rec = orig_p(d, desc)
                if d != name:
                    return cls, zbar, rec
                return (np.zeros(0, np.asarray(cls).dtype), zbar,
                        dict(rec, entries=0, entries_null=0, abl={"arm": arm, "per_entry_classes": "empty"}))

            P_.set(U2C, graph_csr=graph_csr)
            P_.set(PK.AT, load_types=at_load)
            P_.set(PK.P, load_types=p_load)
        yield SimpleNamespace(drop=drop, seen=seen)
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
    elif kind == "links":
        with B.checks(M), on_arm(M, name, arm):
            PK.chain_build("links", name, CARVE, R)
    elif kind == "kt":
        PK.kt_stage(name, CARVE, R)
    elif kind == "record":
        rec = PK.record_carve(name, CARVE, R, step1=lambda *_a: dict(NO_STEP1))
        step_record(R, arm, name, "record", {"inputs": len(rec["inputs"]), "problems": rec["problems"]})
    else:
        raise SystemExit(f"ABL: step {kind}: one of {STEPS}")
    log(f"step {kind} {arm}/{name} {index or shard or ''}: {time.time() - t:.0f} s")
    return 0


def chain(arm, name, lanes=1, host=False):
    """The carve's whole chain, each step its own process; the independent shards, look shards and cache parts of a
    stage run `lanes` at a time. A step already filed is skipped."""
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
    if not all((Path(R.links) / name / CARVE / f"part_{i}of{nc}" / "record.json").exists() for i in range(nc)):
        one("links")
    if not (Path(R.kt) / name / CARVE / "record.json").exists():
        one("kt")
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
    """ML1's roots and gate for an arm; GOK reads on the record's own."""
    if arm == "GOK":
        return dict(ML.ROOTS), ML.GATE
    b = f"outputs/abl/{arm}"
    return ({**ML.ROOTS, "pools": f"{b}/pools", "plook": f"{b}/look", "cache": f"{b}/cache", "rmatch": f"{b}/chains/rmatch",
             "zrc": f"{b}/chains/zrc", "links": f"{b}/links", "kt": f"{b}/kt"}, f"{b}/gate.json")


def read(arm, fit, datasets=None):
    """ML1's read of a fork on the arm's roots and gate (zml1s's code, no stamp); the reads land in the fork."""
    import zml1s as ML
    import zwq as WQ
    import zp1c3 as Z3
    if fit not in FITS or ML.ALL_FITS.get(fit) is None:
        raise SystemExit(f"ABL: {fit} is not one of ML1's zg6r or zrckp fits")
    mlarm, _seed, _split = ML.ALL_FITS[fit]
    datasets = tuple(datasets or (built(arm) if arm != "GOK" else DATASETS))
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
    """A dataset's look and cache go once its twelve reads are filed (they cost lane-hours to regenerate, so only reads
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


def check_gok(fit, ds="webqsp", load_read=None, out=None):
    """ABL check 5: the record re-read through ABL's own read path (arm GOK, forked fit, the record's roots and gate)
    equals ML1's filed read on the same carve, per question, to 1e-4 (ids, gold totals and every measure)."""
    import p3c_claim1 as P3
    load_read = load_read or P3.load_read
    mine = load_read(ROOT, f"outputs/abl/fits/GOK/{fit}", ds)
    ml1 = load_read(ROOT, f"outputs/screen/fits/{fit}", ds)
    ok = (np.array_equal(mine["ids"], ml1["ids"]) and np.array_equal(mine["gt"], ml1["gt"])
          and mine["m"].shape == ml1["m"].shape)
    worst = float(np.max(np.abs(mine["m"] - ml1["m"]))) if ok else None
    ok = bool(ok and worst <= 1e-4)
    rec = {"declared_in": DECLARED, "stage": STAGE, "check": "GOK re-read equals ML1's read (section 4.5)", "fit": fit,
           "dataset": ds, "carve": CARVE, "abl_read_sha256": mine["sha256"], "ml1_read_sha256": ml1["sha256"],
           "questions": int(mine["ids"].size), "max_abs_difference": worst, "equal_to_1e-4": ok,
           "zabl_sha256": sha_src(__file__), "utc": utc()}
    if out is None:
        write_json(arm_dir("GOK") / f"abl_check_{fit}__{ds}.json", rec)
    log(f"check GOK {fit}/{ds}: max |difference| {worst}; {'EQUAL' if ok else 'NOT EQUAL'}")
    return (0 if ok else 1) if out is None else rec


# ── the report ────────────────────────────────────────────────────────────────

def per_q(arm, fit, ds):
    """One fit's per-question R@5 on a carve, from the arm's fork (GOK: ML1's filed read)."""
    import p3c_claim1 as P3
    fdir = f"outputs/screen/fits/{fit}" if arm == "GOK" else f"outputs/abl/fits/{arm}/{fit}"
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


def report():
    rec = {"declared_in": DECLARED, "stage": STAGE, "boot": BOOT, "arms": {}, "missing": []}
    lines = [f"# ABL report (declared in {DECLARED})", "",
             "R@5 x100 on each development carve (s1eval), the pair-1 fits of ML1 fixed, three seeds; arm minus GOK (the record) "
             "with a 95% interval from 2,000 paired bootstrap resamples over questions; call = BELOW / LEVEL / ABOVE; "
             "'below' = fits (of those read) where the arm is below GOK.", "",
             "| arm | dataset | regime | model | GOK | arm | diff (CI) | call | below |", "|---|---|---|---|---:|---:|---|---|---|"]
    for arm in BUILD_ORDER:
        for ds in built(arm):
            for regime in ("in-domain", "held out"):
                for mlarm in ARMS_ML:
                    a, b = cell("GOK", ds, mlarm, regime), cell(arm, ds, mlarm, regime)
                    if a is None or b is None:
                        if (a is not None) and b is None and any(
                                (OUT / "fits" / arm).glob("*")):
                            rec["missing"].append(f"{arm}/{ds}/{mlarm}/{regime}")
                        continue
                    c = compare(a, b)
                    rec["arms"].setdefault(arm, {}).setdefault(ds, {}).setdefault(regime, {})[mlarm] = c
                    lines.append(f"| {arm} | {ds} | {regime} | {mlarm} | {100 * c['a']:.1f} | {100 * c['b']:.1f} | "
                                 f"{100 * c['diff']:+.1f} ({100 * c['ci'][0]:+.1f}, {100 * c['ci'][1]:+.1f}) | {c['call']} | "
                                 f"{c['fits_below']}/{c['fits']} |")
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
]
# a read on the card as B1P's reads ran (zb1p rows): a transient WinError 32 retried, 0.27 of the card's memory, a share of 0.29
READ = ("python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.27 "
        "--gc 0.9 -- outputs/mp_unified/zabl.py")
LANE_GB = {"webqsp": 7.0, "musique": 9.5, "hotpotqa": 9.5, "metaqa": 9.5, "squad": 7.0, "2wiki": 9.5}


def gok_items():
    """Check 5's rows: the record forked and re-read on webqsp/s1eval for seed 0 of both models, then compared to ML1's
    filed read (each a few minutes; the GPU share is the arms' reads')."""
    ins = ";".join(CODE)
    fits = ("scr-ml1-zg6r-s0", "scr-ml1-zrckp-s0")
    rows = [f"line|x||abl-GOK-fork|1|2|{ins}|python outputs/mp_unified/zabl.py fork --arm GOK"]
    names = []
    for f in fits:
        n = f"abl-GOK-read-{f.replace('scr-ml1-', '')}"
        names.append(n)
        rows.append(f"line|x|abl-GOK-fork|{n}|1.1|6|{ins}|{READ} read --arm GOK --name {f} --datasets webqsp|gpu:0.29")
    for f, n in zip(fits, names):
        rows.append(f"line|x|{n}|abl-GOK-check-{f.replace('scr-ml1-', '')}|1|2|{ins}|python outputs/mp_unified/zabl.py check --name {f}")
    return rows


def items(arm, lanes=6, sout=False):
    """Feeder rows (outputs/host_ops/hostfeed_items.txt format) for one arm. The arm's datasets run one after another
    (smallest carve first) so that its disk peak is one dataset's: chain, then the gate on that dataset, then the twelve
    reads on the GPU, then the prune of its look and cache; the next dataset's chain waits on that prune. Arms are
    independent of one another and run side by side as the host's lanes allow. With sout the arm's S-out carves (KB
    datasets, ablation only) under names ending -sout."""
    if arm == "GOK":
        return gok_items()
    table = APPLIES_SOUT if sout else APPLIES
    if arm not in table:
        raise SystemExit(f"ABL: arm {arm} is not built {'as S-out ' if sout else ''}anywhere")
    dss = [d for d in DATASETS if d in table[arm]]
    py = "python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py"
    retry = "python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --"
    ins = ";".join(CODE)
    tag = f"{arm}-sout" if sout else arm
    rows = [f"line|x||abl-{tag}-fork|1|2|{ins}|python outputs/mp_unified/zabl.py fork --arm {arm}"]
    prev = ""
    for ds in dss:
        n = f"abl-{tag}-{ds}"
        rows.append(f"line|x|{prev}|{n}|{2 * lanes}|{lanes * LANE_GB[ds]:g}|{ins}|{retry} {py} chain --arm {arm} --dataset {ds} --lanes {lanes} --host")
        g = f"abl-{tag}-gate-{ds}"
        rows.append(f"line|x|{n}|{g}|2|8|{ins}|{py} gate --arm {arm} --datasets {ds} --host")
        reads = []
        for fit in FITS:
            r = f"abl-{tag}-read-{ds}-{fit.replace('scr-ml1-', '')}"
            reads.append(r)
            rows.append(f"line|x|{g},abl-{tag}-fork|{r}|1.1|6|{ins}|{READ} read --arm {arm} --name {fit} --datasets {ds}|gpu:0.29")
        prev = f"abl-{tag}-prune-{ds}"
        rows.append(f"line|x|{','.join(reads)}|{prev}|1|2|{ins}|python outputs/mp_unified/zabl.py prune --arm {arm} --datasets {ds}")
    return rows


def selftest():
    assert dropped("GOK") == () and dropped("GK") == ("ner",) and dropped("G") == ("ner", "knn")
    assert dropped("OK") == ("structural",) and dropped("GO") == ("knn",) and dropped("0") == FAMILY
    assert set(BUILD_ORDER) | {"GOK"} == set(FLAGS) and len(set(BUILD_ORDER)) == 9
    assert len({FLAGS[a] for a in FLAGS}) == 8, "every subset of {G, O, K} once (Kb and 0b repeat K and 0)"
    assert FLAGS["Kb"] == FLAGS["K"] and FLAGS["0b"] == FLAGS["0"]
    assert len(set(DATASETS)) == 6 and set(PASSAGE) | set(KB) == set(DATASETS) and not set(PASSAGE) & set(KB)
    assert all(a in FLAGS and set(d) <= set(DATASETS) for a, d in {**APPLIES, **APPLIES_SOUT}.items())
    assert set(APPLIES) | set(APPLIES_SOUT) == set(BUILD_ORDER)
    # R moves as one on the passage datasets: no passage dataset is built with the slot and ner apart (arms GK, G, OK, O)
    for a in ("GK", "G", "OK", "O", "Kb", "0b"):
        assert not set(APPLIES.get(a, ())) & set(PASSAGE) and not set(APPLIES_SOUT.get(a, ())) & set(PASSAGE), a
    assert in_domain_splits("musique") == ("WQ-hotpotqa",) and in_domain_splits("hotpotqa") == ("WQ-musique",)
    assert in_domain_splits("metaqa") == () and set(in_domain_splits("2wiki")) == set(SPLITS)
    assert set(in_domain_splits("webqsp")) == set(SPLITS) and set(in_domain_splits("squad")) == set(SPLITS)
    assert len(FITS) == 12 and all(k in SHARDS for k in DATASETS)
    # an empty family store has no edge and the right size
    # (zu2c.csr_store and FamilyStore need the package modules: checked where they import)
    rows = items("GO", 6)
    assert len(rows) == 1 + 6 * (1 + 1 + 12 + 1) and all(r.count("|") >= 7 for r in rows)
    assert sum("gpu:" in r for r in rows) == 72 and all(r.startswith("line|x|") for r in rows)
    names = [r.split("|")[3] for r in rows]
    assert len(names) == len(set(names)), "item names are unique"
    by = {r.split("|")[3]: r.split("|")[2].split(",") for r in rows}
    # the datasets run one after another: each chain waits on the previous dataset's prune, never on a later item
    assert by["abl-GO-musique"] == ["abl-GO-prune-webqsp"] and by["abl-GO-webqsp"] == [""]
    assert all(d in names[:names.index(n)] for n, ds_ in by.items() for d in ds_ if d)
    g = items("GOK")
    assert len(g) == 1 + 2 + 2 and sum("gpu:" in r for r in g) == 2 and all(r.startswith("line|x|") for r in g)
    assert len(items("GK", 6)) == 1 + 2 * 15 and len(items("Kb", 6, sout=True)) == 1 + 2 * 15
    assert len(items("K", 6)) == 1 + 4 * 15 and len(items("0", 6)) == 1 + 4 * 15
    assert items("0", 6)[0].split("|")[3] == "abl-0-fork" and items("0", 6)[1].split("|")[3] == "abl-0-musique" and "-sout-" not in "".join(items("0", 6))
    # check 5 on fake reads: equal reads pass, a 1e-3 change fails
    def fake(delta=0.0):
        def lr(_root, fit, _ds):
            m = np.full((4, 3), 0.5) + (delta if "abl" in fit else 0.0)
            return {"ids": np.array(list("abcd")), "gt": np.ones(4, np.int64), "m": m, "sha256": "x"}
        return lr
    assert check_gok("f", "d", fake(), out=True)["equal_to_1e-4"]
    assert not check_gok("f", "d", fake(1e-3), out=True)["equal_to_1e-4"]
    # the paired comparison: identical cells are LEVEL with difference 0; a cell 0.1 lower in every question is BELOW
    ids = np.array([str(i) for i in range(200)])
    rng = np.random.default_rng(0)
    base = rng.random(200)
    a = {"ids": ids, "gt": np.ones(200, np.int64), "per_fit": [base, base]}
    b = {"ids": ids, "gt": np.ones(200, np.int64), "per_fit": [base - 0.1, base - 0.1]}
    c0, c1 = compare(a, a), compare(a, b)
    assert c0["call"] == "LEVEL" and abs(c0["diff"]) < 1e-12 and c1["call"] == "BELOW" and c1["fits_below"] == 2
    assert abs(c1["diff"] + 0.1) < 1e-9
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
    ap.add_argument("--arm", choices=sorted(FLAGS))
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--datasets")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--index")
    ap.add_argument("--shard")
    ap.add_argument("--lanes", type=int, default=6)
    ap.add_argument("--name")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--sout", action="store_true", help="items: the arm's S-out carves (KB datasets, ablation only)")
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
        return check_gok(a.name, (ds_list or ("webqsp",))[0])
    if a.cmd == "report":
        return report()
    if a.cmd == "items":
        print("\n".join(items(a.arm, a.lanes, a.sout)))
        return 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
