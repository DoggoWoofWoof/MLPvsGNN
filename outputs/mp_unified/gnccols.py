"""GNC's halves, half fits, held-out first passes, assembled first pass, lists and columns
(docs/GNC_CROSSFIT_FIRST_PASS_GNN.md): GNR's NEXT columns (docs/GNR_REANCHORED_NEXT_GNN.md) with the first pass cross-fitted, so
that the anchors of every training question come from a fit that did not train on it.

  halves   LTM's components of each training carve dealt to two halves (largest first, the half with fewer questions so far).
  hfit     ISB's zg6r-m16 (own start, 16 epochs, SWA from 10) on one half of every training carve of the split.
  heldout  GNR's first-pass read of that half fit on the split's four training carves (p@swa alone, the five highest rows of
           every question, checked against lean_gpu.top_hit's golds); the other half's questions are the out-of-sample ones.
  first    the split's first pass in GNR's schema: a training question's five rows from the half fit that did not train on
           it; the s1eval carves' files are GNR's, byte-copied (sha256 checked against GNR's record).
  lists    gnrcols.lists under GNC's stage, declaration and root, on that first pass (LTM's folds checked as there).
  cols     gnrcols.cols under GNC's names (LTM's fold fits read only); the carve's gnc.json beside its column record.
  report   the first passes, the lists' census and the columns beside LTM's (decides nothing); GNC's s1eval columns beside GNR's.

gnrcols.py, ltmcols.py, ltwdiag.py, lean_gpu.py, readswa.py, zml1s.py, ziso.py, zisb.py and zdep.py are imported and not
edited. Everything patched is put back afterwards, also on a failure. GNC writes only under outputs/gnc and the fits
scr-gnc-h*; GNR's first passes, LTM's lists and fold fits, ML1's and ISB's fit folders are read only.

    python outputs/mp_unified/gnccols.py halves --split WQ-musique
    python outputs/mp_unified/gnccols.py hfit --name scr-gnc-h0-s0                (host GPU)
    python outputs/mp_unified/gnccols.py heldout --name scr-gnc-h0-s0             (host GPU)
    python outputs/mp_unified/gnccols.py first --split WQ-musique --seed 0
    python outputs/mp_unified/gnccols.py lists --split WQ-musique --seed 0 --dataset musique --carve s1eval
    python outputs/mp_unified/gnccols.py cols --split WQ-musique --seed 0 --dataset musique --carve s1eval --host
    python outputs/mp_unified/gnccols.py report
    python outputs/mp_unified/gnccols.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("LTM_THREADS", "2"))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

import gnrcols as B  # noqa: E402  (GNR's first pass, lists and columns: patched for a call, not edited)

LTMC, LTW, CH, WQ = B.LTMC, B.LTW, B.CH, B.WQ
DECLARED = "docs/GNC_CROSSFIT_FIRST_PASS_GNN.md"
STAGE = "GNC"
KEY = "gnc"                                     # the stamp's key in a GNC half fit's screen.json and in GNC's records
OUT = ROOT / "outputs" / "gnc"
GNR_OUT = B.OUT                                 # GNR's first passes: read only
LTM_OUT, PKO = B.LTM_OUT, B.PKO
TOP, K, FILL, CAND, FIRST_TOL = B.TOP, B.K, B.FILL, B.CAND, B.FIRST_TOL
HALVES = 2
CONFIG = "2e-3:1e-4:0.1:16:10"                  # ISB's: lr:wd:dropout:epochs:swa_from
SEEDS = (0,)                                    # the screen is seed 0; seeds 1 and 2 follow by amendment
SPLITS = tuple(WQ.SPLITS)
COPY = B.COPY
_B_CODE = B.code_shas


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] gnccols: {msg}", flush=True)


rel, sha_file, write_json, replace = B.rel, B.sha_file, B.write_json, B.replace


def seed_root(seed, out=OUT):
    return B.seed_root(seed, out)


def first_root(seed, out=OUT):
    return B.first_root(seed, out)


def first_record(seed, split, out=OUT):
    return B.first_record(seed, split, out)


def lists_path(seed, split, ds, carve, out=OUT):
    return B.lists_path(seed, split, ds, carve, out)


def census_path(seed, split, ds, carve, out=OUT):
    return B.census_path(seed, split, ds, carve, out)


def held_root(seed, out=OUT):
    return seed_root(seed, out) / "held"


def held_record(name, seed=0, out=OUT):
    return held_root(seed, out) / f"{name}.json"


def halves_dir(out=OUT):
    return Path(out) / "halves"


def halves_record(split, out=OUT):
    return halves_dir(out) / f"{split}.json"


def halves_file(split, ds, carve, out=OUT):
    return halves_dir(out) / f"{split}__{ds}__{carve}.npz"


def code_shas():
    return {"gnccols_sha256": CH.sha_src(__file__), **_B_CODE()}


def half_fit_name(j, seed, split):
    return f"scr-gnc-h{int(j)}-s{int(seed)}{WQ.SUF[split]}"


HALF_FITS = {half_fit_name(j, sd, sp): ("zg6r", sd, sp) for sd in SEEDS for sp in SPLITS for j in range(HALVES)}
HALF_OF = {half_fit_name(j, sd, sp): (sp, j, sd) for sd in SEEDS for sp in SPLITS for j in range(HALVES)}
assert len(HALF_FITS) == 4 == len(HALF_OF)


# ── the halves (CPU) ─────────────────────────────────────────────────────────


def deal(comp):
    """The halves of a carve's questions from their components: components largest first (ties: the lower first question,
    which is the lower component number), each to the half with fewer questions so far (ties: the lower half), as
    ltmcols.comp_folds deals folds. Returns (half of each question, questions per half)."""
    comp = np.asarray(comp, np.int64)
    size = np.bincount(comp) if comp.size else np.zeros(0, np.int64)
    of_comp, load = np.empty(size.size, np.int64), np.zeros(HALVES, np.int64)
    for c in np.lexsort((np.arange(size.size), -size)):
        h = int(np.argmin(load))
        of_comp[c] = h
        load[h] += size[c]
    return of_comp[comp].astype(np.int8), load


def halves(split, root=PKO, ltm_out=LTM_OUT, out=OUT):
    """The split's halves (section 2.1): per training carve LTM's components dealt to two halves, with the carve's
    ids sha256 (the cache's own) and the half of every question; outputs/gnc/halves/<split>.json and one npz per carve."""
    import lean_cache as LC
    t0 = time.time()
    per = {}
    for ds, cv in LTMC.train_carves(split):
        lp = LTMC.lists_path(ltm_out, ds, cv)
        with np.load(lp) as z:
            comp = np.asarray(z["Q_comp"], np.int64)
        parts, recs = LC.part_dirs(ds, cv, Path(root) / "cache")
        ids = [str(x) for r in recs for x in r["ids"]]
        if len(ids) != comp.size:
            raise SystemExit(f"gnc: {ds}/{cv}: LTM's lists hold {comp.size} questions, the cache {len(ids)}")
        h, load = deal(comp)
        if int(load.min()) == 0 or int(load.sum()) != comp.size:
            raise SystemExit(f"gnc: {ds}/{cv}: the halves hold {load.tolist()} of {comp.size} questions")
        if not all(np.unique(h[comp == c]).size == 1 for c in np.unique(comp)):
            raise SystemExit(f"gnc: {ds}/{cv}: a component is in both halves")
        f = halves_file(split, ds, cv, out)
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f.stem + ".tmp.npz")
        np.savez(tmp, half=h, comp=comp)
        replace(tmp, f)
        per[f"{ds}={cv}"] = {"questions": int(comp.size), "components": int(comp.max()) + 1 if comp.size else 0,
                             "half_questions": [int(x) for x in load],
                             "half_components": [int(len(np.unique(comp[h == j]))) for j in range(HALVES)],
                             "carve_ids_sha256": recs[0]["look"]["carve_ids_sha256"],
                             "ids_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
                             "file": f.name, "file_sha256": sha_file(f), "ltm_lists": rel(lp),
                             "ltm_lists_sha256": sha_file(lp)}
        log(f"halves {split} {ds}/{cv}: {comp.size} questions, {per[f'{ds}={cv}']['components']} components -> "
            f"{load.tolist()} questions")
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split,
           "rule": "LTM's components (questions joined by a shared gold node) dealt largest first, ties by the lower first "
                   "question, each to the half with fewer questions so far, ties to the lower half",
           "halves": HALVES, "carves": per, "code": code_shas(), "seconds": time.time() - t0,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(halves_record(split, out), rec)
    return rec


def load_halves(split, out=OUT):
    """(the split's halves record, {(dataset, carve): the half of each question}), every file's sha256 checked."""
    p = halves_record(split, out)
    if not p.is_file():
        raise SystemExit(f"gnc: no halves of {split} ({rel(p)})")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if (rec.get("stage"), rec.get("declared_in"), rec.get("split"), rec.get("halves")) != (STAGE, DECLARED, split, HALVES):
        raise SystemExit(f"gnc: {rel(p)} is not GNC's halves of {split}")
    want = [f"{ds}={cv}" for ds, cv in LTMC.train_carves(split)]
    if list(rec.get("carves") or {}) != want:
        raise SystemExit(f"gnc: the halves of {split} hold {list(rec.get('carves') or {})}, not {want}")
    h = {}
    for ds, cv in LTMC.train_carves(split):
        c = rec["carves"][f"{ds}={cv}"]
        f = halves_file(split, ds, cv, out)
        if sha_file(f) != c["file_sha256"]:
            raise SystemExit(f"gnc: {rel(f)} is not the file the halves of {split} name")
        with np.load(f) as z:
            h[(ds, cv)] = np.asarray(z["half"], np.int8)
        if h[(ds, cv)].size != c["questions"] or [int((h[(ds, cv)] == j).sum()) for j in range(HALVES)] != c["half_questions"]:
            raise SystemExit(f"gnc: {rel(f)} does not hold the halves its record names")
    return rec, h


# ── the half fits (host GPU) ─────────────────────────────────────────────────


def half_of(cls, keep):
    """cls (lean_gpu's CacheCarve of the arm) with the questions of keep only: keep maps (dataset, carve) to (the kept
    questions' indices, the carve's ids sha256). rows is the kept count and batch(qs) takes indices among the kept, mapped
    to the carve's own before its batch runs, so its cache, links and chain arrays stay aligned. A carve not in keep is
    whole."""
    class HalfCarve(cls):
        def __init__(self, ds, carve, *a, **k):
            super().__init__(ds, carve, *a, **k)
            self.keep_q = None
            spec = keep.get((ds, carve))
            if spec is None:
                return
            idx, ids_sha = spec
            idx = np.asarray(idx, np.int64)
            if self.look["carve_ids_sha256"] != ids_sha:
                raise SystemExit(f"gnc: {ds}/{carve}: the cache's ids are not the halves'")
            if idx.size == 0 or int(idx.min()) < 0 or int(idx.max()) >= self.rows or np.unique(idx).size != idx.size:
                raise SystemExit(f"gnc: {ds}/{carve}: the kept questions {idx.size} are not distinct questions of the {self.rows}")
            self.keep_q, self.rows_all, self.rows = idx, self.rows, int(idx.size)

        def batch(self, qs, blocks):
            qs = np.asarray(qs, np.int64)
            if self.keep_q is None:
                return super().batch(qs, blocks)
            if qs.size and (int(qs.min()) < 0 or int(qs.max()) >= self.rows):
                raise SystemExit(f"gnc: {self.ds}/{self.carve}: a batch asked for questions outside the {self.rows} kept")
            return super().batch(self.keep_q[qs], blocks)
    HalfCarve.__name__ = HalfCarve.__qualname__ = f"Half{cls.__name__}"
    return HalfCarve


@contextlib.contextmanager
def on_half_train(LG, keep):
    """lean_gpu.train_cmd run with the carve class at that moment (the arm's, set by lean_screen's patched) subclassed to
    the kept questions; the class put back afterwards."""
    orig = LG.train_cmd

    def train_cmd(a):
        cur = LG.CacheCarve
        LG.CacheCarve = half_of(cur, keep)
        try:
            return orig(a)
        finally:
            LG.CacheCarve = cur
    with WQ.patch(LG, train_cmd=train_cmd):
        yield


@contextlib.contextmanager
def with_fits():
    """GNC's half fits in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    import zml1s as M1
    M1.ALL_FITS.update(HALF_FITS)
    try:
        yield
    finally:
        for k in HALF_FITS:
            M1.ALL_FITS.pop(k, None)


def gnc_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the half fit being run answering for its own (arm, seed, split)."""
    import zml1s as M1
    a = M1._ACTIVE[0]
    if a in HALF_FITS and HALF_FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


@contextlib.contextmanager
def on_names():
    """zml1s with GNC's names, stamp key, stage and declaration; restored after. Enough for a held-out read."""
    import readswa as RS
    import zdep as DEP
    import ziso as ISO
    import zisb as ISB
    import zml1s as M1
    sha = WQ.sha_src(__file__)
    more = {"ziso_sha256": WQ.sha_src(ISO.__file__), "zdep_sha256": WQ.sha_src(DEP.__file__),
            "zisb_sha256": WQ.sha_src(ISB.__file__), **RS.code()}
    code = M1.code_shas
    with with_fits(), WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=gnc_fit_name,
                               code_shas=lambda arm=None: {**code(arm), "gnccols_sha256": sha, **more}):
        yield


def keep_of(split, j, out=OUT):
    """{(dataset, carve): (the indices of half j's questions, the carve's ids sha256)}, and the halves record."""
    rec, hv = load_halves(split, out)
    return {k: (np.flatnonzero(h == j).astype(np.int64), rec["carves"][f"{k[0]}={k[1]}"]["carve_ids_sha256"])
            for k, h in hv.items()}, rec


def stamp_extra(name, out=OUT):
    """What a half fit's ML1 stamp block gains: GNC's config, the half, the halves record and the kept questions."""
    sp, j, _sd = HALF_OF[name]
    keep, rec = keep_of(sp, j, out)
    return {"config": CONFIG, "half": j, "halves_record_sha256": sha_file(halves_record(sp, out)),
            "kept": {f"{ds}={cv}": {"questions": int(idx.size),
                                    "kept_index_sha256": hashlib.sha256(idx.tobytes()).hexdigest()}
                     for (ds, cv), (idx, _s) in keep.items()}}


def stamp_with(orig, out=OUT):
    """zml1s.stamp with a half fit's extras added."""
    def stamp(name, env, what, extra=None):
        if name not in HALF_FITS:
            return orig(name, env, what, extra)
        return orig(name, env, what, {**(extra or {}), **stamp_extra(name, out)})
    return stamp


@contextlib.contextmanager
def on_half(G, RM, name, out=OUT):
    """A half fit's train context: GNC's names and stamp, ISO's own start (its load_encoder in zml1s, its encoder_used in
    zg6r), ISB's config on rmatch's train, and lean_gpu.train_cmd on the half's questions; restored after."""
    import lean_gpu as LG
    import ziso as ISO
    import zisb as ISB
    import zml1s as M1
    if ISB.CONFIG != CONFIG:
        raise SystemExit(f"gnc: ISB's config is {ISB.CONFIG}, not {CONFIG}")
    sp, j, _sd = HALF_OF[name]
    keep, _rec = keep_of(sp, j, out)
    with on_names(), WQ.patch(M1, stamp=stamp_with(M1.stamp, out), load_encoder=ISO.own_load), \
            WQ.patch(G, encoder_used=ISO.own_used), WQ.patch(RM, train=ISB.with_config(RM.train)), \
            on_half_train(LG, keep):
        yield


def verify_half_train(name, base=ROOT, out=OUT):
    """A half fit's record is GNC's: its train read the half's questions of every training carve, none of the other half,
    under ISB's config; returns the stamp block."""
    import zml1s as M1
    if name not in HALF_OF:
        raise SystemExit(f"gnc: {name} is not one of GNC's half fits")
    sp, j, _sd = HALF_OF[name]
    fd = M1.screen_dir(base) / "fits" / name
    for f in ("train.json", "screen.json", "models.pt"):
        if not (fd / f).is_file():
            raise SystemExit(f"gnc: {rel(fd)} lacks {f}")
    tj = json.loads((fd / "train.json").read_text(encoding="utf-8"))
    blk = json.loads((fd / "screen.json").read_text(encoding="utf-8")).get(KEY) or {}
    rec, hv = load_halves(sp, out)
    want = [(ds, cv, int((hv[(ds, cv)] == j).sum())) for ds, cv in LTMC.train_carves(sp)]
    got = [(t["dataset"], t["carve"], t["questions"]) for t in tj.get("train") or []]
    if got != want:
        raise SystemExit(f"gnc: {name} trained on {got}, not on half {j}'s {want}")
    keep, _r = keep_of(sp, j, out)
    kept = {f"{ds}={cv}": {"questions": int(i.size), "kept_index_sha256": hashlib.sha256(i.tobytes()).hexdigest()}
            for (ds, cv), (i, _s) in keep.items()}
    if (blk.get("declared_in"), blk.get("half"), blk.get("config"), blk.get("halves_record_sha256"), blk.get("kept")) != (
            DECLARED, j, CONFIG, sha_file(halves_record(sp, out)), kept):
        raise SystemExit(f"gnc: {name}'s stamp is not GNC's half {j} of {sp} under {CONFIG} on the current halves")
    return blk


def hfit(name, base=ROOT, out=OUT):
    """One half fit on the host's GPU (section 2.2): trained on its half's questions when not trained and stamped before;
    no read here (heldout reads it on the training carves)."""
    import zml1s as M1
    if name not in HALF_FITS:
        raise SystemExit(f"gnc: {name} is not one of GNC's half fits ({', '.join(HALF_FITS)})")
    sp, j, sd = HALF_OF[name]
    G, _U2N = M1.import_arm("zg6r")
    import zisb as ISB
    RM = ISB.trainer()
    env = M1.device_env()
    M1.where_ok(env, "fit")
    M1.gate_ready()
    keep, _rec = keep_of(sp, j, out)
    log(f"{name}: zg6r from its own start, config {CONFIG}, half {j} of {sp}: "
        f"{', '.join(f'{k[0]}/{k[1]} {len(v[0])}' for k, v in keep.items())} questions")
    t0 = time.time()
    with on_half(G, RM, name, out):
        fd = WQ.fit_dir(name)
        if M1.stamped(name, "screen") and (fd / "train.json").is_file() and (fd / "models.pt").is_file():
            log(f"{name}: trained before, kept")
        else:
            M1.clear(fd, "*")
            rc = M1.train(name, env)
            if rc:
                return rc
    verify_half_train(name, base, out)
    log(f"{name}: trained on half {j} in {time.time() - t0:.0f} s; record checked")
    return 0


# ── the held-out first pass (host GPU) ───────────────────────────────────────


def summarise_held(name, src, dst, carves, seen, out=OUT, seed=0):
    """A half fit's held-out record: per training carve its read file (the five rows of every question), the R@5 over all
    questions, over the half it did not train on (the out-of-sample one) and over the half it trained on; the check
    (the carves read are the split's training carves, every question once)."""
    import lean_gpu as LG
    sp, j, _sd = HALF_OF[name]
    mine = json.loads((dst / "read.json").read_text(encoding="utf-8"))
    want = [f"{d}={c}" for d, c in carves]
    if list(mine.get("carves") or {}) != want or [x["carve"] for x in seen] != want:
        raise SystemExit(f"gnc: the held-out read of {name} covered {list(mine.get('carves') or {})} (captured "
                         f"{[x['carve'] for x in seen]}), not {want}")
    sha = sha_file(src / "models.pt")
    if mine.get("candidates") != [CAND] or mine.get("models_sha256") != sha:
        raise SystemExit(f"gnc: the held-out read of {name} used {mine.get('candidates')} of models "
                         f"{mine.get('models_sha256')}, not {CAND} of {sha}")
    rec, hv = load_halves(sp, out)
    per = {}
    for (ds, cv), sv in zip(carves, seen):
        f = dst / "reads" / f"{ds}__{cv}.npz"
        with np.load(f) as z:
            top, hit, gt, T, n, ids = z["top"], z["hit"], z["gold_total"], z["top5"], z["n"], z["ids"]
            cand = [str(x) for x in z["candidates"]]
        h = hv[(ds, cv)]
        if cand != [CAND] or T.shape != (top.shape[1], TOP) or n.size != h.size or int(n.sum()) != sv["rows"]:
            raise SystemExit(f"gnc: {rel(f)} does not hold {CAND}'s five rows of each of the {h.size} questions")
        m = LG.metrics_of(top[0], hit[0], gt)[:, 0]
        ok = gt > 0

        def r5(sel):
            s = ok & sel
            return float(m[s].mean()) if s.any() else None
        per[f"{ds}={cv}"] = {"file": rel(f, held_root(seed, out)), "sha256": sha_file(f), "questions": int(n.size),
                             "rows": int(n.sum()), "with_gold": int(ok.sum()), "R@5": r5(np.ones(n.size, bool)),
                             "R@5_heldout": r5(h != j), "R@5_trained": r5(h == j),
                             "heldout_questions": int((h != j).sum())}
    return {"stage": STAGE, "declared_in": DECLARED, "split": sp, "seed": int(seed), "fit": name, "half": j,
            "candidate": CAND, "models_sha256": sha, "copies": {f: sha_file(dst / f) for f in COPY},
            "halves_record_sha256": sha_file(halves_record(sp, out)), "ok": True, "carves": per,
            "read": {k: mine.get(k) for k in ("seconds", "peak_gpu_gb", "peak_rss_bytes", "rows_cap")},
            "code": code_shas(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def heldout(name, base=ROOT, out=OUT):
    """A trained half fit's first pass over the split's four training carves (section 2.3), GNR's way: p@swa alone, the
    five highest rows of every question kept and checked against top_hit's golds, into GNC's copy of the fit."""
    import lean_gpu as LG
    import readswa as RS
    import zml1s as M1
    if name not in HALF_OF:
        raise SystemExit(f"gnc: {name} is not one of GNC's half fits")
    sp, j, sd = HALF_OF[name]
    t0 = time.time()
    verify_half_train(name, base, out)
    src = M1.screen_dir(base) / "fits" / name
    env = M1.device_env()
    M1.where_ok(env, "read")
    dst = held_root(sd, out) / name
    dst.mkdir(parents=True, exist_ok=True)
    for f in COPY:
        B.copy_checked(src / f, dst / f)
    carves = LTMC.train_carves(sp)
    seen = []
    log(f"held-out pass {name} (half {j} of {sp}): {len(carves)} training carves into {rel(dst)}")
    with on_names():
        _rec, k = M1.fit_stamp(name)
        if M1.device_of(env) != k.get("device"):
            raise SystemExit(f"gnc: {name} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
        with RS.on_read(LG), WQ.patch(LG, read_carve=B.capture(LG.read_carve, seen),
                                      read_cmd=B.into(LG.read_cmd, held_root(sd, out), carves, name)):
            rc, _extra = M1.run_arm(name, env, "read")
    if rc != 0:
        raise SystemExit(f"gnc: the held-out read of {name} returned {rc}")
    rec = summarise_held(name, src, dst, carves, seen, out=out, seed=sd)
    rec.update({"device": M1.device_of(env), "seconds": time.time() - t0})
    write_json(held_record(name, sd, out), rec)
    for key, r in rec["carves"].items():
        log(f"  {key}: R@5 {r['R@5']:.4f}; held-out {r['R@5_heldout']:.4f} ({r['heldout_questions']} questions), "
            f"trained {r['R@5_trained']:.4f}")
    return rec


# ── the assembled first pass (CPU) ───────────────────────────────────────────


def first(seed, split, base=ROOT, out=OUT, gnr_out=GNR_OUT, call=None):
    """The split's first pass in GNR's schema (section 2.4): a training question's five rows from the half fit that did not
    train on it; the s1eval carves' files are GNR's, byte-copied. check = every piece present and agreeing."""
    import lean_gpu as LG
    import zdep as DEP
    import ziso as ISO
    t0 = time.time()
    call = DEP.iso_call(base) if call is None else call
    if call != ISO.KEEP:
        raise SystemExit(f"gnc: ISO's verdict is {call}; GNC's s1eval anchors are ML1's zg6r fit's under KEEP only")
    bname = DEP.base_fit(seed, split, call)
    gp = B.first_record(seed, split, gnr_out)
    if not gp.is_file():
        raise SystemExit(f"gnc: no GNR first pass of {split}, seed {seed} ({rel(gp)})")
    g = json.loads(gp.read_text(encoding="utf-8"))
    if (g.get("stage"), g.get("split"), g.get("seed"), g.get("base"), g.get("check")) != ("GNR", split, int(seed), bname, True):
        raise SystemExit(f"gnc: {rel(gp)} is not a passed GNR first pass of {bname} on {split}, seed {seed}")
    hrec, hv = load_halves(split, out)
    held = {}
    for j in range(HALVES):
        nm = half_fit_name(j, seed, split)
        hp = held_record(nm, seed, out)
        if not hp.is_file():
            raise SystemExit(f"gnc: no held-out pass of {nm} ({rel(hp)})")
        r = json.loads(hp.read_text(encoding="utf-8"))
        if (r.get("stage"), r.get("declared_in"), r.get("fit"), r.get("half"), r.get("ok"),
                r.get("halves_record_sha256")) != (STAGE, DECLARED, nm, j, True, sha_file(halves_record(split, out))):
            raise SystemExit(f"gnc: {rel(hp)} is not a passed held-out pass of {nm} on the current halves")
        held[j] = (nm, r, hp)
    root = first_root(seed, out) / f"asm-{split}"
    (root / "reads").mkdir(parents=True, exist_ok=True)
    train = set(LTMC.train_carves(split))
    per = {}
    for ds, cv in LTMC.col_carves(split):
        key = f"{ds}={cv}"
        f = root / "reads" / f"{ds}__{cv}.npz"
        if (ds, cv) in train:
            reads = []
            for j in range(HALVES):
                nm, r, _hp = held[j]
                c = r["carves"][key]
                rf = held_root(seed, out) / c["file"]
                if sha_file(rf) != c["sha256"]:
                    raise SystemExit(f"gnc: {rel(rf)} is not the file {nm}'s held-out record names")
                with np.load(rf) as z:
                    reads.append({k: z[k] for k in z.files})
            a, b = reads
            h = hv[(ds, cv)]
            for k in ("ids", "n", "gold_total"):
                if not np.array_equal(a[k], b[k]):
                    raise SystemExit(f"gnc: {key}: the two half fits' reads disagree on {k}")
            if a["n"].size != h.size:
                raise SystemExit(f"gnc: {key}: the reads hold {a['n'].size} questions, the halves {h.size}")
            use1 = (h == 0)[None]                 # a half-0 question was trained by fit 0: its anchors are fit 1's
            top5 = np.where(use1.T, b["top5"], a["top5"]).astype(np.int64)
            top = np.where(use1, b["top"], a["top"])
            hit = np.where(use1, b["hit"], a["hit"])
            tmp = f.with_name(f.stem + ".tmp.npz")
            np.savez_compressed(tmp, candidates=np.asarray([CAND]), ids=a["ids"], gold_total=a["gold_total"], top=top,
                                hit=hit, top5=top5, n=a["n"])
            replace(tmp, f)
            gt, ok = a["gold_total"], a["gold_total"] > 0
            r5 = float(LG.metrics_of(top[0], hit[0], gt)[ok].mean(0)[0]) if ok.any() else None
            gc = (g["carves"] or {}).get(key) or {}
            per[key] = {"file": rel(f, first_root(seed, out)), "sha256": sha_file(f), "questions": int(a["n"].size),
                        "rows": int(a["n"].sum()), "with_gold": int(ok.sum()), "R@5": r5, "in_sample": False,
                        "source": "halves", "from_fits": [held[0][0], held[1][0]],
                        "gnr_in_sample_R@5": gc.get("R@5"), "gnr_in_sample": gc.get("in_sample"),
                        "R@5_by_fit_heldout": [held[j][1]["carves"][key]["R@5_heldout"] for j in range(HALVES)]}
        else:
            gc = g["carves"][key]
            gf = B.first_root(seed, gnr_out) / gc["file"]
            if sha_file(gf) != gc["sha256"]:
                raise SystemExit(f"gnc: {rel(gf)} is not the file GNR's first pass names")
            if not (f.is_file() and sha_file(f) == gc["sha256"]):
                shutil.copyfile(gf, f)
            if sha_file(f) != gc["sha256"]:
                raise SystemExit(f"gnc: the copy of {rel(gf)} does not match it")
            per[key] = {"file": rel(f, first_root(seed, out)), "sha256": gc["sha256"], "questions": gc["questions"],
                        "rows": gc["rows"], "with_gold": gc["with_gold"], "R@5": gc["R@5"], "in_sample": False,
                        "source": "gnr", "gnr_file": rel(gf), "base_R@5": gc.get("base_R@5"), "delta": gc.get("delta"),
                        "within": gc.get("within"), "agree": gc.get("agree")}
    models = {f"h{j}": {"fit": held[j][0], "models_sha256": held[j][1]["models_sha256"],
                        "held_record": rel(held[j][2], first_root(seed, out).parent),
                        "held_record_sha256": sha_file(held[j][2])} for j in range(HALVES)}
    ev = [r for k, r in per.items() if r["source"] == "gnr"]
    check = bool(ev) and all(r.get("within") for r in ev) and len(per) == len(LTMC.col_carves(split))
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "base": bname, "candidate": CAND,
           "base_models_sha256": g["base_models_sha256"], "models_sha256": g["models_sha256"], "halves": models,
           "halves_record_sha256": sha_file(halves_record(split, out)), "gnr_first_sha256": sha_file(gp),
           "check": check, "tol": FIRST_TOL, "carves": per, "code": code_shas(), "seconds": time.time() - t0,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(first_record(seed, split, out), rec)
    for key, r in per.items():
        log(f"  {key}: R@5 {r['R@5']:.4f}" + (f" (GNR in sample {r['gnr_in_sample_R@5']:.4f})" if r["source"] == "halves"
                                               else f" (GNR's s1eval, base {r['base_R@5']:.4f})"))
    if not check:
        raise SystemExit(f"gnc: the first pass of {split}, seed {seed} did not pass its check")
    return rec


# ── the lists and the columns (CPU) ──────────────────────────────────────────


@contextlib.contextmanager
def as_gnc():
    """gnrcols with GNC's stage, declaration and code record, for the length of a call; restored after."""
    with WQ.patch(B, STAGE=STAGE, DECLARED=DECLARED, code_shas=code_shas):
        yield


def lists(seed, split, ds, carve, root=PKO, out=OUT, ltm_out=LTM_OUT):
    """gnrcols.lists on GNC's first pass, under GNC's names and root; the census says the anchors are out of sample."""
    with as_gnc():
        sm = B.lists(seed, split, ds, carve, root=root, out=out, ltm_out=ltm_out)
    cp = census_path(seed, split, ds, carve, out)
    c = json.loads(cp.read_text(encoding="utf-8"))
    c["in_sample"] = False
    write_json(cp, c)
    return c


def cols(seed, split, ds, carve, root=PKO, out=OUT, nodes=None, host=False, ltm_out=LTM_OUT):
    """gnrcols.cols under GNC's names and root; gnc.json beside the carve's column record (gnrcols' gnr.json, renamed)."""
    with as_gnc():
        rec = B.cols(seed, split, ds, carve, root=root, out=out, nodes=nodes, host=host, ltm_out=ltm_out)
    cd = LTMC.cols_dir(seed_root(seed, out), split, ds, carve)
    gj = json.loads((cd / "gnr.json").read_text(encoding="utf-8"))
    if gj.get("stage") != STAGE:
        raise SystemExit(f"gnc: {rel(cd / 'gnr.json')} is not GNC's")
    write_json(cd / "gnc.json", gj)
    (cd / "gnr.json").unlink()
    return rec


def same_as_gnr(seed, split, ds, carve, out=OUT, gnr_out=GNR_OUT):
    """Beside, deciding nothing: whether GNC's columns of an s1eval carve equal GNR's (each cache part's next_sha256)."""
    a = LTMC.cols_dir(seed_root(seed, out), split, ds, carve)
    b = LTMC.cols_dir(seed_root(seed, gnr_out), split, ds, carve)
    if not (a / "record.json").is_file() or not (b / "record.json").is_file():
        return None
    pa = sorted(p.name for p in a.iterdir() if p.is_dir())
    pb = sorted(p.name for p in b.iterdir() if p.is_dir())
    if pa != pb:
        return False
    for nm in pa:
        ra = json.loads((a / nm / "record.json").read_text(encoding="utf-8")).get("next_sha256")
        rb = json.loads((b / nm / "record.json").read_text(encoding="utf-8")).get("next_sha256")
        if ra is None or ra != rb:
            return False
    return True


def report(out=OUT, ltm_out=LTM_OUT, seeds=SEEDS, gnr_out=GNR_OUT):
    """Per seed, split and carve: the assembled first pass's R@5 (out of sample on every carve) beside GNR's in-sample R@5
    on the training carves and the base's on the s1eval ones, the lists' census and the columns beside LTM's; GNC's s1eval
    columns beside GNR's. Decides nothing."""
    def load(p):
        return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).is_file() else None
    with as_gnc():
        rep = B.report(out=out, ltm_out=ltm_out, seeds=seeds)
    rows = rep["rows"]
    for r in rows:
        fr = load(first_record(r["seed"], r["split"], out)) or {}
        f = (fr.get("carves") or {}).get(f"{r['dataset']}={r['carve']}") or {}
        r["source"] = f.get("source")
        r["gnr_in_sample_R@5"] = f.get("gnr_in_sample_R@5")
        r["heldout_R@5_by_fit"] = f.get("R@5_by_fit_heldout")
        r["in_sample"] = False
        r["equals_gnr"] = (same_as_gnr(r["seed"], r["split"], r["dataset"], r["carve"], out, gnr_out)
                           if r["carve"] == LTMC.EVAL_CARVE else None)
    rec = {"stage": STAGE, "declared_in": DECLARED, "decides": "nothing", "rows": rows, "code": code_shas(),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(Path(out) / "report.json", rec)

    def fm(x, d=4):
        return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
    md = ["# GNC: the cross-fitted first pass, its lists and its columns (decides nothing)", "",
          f"From {DECLARED}. R@5 of the assembled anchors (out of sample on every carve), GNR's in-sample R@5 beside it on "
          "the training carves, the base's s1eval R@5 on the s1eval carves; the T lists and the share of golds outside the "
          "anchors next to one, beside LTM's (rrf's anchors); the share of rows next to an anchor and the gold share among "
          "them, beside LTM's columns; whether an s1eval carve's columns equal GNR's.", "",
          "| seed | split | carve | GNC first R@5 | GNR in sample | base R@5 | rrf R@5 | T lists (LTM) | "
          "golds next to an anchor (LTM) | rows next to an anchor (LTM) | gold share there (LTM) | = GNR |",
          "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        md.append(f"| {r['seed']} | {r['split']} | {r['dataset']}/{r['carve']} | {fm(r['first_R@5'])} | "
                  f"{fm(r['gnr_in_sample_R@5'])} | {fm(r['base_R@5'])} | {fm(r['rrf_R@5'])} | {fm(r['T_lists'])} "
                  f"({fm(r['ltm_T_lists'])}) | {fm(r['next_share'], 3)} ({fm(r['ltm_next_share'], 3)}) | "
                  f"{fm(r['rows_with_an_anchor'], 3)} ({fm(r['ltm_rows_with_an_anchor'], 3)}) | {fm(r['gold_adjacent'], 3)} "
                  f"({fm(r['ltm_gold_adjacent'], 3)}) | {fm(r['equals_gnr'])} |")
    (Path(out) / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    return rec


# ── self-test (toy carves; no host, no GPU) ──────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def toy_held(tmp, out, sp, j, rng, seed=0):
    """A held-out record of half fit j of the toy split as heldout() leaves it: GNC's copy of the fit with a read.json and
    one read file per training carve (rounded scores, with ties), and the record summarise_held writes."""
    import lean_cache as LC
    name = half_fit_name(j, seed, sp)
    src = Path(tmp) / "fits" / name
    dst = held_root(seed, out) / name
    for p in (src, dst / "reads"):
        p.mkdir(parents=True, exist_ok=True)
    (src / "models.pt").write_bytes(f"toy models {j}".encode())
    (src / "screen.json").write_text("{}", encoding="utf-8")
    for f in COPY:
        B.copy_checked(src / f, dst / f)
    carves, seen, mine = LTMC.train_carves(sp), [], {}
    for ds, cv in carves:
        parts, recs = LC.part_dirs(ds, cv, Path(tmp) / "cache")
        n = np.concatenate([np.load(p / "n.npy") for p in parts]).astype(np.int64)
        g = np.concatenate([np.load(p / "gold.npy") for p in parts]).astype(bool)
        gt = np.concatenate([np.load(p / "gold_total.npy") for p in parts]).astype(np.int32)
        ids = np.asarray([x for r in recs for x in r["ids"]])
        s = np.round(rng.normal(size=int(n.sum())), 1).astype(np.float32)
        T = B.top5(s, n)
        top = B.hits_of(T, g, n).astype(np.int16)[None]
        hit = g[np.r_[0, np.cumsum(n)][:-1] + T[:, 0]].astype(np.uint8)[None]
        np.savez_compressed(dst / "reads" / f"{ds}__{cv}.npz", candidates=np.asarray([CAND]), ids=ids, gold_total=gt,
                            top=top, hit=hit, top5=T, n=n)
        seen.append({"carve": f"{ds}={cv}", "questions": int(n.size), "rows": int(n.sum())})
        mine[f"{ds}={cv}"] = {}
    write_json(dst / "read.json", {"candidates": [CAND], "models_sha256": sha_file(src / "models.pt"), "carves": mine})
    rec = summarise_held(name, src, dst, carves, seen, out=out, seed=seed)
    write_json(held_record(name, seed, out), rec)
    return rec


def selftest():
    import torch
    saved = (LTMC.STAGE, LTMC.DECLARED, LTMC.OUT, LTMC.load_lists, LTMC.load_folds, LTW.DS_K)
    rng = np.random.default_rng(11)
    # 1. deal: balanced, deterministic, components whole, largest first, ties to the lower half
    comp = np.asarray([0, 0, 0, 1, 1, 2, 3, 4, 4, 5])
    h, load = deal(comp)
    assert h.tolist() == [0, 0, 0, 1, 1, 1, 1, 0, 0, 1] or int(load.sum()) == 10, h
    assert int(load.sum()) == 10 and abs(int(load[0]) - int(load[1])) <= 1, load
    assert all(np.unique(h[comp == c]).size == 1 for c in np.unique(comp))
    for _ in range(20):
        c = np.sort(rng.integers(0, 60, 300))
        c = np.unique(c, return_inverse=True)[1]
        hh, ld = deal(c)
        assert int(ld.sum()) == 300 and all(np.unique(hh[c == k]).size == 1 for k in np.unique(c))
        sizes = np.bincount(c)
        assert abs(int(ld[0]) - int(ld[1])) <= int(sizes.max())
        assert np.array_equal(hh, deal(c)[0])
    assert deal(np.zeros(0, np.int64))[0].size == 0

    # 2. the half carve: rows is the kept count, batch maps a kept index to the carve's own, a whole carve is whole, and
    #    a wrong ids sha, a repeated question or one past the carve stops it
    class Base:
        def __init__(self, ds, carve, *a, **k):
            self.ds, self.carve, self.rows = ds, carve, 12
            self.look = {"carve_ids_sha256": "abc"}
            self.seen = []

        def batch(self, qs, blocks):
            self.seen.append([int(x) for x in qs])
            return ("feats", list(qs))
    Half = half_of(Base, {("musique", "fit"): (np.asarray([1, 4, 5, 9]), "abc")})
    c = Half("musique", "fit")
    assert c.rows == 4 and c.rows_all == 12 and c.batch(np.asarray([0, 3]), ["x"]) == ("feats", [1, 9])
    assert c.seen == [[1, 9]] and Half("squad", "fit").rows == 12 and Half("squad", "fit").batch([2, 3], [])[1] == [2, 3]
    must_stop(c.batch, np.asarray([4]), ["x"])
    for bad in ((np.asarray([1, 1]), "abc"), (np.asarray([1, 12]), "abc"), (np.asarray([], np.int64), "abc"),
                (np.asarray([1, 2]), "zzz")):
        must_stop(half_of(Base, {("musique", "fit"): bad}), "musique", "fit")

    # 3. on_half_train: the arm's carve class at the call is the one subclassed, and is put back
    import lean_gpu as LG
    got = []
    orig_t, orig_c = LG.train_cmd, LG.CacheCarve
    try:
        LG.train_cmd, LG.CacheCarve = (lambda a: got.append(LG.CacheCarve) or 0), Base
        with on_half_train(LG, {("musique", "fit"): (np.asarray([0]), "abc")}):
            assert LG.train_cmd(None) == 0
        assert LG.CacheCarve is Base and got[0] is not Base and issubclass(got[0], Base) and LG.train_cmd is not orig_t
    finally:
        LG.train_cmd, LG.CacheCarve = orig_t, orig_c
    assert LG.train_cmd is orig_t and LG.CacheCarve is orig_c

    # 4. toy PKO trees of one split: LTM's lists and fold fits, the halves, two held-out records, GNR's toy first pass
    tmp = Path(tempfile.mkdtemp(prefix="gnc_"))
    try:
        sp = "WQ-hotpotqa"
        bname = "scr-ml1-zg6r-s0-hp"
        M = LTW.unit(rng.normal(size=(400, 24))).astype(np.float32)
        for ds, cv in LTMC.col_carves(sp):
            LTW.toy_tree(tmp, ds, cv, M, rng, nq=30, n=40)
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        ltm, out, gnr = tmp / "ltm", tmp / "gnc", tmp / "gnr"
        for ds, cv in LTMC.col_carves(sp):
            LTMC.lists(ds, cv, root=tmp, out=ltm)
        LTMC.fit(sp, root=tmp, out=ltm, nodes=nodes, epochs=2, min_lists=1)
        must_stop(load_halves, sp, out)
        hr = halves(sp, root=tmp, ltm_out=ltm, out=out)
        rec, hv = load_halves(sp, out)
        assert list(hv) == list(LTMC.train_carves(sp)) and all(
            sum(v["half_questions"]) == 30 and min(v["half_questions"]) > 0 for v in hr["carves"].values())
        f = halves_file(sp, "musique", "s1fit", out)
        f_bytes = f.read_bytes()
        with np.load(f) as z:
            keep = {k: z[k] for k in z.files}
        np.savez(f, **{**keep, "half": 1 - keep["half"]})
        must_stop(load_halves, sp, out)                                  # a changed file is not the one the record names
        f.write_bytes(f_bytes)
        assert load_halves(sp, out)[0]["carves"] == hr["carves"]
        kp, _r = keep_of(sp, 1, out)
        assert all(np.array_equal(v[0], np.flatnonzero(hv[k] == 1)) for k, v in kp.items())
        # the half fits' record: the stamp's extras, and verify_half_train on a made fit folder
        import zml1s as M1
        nm = half_fit_name(1, 0, sp)
        ex = stamp_extra(nm, out)
        assert ex["config"] == CONFIG and ex["half"] == 1 and set(ex["kept"]) == {f"{d}={c}" for d, c in LTMC.train_carves(sp)}
        assert [v["questions"] for v in ex["kept"].values()] == [hr["carves"][k]["half_questions"][1] for k in ex["kept"]]
        fd = M1.screen_dir(tmp) / "fits" / nm
        fd.mkdir(parents=True)
        (fd / "models.pt").write_bytes(b"toy")
        train = {"train": [{"dataset": d, "carve": c, "questions": hr["carves"][f"{d}={c}"]["half_questions"][1]}
                           for d, c in LTMC.train_carves(sp)]}
        write_json(fd / "train.json", train)
        write_json(fd / "screen.json", {KEY: {"declared_in": DECLARED, **ex}})
        assert verify_half_train(nm, tmp, out)["half"] == 1
        write_json(fd / "train.json", {"train": [{**t, "questions": 30} for t in train["train"]]})
        must_stop(verify_half_train, nm, tmp, out)                       # a train on the whole carve
        write_json(fd / "train.json", train)
        for bad in ({"half": 0}, {"halves_record_sha256": "0" * 64}, {"config": "2e-3:1e-4:0.1:8:5"},
                    {"declared_in": "docs/ISB_ZG6RM_16_EPOCHS.md"}):
            write_json(fd / "screen.json", {KEY: {"declared_in": DECLARED, **ex, **bad}})
            must_stop(verify_half_train, nm, tmp, out)
        write_json(fd / "screen.json", {KEY: {"declared_in": DECLARED, **ex}})
        assert verify_half_train(nm, tmp, out)["half"] == 1
        must_stop(verify_half_train, nm, tmp / "nowhere", out)           # no fit folder under another base
        must_stop(verify_half_train, "scr-ml1-zg6r-s0-hp", tmp, out)     # ML1's fit is not a half fit
        # zml1s with GNC's names: the half fits answer for their own (arm, seed, split), ML1's names stay ML1's; restored
        s_m1 = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.code_shas, len(M1.ALL_FITS))
        with on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(HALF_FITS) <= set(M1.ALL_FITS)
            M1._ACTIVE[0] = nm
            try:
                assert M1.pko_fit_name("zg6r", 0, "WQ-hotpotqa") == nm
                assert M1.pko_fit_name("zg6r", 0, "WQ-musique") == M1.fit_name("zg6r", 0, "WQ-musique")
            finally:
                M1._ACTIVE[0] = None
            assert "gnccols_sha256" in M1.code_shas("zg6r") and "zisb_sha256" in M1.code_shas("zg6r")
        assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.code_shas, len(M1.ALL_FITS)) == s_m1
        seen = []
        st = stamp_with(lambda name, env, what, extra=None: seen.append((name, what, extra)), out)
        st("scr-ml1-zg6r-s0-hp", {}, "screen", {"x": 1})
        st(nm, {}, "screen", {"x": 2})
        assert seen[0] == ("scr-ml1-zg6r-s0-hp", "screen", {"x": 1}) and seen[1][2]["x"] == 2 and seen[1][2]["half"] == 1
        # the first pass: no held-out record, then one half only, stops; GNR's toy first pass (its check passed)
        gfr = B.toy_first(tmp, gnr, 0, sp, bname, np.random.default_rng(1))
        assert gfr["check"]
        must_stop(first, 0, sp, base=tmp, out=out, gnr_out=gnr, call="KEEP")
        h0 = toy_held(tmp, out, sp, 0, np.random.default_rng(2))
        assert h0["ok"] and set(h0["carves"]) == {f"{d}={c}" for d, c in LTMC.train_carves(sp)}
        must_stop(first, 0, sp, base=tmp, out=out, gnr_out=gnr, call="KEEP")
        toy_held(tmp, out, sp, 1, np.random.default_rng(3))
        import ziso as ISO
        fr = first(0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.KEEP)
        assert fr["check"] and fr["stage"] == STAGE and all(not r["in_sample"] for r in fr["carves"].values())
        assert fr["halves"]["h0"]["fit"] == half_fit_name(0, 0, sp) and fr["base"] == bname
        for ds, cv in LTMC.col_carves(sp):
            r = fr["carves"][f"{ds}={cv}"]
            with np.load(first_root(0, out) / r["file"]) as z:
                T, ids = z["top5"], z["ids"]
            if (ds, cv) in set(LTMC.train_carves(sp)):
                assert r["source"] == "halves"
                with np.load(held_root(0, out) / h0["carves"][f"{ds}={cv}"]["file"]) as z0, \
                        np.load(held_root(0, out) / json.loads(held_record(half_fit_name(1, 0, sp), 0, out).read_text(
                            encoding="utf-8"))["carves"][f"{ds}={cv}"]["file"]) as z1:
                    t0_, t1_ = z0["top5"], z1["top5"]
                hh = hv[(ds, cv)]
                assert np.array_equal(T[hh == 0], t1_[hh == 0]) and np.array_equal(T[hh == 1], t0_[hh == 1])
                assert (hh == 0).any() and (hh == 1).any() and not np.array_equal(t0_, t1_)
            else:
                assert r["source"] == "gnr" and r["sha256"] == gfr["carves"][f"{ds}={cv}"]["sha256"]
        # a half fit trained and read but whose record names other halves, a GNR first pass that failed, or a ruling
        # other than KEEP stops the first pass
        hp = held_record(half_fit_name(0, 0, sp), 0, out)
        keep_h = hp.read_text(encoding="utf-8")
        write_json(hp, {**json.loads(keep_h), "halves_record_sha256": "0" * 64})
        must_stop(first, 0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.KEEP)
        hp.write_text(keep_h, encoding="utf-8")
        must_stop(first, 0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.REPLACE)
        gp = B.first_record(0, sp, gnr)
        keep_g = gp.read_text(encoding="utf-8")
        write_json(gp, {**json.loads(keep_g), "check": False})
        must_stop(first, 0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.KEEP)
        gp.write_text(keep_g, encoding="utf-8")
        fr = first(0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.KEEP)
        # a half fit's read file changed after its record: the first pass stops
        rf = held_root(0, out) / h0["carves"]["musique=s1fit"]["file"]
        rf_bytes = rf.read_bytes()
        with np.load(rf) as z:
            kz = {k: z[k] for k in z.files}
        np.savez_compressed(rf, **{**kz, "top5": kz["top5"][::-1]})
        must_stop(first, 0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.KEEP)
        rf.write_bytes(rf_bytes)
        fr = first(0, sp, base=tmp, out=out, gnr_out=gnr, call=ISO.KEEP)
        # 5. the lists and the columns on the assembled anchors: GNR's pipeline under GNC's names
        for ds, cv in LTMC.col_carves(sp):
            sm = lists(0, sp, ds, cv, root=tmp, out=out, ltm_out=ltm)
            assert sm["stage"] == STAGE and sm["questions"] == 30 and sm["T"]["lists"] > 0 and sm["in_sample"] is False
        A, _fr, _c = B.anchors_of.__wrapped__(0, sp, "musique", "s1fit", out) if hasattr(B.anchors_of, "__wrapped__") else (
            None, None, None)
        with as_gnc():
            A, _fr, _c = B.anchors_of(0, sp, "musique", "s1fit", out)
        with np.load(first_root(0, out) / fr["carves"]["musique=s1fit"]["file"]) as z:
            assert np.array_equal(A["top5"], z["top5"].astype(np.int64))
        fit_sha = sha_file(ltm / f"fit_{sp}.json")
        for ds, cv in (("musique", "s1fit"), ("musique", "s1eval")):
            rc = cols(0, sp, ds, cv, root=tmp, out=out, nodes=nodes, ltm_out=ltm)
            assert rc["stage"] == STAGE and rc["declared_in"] == DECLARED and rc["identity"]["pass"]
            assert rc["fit_sha256"] == fit_sha and rc["rows"] == 30 * 40
            cd = LTMC.cols_dir(seed_root(0, out), sp, ds, cv)
            gj = json.loads((cd / "gnc.json").read_text(encoding="utf-8"))
            assert gj["stage"] == STAGE and not (cd / "gnr.json").exists() and gj["base"] == bname
            assert gj["first_sha256"] == sha_file(first_record(0, sp, out))
        assert (LTMC.STAGE, LTMC.DECLARED, LTMC.OUT, LTMC.load_lists, LTMC.load_folds) == saved[:5]
        assert B.STAGE == "GNR" and B.DECLARED == "docs/GNR_REANCHORED_NEXT_GNN.md" and B.code_shas is _B_CODE
        rep = report(out=out, ltm_out=ltm, gnr_out=gnr)
        assert sum(r["first_R@5"] is not None for r in rep["rows"]) == len(LTMC.col_carves(sp))
        assert all(r["in_sample"] is False for r in rep["rows"])
        eq = [r["equals_gnr"] for r in rep["rows"] if r["carve"] == "s1eval"]
        assert len(eq) == 12 and all(x is None for x in eq)               # no GNR columns here
        assert "GNC" in (Path(out) / "report.md").read_text(encoding="utf-8")
    finally:
        LTMC.STAGE, LTMC.DECLARED, LTMC.OUT, LTMC.load_lists, LTMC.load_folds, LTW.DS_K = saved
        shutil.rmtree(tmp, ignore_errors=True)
    assert torch.is_tensor(torch.zeros(1))
    log("selftest: the halves' deal and files, the half carve (kept rows, mapped batches, its refusals), the train_cmd "
        "wrapper, the assembled first pass from two held-out records and GNR's (and its refusals), the lists and columns "
        "under GNC's names, and the report pass")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("halves", "hfit", "heldout", "first", "lists", "cols", "report"))
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--name")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.host and a.cmd in ("hfit", "heldout", "halves", "first"):
        import zpkoh
        zpkoh.apply_mirror()
    if a.cmd == "report":
        report()
        return 0
    if a.cmd in ("hfit", "heldout"):
        if not a.name:
            ap.error("--name")
        return hfit(a.name) if a.cmd == "hfit" else (heldout(a.name) and 0)
    if a.split is None:
        ap.error("--split")
    if a.cmd == "halves":
        halves(a.split)
        return 0
    if a.cmd == "first":
        first(a.seed, a.split)
        return 0
    if (a.dataset, a.carve) not in LTMC.col_carves(a.split):
        raise SystemExit(f"gnc: {a.dataset}/{a.carve} is not a carve the split {a.split} reads")
    if a.cmd == "lists":
        lists(a.seed, a.split, a.dataset, a.carve)
        return 0
    if a.cmd == "cols":
        if a.host:
            import lean_host as LH
            LH.substitute()                      # the verified mirror in place of the package, as ltmcols cols --host
        cols(a.seed, a.split, a.dataset, a.carve, host=a.host)
        return 0
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
