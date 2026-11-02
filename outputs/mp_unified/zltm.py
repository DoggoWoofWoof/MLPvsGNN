"""LTM's stage (docs/LTM_NEXT_HOP_MLP.md): zrckpn, the MLP zrckp with LTW's learned next-hop score of each pool row as seven
input columns (the NEXT block, built by ltmcols.py), on PKO's pools under WQ, on the host's CPU with DOB's device, against
DOB's CPU zrckp base and its seed null; the compares and the calls.

The columns come from the first stage's top five rows (fixed before any row is scored), PKO's links and the frozen vectors,
scored by LTW's bil form refit per split and cross-fitted over the split's training carves (ltmcols.py): the MLP reads them
as it reads any cached column, and passes no message. zml1s.py, zdob.py, zp1c3p.py, zdep.py, zltg.py and ltmcols.py are
imported and not edited. For the length of an LTM fit zml1s runs with LTM's names in its fit table, LTM's stamp key, stage
and declaration, zrckpn's stamp and read cap beside zrckp's, DOB's CPU device (zdob's cpu_env, cpu_ok and card), lean_gpu's
pick with NEXT after DOB's nine blocks, and its train and read replaced by this file's: zp1c3p's train and read for zrckpn
under R2P's arm swap. zrckpn is zrckp's model (zp1c3p.ZRCKP) on zp1c3's chain carve with the fit's split's columns beside
it. Every patched name is put back afterwards, also on a failure. The card, PKO's roots and the read cap are zrckp's, as in
DOB's base.

    python outputs/mp_unified/zltm.py fit --name scr-ltm-zrckpn-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zltm.py compare-all                       (each read fit against DOB's base; the base's null)
    python outputs/mp_unified/zltm.py calls                             (outputs/ltm/calls.json and .md)
    python outputs/mp_unified/zltm.py --selftest
"""
import os
import sys

THREADS = 6                                     # DOB's: every LTM fit's torch, OpenMP, MKL and OpenBLAS threads
if __name__ == "__main__":                      # before numpy or torch loads; a CPU fit never sees the GPU
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import zdob as DOB  # noqa: E402  (DOB's device, its base's names and its pick)
import zdep as DEP  # noqa: E402  (musique's hop slices and DEP's sixteen-read rule, beside)
import zltg as LTG  # noqa: E402  (LTG's in-domain rule: its keys and per-read calls, pure; LTG's fits are not touched)
import ltmcols as LTMC  # noqa: E402  (the columns' layout, records and carves)

DECLARED = "docs/LTM_NEXT_HOP_MLP.md"
STAGE = "LTM"
KEY = "ltm"                                     # the stamp's key in an LTM fit's screen.json and read.json and LTM's compares
OUT = "outputs/ltm"                             # calls.json and .md (the columns under cols/, the scorer's work under work/)
ARM = "zrckpn"                                  # zrckp's model on zp1c3's chain carve with NEXT beside it
BASE = DOB.BASE                                 # DOB's CPU zrckp, fitted at three seeds on this device
NEXT = "NEXT"                                   # the block: ltmcols' seven columns
PICK = DOB.PICK + (NEXT,)                       # DOB's nine blocks, then NEXT
P1C3P = "p1c3p"                                 # zp1c3p's block in a fit's screen.json, written for the arm its swap names
ROWS_CAP = M1.ROWS_CAP["zrckp"]                 # zrckp's read cap (65,536)
LTM_CODE = ("zltm", "zdob", "ltmcols", "ltwdiag", "chndiag")    # beside zml1s's code: this stage, DOB's device, the columns
CAND = "p@swa"
CLAIM1_READS = (("WQ-hotpotqa", "musique"), ("WQ-musique", "hotpotqa"))   # C1R's in-domain reads that do not hold
NAMING = "zrckp + LTW's next-hop columns (NEXT: each row against the first stage's top five rows, cross-fitted)"
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT = (
    "ADOPT_FOR_FULL_RUN", "NO_GAIN", "NOT_ADOPTED", "INCOMPLETE", "ADOPT")
assert DOB.THREADS == THREADS and BASE == "zrckp" and ROWS_CAP == 65536
assert (LTMC.DECLARED, LTMC.STAGE, LTMC.WIDTH, len(LTMC.COLS)) == (DECLARED, STAGE, 7, 7)
assert (DEP.ADOPT_FOR_FULL_RUN, DEP.NO_GAIN, DEP.NOT_ADOPTED, DEP.INCOMPLETE, DEP.ADOPT) == (
    ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT)


def fit_name(seed, split):
    return f"scr-ltm-{ARM}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    """zrckpn's fit against DOB's base fit of the same seed and split."""
    return f"scr-ltm-{ARM}-vs-{BASE}-s{seed}{M1.WQ.SUF[split]}"


def null_name(seed, split):
    """DOB's base at seed 1 or 2 against its own seed 0 (LTM's copy of the base's null)."""
    return f"scr-ltm-{BASE}-s{seed}-vs-s0{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
SCREEN = tuple(fit_name(0, sp) for sp in M1.SPLITS)
PAIRS = {pair_name(sd, sp): (fit_name(sd, sp), DOB.fit_name(BASE, sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
NULLS = {null_name(sd, sp): (DOB.fit_name(BASE, sd, sp), DOB.fit_name(BASE, 0, sp))
         for sd in RULE.SEEDS[1:] for sp in M1.SPLITS}
assert len(FITS) == 6 and len(PAIRS) == 6 and len(NULLS) == 4
assert not set(FITS) & (set(M1.ALL_FITS) | set(DOB.FITS) | set(LTG.FITS))
assert all(b in DOB.FITS and DOB.DROP_OF[b] is None for _n, b in PAIRS.values())


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zltm: {msg}", flush=True)


# ── the columns, beside the carve ────────────────────────────────────────────

CUR = {"split": None, "fit_sha256": None, "out": None}   # the split whose columns the carves built now load
LOADED = []                                     # per carve loaded: dataset, carve, its column record's sha256, the scorer


def load_next(ds, carve, split, part_shas, out=None, verify=True):
    """(rows x 7 float16 columns, the carve's column record, that record's sha256) of one carve under the split, checked
    against the carve's cache parts: the record's stage, declaration, split, carve, columns, training flag and identity
    check; each part's cache record sha256, rows, scorer and file (shape, type and, with verify, sha256)."""
    import lean_cache as LC
    cd = LTMC.cols_dir(LTMC.OUT if out is None else out, split, ds, carve)
    p = cd / "record.json"
    if not p.is_file():
        raise SystemExit(f"zltm: no LTM columns of {ds}/{carve} for {split} ({p})")
    rec = json.loads(p.read_text(encoding="utf-8"))
    want = (LTMC.STAGE, LTMC.DECLARED, split, ds, carve, list(LTMC.COLS), (ds, carve) in LTMC.train_carves(split))
    got = tuple(rec.get(k) for k in ("stage", "declared_in", "split", "dataset", "carve", "columns", "training_carve"))
    if got != want or not (rec.get("identity") or {}).get("pass") or not rec.get("fit_sha256"):
        raise SystemExit(f"zltm: {p} is {got} (identity {rec.get('identity')}), not LTM's columns {want}")
    names = [q.get("part") for q in rec.get("parts") or []]
    if names != list(part_shas):
        raise SystemExit(f"zltm: {ds}/{carve}'s columns hold the parts {names}, its cache {list(part_shas)}")
    xs = []
    for q in rec["parts"]:
        pd = cd / q["part"]
        pr = json.loads((pd / "record.json").read_text(encoding="utf-8"))
        pw = (LTMC.STAGE, split, ds, carve, q["part"], part_shas[q["part"]], q["rows"], list(LTMC.COLS), rec["fit_sha256"],
              q["next_sha256"])
        pg = tuple(pr.get(k) for k in ("stage", "split", "dataset", "carve", "part", "cache_record_sha256", "rows", "columns",
                                       "fit_sha256", "next_sha256"))
        if pg != pw:
            raise SystemExit(f"zltm: {pd / 'record.json'} does not name its carve's part, cache record, rows and scorer")
        f = pd / "next.npy"
        if verify and LC.sha_file(f) != q["next_sha256"]:
            raise SystemExit(f"zltm: {f}: sha256 is not its record's")
        x = np.load(f)
        if x.dtype != np.float16 or x.shape != (q["rows"], LTMC.WIDTH):
            raise SystemExit(f"zltm: {f} is {x.dtype} {x.shape}, not float16 ({q['rows']}, {LTMC.WIDTH})")
        xs.append(x)
    X = np.concatenate(xs) if xs else np.zeros((0, LTMC.WIDTH), np.float16)
    if X.shape[0] != rec.get("rows"):
        raise SystemExit(f"zltm: {ds}/{carve}'s column parts hold {X.shape[0]} rows, its record {rec.get('rows')}")
    return X, rec, LC.sha_file(p)


def next_carve(base):
    """base's carve class with the split's NEXT columns beside it: batch() adds them under NEXT for the batch's rows (the
    carve's own row order), block_values() hands them to lean_gpu's dead-block check, widths names their seven columns;
    every other block, side key, row and gold is base's, unchanged. KT_ROOT and CH_ROOT stay base's (zwq.on_wq patches
    them there)."""
    import torch
    import lean_cache as LC

    class NextCarve(base):
        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            sp = CUR["split"]
            if sp is None:
                raise SystemExit(f"zltm: {ds}/{carve} built for {ARM} outside an LTM split")
            X, rec, sha = load_next(ds, carve, sp, self.part_shas, CUR["out"], verify)
            if X.shape[0] != int(self.off_np[-1]):
                raise SystemExit(f"zltm: {ds}/{carve}'s columns hold {X.shape[0]} rows, its carve {int(self.off_np[-1])}")
            if CUR["fit_sha256"] is None:
                CUR["fit_sha256"] = rec["fit_sha256"]
            elif rec["fit_sha256"] != CUR["fit_sha256"]:
                raise SystemExit(f"zltm: {ds}/{carve}'s columns come from the scorer {rec['fit_sha256'][:12]}, this fit's "
                                 f"from {CUR['fit_sha256'][:12]}")
            self.next_X = torch.from_numpy(X).to(self.device)
            self.next_record = {"dataset": ds, "carve": carve, "record_sha256": sha, "fit_sha256": rec["fit_sha256"],
                                "rows": int(X.shape[0]), "training_carve": rec["training_carve"]}
            self.widths = {**self.widths, NEXT: LTMC.WIDTH}
            LOADED.append(self.next_record)

        def nbytes(self):
            return super().nbytes() + self.next_X.numel() * self.next_X.element_size()

        def block_values(self, b):
            if b == NEXT:
                return self.next_X
            return super().block_values(b)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, [b for b in blocks if b != NEXT])
            if NEXT in blocks:
                qs = np.asarray(qs, np.int64)
                cnt = self.n_np[qs]
                idx = np.repeat(self.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
                x = self.next_X[torch.from_numpy(idx).to(self.next_X.device)].to(torch.float32)
                feats[NEXT] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return feats, nq, base_z, gold

    NextCarve.__name__ = NextCarve.__qualname__ = f"Next{base.__name__}"
    return NextCarve


_ARM = []


def import_arm():
    """zp1c3p (zrckp's model, zp1c3's chain carve), imported outside zwq.on_wq, and zrckpn in lean_screen's table: zrckp's
    model on the chain carve with NEXT beside it, registered once."""
    if _ARM:
        return _ARM[0]
    import zp1c3p as Z3P
    S = Z3P.S
    if S.ARMS.get("zrckp") != (Z3P.ZRCKP, Z3P.Z3.KTChainCarveZRC):
        raise SystemExit("zltm: lean_screen's zrckp is not zp1c3p's")
    if ARM in S.ARMS:
        raise SystemExit(f"zltm: lean_screen holds an arm {ARM} that is not this file's")
    S.ARMS[ARM] = (Z3P.ZRCKP, next_carve(Z3P.Z3.KTChainCarveZRC))
    _ARM.append(Z3P)
    return Z3P


@contextlib.contextmanager
def on_arm(W):
    """zp1c3p's arm is zrckpn for the block (R2P's swap), its model and carve from lean_screen's table."""
    with M1.WQ.patch(W.Z3P, ARM=ARM, ARMS={ARM: W.S.ARMS[ARM]}):
        yield W


@contextlib.contextmanager
def on_split(split, fit_sha256=None, out=None):
    """The split whose columns the carves built in the block load (and the scorer they must share, when given); put back
    after, also on a failure."""
    if split not in M1.SPLITS:
        raise SystemExit(f"zltm: {split} is not one of WQ's splits")
    saved = dict(CUR)
    CUR.update(split=split, fit_sha256=fit_sha256, out=out)
    try:
        yield
    finally:
        CUR.clear()
        CUR.update(saved)


def next_block(split):
    """A fit's NEXT block, inside on_split: the split, the columns, the scorer and each carve's column record it read."""
    return {"declared_in": DECLARED, "split": split, "columns": list(LTMC.COLS), "fit_sha256": CUR["fit_sha256"],
            "carves": {f"{d['dataset']}/{d['carve']}": d["record_sha256"] for d in LOADED},
            "training_carves": [f"{d}/{c}" for d, c in LTMC.train_carves(split)]}


def used_ok(tj, name):
    """A fit whose trainer did not take NEXT into every set it fitted, or found it dead, is not LTM's: it stops unstamped."""
    sets, dead = tj.get("sets") or {}, tj.get("dead") or []
    if not sets or any(NEXT not in bl for bl in sets.values()) or NEXT in dead:
        raise SystemExit(f"zltm: {name}'s trainer fitted the sets {sets} with the dead blocks {dead}; unstamped")


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """LTM's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def ltm_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the LTM fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def ltm_code():
    return {f"{m}_sha256": M1.WQ.sha_src(HERE / f"{m}.py") for m in LTM_CODE}


def stamp_with_pick(orig):
    """zml1s.stamp with the fit's pick and the threads its trainer ran on."""
    def stamp(name, env, what, extra=None):
        import torch
        return orig(name, env, what, {**(extra or {}), "pick": list(PICK), "threads": torch.get_num_threads()})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with LTM's names, stamp key, stage and declaration, and zrckpn's stamp and read cap beside zrckp's; restored
    after. Enough for the compares and the calls."""
    code, extra = M1.code_shas, ltm_code()
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=ltm_fit_name,
                                  STAMP={**M1.STAMP, ARM: P1C3P}, ROWS_CAP={**M1.ROWS_CAP, ARM: ROWS_CAP},
                                  code_shas=lambda arm=None: {**code(arm), **extra}):
        yield


@contextlib.contextmanager
def on_pick_next(LG):
    """lean_gpu's pick (DOB's nine blocks, checked) with NEXT after them, and its pns without SEMB; restored after."""
    with DOB.on_pick(None, LG), M1.WQ.patch(LG, SETS={**LG.SETS, "pick": list(PICK),
                                                      "pns": [b for b in PICK if b != "SEMB"]}):
        yield


@contextlib.contextmanager
def on_ltm(Z2=None, LG=None):
    """An LTM fit's whole context: the names, DOB's CPU, zml1s's train and read replaced by LTM's, and the pick with NEXT
    (DOB's checked first); restored after."""
    if Z2 is None:
        import zp1c2 as Z2
    if LG is None:
        import lean_gpu as LG
    with on_names(), M1.WQ.patch(M1, device_env=DOB.cpu_env, ok_where=DOB.cpu_ok, stamp=stamp_with_pick(M1.stamp),
                                 train=train, read=read), \
            M1.WQ.patch(Z2, DEVICE="cpu", on_card=DOB.cpu_card(Z2.on_card)), on_pick_next(LG):
        yield


# ── train and read (zml1s's, for zrckpn) ─────────────────────────────────────


def wq_block(name):
    """zwq.train's wq block for the fit, inside on_ml1 (WQ's declaration, roots, gate and overlay are LTM's and PKO's)."""
    WQ = M1.WQ
    arm, seed, split = M1.ALL_FITS[name]
    return {"declared_in": WQ.DECLARED, "split": split, "seed": seed, "arm": arm, "trains_on": list(WQ.SPLITS[split]),
            "zero_shot": list(WQ.ZERO_SHOT[split]), "roots": dict(WQ.ROOTS), "gate": WQ.sha_file(WQ.ROOT / WQ.GATE),
            "overlay": WQ.sha_file(WQ.ROOT / WQ.OVERLAY)}


def train(name, env):
    """In place of zml1s.train for an LTM fit: zp1c3p's train for zrckpn under the arm swap, on PKO's roots under WQ
    (zwq.train's zrckp branch, as R2P trains zrckpb), every carve with the split's columns; then zwq's wq block, and LTM's
    stamp with the NEXT block (the scorer and each carve's column record)."""
    if name not in FITS:
        raise SystemExit(f"zltm: {name} is not one of LTM's fits")
    arm, seed, split = M1.ALL_FITS[name]
    M1.where_ok(env, "train")
    Z3P = import_arm()
    args = ["train", "--name", name, "--arm", arm, "--seed", str(seed), "--device", "cpu", "--cache-root",
            M1.ROOTS["cache"], "--host"]                  # zwq.train's order; DOB's card sets the device and threads
    rc, nxt = 1, None
    M1._ACTIVE[0] = name
    LOADED.clear()
    try:
        with M1.on_ml1() as WQ:
            if WQ.fit_name(arm, seed, split) != name:
                raise SystemExit(f"zltm: zwq names {arm}'s seed {seed} fit of {split} {WQ.fit_name(arm, seed, split)}")
            WQ.cache_root_ok(M1.ROOTS["cache"])
            WQ.gate_passed()
            with WQ.on_wq() as W, on_arm(W), on_split(split):
                if W.Z3P is not Z3P:
                    raise SystemExit("zltm: zwq's zp1c3p is not the one zrckpn was registered from")
                with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, split: name}):
                    rc = W.Z3P.train(args, split)
                nxt = next_block(split)
                if rc == 0:
                    used_ok(WQ.read_json(WQ.fit_dir(name) / "train.json"), name)
                    miss = sorted(set(nxt["training_carves"]) - set(nxt["carves"]))
                    if miss:
                        raise SystemExit(f"zltm: {name} trained without the columns of {miss}; unstamped")
                    sj = WQ.fit_dir(name) / "screen.json"
                    rec = WQ.read_json(sj)
                    rec["zwq_sha256"] = WQ.sha_src(WQ.__file__)
                    rec["wq"] = wq_block(name)
                    W.LC.write_json(sj, rec)
    finally:
        M1._ACTIVE[0] = None
        LOADED.clear()
    if rc == 0:
        M1.stamp(name, env, "screen", {"what": NAMING, "against": DOB.fit_name(BASE, seed, split), "next": nxt})
        log(f"train {name}: done, stamped (the scorer {nxt['fit_sha256'][:12]}, columns of {sorted(nxt['carves'])})")
    return rc


def read(name, env):
    """In place of zml1s.read for an LTM fit: zp1c3p's read for zrckpn under the arm swap, p@swa on the six s1eval carves
    at zrckp's read cap, on the device the fit was trained on, every carve's columns from the scorer its train read;
    read.json is stamped when the read exits 0 and read every eval carve's columns."""
    if name not in FITS:
        raise SystemExit(f"zltm: {name} is not one of LTM's fits")
    arm, seed, split = M1.ALL_FITS[name]
    M1.where_ok(env, "read")
    _rec, k = M1.fit_stamp(name)
    if M1.device_of(env) != k.get("device"):
        raise SystemExit(f"zltm: {name} was fitted on {k.get('device')}; this read runs on {M1.device_of(env)}")
    nk = k.get("next") or {}
    if (nk.get("declared_in"), nk.get("split"), nk.get("columns")) != (DECLARED, split, list(LTMC.COLS)) or \
            not nk.get("fit_sha256"):
        raise SystemExit(f"zltm: {name} was not trained on LTM's columns of {split} ({nk})")
    Z3P = import_arm()
    argv = ["--name", name, "--device", "cpu", "--host", "--cache-root", M1.ROOTS["cache"], "--rows-cap", str(ROWS_CAP)]
    rc, nxt = 1, None
    M1._ACTIVE[0] = name
    LOADED.clear()
    try:
        with M1.on_ml1() as WQ:
            WQ.cache_root_ok(M1.ROOTS["cache"])
            WQ.gate_passed()
            why = WQ.wq_why(name)
            if why:
                raise SystemExit(f"zltm: {why}")
            with WQ.on_wq() as W, on_arm(W), on_split(split, nk["fit_sha256"]):
                if W.Z3P is not Z3P:
                    raise SystemExit("zltm: zwq's zp1c3p is not the one zrckpn was registered from")
                rc = W.Z3P.read(argv)
                nxt = next_block(split)
    finally:
        M1._ACTIVE[0] = None
        LOADED.clear()
    if rc == 0:
        miss = sorted({f"{ds}/{LTMC.EVAL_CARVE}" for ds in M1.WQ.EVAL_ORDER} - set(nxt["carves"]))
        if miss:
            raise SystemExit(f"zltm: {name}'s read did not load the columns of {miss}; unstamped")
        M1.stamp(name, env, "read", {"next": nxt})
    return rc


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zltm: {name} is not one of LTM's fits ({', '.join(FITS)})")
    import_arm()                                    # outside zwq.on_wq
    with on_ltm():
        return M1.fit(name)


# ── the compares and the calls ───────────────────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: LTM's or DOB's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    if name in DOB.FITS:
        return DOB.fit_block(name, what, base)
    raise SystemExit(f"zltm: {name} is neither LTM's nor DOB's")


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != ROWS_CAP:
        raise SystemExit(f"zltm: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each pair whose two fits are read, through zwq's run_compare, stamped with LTM's block; a compare filed before is
    kept. The full run's pairs are compared once their fits exist."""
    done, ran, wait = [], [], []
    for cname, (new, bse) in {**PAIRS, **NULLS}.items():
        if not (is_read(new, base) and is_read(bse, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zltm: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zltm: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zltm_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


def r5_from(base=ROOT):
    """({(split, dataset): {seed: R@5}} of LTM's fits, the same of DOB's base, problems) from LTM's compares. The base's
    R@5 at a seed must be one number in every compare that holds it."""
    new_r5, base_r5, bad = {}, {}, []

    def put(rd, sd, v, cname):
        d = base_r5.setdefault(rd, {})
        if sd in d and abs(d[sd] - v) > RULE.EPS:
            bad.append(f"{rd[0]} {rd[1]}: the base's seed {sd} gives {d[sd]} in one compare and {v} in {cname}")
        d.setdefault(sd, v)

    for cname, (new, bse) in {**PAIRS, **NULLS}.items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            if cname in PAIRS:
                _a, sd, sp = FITS[new]
                rd = (sp, r["dataset"])
                new_r5.setdefault(rd, {})[sd] = r["new"][0]
                put(rd, sd, r["base"][0], cname)
            else:
                _a, sd, sp = DOB.FITS[new]
                rd = (sp, r["dataset"])
                put(rd, sd, r["new"][0], cname)
                put(rd, 0, r["base"][0], cname)
    return new_r5, base_r5, bad


def slices(new_r5, base_r5, base=ROOT):
    """({(split, slice): {seed: R@5}} of LTM's read fits, the same of DOB's base at its three seeds, problems), by
    zdep.slice_r5 (musique's s1eval read file by the questions' hop count). Each file's all-question R@5 must be its
    compare's musique R@5."""
    new_s, base_s, bad = {}, {}, []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            for f, own, ref in ((fit_name(sd, sp), new_s, new_r5), (DOB.fit_name(BASE, sd, sp), base_s, base_r5)):
                if f in FITS and not is_read(f, base):
                    continue
                s = DEP.slice_r5(f, base)
                if s is None:
                    bad.append(f"{f}: no {DEP.SLICE_DS} read file")
                    continue
                want = ref.get((sp, DEP.SLICE_DS), {}).get(sd)
                if want is not None and abs(s["all"] - want) > DEP.R5_TOL:
                    bad.append(f"{f}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in DEP.SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return new_s, base_s, bad


# ── LTM's rule (section 6): LTG's in-domain rule against DOB's base ──────────


def decide_screen(new, base):
    """The screen: LTG's per-read call (floor max(0.0075, 2 x the base's spread over its three seeds), on the seed-0 delta)
    on LTG's ten deciding reads: the eight reads of datasets their split trains on and the WQ-hotpotqa fits' musique 3-hop
    and 4-hop. A LOSS: NOT_ADOPTED; else a GAIN: ADOPT_FOR_FULL_RUN; else NO_GAIN. The four zero-shot reads and
    WQ-musique's slices are called beside."""
    rows, problems = [], []
    for rd in LTG.deciding_keys():
        b, n = base.get(rd, {}), new.get(rd, {})
        if any(k not in b for k in RULE.SEEDS) or 0 not in n:
            problems.append(f"{rd[0]} {rd[1]}: the base's seeds {sorted(b)}, zrckpn's {sorted(n)}")
            continue
        rows.append(LTG.screen_row(rd, b, n))
    side = [LTG.screen_row(rd, base[rd], new[rd]) for rd in LTG.beside_keys()
            if all(k in base.get(rd, {}) for k in RULE.SEEDS) and 0 in new.get(rd, {})]
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": rows, "beside": side, "counts": {}}
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    v = NOT_ADOPTED if n[RULE.LOSS] else ADOPT_FOR_FULL_RUN if n[RULE.GAIN] else NO_GAIN
    return {"verdict": v, "problems": [], "reads": rows, "beside": side, "counts": n}


def decide_full(new, base):
    """The full run: LTG's zml1.call on each deciding read at three seeds. ADOPT: no LOSS among the ten and a GAIN on at
    least one of the eight in-domain dataset reads; else NOT_ADOPTED."""
    rows = []
    for rd in LTG.deciding_keys():
        b, n = base.get(rd, {}), new.get(rd, {})
        if any(k not in b or k not in n for k in RULE.SEEDS):
            return {"verdict": INCOMPLETE, "problems": [f"{rd[0]} {rd[1]}: a seed is missing"], "reads": rows,
                    "beside": [], "counts": {}}
        rows.append(LTG.full_row(rd, b, n))
    side = [LTG.full_row(rd, base[rd], new[rd]) for rd in LTG.beside_keys()
            if all(k in base.get(rd, {}) and k in new.get(rd, {}) for k in RULE.SEEDS)]
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.UNCONFIRMED, RULE.WITHIN)}
    gain = any(r["call"] == RULE.GAIN and r["read"] in RULE.DATASETS for r in rows)
    return {"verdict": ADOPT if not n[RULE.LOSS] and gain else NOT_ADOPTED, "problems": [], "reads": rows, "beside": side,
            "counts": n}


def fit_problems(base=ROOT):
    """Why a read LTM fit is not the declared one: its device, threads and pick, its gate, its columns (one scorer for its
    train and read, the split's present scorer, every training carve in its train and every eval carve in its read, each
    carve's column record unchanged since), and its trainer's code against its base fit's (every file the base names)."""
    out = []
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    lo = Path(base) / OUT
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = DOB.fit_name(BASE, sd, sp)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        for what, k in (("train", ks), ("read", kr)):
            if k.get("device") != DOB.DEVICE or k.get("threads") != THREADS or k.get("pick") != list(PICK):
                out.append(f"{name}: its {what} ran on {k.get('device')} at {k.get('threads')} threads with the pick "
                           f"{k.get('pick')}")
        nt, nr = ks.get("next") or {}, kr.get("next") or {}
        fr = lo / f"fit_{sp}.json"
        fsha = M1.WQ.sha_file(fr) if fr.is_file() else None
        if not nt.get("fit_sha256") or nt.get("fit_sha256") != nr.get("fit_sha256") or nt.get("fit_sha256") != fsha:
            out.append(f"{name}: its train read the scorer {str(nt.get('fit_sha256'))[:12]}, its read "
                       f"{str(nr.get('fit_sha256'))[:12]}, {sp}'s present fit is {str(fsha)[:12]}")
        miss = sorted((set(f"{d}/{c}" for d, c in LTMC.train_carves(sp)) - set(nt.get("carves") or {}))
                      | ({f"{d}/{LTMC.EVAL_CARVE}" for d in M1.WQ.EVAL_ORDER} - set(nr.get("carves") or {})))
        if miss:
            out.append(f"{name}: no columns of {miss} in its train or read")
        for key, sha in {**(nt.get("carves") or {}), **(nr.get("carves") or {})}.items():
            ds, cv = key.split("/")
            p = LTMC.cols_dir(lo, sp, ds, cv) / "record.json"
            if not p.is_file() or M1.WQ.sha_file(p) != sha:
                out.append(f"{name}: the columns of {key} are not the ones it read")
        if kb.get("device") != ks.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def columns_beside(base=ROOT):
    """Deciding nothing: per split, its scorer's fit record sha and, per carve, the share of rows next to an anchor and
    the A lists' T2 under bil, pair and rrf (ltmcols' carve records)."""
    out = {}
    lo = Path(base) / OUT
    for sp in M1.SPLITS:
        fr = lo / f"fit_{sp}.json"
        d = {"fit_sha256": M1.WQ.sha_file(fr) if fr.is_file() else None, "carves": {}}
        for ds, cv in LTMC.col_carves(sp):
            p = LTMC.cols_dir(lo, sp, ds, cv) / "record.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            a = r.get("A_assigned") or {}
            d["carves"][f"{ds}/{cv}"] = {"training": r.get("training_carve"), "rows_with_an_anchor": r.get(
                "rows_with_an_anchor"), "A_ranked": r.get("A_ranked"), **{f"A_T2_{s}": (a.get(s) or {}).get("T2")
                                                                       for s in ("bil", "pair", "rrf")}}
        out[sp] = d
    return out


def beside(base=ROOT):
    """Deciding nothing: per read LTM fit, the seconds."""
    out = {}
    for name in FITS:
        if not is_read(name, base):
            continue
        tj = M1.screen_dir(base) / "fits" / name / "ml1_timing.json"
        tm = M1.WQ.read_json(tj) if tj.is_file() else {}
        out[name] = {"train_seconds": tm.get("train_seconds"), "read_seconds": tm.get("read_seconds")}
    return out


def calls(base=ROOT):
    new_r5, base_r5, bad = r5_from(base)
    new_s, base_s, bad_s = slices(new_r5, base_r5, base)
    bad += bad_s
    new_all, base_all = {**new_r5, **new_s}, {**base_r5, **base_s}
    screen = decide_screen(new_all, base_all)
    full_ready = all(is_read(f, base) for f in FITS)
    full = decide_full(new_all, base_all) if full_ready else None
    dep_rule = {"screen": DEP.decide_screen(new_all, base_all)["verdict"],
                "full_run": DEP.decide_full(new_r5, base_r5, new_s, base_s)["verdict"] if full_ready else None}
    bad += fit_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "what": NAMING, "base": f"DOB's CPU {BASE}",
           "device": DOB.DEVICE, "threads": THREADS, "pick": list(PICK), "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "problems": bad, "beside": beside(base), "columns": columns_beside(base),
           "claim1_reads": [list(rd) for rd in CLAIM1_READS], "zltm_sha256": M1.WQ.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)

    def mark(r):
        return " (C1R)" if (r["split"], r["read"]) in CLAIM1_READS else ""

    md = [f"# LTM: {ARM} ({NAMING}) against DOB's CPU {BASE}, on PKO's pools, on the host's CPU", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | zrckpn s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']}{mark(r)} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", "", f"counts: {json.dumps(full.get('counts', {}))}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in frows:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['read']}{mark(r)} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
                          f"{r['floor']:.4f} | {r['call']} |")
    cols = [(sp, k, c) for sp, d in rec["columns"].items() for k, c in d["carves"].items()]
    if cols:
        md += ["", "Beside, deciding nothing: the columns per carve (ltmcols' records):", "",
               "| split | carve | training | rows next to an anchor | A ranked | A T2 bil | pair | rrf |",
               "|---|---|---|---:|---:|---:|---:|---:|"]

        def t(v):
            return "-" if v is None else f"{v:.3f}"
        md += [f"| {sp} | {k} | {c['training']} | {t(c['rows_with_an_anchor'])} | {c['A_ranked']} | {t(c['A_T2_bil'])} | "
               f"{t(c['A_T2_pair'])} | {t(c['A_T2_rrf'])} |" for sp, k, c in cols]
    if rec["beside"]:
        md += ["", "| fit | train s | read s |", "|---|---:|---:|"]
        md += [f"| {n} | {b['train_seconds']} | {b['read_seconds']} |" for n, b in rec["beside"].items()]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; DEP's rule beside {dep_rule}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def must_stop(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit:
        return
    raise AssertionError(f"{getattr(fn, '__name__', fn)} passed")


def same(a, b):
    """Two batches' features equal, through tensors, arrays, tuples, lists, dicts and None."""
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and bool(torch.equal(a, b))
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def toy_cols(out, split, ds, carve, cache_root, rng, fit_sha, training=False):
    """ltmcols' layout for one toy carve: seven float16 columns per part (anchors' log-ratios with FILL off them, their max,
    the count of anchors), the part and carve records; the columns."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    cd = LTMC.cols_dir(out, split, ds, carve)
    ps, xs = [], []
    for p, r in zip(parts, recs):
        k = int(r["rows"])
        x = rng.uniform(-LTMC.CLIP, LTMC.CLIP, (k, LTMC.TOP)).astype(np.float32)
        x[rng.random((k, LTMC.TOP)) < 0.6] = LTMC.FILL
        C = np.empty((k, LTMC.WIDTH), np.float16)
        C[:, :LTMC.TOP] = x
        C[:, LTMC.TOP] = x.max(1)
        C[:, LTMC.TOP + 1] = (x > LTMC.FILL).sum(1)
        pd = cd / p.name
        pd.mkdir(parents=True, exist_ok=True)
        np.save(pd / "next.npy", C)
        sha = LC.sha_file(pd / "next.npy")
        M1.WQ.write_json(pd / "record.json", {
            "stage": LTMC.STAGE, "declared_in": LTMC.DECLARED, "split": split, "dataset": ds, "carve": carve,
            "part": p.name, "cache_record_sha256": LC.sha_file(p / "record.json"), "rows": k,
            "queries": int(r["queries"]), "columns": list(LTMC.COLS), "fit_sha256": fit_sha, "training_carve": training,
            "next_sha256": sha})
        ps.append({"part": p.name, "rows": k, "next_sha256": sha})
        xs.append(C)
    M1.WQ.write_json(cd / "record.json", {
        "stage": LTMC.STAGE, "declared_in": LTMC.DECLARED, "split": split, "dataset": ds, "carve": carve,
        "training_carve": training, "rows": int(sum(q["rows"] for q in ps)), "parts": ps, "columns": list(LTMC.COLS),
        "fit_sha256": fit_sha, "identity": {"pass": True}})
    return np.concatenate(xs)


def selftest():
    import shutil
    import tempfile
    import types
    import torch
    import lean_gpu as LG
    import zp1c2 as Z2
    # 1. names and pairs; the pick; the real modules hold what the patches replace
    assert fit_name(0, "WQ-hotpotqa") == "scr-ltm-zrckpn-s0-hp" and fit_name(2, "WQ-musique") == "scr-ltm-zrckpn-s2"
    assert pair_name(0, "WQ-hotpotqa") == "scr-ltm-zrckpn-vs-zrckp-s0-hp"
    assert null_name(1, "WQ-musique") == "scr-ltm-zrckp-s1-vs-s0"
    assert PAIRS[pair_name(1, "WQ-hotpotqa")] == ("scr-ltm-zrckpn-s1-hp", "scr-dob-zrckp-s1-hp")
    assert NULLS[null_name(2, "WQ-hotpotqa")] == ("scr-dob-zrckp-s2-hp", "scr-dob-zrckp-s0-hp")
    assert SCREEN == ("scr-ltm-zrckpn-s0", "scr-ltm-zrckpn-s0-hp")
    assert PICK[:-1] == DOB.PICK and PICK[-1] == NEXT and NEXT not in LG.LM.LEAN and NEXT not in DOB.PICK
    assert tuple(LG.SETS["pick"]) == DOB.PICK and Z2.DEVICE == "cuda" and callable(Z2.on_card)
    assert M1.STAMP["zrckp"] == P1C3P and ARM not in M1.STAMP and ARM not in M1.ROWS_CAP
    names = set(M1.fit.__code__.co_names)
    assert {"train", "read", "device_env", "where_ok", "stamped"} <= names, names
    assert "ok_where" in M1.where_ok.__code__.co_names and "ROWS_CAP" in M1.stamp.__code__.co_names
    assert len(LTG.deciding_keys()) == 10 and len(LTG.beside_keys()) == 6
    # 2. the arm: zrckp's model on zp1c3's chain carve with NEXT; registered once; zp1c3p's swap and its restoring
    Z3P = import_arm()
    S, Z3 = Z3P.S, Z3P.Z3
    model, carve = S.ARMS[ARM]
    assert model is Z3P.ZRCKP and carve.__mro__[1] is Z3.KTChainCarveZRC and carve.__name__ == "NextKTChainCarveZRC"
    assert not {"KT_ROOT", "CH_ROOT", "REL_DIR"} & set(vars(carve)) and import_arm() is Z3P and S.ARMS[ARM][1] is carve
    with M1.WQ.patch(Z3.KTChainCarveZRC, KT_ROOT=Path("kt_x")):
        assert carve.KT_ROOT == Path("kt_x")                  # zwq.on_wq's patch reaches zrckpn's carve
    assert carve.KT_ROOT == Z3.KTChainCarveZRC.KT_ROOT != Path("kt_x")
    W = types.SimpleNamespace(Z3P=Z3P, S=S)
    a0, s0 = Z3P.ARM, dict(Z3P.ARMS)
    with on_arm(W):
        assert Z3P.ARM == ARM and Z3P.ARMS == {ARM: S.ARMS[ARM]}
    assert (Z3P.ARM, Z3P.ARMS) == (a0, s0) == ("zrckp", {"zrckp": S.ARMS["zrckp"]})
    # 3. the columns beside a toy chain carve: loaded per split and checked; batches carry them for their own rows
    LG.bind_device_ops()
    RM, ZL, ZM = Z3P.RM, Z3P.ZL, Z3P.ZM
    tmp = Path(tempfile.mkdtemp(prefix="zltm_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = Z3.kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        ncls = next_carve(cls)
        out, sp, fsha = tmp / "ltm", "WQ-musique", "f" * 64
        X = toy_cols(out, sp, "metaqa", "toy", tmp / "cache", rng, fsha)
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert X.shape == (int(c.off_np[-1]), 7) and len(c.part_shas) == 2
        cd = LTMC.cols_dir(out, sp, "metaqa", "toy")
        rp = cd / "record.json"
        pp = cd / next(iter(c.part_shas)) / "record.json"     # the first part's record
        fp = pp.parent / "next.npy"

        def build(split=sp, **kw):
            with on_split(split, out=out, **kw):
                return ncls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")

        must_stop(ncls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")            # outside a split
        with on_split(sp, out=out):
            cn = ncls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            assert CUR["fit_sha256"] == fsha and [d["dataset"] for d in LOADED] == ["metaqa"]
            nb = next_block(sp)
            assert nb["carves"] == {"metaqa/toy": M1.WQ.sha_file(rp)} and nb["fit_sha256"] == fsha
        assert CUR == {"split": None, "fit_sha256": None, "out": None}
        LOADED.clear()
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        blocks = ["rank", "SEMB"]
        f0, nq0, bz0, g0 = c.batch(qs, blocks)
        f1, nq1, bz1, g1 = cn.batch(qs, blocks + [NEXT])
        assert set(f1) == set(f0) | {NEXT} and all(same(f0[k], f1[k]) for k in f0)
        assert torch.equal(nq0, nq1) and torch.equal(bz0, bz1) and torch.equal(g0, g1)
        want = np.concatenate([X[int(c.off_np[q]):int(c.off_np[q + 1])] for q in qs]).astype(np.float32)
        assert f1[NEXT].dtype == torch.float32 and np.array_equal(f1[NEXT].numpy(), want)
        assert same(cn.batch(qs, blocks)[0], f0) and NEXT not in cn.batch(qs, blocks)[0]
        assert torch.equal(cn.block_values(NEXT), torch.from_numpy(X)) and cn.widths[NEXT] == 7 and NEXT not in c.widths
        assert cn.nbytes() == c.nbytes() + X.nbytes and LG.dead_blocks([cn], [NEXT]) == []
        # the refusals: another split, a changed record, a part's cache record, a column file, two scorers in one fit
        must_stop(build, "WQ-hotpotqa")                        # no columns of the carve for that split
        orig = {p: p.read_bytes() for p in (rp, pp, fp)}
        for p, edit in ((rp, {"training_carve": True}), (rp, {"identity": {"pass": False}}), (rp, {"split": "WQ-hotpotqa"}),
                        (rp, {"columns": list(LTMC.COLS[:-1])}), (pp, {"cache_record_sha256": "0" * 64}),
                        (pp, {"fit_sha256": "e" * 64}), (pp, {"rows": 1})):
            M1.WQ.write_json(p, {**json.loads(orig[p]), **edit})
            must_stop(build)
            p.write_bytes(orig[p])
        assert build().next_record["fit_sha256"] == fsha         # every file put back
        Y = np.load(fp)
        Y[0, 0] = 1.0 if float(Y[0, 0]) != 1.0 else 2.0
        np.save(fp, Y)
        must_stop(build)                                       # the file is not its record's
        np.save(fp, np.load(io_bytes(orig[fp])).astype(np.float32))
        s32 = M1.WQ.sha_file(fp)
        r32 = json.loads(orig[rp])
        r32["parts"][0]["next_sha256"] = s32
        M1.WQ.write_json(rp, r32)
        M1.WQ.write_json(pp, {**json.loads(orig[pp]), "next_sha256": s32})
        must_stop(build)                                       # float32 with matching shas: not ltmcols' float16
        for p in (rp, pp, fp):
            p.write_bytes(orig[p])
        must_stop(build, fit_sha256="e" * 64)                  # a read whose train read another scorer
        assert CUR == {"split": None, "fit_sha256": None, "out": None}
        recs = [rp] + [cd / q / "record.json" for q in c.part_shas]
        saved_recs = {p: p.read_bytes() for p in recs}
        with on_split(sp, out=out):
            build_ok = ncls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            for p in recs:                                     # the whole carve's columns from another scorer
                M1.WQ.write_json(p, {**json.loads(saved_recs[p]), "fit_sha256": "e" * 64})
            LOADED.clear()
            with on_split(sp, out=out):
                assert ncls("metaqa", "toy", "2wiki", "cpu", tmp / "cache").next_record["fit_sha256"] == "e" * 64
            must_stop(ncls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")  # but not beside a carve of fsha's
            for p in recs:
                p.write_bytes(saved_recs[p])
        assert build_ok.next_record["fit_sha256"] == fsha
        LOADED.clear()
        # 4. zrckp's model with NEXT: three columns per NEXT column into its first layer (raw, z, keep); the output reads them
        B = qs.size
        nets = {}
        for k, bl in (("old", blocks), ("new", blocks + [NEXT])):
            torch.manual_seed(0)
            nets[k] = Z3P.ZRCKP(bl, cn.widths, 16, dropout=0.1, seed=0).eval()
        assert nets["new"].l1.in_features == nets["old"].l1.in_features + 2 * 7 + 1
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for prm in nets["new"].parameters():               # the start's zero output layer would hide every input
                prm.copy_(torch.randn(prm.shape, generator=gen) * 0.1)
            s1 = nets["new"](f1, torch.ones(B, 3), nq1, B, bz1)
            f3 = dict(f1)
            f3[NEXT] = f1[NEXT] + 1.0
            s3 = nets["new"](f3, torch.ones(B, 3), nq1, B, bz1)
        assert s1.shape == g1.shape, (s1.shape, g1.shape)
        assert bool(torch.isfinite(s1).all())
        assert not torch.equal(s1, s3), "the output does not read NEXT"
        # 5. a toy fit through lean_gpu's loop with NEXT under zrckpn's arm, and the trainer's sets check
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 2, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert LG.LeanMLP8D is Z3P.ZRCKP and LG.CacheCarve is carve
            fv = LG.fit_variant([cn], blocks + [NEXT], cfg, 0, 16, "none", "cpu", tag="toy-zrckpn")
        assert LG.LeanMLP8D is not Z3P.ZRCKP and LG.CacheCarve is not carve
        assert tuple(fv["swa"]["l1.weight"].shape) == (16, nets["new"].l1.in_features)
        used_ok({"sets": {"pick": list(PICK)}, "dead": []}, "x")
        for tj in ({"sets": {"pick": list(DOB.PICK)}, "dead": []}, {"sets": {"pick": list(PICK)}, "dead": [NEXT]},
                   {"sets": {}, "dead": []}, {"sets": {"pick": list(PICK), "pns": list(DOB.PICK)}, "dead": []}):
            must_stop(used_ok, tj, "x")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        LOADED.clear()
    # 6. the patches, on fakes, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    keep = ("DECLARED", "STAGE", "KEY", "pko_fit_name", "device_env", "ok_where", "stamp", "code_shas", "train", "read",
            "STAMP", "ROWS_CAP")
    saved = {k: getattr(M1, k) for k in keep}
    fz = types.SimpleNamespace(DEVICE="cuda", on_card=lambda argv: argv)
    fl = types.SimpleNamespace(SETS={"pick": list(DOB.PICK), "pns": [b for b in DOB.PICK if b != "SEMB"], "x": ["rank"]})
    sets0 = fl.SETS
    with on_ltm(fz, fl):
        assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
        assert M1.train is train and M1.read is read and M1.ok_where is DOB.cpu_ok and M1.device_env is DOB.cpu_env
        assert M1.STAMP[ARM] == P1C3P and M1.ROWS_CAP[ARM] == ROWS_CAP and M1.ROWS_CAP["zrckp"] == ROWS_CAP
        assert fz.DEVICE == "cpu" and fz.on_card(["--device", "cuda"]) == ["--device", "cpu", "--threads", str(THREADS)]
        assert fl.SETS["pick"] == list(PICK) and fl.SETS["pns"] == [b for b in PICK if b != "SEMB"] and fl.SETS["x"] == [
            "rank"]
        c = M1.code_shas(ARM)
        assert all(f"{m}_sha256" in c for m in M1.CODE + LTM_CODE)
        M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
        assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-ltm-zrckpn-s0-hp"
        assert M1.pko_fit_name("zrckp", 0, "WQ-hotpotqa") == "scr-ml1-zrckp-s0-hp"
        M1._ACTIVE[0] = None
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and {k: getattr(M1, k) for k in keep} == saved
    assert fl.SETS is sets0
    try:
        with on_ltm(fz, fl):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and {k: getattr(M1, k) for k in keep} == saved
    assert fl.SETS is sets0
    try:
        with on_ltm(fz, types.SimpleNamespace(SETS={"pick": list(DOB.PICK[:-1])})):
            raise AssertionError("a changed pick passed")
    except SystemExit:
        pass
    assert len(M1.ALL_FITS) == n0 and {k: getattr(M1, k) for k in keep} == saved
    # 7. the compares' and slices' reading, on records of the filed compares' shape (bases hold a Windows path)
    with tempfile.TemporaryDirectory() as td:
        scr = Path(td) / "outputs" / "screen"
        scr.mkdir(parents=True)
        b5 = {0: 0.50, 1: 0.51, 2: 0.49}                       # the base's R@5 at each seed on every read
        hot = ("WQ-hotpotqa", "musique")

        def write(cname, new, bse, sp, v_new, v_base):
            key = "outputs\\screen\\fits\\" + bse
            rows = [{"dataset": ds, "read": "in-domain", "questions": 10, "base": [v_base, 0.1, 0.1],
                     "new": [v_new + (0.02 if (sp, ds) == hot and cname in PAIRS else 0.0), 0.1, 0.1]}
                    for ds in RULE.DATASETS]
            M1.WQ.write_json(scr / f"{cname}.json", {
                "candidate": CAND, "carve": "s1eval", "bases": [key], "by_base": {key: {"rows": rows}},
                KEY: {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse}})

        for sp in M1.SPLITS:
            write(pair_name(0, sp), fit_name(0, sp), DOB.fit_name(BASE, 0, sp), sp, b5[0], b5[0])
            for sd in (1, 2):
                write(null_name(sd, sp), DOB.fit_name(BASE, sd, sp), DOB.fit_name(BASE, 0, sp), sp, b5[sd], b5[0])
        new, bas, bad = r5_from(td)
        assert not bad and all(bas[rd] == b5 for rd in RULE.READS) and all(sorted(new[rd]) == [0] for rd in RULE.READS)
        assert new[hot] == {0: 0.52}
        # slices: zdep.slice_r5 stood in for by the fits' made-up files; an LTM fit not read is skipped
        for sp in M1.SPLITS:
            (scr / "fits" / fit_name(0, sp)).mkdir(parents=True)
            (scr / "fits" / fit_name(0, sp) / "read.json").write_text("{}", encoding="utf-8")
        files = {}
        for f in list(SCREEN) + [DOB.fit_name(BASE, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS]:
            _a, sd, sp = FITS[f] if f in FITS else DOB.FITS[f]
            files[f] = {"all": (new if f in FITS else bas)[(sp, "musique")][sd],
                        "musique-3hop": 0.42 if f == fit_name(0, hot[0]) else 0.40, "musique-4hop": 0.30}
        with M1.WQ.patch(DEP, slice_r5=lambda f, base=None: files.get(f)):
            ns, bs, bad = slices(new, bas, td)
            assert not bad and ns[(hot[0], "musique-3hop")] == {0: 0.42} and bs[(hot[0], "musique-4hop")] == {
                0: 0.30, 1: 0.30, 2: 0.30} and sorted(ns) == sorted((s, sl) for s in M1.SPLITS for sl in DEP.SLICES)
            v = decide_screen({**new, **ns}, {**bas, **bs})    # floor 0.02 on musique, 0.0075 on its flat slices
            calls_of = {(r["split"], r["read"]): r["call"] for r in v["reads"]}
            assert v["verdict"] == ADOPT_FOR_FULL_RUN and len(v["reads"]) == 10 and len(v["beside"]) == 6, v
            assert calls_of[hot] == calls_of[(hot[0], "musique-3hop")] == RULE.GAIN and v["counts"][RULE.GAIN] == 2
            files[DOB.fit_name(BASE, 1, hot[0])] = None
            assert slices(new, bas, td)[2]                     # a base fit without its file is a problem
            files[DOB.fit_name(BASE, 1, hot[0])] = {"all": 0.6, "musique-3hop": 0.40, "musique-4hop": 0.30}
            assert slices(new, bas, td)[2]                     # a file whose R@5 is not its compare's
        sp = M1.SPLITS[1]
        write(null_name(1, sp), DOB.fit_name(BASE, 1, sp), DOB.fit_name(BASE, 0, sp), sp, b5[1], b5[0] + 0.001)
        assert r5_from(td)[2], "the base's seed 0 at two values passed"
        (scr / f"{null_name(2, sp)}.json").unlink()
        new, bas, _bad = r5_from(td)
        assert decide_screen(new, bas)["verdict"] == INCOMPLETE
    # 8. the rule on made-up reads: LTG's ten keys decide; the zero-shot reads and WQ-musique's slices do not
    keys = LTG.deciding_keys() + LTG.beside_keys()

    def made(vals):
        return {rd: dict(vals) for rd in keys}
    base3, flat = made(b5), made({0: 0.5, 1: 0.5, 2: 0.5})
    assert decide_screen(made({0: 0.50}), base3)["verdict"] == NO_GAIN
    g = made({0: 0.50})
    g[("WQ-hotpotqa", "musique-4hop")] = {0: 0.5075}
    assert decide_screen(g, flat)["verdict"] == ADOPT_FOR_FULL_RUN
    zs = made({0: 0.50})
    zs[LTG.ZERO_SHOT[0]] = {0: 0.40}                           # a zero-shot LOSS is beside
    zs[("WQ-musique", "musique-3hop")] = {0: 0.40}
    v = decide_screen(zs, flat)
    assert v["verdict"] == NO_GAIN and sum(r["call"] == RULE.LOSS for r in v["beside"]) == 2
    lo = made({0: 0.52})
    lo[("WQ-musique", "hotpotqa")] = {0: 0.48}
    assert decide_screen(lo, flat)["verdict"] == NOT_ADOPTED
    full = made({0: 0.50, 1: 0.50, 2: 0.50})
    full[("WQ-hotpotqa", "musique")] = {0: 0.52, 1: 0.51, 2: 0.51}
    assert decide_full(full, flat)["verdict"] == ADOPT
    full[("WQ-hotpotqa", "musique")] = {0: 0.50, 1: 0.50, 2: 0.50}
    full[("WQ-hotpotqa", "musique-3hop")] = {0: 0.52, 1: 0.51, 2: 0.51}   # a GAIN on a slice alone does not adopt
    assert decide_full(full, flat)["verdict"] == NOT_ADOPTED
    full[("WQ-hotpotqa", "musique")] = {0: 0.52, 1: 0.51, 2: 0.51}
    full[("WQ-musique", "2wiki")] = {0: 0.49, 1: 0.49, 2: 0.49}
    assert decide_full(full, flat)["verdict"] == NOT_ADOPTED
    del full[("WQ-musique", "2wiki")][2]
    assert decide_full(full, flat)["verdict"] == INCOMPLETE
    log("selftest: names, pairs and the pick; zrckpn's arm and swap; the columns beside a toy chain carve (alignment, "
        "widths, dead check, every refusal); the model and a toy fit with NEXT; the trainer's sets check; the patches and "
        "their restoring (also on a failure); the compares' and slices' reading; the rule pass")
    return 0


def io_bytes(b):
    import io
    return io.BytesIO(b)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
