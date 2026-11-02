"""LTC's stage (docs/LTC_CHAINED_BEAM_MLP.md): zrckpc, the MLP zrckp with LTM's seven next-hop columns (NEXT) and LTC's
fourteen chained-beam columns (CHAIN, built by ltccols.py) as input blocks, on PKO's pools under WQ, on the host's CPU
with DOB's device, against DOB's CPU zrckp base and its seed null; LTM's zrckpn read beside; the compares and the calls.

zrckpc is LTM's zrckpn with CHAIN beside NEXT. This file runs zltm.py's train, read and fit under LTC's names: as_ltc
sets zltm's arm, fits, pairs, nulls, declaration, stage, stamp key, naming, pick and code list to LTC's, and its arm
registration, split, NEXT block and set check to this file's, and puts every name back afterwards, also on a failure.
zltm's OUT stays LTM's: NEXT's columns and the split's scorer are LTM's, read only. A carve's CHAIN columns are read
from outputs/ltc and must come from the scorer and the LTM column record of the NEXT columns beside them. zml1s.py,
zdob.py, zp1c3p.py, zdep.py, zltg.py, zltm.py, ltmcols.py and ltccols.py are imported and not edited.

    python outputs/mp_unified/zltc.py fit --name scr-ltc-zrckpc-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zltc.py compare-all                       (against DOB's base, the base's null, LTM's zrckpn)
    python outputs/mp_unified/zltc.py calls                             (outputs/ltc/calls.json and .md)
    python outputs/mp_unified/zltc.py --selftest
"""
import os
import sys

THREADS = 6                                     # DOB's: every LTC fit's torch, OpenMP, MKL and OpenBLAS threads
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
import zltg as LTG  # noqa: E402  (LTG's in-domain rule: its keys and per-read calls, pure)
import zltm as LTM  # noqa: E402  (zrckpn's stage: its train, read, fit, NEXT carve, compares' reading and rule)
import ltmcols as LTMC  # noqa: E402  (NEXT's layout and carves)
import ltccols as LTCC  # noqa: E402  (CHAIN's layout and records)

DECLARED = "docs/LTC_CHAINED_BEAM_MLP.md"
STAGE = "LTC"
KEY = "ltc"                                     # the stamp's key in an LTC fit's screen.json and read.json and LTC's compares
OUT = "outputs/ltc"                             # calls.json and .md (the columns under cols/, built by ltccols.py)
ARM = "zrckpc"                                  # zrckp's model on zp1c3's chain carve with NEXT and CHAIN beside it
LTM_ARM = LTM.ARM                               # zrckpn, read beside
BASE = LTM.BASE                                 # DOB's CPU zrckp, fitted at three seeds on this device
NEXT = LTM.NEXT
CHAIN = "CHAIN"                                 # the block: ltccols' fourteen columns
PICK = LTM.PICK + (CHAIN,)                      # DOB's nine blocks, NEXT, then CHAIN
LTC_CODE = LTM.LTM_CODE + ("zltc", "ltccols")   # beside zml1s's code: LTM's list (zltm, zdob, ltmcols, ...), this stage
NAMING = ("zrckp + LTM's next-hop columns (NEXT) + LTC's chained beam of hop-wise softmaxes two and three links out from "
          "the first stage's top five rows (CHAIN)")
HOP2_MARGIN = 0.10                              # section 7 (a): depth 2's AUC this far below depth 1's names the hop-2 scorer
NEXT_FORM_CARVE = ("WQ-hotpotqa", "musique", "s1eval")      # section 7: musique in-domain
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT = (
    LTM.ADOPT_FOR_FULL_RUN, LTM.NO_GAIN, LTM.NOT_ADOPTED, LTM.INCOMPLETE, LTM.ADOPT)
assert LTM.THREADS == THREADS and BASE == "zrckp" and LTM_ARM == "zrckpn" and NEXT == "NEXT" and LTM.OUT == "outputs/ltm"
assert (LTCC.DECLARED, LTCC.STAGE, LTCC.WIDTH, LTMC.WIDTH) == (DECLARED, STAGE, 14, 7) and "zltm" in LTC_CODE


def fit_name(seed, split):
    return f"scr-ltc-{ARM}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    """zrckpc's fit against DOB's base fit of the same seed and split."""
    return f"scr-ltc-{ARM}-vs-{BASE}-s{seed}{M1.WQ.SUF[split]}"


def null_name(seed, split):
    """DOB's base at seed 1 or 2 against its own seed 0 (LTC's copy of the base's null)."""
    return f"scr-ltc-{BASE}-s{seed}-vs-s0{M1.WQ.SUF[split]}"


def ltm_pair_name(seed, split):
    """zrckpc's fit against LTM's zrckpn of the same seed and split (beside; section 7's head-to-head)."""
    return f"scr-ltc-{ARM}-vs-{LTM_ARM}-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
SCREEN = tuple(fit_name(0, sp) for sp in M1.SPLITS)
PAIRS = {pair_name(sd, sp): (fit_name(sd, sp), DOB.fit_name(BASE, sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
NULLS = {null_name(sd, sp): (DOB.fit_name(BASE, sd, sp), DOB.fit_name(BASE, 0, sp))
         for sd in RULE.SEEDS[1:] for sp in M1.SPLITS}
LTM_PAIRS = {ltm_pair_name(sd, sp): (fit_name(sd, sp), LTM.fit_name(sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and len(PAIRS) == 6 and len(NULLS) == 4 and len(LTM_PAIRS) == 6
assert not set(FITS) & (set(M1.ALL_FITS) | set(DOB.FITS) | set(LTG.FITS) | set(LTM.FITS))
assert all(b in DOB.FITS and DOB.DROP_OF[b] is None for _n, b in PAIRS.values())
assert all(b in LTM.FITS for _n, b in LTM_PAIRS.values())
assert not (set(PAIRS) | set(NULLS) | set(LTM_PAIRS)) & (set(LTM.PAIRS) | set(LTM.NULLS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zltc: {msg}", flush=True)


# ── the chain columns, beside NEXT's ─────────────────────────────────────────

CUR = {"split": None, "out": None}              # the split whose CHAIN columns the carves built now load, and their root
LOADED = []                                     # per carve loaded: dataset, carve, its chain record's sha256, its sources
_LTM_ON_SPLIT, _LTM_NEXT_BLOCK = LTM.on_split, LTM.next_block


def load_chain(ds, carve, split, part_shas, nrec, out=None, verify=True):
    """(rows x 14 float16 columns, the carve's chain record, that record's sha256) of one carve under the split, checked
    against the carve's cache parts and the NEXT columns beside them (nrec: the NEXT carve's record): the record's stage,
    declaration, split, carve, columns, beam, training flag and both identities; its scorer and its LTM column record must
    be NEXT's; each part's cache record sha256, rows, scorer and file (shape, type and, with verify, sha256)."""
    import lean_cache as LC
    cd = LTCC.cols_dir(LTCC.OUT if out is None else out, split, ds, carve)
    p = cd / "record.json"
    if not p.is_file():
        raise SystemExit(f"zltc: no LTC columns of {ds}/{carve} for {split} ({p})")
    rec = json.loads(p.read_text(encoding="utf-8"))
    want = (LTCC.STAGE, LTCC.DECLARED, split, ds, carve, list(LTCC.COLS), LTCC.BEAM,
            (ds, carve) in LTMC.train_carves(split))
    got = tuple(rec.get(k) for k in ("stage", "declared_in", "split", "dataset", "carve", "columns", "beam",
                                     "training_carve"))
    if got != want or not (rec.get("identity") or {}).get("pass") or not (rec.get("hop1_identity") or {}).get("pass"):
        raise SystemExit(f"zltc: {p} is {got} (identity {rec.get('identity')}, hop one {rec.get('hop1_identity')}), "
                         f"not LTC's columns {want}")
    if rec.get("fit_sha256") != nrec["fit_sha256"] or rec.get("ltm_record_sha256") != nrec["record_sha256"]:
        raise SystemExit(f"zltc: {p} was built from the scorer {str(rec.get('fit_sha256'))[:12]} and LTM's record "
                         f"{str(rec.get('ltm_record_sha256'))[:12]}; the NEXT columns beside it are "
                         f"{nrec['fit_sha256'][:12]}'s and {nrec['record_sha256'][:12]}")
    names = [q.get("part") for q in rec.get("parts") or []]
    if names != list(part_shas):
        raise SystemExit(f"zltc: {ds}/{carve}'s chain columns hold the parts {names}, its cache {list(part_shas)}")
    xs = []
    for q in rec["parts"]:
        pd = cd / q["part"]
        pr = json.loads((pd / "record.json").read_text(encoding="utf-8"))
        pw = (LTCC.STAGE, split, ds, carve, q["part"], part_shas[q["part"]], q["rows"], list(LTCC.COLS), rec["fit_sha256"],
              q["chain_sha256"])
        pg = tuple(pr.get(k) for k in ("stage", "split", "dataset", "carve", "part", "cache_record_sha256", "rows", "columns",
                                       "fit_sha256", "chain_sha256"))
        if pg != pw:
            raise SystemExit(f"zltc: {pd / 'record.json'} does not name its carve's part, cache record, rows and scorer")
        f = pd / "chain.npy"
        if verify and LC.sha_file(f) != q["chain_sha256"]:
            raise SystemExit(f"zltc: {f}: sha256 is not its record's")
        x = np.load(f)
        if x.dtype != np.float16 or x.shape != (q["rows"], LTCC.WIDTH):
            raise SystemExit(f"zltc: {f} is {x.dtype} {x.shape}, not float16 ({q['rows']}, {LTCC.WIDTH})")
        xs.append(x)
    X = np.concatenate(xs) if xs else np.zeros((0, LTCC.WIDTH), np.float16)
    if X.shape[0] != rec.get("rows"):
        raise SystemExit(f"zltc: {ds}/{carve}'s chain parts hold {X.shape[0]} rows, its record {rec.get('rows')}")
    return X, rec, LC.sha_file(p)


def chain_carve(base):
    """base's carve class with the split's NEXT columns (zltm.next_carve) and CHAIN columns beside it: batch() adds CHAIN
    for the batch's rows (the carve's own row order), block_values() hands it to lean_gpu's dead-block check, widths names
    its fourteen columns; NEXT and every other block, side key, row and gold are the NEXT carve's, unchanged."""
    import torch
    import lean_cache as LC
    nbase = LTM.next_carve(base)

    class ChainCarve(nbase):
        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            sp = CUR["split"]
            if sp is None or sp != LTM.CUR["split"]:
                raise SystemExit(f"zltc: {ds}/{carve} built for {ARM} outside an LTC split (CHAIN's {sp}, NEXT's "
                                 f"{LTM.CUR['split']})")
            X, rec, sha = load_chain(ds, carve, sp, self.part_shas, self.next_record, CUR["out"], verify)
            if X.shape[0] != int(self.off_np[-1]):
                raise SystemExit(f"zltc: {ds}/{carve}'s chain columns hold {X.shape[0]} rows, its carve "
                                 f"{int(self.off_np[-1])}")
            self.chain_X = torch.from_numpy(X).to(self.device)
            self.chain_record = {"dataset": ds, "carve": carve, "record_sha256": sha, "fit_sha256": rec["fit_sha256"],
                                 "ltm_record_sha256": rec["ltm_record_sha256"], "rows": int(X.shape[0])}
            self.widths = {**self.widths, CHAIN: LTCC.WIDTH}
            LOADED.append(self.chain_record)

        def nbytes(self):
            return super().nbytes() + self.chain_X.numel() * self.chain_X.element_size()

        def block_values(self, b):
            if b == CHAIN:
                return self.chain_X
            return super().block_values(b)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, [b for b in blocks if b != CHAIN])
            if CHAIN in blocks:
                qs = np.asarray(qs, np.int64)
                cnt = self.n_np[qs]
                idx = np.repeat(self.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
                x = self.chain_X[torch.from_numpy(idx).to(self.chain_X.device)].to(torch.float32)
                feats[CHAIN] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            return feats, nq, base_z, gold

    ChainCarve.__name__ = ChainCarve.__qualname__ = f"Chain{nbase.__name__}"
    return ChainCarve


_ARM = []


def import_arm():
    """zp1c3p (zrckp's model, zp1c3's chain carve), imported outside zwq.on_wq, and zrckpc in lean_screen's table: zrckp's
    model on the chain carve with NEXT and CHAIN beside it, registered once."""
    if _ARM:
        return _ARM[0]
    import zp1c3p as Z3P
    S = Z3P.S
    if S.ARMS.get("zrckp") != (Z3P.ZRCKP, Z3P.Z3.KTChainCarveZRC):
        raise SystemExit("zltc: lean_screen's zrckp is not zp1c3p's")
    if ARM in S.ARMS:
        raise SystemExit(f"zltc: lean_screen holds an arm {ARM} that is not this file's")
    S.ARMS[ARM] = (Z3P.ZRCKP, chain_carve(Z3P.Z3.KTChainCarveZRC))
    _ARM.append(Z3P)
    return Z3P


@contextlib.contextmanager
def on_split(split, fit_sha256=None, out=None, chain_out=None):
    """zltm's split (NEXT's columns from out, the scorer they must share when given) and LTC's (CHAIN's columns from
    chain_out); LOADED cleared on entry and exit; put back after, also on a failure."""
    saved = dict(CUR)
    LOADED.clear()
    with _LTM_ON_SPLIT(split, fit_sha256, out):
        CUR.update(split=split, out=chain_out)
        try:
            yield
        finally:
            CUR.clear()
            CUR.update(saved)
            LOADED.clear()


def next_block(split):
    """A fit's NEXT block, inside on_split (zltm's: the split, NEXT's columns, the scorer, each carve's NEXT record), with
    the chain's beside it: CHAIN's columns, the beam and each carve's chain record. CHAIN's carves must be NEXT's."""
    nb = _LTM_NEXT_BLOCK(split)
    ch = {f"{d['dataset']}/{d['carve']}": d["record_sha256"] for d in LOADED}
    if set(ch) != set(nb["carves"]):
        raise SystemExit(f"zltc: CHAIN was read on {sorted(ch)} and NEXT on {sorted(nb['carves'])}")
    nb["chain"] = {"declared_in": DECLARED, "columns": list(LTCC.COLS), "beam": LTCC.BEAM, "carves": ch,
                   "ltm_records": {f"{d['dataset']}/{d['carve']}": d["ltm_record_sha256"] for d in LOADED}}
    return nb


def used_ok(tj, name):
    """A fit whose trainer did not take NEXT and CHAIN into every set it fitted, or found either dead, is not LTC's: it
    stops unstamped."""
    sets, dead = tj.get("sets") or {}, tj.get("dead") or []
    if not sets or any(NEXT not in bl or CHAIN not in bl for bl in sets.values()) or NEXT in dead or CHAIN in dead:
        raise SystemExit(f"zltc: {name}'s trainer fitted the sets {sets} with the dead blocks {dead}; unstamped")


def ltc_names():
    """zltm's names as LTC's (OUT stays LTM's: NEXT's columns and the split's scorer)."""
    return {"ARM": ARM, "FITS": FITS, "SCREEN": SCREEN, "PAIRS": PAIRS, "NULLS": NULLS, "DECLARED": DECLARED,
            "STAGE": STAGE, "KEY": KEY, "NAMING": NAMING, "PICK": PICK, "LTM_CODE": LTC_CODE, "fit_name": fit_name,
            "pair_name": pair_name, "null_name": null_name, "import_arm": import_arm, "on_split": on_split,
            "next_block": next_block, "used_ok": used_ok}


@contextlib.contextmanager
def as_ltc():
    """zltm with LTC's names for the block; every name put back after, also on a failure."""
    with M1.WQ.patch(LTM, **ltc_names()):
        yield


def fit(name):
    """zltm's fit under LTC's names: train when needed (zp1c3p's train for zrckpc on DOB's CPU), then read."""
    if name not in FITS:
        raise SystemExit(f"zltc: {name} is not one of LTC's fits ({', '.join(FITS)})")
    with as_ltc():
        return LTM.fit(name)


# ── the compares ─────────────────────────────────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: LTC's (under LTC's names), LTM's or DOB's (zltm's own)."""
    if name in FITS:
        with as_ltc():
            return LTM.fit_block(name, what, base)
    return LTM.fit_block(name, what, base)


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != LTM.ROWS_CAP:
        raise SystemExit(f"zltc: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """Each pair whose two fits are read, through zwq's run_compare, stamped with LTC's block; a compare filed before is
    kept: zrckpc against DOB's base, the base's null under LTC's names, and zrckpc against LTM's zrckpn."""
    done, ran, wait = [], [], []
    for cname, (new, bse) in {**PAIRS, **NULLS, **LTM_PAIRS}.items():
        if not (LTM.is_read(new, base) and LTM.is_read(bse, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zltc: {new} and {bse} ran on different devices: {sorted(dev)}")
        with LTM.with_fits(), as_ltc(), LTM.on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zltc: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zltc_sha256": M1.WQ.sha_src(__file__), "zltm_sha256": M1.WQ.sha_src(LTM.__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


def ltm_r5(base=ROOT):
    """({(split, dataset): {seed: R@5}} of zrckpc, the same of LTM's zrckpn, problems) from LTC's zrckpc-vs-zrckpn
    compares."""
    zc, zn, bad = {}, {}, []
    for cname, (new, bse) in LTM_PAIRS.items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != LTM.CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            bad.append(f"{cname}: not a p@swa compare on s1eval against one base")
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            rd = (sp, r["dataset"])
            zc.setdefault(rd, {})[sd] = r["new"][0]
            zn.setdefault(rd, {})[sd] = r["base"][0]
    return zc, zn, bad


def ltm_side(new_r5, base=ROOT):
    """({(split, read): {seed: R@5}} of LTM's zrckpn, datasets and musique's slices, problems), from LTC's compares
    against it and zdep.slice_r5 on its read files. zrckpc's R@5 there must be its R@5 against DOB's base, and a slice
    file's R@5 over all questions its compare's."""
    zc, zn, bad = ltm_r5(base)
    for rd, d in zc.items():
        for sd, v in d.items():
            w = new_r5.get(rd, {}).get(sd)
            if w is not None and abs(w - v) > RULE.EPS:
                bad.append(f"{rd[0]} {rd[1]}: zrckpc's seed {sd} gives {w} against DOB's base and {v} against zrckpn")
    zs = {}
    for _c, (new, bse) in LTM_PAIRS.items():
        if not LTM.is_read(bse, base):
            continue
        _a, sd, sp = FITS[new]
        s = DEP.slice_r5(bse, base)
        if s is None:
            bad.append(f"{bse}: no {DEP.SLICE_DS} read file")
            continue
        want = zn.get((sp, DEP.SLICE_DS), {}).get(sd)
        if want is not None and abs(s["all"] - want) > DEP.R5_TOL:
            bad.append(f"{bse}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
        for sl in DEP.SLICES:
            if s[sl] is not None:
                zs.setdefault((sp, sl), {})[sd] = s[sl]
    return {**zn, **zs}, bad


# ── LTC's rule (section 5) is LTM's; the head-to-head and the next form (section 7) ──


def head_to_head(full, ltm_full, new, old, problems=()):
    """Section 7, after LTC's full run is ADOPT: if LTM's full run is not ADOPT, zrckpc is the candidate; if both are,
    LTG's full-run call of zrckpc against zrckpn at three seeds on the ten deciding reads: zrckpc on no LOSS and a GAIN
    on at least one of the eight dataset reads, else zrckpn."""
    if full is None or full["verdict"] != ADOPT:
        return {"verdict": "NOT_RUN", "candidate": None, "why": "LTC's full run is not ADOPT", "reads": [], "counts": {}}
    if ltm_full != ADOPT:
        return {"verdict": "NOT_NEEDED", "candidate": ARM, "why": f"LTM's full run is {ltm_full}", "reads": [],
                "counts": {}}
    if problems:
        return {"verdict": INCOMPLETE, "candidate": None, "why": "; ".join(problems), "reads": [], "counts": {}}
    rows = []
    for rd in LTG.deciding_keys():
        b, n = old.get(rd, {}), new.get(rd, {})
        if any(k not in b or k not in n for k in RULE.SEEDS):
            return {"verdict": INCOMPLETE, "candidate": None, "why": f"{rd[0]} {rd[1]}: a seed is missing", "reads": rows,
                    "counts": {}}
        rows.append(LTG.full_row(rd, b, n))
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.UNCONFIRMED, RULE.WITHIN)}
    win = not n[RULE.LOSS] and any(r["call"] == RULE.GAIN and r["read"] in RULE.DATASETS for r in rows)
    return {"verdict": "DECIDED", "candidate": ARM if win else LTM_ARM,
            "why": "no LOSS and an in-domain dataset GAIN against zrckpn" if win else "zrckpn kept", "reads": rows,
            "counts": n}


def next_form(base=ROOT):
    """Section 7's next form on NO_GAIN or NOT_ADOPTED, from the chain record of WQ-hotpotqa's musique/s1eval: (a)
    hop2_scorer when depth 2's AUC is more than HOP2_MARGIN below depth 1's, else (b) cumulative_residual."""
    sp, ds, cv = NEXT_FORM_CARVE
    p = LTCC.cols_dir(Path(base) / OUT, sp, ds, cv) / "record.json"
    if not p.is_file():
        return {"form": None, "why": f"no chain record of {sp} {ds}/{cv}"}
    x = json.loads(p.read_text(encoding="utf-8")).get("reached") or {}
    a1, a2 = ((x.get(k) or {}).get("auc_max_reached") for k in ("1", "2"))
    if a1 is None or a2 is None:
        return {"form": None, "why": f"AUC d1 {a1}, d2 {a2}"}
    return {"form": "hop2_scorer" if a2 < a1 - HOP2_MARGIN else "cumulative_residual", "auc_d1": a1, "auc_d2": a2,
            "margin": HOP2_MARGIN}


def chain_problems(name, split, ks, kr, base=ROOT):
    """Why a read LTC fit's CHAIN is not the declared one: its train and read each name LTC's chain block on the NEXT
    carves' set; each chain record they read is unchanged since, from the split's present LTM fit, and built on the NEXT
    record the fit read beside it."""
    out = []
    fr = Path(base) / LTM.OUT / f"fit_{split}.json"
    fsha = M1.WQ.sha_file(fr) if fr.is_file() else None
    lc = Path(base) / OUT
    for what, k in (("train", ks), ("read", kr)):
        nb = k.get("next") or {}
        ch = nb.get("chain") or {}
        if (ch.get("declared_in"), ch.get("columns"), ch.get("beam")) != (DECLARED, list(LTCC.COLS), LTCC.BEAM):
            out.append(f"{name}: its {what} names no LTC chain block ({ch.get('declared_in')}, beam {ch.get('beam')})")
            continue
        nc, cc = nb.get("carves") or {}, ch.get("carves") or {}
        if set(cc) != set(nc):
            out.append(f"{name}: its {what} read CHAIN on {sorted(cc)} and NEXT on {sorted(nc)}")
        for key, sha in cc.items():
            ds, cv = key.split("/")
            p = LTCC.cols_dir(lc, split, ds, cv) / "record.json"
            if not p.is_file() or M1.WQ.sha_file(p) != sha:
                out.append(f"{name}: the chain columns of {key} are not the ones its {what} read")
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if r.get("fit_sha256") != fsha:
                out.append(f"{name}: the chain columns of {key} come from the scorer {str(r.get('fit_sha256'))[:12]}; "
                           f"{split}'s present fit is {str(fsha)[:12]}")
            if r.get("ltm_record_sha256") != nc.get(key):
                out.append(f"{name}: the chain columns of {key} were built on another LTM record than the NEXT its "
                           f"{what} read")
    return out


def fit_problems(base=ROOT):
    """LTM's checks of a read fit (device, threads, pick, gate, NEXT's scorer and records, code against the base's) under
    LTC's names, and CHAIN's (chain_problems)."""
    with as_ltc():
        out = LTM.fit_problems(base)
    for name, (_a, _sd, sp) in FITS.items():
        if not LTM.is_read(name, base):
            continue
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        out += chain_problems(name, sp, ks, kr, base)
    return out


def columns_beside(base=ROOT):
    """Deciding nothing: per split and carve, ltccols' chain record: rows reached per depth, the missed golds reached, the
    AUC of each depth's max column over its reached rows, the hop-one identity's worst difference and the seconds."""
    out = {}
    lc = Path(base) / OUT
    for sp in M1.SPLITS:
        d = {}
        for ds, cv in LTMC.col_carves(sp):
            p = LTCC.cols_dir(lc, sp, ds, cv) / "record.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            x = r.get("reached") or {}

            def per(field):
                return {k: (x.get(k) or {}).get(field) for k in ("1", "2", "3")}
            d[f"{ds}/{cv}"] = {"training": r.get("training_carve"), "fit_sha256": r.get("fit_sha256"),
                               "rows": per("rows"), "missed_golds_reached": per("missed_golds_reached"),
                               "missed_reached_any": r.get("missed_reached_any"), "auc_max_reached": per("auc_max_reached"),
                               "hop1_max_abs": (r.get("hop1_identity") or {}).get("max_abs"), "seconds": r.get("seconds")}
        out[sp] = d
    return out


def calls(base=ROOT):
    with as_ltc():
        new_r5, base_r5, bad = LTM.r5_from(base)
        new_s, base_s, bad_s = LTM.slices(new_r5, base_r5, base)
        new_all, base_all = {**new_r5, **new_s}, {**base_r5, **base_s}
        screen = LTM.decide_screen(new_all, base_all)
        full_ready = all(LTM.is_read(f, base) for f in FITS)
        full = LTM.decide_full(new_all, base_all) if full_ready else None
        dep_rule = {"screen": DEP.decide_screen(new_all, base_all)["verdict"],
                    "full_run": DEP.decide_full(new_r5, base_r5, new_s, base_s)["verdict"] if full_ready else None}
        secs = LTM.beside(base)
    bad += bad_s + fit_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    zn_all, side_bad = ltm_side(new_r5, base)
    lp = Path(base) / LTM.OUT / "calls.json"
    lrec = M1.WQ.read_json(lp) if lp.is_file() else {}
    ltm_v = {"screen": (lrec.get("screen") or {}).get("verdict"), "full_run": (lrec.get("full_run") or {}).get("verdict")}
    h2h = head_to_head(full, ltm_v["full_run"], new_all, zn_all, side_bad)
    vs_ltm = [{"split": rd[0], "read": rd[1], "seed": sd, "zrckpc": round(new_all[rd][sd], 4),
               "zrckpn": round(zn_all[rd][sd], 4), "delta": round(new_all[rd][sd] - zn_all[rd][sd], 4)}
              for rd in LTG.deciding_keys() + LTG.beside_keys() for sd in RULE.SEEDS
              if sd in new_all.get(rd, {}) and sd in zn_all.get(rd, {})]
    nf = next_form(base)
    nf["applies"] = screen["verdict"] in (NO_GAIN, NOT_ADOPTED) or (full is not None and full["verdict"] == NOT_ADOPTED)
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "what": NAMING, "base": f"DOB's CPU {BASE}",
           "device": DOB.DEVICE, "threads": THREADS, "pick": list(PICK), "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "problems": bad, "beside": secs, "ltm_verdicts": ltm_v, "vs_zrckpn": vs_ltm,
           "vs_zrckpn_problems": side_bad, "head_to_head": h2h, "next_form_by_rule": nf,
           "chain_columns": columns_beside(base), "ltm_columns": LTM.columns_beside(base),
           "claim1_reads": [list(rd) for rd in LTM.CLAIM1_READS], "zltc_sha256": M1.WQ.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)

    def mark(r):
        return " (C1R)" if (r["split"], r["read"]) in LTM.CLAIM1_READS else ""

    def t(v):
        return "-" if v is None else f"{v:.3f}"
    md = [f"# LTC: {ARM} ({NAMING}) against DOB's CPU {BASE}, on PKO's pools, on the host's CPU", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"LTM's verdicts: screen {ltm_v['screen']}, full run {ltm_v['full_run']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | zrckpc s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']}{mark(r)} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", "", f"counts: {json.dumps(full.get('counts', {}))}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            md += [f"| {r['split']} | {r['read']}{mark(r)} | {r['delta'][0]:+.4f} | {r['delta'][1]:+.4f} | "
                   f"{r['delta'][2]:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in frows]
    md += ["", f"## Head-to-head (section 7): {h2h['verdict']}, candidate {h2h['candidate']}", "", h2h["why"]]
    if h2h["reads"]:
        md += ["", "| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {r['delta'][0]:+.4f} | {r['delta'][1]:+.4f} | {r['delta'][2]:+.4f} | "
               f"{r['floor']:.4f} | {r['call']} |" for r in h2h["reads"]]
    md += ["", f"## Next form by rule (section 7, applies: {nf['applies']}): {nf['form']}", "",
           f"WQ-hotpotqa musique/s1eval chain AUC d1 {t(nf.get('auc_d1'))}, d2 {t(nf.get('auc_d2'))} "
           f"(margin {HOP2_MARGIN}){'; ' + nf['why'] if nf.get('why') else ''}"]
    if vs_ltm:
        md += ["", "Beside, deciding nothing: zrckpc - zrckpn (LTM) per read and seed:", "",
               "| split | read | seed | zrckpc | zrckpn | delta |", "|---|---|---:|---:|---:|---:|"]
        md += [f"| {r['split']} | {r['read']}{mark(r)} | {r['seed']} | {r['zrckpc']:.4f} | {r['zrckpn']:.4f} | "
               f"{r['delta']:+.4f} |" for r in vs_ltm]
    cols = [(sp, k, c) for sp, d in rec["chain_columns"].items() for k, c in d.items()]
    if cols:
        md += ["", "Beside, deciding nothing: the chain columns per carve (ltccols' records; depth 1 is LTM's NEXT):", "",
               "| split | carve | training | rows d1, d2, d3 | missed golds d1, d2, d3 | any | AUC d1, d2, d3 | hop one | s |",
               "|---|---|---|---|---|---:|---|---:|---:|"]
        md += [f"| {sp} | {k} | {c['training']} | {', '.join(t(v) for v in c['rows'].values())} | "
               f"{', '.join(t(v) for v in c['missed_golds_reached'].values())} | {t(c['missed_reached_any'])} | "
               f"{', '.join(t(v) for v in c['auc_max_reached'].values())} | {t(c['hop1_max_abs'])} | "
               f"{t(c['seconds'])} |" for sp, k, c in cols]
    if secs:
        md += ["", "| fit | train s | read s |", "|---|---:|---:|"]
        md += [f"| {n} | {b['train_seconds']} | {b['read_seconds']} |" for n, b in secs.items()]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    if side_bad:
        md += ["", "problems beside (zrckpn's reads):"] + [f"- {p}" for p in side_bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; head-to-head {h2h['verdict']}; next form "
        f"{nf['form']} (applies {nf['applies']}); DEP's rule beside {dep_rule}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────

must_stop = LTM.must_stop


def toy_chain(out, split, ds, carve, cache_root, rng, fit_sha, ltm_sha, training=False):
    """ltccols' layout for one toy carve: fourteen float16 columns per part (-8 off the reach), the part and carve
    records; the columns."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    cd = LTCC.cols_dir(out, split, ds, carve)
    ps, xs = [], []
    for p, r in zip(parts, recs):
        k = int(r["rows"])
        C = rng.uniform(-LTCC.CLIP, LTCC.CLIP, (k, LTCC.WIDTH)).astype(np.float16)
        C[rng.random((k, LTCC.WIDTH)) < 0.5] = LTCC.FILL
        pd = cd / p.name
        pd.mkdir(parents=True, exist_ok=True)
        np.save(pd / "chain.npy", C)
        sha = LC.sha_file(pd / "chain.npy")
        M1.WQ.write_json(pd / "record.json", {
            "stage": LTCC.STAGE, "declared_in": LTCC.DECLARED, "split": split, "dataset": ds, "carve": carve,
            "part": p.name, "cache_record_sha256": LC.sha_file(p / "record.json"), "rows": k,
            "queries": int(r["queries"]), "columns": list(LTCC.COLS), "beam": LTCC.BEAM, "fit_sha256": fit_sha,
            "training_carve": training, "chain_sha256": sha})
        ps.append({"part": p.name, "rows": k, "chain_sha256": sha})
        xs.append(C)
    M1.WQ.write_json(cd / "record.json", {
        "stage": LTCC.STAGE, "declared_in": LTCC.DECLARED, "split": split, "dataset": ds, "carve": carve,
        "training_carve": training, "rows": int(sum(q["rows"] for q in ps)), "parts": ps, "columns": list(LTCC.COLS),
        "beam": LTCC.BEAM, "fit_sha256": fit_sha, "ltm_record_sha256": ltm_sha, "identity": {"pass": True},
        "hop1_identity": {"pass": True, "max_abs": 0.0}})
    return np.concatenate(xs)


def selftest():
    import shutil
    import tempfile
    import types
    import torch
    import lean_gpu as LG
    import zp1c2 as Z2
    # 1. names and pairs; the pick; the real modules hold what the patches replace
    assert fit_name(0, "WQ-hotpotqa") == "scr-ltc-zrckpc-s0-hp" and fit_name(2, "WQ-musique") == "scr-ltc-zrckpc-s2"
    assert pair_name(0, "WQ-hotpotqa") == "scr-ltc-zrckpc-vs-zrckp-s0-hp" and null_name(1, "WQ-musique") == \
        "scr-ltc-zrckp-s1-vs-s0"
    assert PAIRS[pair_name(2, "WQ-musique")] == ("scr-ltc-zrckpc-s2", "scr-dob-zrckp-s2")
    assert NULLS[null_name(2, "WQ-hotpotqa")] == ("scr-dob-zrckp-s2-hp", "scr-dob-zrckp-s0-hp")
    assert LTM_PAIRS[ltm_pair_name(1, "WQ-hotpotqa")] == ("scr-ltc-zrckpc-s1-hp", "scr-ltm-zrckpn-s1-hp")
    assert SCREEN == ("scr-ltc-zrckpc-s0", "scr-ltc-zrckpc-s0-hp")
    assert PICK[:-2] == DOB.PICK and PICK[-2:] == (NEXT, CHAIN) and CHAIN not in LG.LM.LEAN and CHAIN not in DOB.PICK
    assert LTC_CODE[:len(LTM.LTM_CODE)] == LTM.LTM_CODE and LTC_CODE[-2:] == ("zltc", "ltccols")
    assert tuple(LG.SETS["pick"]) == DOB.PICK and Z2.DEVICE == "cuda" and ARM not in M1.STAMP and ARM not in M1.ROWS_CAP
    # 2. as_ltc: zltm's names are LTC's in the block, put back after (also on a failure); OUT stays LTM's
    saved = {k: getattr(LTM, k) for k in ltc_names()}
    with as_ltc():
        assert all(getattr(LTM, k) is v for k, v in ltc_names().items()) and LTM.OUT == "outputs/ltm"
        assert LTM.fit_name(0, "WQ-musique") == "scr-ltc-zrckpc-s0" and LTM.LTM_CODE == LTC_CODE
    assert all(getattr(LTM, k) is v for k, v in saved.items()) and LTM.fit_name(0, "WQ-musique") == "scr-ltm-zrckpn-s0"
    try:
        with as_ltc():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert all(getattr(LTM, k) is v for k, v in saved.items())
    # 3. the arm: zrckp's model on zp1c3's chain carve with NEXT and CHAIN; registered once; zp1c3p's swap, restored
    Z3P = import_arm()
    S, Z3 = Z3P.S, Z3P.Z3
    model, carve = S.ARMS[ARM]
    assert model is Z3P.ZRCKP and carve.__mro__[2] is Z3.KTChainCarveZRC and carve.__name__ == "ChainNextKTChainCarveZRC"
    assert carve.__mro__[1].__name__ == "NextKTChainCarveZRC" and import_arm() is Z3P and S.ARMS[ARM][1] is carve
    assert not {"KT_ROOT", "CH_ROOT", "REL_DIR"} & (set(vars(carve)) | set(vars(carve.__mro__[1])))
    with M1.WQ.patch(Z3.KTChainCarveZRC, KT_ROOT=Path("kt_x")):
        assert carve.KT_ROOT == Path("kt_x")                  # zwq.on_wq's patch reaches zrckpc's carve
    W = types.SimpleNamespace(Z3P=Z3P, S=S)
    a0, s0 = Z3P.ARM, dict(Z3P.ARMS)
    with as_ltc(), LTM.on_arm(W):
        assert Z3P.ARM == ARM and Z3P.ARMS == {ARM: S.ARMS[ARM]}
    assert (Z3P.ARM, Z3P.ARMS) == (a0, s0) == ("zrckp", {"zrckp": S.ARMS["zrckp"]})
    # 4. NEXT and CHAIN beside a toy chain carve: loaded per split and checked; batches carry them for their own rows
    LG.bind_device_ops()
    RM, ZL, ZM = Z3P.RM, Z3P.ZL, Z3P.ZM
    tmp = Path(tempfile.mkdtemp(prefix="zltc_"))
    try:
        rng = np.random.default_rng(37)
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
        ccls = chain_carve(cls)
        lo, lc, sp, sp2, fsha = tmp / "ltm", tmp / "ltc", "WQ-musique", "WQ-hotpotqa", "f" * 64
        X = LTM.toy_cols(lo, sp, "metaqa", "toy", tmp / "cache", rng, fsha)
        LTM.toy_cols(lo, sp2, "metaqa", "toy", tmp / "cache", rng, fsha)        # NEXT on the other split, CHAIN not
        nrp = LTMC.cols_dir(lo, sp, "metaqa", "toy") / "record.json"
        Y = toy_chain(lc, sp, "metaqa", "toy", tmp / "cache", rng, fsha, M1.WQ.sha_file(nrp))
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert Y.shape == (int(c.off_np[-1]), 14) and X.shape[0] == Y.shape[0] and len(c.part_shas) == 2
        cd = LTCC.cols_dir(lc, sp, "metaqa", "toy")
        rp = cd / "record.json"
        pp = cd / next(iter(c.part_shas)) / "record.json"     # the first part's record
        fp = pp.parent / "chain.npy"

        def build(split=sp, **kw):
            try:
                with on_split(split, out=lo, chain_out=lc, **kw):
                    return ccls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            finally:
                LTM.LOADED.clear()

        must_stop(ccls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")            # outside a split
        with LTM.on_split(sp, out=lo):                                              # NEXT's split alone
            must_stop(ccls, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        LTM.LOADED.clear()
        with as_ltc(), on_split(sp, out=lo, chain_out=lc):
            cn = ccls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            assert CUR == {"split": sp, "out": lc} and LTM.CUR["fit_sha256"] == fsha
            nb = LTM.next_block(sp)                            # zltm's name: LTC's block under as_ltc
            assert (nb["declared_in"], nb["columns"], nb["fit_sha256"]) == (DECLARED, list(LTMC.COLS), fsha)
            assert nb["carves"] == {"metaqa/toy": M1.WQ.sha_file(nrp)} and nb["chain"]["beam"] == LTCC.BEAM
            assert nb["chain"]["carves"] == {"metaqa/toy": M1.WQ.sha_file(rp)} and nb["chain"]["columns"] == list(
                LTCC.COLS) and nb["chain"]["ltm_records"] == {"metaqa/toy": M1.WQ.sha_file(nrp)}
            LOADED.clear()
            must_stop(LTM.next_block, sp)                      # NEXT read on a carve CHAIN was not
        assert CUR == {"split": None, "out": None} and not LOADED
        assert LTM.CUR == {"split": None, "fit_sha256": None, "out": None}
        LTM.LOADED.clear()
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        blocks = ["rank", "SEMB"]
        f0, nq0, bz0, g0 = c.batch(qs, blocks)
        f1, nq1, bz1, g1 = cn.batch(qs, blocks + [NEXT, CHAIN])
        assert set(f1) == set(f0) | {NEXT, CHAIN} and all(LTM.same(f0[k], f1[k]) for k in f0)
        assert torch.equal(nq0, nq1) and torch.equal(bz0, bz1) and torch.equal(g0, g1)
        rows = np.concatenate([np.arange(int(c.off_np[q]), int(c.off_np[q + 1])) for q in qs])
        assert np.array_equal(f1[NEXT].numpy(), X[rows].astype(np.float32))
        assert f1[CHAIN].dtype == torch.float32 and np.array_equal(f1[CHAIN].numpy(), Y[rows].astype(np.float32))
        f2 = cn.batch(qs, blocks + [NEXT])[0]
        assert CHAIN not in f2 and LTM.same(f2[NEXT], f1[NEXT]) and CHAIN not in cn.batch(qs, blocks)[0]
        assert torch.equal(cn.block_values(CHAIN), torch.from_numpy(Y)) and cn.widths[CHAIN] == 14 and cn.widths[NEXT] == 7
        assert CHAIN not in c.widths and cn.nbytes() == c.nbytes() + X.nbytes + Y.nbytes
        assert LG.dead_blocks([cn], [NEXT, CHAIN]) == []
        # the refusals: another split; an edited carve or part record; a changed file; float32 under matching shas;
        # NEXT's record rewritten after the chain was built
        must_stop(build, sp2)                                  # NEXT's columns of that split, no CHAIN
        orig = {p: p.read_bytes() for p in (rp, pp, fp, nrp)}
        for p, edit in ((rp, {"training_carve": True}), (rp, {"identity": {"pass": False}}),
                        (rp, {"hop1_identity": {"pass": False}}), (rp, {"split": sp2}),
                        (rp, {"columns": list(LTCC.COLS[:-1])}), (rp, {"beam": 2}), (rp, {"fit_sha256": "e" * 64}),
                        (rp, {"ltm_record_sha256": "0" * 64}), (rp, {"stage": "LTM"}),
                        (pp, {"cache_record_sha256": "0" * 64}), (pp, {"fit_sha256": "e" * 64}), (pp, {"rows": 1})):
            M1.WQ.write_json(p, {**json.loads(orig[p]), **edit})
            must_stop(build)
            p.write_bytes(orig[p])
        assert build().chain_record["record_sha256"] == M1.WQ.sha_file(rp)       # every file put back
        Z = np.load(fp)
        Z[0, 0] = 1.0 if float(Z[0, 0]) != 1.0 else 2.0
        np.save(fp, Z)
        must_stop(build)                                       # the file is not its record's
        np.save(fp, np.load(LTM.io_bytes(orig[fp])).astype(np.float32))
        s32 = M1.WQ.sha_file(fp)
        r32 = json.loads(orig[rp])
        r32["parts"][0]["chain_sha256"] = s32
        M1.WQ.write_json(rp, r32)
        M1.WQ.write_json(pp, {**json.loads(orig[pp]), "chain_sha256": s32})
        must_stop(build)                                       # float32 with matching shas: not ltccols' float16
        for p in (rp, pp, fp):
            p.write_bytes(orig[p])
        M1.WQ.write_json(nrp, {**json.loads(orig[nrp]), "note": "rewritten after the chain"})
        assert build.__name__ and LTM.load_next("metaqa", "toy", sp, c.part_shas, lo)[1]["note"]   # NEXT itself loads
        must_stop(build)                                       # but the chain was built on NEXT's earlier record
        nrp.write_bytes(orig[nrp])
        assert build().chain_record["ltm_record_sha256"] == M1.WQ.sha_file(nrp)
        LOADED.clear()
        # 5. zrckp's model with NEXT and CHAIN: 15 and 29 more inputs to its first layer; the output reads CHAIN
        B = qs.size
        nets = {}
        for k, bl in (("old", blocks), ("next", blocks + [NEXT]), ("new", blocks + [NEXT, CHAIN])):
            torch.manual_seed(0)
            nets[k] = Z3P.ZRCKP(bl, cn.widths, 16, dropout=0.1, seed=0).eval()
        assert nets["next"].l1.in_features == nets["old"].l1.in_features + 2 * 7 + 1
        assert nets["new"].l1.in_features == nets["next"].l1.in_features + 2 * 14 + 1
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for prm in nets["new"].parameters():               # the start's zero output layer would hide every input
                prm.copy_(torch.randn(prm.shape, generator=gen) * 0.1)
            s1 = nets["new"](f1, torch.ones(B, 4), nq1, B, bz1)        # keep: one column per block
            f3 = dict(f1)
            f3[CHAIN] = f1[CHAIN] + 1.0
            s3 = nets["new"](f3, torch.ones(B, 4), nq1, B, bz1)
        assert s1.shape == g1.shape and bool(torch.isfinite(s1).all())
        assert not torch.equal(s1, s3), "the output does not read CHAIN"
        # 6. a toy fit through lean_gpu's loop with NEXT and CHAIN under zrckpc's arm, and the trainer's sets check
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 2, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert LG.LeanMLP8D is Z3P.ZRCKP and LG.CacheCarve is carve
            fv = LG.fit_variant([cn], blocks + [NEXT, CHAIN], cfg, 0, 16, "none", "cpu", tag="toy-zrckpc")
        assert LG.LeanMLP8D is not Z3P.ZRCKP and LG.CacheCarve is not carve
        assert tuple(fv["swa"]["l1.weight"].shape) == (16, nets["new"].l1.in_features)
        used_ok({"sets": {"pick": list(PICK)}, "dead": []}, "x")
        for tj in ({"sets": {"pick": list(LTM.PICK)}, "dead": []}, {"sets": {"pick": list(PICK)}, "dead": [CHAIN]},
                   {"sets": {"pick": list(PICK)}, "dead": [NEXT]}, {"sets": {}, "dead": []},
                   {"sets": {"pick": list(PICK), "pns": list(LTM.PICK)}, "dead": []}):
            must_stop(used_ok, tj, "x")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        LTM.LOADED.clear()
        LOADED.clear()
    # 7. zltm's patches under LTC's names, on fakes, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    keep = ("DECLARED", "STAGE", "KEY", "pko_fit_name", "device_env", "ok_where", "stamp", "code_shas", "train", "read",
            "STAMP", "ROWS_CAP")
    saved_m1 = {k: getattr(M1, k) for k in keep}
    fz = types.SimpleNamespace(DEVICE="cuda", on_card=lambda argv: argv)
    fl = types.SimpleNamespace(SETS={"pick": list(DOB.PICK), "pns": [b for b in DOB.PICK if b != "SEMB"], "x": ["rank"]})
    sets0 = fl.SETS
    with as_ltc(), LTM.on_ltm(fz, fl):
        assert len(M1.ALL_FITS) == n0 + 6 and all(M1.ALL_FITS[f] == v for f, v in FITS.items())
        assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and M1.train is LTM.train and M1.read is LTM.read
        assert M1.ok_where is DOB.cpu_ok and M1.device_env is DOB.cpu_env and fz.DEVICE == "cpu"
        assert M1.STAMP[ARM] == LTM.P1C3P and M1.ROWS_CAP[ARM] == LTM.ROWS_CAP and LTM_ARM not in M1.STAMP
        assert fl.SETS["pick"] == list(PICK) and fl.SETS["pns"] == [b for b in PICK if b != "SEMB"]
        cs = M1.code_shas(ARM)
        assert all(f"{m}_sha256" in cs for m in M1.CODE + LTC_CODE)
        M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
        assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-ltc-zrckpc-s0-hp"
        assert M1.pko_fit_name("zrckp", 0, "WQ-hotpotqa") == "scr-ml1-zrckp-s0-hp"
        M1._ACTIVE[0] = None
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and fl.SETS is sets0
    assert all(getattr(M1, k) is v for k, v in saved_m1.items()) and all(getattr(LTM, k) is v for k, v in saved.items())
    try:
        with as_ltc(), LTM.on_ltm(fz, fl):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert len(M1.ALL_FITS) == n0 and fz.DEVICE == "cuda" and fl.SETS is sets0
    assert all(getattr(M1, k) is v for k, v in saved_m1.items()) and all(getattr(LTM, k) is v for k, v in saved.items())
    # 8. the compares' reading under LTC's names, zrckpn's beside, and the calls on records of the filed shapes
    with tempfile.TemporaryDirectory() as td:
        scr = Path(td) / "outputs" / "screen"
        scr.mkdir(parents=True)
        b5 = {0: 0.50, 1: 0.51, 2: 0.49}                       # the base's R@5 at each seed on every read
        hot = ("WQ-hotpotqa", "musique")

        def write(cname, new, bse, sp, v_new, v_base):
            key = "outputs\\screen\\fits\\" + bse
            rows = [{"dataset": ds, "read": "in-domain", "questions": 10, "base": [v_base, 0.1, 0.1],
                     "new": [v_new + (0.02 if (sp, ds) == hot and new in FITS else 0.0), 0.1, 0.1]}
                    for ds in RULE.DATASETS]
            M1.WQ.write_json(scr / f"{cname}.json", {
                "candidate": LTM.CAND, "carve": "s1eval", "bases": [key], "by_base": {key: {"rows": rows}},
                KEY: {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse}})

        for sp in M1.SPLITS:
            write(pair_name(0, sp), fit_name(0, sp), DOB.fit_name(BASE, 0, sp), sp, b5[0], b5[0])
            write(ltm_pair_name(0, sp), fit_name(0, sp), LTM.fit_name(0, sp), sp, b5[0], 0.505)
            for sd in (1, 2):
                write(null_name(sd, sp), DOB.fit_name(BASE, sd, sp), DOB.fit_name(BASE, 0, sp), sp, b5[sd], b5[0])
        assert LTM.r5_from(td) == ({}, {}, [])                 # LTM's own names see none of LTC's compares
        with as_ltc():
            new, bas, bad = LTM.r5_from(td)
        assert not bad and all(bas[rd] == b5 for rd in RULE.READS) and new[hot] == {0: 0.52} and len(new) == 12
        for f in list(SCREEN) + [LTM.fit_name(0, sp) for sp in M1.SPLITS]:
            (scr / "fits" / f).mkdir(parents=True)
            (scr / "fits" / f / "read.json").write_text("{}", encoding="utf-8")
        files = {f: {"all": 0.505, "musique-3hop": 0.41, "musique-4hop": 0.30} for f in
                 [LTM.fit_name(0, sp) for sp in M1.SPLITS]}
        for f in list(SCREEN) + [DOB.fit_name(BASE, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS]:
            _a, sd, sp = FITS[f] if f in FITS else DOB.FITS[f]
            files[f] = {"all": (new if f in FITS else bas)[(sp, "musique")][sd],
                        "musique-3hop": 0.43 if f == fit_name(0, hot[0]) else 0.40, "musique-4hop": 0.30}
        with M1.WQ.patch(DEP, slice_r5=lambda f, base=None: files.get(f)):
            zn, side_bad = ltm_side(new, td)
            assert not side_bad and zn[hot] == {0: 0.505} and zn[(hot[0], "musique-3hop")] == {0: 0.41}
            files[LTM.fit_name(0, hot[0])] = {"all": 0.6, "musique-3hop": 0.41, "musique-4hop": 0.30}
            assert ltm_side(new, td)[1]                        # a slice file whose R@5 is not its compare's
            files[LTM.fit_name(0, hot[0])]["all"] = 0.505
            calls(td)                                          # the fits hold no stamps: INCOMPLETE, every table drawn
            rec = json.loads((Path(td) / OUT / "calls.json").read_text(encoding="utf-8"))
            md = (Path(td) / OUT / "calls.md").read_text(encoding="utf-8")
            assert rec["screen"]["verdict"] == INCOMPLETE and len(rec["screen"]["reads"]) == 10 and rec["problems"]
            assert {(r["split"], r["read"]): r["call"] for r in rec["screen"]["reads"]}[(hot[0], "musique-3hop")] == \
                RULE.GAIN
            assert rec["head_to_head"]["verdict"] == "NOT_RUN" and rec["ltm_verdicts"] == {"screen": None, "full_run": None}
            vs = {(r["split"], r["read"], r["seed"]): r["delta"] for r in rec["vs_zrckpn"]}
            assert vs[(hot[0], "musique", 0)] == round(0.52 - 0.505, 4) and vs[(hot[0], "musique-3hop", 0)] == 0.02
            assert "## Screen: INCOMPLETE" in md and "zrckpc - zrckpn" in md and "Head-to-head" in md
        write(ltm_pair_name(0, hot[0]), fit_name(0, hot[0]), LTM.fit_name(0, hot[0]), hot[0], b5[0] + 0.01, 0.505)
        assert ltm_r5(td)[0][hot] == {0: 0.53} and ltm_side(new, td)[1]   # zrckpc at two values
    # 9. the head-to-head, the next form and the chain checks on made-up records
    keys = LTG.deciding_keys() + LTG.beside_keys()

    def made(vals):
        return {rd: dict(vals) for rd in keys}
    flat, adopt = made({0: 0.5, 1: 0.5, 2: 0.5}), {"verdict": ADOPT}
    assert head_to_head(None, ADOPT, flat, flat)["verdict"] == "NOT_RUN"
    assert head_to_head({"verdict": NOT_ADOPTED}, ADOPT, flat, flat)["candidate"] is None
    for lv in (None, NOT_ADOPTED, INCOMPLETE):
        v = head_to_head(adopt, lv, flat, flat)
        assert (v["verdict"], v["candidate"]) == ("NOT_NEEDED", ARM)
    up = made({0: 0.5, 1: 0.5, 2: 0.5})
    up[("WQ-hotpotqa", "musique")] = {0: 0.52, 1: 0.51, 2: 0.51}
    v = head_to_head(adopt, ADOPT, up, flat)
    assert (v["verdict"], v["candidate"]) == ("DECIDED", ARM) and len(v["reads"]) == 10
    assert head_to_head(adopt, ADOPT, flat, flat)["candidate"] == LTM_ARM            # no GAIN: zrckpn kept
    sl = made({0: 0.5, 1: 0.5, 2: 0.5})
    sl[("WQ-hotpotqa", "musique-3hop")] = {0: 0.52, 1: 0.51, 2: 0.51}
    assert head_to_head(adopt, ADOPT, sl, flat)["candidate"] == LTM_ARM              # a slice GAIN alone does not
    lo_ = dict(up)
    lo_[("WQ-musique", "2wiki")] = {0: 0.48, 1: 0.48, 2: 0.48}
    assert head_to_head(adopt, ADOPT, lo_, flat)["candidate"] == LTM_ARM             # a LOSS
    zs = dict(up)
    zs[LTG.ZERO_SHOT[0]] = {0: 0.40, 1: 0.40, 2: 0.40}
    assert head_to_head(adopt, ADOPT, zs, flat)["candidate"] == ARM                  # a zero-shot LOSS is not deciding
    assert head_to_head(adopt, ADOPT, made({0: 0.5, 1: 0.5}), flat)["verdict"] == INCOMPLETE
    assert head_to_head(adopt, ADOPT, up, flat, ["x"])["verdict"] == INCOMPLETE
    with tempfile.TemporaryDirectory() as td:
        assert next_form(td)["form"] is None
        p = LTCC.cols_dir(Path(td) / OUT, *NEXT_FORM_CARVE) / "record.json"
        p.parent.mkdir(parents=True)
        for a1, a2, want in ((0.80, 0.69, "hop2_scorer"), (0.80, 0.71, "cumulative_residual"), (0.80, None, None)):
            M1.WQ.write_json(p, {"reached": {"1": {"auc_max_reached": a1}, "2": {"auc_max_reached": a2}}})
            assert next_form(td)["form"] == want, (a1, a2)
        lt = Path(td) / LTM.OUT
        lt.mkdir(parents=True)
        (lt / "fit_WQ-musique.json").write_text("{}", encoding="utf-8")
        fs = M1.WQ.sha_file(lt / "fit_WQ-musique.json")
        cp = LTCC.cols_dir(Path(td) / OUT, "WQ-musique", "musique", "s1eval") / "record.json"
        cp.parent.mkdir(parents=True)
        M1.WQ.write_json(cp, {"fit_sha256": fs, "ltm_record_sha256": "a" * 64})

        def stamp(next_carves, chain_carves, **ch):
            return {"next": {"carves": next_carves, "chain": {"declared_in": DECLARED, "columns": list(LTCC.COLS),
                                                              "beam": LTCC.BEAM, "carves": chain_carves, **ch}}}
        good = stamp({"musique/s1eval": "a" * 64}, {"musique/s1eval": M1.WQ.sha_file(cp)})
        assert chain_problems("x", "WQ-musique", good, good, td) == []
        for bad_k in ({"next": {"carves": {"musique/s1eval": "a" * 64}}},                  # no chain block
                      stamp({"musique/s1eval": "a" * 64}, {"musique/s1eval": M1.WQ.sha_file(cp)}, beam=2),
                      stamp({"musique/s1eval": "a" * 64, "2wiki/s1eval": "b" * 64},    # NEXT on a carve CHAIN is not
                            {"musique/s1eval": M1.WQ.sha_file(cp)}),
                      stamp({"musique/s1eval": "a" * 64}, {"musique/s1eval": "0" * 64}),   # the record changed since
                      stamp({"musique/s1eval": "c" * 64}, {"musique/s1eval": M1.WQ.sha_file(cp)})):   # another NEXT
            assert chain_problems("x", "WQ-musique", good, bad_k, td), bad_k
        M1.WQ.write_json(cp, {"fit_sha256": "e" * 64, "ltm_record_sha256": "a" * 64})       # another scorer
        other = stamp({"musique/s1eval": "a" * 64}, {"musique/s1eval": M1.WQ.sha_file(cp)})
        assert chain_problems("x", "WQ-musique", other, other, td)
    log("selftest: names, pairs and the pick; as_ltc and its restoring; zrckpc's arm and swap; NEXT and CHAIN beside a "
        "toy chain carve (alignment, widths, dead check, every refusal); the model (15 + 29 inputs) and a toy fit; the "
        "trainer's sets check; zltm's patches under LTC's names and their restoring (also on a failure); the compares' "
        "reading, zrckpn's beside and the calls; the head-to-head, the next form and the chain checks pass")
    return 0


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
