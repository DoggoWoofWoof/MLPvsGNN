"""GNS's stage (docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md): GNT's learned attention (zgnt.py) with its places moved from the
first stage's top five rows of each question to the base GNN's own five highest rows of a first pass (GNR's, read only).
Every pool row's l1 state gains W_m applied to a weighted sum of the l1 states of those five rows of its question (itself
left out), the weights a softmax over them of u_c W_a u_k + wp * pair_k + wr * res_k + b_k: GNT's term, parameters and
start (W_m and W_a zero, so a fit starts as the GNN of record bit for bit), on GNS's fifteen columns (gnscols.py: GNW's
rule and layout against the first pass's five rows). On PKO's roots, on the host's GPU: the screen at seed 0 on both WQ
splits against the base's seed-0 fits under LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), DEP's sixteen-read
rule beside.

zgnt.py, zgnr.py, zgnw.py, zgnx.py, zgnp.py, gnscols.py, gnrcols.py and gnwcols.py, and every module they import, are
imported and not edited. For the length of a GNS call zgnx's names (its fits, declaration, stage, stamp key, output root,
fit and pair names, code, term record, NEXT block and model patch: zgnt's) are GNS's, its block width is fifteen columns,
and ltmcols' root, stage, declaration, column names and width are GNS's for the call's seed (outputs/gns/s<seed>): zgnx's
carve then loads GNS's columns through zltm.load_next and refuses LTM's, GNW's and GNR's. Everything else in a fit is
GNX's: the carve, the base, the start, the trainer and the read. Every patched name is put back afterwards, also on a
failure. A fit refuses to start unless its split's and seed's first pass is one GNR's fits may stand on
(zgnr.first_problems), the columns' match passed, and its split's column check passed on that first pass (section 4).

    python outputs/mp_unified/zgns.py check-cols --split WQ-musique  (GNS's seed-0 columns of every carve the split loads)
    python outputs/mp_unified/zgns.py fit --name scr-gns-sa-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zgns.py compare-all                  (each GNS fit read against the base's fit of its seed)
    python outputs/mp_unified/zgns.py calls                        (outputs/gns/calls.json and .md)
    python outputs/mp_unified/zgns.py s0stop-check                 (s0stop's check under GNS's entry)
    python outputs/mp_unified/zgns.py s0stop-run -- COMMAND ...    (s0stop's run under GNS's entry)
    python outputs/mp_unified/zgns.py gate --want ADOPT_FOR_FULL_RUN (s0stop's gate under GNS's entry)
    python outputs/mp_unified/zgns.py --selftest
"""
import argparse
import contextlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zgnt as GNT  # noqa: E402  (GNT's term and model patch, which GNS's fits take as they are; not edited)
import zgnr as GNR  # noqa: E402  (GNR's first passes: their checks and beside; not edited)
import zgnw as GNW  # noqa: E402  (GNW's fit checks; not edited)
import zgnx as GNX  # noqa: E402  (GNX's arm, carve, stamp, compares and the pieces of its calls; not edited)
import zgnp as GNP  # noqa: E402  (the beside reads; not edited)
import gnscols as GNSC  # noqa: E402  (GNS's columns: their root per seed, names, layout and scorer; not edited)
import gnrcols as GNRC  # noqa: E402  (GNR's first passes' root; not edited)
import gnwcols as GNWC  # noqa: E402  (GNW's columns: the beside's root and declaration; not edited)
import s0stop as S0  # noqa: E402  (the seed-0 stop, with GNS's entry patched in for a call; not edited)

M1, RULE, ISO, DEP, LTG, LTM, LTMC = GNX.M1, GNX.RULE, GNX.ISO, GNX.DEP, GNX.LTG, GNX.LTM, GNX.LTMC
DECLARED = "docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md"
STAGE = "GNS"
KEY = "gns"                                     # the stamp's key in a GNS fit's screen.json and read.json and GNS's compares
OUT = "outputs/gns"
TAG = "sa"
NEXT = GNX.NEXT
TOP, WIDTH, IN = GNT.TOP, GNT.WIDTH, GNT.IN     # GNW's layout: pair_k, res_k, anchor_k for the five places
TERM_NAMES, TERM_PARAMS = GNT.TERM_NAMES, GNT.TERM_PARAMS
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    GNX.ADOPT_FOR_FULL_RUN, GNX.NO_GAIN, GNX.NOT_ADOPTED, GNX.ADOPT, GNX.INCOMPLETE)
PATCHED = GNT.PATCHED
COLS_PATCHED = GNT.COLS_PATCHED
ORIG = {k: getattr(GNX, k) for k in ("code_extra", "next_block")}
GNX_FITS = dict(GNX.FITS)                       # zgnx's own fit names, taken before any patch
OWN_CODE = ("zgns", "gnscols", "zgnr", "gnrcols") + GNT.OWN_CODE   # then GNT's: zgnt, zgnw, gnwcols, gnpcols, zpfp
assert (GNSC.STAGE, GNSC.DECLARED, Path(GNSC.OUT)) == (STAGE, DECLARED, ROOT / OUT) and GNSC.LTMC is LTMC
assert (WIDTH, IN, TOP, TERM_PARAMS, NEXT) == (15, 15, 5, 32775, "NEXT") and tuple(GNSC.COLS) == tuple(GNWC.COLS)
assert ORIG["next_block"] is GNT.ORIG["next_block"] and Path(GNRC.OUT) == ROOT / GNR.OUT


def fit_name(seed, split):
    return f"scr-gns-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-gns-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (GNX.ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS)
                                            | set(GNX.FITS) | set(GNP.FITS) | set(GNW.FITS) | set(GNT.FITS)
                                            | set(GNR.FITS))
CUR = {"seed": None, "out": None, "gnr": None}  # inside on_gns: the seed whose columns, GNS's root, GNR's root


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zgns: {msg}", flush=True)


def own_shas():
    return {"zgns_sha256": M1.WQ.sha_src(__file__), "gnscols_sha256": M1.WQ.sha_src(GNSC.__file__),
            **GNR.own_shas(), **GNT.own_shas()}


def code_extra():
    """GNX's code beside zml1s's (zgnx, zdep, zltg, zltm, ltmcols), and GNS's own: this file, gnscols.py, zgnr.py,
    gnrcols.py, zgnt.py, zgnw.py, gnwcols.py, gnpcols.py and zpfp.py."""
    return {**ORIG["code_extra"](), **own_shas()}


def term_record(G):
    return {**GNT.term_record(G),
            "term": "GNS's learned attention: GNT's term (each row's l1 state gains W_m applied to the softmax over its "
                    "question's places, itself left out, of u_c W_a u_k + wp * pair_k + wr * res_k + b_k weighting those "
                    "rows' l1 states; W_m and W_a zero-started) with the places the base GNN's own five highest rows of a "
                    "first pass (GNR's), not the first stage's top five"}


def cols_root(seed, out=None):
    """GNS's column root of a seed (out: GNS's root, or a test's)."""
    return GNSC.seed_root(int(seed), Path(GNSC.OUT if out is None else out))


def first_path(seed, split, gnr_out=None):
    """The first pass's record of a split and seed under GNR's root (gnr_out: a test's)."""
    return GNR.first_path(seed, split, gnr_out)


def check_path(seed, split, out=None):
    return cols_root(seed, out) / f"check_{split}.json"


def next_block(split):
    """GNX's NEXT block with GNS's declaration, the columns' root (inside on_gns: GNS's of the call's seed) and
    declaration, the seed and the first pass the columns stand on (its record's path and sha256; None when there is
    none)."""
    if CUR["seed"] is None:
        raise SystemExit("zgns: a NEXT block outside on_gns")
    fp = first_path(CUR["seed"], split, CUR["gnr"])
    return {**ORIG["next_block"](split), "declared_in": DECLARED, "columns_root": M1.WQ.rel(Path(LTMC.OUT)),
            "columns_declared_in": LTMC.DECLARED, "seed": CUR["seed"], "first": M1.WQ.rel(fp),
            "first_sha256": GNR.sha_or_none(fp)}


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_cols(seed=0, out=None):
    """ltmcols' root, stage, declaration, column names and width GNS's for the seed (out: GNS's root, or a test's; the
    columns under its s<seed>); restored after, also on a failure."""
    with M1.WQ.patch(LTMC, OUT=cols_root(seed, out), STAGE=STAGE, DECLARED=DECLARED, COLS=GNSC.COLS, WIDTH=WIDTH):
        yield


@contextlib.contextmanager
def on_gns(seed=0, out=None, gnr_out=None):
    """zgnx with GNS's names, code, term record, NEXT block and width, and GNT's model patch; ltmcols GNS's for the seed
    (on_cols); restored after, also on a failure."""
    saved = dict(CUR)
    CUR.update(seed=int(seed), out=Path(GNSC.OUT if out is None else out),
               gnr=Path(GNRC.OUT if gnr_out is None else gnr_out))
    try:
        with on_cols(seed, out), M1.WQ.patch(GNX, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, OUT=OUT,
                                              fit_name=fit_name, pair_name=pair_name, code_extra=code_extra,
                                              term_record=term_record, next_block=next_block, IN=IN, WIDTH=WIDTH,
                                              on_model=GNT.on_model):
            yield
    finally:
        CUR.clear()
        CUR.update(saved)


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table with GNS's entry (zgns, its root, its two seed-0 fits, LTG's rule, this declaration), in this
    process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, KEY: ("zgns", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                     "ltg", DECLARED)}):
        yield


def stop_record(base=ROOT):
    """GNS's current s0stop.json (s0stop.stop_record under GNS's entry), else None."""
    with on_s0stop():
        return S0.stop_record(KEY, base)


# ── the first pass and the columns a fit stands on (section 4) ───────────────


def check_cols(split, seed=0, base=ROOT, cache_root=None):
    """Every carve a GNS fit of the split loads (its training carves, the six s1eval carves): GNS's column record and
    files of the seed against the carve's cache parts (zltm.load_next, sha256 checked), each record's scorer
    gnscols.fit_sha of the split's and seed's current first pass and its first_sha256 that first pass's record sha256;
    and the first pass one GNR's fits may stand on (zgnr.first_problems); outputs/gns/s<seed>/check_<split>.json."""
    import lean_cache as LC
    if split not in M1.SPLITS:
        raise SystemExit(f"zgns: {split} is not one of WQ's splits")
    croot = Path(cache_root or Path(base) / M1.ROOTS["cache"])
    gout = Path(base) / OUT
    fp = first_path(seed, split, Path(base) / GNR.OUT)
    fsha = GNR.sha_or_none(fp)
    want = GNSC.fit_sha(fsha) if fsha else None
    bad = [f"{split}: {p}" for p in GNR.first_problems(seed, split, DEP.iso_call(base), base)]
    got = {}
    with on_cols(seed, gout):
        for key in GNX.train_carves(split) + GNX.eval_carves():
            ds, cv = key.split("/")
            try:
                parts, _r = LC.part_dirs(ds, cv, croot)
                X, r, sha = LTM.load_next(ds, cv, split, {p.name: LC.sha_file(p / "record.json") for p in parts})
            except SystemExit as e:
                bad.append(f"{split} {key}: {e}")
                continue
            g = {"rows": int(X.shape[0]), "record_sha256": sha, "fit_sha256": r.get("fit_sha256"),
                 "first_sha256": r.get("first_sha256"), "seed": r.get("seed"), "training_carve": r.get("training_carve"),
                 "nan": int(np.isnan(X).sum()), "census": r.get("census")}
            if want is None or (g["fit_sha256"], g["first_sha256"], g["seed"]) != (want, fsha, int(seed)):
                bad.append(f"{split} {key}: its columns stand on the first pass {g['first_sha256']} (scorer "
                           f"{g['fit_sha256']}, seed {g['seed']}), not the current {fsha} (scorer {want}, seed {seed})")
            got[key] = g
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "cache_root": M1.WQ.rel(croot),
           "first": M1.WQ.rel(fp), "first_sha256": fsha, "fit_sha256": want, "carves": got, "problems": bad,
           "pass": not bad, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = check_path(seed, split, gout)
    p.parent.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(p, rec)
    log(f"check-cols {split} s{seed}: {'PASS' if not bad else 'FAIL'}; {len(got)} carves on the first pass "
        f"{(fsha or '-')[:12]}; {bad}")
    return 0 if not bad else 1


def match_problems(base=ROOT):
    """Why GNS's columns' match (outputs/gns/match.json: rrf's top five as the places give gnwcols' own columns) does not
    hold: missing, not passed, or made by other code of the rule."""
    p = Path(base) / OUT / "match.json"
    if not p.is_file():
        return [f"{M1.WQ.rel(p, base)}: the columns' match against gnwcols is missing"]
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r.get("pass"):
        return [f"{M1.WQ.rel(p, base)}: the columns' match against gnwcols did not pass"]
    if r.get("rule_sha256") != GNSC.rule_sha():
        return [f"{M1.WQ.rel(p, base)}: the match was made by other code of the rule ({str(r.get('rule_sha256'))[:12]})"]
    return []


def check_problems(seed, split, base=ROOT):
    """Why the split's and seed's column check does not hold: missing, not passed, or made on another first pass than the
    current one (or by another scorer than gnscols.fit_sha of it)."""
    p = check_path(seed, split, Path(base) / OUT)
    fsha = GNR.sha_or_none(first_path(seed, split, Path(base) / GNR.OUT))
    if not p.is_file():
        return [f"{M1.WQ.rel(p, base)}: the column check of {split}, seed {seed} is missing"]
    r = json.loads(p.read_text(encoding="utf-8"))
    if not r.get("pass"):
        return [f"{M1.WQ.rel(p, base)}: the column check of {split}, seed {seed} did not pass "
                f"({len(r.get('problems') or [])} problems)"]
    got = tuple(r.get(k) for k in ("stage", "split", "seed", "first_sha256", "fit_sha256"))
    if fsha is None or got != (STAGE, split, int(seed), fsha, GNSC.fit_sha(fsha)):
        return [f"{M1.WQ.rel(p, base)}: the column check names {got}, not the current first pass {fsha} of {split}, "
                f"seed {seed}"]
    return []


def gate_problems(seed, split, base=ROOT):
    """Why a GNS fit of the split and seed may not start, or a read one is not the declared one (section 4): the first
    pass (zgnr.first_problems), the columns' match, the split's column check on the current first pass."""
    return (list(GNR.first_problems(seed, split, DEP.iso_call(base), base)) + match_problems(base)
            + check_problems(seed, split, base))


def next_problems(name, seed, split, what, nk, base=ROOT):
    """Why a GNS fit's train or read NEXT block (nk) does not stand on the current first pass of its split and seed: the
    block must name GNS's declaration, the seed, the seed's columns root and GNS's columns' declaration, the first pass's
    record sha256, and gnscols.fit_sha of it as its scorer."""
    fsha = GNR.sha_or_none(first_path(seed, split, Path(base) / GNR.OUT))
    want = (DECLARED, int(seed), M1.WQ.rel(cols_root(seed, Path(base) / OUT)), DECLARED)   # as next_block names it
    got = tuple(nk.get(k) for k in ("declared_in", "seed", "columns_root", "columns_declared_in"))
    out = []
    if got != want:
        out.append(f"{name}: its {what}'s NEXT block names {got}, not GNS's columns {want}")
    if fsha is None or nk.get("first_sha256") != fsha or nk.get("fit_sha256") != GNSC.fit_sha(fsha):
        out.append(f"{name}: its {what} read columns of the first pass {nk.get('first_sha256')} (scorer "
                   f"{nk.get('fit_sha256')}), not the current {fsha}")
    return out


# ── the commands (zgnx's under GNS's names) ──────────────────────────────────


def fit(name, base=ROOT):
    if name not in FITS:
        raise SystemExit(f"zgns: {name} is not one of GNS's fits ({', '.join(FITS)})")
    _a, sd, sp = FITS[name]
    bad = gate_problems(sd, sp, base)
    if bad:                                     # before any import or train: section 4 would call the fit INCOMPLETE
        raise SystemExit(f"zgns: {name}: {'; '.join(bad)}")
    with on_gns(sd):
        return GNX.fit(name)


def compare_all(base=ROOT):
    with on_gns():
        return GNX.compare_all(base)


def fit_block(name, what, base=ROOT):
    """A GNS fit's stamp block (zgnx's fit_block under GNS's names)."""
    with on_gns():
        return GNX.fit_block(name, what, base)


def fit_problems(call, base=ROOT):
    """zgnw.fit_problems with GNS's fits and term (GNT's TERM_PARAMS parameters on the fifteen columns), inside on_gns:
    its term, base and ISO verdict; its NEXT block (the split, the columns, one scorer for train and read, every training
    carve in its train and every s1eval carve in its read); its start; its device; its trainer's code against the base's
    fit of its seed and split."""
    with M1.WQ.patch(GNW, FITS=FITS, TERM_PARAMS=TERM_PARAMS):
        return GNW.fit_problems(call, base)


def gns_problems(base=ROOT):
    """Why a GNS fit that is read is not the declared one beyond zgnw's conditions (section 4): its train's and read's
    NEXT blocks (next_problems), its train's term (GNS's text, GNT's parameters on the fifteen columns) and code (this
    file, gnscols.py, zgnr.py, gnrcols.py, zgnt.py and GNW's), and its split's and seed's gate (gate_problems)."""
    out, own, read = [], own_shas(), []
    with on_gns():
        for name, (_a, sd, sp) in FITS.items():
            if not GNX.is_read(name, base):
                continue
            read.append((name, sd, sp))
            try:
                ks, kr = GNX.fit_block(name, "screen", base), GNX.fit_block(name, "read", base)
            except SystemExit as e:
                out.append(str(e))
                continue
            for what, k in (("train", ks), ("read", kr)):
                out += next_problems(name, sd, sp, what, k.get("next") or {}, base)
            if (ks.get("inputs"), ks.get("columns"), ks.get("term_parameters")) != (IN, list(GNSC.COLS), TERM_PARAMS) \
                    or not str(ks.get("term") or "").startswith("GNS's learned attention"):
                out.append(f"{name}: its train's term takes {ks.get('inputs')} inputs of {ks.get('columns')} with "
                           f"{ks.get('term_parameters')} parameters ({str(ks.get('term'))[:40]!r}), not GNS's")
            code = ks.get("code") or {}
            diff = sorted(k for k, v in own.items() if code.get(k) != v)
            if diff:
                out.append(f"{name}: its train's code is not this GNS's in {diff}")
    for name, sd, sp in read:
        out += [f"{name}: {p}" for p in gate_problems(sd, sp, base)]
    return out


def next_form(screen, full):
    s = (screen or {}).get("verdict")
    v = (full or {}).get("verdict")
    if s == INCOMPLETE or v == INCOMPLETE:
        return "what is missing is fixed in an amendment, before any number is read"
    if s == ADOPT_FOR_FULL_RUN and v in (None, ADOPT):
        return ("the stack takes the attention by an amendment to docs/STK_GNN_STACK.md: of GNT, GNW and GNS, the one that "
                "ADOPTs with the largest mean seed-0 delta over the ten deciding reads (GNS's columns per seed: seeds 1 "
                "and 2 need their own first passes and columns, from the base fits of those seeds, before any stack fit "
                "reads them); GNS's own full run is held for the stack; next form: a second round, the places re-anchored "
                "on GNS's own first-pass top five")
    return ("attention to the in-sample first-pass places closes for the GNN; next form: GNR's first pass cross-fitted "
            "(two half fits of the base per split), so that the training carves' places are out of sample as the s1eval "
            "carves' are, for GNR's columns and GNS's")


def cols_beside(base=ROOT, seed=0):
    """The column checks of the seed (decides nothing): per split the check, and per carve the share of questions whose
    five places are rrf's top five as a set, and the golds among the first pass's five and among rrf's."""
    out = {}
    for sp in M1.SPLITS:
        p = check_path(seed, sp, Path(base) / OUT)
        if not p.is_file():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        out[sp] = {"pass": r.get("pass"),
                   "carves": {k: {x: (g.get("census") or {}).get(x) for x in
                                  ("questions", "same_five_share", "first_golds", "rrf_golds", "with_gold")}
                              for k, g in (r.get("carves") or {}).items()}}
    return out


def calls(base=ROOT):
    """GNS's screen (and its full run, when every fit is read) by LTG's in-domain rule, DEP's sixteen-read rule beside,
    through zgnx's reads, slices and checks under GNS's names; outputs/gns/calls.json and .md."""
    with on_gns():
        call = DEP.iso_call(base)
        new_r5, b0 = GNX.r5_gnx(call, base)
        base3 = DEP.base_seeds(call, base)
        bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in GNS's compare, {base3.get(rd, {}).get(sd)} in its own "
               f"stage's" for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
               if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > GNX.R5_TOL]
        new_s, base_s, bad_s = GNX.slices(call, new_r5, base3, base)
        bad += bad_s
        new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
        screen = LTG.decide_screen(new_all, base_all)
        full_ready = all(GNX.is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
        full = LTG.decide_full(new_all, base_all) if full_ready else None
        dep_rule = {"screen": DEP.decide_screen(new_all, base_all)["verdict"],
                    "full_run": DEP.decide_full(new_r5, base3, new_s, base_s)["verdict"] if full_ready else None}
        bad += fit_problems(call, base)
        params = {f: {k: GNX.fit_block(f, "screen", base).get(k) for k in GNX.term_record_keys()}
                  for f in FITS if GNX.is_read(f, base)}
    bad += gns_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    beside = {st: GNP.screen_beside(st, base) for st in ("gnt", "gnw", "gnr", "gnx")}
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{GNX.ARM} with GNS's learned attention (the base GNN's own "
           "places)", "iso_verdict": call, "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)",
           "screen": screen, "full_run": full, "dep_rule_beside": dep_rule, "next_form": next_form(screen, full),
           "beside": {**beside, "first_pass": GNR.first_beside(base), "columns": cols_beside(base)}, "problems": bad,
           "parameters": params, **own_shas(), "zgnx_sha256": M1.WQ.sha_src(GNX.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    td = {(r["split"], r["read"]): r.get("delta") for r in (beside["gnt"] or {}).get("reads") or []}

    def fd(x):
        return "-" if x is None else f"{x:+.4f}"
    md = [f"# GNS: {GNX.ARM} with learned attention to the base GNN's own top five (a first pass) against {rec['base']}, "
          "on PKO's pools", "", f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"next form: {rec['next_form']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | GNS s0 | delta | GNT delta (rrf's places) | floor | call |",
               "|---|---|---|---:|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {fd(td.get((r['split'], r['read'])))} | {r['floor']:.4f} | {r['call']} |"
               for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in frows:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['read']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['floor']:.4f} | "
                          f"{r['call']} |")
    for st in ("gnt", "gnw", "gnr", "gnx"):
        b = beside[st]
        if b:
            md += ["", f"{st.upper()} beside: screen {b['screen']}, full run {b['full_run']}"]
    fps = ", ".join(f"{sp} check {v['check']}" for sp, v in rec["beside"]["first_pass"].items())
    cbs = []
    for sp, v in rec["beside"]["columns"].items():
        sh = [c["same_five_share"] for c in v["carves"].values() if c.get("same_five_share") is not None]
        cbs.append(f"{sp} check {v['pass']}, places = rrf's top five on {min(sh):.3f}-{max(sh):.3f} of the questions"
                   if sh else f"{sp} check {v['pass']}")
    md += ["", f"First passes: {fps or 'none'}; columns: {'; '.join(cbs) or 'none'}"]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; DEP's rule beside {dep_rule}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def snapshot(G=None):
    s = {**{f"gnx.{k}": getattr(GNX, k) for k in PATCHED}, **{f"ltmcols.{k}": getattr(LTMC, k) for k in COLS_PATCHED},
         **{f"m1.{k}": getattr(M1, k) for k in ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas",
                                                 "load_encoder", "train", "read")}, "n_fits": len(M1.ALL_FITS),
         "cur": dict(CUR), "gnw.FITS": GNW.FITS, "gnw.TERM_PARAMS": GNW.TERM_PARAMS}
    if G is not None:
        s.update({"g.encoder_used": G.encoder_used, "g.init": G.ZG6R.__init__, "g.forward": G.ZG6R.forward,
                  "g.carve_init": G.G6rLinkCarve.__init__, "g.carve_batch": G.G6rLinkCarve.batch})
    return s


def selftest():
    must_stop = GNP.must_stop
    sp, hp = "WQ-musique", "WQ-hotpotqa"
    # 1. names: GNS's own, none another stage's
    assert fit_name(0, sp) == "scr-gns-sa-s0" and fit_name(2, hp) == "scr-gns-sa-s2-hp"
    assert pair_name(1, hp) == "scr-gns-sa-vs-base-s1-hp" and S0.S0_SPLIT == sp and S0.HP_SPLIT == hp
    import zgna as GNA
    import zltx as LTX
    import zlta as ZLA
    assert not set(FITS) & (set(LTM.FITS) | set(GNA.FITS) | set(LTX.FITS) | set(ZLA.FITS) | set(GNX_FITS))
    assert OWN_CODE == ("zgns", "gnscols", "zgnr", "gnrcols", "zgnt", "zgnw", "gnwcols", "gnpcols", "zpfp")
    assert set(own_shas()) == {f"{k}_sha256" for k in OWN_CODE}
    # 2. the patches: zgnx's names, code, width and model patch GNS's (the model patch GNT's own) and ltmcols' root, stage,
    #    declaration, columns and width GNS's of the seed inside; restored after, also on a failure; zgnx's own names,
    #    pairs and code under them; no NEXT block outside; other stages' fits refused; GNS's s0stop entry
    s0 = snapshot()
    with on_gns():
        assert GNX.FITS is FITS and (GNX.DECLARED, GNX.STAGE, GNX.KEY, GNX.OUT) == (DECLARED, STAGE, KEY, OUT)
        assert (GNX.IN, GNX.WIDTH, LTMC.WIDTH, tuple(LTMC.COLS)) == (15, 15, 15, GNWC.COLS)
        assert GNX.on_model is GNT.on_model and GNX.next_block is next_block and GNX.term_record is term_record
        assert (Path(LTMC.OUT), LTMC.STAGE, LTMC.DECLARED) == (ROOT / OUT / "s0", STAGE, DECLARED)
        assert CUR == {"seed": 0, "out": ROOT / OUT, "gnr": ROOT / GNR.OUT}
        assert GNX.pairs(ISO.KEEP)[pair_name(0, sp)] == ("scr-gns-sa-s0", "scr-ml1-zg6r-s0")
        assert GNX.pairs(ISO.REPLACE)[pair_name(0, hp)] == ("scr-gns-sa-s0-hp", "scr-iso-zg6rm-s0-hp")
        cs = GNX.code_extra()
        assert all(f"{k}_sha256" in cs for k in OWN_CODE + ("zgnx", "zdep", "zltg", "zltm", "ltmcols")), cs
        with GNX.on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert not set(GNX_FITS) & set(M1.ALL_FITS) and not set(GNT.FITS) & set(M1.ALL_FITS)
            assert all(f"{k}_sha256" in M1.code_shas(GNX.ARM) for k in OWN_CODE)
        with M1.WQ.patch(GNW, FITS=FITS, TERM_PARAMS=TERM_PARAMS):
            assert GNW.FITS is FITS and GNW.TERM_PARAMS == 32775
    with on_gns(2, Path("x") / "gns", Path("x") / "gnr"):
        assert Path(LTMC.OUT) == Path("x") / "gns" / "s2" and CUR["seed"] == 2 and CUR["gnr"] == Path("x") / "gnr"
    assert snapshot() == s0 and (GNX.IN, LTMC.WIDTH, LTMC.STAGE) == (15, 7, "LTM") and GNX.on_model is not GNT.on_model
    try:
        with on_gns():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert snapshot() == s0
    must_stop(next_block, sp)                                            # outside on_gns
    for other in ("scr-gnt-at-s0", "scr-gnw-mw-s0", "scr-gnr-nx-s0", "scr-gnx-nx-s0"):
        must_stop(fit, other)
    st0 = dict(S0.STAGES)
    with on_s0stop():
        assert S0.STAGES[KEY] == ("zgns", OUT, "scr-gns-sa-s0", "scr-gns-sa-s0-hp", "ltg", DECLARED)
        assert S0.STAGES["gnx"] == st0["gnx"] and S0.deciding("ltg") == S0.deciding(S0.STAGES["gnx"][4])
    assert S0.STAGES == st0 and KEY not in S0.STAGES
    G = GNX.import_g()
    sg = snapshot(G)
    with on_gns():
        rec = GNX.term_record(G)
    assert rec["term"].startswith("GNS's learned attention") and rec["term_parameters"] == 32775 and rec["inputs"] == 15
    assert rec["parameters_h128"] == 212993 and rec["columns"] == list(GNWC.COLS), rec
    assert set(rec) == set(GNX.term_record_keys()) | {"term"}, rec
    # 3. on a toy tree of every carve WQ-musique's fits read, with a toy first pass of the base fit zdep names: a fit's
    #    gate (the first pass, the match, the split's check on the current first pass); the check on GNS's columns; a
    #    changed first pass fails both; GNS's columns load inside on_gns and stop outside it; the NEXT block names GNS's
    #    columns on that first pass
    import lean_cache as LC
    LTW = GNSC.LTW
    tmp = Path(tempfile.mkdtemp(prefix="zgns_"))
    try:
        rng = np.random.default_rng(29)
        M = LTW.unit(rng.normal(size=(300, 16))).astype(np.float32)
        pko, gout, rout = tmp / "pko", tmp / OUT, tmp / GNR.OUT
        for ds, cv in LTMC.col_carves(sp):
            LTW.toy_tree(pko, ds, cv, M, rng, nq=6, n=4 if (ds, cv) == ("squad", "s1eval") else 12)
        nodes = {ds: LTW.FakeNodes(M) for ds in LTMC.DS_K}
        bse = DEP.base_fit(0, sp, ISO.KEEP)
        GNRC.toy_first(pko, rout, 0, sp, bse, np.random.default_rng(5))
        mp = M1.screen_dir(tmp) / "fits" / bse / "models.pt"
        mp.parent.mkdir(parents=True)
        mp.write_bytes(b"toy models")
        fpath = first_path(0, sp, rout)
        fsha = GNR.sha_or_none(fpath)
        with M1.WQ.patch(DEP, iso_call=lambda base=ROOT: ISO.KEEP):
            assert GNR.first_problems(0, sp, ISO.KEEP, tmp) == []
            assert len(gate_problems(0, sp, tmp)) == 2                   # no match, no check
            must_stop(fit, fit_name(0, sp), tmp)
            for ds, cv in LTMC.col_carves(sp):
                GNSC.cols(0, sp, ds, cv, pko, gout, nodes, gnr_out=rout)
            assert check_cols(sp, 0, tmp, pko / "cache") == 0
            ck = json.loads(check_path(0, sp, gout).read_text(encoding="utf-8"))
            assert ck["pass"] and len(ck["carves"]) == 10 and ck["fit_sha256"] == GNSC.fit_sha(fsha), ck["problems"]
            assert ck["carves"]["squad/s1eval"]["census"]["same_five_share"] == 1.0 and check_problems(0, sp, tmp) == []
            assert check_problems(0, hp, tmp) and check_problems(1, sp, tmp)
            M1.WQ.write_json(gout / "match.json", {"pass": True, "rule_sha256": "0" * 64})
            assert len(match_problems(tmp)) == 1
            M1.WQ.write_json(gout / "match.json", {"pass": False, "rule_sha256": GNSC.rule_sha()})
            assert len(match_problems(tmp)) == 1
            M1.WQ.write_json(gout / "match.json", {"pass": True, "rule_sha256": GNSC.rule_sha()})
            assert match_problems(tmp) == [] and gate_problems(0, sp, tmp) == []
            keep = fpath.read_bytes()                                    # another first pass: the gate and the check fail
            fr = json.loads(keep)
            fr["note"] = "another first pass"
            fpath.write_text(json.dumps(fr), encoding="utf-8")
            assert GNR.first_problems(0, sp, ISO.KEEP, tmp) == [] and len(check_problems(0, sp, tmp)) == 1
            ck0 = check_path(0, sp, gout).read_bytes()
            assert check_cols(sp, 0, tmp, pko / "cache") == 1
            ck = json.loads(check_path(0, sp, gout).read_text(encoding="utf-8"))
            assert not ck["pass"] and len(ck["problems"]) == 10, ck["problems"]
            fpath.write_bytes(keep)
            check_path(0, sp, gout).write_bytes(ck0)
            assert gate_problems(0, sp, tmp) == []
            fr["check"] = False                                          # a failed first pass: the gate fails
            fpath.write_text(json.dumps(fr), encoding="utf-8")
            assert len(GNR.first_problems(0, sp, ISO.KEEP, tmp)) == 1 and len(gate_problems(0, sp, tmp)) == 2
            fpath.write_bytes(keep)
            assert gate_problems(0, sp, tmp) == []
        parts, _r = LC.part_dirs("musique", "s1eval", pko / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        must_stop(LTM.load_next, "musique", "s1eval", sp, ps, cols_root(0, gout))    # GNS's columns outside on_gns
        with on_gns(0, gout, rout):
            X, r2, _sha = LTM.load_next("musique", "s1eval", sp, ps)
            assert (r2["stage"], r2["declared_in"], r2["seed"], r2["first_sha256"]) == (STAGE, DECLARED, 0, fsha), r2
            must_stop(LTM.load_next, "musique", "s1eval", hp, ps)        # no WQ-hotpotqa columns
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, cols_root(1, gout))   # nor seed 1's
            with GNX.on_split(sp, r2["fit_sha256"]):
                GNX.LOADED.append({"dataset": "musique", "carve": "s1eval", "record_sha256": "r"})
                nb = GNX.next_block(sp)
            GNX.LOADED.clear()
        want = {"declared_in": DECLARED, "columns_root": M1.WQ.rel(cols_root(0, gout)), "columns_declared_in": DECLARED,
                "seed": 0, "first": M1.WQ.rel(fpath), "first_sha256": fsha, "fit_sha256": GNSC.fit_sha(fsha),
                "columns": list(GNWC.COLS), "split": sp}
        assert {k: nb.get(k) for k in want} == want, nb
        assert next_problems("x", 0, sp, "train", nb, tmp) == []
        GNWC.cols("musique", "s1eval", pko, tmp / "gnw", {"musique": LTW.FakeNodes(M)})
        with on_gns(0, tmp / "gnw", rout):                               # GNW's columns in GNS's place stop
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnw")
        assert snapshot() == s0
        # 4. the stamp under GNS's names: the train's NEXT block is GNS's on its columns and first pass, with GNS's term;
        #    one without columns stops
        seen, orig_stamp = [], M1.stamp
        try:
            M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
            with on_gns(), GNX.on_gnx(G, ISO.KEEP):
                assert (M1.STAGE, M1.KEY) == (STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
                assert G.ZG6R.__init__ is not sg["g.init"]
                with GNX.on_split(hp):
                    must_stop(M1.stamp, fit_name(0, hp), {}, "screen", {"x": 1})
                with GNX.on_split(hp, "c" * 64):
                    GNX.LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                                      for k in GNX.train_carves(hp))
                    M1.stamp(fit_name(0, hp), {}, "screen", {"x": 1})
                GNX.LOADED.clear()
        finally:
            M1.stamp = orig_stamp
            M1._ACTIVE[0] = None
            GNX.LOADED.clear()
        assert [s[:2] for s in seen] == [(fit_name(0, hp), "screen")], seen
        e = seen[0][2]
        assert (e["next"]["declared_in"], e["next"]["fit_sha256"], e["base"]) == (DECLARED, "c" * 64, "scr-ml1-zg6r-s0-hp")
        assert (e["next"]["columns_root"], e["next"]["columns_declared_in"], e["next"]["seed"]) == (f"{OUT}/s0", DECLARED,
                                                                                                    0), e["next"]
        assert e["next"]["first_sha256"] == GNR.sha_or_none(first_path(0, hp)), e["next"]
        assert e["next"]["columns"] == list(GNWC.COLS) and e["inputs"] == 15 and e["term_parameters"] == 32775, e
        assert e["term"].startswith("GNS's learned attention") and e["x"] == 1, e
        assert snapshot(G) == sg and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
        # 5. the checks a read fit faces, on made stamp blocks: a declared fit gives none; GNW's term, GNT's term text,
        #    other code, GNW's columns root, another first pass, another scorer, a missing match or check each give
        #    problems; the calls' records on made reads
        name = fit_name(0, sp)
        croot = M1.WQ.rel(cols_root(0, gout))

        def block(what, **over):
            need = GNX.train_carves(sp) if what == "screen" else GNX.eval_carves()
            k = {"term_parameters": TERM_PARAMS, "inputs": IN, "columns": list(GNWC.COLS), "layers": DEP.BASE_LAYERS,
                 "term": "GNS's learned attention: made", "base": bse, "iso_verdict": ISO.KEEP, "device": {"gpu": "t"},
                 "next": {"declared_in": DECLARED, "columns_root": croot, "columns_declared_in": DECLARED, "seed": 0,
                          "first": "made", "first_sha256": fsha, "split": sp, "columns": list(GNWC.COLS),
                          "fit_sha256": GNSC.fit_sha(fsha), "carves": {c: "r" for c in need}},
                 "encoder": {"fit": "z", "seed": 0, "sha256": "s", "copies": 1}, "code": {**own_shas(), "a_sha256": "1"}}
            nx = over.pop("next", None)
            k.update(over)
            if nx:
                k["next"] = {**k["next"], **nx}
            return k
        blocks_ = {}

        def fb(nm, what, base=ROOT):
            if nm == bse:
                return {"encoder": {"fit": "z", "seed": 0, "sha256": "s"}, "device": {"gpu": "t"},
                        "code": {"a_sha256": "1"}}
            return blocks_[what]
        with M1.WQ.patch(DEP, iso_call=lambda base=ROOT: ISO.KEEP), \
                M1.WQ.patch(GNX, is_read=lambda nm, base=ROOT: nm == name, fit_block=fb):
            for over, n_fit, n_gns in (({}, 0, 0), ({"term_parameters": GNW.TERM_PARAMS}, 2, 1),
                                       ({"term": "GNT's learned attention: made"}, 0, 1),
                                       ({"columns": list(GNWC.GNPC.COLS)}, 2, 1),
                                       ({"code": {**own_shas(), "zgns_sha256": "0" * 64, "a_sha256": "1"}}, 0, 1),
                                       ({"code": {**own_shas(), "gnrcols_sha256": "0" * 64, "a_sha256": "1"}}, 0, 1),
                                       ({"next": {"columns_root": GNW.OUT}}, 0, 2),
                                       ({"next": {"declared_in": GNT.DECLARED}}, 0, 2),
                                       ({"next": {"first_sha256": "e" * 64}}, 0, 2),
                                       ({"next": {"fit_sha256": "g" * 64}}, 0, 2)):
                blocks_.update(screen=block("screen", **dict(over)), read=block("read", **dict(over)))
                with on_gns():
                    fp_ = fit_problems(ISO.KEEP, tmp)
                gp = gns_problems(tmp)
                assert (len(fp_), len(gp)) == (n_fit, n_gns), (over, fp_, gp)
            blocks_.update(screen=block("screen"), read=block("read", next={"fit_sha256": "g" * 64}))
            with on_gns():
                assert len(fit_problems(ISO.KEEP, tmp)) == 1             # train and read on different scorers
            assert len(gns_problems(tmp)) == 1
            blocks_.update(screen=block("screen"), read=block("read"))
            for f in (gout / "match.json", check_path(0, sp, gout)):
                keep_f = f.read_bytes()
                f.unlink()
                assert len(gns_problems(tmp)) == 1
                f.write_bytes(keep_f)
            assert gns_problems(tmp) == []
        # the calls on made reads: GNS's s0 read and declared, every read the rule takes above the base's three seeds
        keys = list(DEP.read_keys())
        base3 = {k: {0: 0.50, 1: 0.51, 2: 0.49} for k in keys}
        new = {k: {0: 0.53} for k in keys}

        def made_calls():
            with M1.WQ.patch(DEP, iso_call=lambda base=ROOT: ISO.KEEP, base_seeds=lambda call, base=ROOT: base3), \
                    M1.WQ.patch(GNX, is_read=lambda nm, base=ROOT: nm == name, fit_block=fb,
                                r5_gnx=lambda call, base=ROOT: (new, {k: {0: v[0]} for k, v in base3.items()}),
                                slices=lambda call, new_r5, b3, base=ROOT: ({}, {}, [])):
                calls(tmp)
            return (json.loads((gout / "calls.json").read_text(encoding="utf-8")),
                    (gout / "calls.md").read_text(encoding="utf-8"))
        ck0 = check_path(0, sp, gout).read_bytes()
        check_path(0, sp, gout).unlink()
        cj, md = made_calls()                                            # no column check: INCOMPLETE
        assert cj["stage"] == STAGE and cj["screen"]["verdict"] == INCOMPLETE, cj["screen"]
        assert len(cj["problems"]) == 1 and "column check" in cj["problems"][0] and "## Screen: INCOMPLETE" in md
        check_path(0, sp, gout).write_bytes(ck0)
        cj, md = made_calls()
        assert cj["screen"]["verdict"] == ADOPT_FOR_FULL_RUN and not cj["problems"] and cj["full_run"] is None, cj
        assert cj["parameters"] == {name: {k: block("screen").get(k) for k in GNX.term_record_keys()}}, cj["parameters"]
        assert cj["next_form"] == next_form(cj["screen"], None) and "| split | read |" in md and cj["beside"]["gnt"] is None
        assert cj["beside"]["first_pass"][sp]["check"] is True and cj["beside"]["columns"][sp]["pass"] is True
        assert "places = rrf's top five on" in md, md
    finally:
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    assert snapshot(G) == sg
    # 6. what follows, by the verdicts
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("the stack takes the attention")
    assert next_form({"verdict": NO_GAIN}, None).startswith("attention to the in-sample first-pass places closes")
    assert next_form({"verdict": NOT_ADOPTED}, None).startswith("attention to the in-sample first-pass places closes")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("attention to the in-sample")
    assert next_form({"verdict": INCOMPLETE}, None).startswith("what is missing")
    log("selftest: names; the patches and their restoring (also on a failure), GNT's model patch, GNS's columns root per "
        "seed, no NEXT block outside, other stages' fits refused, the s0stop entry; GNS's term record; on toy carves of "
        "WQ-musique with a toy first pass: the gate (first pass, match, check) holds a fit, the check passes on GNS's "
        "columns and fails on another first pass, a failed first pass fails the gate, GNS's columns load inside and stop "
        "outside, GNW's stop inside, the NEXT block names GNS's columns on the first pass; the stamp's NEXT block and "
        "term GNS's and its refusal without columns; the read checks on made blocks; the calls on made reads; what "
        "follows")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("check-cols", "fit", "compare-all", "calls", "s0stop-check",
                                               "s0stop-run", "gate"))
    ap.add_argument("--name")
    ap.add_argument("--split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--want")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "check-cols":
        if not a.split:
            ap.error("check-cols needs --split")
        return check_cols(a.split, a.seed)
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    if a.cmd == "s0stop-check":
        with on_s0stop():
            return S0.check(KEY, ROOT)
    if a.cmd == "s0stop-run":
        with on_s0stop():
            return S0.run(KEY, cmd)
    if a.cmd == "gate":
        if not a.want:
            ap.error("gate needs --want")
        with on_s0stop():
            return S0.gate(KEY, [w for w in a.want.split(",") if w])
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
