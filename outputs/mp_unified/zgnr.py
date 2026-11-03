"""GNR's stage (docs/GNR_REANCHORED_NEXT_GNN.md): GNX's arm, the GNN of record with seven NEXT columns as one more
input of its encoder (docs/GNX_NEXT_COLUMNS_IN_GNN.md), reading GNR's columns (gnrcols.py: LTM's T lists and next-hop
columns with the anchors the base GNN's own top five rows of a first pass, scored by LTM's fold fits, read only) in place
of LTM's, whose anchors are rrf's top five. On PKO's roots, on the host's GPU: the screen at seed 0 on both WQ splits
against the base's seed-0 fits under LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), DEP's sixteen-read rule
beside.

zgnx.py and gnrcols.py, and every module they import, are imported and not edited. For the length of a GNR call zgnx's
names (its fits, declaration, stage, stamp key, output root, fit and pair names, code, term record and NEXT block) are
GNR's, and ltmcols' root, stage and declaration are GNR's for the call's seed (outputs/gnr/s<seed>): zgnx's carve then
loads GNR's columns through zltm.load_next and refuses LTM's. Everything else in a fit is GNX's: the term (gnx_w, hidden x
15, zero-started, added to l1's output), the base, the start, the trainer and the read. Every patched name is put back
afterwards, also on a failure. A fit refuses to start when its split's and seed's first pass is not one its columns may
stand on (section 4).

    python outputs/mp_unified/zgnr.py check-cols                   (GNR's seed-0 columns of every carve both splits load)
    python outputs/mp_unified/zgnr.py fit --name scr-gnr-nx-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zgnr.py compare-all                  (each GNR fit read against the base's fit of its seed)
    python outputs/mp_unified/zgnr.py calls                        (outputs/gnr/calls.json and .md)
    python outputs/mp_unified/zgnr.py s0stop-check                 (s0stop's check under GNR's entry)
    python outputs/mp_unified/zgnr.py s0stop-run -- COMMAND ...    (s0stop's run under GNR's entry)
    python outputs/mp_unified/zgnr.py gate --want ADOPT_FOR_FULL_RUN (s0stop's gate under GNR's entry)
    python outputs/mp_unified/zgnr.py --selftest
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
import zgnx as GNX  # noqa: E402  (GNX's arm, carve, term, stamp, compares and the pieces of its calls; not edited)
import gnrcols as GNRC  # noqa: E402  (the first passes' records, GNR's columns' root per seed; not edited)
import s0stop as S0  # noqa: E402  (the seed-0 stop, with GNR's entry patched in for a call; not edited)

M1, RULE, ISO, DEP, LTG, LTM, LTMC = GNX.M1, GNX.RULE, GNX.ISO, GNX.DEP, GNX.LTG, GNX.LTM, GNX.LTMC
DECLARED = "docs/GNR_REANCHORED_NEXT_GNN.md"
STAGE = "GNR"
KEY = "gnr"                                     # the stamp's key in a GNR fit's screen.json and read.json and GNR's compares
OUT = "outputs/gnr"
TAG = GNX.TAG
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    GNX.ADOPT_FOR_FULL_RUN, GNX.NO_GAIN, GNX.NOT_ADOPTED, GNX.ADOPT, GNX.INCOMPLETE)
PATCHED = ("FITS", "DECLARED", "STAGE", "KEY", "OUT", "fit_name", "pair_name", "code_extra", "term_record", "next_block")
ORIG = {k: getattr(GNX, k) for k in ("code_extra", "term_record", "next_block")}
GNX_FITS = dict(GNX.FITS)                       # zgnx's own fit names, taken before any patch
OWN_CODE = ("zgnr", "gnrcols")
assert (GNRC.STAGE, GNRC.DECLARED, Path(GNRC.OUT)) == (STAGE, DECLARED, ROOT / OUT) and GNRC.LTMC is LTMC


def fit_name(seed, split):
    return f"scr-gnr-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-gnr-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (GNX.ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS)
                                            | set(GNX.FITS))
CUR = {"seed": None, "out": None}               # inside on_gnr: the seed whose columns, and the root whose first passes


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zgnr: {msg}", flush=True)


def own_shas():
    return {"zgnr_sha256": M1.WQ.sha_src(__file__), "gnrcols_sha256": M1.WQ.sha_src(GNRC.__file__)}


def code_extra():
    """GNX's code beside zml1s's (zgnx, zdep, zltg, zltm, ltmcols), and GNR's own: this file and gnrcols.py."""
    return {**ORIG["code_extra"](), **own_shas()}


def term_record(G):
    return {**ORIG["term_record"](G),
            "term": "GNR's NEXT columns (LTM's next-hop columns against the base GNN's own top five rows of a first pass) "
                    "into zg6r's encoder ([raw, z, 1] through gnx_w, zero-started, added to l1's output)"}


def first_path(seed, split, out=None):
    """The first pass's record of a split and seed (gnrcols' first) under GNR's root (out: a test's)."""
    return GNRC.first_record(seed, split, Path(GNRC.OUT if out is None else out))


def sha_or_none(p):
    return GNRC.sha_file(p) if Path(p).is_file() else None


def next_block(split):
    """GNX's NEXT block with GNR's declaration, the columns' root (inside on_gnr: GNR's of the call's seed), the seed and
    the first pass the columns stand on (its record's path and sha256; None when there is none)."""
    if CUR["seed"] is None:
        raise SystemExit("zgnr: a NEXT block outside on_gnr")
    fp = first_path(CUR["seed"], split, CUR["out"])
    return {**ORIG["next_block"](split), "declared_in": DECLARED, "columns_root": M1.WQ.rel(Path(LTMC.OUT)),
            "seed": CUR["seed"], "first": M1.WQ.rel(fp), "first_sha256": sha_or_none(fp)}


@contextlib.contextmanager
def on_gnr(seed=0, out=None):
    """zgnx with GNR's names, code, term record and NEXT block, and ltmcols' root, stage and declaration GNR's for the
    seed (out: GNR's root, or a test's; the columns under its s<seed>); restored after, also on a failure."""
    root = Path(GNRC.OUT if out is None else out)
    saved = dict(CUR)
    CUR.update(seed=int(seed), out=root)
    try:
        with M1.WQ.patch(LTMC, OUT=GNRC.seed_root(int(seed), root), STAGE=STAGE, DECLARED=DECLARED), \
                M1.WQ.patch(GNX, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, OUT=OUT, fit_name=fit_name,
                            pair_name=pair_name, code_extra=code_extra, term_record=term_record, next_block=next_block):
            yield
    finally:
        CUR.clear()
        CUR.update(saved)


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table with GNR's entry (zgnr, its root, its two seed-0 fits, LTG's rule, this declaration), in this
    process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, KEY: ("zgnr", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                     "ltg", DECLARED)}):
        yield


def stop_record(base=ROOT):
    """GNR's current s0stop.json (s0stop.stop_record under GNR's entry), else None."""
    with on_s0stop():
        return S0.stop_record(KEY, base)


# ── the first pass and the columns a fit stands on (section 4) ───────────────


def first_problems(seed, split, call, base=ROOT):
    """Why the first pass of a split and seed is not one a GNR fit may stand on: it is missing, is not GNR's of that split
    and seed, is not of the base fit zdep.base_fit names, its models.pt sha256 is not the base's, or its s1eval check
    failed."""
    fp = first_path(seed, split, Path(base) / OUT)
    if not fp.is_file():
        return [f"no first pass of {split}, seed {seed} ({M1.WQ.rel(fp, base)})"]
    rec = json.loads(fp.read_text(encoding="utf-8"))
    out, bse = [], DEP.base_fit(seed, split, call)
    got = tuple(rec.get(k) for k in ("stage", "declared_in", "split", "seed"))
    if got != (STAGE, DECLARED, split, int(seed)):
        out.append(f"{M1.WQ.rel(fp, base)} is {got}, not GNR's first pass of {split}, seed {seed}")
    if rec.get("base") != bse:
        out.append(f"the first pass of {split}, seed {seed} read {rec.get('base')}, not the base fit {bse}")
    mp = M1.screen_dir(base) / "fits" / bse / "models.pt"
    sha = sha_or_none(mp)
    if sha is None or rec.get("base_models_sha256") != sha or rec.get("models_sha256") != sha:
        out.append(f"the first pass of {split}, seed {seed} read the models {rec.get('models_sha256')} (its base "
                   f"{rec.get('base_models_sha256')}), not {bse}'s {sha}")
    if rec.get("check") is not True:
        out.append(f"the first pass of {split}, seed {seed} failed its s1eval check (within {rec.get('tol')})")
    return out


def carve_problems(seed, split, carves, base=ROOT):
    """Why GNR's columns of the carves ({dataset/carve: the column record's sha256, or None for the file's own}) do not
    stand on the split's and seed's current first pass: each carve's gnr.json must be GNR's of the split and seed and name
    that column record and that first pass's record."""
    root = Path(base) / OUT
    fsha, out = sha_or_none(first_path(seed, split, root)), []
    for key, rsha in sorted(carves.items()):
        ds, cv = key.split("/")
        cd = LTMC.cols_dir(GNRC.seed_root(seed, root), split, ds, cv)
        g = json.loads((cd / "gnr.json").read_text(encoding="utf-8")) if (cd / "gnr.json").is_file() else {}
        want = (STAGE, split, int(seed), sha_or_none(cd / "record.json") if rsha is None else rsha, fsha)
        if fsha is None or tuple(g.get(k) for k in ("stage", "split", "seed", "record_sha256", "first_sha256")) != want:
            out.append(f"{split} {key}: its columns are not GNR's of seed {seed} on the current first pass "
                       f"({M1.WQ.rel(cd / 'gnr.json', base)})")
    return out


def next_problems(name, seed, split, what, nk, base=ROOT):
    """Why a GNR fit's train or read NEXT block (nk) does not stand on the current first pass of its split and seed: the
    block must name GNR's declaration, the seed, the seed's columns root and the first pass's record sha256, and each
    carve it loaded must be GNR's columns on that first pass (carve_problems)."""
    root = Path(base) / OUT
    fsha = sha_or_none(first_path(seed, split, root))
    want = (DECLARED, int(seed), M1.WQ.rel(GNRC.seed_root(seed, root)))   # as next_block names it
    got = tuple(nk.get(k) for k in ("declared_in", "seed", "columns_root"))
    out = []
    if got != want:
        out.append(f"{name}: its {what}'s NEXT block names {got}, not GNR's columns {want}")
    if fsha is None or nk.get("first_sha256") != fsha:
        out.append(f"{name}: its {what} read columns of the first pass {nk.get('first_sha256')}, not the current {fsha}")
    return out + [f"{name}: {p}" for p in carve_problems(seed, split, nk.get("carves") or {}, base)]


# ── the commands (zgnx's under GNR's names) ──────────────────────────────────


def fit(name, base=ROOT):
    if name not in FITS:
        raise SystemExit(f"zgnr: {name} is not one of GNR's fits ({', '.join(FITS)})")
    _a, sd, sp = FITS[name]
    bad = first_problems(sd, sp, DEP.iso_call(base), base)
    if bad:                                     # before any import or train: section 4 would call the fit INCOMPLETE
        raise SystemExit(f"zgnr: {name}: {'; '.join(bad)}")
    with on_gnr(sd):
        return GNX.fit(name)


def check_cols(seed=0, base=ROOT):
    """GNX's check under GNR's names and the seed's columns (every carve of both splits loads, one scorer per split;
    outputs/gnr/cols.json), then each split's first pass (first_problems) and each carve's gnr.json against its column
    record and that first pass (carve_problems), added to cols.json."""
    with on_gnr(seed):
        rc = GNX.check_cols()
    call, bad = DEP.iso_call(base), []
    for sp in M1.SPLITS:
        bad += [f"{sp}: {p}" for p in first_problems(seed, sp, call, base)]
        bad += carve_problems(seed, sp, {k: None for k in GNX.train_carves(sp) + GNX.eval_carves()}, base)
    p = Path(base) / OUT / "cols.json"
    rec = M1.WQ.read_json(p)
    rec.update({"seed": int(seed), "gnr_problems": bad, "pass": bool(rec.get("pass")) and not bad})
    M1.WQ.write_json(p, rec)
    log(f"check-cols: first passes and gnr.json {'PASS' if not bad else 'FAIL'}; {bad}")
    return rc or (1 if bad else 0)


def compare_all(base=ROOT):
    with on_gnr():
        return GNX.compare_all(base)


def fit_block(name, what, base=ROOT):
    """A GNR fit's stamp block (zgnx's fit_block under GNR's names)."""
    with on_gnr():
        return GNX.fit_block(name, what, base)


def gnr_problems(call, base=ROOT):
    """Why a GNR fit that is read is not the declared one beyond GNX's conditions (section 4): its train's and read's NEXT
    blocks (next_problems), its train's code (this zgnr.py and gnrcols.py), and its split's and seed's first pass
    (first_problems)."""
    out, own = [], own_shas()
    with on_gnr():
        for name, (_a, sd, sp) in FITS.items():
            if not GNX.is_read(name, base):
                continue
            try:
                ks, kr = GNX.fit_block(name, "screen", base), GNX.fit_block(name, "read", base)
            except SystemExit as e:
                out.append(str(e))
                continue
            for what, k in (("train", ks), ("read", kr)):
                out += next_problems(name, sd, sp, what, k.get("next") or {}, base)
            code = ks.get("code") or {}
            diff = sorted(k for k, v in own.items() if code.get(k) != v)
            if diff:
                out.append(f"{name}: its train's code is not this GNR's in {diff}")
            out += [f"{name}: {p}" for p in first_problems(sd, sp, call, base)]
    return out


def next_form(screen, full):
    s = (screen or {}).get("verdict")
    v = (full or {}).get("verdict")
    if s == INCOMPLETE or v == INCOMPLETE:
        return "what is missing is fixed in an amendment, before any number is read"
    if s == ADOPT_FOR_FULL_RUN and v in (None, ADOPT):
        return ("GNR's input goes to the stack through an STK amendment (a form beside GNX's slot, its columns per seed); "
                "seeds 1 and 2 need their own first passes and columns, from the base fits of those seeds, before any stack "
                "fit reads them; no GNR-alone full run before the 6-7 November freeze")
    return ("the in-sample anchors close; next form, in its own file: the first pass cross-fitted (two half fits of the "
            "base per split), so that the training carves' anchors are out of sample as the s1eval carves' are")


def gnx_beside(base=ROOT):
    """GNX's filed screen (decides nothing): its verdict and its deltas, read by read (rrf's anchors)."""
    p = Path(base) / "outputs" / "gnx" / "calls.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    sc = rec.get("screen") or {}
    return {"screen": sc.get("verdict"), "full_run": (rec.get("full_run") or {}).get("verdict"),
            "reads": [{k: r.get(k) for k in ("split", "read", "delta", "call")}
                      for r in (sc.get("reads") or []) + (sc.get("beside") or [])]}


def first_beside(base=ROOT, seed=0):
    """The first passes of the seed (decides nothing): per split the check and per carve R@5, in sample or the base's filed
    R@5 and the share of questions that agree."""
    out = {}
    for sp in M1.SPLITS:
        fp = first_path(seed, sp, Path(base) / OUT)
        if not fp.is_file():
            continue
        rec = json.loads(fp.read_text(encoding="utf-8"))
        out[sp] = {"check": rec.get("check"), "base": rec.get("base"),
                   "carves": {k: {x: r.get(x) for x in ("R@5", "in_sample", "base_R@5", "agree")}
                              for k, r in (rec.get("carves") or {}).items()}}
    return out


def calls(base=ROOT):
    """GNR's screen (and its full run, when every fit is read) by LTG's in-domain rule, DEP's sixteen-read rule beside,
    through zgnx's reads, slices and checks under GNR's names; outputs/gnr/calls.json and .md."""
    with on_gnr():
        call = DEP.iso_call(base)
        new_r5, b0 = GNX.r5_gnx(call, base)
        base3 = DEP.base_seeds(call, base)
        bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in GNR's compare, {base3.get(rd, {}).get(sd)} in its own "
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
        bad += GNX.fit_problems(call, base)
        params = {f: {k: GNX.fit_block(f, "screen", base).get(k) for k in GNX.term_record_keys()}
                  for f in FITS if GNX.is_read(f, base)}
    bad += gnr_problems(call, base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    gb = gnx_beside(base)
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{GNX.ARM} with GNR's NEXT input (the base GNN's own anchors)",
           "iso_verdict": call, "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen,
           "full_run": full, "dep_rule_beside": dep_rule, "next_form": next_form(screen, full),
           "beside": {"gnx": gb, "first_pass": first_beside(base)}, "problems": bad, "parameters": params,
           **own_shas(), "zgnx_sha256": M1.WQ.sha_src(GNX.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    gd = {(r["split"], r["read"]): r.get("delta") for r in (gb or {}).get("reads") or []}

    def fd(x):
        return "-" if x is None else f"{x:+.4f}"
    md = [f"# GNR: {GNX.ARM} with GNR's NEXT input (anchors: the base GNN's own top five) against {rec['base']}, on "
          "PKO's pools", "", f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"next form: {rec['next_form']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | GNR s0 | delta | GNX delta (rrf's anchors) | floor | call |",
               "|---|---|---|---:|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {fd(gd.get((r['split'], r['read'])))} | {r['floor']:.4f} | {r['call']} |"
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
    if gb:
        md += ["", f"GNX beside (LTM's columns, rrf's anchors): screen {gb['screen']}, full run {gb['full_run']}"]
    fps = ", ".join(f"{sp} check {v['check']}" for sp, v in rec["beside"]["first_pass"].items())
    md += [f"First passes: {fps or 'none'}; per carve in {OUT}/report.md"]
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


def snapshot(G=None):
    s = {**{f"gnx.{k}": getattr(GNX, k) for k in PATCHED}, **{f"ltmcols.{k}": getattr(LTMC, k) for k in
                                                              ("OUT", "STAGE", "DECLARED")},
         **{f"m1.{k}": getattr(M1, k) for k in ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas",
                                                 "load_encoder", "train", "read")}, "n_fits": len(M1.ALL_FITS),
         "cur": dict(CUR)}
    if G is not None:
        s.update({"g.encoder_used": G.encoder_used, "g.init": G.ZG6R.__init__, "g.forward": G.ZG6R.forward,
                  "g.carve_init": G.G6rLinkCarve.__init__, "g.carve_batch": G.G6rLinkCarve.batch})
    return s


def toy_first(base, seed, split, over=None, models=b"m"):
    """A made base fit's models.pt and a first-pass record of the split and seed under base (as gnrcols' first writes
    them), with fields overridden; the record's path."""
    bse = DEP.base_fit(seed, split, ISO.KEEP)
    mp = M1.screen_dir(base) / "fits" / bse / "models.pt"
    mp.parent.mkdir(parents=True, exist_ok=True)
    mp.write_bytes(models)
    sha = GNRC.sha_file(mp)
    rec = {"stage": STAGE, "declared_in": DECLARED, "split": split, "seed": int(seed), "base": bse, "candidate": GNRC.CAND,
           "base_models_sha256": sha, "models_sha256": sha, "check": True, "tol": GNRC.FIRST_TOL, "carves": {}}
    rec.update(over or {})
    fp = first_path(seed, split, Path(base) / OUT)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(rec), encoding="utf-8")
    return fp


def selftest():
    # 1. names: GNR's own, none another stage's
    assert fit_name(0, "WQ-musique") == "scr-gnr-nx-s0" and fit_name(2, "WQ-hotpotqa") == "scr-gnr-nx-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-gnr-nx-vs-base-s1-hp"
    import zrtr as RTR
    import zira as IRA
    import zisb as ISB
    import zgna as GNA
    assert not set(FITS) & (set(RTR.FITS) | set(IRA.FITS) | set(ISB.FITS) | set(LTM.FITS) | set(GNA.FITS))
    # 2. the patches: zgnx's names GNR's and ltmcols' root (the seed's), stage and declaration GNR's inside; restored
    #    after, also on a failure; zgnx's own names, pairs and code under them
    s0 = snapshot()
    with on_gnr():
        assert GNX.FITS is FITS and (GNX.DECLARED, GNX.STAGE, GNX.KEY, GNX.OUT) == (DECLARED, STAGE, KEY, OUT)
        assert GNX.pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-gnr-nx-s0", "scr-ml1-zg6r-s0")
        assert GNX.pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-gnr-nx-s0-hp", "scr-iso-zg6rm-s0-hp")
        assert (Path(LTMC.OUT), LTMC.STAGE, LTMC.DECLARED) == (ROOT / OUT / "s0", "GNR", DECLARED)
        assert CUR == {"seed": 0, "out": ROOT / OUT}
        cs = GNX.code_extra()
        assert all(f"{k}_sha256" in cs for k in OWN_CODE + ("zgnx", "zdep", "zltg", "zltm", "ltmcols")), cs
        with GNX.on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert not set(GNX_FITS) & set(M1.ALL_FITS)                  # zgnx's table is GNR's here
            assert all(f"{k}_sha256" in M1.code_shas(GNX.ARM) for k in OWN_CODE)
    with on_gnr(2):
        assert Path(LTMC.OUT) == ROOT / OUT / "s2" and CUR["seed"] == 2
    assert snapshot() == s0
    try:
        with on_gnr():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert snapshot() == s0
    must_stop(next_block, "WQ-musique")                                  # outside on_gnr
    must_stop(fit, "scr-gnx-nx-s0")                                      # GNX's own fit is not GNR's
    st0 = dict(S0.STAGES)
    with on_s0stop():
        assert S0.STAGES[KEY] == ("zgnr", OUT, "scr-gnr-nx-s0", "scr-gnr-nx-s0-hp", "ltg", DECLARED)
        assert S0.STAGES["gnx"] == st0["gnx"] and S0.deciding("ltg") == S0.deciding(S0.STAGES["gnx"][4])
    assert S0.STAGES == st0 and KEY not in S0.STAGES
    tmp = Path(tempfile.mkdtemp(prefix="zgnr_"))
    try:                                         # GNR's stop record on made files: current only for the read it was made on
        rp = S0.read_path(fit_name(0, "WQ-musique"), tmp)
        rp.parent.mkdir(parents=True)
        rp.write_text("{}", encoding="utf-8")
        (tmp / OUT).mkdir(parents=True)
        M1.WQ.write_json(tmp / OUT / "s0stop.json", {"s0_fit": fit_name(0, "WQ-musique"), "hp_fit": fit_name(0, "WQ-hotpotqa"),
                                                     "s0_read_sha256": S0.sha_file(rp), "decision": S0.STOP})
        assert stop_record(tmp)["decision"] == S0.STOP and KEY not in S0.STAGES
        rp.write_text("{ }", encoding="utf-8")
        assert stop_record(tmp) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 3. the first pass a fit stands on: none, a passed one of the base fit, and each way one is refused
    tmp = Path(tempfile.mkdtemp(prefix="zgnr_"))
    try:
        sp = "WQ-musique"
        assert first_problems(0, sp, ISO.KEEP, tmp)[0].startswith("no first pass of WQ-musique, seed 0")
        (tmp / DEP.ISO_VERDICT).parent.mkdir(parents=True, exist_ok=True)
        M1.WQ.write_json(tmp / DEP.ISO_VERDICT, {"verdict": ISO.KEEP})
        try:                                                             # refused before any import or train
            fit(fit_name(0, sp), tmp)
            raise AssertionError("fit passed without a first pass")
        except SystemExit as e:
            assert "no first pass of WQ-musique, seed 0" in str(e), e
        toy_first(tmp, 0, sp)
        assert first_problems(0, sp, ISO.KEEP, tmp) == []
        assert first_problems(0, sp, ISO.REPLACE, tmp)                   # not of the base fit ISO's REPLACE names
        for over in ({"check": False}, {"base": "scr-ml1-zg6r-s1"}, {"seed": 1}, {"split": "WQ-hotpotqa"},
                     {"models_sha256": "0" * 64}, {"declared_in": "docs/GNX_NEXT_COLUMNS_IN_GNN.md"}):
            toy_first(tmp, 0, sp, over)
            assert len(first_problems(0, sp, ISO.KEEP, tmp)) == 1, over
        fp = toy_first(tmp, 0, sp)
        (M1.screen_dir(tmp) / "fits" / "scr-ml1-zg6r-s0" / "models.pt").write_bytes(b"n")   # the base's models changed
        assert len(first_problems(0, sp, ISO.KEEP, tmp)) == 1 and fp.is_file()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 4. the columns a GNR carve loads are GNR's of the call's seed: GNR-stamped toy columns load inside on_gnr, from
    #    ltmcols' root there, and stop outside it and under another seed; LTM-stamped columns stop inside it; the NEXT
    #    block names GNR's declaration, the seed, its root and the first pass; the columns' provenance checks
    import lean_cache as LC
    import ltwdiag as LTW
    tmp = Path(tempfile.mkdtemp(prefix="zgnr_"))
    groot = tmp / OUT
    try:
        rng = np.random.default_rng(31)
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        fp = toy_first(tmp, 0, sp)
        with on_gnr(0, groot):
            X = LTM.toy_cols(GNRC.seed_root(0, groot), sp, "musique", "s1eval", tmp / "cache", rng, "f" * 64)
            X2, rec, rsha = LTM.load_next("musique", "s1eval", sp, ps)   # out None: ltmcols' OUT, GNR's seed 0 here
            assert np.array_equal(X, X2) and (rec["stage"], rec["declared_in"]) == ("GNR", DECLARED), rec
            with GNX.on_split(sp, "f" * 64):
                GNX.LOADED.append({"dataset": "musique", "carve": "s1eval", "record_sha256": rsha})
                nb = GNX.next_block(sp)
            GNX.LOADED.clear()
        assert (nb["declared_in"], nb["seed"], nb["columns_root"], nb["fit_sha256"]) == (DECLARED, 0, "s0", "f" * 64), nb
        assert nb["first_sha256"] == GNRC.sha_file(fp) and nb["carves"] == {"musique/s1eval": rsha}, nb
        with on_gnr(1, groot):
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps)        # seed 1 has no columns
        Y = LTM.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)
        assert LTM.load_next("musique", "s1eval", sp, ps, tmp / "ltm")[1]["stage"] == "LTM" and not np.array_equal(X, Y)
        must_stop(LTM.load_next, "musique", "s1eval", sp, ps, GNRC.seed_root(0, groot))  # GNR's columns outside on_gnr
        with on_gnr(0, groot):
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "ltm")   # LTM's columns inside
        cd = LTMC.cols_dir(GNRC.seed_root(0, groot), sp, "musique", "s1eval")
        name = fit_name(0, sp)
        assert len(next_problems(name, 0, sp, "read", nb, tmp)) == 1     # no gnr.json beside the columns
        M1.WQ.write_json(cd / "gnr.json", {"stage": STAGE, "split": sp, "seed": 0, "record_sha256": rsha,
                                           "first_sha256": GNRC.sha_file(fp)})
        assert next_problems(name, 0, sp, "read", nb, tmp) == []
        assert carve_problems(0, sp, {"musique/s1eval": None}, tmp) == []
        assert len(carve_problems(0, sp, {"musique/s1eval": "0" * 64}, tmp)) == 1
        assert len(next_problems(name, 0, sp, "read", dict(nb, seed=1), tmp)) == 1
        assert len(next_problems(name, 0, sp, "read", dict(nb, declared_in=GNX.DECLARED), tmp)) == 1
        toy_first(tmp, 0, sp, {"utc": "later"})                             # the first pass changed after the columns
        assert len(next_problems(name, 0, sp, "read", nb, tmp)) == 2
        assert len(carve_problems(0, sp, {"musique/s1eval": None}, tmp)) == 1
        assert snapshot() == s0
    finally:
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. GNX's toy term under GNR's patches: zg6r's toy carve loads GNR-stamped columns, at gnx_w = 0 the forwards are
    #    zg6r's bit for bit, the term trains in lean_gpu's loop; the patches restored
    G = GNX.import_g()
    sg = snapshot(G)
    with on_gnr():
        t = GNX.toy_term(G)
        rec = GNX.term_record(G)
    assert (t["no_split"], t["other_split"], t["other_scorer"]) == ("refused",) * 3, t
    assert t["loaded"] == ["metaqa/toy"] and t["scorer"] == "a" * 64 and t["carve_restored"], t
    assert t["files_are_toy_cols"] and t["carve_cols_are_files"] and t["carve_cols_on_cpu"], t
    assert t["nan_set"] > 0 and t["next_twice"] == "refused", t
    assert t["rank_rows_are_carve_rows"] and t["next_rows_are_carve_rows"] and t["next_nan_free"], t
    assert t["gnx_zero"] and t["gnx_shape"] == (16, GNX.IN) and t["l1_out"] == 16 and t["term_params"] == 16 * GNX.IN, t
    assert not t["plain_has_term"] and t["gnx_in_cleared"], t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"] and t["gnx_grad"] > 0 and t["term_moves_score"], t
    assert t["l1_outside"] == "refused" and t["no_next"] == "refused" and t["model_restored"], t
    assert t["strict_load_patched"] == "passed" and t["strict_load_plain"] == "refused", t
    assert t["fit_gnx_moved"] and t["fit_loss_finite"] and t["fit_restored"], t
    assert rec["term"].startswith("GNR's NEXT columns") and rec["term_parameters"] == 1920 and rec["inputs"] == 15, rec
    assert rec["parameters_h128"] == 182138 and rec["columns"] == list(LTMC.COLS), rec
    assert snapshot(G) == sg
    # 6. the stamp under GNR's names: the train's NEXT block is GNR's, with GNR's term and the first pass's sha256; one
    #    without the columns stops
    seen, orig_stamp = [], M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    tmp = Path(tempfile.mkdtemp(prefix="zgnr_"))
    try:
        fp = toy_first(tmp, 0, "WQ-hotpotqa")
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_gnr(0, tmp / OUT), GNX.on_gnx(G, ISO.KEEP):
            assert (M1.STAGE, M1.KEY) == (STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            with GNX.on_split("WQ-hotpotqa"):
                must_stop(M1.stamp, hp, {}, "screen", {"x": 1})
            with GNX.on_split("WQ-hotpotqa", "c" * 64):
                GNX.LOADED.extend({"dataset": k.split("/")[0], "carve": k.split("/")[1], "record_sha256": "r"}
                                  for k in GNX.train_carves("WQ-hotpotqa"))
                M1.stamp(hp, {}, "screen", {"x": 1})
            GNX.LOADED.clear()
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    assert [s[:2] for s in seen] == [(hp, "screen")], seen
    e = seen[0][2]
    assert (e["next"]["declared_in"], e["next"]["fit_sha256"], e["base"]) == (DECLARED, "c" * 64,
                                                                             "scr-ml1-zg6r-s0-hp"), e
    assert e["next"]["seed"] == 0 and e["next"]["first_sha256"] is not None and e["next"]["columns_root"] == "s0", e
    assert e["term"].startswith("GNR's NEXT columns") and e["term_parameters"] == 1920 and e["x"] == 1, e
    assert snapshot(G) == sg and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
    # 7. what follows, by the verdicts
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("GNR's input goes to the stack")
    assert next_form({"verdict": NO_GAIN}, None).startswith("the in-sample anchors close")
    assert next_form({"verdict": NOT_ADOPTED}, None).startswith("the in-sample anchors close")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("the in-sample anchors")
    assert next_form({"verdict": INCOMPLETE}, None).startswith("what is missing")
    log("selftest: names; the patches and their restoring (also on a failure), the columns' root per seed; the first "
        "pass a fit stands on and its six refusals, the fit refused before any train; GNR-stamped columns load inside, "
        "from the seed's root, and stop outside and under another seed, LTM's stop inside; the NEXT block names GNR's "
        "declaration, seed, root and first pass, and the columns' provenance checks; GNX's toy term under GNR's patches "
        "(zg6r's forwards bit for bit at gnx_w = 0, the term trains in lean_gpu's loop); the stamp's NEXT block GNR's "
        "with GNR's term and its refusal without columns; what follows")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls", "check-cols", "s0stop-check", "s0stop-run",
                                               "gate"))
    ap.add_argument("--name")
    ap.add_argument("--want")
    ap.add_argument("--seed", type=int, default=0)
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
    if a.cmd == "check-cols":
        return check_cols(a.seed)
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
