"""GNP's stage (docs/GNP_PAIR_COLUMNS_IN_GNN.md): GNX's arm, the GNN of record with one more input block in its encoder
(docs/GNX_NEXT_COLUMNS_IN_GNN.md), reading PFS amendment A3's six pair columns (docs/PKO_FEATURE_SCREEN.md sections 11
and 12: the frozen encoder's question-conditional pair scores of every pool row against the first stage's top five,
written by gnpcols.py) in place of LTM's seven NEXT columns. On PKO's roots, on the host's GPU: the screen at seed 0 on
both WQ splits against the base's seed-0 fits under LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), DEP's
sixteen-read rule beside.

zgnx.py, gnpcols.py and zpfp.py, and every module they import, are imported and not edited. For the length of a GNP call
zgnx's names (its fits, declaration, stage, stamp key, output root, fit and pair names, code, term record and NEXT block)
are GNP's, its block width is six columns (IN = 13: [raw, z, 1]), and ltmcols' root, stage, declaration, column names and
width are GNP's: zgnx's carve then loads GNP's columns through zltm.load_next and refuses LTM's and LTA's. Everything else
in a fit is GNX's: the term (gnx_w, hidden x 13, zero-started, added to l1's output), the base, the start, the trainer
and the read. Every patched name is put back afterwards, also on a failure.

    python outputs/mp_unified/zgnp.py check-cols                   (GNP's columns of every carve both splits load)
    python outputs/mp_unified/zgnp.py fit --name scr-gnp-px-s0     (train when needed, then read; kept when read before)
    python outputs/mp_unified/zgnp.py compare-all                  (each GNP fit read against the base's fit of its seed)
    python outputs/mp_unified/zgnp.py calls                        (outputs/gnp/calls.json and .md)
    python outputs/mp_unified/zgnp.py s0stop-check                 (s0stop's check under GNP's entry)
    python outputs/mp_unified/zgnp.py s0stop-run -- COMMAND ...    (s0stop's run under GNP's entry)
    python outputs/mp_unified/zgnp.py gate --want ADOPT_FOR_FULL_RUN (s0stop's gate under GNP's entry)
    python outputs/mp_unified/zgnp.py --selftest
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
import gnpcols as GNPC  # noqa: E402  (GNP's columns: their root, names and layout; not edited after its items run)
import s0stop as S0  # noqa: E402  (the seed-0 stop, with GNP's entry patched in for a call; not edited)

M1, RULE, ISO, DEP, LTG, LTM, LTMC = GNX.M1, GNX.RULE, GNX.ISO, GNX.DEP, GNX.LTG, GNX.LTM, GNX.LTMC
PP = GNPC.PP
DECLARED = GNPC.DECLARED
STAGE = GNPC.STAGE
KEY = "gnp"                                     # the stamp's key in a GNP fit's screen.json and read.json and GNP's compares
OUT = "outputs/gnp"
TAG = "px"
WIDTH = GNPC.WIDTH                              # six columns
IN = 2 * WIDTH + 1                              # raw, z, 1
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    GNX.ADOPT_FOR_FULL_RUN, GNX.NO_GAIN, GNX.NOT_ADOPTED, GNX.ADOPT, GNX.INCOMPLETE)
PATCHED = ("FITS", "DECLARED", "STAGE", "KEY", "OUT", "fit_name", "pair_name", "code_extra", "term_record", "next_block",
           "IN", "WIDTH")
COLS_PATCHED = ("OUT", "STAGE", "DECLARED", "COLS", "WIDTH")
ORIG = {k: getattr(GNX, k) for k in ("code_extra", "term_record", "next_block")}
GNX_FITS = dict(GNX.FITS)                       # zgnx's own fit names, taken before any patch
OWN_CODE = ("zgnp", "gnpcols", "zpfp")
assert (WIDTH, IN, GNX.IN, GNX.HIDDEN) == (6, 13, 15, 128) and Path(GNPC.OUT) == ROOT / OUT


def fit_name(seed, split):
    return f"scr-gnp-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-gnp-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (GNX.ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(LTG.FITS)
                                            | set(GNX.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zgnp: {msg}", flush=True)


def own_shas():
    return {"zgnp_sha256": M1.WQ.sha_src(__file__), "gnpcols_sha256": M1.WQ.sha_src(GNPC.__file__),
            "zpfp_sha256": M1.WQ.sha_src(PP.__file__)}


def code_extra():
    """GNX's code beside zml1s's (zgnx, zdep, zltg, zltm, ltmcols), and GNP's own: this file, gnpcols.py and zpfp.py."""
    return {**ORIG["code_extra"](), **own_shas()}


def term_record(G):
    return {**ORIG["term_record"](G),
            "term": "GNP's pair columns (PFS A3's six question-conditional pair scores of every pool row against the "
                    "first stage's top five) into zg6r's encoder ([raw, z, 1] through gnx_w, zero-started, added to l1's "
                    "output)"}


def next_block(split):
    """GNX's NEXT block with GNP's declaration and the columns' root (inside on_gnp: GNP's)."""
    return {**ORIG["next_block"](split), "declared_in": DECLARED, "columns_root": M1.WQ.rel(Path(LTMC.OUT))}


@contextlib.contextmanager
def on_cols(out=None):
    """ltmcols' root, stage, declaration, column names and width GNP's (out: GNP's root, or a test's); restored after,
    also on a failure."""
    with M1.WQ.patch(LTMC, OUT=Path(GNPC.OUT if out is None else out), STAGE=STAGE, DECLARED=DECLARED,
                     COLS=GNPC.COLS, WIDTH=WIDTH):
        yield


@contextlib.contextmanager
def on_gnp(out=None):
    """zgnx with GNP's names, code, term record, NEXT block and width, and ltmcols GNP's (on_cols); restored after, also
    on a failure."""
    with on_cols(out), M1.WQ.patch(GNX, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, OUT=OUT,
                                   fit_name=fit_name, pair_name=pair_name, code_extra=code_extra,
                                   term_record=term_record, next_block=next_block, IN=IN, WIDTH=WIDTH):
        yield


@contextlib.contextmanager
def on_s0stop():
    """s0stop's stage table with GNP's entry (zgnp, its root, its two seed-0 fits, LTG's rule, this declaration), in this
    process; restored after."""
    with M1.WQ.patch(S0, STAGES={**S0.STAGES, KEY: ("zgnp", OUT, fit_name(0, S0.S0_SPLIT), fit_name(0, S0.HP_SPLIT),
                                                     "ltg", DECLARED)}):
        yield


def stop_record(base=ROOT):
    """GNP's current s0stop.json (s0stop.stop_record under GNP's entry), else None."""
    with on_s0stop():
        return S0.stop_record(KEY, base)


# ── the commands (zgnx's under GNP's names) ──────────────────────────────────


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zgnp: {name} is not one of GNP's fits ({', '.join(FITS)})")
    with on_gnp():
        return GNX.fit(name)


def check_cols():
    with on_gnp():
        return GNX.check_cols()


def compare_all(base=ROOT):
    with on_gnp():
        return GNX.compare_all(base)


def fit_block(name, what, base=ROOT):
    """A GNP fit's stamp block (zgnx's fit_block under GNP's names; STK's 'as it was screened' reads it)."""
    with on_gnp():
        return GNX.fit_block(name, what, base)


def gnp_problems(base=ROOT):
    """Why a GNP fit that is read did not read GNP's columns by this code: its train's and read's NEXT block must name
    GNP's declaration, its train's term the six columns, and its train's code this zgnp.py, gnpcols.py and zpfp.py; and
    the columns' match against zpfp (outputs/gnp/match.json) must have passed."""
    out, own = [], own_shas()
    with on_gnp():
        for name in FITS:
            if not GNX.is_read(name, base):
                continue
            try:
                ks, kr = GNX.fit_block(name, "screen", base), GNX.fit_block(name, "read", base)
            except SystemExit as e:
                out.append(str(e))
                continue
            for what, k in (("train", ks), ("read", kr)):
                if (k.get("next") or {}).get("declared_in") != DECLARED:
                    out.append(f"{name}: its {what}'s NEXT block names {(k.get('next') or {}).get('declared_in')}, "
                               f"not GNP's columns")
            if (ks.get("inputs"), ks.get("columns")) != (IN, list(GNPC.COLS)):
                out.append(f"{name}: its train's term takes {ks.get('inputs')} inputs of {ks.get('columns')}, not GNP's")
            code = ks.get("code") or {}
            diff = sorted(k for k, v in own.items() if code.get(k) != v)
            if diff:
                out.append(f"{name}: its train's code is not this GNP's in {diff}")
    m = Path(base) / OUT / "match.json"
    if any(GNX.is_read(f, base) for f in FITS):
        if not m.is_file():
            out.append(f"{M1.WQ.rel(m)}: the columns' match against zpfp is missing")
        elif not json.loads(m.read_text(encoding="utf-8")).get("pass"):
            out.append(f"{M1.WQ.rel(m)}: the columns' match against zpfp did not pass")
    return out


def next_form(screen, full):
    s = (screen or {}).get("verdict")
    v = (full or {}).get("verdict")
    if s == INCOMPLETE or v == INCOMPLETE:
        return "what is missing is fixed in an amendment, before any number is read"
    if s == ADOPT_FOR_FULL_RUN and v in (None, ADOPT):
        return ("the stack takes the pair columns as STK amendment A3 states; GNP's own full run is held by stk-own-gnp; "
                "next form: the same six columns against the GNN's own top five after a first pass (two-pass refinement)")
    return ("the pair columns as an encoder input close for the GNN; next form: the pair scores as message weights from "
            "the first stage's top five inside message passing")


def pfs_beside(base=ROOT):
    """PFS A3's probe records (decides nothing): per dataset the recall@5 difference and the call."""
    out = {}
    for ds in GNPC.MATCH_DS:
        p = Path(base) / "outputs" / "pfs" / f"pair_{ds}.json"
        if p.is_file():
            r = json.loads(p.read_text(encoding="utf-8"))
            out[ds] = {"d_recall@5": r.get("d_recall@5"), "call": r.get("call"), "questions": r.get("questions")}
    return out or None


def screen_beside(stage, base=ROOT):
    """Another stage's filed screen (decides nothing): its verdict and its reads' deltas."""
    p = Path(base) / "outputs" / stage / "calls.json"
    if not p.is_file():
        return None
    rec = json.loads(p.read_text(encoding="utf-8"))
    sc = rec.get("screen") or {}
    return {"screen": sc.get("verdict"), "full_run": (rec.get("full_run") or {}).get("verdict"),
            "reads": [{k: r.get(k) for k in ("split", "read", "delta", "call")} for r in sc.get("reads") or []]}


def calls(base=ROOT):
    """GNP's screen (and its full run, when every fit is read) by LTG's in-domain rule, DEP's sixteen-read rule beside,
    through zgnx's reads, slices and checks under GNP's names; outputs/gnp/calls.json and .md."""
    with on_gnp():
        call = DEP.iso_call(base)
        new_r5, b0 = GNX.r5_gnx(call, base)
        base3 = DEP.base_seeds(call, base)
        bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in GNP's compare, {base3.get(rd, {}).get(sd)} in its own "
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
    bad += gnp_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{GNX.ARM} with GNP's pair columns", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen, "full_run": full,
           "dep_rule_beside": dep_rule, "next_form": next_form(screen, full),
           "beside": {"pfs_a3": pfs_beside(base), "gnx": screen_beside("gnx", base), "gna": screen_beside("gna", base)},
           "problems": bad, "parameters": params, **own_shas(), "zgnx_sha256": M1.WQ.sha_src(GNX.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# GNP: {GNX.ARM} with the pair columns against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}",
          f"DEP's sixteen-read rule beside: screen {dep_rule['screen']}, full run {dep_rule['full_run']}",
          f"next form: {rec['next_form']}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | base s0, s1, s2 | GNP s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in frows:
                d = r["delta"]
                md.append(f"| {r['split']} | {r['read']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['floor']:.4f} | "
                          f"{r['call']} |")
    for st in ("gnx", "gna"):
        b = rec["beside"][st]
        if b:
            md += ["", f"{st.upper()} beside: screen {b['screen']}, full run {b['full_run']}"]
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
    s = {**{f"gnx.{k}": getattr(GNX, k) for k in PATCHED}, **{f"ltmcols.{k}": getattr(LTMC, k) for k in COLS_PATCHED},
         "zltm.toy_cols": LTM.toy_cols,
         **{f"m1.{k}": getattr(M1, k) for k in ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas",
                                                 "load_encoder", "train", "read")}, "n_fits": len(M1.ALL_FITS)}
    if G is not None:
        s.update({"g.encoder_used": G.encoder_used, "g.init": G.ZG6R.__init__, "g.forward": G.ZG6R.forward,
                  "g.carve_init": G.G6rLinkCarve.__init__, "g.carve_batch": G.G6rLinkCarve.batch})
    return s


def toy_cols6(out, split, ds, carve, cache_root, rng, fit_sha, training=False):
    """zltm.toy_cols for GNP's layout (zltm's own writes LTM's seven columns): six float16 columns per part, in [-1, 1]
    as the pair scores are, with the part and carve records ltmcols' names give inside on_cols; the columns."""
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    cd = LTMC.cols_dir(out, split, ds, carve)
    ps, xs = [], []
    for p, r in zip(parts, recs):
        k = int(r["rows"])
        C = rng.uniform(-1.0, 1.0, (k, LTMC.WIDTH)).astype(np.float16)
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
    # 1. names: GNP's own, none another stage's
    assert fit_name(0, "WQ-musique") == "scr-gnp-px-s0" and fit_name(2, "WQ-hotpotqa") == "scr-gnp-px-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-gnp-px-vs-base-s1-hp"
    import zrtr as RTR
    import zira as IRA
    import zisb as ISB
    import zgna as GNA
    import zlta as ZLA
    assert not set(FITS) & (set(RTR.FITS) | set(IRA.FITS) | set(ISB.FITS) | set(LTM.FITS) | set(GNA.FITS) | set(ZLA.FITS))
    # 2. the patches: zgnx's names, code and width GNP's and ltmcols' root, stage, declaration, columns and width GNP's
    #    inside; restored after, also on a failure; zgnx's own names, pairs and code under them
    s0 = snapshot()
    with on_gnp():
        assert GNX.FITS is FITS and (GNX.DECLARED, GNX.STAGE, GNX.KEY, GNX.OUT) == (DECLARED, STAGE, KEY, OUT)
        assert (GNX.IN, GNX.WIDTH, LTMC.WIDTH, tuple(LTMC.COLS)) == (13, 6, 6, GNPC.COLS)
        assert GNX.pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-gnp-px-s0", "scr-ml1-zg6r-s0")
        assert GNX.pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-gnp-px-s0-hp", "scr-iso-zg6rm-s0-hp")
        assert (Path(LTMC.OUT), LTMC.STAGE, LTMC.DECLARED) == (Path(GNPC.OUT), "GNP", "docs/GNP_PAIR_COLUMNS_IN_GNN.md")
        cs = GNX.code_extra()
        assert all(f"{k}_sha256" in cs for k in OWN_CODE + ("zgnx", "zdep", "zltg", "zltm", "ltmcols")), cs
        with GNX.on_names():
            assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and set(FITS) <= set(M1.ALL_FITS)
            assert not set(GNX_FITS) & set(M1.ALL_FITS)                  # zgnx's table is GNP's here
            assert all(f"{k}_sha256" in M1.code_shas(GNX.ARM) for k in OWN_CODE)
    assert snapshot() == s0 and (GNX.IN, LTMC.WIDTH, LTMC.STAGE) == (15, 7, "LTM")
    try:
        with on_gnp():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert snapshot() == s0
    must_stop(fit, "scr-gnx-nx-s0")                                      # GNX's own fit is not GNP's
    must_stop(fit, "scr-gna-nx-s0")                                      # nor GNA's
    st0 = dict(S0.STAGES)
    with on_s0stop():
        assert S0.STAGES[KEY] == ("zgnp", OUT, "scr-gnp-px-s0", "scr-gnp-px-s0-hp", "ltg", DECLARED)
        assert S0.STAGES["gnx"] == st0["gnx"] and S0.deciding("ltg") == S0.deciding(S0.STAGES["gnx"][4])
    assert S0.STAGES == st0 and KEY not in S0.STAGES
    tmp = Path(tempfile.mkdtemp(prefix="zgnp_"))
    try:                                         # GNP's stop record on made files: current only for the read it was made on
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
    # 3. the columns a GNP carve loads are gnpcols': its columns load inside on_gnp, from ltmcols' root there, and stop
    #    outside it; LTM-stamped columns stop inside it; the NEXT block names GNP's declaration and root
    import lean_cache as LC
    import ltwdiag as LTW
    tmp = Path(tempfile.mkdtemp(prefix="zgnp_"))
    try:
        rng = np.random.default_rng(29)
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        rec = GNPC.cols("musique", "s1eval", tmp, tmp / "gnp", {"musique": PP.FakeNodes(M)})
        with on_gnp(tmp / "gnp"):
            X2, r2, _sha = LTM.load_next("musique", "s1eval", sp, ps)      # out None: ltmcols' OUT, GNP's here
            assert X2.shape == (rec["rows"], 6) and (r2["stage"], r2["declared_in"]) == ("GNP", DECLARED), r2
            with GNX.on_split(sp, rec["fit_sha256"]):
                GNX.LOADED.append({"dataset": "musique", "carve": "s1eval", "record_sha256": "r"})
                nb = GNX.next_block(sp)
            GNX.LOADED.clear()
            assert (nb["declared_in"], Path(nb["columns_root"]).name, nb["fit_sha256"], nb["columns"]) == (
                DECLARED, "gnp", GNPC.rule_sha(), list(GNPC.COLS)), nb
        Y = LTM.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)
        assert LTM.load_next("musique", "s1eval", sp, ps, tmp / "ltm")[1]["stage"] == "LTM" and Y.shape[1] == 7
        must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "gnp")  # GNP's columns outside on_gnp
        with on_gnp(tmp / "gnp"):
            must_stop(LTM.load_next, "musique", "s1eval", sp, ps, tmp / "ltm")   # LTM's columns inside
        assert snapshot() == s0
    finally:
        GNX.LOADED.clear()
        shutil.rmtree(tmp, ignore_errors=True)
    # 4. GNX's toy term under GNP's patches, with six-column toy files (toy_cols6 in zltm's place for the block): zg6r's
    #    toy carve loads GNP-stamped columns, gnx_w is hidden x 13, at gnx_w = 0 the forwards are zg6r's bit for bit, the
    #    term trains in lean_gpu's loop; the patches restored
    G = GNX.import_g()
    sg = snapshot(G)
    with on_gnp(), M1.WQ.patch(LTM, toy_cols=toy_cols6):
        t = GNX.toy_term(G)
        rec = GNX.term_record(G)
    assert (t["no_split"], t["other_split"], t["other_scorer"]) == ("refused",) * 3, t
    assert t["loaded"] == ["metaqa/toy"] and t["scorer"] == "a" * 64 and t["carve_restored"], t
    assert t["files_are_toy_cols"] and t["carve_cols_are_files"] and t["carve_cols_on_cpu"], t
    assert t["nan_set"] > 0 and t["next_twice"] == "refused", t
    assert t["rank_rows_are_carve_rows"] and t["next_rows_are_carve_rows"] and t["next_nan_free"], t
    assert t["gnx_zero"] and t["gnx_shape"] == (16, IN) and t["l1_out"] == 16 and t["term_params"] == 16 * IN, t
    assert not t["plain_has_term"] and t["gnx_in_cleared"], t
    assert t["eval_start_is_zg6r"] and t["train_start_is_zg6r"] and t["gnx_grad"] > 0 and t["term_moves_score"], t
    assert t["l1_outside"] == "refused" and t["no_next"] == "refused" and t["model_restored"], t
    assert t["strict_load_patched"] == "passed" and t["strict_load_plain"] == "refused", t
    assert t["fit_gnx_moved"] and t["fit_loss_finite"] and t["fit_restored"], t
    assert rec["term"].startswith("GNP's pair columns") and rec["term_parameters"] == 128 * 13 == 1664, rec
    assert rec["inputs"] == 13 and rec["parameters_h128"] == 180218 + 1664 == 181882, rec
    assert rec["columns"] == list(GNPC.COLS), rec
    assert snapshot(G) == sg
    # 5. the stamp under GNP's names: the train's NEXT block is GNP's, with GNP's term; one without the columns stops
    seen, orig_stamp = [], M1.stamp
    hp = fit_name(0, "WQ-hotpotqa")
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_gnp(), GNX.on_gnx(G, ISO.KEEP):
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
    assert [s[:2] for s in seen] == [(hp, "screen")], seen
    e = seen[0][2]
    assert (e["next"]["declared_in"], e["next"]["fit_sha256"], e["base"]) == (DECLARED, "c" * 64, "scr-ml1-zg6r-s0-hp"), e
    assert e["next"]["columns"] == list(GNPC.COLS) and e["inputs"] == 13, e
    assert e["term"].startswith("GNP's pair columns") and e["term_parameters"] == 1664 and e["x"] == 1, e
    assert snapshot(G) == sg and GNX.CUR == {"split": None, "fit_sha256": None, "out": None}
    # 6. what follows, by the verdicts
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, None).startswith("the stack takes the pair columns")
    assert next_form({"verdict": NO_GAIN}, None).startswith("the pair columns as an encoder input close")
    assert next_form({"verdict": NOT_ADOPTED}, None).startswith("the pair columns as an encoder input close")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("the pair columns")
    assert next_form({"verdict": INCOMPLETE}, None).startswith("what is missing")
    log("selftest: names; the patches and their restoring (also on a failure); gnpcols' columns load inside, from "
        "ltmcols' root there, and stop outside, LTM's stop inside, the NEXT block names GNP's; GNX's toy term under GNP's "
        "patches with six-column toy files (gnx_w hidden x 13; zg6r's forwards bit for bit at gnx_w = 0; the term trains "
        "in lean_gpu's loop); the stamp's NEXT block GNP's with GNP's term and its refusal without columns; what follows")
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
        return check_cols()
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
