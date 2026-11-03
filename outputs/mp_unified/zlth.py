"""LTH's stage (docs/LTH_RANKED_NEGATIVES_NEXT_MLP.md): zrckph, LTM's MLP zrckpn with its NEXT columns built by lthcols.py
(LTA's columns, each anchor's learned next-hop term over every pool row outside the first stage's top five, linked to it
or not, with the scorer's negatives drawn in proportion to 1 over their rank by rrf, not uniformly), on PKO's pools under
WQ, on the host's CPU with DOB's device, against LTM's zrckpn at its three seeds; the compares and the calls. This file is
zlta.py with LTA's names made LTH's, LTH's next form (section 7) and LTA's filed screen beside; zlta.py is not imported
or edited.

A fit is zltm's fit of zrckpn under LTH's names, stamp key, stage, declaration and naming (zlth and lthcols beside LTM's
code in the stamp), with ltmcols' output root, stage and declaration LTH's for the length of the fit, so zltm's carve loads
LTH's columns and checks them as it checks LTM's. zrckph is zrckpn's model, carve, pick, trainer, read cap and device;
only its NEXT columns differ. zltm, zml1s, ltmcols, zltg and zdep are imported and not edited; every patched name is put
back afterwards, also on a failure. LTM's fits, compares and columns are only read; LTH writes under outputs/lth and its
own fit and compare names.

    python outputs/mp_unified/zlth.py fit --name scr-lth-zrckph-s0-hp   (train when needed, then read; kept when read)
    python outputs/mp_unified/zlth.py compare-all                       (each read fit against LTM's zrckpn, same seed)
    python outputs/mp_unified/zlth.py calls                             (outputs/lth/calls.json and .md)
    python outputs/mp_unified/zlth.py --selftest
"""
import os
import sys

THREADS = 6                                     # zltm's (DOB's): torch, OpenMP, MKL and OpenBLAS threads
if __name__ == "__main__":                      # before numpy or torch loads; a CPU fit never sees the GPU
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = str(THREADS)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import zdep as DEP  # noqa: E402  (musique's hop slices)
import zltg as LTG  # noqa: E402  (LTG's in-domain rule: its keys, pure)
import zltm as ZL  # noqa: E402  (LTM's fit of zrckpn, its names, its rule and its fits' reads)
import lthcols as LTHC  # noqa: E402  (LTH's columns: their stage, declaration and root)

LTMC = ZL.LTMC                                  # ltmcols: the columns' layout, records and carves
DECLARED = "docs/LTH_RANKED_NEGATIVES_NEXT_MLP.md"
STAGE = "LTH"
KEY = "lth"                                     # the stamp's key in an LTH fit's screen.json and read.json and its compares
OUT = "outputs/lth"                             # calls.json and .md (the columns under cols/, the scorer's work under work/)
ARM = ZL.ARM                                    # zrckpn's model, chain carve, pick, trainer, read cap and device
FORM = "zrckph"                                 # the fits' name: zrckpn on LTH's columns
LTH_CODE = ZL.LTM_CODE + ("zlth", "lthcols")    # LTM's files in the code stamp, then this stage's two
CAND = ZL.CAND
NAMING = ("zrckp + LTH's next-hop columns (NEXT over every pool row: each anchor's learned next-hop term on every row "
          "outside the first stage's top five, linked or not, cross-fitted; the scorer's negatives drawn in proportion to "
          "1 over their rank by rrf)")
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE, ADOPT = (ZL.ADOPT_FOR_FULL_RUN, ZL.NO_GAIN, ZL.NOT_ADOPTED,
                                                               ZL.INCOMPLETE, ZL.ADOPT)


def fit_name(seed, split):
    return f"scr-lth-{FORM}-s{seed}{M1.WQ.SUF[split]}"


def compare_name(seed, split):
    """zrckph against LTM's zrckpn of the same seed and split."""
    return f"scr-lth-{FORM}-vs-{ARM}-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
SCREEN = tuple(fit_name(0, sp) for sp in M1.SPLITS)
AGAINST = {fit_name(sd, sp): ZL.fit_name(sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
COMPARES = {compare_name(sd, sp): (fit_name(sd, sp), ZL.fit_name(sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert THREADS == ZL.THREADS and ARM == "zrckpn" and (LTHC.DECLARED, LTHC.STAGE) == (DECLARED, STAGE)
assert LTHC.OUT == ROOT / OUT and LTHC.LTM_OUT == ROOT / ZL.OUT
assert len(FITS) == len(COMPARES) == 6 and set(AGAINST.values()) == set(ZL.FITS)
assert not set(FITS) & (set(M1.ALL_FITS) | set(ZL.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zlth: {msg}", flush=True)


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_zltm():
    """zltm with LTH's fits, declaration, stage, stamp key, code and naming; restored after (also on a failure)."""
    with M1.WQ.patch(ZL, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, LTM_CODE=LTH_CODE, NAMING=NAMING):
        yield


@contextlib.contextmanager
def on_cols(out=None):
    """ltmcols' output root, stage and declaration LTH's for the block: zltm.load_next reads them at call time, so the
    carves a fit builds load LTH's columns (out: LTH's root, or a test's); restored after (also on a failure)."""
    with M1.WQ.patch(LTMC, OUT=Path(LTHC.OUT if out is None else out), STAGE=STAGE, DECLARED=DECLARED):
        yield


@contextlib.contextmanager
def on_names():
    """zml1s with LTH's names (zltm's on_names under on_zltm): enough for the stamps, the compares and the calls."""
    with on_zltm(), ZL.on_names():
        yield


def lth_stamp(orig):
    """zml1s.stamp for an LTH fit: LTM's fit of its seed and split as what it is against, on both records; LTH's naming
    on the train's. A train or read whose NEXT block is not LTH's stops unstamped."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            raise SystemExit(f"zlth: {name} is not one of LTH's fits")
        e = {**(extra or {}), "against": AGAINST[name]}
        nk = e.get("next") or {}
        if nk.get("declared_in") != DECLARED or not nk.get("fit_sha256"):
            raise SystemExit(f"zlth: {name}'s {what} did not read LTH's columns ({nk.get('declared_in')}); unstamped")
        if what == "screen":
            e["what"] = NAMING
        return orig(name, env, what, e)
    return stamp


@contextlib.contextmanager
def on_fit():
    """A fit's patches beside zltm's own: LTH's names in zltm, LTH's columns in ltmcols' place and this file's stamp under
    zml1s's (zltm's on_ltm wraps the stamp it finds)."""
    with on_zltm(), on_cols(), M1.WQ.patch(M1, stamp=lth_stamp(M1.stamp)):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zlth: {name} is not one of LTH's fits ({', '.join(FITS)})")
    with on_fit():
        return ZL.fit(name)


# ── the compares and the reads ───────────────────────────────────────────────


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: LTH's, or LTM's for the fits LTH's are against."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    if name in ZL.FITS:
        return ZL.fit_block(name, what, base)
    raise SystemExit(f"zlth: {name} is neither LTH's nor LTM's")


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != ZL.ROWS_CAP:
        raise SystemExit(f"zlth: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """Each read LTH fit against LTM's zrckpn of its seed and split, through zwq's run_compare, stamped with LTH's block;
    a compare filed before is kept."""
    done, ran, wait = [], [], []
    for cname, (new, bse) in COMPARES.items():
        if not (is_read(new, base) and is_read(bse, base)):
            wait.append(cname)
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zlth: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zlth: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zlth_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


def zrckpn_base(base=ROOT):
    """({read: {seed: R@5}} of LTM's zrckpn at its three seeds on both splits, from LTM's compares and musique's read
    files; problems)."""
    n5, _b5, bad = ZL.r5_from(base)
    out = {rd: dict(v) for rd, v in n5.items()}
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            f = ZL.fit_name(sd, sp)
            s = DEP.slice_r5(f, base)
            want = out.get((sp, DEP.SLICE_DS), {}).get(sd)
            if s is None or want is None or abs(s["all"] - want) > DEP.R5_TOL:
                bad.append(f"{f}: musique's read file gives {None if s is None else round(s['all'], 5)}, its compare {want}")
                continue
            for sl in DEP.SLICES:
                if s[sl] is not None:
                    out.setdefault((sp, sl), {})[sd] = s[sl]
    return out, bad


def lth_reads(base=ROOT):
    """({read: {seed: R@5}} of LTH's read fits from their compares and musique's read files, {read: {seed: the compare's
    base R@5}}, problems)."""
    new, against, bad = {}, {}, []
    for cname, (nm, bse) in COMPARES.items():
        _a, sd, sp = FITS[nm]
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, nm, bse):
            bad.append(f"{cname} is not LTH's compare of {nm} against {bse}")
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            bad.append(f"{cname} is not {CAND} on s1eval against one base")
            continue
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            against.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
        s = DEP.slice_r5(nm, base)
        want = new.get((sp, DEP.SLICE_DS), {}).get(sd)
        if s is None or want is None or abs(s["all"] - want) > DEP.R5_TOL:
            bad.append(f"{nm}: musique's read file gives {None if s is None else round(s['all'], 5)}, its compare {want}")
            continue
        for sl in DEP.SLICES:
            if s[sl] is not None:
                new.setdefault((sp, sl), {})[sd] = s[sl]
    return new, against, bad


# ── the rule (section 5): LTM's, with zrckpn as the base ─────────────────────


def decide_screen(new, base):
    """zltm's screen (LTG's per-read call on its ten deciding reads, floor max(0.0075, 2 x the base's spread)), zrckph at
    seed 0 against zrckpn's three seeds."""
    return ZL.decide_screen(new, base)


def decide_full(new, base):
    """zltm's full run (LTG's zml1.call at three seeds; ADOPT: no LOSS and a GAIN on a dataset read)."""
    return ZL.decide_full(new, base)


def next_form(screen, full):
    """Section 7: what the verdicts name next (decides nothing about this form)."""
    v = (full or {}).get("verdict") if full is not None else None
    s = (screen or {}).get("verdict")
    if v == ADOPT:
        return ("zrckph is the MLP candidate; next form: LTH's draw with "
                "the term on two-link rows (LTM's lists widened by one link: each anchor's term on the rows two links from it)")
    if s in (NO_GAIN, NOT_ADOPTED) or v == NOT_ADOPTED:
        return ("the ranked negatives close; next form: "
                "the term on two-link rows (LTM's lists widened by one link: each anchor's term on the rows two links from it)")
    if s == ADOPT_FOR_FULL_RUN:
        return "the full run (seeds 1 and 2), behind lth-gate"
    return None


def fit_problems(base=ROOT):
    """Why a read LTH fit is not the declared one: its gate; its device, threads, pick and rows cap against LTM's fit of
    its seed and split; what it is against; its columns (LTH's declaration, one scorer for its train and read, the split's
    present LTH fit, every training carve in its train and every eval carve in its read, each carve's record unchanged
    since); its trainer's code against LTM's fit's (every file LTM's names, then zlth and lthcols); and one lthcols.py
    for its code stamp, its scorer and its columns."""
    out = []
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    so = Path(base) / OUT
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = AGAINST[name]
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
            ls, lr = fit_block(bse, "screen", base), fit_block(bse, "read", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            out.append(f"{name} names a gate that is not PKO's present gate")
        for what, k, lk in (("train", ks, ls), ("read", kr, lr)):
            got = tuple(json.dumps(k.get(x), sort_keys=True) for x in ("device", "threads", "pick", "rows_cap"))
            want = tuple(json.dumps(lk.get(x), sort_keys=True) for x in ("device", "threads", "pick", "rows_cap"))
            if got != want or k.get("threads") != THREADS or k.get("pick") != list(ZL.PICK):
                out.append(f"{name}: its {what} ran with {got}, {bse}'s with {want}")
            if k.get("against") != bse:
                out.append(f"{name}: its {what} is against {k.get('against')}, not {bse}")
        nt, nr = ks.get("next") or {}, kr.get("next") or {}
        fr = so / f"fit_{sp}.json"
        frec = json.loads(fr.read_text(encoding="utf-8")) if fr.is_file() else {}
        fsha = M1.WQ.sha_file(fr) if fr.is_file() else None
        if (nt.get("declared_in"), nr.get("declared_in")) != (DECLARED, DECLARED):
            out.append(f"{name}: its train and read read the columns of {nt.get('declared_in')}, {nr.get('declared_in')}")
        if not nt.get("fit_sha256") or nt.get("fit_sha256") != nr.get("fit_sha256") or nt.get("fit_sha256") != fsha:
            out.append(f"{name}: its train read the scorer {str(nt.get('fit_sha256'))[:12]}, its read "
                       f"{str(nr.get('fit_sha256'))[:12]}, {sp}'s present LTH fit is {str(fsha)[:12]}")
        miss = sorted((set(f"{d}/{c}" for d, c in LTMC.train_carves(sp)) - set(nt.get("carves") or {}))
                      | ({f"{d}/{LTMC.EVAL_CARVE}" for d in M1.WQ.EVAL_ORDER} - set(nr.get("carves") or {})))
        if miss:
            out.append(f"{name}: no columns of {miss} in its train or read")
        built = {(frec.get("lth") or {}).get("lthcols_sha256")}
        for key, sha in {**(nt.get("carves") or {}), **(nr.get("carves") or {})}.items():
            ds, cv = key.split("/")
            p = LTMC.cols_dir(so, sp, ds, cv) / "record.json"
            if not p.is_file() or M1.WQ.sha_file(p) != sha:
                out.append(f"{name}: the columns of {key} are not the ones it read")
                continue
            built.add((json.loads(p.read_text(encoding="utf-8")).get("lth") or {}).get("lthcols_sha256"))
        cb, cn = ls.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if diff or not {"zlth_sha256", "lthcols_sha256"} <= set(cn):
            out.append(f"{name}: code differs from {bse}'s in {diff} (zlth and lthcols stamped: "
                       f"{ {'zlth_sha256', 'lthcols_sha256'} <= set(cn)})")
        if built != {cn.get("lthcols_sha256")}:
            out.append(f"{name}: its scorer and columns were built by lthcols.py {sorted(str(x)[:12] for x in built)}, "
                       f"its code stamp names {str(cn.get('lthcols_sha256'))[:12]}")
    return out


# ── beside, deciding nothing ─────────────────────────────────────────────────


def columns_beside(base=ROOT):
    """Per split and carve: the A cases' T2 under bil (and by hop) of LTH's scorer among every row outside the top five,
    its linked and unlinked targets apart and rrf's T2 on the same cases; LTA's (the same cases, uniform negatives) the
    same way; LTM's among linked rows beside (lists of other lengths: not one measure) (lthcols', ltacols' and ltmcols'
    carve records)."""
    out = {}
    so, ao, lo = Path(base) / OUT, Path(base) / "outputs" / "lta", Path(base) / ZL.OUT
    for sp in M1.SPLITS:
        d = {}
        for ds, cv in LTMC.col_carves(sp):
            row = {}
            for who, o in (("lth", so), ("lta", ao), ("ltm", lo)):
                p = LTMC.cols_dir(o, sp, ds, cv) / "record.json"
                if not p.is_file():
                    continue
                r = json.loads(p.read_text(encoding="utf-8"))
                aa = r.get("A_assigned") or {}
                a = aa.get("bil") or {}
                row[who] = {"A_T2_bil": a.get("T2"), "T2_by_hop": a.get("T2_by_hop"), "A_ranked": r.get("A_ranked")}
                if who in ("lth", "lta"):
                    row[who].update({"T2_linked": a.get("T2_linked"), "T2_unlinked": a.get("T2_unlinked"),
                                     "A_T2_rrf": (aa.get("rrf") or {}).get("T2"), "linked_share": r.get("A_linked_share")})
            if row:
                d[f"{ds}/{cv}"] = row
        out[sp] = d
    return out


def lta_beside(base=ROOT):
    """LTA's screen verdict and its zrckpa - zrckpn rows at seed 0 (outputs/lta/calls.json, its uniform negatives), as
    filed."""
    p = Path(base) / "outputs" / "lta" / "calls.json"
    if not p.is_file():
        return None
    r = json.loads(p.read_text(encoding="utf-8"))
    sc = r.get("screen") or {}
    return {"screen": sc.get("verdict"), "full_run": (r.get("full_run") or {}).get("verdict"),
            "vs_zrckpn": [{"split": x.get("split"), "read": x.get("read"), "delta": x.get("delta"), "call": x.get("call")}
                          for x in sc.get("reads") or []]}


def beside(base=ROOT):
    """Per read LTH fit, the seconds."""
    out = {}
    for name in FITS:
        if not is_read(name, base):
            continue
        tj = M1.screen_dir(base) / "fits" / name / "ml1_timing.json"
        tm = M1.WQ.read_json(tj) if tj.is_file() else {}
        out[name] = {"train_seconds": tm.get("train_seconds"), "read_seconds": tm.get("read_seconds")}
    return out


def calls(base=ROOT):
    zb, bad = zrckpn_base(base)
    new, against, own = lth_reads(base)
    bad += own
    for rd, d in sorted(against.items()):
        for sd, v in sorted(d.items()):
            b = zb.get(rd, {}).get(sd)
            if b is None or abs(b - v) > RULE.EPS:
                bad.append(f"{compare_name(sd, rd[0])}: its base gives {rd[1]} {v}, LTM's zrckpn at seed {sd} {b}")
    screen = decide_screen(new, zb)
    full_ready = all(is_read(f, base) for f in FITS)
    full = decide_full(new, zb) if full_ready else None
    bad += fit_problems(base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "form": FORM, "arm": ARM, "what": NAMING,
           "base": "LTM's zrckpn at seeds 0, 1 and 2", "device": ZL.DOB.DEVICE, "threads": THREADS, "pick": list(ZL.PICK),
           "screen": screen, "full_run": full, "next_form": next_form(screen, full), "problems": bad,
           "beside": beside(base), "columns": columns_beside(base), "lta": lta_beside(base),
           "base_reads": {f"{rd[0]} {rd[1]}": {str(k): v for k, v in sorted(s.items())} for rd, s in sorted(zb.items())},
           "zlth_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)

    def t(v):
        return "-" if v is None else f"{v:.3f}"
    md = [f"# LTH: {FORM} ({NAMING}) against LTM's zrckpn, on PKO's pools, on the host's CPU", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    rows = screen["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in screen.get("beside", [])]
    if rows:
        md += ["| split | read | zrckpn s0, s1, s2 | zrckph s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in rows]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", "", f"counts: {json.dumps(full.get('counts', {}))}", ""]
        frows = full["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in full.get("beside", [])]
        if frows:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            md += [f"| {r['split']} | {r['read']} | {r['delta'][0]:+.4f} | {r['delta'][1]:+.4f} | {r['delta'][2]:+.4f} | "
                   f"{r['floor']:.4f} | {r['call']} |" for r in frows]
    md += ["", f"next (section 7): {rec['next_form']}"]
    cols = [(sp, k, c) for sp, d in rec["columns"].items() for k, c in d.items()]
    if cols:
        md += ["", "Beside, deciding nothing: the A cases' T2 under bil, LTH's scorer among every row outside the top five "
               "(linked and unlinked targets apart; rrf's on the same cases), LTA's (uniform negatives) the same way, LTM's "
               "among linked rows (carve records):", "",
               "| split | carve | A ranked | LTH bil | linked | unlinked | LTA bil | linked | unlinked | rrf | LTM bil "
               "| musique 2/3/4-hop, LTH |",
               "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for sp, k, c in cols:
            a, b, u = c.get("ltm") or {}, c.get("lth") or {}, c.get("lta") or {}
            hb = b.get("T2_by_hop") or {}
            hops = ", ".join(t(hb.get(h)) for h in ("2hop", "3hop", "4hop")) if hb else ""
            md.append(f"| {sp} | {k} | {b.get('A_ranked')} | {t(b.get('A_T2_bil'))} | {t(b.get('T2_linked'))} | "
                      f"{t(b.get('T2_unlinked'))} | {t(u.get('A_T2_bil'))} | {t(u.get('T2_linked'))} | "
                      f"{t(u.get('T2_unlinked'))} | {t(b.get('A_T2_rrf'))} | {t(a.get('A_T2_bil'))} | {hops} |")
    if rec["lta"]:
        md += ["", f"LTA's zrckpa (uniform negatives; screen {rec['lta']['screen']}, full run {rec['lta']['full_run']}) - "
               "zrckpn at seed 0, as filed: " + "; ".join(f"{r['split']} {r['read']} {r['delta']:+.4f} {r['call']}"
                                                        for r in rec["lta"]["vs_zrckpn"] if r.get("delta") is not None)]
    if rec["beside"]:
        md += ["", "| fit | train s | read s |", "|---|---:|---:|"]
        md += [f"| {n} | {b['train_seconds']} | {b['read_seconds']} |" for n, b in rec["beside"].items()]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; next: {rec['next_form']}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import lean_cache as LC
    must_stop = ZL.must_stop
    # 1. the names
    assert fit_name(0, "WQ-hotpotqa") == "scr-lth-zrckph-s0-hp" and fit_name(2, "WQ-musique") == "scr-lth-zrckph-s2"
    assert SCREEN == ("scr-lth-zrckph-s0", "scr-lth-zrckph-s0-hp")
    assert COMPARES[compare_name(1, "WQ-hotpotqa")] == ("scr-lth-zrckph-s1-hp", "scr-ltm-zrckpn-s1-hp")
    assert all(FITS[n] == ZL.FITS[AGAINST[n]] for n in FITS)
    assert LTH_CODE[:-2] == ZL.LTM_CODE and LTH_CODE[-2:] == ("zlth", "lthcols")
    assert all((HERE / f"{m}.py").is_file() for m in LTH_CODE)
    # 2. the patches: zltm, ltmcols and zml1s under LTH's names, columns and stamp; all restored, also on a failure
    zl_k = ("FITS", "DECLARED", "STAGE", "KEY", "LTM_CODE", "NAMING")
    lc_k = ("OUT", "STAGE", "DECLARED")
    m1_k = ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas", "train", "read")
    zl0, lc0 = {k: getattr(ZL, k) for k in zl_k}, {k: getattr(LTMC, k) for k in lc_k}
    m10, n0 = {k: getattr(M1, k) for k in m1_k}, len(M1.ALL_FITS)
    with on_fit():
        assert ZL.FITS is FITS and (ZL.DECLARED, ZL.STAGE, ZL.KEY, ZL.LTM_CODE, ZL.NAMING) == (
            DECLARED, STAGE, KEY, LTH_CODE, NAMING)
        assert (LTMC.OUT, LTMC.STAGE, LTMC.DECLARED) == (LTHC.OUT, STAGE, DECLARED)
        assert M1.stamp is not m10["stamp"] and M1.stamp.__qualname__.startswith("lth_stamp")
        assert ZL.next_block("WQ-hotpotqa")["declared_in"] == DECLARED
    with on_names():
        assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and all(M1.ALL_FITS[n] == FITS[n] for n in FITS)
        cs = M1.code_shas(ARM)
        assert {"zlth_sha256", "lthcols_sha256", "zltm_sha256", "ltmcols_sha256", "zml1s_sha256", "zp1c3p_sha256"} <= set(cs)
        assert cs["lthcols_sha256"] == M1.WQ.sha_src(HERE / "lthcols.py")
        M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
        try:
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-lth-zrckph-s0-hp"
        finally:
            M1._ACTIVE[0] = None
    try:
        with on_fit(), on_names():
            raise KeyError("x")
    except KeyError:
        pass
    assert {k: getattr(ZL, k) for k in zl_k} == zl0 and {k: getattr(LTMC, k) for k in lc_k} == lc0
    assert {k: getattr(M1, k) for k in m1_k} == m10 and len(M1.ALL_FITS) == n0 and not set(FITS) & set(M1.ALL_FITS)
    must_stop(fit, ZL.fit_name(0, "WQ-hotpotqa"))
    must_stop(fit_block, "scr-x", "screen")
    # 3. the stamp: LTM's fit as what it is against on both records, the naming on the train's; a record whose NEXT
    #    block is not LTH's, or a name not LTH's, stops unstamped
    got = []
    st = lth_stamp(lambda *a: got.append(a))
    nx = {"declared_in": DECLARED, "fit_sha256": "f" * 64}
    must_stop(st, fit_name(0, "WQ-musique"), "env", "screen", {"what": "x", "next": {**nx, "declared_in": ZL.DECLARED}})
    must_stop(st, fit_name(0, "WQ-musique"), "env", "read", {"next": {"declared_in": DECLARED}})
    must_stop(st, ZL.fit_name(0, "WQ-musique"), "env", "read", {"next": nx})
    assert not got
    st(fit_name(0, "WQ-musique"), "env", "screen", {"what": "x", "against": "y", "next": nx})
    st(fit_name(0, "WQ-musique"), "env", "read", {"next": nx})
    (n_s, _e, w_s, e_s), (n_r, _e2, w_r, e_r) = got
    assert (n_s, w_s, e_s["what"], e_s["against"], e_s["next"]) == (fit_name(0, "WQ-musique"), "screen", NAMING,
                                                                    "scr-ltm-zrckpn-s0", nx)
    assert (n_r, w_r, e_r) == (fit_name(0, "WQ-musique"), "read", {"next": nx, "against": "scr-ltm-zrckpn-s0"})
    # 4. zltm's column loader under on_cols reads LTH's columns from LTH's root at call time (as a fit's carves do) and
    #    refuses LTM's; outside on_cols it reads LTM's and refuses LTH's
    tmp = Path(tempfile.mkdtemp(prefix="zlth_"))
    try:
        rng = np.random.default_rng(13)
        import ltwdiag as LTW
        M = LTW.unit(rng.normal(size=(200, 16))).astype(np.float32)
        LTW.toy_tree(tmp, "musique", "s1eval", M, rng, nq=6, n=20)
        parts, _recs = LC.part_dirs("musique", "s1eval", tmp / "cache")
        ps = {p.name: LC.sha_file(p / "record.json") for p in parts}
        sp = "WQ-hotpotqa"
        with on_cols(tmp / "lth"):
            X = ZL.toy_cols(tmp / "lth", sp, "musique", "s1eval", tmp / "cache", rng, "f" * 64)
            X2, rec, _sha = ZL.load_next("musique", "s1eval", sp, ps)          # out None: ltmcols' OUT, LTH's here
            assert np.array_equal(X, X2) and rec["stage"] == STAGE and rec["declared_in"] == DECLARED
        Y = ZL.toy_cols(tmp / "ltm", sp, "musique", "s1eval", tmp / "cache", rng, "e" * 64)
        assert ZL.load_next("musique", "s1eval", sp, ps, tmp / "ltm")[1]["stage"] == ZL.STAGE == "LTM"
        assert not np.array_equal(X, Y)
        must_stop(ZL.load_next, "musique", "s1eval", sp, ps, tmp / "lth")      # LTH's columns outside on_cols
        with on_cols(tmp / "lth"):
            must_stop(ZL.load_next, "musique", "s1eval", sp, ps, tmp / "ltm")  # LTM's columns inside
        assert {k: getattr(LTMC, k) for k in lc_k} == lc0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the compares' reading on made records under a temporary root: LTM's zrckpn at three seeds (its pair compares)
    #    and LTH's fits against it; the slices from made read files; the calls drawn
    tmp = Path(tempfile.mkdtemp(prefix="zlth_"))
    try:
        scr = M1.screen_dir(tmp)
        scr.mkdir(parents=True)
        b5 = {0: 0.50, 1: 0.51, 2: 0.49}

        def rows(v):
            return [{"dataset": ds, "read": "in-domain", "questions": 10, "base": [v[1], 0.1, 0.1], "new": [v[0], 0.1, 0.1]}
                    for ds in RULE.DATASETS]
        for sp in M1.SPLITS:
            for sd in RULE.SEEDS:
                bse = ZL.DOB.fit_name(ZL.BASE, sd, sp)
                M1.WQ.write_json(scr / f"{ZL.pair_name(sd, sp)}.json", {
                    "candidate": CAND, "carve": "s1eval", "bases": ["outputs\\screen\\fits\\" + bse],
                    "by_base": {"outputs\\screen\\fits\\" + bse: {"rows": rows((b5[sd], 0.45))}},
                    ZL.KEY: {"declared_in": ZL.DECLARED, "compare": ZL.pair_name(sd, sp), "new": ZL.fit_name(sd, sp),
                             "base": bse}})
        hot = ("WQ-hotpotqa", "musique")
        for sp in M1.SPLITS:
            cname, (nm, bse) = compare_name(0, sp), COMPARES[compare_name(0, sp)]
            M1.WQ.write_json(scr / f"{cname}.json", {
                "candidate": CAND, "carve": "s1eval", "bases": ["outputs\\screen\\fits\\" + bse],
                "by_base": {"outputs\\screen\\fits\\" + bse: {"rows": rows((0.50 if sp != hot[0] else 0.515, b5[0]))}},
                KEY: {"declared_in": DECLARED, "compare": cname, "new": nm, "base": bse}})
        files = {}
        for sp in M1.SPLITS:
            for sd in RULE.SEEDS:
                files[ZL.fit_name(sd, sp)] = {"all": b5[sd], "musique-3hop": 0.40, "musique-4hop": 0.30}
            files[fit_name(0, sp)] = {"all": 0.515 if sp == hot[0] else 0.50, "musique-3hop": 0.42, "musique-4hop": 0.30}
        with M1.WQ.patch(DEP, slice_r5=lambda f, base=None: files.get(f)):
            zb, bad = zrckpn_base(tmp)
            assert not bad and all(zb[(sp, d)] == b5 for sp in M1.SPLITS for d in RULE.DATASETS)
            assert zb[(hot[0], "musique-4hop")] == {0: 0.30, 1: 0.30, 2: 0.30}
            new, against, bad = lth_reads(tmp)
            assert not bad and new[hot] == {0: 0.515} and new[(hot[0], "musique-3hop")] == {0: 0.42}
            assert all(v == {0: b5[0]} for v in against.values())
            v = decide_screen(new, zb)          # musique +0.015 < floor 0.02 (2 x rms 0.01); 3-hop +0.02 >= 0.0075
            calls_of = {(r["split"], r["read"]): r["call"] for r in v["reads"]}
            assert v["verdict"] == ADOPT_FOR_FULL_RUN and calls_of[(hot[0], "musique-3hop")] == RULE.GAIN, v
            assert calls_of[hot] == RULE.WITHIN and len(v["reads"]) == 10
            assert fit_problems(tmp) == [] and beside(tmp) == {}
            assert calls(tmp) == 0
            cj = M1.WQ.read_json(tmp / OUT / "calls.json")
            assert cj["screen"]["verdict"] == ADOPT_FOR_FULL_RUN and cj["full_run"] is None and not cj["problems"]
            assert cj["next_form"].startswith("the full run") and (tmp / OUT / "calls.md").is_file()
            assert cj["lta"] is None and "LTA's zrckpa" not in (tmp / OUT / "calls.md").read_text(encoding="utf-8")
            la = tmp / "outputs" / "lta"
            la.mkdir(parents=True)
            M1.WQ.write_json(la / "calls.json", {"screen": {"verdict": NOT_ADOPTED, "reads": [
                {"split": hot[0], "read": "musique-4hop", "delta": -0.0111, "call": RULE.LOSS},
                {"split": hot[0], "read": "musique", "delta": None, "call": None}]}, "full_run": None})
            assert lta_beside(tmp) == {"screen": NOT_ADOPTED, "full_run": None, "vs_zrckpn": [
                {"split": hot[0], "read": "musique-4hop", "delta": -0.0111, "call": RULE.LOSS},
                {"split": hot[0], "read": "musique", "delta": None, "call": None}]}
            assert calls(tmp) == 0 and M1.WQ.read_json(tmp / OUT / "calls.json")["lta"]["screen"] == NOT_ADOPTED
            assert (f"LTA's zrckpa (uniform negatives; screen {NOT_ADOPTED}, full run None) - zrckpn at seed 0, as filed: "
                    f"{hot[0]} musique-4hop -0.0111 {RULE.LOSS}") in (tmp / OUT / "calls.md").read_text(encoding="utf-8")
            files[fit_name(0, hot[0])] = {"all": 0.60, "musique-3hop": 0.42, "musique-4hop": 0.30}
            assert lth_reads(tmp)[2], "a read file whose R@5 is not its compare's passed"
            files[fit_name(0, hot[0])] = {"all": 0.515, "musique-3hop": 0.42, "musique-4hop": 0.30}
            js = scr / f"{compare_name(0, 'WQ-musique')}.json"
            r0 = M1.WQ.read_json(js)
            M1.WQ.write_json(js, {**r0, KEY: {**r0[KEY], "base": "scr-x"}})
            assert lth_reads(tmp)[2] and decide_screen(lth_reads(tmp)[0], zb)["verdict"] == INCOMPLETE
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. the rule on made numbers: zltm's (LTG's ten deciding reads), zrckpn's seeds as the base; the next form
    keys = LTG.deciding_keys() + LTG.beside_keys()
    flat = {rd: {0: 0.5, 1: 0.5, 2: 0.5} for rd in keys}
    same = {rd: {0: 0.5} for rd in keys}
    assert decide_screen(same, flat)["verdict"] == NO_GAIN
    up = {**same, ("WQ-hotpotqa", "musique-4hop"): {0: 0.5075}}
    assert decide_screen(up, flat)["verdict"] == ADOPT_FOR_FULL_RUN
    assert decide_screen({**up, ("WQ-musique", "squad"): {0: 0.49}}, flat)["verdict"] == NOT_ADOPTED
    full = {rd: {0: 0.5, 1: 0.5, 2: 0.5} for rd in keys}
    full[("WQ-hotpotqa", "musique")] = {0: 0.52, 1: 0.51, 2: 0.51}
    assert decide_full(full, flat)["verdict"] == ADOPT
    assert next_form({"verdict": NO_GAIN}, None).startswith("the ranked negatives close; next form: the term on two-link")
    assert next_form({"verdict": NOT_ADOPTED}, None) == next_form({"verdict": NO_GAIN}, None)
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": ADOPT}).startswith("zrckph is the MLP candidate")
    assert next_form({"verdict": ADOPT_FOR_FULL_RUN}, {"verdict": NOT_ADOPTED}).startswith("the ranked negatives close")
    assert next_form({"verdict": INCOMPLETE}, None) is None
    log("selftest: names; the patches and their restoring (also on a failure); the stamp and its refusals; zltm's column "
        "loader under LTH's root and stage (and its refusals either way); the compares' and slices' reading and the "
        "calls on made records; the rule and the next form")
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
