"""RWT's stage (docs/RWT_QUESTION_WEIGHTS_MLP.md): zrckpn, LTM's MLP, with each question's term in the trainer's loss
weighted inside its step by one of two universal forms: F, 1 - the question's softmax mass on its gold rows, and G, its
gold rows capped at five. The weights average to 1 over the step's questions with a gold row and pass no gradient. Seed 0
on WQ-hotpotqa, on the host's CPU, against LTM's zrckpn at seed 0; the compares and the calls.

A fit is zltm's fit under RWT's names, stamp key, stage and declaration (zrwt beside LTM's code in the stamp), with
lean_gpu.listwiseD swapped for the form's weighted loss for the length of the train (fit_variant reads it by its module
name). zltm, zml1s, zltg, zdep and lean_gpu are imported and not edited; every patched name is put back afterwards, also
on a failure. LTM's fits, compares and columns are only read; RWT writes under outputs/rwt and its own fit and compare
names.

    python outputs/mp_unified/zrwt.py fit --name scr-rwt-zrckpnf-s0-hp  (train when needed, then read; kept when read)
    python outputs/mp_unified/zrwt.py compare-all                        (each read fit against LTM's zrckpn at seed 0)
    python outputs/mp_unified/zrwt.py calls                              (outputs/rwt/calls.json and .md)
    python outputs/mp_unified/zrwt.py --selftest
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
import torch  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import zdep as DEP  # noqa: E402  (musique's hop slices)
import zltg as LTG  # noqa: E402  (LTG's in-domain rule: its keys and its per-read call, pure)
import zltm as ZL  # noqa: E402  (LTM's fit of zrckpn, its names and its fits' reads)
import lean_gpu as LG  # noqa: E402  (the trainer's loss, swapped for a train)

DECLARED = "docs/RWT_QUESTION_WEIGHTS_MLP.md"
STAGE = "RWT"
KEY = "rwt"                                     # the stamp's key in an RWT fit's screen.json and read.json and its compares
OUT = "outputs/rwt"                             # calls.json and .md
ARM = ZL.ARM                                    # zrckpn: zrckp with LTM's NEXT columns
SPLIT, SEED = "WQ-hotpotqa", 0                  # the split musique trains on; the screen's seed
CAP, FLAT = 5, 1e-6                             # G's cap (R@5's K); below this mean of 1 - P, F leaves the step unweighted
FORMS = {"f": "F, 1 - the question's softmax mass on its gold rows", "g": "G, the question's gold rows capped at 5"}
RWT_CODE = ZL.LTM_CODE + ("zrwt",)              # LTM's files in the code stamp, then this one
AGAINST = ZL.fit_name(SEED, SPLIT)              # LTM's zrckpn at seed 0 on WQ-hotpotqa
CAND = ZL.CAND
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, INCOMPLETE = (ZL.ADOPT_FOR_FULL_RUN, ZL.NO_GAIN, ZL.NOT_ADOPTED,
                                                        ZL.INCOMPLETE)


def fit_name(form):
    return f"scr-rwt-{ARM}{form}-s{SEED}{M1.WQ.SUF[SPLIT]}"


def compare_name(form):
    return f"scr-rwt-{ARM}{form}-vs-{ARM}-s{SEED}{M1.WQ.SUF[SPLIT]}"


FITS = {fit_name(f): (ARM, SEED, SPLIT) for f in FORMS}
FORM_OF = {fit_name(f): f for f in FORMS}
COMPARES = {compare_name(f): (fit_name(f), AGAINST) for f in FORMS}
NAMING = {f: f"{ARM} (LTM's MLP) with its loss's question weights by {FORMS[f]}" for f in FORMS}
DECIDING = [rd for rd in LTG.deciding_keys() if rd[0] == SPLIT]     # the six WQ-hotpotqa reads LTG's rule decides on
BESIDE = [(SPLIT, ds) for ds in M1.WQ.ZERO_SHOT[SPLIT]]             # its two zero-shot reads
assert THREADS == ZL.THREADS and AGAINST == "scr-ltm-zrckpn-s0-hp" and ZL.FITS[AGAINST] == (ARM, SEED, SPLIT)
assert set(DECIDING) == {(SPLIT, d) for d in ("webqsp", "squad", "musique", "2wiki", "musique-3hop", "musique-4hop")}
assert len(DECIDING) == 6 and set(BESIDE) == {(SPLIT, "hotpotqa"), (SPLIT, "metaqa")}
assert not set(FITS) & (set(M1.ALL_FITS) | set(ZL.FITS)) and len(FITS) == len(COMPARES) == 2


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zrwt: {msg}", flush=True)


# ── the weighted loss (section 2) ────────────────────────────────────────────

ORIG_LOSS = LG.listwiseD
STATS = {}                                      # the last swap's counts: steps, steps weighted, F's flat steps, weights' range


def weights(ng, P, form):
    """(the weights of a step's questions with a gold row, mean 1; whether F found the step flat). ng: their gold rows;
    P: their softmax mass on them (F only)."""
    if form == "unit":
        return torch.ones_like(ng), False
    if form == "g":
        n5 = ng.clamp(max=CAP)
        return n5 / n5.mean(), False
    if form == "f":
        h = 1.0 - P.clamp(0.0, 1.0)
        m = h.mean()
        return (h / m, False) if float(m) >= FLAT else (torch.ones_like(h), True)
    raise SystemExit(f"zrwt: no form {form}")


def listwise_w(scores, gold, nq, B, form):
    """lean_gpu.listwiseD with each question's term weighted by the form; the weights pass no gradient."""
    lp = LG.seg_log_softmaxD(scores, nq, B)
    g = gold.to(scores.dtype)
    ng = torch.zeros(B, device=scores.device).index_add_(0, nq, g)
    pq = -torch.zeros(B, device=scores.device).index_add_(0, nq, lp * g)
    has = ng > 0
    if not bool(has.any()):
        return scores.sum() * 0.0
    t = pq[has] / ng[has]
    with torch.no_grad():
        P = torch.zeros(B, device=scores.device).index_add_(0, nq, lp.exp() * g)[has] if form == "f" else None
        w, flat = weights(ng[has], P, form)
        if STATS.get("form") == form:
            lo, hi = float(w.min()), float(w.max())
            STATS["calls"] += 1
            STATS["weighted"] += int(lo != 1.0 or hi != 1.0)
            STATS["flat"] += int(flat)
            STATS["w_max"] = max(STATS["w_max"], hi)
            STATS["w_min"] = lo if STATS["w_min"] is None else min(STATS["w_min"], lo)
    return (w * t).mean()


@contextlib.contextmanager
def on_loss(form):
    """lean_gpu.listwiseD is the form's weighted loss for the block, its counts started afresh; put back after, also on a
    failure. A swap inside another is refused."""
    if form != "unit" and form not in FORMS:
        raise SystemExit(f"zrwt: no form {form}")
    if LG.listwiseD is not ORIG_LOSS:
        raise SystemExit("zrwt: lean_gpu.listwiseD is not lean_gpu's own")
    STATS.clear()
    STATS.update(form=form, calls=0, weighted=0, flat=0, w_max=0.0, w_min=None)

    def loss(scores, gold, nq, B):
        return listwise_w(scores, gold, nq, B, form)

    with M1.WQ.patch(LG, listwiseD=loss):
        yield STATS


# ── the patches ──────────────────────────────────────────────────────────────

ZL_TRAIN = ZL.train


def train(name, env):
    """zltm's train for an RWT fit (zp1c3p's train for zrckpn, every carve with LTM's columns), under the form's loss."""
    if name not in FORM_OF:
        raise SystemExit(f"zrwt: {name} is not one of RWT's fits")
    with on_loss(FORM_OF[name]):
        return ZL_TRAIN(name, env)


def rwt_stamp(orig):
    """zml1s.stamp for an RWT fit: its form and LTM's fit it is against; on the train's record also its naming and its
    loss's counts. A train whose form weighted no step stops unstamped."""
    def stamp(name, env, what, extra=None):
        form = FORM_OF.get(name)
        if form is None:
            raise SystemExit(f"zrwt: {name} is not one of RWT's fits")
        e = {**(extra or {}), "form": form, "against": AGAINST}
        if what == "screen":
            if STATS.get("form") != form or not STATS.get("calls") or not STATS.get("weighted"):
                raise SystemExit(f"zrwt: {name}'s trainer did not weight its steps by {form} ({STATS}); unstamped")
            e.update(what=NAMING[form], loss=dict(STATS))
        return orig(name, env, what, e)
    return stamp


@contextlib.contextmanager
def on_zltm():
    """zltm with RWT's fits, declaration, stage, stamp key and code; restored after (also on a failure)."""
    with M1.WQ.patch(ZL, FITS=FITS, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, LTM_CODE=RWT_CODE):
        yield


@contextlib.contextmanager
def on_names():
    """zml1s with RWT's names (zltm's on_names under on_zltm): enough for the stamps, the compares and the calls."""
    with on_zltm(), ZL.on_names():
        yield


@contextlib.contextmanager
def on_fit():
    """A fit's patches beside zltm's own: RWT's names in zltm, this file's train in zltm's place and its stamp under
    zml1s's (zltm's on_ltm wraps the train and the stamp it finds)."""
    with on_zltm(), M1.WQ.patch(ZL, train=train), M1.WQ.patch(M1, stamp=rwt_stamp(M1.stamp)):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zrwt: {name} is not one of RWT's fits ({', '.join(FITS)})")
    with on_fit():
        return ZL.fit(name)


# ── the compares and the calls ───────────────────────────────────────────────


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: RWT's, or LTM's for the fit RWT's are against."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    if name == AGAINST:
        return ZL.fit_block(name, what, base)
    raise SystemExit(f"zrwt: {name} is neither RWT's nor {AGAINST}")


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != ZL.ROWS_CAP:
        raise SystemExit(f"zrwt: {name}'s train and read are not on one device with zrckp's rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """Each read RWT fit against LTM's zrckpn at seed 0, through zwq's run_compare, stamped with RWT's block; a compare
    filed before is kept."""
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
            raise SystemExit(f"zrwt: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zrwt: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "form": FORM_OF[new],
                    "device": json.loads(dev.pop()), "zrwt_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before, {len(wait)} waiting for their fits")
    return 0


def ltm_base(base=ROOT):
    """({read: {seed: R@5}} of LTM's zrckpn on WQ-hotpotqa at its three seeds, from LTM's compares and musique's read
    files; problems)."""
    n5, _b5, bad = ZL.r5_from(base)
    bad = [b for b in bad if b.startswith(SPLIT)]
    out = {rd: dict(v) for rd, v in n5.items() if rd[0] == SPLIT}
    for sd in RULE.SEEDS:
        f = ZL.fit_name(sd, SPLIT)
        s = DEP.slice_r5(f, base)
        want = out.get((SPLIT, DEP.SLICE_DS), {}).get(sd)
        if s is None or want is None or abs(s["all"] - want) > DEP.R5_TOL:
            bad.append(f"{f}: musique's read file gives {None if s is None else round(s['all'], 5)}, its compare {want}")
            continue
        for sl in DEP.SLICES:
            if s[sl] is not None:
                out.setdefault((SPLIT, sl), {})[sd] = s[sl]
    return out, bad


def rwt_reads(base=ROOT):
    """({form: {read: seed-0 R@5}} of RWT's fits from their compares and musique's read files, {form: {read: the
    compare's base R@5}}, {form: problems})."""
    new, against, bad = {}, {}, {f: [] for f in FORMS}
    for cname, (nm, bse) in COMPARES.items():
        f = FORM_OF[nm]
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, nm, bse):
            bad[f].append(f"{cname} is not RWT's compare of {nm} against {bse}")
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            bad[f].append(f"{cname} is not {CAND} on s1eval against one base")
            continue
        d, a = new.setdefault(f, {}), against.setdefault(f, {})
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            d[(SPLIT, r["dataset"])] = r["new"][0]
            a[(SPLIT, r["dataset"])] = r["base"][0]
        s = DEP.slice_r5(nm, base)
        want = d.get((SPLIT, DEP.SLICE_DS))
        if s is None or want is None or abs(s["all"] - want) > DEP.R5_TOL:
            bad[f].append(f"{nm}: musique's read file gives {None if s is None else round(s['all'], 5)}, its compare "
                          f"{want}")
            continue
        for sl in DEP.SLICES:
            if s[sl] is not None:
                d[(SPLIT, sl)] = s[sl]
    return new, against, bad


def decide(new, base):
    """Section 4 for one form: LTG's per-read screen call on the six reads (a LOSS: NOT_ADOPTED; else a GAIN:
    ADOPT_FOR_FULL_RUN; else NO_GAIN), the two zero-shot reads beside."""
    rows, problems = [], []
    for rd in DECIDING:
        b = base.get(rd, {})
        if any(k not in b for k in RULE.SEEDS) or rd not in new:
            problems.append(f"{rd[0]} {rd[1]}: zrckpn's seeds {sorted(b)}, the form's read {'present' if rd in new else 'missing'}")
            continue
        rows.append(LTG.screen_row(rd, b, {0: new[rd]}))
    side = [LTG.screen_row(rd, base[rd], {0: new[rd]}) for rd in BESIDE
            if all(k in base.get(rd, {}) for k in RULE.SEEDS) and rd in new]
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": rows, "beside": side, "counts": {},
                "mean_delta": None}
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    v = NOT_ADOPTED if n[RULE.LOSS] else ADOPT_FOR_FULL_RUN if n[RULE.GAIN] else NO_GAIN
    return {"verdict": v, "problems": [], "reads": rows, "beside": side, "counts": n,
            "mean_delta": round(sum(r["delta"] for r in rows) / len(rows), 5)}


def first_full_run(forms):
    """Section 6: of the forms ADOPT_FOR_FULL_RUN, the one with the larger mean delta over the six reads (a tie: F)."""
    passed = [f for f in FORMS if (forms.get(f) or {}).get("verdict") == ADOPT_FOR_FULL_RUN]
    return max(passed, key=lambda f: (forms[f]["mean_delta"], f == "f")) if passed else None


def fit_problems(base=ROOT):
    """{form: why its read fit is not the declared one}: its stamps (form, LTM's fit it is against, its loss's counts,
    PKO's present gate, LTM's fit's device, threads, pick and rows cap), its columns (LTM's fit's own records and scorer,
    the split's present scorer, the records unchanged since) and its trainer's code against LTM's fit's (every file
    LTM's names, and zrwt)."""
    out = {f: [] for f in FORMS}
    gp = Path(base) / M1.GATE
    g = M1.WQ.sha_file(gp) if gp.is_file() else None
    lo_dir = Path(base) / ZL.OUT
    fr = lo_dir / f"fit_{SPLIT}.json"
    fsha = M1.WQ.sha_file(fr) if fr.is_file() else None
    if not any(is_read(n, base) for n in FITS):
        return out
    try:
        ls, lr = fit_block(AGAINST, "screen", base), fit_block(AGAINST, "read", base)
    except SystemExit as e:
        return {f: [str(e)] for f in FORMS}
    for name, f in FORM_OF.items():
        if not is_read(name, base):
            continue
        bad = out[f]
        try:
            ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
        except SystemExit as e:
            bad.append(str(e))
            continue
        if g is None or ks.get("gate") != g or kr.get("gate") != g:
            bad.append(f"{name} names a gate that is not PKO's present gate")
        for what, k, lk in (("train", ks, ls), ("read", kr, lr)):
            got = tuple(json.dumps(k.get(x), sort_keys=True) for x in ("device", "threads", "pick", "rows_cap"))
            want = tuple(json.dumps(lk.get(x), sort_keys=True) for x in ("device", "threads", "pick", "rows_cap"))
            if got != want or k.get("threads") != THREADS or k.get("pick") != list(ZL.PICK):
                bad.append(f"{name}: its {what} ran with {got}, {AGAINST}'s with {want}")
            if (k.get("form"), k.get("against")) != (f, AGAINST):
                bad.append(f"{name}: its {what} names the form {k.get('form')} against {k.get('against')}")
            nk, nl = k.get("next") or {}, lk.get("next") or {}
            if not nk.get("fit_sha256") or nk.get("fit_sha256") != nl.get("fit_sha256") or nk.get("fit_sha256") != fsha:
                bad.append(f"{name}: its {what} read the scorer {str(nk.get('fit_sha256'))[:12]}, {AGAINST}'s "
                           f"{str(nl.get('fit_sha256'))[:12]}, the split's present one {str(fsha)[:12]}")
            if not nk.get("carves") or (nk.get("carves") or {}) != (nl.get("carves") or {}):
                bad.append(f"{name}: its {what} read other column records than {AGAINST}'s")
            for key, sha in (nk.get("carves") or {}).items():
                ds, cv = key.split("/")
                p = ZL.LTMC.cols_dir(lo_dir, SPLIT, ds, cv) / "record.json"
                if not p.is_file() or M1.WQ.sha_file(p) != sha:
                    bad.append(f"{name}: the columns of {key} are not the ones it read")
        lo = ks.get("loss") or {}
        if lo.get("form") != f or not lo.get("calls") or not lo.get("weighted"):
            bad.append(f"{name}: its trainer's loss counts are {lo}")
        cb, cn = ls.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if cn.get(k) != cb[k])
        if diff or "zrwt_sha256" not in cn:
            bad.append(f"{name}: code differs from {AGAINST}'s in {diff} (zrwt stamped: {'zrwt_sha256' in cn})")
    return out


def beside(base=ROOT):
    """Deciding nothing: per read RWT fit, its seconds, its training curve (the weighted loss per epoch, per variant) and
    its loss's counts."""
    out = {}
    for name in FITS:
        if not is_read(name, base):
            continue
        fd = M1.screen_dir(base) / "fits" / name
        tm = M1.WQ.read_json(fd / "ml1_timing.json") if (fd / "ml1_timing.json").is_file() else {}
        tj = M1.WQ.read_json(fd / "train.json") if (fd / "train.json").is_file() else {}
        try:
            lo = fit_block(name, "screen", base).get("loss")
        except SystemExit:
            lo = None
        out[name] = {"train_seconds": tm.get("train_seconds"), "read_seconds": tm.get("read_seconds"), "loss": lo,
                     "curve": {v: [round(e["loss"], 4) for e in vr.get("curve") or []]
                               for v, vr in (tj.get("variants") or {}).items()}}
    return out


def calls(base=ROOT):
    bse, shared = ltm_base(base)
    new, against, own = rwt_reads(base)
    fp = fit_problems(base)
    forms = {}
    for f in FORMS:
        nm = fit_name(f)
        if f not in new and not own[f]:
            forms[f] = {"fit": nm, "verdict": None, "waiting": f"{compare_name(f)} is not filed"}
            continue
        probs = shared + own[f] + fp[f]
        for rd, v in sorted((against.get(f) or {}).items()):
            b0 = bse.get(rd, {}).get(0)
            if b0 is None or abs(b0 - v) > RULE.EPS:
                probs.append(f"{compare_name(f)}: its base gives {rd[1]} {v}, LTM's zrckpn at seed 0 {b0}")
        blk = decide(new.get(f) or {}, bse)
        if probs and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
        blk["problems"] = blk["problems"] + probs
        forms[f] = {"fit": nm, "form": FORMS[f], **blk}
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "split": SPLIT, "seed": SEED, "against": AGAINST,
           "forms": forms, "first_full_run": first_full_run(forms),
           "base_reads": {f"{rd[0]} {rd[1]}": {str(k): v for k, v in sorted(s.items())} for rd, s in sorted(bse.items())},
           "beside": beside(base), "zrwt_sha256": M1.WQ.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# RWT: question weights in {ARM}'s loss, seed 0 on {SPLIT}, against LTM's {AGAINST}", ""]
    for f, blk in forms.items():
        md += [f"## {FORMS[f]}: {blk.get('verdict')}", ""]
        if blk.get("verdict") is None:
            md += [blk["waiting"], ""]
            continue
        md += [f"counts: {json.dumps(blk.get('counts', {}))}; mean delta over the six reads: {blk.get('mean_delta')}", ""]
        rows = blk["reads"] + [dict(r, call=f"{r['call']} (beside)") for r in blk.get("beside", [])]
        if rows:
            md += [f"| read | zrckpn s0, s1, s2 | {f.upper()} s0 | delta | floor | call |", "|---|---|---:|---:|---:|---|"]
            md += [f"| {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | {r['delta']:+.4f} | "
                   f"{r['floor']:.4f} | {r['call']} |" for r in rows]
        if blk["problems"]:
            md += ["", "problems:"] + [f"- {p}" for p in blk["problems"]]
        md.append("")
    md += [f"first full run (section 6): {rec['first_full_run']}"]
    if rec["beside"]:
        md += ["", "| fit | train s | read s | steps | weighted | F flat | w min | w max | curve |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for n, b in rec["beside"].items():
            lo = b["loss"] or {}
            md.append(f"| {n} | {b['train_seconds']} | {b['read_seconds']} | {lo.get('calls')} | {lo.get('weighted')} | "
                      f"{lo.get('flat')} | {lo.get('w_min')} | {lo.get('w_max')} | {json.dumps(b['curve'])} |")
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("; ".join(f"{f}: {blk.get('verdict')} {json.dumps(blk.get('counts', {}))}" for f, blk in forms.items())
        + f"; first full run {rec['first_full_run']}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def seg_case(rng, counts, gold_p):
    """Made scores, gold, nq and B of questions with counts rows each."""
    n = np.asarray(counts, np.int64)
    nq = torch.from_numpy(np.repeat(np.arange(n.size), n))
    s = torch.from_numpy(rng.standard_normal(int(n.sum())).astype(np.float32))
    gold = torch.from_numpy(rng.random(int(n.sum())) < gold_p)
    return s, gold, nq, int(n.size)


def by_hand(s, gold, nq, B, form):
    """(the form's loss in float64 from the arrays, its weights)."""
    s, g, q = s.detach().double().numpy(), gold.numpy().astype(np.float64), nq.numpy()
    ts, ns, Ps = [], [], []
    for i in range(B):
        m = q == i
        if g[m].sum() == 0:
            continue
        x = s[m] - s[m].max()
        lp = x - np.log(np.exp(x).sum())
        ts.append(-(lp * g[m]).sum() / g[m].sum())
        ns.append(g[m].sum())
        Ps.append((np.exp(lp) * g[m]).sum())
    t, n5, h = np.asarray(ts), np.minimum(np.asarray(ns), CAP), 1.0 - np.clip(np.asarray(Ps), 0.0, 1.0)
    w = {"unit": np.ones_like(t), "g": n5 / n5.mean(), "f": h / h.mean() if h.mean() >= FLAT else np.ones_like(t)}[form]
    return float((w * t).mean()), w


def selftest():
    must_stop = ZL.must_stop
    # 1. the names
    assert list(FITS) == ["scr-rwt-zrckpnf-s0-hp", "scr-rwt-zrckpng-s0-hp"]
    assert list(COMPARES) == ["scr-rwt-zrckpnf-vs-zrckpn-s0-hp", "scr-rwt-zrckpng-vs-zrckpn-s0-hp"]
    assert set(FITS.values()) == {("zrckpn", 0, "WQ-hotpotqa")} and set(FORM_OF.values()) == set(FORMS)
    assert RWT_CODE[:-1] == ZL.LTM_CODE and RWT_CODE[-1] == "zrwt" and (HERE / "zrwt.py").is_file()
    # 2. unit is lean_gpu's loss bit for bit, loss and gradient: with a question all gold, one without a gold row, a
    #    step without any gold row and a step of one question
    rng = np.random.default_rng(7)
    for counts, gp in (([3, 5, 1, 7, 4, 9, 2], 0.3), ([4, 4, 4], 0.0), ([2, 6, 3, 8], 0.5), ([5], 1.0)):
        s, gold, nq, B = seg_case(rng, counts, gp)
        if len(counts) == 7:
            gold[:3], gold[3:8] = True, False
        a, b = s.clone().requires_grad_(True), s.clone().requires_grad_(True)
        la, lb = ORIG_LOSS(a, gold, nq, B), listwise_w(b, gold, nq, B, "unit")
        la.backward()
        lb.backward()
        assert torch.equal(la, lb) and torch.equal(a.grad, b.grad), counts
    # 3. G by hand: min(n, 5) over its mean among the step's questions with a gold row; the cap; equal counts are
    #    unit's bits
    s, gold, nq, B = seg_case(rng, [9, 3, 7, 2, 8, 5], 0.6)
    gold[:8] = True                                       # the first question: 8 gold rows of 9
    lg = listwise_w(s, gold, nq, B, "g")
    want, w = by_hand(s, gold, nq, B, "g")
    assert abs(float(lg) - want) <= 1e-5 * max(1.0, abs(want)) and abs(w.mean() - 1.0) < 1e-12
    wg, flat = weights(torch.tensor([8.0, 5.0, 2.0, 1.0]), None, "g")
    assert not flat and float(wg[0]) == float(wg[1]) and abs(float(wg.mean()) - 1.0) < 1e-6
    assert abs(float(wg[0]) / float(wg[2]) - 2.5) < 1e-6
    s1, g1, nq1, B1 = seg_case(rng, [4, 6, 3], 0.0)
    g1[[1, 5, 11]] = True                                 # one gold row each
    assert torch.equal(listwise_w(s1, g1, nq1, B1, "g"), ORIG_LOSS(s1, g1, nq1, B1))
    # 4. F by hand; its weights fall as a question's gold mass rises; its flat step is unit's bits; no gradient passes
    #    through its weights
    s, gold, nq, B = seg_case(rng, [6, 4, 9, 3, 7], 0.35)
    gold[[0, 6, 10, 19, 22]] = True                       # every question has a gold row
    x = s.clone().requires_grad_(True)
    lf = listwise_w(x, gold, nq, B, "f")
    lf.backward()
    want, w = by_hand(s, gold, nq, B, "f")
    assert abs(float(lf.detach()) - want) <= 1e-5 * max(1.0, abs(want)) and abs(w.mean() - 1.0) < 1e-12 and w.std() > 0.05
    y = s.clone().requires_grad_(True)
    lp = LG.seg_log_softmaxD(y, nq, B)
    g = gold.to(torch.float32)
    ng = torch.zeros(B).index_add_(0, nq, g)
    pq = -torch.zeros(B).index_add_(0, nq, lp * g)
    has = ng > 0
    (torch.from_numpy(w.astype(np.float32)) * (pq[has] / ng[has])).mean().backward()
    assert torch.allclose(x.grad, y.grad, atol=1e-6, rtol=1e-5), (x.grad - y.grad).abs().max()
    wf, flat = weights(torch.ones(3), torch.tensor([0.9, 0.5, 0.0]), "f")
    assert not flat and float(wf[0]) < float(wf[1]) < float(wf[2]) and abs(float(wf.mean()) - 1.0) < 1e-6
    wf, flat = weights(torch.ones(2), torch.tensor([1.0, 1.0]), "f")
    assert flat and torch.equal(wf, torch.ones(2))
    s2, g2, nq2, B2 = seg_case(rng, [3, 5], 1.0)
    assert torch.equal(listwise_w(s2, g2, nq2, B2, "f"), ORIG_LOSS(s2, g2, nq2, B2))
    must_stop(weights, torch.ones(2), None, "h")
    # 5. the swap: inside, lean_gpu's loss is the form's and counted; after, lean_gpu's own, also on a failure; no swap
    #    inside another; no unknown form
    with on_loss("g") as st:
        assert LG.listwiseD is not ORIG_LOSS
        assert torch.equal(LG.listwiseD(s, gold, nq, B), listwise_w(s, gold, nq, B, "g")) and st["calls"] == 2
        must_stop(on_loss("f").__enter__)
        assert st["form"] == "g"
    assert LG.listwiseD is ORIG_LOSS
    try:
        with on_loss("f"):
            raise KeyError("x")
    except KeyError:
        pass
    assert LG.listwiseD is ORIG_LOSS
    must_stop(on_loss("h").__enter__)
    # 6. a toy fit through lean_gpu's loop: under unit the trainer's own states and curve bit for bit; F and G move them,
    #    weighting its steps
    LG.bind_device_ops()
    import rmatch as RM
    tmp = Path(tempfile.mkdtemp(prefix="zrwt_"))
    try:
        RM.toy_roots(tmp, np.random.default_rng(31), chunks=(12, 12))
        c = RM.ToyBase("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        ref = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy")
        with on_loss("unit") as st:
            u = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy unit")
        assert st["calls"] == sum(e["steps"] for e in u["curve"]) and st["weighted"] == 0
        assert not LG.same_state(ref["swa"], u["swa"]) and len(ref["states"]) == len(u["states"]) == 3
        assert all(not LG.same_state(p, q) for p, q in zip(ref["states"], u["states"]))
        assert [e["loss"] for e in ref["curve"]] == [e["loss"] for e in u["curve"]]
        for f in FORMS:
            with on_loss(f) as st:
                v = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {f}")
            assert st["calls"] == sum(e["steps"] for e in v["curve"]) and st["weighted"] > 0, (f, dict(st))
            assert LG.same_state(ref["swa"], v["swa"]), f"{f} did not move the fit"
        assert LG.listwiseD is ORIG_LOSS
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the patches: zltm and zml1s under RWT's names, this file's train and stamp; all restored, also on a failure
    zl_k = ("FITS", "DECLARED", "STAGE", "KEY", "LTM_CODE", "train")
    m1_k = ("DECLARED", "STAGE", "KEY", "pko_fit_name", "stamp", "code_shas", "train", "read")
    zl0, m10, n0 = {k: getattr(ZL, k) for k in zl_k}, {k: getattr(M1, k) for k in m1_k}, len(M1.ALL_FITS)
    with on_fit():
        assert ZL.FITS is FITS and (ZL.DECLARED, ZL.STAGE, ZL.KEY, ZL.LTM_CODE) == (DECLARED, STAGE, KEY, RWT_CODE)
        assert ZL.train is train and M1.stamp is not m10["stamp"] and M1.stamp.__qualname__.startswith("rwt_stamp")
    with on_names():
        assert (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY) and all(M1.ALL_FITS[n] == FITS[n] for n in FITS)
        cs = M1.code_shas(ARM)
        assert {"zrwt_sha256", "zltm_sha256", "zml1s_sha256", "zp1c3p_sha256"} <= set(cs)
        M1._ACTIVE[0] = fit_name("g")
        try:
            assert M1.pko_fit_name(ARM, SEED, SPLIT) == fit_name("g")
        finally:
            M1._ACTIVE[0] = None
    try:
        with on_fit(), on_names():
            raise KeyError("x")
    except KeyError:
        pass
    assert {k: getattr(ZL, k) for k in zl_k} == zl0 and {k: getattr(M1, k) for k in m1_k} == m10
    assert len(M1.ALL_FITS) == n0 and not set(FITS) & set(M1.ALL_FITS)
    must_stop(fit, AGAINST)
    must_stop(train, AGAINST, {})
    must_stop(fit_block, "scr-x", "screen")
    # 8. the stamp: the form and LTM's fit on both records, the naming and the loss's counts on the train's; a train
    #    whose form weighted no step, another form's counts or a name not RWT's stops unstamped
    got = []
    st = rwt_stamp(lambda *a: got.append(a))
    STATS.clear()
    STATS.update(form="g", calls=0, weighted=0, flat=0, w_max=0.0, w_min=None)
    must_stop(st, fit_name("g"), "env", "screen", {"what": "x"})
    STATS.update(calls=4, weighted=3, w_max=1.6, w_min=0.4)
    must_stop(st, fit_name("f"), "env", "screen", {})
    must_stop(st, "scr-x", "env", "read", {})
    assert not got
    st(fit_name("g"), "env", "screen", {"what": "x", "against": "y", "next": {"a": 1}})
    st(fit_name("g"), "env", "read", {"next": {"a": 1}})
    (n_s, _e, w_s, e_s), (n_r, _e2, w_r, e_r) = got
    assert (n_s, w_s, e_s["what"], e_s["against"], e_s["form"], e_s["next"], e_s["loss"]["weighted"]) == (
        fit_name("g"), "screen", NAMING["g"], AGAINST, "g", {"a": 1}, 3)
    assert (n_r, w_r, e_r) == (fit_name("g"), "read", {"next": {"a": 1}, "form": "g", "against": AGAINST})
    # 9. the compares' reading, on made records under a temporary root: RWT's stamp, the candidate, the slices' file
    tmp = Path(tempfile.mkdtemp(prefix="zrwt_"))
    try:
        sd = M1.screen_dir(tmp)
        sd.mkdir(parents=True)
        rows = [{"dataset": d, "new": [0.5 + 0.01 * i], "base": [0.5]} for i, d in enumerate(M1.WQ.EVAL_ORDER)]
        for i, (cname, (nm, bse)) in enumerate(COMPARES.items()):
            M1.WQ.write_json(sd / f"{cname}.json", {"candidate": CAND, "carve": "s1eval", "bases": ["b"],
                                                    "by_base": {"b": {"rows": rows}},
                                                    KEY: {"declared_in": DECLARED, "new": nm if i == 0 else "x",
                                                          "base": bse}})
        new, against, bad = rwt_reads(tmp)
        mi = M1.WQ.EVAL_ORDER.index("musique")
        assert set(new) == {"f"} and new["f"][(SPLIT, "musique")] == 0.5 + 0.01 * mi and len(new["f"]) == 6
        assert set(against["f"].values()) == {0.5} and len(bad["f"]) == 1 and "read file" in bad["f"][0]
        assert len(bad["g"]) == 1 and "not RWT's compare" in bad["g"][0]
        assert fit_problems(tmp) == {f: [] for f in FORMS} and beside(tmp) == {}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 10. the rule on made numbers: the six reads decide, a delta at the floor counts, the zero-shot reads do not decide,
    #     a missing read or seed is INCOMPLETE; the first full run by mean delta (a tie: F)
    flat3 = {rd: {0: 0.5, 1: 0.5, 2: 0.5} for rd in DECIDING + BESIDE}
    same = {rd: 0.5 for rd in DECIDING + BESIDE}
    assert decide(same, flat3)["verdict"] == NO_GAIN and decide(same, flat3)["counts"][RULE.WITHIN] == 6
    up = {**same, (SPLIT, "musique-4hop"): 0.51}
    assert decide(up, flat3)["verdict"] == ADOPT_FOR_FULL_RUN
    assert decide({**same, (SPLIT, "musique"): 0.5075}, flat3)["verdict"] == ADOPT_FOR_FULL_RUN
    assert decide({**same, (SPLIT, "musique"): 0.5074}, flat3)["verdict"] == NO_GAIN
    assert decide({**up, (SPLIT, "squad"): 0.49}, flat3)["verdict"] == NOT_ADOPTED
    side = decide({**up, (SPLIT, "hotpotqa"): 0.40}, flat3)
    assert side["verdict"] == ADOPT_FOR_FULL_RUN and [r["call"] for r in side["beside"]].count(RULE.LOSS) == 1
    assert decide({k: v for k, v in same.items() if k != (SPLIT, "webqsp")}, flat3)["verdict"] == INCOMPLETE
    short = {**flat3, (SPLIT, "2wiki"): {0: 0.5, 1: 0.5}}
    assert decide(same, short)["verdict"] == INCOMPLETE
    wide = {**flat3, (SPLIT, "squad"): {0: 0.50, 1: 0.51, 2: 0.49}}     # zml1's spread 0.01: the floor 0.02
    assert decide({**same, (SPLIT, "squad"): 0.515}, wide)["verdict"] == NO_GAIN
    assert decide({**same, (SPLIT, "squad"): 0.525}, wide)["verdict"] == ADOPT_FOR_FULL_RUN
    fa = {"f": {"verdict": ADOPT_FOR_FULL_RUN, "mean_delta": 0.002}, "g": {"verdict": ADOPT_FOR_FULL_RUN, "mean_delta": 0.003}}
    assert first_full_run(fa) == "g" and first_full_run({**fa, "g": {**fa["g"], "mean_delta": 0.002}}) == "f"
    assert first_full_run({**fa, "f": {"verdict": NO_GAIN}}) == "g" and first_full_run({"f": {}, "g": {}}) is None
    log("selftest: names; unit is lean_gpu's loss bit for bit (loss, gradient); G and F by hand, G's cap, F's order and "
        "flat step, no gradient through the weights; the swap and its restoring (also on a failure, never nested); a toy "
        "fit (unit bit for bit, F and G move it); the patches and their restoring; the stamp; the compares' reading; "
        "the rule and the first full run")
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
