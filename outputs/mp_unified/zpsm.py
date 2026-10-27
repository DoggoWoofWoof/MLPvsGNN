"""PSM (docs/PSM_POOL_SOFTMAX_INPUT.md): a pool-softmax input for the MLP, under WQ (docs/WQ_WEBQSP_TRAINS.md, section
9, step 1). zrckp (zp1c3p.py, imported unchanged) reads, per block, the row's raw columns, their z-scores over the
question's pool against its retrieved rows (zret) and the block's keep flag. zrckpg adds, for every fixed block (every
block but SEMB), each column's gap below the pool's soft best: the log-softmax of zret's z over the question's rows at
temperature 1, floored at -10, times the keep flag. The gaps enter the first layer through W_g (hidden x columns, zero
at the start, no bias, drawn from no generator), so at the start zrckpg is zrckp bit for bit.

The input aggregates over the question's rows, but only over the cache's fixed columns, as zret's z-scores do; no
learned value and no score of any row enters it, and it is computed before any row is scored. SEMB (a learned product)
and the KT columns (already positions within the pool's orders) take no gap.

One seed-0 fit per WQ split on WQ's card and roots, read on the six s1eval carves at p@swa, compared with WQ's card fit
of the split (scr-wq-zrckp-s0, -hp) alone, and called under WQ's floors on zrckp (outputs/wq/floors-zrckp.json).

    python outputs/mp_unified/zpsm.py train --split WQ-musique --name scr-psm-zrckpg --device cuda --host \\
        --cache-root outputs/wq/p1c2/cache
    python outputs/mp_unified/zpsm.py read --name scr-psm-zrckpg --device cuda --host --cache-root outputs/wq/p1c2/cache
    python outputs/mp_unified/zpsm.py compare --split WQ-musique
    python outputs/mp_unified/zpsm.py verdict                              (outputs/psm/screen-verdict.json, .md)
    python outputs/mp_unified/zpsm.py --selftest

train and read run only once WQ's summary is BASE_READY and its gate PASS. verdict exits 0 on ADOPT_FOR_FULL_RUN or
NOT_ADOPTED and 2 after writing an INCOMPLETE record.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zwq as WQ  # noqa: E402  (WQ's splits, roots, gate, summary, floors and on_wq)
import zp1c3p as Z3P  # noqa: E402  (zrckp; through it zp1c3, zp1c2, zrm, rmatch, lean_screen3 and lean_gpu)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z3 = Z3P.Z3
Z2, ZL, ZM, RM = Z3.Z2, Z3.ZL, Z3.ZM, Z3.RM
S, LG, LC, SR = Z3.S, Z3.LG, Z3.LC, Z3.SR
S3 = ZM.S3
log = S.log

DECLARED = "docs/PSM_POOL_SOFTMAX_INPUT.md"
STAGE = "PSM"
ARM, BASE = "zrckpg", "zrckp"
SPLITS = tuple(WQ.SPLITS)
FIT = {sp: f"scr-psm-{ARM}{WQ.SUF[sp]}" for sp in SPLITS}
SPLIT_OF = {n: sp for sp, n in FIT.items()}
BASE_FIT = {sp: WQ.fit_name(BASE, 0, sp) for sp in SPLITS}
VERDICT = "outputs/psm/screen-verdict"
TEMP, GAP_FLOOR = 1.0, -10.0
NO_GAP = ("SEMB",)
GAP = {"blocks": "every block of the fit but SEMB", "of": "zret's z (lean_screen3.seg_zscore_ref)",
       "temperature": TEMP, "floor": GAP_FLOOR, "times_keep": True, "gradient": False,
       "weights": "W_g, hidden x gap columns, zero at the start, no bias, drawn from no generator"}
NEXT = {"ADOPT_FOR_FULL_RUN": "zspkg: the same input on the GNN (zspk), then the pair's full run under WQ",
        "GAIN_AND_LOSS": "zrckpu: the gap without the pool-size term, log(N_q p_i) floored at 0",
        "NO_GAIN": "zrckph: the hard gap, z_i minus the pool's largest z, floored at -10",
        "INCOMPLETE": "the missing piece reruns; nothing is decided"}
RULE = ("ADOPT_FOR_FULL_RUN when, under WQ's floors on zrckp, no read of the twelve is LOSS and at least one is GAIN; "
        "NOT_ADOPTED otherwise; INCOMPLETE when a comparison or the floors are missing, not WQ's or PSM's, or not twelve "
        "reads (docs/PSM_POOL_SOFTMAX_INPUT.md, section 5)")
assert len(set(FIT.values())) == len(FIT) == 2 and all(b in WQ.FITS for b in BASE_FIT.values())


# ── the input: each fixed column's gap below the pool's soft best ────────────


@torch.no_grad()
def seg_log_softmax_cols(x, nq, B):
    """Each column's log-softmax over its segment's rows: lean_gpu.seg_log_softmaxD (the op every fit's loss runs) over
    the (segment, column) pairs."""
    N, W = x.shape
    seg = (nq.unsqueeze(1) * W + torch.arange(W, dtype=nq.dtype, device=x.device)).reshape(-1)
    return LG.seg_log_softmaxD(x.reshape(-1), seg, B * W).reshape(N, W)


@torch.no_grad()
def pool_gap(z, nq, B):
    """The gap below the pool's soft best: the log-softmax of z / TEMP over the question's rows, floored at GAP_FLOOR."""
    return seg_log_softmax_cols(z / TEMP, nq, B).clamp_min(GAP_FLOOR)


class GapRet(S3.ZRet):
    """zret's forward with each fixed block's pool gaps (pool_gap of its z, times its keep flag) added to the first
    layer's pre-activation through gap_w (zero at the start, no bias, drawn from no generator)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.gap_blocks = [b for b in self.blocks if b not in NO_GAP]
        self.gap_cols = sum(self.widths[b] for b in self.gap_blocks)
        if not self.gap_cols:
            raise SystemExit(f"{ARM} needs a fixed block")
        self.gap_w = nn.Parameter(torch.zeros(self.l1.out_features, self.gap_cols))

    @torch.no_grad()
    def gap_of(self, feats, b, nq, B, ref, m):
        raw = S.raw_of(self, feats, b, nq)
        if raw is not feats[b]:
            raise SystemExit(f"{ARM}: block {b}'s columns are not the batch's own; a learned block takes no pool gap")
        return pool_gap(S3.seg_zscore_ref(raw, nq, B, ref), nq, B) * m

    def pool_gaps(self, feats, keep, nq, B, ref=None):
        """(N, gap_cols): the fixed blocks' gaps in the model's block order, each times its keep flag."""
        ref = S3.retrieved(feats) if ref is None else ref
        return torch.cat([self.gap_of(feats, b, nq, B, ref, keep[nq, j].unsqueeze(1))
                          for j, b in enumerate(self.blocks) if b not in NO_GAP], 1)

    def forward(self, feats, keep, nq, B, base_z):
        ref = S3.retrieved(feats)
        parts = []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
        x = self.l1(torch.cat(parts, 1)) + Fn.linear(self.pool_gaps(feats, keep, nq, B, ref), self.gap_w)
        h = self.drop(Fn.gelu(x))
        h = self.drop(Fn.gelu(self.l2(h)))
        bz = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
        return self.base_w * bz + self.out(h).squeeze(-1)


class ZRCKPG(Z3P.ZRCKP, GapRet):
    """zrckpg: zrckp (zrc's model and the pointwise KT head) over zret with the pool gaps."""


ARMS = {ARM: (ZRCKPG, Z3.KTChainCarveZRC)}
S.ARMS.update(ARMS)


# ── guards ───────────────────────────────────────────────────────────────────


def base_ready(base=ROOT):
    p = Path(base) / (WQ.SUMMARY + ".json")
    if not p.is_file():
        raise SystemExit(f"zpsm: {WQ.SUMMARY}.json missing (zwq.py summary); PSM runs after WQ's BASE_READY")
    rec = WQ.read_json(p)
    if rec.get("status") != "BASE_READY" or rec.get("declared_in") != WQ.DECLARED:
        raise SystemExit(f"zpsm: {WQ.SUMMARY}.json is {rec.get('status')}, not WQ's BASE_READY")
    return rec


def psm_why(name, base=ROOT):
    """Why the fit record of name is not PSM's fit of its split, or None."""
    split = SPLIT_OF.get(name)
    if split is None:
        return f"{name} is not a PSM fit ({', '.join(FIT.values())})"
    sj = WQ.fit_dir(name, base) / "screen.json"
    if not sj.is_file():
        return f"{name} has no record (outputs/screen/fits/{name}/screen.json)"
    s = WQ.read_json(sj)
    p, w = s.get("psm") or {}, s.get("wq") or {}
    got = (s.get("arm"), s.get("seed"), s.get("split"), p.get("declared_in"), p.get("arm"), p.get("split"),
           w.get("declared_in"), w.get("split"))
    if got != (ARM, 0, split, DECLARED, ARM, split, WQ.DECLARED, split) or "p1c3p" not in s:
        return f"{name}'s record is {got}, p1c3p stamp {'p1c3p' in s}; not PSM's {ARM} seed 0 on {split}"
    return None


@contextlib.contextmanager
def on_arm(W):
    """zp1c3p's train and read for zrckpg: its arm name and arm table swapped for the block, restored after."""
    with WQ.patch(W.Z3P, ARM=ARM, ARMS=dict(ARMS)):
        yield W


# ── train, read, compare ─────────────────────────────────────────────────────


def gap_columns(name):
    """{block: width} of the fit's gap blocks, from its train.json (None before it exists)."""
    tj = WQ.fit_dir(name) / "train.json"
    if not tj.is_file():
        return None
    w = (WQ.read_json(tj).get("variants") or {}).get("p", {}).get("widths") or {}
    return {b: int(v) for b, v in w.items() if b not in NO_GAP}


def train(argv):
    """PSM's fit of a WQ split: zp1c3p's train for zrckpg on WQ's card and roots, then the record's wq and psm blocks."""
    ap = argparse.ArgumentParser(prog="zpsm.py train")
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--cache-root", required=True)
    a = ap.parse_args(argv)
    if a.name != FIT[a.split]:
        raise SystemExit(f"zpsm: PSM's fit of {a.split} is {FIT[a.split]}, not {a.name}")
    WQ.cache_root_ok(a.cache_root)
    summ = base_ready()
    WQ.gate_passed()
    why = WQ.wq_why(BASE_FIT[a.split])
    if why:
        raise SystemExit(f"zpsm: {why}")
    args = ["train", "--name", a.name, "--arm", ARM, "--seed", "0", "--device", a.device,
            "--cache-root", WQ.ROOTS["cache"]] + (["--host"] if a.host else [])
    with WQ.on_wq() as W, on_arm(W):
        with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, a.split: a.name}):
            rc = W.Z3P.train(args, a.split)
        if rc == 0:
            sj = WQ.fit_dir(a.name) / "screen.json"
            rec = WQ.read_json(sj)
            cols = gap_columns(a.name)
            rec["zwq_sha256"], rec["zpsm_sha256"] = WQ.sha_src(WQ.__file__), WQ.sha_src(__file__)
            rec["wq"] = {"declared_in": WQ.DECLARED, "split": a.split, "seed": 0, "arm": ARM,
                         "trains_on": list(WQ.SPLITS[a.split]), "zero_shot": list(WQ.ZERO_SHOT[a.split]),
                         "roots": dict(WQ.ROOTS), "gate": WQ.sha_file(ROOT / WQ.GATE),
                         "overlay": WQ.sha_file(ROOT / WQ.OVERLAY)}
            rec["psm"] = {"declared_in": DECLARED, "arm": ARM, "base_arm": BASE, "split": a.split, "seed": 0,
                          "against": BASE_FIT[a.split], "input": GAP, "gap_columns": cols,
                          "gap_weights": None if cols is None else sum(cols.values()) * int(rec.get("hidden") or 0),
                          "summary": WQ.sha_file(ROOT / (WQ.SUMMARY + ".json")), "summary_utc": summ.get("utc")}
            W.LC.write_json(sj, rec)
            log(f"zpsm train {a.name}: done, stamped ({sum((cols or {}).values())} gap columns)")
    return rc


def read(argv):
    """A PSM fit's read of the six s1eval carves (zp1c3p's read for zrckpg on WQ's roots)."""
    ap = argparse.ArgumentParser(prog="zpsm.py read")
    ap.add_argument("--name", required=True, choices=sorted(SPLIT_OF))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--cache-root", required=True)
    a = ap.parse_args(argv)
    WQ.cache_root_ok(a.cache_root)
    WQ.gate_passed()
    why = psm_why(a.name)
    if why:
        raise SystemExit(f"zpsm: {why}")
    args = ["--name", a.name, "--device", a.device, "--cache-root", WQ.ROOTS["cache"]] + (["--host"] if a.host else [])
    with WQ.on_wq() as W, on_arm(W):
        return W.Z3P.read(args)


def compare(split, base=ROOT):
    """lean_screen's compare of PSM's fit of split with WQ's card fit of the split alone, stamped PSM's."""
    name, b0 = FIT[split], BASE_FIT[split]
    for why in (psm_why(name, base), WQ.wq_why(b0, base)):
        if why:
            raise SystemExit(f"zpsm: {why}")
    six = {f"{d}=s1eval" for d in WQ.EVAL_ORDER}
    for f in (name, b0):
        rj = WQ.fit_dir(f, base) / "read.json"
        if not rj.is_file() or set(WQ.read_json(rj).get("carves") or {}) != six:
            raise SystemExit(f"zpsm: {f} is not read on the six s1eval carves")
    out = f"outputs/screen/{name}"
    rc = WQ.run_compare(f"outputs/screen/fits/{name}", f"outputs/screen/fits/{b0}", out)
    p = Path(base) / (out + ".json")
    if rc == 0 and p.is_file():
        rec = WQ.read_json(p)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "decided_against": b0, "split": split,
                    "zpsm_sha256": WQ.sha_src(__file__)})
        WQ.write_json(p, rec)
    return rc


# ── the verdict (section 5) ──────────────────────────────────────────────────


def compare_rows(split, base=ROOT):
    """The six rows of PSM's comparison of split, or SystemExit naming what is wrong with it."""
    name, b0 = FIT[split], BASE_FIT[split]
    p = Path(base) / "outputs" / "screen" / f"{name}.json"
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{name}.json missing")
    rec = WQ.read_json(p)
    bases = rec.get("bases") or []
    if rec.get("stage") != STAGE or rec.get("declared_in") != DECLARED or rec.get("split") != split:
        raise SystemExit(f"outputs/screen/{name}.json is not PSM's comparison of {split}")
    if len(bases) != 1 or WQ.parts_of(bases[0])[-3:] != ("screen", "fits", b0):
        raise SystemExit(f"outputs/screen/{name}.json is decided against {bases}, not {b0} alone")
    if WQ.parts_of(rec.get("new", ""))[-3:] != ("screen", "fits", name):
        raise SystemExit(f"outputs/screen/{name}.json reads {rec.get('new')}, not {name}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"outputs/screen/{name}.json reads {rec.get('candidate')} on {rec.get('carve')}")
    if sorted(rec.get("trained_on") or []) != sorted(WQ.SPLITS[split]):
        raise SystemExit(f"outputs/screen/{name}.json trained on {rec.get('trained_on')}, not {split}'s")
    for why in (psm_why(name, base), WQ.wq_why(b0, base)):
        if why:
            raise SystemExit(why)
    rows = rec["by_base"][bases[0]]["rows"]
    if sorted(r["dataset"] for r in rows) != sorted(WQ.EVAL_ORDER):
        raise SystemExit(f"outputs/screen/{name}.json reads {[r['dataset'] for r in rows]}, not the six")
    return rows


def verdict(base=ROOT, out=None):
    """PSM's verdict from its two comparisons, every read called again under WQ's floors on zrckp."""
    base = Path(base)
    problems, rows = [], []
    fp = base / (WQ.FLOORS[BASE] + ".json")
    fl = WQ.read_json(fp) if fp.is_file() else {}
    if fl.get("status") != "OK" or fl.get("declared_in") != WQ.DECLARED or fl.get("arm") != BASE:
        problems.append(f"{WQ.FLOORS[BASE]}.json is {fl.get('status')}, not WQ's OK floors on {BASE}")
    fmap = {(r["split"], r["dataset"]): r for r in fl.get("reads") or []}
    for sp in SPLITS:
        try:
            got = compare_rows(sp, base)
        except (SystemExit, KeyError, TypeError) as e:
            problems.append(WQ.clean(e, base))
            continue
        for r in got:
            f = fmap.get((sp, r["dataset"]))
            if f is None:
                problems.append(f"{sp} {r['dataset']}: no floor")
                continue
            if r["base"][0] != f["seed0"]:
                problems.append(f"{sp} {r['dataset']}: compared with seed-0 R@5 {r['base'][0]}, the floors with "
                                f"{f['seed0']}")
                continue
            rows.append({"split": sp, "dataset": r["dataset"], "read": r["read"], "base": r["base"], "new": r["new"],
                         "delta": r["delta"], "lo": r["lo"], "hi": r["hi"], "call_filed": r["call"],
                         "floor": f["floor"], "call": SR.recall(r["call"], r["delta"][0], f["floor"]),
                         "near_floor": SR.near(r["call"], r["delta"][0], f["floor"])})
    if not problems and len(rows) != len(SPLITS) * len(WQ.EVAL_ORDER):
        problems.append(f"{len(rows)} reads, not twelve")
    tag = [f"{r['split']} {r['dataset']}" for r in rows]
    gains = [t for t, r in zip(tag, rows) if r["call"] == "GAIN"]
    losses = [t for t, r in zip(tag, rows) if r["call"] == "LOSS"]
    if problems:
        v, nxt = "INCOMPLETE", NEXT["INCOMPLETE"]
    elif losses or not gains:
        v, nxt = "NOT_ADOPTED", NEXT["GAIN_AND_LOSS" if gains else "NO_GAIN"]
    else:
        v, nxt = "ADOPT_FOR_FULL_RUN", NEXT["ADOPT_FOR_FULL_RUN"]
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "next": nxt, "arm": ARM,
           "base_arm": BASE, "against": dict(BASE_FIT), "fits": dict(FIT), "floors": WQ.FLOORS[BASE] + ".json",
           "floors_sha256": WQ.sha_file(fp) if fp.is_file() else None, "problems": problems, "gains": gains,
           "losses": losses, "reads": rows, "zpsm_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# PSM's screen verdict: **{v}**", "", f"Rule: {RULE}.", "",
          f"{ARM} (zrckp with each fixed column's pool gap) against WQ's card fit of each split, every read called "
          f"under WQ's floors on {BASE}: {len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS "
          f"({', '.join(losses) or 'none'}).", "", f"Next: {nxt}."]
    if problems:
        md += ["", "Problems: " + "; ".join(problems)]
    if rows:
        md += ["", "FC@5 and hit@1 are beside, deciding nothing.", "",
               "| split | dataset | read | base R@5 | new R@5 | delta R@5 [95% CI] | filed | floor | call | delta FC@5 | "
               "delta hit@1 |", "|---|---|---|---:|---:|---|---|---:|---|---:|---:|"]
        for r in rows:
            d, lo, hi = r["delta"], r["lo"], r["hi"]
            md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['base'][0]:.4f} | {r['new'][0]:.4f} | "
                      f"{d[0]:+.4f} [{lo[0]:+.4f}, {hi[0]:+.4f}] | {r['call_filed']} | {r['floor']:.4f} | "
                      f"{'**' + r['call'] + '**' if r['call'] != 'WITHIN' else r['call']} | {d[1]:+.4f} | {d[2]:+.4f} |")
    WQ.write(Path(out) if out else base / VERDICT, rec, md)
    log(f"zpsm verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''})")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


class ToyGap(ZM.ToyZ):
    """zrm's toy carve (about 40% of its rows unretrieved) with question ids and a second fixed block, WALK: three
    columns, one spread, one constant over every pool (z 0, so its gap is -log n), one of ties."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.ids = [f"q{i}" for i in range(self.rows)]
        g = torch.Generator().manual_seed(11)
        w = torch.rand(int(self.off_np[-1]), 3, generator=g) * 4.0
        w[:, 1] = 0.5
        w[:, 2] = (w[:, 2] > 2.0).float()
        self.WK = w
        self.widths = {**self.widths, "WALK": 3}

    def batch(self, qs, blocks):
        feats, nq, bz, gold = super().batch(qs, [b for b in blocks if b != "WALK"])
        if "WALK" in blocks:
            qs = np.asarray(qs, np.int64)
            cnt = self.n_np[qs]
            idx = torch.from_numpy(np.repeat(self.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum())))
            feats["WALK"] = self.WK[idx]
        return feats, nq, bz, gold


def gap_numpy(z, nq, B, floor=GAP_FLOOR):
    """pool_gap by hand, in float64, for the selftest."""
    z, nq = np.asarray(z, np.float64), np.asarray(nq)
    out = np.zeros_like(z)
    for q in range(B):
        r = nq == q
        if r.any():
            zz = z[r] / TEMP
            mx = zz.max(0)
            out[r] = np.maximum(zz - (mx + np.log(np.exp(zz - mx).sum(0))), floor)
    return out


def fake_fit(base, name, arm, split, stamp):
    rec = {"arm": arm, "seed": 0, "split": split, "rc": 0, "p1c3p": {},
           "wq": {"declared_in": WQ.DECLARED, "arm": arm, "seed": 0, "split": split}}
    if stamp:
        rec["psm"] = {"declared_in": DECLARED, "arm": arm, "split": split, "seed": 0}
    WQ.write_json(WQ.fit_dir(name, base) / "screen.json", rec)


def fake_world(base, calls=None, floors=None, seed0=0.5, **over):
    """WQ's floors on zrckp, the four fit records and PSM's two comparisons, in base. calls: {(split, ds): (delta,
    filed call)}; floors: {(split, ds): floor}; over: {split: fields replaced in that comparison}."""
    calls, floors = calls or {}, floors or {}
    reads = [{"split": sp, "dataset": ds, "read": "x", "seed0": seed0, "d1": 0.0, "d2": 0.0, "spread": 0.0,
              "floor": floors.get((sp, ds), WQ.FLOOR), "raised": (sp, ds) in floors}
             for sp in SPLITS for ds in WQ.EVAL_ORDER]
    WQ.write_json(Path(base) / (WQ.FLOORS[BASE] + ".json"),
                  {"stage": WQ.STAGE, "declared_in": WQ.DECLARED, "arm": BASE, "status": "OK", "reads": reads})
    for sp in SPLITS:
        fake_fit(base, FIT[sp], ARM, sp, True)
        fake_fit(base, BASE_FIT[sp], BASE, sp, False)
        rows = []
        for ds in WQ.EVAL_ORDER:
            d, c = calls.get((sp, ds), (0.001, "WITHIN"))
            rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[sp] else "zero-shot", "questions": 100,
                         "base": [seed0, 0.3, 0.2], "new": [round(seed0 + d, 4), 0.3, 0.2], "rrf": [0.4, 0, 0],
                         "delta": [d, 0.002, -0.001], "lo": [d - 0.004, 0, 0], "hi": [d + 0.004, 0, 0], "call": c})
        b = f"outputs\\screen\\fits\\{BASE_FIT[sp]}"
        rec = {"new": f"outputs\\screen\\fits\\{FIT[sp]}", "bases": [b], "candidate": "p@swa", "carve": "s1eval",
               "trained_on": sorted(WQ.SPLITS[sp]), "by_base": {b: {"rows": rows}}, "stage": STAGE,
               "declared_in": DECLARED, "decided_against": BASE_FIT[sp], "split": sp}
        rec.update(over.get(sp, {}))
        WQ.write_json(Path(base) / "outputs" / "screen" / f"{FIT[sp]}.json", rec)


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zrckp over zret with the gaps; zrckp's, zrc's and zspk's arms unchanged; the names are WQ's
    mro = ZRCKPG.__mro__
    assert mro[1:8] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, GapRet, S3.ZRet, LG.LeanMLP8D), mro
    assert S.ARMS[ARM] == ARMS[ARM] == (ZRCKPG, Z3.KTChainCarveZRC)
    assert S.ARMS[BASE] == Z3P.ARMS[BASE] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and Z3P.ARM == BASE
    assert S.ARMS["zrc"] == Z2.ARMS["zrc"] and S.ARMS["zspk"] == Z3.ARMS["zspk"]
    assert FIT == {"WQ-musique": "scr-psm-zrckpg", "WQ-hotpotqa": "scr-psm-zrckpg-hp"}
    assert BASE_FIT == {"WQ-musique": "scr-wq-zrckp-s0", "WQ-hotpotqa": "scr-wq-zrckp-s0-hp"}
    tmp = Path(tempfile.mkdtemp(prefix="zpsm_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        plain = ZL.link_carve(RM.chain_carve(ToyGap, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = Z3.kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB", "WALK"]
        qs = np.r_[np.arange(c.rows)[::-1][:5], np.arange(c.rows)[:4]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        N = int(nq.numel())
        keep = torch.ones(B, len(blocks))
        keep[1, 2] = 0.0
        ref = S3.retrieved(feats)
        assert 0 < float(ref.sum()) < N
        # 2. from the same seed zrckpg is zrckp's state plus gap_w (zero, 16 x 8) and its forward is zrckp's bit for
        #    bit, in eval and with the same dropout draws in training; the global generator is where zrckp leaves it
        nets, after = {}, {}
        for k, cl in (("old", Z3P.ZRCKP), ("new", ZRCKPG)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after["old"], after["new"])
        so, sn = nets["old"].state_dict(), nets["new"].state_dict()
        assert sorted(set(sn) - set(so)) == ["gap_w"] and all(torch.equal(so[k], sn[k]) for k in so)
        assert tuple(sn["gap_w"].shape) == (16, 8) and float(sn["gap_w"].abs().sum()) == 0.0
        assert nets["new"].gap_blocks == ["rank", "WALK"] and nets["new"].gap_cols == 8
        with torch.no_grad():
            for mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out["old"], out["new"]), mode
        # 3. the gaps by hand (zret's z by lean_screen3's float64 reference, then the log-softmax): the constant column
        #    is -log n, the dropped block is zero, and a gap far below the soft best is floored at -10
        net = nets["new"].eval()
        G = net.pool_gaps(feats, keep, nq, B)
        hand = []
        for j, b in enumerate(blocks):
            if b in NO_GAP:
                continue
            z64 = S3.ref_numpy(feats[b], nq, B, ref)
            hand.append(gap_numpy(z64, nq.numpy(), B) * keep[nq, j].unsqueeze(1).numpy())
        H = np.concatenate(hand, 1)
        assert G.shape == (N, 8) and G.dtype == torch.float32 and not G.requires_grad
        assert float(np.abs(G.numpy() - H).max()) < 1e-4, float(np.abs(G.numpy() - H).max())
        n_of = np.bincount(nq.numpy(), minlength=B)
        live = (keep[nq, 2] > 0).numpy()
        assert np.allclose(G[live, 6].numpy(), -np.log(n_of[nq.numpy()][live]), atol=1e-5)
        assert float(G[~live, 5:].abs().sum()) == 0.0 and (~live).any()
        assert float((G <= 0).all()) and float((G >= GAP_FLOOR).all())
        far = pool_gap(torch.tensor([[0.0], [30.0], [29.0]]), torch.tensor([0, 0, 1]), 2)
        assert far[0, 0] == GAP_FLOOR and abs(float(far[1, 0])) < 1e-6 and float(far[2, 0]) == 0.0
        # 4. the forward by hand with gap_w (and the last layer) set: l1 of zret's parts plus the gaps through gap_w
        with torch.no_grad():
            gen = torch.Generator().manual_seed(3)
            net.gap_w.copy_(torch.randn(net.gap_w.shape, generator=gen))
            net.out.weight.copy_(torch.randn(net.out.weight.shape, generator=gen))
            parts = []
            for j, b in enumerate(blocks):
                m = keep[nq, j].unsqueeze(1)
                raw = S.raw_of(net, feats, b, nq)
                parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
            h = Fn.gelu(net.l1(torch.cat(parts, 1)) + torch.from_numpy(H).float() @ net.gap_w.T)
            h = Fn.gelu(net.l2(h))
            bz_ = S3.seg_zscore_ref(feats["rank"][:, S3.I_RRF:S3.I_RRF + 1], nq, B, ref).squeeze(1)
            s_hand = net.base_w * bz_ + net.out(h).squeeze(-1)
            s = net(feats, keep, nq, B, bz)
            assert float((s - s_hand).abs().max()) < 1e-3 and float((s - bz_).abs().max()) > 1e-3
        # 5. no learned value enters the gaps: other parameters, SEMB's inputs and the KT columns leave them unchanged;
        #    one question's fixed columns move its own gaps only; a row permutation permutes them
        with torch.no_grad():
            for p in net.parameters():
                p.add_(torch.randn(p.shape, generator=gen))
        f2 = dict(feats)
        f2["SEMB"] = (torch.randn(feats["SEMB"][0].shape, generator=gen), torch.randn(feats["SEMB"][1].shape, generator=gen))
        f2[Z3.KT_KEY] = torch.rand(feats[Z3.KT_KEY].shape, generator=gen)
        assert torch.equal(net.pool_gaps(f2, keep, nq, B), G)
        f3 = dict(feats)
        q0 = (nq == 0).unsqueeze(1)
        f3["WALK"] = torch.where(q0, torch.rand(feats["WALK"].shape, generator=gen) * 9, feats["WALK"])
        G3 = net.pool_gaps(f3, keep, nq, B)
        assert torch.equal(G3[nq != 0], G[nq != 0]) and not torch.equal(G3[nq == 0], G[nq == 0])
        perm = torch.randperm(N, generator=gen)
        fp = {b: feats[b][perm] for b in ("rank", "WALK")}
        assert float((net.pool_gaps(fp, keep, nq[perm], B) - G[perm]).abs().max()) < 1e-5
        # 6. the guard: a block whose columns are not the batch's own takes no gap
        orig = S.raw_of
        with WQ.patch(S, raw_of=lambda model, fs, b, q: fs[b] * 1.0 if b == "WALK" else orig(model, fs, b, q)):
            WQ.must_stop(net.pool_gaps, feats, keep, nq, B)
            WQ.must_stop(net, feats, keep, nq, B, bz)
        assert S.raw_of is orig
        # 7. a toy fit through lean_gpu's loop: gap_w moves, and a repeat is identical
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            r1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrckpg")
            r2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrckpg repeat")
        assert float(r1["swa"]["gap_w"].abs().sum()) > 0 and tuple(r1["swa"]["gap_w"].shape) == (16, 8)
        assert not LG.same_state(r1["swa"], r2["swa"])
        # 8. train, read and compare refuse what is not PSM's; zp1c3p's arm is restored after on_arm
        WQ.must_stop(train, ["--split", "WQ-musique", "--name", FIT["WQ-hotpotqa"], "--cache-root", WQ.ROOTS["cache"]])
        WQ.must_stop(train, ["--split", "L-musique", "--name", FIT["WQ-musique"], "--cache-root", WQ.ROOTS["cache"]])
        WQ.must_stop(train, ["--split", "WQ-musique", "--name", FIT["WQ-musique"], "--cache-root", "outputs/p1c2/cache"])
        WQ.must_stop(read, ["--name", BASE_FIT["WQ-musique"], "--cache-root", WQ.ROOTS["cache"]])
        W = types_ns(Z3P=Z3P)
        with on_arm(W):
            assert Z3P.ARM == ARM and Z3P.ARMS == ARMS
        assert Z3P.ARM == BASE and Z3P.ARMS[BASE] == (Z3P.ZRCKP, Z3.KTChainCarveZRC)
        T = tmp / "world"
        assert "missing" in WQ.must_stop(base_ready, T)
        WQ.write_json(T / (WQ.SUMMARY + ".json"), {"status": "NOT_READY", "declared_in": WQ.DECLARED})
        assert "NOT_READY" in WQ.must_stop(base_ready, T)
        WQ.write_json(T / (WQ.SUMMARY + ".json"), {"status": "BASE_READY", "declared_in": WQ.DECLARED})
        assert base_ready(T)["status"] == "BASE_READY"
        assert "no record" in psm_why(FIT["WQ-musique"], T) and "not a PSM fit" in psm_why(BASE_FIT["WQ-musique"], T)
        assert "no record" in WQ.must_stop(compare, "WQ-musique", T)
        fake_world(T)
        assert psm_why(FIT["WQ-musique"], T) is None and WQ.wq_why(BASE_FIT["WQ-musique"], T) is None
        assert "not read" in WQ.must_stop(compare, "WQ-musique", T)
        fake_fit(T, FIT["WQ-hotpotqa"], ARM, "WQ-musique", True)
        assert "not PSM's" in psm_why(FIT["WQ-hotpotqa"], T)
        # 9. the verdict: no LOSS and a GAIN adopts; a GAIN under a raised floor is WITHIN; a LOSS, no GAIN, a missing
        #    or foreign comparison, another seed-0 fit or REFUSED floors do not adopt; each names its next form
        cases = (({("WQ-musique", "webqsp"): (0.02, "GAIN")}, {}, {}, "ADOPT_FOR_FULL_RUN", "zspkg"),
                 ({("WQ-musique", "webqsp"): (0.015, "GAIN")}, {("WQ-musique", "webqsp"): 0.02}, {}, "NOT_ADOPTED",
                  "zrckph"),
                 ({("WQ-musique", "webqsp"): (0.02, "GAIN"), ("WQ-hotpotqa", "metaqa"): (-0.03, "LOSS")}, {}, {},
                  "NOT_ADOPTED", "zrckpu"),
                 ({}, {}, {}, "NOT_ADOPTED", "zrckph"),
                 ({("WQ-musique", "webqsp"): (0.02, "GAIN")}, {}, {"WQ-hotpotqa": {"stage": "WQ"}}, "INCOMPLETE",
                  "missing"),
                 ({("WQ-musique", "webqsp"): (0.02, "GAIN")}, {}, {"WQ-hotpotqa": {"bases": ["outputs/screen/fits/x"]}},
                  "INCOMPLETE", "missing"))
        for i, (calls, fls, over, want, nxt) in enumerate(cases):
            W_ = tmp / f"v{i}"
            fake_world(W_, calls, fls, **over)
            v = verdict(W_, W_ / "verdict")
            assert v["verdict"] == want and nxt in v["next"], (i, v["verdict"], v["next"], v["problems"])
            assert (W_ / "verdict.md").is_file() and (len(v["reads"]) == 12) == (want != "INCOMPLETE")
        W_ = tmp / "v6"
        fake_world(W_, {("WQ-musique", "webqsp"): (0.02, "GAIN")})
        WQ.write_json(W_ / (WQ.FLOORS[BASE] + ".json"), {"stage": WQ.STAGE, "declared_in": WQ.DECLARED, "arm": BASE,
                                                          "status": "REFUSED"})
        assert verdict(W_, W_ / "verdict")["verdict"] == "INCOMPLETE"
        W_ = tmp / "v7"
        fake_world(W_, {("WQ-musique", "webqsp"): (0.02, "GAIN")})
        fl = WQ.read_json(W_ / (WQ.FLOORS[BASE] + ".json"))
        fl["reads"][0]["seed0"] = 0.4999
        WQ.write_json(W_ / (WQ.FLOORS[BASE] + ".json"), fl)
        v = verdict(W_, W_ / "verdict")
        assert v["verdict"] == "INCOMPLETE" and any("seed-0 R@5" in p for p in v["problems"]), v["problems"]
        (W_ / "outputs" / "screen" / f"{FIT['WQ-musique']}.json").unlink()
        assert verdict(W_, W_ / "verdict")["verdict"] == "INCOMPLETE"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zpsm selftest: arm {ARM} (zrckp over zret with each fixed column's pool gap, W_g hidden x columns), zrckp's, "
        f"zrc's and zspk's arms unchanged; at the start {ARM} is zrckp bit for bit (eval and training draws, the global "
        f"generator unmoved); the gaps match a float64 reference (constant column -log n, dropped block zero, floor "
        f"-10); the forward by hand; no learned value, SEMB input or KT column moves the gaps, one question's columns "
        f"move its own only, a permutation permutes them; the guard stops a learned block; toy fits move gap_w and "
        f"repeat; train, read and compare refuse what is not PSM's; the verdict's nine cases "
        f"({time.time() - t0:.1f}s): ok")
    return 0


def types_ns(**kw):
    import types
    return types.SimpleNamespace(**kw)


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd, rest = (argv[0], argv[1:]) if argv else (None, [])
    if cmd == "--selftest":
        return selftest()
    if cmd == "train":
        return train(rest)
    if cmd == "read":
        return read(rest)
    if cmd == "compare":
        ap = argparse.ArgumentParser(prog="zpsm.py compare")
        ap.add_argument("--split", required=True, choices=SPLITS)
        return compare(ap.parse_args(rest).split)
    if cmd == "verdict":
        argparse.ArgumentParser(prog="zpsm.py verdict").parse_args(rest)
        return 0 if verdict()["verdict"] != "INCOMPLETE" else 2
    raise SystemExit("zpsm: train, read, compare, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
