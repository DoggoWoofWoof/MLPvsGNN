"""A3z (docs/A3Z_WHOLE_POOL_Z.md): step 1's whole-pool z-scores beside zret's, in both of the lean pair's models. A3,
A3t and A3v put the pool's retrieved share into zret's reference in training (vshare's ranges, then its two-sided
rule); the GNN arm gained nothing under any of them (outputs/a3v/screen-verdict.json), and A3v's verdict named this
form: the share idea as an input before it closes. zret takes every z-score against the pool's retrieved rows; step 1
(lean_mlp) took them against the whole pool. A3z keeps every input zret reads and adds, per row, before any row is
scored:

    per block b (w columns, SEMB's 64 learned ones included)  b's whole-pool z-score (lean_gpu.seg_zscore8D)
    rrf                                                        its whole-pool z-score (step 1's base term), a second
                                                               base term

The columns enter l1's pre-activation through a3z_w (hidden x columns, zero at the start), the base term through
a3z_bw (zero at the start): each arm starts as its base bit for bit, and no random number is drawn. On a pool whose
rows are all retrieved (squad's), or with fewer than two retrieved rows, the columns are zret's own z-scores bit for bit;
they differ only where the pool has two retrieved rows and another row. Nothing reads a score; nothing is normalised
over the question's scores (the whole-pool statistics are of input columns, as zret's are).

    zrckpz  zrckp's model (outputs/mp_unified/zp1c3p.py) with A3z's columns, on zrckp's carve
    zspkz   zspk's model (outputs/mp_unified/zp1c3.py) with A3z's columns; zsp's propagation reads the score they change

    python outputs/mp_unified/za3z.py train --split L-musique --name scr-a3z-zrckpz --arm zrckpz --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za3z.py read --name scr-a3z-zrckpz --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za3z.py compare --new outputs/screen/fits/scr-a3z-zrckpz \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3z-zrckpz
    python outputs/mp_unified/za3z.py pair --arm zrckpz --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za3z.py recall --arm zrckpz --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za3z.py verdict
    python outputs/mp_unified/za3z.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3 as Z3  # noqa: E402  (zspk and its KT carves; through it P1c2's roots, gate and card, zprop, zlink, rmatch)
import zp1c3p as Z3P  # noqa: E402  (zrckp)
import lean_screen3 as S3  # noqa: E402  (zret: seg_zscore_ref, retrieved)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A3Z_WHOLE_POOL_Z.md"
OUT = ROOT / "outputs" / "a3z"
SCREEN = ROOT / "outputs" / "screen"
A3Z = ("a3z_w", "a3z_bw")
SETTINGS = {"block": "each column's whole-pool z-score (lean_gpu.seg_zscore8D, step 1's: population sd, 0 where it is "
                     "under 1e-6), times the block's keep; SEMB's 64 learned columns included",
            "base": "rrf's whole-pool z-score (step 1's base term), through a3z_bw",
            "into": "l1's pre-activation through a3z_w (hidden x columns)", "start": "a3z_w and a3z_bw zero",
            "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpz": "zrckp", "zspkz": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a3z-{a}", "L-hotpotqa": f"scr-a3z-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpz": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkz": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpz": "zrckp with step 1's whole-pool z-scores beside its z-scores",
        "zspkz": "zspk with step 1's whole-pool z-scores beside its z-scores, before its propagation"}
PAIR = {a: SCREEN / f"scr-a3z-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a3z-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa", "musique")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A3z's re-call of the arm, "
        "or not twelve reads (docs/A3Z_WHOLE_POOL_Z.md, section 4)")
# the training pools by kind, counted in each training forward (stamped in each fit's screen.json): under two retrieved
# rows (zret's reference is the whole pool), all rows retrieved (the same), and the rest (the reference differs)
LAST = {"forwards": 0, "questions": 0, "under_two": 0, "all_retrieved": 0, "ref_differs": 0}


# ── the columns ──────────────────────────────────────────────────────────────


def n_extra(blocks, widths):
    """A3z's column count: one per column of every block, SEMB's included."""
    return sum(int(widths[b]) for b in blocks)


def whole_z(x, nq, B):
    """Each column's z-score against its whole pool: step 1's statistic (lean_gpu.seg_zscore8D: population sd, 0 where
    it is under 1e-6), the one zret falls back to."""
    return LG.seg_zscore8D(x, nq, B)


def pool_kinds(ref, nq, B):
    """(under two retrieved rows, all rows retrieved with at least two, the rest) over a batch's B pools."""
    with torch.no_grad():
        n = torch.zeros(B, dtype=ref.dtype, device=ref.device).index_add_(0, nq, torch.ones_like(ref))
        r = torch.zeros(B, dtype=ref.dtype, device=ref.device).index_add_(0, nq, ref)
        under = r < 2
        full = ~under & (r == n)
        return int(under.sum()), int(full.sum()), int((~under & ~full).sum())


class A3ZMix(S3.ZRet):
    """zret's forward with A3z's columns beside (the module's docstring): into l1's pre-activation through a3z_w, and
    rrf's whole-pool z-score through a3z_bw, both zero at the start."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a3z_n = n_extra(self.blocks, self.widths)
        self.a3z_w = nn.Parameter(torch.zeros(self.l1.out_features, self.a3z_n))
        self.a3z_bw = nn.Parameter(torch.zeros(1))

    def a3z_inputs(self, feats, keep, nq, B):
        """(zret's l1 input, A3z's columns, rrf's z against the retrieved rows, rrf's whole-pool z)."""
        ref = S3.retrieved(feats)
        if self.training:
            u, f, d = pool_kinds(ref, nq, B)
            LAST["forwards"] += 1
            LAST["questions"] += int(B)
            LAST["under_two"] += u
            LAST["all_retrieved"] += f
            LAST["ref_differs"] += d
        parts, extra = [], []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
            extra.append(whole_z(raw, nq, B) * m)
        r1 = feats["rank"][:, S3.I_RRF:S3.I_RRF + 1]
        bz = S3.seg_zscore_ref(r1, nq, B, ref).squeeze(1)
        bw = whole_z(r1, nq, B).squeeze(1)
        return torch.cat(parts, 1), torch.cat(extra, 1), bz, bw

    def forward(self, feats, keep, nq, B, base_z):
        x, e, bz, bw = self.a3z_inputs(feats, keep, nq, B)
        h = self.drop(Fn.gelu(self.l1(x) + Fn.linear(e, self.a3z_w)))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * bz + self.a3z_bw * bw + self.out(h).squeeze(-1)


class ZRCKPZ(Z3P.ZRCKP, A3ZMix):
    """zrckpz: zrckp's model with A3z's columns (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3ZMix, ZRet)."""


class ZSPKZ(Z3.ZSPK, A3ZMix):
    """zspkz: zspk's model with A3z's columns; zsp's propagation reads the score they change (MRO: ZSPK, ZProp, KTMix,
    ZRM, ChainMatch, A3ZMix, ZRet)."""


ARMS = {"zrckpz": (ZRCKPZ, Z3.KTChainCarveZRC), "zspkz": (ZSPKZ, Z3.KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def split_of_fit(name):
    for a in ARM_NAMES:
        for sp, n in FIT[a].items():
            if n == name:
                return a, sp
    return None, None


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za3z_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_gpu_sha256": LC.sha_src(LG.__file__),
                    "a3z": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "parameters": list(A3Z), "settings": SETTINGS,
                        "training_pools": dict(LAST),
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za3z: train takes --arm zrckpz or zspkz, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za3z: A3z's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za3z: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a3z" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za3z: {name} is not an A3z fit (arm {rec.get('arm')})")
    if rec["a3z"].get("split") != split:
        raise SystemExit(f"za3z: {name} is {split}'s fit, its record says {rec['a3z'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za3z: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za3z: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp)"),
            ("(rel's screen fits)", f"({b}'s card fits)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A3z's whole-pool z beside) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A3z ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A3z", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za3z_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(arm, screens, out):
    with on_base(arm):
        rec = Z.pair(screens, out)
    restamp(out, arm)
    return rec


def recall(arm, null_files, pair_file, out, base_screens=None):
    with on_base(arm):
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out, arm)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools, then its full run, its own declared file",
        "GAIN_AND_LOSS": "the back-off form, A3zg: A3z's contribution (a3z_w's and a3z_bw's terms) scaled by "
                         "2 sigmoid(<w, c> + b), zero-started so it starts at 1, c the pool's log size and retrieved "
                         "share (input columns only), its own declared file",
        "NO_GAIN": "A3zx, A3z's columns on the adopted arms, R1's zrckpr and G5's zspx, each against its own base, its "
                   "own declared file; the share idea closes as an input only if A3zx finds no GAIN either",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A3z's screen verdict, per arm, from its re-call (section 4); webqsp, metaqa and musique beside, musique's in-domain
    read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A3z" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A3z's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "next_form": next_form(v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A3z", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za3z_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A3z's screen verdict: zrckpz **{per['zrckpz']['verdict']}**, zspkz **{per['zspkz']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    num = lambda x: f"{x:+.4f}" if isinstance(x, (int, float)) else "-"
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** ({WHAT[a]}) against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ". Beside: "
                  + (", ".join(f"{k} {b['call']} ({num(b['delta'])})" for k, b in p["beside"].items()) or "none")
                  + f". Next: {p['next_form']}.")
    md += ["", f"Beside, deciding nothing: musique in-domain ({ck}): "
           + ", ".join(f"{a} {side[a]['call']} ({num(side[a]['delta'])})" for a in ARM_NAMES) + ".",
           "", "This verdict closes only this form (step 1's whole-pool z-scores beside zret's, into l1); "
               "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za3z verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A3z", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def whole_by_hand(x, nq, B, eps=1e-6):
    """The whole-pool z-score by hand in float64 (population sd; 0 where the sd is under eps)."""
    x, nq = x.double().numpy(), nq.numpy()
    out = np.zeros_like(x)
    for q in range(B):
        rows = np.flatnonzero(nq == q)
        for c in range(x.shape[1]):
            col = x[rows, c]
            sd = col.std() if rows.size else 0.0
            out[rows, c] = 0.0 if sd < eps else (col - col.mean()) / sd
    return out


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models with A3ZMix just above zret's forward; the base arms unchanged
    assert S.ARMS["zrckpz"] == (ZRCKPZ, Z3.KTChainCarveZRC) and S.ARMS["zspkz"] == (ZSPKZ, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZM.S3 is S3 and S3.LG is LG
    assert ZRCKPZ.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A3ZMix, S3.ZRet)
    assert ZSPKZ.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A3ZMix, S3.ZRet)
    # 2a. the columns on a hand-made batch of five pools, two columns each:
    #     q0  six rows, v1 and v2 retrieved (A3v's worked example, docs/A3V_VSHARE_TWO_SIDED.md): the two references differ
    #     q1  four rows, all retrieved (squad's kind): the columns are zret's z-scores bit for bit
    #     q2  five rows, one retrieved: zret falls back to the whole pool, so the same
    #     q3  three rows, two retrieved, column 0 constant over them (zret falls back) and column 1 constant: the same
    #     q4  one row: 0 in both
    col0 = [0.81, 0.78, 0.62, 0.55, 0.40, 0.31, 0.9, 0.7, 0.5, 0.2, 0.6, 0.5, 0.45, 0.3, 0.1, 0.5, 0.5, 0.2, 0.7]
    col1 = [1, 2, 3, 4, 5, 6, 3, 1, 4, 1, 2, 7, 1, 8, 2, 4, 4, 4, 5]
    hx = torch.tensor([col0, col1], dtype=torch.float32).T.contiguous()
    hq = torch.tensor([0] * 6 + [1] * 4 + [2] * 5 + [3] * 3 + [4], dtype=torch.int64)
    href = torch.tensor([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1], dtype=torch.float32)
    hw, hz = whole_z(hx, hq, 5), S3.seg_zscore_ref(hx, hq, 5, href)
    assert np.allclose(hw.double().numpy(), whole_by_hand(hx, hq, 5), atol=1e-5)
    assert np.allclose(hw[:6, 0].numpy(), [1.27, 1.10, 0.23, -0.15, -0.98, -1.47], atol=0.006)
    assert np.allclose(hz[:6, 0].numpy(), [1.00, -1.00, -11.67, -16.33, -26.33, -32.33], atol=0.006)
    assert float((hw[:6] - hz[:6]).abs().min()) > 0.1
    assert torch.equal(hw[6:], hz[6:]) and float(hw[15:, 1].abs().max()) == 0.0 and float(hw[18].abs().max()) == 0.0
    assert pool_kinds(href, hq, 5) == (2, 1, 2)                     # under two: q2, q4; all retrieved: q1; else q0, q3
    rng = np.random.default_rng(58)
    tmp = Path(tempfile.mkdtemp(prefix="za3z_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        c_m = cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:9], np.arange(c_m.rows)[:7]].astype(np.int64)
        B = qs.size
        f_, nq_, bz_, _gold = c_m.batch(qs, blocks)
        keep_ = torch.ones(B, len(blocks))
        keep_[3, 1] = 0.0                                               # one question without SEMB: its columns are 0
        # 2b. on the toy batch: the columns are each block's whole-pool z-score by hand, SEMB's learned columns
        #     included; rrf's is the batch's own base_z (step 1's base term); zret's l1 input is unchanged
        torch.manual_seed(0)
        m = ZRCKPZ(blocks, c_m.widths, 16, dropout=0.1, seed=0)
        assert m.a3z_n == n_extra(blocks, c_m.widths) == 5 + 64 and tuple(m.a3z_w.shape) == (16, 69)
        m.eval()
        x, e, bz, bw = m.a3z_inputs(f_, keep_, nq_, B)
        raw_r = f_["rank"]
        semb = S.raw_of(m, f_, "SEMB", nq_).detach()
        m1 = keep_[nq_, 1].unsqueeze(1).numpy()
        want = np.concatenate([whole_by_hand(raw_r, nq_, B), whole_by_hand(semb, nq_, B) * m1], 1)
        assert e.shape == (nq_.numel(), 69) and bool(torch.isfinite(e).all())
        assert np.allclose(e.detach().double().numpy(), want, atol=1e-4), float(np.abs(e.detach().numpy() - want).max())
        assert float(e[nq_ == 3, 5:].detach().abs().max()) == 0.0
        # rrf's whole-pool z-score: on the cache carves it is the batch's own base_z (lean_gpu.CacheCarve z-scores
        # the rank block's rrf column, the one zret's I_RRF reads); the toy carve's base_z takes another column
        assert np.allclose(bw.double().numpy(), whole_by_hand(raw_r[:, S3.I_RRF:S3.I_RRF + 1], nq_, B)[:, 0], atol=1e-5)
        import inspect
        src_init, src_batch = inspect.getsource(LG.CacheCarve.__init__), inspect.getsource(LG.CacheCarve.batch)
        assert 'self.c_rrf = self.span["rank"][0] + LM.SPLIT["rank"].index("rrf")' in src_init
        assert "base_z = seg_zscore8D(rrf.unsqueeze(1), nq, B).squeeze(1)" in src_batch
        assert S3.I_RRF == LM.SPLIT["rank"].index("rrf") and not torch.allclose(bw, bz_, atol=1e-3)
        ret = S3.retrieved(f_)
        kinds = pool_kinds(ret, nq_, B)
        assert sum(kinds) == B and kinds[2] > 0, kinds              # the toy batch has pools where the two differ
        zr = torch.cat([S3.seg_zscore_ref(raw_r, nq_, B, ret), S3.seg_zscore_ref(semb, nq_, B, ret)
                        * torch.from_numpy(m1)], 1)
        assert not torch.allclose(e.detach(), zr, atol=1e-3)
        with torch.no_grad():
            xz = torch.cat(sum(([S.raw_of(m, f_, b, nq_) * keep_[nq_, j].unsqueeze(1),
                                 S3.seg_zscore_ref(S.raw_of(m, f_, b, nq_), nq_, B, ret) * keep_[nq_, j].unsqueeze(1),
                                 keep_[nq_, j].unsqueeze(1)] for j, b in enumerate(blocks)), []), 1)
        assert torch.equal(x.detach(), xz)
        # the kinds are counted in training forwards only
        LAST.update({k: 0 for k in LAST})
        m.a3z_inputs(f_, keep_, nq_, B)
        assert LAST["forwards"] == 0
        m.train()
        m.a3z_inputs(f_, keep_, nq_, B)
        assert LAST == {"forwards": 1, "questions": B, "under_two": kinds[0], "all_retrieved": kinds[1],
                        "ref_differs": kinds[2]}
        # 3. each arm starts as its base bit for bit (same seed, no random number drawn; its scores and every shared
        #    parameter's gradient); with a3z_w and a3z_bw moved, the scorer is zret's on A3z's inputs, and zspkz's
        #    propagation reads that score
        for arm, cls, base, cv in (("zrckpz", ZRCKPZ, Z3P.ZRCKP, c_m), ("zspkz", ZSPKZ, Z3.ZSPK, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            s0, s1 = m0.state_dict(), m1_.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(A3Z) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0}) and all(float(s1[k].abs().max()) == 0 for k in A3Z)
            if arm == "zspkz":
                w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                with torch.no_grad():
                    m0.sp_w2.copy_(w)
                    m1_.sp_w2.copy_(w)
            for mode in ("eval", "train"):
                getattr(m0, mode)()
                getattr(m1_, mode)()
                torch.manual_seed(3)
                y0 = m0(f2, keep_, nq2, B, bz2)
                torch.manual_seed(3)
                y1 = m1_(f2, keep_, nq2, B, bz2)
                assert torch.equal(y0, y1), (arm, mode)
                m0.zero_grad()
                m1_.zero_grad()
                (y0 * torch.linspace(-1, 1, y0.numel())).sum().backward()
                (y1 * torch.linspace(-1, 1, y1.numel())).sum().backward()
                for n_, p0 in m0.named_parameters():
                    g0, g1 = p0.grad, dict(m1_.named_parameters())[n_].grad
                    assert (g0 is None and g1 is None) or torch.equal(g0, g1), (arm, mode, n_)
                # out starts at zero: a3z_w has no gradient at the start, as l1 has none; a3z_bw has one
                assert float(m1_.a3z_bw.grad.abs().max()) > 0 and float(m1_.a3z_w.grad.abs().max()) == 0.0
                assert float(m1_.l1.weight.grad.abs().max()) == 0.0
            m1_.eval()
            with torch.no_grad():
                m1_.a3z_w.copy_(torch.randn(m1_.a3z_w.shape, generator=torch.Generator().manual_seed(9)) * 0.1)
                m1_.a3z_bw.fill_(0.3)
                x, e, bz, bw = m1_.a3z_inputs(f2, keep_, nq2, B)
                h = Fn.gelu(m1_.l1(x) + e @ m1_.a3z_w.T)
                want = m1_.base_w * bz + 0.3 * bw + m1_.out(Fn.gelu(m1_.l2(h))).squeeze(-1)
                sc = A3ZMix.forward(m1_, f2, keep_, nq2, B, bz2)
                assert torch.allclose(sc, want, atol=1e-5) and not torch.allclose(sc, S3.ZRet.forward(m1_, f2, keep_, nq2,
                                                                                                       B, bz2))
                pre = super(ZP.ZProp, m1_).forward(f2, keep_, nq2, B, bz2) if arm == "zspkz" else None
                y1 = m1_(f2, keep_, nq2, B, bz2)
                if arm == "zrckpz":
                    assert torch.allclose(y1, sc, atol=1e-6)                   # toy carves: no KT input, chain gate 0
                else:
                    assert torch.allclose(pre, sc, atol=1e-6)
                    assert torch.allclose(y1, pre + m1_.prop_head(ZP.prop_inputs(pre, f2[ZL.LK_KEY], nq2, B)),
                                          atol=1e-5)
                    assert float((y1 - sc).abs().max()) > 1e-4
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical, A3z's weights learn, and the kinds are
        #    counted; no base arm's model is an A3ZMix (A3z changes no base class)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpz", c_m), ("zspkz", c_g)):
            LAST.update({k: 0 for k in LAST})
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                seen = dict(LAST)
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["a3z_w"].abs().max()) > 0 and float(fit["swa"]["a3z_bw"].abs().max()) > 0
            assert seen["forwards"] > 0 and seen["questions"] == seen["under_two"] + seen["all_retrieved"] + seen[
                "ref_differs"] and seen["ref_differs"] > 0, seen
            assert LAST["forwards"] == 2 * seen["forwards"] and LAST["questions"] == 2 * seen["questions"]
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A3ZMix)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a3z-zspkz-hp") == ("zspkz", "L-hotpotqa") and split_of_fit("scr-p5c-zspko") == (None, None)
    good = ["train", "--arm", "zrckpz", "--name", FIT["zrckpz"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpz"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkz", "--name", FIT["zrckpz"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkz", "--name", FIT["zspkz"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a3z-zspkz", {"arm": "zspko", "p5c": {}}),
                        ("scr-a3z-zspkz-hp", {"arm": "zspkz", "a3z": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A3z's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T_ / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T_ / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T_ / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T_ / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T_ / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "A3z" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A3z ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A3z's whole-pool z beside) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A3zg, no gain -> A3zx
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpz": fake_recall(T_ / "a.json", "zrckpz", {mi: "GAIN"}),
                       "zspkz": fake_recall(T_ / "b.json", "zspkz", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpz"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpz"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspkz"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkz"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpz": {"call": "GAIN", "delta": 0.0},
                                                   "zspkz": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkz"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpz": fake_recall(T_ / "a.json", "zrckpz", {}),
                       "zspkz": fake_recall(T_ / "b.json", "zspkz", {gm: "GAIN"}, stage="A3v")}, T_ / "v")
        assert rec["arms"]["zrckpz"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkz"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpz"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspkz"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpz": T_ / "none.json", "zspkz": fake_recall(T_ / "b.json", "zrckpz", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpz"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkz"]["verdict"] == "INCOMPLETE"
    log(f"za3z selftest: the arms are zrckp's and zspk's models with A3ZMix just above zret's forward; on a hand-made "
        f"batch the columns are the whole-pool z-scores by hand (A3v's worked example: 1.27 .. -1.47 beside zret's 1.00 "
        f".. -32.33) and zret's own bit for bit on pools all retrieved, under two retrieved, constant or of one row; on "
        f"the toy batch they match by hand, SEMB's included (a question without SEMB has none), rrf's is the cache "
        f"carves' base_z (by their source), and the pool kinds are counted in training only; each arm starts as its base bit for bit, scores and "
        f"every shared gradient, drawing no random number; moved, the scorer is zret's on A3z's inputs and zspkz's "
        f"propagation reads it; toy fits repeat bit for bit and A3z's weights learn; train refuses another arm, split, "
        f"fit name, cache root or device; read refuses a fit that is not A3z's or not its split's; relz's pair and "
        f"re-call per arm, stamped A3z's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--split", default="L-musique", choices=S2.SPLITS)
    ap.add_argument("--selftest", action="store_true")
    k, rest = ap.parse_known_args(argv)
    if k.selftest:
        return selftest()
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair(g.arm, [x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(g.arm, null, g.pair, g.out)
        except SystemExit as e:
            log(f"za3z {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za3z: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
