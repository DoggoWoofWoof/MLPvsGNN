"""A8e (docs/A8E_QDEPTH_PRIOR.md): round 11's own form of the question-to-depth prior, inside both of the lean pair's
models. Each row's score takes its question's prior at the row's depth class, learned with the rest of the model:

    s(v) = the base model's s(v) + l_q[depth(v)],    l_q = q U + P_q W + b    (five values per question)

q is the question's embedding (SEMB's input), P_q its pool's profile (the share of its rows in each depth class and
log(1 + its size), qdepth.profile), depth(v) qdepth.depth_of's class of the row from WALK's raw columns (0 a seed, 1-3
the first structural hop, the nearest, 4 unreached). U (1536 x 5), W (6 x 5) and b (5), 7,715 parameters, start at
zero (qdepth.QDepth's), so each arm starts as its base bit for bit; the model's listwise loss trains them with its own
Adam, learning rate and weight decay. The prior is computed by qdepth.QDepth.depth_logits itself. It reads the
question's embedding and the pool's fixed walks: no label, no row's score, nothing passed over graph neighbours. A
question whose WALK or SEMB block is dropped takes none (round 11's rule; fits and reads keep every block).

    zrckpqe  zrckp's model (outputs/mp_unified/zp1c3p.py) plus the prior, added to the final score, on zrckp's carve
    zspkqe   zspk's model (outputs/mp_unified/zp1c3.py) plus the prior, added with the KT head before zsp's
             propagation reads the score, on zspk's carve

No column and no builder (A8's columns are not read): the prior's inputs are blocks of P1c2's cache that both base
models already read.

    python outputs/mp_unified/za8e.py train --split L-musique --name scr-a8e-zrckpqe --arm zrckpqe --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8e.py read --name scr-a8e-zrckpqe --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8e.py compare --new outputs/screen/fits/scr-a8e-zrckpqe \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a8e-zrckpqe
    python outputs/mp_unified/za8e.py pair --arm zrckpqe --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za8e.py recall --arm zrckpqe --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za8e.py verdict
    python outputs/mp_unified/za8e.py --selftest
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
import qdepth as QD  # noqa: E402  (round 11's prior: depth_of, profile, QDepth.depth_logits)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A8E_QDEPTH_PRIOR.md"
OUT = ROOT / "outputs" / "a8e"
SCREEN = ROOT / "outputs" / "screen"
QDP = ("qd_u", "qd_p", "qd_b")
N_QD = QD.Q_DIM * QD.DEPTHS + QD.PROF_W * QD.DEPTHS + QD.DEPTHS
SETTINGS = {"prior": "round 11's (outputs/mp_unified/qdepth.py): l_q = q U + P_q W + b over the five depth classes, "
                     "added at each row's class (qdepth.QDepth.depth_logits)",
            "depths": QD.DEPTHS, "walk": {"width": QD.WALK_W, "first_hop": list(QD.FIRST_HOP), "is_seed": QD.IS_SEED},
            "profile": "qdepth.profile: the depth-class shares and log(1 + the pool's size)",
            "parameters": N_QD, "init": "zero",
            "trained_by": "the model's listwise loss, with its Adam, learning rate and weight decay",
            "dropped_block": "a question whose WALK or SEMB block is dropped takes no prior", "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpqe": "zrckp", "zspkqe": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a8e-{a}", "L-hotpotqa": f"scr-a8e-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpqe": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                         "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkqe": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpqe": "zrckp plus round 11's depth prior", "zspkqe": "zspk plus round 11's depth prior before its propagation"}
PAIR = {a: SCREEN / f"scr-a8e-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a8e-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A8e's re-call of the arm, "
        "or not twelve reads (docs/A8E_QDEPTH_PRIOR.md, section 4)")


# ── the prior ────────────────────────────────────────────────────────────────


def qd_check(blocks, widths):
    """qdepth.QDepth's refusals, before the base model is built: the prior needs WALK (16 wide) and SEMB."""
    if "WALK" not in blocks or "SEMB" not in blocks:
        raise SystemExit("za8e: the depth prior needs the WALK and SEMB blocks")
    if int(widths["WALK"]) != QD.WALK_W:
        raise SystemExit(f"za8e: WALK has {widths['WALK']} columns, lean_mlp's WALK {QD.WALK_W}")


def qd_init(m):
    """Round 11's prior on model m: U, W and b at zero (qdepth.QDepth's; no draw from any generator), and where WALK and
    SEMB sit among its blocks."""
    m.j_walk, m.j_semb = m.blocks.index("WALK"), m.blocks.index("SEMB")
    m.qd_u = nn.Parameter(torch.zeros(QD.Q_DIM, QD.DEPTHS))
    m.qd_p = nn.Parameter(torch.zeros(QD.PROF_W, QD.DEPTHS))
    m.qd_b = nn.Parameter(torch.zeros(QD.DEPTHS))


def qd_term(m, feats, keep, nq, B):
    """(N,) each row's prior l_q[depth(v)], as qdepth.QDepth.forward adds it: from its question's embedding, its pool's
    depth profile and its own depth class (qdepth.QDepth.depth_logits). It takes no score; a question whose WALK or SEMB
    block is dropped takes none."""
    d, lq = QD.QDepth.depth_logits(m, feats, nq, B)
    return (lq[nq] * Fn.one_hot(d, QD.DEPTHS).to(lq.dtype)).sum(1) * (keep[nq, m.j_walk] * keep[nq, m.j_semb])


class ZRCKPQE(Z3P.ZRCKP):
    """zrckpqe: zrckp's model plus round 11's depth prior (zero at the start), added to the final score."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        qd_check(blocks, widths)
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        qd_init(self)

    qd_term = qd_term

    def forward(self, feats, keep, nq, B, base_z):
        return super().forward(feats, keep, nq, B, base_z) + self.qd_term(feats, keep, nq, B)


class QDMix(Z3.KTMix):
    """zrm's score, the KT head, then round 11's depth prior: in zspkqe, the score zsp's propagation reads."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        qd_check(blocks, widths)
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        qd_init(self)

    qd_term = qd_term

    def forward(self, feats, keep, nq, B, base_z):
        return super().forward(feats, keep, nq, B, base_z) + self.qd_term(feats, keep, nq, B)


class ZSPKQE(ZP.ZProp, QDMix):
    """zspkqe: zspk's model with round 11's depth prior added before zsp's propagation reads the score (MRO: ZProp,
    QDMix, KTMix, ZRM)."""


ARMS = {"zrckpqe": (ZRCKPQE, Z3.KTChainCarveZRC), "zspkqe": (ZSPKQE, Z3.KTLinkCarveBase)}
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
        mp = out_root / name / "models.pt"
        rec.update({"za8e_sha256": LC.sha_src(__file__), "qdepth_sha256": LC.sha_src(QD.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a8e": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "prior": list(QDP), "settings": SETTINGS,
                        "prior_norms": QD.head_norms(mp) if mp.exists() else None,
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za8e: train takes --arm zrckpqe or zspkqe, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za8e: A8e's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za8e: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a8e" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za8e: {name} is not an A8e fit (arm {rec.get('arm')})")
    if rec["a8e"].get("split") != split:
        raise SystemExit(f"za8e: {name} is {split}'s fit, its record says {rec['a8e'].get('split')}")
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
        raise SystemExit(f"za8e: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za8e: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A8e's depth prior) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A8e ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A8e", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za8e_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "A8ek, the back-off form: the prior split by the graph's kind (one U, W and b for the KB "
                         "graphs, metaqa and webqsp, and one for the passage graphs, each at zero), so one kind's depth "
                         "weights cannot move the other kind's scores; on both models, its own declared file",
        "NO_GAIN": "A8ex, the same prior on the adopted arms (G5's zspx, added before its propagation reads the "
                   "score, and R1's zrckpr, added to its final score), on both models, its own declared file; the "
                   "depth prior closes as a model term only if A8ex finds no GAIN either",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A8e's screen verdict, per arm, from its re-call (section 4); webqsp and metaqa beside, musique's in-domain read
    of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A8e" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A8e's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems, "next_form": next_form(v, gains, losses),
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                             if r.get("dataset") in BESIDE},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    ck = f"{CLAIM1_READ[0]} {CLAIM1_READ[1]}"
    side = {a: {"call": per[a]["calls"].get(ck), "delta": per[a]["delta_r5"].get(ck)} for a in ARM_NAMES}
    rec = {"stage": "A8e", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za8e_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A8e's screen verdict: zrckpqe **{per['zrckpqe']['verdict']}**, zspkqe **{per['zspkqe']['verdict']}**", "",
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
           "", "This verdict closes only this form (round 11's prior inside each model, trained by its listwise loss); "
           "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za8e verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A8e", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


class ToyWalk(ZM.ToyZ):
    """zrm's toy carve with its questions' ids (P1c3's KT input checks them) and a WALK block in lean_mlp's layout:
    twelve walk columns, the first-hop one-hot of hops 1 to 3 (columns 12-14) and is-seed (15). Rows 0-4 hold the five
    depth classes; row 0 is a seed that is also a first hop, rows 1 and 2 carry a second, farther first hop, and so do
    about a fifth of the other rows of classes 0-2."""

    def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
        super().__init__(ds, carve, basis, device, cache_root, verify, score2)
        self.ids = [f"q{i}" for i in range(self.rows)]
        N = int(self.off_np[-1])
        g = torch.Generator().manual_seed(13)
        w = torch.zeros(N, QD.WALK_W)
        w[:, :12] = torch.rand(N, 12, generator=g) * 3
        k = torch.randint(0, QD.DEPTHS, (N,), generator=g)
        k[:5] = torch.arange(QD.DEPTHS)
        w[k == 0, QD.IS_SEED] = 1.0
        for h in (1, 2, 3):
            w[k == h, QD.FIRST_HOP[h - 1]] = 1.0
        extra = (torch.rand(N, generator=g) < 0.2) & (k < 3)
        extra[:3] = True
        w[extra, QD.FIRST_HOP[2]] = 1.0
        w[0, QD.FIRST_HOP[1]] = 1.0
        self.walk, self.depth = w, k
        self.widths = {**self.widths, "WALK": QD.WALK_W}

    def batch(self, qs, blocks):
        feats, nq, base_z, gold = super().batch(qs, [b for b in blocks if b != "WALK"])
        if "WALK" in blocks:
            qs = np.asarray(qs, np.int64)
            cnt = self.n_np[qs]
            idx = np.repeat(self.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
            feats["WALK"] = self.walk[torch.from_numpy(idx)]
        return feats, nq, base_z, gold


def hand_prior(m, feats, nq, B):
    """float64 by hand from the raw walk flags and the questions' embeddings: (each row's depth class, its l_q there)."""
    W = feats["WALK"].double().numpy()
    qe = feats["SEMB"][0].double().numpy()
    U, P, b = (getattr(m, k).detach().double().numpy() for k in QDP)
    hop = W[:, list(QD.FIRST_HOP)] > 0.5
    d = np.where(W[:, QD.IS_SEED] > 0.5, 0, np.where(hop.any(1), hop.argmax(1) + 1, QD.DEPTHS - 1))
    nqn = nq.numpy()
    out = np.zeros(d.size)
    for q in range(B):
        r = np.flatnonzero(nqn == q)
        prof = np.r_[[np.mean(d[r] == c) for c in range(QD.DEPTHS)], np.log1p(r.size)]
        out[r] = (qe[q] @ U + prof @ P + b)[d[r]]
    return d, out


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models plus the prior, on their bases' own carves; the bases unchanged
    assert S.ARMS["zrckpqe"] == (ZRCKPQE, Z3.KTChainCarveZRC) and S.ARMS["zspkqe"] == (ZSPKQE, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPQE.__mro__[1] is Z3P.ZRCKP and ZSPKQE.__mro__[1:5] == (ZP.ZProp, QDMix, Z3.KTMix, ZM.ZRM)
    assert "zrckpq" not in ARMS and "zspkq" not in ARMS            # A8's arms are not this stage's
    # 2. the prior is round 11's: qdepth's constants and 7,715 parameters
    assert (QD.DEPTHS, QD.WALK_W, QD.FIRST_HOP, QD.IS_SEED, QD.Q_DIM, QD.PROF_W) == (5, 16, (12, 13, 14), 15, 1536, 6)
    assert N_QD == 7715
    rng = np.random.default_rng(83)
    tmp = Path(tempfile.mkdtemp(prefix="za8e_"))
    neutral = "za8e-selftest-frozen"
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        cls_ch = RM.chain_carve(ToyWalk, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng, cache_root=str(tmp / "cache"))
        c_m = Z3.kt_carve(cls_ch, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_g = Z3.kt_carve(cls_lk, tmp / "kt")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert torch.equal(QD.depth_of(c_m.walk), c_m.depth) and QD.depth_of(c_m.walk)[:5].tolist() == [0, 1, 2, 3, 4]
        blocks = ["rank", "SEMB", "WALK"]
        widths = {b: c_m.widths[b] for b in blocks}
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        keep_ = torch.ones(B, len(blocks))
        fm, _nq, _bz, _g = c_m.batch(qs, blocks)
        fg, _nqg, _bzg, _gg = c_g.batch(qs, blocks)
        assert Z3.KT_KEY in fm and Z3.KT_KEY in fg and ZL.LK_KEY not in fm and ZL.LK_KEY in fg
        # 3. the models: from the same seed each arm is its base's state plus the prior at zero, the global stream where
        #    the base leaves it; in eval and in training (the same dropout draws) its scores are the base's bit for bit,
        #    and so is every gradient of the first listwise step but the prior's own, which move (finite, b's summing
        #    to zero over the classes)
        for arm, cls, base, cv in (("zrckpqe", ZRCKPQE, Z3P.ZRCKP, c_m), ("zspkqe", ZSPKQE, Z3.ZSPK, c_g)):
            f_, nq_, bz_, gold_ = cv.batch(qs, blocks)
            nets, after = {}, {}
            for k, cl in (("base", base), ("arm", cls)):
                torch.manual_seed(0)
                nets[k] = cl(blocks, widths, 16, dropout=0.1, seed=0)
                after[k] = torch.rand(4)
            assert torch.equal(after["base"], after["arm"]), arm
            m0, m1 = nets["base"], nets["arm"]
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(QDP) and not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in QDP) == N_QD and all(float(s1[k].abs().max()) == 0.0 for k in QDP)
            ys = {}
            for k, m in nets.items():
                m.train()
                torch.manual_seed(5)
                ys[k] = m(f_, keep_, nq_, B, bz_)
                LG.listwiseD(ys[k], gold_, nq_, B).backward()
            assert torch.equal(ys["base"], ys["arm"]), arm
            p1 = dict(m1.named_parameters())
            for k, p in m0.named_parameters():
                assert (p.grad is None) == (p1[k].grad is None), (arm, k)
                assert p.grad is None or torch.equal(p.grad, p1[k].grad), (arm, k)
            gq = {k: p1[k].grad for k in QDP}
            assert all(bool(torch.isfinite(v).all()) and float(v.abs().sum()) > 0 for v in gq.values()), arm
            assert abs(float(gq["qd_b"].sum())) < 1e-5, float(gq["qd_b"].sum())
            m0.eval()
            m1.eval()
            with torch.no_grad():
                # 4. with the KT head (and zspk's propagation) set alike, still the base; with the prior set, zrckpqe is
                #    the base plus the prior and zspkqe's propagation reads the score with the prior added; the prior
                #    is l_q at the row's own class against a float64 hand computation from the raw walks
                gen = torch.Generator().manual_seed(9)
                for k in ("kt_w2",) + (("sp_w2",) if arm == "zspkqe" else ()):
                    w = torch.randn(getattr(m0, k).shape, generator=gen)
                    getattr(m0, k).copy_(w)
                    getattr(m1, k).copy_(w)
                y0 = m0(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, m1(f_, keep_, nq_, B, bz_)), arm
                for k, sc in zip(QDP, (0.02, 0.5, 0.5)):
                    getattr(m1, k).copy_(torch.randn(getattr(m1, k).shape, generator=gen) * sc)
                term = m1.qd_term(f_, keep_, nq_, B)
                d_h, want = hand_prior(m1, f_, nq_, B)
                assert np.array_equal(d_h, QD.depth_of(f_["WALK"]).numpy()) and len(set(d_h.tolist())) == QD.DEPTHS
                assert np.allclose(term.double().numpy(), want, atol=1e-5, rtol=1e-5) and float(term.abs().max()) > 1e-2
                y1 = m1(f_, keep_, nq_, B, bz_)
                if arm == "zrckpqe":
                    assert torch.equal(y1, y0 + term)
                else:
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    assert torch.equal(pre, Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_) + term)
                    assert torch.equal(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f_[ZL.LK_KEY], nq_, B)))
                    assert float((y1 - y0 - term).abs().max()) > 1e-4         # the propagation read the prior
                # 5. the prior reads the question's embedding and WALK only, never a row's score: from those two alone
                #    it is the same bit for bit, and so it is when every other weight moves every row's score; a
                #    question whose WALK or SEMB is dropped takes none, the others keep theirs
                only = {"WALK": f_["WALK"], "SEMB": (f_["SEMB"][0], None)}
                assert torch.equal(m1.qd_term(only, keep_, nq_, B), term)
                for k, p in m1.named_parameters():
                    if k not in QDP:
                        p.add_(torch.randn(p.shape, generator=gen) * 0.5)
                assert torch.equal(m1.qd_term(f_, keep_, nq_, B), term)
                assert not torch.equal(m1(f_, keep_, nq_, B, bz_), y1)
                q1 = nq_ == 1
                for j in (m1.j_walk, m1.j_semb):
                    k2 = keep_.clone()
                    k2[1, j] = 0.0
                    t2 = m1.qd_term(f_, k2, nq_, B)
                    assert float(t2[q1].abs().max()) == 0.0 and torch.equal(t2[~q1], term[~q1])
        # 6. the term is what round 11's QDepth adds to lean_gpu's model, bit for bit
        f3 = {b: fm[b] for b in blocks}
        nq3, bz3 = _nq, _bz
        torch.manual_seed(0)
        lm = LG.LeanMLP8D(blocks, widths, 16, dropout=0.1, seed=0).eval()
        torch.manual_seed(0)
        qm = QD.QDepth(blocks, widths, 16, dropout=0.1, seed=0).eval()
        torch.manual_seed(0)
        mq = ZRCKPQE(blocks, widths, 16, dropout=0.1, seed=0).eval()
        with torch.no_grad():
            g2 = torch.Generator().manual_seed(4)
            for k, sc in zip(QDP, (0.02, 0.5, 0.5)):
                x = torch.randn(getattr(mq, k).shape, generator=g2) * sc
                getattr(mq, k).copy_(x)
                getattr(qm, k).copy_(x)
            assert not LG.same_state(lm.state_dict(), {k: v for k, v in qm.state_dict().items() if k not in QDP})
            assert torch.equal(qm(f3, keep_, nq3, B, bz3), lm(f3, keep_, nq3, B, bz3) + mq.qd_term(f3, keep_, nq3, B))
        # 7. toy fits through lean_gpu's loop: a repeat is bit for bit and the prior learns; with the prior frozen at
        #    its zero start, each arm's fit is its base's bit for bit on every shared key
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpqe", c_m), ("zspkqe", c_g)):

            class Frozen(ARMS[arm][0]):
                def __init__(self, *a, **k):
                    super().__init__(*a, **k)
                    for p in (self.qd_u, self.qd_p, self.qd_b):
                        p.requires_grad_(False)

            S.ARMS[neutral] = (Frozen, ARMS[arm][1])
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            with S.patched(neutral):
                frz = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, prior frozen at zero")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            S.ARMS.pop(neutral)
            st = lambda f: f["states"] + [f["swa"]]
            assert all(not LG.same_state(a, b) for a, b in zip(st(fit), st(rep))), arm
            assert all(float(fit["swa"][k].abs().max()) > 0 for k in QDP), arm
            for a, b in zip(st(frz), st(base)):
                assert not LG.same_state({k: a[k] for k in b}, b), arm
                assert all(float(a[k].abs().max()) == 0.0 for k in QDP), arm
            assert any(LG.same_state({k: a[k] for k in b}, b) for a, b in zip(st(fit), st(base))), arm
    finally:
        S.ARMS.pop(neutral, None)
        shutil.rmtree(tmp, ignore_errors=True)
    # 8. the refusals: blocks without WALK or SEMB, a WALK of another width; train and read; the fit names give the split
    for bl, wd in ((["rank", "SEMB"], {"rank": 5, "SEMB": 64}), (["rank", "WALK"], {"rank": 5, "WALK": QD.WALK_W}),
                   (["rank", "SEMB", "WALK"], {"rank": 5, "SEMB": 64, "WALK": 15})):
        for cl in (ZRCKPQE, ZSPKQE):
            SR.must_stop(cl, bl, wd, 16)
    assert split_of_fit("scr-a8e-zspkqe-hp") == ("zspkqe", "L-hotpotqa") and split_of_fit("scr-a8-zspkq") == (None, None)
    good = ["train", "--arm", "zrckpqe", "--name", FIT["zrckpqe"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpqe"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zrckpq", "--name", FIT["zrckpqe"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkqe", "--name", FIT["zrckpqe"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkqe", "--name", FIT["zspkqe"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a8e-zspkqe", {"arm": "zspkr", "r1": {}}),
                        ("scr-a8e-zrckpqe", {"arm": "zrckpq", "a8": {"split": "L-musique"}}),
                        ("scr-a8e-zspkqe-hp", {"arm": "zspkqe", "a8e": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 9. relz's pair and re-call for each arm against its base arm's card fits, stamped A8e's; relz restored after
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
            assert pj["stage"] == "A8e" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A8e ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A8e's depth prior) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 10. the verdict, per arm, with the next form each outcome names; webqsp and metaqa beside, and musique's
        #     in-domain read
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpqe": fake_recall(T_ / "a.json", "zrckpqe", {mi: "GAIN"}),
                       "zspkqe": fake_recall(T_ / "b.json", "zspkqe", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpqe"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspkqe"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkqe"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["arms"]["zrckpqe"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["musique_in_domain_beside"] == {"zrckpqe": {"call": "GAIN", "delta": 0.0},
                                                   "zspkqe": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkqe"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpqe": fake_recall(T_ / "a.json", "zrckpqe", {}),
                       "zspkqe": fake_recall(T_ / "b.json", "zspkqe", {gm: "GAIN"}, stage="R1")}, T_ / "v")
        assert rec["arms"]["zrckpqe"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkqe"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpqe"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspkqe"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpqe": fake_recall(T_ / "a.json", "zrckpqe", {}),
                       "zspkqe": fake_recall(T_ / "b.json", "zspkqe", {("L-hotpotqa", "webqsp"): "LOSS"})}, T_ / "v")
        assert [rec["arms"][a]["verdict"] for a in ARM_NAMES] == ["NOT_ADOPTED", "NOT_ADOPTED"]
        assert all(rec["arms"][a]["next_form"] == NEXT["NO_GAIN"] for a in ARM_NAMES)
        assert "A8ex" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpqe": T_ / "none.json",
                       "zspkqe": fake_recall(T_ / "b.json", "zspkq", {gm: "GAIN"}, stage="A8",
                                             declared_in="docs/A8_QDEPTH_INPUT.md")}, T_ / "v")
        assert rec["arms"]["zrckpqe"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkqe"]["verdict"] == "INCOMPLETE"
        assert all(rec["arms"][a]["problems"] for a in ARM_NAMES)
        rec = verdict({"zrckpqe": fake_recall(T_ / "a.json", "zrckpqe", {}, rows=[]),
                       "zspkqe": fake_recall(T_ / "b.json", "zspkqe", {})}, T_ / "v")
        assert rec["arms"]["zrckpqe"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkqe"]["verdict"] == "NOT_ADOPTED"
    log(f"za8e selftest: the arms are zrckp's and zspk's models plus round 11's prior on their own carves; each starts "
        f"as its base bit for bit (state, global stream, scores in eval and training, every first-step gradient but the "
        f"prior's, which move); with the prior set zrckpqe is the base plus l_q at each row's class (a float64 hand "
        f"computation) and zspkqe's propagation reads it; the prior reads WALK and the question's embedding only, never "
        f"a row's score, and is what round 11's QDepth adds; toy fits repeat bit for bit, the prior learns, and frozen "
        f"at zero each fit is its base's; the refusals; relz's pair and re-call against each base arm's card fits, "
        f"stamped A8e's; the verdict per arm with its next form ({time.time() - t0:.1f}s): ok")
    return 0


# ── main ─────────────────────────────────────────────────────────────────────


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
            log(f"za8e {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za8e: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
