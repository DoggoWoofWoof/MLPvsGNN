"""A2 (docs/A2_RANKS_BESIDE_Z.md): pool ranks and top-50 statistics beside zret's z-scores, in both of the lean pair's
models. ztop50 (lean_screen4.py) and prank (lean_screen5.py) each REPLACED the within-pool z-scores, and each gained on
musique zero-shot (+0.139, +0.202) while metaqa in domain lost what the z-scores carry; neither was tried beside the
z-scores, nor on the GNN. A2 keeps every input zret reads and adds, per row, before any row is scored:

    per fixed block b (w columns)  z_top(b)   b's z-score against the pool's 50 best retrieved rows by rrf (ztop50's)
                                   hi(b), lo(b)  60 / (60 + min(rank, 50)) from the pool's top and bottom (prank's)
    SEMB (learned, 64 columns)     z_top(SEMB)
    rrf                            its z-score against the same top 50 (ztop50's base), a second base term

The columns enter l1's pre-activation through a2_w (hidden x columns, zero at the start), the base term through a2_bw
(zero at the start): each arm starts as its base bit for bit, and no random number is drawn. Nothing reads a score;
nothing is normalised over the question's scores (the within-pool statistics are of input columns, as zret's are).

    zrckpa  zrckp's model (outputs/mp_unified/zp1c3p.py) with A2's columns, on zrckp's carve
    zspka   zspk's model (outputs/mp_unified/zp1c3.py) with A2's columns; zsp's propagation reads the score they change

    python outputs/mp_unified/za2rank.py train --split L-musique --name scr-a2-zrckpa --arm zrckpa --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2rank.py read --name scr-a2-zrckpa --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za2rank.py compare --new outputs/screen/fits/scr-a2-zrckpa \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a2-zrckpa
    python outputs/mp_unified/za2rank.py pair --arm zrckpa --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za2rank.py recall --arm zrckpa --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za2rank.py verdict
    python outputs/mp_unified/za2rank.py --selftest
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
import lean_screen4 as S4  # noqa: E402  (ztop50: top_ref)
import lean_screen5 as S5  # noqa: E402  (prank: rank_inputs)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
LM = LG.LM
log = S.log

DECLARED = "docs/A2_RANKS_BESIDE_Z.md"
OUT = ROOT / "outputs" / "a2"
SCREEN = ROOT / "outputs" / "screen"
TOPK = S4.K
A2 = ("a2_w", "a2_bw")
SETTINGS = {"top_k": TOPK, "rank_constant": S5.KR, "rank_cap": S5.CAP, "rank_chunk": S5.COLS,
            "fixed_block": "z against the pool's top 50 retrieved rows by rrf, hi, lo (each times the block's keep)",
            "SEMB": "z against the same top 50", "base": "rrf's z against the same top 50, through a2_bw",
            "into": "l1's pre-activation through a2_w (hidden x columns)", "start": "a2_w and a2_bw zero",
            "score_input": False}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpa": "zrckp", "zspka": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a2-{a}", "L-hotpotqa": f"scr-a2-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpa": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspka": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpa": "zrckp with A2's ranks and top-50 columns beside its z-scores",
        "zspka": "zspk with A2's ranks and top-50 columns beside its z-scores, before its propagation"}
PAIR = {a: SCREEN / f"scr-a2-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a2-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa", "musique")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A2's re-call of the arm, or "
        "not twelve reads (docs/A2_RANKS_BESIDE_Z.md, section 4)")


# ── the columns ──────────────────────────────────────────────────────────────


def n_extra(blocks, widths):
    """A2's column count: three per fixed column, one per SEMB column."""
    return sum(int(widths[b]) if b == "SEMB" else 3 * int(widths[b]) for b in blocks)


def a2_block(raw, b, nq, B, top, m):
    """One block's A2 columns, times its keep m: [z_top] for SEMB, [z_top, hi, lo] for a fixed block."""
    zt = S3.seg_zscore_ref(raw, nq, B, top) * m
    if b == "SEMB":
        return [zt]
    with torch.no_grad():
        hi, lo = S5.rank_inputs(raw.detach(), nq, B)
    return [zt, hi * m, lo * m]


class A2Mix(S3.ZRet):
    """zret's forward with A2's columns beside (the module's docstring): into l1's pre-activation through a2_w, and
    rrf's top-50 z-score through a2_bw, both zero at the start."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.a2_n = n_extra(self.blocks, self.widths)
        self.a2_w = nn.Parameter(torch.zeros(self.l1.out_features, self.a2_n))
        self.a2_bw = nn.Parameter(torch.zeros(1))

    def a2_inputs(self, feats, keep, nq, B):
        """(zret's l1 input, A2's columns, rrf's z against the retrieved rows, rrf's z against the top 50)."""
        rrf = feats["rank"][:, S3.I_RRF]
        ref = S3.retrieved(feats)
        top = S4.top_ref(rrf, nq, B, TOPK)
        parts, extra = [], []
        for j, b in enumerate(self.blocks):
            m = keep[nq, j].unsqueeze(1)
            raw = S.raw_of(self, feats, b, nq)
            parts += [raw * m, S3.seg_zscore_ref(raw, nq, B, ref) * m, m]
            extra += a2_block(raw, b, nq, B, top, m)
        r1 = rrf.unsqueeze(1)
        bz = S3.seg_zscore_ref(r1, nq, B, ref).squeeze(1)
        bt = S3.seg_zscore_ref(r1, nq, B, top).squeeze(1)
        return torch.cat(parts, 1), torch.cat(extra, 1), bz, bt

    def forward(self, feats, keep, nq, B, base_z):
        x, e, bz, bt = self.a2_inputs(feats, keep, nq, B)
        h = self.drop(Fn.gelu(self.l1(x) + Fn.linear(e, self.a2_w)))
        h = self.drop(Fn.gelu(self.l2(h)))
        return self.base_w * bz + self.a2_bw * bt + self.out(h).squeeze(-1)


class ZRCKPA(Z3P.ZRCKP, A2Mix):
    """zrckpa: zrckp's model with A2's columns (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2Mix, ZRet)."""


class ZSPKA(Z3.ZSPK, A2Mix):
    """zspka: zspk's model with A2's columns; zsp's propagation reads the score they change (MRO: ZSPK, ZProp, KTMix,
    ZRM, ChainMatch, A2Mix, ZRet)."""


ARMS = {"zrckpa": (ZRCKPA, Z3.KTChainCarveZRC), "zspka": (ZSPKA, Z3.KTLinkCarveBase)}
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
        rec.update({"za2rank_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen4_sha256": LC.sha_src(S4.__file__),
                    "lean_screen5_sha256": LC.sha_src(S5.__file__), "a2": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "parameters": list(A2), "settings": SETTINGS,
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE), "kt_records": Z3.kt_records(Z3.train_carves(split)),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(Z2.CH_ZRC),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za2rank: train takes --arm zrckpa or zspka, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za2rank: A2's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za2rank: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
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
    if rec.get("arm") not in ARMS or "a2" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za2rank: {name} is not an A2 fit (arm {rec.get('arm')})")
    if rec["a2"].get("split") != split:
        raise SystemExit(f"za2rank: {name} is {split}'s fit, its record says {rec['a2'].get('split')}")
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
        raise SystemExit(f"za2rank: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za2rank: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A2's ranks beside) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A2 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A2", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za2rank_sha256": LC.sha_src(__file__)})
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
        "GAIN_AND_LOSS": "the back-off form, A2g: A2's contribution (a2_w's and a2_bw's terms) scaled by "
                         "2 sigmoid(<w, c> + b), zero-started so it starts at 1, c the pool's log size and retrieved "
                         "share (input columns only), its own declared file",
        "NO_GAIN": "the halves apart: the rank columns alone beside, and the top-50 columns alone beside, each on both "
                   "models, their own declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A2's screen verdict, per arm, from its re-call (section 4); webqsp, metaqa and musique beside, musique's in-domain
    read of both arms side by side, and the next form each outcome names (section 5)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A2" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A2's re-call of {a}")
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
    rec = {"stage": "A2", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za2rank_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A2's screen verdict: zrckpa **{per['zrckpa']['verdict']}**, zspka **{per['zspka']['verdict']}**", "",
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
           "", "This verdict closes only this form (the rank and top-50 columns beside the z-scores, into l1); "
               "docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za2rank verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A2", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def z_by_hand(x, nq, B, ref, eps=1e-6):
    """seg_zscore_ref by hand in float64: z against the pool's reference rows (population sd); a pool with fewer than
    two of them, or a column whose sd over them is under eps, takes the whole pool's z (0 where its sd is under eps)."""
    x, nq, ref = x.double().numpy(), nq.numpy(), ref.numpy() > 0
    out = np.zeros_like(x)
    for q in range(B):
        rows = np.flatnonzero(nq == q)
        if rows.size == 0:
            continue
        rr = rows[ref[rows]]
        for c in range(x.shape[1]):
            col = x[rows, c]
            if rr.size >= 2 and x[rr, c].std() >= eps:
                out[rows, c] = (col - x[rr, c].mean()) / x[rr, c].std()
            else:
                sd = col.std()
                out[rows, c] = 0.0 if sd < eps else (col - col.mean()) / sd
    return out


def selftest():
    import shutil
    import tempfile
    global TOPK
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models with A2Mix just above zret's forward; the base arms unchanged
    assert S.ARMS["zrckpa"] == (ZRCKPA, Z3.KTChainCarveZRC) and S.ARMS["zspka"] == (ZSPKA, Z3.KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZM.S3 is S3 and S4.S3 is S3 and S5.S is S and TOPK == 50 == S4.K and (S5.KR, S5.CAP) == (60.0, 50)
    assert ZRCKPA.__mro__[1:7] == (Z3P.ZRCKP, Z3P.KTMixP, ZM.ZRM, RM.ChainMatch, A2Mix, S3.ZRet)
    assert ZSPKA.__mro__[1:8] == (Z3.ZSPK, ZP.ZProp, Z3.KTMix, ZM.ZRM, RM.ChainMatch, A2Mix, S3.ZRet)
    rng = np.random.default_rng(58)
    tmp = Path(tempfile.mkdtemp(prefix="za2_"))
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
        # 2. the columns against references by hand: hi and lo are prank's ranks (lean_screen5.rank_numpy), z_top the
        #    z-score against ztop50's reference (lean_screen4.top_ref_numpy); at k = 50 (pools under 50 retrieved
        #    rows) z_top is zret's z; at k = 3 it differs on pools with more than three retrieved rows
        torch.manual_seed(0)
        m = ZRCKPA(blocks, c_m.widths, 16, dropout=0.1, seed=0)
        assert m.a2_n == n_extra(blocks, c_m.widths) == 3 * 5 + 64 and tuple(m.a2_w.shape) == (16, 79)
        rrf = f_["rank"][:, S3.I_RRF]
        ret = S3.retrieved(f_)
        for k in (50, 3):
            TOPK = k
            try:
                x, e, bz, bt = m.a2_inputs(f_, keep_, nq_, B)
            finally:
                TOPK = 50
            top = torch.from_numpy(S4.top_ref_numpy(rrf, nq_, B, k)).to(rrf.dtype)
            assert torch.equal(S4.top_ref(rrf, nq_, B, k), top)
            raw_r = f_["rank"]
            semb = S.raw_of(m, f_, "SEMB", nq_).detach()
            m1 = keep_[nq_, 1].unsqueeze(1).numpy()
            want = np.concatenate([z_by_hand(raw_r, nq_, B, top),
                                   S5.KR / (S5.KR + np.minimum(S5.rank_numpy(raw_r, nq_, True), S5.CAP)),
                                   S5.KR / (S5.KR + np.minimum(S5.rank_numpy(raw_r, nq_, False), S5.CAP)),
                                   z_by_hand(semb, nq_, B, top) * m1], 1)
            assert e.shape == (nq_.numel(), 79) and bool(torch.isfinite(e).all())
            assert np.allclose(e.detach().double().numpy(), want, atol=1e-4), float(np.abs(e.detach().numpy() - want).max())
            assert np.allclose(bt.detach().double().numpy(), z_by_hand(rrf.unsqueeze(1), nq_, B, top)[:, 0], atol=1e-5)
            zr = torch.cat([S3.seg_zscore_ref(raw_r, nq_, B, ret),
                            S3.seg_zscore_ref(semb, nq_, B, ret) * torch.from_numpy(m1)], 1)
            same = torch.allclose(torch.cat([e[:, :5], e[:, 15:]], 1).detach(), zr, atol=1e-6)
            many = bool((torch.zeros(B).index_add_(0, nq_, ret) > k).any())
            assert same == (not many), (k, same, many)
            assert float(e[nq_ == 3, 15:].detach().abs().max()) == 0.0
        # 3. each arm starts as its base bit for bit (same seed, no random number drawn; its scores and every shared
        #    parameter's gradient); with a2_w and a2_bw moved, the scorer is zret's on A2's inputs, and zspka's
        #    propagation reads that score
        for arm, cls, base, cv in (("zrckpa", ZRCKPA, Z3P.ZRCKP, c_m), ("zspka", ZSPKA, Z3.ZSPK, c_g)):
            f2, nq2, bz2, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            r0 = torch.get_rng_state()
            torch.manual_seed(0)
            m1_ = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            assert torch.equal(torch.get_rng_state(), r0)
            s0, s1 = m0.state_dict(), m1_.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(A2) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0}) and all(float(s1[k].abs().max()) == 0 for k in A2)
            if arm == "zspka":
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
                # out starts at zero: a2_w has no gradient at the start, as l1 has none; a2_bw has one
                assert float(m1_.a2_bw.grad.abs().max()) > 0 and float(m1_.a2_w.grad.abs().max()) == 0.0
                assert float(m1_.l1.weight.grad.abs().max()) == 0.0
            m1_.eval()
            with torch.no_grad():
                m1_.a2_w.copy_(torch.randn(m1_.a2_w.shape, generator=torch.Generator().manual_seed(9)) * 0.1)
                m1_.a2_bw.fill_(0.3)
                x, e, bz, bt = m1_.a2_inputs(f2, keep_, nq2, B)
                h = Fn.gelu(m1_.l1(x) + e @ m1_.a2_w.T)
                want = m1_.base_w * bz + 0.3 * bt + m1_.out(Fn.gelu(m1_.l2(h))).squeeze(-1)
                sc = A2Mix.forward(m1_, f2, keep_, nq2, B, bz2)
                assert torch.allclose(sc, want, atol=1e-5) and not torch.allclose(sc, S3.ZRet.forward(m1_, f2, keep_, nq2,
                                                                                                       B, bz2))
                pre = super(ZP.ZProp, m1_).forward(f2, keep_, nq2, B, bz2) if arm == "zspka" else None
                y1 = m1_(f2, keep_, nq2, B, bz2)
                if arm == "zrckpa":
                    assert torch.allclose(y1, sc, atol=1e-6)                   # toy carves: no KT input, chain gate 0
                else:
                    assert torch.allclose(pre, sc, atol=1e-6)
                    assert torch.allclose(y1, pre + m1_.prop_head(ZP.prop_inputs(pre, f2[ZL.LK_KEY], nq2, B)),
                                          atol=1e-5)
                    assert float((y1 - sc).abs().max()) > 1e-4
        # 4. toy fits through lean_gpu's loop: a repeat is bit-identical and A2's weights learn; no base arm's model is
        #    an A2Mix (A2 changes no base class)
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv in (("zrckpa", c_m), ("zspka", c_g)):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["a2_w"].abs().max()) > 0 and float(fit["swa"]["a2_bw"].abs().max()) > 0
            assert not issubclass(S.ARMS[BASE_OF[arm]][0], A2Mix)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 5. the refusals; the fit names give the split
    assert split_of_fit("scr-a2-zspka-hp") == ("zspka", "L-hotpotqa") and split_of_fit("scr-p5c-zspko") == (None, None)
    good = ["train", "--arm", "zrckpa", "--name", FIT["zrckpa"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpa"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspka", "--name", FIT["zrckpa"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspka", "--name", FIT["zspka"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a2-zspka", {"arm": "zspko", "p5c": {}}),
                        ("scr-a2-zspka-hp", {"arm": "zspka", "a2": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 6. relz's pair and re-call for each arm against its base arm's card fits, stamped A2's; relz restored after
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
            assert pj["stage"] == "A2" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A2 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A2's ranks beside) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 7. the verdict, per arm, with the next form of section 5: gains with a loss -> A2g, no gain -> the halves
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpa": fake_recall(T_ / "a.json", "zrckpa", {mi: "GAIN"}),
                       "zspka": fake_recall(T_ / "b.json", "zspka", {gm: "GAIN", ("L-hotpotqa", "metaqa"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpa"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zrckpa"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["arms"]["zspka"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspka"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["musique_in_domain_beside"] == {"zrckpa": {"call": "GAIN", "delta": 0.0},
                                                   "zspka": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspka"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpa": fake_recall(T_ / "a.json", "zrckpa", {}),
                       "zspka": fake_recall(T_ / "b.json", "zspka", {gm: "GAIN"}, stage="P5c")}, T_ / "v")
        assert rec["arms"]["zrckpa"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspka"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpa"]["next_form"] == NEXT["NO_GAIN"]
        assert rec["arms"]["zspka"]["next_form"] == NEXT["INCOMPLETE"]
        rec = verdict({"zrckpa": T_ / "none.json", "zspka": fake_recall(T_ / "b.json", "zrckpa", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpa"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspka"]["verdict"] == "INCOMPLETE"
    log(f"za2rank selftest: the arms are zrckp's and zspk's models with A2Mix just above zret's forward; the columns "
        f"match references by hand (prank's ranks, ztop50's z-scores, at k 50 and 3; a question without SEMB has none); "
        f"each arm starts as its base bit for bit, scores and every shared gradient, drawing no random number; moved, "
        f"the scorer is zret's on A2's inputs and zspka's propagation reads it; toy fits repeat bit for bit and A2's "
        f"weights learn; train refuses another arm, split, fit name, cache root or device; read refuses a fit that is "
        f"not A2's or not its split's; relz's pair and re-call per arm, stamped A2's; the verdict's outcomes and next "
        f"forms ({time.time() - t0:.1f}s): ok")
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
            log(f"za2rank {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za2rank: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
