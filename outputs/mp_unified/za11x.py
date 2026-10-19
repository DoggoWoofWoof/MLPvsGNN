"""A11x (docs/A11X_FOURTH_FAMILY_ZSPX.md): A11's fourth link family in all three of G5's zspx steps. A11 put our
LLM-free phrase links (two passages linked when they share a phrase that 2 to 25 passages hold) into zspk's one
propagation step as a fourth family: NOT_ADOPTED with no GAIN (one LOSS, webqsp zero-shot on L-musique, a KB where
zspk4 is zspk by rule). Its section 5 fixed this form for that outcome: the family in each step of zspx, the adopted
GNN, where depth paid. Nothing else changes:

    zspx4   zspx (zg5mix.ZSPX) whose three step heads each also read the fourth family's three columns per row
            (A11's: the mean and soft maximum of the neighbours' z-scores and log1p of the degree), before the GELU:
                s1 = s0 + W2 gelu(W1 x10(s0) + b1 + W4sp x4(s0)) + b2     zspx's step: x4 keeps the gradient, as x10
                s2 = s1 + W2 gelu(W1 r10(s1) + b1 + W4rs x4(s1)) + b2     G3's residual step: no gradient
                s3 = s2 + W2 gelu(W1 x10(s2) + b1 + W4p2 x4(s2)) + b2     G4's second step: no gradient
            (W1, b1, W2, b2 are each head's own.) Each W4 is 32 x 3, from its own generator (seed + 1101, A11's
            draw, for the first; seed + 1102 and + 1103 for the others), at its head's W1 scale: 288 parameters. A
            row with no fourth-family neighbour reads x4 = 0, so at the start (every head's last layer is zero), on
            the KBs and on questions the family does not reach, zspx4 is zspx bit for bit.

The family's edges are A11's build (za11p4.py build, unchanged), beside G3's carve (zspx's).

    python outputs/host_ops/pylib_run.py outputs/mp_unified/za11p4.py build --dataset musique --carve s1fit \\
        --part 0 --host
    python outputs/mp_unified/za11x.py train --split L-musique --name scr-a11x-zspx4 --arm zspx4 --device cuda --host \\
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za11x.py read --name scr-a11x-zspx4 --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za11x.py compare --new outputs/screen/fits/scr-a11x-zspx4 \\
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a11x-zspx4
    python outputs/mp_unified/za11x.py pair --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za11x.py recall --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za11x.py verdict
    python outputs/mp_unified/za11x.py --selftest
The re-call's base screens are A9's (outputs/a9/base-zspx-<split>.json: zspx's card fits against step 1's).
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

import zg5mix as G5  # noqa: E402  (zspx, its carve, its fits; through it zg3res, zg4prop, zp1c3, zprop, zlink)
import za11p4 as ZA  # noqa: E402  (A11's fourth family: its build, records, carve, columns and toy helpers)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

G3, G4, Z3, Z2 = G5.G3, G5.G4, G5.Z3, G5.Z2
ZP, ZL, ZM, RM = G5.ZP, G5.ZL, G5.ZM, G5.RM
Z, R, S2, S, LG, LC, SR = G5.Z, G5.R, G5.S2, G5.S, G5.LG, G5.LC, G5.SR
log = S.log

DECLARED = "docs/A11X_FOURTH_FAMILY_ZSPX.md"
OUT = ROOT / "outputs" / "a11x"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM, BASE_ARM = "zspx4", "zspx"
HEADS = ("sp", "rs", "p2")                          # zspx's step (zspkg's), G3's residual step, G4's second step
W4 = tuple(f"{h}_w4" for h in HEADS)
W4_HID = (ZP.HID, G3.RS_HID, G4.P2_HID)             # each head's hidden width (32)
W4_FAN = (ZP.N_IN, G3.RS_IN, G4.N_IN)               # each head's W1 fan-in (10): W4 is drawn at its head's W1 scale
W4_SEED_OFF = (ZA.P4_SEED_OFF, 1102, 1103)          # the first is A11's own draw
FIT = {"L-musique": "scr-a11x-zspx4", "L-hotpotqa": "scr-a11x-zspx4-hp"}
BASE_FITS = {"L-musique": ("screen", "fits", "scr-g5-zspx"), "L-hotpotqa": ("screen", "fits", "scr-g5-zspx-hp")}
BASE_SCREENS = {sp: f"outputs/a9/base-{BASE_ARM}-{sp}.json" for sp in SPLITS}
AGAINST = "zspx's card fit of each split under lean_gpu's loss (scr-g5-zspx, scr-g5-zspx-hp)"
WHAT = "zspx whose three steps also read our LLM-free phrase links as a fourth family"
SETTINGS = {"base": "zspx (outputs/mp_unified/zg5mix.py), every setting, seed, batch and head of it",
            "family": "A11's fourth family, unchanged: its build, edges and carve (outputs/mp_unified/za11p4.py)",
            "columns": ZA.SETTINGS["columns"],
            "steps": {"sp": "zspx's step (zspkg's form): the columns of s0, with the gradient kept as its own inputs "
                            "keep it",
                      "rs": "G3's residual step: the columns of s1, no gradient (G3's own residual columns need each "
                            "edge's bridge similarity, which G3 builds for zlink's edges only)",
                      "p2": "G4's second step: the columns of s2, no gradient"},
            "where": "inside each step's head, before its GELU: gelu(W1 x + b1 + W4 x4)",
            "weights": {nm: {"shape": [h, ZA.P4_IN], "seed_offset": off, "init": f"uniform in +-1/sqrt({fan})"}
                        for nm, h, fan, off in zip(W4, W4_HID, W4_FAN, W4_SEED_OFF)},
            "parameters_added": sum(h * ZA.P4_IN for h in W4_HID)}
PAIR = SCREEN / "scr-a11x-zspx4-pair"
RECALL = SCREEN / "scr-a11x-zspx4-pair-recall"
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
RULE = ("ADOPT_FOR_FULL_RUN when zspx4's re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A11x's, or not twelve "
        "reads (docs/A11X_FOURTH_FAMILY_ZSPX.md, section 4)")
A11M = "the family's per-row degree (log1p; no score in it) as a fixed column through R1's pointwise head"
NEXT = {"ADOPT_FOR_FULL_RUN": "zspx4 fitted again on P1g-b's pools (A11's build run on them first), then its full run, "
                              f"its own declared file; and A11m for the MLP, {A11M} in zrckp (as zrckpr reads R1's "
                              "columns), its own declared file",
        "GAIN_AND_LOSS": "A11xd, the back-off form: zspx4 under edge-family dropout over its four families (FD 0.25, "
                         "qd_gnn13's fd25 rule through zg2.family_keep, as A9 ran it on zspx), so no step can lean on "
                         f"the fourth family where its mix differs; and A11m for the MLP, {A11M} in zrckp; one "
                         "declared file",
        "NO_GAIN": f"A11m in both models, {A11M} in zrckp (as zrckpr reads R1's columns) and in zspx (as zspxr "
                   "does), each free to learn how much to use it; one declared file. This closes zspx4's form only: "
                   "as a message the family reached zspk's one step (A11) and all three of zspx's steps (A11x) with "
                   "no GAIN, so it goes on as a fixed input",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def p4_inputs_grad(s, p4, nq, B):
    """(N, 3): za11p4.p4_inputs with the gradient kept, zg4prop.prop_inputs_grad's own code (zspx's first step's) with
    every edge in its first family."""
    x = G4.prop_inputs_grad(s, {"u": p4["u"], "v": p4["v"], "f": torch.zeros_like(p4["u"])}, nq, B)
    return x[:, [0, ZP.FAMS, 2 * ZP.FAMS]]


class ZSPX4(G5.ZSPX):
    """zspx4: zspx whose three step heads also read the fourth family's three columns (a W4 each, from its own
    generator)."""

    cols1 = staticmethod(p4_inputs_grad)        # the first step's columns keep the gradient, as that step's inputs do

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        for nm, h, fan, off in zip(W4, W4_HID, W4_FAN, W4_SEED_OFF):
            g = torch.Generator().manual_seed(int(seed) + off)
            setattr(self, nm, nn.Parameter((torch.rand(h, ZA.P4_IN, generator=g) * 2 - 1) / fan ** 0.5))

    def head4(self, h, x, x4):
        """Head h (sp, rs or p2) over its own inputs x, plus its W4 over the fourth family's columns x4 before its
        GELU; with x4 None it is zspx's head, op for op."""
        a = Fn.linear(x, getattr(self, f"{h}_w1"), getattr(self, f"{h}_b1"))
        if x4 is not None:
            a = a + Fn.linear(x4, getattr(self, f"{h}_w4"))
        return Fn.linear(Fn.gelu(a), getattr(self, f"{h}_w2"), getattr(self, f"{h}_b2")).squeeze(-1)

    def forward(self, feats, keep, nq, B, base_z):
        s = super(ZP.ZProp, self).forward(feats, keep, nq, B, base_z)       # what ZProp.forward's super() returns
        if ZL.LK_KEY not in feats:
            return s
        lk, p4 = feats[ZL.LK_KEY], feats.get(ZA.P4_KEY)
        s = s + self.head4("sp", G4.prop_inputs_grad(s, lk, nq, B), None if p4 is None else self.cols1(s, p4, nq, B))
        if G3.RES_KEY in feats:
            s = s + self.head4("rs", G3.res_inputs(s, lk, feats[G3.RES_KEY], nq, B),
                               None if p4 is None else ZA.p4_inputs(s, p4, nq, B))
        return s + self.head4("p2", ZP.prop_inputs(s, lk, nq, B), None if p4 is None else ZA.p4_inputs(s, p4, nq, B))


P4ResKTLinkCarveBase = ZA.p4_carve(G3.ResKTLinkCarveBase)
ARMS = {ARM: (ZSPX4, P4ResKTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"za11x_sha256": LC.sha_src(__file__), "za11p4_sha256": LC.sha_src(ZA.__file__),
                    "zg5mix_sha256": LC.sha_src(G5.__file__), "zg3res_sha256": LC.sha_src(G3.__file__),
                    "zg4prop_sha256": LC.sha_src(G4.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "a11x": {
                        "declared_in": DECLARED, "arm": ARM, "base_arm": BASE_ARM, "device": Z2.DEVICE,
                        "settings": SETTINGS, "build": ZA.BUILD_SETTINGS, "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "p4_records": ZA.p4_records(Z3.train_carves(split)),
                        "res_records": G3.res_records(Z3.train_carves(split)),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm != ARM or S.ARMS.get(arm) != ARMS[ARM]:
        raise SystemExit(f"za11x: train takes --arm {ARM}, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za11x: A11x's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[split]:
        raise SystemExit(f"za11x: zspx4's fit of {split} is {FIT[split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    G3.res_records(Z3.train_carves(split))
    ZA.p4_records(Z3.train_carves(split))
    with Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") != ARM or "a11x" not in rec:
        raise SystemExit(f"za11x: {name} is not an A11x fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    G3.res_records(Z3.eval_carves())
    ZA.p4_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, zspx4 decided against zspx's card fits ──────────


@contextlib.contextmanager
def on_base():
    def base_fit(split):
        if split not in BASE_FITS:
            raise SystemExit(f"za11x: no base fit of {BASE_ARM} on {split}")
        return BASE_FITS[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, BASE_FITS, BASE_SCREENS, base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zspx's card fits scr-g5-zspx and scr-g5-zspx-hp)"),
       ("(rel's screen fits)", "(zspx's card fits)"),
       ("| rel R@5 | relz R@5 |", f"| zspx R@5 | zspx4 R@5 ({WHAT}) |"),
       ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(f"A11x ({DECLARED}): zspx4 ({WHAT}) against zspx.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A11x", "declared_in": DECLARED, "decided_against": AGAINST, "device": Z2.DEVICE,
                    "za11x_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_base():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, base_screens=None):
    with on_base():
        rec = Z.recall(null_files, pair_file, out, base_screens)
    restamp(out)
    return rec


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def verdict(recall_file=None, out=None):
    """A11x's screen verdict from zspx4's re-call (section 4); webqsp and metaqa beside; the next form (section 5)."""
    rp, problems, rows = Path(recall_file or RECALL.with_suffix(".json")), [], []
    if not rp.exists():
        problems.append(f"{rp.name} missing")
    else:
        rec = json.loads(rp.read_text(encoding="utf-8"))
        if rec.get("stage") != "A11x" or rec.get("declared_in") != DECLARED or rec.get("arm") != ARM:
            problems.append(f"{rp.name} is not A11x's re-call of {ARM}")
        rows = rec.get("rows", [])
        if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
            problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
    gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
    losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    out_rec = {"stage": "A11x", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "gains": gains,
               "losses": losses, "problems": problems, "next_form": nxt,
               "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
               "beside": {tag(r): {"call": r.get("call"), "delta": r.get("delta")} for r in rows
                          if r.get("dataset") in BESIDE},
               "recall_sha256": LC.sha_file(rp) if rp.exists() else None,
               "za11x_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

    def fmt(d):
        return f"{d:+.4f}" if isinstance(d, (int, float)) else "n/a"

    md = [f"# A11x's screen verdict: zspx4 **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zspx4** ({WHAT}) against {AGAINST}, after the seed-null re-call: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ". Beside (the KBs, where zspx4 is zspx by "
          "rule): " + (", ".join(f"{k} {b['call']} ({fmt(b['delta'])})" for k, b in out_rec["beside"].items())
                       or "none") + ".",
          f"- **Next form** (fixed in section 5 before any number): {nxt}.", "",
          "R@5 difference against zspx per read:", "", "| read | zspx4 |", "| --- | --- |"]
    md += [f"| {tag(r)} | {fmt(r.get('delta'))} {r.get('call')} |" for r in rows]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), out_rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"za11x verdict: {v}; next: {nxt}")
    return out_rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, calls, stage="A11x", arm=ARM, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": stage, "declared_in": DECLARED if stage == "A11x" else "x", "arm": arm, "verdict": "x",
           "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def same(x, y):
    """Two batch values equal: tensors bit for bit (dtype too), dicts key by key."""
    if isinstance(x, dict) or isinstance(y, dict):
        return (isinstance(x, dict) and isinstance(y, dict) and sorted(x) == sorted(y)
                and all(same(x[k], y[k]) for k in x))
    if torch.is_tensor(x) or torch.is_tensor(y):
        return torch.is_tensor(x) and torch.is_tensor(y) and x.dtype == y.dtype and torch.equal(x, y)
    if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
        return (isinstance(x, np.ndarray) and isinstance(y, np.ndarray) and x.dtype == y.dtype
                and np.array_equal(x, y))
    if isinstance(x, (list, tuple)) or isinstance(y, (list, tuple)):
        return type(x) is type(y) and len(x) == len(y) and all(same(a, b) for a, b in zip(x, y))
    return x == y


def set_w2(models, moved, w):
    """Each head's last layer: w's for the heads in moved, zero for the others, the same in every model given."""
    with torch.no_grad():
        for m in models:
            for h in HEADS:
                getattr(m, f"{h}_w2").copy_(w[h] if h in moved else torch.zeros_like(w[h]))


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arm: zspx plus three W4s on G3's carve with A11's fourth family beside; zspx's, zspk4's and zspk's arms
    #    unchanged; the declared constants
    assert S.ARMS[ARM] == (ZSPX4, P4ResKTLinkCarveBase) and ZSPX4.__mro__[1] is G5.ZSPX and ZA.Z3 is Z3
    assert P4ResKTLinkCarveBase.__mro__[1] is G3.ResKTLinkCarveBase and P4ResKTLinkCarveBase.P4_ROOT == ZA.P4_DIR
    assert S.ARMS[BASE_ARM] == (G5.ZSPX, G3.ResKTLinkCarveBase) and S.ARMS["zspk4"] == (ZA.ZSPK4, ZA.P4KTLinkCarveBase)
    assert S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert W4_HID == (32, 32, 32) and W4_FAN == (10, 10, 10) and SETTINGS["parameters_added"] == 288
    assert len(set(W4_SEED_OFF)) == 3 and not set(W4_SEED_OFF) & {ZP.SEED_OFF, Z3.KT_SEED_OFF, G3.RS_SEED_OFF,
                                                                  G4.P2_SEED_OFF}
    tmp = Path(tempfile.mkdtemp(prefix="za11x_"))
    try:
        rng = np.random.default_rng(1112)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        T, _, _ = G3.toy_embed(tmp, rng)
        # the toy cache's rows get passages (step 1's node array, written before any build ties to the records); the
        # rows of every third question only passages the P form links to none, so some questions have no edge
        ip, col, counts, a = ZA.toy_csr(rng, 32, 0.15)
        NN = 40
        ip, a, counts = np.r_[ip, np.full(NN - 32, ip[-1])], np.pad(a, (0, NN - 32)), dict(counts, passages=NN)
        nodes_of, q0 = {}, 0
        for p in sorted((tmp / "cache" / "metaqa" / "toy").iterdir()):
            r = json.loads((p / "record.json").read_text(encoding="utf-8"))
            n = np.load(p / "n.npy")
            nd = np.concatenate([rng.integers(32, NN, k) if (q0 + j) % 3 == 0 else rng.integers(0, 32, k)
                                 for j, k in enumerate(n.tolist())]).astype(np.int32)
            r["arrays"]["node"] = {"sha256": R.save_npy(p, "node", nd)}
            R.write_json(p / "record.json", r)
            nodes_of[p.name], q0 = nd, q0 + n.size
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(out_root=tmp / "res", cache_root=tmp / "cache", look_root=tmp / "look", lk_root=tmp / "lk",
                  freeze="toy")
        G3.build("metaqa", "toy", 0, nodes=G3.ToyNodes(T), **kw)
        G3.build("metaqa", "toy", 1, nodes=G3.ToyNodes(T), **kw)

        class ToyIds(ZM.ToyZ):
            def __init__(self, *a_, **k):
                super().__init__(*a_, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        cls_lk = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        Z3.toy_kt(tmp / "kt", cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"), rng,
                  cache_root=str(tmp / "cache"))
        cls_res = G3.res_carve(Z3.kt_carve(cls_lk, tmp / "kt"), tmp / "res")        # zspx's carve
        cls = ZA.p4_carve(cls_res, tmp / "p4")                                       # zspx4's
        with ZA.passages(ZA.PASSAGE + ("metaqa",)):
            ZA.build("metaqa", "toy", out_root=tmp / "p4", cache_root=tmp / "cache", csr=(ip, col, counts))
            # 2. the host's state, records without arrays: the records pass, the carve cannot load; A11's build run
            #    again over the existing record writes the arrays back byte for byte and rewrites the record
            d0 = tmp / "p4" / "metaqa" / "toy" / "part_0of2"
            before = {f.name: LC.sha_file(f) for f in sorted(d0.glob("*.npy"))}
            assert sorted(before) == ["e_u.npy", "e_v.npy", "q_ne.npy"]
            for f in d0.glob("*.npy"):
                f.unlink()
            assert set(ZA.p4_records([("metaqa", "toy")], tmp / "p4", tmp / "cache")) == {
                f"metaqa/toy/{k}" for k in nodes_of}
            try:
                cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
                raise AssertionError("the carve loaded a part whose arrays are gone")
            except FileNotFoundError:
                pass
            ZA.build("metaqa", "toy", 0, out_root=tmp / "p4", cache_root=tmp / "cache", csr=(ip, col, counts))
            rec0 = json.loads((d0 / "record.json").read_text(encoding="utf-8"))
            assert {f.name: LC.sha_file(f) for f in sorted(d0.glob("*.npy"))} == before
            assert {f"{k}.npy": x["sha256"] for k, x in rec0["arrays"].items()} == before
            c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        c_kb = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")                     # metaqa as the KB it is
        c_res = cls_res("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert c_kb.p4["e_u"].size == 0 and c_kb.p4["records"] == {}
        # 3. the carve: zspx's batch, every key unchanged, with the fourth family's edges beside; those are the brute
        #    force's on the node arrays; none on the KB
        node = np.concatenate([nodes_of[k] for k in sorted(nodes_of)])
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, gold = c.batch(qs, blocks)
        fr, nq_r, bz_r, gold_r = c_res.batch(qs, blocks)
        assert sorted(feats) == sorted(list(fr) + [ZA.P4_KEY]) and all(same(feats[k], fr[k]) for k in fr)
        assert same(nq, nq_r) and same(bz, bz_r) and same(gold, gold_r)
        assert ZL.LK_KEY in feats and Z3.KT_KEY in feats and G3.RES_KEY in feats
        p4 = feats[ZA.P4_KEY]
        cnt = c.n_np[qs]
        rseg = np.cumsum(cnt) - cnt
        want = []
        for i, q in enumerate(qs):
            lo, hi = int(c.off_np[q]), int(c.off_np[q + 1])
            want += [(int(rseg[i]) + x, int(rseg[i]) + y) for x, y in ZA.by_hand(a, node[lo:hi])]
        assert list(zip(p4["u"].tolist(), p4["v"].tolist())) == want and p4["u"].numel() > 0
        feats_kb, nq_kb, bz_kb, _ = c_kb.batch(qs, blocks)
        assert feats_kb[ZA.P4_KEY]["u"].numel() == 0 and same(nq_kb, nq) and same(bz_kb, bz)
        has = torch.zeros(nq.numel(), dtype=torch.bool)
        has[p4["v"]] = True                                 # rows with a fourth-family neighbour
        q_has = torch.zeros(B, dtype=torch.bool)
        q_has[nq[p4["v"]]] = True
        none_q = ~q_has[nq]                                 # rows of the questions the family does not reach
        assert bool(none_q.any()) and bool((~has & ~none_q).any()), "the toy needs both kinds of question and row"
        keep = torch.ones(B, len(blocks))
        ms = {}
        for k, cl in (("zspx", G5.ZSPX), ("zspx4", ZSPX4), ("zspk4", ZA.ZSPK4)):
            torch.manual_seed(0)
            ms[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        # 4. zspx4's state is zspx's plus the three W4s, each from its own generator at its head's W1 scale; the
        #    first is A11's own draw (zspk4's p4_w)
        sx, s4, sk = (ms[k].state_dict() for k in ("zspx", "zspx4", "zspk4"))
        assert sorted(set(s4) - set(sx)) == sorted(W4) and not LG.same_state({k: s4[k] for k in sx}, sx)
        for nm, h, fan, off in zip(W4, W4_HID, W4_FAN, W4_SEED_OFF):
            g = torch.Generator().manual_seed(off)
            assert torch.equal(s4[nm], (torch.rand(h, ZA.P4_IN, generator=g) * 2 - 1) / fan ** 0.5), nm
        assert torch.equal(s4["sp_w4"], sk["p4_w"]) and not torch.equal(s4["rs_w4"], s4["sp_w4"])
        assert sum(s4[k].numel() for k in W4) == SETTINGS["parameters_added"]
        g = torch.Generator().manual_seed(9)
        w = {h: torch.randn(1, 32, generator=g) for h in HEADS}
        with torch.no_grad():
            # 5. at the start zspx4 is zspx; with one head's last layer moved it moves only the rows with a
            #    fourth-family neighbour, and with zspx's step alone moved it is A11's zspk4 on the same weights
            y0 = ms["zspx"](feats, keep, nq, B, bz)
            assert torch.equal(ms["zspx4"](feats, keep, nq, B, bz), y0)
            for h in HEADS:
                set_w2((ms["zspx"], ms["zspx4"]), (h,), w)
                yx, y4 = ms["zspx"](feats, keep, nq, B, bz), ms["zspx4"](feats, keep, nq, B, bz)
                assert not torch.equal(y4, yx) and torch.equal(y4[~has], yx[~has]), h
                if h == "sp":
                    ms["zspk4"].sp_w2.copy_(w[h])
                    assert torch.equal(y4, ms["zspk4"](feats, keep, nq, B, bz))
            # with all three moved: the questions the family does not reach are zspx's bit for bit, a row without a
            # neighbour in a question it reaches moves (through its question's z-scores in the later steps); on the KB,
            # without the family's key and without links zspx4 is zspx
            set_w2((ms["zspx"], ms["zspx4"]), HEADS, w)
            yx, y4 = ms["zspx"](feats, keep, nq, B, bz), ms["zspx4"](feats, keep, nq, B, bz)
            assert torch.equal(y4[none_q], yx[none_q]) and bool((y4[~has & ~none_q] != yx[~has & ~none_q]).any())
            assert torch.equal(ms["zspx4"](feats_kb, keep, nq, B, bz), yx)
            assert torch.equal(ms["zspx4"]({k: x for k, x in feats.items() if k != ZA.P4_KEY}, keep, nq, B, bz), yx)
            no_lk = {k: x for k, x in feats.items() if k not in (ZL.LK_KEY, G3.RES_KEY, ZA.P4_KEY)}
            assert torch.equal(ms["zspx4"](no_lk, keep, nq, B, bz), ms["zspx"](no_lk, keep, nq, B, bz))
            # 6. the three columns against A11's float64 brute force; rows without a neighbour read 0; the first step's
            #    gradient-kept columns are the same numbers
            s = torch.randn(nq.numel(), generator=g)
            x4 = ZA.p4_inputs(s, p4, nq, B)
            assert torch.allclose(x4.double(), ZA.p4_by_hand(s, p4["u"], p4["v"], nq, B), atol=1e-4)
            assert bool((x4[~has] == 0).all()) and torch.equal(p4_inputs_grad(s, p4, nq, B), x4)
        # 7. the gradient: the first step's columns carry it to the score (A11's do not); in the model it reaches the
        #    scorer through them, and on the KB the scorer's gradient is zspx's
        sg = torch.randn(nq.numel(), generator=torch.Generator().manual_seed(4)).requires_grad_(True)
        xg = p4_inputs_grad(sg, p4, nq, B)
        assert xg.requires_grad and not ZA.p4_inputs(sg, p4, nq, B).requires_grad
        xg[:, :2].sum().backward()
        assert float(sg.grad.abs().max()) > 0

        class NoGrad1(ZSPX4):
            cols1 = staticmethod(ZA.p4_inputs)

        torch.manual_seed(0)
        mng = NoGrad1(blocks, c.widths, 16, dropout=0.1, seed=0).eval()
        mng.load_state_dict(ms["zspx4"].state_dict())
        gw = torch.randn(nq.numel(), generator=torch.Generator().manual_seed(11))
        ga = G5.scorer_grads(ms["zspx4"], feats, keep, nq, B, bz, gw)
        gb = G5.scorer_grads(mng, feats, keep, nq, B, bz, gw)
        with torch.no_grad():
            assert torch.equal(mng(feats, keep, nq, B, bz), y4)
        assert sorted(ga) == sorted(gb) and any(float((ga[k] - gb[k]).abs().max()) > 1e-6 for k in ga)
        gk4 = G5.scorer_grads(ms["zspx4"], feats_kb, keep, nq, B, bz, gw)
        gkx = G5.scorer_grads(ms["zspx"], feats_kb, keep, nq, B, bz, gw)
        assert sorted(gk4) == sorted(gkx) and all(torch.allclose(gk4[k], gkx[k], atol=1e-6) for k in gkx)
        # 8. toy fits through lean_gpu's loop: a repeat is bit for bit; each W4 learns from the family (on the KB only
        #    the weight decay moves it); the family moves the fit; on the KB the fit is zspx's bit for bit on every
        #    shared key
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            fit = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx4")
            rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx4 repeat")
            kb = LG.fit_variant([c_kb], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx4 on a KB")
        with S.patched(BASE_ARM):
            base = LG.fit_variant([c_res], blocks, cfg, 0, 16, "none", "cpu", tag="toy zspx")
        pairs = lambda x, y: zip(x["states"] + [x["swa"]], y["states"] + [y["swa"]])  # noqa: E731
        assert all(not LG.same_state(x, y) for x, y in pairs(fit, rep))
        assert all(not torch.equal(fit["swa"][k], kb["swa"][k]) for k in W4)
        assert any(LG.same_state({k: x[k] for k in y}, y) for x, y in pairs(fit, base))
        assert all(not LG.same_state({k: x[k] for k in y}, y) for x, y in pairs(kb, base))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 9. the refusals
    good = ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspk4", "--name", FIT["L-musique"], "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", ARM, "--name", FIT["L-musique"], "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": BASE_ARM, "g5": {}}), ("x", {"arm": "zspk4", "a11": {}}), ("w", {"arm": ARM})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 10. relz's pair and re-call against zspx's card fits, stamped A11x's; relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        null1 = SR.fake_null(T_ / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
        bs = {sp: str(SR.fake_compare(T_ / BASE_FITS[sp][0], BASE_FITS[sp][2], sp,
                                      {ds: 0.1 for ds in LG.EVAL_ORDER}, arm=BASE_ARM)) for sp in BASE_FITS}
        zb = {sp: "/".join(("outputs",) + BASE_FITS[sp]) for sp in BASE_FITS}
        za = SR.fake_compare(T_ / ARM, FIT["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T_ / ARM, FIT["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T_ / "p")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM, pr["verdict"]
        pj = json.loads((T_ / "p.json").read_text(encoding="utf-8"))
        assert pj["stage"] == "A11x" and pj["decided_against"] == AGAINST and pj["declared_in"] == DECLARED
        r = recall(null1, T_ / "p.json", T_ / "r", bs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T_ / "r.md").read_text(encoding="utf-8")
        assert t.startswith(f"A11x ({DECLARED})") and f"| zspx R@5 | zspx4 R@5 ({WHAT}) |" in t
        assert json.loads((T_ / "r.json").read_text(encoding="utf-8"))["stage"] == "A11x"
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 11. the verdict's outcomes and next forms; the parent's re-call is not this stage's
        rp = T_ / "rc.json"
        gm, lw = ("L-musique", "musique"), ("L-hotpotqa", "webqsp")
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and v["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        v = verdict(fake_recall(rp, {gm: "GAIN", lw: "LOSS"}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["GAIN_AND_LOSS"]
        v = verdict(fake_recall(rp, {}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {lw: "LOSS"}), T_ / "v")
        assert v["verdict"] == "NOT_ADOPTED" and v["next_form"] == NEXT["NO_GAIN"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, stage="A11", arm="zspk4"), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"]
        v = verdict(fake_recall(rp, {gm: "GAIN"}, rows=[]), T_ / "v")
        assert v["verdict"] == "INCOMPLETE" and v["problems"]
        assert verdict(T_ / "none.json", T_ / "v")["verdict"] == "INCOMPLETE"
        v = verdict(fake_recall(rp, {gm: "GAIN"}), T_ / "v")
        assert sorted(v["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "Next form" in (T_ / "v.md").read_text(encoding="utf-8")
    log(f"za11x selftest: zspx4 is zspx plus three W4s (288 parameters; the first A11's own draw) on G3's carve with "
        f"A11's fourth family beside, every other batch key zspx's; A11's build over a records-only part writes its "
        f"arrays back byte for byte; the edges are the brute force's, none on a KB; zspx4 is zspx bit for bit at the "
        f"start, moves only rows with a fourth-family neighbour with one head moved (zspk4 exactly with zspx's step "
        f"alone), and with all three only the questions the family reaches; it is zspx on the KB, without the key and "
        f"without links; the columns match a float64 brute force and the first step's carry the gradient to the "
        f"scorer; toy fits: repeats identical, each W4 learns, on a KB the fit is zspx's on shared keys; refusals; "
        f"relz's pair and re-call stamped A11x's; the verdict's outcomes and next forms ({time.time() - t0:.1f}s): ok")
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
        return 1 if verdict()["verdict"] == "INCOMPLETE" else 0
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"za11x {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za11x: train, read, compare, pair, recall, verdict, or --selftest (builds: za11p4.py build)")


if __name__ == "__main__":
    sys.exit(main())
