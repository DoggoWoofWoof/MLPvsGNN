"""A8 (docs/A8_QDEPTH_INPUT.md): round 11's question-to-depth prior as a fixed input to both of the lean pair's models.
scripts/a8_qdepth_cols.py fits, per screen split, the prior of outputs/mp_unified/qdepth.py on its own (five logits
per question, l_q = x_q Theta + b, x_q the question's embedding and its pool's depth profile, standardised; the
listwise loss of the depth-only score l_q[depth(v)] over the question's rows, L2 on Theta picked on held-out questions;
refit per split without its held-out dataset, cross-fitted over two folds of the training questions), and builds six
columns per row:

    lw      log_softmax(l_q)[depth(v)]: the prior's log-weight of the row's depth class, over the five classes
    d0..d4  the row's depth class (qdepth.depth_of: 0 a seed, 1-3 the first structural hop, 4 unreached), one-hot

float16. The prior runs before either model, on the question's embedding and the pool's fixed walks. No column reads a
label or any row's score, and none passes over graph neighbours at read time (the profile counts the pool's rows by
class). Both models read them through the same pointwise head, Linear(6, 16), GELU, Linear(16, 1) (129 parameters,
last layer zero at the start, first drawn from its own generator, seed + 8801):

    zrckpq  zrckp's model (outputs/mp_unified/zp1c3p.py) plus the head, on zrckp's carve with the columns beside
    zspkq   zspk's model (outputs/mp_unified/zp1c3.py) plus the head, added with the KT head before zsp's propagation
            reads the score, on zspk's carve with the columns beside

Each split reads its own columns (outputs/a8/cols/<split>/...): a fit of L-musique never sees a prior that trained on
musique, nor one of L-hotpotqa on hotpotqa.

    python outputs/mp_unified/za8qd.py train --split L-musique --name scr-a8-zrckpq --arm zrckpq --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8qd.py read --name scr-a8-zrckpq --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za8qd.py compare --new outputs/screen/fits/scr-a8-zrckpq \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a8-zrckpq
    python outputs/mp_unified/za8qd.py pair --arm zrckpq --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za8qd.py recall --arm zrckpq --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za8qd.py verdict
    python outputs/mp_unified/za8qd.py --selftest
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/A8_QDEPTH_INPUT.md"
OUT = ROOT / "outputs" / "a8"
COLS_DIR = OUT / "cols"
SCREEN = ROOT / "outputs" / "screen"
A8_KEY = "_a8"
COLNAMES = ("lw", "d0", "d1", "d2", "d3", "d4")
NCOL = len(COLNAMES)
FOLDS = 2
A8_HID, A8_SEED_OFF = 16, 8801
A8P = ("a8_w1", "a8_b1", "a8_w2", "a8_b2")
N_A8 = A8_HID * NCOL + A8_HID + A8_HID + 1
# scripts/a8_qdepth_cols.py's settings, which every column record carries (the selftest checks they are the same)
BUILD_SETTINGS = {'columns': ['lw', 'd0', 'd1', 'd2', 'd3', 'd4'],
                  'prior': "round 11's (outputs/mp_unified/qdepth.py): l_q = x_q Theta + b over the five depth "
                           "classes, x_q = (q_emb, the pool's depth profile) standardised",
                  'depth': "qdepth.depth_of on WALK's raw columns: 0 a seed, 1-3 the first structural hop (the "
                           "nearest), 4 unreached; the profile is qdepth.profile's, in float64",
                  'objective': "lean_gpu's listwise loss of the depth-only score l_q[depth(v)] over the question's "
                               "rows, each dataset's mean weighted alike, plus lam ||Theta||^2 (b free)",
                  'solver': 'L-BFGS (strong Wolfe), float64, from zero, max_iter 2000, history 20, tolerance_grad '
                            '1e-09, tolerance_change 1e-12; the grid strongest first, each from the previous '
                            'solution',
                  'standardise': 'by the training questions under the same weights (a dimension whose sd is below '
                                 '1e-06 is centred only)',
                  'lams': [0.1, 0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001, 3e-05, 1e-05],
                  'pick': "the lam of the grid whose dataset-balanced mean, over the split's four datasets, of the "
                          "held-out loss (the fold's held-out questions with a gold in the pool) is lowest; ties go "
                          'to the stronger',
                  'folds': 2,
                  'fold': "parity of sha256('dataset:question id') read as an integer",
                  'cross_fit': "each question's columns from the prior of the other fold",
                  'fit': "per split on its four training datasets' fit carves, one fold's questions less 20% held "
                         "out (P5b2's split_of, seed 0), those with a gold in the pool",
                  'cache': "P1c2's (outputs/p1c2/cache); rows in its order",
                  'dtype': 'float16'}
SETTINGS = {"columns": NCOL, "hidden": A8_HID, "seed_offset": A8_SEED_OFF, "parameters": N_A8,
            "score_input": False, "build": BUILD_SETTINGS}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckpq": "zrckp", "zspkq": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a8-{a}", "L-hotpotqa": f"scr-a8-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckpq": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspkq": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckpq": "zrckp plus the A8 head", "zspkq": "zspk plus the A8 head before its propagation"}
PAIR = {a: SCREEN / f"scr-a8-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a8-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A8's re-call of the arm, "
        "or not twelve reads (docs/A8_QDEPTH_INPUT.md, section 4)")


# ── the carve ────────────────────────────────────────────────────────────────


def load_a8(c, root, split, cache_root, verify=True):
    """A carve's six columns of one split on its device (float16), each part tied to its step-1 record."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    xs, shas = [], {}
    for p in parts:
        d = Path(root) / split / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"za8qd: {rp} missing (a8_qdepth_cols.py cols {c.ds} {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("stage") != "A8" or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"za8qd: {rp} is not A8's build")
        if h.get("split") != split or h.get("dataset") != c.ds or h.get("carve") != c.carve:
            raise SystemExit(f"za8qd: {rp} holds {h.get('split')}'s columns of {h.get('dataset')}/{h.get('carve')}, "
                             f"not {split}'s of {c.ds}/{c.carve}")
        if h["step1_record_sha256"] != c.part_shas[p.name]:
            raise SystemExit(f"za8qd: {rp} was built on another step-1 part")
        fn = d / "cols.npy"
        if verify and LC.sha_file(fn) != h["arrays"]["cols"]["sha256"]:
            raise SystemExit(f"za8qd: {fn}: sha256 is not its record's")
        x = np.load(fn)
        if x.dtype != np.float16 or list(x.shape) != h["arrays"]["cols"]["shape"] or x.shape[1:] != (NCOL,):
            raise SystemExit(f"za8qd: {fn}: {x.dtype} {x.shape}, not its record's float16 (rows, {NCOL})")
        xs.append(x)
        shas[p.name] = LC.sha_file(rp)
    x = np.concatenate(xs)
    if x.shape[0] != int(c.off_np[-1]):
        raise SystemExit(f"za8qd: {c.ds}/{c.carve}: {x.shape[0]} rows of columns against the cache's "
                         f"{int(c.off_np[-1])}")
    return torch.from_numpy(x).to(c.device), shas


def a8_batch(c, qs):
    """The columns of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.a8[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def a8_carve(base, a8_root=COLS_DIR):
    """base's carve class with A8's columns of the split set by on_split beside it: batch() adds them under A8_KEY;
    every block, row, side key and gold is base's, unchanged."""

    class A8Carve(base):
        A8_ROOT = a8_root
        A8_SPLIT = None

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            sp = type(self).A8_SPLIT
            if sp not in SPLITS:
                raise SystemExit(f"za8qd: no screen split is set for A8's columns ({sp}); run under on_split")
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.a8_split = sp
            self.a8, self.a8_records = load_a8(self, type(self).A8_ROOT, sp, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.a8.numel() * self.a8.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[A8_KEY] = a8_batch(self, qs)
            return feats, nq, base_z, gold

    A8Carve.__name__ = A8Carve.__qualname__ = f"A8{base.__name__}"
    return A8Carve


A8KTChainCarveZRC = a8_carve(Z3.KTChainCarveZRC)
A8KTLinkCarveBase = a8_carve(Z3.KTLinkCarveBase)
CARVES = (A8KTChainCarveZRC, A8KTLinkCarveBase)


@contextlib.contextmanager
def on_split(split, carves=CARVES):
    """Every A8 carve class reads the split's columns inside, and none outside."""
    if split not in SPLITS:
        raise SystemExit(f"za8qd: A8's columns are built for {', '.join(SPLITS)}, not {split}")
    saved = [c.A8_SPLIT for c in carves]
    for c in carves:
        c.A8_SPLIT = split
    try:
        yield
    finally:
        for c, s in zip(carves, saved):
            c.A8_SPLIT = s


# ── the head ─────────────────────────────────────────────────────────────────


def a8_init(m, seed):
    g = torch.Generator().manual_seed(int(seed) + A8_SEED_OFF)
    m.a8_w1 = nn.Parameter((torch.rand(A8_HID, NCOL, generator=g) * 2 - 1) / NCOL ** 0.5)
    m.a8_b1 = nn.Parameter(torch.zeros(A8_HID))
    m.a8_w2 = nn.Parameter(torch.zeros(1, A8_HID))
    m.a8_b2 = nn.Parameter(torch.zeros(1))


def a8_head(m, x):
    return Fn.linear(Fn.gelu(Fn.linear(x, m.a8_w1, m.a8_b1)), m.a8_w2, m.a8_b2).squeeze(-1)


class ZRCKPQ(Z3P.ZRCKP):
    """zrckpq: zrckp's model plus the A8 head (zero at the start) over each row's own six columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a8_init(self, seed)

    a8_head = a8_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A8_KEY not in feats:
            return s
        return s + self.a8_head(feats[A8_KEY])


class A8Mix(Z3.KTMix):
    """zrm's score, the KT head, then the A8 head: in zspkq, the score zsp's propagation reads."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a8_init(self, seed)

    a8_head = a8_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A8_KEY not in feats:
            return s
        return s + self.a8_head(feats[A8_KEY])


class ZSPKQ(ZP.ZProp, A8Mix):
    """zspkq: zspk's model with the A8 head added before zsp's propagation reads the score (MRO: ZProp, A8Mix, KTMix,
    ZRM)."""


ARMS = {"zrckpq": (ZRCKPQ, A8KTChainCarveZRC), "zspkq": (ZSPKQ, A8KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def a8_records(carves, split, root=COLS_DIR, cache_root=None):
    """Each carve part's A8 record sha256 for the split; stops when one is missing or not A8's of that split."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / split / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"za8qd: {rp} missing (a8_qdepth_cols.py cols {ds} {cv})")
            h = json.loads(rp.read_text(encoding="utf-8"))
            if h.get("declared_in") != DECLARED or h.get("split") != split:
                raise SystemExit(f"za8qd: {rp} is not A8's of {split}")
            out[f"{ds}/{cv}/{p.name}"] = LC.sha_file(rp)
    return out


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
        rec.update({"za8qd_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "a8": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "head": list(A8P), "settings": SETTINGS,
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE),
                        "a8_records": a8_records(Z3.train_carves(split), split),
                        "a8_eval_records": a8_records(Z3.eval_carves(), split),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za8qd: train takes --arm zrckpq or zspkq, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za8qd: A8's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za8qd: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    a8_records(Z3.train_carves(split), split)
    a8_records(Z3.eval_carves(), split)
    with on_split(split), Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a8" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za8qd: {name} is not a A8 fit (arm {rec.get('arm')})")
    if rec["a8"].get("split") != split:
        raise SystemExit(f"za8qd: {name} is {split}'s fit, its record says {rec['a8'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    a8_records(Z3.eval_carves(), split)
    with on_split(split), Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za8qd: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za8qd: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A8's depth prior input) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A8 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A8", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za8qd_sha256": LC.sha_src(__file__)})
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


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools (the prior and its columns fitted and built again "
                              "on P1g-b's cache), then its full run, its own declared file",
        "GAIN_AND_LOSS": "A8k, the back-off form: the six columns split by the graph's kind (the KB graphs' and the "
                         "passage graphs'; twelve columns, the other kind's six at 0), so the KB graphs' depth weights "
                         "cannot move the passage graphs' head weights; on both models, its own declared file",
        "NO_GAIN": "A8e, round 11's own form on both lean models: the prior l_q[depth(v)] added to each row's score "
                   "(qdepth.QDepth's U, W, b at zero), trained by each model's listwise loss, on zrckp and on zspk "
                   "before zsp's propagation reads the score; one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A8's screen verdict, per arm, from its re-call (section 4); webqsp and metaqa beside, musique's in-domain read
    of both arms side by side, and the next form each outcome names."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A8" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A8's re-call of {a}")
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
    rec = {"stage": "A8", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za8qd_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A8's screen verdict: zrckpq **{per['zrckpq']['verdict']}**, zspkq **{per['zspkq']['verdict']}**", "",
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
           "", "This verdict closes only this form (the depth prior's six columns through a pointwise head); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za8qd verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A8", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def toy_cols(root, cache_root, ds, carve, split, rng, **over):
    """Columns of one split for every step-1 part of a toy carve, with the records the builder writes."""
    parts, recs = LC.part_dirs(ds, carve, cache_root)
    for p, r in zip(parts, recs):
        d = Path(root) / split / ds / carve / p.name
        d.mkdir(parents=True, exist_ok=True)
        n_ = int(r["rows"])
        x = np.c_[-4.0 * rng.random(n_), np.eye(5)[rng.integers(0, 5, n_)]].astype(np.float16)
        sha = R.save_npy(d, "cols", x)
        rec = {"declared_in": DECLARED, "stage": "A8", "split": split, "dataset": ds, "carve": carve,
               "part": p.name, "step1_record_sha256": LC.sha_file(p / "record.json"), "settings": BUILD_SETTINGS,
               "arrays": {"cols": {"dtype": x.dtype.str, "shape": list(x.shape), "sha256": sha}}}
        rec.update(over)
        R.write_json(d / "record.json", rec)


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    torch.use_deterministic_algorithms(True)
    # 1. the arms: zrckp's and zspk's models plus the head, on their carves with the columns beside; the bases unchanged
    assert S.ARMS["zrckpq"] == (ZRCKPQ, A8KTChainCarveZRC) and S.ARMS["zspkq"] == (ZSPKQ, A8KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPQ.__mro__[1] is Z3P.ZRCKP and ZSPKQ.__mro__[1:5] == (ZP.ZProp, A8Mix, Z3.KTMix, ZM.ZRM)
    assert A8KTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and A8KTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert vars(A8KTChainCarveZRC)["A8_ROOT"] == COLS_DIR == vars(A8KTLinkCarveBase)["A8_ROOT"]
    assert all(c.A8_SPLIT is None for c in CARVES)
    # 2. the builder's constants are these (the records carry its settings)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import a8_qdepth_cols as BLD
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert (BLD.DECLARED, BLD.COLNAMES, BLD.NCOL, BLD.FOLDS) == (DECLARED, COLNAMES, NCOL, FOLDS)
    assert BLD.BUILD_SETTINGS == BUILD_SETTINGS and BLD.COLS == COLS_DIR and tuple(BLD.SPLITS) == tuple(SPLITS)
    assert {sp: set(BLD.SPLITS[sp]) for sp in SPLITS} == {sp: set(LG.FITS[sp]) for sp in SPLITS}
    assert {(d, cv) for sp in SPLITS for d, cv in BLD.carves_of()[sp]} == {
        c for sp in SPLITS for c in Z3.train_carves(sp)} | set(Z3.eval_carves())
    rng = np.random.default_rng(57)
    tmp = Path(tempfile.mkdtemp(prefix="za8qd_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for sp in SPLITS:
            toy_cols(tmp / "a8", tmp / "cache", "metaqa", "toy", sp, rng)
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        C_m, C_g = a8_carve(cls_ch, tmp / "a8"), a8_carve(cls_lk, tmp / "a8")
        # 3. a carve reads only under on_split, only its split's columns, and only columns built on its parts
        SR.must_stop(C_m, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        SR.must_stop(lambda: on_split("L-2wiki", (C_m,)).__enter__())
        with on_split("L-hotpotqa", (C_m, C_g)):
            c_mh = C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert C_m.A8_SPLIT is None and c_mh.a8_split == "L-hotpotqa"
        with on_split("L-musique", (C_m, C_g)):
            c_m = C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            c_g = C_g("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert not torch.equal(c_m.a8, c_mh.a8)
        bad = tmp / "a8" / "L-musique" / "metaqa" / "toy" / "part_1of2" / "record.json"
        keep = bad.read_bytes()
        for key, val in (("split", "L-hotpotqa"), ("step1_record_sha256", "x"), ("stage", "R1"),
                         ("settings", {**BUILD_SETTINGS, "lams": [1.0]})):
            h = json.loads(keep)
            h[key] = val
            R.write_json(bad, h)
            with on_split("L-musique", (C_m,)):
                SR.must_stop(C_m, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        bad.write_bytes(keep)
        # 4. a batch's columns are its rows' own, in the cache's row order; the MLP's carve holds no edge and gives
        #    the GNN's carve's columns
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c_m.rows)[::-1][:7], np.arange(c_m.rows)[:5]].astype(np.int64)
        B = qs.size
        fm, nq, bz, _gold = c_m.batch(qs, blocks)
        fg, nq_g, bz_g, _ = c_g.batch(qs, blocks)
        assert ZL.LK_KEY not in fm and not hasattr(c_m, "links") and ZL.LK_KEY in fg
        assert torch.equal(fm[A8_KEY], fg[A8_KEY]) and torch.equal(nq, nq_g) and fm[A8_KEY].dtype == torch.float32
        allx = np.concatenate([np.load(tmp / "a8" / "L-musique" / "metaqa" / "toy" / f"part_{i}of2" / "cols.npy")
                               for i in (0, 1)])
        want = np.concatenate([allx[c_m.off_np[q]:c_m.off_np[q] + c_m.n_np[q]] for q in qs]).astype(np.float32)
        assert np.array_equal(fm[A8_KEY].numpy(), want) and want.shape[0] == int(c_m.n_np[qs].sum())
        # 5. the models: each arm starts as its base bit for bit; with the head moved, zrckpq adds the head over the
        #    columns and zspkq's propagation reads the score with the head added; without the columns each is its base
        keep_ = torch.ones(B, len(blocks))
        for arm, cls, base, cv in (("zrckpq", ZRCKPQ, Z3P.ZRCKP, c_m), ("zspkq", ZSPKQ, Z3.ZSPK, c_g)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(A8P) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in A8P) == N_A8 == 129
            m0.eval()
            m1.eval()
            with torch.no_grad():
                if arm == "zspkq":
                    w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
                y0 = m0(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, m1(f_, keep_, nq_, B, bz_))
                m1.a8_w2.copy_(torch.randn(1, A8_HID, generator=torch.Generator().manual_seed(9)))
                y1 = m1(f_, keep_, nq_, B, bz_)
                head = m1.a8_head(f_[A8_KEY])
                assert float(head.abs().max()) > 1e-3
                if arm == "zrckpq":
                    assert torch.allclose(y1 - y0, head, atol=1e-5, rtol=1e-4)    # float32 rounding of y0 + head
                else:
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    kt_only = Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_)
                    assert torch.allclose(pre - kt_only, head, atol=1e-5, rtol=1e-4)
                    assert torch.allclose(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f_[ZL.LK_KEY], nq_, B)),
                                          atol=1e-5)
                    assert float((y1 - y0 - head).abs().max()) > 1e-4
                bare = {k: x for k, x in f_.items() if k != A8_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 6. toy fits through lean_gpu's loop: a repeat is bit-identical and the head learns; on the carve without the
        #    columns each arm's fit is its base's bit for bit on every shared key, its head at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c in (("zrckpq", c_m, cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")),
                                ("zspkq", c_g, cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"))):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no columns")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["a8_w2"].abs().max()) > 0
            for a, b in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k: a[k] for k in b}, b), arm
                assert float(a["a8_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals; the fit names give the split
    assert split_of_fit("scr-a8-zspkq-hp") == ("zspkq", "L-hotpotqa") and split_of_fit("scr-r1-zspkr") == (None, None)
    good = ["train", "--arm", "zrckpq", "--name", FIT["zrckpq"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckpq"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspkq", "--name", FIT["zrckpq"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspkq", "--name", FIT["zspkq"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a8-zspkq", {"arm": "zspkr", "r1": {}}),
                        ("scr-a8-zspkq-hp", {"arm": "zspkq", "a8": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 8. relz's pair and re-call for each arm against its base arm's card fits, stamped A8's; relz restored after
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
            assert pj["stage"] == "A8" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A8 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A8's depth prior input) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict, per arm, with the next form; webqsp and metaqa beside, and musique's in-domain read
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckpq": fake_recall(T_ / "a.json", "zrckpq", {mi: "GAIN"}),
                       "zspkq": fake_recall(T_ / "b.json", "zspkq", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckpq"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspkq"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspkq"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["arms"]["zrckpq"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["musique_in_domain_beside"] == {"zrckpq": {"call": "GAIN", "delta": 0.0},
                                                   "zspkq": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspkq"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckpq": fake_recall(T_ / "a.json", "zrckpq", {}),
                       "zspkq": fake_recall(T_ / "b.json", "zspkq", {gm: "GAIN"}, stage="R1")}, T_ / "v")
        assert rec["arms"]["zrckpq"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspkq"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckpq"]["next_form"] == NEXT["NO_GAIN"]
        rec = verdict({"zrckpq": T_ / "none.json", "zspkq": fake_recall(T_ / "b.json", "zrckpq", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckpq"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspkq"]["verdict"] == "INCOMPLETE"
    log(f"za8qd selftest: the arms are zrckp's and zspk's models plus the head on their carves with the columns beside, "
        f"the MLP's carve without edges; the builder's constants are these; a carve reads only under on_split, only "
        f"its split's columns built on its own parts under the builder's settings; a batch's columns are its rows' own; "
        f"each arm starts as its base bit for bit, zspkq's propagation reads the head, and without the columns each is "
        f"its base; toy fits repeat bit for bit, the head learns, and without the columns each fit is its base's; train "
        f"refuses another arm, split, fit name, cache root or device; read refuses a fit that is not A8's or not its "
        f"split's; relz's pair and re-call against each base arm's card fits, stamped A8's; the verdict per arm with "
        f"its next form ({time.time() - t0:.1f}s): ok")
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
            log(f"za8qd {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za8qd: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
