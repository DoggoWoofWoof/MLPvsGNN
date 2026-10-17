"""A7 (docs/A7_S6_SCORER_INPUT.md): S6's scorer as a fixed input to both of the lean pair's models.
scripts/a7_s6_cols.py builds, per screen split, four columns per cache row from S6's fg-gr scorer with chainscore21's
length offset (chainscore19's joint model at T = 0, refit per split without its held-out dataset, cross-fitted over two
folds of the training questions). With f an MLP over the row's 13 node features, p0 a softmax over the question's
chains of g(chain features) - log C_level,length, and m_c(v) chain c's walk mass at v:

    f       f(v): the scorer's own term
    lm      log(eps + sum_c p0(c) m_c(v)): the chains' weighted mass at v
    has     1 when that mass is above 0
    seed    1 on the question's bucket-0 seeds

float16. The scorer runs before either model, on the question, the first stage's ranks and the pool's fixed walks. No
column is normalised over the question's rows, reads either model's score, or passes over graph neighbours at read
time. Both models read them through the same pointwise head, Linear(4, 16), GELU, Linear(16, 1) (97 parameters, last
layer zero at the start, first drawn from its own generator, seed + 7701):

    zrckps  zrckp's model (outputs/mp_unified/zp1c3p.py) plus the head, on zrckp's carve with the columns beside
    zspks   zspk's model (outputs/mp_unified/zp1c3.py) plus the head, added with the KT head before zsp's propagation
            reads the score, on zspk's carve with the columns beside

Each split reads its own columns (outputs/a7/cols/<split>/...): a fit of L-musique never sees a scorer that trained on
musique, nor one of L-hotpotqa on hotpotqa.

    python outputs/mp_unified/za7s6.py train --split L-musique --name scr-a7-zrckps --arm zrckps --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za7s6.py read --name scr-a7-zrckps --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za7s6.py compare --new outputs/screen/fits/scr-a7-zrckps \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a7-zrckps
    python outputs/mp_unified/za7s6.py pair --arm zrckps --screens S1.json,S2.json --out ...
    python outputs/mp_unified/za7s6.py recall --arm zrckps --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/za7s6.py verdict
    python outputs/mp_unified/za7s6.py --selftest
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

DECLARED = "docs/A7_S6_SCORER_INPUT.md"
OUT = ROOT / "outputs" / "a7"
COLS_DIR = OUT / "cols"
SCREEN = ROOT / "outputs" / "screen"
A7_KEY = "_a7"
COLNAMES = ("f", "lm", "has", "seed")
NCOL = len(COLNAMES)
FOLDS = 2
A7_HID, A7_SEED_OFF = 16, 7701
A7P = ("a7_w1", "a7_b1", "a7_w2", "a7_b2")
N_A7 = A7_HID * NCOL + A7_HID + A7_HID + 1
# scripts/a7_s6_cols.py's settings, which every column record carries (the selftest checks they are the same)
_REG6 = "1,0.25,0.05,0.01,0.002,0"
_STOP = ("the epoch (1 to 12) whose dataset-balanced mean, over the split's four datasets, of the held-out objective "
         "(chainscore19's node loss plus chain loss on the fold's held-out questions of the id build, regime 1) is "
         "lowest; ties go to the earlier")
BUILD_SETTINGS = {"columns": list(COLNAMES),
                  "scorer": "chainscore19's joint model fg-gr (f and g MLPs, T = 0) with chainscore21's lb offset",
                  "read": "regime 1 (the whole carve)", "folds": FOLDS,
                  "fold": "parity of sha256('dataset:question id') read as an integer",
                  "cross_fit": "each question's columns from the scorer of the other fold",
                  "fit": "per split on its four training datasets' fit builds (metaqa's five), one fold's questions "
                         "less 20% held out (P5b2's split_of, seed 0), datasets uniform; " + _STOP,
                  "builds": {"kb": {"metaqa": {"hops": 3, "ks": "2,4"}, "webqsp": {"hops": 2, "ks": "4,16,64,256"}},
                             "transforms": {"al4": {"rows": "even", "hops": 2, "ks": "2,4,16", "regimes": "1,0.05,0"},
                                            "sp4": {"rows": "odd", "hops": 2, "ks": "2,4,16", "regimes": "1,0.05,0"},
                                            "mg3": {"rows": "all", "hops": 3, "ks": "2", "regimes": _REG6},
                                            "rf": {"rows": "even", "hops": 3, "ks": "2,4", "regimes": "1,0.05,0"}},
                             "passages": {"hops": 2, "ks": "4,16,64", "k_fine": 256}, "fit_regimes": _REG6,
                             "read_regimes": "1", "max_chains": 20000},
                  "pools": "P1c2's looks (outputs/p1c2/look); rows in P1c2's cache order", "dtype": "float16"}
SETTINGS = {"columns": NCOL, "hidden": A7_HID, "seed_offset": A7_SEED_OFF, "parameters": N_A7,
            "score_input": False, "build": BUILD_SETTINGS}
SPLITS = Z3.SPLITS
BASE_OF = {"zrckps": "zrckp", "zspks": "zspk"}
ARM_NAMES = tuple(BASE_OF)
FIT = {a: {"L-musique": f"scr-a7-{a}", "L-hotpotqa": f"scr-a7-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckps": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspks": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split ({BASE_FITS[a]['L-musique'][2]}, "
              f"{BASE_FITS[a]['L-hotpotqa'][2]})" for a in ARM_NAMES}
WHAT = {"zrckps": "zrckp plus the A7 head", "zspks": "zspk plus the A7 head before its propagation"}
PAIR = {a: SCREEN / f"scr-a7-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-a7-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
BESIDE = ("webqsp", "metaqa")
CLAIM1_READ = ("L-hotpotqa", "musique")
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A7's re-call of the arm, "
        "or not twelve reads (docs/A7_S6_SCORER_INPUT.md, section 4)")


# ── the carve ────────────────────────────────────────────────────────────────


def load_a7(c, root, split, cache_root, verify=True):
    """A carve's four columns of one split on its device (float16), each part tied to its step-1 record."""
    parts, _recs = LC.part_dirs(c.ds, c.carve, cache_root)
    xs, shas = [], {}
    for p in parts:
        d = Path(root) / split / c.ds / c.carve / p.name
        rp = d / "record.json"
        if not rp.exists():
            raise SystemExit(f"za7s6: {rp} missing (a7_s6_cols.py cols {c.ds} {c.carve})")
        h = json.loads(rp.read_text(encoding="utf-8"))
        if h.get("declared_in") != DECLARED or h.get("stage") != "A7" or h.get("settings") != BUILD_SETTINGS:
            raise SystemExit(f"za7s6: {rp} is not A7's build")
        if h.get("split") != split or h.get("dataset") != c.ds or h.get("carve") != c.carve:
            raise SystemExit(f"za7s6: {rp} holds {h.get('split')}'s columns of {h.get('dataset')}/{h.get('carve')}, "
                             f"not {split}'s of {c.ds}/{c.carve}")
        if h["step1_record_sha256"] != c.part_shas[p.name]:
            raise SystemExit(f"za7s6: {rp} was built on another step-1 part")
        fn = d / "cols.npy"
        if verify and LC.sha_file(fn) != h["arrays"]["cols"]["sha256"]:
            raise SystemExit(f"za7s6: {fn}: sha256 is not its record's")
        x = np.load(fn)
        if x.dtype != np.float16 or list(x.shape) != h["arrays"]["cols"]["shape"] or x.shape[1:] != (NCOL,):
            raise SystemExit(f"za7s6: {fn}: {x.dtype} {x.shape}, not its record's float16 (rows, {NCOL})")
        xs.append(x)
        shas[p.name] = LC.sha_file(rp)
    x = np.concatenate(xs)
    if x.shape[0] != int(c.off_np[-1]):
        raise SystemExit(f"za7s6: {c.ds}/{c.carve}: {x.shape[0]} rows of columns against the cache's "
                         f"{int(c.off_np[-1])}")
    return torch.from_numpy(x).to(c.device), shas


def a7_batch(c, qs):
    """The columns of questions qs, in lean_gpu.CacheCarve.batch's row order, float32 on c's device."""
    qs = np.asarray(qs, np.int64)
    cnt = c.n_np[qs]
    idx = np.repeat(c.off_np[qs] - (np.cumsum(cnt) - cnt), cnt) + np.arange(int(cnt.sum()))
    return c.a7[torch.from_numpy(idx).to(c.device)].to(torch.float32)


def a7_carve(base, a7_root=COLS_DIR):
    """base's carve class with A7's columns of the split set by on_split beside it: batch() adds them under A7_KEY;
    every block, row, side key and gold is base's, unchanged."""

    class A7Carve(base):
        A7_ROOT = a7_root
        A7_SPLIT = None

        def __init__(self, ds, carve, basis, device, cache_root=LC.OUT, verify=True, score2=False):
            sp = type(self).A7_SPLIT
            if sp not in SPLITS:
                raise SystemExit(f"za7s6: no screen split is set for A7's columns ({sp}); run under on_split")
            super().__init__(ds, carve, basis, device, cache_root, verify, score2)
            self.a7_split = sp
            self.a7, self.a7_records = load_a7(self, type(self).A7_ROOT, sp, cache_root, verify)

        def nbytes(self):
            return super().nbytes() + self.a7.numel() * self.a7.element_size()

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            feats[A7_KEY] = a7_batch(self, qs)
            return feats, nq, base_z, gold

    A7Carve.__name__ = A7Carve.__qualname__ = f"A7{base.__name__}"
    return A7Carve


A7KTChainCarveZRC = a7_carve(Z3.KTChainCarveZRC)
A7KTLinkCarveBase = a7_carve(Z3.KTLinkCarveBase)
CARVES = (A7KTChainCarveZRC, A7KTLinkCarveBase)


@contextlib.contextmanager
def on_split(split, carves=CARVES):
    """Every A7 carve class reads the split's columns inside, and none outside."""
    if split not in SPLITS:
        raise SystemExit(f"za7s6: A7's columns are built for {', '.join(SPLITS)}, not {split}")
    saved = [c.A7_SPLIT for c in carves]
    for c in carves:
        c.A7_SPLIT = split
    try:
        yield
    finally:
        for c, s in zip(carves, saved):
            c.A7_SPLIT = s


# ── the head ─────────────────────────────────────────────────────────────────


def a7_init(m, seed):
    g = torch.Generator().manual_seed(int(seed) + A7_SEED_OFF)
    m.a7_w1 = nn.Parameter((torch.rand(A7_HID, NCOL, generator=g) * 2 - 1) / NCOL ** 0.5)
    m.a7_b1 = nn.Parameter(torch.zeros(A7_HID))
    m.a7_w2 = nn.Parameter(torch.zeros(1, A7_HID))
    m.a7_b2 = nn.Parameter(torch.zeros(1))


def a7_head(m, x):
    return Fn.linear(Fn.gelu(Fn.linear(x, m.a7_w1, m.a7_b1)), m.a7_w2, m.a7_b2).squeeze(-1)


class ZRCKPS(Z3P.ZRCKP):
    """zrckps: zrckp's model plus the A7 head (zero at the start) over each row's own four columns."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a7_init(self, seed)

    a7_head = a7_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A7_KEY not in feats:
            return s
        return s + self.a7_head(feats[A7_KEY])


class A7Mix(Z3.KTMix):
    """zrm's score, the KT head, then the A7 head: in zspks, the score zsp's propagation reads."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        a7_init(self, seed)

    a7_head = a7_head

    def forward(self, feats, keep, nq, B, base_z):
        s = super().forward(feats, keep, nq, B, base_z)
        if A7_KEY not in feats:
            return s
        return s + self.a7_head(feats[A7_KEY])


class ZSPKS(ZP.ZProp, A7Mix):
    """zspks: zspk's model with the A7 head added before zsp's propagation reads the score (MRO: ZProp, A7Mix, KTMix,
    ZRM)."""


ARMS = {"zrckps": (ZRCKPS, A7KTChainCarveZRC), "zspks": (ZSPKS, A7KTLinkCarveBase)}
S.ARMS.update(ARMS)


# ── train and read ───────────────────────────────────────────────────────────


def a7_records(carves, split, root=COLS_DIR, cache_root=None):
    """Each carve part's A7 record sha256 for the split; stops when one is missing or not A7's of that split."""
    out = {}
    for ds, cv in carves:
        parts, _ = LC.part_dirs(ds, cv, cache_root or Z2.CACHE)
        for p in parts:
            rp = Path(root) / split / ds / cv / p.name / "record.json"
            if not rp.exists():
                raise SystemExit(f"za7s6: {rp} missing (a7_s6_cols.py cols {ds} {cv})")
            h = json.loads(rp.read_text(encoding="utf-8"))
            if h.get("declared_in") != DECLARED or h.get("split") != split:
                raise SystemExit(f"za7s6: {rp} is not A7's of {split}")
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
        rec.update({"za7s6_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "a7": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "split": split,
                        "device": Z2.DEVICE, "head": list(A7P), "settings": SETTINGS,
                        "pools": "P1c2's (outputs/p1c2/pools)", "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(),
                        "gate": LC.sha_file(Z2.GATE),
                        "a7_records": a7_records(Z3.train_carves(split), split),
                        "a7_eval_records": a7_records(Z3.eval_carves(), split),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"za7s6: train takes --arm zrckps or zspks, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"za7s6: A7's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"za7s6: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    a7_records(Z3.train_carves(split), split)
    a7_records(Z3.eval_carves(), split)
    with on_split(split), Z2.on_p1c2():
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    arm, split = split_of_fit(name)
    if rec.get("arm") not in ARMS or "a7" not in rec or rec.get("arm") != arm:
        raise SystemExit(f"za7s6: {name} is not a A7 fit (arm {rec.get('arm')})")
    if rec["a7"].get("split") != split:
        raise SystemExit(f"za7s6: {name} is {split}'s fit, its record says {rec['a7'].get('split')}")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    a7_records(Z3.eval_carves(), split)
    with on_split(split), Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fits ───────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    if arm not in ARMS:
        raise SystemExit(f"za7s6: --arm takes {' or '.join(ARM_NAMES)}, not {arm}")
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"za7s6: no base fit of {BASE_OF[arm]} on {split}")
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
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (A7's S6 scorer input) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"A7 ({DECLARED}): {arm} ({WHAT[arm]}) against {BASE_OF[arm]}.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "A7", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "za7s6_sha256": LC.sha_src(__file__)})
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


NEXT = {"ADOPT_FOR_FULL_RUN": "the arm fitted again on P1g-b's pools (the scorer and its columns built again on P1g-b's "
                              "looks), then its full run, its own declared file",
        "GAIN_AND_LOSS": "A7k, the back-off form: the four columns split by the graph's kind (the KB graphs' and the "
                         "pseudo-typed passage graphs'; eight columns, the other kind's four at 0), so the KB graphs' "
                         "chain weights cannot move the passage graphs' head weights; on both models, its own declared "
                         "file",
        "NO_GAIN": "A7e: on zspk, chainscore20's carried scorer ena-gr (its GNN f with the EM rounds, K1 above em-gr) "
                   "in the scorer's place, a message-passing form the GNN may read; on zrckp, the fg-gr scorer fitted "
                   "on metaqa's five builds alone (S6's filed setting); one declared file",
        "INCOMPLETE": "the missing re-call is run, nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains else NEXT["NO_GAIN"]
    return NEXT[v]


def verdict(recalls=None, out=None):
    """A7's screen verdict, per arm, from its re-call (section 4); webqsp and metaqa beside, musique's in-domain read
    of both arms side by side, and the next form each outcome names."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems, rows = Path(recalls[a]), [], []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "A7" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not A7's re-call of {a}")
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
    rec = {"stage": "A7", "declared_in": DECLARED, "rule": RULE, "arms": per, "musique_in_domain_beside": side,
           "za7s6_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# A7's screen verdict: zrckps **{per['zrckps']['verdict']}**, zspks **{per['zspks']['verdict']}**", "",
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
           "", "This verdict closes only this form (the S6 scorer's four columns through a pointwise head); docs/SCREENS.md, 3b."]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("za7s6 verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "A7", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
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
        x = np.c_[rng.normal(0.0, 2.0, n_), -6.9 * rng.random(n_), rng.random(n_) < 0.6,
                  rng.random(n_) < 0.05].astype(np.float16)
        sha = R.save_npy(d, "cols", x)
        rec = {"declared_in": DECLARED, "stage": "A7", "split": split, "dataset": ds, "carve": carve,
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
    assert S.ARMS["zrckps"] == (ZRCKPS, A7KTChainCarveZRC) and S.ARMS["zspks"] == (ZSPKS, A7KTLinkCarveBase)
    assert S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    assert ZRCKPS.__mro__[1] is Z3P.ZRCKP and ZSPKS.__mro__[1:5] == (ZP.ZProp, A7Mix, Z3.KTMix, ZM.ZRM)
    assert A7KTChainCarveZRC.__mro__[1] is Z3.KTChainCarveZRC and A7KTLinkCarveBase.__mro__[1] is Z3.KTLinkCarveBase
    assert vars(A7KTChainCarveZRC)["A7_ROOT"] == COLS_DIR == vars(A7KTLinkCarveBase)["A7_ROOT"]
    assert all(c.A7_SPLIT is None for c in CARVES)
    # 2. the builder's constants are these (the records carry its settings)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import a7_s6_cols as BLD
    finally:
        sys.path.remove(str(ROOT / "scripts"))
    assert (BLD.DECLARED, BLD.COLNAMES, BLD.NCOL, BLD.FOLDS) == (DECLARED, COLNAMES, NCOL, FOLDS)
    assert BLD.BUILD_SETTINGS == BUILD_SETTINGS and BLD.COLS == COLS_DIR and tuple(BLD.SPLITS) == tuple(SPLITS)
    assert {sp: set(BLD.SPLITS[sp]) for sp in SPLITS} == {sp: set(LG.FITS[sp]) for sp in SPLITS}
    assert {(d, cv) for sp in SPLITS for d, cv in BLD.carves_of()[sp]} == {
        c for sp in SPLITS for c in Z3.train_carves(sp)} | set(Z3.eval_carves())
    rng = np.random.default_rng(57)
    tmp = Path(tempfile.mkdtemp(prefix="za7s6_"))
    try:
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        for sp in SPLITS:
            toy_cols(tmp / "a7", tmp / "cache", "metaqa", "toy", sp, rng)
        cls_ch = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        cls_lk = ZL.link_carve(cls_ch, tmp / "lk")
        C_m, C_g = a7_carve(cls_ch, tmp / "a7"), a7_carve(cls_lk, tmp / "a7")
        # 3. a carve reads only under on_split, only its split's columns, and only columns built on its parts
        SR.must_stop(C_m, "metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        SR.must_stop(lambda: on_split("L-2wiki", (C_m,)).__enter__())
        with on_split("L-hotpotqa", (C_m, C_g)):
            c_mh = C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert C_m.A7_SPLIT is None and c_mh.a7_split == "L-hotpotqa"
        with on_split("L-musique", (C_m, C_g)):
            c_m = C_m("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            c_g = C_g("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert not torch.equal(c_m.a7, c_mh.a7)
        bad = tmp / "a7" / "L-musique" / "metaqa" / "toy" / "part_1of2" / "record.json"
        keep = bad.read_bytes()
        for key, val in (("split", "L-hotpotqa"), ("step1_record_sha256", "x"), ("stage", "R1"),
                         ("settings", {**BUILD_SETTINGS, "read": "regime 0.05"})):
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
        assert torch.equal(fm[A7_KEY], fg[A7_KEY]) and torch.equal(nq, nq_g) and fm[A7_KEY].dtype == torch.float32
        allx = np.concatenate([np.load(tmp / "a7" / "L-musique" / "metaqa" / "toy" / f"part_{i}of2" / "cols.npy")
                               for i in (0, 1)])
        want = np.concatenate([allx[c_m.off_np[q]:c_m.off_np[q] + c_m.n_np[q]] for q in qs]).astype(np.float32)
        assert np.array_equal(fm[A7_KEY].numpy(), want) and want.shape[0] == int(c_m.n_np[qs].sum())
        # 5. the models: each arm starts as its base bit for bit; with the head moved, zrckps adds the head over the
        #    columns and zspks's propagation reads the score with the head added; without the columns each is its base
        keep_ = torch.ones(B, len(blocks))
        for arm, cls, base, cv in (("zrckps", ZRCKPS, Z3P.ZRCKP, c_m), ("zspks", ZSPKS, Z3.ZSPK, c_g)):
            f_, nq_, bz_, _ = cv.batch(qs, blocks)
            torch.manual_seed(0)
            m0 = base(blocks, cv.widths, 16, dropout=0.1, seed=0)
            torch.manual_seed(0)
            m1 = cls(blocks, cv.widths, 16, dropout=0.1, seed=0)
            s0, s1 = m0.state_dict(), m1.state_dict()
            assert sorted(set(s1) - set(s0)) == sorted(A7P) and set(s0) <= set(s1)
            assert not LG.same_state(s0, {k: s1[k] for k in s0})
            assert sum(s1[k].numel() for k in A7P) == N_A7 == 97
            m0.eval()
            m1.eval()
            with torch.no_grad():
                if arm == "zspks":
                    w = torch.randn(1, ZP.HID, generator=torch.Generator().manual_seed(8))
                    m0.sp_w2.copy_(w)
                    m1.sp_w2.copy_(w)
                y0 = m0(f_, keep_, nq_, B, bz_)
                assert torch.equal(y0, m1(f_, keep_, nq_, B, bz_))
                m1.a7_w2.copy_(torch.randn(1, A7_HID, generator=torch.Generator().manual_seed(9)))
                y1 = m1(f_, keep_, nq_, B, bz_)
                head = m1.a7_head(f_[A7_KEY])
                assert float(head.abs().max()) > 1e-3
                if arm == "zrckps":
                    assert torch.allclose(y1 - y0, head, atol=1e-5)
                else:
                    pre = super(ZP.ZProp, m1).forward(f_, keep_, nq_, B, bz_)
                    kt_only = Z3.KTMix.forward(m1, f_, keep_, nq_, B, bz_)
                    assert torch.allclose(pre - kt_only, head, atol=1e-5)
                    assert torch.allclose(y1, pre + m1.prop_head(ZP.prop_inputs(pre, f_[ZL.LK_KEY], nq_, B)),
                                          atol=1e-5)
                    assert float((y1 - y0 - head).abs().max()) > 1e-4
                bare = {k: x for k, x in f_.items() if k != A7_KEY}
                assert torch.equal(m1(bare, keep_, nq_, B, bz_), m0(bare, keep_, nq_, B, bz_))
        # 6. toy fits through lean_gpu's loop: a repeat is bit-identical and the head learns; on the carve without the
        #    columns each arm's fit is its base's bit for bit on every shared key, its head at the start
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        for arm, cv, bare_c in (("zrckps", c_m, cls_ch("metaqa", "toy", "2wiki", "cpu", tmp / "cache")),
                                ("zspks", c_g, cls_lk("metaqa", "toy", "2wiki", "cpu", tmp / "cache"))):
            with S.patched(arm):
                fit = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}")
                rep = LG.fit_variant([cv], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm} repeat")
                bare = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {arm}, no columns")
            with S.patched(BASE_OF[arm]):
                base = LG.fit_variant([bare_c], blocks, cfg, 0, 16, "none", "cpu", tag=f"toy {BASE_OF[arm]}")
            assert all(not LG.same_state(a, b) for a, b in zip(fit["states"] + [fit["swa"]],
                                                               rep["states"] + [rep["swa"]]))
            assert float(fit["swa"]["a7_w2"].abs().max()) > 0
            for a, b in zip(bare["states"] + [bare["swa"]], base["states"] + [base["swa"]]):
                assert not LG.same_state({k: a[k] for k in b}, b), arm
                assert float(a["a7_w2"].abs().max()) == 0.0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 7. the refusals; the fit names give the split
    assert split_of_fit("scr-a7-zspks-hp") == ("zspks", "L-hotpotqa") and split_of_fit("scr-r1-zspkr") == (None, None)
    good = ["train", "--arm", "zrckps", "--name", FIT["zrckps"]["L-musique"], "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", FIT["zrckps"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, good, "L-hotpotqa")
    SR.must_stop(train, ["train", "--arm", "zspks", "--name", FIT["zrckps"]["L-musique"], "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspks", "--name", FIT["zspks"]["L-musique"], "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T_ = Path(td)
        for nm, rec in (("y", {"arm": "zrckp", "p1c3p": {}}), ("scr-a7-zspks", {"arm": "zspkr", "r1": {}}),
                        ("scr-a7-zspks-hp", {"arm": "zspks", "a7": {"split": "L-musique"}})):
            (T_ / "fits" / nm).mkdir(parents=True)
            LC.write_json(T_ / "fits" / nm / "screen.json", rec)
            SR.must_stop(read, ["--name", nm, "--out-root", str(T_ / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 8. relz's pair and re-call for each arm against its base arm's card fits, stamped A7's; relz restored after
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
            assert pj["stage"] == "A7" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T_ / arm / "pz.json", T_ / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T_ / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"A7 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (A7's S6 scorer input) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 9. the verdict, per arm, with the next form; webqsp and metaqa beside, and musique's in-domain read
        gm, mi = ("L-musique", "musique"), CLAIM1_READ
        rec = verdict({"zrckps": fake_recall(T_ / "a.json", "zrckps", {mi: "GAIN"}),
                       "zspks": fake_recall(T_ / "b.json", "zspks", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T_ / "v")
        assert rec["arms"]["zrckps"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspks"]["verdict"] == "NOT_ADOPTED"
        assert rec["arms"]["zspks"]["next_form"] == NEXT["GAIN_AND_LOSS"]
        assert rec["arms"]["zrckps"]["next_form"] == NEXT["ADOPT_FOR_FULL_RUN"]
        assert rec["musique_in_domain_beside"] == {"zrckps": {"call": "GAIN", "delta": 0.0},
                                                   "zspks": {"call": "WITHIN", "delta": 0.0}}
        assert sorted(rec["arms"]["zspks"]["beside"]) == sorted(f"{sp} {d}" for sp in SPLITS for d in BESIDE)
        assert "closes only this form" in (T_ / "v.md").read_text(encoding="utf-8")
        rec = verdict({"zrckps": fake_recall(T_ / "a.json", "zrckps", {}),
                       "zspks": fake_recall(T_ / "b.json", "zspks", {gm: "GAIN"}, stage="R1")}, T_ / "v")
        assert rec["arms"]["zrckps"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspks"]["verdict"] == "INCOMPLETE"
        assert rec["arms"]["zrckps"]["next_form"] == NEXT["NO_GAIN"]
        rec = verdict({"zrckps": T_ / "none.json", "zspks": fake_recall(T_ / "b.json", "zrckps", {gm: "GAIN"})},
                      T_ / "v")
        assert rec["arms"]["zrckps"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspks"]["verdict"] == "INCOMPLETE"
    log(f"za7s6 selftest: the arms are zrckp's and zspk's models plus the head on their carves with the columns beside, "
        f"the MLP's carve without edges; the builder's constants are these; a carve reads only under on_split, only "
        f"its split's columns built on its own parts under the builder's settings; a batch's columns are its rows' own; "
        f"each arm starts as its base bit for bit, zspks's propagation reads the head, and without the columns each is "
        f"its base; toy fits repeat bit for bit, the head learns, and without the columns each fit is its base's; train "
        f"refuses another arm, split, fit name, cache root or device; read refuses a fit that is not A7's or not its "
        f"split's; relz's pair and re-call against each base arm's card fits, stamped A7's; the verdict per arm with "
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
            log(f"za7s6 {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("za7s6: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
