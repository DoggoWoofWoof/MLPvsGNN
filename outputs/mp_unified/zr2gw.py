"""R2GW (docs/R2GW_R2G_UNDER_WQ.md): R2g (docs/R2_BRIDGE_EM.md, its next form for a GAIN and a LOSS) under WQ on Kaggle's T4,
against WQK's base set (docs/WQK_WQ_ON_T4.md). zrckpbg is zrckpb (zr2br.py, imported unchanged: zrckp plus R2's bridge head
and its EM term) with the head multiplied, per question, by 2 sigmoid(a . c_q + b); a = b = 0 at the start, so it starts as
zrckpb bit for bit. It is trained by zp1c3p's train under an arm swap with zr2br's EM term, at WQK's names and roots with
WQK's mixed KT input (zwqk.py, imported unchanged), as WQK trains zrckp, and read on the device it was fitted on with its
gate per question kept. zrckpb (R2t's arm) is fitted beside it. Each fit is compared with WQK's seed-0 zrckp fit of its
split (zrckpbg also with zrckpb), all on one device, and zrckpbg's re-call under WQK's zrckp seed null decides.

In the kernel (zr2gw_session.py runs each in its own process, after zwqk.py overlay and gate):

    python outputs/mp_unified/zr2gw.py fit --name scr-r2gw-zrckpbg-hp --rows-cap 65536

On the laptop, after `kaggle kernels output` into a new empty folder for each session:

    python outputs/mp_unified/zr2gw.py take --from DIR   # each fit's records, reads and gates per question, never its weights
    python outputs/mp_unified/zr2gw.py compare-all       # outputs/screen/scr-r2gw-<arm>-vs-zrckp[-hp].json
    python outputs/mp_unified/zr2gw.py verdict           # outputs/r2gw/screen-verdict (.json, .md)
    python outputs/mp_unified/zr2gw.py --selftest

This file imports the standard library and zwqk (with it zwq) only; zr2br (torch, zg6, zp1c3p and R2's arms) and torch are
imported by the commands that need them, outside zwq.on_wq. verdict exits 0 on ADOPT_FOR_FULL_RUN or NOT_ADOPTED and 2
after writing an INCOMPLETE record.
"""
import contextlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zwqk as K  # noqa: E402  (the standard library and zwq at import)

WQ = K.WQ
DECLARED = "docs/R2GW_R2G_UNDER_WQ.md"
STAGE = "R2GW"
KEY = "r2gw"                                  # the stamp's key in screen.json, read.json and R2GW's compares
ARM, BESIDE_ARM, BASE = "zrckpbg", "zrckpb", "zrckp"
ARMS = (ARM, BESIDE_ARM)
SEED = 0
SPLITS = tuple(WQ.SPLITS)
READ_CAP = K.ROWS_CAP                         # WQK's rows a read batch (whole questions, so the scores do not change)
NB, R_DIM = 5, 32                             # R2's bridges per question and its bridge space's width
M_DIM = NB - 1                                # the margins of the top-5 bridges
C_DIM = R_DIM + M_DIM                         # c_q: h_q (32) and the margins (4)
N_GATE = C_DIM + 1                            # a (36) and b (1)
R2_N = 59537                                  # R2's parameters (zrckpb's part); zrckpbg's are R2_N + N_GATE
R2_DECLARED = "docs/R2_BRIDGE_EM.md"
R2_CONST = {"K": 5, "r": 32, "LAM": 1.0, "prior_c": 60, "seed_offset": 5502, "head": [3, 16, 1], "w1_scale": 0.1}
R2_PARAMS = ("r2_U", "r2_V", "r2_W1", "r2_W2", "r2_o", "r2_w1", "r2_b1", "r2_w2", "r2_b2")
GATE_PARAMS = ("r2g_a", "r2g_b")
FIT = {(a, sp): f"scr-r2gw-{a}{WQ.SUF[sp]}" for a in ARMS for sp in SPLITS}
FITS = {n: k for k, n in FIT.items()}         # name -> (arm, split)
BASE_FIT = {sp: K.fit_name(BASE, 0, sp) for sp in SPLITS}         # WQK's seed-0 zrckp: the deciding base
ZSPK_CMP = {sp: K.fit_name("zspk", 1, sp) for sp in SPLITS}       # WQK's zspk seed-1 compare: its base column is zspk's R@5
CMP = {(a, sp): f"scr-r2gw-{a}-vs-zrckp{WQ.SUF[sp]}" for a in ARMS for sp in SPLITS}
COMPARES = {CMP[(a, sp)]: (FIT[(a, sp)], [BASE_FIT[sp]] + ([FIT[(BESIDE_ARM, sp)]] if a == ARM else []))
            for a in ARMS for sp in SPLITS}
NEW = "outputs/r2gw/"
SESSIONS = NEW + "sessions"
VERDICT = NEW + "screen-verdict"
CALLS = {"zrckpbg": (ARM, 0), "zrckpb": (BESIDE_ARM, 0), "zrckpbg-vs-zrckpb": (ARM, 1)}     # call -> (compare's arm, base column)
AGAINST = {"zrckpbg": "zrckp", "zrckpb": "zrckp", "zrckpbg-vs-zrckpb": "zrckpb"}
DECIDING = {"zrckpbg": True, "zrckpb": False, "zrckpbg-vs-zrckpb": False}
PAIR = {c: f"{NEW}r2gw-{c}-pair" for c in CALLS}
RECALL = {c: f"{p}-recall" for c, p in PAIR.items()}
NULL_KIND = {"zrckpbg": "WQK's zrckp seed null",
             "zrckpb": "WQK's zrckp seed null (beside)",
             "zrckpbg-vs-zrckpb": "WQK's zrckp seed null, its seed-0 R@5 replaced by R2GW's zrckpb read (beside)"}
FLOORS = K.FLOORS[BASE]
TIMING_FILE = "r2gw_timing.json"
GATES_FILE = "r2gw_gates.npz"
CODE = tuple(dict.fromkeys(K.CODE + ("zr2br", "zn2", "zr2gw")))
WHAT = {ARM: "zrckp + R2's bridge head, the head gated per question (R2g)", BESIDE_ARM: "zrckp + R2's bridge head"}
NAMING_R2T = ("zrckpb is called \"zrckp + R2's bridge head\" until the user rules whether reading the five bridges' vectors "
              "keeps it inside the MLP track (R2, section 2)")
NAMING = NAMING_R2T + "; zrckpbg likewise, and no claim sets either against the GNN as the MLP"
R2E = ("R2e, the edge form: bridge scores only for bridges linked to the row by zlink's edges (B1d: the edge is the bridge), "
       "as fixed per-row link flags for the MLP and inside zg6's messages for the GNN; one declared file")
R2G = {"form": "R2's head times 2 sigmoid(a . c_q + b), per question",
       "c_q": "[h_q = e_q @ r2_U (32, shared with R2's bridge logits, not detached); g_j = (r_j - r_(j+1)) / r_0, j = 0..3, "
              "r_j the stored c_rrf of R2's j-th bridge; g_j = 0 where bridge j+1 is absent or a value is not finite; all "
              "four 0 where r_0 <= 0 or not finite]",
       "c_dim": C_DIM, "n_params": N_GATE,
       "init": "a = 0, b = 0: the gate is 1 and zrckpbg starts as zrckpb bit for bit", "detached": False}
RULE = ("ADOPT_FOR_FULL_RUN when zrckpbg's re-call against WQK's seed-0 zrckp under WQK's zrckp seed null has no LOSS on any "
        "of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when WQK's base verdict is not "
        "BASE_READY, a fit or compare is missing or not R2GW's, R2GW's fits and WQK's seed-0 zrckp fits are not on one "
        "device, a fit or read names a gate that is not a filed PASS gate of an R2GW session, a fit's R2 constants or EM "
        "term are not as declared, zrckpbg's gate is not as declared or its read holds no gate per question, WQK's floors "
        "are not their compares', or a re-call does not hold the twelve reads (docs/R2GW_R2G_UNDER_WQ.md section 5)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zrckpbg fitted again on the pools WQP carried (U2s's pick), then its full run under WQ "
                              "beside zrckp's, each its own declared file",
        "NOT_ADOPTED": "R2e under WQ on the T4, its own declared file (R2's next form after R2g: " + R2E + ")",
        "INCOMPLETE": "what is missing is supplied (WQK's base set, a fit, a compare, a session's gate or a re-call); "
                      "nothing else changes"}
assert len(FIT) == len(FITS) == 4 and not set(FITS) & set(K.FITS)
assert all(K.FITS.get(b) == (BASE, 0, sp) for sp, b in BASE_FIT.items())
assert all(c in K.COMPARES for c in ZSPK_CMP.values()) and FLOORS == "outputs/wqk/floors-zrckp"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zr2gw: {msg}", flush=True)


# ── the margins c_q reads (section 3) ────────────────────────────────────────


def margins_np(rrf, n, br):
    """(Q, 4) float32: g_j = (r_j - r_(j+1)) / r_0 over each question's bridges (br, R2's (Q, 5) pool offsets, -1 absent),
    r_j the RRF value of its j-th bridge; 0 where bridge j+1 is absent or a value is not finite, all 0 where r_0 <= 0 or
    not finite. rrf per row in pool order, n rows per question."""
    import numpy as np
    n, rrf, br = np.asarray(n, np.int64), np.asarray(rrf), np.asarray(br, np.int64)
    off = np.r_[0, np.cumsum(n)]
    if rrf.ndim != 1 or rrf.size != int(off[-1]) or br.ndim != 2 or br.shape != (n.size, NB):
        raise SystemExit(f"zr2gw: margins over {rrf.shape} scores, {n.size} questions and bridges {br.shape}")
    if ((br < -1) | ((br >= 0) & (br >= n[:, None]))).any():
        raise SystemExit("zr2gw: a bridge index lies outside its question's pool")
    idx = np.where(br >= 0, off[:-1, None] + br, 0)
    vals = rrf.astype(np.float64)[idx] if rrf.size else np.zeros(br.shape)
    r = np.where(br >= 0, vals, np.nan)
    top = r[:, :1]
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (r[:, :-1] - r[:, 1:]) / top
    g = np.where(np.isfinite(top) & (top > 0), g, 0.0)
    g[~np.isfinite(g)] = 0.0
    return g.astype(np.float32)


def gate_stats(g):
    """The gate per question of one read: count, mean, spread, extremes, the share below 1 and the quantiles."""
    import numpy as np
    g = np.asarray(g, np.float64)
    if g.size == 0:
        return {"questions": 0}
    q = np.quantile(g, [0.05, 0.25, 0.5, 0.75, 0.95])
    out = {"mean": g.mean(), "std": g.std(), "min": g.min(), "max": g.max(), "share_below_1": (g < 1.0).mean(),
           "q05": q[0], "q25": q[1], "q50": q[2], "q75": q[3], "q95": q[4]}
    return {"questions": int(g.size), **{k: round(float(v), 6) for k, v in out.items()}}


# ── zr2br, imported outside zwq.on_wq, and zrckpbg's classes ─────────────────


def check_r2(R2):
    """zr2br as R2GW's: R2's declaration, constants, parameters, bridges, key and splits, and zrckpb in lean_screen's table."""
    got = (R2.DECLARED, dict(R2.DECLARED_CONST), tuple(R2.R2_PARAMS), R2.n_params(), R2.K, R2.R_DIM, R2.R2_KEY,
           tuple(R2.SPLITS))
    want = (R2_DECLARED, R2_CONST, R2_PARAMS, R2_N, NB, R_DIM, "_r2", ("L-musique", "L-hotpotqa"))
    if got != want:
        raise SystemExit(f"zr2gw: zr2br is {got}, not R2's declared {want}")
    R2.check_constants()
    if R2.S.ARMS.get(BESIDE_ARM) != R2.ARMS[BESIDE_ARM]:
        raise SystemExit(f"zr2gw: lean_screen's {BESIDE_ARM} is not zr2br's")


_R2 = {}


def import_r2():
    """zr2br (torch, zg6, zp1c3p and R2's arms in lean_screen's table), imported outside zwq.on_wq after zg6
    (zwqk.import_g6): both bind their own splits at import. Builds zrckpbg's model and carve once and puts them in
    lean_screen's table. Returns {"R2", "GateMix", "ZRCKPBG", "GCarve", "np", "torch"}."""
    if _R2:
        return _R2
    K.import_g6()
    import numpy as np
    import torch
    import torch.nn as nn
    import zr2br as R2
    check_r2(R2)

    class GateMix:
        """R2Mix's final score with the head scaled per question: s + g_q * head(x_v), g_q = 2 sigmoid(a . c_q + b)."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.r2g_a = nn.Parameter(torch.zeros(C_DIM))
            self.r2g_b = nn.Parameter(torch.zeros(1))

        def r2g_c(self, r2):
            """c_q (B, 36): h_q = e_q @ r2_U (R2's own question projection) and the carve's margins."""
            gm = r2.get("gm")
            if gm is None:
                raise SystemExit("zr2gw: zrckpbg's batch holds no margins (its carve is not R2GW's)")
            h = r2["eq"] @ self.r2_U
            return torch.cat([h, gm.to(h.dtype)], 1)

        def r2g_gate(self, r2):
            return 2.0 * torch.sigmoid(self.r2g_c(r2) @ self.r2g_a + self.r2g_b)

        def forward(self, feats, keep, nq, B, base_z):
            s = super(R2.R2Mix, self).forward(feats, keep, nq, B, base_z)
            if R2.R2_KEY not in feats:
                return s
            r2 = feats[R2.R2_KEY]
            g = self.r2g_gate(r2)
            seen = self.__dict__.get("r2g_seen")
            if seen is not None and not self.training:
                seen.append(g.detach().to("cpu", torch.float32))
            return s + g[nq] * self.r2_head(self.r2_feats(r2, nq))

    class ZRCKPBG(GateMix, R2.ZRCKPB):
        """zrckpbg: zrckpb with the head gated per question (R2g)."""

    class GCarve(R2.R2KTChainCarveZRC):
        """R2's carve with each question's margins beside its bridges: batch() adds them under R2's key."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.r2g_gm = margins_np(R2.rrf_of(self), self.n_np, self.r2_br)

        def nbytes(self):
            return super().nbytes() + int(self.r2g_gm.nbytes)

        def batch(self, qs, blocks):
            feats, nq, base_z, gold = super().batch(qs, blocks)
            gm = torch.from_numpy(self.r2g_gm[np.asarray(qs, np.int64)]).to(self.device)
            feats[R2.R2_KEY]["gm"] = gm
            return feats, nq, base_z, gold

    GCarve.__name__ = GCarve.__qualname__ = "GR2KTChainCarveZRC"
    R2.S.ARMS[ARM] = (ZRCKPBG, GCarve)
    _R2.update(R2=R2, GateMix=GateMix, ZRCKPBG=ZRCKPBG, GCarve=GCarve, np=np, torch=torch)
    return _R2


def same_modules(R2, W):
    """zr2br's modules are zwq's inside zwq.on_wq (W), so the names on_wq patches are the ones zrckp's carve and loop read."""
    pairs = ((R2.Z3, W.Z3), (R2.Z2, W.Z2), (R2.LG, W.LG), (R2.S, W.S), (R2.S2, W.S2), (R2.LC, W.LC),
             (R2.RM, W.Z3.RM), (R2.ZL, W.Z3.ZL), (R2.Z3P, W.Z3P))
    bad = [b.__name__ for a, b in pairs if a is not b]
    if bad:
        raise SystemExit(f"zr2gw: zr2br does not train on zwq's {bad}")


@contextlib.contextmanager
def on_arm(W, arm):
    """zp1c3p's arm is the R2GW arm's for the block (zpsm.on_arm's swap), the model and carve from lean_screen's table."""
    with WQ.patch(W.Z3P, ARM=arm, ARMS={arm: W.S.ARMS[arm]}):
        yield W


GATES = {}


def gate_reader(orig):
    """lean_gpu.read_carve that also keeps zrckpbg's p@swa gate per question: the model collects each batch's gate in
    r2g_seen (evaluation only); batches run in question order, so the gates are the carve's questions' in order."""
    R = import_r2()
    np, torch, GateMix = R["np"], R["torch"], R["GateMix"]

    def read_carve(c, models, *a, **kw):
        cand = [m for nm, m, _bl in models if nm == "p@swa" and isinstance(m, GateMix)]
        if len(cand) != 1:
            raise SystemExit("zr2gw: the read holds no p@swa candidate of zrckpbg")
        m = cand[0]
        m.r2g_seen = []
        try:
            out = orig(c, models, *a, **kw)
            seen = list(m.r2g_seen)
        finally:
            m.__dict__.pop("r2g_seen", None)
        g = torch.cat(seen).numpy() if seen else np.zeros(0, np.float32)
        q = len(c.n_np)
        if g.size != q:
            raise SystemExit(f"zr2gw: {c.ds}={c.carve}: kept {g.size} gates for {q} questions")
        gm = getattr(c, "r2g_gm", None)
        share = round(float((np.abs(gm).sum(1) > 0).mean()), 6) if gm is not None and len(gm) else None
        GATES[f"{c.ds}={c.carve}"] = {"gate": g, "questions_in_carve": q, "share_margins_nonzero": share}
        return out

    return read_carve


@contextlib.contextmanager
def gate_reads(LG, arm):
    """lean_gpu's read_carve keeps the gate per question while zrckpbg is read; zrckpb's read is lean_gpu's own."""
    if arm != ARM:
        yield LG
        return
    with WQ.patch(LG, read_carve=gate_reader(LG.read_carve)):
        yield LG


# ── guards ───────────────────────────────────────────────────────────────────


def arm_split(name):
    if name not in FITS:
        raise SystemExit(f"zr2gw: {name} is not one of R2GW's fits ({', '.join(FIT.values())})")
    return FITS[name]


def where_ok(env, what):
    if not K.ok_where(env):
        raise SystemExit(f"zr2gw: R2GW {what}s in a Kaggle kernel on a {K.GPU} from the image {K.IMAGE_TAG} (section "
                         f"4), not with {K.device_of(env)} ({env.get('where')})")


def cap_ok(rows_cap):
    if rows_cap is None or int(rows_cap) != READ_CAP:
        raise SystemExit(f"zr2gw: R2GW reads with --rows-cap {READ_CAP} (WQK's, section 4), not {rows_cap}")


def wqk_ready(base=None):
    """WQK's verdict record, committed before the stub: BASE_READY under WQK's declaration (section 4)."""
    p = Path(base or WQ.ROOT) / (K.VERDICT + ".json")
    if not p.is_file():
        raise SystemExit(f"zr2gw: {K.VERDICT}.json missing; R2GW trains after WQK's BASE_READY (section 4)")
    rec = WQ.read_json(p)
    if (rec.get("stage"), rec.get("declared_in"), rec.get("verdict")) != (K.STAGE, K.DECLARED, "BASE_READY"):
        raise SystemExit(f"zr2gw: {K.VERDICT}.json is {rec.get('verdict')}, not WQK's BASE_READY")
    return rec


def term_ok(summ, name):
    """A fit whose training had no batch with R2's term is not R2GW's (section 4): it stops unstamped."""
    if not summ.get("batches_with_term"):
        raise SystemExit(f"zr2gw: no training batch of {name} carried R2's term ({summ.get('batches')} batches); unstamped")


# ── train and read in the kernel (section 4) ─────────────────────────────────


def code_shas():
    return {m: WQ.sha_src(HERE / f"{m}.py") for m in CODE}


def fit_records(split):
    """The KT and chain records the fit trained on (zwqk.stamp_fit's), at WQK's roots with the mixed KT input."""
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, W.Z2.on_p1c2():
        return {"kt_records": W.Z3.kt_records(W.Z3.train_carves(split)), "rmatch_records": W.Z3.RM.built_records(),
                "zrc_records": W.Z2._BUILT(W.Z2.CH_ZRC), "zlink_records": W.Z3.ZL.built_records()}


def head_of(arm):
    return {"r2g": R2G if arm == ARM else None, "kt_root": K.ROOTS["kt"], "roots": dict(K.ROOTS), "read_rows": READ_CAP}


def stamp_fit(name, env, R2, summ, base=None, records=None):
    """The fit's record after zp1c3p's train: the wq block at WQK's roots and declaration (zwqk.wq_block), and R2GW's block
    (the device, WQK's gate, overlay, check and verdict, the KT and chain records, R2's constants, parameters and EM term,
    the gate's declaration, the code)."""
    base = Path(base or WQ.ROOT)
    arm, split = arm_split(name)
    sj = WQ.fit_dir(name, base) / "screen.json"
    rec = WQ.read_json(sj)
    records = fit_records(split) if records is None else records
    with WQ.patch(WQ, ROOT=base):
        wq = K.wq_block(arm, SEED, split)
    rec.update({"zwq_sha256": WQ.sha_src(WQ.__file__), "zwqk_sha256": WQ.sha_src(K.__file__),
                "zr2br_sha256": WQ.sha_src(HERE / "zr2br.py"), "zr2gw_sha256": WQ.sha_src(__file__), "wq": wq})
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "what": WHAT[arm], "seed": SEED, "split": split,
                "against": BASE_FIT[split], "device": K.device_of(env), "env": env,
                "gate": WQ.sha_file(base / K.RECORDS["GATE"]), "overlay": WQ.sha_file(base / K.RECORDS["OVERLAY"]),
                "check": WQ.sha_file(base / K.RECORDS["CHECK"]),
                "wqk_verdict": WQ.sha_file(base / (K.VERDICT + ".json")), "cache_root": K.ROOTS["cache"], **records,
                "r2": {"declared_in": R2.DECLARED, "constants": R2.constants(), "params": list(R2.R2_PARAMS),
                       "settings": R2.SETTINGS, "term": summ},
                "head": head_of(arm), "code": code_shas()}
    WQ.write_json(sj, rec)
    return rec


def train(name, env=None):
    """R2GW's fit (section 4): zp1c3p's train for the arm under the arm swap with R2's EM term (LAM 1), at WQK's names and
    roots with WQK's mixed KT input, as zwqk's train runs zrckp; then the fit's stamps."""
    arm, split = arm_split(name)
    env = K.device_env() if env is None else env
    where_ok(env, "train")
    wqk_ready()
    R2 = import_r2()["R2"]
    with K.on_wqk():
        WQ.gate_passed()
    args = ["train", "--name", name, "--arm", arm, "--seed", str(SEED), "--device", "cuda", "--cache-root",
            K.ROOTS["cache"]]
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, on_arm(W, arm):
        same_modules(R2, W)
        with WQ.patch(W.Z3P, FIT={**W.Z3P.FIT, split: name}), R2.r2_em():
            rc = W.Z3P.train(args, split)
    summ = R2.summary()
    if rc == 0:
        term_ok(summ, name)
        stamp_fit(name, env, R2, summ)
        ep = summ.get("epochs") or []
        log(f"train {name}: done, stamped (R2's term on {summ['questions_with_term']} of {summ['questions']} questions"
            + (f"; L_br {ep[0]['br_nll_mean']:.2f} -> {ep[-1]['br_nll_mean']:.2f})" if ep else ")"))
    return rc


def fit_stamp(name, fits_root=None):
    """An R2GW fit's screen.json and its R2GW block, checked against the fit's name."""
    arm, split = arm_split(name)
    sj = Path(fits_root or WQ.fit_dir(name).parent) / name / "screen.json"
    rec = WQ.read_json(sj) if sj.is_file() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("arm"), k.get("seed"), k.get("split")) != \
            (arm, DECLARED, name, arm, SEED, split):
        raise SystemExit(f"zr2gw: {name} is not R2GW's {arm} fit of {split} (no R2GW stamp)")
    return rec, k


def stamp_read(name, env, kt, rows_cap, gates, base=None):
    """read.json's R2GW block; for zrckpbg also r2gw_gates.npz (the gate per question, one array per carve) and the
    gate's statistics per carve."""
    base = Path(base or WQ.ROOT)
    arm, split = arm_split(name)
    fd = WQ.fit_dir(name, base)
    rj = fd / "read.json"
    rec = WQ.read_json(rj)
    stats, gsha = None, None
    if arm == ARM:
        import numpy as np
        stats = {c: {**gate_stats(d["gate"]), "questions_in_carve": d["questions_in_carve"],
                     "share_margins_nonzero": d["share_margins_nonzero"]} for c, d in gates.items()}
        np.savez_compressed(fd / GATES_FILE, **{c.replace("=", "__"): d["gate"] for c, d in gates.items()})
        gsha = WQ.sha_file(fd / GATES_FILE)
    rec["zr2gw_sha256"] = WQ.sha_src(__file__)
    rec[KEY] = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": SEED, "split": split,
                "device": K.device_of(env), "env": env, "gate": WQ.sha_file(base / K.RECORDS["GATE"]),
                "kt_records": kt, "rows_cap": int(rows_cap), "gate_stats": stats, "gates_sha256": gsha}
    WQ.write_json(rj, rec)
    return rec


def read(name, env=None, rows_cap=READ_CAP):
    """The fit read by zp1c3p's read for its arm with p@swa on the six s1eval carves, --rows-cap 65536, on the device it was
    fitted on (zwqk's read, under the arm swap); zrckpbg's gate per question is kept. read.json is stamped when the read
    exits 0."""
    arm, split = arm_split(name)
    cap_ok(rows_cap)
    env = K.device_env() if env is None else env
    where_ok(env, "read")
    rec, k = fit_stamp(name)
    if K.device_of(env) != k.get("device"):
        raise SystemExit(f"zr2gw: {name} was fitted on {k.get('device')}; this read runs on {K.device_of(env)}")
    got = (k.get("r2") or {}).get("constants")
    if got != R2_CONST:
        raise SystemExit(f"zr2gw: {name} was fitted with R2 constants {got}, not R2's declared {R2_CONST}")
    R2 = import_r2()["R2"]
    with K.on_wqk():
        WQ.cache_root_ok(K.ROOTS["cache"])
        WQ.gate_passed()
    argv = ["--name", name, "--device", "cuda", "--cache-root", K.ROOTS["cache"], "--rows-cap", str(READ_CAP)]
    GATES.clear()
    rc = 1
    try:
        with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W, on_arm(W, arm):
            same_modules(R2, W)
            with gate_reads(W.LG, arm):
                rc = W.Z3P.read(argv)
            kt = W.Z3.kt_records(W.Z3.eval_carves())
        rj = WQ.fit_dir(name) / "read.json"
        if rc == 0 and rj.is_file():
            stamp_read(name, env, kt, rows_cap, dict(GATES))
    finally:
        GATES.clear()
    return rc


def stamped(p, name):
    try:
        k = WQ.read_json(p).get(KEY) or {}
    except (OSError, ValueError):
        return False
    return k.get("declared_in") == DECLARED and k.get("fit") == name


def fit(name, rows_cap=READ_CAP):
    """One fit in the kernel (zwqk's fit, for R2GW): a read R2GW stamped (with zrckpbg's gates) is kept; a fit trained and
    stamped but not read is only read; otherwise the fit's files are unlinked and it trains and reads. r2gw_timing.json
    holds the seconds and the peak."""
    arm, _split = arm_split(name)
    cap_ok(rows_cap)
    fd = WQ.fit_dir(name)
    if stamped(fd / "read.json", name) and (arm != ARM or (fd / GATES_FILE).is_file()):
        log(f"fit {name}: read before, kept")
        return 0
    env = K.device_env()
    where_ok(env, "train")
    wqk_ready()
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").is_file() and (fd / "models.pt").is_file()
    if not trained:
        K.clear(fd, "*")
        K.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    K.clear(fd / "reads", "*")
    K.clear(fd, "read.json")
    K.clear(fd, GATES_FILE)
    rc = read(name, env, rows_cap)
    WQ.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(time.time() - t1, 1), "peak_gb": K.peak_gb(),
                                     "rows_cap": int(rows_cap), "exit": rc, "declared_in": DECLARED})
    return rc


# ── on the laptop, after the sessions: take, compare-all, verdict (sections 4 and 5) ──


def fit_device(name, fits_root=None):
    """An R2GW fit's device: its screen.json's R2GW block, and its read.json's on the same device with --rows-cap 65536."""
    _rec, k = fit_stamp(name, fits_root)
    rj = Path(fits_root or WQ.fit_dir(name).parent) / name / "read.json"
    r = WQ.read_json(rj) if rj.is_file() else {}
    rk = r.get(KEY) or {}
    if (rk.get("declared_in"), rk.get("fit"), rk.get("device"), rk.get("rows_cap")) != \
            (DECLARED, name, k.get("device"), READ_CAP):
        raise SystemExit(f"zr2gw: {name} has no R2GW read on the device it was fitted on")
    return k["device"]


def pair_device(names, base):
    """The one device the fits of names were fitted and read on: R2GW's through fit_device, WQK's through zwqk's."""
    fr = K.fits_dir(base)
    devs = {n: (fit_device(n, fr) if n in FITS else K.fit_device(n, fr)) for n in names}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zr2gw: {names[0]} and its bases ran on different devices or images: {devs}")
    return devs[names[0]]


def split_device(sp, base):
    """The one device R2GW's two fits of sp and WQK's seed-0 zrckp fit of sp ran on."""
    return pair_device([FIT[(ARM, sp)], FIT[(BESIDE_ARM, sp)], BASE_FIT[sp]], base)


def take(src, base=None):
    """A kernel's output (src): each R2GW fit copied to outputs/screen/fits/<name>, its records, reads and gates per
    question, never its weights; the session's gate.json and overlay.json to outputs/r2gw/sessions/<the gate's
    sha256[:12]>. A fit without R2GW's stamp and read on one device is left (named); a folder not R2GW's, or a file
    already there with other bytes, stops the take. Returns (copied, left)."""
    base, src = Path(base or WQ.ROOT), Path(src)
    fits = src / "fits"
    g, o = src / "wqk" / "gate.json", src / "wqk" / "overlay.json"
    if not (g.is_file() and o.is_file()):
        raise SystemExit(f"zr2gw take: {src.name} holds no wqk/gate.json and wqk/overlay.json (the session's records)")
    names = sorted(p.name for p in fits.iterdir() if p.is_dir()) if fits.is_dir() else []
    bad = [n for n in names if n not in FITS]
    if bad:
        raise SystemExit(f"zr2gw take: {bad} are not R2GW's fits")
    done, left = [], []
    for n in names:
        try:
            fit_device(n, fits)
            done.append(n)
        except SystemExit as e:
            left.append(f"{n}: {e}")
    plan = [(f, K.fits_dir(base) / n / f.relative_to(fits / n)) for n in done
            for f in sorted(p for p in (fits / n).rglob("*") if p.is_file())
            if f.name not in K.NEVER and f.suffix not in (".tmp", ".part")]
    sess = base / SESSIONS / WQ.sha_file(g)[:12]
    plan += [(g, sess / "gate.json"), (o, sess / "overlay.json")]
    for f, t in plan:
        if t.exists() and WQ.sha_file(t) != WQ.sha_file(f):
            raise SystemExit(f"zr2gw take: {WQ.rel(t, base)} exists with other bytes")
    took = []
    for f, t in plan:
        if not t.exists():
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(WQ.rel(t, base))
    log(f"take: {len(done)} fits, {len(took)} files copied; the gate {sess.name}" + (f"; left: {left}" if left else ""))
    return took, left


def compare_all(base=None):
    """lean_screen's compare of each R2GW fit with WQK's seed-0 zrckp fit of its split (first: it decides; zrckpbg also
    with the split's zrckpb), run from the repository's root by zwq's run_compare, every fit on one device; each stamped
    with R2GW's record and that device. A compare R2GW stamped from the same fits is kept."""
    base = Path(base or WQ.ROOT)
    import_r2()
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = K.screen_dir(base) / f"{cname}.json"
        if js.is_file():
            k = WQ.read_json(js).get(KEY) or {}
            if (k.get("declared_in"), k.get("new"), k.get("bases")) == (DECLARED, new, list(bases)):
                done.append(cname)
                continue
        dev = pair_device([new] + list(bases), base)
        with K.on_wqk():
            for f in bases:
                if f in K.FITS:
                    why = WQ.wq_why(f, base)
                    if why:
                        raise SystemExit(f"zr2gw: {why}")
            rc = WQ.run_compare(f"outputs/screen/fits/{new}", ",".join(f"outputs/screen/fits/{b}" for b in bases),
                                f"outputs/screen/{cname}")
        if rc or not js.is_file():
            raise SystemExit(f"zr2gw: the compare {cname} exited {rc}")
        rec = WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                    "zr2gw_sha256": WQ.sha_src(__file__)}
        WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """An R2GW compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.is_file():
        raise SystemExit(f"outputs/screen/{cname}.json missing")
    rec = WQ.read_json(p)
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"outputs/screen/{cname}.json is not R2GW's compare of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"outputs/screen/{cname}.json was made on {k.get('device')}, not on the fits' device {dev}")
    got = [WQ.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if WQ.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"outputs/screen/{cname}.json compares {rec.get('new')} with {rec.get('bases')}")
    split = FITS[new][1]
    if sorted(rec.get("trained_on") or []) != sorted(WQ.SPLITS[split]):
        raise SystemExit(f"outputs/screen/{cname}.json trained on {rec.get('trained_on')}, not {split}'s")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"outputs/screen/{cname}.json reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on "
                         f"s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(WQ.EVAL_ORDER):
            raise SystemExit(f"outputs/screen/{cname}.json reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


def filed_gates(base):
    """zwqk.filed_gates on R2GW's sessions: ({gate sha256: its record}, problems), each a WQK PASS gate on the reproduced
    check and its session's overlay."""
    with WQ.patch(K, SESSIONS=SESSIONS):
        return K.filed_gates(base)


def fit_problems(base, gates):
    """Why an R2GW fit is not section 4's: its train exit, its six reads, the gates its train, wq block and read name, its
    overlay, R2's constants and EM term, and for zrckpbg its gate's declaration and the gate per question of its read."""
    out = []
    fr = K.fits_dir(base)
    for (arm, _sp), n in FIT.items():
        sj, rj = fr / n / "screen.json", fr / n / "read.json"
        if not (sj.is_file() and rj.is_file()):
            continue                                  # fit_device names it
        s, r = WQ.read_json(sj), WQ.read_json(rj)
        k, rk = s.get(KEY) or {}, r.get(KEY) or {}
        if s.get("rc") != 0:
            out.append(f"{n}: its train did not exit 0")
        if set(r.get("carves") or {}) != set(K.SIX):
            out.append(f"{n}: not read on the six s1eval carves")
        for what, g in (("train", k.get("gate")), ("wq block", (s.get("wq") or {}).get("gate")), ("read", rk.get("gate"))):
            if g not in gates:
                out.append(f"{n}: its {what} names the gate {str(g)[:12]}, not a filed R2GW PASS gate")
        if k.get("gate") in gates and (gates[k["gate"]].get("records") or {}).get("overlay") != k.get("overlay"):
            out.append(f"{n}: its overlay is not its gate's")
        r2 = k.get("r2") or {}
        term = r2.get("term") or {}
        if (r2.get("declared_in"), r2.get("constants"), r2.get("params"), term.get("lam")) != \
                (R2_DECLARED, R2_CONST, list(R2_PARAMS), 1.0) or not term.get("questions_with_term"):
            out.append(f"{n}: R2's constants or EM term are not as declared")
        h = k.get("head") or {}
        if arm == ARM:
            if h.get("r2g") != R2G:
                out.append(f"{n}: its gate is not R2g's as declared")
            if set(rk.get("gate_stats") or {}) != set(K.SIX):
                out.append(f"{n}: its read holds no gate per question on the six carves")
        elif h.get("r2g") is not None:
            out.append(f"{n}: zrckpb is gated (its head names a gate)")
    return out


def restamp(pair, rcl, c):
    """R2GW's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for p in (pair, rcl):
        js = Path(str(p) + ".json")
        rec = WQ.read_json(js)
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": CALLS[c][0], "against": AGAINST[c],
                    "null_kind": NULL_KIND[c], "deciding": DECIDING[c], "zr2gw_sha256": WQ.sha_src(__file__)})
        WQ.write_json(js, rec)
        md = Path(str(p) + ".md")
        if md.is_file():
            head_ = (f"R2GW ({DECLARED}): {CALLS[c][0]} against {AGAINST[c]} under WQ on the T4"
                     + (", deciding." if DECIDING[c] else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(base, dev, c):
    """screen_pair over R2GW's two compares of the call's arm (column CALLS[c][1] of each: WQK's seed-0 zrckp, or for
    zrckpbg-vs-zrckpb the split's zrckpb), then screen_recall under WQK's zrckp null. For zrckpbg-vs-zrckpb the null's
    seed-0 R@5 is R2GW's zrckpb read (the new column of zrckpb's own compare), which the zrckpbg compare's column must
    equal; the floors stay zrckp's."""
    import screen_pair as SP
    import screen_recall as SR
    base = Path(base)
    scr = K.screen_dir(base)
    arm, col = CALLS[c]

    def load(fn):
        cn = next(x for x in COMPARES if Path(fn).stem == x)
        _rec, rows = cmp_rows(scr / f"{cn}.json", cn, dev)
        return FITS[COMPARES[cn][0]][1], [rows[COMPARES[cn][1][col]][ds] for ds in WQ.EVAL_ORDER]

    pair, rcl = K.at(base, PAIR[c]), K.at(base, RECALL[c])
    pair.parent.mkdir(parents=True, exist_ok=True)
    with WQ.patch(SP, load=load):
        SP.pair([K.at(base, f"outputs/screen/{CMP[(arm, sp)]}.json") for sp in SPLITS], pair)
    null, sources = K.wqk_null(BASE, base, dev, SR.floor_of)
    if c == "zrckpbg-vs-zrckpb":
        for sp in SPLITS:
            cn = CMP[(BESIDE_ARM, sp)]
            _r, kb = cmp_rows(scr / f"{cn}.json", cn, dev)
            null[sp] = {ds: {**v, "seed0": kb[BASE_FIT[sp]][ds]["new"][0]} for ds, v in null[sp].items()}
            sources[f"R2GW zrckpb {sp}"] = f"outputs/screen/{cn}.json"
    with WQ.patch(SR, load_null=lambda _files: (null, sources)):
        rec = SR.pair([], Path(str(pair) + ".json"), rcl)
    restamp(pair, rcl, c)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def timing(name, fits_root):
    return K.timing(name, fits_root, TIMING_FILE)


def f4(x, signed=False):
    return "n/a" if x is None else K.fmt(x, signed)


def verdict(base=None, out=None):
    """R2GW's verdict by the rule from zrckpbg's re-call against WQK's seed-0 zrckp under WQK's zrckp null (sections 4
    and 5). Beside, deciding nothing: zrckpb against zrckp, zrckpbg against zrckpb, the gate per question, R@5 per read,
    R2's term per fit and seconds and memory; outputs/r2gw/screen-verdict."""
    import screen_recall as SR
    base = Path(base or WQ.ROOT)
    od = Path(out) if out else base / VERDICT
    problems = []
    try:
        wqk_ready(base)
    except SystemExit as e:
        problems.append(str(e))
    gates, gp = filed_gates(base)
    problems += gp
    devs = {}
    for sp in SPLITS:
        try:
            devs[sp] = split_device(sp, base)
        except SystemExit as e:
            problems.append(str(e))
    problems += fit_problems(base, gates)
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"R2GW's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    rows = {}
    for cn in COMPARES:
        try:
            rows[cn] = cmp_rows(K.screen_dir(base) / f"{cn}.json", cn, dev)[1]
        except SystemExit as e:
            problems.append(str(e))
    problems += K.floors_problems(base, SR.floor_of)
    rc = {}
    if not problems:
        try:
            for c in CALLS:
                rc[c] = recall(base, dev, c)
        except (SystemExit, KeyError) as e:
            problems.append(WQ.clean(e, base))
    whole = sorted((sp, ds) for sp in SPLITS for ds in WQ.EVAL_ORDER)
    for c, r in rc.items():
        if sorted((x["split"], x["dataset"]) for x in r["rows"]) != whole:
            problems.append(f"the re-call of {c} against {AGAINST[c]} holds {len(r['rows'])} reads, not the twelve of "
                            f"{', '.join(SPLITS)}")
    rrows = rc["zrckpbg"]["rows"] if "zrckpbg" in rc and not problems else []
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = NEXT[v]
    wv = base / (K.VERDICT + ".json")
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": BASE,
           "fits": {a: {sp: FIT[(a, sp)] for sp in SPLITS} for a in ARMS}, "against": dict(BASE_FIT),
           "floors": FLOORS + ".json", "null_kind": NULL_KIND["zrckpbg"], "device": dev, "gates": sorted(gates),
           "next_form": nxt, "gains": gains, "losses": losses, "problems": problems,
           "calls": {tag(r): r["call"] for r in rrows}, "delta_r5": {tag(r): r["delta"] for r in rrows},
           "naming": NAMING, "wqk_verdict": WQ.sha_file(wv) if wv.is_file() else None,
           "zwqk_sha256": WQ.sha_src(K.__file__), "zr2gw_sha256": WQ.sha_src(__file__), "utc": WQ.utc()}
    md = [f"# R2GW, R2g under WQ on Kaggle's T4: zrckpbg **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zrckpbg** (zrckp + R2's bridge head, the head multiplied per question by 2 sigmoid(a . c_q + b)) against "
          f"WQK's seed-0 zrckp on the T4, re-called under WQK's zrckp seed null: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 7 before any number): {nxt}.", f"- Device: {json.dumps(dev)}.",
          f"- Gates: {', '.join(g[:12] for g in sorted(gates)) or 'none'}.", f"- Naming: {NAMING}."]
    beside = {}
    if v != "INCOMPLETE":
        fr, scr = K.fits_dir(base), K.screen_dir(base)
        zspk = {}
        for sp in SPLITS:
            _r, zr = K.cmp_rows(scr / f"{ZSPK_CMP[sp]}.json", ZSPK_CMP[sp], dev)
            zspk[sp] = {ds: zr[ds]["base"][0] for ds in WQ.EVAL_ORDER}
        side = {c: {tag(r): r["call"] for r in rc[c]["rows"]} for c in ("zrckpb", "zrckpbg-vs-zrckpb")}
        by = {c: {"gains": [k_ for k_, x in d.items() if x == "GAIN"], "losses": [k_ for k_, x in d.items() if x == "LOSS"]}
              for c, d in side.items()}
        gate_of = {sp: ((WQ.read_json(fr / FIT[(ARM, sp)] / "read.json").get(KEY) or {}).get("gate_stats") or {})
                   for sp in SPLITS}
        r5 = {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": zspk[sp][ds], "zrckp": r["base"],
                          "zrckpb": rows[CMP[(BESIDE_ARM, sp)]][BASE_FIT[sp]][ds]["new"][0], "zrckpbg": r["new"],
                          "floor": r["floor"], "call": r["call"], "zrckpb_vs_zrckp": side["zrckpb"][tag(r)],
                          "zrckpbg_vs_zrckpb": side["zrckpbg-vs-zrckpb"][tag(r)], "gate": gate_of[sp].get(f"{ds}=s1eval")}
        term = {}
        for (_a, _sp), n in FIT.items():
            t = ((WQ.read_json(fr / n / "screen.json").get(KEY) or {}).get("r2") or {}).get("term") or {}
            ep = t.get("epochs") or []
            term[n] = {"questions": t.get("questions"), "questions_with_term": t.get("questions_with_term"),
                       "share_with_term": t.get("share_with_term"),
                       "br_nll_first": ep[0].get("br_nll_mean") if ep else None,
                       "br_nll_last": ep[-1].get("br_nll_mean") if ep else None}
        tm = {sp: {"zrckpbg": timing(FIT[(ARM, sp)], fr), "zrckpb": timing(FIT[(BESIDE_ARM, sp)], fr),
                   "zrckp": K.timing(BASE_FIT[sp], fr, "wqk_timing.json")} for sp in SPLITS}
        rp = Path(str(K.at(base, RECALL["zrckpbg"])) + ".json")
        beside = {"calls": by, "r5": r5, "terms": term, "timing": tm, "pair": PAIR["zrckpbg"] + ".json",
                  "recall": RECALL["zrckpbg"] + ".json", "recall_sha256": WQ.sha_file(rp),
                  "pairs": {c: PAIR[c] + ".json" for c in CALLS}, "recalls": {c: RECALL[c] + ".json" for c in CALLS}}
        md += ["", "Beside, deciding nothing.", "",
               f"- zrckpb against WQK's seed-0 zrckp under the same null: {len(by['zrckpb']['gains'])} GAIN "
               f"({', '.join(by['zrckpb']['gains']) or 'none'}), {len(by['zrckpb']['losses'])} LOSS "
               f"({', '.join(by['zrckpb']['losses']) or 'none'}).",
               f"- zrckpbg against zrckpb (the null's seed-0 R@5 is R2GW's zrckpb read): "
               f"{len(by['zrckpbg-vs-zrckpb']['gains'])} GAIN ({', '.join(by['zrckpbg-vs-zrckpb']['gains']) or 'none'}), "
               f"{len(by['zrckpbg-vs-zrckpb']['losses'])} LOSS ({', '.join(by['zrckpbg-vs-zrckpb']['losses']) or 'none'}).",
               "", "R@5 per read on the T4, and zrckpbg's gate per question on the read (mean, share of questions below 1):",
               "", "| read | rrf | zspk | zrckp | zrckpb | zrckpbg | floor | call | zrckpb vs zrckp | zrckpbg vs zrckpb "
                   "| gate mean | share below 1 |", "|---|---:|---:|---:|---:|---:|---:|---|---|---|---:|---:|"]
        for k_, t in r5.items():
            gs = t["gate"] or {}
            md.append(f"| {k_} ({t['read']}) | {f4(t['rrf'])} | {f4(t['zspk'])} | {f4(t['zrckp'])} | {f4(t['zrckpb'])} "
                      f"| {f4(t['zrckpbg'])} | {f4(t['floor'])} | {t['call']} | {t['zrckpb_vs_zrckp']} "
                      f"| {t['zrckpbg_vs_zrckpb']} | {f4(gs.get('mean'))} | {f4(gs.get('share_below_1'))} |")
        md += ["", "R2's EM term on each fit (the share of training questions it reached, L_br at the first and the last "
                   "epoch):", "", "| fit | questions | with the term | share | L_br first | L_br last |",
               "|---|---:|---:|---:|---:|---:|"]
        for n, t in term.items():
            md.append(f"| {n} | {t['questions']} | {t['questions_with_term']} | {f4(t['share_with_term'])} "
                      f"| {f4(t['br_nll_first'])} | {f4(t['br_nll_last'])} |")

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        md += ["", "Seconds and memory: zrckpbg, and in brackets zrckpb and WQK's seed-0 zrckp of the same split, all on "
                   "the T4:", "", "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |",
               "|---|---:|---:|---|---:|"]
        for sp in SPLITS:
            a_, b_, c_ = tm[sp]["zrckpbg"], tm[sp]["zrckpb"], tm[sp]["zrckp"]
            md.append(f"| {sp} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}, {s0(c_['train_seconds'])}) | "
                      f"{s0(a_['read_seconds'])} ({s0(b_['read_seconds'])}, {s0(c_['read_seconds'])}) | "
                      f"{f4(a_['train_peak_gpu_gb'])} / {f4(a_['read_peak_gpu_gb'])} ({f4(b_['train_peak_gpu_gb'])} / "
                      f"{f4(b_['read_peak_gpu_gb'])}, {f4(c_['train_peak_gpu_gb'])} / {f4(c_['read_peak_gpu_gb'])}) | "
                      f"{f4(a_['peak_gb'])} ({f4(b_['peak_gb'])}, {f4(c_['peak_gb'])}) |")
    rec["beside"] = beside
    WQ.write(od, rec, md)
    log(f"verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


def real_state():
    """What R2GW has under the real folders: its fits, its compares and everything under outputs/r2gw."""
    w = WQ.ROOT / NEW
    return (sorted(p.name for p in K.fits_dir(WQ.ROOT).glob("scr-r2gw-*")),
            sorted(p.name for p in K.screen_dir(WQ.ROOT).glob("scr-r2gw-*")),
            sorted((p.relative_to(w).as_posix(), p.stat().st_mtime_ns) for p in w.rglob("*")) if w.is_dir() else [])


# ── selftest ─────────────────────────────────────────────────────────────────

FAKE_GATE = {"questions": 100, "mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0, "share_below_1": 0.0, "q05": 1.0,
             "q25": 1.0, "q50": 1.0, "q75": 1.0, "q95": 1.0, "questions_in_carve": 100, "share_margins_nonzero": 0.9}


def fake_fit(fits_root, name, gate_sha, overlay_sha, dev=None, read_=True, rows_cap=READ_CAP, r2=None, r2g="declared",
             stats=True):
    """An R2GW fit as the kernel leaves it: screen.json (WQK's wq block, R2GW's block) and read.json with their stamps,
    train.json, r2gw_timing.json, a read and models.pt. No gates npz: the take copies what the fit holds."""
    arm, sp = FITS[name]
    d = Path(fits_root) / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    dev = dev or K.device_of(K.TOY_ENV)
    r2 = r2 or {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(R2_PARAMS),
                "term": {"lam": 1.0, "questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                         "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}}
    k = {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": SEED, "split": sp, "device": dev, "gate": gate_sha,
         "overlay": overlay_sha, "r2": r2, "head": {"r2g": (R2G if arm == ARM else None) if r2g == "declared" else r2g}}
    WQ.write_json(d / "screen.json", {"arm": arm, "seed": SEED, "split": sp, "rc": 0, KEY: k,
                                      "wq": {"declared_in": K.DECLARED, "split": sp, "seed": SEED, "arm": arm,
                                             "gate": gate_sha}})
    WQ.write_json(d / "train.json", {"seconds": 150.0, "peak_gpu_gb": 3.5, "data_gb": 2.0})
    WQ.write_json(d / TIMING_FILE, {"peak_gb": 9.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if read_:
        WQ.write_json(d / "read.json", {
            "seconds": 12.0, "peak_gpu_gb": 4.5, "carves": {c: {} for c in K.SIX},
            KEY: {"declared_in": DECLARED, "fit": name, "arm": arm, "seed": SEED, "split": sp, "device": dev,
                  "gate": gate_sha, "rows_cap": rows_cap,
                  "gate_stats": {c: dict(FAKE_GATE) for c in K.SIX} if arm == ARM and stats else None}})
    return d


def toy_batch(R2, torch, seed=0):
    """A batch of two questions (3 and 4 rows) in the form R2's model reads: x, R2's inputs under R2's key, the margins."""
    g = torch.Generator().manual_seed(seed)
    sizes, off = (3, 4), (0, 3)
    nq = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
    n, nb = int(nq.numel()), len(sizes)
    bri = torch.full((nb, NB), -1, dtype=torch.long)
    for q, s in enumerate(sizes):
        for j in range(min(NB, s)):
            bri[q, j] = off[q] + j
    r2 = {"bri": bri, "eq": torch.randn(nb, R2.Q_DIM, generator=g) * 0.03,
          "p": torch.randn(n, R2.P_DIM, generator=g) * 0.1,
          "gm": torch.tensor([[0.5, 0.25, 0.125, 0.0625], [0.0, 0.0, 0.0, 0.0]])}
    feats = {"x": torch.randn(n, 2, generator=g), R2.R2_KEY: r2}
    return feats, nq, nb, torch.randn(n, generator=g)


def selftest():
    import tempfile
    import types
    t0 = time.time()
    before = real_state()
    stops, TOY = K.stops, K.TOY_ENV
    me = sys.modules[__name__]
    R = import_r2()
    R2, GateMix, ZRCKPBG, GCarve, np, torch = R["R2"], R["GateMix"], R["ZRCKPBG"], R["GCarve"], R["np"], R["torch"]
    nn = torch.nn
    # 1. the names: four fits, four compares on WQK's seed-0 zrckp, WQK's zspk compare, every module the stamps hash, R2's
    #    next-form texts, R2t's naming text
    assert FIT == {(ARM, "WQ-musique"): "scr-r2gw-zrckpbg", (ARM, "WQ-hotpotqa"): "scr-r2gw-zrckpbg-hp",
                   (BESIDE_ARM, "WQ-musique"): "scr-r2gw-zrckpb", (BESIDE_ARM, "WQ-hotpotqa"): "scr-r2gw-zrckpb-hp"}
    assert COMPARES["scr-r2gw-zrckpbg-vs-zrckp-hp"] == ("scr-r2gw-zrckpbg-hp",
                                                        ["scr-wqk-zrckp-s0-hp", "scr-r2gw-zrckpb-hp"])
    assert COMPARES["scr-r2gw-zrckpb-vs-zrckp"] == ("scr-r2gw-zrckpb", ["scr-wqk-zrckp-s0"]) and len(COMPARES) == 4
    assert ZSPK_CMP["WQ-musique"] == "scr-wqk-zspk-s1" and BASE_FIT["WQ-hotpotqa"] == "scr-wqk-zrckp-s0-hp"
    assert all((HERE / f"{m}.py").is_file() for m in CODE) and len(set(CODE)) == len(CODE)
    assert set(NEXT) == {"ADOPT_FOR_FULL_RUN", "NOT_ADOPTED", "INCOMPLETE"}
    assert NAMING_R2T == K.source_const(HERE / "zr2t.py", "NOT_BESIDE")["zrckpb"]
    assert R2E == R2.NEXT["NO_GAIN"] and R2.NEXT["GAIN_AND_LOSS"].startswith("R2g")
    assert R2.constants() == R2_CONST and R2.n_params() == R2_N == 59537 and (C_DIM, N_GATE) == (36, 37)
    assert R2.S.ARMS[ARM] == (ZRCKPBG, GCarve) and R2.S.ARMS[BESIDE_ARM] == R2.ARMS[BESIDE_ARM]
    # 2. zr2br: R2's declared constants, parameters, bridges and splits, else refused
    check_r2(R2)
    with WQ.patch(R2, DECLARED_CONST={**R2.DECLARED_CONST, "K": 6}):
        stops(check_r2, R2, has="not R2's declared")
    # 3. the margins by hand: ties, a NaN, a short pool, an empty pool, a single row, a negative top
    n = np.array([6, 3, 1, 0, 3, 2, 3])
    blocks = [[0.25, 1, 0.5, 0.125, 0.0625, 0.03125], [0.5, 0.25, 1.0], [0.7], [], [1.0, float("nan"), 0.5],
              [-1.0, -2.0], [0.5, 0.5, 0.25]]
    rrf = np.array([x for b in blocks for x in b], np.float32)
    br = R2.bridges_np(rrf, n)
    want = np.array([[.5, .25, .125, .0625], [.5, .25, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [.5, 0, 0, 0], [0, 0, 0, 0],
                     [0, .5, 0, 0]], np.float32)
    got = margins_np(rrf, n, br)
    assert got.dtype == np.float32 and np.array_equal(got, want), got
    stops(margins_np, rrf[:-1], n, br, has="margins over")
    for bad in (5, -2):
        br2 = br.copy()
        br2[2, 0] = bad
        stops(margins_np, rrf, n, br2, has="outside its question's pool")
    assert margins_np(np.zeros(0, np.float32), np.zeros(0, np.int64), np.zeros((0, NB), np.int64)).shape == (0, M_DIM)
    # 4. the carve: R2's carve with each question's margins in its batches
    assert (GCarve.__mro__[1], GCarve.__name__) == (R2.R2KTChainCarveZRC, "GR2KTChainCarveZRC")
    assert ZRCKPBG.__mro__[1:4] == (GateMix, R2.ZRCKPB, R2.R2Mix)
    gm = np.array([[0.5, 0, 0, 0], [0.25, 0.125, 0, 0], [0, 0, 0, 0]], np.float32)

    def base_batch(self, qs, blocks_):
        return {R2.R2_KEY: {"bri": 0}}, "nq", "base_z", "gold"

    with WQ.patch(R2.R2KTChainCarveZRC, batch=base_batch, nbytes=lambda self: 100):
        c = GCarve.__new__(GCarve)
        c.r2g_gm, c.device = gm, "cpu"
        f_, nq_, bz_, gd_ = c.batch([2, 0], None)
        assert torch.equal(f_[R2.R2_KEY]["gm"], torch.from_numpy(gm[[2, 0]])) and (nq_, bz_, gd_) == ("nq", "base_z", "gold")
        assert c.nbytes() == 100 + gm.nbytes
    # 5. the model on a toy base: the gate is 1 and zrckpbg is zrckpb bit for bit; c_q by hand; 37 gate numbers; a moved
    #    gate scales the head; no R2 input leaves the base score; a batch without margins is refused
    class ToyBase(nn.Module):
        def __init__(self, blocks_, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
            super().__init__()
            self.w = nn.Parameter(torch.tensor([0.5, -0.25]))

        def forward(self, feats, keep, nq, B, base_z):
            return feats["x"] @ self.w + base_z

    class ToyB(R2.R2Mix, ToyBase):
        pass

    class ToyBG(GateMix, ToyB):
        pass

    feats, nq, nb, base_z = toy_batch(R2, torch)
    r2 = feats[R2.R2_KEY]
    mb, mg = ToyB(None, None), ToyBG(None, None)
    with torch.no_grad():
        mb.r2_w2.copy_(torch.randn(mb.r2_w2.shape, generator=torch.Generator().manual_seed(1)) * 0.2)
        mb.r2_b2.fill_(0.05)
    res = mg.load_state_dict(mb.state_dict(), strict=False)
    assert set(res.missing_keys) == set(GATE_PARAMS) and not res.unexpected_keys
    count = lambda m, pre: sum(p.numel() for k_, p in m.named_parameters() if k_.startswith(pre))     # noqa: E731
    assert (count(mb, "r2"), count(mg, "r2"), count(mg, "r2g_")) == (R2_N, R2_N + N_GATE, N_GATE)
    assert [k_ for k_, _p in mg.named_parameters() if k_.startswith("r2g_")] == list(GATE_PARAMS)
    c_q = mg.r2g_c(r2)
    assert c_q.shape == (nb, C_DIM) and torch.equal(c_q, torch.cat([r2["eq"] @ mg.r2_U, r2["gm"]], 1))
    assert torch.equal(mg.r2g_gate(r2), torch.ones(nb))
    for mode in ("eval", "train"):
        getattr(mb, mode)(), getattr(mg, mode)()
        assert torch.equal(mb(feats, None, nq, nb, base_z), mg(feats, None, nq, nb, base_z)), mode
    assert float(mb.r2_head(mb.r2_feats(r2, nq)).abs().max()) > 0
    with torch.no_grad():
        mg.r2g_a.copy_(torch.linspace(-0.05, 0.05, C_DIM))
        mg.r2g_b.fill_(0.3)
    y, gq = mg(feats, None, nq, nb, base_z), mg.r2g_gate(r2)
    assert torch.allclose(gq, 2 * torch.sigmoid(c_q @ mg.r2g_a + mg.r2g_b)) and not torch.equal(gq, torch.ones(nb))
    s_ = feats["x"] @ mg.w + base_z
    assert torch.allclose(y, s_ + gq[nq] * mg.r2_head(mg.r2_feats(r2, nq))) and not torch.equal(y, mb(feats, None, nq, nb, base_z))
    assert torch.equal(mg({"x": feats["x"]}, None, nq, nb, base_z), s_)
    stops(mg, {"x": feats["x"], R2.R2_KEY: {k_: v for k_, v in r2.items() if k_ != "gm"}}, None, nq, nb, base_z,
          has="holds no margins")
    # 6. R2's EM term: the same bridge logits kept with and without the gate, a batch kept twice refused; the gate's gradient
    #    is exactly 0 while the head's last layer is 0 and not 0 once it has moved; a fit with no term is refused
    mg0 = ToyBG(None, None)
    with R2.r2_em():
        mb.train(), mg.train()
        R2.clear_stash()
        mb(feats, None, nq, nb, base_z)
        kb = [R2.STASH[k_] for k_ in ("S", "M", "lpt")]
        R2.clear_stash()
        mg(feats, None, nq, nb, base_z)
        kg = [R2.STASH[k_] for k_ in ("S", "M", "lpt")]
        assert all(torch.equal(a, b) for a, b in zip(kb, kg)) and R2.STASH["N"] == int(nq.numel())
        stops(mg, feats, None, nq, nb, base_z, has="kept and never scored")
        R2.clear_stash()
    assert R2.STASH["S"] is None
    mg0(feats, None, nq, nb, base_z).sum().backward()
    assert int(torch.count_nonzero(mg0.r2g_a.grad)) == 0 and int(torch.count_nonzero(mg0.r2g_b.grad)) == 0
    assert int(torch.count_nonzero(mg0.r2_w2.grad)) > 0
    mg.zero_grad()
    mg(feats, None, nq, nb, base_z).sum().backward()
    assert int(torch.count_nonzero(mg.r2g_a.grad)) > 0 and int(torch.count_nonzero(mg.r2g_b.grad)) > 0
    stops(term_ok, {"batches": 7, "batches_with_term": 0}, "scr-r2gw-zrckpbg", has="carried R2's term (7 batches)")
    term_ok({"batches": 7, "batches_with_term": 3}, "scr-r2gw-zrckpbg")
    # 7. the gate per question in a read: only p@swa's, in question order, the patch removed after (an error too); a read
    #    without a gated p@swa, or with a count off, refused; zrckpb's read is not patched
    mg.eval(), mb.eval()
    models = [("p", mb, []), ("p@swa", mg, [])]
    cs = types.SimpleNamespace(ds="metaqa", carve="s1eval", n_np=np.ones(4), r2g_gm=np.array(
        [[0, 0, 0, 0], [.5, 0, 0, 0], [0, 0, 0, 0], [.1, .2, 0, 0]], np.float32))

    def fake_orig(c_, models_, *a, **kw):
        for _nm, m, _bl in models_:
            m(feats, None, nq, nb, base_z)
            m(feats, None, nq, nb, base_z)
        return ("out", a, kw)

    GATES.clear()
    assert gate_reader(fake_orig)(cs, models, 65536, first_q=3) == ("out", (65536,), {"first_q": 3})
    d = GATES["metaqa=s1eval"]
    with torch.no_grad():
        g2 = mg.r2g_gate(r2)
    assert np.array_equal(d["gate"], torch.cat([g2, g2]).numpy()) and (d["questions_in_carve"], d["share_margins_nonzero"]) == (4, 0.5)
    assert "r2g_seen" not in mg.__dict__ and "r2g_seen" not in mb.__dict__
    GATES.clear()

    def bad_orig(c_, models_, *a, **kw):
        raise RuntimeError("inside")

    try:
        gate_reader(bad_orig)(cs, models)
        raise AssertionError("did not raise")
    except RuntimeError:
        pass
    assert "r2g_seen" not in mg.__dict__ and not GATES
    stops(gate_reader(fake_orig), cs, [("p", mg, [])], has="holds no p@swa candidate of zrckpbg")
    stops(gate_reader(fake_orig), cs, [("p@swa", mb, [])], has="holds no p@swa candidate of zrckpbg")
    stops(gate_reader(fake_orig), types.SimpleNamespace(ds="squad", carve="s1eval", n_np=np.ones(3)), models,
          has="squad=s1eval: kept 4 gates for 3 questions")
    assert "r2g_seen" not in mg.__dict__ and not GATES
    lg = types.SimpleNamespace(read_carve=fake_orig)
    with gate_reads(lg, BESIDE_ARM):
        assert lg.read_carve is fake_orig
    with gate_reads(lg, ARM):
        assert lg.read_carve is not fake_orig
    assert lg.read_carve is fake_orig
    assert gate_stats([0.5, 1.0, 1.5])["share_below_1"] == 0.333333 and gate_stats([]) == {"questions": 0}
    # 8. inside zwq.on_wq: zr2br trains on zwq's modules; the arm swap is zp1c3p's for the block and restored after
    with K.on_mixed_kt(), K.on_wqk(), WQ.on_wq() as W:
        same_modules(R2, W)
        other = types.SimpleNamespace(Z3=W.Z3, Z2=W.Z2, LG=json, S=W.S, S2=W.S2, LC=W.LC, Z3P=W.Z3P)
        stops(same_modules, R2, other, has="does not train on zwq's ['json']")
        saved = (W.Z3P.ARM, dict(W.Z3P.ARMS))
        for a_ in ARMS:
            with on_arm(W, a_) as W2:
                assert W2 is W and W.Z3P.ARM == a_ and W.Z3P.ARMS == {a_: R2.S.ARMS[a_]}
        assert (W.Z3P.ARM, dict(W.Z3P.ARMS)) == saved
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        B = td / "repo"
        # 9. where, the read cap, WQK's verdict: train, read and fit refuse before reading any data
        stops(arm_split, "scr-wqk-zrckp-s0", has="not one of R2GW's fits")
        stops(train, "scr-r2-zrckpb", TOY, has="not one of R2GW's fits")
        for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"}, {"image": "kaggle-gpu:other"}):
            stops(where_ok, {**TOY, **bad}, "train", has="R2GW trains in a Kaggle kernel")
            stops(train, FIT[(ARM, "WQ-musique")], {**TOY, **bad}, has="R2GW trains in a Kaggle kernel")
            stops(read, FIT[(BESIDE_ARM, "WQ-hotpotqa")], {**TOY, **bad}, has="R2GW reads in a Kaggle kernel")
        where_ok(TOY, "read")
        cap_ok(READ_CAP)
        for cap in (0, None, 32768):
            stops(cap_ok, cap, has="--rows-cap 65536")
            stops(read, FIT[(ARM, "WQ-musique")], TOY, rows_cap=cap, has="--rows-cap 65536")
            stops(fit, FIT[(ARM, "WQ-musique")], rows_cap=cap, has="--rows-cap 65536")
        stops(fit, FIT[(BESIDE_ARM, "WQ-hotpotqa")], has="R2GW trains in a Kaggle kernel")     # the laptop is not a T4
        stops(wqk_ready, B, has="base-verdict.json missing")
        wv = B / (K.VERDICT + ".json")
        with WQ.patch(WQ, ROOT=B):
            stops(train, FIT[(ARM, "WQ-musique")], TOY, has="base-verdict.json missing")
            for bad in ({"verdict": "INCOMPLETE"}, {"declared_in": WQ.DECLARED}, {"stage": "WQ"}):
                WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY", **bad})
                stops(wqk_ready, B, has="not WQK's BASE_READY")
                stops(train, FIT[(BESIDE_ARM, "WQ-musique")], TOY, has="not WQK's BASE_READY")
        WQ.write_json(wv, {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        assert wqk_ready(B)["verdict"] == "BASE_READY"
        shutil.rmtree(B / "outputs")
        # 10. the stamps of all four fits (WQK's wq block at the toy root, R2GW's block), fit_stamp, the read's stamp
        #     with the gate per question, fit_device; then the read's refusals on the stamped fits
        B7 = td / "stamp"
        for k_, r_ in (("GATE", {"verdict": "PASS"}), ("OVERLAY", {"status": "OK"}), ("CHECK", {"status": "REPRODUCED"})):
            WQ.write_json(B7 / K.RECORDS[k_], {"declared_in": K.DECLARED, **r_})
        WQ.write_json(B7 / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        summ = {"lam": 1.0, "batches": 10, "batches_with_term": 9, "questions": 100, "questions_with_term": 90,
                "share_with_term": 0.9, "epochs": [{"br_nll_mean": 4.6}, {"br_nll_mean": 3.9}]}
        fr7 = K.fits_dir(B7)
        for nm, (a_, sp_) in FITS.items():
            WQ.write_json(WQ.fit_dir(nm, B7) / "screen.json", {"arm": a_, "seed": SEED, "split": sp_, "rc": 0})
            stops(fit_stamp, nm, fr7, has="no R2GW stamp")
            stamp_fit(nm, TOY, R2, summ, B7, {"kt_records": {"webqsp/fit": "k"}})
            s7, k7 = fit_stamp(nm, fr7)
            assert (k7["declared_in"], k7["device"], k7["against"], k7["what"]) == (DECLARED, K.device_of(TOY),
                                                                                   BASE_FIT[sp_], WHAT[a_])
            assert (s7["wq"]["declared_in"], s7["wq"]["gate"]) == (K.DECLARED, WQ.sha_file(B7 / K.RECORDS["GATE"]))
            assert k7["gate"] == s7["wq"]["gate"] and k7["wqk_verdict"] == WQ.sha_file(B7 / (K.VERDICT + ".json"))
            assert (k7["r2"]["constants"], k7["r2"]["params"], k7["r2"]["term"]) == (R2_CONST, list(R2_PARAMS), summ)
            assert k7["head"]["r2g"] == (R2G if a_ == ARM else None) and k7["head"]["read_rows"] == READ_CAP
            assert s7["zr2br_sha256"] == WQ.sha_src(HERE / "zr2br.py") and s7["zr2gw_sha256"] == WQ.sha_src(__file__)
            assert set(k7["code"]) == set(CODE) and k7["kt_records"] == {"webqsp/fit": "k"}
        stops(fit_stamp, FIT[(ARM, "WQ-musique")], fr7 / "none", has="no R2GW stamp")
        n7, nb7 = FIT[(ARM, "WQ-hotpotqa")], FIT[(BESIDE_ARM, "WQ-hotpotqa")]
        for nm in (n7, nb7):
            WQ.write_json(WQ.fit_dir(nm, B7) / "read.json", {"seconds": 12.0, "carves": {c: {} for c in K.SIX}})
        stops(fit_device, n7, fr7, has="has no R2GW read on the device it was fitted on")
        gates = {"metaqa=s1eval": {"gate": np.array([0.5, 1.0, 1.5], np.float32), "questions_in_carve": 3,
                                   "share_margins_nonzero": 1.0}}
        r7 = stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, gates, B7)
        rk = r7[KEY]
        assert rk["gate_stats"]["metaqa=s1eval"] == {**gate_stats([0.5, 1.0, 1.5]), "questions_in_carve": 3,
                                                     "share_margins_nonzero": 1.0}
        assert rk["gate_stats"]["metaqa=s1eval"]["share_below_1"] == 0.333333 and rk["rows_cap"] == READ_CAP
        gz = WQ.fit_dir(n7, B7) / GATES_FILE
        assert rk["gates_sha256"] == WQ.sha_file(gz) and list(np.load(gz).files) == ["metaqa__s1eval"]
        assert fit_device(n7, fr7) == K.device_of(TOY)
        rb = stamp_read(nb7, TOY, {"webqsp/fit": "k"}, READ_CAP, {}, B7)[KEY]
        assert rb["gate_stats"] is None and rb["gates_sha256"] is None and not (WQ.fit_dir(nb7, B7) / GATES_FILE).exists()
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, 32768, gates, B7)
        stops(fit_device, n7, fr7, has="has no R2GW read on the device it was fitted on")
        stamp_read(n7, TOY, {"webqsp/fit": "k"}, READ_CAP, gates, B7)
        assert stamped(WQ.fit_dir(n7, B7) / "read.json", n7) and not stamped(WQ.fit_dir(n7, B7) / "read.json", nb7)
        boom = types.SimpleNamespace(n=0)

        def gate_boom(base=None):
            boom.n += 1
            raise SystemExit("zwq: outputs/wqk/gate.json is FAIL, not WQ's PASS")

        with WQ.patch(WQ, fit_dir=lambda name, base=B7: Path(base) / "outputs" / "screen" / "fits" / name):
            stops(read, n7, {**TOY, "cudnn": 1}, has="this read runs on")
            sj7 = WQ.fit_dir(n7, B7) / "screen.json"
            keep7 = sj7.read_bytes()
            s7 = WQ.read_json(sj7)
            s7[KEY]["r2"]["constants"] = {**R2_CONST, "K": 6}
            WQ.write_json(sj7, s7)
            stops(read, n7, TOY, has="fitted with R2 constants")
            sj7.write_bytes(keep7)
            with WQ.patch(WQ, gate_passed=gate_boom):
                stops(read, n7, TOY, has="is FAIL")
            assert boom.n == 1
        # 11. take, compare-all and the verdict on a toy of WQK's base set (zwqk's toy, its fits, compares and floors) and
        #     an R2GW session: ADOPT_FOR_FULL_RUN, NOT_ADOPTED both ways, and INCOMPLETE for each missing or wrong piece
        K.toy_build(B)
        gw, ow = K.fake_session(td / "wqk_out", B, tag="w")
        for nm in K.FITS:
            K.fake_fit(K.fits_dir(B), nm, gw, ow)

        def wqk_run(new, base_fit, out):                # zwqk's selftest's: WQK's compares on the toy
            new, b0, c_ = Path(new).name, Path(base_fit).name, Path(out).name
            arm, seed, sp = K.FITS[new]
            dd = {ds: (0.002, -0.002)[seed - 1] for ds in WQ.EVAL_ORDER} if arm != "zg6" else {}
            if (arm, sp) == ("zrckp", "WQ-musique"):
                dd["musique"] = (0.02, -0.01)[seed - 1]
            K.fake_compare(B, c_, new, b0, dd, base_r5=0.59 if arm == "zrckp" else 0.6)
            return 0

        why = WQ.wq_why
        with WQ.patch(WQ, run_compare=wqk_run, wq_why=lambda name, base=B: why(name, base)):
            assert K.compare_all(B) == 0
        assert K.floors(B) == 0
        WQ.write_json(B / (K.VERDICT + ".json"), {"stage": K.STAGE, "declared_in": K.DECLARED, "verdict": "BASE_READY"})
        out_g = td / "r2gw_out"
        gg, og = K.fake_session(out_g, B, tag="g")
        names = list(FITS)
        for nm in names[:3]:
            fake_fit(out_g / "fits", nm, gg, og)
        fake_fit(out_g / "fits", names[3], gg, og, read_=False)
        took, left = take(out_g, B)
        assert len(took) == 3 * 5 + 2 and len(left) == 1 and "has no R2GW read" in left[0], (took, left)
        fr = K.fits_dir(B)
        assert not (fr / names[0] / "models.pt").exists() and (fr / names[0] / TIMING_FILE).is_file()
        assert (B / SESSIONS / gg[:12] / "gate.json").is_file() and not (fr / names[3]).exists()
        fake_fit(out_g / "fits", names[3], gg, og)
        took, left = take(out_g, B)
        assert len(took) == 5 and not left and take(out_g, B) == ([], [])
        (fr / names[0] / "train.json").write_text("{}", encoding="utf-8")
        stops(take, out_g, B, has="exists with other bytes")
        shutil.copyfile(out_g / "fits" / names[0] / "train.json", fr / names[0] / "train.json")
        (out_g / "fits" / BASE_FIT["WQ-musique"]).mkdir()
        stops(take, out_g, B, has="are not R2GW's fits")
        (out_g / "fits" / BASE_FIT["WQ-musique"]).rmdir()
        stops(take, td / "wqk_out" / "wqk", B, has="holds no wqk/gate.json")
        cur = {}

        def r2gw_run(new, base_fit, out):               # lean_screen's compare with one or two bases, in its record's form
            import lean_screen as S
            n_, c_ = Path(new).name, Path(out).name
            arm, sp = FITS[n_]
            bs = [Path(b).name for b in base_fit.split(",")]
            assert bs == COMPARES[c_][1] and c_ == CMP[(arm, sp)], (bs, c_)
            by = {}
            for b in bs:
                rows = []
                for ds in WQ.EVAL_ORDER:
                    nw = round(0.59 + cur.get((arm, sp, ds), 0.0), 4)
                    r5 = 0.59 if b == BASE_FIT[sp] else round(0.59 + cur.get((BESIDE_ARM, sp, ds), 0.0), 4)
                    dl = round(nw - r5, 4)
                    lo, hi = round(dl - 0.004, 4), round(dl + 0.004, 4)
                    rows.append({"dataset": ds, "read": "in-domain" if ds in WQ.SPLITS[sp] else "zero-shot",
                                 "questions": 100, "base": [r5, 0.5, 0.4], "new": [nw, 0.5, 0.4],
                                 "rrf": [0.4, 0.3, 0.2], "delta": [dl, 0, 0], "lo": [lo, 0, 0], "hi": [hi, 0, 0],
                                 "call": S.call(dl, lo, hi)})
                by[f"outputs/screen/fits/{b}"] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]),
                                                  "decides": b == bs[0]}
            bp = list(by)
            WQ.write_json(K.screen_dir(B) / f"{c_}.json", {
                "new": f"outputs/screen/fits/{n_}", "bases": bp, "candidate": "p@swa", "carve": "s1eval",
                "trained_on": sorted(WQ.SPLITS[sp]), "by_base": by, "verdict": by[bp[0]]["verdict"]})
            return 0

        def run(deltas):
            cur.clear()
            cur.update(deltas)
            for c_ in COMPARES:
                (K.screen_dir(B) / f"{c_}.json").unlink(missing_ok=True)
            with WQ.patch(WQ, run_compare=r2gw_run):
                assert compare_all(B) == 0 and compare_all(B) == 0
            return verdict(B)

        v = run({(ARM, "WQ-hotpotqa", "squad"): 0.03, (BESIDE_ARM, "WQ-musique", "metaqa"): 0.02})
        assert v["verdict"] == "ADOPT_FOR_FULL_RUN" and not v["problems"], v["problems"]
        assert (v["gains"], v["losses"], v["next_form"]) == (["WQ-hotpotqa squad"], [], NEXT["ADOPT_FOR_FULL_RUN"])
        assert v["gates"] == [gg] and v["device"] == K.device_of(TOY) and len(v["calls"]) == 12 and v["naming"] == NAMING
        k_ = WQ.read_json(K.screen_dir(B) / f"{CMP[(ARM, 'WQ-hotpotqa')]}.json")[KEY]
        assert (k_["new"], k_["bases"], k_["device"]) == (FIT[(ARM, "WQ-hotpotqa")],
                                                          [BASE_FIT["WQ-hotpotqa"], FIT[(BESIDE_ARM, "WQ-hotpotqa")]],
                                                          K.device_of(TOY))
        b = v["beside"]
        assert b["calls"]["zrckpb"] == {"gains": ["WQ-musique metaqa"], "losses": []}
        assert b["calls"]["zrckpbg-vs-zrckpb"] == {"gains": ["WQ-hotpotqa squad"], "losses": ["WQ-musique metaqa"]}
        assert b["r5"]["WQ-musique metaqa"] == {
            "read": "zero-shot", "rrf": 0.4, "zspk": 0.6, "zrckp": 0.59, "zrckpb": 0.61, "zrckpbg": 0.59, "floor": 0.0075,
            "call": "WITHIN", "zrckpb_vs_zrckp": "GAIN", "zrckpbg_vs_zrckpb": "LOSS", "gate": FAKE_GATE}
        assert b["r5"]["WQ-musique musique"]["floor"] == round(2 * (((0.02 ** 2 + 0.01 ** 2) / 2) ** 0.5), 6)
        assert b["terms"][FIT[(ARM, "WQ-musique")]] == {"questions": 100, "questions_with_term": 90, "share_with_term": 0.9,
                                                        "br_nll_first": 4.6, "br_nll_last": 3.9}
        assert b["timing"]["WQ-musique"]["zrckpbg"] == {"train_seconds": 150.0, "read_seconds": 12.0,
                                                        "train_peak_gpu_gb": 3.5, "read_peak_gpu_gb": 4.5, "data_gb": 2.0,
                                                        "peak_gb": 9.0}
        assert b["timing"]["WQ-hotpotqa"]["zrckp"]["peak_gb"] == 8.0
        assert b["recall_sha256"] == WQ.sha_file(B / (RECALL["zrckpbg"] + ".json"))
        for c_ in CALLS:
            r_ = WQ.read_json(B / (RECALL[c_] + ".json"))
            assert (r_["stage"], r_["against"], r_["deciding"], len(r_["rows"])) == (STAGE, AGAINST[c_], DECIDING[c_], 12)
            assert (B / (PAIR[c_] + ".md")).read_text(encoding="utf-8").startswith(f"R2GW ({DECLARED})")
        r1 = WQ.read_json(B / (RECALL["zrckpbg-vs-zrckpb"] + ".json"))
        assert {(r["split"], r["dataset"]): r["base"] for r in r1["rows"]}[("WQ-musique", "metaqa")] == 0.61
        md = (B / (VERDICT + ".md")).read_text(encoding="utf-8")
        assert "zrckpbg **ADOPT_FOR_FULL_RUN**" in md and "| WQ-hotpotqa squad (in-domain) | 0.4000 | 0.6000 | 0.5900 |" in md
        v = run({(ARM, "WQ-hotpotqa", "squad"): 0.03, (ARM, "WQ-musique", "webqsp"): -0.03})
        assert (v["verdict"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", ["WQ-musique webqsp"], NEXT["NOT_ADOPTED"])
        v = run({(ARM, "WQ-musique", "musique"): 0.02})        # a filed GAIN within the raised floor (0.0316)
        assert (v["verdict"], v["gains"], v["losses"], v["next_form"]) == ("NOT_ADOPTED", [], [], NEXT["NOT_ADOPTED"])
        assert v["calls"]["WQ-musique musique"] == "WITHIN" and v["delta_r5"]["WQ-musique musique"] == 0.02
        r0 = WQ.read_json(B / (RECALL["zrckpbg"] + ".json"))
        assert [r["call_filed"] for r in r0["rows"] if (r["split"], r["dataset"]) == ("WQ-musique", "musique")] == ["GAIN"]

        def incomplete(has):
            r = verdict(B)
            assert r["verdict"] == "INCOMPLETE" and r["beside"] == {} and r["next_form"] == NEXT["INCOMPLETE"] and \
                any(has in p for p in r["problems"]), (has, r["problems"])

        def away(p, has):
            keep_ = p.read_bytes()
            p.unlink()
            incomplete(has)
            p.write_bytes(keep_)

        cg = K.screen_dir(B) / f"{CMP[(ARM, 'WQ-hotpotqa')]}.json"
        away(cg, f"{CMP[(ARM, 'WQ-hotpotqa')]}.json missing")
        away(B / (K.VERDICT + ".json"), "base-verdict.json missing")
        keep = cg.read_bytes()
        rc_ = WQ.read_json(cg)
        rc_[KEY]["new"] = "scr-r2gw-zrckpb-hp"
        WQ.write_json(cg, rc_)
        incomplete("is not R2GW's compare of")
        cg.write_bytes(keep)
        n0, nb0 = FIT[(ARM, "WQ-musique")], FIT[(BESIDE_ARM, "WQ-musique")]
        bad_r2 = {"declared_in": R2_DECLARED, "constants": {**R2_CONST, "K": 6}, "params": list(R2_PARAMS),
                  "term": {"lam": 1.0, "questions_with_term": 90}}
        no_term = {"declared_in": R2_DECLARED, "constants": dict(R2_CONST), "params": list(R2_PARAMS),
                   "term": {"lam": 1.0, "questions_with_term": 0}}
        for nm, kw, has in (
                (n0, {"dev": {**K.device_of(TOY), "gpu": "Tesla P100"}}, "ran on different devices"),
                (n0, {"gate_sha": "f" * 64}, "names the gate ffffffffffff, not a filed R2GW PASS gate"),
                (n0, {"r2": bad_r2}, f"{n0}: R2's constants or EM term are not as declared"),
                (nb0, {"r2": no_term}, f"{nb0}: R2's constants or EM term are not as declared"),
                (n0, {"rows_cap": 32768}, "has no R2GW read on the device it was fitted on"),
                (n0, {"r2g": None}, f"{n0}: its gate is not R2g's as declared"),
                (n0, {"stats": False}, f"{n0}: its read holds no gate per question on the six carves"),
                (nb0, {"r2g": R2G}, f"{nb0}: zrckpb is gated")):
            fake_fit(fr, nm, **{"gate_sha": gg, "overlay_sha": og, **kw})
            incomplete(has)
            fake_fit(fr, nm, gg, og)
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
        out2 = td / "r2gw_out2"
        g2_, _o2 = K.fake_session(out2, B, verdict_="FAIL", tag="h")
        take(out2, B)
        incomplete("FAIL, not WQK's PASS")
        shutil.rmtree(B / SESSIONS / g2_[:12])
        p = K.screen_dir(B) / "scr-wqk-zrckp-s1.json"
        keep = p.read_bytes()
        r = WQ.read_json(p)
        r["by_base"]["outputs/screen/fits/scr-wqk-zrckp-s0"]["rows"][2]["delta"][0] = 0.05
        WQ.write_json(p, r)
        incomplete("floors-zrckp.json is not its compares' floors")
        p.write_bytes(keep)

        def short(base, dev, c_):
            rec = recall.__wrapped__(base, dev, c_)
            rec["rows"] = rec["rows"][:-1]
            return rec

        short_fn = types.FunctionType(short.__code__, {**globals(), "recall": types.SimpleNamespace(__wrapped__=recall)})
        with WQ.patch(me, recall=short_fn):
            incomplete("holds 11 reads, not the twelve")
        assert verdict(B)["verdict"] == "NOT_ADOPTED"
    # 12. nothing left patched in zr2br or lean_gpu, no gate state kept, and nothing written under R2GW's real folders
    assert R2.STASH["S"] is None and not GATES and "r2g_seen" not in mg.__dict__
    assert real_state() == before, (real_state(), before)
    print(f"zr2gw selftest: ok ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    import argparse
    ap = argparse.ArgumentParser(prog="zr2gw.py", allow_abbrev=False)
    ap.add_argument("cmd", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=READ_CAP)
    ap.add_argument("--from", dest="src")
    a = ap.parse_args(argv)
    src = Path(a.src).resolve() if a.src else None
    os.chdir(WQ.ROOT)
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    if a.cmd == "take":
        if src is None:
            ap.error("take takes --from")
        take(src)
        return 0
    if a.cmd == "compare-all":
        return compare_all()
    return 0 if verdict()["verdict"] != "INCOMPLETE" else 2


if __name__ == "__main__":
    sys.exit(main())
