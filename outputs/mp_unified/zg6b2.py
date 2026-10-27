"""G6b2 (docs/G6B2_ROW_BACKOFF.md): G6b's next form after its GAIN and LOSS (docs/G6B_GATED_BACKOFF.md, sections 6
and 9), a per-row back-off, on Kaggle's T4 against KG's base set.

zg6b2 is zg6b's network (zg6's, scored twice from one encoder pass and one context: over the question's pool edges,
s_e, and with no edge reaching its layers, s_t) with the gate moved from the question to the row:
    score_r = g_r s_e,r + (1 - g_r) s_t,r,   g_r = sigmoid(a . sh_q + c . log1p(deg_r) + b),
sh_q being zg6b's family shares among question q's edges that reach the layers, and deg_r row r's in-degree per family
over those edges (an edge u -> v counted at v, after the family dropout in training: zg6.g6_layers' deg before its
clamp). a, b and c start at 0, so g = 0.5 and the score is zg6b's at the start. A row no edge reaches has s_e = s_t, so
its score is the twin's whatever its gate, and it gives the gate no gradient. Everything else is zg6b's, and through it
zg6's. 121,809 own parameters at hidden 128: zg6b's 121,806 and c's 3.

The fits are zg6b2's two on U2d's base build (scr-g6b2-zg6b2 on L-musique, scr-g6b2-zg6b2-hp on L-hotpotqa, seed 0),
trained and read through zu2d.py's bodies as KG runs them, on KG's device. In the kernel (zkg_session.py runs it, one
process per fit):

    python outputs/mp_unified/zg6b2.py fit --name N [--rows-cap R]

On the laptop, after `kaggle kernels output` into a new empty folder (KG's BASE_READY and G6b's verdict filed):

    python outputs/mp_unified/zg6b2.py take --from DIR       # each fit's records, reads and gate file, never its weights
    python outputs/mp_unified/zg6b2.py compare-all           # against KG's zspk (deciding), KG's zg6 and G6b's zg6b
    python outputs/mp_unified/zg6b2.py verdict               # G6's rule under KG's zspk null, and the beside tables
    python outputs/mp_unified/zg6b2.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zg6b as ZB  # noqa: E402  (G6b's network, gate and fits; through it KG's device, null and base set)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

KG = ZB.KG
Z, G6, G2 = KG.Z, KG.G6, KG.G2
Z2, Z3, RM, ZL, SP = KG.Z2, KG.Z3, KG.RM, KG.ZL, KG.SP
S, S2, LG, LC, SR = KG.S, KG.S2, KG.LG, KG.LC, KG.SR
log = KG.log

DECLARED = "docs/G6B2_ROW_BACKOFF.md"
STAGE = "G6b2"
KEY = "g6b2"                                 # the stamp's key in screen.json, read.json and G6b2's compares
ARM = "zg6b2"
SEED = 0
BUILD, SPLITS, SUF = KG.BUILD, KG.SPLITS, KG.SUF
FAMS, KT_KEY, LK_KEY, ZCLIP = G6.FAMS, G6.KT_KEY, G6.LK_KEY, G6.ZCLIP
SCREEN = KG.SCREEN                           # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "g6b2"
VERDICT = "screen-verdict"
AGAINST = {0: "zspk", 1: "zg6", 2: "zg6b"}
PAIR = {0: "g6b2-zg6b2-pair", 1: "g6b2-zg6b2-vs-zg6-pair", 2: "g6b2-zg6b2-vs-zg6b-pair"}
RECALL = {k: f"{p}-recall" for k, p in PAIR.items()}
NULL_KIND = {0: "KG's zspk seed null",
             1: "KG's zspk seed null, its seed-0 R@5 replaced by KG's zg6 read (beside)",
             2: "KG's zspk seed null, its seed-0 R@5 replaced by G6b's zg6b read (beside)"}
GATE_FILE = "g6b2_gate.json"
TIMING_FILE = "g6b2_timing.json"
SHARE_MIN = G6.SHARE_MIN
G6B2_NAMES = ("g6b2_row_w",)
GATE_NAMES = ZB.G6B_NAMES + G6B2_NAMES
N_PARAMS = ZB.N_PARAMS + FAMS
DEGREES = (1, 3, 10, 30, 100)                # in-degrees at which the gate beside shows a row of one family alone
KEY_STRIDE = 1 << 32                         # a reached row's key beside: question * KEY_STRIDE + its row (zlink's e_v)


def fit_name(split):
    return f"scr-g6b2-{ARM}{SUF[split]}"


FITS = {fit_name(sp): sp for sp in SPLITS}
ZSPK, ZG6 = ZB.ZSPK, ZB.ZG6                                         # KG's zspk (deciding) and zg6 (beside), seed 0
ZG6B = {sp: ZB.fit_name(sp) for sp in SPLITS}                      # beside: G6b's zg6b, on the same T4
CMP = {sp: f"scr-g6b2-{ARM}-vs-zspk{SUF[sp]}" for sp in SPLITS}
COMPARES = {CMP[sp]: (fit_name(sp), [ZSPK[sp], ZG6[sp], ZG6B[sp]]) for sp in SPLITS}
KG_ZG6_CMP = ZB.KG_ZG6_CMP                                         # zg6's R@5 as KG filed it
G6B_CMP = ZB.CMP                                                   # zg6b's R@5 as G6b filed it (its compares' new)
KG_ZRCKP = ZB.KG_ZRCKP                                             # zrckp's seed-0 R@5 (the MLP)
assert all(b in KG.FITS for sp in SPLITS for b in (ZSPK[sp], ZG6[sp])) and all(ZG6B[sp] in ZB.FITS for sp in SPLITS)

RULE = ("ADOPT_FOR_FULL_RUN when zg6b2's re-call against KG's zspk under KG's zspk seed null has no LOSS on any of its "
        "twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not "
        "BASE_READY, G6b's verdict is not NOT_ADOPTED with G6b2 next, a fit or compare is missing or not G6b2's, the "
        "fits are not on KG's device, or a re-call does not hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md "
        "section 4, on the T4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg6b2 fitted again on U2s's carried pools, then its full run on all six splits beside "
                              "zspk's, then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) "
                              "measured again; each its own declared file",
        "GAIN_AND_LOSS": G6.NEXT["NO_GAIN"],
        "NO_GAIN": G6.NEXT["NO_GAIN"],
        "INCOMPLETE": "what is missing is supplied (KG's base set, G6b's verdict, a fit, a compare or a re-call); "
                      "nothing else changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def row_degree(v, f, N, dt):
    """(N, FAMS): each row's in-degree per family over the edges (v their targets, f their families), zg6.g6_layers'
    deg before its clamp; 0 for a row no edge reaches. Counts are whole numbers, exact in any summation order."""
    one = torch.ones(v.numel(), dtype=dt, device=v.device)
    return torch.zeros(N * FAMS, dtype=dt, device=v.device).index_add_(0, v * FAMS + f, one).view(N, FAMS)


def g6b2_parts(m, H, xb, lk, nq, B, u01=None):
    """(score, s_e, s_t, g, sh, ld). s_e and s_t as zg6b.g6b_parts computes them, op for op; sh (B x FAMS) the family
    shares of each question's edges that reach the layers, ld (N x FAMS) log1p of each row's in-degree per family over
    them; g (N,) each row's gate sigmoid(a . sh[q] + c . ld + b); score = g s_e + (1 - g) s_t on each row."""
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    w = G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f = G6.edges_of(lk, nq, B, u01, H.device)
    he = G6.g6_layers(m, h0, xb, w, u, v, f, nq, B)
    ht = G6.g6_layers(m, h0, xb, w, *G6.edges_of(None, nq, B, None, H.device), nq, B)
    skip = Fn.linear(h0, m.g6_skip_w)
    s_e = (Fn.linear(he, m.g6_out_w) + skip + m.g6_out_b).squeeze(-1)
    s_t = (Fn.linear(ht, m.g6_out_w) + skip + m.g6_out_b).squeeze(-1)
    sh = ZB.shares(f, nq[v], B, H.dtype)
    ld = torch.log1p(row_degree(v, f, H.shape[0], H.dtype))
    zq = Fn.linear(sh, m.g6b_gate_w, m.g6b_gate_b).squeeze(-1)
    g = torch.sigmoid(zq[nq] + Fn.linear(ld, m.g6b2_row_w).squeeze(-1))
    return g * s_e + (1.0 - g) * s_t, s_e, s_t, g, sh, ld


class ZG6B2(ZB.ZG6B):
    """zg6b2: zg6b with its gate moved to the row: the question's edge-family shares and the row's own log1p in-degree
    per family (equal weight at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.g6b2_row_w = nn.Parameter(torch.zeros(1, FAMS))

    def g6b2_scores(self, feats, keep, nq, B):
        if KT_KEY not in feats:
            raise RuntimeError("zg6b2: no KT input in the batch (P1c3's carve adds it)")
        if LK_KEY not in feats:
            raise RuntimeError("zg6b2: no pool edges in the batch (zlink's carve adds them)")
        H = self.encode(feats, keep, nq, B)
        xb = torch.cat([feats[KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
        u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training else None
        return g6b2_parts(self, H, xb, feats[LK_KEY], nq, B, u01)

    def forward(self, feats, keep, nq, B, base_z):
        """The mixed score; base_z (rrf's z-score, the MLP's base) is never read."""
        return self.g6b2_scores(feats, keep, nq, B)[0]


ARMS = {ARM: (ZG6B2, Z3.KTLinkCarveBase)}
if S.ARMS.get(ARM) not in (None, ARMS[ARM]):
    raise SystemExit(f"zg6b2: lean_screen's arm {ARM} is already {S.ARMS[ARM]}")
S.ARMS.update(ARMS)
SETTINGS = {"network": "zg6b's (docs/G6B_GATED_BACKOFF.md, section 2), and through it zg6's", "zg6b": ZB.SETTINGS,
            "gate": "sigmoid(a . sh_q + c . log1p(deg_r) + b) per row; sh_q zg6b's family shares of the row's question, "
                    "deg_r the row's in-degree per family over the edges that reach the layers (after the family "
                    "dropout in training), 0 for a row no edge reaches; a, b and c 0 at the start",
            "score": "g_r s_e + (1 - g_r) s_t on each row", "gate_parameters": 2 * FAMS + 1,
            "parameters_at_hidden_128": N_PARAMS}


# ── the device, train and read (KG's, with G6b2's arm, names and stamp) ─────


def where_ok(env, what):
    if not KG.ok_where(env):
        raise SystemExit(f"zg6b2: G6b2 {what}s in a Kaggle kernel on a {KG.GPU} from the image {KG.IMAGE_TAG} (KG's "
                         f"device), not with {KG.device_of(env)} ({env.get('where')})")


def check_arm():
    if S.ARMS.get(ARM) != ARMS[ARM]:
        raise SystemExit(f"zg6b2: the arm {ARM} is {S.ARMS.get(ARM)}, not G6b2's {ARMS[ARM]}")


def code_shas():
    return {**ZB.code_shas(), "zg6b2_sha256": LC.sha_src(__file__)}


def head():
    return {"model": SETTINGS, "kt_root": Z.rel(Z.roots(BUILD)["kt"])}


def train(name, env=None):
    """One of G6b2's two fits on U2d's base build: zu2d.train's body as KG runs it, with G6b2's arm, name and stamp."""
    if name not in FITS:
        raise SystemExit(f"zg6b2: {name} is not one of G6b2's fits ({', '.join(FITS)})")
    split = FITS[name]
    check_arm()
    env = KG.device_env() if env is None else env
    where_ok(env, "train")
    argv = ["train", "--name", name, "--arm", ARM, "--device", "cuda", "--cache-root", KG.cache_arg()]
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(SEED)]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        rc = RM.train(argv, split)
        stamp_fit(name, env, g, kt, r)
    return rc


def stamp_fit(name, env, g, kt, r, out_root=None):
    split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    rec.update({**code_shas(), KEY: {
        "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split,
        "kg_bases": {"zspk": ZSPK[split], "zg6": ZG6[split]}, "g6b_beside": ZG6B[split], "device": KG.device_of(env),
        "env": env, "graph": Z.GRAPH[BUILD], "pools": g.get("construction"), "rule": Z.ZP.RULE[BUILD],
        "cache_root": Z.rel(r["cache"]), "gate": LC.sha_file(r["gate"]), "head": head(), "kt_records": kt,
        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
        "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A G6b2 fit's screen.json and its G6b2 record, checked against the fit's name."""
    split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("build"), k.get("arm"), k.get("seed"),
            k.get("split")) != (ARM, DECLARED, name, BUILD, ARM, SEED, split):
        raise SystemExit(f"zg6b2: {sj} is not G6b2's {ARM} fit of {split}")
    return rec, k


def read(name, env=None, rows_cap=0):
    """The fit read with p@swa on the six s1eval carves of U2d's base build, on the device it was fitted on."""
    if name not in FITS:
        raise SystemExit(f"zg6b2: {name} is not one of G6b2's fits")
    split = FITS[name]
    env = KG.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if KG.device_of(env) != k.get("device"):
        raise SystemExit(f"zg6b2: {name} was fitted on {k.get('device')}; this read runs on {KG.device_of(env)}")
    argv = ["--name", name, "--device", "cuda", "--cache-root", KG.cache_arg()]
    argv += ["--rows-cap", str(int(rows_cap))] if rows_cap else []
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        rc = RM.read(argv)
    rj = S.OUT / "fits" / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), KEY: {
            "declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split,
            "device": KG.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "rows_cap": int(rows_cap)}})
        LC.write_json(rj, rec)
    return rc


def stamped(p, name):
    try:
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    k = rec.get(KEY) or {}
    return k.get("declared_in") == DECLARED and k.get("fit") == name


# ── the gate, beside: its parameters, and its value on each s1eval question and reached row ──


def row_gate_values(a, b, c, links_root, ds, carve, keys=False):
    """The gate on one carve from zlink's arrays (each checked against its part's record), in float64, part by part:
    {"gq": each question's sigmoid(a . sh + b), "none": questions without edges, "rows": the carve's rows, "g": each
    reached row's sigmoid(a . sh_q + c . ld + b), "ld_sum": the reached rows' log1p in-degrees summed per family}; with
    keys, also "keys" (q * KEY_STRIDE + the row's index in its question, q counted over the carve) and "ld"."""
    a, c = np.asarray(a, np.float64), np.asarray(c, np.float64)
    out = {"gq": [], "none": 0, "rows": 0, "g": [], "ld_sum": np.zeros(FAMS), "keys": [], "ld": []}
    q0 = 0
    for p in ZB.link_parts(links_root, ds, carve):
        h = json.loads((p / "record.json").read_text(encoding="utf-8"))
        arr = {}
        for k in ("q_ne", "e_v", "e_fam"):
            fn = p / f"{k}.npy"
            if LC.sha_file(fn) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"zg6b2: {fn}: sha256 is not its record's")
            arr[k] = np.load(fn)
        qn = arr["q_ne"].astype(np.int64)
        q = np.repeat(np.arange(qn.size), qn)
        fam = arr["e_fam"].astype(np.int64)
        if arr["e_v"].size and not 0 <= int(arr["e_v"].min()) <= int(arr["e_v"].max()) < KEY_STRIDE:
            raise SystemExit(f"zg6b2: {p}: a row index outside [0, {KEY_STRIDE})")
        cnt = np.bincount(q * FAMS + fam, minlength=qn.size * FAMS).reshape(qn.size, FAMS)
        zq = cnt / np.maximum(cnt.sum(1, keepdims=True), 1) @ a + b
        uk, inv = np.unique(q * KEY_STRIDE + arr["e_v"].astype(np.int64), return_inverse=True)
        ld = np.log1p(np.bincount(inv.reshape(-1) * FAMS + fam, minlength=uk.size * FAMS).reshape(uk.size, FAMS))
        out["gq"].append(1.0 / (1.0 + np.exp(-zq)))
        out["g"].append(1.0 / (1.0 + np.exp(-(zq[uk // KEY_STRIDE] + ld @ c))))
        out["ld_sum"] += ld.sum(0)
        out["none"] += int((qn == 0).sum())
        out["rows"] += int(h["rows"])
        if keys:
            out["keys"].append(uk + q0 * KEY_STRIDE)
            out["ld"].append(ld)
        q0 += qn.size
    for k in ("gq", "g") + (("keys", "ld") if keys else ()):
        out[k] = np.concatenate(out[k])
    if not keys:
        del out["keys"], out["ld"]
    return out


def qstats(x):
    """Mean, 10th, 50th and 90th percentiles, rounded; None for no values."""
    if x.size == 0:
        return {"mean": None, "q10": None, "q50": None, "q90": None}
    q10, q50, q90 = (float(v) for v in np.quantile(x, [0.1, 0.5, 0.9]))
    return {"mean": round(float(x.mean()), 4), "q10": round(q10, 4), "q50": round(q50, 4), "q90": round(q90, 4)}


def gate_stats(st, links_root, carve="s1eval", datasets=None):
    """The gate of a p@swa state: a, b and c; its value on a row of a question of one family alone, unreached and
    reached by k edges of that family (k in DEGREES), and on a row of a question without edges; on each dataset's
    questions sigmoid(a . sh + b) and the share without edges; over the rows the edges reach, their share of the carve's
    rows, the row gate and their mean log1p in-degree per family."""
    a = [float(x) for x in st["g6b_gate_w"].double().view(-1)]
    b = float(st["g6b_gate_b"].double().view(-1)[0])
    c = [float(x) for x in st["g6b2_row_w"].double().view(-1)]
    out = {}
    for ds in datasets or LG.EVAL_ORDER:
        r = row_gate_values(a, b, c, links_root, ds, carve)
        nq_, nr = int(r["gq"].size), int(r["g"].size)
        out[ds] = {"questions": nq_, "no_edges": r["none"], "no_edges_share": round(r["none"] / max(nq_, 1), 4),
                   "question_gate": qstats(r["gq"]), "rows": r["rows"], "reached": nr,
                   "reached_share": round(nr / max(r["rows"], 1), 4), "row_gate": qstats(r["g"]),
                   "log1p_in_degree": [round(float(x) / max(nr, 1), 4) for x in r["ld_sum"]]}
    return {"a": [round(x, 6) for x in a], "b": round(b, 6), "c": [round(x, 6) for x in c],
            "no_edges": round(ZB.sigmoid(b), 4), "pure_family": [round(ZB.sigmoid(a[f] + b), 4) for f in range(FAMS)],
            "pure_family_by_degree": {str(k): [round(ZB.sigmoid(a[f] + c[f] * math.log1p(k) + b), 4)
                                               for f in range(FAMS)] for k in DEGREES},
            "carve": carve, "by_dataset": out}


def gate_beside(name, fits_root=None, links_root=None, carve="s1eval", datasets=None):
    """g6b2_gate.json in the fit's folder: the gate of its p@swa state (models.pt) on the six s1eval carves."""
    fd = Path(fits_root or S.OUT / "fits") / name
    blob = torch.load(fd / "models.pt", map_location="cpu", weights_only=False)
    rec = {"stage": STAGE, "declared_in": DECLARED, "fit": name, "candidate": "p@swa",
           **gate_stats(blob["states"]["p@swa"], links_root or Z.roots(BUILD)["links"], carve, datasets)}
    LC.write_json(fd / GATE_FILE, rec)
    return rec


def fit(name, rows_cap=0):
    """One fit in the kernel: a read G6b2 stamped, with its gate file, is kept; a fit trained and stamped but not read
    is only read; otherwise the fit's files are unlinked and it trains and reads. Then the gate file; g6b2_timing.json
    holds the seconds and peaks."""
    if name not in FITS:
        raise SystemExit(f"zg6b2: {name} is not one of G6b2's fits")
    fd = S.OUT / "fits" / name
    if stamped(fd / "read.json", name) and (fd / GATE_FILE).exists():
        log(f"zg6b2 fit {name}: read before, kept")
        return 0
    env = KG.device_env()
    where_ok(env, "train")
    t0 = time.time()
    trained = stamped(fd / "screen.json", name) and (fd / "train.json").exists() and (fd / "models.pt").exists()
    if not trained:
        KG.clear(fd, "*")
        KG.clear(fd / "reads", "*")
        rc = train(name, env)
        if rc:
            return rc
    t1 = time.time()
    rc = 0
    if not stamped(fd / "read.json", name):
        KG.clear(fd / "reads", "*")
        KG.clear(fd, "read.json")
        rc = read(name, env, rows_cap)
    t2 = time.time()
    if rc == 0:
        gate_beside(name)
    LC.write_json(fd / TIMING_FILE, {"fit": name, "trained_here": not trained, "train_seconds": round(t1 - t0, 1),
                                     "read_seconds": round(t2 - t1, 1), "gate_seconds": round(time.time() - t2, 1),
                                     "peak_gb": KG.peak_gb(), "rows_cap": int(rows_cap), "exit": rc,
                                     "declared_in": DECLARED})
    return rc


# ── the laptop: take and the compares ────────────────────────────────────────

NEVER = KG.NEVER
FITS_ROOT = SCREEN / "fits"                  # the laptop's fits, relative: its commands run from the repository's root


def fit_device(name, out_root=None):
    """A G6b2 fit's device: its screen.json's G6b2 record, and its read.json's on the same device."""
    out_root = Path(out_root or FITS_ROOT)
    _rec, k = fit_stamp(name, out_root)
    rj = out_root / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zg6b2: {rj} is not G6b2's read of {name} on the device it was fitted on")
    return k["device"]


def take(src, out_root=None):
    """Each G6b2 fit under src (a kernel's output: fits/<name>, or <name>) copied to outputs/screen/fits/<name>: its
    records, reads and gate file, never its weights. A fit must be stamped and read by G6b2; a file there already must
    be the same bytes. Other stages' fits in the same output are theirs to take."""
    src = Path(src)
    base = src / "fits" if (src / "fits").is_dir() else src
    dst_root = Path(out_root or FITS_ROOT)
    took, kept = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("scr-g6b2-")):
        if d.name not in FITS:
            raise SystemExit(f"zg6b2 take: {d.name} is not one of G6b2's fits")
        fit_device(d.name, base)
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            if f.name in NEVER or f.suffix in (".tmp", ".part"):
                continue
            t = dst_root / d.name / f.relative_to(d)
            if t.exists():
                if LC.sha_file(t) != LC.sha_file(f):
                    raise SystemExit(f"zg6b2 take: {t} exists with other bytes")
                kept.append(str(t))
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(str(t))
    log(f"zg6b2 take: {len(took)} files copied, {len(kept)} there before")
    return took


def base_device(name, out_root=None):
    """A base's device: G6b's zg6b by G6b's stamps, KG's fits by KG's."""
    return ZB.fit_device(name, out_root) if name in ZB.FITS else KG.fit_device(name, out_root)


def one_device(new, bases, out_root=None):
    devs = {new: fit_device(new, out_root), **{b: base_device(b, out_root) for b in bases}}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zg6b2: G6b2's fit and its bases ran on different devices or images: {devs}")
    return devs[new]


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of a G6b2 fit with KG's zspk (first: it decides), KG's zg6 and G6b's zg6b, one device, the
    paths relative to the repository, stamped with G6b2's record."""
    new, bases = COMPARES[cname]
    dev = one_device(new, bases, out_root)
    fits = Path(out_root or FITS_ROOT)
    out = Path(screen or SCREEN) / cname
    with Z.at_root():
        def r_(p):
            p = Path(p)
            return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "device": dev,
                "zg6b2_sha256": LC.sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(out_root=None, screen=None):
    done, ran = [], []
    for cname, (new, bases) in COMPARES.items():
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            k = json.loads(js.read_text(encoding="utf-8")).get(KEY) or {}
            if k.get("declared_in") == DECLARED and k.get("new") == new and k.get("bases") == list(bases):
                done.append(cname)
                continue
        compare_cmd(cname, out_root, screen)
        ran.append(cname)
    log(f"zg6b2 compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A G6b2 compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.exists():
        raise SystemExit(f"zg6b2: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zg6b2: {p} is not G6b2's compare {cname} of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zg6b2: {p} was made on {k.get('device')}, not on the fits' device {dev}")
    got = [SR.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if SR.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"zg6b2: {p} compares {rec.get('new')} with {rec.get('bases')}, not {new} with {bases}")
    split = FITS[new]
    if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
        raise SystemExit(f"zg6b2: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zg6b2: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zg6b2: {p} reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


# ── the verdict ──────────────────────────────────────────────────────────────


def g6b_filed(g6b_out=None):
    """Problems unless G6b's filed verdict is NOT_ADOPTED under G6b's declaration with G6b2 its next form."""
    p = Path(g6b_out or ZB.OUT) / f"{ZB.VERDICT}.json"
    if not p.exists():
        return [f"{p} missing (G6b's verdict)"]
    gv = json.loads(p.read_text(encoding="utf-8"))
    if (gv.get("stage"), gv.get("declared_in"), gv.get("verdict"), gv.get("next_form")) != \
            (ZB.STAGE, ZB.DECLARED, "NOT_ADOPTED", ZB.NEXT["GAIN_AND_LOSS"]):
        return [f"G6b's verdict is {gv.get('verdict')} ({gv.get('declared_in')}) with next form "
                f"{str(gv.get('next_form'))[:40]!r}..., not NOT_ADOPTED with G6b2 next"]
    return []


def restamp(od, which):
    """G6b2's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for name in (PAIR[which], RECALL[which]):
        js = (od / name).with_suffix(".json")
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "against": AGAINST[which],
                    "null_kind": NULL_KIND[which], "deciding": which == 0, "zg6b2_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)
        md = js.with_suffix(".md")
        if md.exists():
            head_ = (f"G6b2 ({DECLARED}): zg6b2 against {'G6b' if which == 2 else 'KG'}'s {AGAINST[which]} on the T4"
                     + (", deciding." if which == 0 else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(scr, od, dev, which):
    """screen_pair over G6b2's two compares against base `which` (0: KG's zspk, deciding; 1: KG's zg6 and 2: G6b's
    zg6b, beside), then screen_recall under KG's zspk null. Beside, the null's seed-0 R@5 is the base's read as filed
    (zg6's: scr-kg-zg6-vs-zspk's new column; zg6b's: scr-g6b-zg6b-vs-zspk's), which the compare's column must equal;
    the floors stay zspk's."""
    def load(fn):
        p = Path(fn)
        c = next(c for c in COMPARES if p.stem == c)
        _rec, rows = cmp_rows(p, c, dev)
        return FITS[COMPARES[c][0]], [rows[COMPARES[c][1][which]][ds] for ds in LG.EVAL_ORDER]

    saved = SP.load
    SP.load = load
    try:
        SP.pair([scr / f"{CMP[sp]}.json" for sp in SPLITS], od / PAIR[which])
    finally:
        SP.load = saved
    null, sources = KG.kg_null("zspk", scr, dev)
    for sp in SPLITS:
        if which == 1:
            _r, kz = KG.cmp_rows(scr / f"{KG_ZG6_CMP[sp]}.json", KG_ZG6_CMP[sp], dev)
            r5 = {ds: kz[ds]["new"][0] for ds in LG.EVAL_ORDER}
            sources[f"KG zg6 {sp}"] = str(scr / f"{KG_ZG6_CMP[sp]}.json")
        elif which == 2:
            _r, gz = ZB.cmp_rows(scr / f"{G6B_CMP[sp]}.json", G6B_CMP[sp], dev)
            r5 = {ds: gz[ZSPK[sp]][ds]["new"][0] for ds in LG.EVAL_ORDER}
            sources[f"G6b zg6b {sp}"] = str(scr / f"{G6B_CMP[sp]}.json")
        else:
            continue
        null[sp] = {ds: {**v, "seed0": r5[ds]} for ds, v in null[sp].items()}
    saved = SR.load_null
    SR.load_null = lambda _files: (null, sources)
    try:
        rec = SR.pair([], (od / PAIR[which]).with_suffix(".json"), od / RECALL[which])
    finally:
        SR.load_null = saved
    restamp(od, which)
    return rec


def tag(r):
    return f"{r['split']} {r['dataset']}"


def timing(name, fits_root):
    d = Path(fits_root) / name
    t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
    r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
    k = json.loads((d / TIMING_FILE).read_text(encoding="utf-8")) if (d / TIMING_FILE).exists() else {}
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb"),
            "gate_seconds": k.get("gate_seconds")}


fmt = KG.fmt


def calls_of(rc):
    c = {tag(r): r["call"] for r in rc["rows"]}
    return c, [k for k, x in c.items() if x == "GAIN"], [k for k, x in c.items() if x == "LOSS"]


def verdict(screen=None, out=None, out_root=None, kg_out=None, g6b_out=None):
    """G6b2's verdict by G6's rule from zg6b2's re-call against KG's zspk under KG's zspk null (sections 4 and 6).
    Beside, deciding nothing: zg6b2 against KG's zg6 and G6b's zg6b under zspk's floors, R@5 per read with zrckp's and
    the MLP share, the gate on each dataset, and seconds and memory beside G6b's zg6b and KG's zg6."""
    scr, od = Path(screen or SCREEN), Path(out or OUT)
    fits_root = Path(out_root or FITS_ROOT)
    od.mkdir(parents=True, exist_ok=True)
    kdev, problems = ZB.kg_device(kg_out)
    problems += g6b_filed(g6b_out)
    devs = {}
    for n in FITS:
        try:
            devs[n] = fit_device(n, out_root)
        except SystemExit as e:
            problems.append(str(e))
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"G6b2's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    if dev is not None and kdev is not None and dev != kdev:
        problems.append(f"G6b2's fits ran on {dev}, KG's base set on {kdev}")
    rows = {}
    for c in COMPARES:
        try:
            rows[c] = cmp_rows(scr / f"{c}.json", c, dev)[1]
        except SystemExit as e:
            problems.append(str(e))
    rc = {}
    if not problems:
        try:
            for which in AGAINST:
                rc[which] = recall(scr, od, dev, which)
        except SystemExit as e:
            problems.append(str(e))
    whole = sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER)
    for which, r in rc.items():
        if sorted((x["split"], x["dataset"]) for x in r["rows"]) != whole:
            problems.append(f"the re-call against {AGAINST[which]} holds {len(r['rows'])} reads, not the twelve of "
                            f"{', '.join(SPLITS)}")
    rrows = rc[0]["rows"] if 0 in rc and not problems else []
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"]
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": "zspk",
           "null_kind": NULL_KIND[0], "device": dev, "next_form": nxt, "gains": gains, "losses": losses,
           "problems": problems, "calls": {tag(r): r["call"] for r in rrows},
           "delta_r5": {tag(r): r["delta"] for r in rrows}, "zg6b2_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G6b2's screen verdict: zg6b2 **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6b2** (zg6's network over the edges and without them, mixed per row by a gate on the question's "
          f"edge-family shares and the row's own log1p in-degree per family) against KG's zspk on the T4, re-called "
          f"under KG's zspk seed null: {len(gains)} GAIN ({', '.join(gains) or 'none'}), {len(losses)} LOSS "
          f"({', '.join(losses) or 'none'})" + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 6 before any number): {nxt}.", f"- Device: {json.dumps(dev)}."]
    beside = {}
    if v != "INCOMPLETE":
        zrckp = {}
        for sp in SPLITS:
            _r, zr = KG.cmp_rows(scr / f"{KG_ZRCKP[sp]}.json", KG_ZRCKP[sp], dev)
            zrckp[sp] = {ds: zr[ds]["base"][0] for ds in LG.EVAL_ORDER}
        vz, gz, lz = calls_of(rc[1])
        vb, gb, lb = calls_of(rc[2])
        r5, share = {}, {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            rw = rows[CMP[sp]]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": r["base"], "zg6": rw[ZG6[sp]][ds]["base"][0],
                          "zg6b": rw[ZG6B[sp]][ds]["base"][0], "zg6b2": r["new"], "zrckp": zrckp[sp][ds],
                          "floor": r["floor"], "call": r["call"], "call_vs_zg6": vz[tag(r)],
                          "call_vs_zg6b": vb[tag(r)]}
            share[tag(r)] = G6.mlp_share(zrckp[sp][ds], r["new"], r["rrf"])
        gate = {}
        for n, sp in FITS.items():
            p = fits_root / n / GATE_FILE
            gate[sp] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        tm = {sp: {"zg6b2": timing(n, fits_root), "zg6b": ZB.timing(ZG6B[sp], fits_root),
                   "zg6": KG.timing(ZG6[sp], fits_root)} for n, sp in FITS.items()}
        beside = {"vs_zg6": {"gains": gz, "losses": lz}, "vs_zg6b": {"gains": gb, "losses": lb}, "r5": r5,
                  "mlp_share": share, "gate": gate, "timing": tm,
                  "pair": (od / PAIR[0]).with_suffix(".json").as_posix(),
                  "recall": (od / RECALL[0]).with_suffix(".json").as_posix(),
                  "recall_sha256": LC.sha_file((od / RECALL[0]).with_suffix(".json"))}
        md += ["", "Beside, deciding nothing.", ""]
        for w, (g_, l_) in (("KG's zg6", (gz, lz)), ("G6b's zg6b", (gb, lb))):
            md.append(f"- zg6b2 against {w} under zspk's floors: {len(g_)} GAIN ({', '.join(g_) or 'none'}), "
                      f"{len(l_)} LOSS ({', '.join(l_) or 'none'}).")
        md += ["", f"R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6b2 - rrf) where zg6b2's gain over rrf "
                   f"exceeds {SHARE_MIN}:", "",
               "| read | rrf | zspk | zg6 | zg6b | zg6b2 | zrckp (MLP) | MLP share | floor | call | vs zg6 | vs zg6b |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
        for k, t in r5.items():
            md.append(f"| {k} ({t['read']}) | {fmt(t['rrf'])} | {fmt(t['zspk'])} | {fmt(t['zg6'])} | {fmt(t['zg6b'])} | "
                      f"{fmt(t['zg6b2'])} | {fmt(t['zrckp'])} | {fmt(share[k])} | {fmt(t['floor'])} | {t['call']} | "
                      f"{t['call_vs_zg6']} | {t['call_vs_zg6b']} |")
        md += ["", "The gate of each fit's p@swa: g = sigmoid(a . sh_q + c . log1p(deg_r) + b) on each row. Its value on "
                   "a row of a question of one family alone, unreached and reached by k edges of that family, and on a "
                   "row of a question without edges; on each dataset's s1eval questions the question part sigmoid(a . "
                   "sh + b) (mean, 10th / 50th / 90th percentile), and over the rows the edges reach their share of the "
                   "carve's rows, the row gate (mean and percentiles) and their mean log1p in-degree per family:", ""]
        for sp in SPLITS:
            gt = gate[sp]
            if gt is None:
                md.append(f"- {sp}: no gate file.")
                continue
            md.append(f"- {sp}: a = {gt['a']}, b = {gt['b']}, c = {gt['c']}; one family alone, unreached "
                      f"{gt['pure_family']}; reached by " + "; ".join(f"{k}: {x}" for k, x in
                                                                    gt["pure_family_by_degree"].items())
                      + f"; no edges {gt['no_edges']}.")
            for ds in LG.EVAL_ORDER:
                x = gt["by_dataset"][ds]
                qg, rg = x["question_gate"], x["row_gate"]
                md.append(f"  - {ds}: question part {fmt(qg['mean'])} ({fmt(qg['q10'])} / {fmt(qg['q50'])} / "
                          f"{fmt(qg['q90'])}); reached rows {fmt(x['reached_share'])} of {x['rows']}, row gate "
                          f"{fmt(rg['mean'])} ({fmt(rg['q10'])} / {fmt(rg['q50'])} / {fmt(rg['q90'])}), log1p in-degree "
                          f"{x['log1p_in_degree']}; questions without edges {fmt(x['no_edges_share'])}.")
        md += ["", "Seconds and memory: zg6b2, and in brackets G6b's zg6b and KG's zg6 of the same split, all on the T4:",
               "", "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]

        def s0(x):
            return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
        for sp in SPLITS:
            a_, b_, c_ = tm[sp]["zg6b2"], tm[sp]["zg6b"], tm[sp]["zg6"]
            md.append(f"| {sp} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}, {s0(c_['train_seconds'])}) | "
                      f"{s0(a_['read_seconds'])} ({s0(b_['read_seconds'])}, {s0(c_['read_seconds'])}) | "
                      f"{fmt(a_['train_peak_gpu_gb'])} / {fmt(a_['read_peak_gpu_gb'])} ({fmt(b_['train_peak_gpu_gb'])} / "
                      f"{fmt(b_['read_peak_gpu_gb'])}, {fmt(c_['train_peak_gpu_gb'])} / {fmt(c_['read_peak_gpu_gb'])}) | "
                      f"{fmt(a_['peak_gb'])} ({fmt(b_['peak_gb'])}, {fmt(c_['peak_gb'])}) |")
    rec["beside"] = beside
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg6b2 verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = KG.TOY_ENV
stops = KG.stops


def fake_fit(td, name, dev=None, read=True, stamp=True, gate=True):
    """A G6b2 fit's screen.json and read.json (the stamps fit_device reads), with train.json, g6b2_timing.json, a read,
    models.pt and the gate file."""
    dev = dev or KG.device_of(TOY_ENV)
    split = FITS[name]
    d = Path(td) / "fits" / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split, "device": dev}
    LC.write_json(d / "screen.json", {"arm": ARM, "split": split, "seed": SEED, **({KEY: k} if stamp else {})})
    LC.write_json(d / "train.json", {"seconds": 160.0, "peak_gpu_gb": 3.6, "data_gb": 2.0})
    LC.write_json(d / TIMING_FILE, {"peak_gb": 9.5, "gate_seconds": 20.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if gate:
        qs_ = {"mean": 0.6, "q10": 0.5, "q50": 0.6, "q90": 0.7}
        LC.write_json(d / GATE_FILE, {
            "a": [0.5, -1.0, 0.25], "b": 0.1, "c": [-0.2, 0.1, 0.0], "no_edges": 0.525,
            "pure_family": [0.6457, 0.2891, 0.5866], "pure_family_by_degree": {str(k_): [0.6, 0.3, 0.6] for k_ in DEGREES},
            "carve": "s1eval", "by_dataset": {ds: {"questions": 10, "no_edges": 1, "no_edges_share": 0.1,
                                                   "question_gate": qs_, "rows": 100, "reached": 80,
                                                   "reached_share": 0.8, "row_gate": qs_,
                                                   "log1p_in_degree": [1.0, 0.5, 0.25]} for ds in LG.EVAL_ORDER}})
    if read:
        LC.write_json(d / "read.json", {"seconds": 13.0, "peak_gpu_gb": 4.6, KEY: dict(k)})
    return d


def fake_cmp(td, cname, deltas=None, zspk_r5=0.6, zg6_r5=None, zg6b_r5=None, dev=None, stamp=True, screen=None):
    """One of G6b2's compares in lean_screen's form (G6b2-stamped), against KG's zspk and zg6 and G6b's zg6b (zg6_r5 and
    zg6b_r5 {dataset: R@5}, 0.6 where absent); deltas {dataset: zg6b2 minus zspk}."""
    new, bases = COMPARES[cname]
    split = FITS[new]
    deltas, zg6_r5, zg6b_r5 = deltas or {}, zg6_r5 or {}, zg6b_r5 or {}
    bp = [str(Path(td) / "fits" / b) for b in bases]
    by = {}
    for b, r5 in zip(bp, ({ds: zspk_r5 for ds in LG.EVAL_ORDER}, {ds: zg6_r5.get(ds, 0.6) for ds in LG.EVAL_ORDER},
                          {ds: zg6b_r5.get(ds, 0.6) for ds in LG.EVAL_ORDER})):
        rows = []
        for ds in LG.EVAL_ORDER:
            nw = round(zspk_r5 + deltas.get(ds, 0.0), 4)
            d = round(nw - r5[ds], 4)
            lo, hi = round(d - 0.004, 4), round(d + 0.004, 4)
            rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot", "questions": 100,
                         "base": [r5[ds], 0.5, 0.4], "new": [nw, 0.5, 0.4], "rrf": [0.4, 0.3, 0.2], "delta": [d, 0, 0],
                         "lo": [lo, 0, 0], "hi": [hi, 0, 0], "call": S.call(d, lo, hi)})
        by[b] = {"rows": rows, "verdict": S.verdict([r["call"] for r in rows]), "decides": b == bp[0]}
    rec = {"new": str(Path(td) / "fits" / new), "bases": bp, "candidate": "p@swa", "carve": "s1eval",
           "trained_on": sorted(LG.FITS[split]), "by_base": by, "verdict": by[bp[0]]["verdict"]}
    if stamp:
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases),
                    "device": dev or KG.device_of(TOY_ENV)}
    p = Path(screen or td) / f"{cname}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rec), encoding="utf-8")
    return p


def fake_set(td, deltas=None, g6b=None, kg_g6=None, big=None, kg="BASE_READY", g6b_v=("NOT_ADOPTED", None), dev=None,
             zg6_r5=None, zg6b_r5=None):
    """KG's fake base set and verdict and G6b's fits and compares (zg6b.fake_set; g6b {(split, dataset): zg6b minus
    zspk}), G6b's verdict (g6b_v: verdict and next form, G6b2 when None; None: no file), and G6b2's two fits and
    compares; deltas {(split, dataset): zg6b2 minus zspk}; kg_g6 {(split, dataset): zg6 minus zspk}; big {(split,
    dataset): (D1, D2)} for zspk's null."""
    td = Path(td)
    deltas, g6b, kg_g6 = deltas or {}, g6b or {}, kg_g6 or {}
    ZB.fake_set(td, g6b, kg_g6=kg_g6, big=big, kg=kg)
    if g6b_v:
        LC.write_json(td / "g6b" / f"{ZB.VERDICT}.json", {"stage": ZB.STAGE, "declared_in": ZB.DECLARED,
                                                          "verdict": g6b_v[0],
                                                          "next_form": g6b_v[1] or ZB.NEXT["GAIN_AND_LOSS"]})
    for n, sp in FITS.items():
        fake_fit(td, n, dev=dev)
        z6 = (zg6_r5 or {}).get(sp) or {ds: round(0.6 + kg_g6.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        zb = (zg6b_r5 or {}).get(sp) or {ds: round(0.6 + g6b.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        fake_cmp(td, CMP[sp], {ds: x for (s, ds), x in deltas.items() if s == sp}, zg6_r5=z6, zg6b_r5=zb, dev=dev)


def real_state():
    """What G6b2 has under the real folders: its fits, its compares and outputs/g6b2."""
    return (sorted(p.name for p in (S.OUT / "fits").glob("scr-g6b2-*")),
            sorted(p.name for p in (ROOT / SCREEN).glob("scr-g6b2-*")), (ROOT / OUT).exists() and
            sorted(p.name for p in (ROOT / OUT).iterdir()))


def brute_degree(lk, N, keep_e=None):
    """(N, FAMS) in-degrees per family by python loops over the edges (keep_e: the edges kept)."""
    cnt = [[0] * FAMS for _ in range(N)]
    ks = keep_e.tolist() if keep_e is not None else [True] * lk["v"].numel()
    for v_, f_, k_ in zip(lk["v"].tolist(), lk["f"].tolist(), ks):
        if k_:
            cnt[v_][f_] += 1
    return torch.tensor(cnt, dtype=torch.float32)


def by_hand(m, H, xb, lk, nq, B, keep_e=None):
    """zg6b2's score with python loops: zg6.by_hand's two passes, the shares and in-degrees counted edge by edge."""
    dg = brute_degree(lk, nq.numel(), keep_e)
    bs = ZB.brute_shares(lk, nq, B, keep_e)
    g = torch.sigmoid((bs @ m.g6b_gate_w.view(-1) + m.g6b_gate_b)[nq] + torch.log1p(dg) @ m.g6b2_row_w.view(-1))
    return g * G6.by_hand(m, H, xb, lk, nq, B, False, keep_e) + (1 - g) * G6.by_hand(m, H, xb, lk, nq, B, True), g, dg


def selftest():
    import tempfile
    t0 = time.time()
    before = real_state()
    LG.bind_device_ops()
    # 1. the arm on P1c3's link carve, with zg6b's and zg6's classes below it; earlier arms unchanged; names and bases
    assert S.ARMS[ARM] == (ZG6B2, Z3.KTLinkCarveBase) and ZG6B2.__mro__[1:4] == (ZB.ZG6B, G6.G6Free, G6.ZM.ZRM)
    assert S.ARMS["zg6b"] == ZB.ARMS["zg6b"] and S.ARMS["zg6"] == G6.ARMS["zg6"] and S.ARMS["zspk"] == Z.ARMS["zspk"]
    assert N_PARAMS == 121809 and sorted(FITS) == ["scr-g6b2-zg6b2", "scr-g6b2-zg6b2-hp"] and not ZG6B2.TWIN
    assert COMPARES["scr-g6b2-zg6b2-vs-zspk-hp"] == ("scr-g6b2-zg6b2-hp", ["scr-kg-base-zspk-hp", "scr-kg-base-zg6-hp",
                                                                          "scr-g6b-zg6b-hp"])
    assert G6B_CMP["L-musique"] == "scr-g6b-zg6b-vs-zspk" and NEXT["GAIN_AND_LOSS"] == NEXT["NO_GAIN"] == ZB.NEXT["NO_GAIN"]
    assert not any(n.startswith("scr-g6b-") for n in FITS) and all(n.startswith("scr-g6b-") for n in ZG6B.values())
    # 2. where: a Kaggle T4 on KG's image only; train and read refuse anything else before touching data
    for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"},
                {"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}):
        stops(train, "scr-g6b2-zg6b2", {**TOY_ENV, **bad}, has="G6b2 trains in a Kaggle kernel")
        stops(read, "scr-g6b2-zg6b2-hp", {**TOY_ENV, **bad}, has="G6b2 reads in a Kaggle kernel")
    stops(train, "scr-g6b-zg6b", TOY_ENV, has="not one of G6b2's fits")
    stops(read, "scr-g6b2-zg6b", TOY_ENV, has="not one of G6b2's fits")
    tmp = Path(tempfile.mkdtemp(prefix="zg6b2_"))
    try:
        rng = np.random.default_rng(31)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        plain = ZL.link_carve(RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
        cls = Z3.kt_carve(plain, tmp / "kt")
        c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(c.rows)[::-1][:7], np.arange(c.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _gold = c.batch(qs, blocks)
        N = nq.numel()
        lk = feats[LK_KEY]
        assert lk["u"].numel() > 0 and KT_KEY in feats and len(set(lk["f"].tolist())) >= 2
        keep = torch.ones(B, len(blocks))
        # 3. from seed 0: zg6b2's construction moves the global generator as zg6's and zg6b's do; its state is zg6b's
        #    plus c (zero), every shared tensor equal; the family dropout's generator starts where zg6b's does
        nets, after = {}, {}
        for k, cl in (("zg6", G6.ZG6), ("zg6t", G6.ZG6T), ("zg6b", ZB.ZG6B), (ARM, ZG6B2)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after["zg6"]) and torch.equal(after[ARM], after["zg6b"])
        m, zb, z6, zt = nets[ARM], nets["zg6b"], nets["zg6"], nets["zg6t"]
        sm, sb = m.state_dict(), zb.state_dict()
        assert sorted(sm) == sorted(set(sb) | set(G6B2_NAMES)) and all(torch.equal(sm[k], sb[k]) for k in sb)
        assert all(float(sm[k].abs().sum()) == 0 for k in GATE_NAMES)
        assert sum(int(sm[k].numel()) for k in G6.G6_NAMES + GATE_NAMES) == G6.n_params(16) + 2 * FAMS + 1
        assert torch.equal(m.g6_gen.get_state(), zb.g6_gen.get_state())
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for n in G6.G6_NAMES:
                getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
            shared = {k: x for k, x in m.state_dict().items() if k not in G6B2_NAMES}
            zb.load_state_dict(shared)
            z6.load_state_dict({k: x for k, x in shared.items() if k not in ZB.G6B_NAMES})
            zt.load_state_dict({k: x for k, x in shared.items() if k not in ZB.G6B_NAMES})
            for x in nets.values():
                x.eval()
            # 4. at the start (a, b, c 0): s_e, s_t and the shares are zg6b's bit for bit (zg6's score and its twin's);
            #    the gate is 0.5 on every row and the score zg6b's, bit for bit; base_z is never read
            H = m.encode(feats, keep, nq, B)
            xb = torch.cat([feats[KT_KEY], m.chain(feats, keep, nq)], 1)
            y, se, st, g, sh, ld = m.g6b2_scores(feats, keep, nq, B)
            yb, seb, stb, _gb, shb = zb.g6b_scores(feats, keep, nq, B)
            assert torch.equal(se, seb) and torch.equal(st, stb) and torch.equal(sh, shb)
            assert torch.equal(se, z6(feats, keep, nq, B, bz)) and torch.equal(st, zt(feats, keep, nq, B, bz))
            assert torch.equal(g, torch.full((N,), 0.5)) and torch.equal(y, yb) and torch.equal(y, zb(feats, keep, nq, B, bz))
            assert torch.equal(m(feats, keep, nq, B, torch.full_like(bz, float("nan"))), y)
            assert not torch.allclose(se, st)
            # 5. a random a and b with c 0: zg6b's gate on every row and zg6b's score (to an ulp: the sigmoid runs on
            #    rows, not questions)
            gw_, gb_ = torch.randn(1, FAMS, generator=gen), torch.randn(1, generator=gen)
            for x in (m, zb):
                x.g6b_gate_w.copy_(gw_)
                x.g6b_gate_b.copy_(gb_)
            y, se, st, g, sh, ld = m.g6b2_scores(feats, keep, nq, B)
            yb, _seb, _stb, gb2, _shb = zb.g6b_scores(feats, keep, nq, B)
            assert torch.allclose(g, gb2[nq], atol=1e-6, rtol=0) and torch.allclose(y, yb, atol=1e-6, rtol=1e-6)
            # 6. a random c: the in-degrees are a count by loops, bit for bit, and the score the model by hand
            m.g6b2_row_w.copy_(torch.randn(1, FAMS, generator=gen))
            y, se, st, g, sh, ld = m.g6b2_scores(feats, keep, nq, B)
            want, gw, dg = by_hand(m, H, xb, lk, nq, B)
            assert torch.equal(row_degree(lk["v"], lk["f"], N, torch.float32), dg) and torch.equal(ld, torch.log1p(dg))
            assert torch.allclose(g, gw, atol=1e-6) and torch.allclose(y, want, atol=1e-5, rtol=1e-5), \
                float((y - want).abs().max())
            reached = dg.sum(1) > 0
            assert 0 < int(reached.sum()) < N and float(g[reached].std()) > 0
            # 7. a row no edge reaches in a question with edges: its two scores are equal bit for bit, its score the
            #    twin's, its gate the question part alone; such rows give the gate no gradient; a batch without the
            #    edges or the KT input stops
            r0 = int(lk["v"][0])
            q0 = int(nq[r0])
            f3 = dict(feats)
            f3[LK_KEY] = {x: t[lk["v"] != r0] for x, t in lk.items()}
            assert bool((nq[f3[LK_KEY]["v"]] == q0).any())
            y3, se3, st3, g3, sh3, ld3 = m.g6b2_scores(f3, keep, nq, B)
            assert torch.equal(se3[r0], st3[r0]) and float(ld3[r0].abs().sum()) == 0
            assert abs(float(y3[r0]) - float(st3[r0])) <= 1e-6 * max(1.0, abs(float(st3[r0])))
            zq0 = float(sh3[q0] @ m.g6b_gate_w.view(-1) + m.g6b_gate_b)
            assert abs(float(g3[r0]) - ZB.sigmoid(zq0)) < 1e-6
            un = brute_degree(f3[LK_KEY], N).sum(1) == 0
            assert torch.equal(se3[un], st3[un])
            for bad in ({k: x for k, x in feats.items() if k != LK_KEY}, {k: x for k, x in feats.items() if k != KT_KEY}):
                try:
                    m(bad, keep, nq, B, bz)
                except RuntimeError:
                    continue
                raise AssertionError("zg6b2 ran without its edges or its KT input")
        wz = torch.randn(N, generator=torch.Generator().manual_seed(13)) * un.float()
        gz = G6.grads(m, f3, keep, nq, B, bz, wz)
        assert all(float(gz[k].abs().sum()) == 0 for k in GATE_NAMES), {k: float(gz[k].abs().sum()) for k in GATE_NAMES}
        assert G6.nonzero(gz["g6_out_w"])
        with torch.no_grad():
            # 8. the family dropout: no draw at read; in training one B x FAMS draw from the module's generator, the
            #    shares and the in-degrees over the edges zg2.family_keep keeps, the score the model by hand over them;
            #    the global generator moves as zg6b's
            g0 = m.g6_gen.get_state()
            m(feats, keep, nq, B, bz)
            assert torch.equal(m.g6_gen.get_state(), g0)
            m.train()
            m.drop.p = 0.0
            for _ in range(50):
                g0 = m.g6_gen.get_state()
                u01 = torch.rand(B, FAMS, generator=m.g6_gen)
                kd = G2.family_keep(lk["f"], nq[lk["v"]], B, u01)
                if 0 < int(kd.sum()) < kd.numel():
                    break
            assert 0 < int(kd.sum()) < kd.numel()
            g1 = m.g6_gen.get_state()
            m.g6_gen.set_state(g0)
            y, se, st, g, sh, ld = m.g6b2_scores(feats, keep, nq, B)
            assert torch.equal(m.g6_gen.get_state(), g1)
            want, gw, dg = by_hand(m, H, xb, lk, nq, B, kd)
            assert torch.equal(ld, torch.log1p(dg)) and torch.allclose(sh, ZB.brute_shares(lk, nq, B, kd), atol=1e-6)
            assert torch.allclose(y, want, atol=1e-5, rtol=1e-5), float((y - want).abs().max())
            m.drop.p = 0.1
            zb.load_state_dict({k: x for k, x in m.state_dict().items() if k not in G6B2_NAMES})
            zb.g6_gen.set_state(m.g6_gen.get_state())
            zb.train()
            torch.manual_seed(9)
            m(feats, keep, nq, B, bz)
            ra = torch.rand(4)
            torch.manual_seed(9)
            zb(feats, keep, nq, B, bz)
            assert torch.equal(ra, torch.rand(4)) and torch.equal(m.g6_gen.get_state(), zb.g6_gen.get_state())
            m.eval()
            zb.eval()
        # 9. gradients reach every parameter: the encoder, the chain match, each layer, the context, the readout and
        #    the gate's a, b and c
        with torch.no_grad():
            m.cm_b.copy_(torch.randn(m.cm_b.shape, generator=gen) * 0.1)
            m.cm_d.copy_(torch.randn(m.cm_d.shape, generator=gen) * 0.1)
        ga = G6.grads(m, feats, keep, nq, B, bz, torch.randn(N, generator=torch.Generator().manual_seed(11)))
        assert not [k for k, x in ga.items() if not G6.nonzero(x)], [k for k, x in ga.items() if not G6.nonzero(x)]
        assert all(G6.nonzero(ga[k]) for k in GATE_NAMES) and G6.nonzero(ga["l1.weight"])
        # 10. toy fits through lean_gpu's loop: a repeat is bit for bit; the encoder, the stack, the readout and the
        #     gate's a, b and c move
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zg6b2")
            f2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zg6b2 repeat")
        assert all(not LG.same_state(a, b) for a, b in zip(f1["states"] + [f1["swa"]], f2["states"] + [f2["swa"]]))
        torch.manual_seed(0)
        s0 = ZG6B2(blocks, f1["widths"], 16, dropout=0.1, seed=0).state_dict()
        assert sorted(s0) == sorted(f1["swa"])
        must = ["l1.weight", "l2.weight", "U", "g6_in_w", "g6_out_w", "g6_skip_w", *GATE_NAMES] + \
            [lay[i] for lay in G6.G6_LAYER for i in (0, 1, 2, 3, 4, 5, 6)]
        still = [k for k in must if torch.equal(s0[k], f1["swa"][k])]
        assert not still, still
        # 11. the gate beside: from the toy's link arrays, each question's part and each reached row's gate and
        #     in-degrees are the model's own (eval) and a brute count's; the stats are theirs; an array with other bytes
        #     stops
        m.eval()
        st_ = {k: x.clone() for k, x in m.state_dict().items()}
        a_ = [float(x) for x in st_["g6b_gate_w"].view(-1)]
        b_ = float(st_["g6b_gate_b"])
        c_ = [float(x) for x in st_["g6b2_row_w"].view(-1)]
        allq = np.arange(c.rows, dtype=np.int64)
        fa, nqa, _bza, _g = c.batch(allq, blocks)
        with torch.no_grad():
            _ya, _sea, _sta, gma, sha, lda = m.g6b2_scores(fa, torch.ones(allq.size, len(blocks)), nqa, allq.size)
        rch = torch.nonzero(lda.sum(1) > 0).view(-1)
        cnt = torch.bincount(nqa, minlength=allq.size)
        start = torch.cumsum(cnt, 0) - cnt
        key_m = (nqa[rch] * KEY_STRIDE + (rch - start[nqa[rch]])).numpy()
        rv = row_gate_values(a_, b_, c_, tmp / "lk", "metaqa", "toy", keys=True)
        gq_m = torch.sigmoid(sha.double() @ torch.tensor(a_, dtype=torch.float64) + b_).numpy()
        assert np.array_equal(rv["keys"], key_m) and rv["gq"].size == c.rows and rv["rows"] == nqa.numel()
        assert np.allclose(rv["gq"], gq_m, atol=1e-6) and np.allclose(rv["g"], gma[rch].double().numpy(), atol=1e-6)
        assert np.allclose(rv["ld"], lda[rch].double().numpy(), atol=1e-6)
        brute_k, brute_g, brute_ld, nn0, qg = [], [], [], 0, 0
        for p in ZB.link_parts(tmp / "lk", "metaqa", "toy"):
            qn, ev, ef = (np.load(p / f"{k}.npy").tolist() for k in ("q_ne", "e_v", "e_fam"))
            e = 0
            for k in qn:
                fc, deg = [0] * FAMS, {}
                for v_, f_ in zip(ev[e:e + k], ef[e:e + k]):
                    fc[f_] += 1
                    deg.setdefault(v_, [0] * FAMS)[f_] += 1
                e += k
                nn0 += k == 0
                zq = sum(a_[f_] * fc[f_] / max(k, 1) for f_ in range(FAMS)) + b_
                for v_ in sorted(deg):
                    lg_ = [math.log1p(x) for x in deg[v_]]
                    brute_k.append(qg * KEY_STRIDE + v_)
                    brute_ld.append(lg_)
                    brute_g.append(ZB.sigmoid(zq + sum(c_[f_] * lg_[f_] for f_ in range(FAMS))))
                qg += 1
            assert e == len(ef)
        assert np.array_equal(rv["keys"], np.asarray(brute_k)) and np.allclose(rv["g"], brute_g, atol=1e-12)
        assert np.allclose(rv["ld"], brute_ld, atol=1e-12) and rv["none"] == nn0
        gs = gate_stats(st_, tmp / "lk", "toy", ["metaqa"])
        x = gs["by_dataset"]["metaqa"]
        assert x["questions"] == c.rows and x["no_edges"] == nn0 and x["reached"] == len(brute_g)
        assert x["row_gate"] == qstats(np.asarray(brute_g)) and x["question_gate"] == qstats(rv["gq"])
        assert x["reached_share"] == round(len(brute_g) / nqa.numel(), 4)
        assert x["log1p_in_degree"] == [round(float(v) / len(brute_ld), 4) for v in np.asarray(brute_ld).sum(0)]
        assert gs["c"] == [round(v, 6) for v in c_] and gs["pure_family_by_degree"]["10"] == \
            [round(ZB.sigmoid(a_[f] + c_[f] * math.log1p(10) + b_), 4) for f in range(FAMS)]
        assert gs["no_edges"] == round(ZB.sigmoid(b_), 4) and gs["pure_family"] == [round(ZB.sigmoid(a_[f] + b_), 4)
                                                                                    for f in range(FAMS)]
        fd = tmp / "gfits" / "scr-g6b2-zg6b2"
        fd.mkdir(parents=True)
        torch.save({"states": {"p@swa": st_}, "candidates": ["p@swa"]}, fd / "models.pt")
        gb = gate_beside("scr-g6b2-zg6b2", tmp / "gfits", tmp / "lk", "toy", ["metaqa"])
        assert json.loads((fd / GATE_FILE).read_text(encoding="utf-8")) == gb and gb["by_dataset"]["metaqa"] == x
        assert gb["fit"] == "scr-g6b2-zg6b2" and gb["candidate"] == "p@swa" and gb["carve"] == "toy"
        arr = ZB.link_parts(tmp / "lk", "metaqa", "toy")[0] / "e_v.npy"
        keep_b = arr.read_bytes()
        arr.write_bytes(keep_b[:-1] + bytes([keep_b[-1] ^ 1]))
        stops(row_gate_values, [0.0] * FAMS, 0.0, [0.0] * FAMS, tmp / "lk", "metaqa", "toy",
              has="sha256 is not its record's")
        arr.write_bytes(keep_b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "fits"
        # 12. stamps and devices: a fit not G6b2's, a read on another device; take copies records, reads and the gate
        #     file, never models.pt, and leaves KG's and G6b's fits to them
        fake_fit(td, "scr-g6b2-zg6b2")
        assert fit_device("scr-g6b2-zg6b2", fits) == KG.device_of(TOY_ENV)
        fake_fit(td, "scr-g6b2-zg6b2-hp", stamp=False)
        stops(fit_device, "scr-g6b2-zg6b2-hp", fits, has="not G6b2's zg6b2 fit")
        d = fake_fit(td, "scr-g6b2-zg6b2-hp")
        LC.write_json(d / "read.json", {KEY: {"declared_in": DECLARED, "fit": "scr-g6b2-zg6b2-hp",
                                              "device": {**KG.device_of(TOY_ENV), "gpu": "Tesla P100"}}})
        stops(fit_device, "scr-g6b2-zg6b2-hp", fits, has="on the device it was fitted on")
        src = td / "out"
        fake_fit(src, "scr-g6b2-zg6b2")
        KG.fake_fit(src, "scr-kg-base-zg6")
        ZB.fake_fit(src, "scr-g6b-zg6b")
        took = take(src, td / "repo_fits")
        assert sorted(Path(t).name for t in took) == sorted(["g6b2_gate.json", "g6b2_timing.json", "metaqa__s1eval.npz",
                                                             "read.json", "screen.json", "train.json"]), took
        assert not (td / "repo_fits" / "scr-g6b2-zg6b2" / "models.pt").exists()
        assert not (td / "repo_fits" / "scr-kg-base-zg6").exists() and not (td / "repo_fits" / "scr-g6b-zg6b").exists()
        assert take(src, td / "repo_fits") == []
        (td / "repo_fits" / "scr-g6b2-zg6b2" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, src, td / "repo_fits", has="other bytes")
        fake_fit(src, "scr-g6b2-zg6b2-hp", read=False)
        stops(take, src, td / "d2", has="not G6b2's read")
        shutil.rmtree(src / "fits" / "scr-g6b2-zg6b2-hp")
        (src / "fits" / "scr-g6b2-zg6b").mkdir()
        stops(take, src, td / "d3", has="not one of G6b2's fits")
        # 13. one device across G6b2's fit, KG's two bases and G6b's zg6b
        three = ["scr-kg-base-zspk", "scr-kg-base-zg6", "scr-g6b-zg6b"]
        KG.fake_fit(td, "scr-kg-base-zspk")
        KG.fake_fit(td, "scr-kg-base-zg6")
        ZB.fake_fit(td, "scr-g6b-zg6b")
        assert one_device("scr-g6b2-zg6b2", three, fits) == KG.device_of(TOY_ENV)
        ZB.fake_fit(td, "scr-g6b-zg6b", dev={**KG.device_of(TOY_ENV), "cudnn": 1})
        stops(one_device, "scr-g6b2-zg6b2", three, fits, has="different devices")
    # 14. the verdict: a GAIN and no LOSS adopts for the full run; every NOT_ADOPTED gives G6r (a GAIN and a LOSS, no
    #     GAIN, a GAIN under a floor the null raises, a LOSS alone); INCOMPLETE without KG's BASE_READY, without G6b's
    #     verdict or with another, without a compare, with the fits on another device than KG's, or with a zg6 or zg6b
    #     column that is not the read as filed; beside: the zg6 and zg6b re-calls, R@5, the MLP share, gate and timing
    gm_, lw = ("L-musique", "metaqa"), ("L-hotpotqa", "webqsp")
    g6b = {gm_: 0.015, lw: -0.02, ("L-hotpotqa", "musique"): 0.01}
    cases = (({gm_: 0.02}, None, "ADOPT_FOR_FULL_RUN", NEXT["ADOPT_FOR_FULL_RUN"]),
             ({gm_: 0.02, lw: -0.03}, None, "NOT_ADOPTED", NEXT["GAIN_AND_LOSS"]),
             ({}, None, "NOT_ADOPTED", NEXT["NO_GAIN"]),
             ({("L-musique", "musique"): 0.02}, {("L-musique", "musique"): (0.02, -0.01)}, "NOT_ADOPTED",
              NEXT["NO_GAIN"]),
             ({lw: -0.03}, None, "NOT_ADOPTED", NEXT["NO_GAIN"]))
    for i, (deltas, big, want, nxt) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_set(td, deltas, g6b, kg_g6={("L-hotpotqa", "2wiki"): 0.01}, big=big)
            v = verdict(td, td / "g6b2", td / "fits", td / "kg", td / "g6b")
            assert v["verdict"] == want and v["next_form"] == nxt, (i, v["verdict"], v["problems"])
            b = v["beside"]
            assert b["r5"]["L-musique metaqa"]["zrckp"] == 0.59 and b["r5"]["L-hotpotqa 2wiki"]["zg6"] == 0.61
            assert b["r5"]["L-musique metaqa"]["zg6b"] == 0.615 and b["r5"]["L-hotpotqa webqsp"]["zg6b"] == 0.58
            assert b["r5"]["L-musique squad"]["zg6b2"] == 0.6 and b["mlp_share"]["L-musique squad"] == round(0.19 / 0.2, 4)
            assert b["gate"]["L-musique"]["c"] == [-0.2, 0.1, 0.0] and b["timing"]["L-hotpotqa"]["zg6b2"]["train_seconds"] == 160
            assert b["timing"]["L-hotpotqa"]["zg6b"]["train_seconds"] == 150.0
            assert b["timing"]["L-hotpotqa"]["zg6"]["train_seconds"] == 100.0
            for which in (1, 2):
                rz = json.loads((td / "g6b2" / f"{RECALL[which]}.json").read_text(encoding="utf-8"))
                assert rz["stage"] == STAGE and rz["against"] == AGAINST[which] and not rz["deciding"]
                assert len(rz["rows"]) == 12 and rz["null_kind"] == NULL_KIND[which]
            rd = json.loads((td / "g6b2" / f"{RECALL[0]}.json").read_text(encoding="utf-8"))
            assert rd["deciding"] and rd["against"] == "zspk" and len(rd["rows"]) == 12
            if i == 0:
                assert v["gains"] == ["L-musique metaqa"] and b["vs_zg6"]["gains"] == ["L-musique metaqa"]
                assert b["vs_zg6b"]["gains"] == ["L-hotpotqa webqsp"] and b["vs_zg6b"]["losses"] == ["L-hotpotqa musique"]
            if i == 1:
                assert b["vs_zg6b"]["losses"] == ["L-hotpotqa musique", "L-hotpotqa webqsp"], b["vs_zg6b"]
            if i == 3:
                assert rd["changed"] == ["L-musique musique GAIN -> WITHIN"], rd["changed"]
            assert (td / "g6b2" / f"{VERDICT}.md").read_text(encoding="utf-8").startswith("# G6b2's screen verdict")
    for case in ("no_kg", "kg_incomplete", "no_g6b", "g6b_adopted", "g6b_other_next", "no_compare", "other_device",
                 "zg6_column", "zg6b_column"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            kw = {"no_kg": {"kg": None}, "kg_incomplete": {"kg": "INCOMPLETE"}, "no_g6b": {"g6b_v": None},
                  "g6b_adopted": {"g6b_v": ("ADOPT_FOR_FULL_RUN", ZB.NEXT["ADOPT_FOR_FULL_RUN"])},
                  "g6b_other_next": {"g6b_v": ("NOT_ADOPTED", ZB.NEXT["NO_GAIN"])},
                  "other_device": {"dev": {**KG.device_of(TOY_ENV), "cudnn": 1}},
                  "zg6_column": {"zg6_r5": {"L-musique": {ds: 0.6 for ds in LG.EVAL_ORDER} | {"squad": 0.61}}},
                  "zg6b_column": {"zg6b_r5": {"L-hotpotqa": {ds: 0.6 for ds in LG.EVAL_ORDER}}}}.get(case, {})
            fake_set(td, {gm_: 0.02}, g6b, **kw)
            if case == "no_compare":
                (td / f"{CMP['L-hotpotqa']}.json").unlink()
            v = verdict(td, td / "g6b2", td / "fits", td / "kg", td / "g6b")
            assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"] and v["beside"] == {}, \
                (case, v["verdict"], v["problems"])
            has = {"no_kg": "KG's base verdict", "kg_incomplete": "not BASE_READY", "no_g6b": "G6b's verdict",
                   "g6b_adopted": "not NOT_ADOPTED with G6b2 next", "g6b_other_next": "not NOT_ADOPTED with G6b2 next",
                   "no_compare": "missing", "other_device": "KG's base set on",
                   "zg6_column": "compared with seed-0 R@5", "zg6b_column": "compared with seed-0 R@5"}[case]
            assert any(has in p for p in v["problems"]), (case, v["problems"])
    # 15. nothing written under the real folders
    assert real_state() == before, (real_state(), before)
    log(f"zg6b2 selftest: the arm on P1c3's link carve with zg6b's and zg6's classes below it, earlier arms unchanged; "
        f"where-checks; from seed 0 zg6b's draws and state plus a zero c ({N_PARAMS} parameters at hidden 128); at the "
        f"start zg6b's score bit for bit; zg6b's gate with c 0; the in-degrees by loops and the score by hand; a row no "
        f"edge reaches scores as the twin and gives the gate no gradient; the family dropout's draws and the global "
        f"generator as zg6b's; gradients reach every parameter; toy fits repeat bit for bit and move a, b and c; the "
        f"gate beside equals the model's own and a brute count, row by row; stamps, take and one device across the "
        f"three bases; the verdict's outcomes, next forms, refusals and beside rows; nothing written under G6b2's "
        f"folders ({time.time() - t0:.1f}s): ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("cmd", nargs="?", choices=("fit", "take", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--rows-cap", type=int, default=0)
    ap.add_argument("--from", dest="src")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        if not a.name:
            ap.error("fit takes --name")
        return fit(a.name, a.rows_cap)
    with Z.at_root():
        if a.cmd == "take":
            if not a.src:
                ap.error("take takes --from")
            take(a.src)
            return 0
        if a.cmd == "compare-all":
            return compare_all()
        if a.cmd == "verdict":
            return 1 if verdict()["verdict"] == "INCOMPLETE" else 0
    ap.error("fit, take, compare-all, verdict or --selftest")


if __name__ == "__main__":
    sys.exit(main())
