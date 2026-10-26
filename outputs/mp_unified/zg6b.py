"""G6b (docs/G6B_GATED_BACKOFF.md): G6's next form after a GAIN and a LOSS (docs/G6_FREE_GNN.md, section 5; GL,
section 7), a per-question back-off, on Kaggle's T4 against KG's base set.

zg6b is zg6's network scored twice from one encoder pass and one context: over the question's pool edges (zg6's score,
s_e) and with no edge reaching its layers (its twin's score, s_t: the same parameters, context and readout, G6's zg6t).
A gate on the question's edge-family shares mixes the two per question:
    score = g s_e + (1 - g) s_t,   g = sigmoid(a . sh + b),
sh being each family's share among the question's edges that reach the layers (after the family dropout in training;
0 for a question without edges). a and b start at 0, so g = 0.5. Everything else is zg6's: its encoder, boundary,
context, three layers, unbounded readout, family dropout (the same draws from the same generator) and card. 121,806 own
parameters at hidden 128: zg6's 121,802 and the gate's 4.

The fits are zg6b's two on U2d's base build (scr-g6b-zg6b on L-musique, scr-g6b-zg6b-hp on L-hotpotqa, seed 0), trained
and read through zu2d.py's bodies as KG runs them, on KG's device. In the kernel (zkg_session.py runs it, one process
per fit):

    python outputs/mp_unified/zg6b.py fit --name N [--rows-cap R]

On the laptop, after `kaggle kernels output` into a new empty folder and KG's BASE_READY:

    python outputs/mp_unified/zg6b.py take --from DIR       # each fit's records, reads and gate file, never its weights
    python outputs/mp_unified/zg6b.py compare-all           # zg6b against KG's zspk (deciding) and KG's zg6 (beside)
    python outputs/mp_unified/zg6b.py verdict               # G6's rule under KG's zspk null, and the beside tables
    python outputs/mp_unified/zg6b.py --selftest
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

import zkg as KG  # noqa: E402  (KG's device, null and base set; through it zu2d, zg6 and P1c2's card)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z, G6, G2 = KG.Z, KG.G6, KG.G2
Z2, Z3, RM, ZL, SP = KG.Z2, KG.Z3, KG.RM, KG.ZL, KG.SP
S, S2, LG, LC, SR = KG.S, KG.S2, KG.LG, KG.LC, KG.SR
log = KG.log

DECLARED = "docs/G6B_GATED_BACKOFF.md"
STAGE = "G6b"
KEY = "g6b"                                  # the stamp's key in screen.json, read.json and G6b's compares
ARM = "zg6b"
SEED = 0
BUILD, SPLITS, SUF = KG.BUILD, KG.SPLITS, KG.SUF
FAMS, KT_KEY, LK_KEY, ZCLIP = G6.FAMS, G6.KT_KEY, G6.LK_KEY, G6.ZCLIP
SCREEN = KG.SCREEN                           # relative: every laptop command runs from the repository's root
OUT = Path("outputs") / "g6b"
VERDICT = "screen-verdict"
PAIR = {0: "g6b-zg6b-pair", 1: "g6b-zg6b-vs-zg6-pair"}
RECALL = {0: "g6b-zg6b-pair-recall", 1: "g6b-zg6b-vs-zg6-pair-recall"}
AGAINST = {0: "zspk", 1: "zg6"}
GATE_FILE = "g6b_gate.json"
TIMING_FILE = "g6b_timing.json"
SHARE_MIN = G6.SHARE_MIN
G6B_NAMES = ("g6b_gate_w", "g6b_gate_b")
N_PARAMS = G6.n_params(128) + FAMS + 1


def fit_name(split):
    return f"scr-g6b-{ARM}{SUF[split]}"


FITS = {fit_name(sp): sp for sp in SPLITS}
ZSPK = {sp: KG.fit_name("zspk", 0, sp) for sp in SPLITS}            # the deciding base: KG's zspk, seed 0
ZG6 = {sp: KG.fit_name("zg6", 0, sp) for sp in SPLITS}              # beside: KG's zg6
CMP = {sp: f"scr-g6b-{ARM}-vs-zspk{SUF[sp]}" for sp in SPLITS}
COMPARES = {CMP[sp]: (fit_name(sp), [ZSPK[sp], ZG6[sp]]) for sp in SPLITS}
KG_ZG6_CMP = {sp: f"scr-kg-zg6-vs-zspk{SUF[sp]}" for sp in SPLITS}  # zg6's R@5 as KG filed it
KG_ZRCKP = {sp: KG.null_name("zrckp", 1, sp) for sp in SPLITS}      # its base column: zrckp's seed-0 R@5 (the MLP)
assert all(b in KG.FITS for _n, bs in COMPARES.values() for b in bs) and all(c in KG.COMPARES for c in KG_ZG6_CMP.values())

RULE = ("ADOPT_FOR_FULL_RUN when zg6b's re-call against KG's zspk under KG's zspk seed null has no LOSS on any of its "
        "twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when KG's base verdict is not "
        "BASE_READY, a fit or compare is missing or not G6b's, the fits are not on KG's device, or the re-call does not "
        "hold the twelve reads (G6's rule, docs/G6_FREE_GNN.md section 4, on the T4)")
NEXT = {"ADOPT_FOR_FULL_RUN": "zg6b fitted again on P1g-b's pools, then its full run on all six splits beside zspk's, "
                              "then Claim 1 (P3c's intervals) and Claim 3 (cold latency, the C stages' timing) measured "
                              "again; each its own declared file",
        "GAIN_AND_LOSS": "G6b2, a per-row back-off: the gate also reads the row's own log1p in-degree per family over "
                         "the edges that reach the layers, so a row the edges barely reach can lean on the twin; one "
                         "declared file",
        "NO_GAIN": G6.NEXT["NO_GAIN"],
        "INCOMPLETE": "what is missing is supplied (KG's base set, a fit, a compare or the re-call); nothing else "
                      "changes"}


def next_form(v, gains, losses):
    if v == "NOT_ADOPTED":
        return NEXT["GAIN_AND_LOSS"] if gains and losses else NEXT["NO_GAIN"]
    return NEXT[v]


# ── the model ────────────────────────────────────────────────────────────────


def shares(f, qv, B, dt):
    """(B, FAMS): each question's share of each family among its edges (f the edges' families, qv their questions); 0
    for a question without edges. Counts are whole numbers, exact in any summation order."""
    one = torch.ones(f.numel(), dtype=dt, device=f.device)
    cnt = torch.zeros(B * FAMS, dtype=dt, device=f.device).index_add_(0, qv * FAMS + f, one).view(B, FAMS)
    return cnt / cnt.sum(1, keepdim=True).clamp_min(1.0)


def g6b_parts(m, H, xb, lk, nq, B, u01=None):
    """(score, s_e, s_t, g, sh). s_e is zg6.g6_stack's score over the edges lk (less the families the dropout drops when
    u01 is given), s_t its score with no edges (the twin), both from one h0 and context, op for op; g (B,) the gate on
    the shares sh (B x FAMS) of the edges that reach the layers; score = g s_e + (1 - g) s_t on each row."""
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
    sh = shares(f, nq[v], B, H.dtype)
    g = torch.sigmoid(Fn.linear(sh, m.g6b_gate_w, m.g6b_gate_b)).squeeze(-1)
    gq = g[nq]
    return gq * s_e + (1.0 - gq) * s_t, s_e, s_t, g, sh


class ZG6B(G6.G6Free):
    """zg6b: zg6's network over the question's edges and with none, mixed by a gate on the question's edge-family
    shares (equal weight at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        self.g6b_gate_w = nn.Parameter(torch.zeros(1, FAMS))
        self.g6b_gate_b = nn.Parameter(torch.zeros(1))

    def g6b_scores(self, feats, keep, nq, B):
        if KT_KEY not in feats:
            raise RuntimeError("zg6b: no KT input in the batch (P1c3's carve adds it)")
        if LK_KEY not in feats:
            raise RuntimeError("zg6b: no pool edges in the batch (zlink's carve adds them)")
        H = self.encode(feats, keep, nq, B)
        xb = torch.cat([feats[KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
        u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training else None
        return g6b_parts(self, H, xb, feats[LK_KEY], nq, B, u01)

    def forward(self, feats, keep, nq, B, base_z):
        """The mixed score; base_z (rrf's z-score, the MLP's base) is never read."""
        return self.g6b_scores(feats, keep, nq, B)[0]


ARMS = {ARM: (ZG6B, Z3.KTLinkCarveBase)}
if S.ARMS.get(ARM) not in (None, ARMS[ARM]):
    raise SystemExit(f"zg6b: lean_screen's arm {ARM} is already {S.ARMS[ARM]}")
S.ARMS.update(ARMS)
SETTINGS = {"network": "zg6's (docs/G6_FREE_GNN.md, section 2)", "zg6": G6.SETTINGS,
            "passes": "two from one encoder pass and one context: over the question's pool edges (zg6's score) and "
                      "with no edge reaching the layers (its twin's score)",
            "gate": "sigmoid(a . sh + b) per question; sh each family's share among the question's edges that reach "
                    "the layers (after the family dropout in training), 0 without edges; a and b 0 at the start",
            "score": "g s_e + (1 - g) s_t", "gate_parameters": FAMS + 1, "parameters_at_hidden_128": N_PARAMS}


# ── the device, train and read (KG's, with G6b's arm, names and stamp) ──────


def where_ok(env, what):
    if not KG.ok_where(env):
        raise SystemExit(f"zg6b: G6b {what}s in a Kaggle kernel on a {KG.GPU} from the image {KG.IMAGE_TAG} (KG's "
                         f"device), not with {KG.device_of(env)} ({env.get('where')})")


def check_arm():
    if S.ARMS.get(ARM) != ARMS[ARM]:
        raise SystemExit(f"zg6b: the arm {ARM} is {S.ARMS.get(ARM)}, not G6b's {ARMS[ARM]}")


def code_shas():
    return {**KG.code_shas(), "zg6b_sha256": LC.sha_src(__file__)}


def head():
    return {"model": SETTINGS, "kt_root": Z.rel(Z.roots(BUILD)["kt"])}


def train(name, env=None):
    """One of G6b's two fits on U2d's base build: zu2d.train's body as KG runs it, with G6b's arm, name and stamp."""
    if name not in FITS:
        raise SystemExit(f"zg6b: {name} is not one of G6b's fits ({', '.join(FITS)})")
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
        "kg_bases": {"zspk": ZSPK[split], "zg6": ZG6[split]}, "device": KG.device_of(env), "env": env,
        "graph": Z.GRAPH[BUILD], "pools": g.get("construction"), "rule": Z.ZP.RULE[BUILD],
        "cache_root": Z.rel(r["cache"]), "gate": LC.sha_file(r["gate"]), "head": head(), "kt_records": kt,
        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(r["ch_zrc"]),
        "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    """A G6b fit's screen.json and its G6b record, checked against the fit's name."""
    split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    k = rec.get(KEY) or {}
    if (rec.get("arm"), k.get("declared_in"), k.get("fit"), k.get("build"), k.get("arm"), k.get("seed"),
            k.get("split")) != (ARM, DECLARED, name, BUILD, ARM, SEED, split):
        raise SystemExit(f"zg6b: {sj} is not G6b's {ARM} fit of {split}")
    return rec, k


def read(name, env=None, rows_cap=0):
    """The fit read with p@swa on the six s1eval carves of U2d's base build, on the device it was fitted on."""
    if name not in FITS:
        raise SystemExit(f"zg6b: {name} is not one of G6b's fits")
    split = FITS[name]
    env = KG.device_env() if env is None else env
    where_ok(env, "read")
    _rec, k = fit_stamp(name)
    if KG.device_of(env) != k.get("device"):
        raise SystemExit(f"zg6b: {name} was fitted on {k.get('device')}; this read runs on {KG.device_of(env)}")
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


# ── the gate, beside: its parameters and its value on each s1eval question ──


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def link_parts(links_root, ds, carve):
    """zlink's parts of a carve in question order (lean_cache.part_dirs' order), all of them."""
    d = Path(links_root) / ds / carve
    parts = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not parts or {int(p.name.split("of")[1]) for p in parts} != {len(parts)}:
        raise SystemExit(f"zg6b: the link parts under {d} ({[p.name for p in parts]}) do not make one whole carve")
    return parts


def gate_values(a, b, links_root, ds, carve):
    """(g per question in the carve's order, questions without edges): the gate sigmoid(a . sh + b) on each question's
    family shares, from zlink's arrays of the carve (each array checked against its part's record) in float64."""
    parts = link_parts(links_root, ds, carve)
    gs, none = [], 0
    for p in parts:
        h = json.loads((p / "record.json").read_text(encoding="utf-8"))
        arr = {}
        for k in ("q_ne", "e_fam"):
            fn = p / f"{k}.npy"
            if LC.sha_file(fn) != h["arrays"][k]["sha256"]:
                raise SystemExit(f"zg6b: {fn}: sha256 is not its record's")
            arr[k] = np.load(fn)
        qn = arr["q_ne"].astype(np.int64)
        q = np.repeat(np.arange(qn.size), qn)
        cnt = np.bincount(q * FAMS + arr["e_fam"].astype(np.int64), minlength=qn.size * FAMS).reshape(qn.size, FAMS)
        sh = cnt / np.maximum(cnt.sum(1, keepdims=True), 1)
        gs.append(1.0 / (1.0 + np.exp(-(sh @ np.asarray(a, np.float64) + b))))
        none += int((qn == 0).sum())
    return np.concatenate(gs), none


def gate_stats(st, links_root, carve="s1eval", datasets=None):
    """The gate of a p@swa state: its parameters, its value on each pure family and without edges, and on each
    dataset's questions: mean, 10th, 50th and 90th percentiles, and the share of questions without edges."""
    a = [float(x) for x in st["g6b_gate_w"].double().view(-1)]
    b = float(st["g6b_gate_b"].double().view(-1)[0])
    out = {}
    for ds in datasets or LG.EVAL_ORDER:
        g, none = gate_values(a, b, links_root, ds, carve)
        q10, q50, q90 = (float(x) for x in np.quantile(g, [0.1, 0.5, 0.9]))
        out[ds] = {"questions": int(g.size), "no_edges": none, "no_edges_share": round(none / max(g.size, 1), 4),
                   "mean": round(float(g.mean()), 4), "q10": round(q10, 4), "q50": round(q50, 4), "q90": round(q90, 4)}
    return {"a": [round(x, 6) for x in a], "b": round(b, 6),
            "pure_family": [round(sigmoid(a[f] + b), 4) for f in range(FAMS)], "no_edges": round(sigmoid(b), 4),
            "carve": carve, "by_dataset": out}


def gate_beside(name, fits_root=None, links_root=None, carve="s1eval", datasets=None):
    """g6b_gate.json in the fit's folder: the gate of its p@swa state (models.pt) on the six s1eval carves."""
    fd = Path(fits_root or S.OUT / "fits") / name
    blob = torch.load(fd / "models.pt", map_location="cpu", weights_only=False)
    rec = {"stage": STAGE, "declared_in": DECLARED, "fit": name, "candidate": "p@swa",
           **gate_stats(blob["states"]["p@swa"], links_root or Z.roots(BUILD)["links"], carve, datasets)}
    LC.write_json(fd / GATE_FILE, rec)
    return rec


def fit(name, rows_cap=0):
    """One fit in the kernel: a read G6b stamped, with its gate file, is kept; a fit trained and stamped but not read is
    only read; otherwise the fit's files are unlinked and it trains and reads. Then the gate file; g6b_timing.json holds
    the seconds and peaks."""
    if name not in FITS:
        raise SystemExit(f"zg6b: {name} is not one of G6b's fits")
    fd = S.OUT / "fits" / name
    if stamped(fd / "read.json", name) and (fd / GATE_FILE).exists():
        log(f"zg6b fit {name}: read before, kept")
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
    """A G6b fit's device: its screen.json's G6b record, and its read.json's on the same device."""
    out_root = Path(out_root or FITS_ROOT)
    _rec, k = fit_stamp(name, out_root)
    rj = out_root / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    rk = r.get(KEY) or {}
    if rk.get("declared_in") != DECLARED or rk.get("fit") != name or rk.get("device") != k.get("device"):
        raise SystemExit(f"zg6b: {rj} is not G6b's read of {name} on the device it was fitted on")
    return k["device"]


def take(src, out_root=None):
    """Each G6b fit under src (a kernel's output: fits/<name>, or <name>) copied to outputs/screen/fits/<name>: its
    records, reads and gate file, never its weights. A fit must be stamped and read by G6b; a file there already must be
    the same bytes. KG's fits in the same output are KG's to take (zkg.py take)."""
    src = Path(src)
    base = src / "fits" if (src / "fits").is_dir() else src
    dst_root = Path(out_root or FITS_ROOT)
    took, kept = [], []
    for d in sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith("scr-g6b-")):
        if d.name not in FITS:
            raise SystemExit(f"zg6b take: {d.name} is not one of G6b's fits")
        fit_device(d.name, base)
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            if f.name in NEVER or f.suffix in (".tmp", ".part"):
                continue
            t = dst_root / d.name / f.relative_to(d)
            if t.exists():
                if LC.sha_file(t) != LC.sha_file(f):
                    raise SystemExit(f"zg6b take: {t} exists with other bytes")
                kept.append(str(t))
                continue
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, t)
            took.append(str(t))
    log(f"zg6b take: {len(took)} files copied, {len(kept)} there before")
    return took


def one_device(new, bases, out_root=None):
    devs = {new: fit_device(new, out_root), **{b: KG.fit_device(b, out_root) for b in bases}}
    if len({json.dumps(d, sort_keys=True) for d in devs.values()}) != 1:
        raise SystemExit(f"zg6b: G6b's fit and KG's bases ran on different devices or images: {devs}")
    return devs[new]


def compare_cmd(cname, out_root=None, screen=None):
    """lean_screen's compare of a G6b fit with KG's zspk (first: it decides) and KG's zg6, one device, the paths relative
    to the repository, stamped with G6b's record."""
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
                "zg6b_sha256": LC.sha_src(__file__)}
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
    log(f"zg6b compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def cmp_rows(js, cname, dev=None):
    """A G6b compare's rows {base fit: {dataset: row}}, after its checks; SystemExit naming what is wrong."""
    p = Path(js)
    if not p.exists():
        raise SystemExit(f"zg6b: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    new, bases = COMPARES[cname]
    k = rec.get(KEY) or {}
    if (k.get("declared_in"), k.get("compare"), k.get("new"), k.get("bases")) != (DECLARED, cname, new, list(bases)):
        raise SystemExit(f"zg6b: {p} is not G6b's compare {cname} of {new} with {bases}")
    if dev is not None and k.get("device") != dev:
        raise SystemExit(f"zg6b: {p} was made on {k.get('device')}, not on the fits' device {dev}")
    got = [SR.parts_of(b)[-1:] for b in rec.get("bases") or []]
    if SR.parts_of(rec.get("new", ""))[-1:] != (new,) or got != [(b,) for b in bases]:
        raise SystemExit(f"zg6b: {p} compares {rec.get('new')} with {rec.get('bases')}, not {new} with {bases}")
    split = FITS[new]
    if sorted(rec.get("trained_on", [])) != sorted(LG.FITS[split]):
        raise SystemExit(f"zg6b: {p} trained on {rec.get('trained_on')}, not {split}'s {sorted(LG.FITS[split])}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zg6b: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    out = {}
    for b, bp in zip(bases, rec["bases"]):
        rows = {r["dataset"]: r for r in rec["by_base"][bp]["rows"]}
        if sorted(rows) != sorted(LG.EVAL_ORDER):
            raise SystemExit(f"zg6b: {p} reads {sorted(rows)} against {b}, not the six datasets")
        out[b] = rows
    return rec, out


# ── the verdict ──────────────────────────────────────────────────────────────


def kg_device(kg_out=None):
    """KG's base verdict: (its device, problems); BASE_READY under KG's declaration or a problem."""
    p = Path(kg_out or KG.OUT) / f"{KG.VERDICT}.json"
    if not p.exists():
        return None, [f"{p} missing (KG's base verdict)"]
    kv = json.loads(p.read_text(encoding="utf-8"))
    if (kv.get("stage"), kv.get("declared_in"), kv.get("verdict")) != (KG.STAGE, KG.DECLARED, "BASE_READY"):
        return None, [f"KG's base verdict is {kv.get('verdict')} ({kv.get('declared_in')}), not BASE_READY"]
    return kv.get("device"), []


def restamp(od, which):
    """G6b's fields on the pair and re-call screen_pair and screen_recall wrote; a header on their markdown."""
    for name in (PAIR[which], RECALL[which]):
        js = (od / name).with_suffix(".json")
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": STAGE, "declared_in": DECLARED, "arm": ARM, "against": AGAINST[which],
                    "null_kind": "KG's zspk seed null" + ("" if which == 0 else ", its seed-0 R@5 replaced by KG's zg6 "
                                                                                "read (beside)"),
                    "deciding": which == 0, "zg6b_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)
        md = js.with_suffix(".md")
        if md.exists():
            head_ = (f"G6b ({DECLARED}): zg6b against KG's {AGAINST[which]} on the T4"
                     + (", deciding." if which == 0 else ", beside, deciding nothing.") + "\n\n")
            md.write_text(head_ + md.read_text(encoding="utf-8"), encoding="utf-8")


def recall(scr, od, dev, which):
    """screen_pair over G6b's two compares against base `which` (0: KG's zspk, deciding; 1: KG's zg6, beside), then
    screen_recall under KG's zspk null. Against zg6 the null's seed-0 R@5 is zg6's read as KG filed it (scr-kg-zg6-vs-
    zspk's new column), which the compare's zg6 column must equal; the floors stay zspk's."""
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
    if which == 1:
        for sp in SPLITS:
            _r, kz = KG.cmp_rows(scr / f"{KG_ZG6_CMP[sp]}.json", KG_ZG6_CMP[sp], dev)
            null[sp] = {ds: {**v, "seed0": kz[ds]["new"][0]} for ds, v in null[sp].items()}
        sources = {**sources, **{f"KG zg6 {sp}": str(scr / f"{KG_ZG6_CMP[sp]}.json") for sp in SPLITS}}
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


def mlp_share(mlp, arm, rrf):
    """(zrckp - rrf) / (zg6b - rrf) on R@5; None where a number is missing or zg6b's gain over rrf is SHARE_MIN or
    less."""
    return G6.mlp_share(mlp, arm, rrf)


def timing(name, fits_root):
    d = Path(fits_root) / name
    t = json.loads((d / "train.json").read_text(encoding="utf-8")) if (d / "train.json").exists() else {}
    r = json.loads((d / "read.json").read_text(encoding="utf-8")) if (d / "read.json").exists() else {}
    k = json.loads((d / TIMING_FILE).read_text(encoding="utf-8")) if (d / TIMING_FILE).exists() else {}
    return {"train_seconds": t.get("seconds"), "read_seconds": r.get("seconds"), "train_peak_gpu_gb": t.get("peak_gpu_gb"),
            "read_peak_gpu_gb": r.get("peak_gpu_gb"), "data_gb": t.get("data_gb"), "peak_gb": k.get("peak_gb")}


fmt = KG.fmt


def verdict(screen=None, out=None, out_root=None, kg_out=None):
    """G6b's verdict by G6's rule from zg6b's re-call against KG's zspk under KG's zspk null (sections 4 and 5). Beside,
    deciding nothing: zg6b against KG's zg6 under zspk's floors, R@5 per read with zrckp's and the MLP share, the gate on
    each dataset, and seconds and memory beside KG's zg6."""
    scr, od = Path(screen or SCREEN), Path(out or OUT)
    fits_root = Path(out_root or FITS_ROOT)
    od.mkdir(parents=True, exist_ok=True)
    kdev, problems = kg_device(kg_out)
    devs = {}
    for n in FITS:
        try:
            devs[n] = fit_device(n, out_root)
        except SystemExit as e:
            problems.append(str(e))
    keys = {json.dumps(d, sort_keys=True) for d in devs.values()}
    if len(keys) > 1:
        problems.append(f"G6b's fits ran on {len(keys)} devices: {sorted(keys)}")
    dev = json.loads(next(iter(keys))) if len(keys) == 1 else None
    if dev is not None and kdev is not None and dev != kdev:
        problems.append(f"G6b's fits ran on {dev}, KG's base set on {kdev}")
    rows = {}
    for c in COMPARES:
        try:
            rows[c] = cmp_rows(scr / f"{c}.json", c, dev)[1]
        except SystemExit as e:
            problems.append(str(e))
    rc, bz = None, None
    if not problems:
        try:
            rc = recall(scr, od, dev, 0)
            bz = recall(scr, od, dev, 1)
        except SystemExit as e:
            problems.append(str(e))
    rrows = rc["rows"] if rc else []
    if rc and sorted((r["split"], r["dataset"]) for r in rrows) != sorted((sp, ds) for sp in SPLITS
                                                                           for ds in LG.EVAL_ORDER):
        problems.append(f"the re-call holds {len(rrows)} reads, not the twelve of {', '.join(SPLITS)}")
    gains = [tag(r) for r in rrows if r["call"] == "GAIN"] if not problems else []
    losses = [tag(r) for r in rrows if r["call"] == "LOSS"] if not problems else []
    v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
    nxt = next_form(v, gains, losses)
    rec = {"stage": STAGE, "declared_in": DECLARED, "rule": RULE, "verdict": v, "arm": ARM, "base_arm": "zspk",
           "null_kind": "KG's zspk seed null", "device": dev, "next_form": nxt, "gains": gains, "losses": losses,
           "problems": problems, "calls": {tag(r): r["call"] for r in rrows},
           "delta_r5": {tag(r): r["delta"] for r in rrows}, "zg6b_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# G6b's screen verdict: zg6b **{v}**", "", f"Rule: {RULE}.", "",
          f"- **zg6b** (zg6's network over the edges and without them, mixed per question by a gate on the question's "
          f"edge-family shares) against KG's zspk on the T4, re-called under KG's zspk seed null: {len(gains)} GAIN "
          f"({', '.join(gains) or 'none'}), {len(losses)} LOSS ({', '.join(losses) or 'none'})"
          + (f"; problems: {'; '.join(problems)}" if problems else "") + ".",
          f"- **Next form** (fixed in section 6 before any number): {nxt}.", f"- Device: {json.dumps(dev)}."]
    beside = {}
    if v != "INCOMPLETE":
        zrckp = {}
        for sp in SPLITS:
            _r, zr = KG.cmp_rows(scr / f"{KG_ZRCKP[sp]}.json", KG_ZRCKP[sp], dev)
            zrckp[sp] = {ds: zr[ds]["base"][0] for ds in LG.EVAL_ORDER}
        vz = {tag(r): r["call"] for r in bz["rows"]}
        r5, share = {}, {}
        for r in rrows:
            sp, ds = r["split"], r["dataset"]
            zg6 = rows[CMP[sp]][ZG6[sp]][ds]["base"][0]
            r5[tag(r)] = {"read": r["read"], "rrf": r["rrf"], "zspk": r["base"], "zg6": zg6, "zg6b": r["new"],
                          "zrckp": zrckp[sp][ds], "floor": r["floor"], "call": r["call"], "call_vs_zg6": vz[tag(r)]}
            share[tag(r)] = mlp_share(zrckp[sp][ds], r["new"], r["rrf"])
        gate = {}
        for n, sp in FITS.items():
            p = fits_root / n / GATE_FILE
            gate[sp] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        tm = {sp: {"zg6b": timing(n, fits_root), "zg6": KG.timing(ZG6[sp], fits_root)} for n, sp in FITS.items()}
        beside = {"vs_zg6": {"gains": [k for k, c in vz.items() if c == "GAIN"],
                             "losses": [k for k, c in vz.items() if c == "LOSS"]},
                  "r5": r5, "mlp_share": share, "gate": gate, "timing": tm,
                  "pair": (od / PAIR[0]).with_suffix(".json").as_posix(),
                  "recall": (od / RECALL[0]).with_suffix(".json").as_posix(),
                  "recall_sha256": LC.sha_file((od / RECALL[0]).with_suffix(".json"))}
        md += ["", "Beside, deciding nothing.", "",
               f"- zg6b against KG's zg6 under zspk's floors: {len(beside['vs_zg6']['gains'])} GAIN "
               f"({', '.join(beside['vs_zg6']['gains']) or 'none'}), {len(beside['vs_zg6']['losses'])} LOSS "
               f"({', '.join(beside['vs_zg6']['losses']) or 'none'}).", "",
               f"R@5 per read on the T4; the MLP share = (zrckp - rrf) / (zg6b - rrf) where zg6b's gain over rrf "
               f"exceeds {SHARE_MIN}:", "",
               "| read | rrf | zspk | zg6 | zg6b | zrckp (MLP) | MLP share | floor | call | vs zg6 |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
        for k, t in r5.items():
            md.append(f"| {k} ({t['read']}) | {fmt(t['rrf'])} | {fmt(t['zspk'])} | {fmt(t['zg6'])} | {fmt(t['zg6b'])} | "
                      f"{fmt(t['zrckp'])} | {fmt(share[k])} | {fmt(t['floor'])} | {t['call']} | {t['call_vs_zg6']} |")
        md += ["", "The gate of each fit's p@swa: g = sigmoid(a . sh + b); its value on a question of one family alone, "
                   "without edges, and over each dataset's s1eval questions (mean, 10th / 50th / 90th percentile; the "
                   "share of questions without edges):", ""]
        for sp in SPLITS:
            gt = gate[sp]
            if gt is None:
                md.append(f"- {sp}: no gate file.")
                continue
            md.append(f"- {sp}: a = {gt['a']}, b = {gt['b']}; one family alone {gt['pure_family']}, no edges "
                      f"{gt['no_edges']}.")
            for ds in LG.EVAL_ORDER:
                x = gt["by_dataset"][ds]
                md.append(f"  - {ds}: mean {x['mean']:.4f} ({x['q10']:.4f} / {x['q50']:.4f} / {x['q90']:.4f}); "
                          f"without edges {x['no_edges_share']:.4f} of {x['questions']}.")
        md += ["", "Seconds and memory: zg6b and, in brackets, KG's zg6 of the same split, both on the T4:", "",
               "| split | train s | read s | peak GPU GB, train / read | peak RAM GB |", "|---|---:|---:|---|---:|"]
        for sp in SPLITS:
            a_, b_ = tm[sp]["zg6b"], tm[sp]["zg6"]

            def s0(x):
                return f"{x:.0f}" if isinstance(x, (int, float)) else "n/a"
            md.append(f"| {sp} | {s0(a_['train_seconds'])} ({s0(b_['train_seconds'])}) | {s0(a_['read_seconds'])} "
                      f"({s0(b_['read_seconds'])}) | {fmt(a_['train_peak_gpu_gb'])} / {fmt(a_['read_peak_gpu_gb'])} "
                      f"({fmt(b_['train_peak_gpu_gb'])} / {fmt(b_['read_peak_gpu_gb'])}) | {fmt(a_['peak_gb'])} "
                      f"({fmt(b_['peak_gb'])}) |")
    rec["beside"] = beside
    LC.write_json(od / f"{VERDICT}.json", rec)
    (od / f"{VERDICT}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zg6b verdict: {v} ({len(gains)} GAIN, {len(losses)} LOSS{'; ' + '; '.join(problems) if problems else ''}); "
        f"next: {nxt}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────

TOY_ENV = KG.TOY_ENV
stops = KG.stops


def fake_fit(td, name, dev=None, read=True, stamp=True, gate=True):
    """A G6b fit's screen.json and read.json (the stamps fit_device reads), with train.json, g6b_timing.json, a read,
    models.pt and the gate file."""
    dev = dev or KG.device_of(TOY_ENV)
    split = FITS[name]
    d = Path(td) / "fits" / name
    (d / "reads").mkdir(parents=True, exist_ok=True)
    k = {"declared_in": DECLARED, "fit": name, "build": BUILD, "arm": ARM, "seed": SEED, "split": split, "device": dev}
    LC.write_json(d / "screen.json", {"arm": ARM, "split": split, "seed": SEED, **({KEY: k} if stamp else {})})
    LC.write_json(d / "train.json", {"seconds": 150.0, "peak_gpu_gb": 3.5, "data_gb": 2.0})
    LC.write_json(d / TIMING_FILE, {"peak_gb": 9.0})
    (d / "models.pt").write_bytes(b"weights")
    (d / "reads" / "metaqa__s1eval.npz").write_bytes(b"read")
    if gate:
        LC.write_json(d / GATE_FILE, {"a": [0.5, -1.0, 0.25], "b": 0.1, "pure_family": [0.6457, 0.2891, 0.5866],
                                      "no_edges": 0.525, "carve": "s1eval", "by_dataset": {
                                          ds: {"questions": 10, "no_edges": 1, "no_edges_share": 0.1, "mean": 0.6,
                                               "q10": 0.5, "q50": 0.6, "q90": 0.7} for ds in LG.EVAL_ORDER}})
    if read:
        LC.write_json(d / "read.json", {"seconds": 12.0, "peak_gpu_gb": 4.5, KEY: dict(k)})
    return d


def fake_cmp(td, cname, deltas=None, zspk_r5=0.6, zg6_r5=None, dev=None, stamp=True, screen=None):
    """One of G6b's compares in lean_screen's form (G6b-stamped), against KG's zspk and zg6 (zg6_r5 {dataset: R@5},
    0.6 where absent); deltas {dataset: zg6b minus zspk}."""
    new, bases = COMPARES[cname]
    split = FITS[new]
    deltas, zg6_r5 = deltas or {}, zg6_r5 or {}
    bp = [str(Path(td) / "fits" / b) for b in bases]
    by = {}
    for b, r5 in zip(bp, ({ds: zspk_r5 for ds in LG.EVAL_ORDER}, {ds: zg6_r5.get(ds, 0.6) for ds in LG.EVAL_ORDER})):
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


def fake_set(td, deltas=None, kg_g6=None, big=None, kg="BASE_READY", dev=None, zg6_r5=None):
    """KG's fake base set (zkg.fake_all) with its base verdict, and G6b's two fits and compares; deltas {(split,
    dataset): zg6b minus zspk}; kg_g6 {(split, dataset): zg6 minus zspk} in KG's compares; big {(split, dataset): (D1,
    D2)} for zspk's null."""
    td = Path(td)
    deltas, kg_g6 = deltas or {}, kg_g6 or {}
    KG.fake_all(td, g6=kg_g6, big=big)
    if kg:
        LC.write_json(td / "kg" / f"{KG.VERDICT}.json", {"stage": KG.STAGE, "declared_in": KG.DECLARED, "verdict": kg,
                                                         "device": KG.device_of(TOY_ENV)})
    for n, sp in FITS.items():
        fake_fit(td, n, dev=dev)
        z6 = (zg6_r5 or {}).get(sp) or {ds: round(0.6 + kg_g6.get((sp, ds), 0.0), 4) for ds in LG.EVAL_ORDER}
        fake_cmp(td, CMP[sp], {ds: x for (s, ds), x in deltas.items() if s == sp}, zg6_r5=z6, dev=dev)


def real_state():
    """What G6b has under the real folders: its fits, its compares and outputs/g6b."""
    return (sorted(p.name for p in (S.OUT / "fits").glob("scr-g6b-*")),
            sorted(p.name for p in (ROOT / SCREEN).glob("scr-g6b-*")), (ROOT / OUT).exists() and
            sorted(p.name for p in (ROOT / OUT).iterdir()))


def brute_shares(lk, nq, B, keep_e=None):
    """(B, FAMS) family shares by python loops over the edges (keep_e: the edges kept)."""
    cnt = [[0] * FAMS for _ in range(B)]
    ks = keep_e.tolist() if keep_e is not None else [True] * lk["v"].numel()
    for v_, f_, k_ in zip(lk["v"].tolist(), lk["f"].tolist(), ks):
        if k_:
            cnt[int(nq[v_])][f_] += 1
    return torch.tensor([[c / max(sum(r), 1) for c in r] for r in cnt])


def selftest():
    import tempfile
    t0 = time.time()
    before = real_state()
    LG.bind_device_ops()
    # 1. the arm on P1c3's link carve; zg6's class below it; earlier arms unchanged; the names and bases
    assert S.ARMS[ARM] == (ZG6B, Z3.KTLinkCarveBase) and ZG6B.__mro__[1:3] == (G6.G6Free, G6.ZM.ZRM) and not ZG6B.TWIN
    assert S.ARMS["zg6"] == G6.ARMS["zg6"] and S.ARMS["zg6t"] == G6.ARMS["zg6t"] and S.ARMS["zspk"] == Z.ARMS["zspk"]
    assert N_PARAMS == 121806 and sorted(FITS) == ["scr-g6b-zg6b", "scr-g6b-zg6b-hp"]
    assert COMPARES["scr-g6b-zg6b-vs-zspk-hp"] == ("scr-g6b-zg6b-hp", ["scr-kg-base-zspk-hp", "scr-kg-base-zg6-hp"])
    assert KG_ZRCKP["L-musique"] == "scr-kg-null-zrckp-s1" and KG_ZG6_CMP["L-hotpotqa"] == "scr-kg-zg6-vs-zspk-hp"
    # 2. where: a Kaggle T4 on KG's image only; train and read refuse anything else before touching data
    for bad in ({"where": "elsewhere"}, {"gpu": "Tesla P100-PCIE-16GB"},
                {"image": "kaggle-gpu:py3.13.15:torch2.12.0+cu128:numpy2.1.3"}):
        stops(train, "scr-g6b-zg6b", {**TOY_ENV, **bad}, has="G6b trains in a Kaggle kernel")
        stops(read, "scr-g6b-zg6b-hp", {**TOY_ENV, **bad}, has="G6b reads in a Kaggle kernel")
    stops(train, "scr-kg-base-zg6", TOY_ENV, has="not one of G6b's fits")
    stops(read, "scr-g6b-zg6", TOY_ENV, has="not one of G6b's fits")
    tmp = Path(tempfile.mkdtemp(prefix="zg6b_"))
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
        # 3. from seed 0: zg6b's construction moves the global generator as zg6's does; its state is zg6's plus the
        #    gate (zero), every shared tensor equal; the family dropout's generator starts where zg6's does
        nets, after = {}, {}
        for k, cl in (("zg6", G6.ZG6), ("zg6t", G6.ZG6T), (ARM, ZG6B)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after[ARM], after["zg6"])
        m, z6, zt = nets[ARM], nets["zg6"], nets["zg6t"]
        sm, s6 = m.state_dict(), z6.state_dict()
        assert sorted(sm) == sorted(set(s6) | set(G6B_NAMES)) and all(torch.equal(sm[k], s6[k]) for k in s6)
        assert float(sm["g6b_gate_w"].abs().sum()) == 0 and float(sm["g6b_gate_b"].abs().sum()) == 0
        assert sum(int(sm[k].numel()) for k in G6.G6_NAMES + G6B_NAMES) == G6.n_params(16) + FAMS + 1
        assert torch.equal(m.g6_gen.get_state(), z6.g6_gen.get_state())
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for n in G6.G6_NAMES:
                getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
            shared = {k: x for k, x in m.state_dict().items() if k not in G6B_NAMES}
            z6.load_state_dict(shared)
            zt.load_state_dict(shared)
            for x in nets.values():
                x.eval()
            # 4. at the start (gate 0): s_e is zg6.g6_stack over the edges and s_t with none, bit for bit; zg6 and
            #    zg6t with the same weights give them; the score is half of each, bit for bit
            H = m.encode(feats, keep, nq, B)
            xb = torch.cat([feats[KT_KEY], m.chain(feats, keep, nq)], 1)
            y, se, st, g, sh = m.g6b_scores(feats, keep, nq, B)
            assert torch.equal(se, G6.g6_stack(m, H, xb, lk, nq, B)) and torch.equal(st, G6.g6_stack(m, H, xb, None, nq, B))
            assert torch.equal(se, z6(feats, keep, nq, B, bz)) and torch.equal(st, zt(feats, keep, nq, B, bz))
            assert torch.equal(g, torch.full((B,), 0.5)) and torch.equal(y, 0.5 * se + 0.5 * st)
            assert torch.equal(m(feats, keep, nq, B, torch.full_like(bz, float("nan"))), y)
            assert not torch.allclose(se, st)
            # 5. set: with a random gate the score is the model by hand (zg6's two passes by hand, the shares counted
            #    by loops)
            m.g6b_gate_w.copy_(torch.randn(1, FAMS, generator=gen))
            m.g6b_gate_b.copy_(torch.randn(1, generator=gen))
            y, se, st, g, sh = m.g6b_scores(feats, keep, nq, B)
            bs = brute_shares(lk, nq, B)
            assert torch.allclose(sh, bs, atol=1e-6)
            gw = torch.sigmoid(bs @ m.g6b_gate_w.view(-1) + m.g6b_gate_b)
            want = gw[nq] * G6.by_hand(m, H, xb, lk, nq, B, False) + (1 - gw[nq]) * G6.by_hand(m, H, xb, lk, nq, B, True)
            assert torch.allclose(y, want, atol=1e-5, rtol=1e-5), float((y - want).abs().max())
            # 6. a question without edges: its two scores are equal, bit for bit; its gate is sigmoid(b); and a batch
            #    without the edges or the KT input stops
            q0 = int(nq[lk["v"][0]])
            f3 = dict(feats)
            ke = nq[lk["v"]] != q0
            f3[LK_KEY] = {x: t[ke] for x, t in lk.items()}
            y3, se3, st3, g3, sh3 = m.g6b_scores(f3, keep, nq, B)
            rows0 = nq == q0
            assert torch.equal(se3[rows0], st3[rows0]) and torch.equal(y3[rows0], st3[rows0])
            assert float(sh3[q0].abs().sum()) == 0 and abs(float(g3[q0]) - sigmoid(float(m.g6b_gate_b))) < 1e-6
            for bad in ({k: x for k, x in feats.items() if k != LK_KEY}, {k: x for k, x in feats.items() if k != KT_KEY}):
                try:
                    m(bad, keep, nq, B, bz)
                except RuntimeError:
                    continue
                raise AssertionError("zg6b ran without its edges or its KT input")
            # 7. the family dropout: no draw at read; in training one B x FAMS draw from the module's generator, the
            #    score the model by hand over the edges zg2.family_keep keeps, the shares over those edges; the global
            #    generator moves as zg6's
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
            y, se, st, g, sh = m.g6b_scores(feats, keep, nq, B)
            assert torch.equal(m.g6_gen.get_state(), g1)
            bs = brute_shares(lk, nq, B, kd)
            assert torch.allclose(sh, bs, atol=1e-6)
            gw = torch.sigmoid(bs @ m.g6b_gate_w.view(-1) + m.g6b_gate_b)
            want = gw[nq] * G6.by_hand(m, H, xb, lk, nq, B, False, kd) + (1 - gw[nq]) * G6.by_hand(m, H, xb, lk, nq, B,
                                                                                               True)
            assert torch.allclose(y, want, atol=1e-5, rtol=1e-5), float((y - want).abs().max())
            m.drop.p = 0.1
            z6.load_state_dict({k: x for k, x in m.state_dict().items() if k not in G6B_NAMES})
            z6.g6_gen.set_state(m.g6_gen.get_state())
            z6.train()
            torch.manual_seed(9)
            m(feats, keep, nq, B, bz)
            ra = torch.rand(4)
            torch.manual_seed(9)
            z6(feats, keep, nq, B, bz)
            assert torch.equal(ra, torch.rand(4)) and torch.equal(m.g6_gen.get_state(), z6.g6_gen.get_state())
            m.eval()
            z6.eval()
        # 8. gradients reach every parameter: the encoder, the chain match, each layer, the context, the readout and
        #    the gate
        with torch.no_grad():
            m.cm_b.copy_(torch.randn(m.cm_b.shape, generator=gen) * 0.1)
            m.cm_d.copy_(torch.randn(m.cm_d.shape, generator=gen) * 0.1)
        ga = G6.grads(m, feats, keep, nq, B, bz, torch.randn(N, generator=torch.Generator().manual_seed(11)))
        assert not [k for k, x in ga.items() if not G6.nonzero(x)], [k for k, x in ga.items() if not G6.nonzero(x)]
        assert G6.nonzero(ga["g6b_gate_w"]) and G6.nonzero(ga["g6b_gate_b"]) and G6.nonzero(ga["l1.weight"])
        # 9. toy fits through lean_gpu's loop: a repeat is bit for bit; the encoder, the stack, the readout and the
        #    gate move
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            f1 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zg6b")
            f2 = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zg6b repeat")
        assert all(not LG.same_state(a, b) for a, b in zip(f1["states"] + [f1["swa"]], f2["states"] + [f2["swa"]]))
        torch.manual_seed(0)
        s0 = ZG6B(blocks, f1["widths"], 16, dropout=0.1, seed=0).state_dict()
        assert sorted(s0) == sorted(f1["swa"])
        must = ["l1.weight", "l2.weight", "U", "g6_in_w", "g6_out_w", "g6_skip_w", *G6B_NAMES] + \
            [lay[i] for lay in G6.G6_LAYER for i in (0, 1, 2, 3, 4, 5, 6)]
        still = [k for k in must if torch.equal(s0[k], f1["swa"][k])]
        assert not still, still
        # 10. the gate beside: from the toy's link arrays, each question's gate is the model's own (eval), and the
        #     stats are the brute count's; an array with other bytes stops
        m.eval()
        st = {k: x.clone() for k, x in m.state_dict().items()}
        a_, b_ = [float(x) for x in st["g6b_gate_w"].view(-1)], float(st["g6b_gate_b"])
        allq = np.arange(c.rows, dtype=np.int64)
        fa, nqa, _bza, _g = c.batch(allq, blocks)
        with torch.no_grad():
            gm = m.g6b_scores(fa, torch.ones(allq.size, len(blocks)), nqa, allq.size)[3]
        gv, none = gate_values(a_, b_, tmp / "lk", "metaqa", "toy")
        assert gv.size == c.rows and np.allclose(gv, gm.double().numpy(), atol=1e-6), float(np.abs(gv - gm.numpy()).max())
        brute, nn0 = [], 0
        for p in link_parts(tmp / "lk", "metaqa", "toy"):
            qn, ef = np.load(p / "q_ne.npy").tolist(), np.load(p / "e_fam.npy").tolist()
            e = 0
            for k in qn:
                cnt = [0] * FAMS
                for f_ in ef[e:e + k]:
                    cnt[f_] += 1
                e += k
                nn0 += k == 0
                brute.append(sigmoid(sum(a_[f_] * cnt[f_] / max(k, 1) for f_ in range(FAMS)) + b_))
            assert e == len(ef)
        assert np.allclose(gv, brute, atol=1e-12) and none == nn0, (none, nn0)
        gs = gate_stats(st, tmp / "lk", "toy", ["metaqa"])
        x = gs["by_dataset"]["metaqa"]
        assert x["questions"] == c.rows and x["mean"] == round(float(np.mean(gv)), 4) and x["no_edges"] == nn0
        assert (x["q10"], x["q50"], x["q90"]) == tuple(round(float(q), 4) for q in np.quantile(gv, [0.1, 0.5, 0.9]))
        assert gs["no_edges"] == round(sigmoid(b_), 4) and gs["pure_family"] == [round(sigmoid(a_[f] + b_), 4)
                                                                                for f in range(FAMS)]
        fd = tmp / "gfits" / "scr-g6b-zg6b"
        fd.mkdir(parents=True)
        torch.save({"states": {"p@swa": st}, "candidates": ["p@swa"]}, fd / "models.pt")
        gb = gate_beside("scr-g6b-zg6b", tmp / "gfits", tmp / "lk", "toy", ["metaqa"])
        assert json.loads((fd / GATE_FILE).read_text(encoding="utf-8")) == gb and gb["by_dataset"]["metaqa"] == x
        assert gb["fit"] == "scr-g6b-zg6b" and gb["candidate"] == "p@swa" and gb["carve"] == "toy"
        p = link_parts(tmp / "lk", "metaqa", "toy")[0]
        arr = p / "e_fam.npy"
        keep_b = arr.read_bytes()
        arr.write_bytes(keep_b[:-1] + bytes([keep_b[-1] ^ 1]))
        stops(gate_values, [0.0] * FAMS, 0.0, tmp / "lk", "metaqa", "toy", has="sha256 is not its record's")
        arr.write_bytes(keep_b)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        fits = td / "fits"
        # 11. stamps and devices: a fit not G6b's, a read on another device; take copies records, reads and the gate
        #     file, never models.pt, and leaves KG's fits to KG
        fake_fit(td, "scr-g6b-zg6b")
        assert fit_device("scr-g6b-zg6b", fits) == KG.device_of(TOY_ENV)
        fake_fit(td, "scr-g6b-zg6b-hp", stamp=False)
        stops(fit_device, "scr-g6b-zg6b-hp", fits, has="not G6b's zg6b fit")
        d = fake_fit(td, "scr-g6b-zg6b-hp")
        LC.write_json(d / "read.json", {KEY: {"declared_in": DECLARED, "fit": "scr-g6b-zg6b-hp",
                                              "device": {**KG.device_of(TOY_ENV), "gpu": "Tesla P100"}}})
        stops(fit_device, "scr-g6b-zg6b-hp", fits, has="on the device it was fitted on")
        src = td / "out"
        fake_fit(src, "scr-g6b-zg6b")
        KG.fake_fit(src, "scr-kg-base-zg6")
        took = take(src, td / "repo_fits")
        assert sorted(Path(t).name for t in took) == sorted(["g6b_gate.json", "g6b_timing.json", "metaqa__s1eval.npz",
                                                             "read.json", "screen.json", "train.json"]), took
        assert not (td / "repo_fits" / "scr-g6b-zg6b" / "models.pt").exists()
        assert not (td / "repo_fits" / "scr-kg-base-zg6").exists() and take(src, td / "repo_fits") == []
        (td / "repo_fits" / "scr-g6b-zg6b" / "train.json").write_text("{}", encoding="utf-8")
        stops(take, src, td / "repo_fits", has="other bytes")
        fake_fit(src, "scr-g6b-zg6b-hp", read=False)
        stops(take, src, td / "d2", has="not G6b's read")
        shutil.rmtree(src / "fits" / "scr-g6b-zg6b-hp")
        (src / "fits" / "scr-g6b-zg6").mkdir()
        stops(take, src, td / "d3", has="not one of G6b's fits")
        # 12. one device across G6b's fit and KG's bases
        KG.fake_fit(td, "scr-kg-base-zspk")
        KG.fake_fit(td, "scr-kg-base-zg6")
        assert one_device("scr-g6b-zg6b", ["scr-kg-base-zspk", "scr-kg-base-zg6"], fits) == KG.device_of(TOY_ENV)
        KG.fake_fit(td, "scr-kg-base-zg6", dev={**KG.device_of(TOY_ENV), "cudnn": 1})
        stops(one_device, "scr-g6b-zg6b", ["scr-kg-base-zspk", "scr-kg-base-zg6"], fits, has="different devices")
    # 13. the verdict: a GAIN and no LOSS adopts for the full run; a GAIN and a LOSS gives G6b2; no GAIN (a GAIN under
    #     a floor the null raises included) gives G6r; INCOMPLETE without KG's BASE_READY, without a compare, with the
    #     fits on another device than KG's, or with a zg6 column that is not KG's zg6 read; beside: the zg6 re-call,
    #     R@5 with zrckp and the MLP share, the gate, the timing
    gm_, lw = ("L-musique", "metaqa"), ("L-hotpotqa", "webqsp")
    cases = (({gm_: 0.02}, None, "ADOPT_FOR_FULL_RUN", NEXT["ADOPT_FOR_FULL_RUN"]),
             ({gm_: 0.02, lw: -0.03}, None, "NOT_ADOPTED", NEXT["GAIN_AND_LOSS"]),
             ({}, None, "NOT_ADOPTED", NEXT["NO_GAIN"]),
             ({("L-musique", "musique"): 0.02}, {("L-musique", "musique"): (0.02, -0.01)}, "NOT_ADOPTED",
              NEXT["NO_GAIN"]),
             ({lw: -0.03}, None, "NOT_ADOPTED", NEXT["NO_GAIN"]))
    for i, (deltas, big, want, nxt) in enumerate(cases):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            fake_set(td, deltas, kg_g6={("L-hotpotqa", "2wiki"): 0.01}, big=big)
            v = verdict(td, td / "g6b", td / "fits", td / "kg")
            assert v["verdict"] == want and v["next_form"] == nxt, (i, v["verdict"], v["problems"])
            b = v["beside"]
            assert b["r5"]["L-musique metaqa"]["zrckp"] == 0.59 and b["r5"]["L-hotpotqa 2wiki"]["zg6"] == 0.61
            assert b["r5"]["L-musique squad"]["zg6b"] == 0.6 and b["mlp_share"]["L-musique squad"] == round(0.19 / 0.2, 4)
            assert b["gate"]["L-musique"]["no_edges"] == 0.525 and b["timing"]["L-hotpotqa"]["zg6b"]["train_seconds"] == 150
            assert b["timing"]["L-hotpotqa"]["zg6"]["train_seconds"] == 100.0
            rz = json.loads((td / "g6b" / f"{RECALL[1]}.json").read_text(encoding="utf-8"))
            assert rz["stage"] == STAGE and rz["against"] == "zg6" and not rz["deciding"]
            rd = json.loads((td / "g6b" / f"{RECALL[0]}.json").read_text(encoding="utf-8"))
            assert rd["deciding"] and rd["against"] == "zspk" and len(rd["rows"]) == 12
            if i == 0:
                assert v["gains"] == ["L-musique metaqa"] and b["vs_zg6"]["gains"] == ["L-musique metaqa"]
            if i == 3:
                assert rd["changed"] == ["L-musique musique GAIN -> WITHIN"], rd["changed"]
            assert (td / "g6b" / f"{VERDICT}.md").read_text(encoding="utf-8").startswith("# G6b's screen verdict")
    for case in ("no_kg", "kg_incomplete", "no_compare", "other_device", "zg6_column"):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            if case == "other_device":
                fake_set(td, {gm_: 0.02}, dev={**KG.device_of(TOY_ENV), "cudnn": 1})
            elif case == "zg6_column":
                fake_set(td, {gm_: 0.02}, zg6_r5={"L-musique": {ds: 0.6 for ds in LG.EVAL_ORDER} | {"squad": 0.61}})
            else:
                fake_set(td, {gm_: 0.02}, kg={"no_kg": None, "kg_incomplete": "INCOMPLETE"}.get(case, "BASE_READY"))
            if case == "no_compare":
                (td / f"{CMP['L-hotpotqa']}.json").unlink()
            v = verdict(td, td / "g6b", td / "fits", td / "kg")
            assert v["verdict"] == "INCOMPLETE" and v["next_form"] == NEXT["INCOMPLETE"] and v["beside"] == {}, \
                (case, v["verdict"], v["problems"])
            has = {"no_kg": "KG's base verdict", "kg_incomplete": "not BASE_READY", "no_compare": "missing",
                   "other_device": "KG's base set on", "zg6_column": "compared with seed-0 R@5"}[case]
            assert any(has in p for p in v["problems"]), (case, v["problems"])
    # 14. nothing written under the real folders
    assert real_state() == before, (real_state(), before)
    log(f"zg6b selftest: the arm on P1c3's link carve with zg6's class below it, earlier arms unchanged; where-checks; "
        f"from seed 0 zg6's draws and state plus a zero gate ({N_PARAMS} parameters at hidden 128); at the start zg6's "
        f"score and its twin's bit for bit, half of each; by hand with a random gate; a question without edges; the "
        f"family dropout's draws and the global generator as zg6's; gradients reach every parameter; toy fits repeat "
        f"bit for bit and move the gate; the gate beside equals the model's own gate and a brute count; stamps, take "
        f"and one device; the verdict's outcomes, next forms, refusals and beside rows; nothing written under G6b's "
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
