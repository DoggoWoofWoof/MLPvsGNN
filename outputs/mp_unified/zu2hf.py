"""U2h's full run (docs/U2H_FULL_RUN.md): the lean pair with KT_W (zu2h's zrckpw and zspkw) on all six splits under the
seed null of every split, on U2d's base build, with zg6 on the same input (zg6w) beside it.
    fits     scr-u2hf-<arm>-<split>, scr-u2hf-base-<arm>[-s<seed>]-<split>, scr-u2hf-zg6w-<split>,
             scr-u2hf-base-zg6-<split>, and the device probes scr-u2hf-probe-l4 / -t4 (zrckp, seed 0, J5)
    reused   U2h's and U2d's fits and compares of L-musique and L-hotpotqa, as filed (Modal's L40S)
    device   D, the probe GPU with the lower measured cost per fit (section 3), filed in outputs/u2hf/device.json
On Modal (zu2d_modal.py's hf stages run these in the container's process):
    python outputs/mp_unified/zu2hf.py train --split J5 --name scr-u2hf-zrckpw-j5 --arm zrckpw --device cuda \\
        --cache-root outputs/u2d/base/cache --modal
    python outputs/mp_unified/zu2hf.py read --name scr-u2hf-zrckpw-j5 --device cuda --cache-root outputs/u2d/base/cache \\
        --modal
On the laptop, from the fetched fits:
    python outputs/mp_unified/zu2hf.py plan
    python outputs/mp_unified/zu2hf.py device
    python outputs/mp_unified/zu2hf.py compare-all
    python outputs/mp_unified/zu2hf.py decide
    python outputs/mp_unified/zu2hf.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zu2h as ZH  # noqa: E402  (U2h's arms, carves and KT_W records; through it zu2g, zu2d, zu2d_pools, zg6)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

Z = ZH.Z                                       # zu2d
Z3, Z2, ZM, ZL, RM = ZH.Z3, ZH.Z2, ZH.ZM, ZH.ZL, ZH.RM
S, S2, LG, LC, SR, R, G6 = ZH.S, ZH.S2, ZH.LG, ZH.LC, ZH.SR, ZH.R, ZH.G6
log, rel, sha_src = ZH.log, ZH.rel, ZH.sha_src

DECLARED = "docs/U2H_FULL_RUN.md"
OUT = ROOT / "outputs" / "u2hf"
SCREEN = ROOT / "outputs" / "screen"
DEVICE = OUT / "device.json"
BUILD = "base"
SPLITS = tuple(S2.SPLITS)
OLD = tuple(ZH.SPLITS)                         # U2h's screen splits: fits and compares reused as filed
NEW = tuple(sp for sp in SPLITS if sp not in OLD)
assert NEW == ("J5", "L-metaqa", "L-squad", "L-2wiki") and OLD == ("L-musique", "L-hotpotqa")
SUF = {sp: "-" + sp.lower().replace("-", "") for sp in SPLITS}    # -j5, -lmetaqa, -lsquad, -lmusique, -lhotpotqa, -l2wiki
ARMS_H = ZH.ARMS_H
BASE_ARM = dict(ZH.BASE_ARM)                   # zrckpw -> zrckp, zspkw -> zspk
LEAN = ("zrckp", "zspk")
SEEDS = (0, *SR.SEEDS)
KTW_KEY, KT_KEY, NKT = ZH.KTW_KEY, ZH.KT_KEY, ZH.NKT
IMAGE_TAG = Z.IMAGE_TAG

# ── the device (section 3) ───────────────────────────────────────────────────

GPUS = ("L4", "T4")                            # probed; the order breaks a tie
FALLBACK = "L40S"                              # U2h's device, when neither probe trains and reads
PRICE = {"L4": 0.80, "T4": 0.59, "L40S": 1.95}  # $ per GPU-hour, Modal's list price on 23 October 2026
CPUS, CPU_H, MEM_GIB, GIB_H = 4, 0.047, 16, 0.008
TIE = 0.01
GPU_RE = {"L4": r"\bL4\b", "T4": r"\bT4\b", "L40S": r"\bL40S\b"}
PROBE = {g: f"scr-u2hf-probe-{g.lower()}" for g in GPUS}
PROBE_OF = {v: k for k, v in PROBE.items()}


def hourly(g):
    return PRICE[g] + CPUS * CPU_H + MEM_GIB * GIB_H


def gpu_is(name, g):
    """Whether a CUDA device name (torch.cuda.get_device_name) is the GPU type g ("NVIDIA L40S" is not an L4)."""
    return bool(re.search(GPU_RE[g], str(name or "")))


def choose(seconds):
    """D and each probe's cost per fit from {gpu: total seconds, or None when the probe did not train and read}."""
    cost = {g: round(s / 3600.0 * hourly(g), 4) for g, s in seconds.items() if s}
    if not cost:
        return FALLBACK, cost
    lo = min(cost.values())
    return next(g for g in GPUS if g in cost and cost[g] <= lo * (1 + TIE)), cost


def device_rec(path=None):
    p = Path(path or DEVICE)
    if not p.exists():
        raise SystemExit(f"zu2hf: {p} missing (zu2hf.py device, after the probes; section 3)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    if rec.get("declared_in") != DECLARED or rec.get("D") not in (*GPUS, FALLBACK):
        raise SystemExit(f"zu2hf: {p} is not U2h's full run's device record")
    return rec


def device(path=None):
    return device_rec(path)["D"]


def gpu_for(name):
    """The GPU a fit must run on: a probe's own, every other fit's D."""
    if name not in FITS:
        raise SystemExit(f"zu2hf: {name} is not one of the full run's fits")
    return PROBE_OF.get(name) or device()


def where_ok(where, env, name, what):
    g = gpu_for(name)
    if not (where == "modal" and env.get("where") == "modal" and env.get("image") == IMAGE_TAG
            and gpu_is(env.get("gpu"), g)):
        raise SystemExit(f"zu2hf: {name} {what}s on Modal's {g} from the image {IMAGE_TAG} (section 3), not on {where} "
                         f"with {Z.device_of(env)}")


# ── zg6w (section 2) ─────────────────────────────────────────────────────────

D6, LAYERS, FAMS, LCLIP, ZCLIP = G6.D, G6.LAYERS, G6.FAMS, G6.LCLIP, G6.ZCLIP
KW_IN = "g6_in_kw"
KW_LAYER = tuple(f"g6_{k}_kw" for k in range(LAYERS))
assert len(G6.G6_LAYER) == LAYERS
N_KW = D6 * NKT * (1 + LAYERS)                 # 2,048


def g6w_layers(m, h, xb, kw, w, u, v, f, nq, B):
    """zg6.g6_layers, op for op, with KT_W's term added last in each layer's pre-activation."""
    N = h.shape[0]
    dev, dt = h.device, h.dtype
    E = u.numel()
    at = v * FAMS + f
    one = torch.ones(E, dtype=dt, device=dev)
    deg = torch.zeros(N * FAMS, dtype=dt, device=dev).index_add_(0, at, one).view(N, FAMS, 1).clamp_min(1.0)
    qv = nq[v]
    for lay, kn in zip(G6.G6_LAYER, KW_LAYER):
        wm, wa, fb, ws, wmean, wagg, wx, b = (getattr(m, n) for n in lay)
        c = torch.zeros(B, D6, dtype=dt, device=dev).index_add_(0, nq, w * h)
        hm = Fn.linear(h, wm).view(N, FAMS, D6)
        msg = hm[u, f]
        lg = ((Fn.linear(h, wa)[u] * c[qv]).sum(1) / math.sqrt(D6) + fb[f]).clamp(-LCLIP, LCLIP)
        mean = torch.zeros(N * FAMS, D6, dtype=dt, device=dev).index_add_(
            0, at, torch.sigmoid(lg).unsqueeze(1) * msg).view(N, FAMS, D6) / deg
        ex = torch.exp(lg)
        num = torch.zeros(N, D6, dtype=dt, device=dev).index_add_(0, v, ex.unsqueeze(1) * msg)
        den = torch.zeros(N, dtype=dt, device=dev).index_add_(0, v, ex)
        att = num / den.clamp_min(1e-30).unsqueeze(1)
        pre = Fn.linear(h, ws) + Fn.linear(mean.reshape(N, FAMS * D6), wmean) + Fn.linear(att, wagg) + \
            Fn.linear(xb, wx) + b + Fn.linear(kw, getattr(m, kn))
        h = h + Fn.gelu(Fn.layer_norm(pre, (D6,)))
    return h


def g6w_stack(m, H, xb, kw, lk, nq, B, u01=None):
    """zg6.g6_stack with KT_W's term added last in h0's pre-activation, and g6w_layers."""
    xb, kw = xb.to(H.dtype), kw.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b) + Fn.linear(kw, getattr(m, KW_IN)))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-ZCLIP, ZCLIP)
    w = G6.G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f = G6.edges_of(lk, nq, B, u01, H.device)
    h = g6w_layers(m, h0, xb, kw, w, u, v, f, nq, B)
    return (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)


class ZG6W(G6.ZG6):
    """zg6w: zg6 with KT_W entering h0 and every layer through zero-initialised weights (zg6's score at the start)."""

    def __init__(self, blocks, widths, hidden=128, dropout=0.1, seed=0, arm="ctl", ctx="none"):
        super().__init__(blocks, widths, hidden, dropout, seed, arm, ctx)
        setattr(self, KW_IN, nn.Parameter(torch.zeros(D6, NKT)))
        for kn in KW_LAYER:
            setattr(self, kn, nn.Parameter(torch.zeros(D6, NKT)))

    def forward(self, feats, keep, nq, B, base_z):
        """zg6's forward with KT_W beside the boundary; base_z is never read."""
        if KT_KEY not in feats:
            raise RuntimeError("zg6w: no KT input in the batch (P1c3's carve adds it)")
        if G6.LK_KEY not in feats:
            raise RuntimeError("zg6w: no pool edges in the batch (zlink's carve adds them)")
        if KTW_KEY not in feats:
            raise RuntimeError("zg6w: no KT_W input in the batch (zu2h's carve adds it)")
        H = self.encode(feats, keep, nq, B)
        xb = torch.cat([feats[KT_KEY].to(H.dtype), self.chain(feats, keep, nq).to(H.dtype)], 1)
        u01 = torch.rand(B, FAMS, generator=self.g6_gen) if self.training else None
        return g6w_stack(self, H, xb, feats[KTW_KEY], feats[G6.LK_KEY], nq, B, u01)


ARMS = {"zg6w": (ZG6W, ZH.KTWLinkCarveBase)}
if S.ARMS.get("zg6w") not in (None, ARMS["zg6w"]):
    raise SystemExit(f"zu2hf: lean_screen's arm zg6w is already {S.ARMS['zg6w']}")
S.ARMS.update(ARMS)


def arm_entry(kind, arm):
    if kind == "arm":
        return ZH.ARMS[arm]
    if kind == "zg6w":
        return ARMS[arm]
    return Z.ARMS[arm]


def head_of(kind, arm):
    if kind == "arm":
        return ZH.head_of(arm)
    if kind == "zg6w":
        return {"model": G6.SETTINGS, "ktw": {"columns": [f"{c} on W2_S" for c in ZH.KT_COLS],
                                              "enters": "h0 and each layer's pre-activation, added last",
                                              "init": "zero", "parameters": N_KW, "ktw_root": rel(ZH.KTW)}}
    return Z.head_of(arm, BUILD)


# ── the fits (section 2) ─────────────────────────────────────────────────────


def fit_name(kind, arm, seed, split):
    if kind == "arm":
        return f"scr-u2hf-{arm}{SUF[split]}"
    if kind == "zg6w":
        return f"scr-u2hf-zg6w{SUF[split]}"
    return f"scr-u2hf-base-{arm}{f'-s{seed}' if seed else ''}{SUF[split]}"


def _fits():
    out = {}
    for g in GPUS:
        out[PROBE[g]] = ("probe", "zrckp", 0, "J5")
    for sp in NEW:
        for a in ARMS_H:
            out[fit_name("arm", a, 0, sp)] = ("arm", a, 0, sp)
        for b in LEAN:
            for sd in SEEDS:
                if (b, sd, sp) != ("zrckp", 0, "J5"):          # the probe on D
                    out[fit_name("base", b, sd, sp)] = ("base", b, sd, sp)
    out[fit_name("base", "zrckp", 0, "J5")] = ("base", "zrckp", 0, "J5")     # run only when D falls back to L40S
    for sp in NEW + OLD:
        out[fit_name("zg6w", "zg6w", 0, sp)] = ("zg6w", "zg6w", 0, sp)
        out[fit_name("base", "zg6", 0, sp)] = ("base", "zg6", 0, sp)
    return out


FITS = _fits()
assert len(FITS) == 2 + 8 + 23 + 1 + 12 == 46
KTW_KINDS = ("arm", "zg6w")
GROUPS = {"probe": [PROBE[g] for g in GPUS],
          "arms": [fit_name("arm", a, 0, sp) for sp in NEW for a in ARMS_H],
          "base": [fit_name("base", b, 0, sp) for sp in NEW for b in LEAN if (b, sp) != ("zrckp", "J5")],
          "fallback": [fit_name("base", "zrckp", 0, "J5")],
          "null": [fit_name("base", b, sd, sp) for sp in NEW for b in LEAN for sd in SR.SEEDS],
          "zg6w": [fit_name("zg6w", "zg6w", 0, sp) for sp in NEW + OLD],
          "zg6": [fit_name("base", "zg6", 0, sp) for sp in NEW + OLD]}
WHERE = {"probe": "workspace 3", "arms": "workspace 4", "base": "workspace 3",
         "fallback": "workspace 3, only when D is L40S", "null": "workspace 5",
         "zg6w": "workspace 4", "zg6": "workspace 3 or 5"}
assert sorted(n for g in GROUPS.values() for n in g) == sorted(FITS)


def arm_fit(arm, sp):
    return ZH.fit_name(arm, sp) if sp in OLD else fit_name("arm", arm, 0, sp)


def base_fit(b, seed, sp, dev=None):
    """The base arm b's fit with this seed on split sp: U2d's on the screen splits, the full run's on the new ones (zrckp's
    J5 seed-0 fit is the probe on D); zg6's is the full run's on every split."""
    if b == "zg6":
        return fit_name("base", "zg6", 0, sp)
    if sp in OLD:
        return Z.fit_name("base", b, seed, sp)
    if (b, seed, sp) == ("zrckp", 0, "J5"):
        return PROBE[dev or device()] if (dev or device()) in GPUS else fit_name("base", b, seed, sp)
    return fit_name("base", b, seed, sp)


# ── train and read ───────────────────────────────────────────────────────────


def code_shas():
    return {**ZH.code_shas(), "zu2hf_sha256": sha_src(__file__)}


def train(argv, split, where="laptop", env=None):
    """One of the full run's fits: its kind, arm, seed and split from its name, U2d's card on the GPU, every carve on U2d's
    base build (KT_W beside its KT for the arms and zg6w); the base gate PASS; stamped with the device it ran on."""
    arm = R.arm_of(argv)
    name, _ = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2hf: {name} is not one of the full run's fits (section 2)")
    kind, farm, seed, fsplit = FITS[name]
    if arm != farm or split != fsplit:
        raise SystemExit(f"zu2hf: {name} is {farm}'s fit of {fsplit}, not {arm}'s of {split}")
    if S.ARMS.get(arm) != arm_entry(kind, arm):
        raise SystemExit(f"zu2hf: the arm {arm} is {S.ARMS.get(arm)}, not {arm_entry(kind, arm)}")
    bad = G6.fixed_flags(argv, Z.REFUSED_TRAIN)
    if bad:
        raise SystemExit(f"zu2hf: the full run trains on U2d's card (seed from the fit's name); {bad} would change that")
    env = Z.device_env() if env is None else env
    where_ok(where, env, name, "train")
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv) + ["--seed", str(seed)]
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.train_carves(split), r["kt"])
        ktw = ZH.ktw_records(Z3.train_carves(split)) if kind in KTW_KINDS else None
        rc = RM.train(argv, split)
        stamp_fit(argv, name, env, g, kt, ktw)
    return rc


def stamp_fit(argv, name, env, g, kt, ktw):
    kind, arm, seed, split = FITS[name]
    _, out_root = S2.where(argv)
    sj = Path(out_root) / name / "screen.json"
    if not sj.exists():
        return
    rec = json.loads(sj.read_text(encoding="utf-8"))
    r = Z.roots(BUILD)
    rec.update({**code_shas(), "u2hf": {
        "declared_in": DECLARED, "fit": name, "kind": kind, "build": BUILD, "arm": arm, "seed": seed, "split": split,
        "gpu_rule": PROBE_OF.get(name, "D"), "device": Z.device_of(env), "env": env, "graph": Z.GRAPH[BUILD],
        "beside": ZH.GRAPH_W if kind in KTW_KINDS else None, "pools": g.get("construction"),
        "rule": ZH.ZP.RULE[BUILD], "cache_root": rel(r["cache"]), "gate": LC.sha_file(r["gate"]),
        "head": head_of(kind, arm), "kt_records": kt, "ktw_records": ktw, "rmatch_records": RM.built_records(),
        "zrc_records": Z2._BUILT(r["ch_zrc"]), "zlink_records": ZL.built_records()}})
    LC.write_json(sj, rec)


def fit_stamp(name, out_root=None):
    kind, arm, seed, split = FITS[name]
    sj = Path(out_root or S.OUT / "fits") / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    u = rec.get("u2hf") or {}
    if (rec.get("arm"), u.get("declared_in"), u.get("fit"), u.get("build"), u.get("arm"), u.get("seed"),
            u.get("split")) != (arm, DECLARED, name, BUILD, arm, seed, split):
        raise SystemExit(f"zu2hf: {sj} is not the full run's {arm} fit of {split} with seed {seed}")
    return rec, u


def read(argv, where="laptop", env=None):
    """The fit read with p@swa on the six s1eval carves of the base build (KT_W beside for the arms and zg6w), on the
    device it was fitted on."""
    name, out_root = S2.where(argv)
    if name not in FITS:
        raise SystemExit(f"zu2hf: {name} is not one of the full run's fits")
    kind, arm, seed, split = FITS[name]
    bad = G6.fixed_flags(argv, Z.REFUSED_READ)
    if bad:
        raise SystemExit(f"zu2hf: the full run reads p's fits on the six s1eval carves of the base build; {bad} would "
                         f"change that")
    _rec, u = fit_stamp(name, out_root)
    env = Z.device_env() if env is None else env
    where_ok(where, env, name, "read")
    if Z.device_of(env) != u.get("device"):
        raise SystemExit(f"zu2hf: {name} was fitted on {u.get('device')}; this read runs on {Z.device_of(env)}")
    with Z.on_u2d(BUILD) as r:
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        g = Z.gate_passed(BUILD)
        kt = Z3.kt_records(Z3.eval_carves(), r["kt"])
        ktw = ZH.ktw_records(Z3.eval_carves()) if kind in KTW_KINDS else None
        rc = RM.read(argv)
    rj = Path(out_root) / name / "read.json"
    if rj.exists():
        rec = json.loads(rj.read_text(encoding="utf-8"))
        rec.update({**code_shas(), "u2hf": {
            "declared_in": DECLARED, "fit": name, "kind": kind, "build": BUILD, "arm": arm, "seed": seed,
            "split": split, "device": Z.device_of(env), "env": env, "gate": LC.sha_file(Z.roots(BUILD)["gate"]),
            "pools": g.get("construction"), "kt_records": kt, "ktw_records": ktw}})
        LC.write_json(rj, rec)
    return rc


# ── the device record (section 3) ────────────────────────────────────────────


def fit_device(name, out_root=None):
    """A fetched fit's device: the full run's from its u2hf stamps (the read on the fit's device), U2h's and U2d's through
    their modules."""
    if name not in FITS:
        return ZH.fit_device(name, out_root)
    _rec, u = fit_stamp(name, out_root)
    rj = Path(out_root or S.OUT / "fits") / name / "read.json"
    r = json.loads(rj.read_text(encoding="utf-8")) if rj.exists() else {}
    ru = r.get("u2hf") or {}
    if ru.get("declared_in") != DECLARED or ru.get("fit") != name or ru.get("device") != u.get("device"):
        raise SystemExit(f"zu2hf: {rj} is not the full run's read of {name} on the device it was fitted on")
    return u["device"]


def probe_seconds(g, out_root=None):
    """A probe's total seconds from its timing.json, when it trained and read in one container on its own GPU; else
    None (with the reason)."""
    name = PROBE[g]
    fd = Path(out_root or S.OUT / "fits") / name
    try:
        dev = fit_device(name, out_root)
    except SystemExit as e:
        return None, str(e)
    tp = fd / "timing.json"
    t = json.loads(tp.read_text(encoding="utf-8")) if tp.exists() else {}
    if t.get("fit") != name or t.get("gpu") != g or not t.get("trained_here") or not gpu_is(dev.get("gpu"), g):
        return None, f"{tp}: not one container's train and read of {name} on {g}"
    return float(t["total_seconds"]), None


def device_cmd(out_root=None, out=None):
    seconds, why = {}, {}
    for g in GPUS:
        seconds[g], why[g] = probe_seconds(g, out_root)
    d, cost = choose(seconds)
    rec = {"declared_in": DECLARED, "stage": "U2hf device", "D": d, "seconds": seconds, "cost_per_fit": cost,
           "not_used": {g: w for g, w in why.items() if w}, "prices": PRICE,
           "container": {"cpus": CPUS, "cpu_hour": CPU_H, "memory_gib": MEM_GIB, "gib_hour": GIB_H}, "tie": TIE,
           "rule": "the probe GPU with the lower cost per fit among the probes that trained and read; within 1%: L4; "
                   "neither: L40S (docs/U2H_FULL_RUN.md, section 3)",
           "base_fit_zrckp_j5": PROBE[d] if d in GPUS else None, "zu2hf_sha256": sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(Path(out or DEVICE), rec)
    log(f"zu2hf device: D = {d}; cost per fit {cost}; not used {rec['not_used'] or 'none'}")
    return rec


# ── the compares ─────────────────────────────────────────────────────────────


def compares(dev=None):
    """{name: (new, bases, one_device)}: the decision compares and the null on the four new splits, zg6w against zg6 on
    every split, and the two probes (device noise, beside)."""
    d = dev or device()
    out = {}
    for sp in NEW:
        for arm in ARMS_H:
            out[f"scr-u2hf-dh-{arm}{SUF[sp]}"] = (arm_fit(arm, sp), [base_fit(BASE_ARM[arm], 0, sp, d)], True)
        for b in LEAN:
            for sd in SR.SEEDS:
                out[f"scr-u2hf-null-{b}-s{sd}{SUF[sp]}"] = (base_fit(b, sd, sp, d), [base_fit(b, 0, sp, d)], True)
    for sp in NEW + OLD:
        out[f"scr-u2hf-g6-zg6w{SUF[sp]}"] = (fit_name("zg6w", "zg6w", 0, sp), [base_fit("zg6", 0, sp)], True)
    if d in GPUS:
        other = next(g for g in GPUS if g != d)
        out["scr-u2hf-probe"] = (PROBE[other], [PROBE[d]], False)
    return out


def devices_of(names, one=True, out_root=None):
    devs = {n: fit_device(n, out_root) for n in names}
    same = len({json.dumps(x, sort_keys=True) for x in devs.values()}) == 1
    if one and not same:
        raise SystemExit(f"zu2hf: the fits ran on different devices or images: {devs}")
    return devs, same


def compare_cmd(cname, new, bases, one=True, out_root=None, screen=None):
    devs, same = devices_of([new, *bases], one, out_root)
    fits = Path(out_root or S.OUT / "fits")
    out = Path(screen or SCREEN) / cname

    def r_(p):
        p = Path(p)
        return p.relative_to(ROOT).as_posix() if p.is_absolute() and p.is_relative_to(ROOT) else str(p)

    with Z.at_root():
        rc = S2.main(["compare", "--new", r_(fits / new), "--base", ",".join(r_(fits / b) for b in bases),
                      "--out", r_(out)])
    js = out.with_suffix(".json")
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec["u2hf"] = {"declared_in": DECLARED, "compare": cname, "new": new, "bases": list(bases), "devices": devs,
                   "one_device": same, "zu2hf_sha256": sha_src(__file__)}
    LC.write_json(js, rec)
    return rc


def compare_all(only=None, out_root=None, screen=None):
    done, ran, skipped = [], [], []
    for cname, (new, bases, one) in compares().items():
        if only and cname not in only:
            continue
        js = Path(screen or SCREEN) / f"{cname}.json"
        if js.exists():
            u = json.loads(js.read_text(encoding="utf-8")).get("u2hf") or {}
            if u.get("declared_in") == DECLARED and u.get("new") == new and u.get("bases") == list(bases):
                done.append(cname)
                continue
        try:
            compare_cmd(cname, new, bases, one, out_root, screen)
            ran.append(cname)
        except SystemExit as e:
            skipped.append(f"{cname}: {e}")
    log(f"zu2hf compare-all: {len(ran)} run, {len(done)} filed before, {len(skipped)} skipped"
        + (f" ({'; '.join(skipped)})" if skipped else ""))
    return 1 if skipped else 0


# ── the null and the grade (section 4) ───────────────────────────────────────


def _load_cmp(p, stamp, declared, new, base, split):
    """A compare's rows, after checking it is the declared one: its stamp, its new fit and deciding base, p@swa on
    s1eval, the split's fit carves, one device and six datasets."""
    p = Path(p)
    if not p.exists():
        raise SystemExit(f"zu2hf: {p} missing")
    rec = json.loads(p.read_text(encoding="utf-8"))
    st = rec.get(stamp) or {}
    if st.get("declared_in") != declared:
        raise SystemExit(f"zu2hf: {p} is not stamped by {declared}")
    if st.get("one_device") is not True:
        raise SystemExit(f"zu2hf: {p}'s fits ran on different devices")
    if SR.parts_of(rec["new"])[-1] != new or SR.parts_of(rec["bases"][0])[-1] != base:
        raise SystemExit(f"zu2hf: {p} compares {rec['new']} with {rec['bases'][0]}, not {new} with {base}")
    if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval":
        raise SystemExit(f"zu2hf: {p} reads {rec.get('candidate')} on {rec.get('carve')}, not p@swa on s1eval")
    if sorted(rec["trained_on"]) != sorted(LG.FITS[split]):
        raise SystemExit(f"zu2hf: {p} trained on {rec['trained_on']}, not {split}'s {sorted(LG.FITS[split])}")
    rows = rec["by_base"][rec["bases"][0]]["rows"]
    if sorted(r["dataset"] for r in rows) != sorted(LG.EVAL_ORDER):
        raise SystemExit(f"zu2hf: {p} reads {[r['dataset'] for r in rows]}, not the six datasets")
    return rows


def null_of(b, screen=None, dev=None):
    """The base arm b's seed null over all six splits in screen_recall.load_null's form ({split: {dataset: read}},
    sources): U2d's filed null on the screen splits, the full run's null compares on the new ones."""
    screen = Path(screen or SCREEN)
    null, sources = Z.u2d_null(b, Z.null_files(b, screen))
    d = dev or device()
    for sp in NEW:
        got = {}
        for sd in SR.SEEDS:
            p = screen / f"scr-u2hf-null-{b}-s{sd}{SUF[sp]}.json"
            rows = _load_cmp(p, "u2hf", DECLARED, base_fit(b, sd, sp, d), base_fit(b, 0, sp, d), sp)
            got[sd] = {r["dataset"]: r for r in rows}
            sources[f"U2hf {b} {sp} seed {sd}"] = rel(p)
        null[sp] = {}
        for ds in LG.EVAL_ORDER:
            r1, r2 = got[1][ds], got[2][ds]
            if r1["base"][0] != r2["base"][0]:
                raise SystemExit(f"zu2hf: {sp} {ds}: {b}'s two seeds are compared with different seed-0 R@5")
            d1, d2 = r1["delta"][0], r2["delta"][0]
            null[sp][ds] = {"read": r1["read"], "seed0": r1["base"][0], "d1": d1, "d2": d2,
                            "spread": math.sqrt((d1 * d1 + d2 * d2) / 2.0), "floor": SR.floor_of(d1, d2)}
    return null, sources


def decision_file(arm, sp, screen=None):
    screen = Path(screen or SCREEN)
    if sp in OLD:
        return screen / f"{ZH.decision_cmp('H', arm, sp)}.json", "u2h", ZH.DECLARED
    return screen / f"scr-u2hf-dh-{arm}{SUF[sp]}.json", "u2hf", DECLARED


def grade(arm, screen=None, dev=None):
    """The arm's 36 reads (lean_screen2.grade's rows), each called under its floor from the base arm's seed null."""
    problems, rows, sources = [], [], {}
    b = BASE_ARM[arm]
    d = dev or device()
    try:
        null, nsrc = null_of(b, screen, d)
    except SystemExit as e:
        problems.append(str(e))
        null, nsrc = {}, {}
    for sp in SPLITS:
        p, stamp, declared = decision_file(arm, sp, screen)
        sources[sp] = rel(p)
        try:
            got = _load_cmp(p, stamp, declared, arm_fit(arm, sp), base_fit(b, 0, sp, d), sp)
        except SystemExit as e:
            problems.append(str(e))
            continue
        for r in got:
            rows.append({"split": sp, "dataset": r["dataset"], "read": r["read"],
                         "primary": S2.primary(sp, r["dataset"]), "base": r["base"][0], "new": r["new"][0],
                         "rrf": r["rrf"][0], "delta": r["delta"][0], "lo": r["lo"][0], "hi": r["hi"][0],
                         "delta_fc5": r["delta"][1], "delta_hit1": r["delta"][2], "call": r["call"]})
    called = []
    for r in rows:
        try:
            called.append(SR.recall_row(r, null, sources[r["split"]]) if null else
                          {**r, "call_filed": r["call"], "covered": False, "floor": None, "near_floor": False})
        except SystemExit as e:
            problems.append(str(e))
    n_prim = sum(r["primary"] for r in called)
    if len(called) != len(SPLITS) * len(LG.EVAL_ORDER) or n_prim != 11:
        problems.append(f"{arm}: {len(called)} reads ({n_prim} primary), not 36 (11)")
    gains = [f"{r['split']} {r['dataset']}" for r in called if r["primary"] and r["call"] == "GAIN"]
    all_gains = [f"{r['split']} {r['dataset']}" for r in called if r["call"] == "GAIN"]
    losses = [f"{r['split']} {r['dataset']}" for r in called if r["call"] == "LOSS"]
    v = "INCOMPLETE" if problems else "ADOPT" if gains and not losses else "NOT_ADOPTED"
    return {"arm": arm, "base_arm": b, "verdict": v, "problems": problems, "primary_gains": gains,
            "gains": all_gains, "losses": losses, "rows": called, "sources": sources, "null": nsrc,
            "rule": SR.RULE, "floor_min": S.FLOOR}


NEXT = {"ADOPT": "the lean pair becomes (zrckpw, zspkw); next, each in its own file: KT_W's cold serving cost (P4c's "
                 "measure, on W2_S's walk), Claim 1's intervals for the new pair, and zg6w's own full run against zg6 "
                 "under zg6's seed null if its readings here show a gain",
        "NOT_ADOPTED_LOSS": "a per-question gate on the KT_W head, its weight per question from inputs computed before "
                            "any row is scored, in its own file (SCREENS 3b's back-off form)",
        "NOT_ADOPTED_NO_GAIN": "KT_W stays out of the lean pair; W2_S's signal goes on through the link-precision "
                               "features (the hyperlink-gap plan's step 1), beside KT_W as inputs to both models, in its "
                               "own file",
        "INCOMPLETE": "the missing fits and compares run when credit allows (November's credit, or credit the user "
                      "adds); nothing else changes",
        "always": "the hyperlink-gap plan (link precision; a hyperlink scorer learned on 2wiki and applied unchanged "
                  "elsewhere; corpus aliases), each step in its own declared file, beside this run"}


def pair_verdict(grades):
    problems = [p for g in grades.values() for p in g["problems"]]
    losses = [f"{a} {x}" for a, g in grades.items() for x in g["losses"]]
    gains = [f"{a} {x}" for a, g in grades.items() for x in g["primary_gains"]]
    if problems:
        return "INCOMPLETE", NEXT["INCOMPLETE"], problems, gains, losses
    if gains and not losses:
        return "ADOPT", NEXT["ADOPT"], problems, gains, losses
    return "NOT_ADOPTED", NEXT["NOT_ADOPTED_LOSS" if losses else "NOT_ADOPTED_NO_GAIN"], problems, gains, losses


# ── beside, deciding nothing (section 5) ─────────────────────────────────────


def _rows_or_none(p, stamp, declared, new, base, split, one=True):
    try:
        if one:
            return {r["dataset"]: r for r in _load_cmp(p, stamp, declared, new, base, split)}
        rec = json.loads(Path(p).read_text(encoding="utf-8"))
        return {r["dataset"]: r for r in rec["by_base"][rec["bases"][0]]["rows"]}
    except (SystemExit, OSError, ValueError, KeyError):
        return None


def beside(grades, screen=None, dev=None):
    screen = Path(screen or SCREEN)
    d = dev or device()
    g6, share = {}, []
    mlp = {(r["split"], r["dataset"]): r for r in grades["zrckpw"]["rows"]}
    for sp in NEW + OLD:
        rows = _rows_or_none(screen / f"scr-u2hf-g6-zg6w{SUF[sp]}.json", "u2hf", DECLARED,
                             fit_name("zg6w", "zg6w", 0, sp), base_fit("zg6", 0, sp), sp)
        if rows is None:
            continue
        g6[sp] = {ds: {"delta": r["delta"], "lo": r["lo"][0], "hi": r["hi"][0], "zg6w_r5": r["new"][0],
                       "zg6_r5": r["base"][0], "rrf": r["rrf"][0]} for ds, r in rows.items()}
        for ds, r in rows.items():
            m = mlp.get((sp, ds))
            gnn = r["new"][0] - r["rrf"][0]
            if m is None or gnn <= ZH.G6.SHARE_MIN:
                continue
            share.append({"split": sp, "dataset": ds, "mlp_gain": round(m["new"] - m["rrf"], 6),
                          "gnn_gain": round(gnn, 6), "share": round((m["new"] - m["rrf"]) / gnn, 4),
                          "across_devices": sp in OLD})
    probe = None
    if d in GPUS:
        other = next(g for g in GPUS if g != d)
        rows = _rows_or_none(screen / "scr-u2hf-probe.json", "u2hf", DECLARED, PROBE[other], PROBE[d], "J5", one=False)
        if rows:
            probe = {"other": other, "D": d, "delta_r5": {ds: r["delta"][0] for ds, r in rows.items()}}
    return {"zg6w_vs_zg6": g6, "mlp_share": share, "device_noise": probe}


# ── the verdict ──────────────────────────────────────────────────────────────


def fmt(x, signed=True):
    return "—" if x is None else (f"{x:+.4f}" if signed else f"{x:.4f}")


def grade_md(g):
    md = [f"# Full run of {g['arm']} (KT_W beside KT) against {g['base_arm']}: {g['verdict']}", "",
          f"R@5 of {g['arm']}'s p@swa minus {g['base_arm']}'s seed-0 p@swa of the same split on U2d's base build, on the "
          "six s1eval carves (95% question bootstrap). Each read is called under its floor from "
          f"{g['base_arm']}'s own seed null on that split (docs/U2H_FULL_RUN.md, section 4). Bold: the eleven primary "
          "reads. zs: zero-shot.", "", "| split | " + " | ".join(LG.EVAL_ORDER) + " |",
          "|---|" + "---|" * len(LG.EVAL_ORDER)]
    for sp in SPLITS:
        cells = []
        for ds in LG.EVAL_ORDER:
            r = next((x for x in g["rows"] if x["split"] == sp and x["dataset"] == ds), None)
            if r is None:
                cells.append("—")
                continue
            c = f"{r['delta']:+.4f} {r['call']}{' zs' if r['read'] == 'zero-shot' else ''}"
            if r.get("floor") is not None:
                c += f" (floor {r['floor']:.4f})"
            cells.append(f"**{c}**" if r["primary"] else c)
        md.append(f"| {sp} | " + " | ".join(cells) + " |")
    md += ["", f"Primary GAINs: {', '.join(g['primary_gains']) or 'none'}. LOSSes: {', '.join(g['losses']) or 'none'}. "
               f"GAINs in all: {len(g['gains'])} of {len(g['rows'])}."]
    if g["problems"]:
        md += ["", "Problems:"] + [f"- {p}" for p in g["problems"]]
    return md


def decide(screen=None, out=None, dev=None):
    d = dev or device()
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)
    grades = {arm: grade(arm, screen, d) for arm in ARMS_H}
    v, nxt, problems, gains, losses = pair_verdict(grades)
    bs = beside(grades, screen, d)
    for arm, g in grades.items():
        LC.write_json(out / f"grade-{arm}.json", {**g, "declared_in": DECLARED, "stage": "U2hf",
                                                  "zu2hf_sha256": sha_src(__file__)})
        (out / f"grade-{arm}.md").write_text("\n".join(grade_md(g)) + "\n", encoding="utf-8")
    rec = {"declared_in": DECLARED, "stage": "U2hf", "D": d, "verdict": v, "primary_gains": gains, "losses": losses,
           "problems": problems, "arms": {a: g["verdict"] for a, g in grades.items()}, "next": [nxt, NEXT["always"]],
           "beside": bs, "zu2hf_sha256": sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    LC.write_json(out / "verdict.json", rec)
    md = [f"# U2h's full run: {v}", "",
          f"The lean pair with KT_W (zrckpw, zspkw) against (zrckp, zspk) on U2d's base build, six splits per arm; the new "
          f"splits on {d}, L-musique and L-hotpotqa as filed on L40S (docs/U2H_FULL_RUN.md).", "",
          f"- Primary GAINs: {', '.join(gains) or 'none'}", f"- LOSSes: {', '.join(losses) or 'none'}",
          f"- Each arm's own verdict (beside): " + ", ".join(f"{a} {g['verdict']}" for a, g in grades.items()),
          f"- Next: {nxt}", f"- Always: {NEXT['always']}"]
    if problems:
        md += ["", "Problems:"] + [f"- {p}" for p in problems]
    if bs["zg6w_vs_zg6"]:
        md += ["", "## Beside: zg6w against zg6 (delta R@5)", "", "| split | " + " | ".join(LG.EVAL_ORDER) + " |",
               "|---|" + "---|" * len(LG.EVAL_ORDER)]
        for sp, t in bs["zg6w_vs_zg6"].items():
            md.append(f"| {sp} | " + " | ".join(fmt(t[ds]["delta"][0]) if ds in t else "—" for ds in LG.EVAL_ORDER)
                      + " |")
    if bs["mlp_share"]:
        md += ["", "## Beside: zrckpw's share of zg6w's R@5 gain over rrf", "",
               "| split | dataset | MLP gain | GNN gain | share |", "|---|---|---:|---:|---:|"]
        for s in bs["mlp_share"]:
            md.append(f"| {s['split']}{' (across devices)' if s['across_devices'] else ''} | {s['dataset']} | "
                      f"{s['mlp_gain']:+.4f} | {s['gnn_gain']:+.4f} | {s['share']:.2f} |")
    if bs["device_noise"]:
        dn = bs["device_noise"]
        md += ["", f"Device noise (zrckp J5 seed 0, {dn['other']} minus {dn['D']}): "
               + ", ".join(f"{ds} {x:+.4f}" for ds, x in dn["delta_r5"].items())]
    (out / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zu2hf decide: {v}; primary GAINs {gains or 'none'}; LOSSes {losses or 'none'}"
        + (f"; problems {problems}" if problems else ""))
    return rec


def plan(dev=None):
    try:
        d = dev or device()
    except SystemExit:
        d = None
    for g, names in GROUPS.items():
        print(f"{g} ({WHERE[g]}, {len(names)}): {','.join(names)}")
    print(f"D = {d or 'not filed yet (section 3)'}")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _fake_cmp(path, stamp, declared, new, base, split, deltas, base_r5=0.5, one=True):
    rows = []
    for ds in LG.EVAL_ORDER:
        x = deltas.get(ds, 0.0)
        call = "GAIN" if x >= S.FLOOR else "LOSS" if x <= -S.FLOOR else "WITHIN"
        rows.append({"dataset": ds, "read": "in-domain" if ds in LG.FITS[split] else "zero-shot",
                     "base": [base_r5, 0.3, 0.4], "new": [base_r5 + x, 0.3, 0.4], "rrf": [base_r5 - 0.1, 0.2, 0.3],
                     "delta": [x, 0.0, 0.0], "lo": [x - 0.002, 0.0, 0.0], "hi": [x + 0.002, 0.0, 0.0], "call": call})
    b = f"outputs/screen/fits/{base}"
    rec = {"new": f"outputs/screen/fits/{new}", "bases": [b], "candidate": "p@swa", "carve": "s1eval",
           "trained_on": list(LG.FITS[split]), "by_base": {b: {"rows": rows}},
           stamp: {"declared_in": declared, "one_device": one}}
    Path(path).write_text(json.dumps(rec), encoding="utf-8")


def _fake_run(td, d, arm_deltas=None, null_d=None, drop=(), wrong_base=(), two_devices=()):
    """Every compare the decision reads, in td: the arms' decision compares (U2h's on the screen splits) and the new
    splits' null; U2d's screen-split null comes from a patched zu2d.u2d_null."""
    arm_deltas = arm_deltas or {}
    null_d = null_d or {}
    for arm in ARMS_H:
        b = BASE_ARM[arm]
        for sp in SPLITS:
            p, stamp, declared = decision_file(arm, sp, td)
            if (arm, sp) in drop:
                continue
            base = "scr-elsewhere" if (arm, sp) in wrong_base else base_fit(b, 0, sp, d)
            _fake_cmp(p, stamp, declared, arm_fit(arm, sp), base, sp, arm_deltas.get((arm, sp), {}),
                      one=(arm, sp) not in two_devices)
        for sp in NEW:
            for sd in SR.SEEDS:
                _fake_cmp(td / f"scr-u2hf-null-{b}-s{sd}{SUF[sp]}.json", "u2hf", DECLARED, base_fit(b, sd, sp, d),
                          base_fit(b, 0, sp, d), sp, null_d.get((b, sp, sd), {}))


def _toy_zg6w(tmp):
    """zg6's selftest's toy carve (metaqa toy chains, links and KT), a batch, and a random KT_W beside it."""
    rng = np.random.default_rng(31)
    RM.toy_roots(tmp, rng, chunks=(12, 12))
    RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
             rel_dir=tmp / "rel")
    ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")

    class ToyIds(ZM.ToyZ):
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
    feats, nq, bz, _gold = c.batch(qs, blocks)
    feats[KTW_KEY] = torch.from_numpy(rng.random((nq.numel(), NKT)).astype(np.float32))
    return c, blocks, feats, nq, bz, qs.size


def selftest():
    import shutil
    import tempfile
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the fits, groups and compares
    assert len(set(SUF.values())) == 6 and len(FITS) == 46 and len(GROUPS["null"]) == 16
    assert [len(GROUPS[g]) for g in ("probe", "arms", "base", "fallback", "null", "zg6w", "zg6")] == [2, 8, 7, 1, 16, 6, 6]
    for d in GPUS:
        cm = compares(d)
        assert len(cm) == 8 + 16 + 6 + 1 and cm["scr-u2hf-probe"][2] is False
        assert base_fit("zrckp", 0, "J5", d) == PROBE[d] and cm["scr-u2hf-dh-zrckpw-j5"][1] == [PROBE[d]]
        used = {n for new, bs, _ in cm.values() for n in [new, *bs]}
        assert used <= set(FITS) and set(FITS) - used == set(GROUPS["fallback"]), set(FITS) - used
    cm = compares("L40S")
    assert len(cm) == 30 and base_fit("zrckp", 0, "J5", "L40S") == "scr-u2hf-base-zrckp-j5"
    used = {n for new, bs, _ in cm.values() for n in [new, *bs]}
    assert used <= set(FITS) and set(FITS) - used == set(PROBE.values()), set(FITS) - used
    assert base_fit("zspk", 2, "L-hotpotqa", "L4") == Z.fit_name("base", "zspk", 2, "L-hotpotqa")
    assert arm_fit("zspkw", "L-musique") == ZH.fit_name("zspkw", "L-musique") == "scr-u2h-zspkw"
    assert decision_file("zrckpw", "L-hotpotqa")[0].name == "scr-u2h-dh-zrckpw-hp.json"
    assert sum(S2.primary(sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER) == 11
    # 2. the GPU names and the rule
    assert gpu_is("NVIDIA L4", "L4") and not gpu_is("NVIDIA L40S", "L4") and gpu_is("NVIDIA L40S", "L40S")
    assert gpu_is("Tesla T4", "T4") and not gpu_is("NVIDIA L4", "T4") and not gpu_is(None, "L4")
    assert choose({"L4": 1000.0, "T4": 2000.0})[0] == "L4" and choose({"L4": 1000.0, "T4": 1200.0})[0] == "T4"
    lo = 1000.0 * hourly("L4") / hourly("T4")
    assert choose({"L4": 1000.0, "T4": lo * 0.999})[0] == "L4"                  # within 1%: L4
    assert choose({"L4": 1000.0, "T4": lo * 0.95})[0] == "T4"
    assert choose({"L4": None, "T4": 900.0})[0] == "T4" and choose({"L4": None, "T4": None}) == ("L40S", {})
    assert abs(hourly("L4") - 1.116) < 1e-9 and abs(hourly("T4") - 0.906) < 1e-9
    tmp = Path(tempfile.mkdtemp(prefix="zu2hf_"))
    saved = {k: globals()[k] for k in ("DEVICE",)}
    saved_null = Z.u2d_null
    try:
        # 3. the device record and where fits may run
        globals()["DEVICE"] = tmp / "device.json"
        try:
            device()
            raise AssertionError("a missing device record was read")
        except SystemExit:
            pass
        LC.write_json(DEVICE, {"declared_in": DECLARED, "D": "T4"})
        assert device() == "T4" and gpu_for(PROBE["L4"]) == "L4" and gpu_for("scr-u2hf-zrckpw-j5") == "T4"
        env = {"where": "modal", "image": IMAGE_TAG, "gpu": "Tesla T4"}
        where_ok("modal", env, "scr-u2hf-zrckpw-j5", "train")
        where_ok("modal", {**env, "gpu": "NVIDIA L4"}, PROBE["L4"], "train")
        for bad in ({**env, "gpu": "NVIDIA L4"}, {**env, "image": "other"}, {**env, "where": "elsewhere"}):
            try:
                where_ok("modal", bad, "scr-u2hf-zrckpw-j5", "train")
                raise AssertionError(f"{bad} was accepted")
            except SystemExit:
                pass
        try:
            where_ok("laptop", env, "scr-u2hf-zrckpw-j5", "train")
            raise AssertionError("the laptop was accepted")
        except SystemExit:
            pass
        # 4. zg6w: zg6's construction and draws, 2,048 zero weights more, zg6's score bit for bit at the start (eval and
        #    training mode), gradients reaching every KT_W weight, and no score without KT_W
        assert S.ARMS["zg6w"] == (ZG6W, ZH.KTWLinkCarveBase) and S.ARMS["zg6"] == G6.ARMS["zg6"]
        assert ZG6W.__mro__[1] is G6.ZG6 and not ZG6W.TWIN and N_KW == 2048
        c, blocks, feats, nq, bz, B = _toy_zg6w(tmp / "toy")
        keep = torch.ones(B, len(blocks))
        nets, after = {}, {}
        for k, cl in (("zg6", G6.ZG6), ("zg6w", ZG6W)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, c.widths, 16, dropout=0.1, seed=0)
            after[k] = torch.rand(4)
        assert torch.equal(after["zg6"], after["zg6w"])
        s6, sw = nets["zg6"].state_dict(), nets["zg6w"].state_dict()
        assert sorted(set(sw) - set(s6)) == sorted((KW_IN, *KW_LAYER)) and set(s6) <= set(sw)
        assert all(torch.equal(s6[k], sw[k]) for k in s6) and all(not sw[k].any() for k in (KW_IN, *KW_LAYER))
        assert sum(sw[k].numel() for k in (KW_IN, *KW_LAYER)) == N_KW
        with torch.no_grad():
            for x in nets.values():
                x.eval()
            assert torch.equal(nets["zg6"](feats, keep, nq, B, bz), nets["zg6w"](feats, keep, nq, B, bz))
        ys = {}
        for k, m in nets.items():
            m.train()
            torch.manual_seed(5)
            ys[k] = m(feats, keep, nq, B, bz)
        assert torch.equal(ys["zg6"], ys["zg6w"])
        ys["zg6w"].pow(2).sum().backward()
        assert all(getattr(nets["zg6w"], k).grad is not None and getattr(nets["zg6w"], k).grad.abs().sum() > 0
                   for k in (KW_IN, *KW_LAYER))
        no_w = {k: v for k, v in feats.items() if k != KTW_KEY}
        try:
            nets["zg6w"](no_w, keep, nq, B, bz)
            raise AssertionError("zg6w scored without KT_W")
        except RuntimeError:
            pass
        # 5. the grade and the pair's verdict on fake compares, U2d's screen-split null patched in
        def fake_null(b, files=None, big=None):
            null = {sp: {ds: {"read": "x", "seed0": 0.5, "d1": 0.001, "d2": -0.001, "spread": 0.001,
                              "floor": S.FLOOR} for ds in LG.EVAL_ORDER} for sp in OLD}
            return null, {f"U2d {b} {sp}": "fake" for sp in OLD}

        Z.u2d_null = fake_null
        for d in ("L4", "T4", "L40S"):
            LC.write_json(DEVICE, {"declared_in": DECLARED, "D": d})
            sc = tmp / f"screen-{d}"
            sc.mkdir()
            gain = {("zrckpw", "J5"): {"2wiki": 0.012}}
            _fake_run(sc, d, gain)
            v = decide(sc, sc / "out")
            assert v["verdict"] == "ADOPT" and v["primary_gains"] == ["zrckpw J5 2wiki"], v
            assert v["arms"] == {"zrckpw": "ADOPT", "zspkw": "NOT_ADOPTED"}
            g = json.loads((sc / "out" / "grade-zrckpw.json").read_text(encoding="utf-8"))
            assert len(g["rows"]) == 36 and all(r["covered"] for r in g["rows"])
        LC.write_json(DEVICE, {"declared_in": DECLARED, "D": "L4"})
        cases = {
            "loss": (dict(arm_deltas={("zrckpw", "J5"): {"2wiki": 0.012}, ("zspkw", "L-squad"): {"metaqa": -0.009}}),
                     "NOT_ADOPTED", NEXT["NOT_ADOPTED_LOSS"]),
            "no gain": (dict(arm_deltas={("zrckpw", "L-squad"): {"metaqa": 0.012}}), "NOT_ADOPTED",
                        NEXT["NOT_ADOPTED_NO_GAIN"]),
            "raised floor": (dict(arm_deltas={("zrckpw", "J5"): {"2wiki": 0.012}},
                                  null_d={("zrckp", "J5", 1): {"2wiki": 0.009}, ("zrckp", "J5", 2): {"2wiki": -0.007}}),
                             "NOT_ADOPTED", NEXT["NOT_ADOPTED_NO_GAIN"]),
            "missing": (dict(arm_deltas={("zrckpw", "J5"): {"2wiki": 0.012}}, drop=(("zspkw", "L-2wiki"),)),
                        "INCOMPLETE", NEXT["INCOMPLETE"]),
            "wrong base": (dict(arm_deltas={("zrckpw", "J5"): {"2wiki": 0.012}}, wrong_base=(("zspkw", "J5"),)),
                           "INCOMPLETE", NEXT["INCOMPLETE"]),
            "two devices": (dict(arm_deltas={("zrckpw", "J5"): {"2wiki": 0.012}}, two_devices=(("zrckpw", "L-squad"),)),
                            "INCOMPLETE", NEXT["INCOMPLETE"]),
        }
        for k, (kw, want, nxt) in cases.items():
            sc = tmp / f"case-{k.replace(' ', '_')}"
            sc.mkdir()
            _fake_run(sc, "L4", **kw)
            v = decide(sc, sc / "out")
            assert v["verdict"] == want and v["next"][0] == nxt, (k, v["verdict"], v["problems"])
        g = json.loads((tmp / "case-raised_floor" / "out" / "grade-zrckpw.json").read_text(encoding="utf-8"))
        r = next(x for x in g["rows"] if x["split"] == "J5" and x["dataset"] == "2wiki")
        assert r["call_filed"] == "GAIN" and r["call"] == "WITHIN" and abs(r["floor"] - SR.floor_of(0.009, -0.007)) < 1e-6
        assert abs(SR.floor_of(0.009, -0.007) - 0.016125) < 1e-6, r
    finally:
        Z.u2d_null = saved_null
        globals().update(saved)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"zu2hf selftest: OK ({time.time() - t0:.1f}s)", flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--selftest"]:
        return selftest()
    cmd, rest = (argv[0], argv[1:]) if argv else (None, [])
    if cmd in ("train", "read"):
        ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        ap.add_argument("--split", default="J5", choices=SPLITS)
        ap.add_argument("--modal", action="store_true")
        k, rest = ap.parse_known_args(rest)
        where = "modal" if k.modal else "laptop"
        return train(["train"] + rest, k.split, where) if cmd == "train" else read(rest, where)
    if cmd == "plan" and not rest:
        return plan()
    if cmd == "device" and not rest:
        with Z.at_root():
            device_cmd()
        return 0
    if cmd == "compare-all" and not rest:
        with Z.at_root():
            return compare_all(screen=Path("outputs/screen"))
    if cmd == "decide" and not rest:
        with Z.at_root():
            return 0 if decide(screen=Path("outputs/screen"))["verdict"] != "INCOMPLETE" else 1
    raise SystemExit("zu2hf: train, read, plan, device, compare-all, decide, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
