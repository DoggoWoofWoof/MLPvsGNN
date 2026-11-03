"""IRA's stage (docs/IRA_ITERATIVE_ALIGNMENT.md): IsoNet++'s iteratively refined alignment (Ramachandran et al., NeurIPS 2024)
on the GNN of record. RTR's two rounds of zg6r's three layers (zrtr.rounds_stack) with, between them, a plan of SLOTS hop slots
over the question's rows: Sinkhorn with a dustbin on logits from round one's clipped z-score and last states. The plan enters
round two in each row's start (the slots it holds, as learned vectors, so its neighbours' messages carry them) and in the
question's context (the slots' mass, mixed with RTR's context by a learned weight that starts at 0). At the start IRA's score
is RTR's. On PKO's roots, on the host's GPU: the screen at seed 0 on both WQ splits against the base's seed-0 fit, on the twelve
reads and on musique's 3-hop and 4-hop slices, under DEP's rule (zdep's functions); the full run (seeds 1 and 2) when the screen
asks for it.

The base is the GNN of record after ISO, read from ISO's filed verdict as DEP reads it (zdep.iso_call, zdep.base_fit). An IRA
fit runs under RTR's condition, when DEP's filed calls keep three layers (zrtr.dep_call). zml1s.py, zg6r.py, ziso.py, zdep.py
and zrtr.py are imported and not edited. For the length of an IRA fit, zg6r's g6r_stack is IRA's stack and ZG6R's __init__
adds IRA's parameters after zg6r's own; zml1s runs with IRA's names, stamp key, stage and declaration, and on REPLACE with
ISO's own-start patches. Every patched name is put back afterwards, also on a failure.

    python outputs/mp_unified/zira.py fit --name scr-ira-k4-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zira.py compare-all                    (each IRA fit read against the base's fit of its seed)
    python outputs/mp_unified/zira.py calls                          (outputs/ira/calls.json and .md)
    python outputs/mp_unified/zira.py --selftest
"""
import argparse
import contextlib
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402
import ziso as ISO  # noqa: E402
import zdep as DEP  # noqa: E402  (the base after ISO, the slices and both rules; DEP's fits are not touched)
import zrtr as RTR  # noqa: E402  (the rounds IRA starts from and RTR's condition; RTR's fits are not touched)
import ml1start as MS  # noqa: E402  (on ISO's KEEP, ML1's zspk start read under ML1's names: amendment A1)
import readswa as RS  # noqa: E402  (a fit's read keeps p@swa only: amendment A2)

DECLARED = "docs/IRA_ITERATIVE_ALIGNMENT.md"
STAGE = "IRA"
KEY = "ira"                                     # the stamp's key in an IRA fit's screen.json and read.json and IRA's compares
OUT = "outputs/ira"
ARM = DEP.ARM                                   # zrm's arm: the model is zg6r's with RTR's rounds and the plan
TAG = "k4"
ROUNDS = RTR.ROUNDS                             # 2
SLOTS = 4                                       # one per hop of musique's longest questions
SINK_ITERS = 50                                 # each a column step, then a row step; ten hold the slots only to ira_bz ~4
SHARP = 25.0                                    # the self-test's sharp ira_bz: a softmax puts 0.993 on the top row
A_SEED_OFF = 7919                               # ira_a's own generator: the fit's seed plus this
IRA_NAMES = ("ira_e", "ira_E", "ira_a", "ira_bz", "ira_b0", "ira_lam")
RTR_CALLS = Path(RTR.OUT) / "calls.json"
CODE_SKIP = DEP.CODE_SKIP                       # stage modules, not the trainer's code
R5_TOL = DEP.R5_TOL
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    DEP.ADOPT_FOR_FULL_RUN, DEP.NO_GAIN, DEP.NOT_ADOPTED, DEP.ADOPT, DEP.INCOMPLETE)


def fit_name(seed, split):
    return f"scr-ira-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-ira-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS) | set(RTR.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zira: {msg}", flush=True)


# ── when IRA runs, and its base ──────────────────────────────────────────────


def dep_call(base=ROOT):
    """RTR's condition (zrtr.dep_call): DEP's filed calls keep three layers. Returns which; anything else stops an IRA fit."""
    return RTR.dep_call(base)


def pairs(call):
    return {pair_name(sd, sp): (fit_name(sd, sp), DEP.base_fit(sd, sp, call)) for sd in RULE.SEEDS for sp in M1.SPLITS}


# ── the plan and the rounds ──────────────────────────────────────────────────


def n_ira(G):
    """IRA's parameters: ira_e (D), ira_E and ira_a (SLOTS x D each), ira_bz, ira_b0 and ira_lam (one each)."""
    return int(G.D + 2 * SLOTS * G.D + 3)


def seg_lse(torch, x, nq, B):
    """Per question and column, the log-sum-exp of x (N, C) over the question's rows: (B, C)."""
    C = x.shape[1]
    mx = torch.full((B, C), -float("inf"), dtype=x.dtype, device=x.device).scatter_reduce(
        0, nq.unsqueeze(1).expand(-1, C), x, reduce="amax", include_self=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx)).detach()
    den = torch.zeros(B, C, dtype=x.dtype, device=x.device).index_add_(0, nq, (x - mx[nq]).exp())
    return den.clamp_min(1e-30).log() + mx


def plan(G, m, z, h1, nq, B, full=False):
    """IRA's plan (N, SLOTS), or with the dustbin's column (N, SLOTS + 1) when full: Sinkhorn in log space, in float32, per
    question, on slot k's logits ira_bz z + ira_a[k] . h1 / sqrt(D) and the dustbin's ira_b0. Each row's mass over the slots
    and the dustbin is 1; each slot's mass over the question's n rows is t = min(1, n / (SLOTS + 1)); the dustbin's n - SLOTS t.
    SINK_ITERS iterations, each a column step and then a row step, so the rows hold exactly."""
    torch = G.torch
    N, dev = z.shape[0], z.device
    zf, hf = z.float(), h1.float()
    lg = torch.cat([m.ira_bz.float() * zf.unsqueeze(1) + hf @ m.ira_a.float().t() / math.sqrt(G.D),
                    m.ira_b0.float().view(1, 1).expand(N, 1)], 1)
    n = torch.zeros(B, dtype=torch.float32, device=dev).index_add_(0, nq, torch.ones(N, dtype=torch.float32, device=dev))
    t = (n / (SLOTS + 1)).clamp(max=1.0)
    tgt = torch.cat([t.unsqueeze(1).expand(B, SLOTS), (n - SLOTS * t).unsqueeze(1)], 1).clamp_min(1e-30).log()
    u = torch.zeros(N, dtype=torch.float32, device=dev)
    v = None
    for _i in range(SINK_ITERS):
        v = tgt - seg_lse(torch, lg + u.unsqueeze(1), nq, B)
        u = -torch.logsumexp(lg + v[nq], dim=1)
    p = (lg + u.unsqueeze(1) + v[nq]).exp()
    return p if full else p[:, :SLOTS]


def ira_stack(G, m, H, xb, lk, gd, nq, B, u01=None):
    """zg6r's score (N,) after RTR's two rounds with IRA's plan between them. Round one is zg6r.g6r_stack, op for op. Round two
    starts each row from h0 + z ira_e + P ira_E, z round one's clipped z-score over the question's rows and P the plan from z and
    round one's last states, and takes as the question's context the softmax of z (RTR's) moved toward the slots' mass by
    clip(ira_lam, 0, 1). The readout is zg6r's, on round two's start and end states. The family dropout's one draw (u01, in
    training) holds for both rounds."""
    torch, Fn = G.torch, G.Fn
    xb = xb.to(H.dtype)
    h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
    s0 = Fn.linear(h0, m.g6_ctx_w).squeeze(-1)
    z0 = G.LG.seg_zscore8D(s0.unsqueeze(1), nq, B).squeeze(1).clamp(-G.ZCLIP, G.ZCLIP)
    w = G.G2.seg_softmax_rows(z0, nq, B).unsqueeze(1)
    u, v, f, ci = lk["u"], lk["v"], lk["f"], gd["ci"]
    if u01 is not None:
        ke = G.G2.family_keep(f, nq[v], B, u01)
        u, v, f, ci = u[ke], v[ke], f[ke], ci[ke]
    te = torch.nonzero(ci >= 0).view(-1)
    codes = cte = None
    if te.numel():
        if gd["rel"] is None:
            raise RuntimeError("zg6r: typed edges in the batch without their relation vectors")
        codes = G.rel_codes(m, gd["rel"], gd["crel"], gd["cdir"], H.dtype)
        cte = ci[te]
    h = G.g6r_layers(m, h0, xb, w, u, v, f, nq, B, te, codes, cte)
    s = (Fn.linear(h, m.g6_out_w) + Fn.linear(h0, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
    z = G.LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1).clamp(-G.ZCLIP, G.ZCLIP)
    wr = G.G2.seg_softmax_rows(z, nq, B)
    p = plan(G, m, z, h, nq, B).to(h0.dtype)
    ms = p.sum(1)
    ws = ms / torch.zeros(B, dtype=ms.dtype, device=ms.device).index_add_(0, nq, ms).clamp_min(1e-30)[nq]
    w2 = (wr + m.ira_lam.clamp(0.0, 1.0) * (ws - wr)).unsqueeze(1)
    hr = h0 + z.unsqueeze(1) * m.ira_e + p @ m.ira_E
    h = G.g6r_layers(m, hr, xb, w2, u, v, f, nq, B, te, codes, cte)
    return (Fn.linear(h, m.g6_out_w) + Fn.linear(hr, m.g6_skip_w) + m.g6_out_b).squeeze(-1)


@contextlib.contextmanager
def on_model(G):
    """For the block: zg6r's g6r_stack is ira_stack (zg6r's forward calls it by name), and a ZG6R built gets IRA's parameters
    after zg6r's own __init__: ira_a from its own generator at the model's seed + A_SEED_OFF, the rest constants, so every
    other start and draw is zg6r's. Both are restored after, also on a failure."""
    orig_init = G.ZG6R.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        seed = int(k.get("seed", a[4] if len(a) > 4 else 0))
        g = G.torch.Generator().manual_seed(seed + A_SEED_OFF)
        self.ira_e = G.nn.Parameter(G.torch.zeros(G.D))
        self.ira_E = G.nn.Parameter(G.torch.zeros(SLOTS, G.D))
        self.ira_a = G.nn.Parameter(G.torch.randn(SLOTS, G.D, generator=g))
        self.ira_bz = G.nn.Parameter(G.torch.ones(1))
        self.ira_b0 = G.nn.Parameter(G.torch.zeros(1))
        self.ira_lam = G.nn.Parameter(G.torch.zeros(1))

    def stack(m, H, xb, lk, gd, nq, B, u01=None):
        return ira_stack(G, m, H, xb, lk, gd, nq, B, u01)

    with M1.WQ.patch(G, g6r_stack=stack), M1.WQ.patch(G.ZG6R, __init__=init):
        yield


def ira_record(G):
    return {"rounds": ROUNDS, "slots": SLOTS, "sinkhorn_iters": SINK_ITERS, "ira_parameters": n_ira(G),
            "parameters_h128": int(G.N_PARAMS + n_ira(G)), "layers": int(G.LAYERS)}


def ira_record_keys():
    return ("rounds", "slots", "sinkhorn_iters", "ira_parameters", "parameters_h128", "layers")


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """IRA's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def ira_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the IRA fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def stamp_with_plan(orig, call, dep, rec):
    """zml1s.stamp with the plan's record, the base, ISO's verdict and DEP's call."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        return orig(name, env, what, {**(extra or {}), **rec, "base": DEP.base_fit(sd, sp, call), "iso_verdict": call,
                                      "dep_call": dep})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with IRA's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    sha, rsha, dsha = M1.WQ.sha_src(__file__), M1.WQ.sha_src(RTR.__file__), M1.WQ.sha_src(DEP.__file__)
    code, ms = M1.code_shas, {**MS.code(), **RS.code()}
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=ira_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zira_sha256": sha, "zrtr_sha256": rsha,
                                                              "zdep_sha256": dsha, **ms}):
        yield


@contextlib.contextmanager
def on_fit(G, call, dep):
    """An IRA fit's context: IRA's names and stamp, the plan's model, the start (on REPLACE ISO's own: its load_encoder in
    zml1s, its encoder_used in zg6r; on KEEP ML1's zspk start read under ML1's names: ml1start's ml1_load in zml1s) and a
    read of p@swa alone (readswa's on_read in lean_gpu: amendment A2); restored after."""
    own = call == ISO.REPLACE
    with on_names(), M1.WQ.patch(M1, stamp=stamp_with_plan(M1.stamp, call, dep, ira_record(G)),
                                 load_encoder=ISO.own_load if own else MS.ml1_load), \
            (M1.WQ.patch(G, encoder_used=ISO.own_used) if own else contextlib.nullcontext()), on_model(G), RS.on_read():
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zira: {name} is not one of IRA's fits ({', '.join(FITS)})")
    dep = dep_call()
    call = DEP.iso_call()
    G, _U2N = M1.import_arm(ARM)
    if G.LAYERS != DEP.BASE_LAYERS:
        raise SystemExit(f"zira: zg6r holds {G.LAYERS} layers, not {DEP.BASE_LAYERS}")
    _a, sd, sp = FITS[name]
    log(f"{name}: {ROUNDS} rounds with {SLOTS} slots on {DEP.base_fit(sd, sp, call)}'s start (ISO {call}; DEP's {dep}); "
        f"{ira_record(G)}")
    with on_fit(G, call, dep):
        return M1.fit(name)


# ── the compares, the slices and the calls ───────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: IRA's, ISO's or ML1's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    return DEP.fit_block(name, what, base)


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zira: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each IRA fit that is read against the base's fit of its seed and split, through zwq's run_compare; a compare filed
    before is kept. The full run's fits are compared once they exist."""
    call = DEP.iso_call(base)
    done, ran = [], []
    for cname, (new, bse) in pairs(call).items():
        if not is_read(new, base):
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zira: {new} and {bse} ran on different devices: {sorted(dev)}")
        with ISO.with_fits(), on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zira: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "iso_verdict": call,
                    "device": json.loads(dev.pop()), "zira_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_ira(call, base=ROOT):
    """({(split, dataset): {seed: R@5}} of IRA's fits, the same of their bases) from IRA's compares."""
    new_r5, base_r5 = {}, {}
    for cname, (new, bse) in pairs(call).items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != DEP.CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def slices(call, new_r5, base3, base=ROOT):
    """({(split, slice): {seed: R@5}} of IRA's read fits, the same of the base's three seeds, problems), by zdep.slice_r5.
    Each file's all-question R@5 must be its compare's musique R@5."""
    new_s, base_s, bad = {}, {}, []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            for f, own, ref in ((fit_name(sd, sp), new_s, new_r5), (DEP.base_fit(sd, sp, call), base_s, base3)):
                if f in FITS and not is_read(f, base):
                    continue
                s = DEP.slice_r5(f, base)
                if s is None:
                    bad.append(f"{f}: no {DEP.SLICE_DS} read file")
                    continue
                want = ref.get((sp, DEP.SLICE_DS), {}).get(sd)
                if want is not None and abs(s["all"] - want) > R5_TOL:
                    bad.append(f"{f}: its read file gives {DEP.SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in DEP.SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return new_s, base_s, bad


def fit_problems(call, base=ROOT):
    """Why an IRA fit that is read is not the declared one: its rounds, slots, base and ISO verdict, its start, its device,
    and its trainer's code against the base's fit of its seed and split."""
    out = []
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = DEP.base_fit(sd, sp, call)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k in (("train", ks), ("read", kr)):
            got = (k.get("rounds"), k.get("slots"), k.get("layers"), k.get("base"), k.get("iso_verdict"))
            if got != (ROUNDS, SLOTS, DEP.BASE_LAYERS, bse, call):
                out.append(f"{name}: its {what} names {got[0]} rounds, {got[1]} slots, {got[2]} layers on {got[3]} "
                           f"(ISO {got[4]})")
            if not str(k.get("dep_call") or "").endswith((NO_GAIN, NOT_ADOPTED, RTR.REOPENED)):   # A5: or reopened
                out.append(f"{name}: its {what} ran under DEP's {k.get('dep_call')}")
        enc, benc = ks.get("encoder") or {}, kb.get("encoder") or {}
        if call == ISO.REPLACE:                     # zg6r-m's own draw: no zspk tensor copied
            ok = enc.get("start") == "own" and enc.get("copies") == 0
        else:                                       # zg6r's start: the zspk encoder its base fit copied
            ok = benc.get("fit") is not None and int(enc.get("copies") or 0) >= 1 and all(
                enc.get(x) == benc.get(x) for x in ("fit", "seed", "sha256"))
        if not ok:
            out.append(f"{name}: its encoder's start ({enc.get('start') or enc.get('fit')}, {enc.get('copies')} copies) "
                       f"is not {bse}'s (ISO {call})")
        if ks.get("device") != kb.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if k not in CODE_SKIP and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def plan_numbers(name, base=ROOT):
    """Beside (section 5): the fitted plan's numbers from the fit's models.pt (p@swa); None where the file is not here (the
    weights stay on the host)."""
    p = M1.screen_dir(base) / "fits" / name / "models.pt"
    if not p.is_file():
        return None
    import torch
    st = (torch.load(p, map_location="cpu", weights_only=False).get("states") or {}).get(DEP.CAND)
    if not st or any(k not in st for k in IRA_NAMES):
        return {"missing": sorted(k for k in IRA_NAMES if not st or k not in st)}
    lam = float(st["ira_lam"].float().view(-1)[0])
    return {"lam_raw": lam, "lam": min(1.0, max(0.0, lam)), "bz": float(st["ira_bz"].float().view(-1)[0]),
            "b0": float(st["ira_b0"].float().view(-1)[0]), "norm_E": float(st["ira_E"].float().norm()),
            "norm_e": float(st["ira_e"].float().norm()), "norm_a": float(st["ira_a"].float().norm())}


def next_form(verdict, plans):
    """Section 6's next form on NO_GAIN or NOT_ADOPTED: 'coverage' when every screen fit leaves lambda at 0 (the slots' context
    unused), else 'three_rounds'; None on any other verdict or without every screen fit's numbers."""
    if verdict not in (NO_GAIN, NOT_ADOPTED):
        return None
    lams = [(plans.get(fit_name(0, sp)) or {}).get("lam") for sp in M1.SPLITS]
    if any(x is None for x in lams):
        return None
    return "coverage" if all(x <= 0.0 for x in lams) else "three_rounds"


def rtr_beside(base=ROOT):
    """Beside (section 5): RTR's screen against the same base, when filed."""
    p = Path(base) / RTR_CALLS
    if not p.is_file():
        return None
    sc = M1.WQ.read_json(p).get("screen") or {}
    return {"verdict": sc.get("verdict"), "counts": sc.get("counts"),
            "reads": [{k: r.get(k) for k in ("split", "read", "new", "delta", "call")} for r in sc.get("reads") or []]}


def calls(base=ROOT):
    call = DEP.iso_call(base)
    new_r5, b0 = r5_ira(call, base)
    base3 = DEP.base_seeds(call, base)
    bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in IRA's compare, {base3.get(rd, {}).get(sd)} in its own stage's"
           for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
           if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > R5_TOL]
    new_s, base_s, bad_s = slices(call, new_r5, base3, base)
    bad += bad_s
    new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
    screen = DEP.decide_screen(new_all, base_all)
    full_ready = all(is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
    full = DEP.decide_full(new_r5, base3, new_s, base_s) if full_ready else None
    bad += fit_problems(call, base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    dp = Path(base) / RTR.DEP_CALLS
    dep = M1.WQ.read_json(dp) if dp.is_file() else {}
    plans = {f: plan_numbers(f, base) for f in FITS if is_read(f, base)}
    last = (full or {}).get("verdict") if full is not None else screen["verdict"]
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{ARM} with {ROUNDS} rounds and {SLOTS} aligned slots",
           "iso_verdict": call, "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)",
           "dep": {"screen": (dep.get("screen") or {}).get("verdict"), "full_run": (dep.get("full_run") or {}).get("verdict")},
           "screen": screen, "full_run": full, "problems": bad,
           "parameters": {f: {k: fit_block(f, "screen", base).get(k) for k in ira_record_keys()}
                          for f in FITS if is_read(f, base)},
           "beside": {"rtr_screen": rtr_beside(base), "plan": plans},
           "next_form_by_rule": next_form(last, plans),
           "zira_sha256": M1.WQ.sha_src(__file__), "zrtr_sha256": M1.WQ.sha_src(RTR.__file__),
           "zdep_sha256": M1.WQ.sha_src(DEP.__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# IRA: {rec['arm']} against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    if screen["reads"]:
        md += ["| split | read | base s0, s1, s2 | IRA s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
        md += [f"| {r['split']} | {r['read']} | {', '.join(f'{x:.4f}' for x in r['base'])} | {r['new']:.4f} | "
               f"{r['delta']:+.4f} | {r['floor']:.4f} | {r['call']} |" for r in screen["reads"]]
    if full is not None:
        md += ["", f"## Full run: {full['verdict']}", ""]
        if full["reads"]:
            md += ["| split | read | d0 | d1 | d2 | floor | call |", "|---|---|---:|---:|---:|---:|---|"]
            for r in full["reads"] + full["slices"]:
                d = r["delta"]
                md.append(f"| {r['split']} | {r.get('dataset', r.get('read'))} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | "
                          f"{r['floor']:.4f} | {r['call']} |")
    md += ["", "## Beside (deciding nothing)", ""]
    rb = rec["beside"]["rtr_screen"]
    md.append(f"RTR's screen on the same base: {rb['verdict']} {json.dumps(rb.get('counts') or {})}" if rb else
              "RTR's screen: not filed")
    if plans:
        md += ["", "| fit | lambda | ira_bz | ira_b0 | norm E | norm e | norm a |", "|---|---:|---:|---:|---:|---:|---:|"]
        for f, pn in plans.items():
            if pn and "lam" in pn:
                md.append(f"| {f} | {pn['lam']:.4f} | {pn['bz']:.4f} | {pn['b0']:.4f} | {pn['norm_E']:.4f} | "
                          f"{pn['norm_e']:.4f} | {pn['norm_a']:.4f} |")
            else:
                md.append(f"| {f} | {pn} | | | | | |")
    md += ["", f"next form by section 6's rule: {rec['next_form_by_rule']}"]
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; next form {rec['next_form_by_rule']}; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_plan(G):
    """The plan on three toy questions (3, 9 and 40 rows): its marginals, the questions' independence, the rows' permutation,
    and with a sharp z (ira_bz SHARP) and the slots alike, at least 0.8 of a slot on each of the four top rows where a softmax
    puts 0.99 on the top row and under 0.01 on the next."""
    import types
    torch = G.torch
    gen = torch.Generator().manual_seed(5)
    ns = (3, 9, 40)
    nq = torch.cat([torch.full((n,), q, dtype=torch.long) for q, n in enumerate(ns)])
    B, N = len(ns), int(sum(ns))
    z = torch.randn(N, generator=gen) * 1.5
    h1 = torch.randn(N, G.D, generator=gen)
    m = types.SimpleNamespace(ira_bz=torch.ones(1), ira_b0=torch.zeros(1), ira_a=torch.randn(SLOTS, G.D, generator=gen))
    out = {}
    pf = plan(G, m, z, h1, nq, B, full=True)
    out["rows_max_abs"] = float((pf.sum(1) - 1).abs().max())
    col = torch.zeros(B, SLOTS).index_add_(0, nq, pf[:, :SLOTS])
    t = torch.tensor([min(1.0, n / (SLOTS + 1)) for n in ns])
    out["slots_max_abs"] = float((col - t.unsqueeze(1)).abs().max())
    out["nonneg"] = bool((pf >= 0).all())
    sub = nq < 2                                                    # questions 0 and 1 alone
    ps = plan(G, m, z[sub], h1[sub], nq[sub], 2)
    out["independent_max_abs"] = float((ps - pf[sub][:, :SLOTS]).abs().max())
    perm = torch.randperm(N, generator=gen)
    pp = plan(G, m, z[perm], h1[perm], nq[perm], B)
    out["permuted_max_abs"] = float((pp - pf[perm][:, :SLOTS]).abs().max())
    zs = z.clone()
    q2 = torch.nonzero(nq == 2).view(-1)
    zs[q2] = torch.linspace(4.0, -3.8, q2.numel())                  # q2's rows in falling order, 0.2 apart
    sharp = types.SimpleNamespace(ira_bz=torch.full((1,), SHARP), ira_b0=torch.zeros(1), ira_a=torch.zeros(SLOTS, G.D))
    mass = plan(G, sharp, zs, h1, nq, B).sum(1)[q2]
    out["sharp_top4_min"] = float(mass[:4].min())
    out["sharp_rest_sum"] = float(mass[4:].sum())
    sm = torch.softmax(SHARP * zs[q2], 0)
    out["softmax_top1"], out["softmax_second"] = float(sm[0]), float(sm[1])
    return out


def toy_rounds(G):
    """The rounds on zg6r's own toy carve (RTR's toy, after zdep's: zg6r's self-test steps 2 and 3), zg6r's parameters drawn at
    random: at the start IRA's score is RTR's two rounds bit for bit (RTR's e set to IRA's, typed edges in); zg6r's forward is
    IRA's stack; the score moves with ira_E and with lambda; a training forward (checkpointed) and backward reaches every IRA
    parameter; the parameter count; the state dict's strict load with and without the patch; the patch's restoring."""
    import tempfile
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    with tempfile.TemporaryDirectory(prefix="zira_") as td:
        tmp = Path(td)
        rng = np.random.default_rng(31)
        G.RM.toy_roots(tmp, rng, chunks=(12, 12))
        G.RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                   rel_dir=tmp / "rel")
        G.ZL.build("metaqa", "toy", out_root=tmp / "lk", cache_root=tmp / "cache", look_root=tmp / "look")
        kw = dict(look_root=tmp / "look", cache_root=tmp / "cache", links_root=tmp / "lk", ch_root=tmp / "ch")

        class ToyIds(G6.ZM.ToyZ):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.ids = [f"q{i}" for i in range(self.rows)]

        with G.U2N.extra_carves(("metaqa", "toy")):
            G.U2N.build_typed("metaqa", "toy", tmp / "look", tmp / "typed", **kw)
            plain = G.ZL.link_carve(G.RM.chain_carve(ToyIds, tmp / "ch", tmp / "rel"), tmp / "lk")
            c0 = plain("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            G.Z3.toy_kt(tmp / "kt", c0, rng, cache_root=str(tmp / "cache"))
            cls = G.g6r_carve(G.Z3.kt_carve(plain, tmp / "kt"), tmp / "typed")
            ce = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(ce.rows)[::-1][:7], np.arange(ce.rows)[:5]].astype(np.int64)
        B = qs.size
        feats, nq, bz, _gold = ce.batch(qs, blocks)
        keep = torch.ones(B, len(blocks))
        gd, lk = feats[G.G6R_KEY], feats[G.LK_KEY]
        out["typed_edges"] = int((gd["ci"] >= 0).sum())
        out["edges"] = int(lk["u"].numel())
        orig_stack, orig_init = G.g6r_stack, G.ZG6R.__init__
        names = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
        gen = torch.Generator().manual_seed(3)
        with on_model(G):
            torch.manual_seed(0)
            m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
            want_a = torch.randn(SLOTS, G.D, generator=torch.Generator().manual_seed(0 + A_SEED_OFF))
            out["starts"] = bool((m.ira_e == 0).all() and (m.ira_E == 0).all() and (m.ira_lam == 0).all()
                                 and (m.ira_bz == 1).all() and (m.ira_b0 == 0).all() and torch.equal(m.ira_a.detach(), want_a))
            out["parameters_h16"] = sum(int(getattr(m, k).numel()) for k in names + IRA_NAMES)
            out["want_h16"] = G6.n_params(16) + G.N_G6R + n_ira(G)
            with torch.no_grad():
                for k in names + ("ira_e",):
                    getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
            m.eval()
            with torch.no_grad():
                H = m.encode(feats, keep, nq, B)
                xb = torch.cat([feats[G.KT_KEY], m.chain(feats, keep, nq)], 1).to(H.dtype)
                got = ira_stack(G, m, H, xb, lk, gd, nq, B, None)
                object.__setattr__(m, "rtr_e", m.ira_e.detach())    # RTR's e, outside the module's parameters
                try:
                    rtr = RTR.rounds_stack(G, m, H, xb, lk, gd, nq, B, None, 2)
                finally:
                    object.__delattr__(m, "rtr_e")
                out["start_is_rtr"] = bool(torch.equal(got, rtr))
                out["forward_is_ira"] = bool(torch.equal(m(feats, keep, nq, B, bz), got))
                out["two_rounds_move"] = not bool(torch.equal(got, orig_stack(m, H, xb, lk, gd, nq, B)))
                m.ira_E.copy_(torch.randn(SLOTS, G.D, generator=gen) * 0.3)
                out["moves_with_E"] = not bool(torch.equal(ira_stack(G, m, H, xb, lk, gd, nq, B, None), got))
                m.ira_E.zero_()
                m.ira_lam.fill_(0.5)
                out["moves_with_lam"] = not bool(torch.equal(ira_stack(G, m, H, xb, lk, gd, nq, B, None), got))
                m.ira_E.copy_(torch.randn(SLOTS, G.D, generator=gen) * 0.3)
            m.train()
            m.zero_grad()
            yt = m(feats, keep, nq, B, bz)
            yt.sum().backward()
            last = tuple(G6.G6_LAYER[-1]) + tuple(G.G6R_LAYER[-1])
            out["train_finite"] = bool(torch.isfinite(yt).all())
            out["ira_no_grad"] = [k for k in IRA_NAMES if getattr(m, k).grad is None
                                  or float(getattr(m, k).grad.abs().sum()) == 0]
            out["last_no_grad"] = [k for k in last if getattr(m, k).grad is None or float(getattr(m, k).grad.abs().sum()) == 0]
            sd = m.state_dict()
            G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(sd, strict=True)
            out["strict_load_patched"] = "passed"
        out["restored"] = G.g6r_stack is orig_stack and G.ZG6R.__init__ is orig_init
        plain_m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
        out["plain_has_ira"] = any(hasattr(plain_m, k) for k in IRA_NAMES)
        try:
            plain_m.load_state_dict(sd, strict=True)
            out["strict_load_plain"] = "passed"
        except RuntimeError:
            out["strict_load_plain"] = "refused"
    return out


def selftest():
    import tempfile
    import torch
    assert fit_name(0, "WQ-musique") == "scr-ira-k4-s0" and fit_name(2, "WQ-hotpotqa") == "scr-ira-k4-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-ira-k4-vs-base-s1-hp"
    assert pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-ira-k4-s0", "scr-ml1-zg6r-s0")
    assert pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-ira-k4-s0-hp", "scr-iso-zg6rm-s0-hp")
    # 1. RTR's condition opens IRA: missing or foreign calls stop it
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / RTR.DEP_CALLS
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"declared_in": DEP.DECLARED, "screen": {"verdict": NO_GAIN}}), encoding="utf-8")
        assert dep_call(td) == "screen NO_GAIN"
        p.write_text(json.dumps({"declared_in": DEP.DECLARED, "screen": {"verdict": ADOPT_FOR_FULL_RUN}}), encoding="utf-8")
        for path in (td, Path(td) / "none"):
            try:
                dep_call(path)
                raise AssertionError("DEP's screen asking for its full run, or no calls, must stop")
            except SystemExit:
                pass
    # 2. the plan, and the rounds on zg6r's toy carve
    G, _U2N = M1.import_arm(ARM)
    assert G.LAYERS == DEP.BASE_LAYERS == 3 and G.D == 64, (G.LAYERS, G.D)
    tp = toy_plan(G)
    assert tp["rows_max_abs"] < 1e-5 and tp["slots_max_abs"] < 1e-2 and tp["nonneg"], tp
    assert tp["independent_max_abs"] < 1e-6 and tp["permuted_max_abs"] < 1e-5, tp
    assert tp["sharp_top4_min"] > 0.8 and tp["sharp_rest_sum"] < 0.1, tp
    assert tp["softmax_top1"] > 0.99 and tp["softmax_second"] < 0.01, tp
    t = toy_rounds(G)
    assert t["typed_edges"] > 0 and t["edges"] > t["typed_edges"], t
    assert t["starts"] and t["parameters_h16"] == t["want_h16"], t
    assert t["start_is_rtr"] and t["forward_is_ira"] and t["two_rounds_move"], t
    assert t["moves_with_E"] and t["moves_with_lam"] and t["train_finite"], t
    assert not t["ira_no_grad"] and not t["last_no_grad"], t
    assert t["strict_load_patched"] == "passed" and t["restored"] and not t["plain_has_ira"], t
    assert t["strict_load_plain"] == "refused", t
    assert ira_record(G) == {"rounds": 2, "slots": 4, "sinkhorn_iters": 50, "ira_parameters": 579,
                             "parameters_h128": 180218 + 579, "layers": 3}, ira_record(G)
    # 3. the patches, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder)
    gsaved = (G.encoder_used, G.g6r_stack, G.ZG6R.__init__)
    seen = []
    orig_stamp = M1.stamp
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_fit(G, ISO.REPLACE, "screen NO_GAIN"):
            assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
            assert M1.load_encoder is ISO.own_load and G.encoder_used is ISO.own_used
            assert G.g6r_stack is not gsaved[1] and G.ZG6R.__init__ is not gsaved[2]
            cs = M1.code_shas(ARM)
            assert "zira_sha256" in cs and "zrtr_sha256" in cs and "zdep_sha256" in cs and "ml1start_sha256" in cs
            assert "readswa_sha256" in cs and RS.installed()
            M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-ira-k4-s0-hp"
            assert M1.pko_fit_name(ARM, 1, "WQ-hotpotqa") == "scr-ml1-zg6r-s1-hp"
            M1._ACTIVE[0] = None
            M1.stamp(fit_name(0, "WQ-hotpotqa"), {}, "screen", {"x": 1})
        assert seen[-1] == ("scr-ira-k4-s0-hp", "screen", {"x": 1, **ira_record(G), "base": "scr-iso-zg6rm-s0-hp",
                                                            "iso_verdict": ISO.REPLACE, "dep_call": "screen NO_GAIN"}), seen
        assert (G.encoder_used, G.g6r_stack, G.ZG6R.__init__) == gsaved and len(M1.ALL_FITS) == n0 and not RS.installed()
        with on_fit(G, ISO.KEEP, "screen NOT_ADOPTED"):
            assert M1.load_encoder is MS.ml1_load and G.encoder_used is gsaved[0] and M1.KEY == KEY and RS.installed()
            assert G.g6r_stack is not gsaved[1]
        try:
            with on_fit(G, ISO.REPLACE, "screen NO_GAIN"):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder) == saved
    assert (G.encoder_used, G.g6r_stack, G.ZG6R.__init__) == gsaved and len(M1.ALL_FITS) == n0 and not RS.installed()
    # 4. the beside numbers and section 6's rule for the next form
    with tempfile.TemporaryDirectory() as td:
        nm = fit_name(0, "WQ-musique")
        fd = M1.screen_dir(td) / "fits" / nm
        fd.mkdir(parents=True)
        st = {"ira_lam": torch.tensor([-0.2]), "ira_bz": torch.tensor([1.5]), "ira_b0": torch.tensor([0.3]),
              "ira_E": torch.ones(SLOTS, 64), "ira_e": torch.zeros(64), "ira_a": torch.ones(SLOTS, 64)}
        torch.save({"states": {DEP.CAND: st}}, fd / "models.pt")
        pn = plan_numbers(nm, td)
        assert pn["lam_raw"] == float(torch.tensor(-0.2)) and pn["lam"] == 0.0 and abs(pn["norm_E"] - 16.0) < 1e-5, pn
        assert plan_numbers(fit_name(0, "WQ-hotpotqa"), td) is None
    zero, used = {"lam": 0.0}, {"lam": 0.3}
    both = {fit_name(0, sp): zero for sp in M1.SPLITS}
    assert next_form(NO_GAIN, both) == "coverage" and next_form(NOT_ADOPTED, both) == "coverage"
    assert next_form(NO_GAIN, {**both, fit_name(0, "WQ-hotpotqa"): used}) == "three_rounds"
    assert next_form(ADOPT_FOR_FULL_RUN, both) is None and next_form(NO_GAIN, {fit_name(0, "WQ-musique"): zero}) is None
    # 5. DEP's rule, as IRA calls it: sixteen reads, the floors from the base's spread
    rk = DEP.read_keys()
    assert len(rk) == 16 and len(set(rk)) == 16
    bs = {0: 0.50, 1: 0.51, 2: 0.49}                             # spread 0.01, floor 0.02
    base_ = {rd: dict(bs) for rd in rk}
    new_ = {rd: {0: 0.50} for rd in rk}
    assert DEP.decide_screen(new_, base_)["verdict"] == NO_GAIN
    new_[("WQ-hotpotqa", "musique-4hop")] = {0: 0.52}
    assert DEP.decide_screen(new_, base_)["verdict"] == ADOPT_FOR_FULL_RUN
    new_[RULE.READS[0]] = {0: 0.48}
    assert DEP.decide_screen(new_, base_)["verdict"] == NOT_ADOPTED
    log("selftest: names and pairs, RTR's condition opening IRA, the plan on toy questions (rows within "
        f"{tp['rows_max_abs']:.1e}, slots within {tp['slots_max_abs']:.1e}; a sharp z puts at least {tp['sharp_top4_min']:.3f} "
        f"of a slot on each of the four top rows, where a softmax puts {tp['softmax_top1']:.3f} on one), the rounds "
        f"on zg6r's toy carve ({t['typed_edges']} typed of {t['edges']} edges: RTR's two rounds bit for bit at the start, every "
        f"IRA parameter's gradient, {t['parameters_h16']} parameters at hidden 16, the strict load under the patch and its "
        "refusal without), the patches and their restoring (also on a failure), the beside numbers and section 6's rule, and "
        "DEP's rule pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
