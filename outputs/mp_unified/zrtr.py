"""RTR's stage (docs/RTR_GO_BACK_ROUNDS.md): MUSIQUE_DIAGNOSIS step 3's other depth form, after DEP (docs/DEP_GNN_DEPTH.md
section 6). The GNN of record with a second round of its own three message-passing layers that goes back to the neighbours of
round one's top candidates. Round two's question context is the softmax over the question's rows of round one's clipped
z-scored score (zg6r's rule for its first context, applied to its own score), and each row starts round two from its first
state plus its round-one z-score times a learned vector (64 numbers, starting at 0). The layers and the readout are round
one's; the score, in training and at reading, is round two's. On PKO's roots, on the host's GPU: the screen at seed 0 on both
WQ splits against the base's seed-0 fit, on the twelve reads and on musique's 3-hop and 4-hop slices, under DEP's rule (zdep's
functions); the full run (seeds 1 and 2) when the screen asks for it.

The base is the GNN of record after ISO, read from ISO's filed verdict as DEP reads it (zdep.iso_call, zdep.base_fit). An RTR
fit runs only when DEP's filed calls keep three layers: DEP's screen NO_GAIN or NOT_ADOPTED, or its full run NOT_ADOPTED.
zml1s.py, zg6r.py, ziso.py and zdep.py are imported and not edited. For the length of an RTR fit, zg6r's g6r_stack is the
rounds' stack and ZG6R's __init__ adds the vector after zg6r's own; zml1s runs with RTR's names, stamp key, stage and
declaration, and on REPLACE with ISO's own-start patches. Every patched name is put back afterwards, also on a failure.

    python outputs/mp_unified/zrtr.py fit --name scr-rtr-r2-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/zrtr.py compare-all                    (each RTR fit read against the base's fit of its seed)
    python outputs/mp_unified/zrtr.py calls                          (outputs/rtr/calls.json and .md)
    python outputs/mp_unified/zrtr.py --selftest
"""
import argparse
import contextlib
import json
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

DECLARED = "docs/RTR_GO_BACK_ROUNDS.md"
STAGE = "RTR"
KEY = "rtr"                                     # the stamp's key in an RTR fit's screen.json and read.json and RTR's compares
OUT = "outputs/rtr"
ARM = DEP.ARM                                   # zrm's arm: the model is zg6r's with a second round
TAG = "r2"
ROUNDS = 2
DEP_CALLS = Path(DEP.OUT) / "calls.json"
CODE_SKIP = DEP.CODE_SKIP                       # stage modules, not the trainer's code
R5_TOL = DEP.R5_TOL
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    DEP.ADOPT_FOR_FULL_RUN, DEP.NO_GAIN, DEP.NOT_ADOPTED, DEP.ADOPT, DEP.INCOMPLETE)


def fit_name(seed, split):
    return f"scr-rtr-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-rtr-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS) | set(DEP.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zrtr: {msg}", flush=True)


# ── when RTR runs, and its base ──────────────────────────────────────────────


def dep_call(base=ROOT):
    """DEP's filed calls keep three layers (DEP section 6): its screen NO_GAIN or NOT_ADOPTED, or its full run NOT_ADOPTED.
    Returns which; anything else stops an RTR fit."""
    p = Path(base) / DEP_CALLS
    rec = M1.WQ.read_json(p) if p.is_file() else {}
    if rec.get("declared_in") != DEP.DECLARED:
        raise SystemExit(f"zrtr: {DEP_CALLS} is not DEP's filed calls; RTR runs after DEP's verdict")
    sc = (rec.get("screen") or {}).get("verdict")
    fr = (rec.get("full_run") or {}).get("verdict")
    if sc in (NO_GAIN, NOT_ADOPTED):
        return f"screen {sc}"
    if sc == ADOPT_FOR_FULL_RUN and fr == NOT_ADOPTED:
        return f"full run {fr}"
    raise SystemExit(f"zrtr: DEP's screen is {sc} and its full run {fr}; RTR runs when three layers stay (DEP section 6)")


def pairs(call):
    return {pair_name(sd, sp): (fit_name(sd, sp), DEP.base_fit(sd, sp, call)) for sd in RULE.SEEDS for sp in M1.SPLITS}


# ── the rounds ───────────────────────────────────────────────────────────────


def rounds_stack(G, m, H, xb, lk, gd, nq, B, u01=None, rounds=ROUNDS):
    """zg6r's score (N,) after `rounds` rounds. Round one is zg6r.g6r_stack, op for op. Each later round runs the same layers
    again from h0 + z * rtr_e, z the last round's clipped z-score over the question's rows, with the question's context
    weights the softmax of z over those rows (zg6r's rule for its first context, on the last round's score). The readout is
    zg6r's, on the round's start and end states. The family dropout's one draw (u01, in training) holds for every round."""
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
    for _r in range(1, rounds):
        z = G.LG.seg_zscore8D(s.unsqueeze(1), nq, B).squeeze(1).clamp(-G.ZCLIP, G.ZCLIP)
        w = G.G2.seg_softmax_rows(z, nq, B).unsqueeze(1)
        hr = h0 + z.unsqueeze(1) * m.rtr_e
        h = G.g6r_layers(m, hr, xb, w, u, v, f, nq, B, te, codes, cte)
        s = (Fn.linear(h, m.g6_out_w) + Fn.linear(hr, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
    return s


@contextlib.contextmanager
def on_rounds(G):
    """For the block: zg6r's g6r_stack is rounds_stack at ROUNDS rounds (zg6r's forward calls it by name), and a ZG6R built
    gets rtr_e (D numbers, 0) after zg6r's own __init__, which takes no draw, so every other start is zg6r's. Both are
    restored after, also on a failure."""
    orig_init = G.ZG6R.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        self.rtr_e = G.nn.Parameter(G.torch.zeros(G.D))

    def stack(m, H, xb, lk, gd, nq, B, u01=None):
        return rounds_stack(G, m, H, xb, lk, gd, nq, B, u01, ROUNDS)

    with M1.WQ.patch(G, g6r_stack=stack), M1.WQ.patch(G.ZG6R, __init__=init):
        yield


def rounds_record(G):
    return {"rounds": ROUNDS, "rounds_parameters": int(G.D), "parameters_h128": int(G.N_PARAMS + G.D),
            "layers": int(G.LAYERS)}


def rounds_record_keys():
    return ("rounds", "rounds_parameters", "parameters_h128", "layers")


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """RTR's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def rtr_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the RTR fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def stamp_with_rounds(orig, call, dep, rec):
    """zml1s.stamp with the rounds' record, the base, ISO's verdict and DEP's call."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        return orig(name, env, what, {**(extra or {}), **rec, "base": DEP.base_fit(sd, sp, call), "iso_verdict": call,
                                      "dep_call": dep})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with RTR's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    sha, dsha = M1.WQ.sha_src(__file__), M1.WQ.sha_src(DEP.__file__)
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=rtr_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zrtr_sha256": sha, "zdep_sha256": dsha}):
        yield


@contextlib.contextmanager
def on_rtr(G, call, dep):
    """An RTR fit's context: RTR's names and stamp, the rounds, and on REPLACE ISO's own start (its load_encoder in zml1s,
    its encoder_used in zg6r); restored after."""
    own = call == ISO.REPLACE
    with on_names(), M1.WQ.patch(M1, stamp=stamp_with_rounds(M1.stamp, call, dep, rounds_record(G)),
                                 **({"load_encoder": ISO.own_load} if own else {})), \
            (M1.WQ.patch(G, encoder_used=ISO.own_used) if own else contextlib.nullcontext()), on_rounds(G):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zrtr: {name} is not one of RTR's fits ({', '.join(FITS)})")
    dep = dep_call()
    call = DEP.iso_call()
    G, _U2N = M1.import_arm(ARM)
    if G.LAYERS != DEP.BASE_LAYERS:
        raise SystemExit(f"zrtr: zg6r holds {G.LAYERS} layers, not {DEP.BASE_LAYERS}")
    _a, sd, sp = FITS[name]
    log(f"{name}: {ROUNDS} rounds on {DEP.base_fit(sd, sp, call)}'s start (ISO {call}; DEP's {dep}); {rounds_record(G)}")
    with on_rtr(G, call, dep):
        return M1.fit(name)


# ── the compares, the slices and the calls ───────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: RTR's, ISO's or ML1's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    return DEP.fit_block(name, what, base)


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zrtr: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each RTR fit that is read against the base's fit of its seed and split, through zwq's run_compare; a compare filed
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
            raise SystemExit(f"zrtr: {new} and {bse} ran on different devices: {sorted(dev)}")
        with ISO.with_fits(), on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zrtr: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "iso_verdict": call,
                    "device": json.loads(dev.pop()), "zrtr_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_rtr(call, base=ROOT):
    """({(split, dataset): {seed: R@5}} of RTR's fits, the same of their bases) from RTR's compares."""
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
    """({(split, slice): {seed: R@5}} of RTR's read fits, the same of the base's three seeds, problems), by zdep.slice_r5.
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
    """Why an RTR fit that is read is not the declared one: its rounds, base and ISO verdict, its start, its device, and its
    trainer's code against the base's fit of its seed and split."""
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
            if (k.get("rounds"), k.get("layers"), k.get("base"), k.get("iso_verdict")) != (ROUNDS, DEP.BASE_LAYERS, bse, call):
                out.append(f"{name}: its {what} names {k.get('rounds')} rounds of {k.get('layers')} layers on {k.get('base')} "
                           f"(ISO {k.get('iso_verdict')})")
            if not str(k.get("dep_call") or "").endswith((NO_GAIN, NOT_ADOPTED)):
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


def calls(base=ROOT):
    call = DEP.iso_call(base)
    new_r5, b0 = r5_rtr(call, base)
    base3 = DEP.base_seeds(call, base)
    bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in RTR's compare, {base3.get(rd, {}).get(sd)} in its own stage's"
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
    dp = Path(base) / DEP_CALLS
    dep = M1.WQ.read_json(dp) if dp.is_file() else {}
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{ARM} with {ROUNDS} rounds of its layers", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)",
           "dep": {"screen": (dep.get("screen") or {}).get("verdict"), "full_run": (dep.get("full_run") or {}).get("verdict")},
           "screen": screen, "full_run": full, "problems": bad,
           "parameters": {f: {k: fit_block(f, "screen", base).get(k) for k in rounds_record_keys()}
                          for f in FITS if is_read(f, base)},
           "zrtr_sha256": M1.WQ.sha_src(__file__), "zdep_sha256": M1.WQ.sha_src(DEP.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# RTR: {ARM} with {ROUNDS} rounds of its layers against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    if screen["reads"]:
        md += ["| split | read | base s0, s1, s2 | RTR s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
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
    if bad:
        md += ["", "problems:"] + [f"- {p}" for p in bad]
    (out / "calls.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"screen {screen['verdict']} {json.dumps(screen.get('counts', {}))}"
        + (f"; full run {full['verdict']}" if full else "") + f"; {OUT}/calls.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_rounds(G):
    """The rounds on zg6r's own toy carve (zdep's toy: zg6r's self-test steps 2 and 3), parameters drawn at random: round one
    against zg6r's stack bit for bit (typed edges in); round two by hand through zg6's own layers (typed edges taken out,
    where zg6r's layer is zg6's op for op); round two with rtr_e at 0; a training forward (checkpointed) and backward; the
    parameter count; the state dict's strict load with and without the patch; the patch's restoring."""
    import tempfile
    import torch
    Fn = torch.nn.functional
    G6 = G.G6
    G.LG.bind_device_ops()
    out = {}
    with tempfile.TemporaryDirectory(prefix="zrtr_") as td:
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
        names = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES) + ("rtr_e",)
        gen = torch.Generator().manual_seed(3)
        with on_rounds(G):
            torch.manual_seed(0)
            m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
            out["e_starts_at_zero"] = bool((m.rtr_e == 0).all())
            out["parameters_h16"] = sum(int(getattr(m, k).numel()) for k in names)
            out["want_h16"] = G6.n_params(16) + G.N_G6R + G.D
            with torch.no_grad():
                for k in names:
                    getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
            m.eval()
            with torch.no_grad():
                H = m.encode(feats, keep, nq, B)
                xb = torch.cat([feats[G.KT_KEY], m.chain(feats, keep, nq)], 1).to(H.dtype)
                one = rounds_stack(G, m, H, xb, lk, gd, nq, B, None, 1)
                out["round1_is_zg6r"] = bool(torch.equal(one, orig_stack(m, H, xb, lk, gd, nq, B)))
                y = m(feats, keep, nq, B, bz)
                two = rounds_stack(G, m, H, xb, lk, gd, nq, B, None, 2)
                out["forward_is_two_rounds"] = bool(torch.equal(y, two)) and not bool(torch.equal(y, one))
                gd0 = dict(gd, ci=torch.full_like(gd["ci"], -1))           # typed edges out: zg6r's layer is zg6's
                got = rounds_stack(G, m, H, xb, lk, gd0, nq, B, None, 2)
                s1 = G6.g6_stack(m, H, xb, lk, nq, B)
                z = G.LG.seg_zscore8D(s1.unsqueeze(1), nq, B).squeeze(1).clamp(-G.ZCLIP, G.ZCLIP)
                w = G.G2.seg_softmax_rows(z, nq, B).unsqueeze(1)
                h0 = Fn.gelu(Fn.linear(torch.cat([H, xb], 1), m.g6_in_w, m.g6_in_b))
                hr = h0 + z.unsqueeze(1) * m.rtr_e
                h = G6.g6_layers(m, hr, xb, w, lk["u"], lk["v"], lk["f"], nq, B)
                want = (Fn.linear(h, m.g6_out_w) + Fn.linear(hr, m.g6_skip_w) + m.g6_out_b).squeeze(-1)
                out["round2_by_hand_max_abs"] = float((got - want).abs().max())
                e = m.rtr_e.detach().clone()
                m.rtr_e.zero_()
                out["round2_moves_with_e_zero"] = not bool(torch.equal(rounds_stack(G, m, H, xb, lk, gd, nq, B, None, 2), one))
                m.rtr_e.copy_(e)
            m.train()
            m.zero_grad()
            yt = m(feats, keep, nq, B, bz)
            yt.sum().backward()
            last = tuple(G6.G6_LAYER[-1]) + tuple(G.G6R_LAYER[-1])
            out["train_finite"] = bool(torch.isfinite(yt).all())
            out["e_grad"] = float(m.rtr_e.grad.abs().sum()) if m.rtr_e.grad is not None else 0.0
            out["last_no_grad"] = [k for k in last if getattr(m, k).grad is None or float(getattr(m, k).grad.abs().sum()) == 0]
            sd = m.state_dict()
            G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0).load_state_dict(sd, strict=True)
            out["strict_load_patched"] = "passed"
        out["restored"] = G.g6r_stack is orig_stack and G.ZG6R.__init__ is orig_init
        plain_m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
        out["plain_has_e"] = hasattr(plain_m, "rtr_e")
        try:
            plain_m.load_state_dict(sd, strict=True)
            out["strict_load_plain"] = "passed"
        except RuntimeError:
            out["strict_load_plain"] = "refused"
    return out


def selftest():
    import tempfile
    assert fit_name(0, "WQ-musique") == "scr-rtr-r2-s0" and fit_name(2, "WQ-hotpotqa") == "scr-rtr-r2-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-rtr-r2-vs-base-s1-hp"
    assert pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-rtr-r2-s0", "scr-ml1-zg6r-s0")
    assert pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-rtr-r2-s0-hp", "scr-iso-zg6rm-s0-hp")
    # 1. DEP's filed calls open RTR only when three layers stay
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / DEP_CALLS
        p.parent.mkdir(parents=True)
        for sc, fr, want in ((NO_GAIN, None, "screen NO_GAIN"), (NOT_ADOPTED, None, "screen NOT_ADOPTED"),
                             (ADOPT_FOR_FULL_RUN, NOT_ADOPTED, "full run NOT_ADOPTED"), (ADOPT_FOR_FULL_RUN, None, None),
                             (ADOPT_FOR_FULL_RUN, ADOPT, None), (ADOPT_FOR_FULL_RUN, INCOMPLETE, None), (INCOMPLETE, None, None)):
            p.write_text(json.dumps({"declared_in": DEP.DECLARED, "screen": {"verdict": sc},
                                     "full_run": {"verdict": fr} if fr else None}), encoding="utf-8")
            try:
                got = dep_call(td)
            except SystemExit:
                got = None
            assert got == want, (sc, fr, got)
        p.write_text(json.dumps({"declared_in": "elsewhere", "screen": {"verdict": NO_GAIN}}), encoding="utf-8")
        for path in (td, Path(td) / "none"):
            try:
                dep_call(path)
                raise AssertionError("calls not DEP's, or missing, must stop")
            except SystemExit:
                pass
    # 2. the rounds on zg6r's toy carve
    G, _U2N = M1.import_arm(ARM)
    assert G.LAYERS == DEP.BASE_LAYERS == 3 and G.D == 64, (G.LAYERS, G.D)
    t = toy_rounds(G)
    assert t["typed_edges"] > 0 and t["edges"] > t["typed_edges"], t
    assert t["e_starts_at_zero"] and t["parameters_h16"] == t["want_h16"], t
    assert t["round1_is_zg6r"] and t["forward_is_two_rounds"] and t["round2_by_hand_max_abs"] < 1e-5, t
    assert t["round2_moves_with_e_zero"] and t["train_finite"] and t["e_grad"] > 0 and not t["last_no_grad"], t
    assert t["strict_load_patched"] == "passed" and t["restored"] and not t["plain_has_e"], t
    assert t["strict_load_plain"] == "refused", t
    assert rounds_record(G) == {"rounds": 2, "rounds_parameters": 64, "parameters_h128": 180218 + 64, "layers": 3}
    # 3. the patches, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder)
    gsaved = (G.encoder_used, G.g6r_stack, G.ZG6R.__init__)
    seen = []
    orig_stamp = M1.stamp
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_rtr(G, ISO.REPLACE, "screen NO_GAIN"):
            assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
            assert M1.load_encoder is ISO.own_load and G.encoder_used is ISO.own_used
            assert G.g6r_stack is not gsaved[1] and G.ZG6R.__init__ is not gsaved[2]
            cs = M1.code_shas(ARM)
            assert "zrtr_sha256" in cs and "zdep_sha256" in cs
            M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-rtr-r2-s0-hp"
            assert M1.pko_fit_name(ARM, 1, "WQ-hotpotqa") == "scr-ml1-zg6r-s1-hp"
            M1._ACTIVE[0] = None
            M1.stamp(fit_name(0, "WQ-hotpotqa"), {}, "screen", {"x": 1})
        assert seen[-1] == ("scr-rtr-r2-s0-hp", "screen", {"x": 1, **rounds_record(G), "base": "scr-iso-zg6rm-s0-hp",
                                                            "iso_verdict": ISO.REPLACE, "dep_call": "screen NO_GAIN"}), seen
        assert (G.encoder_used, G.g6r_stack, G.ZG6R.__init__) == gsaved and len(M1.ALL_FITS) == n0
        with on_rtr(G, ISO.KEEP, "screen NOT_ADOPTED"):
            assert M1.load_encoder is saved[6] and G.encoder_used is gsaved[0] and M1.KEY == KEY
            assert G.g6r_stack is not gsaved[1]
        try:
            with on_rtr(G, ISO.REPLACE, "screen NO_GAIN"):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
        M1._ACTIVE[0] = None
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder) == saved
    assert (G.encoder_used, G.g6r_stack, G.ZG6R.__init__) == gsaved and len(M1.ALL_FITS) == n0
    # 4. DEP's rule, as RTR calls it: sixteen reads, the floors from the base's spread
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
    log("selftest: names and pairs, DEP's calls opening RTR only when three layers stay, the rounds on zg6r's toy carve "
        f"({t['typed_edges']} typed of {t['edges']} edges: round one zg6r's bit for bit, round two within "
        f"{t['round2_by_hand_max_abs']:.1e} of zg6's layers by hand, rtr_e's gradient, {t['parameters_h16']} parameters at "
        "hidden 16, the strict load under the patch and its refusal without), the patches and their restoring (also on a "
        "failure), and DEP's rule pass")
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
