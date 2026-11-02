"""DEP's stage (docs/DEP_GNN_DEPTH.md): MUSIQUE_DIAGNOSIS step 3's depth form on the GNN of record. The GNN of record with a
fourth message-passing layer (zg6's LAYERS, and with it zg6r's, 3 -> 4), on PKO's roots, on the host's GPU: the screen at seed
0 on both WQ splits against the base's seed-0 fit, on the twelve reads and on musique's 3-hop and 4-hop slices; the full run
(seeds 1 and 2) when the screen asks for it.

The base is the GNN of record after ISO (docs/ISO_ZG6R_OWN_START.md section 6), read from ISO's filed verdict: on KEEP, zg6r
(ML1's fits, started from ML1's seed-0 zspk encoder); on REPLACE, zg6r-m (ISO's fits, their own start). The four-layer arm
takes the base's start. zml1s.py, zg2.py, zg6.py, zg6r.py and ziso.py are imported and not edited: zg2's LAYERS is set to 4
before zg6 is first imported (zg6 and zg6r bind the count, and every name and size built from it, at import), and a fit
refuses when zg6 is already in its process. For the length of a DEP fit zml1s runs with DEP's names, stamp key, stage and
declaration, and on REPLACE with ISO's own-start patches. Every patched name is put back afterwards, also on a failure.

    python outputs/mp_unified/zdep.py fit --name scr-dep-l4-s0       (train when needed, then read; kept when read before)
    python outputs/mp_unified/zdep.py compare-all                     (each DEP fit read against the base's fit of its seed)
    python outputs/mp_unified/zdep.py calls                           (outputs/dep/calls.json and .md)
    python outputs/mp_unified/zdep.py --selftest
"""
import argparse
import contextlib
import json
import subprocess
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

DECLARED = "docs/DEP_GNN_DEPTH.md"
STAGE = "DEP"
KEY = "dep"                                     # the stamp's key in a DEP fit's screen.json and read.json and DEP's compares
OUT = "outputs/dep"
ARM = "zg6r"                                    # zrm's arm: the model is zg6r's with one more layer
TAG = "l4"
LAYERS, BASE_LAYERS = 4, 3
ISO_VERDICT = "outputs/iso/verdict.json"
SLICE_DS, CAND = "musique", "p@swa"
SLICES = {"musique-3hop": "3hop", "musique-4hop": "4hop"}       # musique's question ids start with their hop count
CODE_SKIP = ("zml1s_sha256", "ziso_sha256")    # stage modules, not the trainer's code
R5_TOL = 6e-5                                   # a read file's R@5 against its compare's (rounded to four places)
ADOPT_FOR_FULL_RUN, NO_GAIN, NOT_ADOPTED, ADOPT, INCOMPLETE = (
    "ADOPT_FOR_FULL_RUN", "NO_GAIN", "NOT_ADOPTED", "ADOPT", "INCOMPLETE")


def fit_name(seed, split):
    return f"scr-dep-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-dep-{TAG}-vs-base-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and not set(FITS) & (set(M1.ALL_FITS) | set(ISO.FITS))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zdep: {msg}", flush=True)


# ── the base and the four layers ─────────────────────────────────────────────


def iso_call(base=ROOT):
    """ISO's filed verdict, REPLACE or KEEP; anything else stops a DEP fit, compare or call (section 2)."""
    p = Path(base) / ISO_VERDICT
    v = M1.WQ.read_json(p).get("verdict") if p.is_file() else None
    if v not in (ISO.REPLACE, ISO.KEEP):
        raise SystemExit(f"zdep: ISO's verdict is {v}; DEP's base is the GNN of record after ISO")
    return v


def base_fit(seed, split, call):
    return ISO.fit_name(seed, split) if call == ISO.REPLACE else M1.fit_name(ARM, seed, split)


def pairs(call):
    return {pair_name(sd, sp): (fit_name(sd, sp), base_fit(sd, sp, call)) for sd in RULE.SEEDS for sp in M1.SPLITS}


def import_deep():
    """(zg6r with LAYERS layers, zu2n): zg2's LAYERS set before zg6's first import, then zml1s.import_arm's import and its
    checks; the counts checked after."""
    if "zg6" in sys.modules or "zg6r" in sys.modules:
        raise SystemExit("zdep: zg6 was imported before its layer count was set; a DEP fit runs in a process of its own")
    import zg2 as G2
    if G2.LAYERS != BASE_LAYERS:
        raise SystemExit(f"zdep: zg2 holds {G2.LAYERS} layers, not {BASE_LAYERS}")
    G2.LAYERS = LAYERS
    G, U2N = M1.import_arm(ARM)
    import zg6 as G6
    got = (G6.LAYERS, G.LAYERS, len(G6.G6_LAYER), len(G.G6R_LAYER), len(G.G6R_NAMES), G6.G2 is G2)
    if got != (LAYERS, LAYERS, LAYERS, LAYERS, 2 + 4 * LAYERS, True):
        raise SystemExit(f"zdep: the layer counts are {got}, not {LAYERS} throughout")
    return G, U2N


def depth_record():
    import zg6 as G6
    import zg6r as G6R
    return {"layers": G6.LAYERS, "zg6_parameters_h128": G6.n_params(128), "g6r_parameters": G6R.N_G6R}


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    """DEP's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def dep_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the DEP fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def stamp_with_depth(orig, call):
    """zml1s.stamp with the layer count, the parameter counts, the base and ISO's verdict."""
    def stamp(name, env, what, extra=None):
        if name not in FITS:
            return orig(name, env, what, extra)
        _a, sd, sp = FITS[name]
        return orig(name, env, what, {**(extra or {}), **depth_record(), "base": base_fit(sd, sp, call), "iso_verdict": call})
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with DEP's names, stamp key, stage and declaration; restored after. Enough for the compares and the calls."""
    sha = M1.WQ.sha_src(__file__)
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=dep_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zdep_sha256": sha}):
        yield


@contextlib.contextmanager
def on_dep(G, call):
    """A DEP fit's context: DEP's names and stamp, and on REPLACE ISO's own start (its load_encoder in zml1s, its
    encoder_used in zg6r); restored after."""
    own = call == ISO.REPLACE
    with on_names(), M1.WQ.patch(M1, stamp=stamp_with_depth(M1.stamp, call),
                                 **({"load_encoder": ISO.own_load} if own else {})), \
            (M1.WQ.patch(G, encoder_used=ISO.own_used) if own else contextlib.nullcontext()):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zdep: {name} is not one of DEP's fits ({', '.join(FITS)})")
    call = iso_call()
    G, _U2N = import_deep()
    log(f"{name}: {LAYERS} layers on {base_fit(FITS[name][1], FITS[name][2], call)}'s start (ISO {call}); {depth_record()}")
    with on_dep(G, call):
        return M1.fit(name)


# ── the compares, the slices and the calls ───────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: DEP's, ISO's or ML1's."""
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]
    if name in ISO.FITS:
        return ISO.fit_block(name, what, base)
    return M1.fit_stamp(name, M1.screen_dir(base) / "fits", what)[1]


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zdep: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    """Each DEP fit that is read against the base's fit of its seed and split, through zwq's run_compare; a compare filed
    before is kept. The full run's fits are compared once they exist."""
    call = iso_call(base)
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
            raise SystemExit(f"zdep: {new} and {bse} ran on different devices: {sorted(dev)}")
        with ISO.with_fits(), on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zdep: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "iso_verdict": call,
                    "device": json.loads(dev.pop()), "zdep_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_dep(call, base=ROOT):
    """({(split, dataset): {seed: R@5}} of DEP's fits, the same of their bases) from DEP's compares."""
    new_r5, base_r5 = {}, {}
    for cname, (new, bse) in pairs(call).items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != CAND or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def base_seeds(call, base=ROOT):
    """{(split, dataset): {seed: R@5}} of the base at its three seeds: ML1's compares for zg6r, ISO's for zg6r-m."""
    if call == ISO.REPLACE:
        return ISO.r5_pair(base)[0]
    return M1.r5_by_arm(base).get(ARM, {})


def slice_r5(name, base=ROOT):
    """{'all': R@5, slice: R@5} of a fit's p@swa on musique's s1eval read by the questions' hop count (questions with a
    gold), from its read file; None when the file is missing."""
    p = M1.screen_dir(base) / "fits" / name / "reads" / f"{SLICE_DS}__s1eval.npz"
    if not p.is_file():
        return None
    z = np.load(p)
    k = [str(c) for c in z["candidates"]].index(CAND)
    ids = np.asarray([str(x) for x in z["ids"]])
    gt = z["gold_total"].astype(np.int64)
    ok = gt > 0
    r = np.where(ok, z["top"][k].astype(np.float64) / np.maximum(gt, 1), 0.0)
    out = {"all": float(r[ok].mean())}
    for s, pre in SLICES.items():
        m = ok & np.char.startswith(ids, pre)
        out[s] = float(r[m].mean()) if m.any() else None
    return out


def slices(call, new_r5, base3, base=ROOT):
    """({(split, slice): {seed: R@5}} of DEP's read fits, the same of the base's three seeds, problems). Each file's
    all-question R@5 must be its compare's musique R@5."""
    new_s, base_s, bad = {}, {}, []
    for sp in M1.SPLITS:
        for sd in RULE.SEEDS:
            for f, own, ref in ((fit_name(sd, sp), new_s, new_r5), (base_fit(sd, sp, call), base_s, base3)):
                if f in FITS and not is_read(f, base):
                    continue
                s = slice_r5(f, base)
                if s is None:
                    bad.append(f"{f}: no {SLICE_DS} read file")
                    continue
                want = ref.get((sp, SLICE_DS), {}).get(sd)
                if want is not None and abs(s["all"] - want) > R5_TOL:
                    bad.append(f"{f}: its read file gives {SLICE_DS} R@5 {s['all']:.5f}, its compare {want}")
                for sl in SLICES:
                    if s[sl] is not None:
                        own.setdefault((sp, sl), {})[sd] = s[sl]
    return new_s, base_s, bad


def read_keys():
    return list(RULE.READS) + [(sp, sl) for sp in M1.SPLITS for sl in SLICES]


def decide_screen(new, base):
    """Section 4, the screen. Per read (the twelve and the four slices): floor = max(0.0075, 2 x the base's spread over its
    three seeds); the seed-0 delta at or below minus the floor is a LOSS, at or above it a GAIN. A LOSS anywhere:
    NOT_ADOPTED; else a GAIN anywhere: ADOPT_FOR_FULL_RUN; else NO_GAIN."""
    rows, problems = [], []
    for rd in read_keys():
        b, n = base.get(rd, {}), new.get(rd, {})
        if any(k not in b for k in RULE.SEEDS) or 0 not in n:
            problems.append(f"{rd[0]} {rd[1]}: the base's seeds {sorted(b)}, DEP's {sorted(n)}")
            continue
        s = RULE.spread(b)
        floor = RULE.pair_floor([s, s])
        d = n[0] - b[0]
        c = RULE.LOSS if d <= -floor + RULE.EPS else RULE.GAIN if d >= floor - RULE.EPS else RULE.WITHIN
        rows.append({"split": rd[0], "read": rd[1], "base": [round(b[k], 4) for k in RULE.SEEDS], "new": round(n[0], 4),
                     "delta": round(d, 4), "floor": round(floor, 4), "call": c})
    if problems:
        return {"verdict": INCOMPLETE, "problems": problems, "reads": rows, "counts": {}}
    n = {c: sum(1 for r in rows if r["call"] == c) for c in (RULE.GAIN, RULE.LOSS, RULE.WITHIN)}
    v = NOT_ADOPTED if n[RULE.LOSS] else ADOPT_FOR_FULL_RUN if n[RULE.GAIN] else NO_GAIN
    return {"verdict": v, "problems": [], "reads": rows, "counts": n}


def decide_full(new, base, new_s, base_s):
    """Section 4, the full run: zml1.verdict on the twelve reads, and each slice called by zml1.call with the pair floor of
    both arms' spreads. ADOPT: zml1 says ADOPT and no slice is a LOSS."""
    v = RULE.verdict(new, base)
    if v["verdict"] == INCOMPLETE:
        return {"verdict": INCOMPLETE, "problems": v["problems"], "reads": [], "slices": []}
    rows = []
    for rd in [(sp, sl) for sp in M1.SPLITS for sl in SLICES]:
        b, n = base_s.get(rd, {}), new_s.get(rd, {})
        if any(k not in b or k not in n for k in RULE.SEEDS):
            return {"verdict": INCOMPLETE, "problems": [f"{rd[0]} {rd[1]}: a seed's slice is missing"], "reads": v["reads"],
                    "slices": rows}
        d = {k: n[k] - b[k] for k in RULE.SEEDS}
        fl = RULE.pair_floor([RULE.spread(n), RULE.spread(b)])
        rows.append({"split": rd[0], "read": rd[1], "delta": {k: round(d[k], 4) for k in RULE.SEEDS}, "floor": round(fl, 6),
                     "call": RULE.call(d, fl)})
    ok = v["verdict"] == ADOPT and not any(r["call"] == RULE.LOSS for r in rows)
    return {"verdict": ADOPT if ok else NOT_ADOPTED, "problems": [], "reads": v["reads"], "slices": rows,
            "counts": v.get("counts", {})}


def fit_problems(call, base=ROOT):
    """Why a DEP fit that is read is not the declared one: its layers, base and ISO verdict, its start, its device, and its
    trainer's code against the base's fit of its seed and split."""
    out = []
    for name, (_a, sd, sp) in FITS.items():
        if not is_read(name, base):
            continue
        bse = base_fit(sd, sp, call)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k in (("train", ks), ("read", kr)):
            if (k.get("layers"), k.get("base"), k.get("iso_verdict")) != (LAYERS, bse, call):
                out.append(f"{name}: its {what} names {k.get('layers')} layers on {k.get('base')} (ISO {k.get('iso_verdict')})")
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
    call = iso_call(base)
    new_r5, b0 = r5_dep(call, base)
    base3 = base_seeds(call, base)
    bad = [f"{rd[0]} {rd[1]} seed {sd}: the base gives {v} in DEP's compare, {base3.get(rd, {}).get(sd)} in its own stage's"
           for rd, by in sorted(b0.items()) for sd, v in sorted(by.items())
           if base3.get(rd, {}).get(sd) is None or abs(v - base3[rd][sd]) > R5_TOL]
    new_s, base_s, bad_s = slices(call, new_r5, base3, base)
    bad += bad_s
    new_all, base_all = {**new_r5, **new_s}, {**base3, **base_s}
    screen = decide_screen(new_all, base_all)
    full_ready = all(is_read(fit_name(sd, sp), base) for sd in RULE.SEEDS for sp in M1.SPLITS)
    full = decide_full(new_r5, base3, new_s, base_s) if full_ready else None
    bad += fit_problems(call, base)
    for blk in (screen, full):
        if blk is not None and bad and blk["verdict"] != INCOMPLETE:
            blk["verdict"] = INCOMPLETE
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": f"{ARM} with {LAYERS} layers", "iso_verdict": call,
           "base": "zg6r-m (ISO)" if call == ISO.REPLACE else "zg6r (ML1)", "screen": screen, "full_run": full,
           "problems": bad, "parameters": {f: {k: fit_block(f, "screen", base).get(k) for k in depth_record_keys()}
                                           for f in FITS if is_read(f, base)},
           "zdep_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "calls.json", rec)
    md = [f"# DEP: {ARM} with {LAYERS} message-passing layers against {rec['base']}, on PKO's pools", "",
          f"## Screen: {screen['verdict']}", "", f"counts: {json.dumps(screen.get('counts', {}))}", ""]
    if screen["reads"]:
        md += ["| split | read | base s0, s1, s2 | DEP s0 | delta | floor | call |", "|---|---|---|---:|---:|---:|---|"]
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


def depth_record_keys():
    return ("layers", "zg6_parameters_h128", "g6r_parameters")


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_forward(G):
    """The four-layer zg6r on zg6r's own toy carve (its self-test's steps 2 and 3), its parameters drawn at random: the score
    in eval against zg6r's by-hand layers, a training forward (checkpointed) and backward, and the parameters of the last
    layer that get no gradient in eval."""
    import tempfile
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    with tempfile.TemporaryDirectory(prefix="zdep_") as td:
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
        torch.manual_seed(0)
        m = G.ZG6R(blocks, ce.widths, 16, dropout=0.1, seed=0)
        names = tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES)
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for k in names:
                getattr(m, k).copy_(torch.randn(getattr(m, k).shape, generator=gen) * 0.3)
        m.eval()
        with torch.no_grad():
            H = m.encode(feats, keep, nq, B)
            xb = torch.cat([feats[G.KT_KEY], m.chain(feats, keep, nq)], 1)
            y = m(feats, keep, nq, B, bz)
            want = G.by_hand(m, H, xb, feats[G.LK_KEY], feats[G.G6R_KEY], nq, B)
        m.zero_grad()
        m(feats, keep, nq, B, bz).sum().backward()
        last = tuple(G6.G6_LAYER[-1]) + tuple(G.G6R_LAYER[-1])
        no_grad = [k for k in last if getattr(m, k).grad is None or float(getattr(m, k).grad.abs().sum()) == 0]
        m.train()
        m.zero_grad()
        yt = m(feats, keep, nq, B, bz)
        yt.sum().backward()
        return {"parameters_h16": sum(int(getattr(m, k).numel()) for k in names), "want_h16": G6.n_params(16) + G.N_G6R,
                "by_hand_max_abs": float((y - want).abs().max()), "train_finite": bool(torch.isfinite(yt).all()),
                "train_grad": any(getattr(m, k).grad is not None for k in last), "last_layer": len(last),
                "last_no_grad": no_grad}


def layers_check():
    """Run in a process of its own by the self-test: the four-layer import, its counts, a toy forward and backward, and the
    refusal of a second import."""
    G, _U2N = import_deep()
    import zg6 as G6
    rec = {**depth_record(), "g6r_layers": G.LAYERS, "g6_names": len(G6.G6_NAMES), "g6r_names": len(G.G6R_NAMES),
           "toy": toy_forward(G)}
    try:
        import_deep()
        rec["second_import"] = "passed"
    except SystemExit:
        rec["second_import"] = "refused"
    print("LAYERS_CHECK " + json.dumps(rec), flush=True)
    return 0


def selftest():
    import tempfile
    import types
    assert fit_name(0, "WQ-musique") == "scr-dep-l4-s0" and fit_name(2, "WQ-hotpotqa") == "scr-dep-l4-s2-hp"
    assert pair_name(1, "WQ-hotpotqa") == "scr-dep-l4-vs-base-s1-hp"
    assert pairs(ISO.KEEP)[pair_name(0, "WQ-musique")] == ("scr-dep-l4-s0", "scr-ml1-zg6r-s0")
    assert pairs(ISO.REPLACE)[pair_name(0, "WQ-hotpotqa")] == ("scr-dep-l4-s0-hp", "scr-iso-zg6rm-s0-hp")
    # 1. ISO's verdict picks the base; any other verdict stops
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / ISO_VERDICT
        p.parent.mkdir(parents=True)
        for v in (ISO.REPLACE, ISO.KEEP):
            p.write_text(json.dumps({"verdict": v}), encoding="utf-8")
            assert iso_call(td) == v
        for v in (ISO.INCOMPLETE, None):
            p.write_text(json.dumps({"verdict": v}), encoding="utf-8")
            try:
                iso_call(td)
                raise AssertionError(f"ISO {v} passed")
            except SystemExit:
                pass
    # 2. the four layers, in a process of its own (zg6 binds its count at import)
    pr = subprocess.run([sys.executable, str(Path(__file__)), "layers-check"], capture_output=True, text=True, timeout=900)
    line = [x for x in (pr.stdout or "").splitlines() if x.startswith("LAYERS_CHECK ")]
    assert pr.returncode == 0 and line, (pr.returncode, (pr.stderr or "")[-1500:])
    rec = json.loads(line[0].split(" ", 1)[1])
    base_g6, per = 121802, (121802 - (64 * (128 + 10) + 64 + 64 + 2 * 64 + 1)) // 3
    assert rec["layers"] == rec["g6r_layers"] == LAYERS and rec["zg6_parameters_h128"] == base_g6 + per, rec
    assert rec["g6r_names"] == 2 + 4 * LAYERS and rec["second_import"] == "refused", rec
    t = rec["toy"]
    assert t["parameters_h16"] == t["want_h16"] and t["by_hand_max_abs"] < 1e-4 and t["train_finite"], t
    assert t["train_grad"] and t["last_layer"] == 12 and not t["last_no_grad"], t
    # 3. the patches, on fakes, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder)
    seen = []
    Gf = types.SimpleNamespace(encoder_used="orig")
    orig_stamp = M1.stamp
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, dict(extra or {})))
        with on_dep(Gf, ISO.REPLACE):
            assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
            assert M1.load_encoder is ISO.own_load and Gf.encoder_used is ISO.own_used and "zdep_sha256" in M1.code_shas(ARM)
            M1._ACTIVE[0] = fit_name(0, "WQ-hotpotqa")
            assert M1.pko_fit_name(ARM, 0, "WQ-hotpotqa") == "scr-dep-l4-s0-hp"
            assert M1.pko_fit_name(ARM, 1, "WQ-hotpotqa") == "scr-ml1-zg6r-s1-hp"
            M1._ACTIVE[0] = None
            dr = {"layers": LAYERS, "zg6_parameters_h128": 1, "g6r_parameters": 2}
            with M1.WQ.patch(sys.modules[__name__], depth_record=lambda: dr):
                M1.stamp(fit_name(0, "WQ-hotpotqa"), {}, "screen", {"x": 1})
        assert seen[-1] == ("scr-dep-l4-s0-hp", "screen", {"x": 1, **dr, "base": "scr-iso-zg6rm-s0-hp", "iso_verdict": ISO.REPLACE})
        assert Gf.encoder_used == "orig" and len(M1.ALL_FITS) == n0
        with on_dep(Gf, ISO.KEEP):
            assert M1.load_encoder is saved[6] and Gf.encoder_used == "orig" and M1.KEY == KEY
        try:
            with on_dep(Gf, ISO.REPLACE):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder) == saved
    assert len(M1.ALL_FITS) == n0 and Gf.encoder_used == "orig"
    # 4. the slices from a made-up read file: hop prefixes, questions without a gold left out, p@swa's row
    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / "outputs" / "screen" / "fits" / "f" / "reads"
        d.mkdir(parents=True)
        ids = ["2hop__a", "3hop1__b", "3hop2__c", "4hop1__d", "4hop3__e", "2hop__f"]
        np.savez_compressed(d / f"{SLICE_DS}__s1eval.npz", candidates=np.asarray(["p@ep0", CAND]), ids=np.asarray(ids),
                            gold_total=np.asarray([2, 2, 3, 4, 0, 2], np.int32),
                            top=np.asarray([[0] * 6, [1, 2, 1, 2, 0, 2]], np.int16))
        with M1.WQ.patch(M1, screen_dir=lambda base=ROOT: Path(base) / "outputs" / "screen"):
            s = slice_r5("f", td)
        assert abs(s["all"] - (0.5 + 1 + 1 / 3 + 0.5 + 1) / 5) < 1e-12 and abs(s["musique-3hop"] - (1 + 1 / 3) / 2) < 1e-12
        assert s["musique-4hop"] == 0.5
    # 5. the screen's rule on made-up reads
    def made(base_seeds_, delta):
        b = {rd: dict(base_seeds_) for rd in read_keys()}
        n = {rd: {0: base_seeds_[0] + delta.get(rd, 0.0)} for rd in read_keys()}
        return n, b
    bs = {0: 0.50, 1: 0.51, 2: 0.49}                             # spread 0.01, floor 0.02
    v = decide_screen(*made(bs, {}))
    assert v["verdict"] == NO_GAIN and all(r["floor"] == 0.02 for r in v["reads"]) and len(v["reads"]) == 16
    sl4 = ("WQ-hotpotqa", "musique-4hop")
    assert decide_screen(*made(bs, {sl4: 0.02}))["verdict"] == ADOPT_FOR_FULL_RUN
    assert decide_screen(*made(bs, {sl4: 0.03, RULE.READS[0]: -0.02}))["verdict"] == NOT_ADOPTED
    assert decide_screen(*made({0: 0.5, 1: 0.5, 2: 0.5}, {RULE.READS[3]: 0.0075}))["verdict"] == ADOPT_FOR_FULL_RUN
    assert decide_screen(*made(bs, {RULE.READS[3]: 0.0199}))["verdict"] == NO_GAIN
    n, b = made(bs, {})
    del b[sl4][2]
    assert decide_screen(n, b)["verdict"] == INCOMPLETE
    # 6. the full run's rule: zml1's verdict, and a slice LOSS blocks an ADOPT
    def full(dl, sl_dl):
        nr = {rd: {k: bs[k] + dl.get(rd, 0.0) + (0.001 if k else 0.0) for k in RULE.SEEDS} for rd in RULE.READS}
        br = {rd: dict(bs) for rd in RULE.READS}
        ks = [(sp, x) for sp in M1.SPLITS for x in SLICES]
        ns = {rd: {k: bs[k] + sl_dl.get(rd, 0.0) for k in RULE.SEEDS} for rd in ks}
        return decide_full(nr, br, ns, {rd: dict(bs) for rd in ks})
    assert full({}, {})["verdict"] == NOT_ADOPTED
    assert full({RULE.READS[4]: 0.03}, {})["verdict"] == ADOPT
    assert full({RULE.READS[4]: 0.03}, {sl4: -0.05})["verdict"] == NOT_ADOPTED
    log("selftest: names and pairs, ISO's verdict picking the base, the four-layer import in its own process (counts, "
        f"parameters, the toy forward within {t['by_hand_max_abs']:.1e} of the by-hand layers, the fourth layer's gradients, the "
        "refusal of a second), the patches and their restoring (also on a failure), the slices from a read file, and both rules "
        "pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "calls", "layers-check"))
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
    if a.cmd == "layers-check":
        return layers_check()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
