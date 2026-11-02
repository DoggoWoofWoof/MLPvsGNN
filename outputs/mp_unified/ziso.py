"""ISO's stage (docs/ISO_ZG6R_OWN_START.md): zg6r-m, zg6r started from its own drawn encoder instead of the encoder of ML1's
seed-0 zspk fit, at three seeds on both WQ splits, on PKO's roots, on the host's GPU; its compares against ML1's zg6r fits of
the same seed and split; the verdict.

zml1s.py is imported and not edited. Its fit, train and read run with ISO's six names added to its fit table, its stamp key,
stage and declaration set to ISO's for the length of an ISO fit, its encoder load replaced by an own-start record (no tensors),
and zg6r's encoder_used check replaced by its converse (no zspk tensor was copied into any zg6r). Every patched name is put
back afterwards, also on a failure. The model, its trainer, its inputs, PKO's roots and the read cap are ML1's zg6r's.

    python outputs/mp_unified/ziso.py fit --name scr-iso-zg6rm-s0      (train when needed, then read; kept when read before)
    python outputs/mp_unified/ziso.py compare-all                      (the six compares against ML1's zg6r fits, CPU)
    python outputs/mp_unified/ziso.py verdict                          (outputs/iso/verdict.json and .md)
    python outputs/mp_unified/ziso.py --selftest
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

import zml1 as RULE  # noqa: E402  (stdlib only)
import zml1s as M1  # noqa: E402

DECLARED = "docs/ISO_ZG6R_OWN_START.md"
STAGE = "ISO"
KEY = "iso"                                     # the stamp's key in an ISO fit's screen.json and read.json and ISO's compares
OUT = "outputs/iso"
ARM = "zg6r"                                    # zrm's arm: the model is zg6r's; only the encoder's start differs
TAG = "zg6rm"
OWN = {"start": "own", "fit": None, "elements": 0,
       "note": "zg6r's encoder drawn by zrm's init at the fit's seed; no zspk tensor copied"}


def fit_name(seed, split):
    return f"scr-iso-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-iso-{TAG}-vs-zg6r-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(sd, sp): (ARM, sd, sp) for sd in RULE.SEEDS for sp in M1.SPLITS}
PAIRS = {pair_name(sd, sp): (fit_name(sd, sp), M1.fit_name(ARM, sd, sp)) for sd in RULE.SEEDS for sp in M1.SPLITS}
assert len(FITS) == 6 and len(PAIRS) == 6 and not set(FITS) & set(M1.ALL_FITS)
ML1_KEY, ML1_DECLARED = M1.KEY, M1.DECLARED
REPLACE, KEEP, INCOMPLETE = "REPLACE", "KEEP", "INCOMPLETE"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] ziso: {msg}", flush=True)


@contextlib.contextmanager
def with_fits():
    """ISO's names in zml1s's fit table (in place, so on_ml1's FITS sees them); removed afterwards."""
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def iso_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the ISO fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def own_load(split):
    """In place of zml1s.load_encoder: no tensors (zg6r then keeps its own draw) and the record of that."""
    return None, {**OWN, "split": split}


def own_used(name, enc, out_root=None):
    """In place of zg6r.encoder_used: the fit trained, and no zspk tensor was copied into any zg6r built meanwhile."""
    if enc.get("start") != "own" or enc.get("copies", -1) != 0:
        raise SystemExit(f"ziso: {name}: the encoder's start is {enc.get('start')} with {enc.get('copies')} copies, "
                         f"not its own draw with none")
    p = Path(out_root or M1.WQ.fit_dir(name).parent) / name / "models.pt"
    if not p.is_file():
        raise SystemExit(f"ziso: {name}: no models.pt after the train")


@contextlib.contextmanager
def on_iso(G=None):
    """zml1s (and zg6r, when given) with ISO's names, stamp, declaration and own start; restored after."""
    sha = M1.WQ.sha_src(__file__)
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=iso_fit_name,
                                  load_encoder=own_load,
                                  code_shas=lambda arm=None: {**code(arm), "ziso_sha256": sha}), \
            (M1.WQ.patch(G, encoder_used=own_used) if G is not None else contextlib.nullcontext()):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"ziso: {name} is not one of ISO's fits ({', '.join(FITS)})")
    G, _U2N = M1.import_arm(ARM)
    with on_iso(G):
        return M1.fit(name)


# ── the compares and the verdict ─────────────────────────────────────────────


def fit_block(name, what, base=ROOT):
    """A fit's stamp block: ISO's for ISO's fits, ML1's for ML1's."""
    fr = M1.screen_dir(base) / "fits"
    if name in FITS:
        with on_iso():
            return M1.fit_stamp(name, fr, what)[1]
    return M1.fit_stamp(name, fr, what)[1]


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"ziso: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def compare_all(base=ROOT):
    """ISO's six compares, each fit against ML1's zg6r fit of the same seed and split, through zwq's run_compare."""
    done, ran = [], []
    for cname, (new, bse) in PAIRS.items():
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"ziso: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_iso(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"ziso: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "ziso_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_pair(base=ROOT):
    """({(split, dataset): {seed: R@5}} of zg6r-m, the same of zg6r) from ISO's compares; one missing leaves its reads out."""
    new_r5, base_r5 = {}, {}
    for cname, (new, bse) in PAIRS.items():
        p = M1.screen_dir(base) / f"{cname}.json"
        if not p.is_file():
            continue
        rec = M1.WQ.read_json(p)
        k = rec.get(KEY) or {}
        if (k.get("declared_in"), k.get("new"), k.get("base")) != (DECLARED, new, bse):
            continue
        if rec.get("candidate") != "p@swa" or rec.get("carve") != "s1eval" or len(rec.get("bases") or []) != 1:
            continue
        _a, sd, sp = FITS[new]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5.setdefault((sp, r["dataset"]), {})[sd] = r["new"][0]
            base_r5.setdefault((sp, r["dataset"]), {})[sd] = r["base"][0]
    return new_r5, base_r5


def decide(new, base):
    """ISO section 4. ML1's calls and pair floors (zml1.verdict) on zg6r-m against zg6r; a read is ISO's LOSS when ML1
    calls it a LOSS or the mean of its three deltas is at or below minus its floor. REPLACE: no ISO LOSS on any read."""
    v = RULE.verdict(new, base)
    if v["verdict"] == "INCOMPLETE":
        return {"verdict": INCOMPLETE, "problems": v["problems"], "reads": [], "counts": {}}
    for r in v["reads"]:
        rd = (r["split"], r["dataset"])
        m = sum(new[rd][k] - base[rd][k] for k in RULE.SEEDS) / len(RULE.SEEDS)
        r["mean_delta"] = round(m, 4)
        r["iso_loss"] = r["call"] == RULE.LOSS or m <= -r["floor"] + RULE.EPS
    n = dict(v["counts"], ISO_LOSS=sum(1 for r in v["reads"] if r["iso_loss"]))
    return {"verdict": REPLACE if n["ISO_LOSS"] == 0 else KEEP, "counts": n, "problems": [], "reads": v["reads"]}


def fit_problems(base=ROOT):
    """Why an ISO fit is not the declared one: its own start, its device and its trainer's code against ML1's zg6r fit."""
    out = []
    for new, (_a, sd, sp) in FITS.items():
        bse = M1.fit_name(ARM, sd, sp)
        try:
            kn, kb = fit_block(new, "screen", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        enc = kn.get("encoder") or {}
        if enc.get("start") != "own" or enc.get("copies") != 0:
            out.append(f"{new}: its encoder's start is {enc.get('start')} with {enc.get('copies')} copies")
        if kn.get("device") != kb.get("device"):
            out.append(f"{new}: device {kn.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, kn.get("code") or {}
        diff = sorted(k for k in cb if k != "zml1s_sha256" and cn.get(k) != cb[k])
        if diff:
            out.append(f"{new}: code differs from {bse}'s in {diff}")
    return out


def beside(new, base=ROOT):
    """Section 5: per read, the seed means of zspk, zg6r (ML1's compares) and zg6r-m, and the part of zg6r's gain over zspk
    that zg6r-m keeps, (zg6r-m - zspk) / (zg6r - zspk), where that gain exceeds 0.002. Decides nothing."""
    ml1 = M1.r5_by_arm(base)
    g, s = ml1.get(ARM, {}), ml1.get(M1.BASE_OF[ARM], {})
    mean = lambda d: sum(d.values()) / len(d)  # noqa: E731
    out = []
    for rd in RULE.READS:
        if rd in new and rd in g and rd in s and new[rd] and g[rd] and s[rd]:
            den = mean(g[rd]) - mean(s[rd])
            out.append({"split": rd[0], "dataset": rd[1], "zspk": round(mean(s[rd]), 4), "zg6r": round(mean(g[rd]), 4),
                        "zg6r_m": round(mean(new[rd]), 4),
                        "kept": round((mean(new[rd]) - mean(s[rd])) / den, 3) if den > 0.002 else None})
    return out


def verdict(base=ROOT):
    new, bse = r5_pair(base)
    v = decide(new, bse)
    bad = fit_problems(base)
    if bad and v["verdict"] != INCOMPLETE:
        v = {**v, "verdict": INCOMPLETE}
    v["problems"] = v.get("problems", []) + bad
    rec = {"stage": STAGE, "declared_in": DECLARED, "arm": "zg6r-m", "base": "zg6r (ML1)", **v, "beside": beside(new, base),
           "ziso_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "verdict.json", rec)
    md = ["# ISO: zg6r-m (zg6r from its own encoder start) against zg6r (from zspk's), on PKO's pools", "",
          f"## {rec['verdict']}", "", f"counts: {json.dumps(rec.get('counts', {}))}", ""]
    if rec["reads"]:
        md += ["| split | dataset | d0 | d1 | d2 | mean | floor | ML1 call | ISO loss |", "|---|---|---:|---:|---:|---:|---:|---|---|"]
        for r in rec["reads"]:
            d = r["delta"]
            md.append(f"| {r['split']} | {r['dataset']} | {d[0]:+.4f} | {d[1]:+.4f} | {d[2]:+.4f} | {r['mean_delta']:+.4f} | "
                      f"{r['floor']:.4f} | {r['call']} | {'yes' if r['iso_loss'] else 'no'} |")
    if rec["beside"]:
        md += ["", "Beside, deciding nothing (seed means; kept = (zg6r-m - zspk) / (zg6r - zspk)):", "",
               "| split | dataset | zspk | zg6r | zg6r-m | kept |", "|---|---|---:|---:|---:|---:|"]
        md += [f"| {b['split']} | {b['dataset']} | {b['zspk']:.4f} | {b['zg6r']:.4f} | {b['zg6r_m']:.4f} | "
               f"{'' if b['kept'] is None else b['kept']} |" for b in rec["beside"]]
    if rec["problems"]:
        md += ["", "problems:"] + [f"- {p}" for p in rec["problems"]]
    (out / "verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"verdict {rec['verdict']} {json.dumps(rec.get('counts', {}))}; {OUT}/verdict.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def selftest():
    import types
    assert sorted(FITS) == sorted(f"scr-iso-zg6rm-s{k}{s}" for k in (0, 1, 2) for s in ("", "-hp"))
    assert all(PAIRS[c][1] in M1.FITS and M1.FITS[PAIRS[c][1]] == FITS[PAIRS[c][0]] for c in PAIRS)
    n0, saved = len(M1.ALL_FITS), (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.load_encoder, M1.code_shas)
    G = types.SimpleNamespace(encoder_used="orig")
    with on_iso(G):
        assert len(M1.ALL_FITS) == n0 + 6 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
        assert G.encoder_used is own_used and M1.load_encoder("WQ-musique") == (None, {**OWN, "split": "WQ-musique"})
        assert "ziso_sha256" in M1.code_shas("zg6r") and "zg6r_sha256" in M1.code_shas("zg6r")
        M1._ACTIVE[0] = "scr-iso-zg6rm-s1-hp"
        assert M1.pko_fit_name("zg6r", 1, "WQ-hotpotqa") == "scr-iso-zg6rm-s1-hp"
        assert M1.pko_fit_name("zspk", 0, "WQ-hotpotqa") == "scr-ml1-zspk-s0-hp"
        assert M1.pko_fit_name("zg6r", 0, "WQ-hotpotqa") == "scr-ml1-zg6r-s0-hp"
        M1._ACTIVE[0] = None
        for bad in ({"start": "own", "copies": 1}, {"start": None, "copies": 0}):
            try:
                own_used("x", bad, "nowhere")
                raise AssertionError("own_used passed a copied start")
            except SystemExit:
                pass
    assert len(M1.ALL_FITS) == n0 and G.encoder_used == "orig"
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.load_encoder, M1.code_shas) == saved
    try:
        with on_iso(G):
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert len(M1.ALL_FITS) == n0 and G.encoder_used == "orig" and M1.KEY == ML1_KEY
    # the rule on made-up reads: flat (REPLACE), one seed-0 LOSS (KEEP), a mean LOSS seed 0 does not call (KEEP), a hole
    base = {rd: {0: 0.50, 1: 0.52, 2: 0.48} for rd in RULE.READS}
    same = {rd: dict(v) for rd, v in base.items()}
    assert decide(same, base)["verdict"] == REPLACE
    one = {rd: dict(v) for rd, v in base.items()}
    rd0 = RULE.READS[3]
    one[rd0] = {0: 0.40, 1: 0.42, 2: 0.38}                      # every seed 0.10 down; floor 0.04
    v = decide(one, base)
    assert v["verdict"] == KEEP and v["counts"]["ISO_LOSS"] == 1 and v["counts"]["LOSS"] == 1
    base2 = {rd: dict(v) for rd, v in base.items()}
    base2[rd0] = {0: 0.50, 1: 0.50, 2: 0.50}
    mean = {rd: dict(v) for rd, v in base2.items()}
    mean[rd0] = {0: 0.494, 1: 0.490, 2: 0.490}                 # seed 0 -0.006 (not ML1's LOSS), mean -0.0087, floor 0.0075
    v = decide(mean, base2)
    r = [x for x in v["reads"] if (x["split"], x["dataset"]) == rd0][0]
    assert r["call"] == RULE.WITHIN and r["iso_loss"] and r["floor"] == 0.0075 and v["verdict"] == KEEP
    mean[rd0] = {0: 0.494, 1: 0.494, 2: 0.494}                 # mean -0.006: within
    assert decide(mean, base2)["verdict"] == REPLACE
    up = {rd: {k: x + 0.05 for k, x in v_.items()} for rd, v_ in base.items()}      # above the 0.04 floor
    v = decide(up, base)
    assert v["verdict"] == REPLACE and v["counts"]["GAIN"] == 12
    hole = {rd: dict(v_) for rd, v_ in base.items()}
    del hole[rd0][2]
    assert decide(hole, base)["verdict"] == INCOMPLETE
    log("selftest: names, pairs, the patches and their restoring (also on a failure), own_used, and the rule pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "verdict"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "verdict":
        return verdict()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
