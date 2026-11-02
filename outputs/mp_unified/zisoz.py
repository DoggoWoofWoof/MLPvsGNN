"""ISO amendment A1 (docs/ISO_ZG6R_OWN_START.md section 10): ISO section 7's second isolation arm, zg6-z. zg6 (no relation
messages) started from ML1's seed-0 zspk encoder, at seed 0 on both WQ splits, on PKO's roots, on the host's GPU, read against
ML1's zg6r at seed 0. Reported beside; it decides nothing.

zg6-z runs zg6r's route (zml1s.run_arm's zg6r branch: PKO's roots, ML1's encoder start, the card, the seed, the read cap) with
U2n's typed links removed where they are loaded: zu2n.load_typed answers None for every carve, as it does for a passage graph.
zg6r's self-test shows that zg6r's score is then zg6's bit for bit, in eval and in training with the same draws, and that its
construction moves the global generator as zg6's does; so the model trained and read is zg6, with G6r's parameters built and
never used. This module's self-test shows the same on zg6r's toy KB carve, whose typed links are real. zml1s.py, zg6r.py and
zu2n.py are imported and not edited; every patched name is put back afterwards, also on a failure.

    python outputs/mp_unified/zisoz.py fit --name scr-iso-zg6z-s0       (train when needed, then read; kept when read before)
    python outputs/mp_unified/zisoz.py compare-all                      (each zg6-z fit read against ML1's zg6r seed 0)
    python outputs/mp_unified/zisoz.py report                           (outputs/iso/zg6z.json and .md)
    python outputs/mp_unified/zisoz.py --selftest
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
import ml1start as MS  # noqa: E402  (ML1's zspk start read under ML1's names: ISO's amendment A2)

DECLARED = "docs/ISO_ZG6R_OWN_START.md"
STAGE = "ISO"
KEY = "isoz"                                    # the stamp's key in a zg6-z fit's screen.json and read.json and its compares
OUT = "outputs/iso"
ARM = "zg6r"                                    # zrm's arm: the route is zg6r's, with no typed link
TAG = "zg6z"
REMOVED = {"typed_links": "removed", "model": "zg6 (zg6r with no typed link: G6r's parameters built and unused)"}
R5_TOL = 6e-5
GAIN_MIN = 0.002                                # the part kept is shown where zg6r's gain over zspk exceeds this (ISO section 5)


def fit_name(seed, split):
    return f"scr-iso-{TAG}-s{seed}{M1.WQ.SUF[split]}"


def pair_name(seed, split):
    return f"scr-iso-{TAG}-vs-zg6r-s{seed}{M1.WQ.SUF[split]}"


FITS = {fit_name(0, sp): (ARM, 0, sp) for sp in M1.SPLITS}
PAIRS = {pair_name(0, sp): (fit_name(0, sp), M1.fit_name(ARM, 0, sp)) for sp in M1.SPLITS}
assert len(FITS) == 2 and not set(FITS) & set(M1.ALL_FITS)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] zisoz: {msg}", flush=True)


# ── the patches ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def with_fits():
    M1.ALL_FITS.update(FITS)
    try:
        yield
    finally:
        for k in FITS:
            M1.ALL_FITS.pop(k, None)


def z_fit_name(arm, seed, split):
    """zml1s.pko_fit_name with the zg6-z fit being run answering for its own (arm, seed, split)."""
    a = M1._ACTIVE[0]
    if a in FITS and FITS[a] == (arm, seed, split):
        return a
    return M1.fit_name(arm, seed, split)


def no_typed(c, root, cache_root, lk, verify=True):
    """In place of zu2n.load_typed: no typed link on any carve, as on a passage graph."""
    return None


def stamp_removed(orig):
    def stamp(name, env, what, extra=None):
        return orig(name, env, what, {**(extra or {}), **REMOVED} if name in FITS else extra)
    return stamp


@contextlib.contextmanager
def on_names():
    """zml1s with zg6-z's names, stamp key, stage and declaration; restored after. Enough for the compares and the report."""
    sha, ms = M1.WQ.sha_src(__file__), MS.code()
    code = M1.code_shas
    with with_fits(), M1.WQ.patch(M1, DECLARED=DECLARED, STAGE=STAGE, KEY=KEY, pko_fit_name=z_fit_name,
                                  code_shas=lambda arm=None: {**code(arm), "zisoz_sha256": sha, **ms}):
        yield


@contextlib.contextmanager
def on_z(U2N):
    """A zg6-z fit's context: zg6-z's names and stamp, ML1's zspk start read under ML1's names (ml1start's ml1_load in
    zml1s), and no typed link loaded; restored after."""
    with on_names(), M1.WQ.patch(M1, stamp=stamp_removed(M1.stamp), load_encoder=MS.ml1_load), \
            M1.WQ.patch(U2N, load_typed=no_typed):
        yield


def fit(name):
    if name not in FITS:
        raise SystemExit(f"zisoz: {name} is not one of zg6-z's fits ({', '.join(FITS)})")
    G, U2N = M1.import_arm(ARM)
    if G.U2N is not U2N:
        raise SystemExit("zisoz: zg6r's zu2n is not the one patched")
    with on_z(U2N):
        return M1.fit(name)


# ── the compares and the report ──────────────────────────────────────────────


def fit_block(name, what, base=ROOT):
    fr = M1.screen_dir(base) / "fits"
    if name in FITS:
        with on_names():
            return M1.fit_stamp(name, fr, what)[1]
    return M1.fit_stamp(name, fr, what)[1]


def device_of_fit(name, base=ROOT):
    ks, kr = fit_block(name, "screen", base), fit_block(name, "read", base)
    if ks.get("device") != kr.get("device") or kr.get("rows_cap") != M1.ROWS_CAP[ARM]:
        raise SystemExit(f"zisoz: {name}'s train and read are not on one device with zg6r's rows cap")
    return ks["device"]


def is_read(name, base=ROOT):
    return (M1.screen_dir(base) / "fits" / name / "read.json").is_file()


def compare_all(base=ROOT):
    done, ran = [], []
    for cname, (new, bse) in PAIRS.items():
        if not is_read(new, base):
            continue
        js = M1.screen_dir(base) / f"{cname}.json"
        if js.is_file() and (M1.WQ.read_json(js).get(KEY) or {}).get("new") == new:
            done.append(cname)
            continue
        dev = {json.dumps(device_of_fit(f, base), sort_keys=True) for f in (new, bse)}
        if len(dev) != 1:
            raise SystemExit(f"zisoz: {new} and {bse} ran on different devices: {sorted(dev)}")
        with on_names(), M1.on_ml1():
            rc = M1.WQ.run_compare(f"outputs/screen/fits/{new}", f"outputs/screen/fits/{bse}", f"outputs/screen/{cname}")
        if rc:
            raise SystemExit(f"zisoz: the compare {cname} exited {rc}")
        rec = M1.WQ.read_json(js)
        rec[KEY] = {"declared_in": DECLARED, "compare": cname, "new": new, "base": bse, "device": json.loads(dev.pop()),
                    "zisoz_sha256": M1.WQ.sha_src(__file__)}
        M1.WQ.write_json(js, rec)
        ran.append(cname)
    log(f"compare-all: {len(ran)} run, {len(done)} filed before")
    return 0


def r5_z(base=ROOT):
    """({(split, dataset): R@5} of zg6-z at seed 0, the same of ML1's zg6r seed 0 as the compares give it)."""
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
        sp = FITS[new][2]
        for r in rec["by_base"][rec["bases"][0]]["rows"]:
            new_r5[(sp, r["dataset"])] = r["new"][0]
            base_r5[(sp, r["dataset"])] = r["base"][0]
    return new_r5, base_r5


def fit_problems(base=ROOT):
    """Why a zg6-z fit is not the declared one: no typed link in its train and read, ML1's zg6r seed-0 encoder start, the
    device and the trainer's code of ML1's zg6r fit of its split."""
    out = []
    for name, (_a, sd, sp) in FITS.items():
        bse = M1.fit_name(ARM, sd, sp)
        try:
            ks, kr, kb = fit_block(name, "screen", base), fit_block(name, "read", base), fit_block(bse, "screen", base)
        except SystemExit as e:
            out.append(str(e))
            continue
        for what, k in (("train", ks), ("read", kr)):
            if k.get("typed_links") != REMOVED["typed_links"]:
                out.append(f"{name}: its {what} does not name its typed links removed")
        enc, benc = ks.get("encoder") or {}, kb.get("encoder") or {}
        if benc.get("fit") is None or int(enc.get("copies") or 0) < 1 or any(
                enc.get(x) != benc.get(x) for x in ("fit", "seed", "sha256")):
            out.append(f"{name}: its encoder ({enc.get('fit')}, {enc.get('copies')} copies) is not {bse}'s start")
        if ks.get("device") != kb.get("device"):
            out.append(f"{name}: device {ks.get('device')}, {bse}'s {kb.get('device')}")
        cb, cn = kb.get("code") or {}, ks.get("code") or {}
        diff = sorted(k for k in cb if k != "zml1s_sha256" and cn.get(k) != cb[k])
        if diff:
            out.append(f"{name}: code differs from {bse}'s in {diff}")
    return out


def beside_rows(new, base0, ml1):
    """Per read: zg6-z, zg6r and zspk at seed 0, zg6-z's delta from zg6r with zg6r's seed floor beside it, and the part of
    zg6r's gain over zspk that zg6-z keeps, (zg6-z - zspk) / (zg6r - zspk), where that gain exceeds GAIN_MIN. Problems when
    a value is missing or zg6r's seed 0 is not the same in both stages."""
    rows, bad = [], []
    for rd in RULE.READS:
        g, z = ml1.get(ARM, {}).get(rd, {}), ml1.get("zspk", {}).get(rd, {})
        n, b0 = new.get(rd), base0.get(rd)
        if n is None or b0 is None or any(k not in g for k in RULE.SEEDS) or 0 not in z:
            bad.append(f"{rd[0]} {rd[1]}: a value is missing")
            continue
        if abs(b0 - g[0]) > R5_TOL:
            bad.append(f"{rd[0]} {rd[1]}: zg6r seed 0 is {b0} in zg6-z's compare, {g[0]} in ML1's")
        s = RULE.spread(g)
        gain = g[0] - z[0]
        rows.append({"split": rd[0], "dataset": rd[1], "zg6z": round(n, 4), "zg6r": round(g[0], 4), "zspk": round(z[0], 4),
                     "delta": round(n - g[0], 4), "floor": round(RULE.pair_floor([s, s]), 4),
                     "kept": round((n - z[0]) / gain, 3) if gain > GAIN_MIN else None})
    return rows, bad


def report(base=ROOT):
    new, base0 = r5_z(base)
    rows, bad = beside_rows(new, base0, M1.r5_by_arm(base))
    bad += fit_problems(base)
    rec = {"stage": STAGE, "declared_in": DECLARED, "amendment": "A1 (section 10)", "arm": "zg6-z: " + REMOVED["model"],
           "decides": "nothing (ISO section 7)", "complete": not bad, "rows": rows, "problems": bad,
           "zisoz_sha256": M1.WQ.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(base) / OUT
    out.mkdir(parents=True, exist_ok=True)
    M1.WQ.write_json(out / "zg6z.json", rec)
    md = ["# ISO A1: zg6-z (zg6, no relation messages) beside zg6r, seed 0, on PKO's pools", "",
          "Reported beside; decides nothing. kept = (zg6-z - zspk) / (zg6r - zspk) where zg6r's gain exceeds "
          f"{GAIN_MIN}; floor = zg6r's seed floor, for scale.", "",
          "| split | read | zspk s0 | zg6r s0 | zg6-z s0 | delta | floor | kept |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {r['split']} | {r['dataset']} | {r['zspk']:.4f} | {r['zg6r']:.4f} | {r['zg6z']:.4f} | {r['delta']:+.4f} | "
           f"{r['floor']:.4f} | {'' if r['kept'] is None else format(r['kept'], '.3f')} |" for r in rows]
    if bad:
        md += ["", "INCOMPLETE:"] + [f"- {p}" for p in bad]
    (out / "zg6z.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"report: {len(rows)} reads, {'complete' if not bad else f'{len(bad)} problems'}; {OUT}/zg6z.md")
    return 0


# ── self-test (laptop or host, CPU, no fit) ──────────────────────────────────


def toy_same_as_zg6(G):
    """zg6r's toy KB carve (its self-test's steps 2 and 3) built twice, with its typed links and with them removed by
    no_typed; zg6 and zg6r from seed 0 with the same zg6 state: removed, zg6r's score is zg6's bit for bit in eval and in
    training with the same draws; with the links, the scores differ."""
    import tempfile
    import numpy as np
    import torch
    G6 = G.G6
    G.LG.bind_device_ops()
    with tempfile.TemporaryDirectory(prefix="zisoz_") as td:
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
            ct = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
            with M1.WQ.patch(G.U2N, load_typed=no_typed):
                cz = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        qs = np.r_[np.arange(ct.rows)[::-1][:7], np.arange(ct.rows)[:5]].astype(np.int64)
        B = qs.size
        ft, nq, bz, _g = ct.batch(qs, blocks)
        fz, nqz, bzz, _g = cz.batch(qs, blocks)
        keep = torch.ones(B, len(blocks))
        nets = {}
        for k, cl in (("zg6", G6.ZG6), (ARM, G.ZG6R)):
            torch.manual_seed(0)
            nets[k] = cl(blocks, ct.widths, 16, dropout=0.1, seed=0)
        m, z6 = nets[ARM], nets["zg6"]
        gen = torch.Generator().manual_seed(3)
        with torch.no_grad():
            for n in tuple(G6.G6_NAMES) + tuple(G.G6R_NAMES):
                getattr(m, n).copy_(torch.randn(getattr(m, n).shape, generator=gen) * 0.3)
            z6.load_state_dict({k: x for k, x in m.state_dict().items() if k not in G.G6R_NAMES})
        out = {"typed_edges_with": int((ft[G.G6R_KEY]["ci"] >= 0).sum()), "typed_edges_removed":
               int((fz[G.G6R_KEY]["ci"] >= 0).sum()), "rel_removed_none": fz[G.G6R_KEY]["rel"] is None,
               "typed_none": cz.u2n_typed is None, "same_rows": bool(torch.equal(nq, nqz) and torch.equal(bz, bzz))}
        for x in nets.values():
            x.eval()
        with torch.no_grad():
            out["eval_equal_removed"] = bool(torch.equal(m(fz, keep, nq, B, bz), z6(fz, keep, nq, B, bz)))
            out["eval_differs_with"] = not bool(torch.allclose(m(ft, keep, nq, B, bz), z6(ft, keep, nq, B, bz)))
        ys = {}
        for k, x in nets.items():
            x.train()
            x.g6_gen.manual_seed(77)
            torch.manual_seed(5)
            ys[k] = x(fz, keep, nq, B, bz)
        out["train_equal_removed"] = bool(torch.equal(ys[ARM], ys["zg6"]))
        return out


def selftest():
    import tempfile
    import types
    assert fit_name(0, "WQ-musique") == "scr-iso-zg6z-s0" and fit_name(0, "WQ-hotpotqa") == "scr-iso-zg6z-s0-hp"
    assert PAIRS[pair_name(0, "WQ-hotpotqa")] == ("scr-iso-zg6z-s0-hp", "scr-ml1-zg6r-s0-hp")
    import ziso
    import zdep
    assert not set(FITS) & (set(ziso.FITS) | set(zdep.FITS)) and KEY not in (ziso.KEY, zdep.KEY, M1.KEY)
    # 1. the patches, on a fake zu2n, and their restoring (also on a failure)
    n0 = len(M1.ALL_FITS)
    saved = (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder)
    U = types.SimpleNamespace(load_typed="orig")
    seen = []
    orig_stamp = M1.stamp
    try:
        M1.stamp = lambda name, env, what, extra=None: seen.append((name, what, extra))
        with on_z(U):
            assert len(M1.ALL_FITS) == n0 + 2 and (M1.DECLARED, M1.STAGE, M1.KEY) == (DECLARED, STAGE, KEY)
            assert U.load_typed is no_typed and M1.load_encoder is MS.ml1_load and "zisoz_sha256" in M1.code_shas(ARM)
            assert "ml1start_sha256" in M1.code_shas(ARM)
            M1._ACTIVE[0] = fit_name(0, "WQ-musique")
            assert M1.pko_fit_name(ARM, 0, "WQ-musique") == "scr-iso-zg6z-s0"
            assert M1.pko_fit_name("zspk", 0, "WQ-musique") == "scr-ml1-zspk-s0"
            M1._ACTIVE[0] = None
            M1.stamp(fit_name(0, "WQ-hotpotqa"), {}, "read", {"x": 1})
            M1.stamp("scr-ml1-zg6r-s0", {}, "read", {"x": 2})
        assert seen == [("scr-iso-zg6z-s0-hp", "read", {"x": 1, **REMOVED}), ("scr-ml1-zg6r-s0", "read", {"x": 2})]
        try:
            with on_z(U):
                raise RuntimeError("inside")
        except RuntimeError:
            pass
    finally:
        M1.stamp = orig_stamp
    assert (M1.DECLARED, M1.STAGE, M1.KEY, M1.pko_fit_name, M1.stamp, M1.code_shas, M1.load_encoder) == saved
    assert len(M1.ALL_FITS) == n0 and U.load_typed == "orig"
    # 2. the real modules: zu2n's loader is what zg6r's carve calls, through zg6r's own zu2n
    G, U2N = M1.import_arm(ARM)
    import inspect
    src = inspect.getsource(G.g6r_carve)
    assert G.U2N is U2N and "U2N.load_typed(" in src and list(inspect.signature(U2N.load_typed).parameters) == [
        "c", "root", "cache_root", "lk", "verify"]
    # 3. on zg6r's toy KB carve: removed, zg6r is zg6 bit for bit; with the links, it is not
    t = toy_same_as_zg6(G)
    assert t["typed_edges_with"] > 0 and t["typed_edges_removed"] == 0 and t["rel_removed_none"] and t["typed_none"], t
    assert t["same_rows"] and t["eval_equal_removed"] and t["eval_differs_with"] and t["train_equal_removed"], t
    assert U2N.load_typed is not no_typed
    # 4. the report's rows on made-up values: the part kept, the floor, a missing value, a base that disagrees
    rd0, rd1 = RULE.READS[0], RULE.READS[1]
    ml1 = {ARM: {rd: {0: 0.60, 1: 0.61, 2: 0.59} for rd in RULE.READS}, "zspk": {rd: {0: 0.50} for rd in RULE.READS}}
    new = {rd: 0.55 for rd in RULE.READS}
    base0 = {rd: 0.60 for rd in RULE.READS}
    rows, bad = beside_rows(new, base0, ml1)
    assert not bad and len(rows) == 12 and all(r["kept"] == 0.5 and r["floor"] == 0.02 and r["delta"] == -0.05 for r in rows)
    ml1["zspk"][rd1] = {0: 0.599}
    rows, bad = beside_rows({**new, rd0: None}, {**base0, rd1: 0.6002}, ml1)
    assert len(rows) == 11 and len(bad) == 2 and [r["kept"] for r in rows if (r["split"], r["dataset"]) == rd1] == [None]
    with tempfile.TemporaryDirectory() as td:
        assert r5_z(td) == ({}, {}) and not is_read(fit_name(0, "WQ-musique"), td)
    log("selftest: names and pairs, the patches and their restoring (also on a failure), zg6r's carve calling the patched "
        "zu2n loader, zg6r's toy KB carve (typed links removed: zg6's score bit for bit in eval and training; with them: "
        f"differs, {t['typed_edges_with']} typed edges), and the report's rows pass")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("fit", "compare-all", "report"))
    ap.add_argument("--name")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "fit":
        return fit(a.name)
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "report":
        return report()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
