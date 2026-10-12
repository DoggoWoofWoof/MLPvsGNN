"""P1c's screen (docs/P1C_POOLS_RETRAINED.md): the MLP base (zrc) and the GNN base (zsp), unchanged, refit on P1c's pools
(P1b2's chosen construction at each question's frozen size, scripts/p1c_pools.py) on today's graph. Every command is
zrct.py's, zg1.py's or rmatch.py's with P1c's roots in place of today's:
    looks   outputs/p1c/look   (outputs/mp_unified/look_p1c.py)
    caches  outputs/p1c/cache  (lean_cache on those looks; --cache-root, required on train and read)
    chains  outputs/p1c/chains/rmatch (rmatch's build) and outputs/p1c/chains/zrc (zrc's, against rmatch's)
    links   outputs/p1c/links  (zlink's build: the pool edges inside each P1c pool)
The relation tables are today's (rmatch's REL_DIR): the graph does not change. The models, arm names (zrc, zsp),
settings, seeds and batches are zrct's and zprop's. Four fits on the card: zrc and zsp on L-musique and L-hotpotqa, each
read on the six s1eval carves of P1c's pools and compared with its own arm's card fit of the split on today's pools
(zrc: scr-zrct, scr-zrct-hp; zsp: scr-zspg, scr-zspg-hp), the same questions and gold totals; R@5 over the question's
golds, so a gold the pools lose counts as a miss. The pair and its re-call under the seed null are relz's, decided
against those fits and stamped as P1c's.

    python outputs/mp_unified/zp1c.py build --what rmatch|zrc|links --dataset metaqa --carve fit --host
    python outputs/mp_unified/zp1c.py train --split L-musique --name scr-p1c-zrc --arm zrc --device cuda --host \\
        --cache-root outputs/p1c/cache
    python outputs/mp_unified/zp1c.py read --name scr-p1c-zrc --device cuda --host --cache-root outputs/p1c/cache
    python outputs/mp_unified/zp1c.py compare --new outputs/screen/fits/scr-p1c-zrc \\
        --base outputs/screen/fits/scr-zrct,outputs/step1/fits/L-musique --out outputs/screen/scr-p1c-zrc
    python outputs/mp_unified/zp1c.py pair --arm zrc --screens outputs/screen/scr-p1c-zrc.json,... --out ...
    python outputs/mp_unified/zp1c.py recall --arm zrc --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zp1c.py --selftest
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

import zprop as ZP  # noqa: E402  (sets the cuBLAS workspace before torch loads)
import zrc as ZC  # noqa: E402

ZL, ZM, RM = ZP.ZL, ZP.ZM, ZP.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
log = S.log

DECLARED = "docs/P1C_POOLS_RETRAINED.md"
OUT = ROOT / "outputs" / "p1c"
LOOK, CACHE = OUT / "look", OUT / "cache"
CH_RM, CH_ZRC, LINKS = OUT / "chains" / "rmatch", OUT / "chains" / "zrc", OUT / "links"
GATE = OUT / "gate.json"
DEVICE = "cuda"
ARMS = {"zrc": (ZM.ZRM, ZC.ChainCarveZRC), "zsp": (ZP.ZProp, ZL.LinkCarveBase)}
BASE_FITS = {"zrc": {"L-musique": ("screen", "fits", "scr-zrct"), "L-hotpotqa": ("screen", "fits", "scr-zrct-hp")},
             "zsp": {"L-musique": ("screen", "fits", "scr-zspg"), "L-hotpotqa": ("screen", "fits", "scr-zspg-hp")}}
BASE_SCREENS = {"zrc": {sp: f"outputs/u1c/base-zrct-{sp}.json" for sp in BASE_FITS["zrc"]},
                "zsp": {sp: f"outputs/zg1/base-zspg-{sp}.json" for sp in BASE_FITS["zsp"]}}
AGAINST = {"zrc": "zrc's card fit of each split on today's pools (scr-zrct)",
           "zsp": "zsp's card fit of each split on today's pools (scr-zspg)"}
_BUILT = RM.built_records
_ZL_BUILT = ZL.built_records


# ── P1c's roots ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_p1c():
    """rmatch's, zrc's and zlink's carves (and their records) read P1c's chains and links; restored after. zlink's
    carve extends rmatch's, so it reads rmatch's chains. The relation tables are today's."""
    saved = (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT, RM.built_records,
             ZL.built_records)
    RM.ChainCarveBase.CH_ROOT = CH_RM
    ZC.ChainCarveZRC.CH_ROOT = CH_ZRC
    ZL.LinkCarveBase.LK_ROOT = LINKS
    RM.built_records = lambda root=CH_RM: _BUILT(root)
    ZL.built_records = lambda root=LINKS: _ZL_BUILT(root)
    try:
        yield
    finally:
        (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT, RM.built_records,
         ZL.built_records) = saved


def cache_root_of(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cache-root")
    k, _ = ap.parse_known_args(argv)
    if not k.cache_root or Path(k.cache_root).resolve() != CACHE.resolve():
        raise SystemExit(f"zp1c: train and read take --cache-root {CACHE.relative_to(ROOT).as_posix()}")


def on_card(argv):
    dev, _t = ZP.device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zp1c: P1c trains and reads on the {DEVICE}, as its bases did, not {dev}")
    return list(argv) if "--device" in argv else list(argv) + ["--device", DEVICE]


def gate_passed(gate=GATE):
    if not Path(gate).exists():
        raise SystemExit(f"zp1c: {gate} missing (scripts/p1c_pools.py gate)")
    rec = json.loads(Path(gate).read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("declared_in") != DECLARED:
        raise SystemExit(f"zp1c: {gate} is {rec.get('verdict')}, not P1c's PASS")
    return rec


# ── build, train, read, compare ──────────────────────────────────────────────


def build(what, ds, carve, host):
    placement = {"where": "host" if host else "laptop"}
    if what == "rmatch":
        RM.build(ds, carve, CH_RM, cache_root=CACHE, look_root=LOOK, placement=placement)
    elif what == "zrc":
        ZC.build(ds, carve, CH_ZRC, cache_root=CACHE, look_root=LOOK, placement=placement, rm_root=CH_RM)
    elif what == "links":
        ZL.build(ds, carve, LINKS, cache_root=CACHE, look_root=LOOK, placement=placement)
    else:
        raise SystemExit(f"zp1c build: --what rmatch, zrc or links, not {what}")
    return 0


def stamp(argv, arm):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = gate_passed()
        rec.update({"zp1c_sha256": LC.sha_src(__file__), "p1c": {
            "declared_in": DECLARED, "arm": arm, "device": DEVICE, "graph": "today's",
            "pools": f"P1b2's {g.get('construction')} at each question's frozen size (outputs/p1c/pools)",
            "cache_root": CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(GATE),
            "rmatch_records": RM.built_records(), "zrc_records": _BUILT(CH_ZRC),
            "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS:
        raise SystemExit(f"zp1c: train takes --arm zrc or zsp, not {arm}")
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zp1c: the arm {arm} is {S.ARMS.get(arm)}, not {ARMS[arm]}")
    if split not in BASE_FITS[arm]:
        raise SystemExit(f"zp1c: P1c's screen fits L-musique and L-hotpotqa, not {split}")
    cache_root_of(argv)
    argv = on_card(argv)
    gate_passed()
    with on_p1c():
        rc = RM.train(argv, split)
        stamp(argv, arm)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    rec = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))
    if rec.get("arm") not in ARMS or "p1c" not in rec:
        raise SystemExit(f"zp1c: {name} is not a P1c fit (arm {rec.get('arm')})")
    cache_root_of(argv)
    with on_p1c():
        return RM.read(on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its own card fit ───────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zp1c: no base fit of {arm} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, arm, fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b = {"zrc": "scr-zrct", "zsp": "scr-zspg"}[arm]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({arm}'s card fits {b} and {b}-hp, today's pools)"),
            ("(rel's screen fits)", f"({arm}'s card fits on today's pools)"),
            ("| rel R@5 | relz R@5 |", f"| {arm} R@5 (today's) | {arm} R@5 (P1c) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"P1c ({DECLARED}): {arm} on P1c's pools against {arm} on today's.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "P1c", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": DEVICE,
                    "zp1c_sha256": LC.sha_src(__file__)})
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


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    t0 = time.time()
    for arm, want in ARMS.items():
        assert S.ARMS[arm] == want, arm
    assert ZL.LinkCarveBase.__mro__[1] is RM.ChainCarveBase
    before = (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT)
    rel_before = (RM.ChainCarveBase.REL_DIR, ZC.ChainCarveZRC.REL_DIR, ZL.LinkCarveBase.REL_DIR)
    with on_p1c():
        assert (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT) == (CH_RM, CH_ZRC, LINKS)
        assert ZL.LinkCarveBase.CH_ROOT == CH_RM          # zsp's chains are rmatch's, through the class it extends
        assert (RM.ChainCarveBase.REL_DIR, ZC.ChainCarveZRC.REL_DIR, ZL.LinkCarveBase.REL_DIR) == rel_before
    assert (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT) == before
    assert rel_before == (RM.REL_DIR,) * 3
    assert RM.built_records is _BUILT and ZL.built_records is _ZL_BUILT
    good = ["train", "--arm", "zrc", "--name", "x", "--cache-root", "outputs/p1c/cache"]
    SR.must_stop(train, ["train", "--arm", "zrm", "--name", "x", "--cache-root", "outputs/p1c/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, ["train", "--arm", "zsp", "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zsp", "--name", "x", "--cache-root", "outputs/u1c/cache"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    SR.must_stop(gate_passed, OUT / "no-such-gate.json")
    SR.must_stop(build, "nothing", "metaqa", "fit", False)
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        g = T / "g.json"
        LC.write_json(g, {"verdict": "FAIL", "declared_in": DECLARED})
        SR.must_stop(gate_passed, g)
        LC.write_json(g, {"verdict": "PASS", "declared_in": "docs/U1C_RETRAIN_ON_U.md"})
        SR.must_stop(gate_passed, g)
        LC.write_json(g, {"verdict": "PASS", "declared_in": DECLARED})
        assert gate_passed(g)["verdict"] == "PASS"
        (T / "fits" / "y").mkdir(parents=True)
        LC.write_json(T / "fits" / "y" / "screen.json", {"arm": "zrc", "u1c": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T / "fits"), "--cache-root", "outputs/p1c/cache"])
        # relz's pair and re-call for each arm against its own card fits, relz restored after
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARMS:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=arm)) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", f"scr-p1c-{arm}", "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", f"scr-p1c-{arm}-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(T / arm / "z2", f"scr-p1c-{arm}", "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="outputs/screen/fits/scr-zrm")
            SR.must_stop(pair, arm, [zw, zh], T / arm / "bad")
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm, pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "P1c" and pj["decided_against"] == AGAINST[arm]
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"P1c ({DECLARED})") and f"| {arm} R@5 (today's) | {arm} R@5 (P1c) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zp1c selftest: arms zrc and zsp as registered; P1c's chain and link roots in place and restored, today's "
        f"relation tables untouched; train takes no other arm, split, cache root or device and needs P1c's gate PASS; "
        f"read refuses a fit that is not P1c's; relz's pair and re-call run against each arm's card fits "
        f"({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("--what", required=True)
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(rest)
        return build(b.what, b.dataset, b.carve, b.host)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=sorted(ARMS))
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
            log(f"zp1c {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zp1c: build, train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
