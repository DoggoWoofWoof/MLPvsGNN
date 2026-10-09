"""U1c's screen (docs/U1C_RETRAIN_ON_U.md): the MLP base (zrc) and the GNN base (zsp), unchanged, refit on U1d's
universal graph and step 4h's chosen pools. Every command is zrct.py's, zg1.py's or rmatch.py's with U1c's roots in place
of today's:
    looks   outputs/u1c/look   (outputs/mp_unified/look_u1c.py)
    caches  outputs/u1c/cache  (lean_cache on those looks; --cache-root, required on train and read)
    chains  outputs/u1c/chains/rmatch (rmatch's build) and outputs/u1c/chains/zrc (zrc's, against rmatch's)
    links   outputs/u1c/links  (zlink's build: the pool edges, now U1d's structural_U inside each pool)
The models, arm names (zrc, zsp), settings, seeds and batches are zrct's and zprop's. Four fits on the card: zrc and zsp
on L-musique and L-hotpotqa, each read on the six s1eval carves of U1c's pools and compared with its own arm's card fit
of the split on today's graph and pools (zrc: scr-zrct, scr-zrct-hp; zsp: scr-zspg, scr-zspg-hp, round thirty-one's),
the same questions and gold totals; R@5 over the question's golds. The pair and its re-call under the seed null are
relz's, decided against those fits and stamped as U1c's.

    python outputs/mp_unified/zu1c.py build --what rmatch|zrc|links --dataset metaqa --carve fit --host
    python outputs/mp_unified/zu1c.py train --split L-musique --name scr-u1c-zrc --arm zrc --device cuda --host \\
        --cache-root outputs/u1c/cache
    python outputs/mp_unified/zu1c.py read --name scr-u1c-zrc --device cuda --host --cache-root outputs/u1c/cache
    python outputs/mp_unified/zu1c.py compare --new outputs/screen/fits/scr-u1c-zrc \\
        --base outputs/screen/fits/scr-zrct,outputs/step1/fits/L-musique --out outputs/screen/scr-u1c-zrc
    python outputs/mp_unified/zu1c.py compare --new outputs/screen/fits/scr-zrct --base outputs/step1/fits/L-musique \\
        --out outputs/u1c/base-zrct-L-musique
    python outputs/mp_unified/zu1c.py pair --arm zrc --screens outputs/screen/scr-u1c-zrc.json,... --out ...
    python outputs/mp_unified/zu1c.py recall --arm zrc --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zu1c.py --selftest
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

OUT = ROOT / "outputs" / "u1c"
LOOK, CACHE = OUT / "look", OUT / "cache"
CH_RM, CH_ZRC, LINKS = OUT / "chains" / "rmatch", OUT / "chains" / "zrc", OUT / "links"
GATE = OUT / "gate-screen.json"
DEVICE = "cuda"
ARMS = {"zrc": (ZM.ZRM, ZC.ChainCarveZRC), "zsp": (ZP.ZProp, ZL.LinkCarveBase)}
BASE_FITS = {"zrc": {"L-musique": ("screen", "fits", "scr-zrct"), "L-hotpotqa": ("screen", "fits", "scr-zrct-hp")},
             "zsp": {"L-musique": ("screen", "fits", "scr-zspg"), "L-hotpotqa": ("screen", "fits", "scr-zspg-hp")}}
BASE_SCREENS = {"zrc": {sp: f"outputs/u1c/base-zrct-{sp}.json" for sp in BASE_FITS["zrc"]},
                "zsp": {sp: f"outputs/zg1/base-zspg-{sp}.json" for sp in BASE_FITS["zsp"]}}
AGAINST = {"zrc": "zrc's card fit of each split on today's graph and pools (scr-zrct)",
           "zsp": "zsp's card fit of each split on today's graph and pools (scr-zspg)"}
_BUILT = RM.built_records
_ZL_BUILT = ZL.built_records


# ── U1c's roots ──────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_u1c():
    """rmatch's, zrc's and zlink's carves (and their records) read U1c's chains and links; restored after."""
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
        raise SystemExit(f"zu1c: train and read take --cache-root {CACHE.relative_to(ROOT).as_posix()}")


def on_card(argv):
    dev, _t = ZP.device_of(argv)
    if dev != DEVICE:
        raise SystemExit(f"zu1c: U1c trains and reads on the {DEVICE}, as its bases did, not {dev}")
    return list(argv) if "--device" in argv else list(argv) + ["--device", DEVICE]


def gate_passed(gate=GATE):
    if not Path(gate).exists():
        raise SystemExit(f"zu1c: {gate} missing (scripts/u1c_host.py gate --screen)")
    rec = json.loads(Path(gate).read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("declared_in") != "docs/U1C_RETRAIN_ON_U.md":
        raise SystemExit(f"zu1c: {gate} is {rec.get('verdict')}, not U1c's PASS")
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
        raise SystemExit(f"zu1c build: --what rmatch, zrc or links, not {what}")
    return 0


def stamp(argv, arm):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zu1c_sha256": LC.sha_src(__file__), "u1c": {
            "declared_in": "docs/U1C_RETRAIN_ON_U.md", "arm": arm, "device": DEVICE,
            "graph": "U1d's structural_U (outputs/u1d/<D>/graph_structural_u.npz)",
            "pools": "step 4h's chosen configuration (outputs/step4h/choice.json), outputs/u1c/pools",
            "cache_root": CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(GATE),
            "rmatch_records": RM.built_records(), "zrc_records": _BUILT(CH_ZRC),
            "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS:
        raise SystemExit(f"zu1c: train takes --arm zrc or zsp, not {arm}")
    if S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zu1c: the arm {arm} is {S.ARMS.get(arm)}, not {ARMS[arm]}")
    if split not in BASE_FITS[arm]:
        raise SystemExit(f"zu1c: U1c's screen fits L-musique and L-hotpotqa, not {split}")
    cache_root_of(argv)
    argv = on_card(argv)
    gate_passed()
    with on_u1c():
        rc = RM.train(argv, split)
        stamp(argv, arm)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    rec = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))
    if rec.get("arm") not in ARMS or "u1c" not in rec:
        raise SystemExit(f"zu1c: {name} is not a U1c fit (arm {rec.get('arm')})")
    cache_root_of(argv)
    with on_u1c():
        return RM.read(on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its own card fit ───────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zu1c: no base fit of {arm} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, arm, fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b = {"zrc": "scr-zrct", "zsp": "scr-zspg"}[arm]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({arm}'s card fits {b} and {b}-hp, today's graph and pools)"),
            ("(rel's screen fits)", f"({arm}'s card fits on today's graph and pools)"),
            ("| rel R@5 | relz R@5 |", f"| {arm} R@5 (today's) | {arm} R@5 (U1c) |"),
            ("section 2 and the tenth round", "section 2, and docs/U1C_RETRAIN_ON_U.md"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"U1c (docs/U1C_RETRAIN_ON_U.md): {arm} on U1d's graph and step 4h's pools against {arm} on "
                      f"today's.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "U1c", "declared_in": "docs/U1C_RETRAIN_ON_U.md", "decided_against": AGAINST[arm],
                    "device": DEVICE, "zu1c_sha256": LC.sha_src(__file__)})
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
    with on_u1c():
        assert (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT) == (CH_RM, CH_ZRC, LINKS)
        assert ZL.LinkCarveBase.CH_ROOT == CH_RM          # zsp's chains are rmatch's, through the class it extends
    assert (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT) == before
    assert RM.built_records is _BUILT and ZL.built_records is _ZL_BUILT
    good = ["train", "--arm", "zrc", "--name", "x", "--cache-root", "outputs/u1c/cache"]
    SR.must_stop(train, ["train", "--arm", "zrm", "--name", "x", "--cache-root", "outputs/u1c/cache"], "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, ["train", "--arm", "zsp", "--name", "x"], "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    SR.must_stop(gate_passed, OUT / "no-such-gate.json")
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        g = T / "g.json"
        LC.write_json(g, {"verdict": "FAIL", "declared_in": "docs/U1C_RETRAIN_ON_U.md"})
        SR.must_stop(gate_passed, g)
        LC.write_json(g, {"verdict": "PASS", "declared_in": "docs/U1C_RETRAIN_ON_U.md"})
        assert gate_passed(g)["verdict"] == "PASS"
        (T / "fits" / "y").mkdir(parents=True)
        LC.write_json(T / "fits" / "y" / "screen.json", {"arm": "zrc"})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T / "fits"), "--cache-root", "outputs/u1c/cache"])
        # relz's pair and re-call for each arm against its own card fits, relz restored after
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARMS:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=arm)) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", f"scr-u1c-{arm}", "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", f"scr-u1c-{arm}-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(T / arm / "z2", f"scr-u1c-{arm}", "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="outputs/screen/fits/scr-zrm")
            SR.must_stop(pair, arm, [zw, zh], T / arm / "bad")
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm, pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "U1c" and pj["decided_against"] == AGAINST[arm]
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith("U1c (docs/U1C_RETRAIN_ON_U.md)") and f"| {arm} R@5 (today's) | {arm} R@5 (U1c) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zu1c selftest: arms zrc and zsp as registered; U1c's chain and link roots in place and restored; train takes "
        f"no other arm, split, cache root or device and needs the gate's PASS; read refuses a fit that is not U1c's; "
        f"relz's pair and re-call run against each arm's card fits ({time.time() - t0:.1f}s): ok")
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
            log(f"zu1c {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zu1c: build, train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
