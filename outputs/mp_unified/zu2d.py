"""U2d's screen (docs/U2D_REFIT_ON_U2E.md): zrckp, zspk and zg6 refit on two builds of P1c2's eleven carves, today's
pools rebuilt (base) and U2e's pick on W2_S (new), each build with its own roots (outputs/u2d/<build>/):
    looks   look           (outputs/mp_unified/look_u2d.py --build B)
    caches  cache          (lean_cache on those looks, step 1's bases copied in; --cache-root on train and read)
    chains  chains/rmatch and chains/zrc (rmatch's and zrc's builds, metaqa and webqsp)
    links   links          (zlink's build: the pool edges inside each pool)
    KT      kt             (zu2d_pools.py kt: P1c3's eight columns, and on the new build the seven via columns)
    gate    gate.json      (zu2d_pools.py gate)
The relation tables are today's on both builds. Every build call is zp1c2's (rmatch.build, zrc.build, zlink.build,
unchanged) inside P1c2's chain and link binding (zp1c2.on_p1c2), with the build's roots in place of P1c2's.

This is the build part, committed before the builds run. The fits, reads, null and decisions (section 5 to 7) follow
in this file before the first fit.

    python outputs/mp_unified/zu2d.py build --build base|new --what rmatch|zrc|links --dataset D --carve C [--modal]
    python outputs/mp_unified/zu2d.py --selftest
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk, KT carves and P1c2's roots, gate and card)
import zu2d_pools as ZP  # noqa: E402  (U2d's roots, declaration and gate)

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZC, ZL, RM = Z3.ZC, Z3.ZL, Z3.RM
log = Z3.S.log

DECLARED = ZP.DECLARED
BUILDS = ZP.BUILDS
WHAT = ("rmatch", "zrc", "links")
TYPED = tuple(RM.TYPED)                    # rmatch's and zrc's chains: the KB datasets
Z2_ROOTS = ("LOOK", "CACHE", "CH_RM", "CH_ZRC", "LINKS", "GATE")


# ── each build's roots ───────────────────────────────────────────────────────


def roots(build):
    """zu2d_pools' roots of the build, with the chains and links beside them."""
    r = ZP.roots(build)
    b = r["pools"].parent
    return {**r, "ch_rm": b / "chains" / "rmatch", "ch_zrc": b / "chains" / "zrc", "links": b / "links"}


@contextlib.contextmanager
def on_u2d(build):
    """P1c2's roots (looks, caches, chains, links, gate) and P1c3's KT root and declaration replaced by the build's,
    then P1c2's chain and link binding entered on them; all restored after. The relation tables are today's."""
    r = roots(build)
    saved = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
             Z3.DECLARED)
    for k, v in zip(Z2_ROOTS, (r["look"], r["cache"], r["ch_rm"], r["ch_zrc"], r["links"], r["gate"])):
        setattr(Z2, k, v)
    Z3.KTChainCarveZRC.KT_ROOT = Z3.KTLinkCarveBase.KT_ROOT = r["kt"]
    Z3.DECLARED = DECLARED                   # zp1c3.load_kt and kt_records check each KT record's declaration
    try:
        with Z2.on_p1c2():
            yield r
    finally:
        for k, v in saved[0].items():
            setattr(Z2, k, v)
        Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED = saved[1:]


def gate_passed(build):
    """The build's gate (zu2d_pools.py gate): PASS, under U2d's declaration, for this build."""
    gate = roots(build)["gate"]
    if not gate.exists():
        raise SystemExit(f"zu2d: {gate} missing (zu2d_pools.py gate --build {build})")
    rec = json.loads(gate.read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("declared_in") != DECLARED or rec.get("build") != build:
        raise SystemExit(f"zu2d: {gate} is {rec.get('verdict')} ({rec.get('declared_in')}, {rec.get('build')}), "
                         f"not U2d's PASS for the {build} build")
    return rec


def placement_of(where):
    if where not in ("laptop", "host", "modal"):
        raise SystemExit(f"zu2d: placement {where}")
    return {"where": where} if where != "modal" else {"where": "modal", "node": platform.node()}


# ── build ────────────────────────────────────────────────────────────────────


def build(b, what, ds, carve, where="laptop"):
    """One chain or link build of a carve on the build's roots (zp1c2.build's calls, with the placement named)."""
    if b not in BUILDS or what not in WHAT:
        raise SystemExit(f"zu2d build: --build {BUILDS}, --what {WHAT}")
    if what in ("rmatch", "zrc") and ds not in TYPED:
        raise SystemExit(f"zu2d build: {what} builds the typed datasets {TYPED}, not {ds}")
    pl = placement_of(where)
    with on_u2d(b) as r:
        if what == "rmatch":
            RM.build(ds, carve, r["ch_rm"], cache_root=r["cache"], look_root=r["look"], placement=pl)
        elif what == "zrc":
            ZC.build(ds, carve, r["ch_zrc"], cache_root=r["cache"], look_root=r["look"], placement=pl,
                     rm_root=r["ch_rm"])
        else:
            ZL.build(ds, carve, r["links"], cache_root=r["cache"], look_root=r["look"], placement=pl)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    filed = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
             Z3.DECLARED, RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT,
             RM.built_records, ZL.built_records)
    for b in BUILDS:
        r = roots(b)
        base = ROOT / "outputs" / "u2d" / b
        want = {"look": base / "look", "cache": base / "cache", "kt": base / "kt", "gate": base / "gate.json",
                "ch_rm": base / "chains" / "rmatch", "ch_zrc": base / "chains" / "zrc", "links": base / "links"}
        assert all(r[k] == v for k, v in want.items()), (b, r)
        with on_u2d(b) as rr:
            assert rr == r
            assert (Z2.LOOK, Z2.CACHE, Z2.CH_RM, Z2.CH_ZRC, Z2.LINKS, Z2.GATE) == (
                r["look"], r["cache"], r["ch_rm"], r["ch_zrc"], r["links"], r["gate"])
            assert Z3.KTChainCarveZRC.KT_ROOT == Z3.KTLinkCarveBase.KT_ROOT == r["kt"]
            assert Z3.DECLARED == DECLARED
            assert RM.ChainCarveBase.CH_ROOT == r["ch_rm"] and ZC.ChainCarveZRC.CH_ROOT == r["ch_zrc"]
            assert ZL.LinkCarveBase.LK_ROOT == r["links"]
        now = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
               Z3.DECLARED, RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT,
               RM.built_records, ZL.built_records)
        assert now == filed, f"{b}: the filed roots are not restored"
    try:
        with on_u2d("new"):
            raise KeyError("inside")
    except KeyError:
        pass
    assert (Z2.LOOK, Z3.DECLARED, RM.built_records) == (filed[0]["LOOK"], filed[3], filed[7]), "restored on an error"
    for args in (("old", "links", "musique", "fit"), ("new", "chains", "musique", "fit"),
                 ("new", "rmatch", "musique", "fit"), ("base", "zrc", "2wiki", "s1eval")):
        try:
            build(*args)
            raise AssertionError(f"build {args} ran")
        except SystemExit:
            pass
    with tempfile.TemporaryDirectory() as td:
        saved = ZP.OUT
        ZP.OUT = Path(td)
        try:
            for b, rec, ok in (("base", None, False), ("base", {"verdict": "PASS", "declared_in": DECLARED,
                                                                 "build": "base"}, True),
                               ("new", {"verdict": "PASS", "declared_in": DECLARED, "build": "base"}, False),
                               ("new", {"verdict": "FAIL", "declared_in": DECLARED, "build": "new"}, False),
                               ("new", {"verdict": "PASS", "declared_in": Z3.DECLARED, "build": "new"}, False)):
                g = roots(b)["gate"]
                if rec is not None:
                    g.parent.mkdir(parents=True, exist_ok=True)
                    g.write_text(json.dumps(rec), encoding="utf-8")
                try:
                    gate_passed(b)
                    assert ok, (b, rec)
                except SystemExit:
                    assert not ok, (b, rec)
        finally:
            ZP.OUT = saved
    assert placement_of("modal")["where"] == "modal" and placement_of("host") == {"where": "host"}
    log("zu2d selftest: each build's roots in place and the filed roots restored after (also on an error), the build "
        "refusals, the gate's refusals and the placements: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("build",))
    ap.add_argument("--build", choices=BUILDS)
    ap.add_argument("--what", choices=WHAT)
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--modal", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd != "build" or not (a.build and a.what and a.dataset and a.carve):
        ap.error("build --build B --what W --dataset D --carve C, or --selftest")
    if a.host and a.modal:
        ap.error("--host or --modal, not both")
    return build(a.build, a.what, a.dataset, a.carve, "modal" if a.modal else "host" if a.host else "laptop")


if __name__ == "__main__":
    sys.exit(main())
