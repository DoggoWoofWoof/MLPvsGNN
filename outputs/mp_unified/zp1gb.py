"""P1g-b's screen (docs/P1GB_POOLS_REFIT.md): the lean pair, unchanged, refit on P1g-b's pools (P1g2's chosen rule,
scripts/p1gb_pools.py) on today's graph. Every command is P1c3's and P1c3p's (outputs/mp_unified/zp1c3.py and zp1c3p.py,
imported unchanged) with P1g-b's roots in place of P1c2's:
    looks   outputs/p1gb/look   (outputs/mp_unified/look_p1gb.py)
    caches  outputs/p1gb/cache  (lean_cache on those looks; --cache-root, required on train and read)
    chains  outputs/p1gb/chains/rmatch and outputs/p1gb/chains/zrc (rmatch's and zrc's builds)
    links   outputs/p1gb/links  (zlink's build: the pool edges inside each P1g-b pool)
    KT      outputs/p1gb/kt     (scripts/p1gb_pools.py kt: P1c3's eight columns, the fill column marking fill_f)
The relation tables are today's: the graph does not change.

Two arms, the lean pair as filed:
    zrckp  the MLP (zp1c3p's: zrc's model plus the pointwise KT head), on zrc's chain carve with the KT input beside
    zspk   the GNN (zp1c3's: zsp's model plus the KT head), on zlink's carve with the KT input beside
Their settings, seeds and batches are unchanged. One card fit per arm on L-musique and L-hotpotqa, read on the six
s1eval carves of P1g-b's pools, and compared with the same arm's card fit of the split on P1c2's pools (zrckp:
scr-p1c3p-zrckp, -hp; zspk: scr-p1c3-zspk, -hp), the same questions and gold totals; R@5 over the question's golds,
so a gold a pool misses counts as a miss. The pair and its re-call under the seed null are relz's, stamped as
P1g-b's; the verdict is this file's (P1c2's rule).

    python outputs/mp_unified/zp1gb.py build --what rmatch|zrc|links --dataset metaqa --carve fit --host
    python outputs/mp_unified/zp1gb.py train --split L-musique --name scr-p1gb-zrckp --arm zrckp --device cuda \\
        --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/zp1gb.py read --name scr-p1gb-zrckp --device cuda --host --cache-root outputs/p1gb/cache
    python outputs/mp_unified/zp1gb.py compare --new outputs/screen/fits/scr-p1gb-zrckp \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-p1gb-zrckp
    python outputs/mp_unified/zp1gb.py pair --arm zrckp --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zp1gb.py recall --arm zrckp --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zp1gb.py verdict
    python outputs/mp_unified/zp1gb.py --selftest
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

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk, KT carves and P1c2's roots, gate and card)

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/P1GB_POOLS_REFIT.md"
OUT = ROOT / "outputs" / "p1gb"
LOOK, CACHE, KT_ROOT = OUT / "look", OUT / "cache", OUT / "kt"
CH_RM, CH_ZRC, LINKS = OUT / "chains" / "rmatch", OUT / "chains" / "zrc", OUT / "links"
GATE = OUT / "gate.json"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
ARM_NAMES = ("zrckp", "zspk")
ARMS = {"zrckp": Z3P.ARMS["zrckp"], "zspk": Z3.ARMS["zspk"]}
HEAD = {"zrckp": Z3P.SETTINGS, "zspk": Z3.SETTINGS}
FIT = {a: {"L-musique": f"scr-p1gb-{a}", "L-hotpotqa": f"scr-p1gb-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckp": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspk": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                      "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/p1gb/base-{a}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{a}'s card fit of each split on P1c2's pools ({BASE_FITS[a]['L-musique'][2]})" for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-p1gb-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-p1gb-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
RULE = ("ADOPT_FOR_FULL_RUN when neither arm's re-call under the seed null has a LOSS on any of its twelve reads and "
        "the two arms together have at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when a re-call is missing, not "
        "P1g-b's, or not twelve reads (docs/P1GB_POOLS_REFIT.md, P1c2's rule)")
Z2_ROOTS = ("LOOK", "CACHE", "CH_RM", "CH_ZRC", "LINKS", "GATE")


# ── P1g-b's roots ────────────────────────────────────────────────────────────


@contextlib.contextmanager
def on_p1gb():
    """P1c2's roots (looks, caches, chains, links, gate) and P1c3's KT root and declaration replaced by P1g-b's, then
    P1c2's chain and link binding entered on them; all restored after. The relation tables are today's."""
    saved = ({k: getattr(Z2, k) for k in Z2_ROOTS}, Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT,
             Z3.DECLARED)
    for k, v in zip(Z2_ROOTS, (LOOK, CACHE, CH_RM, CH_ZRC, LINKS, GATE)):
        setattr(Z2, k, v)
    Z3.KTChainCarveZRC.KT_ROOT = Z3.KTLinkCarveBase.KT_ROOT = KT_ROOT
    Z3.DECLARED = DECLARED                   # zp1c3.load_kt and kt_records check each KT record's declaration
    try:
        with Z2.on_p1c2():
            yield
    finally:
        for k, v in saved[0].items():
            setattr(Z2, k, v)
        Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED = saved[1:]


def gate_passed(gate=None):
    gate = Path(gate or GATE)
    if not gate.exists():
        raise SystemExit(f"zp1gb: {gate} missing (scripts/p1gb_pools.py gate)")
    rec = json.loads(gate.read_text(encoding="utf-8"))
    if rec.get("verdict") != "PASS" or rec.get("declared_in") != DECLARED:
        raise SystemExit(f"zp1gb: {gate} is {rec.get('verdict')} ({rec.get('declared_in')}), not P1g-b's PASS")
    return rec


# ── build, train, read, compare ──────────────────────────────────────────────


def build(what, ds, carve, host):
    with on_p1gb():
        return Z2.build(what, ds, carve, host)


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        g = gate_passed()
        head = dict(HEAD[arm], kt_root=KT_ROOT.relative_to(ROOT).as_posix())
        rec.update({"zp1gb_sha256": LC.sha_src(__file__), "zp1c3p_sha256": LC.sha_src(Z3P.__file__),
                    "zp1c3_sha256": LC.sha_src(Z3.__file__), "zp1c2_sha256": LC.sha_src(Z2.__file__), "p1gb": {
                        "declared_in": DECLARED, "arm": arm, "device": Z2.DEVICE, "graph": "today's",
                        "pools": f"{', '.join(g.get('construction', []))} (outputs/p1gb/pools)",
                        "cache_root": CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(GATE), "head": head,
                        "kt_records": Z3.kt_records(Z3.train_carves(split), KT_ROOT),
                        "rmatch_records": RM.built_records(), "zrc_records": Z2._BUILT(CH_ZRC),
                        "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zp1gb: train takes --arm zrckp or zspk as filed, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zp1gb: P1g-b's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zp1gb: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    with on_p1gb():
        Z2.cache_root_of(argv)
        argv = Z2.on_card(argv)
        gate_passed()
        Z3.kt_records(Z3.train_carves(split), KT_ROOT)
        rc = RM.train(argv, split)
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "p1gb" not in rec:
        raise SystemExit(f"zp1gb: {name} is not a P1g-b fit (arm {rec.get('arm')})")
    with on_p1gb():
        Z2.cache_root_of(argv)
        Z3.kt_records(Z3.eval_carves(), KT_ROOT)
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its own card fit on P1c2's pools ───────────────────────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zp1gb: no base fit of {arm} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, arm, fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b = BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({arm}'s card fits {b} and {b}-hp, P1c2's pools)"),
            ("(rel's screen fits)", f"({arm}'s card fits on P1c2's pools)"),
            ("| rel R@5 | relz R@5 |", f"| {arm} R@5 (P1c2's pools) | {arm} R@5 (P1g-b's) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"P1g-b ({DECLARED}): {arm} on P1g-b's pools against {arm} on P1c2's.\n\n" + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "P1gb", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zp1gb_sha256": LC.sha_src(__file__)})
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


# ── the verdict ──────────────────────────────────────────────────────────────


def tag(r):
    return f"{r['split']} {r['dataset']}"


def recall_rows(rp, arm, problems):
    rp = Path(rp)
    if not rp.exists():
        problems.append(f"{rp.name} missing")
        return []
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("stage") != "P1gb" or rec.get("declared_in") != DECLARED or rec.get("arm") != arm:
        problems.append(f"{rp.name} is not P1g-b's re-call of {arm}")
    rows = rec.get("rows", [])
    got = sorted((r.get("split"), r.get("dataset")) for r in rows)
    if got != sorted((sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
        problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six datasets")
    return rows


def verdict(recalls=None, out=None):
    """P1g-b's screen verdict from the two arms' re-calls (P1c2's rule)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    problems, per = [], {}
    for a in ARM_NAMES:
        rows = recall_rows(recalls[a], a, problems)
        per[a] = {"gains": [tag(r) for r in rows if r.get("call") == "GAIN"],
                  "losses": [tag(r) for r in rows if r.get("call") == "LOSS"],
                  "calls": {tag(r): r.get("call") for r in rows},
                  "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "recall_sha256": LC.sha_file(recalls[a]) if Path(recalls[a]).exists() else None}
    if problems:
        v = "INCOMPLETE"
    elif any(per[a]["losses"] for a in ARM_NAMES) or not any(per[a]["gains"] for a in ARM_NAMES):
        v = "NOT_ADOPTED"
    else:
        v = "ADOPT_FOR_FULL_RUN"
    rec = {"stage": "P1gb", "declared_in": DECLARED, "rule": RULE, "verdict": v, "arms": per, "problems": problems,
           "zp1gb_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# P1g-b's screen verdict: **{v}**", "", f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'}).")
    if problems:
        md += ["", "Problems: " + "; ".join(problems)]
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    counts = ", ".join(f"{a} {len(per[a]['gains'])} GAIN {len(per[a]['losses'])} LOSS" for a in ARM_NAMES)
    log(f"zp1gb verdict: {v} ({'; '.join(problems) if problems else counts})")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "P1gb", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import tempfile
    t0 = time.time()
    # 1. the arms are the lean pair as filed
    assert ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC) and ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    for a in ARM_NAMES:
        assert S.ARMS[a] == ARMS[a], a
    # 2. P1g-b's roots in place inside, P1c2's and P1c3's restored after; today's relation tables untouched
    before = {k: getattr(Z2, k) for k in Z2_ROOTS}
    kt_before = (Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED)
    cls_before = (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT)
    rel_before = (RM.ChainCarveBase.REL_DIR, ZC.ChainCarveZRC.REL_DIR, ZL.LinkCarveBase.REL_DIR)
    with on_p1gb():
        assert [getattr(Z2, k) for k in Z2_ROOTS] == [LOOK, CACHE, CH_RM, CH_ZRC, LINKS, GATE]
        assert (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT) == (CH_RM, CH_ZRC, LINKS)
        assert ZL.LinkCarveBase.CH_ROOT == CH_RM
        assert Z3.KTChainCarveZRC.KT_ROOT == KT_ROOT and Z3.KTLinkCarveBase.KT_ROOT == KT_ROOT
        assert Z3.DECLARED == DECLARED
        assert (RM.ChainCarveBase.REL_DIR, ZC.ChainCarveZRC.REL_DIR, ZL.LinkCarveBase.REL_DIR) == rel_before
        SR.must_stop(Z2.cache_root_of, ["--cache-root", "outputs/p1c2/cache"])
        Z2.cache_root_of(["--cache-root", "outputs/p1gb/cache"])
    assert {k: getattr(Z2, k) for k in Z2_ROOTS} == before
    assert (Z3.KTChainCarveZRC.KT_ROOT, Z3.KTLinkCarveBase.KT_ROOT, Z3.DECLARED) == kt_before
    assert (RM.ChainCarveBase.CH_ROOT, ZC.ChainCarveZRC.CH_ROOT, ZL.LinkCarveBase.LK_ROOT) == cls_before
    assert RM.built_records is Z2._BUILT and ZL.built_records is Z2._ZL_BUILT
    assert all("p1gb" in p.as_posix() for p in (LOOK, CACHE, KT_ROOT, CH_RM, CH_ZRC, LINKS, GATE))
    # 3. the refusals: arm, split, fit name, cache root, device, gate, a fit that is not P1g-b's
    good = ["train", "--arm", "zrckp", "--name", "scr-p1gb-zrckp", "--cache-root", "outputs/p1gb/cache"]
    SR.must_stop(train, ["train", "--arm", "zrc", "--name", "scr-p1gb-zrckp", "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", "scr-p1gb-zrckp-hp", "--cache-root",
                         "outputs/p1gb/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspk", "--name", "scr-p1gb-zspk", "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    SR.must_stop(build, "nothing", "metaqa", "fit", False)
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        g = T / "g.json"
        SR.must_stop(gate_passed, T / "none.json")
        LC.write_json(g, {"verdict": "FAIL", "declared_in": DECLARED})
        SR.must_stop(gate_passed, g)
        LC.write_json(g, {"verdict": "PASS", "declared_in": Z2.DECLARED})
        SR.must_stop(gate_passed, g)
        LC.write_json(g, {"verdict": "PASS", "declared_in": DECLARED})
        assert gate_passed(g)["verdict"] == "PASS"
        (T / "fits" / "y").mkdir(parents=True)
        LC.write_json(T / "fits" / "y" / "screen.json", {"arm": "zrckp", "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T / "fits"), "--cache-root", "outputs/p1gb/cache"])
        # 4. relz's pair and re-call for each arm against its own card fits on P1c2's pools, relz restored after
        saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=arm)) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            zw = SR.fake_compare(T / arm / "z2", FIT[arm]["L-musique"], "L-musique", {}, arm=arm, base_r5=0.6,
                                 base="outputs/screen/fits/scr-zrm")
            SR.must_stop(pair, arm, [zw, zh], T / arm / "bad")
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm, pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "P1gb" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"P1g-b ({DECLARED})") and f"| {arm} R@5 (P1c2's pools) | {arm} R@5 (P1g-b's) |" in t
            rj = json.loads((T / arm / "rz.json").read_text(encoding="utf-8"))
            assert rj["stage"] == "P1gb" and rj["arm"] == arm
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
        # 5. the verdict's outcomes
        def rc(a, calls, **over):
            return fake_recall(T / f"v-{a}.json", a, calls, **over)

        def v(ra, rs):
            return verdict({"zrckp": ra, "zspk": rs}, T / "verdict")["verdict"]

        gm = ("L-musique", "musique")
        assert v(rc("zrckp", {gm: "GAIN"}), rc("zspk", {})) == "ADOPT_FOR_FULL_RUN"
        assert v(rc("zrckp", {gm: "GAIN"}), rc("zspk", {("L-hotpotqa", "webqsp"): "LOSS"})) == "NOT_ADOPTED"
        assert v(rc("zrckp", {}), rc("zspk", {})) == "NOT_ADOPTED"
        assert v(rc("zrckp", {gm: "GAIN"}), rc("zspk", {}, stage="P1c3")) == "INCOMPLETE"
        assert v(rc("zrckp", {gm: "GAIN"}), T / "none.json") == "INCOMPLETE"
        assert v(fake_recall(T / "wrong-arm.json", "zspk", {gm: "GAIN"}), rc("zspk", {})) == "INCOMPLETE"
        short = fake_recall(T / "short.json", "zspk", {})
        sj = json.loads(short.read_text(encoding="utf-8"))
        sj["rows"] = sj["rows"][:11]
        LC.write_json(short, sj)
        assert v(rc("zrckp", {gm: "GAIN"}), short) == "INCOMPLETE"
    log(f"zp1gb selftest: the arms are the lean pair as filed; P1g-b's looks, caches, chains, links, gate and KT root in "
        f"place and P1c2's and P1c3's restored, today's relation tables untouched; train refuses another arm, split, "
        f"fit name, cache root or device and needs P1g-b's gate PASS; read refuses a fit that is not P1g-b's; relz's "
        f"pair and re-call run against each arm's card fits on P1c2's pools and are stamped P1g-b's; the verdict's "
        f"three outcomes ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "verdict":
        return 0 if verdict()["verdict"] != "INCOMPLETE" else 1
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--arm", required=True, choices=ARM_NAMES)
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
            log(f"zp1gb {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zp1gb: build, train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
