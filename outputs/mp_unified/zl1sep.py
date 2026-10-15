"""L1, the first ledger screen (docs/L1_SEP_LEAN_PAIR.md): the lean pair trained with zsep's objective. zsep
(outputs/mp_unified/zsep.py, imported unchanged) scores each gold row against its question's non-gold rows only:
lean_gpu's listwise loss puts a question's other golds in each gold's denominator, so once one gold leads it is pushed
down and the non-gold rows above a trailing gold get almost no push. On zret (the sixteenth round) zsep was NOT_ADOPTED
after its re-call (docs/FULL_ROUND16.md); the lean pair is a different model on KIND_T's pools, where multi-gold
questions (musique's chains, hotpotqa and 2wiki's pairs) are the ones the pair misses.

Two arms, each the lean pair's arm unchanged (model, carve, KT input, settings, seeds, batches) with zsep's loss in
training only:
    zrckps  zrckp's model and carve (outputs/mp_unified/zp1c3p.py), trained with zsep.listwise_sep
    zspks   zspk's model and carve (outputs/mp_unified/zp1c3.py), trained with zsep.listwise_sep
Reads and serving are the arm's own. One card fit per arm on L-musique and L-hotpotqa, on P1c2's roots and P1c3's KT
input, read on the six s1eval carves and compared with the same arm's card fit of the split under lean_gpu's loss
(zrckp: scr-p1c3p-zrckp, -hp; zspk: scr-p1c3-zspk, -hp), which decides; step 1's fits beside. The pair and its re-call
under the seed null are relz's, stamped as L1's; the verdict is this file's.

    python outputs/mp_unified/zl1sep.py train --split L-musique --name scr-l1-zrckps --arm zrckps --device cuda \\
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zl1sep.py read --name scr-l1-zrckps --device cuda --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/zl1sep.py compare --new outputs/screen/fits/scr-l1-zrckps \\
        --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-l1-zrckps
    python outputs/mp_unified/zl1sep.py pair --arm zrckps --screens S1.json,S2.json --out ...
    python outputs/mp_unified/zl1sep.py recall --arm zrckps --null N1,N2,N3,N4 --pair P.json --out ...
    python outputs/mp_unified/zl1sep.py verdict
    python outputs/mp_unified/zl1sep.py --selftest
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

import zp1c3p as Z3P  # noqa: E402  (zrckp; through it P1c3's zspk and KT carves, P1c2's roots, gate and card)
import zsep as ZS  # noqa: E402  (the objective, unchanged)

import numpy as np  # noqa: E402
import torch  # noqa: E402

Z3 = Z3P.Z3
Z2 = Z3.Z2
ZP, ZC, ZL, ZM, RM = Z3.ZP, Z3.ZC, Z3.ZL, Z3.ZM, Z3.RM
Z, R, S2, S, LG, LC, SR = Z3.Z, Z3.R, Z3.S2, Z3.S, Z3.LG, Z3.LC, Z3.SR
log = S.log

DECLARED = "docs/L1_SEP_LEAN_PAIR.md"
OUT = ROOT / "outputs" / "l1"
SCREEN = ROOT / "outputs" / "screen"
SPLITS = Z3.SPLITS
BASE_OF = {"zrckps": "zrckp", "zspks": "zspk"}
ARM_NAMES = tuple(BASE_OF)
ARMS = {"zrckps": Z3P.ARMS["zrckp"], "zspks": Z3.ARMS["zspk"]}
S.ARMS.update(ARMS)
FIT = {a: {"L-musique": f"scr-l1-{a}", "L-hotpotqa": f"scr-l1-{a}-hp"} for a in ARM_NAMES}
BASE_FITS = {"zrckps": {"L-musique": ("screen", "fits", "scr-p1c3p-zrckp"),
                        "L-hotpotqa": ("screen", "fits", "scr-p1c3p-zrckp-hp")},
             "zspks": {"L-musique": ("screen", "fits", "scr-p1c3-zspk"),
                       "L-hotpotqa": ("screen", "fits", "scr-p1c3-zspk-hp")}}
BASE_SCREENS = {a: {sp: f"outputs/l1/base-{BASE_OF[a]}-{sp}.json" for sp in SPLITS} for a in ARM_NAMES}
AGAINST = {a: f"{BASE_OF[a]}'s card fit of each split under lean_gpu's loss ({BASE_FITS[a]['L-musique'][2]})"
           for a in ARM_NAMES}
PAIR = {a: SCREEN / f"scr-l1-{a}-pair" for a in ARM_NAMES}
RECALL = {a: SCREEN / f"scr-l1-{a}-pair-recall" for a in ARM_NAMES}
VERDICT = OUT / "screen-verdict"
RULE = ("per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at "
        "least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not L1's, or not twelve reads "
        "(docs/L1_SEP_LEAN_PAIR.md, section 3)")


# ── train and read ───────────────────────────────────────────────────────────


def stamp(argv, arm, split):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zl1sep_sha256": LC.sha_src(__file__), "zsep_sha256": LC.sha_src(ZS.__file__),
                    "zp1c3p_sha256": LC.sha_src(Z3P.__file__), "zp1c3_sha256": LC.sha_src(Z3.__file__),
                    "zp1c2_sha256": LC.sha_src(Z2.__file__), "l1": {
                        "declared_in": DECLARED, "arm": arm, "base_arm": BASE_OF[arm], "device": Z2.DEVICE,
                        "loss": "zsep.listwise_sep: each gold against the non-gold rows (training only)",
                        "census": ZS.STATS["census"], "pools": "P1c2's (outputs/p1c2/pools)",
                        "cache_root": Z2.CACHE.relative_to(ROOT).as_posix(), "gate": LC.sha_file(Z2.GATE),
                        "kt_records": Z3.kt_records(Z3.train_carves(split)), "rmatch_records": RM.built_records(),
                        "zrc_records": Z2._BUILT(Z2.CH_ZRC), "zlink_records": ZL.built_records()}})
        LC.write_json(sj, rec)


def train(argv, split):
    arm = R.arm_of(argv)
    if arm not in ARMS or S.ARMS.get(arm) != ARMS[arm]:
        raise SystemExit(f"zl1sep: train takes --arm zrckps or zspks, not {arm} ({S.ARMS.get(arm)})")
    if split not in SPLITS:
        raise SystemExit(f"zl1sep: L1's screen fits L-musique and L-hotpotqa, not {split}")
    name, _ = S2.where(argv)
    if name != FIT[arm][split]:
        raise SystemExit(f"zl1sep: {arm}'s fit of {split} is {FIT[arm][split]}, not {name}")
    Z2.cache_root_of(argv)
    argv = Z2.on_card(argv)
    Z2.gate_passed()
    Z3.kt_records(Z3.train_carves(split))
    ZS.STATS["census"] = None
    with Z2.on_p1c2(), ZS.sep_loss():
        rc = RM.train(argv, split)
    with Z2.on_p1c2():
        stamp(argv, arm, split)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    rec = json.loads(sj.read_text(encoding="utf-8")) if sj.exists() else {}
    if rec.get("arm") not in ARMS or "l1" not in rec:
        raise SystemExit(f"zl1sep: {name} is not an L1 fit (arm {rec.get('arm')})")
    Z2.cache_root_of(argv)
    Z3.kt_records(Z3.eval_carves())
    with Z2.on_p1c2():
        return RM.read(Z2.on_card(argv))


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz's pair and re-call, each arm decided against its base arm's card fit ────────────────────────────────


@contextlib.contextmanager
def on_base(arm):
    fits = BASE_FITS[arm]

    def base_fit(split):
        if split not in fits:
            raise SystemExit(f"zl1sep: no base fit of {BASE_OF[arm]} on {split}")
        return fits[split]

    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = arm, BASE_OF[arm], fits, BASE_SCREENS[arm], base_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


def fix(arm):
    b, f = BASE_OF[arm], BASE_FITS[arm]["L-musique"][2]
    return (("(rel's screen fits scr-rel and scr-rel-hp)", f"({b}'s card fits {f} and {f}-hp, lean_gpu's loss)"),
            ("(rel's screen fits)", f"({b}'s card fits, lean_gpu's loss)"),
            ("| rel R@5 | relz R@5 |", f"| {b} R@5 | {arm} R@5 (zsep's loss) |"),
            ("section 2 and the tenth round", f"section 2, and {DECLARED}"))


def restamp(out, arm):
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in fix(arm):
            t = t.replace(a, b)
        md.write_text(f"L1 ({DECLARED}): {arm} ({BASE_OF[arm]} trained with zsep's loss) against {BASE_OF[arm]}.\n\n"
                      + t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"stage": "L1", "declared_in": DECLARED, "decided_against": AGAINST[arm], "device": Z2.DEVICE,
                    "zl1sep_sha256": LC.sha_src(__file__)})
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


def verdict(recalls=None, out=None):
    """L1's screen verdict, per arm, from its re-call (section 3)."""
    recalls = recalls or {a: RECALL[a].with_suffix(".json") for a in ARM_NAMES}
    per = {}
    for a in ARM_NAMES:
        rp, problems = Path(recalls[a]), []
        rows = []
        if not rp.exists():
            problems.append(f"{rp.name} missing")
        else:
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if rec.get("stage") != "L1" or rec.get("declared_in") != DECLARED or rec.get("arm") != a:
                problems.append(f"{rp.name} is not L1's re-call of {a}")
            rows = rec.get("rows", [])
            if sorted((r.get("split"), r.get("dataset")) for r in rows) != sorted(
                    (sp, ds) for sp in SPLITS for ds in LG.EVAL_ORDER):
                problems.append(f"{rp.name}: {len(rows)} reads, not the twelve of {', '.join(SPLITS)} on the six")
        gains = [tag(r) for r in rows if r.get("call") == "GAIN"]
        losses = [tag(r) for r in rows if r.get("call") == "LOSS"]
        v = "INCOMPLETE" if problems else ("NOT_ADOPTED" if losses or not gains else "ADOPT_FOR_FULL_RUN")
        per[a] = {"verdict": v, "gains": gains, "losses": losses, "problems": problems,
                  "calls": {tag(r): r.get("call") for r in rows}, "delta_r5": {tag(r): r.get("delta") for r in rows},
                  "recall_sha256": LC.sha_file(rp) if rp.exists() else None}
    rec = {"stage": "L1", "declared_in": DECLARED, "rule": RULE, "arms": per, "zl1sep_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = [f"# L1's screen verdict: zrckps **{per['zrckps']['verdict']}**, zspks **{per['zspks']['verdict']}**", "",
          f"Rule: {RULE}.", ""]
    for a in ARM_NAMES:
        p = per[a]
        md.append(f"- **{a}** against {AGAINST[a]}, after the seed-null re-call: {len(p['gains'])} GAIN "
                  f"({', '.join(p['gains']) or 'none'}), {len(p['losses'])} LOSS ({', '.join(p['losses']) or 'none'})"
                  + (f"; problems: {'; '.join(p['problems'])}" if p["problems"] else "") + ".")
    out = Path(out or VERDICT)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("zl1sep verdict: " + ", ".join(f"{a} {per[a]['verdict']}" for a in ARM_NAMES))
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def fake_recall(path, arm, calls, **over):
    rows = [{"split": sp, "dataset": ds, "read": "x", "delta": 0.0, "call": calls.get((sp, ds), "WITHIN")}
            for sp in SPLITS for ds in LG.EVAL_ORDER]
    rec = {"stage": "L1", "declared_in": DECLARED, "arm": arm, "verdict": "x", "rows": rows}
    rec.update(over)
    LC.write_json(path, rec)
    return path


def selftest():
    import tempfile
    t0 = time.time()
    # 1. the arms are the lean pair's, unchanged; the objective swaps lean_gpu's loss in training only, restored after
    assert S.ARMS["zrckps"] == S.ARMS["zrckp"] == (Z3P.ZRCKP, Z3.KTChainCarveZRC)
    assert S.ARMS["zspks"] == S.ARMS["zspk"] == (Z3.ZSPK, Z3.KTLinkCarveBase)
    saved = LG.listwiseD, LG.fit_variant
    with ZS.sep_loss():
        assert LG.listwiseD is ZS.listwise_sep and LG.fit_variant is not saved[1]
    assert (LG.listwiseD, LG.fit_variant) == saved
    # 2. the loss: one gold equals lean_gpu's; several golds differ and match the float64 reference
    g = torch.Generator().manual_seed(7)
    s = torch.randn(30, generator=g, dtype=torch.float64)
    nq = torch.repeat_interleave(torch.arange(3), torch.tensor([10, 12, 8]))
    one = torch.zeros(30, dtype=torch.float64)
    one[[2, 14, 25]] = 1.0
    s32, one32 = s.float(), one.float()
    assert abs(float(ZS.listwise_sep(s32, one32, nq, 3)) - float(saved[0](s32, one32, nq, 3))) < 1e-5
    many = one.clone()
    many[[3, 4, 15]] = 1.0
    ref, _grad = ZS.sep_numpy(s.numpy(), many.numpy() > 0, nq.numpy(), 3)
    assert abs(float(ZS.listwise_sep(s, many, nq, 3)) - ref) < 1e-8
    assert abs(float(ZS.listwise_sep(s32, many.float(), nq, 3)) - float(saved[0](s32, many.float(), nq, 3))) > 1e-3
    # 3. the refusals
    good = ["train", "--arm", "zrckps", "--name", "scr-l1-zrckps", "--cache-root", "outputs/p1c2/cache"]
    SR.must_stop(train, ["train", "--arm", "zrckp", "--name", "scr-l1-zrckps", "--cache-root", "outputs/p1c2/cache"],
                 "L-musique")
    SR.must_stop(train, good, "L-2wiki")
    SR.must_stop(train, ["train", "--arm", "zrckps", "--name", "scr-l1-zrckps-hp", "--cache-root",
                         "outputs/p1c2/cache"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zspks", "--name", "scr-l1-zspks", "--cache-root", "outputs/p1gb/cache"],
                 "L-musique")
    SR.must_stop(train, good + ["--device", "cpu"], "L-musique")
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        (T / "fits" / "y").mkdir(parents=True)
        LC.write_json(T / "fits" / "y" / "screen.json", {"arm": "zrckp", "p1c3p": {}})
        SR.must_stop(read, ["--name", "y", "--out-root", str(T / "fits"), "--cache-root", "outputs/p1c2/cache"])
        # 4. relz's pair and re-call for each arm against its base arm's card fits, relz restored after
        saved_z = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
        for arm in ARM_NAMES:
            fits = BASE_FITS[arm]
            null1 = SR.fake_null(T / arm / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})
            bs = {sp: str(SR.fake_compare(T / arm / fits[sp][0], fits[sp][2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                          arm=BASE_OF[arm])) for sp in fits}
            zb = {sp: "/".join(("outputs",) + fits[sp]) for sp in fits}
            za = SR.fake_compare(T / arm / "z", FIT[arm]["L-musique"], "L-musique", {"webqsp": 0.0215}, arm=arm,
                                 base_r5=0.6, base=zb["L-musique"])
            zh = SR.fake_compare(T / arm / "z", FIT[arm]["L-hotpotqa"], "L-hotpotqa", {"hotpotqa": -0.0147}, arm=arm,
                                 base_r5=0.6, base=zb["L-hotpotqa"])
            pr = pair(arm, [za, zh], T / arm / "pz")
            assert pr["verdict"] == "MIXED" and pr["arm"] == arm and pr["base_arm"] == BASE_OF[arm], pr["verdict"]
            pj = json.loads((T / arm / "pz.json").read_text(encoding="utf-8"))
            assert pj["stage"] == "L1" and pj["decided_against"] == AGAINST[arm] and pj["declared_in"] == DECLARED
            r = recall(arm, null1, T / arm / "pz.json", T / arm / "rz", bs)
            assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
            t = (T / arm / "rz.md").read_text(encoding="utf-8")
            assert t.startswith(f"L1 ({DECLARED})") and f"| {BASE_OF[arm]} R@5 | {arm} R@5 (zsep's loss) |" in t
        assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved_z
        # 5. the verdict, per arm
        gm = ("L-musique", "musique")
        rec = verdict({"zrckps": fake_recall(T / "a.json", "zrckps", {gm: "GAIN"}),
                       "zspks": fake_recall(T / "b.json", "zspks", {gm: "GAIN", ("L-hotpotqa", "webqsp"): "LOSS"})},
                      T / "v")
        assert rec["arms"]["zrckps"]["verdict"] == "ADOPT_FOR_FULL_RUN"
        assert rec["arms"]["zspks"]["verdict"] == "NOT_ADOPTED"
        rec = verdict({"zrckps": fake_recall(T / "a.json", "zrckps", {}),
                       "zspks": fake_recall(T / "b.json", "zspks", {gm: "GAIN"}, stage="P1c3")}, T / "v")
        assert rec["arms"]["zrckps"]["verdict"] == "NOT_ADOPTED" and rec["arms"]["zspks"]["verdict"] == "INCOMPLETE"
        rec = verdict({"zrckps": T / "none.json", "zspks": fake_recall(T / "b.json", "zrckps", {gm: "GAIN"})}, T / "v")
        assert rec["arms"]["zrckps"]["verdict"] == "INCOMPLETE" and rec["arms"]["zspks"]["verdict"] == "INCOMPLETE"
    log(f"zl1sep selftest: the arms are the lean pair's; zsep's loss in training only, restored after; one gold gives "
        f"lean_gpu's loss, several match the float64 reference; train refuses another arm, split, fit name, cache root "
        f"or device; read refuses a fit that is not L1's; relz's pair and re-call against each base arm's card fits, "
        f"stamped L1's; the verdict per arm ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "verdict":
        return 0 if all(v["verdict"] != "INCOMPLETE" for v in verdict()["arms"].values()) else 1
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
            log(f"zl1sep {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zl1sep: train, read, compare, pair, recall, verdict, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
