"""Screens, thirteenth round (docs/SCREENS.md): the two adopted arms together, on zret's base. zgs is zret's model
(lean_screen3.ZRet: every block's within-pool z-score, and rrf's base z-score, taken against the pool's retrieved rows)
trained by gsurg's loop (lean_screen5.fit_gsurg: gradient surgery across the training datasets). Both parts are used
unchanged; no new column, block or hyperparameter. Two fits (L-musique and L-hotpotqa), each compared with zret's fit
of its split (that comparison decides) and with gsurg's and step 1's (reported); the pair, its re-call under the seed
null and the full run's grade are relz.py's, run under zgs's name with zret's fits in place of rel's.

    python outputs/mp_unified/zgs.py smoke --device cuda --host
    python outputs/mp_unified/zgs.py train --split L-musique --name scr-zgs --arm zgs --device cuda --host
    python outputs/mp_unified/zgs.py read --name scr-zgs --device cuda --host
    python outputs/mp_unified/zgs.py compare --new outputs/screen/fits/scr-zgs \\
        --base outputs/screen/fits/scr-zret,outputs/screen/fits/scr-gsurg,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zgs
    python outputs/mp_unified/zgs.py pair --screens outputs/screen/scr-zgs.json,outputs/screen/scr-zgs-hp.json \\
        --out outputs/screen/scr-zgs-pair
    python outputs/mp_unified/zgs.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zgs-pair.json \\
        --out outputs/screen/scr-zgs-pair-recall
    python outputs/mp_unified/zgs.py grade --full-root outputs/full_zgs \\
        --reuse L-musique=outputs/screen/scr-zgs.json,L-hotpotqa=outputs/screen/scr-zgs-hp.json
    python outputs/mp_unified/zgs.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json). A read's floor is the
null's; its base R@5 is zret's fit's (outputs/screen/scr-zret.json and outputs/full_zret/compare-L-hotpotqa.json, both
compared with the null's seed 0). The full run's re-grade is nullx.py's regrade with zret's six comparisons as
--base-compares.

zret's fits: L-musique is its screen fit (outputs/screen/fits/scr-zret); every other split is its full run's
(outputs/full_zret/fits/<split>), L-hotpotqa included, since zret's screen predates the two-fit screens.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import relz as Z  # noqa: E402
import lean_screen3 as S3  # noqa: E402
import lean_screen5 as S5  # noqa: E402

S2, S = Z.S2, Z.S
LG, LC = S.LG, S.LC
log = S.log
ARM, BASE_ARM = "zgs", "zret"
S.ARMS.update({ARM: S.ARMS[BASE_ARM]})


# ── train and read ───────────────────────────────────────────────────────────


def train(argv, split):
    """lean_screen2's train with zret's model and gsurg's loop."""
    arm = S5.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zgs: train takes --arm {ARM}, not {arm}")
    with S5.gsurg_loop():
        rc = S2.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec["zgs_sha256"] = LC.sha_src(__file__)
        rec["lean_screen3_sha256"] = LC.sha_src(S3.__file__)
        rec["lean_screen5_sha256"] = LC.sha_src(S5.__file__)
        rec["zgs"] = {"model": "lean_screen3.ZRet", "loop": "lean_screen5.fit_gsurg"}
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zgs: {name} was trained as {arm}, not {ARM}")
    return S2.main(["read"] + argv)


# ── relz.py's pair, re-call and grade, decided against zret's fits ───────────


ZRET_FITS = {"L-musique": ("screen", "fits", "scr-zret"), "L-hotpotqa": ("full_zret", "fits", "L-hotpotqa")}
ZRET_SCREENS = {"L-musique": "outputs/screen/scr-zret.json", "L-hotpotqa": "outputs/full_zret/compare-L-hotpotqa.json"}


def zret_fit(split):
    """zret's fit of a split: its screen fit on L-musique, its full run's fit on every other split."""
    return ZRET_FITS.get(split, ("full_zret", "fits", split))


def zret_compares():
    """zret's six comparisons with step 1's fits (nullx.py regrade's --base-compares)."""
    return {sp: (ZRET_SCREENS["L-musique"] if sp == "L-musique" else f"outputs/full_zret/compare-{sp}.json")
            for sp in S2.SPLITS}


@contextlib.contextmanager
def on_zret():
    """relz.py's records under zgs's name, each read decided against zret's fit of its split; restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZRET_FITS, ZRET_SCREENS, zret_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zret's fits: its screen fit scr-zret and its full run's "
                                                      "L-hotpotqa fit)"),
       ("(rel's screen fits)", "(zret's fits)"),
       ("| rel R@5 | relz R@5 |", "| zret R@5 | zgs R@5 |"),
       ("section 2 and the tenth round", "section 2 and the thirteenth round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND13.md"))


def restamp(out):
    """A record relz.py wrote under zgs's name: zret named as the base in its md, this file's sha added."""
    out = Path(out)
    md = out.with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        for a, b in FIX:
            t = t.replace(a, b)
        md.write_text(t, encoding="utf-8")
    js = out.with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec["decided_against"] = "zret's fit of each split"
        rec["zgs_sha256"] = LC.sha_src(__file__)
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zret():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zret_screens=None):
    with on_zret():
        rec = Z.recall(null_files, pair_file, out, zret_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zret():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── smoke ────────────────────────────────────────────────────────────────────


def smoke(device, host, out_root=None):
    """zret once, gsurg once and zgs twice (the repeat must be IDENTICAL), one epoch each on 2wiki's and hotpotqa's
    select carves (gsurg needs two datasets to project), each read on 2wiki's. zgs must hold zret's model (its state's
    keys), its loop must have met conflicting pairs, and its scores must differ from zret's (the same init and
    batches, so a difference is the surgery) and from gsurg's (the same loop, so a difference is zret's forward)."""
    t0 = time.time()
    root = Path(out_root or S.OUT / "smoke13")
    h = ["--host"] if host else []
    common = ["--train", "2wiki=select,hotpotqa=select", "--basis", "2wiki", "--variants", "p", "--config",
              "2e-3:1e-4:0.1:1:0", "--device", device, "--out-root", str(root)] + h
    runs = {BASE_ARM: (S2.train, "1"), "gsurg": (S5.train, "1"), ARM: (train, "2")}
    rec, ok = {}, True
    for arm, (fn, rep) in runs.items():
        nm = f"smoke-{arm}"
        rc = fn(["train", "--name", nm, "--arm", arm, "--repeat", rep] + common, "L-musique")
        rr = S2.main(["read", "--name", nm, "--read", "2wiki=select", "--device", device, "--out-root", str(root)] + h)
        sj = json.loads((root / nm / "screen.json").read_text(encoding="utf-8"))
        tj = json.loads((root / nm / "train.json").read_text(encoding="utf-8"))
        cur = tj["variants"]["p"]["curve"][0]
        blob = torch.load(root / nm / "models.pt", map_location="cpu", weights_only=False)
        rec[arm] = {"train_rc": rc, "repeats": int(rep), "read_rc": rr, "screen_arm": sj.get("arm"),
                    "gsurg": {k: cur["gsurg"][k] for k in ("pairs", "conflicts", "share")} if "gsurg" in cur else None,
                    "state_keys": sorted(blob["states"][sorted(blob["states"])[0]])}
        ok = ok and rc == 0 and rr == 0 and sj.get("arm") == arm
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    rec[ARM]["repeat"] = "IDENTICAL" if rec[ARM]["train_rc"] == 0 else "DIFFERENT"
    rec["zgs_met_conflicts"] = bool(rec[ARM]["gsurg"] and rec[ARM]["gsurg"]["conflicts"] > 0)
    rec["zret_has_no_surgery"] = rec[BASE_ARM]["gsurg"] is None
    rec["zgs_is_zrets_model"] = rec[ARM]["state_keys"] == rec[BASE_ARM]["state_keys"]
    ok = ok and rec["zgs_met_conflicts"] and rec["zret_has_no_surgery"] and rec["zgs_is_zrets_model"]
    npz = {a: root / f"smoke-{a}" / "reads" / "2wiki__select.npz" for a in runs}
    if all(p.exists() for p in npz.values()):
        r = {a: np.load(p) for a, p in npz.items()}
        ids = {a: [str(x) for x in v["ids"]] for a, v in r.items()}
        rec["same_questions"] = ids[ARM] == ids[BASE_ARM] == ids["gsurg"]
        rec["scores_differ_from_zret"] = bool(not np.array_equal(r[ARM]["scores64"], r[BASE_ARM]["scores64"]))
        rec["scores_differ_from_gsurg"] = bool(not np.array_equal(r[ARM]["scores64"], r["gsurg"]["scores64"]))
        k = [str(x) for x in r[ARM]["candidates"]].index("p@ep0")
        rec["p@ep0_hit@1"] = {a: float(v["hit"][k].mean()) for a, v in r.items()}
        ok = ok and rec["same_questions"] and rec["scores_differ_from_zret"] and rec["scores_differ_from_gsurg"]
    else:
        rec["same_questions"] = None
        ok = False
    for a in runs:
        rec[a].pop("state_keys")
    rec.update({"ok": ok, "seconds": time.time() - t0, "script_sha256": LC.sha_src(__file__)})
    LC.write_json(root / "smoke.json", rec)
    log(f"smoke13: {json.dumps(rec)}; {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm is zret's model on step 1's carve; zret's and gsurg's own arms are unchanged
    assert S.ARMS[ARM] == (S3.ZRet, S.ARMS["base"][1]) and S.ARMS[BASE_ARM] == S.ARMS[ARM]
    assert S.ARMS["gsurg"] == S.ARMS["base"]
    # 2. training swaps in gsurg's loop and zret's model, and restores both
    orig_fit, orig_cls = LG.fit_variant, LG.LeanMLP8D
    with S5.gsurg_loop(), S.patched(ARM):
        assert LG.fit_variant is S5.fit_gsurg and LG.LeanMLP8D is S3.ZRet
    assert LG.fit_variant is orig_fit and LG.LeanMLP8D is orig_cls
    try:
        train(["train", "--name", "x", "--arm", BASE_ARM], "L-musique")
        raise AssertionError("zgs trained another arm")
    except SystemExit as e:
        assert "takes --arm zgs" in str(e)
    # 3. on two toy datasets that disagree: gsurg's loop over zret's model builds a ZRet, meets conflicts and moves the
    #    fit off zret's own loop; without the surgery it is zret's loop bit for bit
    cfg = {"lr": 2e-3, "wd": 1e-4, "dropout": 0.1, "epochs": 2, "swa_from": 1, "cos": 0, "adamw": 0, "drop": 0}
    wd = {"rank": 5, "WALK": 4}
    a = S5.ToyCarve("dsa", np.random.default_rng(1).integers(3, 20, 60), 11, wd, sign=1.0)
    b = S5.ToyCarve("dsb", np.random.default_rng(2).integers(3, 20, 40), 12, wd, sign=-1.0)
    for c in (a, b):                    # zret's base score reads the rrf column: the datasets disagree on it too
        c.F["rank"][:, S3.I_RRF] = c.F["rank"][:, 0]
    blocks = ["rank", "WALK"]
    with S.patched(ARM):
        ref = LG.fit_variant([a, b], blocks, cfg, 0, 16, "none", "cpu")
        off = S5.fit_gsurg([a, b], blocks, cfg, 0, 16, "none", "cpu", surgery=False)
        on = S5.fit_gsurg([a, b], blocks, cfg, 0, 16, "none", "cpu")
        m = LG.LeanMLP8D(blocks, wd, 16)
    assert isinstance(m, S3.ZRet)
    assert not LG.same_state(ref["swa"], off["swa"]) and LG.same_state(ref["swa"], on["swa"])
    assert on["curve"][0]["gsurg"]["conflicts"] > 0
    # 4. relz's pair, re-call and grade under zgs's name, decided against zret's fits; relz restored after (an error
    #    too); restamp names zret and this round
    assert zret_fit("L-musique") == ("screen", "fits", "scr-zret") and zret_fit("J5") == ("full_zret", "fits", "J5")
    assert zret_fit("L-hotpotqa") == ("full_zret", "fits", "L-hotpotqa")
    bc = zret_compares()
    assert sorted(bc) == sorted(S2.SPLITS) and bc["L-hotpotqa"] == ZRET_SCREENS["L-hotpotqa"]
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zret():
        assert Z.tag({})["arm"] == ARM and Z.rel_fit("L-squad") == ("full_zret", "fits", "L-squad")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    try:
        with on_zret():
            raise KeyError("x")
    except KeyError:
        pass
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    SR = Z.SR
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / zret_fit(sp)[0], zret_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZRET_FITS}                                           # zret's fits against step 1's (0.5)
        zb = {sp: "/".join(("outputs",) + zret_fit(sp)) for sp in ZRET_FITS}
        za = SR.fake_compare(T / "z", "scr-zgs", "L-musique", {"musique": 0.0842}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zgs-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM and pr["base_arm"] == BASE_ARM
        assert "scr-zret" in (T / "pz.md").read_text(encoding="utf-8")
        assert json.loads((T / "pz.json").read_text(encoding="utf-8"))["decided_against"] == "zret's fit of each split"
        rc = recall(null1, T / "pz.json", T / "rz", zs)
        assert rc["verdict"] == "PROMISING" and rc["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], rc["changed"]
        assert all(r["base"] == 0.6 for r in rc["rows"])
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "| zret R@5 | zgs R@5 |" in t and "thirteenth round" in t and "rel's" not in t
        st1 = SR.fake_compare(T / "z2", "scr-zgs", "L-musique", {}, arm=ARM)       # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZRET_FITS:
                d = {"musique": 0.02} if sp == "J5" else {}
                SR.fake_compare(full / "src", f"fit-{sp}", sp, d, arm=ARM,
                                base="/".join(("outputs",) + zret_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM
        assert "docs/FULL_ROUND13.md" in (full / "grade.md").read_text(encoding="utf-8")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zgs selftest: the arm is zret's model; training swaps in gsurg's loop and zret's model and restores both; on "
        f"two disagreeing toy datasets the loop builds a ZRet, meets conflicts and moves the fit, and without the surgery "
        f"is zret's loop bit for bit; relz's pair, re-call and grade run under zgs's name against zret's fits (a pair "
        f"against step 1's is refused), name zret and this round, and relz is restored ({time.time() - t0:.1f}s): ok")
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
    if k.cmd == "smoke":
        sp = argparse.ArgumentParser()
        sp.add_argument("--device", default="cpu")
        sp.add_argument("--host", action="store_true")
        sp.add_argument("--out-root")
        s = sp.parse_args(rest)
        return smoke(s.device, s.host, s.out_root)
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return S2.main([k.cmd] + rest)
    if k.cmd in ("pair", "recall", "grade"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--full-root")
        gp.add_argument("--reuse", default="")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out, "grade": g.full_root}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out; "
                     "grade --full-root")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                recall(null, g.pair, g.out)
            else:
                reuse = dict(x.split("=", 1) for x in g.reuse.split(",") if x)
                v = grade(g.full_root, reuse, g.out or str(Path(g.full_root) / "grade"))["verdict"]
                return 1 if v == "INCOMPLETE" else 0
        except SystemExit as e:
            log(f"zgs {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zgs: train, read, compare, pair, recall, grade, smoke, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
