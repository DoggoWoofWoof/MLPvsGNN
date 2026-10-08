"""Screens, twenty-first round (docs/SCREENS.md; full run docs/FULL_ROUND21.md): gsurg's loop on zrm, the base.

zrg is zrm (zrm.ZRM over rmatch.ChainCarveBase: rmatch's gated chain match added to zret's model; the base since
docs/BASE_ZRM_ZRS.md re-graded ADOPT) trained by gsurg's loop (lean_screen5.fit_gsurg: at each step, the gradient on
each training dataset's questions loses its component along the gradient of every other dataset in the step that it
conflicts with). zgs (the thirteenth round) is the same loop on zret's model. The amendment of 8 October
(docs/SCREENS.md, 'Amendment: with zrm the base') sends a PROMISING re-call of zgs to a combination with zrm, declared
before its numbers; this is that round. Both parts are used unchanged: zrm.py's model, settings and training
(rmatch.py's train, lean_screen2's) run inside lean_screen5's gsurg_loop (lean_gpu's fit_variant swapped for fit_gsurg
while a fit trains; each epoch's surgery counts land in train.json's curve). No new column, block or hyperparameter.
The loop acts in training only, so reads and serving are zrm's. On L-metaqa's fit (no typed graph in training) zrm is
zret's model bit for bit, so there zrg is zgs's model and loop.

    python outputs/mp_unified/zrg.py train --split L-musique --name scr-zrg --arm zrg --device cuda --host
    python outputs/mp_unified/zrg.py read --name scr-zrg --device cuda --host
    python outputs/mp_unified/zrg.py compare --new outputs/screen/fits/scr-zrg \\
        --base outputs/screen/fits/scr-zrm,outputs/screen/fits/scr-zgs,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zrg
    python outputs/mp_unified/zrg.py pair --screens outputs/screen/scr-zrg.json,outputs/screen/scr-zrg-hp.json \\
        --out outputs/screen/scr-zrg-pair
    python outputs/mp_unified/zrg.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrg-pair.json \\
        --out outputs/screen/scr-zrg-pair-recall
    python outputs/mp_unified/zrg.py gate --recall outputs/screen/scr-zrg-pair-recall.json \\
        --base outputs/zbase/grade-nullx.json
    python outputs/mp_unified/zrg.py grade --full-root outputs/full_zrg \\
        --reuse L-musique=outputs/screen/scr-zrg.json,L-hotpotqa=outputs/screen/scr-zrg-hp.json
    python outputs/mp_unified/zrg.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zrm's fit of its split: relz.py's pair, re-call and grade under zrg's name with zrm's fits in
place of rel's (zrc.py's mapping: zrm's screen fits scr-zrm and scr-zrm-hp, its full run's fits on the other splits; the
re-call's and the re-grade's base R@5 from outputs/zrc/base-zrm-<split>.json). zgs's screen fits and step 1's are
reported beside. The full run starts only when the screen's re-call is PROMISING and zrm is the base: `gate`.

Speed: the loop changes training only (about 3.5 times lean_gpu's loop per epoch: each step takes every other present
dataset's gradient). zrg reads and serves as zrm does, so any latency figure for it is zrm's, and cold (8216ffe): each
question timed from scratch, the walk, the move of its entries to the device and the forward, with no warm-up pass and
nothing kept from an earlier question.
"""
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import zrc as ZC  # noqa: E402
import lean_screen5 as S5  # noqa: E402

ZM, RM = ZC.ZM, ZC.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
S3 = ZM.S3
log = S.log

ARM, BASE_ARM = "zrg", "zrm"
S.ARMS.update({ARM: (ZM.ZRM, RM.ChainCarveBase)})


# ── train and read ───────────────────────────────────────────────────────────


def looped(fit_dir):
    """True when every epoch of every variant in the fit's train.json carries gsurg's surgery counts."""
    p = Path(fit_dir) / "train.json"
    if not p.exists():
        return False
    v = json.loads(p.read_text(encoding="utf-8")).get("variants", {})
    return bool(v) and all(r.get("curve") and all("gsurg" in e for e in r["curve"]) for r in v.values())


def train(argv, split):
    """rmatch.py's train (zrm's model and settings) for the arm zrg, inside lean_screen5's gsurg_loop. A fit whose
    curve lacks the surgery counts trained outside the loop and stops (exit 1)."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zrg: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZM.ZRM, RM.ChainCarveBase) or S.ARMS.get(BASE_ARM) != S.ARMS.get(ARM):
        raise SystemExit(f"zrg: the arm {ARM} is {S.ARMS.get(ARM)}, not zrm's model on rmatch's chain carve")
    with S5.gsurg_loop():
        rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zrg_sha256": LC.sha_src(__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__), "lean_screen5_sha256": LC.sha_src(S5.__file__),
                    "zrg": {"model": "zrm.ZRM (rmatch.ChainMatch over lean_screen3.ZRet)", "carve": "rmatch.ChainCarveBase",
                            "loop": "lean_screen5.fit_gsurg"}})
        LC.write_json(sj, rec)
    if rc == 0 and not looped(out_root / name):
        raise SystemExit("zrg: the fit's curve has no surgery counts (trained outside gsurg's loop); not this round's arm")
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zrg: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair, re-call and grade, decided against zrm's fits ────────────


@contextlib.contextmanager
def on_zrm():
    """relz.py's records under zrg's name, each read decided against zrm's fit of its split (zrc.py's mapping); relz
    restored after."""
    with ZC.on_zrm():
        Z.ARM = ARM
        yield


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrm's screen fits scr-zrm and scr-zrm-hp)"),
       ("(rel's screen fits)", "(zrm's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrm R@5 | zrg R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-first round"),
       ("docs/FULL_ROUND10.md", "docs/FULL_ROUND21.md"))


def restamp(out):
    """A record relz.py wrote under zrg's name: zrm named as the base in its md, this round and this file's sha added."""
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
        rec.update({"decided_against": "zrm's fit of each split", "round": "twenty-first", "zrg_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrm():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrm_screens=None):
    with on_zrm():
        rec = Z.recall(null_files, pair_file, out, zrm_screens)
    restamp(out)
    return rec


def grade(full_root, reuse, out, check_fits=True):
    with on_zrm():
        rec = Z.grade(full_root, reuse, out, check_fits)
    restamp(out)
    return rec


# ── the full run's gate ──────────────────────────────────────────────────────


def gate(recall_file, base_file):
    """0 when the screen's re-call is PROMISING and zrm is the base (docs/BASE_ZRM_ZRS.md's re-grade ADOPT), 1
    otherwise; a missing file, or one that is not the declared record, stops (2)."""
    got = []
    for fn, arm, want in ((recall_file, ARM, "zrm's fit of each split"), (base_file, "zrm", "zrs's fit of each split")):
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zrg gate: {p} missing")
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("arm") != arm or rec.get("decided_against") != want:
            raise SystemExit(f"zrg gate: {p} is arm {rec.get('arm')}'s, decided against {rec.get('decided_against')!r}; "
                             "not the declared record")
        got.append(rec.get("verdict"))
    if "null_splits" not in json.loads(Path(base_file).read_text(encoding="utf-8")):
        raise SystemExit(f"zrg gate: {base_file} is not a re-grade under the null")
    ok = got[0] == "PROMISING" and got[1] == "ADOPT"
    log(f"zrg gate: the re-call {got[0]}, zrm against zrs {got[1]} -> {'the full run starts' if ok else 'no full run'}")
    return 0 if ok else 1


# ── selftest ─────────────────────────────────────────────────────────────────


class Signed:
    """A toy carve whose golds lean on its first rank column with the given sign, with or without its chains: two of
    them on the same rows and golds disagree on that column (lean_screen5's toy datasets disagree the same way)."""

    def __init__(self, c, ds, sign, chains):
        self.c, self.ds, self.sign, self.keep = c, ds, sign, chains
        self.carve, self.chains = c.carve, (c.chains if chains else None)
        self.rows, self.widths, self.n_np, self.off_np = c.rows, c.widths, c.n_np, c.off_np

    def nbytes(self):
        return 0

    def batch(self, qs, blocks):
        feats, nq, bz, gold = self.c.batch(qs, blocks)
        feats = {k: v for k, v in feats.items() if self.keep or k != RM.CH_KEY}
        r = feats["rank"].clone()
        r[:, 0] += self.sign * 2.5 * gold.to(r.dtype)
        feats["rank"] = r
        return feats, nq, bz, gold


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm is zrm's model on rmatch's chain carve; zrm's, zgs's and gsurg's own arms unchanged; train takes no
    #    other arm
    assert S.ARMS[ARM] == (ZM.ZRM, RM.ChainCarveBase) == S.ARMS[BASE_ARM]
    assert S.ARMS["zgs"] == (S3.ZRet, S.ARMS["base"][1]) and S.ARMS["gsurg"] == S.ARMS["base"]
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--name", "x"], "L-musique")
    # 2. training swaps in gsurg's loop and zrm's model and carve, and restores all three
    orig = LG.fit_variant, LG.LeanMLP8D, LG.CacheCarve
    with S5.gsurg_loop(), S.patched(ARM):
        assert LG.fit_variant is S5.fit_gsurg and LG.LeanMLP8D is ZM.ZRM and LG.CacheCarve is RM.ChainCarveBase
    assert (LG.fit_variant, LG.LeanMLP8D, LG.CacheCarve) == orig
    tmp = Path(tempfile.mkdtemp(prefix="zrg_"))
    try:
        # 3. on rmatch's toy chain carve (typed) and the same rows untyped, the two leaning opposite ways on one rank
        #    column: gsurg's loop over zrm's model is zrm's own loop bit for bit until its first conflict and moves
        #    every state after it, its repeat identical; without the surgery it is zrm's loop bit for bit; the match's
        #    gates train
        rng = np.random.default_rng(19)
        RM.toy_roots(tmp, rng, chunks=(30, 30))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert c.chains is not None
        tr = [Signed(c, "metaqa", 1.0, True), Signed(c, "2wiki", -1.0, False)]
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        with S.patched(ARM):
            assert LG.fit_variant is orig[0]
            ref = LG.fit_variant(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrm/p")
            off = S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrg/p no surgery", surgery=False)
            on = [S5.fit_gsurg(tr, blocks, cfg, 0, 16, "none", "cpu", tag=f"toy-zrg/p {r}") for r in range(2)]
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        assert sorted(on[0]["swa"]) == sorted(ref["swa"])
        assert all(not LG.same_state(a, b) for a, b in zip(every(off), every(ref)))
        first = next((e for e, r in enumerate(on[0]["curve"]) if r["gsurg"]["conflicts"] > 0), None)
        assert first is not None, [r["gsurg"] for r in on[0]["curve"]]
        assert all(not LG.same_state(a, b) for a, b in zip(on[0]["states"][:first], ref["states"][:first]))
        assert all(LG.same_state(a, b) for a, b in zip(every(on[0])[first:], every(ref)[first:]))
        assert all(not LG.same_state(a, b) for a, b in zip(every(on[0]), every(on[1])))
        assert all(e["gsurg"]["pairs"] > 0 for e in on[0]["curve"])
        assert float(on[0]["swa"]["cm_gate"].abs().sum()) > 0
        # 4. the curve check: surgery counts in every epoch pass, a curve without them (lean_gpu's loop) does not
        for name, f in (("on", on[0]), ("ref", ref)):
            (tmp / "fits" / name).mkdir(parents=True)
            LC.write_json(tmp / "fits" / name / "train.json", {"variants": {"p": {"curve": f["curve"]}}})
        assert looped(tmp / "fits" / "on") and not looped(tmp / "fits" / "ref") and not looped(tmp / "fits" / "none")
        # 5. read refuses another arm's fit
        LC.write_json(tmp / "fits" / "ref" / "screen.json", {"arm": BASE_ARM})
        try:
            read(["--name", "ref", "--out-root", str(tmp / "fits")])
            raise AssertionError("zrg read zrm's fit")
        except SystemExit as e:
            assert "trained as zrm" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 6. relz's pair, re-call and grade under zrg's name, decided against zrm's fits; relz restored after
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrm():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
        assert Z.rel_fit("J5") == ("full_zrm", "fits", "J5") and Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrm")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)

        # 7. the gate: PROMISING and ADOPT -> 0; anything else -> 1; a wrong or missing record stops
        def rec(name, **kw):
            p = T / f"{name}.json"
            p.write_text(json.dumps(kw), encoding="utf-8")
            return str(p)

        rc_p = rec("rp", arm=ARM, decided_against="zrm's fit of each split", verdict="PROMISING")
        rc_m = rec("rm", arm=ARM, decided_against="zrm's fit of each split", verdict="MIXED")
        rc_g = rec("rg", arm="zgs", decided_against="zret's fit of each split", verdict="PROMISING")
        b_a = rec("ba", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT", null_splits=["J5"])
        b_n = rec("bn", arm="zrm", decided_against="zrs's fit of each split", verdict="NOT_ADOPTED", null_splits=["J5"])
        b_f = rec("bf", arm="zrm", decided_against="zrs's fit of each split", verdict="ADOPT")         # filed, not nullx
        assert gate(rc_p, b_a) == 0 and gate(rc_m, b_a) == 1 and gate(rc_p, b_n) == 1
        SR.must_stop(gate, rc_g, b_a)
        SR.must_stop(gate, rc_p, b_f)
        SR.must_stop(gate, b_a, rc_p)
        SR.must_stop(gate, rc_p, str(T / "none.json"))
        # 8. the pair, the re-call and the grade against zrm's fits, named for this round
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / ZC.zrm_fit(sp)[0], ZC.zrm_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZC.ZRM_FITS}                                            # zrm's fits against step 1's (0.6)
        zb = {sp: "/".join(("outputs",) + ZC.zrm_fit(sp)) for sp in ZC.ZRM_FITS}
        za = SR.fake_compare(T / "z", "scr-zrg", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zrg-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        st1 = SR.fake_compare(T / "z2", "scr-zrg", "L-musique", {}, arm=ARM)        # decided against step 1's fit
        SR.must_stop(pair, [st1, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-first" and pj["decided_against"] == "zrm's fit of each split"
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "twenty-first round" in t and "tenth round" not in t and "| zrm R@5 | zrg R@5 |" in t
        assert gate(str(T / "rz.json"), b_a) == 0
        full = T / "full"
        full.mkdir()
        for sp in S2.SPLITS:
            if sp not in ZC.ZRM_FITS:
                SR.fake_compare(full / "src", f"fit-{sp}", sp, {"webqsp": 0.03} if sp == "J5" else {}, arm=ARM,
                                base="/".join(("outputs",) + ZC.zrm_fit(sp))).replace(full / f"compare-{sp}.json")
        g = grade(full, {"L-musique": str(za), "L-hotpotqa": str(zh)}, full / "grade", check_fits=False)
        assert g["verdict"] == "NOT_ADOPTED" and g["losses"] == 1 and g["arm"] == ARM, (g["verdict"], g["losses"])
        t = (full / "grade.md").read_text(encoding="utf-8")
        assert "docs/FULL_ROUND21.md" in t and "FULL_ROUND10" not in t
    log(f"zrg selftest: the arm is zrm's model on rmatch's chain carve and train takes no other; gsurg's loop over zrm's "
        f"model is zrm's loop until its first conflict and moves every state after it, its repeat identical, and without "
        f"the surgery is zrm's loop bit for bit; a curve without surgery counts stops a fit; the gate needs a PROMISING re-call and zrm "
        f"ADOPTED against zrs under the null; relz's pair, re-call and grade run against zrm's fits and name this round "
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
    if k.cmd == "train":
        return train([k.cmd] + rest, k.split)          # as zrm.py's main
    if k.cmd == "read":
        return read(rest)
    if k.cmd == "compare":
        return compare(rest)
    if k.cmd == "gate":
        gp = argparse.ArgumentParser()
        gp.add_argument("--recall", required=True)
        gp.add_argument("--base", required=True)
        g = gp.parse_args(rest)
        try:
            return gate(g.recall, g.base)
        except SystemExit as e:
            log(f"zrg gate: {e}")
            return 2
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
            log(f"zrg {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zrg: train, read, compare, pair, recall, gate, grade, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
