"""Screens, twenty-eighth round (docs/SCREENS.md): zrc's entries under zkind's head, decided against zrc, the base.

zck is zkind's model (zkind.ZKind: zrm.ZRM whose output layer's weights and bias and rrf's base weight take learned
offsets on a batch with chain entries, zero at the start) over zrc's builds (zrc.ChainCarveZRC: each typed row keeps the
64 entries w0 ranks highest). zrc is the base since docs/BASE_ZRC_ZKIND.md (NOT_ADOPTED 08:01: zkind's fits level with
zrc's). Both declarations name this combination as a round of its own: zrc changes which entries a typed row keeps, and
zkind the head that reads them. The offsets add no draw, so from the same seed zck is zrc's model and state plus three
zero offsets, and its forward is zrc's bit for bit until an offset moves. No new column, block, carve, graph or
hyperparameter; settings and training are zrm's (rmatch.py's train).

    python outputs/mp_unified/zck.py train --split L-musique --name scr-zck --arm zck --device cuda --host
    python outputs/mp_unified/zck.py read --name scr-zck --device cuda --host
    python outputs/mp_unified/zck.py compare --new outputs/screen/fits/scr-zck \\
        --base outputs/screen/fits/scr-zrct,outputs/screen/fits/scr-zrm,outputs/step1/fits/L-musique \\
        --out outputs/screen/scr-zck
    python outputs/mp_unified/zck.py pair --screens outputs/screen/scr-zck.json,outputs/screen/scr-zck-hp.json \\
        --out outputs/screen/scr-zck-pair
    python outputs/mp_unified/zck.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zck-pair.json \\
        --out outputs/screen/scr-zck-pair-recall
    python outputs/mp_unified/zck.py --selftest
N1..N4 are the seed null's comparisons (outputs/screen/scr-null-s1, -s2, -s1-hp, -s2-hp .json).

Each read is decided against zrc's card fit of its split (scr-zrct, scr-zrct-hp): relz.py's pair and re-call under zck's
name with zrc's fits in place of rel's; the re-call's base R@5 from docs/BASE_ZRC_ZKIND.md's
outputs/zbase2/base-zrc-<split>.json (zrc's fits against step 1's). zrm's and step 1's fits are reported beside.

Speed: zkind's offsets add one product per row on a typed graph; zrc's entries are built ahead as zrc's are. Any latency
figure for zck is cold (8216ffe), timed in its own declared stage.
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

import zkind as ZK  # noqa: E402

ZC, ZM, RM = ZK.ZC, ZK.ZM, ZK.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
S3 = ZM.S3
log = S.log

ARM, BASE_ARM = "zck", "zrc"
DECIDED = "zrc's fit of each split"
ZRC_FITS = {"L-musique": ("screen", "fits", "scr-zrct"), "L-hotpotqa": ("screen", "fits", "scr-zrct-hp")}
ZRC_SCREENS = {sp: f"outputs/zbase2/base-zrc-{sp}.json" for sp in ZRC_FITS}

S.ARMS.update({ARM: (ZK.ZKind, ZC.ChainCarveZRC)})


def zrc_fit(split):
    """zrc's fit of a split: its screen fits on L-musique and L-hotpotqa, its full run's (outputs/full_zrct) elsewhere."""
    return ZRC_FITS.get(split, ("full_zrct", "fits", split))


# ── train, read and compare ──────────────────────────────────────────────────


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zck: zkind's model over zrc's builds; records stamped."""
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zck: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZK.ZKind, ZC.ChainCarveZRC) or S.ARMS.get(BASE_ARM) != (ZM.ZRM, ZC.ChainCarveZRC):
        raise SystemExit(f"zck: the arm {ARM} is {S.ARMS.get(ARM)}, not zkind's model over zrc's builds")
    rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zck_sha256": LC.sha_src(__file__), "zkind_sha256": LC.sha_src(ZK.__file__),
                    "zrc_sha256": LC.sha_src(ZC.__file__), "zrm_sha256": LC.sha_src(ZM.__file__),
                    "lean_screen3_sha256": LC.sha_src(S3.__file__),
                    "zck": {"model": "zkind.ZKind (zrm.ZRM with offsets on a typed graph's output layer and base "
                                     "weight)", "carve": "zrc.ChainCarveZRC", "entries": "outputs/zrc/cache",
                            "rule": ZC.RULE, "offsets": list(ZK.KD)},
                    "zrc_records": RM.built_records(ZC.CH_OUT)})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zck: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair and re-call, decided against zrc's fits ───────────────────


@contextlib.contextmanager
def on_zrc():
    """relz.py's records under zck's name, each read decided against zrc's fit of its split; relz restored after."""
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZRC_FITS, ZRC_SCREENS, zrc_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrc's screen fits scr-zrct and scr-zrct-hp)"),
       ("(rel's screen fits)", "(zrc's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrc R@5 | zck R@5 |"),
       ("section 2 and the tenth round", "section 2 and the twenty-eighth round"))


def restamp(out):
    """A record relz.py wrote under zck's name: zrc named as the base in its md, this round and this file's sha added."""
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
        rec.update({"decided_against": DECIDED, "round": "twenty-eighth", "zck_sha256": LC.sha_src(__file__)})
        LC.write_json(js, rec)


def pair(screens, out):
    with on_zrc():
        rec = Z.pair(screens, out)
    restamp(out)
    return rec


def recall(null_files, pair_file, out, zrc_screens=None):
    with on_zrc():
        rec = Z.recall(null_files, pair_file, out, zrc_screens)
    restamp(out)
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    # 1. the arm: zkind's model over zrc's builds; zrc's arm unchanged; train takes no other arm
    assert S.ARMS[ARM] == (ZK.ZKind, ZC.ChainCarveZRC) and S.ARMS[BASE_ARM] == (ZM.ZRM, ZC.ChainCarveZRC)
    assert ZC.ChainCarveZRC.CH_ROOT == ZC.CH_OUT and ZK.ZKind.__mro__[1] is ZM.ZRM
    SR.must_stop(train, ["train", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--arm", "zkind", "--name", "x"], "L-musique")
    assert zrc_fit("J5") == ("full_zrct", "fits", "J5") and zrc_fit("L-hotpotqa") == ("screen", "fits", "scr-zrct-hp")
    tmp = Path(tempfile.mkdtemp(prefix="zck_"))
    try:
        # 2. from the same seed zck is zrc's state plus zkind's three zero offsets, and its forward is zrc's bit for
        #    bit on a typed batch, in eval and with the same dropout draws in training (a toy chain carve built as
        #    zrc's builds are read: rmatch's chain_carve over a build root)
        rng = np.random.default_rng(28)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        c = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        assert c.chains is not None
        blocks = ["rank", "SEMB"]
        qs = np.arange(min(c.rows, 8), dtype=np.int64)
        B = qs.size
        keep = torch.ones(B, len(blocks))
        feats, nq, bz, _g = c.batch(qs, blocks)
        assert RM.CH_KEY in feats
        nets = {}
        for k, cls in ((BASE_ARM, ZM.ZRM), (ARM, ZK.ZKind)):
            torch.manual_seed(0)
            nets[k] = cls(blocks, c.widths, 16, dropout=0.1, seed=0)
        sz, sk = nets[BASE_ARM].state_dict(), nets[ARM].state_dict()
        assert sorted(set(sk) - set(sz)) == sorted(ZK.KD) and all(torch.equal(sz[k], sk[k]) for k in sz)
        with torch.no_grad():
            for train_mode in (False, True):
                out = {}
                for k, m in nets.items():
                    m.train(train_mode)
                    torch.manual_seed(5)
                    out[k] = m(feats, keep, nq, B, bz)
                assert torch.equal(out[ARM], out[BASE_ARM]), train_mode
        # 3. read refuses another arm's fit
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": "zkind"})
        try:
            read(["--name", "x", "--out-root", str(tmp / "fits")])
            raise AssertionError("zck read zkind's fit")
        except SystemExit as e:
            assert "trained as zkind" in str(e)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # 4. relz's names swapped for zrc's fits and restored, after an error too
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrc():
        assert Z.tag({})["arm"] == ARM and Z.BASE_ARM == BASE_ARM
        assert Z.rel_fit("J5") == ("full_zrct", "fits", "J5") and Z.rel_fit("L-musique") == ("screen", "fits", "scr-zrct")
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    try:
        with on_zrc():
            raise KeyError("x")
    except KeyError:
        pass
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    # 5. the pair and the re-call against zrc's screen fits, named for this round; a pair decided against zrm's stops
    with tempfile.TemporaryDirectory() as td:
        T = Path(td)
        null1 = SR.fake_null(T / "null", "n1", {("L-hotpotqa", "hotpotqa"): (-0.010, 0.012)})   # floor 0.0221
        zs = {sp: str(SR.fake_compare(T / zrc_fit(sp)[0], zrc_fit(sp)[2], sp, {ds: 0.1 for ds in LG.EVAL_ORDER},
                                      arm=BASE_ARM))
              for sp in ZRC_FITS}                                               # zrc's fits against step 1's (0.6)
        zb = {sp: "/".join(("outputs",) + zrc_fit(sp)) for sp in ZRC_FITS}
        za = SR.fake_compare(T / "z", "scr-zck", "L-musique", {"webqsp": 0.0215}, arm=ARM, base_r5=0.6,
                             base=zb["L-musique"])
        zh = SR.fake_compare(T / "z", "scr-zck-hp", "L-hotpotqa", {"hotpotqa": -0.0147}, arm=ARM, base_r5=0.6,
                             base=zb["L-hotpotqa"])
        zm = SR.fake_compare(T / "z2", "scr-zck", "L-musique", {}, arm=ARM, base_r5=0.6,
                             base="outputs/screen/fits/scr-zrm")                  # decided against zrm's fit
        SR.must_stop(pair, [zm, zh], T / "bad")
        pr = pair([za, zh], T / "pz")
        assert pr["verdict"] == "MIXED" and pr["arm"] == ARM
        pj = json.loads((T / "pz.json").read_text(encoding="utf-8"))
        assert pj["round"] == "twenty-eighth" and pj["decided_against"] == DECIDED
        r = recall(null1, T / "pz.json", T / "rz", zs)
        assert r["verdict"] == "PROMISING" and r["changed"] == ["L-hotpotqa hotpotqa LOSS -> WITHIN"], r["changed"]
        t = (T / "rz.md").read_text(encoding="utf-8")
        assert "twenty-eighth round" in t and "tenth round" not in t and "| zrc R@5 | zck R@5 |" in t
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zck selftest: the arm is zkind's model over zrc's builds and train takes no other; from the same seed it is "
        f"zrc's state plus zero offsets and zrc's forward bit for bit; read refuses another arm's fit; relz's pair and "
        f"re-call run against zrc's screen fits, name this round, and refuse a pair decided against zrm's "
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
    if k.cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out")
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        need = {"pair": g.screens and g.out, "recall": g.pair and len(null) == 4 and g.out}[k.cmd]
        if not need:
            gp.error(f"{k.cmd}: pair needs --screens A,B and --out; recall --pair, the four --null files and --out")
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            else:
                recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"zck {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zck: train, read, compare, pair, recall, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
