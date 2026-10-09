"""Diagnostic D3 (docs/DIAG_DATA_SCALE.md): is zrc, the MLP's base, limited by its training data?

The fit carves are capped at 6,000 questions per dataset (outputs/m3b/carves.json, "why_capped": a CPU-era compute
bound), drawn by a stride from train pools of 20k to 330k. Before building more fit carves (looks, caches and chain
builds for ~26k new questions: CPU, memory and disk the link and pool stages hold now), this asks whether more would
help, from the other side: zrc refit on a stride of its own fit carves, every 2nd and every 4th question of each
training dataset (the carve rule's own stride, in carve order), with everything else zrc's (zrm.ZRM over
zrc.ChainCarveZRC, rmatch.py's train, zrm's settings, seed 0, p@swa, the same epochs). A fit's other inputs (the
context statistics, the select carve) are its subset's or unchanged; no row, column, pool or score is new.

    python outputs/mp_unified/zds.py train --split L-musique --every 2 --name dsx-h2 --arm zds --device cuda --host
    python outputs/mp_unified/zds.py read --name dsx-h2 --device cuda --host
    python outputs/mp_unified/zds.py compare --new outputs/screen/fits/dsx-h2 \\
        --base outputs/screen/fits/scr-zrct,outputs/step1/fits/L-musique --out outputs/screen/dsx-h2
    python outputs/mp_unified/zds.py pair --screens outputs/screen/dsx-h2.json,outputs/screen/dsx-h2-hp.json \\
        --out outputs/screen/dsx-h2-pair
    python outputs/mp_unified/zds.py recall --null N1,N2,N3,N4 --pair outputs/screen/dsx-h2-pair.json \\
        --out outputs/screen/dsx-h2-pair-recall
    python outputs/mp_unified/zds.py decide --half outputs/screen/dsx-h2-pair-recall.json \\
        --quarter outputs/screen/dsx-q4-pair-recall.json --out outputs/screen/dsx-decide
    python outputs/mp_unified/zds.py --selftest

Each read is the subset fit minus zrc's fit of its split (scr-zrct, scr-zrct-hp), re-called under the seed null as a
screen's. Here a LOSS says the full carve beats its subset by more than seed noise. `decide` applies the declared rule
(docs/DIAG_DATA_SCALE.md) to the half fits; the quarter fits are reported for the curve's shape only.
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

import zof as ZO  # noqa: E402

ZC, ZM, RM = ZO.ZC, ZO.ZM, ZO.RM
Z, R = RM.Z, RM.R
S2, S = RM.S2, RM.S
LG, LC = S.LG, S.LC
SR = Z.SR
log = S.log

ARM, BASE_ARM = "zds", "zrc"
DECIDED = ZO.DECIDED
EVERY = (1, 2, 4)
MIN_LOSS = 2
S.ARMS.update({ARM: (ZM.ZRM, ZC.ChainCarveZRC)})


# ── the subset ───────────────────────────────────────────────────────────────


class Sub:
    """A training carve seen through a stride: question j of the subset is question keep[j] of the carve. Every other
    attribute is the carve's own."""

    def __init__(self, carve, every):
        self._c = carve
        self.keep = np.arange(0, carve.rows, every, dtype=np.int64)
        self.rows = int(self.keep.size)

    def __getattr__(self, k):
        return getattr(self._c, k)

    def batch(self, qs, *a, **kw):
        return self._c.batch(self.keep[np.asarray(qs, np.int64)], *a, **kw)


class subset:
    """lean_gpu's fit loop given each training carve through Sub(every); restored on the way out. The counts of the
    last fit are kept for the record."""

    def __init__(self, every):
        if every not in EVERY:
            raise SystemExit(f"zds: --every takes {EVERY}, not {every}")
        self.every, self.last = every, []

    def __enter__(self):
        self.saved = orig = LG.fit_variant

        def fit_variant(tr, *args, **kw):
            sub = [Sub(c, self.every) for c in tr]
            self.last = [{"dataset": c.ds, "carve": c.carve, "questions": int(c.rows), "kept": s.rows}
                         for c, s in zip(tr, sub)]
            log("  zds: every %d: %s" % (self.every, ", ".join(f"{x['dataset']} {x['kept']}/{x['questions']}"
                                                                for x in self.last)))
            return orig(sub, *args, **kw)

        LG.fit_variant = fit_variant
        return self

    def __exit__(self, *exc):
        LG.fit_variant = self.saved
        return False


# ── train, read, compare ─────────────────────────────────────────────────────


def every_of(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--every", type=int)
    k, rest = ap.parse_known_args(argv)
    if k.every is None:
        raise SystemExit("zds: train needs --every 2 or 4")
    return k.every, rest


def train(argv, split):
    """rmatch.py's train (zrm's settings) for the arm zds, zrc's model and builds, on every k-th fit question."""
    every, argv = every_of(argv)
    arm = R.arm_of(argv)
    if arm != ARM:
        raise SystemExit(f"zds: train takes --arm {ARM}, not {arm}")
    if S.ARMS.get(ARM) != (ZM.ZRM, ZC.ChainCarveZRC) or S.ARMS.get(BASE_ARM) != S.ARMS[ARM]:
        raise SystemExit(f"zds: the arm {ARM} is {S.ARMS.get(ARM)}, not zrc's model on zrc's carve")
    with subset(every) as sb:
        rc = RM.train(argv, split)
    name, out_root = S2.where(argv)
    sj = out_root / name / "screen.json"
    if sj.exists():
        rec = json.loads(sj.read_text(encoding="utf-8"))
        rec.update({"zds_sha256": LC.sha_src(__file__), "zrc_sha256": LC.sha_src(ZC.__file__),
                    "zrm_sha256": LC.sha_src(ZM.__file__),
                    "zds": {"model": "zrm.ZRM over zrc.ChainCarveZRC (zrc's arm)", "every": every,
                            "subset": "every k-th question of each fit carve, in carve order", "last_fit": sb.last},
                    "zrc_records": RM.built_records(ZC.CH_OUT)})
        LC.write_json(sj, rec)
    return rc


def read(argv):
    name, out_root = S2.where(argv)
    arm = json.loads((out_root / name / "screen.json").read_text(encoding="utf-8"))["arm"]
    if arm != ARM:
        raise SystemExit(f"zds: {name} was trained as {arm}, not {ARM}")
    return RM.read(argv)


def compare(rest):
    return S2.main(["compare"] + rest)


# ── relz.py's pair and re-call against zrc's fits (zof's), under this name ───


@contextlib.contextmanager
def on_zrc():
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = ARM, BASE_ARM, ZO.ZRC_FITS, ZO.ZRC_SCREENS, ZO.zrc_fit
    try:
        yield
    finally:
        Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit = saved


FIX = (("(rel's screen fits scr-rel and scr-rel-hp)", "(zrc's screen fits scr-zrct and scr-zrct-hp)"),
       ("(rel's screen fits)", "(zrc's screen fits)"),
       ("| rel R@5 | relz R@5 |", "| zrc R@5 | subset R@5 |"),
       ("section 2 and the tenth round", "docs/DIAG_DATA_SCALE.md"))


def restamp(out):
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
        rec.update({"decided_against": DECIDED, "diagnostic": "D3 data scale", "zds_sha256": LC.sha_src(__file__)})
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


# ── the declared rule ────────────────────────────────────────────────────────


def tally(rec):
    rows = rec["rows"]
    return {"loss": [f"{r['split']} {r['dataset']} {r['read']}" for r in rows if r["call"] == "LOSS"],
            "gain": [f"{r['split']} {r['dataset']} {r['read']}" for r in rows if r["call"] == "GAIN"],
            "mean_delta": float(np.mean([r["delta"] for r in rows])) if rows else 0.0, "reads": len(rows)}


def decide(half, quarter, out):
    """DATA_LIMITED when the half fits read LOSS against zrc's under the seed null on MIN_LOSS or more of their reads,
    NOT_DATA_LIMITED otherwise. The quarter fits are reported only."""
    recs = {}
    for k, fn in (("half", half), ("quarter", quarter)):
        if fn is None:
            continue
        p = Path(fn)
        if not p.exists():
            raise SystemExit(f"zds decide: {p} missing")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("arm") != ARM or r.get("decided_against") != DECIDED or "rows" not in r:
            raise SystemExit(f"zds decide: {p} is not a zds re-call against {DECIDED}")
        recs[k] = r
    if "half" not in recs:
        raise SystemExit("zds decide: the half fits' re-call is required")
    t = {k: tally(r) for k, r in recs.items()}
    v = "DATA_LIMITED" if len(t["half"]["loss"]) >= MIN_LOSS else "NOT_DATA_LIMITED"
    rec = {"verdict": v, "rule": f"DATA_LIMITED if the half fits read LOSS (seed null) on >= {MIN_LOSS} reads",
           "half": t["half"], "quarter": t.get("quarter"), "files": {k: str(Path(f)) for k, f in
                                                                    (("half", half), ("quarter", quarter)) if f},
           "zds_sha256": LC.sha_src(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# D3 data scale: zrc on a stride of its fit carves", "",
          f"**{v}** ({rec['rule']}; docs/DIAG_DATA_SCALE.md).", "",
          "| split | dataset | read | zrc R@5 | half delta | half call | quarter delta | quarter call |",
          "| --- | --- | --- | ---: | ---: | --- | ---: | --- |"]
    qrows = {(r["split"], r["dataset"], r["read"]): r for r in recs.get("quarter", {}).get("rows", [])}
    for r in recs["half"]["rows"]:
        q = qrows.get((r["split"], r["dataset"], r["read"]))
        qd = f"{q['delta']:+.4f}" if q else ""
        md.append(f"| {r['split']} | {r['dataset']} | {r['read']} | {r['base']:.4f} | {r['delta']:+.4f} | "
                  f"{r['call']} | {qd} | {q['call'] if q else ''} |")
    md += ["", f"Half: {len(t['half']['loss'])} LOSS, {len(t['half']['gain'])} GAIN of {t['half']['reads']} reads, "
           f"mean delta {t['half']['mean_delta']:+.4f}."]
    if "quarter" in t:
        md.append(f"Quarter: {len(t['quarter']['loss'])} LOSS, {len(t['quarter']['gain'])} GAIN of "
                  f"{t['quarter']['reads']} reads, mean delta {t['quarter']['mean_delta']:+.4f}.")
    out = Path(out)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"zds decide: {v}; half {len(t['half']['loss'])} LOSS of {t['half']['reads']}")
    return rec


# ── selftest ─────────────────────────────────────────────────────────────────


class Spy:
    """Records every question index a fit asks its carve for."""

    def __init__(self, carve):
        self._c, self.asked = carve, []

    def __getattr__(self, k):
        return getattr(self._c, k)

    def batch(self, qs, *a, **kw):
        self.asked.append(np.asarray(qs, np.int64).copy())
        return self._c.batch(qs, *a, **kw)


def selftest():
    t0 = time.time()
    LG.bind_device_ops()
    SR.must_stop(train, ["train", "--arm", ARM, "--name", "x"], "L-musique")            # no --every
    SR.must_stop(train, ["train", "--every", "3", "--arm", ARM, "--name", "x"], "L-musique")
    SR.must_stop(train, ["train", "--every", "2", "--arm", BASE_ARM, "--name", "x"], "L-musique")
    tmp = Path(tempfile.mkdtemp(prefix="zds_"))
    try:
        rng = np.random.default_rng(33)
        RM.toy_roots(tmp, rng, chunks=(12, 12))
        RM.build("metaqa", "toy", out_root=tmp / "ch", cache_root=tmp / "cache", look_root=tmp / "look",
                 rel_dir=tmp / "rel")
        cls = RM.chain_carve(ZM.ToyZ, tmp / "ch", tmp / "rel")
        c = cls("metaqa", "toy", "2wiki", "cpu", tmp / "cache")
        blocks = ["rank", "SEMB"]
        cfg = {"lr": 2e-2, "wd": 1e-4, "dropout": 0.1, "epochs": 3, "swa_from": 1, "cos": False, "adamw": False,
               "drop": False}
        every = lambda f: f["states"] + [f["swa"]]  # noqa: E731
        saved_arms = dict(S.ARMS)
        S.ARMS.update({ARM: (ZM.ZRM, cls)})
        try:
            with S.patched(ARM):
                ref = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-zrc")
                # 1. every 1 is the plain fit bit for bit
                with subset(1):
                    one = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-every1")
                assert all(torch.equal(a[k], b[k]) for a, b in zip(every(ref), every(one)) for k in a)
                # 2. every 2: the loop asks only for the kept questions, each once per epoch; the fit moves
                spy = Spy(c)
                with subset(2) as sb:
                    two = LG.fit_variant([spy], blocks, cfg, 0, 16, "none", "cpu", tag="toy-every2")
                kept = np.arange(0, c.rows, 2)
                asked = np.concatenate(spy.asked)
                assert set(asked.tolist()) == set(kept.tolist()), (sorted(set(asked.tolist())), kept)
                assert sb.last == [{"dataset": "metaqa", "carve": "toy", "questions": c.rows, "kept": kept.size}]
                assert not all(torch.equal(a[k], b[k]) for a, b in zip(every(ref), every(two)) for k in a)
                # 3. a repeat is identical; the loop is restored on the way out
                with subset(2):
                    rep = LG.fit_variant([c], blocks, cfg, 0, 16, "none", "cpu", tag="toy-every2 repeat")
                assert all(torch.equal(a[k], b[k]) for a, b in zip(every(two), every(rep)) for k in a)
        finally:
            S.ARMS.clear()
            S.ARMS.update(saved_arms)
        assert LG.fit_variant.__name__ == "fit_variant" and LG.fit_variant.__module__ == LG.__name__
        (tmp / "fits" / "x").mkdir(parents=True)
        LC.write_json(tmp / "fits" / "x" / "screen.json", {"arm": BASE_ARM})
        SR.must_stop(read, ["--name", "x", "--out-root", str(tmp / "fits")])
        # 4. the rule: >= MIN_LOSS LOSS reads in the half fits -> DATA_LIMITED; a wrong record stops
        def rr(name, calls, arm=ARM):
            rows = [{"split": "L-musique", "dataset": f"d{i}", "read": "in-domain", "base": 0.5, "delta": -0.01,
                     "call": cl} for i, cl in enumerate(calls)]
            p = tmp / f"{name}.json"
            LC.write_json(p, {"arm": arm, "decided_against": DECIDED, "rows": rows})
            return str(p)

        assert decide(rr("h2", ["LOSS", "LOSS", "WITHIN"]), None, tmp / "d1")["verdict"] == "DATA_LIMITED"
        assert decide(rr("h1", ["LOSS", "WITHIN", "GAIN"]), rr("q", ["LOSS"] * 3), tmp / "d2")["verdict"] == \
            "NOT_DATA_LIMITED"
        assert "| quarter delta |" in (tmp / "d2.md").read_text(encoding="utf-8")
        SR.must_stop(decide, rr("bad", ["LOSS"] * 3, arm="zof"), None, tmp / "d3")
        SR.must_stop(decide, str(tmp / "none.json"), None, tmp / "d4")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    saved = Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit
    with on_zrc():
        assert Z.ARM == ARM and Z.REL_FITS == ZO.ZRC_FITS
    assert (Z.ARM, Z.BASE_ARM, Z.REL_FITS, Z.REL_SCREENS, Z.rel_fit) == saved
    log(f"zds selftest: every 1 is zrc's fit bit for bit; every 2 asks only for the kept questions and moves the fit; "
        f"repeats are identical; the rule and its records ({time.time() - t0:.1f}s): ok")
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
    if k.cmd in ("pair", "recall", "decide"):
        gp = argparse.ArgumentParser()
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--half")
        gp.add_argument("--quarter")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(rest)
        null = [x for x in g.null.split(",") if x]
        try:
            if k.cmd == "pair":
                pair([x for x in g.screens.split(",") if x], g.out)
            elif k.cmd == "recall":
                if not g.pair or len(null) != 4:
                    gp.error("recall needs --pair and the four --null files")
                recall(null, g.pair, g.out)
            else:
                decide(g.half, g.quarter, g.out)
        except SystemExit as e:
            log(f"zds {k.cmd}: {e}")
            return 2
        return 0
    raise SystemExit("zds: train, read, compare, pair, recall, decide, or --selftest")


if __name__ == "__main__":
    sys.exit(main())
