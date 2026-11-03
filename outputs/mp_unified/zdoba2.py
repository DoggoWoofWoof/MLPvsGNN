"""DOB amendment A2 (docs/DOB_DROP_ONE_BLOCK_CPU.md section 10): the xrank arm withdrawn, before any xrank fit trained.
zrckp's model (lean_screen3's zret) z-scores every input against the pool's retrieved rows, which the rank block's rrf
column marks, so the rank block cannot leave the pick; it stays in every arm, as SEMB does.

zdob.py is imported and not edited: its sha is in every DOB fit's code record, and section 4 checks each fit's code
against its base's. For the length of the compares or the calls, zdob's DROPS, FITS, DROP_OF and PAIRS lose xrank's two
fits and two compares, and are put back afterwards, also on a failure. The calls record and its page name the amendment.

    python outputs/mp_unified/zdoba2.py compare-all      (the 18 compares, CPU)
    python outputs/mp_unified/zdoba2.py calls            (outputs/dob/calls.json and .md)
    python outputs/mp_unified/zdoba2.py --selftest
"""
import os
import sys

if __name__ == "__main__":                      # zdob's, before numpy or torch loads: 6 threads, the GPU never seen
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[_v] = "6"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zdob as D  # noqa: E402  (not edited)

assert D.THREADS == 6
AMENDMENT = "A2"
WITHDRAWN = {"xrank": "rank"}
KEPT = ("SEMB", "rank")                          # in every arm: rmatch requires SEMB, zret the rank block
NOTE = ("Amendment A2: xrank withdrawn before any xrank fit trained (zrckp's model z-scores every input against the "
        "retrieved rows, which the rank block's rrf column marks); seven drops, 20 fits, 18 compares.")


def tables():
    """zdob's DROPS, FITS, DROP_OF and PAIRS without the withdrawn arm."""
    gone = {D.fit_name(t, 0, sp) for t in WITHDRAWN for sp in D.M1.SPLITS}
    drops = {t: b for t, b in D.DROPS.items() if t not in WITHDRAWN}
    fits = {k: v for k, v in D.FITS.items() if k not in gone}
    drop_of = {k: v for k, v in D.DROP_OF.items() if k not in gone}
    pairs = {k: v for k, v in D.PAIRS.items() if v[0] not in gone}
    assert len(fits) == 20 and len(pairs) == 18 and len(drops) == 7, (len(fits), len(pairs), len(drops))
    assert set(drops.values()) | set(KEPT) == set(D.PICK) and not set(drops) & set(WITHDRAWN)
    return drops, fits, drop_of, pairs


@contextlib.contextmanager
def amended():
    drops, fits, drop_of, pairs = tables()
    with D.M1.WQ.patch(D, DROPS=drops, FITS=fits, DROP_OF=drop_of, PAIRS=pairs):
        yield


def stamp_calls(out):
    """The calls record and page in out name the amendment."""
    js = Path(out) / "calls.json"
    rec = json.loads(js.read_text(encoding="utf-8"))
    rec["amended_in"] = {"amendment": AMENDMENT, "withdrawn": WITHDRAWN, "kept_in_every_arm": list(KEPT),
                         "zdoba2_sha256": D.M1.WQ.sha_src(__file__)}
    D.M1.WQ.write_json(js, rec)
    md = Path(out) / "calls.md"
    md.write_text(md.read_text(encoding="utf-8") + "\n" + NOTE + "\n", encoding="utf-8")
    return rec


def compare_all(base=D.ROOT):
    with amended():
        return D.compare_all(base)


def calls(base=D.ROOT):
    with amended():
        rc = D.calls(base)
    stamp_calls(Path(base) / D.OUT)
    return rc


def selftest():
    n_fits, n_pairs, drops0 = len(D.FITS), len(D.PAIRS), dict(D.DROPS)
    assert (n_fits, n_pairs, len(drops0)) == (22, 20, 8) and "xrank" in drops0
    with amended():
        assert "xrank" not in D.DROPS and (len(D.FITS), len(D.PAIRS), len(D.DROPS)) == (20, 18, 7)
        assert not [k for k in D.FITS if "-xrank-" in k] and not [k for k in D.PAIRS if "-xrank-" in k]
        assert all(D.PAIRS[c][1] == D.fit_name(D.BASE, 0, D.FITS[D.PAIRS[c][0]][2]) for c in D.PAIRS)
        try:
            D.fit(D.fit_name("xrank", 0, "WQ-musique"))
            raise AssertionError("an xrank fit passed")
        except SystemExit:
            pass
    assert (len(D.FITS), len(D.PAIRS), D.DROPS) == (n_fits, n_pairs, drops0)
    try:
        with amended():
            raise RuntimeError("inside")
    except RuntimeError:
        pass
    assert (len(D.FITS), len(D.PAIRS), D.DROPS) == (n_fits, n_pairs, drops0)

    # the rule on made-up reads without xrank's: CALLED under the amendment, INCOMPLETE without it
    def made(base_seeds, deltas, tags):
        r = {}
        for rd in D.RULE.READS:
            d = r.setdefault(rd, {})
            for k in D.RULE.SEEDS:
                d[D.fit_name(D.BASE, k, rd[0])] = base_seeds[k]
            for t in tags:
                d[D.fit_name(t, 0, rd[0])] = base_seeds[0] + deltas.get((t, rd), 0.0)
        return r
    b = {0: 0.50, 1: 0.51, 2: 0.49}                              # spread 0.01, floor 0.02
    rd0 = D.RULE.READS[2]
    tags7 = [t for t in drops0 if t not in WITHDRAWN]
    r5 = made(b, {("xwalk", rd0): -0.03, ("xdcos", rd0): +0.03}, tags7)
    with amended():
        v = D.decide(r5)
    assert v["verdict"] == "CALLED" and set(v["arms"]) == set(tags7), v["verdict"]
    assert v["arms"]["xwalk"]["call"] == D.CARRIES and v["arms"]["xdcos"]["call"] == D.HURTS
    assert D.decide(r5)["verdict"] == D.INCOMPLETE
    # the calls record and page name the amendment
    with tempfile.TemporaryDirectory() as tmp:
        (Path(tmp) / "calls.json").write_text(json.dumps({"stage": "DOB", "verdict": "CALLED"}), encoding="utf-8")
        (Path(tmp) / "calls.md").write_text("# DOB\n", encoding="utf-8")
        rec = stamp_calls(tmp)
        assert rec["amended_in"]["withdrawn"] == WITHDRAWN and rec["verdict"] == "CALLED"
        assert (Path(tmp) / "calls.md").read_text(encoding="utf-8").endswith(NOTE + "\n")
    D.selftest()
    print("zdoba2: selftest: the tables without xrank (20 fits, 18 compares, 7 drops), their restoring (also on a "
          "failure), xrank's fit refused under the amendment, the rule CALLED on seven drops and INCOMPLETE without "
          "the amendment, the calls' stamp, and zdob's own self-test pass", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("compare-all", "calls"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "compare-all":
        return compare_all()
    if a.cmd == "calls":
        return calls()
    ap.error("a command or --selftest")


if __name__ == "__main__":
    sys.exit(main())
