"""SWP Part B's report (zswp.report) with the per-resample paired bootstrap in place of zabl.compare.

zabl.compare's bootstrap line averages each chunk of 100 resamples into one number, so its 2,000 resamples are 20 values and its
intervals are about 10 times too narrow (found 8 Nov 18:45; zswpcr.cmp is the corrected comparison, same seed, same resamples, each
resample the mean of its own question draw). zswp.py and zabl.py are not edited: the S-world (Part C) and the N-world (Part E) pin every
code file of the record until their reads end. This file swaps the comparison in-process and writes outputs/swp/report_pb.json and
outputs/swp/report_pb.md beside zswp's own report (whose intervals are the narrow ones and are not quoted).

    python outputs/mp_unified/zswpbr.py report
    python outputs/mp_unified/zswpbr.py selftest
"""
import contextlib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zswp as W  # noqa: E402  (Part B: cells, report and tables; not edited)
import zswpcr as CR  # noqa: E402  (Part C's report: the corrected paired bootstrap)

Z = W.Z
STAGE = "SWP-B-report-pb"
NOTE = ("95% intervals from a paired bootstrap over the questions, each of the 2,000 resamples the mean of its own draw "
        "(zswpcr.cmp; zabl.compare, used by zswp.py's own report.md, averages chunks of 100 resamples: intervals about 10x too narrow)")


def compare(a, b):
    return CR.cmp(Z, a, b)


@contextlib.contextmanager
def patched(out=None):
    """zabl.compare -> the corrected comparison; zswp's report.json/report.md -> report_pb.json/report_pb.md (out: another folder,
    for the self-test). Everything is put back on exit."""
    keep = (Z.compare, W.write_json, W.write_md, W.OUT)
    base = Path(out) if out else W.OUT

    def write_json(path, rec):
        path = Path(path)
        if path.name == "report.json":
            rec = dict(rec, bootstrap=NOTE, report_stage=STAGE, zswpbr_sha256=W.sha_src(__file__), zswpcr_sha256=W.sha_src(CR.__file__))
            path = base / "report_pb.json"
        return keep[1](path, rec)

    def write_md(rec):
        with tempfile.TemporaryDirectory() as td:
            W.OUT = Path(td)
            try:
                keep[2](rec)
                text = (Path(td) / "report.md").read_text(encoding="utf-8")
            finally:
                W.OUT = base if out else keep[3]
        head, _, rest = text.partition("\n")
        (base / "report_pb.md").write_text(head + " (per-resample bootstrap)\n\n" + NOTE + ".\n" + rest, encoding="utf-8")

    Z.compare, W.write_json, W.write_md = compare, write_json, write_md
    if out:
        W.OUT = base
    try:
        yield
    finally:
        Z.compare, W.write_json, W.write_md, W.OUT = keep


def report(out=None):
    with patched(out):
        return W.report()


def selftest():
    import numpy as np
    rng = np.random.default_rng(0)
    n = 1500
    ids, gt = np.arange(n), np.ones(n, dtype=int)
    a = {"ids": ids, "gt": gt, "per_fit": [rng.random(n) < 0.6 for _ in range(2)]}
    b = {"ids": ids, "gt": gt, "per_fit": [rng.random(n) < 0.62 for _ in range(2)]}
    a["per_fit"] = [x.astype(float) for x in a["per_fit"]]
    b["per_fit"] = [x.astype(float) for x in b["per_fit"]]
    old, new = Z.compare(a, b), compare(a, b)
    assert abs(old["diff"] - new["diff"]) < 1e-12 and old["questions"] == new["questions"] == n
    wo, wn = old["ci"][1] - old["ci"][0], new["ci"][1] - new["ci"][0]
    assert wn > 5 * wo, (wo, wn)                                   # zabl.compare's interval is about 10x too narrow
    keep = (Z.compare, W.write_json, W.write_md, W.OUT)
    with tempfile.TemporaryDirectory() as td:
        assert report(td) == 0
        rec = W.read_json(Path(td) / "report_pb.json")
        md = (Path(td) / "report_pb.md").read_text(encoding="utf-8")
        assert rec["bootstrap"] == NOTE and rec["report_stage"] == STAGE and len(rec["missing"]) > 0
        assert md.startswith("# SWP part B") and "(per-resample bootstrap)" in md.splitlines()[0] and NOTE in md
        assert not (Path(td) / "report.json").exists() and not (Path(td) / "report.md").exists()
        try:
            with patched(td):
                assert Z.compare is compare and W.OUT == Path(td)
                raise RuntimeError("an error inside the patch")
        except RuntimeError:
            pass
    assert (Z.compare, W.write_json, W.write_md, W.OUT) == keep, "patched() must put everything back"
    print(f"zswpbr self-test: OK (synthetic diff {100 * new['diff']:+.2f}: zabl.compare width {100 * wo:.2f}, per-resample {100 * wn:.2f})")
    return 0


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    if a[:1] == ["selftest"]:
        return selftest()
    if a[:1] == ["report"]:
        return report()
    raise SystemExit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
