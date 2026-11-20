"""ENC E3W's report (docs/ENC_ENCODER_COMPARISON.md section 19.5): E2's report code (zenc2r.py, unchanged) on the e3v
world against the frozen world, with E2's qwen3e06 and mpnet cells beside.

    python outputs/mp_unified/zenc3r.py collect    (the host, in the record's tree: the e3v and frozen worlds' complete
                                                    D83E reads and their s1eval pool parts -> outputs/enc/e3w/collect/)
    python outputs/mp_unified/zenc3r.py files      (the rx fetch globs the report reads)
    python outputs/mp_unified/zenc3r.py report     (the laptop: outputs/enc/e3w/report.json and report.md)
    python outputs/mp_unified/zenc3r.py selftest

The rule is 12.9 and 12.11's (zenc2r's docstring): R@5 per question, e3v's world minus the frozen world, paired 95%
bootstrap (2,000, seed 0); per deciding model (GNX, zrckpn) over SQuAD and MuSiQue s1eval, KEEPS, CLOSE or LOSES. The
audited runs are E2's folders (outputs/enc/e2/audit/<enc>/<tag>, where zenc2w.run_in writes them); e3v's tags and
reruns are in docs/ENC_E3W_RUNS.json, so a rerun named there changes no pinned code. Nothing here trains or reads a
model; nothing is written outside outputs/enc/e3w/.
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zenc2r as R  # noqa: E402

ROOT = R.ROOT
ARM = "e3v"
DIM = 1152
RUNS3 = ROOT / "docs" / "ENC_E3W_RUNS.json"
E2R = ROOT / "outputs" / "enc" / "e2" / "report.json"

R.STAGE = "ENC E3W: report"
R.DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 19 (sections 12.9 to 12.11's rule)"
R.ARMS = (ARM,)
R.ENCS = (R.REF, ARM)
R.DIMS = dict(R.DIMS, **{ARM: DIM})


def load_runs(path=RUNS3):
    tags, sup = R.runs_table(path)
    R.TAGS = dict(R.TAGS, **tags)
    R.SUPERSEDED = dict(R.SUPERSEDED, **sup)


def configure(root=ROOT, worlds=None, runs=RUNS3):
    root = Path(root)
    R.configure(root, worlds)
    R.BASE["out"] = root / "outputs" / "enc" / "e3w"
    R.BASE["col"] = R.BASE["out"] / "collect"
    R.BASE["audit"] = root / "outputs" / "enc" / "e2" / "audit"
    load_runs(runs)


configure()
_cost_of = R.cost_of


def cost_of(enc):
    c = _cost_of(enc)
    if enc == ARM:
        pj = R.BASE["out"] / "params.json"
        if pj.is_file():
            p = R.read_json(pj)
            c["params"], c["params_parts"] = p["total"], p["params"]
    return c


R.cost_of = cost_of


def e2_beside():
    """E2's qwen3e06 and mpnet cells and calls (outputs/enc/e2/report.json), for the table beside."""
    if not E2R.is_file():
        return None
    e2 = R.read_json(E2R)
    return {"file": "outputs/enc/e2/report.json", "sha256": R.sha_file(E2R), "calls": e2.get("calls"),
            "cells": {k: {x: v.get(x) for x in ("frozen", "arm", "pts", "ci", "call", "won_back_pts")}
                      for k, v in e2.get("cells", {}).items()}}


def report():
    rc = R.report()
    out = R.BASE["out"]
    rep = R.read_json(out / "report.json")
    rep["beside_e2"] = e2_beside()
    R.write_json(out / "report.json", rep)
    md = (out / "report.md").read_bytes().decode("utf-8").split("\n")
    md[0] = "# ENC E3W: the models on e3v's vectors (gte-modernbert + granite-small, fused), in-domain (report)"
    b = rep["beside_e2"]
    if b:
        md += ["## Beside: E2's arms (outputs/enc/e2/report.json)", "",
               "| arm | model | carve | frozen | arm | model gap | won back |", "|---|---|---|---|---|---|---|"]
        for k, c in sorted(b["cells"].items()):
            arm, kind, ds = k.split()
            if c.get("pts") is None:
                continue
            md.append(f"| {arm} | {R.ROLE.get(kind, kind)} | {ds} | {c['frozen']} | {c['arm']} | {c['pts']:+.2f} "
                      f"[{c['ci'][0]:+.2f}, {c['ci'][1]:+.2f}] {c['call']} | {c.get('won_back_pts', 'n/a')} |")
        md += ["", "E2's calls: " + "; ".join(f"{a}: " + ", ".join(f"{k} {v['call']}" for k, v in cs.items())
                                               for a, cs in (b["calls"] or {}).items()), ""]
    (out / "report.md").write_bytes("\n".join(md).encode("utf-8"))
    return rc


def files():
    globs = ["outputs/enc/e3w/collect/**", f"outputs/enc/e2/audit/{ARM}/*/run.json",
             f"outputs/enc/e2/audit/{ARM}/*/summary.json", f"outputs/enc/e0/{ARM}/squad.*", f"outputs/enc/e0/{ARM}/musique.*",
             "outputs/enc/e3w/gv.json", "outputs/enc/e3w/params.json", "outputs/enc/e3w/mark.json",
             f"outputs/enc/e2/first/{ARM}/*.json", f"outputs/enc/e2/pad/{ARM}/*/pad.json", f"outputs/enc/e2/bases/{ARM}/*.json"]
    for g in globs:
        print(g)
    return 0


def selftest():
    assert R.ARMS == (ARM,) and R.ENCS == ("frozen", ARM) and R.DIMS[ARM] == DIM
    assert R.TAGS[ARM] and all(t.endswith("-") for t in R.TAGS[ARM]) and "frozen" in R.TAGS and "qwen3e06" in R.TAGS
    assert R.BASE["out"].parts[-2:] == ("enc", "e3w") and R.BASE["audit"].parts[-3:] == ("enc", "e2", "audit")
    assert R.BASE["worlds"][ARM].name == "e2world_e3v" and R.BASE["worlds"]["frozen"].name == "e2world_frozen"
    with tempfile.TemporaryDirectory() as td:
        ws = Path(td) / "ws"
        (ws / "docs").mkdir(parents=True)
        shutil.copy2(RUNS3, ws / "docs" / RUNS3.name)
        shutil.copy2(R.RUNS, ws / "docs" / R.RUNS.name)
        configure(ws, runs=ws / "docs" / RUNS3.name)
        assert R.report() == 0                                       # no runs and no reads: no cells, nothing called
        rep = R.read_json(ws / "outputs" / "enc" / "e3w" / "report.json")
        assert rep["cells"] == {} and rep["ga"][ARM]["verdict"] == "NO_RUNS" and rep["calls"] == {}
        assert not (ws / "outputs" / "enc" / "e2" / "report.json").exists()
    configure()
    print("zenc3r self-test: OK")
    return 0


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="zenc3r.py")
    ap.add_argument("cmd", choices=("collect", "files", "report", "selftest"))
    a = ap.parse_args(argv)
    return {"collect": R.collect, "files": files, "report": report, "selftest": selftest}[a.cmd]()


if __name__ == "__main__":
    raise SystemExit(main())
