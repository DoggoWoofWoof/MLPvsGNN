"""Does cuda_alloc.py change what a GPU run computes? (systems check; no science; its outputs are never graded)

    python outputs/host_ops/cuda_alloc_check.py --out outputs/mp_unified/smoke30/capchk/check.json

Runs part 14's rga L3 GPU smoke (cs30-smoke-rga-L3-gpu's command, outputs under smoke30/capchk/) three times, one
after another on the same card: A and A2 as the queued items run today, B under cuda_alloc.py with --gc 0.05, so the
allocator frees cached blocks on nearly every miss (the path a bounded pool takes, exercised far more often than the
queued items' --gc 0.6 would). Then it compares A with A2 (is the smoke itself repeatable on this card?) and A with B:
every array in the rows file, the record's state_sha256 and every record field except timing and placement fields.

verdict: "IDENTICAL" (A = A2 = B), "CAP_DIFFERS" (A = A2, B differs: the runner must not be used),
"NOT_REPEATABLE" (A differs from A2: identity cannot decide; the fields that differ are listed).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
B = "outputs/mp_unified/lean/"
SMOKE = ["outputs/mp_unified/chainscore30.py", "train", "--cls", "rga", "--axis", "L", "--value", "3", "--threads", "1",
         "--smoke", "--", "train", "--arm", "rga", "--seed", "0", "--device", "cuda",
         "--map-from", B + "cs22.json", "--cache", "outputs/mp_unified/cache", "--fit", B + "cs19-mq-fit-id.npz",
         "--aug", "al4=" + B + "cs19-mq-fit-al4.npz", "--aug", "sp4=" + B + "cs19-mq-fit-sp4.npz",
         "--aug", "mg3=" + B + "cs19-mq-fit-mg3.npz", "--aug", "rf=" + B + "cs19-mq-fit-rf.npz",
         "--select", B + "cs19-mq-select.npz", "--read", "metaqa=" + B + "cs19-mq-x1f.npz",
         "--read", "webqsp=" + B + "cs19-wq.npz", "--read", "webqsp_sf=" + B + "cs19-wq-sf.npz",
         "--pread", "2wiki=" + B + "cs21-2w-x1.npz", "--pread", "hotpotqa=" + B + "cs21-hp-x1.npz",
         "--edges", "fit=" + B + "cs20e-mq-fit-id.npz", "--edges", "al4=" + B + "cs20e-mq-fit-al4.npz",
         "--edges", "sp4=" + B + "cs20e-mq-fit-sp4.npz", "--edges", "mg3=" + B + "cs20e-mq-fit-mg3.npz",
         "--edges", "rf=" + B + "cs20e-mq-fit-rf.npz", "--edges", "select=" + B + "cs20e-mq-select.npz",
         "--edges", "metaqa=" + B + "cs20e-mq-x1f.npz", "--edges", "webqsp=" + B + "cs20e-wq.npz",
         "--edges", "webqsp_sf=" + B + "cs20e-wq-sf.npz", "--edges", "2wiki=" + B + "cs21e-2w-x1.npz",
         "--edges", "hotpotqa=" + B + "cs21e-hp-x1.npz"]
TAIL = ["--epochs", "1", "--per-epoch", "128", "--smoke", "--read-limit", "64"]
SKIP_SUB = ("time", "seconds", "timing", "wall", "placement")       # timing and placement fields
SKIP_KEY = {"started", "ended", "host", "argv", "out", "rows_out", "state_out", "inputs"}   # paths and clocks


def run(tag, d, capped):
    o = f"{d}/{tag}-rga-L3"          # relative and with '/': chainscore30 checks the smoke prefix as a string
    cmd = [sys.executable] + (["outputs/host_ops/cuda_alloc.py", "--frac", "0.95", "--gc", "0.05", "--"] if capped else [])
    cmd += SMOKE + ["--out", f"{o}.json", "--rows-out", f"{o}.rows.npz", "--state-out", f"{o}.pt"] + TAIL
    print(f"[check] {tag}: {' '.join(cmd[:6])} ...", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    print(f"[check] {tag}: rc {rc}, {time.time() - t0:.0f} s", flush=True)
    if rc:
        raise SystemExit(f"{tag} failed (rc {rc})")
    return o, round(time.time() - t0, 1)


def flat(x, pre=""):
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            if str(k) in SKIP_KEY or any(s in str(k).lower() for s in SKIP_SUB):
                continue
            out.update(flat(v, f"{pre}{k}."))
        return out
    if isinstance(x, list):
        out = {}
        for i, v in enumerate(x):
            out.update(flat(v, f"{pre}{i}."))
        return out
    return {pre.rstrip("."): x}


def diff(o1, o2):
    bad = []
    j1 = flat(json.loads((ROOT / f"{o1}.json").read_text(encoding="utf-8")))
    j2 = flat(json.loads((ROOT / f"{o2}.json").read_text(encoding="utf-8")))
    for k in sorted(set(j1) | set(j2)):
        if j1.get(k, "<missing>") != j2.get(k, "<missing>"):
            bad.append(f"json {k}: {j1.get(k, '<missing>')!r} vs {j2.get(k, '<missing>')!r}"[:300])
    with np.load(ROOT / f"{o1}.rows.npz", allow_pickle=False) as r1, \
            np.load(ROOT / f"{o2}.rows.npz", allow_pickle=False) as r2:
        for k in sorted(set(r1.files) | set(r2.files)):
            if k not in r1.files or k not in r2.files:
                bad.append(f"rows {k}: present in one only")
                continue
            x, y = r1[k], r2[k]
            if x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes():
                bad.append(f"rows {k}: {x.dtype}{x.shape} vs {y.dtype}{y.shape}, not byte-equal")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rel = Path(a.out.replace("\\", "/"))
    if rel.is_absolute() or not rel.as_posix().startswith("outputs/mp_unified/smoke30/"):
        raise SystemExit("--out is a relative path under outputs/mp_unified/smoke30/")
    out = ROOT / rel
    d = rel.parent.as_posix()
    (ROOT / d).mkdir(parents=True, exist_ok=True)
    oa, ta = run("A", d, False)
    ob, tb = run("B", d, True)
    oa2, ta2 = run("A2", d, False)
    rep, cap = diff(oa, oa2), diff(oa, ob)
    verdict = "NOT_REPEATABLE" if rep else ("CAP_DIFFERS" if cap else "IDENTICAL")
    js = {"verdict": verdict, "seconds": {"A": ta, "B": tb, "A2": ta2}, "A_vs_A2": rep, "A_vs_B": cap,
          "B_runner": "cuda_alloc.py --frac 0.95 --gc 0.05", "smoke": "cs30-smoke-rga-L3-gpu's command"}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(js, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"[check] verdict {verdict}; A vs A2: {len(rep)} differences, A vs B: {len(cap)}", flush=True)
    for x in (rep + cap)[:40]:
        print("   ", x, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
