"""Do two GPU runs sharing the card compute what one run alone does? (systems check; no science; never graded)

    python outputs/host_ops/cuda_corun_check.py --out outputs/mp_unified/smoke30/corunchk/check.json [--frac 0.45]

6 Oct 2026, 20:20: a sub-second probe of the card (gpu_busy_probe.py) during cs31-rga-kp-14-s0's epochs read 50-60%
busy in every 5 s slice: one GNN run leaves about half the card idle (its own CPU-side work between kernels). Two runs
can share the card only if each one's PyTorch pool is capped so that both fit (two unbounded pools spilled into system
memory and thrashed, 4 Oct). This check gates that change, the way cuda_alloc_check.py gated the bounded pool.

It runs part 14's rga L3 GPU smoke (the smoke the queued GPU items depend on; torch peak 9.44 GB, more than the part-15
rga arms' 6.7 GB) as A: alone, under cuda_alloc.py --gc 0.9 (how the queued items run today); then C1 and C2 at the
same time, each under cuda_alloc.py --frac FRAC --gc 0.9. It compares C1 and C2 with A: every array in the rows file
and every record field except timing and placement fields.

verdict: "IDENTICAL" (A = C1 = C2), "CORUN_DIFFERS" (the runs must not share the card), or a failed run's code.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuda_alloc_check import SMOKE, TAIL, diff        # noqa: E402  (the same smoke and the same comparison)


def cmd(tag, d, frac):
    o = f"{d}/{tag}-rga-L3"          # relative and with '/': chainscore30 checks the smoke prefix as a string
    c = [sys.executable, "outputs/host_ops/cuda_alloc.py", "--frac", f"{frac:g}", "--gc", "0.9", "--"]
    c += SMOKE + ["--out", f"{o}.json", "--rows-out", f"{o}.rows.npz", "--state-out", f"{o}.pt"] + TAIL
    return o, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.45)
    a = ap.parse_args()
    rel = Path(a.out.replace("\\", "/"))
    if rel.is_absolute() or not rel.as_posix().startswith("outputs/mp_unified/smoke30/"):
        raise SystemExit("--out is a relative path under outputs/mp_unified/smoke30/")
    out = ROOT / rel
    d = rel.parent.as_posix()
    (ROOT / d).mkdir(parents=True, exist_ok=True)

    oa, ca = cmd("A", d, 1.0)
    print(f"[corun] A alone: {' '.join(ca[:8])} ...", flush=True)
    t0 = time.time()
    rc = subprocess.run(ca, cwd=ROOT).returncode
    ta = round(time.time() - t0, 1)
    print(f"[corun] A: rc {rc}, {ta} s", flush=True)
    if rc:
        raise SystemExit(f"A failed (rc {rc})")

    o1, c1 = cmd("C1", d, a.frac)
    o2, c2 = cmd("C2", d, a.frac)
    print(f"[corun] C1 and C2 together, each --frac {a.frac:g}", flush=True)
    t0 = time.time()
    p1 = subprocess.Popen(c1, cwd=ROOT)
    p2 = subprocess.Popen(c2, cwd=ROOT)
    r1, r2 = p1.wait(), p2.wait()
    tc = round(time.time() - t0, 1)
    print(f"[corun] C1 rc {r1}, C2 rc {r2}, {tc} s for both", flush=True)
    if r1 or r2:
        raise SystemExit(f"a shared run failed (C1 rc {r1}, C2 rc {r2})")

    d1, d2 = diff(oa, o1), diff(oa, o2)
    verdict = "CORUN_DIFFERS" if (d1 or d2) else "IDENTICAL"
    js = {"verdict": verdict, "seconds": {"A": ta, "C1_and_C2": tc}, "A_vs_C1": d1, "A_vs_C2": d2,
          "A_runner": "cuda_alloc.py --frac 1 --gc 0.9", "C_runner": f"cuda_alloc.py --frac {a.frac:g} --gc 0.9",
          "smoke": "cs30-smoke-rga-L3-gpu's command"}
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(js, indent=1), encoding="utf-8")
    os.replace(tmp, out)
    print(f"[corun] verdict {verdict}; A vs C1: {len(d1)} differences, A vs C2: {len(d2)}", flush=True)
    for x in (d1 + d2)[:40]:
        print("   ", x, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
