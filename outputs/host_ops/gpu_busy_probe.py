"""Read-only probe: how busy the GPU is over a window, at sub-second resolution (systems code; no science).

    python outputs/host_ops/gpu_busy_probe.py [--seconds 180] [--ms 200] [--bin 5]

The feeder's utilization log samples nvidia-smi once a minute, and one sample cannot tell a GPU that idles a few
seconds in every epoch from one that is busy throughout. This runs `nvidia-smi --query-gpu=utilization.gpu,memory.used
-lms MS` for SECONDS and prints the mean utilization, the share of samples at 0% and at 90% or more, and the mean over
each BIN-second slice, so a CPU-only phase inside every epoch shows up as a repeating dip.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--ms", type=int, default=200)
    ap.add_argument("--bin", type=float, default=5.0)
    a = ap.parse_args(argv)
    exe = shutil.which("nvidia-smi") or os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32",
                                                     "nvidia-smi.exe")
    cmd = [exe, "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", f"-lms={a.ms}"]
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, text=True, bufsize=1)
    rows = []
    try:
        for line in p.stdout:
            t = time.time() - t0
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 2:
                try:
                    rows.append((t, float(f[0]), float(f[1]) / 1024))
                except ValueError:
                    pass
            if t >= a.seconds:
                break
    finally:
        p.kill()
    if not rows:
        print("no samples")
        return 1
    u = [r[1] for r in rows]
    n = len(u)
    print(f"{n} samples over {rows[-1][0]:.0f} s (every {a.ms} ms asked; {rows[-1][0] / max(n - 1, 1) * 1000:.0f} ms "
          f"got): mean {sum(u) / n:.1f}%, at 0% {sum(x == 0 for x in u) / n:.2f}, below 30% "
          f"{sum(x < 30 for x in u) / n:.2f}, 90% or more {sum(x >= 90 for x in u) / n:.2f}; memory "
          f"{min(r[2] for r in rows):.1f}-{max(r[2] for r in rows):.1f} GB")
    bins = {}
    for t, x, _ in rows:
        bins.setdefault(int(t // a.bin), []).append(x)
    line = []
    for k in sorted(bins):
        line.append(f"{sum(bins[k]) / len(bins[k]):3.0f}")
    for i in range(0, len(line), 24):
        print(f"{i * a.bin:6.0f}s " + " ".join(line[i:i + 24]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
