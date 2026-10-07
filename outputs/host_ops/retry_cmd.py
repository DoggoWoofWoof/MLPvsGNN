"""Run a command; rerun it when it failed on a transient Windows file race (systems code; no science).

    python outputs/host_ops/retry_cmd.py [--tries 4] [--wait 30] -- COMMAND ...

A rename or delete on the host can fail for a moment while another process (the virus scanner, the indexer, a sync
read) holds a file open: "PermissionError: [WinError 5] Access is denied" on os.replace of a directory just written
(cs25c-wq-sf, 5 Oct 00:59), or "[WinError 32]" (in use). The command's output passes through unchanged. When the
command exits nonzero and the last 4,000 characters of its output name WinError 5 or 32, it is run again after --wait
seconds, up to --tries runs in all; any other failure ends at once with the command's code. Only commands that are safe
to rerun belong here (cs_cache.py build keeps a current entry and rebuilds a missing one).

--cuda also reruns a command whose output ends on a CUDA fault ("CUDA error", torch.AcceleratorError, a CUBLAS or
CUDNN status, "CUDA out of memory"). cs30-sg2g-dirfwd-s1 (5 Oct 11:50) died in epoch 7 on "CUDA error: an illegal
memory access was encountered" while seeds 0 and 2 of the same arm passed, and in the feeder a failed run drops every
grade that reads it. A seeded training run started again from scratch is the same run, so the GPU items run under
--cuda; a fault that repeats on every try still ends with the command's code.

Under --cuda, a run that failed on "CUDA out of memory" while running under a capped PyTorch pool (cuda_alloc.py
--frac F, F < 1: the host feeder caps the runs that share the card) is run again under a larger cap: what torch's
message says the failed run asked for (allocated + reserved but unallocated + the request, over the card's capacity)
plus --oom-margin, and at least F + --oom-step, at most 1. Nothing else in the command changes. Until 7 Oct 08:55 the
rerun had --frac 1 (6 Oct 20:55): J5 (cap 0.28, asked 6.84 GiB = 0.285) then held 11.5 GiB, the caching allocator
keeping what it freed, and with two capped fits beside it the card spilled into system memory for 35 min; L-metaqa's
uncapped rerun held 6.39 GiB where 3.92 was allocated. A larger cap keeps the pool bounded, and the feeder counts what a
run holds beyond its GPU share as taken.
"""
import argparse
import math
import re
import subprocess
import sys
import time

RACE = ("[WinError 5]", "[WinError 32]")
CUDA = ("CUDA error", "AcceleratorError", "CUBLAS_STATUS_", "CUDNN_STATUS_", "CUDA out of memory")


def frac_of(cmd):
    """cuda_alloc.py's --frac F in cmd (the runner's own options, before its --), or None."""
    for i, t in enumerate(cmd):
        if t.replace("\\", "/").endswith("cuda_alloc.py"):
            j = i + 1
            while j < len(cmd) - 1 and cmd[j] != "--":
                if cmd[j] == "--frac":
                    return float(cmd[j + 1])
                j += 1
            return None
    return None


def with_frac(cmd, frac):
    """cmd with cuda_alloc.py's --frac set to frac."""
    out = list(cmd)
    for i, t in enumerate(out):
        if t.replace("\\", "/").endswith("cuda_alloc.py"):
            j = i + 1
            while j < len(out) - 1 and out[j] != "--":
                if out[j] == "--frac":
                    out[j + 1] = f"{frac:g}"
                j += 1
            break
    return out


def _gib(num, unit):
    return float(num) / (1024.0 if unit == "MiB" else 1.0)


def asked_frac(tail):
    """What torch's out-of-memory message says the run asked for, as a fraction of the card: (allocated + reserved but
    unallocated + the request) / total capacity; None when the message does not say."""
    t = tail.decode("utf-8", "replace")
    req = re.findall(r"Tried to allocate ([\d.]+) (MiB|GiB)", t)
    alloc = re.findall(r"([\d.]+) (MiB|GiB) is allocated by PyTorch", t)
    resv = re.findall(r"([\d.]+) (MiB|GiB) is reserved by PyTorch but unallocated", t)
    cap = re.findall(r"total capacity of ([\d.]+) (MiB|GiB)", t)
    if not (req and alloc and cap):
        return None
    total = _gib(*cap[-1])
    need = _gib(*req[-1]) + _gib(*alloc[-1]) + (_gib(*resv[-1]) if resv else 0.0)
    return need / total if total > 0 else None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--wait", type=float, default=30.0)
    ap.add_argument("--cuda", action="store_true", help="also rerun on a CUDA fault")
    ap.add_argument("--oom-step", type=float, default=0.05,
                    help="an out-of-memory rerun's cap is at least the failed cap plus this")
    ap.add_argument("--oom-margin", type=float, default=0.02,
                    help="added to what the failed run asked for (fraction of the card)")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("no command")
    rc = 1
    for k in range(1, a.tries + 1):
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tail = b""
        for chunk in iter(lambda: p.stdout.read1(1 << 14), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            tail = (tail + chunk)[-4000:]
        rc = p.wait()
        if rc == 0:
            return 0
        race = [r for r in RACE + (CUDA if a.cuda else ()) if r.encode() in tail]
        if not race or k == a.tries:
            if race:
                print(f"[retry_cmd] run {k} of {a.tries} failed on {race}; no runs left (rc {rc})", flush=True)
            return rc
        print(f"[retry_cmd] run {k} of {a.tries} failed on {race} (rc {rc}); again in {a.wait:g} s", flush=True)
        f = frac_of(cmd)
        if a.cuda and b"CUDA out of memory" in tail and f is not None and f < 1:
            asked = asked_frac(tail)
            nf = max(f + a.oom_step, (asked or 0.0) + a.oom_margin)
            nf = min(1.0, math.ceil(nf * 100 - 1e-9) / 100)
            cmd = with_frac(cmd, nf)
            print(f"[retry_cmd] out of memory under a capped pool (--frac {f:g}"
                  + (f", asked {asked:.3f} of the card" if asked is not None else "")
                  + f"): the next run has --frac {nf:g}", flush=True)
        time.sleep(a.wait)
    return rc


if __name__ == "__main__":
    sys.exit(main())
