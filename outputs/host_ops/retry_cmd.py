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
--frac F, F < 1: the host feeder caps two runs that share the card) is run again with --frac 1: the same run with the
whole card's bound, which cannot hit the same cap again (6 Oct, 20:55). Nothing else in the command changes.
"""
import argparse
import subprocess
import sys
import time

RACE = ("[WinError 5]", "[WinError 32]")
CUDA = ("CUDA error", "AcceleratorError", "CUBLAS_STATUS_", "CUDNN_STATUS_", "CUDA out of memory")


def uncap(cmd):
    """cmd with cuda_alloc.py's --frac F (the runner's own options, before its --) set to 1."""
    out = list(cmd)
    for i, t in enumerate(out):
        if t.replace("\\", "/").endswith("cuda_alloc.py"):
            j = i + 1
            while j < len(out) - 1 and out[j] != "--":
                if out[j] == "--frac":
                    out[j + 1] = "1"
                j += 1
            break
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--wait", type=float, default=30.0)
    ap.add_argument("--cuda", action="store_true", help="also rerun on a CUDA fault")
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
        if a.cuda and b"CUDA out of memory" in tail and uncap(cmd) != cmd:
            cmd = uncap(cmd)
            print("[retry_cmd] out of memory under a capped pool: the next run has the whole card's bound (--frac 1)",
                  flush=True)
        time.sleep(a.wait)
    return rc


if __name__ == "__main__":
    sys.exit(main())
