"""Run a Python script with PyTorch's CUDA caching allocator bounded, so it reuses and frees its cache instead of
spilling into system memory (systems code; no science).

    python outputs/host_ops/cuda_alloc.py [--frac 1.0] [--gc 0.6] -- SCRIPT.py ARGS ...

6 Oct 2026, 19:10: the GPU memory probe (gpu_mem_probe.py) found cs30-rga-T4-s0 holding 23.4 GB of the card's 24 GB
plus 12.3 GB of shared (system) memory, 35.6 GB committed, while its own log put torch's peak allocation at 11.0 GB
(cs31-rga-kn-14-s0 next: 6.7 GB allocated, 22.8 GB on the card). On Windows (WDDM) the driver's system-memory fallback
lets cudaMalloc succeed past the card's memory, so the caching allocator never meets an out-of-memory, never frees its
cache, and its pool grows into system memory. The spill takes host memory that the job's rx request does not cover
(L3's host peak 17.8 GB against L0's 8.9) and moves blocks across PCIe. It is not why T3, T4 and L3 are slow: their
epochs are 2.5-3.5x T0's from epoch 0, before any spill (more EM rounds and layers); T4's epochs went from 113 s
(epoch 0) to 109-136 s.

This runner sets PYTORCH_CUDA_ALLOC_CONF=garbage_collection_threshold:GC (unless the caller set one) and, when CUDA
starts, torch.cuda.set_per_process_memory_fraction(FRAC). With a fraction set, the allocator frees old unused cached
blocks once its pool passes GC x FRAC of the card, and before raising out-of-memory past FRAC it frees every unused
cached block and retries; the pool then stays near the job's live use (GC is what keeps it there). The default FRAC 1.0
lets a run allocate as much as before, so no run that fit on the card can fail under the runner; what stays outside
the pool (the CUDA context, other processes) can still push a pool near the card's size a little past it. Where tensors
are placed does not change what the kernels compute: the deterministic settings (cs_dev.py, device_placement "det") are
untouched, and before any queued GPU item ran under this runner, cuda_alloc_check.py compared a smoke's rows and
record with and without it.

The script runs in this process (runpy, as __main__, with sys.argv = [SCRIPT, ARGS...]). torch is not imported here
before the script's own imports, so the thread and CUBLAS variables a script sets before importing numpy and torch take
effect as before; the fraction is queued with torch.cuda._lazy_call at the script's first import of torch.
"""
import argparse
import builtins
import os
import runpy
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=1.0, help="per-process fraction of the card's memory")
    ap.add_argument("--gc", type=float, default=0.6, help="garbage_collection_threshold, a fraction of FRAC")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd or not cmd[0].endswith(".py"):
        ap.error("need SCRIPT.py [ARGS ...] after --")
    if not (0.0 < a.frac <= 1.0 and 0.0 < a.gc < 1.0):
        ap.error("--frac in (0, 1], --gc in (0, 1)")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", f"garbage_collection_threshold:{a.gc:g}")
    frac = float(a.frac)
    orig = builtins.__import__
    done = []

    def bound():
        import torch
        torch.cuda.set_per_process_memory_fraction(frac)
        print(f"[cuda_alloc] fraction {frac:g} of {torch.cuda.get_device_name(torch.cuda.current_device())}, "
              f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF')}", flush=True)

    def hooked(name, globals=None, locals=None, fromlist=(), level=0):
        m = orig(name, globals, locals, fromlist, level)
        if not done and level == 0 and name.split(".")[0] == "torch":
            t = sys.modules.get("torch")
            tc = getattr(t, "cuda", None) if t is not None else None
            if tc is not None and hasattr(tc, "_lazy_call") and hasattr(tc, "set_per_process_memory_fraction"):
                done.append(1)
                builtins.__import__ = orig
                tc._lazy_call(bound)
        return m

    builtins.__import__ = hooked
    script = cmd[0]
    sys.argv = list(cmd)
    sys.path[0] = os.path.dirname(os.path.abspath(script))     # as `python SCRIPT.py` would have it
    runpy.run_path(script, run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
