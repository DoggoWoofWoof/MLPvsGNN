"""Self-test of cuda_alloc.py's peak report (systems check; no science).

    python outputs/host_ops/cuda_alloc.py --frac 0.1 -- outputs/host_ops/cuda_alloc_selftest.py

Allocates 1 GiB and frees it, resets the peak statistics as a training script does each epoch, allocates 0.5 GiB
and exits. torch then reports 0.50 GB; the runner must print "peak allocated 1.00 GB" (the reset folded in).
"""
import torch

x = torch.empty(2 ** 28, dtype=torch.float32, device="cuda")        # 1 GiB
torch.cuda.synchronize()
del x
torch.cuda.reset_peak_memory_stats()                                # torch's peak restarts from 0 allocated
y = torch.empty(2 ** 27, dtype=torch.float32, device="cuda")        # 0.5 GiB
torch.cuda.synchronize()
print(f"[selftest] after the reset torch reports {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)
