"""Self-test of cuda_alloc.py's peak report (systems check; no science).

    python outputs/host_ops/cuda_alloc.py --frac 0.1 -- outputs/host_ops/cuda_alloc_selftest.py

Allocates 1 GiB, resets the peak statistics as a training script does each epoch, allocates 0.5 GiB and exits. The
runner must print "peak allocated 1.00 GB" (the reset folded in), not 0.50.
"""
import torch

x = torch.empty(2 ** 28, dtype=torch.float32, device="cuda")        # 1 GiB
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
del x
y = torch.empty(2 ** 27, dtype=torch.float32, device="cuda")        # 0.5 GiB
torch.cuda.synchronize()
print(f"[selftest] after the reset torch reports {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)
