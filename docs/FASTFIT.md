# fastfit: lean_gpu's fits with X in host memory, checked bit for bit (3 November)

`outputs/host_ops/fastfit.py` runs a command with two changes to lean_gpu, both made in-process when lean_gpu is
imported. lean_gpu itself is not edited.

1. **Host X.** CacheCarve's fixed blocks (the float16 matrix X) stay in host memory, and everything else goes to the
   device as before. `batch()` copies each batch's rows of X to the device. A carve class whose own code reads `self.X`
   keeps X on the device. zg6r's carve takes its batch from CacheCarve's, so its X moves too.
2. **Prefetch.** fit_variant's loop builds the next step's batch on a thread. The steps, their order, their random draws
   and the step's code are unchanged.

fastfit checks the sha of each function it replaces and refuses to patch on a mismatch.

## The check (`outputs/host_ops/fastfit_check.py`, outputs/fastfit_check/check.json)

The check is one small real fit on the host's card, run three ways in their own processes under `cuda_alloc --frac 0.15`:
lean_gpu's train on PKO's cache, squad/fit and webqsp/fit, two epochs, seed 0. It runs under a scratch name and root,
and no stage's fit is read or written.

- **plain:** no fastfit.
- **host:** `fastfit.py --no-prefetch`.
- **both:** `fastfit.py`.

Each run enters lean_gpu as an imported module, as the fits do (`--lg-main`). A run counts only when its log shows the
changes its mode makes. The first check, at 19:23 IST, ran lean_gpu as `__main__`, so fastfit never patched it, and it
is not a result.

| run | seconds fitting | peak on the card | vs plain |
|---|---|---|---|
| plain | 79.8 | 2.89 GB | |
| host | 80.8 | 1.80 GB | IDENTICAL |
| both | 79.6 | 1.83 GB | IDENTICAL |

"IDENTICAL" means every state dict, compared tensor by tensor at the bit (each epoch's and the swa average), plus the
fit's metadata and its loss-and-steps curve, equals plain's.

## Reading

- fastfit changes no number.
- On this fit it saves memory (the peak is 38% lower, because webqsp/fit's 1.00 GB of X moved to the host), not time.
  Prefetch waited 0 s on a batch, so the steps were bound by the device, not by building batches.
- The queue's remaining GPU fits are not re-booked. Their WQ-hotpotqa fits take 0.83 of the card one at a time, and the
  saving per fit, about X's size, would not let two of them share the card.
- fastfit stays available as a bit-identical memory saver, for when the card is shared by fits that X alone keeps
  apart.
