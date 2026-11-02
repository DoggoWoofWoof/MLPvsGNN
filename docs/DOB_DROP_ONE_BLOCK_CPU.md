# DOB: zrckp with one input block dropped, on PKO's pools, on the host's CPU

Declared 2 November 2026, before any DOB fit. Stage module: `outputs/mp_unified/zdob.py`. It imports `zml1s.py` and does not
edit it.

## 1. The question

- **What PFS named.** PFS (docs/PKO_FEATURE_SCREEN.md) found that PKO's cached columns separate the golds the first stage
  misses on every dataset (NO_CARRIER on none). It named the next screen: a block ablation (drop-one) on the carrying blocks.
  Adding information did not help: the edge scalars (PFS-B) and the similarity to the first stage's top rows (A2) added
  nothing.
- **Where the limit is.** The knee (PKO_DEPTH_KNEE section 7) and C1R (section 5) both put it in the ranker. On musique, the
  deeper pool's reach is never ranked into the top five, and the MLP keeps 0.84 to 0.89 of zg6r's gain there.
- **What the user asked, 31 October.** "use the cpu to find the best possible features to give with these new pools", and "if
  we can run some mlps in cpu then do it". The host's CPU is idle while ISO holds its GPU.
- **So the question is:** on PKO's pools, does dropping any one of zrckp's input blocks raise R@5, which would mean the block
  costs ranking? And which blocks does R@5 fall without, which would mean they carry it?

## 2. What runs

- **The arms.**
  - zrckp, ML1's MLP of record (zrm's model with the KT pointwise head), with its whole pick.
  - Eight arms, each with one block left out of the pick: rank, dense_cos, topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS,
    WALKF.
  - SEMB stays in every arm, because rmatch requires it. The KT input stays too.
  - Everything else is ML1's zrckp: the trainer, the card, PKO's roots and typed links, and the read cap (65,536).
- **The device: the host's CPU, 6 threads per fit.** The GPU is hidden from the process. torch, OpenMP, MKL and OpenBLAS all
  run 6 threads.
  - A new device needs its own bases and null (13 October rule), so the base is fitted again on the CPU at three seeds.
  - No GPU fit is compared with a CPU fit.
- **The fits, 22.**
  - The base at seeds 0, 1 and 2 on both WQ splits (6): `scr-dob-zrckp-s<k>`, with `-hp` on WQ-hotpotqa.
  - Each drop at seed 0 on both splits (16): `scr-dob-<tag>-s0`, with `-hp`. The tags are xrank, xdcos, xtopo, xdepth,
    xseed, xwalk, xdists and xwalkf.
  - Each fit is stamped under the key `dob` with its dropped block, its pick and its threads.
- **The compares, 20,** through zwq's `run_compare`, as ML1's:
  - each drop's seed-0 fit against the base's seed 0 (16);
  - the base's seeds 1 and 2 against its seed 0 (4, the null).

## 3. The reads

As ML1 section 3: twelve reads, two splits by six datasets, p@swa on the s1eval carves.

## 4. The rule (fixed now)

**Per read.**

- The floor is max(0.0075, 2 x the CPU base's spread). The spread is the root mean square of the base's seed-1 and seed-2
  differences from its seed 0 (`zml1.spread`, then `zml1.pair_floor` with the base's spread twice).
- A drop's delta is its seed-0 R@5 minus the base's seed-0 R@5.
- **LOSS:** the delta is at or below minus the floor. **GAIN:** at or above the floor. **WITHIN:** otherwise.

**Per drop.**

- **HURTS:** at least one GAIN and no LOSS. The block costs R@5 on PKO's pools.
- **CARRIES:** at least one LOSS and no GAIN. The block carries R@5.
- **MIXED:** both GAINs and LOSSes. It is reported and decides nothing, because a choice that depends on the dataset is not
  allowed.
- **IDLE:** every read is WITHIN.

**INCOMPLETE** when any of these holds:

- a fit or a compare is missing;
- a fit's train or read did not run on the CPU at 6 threads;
- a fit's stamp does not name its dropped block and its pick;
- a fit's code differs from its split's base's;
- the base's seed 0 gives two different R@5 values in two compares.

A drop is adopted only on R@5, by ML1's rule. It is never adopted for speed.

## 5. Reported beside, deciding nothing

- Each drop's mean delta over the twelve reads.
- The CPU base's seed means beside ML1's zrckp on the GPU.
- Seconds per fit (`ml1_timing.json` in each fit's folder).

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| HURTS on a block | Its full run, in an amendment: the drop at seeds 1 and 2 on the CPU, judged by ML1's rule (`zml1.verdict`) against the CPU base. On ADOPT, the drop goes to zg6r first, on the GPU, in its own declared file. The program (section 12) applies a screen that passes on the CPU to zg6r first and to the MLP second. |
| CARRIES | The block stays. The list says which inputs carry the ranking on PKO's pools, per dataset. It goes into the paper's input table and into the GNN's depth form. |
| no HURTS anywhere | The pick stays. The knee says the ranker is the limit, so the next CPU form is zrckp's capacity (hidden width and epochs), in its own file. |

A verdict closes only the form it tested.

## 7. Risks, stated before any number

- **Seed 0 only for the drops.** This is a one-fit screen. The full run decides any adoption.
- **The CPU is not the GPU.** Reductions run in another order. That is why the base is fitted again here and the null is the
  CPU's own.
- **Time.** ML1's zrckp fits took 16 to 41 minutes to train and 11 to 22 minutes to read on the GPU, and the CPU is slower. The
  first base fits measure it. The items' memory follows their measured peaks.
- **Blocks interact.** Dropping one block at a time says nothing about pairs. For example, WALK and WALKF read the same walk.

## 8. How it runs

Feeder items run on the host's CPU, 6 CPUs each, so four run at a time. They do not touch the GPU.

1. The base at seed 0 on both splits, then seeds 1 and 2.
2. The sixteen drops.
3. `dob-compare`, after all 22 fits.
4. `dob-calls`, after the compare.

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zdob.py fit --name NAME
python outputs/mp_unified/zdob.py compare-all
python outputs/mp_unified/zdob.py calls          (outputs/dob/calls.json and .md)
```

The self-test, `python outputs/mp_unified/zdob.py --selftest`, passed before this commit. It covers:

- the names and the pairs;
- the card's argv;
- that the real modules hold the names the patches replace (lean_gpu's pick, zp1c2's card);
- the patches, and that they are put back, also on a failure;
- the guard against a changed pick;
- the rule on made-up reads: IDLE, CARRIES, HURTS, MIXED, the 0.0075 floor, and INCOMPLETE on a hole.

## 9. Amendment A1, 2 November 2026, before any fit trained: the CPU environment's torch

The first three items (`dob-zrckp-s0`, `-s0-hp`, `-s1`) stopped at zml1s's device check, about a second in, before any
training. The host runs CPU items in its environment for CPU work, whose torch is `2.8.0+cpu`. The module had pinned the
GPU environment's build (`2.8.0+cu128`). The device record now pins `2.8.0+cpu`; the rest of the check is unchanged: the CPU,
6 threads, and the GPU not visible. Every DOB fit runs in that one environment, so the base and the drops share it. Nothing
else changes. The feeder sends each stopped item once more on its own.
