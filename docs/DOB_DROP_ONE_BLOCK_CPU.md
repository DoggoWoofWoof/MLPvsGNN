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

## 10. Amendment A2, 3 November 2026, before any xrank fit trained: the xrank arm withdrawn

- **What stopped.** `dob-xrank-s0` stopped when its model was built, before any training: "zret needs the rank block: its
  rrf column marks the retrieved rows" (lean_screen3.py). zrckp's model z-scores every input against the pool's retrieved
  rows, and the rank block's rrf column is what marks them. Without the block it is another model, not zrckp with one
  input left out.
- **The change.** The xrank arm is withdrawn. The rank block stays in every arm, as SEMB does. Seven drops remain:
  dense_cos, topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS and WALKF. The fits are 20 (the base's six and the drops'
  fourteen) and the compares 18 (the drops' fourteen and the null's four). Section 4's rule is unchanged over the seven
  drops, and its "a fit or a compare is missing" counts these. DOB does not measure the rank block's share of R@5.
- **The code.** zdob.py is not edited: its sha is in every DOB fit's code record, and section 4 checks each fit's code
  against its base's. `outputs/mp_unified/zdoba2.py` imports it and, for the length of the compares and the calls, takes
  xrank's two fits and two compares out of zdob's tables, then puts them back, also on a failure. The calls record and
  page name the amendment. Its self-test passed before this commit; it covers the tables, their restoring, the refusal of
  an xrank fit, the rule called on seven drops (INCOMPLETE without the amendment), the calls' stamp, and zdob's own
  self-test.
- **The queue.** `dob-xrank-s0` and `dob-xrank-s0-hp` leave it. `dob-compare` and `dob-calls` run
  `zdoba2.py compare-all` and `zdoba2.py calls`, after `dob-a2-selftest`.
- **What has been read.** Nothing: DOB's compares and calls have not run.

## 11. Results, 4 November 2026 (00:22 IST)

`zdoba2.py compare-all` and `calls` ran on the host after A2's self-test, on 4 November at 00:21 and 00:22 IST. The
records are `outputs/dob/calls.json` and `calls.md`. The verdict is CALLED, with no problems.

**Under DOB's rule (section 4), on the twelve reads:**

| drop | block | call | GAIN reads | LOSS reads | mean delta |
| --- | --- | --- | --- | --- | ---: |
| xdcos | dense_cos | MIXED | WQ-musique musique, WQ-hotpotqa metaqa | WQ-musique hotpotqa, WQ-musique 2wiki | +0.0082 |
| xtopo | topo_STRUCT | HURTS | WQ-musique webqsp | | +0.0092 |
| xdepth | depth_STRUCT | HURTS | WQ-hotpotqa metaqa | | +0.0111 |
| xseed | SEED | HURTS | WQ-hotpotqa metaqa | | +0.0056 |
| xwalk | WALK | HURTS | WQ-musique webqsp | | +0.0049 |
| xdists | DISTS | IDLE | | | +0.0022 |
| xwalkf | WALKF | HURTS | WQ-musique musique | | +0.0019 |

- Each HURTS stands on one GAIN. Every one of those GAINs is on a read outside the WWW rule's six: metaqa, webqsp, or
  musique where the WQ-musique split holds it out.
- The largest deltas are on WQ-musique metaqa, where the CPU base's own seeds run from 0.341 to 0.386 (floor 0.0743).

**Under the WWW rule** (docs/WWW_SCOPE_RULE.md section 2), which decides the queue until the freeze. Section 12 says how
DOB's per-drop calls are read; the records are `outputs/dob/www_screen.json` and `.md`.

| drop | WWW screen | the deciding reads |
| --- | --- | --- |
| xdcos | NOT_ADOPTED | LOSS on WQ-musique hotpotqa (−0.0105) and 2wiki (−0.0080), floor 0.0075 |
| xtopo | ADOPT_FOR_FULL_RUN | GAIN on WQ-hotpotqa musique-3hop (+0.0154, floor 0.0075) |
| xdepth, xseed, xwalk, xdists, xwalkf | NO_GAIN | all six WITHIN |

**What the screen says.**
- dense_cos carries the in-domain reads of the WQ-musique split, hotpotqa and 2wiki. It is the only block whose drop
  costs a WWW read.
- Dropping any other single block leaves all six WWW reads WITHIN, except xtopo's 3-hop slice. On PKO's pools no other
  block is needed on its own. One drop at a time says nothing about pairs (section 7): WALK and WALKF read the same walk.

**Beside, deciding nothing.**
- The CPU base's seed means are within 0.004 of ML1's zrckp on the GPU on the four WWW dataset reads. The largest gap
  over the twelve is WQ-musique metaqa, 0.3645 against 0.4008.
- The 20 fits took 1.4 to 2.5 hours each on 6 threads, with measured peaks of 26.6 to 30.9 GB (the feeder's records).

**What follows.**
- xtopo's full run: section 12.
- The other four HURTS drops (xdepth, xseed, xwalk, xwalkf) owe full runs under DOB's own rule. They are NO_GAIN under
  the WWW rule, so their full runs wait until after the freeze, as DEP's do (docs/WWW_SCOPE_RULE.md section 3).
- dense_cos stays in the pick. It is recorded as the block that carries the WQ-musique split's in-domain reads.

## 12. Amendment A3, 4 November 2026 (about 00:50 IST), before any xtopo seed-1 or seed-2 fit

**Why.** DOB's calls hold one verdict per drop. zwww reads one screen block per stage, so its record for DOB
(`outputs/www/dob.json`) has no screen. The WWW rule decides the queue until the freeze, so each drop is read on its
six reads here. Module: `outputs/mp_unified/zdoba3.py`. It imports zdob, zdoba2, zdep and zwww and edits none of them;
A2's tables stay in force.

**The WWW reading of the screen** (`zdoba3.py www-screen`):
- The four dataset reads are DOB's calls rows, unchanged: the same base, floor, delta and call.
- musique's 3-hop and 4-hop slices on the WQ-hotpotqa split come from the fits' musique read files, through
  `zdep.slice_r5`, as DEP, LTM and MPP read them. Each file's all-question R@5 must equal DOB's compare within
  `zdep.R5_TOL`. The floor and the call are section 4's, from the CPU base's three seeds on the slice.
- Each drop's verdict is `zwww.screen_verdict` on the six rows.
- The slices were computed after DOB's calls were read, by a method fixed before them (the WWW rule and zdep). Nothing
  in the reading is chosen here.

**xtopo's full run** (section 6, "HURTS on a block", and the WWW screen's ADOPT_FOR_FULL_RUN):
- **The fits.** xtopo at seeds 1 and 2 on both splits: `scr-dob-xtopo-s1`, `-s2`, `-s1-hp` and `-s2-hp`. They run
  through zdob's fit, on the host's CPU at 6 threads, with the four names added to A2's tables for the length of the
  call.
- **The compares.** Each fit against the base's fit at the same seed (`scr-dob-xtopo-vs-zrckp-s<k>`), through zdob's
  compare-all. A2's 18 compares are kept.
- **The calls** (`outputs/dob/full_xtopo.json` and `.md`):
  - own: ML1's rule (`zml1.verdict`), xtopo's three seeds against the CPU base's three on the twelve reads;
  - WWW: `zwww.full_verdict` on the six reads, each called by `zml1.call` at three seeds with `zml1.pair_floor`'s
    floor. ADOPT needs no LOSS among the six and a GAIN on one of the four dataset reads, so the 3-hop slice alone
    cannot adopt it;
  - INCOMPLETE on any of section 4's problems (`zdob.fit_problems` over all 24 fits) or on a read file that does not
    give its compare's R@5.

**What follows (fixed now).**
- WWW ADOPT: topo_STRUCT leaves the pick, in its own declared file: zg6r first, on the GPU (section 6), then zrckpn, the
  MLP of record since LTM.
- WWW NOT_ADOPTED: topo_STRUCT stays. With no drop adopted, the next CPU form is section 6's capacity form (hidden width
  and epochs), on zrckpn, in its own file.
- The own verdict is filed beside it. It decides after the freeze, together with the other four drops' full runs.

**The queue.**
- `dob-a3-selftest` runs first, then the four fits: 6 CPUs each, with memory from their measured peaks (28 GB, and
  32 GB on WQ-hotpotqa). `dob-a3-compare` and `dob-a3-calls` follow.
- The block sits after every pending GPU fit in the feeder's file, so a fit that waits for memory holds no GPU item.
- U2w's stage B items, when declared, go ahead of it (the user's 4 November priority).

**The self-test**, `python outputs/mp_unified/zdoba3.py --selftest`, passed on the laptop's CPU before this commit. It
covers:
- the tables (24 fits, 22 compares) and their restoring, also on a failure, and the refusal of every other fit;
- the WWW screen per drop on made-up calls and read files: a dataset LOSS, a slice LOSS, a slice GAIN alone, and a read
  file off its compare;
- the full run's two rules: a slice GAIN alone, a slice LOSS, webqsp beside, and a slice missing;
- A2's and zdob's own self-tests.

```
python outputs/mp_unified/zdoba3.py www-screen
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zdoba3.py fit --name scr-dob-xtopo-s1
python outputs/mp_unified/zdoba3.py compare-all
python outputs/mp_unified/zdoba3.py calls
```

## 13. Results of A3, 4 November 2026 (07:58 IST)

`zdoba3.py compare-all` and `calls` ran on the host on 4 November at 07:58 IST, after the four xtopo fits. The records
are `outputs/dob/full_xtopo.json` and `.md`, with no problems.

| rule | reads | verdict |
| --- | --- | --- |
| own (ML1's, `zml1.verdict`) | twelve | **NOT_ADOPTED**: 11 WITHIN, 1 UNCONFIRMED_GAIN (WQ-musique webqsp) |
| WWW (`zwww.full_verdict`) | six | **NOT_ADOPTED**: all six WITHIN |

- The screen's one GAIN, WQ-hotpotqa musique-3hop at seed 0 (+0.0154), does not hold at seeds 1 and 2 (−0.0070 and
  +0.0018). Its three-seed floor is 0.0270.
- WQ-musique webqsp gives +0.0134, −0.0004 and +0.0016 against a floor of 0.0131. It is outside the WWW rule's six.

**What follows (section 12, fixed before the fits).**
- topo_STRUCT stays in the pick.
- No drop is adopted, so the next CPU form is section 6's capacity form (hidden width and epochs), on zrckpn, the MLP of
  record since LTM, in its own declared file.
- The own verdict is filed here. It decides after the freeze, together with the other four drops' full runs
  (section 11).
