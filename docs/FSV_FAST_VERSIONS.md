# FSV: fast versions of the MLP and the GNN of record, cut by refit, then timed cold and warm

Declared 4 November 2026 (about 03:10 IST), before any FSV fit and any FSV timed number. Stage module:
`outputs/mp_unified/zfsv.py` (the MLP's fits, every compare, the calls). The GNN's fits are wired by amendment A1, in
its own module, before any GNN fit. The timing's code and its development check come in an amendment, before any timed
number. Records: `outputs/fsv/`.

## 1. Why

**What was asked (3 November).** Claim 3 compares the fast versions of the best MLP and the best GNN, "without losing
anything", timed cold and warm. "The fast versions are basically feature cuts that the respective models dont need." The
GNN is its own model, never the MLP plus message passing. This replaces the 12 October rule that speed work keeps every
input:
- the full models keep every input;
- a fast version may leave out only the inputs that its own refit shows it does not need.

**The models of record.** These are zrckpn, the MLP (LTM, ADOPT 7874e4d), and zg6r, the GNN (ML1, ADOPT). Neither has
been timed.

**What is known about which inputs each model needs.**
- **F1 (12 October), by permutation without refit:** zrc and zsp read every block above F1's 0.002 bar. Permutation
  shows what a trained model reads. It does not show what the model needs, because a refit without a block can make up
  for it.
- **DOB (filed 4 November, 00:22 IST), by refit:** zrckp was refitted without one block at a time at seed 0. Six blocks
  lost no read of the twelve:
  - topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS and WALKF.
  - dense_cos lost two in-domain reads (MIXED).
  - One drop at a time says nothing about pairs (DOB section 7).

**So the question is:** refitted without groups of those six blocks, does each model lose any read?
- The largest cut that loses none is that model's fast version.
- The full and fast versions are then timed.

## 2. The cuts (fixed now)

**The candidates** are DOB's six blocks.

**Never cut:**
- rank: zrckp's model z-scores every input against the rows the rrf column marks (DOB amendment A2);
- SEMB: rmatch requires it;
- dense_cos: DOB found a LOSS without it on two in-domain reads;
- the KT input;
- NEXT: the block that makes zrckpn;
- the GNN's links and messages: that is its message passing.

**The groups.** The candidates are grouped by the computation they share when a question is served, so a cut removes a
whole computation:

| group | blocks | what serving no longer computes |
|---|---|---|
| G_struct | topo_STRUCT, depth_STRUCT | the look's structural columns |
| G_store | SEED, DISTS | the decoded store's lines and pair distances (lean_mlp3.store_query) |
| G_walk | WALK, WALKF | lean_query's walks over the pool's families, and the hop-1 seed walks (lean_mlp, lean_mlp2) |

**The levels.** They are cumulative and are tried in this order, the largest cut first:

| level | cut | the pick keeps (the MLP adds NEXT) |
|---|---|---|
| L3 | G_struct, G_store, G_walk | rank, dense_cos, SEMB |
| L2 | G_struct, G_walk | rank, dense_cos, SEMB, SEED, DISTS |
| L1 | G_struct | rank, dense_cos, SEMB, SEED, WALK, DISTS, WALKF |

- Each step back restores the group F1's permutation found most read:
  - first SEED and DISTS (mean drops 0.0296 and 0.0158 for zrc);
  - then WALK and WALKF (0.0219 and 0.0086).
- G_struct (0.0056 and 0.0027) is cut at every level.
- The order and the groups are the same for both models. Each model's own refits decide its own level.

## 3. What runs

**The MLP.**
- zrckpn cut at level L is LTM's zrckpn with L's blocks left out of lean_gpu's pick. NEXT is kept, after the kept
  blocks.
- Everything else is LTM's: zltm's trainer, the card, the NEXT columns of each split, PKO's roots, the read cap
  (65,536) and DOB's device (the host's CPU, 6 threads, torch 2.8.0+cpu).
- **Fits:** `scr-fsv-zrckpn-L<l>-s<k>` on WQ-musique and `scr-fsv-zrckpn-L<l>-s<k>-hp` on WQ-hotpotqa.
- **Each fit is compared** with LTM's zrckpn fit of the same seed and split (`scr-ltm-zrckpn-s<k>`, `-hp`), on the same
  device.

**The GNN** (wired by amendment A1, before any GNN fit).
- zg6r cut at level L is ML1's zg6r with L's blocks left out of the pick, on ML1's device (the host's GPU).
- Its encoder starts from zspk fitted with the same cut (`scr-fsv-zspk-L<l>-s0`, `-hp`), as ML1's zg6r starts from
  ML1's zspk. A cut encoder never starts from weights that saw the cut blocks.
- **Each zg6r fit is compared** with ML1's zg6r of the same seed and split.

**The full model's R@5 at its three seeds** comes from the filed compares that hold it:
- zrckpn's from LTM's compares (`scr-ltm-zrckpn-vs-zrckp-s<k>`);
- zg6r's from ML1's (`scr-ml1-zg6r-vs-zspk-s<k>`).

FSV's own compares must give the same R@5 for the full model at a seed, within 1e-9.

## 4. The reads and the rule (fixed now)

**The deciding reads, eight.**
- The six in-domain reads that decide until the freeze (the rule of 6cb8f39, its section 2):
  - WQ-musique: hotpotqa and 2wiki;
  - WQ-hotpotqa: musique, 2wiki, and musique's 3-hop and 4-hop slices.
- The two datasets each split holds out: musique under WQ-musique, and hotpotqa under WQ-hotpotqa.
- "Without losing anything" covers the held-out reads C1R reports, so they decide here. The six-read rule adopts a
  better model; this stage admits a cut, so it is stricter.
- **Beside, deciding nothing:** webqsp, squad and metaqa on both splits, and WQ-musique's musique slices.
- The slices are read from each fit's musique read file through `zdep.slice_r5`, as LTM and DOB A3 read them. Each
  file's all-question R@5 must equal its compare's within `zdep.R5_TOL`.

**The screen** (seed 0, both splits). Per deciding read, `zltg.screen_row`:
- delta = the cut fit's R@5 minus the full model's R@5, both at seed 0;
- the floor is max(0.0075, 2 × the full model's spread over its three seeds);
- LOSS when the delta is at or below minus the floor.
- **PASS:** no LOSS among the eight. **FAIL:** any LOSS.

**The full run** (seeds 1 and 2, both splits). Per deciding read, `zltg.full_row`: `zml1.call` over the three seeds,
with the pair floor of both arms' spreads.
- **ADMITTED:** no LOSS among the eight.
- **NOT_ADMITTED:** any LOSS.

**INCOMPLETE** when any of these holds. It takes precedence over FAIL and NOT_ADMITTED, as in LTM:
- a fit, compare or read file is missing;
- a fit did not run on its full model's device, threads and read cap, or not with its level's pick;
- a fit read a gate, a NEXT scorer or NEXT columns other than its full model's fit read, or its trainer's code differs
  from that fit's (every file the full fit's stamp names; none has changed since LTM's declaration, 3aa0b58);
- the full model's R@5 at a seed differs between two compares;
- a read file's musique R@5 differs from its compare's.

**The order, per model.**
1. L3's screen runs first.
2. FAIL: the next level's screen runs.
3. PASS: that level's full run runs.
4. NOT_ADMITTED: the next level's screen runs.
5. The first ADMITTED level is the model's fast version.

If no level is admitted, the fast version is the full model, and its only lever is faster exact forms (section 5).

**What admission does not do.**
- A GAIN never admits a cut by itself and never picks a level.
- An admitted cut is a fast version, not a better model. Adopting a cut as the model of record would need an adoption
  under the six-read rule, in its own file.

## 5. The timing (fixed now in principle; the code and its development check in an amendment before any timed number)

**What is timed.** Four versions per question, on the same questions:
- the full MLP and the fast MLP;
- the full GNN and the fast GNN.

A fast version that equals its full model is timed once.

**The stages of a question:**
- (a) the shared pool build, pick O, from the first-stage lists;
- (b) each version's own inputs, from the pool;
- (c) its forward.

**What is reported:**
- **Model-only** is (b) + (c). **End to end** is (a) + (b) + (c).
- The first-stage retrieval (the question's encoding, and the dense and SPLADE lists) is shared by every version. It is
  reported beside, not inside.

**Cold** (the 8 October rule):
- batch 1 and one thread;
- every per-question structure is computed from scratch for each question, and nothing is kept from an earlier
  question;
- index-time structures (the graph, the stores, the weights) are loaded before timing and reported on their own rows.

**Warm** sits beside cold, never instead of it, and is labelled warm:
- the per-question inputs are already built and in memory;
- the forward alone is timed, on one thread, at batch 1 and at batch 16.

**Forms.** Each version is timed at its fastest exact form. A form is exact when its scores are within 1e-4 of the fit's
filed read and its top 5 is the same (C3's rule). Every exact form available to one model is given to the other where
they share a stage.

**Questions:** the first 200 s1eval questions of 2wiki, hotpotqa and musique, in the look's order.

**Device.** The laptop, as in C12 and P4c:
- pinned to LP 2, with EcoQoS opted out;
- the session idle;
- the power state recorded.

If the laptop cannot hold a dataset's index, the amendment names the device before any timed number.

**Checks,** in a separate pass after the timed one, untimed:
- each timed question's scores against the fit's filed read file;
- its pool against PKO's filed pool.

A question that fails a check is not kept. More than 2% failing stops the dataset (P4a's rule).

**Reported, with no call:**
- per stage, p50 and p95 in ms;
- model-only and end to end;
- GNN / MLP ratios, full against full and fast against fast, paired by question, with 95% intervals over 2,000 question
  resamples (seed 0);
- the warm batch-1 and batch-16 times.

Claim 3 is stated from this record, whatever the ratios turn out to be.

## 6. What follows (fixed now)

- **Each model's fast version** is its first ADMITTED level, or the full model if none is admitted.
- **The frozen pair.** If the pair of record at the freeze (00:00 IST, 7 November) is not zg6r and zrckpn, the admitted
  levels are screened again on the frozen models before any timing. The order and the rule stay the same; an amendment
  names the fits.
- **A verdict closes only the level it tested.** A cut outside these levels (for example, dense_cos) needs its own file.

## 7. Risks, stated before any number

- **The candidates come from DOB's single drops on zrckp.** zrckpn adds NEXT, and the GNN passes messages, so either
  may need a block DOB did not. The screens decide that, for each model.
- **The deciding reads include the held-out reads,** whose floors are wider. A cut can fail on them by chance; the
  floors are each full model's own.
- **Nested levels:** a FAIL at L3 does not say which group was needed. The order says only which group comes back first.
- **CPU time.** zrckpn's LTM fits took 1.7 to 2.5 hours each at 6 threads, with peaks of 26.8 to 31.0 GB (the feeder's
  records).
- **GPU time.** One level's GNN screen is two zspk fits (47 to 80 minutes each in ML1) and two zg6r fits (2.2 to 3.3
  hours each). Its full run is four more zg6r fits.
- **The cut fits may not be in time for the freeze.** In that case the timing reads the full models, and the fast
  versions are timed when they are admitted.

## 8. How it runs (the MLP)

Feeder items run on the host's CPU, 6 CPUs each. Each fit reserves its full model's measured peak (26.8 GB, and 31.0 GB
on WQ-hotpotqa).

```
python outputs/mp_unified/zfsv.py --selftest
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zfsv.py fit --name NAME
python outputs/mp_unified/zfsv.py compare-all
python outputs/mp_unified/zfsv.py calls          (outputs/fsv/calls.json and .md)
```

**The queue.**
- `fsv-selftest` runs first, then L3's two screen fits, then `fsv-compare` and `fsv-calls`.
- A later level's items, or a full run's, are added only after the calls name them (section 4's order).
- The items sit right after U2x's merges, ahead of DOB A3's last pending fit (`dob-xtopo-s2-hp`), for D1's reason
  below.

**D1, the GPU's order** (default first). The GNN's screen goes after the GPU items queued now:
- these are the WQ-hotpotqa halves of RTR, IRA, GNP, GNW, LTX, GNR, GNT and GNS;
- the GNN's screen goes ahead of any full run those screens open.

This puts measuring Claim 3 ahead of further model iterations, as the user's 4 November order puts it ("verifying the
claims and measuring all"). It may push a full run past the freeze. The user decides.

**The self-test** (`python outputs/mp_unified/zfsv.py --selftest`) passed on the laptop before this commit. It covers:
- the names, the levels' picks and the groups;
- the patches of zltm's table, pick, naming and code, and of zml1s's declaration, stage and key, and that each is put
  back, also on a failure;
- the stamp's FSV block;
- the screen and the full run on made-up R@5: PASS, FAIL, ADMITTED and NOT_ADMITTED, the 0.0075 floor and a wider one,
  an unconfirmed seed-0 gain, a held-out read and a slice deciding, a beside read not deciding, and INCOMPLETE on a hole
  (holding a LOSS back) and on any problem;
- section 4's order over the levels;
- the R@5 reader (two R@5 for one seed, a missing or foreign compare) and the slices (a read file whose musique R@5 is not
  its compare's).
