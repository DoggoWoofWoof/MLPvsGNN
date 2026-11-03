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

## 9. Amendment A1: the GNN's cut fits (4 November, about 05:00 IST, before any GNN fit)

This wires section 3's GNN part. Module: `outputs/mp_unified/zfsvg.py` (new). `zfsv.py` is not edited. Sections 2, 4, 6
and 7 stand as written.

**The fits.**

| fit | what | compared with |
|---|---|---|
| `scr-fsv-zspk-L<l>-s0`, `-hp` | zspk cut at L, seed 0: the start of the level's zg6r fits of its split | nothing (its read is checked, not compared) |
| `scr-fsv-zg6r-L<l>-s<k>`, `-hp` | zg6r cut at L | ML1's `scr-ml1-zg6r-s<k>`, `-hp`, in `scr-fsv-zg6r-L<l>-vs-full-s<k>`, `-hp` |

**How a fit runs.** Each fit is zml1s's own fit, a train and then a read. Everything is ML1's: the arm, the card, PKO's
roots and gate, the read cap (zspk 65,536, zg6r 32,768) and the device (the host's GPU, torch 2.8.0+cu128, deterministic
algorithms, TF32 off, cuBLAS workspace :4096:8). These in-process patches apply, and each is put back afterwards, also
on a failure:
- FSV's declaration, stage and stamp key (`fsv`). zml1s's fit table holds FSV's GNN fits, and its fit naming answers
  for the fit being run.
- lean_gpu's pick and pns (the pick without SEMB) without the level's cut. DOB's nine blocks are checked first. zrm
  builds its block encoder from this pick when the model is built (`zrm.py`, line 176).
- Reads keep p@swa alone (readswa, as in DEP, RTR and IRA).
- The stamp adds A1's block: the level, groups, cut, pick, full fit and start, and the pick and pns in force when the
  stamp is written.
- The code list adds the sha256 of `zfsvg.py`, `zfsv.py` and `readswa.py`.

**The start.** zfsvg's loader replaces `zml1s.load_encoder` for a cut zg6r fit. It loads the p@swa encoder of the cut
zspk of the fit's level and split, and stops unless all of these hold:
- the cut zspk was fitted and read by FSV with A1's block at that level, trained the level's pick (its `train.json`),
  and had that pick in force when stamped;
- its `models.pt` is that fit's seed 0, with a p@swa;
- it holds every encoder tensor zg6rw names, at the cut encoder's size: ML1's 224,737 numbers less 128 x (2w + 1) for
  each cut block of width w. These are the columns zrm's first layer takes per block: its raw values, z-scored values
  and mask. ML1's seed-0 zspk fits record the same widths in both splits, so the sizes are 203,489 at L3, 208,097 at L2
  and 217,057 at L1.

zg6r's own checks then apply unchanged:
- each tensor's shape must match the cut model's (`copy_encoder`);
- at least one zg6r must be built from the encoder;
- the fit's basis and hidden size must be the encoder's (`encoder_used`).

A cut zspk starts from a fresh draw, as ML1's zspk did. So no cut encoder starts from weights that saw a cut block.

**After each fit.** The fit's `train.json` must name the level's pick: its variant p's blocks and its recorded pick.
Otherwise the item exits 3, because the patch did not reach the trainer. `retry_cmd` reruns only file races and CUDA
faults, so exit 3 is not rerun.

**The compares and the calls.**
- `compare-all` runs zml1s's compare of each read cut zg6r against ML1's zg6r of the same seed and split, under FSV's
  names. Each compare record adds an FSV block: the amendment, the level, the two fits and the device.
- The full model's R@5 at each seed is section 3's: from ML1's compares `scr-ml1-zg6r-vs-zspk-s<k>`, `-hp`. FSV's
  compares must agree with them within 1e-9.
- The rule is section 4's, unchanged: the eight deciding reads, the screen and the full run, INCOMPLETE first, and the
  order L3, L2, L1. The slices are read as section 4 reads them.
- `calls` writes `outputs/fsv/calls_gnn.json` and `calls_gnn.md`.

**INCOMPLETE** (section 4) also covers these, for a cut fit and for its start, against its full fit (ML1's fit of the
same arm, seed and split):
- it names a gate other than PKO's present gate;
- its train and read stamps and its full fit's two stamps do not all name one device;
- a read cap is not the arm's;
- its trainer's code differs from its full fit's. This covers every file the full fit's stamp names, except the three
  stage modules DEP also leaves out (`zml1s.py`, `ziso.py`, `ml1start.py`);
- its `train.json` does not name the level's pick, or its hidden size, basis, config or device flags differ from its
  full fit's. The flags are deterministic algorithms, TF32, matmul precision, device, torch, cuBLAS workspace and torch
  threads;
- for a cut zg6r, its encoder record does not name all of: its start (read, seed 0), the sha256 of the start's present
  `models.pt`, at least one copy, the level, and the level's size.

**The queue (the host's GPU).** D1 stands. The GNN's L3 screen goes after the six WQ-hotpotqa halves not yet run (GNP,
GNW, LTX, GNR, GNT, GNS; RTR's has run and IRA's is running), and ahead of every full run those screens may open.

| item | GPU share | GB | after |
|---|---:|---:|---|
| `fsvg-selftest` | - | 2 | - |
| `fsvg-zspk-L3-s0` | 0.42 | 8 | the self-test, PKO's gate, ISO's verdict, the start and readswa checks |
| `fsvg-zspk-L3-s0-hp` | 0.57 | 12 | the same |
| `fsvg-zg6r-L3-s0` | 0.42 | 9 | `fsvg-zspk-L3-s0` |
| `fsvg-zg6r-L3-s0-hp` | 0.83 | 12 | `fsvg-zspk-L3-s0-hp` |
| `fsvg-compare` | - | 8 | both zg6r fits |
| `fsvg-calls` | - | 4 | the compare |

The shares and memory hold ML1's measured train peaks on the 24 GB card. The PyTorch pool caps are 0.4, 0.55, 0.4 and
0.8 (`cuda_alloc.py`).

| arm | GPU GB, WQ-musique | GPU GB, WQ-hotpotqa | memory GB, WQ-musique | memory GB, WQ-hotpotqa |
|---|---:|---:|---:|---:|
| zspk | 7.2 | 11.7 | 7.5 | 11.2 |
| zg6r | 7.8-8.0 | 17.7-18.4 | 7.3-8.7 | 11.4 |

A cut model takes fewer columns, so its peaks should not be higher. The two starts fit on the card together.

The later items are added only after `calls_gnn` names them (section 4's order): a full run's seeds 1 and 2, or the
next level's screen.

**The self-test** (`python outputs/mp_unified/zfsvg.py --selftest`) passed on the laptop before this commit. It covers:
- the names and the levels' picks;
- the encoder's size at each level, against zrm's block encoder built at ML1's widths;
- the widths' record, and its refusal of a start not at DOB's pick;
- the patches (names, fit table, pick, stamp, loader, code), and that each is put back: after the fit, after a failure
  inside it, and when lean_gpu is not at DOB's pick;
- the stamp's A1 block;
- the loader on a made-up start, and each refusal: a fit of the other split, an unread start, a start trained at
  another pick, another fit's `models.pt`, one number short, and a missing tensor;
- the problems check on a pair shaped as ML1's real records (`train.json`'s config, flags, sets and variant; the
  stamps' device, gate, read cap and code). It runs clean, then with each of nine faults alone, and with the full fit
  at another read cap;
- the R@5 reader: two R@5 for one seed, a compare not A1's, and ML1's compares missing;
- the slices: a missing read file, and a read file whose musique R@5 is not its compare's;
- the rule: PASS, FAIL, INCOMPLETE on a problem, and ADMITTED on all three seeds.

**Commit policy.** Commit only `outputs/fsv/calls_gnn.json` and `calls_gnn.md`, masked with `mask_host.py` and
`mask_md.py`, with `git add -f`. Never commit the fits, compares or read files.
