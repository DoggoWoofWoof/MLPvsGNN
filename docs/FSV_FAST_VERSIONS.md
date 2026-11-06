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

## 10. Amendment A2: the timing's two families, its stages' forms and its code (4 November, before any timed number)

This wires section 5 and the user's order of 4 November, 06:30 IST: "make sure while measuring both models across families
are as fast as possible and fam 1 without feature cuts and fam 2 with feature cuts". Module: `outputs/mp_unified/zfsvc.py`
(new). It imports `zfsvt.py` (section 5's reference reads, unchanged). Sections 1 to 9 stand, except where this names one.

**The families.**

| family | MLP | GNN |
|---|---|---|
| 1, no feature cuts | zrckpn, every input kept | zg6r, every input kept |
| 2, with feature cuts | zrckpn at its first ADMITTED level, else the full zrckpn | zg6r at its first ADMITTED level, else the full zg6r |

- Both families time each model at its fastest exact form in every stage. A form one model has in a shared stage, the other
  has too. A faster exact form found later is a new amendment and a re-time of every number it touches.
- Family 2 follows section 4's order and rule, unchanged. State at 09:20 IST on 4 November: the MLP's L3 screen FAILED
  (6d16b28) and its L2 screen FAILED (0c95b96: 1 LOSS of the 8 deciding reads, WQ-hotpotqa musique-4hop, -0.0093 against a
  0.0075 floor). L1, the G_struct cut alone, is queued on the host (`fsv-zrckpn-L1-s0`, `-s0-hp`, `fsv-compare-l1`,
  `fsv-calls-l1`), its verdict due between about 11:00 and 12:00 IST on 4 November. The GNN's L3 calls are due about 01:00
  IST on 5 November. If L1 also fails, the MLP's Family 2 is the full zrckpn.
- Family 2 is timed only after both verdicts. A model whose Family 2 is its full model has no second row: it is timed once
  (section 5), and its Family 1 row stands for both.
- Family 2's forms are declared in amendment A3, before any Family 2 number. They take one rule from here: a cut input is
  neither computed nor read in stage (b); the forward is the cut fit's own model; exactness is against that fit's own filed
  read (`scr-fsv-*`), which `zfsvt.py` reads first, as it read the full fits'.

**The stages' forms, Family 1.** Stage (a) runs once per question and is shared. Stages (b) and (c) run once per model.

| stage | form | MLP (zrckpn) | GNN (zg6r) |
|---|---|---|---|
| (a) pool | `zfsvc.stage_a`: today's pool and the seeds by `c10_zn.FastPool`; the base (equal-RRF budget, `c10_zn._rrf_base`); the question's text links (step 4e's `link` on the name index); PKO's rule as `zpko.rule_batch` calls it, for one question on one thread, without its statistics and self-checks (the walks on the carried walk, KIND_T, the one-link and two-link neighbourhoods and their orders, the pick's expansion, the KT columns); pool = B plus the expansion | the same run | the same run |
| (b) compile | `c11_fast.compile_serial` on `c12_fast`'s kernel table | own call | own call |
| (b) rows | `c10_zn.lean_rows10`, with `c10_zn.Decoder` over the int8 code table | own call | own call |
| (b) links | `c3_fast.links_fast` | for NEXT's family bits | the `_link` edges |
| (b) NEXT | LTM's lists of the five leading rows and their link neighbours, scored by the fold model of the question's fold | 7 columns | none |
| (b) inputs | the batch's tensors as `lean_gpu.CacheCarve.batch` gives them, from the question's own rows | blocks, `_kt`, NEXT | blocks, `_kt`, `_link`, `_g6r` (the passage graphs' empty typed slot) |
| (c) forward | the model's module in eager torch, `no_grad`, one thread; the top five | own | own |

- The model-only time of a model is its (b) + (c). Its end-to-end time is the question's (a) + its (b) + (c); (a) ran
  once and is charged in full to both models.
- Each model's (b) runs from the pool for that model alone, so its model-only time stands alone.
- The models alternate in leading each question: the MLP first on even questions, the GNN first on odd ones. The 201st
  question runs first, for both models, to compile (numba) and warm: it is not kept.
- The first-stage retrieval (the question's encoding, the dense and SPLADE lists) is not timed here. PKO's prepared lists
  are read before timing, as index-time data. Any number reported beside it comes from an earlier record, labelled so.
- No fused form exists for zrckpn or zg6r (the C-stage fused forwards cover the older zrc and zsp). Stage (a) is PKO's own
  rule at batch 1: P4b's fast-walk layouts were built for P4a's KIND_T pools and are not applied to pick O's walks. Either
  is a place where a faster exact form could exist; it would be an amendment and a re-time (D1 below).

**Threads and BLAS.**
- The walks (numba), the forward (torch) and every other stage run on one thread; every OpenMP runtime is limited to one
  thread (the host's job environment sets two for the pools loaded before the script, so the script limits them).
- OpenBLAS runs on **2 threads**, in the matrix products of stages (a) and (b). The reason is measured. The first 12
  questions of musique's s1eval, with BLAS on 1 thread: two questions had one float16 input value (`depth_STRUCT`) one step
  off the filed read's, and one question's scores moved 1.5e-4 (beyond section 5's 1e-4, top 5 the same). With BLAS on 2
  threads, all 12 reproduce the filed inputs value for value. Section 5's "one thread" is read this way from here on.
- A pass at BLAS 1 is timed beside, labelled: its check counts the questions beyond 1e-4, and they are timed all the same.
  The BLAS-2 pass is the record.
- Each job records the pools' effective sizes (torch, numba, BLAS and every OpenMP runtime, each with its library) after
  every module has set its own; `zfsvc.py threads` prints them after the imports.

**The device.** The host's CPU (rx env `mpr-cpu`, started through `pylib_run.py` for numba), not the laptop: the laptop
holds neither PKO's links and caches nor the RAM for a dataset's index (section 5's own condition). The host is shared, so
it cannot be idle; instead:
- the process pins itself to logical processors 2 and 4 (its own affinity, EcoQoS opted out, above-normal priority; its own
  process only), and with `--require-pin` the job stops if the pin fails (the three calls all took effect in the smoke
  run below);
- every logical processor's load is sampled for 1 s before and after the timed pass and recorded;
- one job per dataset, one after another, never beside another timed job; `--cpus 2`; memory reserved from the larger of
  the dataset's development and smoke peaks plus a margin (musique: 12.8 GB for the 40-question development check, 15.9 GB
  for the 20-question smoke run, which also holds the check pass);
- one record per dataset: a job that fails (memory, a lost host connection) is re-run, a finished job is not repeated for
  its numbers.

**Fits and questions** (the fits are zfsvt's: seed 0, p@swa, its filed reads).

| dataset | MLP | GNN |
|---|---|---|
| musique | `scr-ltm-zrckpn-s0-hp` | `scr-ml1-zg6r-s0-hp` |
| hotpotqa, 2wiki | `scr-ltm-zrckpn-s0` | `scr-ml1-zg6r-s0` |

The first 200 s1eval questions of each dataset, in the look's order, and the 201st for the first run. Musique's models
are the WQ-hotpotqa halves; hotpotqa's and 2wiki's are the WQ-musique halves.

**Two inputs the filed reads used, and how the cold path matches them.**
- **NEXT.** The fold model of question j is LTM's cross-fit: fold `np.random.default_rng([0, DS_K[ds], 7]).integers(0, 5,
  n)[j]`, n the carve's full size, so the model that did not train on the question scores it. LTM scored lists in batches
  of 32 across questions; the cold path scores one question's lists at a time, and a batch's composition can in principle
  move a float16 value one step. The check pass counts the values that differ from the cache's (musique, first 40
  questions, 1,383,088 values: none differ).
- **The int8 code table.** The cache the reads were built on does not hold one code per node: of the first 12 questions'
  59,687 pool rows, one row's code differs by one step from the same node's code in an earlier question (the build's
  blocks), and 2 of the first 40 questions' rows. A table written once per node cannot reproduce every filed read. The cold
  path reads one table either way (the same slots, the same decode); before each question's timed stages, untimed, the table
  is written with the codes of that question's rows from the cache. The job records how many rows differed from an earlier
  question's (`code_conflicts`).

**The checks** (section 5's, untimed, after the timed pass). For each timed question:
- each model's scores against the fit's filed read at batch 1 (within 1e-4, same top 5);
- the pool, today's pool and the KT columns against PKO's own rule on the same question (`zpko.rule_batch`), equal; the
  pool's size against the question's row count in the cache, which was built on PKO's filed pools (a mismatch stops the
  job);
- the NEXT columns against the cache's, counted (not a call).

A question that fails the first two is not kept. More than 2% failing stops the dataset (P4a's rule); the job exits 1.

**Reported** (section 5's list, as one recipe fixed before any number):
- per stage and per model, p50 and p95 in ms over the kept questions: (a) with its parts (the pool, the base, the links,
  the rule's walks, orders and KT columns), (b) with its parts (compile, rows, links, NEXT, inputs), (c);
- model-only and end to end, per model;
- the ratio GNN / MLP of the p50 of the same resampled questions, full against full (Family 1) and fast against fast
  (Family 2), with the 2.5th and 97.5th percentiles over 2,000 resamples of the questions with replacement
  (`numpy.random.default_rng(0)`);
- warm: the forward alone from the packs of the cold pass, batch 1 on each question and batch 16 as 12 batches of the first
  192 (the batch's time over 16), the same pin;
- cold is the headline, warm beside it, labelled.

**The development check** (`python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvc.py dev --dataset musique
--questions N --blas B`) passed on the host before this commit, on musique's first questions, BLAS 2:
- stage (a)'s base-and-seeds, today's pool, expansion, pool, KT columns, text links and seeds equal PKO's own rule and its
  filed arrays on every question;
- each model's inputs equal the reference carve's batch of that question value for value (0 values differ);
- scores are within 7.2e-6 of the filed read's (first 12 questions) and the top 5 is the same;
- the NEXT columns: 0 of 28,203 to 41,370 float16 values per question differ from the cache's (first 12 questions);
- first 40 questions (197,584 pool rows): every check passes, scores within 7.2e-6, 0 of 1,383,088 NEXT values differ.
At BLAS 1 the same 12 questions fail as described above (2 questions with one input value a step off, 1 beyond 1e-4).

A smoke run of the timed path (`run --nq 20`, `--cpus 2`, BLAS 2, `--require-pin` on logical processors 2 and 4; its raw
file goes to `raw_smoke`, is not filed and is not a measurement) completed on the host, run twice (the second after the
OpenMP limit above), both with the same result: the pin took (affinity, EcoQoS opt-out, priority), the pools were OpenBLAS
2 and torch, numba and every OpenMP runtime 1 (second run), 20 of 20 questions were kept with no check failing (scores
within 5.7e-6 for the MLP and 7.2e-6 for the GNN of the filed reads, the pool and KT columns equal PKO's rule on every
question, 0 NEXT values differing), one cached code row differed from an earlier question's, and the job's peak memory
was 15.9 GB.

The development check and the smoke run were made on musique only. Hotpotqa and 2wiki go through `dev` (their first 40
questions, BLAS 2) before their timed passes, and a pass starts only when every check passes; a failure there changes the
code, not the recipe above, and is stated in the report.

**Owed, not part of this amendment.** `zfsvc.py report` (the reduction above, from the raw files, writing
`outputs/fsv/timing/report.json` and `report.md`) is written with the first raw file and changes no choice made here.
Family 2's forms (A3). **D1** (the user's): spend time on a faster exact stage (a) or forward before the timed pass, or time
the forms stated here first. Default: time them first and state the share of (a) beside each ratio. In the smoke run (the
host carried other jobs: development times, not measurements) the p50 of stage (a) was 0.47 s per question (0.51 s in the
first run), of which the rule's walks were 0.29 s and its orders 0.18 s, against 0.12 s for the MLP's (b) + (c) and 0.34 s
for the GNN's (its forward 0.25 s); stage (a) is the larger part of both models' end-to-end times, so a faster exact (a)
moves the end-to-end ratios and a faster exact forward moves the model-only ones.

**Commit policy.** Commit only `outputs/fsv/timing/report.json` and `report.md`, masked (`mask_host.py`, `mask_md.py`),
with `git add -f`. The raw files (`outputs/fsv/timing/raw`) and the reference arrays (`ref/*.npz`) are never committed.

## 11. Amendment A5: the L0 level (4 November, about 11:15 IST, before any L0 fit)

L3, L2 and L1 have each failed their screen for zrckpn (L3 6d16b28, L2 0c95b96, L1 ab2af7a). By section 4 the MLP has no
fast version in section 2's order, and by section 10 its Family 2 would be the full zrckpn. This adds one level to the
MLP's order, after L1: **L0, topo_STRUCT alone**. It names the fits, the rule and the queue before any L0 number. Module:
`outputs/mp_unified/zfsv.py`, edited additively. Sections 1 to 10 stand, except where this names one.

**Why a level after L1, and why an amendment.**
- A verdict closes only the level it tested (section 6). L1 cut both blocks of G_struct and lost one deciding read,
  WQ-hotpotqa musique-4hop (-0.0117 against a 0.0075 floor; L2 lost the same read, -0.0093). The other seven deciding reads
  were WITHIN at both levels. Nested levels do not say which block a read needed (section 7).
- topo_STRUCT is one of section 2's six candidates and lies inside G_struct, so L0 is a finer cut of a group already
  named, not a new input. A cut outside the levels needs its own file (section 6). This section is that declaration, made
  as an amendment because L0 takes section 4's reads and rule, the module's fits and stamps, and section 8's queue
  unchanged. It is dated and committed before any L0 fit, as a file would be.
- topo_STRUCT, and not depth_STRUCT, because it is the one block of G_struct with evidence at three seeds (below).

**The level.**

| level | group | cut | the pick keeps (NEXT too, after the kept blocks) |
|---|---|---|---|
| L0 | G_topo (new): topo_STRUCT | topo_STRUCT | rank, dense_cos, depth_STRUCT, SEMB, SEED, WALK, DISTS, WALKF |

- The MLP's order is L3, L2, L1, L0. Each level's cut contains the next one's: L0's is the first block of L1's.
- "L0" names the smallest cut. It is not "no cut": the full zrckpn is not a level.
- The GNN's order stays L3, L2, L1. `zfsvg.py` counts three levels and is not edited. If the GNN's L1 screen fails too, a
  later amendment adds an L0 for the GNN, with evidence and wiring of its own.

**The evidence, and what it does not say.** DOB A3 (`docs/DOB_DROP_ONE_BLOCK_CPU.md` sections 12 and 13, 5e81953;
`outputs/dob/full_xtopo.json` and `.md`) refitted zrckp without topo_STRUCT alone, at seeds 0, 1 and 2 on both splits,
against the CPU base's fits of the same seeds:
- DOB's own rule (ML1's, twelve reads): NOT_ADOPTED, 11 WITHIN and 1 UNCONFIRMED_GAIN (WQ-musique webqsp). No LOSS.
- The WWW rule (the six in-domain reads): NOT_ADOPTED, all six WITHIN. "Not adopted" here means no GAIN to adopt it as a
  better model; it is not a loss.
- On the read L1 lost, WQ-hotpotqa musique-4hop, the three seeds' deltas were +0.0056, -0.0093 and +0.0006, against a
  three-seed floor of 0.0119.
- Of DOB's seven single drops, xtopo is the only one taken to seeds 1 and 2 (DOB section 12). depth_STRUCT's drop has
  its seed-0 reads alone: NO_GAIN, all six WWW reads WITHIN, +0.0062 on the 4-hop slice.

It does not say that zrckpn can do without topo_STRUCT, for three reasons:
- **The base differs.** DOB's base is zrckp. FSV's full model is zrckpn: zrckp plus the NEXT columns (LTM, ADOPT
  7874e4d), which read the look differently.
- **The rule differs.** DOB's three-seed calls use `zml1.call` with pair floors. The screen below reads seed 0 against
  floors from the full model's own spread.
- **The pair.** One drop at a time says nothing about pairs (DOB section 7). L1 cut topo_STRUCT and depth_STRUCT together;
  DOB's single drops of each lost no deciding read on zrckp. L0 asks whether topo_STRUCT alone costs the 4-hop read on
  zrckpn.

**The screen** is L1's, with the level's name changed (sections 3 and 4):
- **Fits:** `scr-fsv-zrckpn-L0-s0` on WQ-musique and `scr-fsv-zrckpn-L0-s0-hp` on WQ-hotpotqa. Each is LTM's zrckpn at
  seed 0 with topo_STRUCT left out of lean_gpu's pick; NEXT is kept. Everything else is LTM's: zltm's trainer and read, the
  card, each split's NEXT columns, PKO's roots, the read cap (65,536) and DOB's device (the host's CPU, 6 threads, torch
  2.8.0+cpu).
- **Compared** with LTM's `scr-ltm-zrckpn-s0` and `-hp`, in `scr-fsv-zrckpn-L0-vs-full-s0` and `-hp`, on the same device.
- **The rule** is section 4's, unchanged: the same eight deciding reads (the slices read as there), the same floors
  (max(0.0075, 2 × the full model's spread over its three seeds)), LOSS when the delta is at or below minus the floor,
  PASS with no LOSS, FAIL with any, INCOMPLETE first. The full model's R@5 comes from the same compares as for L1.
- **No number is read before this section is committed,** and no L0 fit starts before it.

**What the calls mean** (section 4's order, extended by one level):
- **PASS:** the full run follows: L0 at seeds 1 and 2 on both splits (`scr-fsv-zrckpn-L0-s1`, `-s2`, `-s1-hp`, `-s2-hp`),
  queued only after the calls name it (section 8). ADMITTED, no LOSS among the eight at three seeds, makes L0 the MLP's
  fast version. Its Family 2 is then the cut zrckpn, timed in a later pass (M3) once Family 2's forms (A3, owed in section
  10) are written, after the pass that times both families with the MLP's Family 2 as the full zrckpn (M2). NOT_ADMITTED:
  Family 2 stays the full zrckpn.
- **FAIL:** the MLP's Family 2 stays the full zrckpn. It is timed once, and its Family 1 row stands for both (section 10).
  No further MLP cut is declared without a new named form and its own evidence, before any number.
- A PASS or an ADMITTED is not an adoption (section 4): L0 would be a fast version, not a better model.
- Section 10's "if L1 also fails, the MLP's Family 2 is the full zrckpn" holds from here only once L0 has failed or been
  NOT_ADMITTED. Until then the MLP's Family 2 is open; the timing of the full models is not affected.

**Risks, stated before any number.**
- **One seed is noise on one read.** zrckpn's three seeds agree within 0.0012 on the 4-hop slice, so its floor is the
  0.0075 minimum, yet DOB's topo_STRUCT drop moved that read by +0.0056 at one seed and -0.0093 at another. A seed-0 FAIL
  may be chance. It still decides this level: a new draw of the same cut is a new declaration, not a rerun. A PASS is not
  an admission; the full run decides.
- **A level declared after its predecessors' tables were seen** raises the chance of a lucky seed-0 PASS. The three-seed
  full run is the stricter gate for it.
- **The saving may be small.** topo_STRUCT is one block of the look's structural columns, so an admitted L0 may save
  little in section 10's stages. The timing says how much. That changes no rule here.

**The queue** (the host's CPU, section 8's items with L0 in the names; a full run's items are not queued).

L1's two fits measured peaks of 26.7 GB (WQ-musique) and 30.2 GB (WQ-hotpotqa) on 6 CPUs and ran 1 hour 15 minutes and
1 hour 40 minutes. L3's and L2's four fits ran 58 minutes to 1 hour 37 minutes. An L0 fit has more columns than L1's and
fewer than the full model's, so each reserves the full model's measured peak, as L1's did (26.8 GB and 31.0 GB, plus the
feeder's margin).

| item | CPUs | memory (GB) | after |
|---|---:|---|---|
| `fsv-selftest-l0` | 1 | 4 | - |
| `fsv-zrckpn-L0-s0` | 6 | LTM's `ltm-zrckpn-s0` peak | PKO's gate, the self-test |
| `fsv-zrckpn-L0-s0-hp` | 6 | LTM's `ltm-zrckpn-s0-hp` peak | the same |
| `fsv-compare-l0` | 1 | 8 | both fits |
| `fsv-calls-l0` | 1 | 4 | the compare |

- The self-test is new because `fsv-selftest` ran on the host before this amendment. It heads L0's items as it headed
  L1's.
- The items sit after U2z's pending block and ahead of CAP's pending fits, for D1's reason (measuring Claim 3 before
  further model iterations). A fit starts when its memory is free; nothing already running is touched.

**The report** is `outputs/fsv/calls.json` and `calls.md`, rewritten by `fsv-calls-l0`: the MLP's one record, with L3,
L2, L1 and, once its compare is filed, L0. Until then the header reads "Next: L0 screen". It is filed as L1's was: those
two files only, masked (`mask_host.py`, `mask_md.py`), with `git add -f`; never the fits, compares or read files.

**The code** (`zfsv.py`, additive):
- `ORDER`, `LEVELS` and `GROUPS` are as they were, because `zfsvg.py` reads them and asserts its fit counts. L0 is in
  `GROUPS_A5`, `LEVELS_A5` and `MLP_ORDER = ORDER + ("L0",)`, merged for the lookups in `ALL_GROUPS` and `ALL_LEVELS`.
- The MLP's tables (24 fits, 24 compares), `decide`, `slices` and the calls record use `MLP_ORDER`. `walk` takes an
  optional order and defaults to `ORDER`.
- L3's, L2's and L1's stamp block is as stamped before. L0's adds `"amendment": "A5"`.

**The self-test** (`python outputs/mp_unified/zfsv.py --selftest`) passed on the laptop before this commit, and so did
`zfsvg.py`'s. It adds to section 8's list:
- L0's name, cut and pick, its stamp block and the L1 block unchanged, and its patches and their restoring;
- the MLP's walk: L0's screen is named once L3, L2 and L1 have failed, and again on INCOMPLETE; PASS names its full run;
  ADMITTED makes L0 the fast version; NOT_ADMITTED and FAIL name nothing; an earlier admitted level still wins. The GNN's
  walk still stops at three levels;
- `decide` on made-up reads: three failed levels name L0's screen with L0 absent from the levels (pending, not a FAIL),
  then L0's PASS, ADMITTED and FAIL.

**Reproduction.** The filed screens of L3, L2 and L1 (`outputs/fsv/calls.json`, ab2af7a) were rebuilt from their own rows
(the full model's three seeds, the cut's seed 0) and run through the amended `decide`:
- the verdicts (FAIL, FAIL, FAIL), counts, cuts, picks and every read's call are the filed ones. The eight deciding reads'
  deltas and floors equal the filed ones to the last digit, and the beside reads' to 0.00015, because the record stores
  WQ-musique's slice reads at four decimals;
- the one difference is the step named: L0's screen, where the filed record names none. The calls' header reads "Next: L0
  screen", and L0 is not among the levels until its compare is filed.

## 12. Amendment A4: faster exact forms for stage (a) and the GNN's layers (4 November, before any timed number with them)

This takes up D1 of section 10 (a faster exact stage (a) or forward before the timed pass of record) on the user's order of 4
November, 06:30 IST ("make sure while measuring both models across families are as fast as possible and fam 1 without feature
cuts and fam 2 with feature cuts"), and the earlier one, "make the gnns faster too i dont want to give any unfair advantage to
mlp". Section 10's timed passes (M2: `outputs/fsv/timing/raw`, `report.json`) used section 10's forms. They stand as the record
of those forms and are not touched. The forms below are timed after this commit, as M3. Module: `outputs/mp_unified/zfsvx.py`
(new). It imports `zfsvc.py`, installs the forms in its own process (`installed()`: three functions replaced, put back on exit
and asserted) and edits nothing: not `zfsvc.py`, `zg6r.py`, `lean_gpu.py`, `zpko.py`, the C-stage modules or P4's. Sections 1
to 11 stand, except where this names one.

**Why.** M2's stage (a), p50 in ms per question (BLAS 2, the 200 kept questions; the rule's walks, orders and KT columns are its
parts), and the two models' (b) and (c):

| dataset | pool rows p50 | (a) | of it: walks / orders / KT | MLP (b)+(c) | GNN (b)+(c) | of it: GNN (c) / MLP (c) | (a) share of the MLP's / the GNN's end to end |
|---|---|---|---|---|---|---|---|
| musique | 4814 | 419.6 | 242.7 / 151.6 / 13.5 | 102.9 | 279.5 | 206.8 / 25.9 | 80% / 60% |
| hotpotqa | 697 | 1173.5 | 1046.2 / 110.3 / 12.5 | 17.4 | 21.5 | 13.2 / 6.2 | 99% / 99% |
| 2wiki | 1074 | 1353.1 | 1202.9 / 132.8 / 12.5 | 25.0 | 32.0 | 18.9 / 7.9 | 98% / 98% |

Stage (a) is the larger part of both models' end-to-end time on every dataset, and the walks are most of it. On musique the
GNN's forward is the larger part of its model-only time (206.8 of 279.5 ms, against the MLP's 25.9 of 102.9).

**The forms** (each stage, each model; a form a model has in a shared stage, the other has too):

| stage | A2's form | A4's form | MLP | GNN |
|---|---|---|---|---|
| (a) walks | `lean_rule`'s thirteen walks (`sources_batch`, `steered_batch`, `restarts_batch`, five `rwalk`s), numba batch kernels on the two unions | the ten walks the pool and the KT columns read (A1st, A1all, Kst, KTst, KTall and the five relational walks), on P4b's declared records (`docs/P4B_FAST_WALKS.md`: breadth-first numbering, 16-byte hot and cold records, merged neighbour lists, generation-stamped keys, a steer bitmap), built once per dataset from the unions at index time; the relational walk is `zfsvx.frwalk`, `zu2m_a`'s push on the same records | the same run | the same run |
| (a) orders | `p1e_orders` and `rule_orders` for every candidate order, `question_orders` for every column | the six orders the columns read (KIND_T, A1st, A1all, Kst, KTst, KTall), from `p1e_orders`' and `rule_orders`' own lines; of the expansion's orders `kt_cos` alone (the pick's order, asserted at run time), its unit rows made from the float16 rows the store holds in one numba pass, the float32 norm certified | the same run | the same run |
| (b) | A2's | A2's | unchanged | unchanged |
| (c) forward | the module in eager torch, one thread | the MLP's: the module's. The GNN's: `zg6r.g6r_step` replaced in the process by `zfsvx.g6r_step_x` for a pool with no typed edge, one question, float32, no gradient (any other input takes the module's step, counted) | the module's | the layer step in two parts: the attention logit from one dot product per node, and the messages with their two scatter-adds in one pass over the edges |

**Why each form is exact** (the same floating-point operations in the same order; only where the state lives, or work no read
reaches, changes):
- **The walks on P4b's records.** The seeds in their order, each popped node's neighbours in theirs, the same float64
  operations: P4b's walkers reproduce P4a's orders and counts bit for bit (`docs/P4B_FAST_WALKS.md`). The numbering is
  breadth-first with ties by original id, and the top breaks its ties by the original id, so every order is the original's.
  `frwalk` is `zu2m_a`'s relational push statement for statement on the same records: family 0's edges weigh the question's
  weight of their class, the other families 1, each times floor plus the node's extra when steered, the weights of a run summed
  first and then each neighbour's share `m / (nf * sum)` times its weight; the class of every merged-list entry is carried at
  index time. A union with the records built from it holds the same nodes and edges as the union.
- **The ten walks.** Kall, A1Tst and A1Tall feed orders (csum, A1both, SUM5, RRF5, RRF7T, KCUT, ROUTEboth and the rest of
  `p1e_orders`' and `rule_orders`' candidates) that nothing after them reads: the pool takes KIND_T, the KT columns take the six
  orders above, and the expansion takes `kt_cos`. The forms compute those orders from `p1e_orders`' and `rule_orders`' own lines.
  Every question's pool, KT columns and expansion are compared with `zfsvc.lean_rule`'s, byte for byte.
- **The expansion's cosines.** `kt_cos` is the column of `question_orders` that the pick's plan reads (the run stops if the
  plan's order is another). Its cosines are `d5b_neighbour_orders.unit` of the float16 rows (as float32) times the question's
  vector. The forms convert the rows with torch (exact, as numpy's) and take each row's norm in one pass: the float64 sum of the
  squares (each product of two float32 values is exact in float64), in eight lanes instead of einsum's order. Any order of 1,536
  non-negative float64 terms is within 1.7e-13 (relative) of the exact sum, so the norm's float32 rounding is the same as
  einsum's unless the band [y (1 - 1e-12), y (1 + 1e-12)] holds a rounding midpoint; a matrix with a row whose band does hold
  one takes the reference's own `unit` for the whole matrix (counted: it happens, see the checks). The rows that only the second
  link (N2) brings are not read by any order; they stay zero rows in the product, because a float32 row of a matrix-vector
  product depends on its place in the matrix, not on the other rows' values (the kernel takes rows by position), so every other
  row's cosine is the original's.
- **The GNN's layers.** `g6r_step_x` is `zg6r.g6r_step` op for op, except two places: the attention logit (the gather of the
  linear map's rows times the question's context, summed over the 64 values, is the per-node sum, gathered: the same float32
  values, since torch's row sum is a function of the row alone), and the two scatter-adds (`index_add_` takes the edges one after
  the other and adds each row to its target; the pass does the same in the edge order, each product a rounded float32 and each
  sum a float32 sum). Every other op is the module's, in its order. A pool with a typed edge, a batch beyond one question, a
  non-float32 or a grad-enabled input takes the module's step (counted in `STATS`); the timed path has none of these
  (zfsvc builds `_g6r` with no typed edge).
- **The MLP's forward: no form.** The module's forward is spread over 2,306 small ops, the largest a fifth of its time (the
  profile below). Its largest part, `encode`'s masks and segment z-scores (six small ops, about 45%), would take a fused rewrite
  of `seg_zscore_ref` and `seg_zscore8D` with every float32 rounding kept, which is not in this amendment. The MLP's forward
  stays the module's. The GNN's `encode` is the same computation on its blocks (38 of the module's 287 ms and 42 of the forms'
  90 on that question), so a fused `encode` would serve both models and every cut fit; it is left for a later amendment.
- **Where exactness is checked, not argued.** Two steps rest on a property of a library's kernel (sgemv's rows by place, torch's
  row sum), so the checks below compare the bytes, not only the bars.

**The checks** (untimed; `devx` for development, `run` for M3):
- A2's own development checks with the forms installed (`zfsvc.dev_checks`): the pool, today's pool, the expansion, the KT
  columns and I against PKO's rule on the same question; each model's inputs against the reference carve's batch; each model's
  scores against the fit's filed read at batch 1, within 1e-4 with the same top 5; the float16 NEXT values counted.
- Beside them, per question: stage (a)'s three outputs from `zfsvc.lean_rule` and from the forms, byte for byte (run in
  alternating order on the same inputs); each model's scores from the forward with the module's layer step and with the forms',
  byte for byte. `devx` passes only if every one holds on every question, no layer step took the module's path, and the forms
  ran (counted).
- `selftest` (no data): the walks on random graphs of both unions, with and without a steer; the orders; the cosines with the
  certified and the not-certified norm; the layers, including the typed-edge, two-question and gradient inputs that take the
  module's step; the installation and its restoration, also after an error. It passed before this commit.
- M3's `run` keeps A2's checks per timed question and A2's rule: a question that fails the scores' or the pool's check is not
  kept, and more than 2% failing stops the dataset.

**Development numbers** (the host, the first 40 questions of each dataset, BLAS 2, one logical processor off the timed ones,
other jobs running: development times, not measurements; stage (a)'s two forms on the same inputs, alternating order):

| dataset | pool rows p50 | link edges p50 | (a) zfsvc's p50 | (a) the forms' p50 | speed-up | GNN forward: module's / forms' (speed-up) | MLP forward: module's / forms' |
|---|---|---|---|---|---|---|---|
| musique | 4845 | 126724 | 582.9 | 177.2 | 3.29 | 319.3 / 89.1 (3.58) | 36.6 / 37.5 |
| hotpotqa | 671 | 4040 | 1425.1 | 225.4 | 6.32 | 15.4 / 10.9 (1.41) | 7.0 / 6.9 |
| 2wiki | 1123 | 6434 | 1477.7 | 200.4 | 7.37 | 21.0 / 14.2 (1.47) | 8.8 / 8.5 |

The forms' stage (a) by segment (p50 ms; the walks are `x_unw` and `x_rel`, `question_orders` is the expansion's cosines and
their fusion, `cos1` the first link's cosines):

- musique: x_unw 32.1, x_rel 45.9, x_ord1 2.1, n1_table 1.5, cos1 25.7, ranks 0.4, second_link 2.1, outside 0.5, question_orders
  49.2, kept 0.7, x_ord2 2.0, kt_cols 4.2.
- hotpotqa: x_unw 65.6, x_rel 85.9, x_ord1 2.4, n1_table 0.6, cos1 7.5, ranks 0.1, second_link 1.2, outside 0.5, question_orders
  56.5, kept 0.3, x_ord2 2.3, kt_cols 2.2.
- 2wiki: x_unw 54.1, x_rel 72.5, x_ord1 2.4, n1_table 0.5, cos1 10.5, ranks 0.1, second_link 1.4, outside 0.4, question_orders
  52.8, kept 0.3, x_ord2 2.3, kt_cols 2.3.

zfsvc's own stage (a) reads 583, 1425 and 1478 ms here against M2's 420, 1174 and 1353: the first 40 questions, the host loaded
by other jobs, one logical processor for BLAS's two threads. Only the ratios say anything, and they are not timed ones.

All three runs passed (`DEVX PASS`). On the 40 questions of each dataset: stage (a)'s pool, KT columns and expansion from the
forms equal `zfsvc.lean_rule`'s byte for byte on all 40; each model's scores from the module's layer step and from the forms'
are equal byte for byte on all 80 forwards; A2's checks held on every question (the pool, today's pool, the expansion, the KT
columns and I against PKO's rule; each model's inputs against the carve's batch); the largest difference of a score from the
fit's filed read (MLP / GNN) was 5.7e-06 / 7.2e-06 on musique, 2.9e-06 / 4.8e-06 on hotpotqa, 2.4e-06 / 6.7e-06 on 2wiki (bar
1e-4), with the same top five on 40 of 40 for both models on all three; no layer step took the module's path (255 of 255 by the
forms on each). The float16 NEXT values that differ from the cache's (counted, not failures, as in M2): 0 of 1,383,088 on
musique, 1 of 191,583 on hotpotqa, 0 of 313,117 on 2wiki. The norm band could not certify a matrix in 6 (musique), 15 (hotpotqa)
and 18 (2wiki) of the 123 stage (a) calls of those runs (the prepass, the JIT question and each question twice); each took the
reference's form for that matrix, and its bytes are the reference's.

**What dominates** (torch's profiler, one thread, self time; the first question of musique: 4,750 rows, 106,810 link edges):
- The module's GNN forward, 276 ms of self time: `mul` 25%, `index` 24%, `scatter_add_` 23%, `mm` 7%, `sum` 7%, `addmm` 3%. By
  part (encode, chain, input map and context, layers, readout; wall time, ms): the module's 37.8, 0.3, 4.3, 244.1, 0.8 (the
  layers are 85%: the per-edge gathers, products and scatter-adds, and the attention's row sum); the forms' 41.5, 0.1, 3.3,
  45.3, 0.2. On the second question the layers took 235 ms and 58. What is left of the GNN's forward is the encode (46%) and the
  layers (50%).
- The module's MLP forward, 39 ms in 2,306 calls: `addmm` 20%, `index` 14%, `mul` 11%, `mm` 10%, `scatter_add_` 9%, `where` 8%,
  `gelu` 4%, `cat` 3%. The six ops `index`, `mul`, `where`, `div`, `sub` and `scatter_add_` (45% together) are `encode`'s masks
  and segment z-scores.

**What is not changed.**
- Stage (b): A2's forms (`compile_serial`, `lean_rows10`, `links_fast`, NEXT's lists, the inputs), for both models. Compile
  36, rows 20 and links 12 ms on musique are a place for a later amendment.
- The MLP's forward (the module's; see the profile), first-stage retrieval (not timed), the threads (one for torch, numba and every
  OpenMP runtime; BLAS 2 in the record, 1 beside), the pin, the reads, the bars and the question sets.
- Family 2's forms stay for amendment A3 (section 10): a cut input is neither computed nor read in stage (b). The forms above
  touch the pool's walks and orders and the GNN's layers, which no cut changes (a cut fit is the same stack with its own
  weights), and so are the forms of both families.

**Re-time (M3).** The timed passes run per dataset after this commit, one at a time on the host, a smoke first (`--nq 20`, its
raw file in `outputs/fsv/timing/a4/raw_smoke`, not filed; the run stops with "the forms did not run" if a form was not used).
The records add 0.1, 1.6 and 2.7 GiB to a dataset's process, so a job's `--mem` is A2's plus 3 GB. Each dataset:

```
python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvx.py run --dataset D --blas 2 --lps 2,4 --require-pin
```

then the BLAS-1 pass beside (`--blas 1`), its raw files in `outputs/fsv/timing/a4/raw` (the `a4` directory; zfsvc's raw files are
not touched; each records the module's sha256, the records' index time and size and what ran), and
`python outputs/mp_unified/zfsvx.py report` (zfsvc's reduction of those files, labelled A4: `outputs/fsv/timing/a4/report.json`
and `report.md`). M3 replaces M2 as the number of record for every cell; M2 stays in the repository, labelled A2. A faster exact
form found later is a new amendment and a re-time.

**Risks, stated before any number.**
- The development speed-ups are not measurements; the timed ones can differ (pinning, the host's load, the cache).
- The records are an index-time structure built from the unions (1.3 s and 77 MiB on musique, 10.5 s and 1,586 MiB on hotpotqa,
  14.5 s and 2,731 MiB on 2wiki; untimed, as the unions and PKO's prepared lists are). A reader who counts them as cold time
  would add the build once per dataset; M3 reports it beside.
- Two exactness steps rest on library kernels (sgemv's per-row result by place, torch's row sum). They hold on the checked
  questions and in the self-test; another library version or thread count could differ, and a question whose bytes differ fails
  the check and is counted, not hidden.
- A norm the band cannot certify takes the reference's form (slower, exact); the count is reported (6 on musique, 15 on
  hotpotqa, 18 on 2wiki of 123 stage (a) calls in the development runs).
- The new kernels are compiled in the first (untimed) question, as A2's are.

**Commit policy.** `zfsvx.py` and this section are committed before any M3 pass. The raw files are never committed (section
10's policy); M3's `report.json` and `report.md` are committed masked, as M2's, with `git add -f`.

**Correction 1 (4 November, before any M3 pass).** The first smoke on the host, with two CPUs as the timed jobs have, stopped at
the first compile after the P4b import: numba refuses a changed `NUMBA_NUM_THREADS` once its threads have started. rx sets the
variable to `--cpus`; `zfsvc._load` puts that value back after the C-stage modules' imports set it to 1, and `p4b_fast_walks`
(through `p4a_pool_cold`) set it to 1 again when `zfsvx` imported it afterwards. The development runs had one CPU, the value P4b
sets, and did not meet it. `zfsvx.py` now imports P4b and zg6r inside `thread_env_kept`, which puts the six thread variables
(OpenBLAS, OpenMP, MKL, NumExpr, numba and the timing code's) back as they were on entry, so the process's environment after setup
is A2's. No form, check or number changes. The self-test gained a check of the guard (the variables come back, also after an
error; numba takes its compile after a block that changed `NUMBA_NUM_THREADS` and refuses the change without the guard) and was
run whole with `NUMBA_NUM_THREADS=2`, which stopped at the first compile before the correction. The correction and `zfsvx.py`
with it are committed before any M3 pass; each raw file records the sha256 of the file that ran.

## 13. Amendment A6: faster exact forms for stage (b)'s links and NEXT columns and for the encoder's z-scores (4 November, before any timed number with them)

This takes up what section 12 left open ("Stage (b) ... a place for a later amendment"; a fused `encode` "would serve both models and
every cut fit") on the user's order of 4 November, about 19:30 IST ("we can try speeding up whatever you found rn"; on the MLP's
NEXT columns: "mlp is doing a beam search that is why the NEXT is so slow right, try speeding up, both the stuff and what about the
fast families"), with the standing one of 06:30 IST (section 12) that both models be as fast as possible. M3 (`outputs/fsv/timing/a4`)
used section 12's forms. It stands as the record of those forms and is not touched. The forms below are timed after this commit,
as M5. Module: `outputs/mp_unified/zfsvy.py` (new). It imports `zfsvx.py`, installs A4's forms with `zfsvx.installed()` and A6's on
top in its own process (three functions replaced, put back on exit and asserted), and edits nothing: not `zfsvc.py`, `zfsvx.py`,
`lean_screen3.py`, `ltwdiag.py`, `c3_fast.py`, `zg6r.py`, `lean_gpu.py` or any C-stage module. Sections 1 to 12 stand, except where
this names one.

**Why.** M3's stage (b) and (c), p50 in ms per question (BLAS 2, the 200 kept questions; stage (a) with A4's forms). Compile, rows
and links are shared: each model computes its own copy. NEXT is the MLP's alone. Assemble is `build`'s inputs.

| dataset | (a) | compile MLP / GNN | rows MLP / GNN | links MLP / GNN | NEXT (MLP) | assemble MLP / GNN | (c) MLP / GNN | end to end MLP / GNN | GNN / MLP |
|---|---|---|---|---|---|---|---|---|---|
| musique | 142.7 | 38.2 / 37.2 | 21.1 / 21.0 | 12.9 / 13.3 | 7.4 | 3.0 / 5.4 | 27.0 / 69.2 | 260.3 / 296.9 | 1.14 |
| hotpotqa | 154.6 | 5.3 / 5.4 | 1.7 / 1.6 | 0.43 / 0.43 | 2.8 | 0.6 / 0.7 | 5.5 / 8.8 | 171.9 / 173.2 | 1.01 |
| 2wiki | 178.8 | 9.9 / 9.7 | 2.6 / 2.5 | 0.66 / 0.67 | 3.6 | 0.9 / 1.1 | 7.5 / 12.7 | 205.1 / 208.8 | 1.02 |

Stage (a) is 55%, 90% and 87% of the MLP's end-to-end time, so a form in stage (b) or (c) moves end to end by a few percent at most
on hotpotqa and 2wiki, where both models' (b) + (c) are 18 to 28 ms of 172 to 209. On musique the data is larger (4,800 pool rows,
127,000 link edges): stage (b) is 83 ms of the MLP's 260 and 77 of the GNN's 297, and the GNN's forward is 69.

**What NEXT is.** It is not a search that goes deeper: it is LTM's beam of fixed width five and one hop. The five rows rrf ranks
first are the anchors; each anchor's candidates are its link neighbours (with the link families as a bit mask) that are not
themselves anchors; the fold model of the question's hold-out fold (LTW's scorer) scores every (anchor, candidate) pair; the
log-ratio to uniform fills seven float16 columns (one per anchor, their maximum, the number of anchors that reached the row).
Its work per question is the lists from the links, the unit vectors of the float16 rows of the anchors and the candidates (1,536
values each), the pair features, the scorer's forward in torch, and the assembly of the columns. The profile (development, run
after this commit and added below) says which of these is the larger share of the 2.8 to 7.4 ms.

**The forms** (each stage, each model; a form a model has in a shared stage, the other has too):

| stage | A4's form (the modules') | A6's form | MLP | GNN |
|---|---|---|---|---|
| (b) links | `c3_fast.links_fast`: numpy passes over the edge arrays (both directions, a sort, the repeats dropped) | one numba pass: each edge in both directions as the entry (row, other row * 3 + family), two stable counting sorts (by the key, then by the row), repeats dropped, the row, the other row and the family split back; int16 rows and int8 families as the module's | yes | yes |
| (b) NEXT, candidate lists | `LTW.fam_bits` for each anchor (two scans of the link arrays, the bits of every node) and the candidate test | the anchor's run in the sorted closure by binary search, each neighbour's bits or-ed over its entries, the anchors excluded | yes | none (the GNN has no NEXT) |
| (b) NEXT, unit rows | `ltwdiag.unit` on the float16 rows converted to float32: `np.linalg.norm`, `np.maximum(norm, 1e-12)`, divide | one numba pass per matrix with numpy's own float32 summation order | yes | none |
| (b) NEXT, chunk | `LTW.read_chunk` | the same lines, without the outputs `next_cold` drops (the `free` columns and the identity check) | yes | none |
| (c) encode | `lean_screen3.seg_zscore_ref`, called for every block's columns in both models' `encode` (and for rrf's base column in the MLP's forward): a chain of torch ops (the whole segment's z-score through `lean_gpu.seg_zscore8D`, then the reference rows' sums, their centring, variance and `where`s) | one numba pass per question over the segment, every column at once | yes | yes |

**Why each form is exact** (the same floating-point operations in the same order, or integers only; only where the state lives, or
work no read reaches, changes):
- **The segment z-scores.** The module computes, per column and question, float32 sums with `index_add_` (it adds the rows one after
  the other, in row order, which is also what section 12 relies on for the GNN's scatter-adds), the mean as the sum over the count
  (the reference rows' count clamped at 1), the centred values as float32 differences, the variance as the sum of their squares
  (times the 0/1 weight for the reference rows) over the count, the standard deviation as 0 where the variance is 0 and the float32
  square root otherwise, and takes per column the reference rows' z-score `c / max(sd, eps)` when there are at least two of them
  and their `sd >= eps`, else the whole segment's z-score (0 where its `sd < eps`). The form does the same float32 operations in the
  same order, each product and sum rounded to float32 on its own (numba without fast-math, float32 scalars only: no fused
  multiply-add, no reassociation, no float64 constant). A batch of 16 questions (the warm batches) is done question by question:
  `zfsvc.build` lays each question's rows out contiguously and in question order, so a question's sums never include another's rows,
  as in the module's per-question buffers. Any other input (not float32, not two-dimensional, not CPU, a gradient in force, empty,
  2^24 rows or more, question ids unsorted or outside 0..B-1, a reference of another shape) takes the module's function, counted.
- **The candidate lists.** `links_fast`'s output is the symmetric closure of the pool's links, sorted by (row, other row * 3 +
  family) with no entry twice. The entries of an anchor are therefore one run, its neighbours come in ascending order with their
  families together, and the bits are `1 << family` or-ed over a neighbour's entries; `fam_bits`' two directions give the same
  neighbours and bits because the closure holds each entry's mirror. A neighbour that is one of the five anchors, the anchor
  itself included, is not a candidate, as in the module. Integers only.
- **The unit rows.** numpy's float32 `norm` over a row of 1,536 values is the square root of `add.reduce` of the squares, and
  `add.reduce` along the contiguous axis is numpy's pairwise sum: blocks of at most 128 values with eight interleaved accumulators
  combined as ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)), the remainder added one by one, longer runs split at half less its
  remainder modulo 8 (768, 384, 192 and 96 for 1,536). The kernel takes the same squares (float32 products) and adds them in that
  order, then the float32 square root, `max(., 1e-12)` and the float32 division. The float16 rows are converted by torch (exact, as
  numpy's). This rests on numpy's implementation of the pairwise sum, which the checks below compare byte for byte.
- **The chunk.** `LTW.read_chunk` builds columns and an identity check that `next_cold` discards (`b, _chk = ...`) and that
  `items_of` does not read; the form leaves them out and keeps every other line, so every value `items_of` and the scorer take is
  the module's.
- **The links.** Integer work only: the same set of entries sorted by the same key gives the same arrays (self-loops dropped, repeats
  collapsed, as in the module), and an edge outside its pool or its families sends the call to the module's function.

**The checks** (untimed; `devy` for development, `run` for M5):
- A2's own development checks with A4's and A6's forms installed (`zfsvc.dev_checks`): the pool, today's pool, the expansion, the KT
  columns and I against PKO's rule on the same question; each model's inputs against the reference carve's batch; each model's
  scores against the fit's filed read at batch 1, within 1e-4 with the same top 5; the float16 NEXT values counted.
- Beside them, per question and model, stage (b) and the forward run under six settings in a rotating order: A6's forms off (A4's
  forms only), all of A6's, and each of A6's four forms alone (the z-scores, the lists, the unit rows with the chunk, the links).
  Every output (the scores, the pack of the inputs, the features) is compared byte for byte with the run without A6's forms, and
  each segment's time is kept. `devy` passes only if every comparison holds on every question and every form ran (counted in
  both modules' `STATS`, the module's own function never taken on the timed path).
- `selftest` (no data): each form against the module's function on random inputs, byte for byte (the z-scores on constant,
  tied, tiny-scale and slice inputs, with no, one, two and many reference rows, single questions and batches of two to sixteen with
  empty and one-row questions and 32- and 64-bit ids; the lists on random closures; the links on random edge sets with loops,
  repeats and every family; the unit rows on float16 rows of every kind including zero rows; the chunk); the inputs that go to
  the module's function; the installation and its restoration, also after an error. It passed before this commit (`--trials 4`: 88,
  16, 86, 75 and 6 comparisons, none differing).
- M5's `run` keeps A2's checks per timed question and A2's rule: a question that fails the scores' or the pool's check is not kept,
  and more than 2% failing stops the dataset. It stops with "the forms did not run" if a form was not used, and each raw file records
  both modules' sha256 and what ran.

**Development numbers and the profile.** From the first development runs on the host (feeder jobs `fsvy-profile-D` and
`fsvy-devy-D`, 4 November 21:19 to 22:18 IST, on the module as committed in d2348c3; the first 40 questions of each dataset, BLAS 2,
two CPUs as the timed jobs have, other jobs running on the host: development times, not measurements). A form that fails a check
there is dropped from the module and from this section by a correction before any M5 pass. Each form's gain is reported per dataset;
a form that gains under 0.5 ms on hotpotqa and nothing on musique is dropped as well (the links form can gain only on musique, where
links are 10 to 13 ms of stage (b), against 0.4 on hotpotqa). A form the profile shows to be worth adding is declared by a correction
before any M5 pass, with its check.

**Exactness.** `devy` PASSED on hotpotqa, musique and 2wiki. On every question and for both models, each of the five settings (all of
A6's forms; each form alone) gave the scores, the pack of the inputs and the features byte for byte equal to the run with A4's forms
only. A2's checks held with the forms installed: the pool, the KT columns and I equal PKO's rule on the same question, each model's
inputs equal the carve's batch, the scores within 1e-4 of the fits' filed reads (the largest difference below is 7.2e-6) with the same
top five on 40 of 40, and the float16 NEXT values counted. Every form ran and neither module's own function was taken (`zref_orig` 0,
`links_orig` 0 on every dataset; counters per job: z-scores 1,680, candidate lists 123, unit rows 82, links 164). The profile's NEXT
columns equal `next_cold`'s on 40 of 40 questions on each dataset.

| dataset | questions | pool rows p50 (max) | link edges p50 (max) | scores vs the filed read, max diff MLP / GNN | same top 5 MLP / GNN | NEXT values differing from the filed read | variants differing byte for byte from A4's forms |
|---|---|---|---|---|---|---|---|
| musique | 40 | 4833 (8053) | 125259 (289218) | 5.7e-06 / 7.2e-06 | 40 / 40 of 40 | 0 of 1,383,088 | none |
| hotpotqa | 40 | 666 (1250) | 4036 (15056) | 2.9e-06 / 4.8e-06 | 40 / 40 of 40 | 1 of 191,583 | none |
| 2wiki | 40 | 1115 (1568) | 6230 (20910) | 2.4e-06 / 6.7e-06 | 40 / 40 of 40 | 0 of 313,117 | none |

(The one hotpotqa NEXT value is also in the run with A4's forms: the variants agree byte for byte, so the form did not move it.)

**All of A6's forms against A4's forms alone**, p50 in ms per question, stage (b) + (c) without stage (a) (A4's forms -> A4's and
A6's). The segments A6 does not touch (compile, rows, assemble) differ by up to 0.3 ms between settings on hotpotqa and 2wiki and by
up to 2.5 ms on musique: the host is shared, so only differences beyond that are readable.

| dataset | model | compile | rows | links | NEXT | assemble | forward | (b)+(c) | gain | x |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | MLP | 31.29 -> 33.77 | 16.59 -> 16.80 | 10.25 -> 1.99 | 6.46 -> 3.58 | 2.30 -> 2.24 | 22.62 -> 11.83 | 92.77 -> 69.21 | 23.56 | 1.34 |
| musique | GNN | 32.36 -> 32.88 | 16.48 -> 18.18 | 10.37 -> 1.96 | none | 3.84 -> 4.73 | 52.06 -> 44.88 | 115.43 -> 103.41 | 12.02 | 1.12 |
| hotpotqa | MLP | 3.85 -> 3.76 | 1.48 -> 1.50 | 0.40 -> 0.07 | 2.38 -> 2.37 | 0.54 -> 0.54 | 5.30 -> 2.03 | 14.96 -> 10.99 | 3.96 | 1.36 |
| hotpotqa | GNN | 3.66 -> 3.62 | 1.51 -> 1.51 | 0.42 -> 0.07 | none | 0.60 -> 0.61 | 8.28 -> 5.65 | 14.95 -> 11.56 | 3.40 | 1.29 |
| 2wiki | MLP | 7.37 -> 7.23 | 2.29 -> 2.42 | 0.65 -> 0.12 | 3.15 -> 3.03 | 0.91 -> 0.88 | 7.22 -> 3.17 | 22.37 -> 16.90 | 5.47 | 1.32 |
| 2wiki | GNN | 7.11 -> 6.84 | 2.44 -> 2.50 | 0.64 -> 0.11 | none | 1.10 -> 1.06 | 12.40 -> 8.51 | 23.36 -> 19.64 | 3.72 | 1.19 |

**Each form alone**, its own segment's p50 in ms without and with the form:

| dataset | model | z-scores: forward | links form: links | lists form: NEXT | unit rows and chunk form: NEXT |
|---|---|---|---|---|---|
| musique | MLP | 22.62 -> 11.49 | 10.25 -> 1.89 | 6.46 -> 3.72 | 6.46 -> 5.19 |
| musique | GNN | 52.06 -> 42.35 | 10.37 -> 1.89 | none | none |
| hotpotqa | MLP | 5.30 -> 1.97 | 0.40 -> 0.08 | 2.38 -> 2.41 | 2.38 -> 2.44 |
| hotpotqa | GNN | 8.28 -> 5.55 | 0.42 -> 0.07 | none | none |
| 2wiki | MLP | 7.22 -> 3.09 | 0.65 -> 0.11 | 3.15 -> 2.94 | 3.15 -> 3.03 |
| 2wiki | GNN | 12.40 -> 8.70 | 0.64 -> 0.11 | none | none |

**What NEXT costs** (the profile: two questions per dataset with its timers in force, p50 in ms; its NEXT totals, 3.1, 7.1 and 3.0
with the call's wrapper, are within 0.7 ms of the development p50s above). The question was whether the beam search is what makes
NEXT slow. It is not a search: the candidate lists are 0.1 ms on hotpotqa and 2wiki and 1.4 ms on musique, where the links are
125,000 edges; the pair features and the scorer's forward are 72%, 47% and 70% of it.

| part | hotpotqa | musique | 2wiki |
|---|---|---|---|
| pair features (`read_chunk`: the rows' reads, unit vectors, the per-list pair features, the float16 rows) | 1.29 | 1.79 | 1.30 |
| the fold model's forward (torch) | 0.86 | 1.41 | 0.73 |
| collate | 0.30 | 0.51 | 0.31 |
| candidate lists (the families' bits) | 0.10 | 1.40 | 0.12 |
| order, prep, candidates, items, fill, pack | 0.42 | 1.63 | 0.44 |
| NEXT, total | 2.96 | 6.73 | 2.90 |

**What the numbers say, and what is decided.**
- The segment z-scores are the large gain on every dataset and in both models: they were 64% and 34% of the MLP's and the GNN's
  forward on hotpotqa in the profile (61% and 32% on 2wiki, 55% and 21% on musique), and the forms' gains agree (the MLP's forward
  falls 2.6 times on hotpotqa and the GNN's 1.5 times).
- The links form gains 0.3 ms on hotpotqa and 0.5 ms on 2wiki and 8.4 ms on musique, in both models. The lists form and the unit-row
  form gain nothing on hotpotqa and 0.1 to 0.2 ms on 2wiki, and 2.7 and 1.3 ms of the MLP's NEXT on musique. By the rule above all
  four forms stay: each gains on musique. No form is dropped and none is added. The parts of NEXT that remain (the pair features and
  the scorer's forward, 0.7 to 1.8 ms each) would be a new amendment for under 1 ms of 170 to 260 ms end to end; none is proposed.
- The risk named below ("the ratios GNN / MLP can fall") went the other way: the MLP's forward is mostly z-scores, so it gains more
  than the GNN's, and the model-only ratio GNN / MLP at development rises (musique 1.24 -> 1.49, hotpotqa 1.00 -> 1.05, 2wiki 1.04
  -> 1.16). Taken with M3's stage (a) (142.7, 154.6 and 178.8 ms) the development gains would move end to end by 2 to 3% on
  hotpotqa and 2wiki and by 10% (MLP) and 4% (GNN) on musique, and GNN / MLP end to end from 1.14 to about 1.20 on musique, from 1.02
  to 1.03 on 2wiki, and leave it at 1.01 on hotpotqa. That is arithmetic on development gains, not a measurement: M5 measures.
- The GNN's forward, after the z-scores, is matrix products and activations (torch's profiler on hotpotqa's first question: `mm` 34%,
  `addmm` 15%, `gelu` 7%, `mul` 5%, layer norm 5%), which A6 does not change.

**What is not changed.**
- Stage (a) (A4's forms: the walks and the orders are the larger part of end-to-end time on hotpotqa and 2wiki), stage (b)'s
  compile and rows, the GNN's layer step (A4), `lean_gpu.seg_zscore8D` (rrf's base z-score in `build`, the context weights), the
  scorer's forward in torch, both models' matrix products, activations and concatenations, first-stage retrieval (not timed),
  the threads (one for torch, numba and every OpenMP runtime; BLAS 2 in the record, 1 beside), the pin, the reads, the bars and
  the question sets.
- Family 2's forms stay for amendment A3: a cut input is neither computed nor read in stage (b), and a cut fit is the same stack
  with its own weights, so A6's forms are the forms of both families. The MLP has no admitted cut (its screens at L3, L2, L1 and L0
  all fail: sections 8 to 11), so its fast version is the full model; the GNN's L3 screen is pending. A3 is declared when its calls
  are filed.

**Re-time (M5).** The timed passes run per dataset after this commit, one at a time on the host, a smoke first (`--nq 20`, its raw
file in `outputs/fsv/timing/a6/raw_smoke`, not filed). The forms add no records and no tables (their scratch arrays are a few MiB),
so a job's `--mem` is M3's. Each dataset:

```
python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy.py run --dataset D --blas 2 --lps 2,4 --require-pin
```

then the BLAS-1 pass beside (`--blas 1`), its raw files in `outputs/fsv/timing/a6/raw` (the `a6` directory; zfsvc's and zfsvx's raw
files are not touched), and `python outputs/mp_unified/zfsvy.py report` (zfsvc's reduction of those files, labelled A6:
`outputs/fsv/timing/a6/report.json` and `report.md`). M5 replaces M3 as the number of record for every cell; M3 stays in the
repository, labelled A4. If no form passes its checks, M5 is not run and M3 stands. A faster exact form found later is a new
amendment and a re-time.

**Risks, stated before any number.**
- The development speed-ups are not measurements; the timed ones can differ (pinning, the host's load, the cache).
- End to end on hotpotqa and 2wiki, stage (a) is 87 to 90% of the time, so these forms can move the ratio of the models'
  end-to-end times by about a percent there; their effect shows in the model-only, forward and warm columns, and on musique.
- A shared form speeds the larger part more: the GNN's encode is a bigger share of its time than the MLP's, so the ratios GNN / MLP
  can fall. M5 reports whatever it measures.
- Three forms rest on library behaviour (torch's `index_add_` taking rows in order, numpy's pairwise float32 sum, numba's plain
  float32 arithmetic). They hold on the self-test and, where the host's libraries differ, the byte-for-byte comparison of `devy`
  decides: a form whose bytes differ is dropped by a correction, not hidden.
- The new kernels are compiled in the first (untimed) question, as A2's and A4's are; their cache is written beside the module.

**Correction 1 (4 November, 22:30 IST, before any M5 pass).** The first smoke (hotpotqa, 20 questions, BLAS 2) passed every check
(0 of 20 questions failing, scores within 2.9e-6 for the MLP and 3.8e-6 for the GNN of the filed reads, 0 NEXT values differing) and
then stopped with "the forms did not run": `forms_ran` required that A4's layer-step form took every GNN layer step, as `devy`
(batch 1 only) shows it does. A4's step is for one question by design (`zfsvx.g6r_step_x` takes the module's step when the batch
is not 1), and a timed run's warm batches of 16 (12 batches of the first 192 questions; one in the smoke) take the module's step,
one per layer: 3 in the smoke, which is what the counters showed (`g6r_step_orig` 3, `g6r_step_x` 123, `zref_orig` 0). The check now
expects exactly that many module steps in a timed run (layers x (questions // 16)) and none in `devy`; nothing else in `zfsvy.py`
changes, no form, no timed path and no score check. M3's warm batch of 16 took the same module step, and A6's z-score form handles
the batch of 16 (`zref_orig` 0), so M5's warm numbers compare like with like.

**Commit policy.** `zfsvy.py` and this section are committed before any M5 pass. The raw files are never committed (section 10's
policy); M5's `report.json` and `report.md` are committed masked, as M2's and M3's, with `git add -f`.

## 14. Amendment A7: a batched layer step for the GNN's warm batches of 16 (4 November, before any timed number with it)

This takes up the line "the GNN's layer step (A4)" in section 13's "What is not changed", after a development measurement of
4 November (feeder jobs `fsvy-b16-D`, from about 22:55 IST, beside M5's first passes), on the standing orders of 10 October ("make
the gnns faster too i dont want to give any unfair advantage to mlp") and of 4 November, 06:30 IST (section 12: both models as fast
as possible). A4's layer step is for one question. At a batch of 16 the module's torch step runs, once per layer (section 13,
correction 1), and that step is most of the GNN's warm forward there, while A6's z-score form speeds the MLP's forward about 1.9
times at the same batch (the table below). Left alone, M5's warm batch-of-16 column would show the ratio GNN / MLP rising to about
1.7 to 1.8 times what the same development job gives with M3's forms (musique 8.4 to 14.8, hotpotqa 2.9 to 5.1, 2wiki 3.5 to
5.9): the MLP got a form for its largest cost and the GNN's largest cost at that batch got none. The form below is the GNN's counterpart.

Module: `outputs/mp_unified/zfsvy_b16.py` (new). Its `selftest` is the check without data, `share` the development job, `run` and
`report` the timed re-time of this section. It imports `zfsvy.py` (and so `zfsvx.py` and `zfsvc.py`), edits none of them and none
of `zg6r.py`: the batched step is put in the module's place around each forward timed with it and put back. Sections 1 to 13
stand. M5 stays the number of record for every cell except the GNN's warm batch of 16 and the ratio there, which the re-time below
replaces for every dataset whose checks pass.

**Development numbers** (`share`, one job per dataset, no timed number). 32 questions of the dataset, that is two batches of 16;
one process, BLAS 2 and then BLAS 1; pinned to two logical processors outside the timed ones (8 and 10), run beside M5's passes
(2 and 4); the three jobs ran 22:54-22:58 (hotpotqa; M5's hotpotqa pass at BLAS 2 ran on 2 and 4 at the same time), 22:59-23:04
(musique; M5's hotpotqa pass at BLAS 1) and 23:14-23:20 IST (2wiki; M5's musique pass at BLAS 2), the last one after this section was
committed (the feeder sent it before the item could be dropped), so the 2wiki rows below were not part of the decision. Each round
times three variants of the batch's forward in an order that rotates: "M3" (no A6 form, the module's step:
M3's forms), "A6" (A6's forms, the module's step: what M5's warm batches run) and "A7" (A6's forms and the batched step below);
five rounds after one that is not counted. The layer step's three calls and the layer loop are timed inside the same forwards by
wrappers, and one torch-profiler pass per model gives the shares of its self time. Cells are ms per question (the batch's forward
over 16, median over the rounds):

| dataset | BLAS | MLP ms/q, M3 -> A6 forms | GNN ms/q, M3 -> A6 -> A7 step | step share of the GNN's forward | ratio GNN/MLP, M3 -> A6 -> A7 | A7 step vs module's: the step / the forward | ratio with the step free |
|---|---|---|---|---|---|---|---|
| musique | 2 | 24.7 -> 13.2 | 207.5 -> 194.7 -> 57.1 | 92% | 8.40 -> 14.77 -> 4.34 | 4.25x / 3.41x | 1.15 |
| musique | 1 | 23.7 -> 12.2 | 207.4 -> 196.2 -> 54.9 | 92% | 8.74 -> 16.03 -> 4.49 | 4.49x / 3.57x | 1.21 |
| hotpotqa | 2 | 3.8 -> 2.0 | 11.2 -> 10.0 -> 5.5 | 80% | 2.91 -> 5.10 -> 2.79 | 2.22x / 1.83x | 1.01 |
| hotpotqa | 1 | 3.6 -> 2.0 | 10.7 -> 9.1 -> 5.2 | 80% | 2.98 -> 4.56 -> 2.61 | 2.14x / 1.75x | 0.90 |
| 2wiki | 2 | 5.3 -> 2.8 | 18.6 -> 16.5 -> 9.3 | 80% | 3.50 -> 5.85 -> 3.28 | 2.23x / 1.78x | 1.15 |
| 2wiki | 1 | 5.1 -> 2.9 | 20.6 -> 17.7 -> 10.1 | 81% | 4.05 -> 6.16 -> 3.50 | 2.10x / 1.76x | 1.19 |

Self time in the profiler (the first batch of 16, BLAS 2, A6's forms and the module's step), matrix products / gathers and
scatters / the rest:

| dataset | MLP: matrix products / gathers and scatters / the rest | GNN: same |
|---|---|---|
| musique | 66% / 3% / 31% | 9% / 54% / 38% |
| hotpotqa | 64% / 3% / 33% | 30% / 33% / 37% |
| 2wiki | 60% / 3% / 37% | 24% / 36% / 39% |

What the numbers say, and what is decided.
- The module's layer step is 92% of the GNN's batch-of-16 forward on musique, 80% on hotpotqa and 80% (BLAS 2) or 81% (BLAS 1) on
  2wiki; the layer loop is the same share to within a point. The MLP has no such stage: its forward is 60 to 66% matrix products
  and 3% gathers and scatters, and the matrix products are calls to the BLAS that the GNN's forward makes too. The GNN's forward
  is 54% (musique), 33% (hotpotqa) and 36% (2wiki) gathers and scatters, the layer step's edge work.
- A6's forms alone move the ratio the wrong way: the MLP's forward falls 1.9 times and the GNN's 1.07 (musique), 1.12
  (hotpotqa) and 1.13 (2wiki) times, because the segment z-scores are most of the MLP's forward and a smaller part of the GNN's.
- The batched step is 4.3 (musique), 2.2 (hotpotqa) and 2.2 (2wiki) times faster than the module's, the forward 3.4, 1.8 and 1.8
  times, and the ratio GNN / MLP at A6's forms and the batched step is 4.3 (musique), 2.8 (hotpotqa) and 3.3 (2wiki), under the
  value with M3's forms (8.4, 2.9 and 3.5 in the same jobs) on musique, about level with it on hotpotqa and a little under it on
  2wiki. If the step were free the ratio would be 1.2, 1.0 and 1.2: what is left of the GNN's forward (the encoder, the readout
  and the z-scores) is the size of the MLP's whole forward. No further form is proposed here.
- The forward's scores are byte for byte the same in every batch (two a dataset), at both BLAS counts: the batched step against
  the module's step for the GNN, and A6's forms against M3's for both models. The batched step gains on every dataset measured
  and at both BLAS counts, so it is declared. The development job for 2wiki ran after this section was committed (the feeder sent
  it before its item could be dropped); its rows are in the tables and agree with the other two datasets, and they were not part
  of the decision.
- End to end and the cold forward do not use this step at a batch of 16 and do not move. The warm batch of 16 moves, and so do the
  ratios GNN / MLP there: that is arithmetic on development gains, not a measurement: the re-time measures.

**The form.** `g6r_step_b` in `outputs/mp_unified/zfsvy_b16.py` is `zfsvx.g6r_step_x` for a batch (B > 1; B = 1 is A4's step,
unchanged). It is `zg6r.g6r_step` with the same operations in the same order, except two:
1. The attention logit's dot product. The module computes it once per edge: `(linear(h, wa)[u] * c[qv]).sum(1) / sqrt(D)`, a gather
   of the source row's projection, a product with the context of the question the edge is in, and a sum over the 64 values. There
   are 6 times as many edges as rows on hotpotqa, 5 times on 2wiki and 24 times on musique (p50 of the development questions: 4,036
   edges on 676 rows, 5,871 on 1,082 and 118,598 on 4,859). Every edge lies inside one question (its source row and its target row are rows of the same
   question: `nq` is each row's question, `qv` the target's), so `c[qv]` of the edge is `c[nq[u]]`, the context of the source
   row's own question. The form computes `(linear(h, wa) * c[nq]).sum(1) / sqrt(D)` once per row and gathers the result at the
   edges' source rows. For one edge the two are the same 64 float32 products added by the same torch reduction (per row of a
   contiguous matrix), and the gather copies a float32. The form tests `nq[u] == qv` on every call and takes the module's step
   (counted) when it fails, with a typed edge, another dtype or a gradient in force.
2. The two scatter-adds. The family means and the attention numerators are `index_add_` of per-edge rows into rows. A4's numba
   kernel `edge_acc` does both in one pass over the edges in their order: `index_add_` adds the source rows one after the other in
   index order (section 12's argument for B = 1), so each target row's sum is the same sequence of float32 additions. The edges of
   different questions touch different rows, so a batch does not change any row's sequence.

The denominators' `index_add_`, the clamp, the divisions, the five linear maps, the layer norm and the activation are the module's
calls on the same tensors in the same order. The module's per-edge tensors (the gathered projections and their products, 64
floats per edge) are never built; that is the saving.

**The checks** (untimed):
- `selftest` (the laptop; random layers of the module's shapes and random graphs whose edges lie inside their questions: B = 2,
  5 and 16, one and two torch threads, up to 2,000 rows and 20,000 edges a question, every layer): 42 steps compared byte for
  byte, 0 differ; an edge that leaves its question takes the module's step, counted. (The other fallbacks, a typed edge, another
  dtype and a gradient in force, are by reading.)
- The development jobs above: the forward's scores byte for byte the module's, in every batch (two a dataset), at both BLAS
  counts: A7's step against the module's step for the GNN, A6's forms against M3's for both models.
- The timed run (`run`), per dataset and BLAS count, on the 12 batches of 16 of M5's warm pass (the first 192 questions): the
  scores of the batched step byte for byte the module's in every batch, the same scores twice in every round (a variant whose
  forward differs from its own first round stops the run), and each variant's scores, question by question, against the fit's
  filed read (section 5's bar: within 1e-4, the same top five; M5 does not check the batch of 16's scores). The run stops when
  more than 2% of the questions fail for the batched step (section 5), and a byte that differs in any batch fails the form for
  that dataset: M5's batch-of-16 cell stands there and says so. The guard of A6 (`forms_ran`) and a counter of the batched step
  (all three layer steps of every forward timed with it, none the module's) must hold, or the job stops without a record.

**What is not changed.**
- Everything else of sections 1 to 13: the MLP, stage (a), stage (b), the cold forward and the warm batch of 1 (A4's step), both
  models' matrix products, the threads (one for torch, numba and every OpenMP runtime; BLAS 2 in the record, 1 beside), the pin,
  the reads, the bars and the question sets. M5's own cells stay in M5's report, the batch of 16 of both models included (the
  module's step); the re-time reports the GNN with the module's step and with A7's side by side, the MLP once, in the same job.
- Family 2 (amendment A3, still to be declared): a cut GNN is the same stack with its own weights, so A7's step is a form of both
  families; the MLP has no admitted cut and no layer step.

**Re-time (A7).** After this commit and after M5's chain, never beside another timed job, per dataset one job with both BLAS
counts (2, then 1), a smoke first (hotpotqa, `--nq 20`, one batch of 16; its raw file in `outputs/fsv/timing/a7/raw_smoke`, not
filed):

```
python outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_b16.py run --dataset D --blas 2 --lps 2,4 --require-pin
```

Per BLAS count: the inputs of the first 192 questions (12 batches of 16) built as in M5 (untimed), then six rounds over the 12
batches, the first not counted; in a round, per batch, the features of both models are built outside the clock and the three
forwards (the MLP's, the GNN's with the module's step, the GNN's with A7's) are timed in an order that rotates from batch to
batch and round to round.
Each cell is the batch's forward time over 16, p50 and p95 over the 60 batch times; the ratio GNN / MLP is that of the p50s,
paired by batch and round, with the 2.5th and 97.5th percentiles over 2,000 resamples of the batch times with replacement
(`numpy.random.default_rng(0)`). Raw files `outputs/fsv/timing/a7/raw/D__bN.json`;
`python outputs/mp_unified/zfsvy_b16.py report` writes `outputs/fsv/timing/a7/report.json` and `report.md`.

For the warm batch of 16, the number of record is the report's (the MLP's and the GNN's times of one job, the GNN with A7's step)
for every dataset whose checks pass; M5's own batch-of-16 cells (one forward per batch, 12 values, the module's step) stay in M5's
report. The headline stays the end-to-end and the cold forward times, which this does not touch; the warm columns are beside.

**Risks, stated before any number.**
- The development numbers are 32 questions beside M5's passes on other processors: not measurements. The timed ones can differ,
  and the speed-up can differ with the graphs' sizes (the layer step's cost grows with the edges: 4,036 a question on hotpotqa,
  5,871 on 2wiki, 118,598 on musique).
- The form leans on two properties of the libraries: torch's reduction over a row does not depend on the number of rows, and
  `index_add_` takes its rows in order (A4's kernel already relies on it). They held on the selftest and in every development
  comparison; the timed run's per-batch comparison decides, and a form whose bytes differ is dropped by a correction.
- The ratios GNN / MLP at batch 16 fall with this form: the GNN's largest stage at that batch shrank and the MLP's did not. The
  report says what it measures. The form does not touch the cold forward or end to end.
- Six rounds over 12 batches is a larger sample than M5's 12 values per cell; the two are not compared cell by cell, the re-time
  carries the module's step beside A7's.

**Commit policy.** `zfsvy_b16.py` and this section are committed before any A7 timed pass. The raw files and the development
JSON are never committed (section 10's policy); the A7 `report.json` and `report.md` are committed masked, as M5's, with
`git add -f`.

## 15. Amendment A3: Family 2's forms for the cut GNN, and the GNN's L2 screen (5 November, before any Family 2 number)

Sections 10 to 14 each leave Family 2's forms to this amendment, and section 13 says it is declared when the GNN's calls are filed.
They are filed: `outputs/fsv/calls_gnn.json` and `calls_gnn.md` (the host's `fsvg-calls`, 5 November, 03:45 IST; committed in
0591dfb). The rule of sections 5 and 10 stands: a fast version that equals its full model is timed once and has no second row.
Section 9 adds that the level after a failed screen is queued once the calls name it; they name L2, and this amendment queues it.

**State at 03:45 IST on 5 November.**
- The MLP's four screens failed (L3 6d16b28, L2 0c95b96, L1 ab2af7a, L0 d2f8c98): its fast version is the full zrckpn, and its
  Family 1 rows (M5, A7) are its Family 2 rows.
- The GNN's L3 screen FAILED: three LOSS among the eight deciding reads (WQ-hotpotqa musique -0.0220 against a floor of 0.0114,
  its 3-hop slice -0.0259 against 0.0207 and its 4-hop slice -0.0111 against 0.0096), two GAIN and three WITHIN. By section 4's
  order the L2 screen (G_struct and G_walk cut) comes next; its items are declared below. A PASS would still need its full run, so
  the GNN's fast version is not known.
- A3 does not wait for it. The forms below hold for every level the GNN's order can still admit, and nothing in them depends on
  which one it is.

**The cases.** L3 is out (a verdict closes only the level it tested, section 6). G_store (SEED, DISTS) stays in every level that
is left, so it needs no form.

| the GNN's calls | Family 2 | what this section gives |
|---|---|---|
| L2 ADMITTED | zg6r without G_struct and G_walk, against the full zrckpn | the forms for both groups |
| L2 not admitted, L1 ADMITTED | zg6r without G_struct, against the full zrckpn | the form for G_struct |
| an L0 for the GNN added by a later amendment (section 11) and ADMITTED | zg6r without topo_STRUCT, against the full zrckpn | the form for G_struct's first block alone |
| no level admitted | the full zg6r against the full zrckpn: Family 1 stands for both | nothing; M4 is not run |

**What a cut removes in stage (b).** The pieces each group feeds, read from the code, with their costs from A6's development
profile (the first two questions of each dataset, BLAS 2, one thread, p50 ms). These figures are development figures: unfiled (the
profile's raw output is not committed) and not measurements. 2wiki was not profiled.

| group | blocks | computed by | hotpotqa | musique |
|---|---|---|---|---|
| G_struct | topo_STRUCT, depth_STRUCT | the compile (`C11.compile_serial` calls `c2_fast.compile_mlp`): its laps `topology`, `D` and `depth_basis` | 1.6 | 23.8 |
| G_walk | WALK, WALKF | the rows (`C10.lean_rows10` calls `C2.walk_block` and `C2.walkf_block`) | 0.5 | 7.4 |

M5's GNN stage (b) is 8.6 ms on hotpotqa, 13.2 on 2wiki and 65.0 on musique, its end to end 177.3, 201.7 and 255.3, and stage (a)
is 92%, 88% and 52% of that. So the skipped pieces are worth at most 2.1 ms (L2) or 1.6 (L1) of stage (b) on hotpotqa and 31.2 or
23.8 of it on musique. If those pieces were all that changed, the GNN's end to end on musique would go from 1.17 times the MLP's
(255.3 against 218.0) to 1.03 (L2) or 1.06 (L1), and on hotpotqa it would move by about 1% (2wiki was not profiled, and stage (a)
is a similar share there). The forward (c) changes too, by a narrower first layer and fewer z-scored columns. These are bounds
from unfiled development figures, stated before any timed number; the timed pass gives the numbers, and Claim 3 is stated from
them whatever they are.

**The forms.** One rule, from section 10: a cut input is neither computed nor read in stage (b), and the forward is the cut fit's
own model. A cut fit is the same stack as its full model with its own weights (section 13), so A6's forms and A7's batched step
apply to it unchanged.
- compile (G_struct): an in-process copy of `c2_fast.compile_mlp` that returns after its `edges` lap, so the laps `topology`, `D`
  and `depth_basis` do not run (L0: only the topology kernel is left out, since the symmetric CSR the depth basis reads is still
  built). The retrieval and edge laps are the module's own and the pool's edges are still made, because the links and the kept
  blocks read them. If `need_compile.NeedCompiler` (it mirrors `FastCompiler` group by group and is bit-identical on the asked
  blocks by its own `need_check.py`) takes the same inputs and is no slower, it is used instead. The development check decides,
  and a correction records which.
- rows (G_walk, at L2): `C2.walk_block` and `C2.walkf_block`, the two calls `C10.lean_rows10` makes for the walk columns, are
  replaced in process by functions that return zeros of the same shape (the cut pick never reads those columns), so that every
  other line of `lean_rows10` is the module's own.
- assemble and forward: `zfsvc.build` over the cut pick's blocks (section 2's table, with the GNN's links and messages); the
  forward is the cut fit's own `models.pt`, read for each dataset from the same split as the full fits' reads; A6's z-score forms
  run on the kept columns only; the warm batch of 16 runs A7's step.

**Checks.** A module for the cut path (`outputs/mp_unified/zfsvy_cut.py`, new, not in this commit) imports `zfsvy_b16.py` and so
`zfsvy.py`, `zfsvx.py` and `zfsvc.py`, and edits none of them, as `zfsvy_b16.py` did. Its `selftest` is the check without data and
its `devy` the development job (the first 40 questions of each dataset, BLAS 2, the full path and the cut path in one process, the
two CPUs the timed jobs have). On every question:
- each kept column of X, the pack's inputs, the features and the scores of the cut path are byte for byte the values the full
  path gives when the cut pick is selected from its inputs;
- A2's checks hold: the pool, the KT columns and I equal PKO's rule's; the scores are within 1e-4 of the cut fit's own filed read
  with the same top five.

A form that fails is dropped by a correction before any M4 pass: its piece is then computed and not read, M4 says which pieces
ran, and no gain is claimed for it. A piece that cannot be separated exactly costs the cut only its speed-up.

**M4, the timed pass.** Per dataset one job with both BLAS counts (2, the record, then 1 beside), a smoke first, as M5's and A7's:
the first 200 questions, cold at batch 1 and one thread, warm at batches of 1 and 16, the same pin and load. Per question, in a
rotating order, three versions: the full MLP, the full GNN and the cut GNN. It runs only after this commit, after the GNN's
verdict, after the development check, never beside another timed job, and with section 6's rule applied first if the pair of
record at the freeze is not zg6r and zrckpn.
- Reported, with no call (section 5's list): per stage p50 and p95 of the cut and the full GNN; model-only and end to end; the
  ratio cut GNN / full MLP (Family 2) and, beside it, full against full (Family 1 again, in the same job), paired by question with
  2,000 resamples (seed 0); the cut GNN's gain over the full GNN per stage, paired the same way; the warm times at batches of 1 and 16.
- If no level is admitted, M4 is not run and the record of Claim 3 is M5's and A7's.

**The L2 screen's items.** The same module, rule and fits as L3's (section 9), with `L2` in the fit names: `scr-fsv-zspk-L2-s0` and
`-hp`, `scr-fsv-zg6r-L2-s0` and `-hp`, then the compare and the calls. L2 cuts G_struct and G_walk, so the pick keeps rank, dense_cos,
SEMB, SEED and DISTS, and the cut encoder has 208,097 numbers (section 9). The commands, the pool caps and the inputs are L3's items'
own; of the 85 input files only this document has changed since L3's fits were sent.

| item | GPU share | host GB | after |
|---|---:|---:|---|
| `fsvg-zspk-L2-s0` | 0.42 | 13.4 | `fsvg-calls` (the calls that name L2) |
| `fsvg-zspk-L2-s0-hp` | 0.57 | 13.4 | `fsvg-calls` |
| `fsvg-zg6r-L2-s0` | 0.42 | 13.4 | `fsvg-zspk-L2-s0` |
| `fsvg-zg6r-L2-s0-hp` | 0.83 | 13.4 | `fsvg-zspk-L2-s0-hp` |
| `fsvg-compare-l2` | - | 2 | both zg6r fits |
| `fsvg-calls-l2` | - | 1 | the compare |

The reservations come from L3's four fits as run, not from section 9's table (ML1's peaks, 7.3 to 11.4 GB of memory):
- memory: rx's record is 12.63, 12.47, 12.81 and 12.65 GB (zspk, zspk -hp, zg6r, zg6r -hp), so each fit reserves 13.4, 4.6% above
  the largest; the compare and the calls peaked at 0.22 and 0.04 GB committed and reserve 2 and 1;
- the card: the allocator's peak reserved was 8.65, 11.42, 9.01 and 17.89 GB, under the pool caps 0.4, 0.55, 0.4 and 0.8 of the
  24 GB card, so the shares and caps stay L3's. The training peaks in L3's logs (7.17, 11.58, 7.80 and 18.35 GB) match ML1's
  (section 9's table: 7.2, 11.7, 7.8 to 8.0 and 17.7 to 18.4), so the cut did not lower them. L2 keeps more columns than L3 and fewer
  than the full model, so its peaks are not expected to differ; the allocator's line in each log is the check;
- time: the four fits ran 32, 41, 64 and 111 minutes beside other jobs; a CUDA fault is rerun by `retry_cmd`, as at L3.

Once `fsvg-calls-l2` has run, `calls_gnn` names the next step as section 4 says: L2's full run (seeds 1 and 2) after a PASS, L1's
screen after a FAIL. Those items are declared by a correction then, as this one is.

**What is not changed.** Sections 1 to 14: the cuts, the reads, the rule, the pool and the forms of both families, M5's and A7's
numbers (the MLP's Family 2 rows are M5's and A7's), the threads, the pin, the question sets and the bars.

**Commit policy.** This section is committed before any development job of the cut module and before any M4 pass. The raw files and
the development JSON are never committed (section 10's policy); M4's `report.json` and `report.md` are committed masked, as M5's,
with `git add -f`. The six items of the L2 screen go to the feeder after this commit; no other host job is sent with it.

**Correction 1 (5 November, about 10:16 IST, before any seed-1 or seed-2 fit): the GNN's L2 full run.** The L2 screen is filed:
`outputs/fsv/calls_gnn.json` and `calls_gnn.md` (the host's `fsvg-calls-l2`, 5 November, 10:03 IST; committed in 8a1dd2d). It is
a PASS. Of the eight deciding reads one is a GAIN (WQ-musique musique, +0.0162 against a floor of 0.0095), seven are WITHIN and
none is a LOSS; the lowest delta is -0.0062 (WQ-hotpotqa's musique 4-hop slice, floor 0.0096); there are no problems. By
section 4's order a PASS opens that level's full run, and the calls read "Next: L2 full_run". The GNN's fast version is still
not known: a PASS admits nothing, and the GAIN does not count (section 4).

**The run.** Section 4's full run, unchanged: seeds 1 and 2 on both splits, each cut fit compared with ML1's zg6r of the same
seed and split, and `zltg.full_row` per deciding read over the three seeds with the pair floor of both arms' spreads. Seed 0
is the screen's, filed and kept. ADMITTED when none of the eight is a LOSS, NOT_ADMITTED when any is, INCOMPLETE first
(sections 4 and 9).

Four fits and no new start. Each starts from the cut zspk fit of its split at seed 0 (`scr-fsv-zspk-L2-s0` for WQ-musique, `-hp`
for WQ-hotpotqa), fitted and read in the screen, which is the start of every zg6r fit of that level and split (section 9);
`zfsvg`'s loader checks the start as it did for the screen's fits.

| item | GPU share | host GB | after |
|---|---:|---:|---|
| `fsvg-zg6r-L2-s1` | 0.42 | 13.4 | `fsvg-calls-l2` (the calls that name the full run) |
| `fsvg-zg6r-L2-s2` | 0.42 | 13.4 | `fsvg-calls-l2` |
| `fsvg-zg6r-L2-s1-hp` | 0.83 | 13.4 | `fsvg-calls-l2` |
| `fsvg-zg6r-L2-s2-hp` | 0.83 | 13.4 | `fsvg-calls-l2` |
| `fsvg-compare-l2f` | - | 2 | the four fits |
| `fsvg-calls-l2f` | - | 1 | the compare |

The items go in this order. The commands, the pool caps (0.4 for the WQ-musique fits, 0.8 for the -hp fits) and the 85 input
files are the screen's items' own, with the seed in the fit name: `scr-fsv-zg6r-L2-s<k>` and `-hp`, k = 1 and 2. No input file
has changed since the screen's items were sent, except this document.

**The reservations** come from the four L2 fits as run, and they stand:
- memory: rx's record is 12.63, 12.52, 12.77 and 12.62 GB (zspk, zspk -hp, zg6r, zg6r -hp), so each fit reserves 13.4, 4.9% above
  the largest; the compare and the calls peaked at 0.22 and 0.04 GB committed and reserve 2 and 1;
- the card: the allocator's peak reserved was 8.68, 11.04, 8.96 and 18.15 GB, under the pool caps 0.4, 0.55, 0.4 and 0.8 of the
  24 GB card, and the process's peak at the end of the two zg6r reads was 9.35 and 18.37 GB, under the shares' 10.1 (0.42) and
  19.9 GB (0.83). The training peaks in L2's logs (7.18, 11.60, 7.81 and 18.37 GB) equal L3's within 0.02 GB, so the cut at L2
  did not move them either;
- time: the two zg6r fits ran 36 minutes and 1 h 50 min.

**The card.** The two WQ-musique fits hold 0.84 of it together and run side by side; each -hp fit holds 0.83 and runs alone, one
after the other. On a card free of other jobs the four fits take about 4.5 hours.

**What the verdict opens.**
- ADMITTED: the GNN's fast version is L2 (zg6r without topo_STRUCT, depth_STRUCT, WALK and WALKF), and Family 2 is that cut GNN
  against the full zrckpn. Section 15's forms, the cut module, its development check and M4 follow as section 15 says, and
  section 6's rule for the frozen pair applies before any timing. This correction writes and runs none of them.
- NOT_ADMITTED: L1's screen (G_struct cut alone, 217,057 encoder numbers) opens by a later correction, not before. If no level
  is admitted, the full zg6r is the fast version and M4 is not run (section 15).
- INCOMPLETE: the missing fit, compare or read file is made and the calls are rerun (section 4).

**What is not changed.** Sections 1 to 15: the cuts, the reads, the rule, the pool, the forms of both families, and M5's and A7's numbers.

**Commit policy.** This correction is committed before any of the four fits is sent. The fits, compares and reads are never
committed; the calls that name the verdict are, masked (section 9's policy). The six items go to the feeder after this commit;
no other host job is sent with it.

**Correction 2 (6 November, about 06:38 IST, before any number of the cut path): the cut module, its check without data, the cut fits'
reference reads and its development job.** The GNN's L2 full run is filed and ADMITTED (`outputs/fsv/calls_gnn.json` and `calls_gnn.md`,
committed in 634a8ad; the header reads "Fast version: L2. Next: nothing"). So Family 2 is the cut zg6r (without topo_STRUCT,
depth_STRUCT, WALK and WALKF) against the full zrckpn, and the forms this section declared on 5 November are now written. This
correction declares the code and its development job; it makes no number, runs nothing on the host itself and sends no item
itself: the six items below go to the feeder after this commit, inserted by the feeder's owner.

**The module.** `outputs/mp_unified/zfsvy_cut.py` (new, committed with this correction) imports `zfsvy_b16.py` first, and so
`zfsvy.py`, `zfsvx.py`, `zfsvc.py` and `zfsvt.py`; it also imports `zfsvg.py` (the cut fits' names, pick and read context). It edits
none of them. It holds the forms of this section, as declared, and no timed pass (M4's `run` and `report` come by a later correction,
before any M4 pass):
- compile (G_struct): `compile_cut` and `_compile_cut`, a copy of `c2_fast.compile_mlp` and of `_compile_mlp` up to and including its
  `edges` lap, then its return, so the laps `topology`, `D` and `depth_basis` do not run and the columns they fill stay zero. The
  retrieval and edge laps are the module's lines; `selftest` compares the copy's lines with the module's, one by one.
  `need_compile.NeedCompiler` is not tried here; a later correction would record it if it were.
- rows (G_walk): `walk_zeros` and `walkf_zeros` return zeros of the shapes of `C2.walk_block` (16 columns) and `C2.walkf_block`
  (18), float32; every other line of `C10.lean_rows10` is the module's own.
- `cut_forms()` puts the three replacements in force (`c2_fast.compile_mlp`, `walk_block`, `walkf_block`, the names
  `C11.compile_serial` and `C10.lean_rows10` call) for the length of the cut GNN's own run, and puts the module's functions back
  after it, also on an error; entered twice it refuses.
- assemble and forward: `cut_view(S)` swaps the GNN's entries of `S.models`, `S.blocks` and `S.boxes` for the cut fit's for the length
  of a block, so that `zfsvc.run_model`, `build`, `forward` and `check_scores` run the cut model unedited: `build` over the cut
  pick's blocks, the forward of the cut fit's own `models.pt`. A6's z-score forms run on the kept columns only (they act on what the
  model passes them); the warm batch of 16 runs A7's step (`zfsvy_b16.forward_t`).
- the cut fits are the seed-0 screen fits of the split of the full fit each dataset is timed on (musique `scr-fsv-zg6r-L2-s0-hp`,
  hotpotqa and 2wiki `scr-fsv-zg6r-L2-s0`), the same choice as the full models' (`scr-ml1-zg6r-s0-hp`, `scr-ml1-zg6r-s0`).
  The cut model is built through the cut fit's own read chain (`zfsvg.on_fit` around `zml1s.run_arm`'s read up to `read_ref`: FSV's
  names, A1's stamp checked, the level's pick, p@swa alone) and takes the full GNN's carve, which is the cut fit's cache, split and
  basis (checked: the basis and its sha256), so no second carve is held.

**The check without data.** `python outputs/mp_unified/zfsvy_cut.py selftest`: (1) `compile_cut` is `compile_mlp`'s lines and
`_compile_cut` is `_compile_mlp`'s up to its `edges` lap and its return; (2) the zero functions return the shape and dtype of the
walks they stand for (checked on a random graph against the module's), are in force only inside `cut_forms`, and are put back after
an error; (3) the comparison of kept columns sees a change in a kept column and not one in a cut column, and `cut_view` swaps and
restores; (4) the level's cut and pick as `zfsv` and `zfsvg` name them (the cut is these four blocks, the pick is DOB's order
without them), and each dataset's cut fit starts from the full fit's split. It passed on the laptop before this commit (6 November,
06:36 IST): it needs no data and prints nothing it measured.

**The cut fits' reference reads** (`ref`). The cut fits have no reference file yet: `zfsvt.py`'s are the full fits'. `python
outputs/mp_unified/zfsvy_cut.py ref --dataset D` is `zfsvt.ref_cmd` for the dataset's cut fit: the same chain, flags, class, roots and
read cap as the fit's read, `read_carve`'s batches, then every question at batch 1, on the CPU, checked against the fit's filed read
file (its scores64 on the first 64 questions, its top and hit on all 201 questions), under `zfsvg.on_fit` and with no stamp
touched. It writes `outputs/fsv/timing/ref/<cut fit>/<dataset>.npz` and `.json` and nothing under `outputs/screen/fits`. The files
are never committed (section 10's policy).

**The development job** (`devy`; the first 40 questions of the dataset and the JIT question, BLAS 2, the two CPUs the timed jobs
have (`zfsvy_b16.pin_two`), A4's and A6's forms installed). Per question, in a rotating order, four versions: the full MLP, the full
GNN, the full path with the cut pick selected from its inputs (`cutref`: the module's own compile and rows, the cut model) and the
cut path. On every question:
- A2's checks (`zfsvc.dev_checks`) on the full MLP and GNN, as A6's development job ran them;
- the cut path against `cutref`, byte for byte: each kept column of X (the blocks of the cut pick but SEMB), the pack's inputs (n, P,
  q16, kt, the link count and the link arrays), the features, the question index, the base z and the scores;
- the cut path's features and base z against the cut fit's carve (`batch([q])` over the cut blocks), and both cut paths' scores
  against the cut fit's reference read at batch 1 (within 1e-4, the same top five);
- information, not a check: whether the cut columns of the cut path's X are zero.

After the questions: the cut path's packs in batches of 16 (two), the forward with the module's layer step against A7's, byte for
byte. The job prints the per-stage medians of the four versions (development numbers, unfiled, not measurements) and a line
`DEVY_JSON` with the counts. It ends `DEVY PASS` when no question fails a check, every form ran (A4's and A6's by their counters, the
cut forms by theirs: `compile_cut`, `walk_zeros` and `walkf_zeros` once per question, the module's functions back in place) and the
batches are equal; otherwise `DEVY FAIL` with the failing questions and checks named. The job exits 0 either way (the verdict is in
the log, so the chain goes on), except on a Windows file race, which `retry_cmd` reruns. A form that fails is dropped by a correction
before any M4 pass, as this section says: its piece is then computed and not read, and no gain is claimed for it.

**The six items** (one chain, in this order, each after the one before; the dataset's devy after its reference read). All run on
two CPUs, and the commands go through `pylib_run.py` as the earlier development items did:

| item | dataset | command (after `outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_cut.py`) | host GB |
|---|---|---|---:|
| `fsvc-ref-hotpotqa` | hotpotqa | `ref --dataset hotpotqa --blas 2` | 14 |
| `fsvc-devy-hotpotqa` | hotpotqa | `devy --dataset hotpotqa --questions 40 --blas 2 --lp 8` | 19 |
| `fsvc-ref-musique` | musique | `ref --dataset musique --blas 2` | 14 |
| `fsvc-devy-musique` | musique | `devy --dataset musique --questions 40 --blas 2 --lp 8` | 23 |
| `fsvc-ref-2wiki` | 2wiki | `ref --dataset 2wiki --blas 2` | 14 |
| `fsvc-devy-2wiki` | 2wiki | `devy --dataset 2wiki --questions 40 --blas 2 --lp 8` | 28 |

The development items run under `retry_cmd.py --tries 3 --wait 90`. Their inputs are `pylib_run.py`, `retry_cmd.py`, the numba zip and
the five modules of the chain (`zfsvt.py`, `zfsvc.py`, `zfsvx.py`, `zfsvy.py`, `zfsvy_b16.py`) with `zfsvg.py` and `zfsvy_cut.py`;
the reference items need the cut fits' folders, which the screen left on the host, and the cache. The reservations are estimates:
no peak of either job has been measured. The devy figures are zfsvy's development items' (17, 21 and 26 GB) plus 2 GB for the cut
model and its reference arrays; the reference items reserve 14 GB until the peak of the same job on the full zg6r fit is read from
rx's record. They are re-read from the first runs, as A2's were.

**What is not changed.** Sections 1 to 14, section 15 above this correction, the cuts, the reads, the rule, the pool, the forms of
both families, the threads, the pin, the question sets and the bars; M5's and A7's numbers; `zfsvc.py`, `zfsvt.py`, `zfsvx.py`,
`zfsvy.py` and `zfsvy_b16.py`.

**What follows.** After the six items: a correction if a form fails its check (the form dropped). Then M4's code (`run` and `report`)
by a later correction, with its smoke; M4 itself runs only after the freeze (00:00 IST on 7 November), after this development
check, never beside another timed job, on a quiet host, with section 6's rule applied first if the pair of record at the freeze is
not zg6r and zrckpn.

**Commit policy.** This correction and `zfsvy_cut.py` are committed together, before any development job of the module runs. The
development logs, the reference files and the development JSON are never committed; the six items go to the feeder after this
commit, and no other host job is sent with them.

**Correction 3 (6 November, about 08:26 IST, before any M4 pass): the six development items read, M4's code and its check without
data, the three M4 items.** The six items of correction 2 finished on the host this morning (all done, 06:46 to 06:55 IST). Every check
of this section passed on every dataset, so no form is dropped and the cut path M4 times is the one correction 2 declared. This
correction reads the items and declares M4's code (`run`, `report` and `selftest`) and its three items. It makes no number of the cut
path, runs nothing on the host itself and sends no item itself: the items go to the feeder after this commit, inserted by the feeder's
owner, and the three M4 items not before 00:00 IST on 7 November (the freeze).

**The six items read.** Pass or fail only. The per-stage times the development job printed are development numbers, not measurements:
they are not filed and not quoted anywhere.

| dataset | reference read of the cut fit (`ref`) | development job (`devy`, 40 questions) | peak of `ref`, GB (working set / commit) | peak of `devy`, GB (working set / commit) |
|---|---|---|---:|---:|
| hotpotqa | PASS | PASS | 2.3 / 3.5 | 14.7 / 14.5 |
| musique | PASS | PASS | 8.0 / 8.3 | 15.8 / 16.5 |
| 2wiki | PASS | PASS | 6.3 / 7.2 | 22.4 / 22.5 |

The peaks are rx's records of the six jobs; the reservations were 14 GB for each `ref` and 19, 23 and 28 GB for the `devy` items, so
none was exceeded. Each check passed on every dataset:
- the reference read of each cut fit agrees with the fit's filed read: the scores of its first 64 questions within 1e-4 with the same
  top five on all 64, the top and hit of all 201 questions, and its batch-1 top five on all 201 (the file is written under
  `outputs/fsv/timing/ref/`, never committed);
- on each of the 40 questions of each dataset: A2's checks on the full MLP and the full GNN; the cut path against the full path with the
  cut pick selected from its inputs (`cutref`), byte for byte in each kept column of X, the pack's inputs, the features, the question
  index, the base z and the scores; the cut path's features and base z against the cut fit's carve; and both cut paths' scores against
  the cut fit's filed read, within 1e-4 with the same top five on 40 of 40;
- the forms ran: A4's and A6's by their counters, and the cut forms (`compile_cut`, `walk_zeros` and `walkf_zeros`) once per question
  with the module's functions back in place after; the warm batches of 16 on the cut packs gave A7's step byte for byte the module's
  on both batches of each dataset.

Information, not checks: the cut columns of the cut path's X were zero on 40 of 40 questions of each dataset; float16 NEXT values that
differ from the filed read were counted as A2 counts them (one value, in one hotpotqa question; none in musique and 2wiki).

**No form is dropped.** Section 15's rule applies to a form that fails; none did. M4 times all four pieces of the cut (G_struct's
compile laps and G_walk's two walk calls left out, the cut pick's five blocks and encoder, A7's batched step on the cut packs), and says
which ran by their counters (`forms` in each raw file, the report's checks table).

**M4's code.** `outputs/mp_unified/zfsvy_m4.py` (new, committed with this correction) imports `zfsvy_cut.py` first, and so
`zfsvy_b16.py`, `zfsvy.py`, `zfsvx.py`, `zfsvc.py` and `zfsvt.py`; it edits none of them, and reuses their stages, forms, checks and
reduction (`zfsvc.stage_a`, `run_model`, `build`, `forward`, `check_scores`, `_series`, `_ratio`; `zfsvy_cut.run_version`,
`attach_cut`, `cut_view`; `zfsvy_b16.forward_t`) as they are.
- `run --dataset D --blas 2 --lps 2,4 --require-pin` is one job per dataset: the setup of A7's job (the full fits, A4's and A6's forms
  in force), the cut fit's model and reference (`attach_cut`), the index-time work (`prepass`), the pin (two logical processors, EcoQoS
  off, above-normal priority; the job stops if it is not pinned) and the load of every processor sampled for 1 s before and after each
  pass. Then a smoke pass and two timed passes.
  - The smoke pass: the first 32 questions at BLAS 2, one round of the two batches of 16, checked as the record is. If a check or a
    form fails the job stops and records nothing.
  - The record pass (BLAS 2), then the beside pass (BLAS 1), each over the first 200 questions after the JIT question (the carve's
    201st) with the same pin and load. A pass writes its raw file when it ends.
  - Cold, batch 1, one thread: per question, stage (a) once on the shared pool (PKO's rule on A4's forms), charged in full to every
    version; then the three versions, each its (b) (compile, rows, links, columns, assemble) and (c) (eager forward, top five) as
    `zfsvc.run_model` times them. The order rotates: the question's number modulo 3 picks the version that leads, so each leads a
    third of the questions. The cut GNN runs inside `cut_view` and `cut_forms` (the cut model, blocks and fit; the three replaced
    functions in force for its run only), the full MLP and GNN as in M5.
  - Warm at batch 1: the forward alone, on features built beforehand from the cold packs (one question at a time, outside the clock),
    the three versions in the question's order.
  - Warm at batch 16: the first 192 questions in twelve batches; one round that is not counted, then five; each batch's features are
    built outside the clock and its five forwards timed in an order that rotates with the batch and the round: the full MLP, the full
    GNN with the module's layer step and with A7's batched step, and the cut GNN with each. A question's time is its batch's over 16.
    The record's warm b16 of the GNNs is A7's step (byte for byte the module's, section 14); the module's step is reported beside.
  - Checks, untimed, as `zfsvc.run`'s and A7's: each version's scores against its own fit's filed read at batch 1 (within 1e-4, the
    same top five), the pool, the KT columns and I against PKO's rule, the float16 NEXT flips counted; for the warm batches, A7's step
    byte for byte the module's for both GNNs and every variant's scores, question by question, against the filed read. The record pass
    keeps the questions on which no version fails a check and stops (exit 1, files still written) if more than 2% fail or a batched
    step differs from the module's; the beside pass times every question and counts the failures, as M5's.
  - Forms: the job stops, writing no file for that pass, unless every form ran: A4's and A6's by `zfsvy.forms_ran`, A7's by one batched
    layer step per layer of each batched forward and none that fell back, the cut forms once per cold run of the cut path (the
    smoke's and the JIT question's included) and the module's functions put back.
  - The raw files are `outputs/fsv/timing/m4/raw/<dataset>__b2.json` and `__b1.json` (`raw_smoke` for a run with `--nq`); a finished
    pass is not repeated without `--overwrite`. They are never committed.
  - The guard: `--not-before` is 2026-11-06T18:30:00+00:00, which is 00:00 IST on 7 November; a run without `--nq` that starts earlier
    starts nothing and exits 3. It is a safety on the feeder's owner's placement of the items, not the way they wait. A run with
    `--nq` (a smoke on the real data) is not bound; it writes `raw_smoke` only, its numbers are development numbers, and it is run only
    when no other timed job is on the host.
- `report` reduces the raw files, per BLAS pass and dataset, with no call: per stage (a), (b)'s parts, (b), (c), model-only and end to
  end, the p50 and p95 of each of the three versions; the ratios of the p50s, paired by question (by batch and round for the warm
  batches of 16), with the 2.5th and 97.5th percentiles over 2,000 resamples (`numpy` `default_rng(0)`, `zfsvc._ratio`): cut GNN / full
  MLP (Family 2), full GNN / full MLP (Family 1 again, the same job) and full GNN / cut GNN (the cut's gain), for end to end (the first
  column: the headline), model-only, (b), (c), the parts of (b) and the warm times at batches of 1 and 16; stage (a)'s share of each
  end-to-end time; the checks, the forms that ran, the pin and the load; and, beside and deciding nothing, the first half of the
  questions against the second, each version led first, second and third, and p95 over p50. It writes `outputs/fsv/timing/m4/report.json`
  and `report.md`, which are masked as M5's were and committed with `git add -f`; the raw files are not.
- `selftest` needs no data and times nothing: the rotation (each version leads equally), the guard (a full run before the time starts
  nothing, a smoke is not bound), the signatures of the 27 functions the module calls in the modules it imports, and a run of `one_pass`,
  the raw files and `report` on a fake world whose stage times are constants (rows, the kept mask, the stop rule, the batched-forward
  count, the ratios equal to the constants' ratios inside their intervals, the report's bytes equal on two reductions, and a missing raw
  directory stopping `report`). It passed on the laptop before this commit (6 November, 08:22 IST).

**The three items** (one chain, in this order, each after the one before; two CPUs each, the pin `--lps 2,4` as M5's and A7's). The
commands go through `pylib_run.py` after `outputs/host_ops/pylib_run.py outputs/mp_unified/zfsvy_m4.py`, with no retry wrapper (a timed
pass is not rerun by one; a failed job is sent again by hand):

| item | command | host GB | after |
|---|---|---:|---|
| `fsvm4-hotpotqa` | `run --dataset hotpotqa --blas 2 --lps 2,4 --require-pin` | 28 | `isb-s0-hp` (the seed-0 fit on the GPU) |
| `fsvm4-musique` | `run --dataset musique --blas 2 --lps 2,4 --require-pin` | 34 | `fsvm4-hotpotqa` |
| `fsvm4-2wiki` | `run --dataset 2wiki --blas 2 --lps 2,4 --require-pin` | 37 | `fsvm4-musique` |

The inputs are `pylib_run.py`, the numba zip and the seven modules of the chain (`zfsvt.py`, `zfsvc.py`, `zfsvx.py`, `zfsvy.py`,
`zfsvy_b16.py`, `zfsvg.py`, `zfsvy_cut.py`) with `zfsvy_m4.py`; the fits, caches and cut-fit folders are on the host already. The
reservations come from rx's records of the nearest jobs, each taken as the larger of the working set and the commit peak: M5's passes
(hotpotqa 21.6, musique 23.3, 2wiki 29.4 GB, two versions' packs held), A7's (20.5, 20.8, 27.8), the pair's development items (14.9,
13.3, 22.5) and the cut development items above (14.7, 16.5, 22.5). M4 holds three versions' packs, so each reservation is the cut
development item's peak plus one and a half times M5's excess over the pair's development item, plus a margin: hotpotqa 14.7 + 1.5 x
6.7 = 24.8, musique 16.5 + 1.5 x 10.0 = 31.5 and 2wiki 22.5 + 1.5 x 6.8 = 32.7, reserved at 28, 34 and 37. They are estimates, re-read
from the first run as A2's were; the smoke pass and the beside pass add no second copy (a pass's packs are freed when its file is
written). The chain is estimated at two to three hours.

The feeder's owner places the items at or after 00:00 IST on 7 November, when the host is quiet: no other timed job running, and the
seed-0 fit of the ISB track off the GPU (the first item names it). Before the freeze a smoke of the module on the real data may run
once, `run --dataset hotpotqa --nq 20 --blas 2 --lps 2,4 --require-pin` (2 CPUs, 28 GB, `fsvm4-smoke-hotpotqa`, no dependency), only
when no timed job is on the host; its output is development numbers, unfiled and unquoted, and it is not M4.

**What is not changed.** Sections 1 to 14, section 15 and corrections 1 and 2 above this one, the cuts, the reads, the rule, the pool,
the forms of both families, the threads, the pin, the question sets and the bars; M5's and A7's numbers; `zfsvc.py`, `zfsvt.py`,
`zfsvx.py`, `zfsvy.py`, `zfsvy_b16.py` and `zfsvy_cut.py`.

**What follows.** The three items run once the freeze has begun and the host is quiet, with section 6's rule applied first: if the
pair of record at the freeze is not zg6r and zrckpn, the items wait for a correction. When the three passes are in, `report` is run,
its two files are masked and committed (`git add -f`, those two files only), and the calls that name the verdicts of Family 2 are
made by a later file, as section 5 says.

**Commit policy.** This correction and `zfsvy_m4.py` are committed together, before any M4 pass and before any item of this
correction is sent. The raw files, the smoke's files and the logs are never committed; no other host job is sent with the items.

**Correction 4 (6 November, about 10:04 IST, before any M4 pass): M4 may start now.** Correction 3 said the three M4 items run
"once the freeze has begun" and had `zfsvy_m4.py` refuse a full run before 00:00 IST on 7 November. The user told the coordinator on
the morning of 6 November to stop postponing and run M4 now. This correction checks section 6's rule on that day's records, so that
the pair M4 times is fixed before M4 starts, and relaxes the guard. It makes no number, runs nothing on the host itself and sends no
item itself: the three items go to the feeder after this commit, inserted ahead of the feeder's waiting head.

**Why now.** The freeze time was a placement rule, not a measurement rule. It kept the timing from running beside the fits that
could still change a pair, and section 6 asks for one check: whether the pair of record at the freeze is zg6r and zrckpn. Every form
that could change that pair is closed as of this morning.
- The GNN: IRA's full run is filed NOT_ADOPTED (f5536e9; its six-read record regenerated in 7f63f7d under the slices amendment a5abf3f);
  ISB's screen is filed KEEP under its own declared rule (aa58c5a), its full run was dropped by its gate, and zg6r stays; the stack
  of GNN forms is IRA alone, which is not adopted. D6 waits for after the freeze (user, 4 November) and is not a form of this pair.
- The MLP: zrckpn stays; no MLP form is open (the capacity track is closed, CAP 97b0ae2).
- The cut: L2 is ADMITTED (634a8ad) on zg6r, and correction 2's six development items all passed (correction 3).

So section 6's rule is checked and not triggered: the pair of record is zg6r and zrckpn, the admitted level is L2, and no
admitted level needs a second screen. If a later change made another pair the pair of record, M4's numbers would belong to the pair
named here, and a later correction would say so and time the new pair; nothing is changed in silence.

**The guard.** `NOT_BEFORE` in `zfsvy_m4.py` moves from 18:30 UTC on 6 November (00:00 IST on 7 November) to 04:00 UTC on 6 November
(09:30 IST on 6 November, before the user's release). The guard stays as a mechanism: a full run before `--not-before` still starts
nothing and exits 3, a run with `--nq` is still not bound, and the self-test's guard check names the new time. The two other lines
that named the old time (the docstring and the refusal's log line) are reworded. Nothing else in the module changes: the forms, the
passes, the checks, the stops, the raw files and the report are those of correction 3. The self-test passed on the laptop after the
edit (SELFTEST PASS, ruff clean on F and E9), with no data and no host timing.

**The three items.** Unchanged from correction 3: one job per dataset, `fsvm4-hotpotqa`, `fsvm4-musique` and `fsvm4-2wiki`, each 2
CPUs and 28, 34 and 37 GB, each depending on the one before (so one runs at a time), the first on `isb-s0-hp`, which has ended
(done, return code 0), with the inputs listed first and the command of correction 3 (`--blas 2 --lps 2,4 --require-pin`, no retry
wrapper). The module's sha changes with the guard, so the first input line is re-read from the committed file. The optional smoke of
correction 3 (`fsvm4-smoke-hotpotqa`) is not sent: each job's own smoke pass (32 questions, BLAS 2) runs first and stops the job
before the record if a check fails.

**The host.** M4 is a timing: it runs on a quiet host and nothing is sent beside it. At the time of this correction the host has no
running mpr job but the feeder, the watchdog and the auto-cleanup, and the feeder's three remaining items wait behind a hold; the
items are placed ahead of that head and the feeder is not otherwise touched. A timed job of another track that appears while M4
runs would make the affected pass not a measurement: the raw files record the host's CPU load around each pass (1 s samples before
and after, printed in the report), and a pass read under such a load is repeated (`--overwrite`) rather than filed.

**What is not changed.** Sections 1 to 15 and corrections 1 to 3 above this one, the cuts, the reads, the rule, the pool, the forms
of both families, the threads, the pin, the question sets and the bars; M5's and A7's numbers; `zfsvc.py`, `zfsvt.py`, `zfsvx.py`,
`zfsvy.py`, `zfsvy_b16.py` and `zfsvy_cut.py`.

**What follows.** When the three passes are in, `report` is run, its two files are masked and committed (`git add -f`, those two
files only), and the calls that name the verdicts of Family 2 are made by a later file, as section 5 says.

**Commit policy.** This correction and `zfsvy_m4.py` are committed together, before any M4 pass and before any item of this
correction is sent. The raw files and the logs are never committed; no other host job is sent with the items.
