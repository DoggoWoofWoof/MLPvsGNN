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
