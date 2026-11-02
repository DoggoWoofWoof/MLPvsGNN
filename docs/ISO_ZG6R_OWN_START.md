# ISO: zg6r from its own encoder start (zg6r-m), against zg6r, on PKO's pools

Declared 2 November 2026, before any ISO fit. Stage module: `outputs/mp_unified/ziso.py`. It imports `zml1s.py` and does not
edit it. This is the first isolation arm that ML1 section 6 names (G6RW section 7, third row).

## 1. The question

ML1 adopted zg6r (docs/ML1_REFITS_ON_PKO.md, section 13). zg6r is its own model in every part but one. Its encoder (zrm's
block encoder and chain match, `ENC_NAMES`, 224,737 numbers) starts from ML1's seed-0 zspk fit of the split. zspk is the lean
GNN: the MLP plus one propagation step. The user said on 31 October: "gnn should not be built over mlp it needs to be the best
model we can get in the gnn".

So this stage asks whether zg6r needs that start. **zg6r-m** is zg6r with its encoder drawn by zrm's own init at the fit's
seed; everything else is ML1's zg6r.

## 2. What runs

- **The arm, zg6r-m.** Everything is ML1's zg6r's: the model, the trainer (zrm's train through `zml1s.run_arm`), the inputs
  (PKO's roots, the KT input, PKO's typed links), the card, the read cap (32,768) and the device (the host's GPU, ML1 A1).
  There is one difference. `zml1s.load_encoder` is replaced by an own-start record that holds no tensors, so no zspk tensor
  is copied. `zg6r.encoder_used` is replaced by its converse: a fit stamped by ISO must show its own start with no copies.
- **The fits.** Three seeds (0, 1, 2) on the two WQ splits, so six fits: `scr-iso-zg6rm-s<k>` (WQ-musique) and
  `scr-iso-zg6rm-s<k>-hp` (WQ-hotpotqa). Each is stamped under the key `iso` with this file as its declaration.
- **The base.** ML1's zg6r fit of the same seed and split, read on the same GPU. No base is fitted again.
- **GPU shares.** As ML1's zg6r items after its A3: WQ-musique at gpu 0.42 (cap 0.40), WQ-hotpotqa at gpu 0.75 (cap 0.72).

## 3. The reads and the floors

As ML1 section 3. There are twelve reads (two splits, six datasets, p@swa on the s1eval carves), each from a compare of a
zg6r-m fit against its zg6r base through zwq's `run_compare`. delta_k is R@5 of zg6r-m minus R@5 of zg6r at seed k. The floor
is ML1's pair floor: max(0.0075, 2 x the root mean square of the two arms' spreads).

## 4. The rule (fixed now)

Each read gets ML1's call (GAIN, LOSS, WITHIN, UNCONFIRMED_GAIN), by `zml1.verdict`. A read is **ISO's LOSS** when ML1 calls
it a LOSS, or when the mean of its three deltas is at or below minus its floor. The second clause catches a loss that every
seed agrees on even when seed 0 is lucky.

- **REPLACE:** no ISO LOSS on any of the twelve reads. zg6r-m becomes the GNN of record.
- **KEEP:** otherwise. zg6r stays.
- **INCOMPLETE:** any of these:
  - a fit or a compare is missing;
  - a fit's stamp does not show its own start with no copies;
  - a fit's device differs from its base's;
  - a fit's trainer code differs from its base's (zml1s's own sha aside);
  - an arm has fewer than twelve reads at three seeds.

**Why REPLACE needs no GAIN.** The question is whether the start is needed. The user's ruling is that the GNN should not be
built over the MLP, so at parity the model without the MLP in it is preferred. GAINs are reported beside the verdict.

## 5. Reported beside, deciding nothing

- ML1's GAIN and UNCONFIRMED_GAIN counts for zg6r-m against zg6r, and each read's mean delta.
- Per read, the seed means of zspk, zg6r and zg6r-m, and the part of zg6r's gain over zspk that zg6r-m keeps:
  (zg6r-m - zspk) / (zg6r - zspk), where zg6r - zspk exceeds 0.002. zspk and zg6r come from ML1's compares.
- Seconds per fit (`ml1_timing.json` in each fit folder).

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| REPLACE | zg6r-m is the GNN of record. Claim 1 and Claim 3 are measured against it, and the depth form (MUSIQUE_DIAGNOSIS step 3) is built on it. |
| KEEP | zg6r stays, and the start carries part of its gain. The next form, in one declared file, is zg6r-m trained for the epochs that zspk's fit and zg6r's fit took together. That gives the GNN the MLP's training budget without the MLP's weights. |
| INCOMPLETE | What is missing is supplied; nothing else changes. |

A verdict closes only the form it tested.

## 7. The second isolation arm, zg6-z (declared now, wired in an amendment)

zg6-z is zg6 (no relation messages) started from ML1's seed-0 zspk encoder. It runs at seed 0 on both splits (two fits) and
is compared with ML1's zg6r at seed 0. It is reported beside and decides nothing: it says how much of zg6r's gain comes from
the relation messages.

zg6 under WQ runs through WQK's own route (its train refuses `--seed`). Its wiring onto PKO's roots is an amendment to this
file, with its self-test, before its fits. Its items go behind zg6r-m's.

## 8. Risks, stated before any number

- **The schedule.** zg6r's schedule (epochs, learning rate) was fixed for a warm encoder, and it may be too short from a cold
  draw. A KEEP may then mean "too few epochs" rather than "the MLP's weights are needed". Section 6's KEEP row tests exactly
  that.
- **Seeds.** There are three per split. A LOSS on seed 0 alone is as noisy as ML1's; the mean clause catches losses that all
  the seeds agree on.
- **Read noise.** The read on this GPU does not repeat bit for bit (ML1 A1), so differences far below the floors are read
  noise.
- **metaqa.** Its floors (0.025 to 0.073 in ML1) are wide, so its zero-shot reads decide little.

## 9. How it runs

Feeder items run GPU-first, behind `knee-zrckp-s0-b`:

1. `iso-zg6rm-s0`, `-s1`, `-s2` (WQ-musique, gpu 0.42).
2. `iso-zg6rm-s0-hp`, `-s1-hp`, `-s2-hp` (gpu 0.75).
3. `iso-compare` (CPU), after all six fits.
4. `iso-verdict`, after the compare.

Each fit runs:

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac F --gc 0.9 -- outputs/mp_unified/ziso.py fit --name NAME
python outputs/mp_unified/ziso.py compare-all
python outputs/mp_unified/ziso.py verdict        (outputs/iso/verdict.json and .md)
```

The self-test, `python outputs/mp_unified/ziso.py --selftest`, passed before this commit. It covers:

- the names and the pairs;
- the patches, and that they are put back, also on a failure;
- `own_used`'s refusals;
- the rule on made-up reads: REPLACE when flat or up; KEEP on a seed-0 LOSS and on a mean-only LOSS; INCOMPLETE on a hole.

Not claimed: anything about the encoder's start on other pools or regimes.

## 10. Amendment A1, 2 November 2026, before any zg6-z fit: zg6-z's wiring

**The route.** Section 7 said that zg6 under WQ runs through WQK's own route, whose train refuses `--seed`, and that its wiring
onto PKO's roots would come here. zg6-z runs on zg6r's own route instead: zml1s's zg6r branch, with PKO's roots, ML1's encoder
start (the split's zspk seed-0 encoder), the card, the seed and the read cap of 32,768. Only U2n's typed links are removed, where
they are loaded: `zu2n.load_typed` answers None for every carve, as it does for a passage graph. The module is
`outputs/mp_unified/zisoz.py`. It imports zml1s, zg6r and zu2n and edits none of them.

**Why that is zg6.**

- zg6r's self-test (zg6r.py, step 5) shows that with no typed edge, zg6r's score is zg6's bit for bit, in eval and in training
  with the same draws. It also shows that zg6r's construction moves the global generator as zg6's does.
- So the model trained and read is zg6. G6r's parameters are built and never used.
- zisoz's self-test shows the same on zg6r's toy KB carve, whose typed links are real (106 typed edges in the batch). With the
  links removed, zg6r's score equals zg6's bit for bit in eval and in training. With them, it differs.

This is a change of route, disclosed here: zg6's own class is not run; zg6r's class with no typed link is.

**What runs.**

- **Fits.** `scr-iso-zg6z-s0` (WQ-musique) and `scr-iso-zg6z-s0-hp` (WQ-hotpotqa). Each trains and reads in one process with the
  links removed. Each is stamped under the key `isoz`, which names the typed links as removed. Any later read of these fits must
  remove the links too.
- **Compares.** Each fit against ML1's zg6r seed 0 of its split (`scr-iso-zg6z-vs-zg6r-s0`, with `-hp`).
- **The report** (`outputs/iso/zg6z.json` and `.md`). Per read:
  - zspk, zg6r and zg6-z at seed 0;
  - zg6-z's delta from zg6r, with zg6r's seed floor beside it for scale;
  - the part of zg6r's gain over zspk that zg6-z keeps, (zg6-z - zspk) / (zg6r - zspk), where that gain exceeds 0.002.

  It decides nothing, as section 7 says.

**The report is incomplete when any of these holds:**

- a fit or a compare is missing;
- a fit does not name its typed links as removed;
- its encoder start is not ML1's zg6r seed-0 start;
- its device or its trainer's code differs from that fit's;
- zg6r's seed 0 differs between the two stages.

**Where its items go.** The GPU shares are zg6r's: 0.42 with cap 0.40 on WQ-musique, and 0.75 with cap 0.72 on WQ-hotpotqa. The
items go behind zg6r-m's, as section 7 says.

- The musique-split fit sits between DEP's two fits (docs/DEP_GNN_DEPTH.md), so it can share the card with DEP's musique-split
  fit.
- The hotpotqa-split fit goes after DEP's.
- Then `iso-zg6z-compare` and `iso-zg6z-report` run on the CPU.

**What stays the same.** Sections 1 to 9: ISO's arm, rule and verdict. zg6-z is not in the verdict.

The self-test, `python outputs/mp_unified/zisoz.py --selftest`, passed before this commit. It covers:

- the names and the pairs;
- the patches, and that they are put back, also on a failure;
- that zg6r's carve calls the loader that is patched, through zg6r's own zu2n;
- the toy KB carve with its typed links and with them removed;
- the report's rows: the part kept, the floor, a missing value, and a base that disagrees.

## 11. Results (filed 2 November 2026): KEEP

`ziso.py verdict` ran on the host at 17:48 UTC (ziso.py sha256 22eba3ff..., the file of A1). The records are
`outputs/iso/verdict.json` and `.md`. All six fits and twelve compares are present and pass section 4's checks: no problems.

**Counts:** GAIN 1, LOSS 1, UNCONFIRMED_GAIN 0, WITHIN 10; ISO LOSS 1. Under section 4, one ISO LOSS gives **KEEP**: zg6r stays
the GNN of record.

| split | dataset | d0 | d1 | d2 | mean | floor | ML1 call | ISO loss |
|---|---|---:|---:|---:|---:|---:|---|---|
| WQ-musique | metaqa | -0.0446 | +0.0057 | +0.0166 | -0.0074 | 0.0621 | WITHIN | no |
| WQ-musique | squad | -0.0022 | +0.0002 | -0.0009 | -0.0010 | 0.0075 | WITHIN | no |
| WQ-musique | musique | +0.0202 | +0.0249 | +0.0241 | +0.0231 | 0.0110 | GAIN | no |
| WQ-musique | hotpotqa | -0.0081 | -0.0075 | -0.0005 | -0.0054 | 0.0075 | LOSS | yes |
| WQ-musique | 2wiki | -0.0017 | -0.0027 | +0.0067 | +0.0008 | 0.0075 | WITHIN | no |
| WQ-musique | webqsp | +0.0142 | +0.0074 | +0.0241 | +0.0152 | 0.0142 | WITHIN | no |
| WQ-hotpotqa | metaqa | -0.0628 | -0.0786 | -0.0325 | -0.0580 | 0.0637 | WITHIN | no |
| WQ-hotpotqa | squad | -0.0006 | -0.0007 | +0.0003 | -0.0003 | 0.0075 | WITHIN | no |
| WQ-hotpotqa | musique | +0.0056 | -0.0089 | -0.0038 | -0.0024 | 0.0126 | WITHIN | no |
| WQ-hotpotqa | hotpotqa | -0.0117 | -0.0127 | -0.0019 | -0.0088 | 0.0200 | WITHIN | no |
| WQ-hotpotqa | 2wiki | -0.0031 | -0.0056 | +0.0002 | -0.0028 | 0.0075 | WITHIN | no |
| WQ-hotpotqa | webqsp | +0.0214 | +0.0261 | +0.0539 | +0.0338 | 0.0292 | WITHIN | no |

- **The ISO LOSS** is hotpotqa under WQ-musique. ML1's rule calls it a LOSS: seed 0 at -0.0081 and seed 1 at -0.0075 against a
  floor of 0.0075. The mean is -0.0054.
- **The GAIN** is musique under WQ-musique, the main target: +0.0202, +0.0249 and +0.0241 against a floor of 0.0110.
- webqsp rises under both splits (means +0.0152 and +0.0338), within the floors. metaqa under WQ-hotpotqa falls (mean
  -0.0580), within its floor of 0.0637.

**Beside, deciding nothing** (seed means; kept = (zg6r-m - zspk) / (zg6r - zspk), where zg6r's gain exceeds 0.002):

| split | dataset | zspk | zg6r | zg6r-m | kept |
|---|---|---:|---:|---:|---:|
| WQ-musique | metaqa | 0.3574 | 0.3518 | 0.3443 |  |
| WQ-musique | squad | 0.9139 | 0.9150 | 0.9140 |  |
| WQ-musique | musique | 0.4934 | 0.5086 | 0.5317 | 2.511 |
| WQ-musique | hotpotqa | 0.8440 | 0.8584 | 0.8530 | 0.626 |
| WQ-musique | 2wiki | 0.9105 | 0.9260 | 0.9268 | 1.049 |
| WQ-musique | webqsp | 0.6403 | 0.6604 | 0.6756 | 1.76 |
| WQ-hotpotqa | metaqa | 0.3199 | 0.3691 | 0.3111 | -0.179 |
| WQ-hotpotqa | squad | 0.9133 | 0.9135 | 0.9132 |  |
| WQ-hotpotqa | musique | 0.5788 | 0.5946 | 0.5923 | 0.851 |
| WQ-hotpotqa | hotpotqa | 0.8091 | 0.7929 | 0.7841 |  |
| WQ-hotpotqa | 2wiki | 0.9143 | 0.9287 | 0.9259 | 0.804 |
| WQ-hotpotqa | webqsp | 0.6345 | 0.6564 | 0.6902 | 2.541 |

On musique under WQ-musique, zg6r-m keeps 2.5 times zg6r's gain over zspk (0.5317 against 0.5086 and 0.4934). On hotpotqa there
it keeps 0.63 of it, and on metaqa under WQ-hotpotqa none (-0.18).

**What follows (section 6, KEEP row).**

- zg6r stays the GNN of record. The stages built on the base after ISO take KEEP's branch: LTG, DEP, RTR and IRA run on ML1's
  zg6r, and their fits start from ML1's zspk encoder. Their amendments of 2 November, and A2 below for zg6-z, read that start
  under ML1's own names.
- The start carries part of zg6r's gain on hotpotqa and metaqa, but not on musique, where the own start is above zg6r at all
  three seeds.
- The next form, in its own declared file, is zg6r-m trained for the epochs zspk's fit and zg6r's fit took together. Both ran
  8 epochs (SWA from epoch 2), so zg6r-m trains for 16.

## 12. Amendment A2, 2 November 2026, before any zg6-z fit: ML1's start read under ML1's names

zg6-z starts from ML1's seed-0 zspk encoder (A1). Its fit context sets zml1s's declaration and key to zg6-z's (`isoz`). zml1s's
loader then refuses ML1's own read stamp in the zspk fit, as it did for LTG's first fit (docs/LTG_NEXT_HOP_GNN.md, amendment
A1, gives the cause). No zg6-z fit had started.

**The change.**

- `zisoz.py`'s fit context installs `ml1start.ml1_load` in zml1s. That is ML1's loader with ML1's declaration, stage and key in
  force for the length of the load only. The tensors and the record are the loader's, bit for bit, and a zspk fit without
  ML1's read stamp is still refused.
- Each fit's code block also names `ml1start_sha256`. The report's code check compares the base's keys only, so it does not
  change.

**What does not change.** Sections 1 to 11, and A1's route, fits, compares and report.

**The queue.** `iso-zg6z-s0` and `iso-zg6z-s0-hp` also wait on `ml1start-check` (LTG's A1).

**Self-test.** `python outputs/mp_unified/zisoz.py --selftest` passed on the laptop's CPU before this commit. It now checks that
the fit context installs `ml1_load` and that the code block names ml1start.
