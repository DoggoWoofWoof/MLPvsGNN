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
