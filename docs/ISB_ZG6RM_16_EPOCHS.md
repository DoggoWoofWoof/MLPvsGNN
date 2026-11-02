# ISB: zg6r from its own encoder start, trained for 16 epochs (zg6r-m16), against zg6r, on PKO's pools

Declared 3 November 2026, before any ISB fit. Stage module: `outputs/mp_unified/zisb.py`. It imports `zml1s.py`, `ziso.py`,
`zdep.py`, `readswa.py` and `rmatch.py` and edits none of them. This is the next form that ISO's section 6 fixed for a KEEP.

## 1. The question

ISO filed KEEP (docs/ISO_ZG6R_OWN_START.md, section 11). zg6r-m (zg6r with its encoder drawn at the fit's seed, no zspk
tensor) gained on musique under WQ-musique (+0.0231 over three seeds, floor 0.0110). It lost on hotpotqa under WQ-musique
(deltas -0.0081, -0.0075, -0.0005 against a floor of 0.0075). The other ten reads were within their floors.

ISO's section 8 named the likely cause: zg6r's schedule was fixed for a warm encoder. zg6r's encoder had trained for 8 epochs
in ML1's zspk fit before zg6r's own 8 epochs, while zg6r-m had 8 epochs in all. ISO's section 6 fixed the next form: zg6r-m
trained for the epochs that zspk's fit and zg6r's fit took together, 8 + 8 = 16. The user's ruling stands: the GNN should not
be built over the MLP. So this stage asks whether the GNN without the MLP's weights reaches zg6r once it has the same budget.

## 2. What runs

- **The arm, zg6r-m16.** ISO's zg6r-m with lean_gpu's config `2e-3:1e-4:0.1:16:10` in place of `2e-3:1e-4:0.1:8:2`
  (lr:wd:dropout:epochs:swa_from). The learning rate, weight decay and dropout are unchanged. The run is 16 epochs, and SWA
  averages epochs 10 to 15: six epochs at the end, as zg6r's 2 to 7.
  - Everything else is ISO's zg6r-m, and so ML1's zg6r: the model, the trainer (zrm's train through `zml1s.run_arm`), the
    inputs (PKO's roots, the KT input, PKO's typed links), the card, the read cap (32,768) and the device (the host's GPU).
  - The start is ISO's: `zml1s.load_encoder` is ISO's `own_load` (no tensors) and `zg6r.encoder_used` is ISO's `own_used`
    (a fit must show its own start with no copies).
- **How the config reaches the trainer.** `rmatch.train`'s argv gets `--config 2e-3:1e-4:0.1:16:10` appended, as zep.py
  gave zrc its own. lean_screen's parser takes `--config` (default lean_gpu.CONFIG) and hands it to `lean_gpu.train_cmd`,
  whose fit loop is the only reader of the epoch count on zg6r's route. An argv that names a config already is refused.
- **What is saved and read.** The train saves p@ep0 to p@ep15 and p@swa (17 states). The read keeps p@swa alone (readswa),
  which is the only candidate any rule here reads.
- **The fits.** Three seeds (0, 1, 2) on the two WQ splits, so six fits: `scr-isb-zg6rm16-s<k>` (WQ-musique) and
  `scr-isb-zg6rm16-s<k>-hp` (WQ-hotpotqa). Each is stamped under the key `isb` with this file as its declaration. The stamp
  also names the config, ML1's config and the base.
- **The base.** ML1's zg6r fit of the same seed and split (`scr-ml1-zg6r-s<k>`, `-hp`), read on the same GPU. No base is
  fitted again. ISO's zg6r-m fits are read beside.
- **GPU shares.** As ISO's: WQ-musique at gpu 0.42 (cap 0.40), WQ-hotpotqa at gpu 0.75 (cap 0.72).

## 3. The reads and the floors

As ML1 section 3. There are twelve reads: two splits by six datasets, R@5 of p@swa on the s1eval carves. Each comes from a
compare of an ISB fit against ML1's zg6r fit of its seed and split, through zwq's `run_compare`. delta_k is zg6r-m16's R@5
minus zg6r's at seed k. The full run's floor is ML1's pair floor: max(0.0075, 2 x the root mean square of the two arms'
spreads).

## 4. The rule (fixed now)

**The screen** runs seed 0 on both splits, two fits.

- Each read's seed-0 delta is called as ML1's rule calls seed 0. A LOSS stands on seed 0 alone (ML1 section 4), so the
  screen calls it as the full run would.
- The floor is ML1's pair floor, with zg6r-m's spread over ISO's three seeds standing in for zg6r-m16's: the same own draw,
  at 8 epochs. zg6r's three seeds share one encoder start (ML1's seed-0 zspk), so its spread alone would leave out the
  draw's noise.

| outcome | when |
| --- | --- |
| KEEP | a LOSS on any read. zg6r stays, and seeds 1 and 2 do not run. |
| FULL_RUN | no LOSS. Seeds 1 and 2 run on both splits. Parity is enough, as for ISO's REPLACE. |
| INCOMPLETE | a fit, a compare, a base seed or a zg6r-m seed is missing, or a fit problem below. |

**The full run** is decided by ISO's rule (`ziso.decide`) on the twelve reads at three seeds, against ML1's zg6r:

- **REPLACE:** no ISO LOSS. An ISO LOSS is ML1's LOSS, or a mean delta at or below minus the floor.
- **KEEP:** otherwise.

**INCOMPLETE** also when any of these holds for an ISB fit that is read:

- its stamp does not show its own start with no copies;
- its stamps, screen.json or train.json do not carry ISB's config: train.json's must be the base's with epochs 16 and
  swa_from 10;
- its train did not save p@ep0 to p@ep15 and p@swa with SWA over epochs 10 to 15;
- its read lists more than p@swa;
- its device differs from its base's;
- its trainer's code differs from its base's over the base's keys (zml1s's own sha aside, as ISO's);
- a base's R@5 in ISB's compare differs from ML1's by more than 6e-5.

**Why REPLACE needs no GAIN.** It is ISO's reason: at parity, the GNN without the MLP in it is preferred.

## 5. Reported beside, deciding nothing

- Per read, over the seeds zg6r-m16 has: the means of zspk and zg6r (ML1's compares), zg6r-m (ISO's) and zg6r-m16, and
  zg6r-m16 - zg6r-m (the 8 more epochs from the same kind of start).
- The part of zg6r's gain over zspk that zg6r-m16 keeps, (zg6r-m16 - zspk) / (zg6r - zspk), where zg6r - zspk exceeds 0.002.
- musique's 3-hop and 4-hop slices (DEP's, from the read files) of each ISB fit against ML1's zg6r of its seed and split.
- Seconds per fit (`ml1_timing.json`).

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| REPLACE | zg6r-m16 replaces zg6r as the base that the GNN's forms are built on: the GNN without the MLP's weights, at parity with zg6r on all twelve reads. Where a form has been adopted on zg6r (DEP, RTR, IRA or LTG), that form stays the GNN of record until its own screen on zg6r-m16, in its own declared file, carries it over; until then zg6r-m16 is reported beside it. Otherwise zg6r-m16 is the GNN of record. Claims 1 and 3 are measured against the GNN of record. |
| KEEP (screen or full run) | zg6r stays. The next form, in its own declared file, is zg6r-m in zg6r's two phases. Each ISO zg6r-m fit is continued from its p@swa state for 8 epochs, with a fresh Adam optimizer and SWA from epoch 2. That is zg6r's schedule, with its reset at epoch 8, and no MLP weights. It tests the reset (section 7) rather than the epoch count. |
| INCOMPLETE | What is missing is supplied; nothing else changes. |

**The freeze.** Results freeze on 6-7 November. A full run not read by then does not change the GNN of record for the frozen
results. Its reads are filed when they land.

A verdict closes only the form it tested.

## 7. Risks, stated before any number

- **No reset.** zg6r's 16 epochs (zspk's 8, then zg6r's 8) had a fresh Adam state and a second SWA window at epoch 8.
  zg6r-m16 has one run, one optimizer and one SWA window. A KEEP may be the reset's doing, which is why section 6's KEEP row
  tests it.
- **A constant learning rate.** lean_gpu's fit loop refuses cos, adamw and drop, so all 16 epochs run at 2e-3. Late epochs
  may overfit the WQ splits' train rows; SWA over the last six is the guard.
- **One seed at the screen.** The floor's second spread is zg6r-m's, not zg6r-m16's. A screen LOSS that zg6r-m16's own
  spread would widen past is still called a LOSS.
- **Read noise.** The read on this GPU does not repeat bit for bit (ML1 A1), so differences far below the floors are noise.
- **metaqa.** Its floors (0.062 and 0.064 at the screen) are wide, so its zero-shot reads decide little.
- **Time.** ISO's zg6r-m fits trained in 4,166-4,598 s (WQ-musique) and 7,321-7,574 s (WQ-hotpotqa). Twice the epochs is
  about 2.3-2.6 h and 4.1-4.2 h per fit, plus a read of p@swa alone. The full run's four fits take about 13 fit-hours. That is
  about 11 hours on the card, with the two WQ-musique fits side by side.

## 8. How it runs

Feeder items, GPU-first. ISB's screen goes ahead of zg6-z's (ISO section 7, which decides nothing):

1. `isb-selftest` (CPU).
2. `isb-s0` (gpu 0.42) before `iso-zg6z-s0`, and `isb-s0-hp` (gpu 0.75) before `iso-zg6z-s0-hp`.
3. `isb-compare` and then `isb-calls` (CPU), after both screen fits.
4. `isb-gate`: `verdict_gate.py --json outputs/isb/calls.json --key screen.verdict --want FULL_RUN`.
5. `isb-s1`, `isb-s2` (gpu 0.42) and `isb-s1-hp`, `isb-s2-hp` (gpu 0.75), behind the gate.
6. `isb-compare-full` and `isb-calls-full` (CPU), after the four.

Each fit runs:

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac F --gc 0.9 -- outputs/mp_unified/zisb.py fit --name NAME
python outputs/mp_unified/zisb.py compare-all
python outputs/mp_unified/zisb.py calls          (outputs/isb/calls.json and .md)
```

The self-test, `python outputs/mp_unified/zisb.py --selftest`, passed on the laptop's CPU before this commit. It covers:

- the names and the pairs;
- the config: lean_mlp5's parse and lean_gpu's candidate names, with lean_gpu.CONFIG as ML1's config;
- `with_config`'s argv, and its refusal of a second config;
- the trainer's route: `zml1s.run_arm` calls `W.Z3.RM.train`, and zwq's zp1c3p hands it the rmatch that is patched;
- `on_isb`'s patches, and that they are put back, also on a failure;
- the stamp's config and base;
- `fit_problems` on fixtures in the records' shapes, with each departure named;
- the screen rule on made-up reads: FULL_RUN flat or up, KEEP on a seed-0 LOSS, the stand-in spread widening the floor, and
  INCOMPLETE on a hole in the reads or in zg6r-m's seeds;
- `calls` on an empty root (INCOMPLETE).

A laptop dry run of `calls` was also made, with ISO's filed compares relabelled as ISB's. It reproduced ISO's seed-0 calls
(GAIN on musique under WQ-musique, +0.0202; LOSS on hotpotqa under WQ-musique, -0.0081 at a floor of 0.0075) and found no
problem in the bases.

Not claimed: anything about the encoder's start on other pools or regimes, or about schedules other than this one.
