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

## 9. Amendment A1, 3 November 2026, before any ISB fit: ISB waits until after the freeze

**What happened.** On 3 November the user asked to finish faster. ISB's screen needs about 7.5 hours of the card, and its
full run about 12 more. Five GNN forms are screened on zg6r before the 6-7 November freeze (docs/STK_GNN_STACK.md section 1).
An ISB REPLACE would change the base under all five. Section 6 would then keep a form adopted on zg6r as the GNN of record
until its own screen on zg6r-m16 carries it over, and those screens do not fit before the freeze.

**The change.** ISB's fits wait until after the freeze. `isb-s0` and `isb-s0-hp` wait on `hold-isb-after-freeze`, a name no
job has. The feeder keeps them waiting and does not drop them. ISB's full-run items wait behind `isb-gate`. GNX's screen
(docs/GNX_NEXT_COLUMNS_IN_GNN.md) takes ISB's place on the card. The hold was put in the queue earlier on 3 November, before
this amendment. No ISB fit has run.

**What does not change.** Sections 1 to 8: the arm, the fits, the rule and what follows. After the freeze, an amendment that
changes only these items' dependencies releases the hold.

## 10. Amendment A2, 4 November 2026, before any ISB fit: ISB's hold is released and ISB goes behind IRA's seeds

**What happened.** A1 parked ISB because an ISB REPLACE would change the base under the GNN forms screened on zg6r. By 23:24 IST on
4 November every one of them had closed on zg6r: GNX NO_GAIN (ec09b5a), GNP NO_GAIN (033d1a9), GNW NO_GAIN under the six-read rule
(86d8508), LTX NO_GAIN (99caf82), GNR NO_GAIN (c33bb44), GNT NOT_ADOPTED (4d62c5d) and GNS NO_GAIN (bbccb41). Only IRA's screen passed
(fc2186e), and its full run is queued. The stack gate found IRA alone ("no stack"). The card now holds FSV's GNN L3 fits and IRA's
seeds, well under the 24 hours of declared GPU work the user asked to keep queued (17 and 20 October). ISB's screen (about 7.5 hours)
and full run (about 12 hours) still end before the freeze (00:00 IST on 7 November) if they start behind IRA's four fits.

**The change.** `isb-s0` and `isb-s0-hp` no longer wait on `hold-isb-after-freeze`. Both items move in the queue to just after IRA's
seeds (`ira-re-compare-full`), so that IRA's fits take the card first; ISB's other items stay behind `isb-gate`. If ISB's full run
cannot be filed before the freeze, its screen is filed beside and nothing replaces zg6r: a form adopted on zg6r stays the GNN of
record (section 6).

**What does not change.** Sections 1 to 8: the arm, the fits, the rule and what follows.

## 11. Results

**The screen, 6 November 2026 (filed 09:04 IST).** Filed in `outputs/isb/calls.json` and `calls.md` (made on the host by `isb-calls` from `scr-isb-zg6rm16-s0` and `scr-isb-zg6rm16-s0-hp` beside ML1's zg6r, ISO's zg6r-m and the base seeds). The seed-0 fits trained for 20,476 s (`-s0`) and 13,017 s (`-s0-hp`) on the host's GPU; no problems.

| rule | verdict | counts | the reads that decide |
|---|---|---|---|
| own (section 4, twelve reads) | **KEEP** | 3 GAIN, 2 LOSS, 7 WITHIN | LOSS: WQ-musique metaqa -0.0936 (floor 0.0621) and WQ-hotpotqa metaqa -0.1104 (0.0637); GAIN: WQ-musique musique +0.0148 (0.0110) and webqsp +0.0212 (0.0142), WQ-hotpotqa musique +0.0154 (0.0126) |

- **What the rule does.** KEEP on a LOSS on any read: zg6r stays and seeds 1 and 2 do not run (section 4). `isb-gate` stopped at 09:05 and the feeder dropped `isb-calls-full` at 09:18, as declared. A verdict closes only the form it tested.
- **Beside, deciding nothing (section 5).** musique by hop count, zg6r-m16 minus zg6r at seed 0: WQ-musique 3-hop +0.0031 and 4-hop +0.0099; WQ-hotpotqa 3-hop +0.0206 and 4-hop +0.0043. On the metaqa reads zg6r-m16 keeps a negative share of zg6r's gain over zspk (-16.0 and -2.2 times); on musique and webqsp it gains.
- **The six-read rule (`zwww.py`, probe only).** `zwww.py call --stage isb` prints the screen as INCOMPLETE: it finds four of the six deciding reads (one GAIN, WQ-hotpotqa musique; three WITHIN) and reports WQ-hotpotqa's musique-3hop and musique-4hop as missing, because `calls.json` keeps the slices at its top level with deltas only, with no floors and no calls. Both deltas are positive (+0.0206, +0.0043), so neither read can be a LOSS: if both are WITHIN or GAIN, the six-read rule would read ADOPT_FOR_FULL_RUN where ISB's own rule reads KEEP. The two rules disagree on the form; ISB's own gate is the declared one, so nothing ran on the six-read call. `zwww.py` is unchanged and no `outputs/www/isb.json` is filed.
- **What follows (section 6's table, fixed before the fit):** KEEP, so zg6r stays. The next form named there is zg6r-m in zg6r's two phases (each ISO zg6r-m fit continued from its p@swa state for 8 epochs, fresh Adam, SWA from epoch 2); it needs its own declared file and nothing is declared here. Results freeze on 6-7 November, so nothing from ISB reaches the frozen results.

## 12. Amendment A3, 6 November 2026, before any seed-1 or seed-2 fit: ISB's full run is run beside, as a record

**What happened.** The screen is filed (section 11): KEEP on two zero-shot LOSSes, so section 4 stops seeds 1 and 2 and `isb-gate` stopped (exit 1), and the feeder dropped the waiting full-run items, `isb-calls-full` among them, at 09:18 IST because their dependency had ended badly. The host's GPU then had no declared work, and the user's standing rule is that it is not left idle (17 and 20 October; 6 November). The six-read rule cannot call the screen (section 11: INCOMPLETE), so there is no second rule that sends the form to its full run either.

**The change.** The four full-run fits (`isb-s1`, `isb-s2`, `isb-s1-hp`, `isb-s2-hp`), `isb-compare-full` and `isb-calls-full` are run anyway, as a record beside zg6r, deciding nothing:
- They no longer wait on `isb-gate`; their other dependencies (`pkoh-gate`, `readswa-check`) stay. They also wait on the end of the timed pass of FSV's M4 (`fsvm4b-2wiki`, the last of its three items, as correction 5 requeued them), because that pass needs a quiet host. The four fits and the two CPU items go to the feeder under new names (`isbf-s1`, `isbf-s2`, `isbf-s1-hp`, `isbf-s2-hp`, `isbf-compare-full`, `isbf-calls-full`) because the feeder dropped the old ones for good; the fit names inside the commands (`scr-isb-zg6rm16-sN`, `-sN-hp`) are section 8's and unchanged.
- The commands, the fits, the configuration, the seeds and the cards are section 8's, unchanged. `zisb.py` is unchanged.
- The verdict of the full run is ISO's rule on the twelve reads at three seeds against ML1's zg6r (section 4), filed as `full_run` in `outputs/isb/calls.json` and `calls.md`. Beside it, the six-read rule's call is made as `zwww.py` makes it for any block that has the reads; if the record keeps the musique slices outside the reads again, the call stays INCOMPLETE and is filed as such.

**What does not change.** Sections 1 to 11: the arm, the rule and the screen's verdict. KEEP stays the screen's verdict. The freeze stays: results freeze on 6-7 November and a full run not read by then does not change the GNN of record (section 6); nothing from this run reaches the frozen results, and zg6r is the GNN of record for Claims 1 and 3 whatever it reads. A REPLACE would only make zg6r-m16 the base for forms declared after the freeze, by section 6's table, in their own files. A verdict closes only the form it tested.

## 13. Results of A3's full run, 11 November 2026: filed beside zg6r, deciding nothing

**The record.** `isbf-calls-full` remade `outputs/isb/calls.json` and `calls.md` on the host at 17:23 IST, from the six ISB fits beside ML1's zg6r, ISO's zg6r-m and the base seeds. The files here are the host's, byte for byte (`calls.json` sha256 0c60a66e417049f3...). The `screen` block is unchanged from section 11, `full_run` is new, and neither has a problem. Seeds 1 and 2 trained for 8,366 and 8,349 s (WQ-musique) and 13,729 and 12,320 s (WQ-hotpotqa) on the host's GPU.

- **How the last fit ran.** `isbf2-s2-hp` (the fit `scr-isb-zg6rm16-s2-hp`) ran out of GPU memory at SWA's start, right after epoch 10, at 13:45 IST on 11 November, under cuda_alloc's cap of 0.72 of the card. `retry_cmd` restarted it from epoch 0 under 0.77, as it is built to. The second run repeated the first's training loss and step count at every epoch from 0 to 10, to four decimals, so the restart changed no result. The train seconds above are the second run's.

| rule | verdict | counts | the reads that decide |
|---|---|---|---|
| ISO's (section 4: `ziso.decide` on the twelve reads at three seeds, against ML1's zg6r) | **KEEP** | 1 GAIN, 3 LOSS, 8 WITHIN; 3 ISO LOSS | WQ-musique metaqa (mean -0.0745, floor 0.0292), WQ-hotpotqa metaqa (-0.1506, 0.0665), WQ-hotpotqa hotpotqa (seed 0 -0.0179, floor 0.0176) |
| six-read (`zwww.py call --stage isb`, beside) | **INCOMPLETE** | 4 WITHIN of the six; 2 missing | WQ-musique hotpotqa +0.0009 and 2wiki +0.0073, WQ-hotpotqa musique +0.0050 and 2wiki +0.0030 (means); WQ-hotpotqa musique-3hop and musique-4hop missing |

- **ISO's rule, read by read.** Each call is ML1's: a GAIN needs seed 0 at the floor and seeds 1 and 2 above zero, and a LOSS stands on seed 0 alone. The floor is now ML1's pair floor with zg6r-m16's own spread.
  - All three ISO LOSSes are on data the split did not train on. metaqa, zero-shot under both splits, falls at every seed: -0.0936, -0.0697, -0.0603 under WQ-musique and -0.1104, -0.1974, -0.1439 under WQ-hotpotqa.
  - hotpotqa under WQ-hotpotqa, which holds it out, is a LOSS on seed 0 alone: -0.0179, 0.0003 past its floor of 0.0176. Seeds 1 and 2 read -0.0129 and -0.0030, and the mean, -0.0113, is inside the floor. At the screen it was WITHIN against the stand-in floor of 0.0200. A call this thin can turn on read noise (section 7); the two metaqa LOSSes decide KEEP without it.
  - The one GAIN, webqsp under WQ-musique, is as thin: seed 0 +0.0212 against a floor of 0.0211, with seeds 1 and 2 at +0.0098 and +0.0375.
  - Two of the screen's three GAINs do not hold at three seeds. musique under WQ-musique, which holds it out, rises at every seed (+0.0148, +0.0184, +0.0267; mean +0.0200), but its floor widened from 0.0110 to 0.0180 and seed 0 is under it, so it is WITHIN. musique under WQ-hotpotqa is WITHIN: +0.0154, -0.0037, +0.0034 against 0.0168.
  - Every in-domain read of 2wiki, hotpotqa and musique is WITHIN, with means from +0.0009 to +0.0073.
- **The six-read rule.** As at the screen, `calls.json` keeps the musique slices at its top level with deltas only. `zwww.py` finds four of the six reads and calls the full run INCOMPLETE, and `outputs/www/isb.json` files that call, as A3 asked; `zwww.py` is unchanged. The missing reads cannot change the outcome: the four found are WITHIN, and this rule adopts a full run only with a GAIN on one of its four dataset reads, so every call the two slices could take gives NOT_ADOPTED. Their deltas, zg6r-m16 - zg6r at seeds 0, 1 and 2: 3-hop +0.0206, -0.0026, -0.0035; 4-hop +0.0043, -0.0049, +0.0099. So both rules leave zg6r in place. The disagreement that section 11 found at the screen came from WQ-hotpotqa musique's seed-0 GAIN, and it does not recur at three seeds.
- **Beside, deciding nothing (section 5; levels are means over the three seeds).**
  - On the in-domain reads zg6r-m16 sits level with zg6r: WQ-musique hotpotqa 0.8592 against 0.8584 and 2wiki 0.9333 against 0.9260; WQ-hotpotqa musique 0.5997 against 0.5946 and 2wiki 0.9317 against 0.9287.
  - On webqsp its means are higher under both splits (0.6832 against 0.6604; 0.6784 against 0.6564): a GAIN under WQ-musique, WITHIN under WQ-hotpotqa.
  - On metaqa it falls below zspk as well as zg6r: 0.2772 against 0.3574 and 0.3518 under WQ-musique, and 0.2185 against 0.3199 and 0.3691 under WQ-hotpotqa. The kept column in `calls.md` is read through these levels.
  - The 8 more epochs from the same kind of start (zg6r-m16 - zg6r-m) cost metaqa most: -0.0671 and -0.0926, then -0.0118 on webqsp under WQ-hotpotqa. Every other read moves between -0.0031 and +0.0076. That is what section 7's constant-learning-rate risk would look like, but nothing here tests it.
  - musique by hop count under WQ-musique, which holds musique out: the deltas are positive at every seed on both slices (3-hop +0.0031, +0.0114, +0.0259; 4-hop +0.0099, +0.0117, +0.0253). Under WQ-hotpotqa they are mixed, as listed above.
- **What follows (section 6's table, fixed before the fit).** KEEP, as the screen filed: zg6r stays. The next form named there is zg6r-m in zg6r's two phases: each ISO zg6r-m fit continued from its p@swa state for 8 epochs, with a fresh Adam optimizer and SWA from epoch 2. It tests the reset rather than the epoch count (section 7). It needs its own declared file, and none is declared here. Nothing from this run reaches the frozen results (A3). A verdict closes only the form it tested.
