# The WWW rule: the three WWW datasets decide

Declared 3 November 2026, before GNS's seed-0 read and before any WQ-hotpotqa read of GNA, GNP, GNW, LTX, GNR, GNT or
GNS. Code: `outputs/mp_unified/zwww.py`. Audit: `outputs/www/audit.md`.

## 1. Why

The WWW 2027 paper reports three datasets: 2wiki, HotpotQA and MuSiQue (paper/www2027_v2/main.tex, Setup: HippoRAG 2's benchmark, 1,000 questions from each). MetaQA,
WebQSP and SQuAD belong to the journal version (paper/www2027_v2/PLAN.md section 11). The stages' rules were written for
all six datasets:

- LTG's in-domain rule decides on ten reads. Four of them are webqsp and squad, two in each split.
- DEP's sixteen-read rule also decides on the zero-shot reads, metaqa among them.

So reads outside the paper decided screens:

- GNW's and GNT's seed-0 stops were STOP on one read each: WQ-musique webqsp, −0.0124 and −0.0108. Their WQ-hotpotqa fits
  were never run.
- DEP's screen is ADOPT_FOR_FULL_RUN on one GAIN: WQ-hotpotqa metaqa, +0.0594. Metaqa never trains, and it is not in the
  paper. DEP's six WWW reads are all WITHIN. That adoption queued DEP's full run, gave DEP a stack seat, and kept RTR's
  and IRA's screens shut (each opens only when DEP's screen fails).

## 2. The rule

The six deciding reads are the in-domain reads of the three datasets:

| split | reads |
|---|---|
| WQ-musique | hotpotqa, 2wiki |
| WQ-hotpotqa | musique, 2wiki, musique-3hop, musique-4hop |

The rule takes each read's call from the stage's own calls.json row: the same base, floor (max(0.0075, 2 × the base's
spread) for a screen) and delta.

- **Screen.** A LOSS among the six gives NOT_ADOPTED; otherwise a GAIN gives ADOPT_FOR_FULL_RUN; otherwise NO_GAIN.
- **Full run.** ADOPT needs no LOSS among the six and a GAIN on one of the four dataset reads, as in LTG's decide_full.
  Anything else is NOT_ADOPTED.
- **Seed-0 stop.** A LOSS on WQ-musique hotpotqa or 2wiki is STOP. As in s0stop, a stale record, a problem or a missing
  read is GO.
- **Missing reads.** A LOSS decides even when another read is missing. Any other verdict with a read missing is
  INCOMPLETE, and so is any verdict when the stage's calls report a problem.
- **Beside.** webqsp, squad, the zero-shot reads (metaqa, held-out musique and hotpotqa) and the WQ-musique fits' musique
  slices are still reported. They decide nothing.

The rule covers every screen, full-run and stop verdict filed from this commit until the freeze (00:00 IST, 7 November),
for both the GNN and the MLP stages.

**Both verdicts are filed.** Each stage still writes its own calls.json under its own rule (the journal's six datasets);
zwww reads it and writes `outputs/www/<stage>.json`, which holds both, `www` and `own`, for the screen, the full run and
the stop. Every filing from this commit reports both. The WWW verdict decides the queue until the freeze. After it, the
journal's work (six datasets and generalization) reads the own verdicts again, and its fits take new item names.

## 3. Filed verdicts under the rule

`zwww.py audit` re-reads every filed record. Three verdicts change:

| stage | own | under the WWW rule | what follows |
|---|---|---|---|
| GNW | seed-0 STOP (webqsp −0.0124) | GO: hotpotqa −0.0043, 2wiki +0.0007, both WITHIN | its WQ-hotpotqa seed-0 fit runs |
| GNT | seed-0 STOP (webqsp −0.0108) | GO: hotpotqa −0.0014, 2wiki +0.0024, both WITHIN | its WQ-hotpotqa seed-0 fit runs |
| DEP | ADOPT_FOR_FULL_RUN (metaqa +0.0594) | NO_GAIN: the six WITHIN | its full run waits for the journal; RTR and IRA open (section 4) |

Nothing else changes:

- LTM stays ADOPT: its GAINs are 2wiki and musique-3hop.
- LTG, LTA, R2P and both RWT forms stay NOT_ADOPTED: each lost a musique read.
- GNX, LTC, LTS and MPP stay NO_GAIN.
- GNA, GNP, GNR and LTX stay GO.

DEP's own verdict stands. Its full run is deferred to after the freeze, not withdrawn: metaqa is the journal's, and DEP
is its first generalization candidate. GNT's beside reads point the same way: on its WQ-musique seed-0 fit, metaqa +0.0444
and musique, held out there, +0.0098, both GAIN.

These three are re-read after their numbers were seen. The rule changes which reads decide; the reason, the paper's
scope, comes from the draft, not from these numbers. It only removes reads, so no WWW read's call gets better. The webqsp
and metaqa calls stay in the records beside the rule. A form that passes a screen still needs its full run (seeds 1 and
2) under the same rule. The paper's numbers are B1 reads, and no selection reads B1.

## 4. The queue

These feeder items are changed in place, before they are sent:

- **Stop wrappers.** `gnw-mw-s0-hp`, `gnt-at-s0-hp`, `gns-sa-s0-hp`, `rtr-re-r2-s0-hp` and `ira-re-k4-s0-hp` run their
  fit through `zwww.py s0stop-run --stage <stage> --` in place of the stage's own wrapper. The command after it does not
  change.
- **Screen gates.** `gna-gate`, `gnp-gate`, `gnw-gate`, `ltx-gate`, `gnr-gate`, `gnt-gate`, `gns-gate`, `rtr-re-gate`,
  `ira-re-gate`, `lth-gate` and `lt2-gate` become `zwww.py gate --stage <stage> --want ADOPT_FOR_FULL_RUN`.
- **DEP's full run.** `dep-l4-s1`, `-s2`, `-s1-hp` and `-s2-hp` also wait on a new item, `dep-www-gate`, which runs
  `zwww.py gate --stage dep --want ADOPT_FOR_FULL_RUN`. DEP is NO_GAIN, so the gate stops and the feeder drops the four
  fits, two of them WQ-hotpotqa fits at 0.75 of the card. The journal's DEP full run comes back after the freeze under
  new names (a dropped name does not come back).
- **The stack.** Its amendment A6 (docs/STK_GNN_STACK.md section 15) applies the rule to the members, the pick, the
  attention form and the rounds' opening.

## 5. What does not change

- The fits, their training sets (webqsp and squad still train), the bases, the floors and the per-read calls.
- Each stage's own calls.json, B1 and Claims 1–3.

The training mix stays until the freeze. Every base, floor and queued fit is on the WQ mix. Dropping webqsp and squad
would re-base every stage (twelve base fits) inside the last three days.

## 6. Self-test

`zwww.py --selftest` passes on the laptop's CPU. Its cases:

- A webqsp LOSS alone decides nothing; a squad or metaqa GAIN alone is NO_GAIN.
- A musique-4hop LOSS beats a 2wiki GAIN.
- A LOSS decides with reads missing; a GAIN with a read missing is INCOMPLETE.
- R2P's `dataset` rows are read.
- A full run needs a GAIN on a dataset read: a slice GAIN alone is not enough, and a squad LOSS does not count.
- The stop: GO on webqsp's LOSS, STOP on hotpotqa's or 2wiki's, GO when stale, on a problem or with a read missing.
- The wrapper runs or skips its command, and the gate gives codes 0, 1 and 2.
- The audit on RWT's two-form record.

## 7. Amendment, 6 November 2026: a block's slices rows are read

`zwww.py` read a block's `reads` and `beside` rows. IRA's full run files its musique-3hop and musique-4hop rows (both
splits, four rows) under `slices`, so two of the six deciding reads were not found, and its WWW full-run verdict printed
INCOMPLETE (f5536e9 filed it so; read by hand it is NOT_ADOPTED: 6 WITHIN, no LOSS, no GAIN). `rows_of` now reads
`slices` as well. The six reads, the calls and the rule are as declared above; the WQ-musique slices stay beside.

A survey of the 24 filed `calls.json` files (30 blocks) found one block with a `slices` key, IRA's full run, and no other
block whose verdict or missing reads change. The self-test gained a case for a block that keeps its slices under `slices`:
read whole, a slice LOSS decides, and without them the reads are missing. `outputs/www/ira.json` is regenerated after
this commit; the expected reading is WWW NOT_ADOPTED, own NOT_ADOPTED, six WITHIN, none missing.
