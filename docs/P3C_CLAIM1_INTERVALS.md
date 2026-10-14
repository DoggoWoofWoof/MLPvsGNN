# P3c: Claim 1 for the lean pair, with intervals

Declared 14 October 2026, before any number was read. A read of filed fits: nothing trains, nothing is adopted.
Script: `scripts/p3c_claim1.py`. Record: `outputs/p3c/claim1.{json,md}`.

## 1. Why

Claim 1 (docs/PAPER_CLAIMS_2026_10_09.md) says the MLP keeps at least 90% of the GNN's R@5 gain over the first
stage, "with a 95% bootstrap interval", on each dataset in-domain, and that zero-shot it is not BELOW the GNN by more
than the in-domain gap. Since 14 October the lean pair is (zrckp, zspk) (docs/P1C3P_FULL_RUN.md). Its shares so far
are point estimates from the J5 comparisons' means (0.93 to 1.29 on R@5). This stage gives every share its interval
and reads both clauses, on the same per-question reads the full runs filed.

## 2. What is read

**The fits** (seed 0, candidate p@swa, P1c2's pools with the KT input), one per model and split:

| split | zrckp (the MLP) | zspk (the GNN) |
| --- | --- | --- |
| J5, L-2wiki, L-squad, L-metaqa | `outputs/full_zrckp/fits/S` | `outputs/full_zspk/fits/S` |
| L-musique | `outputs/screen/fits/scr-p1c3p-zrckp` | `outputs/screen/fits/scr-p1c3-zspk` |
| L-hotpotqa | `outputs/screen/fits/scr-p1c3p-zrckp-hp` | `outputs/screen/fits/scr-p1c3-zspk-hp` |

**The reads:** each fit's `reads/<dataset>__s1eval.npz` on the six datasets: per question, the candidate's top-5 gold
count and top-1 hit, and the same for RRF (the first reference, `rrf`). Measures are lean_gpu's metrics_of (R@5,
FC@5, hit@1) over the questions with at least one gold, as in every comparison.

**The checks come first.** The stage refuses, and writes nothing, unless:
- every fit's `screen.json` names the declared arm, split and seed 0;
- every read's means (R@5, FC@5, hit@1 to four decimals), RRF's means and its question count equal its filed
  comparison's row: `outputs/full_<arm>/compare-S.json` for the full runs' splits and
  `outputs/screen/<screen>{,-hp}.json` for the screens' (each record's deciding base);
- on each split and dataset, the MLP's and the GNN's reads hold the same questions, in the same order, with the same
  gold counts and the same RRF columns;
- on each trained dataset, its J5 read and its leave-out read hold the same questions.

## 3. The rule, fixed before any number

**The bootstrap.** For each read, 2,000 resamples of its questions with replacement, drawn exactly as lean_screen's
comparison draws them (`default_rng(seed)`, chunks of 100, seed = the dataset's index in metaqa, squad, musique,
hotpotqa, 2wiki, webqsp). Every quantity of a read uses the same resamples for the MLP, the GNN and RRF; the
zero-shot clause uses the same resamples for a dataset's J5 and leave-out reads. Intervals are the 2.5% and 97.5%
order statistics of the 2,000 values (no interpolation).

**The share** on a measure: (MLP − RRF) / (GNN − RRF), from the means. A resample where the GNN's gain is 0 or below
gives −infinity (the GNN has nothing to keep, which counts against the claim).

**The in-domain clause** (R@5, the J5 read of each of the five trained datasets; webqsp never trains, so it has no
in-domain read):
- **NO_GNN_GAIN** if the GNN's gain over RRF is at most 0.002 (P3a's margin) or its interval does not lie above 0;
- **HOLDS** if the share's interval lies at or above 0.90;
- **NOT_SHOWN** if the share is at least 0.90 but its interval reaches below 0.90;
- **MISSES** if the share is below 0.90.

**The zero-shot clause** (R@5):
- On each trained dataset D, its leave-out read (L-D): the zero-shot gap is the GNN's mean minus the MLP's there; the
  allowed gap is the same difference on D's J5 read, floored at 0 (an MLP above its GNN in-domain is allowed no
  zero-shot shortfall); the excess is the first minus the second.
- On webqsp, J5's read (zero-shot, as on every split): the allowed gap is 0.
- **BELOW** if the excess is more than 0.002 and its interval lies above 0; **HOLDS** otherwise.

**The call:**
- **CLAIM1_HOLDS:** the in-domain clause HOLDS on all five trained datasets and the zero-shot clause HOLDS on all six.
- **CLAIM1_NOT_ALL:** otherwise. The record names every dataset and clause that is not HOLDS, with its numbers. As the
  claims doc fixes, a dataset that misses the bar is reported as missing it.

**Reported beside, deciding nothing:** the FC@5 and hit@1 shares, with intervals, on every read; the in-domain shares
on the five leave-out fits; webqsp's zero-shot gap on the five leave-out fits under the same rule.

## 4. What it means

- These are development reads on s1eval, the carve the screens and full runs read. The paper's numbers come from the
  confirmation run, on questions never used for selection.
- The universal pair (twin0, gnn0) is not read here: it has no leave-out fits (an open issue in
  docs/PROJECT_TIMELINE.md section 6). Its shares stay P3a's point estimates.
- Whatever the call, it changes no model, pool or input.

## 5. Running it

```
python scripts/p3c_claim1.py --selftest
python scripts/p3c_claim1.py run --limit 50      # development: steps and checks only, no number
python scripts/p3c_claim1.py run                 # the record
```

It runs on the host, where the reads are (about 13 MB per fit), on one CPU. The self-test covers the calls on
made-up statistics, the share's −infinity rule, exact shares of 1 and 0, the resamples equal to lean_screen's, the
zero-shot floor at 0, a whole run on made-up reads, and the refusals (a filed mean the read does not reproduce, a
wrong arm, different questions).

## Results

Read 14 October, 21:22 (one job, 41 s; record stamped 15:52:38 UTC; script sha256 9898a91e…). Every check held:
72 reads (36 per model), every fit's stamp, and every read reproduces its filed comparison.

**Call: CLAIM1_NOT_ALL.** The zero-shot clause HOLDS on all six datasets. The in-domain clause HOLDS on three of the
five trained datasets (metaqa, hotpotqa, 2wiki) and is **NOT_SHOWN** on two (squad, musique): their shares are above
0.90, but their intervals reach below it. No dataset MISSES.

| dataset | questions | GNN's gain over RRF [95%] | MLP's share [95%] | in-domain | zero-shot excess [95%] | zero-shot |
| --- | ---: | --- | --- | --- | --- | --- |
| metaqa | 9,785 | +0.7964 [+0.7896, +0.8027] | 0.997 [0.996, 0.998] | HOLDS | −0.0389 [−0.0447, −0.0332] | HOLDS |
| squad | 11,873 | +0.0045 [+0.0012, +0.0077] | 1.302 [0.859, 2.824] | NOT_SHOWN | +0.0019 [−0.0006, +0.0043] | HOLDS |
| musique | 2,417 | +0.1065 [+0.0965, +0.1160] | 0.933 [0.881, 0.983] | NOT_SHOWN | −0.0181 [−0.0266, −0.0097] | HOLDS |
| hotpotqa | 7,405 | +0.2198 [+0.2137, +0.2259] | 1.010 [0.999, 1.020] | HOLDS | −0.0014 [−0.0045, +0.0016] | HOLDS |
| 2wiki | 12,576 | +0.3272 [+0.3226, +0.3319] | 0.986 [0.981, 0.991] | HOLDS | +0.0012 [−0.0015, +0.0039] | HOLDS |
| webqsp | 1,503 | (never trains) | | | −0.0698 [−0.0870, −0.0523] | HOLDS |

The zero-shot excess is the GNN's lead over the MLP on the held-out read minus its lead in-domain (floored at 0); a
negative excess means the MLP did better zero-shot than its in-domain gap allows. The full tables are in
`outputs/p3c/claim1.md`.

**What it says:**
- **squad:** the first stage already reaches 0.9054 and the GNN adds 0.0045. A share of a gain that small is a ratio of
  two small numbers, so its interval runs from 0.86 to 2.8. The MLP's mean is above the GNN's there (0.9112 against
  0.9099); what fails is the interval, not the point. The clause's rule does not change after the read: squad is
  reported as NOT_SHOWN.
- **musique:** the MLP keeps 93% of the GNN's gain on these 2,417 questions, but as little as 88% on resamples. It is
  where the MLP sits furthest below its GNN in-domain (0.5728 against 0.5800). On the four leave-out fits that train
  on musique, its share is 0.943 to 1.052, and on FC@5 its J5 share is 0.990 [0.918, 1.065].
- **Zero-shot:** the MLP is never below the GNN by more than its in-domain gap allows. It is above the GNN on metaqa
  held out (by 0.037) and on webqsp (by 0.070). The largest excess, squad's +0.0019, is under the 0.002 margin and its
  interval reaches below 0. On the five leave-out fits, webqsp's would-be calls all HOLD as well.

**Beside, deciding nothing:** hit@1 is the MLP's weak measure on hotpotqa: its share is 0.844 [0.480, 1.227] on J5 and
0.46 to 0.64 on three leave-out fits (L-musique, L-squad, L-metaqa). This is the hit@1 gap already on the ledger
(docs/PROJECT_TIMELINE.md section 6). On 2wiki the hit@1 share is 0.915 [0.893, 0.936].

**What it changes:** nothing. No model, pool or input changes. Claim 1 is reported as written: in-domain it holds on
three of five datasets with intervals, and is not shown on squad and musique; zero-shot it holds on all six. These are
development reads. The confirmation run reads both clauses again on its own questions.
