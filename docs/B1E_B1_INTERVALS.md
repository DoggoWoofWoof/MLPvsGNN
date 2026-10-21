# B1e: on B1, how much of the GNN's gain the MLP keeps, with intervals

Declared 21 October 2026, before any number. This stage reads B1b's filed reads and nothing else: nothing trains,
nothing is adopted, and no model, pool, graph or read changes.

## Why

B1b (docs/B1B_MODELS_ON_HIPPORAG2.md) put our two lean models on HippoRAG 2's three settings: 1,000 questions each
from MuSiQue, 2Wiki and HotpotQA, each over its own small corpus. It filed each model's R@5 with an interval against
the published bar. The MLP's share of the GNN's gain is filed as a point only (`mlp_share_of_gnn_gain` in
`outputs/b1b/report.json`).

Claim 1 (docs/PAPER_CLAIMS_2026_10_09.md) is about that share and asks for a 95% interval. P3c
(docs/P3C_CLAIM1_INTERVALS.md) gave the interval on s1eval, for the later pair (zrckp, zspk). Nobody has given it for
the pair B1b read (zrc, the MLP; zsp, the GNN) on B1's questions.

B1e fills that gap with P3c's estimator, unchanged, so a write-up of B1 can quote an interval instead of a point.

## What it reads

- **The reads:** `outputs/b1b/fits/{zrc,zsp}/{J5,L-musique,L-2wiki,L-hotpotqa}/reads/{musique,2wiki,hotpotqa}__b1.npz`.
  That is 24 reads. Each read uses candidate p@swa and its own first-stage reference (`rrf`, the read's first `refs`
  entry).
- **The checks.** Any failure stops the stage before a number is computed.
  - **B1a's questions.** Every read holds B1a's 1,000 questions, with B1a's gold counts, in B1a's order
    (`outputs/bench/hipporag2/<setting>/queries.json`).
  - **B1b's means, where B1b filed them.** B1b filed J5 and each setting's own held-out fit. On those reads, R@5,
    FC@5 and hit@1 must equal `outputs/b1b/report.json` to four decimals, and so must the RRF row.
  - **The other leave-out fits.** On these, the setting is in the fit's training set. B1b ran these reads but did not
    file them, so they are checked against B1a's questions only.
  - **Pairs.** The MLP's and the GNN's reads of one fit and setting must hold the same questions and identical RRF
    columns.

## What it computes

The estimator is P3c's, imported from `scripts/p3c_claim1.py` and not changed:
- `pair_stats` and `excess_stats`;
- the same question resamples for the MLP, the GNN and RRF: 2,000 of them, from `default_rng(seed)`;
- the seed is the setting's place in P3c's `EVAL_ORDER` (musique 2, hotpotqa 3, 2wiki 4);
- the 2.5% and 97.5% points as order statistics.

On each of the 12 (fit, setting) pairs, it gives these, each a mean over B1's 1,000 questions with its interval:
1. **The MLP's share of the GNN's gain over RRF,** (MLP − RRF) / (GNN − RRF), on R@5, FC@5 and hit@1.
2. **The paired gap,** GNN − MLP, on the same three measures.
3. **The gains over RRF** of each model.

**The zero-shot excess.** On each setting, the held-out fit's gap goes beyond the J5 gap by
mean(GNN_held − MLP_held) − max(mean(GNN_J5 − MLP_J5), 0), on R@5. This is P3c's `excess_stats`.

## The calls

These are P3c's rules, imported unchanged (`in_domain_call`, `zero_shot_call`):

- **In-domain clause, per setting** (J5, R@5):
  - **HOLDS** when the share's lower bound is at least 0.90.
  - **NOT_SHOWN** when the point is at least 0.90 but the lower bound is below.
  - **MISSES** when the point is below 0.90.
  - **NO_GNN_GAIN** when the GNN's gain is at most 0.002, or its lower bound is at most 0.
- **Zero-shot clause, per setting** (R@5):
  - **BELOW** when the excess is above 0.002 and its lower bound is above 0.
  - **HOLDS** otherwise.
- **The stage's call:**
  - **B1E_CLAIM1_HOLDS** when all three in-domain clauses and all three zero-shot clauses hold.
  - **B1E_CLAIM1_NOT_ALL** otherwise. The tables say which clause fell.

**Beside, deciding nothing:**
- the shares on FC@5 and hit@1;
- the paired gaps with their intervals;
- every read of the other leave-out fits.

## What it does not do

- It adopts nothing and changes no filed verdict. B1b's ABOVE, AT and BELOW calls against HippoRAG 2 stand as filed.
- It adds no fit, model, pool, graph or read. B1 has only these three settings, so metaqa, squad and webqsp are not
  in it.
- It is not the confirmation run (docs/PROGRAM_2026_10_09.md). These are development numbers, like B1b's.

## Running it

    python scripts/b1e_intervals.py run      -> outputs/b1e/intervals.json, outputs/b1e/intervals.md
    python scripts/b1e_intervals.py --selftest

## Results

Not yet run.
