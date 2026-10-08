# Two diagnostics of zrm's lost R@5: the bridge (D1) and the feature ceiling (D2)

Declared 8 October 2026, about 14:20, before any of their numbers. Neither trains a retriever or changes zrm, its
fits, the graph, the encoder or any feature: both read zrm's screen fits (`outputs/screen/fits/scr-zrm`, L-musique;
`scr-zrm-hp`, L-hotpotqa) on CPU through zrm's own read, on a copy of each fit's `models.pt` and `screen.json` in a
work folder, so the sealed fit folders are never written. Each verdict says where the next declared screen should
look. A diagnostic is not a result, and no number from it is cited as one.

## Why

The published systems we compare with differ in protocol: they train on the test dataset, KB systems are given the
topic entity, rerankers see a fixed candidate set, and the LLM systems retrieve with larger encoders. Those
differences do not explain all of the gap. Two causes are open, and each has a test that needs no new graph and no
new model:

- **No hop conditioning.** zrm scores each pool row on its own. Where it loses R@5 on musique and 2wiki, it mostly
  finds part of a question's golds: 0.74 to 0.77 of musique's lost R@5, and 0.88 of 2wiki's
  (`outputs/diag/goldsplit-zrm.md`). Multi-hop systems pick the second hop given the first. If the golds zrm misses
  sit next to the ones it finds in the question's pool graph, conditioning could reach them.
- **The scorer or the inputs.** Either zrm does not extract all that its per-row inputs hold (a training gap), or the
  inputs themselves hold no more (a feature limit).

## D1, the bridge (`outputs/mp_unified/bridgediag.py`)

For each of the two fits, on the six s1eval carves, zrm's p@swa scores are kept per row. The pool's edges come from
the look's chunks: structural, NER and kNN, as built for the cache, checked row for row against the cache's ids, pool
sizes and gold flags. Every question's top five must match zrm's read.

Measured over the questions zrm finds partly (at least one gold in its top five, at least one in-pool gold outside):

- **adj:** the share of missed in-pool golds next to a found gold, for any family and per family, and within two hops.
- **lift:** adj divided by the same share among non-gold rows outside the top five.
- **nb rank:** a missed gold's rank by zrm's score among the found golds' neighbours outside the top five.

Measured as R@5 against zrm over every question with a gold:

- **C1 (label-free):** zrm's top three, then the two highest-scored neighbours of its top-1 not yet chosen, then zrm's
  order.
- **C2 (label-free):** zrm's top three, the best neighbour of the top-1, then the best neighbour of the top-2.
- **O1 (oracle hop 1):** C1 anchored on the highest-ranked gold in zrm's top five (zrm's top five when it holds none).
- **O2 (reach bound):** zrm's found golds plus every missed in-pool gold next to one, at most five.

**Identity.** zrm's R@5 here must lie within 0.002 of the fit's filed read (CPU against the card), or the run exits 1
and gives no verdict.

**Rule** (on the multi-hop passage reads, musique and 2wiki, checked in each fit):

- **REACHABLE:** in every fit, on musique or 2wiki, adj is at least 0.5 and lift at least 2.
- **ORACLE PAYS:** in every fit, on musique or 2wiki, O1 is at least +0.02 R@5 over zrm.
- **LABEL-FREE PAYS (per rule, C1 or C2):** in every fit, on musique or 2wiki, the rule is at least +0.0075 over zrm,
  and on none of the fit's six reads is it more than 0.0075 under.

**Verdict:**

| Verdict | When | What follows |
| --- | --- | --- |
| NOT_REACHABLE | not REACHABLE | The missed golds are not linked to the found ones in the pool graph. Conditioning on this graph cannot reach them, and new graphs are ruled out. The next screens go to the inputs (D2). |
| PARAMETER_FREE | REACHABLE and a label-free rule pays | That rule, as a fixed second stage over zrm's scores, is screened on all six datasets. |
| CONDITION | REACHABLE and the oracle pays, but no label-free rule does | A learned score of a row given zrm's top pick, trained on the existing graph (an objective, not a new graph), is declared as a screen on all six datasets. |
| REACHABLE_NOT_RANKED | REACHABLE, but the oracle does not pay | The missed golds are linked, but zrm's score does not single them out among the found golds' neighbours. Conditioning needs a pairwise score before it can help. Reported, and weighed with D2. |

## D2, the feature ceiling (`outputs/mp_unified/gbmdiag.py`)

Gradient-boosted trees: scikit-learn's HistGradientBoostingClassifier, 1.7.1 on the host. The settings are fixed in
the script: 400 iterations at most, learning rate 0.1, 63 leaves, 50 rows per leaf, L2 1.0, early stopping on a tenth
of the rows, seed 0.

- **Training.** Pointwise, on the split's fit carves. Per dataset, up to 8,000 questions with an in-pool gold. Each
  question contributes its in-pool golds, its 20 highest-scored non-golds by zrm and 20 random non-golds; its golds
  weigh 1 in total, and so do its non-golds.
- **Reading.** On the six s1eval carves, against the same split's zrm fit.

There are two variants:

- **G0** sees the cache's 86 columns, their z-scores within the pool against its retrieved rows (lean_screen3's
  reference), and log1p(pool size). This is a subset of zrm's inputs: no SEMB and no chain match. Reported only.
- **G1** sees G0's inputs plus zrm's own score of the row. **It decides.** Boosting from zrm's score finds what the
  per-row inputs hold beyond what zrm takes from them. On the fit carves zrm's score is in-sample, so a G1 gain is if
  anything understated.

**Identity.** As in D1: 0.002 against the filed read.

**Rule** (on the in-domain reads, the four training datasets of each split, eight reads in all):

| Verdict | When | What follows |
| --- | --- | --- |
| TRAINING_GAP | G1 is at least +0.02 R@5 over zrm on at least two of the eight | zrm leaves signal in its inputs. Capacity and objective changes on the same inputs are the next screens. |
| FEATURE_LIMIT | G1 is at most +0.01 over zrm on all eight | A stronger scorer on these per-row inputs gains nothing. The gap is in the inputs, and the next screens are new preprocessing of the existing graph (for example, D1's conditioning). |
| MIXED | otherwise | Reported per read. |

The zero-shot reads are reported but do not decide.

## Runs

These are host CPU items:

- `diag-bridge-zrm` and `diag-bridge-zrm-hp`: 8 CPUs each.
- `diag-gbm-L-musique` and `diag-gbm-L-hotpotqa`: 8 CPUs each.
- Then the two verdicts.

Records go to `outputs/diag/bridge-*` and `gbm-*`, with host paths cut before committing. The work folders
(`outputs/diag/bridge-work`, `gbm-work`) are never committed.

Any latency figure is cold. These diagnostics time nothing.

## D1's result: REACHABLE_NOT_RANKED (filed about 14:15)

`outputs/diag/bridge-scr-zrm.md`, `bridge-scr-zrm-hp.md` and `bridge-zrm.json`. All twelve reads reproduce zrm's filed
R@5 exactly. The changes in R@5 are against zrm. Each pair gives L-musique's fit, then L-hotpotqa's.

| read | adj | lift | 2-hop | missed gold ranked 1st or 2nd among found golds' neighbours | C1 | O1 | O2 (reach bound) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 0.378, 0.375 | 14.7, 13.9 | 0.77 | 0.22, 0.28 | −0.010, −0.011 | −0.014, −0.016 | +0.125, +0.116 |
| 2wiki | 0.773, 0.778 | 6.1, 6.1 | 0.81 | 0.71, 0.69 | −0.020, −0.017 | −0.012, −0.010 | +0.061, +0.063 |
| hotpotqa | 0.931, 0.920 | 6.3, 7.0 | 0.93 | 0.61, 0.57 | −0.006, +0.011 | +0.006, +0.017 | +0.048, +0.098 |
| metaqa | 0.075, 0.075 | 1.4, 1.5 | 0.51 | 0.45 | −0.063, −0.062 | −0.064, −0.065 | +0.003, +0.002 |
| webqsp | 0.175, 0.164 | 6.1, 6.3 | 0.98 | 0.22 | −0.016, −0.002 | −0.055, −0.043 | +0.022, +0.019 |

- **The bridge exists on the passage graphs.**
  - On 2wiki and hotpotqa, 0.77 to 0.93 of the golds zrm misses in partly-found questions sit next to a found gold,
    mostly by structural edges. That is about 6 to 7 times the rate of non-gold rows.
  - On musique, 0.38 are next to a found gold, mostly by NER edges, and 0.77 within two hops. The lift is about 14,
    because few non-gold rows are next to a found gold.
  - The reach bound O2 is +0.06 R@5 on 2wiki, +0.05 to +0.10 on hotpotqa and +0.12 on musique.
- **Fixed re-ranks do not use it.**
  - Even anchored on a known gold (O1), filling ranks 4 and 5 with the anchor's best-scored neighbours loses R@5 on
    musique and 2wiki. Those ranks often already hold golds, and zrm's score does not single the missed gold out
    among the anchor's neighbours on musique (median rank 5 to 9 of about 45).
  - On 2wiki and hotpotqa the missed gold is first or second among them 0.57 to 0.71 of the time. A hard swap still
    costs more than it gains.
- **The KB answer sets are not chains.** On metaqa only 0.075 of missed golds sit next to a found one, with a lift of
  1.4. Conditioning on a found answer does not apply there.
- **What follows (by the rule).** This verdict is reported and weighed with D2. The signal a next screen would use is a
  soft one: a row's link to the rows zrm ranks highly, as a learned input or objective term inside the scorer, not a
  hard swap. It is weighed with D2's verdict before any screen is declared.
