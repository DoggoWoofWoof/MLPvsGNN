# BRG5: a learned candidate order, fitted on other questions of the same split (screen)

Declared 28 Nov 2026 19:32 IST, before any number of this page. Screen on the subsets of BRG1 to BRG4; nothing is adopted. Module: `outputs/mp_unified/zbrg5.py` (new; its self-test runs before any row; sha recorded in section 6).

## 1. Why

BRG1 to BRG4 closed the parameter-free levers of the budgeted walk on the sets below the target: the candidate order (support, a ppr-style sum, their fusion, the question's cosine), the scoring window, depth (3 to 5 hops), a beam of 100 and the per-node cap are LEVEL or BELOW the cell's own on every screened set, and none is ABOVE. What the pages also show is headroom: the golds are among the candidates, and a perfect order at the cell's own frontier would give (points over ALL, BRG2 and BRG4): Prime +9.8, WebQSP +5.4, MAG +5.0, MuSiQue +3.1, Amazon +0.8, and MetaQA +7.0 at a per-node cap of 50.

The user's ruling of 26 Nov was to push the rules as far as they go and then offer one model that walks and ranks together as the alternative. The four pages say the rules are at their end for the order. This page is the cheapest form of the alternative: no change to the encoder, no GPU, no new graph. A scorer is fitted on the golds of other questions of the same split and orders each hop's candidates by features that already exist in the walk: the question (the cosine of the candidate and of its parents), and the edge (the family that names it, its place in the parent's row, how many distinct parents name it, where those parents stand). The question it answers is whether the headroom is reachable by a scorer that sees the question and the edge together, which no single key of the earlier pages does.

It is not a model of record and not one of the two families (MLP, GNN): it is a diagnostic fitted per set on a few hundred questions, to decide whether the joint stage is worth its GPU time and which inputs it needs.

## 2. What runs

On five sets with a budget-following cell (the cells of BRG1, `B.CELL`): MuSiQue, WebQSP, Prime, MAG, Amazon. MetaQA's cap walk follows by an amendment after this read.

- **Questions.** Evaluation = the three chunks at positions 0, 1, 2 of the set's seeded chunk order (the 384 questions of BRG1 to BRG4). Training = the chunks at positions 3, 4, 5 (up to 384 further questions). The two are disjoint questions of the same split (G-L1); a fit on the set's own train split is a later step if a variant is carried.
- **Candidates and labels.** The cell's control walk on a training question gives, at every hop, the distinct fresh candidates in the control order. A candidate is a positive if it is a gold node of the question. Negatives are subsampled uniformly (seed 20261128) to at most 400,000 rows per hop (a uniform subsample of negatives shifts the logit by a constant and leaves the order unchanged).
- **Features (12, per candidate and hop, all from the walk's own lists).** (1) log(1 + place in the control order); (2) the cosine of the candidate to the question; (3) its place by cosine among the hop's candidates, as a share of the list; (4) log(1 + the number of distinct frontier nodes that name it); (5) the sum over those frontier nodes of 1 / (RRF_C + the node's place in the frontier + 1); (6) log(1 + the smallest place any parent's row gives it); (7) the number of distinct families that name it; (8) the largest cosine to the question among its parents; (9) feature 2 minus feature 8; (10) log(1 + the best frontier place among its parents); (11) log(1 + the number of the hop's candidates); (12) log of the hop's budget over the number of candidates.
- **Estimators (two, one fit per hop).** `lr`: logistic regression on standardized features (scikit-learn, L2, C = 1, 200 iterations). `gb`: histogram gradient boosting (scikit-learn, 100 iterations, depth 4, learning rate 0.1, no early stopping, seed 0). A hop with fewer than 20 positives in training keeps the control order (reported).
- **Variants.** `control` (the cell's order), `lr`, `gb`. The walk is the cell's: the same seeds, hops, window, budget per hop and visited cap; at each hop the distinct fresh candidates are ordered by the estimator's score, highest first, ties to the control order, and the first `budget` are kept. The pool is the head and the expansion, graded as in BRG2.
- **Read (as BRG2).** ALL per variant, the paired difference to the control with a bootstrap interval (2,000 resamples, seed 20261128) and the call ABOVE / LEVEL / BELOW at the 1.0-point line (`zdrp2.gain_call`); mean pool size and its ratio to the control's; the ceiling (golds among the head and every hop's candidates); per hop the golds among the candidates and among the kept; by gold count. Beside them, per hop, the area under the ROC curve of the control place (negated), `lr` and `gb` over the evaluation questions' candidates of the control walk (all positives and a uniform subsample of at most 100,000 negatives per hop per chunk).

## 3. Gates (no number is read unless all hold)

- **G-L0.** The `control` variant equals the engine's expansion and grade on every evaluation question (0 mismatches).
- **G-L1.** The training and evaluation questions are disjoint, at least 2 training chunks exist, and the positives collected in training equal the golds among the control walk's candidate lists (recounted from the lists).
- **G-L2.** Every feature is finite on every collected row.

## 4. What the read may say

A call per set and estimator: ABOVE, LEVEL or BELOW the control. A set is `reached` if the better estimator is ABOVE with a mean pool no more than 2 percent above the control's. A `reached` set is a screen result on a held-out part of the same split: it carries nothing. A carried result would need (a) a fit on the set's own train split, (b) a whole-split read of the evaluation split, and (c) the same recipe (features and estimator) on all ten datasets with the same gates; the weights are fitted per set, which is why this is a diagnostic and not a rule. If no set is `reached`, the headroom is not reachable by these inputs and the joint stage needs inputs beyond them (the vectors of the candidate and its parents, not their cosines alone); that is then the first design question for it.

## 5. Priors (written before any number of this page)

My expectation on the numbers is honestly uncertain: the control order is one of the inputs of both estimators, so neither should fall far below it, but the single keys BRG2 tried each fell 4 to 11 points on the sets with the most headroom.

- **L1.** On at least one of Prime, MAG and WebQSP (the three sets with ceilings of 5 to 10 points), `gb` is ABOVE the control.
- **L2.** `gb` is at least LEVEL (never BELOW) on all five sets.
- **L3.** On MuSiQue and Amazon neither estimator is ABOVE (the ceilings are +3.1 and +0.8; MuSiQue's lever is links).
- **L4.** On every set and hop with at least 50 positive candidates in the evaluation sample, `gb`'s area under the curve is above the control place's.
- **L5.** Both estimators' mean pool is within 2 percent of the control's on every set.
- **L6.** G-L0, G-L1 and G-L2 hold on all five sets.

## 6. Rows

Five streams on the host's CPU through `rx exec` (`screen --kb prime`, `--kb mag`, `--kb amazon`, `--ds musique`, `--ds webqsp`), after the self-test and a smoke of 8 questions. Training matrices are saved per set under `outputs/brg5/<name>/` so a restart does not collect them again. No new row enters the feeder. Module sha256 prefix: recorded in section 7 when the rows start.

## 7. Results (28 Nov 2026 20:03 IST)

`screen` on the five sets, module sha256 prefix `bfbd4612f8d77c3b` (local = host; committed a1b12e03), self-test PASS on the laptop and the host, smoke of 24 questions on WebQSP before the rows. G-L0 PASS on all 15 evaluation chunks (0 mismatches between the control walk and the engine's expansion and grade), G-L1 PASS (training chunks at positions 3 to 5, disjoint from the evaluation chunks at 0 to 2; 0 recount mismatches), G-L2 PASS (0 non-finite rows). Records: `outputs/brg5/<set>/gate_screen.json`, `fit.json`, `scr_c*.npz`, `auc_c*.npz` and `outputs/brg5/read_<set>.json` (the training matrices stay on the host). Screen reads on 369 to 384 questions; the training questions are other questions of the same split.

| set | variant | ALL | golds in the pool | mean nodes (over control) | ceiling ALL | minus control [95% interval], call |
| --- | --- | --- | --- | --- | --- | --- |
| Prime (369 questions) | control | 86.7 | 89.5 | 2,076 | 96.5 | |
| | `lr` | **90.2** | 92.3 | 2,075 (1.000) | 96.7 | **+3.52 [+1.63, +5.69] ABOVE** |
| | `gb` | 85.6 | 89.1 | 2,124 (1.023) | 96.7 | -1.08 [-3.52, +1.08] LEVEL |
| MAG | control | 92.4 | 93.2 | 2,045 | 97.4 | |
| | `lr` | 93.5 | 94.9 | 2,046 (1.001) | 97.7 | +1.04 [+0.00, +2.34] LEVEL |
| | `gb` | 88.5 | 92.4 | 2,108 (1.031) | 96.9 | -3.91 [-6.25, -1.82] BELOW |
| WebQSP | control | 87.8 | 63.0 | 2,097 | 93.2 | |
| | `lr` | 89.6 | 73.4 | 2,102 (1.003) | 92.7 | +1.82 [-0.26, +3.91] LEVEL |
| | `gb` | 82.0 | 60.6 | 2,113 (1.008) | 93.0 | -5.73 [-8.59, -3.12] BELOW |
| MuSiQue | control | 86.2 | 95.0 | 2,467 | 89.3 | |
| | `lr` | 85.4 | 94.8 | 2,467 (1.000) | 89.3 | -0.78 [-2.08, +0.26] LEVEL |
| | `gb` | 83.3 | 94.0 | 2,481 (1.006) | 89.3 | -2.86 [-4.69, -1.30] BELOW |
| Amazon | control | 85.7 | 93.2 | 2,440 | 86.5 | |
| | `lr` | 85.9 | 93.2 | 2,440 (1.000) | 86.5 | +0.26 [+0.00, +0.78] LEVEL |
| | `gb` | 85.9 | 93.2 | 2,441 (1.000) | 86.5 | +0.26 [+0.00, +0.78] LEVEL |

Share of the golds among a hop's candidates that the walk keeps (control to `lr`, hop 2 / hop 3): Prime 83.5 to 89.0 / 41.3 to 58.2; MAG 83.0 to 89.3 / 39.7 to 46.3; WebQSP 70.2 to 95.0 / 25.9 to 43.9; MuSiQue 95.2 to 94.2 (no hop 3); Amazon 99.1 to 99.3 (no hop 3). `gb` keeps fewer than the control at hops 2 and 3 on Prime (74.6 / 29.5), MAG (71.5 / 11.8) and WebQSP (62.3 / 22.8), and at hop 2 on MuSiQue (87.6).

Area under the ROC curve of the order over the evaluation candidates of the control walk (control place / `lr` / `gb`; the positives per hop are 75 to 1,066):

| set | hop 1 | hop 2 | hop 3 |
| --- | --- | --- | --- |
| Prime | 0.909 / 0.917 / 0.900 | 0.921 / 0.942 / 0.842 | 0.846 / 0.906 / 0.759 |
| MAG | 0.902 / 0.946 / 0.934 | 0.954 / 0.975 / 0.830 | 0.858 / 0.904 / 0.496 |
| WebQSP | 0.702 / 0.733 / 0.808 | 0.927 / 0.978 / 0.803 | 0.853 / 0.894 / 0.770 |
| MuSiQue | 0.908 / 0.910 / 0.845 | 0.966 / 0.966 / 0.921 | none |
| Amazon | 0.839 / 0.912 / 0.892 | 0.959 / 0.974 / 0.945 | none |

**Calls on the priors.**

- **L1 NOT HELD as worded.** `gb` is LEVEL on Prime (-1.08) and BELOW on MAG (-3.91) and WebQSP (-5.73). The estimator that is ABOVE is `lr` (Prime +3.52 [+1.63, +5.69]); I had expected the trees to be the stronger.
- **L2 NOT HELD.** `gb` is BELOW on MuSiQue (-2.86), MAG (-3.91) and WebQSP (-5.73).
- **L3 HELD.** On MuSiQue (`lr` -0.78 LEVEL, `gb` -2.86 BELOW) and Amazon (+0.26 LEVEL for both) neither estimator is ABOVE.
- **L4 NOT HELD.** `gb`'s area is below the control place's at hop 2 on Amazon (0.945 against 0.959), MuSiQue (0.921 against 0.966), Prime, MAG and WebQSP, at hop 3 on Prime, MAG and WebQSP, and at hop 1 on MuSiQue (0.845 against 0.908). `lr`'s area is above the control's at every hop on Prime, MAG, WebQSP and Amazon and equal on MuSiQue (0.910 and 0.966 against 0.908 and 0.966).
- **L5 NOT HELD for `gb`.** Its pool is 1.023 times the control's on Prime and 1.031 on MAG (within 1.02 expected); `lr`'s is within 1.003 everywhere.
- **L6 HELD.** G-L0, G-L1 and G-L2 hold on all five sets.

**By the page's own rule (section 4)** one set is `reached`: Prime with `lr`. MAG (+1.04, lower end 0.00) and WebQSP (+1.82, lower end -0.26) are positive and LEVEL. Over the control's own ceiling (ceiling ALL minus control ALL: Prime 9.8, WebQSP 5.4, MAG 5.0), `lr` closes about 36, 34 and 21 percent of it (my arithmetic).

**What it says.** A linear scorer on the 12 features of the walk, fitted on 3 chunks of other questions, orders the hop's candidates better than the cell's own order everywhere the budget binds (area under the curve up at every hop of Prime, MAG, WebQSP) and, at the same pool size, lifts ALL on Prime by 3.5 points (ABOVE), MAG by 1.0 and WebQSP by 1.8 (LEVEL). The boosted trees fall below the control wherever the training rows are few (about 70 to 1,000 positives a hop), the usual overfit. MuSiQue and Amazon gain nothing, as their misses are links and a lexical first stage. This is the first lever of BRG1 to BRG5 that is ABOVE at an unchanged pool size; it is a screen on a held-out part of the same split with 369 questions, fitted per set, and carries nothing (section 4). The question it leaves is whether more training questions and a regularized non-linear form lift MAG and WebQSP to ABOVE: Amendment 1.

## 8. Amendment 1 (28 Nov 2026 20:03 IST): more training questions, and a regularized non-linear form

Declared after section 7 and before any number of this amendment. A new module `outputs/mp_unified/zbrg5b.py` imports `zbrg5.py` (not edited: its sha is `bfbd4612f8d77c3b`); its self-test runs before any row and its sha is recorded in section 9.

**What runs.** The same five sets, evaluation chunks and gates as section 2 (G-L0 on the `control` variant), five variants: `control`; `lr` (the filed fit on the training chunks at positions 3 to 5); `lr6` (the same estimator fitted on the chunks at positions 3 to 8 that exist, at least 3 and up to 6, the negatives subsampled per chunk as before); `lr2` (logistic regression, C = 1, 200 iterations, on the 12 features and all their 78 pairwise products and squares, standardized before and after the expansion, fitted on positions 3 to 5); `lr26` (the same on positions 3 to 8). A hop with fewer than 20 positives keeps the control order. **G-L3:** the `lr` variant of this run reproduces section 7's `lr` row of every question (the golds in the pool and the pool size equal on all questions of all sets), else no number is read. The read adds, for `lr6`, `lr2` and `lr26`, the paired difference to `lr` with the same bootstrap and calls.

**Priors (before any number of this amendment).**

- **M1.** `lr6` is at least LEVEL against `lr` (no lower than -0.5 point) on every set, and ABOVE the control on at least one of MAG and WebQSP.
- **M2.** `lr2` is not more than 1.0 point above `lr` on any set (the trees' fall says the non-linear forms overfit at this size).
- **M3.** On Prime, `lr6` and `lr26` stay ABOVE the control.
- **M4.** On MuSiQue and Amazon none of the four new variants is ABOVE the control.
- **M5.** Every variant's mean pool is within 2 percent of the control's.
- **M6.** G-L3 holds and the gates of section 3 hold.

## 9. Results of Amendment 1

None yet.
