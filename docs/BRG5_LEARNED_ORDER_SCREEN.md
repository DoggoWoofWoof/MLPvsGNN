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

## 7. Results

None yet.
