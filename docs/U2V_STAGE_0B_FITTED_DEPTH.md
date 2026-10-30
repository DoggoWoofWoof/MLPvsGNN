# U2v stage 0b: can a fitted combination of the stored per-question measures predict a question's needed depth?

Declared 30 October 2026, before any number of this stage. Stage 2 of [PROGRAM_2026_10_30.md](PROGRAM_2026_10_30.md);
the cheap, no-GPU first look at the latent depth of section 2 there. Code: `outputs/mp_unified/zu2v0b.py`. Outputs: only
under `outputs/u2v/stage0b/`. No filed array, pool, fit or record changes. Nothing here is adopted.

## 1. The question

[U2T_STORED_ARRAY_DIAGNOSTIC.md](U2T_STORED_ARRAY_DIAGNOSTIC.md) sections 4 and 7 found that no single stored per-question
measure predicts the rows a question's golds need with one sign on four datasets (best |rho| 0.43), and that the dense top
score is the best single first-stage signal (tail AUC 0.64 to 0.66). Every one of those reads was one measure at a time and
parameter-free, because "pools first, parameter-free" held learning back. The user relaxed that for declared stages on
30 October. This stage fits one linear combination of all of them and asks whether it predicts depth on a dataset it was
not fitted on. If it does not, stages 5 to 7 of the program (a learned selector and its EM) cannot count on the stored
measures and must bring a new signal (the model's scores, the question's content). If it does, it is the baseline those
stages must beat.

## 2. What is read

- **Datasets and carves:** metaqa, squad, musique, hotpotqa, 2wiki on s1sel; webqsp on s1eval, beside (it never moves a
  choice). Weights are U2s's (`outputs/u2s/verdict.json`, `weights`).
- **The depth y:** a question's needed rows beyond B on U2s's rrf50 + none cell, kind_t, order `kt` (cell 0, `M.column(0, 0, kt)`
  as `predict_need.py` reads it), over the questions whose golds are all reached. y = log(1 + need).
- **Features (pre-scoring, gold-free):**
  - S: the stored columns of `outputs/u2s/stats/<dataset>__<carve>.npz`: the 30 `mq` columns, `n_n1`, `n_r0`, `taken`,
    `n_n2`, `n_u` at 0, 20, ..., 140, `nw` column 0 and `bs` column 0, exactly the set `predict_need.py` read.
  - F: the fourteen first-stage measures of `outputs/u2t/diag/fs_probe.py` (dense and SPLADE scores, gaps, spreads,
    agreement), as filed in 9114a11. Their arrays are local-only; their sha256 are recorded in the result.
  - The verdict reads S + F. S alone is shown beside and decides nothing.
- **The transform:** each feature column is rank-normalized within its own dataset to (rank - 0.5) / n, ties averaged. No
  label is read, the rule is the same on every dataset, and a threshold learned on four datasets means the same on the fifth.

## 3. The predictor and the cut

- **Fit:** ridge regression of y on the rank-normalized features with an unpenalized intercept, penalty 10.0, closed form,
  on the reachable questions of the fitting datasets. Nothing is tuned.
- **Budget:** a question's budget is `K_q = ceil(exp(y_hat + c) - 1)`, clipped to [0, 30,000]. One offset `c` is shared by every
  question and every dataset, from {-1.0, -0.5, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0}.
- **The control:** the same grading with K the same for every question, from U2s's FIX budgets (25 to 20,000). It is the
  fixed rule, and it is what the fitted predictor has to beat on the same data and the same folds.
- **Grading:** U2s's own: `column_table` on the stored `pos` and `lens`, over the four cells (rrf50 + none, h1, h2, h3), the
  four orders (kt, kt_n1, kt_cos+n2_30, kt_cos+n2_100) and U2s's ten mass cuts. A choice is the (cell, order, cut, c or K)
  of least weighted rows on the fitting datasets such that every fitting dataset has ALL >= min(level, its ceiling - 0.005).
  Levels 0.90, 0.92 and 0.95.

## 4. The read that decides: leave one dataset out

For each of the five datasets in turn: fit the ridge on the other four, choose the cell, order, cut and offset on those four
as section 3 says, and read the held-out dataset's ALL and rows at that choice. The aggregate is the weighted mean over the
five reads, each under its own fold. Beside, and deciding nothing: the in-sample choice on all five (the fit and the choice
on the same data); webqsp read with the fit on all five; per held-out dataset, Spearman's rho of y_hat against y and the AUC for
"needs 1,000 or more rows"; the ridge's largest standardized weights from the all-five fit.

## 5. The rule (fixed now)

Read at level 0.92. Let Rf and Af be the fitted predictor's weighted leave-one-out rows and ALL, and Rc and Ac the control's.

| verdict | when |
| --- | --- |
| DEPTH_PREDICTABLE | Af >= 0.91, Rf <= 1,953.5 (U2s's cap), Rf <= 0.90 x Rc, and Af >= Ac - 0.005 |
| DEPTH_PARTIAL | Rf <= 0.90 x Rc and Af >= Ac - 0.005, but not DEPTH_PREDICTABLE |
| NO_DEPTH_SIGNAL | otherwise |

The rule reads the leave-one-out aggregate, never the in-sample choice. A feature-set or level other than the two above is
not read for the verdict.

## 6. What each outcome leads to (fixed now)

- **DEPTH_PREDICTABLE:** the fitted combination is distilled to a fixed router (program stage 7's form) and graded in its own
  file; it is the baseline the learned selector (stage 5) must beat at equal rows. The level ruling is put to the user with
  this curve beside the other two.
- **DEPTH_PARTIAL:** stage 5 proceeds with this as its baseline; the cap is not claimed.
- **NO_DEPTH_SIGNAL:** the stored and first-stage measures do not predict depth across datasets, fitted or not. This closes
  only a linear fit on those measures. Stage 5's selector must use a new signal (the ranker's scores, the question's content
  or the walk's mass after a pass), and the level ruling goes to the user as a construction choice (pick O's procedure is the
  default).

## 7. How it runs, and the checks

Laptop, a minute or two. `zu2v0b.py selftest` first: the ridge equals numpy's least squares on a toy, the rank
transform averages ties, the selection and the leave-one-out read pick the cheapest feasible choice and read the held-out
dataset on a hand-built grid, a feature that equals y is recovered, pure noise lands on the control, and on the real arrays
the recomputed need and reach equal `fs_probe`'s stored ones and the control, chosen on all five, costs no less than U2s's
shared figures at the three levels (1,871.2 / 2,354 / 3,311; the control is inside the family U2s took the minimum over).
`zu2v0b.py run` then writes `outputs/u2v/stage0b/result.json` and a log; `zu2v0b.py verdict` applies section 5.

## 8. Not shown, and not claimed

A model's scores as a signal; PKO's pools; the walk's mass after a pass (RRF5T agreement, a mass cut); a nonlinear predictor;
a question-content signal; any change in R@5 or latency. A linear fit that fails does not show depth is unpredictable; it
closes one form on the measures that are stored.

## 9. Results (30 October, run after the declaration; `outputs/u2v/stage0b/`)

**Stage 0b verdict: NO_DEPTH_SIGNAL** (`verdict.json`; S + F, level 0.92). With the declared offsets (-1.0 to 4.0) the fitted
predictor has no choice that reaches the level on the fitting datasets in any of the five folds, at any of the three levels,
so there is no fitted row to compare; section 5 reads that as NO_DEPTH_SIGNAL. The control (one K for every question), leave
one dataset out, weighted: 1,902.9 rows at ALL 0.9574 (level 0.90), 2,414.2 at 0.9636 (0.92), 3,182.1 at 0.9725 (0.95).

**Does it predict at all?** Held-out, per dataset, y_hat against y (Spearman's rho; AUC for "needs 1,000 or more rows"):

| held out | S + F rho | S + F AUC | S rho | S AUC |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 0.317 | 0.742 | 0.139 | 0.654 |
| squad | 0.155 | 0.839 | 0.089 | 0.631 |
| musique | 0.218 | 0.568 | 0.104 | 0.542 |
| hotpotqa | 0.025 | 0.473 | -0.062 | 0.443 |
| 2wiki | 0.106 | 0.627 | 0.187 | 0.569 |

The sign is positive on all five for S + F, with hotpotqa near zero; the size is small, and the first-stage measures add
most of what there is (S alone is near chance on musique and hotpotqa). These held-out figures are no larger than the best
single measures' in-sample ones (|rho| 0.43 metaqa, 0.41 2wiki, 0.18 musique, 0.21 hotpotqa, U2T section 4), so the fit
did not find a combination that predicts better than the best single measure. The ridge's largest weights fall on adjacent mass columns with opposite signs (mq26 and mq27), which is
collinearity, and are not read.

**Beside, written after the numbers, deciding nothing** (`extended_offsets.py`, `frontier_extended.py`; same code and folds,
offsets extended to 9.0): the declared grid was too short for the fitted rule to reach the level. The offsets it needed
were 6 to 8.5 (a budget hundreds to thousands of times exp(y_hat)) on every dataset but musique (4.5 to 5.5), which means the prediction
does not separate the tail and the rule is covering it by giving everyone a large budget. With the offsets extended the
fitted rule reaches every level and costs 6 to 10 percent fewer rows than the control (2,226 against 2,414 at 0.92, 2,991
against 3,182 at 0.95), but its held-out ALL is lower, and compared at matched rows over 18 levels (0.80 to 0.97) the
fitted rule is below the control's held-out ALL at every level, on both feature sets: the gap runs from -0.0014 to -0.0153
(S + F at 0.92: 0.9584 against 0.9614 at 2,226 rows). The saving is a move along the fixed-K frontier, not a better one.
The extended rule still fails section 5 at 0.92 (2,226 rows is above 0.90 x 2,414 and its ALL is 0.0052 under the control's).
Held-out musique stays the weak fold (ALL 0.74 to 0.83 at the offsets chosen on the other four), as in 9114a11.

**What it leads to** (section 6, NO_DEPTH_SIGNAL): the stored and first-stage measures do not predict a question's needed
depth across datasets well enough to beat one K, fitted or not. This closes only a ridge fit on those measures. Program
stage 5 (the recall-trained selector) must bring a new signal: the ranker's scores, the question's content, or the walk's
mass after a pass. Stage 0 of U2v showed the ranker's scores are dataset-bound zero-shot, so that signal is the one stage 5
has to prove. The level ruling goes back to the user as a construction choice; pick O's procedure remains the default and
PKO proceeds on it.

**Not shown.** A quantile (pinball) fit, which suits a cut better than a mean fit and is what stage 5 declares; a nonlinear
predictor; the question's text; the ranker's scores; the walk's mass after a pass; PKO's pools; any change in R@5 or latency.
