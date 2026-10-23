# U2k: a learned link scorer (step 2 of the hyperlink-gap plan)

Declared 23 October 2026, before any number. It follows [U2J_LINK_PRECISION.md](U2J_LINK_PRECISION.md) and the
user's instruction of 23 October: "yeah do the 3 steps to fix". Code: `outputs/mp_unified/zu2k.py`. Outputs: only
under `outputs/u2k/`. No filed graph, pool, fit or record changes.

## 1. The question

U2j orders W2_S's rows by fixed rules. Step 2 asks whether one score learned from the evidence of a link can tell the
links an editor would choose from the rest, and transfer. The score is fitted to one dataset's hyperlinks and then
applied unchanged to datasets it never saw. It is never fitted to a question, a gold or a target dataset's
hyperlinks. The datasets with hyperlinks in their package graph (PKG) supply the labels; the headline graph stays
LLM-free and hyperlink-free on every dataset it is applied to.

## 2. The features of a directed mention link u → v

The evidence is U2i's pass's, read as U2j reads it (the rebuilt links must equal U2i's `links.npz`, else REFUSED).

| feature | what |
| --- | --- |
| lead | the first mention is in u's first sentence |
| pos | the first mention's offset over u's text length |
| count | log(1 + the mention's occurrences in u) |
| indeg | log(1 + v's W2 mention links in) |
| outdeg | log(1 + u's W2 mention links out) |
| caps | cap / (cap + lower + 1) of the surface, over the corpus |
| capmid | the surface written exactly as titled and capitalized, mid-sentence, in u |
| tokens | min(the surface's token count, 8) |
| mutual | v → u is a W2 mention link too |

U2j's alias flag is left out: it is 0 on every W2 link. Each feature is standardized by its mean and standard
deviation over the dataset's own W2 links (no label is read; the same rule on every dataset).

## 3. The scorers

- **The label** of a directed W2 link on the fit dataset: its pair is in that dataset's PKG.
- **The fit:** logistic regression minimizing the summed log-loss plus ½ · 1.0 · ‖w‖² (the intercept unpenalized;
  scikit-learn's objective at C = 1, which the selftest checks), on a uniform sample of 2,000,000 of the fit dataset's
  W2 links (numpy RandomState(0), without replacement; all links if fewer). Newton's method until no weight moves by
  more than 1e-8, at most 50 steps. Nothing is tuned.
- **Scorer A** is fit on 2wiki and **scorer H** on hotpotqa.
- **τ (the prune threshold):** the fit dataset's score at its (1 − p) quantile, where p is the share of its W2 links
  that are hyperlinked. It keeps about as many links as the hyperlinks among them, and is applied unchanged elsewhere.
- **Which scorer grades which dataset (always zero-shot):** hotpotqa, musique and squad are graded with A; 2wiki is
  graded with H. Each scorer on its own fit dataset is reported beside, as an in-domain upper bound.

## 4. The graph forms

Over W2_S, with s(u, v) the higher of the scores of the pair's directed W2 links:

| form | graph | row order |
| --- | --- | --- |
| R0 | W2_S | node id (today's) |
| K1 | W2_S | s(u, v), highest first; sibling-only entries last; ties to node id |
| K2 | W2_S without the mention pairs whose directed links all score below τ; siblings kept | node id |
| K3 | K2's graph | K1's order |

Declared order: K1, K2, K3.

## 5. The grading

Per passage dataset, on s1sel, as U2j grades:
- **The frozen I's** ALL, recall and mean pool (rows reordered as in U2j, section 4).
- **The bridge** at 5 and 25.
- **The reach** of B ∪ X(k) (zu2b.reach_curves, unchanged): all-of-X ALL and rows, and the choice score. Reach depends
  only on the graph, so it is computed for R0, K2 and PKG; K1 shares R0's and K3 shares K2's.
- **The gap share**, `(m(form) − m(R0)) / (m(PKG) − m(R0))`, for I's ALL, the choice score and the bridge at 25, shown
  when the denominator is at least 0.005.

## 6. Beside, deciding nothing

On each passage dataset, for A and for H: the AUC and the average precision of the score against the dataset's own
hyperlinks; the share of W2 links kept at τ; the hyperlinked share of the kept links (precision); and the share of
the hyperlinked W2 links kept (recall).

## 7. The choice

- **Score:** the mean I's ALL over musique, hotpotqa and 2wiki, on s1sel, each graded with its zero-shot scorer.
- **Best:** the form with the highest score; among forms within 0.002 of it, the first in the declared order.
- **ADOPT_FOR_POOLS** when the best's score is at least R0's plus 0.002, and on no dataset does its I's ALL or its
  choice score fall below R0's by more than 0.002.
- **NO_GAIN** otherwise. It closes only K1, K2 and K3 as graph forms.
- **REFUSED** when U2i's gate 1 or U2j's rebuild of the links fails on a dataset graded, or a grade is missing.

## 8. What follows (fixed now)

- **ADOPT_FOR_POOLS:** U2e's pool rule is rebuilt on the chosen form's graph in its own file; then the model cell.
- **Either way:** the score s(u, v) joins U2j's eight features as an input to both models, in the file U2j section 7
  names. The GNN reads it on its edges; the MLP reads per-row summaries over the row's links to the question's
  first-stage rows, computed before any row is scored.
- **Caveat:** the four passage corpora are all Wikipedia, so this measures transfer between Wikipedia corpora only.
  metaqa and webqsp (KB graphs) are untouched.

## 9. How it runs (laptop)

    python outputs/mp_unified/zu2k.py --selftest
    python outputs/mp_unified/zu2k.py fit --dataset 2wiki       # scorer A; after zu2i.py pass --dataset 2wiki
    python outputs/mp_unified/zu2k.py fit --dataset hotpotqa    # scorer H
    python outputs/mp_unified/zu2k.py grade --dataset D
    python outputs/mp_unified/zu2k.py beside --dataset D
    python outputs/mp_unified/zu2k.py choose
    python outputs/mp_unified/zu2k.py report

## 10. Results

Filed after the run: `outputs/u2k/scorer_<D>.json`, `outputs/u2k/<D>/grade_s1sel.json`, `outputs/u2k/<D>/beside.json`,
`outputs/u2k/choice.json`, `outputs/u2k/report.md`.
