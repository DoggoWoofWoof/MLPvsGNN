# P1f: one pool-size rule for every dataset

Declared 15 October 2026, before any P1f number exists. Stage 3 of docs/PROGRAM_2026_10_15.md ("a common rule for
the dataset pools"). Code: `scripts/p1f_pool_size.py`. Status: declared; results in section 7.

## 1. Why

P1c2's pools (docs/P1C2_POOLS_KIND_T.md) take KIND_T's order, P1e's universal order, but they cut it at each
question's frozen size |I_q|. Those sizes come from step 1's per-dataset recipes, and those recipes were chosen on
s1eval golds. That is the audit's HIGH item: the order is one rule for all six datasets, but the size is not. Today's
mean pool sizes are metaqa 2018, squad 50, musique 2092, hotpotqa 94, 2wiki 106 and webqsp 2120.

P1f replaces the size with one label-free rule, the same for all six datasets. It does not change the content: the
pool is still the question's base and seeds (step 1's `bs`), then KIND_T's nodes outside them, in KIND_T's order. The
only thing a rule sets is how many of those nodes the pool takes.

## 2. Inputs (read only)

- P1e's walks, kinds and orders, imported unchanged (`scripts/p1e_anchor_kind.py`, through it P1d's five sources and
  step 4e's carves, linking and graphs), on the host mirror.
- Carves: s1sel on the five training datasets chooses the rule. s1eval on all six is computed and reported beside the
  choice; the choice never reads it.
- The cost weights: each dataset's fit and s1eval question counts, from P1c2's pool manifests
  (`outputs/p1c2/pools/{ds}/pools.json`; musique's fit carve is s1fit). These are metaqa 15,745, squad 17,729,
  musique 7,018, hotpotqa 13,335 and 2wiki 18,504.

## 3. What is measured, per question

- **Gold depth:** 0 for a gold in `bs`. Otherwise, one plus its index among KIND_T's nodes outside `bs`. A gold that
  KIND_T's 7,000 nodes do not hold has no depth (never covered).
- **`bs`'s size, today's size |I_q|, and how many KIND_T nodes lie outside `bs`.**
- **Label-free signals:**
  - the question's kind (entity or passage) and ROUTE's condition (both P1e's);
  - |L| and |L*|;
  - for each of P1d's five sources (A1st, A1all, Kst, Kall, R0) and for `max4` (the largest of the four walks):
    - the nucleus size n(τ), the fewest nodes, in the source's order, whose mass reaches τ of the source's total;
    - the perplexity exp(H) of its masses.
  - An empty source has n(τ) = 0 and perplexity 0.

**Checks**, on every carve, or the script stops:
- KIND_T's gold positions equal P1e's filed `pos_{ds}__{carve}.npz`, question for question.
- On s1eval, today's sizes reproduce P1c2's filed pool coverage (ALL and recall, to four decimals).

## 4. The rules (fixed before any number)

A rule gives each question a total size N_q. The pool is `bs`, then the first max(0, N_q − |bs|) KIND_T nodes outside
`bs`. Rows are |bs| plus the nodes taken (fewer when KIND_T runs out). Every N_q is rounded and clipped to [0, 7000].
`SIZES` = 50, 75, 100, 150, 200, 300, 500, 700, 1000, 1500, 2000, 3000, 5000.

| family | N_q | grid |
| --- | --- | --- |
| FIX(N) | N | N in SIZES (13 rules) |
| KIND(N_e, N_p) | N_e for an entity question, N_p for a passage question | both in SIZES (169) |
| NUC(s, τ, c) | c · n_s(τ) | s in A1st, A1all, Kst, Kall, R0, max4; τ in 0.5, 0.8, 0.9, 0.95, 0.99; c in 0.25, 0.5, 1, 2, 4 (150) |
| PPL(s, c) | c · exp(H_s) | s as above; c in 0.5, 1, 2, 4, 8, 16, 32 (42) |

**Not in any rule:**
- a dataset's name or any dataset-level statistic (corpus size, mean degree);
- a gold, an answer, or a score from a trained model.

The signals belong to the question and its walk.

## 5. The choice (s1sel only)

For a rule r, on each training dataset d's s1sel:
- ALL_d(r): the share of questions whose every gold is in the pool. A question with no golds counts 0, as in P1a.
- rows_d(r): the mean pool size.
- The cost of r is the question-weighted mean of rows_d(r), with weights w_d = fit + s1eval questions (section 2).
- "Today" is |I_q|, evaluated the same way on s1sel.

1. **Budget.** cost(r) ≤ cost(today).
2. **Floor.** ALL_d(r) ≥ ALL_d(today) − 0.01 on each of the five datasets.
3. **Pick.** Among the rules that pass both: the highest equal-weight mean of ALL_d. Ties within 0.001 go to the
   lower cost, then to the earlier rule in the order of section 4.

The verdicts:
- **`ADOPT_RULE`:** a rule passes both. It becomes P1f-b's pool size, and P1f-b is declared in its own file before any
  pool is rebuilt. Rebuilding means pools, looks, caches, KT, links and the lean pair's refits; P1f-b's call is made
  on the models.
- **`NONE_MEETS`:** no rule passes. Today's sizes stay, and the HIGH item stays open, said so in the program. The
  report gives the frontier (each family's best mean ALL at or under the budget, and each dataset's shortfall) so the
  next family is chosen from it.

## 6. Reported beside (never read by the choice)

- The chosen rule (or each family's best) on s1eval: all six datasets' ALL, recall and rows, today's beside.
- webqsp is zero-shot: its row count under the rule is what the lean pair would pay there.
- Each dataset's s1sel recall, and the share of questions at the 7,000 cap.

## 7. Results

Not run yet.
