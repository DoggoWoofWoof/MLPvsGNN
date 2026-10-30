# U2t groundwork: what U2s's stored arrays say about one construction for all datasets

Filed 30 October 2026. **Written after the numbers, and said so.** This is a diagnostic, not a stage: nothing here is
declared, adopted or closed, and no verdict follows from it. It was run on the laptop in about an hour, from U2s's
filed per-question arrays (`outputs/u2s/stats/<D>__s1sel.npz`, 16 MB in all; webqsp on s1eval, beside). No walk was run,
no model was read, and no test split was touched. The scripts and their printed outputs are in `outputs/u2t/diag/`.
The U2t stage itself (section 6) is declared in its own file before any of its numbers.

## 1. Why it was run

The user's direction of 30 October 2026: "close the gap to hyperlinks and unify construction for once and for all
target 2k rows min and we will see if we can go lower, and why hasnt better pools given better ranking till now".
U2s (6dbf4b2) filed NOT_UNIFIED: the cheapest universal 0.950 point costs 3,311.4 rows, 1.7 times the 1,953.5 cap.
The question here is where that cost sits, before a new form is declared.

**The loading check.** `diag_unify.py` reproduces U2s's filed universal kind_t frontier: 3,311.4 rows at level 0.950,
`rrf50+none/kind_t/kt_cos+n2_100:mass_A1all_t0.9_f4_k0@0.8`, with ALL 0.9506 (metaqa), 0.9993 (squad), 0.9511
(musique), 0.9834 (hotpotqa), 0.99 (2wiki).

## 2. Rows against level

Candidates are U2s's kind_t ones (12 constructions, 14 orders, 219 budgets, 10 cuts). Targets are U2s's:
min(level, C_d − 0.005) on each of the five training datasets, cost P1f's question-weighted mean rows. "Shared" is
one candidate for all five; "own" is each dataset's cheapest candidate on its own, weighted the same way.

| level | shared | own, weighted | metaqa own | musique own | hotpotqa own | 2wiki own |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.80 | 585.7 | 154.1 | 264.1 | 610.0 | 53.0 | 60.0 |
| 0.85 | 1,035.2 | 228.8 | 402.7 | 1,045.9 | 60.0 | 63.9 |
| 0.90 | 1,871.2 | 370.1 | 661.9 | 1,832.5 | 88.3 | 77.0 |
| 0.92 | 2,354.4 | 479.8 | 838.5 | 2,455.4 | 124.9 | 93.0 |
| 0.94 | 2,933.2 | 668.1 | 1,209.3 | 3,113.0 | 217.7 | 197.0 |
| 0.95 | 3,311.4 | 821.4 | 1,519.5 | 3,470.1 | 314.7 | 327.2 |
| 0.96 | 3,894.2 | 1,029.6 | 1,893.9 | 4,122.4 | 500.3 | 441.3 |
| 0.97 | 5,136.1 | 1,375.7 | 2,238.5 | 5,343.0 | 1,013.6 | 668.0 |

(Squad is 50.0 at every level. Webqsp, read beside at the shared pick, holds 0.834 at 1,218 rows for level 0.90 and
0.880 at 2,998 rows for level 0.95.)

- The shared cost is about **musique's own cost** at every level (1,871 against 1,833 at 0.90; 3,311 against 3,470 at
  0.95). The shared scale is musique's depth, paid on every dataset. Taken dataset by dataset, 0.95 costs 821 rows.
- The 1,953.5-row cap is met by a universal rule from the current families at **level about 0.90**, not 0.95.

## 3. Where the cost sits: a tail

Rows beyond B that a question's golds need in order kt (rrf50+none, kind_t) (`explore_need.py`, `predict_need.py`). The
first two columns are shares of all questions; the median, 90th and 95th are over the questions that need any walk node:

| dataset | questions with every gold in B | need ≥ 1,000 walk nodes | median | 90th | 95th |
| --- | ---: | ---: | ---: | ---: | ---: |
| metaqa | 0.019 | 0.066 | 14 | 610 | 1,385 |
| squad | 0.979 | 0.003 | 88 | 1,695 | 2,507 |
| musique | 0.368 | 0.124 | 146 | 2,379 | 4,049 |
| hotpotqa | 0.662 | 0.015 | 5 | 364 | 858 |
| 2wiki | 0.348 | 0.019 | 4 | 80 | 506 |
| webqsp (s1eval) | 0.140 | 0.103 | 49 | 1,229 | 2,389 |

The last three to five points of coverage buy most of the rows.

## 4. Nothing stored predicts a question's depth

`predict_need.py` takes every per-question measure U2s stored (the 30 `mq` mass columns, `n_n1`, `n_r0`, `n_n2`,
`taken`, a spread of `n_u`, `nw`, `bs`, the spread of the 14 list lengths) and reads Spearman's rho against the
question's needed rows, and the AUC for "needs ≥ 1,000". No fit.

| dataset | strongest measure | rho |
| --- | --- | ---: |
| metaqa | mq15 | +0.43 |
| squad | n_n1_0 | +0.14 |
| musique | mq14 | +0.18 |
| hotpotqa | mq16 | −0.21 |
| 2wiki | mq24 | +0.41 |
| webqsp | n_n2_2 | −0.23 |

**No measure has |rho| ≥ 0.15 with the same sign on four datasets.** The mass measures track depth on metaqa and
2wiki, hardly on musique, and the other way on hotpotqa. So U2s's MASS and ADA budgets are as much as the stored measures
can give; a universal per-question budget needs a measure that is not stored.

## 5. Orders do not cut the tail; per-question budgets have the headroom

`orders_tail.py`: rows beyond B that hold every gold of 95% of the questions, for each of the 14 orders, and for an
oracle that picks the best order per question.

| dataset | kt | best single order | oracle, best order per question |
| --- | ---: | ---: | ---: |
| metaqa | 1,704 | 1,704 (kt) | 1,520 |
| musique | 5,518 | 3,511 (kt_cos+n2_30) | 2,392 |
| hotpotqa | 322 | 296 (kt_cos+n2_100) | 165 |
| 2wiki | 294 | 294 (kt) | 247 |

The deep golds are deep in every order. Even a perfect router among the 14 orders leaves musique at about 2.4k rows
beyond B's 50, above the cap. The order is not the lever.

A budget that knows each question's needed rows (a bound, since it reads the golds; not a rule) costs **111 rows** at
level 0.95 and 121 at 0.96, weighted (`diag_unify.py`): metaqa 171.5, squad 50, musique 372.6, hotpotqa 57.0, 2wiki 57.4.
The universal rule pays 3,311 against that 111. The headroom is in the per-question budget.

## 6. What this sets for U2t (declared next, in its own file)

1. A universal rule from the families already graded reaches the cap only at level about 0.90. Holding 0.95 at 2k rows
   needs the per-question budget of section 5 from a signal that exists on every dataset, is read before any gold, and
   carries no parameter fit (the pools-first, parameter-free rule).
2. The named, ungraded forms need measures U2s did not store: RRF5T's agreement across its five orders, and a cut over
   the walk's mass. Both need a pass over the walks; U2s's own pass took 14 to 16 hours on the full grid, and a narrower
   pass (one construction, a few orders) is the cost to declare.
3. A signal from the first stage and the question alone (the first-stage lists' agreement, how much of the question B
   covers) needs no walk and is the cheap form to try first.
4. Not shown here, and not claimed: whether any such signal predicts depth; how many rows ranking actually uses
   (the R@5 against pool-depth curve needs models on deeper pools, which wait for PKO's pools); anything on zero-shot
   transfer. The ALL level is a reach measure. Why reach gains did not become ranking gains is in
   docs/U2D_REFIT_ON_U2E.md and docs/U2G_GRAPH_AND_RULE.md: reach was not the binding stage on hotpotqa and 2wiki, and
   the graph without hyperlinks cost the models more than the larger pools gained.

The data are s1sel (about 1,500 questions per dataset) and the kind_t recipe on U2s's 12 constructions. A level read
from 1,500 questions carries a sampling error; the differences above are large against it, but the table is a map, not a
verdict.
