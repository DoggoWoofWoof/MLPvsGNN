# P1g: one pool rule for all six, today's pool plus nodes reached through the base's own links

Declared 15 October 2026, before any P1g number exists. Code: `scripts/p1g_pool_beam.py`. Pools only: nothing trains
or reads, and nothing is fit. Status: declared.

## 1. Why

The user's target (15 October): **ALL ≥ 0.95 on every dataset, coverage first, then the fewest rows.** Today's s1sel
pools (P1c2's: base and seeds, then KIND_T's fill at each question's frozen size) give:

| dataset | ALL | mean rows |
| --- | ---: | ---: |
| metaqa | 0.952 | 2,022 |
| squad | 0.979 | 50 |
| musique | **0.885** | 2,094 |
| hotpotqa | 0.956 | 94 |
| 2wiki | 0.953 | 105 |
| webqsp (s1eval, beside) | **0.841** | 2,120 |

P1f (NONE_MEETS) showed that no size rule over KIND_T's order closes the gap: musique needs about 3,900 nodes of any
filed order to reach 0.95. D5 and D5b ([D5_GOLD_REACH.md](D5_GOLD_REACH.md)) showed why:
- musique's missed golds are one or two links from the base (99.5% of questions have every gold within two), but they
  are bridge passages: in neither first-stage list, and buried among about 6,500 one-link neighbours by first-stage
  weight;
- an order that reads the **parent** passage finds them earlier. Per-parent beams, which keep each strong base node's
  few best neighbours by the question (or by the part of the question the parent does not already answer), reach them
  in far fewer rows than any global order.
- metaqa's misses are 3-hop answers that KIND_T's walk already reaches. They need KIND_T's fill, not a beam.

- On musique the beams add little: today's pool plus the best beam gives 0.906 (D5b: res, P = 200, m = 10). Today's
  pool plus N1's first 7,000 by cos gives 0.945.
- KIND_T's own order holds every gold of 0.950 of musique's s1sel questions at its 7,000 nodes (D5). P1f's size
  rules had to stay within today's rows; coverage now comes first, so P1g lets the fill run longer along the same
  order.

So P1g keeps KIND_T's fill, lets it be shorter or longer, and adds a small, parent-guided set on top, with one rule for
all six.

## 2. The candidates

A candidate is a pair **(f, X)**. The pool is:
- **B:** the question's base and seeds, unchanged (step 1's, as every pool stage opens them);
- **the first round(f × |fill|) of KIND_T's nodes outside B,** where |fill| is today's fill size and
  f ∈ {0.25, 0.5, 1, 1.5, 2, 3, 4}. f = 1 is today's fill; f > 1 goes on along KIND_T's same order (P1e's, unchanged)
  and stops at its end;
- **X:** one extra set of nodes outside B, from one of three families.

| family | X | grid |
| --- | --- | --- |
| none | nothing | (f = 1 with none is **today's pool**) |
| beam1 | B's first P nodes by first-stage weight (P1e's steer); from each, its first m neighbours outside B over the three link families, by score | score ∈ {cos, res}, P ∈ {10, 20, 50, 100, 200}, m ∈ {1, 2, 3, 5, 10} |
| beam2 | beam1, plus each beam1 node's first m neighbours outside B by the same score, with that node as the parent | P ∈ {10, 20, 50}, m ∈ {1, 2, 3, 5} |
| cut | the first k of B's one-link neighbours (N1) under one global order | order ∈ {w, rrf, cos, res}, k ∈ {50, 100, 200, 300, 500, 1,000, 2,000, 3,000, 5,000, 7,000} |

The scores and orders are D5b's, imported unchanged (`scripts/d5b_neighbour_orders.py`). ê are the frozen encoder's
dense embeddings at unit length:
- **cos:** ê_q · ê_v.
- **res:** the parent's residual cos(q − (q·p̂)p̂, v): the question with what the parent p already says taken out.
  In N1's global order, the best parent's.
- **w:** the summed first-stage weight of v's B-parents (D5's weight order). **rrf:** RRF of the cos and w orders.

That is 115 extras × 7 fill fractions = **805 candidates**. Every one is the same rule on every dataset, with no
dataset name, no gold and nothing learned. Its few numbers (f, P, m, k) are chosen once, on s1sel, by section 5.

## 3. What is measured

Per question and candidate: **rows** = |B ∪ fill_f ∪ X| (exact: X's nodes already in the fill are not counted twice),
and per gold whether that set holds it. Per carve and candidate: ALL, gold recall, mean rows and the 90th percentile of
rows.

- **Carves:** s1sel of the five training datasets decides. webqsp's s1eval is beside only, since webqsp's golds never
  design a rule. After the verdict, the chosen candidate and today's pool are measured on each training dataset's
  s1eval, beside (`eval`).
- **Checks** (the stage stops on any failure):
  - the carve's golds equal P1f's gold pointers;
  - |B| and |I_q| equal P1f's filed `bs` and `inc`;
  - **today's pool (f = 1, none) holds exactly the golds P1f's filed depths say it holds, with exactly B + |fill|
    rows,** on every question.
- **Selftest** (`python scripts/p1g_pool_beam.py --selftest`, passed before this commit): the 115 extras and
  805 candidates, named uniquely; D5b's toy graph gives D5b's own beams (one and two steps) and cuts; a beam computed alone
  equals the same beam computed in the grid; positions in the fill; rows and held golds of the union by hand, fill
  fractions above 1 and the order's end included; the summary; and the choice's three outcomes.

## 4. The order of the candidates

f ∈ (0.25, 0.5, 1, 1.5, 2, 3, 4), and within each: none; then beam1 and beam2 by score (cos, res), P and m ascending; then the cuts
by order (w, rrf, cos, res) and k ascending. The order breaks exact ties in cost.

## 5. The rule, fixed before any number

Let ALL_d(c) and rows_d(c) be candidate c's ALL and mean rows on dataset d's s1sel, for the five training datasets.
- **Ceiling:** C_d = the best ALL_d over all 805 candidates.
- **Target:** t_d = min(0.95, C_d − 0.005). Where the family cannot reach 0.95, a dataset must come within 0.005 of
  the best it can do (0.005 is about 8 of 1,500 questions, under one standard error of ALL there).
- **Cost:** P1f's: the question-weighted mean of the five datasets' mean rows (weights: fit plus s1eval questions,
  from P1c2's manifests).
- **The choice:**
  - candidates with ALL_d(c) ≥ t_d on all five datasets **meet**. The one with the lowest cost is chosen (ties by
    section 4's order).
  - **TARGET_MET** when it meets and every t_d is 0.95; **CEILING_MET** when it meets but some dataset's target is its
    ceiling, below 0.95;
  - **NONE_MEETS** when no candidate meets every target. The record names the candidate with the smallest worst
    shortfall (ties by cost), and nothing is adopted.

Coverage comes first: no candidate below a target can win on rows.

**Beside, deciding nothing:**
- webqsp's s1eval under the chosen candidate and today's;
- s1eval of the five training datasets under the chosen candidate and today's (`eval`);
- the cheapest candidate meeting every target in each family (fill fraction and extra family);
- each dataset's own cheapest candidate reaching its own target: a per-dataset pick, an upper bound only.

## 6. What it means, and what follows

- **TARGET_MET or CEILING_MET:** the chosen candidate becomes the final pool. **P1g-b**, declared in its own file,
  writes the pools for the fit and s1eval carves, computes their features (the feature compile and P1c3's KT columns,
  unchanged) and refits the lean pair on them (a screen, then a full run). The models are judged on R@5 and the claims
  as before. The new rows' feature cost is reported beside P4's timings.
- **NONE_MEETS:** today's pools stay. The record says how far each family gets and at what cost, for the next rule.

## 7. Before this file

- D5 and D5b's numbers came first ([D5_GOLD_REACH.md](D5_GOLD_REACH.md)): they are why the beams and cuts are the
  families here. They are diagnostics on the same s1sel carves, and the choice is made on s1sel by section 5, as
  every pool stage's is. s1eval is not read before the verdict.
- A development run on the first 30 s1sel musique questions, on the laptop, checked the APIs and today's-pool check. It
  wrote nothing and logged no number. The longer fills (f > 1) and the cuts at 5,000 and 7,000 were added after it,
  before any P1g number, because of D5's KIND_T line and D5b's musique numbers above. A second development run (30
  questions, the longer fills) passed the same check, and the selftest covers them.
- D5b's musique, metaqa and squad numbers were read before this commit; hotpotqa, 2wiki and webqsp's were still
  running.

## 8. Running it

- **Host items** (`p1g-stats-*`), one per dataset, 2 CPUs each:
  `python outputs/host_ops/pylib_run.py scripts/p1g_pool_beam.py stats DATASET --host`.
- **Then** on the laptop, after the fetch: `python scripts/p1g_pool_beam.py verdict`; the host's `eval` items for the
  five s1eval carves; then `report`.
- **Disk:** under 0.1 GB (per carve, a rows matrix of questions × 805 and a held matrix of golds × 805).
- **ETA:** D5b's runs took 6 minutes (metaqa) to about an hour (webqsp) per carve. P1g adds KIND_T's order (P1f's took
  2-4 minutes per s1sel carve) and drops D5's two-link expansion, so about the same.
- The encoder stays frozen. Test splits are never read. webqsp's golds never choose.

## 9. Results

Filed 17 October 2026, 03:05 (`outputs/p1g/verdict.json`), from the six `stats` runs that finished on the host on
15 October. **TARGET_MET.**

Chosen on s1sel: `f3+beam1_res_P200_m10` (three times today's fill along KIND_T's order, plus beam1 by the residual
score from B's first 200 nodes, 10 neighbours each). Its cost is 2,263.4 rows against today's 699.7.

| dataset | target | today ALL / rows | chosen ALL / rows |
| --- | ---: | --- | --- |
| metaqa | 0.9500 | 0.9519 / 2,022 | 0.9886 / 6,184 |
| squad | 0.9500 | 0.9793 / 50 | 0.9880 / 298 |
| musique | 0.9500 | 0.8846 / 2,094 | 0.9537 / 6,280 |
| hotpotqa | 0.9500 | 0.9556 / 94 | 0.9861 / 478 |
| 2wiki | 0.9500 | 0.9525 / 105 | 0.9693 / 573 |
| webqsp (s1eval, beside) | | 0.8410 / 2,120 | 0.9215 / 6,314 |

**Beside: the cheapest rule per dataset** (an upper bound on what one rule could save, never a pick):
- metaqa `f1+none`, 2,022 rows (today's pool);
- squad `f0.25+none`, 50 rows;
- musique `f2+cut_rrf_3000`, 5,612 rows;
- hotpotqa `f0.5+beam2_res_P10_m1`, 86 rows;
- 2wiki `f1+none`, 105 rows (today's pool).

musique sets the price. Only it needs thousands of extra rows to reach 0.95, and one rule for all six then gives
hotpotqa and 2wiki five to fourteen times the rows they need.

P1g-b builds P1g2's choice, not this one: P1g2 chose over these 805 rules and their gated forms together
([P1G2_GATED_EXTRA.md](P1G2_GATED_EXTRA.md), section 6).
