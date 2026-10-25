# U2m: one construction and one walk recipe for all six, then the relational walk

Declared 25 October 2026, before any U2m number. This is stage U2m of docs/PROGRAM_2026_10_24.md (section 3). Code:
`outputs/mp_unified/zu2m.py`, which imports U2e, U2c, P1e, P1d, D5b and M3B's compile unchanged. Pools only: nothing
trains, nothing is fit, and no construction, recipe, order, budget or cut reads a gold. The choice reads golds on
s1sel, as every pool stage's choice does.

## 1. Why

**The construction is per dataset.** Two parts of today's pools come from the 13 Sep frozen construction
(configs/m3b_controlled_comparison.yaml, `candidate_contract_frozen_2026_09_13`), which is set per dataset:

| dataset | base | expansion I (regime: hop setting) |
| --- | --- | --- |
| metaqa | equal_rrf_budget_50 | STRUCT: h3_c25_v2000 |
| squad | equal_rrf_budget_50 | none (RETRIEVAL) |
| musique | equal_rrf_budget_200 | FULL: h2_c25 |
| hotpotqa | equal_rrf_budget_50 | STRUCT: h1_c25 |
| 2wiki | equal_rrf_budget_50 | STRUCT: h1_c25 |
| webqsp | dense_top200 | FULL: h3_c25_v2000 |

The rest is already one rule. KIND_T's walks, U2e's N1 and N2, and the models' compiled edges read all three families
on every dataset. The construction still reaches U2e's pick (docs/U2E_POOLS_TO_THE_CEILING.md, section 10) in three
ways:
- B, the base and seeds, is in every pool;
- I is one of U_0's parts;
- the ADA budgets scale today's fill, |I_q| − |B_q|, which is 0 on squad.

So U2e's pick is one rule over per-dataset inputs. The program (section 3, U2m c) asks for one base and one
expansion for all six.

**KIND_T switches on a question kind.** A question is an entity question when every node it starts from is name-only.
KIND_T then walks ROUTEst; otherwise it walks RRF5T (docs/P1E_ANCHOR_KIND.md). In practice this is a KB switch:
- on the fit carves every metaqa question is an entity question and no passage-corpus question is;
- on webqsp, 96.7% are.

The user asked on 24 October whether the pools should rely on it. P1e's record of what the switch was worth: P1d's
RRF5, which has no switch, was below P1a's bar at 29 of metaqa's size points (largest gap −0.114) and 3 of webqsp's
(−0.058). KIND_T brought those to 4 points (−0.005) and none. That was measured at |I_q|. U2e's fused orders and
budgets have never been tried without the switch.

## 2. The forms and their order

- **(c) one construction** and **(e) one recipe** run first, in one pass, because they share the walks. Sections 3
  to 9 declare them.
- **(b)** the list-whole cut, **(d)** the mass split by arrival type and **(a)** the relational walk run after
  (c) and (e)'s verdict, on the pools it carries forward. Section 10 fixes what each one is now. Any grid or check
  section 10 leaves open goes into a dated section of this file before that form's code runs.

## 3. The graph, the carves and the walks

These are U2e's, unchanged:
- the graph is W2_S, swapped in for `structural` on musique, squad, hotpotqa and 2wiki (`zu2c.graph_csr`,
  `zu2c.csr_store`); metaqa and webqsp keep their KB;
- the carve is s1sel on the five training datasets, with webqsp's s1eval beside;
- the walks are P1d's five sources and P1e's four steered walks. They read the seeds, the first stage's lists and
  the links. They read no base and no I, so a question walks once for every construction.

## 4. The constructions (c)

A construction is a base and an expansion:
- B = the base ∪ the seeds. The seeds are unchanged: the dense top 5, then the SPLADE top 5.
- I = B ∪ `m3b_pools.expand_hops(seeds, FULL, setting)`, through `m3b_compile.build_pool`, both unchanged.
  - FULL is the three families (structural, ner, knn), with W2_S as structural on the passage graphs.
  - Each family is symmetrised and ordered as configs/m3a_headroom.yaml's `graph_regimes` says.

**The grid** is every pairing of a base with an expansion, 12 constructions:
- **Bases:** `equal_rrf_budget_50`, `equal_rrf_budget_200`, `dense_top200`. These are the three the 13 Sep
  construction uses, built by `m3b_compile.base_rows`.
- **Expansions:** `none` (I = B), `FULL:h1_c25`, `FULL:h2_c25`, `FULL:h3_c25_v2000`. These are the three hop
  settings it uses.
- Names are `<base>+<expansion>`, for example `rrf50+FULL:h2_c25`, with the bases written `rrf50`, `rrf200` and
  `dense200`.

**Beside: each dataset's own construction** (`own`), from section 1's table.
- It is a grid member on three datasets: squad's is `rrf50+none`, musique's `rrf200+FULL:h2_c25` and webqsp's
  `dense200+FULL:h3_c25_v2000`.
- On metaqa, hotpotqa and 2wiki it is a STRUCT expansion, which is not in the grid. It is built as U2e builds it,
  through `m3b_compile.prepare`.

What changes with the construction:
- **With the base only:** B; N1 (D5b's `n1_table` over the three families, weighted over B); its cosine ranks; R0
  outside B; N2 (U2e's `second_link`).
- **With the whole construction:** I, and today's fill.

## 5. The recipes (e)

A recipe is the walk order that stands where U2e's KT stands:
- **`kind_t`**: KIND_T, with P1e's switch. This is U2e's KT.
- **`rrf5t`**: RRF5T on every question. This is P1e's passage recipe: RRF over A1st, A1all, KTst, KTall and R0.
- **`both`**: P1d's `rrf` (constant 60, cut at 7,000 like every P1e order) of ROUTEst and RRF5T, on every question.

`rrf5t` and `both` read no kind; they are the kind-free recipes. ROUTEst on every question alone is not a recipe:
off P1d's route it is A1st, and `both` holds it.

For each construction and recipe:
- **The orders** are U2e's 14 (its section 4), built by `zu2e.question_orders` unchanged. Its KT argument is the
  recipe's order outside that construction's B.
- **Today's fill** is U2e's definition with the recipe's order: the first max(0, |I_q| − |B_q|) nodes of that order
  outside B, so min(|I_q| − |B_q|, the order's length outside B).
- On the own construction with `kind_t`, both are U2e's, question for question (section 8).

## 6. The candidates

Per construction and recipe, there are U2e's 13,860 candidates: 14 orders × 99 budgets (19 FIX, 80 ADA) × P1h's 10
τ, with U2e's budget, cut and evaluation.
- **Universal:** the 12 constructions × the 3 recipes, 498,960 candidates, named
  `<construction>/<recipe>/<order>:<budget>@<τ>`.
- **Own:** the own construction × the 3 recipes, 41,580 candidates, beside. `own/kind_t/…` is U2e's own candidate
  set, so U2e's pick is `own/kind_t/kt_cos:ada_f12_k0@0.85`.

Ties go to the earlier candidate, in this order:
1. the construction: bases, then expansions, each in section 4's order;
2. the recipe: `kind_t`, `rrf5t`, `both`;
3. U2e's own order within a recipe.

U2c's candidates do not enter. U2e's pick already beats U2c's chosen within the cap.

## 7. The choice

U2e's rule, with its cost and cap:
- cost: P1f's question-weighted mean rows, with U2c's weights;
- levels: t from 0.950 to 1.000 in steps of 0.005;
- targets: min(t, C_d − 0.005) on each training dataset d, where C_d is the best ALL any universal candidate
  reaches on d;
- the cap: 1,953.5 rows.

Two frontiers are built over the universal candidates. Each one's pick is its point at the highest level within the
cap:
- **frontier C:** the `kind_t` candidates, giving pick C at level L_C;
- **frontier E:** the `rrf5t` and `both` candidates, giving pick E at level L_E.

The incumbent is U2e's pick, at level 0.955 and 1,812.6 rows. A frontier with no point within the cap has no pick.

| verdict (c) | when |
| --- | --- |
| UNIFIED_RAISED | L_C > 0.955 |
| UNIFIED_HOLDS | L_C = 0.955 |
| UNIFIED_AT_A_PRICE | L_C = 0.950 |
| NOT_UNIFIED | frontier C has no pick |

| verdict (e) | when |
| --- | --- |
| KIND_FREE_NO_PRICE | E has a pick and L_E ≥ L_C (or C has none) |
| KIND_FREE_AT_A_PRICE | L_E = L_C − 0.005, P1g's tolerance |
| KIND_FREE_TOO_DEAR | otherwise |

**Carried forward** to (b), (d), (a) and U2n:
- pick E, when (e) is KIND_FREE_NO_PRICE or KIND_FREE_AT_A_PRICE;
- otherwise pick C, when (c) is not NOT_UNIFIED;
- otherwise U2e's pick stays, and the best universal points are filed with their shortfall.

**Filed beside, past the cap too:**
- frontiers C and E, and the own construction's frontier for each recipe, with each dataset's ALL and rows at every
  level;
- the cheapest point at every level for each construction, each recipe and each U2e order family;
- per dataset, the fewest mean rows reaching each level. This is an upper bound only;
- each dataset's union ceiling: ALL when a gold counts as held if any order of any construction and recipe holds it;
- the carried pick's misses by U2e's causes (TAU_CUT, BUDGET, UNREACHED), with reach meaning any U2m order holds the
  gold;
- the price of the carried pick against U2e's pick: per dataset ALL and rows, and the cost;
- webqsp's s1eval for every pick and every level's point. It never moves a choice.

One caution: the frontier chooses among about half a million correlated candidates on s1sel. The carried pools are
judged where U2n grades them, on s1eval and zero-shot.

## 8. Checks

Any failure stops the dataset, and nothing is written.

**Every question, against U2e's filed arrays** (outputs/u2e/stats, the same graph and carve). The own construction
with `kind_t` must reproduce, exactly:
- |B|, today's fill, |I|;
- the 14 orders' lengths, and every gold's place in each;
- |N1|, |R0|, |I outside B|, |N2(M)|, |U_M|, the parents taken and the hubs passed over.

On squad, musique and webqsp the own construction is a grid cell, so that grid cell itself is what is checked.

**The node rows.** A question's node rows are read once and served from memory to every construction and recipe.
- The cosines depend on the rows' layout. Measured before this declaration: across 200 random trials, 592 rows'
  float32 products differed in their last bits between two matrix layouts. So every order is built exactly as U2e
  builds it: its own matrices, from rows equal to DenseNodes' bytes.
- Development runs compare every served read with DenseNodes.read. The reproduction above checks it on every
  question.

**In the code:**
- every order passes U2e's `orders_exact`;
- every construction's B holds the seeds, and its I holds its B;
- the `none` expansion's I is its B.

**In the verdict:** U2m's table over the own/`kind_t` columns must equal U2e's `u2e_table` on U2e's filed arrays,
ALL and rows, for all 13,860 of U2e's candidates.

**Self-test** (`python outputs/mp_unified/zu2m.py --selftest`, passed before this commit):
- the grid and its names;
- the `both` recipe against a direct RRF;
- today's fill;
- the column layout;
- the vectorized table against U2e's `evaluate` and `summ` on toy arrays;
- the two frontiers, the verdicts and what is carried, on toy tables;
- the shards and the merge.

**Development runs** (`--limit`) write nothing and log no number. Before this commit, squad's and metaqa's (30
questions each) passed every check above, with every served row verified.

## 9. Where it runs, and the outputs

- **The laptop** runs squad, metaqa, musique and webqsp: `python outputs/mp_unified/zu2m.py stats D [--shard I/K]`,
  with 2 BLAS threads, as U2e's shards ran.
- **Modal** runs hotpotqa and 2wiki, whose graphs need more memory than the laptop has. It uses
  `outputs/mp_unified/zu2m_modal.py`, with U2e's image, pins, mirror and volumes. U2e's filed arrays are read from
  the volume where U2e wrote them. U2m writes to a new directory named by zu2m.py's sha256.
- The section 8 reproduction is what ties the laptop to Modal. U2e's arrays were written on Modal, so a laptop
  dataset that passes it is bit-identical to a Modal run.
- `zu2m.py merge D --parts K` joins the shards, and `zu2m.py verdict` runs on the laptop. The stats go to
  `outputs/u2m/stats/<D>__<carve>.npz` and `.json`, and the verdict to `outputs/u2m/verdict_ce.json` and `.md`.
- The arrays stay uncommitted.

## 10. What follows (fixed now)

**(b) The list-whole cut**, on the carried construction and recipe.
- A list is one node u's neighbours through one edge type in one direction: (u, relation, direction) on a typed KB
  edge, and (u, family) on an untyped family. u ranges over the pool, B included.
- After P1h's cut at τ, every list of at most G nodes that the budgeted build order holds, and that the cut splits,
  is restored whole: its members past the cut rejoin the pool.
- G is 10, 25, 50, 100, 200 or 500, applied to every point of the carried frontier family.
- The pick follows section 7's rule. The check: with G = 0, the carried frontier comes back exactly.
- U2l's diagnostic read this before this declaration: on webqsp, ALL would go from 0.893 to 0.948 if the pick's
  cut kept its lists whole. It decides nothing.

**(d) The mass split by arrival type.** The carried recipe's walk mass, split by the family of the edge it arrived by
last, and on the KB by the relation:
- p(v) = Σ_type p_type(v);
- the split kernel's p is bit-identical to the original kernel's, so every order is unchanged;
- the parts sum to p within float32 rounding.

It is an input for U2n's models, graded there, never a pool rule.

**(a) The relational walk.** Within a family, a node's mass goes to its neighbours in proportion to ω(q, type) times
the first stage's steer. With every ω equal to 1, the walk is the carried recipe's, bit for bit. The types:
- **On the KB:** its relations, with their filed relation vectors.
- **On a passage graph:**
  - (i) the family, the link kind (mention or sibling, from U2b's link pass) and the direction;
  - (ii) the shared entity as the link's label, with a vector derived from existing ones: the entity's own node
    vector, or the mean of the passages that mention it;
  - (iii) EM latent types, fit on the graph alone, with no question and no gold.

No form calls the encoder on new text (the user's ruling of 24 October). A7's pseudo-typed passage edges
(docs/A7_S6_SCORER_INPUT.md) are cited beside: their gains were on webqsp only. (a)'s grid (the temperature and the
type forms) and its checks are fixed in a dated section of this file before its code runs.

A verdict closes only the form tested (docs/SCREENS.md, section 3b).

## 11. (b) the list-whole cut: its reading, fixed 25 October 2026

Written before (c) and (e)'s verdict and before any (b) code or number. Code: `outputs/mp_unified/zu2m_b.py`, which
imports zu2m.py, U2e and P1h unchanged (zu2m.py is not edited). Pools only: nothing trains, and no list, restore or
size reads a gold.

**One candidate.** A candidate is (c) and (e)'s `<construction>/<recipe>/<order>:<budget>@<τ>`. Per question:
- the order is the column's order outside B, rebuilt by section 5's path;
- kk = min(its length, the budget), keep = P1h's `keep_counts(|B|, kk, τ)`;
- the build-held set H = B ∪ order[:kk], and the kept pool P = B ∪ order[:keep];
- a node's place is 0 in B, p for the order's p-th node, and none outside H.

**The lists.** For each u in P, B included:
- on an untyped family (W2_S, ner, knn on the passage graphs; any untyped family on a KB), one list: u's neighbours
  in that family's symmetrised store, as FULL reads it;
- on a typed KB's structural family, one list per (relation, direction) in u's typed row, its members once each;
- u itself is never a member of its own list.

**The restore at G.** A list L is restored when all four hold:
- |L| ≤ G;
- H holds every member (the budgeted build order holds it);
- some member is in P (the cut kept part of it);
- some member is in order[keep:kk] (the cut dropped part of it).

The restored rows are the union, over the restored lists, of their members in order[keep:kk]. It is one pass: a
restored node's own lists are not read. The candidate at G has rows |B| + keep + |restored|, and a gold is held
when its place is 0, or between 1 and keep, or it is restored. At G = 0 nothing is restored, so the candidate is
(c) and (e)'s, unchanged.

**The candidates.** The carried frontier is the one (c) and (e) carried: frontier E or C, or U2e's own `kind_t`
frontier if U2e's pick stays.
- Every point of that frontier, at every level from 0.950 to 1.000, past the cap too, crossed with G in {0, 10, 25,
  50, 100, 200, 500}. Those are at most 77 candidates.
- Their order is G ascending, then the points' order in (c) and (e)'s candidate index. So ties go to the smaller G,
  and G = 0 comes first.

**The choice** is section 7's rule over these candidates. It uses the same cost, levels and cap, and the targets
min(t, C_d − 0.005) with (c) and (e)'s filed ceilings C_d. So a level means what it meant in (c) and (e)'s verdict.

| verdict (b) | when |
| --- | --- |
| LIST_WHOLE_RAISED | the pick's level is above the carried pick's |
| LIST_WHOLE_CHEAPER | the same level, G > 0, and a lower cost than the carried pick |
| NO_GAIN | otherwise: the pick is the carried one at G = 0 |

**Carried forward** to (d), (a) and U2n: the (b) pick with its G when RAISED or CHEAPER, otherwise (c) and (e)'s
carried pick.

**Filed beside:**
- each candidate's ALL and rows per dataset, and webqsp's s1eval;
- the mean restored rows;
- the carried pick's misses that each G recovers, by section 7's causes.

webqsp never moves the choice.

**Checks.** Any failure stops the dataset, and nothing is written.
- Every question: each rebuilt column's length and every gold's place equal U2m's filed arrays (`lens`, `pos`) for
  that column.
- In the code: the restored rows lie in order[keep:kk], so rows ≤ |B| + kk; what is held at G holds at every
  larger G.
- In the verdict, at G = 0:
  - every point's ALL and rows per dataset equal (c) and (e)'s filed values;
  - the frontier over the G = 0 candidates returns the carried frontier's picks at every level.
- Self-test (`--selftest`), before the commit that adds the code: the vectorized restore against a per-list loop,
  and the typed grouping against a per-edge loop, both on toy orders and graphs.
- Development runs (`--limit`) write nothing and log no number.

**Where it runs.** As section 9 says:
- the laptop runs squad, metaqa, musique and webqsp;
- Modal runs hotpotqa and 2wiki (`outputs/mp_unified/zu2m_b_modal.py`, U2m's image and volumes, a new output
  directory named by zu2m_b.py's sha256).

The stats go to `outputs/u2m/b/stats/<D>__<carve>.npz` and `.json`, and the verdict to `outputs/u2m/b/verdict_b.json`
and `.md`. The arrays stay uncommitted.

## 12. (c) and (e): filed 25 October 2026

**NOT_UNIFIED, KIND_FREE_TOO_DEAR.** U2e's pick stays: `own/kind_t/kt_cos:ada_f12_k0@0.85`, at level 0.955 and
1,812.6 rows (outputs/u2m/verdict_ce.json and .md). Section 8's check held: U2m's table on the own construction with
`kind_t` equals U2e's table for all 13,860 of U2e's candidates.

**No universal candidate meets 0.950 within the cap** (1,953.5 rows).

| frontier | cheapest at 0.950 | cost | multiple of the cap |
| --- | --- | ---: | ---: |
| C (`kind_t`) | `dense200+FULL:h2_c25/kind_t/kt_cos+n2_30:ada_f8_k0@0.8` | 3,641.3 | 1.86 |
| E (kind-free) | `rrf50+FULL:h3_c25_v2000/both/kt_cos+n2_10:ada_f2_k0@1` | 3,993.8 | 2.04 |
| own construction, `kind_t` (beside) | U2e's order family | 1,497.8 | 0.77 |

- **Where the rows go.** The universal 0.950 point holds squad at 0.9993 with 2,981 rows. The own construction holds
  it at 0.9793 with 50 rows, because squad's expansion is `none` and its fill is 0. Under one construction with an
  expansion, the budgets that scale with fill give squad thousands of rows. A FIX budget gives every dataset the same
  rows. The fewest mean rows any universal candidate needs for 0.950, one dataset at a time (an upper bound): squad
  50, hotpotqa 323.4, 2wiki 326.1, metaqa 1,817.5, musique 3,858.4.
- **The switch.** On the own construction the kind-free recipes cost more at every level: at 0.950, `both` costs
  1,723.0 (+15%) and `rrf5t` 2,466.6 (+65%), against `kind_t`'s 1,497.8. So KIND_T's switch is worth its keep where
  the construction is per dataset.
- **Union ceilings** (a gold is held if any order of any construction and recipe holds it): metaqa 0.9967, squad
  0.9993, musique 0.9811, hotpotqa 0.9887, 2wiki 0.9960, webqsp 0.9647.
- **The carried pick's misses** (golds), by section 7's causes:

| dataset | missed | TAU_CUT | BUDGET | UNREACHED |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 329 | 287 | 8 | 34 |
| squad | 31 | 0 | 30 | 1 |
| musique | 62 | 22 | 9 | 31 |
| hotpotqa | 69 | 13 | 38 | 18 |
| 2wiki | 50 | 13 | 31 | 6 |
| webqsp (s1eval) | 2,077 | 664 | 369 | 1,044 |

TAU_CUT is the cause (b) restores: 87% of metaqa's misses and 32% of webqsp's.

**What follows.**
- (b) runs on U2e's own `kind_t` frontier, as section 11 says for this case, then (d) and (a).
- The verdict closes only these 12 constructions with U2e's budgets (docs/SCREENS.md, section 3b). The next form of
  (c) is one construction whose budget is read from each question's own walk mass, not from today's fill, so a
  question whose walk concentrates gets few rows on any dataset. It is declared in its own file.

## 13. (d) the mass split by arrival type: its reading, fixed 25 October 2026

Written before (b)'s verdict and before any (d) code or number. Code: `outputs/mp_unified/zu2m_d.py`, which imports
zu2m.py, zu2m_b.py, U2e, P1d and P1e unchanged. Nothing trains and nothing is fit. Only the beside table reads golds.
(d) has no verdict: it is an input for U2n's models, graded there (section 10), never a pool rule.

**The walks.** These are the five walks the carried recipe, `kind_t`, reads:
- ROUTEst reads Kst and A1st;
- RRF5T reads A1st, A1all, KTst, KTall and the unwalked R0.

So the five are A1st, A1all and Kst (P1d's sources, step 4e's kernel) and KTst and KTall (P1e's steered kernel).
Every question gets all five, whatever its kind, so a model's columns do not depend on the switch.

**The types.** A row v's mass is p(v) = α × the mass that reached v before its last push. Each arrival has a type:
- **restart:** the walk's restart weight on v;
- **the family of the edge it came over:** structural (the KB's own graph on metaqa and webqsp, W2_S on the passage
  graphs), ner or knn;
- **on metaqa and webqsp's structural family, also its (relation, direction):** the entry for the sending node u in v's
  typed row. Direction 1 means v is the stored source of the fact, 2 that v is its destination. When v's typed row
  holds several entries for u, the arrival is split equally among them.

p_type(v) is α × the type's arrivals before v's last push. So:
- p(v) is the sum of its four parts (restart, structural, ner, knn);
- on the KB, p_structural(v) is the sum of its (relation, direction) parts.

On the st walks only restart and structural can be nonzero.

**The kernel.** The split kernels are step 4e's and P1e's push, with the same statements on p and r in the same order.
Beside r, they keep each type's pending and settled amounts for the carried pool's rows only. An arrival at a pool
row adds to its type's pending amount. A push of the row settles α × each pending amount and clears it. So p, and
every order, is the original kernel's. Only the arrivals at pool rows are kept, so the memory is the pool's, not the
graph's.

**The rows.** These are the rows of the pick section 11 carries, on every question of the carve: B, order[:keep], and
the restored rows when that pick has G > 0.

**The arrays.** Per dataset and carve:
- the rows (with a question pointer);
- for each row and walk, the four parts and p, in float32;
- the walk's total over its top NMAX nodes, so the masses can be scaled as P1b's walk_mass scales them;
- on the KB, for each walk, sparse (row, relation, direction, mass) entries.

The arrays go to `outputs/u2m/d/stats/<D>__<carve>.npz` and `.json`, and stay uncommitted.

**Checks.** Any failure stops the dataset, and nothing is written.
- **Orders and masses.** For every question and walk, the split kernel's top-NMAX order equals the original kernel's.
  On A1st, A1all and Kst, its scaled masses also equal walk_mass's, bit for bit.
- **Parts.** In float64, before storing, |Σ parts − p(v)| ≤ 1e-9 × p(v) + 1e-15 on every row and walk. On the KB, the
  (relation, direction) parts meet the same bound against p_structural(v).
- **Rows.** Each question's row count equals the carried pick's filed rows: (b)'s at its G, or (c) and (e)'s when
  (b) is NO_GAIN. Its held golds equal the filed ones.
- **Self-test** (`--selftest`), before the commit that adds the code. On toy graphs, typed and untyped, steered and
  unsteered, the split kernel's p equals the original's bit for bit. Its parts also equal a per-arrival log, settled
  push by push in plain Python.
- **Development runs** (`--limit`) write nothing and log no number.

**Beside, deciding nothing.** Per dataset and walk:
- the mean share of each type in a row's mass, over gold rows and over the other rows;
- on the KB, the five (relation, direction) pairs with the largest share of the structural mass on gold rows.

webqsp's s1eval is shown beside, as everywhere.

**Where it runs.** As section 9 says:
- the laptop runs squad, metaqa, musique and webqsp;
- Modal runs hotpotqa and 2wiki, after (b)'s runs, if the workspace's remaining credit covers them. If it does not,
  those two are held, and the record says so.

U2n computes the same arrays on its fit and evaluation carves with these kernels, since the kernels are the form.

**What follows.** (a)'s grid (temperature and type forms) and its checks go into section 14 before (a)'s code.

## 14. (a) the relational walk: its reading, fixed 25 October 2026

Written before (b)'s verdict and before any (a) code or number. Code: `outputs/mp_unified/zu2m_a.py`, which imports
zu2m.py, zu2m_b.py, U2e, P1d and P1e unchanged. Pools only: nothing trains, and no weight, type or fit reads a
question's gold or label. The choice reads golds on s1sel, as every pool stage's choice does.

**The weight.** The five walks the carried recipe reads (section 13: A1st, A1all, Kst, KTst and KTall) keep their
restarts, their α, ε and NMAX, and their equal split over a node's non-empty families. Within the structural family,
node u's pushed mass goes to its neighbour v in proportion to

    ω_q(t(u→v)) × s(v)

- s(v) is the walk's own weight: 1 on A1 and K's walks, and P1e's steer, the first stage's RRF weight, on KT's.
- t(u→v) is the arrival's type, from the form below.
- ω_q(t) = exp(β × z_q(t)), where z_q(t) is c_q(t) = cos(q, r_t) standardized over the graph's type inventory:
  minus the mean over the types, divided by their standard deviation, unweighted. q is the question's frozen dense
  vector, the one U2e's `cos` reads (`prep.qemb`). r_t is the type's vector.
- The ner and knn families keep their split (ω = 1). Their edges carry no type in either form.
- β = 0 gives ω = 1 on every edge, so the walk is the carried one. β = 0 runs the carried kernels.

**The types.** Two forms, each one rule on all six:
- **RWr.** On the KB's structural family (metaqa, webqsp), t is the fact's relation. r_t is its filed text vector
  (`outputs/m3b/relations/<D>_rel_embeddings.npy`, the frozen encoder, 13 September), with both shas checked against
  that directory's RECORD.json. The relation vector has no direction, so both directions of a fact share it. When
  u and v share several facts, the arrival's ω is the mean of their ω. On the passage graphs, RWr is RWe.
- **RWe.** Latent types, fit by EM on the graph alone, with no question and no gold, on every dataset (the KB's
  relation labels are not read).
  - An arrival u→v of the structural family has the feature f = (x_v − x_u)/‖x_v − x_u‖, from the frozen node
    vectors x (`DenseNodes`, as U2e reads them). Its reverse has −f.
  - The EM is spherical k-means (hard EM for a mixture of von Mises–Fisher components with one shared
    concentration), K = 64.
  - The fit sample is 200,000 stored pairs (all of them if fewer), uniform without replacement (numpy
    RandomState(0)), each in both directions. The start is 64 of the sample's features drawn by the same generator. The fit runs 25 steps, or
    stops when no assignment changes. An empty component keeps its centre.
  - Each arrival takes the component of highest μ_k · f. On every edge that is the argmax over k of
    S[v, k] − S[u, k], with S = X μᵀ, because ‖x_v − x_u‖ does not depend on k. An arrival with x_v = x_u takes a
    reserved null type with z = 0.
  - r_t is the unit centre μ_t. So ω weighs an arrival by how far its type's mean direction moves toward the
    question.

**Why not (i) and (ii) here.**
- Form (i) (the family, the link kind and the direction) has no vector. So a gold-free ω could not depend on the
  question. It goes to U2n as an edge-type input, where the models learn the weight.
- Form (ii)'s first vector, the entity's own node vector, makes ω a function of the target's own cosine, which the
  steer already reads on the KT walks.
- Its second vector, the mean of the passages that mention the entity, needs each link's entity label. U2n's EL
  inputs carry that label.

Both go to U2n, and neither is closed by this file.

**The grid.** The form is RWr or RWe, and β is 0.5, 1 or 2. That gives six universal points, plus β = 0, the
carried walk. On the passage graphs RWr and RWe coincide, so each passage dataset walks three configurations and each
KB dataset six. K = 64 is fixed. Another K, or EM over the ner and knn families, is a later form.

**The candidates.**
- The carried frontier family is the one (b)'s verdict carries: its points at the G of (b)'s pick, or at G = 0 when
  (b) is NO_GAIN.
- Each point is crossed with the six grid points. At G > 0 the restore is (b)'s, recomputed on the new order.
- At β = 0 the family is the carried frontier itself. The order is β = 0 first, then the forms in the order above,
  then β ascending, then the points' order. So ties go to the carried walk, and then to the weaker weight.
- Each candidate's order is the carried order rebuilt on the relational walks:
  - KIND_T's switch, RRF5T's R0, N1, the cosine order of U_0, U2e's budgets and P1h's cut are unchanged;
  - U_0 changes only through KIND_T's order.

**The choice** is section 7's rule, with (b)'s cost, levels, cap and targets.

| verdict (a) | when |
| --- | --- |
| RELATIONAL_RAISED | the pick's level is above the carried pick's |
| RELATIONAL_CHEAPER | the same level, β > 0, and a lower cost than the carried pick |
| NO_GAIN | otherwise: the pick is the carried one at β = 0 |

**Carried forward** to U2n: the (a) pick with its form and β when RAISED or CHEAPER, otherwise (b)'s carried pick.
When (a) is RAISED or CHEAPER, section 13's arrays are computed again on its walks for U2n.

**Checks.** Any failure stops the dataset, and nothing is written.
- **β = 0.** For every question, the β = 0 candidates' rows and held golds equal (b)'s filed table at the carried G.
- **The relational kernel.** It weighs each edge by w = ω × s(v) in the carried spread's statements: the total, the
  base m / (nf × total) and each amount base × w. With every ω equal to 1, w is s(v) exactly. So its p and its
  top-NMAX order equal the carried kernel's, bit for bit (self-test on toy graphs, typed and untyped, steered and
  unsteered). On toys with ω ≠ 1, its p equals a
  plain Python push within 1e-12 relative.
- **The types.** The relation vectors' and vocabularies' shas equal RECORD.json's. The EM fit is a function of the
  graph and the node vectors alone. It is fit twice in the self-test with the same result. Its centres' sha256 is
  filed.
- **No gold before the choice.** The walk, ω and EM code paths take no gold argument.
- **Development runs** (`--limit`) write nothing and log no number.

**Beside, deciding nothing.**
- Per graph: the EM components' sizes and their mean cosine to their centre.
- On the KB: the normalized mutual information between the EM type and the fact's relation, over the structural
  edges. This shows whether EM recovers the relations without their labels.
- Per dataset, form and β: ALL and rows at the carried pick's point.
- The (a) pick's misses by section 7's causes, against the carried pick's.
- The mean share of a question's ω mass on its top type.
- webqsp's s1eval for every candidate, as everywhere.

**Where it runs.** As section 13 says:
- the laptop runs squad, metaqa, musique and webqsp, after (d);
- hotpotqa and 2wiki run on Modal if the remaining credit covers them, or on the lab host once it is reachable.
  Otherwise they are held, and with them the verdict.
- The EM fit runs where the dataset's walks run.
- The arrays go to `outputs/u2m/a/stats/<D>__<carve>.npz` and `.json`, and stay uncommitted.

**The families.** (a) runs on the families the carried recipe walks today: structural, ner and knn. U2l's choice (E1
in place of W2_S and NER) changes the families. It is declared in its own dated section of this file, with its own
run, and the relational weight applies to whatever the structural family is there.

A verdict closes only the form tested (docs/SCREENS.md, section 3b): here RWr and RWe at K = 64 with these β.

## 15. (b): filed 25 October 2026

**NO_GAIN.** The pick stays U2e's, `own/kind_t/kt_cos:ada_f12_k0@0.85` at G = 0: level 0.955, 1,812.6 rows, with a
cap of 1,953.5 (outputs/u2m/b/verdict_b.json and .md). There were 56 candidates: the 8 points of the own `kind_t`
frontier × the 7 values of G. Section 11's checks held:
- at G = 0, every point's ALL and rows equal (c) and (e)'s filed values on all six datasets;
- the frontier over the G = 0 candidates returns the carried frontier's pick at every level.

**At the carried point, by G** (ALL / mean rows; squad holds 0.9793 / 50 at every G, because it restores nothing;
webqsp is s1eval, beside):

| G | cost | metaqa | musique | hotpotqa | 2wiki | webqsp |
| ---: | ---: | --- | --- | --- | --- | --- |
| 0 | 1,812.6 | 0.9619 / 4,384.4 | 0.9635 / 5,437.4 | 0.9556 / 561.2 | 0.9666 / 839.9 | 0.8929 / 4,754.6 |
| 10 | 2,220.5 | 0.9840 / 5,695.7 | 0.9648 / 6,569.7 | 0.9562 / 588.3 | 0.9672 / 869.7 | 0.9295 / 5,443.2 |
| 50 | 2,446.5 | 0.9840 / 6,042.1 | 0.9681 / 8,081.1 | 0.9562 / 594.7 | 0.9679 / 880.8 | 0.9428 / 5,923.1 |
| 500 | 2,617.2 | 0.9840 / 6,302.6 | 0.9726 / 9,249.9 | 0.9562 / 595.0 | 0.9679 / 882.8 | 0.9474 / 6,267.6 |

**Why there is no gain.**
- **The restore works where the cut drops part of a small list.** On the KB, the restored lists are (relation,
  direction) groups.
  - metaqa: at G = 10 it recovers 174 of its 287 TAU_CUT misses, and ALL goes from 0.9619 to 0.9840.
  - webqsp: 257 golds at G = 10 and 437 at G = 500, of its 664 TAU_CUT misses.
- **The level is set by the weakest dataset.** That is hotpotqa, and it gains one gold (0.9556 → 0.9562). 38 of its
  69 misses are BUDGET, and only 13 are TAU_CUT. 2wiki gains 2 of 50, and squad has no TAU_CUT misses.
- **The rows cost.** At G = 10, metaqa restores 1,311.4 rows and musique 1,132.3, so the cost rises from 1,812.6 to
  2,220.5, which is 1.14× the cap.
- **On the frontier.** G > 0 first appears at level 0.990 (`kt_cos+n2_30:fix_15000@0.95` at G = 10, 7,288.5 rows),
  far past the cap.

**The carried pick's misses recovered** (golds; G = 0 misses by cause, then recovered at G):

| dataset | missed | TAU_CUT / BUDGET / UNREACHED | G = 10 | G = 50 | G = 500 |
| --- | ---: | --- | ---: | ---: | ---: |
| metaqa | 329 | 287 / 8 / 34 | 174 | 175 | 175 |
| squad | 31 | 0 / 30 / 1 | 0 | 0 | 0 |
| musique | 62 | 22 / 9 / 31 | 2 | 8 | 16 |
| hotpotqa | 69 | 13 / 38 / 18 | 1 | 1 | 1 |
| 2wiki | 50 | 13 / 31 / 6 | 1 | 2 | 2 |
| webqsp (s1eval) | 2,077 | 664 / 369 / 1,044 | 257 | 334 | 437 |

**What follows.**
- (d) and (a) read the carried pick at G = 0, as sections 13 and 14 say for NO_GAIN.
- The verdict closes only this restore with these G (docs/SCREENS.md, section 3b). The restore has signal where lists
  are typed, but its rows grow with the lists.
- The next form is a restore with a row budget. Lists are taken in the order of their best member's place until the
  restored rows reach a fixed share of keep. A question with many small lists then gets few rows, on any dataset. It
  is declared in its own dated section before any code.
- Restored-list membership is also an input U2n can learn from: (d)'s split already carries the structural and
  (relation, direction) parts.

## 16. (d) on hotpotqa and 2wiki: where it runs, 25 October 2026 (logistics only, before any of their numbers)

Section 13 sends (d)'s hotpotqa and 2wiki to Modal if the credit covers them, and holds them otherwise.
- W5 has about $1.30 left after (a)'s Modal run, which is too little, and the lab host is unreachable.
- So the laptop also runs them, once U2m's (a) and (d) laptop runs and U2o's laptop runs have finished.
- A machine cannot change a number here. Section 13's checks hold every question's walk order, masses and rows to the
  filed ones, or the run stops.
- (a)'s filing follows as section 18; (d) finished first and is filed as section 17.

## 17. (d): filed 25 October 2026 (beside; no verdict)

(d) has no verdict (section 13). Its arrays are an input for U2n's models. They now exist on all six datasets at the
carried pick, `own/kind_t/kt_cos:ada_f12_k0@0.85` at G = 0 (outputs/u2m/d/beside_d.json and .md; one record per
dataset under outputs/u2m/d/stats/, with the arrays uncommitted).
- **Checks.** Section 13's checks held on all six:
  - every walk's order equals the original kernel's;
  - the A1 and K masses equal walk_mass's, bit for bit;
  - the parts sum to p within the bound;
  - each question's rows and held golds equal the carried pick's.
  - On metaqa and webqsp the (relation, direction) parts also sum to the structural part. That check does not apply
    on the passage graphs, whose records read False for it.
- **Where it ran.** All six ran on the laptop. The first attempt on squad, metaqa, musique and webqsp stopped while
  parsing (b)'s pick (fixed in 17fc610) and wrote nothing; the files here are the reruns. hotpotqa and 2wiki ran as
  section 16 says.

**Beside: where a row's mass comes from.** Mean shares over the rows with mass, on KTall (the steered walk over all
three families). squad's construction has no K walks, so its row reads A1all.

| dataset | restart, gold / other | ner + knn, gold / other | structural share of the edge-borne mass, gold / other |
| --- | --- | --- | --- |
| metaqa | 0.0001 / 0.0006 | 0.126 / 0.256 | 0.874 / 0.744 |
| squad (A1all) | 0.874 / 0.617 | 0.101 / 0.253 | 0.203 / 0.340 |
| musique | 0.403 / 0.001 | 0.326 / 0.597 | 0.454 / 0.403 |
| hotpotqa | 0.509 / 0.005 | 0.202 / 0.532 | 0.589 / 0.465 |
| 2wiki | 0.570 / 0.001 | 0.142 / 0.519 | 0.671 / 0.481 |
| webqsp (s1eval, beside) | 0.002 / 0.001 | 0.142 / 0.348 | 0.858 / 0.652 |

- **Restart.** On the passage graphs' steered walks, 0.40 to 0.57 of a gold row's mass is restart, against under
  0.006 on other rows. These are golds among the seeds, which the models already see.
- **Structural.** Of the mass that arrives over an edge, gold rows draw a larger share from the structural family
  than other rows on five of six datasets. squad is the exception; there a gold row's mass is mostly restart.
- **ner and knn.** On every dataset, ner and knn arrivals make up a smaller share of a gold row's mass than of
  another row's. That agrees with U2l's choice, which drops NER from the passage graphs' walks (U2l, section 10).
- **Relations, on the KBs.**
  - metaqa: on the KT walks, the gold rows' structural mass comes most from starred_actors and written_by (the row as
    the stored source). On the A1 walks it comes most from has_genre and release_year (the row as the destination).
  - webqsp: the mass spreads over many relations, with no pair above 0.075.

**What follows.** U2n reads the parts as inputs, computed with these kernels on its fit and evaluation carves
(section 13). If a later pool stage changes the carried pick (U2o, or E1 in its own section), the arrays are computed
again on the new pick, in their own file. Nothing here closes or decides anything.

## 18. (a): filed 25 October 2026

**RELATIONAL_CHEAPER.** The pick is `own/kind_t/kt_cos:ada_f12_k0@0.85` at G = 0 with RWe at β = 0.5: level 0.955
for 1,809.6 rows, against (b)'s 1,812.6 for the same point at β = 0, with a cap of 1,953.5
(outputs/u2m/a/verdict_a.json and .md). There were 56 candidates: the 8 points of the carried frontier at G = 0 × the
7 configurations. These are development numbers: s1sel, with webqsp's s1eval beside.
- **Checks.** Section 14's checks held on all six.
  - At β = 0, every question's rows and held golds equal (b)'s filed table at G = 0.
  - In the verdict, the β = 0 candidates equal (b)'s filed values on all six datasets.
  - The kernel's self-test (every ω 1 equals the carried kernels bit for bit; ω ≠ 1 equals a plain push within
    1e-12) passed before the code's commit.
- **Where it ran.**
  - squad, metaqa, musique and webqsp ran on the laptop.
  - hotpotqa and 2wiki ran on Modal, in 4 question shards each, joined by `merge`, with their EM types fit there.
  - The verdict reads every dataset's types record. hotpotqa's and 2wiki's were pulled from the volume for it, and
    each array's sha256 equals its record's and the stats' `types_sha256`.

**At the carried pick's point** (ALL / mean rows; squad holds 0.9793 / 50.0 in every configuration; webqsp is s1eval,
beside):

| configuration | cost | metaqa | musique | hotpotqa | 2wiki | webqsp |
| --- | ---: | --- | --- | --- | --- | --- |
| β = 0 | 1,812.6 | 0.9619 / 4,384.4 | 0.9635 / 5,437.4 | 0.9556 / 561.2 | 0.9666 / 839.9 | 0.8929 / 4,754.6 |
| RWr, β = 0.5 | 1,812.9 | 0.9880 / 4,384.9 | 0.9654 / 5,439.4 | 0.9569 / 561.2 | 0.9652 / 839.9 | 0.9208 / 4,759.3 |
| RWr, β = 1 | 1,817.4 | 0.9900 / 4,396.8 | 0.9674 / 5,459.6 | 0.9556 / 561.2 | 0.9619 / 839.9 | 0.9308 / 4,794.4 |
| RWr, β = 2 | 1,793.5 | 0.9733 / 4,255.5 | 0.9654 / 5,530.0 | 0.9463 / 561.2 | 0.9525 / 839.9 | 0.9215 / 4,848.2 |
| RWe, β = 0.5 | 1,809.6 | 0.9653 / 4,369.9 | 0.9654 / 5,439.4 | 0.9569 / 561.2 | 0.9652 / 839.9 | 0.9069 / 4,747.0 |
| RWe, β = 1 | 1,810.6 | 0.9439 / 4,365.4 | 0.9674 / 5,459.6 | 0.9556 / 561.2 | 0.9619 / 839.9 | 0.9082 / 4,768.3 |
| RWe, β = 2 | 1,800.9 | 0.8584 / 4,289.7 | 0.9654 / 5,530.0 | 0.9463 / 561.2 | 0.9525 / 839.9 | 0.8809 / 4,841.4 |

**What the verdict says.**
- **The saving is small.** At the same level the pick costs 3.0 rows less. Against β = 0, its ALL rises on metaqa
  (0.9619 → 0.9653), musique (0.9635 → 0.9654) and hotpotqa (0.9556 → 0.9569), and falls on 2wiki (0.9666 → 0.9652).
  The frontier's 0.95 point moves from 1,497.8 to 1,495.1. No point reaches 0.96 within the cap.
- **On the KBs, RWr is the larger signal.** The KB relations' text, weighted toward the question, takes metaqa from
  0.9619 to 0.9880 at β = 0.5 and to 0.9900 at β = 1, and webqsp's s1eval from 0.8929 to 0.9208 and 0.9308.
- **Why the rule takes RWe.** On the passage graphs RWr is RWe, so the two forms differ only on the KBs.
  - The level is set by hotpotqa, which holds 0.9569 under both, below 0.96. So metaqa's gain does not move the
    level.
  - At 0.955 the rule takes the lower cost: RWe's KB walk keeps fewer metaqa rows (4,369.9 against 4,384.9), so it
    costs 3.3 less.
- **β = 2 is too sharp.** hotpotqa falls to 0.9463 and 2wiki to 0.9525 under both forms, and RWe takes metaqa to
  0.8584.
- **The pick's misses.**
  - metaqa: 329 missed golds become 207 (141 TAU_CUT and 1 UNREACHED recovered, 20 TAU_CUT lost).
  - webqsp (s1eval): 2,077 become 2,011 (132 TAU_CUT, 14 BUDGET and 4 UNREACHED recovered, 84 TAU_CUT lost).
  - musique, hotpotqa and 2wiki: 62 → 59, 69 → 67 and 50 → 52.
  - squad: unchanged at 31.
- **The EM types (beside).**
  - The fit ran its 25 steps on every graph without reaching a step with no change.
  - The mean cosine to the centre is 0.29 (squad) to 0.52 (metaqa).
  - The NMI between the EM type and the fact's relation is 0.355 on metaqa and 0.480 on webqsp. So EM recovers part
    of the relations without their labels.

**What follows.**
- **The carried walk.** It is (a)'s pick: `own/kind_t/kt_cos:ada_f12_k0@0.85` at G = 0, RWe at β = 0.5, level 0.955,
  1,809.6 rows.
  - As section 14 says, section 13's arrays ((d)) are computed again on its walks for U2n, in their own file.
  - U2p (docs/U2P_LINK_SCORE_WALK.md) puts U2k's link score into this walk.
  - E1 (section 19) is measured on the walk its code reads, the β = 0 one, against U2e's pick on that walk.
- **This verdict closes only RWr and RWe at K = 64 with these β under section 7's rule** (docs/SCREENS.md,
  section 3b).
  - Section 7's rule values coverage only up to the level, and the level is set by the passage graphs. So RWr's KB
    gain is signal the rule does not carry.
  - It reaches both of U2n's models as an input: each arrival's relation type and its ω_q, so the models can learn
    the weight.
  - EM over the ner and knn families, and another K, stay open.

## 19. E1 as the entity family: its reading, fixed 25 October 2026 (before its code)

U2l filed CHOSEN E1 (docs/U2L_ENTITY_MATCH.md, section 10) and fixed what follows: U2m reads E1 in place of W2_S and
NER as its entity family, declared in a dated section here before its code. This is that section. Code:
`outputs/mp_unified/zu2m_e1.py`, which imports zu2m.py and zu2o.py unchanged. Outputs go only under
`outputs/u2m/e1/`. Pools only: nothing trains.

**What changes: only the `ner` slot** of the three families (structural, ner, knn).
- **On a passage graph** (musique, squad, hotpotqa, 2wiki) the slot is empty. E1 with the sibling links is already
  the structural graph: U2l's gate 1 found E1 ∪ R equal to U2b's filed W2_S, array for array, and that file is the one
  U2m swaps in. So the walks read W2_S and kNN, without the package's NER.
- **On a KB** (metaqa, webqsp) the slot is U2l's IS (`outputs/u2l/el/<D>/el.npz`), undirected and collapsed by
  `zu2b.undirected_csr` as U2l graded it: rows in node-id order, self-loops kept. Every entry has weight 1; no pool
  rule reads a family's weight. The stored triples and kNN stay.

**What that moves, per question:**
- the walks on the `all` union (A1all, Kall, KTall, A1Tall), and through them KIND_T;
- N1 and N2, which read all three families;
- the I where the own construction reads ner: musique (FULL:h2_c25) and webqsp (FULL:h3_c25_v2000);
- so the own cell's `kind_t` columns: the fill, the 14 orders' lengths and every gold's place.
- The `st` walks (A1st, Kst, KTst, A1Tst), R0, B, the seeds and the question's kind do not change. Neither does the I
  on metaqa, hotpotqa and 2wiki (STRUCT) or on squad (RETRIEVAL).

**The candidates and the choice** are U2o's (docs/U2O_K1_I_ORDER.md, section 3), with E1's columns:
- U2e's 13,860 candidates on the own construction with `kind_t`, on U2m's arrays with the own cell's `kind_t` columns
  (`fill`, `lens`, `pos`) replaced by E1's on all six datasets;
- section 7's rule, with P1f's weights, the cap of 1,953.5 and (c) and (e)'s filed ceilings.

| verdict | when |
| --- | --- |
| E1_RAISED | the pick's level is above 0.955 |
| E1_CHEAPER | the pick's level is 0.955 and its cost is below 1,812.6 |
| NO_GAIN | otherwise: U2e's pick stays |

The incumbent is U2e's pick on the walk zu2m_e1.py reads, the β = 0 one: `own/kind_t/kt_cos:ada_f12_k0@0.85` at
level 0.955 for 1,812.6 rows, as (b) left it (section 15), and as U2o carried it. (a)'s pick, RWe at β = 0.5 for
1,809.6 rows (section 18), is now the carried walk. E1 is still measured on the walk its code reads, so the rule
compares E1 with the same walk without it. If E1 gains, E1 on (a)'s walk is declared in its own section before it
runs.

**Carried forward.**
- RAISED or CHEAPER: E1 becomes the pools' entity family. U2m's (d) and (a) arrays, which read the carried pools, are
  computed again on it for U2n, each in its own file.
- NO_GAIN: the pools keep the package's NER.
- Either way, EL's kinds (IS, CO_t, CO_s) enter U2n's models as edge types (U2l, section 8).
- U2o's K1 order and U2p's link weight (docs/U2P_LINK_SCORE_WALK.md) are different forms, tested on the filed
  families. If E1 and one of them both gain, their combination is declared in its own file before it runs.

**Checks.** Any failure stops the dataset, and nothing is written.
- Each question walks twice: on the filed families and on E1's.
  - On the filed families, |B|, |I|, the fill, the 14 orders' lengths, every gold's place and which golds the I holds
    equal U2m's filed own `kind_t` columns, question for question.
  - Where the construction reads ner (musique, webqsp), the I is rebuilt on both families, and the one rebuilt on the
    filed families equals `prepare`'s pools. Elsewhere both sides take `prepare`'s pools, which E1 cannot move.
  - On E1's families, the `st` walks, R0 and the kind equal the filed ones.
- E1's store: `el.npz` carries its build record's sha256; on a passage graph U2l's gate 1 is PASS on W2_S and U2b's
  filed W2_S is the file U2m swaps in; U2l's choice is CHOSEN E1.
- In the verdict, on the filed families: the table equals U2e's for all 13,860 candidates, the frontier equals the
  filed own `kind_t` frontier, and the rule returns U2e's pick.
- **Self-test** (`--selftest`), passed before the code's commit: `undirected_csr` against a brute force; E1 in the
  slot changes the `all` union only; the package's NER in the slot gives the filed union back; the empty slot; the
  summary; the verdicts.
- **Development runs** (`--limit`) write nothing and log no number. With `--control`, the package's NER stands in
  E1's slot, so every question's E1 side must equal its filed side. 30-question runs on metaqa and musique, with and
  without `--control`, passed before the code's commit.

**Beside, deciding nothing:**
- per dataset, the I's ALL and size on both families, and the share of questions whose KT order E1 leaves unchanged;
- U2e's point under E1, per dataset;
- the misses by U2m's causes (TAU_CUT, BUDGET, UNREACHED);
- webqsp's s1eval, as everywhere.

**Where it runs.** The laptop, one job at a time for memory, after U2p's laptop runs: metaqa, squad, musique and
webqsp first, then hotpotqa and 2wiki. The lab host takes them if it is reachable first. W5's credit (about $1.30) is
too little for Modal. The filed-family reproduction is what ties a machine to U2m's filed run.

A verdict closes only the form tested (docs/SCREENS.md, section 3b): E1 in the ner slot, with the `kind_t` recipe and
U2e's grid. On a passage graph E3 ∪ R ∪ kNN is the filed families (U2l, section 10), so the forms left open are EL's
kinds as typed edges: as walk types in (a)'s relational walk, and as edge types in U2n's models.
