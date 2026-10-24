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
