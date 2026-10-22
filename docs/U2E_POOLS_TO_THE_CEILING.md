# U2e: the pools toward the ceiling, with a second link and fused orders

Declared 22 October 2026, before any U2e number. The user asked on 22 October whether 0.95 is the most the pools can
reach, whether they can get as close to 100% as possible, and whether a better universal link rule and more precise
pools exist. Code: `outputs/mp_unified/zu2e.py`, importing U2c, P1g, P1h, P1f, P1d and D5b unchanged. Pools only:
nothing trains, nothing is fit, and no order, budget or cut reads a gold. The choice reads golds on s1sel, as every
pool stage's choice does.

## 1. Why

U2c (docs/U2C_POOLS_ON_CHOSEN.md, section 9) chose `g:f3+cut_cos_7000@0.9` on W2_S and filed TARGET_MET: ALL ≥ 0.95
on every training dataset's s1sel, at a cost of 1,953.5 rows. 0.95 was the user's target, not a ceiling.

**Read before this declaration.** The numbers below come from U2c's filed s1sel arrays (webqsp's s1eval). They were
read to frame the question and are not results of this stage. "All sources" counts a gold held when any U2c candidate,
KIND_T's full order, the first stage's lists, B's one-link neighbours or the frozen I holds it.

| dataset | U2c's chosen ALL | its rows | the best single U2c candidate | all sources |
| --- | --- | --- | --- | --- |
| metaqa | 0.972 | 3,929 | 0.992 | 0.996 |
| squad | 0.979 | 50 | 0.993 | 0.999 |
| musique | 0.951 | 5,651 | 0.971 | 0.977 |
| hotpotqa | 0.954 | 964 | 0.960 | 0.987 |
| 2wiki | 0.961 | 1,407 | 0.966 | 0.995 |
| webqsp (beside) | 0.878 | 4,555 | 0.949 | 0.958 |

- **Most of the gap is order, not reach.** On hotpotqa and 2wiki, the golds no U2c candidate holds but KIND_T's
  longer order does sit around its 1,100th to 1,200th node. KIND_T's first 500 nodes alone hold ALL 0.960 on hotpotqa
  and 0.967 on 2wiki, at about 550 rows.
- **The rest needs links.** Golds outside every source: musique 38 (35 questions), hotpotqa 21, 2wiki 7, metaqa 42
  (6 questions), squad 1, webqsp 1,391 (63 questions).

U2e therefore does two things. It tests U2B section 7's named next link form, a second link from the question's best
X rows. It also fuses the sources into single orders per question with per-question budgets, so a pool can hold more
golds in fewer rows. It files the whole coverage-against-rows frontier up to each dataset's ceiling, and picks the
highest coverage that fits in no more rows than U2c's chosen pool.

## 2. The graph and the carves

The graph is U2c's chosen graph: W2_S swapped in for `structural` on musique, squad, hotpotqa and 2wiki, exactly as U2c
swaps it (`zu2c.graph_csr` and `zu2c.csr_store`, unchanged). metaqa and webqsp keep their KB. The carve is s1sel on the
five training datasets, with webqsp's s1eval beside. B (base and seeds), KIND_T's order, today's fill and the frozen I
are U2c's, rebuilt the same way and checked against its filed arrays (section 7).

## 3. The sources

Per question, every source is a set of nodes outside B:

- **KT:** KIND_T's order (U2c's `full`; at most 7,000 nodes with B).
- **N1:** B's one-link neighbours over the three families (structural, ner, knn), from D5b's `n1_table`.
- **R0:** the first stage's RRF lists (P1d's R0: the dense and SPLADE lists, 1,000 deep each).
- **I:** the frozen I, rebuilt on the graph.
- **N2(M):** the second link. These are the neighbours, over the three families, of N1's first M nodes by the
  question's cosine, outside B and N1. M is 10, 30 or 100. The parents are taken in that cosine order, under two
  guards that are P1f's 7,000-node cap (NMAX), the cap on every pool order:
  - a parent with more than 7,000 links is passed over (a hub; W2_S's largest node has 621,403 links on 2wiki), and the
    next N1 node takes its place;
  - the parents stop at the first one after which N2 holds at least 7,000 nodes. That parent is kept whole.

  N2(10), N2(30) and N2(100) are nested. Each question records the parents taken and the hubs passed over.

U_M = KT ∪ N1 ∪ R0 ∪ I ∪ N2(M), and U_0 has no second link.

**The cosine.** cos(v) = q · v̂, the question's unit embedding against v's unit dense embedding, both float32, as
D5b computes it. N1's values are D5b's own expression over N1 (`unit(read(N1)) @ q`), and the other nodes are read in
fixed chunks of 65,536 rows.

## 4. The orders

All ties go by node id. RRF is P1d's `rrf` with the first stage's constant (60), uncut: a node scores the sum over the
orders that hold it of 1/(60 + its rank).

| order | definition | with N2? |
| --- | --- | --- |
| `kt` | KIND_T's order | no |
| `kt_n1` | RRF of KT and N1 by cosine | no |
| `cos` | U_M by cosine | M = 0, 10, 30, 100 |
| `kt_cos` | RRF of KT and `cos` | M = 0, 10, 30, 100 |
| `kt_cos_r0` | RRF of KT, `cos` and R0 (in R0's own order) | M = 0, 10, 30, 100 |

That is 14 orders. An order with N2 is named with `+n2_M`, for example `kt_cos+n2_30`.

## 5. The budgets, the cut and the candidates

A pool is B, then the first k_q nodes of one order (all of it when the order is shorter), then P1h's cut at τ on that
build order. The cut keeps the shortest prefix whose rank mass reaches τ of the total, never fewer than |B|
(docs/P1H_FEWER_ROWS.md, section 2). k_q is set in one of two ways:

- **FIX(k):** the same k for every question, with k in 25, 50, 100, 150, 200, 300, 400, 500, 750, 1,000, 1,500,
  2,000, 3,000, 4,000, 5,000, 7,000, 10,000, 15,000 and 20,000 (19 budgets).
- **ADA(f, k0):** max(k0, round(f × today's fill)), with today's fill from U2c (|I_q| − |B_q| on the graph). f is 1,
  1.5, 2, 3, 4, 6, 8, 12, 16 or 24, and k0 is 0, 25, 50, 100, 200, 300, 500 or 1,000 (80 budgets).

τ takes P1h's ten values: 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99 and 1. That gives 14 × 99 × 10 = 13,860
candidates, named `<order>:<budget>@<τ>` (for example `kt_cos+n2_30:fix_500@0.9`).

U2c's 6,706 candidates enter too, read from its filed arrays and unchanged, named `u2c:<name>`. They come first, so
ties go to them. There are **20,566 candidates** in all.

## 6. The choice

On the five training datasets' s1sel:

- **Ceiling.** C_d is the best ALL any candidate reaches on dataset d.
- **Levels.** t runs from 0.950 to 1.000 in steps of 0.005. At each level, the target on dataset d is
  min(t, C_d − 0.005), with 0.005 being P1g's tolerance. The frontier's point at t is the cheapest candidate that meets
  every target, by P1f's question-weighted mean rows (U2c's weights). Ties go to the earlier candidate.
- **The cap.** The cap is U2c's chosen cost, 1,953.5 rows. P1g-b's refit ran on pools of 1,975.5 rows
  (docs/P1GB_POOLS_REFIT.md), which used about 120 GB on the host and most of its GPU. The cap keeps U2d's refit
  within that size.
- **The pick** is the frontier's point at the highest level whose cost is within the cap.

| verdict | when | what goes to U2d |
| --- | --- | --- |
| RAISED | the pick's level is above 0.95 | the pick |
| FEWER_ROWS | the level is 0.95 and the pick costs at most 0.9 of U2c's chosen (P1h's MIN_CUT, 0.10) | the pick |
| NO_CHANGE | otherwise | U2c's chosen, unchanged |

**Filed beside, past the cap too:**
- the whole frontier, with each dataset's ALL and rows at every level;
- the cheapest point of each order family (U2c's, `kt`, `kt_n1`, `cos`, `kt_cos`, `kt_cos_r0`) and of each M at every
  level;
- per dataset, the fewest mean rows that reach each level. This is an upper bound only: a per-dataset pick is not a
  universal rule;
- each dataset's union ceiling: ALL when a gold counts as held if any source or U2c candidate holds it;
- webqsp's s1eval, for the pick and for each level's point. It never moves the choice.

## 7. Checks

Any failure stops the item, and nothing is written.

**Every question, against U2c's filed arrays on the same graph** (the chosen graph's runs; metaqa and webqsp: U2c's
`today` runs, the same KB):
1. |B|, today's fill and |I_q|; per gold, KIND_T's place, whether the first stage's lists hold it, whether it is one
   link from B and whether the frozen I holds it; and the graph's npz sha256.
2. U2c's `place_question`, unchanged, is given B, KIND_T's order, today's fill, and N1 cut at 7,000 by U2e's cosine
   ranks. For `f3+cut_cos_7000` and `f3+none` (U2c's chosen rule and its shut gate), it must give U2c's filed rows and
   build-order places. This checks N1, its cosine order and KIND_T's order together.

**In the code:** every order holds each node once, none of B, and exactly its sources' nodes. Every gold's place in
every order is its index there.

**Self-test** (`python outputs/mp_unified/zu2e.py --selftest`, passed before this commit):
- the second link, its guards and its nesting against a direct build on toy graphs;
- every order against a direct sort or RRF;
- the gold places;
- the budgets, the cut and the summary against a per-question count;
- the frontier, the pick and the verdict on toy tables;
- the shards and the merge.

**Development runs** (`--limit`) write nothing and log no number.

## 8. Where it runs, and the outputs

On Modal, in question shards, as U2c's stats ran (`outputs/mp_unified/zu2e_modal.py`, the same image, pins and
mirror):

    modal run outputs/mp_unified/zu2e_modal.py --stage dev --datasets D [--limit 30]
    modal run --detach outputs/mp_unified/zu2e_modal.py --stage stats --datasets D1,D2,... --shards K
    python outputs/mp_unified/zu2e.py verdict

- `zu2e.py stats D --shard I/K` runs every check on the I-th of K question ranges.
- `zu2e.py merge D --parts K` joins the parts.
- U2c's filed arrays are read from the volume where U2c's run wrote them, and are never written.
- U2e's outputs go to a new directory on the volume, named by zu2e.py's sha256. They are fetched to
  `outputs/u2e/stats/<D>__<carve>.npz` and `.json`.
- The verdict runs on the laptop, from the fetched stats and U2c's filed arrays, and writes `outputs/u2e/verdict.json`
  and `verdict.md`.

The arrays stay uncommitted.

## 9. What follows (fixed now)

**U2d** is declared on its own after this verdict. It builds U2d's pool on the fit and s1eval carves of all six
datasets: the pick when RAISED or FEWER_ROWS, U2c's chosen when NO_CHANGE. It grades s1eval beside, builds the KT input
on those pools, and refits the lean pair (zrckp, zspk) and the free GNN (G6). Each model adopts the pools only through
its own declared screen.

The second link counts as reaching a gold only through an order that holds it. If no order with N2 is on the frontier
within the cap, the second link is reported with its own union ceiling and is not taken. U2B section 7's other next
form, the question-guided hop, stays named for the golds outside every source.

A verdict closes only the form tested (docs/SCREENS.md, section 3b).
