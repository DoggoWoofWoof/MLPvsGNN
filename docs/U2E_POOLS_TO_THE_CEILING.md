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

## 10. Results (22 October)

**RAISED.** The pick is `kt_cos:ada_f12_k0@0.85`. Per question, it fuses KIND_T's order with the cosine order of U_0
by RRF, then takes the first 12 × today's fill nodes, then applies P1h's cut at τ = 0.85. There is no floor: a question
whose fill today is 0 keeps B alone, as the gate did. It meets level 0.955 on every training dataset at 1,812.6 rows,
0.928 of U2c's chosen pool (1,953.5).

The records are `outputs/u2e/verdict.json` and `verdict.md`, written by the committed `zu2e.py verdict` (sha256
`9f781ecf0744…`). The stats records are `outputs/u2e/stats/<D>__<carve>.json`.

- **Checks.** Every check held on every question of all six datasets, run in 12 question shards per dataset on Modal:
  - the graph equals U2c's, and the seeds equal G0's;
  - B, today's fill, |I| and every gold flag equal U2c's filed arrays;
  - U2c's chosen rule and its shut gate are reproduced row for row and place for place;
  - every order is exact.
- **Re-check.** The pick and the frontier's points at 0.95, 0.96 and 1.00 were recomputed from the stats arrays
  outside zu2e's code, with a per-question loop for the cut. Every ALL, mean rows and cost matches the record.

The pick against U2c's chosen pool (s1sel; webqsp's s1eval beside). "Recall" is the share of golds in the pool:

| dataset | the pick: ALL | recall | rows | rows p90 | U2c's chosen: ALL | recall | rows | ceiling (best candidate) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 0.962 | 0.991 | 4,384 | 4,573 | 0.972 | 0.991 | 3,929 | 0.996 |
| squad | 0.979 | 0.979 | 50 | 50 | 0.979 | 0.979 | 50 | 0.999 |
| musique | 0.964 | 0.986 | 5,437 | 6,892 | 0.951 | 0.981 | 5,651 | 0.980 |
| hotpotqa | 0.956 | 0.977 | 561 | 798 | 0.954 | 0.976 | 964 | 0.986 |
| 2wiki | 0.967 | 0.988 | 840 | 1,090 | 0.961 | 0.986 | 1,407 | 0.995 |
| webqsp (s1eval, beside) | 0.893 | 0.959 | 4,755 | 5,693 | 0.878 | 0.944 | 4,555 | 0.961 |

- **musique** rises 0.012 on 4% fewer rows. **hotpotqa and 2wiki** rise a little (0.001 and 0.005) on 42% and 40% fewer
  rows.
- **metaqa** falls 0.010 on 12% more rows. Its misses are the cut's (287 of 329).
- **squad** is unchanged: its fill today is 0 on every question, so the pick keeps B.
- **webqsp** rises 0.015 beside.

**How close to 100%.** The frontier is the cheapest candidate that meets every training dataset's target, as cost in
rows. From 0.98 up, each dataset's target stops at its ceiling less 0.005 (section 6), so the last row has every
dataset within 0.005 of its ceiling:

| level | cost | against U2c's chosen | candidate | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (beside) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.950 | 1,497.8 | 0.77 | `kt_cos+n2_100:ada_f12_k0@0.8` | 0.953 | 0.979 | 0.956 | 0.952 | 0.965 | 0.879 |
| 0.955 | 1,812.6 | 0.93 | `kt_cos:ada_f12_k0@0.85` | 0.962 | 0.979 | 0.964 | 0.956 | 0.967 | 0.893 |
| 0.960 | 1,955.1 | 1.001 | `kt_cos:ada_f16_k300@0.85` | 0.962 | 0.993 | 0.964 | 0.960 | 0.973 | 0.893 |
| 0.965 | 2,324.3 | 1.19 | `kt_cos+n2_100:ada_f3_k1000@0.975` | 0.971 | 0.997 | 0.965 | 0.966 | 0.971 | 0.898 |
| 0.970 | 3,364.3 | 1.72 | `kt_cos:ada_f24_k0@0.95` | 0.989 | 0.979 | 0.975 | 0.971 | 0.983 | 0.919 |
| 0.975 | 4,542.4 | 2.33 | `kt_cos:ada_f24_k1000@1` | 0.996 | 0.997 | 0.976 | 0.976 | 0.987 | 0.957 |
| 0.980, 0.985 | 6,800.2 | 3.48 | `kt_cos+n2_30:fix_15000@0.95` | 0.989 | 0.999 | 0.976 | 0.985 | 0.995 | 0.917 |
| 0.990 to 1.000 | 7,404.2 | 3.79 | `kt_cos:fix_15000@0.975` | 0.993 | 0.999 | 0.976 | 0.985 | 0.995 | 0.935 |

- **Within the cap**, 0.955 is the most. 0.96 costs 1,955.1 rows, 1.6 rows over the cap.
- **Above that, rows grow fast.** 0.97 costs 1.7 times U2c's rows and 0.975 costs 2.3 times. Within 0.005 of every
  ceiling, the cost is 3.8 times (7,404 rows).
- `verdict.md` gives each point's mean rows per dataset.

**What limits the ceilings.** ALL for KIND_T's whole order, for every source together, and the golds outside every
source:

| dataset | ceiling | every source | KIND_T alone | the second link adds | golds outside every source (questions) | of them, the hyperlinks reach |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 0.996 | 0.996 | 0.990 | 0 | 42 (6) | KB |
| squad | 0.999 | 0.999 | 0.999 | 0 | 1 (1) | 0 |
| musique | 0.980 | 0.981 | 0.954 | +0.0045 | 32 (29) | 1 |
| hotpotqa | 0.986 | 0.987 | 0.985 | +0.0007 | 20 (19) | 17 |
| 2wiki | 0.995 | 0.995 | 0.995 | +0.0006 | 7 (7) | 4 |
| webqsp (beside) | 0.961 | 0.961 | 0.889 | +0.0046 | 1,194 (58) | KB |

"The second link adds" is ALL of U_100 less ALL of U_0.

- **Reach no longer limits hotpotqa and 2wiki.** KIND_T's whole order (at most 7,000 nodes) already holds every gold
  for 0.985 and 0.995 of the questions. The pools fall short of that because the golds sit deep in the order, and
  taking more of it costs rows.
- **The second link adds little reach.** N2(100) adds 0.0045 on musique and 0.0046 on webqsp, under 0.001 on hotpotqa
  and 2wiki, and nothing on metaqa and squad. Its orders are on the frontier at 0.95, 0.965 and 0.98–0.985. There,
  N2's nodes move the cosine ranks, and with them the RRF order and the cut. The pick has no second link. The hub
  guard passed over a parent on 0 (metaqa, squad) to 117 (2wiki) questions.
- **The families.** `kt_cos` is the cheapest family at every level. Adding the first stage's lists (`kt_cos_r0`) is
  never cheaper. `kt_n1` reaches 0.97 at 3,768.5 rows. U2c's grid reaches 0.955 at 2,580 rows, and KIND_T alone reaches
  0.95 at 2,473: past 0.954 musique needs B's one-link neighbours. The cosine order alone needs 8,206 rows for 0.95.

**The ledger:** the pick's misses (s1sel; webqsp s1eval):

| dataset | golds | missed | TAU_CUT | BUDGET | UNREACHED | questions missing a gold |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | 11,367 | 329 | 287 | 0 | 42 | 57 |
| squad | 1,498 | 31 | 0 | 30 | 1 | 31 |
| musique | 4,064 | 62 | 22 | 8 | 32 | 56 |
| hotpotqa | 3,016 | 69 | 13 | 36 | 20 | 67 |
| 2wiki | 3,646 | 50 | 13 | 30 | 7 | 50 |
| webqsp | 10,583 | 2,077 | 664 | 219 | 1,194 | 161 |

UNREACHED here means outside every source.

- **On squad, hotpotqa and 2wiki, most misses are the budget's.** These are golds the order holds past the
  question's budget: 30 of squad's 31 (its questions with no fill today), 36 of hotpotqa's 69 and 30 of 2wiki's 50.
  musique's are mostly golds no source reaches (32 of 62).
- **metaqa's misses are the cut's.**
- **Golds no source reaches are few:** 1 to 42 on the training datasets. On hotpotqa, 17 of the 20 are golds only the
  hyperlinks reach.

**What this answers.**
- **0.95 is not the ceiling.** On the LLM-free graph, a pool can hold every gold for 0.980 (musique) to 0.999 (squad)
  of the questions.
- **Within today's rows, 0.955 on every training dataset is the most.**
- **The last points cost rows, not links.** The remaining golds are mostly deep in orders the pools already have.

**What follows** (section 9, RAISED):
- U2d builds the pick on the fit and s1eval carves of all six datasets, declared in its own file.
- U2B section 7's other next form, the question-guided hop, stays named for the golds outside every source (32 on
  musique, 20 on hotpotqa).
- Moving coverage past 0.955 without more rows needs a better parameter-free order deep in KIND_T's walk, as its own
  stage.
