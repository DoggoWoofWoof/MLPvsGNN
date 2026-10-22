# U2f: where each question's pool stops, without the golds

Declared 22 October 2026, before any U2f number. The user asked on 22 October: "cant we reduce the pool sizes and be
more precise maintaining the all coverage". Code: `outputs/mp_unified/zu2f.py`, importing U2e, U2c, P1e, P1d, P1h, P1g
and P1f unchanged, and `outputs/mp_unified/zu2f_modal.py`. Pools only: nothing trains, nothing is fit, and no rule
reads a gold or a dataset's name. The choice reads golds on s1sel, as every pool stage's choice does.

## 1. Why

U2e (docs/U2E_POOLS_TO_THE_CEILING.md, section 10) picked `kt_cos:ada_f12_k0@0.85`: ALL ≥ 0.955 on every training
dataset's s1sel at a cost of 1,812.6 rows. U2d refits the models on it.

**Read before this declaration.** These numbers come from U2e's filed s1sel arrays (webqsp's s1eval). They were read
to frame the question and are not results of this stage. Both stops below read golds, so neither is a rule: a
**perfect stop** gives each question exactly the rows down to its deepest gold in `kt_cos`. **One depth** gives every
question of a dataset the same depth, the least that reaches the level.

| dataset | U2e's pick: ALL | rows | perfect stop at that ALL | one depth at that ALL |
| --- | --- | --- | --- | --- |
| metaqa | 0.962 | 4,384 | 349 | 4,168 |
| squad | 0.979 | 50 | 50 | 50 |
| musique | 0.964 | 5,437 | 548 | 5,334 |
| hotpotqa | 0.956 | 561 | 69 | 525 |
| 2wiki | 0.967 | 840 | 96 | 832 |
| webqsp (beside) | 0.893 | 4,755 | 531 | 4,784 |

At ALL 0.955 on all five, the perfect stop costs 166.2 rows and one depth per dataset 1,482.6.

- **The room is in the stop, not the order.** A perfect stop needs about a tenth of the pick's rows. The pick's
  budget does about as well as one depth per dataset: it tells the questions apart no better than that.
- **The pick's budget is 12 × today's fill.** Today's fill holds step 1's per-dataset sizes: P1f, section 7 says
  "they come from step 1's recipes, which were chosen on s1eval golds". That is the HIGH item. No U2f rule reads
  today's fill.

**What was tried, and how U2f differs.**
- **P1f** (15 October, NONE_MEETS) sized each pool as c × a walk's nucleus size or perplexity, spent down KIND_T's
  fused order. Its finding: "The signals do not tell a musique question that needs 2,000 rows from a hotpotqa or
  2wiki question that needs 100."
- **P1h** (20 October) cut by rank mass. That depends on the order's length alone.
- **U2f keeps sets of nodes, not a depth.** A node is kept when one of KIND_T's own sources holds it inside that
  source's own mass nucleus, or when its cosine lies in a band set by the question's own first-stage cosines.
- **What is new to U2f:**
  - the cosine band, a signal no earlier rule had;
  - the true masses of the steered walks (P1f did not have them);
  - a per-question union in place of one size;
  - W2_S and U2e's orders;
  - no per-dataset size.

The sizes of these sets spent down U2e's orders are a second family (section 4), so the stop and the order can be
told apart.

## 2. The graph, the carves and the orders

The graph, the carves and the orders are U2e's:
- **Graph:** W2_S on the passage datasets, and the KB on metaqa and webqsp.
- **Carves:** s1sel on the five training datasets, with webqsp's s1eval beside.
- **Per question:** B (base and seeds), KIND_T's order, the sources U_0 = KT ∪ N1 ∪ R0 ∪ I, and the 14 orders. These
  are rebuilt by U2e's own steps and checked against U2e's filed arrays (section 6).

## 3. What each node carries

The nodes are those outside B that U_0 holds or that one of KIND_T's sources lists. Each carries three values:

- **r(v):** its place in `kt_cos` (U2e's pick's order, M = 0), from 1. It is ∞ outside that order.
- **e(v), the entry level.** KIND_T's own sources for the question are:
  - **entity question on ROUTE's condition:** Kst and A1st (ROUTEst, Kst backfilled by A1st);
  - **entity question otherwise:** A1st;
  - **passage question:** A1st, A1all, KTst, KTall and R0 (RRF5T's five).

  Kind and ROUTE's condition are P1e's and P1d's. The masses:
  - A1st, A1all and Kst carry P1d's walk masses, and R0 carries P1d's RRF weights. Each sums to 1 over the source's
    listed nodes.
  - **KTst and KTall** carry the steered walk's mass p, read from a copy of P1e's batch kernel that also returns p for
    the nodes it lists. It calls P1e's push and step 4's top-k unchanged. Its orders are checked equal to P1e's, node
    for node. Each is scaled to sum to 1 over its listed nodes, as P1d's walks are.

  For a source s, e_s(v) is the share of s's mass on the nodes before v in s's order. v lies inside s's τ-nucleus (the
  fewest nodes whose mass reaches τ) exactly when e_s(v) < τ. e(v) is the least e_s(v) over the sources that list v,
  and ∞ when none does. A listed node's e(v) is at most 1 − 10⁻¹², so τ = 1 keeps every node a source lists.
- **c(v) = cos(q, v):** U2e's cosine (float32, D5b's expression) for v in U_0, and −2 outside U_0. A non-finite
  cosine is −2 too, and is counted.

**The band's thresholds**, per question: θ(β, δ) = Q_β(c(B)) − δ.
- c(B) is the cosines of B's nodes, by the same expression.
- Q_β is numpy's quantile (linear method), computed in float64 over B's finite cosines: β = 1 is B's highest cosine
  and β = 0 its lowest. B's non-finite cosines are counted. When B has no finite cosine, every band is empty.
- β takes 1, 0.9, 0.75, 0.5, 0.25, 0.1 and 0. δ takes 0, 0.025, 0.05, 0.1, 0.15 and 0.2.
- That gives 42 thresholds, plus "no band" (θ = ∞). The 43 bands run: no band first, then β from 1 down to 0 and,
  within each β, δ from 0 up.

## 4. The rules

Each rule gives one pool per question. None reads a gold, today's fill or the dataset.

- **SET(k0, τ, band):** B ∪ {v : r(v) ≤ k0, or e(v) < τ, or c(v) ≥ θ}.
  - k0 is 0, 25, 50, 100 or 200.
  - τ is 0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.975, 0.99, 0.995, 0.999 or 1.
  - band is one of the 43.
  - That is 3,010 rules. k0 = 0, τ = 0 and no band is B alone.
- **DEPTH(order, x, SET):** B, then the order's first k_q = rint(x × n_q) nodes, where n_q is the number of nodes the
  SET rule keeps outside B.
  - The order is `kt_cos` or `kt` (U2e's).
  - x is 0.5, 1 or 2.
  - rint rounds halves to even.
  - Rows are |B| + min(k_q, the order's length).
  - That is 18,060 rules: each set's size, spent down a fused order.

There are 21,070 candidates in all. The SET rules come first, in the order k0, τ, band. The DEPTH rules follow:
`kt_cos` then `kt`, then x, then the SET rule. Ties go to the earlier.

## 5. The choice (s1sel)

- **ALL and rows** are P1g's: whole questions, and a question with no gold counts 0. The cost is P1f's
  question-weighted mean rows over the five training datasets, with U2c's weights.
- **Levels and targets** are U2e's. Each level runs from 0.950 to 1.000 in steps of 0.005. Dataset d's target is
  min(t, C_d − 0.005), with C_d U2e's filed ceilings. The frontier's point at t is the cheapest candidate meeting every
  target.
- **The incumbent** is U2e's pick, at level 0.955 and a cost of 1,812.6. Both are read from `outputs/u2e/verdict.json`,
  which must be the committed zu2e.py's RAISED record.
- **The pick** is the frontier's point at 0.955, the incumbent's level.

| verdict | when | what follows (section 8) |
| --- | --- | --- |
| FEWER_ROWS | the pick costs at most 0.9 × 1,812.6 (P1h's MIN_CUT, 0.10) | the pick |
| SAME_ROWS | it costs at most 1,812.6 | the pick: the same rows from a rule with no per-dataset size |
| NO_CHANGE | no candidate meets 0.955 within 1,812.6 | U2e's pick stays |

**Filed beside, deciding nothing:**
- the whole frontier, with each dataset's ALL and rows, and the highest level within 1,812.6;
- the cheapest point at every level of each family:
  - SET with the band alone (τ = 0, k0 = 0), the nuclei alone (no band, k0 = 0), both, and with k0 > 0;
  - DEPTH per order and x;
- per dataset, the fewest rows reaching each level (an upper bound only);
- the room: the perfect stop's cost at 0.955 in `kt_cos` and in `kt`, read from golds, and the share of it the pick
  captures: (1,812.6 − pick's cost) / (1,812.6 − the perfect stop's cost in `kt_cos`);
- webqsp's s1eval for the pick and at every level. It never moves the choice.

## 6. Checks

Any failure stops the item, and nothing is written.

**Every question, against U2e's filed arrays** (`outputs/u2e/stats`, the committed zu2e.py's):
- the graph against the one U2e's record names, and the golds' pointers against U2e's;
- the per-question values: |B|, today's fill, |I|, the 14 orders' lengths, |N1|, |R0|, |I outside B|, |N2(M)|,
  |U_M|, the parents taken and the hubs passed over;
- every gold's place in each of the 14 orders, its first N2 parent, and its flags (listed, one link, KIND_T's place,
  in I).

These cover U2c's checks too, since U2e's arrays passed them.

**In the code:**
- The copied kernel's orders equal P1e's KTst and KTall, node for node.
- Every source's masses are finite, non-negative and non-increasing along its order, and sum to 1 within 10⁻⁹
  (an empty source has none).
- U_0 holds no node of B.
- SET at k0, τ = 0 and no band keeps |B| + min(k0, |kt_cos|) rows.
- A gold's r, e and c are its node's.
- A non-finite cosine is counted and filed, and counts as outside every band.

**Self-test** (`python outputs/mp_unified/zu2f.py --selftest`, passed before this commit):
- the entry levels against a per-node loop;
- the rows table against a direct count of each set;
- the thresholds against a direct quantile;
- SET and DEPTH against direct per-question pools;
- the columnwise summary against P1g's `summarize`;
- the frontier, the pick and the verdict on toy tables;
- the copied kernel against P1e's on toy graphs;
- the copied orders function against U2e's;
- the shards and the merge.

**Development runs** (`--limit`) write nothing and log no number.

## 7. Where it runs, and the outputs

On Modal, in question shards, from the committed code:

    modal run outputs/mp_unified/zu2f_modal.py --stage upload
    modal run outputs/mp_unified/zu2f_modal.py --stage probe
    modal run outputs/mp_unified/zu2f_modal.py --stage dev --datasets D [--limit 30]
    modal run --detach outputs/mp_unified/zu2f_modal.py --stage stats --datasets D1,D2,... --shards K
    modal run outputs/mp_unified/zu2f_modal.py --stage fetch
    python outputs/mp_unified/zu2f.py verdict

**The container is U2d's container.**
- It runs the same CPU image and pins.
- It unpacks HEAD's tracked files, then overlays U2c's repository files, U2d's inputs, U2c's and U2e's filed arrays
  and U2b's graphs.
- It builds the same mirror from the package files on the volumes.
- It refuses to run unless U2d's inputs and mirror verify records are VERIFIED for this manifest.

**Volumes and files:**
- U2f writes only a new volume, mpr-u2f: its code archive, and its outputs under `/out/<the first 12 hex of zu2f.py's
  sha256>`.
- The other volumes are mounted read-only.
- The stats are fetched to `outputs/u2f/stats/<D>__<carve>.npz` and `.json`.
- The verdict writes `outputs/u2f/verdict.json` and `verdict.md` on the laptop.
- The arrays stay uncommitted.

## 8. What follows (fixed now)

- **FEWER_ROWS or SAME_ROWS:** U2g is declared in its own file. It builds the pick's pools on the fit and s1eval
  carves of all six datasets, grades s1eval beside, builds the KT input on them, and refits the lean pair and G6 once.
  That is the final refit on the final pools. U2d's results are filed as they come, for U2e's pools.
- **NO_CHANGE:** U2e's pick stays the pools, and U2d's refit is the refit on them. The room between the pick and the
  perfect stop is then the rerankers' to take. That is a learned cut, in its own stage, after the pools are final.

A verdict closes only the form tested (docs/SCREENS.md, section 3b).

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 9. Results

Filed 22 October 2026: `outputs/u2f/verdict.json` and `verdict.md`, and the stats records `outputs/u2f/stats/*.json`
(code 5f49ca2, outputs under `/out/1418879a1d22`). Development numbers.

**The run.** All six datasets ran in 12 question shards each, and all 72 parts merged with no failure. The longest part
took 449 seconds (2wiki). Every check in section 6 passed on every question:
- U2e's per-question and per-gold values were reproduced exactly;
- the copied kernel's orders equal P1e's KTst and KTall;
- every source's masses are distributions;
- the orders are exact;
- SET at k0 with τ = 0 and no band keeps |B| + min(k0, |kt_cos|) rows;
- each gold's r, e and c are its node's.

No cosine was non-finite.

**Verdict: NO_CHANGE.** No rule meets 0.955 within 1,812.6 rows, so U2e's pick stays the pools.
- **The pick.** The cheapest rule at 0.955 is `depth:kt_cos:x1:k0:t0.7:b0.9d0.2`. It keeps B, then as many of kt_cos's
  nodes as SET(k0 = 0, τ = 0.7, band b0.9d0.2) keeps outside B. It costs 2,739.1 rows, 1.51 times U2e's pick.
- **No level fits within 1,812.6.** The cheapest point on the frontier, at 0.950, costs 2,381.3.

| dataset | the pick: ALL | rows | U2e's pick: ALL | rows |
| --- | --- | --- | --- | --- |
| metaqa | 0.956 | 3,436 | 0.962 | 4,384 |
| squad | 0.999 | 1,416 | 0.979 | 50 |
| musique | 0.956 | 4,677 | 0.964 | 5,437 |
| hotpotqa | 0.978 | 2,446 | 0.956 | 561 |
| 2wiki | 0.987 | 2,890 | 0.967 | 840 |
| webqsp (s1eval, beside) | 0.842 | 2,499 | 0.893 | 4,755 |

**Where the rows go.** This is each dataset's share of the cost (the question-weighted mean):

| | metaqa | squad | musique | hotpotqa | 2wiki | cost |
| --- | --- | --- | --- | --- | --- | --- |
| the pick | 748 | 347 | 454 | 451 | 739 | 2,739.1 |
| U2e's pick | 954 | 12 | 528 | 104 | 215 | 1,812.6 |

- The pick saves 280.2 rows on metaqa and musique together, and spends 1,206.5 more on squad, hotpotqa and 2wiki.
- U2e's budget is 12 × |fill|, where |fill| is the size of the pool that today's rule prepares on W2_S, less B. On squad
  |fill| is 0, so U2e's pick keeps B there.
- Section 4 bars U2f's rules from reading today's fill. With the node values alone, no single rule stops early on
  squad, hotpotqa and 2wiki and also late on metaqa and musique.
- Each dataset's own best rule (an upper bound only) reaches 0.955 at 2,089 rows on metaqa, 50 on squad, 4,153 on
  musique, 371 on hotpotqa and 379 on 2wiki. Together those cost 1,035.4. Per-dataset rules are not allowed, so this
  only shows how far apart the datasets' best stops are.

**The families** (the cheapest rule of each at 0.955):

| family | cost |
| --- | --- |
| DEPTH on kt_cos, x = 1 | 2,739.1 |
| DEPTH on kt_cos, x = 0.5 | 2,797.0 |
| DEPTH on kt_cos, x = 2 | 2,938.7 |
| SET with k0 > 0 | 4,757.0 |
| SET with k0 = 0 | 5,015.1 |
| SET, the band alone | 6,920.6 |
| SET, the nuclei alone | 8,526.3 |

- A count spent down kt_cos beats any set of nodes kept by their own values.
- DEPTH on kt reaches 0.950 at best (4,036.0 at x = 1) and never 0.955: KIND_T's whole order holds every gold for only
  0.954 of musique's questions (U2e, section 10).

**The room** (reads golds). A perfect stop in kt_cos costs 166.2 rows at 0.955:

| metaqa | squad | musique | hotpotqa | 2wiki |
| --- | --- | --- | --- | --- |
| 321 | 50 | 505 | 69 | 88 |

That is 9% of U2e's pick. So kt_cos puts the golds early, and the stop is what costs rows. The pick captures −0.56 of
the room, because it costs more than U2e's pick. In kt, musique cannot reach 0.955, so kt's perfect stop has no cost.

**The stats** (s1sel; webqsp's s1eval):

| dataset | B | V | kt_cos | kt | entity questions | route questions | ALL B | ALL kt_cos | ALL listed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 50.0 | 11,678.1 | 9,377.9 | 6,981.0 | 1,497 | 1,421 | 0.019 | 0.996 | 0.995 |
| squad | 50.0 | 8,472.5 | 7,102.4 | 6,933.2 | 0 | 0 | 0.979 | 0.999 | 0.999 |
| musique | 200.0 | 17,473.3 | 12,245.5 | 6,800.0 | 0 | 0 | 0.502 | 0.976 | 0.972 |
| hotpotqa | 50.0 | 17,328.0 | 7,575.3 | 6,950.0 | 0 | 0 | 0.663 | 0.985 | 0.990 |
| 2wiki | 50.0 | 19,135.8 | 8,035.3 | 6,950.0 | 0 | 0 | 0.348 | 0.995 | 0.997 |
| webqsp | 200.5 | 12,210.9 | 11,221.3 | 6,834.6 | 1,453 | 440 | 0.279 | 0.957 | 0.914 |

The targets use U2e's ceilings, which count candidates with the second link. On musique, kt_cos alone ends at 0.976,
below U2e's 0.980, so musique's target from 0.975 up is 0.975.

**What follows (section 8, fixed before any number).** U2e's pick stays the pools, and U2d's refit is the refit on them.
U2d filed NOT_ADOPTED on all four of its decisions (docs/U2D_REFIT_ON_U2E.md, section 11). On these pools and W2_S, the
lean pair loses R@5 on hotpotqa and 2wiki against P1c2's pools on today's graph, at about the same coverage. So U2d's
next form comes first: the 2 × 2, which says whether the graph or the pool rule costs that R@5. The room is then the
rerankers' to take, as a learned cut in its own stage. Section 8's U2g, the final refit on a new pick, is not declared.

This closes only the forms tested: gold-free stops from the node values r, e and c, and set sizes spent down kt_cos or
kt. The next form is the learned cut: a stop learned from the same node values together with |fill|.
