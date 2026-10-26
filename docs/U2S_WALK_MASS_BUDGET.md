# U2s: one construction for all six, with each question's budget read from its own walk mass

Declared 27 October 2026, before any U2s code or number. This is the next form of U2m (c), named when (c) was filed
(docs/U2M_POOLS_THREE_FAMILIES.md, section 12). Code: `outputs/mp_unified/zu2s.py`, which imports U2m, U2q, U2p,
U2m (a) and U2e unchanged. Pools only: nothing trains, nothing is fit, and no construction, recipe, order, budget or
cut reads a gold. The choice reads golds on s1sel, as every pool stage's choice does.

## 1. Why

**(c) found no universal construction within the cap** (NOT_UNIFIED, 73ada1a). Under U2e's budgets no one base and
expansion met 0.950 on every training dataset within 1,953.5 rows. The cheapest universal 0.950 point cost 3,641 rows.
- **Where the rows went.** U2e's ADA budgets scale today's fill, |I_q| − |B_q|, so once a construction has an
  expansion, squad's questions get thousands of rows. A FIX budget gives every dataset the same rows.
- **Per dataset**, the fewest mean rows any universal candidate needs for 0.950 were: squad 50, hotpotqa 323.4,
  2wiki 326.1, metaqa 1,817.5 and musique 3,858.4. No single budget scale fits all five.

**The pick is still one rule over per-dataset inputs.** Today's pick (U2p's, section 2) takes three things from the
13 September construction, which is set per dataset:
- the base B;
- the expansion I;
- the budget's scale, today's fill, which is the size of that dataset's I.

**The walk is already one rule on every dataset.** A question whose walk keeps its mass on a few nodes needs few rows.
One whose walk spreads needs more. That reading is per question, comes from the same rule on every dataset, and needs
no per-dataset input. So it can stand where today's fill stands. (c)'s filing named this form: "one construction whose
budget is read from each question's own walk mass, not from today's fill".

The user's ruling, quoted in docs/U2R_NAME_RULE.md:

> yeah i mean unify it properly so we have a strong claim (25 October)

## 2. The walk, the graph and the carves

**The walk is the carried one, as U2q reads it** (`zu2q.carried_walk`, unchanged):
- on musique, squad, hotpotqa and 2wiki: U2p's pair weight at β = 0.5 (U2p's verdict LINKWEIGHT_RAISED, 1e2fd43);
- on metaqa and webqsp: U2m (a)'s RWe at β = 0.5.

Each question walks the five walks kind_t reads (A1st, A1all, Kst, KTst and KTall) with (a)'s kernel
(`zu2m_a.rwalk`). The kernel returns each walk's nodes by mass, descending, and their raw masses, at most P1d's NMAX
(7,000) nodes each.

**The incumbent** is U2p's pick: `own/kind_t/kt_cos:ada_f16_k300@0.85|G0|RWe@0.5|LW@0.5`, at level 0.96 and a cost of
1,953.4 rows. Its lowest ALL is hotpotqa's, 0.9602.

**The graph, the seeds and the carves** are U2m's:
- W2_S stands in for structural on the passage graphs;
- the seeds are G0's;
- the carve is s1sel on the five training datasets, with webqsp's s1eval beside.

**E1** (U2m section 19) runs on U2e's walk at β = 0, and its verdict is filed in U2m section 20.
- If it files E1_RAISED or E1_CHEAPER, E1 on the carried walk is its own file, as section 19 says. When U2s carries
  a universal pick, that file recomputes U2s's verdict on E1's walk.
- U2s itself runs on the walk filed today.

## 3. The constructions, recipes and orders

All of these are U2m's, unchanged.
- **The 12 universal constructions** (section 4 of U2M): the bases equal_rrf_budget_50, equal_rrf_budget_200 and
  dense_top200, each with the expansions none, FULL:h1_c25, FULL:h2_c25 and FULL:h3_c25_v2000. They are built by the
  same calls: `m3b_compile.base_rows`, `m3b_pools.expand_hops` over FULL, and `m3b_compile.build_pool`. **The own
  construction** is beside, as in (c).
- **The three recipes** (section 5 of U2M): kind_t, rrf5t and both, here on the carried walk.
- **U2e's 14 orders** for each construction and recipe, from `zu2e.question_orders`.

## 4. The budgets

A pool is B, then the first k_q nodes of one order (the whole order when it is shorter), then P1h's cut at τ. This is
U2e's rule (docs/U2E_POOLS_TO_THE_CEILING.md, section 5), unchanged. k_q comes from one of three forms:

- **FIX(k)**: U2e's 19 budgets.
- **ADA(f, k0)**: U2e's 80 budgets, max(k0, round(f × today's fill)). Today's fill comes from the construction's own
  I, as in (c).
- **MASS(w, τ_w, f, k0)**, new: max(k0, round(f × m_q)). m_q is the number of nodes walk w needs, outside B, to hold
  τ_w of its mass outside B. It is computed as follows:
  - take the walk's nodes outside B, in the kernel's order, with their raw masses, and let T be their total;
  - m_q is the smallest m whose first m masses sum to at least τ_w × T;
  - m_q = 0 when the walk has no node outside B.

  The grid:
  - w: A1all or KTall. Both walk all three families. A1all restarts as HippoRAG 2 does; KTall restarts uniformly over
    L* and is steered by the first stage.
  - τ_w: 0.5, 0.8, 0.9, 0.95 or 0.99.
  - f: 1, 2, 4 or 8.
  - k0: 0, 50 or 300.

  That is 120 budgets, named `mass_<w>_t<τ_w>_f<f>_k<k0>`, for example `mass_KTall_t0.9_f4_k50`.

  m_q is counted outside B, so it depends on the base. Each question records m_q for each of the 3 bases, 2 walks and
  5 values of τ_w. The sum over the first m masses is taken in float64, in the kernel's order.

So there are **219 budgets** in all: U2e's 99, then the 120 MASS budgets, in the order written above.

## 5. The candidates

Each construction and recipe has 14 orders × 219 budgets × P1h's 10 values of τ = 30,660 candidates.
- **Universal:** the 12 constructions × the 3 recipes, 1,103,760 candidates, named
  `<construction>/<recipe>/<order>:<budget>@<τ>`.
- **Own:** the own construction × the 3 recipes, 91,980 candidates.

Ties go to the earlier candidate, in this order:
1. the construction, then the recipe, each in U2m's order;
2. within a recipe, U2e's order (the order, then the budget, then τ from 1 down), with the MASS budgets after U2e's
   99.

## 6. The choice

U2m section 7's rule, unchanged:
- **cost:** P1f's question-weighted mean rows, with P1f's weights (metaqa 15,745, squad 17,729, musique 7,018,
  hotpotqa 13,335, 2wiki 18,504);
- **levels:** t from 0.950 to 1.000 in steps of 0.005;
- **targets:** min(t, C_d − 0.005) on each training dataset d, where C_d is the best ALL any universal candidate
  reaches on d;
- **the cap:** 1,953.5 rows.

Each frontier's pick is its point at the highest level within the cap. There are three frontiers:
- **C**, over the universal kind_t candidates: pick C, at level L_C;
- **E**, over the universal rrf5t and both candidates: pick E, at level L_E;
- **O**, over the own construction's kind_t candidates, all 219 budgets: pick O, at level L_O.

O asks a separate question: does the own construction do better on the carried walk with the full budget set? U2p
recomputed only the carried frontier's 8 points on this walk, so O also contains U2e's budgets at this walk's values.
O holds the incumbent, so L_O ≥ 0.96.

| verdict (c) | when |
| --- | --- |
| UNIFIED_RAISED | L_C > 0.96 |
| UNIFIED_HOLDS | L_C = 0.96 |
| UNIFIED_AT_A_PRICE | L_C = 0.955, one step below the incumbent (P1g's tolerance) |
| NOT_UNIFIED | otherwise: C has no pick, or L_C < 0.955 |

| verdict (e) | when |
| --- | --- |
| KIND_FREE_NO_PRICE | E has a pick at L_E ≥ 0.955, and C has none or L_E ≥ L_C |
| KIND_FREE_AT_A_PRICE | L_E = L_C − 0.005 and L_E ≥ 0.955 |
| KIND_FREE_TOO_DEAR | otherwise |

| verdict (o) | when |
| --- | --- |
| OWN_RAISED | L_O > 0.96 |
| OWN_CHEAPER | L_O = 0.96 and pick O costs less than 1,953.4 |
| NO_GAIN | otherwise |

(o) also names pick O's budget form (FIX, ADA or MASS).

**What is carried forward** to (b)'s re-read, U2q's recompute and U2n, taking the first that applies:
1. pick E, when (e) is KIND_FREE_NO_PRICE or KIND_FREE_AT_A_PRICE;
2. pick C, when (c) is not NOT_UNIFIED;
3. pick O, when (o) is OWN_RAISED or OWN_CHEAPER;
4. otherwise U2p's pick stays.

The first two put one rule on all six. A universal pick is carried at a price of at most one step (0.005), as (c)
declared, because the user asked for one rule.

One caution, as in (c): the frontier chooses among about 1.2 million correlated candidates on s1sel. The carried
pools are judged where U2n grades them, on s1eval and zero-shot.

## 7. Checks

Any failure stops the dataset, and nothing is written.

**Every question, against (c)'s filed arrays** (outputs/u2m/stats, on the same graph and carve). These do not depend
on the walk:
- for every construction: |B|, |I| and |I outside B|;
- for every base: |N1|, |R0|, N2's counts at M = 10, 30 and 100, the parents taken and the hubs passed over;
- every gold's in_I and one_link.

**Every question, against the carried frontier's filed arrays** (U2p's at its β on the passage graphs, (a)'s on the
KBs; their pick column is `zu2q.filed_rows`'s). On the own construction with kind_t, each of the carried frontier's 8
points must give exactly the filed rows and held golds. The incumbent is one of them (kt_cos, ADA(16, 300), τ = 0.85,
G = 0).

**The walk:**
- the graph is U2m's, the seeds are G0's, and the EM and score types are the ones (a)'s and U2p's stats read. These
  are U2q's checks.
- KIND_T is ROUTEst or RRF5T by the kind;
- every order passes U2e's `orders_exact`.

**m_q:** for every base and walk:
- m_q is non-decreasing in τ_w;
- m_q is at most the walk's node count outside B;
- m_q is 0 exactly when that count is 0.

**In the verdict:** on U2s's arrays, the own construction with kind_t gives U2p's filed ALL, rows and cost at each
of the 8 points, on all six datasets. The incumbent's point gives level 0.96 at 1,953.4.

**The self-test**, `python outputs/mp_unified/zu2s.py --selftest`, passes before the code is committed. It covers:
- m_q against a brute count on toy walks (ties, B nodes inside the list, an empty walk, a walk all inside B);
- the 219 budgets, their names and the column layout;
- the vectorized table against U2e's `evaluate` and `summ` on toy arrays, for FIX, ADA and MASS;
- the three frontiers, the verdicts and what is carried, on toy tables;
- the shards and the merge.

**Development runs** (`--limit 30`, on metaqa and musique) pass every check above before the code commit. They verify
every served node row against DenseNodes, write nothing and log no number.

## 8. Where it runs, and the outputs

- **The laptop** runs every dataset, one at a time, with 2 BLAS threads: `python outputs/mp_unified/zu2s.py stats D
  [--shard I/K]`.
  - hotpotqa and 2wiki run in question shards, as E1's resumed run does, so a shutdown loses at most one shard.
  - If a shard does not fit in the laptop's memory, it runs on Modal with U2m's image and volumes. The cost is priced
    from one timed shard before the rest launch.
- **Joining and the verdict:** `zu2s.py merge D --parts K` joins the shards, and `zu2s.py verdict` runs on the
  laptop.
- **The outputs:**
  - stats: `outputs/u2s/stats/<D>__<carve>.npz` (uncommitted) and `.json` (committed);
  - verdict: `outputs/u2s/verdict.json` and `.md`.
- **ETA (estimate).** (c)'s stats took 5,223 to 8,604 s per dataset on the laptop, 12.3 hours in all. U2s adds the
  carried walk at one β, about a quarter of U2p's 2,240 to 7,077 s per passage dataset. That is about 1.6 to 2.9 hours
  per dataset, 14 to 16 hours in all, then the verdict.

## 9. Reported beside, deciding nothing

- frontiers C, E and O at every level, past the cap too, with each dataset's ALL and rows;
- the cheapest point at every level for each budget form (FIX, ADA, MASS), each construction and each recipe;
- per dataset, the fewest mean rows reaching each level. This is an upper bound only;
- each dataset's union ceiling;
- the misses of each carried candidate by U2e's causes (TAU_CUT, BUDGET, UNREACHED);
- per dataset, walk and τ_w: m_q's median and 90th percentile, and its Spearman correlation with today's fill on the
  own construction. This shows how far the walk's concentration tracks today's per-dataset fill;
- webqsp's s1eval for every pick and every level's point. It never moves a choice.

## 10. What follows (fixed now)

**A universal pick is carried (step 1 or 2 of section 6).** The pools become one rule on all six: one base, one
expansion, one walk recipe and one budget form.
- U2n builds on these pools.
- (b)'s parked restore is read again on them, in a dated section of U2M. hotpotqa's binding may change, which is the
  reason (b)'s park gives.
- U2q's arrays are recomputed on the new rows for U2n, in their own file, as U2q's section 3 provides.

**Pick O is carried (step 3).** The own construction is kept with pick O's budget. The unified form's next form is
then the one below.

**Nothing is carried (step 4).** U2p's pick stays. The next forms of (c):
- a budget read from the recipe's own fused score: RRF5T's agreement across its five orders, in place of one walk's
  mass;
- P1h's cut taken over the walk's mass in place of rank mass.

Each verdict closes only the form it tests (docs/SCREENS.md, section 3b).
