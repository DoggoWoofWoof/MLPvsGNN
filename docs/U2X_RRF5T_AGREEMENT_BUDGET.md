# U2x: a pool budget read from RRF5T's agreement across its five orders

Declared 4 November 2026 (about 01:15 IST), before any U2x number. Module: `outputs/mp_unified/zu2x.py`. It imports
`zu2s` and `zu2w` unchanged, and through them U2m, U2e, P1e and P1f. Records: `outputs/u2x/stats/<D>__s1sel.agree.npz`
(uncommitted) and `.agree.json`, then `outputs/u2x/choice.json` and `choice.md`.

Stage A is pools only: nothing trains and nothing is fit. The stats pass runs U2s's walks again, unchanged, to record
one more count per question. The choice reads golds on s1sel, as every pool stage's choice does
(docs/U2S_WALK_MASS_BUDGET.md, section 6).

## 1. Why

**U2w found no unified rule** (docs/U2W_ONE_POOL_RULE_THREE_DATASETS.md, section 8; filed in 2505e38). Its verdicts
were NOT_UNIFIED and KIND_FREE_TOO_DEAR:
- no universal candidate is within the 1,953.5-row cap at any level;
- at pick O's level, 0.96, the cheapest C and E point costs 4,215.6 rows, 2.41 × pick O's 1,751.9, past the declared
  row price of 2 ×.

**The cause U2w found.** Every universal candidate pays musique's depth on all three datasets.
- The fewest mean rows any universal candidate needs at 0.96 is 4,122.4 on musique, 467.5 on hotpotqa and 441.3 on
  2wiki.
- At 0.96, the cheapest point of each budget form (FIX, ADA, MASS) and of each of the 12 constructions lie within 6% of
  one another.
- No budget form in U2s's set reads how deep a question's golds lie.

**The form tested here** is the first that U2w's rule 4 and U2s's section 10 name: a budget read from the recipe's own
fused score, RRF5T's agreement across its five orders, in place of one walk's mass.
- When the five orders agree, the fused score sits on few nodes outside the base, and a small budget holds most of it.
- When they disagree, as on deep or ambiguous questions, the fused score spreads, and the budget grows.

The question is whether this signal tracks a question's depth better than one walk's mass does. If it does, one
construction could hold pick O's level on all three datasets at a cost within the cap.

## 2. The AGREE budget

**RRF5T** is P1e's order (`scripts/p1e_anchor_kind.py`, `p1e_orders`).
- It is P1d's RRF (`scripts/p1d_free_pool.py`, `rrf`) over five orders: A1st, A1all, KTst, KTall and R0.
- It uses the first stage's constant c and is cut at NMAX = 7,000 nodes.
- A node's fused score is the sum, over the non-empty orders, of 1 / (c + its rank), with ranks from 1.

**AGREE(τ_w, f, k0)** sets k_q = max(k0, round(f × a_q)). a_q is U2s's m_q (docs/U2S_WALK_MASS_BUDGET.md, section 4),
with RRF5T's fused list in place of one walk's:
- take RRF5T's nodes outside B, in RRF5T's order, with their fused scores, and let T be their total;
- a_q is the smallest m whose first m fused scores sum, in float64, to at least τ_w × T;
- a_q = 0 when RRF5T has no node outside B.

**The grid is MASS's**, fixed before any number:
- τ_w: 0.5, 0.8, 0.9, 0.95 or 0.99;
- f: 1, 2, 4 or 8;
- k0: 0, 50 or 300.

That gives 60 budgets, named `agree_t<τ_w>_f<f>_k<k0>`, for example `agree_t0.9_f4_k50`. Nothing is fitted.

a_q is counted outside B, so it depends on the base. Each question records a_q for each of U2s's 3 bases and 5 values
of τ_w, with RRF5T's node count outside each B. A construction's budget reads its own base's counts.

**The pool rule is unchanged.** A pool is B, then the first k_q nodes of one order, then P1h's cut at τ (U2e's rule,
docs/U2E_POOLS_TO_THE_CEILING.md, section 5). Only k_q is new. An AGREE budget reads RRF5T whatever order fills the
pool, as a MASS budget reads its walk.

## 3. The candidates and the choice

**The candidates:**
- U2s's 1,103,760 universal candidates, unchanged;
- 302,400 AGREE candidates: 12 constructions × 3 recipes × 14 orders × 60 budgets × P1h's 10 values of τ, named
  `<construction>/<recipe>/<order>:<budget>@<τ>`;
- the own construction's 91,980 candidates, which include pick O, read as in U2w for the incumbent.

Ties go to the earlier candidate: U2s's candidates first, in U2s's order, then the AGREE candidates in the same layout.

**The choice is U2w's sections 3 and 4, unchanged**, through zu2w's own functions:
- the cost is P1f's question-weighted mean rows over the three datasets;
- t runs from 0.950 to 1.000 in steps of 0.005;
- the target on dataset d is min(t, C_d − 0.005), where C_d is the best ALL any universal candidate reaches on d,
  U2s's or AGREE's;
- the cap is 1,953.5 rows;
- frontier C runs over the kind_t candidates, and E over the rrf5t and both candidates;
- the verdict table, E's verdicts and the carry order are `zu2w.decide`'s, with the row price of 2 × the incumbent's
  cost.

## 4. Checks

Any failure stops the run, and nothing is written.

**The stats pass** runs `zu2s.stats_stage` unchanged, with three in-process patches restored afterwards:
1. `p1e_anchor_kind.p1e_orders` is wrapped. After the original runs, RRF5T's fused scores are rebuilt from its five
   orders, and their order must equal RRF5T's as P1e serves it, question by question.
2. `zu2s.mass_counts` is wrapped. On the first walk of each base, it also counts RRF5T's fused scores outside that
   base's B.
3. `zu2s.OUT` is set to `outputs/u2x`, so nothing under `outputs/u2s` is written.

So every check of U2s's section 7 runs again:
- (c)'s filed arrays;
- the carried frontier's filed points;
- the walk's graph, seeds and types;
- `orders_exact`;
- m_q's bounds.

On top of those:
- a_q keeps m_q's bounds: it is non-decreasing in τ_w, at most RRF5T's count outside B, and 0 exactly when that count
  is 0;
- each question has one count per base;
- the question count equals the stats record's, shard by shard and after the merge.

**The choice** checks, before any number is written:
- U2s's filed stats are the ones U2w read (sha256);
- the rerun's arrays (`outputs/u2x/stats`) equal U2s's filed arrays, array for array. So the patched pass is U2s's
  pass;
- the fused-score counts cover the rerun's questions and bases;
- pick O's filed point and U2w's filed frontier C points are reproduced from the same tables;
- P1f's weights are U2w's.

**The self-test** (`python outputs/mp_unified/zu2x.py --selftest`) passed before this commit. It checks:
- the fused order equals P1d's `rrf` on 200 random cases with overlaps, empty orders and ties;
- the 60 AGREE budgets give exactly MASS's k_q on the same counts;
- the capture records one count per base per question and stops on a short question;
- the AGREE tables equal U2e's `evaluate` at sampled candidates on made-up arrays;
- the names decode, and the verdict is `zu2w.decide`.

**Development runs** (`dev D --limit 5`, on the host for each of the three) must pass every check of the stats pass
before the full run. They write nothing and log only the first question's counts.

## 5. The verdict and what follows (fixed now)

The verdict and what is carried follow U2w's section 4. Beside, deciding nothing:
- every level's point on C and E, past the cap too, with each dataset's ALL and rows, and each point's budget form;
- the frontiers over the AGREE candidates alone;
- the cheapest AGREE point at the incumbent's level;
- the ceilings with and without the AGREE candidates.

**A pick is carried.** Stages B and C follow U2w's sections 5 and 6 unchanged:
- the pools are built by PKO's steps with the construction replaced;
- their coverage is measured on s1eval and B1;
- zg6r and zrckpn are refitted on them, with the same LOSS rule for admission.

The build's entry point and its checks go in a dated section of this file before it runs.

**Nothing is carried.** Pick O stays the pools of record. The next form is U2s's section 10's second: P1h's cut taken
over the walk's mass in place of rank mass, in its own file.

A verdict closes only the form it tests (docs/SCREENS.md, section 3b).

**D1, the timing (default first).** As U2w's D1:
- stage B's build goes into the host queue ahead of the gated fits as soon as a pick is carried;
- if stage C cannot be read before the freeze (00:00 IST, 7 November), pick O stays for the models of record, and
  the unified rule's coverage is reported beside;
- the user decides at the freeze.

**D2, the row price (default first): keep U2w's declared 2 ×.** The reasons:
- raising a declared bound after U2w's 2.41 × was seen would be post hoc;
- pools 2.4 × pick O's would need roughly 175 GB of build space on the host, against about 91 GB free (an estimate
  from PKO's 73 GB);
- refits on them would not finish before the freeze.

The user may instead accept U2w's C point at 0.96 (4,215.6 rows). That is the user's decision, not this file's.

## 6. Running

```
python outputs/mp_unified/zu2x.py --selftest
python outputs/mp_unified/zu2x.py dev D --limit 5 --threads 2       (D = 2wiki, hotpotqa, musique: the gate)
python outputs/mp_unified/zu2x.py stats D --shard I 3 --threads 2   (I = 0, 1, 2)
python outputs/mp_unified/zu2x.py merge D 3
python outputs/mp_unified/zu2x.py choose                             (laptop, after fetching outputs/u2x/stats)
```

**Where it runs.** The pass runs on the host CPU, because the laptop's memory is short for U2s's pass.
- Each shard asks for 2 CPUs and 12 GB. No peak has been measured yet; the development runs give one.
- The inputs are U2s's, as its section 8 lists, with U2s's filed stats for the three datasets.

**ETA (estimate).**
- About 1 to 1.5 hours per shard when the nine run side by side, then the merges.
- Then the choice on the laptop: U2w's took 1,806 s, and U2x adds 27% more candidates.
