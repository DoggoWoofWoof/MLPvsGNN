# U2w: one pool rule on 2wiki, hotpotqa and musique

Declared 4 November 2026 (about 00:30 IST), before any U2w number. Module: `outputs/mp_unified/zu2w.py`. It imports
`zu2s` and, through it, U2m, U2e and P1f, all unchanged. Records: `outputs/u2w/choice.json` and `choice.md`.

Stage A is pools only: nothing trains, nothing is fit, no walk runs and no candidate is new. The choice reads golds on
s1sel, as every pool stage's choice does (docs/U2S_WALK_MASS_BUDGET.md, section 6).

## 1. Why

**The pools of record are one rule over a per-dataset construction.** Pick O, `own/kind_t/kt_cos:ada_f16_k0@0.85`, was
carried by U2s and built by PKO. It applies one recipe, order, budget and cut to each dataset's own construction, and
those constructions differ (docs/U2M_POOLS_THREE_FAMILIES.md, section 1):

| dataset | base | expansion I |
| --- | --- | --- |
| hotpotqa | equal_rrf_budget_50 | STRUCT: h1_c25 |
| 2wiki | equal_rrf_budget_50 | STRUCT: h1_c25 |
| musique | equal_rrf_budget_200 | FULL: h2_c25 |

The ADA budget scales each question's fill, |I_q| − |B_q|, so it inherits that difference.

**U2s's universal search had to meet six datasets.** It found NOT_UNIFIED: C's cheapest 0.950 point costs 3,311.4
rows, 1.7 times the cap, with metaqa and squad among the targets. The claims are read on 2wiki, hotpotqa and musique
only, and no search has been run on those three alone.

**The user's ruling (4 November):** unifying the pools across 2wiki, hotpotqa and musique is the main priority. It
comes before verifying and measuring the claims. This reverses the 31 October ruling that pick O "is the best rule we
have". Pick O stays the pools of record until a unified pick passes stage C.

**The question.** Is there one construction, recipe, order, budget and cut for the three datasets that holds pick O's
coverage level within the cap? The candidates are U2s's own, read from its filed arrays.

## 2. What stage A reads

- **U2s's filed s1sel stats** for 2wiki, hotpotqa and musique (`outputs/u2s/stats/<D>__s1sel.npz` and `.json`), through
  `zu2s.load_u2s`.
  - That call checks each array's sha256, the record's script sha256 (zu2s.py as committed in 9ab7afe), its checks and
    its summary.
  - The run also repeats U2s's verdict checks: the stats are on the walk U2p's filed verdict carries, and their questions
    and golds are U2m's.
- **The candidates:**
  - U2s's 1,103,760 universal ones: 12 constructions × 3 recipes × 14 orders × 219 budgets × 10 values of τ.
  - The own construction's 91,980, which include pick O.
  - Each candidate's ALL and mean rows come from `zu2s.table_arrays`, unchanged.

**Checks before any number is written** (the run stops if one fails):
- Pick O's ALL and rows on each of the three equal its filed point on U2s's frontier O at 0.96
  (outputs/u2s/verdict.json): musique 0.9668 at 5,522.2 rows, hotpotqa 0.9602 at 713.1 and 2wiki 0.9699 at 1,070.5.
- U2s's filed frontier C points, at all 11 levels from 0.950 to 1.000, give their filed ALL and rows on each of the
  three.
- The weights are U2s's filed ones for these three datasets.

## 3. The choice: U2s's section 6, on the three datasets

- **Cost.** P1f's question-weighted mean rows over the three. The weights are P1f's own (musique 7,018, hotpotqa 13,335,
  2wiki 18,504), through `p1f_pool_size.cost` with those three keys.
- **Levels.** t runs from 0.950 to 1.000 in steps of 0.005. The target on dataset d is min(t, C_d − 0.005), where C_d
  is the best ALL any universal candidate reaches on d.
- **The cap.** 1,953.5 rows, U2s's, unchanged.
- **Frontiers.** C, over the universal kind_t candidates, and E, over the universal rrf5t and both candidates. Each
  frontier's pick is its point at the highest level within the cap. O, the own construction's kind_t frontier, is
  reported beside. It is per dataset, so it never counts as unified.
- **The incumbent** is pick O on the three, read from the same tables. Its level L_I is the highest level whose targets
  it meets, and its cost is its P1f cost on the three. From U2s's filed per-dataset values, L_I should be 0.96, with
  hotpotqa binding at 0.9602; the run computes both.
- **Ties** go to the earlier candidate in U2s's order.

## 4. The verdict and what is carried (fixed now)

| verdict | when |
| --- | --- |
| UNIFIED_RAISED | C's pick is within the cap, at a level above L_I |
| UNIFIED_HOLDS | C's pick is within the cap, at L_I |
| UNIFIED_AT_A_PRICE | C's pick is within the cap, at L_I − 0.005 (one step, U2s's tolerance) |
| UNIFIED_AT_A_ROW_PRICE | none of the above; C's cheapest point at L_I, past the cap, costs at most 2 × the incumbent's cost |
| NOT_UNIFIED | otherwise |

E's verdict is U2s's (e), with the same row-price row added: KIND_FREE_NO_PRICE, KIND_FREE_AT_A_PRICE,
KIND_FREE_AT_A_ROW_PRICE (E's point at L_I is within 2 × the incumbent's cost and costs no more than C's) or
KIND_FREE_TOO_DEAR.

**The row price** is declared because the user made unification the main priority. It is bounded at twice pick O's cost,
so a unified rule cannot buy its level with pools of any size. The price in rows is reported with the pick.

**What is carried to stage B**, the first that applies:
1. E's pick, when (e) is KIND_FREE_NO_PRICE or KIND_FREE_AT_A_PRICE;
2. C's pick, when (c) is UNIFIED_RAISED, UNIFIED_HOLDS or UNIFIED_AT_A_PRICE;
3. the cheaper of C's and E's points at L_I, ties to C, when either is AT_A_ROW_PRICE;
4. nothing: pick O stays, and the next forms are U2s's section 10: a budget read from RRF5T's agreement across its
   five orders, and P1h's cut taken over the walk's mass.

A verdict closes only the form it tests (docs/SCREENS.md, section 3b).

**Beside, deciding nothing:**
- every level's point on C, E and O, past the cap too, with each dataset's ALL and rows;
- the cheapest point of each budget form (FIX, ADA, MASS) and of each construction at L_I;
- per dataset, the fewest mean rows any universal candidate needs for each level. This is an upper bound only;
- U2s's six-dataset pick C at 0.950, read on the three.

**One caution, as in U2s.** The frontier chooses among about 1.1 million correlated candidates on about 1,500
questions per dataset. The carried pick is judged on s1eval and on B1, which the choice never reads.

## 5. Stage B: the pools and their coverage (declared now; the build code in a dated section first)

When stage A carries a pick, its pools are built by PKO's steps with the construction replaced. That needs a new
build entry point, because `zpko.py` is fixed to pick O's own construction. Its code and checks go in a dated section
of this file before it runs. They include the carried pick's rows and held golds reproducing U2s's filed arrays on
s1sel, question for question.

**The carves** are the ones the WQ fits read for the three datasets (fit, s1sel and s1eval), and B1's three settings
(HippoRAG 2's 1,000 questions and corpora).

**The coverage**, measured with no model on s1eval and on B1, for the carried pick and for pick O on the same questions:
- ALL (every gold in the pool) and gold recall (the share of golds in the pool);
- mean, median and 90th-percentile rows;
- rows as a share of the corpus;
- on B1, beside HippoRAG 2's candidate set, which is every passage, because its walk scores the whole graph.

Stage B decides nothing about models: it is the coverage and localization record.

## 6. Stage C: the models on the unified pools (declared now)

- **The fits.** zg6r and zrckpn are refitted on the carried pick's pools under WQ-musique and WQ-hotpotqa, with ML1's
  recipes and seed 0.
- **The reads** are C1R's twelve on s1eval, plus B1's three settings.
- **Each read's call** is the paired R@5 difference, unified − pick O, for the same model, by question, with a 2,000
  resample 95% interval:
  - LOSS when the point is below −0.0075 and the interval's upper end is below 0;
  - GAIN when the point is above +0.0075 and the lower end is above 0;
  - WITHIN otherwise.
- **Admission.** The unified pools become the pools of record when neither model has a LOSS among the six in-domain
  reads of docs/WWW_SCOPE_RULE.md, section 2. A unified rule needs to cost nothing measurable, not to gain.
  - Otherwise pick O stays for the models, and the unified pools are reported beside with their coverage and their
    LOSS reads.
  - Zero-shot and B1 reads are reported and decide nothing.

**D1, the timing (default first).**
- The default is that stage B's build goes into the host queue ahead of the gated s1 and s2 fits, as soon as stage A
  carries a pick, because the user made this the main priority.
- If stage C cannot be read before the freeze (00:00 IST, 7 November), pick O stays for the models of record, and the
  unified rule's coverage is reported beside. The user decides at the freeze.

## 7. Running stage A

```
python outputs/mp_unified/zu2w.py --selftest
python outputs/mp_unified/zu2w.py choose          (laptop, minutes)
```

The self-test runs on made-up tables:
- the cost over three keys equals the hand-computed weighted mean;
- the frontier and the pick within the cap match a hand-built case;
- the verdict table gives each of its five outcomes, and E's four, on inputs built for them;
- the carry order picks E, then C, then the row-priced point, then nothing.

## 8. Results (filed 4 November 2026, about 01:00 IST)

The run, `zu2w.py choose` on the laptop, took 1,806 s. Every check passed: pick O's filed point and U2s's 11 filed
frontier C points were reproduced on the three datasets; the walk, questions and weights are U2s's. Records:
`outputs/u2w/choice.json` and `choice.md`.

**Verdicts: NOT_UNIFIED and KIND_FREE_TOO_DEAR. Nothing is carried.** ALL on s1sel, with mean rows in brackets:

| pools | level | cost (rows) | musique | hotpotqa | 2wiki |
| --- | ---: | ---: | --- | --- | --- |
| pick O, the incumbent (per-dataset construction) | 0.96 | 1,751.9 | 0.9668 (5,522.2) | 0.9602 (713.1) | 0.9699 (1,070.5) |
| C's cheapest at 0.96: `rrf50+FULL:h2_c25/kind_t/kt_cos+n2_10:ada_f6_k0@0.85` | 0.96 | 4,215.6 | 0.9602 (4,193.9) | 0.9834 (4,025.2) | 0.9913 (4,361.1) |
| E's cheapest at 0.96: the same point with the rrf5t recipe | 0.96 | 4,215.6 | 0.9602 (4,193.9) | 0.9834 (4,025.2) | 0.9913 (4,361.1) |

- The ceilings C_d are musique 0.9811, hotpotqa 0.9881 and 2wiki 0.996. The incumbent's level L_I is 0.96, with
  hotpotqa binding at 0.9602.
- No universal point is within the 1,953.5-row cap at any level. C's cheapest 0.950 point costs 3,528.4 rows (1.81 ×
  the cap) and E's 3,522.1.
- At L_I, C's and E's cheapest points cost 4,215.6 rows, 2.41 × the incumbent's. That is past the declared row price of
  2 ×, so neither AT_A_ROW_PRICE row applies.

**Where the price comes from (beside, deciding nothing).** Every universal candidate pays musique's depth on all three
datasets.
- The fewest mean rows any universal candidate needs at 0.96 is 4,122.4 on musique, but 467.5 on hotpotqa and 441.3 on
  2wiki. These are upper bounds per dataset; no one candidate reaches all three.
- C's pick at 0.96 holds hotpotqa and 2wiki far above their targets (0.9834 and 0.9913) with about 4,000 rows each,
  against pick O's 713.1 and 1,070.5. On musique it needs fewer rows than pick O (4,193.9 against 5,522.2).
- At 0.96 the cheapest point of each budget form (FIX 4,252.5, ADA 4,215.6, MASS 4,230.7) and of each of the 12
  constructions (4,215.6 to 4,438.9) lie within 6% of one another. No budget form in U2s's set reads how deep a
  question's golds lie.

**What follows (section 4, rule 4, and U2s's section 10).** Pick O stays the pools of record. The next forms of the
universal construction, each in its own declared file, are:
1. a budget read from the recipe's own fused score, RRF5T's agreement across its five orders, in place of one walk's
   mass;
2. P1h's cut taken over the walk's mass in place of rank mass.

Both target the cause above: each question's depth, which the current budget forms do not see. Stages B and C wait for
a carried pick. Pick O's coverage on s1eval and B1 (section 5's measures, for pick O alone) is the coverage and
localization record either way, so it is measured now from PKO's and B1P's filed pools, in its own file.

Each verdict closes only the form it tests (docs/SCREENS.md, section 3b).
