# Step 4f: smaller pools at step 4e's coverage (a diagnostic)

Declared 9 October 2026, before any number of it. A diagnostic: nothing here is adopted, and no look, cache or fit is
built.

## 1. Why

Step 4e's chosen pools (A3, k = 2) reach "every gold in the pool" (ALL) of 0.909 to 0.992 on the six s1eval carves.
On metaqa, musique and webqsp, though, a pool holds about 7,000 rows. Those three datasets hold about 90% of step 4e's
look bytes (46 GiB of looks on the host, with the caches still to come), and looks grow with the pool.

The size was never chosen for itself:
- U_q (about 3,150 rows there) inherits step 1's frozen pool size: B_q = |I_q| minus the base.
- Step 4e then adds k × B_q walk nodes, and its rule picks k on coverage alone. A row costs nothing under that rule.

A pool is a union of fixed parts, with no single order to cut. This step asks how small a pool can be if every
candidate is ranked in one order and the order is cut at N rows.

## 2. The orders

Each order is a fixed rule over the existing graph and the existing dense and SPLADE top-1000 lists. As in step 4e,
there is no transformer, no LLM, no new graph, no new embedding and no finer text unit. The encoder stays frozen.

| order | rule |
| --- | --- |
| O1 walk | step 4e's A3 walk (the same restart, kernel, alpha 0.5 and eps), with nothing excluded: seeds, linked nodes and list rows rank by their own walk mass |
| O2 fused | reciprocal-rank fusion of three lists, O1, dense and SPLADE, by 1/(c + rank) summed (c is the config's equal-RRF constant); ties by node id |

- squad has no graph, so both orders are the RRF order of its two lists.
- A pool of budget N is the order's first N rows. If an order is shorter than N, its pool is the whole order (the
  mean size is filed).
- The grid of budgets is N ∈ {25, 50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000, 5000, 7000}.
- Coverage is ALL and recall at each N. The walk's order is computed up to 7,000 rows.

## 3. The rule, fixed before any number

The choice carves are step 4e's: the five s1sel carves, and webqsp's s1eval.

1. **The target.** For each dataset, the target is step 4e's chosen pools' ALL on its choice carve minus 0.005 (step
   4e's K_TOL). That ALL is read from outputs/step4e/coverage.json (A3, k = 2).
2. **The budget.** N\* is the smallest grid N at which O1 or O2 reaches the target. If both reach it at the same N, O1
   is taken, since it is the simpler rule. A dataset where neither order reaches the target by N = 7,000 has no N\*,
   and its best ALL is filed.
3. **The verdict.** The size ratio is N\* over step 4e's chosen mean pool size on the choice carve. The verdict is:

   | verdict | when |
   | --- | --- |
   | SHRINKS | every dataset has an N\*, and the ratio is at most 0.5 on metaqa, musique and webqsp (the datasets that hold the bytes) |
   | PARTIAL | at least one of those three has a ratio at most 0.5, but SHRINKS does not hold |
   | NO_SHRINK | otherwise |

4. **The s1eval check.** The s1eval ALL at each dataset's N\* (with the chosen order) is filed beside step 4e's s1eval
   ALL for all six datasets. It is read only, never used to choose.
5. **What follows.** Under SHRINKS or PARTIAL, budgeted pools are a rival to step 4e's pools. Their looks, caches,
   identity gate and zrm screen are a later stage, declared in its own file. The estimated look bytes at N\* (step 4e's
   measured look bytes per row times N\* rows) are filed with the verdict.

## 4. Running it

The step runs on the laptop CPU, on the same prepared inputs as step 4e's coverage stage:

- `python scripts/step4f_pool_budget.py coverage [--threads 5]` writes outputs/step4f/budget.json.
- `python scripts/step4f_pool_budget.py verdict` writes outputs/step4f/verdict.json and verdict.md.
- `python scripts/step4f_pool_budget.py --selftest` runs the self-test.

## Results

(Filed after the run.)
