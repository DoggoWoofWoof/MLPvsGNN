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

Run 9 October 2026. squad and hotpotqa ran on the laptop; the other carves ran on the host, one job per dataset
(outputs/step4f/run_one.py, through outputs/host_ops/pylib_run.py for numba). hotpotqa also ran on the host as the
device control: both of its carves' curves are IDENTICAL to the laptop's. Records: outputs/step4f/budget.json,
verdict.json, verdict.md.

**Verdict: `NO_SHRINK`.** No order holds step 4e's all-gold rate (ALL) within 0.005 at half of step 4e's pool size on
metaqa, musique or webqsp.

| dataset | step 4e ALL / pool | target | N* (order) | pool at N* | ratio | s1eval ALL at N* / step 4e | est. looks GiB (4e) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 0.9599 / 7,093 | 0.9549 | none (-) | - | - | - / 0.9526 | - (23.0) |
| squad | 0.992 / 150 | 0.987 | 100 (O1) | 100.0 | 0.666 | 0.9885 / 0.9921 | 0.4 (0.6) |
| musique | 0.9498 / 6,921 | 0.9448 | 7000 (O1) | 7000.0 | 1.011 | 0.9404 / 0.9367 | 16.1 (15.9) |
| hotpotqa | 0.9775 / 188 | 0.9725 | 500 (O1) | 500.0 | 2.654 | 0.9816 / 0.9787 | 1.3 (0.5) |
| 2wiki | 0.9465 / 220 | 0.9415 | 750 (O1) | 750.0 | 3.415 | 0.9573 / 0.945 | 3.1 (0.9) |
| webqsp | 0.9088 / 7,213 | 0.9038 | none (-) | - | - | - / 0.9088 | - (3.5) |

ALL on the choice carve at N rows, O1 / O2 (U = step 4e's union of the first-stage lists, mean rows per question):

| dataset (carve) | U | N=100 | N=500 | N=1000 | N=3000 | N=7000 | machine |
| --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa (s1sel) | 3,150 | 0.522 / 0.173 | 0.678 / 0.569 | 0.755 / 0.655 | 0.836 / 0.835 | 0.900 / 0.900 | host |
| squad (s1sel) | 50 | 0.989 / 0.989 | 0.995 / 0.995 | 0.997 / 0.997 | 0.998 / 0.998 | 0.998 / 0.998 | laptop |
| musique (s1sel) | 3,134 | 0.514 / 0.497 | 0.737 / 0.726 | 0.810 / 0.799 | 0.893 / 0.893 | 0.945 / 0.946 | host |
| hotpotqa (s1sel) | 100 | 0.867 / 0.771 | 0.981 / 0.942 | 0.991 / 0.979 | 0.997 / 0.997 | 0.999 / 0.999 | laptop |
| 2wiki (s1sel) | 109 | 0.743 / 0.433 | 0.940 / 0.833 | 0.961 / 0.922 | 0.969 / 0.969 | 0.977 / 0.977 | host |
| webqsp (s1eval) | 3,374 | 0.422 / 0.244 | 0.590 / 0.520 | 0.667 / 0.598 | 0.810 / 0.822 | 0.852 / 0.866 | host |

What this says:
- **Big pools.** On metaqa and webqsp, a ranked cut of the A3 walk never reaches step 4e's ALL, even at 7,000 rows:
  4e's pool, built as U plus k x B walk nodes, holds golds that sit low in any single ranking. On musique the walk order
  needs all 7,000 rows to reach it. So a single ordering cannot make these pools smaller: the coverage lives in the
  union itself, not near the top of any one list.
- **Small pools.** On hotpotqa and 2wiki, step 4e's pools are already small (188 and 220 rows), and a single ranked
  order needs 500 and 750 rows for the same coverage. Step 4e's union is the cheaper object there too.
- **squad** is the one dataset where a cut shrinks the pool (100 rows against 150); squad's pools are tiny either way.
- **Fusing the first-stage lists into the walk order (O2)** is below O1 up to 1,000 rows on every dataset, and helps
  only on the largest cuts (webqsp +0.012 at 3,000 and +0.014 at 7,000; musique +0.001 at 7,000), still short of
  step 4e's ALL.

What follows: step 4e's pools stand as the P1 candidate. The disk cost of their looks is the price of their coverage,
and is managed by cleanup, not by cutting the pools. A smaller pool would need a learned pruner (a first-pass scorer
over the union), which is a model stage, not a pool budget; it is not declared here.
