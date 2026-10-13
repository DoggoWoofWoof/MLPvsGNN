# P1b: one restart for all six datasets, its source weights set per question and fit by EM (a diagnostic)

Declared 13 October 2026, committed at 03:07 (3941ebe), before any number of the stage (the first job started at
03:08). P1's second stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), after [P1A_POOL_SCOREBOARD.md](P1A_POOL_SCOREBOARD.md). Nothing here is
adopted, and no model trains or reads.

## 1. The gap P1a measured

On s1eval, our pool construction (A3, step 4e's chosen restart) is below the best LLM-free SOTA order on five datasets.
squad has no graph, so every order there is its RRF order. The order that leads differs by graph:

| dataset | the order that leads | where A3 is below it |
| --- | --- | --- |
| hotpotqa | A1, HippoRAG 2's restart rule | N <= 750: 0.512 against 0.830 at N = 25 |
| 2wiki | A1 | N <= 300: 0.578 against 0.838 at N = 50; A3 leads at N = 500 to 2,000 |
| musique | A1 | N <= 300 (up to 0.027) and N = 4,000 (0.003) |
| metaqa | LINKED, the walk from our linked entities only | every N: 0.237 against 0.546 at N = 25, 0.800 against 0.959 at 2,000 |
| webqsp | A1 | every N: 0.005 to 0.043 |

Two causes are visible in P1a's own columns:
- **The linked names dilute the head on passages.** A3 restarts from the seeds and the linked names at equal weight. On
  hotpotqa, the walk from the linked names alone holds 0.617 at N = 100, against A0's 0.947 from the seeds alone.
- **The lists waste half the restart on metaqa.** KB nodes carry their names only, and the dense and SPLADE lists hold
  every gold of 0.079 and 0.202 of the questions at N = 1,000 (ALL). A3 and A1 put half their restart there. LINKED
  puts none there and leads.

No fixed restart is best on all six. HippoRAG 2's rule leads on the passage graphs and webqsp, and the KB-QA line's
entity walk leads on metaqa. P1b asks whether one restart, whose source weights a question's own label-free features
set, is at or above the best of them on every dataset.

## 2. The construction (MX)

- **Six sources per question**, each a distribution over nodes, all on the dataset's regime union graph with step 4e's
  kernel (its alpha and eps), nothing excluded:

  | zero-step | walked |
  | --- | --- |
  | S0: today's seeds, uniform | S: the walk from S0 (P1a's A0) |
  | L0: our exact-match linked nodes, uniform (step 4e's linking) | L: the walk from L0 (P1a's LINKED) |
  | R0: the RRF-weighted dense and SPLADE top-1,000 lists | R: the walk from R0 |

  A walk's masses are its top 7,000 nodes', scaled to sum to 1. squad has no graph, so there S, L and R are S0, L0
  and R0. A1 is half S0 and half R0 walked together, so it lies inside the mixture's span (up to the walk's push
  threshold).
- **The mixture.** m_q = sum over k of w_k(q) m_k, with w(q) = softmax(W x(q)). x(q) holds fifteen label-free features
  of the question's own lists, links and walks:
  - the number of seeds and of linked nodes (log), and whether there is no linked node;
  - the dense and SPLADE top scores, and their gaps to the tenth;
  - the share of linked nodes inside the lists' top 100, and the share of seeds that are linked;
  - each walk's top mass (log) and its top-10 mass.

  No feature reads a gold or names a dataset.
- **The pool at N** is the mixture's top N, ties by node position.
- **One parameter set for all six**: W, six sources by sixteen weights (the fifteen features and a bias).

## 3. The fit: EM, on the training datasets' fit carves only

- **Data.** The fit carves of the five training datasets, whole: fit on metaqa, squad, hotpotqa and 2wiki, and s1fit
  on musique (4,601 to 5,960 questions each). s1sel and s1eval are never fit. webqsp never fits, so its read is
  zero-shot.
- **The likelihood** is that of each question's golds under its mixture. Every question weighs the same within its
  dataset, every dataset weighs the same, and the golds of a question share its weight. A gold that no source reaches
  carries none.
- **EM.**
  - E-step: each gold's responsibility over the six sources, w_k(q) m_k(g) normalized.
  - M-step: the softmax regression of W on those responsibilities, with an L2 penalty of 1e-3 (L-BFGS, to
    convergence).
  - W starts at zero (equal weights). The loop stops when the penalized log-likelihood gains less than 1e-9 relatively,
    or after 200 iterations.

  Nothing is selected: the features, the penalty and the stopping rule are fixed here.
- **Features are standardized** by their mean and spread over the fit questions, with every dataset weighted equally.

## 4. The call, fixed before any number

For each dataset's s1eval carve:
- **The check first.** P1b's S and L walks must equal P1a's A0 and LINKED curves at every N on both measures (on squad,
  L only). Otherwise the comparison with P1a's records is unsound and the stage stops.
- At every N of P1a's grid (25 to 7,000), and on both measures (ALL and gold recall), MX is compared with the best of
  P1a's LLM-free SOTA orders (A1, A2, LINKED, DENSE, SPLADE, RRF) at that N.
- **AT_OR_ABOVE:** MX is never more than 0.002 below the best (step 4e's tie).
- **BELOW:** anything else. The record names every N and measure where MX is below, the order that leads there, and
  the largest gap.

Reported beside each call, deciding nothing:
- the points where MX is more than 0.002 above the best, and MX against A3;
- MX at the matched sizes of today's pools (I_q) and step 4e's (P4e), beside A3, the best SOTA order and the sets;
- the mean source weights per dataset, on the fit and on each read;
- on metaqa and webqsp, MX at ReaRev's m (500, 2,000) beside the assigned-entity walk (ORACLE, a diagnostic), and the
  smallest N at which MX's ALL reaches ORACLE's at m.

## 5. What follows

- **All six AT_OR_ABOVE:** MX is the pool construction the models read next. P1c, in its own file, rebuilds the
  models' pools as MX's top N at each question's own |I_q|, so the look cost is unchanged, and runs one fit as a
  screen.
- **BELOW anywhere:** MX is not taken as is. The mean weights show whether the gate missed (it weighted a source that
  is not the leading one) or the sources fall short (no weighting of them reaches the leading order). The next change
  is declared in its own file.

Flagged, as every pool so far: the walks are propagation over the graph. The pool is built before either model runs,
and both models of a family read the same pool, so the MLP's own scoring stays free of message passing. Today's pools
are already built by graph hops from the seeds.

## 6. How it runs

- **Code:** `scripts/p1b_restart_mixture.py`. Its self-test passes:
  - the walk with masses orders as step 4e's walk does;
  - a one-hot mixture reproduces its source;
  - EM's log-likelihood never falls, and on a toy problem EM picks the right source per question.
- **Host jobs** use step 4f's pattern (the host mirror through `--host`, numba through `outputs/host_ops/pylib_run.py`),
  with P1a's sizes (5 CPUs, 12 GB; squad 1 CPU, 4 GB):
  1. `gather` on each training dataset's fit carve writes `outputs/p1b/gather_<dataset>.{npz,json}`: features and the
     golds' masses, a few MB.
  2. `fit` writes `outputs/p1b/gate.json` once all five gathers are in.
  3. `apply` on s1sel and s1eval writes `outputs/p1b/mx_<dataset>.json`.
  4. `verdict` writes `outputs/p1b/verdict.{json,md}`. It refuses parts from different scripts, gates or freezes.
- **ETA:**
  - P1a's walks ran at about 19 ms per question and walk (5 threads).
  - gather: three walks on about 6,000 questions, about 10 minutes per dataset with loading.
  - fit: a few minutes.
  - apply: 2wiki's 14,072 questions take about 15 minutes.
  - The call should land about 45 minutes after the start.
- The encoder stays frozen. No model trains or reads, nothing is adopted, and test splits are never read.

## 7. Results (filed 13 October, 03:50)

Records: `outputs/p1b/` (gate.json, gather_*, mx_*, verdict.json and verdict.md). The check passed on every carve: S and
L equal P1a's A0 and LINKED at every point (on squad, L). EM stopped after 195 iterations.

**Correction, 13 October, about 08:45 (found by P2b's check, [P2B_FAMILY_GATE.md](P2B_FAMILY_GATE.md) section 4).**
- **What is wrong.** In `mx_<dataset>.json`, the at-size values of S and L (`at_I_q` and `at_P4e`, ALL and recall) are
  not the walks' values.
- **Why.** The apply stage's list of orders names S and L twice (`DIAG = ("MX", "S", "L", "A1MIX") + SOURCES`), so each
  question's positions were appended twice. The at-size reading then pairs the first half of the questions with
  shifted sizes and divides by twice the count, which comes out at about half the true value (metaqa's s1sel: S
  0.498 filed, 0.8965 true).
- **Shown exactly.** On four carves, P1b's walks were rebuilt with this script's own functions: metaqa's and
  musique's s1sel and s1eval, S and L at both sizes, 16 points.
  - read with the doubled list, they give the filed values;
  - read with each question once, they give P1a's A0 and LINKED at the same sizes.
- **What to read instead.** The true values are P1a's A0 and LINKED at the same sizes, in
  `outputs/p1a/scoreboard_<dataset>.json`.
- **What is unaffected.** The curves (means over the list) were checked above against P1a's at every N. MX, the other
  orders and the call are unaffected too, and so is every number in this file. P1b2 never read those values. The
  records and the script stay as filed.

**The call on s1eval: BELOW on five datasets, AT_OR_ABOVE on squad.**

| dataset | call | where MX is below the best SOTA order | where it is above |
| --- | --- | --- | --- |
| metaqa | BELOW | 28 of 30 points, under LINKED by up to 0.037 (0.923 against 0.959 at N = 2,000) | nowhere |
| squad | AT_OR_ABOVE | nowhere | nowhere |
| musique | BELOW | one point, N = 300 (ALL): 0.732 against A1's 0.737 | every N, by up to 0.030 |
| hotpotqa | BELOW | the head: 0.806 against A1's 0.830 at N = 25; also N = 750 and 3,000 (0.002) | nowhere |
| 2wiki | BELOW | one point, N = 4,000 (ALL): 0.9717 against A1's 0.9739 | N <= 750, by up to 0.015 |
| webqsp | BELOW | N = 25 and 50: 0.256 against A1's 0.301 at N = 25 | N >= 100, by up to 0.060 (N = 500) |

On s1sel (beside): musique is AT_OR_ABOVE; metaqa and hotpotqa are BELOW as on s1eval, and 2wiki only at N = 4,000 and
7,000.

**At each question's own size of today's pools (|I_q|, ALL, s1eval), the same look cost:**

| dataset | today's pools | A3 | MX | best SOTA order |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 0.895 | 0.800 | 0.923 | 0.961 (LINKED) |
| squad | 0.980 | 0.981 | 0.981 | 0.981 (A1) |
| musique | 0.803 | 0.884 | 0.892 | 0.883 (A2) |
| hotpotqa | 0.968 | 0.857 | 0.969 | 0.973 (A1) |
| 2wiki | 0.906 | 0.772 | 0.914 | 0.911 (A1) |
| webqsp | 0.688 | 0.759 | 0.806 | 0.790 (A1) |

MX is above today's pools on all six at the same size (+0.001 to +0.118) and above A3 on every graph dataset.

KB line: at ReaRev's m, MX holds 0.840 against ORACLE's 0.903 on metaqa (m = 500) and 0.801 against 0.816 on webqsp
(m = 2,000). It reaches ORACLE's ALL at N = 1,500 and 3,000.

**Why it is below (section 5's test): the gate missed; the sources do not fall short.**
- **metaqa.** The leading order, LINKED, is the source L itself. The gate gives L 0.85 on average and spreads 0.15
  over the rest, mostly R (0.09). Those sources' top nodes then sit ahead of L's mid-ranked golds.
- **hotpotqa, and webqsp's head.** The leading order, A1, walks half from the seeds and half from the lists. The
  half-and-half mixture of P1b's own S and R (A1MIX, recorded beside) tracks A1 within 0.006 at every N shown. Yet the
  gate gives the list side (R0 and R) 0.029 on hotpotqa and 0.088 on webqsp, where A1 gives it half.
- **The cause is the objective.** EM maximizes the golds' mass under the mixture; the call reads their rank.
  - The walk from the seeds holds its mass near them, so a gold it holds carries more mass than under the list walk,
    even where the list walk ranks it higher.
  - EM therefore moves weight to the peaked source, and it hedges against misses.
  - Neither helps the head.

**What follows (section 5).** MX is not taken as is. The next change is declared in its own file,
[P1B2_RANK_GATE.md](P1B2_RANK_GATE.md):
- it keeps the six sources and adds HippoRAG 2's own walk (A1) as a seventh;
- it fits the gate by EM to the golds' ranks over fixed blends of the sources, rather than to their mass.
