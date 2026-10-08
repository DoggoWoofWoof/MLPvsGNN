# Step 4d: union pools, the frozen pool and the walk pool together

Declared 8 October 2026 at about 21:10, before any number of its pools stage, looks, fits or reads. The user (21:00):
"do we need to improve the quality of the graph pool we provide to both models? any better than what currently exists
we do not want oracle failures to dissuade our coverage".

## 1. Question

Every model reads only the rows its question's pool holds. A gold outside the pool is a miss for every model, and no
training can recover it. Today's pools are the frozen construction I_q (step 1's caches; every screen since). Step 4b's
walk P_F, run on each dataset's own frozen graph, puts more golds in the pool on musique and webqsp. Step 4c replaced I_q
by P_F at the same size. On metaqa that lost golds the frozen expansion's three hops held, so metaqa's golds in the pool
fell from 0.957 to 0.947. The step-4c screen, scr-pools, was MIXED.

**Step 4d keeps both.** The union pool U_q is I_q plus the walk's first B_q nodes that I_q does not hold (B_q is step
4c's budget, |I_q| minus base and seeds). So:
- U_q holds every node of I_q, so no question loses a gold its pool holds today;
- U_q holds every gold P_F's pool holds.

Does the union lift what both models can reach, and do they reach it?

## 2. What is already known (before this file; step 4b's filed arrays)

At 21:00, before this declaration, the union's coverage was computed from step 4b's filed eval arrays
(`outputs/step4b/eval/<ds>.npz`: each gold's place in I_q, in base and seeds, and its rank in the walk). These are the
s1eval questions:

| dataset | every gold in the pool: I | P_F | U | golds in the pool: I | P_F | U | pool size I | U at most |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| metaqa | 0.895 | 0.898 | 0.941 | 0.957 | 0.947 | 0.977 | 2,018 | 3,985 |
| squad | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 0.980 | 50 | 50 |
| musique | 0.803 | 0.835 | 0.844 | 0.921 | 0.934 | 0.939 | 2,092 | 3,985 |
| hotpotqa | 0.968 | 0.969 | 0.972 | 0.980 | 0.980 | 0.982 | 94 | 137 |
| 2wiki | 0.906 | 0.908 | 0.909 | 0.964 | 0.965 | 0.965 | 106 | 161 |
| webqsp | 0.688 | 0.782 | 0.838 | 0.811 | 0.883 | 0.921 | 2,120 | 4,039 |

Twice the walk budget, with I_q kept, adds at most 0.007 anywhere, so the union at step 4c's budget is the step. squad
has no expansion graph: its pool is retrieval's 50 rows, and 2% of its golds lie outside them. The "at most" pool size
is |I_q| + B_q; the walk's nodes that I_q already holds make U_q smaller, and the pools stage files the exact sizes.

The models are far below even today's pools: zrm's R@5 on musique is 0.52 to 0.56 against 0.92 of golds in the pool,
and on webqsp (zero-shot) 0.24 to 0.31 against 0.81. The pools are not the main gap. But a gold outside the pool is
lost to every model, and on webqsp, musique and metaqa's three-hop questions the union returns 3 to 11 points of golds.

## 3. Stages

Each stage is added to `scripts/step4d_union_pools.py` (or a look wrapper beside step 4c's) before it runs. A stage's
code may be written after this file; its rule may not change.

1. **pools** (laptop, numba, about 30 minutes): the union pools of step 4c's 21 carves, with the exact pool sizes
   and coverage of I_q, P_F and U_q per carve, filed in `outputs/step4d/pools/pools.json`. The walk order must equal
   step 4b's on the s1sel and s1eval carves, and every U_q must hold its I_q, or the stage stops.
2. **looks** (host CPU): look_x_six's scoring pass on the union pools, step 4c's look wrapper with the union
   replacement. The replacement checks that the base, seeds and frozen pool are the filed ones, and that the union
   pool is base and seeds plus the filed expansion. Output `outputs/step4d/look`.
3. **caches** (host CPU): lean_cache.py on those looks, step 1's part counts; then rmatch's chain caches for zrm on the
   union caches.
4. **identity gate:** squad's carves equal step 1's caches array by array (its pools do not change). On every other
   carve, each question's rows are |U_q|. The rows of I_q inside U_q carry the same compiled columns as step 1's
   cache, wherever a column does not depend on the pool's other rows.
5. **screen** (card): zrm, the base, trained on the union caches for L-musique and L-hotpotqa (scr-zrm-u, scr-zrm-u-hp),
   with zrm's settings, seed 0, p@swa, and read on the six union s1eval carves.

## 4. Verdict

- **Comparison.** Each read against zrm's screen fit of the split on today's pools (scr-zrm, scr-zrm-hp), question by
  question. lean_screen's comparison: R@5, a 2,000-resample question bootstrap, GAIN, LOSS or WITHIN at the floor.
- **The comparison's denominator.** R@5's denominator must be the question's golds, not the golds its pool holds, so
  that a gold the union adds counts on both sides. The comparison refuses two reads whose gold totals differ. If the
  union's gold totals differ from today's, the stage stops, and the denominator is fixed in an amendment to this file
  before any comparison's number is read.
- **The seed null.** The pair and the re-call run under the seed null, as every round (relz.py, decided against zrm's
  fits, zrc's mapping).
- **PROMISING** is at least one GAIN and no LOSS among the twelve reads. It opens a full run, declared in its own file
  before any of its numbers. An ADOPT there moves both models, the MLP and the GNN track, to the union pools; every
  base is then refit on them.
- **Otherwise** today's pools stay.

## 5. What it changes and what it does not

- No new graph, encoder, embedding or text. The walk runs on each dataset's own frozen expansion graph (step 4b), with
  step 4b's restart value. Only the rows in each pool change.
- Labels never pick a row: the union is built from the seeds, base and walk alone. Golds are read only to file
  coverage.
- webqsp never trains. Test splits are never read.
- **Speed.** On metaqa, musique and webqsp, pools grow up to about twice their size (exact sizes from stage 1). The
  per-question compile and the forward grow with them. Any latency figure is cold (8216ffe), and the step reports
  the pool sizes beside every number.

## 6. Order

Stage 1 starts on the laptop now. Stages 2 and 3 queue on the host CPU behind rounds twenty-four and twenty-five's
CPU items. Step 4c's looks took about 3 hours of wall time; the union's are up to about twice as large. Stage 5's
fits queue on the card behind round twenty-five's. Each stage's numbers are filed as they land.

## Results
