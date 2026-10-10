# U1e: damped links (a cap on a link target's in-degree)

Declared 10 October 2026, about 07:55, before any U1e graph is built or any pool is read. It follows U1d
(docs/U1D_PRECISE_LINKS.md) and step 4h (docs/STEP4H_U_POOLS.md).

## Why

U1d's chosen links (V2) recover 0.57 of 2wiki's hyperlinks and 0.68 of hotpotqa's, at precision 0.23 and 0.32. The most
linked targets are generic one-word titles. On webqsp the mention links reach a mean degree of 246 beside the KB's 6.4.
Step 4h found that GU loses 0.036 ALL on webqsp and 0.06–0.08 on 2wiki and hotpotqa against today's graph.

The U1e design look (outputs/mp_unified/u1e_look.py, 6c66198) read U1d's V2 links as they are. It reported
precision against the hyperlinks by the target's in-degree. That look is not a result, and nothing below is chosen
from it:
- **Precision falls with the target's in-degree.** On 2wiki it is 0.75 at in-degree 1, 0.35 at 128–256, 0.12 at
  32k–64k and 0.01–0.02 above 131k. On hotpotqa it is 0.73, 0.43, 0.15 and 0.0001–0.08.
- **A fixed cap on a target's in-degree lifts F1 on both datasets.** The mean F1 is 0.431 at a cap of 1,024, 0.434
  at 4,096 and 0.438 at 16,384, against 0.379 with no cap.
- **The caps bite differently on the six datasets.** At 4,096 they remove 0.51 of 2wiki's links, 0.46 of hotpotqa's,
  0.84 of webqsp's, 0.08 of musique's and none of squad's or metaqa's. (metaqa's largest in-degree is 577; squad's is
  837.)

## The rule

**GU_c**: U1d's structural_U (outputs/u1d/<D>/graph_structural_u.npz, as its build.json names it) with every mention
link removed whose target has more than c mention links in. The in-degree is counted over the mention links only.
- Every KB triple is kept on metaqa and webqsp.
- One c is used for all six datasets. Nothing is chosen per dataset.

**Candidates**: c ∈ {1,024, 4,096, 16,384} and c = ∞ (GU itself).
- Where a cap removes no link (metaqa and squad at every c; musique at 16,384), GU_c is GU. Its coverage is step 4h's
  filed GU record and is not run again. The build checks that the edge set is identical.

**Pools**: step 4h's coverage stage (scripts/step4h_u_pools.py coverage) is run unchanged on GU_c, with G0 alongside
as its identity check against steps 4c–4e.

**Choice**:
- **The read**: step 4h's chosen configuration, W1. Its mean ALL is taken over step 4h's six reads: the five s1sel
  carves and webqsp's s1eval.
- **The rule**: the highest mean ALL wins. Within 0.002 of the best, the smaller c wins (fewer links, a cheaper
  compile). The mean pool size must stay at most step 4e's, as step 4h requires.
- **The report**: hyperlink F1 for each GU_c on 2wiki and hotpotqa, beside the choice. It does not choose.

## What follows

- **If c = ∞ is chosen**, U1d's graph and U1c's retrain stand.
- **Otherwise**, a U1c rerun on GU_c, in its own file: the same models, fits and reads as U1c, on GU_c's step 4h
  pools.
- Either way, U1c's screen on GU keeps running. Its result is filed as declared.

## How it runs

```
python outputs/mp_unified/u1e.py build --dataset D --cap c --host        -> outputs/u1e/c<c>/<D>/graph_structural_u.npz, build.json
python outputs/host_ops/pylib_run.py outputs/mp_unified/u1e.py coverage --dataset D --cap c --threads 5 --host
                                                                           -> outputs/u1e/c<c>/coverage_<D>.json
python outputs/mp_unified/u1e.py choose                                   -> outputs/u1e/choice.json, report.md
```

- **Build**: it refuses if U1d's npz differs from its build.json sha256, or if the KB triple count changed.
- **Coverage**: step 4h's coverage_stage with its U1D root pointed at outputs/u1e/c<c> and its OUT at the same folder.
  Nothing else in step 4h changes.
- **Disk**: each GU_c is at most U1d's npz (webqsp 3.2 GB, 2wiki 0.7 GB, hotpotqa 0.3 GB), about 10 GB in all. The
  coverage records are small.

## Results

Not yet run.
