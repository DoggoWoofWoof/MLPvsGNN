# Set-coverage Offset variant result

> **Historical negative result:** this completed gate remains part of the
> evidence trail. It predates SA-MLP and is superseded as the main project result
> by `SA_MLP_CONFIRMATION_RESULTS.md`; no second coverage remedy was launched.

> **Status:** `PREREGISTERED_COVERAGE_VARIANT_GATE`  
> **Run date:** 2026-08-25  
> **Frozen formulation:** commit `364183f`  
> **Seeds:** 0, 1, 2, 3, 4  
> **Outcome:** preregistered primary criterion failed on both datasets

## Decision

The exact injective set-assignment objective **does not repair Offset's coverage
failure**. It makes retrieval worse than the original K=4 model on both
datasets. No coefficient or formulation was changed after observing test data,
and no second coverage variant was run.

The result distinguishes diagnosis from remedy:

- Offset's original coverage loss does grow with answer multiplicity.
- Requiring distinct directions to explain distinct positives, plus the frozen
  diversity penalty, is not a successful solution under this protocol.
- The plain-MLP replication remained the strongest result among the
  topology-free models tested at this stage.

## Variant effectiveness

Values are mean +/- sample standard deviation over five seeds.

| Dataset | R@1 | R@5 | R@20 | MRR | FullCov@20 | Params | ms/query | GPU MiB | Train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2Wiki | 39.40 +/- 0.06 | 46.45 +/- 0.36 | 52.89 +/- 0.20 | 89.98 +/- 0.14 | 17.81 +/- 0.43 | 221,504 | 0.0768 | 47.95 | 28.98 |
| MuSiQue | 37.41 +/- 0.11 | 66.43 +/- 0.44 | 79.71 +/- 0.32 | 87.56 +/- 0.20 | 59.34 +/- 0.69 | 221,504 | 0.0767 | 45.88 | 37.50 |

Inference remains effectively the same fixed-cost four-direction operation as
original K=4. Training is 31% slower on 2Wiki and 28% slower on MuSiQue because
the objective evaluates exact injective assignments.

## Paired variant minus original K=4

Differences are percentage points with paired five-seed 95% t intervals.
Positive would favor the set-coverage variant.

| Dataset | dR@1 [95% CI] | dR@5 [95% CI] | dR@20 [95% CI] | dMRR [95% CI] | dFC@20 [95% CI] |
|---|---:|---:|---:|---:|---:|
| 2Wiki | -0.137 [-0.336,+0.063] | -2.813 [-3.362,-2.265] | -2.357 [-2.794,-1.919] | -0.421 [-0.849,+0.008] | -3.800 [-4.788,-2.812] |
| MuSiQue | -0.632 [-0.726,-0.537] | -3.471 [-3.845,-3.097] | -2.332 [-2.704,-1.960] | -1.404 [-1.615,-1.193] | -4.050 [-4.575,-3.525] |

The primary success requirement was at least +2.0 R@20 on **both** datasets
with each paired interval lower bound above zero. Both observed means are
negative with upper bounds below zero.

## Relative to the frozen GNN

| Dataset | dR@1 | dR@5 | dR@20 | dMRR | dFC@20 |
|---|---:|---:|---:|---:|---:|
| 2Wiki variant - GAT | +13.193 | -1.720 | -11.107 | +19.810 | -15.187 |
| MuSiQue variant - GCN | +0.921 | -9.192 | -7.063 | -0.383 | -12.401 |

The variant retains 99.0% of original K=4's R@1 advantage on 2Wiki but only
59.3% on MuSiQue, below the frozen 75% target there. It closes -26.9% of the
2Wiki and -49.3% of the MuSiQue K=4-to-GNN R@20 gap: negative values mean the
coverage gap becomes larger.

## Answer-count behavior

Variant-minus-original-K=4 R@20 differences remain negative in every group:

| Dataset | 2 gold | 3 gold | 4 gold |
|---|---:|---:|---:|
| 2Wiki | -2.57 | n/a | -1.56 |
| MuSiQue | -2.07 | -3.38 | -1.84 |

The new objective does not selectively recover larger answer sets; it degrades
the original K=4 scorer broadly. Its validation R@5 improves across training
epochs and the assignment/diversity losses decrease, so the result is not an
obvious crash or absence of optimization. The optimized objective is simply
misaligned with retrieval quality in this setting.

## Consequence for the paper

Do not claim that a coverage-aware Offset solves message passing's breadth
advantage. The defensible findings are now:

1. parameter-matched plain MLP beats the frozen message-passing comparator at
   R@5 on both 2Wiki and MuSiQue across five paired seeds;
2. it does so at about four times lower operator latency and substantially lower
   incremental memory, but not with 2-4x fewer parameters;
3. Offset has a reproducible early-rank/coverage tradeoff that worsens with
   answer multiplicity;
4. the first preregistered assignment-based remedy fails, showing that the
   tradeoff is not trivially removed by allocating one direction per positive.

This negative result should remain in the paper or appendix because it prevents
post-hoc architecture searching from masquerading as mechanism. Generic
topology perturbations, dataset expansion, and any second coverage variant were
not launched.

Raw variant results and paired analysis are retained under
`outputs/coverage_variant/`.
