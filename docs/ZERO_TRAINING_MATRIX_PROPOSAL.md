# Proposal: the zero-training dataset x regime matrix

**Status: `PROPOSED_NOT_DECLARED`.** This is a proposal. It is **not** a
declaration, it authorises nothing, and no part of it may be launched on the
strength of this document. No declaration, no launch.

D10 was the last 2Wiki-only feature-development stage. Everything in the frozen
[feature catalog](GRAPH_CONTEXT_FEATURE_CATALOG.md) is one dataset and one
seed. The obvious next question -- does any of it transfer -- cannot be
answered by fitting more arms on 2Wiki, and should not be answered by fitting
arms on five more datasets before anyone has looked at what those datasets
contain.

So the first matrix trains nothing.

## The matrix

Six datasets x three regimes.

| dataset | queries | note |
| --- | ---: | --- |
| `squad_clean` | 130,319 | single-hop |
| `musique_clean` | 19,938 | multi-hop |
| `2wiki_clean` | 15,000 | the development dataset; every D-stage result |
| `hotpotqa_clean` | 97,852 | multi-hop |
| `metaqa` | 407,513 | KB-style, by far the largest |
| `webqsp` | 1,578 | KB-style, by far the densest graph |

| regime | node space | graph |
| --- | --- | --- |
| R1 | `Cq` | `G[Cq]`, the frozen candidate-induced graph |
| R2 | `Cq` | `TARGET_H1(Cq)`, restored one-hop context |
| R3 | `Cq'` | `TARGET_H1(Cq')`, a re-derived pool |

Scoring stays `scored_nodes == Cq` in every regime, and `Cq` remains a subset
of the node universe. R1 and R2 differ only in which edges the features may
see; R3 differs in which candidates exist at all.

**R3 is blocked in paper 1 and this proposal does not ask for it.** The frozen
`diagnostic_contract` in [`configs/candidate_headroom.yaml`](../configs/candidate_headroom.yaml)
sets `candidate_regeneration: prohibited_in_paper_1`,
`candidate_admission: prohibited_in_paper_1` and
`graph_expansion: prohibited_in_paper_1`, and records that reachable missing
golds are a paper-2 question. A re-derived `Cq'` is candidate regeneration by
definition. Twelve cells are runnable; R3's six are named here so that their
absence is deliberate, and they belong to paper 2 unless that contract is
changed by its own declaration rather than by this one.

## What is measured

**No neural training. No fitted arm. No model selection of any kind.** The
matrix produces distributions and counts, and nothing that could rank a model.

**1. Candidate headroom.** Reuse the definitions that already exist in
[`src/mp_retrieval/candidate_headroom.py`](../src/mp_retrieval/candidate_headroom.py)
rather than restating them: `any_gold_at_pool`, `all_gold_at_pool`,
`gold_fraction_at_pool_macro`, the micro and macro missing-gold fractions, and
the `recall_ceiling@K` / `recall_ceiling_perfect_retrieval@K` split that keeps
the candidate-generation cap apart from the cut-off cap. That layer already
covers all six datasets and is read-only over frozen pools. Restating the
formulas in a new module would create a second definition of the same quantity,
which is the failure this reuse exists to avoid.

**2. Feature occupancy.** Per dataset, per regime, per local column: nonzero
rate, and the value distribution over candidate rows. On 2Wiki, 666,517 of
4,851,276 rows carry any nonzero PATH value. Whether that ratio is a 2Wiki
property or a general one is currently unknown, and a family that is nonzero on
2% of rows somewhere else is a different feature there.

**3. Family correlation and redundancy.** The pairwise structure among the nine
catalog families, within each dataset. D3 through D10 measured increments, not
redundancy; whether GEOMETRY and SUPPORT carry the same information is a
question no fitted arm answered.

**4. Graph size and construction cost.** Nodes, directed edges, induced-edge
counts per query, and construction latency at p50, p95 and p99 -- per dataset,
per regime, per stage of the build. D10 showed the same code reading 1.2449 of
the historical build on one container and 97.79% on another, so latency is
reported as per-query percentiles on a stated container, never as a total.

## Why zero training is the point

Three of the nine families have no fitted evidence at all, and the two that do
have replacement evidence produced a match and a mixed result. Fitting arms on
five new datasets now would produce thirty more increments over a base arm
chosen by 2Wiki, and every one of them would inherit that choice.

The occupancy and redundancy measurements are also the cheaper ones. If a
family is nonzero on almost no rows in a dataset, its increment there is
predictable without fitting anything, and knowing that before spending GPU time
is the whole argument for measuring first.

## Cost

Not budgeted here, deliberately.

D10 measured roughly 6.8 ms per query end to end for 2Wiki construction --
50.7 s of historical build plus 40.86 s of shared extraction over 13,500
queries. Applied naively to 672,200 queries across all six datasets that is
about 1.3 hours of CPU work, and that number should not be trusted: WebQSP's
sealed graph carries 13,379,166 directed edges against 2Wiki's 855,146, and
MetaQA has twenty-seven times 2Wiki's query count. The per-query cost is a
function of pool size and local density, and neither is known outside 2Wiki.

**A declaration should set the budget from a 1,000-query probe per dataset, not
from this extrapolation.** The headroom layer already runs CPU-only at 8 cores
and 32 GB, so the matrix is plausibly a CPU job with no GPU line at all -- but
that is a claim to check in the probe, not one to file.

## What this proposal explicitly does not ask for

- No neural training anywhere in the first matrix
- No GNN, as a QLS component or as a selector of QLS features
- No candidate regeneration, admission or graph expansion; R3 stays closed
- No modification of any frozen candidate pool, hash or tag
- No resumption of E2, which is paused, and nothing that opens F, which is
  sealed
- No workspace migration
- No test-split read

## What would have to be filed before anything runs

A declaration naming: the exact regimes, the exact per-dataset query caps and
how they are selected, the container and the cost ceiling, which measurements
are reported and at what precision, the abort rule stated in machine-robust
per-query percentiles rather than wall-clock totals, and what each possible
outcome would and would not authorise -- all before the first number is opened.

Until that exists, this document is a proposal and nothing more.
