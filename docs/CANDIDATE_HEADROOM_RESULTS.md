# Candidate-Generation Headroom: Results

Status: **complete for all six frozen datasets**. Diagnostic only. No candidate
pool was modified, expanded, reordered, or regenerated, and no frozen hash
changed. See `CANDIDATE_HEADROOM_PROTOCOL.md` for the contract.

Source: `outputs/candidate_headroom/<dataset>.json`. Every block below is the
**test split**, computed on the same query set as the reported confirmation
(query-count identity verified per dataset).

## 1. What was ever achievable

`g` = gold nodes per query, `p` = golds present in the frozen pool.
Coverage is `p/g`; the Recall@K ceiling is `min(p, K)/g`.

| dataset | test queries | golds/query | coverage (micro) | coverage (macro) | ceiling@1 | ceiling@5 | ceiling@20 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki_clean | 1500 | 2.42 | 0.7655 | 0.7967 | 0.4472 | 0.7967 | 0.7967 |
| musique_clean | 1995 | 2.34 | 0.9329 | 0.9405 | 0.4487 | 0.9405 | 0.9405 |
| webqsp | 159 | 6.17 | 0.4526 | 0.4906 | 0.3894 | 0.4501 | 0.4762 |
| hotpotqa_clean | 9786 | 2.00 | 0.9295 | 0.9295 | 0.4989 | 0.9295 | 0.9295 |
| squad_clean | 13033 | 1.00 | 0.9949 | 0.9949 | 0.9949 | 0.9949 | 0.9949 |
| metaqa | 39093 | 7.75 | 0.1405 | 0.3351 | 0.2638 | 0.3262 | 0.3349 |

Two caps are visible and must not be conflated.

* **The cut-off cap.** HotpotQA has exactly 2.00 golds per query, so Recall@1
  can never exceed 0.50 there however good the reranker is; the measured
  ceiling@1 is 0.4989. MetaQA (7.75 golds) and WebQSP (6.17) are capped harder
  still. Quoting `p/g` as an oracle Recall@K would have overstated all three.
* **The candidate-generation cap.** SQuAD loses essentially nothing to
  candidate generation; MetaQA and WebQSP lose about half their achievable
  recall before any model runs.

| dataset | ceiling@5 | perfect-retrieval ceiling@5 | recall lost to candidate generation@5 | test queries with no gold in pool |
| --- | ---: | ---: | ---: | ---: |
| 2wiki_clean | 0.7967 | 1.0000 | 0.2033 | 0 |
| musique_clean | 0.9405 | 1.0000 | 0.0595 | 0 |
| webqsp | 0.4501 | 0.8896 | 0.4395 | 58 |
| hotpotqa_clean | 0.9295 | 1.0000 | 0.0705 | 21 |
| squad_clean | 0.9949 | 1.0000 | 0.0051 | 67 |
| metaqa | 0.3262 | 0.8257 | 0.4995 | 18939 |

MetaQA has 18,939 of 39,093 test queries (48.4%) with no
gold node in the pool at all. Those queries score zero for every model, at
every cut-off, by construction.

## 2. Reported results beside their ceiling

Test Recall@5, five-seed mean, from the frozen SA-MLP confirmation.
`attainment` is reported / ceiling: the share of achievable recall the model
actually captured.

| dataset | QLS-MLP R@5 | seed-aware GNN R@5 | ceiling@5 | QLS attainment | GNN attainment | GNN - QLS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki_clean | 0.6840 | 0.6985 | 0.7967 | 0.859 | 0.877 | +0.0144 |
| musique_clean | 0.8028 | 0.8124 | 0.9405 | 0.854 | 0.864 | +0.0096 |
| webqsp | 0.3337 | 0.3309 | 0.4501 | 0.741 | 0.735 | -0.0027 |
| hotpotqa_clean | 0.7713 | 0.7766 | 0.9295 | 0.830 | 0.835 | +0.0053 |
| squad_clean | 0.8923 | 0.8933 | 0.9949 | 0.897 | 0.898 | +0.0010 |
| metaqa | 0.3011 | 0.3013 | 0.3262 | 0.923 | 0.924 | +0.0002 |

This reorders the cross-dataset story.

* **MetaQA is not a modelling failure.** Its raw Recall@5 of 0.30 is the worst
  in the table, yet it is 92% of everything the candidate set allowed. The
  headroom that remains there is upstream, not in the reranker.
* **WebQSP is the opposite case.** Its reported 0.334 sits against a ceiling of
  0.450, the lowest attainment in the table at 0.741. WebQSP has the *most*
  unexploited reranking headroom, not the least. The reported number could not
  be interpreted before the ceiling was put beside it.
* **The QLS-vs-GNN contrast is unaffected.** Both models see identical pools,
  so the ceiling cancels in the paired contrast. Every gap in the last column
  remains a clean reranking result.

## 3. Ceiling growth across the Package C budget sweep

Recall@5 ceiling per equal-RRF budget (test split).

| dataset | budget 50 | budget 100 | budget 200 | budget 400 | frozen union |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2wiki_clean | 0.7480 | 0.7637 | 0.7793 | 0.7967 | 0.7967 |
| musique_clean | 0.8620 | 0.8949 | 0.9208 | 0.9405 | 0.9405 |
| webqsp | 0.3002 | 0.3509 | 0.3956 | 0.4501 | 0.4501 |
| hotpotqa_clean | 0.8767 | 0.9004 | 0.9160 | 0.9295 | 0.9295 |
| squad_clean | 0.9804 | 0.9886 | 0.9933 | 0.9949 | 0.9949 |
| metaqa | 0.2104 | 0.2462 | 0.2862 | 0.3262 | 0.3262 |

Budget 400 reproduces the frozen union exactly on every dataset, which is the
expected identity and an internal check on the reconstruction.

**This is a required companion to Package C.** The ceiling rises monotonically
with budget on every dataset, so a raw metric gain from budget 50 to 400 is
partly the ceiling moving rather than the model reranking better. MetaQA's
ceiling grows from 0.2104 to
0.3262 across the sweep, a swing of
0.1158
that no model choice can be credited with. Package C effect sizes must be read
against the per-budget ceiling, not in absolute terms.

## 4. Where the two retrievers disagree

Coverage (micro) by source, test split.

| dataset | dense top-200 | SPLADE top-200 | frozen union | union - best single source |
| --- | ---: | ---: | ---: | ---: |
| 2wiki_clean | 0.7380 | 0.7276 | 0.7655 | 0.0275 |
| musique_clean | 0.9079 | 0.8670 | 0.9329 | 0.0251 |
| webqsp | 0.3242 | 0.3598 | 0.4526 | 0.0928 |
| hotpotqa_clean | 0.8882 | 0.8879 | 0.9295 | 0.0413 |
| squad_clean | 0.9668 | 0.9847 | 0.9949 | 0.0102 |
| metaqa | 0.1215 | 0.0637 | 0.1405 | 0.0190 |

WebQSP gains the most from fusing the two sources; dense and SPLADE fail on
largely different queries there. MetaQA is the pathological case: both
retrievers are weak and fusing them still reaches only 0.141 coverage.

## 5. Missing-gold reachability

Read-only bounded BFS on the undirected global graph, starting from the frozen
retrieval seeds (dense top-5 union SPLADE top-5), test split, three hops.
Buckets are disjoint shortest-hop distances.

| dataset | queries missing a gold | missing golds | d=1 | d=2 | d=3 | >3 or unreachable | within 3 | frontier-capped |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki_clean | 673 | 852 | 84.0% | 13.8% | 1.8% | 0.4% | 99.6% | 0 |
| musique_clean | 286 | 313 | 26.5% | 42.5% | 23.0% | 8.0% | 92.0% | 0 |
| webqsp | 98 | 537 | 24.4% | 40.0% | 32.2% | 3.4% | 96.6% | 0 |
| hotpotqa_clean | 1,358 | 1,379 | 70.8% | 27.8% | 1.3% | 0.0% | 100.0% | 0 |
| squad_clean | 67 | 67 | 11.9% | 37.3% | 43.3% | 7.5% | 92.5% | 0 |
| metaqa | 29,079 | 260,233 | 20.7% | 37.8% | 41.4% | 0.0% | 100.0% | 0 |

No query hit the frontier budget on any dataset, so no missing gold is
mislabelled as unreachable because of a compute limit.

Between 92% and 100% of missing gold evidence sits within three hops of nodes
the retriever already returned. On 2Wiki, 84% is a single hop away; on MetaQA,
all 260,233 missing golds are reachable within three.

**This is a measurement, not a license.** These nodes are not admitted to any
Paper-1 candidate pool, and no expansion, admission, or regeneration was
performed. Candidate expansion remains a Paper-2 / G2 question, and the
Paper-1 protocol is unchanged by this result. What the number establishes is
that MetaQA's and WebQSP's low ceilings are a property of the *candidate
generator*, not of the corpus: the evidence is present and topologically
adjacent, and the frozen fusion rule simply does not reach it.

## 6. Interpretation rules now in force

1. Report every primary metric beside its ceiling on the same split and K.
2. Never quote `p/g` coverage as an oracle Recall@K; quote `min(p, K)/g`.
3. Do not compare absolute metric levels across datasets without their
   ceilings; cross-dataset differences are confounded by candidate coverage.
4. Read Package C budget effects against the per-budget ceiling.
5. Nothing in this document may change candidate generation, budgets, model
   selection, or perturbation rates for Paper 1.
