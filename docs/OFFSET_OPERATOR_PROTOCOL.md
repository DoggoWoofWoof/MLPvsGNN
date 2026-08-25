# Frozen screening protocol: relational operators versus message passing

> **Status:** `SCREENING_ONLY_NOT_PAPER_FINAL`  
> **Frozen:** 2026-08-25  
> **Protocol ancestor:** `paper-protocol-v0`  
> **Controlling configuration:** `configs/operator_screen.yaml`

## Decision question

Can a lightweight query-conditioned relational MLP recover graph-defined
relevance without explicit message passing, while improving the retrieval and
efficiency tradeoff relative to strong GNN operators?

This is a standalone retrieval/graph-learning study. C-RAG is a read-only source
of frozen data artifacts only. Its router, L1/L2/L3 system, fusion stack,
traversal, and final architecture are outside this paper.

The first screen is a decision experiment, not evidence for a final paper. It
asks whether the relational-operator direction is promising enough to justify
canonical five-seed evaluation. It must not be used to claim that an MLP is
universally better than a GNN.

## Immediate experimental set

| Dataset | Queries | Nodes | Query/node dim. | Graph edges | Canonical train/val/test |
|---|---:|---:|---:|---:|---:|
| WebQSP | 1,578 | 781,485 | 1,536 | 13,379,166 | 1,104 / 315 / 159 |
| 2Wiki | 15,000 | 65,865 | 1,536 | 855,146 | 10,500 / 3,000 / 1,500 |
| MuSiQue | 19,938 | 13,672 | 1,536 | 280,108 | 13,956 / 3,987 / 1,995 |

Every row uses the complete `queries_all`, `dense_top200_all`, and
`splade_top200_all` artifacts. HotpotQA, MetaQA, and SQuAD are excluded from
this screen and their ongoing encodings must not be duplicated.

## Frozen inputs and candidates

For query \(q\), the common candidate set is the stable union

\[
C_q = \operatorname{unique}(C_q^{dense@200} \Vert C_q^{splade@200}),
\]

preserving dense order and then adding unseen SPLADE candidates. The anchor is
fixed a priori as dense rank 1. All models receive the same frozen node and
query embeddings, candidate order, labels, negatives, split, seed, and
listwise multi-positive loss. Training includes only queries for which at least
one gold item is in the common pool; evaluation includes every query.

The frozen loader records hashes for source files and hashes of the realized
candidate, label, and split contracts. Those hashes must agree for all seven
models within a dataset.

Observed candidate diagnostics, before learning:

| Dataset | Mean pool size | Pool-size range | Mean recall ceiling | Queries with any in-pool gold |
|---|---:|---:|---:|---:|
| WebQSP | 343.16 | 241-400 | 0.5063 | 0.6572 |
| 2Wiki | 359.30 | 244-398 | 0.7905 | 0.9999 |
| MuSiQue | 331.87 | 222-398 | 0.9384 | 0.9992 |

These values are data diagnostics, not model results. In particular, WebQSP
requires both unconditional metrics and metrics conditional on candidate
availability so candidate generation cannot masquerade as an operator effect.

## Operators under test

All operators first map frozen 1,536-dimensional inputs into the same hidden
width. The screening width, depth, optimizer, learning rate, dropout,
temperature, epochs, batch size, and seed are fixed in the controlling YAML.

### Non-graph controls

`plain_mlp` predicts a query-conditioned target with a small MLP and scores
candidates by cosine similarity. It neither reads the anchor nor adjacency.

`offset_mlp` implements the preregistered relational translation:

\[
z = \operatorname{normalize}\left(a + g(q,a)\right),
\qquad
s_i = \cos(z,x_i)/\tau,
\]

where \(a\) is the frozen dense-rank-1 anchor and \(g\) is a small MLP.

`offset_mlp_k4` predicts four fixed relation directions and scores each
candidate by its maximum cosine similarity to those targets:

\[
z_k = \operatorname{normalize}\left(a + g_k(q,a)\right),
\quad k\in\{1,2,3,4\}.
\]

Neither offset model reads graph topology during training or inference.

### Message-passing comparators

The comparators are one-layer GCN, GraphSAGE, GATv2, and GIN. Each receives the
same candidate features and query state plus one privileged input: the induced
adjacency among candidates in \(C_q\). No GNN receives a larger candidate set,
different labels, or a different loss. Empty induced graphs remain valid
examples.

## Screening budget and selection

- Models: `plain_mlp`, `offset_mlp`, `offset_mlp_k4`, `gcn`, `sage`, `gat`, `gin`.
- One fixed seed: 0.
- Three epochs, batch size 16, hidden width 64, one message-passing layer.
- Shared optimizer settings: learning rate 0.001 and weight decay 0.0001.
- Checkpoint selection: validation R@5 only.
- Best GNN: highest validation R@5, selected independently per dataset.
- Reported Offset-minus-GNN differences: test metrics for both offset variants
  minus that validation-selected best GNN.

This uniform budget is intended for screening. It is not a claim that each
architecture is exhaustively tuned. Any paper-valid comparison must use an
equal, preregistered validation-only hyperparameter budget and canonical seeds.

## Required outputs

For every dataset/model pair, retain:

- R@1, R@5, R@20, MRR, and FullCov@1/@5/@20;
- conditional recall, hit rate, FullCov, and MRR among queries with in-pool gold;
- candidate ceiling and candidate-availability rate;
- total and trainable parameter counts;
- end-to-end training time and graph-preprocessing time;
- synchronized inference latency per query and throughput;
- total and incremental peak allocated GPU memory;
- validation checkpoint metric, seed, source hashes, contract hashes, and
  query-level predictions/metrics.

The compact decision table is:

`dataset | model | R@1 | R@5 | R@20 | MRR | FullCov@20 | params | latency | memory`

It must be accompanied by exact `offset_mlp - best GNN` and
`offset_mlp_k4 - best GNN` test differences. Accuracy-efficiency plots are
created only if an offset model is promising.

## Interpretation gate

The first screen can support one of three decisions:

1. **Promising:** an offset model is competitive with or better than the
   validation-selected GNN on retrieval and shows a plausible efficiency gain.
2. **Mixed:** the offset model helps in some relational regimes but not others;
   investigate the dataset/graph property that predicts the difference.
3. **Not promising:** gains disappear against multiple strong GNNs or depend on
   an efficiency measurement artifact; return to the topology-quality phase
   diagram rather than tuning levels to force an MLP win.

No subsequent expensive experiment may launch until the compact table is
written and reviewed. No conclusion may be based on the development smoke.

## Conditional follow-up, not part of this run

Only if the result gate is passed:

1. Freeze the promising configuration and run five canonical seeds.
2. Add HotpotQA when its complete common candidate cache exists, then MetaQA
   as a relational regime and SQuAD as a graph-light control.
3. Run required ablations: plain versus offset, K=1 versus K>1, hard negatives,
   no query condition, no anchor condition, parameter-matched and
   best-practical comparisons, and an identical-input/loss replay check.
4. Measure scaling at approximately 10k, 50k, 100k, and 500k+ nodes without
   claiming topology-independent cost beyond what the measurements show.
5. Only then resume controlled topology and feature-quality perturbations and
   mechanistic analysis.

The topology predictor, typed-edge ablations, broad dataset expansion, and
large rewiring sweeps remain explicitly deferred.
