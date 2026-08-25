# Six-dataset main-table protocol

Status: **frozen before any new WebQSP, HotpotQA, SQuAD, or MetaQA test result**.

## Paper claim under test

This study does not assume that MLPs always beat GNNs. It asks when
candidate-graph message passing earns its additional topology and systems cost.
The falsifiable outcomes are that message passing helps, is neutral, or hurts
retrieval under a fixed feature/candidate/training contract.

Finding A is already frozen and is not retuned: width-64 plain MLP beats the
validation-selected GAT on 2Wiki by 2.077 R@5 points and the selected GCN on
MuSiQue by 1.103 R@5 points across five paired seeds. The corresponding paired
95% intervals are [1.22, 2.94] and [0.57, 1.63] percentage points. Those two
results come only from the committed confirmation artifacts.

## New main-table experiment

For WebQSP, HotpotQA-clean, SQuAD-clean, and MetaQA, select one comparator from
GCN, GraphSAGE, GATv2, and GIN using validation R@5 only. Architecture selection
uses seed 0 and the same already-frozen three-epoch budget. Test metrics are not
computed during selection. Ties follow the preregistered order GCN, SAGE, GAT,
GIN.

After selection, compare width-64 plain MLP with the selected width-64 GNN on
seeds 0--4. Seed 0 reuses the selected validation checkpoint; it is not tuned or
retrained after architecture selection. Every comparison has identical frozen
1,536-dimensional node/query embeddings, stable dense-then-SPLADE candidate
unions, gold labels, multi-positive listwise loss, splits, optimizer, epoch
budget, and seed. Candidate-induced adjacency is the GNN's only extra input.

The primary table reports R@1, R@5, R@20, MRR, and FullCov@20, plus paired
seed-level MLP-minus-GNN 95% Student-t intervals. Metrics conditional on at
least one retrievable gold are placed in a separate table so candidate recall
cannot be mistaken for reranker quality.

## MetaQA identity and hop contract

MetaQA uses entity string gold IDs rather than numeric document suffixes. Its
row-ordered `node_ids.json` sidecar is derived from the frozen SPLADE
`id_to_idx` mapping and fingerprinted with the six core source artifacts. A
partition ID is never treated as a node row. The native manifest hop label is
carried through each query record and results are aggregated independently for
1-hop, 2-hop, and 3-hop test queries without changing or selecting models.

## Systems contract

Latency uses batch size 16, one warmup batch, and the median of five full-test
passes per trained seed. Report throughput, total and incremental peak GPU
memory, parameter counts, training time, and cold candidate-induced topology
preprocessing. The topology builder is an edge-exact vectorized CSR expansion
with packed int32 storage; optimization is accepted only after edge-for-edge
parity tests against the original implementation.

## Stop rule

No Offset rescue, new architecture, query-level mechanism model, or graph
corruption is run until the six-dataset effectiveness table, MetaQA hop table,
and systems table all exist. New outcomes are reported regardless of direction.
