# CRAG extraction audit

## Safety boundary

`C:\Users\Swastik\Desktop\CRAG` is a read-only reference for this project.
Nothing is moved, deleted, renamed, or edited there. The new Git repository is
`C:\Users\Swastik\Desktop\message-passing-retrieval`.

## Exact provenance retained

The following tracked CRAG files were copied unchanged under
`legacy/crag_snapshot/`:

- `src/alignment/mlp_encoder.py`
- `src/alignment/gnn_encoders.py`
- `src/alignment/coverage_losses.py`
- `src/alignment/train_mlp.py`
- `src/alignment/train_alignment.py`
- `src/experiments/train_gnn.py`
- `results/gnn_ablation/{2wiki_clean,hotpotqa_clean,musique_clean}_gnn.json`

These snapshots exist to preserve provenance and identify old failure modes.
They are not imported by the new package.

## Why the old numbers are preliminary only

1. The original `train_alignment.py` GNN path detached partition embeddings;
   convolution weights received no useful gradient.
2. `train_gnn.py` fixed that issue with live-positive backprop, but its negative
   bank remained detached and the comparison was not parameter matched.
3. The task was partition-level L1 routing, whereas the strongest current MLP
   evidence is the L2 candidate scorer.
4. HotpotQA GNN execution was deferred, not completed.
5. Existing query caches do not constitute canonical dataset splits.
6. Edge types were flattened in `graph.pt`, preventing a clean edge-semantics
   claim.

The corrected L1 results are useful for forming the hypothesis that strong
semantic features plus noisy/high-degree neighborhoods favor an MLP. They are
not evidence for the final paper claim.

## Rewritten as independent components

| Need | New component | Reason for rewrite |
|---|---|---|
| data schema | `mp_retrieval.data`, `mp_retrieval.l2_data` | removes CRAG engine/FAISS coupling |
| L2 input construction | `mp_retrieval.l2_features` | guarantees the same inputs for MLP and GNN |
| paired models | `mp_retrieval.l2_models`, `mp_retrieval.models` | parameter matching and GAT support |
| topology controls | `mp_retrieval.perturbations` | reproducible, seed-controlled interventions |
| mechanism probes | `mp_retrieval.representation` | rank, cosine, energy, hub, gradient audits |
| metrics/statistics | `mp_retrieval.retrieval`, `mp_retrieval.graph_stats` | paired query-level evaluation |
| CRAG conversion | `scripts/export_crag_l2.py` | reads CRAG without making CRAG a dependency |

## Deliberately excluded

- C-RAG L1/L2/L3 architecture, routers, traversal, and universal-substrate code
- GraphRAG/HippoRAG comparison machinery
- paper drafts and C-RAG claims
- BM25, SPLADE, ColBERT, FAISS runtimes except as frozen upstream evidence
- checkpoints and caches
- generated logs, reports, scratchpads, and archived/leaked results
- large corpora and embeddings

## Useful read-only data inventory

Approximate current CRAG storage for the most relevant datasets:

| Dataset | CRAG artifact size | Current status for this paper |
|---|---:|---|
| 2Wiki-clean | 2.65 GB | best first L2 engineering substrate |
| MuSiQue-clean | 1.10 GB | strong multi-hop validation substrate |
| HotpotQA-clean | 14.03 GB | incomplete queries; pilot until rebuilt |
| MetaQA | 2.10 GB | typed relational positive-control target |
| WebQSP | large full KB | requires canonical typed-edge re-export |
| SQuAD-clean | large | graph-irrelevance negative control |

Current `signals_*_gte_qwen.npz` caches are small enough to export locally but
contain test queries only. The exporter labels them as pilot-only.

