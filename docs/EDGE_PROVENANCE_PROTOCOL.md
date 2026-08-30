# Package B: Edge-Provenance Protocol

Status: **frozen before any Package B test result is opened**.

## Question

Does the value of learned message passing depend on whether graph edges encode
native/symbolic relations, embedding similarity, or their union?

The clean six-dataset result alone cannot answer this because its frozen
`graph.pt` flattens edge semantics. SHA-256 verification shows that every clean
standalone graph is byte-identical to CRAG's `gte_qwen/graph.pt`, also called
variant A. Thus the clean result used:

```text
A = STRUCT union kNN
```

It did not use the NER-expanded variant C.

## Read-only reconstruction

The source CRAG repository and Modal source volume are read-only inputs. New
sidecars are written only to this standalone project's result volume.

- `STRUCT`: document-native `neighbors` from the source master-node manifest.
  Depending on the dataset, these represent document/title links or KB
  relations.
- `NER`: the frozen `ner_edges_w_df25.pkl` adjacency.
- `kNN`: exact set difference `A - STRUCT`; no embeddings are searched and no
  neighbors are regenerated.

All undirected edges are canonicalized as `(min(u,v), max(u,v))`. Node-row
alignment must pass either an exact numeric-suffix check or exact equality with
the standalone `node_ids.json` sidecar. Reconstruction fails if any STRUCT edge
is absent from A.

The exporter records edge counts, overlap counts, node-order hashes, and a
serialization-independent SHA-256 digest for every edge set. It also records
stored versus unique directed edge counts in sealed A.

The audit found that sealed A is bidirectionally closed but contains duplicate
directed edges. B/C construction deduplicates adjacency. Therefore a new
deduplicated **simple A** run is mandatory: otherwise duplicate message
weighting would be confounded with edge semantics.

## Frozen graph anchors

| Graph | Definition | Role |
|---|---|---|
| sealed A multigraph | stored `STRUCT union kNN`, including duplicates | Reuse the completed five-seed result |
| simple A | deduplicated bidirectional `STRUCT union kNN` | New normalization control |
| symbolic B | `STRUCT union NER` | Non-embedding relational/symbolic anchor |
| kNN-only | `A - STRUCT` | Embedding-similarity topology anchor |
| full C | `A union NER` | All recovered edge sources |

STRUCT-only and NER-only sidecars are exported for provenance inspection but
are not trained or used for headline selection in this package.

## Matched comparison

For simple A, symbolic B, kNN-only, and full C, train exactly:

```text
QLS-MLP
seed-aware validation-selected GNN
```

using seeds 0/1/2/3/4. Candidate IDs/order, node and query embeddings, labels,
canonical splits, retrieval seeds, loss, optimizer, epoch budget, selected GNN
family, and parameter regime remain frozen. The global edge family is the only
changed input. QLS features and the GNN candidate-induced adjacency are rebuilt
from the same selected graph.

The primary endpoint is paired `GNN - QLS` R@5. R@1, R@20, MRR, FullCov@20,
latency, memory, topology build time, and QLS feature build time are mandatory
secondary outputs. No family may be hidden after outcomes are observed.

## Interpretation

This package is not designed to find whichever graph makes QLS win. It tests
whether learned propagation and fixed graph computation respond differently to
edge semantics. A useful result can be bidirectional: message passing may help
on a sparse relational graph and hurt on semantically redundant or noisy kNN
topology.

No topology corruption levels, feature degradation levels, or phase-diagram
predictor may be selected from these test outcomes. Those belong to a separate
frozen package.
