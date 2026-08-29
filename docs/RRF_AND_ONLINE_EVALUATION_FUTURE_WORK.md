# RRF Fusion and Unseen-Embedding Systems Evaluation

Status: **deferred future work**. This document does not change the sealed
SA-MLP confirmation protocol, checkpoints, test results, or statistical
decisions. Any execution requires a new preregistration and new result files.

## Why this follow-up is necessary

The completed confirmation answers a controlled model question on frozen
candidate pools. It does not yet answer two production-facing questions:

1. Would a stronger parameter-free fusion of the Dense and SPLADE rankings
   improve the retrieval prior supplied to every model?
2. What is the post-retrieval latency for a genuinely unseen query embedding
   when every query-dependent graph and ranking operation is performed on
   demand?

These questions must remain separate. RRF is an effectiveness/prior control.
Uncached post-retrieval timing is a systems-accounting correction.

## Scope boundary

This project is not constructing or benchmarking an upstream retrieval system.
The real-world serving interface begins at:

```text
query embedding
+ upstream Dense ranked candidate IDs
+ upstream SPLADE ranked candidate IDs
```

Raw query text, tokenization, query-encoder serving, ANN search, and SPLADE
index serving remain upstream and out of scope. The existing frozen query
embeddings and ranked IDs are sufficient to exercise this interface. A future
stream of new embeddings may be exported by an upstream retriever without
adding raw query text to the standalone data contract.

## Audit of the current frozen artifacts

The six complete sources contain:

```text
dense_top200_all.npy
splade_top200_all.npy
```

They preserve ranked global node IDs, but do not preserve matching raw score
arrays. This has two consequences:

- standard Reciprocal Rank Fusion can be evaluated exactly from the stored
  ranks;
- raw-score fusion such as calibrated CombSUM cannot be reconstructed from
  these files and would require rerunning the retrievers or exporting scores.

The current candidate pool is a stable union: Dense top-200 first, followed by
previously unseen SPLADE top-200 nodes. The final learned rankers are intended
to be candidate-order equivariant, so this ordering is a deterministic data
contract rather than an explicit Dense/SPLADE fusion score. The current seed
prior is binary membership in the stable union of Dense top-5 and SPLADE top-5.

## Future RRF experiment

### Locked parameter-free baseline

For source rankings \(s \in \{\text{Dense},\text{SPLADE}\}\), use one-indexed
rank \(r_s(d)\) and the preregistered score

\[
\operatorname{RRF}(d)=
\sum_s \frac{\mathbf{1}[d\in s]}{60+r_s(d)}.
\]

An absent document contributes zero for that source. Ties must be broken by a
frozen deterministic rule, such as global node ID after the existing stable
source order. The constant 60 is locked before validation or test evaluation.

The first table should report, on the unchanged candidate union:

- Dense rank alone;
- SPLADE rank alone;
- equal RRF;
- the frozen learned models.

Report R@1, R@5, R@20, MRR, FullCov@20, candidate-conditional metrics, and
candidate ceiling. Equal RRF cannot improve candidate ceiling when it only
reorders the same union; it can improve where in-pool gold nodes appear.

### Validation-only weighted RRF

Weighted RRF is a secondary control:

\[
\operatorname{WRRF}(d)=
\frac{w_D\mathbf{1}[d\in D]}{60+r_D(d)}+
\frac{w_S\mathbf{1}[d\in S]}{60+r_S(d)}.
\]

The weight grid and tie rule must be preregistered. Select weights using
validation R@5 only, then evaluate test once. Report equal RRF regardless of
whether weighted RRF improves validation. Do not select weights, the RRF
constant, or the number of seeds using test outcomes.

### RRF as a shared model prior

RRF may help SA-MLP by selecting better structural seeds, but that must not
create a new information asymmetry. A fair new comparison should include:

1. the sealed binary top-5-union prior as a reference;
2. an RRF scalar feature supplied identically to seed-only MLP, SA-MLP, and
   seed-aware GNN;
3. RRF top-\(K\) seeds supplied identically to SA-MLP and the seed-aware GNN;
4. the same candidates, labels, loss, optimizer seeds, and validation rule.

Changing the structural seeds changes both SA features and the GNN's seed
indicator. All affected models must therefore be retrained under a newly frozen
protocol. These results must not be spliced into the sealed confirmation table.

Candidate fusion and candidate selection must also be distinguished:

- **Rerank the full union:** same candidate ceiling; tests fusion quality.
- **Truncate an RRF-ranked union to a fixed budget:** changes the candidate
  pool and ceiling; tests candidate-generation efficiency and requires a new
  matched data contract.

## Audit of the current latency measurement

The reported 2.49--7.08x SA-MLP/GNN ratio is a valid **warm-cache
candidate-reranking** comparison. It is not a complete uncached
post-retrieval serving measurement for an unseen query embedding.

The timed loop currently includes:

- gathering frozen candidate and query embeddings;
- loading SA rows from the packed structural-feature cache;
- loading GNN edges from the packed candidate-topology cache;
- host/device tensor preparation performed inside the scoring path;
- the learned forward pass; and
- copying model scores back to CPU.

It excludes or performs before the timed loop:

- candidate union or fusion;
- construction of the candidate-induced topology;
- computation of query-local SA distance/path/PPR features;
- corpus-static graph-feature construction;
- model and index cold start; and
- final metric computation.

Query encoding and Dense/SPLADE retrieval are also excluded, but they are
shared upstream services outside this paper's ranker boundary rather than
missing ranker operations.

Both graph-aware methods currently benefit from preprocessing. The GNN reads a
prepacked candidate-induced topology, while SA-MLP reads that topology's
derived query-local structural summaries. A GNN can consume newly induced
adjacency without computing SA summaries, but it is not topology-extraction
free. For a new query, both methods require candidate graph induction; SA-MLP
then requires additional query-local fixed graph computation.

The previously reported 9.3--20.5 seconds is bulk preprocessing over each
frozen dataset and includes corpus-static and query-local work. Dividing that
number by the query count is not an acceptable substitute for online timing:
bulk vectorization, JIT warmup, memory locality, and batching differ from a
single unseen-embedding service path.

## Required unseen-embedding latency protocol

### Allowed corpus-static preprocessing

The following can reasonably be built once for a fixed corpus and reported
separately:

- document/entity embeddings and Dense ANN index;
- SPLADE document index;
- global graph CSR;
- corpus-static degree/PageRank/coreness-style node values;
- trained model checkpoints; and
- immutable node-ID mappings.

Storage size, build time, refresh time, and peak memory for these assets must be
reported. Dynamic-graph invalidation cost is a separate systems endpoint.

### Post-retrieval query-dependent work that must be charged

For a previously unseen query embedding and upstream candidate rankings, the
post-retrieval timer must include:

1. equal/weighted RRF or stable-union construction;
2. candidate embedding and metadata gathering;
3. candidate-induced subgraph extraction for every graph-aware method;
4. seed selection;
5. query-local distance, path, connectivity, and PPR computation for SA-MLP;
6. device transfer and model forward pass;
7. final top-\(K\) selection.

No cache keyed by query ID, query embedding, candidate set, seed set, or
candidate-induced topology may be read in the uncached unseen-embedding
condition.

### Two latency views

Report both rather than replacing the existing diagnostic:

| View | Scope | Purpose |
| --- | --- | --- |
| Cached reranker | Frozen candidates and packed per-query caches | Isolates learned scoring/operator cost and preserves the completed result |
| Uncached post-retrieval | Starts from an unseen query embedding and upstream Dense/SPLADE rankings, then builds the fusion, induced topology, and SA features online | Supports a real-world ranker speed claim |

The paper boundary is:

```text
upstream query embedding + ranked candidate IDs
                       |
                       v
candidate fusion + method-specific graph work + learned ranking + top-K
```

Upstream encoder and retriever latency should be disclosed as out of scope, not
silently described as measured. No end-to-end retrieval-system speed claim will
be made.

### Measurement conditions

Use the same hardware, software image, loaded corpus indexes, process lifetime,
candidate budget, numerical precision, and concurrency for all models. At
minimum report:

- batch size 1 and the existing throughput-oriented batch size 16;
- p50, p95, and p99 latency, not only the mean;
- queries/second under fixed concurrency;
- peak GPU allocation and process RSS;
- CPU thread count and GPU utilization;
- cache/index storage and graph-feature storage;
- cold-start time separately from steady-state service time; and
- at least one embedding/candidate stream whose query IDs were never used to
  build query-local topology or feature caches.

CUDA work must be synchronized around timers. Warmups must be identical. Top-K
selection must occur inside the timed production path, while evaluation metric
aggregation remains outside it.

## Interpretation gate

Until this protocol is executed, the correct claim is:

> SA-MLP is 2.49--7.08x faster than the seed-aware GNN in warm-cache
> candidate-reranking latency while using substantially less incremental GPU
> memory.

It is not yet justified to claim a 2.49--7.08x uncached post-retrieval speedup
for unseen embeddings. The on-demand SA computation may shrink, eliminate, or
reverse the ratio; the purpose of the future experiment is to measure that
outcome rather than assume it.

## Decision order on resumption

1. Implement and unit-test equal RRF over the frozen ranked IDs.
2. Freeze the RRF tie rule and effectiveness protocol before test evaluation.
3. Implement a cache-disabled on-demand candidate-graph and SA-feature path.
4. Verify numerical parity between cached and on-demand SA features and between
   cached and on-demand candidate-induced GNN topology.
5. Freeze the systems protocol and hardware image.
6. Run screening timings without inspecting effectiveness test metrics.
7. Run the canonical unseen-embedding timing benchmark and report every
   dataset.
8. Only then decide whether the paper can make an uncached post-retrieval speed
   claim.

The wider set of submission-critical controls and the bounded real-world query
plan are recorded in
[the paper-readiness audit](PAPER_READINESS_AND_REAL_WORLD_FUTURE_WORK.md).
