# Package D: Uncached Unseen-Embedding Systems Protocol

Status: **frozen before timing**. Execution is gated on completed Package C
budget-400 checkpoints and exact cached/on-demand parity.

## Boundary

This is a post-retrieval ranker benchmark, not a raw-query retrieval-system
benchmark. Each request begins with:

```text
held-out query embedding
+ Dense ranked document/entity IDs
+ SPLADE ranked document/entity IDs
```

Raw text, tokenization, query encoding, Dense ANN search, and SPLADE index
serving are shared upstream operations and remain out of scope.

The timed path charges:

1. equal-RRF candidate ordering and frozen seed construction;
2. candidate embedding gathering;
3. candidate-induced topology extraction from the corpus CSR;
4. QLS distance/path/PPR summaries, for QLS only;
5. host/device transfer;
6. learned forward pass; and
7. top-20 selection.

No cache keyed by query ID, embedding, candidate set, seed set, topology, or
QLS local features may be read in the timed path. Corpus node embeddings,
global graph CSR, corpus-static QLS node features, and trained checkpoints are
allowed static serving assets.

## Matched checkpoint

The benchmark uses seed-0 QLS-MLP and seed-aware GNN checkpoints from the
frozen equal-RRF Package C budget-400 condition. This avoids timing a model on
a candidate contract it was not trained to score. Model weights do not affect
operator shape, but their file hashes are still verified.

The QLS online summaries are quantized to the same float16 local-feature
contract used in training and converted to float32 for inference. Node,
query, and corpus-static features remain float32.

## Mandatory parity gate

Before timing, 32 deterministic held-out queries must show:

- exact candidate-rule/budget agreement with Package C;
- bit-exact candidate-induced edge arrays against the cached reference; and
- bit-exact QLS local features after float16 quantization.

The cached topology and local-feature stores may be read only for this untimed
gate and are released before the timing loop. Any parity failure blocks the
benchmark.

## Measurement

Use up to 1,024 evenly spaced queries in canonical test order, all unseen
during ranker training. WebQSP uses its full smaller test split. Run batch sizes
1 and 16 for three passes, after 64 warmup queries. Alternate method order by
batch to reduce systematic thermal/order bias and synchronize CUDA around the
forward region.

Report stage and total mean/p50/p95/p99 latency, per-query latency, throughput,
cold start, CPU thread count, static asset bytes/hashes, and per-method peak
RSS, total GPU allocation, and incremental GPU allocation above the ready
baseline. The runner resets CUDA peak accounting immediately before each
method batch; it never infers a method-level peak from a joint run. Preserve
the Package C cached-reranker latency as a separate operator-only view; do not
replace or splice it into this uncached table.

The result may support an uncached speed claim only after these measurements.
Until then, all existing speed ratios remain warm-cache claims.
