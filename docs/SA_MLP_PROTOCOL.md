# Structure-Aware MLP screening protocol

Status: **frozen before any real SA-MLP test metric is computed**. The completed
six-dataset result remains immutable at tag `six-dataset-protocol-v1` and
commit `2f7f1aa`; this experiment writes separate caches and outputs.

## Question and falsifiable outcome

The experiment tests whether fixed structural summaries can recover the
retrieval benefit currently obtained by learned message passing. It does not
test whether MLPs universally beat GNNs and does not alter the frozen plain-MLP
or selected-GNN results.

For dataset `d`, the one-seed screening statistic is

\[
\operatorname{closure}_d =
\frac{R@5_{SA,d,0}-R@5_{MLP,d,0}}
     {R@5_{GNN,d,0}-R@5_{MLP,d,0}}.
\]

All three terms use seed 0. The screen passes only if the fully combined
SA-MLP closes at least 50% of the positive frozen GNN gap on at least two of
MetaQA, WebQSP, and HotpotQA. Passing permits an architecture freeze followed
by confirmation; failure stops the direction without inventing more variants.

## Frozen baselines and split contract

The exact main-table artifacts provide the baseline seed-0 predictions and
the validation-selected comparator: GAT on MetaQA and WebQSP, and GIN on
HotpotQA. Neither baseline is retrained. SA-MLP uses the identical 1,536-D
node/query arrays, dense-then-SPLADE candidate union, gold labels,
multi-positive listwise loss, official split, optimizer, three-epoch budget,
and seed. Test is scored once after validation-only epoch selection.

## Fixed structural inputs

Retrieval seeds are the stable union of frozen dense top-5 and SPLADE top-5
nodes. They never use relevance labels. Query-local algorithms operate on the
same candidate-induced directed adjacency supplied to the frozen GNN.

The seven static node descriptors are log out-degree, log in-degree, log total
degree, PageRank, total-degree percentile as a hub score, exact coreness on the
symmetrized simple graph, and a deterministic 64-wedge clustering estimate.
Continuous static columns are normalized graph-wide by z-scoring, clipping to
five standard deviations, and dividing by five.

The ten query-local columns are four minimum-distance buckets (0, 1, 2, and
3+/unreachable), normalized seed-connection count, normalized directed path
counts of length 1/2/3, eight-step personalized PageRank, and normalized common
out-neighbor count with the seed neighborhood. Counts use `log1p` followed by
division by the within-query maximum. No edge confidence is fabricated when
the frozen graph lacks it.

Feature generation receives node IDs, candidate IDs, retrieval-seed IDs, and
topology only. It has no label argument. Static and query-local arrays are
cached in packed row order with their input hashes, dimensions, build time,
and byte size. Preprocessing time and cache storage are reported separately
from online latency.

## Models and parameter matching

The ladder contains the frozen plain MLP plus four new scorers:

1. interaction: projected `q`, `x`, `q*x`, `abs(q-x)`, cosine, and dot;
2. static-structure: projected `q`, `x`, and seven static descriptors;
3. query-local-structure: projected `q`, `x`, and ten local descriptors;
4. SA-MLP: interaction, static, and query-local inputs together.

Every new scorer has separate linear 1,536-to-64 projections for `q` and `x`
and a two-layer scalar scoring head. Its head width is chosen deterministically
to be closest to the already-frozen selected GNN's parameter count, within 256
parameters. It has no convolution, adjacency argument, neighbor tensor, or
learned graph operation. A separately validation-selected practical width is
deferred until this one-seed gate passes, preventing an unregistered 12-model
screen.

## Systems and diagnostic reporting

Each new model reports R@1/5/20, MRR, FullCov, conditional metrics, parameter
count, training time, median-of-five online latency, throughput, total and
incremental peak GPU allocation, and total/incremental process RSS. MetaQA is
additionally reported at native 1/2/3-hop granularity. Feature preprocessing
seconds and cache bytes are reported once per dataset.

## Interpretation limits

Success would show that explicit fixed graph summaries can substitute for
some benefit of the tested GNN under this retrieval protocol. It would not
show that the graph is unnecessary, that every message-passing architecture
is dominated, or that cached graph algorithms are free. Failure would reject
this particular compact descriptor set, not all non-message-passing models.
