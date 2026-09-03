# Stage D8: the corrected support representation matches the proxy it replaces, and does not beat it

D7 found SUPPORT the one residual family that buys depth without selling the
head: +0.64 R@5 and +0.40 FullCov@20 at -0.03 R@1 and *+0.09* MRR. But the
column it measured, `seed_connections`, counts induced graph edges rather than
distinct retrieval seeds. D8 replaces the counting rule and asks whether the
corrected representation actually beats the proxy.

**The selection rule was corrected first, and filed before any D8 number
existed.** The project rule is not *choose the largest R@5*; it is *choose the
smallest, cheapest representation that remains on the effectiveness/system
Pareto frontier*. Under the old reading D7's `dominant_family` fired for PATHS
(+0.82 R@5 against SUPPORT's +0.64) and a PATHS replacement was the proposal
that followed. Under the correct rule SUPPORT is the Pareto move: one column
instead of three, head-neutral instead of head-negative, more coverage, and a
cheaper bounded computation. D7's thresholds and verdicts are untouched and
PATHS still measured +0.82.

**One new run.** `D6_BASE_13` is the exact matched 13-column control and
`D7_SUPPORT_13` the historical proxy under replacement, both fitted at this
architecture, seed, split and budget; both are reused, not refitted. Only
`D8_DISTINCT_SUPPORT_13` was trained.

2Wiki validation, seed 0, `G[Cq]`, three epochs, all three arms at **213,689**
parameters, head width 61 and `local_dim` 13. No architecture change.

## The historical formula, read from the kernel

VERIFIED FROM CODE — `src/mp_retrieval/structural_features.py:340-425`, not
inferred from the column's name.

    for each induced edge (u, v) of G[Cq]:
        if u is a retrieval seed:  connections[v] += 1
        if v is a retrieval seed:  connections[u] += 1
    column = log1p(connections) / max over the query's local node space

| property | value |
|---|---|
| quantity | induced edge **endpoints** incident between candidate and any seed — an edge count, not a seed count |
| hop radius | 1, direct adjacency inside `G[Cq]` |
| multiplicity possible | **yes** |
| reverse edges | count separately — a reciprocal pair `s->d`, `d->s` from **one** seed scores 2 |
| duplicate edges | count separately — each parallel stored edge scores 1 |
| self-loop on a seed | scores 2, because both branches fire |
| direction | ignored in effect: both orientations credit the non-seed endpoint |
| a seed adjacent to a seed | accrues support |
| normalisation | `log1p`, then divide by the per-query maximum |
| second normalisation | `candidate_readout` divides columns 4-9 again by the per-candidate maximum; under `CAND` the node space *is* the pool, so that division is by exactly 1.0 |
| maximum observed raw value | **28** (validation), **30** (all opened queries) |

Two defects, not one. The count is wrong (edges, not seeds), and the value is
not intrinsic: a candidate's reported number moves when a better-supported
candidate enters the pool, and every query with any support reports its best
candidate as exactly 1.0.

Computation path: `run_graph_context_d1.build_local_features` ->
`graph_context.qls_local_features` -> `structural_features._local_feature_chunk`
-> `graph_context.candidate_readout`.

## The replacement, defined before implementation

One replacement for one column. No support@1/@2/@3, no weighted support, no
path diversity, no diffusion.

    distinct_seed_support, replacing seed_connections at column 4

    s supports d  iff  an induced edge s->d or d->s exists in G[Cq] and s is a seed
    support_count(d)    = |{ s in Sq : s supports d }|
    support_fraction(d) = support_count(d) / |Sq|          <- the canonical scalar

The relation is *identical* to the historical one. Only the counting rule
changes: hop radius still 1, same induced edge list, a seed adjacent to a seed
still accrues support, same pool, same prior, same distance geometry, same
architecture.

`support_fraction` is canonical because it is bounded in [0,1] by construction,
means the same thing in every query, and needs no per-query normaliser — which
addresses the second defect, not only the first.

## What the two signals actually measure — before any effectiveness number

**This is not a selection criterion.** It is reported first so that any
effectiveness difference is read in the light of how much the two signals
differ at all. Measured on **validation feature construction**, the population
the effectiveness numbers come from; the same statistic over all 13,500 opened
queries is recorded beside it and agrees closely.

3,000 validation queries, 1,079,071 candidate rows, 5-10 seeds per query
(mean 7.84).

| | validation | all opened |
|---|---|---|
| rows with any support | 50,003 (**4.63%**) | 224,317 (4.62%) |
| corr(historical column, support_fraction) | 0.891 | 0.884 |
| corr(raw edge count, distinct seed count) | **0.961** | 0.962 |
| corr on supported rows only | 0.857 | 0.865 |
| rows where edges **exceed** distinct seeds | 50,003 | 224,317 |
| — as a fraction of **supported** rows | **100.0%** | 100.0% |
| max distinct support | 7 | 8 |
| max raw edge count | 28 | 30 |
| mean distinct support, supported rows | 1.228 | 1.238 |
| mean raw edge count, supported rows | 3.751 | 3.768 |

**The defect is universal where it can occur, and rare overall.** Every single
supported row over-counts — the historical column never once equals the
distinct seed count when there is any support at all — but only 4.63% of
candidate rows have any support to over-count. On the other 95.4% both signals
are zero and identical.

Where it occurs it is large: 3.75 edge endpoints per 1.23 distinct supporting
seeds, about **3.05 edges per distinct seed**. That is more than reciprocity
alone (2x) would produce, so parallel stored edges contribute as well
(INFERENCE, from the ratio; the reciprocity and parallel-edge mechanisms
themselves are VERIFIED FROM CODE above).

Corroborating: **no candidate row has induced degree exactly 1.**

| induced degree | rows | rows where edges exceed distinct | mean distinct | mean raw edges |
|---|---|---|---|---|
| 0 | 408,685 | 0 | 0.000 | 0.000 |
| 1 | **0** | 0 | — | — |
| 2-4 | 362,134 | 12,814 (3.5%) | 0.036 | 0.101 |
| 5+ | 308,252 | 37,189 (12.1%) | 0.157 | 0.490 |

An empty degree-1 bucket means 2Wiki's induced edges are stored in both
directions, so every connected candidate has at least two incident endpoints.
That is exactly the reciprocal-pair mechanism, visible in the degree
distribution rather than only in the kernel.

**But the two signals order candidates almost identically.** Over 194,238,384
within-query candidate pairs:

| | pairs | fraction |
|---|---|---|
| tied under both rules | 177,242,451 | 91.25% |
| concordant | 16,814,163 | 8.66% |
| tied by distinct support only | 168,254 | 0.087% |
| tied by the historical column only | 13,307 | 0.0069% |
| **strictly reversed** | **209** | **0.00011%** |
| **ordered differently at all** | **181,770** | **0.094%** |

Fewer than one pair in a thousand is ordered differently, and about one in a
million is actually reversed. Nearly all of the disagreement is the corrected
rule declaring a tie the proxy broke — which is what collapsing edge
multiplicity onto seed identity does.

**So: on 2Wiki the representation defect is real, universal wherever support
exists, and almost inconsequential for ranking.** The two signals are close to
the same signal here. Whatever the effectiveness numbers say next, they cannot
be read as evidence that fixing a defect that barely reorders anything is what
moved them.

## The ladder

| arm | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---|---|---|---|
| `D6_BASE_13` — retrieval prior + seed geometry (reused) | 39.39 | 71.38 | 77.70 | 93.88 | 51.90 |
| `D7_SUPPORT_13` — historical edge count (reused) | 39.37 | 72.02 | 77.90 | 93.96 | 52.30 |
| `D8_DISTINCT_SUPPORT_13` — corrected representation | 39.29 | 71.59 | 77.86 | 93.85 | 52.23 |

## The three increments

| increment | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---|---|---|---|
| **H - B** historical support given retrieval + geometry | -0.03 | +0.64 | +0.20 | +0.09 | +0.40 |
| **V - B** distinct support given retrieval + geometry | -0.10 | +0.21 | +0.16 | -0.03 | +0.33 |
| **V - H** corrected representation over historical proxy | -0.07 | **-0.43** | -0.04 | -0.12 | -0.07 |

`H - B` reproduces D7's `delta_support` exactly; the runner asserts it to 1e-9.

**Verdict: `DISTINCT SUPPORT PARETO-MATCHES HISTORICAL`.** Every `V - H` metric
is inside the +/-0.50 pp material band, so neither the better nor the worse set
is non-empty and the trichotomy returns the middle label.

**The margin, recorded rather than used.** R@5 at -0.43 misses the FAILS
threshold by **0.07 pp**. That is the same distance by which D7's DIFFUSION
missed PROMISING, and it is recorded here for the same reason: the threshold
was fixed in advance and does not move because a result landed near it.

**And the direction is not neutral.** All five `V - H` metrics are negative.
Each one individually is inside the band, but five out of five in the same
direction is not what a coin-flip tie looks like. The honest statement is:

> The corrected representation did **not** beat the proxy on effectiveness. It
> matches within the band and is consistently, slightly behind. It wins on
> representation and system grounds, not on accuracy.

That is rule B of the declaration, applied as written — *do not require an
accuracy increase when representation and system cost improve* — and not more
than rule B. Rule A did not fire: `rule_a_r5_gain_with_head_maintained` is
false, so nothing here promotes retrieval-weighted distinct support.

Against the base, the corrected column still pays: `V - B` keeps +0.21 R@5,
+0.16 R@20 and +0.33 FullCov@20. Distinct seed support is a useful signal. It
is just a slightly weaker one than the over-counting proxy, on this data.

## By stratum

`V - H`, on the 3,000 validation queries split by gold-node degree.

| stratum | queries | R@1 | R@5 | R@20 | MRR | FullCov@20 |
|---|---|---|---|---|---|---|
| isolated | 834 | -0.06 | -0.09 | +0.09 | -0.04 | +0.24 |
| degree_1 | 750 | +0.30 | -0.83 | -0.13 | +0.31 | -0.27 |
| low_degree | 1,218 | -0.37 | -0.37 | -0.12 | -0.45 | -0.25 |
| ordinary | 198 | +0.25 | -0.76 | +0.25 | -0.07 | +0.51 |

The R@5 loss is not concentrated in the stratum where support matters least. It
is spread across `degree_1` (-0.83), `ordinary` (-0.76) and `low_degree`
(-0.37), and is near zero only on `isolated` — the stratum where the gold node
has no edges and neither counting rule can say anything. That is consistent
with the pooled reading: where support exists at all, the proxy is very
slightly the better ranking signal here.

## Injection discipline

The declaration required the replacement to enter *after* historical candidate
readout, so the learner receives `distinct_support / |Sq|` with no second
rescaling.

- The injected column equals `count / num_seeds` elementwise over all 4,851,276
  candidate rows, `max_abs_diff 0.0`. **This is the gate.** It is exact, and it
  cannot hold under a second per-query rescaling unless that rescaling divided
  by 1.0 and therefore changed nothing.
- `normalised_columns` stays `(4,5,6,7,8,9)` and was **not** extended to the
  replacement; the scalar is appended after `candidate_readout`, exactly as D5's
  graded retrieval prior is.
- Corroborating, not enforcing: 13,480 of the 13,498 queries with any support
  report a column maximum strictly below 1.0. Under a second per-query
  rescaling that count would be 0.
- Column range over the whole block: [0.0, 1.0].

## The match

Every tensor invariant held at `max_abs_diff 0.0` over 4,851,276 rows:
distance geometry equal to both `D6_BASE_13`'s and `D7_SUPPORT_13`'s; prior
columns equal to the injected prior and to both reused arms'; column 4 equal to
the injected scalar; the historical arm carrying the frozen historical column.
`V` differs from `H` and from `B` in `seed_connections` and in nothing else;
every other residual column is exactly zero in both.

## Reuse instead of a third run

33 conditions were checked between `stage_d8` and `stage_d7` — dataset, seed,
data fingerprint, candidate contract, context, model, node count, both feature
schemas, `local_dim`, head width, parameters for all three arms, static feature
source, RRF constant, normalised columns, splits, and every training
hyperparameter — and all 33 matched.

Because D6's and D7's arms were fitted in other containers, eight
content-sensitive statistics were recomputed from D8's own rebuild and compared
to what D7 recorded: 4,851,276 candidate rows, 2,999,730 residual nonzero
entries, **224,317 historical-support nonzero rows** (checked separately,
because a drift confined to the one column D8 replaces would otherwise hide
inside the six-column total), 548,724 prior rows ranked by both and 0 by
neither, prior-vs-D4 difference 0.0, the seed identity proof at 0 mismatches
over 105,675 seed rows, and the gold stratum counts. All matched. On failure the
rule was to stop and inspect, never to refit around it.

## Systems

Incremental cost of constructing the new signal only. The whole QLS pipeline's
time is not attributed to one feature.

| | p50 | p95 | p99 | mean | max |
|---|---|---|---|---|---|
| **distinct-support kernel** (ms/query) | **0.080** | **0.095** | **0.119** | 0.123 | 581 |
| shared graph preparation (ms/query) | 2.832 | 3.227 | 3.427 | 2.905 | 4.385 |
| historical local feature build (ms/query) | 3.100 | 4.083 | 5.131 | 3.714 | 5,803 |

- **Kernel: 1.66 s total, 3.23% of the 51.4 s historical build.**
- Extracting the induced edges costs 39.22 s and is work the historical build
  already does; a production pipeline would share it, so it is reported
  separately rather than charged to the new feature. Charged together the two
  come to 42.0 s, 81.7% of the historical build — which is why the split
  matters.
- The 581 ms maximum is the first call only: numba compiles the kernel on the
  first query, against a p99 of 0.119 ms (INFERENCE, from the shape; the image
  carries `numba==0.60.0` and this is its first execution on Modal).
- **Temporary workspace: 3,184 bytes**, the largest query's mask array —
  `n * ceil(|Sq|/64) * 8` with `words_per_node = 1` for 2Wiki, so 398 nodes.
- Peak process RSS: 5.47 GB (whole process, dominated by the 4.85M-row feature
  block, not by this kernel).
- **One fixed graph pass.** Does not iterate to convergence. `O(|E[Cq]|)` to
  accumulate plus `O(n * words)` to zero the masks.
- No SIMD, numba specialisation or CUDA work was done. The declaration deferred
  it: D8 is first an information-value test, and the cost gate — *stop before
  training if construction exceeds the historical build* — did not fire
  (42.0 s against 51.4 s).

## Cost

One fitted model on one shared build: 51.4 s feature build, 2.1 s static, 10.3 s
assembling and proving the prior, 42.0 s building the replacement, 18.2 s
training and 0.35 s inference — **124.3 s accounted**, about **0.035
GPU-hours** and **$0.06** at $1.734/h, before container overhead. The
authorisation filed before launch was <= 0.08 GPU-hours and <= $0.40. The
gate's 0.2 h / $0.87 is D1's stage-independent whole-split ceiling and was never
a D8 forecast.

## What this does not establish

One replacement for one historical column. D8 says nothing about weighted
support, path diversity or diffusion.

**A Pareto match is not interchangeability.** It is evidence about this dataset.
On a graph that stores edges once rather than reciprocally, or with denser
multiplicity, the two signals would diverge further and this comparison would
have to be redone.

Not a statistical claim: -0.43 against a 0.50 threshold is one point estimate at
one dataset and one seed.

Not an architecture result: the 13-column block is a diagnostic, three columns
wider and 183 parameters larger than the frozen model, and is not the QLS-v2
input.

PATHS remains **PROMISING / DEFERRED**, measured at +0.82 R@5 in D7 — deferred
because SUPPORT was the Pareto-efficient first move, not because it lost.
Nothing in D8 changes that status.

One dataset, one seed, development evidence on `G[Cq]`. No GNN, no message
passing, no test split read.

## Verdict

    DISTINCT SUPPORT PARETO-MATCHES HISTORICAL

    representation   corrected: distinct seeds, not edge endpoints
    range            bounded [0,1] by construction, no per-query normaliser
    system cost      1.66 s, 3.23% of the historical build, one graph pass
    effectiveness    V - H negative on all five metrics, all inside +/-0.50 pp
                     R@5 -0.43, which misses FAILS by 0.07 pp
    carried forward  as the QLS-v2 support column, on Pareto grounds only
