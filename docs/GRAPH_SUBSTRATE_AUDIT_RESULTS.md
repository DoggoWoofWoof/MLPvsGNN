# Phase −1 results: what the retrieval graph actually is

This reports what `scripts/run_graph_substrate_audit.py` measured, laid out by
`scripts/analyze_graph_substrate.py`. It measures the substrate the frozen
operator already ran on; it changes no pool, admits no node, and moves no
frozen number. `docs/GRAPH_SUBSTRATE_AUDIT_PROTOCOL.md` fixed every definition
and every reporting rule before any of this was read, including the rule that
no adequacy threshold exists — so nothing below is called adequate, sufficient,
or shallow. The quantities are reported; the reader draws the line.

Three things the protocol insisted on are load-bearing in the tables and easy
to lose if skimmed:

- **Two connectivity notions, never merged.** Symmetrised connectivity answers
  "are these nodes related at all". Message flow answers "can this candidate
  actually receive that node's signal". Both are reported side by side.
- **Two retention aggregation levels, never mixed.** `node-pooled` pools every
  candidate across queries; `query-mean` averages per-query fractions. They
  have different denominators and they disagree; both are given.
- **Expansion headroom is oracle-only.** It is measured on a 512-query sample
  and admits nothing to any candidate pool. `candidate coverage != metric
  ceiling` still holds, and Paper-1 candidates are untouched.

## Status

| dataset | queries | nodes | stored directed edges | audit |
|---|---:|---:|---:|---|
| 2wiki_clean | 15,000 | 65,865 | 855,146 | complete, 4 graphs |
| musique_clean | 19,938 | 13,672 | 280,108 | complete, 4 graphs |
| metaqa | 407,513 | 40,151 | 585,728 | complete, 4 graphs |
| squad_clean | 130,319 | 19,029 | 2,857,316 | complete, 4 graphs |
| webqsp | 1,578 | 781,485 | 13,379,166 | complete, 4 graphs |
| hotpotqa_clean | 97,852 | — | — | queued, no output yet |

Every table below covers the five complete datasets on the `validation` split.
hotpotqa_clean is appended when its audit finishes; no conclusion here is
stated as holding for it.

## The headline measurements

**The two connectivity notions coincide, on every graph audited.** Not by
assumption — the audit computes the symmetrised receptive field and the exact
directed message-flow receptive field separately, and the difference is zero at
R1, R2 and R3 on all twenty graph-splits. The stored graphs are symmetric in
their reachability even where they are not symmetric in their storage. The
distinction the protocol insisted on turns out not to bite here; it still had
to be measured rather than assumed, and it must be re-measured on
hotpotqa_clean.

**A substantial fraction of candidates are scored as though the graph were not
there, and how large that fraction is depends strongly on the dataset.** On the
sealed graph the isolated fraction runs 0.175 (squad), 0.190 (webqsp), 0.224
(musique), 0.375 (2wiki), 0.412 (metaqa). An isolated candidate is not dropped;
it is scored, receiving only its operator-inserted self-loop. For that fraction
of the pool the one-layer GNN is a plain MLP. The fraction is *identical at R1,
R2 and R3* and equal to the measured `isolated_fraction` — verified numerically
on every graph-split, not inferred. A node with no induced neighbours gains
none at greater depth.

**Depth varies by a factor of seven at one hop and thirty-six at three, so
"shallow" is a property of a substrate and not of the method.** Median
symmetrised R1 on the sealed graph is 0.77 (metaqa), 1.01 (2wiki), 1.89
(musique), 2.07 (webqsp), 5.31 (squad). At R3 the spread runs 3.09, 8.00,
21.19, 64.41 and 110.38. The
historical operator is one layer, so R1 is what it used. R2 and R3 are reported
because the protocol requires depth to be measured rather than asserted, and
because the pre-registered GNN controls will be run at matched reach.

**Induced reachability and global reachability diverge with depth, by very
different margins.** Retrieval seeds reach the same fraction at one hop either
way, as they must. At three hops the induced and global figures are 14.4% and
97.3% on 2wiki, 20.2% and 87.2% on metaqa, 38.1% and 77.4% on musique, 51.2%
and 86.6% on squad, and 53.9% and 84.6% on webqsp. On 2wiki almost everything a
conventional GNN would reach at depth lies outside the scoring set; on webqsp
and squad rather less than half does.

**Retention is low everywhere, and lowest where the global graph is densest.**
On the sealed graph the node-pooled median retention is 0.080 (squad), 0.111
(2wiki), 0.154 (metaqa), 0.200 (musique), 0.250 (webqsp). Squad is the case
that explains the ordering: its median global degree is 41 against 6–7 for the
other four, so its induced graph is simultaneously the densest measured (median
R1 5.31) and the least retentive. Boundary-cut ratios run 0.700 (webqsp) to
0.904 (metaqa). Retrieval is cutting through neighbourhoods rather than
selecting graph-coherent regions.

**Gold path preservation is high; bridge loss is small but not zero, and
concentrated on one dataset.** Gold targets stay connected inside the induced
subgraph in 90.1% (metaqa) to 98.9% (webqsp) of all measured gold targets,
against 99.0% to 100.0% connected globally. The conditional form is
`bridge_loss@h`, the fraction of targets genuinely within `h` hops of a seed
that the induced graph cannot reach within `h` at all: at three hops that is
1.1% (webqsp), 1.2% (squad), 1.4% (2wiki), 3.6% (musique) and 9.9% (metaqa). It
is zero at one hop everywhere, which is expected — a direct seed-to-target edge
survives induction whenever both endpoints are in the pool. Bridge loss is the
`seed -> non-candidate -> candidate` pattern specifically, and neither a GNN
layer nor a QLS hop feature can see it.

Two fields in the path-preservation table are per-query **counts**, not
fractions, and are labelled as such: `lost by induction` and `targets`. On
metaqa the mean is 0.252 lost gold targets per query out of 2.10 gold targets
per query. They are counts because the audit records them as counts; converting
them to a rate here would mix mean-of-fractions with ratio-of-means.

## Duplicate messages are a third to over half of the operator's work

`dataset_default` and `baseline_a_simple` are **the same undirected graph**.
That is by construction, not coincidence: `configs/edge_provenance.yaml`
defines the latter as the deduplicated bidirectional projection of the former
and names it the mandatory duplicate-normalization control. The audit confirms
it independently — the two families hash to the same undirected edge key on all
five datasets — and the stored directed edge counts it read match the counts
Package B recorded for the families it trained, so both packages read the same
frozen artifacts.

Because every structural quantity is identical between the pair, any difference
in their behaviour is attributable to message multiplicity alone. That
difference is large in the operator's work and small in its output:

| dataset | duplicate fraction (sealed) | messages consumed, sealed → dedup | GNN R@5, sealed → dedup |
|---|---:|---:|---:|
| 2wiki_clean | 0.345 | 1065.4 → 823.8 | 0.6985 → 0.7005 (+0.20 pp) |
| musique_clean | 0.377 | 1888.8 → 1268.4 | 0.8124 → 0.8125 (+0.01 pp) |
| metaqa | 0.402 | 1115.6 → 813.2 | 0.3013 → 0.3012 (−0.01 pp) |
| squad_clean | 0.455 | 6484.0 → 3509.7 | 0.8933 → 0.8936 (+0.03 pp) |
| webqsp | 0.536 | 2601.8 → 1350.0 | 0.3309 → 0.3276 (−0.33 pp) |

The recall figures are Package B's, from
`docs/EDGE_PROVENANCE_AND_HEADROOM_RESULTS.md`; they are separately trained
models on the two graphs, not one model with messages removed. Across all six
datasets Package B measured, the largest absolute gap between the pair is 0.43
pp (hotpotqa_clean), in the direction of the deduplicated graph being *worse*.
On webqsp, where a majority of all messages are duplicates, removing them costs
0.33 pp. Between a third and over half of the messages the operator consumes
are exact duplicates, and removing them moves recall@5 by less than half a
point in either direction on every dataset measured.

## Structural edges and kNN edges are different graphs

Package B showed a graph can look connected through kNN edges while its
relational graph is fragmented. The audit measures both halves directly, and
the picture is consistent across all five datasets: kNN edges supply local
density, structural edges supply reach.

Isolation alone does not separate them — on 2wiki the structural-only graph
leaves 77.5% of candidates isolated against 48.8% for kNN-only, while on
musique the ordering reverses (67.8% against 32.3%). Two other measurements
separate them cleanly and in the same direction every time. Global seed
reachability at three hops:

| dataset | kNN-only, global @3 | structural-only, global @3 |
|---|---:|---:|
| 2wiki_clean | 0.149 | 0.870 |
| musique_clean | 0.226 | 0.568 |
| metaqa | 0.088 | 0.770 |
| squad_clean | 0.163 | 0.663 |
| webqsp | 0.160 | 0.810 |

And whether a gold target is reachable from a seed *at all in the full graph*,
before any induction:

| dataset | kNN-only, gold connected globally | structural-only, gold connected globally |
|---|---:|---:|
| 2wiki_clean | 0.934 | 0.997 |
| musique_clean | 0.883 | 0.950 |
| metaqa | 0.515 | 1.000 |
| squad_clean | 0.965 | 0.983 |
| webqsp | 0.338 | 1.000 |

On webqsp and metaqa the kNN graph does not contain a path to most gold targets
even before candidate induction, while the structural graph contains one for
essentially all of them. Whatever connectivity kNN contributes is local; the
sealed graph's reach comes from its structural edges.

## Expansion headroom — oracle only

The protocol keeps this question separate from everything above, and separate
from Paper-1's candidate contract. It asks how much a hop-expanded pool *would*
contain, not what any system should retrieve. `admits_nothing_to_the_pool` is
recorded true on every audit. No result in Packages A–E depends on it, and
nothing here licenses training on an expanded pool.

The multipliers are steep and they are not uniform. One hop from the retrieval
seeds grows the union by 1.10x (2wiki) to 1.83x (squad); two hops by 6.48x
(musique) to 110.59x (2wiki); three hops by 23.72x (musique) to 750.91x
(webqsp). The kNN-only graph barely expands at all — 1.05x / 1.26x / 1.94x on
2wiki, 1.04x / 1.20x / 1.69x on webqsp — which is the same fact as its short
global reach seen from the other side.

## Measurements

### Candidate-induced connectivity (validation split)

| dataset | graph | mean cand | mean directed edges | isolated | degree 1 | degree 2+ |
|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 359.7 | 464.1 | 0.375 | 0.291 | 0.333 |
| 2wiki_clean | structural | 359.7 | 117.7 | 0.775 | 0.165 | 0.060 |
| 2wiki_clean | kNN only | 359.7 | 346.5 | 0.488 | 0.259 | 0.252 |
| 2wiki_clean | baseline A | 359.7 | 464.1 | 0.375 | 0.291 | 0.333 |
| metaqa | sealed A | 373.1 | 440.1 | 0.412 | 0.320 | 0.268 |
| metaqa | structural | 373.1 | 228.1 | 0.604 | 0.302 | 0.094 |
| metaqa | kNN only | 373.1 | 212.0 | 0.644 | 0.227 | 0.129 |
| metaqa | baseline A | 373.1 | 440.1 | 0.412 | 0.320 | 0.268 |
| musique_clean | sealed A | 332.0 | 936.4 | 0.224 | 0.217 | 0.559 |
| musique_clean | structural | 332.0 | 431.3 | 0.678 | 0.143 | 0.179 |
| musique_clean | kNN only | 332.0 | 505.1 | 0.323 | 0.258 | 0.419 |
| musique_clean | baseline A | 332.0 | 936.4 | 0.224 | 0.217 | 0.559 |
| squad_clean | sealed A | 318.9 | 3190.8 | 0.175 | 0.153 | 0.672 |
| squad_clean | structural | 318.9 | 2806.2 | 0.509 | 0.076 | 0.416 |
| squad_clean | kNN only | 318.9 | 384.6 | 0.457 | 0.215 | 0.329 |
| squad_clean | baseline A | 318.9 | 3190.8 | 0.175 | 0.153 | 0.672 |
| webqsp | sealed A | 344.5 | 1005.5 | 0.190 | 0.248 | 0.563 |
| webqsp | structural | 344.5 | 442.9 | 0.420 | 0.420 | 0.160 |
| webqsp | kNN only | 344.5 | 562.6 | 0.388 | 0.180 | 0.432 |
| webqsp | baseline A | 344.5 | 1005.5 | 0.190 | 0.248 | 0.563 |

### Global-neighbourhood retention

| dataset | graph | mean (node-pooled) | median (node-pooled) | median global degree | ret = 0 (query-mean) | ret < 10% | ret < 25% | boundary cut |
|---|---|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 0.169 | 0.111 | 7.0 | 0.375 | 0.452 | 0.709 | 0.839 |
| 2wiki_clean | structural | 0.072 | 0.000 | 3.0 | 0.757 | 0.798 | 0.878 | 0.921 |
| 2wiki_clean | kNN only | 0.227 | 0.111 | 4.0 | 0.484 | 0.489 | 0.581 | 0.773 |
| 2wiki_clean | baseline A | 0.169 | 0.111 | 7.0 | 0.375 | 0.452 | 0.709 | 0.839 |
| metaqa | sealed A | 0.208 | 0.154 | 7.0 | 0.412 | 0.539 | 0.784 | 0.904 |
| metaqa | structural | 0.202 | 0.061 | 4.0 | 0.604 | 0.668 | 0.817 | 0.933 |
| metaqa | kNN only | 0.258 | 0.000 | 3.0 | 0.624 | 0.628 | 0.677 | 0.816 |
| metaqa | baseline A | 0.208 | 0.154 | 7.0 | 0.412 | 0.539 | 0.784 | 0.904 |
| musique_clean | sealed A | 0.280 | 0.200 | 7.0 | 0.224 | 0.314 | 0.529 | 0.792 |
| musique_clean | structural | 0.156 | 0.000 | 2.0 | 0.594 | 0.680 | 0.778 | 0.883 |
| musique_clean | kNN only | 0.367 | 0.333 | 4.0 | 0.314 | 0.315 | 0.389 | 0.633 |
| musique_clean | baseline A | 0.280 | 0.200 | 7.0 | 0.224 | 0.314 | 0.529 | 0.792 |
| squad_clean | sealed A | 0.219 | 0.080 | 41.0 | 0.175 | 0.531 | 0.677 | 0.881 |
| squad_clean | structural | 0.117 | 0.034 | 38.0 | 0.302 | 0.717 | 0.849 | 0.897 |
| squad_clean | kNN only | 0.383 | 0.333 | 3.0 | 0.364 | 0.365 | 0.415 | 0.616 |
| squad_clean | baseline A | 0.219 | 0.080 | 41.0 | 0.175 | 0.531 | 0.677 | 0.881 |
| webqsp | sealed A | 0.348 | 0.250 | 6.0 | 0.190 | 0.225 | 0.423 | 0.700 |
| webqsp | structural | 0.342 | 0.214 | 2.0 | 0.420 | 0.441 | 0.495 | 0.768 |
| webqsp | kNN only | 0.390 | 0.333 | 4.0 | 0.379 | 0.381 | 0.430 | 0.606 |
| webqsp | baseline A | 0.348 | 0.250 | 6.0 | 0.190 | 0.225 | 0.423 | 0.700 |

### Effective receptive field -- the two notions, kept apart

| dataset | graph | sym R1 | sym R2 | sym R3 | flow R1 | flow R2 | flow R3 | coincide | zero fraction |
|---|---|---|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 1.01 | 1.89 | 3.09 | 1.01 | 1.89 | 3.09 | yes | 0.375 |
| 2wiki_clean | structural | 0.01 | 0.18 | 0.29 | 0.01 | 0.18 | 0.29 | yes | 0.775 |
| 2wiki_clean | kNN only | 0.53 | 0.92 | 1.32 | 0.53 | 0.92 | 1.32 | yes | 0.488 |
| 2wiki_clean | baseline A | 1.01 | 1.89 | 3.09 | 1.01 | 1.89 | 3.09 | yes | 0.375 |
| metaqa | sealed A | 0.77 | 3.18 | 8.00 | 0.77 | 3.18 | 8.00 | yes | 0.412 |
| metaqa | structural | 0.23 | 1.66 | 3.22 | 0.23 | 1.66 | 3.22 | yes | 0.604 |
| metaqa | kNN only | 0.08 | 0.10 | 0.11 | 0.08 | 0.10 | 0.11 | yes | 0.644 |
| metaqa | baseline A | 0.77 | 3.18 | 8.00 | 0.77 | 3.18 | 8.00 | yes | 0.412 |
| musique_clean | sealed A | 1.89 | 7.09 | 21.19 | 1.89 | 7.09 | 21.19 | yes | 0.224 |
| musique_clean | structural | 0.20 | 1.63 | 6.12 | 0.20 | 1.63 | 6.12 | yes | 0.678 |
| musique_clean | kNN only | 1.23 | 2.55 | 3.89 | 1.23 | 2.55 | 3.89 | yes | 0.323 |
| musique_clean | baseline A | 1.89 | 7.09 | 21.19 | 1.89 | 7.09 | 21.19 | yes | 0.224 |
| squad_clean | sealed A | 5.31 | 26.72 | 64.41 | 5.31 | 26.72 | 64.41 | yes | 0.175 |
| squad_clean | structural | 3.53 | 18.50 | 40.04 | 3.53 | 18.50 | 40.04 | yes | 0.509 |
| squad_clean | kNN only | 0.74 | 1.32 | 1.84 | 0.74 | 1.32 | 1.84 | yes | 0.457 |
| squad_clean | baseline A | 5.31 | 26.72 | 64.41 | 5.31 | 26.72 | 64.41 | yes | 0.175 |
| webqsp | sealed A | 2.07 | 66.34 | 110.38 | 2.07 | 66.34 | 110.38 | yes | 0.190 |
| webqsp | structural | 0.64 | 62.73 | 99.85 | 0.64 | 62.73 | 99.85 | yes | 0.420 |
| webqsp | kNN only | 1.15 | 2.33 | 3.35 | 1.15 | 2.33 | 3.35 | yes | 0.388 |
| webqsp | baseline A | 2.07 | 66.34 | 110.38 | 2.07 | 66.34 | 110.38 | yes | 0.190 |

### Operator message load

| dataset | graph | unique edges | stored messages | duplicate fraction | operator self-loops | messages consumed |
|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 464.1 | 705.7 | 0.345 | 359.7 | 1065.4 |
| 2wiki_clean | structural | 117.7 | 117.7 | 0.000 | 359.7 | 477.3 |
| 2wiki_clean | kNN only | 346.5 | 346.5 | 0.000 | 359.7 | 706.2 |
| 2wiki_clean | baseline A | 464.1 | 464.1 | 0.000 | 359.7 | 823.8 |
| metaqa | sealed A | 440.1 | 742.6 | 0.402 | 373.1 | 1115.6 |
| metaqa | structural | 228.1 | 228.1 | 0.000 | 373.1 | 601.2 |
| metaqa | kNN only | 212.0 | 212.0 | 0.000 | 373.1 | 585.0 |
| metaqa | baseline A | 440.1 | 440.1 | 0.000 | 373.1 | 813.2 |
| musique_clean | sealed A | 936.4 | 1556.9 | 0.377 | 332.0 | 1888.8 |
| musique_clean | structural | 431.3 | 431.3 | 0.000 | 332.0 | 763.3 |
| musique_clean | kNN only | 505.1 | 505.1 | 0.000 | 332.0 | 837.1 |
| musique_clean | baseline A | 936.4 | 936.4 | 0.000 | 332.0 | 1268.4 |
| squad_clean | sealed A | 3190.8 | 6165.1 | 0.455 | 318.9 | 6484.0 |
| squad_clean | structural | 2806.2 | 2806.2 | 0.000 | 318.9 | 3125.1 |
| squad_clean | kNN only | 384.6 | 384.6 | 0.000 | 318.9 | 703.5 |
| squad_clean | baseline A | 3190.8 | 3190.8 | 0.000 | 318.9 | 3509.7 |
| webqsp | sealed A | 1005.5 | 2257.3 | 0.536 | 344.5 | 2601.8 |
| webqsp | structural | 442.9 | 442.9 | 0.000 | 344.5 | 787.4 |
| webqsp | kNN only | 562.6 | 562.6 | 0.000 | 344.5 | 907.1 |
| webqsp | baseline A | 1005.5 | 1005.5 | 0.000 | 344.5 | 1350.0 |

### Seed reachability -- induced versus global

| dataset | graph | induced @1 | induced @2 | induced @3 | global @1 | global @2 | global @3 |
|---|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 0.060 | 0.104 | 0.144 | 0.060 | 0.765 | 0.973 |
| 2wiki_clean | structural | 0.031 | 0.039 | 0.041 | 0.031 | 0.744 | 0.870 |
| 2wiki_clean | kNN only | 0.052 | 0.082 | 0.109 | 0.052 | 0.091 | 0.149 |
| 2wiki_clean | baseline A | 0.060 | 0.104 | 0.144 | 0.060 | 0.765 | 0.973 |
| metaqa | sealed A | 0.067 | 0.139 | 0.202 | 0.067 | 0.399 | 0.872 |
| metaqa | structural | 0.047 | 0.085 | 0.103 | 0.047 | 0.353 | 0.770 |
| metaqa | kNN only | 0.044 | 0.061 | 0.074 | 0.044 | 0.065 | 0.088 |
| metaqa | baseline A | 0.067 | 0.139 | 0.202 | 0.067 | 0.399 | 0.872 |
| musique_clean | sealed A | 0.114 | 0.261 | 0.381 | 0.114 | 0.399 | 0.774 |
| musique_clean | structural | 0.069 | 0.153 | 0.193 | 0.069 | 0.307 | 0.568 |
| musique_clean | kNN only | 0.072 | 0.130 | 0.185 | 0.072 | 0.139 | 0.226 |
| musique_clean | baseline A | 0.114 | 0.261 | 0.381 | 0.114 | 0.399 | 0.774 |
| squad_clean | sealed A | 0.187 | 0.371 | 0.512 | 0.187 | 0.504 | 0.866 |
| squad_clean | structural | 0.150 | 0.271 | 0.342 | 0.150 | 0.411 | 0.663 |
| squad_clean | kNN only | 0.064 | 0.109 | 0.147 | 0.064 | 0.112 | 0.163 |
| squad_clean | baseline A | 0.187 | 0.371 | 0.512 | 0.187 | 0.504 | 0.866 |
| webqsp | sealed A | 0.156 | 0.454 | 0.539 | 0.156 | 0.574 | 0.846 |
| webqsp | structural | 0.125 | 0.419 | 0.483 | 0.125 | 0.548 | 0.810 |
| webqsp | kNN only | 0.065 | 0.110 | 0.146 | 0.065 | 0.114 | 0.160 |
| webqsp | baseline A | 0.156 | 0.454 | 0.539 | 0.156 | 0.574 | 0.846 |

### Gold path preservation and bridge loss

| dataset | graph | targets (count/query) | connected globally | connected induced | lost by induction (count/query) | distance inflated | bridge loss @1 | @2 | @3 |
|---|---|---|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 1.85 | 0.998 | 0.984 | 0.032 | 0.001 | 0.000 | 0.013 | 0.014 |
| 2wiki_clean | structural | 1.85 | 0.997 | 0.973 | 0.053 | 0.001 | 0.000 | 0.023 | 0.024 |
| 2wiki_clean | kNN only | 1.85 | 0.934 | 0.928 | 0.013 | 0.000 | 0.000 | 0.003 | 0.006 |
| 2wiki_clean | baseline A | 1.85 | 0.998 | 0.984 | 0.032 | 0.001 | 0.000 | 0.013 | 0.014 |
| metaqa | sealed A | 2.10 | 1.000 | 0.901 | 0.252 | 0.024 | 0.000 | 0.078 | 0.099 |
| metaqa | structural | 2.10 | 1.000 | 0.838 | 0.519 | 0.012 | 0.000 | 0.094 | 0.162 |
| metaqa | kNN only | 2.10 | 0.515 | 0.494 | 0.054 | 0.001 | 0.000 | 0.016 | 0.046 |
| metaqa | baseline A | 2.10 | 1.000 | 0.901 | 0.252 | 0.024 | 0.000 | 0.078 | 0.099 |
| musique_clean | sealed A | 2.17 | 0.990 | 0.954 | 0.085 | 0.005 | 0.000 | 0.021 | 0.036 |
| musique_clean | structural | 2.17 | 0.950 | 0.893 | 0.136 | 0.003 | 0.000 | 0.037 | 0.058 |
| musique_clean | kNN only | 2.17 | 0.883 | 0.870 | 0.030 | 0.000 | 0.000 | 0.005 | 0.014 |
| musique_clean | baseline A | 2.17 | 0.990 | 0.954 | 0.085 | 0.005 | 0.000 | 0.021 | 0.036 |
| squad_clean | sealed A | 1.00 | 0.996 | 0.984 | 0.012 | 0.002 | 0.000 | 0.006 | 0.012 |
| squad_clean | structural | 1.00 | 0.983 | 0.967 | 0.015 | 0.001 | 0.000 | 0.009 | 0.016 |
| squad_clean | kNN only | 1.00 | 0.965 | 0.963 | 0.001 | 0.000 | 0.000 | 0.001 | 0.002 |
| squad_clean | baseline A | 1.00 | 0.996 | 0.984 | 0.012 | 0.002 | 0.000 | 0.006 | 0.012 |
| webqsp | sealed A | 4.27 | 1.000 | 0.989 | 0.033 | 0.012 | 0.000 | 0.026 | 0.011 |
| webqsp | structural | 4.27 | 1.000 | 0.979 | 0.057 | 0.007 | 0.000 | 0.029 | 0.021 |
| webqsp | kNN only | 4.27 | 0.338 | 0.327 | 0.043 | 0.000 | 0.000 | 0.006 | 0.044 |
| webqsp | baseline A | 4.27 | 1.000 | 0.989 | 0.033 | 0.012 | 0.000 | 0.026 | 0.011 |

### Graph-expansion headroom -- ORACLE ONLY, admits nothing to any pool

| dataset | graph | n | mean cand | U_seed H=1 | H=2 | H=3 | U_target H=1 | H=2 | H=3 |
|---|---|---|---|---|---|---|---|---|---|
| 2wiki_clean | sealed A | 512 | 359.2 | 1.10 | 110.59 | 167.17 | 5.51 | 129.73 | 175.50 |
| 2wiki_clean | structural | 512 | 359.2 | 1.05 | 110.36 | 137.87 | 2.89 | 125.15 | 146.35 |
| 2wiki_clean | kNN only | 512 | 359.2 | 1.05 | 1.26 | 1.94 | 3.80 | 10.66 | 24.86 |
| 2wiki_clean | baseline A | 512 | 359.2 | 1.10 | 110.59 | 167.17 | 5.51 | 129.73 | 175.50 |
| metaqa | sealed A | 512 | 360.8 | 1.49 | 22.78 | 81.68 | 19.82 | 80.60 | 111.28 |
| metaqa | structural | 512 | 360.8 | 1.47 | 21.71 | 70.60 | 18.34 | 76.70 | 110.39 |
| metaqa | kNN only | 512 | 360.8 | 1.02 | 1.12 | 1.40 | 3.25 | 8.12 | 17.29 |
| metaqa | baseline A | 512 | 360.8 | 1.49 | 22.78 | 81.68 | 19.82 | 80.60 | 111.28 |
| musique_clean | sealed A | 512 | 333.5 | 1.21 | 6.48 | 23.72 | 7.08 | 27.39 | 38.70 |
| musique_clean | structural | 512 | 333.5 | 1.18 | 5.77 | 17.96 | 5.35 | 22.32 | 28.13 |
| musique_clean | kNN only | 512 | 333.5 | 1.03 | 1.15 | 1.50 | 3.15 | 7.13 | 13.43 |
| musique_clean | baseline A | 512 | 333.5 | 1.21 | 6.48 | 23.72 | 7.08 | 27.39 | 38.70 |
| squad_clean | sealed A | 512 | 316.1 | 1.83 | 15.66 | 47.28 | 17.84 | 48.34 | 58.96 |
| squad_clean | structural | 512 | 316.1 | 1.82 | 14.58 | 38.89 | 16.68 | 40.33 | 41.65 |
| squad_clean | kNN only | 512 | 316.1 | 1.01 | 1.07 | 1.20 | 2.58 | 5.19 | 8.98 |
| squad_clean | baseline A | 512 | 316.1 | 1.83 | 15.66 | 47.28 | 17.84 | 48.34 | 58.96 |
| webqsp | sealed A | 315 | 344.5 | 1.21 | 82.24 | 750.91 | 7.20 | 375.42 | 1772.93 |
| webqsp | structural | 315 | 344.5 | 1.17 | 81.36 | 717.24 | 5.21 | 364.94 | 1710.65 |
| webqsp | kNN only | 315 | 344.5 | 1.04 | 1.20 | 1.69 | 3.25 | 9.53 | 25.45 |
| webqsp | baseline A | 315 | 344.5 | 1.21 | 82.24 | 750.91 | 7.20 | 375.42 | 1772.93 |

### Provenance aliasing -- families that are the same undirected graph

| dataset | families | undirected key | stored directed edges | stored symmetric | messages consumed |
|---|---|---|---|---|---|
| 2wiki_clean | `baseline_a_simple` + `dataset_default` | `bfef9e50fcae3e2e` | 521614 / 855146 | yes / no | 823.8 / 1065.4 |
| metaqa | `baseline_a_simple` + `dataset_default` | `54fbd3708ac9fe9d` | 329374 / 585728 | yes / no | 813.2 / 1115.6 |
| musique_clean | `baseline_a_simple` + `dataset_default` | `9b4795052dfe01cb` | 157898 / 280108 | yes / no | 1268.4 / 1888.8 |
| squad_clean | `baseline_a_simple` + `dataset_default` | `be333097e2aa8271` | 1445712 / 2857316 | yes / no | 3509.7 / 6484.0 |
| webqsp | `baseline_a_simple` + `dataset_default` | `369abaabd9ca5b7d` | 6621594 / 13379166 | yes / no | 1350.0 / 2601.8 |
## What this does not touch

Nothing in Phase −1 modifies a frozen result. The audit opens the sealed
artifacts read-only and writes only under `outputs/graph_substrate_audit/`.
Packages A, B, C, D and E1 are unchanged; E1 is not restarted; the candidate
pools, their hashes and their ceilings are untouched; CRAG is not read from or
written to by this package at all. The pre-registered GNN controls named in
protocol §7 have **not** been run, and their outcomes will not be used to
select QLS features.

## Reproducing the tables

```bash
python scripts/analyze_graph_substrate.py
```

That reads the per-dataset `substrate.json` files under
`outputs/graph_substrate_audit/`, writes `summary.json` beside them, and prints
the report. Then:

```bash
python scripts/render_substrate_tables.py --output tables.md
```

renders the `## Measurements` section verbatim from that summary. Every number
in this document is read from it; none is computed here, and the prose above
cites only figures that appear in a table below it. When the one outstanding
audit finishes, re-running both commands and re-splicing is the whole update.
