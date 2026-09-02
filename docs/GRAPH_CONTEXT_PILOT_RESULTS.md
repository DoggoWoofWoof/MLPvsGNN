# Graph-context pilot: results

**Question.** What is the smallest defensible graph context that repairs the
graph starvation Phase -1 found, without turning every query into a whole-graph
computation?

**Answer.** `TARGET_H1 = Cq u N1_in(Cq)` -- the candidate pool plus the
in-neighbours of candidates. It restores radius-1 message retention to 1.0000 on
all six datasets, removes every isolated candidate, and costs 2,012-7,520
context nodes and 8.7-170.1 ms p95 per query. `SEED_H2`, `PATH_H2` and
`BRIDGE_H2` are dominated and killed. Nothing was trained and no GPU was used.

**What that sentence is not.** `TARGET_H1` is the leading *graph-information
repair regime*. It is not the final QLS-v2 serving algorithm, and this document
makes no effectiveness claim at all. Two distinctions carry the whole reading of
these results and are easy to collapse by accident:

| | |
|---|---|
| **graph-information privilege** | which structural information a system is *permitted* to see. `TARGET_H1` is a candidate for this. |
| **QLS feature backend** | how the permitted statistics are *computed*. Materialising `G[Cq u N1_in(Cq)]` is one implementation; it is not the only one, and the 170 ms tail belongs to the implementation rather than to the regime. |

| | |
|---|---|
| **structural identity** | retention 1.0000 and isolate fraction 0.0000 follow largely from including a candidate's global one-hop neighbourhood. They show the construction performs its intended repair. They are not evidence that the repaired information helps ranking. |
| **empirical finding** | the restored reach concentrates on exactly the candidates Phase -1 identified as starved, monotonically in prior degree, on all six datasets -- and the whole thing is computationally measurable at 8.7-170.1 ms p95. |

Read section 6 before quoting the context sizes: on the three small corpora
those 2k-7.5k nodes are 16-30% of the whole graph at the median, and calling
that a "bounded local context" without saying so would be false.

Every number below is MEASURED on the validation split unless labelled
otherwise. Sections 4 and 5 separate identities from findings line by line,
section 8 states the property that gives the arm its interpretation, and
section 10 lists what this does not establish. **We have not shown that
restoring this information improves retrieval ranking. That is Stage D0's
question, and it is the next thing to earn.**

## 1. What was run

Stages A-C of the ladder: feature-only, read-only, no model trained, no GPU, no
gold id, no test split, no candidate pool touched, no GNN anywhere.

| stage | what | cost |
|---|---|---|
| A | synthetic correctness, 1,337 tests | seconds, local |
| B | 25 validation queries, 2wiki + hotpotqa, six arms | 2 containers, 33 s + 278 s |
| C | 300 validation queries, all six datasets, three arms | 6 containers, 35-351 s |

No query was skipped for want of a retrieval seed at either stage: 25/25 and
300/300 on every dataset.

The scored set is exactly `Cq` in every arm. Context nodes contribute structure
and are never ranked, so the candidate ceiling is identical across arms by
construction and none of this is a candidate-generation change.

## 2. The arms

`Sq` is a subset of `Cq` by construction of the frozen artifact
(`seed_global = pool[seed_local]`, VERIFIED FROM CODE), so `Cq u Sq == Cq` and
the seed arms differ from the target arms only in what they expand from. `Sq` is
much the smaller set: `SEED_H1` adds 9-56% more nodes at the median where
`TARGET_H1` adds 470-1977%, on a median `Cq` of 316-364 candidates. The
seeds-per-query count is not recorded in the pilot's result files and is not
asserted here.

| arm | definition | status |
|---|---|---|
| `CAND` | `Uq = Cq` | historical control |
| `SEED_H1` | `Uq = Cq u N1_out(Sq)` | survives, weakly |
| `TARGET_H1` | `Uq = Cq u N1_in(Cq)` | **survives, recommended** |
| `SEED_H2` | `Uq = Cq u N<=2_out(Sq)` | killed at B |
| `PATH_H2` | `Uq = Cq u {v : s -> v -> c, s in Sq, c in Cq, s != c}` | killed at B |
| `BRIDGE_H2` | `Uq = Cq u {v : c1 -> v -> c2, c1,c2 in Cq, c1 != c2}` | killed at B |

`SEED_H3`, `TARGET_H2` and `TARGET_H3` were killed before any compute, on
Phase -1's already-measured corpus share (33.0-100.0% of the graph). They are
not context; they are the corpus.

In-neighbours and not out-neighbours for `TARGET_H1`: every frozen operator
flows source-to-target, so `N1_in(Cq)` is what a one-layer GNN on `G[Cq]`
actually lost.

## 3. The historical substrate is as starved as Phase -1 said

MEASURED, 300 validation queries per dataset, on `CAND`:

| dataset | median radius-1 retention | boundary cut | isolated candidates |
|---|---:|---:|---:|
| squad_clean | 0.1096 | 0.8811 | 17.1% |
| hotpotqa_clean | 0.0908 | 0.8893 | 33.9% |
| 2wiki_clean | 0.1200 | 0.8425 | 36.8% |
| metaqa | 0.1591 | 0.9173 | 24.6% |
| musique_clean | 0.2195 | 0.7985 | 21.7% |
| webqsp | 0.3170 | 0.7000 | 19.1% |

Between 70% and 92% of a candidate's stored one-hop messages never reach it, and
between 17% and 37% of candidates have no neighbour at all inside `G[Cq]`. This
is not "a sparse graph". Three things are separate and stay separate:
candidate-to-candidate edges are preserved exactly at radius 1; the full global
one-hop neighbourhood is badly truncated; the induced receptive field is
therefore sparse.

## 4. What the repair is: the identity half

Retention rises to **1.0000 on all six datasets**, boundary cut to 0.0000 on
five and 0.0015 on hotpotqa, and the isolated fraction to 0.0000 everywhere.

**These are largely consequences of the definition, and must be reported as
such.** Retention counts a candidate's stored out-edges landing inside `Uq`, and
`TARGET_H1` admits every node with an edge into a candidate. If every stored
edge carries its reverse, then `v -> w` with `v` a candidate implies `w -> v`,
so `w` is an in-neighbour of a candidate and is already inside
`Cq u N1_in(Cq)`; retention is then 1.0000 by construction, and an isolated
candidate would need a candidate with no stored neighbour at all. Including a
candidate's global one-hop neighbourhood repairs a radius-1 truncation because
that is what it was built to do.

That premise is VERIFIED directly on 2wiki -- its 855,146 stored edges are
521,614 distinct pairs and every pair carries its reverse. On the other four it
is an INFERENCE from the substrate audit, which found the symmetrised and
directed receptive fields indistinguishable across all nine summary statistics
at three depths but retained aggregates rather than per-node sets. hotpotqa is
the measured exception and behaves accordingly: its directed message-flow
receptive field is strictly smaller than its symmetrised one at every hop, and
it is the one dataset here with a non-zero boundary cut under `TARGET_H1`
(0.0015). The identity is therefore not free -- the 0.0000 on the other five is
a measurement that the premise holds there, not a tautology over all graphs.

**What this establishes:** the construction performs its intended repair, and it
does so completely rather than partially. **What it does not establish:** that
the repaired information is worth anything to a ranker. Quoting 1.0000 retention
as evidence that "we fixed the graph" would be quoting a definition.

## 5. What the repair is: the empirical half

Two things here are genuine findings rather than restatements of the arm.

**Where the restored reach lands.** Fraction of candidates brought strictly
closer to a seed than `G[Cq]` had them, stratified by the induced degree the
candidate had before any context was restored:

| dataset | isolated (deg 0) | deg 1 | deg 2-4 | deg 5+ |
|---|---:|---:|---:|---:|
| hotpotqa_clean | 0.9941 | 0.8108 | 0.5594 | 0.2390 |
| 2wiki_clean | 0.9753 | 0.9202 | 0.7696 | 0.5315 |
| metaqa | 0.9512 | 0.4862 | 0.2748 | 0.1469 |
| squad_clean | 0.9241 | 0.7522 | 0.4996 | 0.1778 |
| webqsp | 0.9048 | 0.4266 | 0.3615 | 0.1403 |
| musique_clean | 0.8731 | 0.7327 | 0.4946 | 0.1887 |

Monotone decreasing in prior degree on all six datasets without exception, and
87-99% of formerly isolated candidates helped against 14-53% of the well
connected ones. Nothing in the definition of `Cq u N1_in(Cq)` requires this: an
arm that widened the context and moved every candidate a little would satisfy
retention 1.0000 just as well and would be rescaling, not repair. That is the
preregistered advancement criterion, and the restored information demonstrably
reaches the candidates Phase -1 identified as starved.

**That the whole thing is computationally measurable.** The regime is not a
thought experiment about what a bigger context would contain: it is 2,012-7,520
median context nodes and 8.7-170.1 ms p95 per query on real hardware, small
enough that the next question can be asked at all.

Neither finding is an effectiveness result. "The restored information reaches
the starved candidates" and "the restored information helps rank them" are
different claims, and only the first is measured here.

## 6. What was killed, and why it cannot come back

| arm | dominated by, at Stage B | on |
|---|---|---|
| `PATH_H2` | `SEED_H1`, `BRIDGE_H2`, `SEED_H2`, `TARGET_H1` | both datasets |
| `BRIDGE_H2` | `TARGET_H1`, `SEED_H2` | both datasets |
| `SEED_H2` | `TARGET_H1` | both datasets |

`PATH_H2` and `BRIDGE_H2` die **structurally**, not narrowly. Each is a subset
of a cheaper arm -- `PATH_H2` of `SEED_H1`, `BRIDGE_H2` of `TARGET_H1` -- because
identifying a path node needs a second matrix step that a radius rule does not
take. `PATH_H2` recovered what `SEED_H1` recovered to three decimal places
(0.294 against 0.294 on 2wiki, 0.310 against 0.310 on hotpotqa) at 6.2x and 2.8x
the build cost, and its seed-reach vector is identical to `SEED_H1`'s to four
decimal places on both datasets. No quantity of further queries reverses a
subset relation.

That the literature-style enclosing contexts are dominated by a plain radius-1
in-neighbourhood is a result rather than a formality, and it is why they were
built and run rather than argued about.

`SEED_H2` failed worse than the filed hypothesis predicted. It was expected to
advance on information and fail on cost. It lost on both: 0.811 and 0.950 median
retention against `TARGET_H1`'s 1.000, while holding 60.2% and 52.9% of the
corpus graph and costing 74.4 ms and 2,119.4 ms end-to-end at the median.

`SEED_H1` survives the preregistered Pareto rule -- it is dominated only on
hotpotqa and webqsp, being marginally cheaper to build on the other four -- and
is retained as an ablation rather than a recommendation. Its seed-distance
improvement is 36-92% of `TARGET_H1`'s depending on dataset, and on metaqa,
musique, squad and webqsp it helps only 15-24% of formerly isolated candidates
against `TARGET_H1`'s 87-95%.

## 7. The context is bounded in nodes, not in share of the graph

This is the caveat that matters most for how the result is stated.

| dataset | corpus nodes | `TARGET_H1` p50 | p95 | max | share p50 | share max |
|---|---:|---:|---:|---:|---:|---:|
| webqsp | 781,485 | 2,012 | 4,764 | 17,029 | 0.26% | 2.18% |
| hotpotqa_clean | 507,494 | 2,890 | 7,509 | 142,073 | 0.57% | 28.00% |
| 2wiki_clean | 65,865 | 2,060 | 2,616 | 3,188 | 3.13% | 4.84% |
| metaqa | 40,151 | 7,520 | 12,039 | 13,349 | 18.73% | 33.25% |
| squad_clean | 19,029 | 5,752 | 8,378 | 9,561 | 30.23% | 50.24% |
| musique_clean | 13,672 | 2,248 | 3,355 | 4,167 | 16.44% | 30.48% |

The absolute size is stable and small: 2,012-7,520 nodes at the median and at
most 12,039 at p95, on corpora spanning 13.7k to 781k nodes. That is the number
that determines serving cost, and it is the number that should be quoted.

The **share** is not small on the small corpora. On squad the median context is
30% of the graph and one query in the sample reaches 50%. That is not because
the rule is expansive; it is because a fixed-size one-hop neighbourhood of
316-364 candidates is a large fraction of a 19,029-node graph. Describing
`TARGET_H1` on squad, metaqa or musique as a "bounded local context" without
that qualification would be false, and the qualification belongs in the paper
next to the claim.

hotpotqa's 142,073-node maximum is one query in three hundred against a p95 of
7,509 and a p99 of 21,778. It is a hub effect in the tail, and any serving story
needs a cap for it.

## 8. Why `TARGET_H1` is the right repair: two-hop path preservation

Retention and isolation say the arm restores a candidate's *neighbours*. The
property that gives it a precise interpretation is stronger, and it is a
theorem rather than a measurement.

**Claim.** Let `U = Cq u N1_in(Cq)` with `N1_in(Cq) = {v : exists c in Cq, v -> c}`.
For any `a, b` in `Cq` and any `v` with `a -> v` and `v -> b`, the whole path
lies inside `G[U]`.

**Proof.** `v -> b` with `b` in `Cq` puts `v` in `N1_in(Cq)`, so `v` is in `U`.
`a` and `b` are in `Cq`, a subset of `U`. Both edges therefore have both
endpoints in `U` and survive vertex induction. ∎

No symmetry assumption is used, so this holds on hotpotqa exactly as on the five
reachability-symmetric graphs. Three consequences matter:

* **The seed case is a special case.** `Sq` is a subset of `Cq` (VERIFIED FROM
  CODE), so every `s -> v -> d` from a retrieval seed to a candidate is
  preserved. `TARGET_H1` restores the *complete* candidate-endpoint two-hop
  context without needing `TARGET_H2` -- which Phase -1 measured at 93.2% of the
  hotpotqa graph.
* **The path arms could not have won.** `PATH_H2` and `BRIDGE_H2` admit only
  nodes carrying a directed two-path into `Cq`, and every such node points at a
  candidate. Both are subsets of `TARGET_H1` on every graph, by the claim rather
  than by measurement. Stage B observed the nesting; this explains it.
* **It is the precise sense in which `TARGET_H1` beats `SEED_H1`.** `SEED_H1`
  preserves `s -> v -> d` for a *seed* `s`, because the bridge is an
  out-neighbour of a seed. It carries no guarantee for `c -> v -> d` between two
  ordinary candidates, and on these graphs it loses some. That is the same fact
  as the descriptor saturation in section 10, seen from the other side: the
  frozen distance-2 bucket asks only about seed two-paths, which is exactly the
  subset both arms preserve.

**Sharpness, stated rather than glossed.** The claim is about *directed*
two-paths. An undirected one has four orientations, and three put an out-edge
from the bridge into `Cq`:

```text
a -> v -> b     v -> b, b in Cq        bridge in U
a <- v -> b     v -> a, a in Cq        bridge in U
a <- v <- b     v -> a, a in Cq        bridge in U
a -> v <- b     no out-edge into Cq    bridge in U only if v -> Cq anyway
```

So the one pattern `TARGET_H1` can lose is the common successor `a -> v <- b`: a
node two distinct candidates both point at, which points at no candidate itself.
On a graph whose stored edges all carry their reverse it cannot exist, and
`Cq u N1(Cq)` preserves every undirected two-path between candidates. On
hotpotqa it can, and `two_path_preservation` counts it rather than assuming it
away.

Pinned in [`tests/test_two_path_preservation.py`](../tests/test_two_path_preservation.py)
against brute-force enumeration of actual paths -- an identity verified with the
same expression that computes it is verified by nothing -- on directed and
symmetric random graphs, including the non-degeneracy checks that the
common-successor exception and the `SEED_H1` gap are both reachable rather than
theoretical.

## 9. Cost

MEASURED on the Modal image, milliseconds per query, validation split, 300
queries. `total` is context construction plus the frozen QLS-v1 descriptor
kernel; both are paid by any system using the context. The seed-distance
diagnostic is excluded -- it is a measurement instrument, never paid at serving
time.

| dataset | `TARGET_H1` build p95 | features p95 | total p95 | total p99 |
|---|---:|---:|---:|---:|
| 2wiki_clean | 1.9 | 6.9 | 8.7 | 9.8 |
| musique_clean | 0.6 | 12.7 | 13.2 | 14.9 |
| metaqa | 1.7 | 27.6 | 29.3 | 32.5 |
| webqsp | 21.9 | 41.8 | 62.8 | 89.4 |
| squad_clean | 5.3 | 155.2 | 160.1 | 169.4 |
| hotpotqa_clean | 31.7 | 143.5 | 170.1 | 205.9 |

Context construction is the cheap half everywhere. The descriptor kernel
dominates, and on squad it is 155 ms for an 8,378-node p95 context -- the cost
scales with context nodes, not with corpus size.

**This is the cost of one implementation, not of the regime.** Every number in
the table comes from materialising `G[Cq u N1_in(Cq)]` explicitly and running
the frozen kernel over it. The statistics that make the regime what it is --
seed support, distance at most 2, bridge and predecessor counts -- are
determined by the global CSR and the frozen pool, not by the subgraph object,
and section 8's theorem is what makes that concrete: every `s -> v -> d` has its
bridge in `N1_in(Cq)`, so the same counts are obtainable by marking
seed-neighbour membership and scanning candidate adjacency rows without ever
building a thousands-node subgraph. Whether an exact implicit form matches the
explicit one bit for bit is an open engineering question, deliberately not
pursued here: the 170 ms tail is not worth optimising until the information is
known to be worth having.

Whole pilot: 8 CPU containers, no GPU, 1,232 container-seconds = 0.34 CPU-hours,
about $0.22 at the $0.634/h substrate shape, against declared ceilings of
2.5 + 3.25 CPU-hours and $2.00 + $5.15. Two further containers were spent for
two seconds each on an image that shipped without `torch-geometric`; a test now
asserts every registered image carries the same pin.

## 10. A measurement was re-specified before it was taken

The filed declaration made "feature movement vs QLS-v1 on `G[Cq]`" the primary
signal, and its stopping rule killed an arm whose features did not move. Both
are vacuous above `SEED_H1`, provably and on any graph.

Columns 0-3 of the frozen QLS-v1 descriptor are one-hot over
`{0, 1, 2, >=3-or-unreachable}`. Reaching bucket 2 requires a path
`seed -> x -> candidate` with a single intermediate, and every such `x` is an
out-neighbour of a seed -- which is exactly what `SEED_H1` admits. The raw
seed-incidence column is pinned for the same reason: `Sq` is inside `Cq`, so
every seed-candidate edge is already in `G[Cq]`. The remaining six columns are
each divided by a per-query maximum over the whole local node space, so they move
under a wider context whether or not a candidate's own topology did.

Verified over 480 arm comparisons on 120 random graphs, symmetric and directed:
zero bucket movements and zero raw-count changes above `SEED_H1`, with the
normalised seed-incidence column differing on 17 of them -- the rescaling
confound, and nothing else.

Confirmed at scale in both stages. `reach<=1` is identical across all six arms
including `CAND`; `reach<=2` is identical across all five non-`CAND` arms. The
arms separate only at distance 3 and beyond, which is the range the descriptor
collapses into one class.

Both descriptor measures were computed anyway, and Stage B shows exactly what
the withdrawal avoided. `distance_bucket_changed` is **identical across all five
non-`CAND` arms** -- 0.6674 on 2wiki and 0.6559 on hotpotqa, to four decimal
places, for `SEED_H1`, `TARGET_H1`, `SEED_H2`, `PATH_H2` and `BRIDGE_H2` alike.
Had the declared measure not been withdrawn it would have ranked nothing, and
its stopping rule -- kill an arm whose features do not move -- would have killed
nothing either. `any_column_changed` does separate the arms (0.8499 to 0.9970 on
2wiki), and Stage A establishes that what separates them there is the per-query
rescaling rather than any candidate's own topology.

The replacement is seed distance recomputed through each arm's own context,
uncapped to four hops and divided by nothing, with `distance_improved` as
primary. Both descriptor measures are still computed and still reported.

## 11. What this does not establish

* **No effectiveness claim.** Nothing was trained. Retention, reach and latency
  are structural and systems measurements. Whether a restored context improves
  ranking is the next question and is not answered here.
* **No QLS-v2 versus GNN comparison.** The fairness contract governing it is
  filed separately and requires both sides to receive the same context. A
  QLS-v2 result under `TARGET_H1` against a GNN on `G[Cq]` would measure the
  substrate, not the method.
* **No claim that a cited paper defines this procedure.** Enclosing subgraphs,
  seed-anchored expansions and radius-bounded contexts are all standard;
  `Cq u N1_in(Cq)` over a frozen candidate pool with a fixed scoring universe is
  ours and is stated in full.
* **Validation only.** The test split was not read.
* **Three arms were judged on two datasets.** `PATH_H2`, `BRIDGE_H2` and
  `SEED_H2` were killed on Stage B's 2wiki and hotpotqa and were never run on
  the other four. The kills for `PATH_H2` and `BRIDGE_H2` rest on subset
  relations and hold everywhere; `SEED_H2`'s rests on a measurement of two
  datasets, and reviving it would mean running the remaining four.
* **`TARGET_H1`'s tail is not solved.** hotpotqa's 142,073-node maximum is
  reported, not handled. Any cap must be defined from the Stage-C size
  distributions and systems constraints, never tuned against ranking outcomes.
* **`TARGET_H1` is not frozen as the QLS-v2 backend.** It is a graph-information
  privilege, and the explicit subgraph is one way to compute statistics from
  that privilege. Section 9 says why the two should not be equated.
* **The fair GNN protocol is not settled, and cannot be depth-blind.** A
  one-layer GNN's natural receptive field matches `TARGET_H1`; a two- or
  three-layer one does not, and would need `TARGET_H2` or standard neighbour
  sampling. Declaring `TARGET_H1` the context for all depths would cripple the
  deeper baselines, which the fairness contract forbids.

### What has to be earned next

> Restoring the graph information that candidate induction removed materially
> improves retrieval ranking.

That sentence is unproven. Stage D0 asks it without training anything -- do the
structural quantities computed under `TARGET_H1` separate relevant candidates
from irrelevant ones better than the same quantities under `CAND`, on the
Stage-C sample, using validation labels for scoring only. If they do not, and in
particular if they do not on the starved strata where the repair lands, the
right result is that historical induction removed a great deal of graph and
restoring it in this form did not help ranking. No GPU is justified before that
question is answered.

## 12. Full tables

Rendered by `scripts/render_graph_context_tables.py` from the Stage C result
files. Regenerate rather than retype:

```bash
PYTHONPATH="src;." python scripts/render_graph_context_tables.py --root outputs/graph_context_pilot/stage_c --split validation
```

Split: `validation`. Datasets: `2wiki_clean`, `hotpotqa_clean`, `metaqa`, `musique_clean`, `squad_clean`, `webqsp`.

Queries measured, and how many were skipped for carrying no retrieval seed:

| dataset | requested | measured | skipped, no seeds |
|:-------|---------:|--------:|-----------------:|
| 2wiki_clean | 300 | 300 | 0 |
| hotpotqa_clean | 300 | 300 | 0 |
| metaqa | 300 | 300 | 0 |
| musique_clean | 300 | 300 | 0 |
| squad_clean | 300 | 300 | 0 |
| webqsp | 300 | 300 | 0 |

### 2wiki_clean

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 364 | 387 | 0.00 | 0.0055 | 667 |
| SEED_H1 | 398 | 438 | 0.09 | 0.0060 | 2,225 |
| TARGET_H1 | 2,060 | 2,616 | 4.70 | 0.0313 | 22,077 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.1200 | 0.1691 | 0.8425 | 0.3682 |
| SEED_H1 | 0.2959 | 0.3319 | 0.6924 | 0.1067 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.0599 | 0.1026 | 0.1434 | 0.1791 | 0.8209 |
| SEED_H1 | 0.7370 | 0.0599 | 0.7657 | 0.8377 | 0.8531 | 0.1469 |
| TARGET_H1 | 0.8778 | 0.0599 | 0.7657 | 0.9737 | 0.9922 | 0.0078 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.6996 | 0.8172 | 0.7366 | 0.5227 |
| TARGET_H1 | 0.9753 | 0.9202 | 0.7696 | 0.5315 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.0 | 0.0 | 0.0 | 3.7 | 3.7 |
| SEED_H1 | 1.3 | 1.6 | 1.8 | 3.6 | 5.2 |
| TARGET_H1 | 1.5 | 1.9 | 2.2 | 6.9 | 8.7 |

### hotpotqa_clean

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 346 | 385 | 0.00 | 0.0007 | 860 |
| SEED_H1 | 408 | 533 | 0.17 | 0.0008 | 277,558 |
| TARGET_H1 | 2,890 | 7,509 | 7.18 | 0.0057 | 535,282 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.0908 | 0.1299 | 0.8893 | 0.3391 |
| SEED_H1 | 0.3086 | 0.3460 | 0.7041 | 0.0410 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0015 | 0.0000 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.1031 | 0.2609 | 0.3463 | 0.4000 | 0.6000 |
| SEED_H1 | 0.6711 | 0.1031 | 0.8970 | 0.9380 | 0.9431 | 0.0569 |
| TARGET_H1 | 0.7289 | 0.1031 | 0.8970 | 0.9986 | 0.9999 | 0.0001 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.8462 | 0.7861 | 0.5545 | 0.2378 |
| TARGET_H1 | 0.9941 | 0.8108 | 0.5594 | 0.2390 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.1 | 0.3 | 0.3 | 65.5 | 65.6 |
| SEED_H1 | 24.8 | 35.3 | 39.5 | 86.6 | 116.4 |
| TARGET_H1 | 23.6 | 31.7 | 34.6 | 143.5 | 170.1 |

### metaqa

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 364 | 391 | 0.00 | 0.0091 | 1,319 |
| SEED_H1 | 398 | 1,304 | 0.09 | 0.0099 | 1,833 |
| TARGET_H1 | 7,520 | 12,039 | 19.77 | 0.1873 | 99,531 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.1591 | 0.2118 | 0.9173 | 0.2459 |
| SEED_H1 | 0.2032 | 0.2559 | 0.8782 | 0.1873 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.0938 | 0.3459 | 0.4886 | 0.5626 | 0.4374 |
| SEED_H1 | 0.2171 | 0.0938 | 0.4818 | 0.6615 | 0.7132 | 0.2868 |
| TARGET_H1 | 0.5064 | 0.0938 | 0.4818 | 0.8747 | 0.9818 | 0.0182 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.2366 | 0.2556 | 0.1812 | 0.1232 |
| TARGET_H1 | 0.9512 | 0.4862 | 0.2748 | 0.1469 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.0 | 0.0 | 0.0 | 7.0 | 7.0 |
| SEED_H1 | 1.0 | 1.4 | 1.7 | 7.7 | 8.9 |
| TARGET_H1 | 1.4 | 1.7 | 1.9 | 27.6 | 29.3 |

### musique_clean

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 336 | 373 | 0.00 | 0.0246 | 1,371 |
| SEED_H1 | 386 | 526 | 0.13 | 0.0283 | 3,763 |
| TARGET_H1 | 2,248 | 3,355 | 5.67 | 0.1644 | 72,109 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.2195 | 0.2880 | 0.7985 | 0.2170 |
| SEED_H1 | 0.2872 | 0.3600 | 0.7112 | 0.1833 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.1264 | 0.2659 | 0.3865 | 0.4702 | 0.5298 |
| SEED_H1 | 0.2332 | 0.1264 | 0.4011 | 0.5556 | 0.6292 | 0.3708 |
| TARGET_H1 | 0.5705 | 0.1264 | 0.4011 | 0.7742 | 0.9435 | 0.0565 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.1488 | 0.3028 | 0.2890 | 0.1390 |
| TARGET_H1 | 0.8731 | 0.7327 | 0.4946 | 0.1887 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.0 | 0.0 | 0.0 | 3.6 | 3.6 |
| SEED_H1 | 0.4 | 0.5 | 0.7 | 2.9 | 3.4 |
| TARGET_H1 | 0.4 | 0.6 | 0.7 | 12.7 | 13.2 |

### squad_clean

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 316 | 367 | 0.00 | 0.0166 | 3,567 |
| SEED_H1 | 506 | 1,125 | 0.56 | 0.0266 | 38,192 |
| TARGET_H1 | 5,752 | 8,378 | 17.32 | 0.3023 | 1,230,989 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.1096 | 0.2457 | 0.8811 | 0.1711 |
| SEED_H1 | 0.2660 | 0.3875 | 0.7025 | 0.1424 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.1983 | 0.3644 | 0.4918 | 0.5718 | 0.4282 |
| SEED_H1 | 0.2188 | 0.1983 | 0.4869 | 0.6466 | 0.7146 | 0.2854 |
| TARGET_H1 | 0.4956 | 0.1983 | 0.4869 | 0.8414 | 0.9708 | 0.0292 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.1664 | 0.3013 | 0.2966 | 0.1447 |
| TARGET_H1 | 0.9241 | 0.7522 | 0.4996 | 0.1778 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.0 | 0.0 | 0.0 | 13.2 | 13.3 |
| SEED_H1 | 2.8 | 3.9 | 4.2 | 22.4 | 26.1 |
| TARGET_H1 | 4.6 | 5.3 | 5.5 | 155.2 | 160.1 |

### webqsp

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 350 | 392 | 0.00 | 0.0004 | 2,164 |
| SEED_H1 | 388 | 522 | 0.09 | 0.0005 | 3,093 |
| TARGET_H1 | 2,012 | 4,764 | 4.81 | 0.0026 | 28,488 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.3170 | 0.3570 | 0.7000 | 0.1909 |
| SEED_H1 | 0.3822 | 0.4170 | 0.6370 | 0.1451 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.1562 | 0.4530 | 0.5387 | 0.5656 | 0.4344 |
| SEED_H1 | 0.1575 | 0.1562 | 0.5734 | 0.6717 | 0.6963 | 0.3037 |
| TARGET_H1 | 0.4368 | 0.1562 | 0.5734 | 0.8445 | 0.9672 | 0.0328 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.2389 | 0.1546 | 0.1605 | 0.0787 |
| TARGET_H1 | 0.9048 | 0.4266 | 0.3615 | 0.1403 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.0 | 0.2 | 0.3 | 34.9 | 34.9 |
| SEED_H1 | 18.6 | 22.0 | 24.2 | 34.3 | 55.0 |
| TARGET_H1 | 18.2 | 21.9 | 25.7 | 41.8 | 62.8 |

### Frontier

Pareto domination on (`build` p95 latency, `retention_median`). An arm is killed only where every measured dataset agrees it is dominated.

| arm | 2wiki_clean | hotpotqa_clean | metaqa | musique_clean | squad_clean | webqsp | verdict |
|:---|:-----------|:--------------|:------|:-------------|:-----------|:------|:-------|
| CAND | on the frontier | on the frontier | on the frontier | on the frontier | on the frontier | on the frontier | survives |
| SEED_H1 | on the frontier | TARGET_H1 | on the frontier | on the frontier | on the frontier | TARGET_H1 | survives |
| TARGET_H1 | on the frontier | on the frontier | on the frontier | on the frontier | on the frontier | on the frontier | survives |


## 13. Stage B, six arms, 25 queries

The run the kills were made on. Retained because three arms exist nowhere else.

Split: `validation`. Datasets: `2wiki_clean`, `hotpotqa_clean`.

Queries measured, and how many were skipped for carrying no retrieval seed:

| dataset | requested | measured | skipped, no seeds |
|:-------|---------:|--------:|-----------------:|
| 2wiki_clean | 25 | 25 | 0 |
| hotpotqa_clean | 25 | 25 | 0 |

### 2wiki_clean

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 361 | 381 | 0.00 | 0.0055 | 552 |
| SEED_H1 | 403 | 453 | 0.10 | 0.0061 | 2,312 |
| TARGET_H1 | 2,039 | 2,612 | 4.62 | 0.0310 | 21,182 |
| SEED_H2 | 39,647 | 40,463 | 109.04 | 0.6019 | 499,080 |
| PATH_H2 | 380 | 401 | 0.05 | 0.0058 | 2,114 |
| BRIDGE_H2 | 669 | 1,107 | 0.82 | 0.0102 | 6,096 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.1003 | 0.1561 | 0.8521 | 0.3896 |
| SEED_H1 | 0.2941 | 0.3290 | 0.6888 | 0.1148 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| SEED_H2 | 0.8107 | 0.7693 | 0.2218 | 0.0123 |
| PATH_H2 | 0.2938 | 0.3225 | 0.6969 | 0.1158 |
| BRIDGE_H2 | 0.5939 | 0.5875 | 0.3892 | 0.0507 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.0580 | 0.0956 | 0.1306 | 0.1583 | 0.8417 |
| SEED_H1 | 0.7429 | 0.0580 | 0.7630 | 0.8299 | 0.8447 | 0.1553 |
| TARGET_H1 | 0.8918 | 0.0580 | 0.7630 | 0.9727 | 0.9932 | 0.0068 |
| SEED_H2 | 0.8826 | 0.0580 | 0.7630 | 0.9727 | 0.9840 | 0.0160 |
| PATH_H2 | 0.7429 | 0.0580 | 0.7630 | 0.8299 | 0.8447 | 0.1553 |
| BRIDGE_H2 | 0.8309 | 0.0580 | 0.7630 | 0.8934 | 0.9325 | 0.0675 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.6938 | 0.8109 | 0.7702 | 0.4842 |
| TARGET_H1 | 0.9765 | 0.9150 | 0.7996 | 0.4912 |
| SEED_H2 | 0.9578 | 0.9109 | 0.7976 | 0.4912 |
| PATH_H2 | 0.6938 | 0.8109 | 0.7702 | 0.4842 |
| BRIDGE_H2 | 0.8418 | 0.8881 | 0.7964 | 0.4912 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.0 | 0.0 | 0.0 | 5.3 | 5.3 |
| SEED_H1 | 1.3 | 1.4 | 1.5 | 3.5 | 4.8 |
| TARGET_H1 | 1.8 | 1.9 | 2.0 | 6.0 | 7.9 |
| SEED_H2 | 3.4 | 3.5 | 3.6 | 77.8 | 81.2 |
| PATH_H2 | 5.3 | 8.7 | 8.9 | 4.3 | 12.7 |
| BRIDGE_H2 | 4.5 | 5.1 | 5.2 | 4.5 | 9.6 |

### hotpotqa_clean

**Size.**

| arm | context nodes p50 | p95 | added ratio p50 | share of graph p50 | context edges p50 |
|:---|-----------------:|---:|---------------:|------------------:|-----------------:|
| CAND | 350 | 380 | 0.00 | 0.0007 | 824 |
| SEED_H1 | 411 | 514 | 0.17 | 0.0008 | 274,413 |
| TARGET_H1 | 3,170 | 10,058 | 7.65 | 0.0062 | 549,852 |
| SEED_H2 | 268,363 | 333,262 | 780.43 | 0.5288 | 9,215,387 |
| PATH_H2 | 395 | 450 | 0.09 | 0.0008 | 274,267 |
| BRIDGE_H2 | 865 | 1,594 | 1.45 | 0.0017 | 434,957 |

**Structural recovery**, on Phase -1's own definitions.

| arm | retention p50 | retention mean | boundary cut | isolated fraction |
|:---|-------------:|--------------:|------------:|-----------------:|
| CAND | 0.0888 | 0.1320 | 0.8995 | 0.3413 |
| SEED_H1 | 0.3104 | 0.3450 | 0.7233 | 0.0436 |
| TARGET_H1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 |
| SEED_H2 | 0.9496 | 0.9083 | 0.1161 | 0.0005 |
| PATH_H2 | 0.3104 | 0.3412 | 0.7335 | 0.0436 |
| BRIDGE_H2 | 0.7079 | 0.6850 | 0.4347 | 0.0035 |

**Seed reach.** Recomputed through each arm's own context, uncapped to 4 hops and divided by nothing.

| arm | closer than G[Cq] | reach<=1 | reach<=2 | reach<=3 | reach<=4 | unreached at 4 |
|:---|-----------------:|--------:|--------:|--------:|--------:|--------------:|
| CAND | 0.0000 | 0.1008 | 0.2481 | 0.3518 | 0.4164 | 0.5836 |
| SEED_H1 | 0.6817 | 0.1008 | 0.9040 | 0.9388 | 0.9418 | 0.0582 |
| TARGET_H1 | 0.7409 | 0.1008 | 0.9040 | 0.9992 | 1.0000 | 0.0000 |
| SEED_H2 | 0.7403 | 0.1008 | 0.9040 | 0.9992 | 0.9993 | 0.0007 |
| PATH_H2 | 0.6817 | 0.1008 | 0.9040 | 0.9388 | 0.9418 | 0.0582 |
| BRIDGE_H2 | 0.7366 | 0.1008 | 0.9040 | 0.9930 | 0.9957 | 0.0043 |

**Where the improvement lands**, by the induced degree the candidate had in `G[Cq]` before any context was restored.

| arm | isolated | degree_1 | low_degree | ordinary |
|:---|--------:|--------:|----------:|--------:|
| CAND | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| SEED_H1 | 0.8462 | 0.7626 | 0.5769 | 0.2921 |
| TARGET_H1 | 0.9970 | 0.7865 | 0.5809 | 0.2921 |
| SEED_H2 | 0.9954 | 0.7865 | 0.5809 | 0.2921 |
| PATH_H2 | 0.8462 | 0.7626 | 0.5769 | 0.2921 |
| BRIDGE_H2 | 0.9847 | 0.7865 | 0.5809 | 0.2921 |

**Latency, milliseconds per query per arm.**

| arm | build p50 | build p95 | build p99 | features p95 | total p95 |
|:---|---------:|---------:|---------:|------------:|---------:|
| CAND | 0.1 | 0.1 | 0.1 | 84.1 | 84.2 |
| SEED_H1 | 61.8 | 73.5 | 77.4 | 103.9 | 173.5 |
| TARGET_H1 | 62.1 | 74.0 | 74.9 | 157.2 | 221.8 |
| SEED_H2 | 128.3 | 136.4 | 147.3 | 2,504.3 | 2,635.6 |
| PATH_H2 | 189.7 | 204.4 | 208.1 | 104.7 | 302.2 |
| BRIDGE_H2 | 178.3 | 199.5 | 203.3 | 112.6 | 306.8 |

### Frontier

Pareto domination on (`build` p95 latency, `retention_median`). An arm is killed only where every measured dataset agrees it is dominated.

| arm | 2wiki_clean | hotpotqa_clean | verdict |
|:---|:-----------|:--------------|:-------|
| CAND | on the frontier | on the frontier | survives |
| SEED_H1 | on the frontier | on the frontier | survives |
| TARGET_H1 | on the frontier | on the frontier | survives |
| SEED_H2 | TARGET_H1 | TARGET_H1 | killed |
| PATH_H2 | BRIDGE_H2, SEED_H1, SEED_H2, TARGET_H1 | BRIDGE_H2, SEED_H1, SEED_H2, TARGET_H1 | killed |
| BRIDGE_H2 | SEED_H2, TARGET_H1 | SEED_H2, TARGET_H1 | killed |

