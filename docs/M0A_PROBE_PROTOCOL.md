# M0A: a contract probe for three candidate/context regimes

This is the protocol for `configs/m0a_probe.yaml`. It is filed before any M0A
number exists. Nothing here is a result, and no downstream stage may cite M0A
as evidence that expansion helps or does not help.

## What the probe is for

Package D ended with a feature catalogue and a negative replacement result. The
open question it could not touch is whether the pool being scored is the thing
holding the metric down. That question needs three regimes, and before any of
them can be compared it needs to be shown that they are implementable,
inference-safe, budget-controlled and affordable.

M0A answers only that. One hundred validation queries on three datasets cannot
support an effectiveness claim, and this stage does not make one. Validating
cheaply is the point: a matrix multiplied out over six datasets before the
contract holds would be an expensive way to discover a leak.

## The three regimes

| | scored set | graph context | may change the candidate oracle |
|---|---|---|---|
| **R1** HISTORICAL | `Cq` | `G[Cq]` | no |
| **R2** REPAIRED_GRAPH_CONTEXT | `Cq` | `TARGET_H1(Cq) = Cq union N1(Cq)` | no |
| **R3** EXPANDED_CANDIDATES | `Cq'` | `TARGET_H1(Cq')` | yes |

R2 changes what a candidate can see. It does not change what is scored, so its
candidate ceiling is mathematically identical to R1's. That identity is
asserted per cell rather than assumed, because the entire reading of R2 depends
on it: if R2 moves, the movement came from context, and a drifting ceiling
would mean the candidate set had drifted too.

R3 is the only regime permitted to move the candidate oracle. This is a new
declared object, `oracle(Cq')`, not a reinterpretation of the historical one.

## The historical contract is preserved, not relaxed

`configs/candidate_headroom.yaml` sets `candidate_regeneration`,
`candidate_admission` and `graph_expansion` to `prohibited_in_paper_1`. That
contract is not edited. It governs `oracle(Cq)` over a pool fixed by
construction, and that object is unchanged and unreinterpreted.

R3 needs a different namespace because it is a different object:
`src/mp_retrieval/candidate_expansion_v2.py` and
`src/mp_retrieval/headroom_v2.py`. Neither writes into a historical output
path. `headroom_v2` imports the historical `present_counts` and
`headroom_metrics` rather than restating them, and a test asserts they are the
same objects, so there is exactly one definition of AnyGold in the repository.

## What candidate expansion may read

Permitted: the frozen query embedding, the frozen node embeddings, the frozen
retrieval seeds and dense-rank-1 anchor, and graph topology.

Forbidden: a learned relational offset, a learned router, a GNN candidate
generator, neural query expansion, the dataset identifier, gold nodes, gold
relations, supporting-fact information, target-test tuning, and any fitted
parameter whatsoever.

`Cq'` is a pure function of the permitted inputs. Same inputs, same output, on
any machine, with no random state and no split-dependent branch.

## The primary method is a reconstruction, not an invention

CRAG level 1 routes with a shared MLP producing K relational offsets around the
nearest dense document. This repository's `offset_mlp` carries the same idea:
score `cos(normalize(a + g(q, a)), x)` with `a` the frozen dense rank 1 and `g`
a small learned network. Both are learned, and both are read-only references
here.

The shape is kept and the learned part is deleted:

- `g(q, a)` asks which direction to move from the anchor. The parameter-free
  residual already present in the data is `r_q = normalize(e_q - x_a)`.
- The directions available are not a learned `K`. They are the displacements
  the graph already contains, `d(u, v) = normalize(x_v - x_u)`, one per edge.
- Compatibility is `cos(r_q, d(u, v))`. No parameters, no temperature, no
  threshold fitted to data.

An edge whose displacement has no usable norm -- two nodes at the same
coordinate -- has no direction to be compatible with, so it is dropped and
counted rather than scored as zero.

**The frontier is the undirected one-hop neighbourhood of `Sq`.** All three
frozen graphs are stored asymmetric, so walking stored edges in one orientation
would be a choice between two directions that cannot be made from evidence
before the probe runs and must not be made after it. The runner symmetrises the
frozen CSR in memory, reads it, and never writes it back. Orientation costs
nothing here in any case: the score is a displacement in embedding space from
the seed to the neighbour, `normalize(x_v - x_seed)`, which does not depend on
which way the stored edge points. Whether each graph was already symmetric is
reported per dataset.

A query with no frozen retrieval seeds has no frontier and admits nothing. The
count of such queries is reported rather than dropped silently.

`cos(e_q, x_v)` is recorded per admitted node as a **diagnostic** so a reader
can see whether admitted nodes were merely dense-similar. It never enters the
admission score.

**The control**, `STRUCTURAL_NEIGHBOUR`, walks the identical frontier under the
identical caps and admits in ascending global node id with no direction scoring
at all. It exists so a movement in the directional arm reads against arbitrary
bounded expansion of the same neighbourhood, rather than against no expansion.

Exactly two methods. A sweep of parameter-free variants is not run before the
contract is shown to hold.

## The budget is fixed before the outcome

| bound | value |
|---|---|
| hop cap | 1 |
| per-seed cap | 16 |
| graph expansion cap | 128 |
| neighbour scan cap per seed | 4096 |
| tie-break | ascending global node id |

**The headline pool is matched**: `|Cq'| == |Cq| exactly, per query`. Admitted
nodes displace the tail of `Cq` in frozen candidate order -- dense first, then
unseen SPLADE -- so expansion trades rather than adds. Seeds and the anchor are
protected from eviction, because `Sq` is a subset of `Cq` by construction and
every downstream feature already assumes it; evicting a seed would break the
contract rather than test the expansion.

Under this rule the oracle can fall as well as rise. That is the property that
makes the comparison mean anything: R3 must not win merely because its
candidate set is unbounded.

The **additive** pool, `stable_union(Cq, Aq)`, is reported separately as a
diagnostic. It separates "this node is reachable and directionally compatible"
from "this node is worth a slot", which the matched number folds together. It
is never the matched-budget headline.

If the neighbour scan cap fires, the affected queries and seeds are reported. A
truncation nobody records reads afterwards as full coverage.

## Edge provenance, for expansion only

Three families are read from the Package B reconstruction in the frozen node
coordinate system: `structural_only`, `knn_only`, `baseline_a_simple`. The
question is whether recovered candidates come from real structural topology or
from embedding similarity recycled through kNN edges.

This is a diagnostic for candidate expansion and headroom. It is not multiplied
into later training experiments. R1 and R2 use the dataset default graph.

## Datasets and sampling

`squad_clean` (weak structural, single hop), `2wiki_clean` (multi-hop text),
`metaqa` (knowledge base). One hundred queries each, validation split, taken as
a deterministic prefix of the split order -- never sampled. The test split is
not read.

## What is measured

Candidate and context: candidate count median/p95/max, context node count,
induced edge count. Headroom: AnyGold, AllGold, GoldFraction, oracle recall at
1/5/20, full-coverage ceiling at 20, missing golds recovered, recovered gold
per added candidate. Systems: expansion and context-build latency p50/p95/p99
per query, peak RSS, temporary workspace bytes.

Latency is reported as per-query percentiles on a stated container, never as a
wall-clock total. D10 measured the same construction at 1.2449 of the
historical build on one host and 0.9779 on another; a totals race measures the
machine.

## Invariants asserted per cell

- the frozen candidate contract is proved bit-exact against the registered
  confirmation before anything is measured over the pool, so R1 is the
  historical object rather than a re-derivation that resembles it
- `scored_nodes == Cq` for R1 and R2, `== Cq'` for R3; `Cq` is a subset of `Uq`
- **`R1 ceiling == R2 ceiling` exactly**
- only R3 changes the candidate oracle
- the matched-budget pool size equals `|Cq|` for every query
- expansion output is byte-identical across repeats
- expansion output is byte-identical under a gold permutation
- no historical headroom output path is written

## Advancement to M0B

M0B is not automatic. Every invariant must pass, the leakage tests must pass
including the gold-permutation one, the matched-budget rule must hold for every
query in every cell, latency percentiles must be recorded per dataset, and the
scan cap must be reported wherever it fired.

Then **one** of:

- **movement** -- at least one dataset shows matched-budget AnyGold or oracle
  recall@5 moving by at least 1.0 point in at least one edge family, under
  either expansion method; or
- **decisive null** -- every dataset's matched-budget oracle recall@5 moves by
  less than 0.25 points *and* the additive diagnostic moves by less than 1.0
  point, which would say directional expansion is inert at this budget and is
  worth confirming across all six datasets precisely because it is cheap.

If the result falls between the two -- too small to act on, too large to call
inert -- M0B is **not** launched on this evidence. The budget or the frontier
is reconsidered in a new declaration. These thresholds are not moved after
seeing M0A.

## Compute

CPU only, zero GPU hours authorised, three jobs on 4 CPUs and 16 GiB, one hour
each, a cost ceiling of $0.60.

The estimate was made before the launch, on the host, against synthetic graphs
carrying each dataset's exact node and edge counts and the frozen feature
dimension of 1536. That is a container estimate from host timings, not a
measurement of the container, and it is reported as such.

| | manifest build | operators | expansion | context | job |
|---|---|---|---|---|---|
| squad_clean | 50 s | 11 s | 114 ms/q | 20 ms/q | ~170 s |
| 2wiki_clean | 6 s | 2 s | 6 ms/q | 12 ms/q | ~30 s |
| metaqa | 156 s | 2 s | 9 ms/q | 6 ms/q | ~180 s |

Each dataset runs 12 R3 cells and 600 expansions, for 380 seconds of work in
total. Priced by the project's own `compute_budget` rates -- $0.3172 an hour
for this shape, at the standing assumption that work is 40% of billed time --
that is about $0.08, or $0.25 if the estimate is out by a factor of three.
Both are inside the ceiling.

The 8 CPU / 32 GiB shape was inherited from the full-dataset headroom job and
is not what this needs. M0A memory-maps every array and scores 100 queries, so
peak RSS is dominated by the retained per-query objects -- 3 to 4 GiB on
metaqa, the largest manifest -- and by the traversal operators, a few hundred
MiB. The work is memory-bound gathers rather than matrix products, so four
cores lose nothing.

A dataset is aborted before reporting if its expansion p95 per query exceeds
its context-build p95 per query by more than a factor of four on the same
container, or if the temporary workspace exceeds 1 MiB per query.

## What M0A does not authorise

M0B, any neural training, any GNN, resuming E2, opening F, a workspace
migration, a test-split read, modification of any frozen pool or hash, a sweep
of further expansion variants, or citing M0A as evidence that expansion helps
or does not help. Regardless of M0A's outcome.
