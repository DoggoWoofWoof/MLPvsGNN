# The frozen feature catalog: nine families, and what each one is evidence for

**Status:** `FROZEN_AT_D10`
**Frozen by:** stage D10, the last 2Wiki-only feature-development stage
**Machine-readable source:** `feature_catalog` in
[`configs/graph_context_pilot.yaml`](../configs/graph_context_pilot.yaml)

This is a per-family record, **not a ranking** and not a model. No family below
is named best. Two of the nine -- PROVENANCE and NODE ROLE -- have no fitted arm
at all. Two more were classified `NEGLIGIBLE` on one historical column each and
never revisited. Of the two whose corrected replacement was actually built and
fitted, one matched the historical proxy and one came out mixed against it.
Neither gained.

Every number here is 2Wiki, seed 0, `CAND`, `G[Cq]`, validation, one
architecture at 213,689 parameters. It is development evidence, not an
evaluation.

## The subset the ladder actually selected

GEOMETRY plus RETRIEVAL: local columns 0-3 and 10-12, **seven of thirteen**.
Columns 4 through 9 are exactly zero in `D6_BASE_13` and in every base arm
after it. Every family tested after D6 was added to that base one column group
at a time, and **none of them was admitted into it**.

That is a 2Wiki result. It is **not declared universal**, and the zero-training
dataset x regime matrix proposed after D10 exists precisely because whether it
transfers is unmeasured.

## Catalogued membership

What each family is understood to contain. This is deliberately wider than the
columns the frozen 13-dimension block carries: a member named here has been
catalogued, **not tested and not admitted**.

```text
RETRIEVAL     dense RR; SPLADE RR; dense/SPLADE agreement
SEED          seed identity
GEOMETRY      seed-relative distance
SUPPORT       historical edge-count support; distinct-support replacement
PATH          historical directed walks; branch-diversity replacement
DIFFUSION     historical personalized PPR; bounded replacement if later justified
TOPOLOGY      local-neighbourhood statistic
PROVENANCE    native evidence; kNN evidence
NODE ROLE     scored candidate; context-only node
```

**PROVENANCE and NODE ROLE are catalogued hypotheses, not tested winners.**
Neither has a fitted arm at this architecture, and no paper text may present
either as having been tested and won.

The provisional strong subset -- 2Wiki, `CAND`, seed 0 -- is only:

```text
RETRIEVAL + SEED + GEOMETRY
```

and it is not universal.

## The nine families

| family | cols | status on 2Wiki | evidence |
| --- | --- | --- | --- |
| RETRIEVAL | 10-12 | `ADMITTED` | D4, D5 |
| SEED | 0 | `ADMITTED` | D3 |
| GEOMETRY | 0-3 | `ADMITTED` | D3, D5, D6 |
| SUPPORT | 4 | `CARRIED_NOT_ON_ACCURACY` | D7, D8 |
| PATH | 5-7 | `PROMISING_REPLACEMENT_FAILED` | D7, D9, D10 |
| DIFFUSION | 8 | `NEGLIGIBLE` | D7 |
| TOPOLOGY | 9 | `NEGLIGIBLE` | D7 |
| PROVENANCE | -- | `MEASURED_AS_A_GRAPH_AXIS` | edge-provenance results |
| NODE ROLE | -- | `NOT_TESTED` | D0b, for its one probe |

### RETRIEVAL -- `ADMITTED`

Graded retrieval prior over dense and SPLADE ranks, columns 10-12.

D4 `delta_retrieval_quality` +9.18 / +4.13 / +1.55 / +13.03 / +2.43.
D5 `delta_prior_given_geometry` +9.93 / +4.68 / +0.45 / +14.10 / +0.60.
As R@1 / R@5 / R@20 / MRR / FullCov@20.

The largest single increment anywhere in the D-ladder, positive on every metric
in both stages. A later stage may keep it in any base arm. It does not
authorise the claim that retrieval rank dominates on any other dataset, or that
this particular grading is the best encoding of rank.

### SEED -- `ADMITTED`

Column 0, `distance_0`, which is exactly the indicator that a candidate is a
retrieval seed.

D3 `delta_seed` +18.20 R@5 over `ZERO_LOCAL` -- 87.4% of the R@5 gap, 67.0% of
R@20, 59.5% of FullCov@20. The single largest jump in the D3 ladder, from one
bit per candidate.

It is not a separate column: it is the zero bucket of the GEOMETRY one-hot, so
admitting GEOMETRY admits it. It does not authorise counting SEED and GEOMETRY
as two independent families when summing contributions.

### GEOMETRY -- `ADMITTED`

Seed-distance one-hot, columns 0-3: buckets 0, 1, 2 and 3-plus-or-unreachable.

D3 `delta_distance` +1.49 R@5 over `SEED_ID_ONLY`.
D5 `delta_geometry_given_prior` +0.23 / +2.05 / +4.03 / +0.42 / +6.93.

Positive on every metric conditional on the retrieval prior, and the block
every later base arm is built from. D3 verified the group is a complete
one-hot, so dropping a bucket changes what is encoded. It does not authorise a
finer distance encoding, a weighted distance, or a hop bound other than the
frozen three.

### SUPPORT -- `CARRIED_NOT_ON_ACCURACY`

Column 4. Historical `seed_connections`; the corrected form is distinct
supporting seeds over bounded bitsets.

D7 `delta_support` -0.03 / +0.64 / +0.20 / +0.09 / +0.40 against `D6_BASE_13`.
D8 `V - H` -0.07 / -0.43 / -0.04 / -0.12 / -0.07, every metric inside the filed
+/-0.50 material band.

The corrected representation matched the historical proxy and **beat it on
nothing**. It is carried on grounds filed in advance -- bounded [0,1] by
construction, no per-query normaliser, 1.66 s and one fixed graph pass -- and
explicitly not on accuracy. A later stage may use distinct supporting seeds in
place of the historical edge count and say the exchange was Pareto-neutral on
2Wiki. It does not authorise the claim that the corrected form is more
accurate, or a weighted support variant, which was named and not tested.

### PATH -- `PROMISING_REPLACEMENT_FAILED`

Columns 5-7. Historical `paths_length_1/2/3`; the corrected form is bounded
branch diversity per hop.

D7 `delta_paths` -0.12 / +0.82 / +0.13 / -0.15 / +0.30 against `D6_BASE_13`.
D10 `V - H` +0.48 / -0.67 / +0.04 / +0.62 / +0.03.

The historical form held the largest R@5 gain of the four D7 families. The
replacement is mixed against it: materially better on MRR, materially worse on
R@5, at the filed 0.005 threshold. The two forms move different metrics in
opposite directions from the same base, so neither subsumes the other.

The D10 classification is contested between two preregistered statements of the
bands; see the D10 stage block and
[`GRAPH_CONTEXT_D10_RESULTS.md`](GRAPH_CONTEXT_D10_RESULTS.md). It does not
authorise adding walk counts back as established useful, attributing either
direction to walk multiplicity alone, or a weighted, disjoint or motif path
variant -- D10 changed the counting rule, the numerical transform and the
calibration together and has no arm that separates them.

### DIFFUSION -- `NEGLIGIBLE`

Column 8, historical `personalized_pagerank`.

D7 `delta_diffusion` -0.13 / +0.43 / +0.17 / -0.06 / +0.33 against
`D6_BASE_13`.

One fitted one-family arm, classified negligible against the filed band. The
corrected candidate D7 named -- bounded H1, H2, H3 and truncated PPR at
deterministic work -- was never built or fitted. It does not authorise the
claim that diffusion carries no signal: what was measured is one historical
encoding of it, once.

### TOPOLOGY -- `NEGLIGIBLE`

Column 9, historical `common_out_neighbors_with_seed_neighborhood`.

D7 `delta_neighbourhood` -0.22 / +0.07 / +0.08 / -0.19 / +0.17 against
`D6_BASE_13`.

The weakest of the four D7 families, negative on R@1 and MRR. No corrected
bounded topology statistic was built or fitted. It does not authorise the claim
that cheap candidate topology is worthless, which D7 was explicit is not what a
one-column arm tests.

### PROVENANCE -- `MEASURED_AS_A_GRAPH_AXIS`

No columns. The family is the edge-construction axis: which edges exist, from
which source. Five edge families were compared on three datasets in
[`EDGE_PROVENANCE_RESULTS.md`](EDGE_PROVENANCE_RESULTS.md), with QLS-MLP and a
learned reference measured on each.

Real measurement, different object. It says which graph to build, not which
per-candidate feature to score with, and no column of the thirteen belongs to
it. A later stage may choose an edge family using that evidence and must keep
it separate from the feature question when summarising. It does not authorise
treating an edge-family result as a feature-family result, or reading a
QLS-versus-GNN gap in that table as evidence about which QLS features to admit.
GNN outcomes are not permitted to select QLS features.

### NODE ROLE -- `NOT_TESTED`

No columns and no members. The only member ever probed is `bridge_support`,
which is not in the frozen block.

At D0b, added to an arm already holding the ten frozen local columns over
`TARGET_H1`, `bridge_support` moved validation R@1 by +0.53 and R@20 by -0.10,
and changed R@5 and R@20 by exactly zero on the isolated stratum it was built
for. KILLED. Its D0 univariate AUC of 0.9513 did not survive as incremental
ranking value.

The family itself has no fitted arm at this architecture. It is listed so that
its absence is explicit rather than accidental. It does not authorise
reintroducing bridge_support without a new declaration, or claiming node role
was tested and failed -- it was not tested.

## What the pattern across the nine is, and is not

Three families have no replacement evidence at all. DIFFUSION and TOPOLOGY were
each fitted once in their historical form and classified negligible, and the
corrected candidates D7 named for both were never built. NODE ROLE has no
member.

Of the two families that do have replacement evidence, SUPPORT produced a match
and PATH produced a mixed result. **Neither produced a gain.**

Stated as inference, not measurement: two corrected structural representations
were built carefully, and neither beat the historical proxy it replaced, on the
dataset where the historical proxies were tuned. That is evidence about how
much further per-candidate structural engineering is worth on 2Wiki. It is
**not** evidence that structure is exhausted, and it says nothing about
datasets where the frozen block was never fitted.

## What this catalog does not settle

It is one dataset and one seed. It does not rank the families against each
other, because the increments were measured against different bases at
different stages and are not commensurable: D3's are over `ZERO_LOCAL`, D4 and
D5's are conditional increments in a two-way decomposition, and D7's, D8's and
D10's are one column group added to `D6_BASE_13`. Reading a ranking out of the
table above would be reading something that was not measured.

Nor does it fix a model. The subset the ladder selected is the base arm the
stages happened to build on, and admission was never the question any single
stage asked.
