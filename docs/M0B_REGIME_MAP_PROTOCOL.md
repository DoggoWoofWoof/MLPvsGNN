# M0B — six-dataset zero-training regime map

Declaration: [`configs/m0b_regime_map.yaml`](../configs/m0b_regime_map.yaml).
Filed before any M0B number exists.

A zero-training map across all six sealed datasets. It trains nothing, fits
nothing, reads no test split, and authorises only itself — step 1 of an
eight-step plan ending in a hard stop before any fitting.

## 1. The question

M0A measured that bounded structural expansion recovers missing gold
candidates. M0A.1 measured where those recovered golds actually came from:
`BEYOND_U2_RECOVERED == 0` on every cell tested — every gold R3 recovered was
already sitting inside R2's `TARGET_H1(Cq)` context.

That finding splits one question into two:

- **R2 fixes graph starvation.** It exposes more topology as context. It does
  not change which nodes are eligible to be returned.
- **R3 fixes candidate eligibility.** It permits a bounded, deterministic
  subset of R2's own context to become scoreable.

M0A.1 answered this on three datasets, at one hundred queries each, as a
saturation curve. It did not ask whether the split holds on the other three
sealed datasets, whether the nine-family feature catalog transfers beyond the
one dataset it was developed on, or where — per dataset, per regime — graph
context alone is the limit versus candidate coverage. That is what M0B maps.

**Restated plainly, in the locked wording from M0A.1, unchanged:**

> R2 already exposed the answer; R3 just makes it scoreable.

M0B does not re-ask this. It asks whether the same sentence holds everywhere.

## 2. M0A and M0A.1 are history

Both verdicts stand exactly as filed:

```text
M0A:   MOVEMENT_UNDER_A_BREACHED_ABORT_RULE
M0A.1: ADVANCE
```

M0B does not rewrite, soften, or re-derive either. Nothing measured here may
be used to restate an M0A or M0A.1 outcome. Carried unchanged:

1. Native structural expansion materially moves candidate headroom, most on
   2wiki_clean and metaqa.
2. kNN-only expansion recovers zero golds in every tested cell (M0A).
3. `L1_DIRECTIONAL` and `STRUCTURAL_NEIGHBOUR` produce identical observed
   candidate-headroom outcomes; directional is demoted to ablation-only, not
   mainline, and its artifacts are retained rather than deleted.
4. `BEYOND_U2_RECOVERED == 0` on every M0A.1 cell (4 of 6 (dataset, family)
   cells had any recoverable gold at all; `squad_clean` had none in either
   family and correctly did not inform the universal-budget search).
5. `+64` is the smallest saturation point that reaches full recovery on every
   cell that had any recovery to reach, on the three datasets probed.

### The biggest scientific implication, restated

A `TARGET_H1`-on-QLS null result (more context, same scoreable candidates,
roughly zero QLS gain) does **not** mean the extra graph nodes were
irrelevant. On metaqa some of them were the missing gold answers, just not
eligible to be returned. R2 and R3 are separate causal questions, and a null
on one says nothing about the other.

## 3. Regimes and sets

| regime | scored nodes | graph | changes the candidate oracle |
|---|---|---|---|
| R1 | `Cq` | `G[Cq]` | no |
| R2 | `Cq` | `TARGET_H1(Cq)` (`= U2`) | no |
| R3 | `Cq_struct` | `TARGET_H1(Cq_struct)` | yes — disclosed, see §4 |

```text
Cq          the historical scored candidate set, frozen order, bit-exact
U2          TARGET_H1(Cq) -- in-neighbours only, unchanged contract
A64         STRUCTURAL_NEIGHBOUR admission at graph_expansion_cap=64
Cq_struct   stable_union(Cq, A64) == expand(...).additive_pool, no separate union step
```

**`A64` is M0A.1's saturation-curve `budget=64` point, reused unchanged — not
M0A.1's own headline arm, which used `cap=128`, and not M0A's original
`cap=128` default.** Constructed as:

```python
candidate_expansion_v2.expand(
    method="STRUCTURAL_NEIGHBOUR",
    budget=ExpansionBudget(
        hop_cap=1, per_seed_cap=16,
        graph_expansion_cap=64, neighbour_scan_cap_per_seed=4096,
    ),
).admitted
```

Universal and dataset-agnostic by declaration. If a dataset's cost or
recovery at `+64` turns out pathological, the response is **report and
stop** — never a silent per-dataset budget change.

### Which graph `A64` walks

M0A.1's `edge_provenance_control` compared `structural_only` against
`baseline_a_simple` for structural admission specifically and found them
behave identically on every measured cell. `A64` is built over
**`structural_only`** — the simpler, unambiguously native-only member of an
interchangeable pair. This was an open choice before this document existed;
it is made once, here, disclosed, and not re-derived per dataset.
`baseline_a_simple` is retained as a single cross-check cell, matching
M0A.1's own precedent of measuring a provenance cross-check without
multiplying it through the whole matrix.

This is a different axis from the **PROVENANCE feature family** below
(native / kNN / union edges backing `G[Cq]` and `TARGET_H1` generally,
already measured on all six datasets in
[`docs/EDGE_PROVENANCE_RESULTS.md`](EDGE_PROVENANCE_RESULTS.md)). Both
concern "which edges"; neither stands in for the other.

## 4. Why the paper-1 candidate-admission prohibition does not block this

`configs/candidate_headroom.yaml`'s `diagnostic_contract` sets
`candidate_regeneration`, `candidate_admission` and `graph_expansion` to
`prohibited_in_paper_1`, and records `reachable_missing_golds_are_a_paper_2_question:
true`. **That contract is not edited, reinterpreted, or relaxed anywhere in
this stage.**

It governs the historical scientific object `oracle(Cq)` over the fixed
frozen pool. That object is unchanged: R1 and R2 still score exactly `Cq`.
R3 scores `Cq_struct`, an explicitly separate, disclosed object.

`configs/m0a_probe.yaml` already crossed this exact line, under the
identical `why_a_new_namespace` key, when M0A was declared. Its code lives
in `src/mp_retrieval/candidate_expansion_v2.py` and `headroom_v2.py` — a
namespace built to study `oracle(Cq')` as a declared intervention, never
writing into a historical path. M0B reuses that namespace and that
precedent rather than asking the question a second time.

[`docs/ZERO_TRAINING_MATRIX_PROPOSAL.md`](ZERO_TRAINING_MATRIX_PROPOSAL.md)
named R3 as blocked and reserved the decision for *"its own declaration
rather than this one."* This file is that declaration, scoped to
zero-training measurement of R3 across six datasets. It does not reopen
paper 1's already-sealed content, and it authorises **measuring** R3 —
nothing about what measuring it would imply downstream.

## 5. Datasets and sampling

Six datasets, all sealed (`outputs/sa_mlp_confirmation/<key>.json`,
`status: SA_MLP_CONFIRMATION_DATASET_COMPLETE`):

| dataset | regime | queries | M0A.1 measured? |
|---|---|---:|---|
| `squad_clean` | single-hop | 130,319 | yes |
| `musique_clean` | multi-hop text | 19,938 | no |
| `2wiki_clean` | multi-hop text | 15,000 | yes |
| `hotpotqa_clean` | multi-hop text | 97,852 | no |
| `metaqa` | knowledge base | 407,513 | yes |
| `webqsp` | knowledge base, densest graph | 1,578 | no |

`webqsp` carries 13,379,166 directed edges against `2wiki_clean`'s 855,146 —
roughly 15.6× denser. Per-query cost is a function of pool size and local
density, neither known outside the three M0A.1-measured datasets. It is
named here as the specific cell the compute probe (§8) must clear before
launch, not shrunk pre-emptively below the filed sampling convention.

**Sampling.** The headline run reuses M0A and M0A.1's own convention
unchanged: 100 queries per dataset, validation split, deterministic prefix.
One consistent, cross-validated convention across the whole track rather
than a new one invented per stage — revisable before step 4 actually
launches, since this document does not launch it. A separate five-query
per-dataset, in-process, no-Modal smoke sample (step 2) checks construction
correctness before any container spend.

## 6. Feature catalog

Reused verbatim from
[`configs/graph_context_pilot.yaml#feature_catalog`](../configs/graph_context_pilot.yaml)
(`status: FROZEN_AT_D10`). **Not substituted with the 2Wiki-selected winner
subset** (GEOMETRY + RETRIEVAL, seven of thirteen columns) — that subset is
explicitly not declared universal by its own frozen doc, and whether it
transfers is the open question M0B exists to answer. All nine families are
carried on all six datasets.

Every family's frozen evidence today is R1 only, 2wiki_clean, seed 0. Three
gaps, named rather than papered over:

**PROVENANCE is reused, not re-measured.**
[`docs/EDGE_PROVENANCE_RESULTS.md`](EDGE_PROVENANCE_RESULTS.md) already
covers all six datasets across `baseline_a_simple` / `knn_only` /
`full_union_c`. M0B folds that table into the feature/regime map as the
PROVENANCE row, cited rather than rerun.

**NODE_ROLE — implemented as Safeguard A, not deferred.**
`graph_context.qls_local_features`'s own docstring is explicit: *"Only the
`pool` rows are returned — context nodes are never scored."* The only member
ever probed, `bridge_support`, was killed at D0b and stays killed — this is
not its reintroduction. `overlap_audit.node_roles` assigns exactly one role
per node per (dataset, regime) cell — `RETRIEVAL_CANDIDATE` (∈ `Cq`),
`STRUCTURAL_SCORED_CANDIDATE` (∈ `scored \ Cq`; empty under R1/R2, `= A64`
under R3), `CONTEXT_ONLY` (∈ `context \ scored`; empty under R1) — as
bookkeeping over sets M0B already constructs, not a new fitted feature.
`node_roles` raises if `Cq ⊄ scored` or `scored ⊄ context`, so it doubles as
a wiring check on the caller's regime construction. Given both containments
hold, the three roles are provably an exact partition of `context` — proved
in the module docstring and checked by a 200-trial randomized sweep in
`tests/test_overlap_audit.py`, not asserted on a single hand-picked example.
**These are deterministic set-membership labels only — nothing here is
trained, and M0B does not call a role "useful" before any trained-screen
phase exists.** `dense_rr = 0`, `splade_rr = 0`, `agreement = 0` mark "not in
that ranking" for any node without one, reusing `rank_feature_rows`'s
existing convention (today scoped to `Cq` members only) extended to the
wider role universe; each such zero is paired with its role flag so it
cannot be read as "ranked and scored poorly."

**A new measurement beyond M0A.1.** M0A.1 classified only golds. M0B also
asks: of *all* `A64`-admitted nodes, not just the gold-recovering ones, what
fraction is already in `U2`? The gold-only half already exists
(`overlap_audit.classify_query_golds`); the all-admitted-nodes half is new
code — `overlap_audit.admitted_node_overlap` /
`aggregate_admitted_node_overlap` — added to `overlap_audit.py` rather than
duplicated elsewhere.

**Resolving the directionality question, before step 2, as required.**
`A64(q) ⊆ U2` is **not** a mathematical identity, and no test in this
codebase asserts it as one. Tracing both constructions:

- `U2 = TARGET_H1(Cq)` walks true in-neighbours of the *raw, directed*
  `graph.pt` (`dataset.rowptr`/`dataset.col` → `build_operators` →
  `operators.forward`).
- `A64` walks the frontier of the *symmetrised* `structural_only` edge
  family — reconstructed from per-document neighbour lists in
  `edge_provenance.py`, a different source from `graph.pt`, then explicitly
  passed through `_undirected()` / `symmetric_csr()` in
  `scripts/run_m0a1_overlap.py` before `expand()` ever sees it.

Containment needs two independent, dataset-specific facts to both hold,
neither of which follows from the code's structure alone:

1. **`graph.pt` (sealed A) is bidirectionally closed for that dataset** —
   symmetrising it would change nothing. Verified directly from
   `outputs/edge_provenance_analysis.json`'s `graph_audits` block: true for
   `squad_clean`, `2wiki_clean`, `musique_clean`, `metaqa`, and `webqsp`;
   **false only for `hotpotqa_clean`**, whose directed receptive field is
   strictly smaller than its symmetrised one.
2. **`structural_only`'s edges are fully covered by sealed A's edges** —
   tracked by `edge_provenance.reconstruct_edge_families()`'s own
   `structural_coverage_by_sealed_a` diagnostic. That diagnostic's existence
   is itself evidence the repo's own authors did not assume full coverage;
   it is measured per dataset, not guaranteed.

Fact 2 is not independently confirmed to be `1.0` on any dataset from
artifacts already on hand. Per the user's instruction — *"if yes, prove it
... if no, retain the declared empirical containment-rate diagnostic and do
not overclaim"* — this is a **no**: containment is retained as the declared
empirical rate (`aggregate_admitted_node_overlap`'s `containment_rate`),
measured on all six datasets, including the five that are bidirectionally
closed. `hotpotqa_clean` is flagged as the dataset where a rate below `1.0`
would be least surprising; a rate below `1.0` on any of the other five would
be the more informative result, since it would isolate fact 2 (coverage) as
the cause rather than fact 1 (closure).

## 7. Graph-context diagnostics — "Phase −1 / Stage C"

Reused, not reimplemented. `src/mp_retrieval/graph_context.py` is already
generic over scored set, context, and graph: `context_nodes` (R1 = `CAND`,
R2 = `TARGET_H1` already registered arms), `candidate_structure` (isolate
fraction, radius-1 retention), `context_report` (size / retention /
isolation / boundary-cut), `two_path_preservation` (candidate-endpoint
two-hop bridge retention), `seed_distance` (seed-relative reach, generic
over `(nodes, pool, seeds)`).

R3 is a new **arm**, not new code — these functions take the scored set and
graph as arguments, so R3 is a call with `Cq_struct` and
`TARGET_H1(Cq_struct)`, the same way R2 was a call with `TARGET_H1(Cq)`.

A thin wrapper is still needed: no single existing call returns
`context_report`, `two_path_preservation`, and `seed_distance`-derived reach
together, and none of them aggregates across queries into a per-(dataset,
regime) cell. The wrapper calls `graph_context.py` directly — it does not
route through `scripts/run_graph_context_pilot.py`, which inlines the same
math separately and mixes in QLS-feature "movement" diagnostics M0B has no
use for.

## 8. Classification labels — predeclared

Filed before any M0B result is read, exactly as the user's constraint
requires: *never call a feature useful before fitting.*

Applies per (dataset, regime, feature family) cell. **Labels are not
mutually exclusive** — a cell may be `SIGNAL_PRESENT` and `HIGH_COST` at
once, mirroring M0A.1's own not-disjoint cross-tabulation discipline. No
later stage may collapse them into a single ranking.

| label | predicate |
|---|---|
| `NO_VARIATION` | every column: nonzero rate `< 0.5%` or `> 99.5%`, and value stddev `< 1e-6` |
| `REDUNDANT` | max \|Pearson r\| ≥ 0.95 against the frozen base or another family in the same cell |
| `SIGNAL_PRESENT` | not `NO_VARIATION`, not `REDUNDANT`, nonzero rate in `[2%, 98%]` on ≥1 column |
| `HIGH_COST` | p95 construction cost > 4.0× the cheapest family's p95 in the same cell |
| `CANDIDATE_FOR_TRAINED_SCREEN` | `SIGNAL_PRESENT`, not `REDUNDANT`, not in the 2Wiki base, not `HIGH_COST` (or `HIGH_COST` but flagged for a representative probe rather than excluded) |

`NO_VARIATION` and `REDUNDANT` read structure only — no gold label enters
either. A gold-conditioned diagnostic (in the spirit of D0's one-off
univariate rank-AUC probe) may be reported *separately*, explicitly labelled
as a diagnostic, and may inform step 7's shortlist — but it is never an
input to these five predicates. Kept apart because a gold-blind
classification rule and a gold-conditioned diagnostic look similar enough to
blur together silently, and the whole track's discipline is not letting one
measurement stand in for another. `HIGH_COST`'s `4.0×` factor is the
systems contract's own `latency_factor`, reused unchanged rather than refit
for this purpose.

## 9. Systems contract and invariants

Thresholds inherited unchanged from
`configs/m0a1_overlap.yaml#systems_contract_for_the_structural_arm`:
latency factor `4.0`, peak RSS `14 GiB`. Measured **independently** for
candidate admission, `TARGET_H1` construction, feature construction, and
shared graph extraction — mirroring M0A.1's "does not inherit the
directional verdict" discipline. `webqsp` is named as the cell most likely
to test the RSS bound; a breach there is report-and-stop, not a silent
threshold change.

Invariants, asserted in the runner rather than assumed, on every dataset:
`R1 == R2` bit-exact, `Cq` bit-exact against the frozen artifact, `A64`
disjoint from `Cq`, `Cq ⊆ Cq_struct` and `Cq ⊆ U2`, `Cq_struct ==
additive_pool`, gold-permutation byte-identity (the standing parametrized
test, not re-derived per dataset), plus the two admitted-node-overlap
fractions from §6 (all-admitted and gold-only). The R3-specific checks
(`A64` disjoint from `Cq`, the admitted delta bounded by the universal cap,
`Cq_struct == Cq ∪ A64` with no separate union step, `Cq ⊆ Cq_struct`) are
`overlap_audit.regime_set_invariants`, run per query, not per dataset
sample — a single query breaching one fails the run rather than being
averaged away. `scored_R1 == scored_R2` is true by construction (both
regimes score the same `Cq` array) and is checked once as a wiring assertion
where the runner builds the two regimes, not re-derived per query.
`node_roles` (§6) independently re-checks `Cq ⊆ scored ⊆ context` for every
regime it is called on, so a violation surfaces twice through unrelated code
paths rather than once.

## 10. Compute — a plan, not a final number

**Reused, already measured** (peak RSS, and the expansion/context p95 ratio
— both real, from the M0A.1 run, not re-estimated):

| dataset | peak RSS | expansion/context p95 ratio (`structural_only`) |
|---|---:|---:|
| `squad_clean` | 4.79 GB | 0.176 |
| `2wiki_clean` | 3.63 GB | 0.172 |
| `metaqa` | 7.68 GB | 0.344 |

**Disclosure:** the M0A.1 runner persisted the ratio and peak RSS but not
the raw p50/p95/p99 latencies, so absolute per-query cost is not available
even for these three datasets from that artifact alone. The step-2/3 probe
re-measures absolute latency for all six datasets, not only the three new
ones.

**Reused, already measured — R1 and R2, all six datasets, Stage C evidence**
(`docs/GRAPH_CONTEXT_PILOT_RESULTS.md`, 300 validation queries per dataset,
real Modal hardware, $0.22 total pilot spend, not re-estimated). That
pilot's `CAND` arm is exactly R1's `scored=Cq, graph=G[Cq]`; its `TARGET_H1`
arm is exactly R2's `context=TARGET_H1(Cq)=U2`. Confirmed by reading
`scripts/run_graph_context_pilot.py:170` directly: the pilot's `feature_ms`
column is timing the identical call this runner makes —
`qls_local_features(..., edge_source=operators.edge_source)` — not an
approximation of it. Milliseconds per query, p95 (p99 in parens where the
per-dataset table reports it):

| dataset | R1 build | R1 features | R2 (`TARGET_H1`) build | R2 features | R2 total | U2 context nodes p95 |
|---|---:|---:|---:|---:|---:|---:|
| `2wiki_clean` | 0.0 | 3.7 | 1.9 | 6.9 | 8.7 | 2,616 |
| `hotpotqa_clean` | 0.3 | 65.5 | 31.7 | 143.5 | 170.1 (205.9) | 7,509 |
| `metaqa` | 0.0 | 7.0 | 1.7 | 27.6 | 29.3 | 12,039 |
| `musique_clean` | 0.0 | 3.6 | 0.6 | 12.7 | 13.2 | 3,355 |
| `squad_clean` | 0.0 | 13.2 | 5.3 | 155.2 | 160.1 | 8,378 |
| `webqsp` | 0.2 | 34.9 | 21.9 | 41.8 | 62.8 (89.4) | 4,764 |

This closes most of the "unmeasured" gap below for R1 and R2 specifically —
`musique_clean` and `hotpotqa_clean` are not actually unmeasured at those two
regimes, and `webqsp` is not either. The step-2/3 probe still re-measures
R1/R2 (the runner computes all three regimes every call; there is no code
path that computes R3 alone), but its R1/R2 numbers now serve as a
consistency cross-check against this much larger 300-query sample rather
than as the only evidence for them.

**Genuinely unmeasured anywhere:** R3's `A64` admission cost (`expand`'s
`STRUCTURAL_NEIGHBOUR` rule over the symmetrised `structural_only` family
at `graph_expansion_cap=64`) — the pilot above never runs a capped
expansion over a *different* edge-provenance graph, only `TARGET_H1` over
the raw directed `graph.pt`; `Cq_struct`'s own `TARGET_H1(Cq_struct)=U3`
build and feature cost — plausibly close to R2's given the pilot's own
finding that "the cost scales with context nodes, not with corpus size"
(§9 there), since `Cq_struct` is at most 64 nodes larger than `Cq`, but
this is an extrapolation from a different arm, not a measurement, and is
not asserted as one; and peak RSS for the full R1+R2+R3 pipeline together
(the pilot's cost section reports container-seconds and CPU-hours, never
RSS). These three are what the step-2/3 probe exists to measure — not
R1/R2 construction cost in general, which is already evidenced above.

**Implemented as Safeguard C:** `scripts/run_m0b_webqsp_probe.py` — webqsp
only, 10-20 queries, real per-query timing for build/feature/seed-distance
cost on R1/R2/R3, A64 admission cost, context node/edge counts, peak RSS,
and the declared (not measured — nothing here writes to disk, so there is
no filesystem workspace to instrument; `run_m0a1_overlap.py`'s own
`temporary_workspace_bytes` convention is reused, scaled to this cap of 64)
temporary-workspace bound. Deliberately excludes `two_path_preservation`
from per-query timing, matching `scripts/run_graph_context_pilot.py`'s own
scope: it is a proof-verification tool pinned by
`tests/test_two_path_preservation.py`, not a per-query serving cost.
Deliberately excludes gold-conditioned statistics entirely, per "no need for
meaningful retrieval statistics from those 10 queries" — Safeguard B's
set-membership invariants and the A64-in-U2 containment rate are still
checked and still gate the run, since neither reads a gold label.
`tests/test_m0b_webqsp_probe.py` proves the wiring locally on a synthetic
dataset before any real Modal spend, the same discipline as
`run_m0b_regime_map.py`.

**Launched for real, 2026-09-03.** The first attempt failed on contact with
real data: `qls_local_features` requires its `nodes=` argument pre-sorted
(it uses `np.searchsorted` against it and does not sort it itself, unlike
its `pool=` argument, which it dedupes internally) but R1 passed the raw
candidate pool as `nodes=` unsorted, raising `IndexError` inside the shipped
kernel on webqsp's real (non-monotonic, retrieval-ranked) candidate order.
`run_graph_context_pilot.py:157` had always sorted+deduped `pool` before
using it as any arm's `nodes=`; this runner's R1 arm — the one arm that
aliases `pool` directly as `nodes` rather than getting it from
`context_nodes` (which sorts internally) — had not. Fixed with one
`np.unique` call ahead of the per-query loop. The synthetic fixture had
been silently accident-proof against this: its candidate rows were built
ascending with a small constant shift between `dense`/`splade`, which
`complete_data._stable_union`'s first-occurrence dedup happens to
reassemble back into a fully sorted sequence regardless. Reordered to
descending (same value set, so no other fixture invariant moved) so the
suite now reproduces the failure without the fix and confirms it with the
fix — both checked before relaunching. No test-suite regression: 5713
passed, 108 skipped (pre-existing, unrelated) on the full local run.

**Real results** (`outputs/m0b_webqsp_probe/webqsp.json`, 15 queries,
validation split, `structural_only`, cap=64):

All four Safeguard B invariants held. A64 admitted a median of 16 nodes/query
(p95 34.7, max 41 — well inside the cap=64 budget) at negligible cost (p99
0.90 ms/query). Containment: **265/265 admitted-node instances were already
inside U2** — `containment_rate = 1.0`, no `admitted_beyond_u2` — the same
`BEYOND_U2_RECOVERED == 0` shape M0A.1 found on its own three datasets, now
also true on `webqsp`'s A64/U2 pair specifically. Peak RSS was 4.59 GB
against the 16 GB container (comfortable, and `webqsp` is the largest of the
six graphs by a wide margin).

R1's raw `feature_latency_ms` (p50 36.6, p95 3640.8, p99 10365.8, max
12047.0) looks alarming read as a serving-cost tail but is not one:
`_local_feature_chunk` is `@njit(cache=True, parallel=True)`
(`structural_features.py:339`), so a fresh container pays parallel
JIT-compilation once, on whichever call reaches the kernel first — R1's,
since it runs first in the per-query loop. The arithmetic confirms a single
outlier: `mean × 15 − max ≈ 510.7 ms` spread over the other 14 queries
averages `≈36.5 ms`, matching the p50 almost exactly, and R3's own feature
timings (called later, after the kernel is already compiled) show no such
cliff. Steady-state R1 feature cost is `≈36.6 ms` (p50), consistent with the
pilot's own webqsp p95 of 34.9 ms above. The one-time compile cost is real
and is charged separately, once per container, in the estimate below — not
folded into a misleading per-query figure.

R2 and R3 build/feature/distance costs (p50/p95/p99/max, all `ms`):

| regime | build | features | seed_distance (diagnostic) | context nodes (median/p95/max) | context edges (median/p95/max) |
|---|---|---|---|---|---|
| R1 | 0 (no build) | p50 36.6 / p99 10365.8¹ | p50 76.6 / p99 96.0 | 313 / 382 / 392 | 2,468 / 3,560 / 3,630 |
| R2 | p50 18.7 / p99 21.0 | p50 39.9 / p99 52.7 | p50 81.6 / p99 91.0 | 1,574 / 6,221 / 11,894 | 25,714 / 74,614 / 159,570 |
| R3 | p50 19.6 / p99 24.6 | p50 125.8 / p99 311.6 | p50 79.5 / p99 99.9 | 29,614 / 69,887 / 72,546 | 595,262 / 1,944,817 / 2,034,204 |

¹ p99 is the JIT-compilation artifact explained above; steady-state is the p50.

R3's context is far larger than R2's (edges: median 595K vs. 26K, a ~23×
jump) — A64's structural admissions evidently sit near much denser local
neighbourhoods than the frozen candidates alone reach — but feature cost
grows sub-linearly with it (features: p99 311.6 ms vs. R2's 52.7 ms, a
~5.9× jump against a ~23× edge-count jump), and build/diagnostic cost barely
moves (build ×1.17, `seed_distance` ×1.10). This ratio — not `webqsp`'s
absolute R3 numbers — is what the estimate below extrapolates to the other
five datasets, for the reason in the next paragraph.

**Correction to "`webqsp` is the named risk":** the already-measured R1/R2
table above shows `webqsp` is *not* the worst case for feature latency —
`squad_clean` (155.2 ms) and `hotpotqa_clean` (143.5 ms) both already cost
more at R2 than `webqsp` does (41.8 ms), despite `webqsp`'s graph being
~15.6× larger by edge count. Feature cost tracks context size, not corpus
size, and nothing says the two must correlate. `webqsp` stays the named risk
for *memory* (largest graph, so the peak-RSS measurement above is the
relevant worst case there) but is very likely **not** the per-query latency
worst case for R3 — extrapolating "`webqsp`'s absolute R3 numbers apply
everywhere" would have understated `squad_clean` and `hotpotqa_clean`
specifically. The estimate below applies `webqsp`'s measured R3:R2 *ratio*
to each dataset's own already-measured R2 baseline instead, which carries
that risk forward correctly.

**Six-dataset, 100-query estimate for step 4**, per-query cost per dataset
= R1 (p50 features + `webqsp`'s p99 diagnostic as a uniform proxy, since
`seed_distance` moved only 76.6→99.9 ms across all three of `webqsp`'s own
regimes and is not obviously context-size-driven) + R2 (p99, all real,
per-dataset) + A64 (`webqsp`'s p99, 0.90 ms, uniform — admission cost is
bounded by the fixed per-seed/neighbour-scan caps, not context size) + R3
(build ≈ R2 build × 1.17, features ≈ R2 features × 5.91, distance =
`webqsp`'s p99 99.9 ms uniform, all per-dataset-scaled, not `webqsp`'s
absolute numbers):

| dataset | R1 (ms) | R2 (ms) | A64 (ms) | R3 est. (ms) | total/query (ms) | ×100 (s) |
|---|---:|---:|---:|---:|---:|---:|
| `2wiki_clean` | 99.7 | 99.8 | 0.9 | 142.9 | 343.3 | 34.3 |
| `hotpotqa_clean` | 161.5 | 266.2 | 0.9 | 985.1 | 1413.7 | 141.4 |
| `metaqa` | 103.0 | 120.3 | 0.9 | 265.0 | 489.2 | 48.9 |
| `musique_clean` | 99.6 | 104.3 | 0.9 | 175.7 | 380.5 | 38.1 |
| `squad_clean` | 109.2 | 251.5 | 0.9 | 1023.3 | 1384.9 | 138.5 |
| `webqsp` (real, not estimated) | 132.6 | 164.7 | 0.9 | 436.1 | 734.3 | 73.4 |

Sum of the six ×100-query columns ≈ 474.6 s of query-loop compute. Adding a
generous 60 s/container for image pull, dataset load, CSR construction and
the one-time JIT-compile (`webqsp`'s own observed max of 12.0 s already sits
inside this margin) × 6 containers = 360 s. Total ≈ 834.6 s ≈ **0.232
compute-hours**, at the $0.634/h CPU substrate rate `docs/COMPUTE_LEDGER.md`
already documents (the same 4-CPU/16 GB shape this probe used) ≈ **$0.147**.

That lands within a factor of two of Stage C's own historical actual/ceiling
pair ($0.16 actual against a $5.15 ceiling) — a reassuring consistency
check, not a coincidence being relied on. Following the same margin this
repo's own ledger uses throughout (`docs/COMPUTE_LEDGER.md`: every stage
files a ceiling roughly 30-40× its actual cost, not a tight bound), the
filed ceiling below is **$5.00** — about 34× this point estimate — rather
than the point estimate itself, so ordinary container-to-container variance
does not force a re-file.

This does not trigger the stop condition: the measured-and-extrapolated
spend is far below the $3.00 placeholder it replaces, not materially above
it, and every Safeguard B invariant held on real data. Step 4 is authorised
to proceed on this basis.

**Step 2, launched for real, 2026-09-03.** All six datasets,
`scripts/modal_m0b_regime_map.py` (`run_regime_map_smoke`), 5 queries each,
validation split, deterministic prefix, `structural_only`, cap=64 —
`outputs/m0b_regime_map/smoke/*.json` (gitignored, not committed; this
section is the record). Every job returned `M0B_REGIME_MAP_COMPLETE`: no
Safeguard B invariant breached on any dataset, including the newly-tightened
`scored_r1_equals_scored_r2` (now a real per-query array-equality check, not
an assumed label — see `scripts/run_m0b_regime_map.py`).

*Containment, reported as the two separate figures the user's own
instruction requires — never conflated:*

| dataset | all-admitted-node containment | recovered-gold instances | recovered gold already in U2 |
|---|---:|---:|---:|
| `squad_clean` | 210/210 = 1.0 | 0 | n/a (none recovered at this sample size) |
| `2wiki_clean` | 112/112 = 1.0 | 3 | 3/3 = 1.0 |
| `musique_clean` | 136/136 = 1.0 | 0 | n/a |
| `hotpotqa_clean` | 159/159 = 1.0 | 0 | n/a |
| `metaqa` | 101/101 = 1.0 | 2 | 2/2 = 1.0 |
| `webqsp` | 82/82 = 1.0 | 4 | 4/4 = 1.0 |

Every dataset that recovered any gold at all found it already inside U2 —
`BEYOND_U2_RECOVERED == 0`, the same shape M0A.1 and Safeguard C already
found, now replicated on three more datasets. **`hotpotqa_clean`'s all-node
containment is also 1.0 on this sample** — but this is a 5-query measurement
of a 5-query sample, not a closure of the directionality risk flag: `A64 ⊆
U2` is not a mathematical identity for `hotpotqa_clean` (its `graph.pt` is
not bidirectionally closed, unlike the other five), so a gap can only be
ruled out by measurement, not assumed from this result. The flag stays open
into the 100-query headline run.

*Cold start, confirmed real and isolated to R1 query 0 on all six datasets*
(`ms`, `structural_only`):

| dataset | R1 cold-start compile | R1 steady-state p50 | R2 raw range | R3 raw range |
|---|---:|---:|---:|---:|
| `squad_clean` | 12,601.4 | 8.1 | 104.9–167.8 | 95.9–165.0 |
| `2wiki_clean` | 8,443.6 | 3.3 | 5.1–7.8 | 78.0–90.0 |
| `musique_clean` | 8,715.7 | 1.4 | 5.9–14.8 | 11.6–14.4 |
| `hotpotqa_clean` | 11,953.5 | 52.3 | 87.4–122.6 | 184.6–1,160.4 |
| `metaqa` | 11,686.3 | 1.7 | 3.8–5.6 | 6.0–19.9 |
| `webqsp` | 8,792.6 | 56.4 | 59.3–69.0 | 55.4–383.8 |

No R2 or R3 raw array shows a cold-start-shaped outlier (one value orders of
magnitude above the rest) on any dataset — the one-time Numba parallel-JIT
compile really does land exactly once, on R1's first call, exactly as
designed. `cold_start_compile_ms` is reported per dataset, split from
`steady_state`, and the full raw array is retained unfiltered in every
regime's `feature_latency_ms` block — nothing here is discarded.

**A genuine finding, not noise: `hotpotqa_clean`'s R3 context explodes
relative to R2**, far more than any other dataset:

| dataset | R2 context nodes (median) | R3 context nodes (median) | growth |
|---|---:|---:|---:|
| `squad_clean` | 6,120 | 6,404 | 1.05x |
| `2wiki_clean` | 2,039 | 40,737 | 20.0x |
| `musique_clean` | 2,402 | 3,882 | 1.6x |
| `hotpotqa_clean` | 1,935 | 126,056 | **65.1x** |
| `metaqa` | 2,383 | 7,893 | 3.3x |
| `webqsp` | 1,574 | 29,614 | 18.8x |

This is why `hotpotqa_clean`'s R3 feature cost is both the highest of the
six and the noisiest within just 5 queries (184.6–1,160.4 ms, a 6.3x spread,
climbing rather than flat) — R3 feature cost tracks context size, and
`hotpotqa_clean`'s A64 admissions evidently sit near much denser local
neighbourhoods than the other five datasets', on the queries sampled here.
Worth watching at the 100-query headline scale, not a reason to stop: every
Safeguard B invariant held (including
`structural_only.admitted_delta_within_universal_cap`, so A64 itself never
exceeded its own +64 budget — the blowup is in the *context* TARGET_H1 walks
outward from the admitted nodes, not in admission itself), and peak RSS
stayed well inside the container even for this dataset (below).

**Correction to "`webqsp` is the named risk" — memory, not just latency:**
measured peak RSS at smoke scale is actually highest for `metaqa` (7.69 GB),
not `webqsp` (4.62 GB), contradicting the edge-count-based assumption that
the largest graph (`webqsp`, ~15.6x `2wiki_clean`'s edge count) would also
show the largest peak RSS:

| dataset | peak RSS (GB) |
|---|---:|
| `squad_clean` | 4.79 |
| `2wiki_clean` | 3.67 |
| `musique_clean` | 3.63 |
| `hotpotqa_clean` | 5.78 |
| `metaqa` | 7.69 |
| `webqsp` | 4.62 |

Both remain far under the 16 GB container (highest at 48%), so this does not
change Step 4's authorisation, but the *reason* previously named for
tracking `webqsp` specifically no longer holds for memory — peak RSS does
not track graph edge count here. `webqsp` does stay correctly named as not
the R3 *latency* risk (the earlier correction in this section), and no
dataset's memory margin is remotely tight, so this is recorded as a
correction for accuracy, not a new risk requiring its own gate.

## Step 3: the compute projection, updated from real six-dataset measurement

The step-4 estimate above (`$0.147`) applied `webqsp`'s measured R3:R2
feature-cost *ratio* to each dataset's own R2 baseline, because Safeguard C
only ever measured `webqsp`. Step 2 now measures R1, R2, A64 admission, and
R3 directly on all six datasets, so the ratio-extrapolation is replaced with
each dataset's own real numbers. Per-query cost, p99 of the 5-query smoke
sample per component (`ms`):

| dataset | R1 feat | R2 build | R2 feat | A64 | R3 build | R3 feat | total/query | ×100 (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `squad_clean` | 15.7 | 3.9 | 167.4 | 0.8 | 3.3 | 164.5 | 355.7 | 35.6 |
| `2wiki_clean` | 3.6 | 2.4 | 7.7 | 0.7 | 3.1 | 89.9 | 107.5 | 10.7 |
| `musique_clean` | 1.5 | 0.9 | 14.7 | 0.7 | 0.7 | 14.3 | 32.7 | 3.3 |
| `hotpotqa_clean` | 54.9 | 30.6 | 121.6 | 0.8 | 30.1 | 1,145.0 | 1,382.9 | 138.3 |
| `metaqa` | 2.2 | 1.1 | 5.5 | 0.6 | 1.3 | 19.8 | 30.6 | 3.1 |
| `webqsp` | 70.6 | 66.9 | 68.9 | 1.0 | 67.7 | 378.2 | 653.2 | 65.3 |

At `n=5`, p99 sits near each dataset's own observed max, which is the safe
direction for a projection feeding a gate — this deliberately does not
smooth over `hotpotqa_clean`'s climbing R3 tail.

Sum of the six ×100-query columns ≈ **256.3 s**. Each dataset's own real,
measured `cold_start_compile_ms` (table above) is added once per container
rather than assumed — sum ≈ **62.2 s**. The remaining per-container margin
(image pull, dataset load, CSR construction — still not separately measured)
is reduced from the prior estimate's 60 s to **30 s × 6 = 180 s**, since the
JIT-compile piece it used to cover is now measured exactly rather than
folded in as slack. Total ≈ **498.4 s ≈ 0.1385 compute-hours**, at the
$0.634/h CPU rate `docs/COMPUTE_LEDGER.md` documents ≈ **$0.088**.

This is *lower* than the ratio-extrapolated $0.147, not higher — the ratio
method overstated cost for datasets whose real R3:R2 ratio came in below
`webqsp`'s 5.91x (e.g. `musique_clean`'s real ratio is ≈0.98x) by more than
it understated `hotpotqa_clean`'s (whose real ratio, ≈9.4x, does exceed
5.91x — consistent with the context-blowup finding above, and exactly the
kind of gap a six-dataset measurement is supposed to catch that a
single-dataset ratio cannot). The filed ceiling stays **$5.00**, unchanged —
now roughly a **57x** margin rather than the prior 34x, more conservative,
not less.

This does not trigger the stop condition: the newly-measured spend is
materially *below* the already-authorised $0.147 estimate, not above it, and
every Safeguard B invariant held on real data across all six datasets. Step
4 is authorised to proceed on this basis.

## 11. What this declaration does not authorise

Only step 1 — filing this declaration. Not authorised: the step-2 smoke
sample, a final compute estimate, the six-dataset launch, invariant
verification, the feature/regime map itself, the trained-screen proposal, or
any fitting of any kind. No QLS training. No GNN training. No feature
selection by model outcome. No test-split read. No workspace migration. M0A
and M0A.1's verdicts are not rewritten. `L1_DIRECTIONAL`'s artifacts are not
deleted. The `+64` budget is not tuned per dataset under any circumstance
short of report-and-stop. The 2Wiki winner subset does not replace the full
nine-family catalog anywhere. The trained matrix is never multiplied by all
three provenance families automatically.

The full plan, restated from the authorising directive:

```text
1. file declaration                                    <- this document
2. validate +64 regime construction on small real samples
3. estimate compute
4. run all-six zero-training matrix
5. verify invariants
6. construct feature/regime map
7. propose smallest trained-screen phase
8. STOP
```

Step 8 is a hard stop: no fitting of any kind begins without a further,
separate, explicit authorisation after step 7's proposal is reviewed. Step 7
itself proposes representative cells first, never `6 × 3 × features`
exhaustively by default — the matrix expands further only if representative
fits show that dataset or regime dependence can change the scientific
conclusion.

## 12. What actually happened

```text
1. file declaration                                    DONE (this document)
2. validate +64 regime construction on small real samples   DONE (section 10)
3. estimate compute                                     DONE (section 10, $0.088)
4. run all-six zero-training matrix                     DONE
5. verify invariants                                    DONE
6. construct feature/regime map                         DONE
7. propose smallest trained-screen phase                DONE
8. STOP                                                 HERE
```

Steps 4 through 8's full results — the real per-dataset invariant table, the
two-figure containment report (including the `hotpotqa_clean` directionality
gap this section named as a risk, now measured at 0.36%, and the one
recovered-gold instance A64 alone supplied there), the candidate-eligibility
vs. graph-availability bottleneck map across all six datasets, the
nine-family computability-per-regime map, and the proposed minimum
trained-screen matrix — are filed in
[`M0B_REGIME_MAP_RESULTS.md`](M0B_REGIME_MAP_RESULTS.md), not restated here.
This section exists only as the pointer from the declaration's own plan to
where it was executed. Nothing beyond step 7's proposal is authorised; the
next action on any proposed cell requires its own declaration.
