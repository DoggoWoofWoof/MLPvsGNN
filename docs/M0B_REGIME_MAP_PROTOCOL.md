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

**NODE_ROLE has no implementation.** `graph_context.qls_local_features`'s own
docstring is explicit: *"Only the `pool` rows are returned — context nodes
are never scored."* The only member ever probed, `bridge_support`, was
killed at D0b and stays killed — this is not its reintroduction. M0B adds a
role label per node per (dataset, regime) cell — `retrieval_candidate`
(∈ `Cq`), `structural_scored_candidate` (∈ `Cq_struct \ Cq`, R3 only),
`context_only` (∈ `TARGET_H1(Cq_struct) \ Cq_struct`) — as bookkeeping over
sets M0B already constructs, not a new fitted feature. `dense_rr = 0`,
`splade_rr = 0`, `agreement = 0` mark "not in that ranking" for any node
without one, reusing `rank_feature_rows`'s existing convention (today scoped
to `Cq` members only) extended to the wider role universe; each such zero is
paired with its role flag so it cannot be read as "ranked and scored
poorly."

**A new measurement beyond M0A.1.** M0A.1 classified only golds. M0B also
asks: of *all* `A64`-admitted nodes, not just the gold-recovering ones, what
fraction is already in `U2`? `A64(q) ⊆ U2` is not guaranteed by
construction — `STRUCTURAL_NEIGHBOUR`'s frontier is undirected, `U2`'s is
the in-neighbour `TARGET_H1` contract — so this is measured, not assumed.
The gold-only half already exists (`overlap_audit.classify_query_golds`);
only the all-admitted-nodes half is new code, added to
`overlap_audit.py` rather than duplicated elsewhere.

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
fractions from §6 (all-admitted and gold-only).

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

**Unmeasured:** `musique_clean` and `hotpotqa_clean` at any regime;
`webqsp`, flagged above as the named density risk; R3's absolute
construction cost on any dataset (M0A.1 measured only a ratio).

A placeholder ceiling of **$3.00** is filed so this document is
self-contained, and is explicitly **not final** —
`docs/ZERO_TRAINING_MATRIX_PROPOSAL.md` already states a declaration should
set the budget from a probe, not an extrapolation, and `webqsp`'s ~15.6×
edge-count density makes extrapolating from three datasets to six
unreliable. The number that actually gates step 4's launch comes from step
3's measured-probe-based estimate, filed as its own artifact.

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
