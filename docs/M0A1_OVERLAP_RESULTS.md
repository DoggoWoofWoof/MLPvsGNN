# M0A.1 R2/R3 overlap audit — results

**Status: M0A.1 COMPLETE. No launch is authorised by this document.**

Declaration: [`configs/m0a1_overlap.yaml`](../configs/m0a1_overlap.yaml).
Protocol: [`M0A1_OVERLAP_PROTOCOL.md`](M0A1_OVERLAP_PROTOCOL.md).
Runner: [`scripts/run_m0a1_overlap.py`](../scripts/run_m0a1_overlap.py).
Rule applied by [`scripts/analyze_m0a1_overlap.py`](../scripts/analyze_m0a1_overlap.py),
which reads its thresholds from the declaration and cannot choose them.

M0A's own verdict, `MOVEMENT_UNDER_A_BREACHED_ABORT_RULE`
([`M0A_PROBE_RESULTS.md`](M0A_PROBE_RESULTS.md)), is unchanged by anything
below. The abort fired only on the `L1_DIRECTIONAL` arm; that arm was not
rerun here. Every claim below is labelled: `MEASURED` is read off this run's
own artifact, `REPRODUCED` is read back from M0A's stored artifact rather than
recomputed, `STRUCTURAL IDENTITY` is checked live against the running code
rather than recalled, and `INFERENCE` is a reading of measured numbers.

---

## 1. The verdict

**`ADVANCE`** — the simple `STRUCTURAL_NEIGHBOUR` R3 construction clears all
four filed conditions. MEASURED, by the rule in
`advancement_to_m0b.advance_the_simple_structural_r3_construction_only_if_all_hold`,
applied mechanically by `scripts/analyze_m0a1_overlap.py`. Advancement is not
automatic (`advancement_to_m0b.is_automatic: false`): this is a reading, not a
launch, and no directional method was required to reach it.

| condition | holds |
|---|---|
| 1. leakage and invariants pass | true |
| 2. materially improves candidate headroom | true |
| 3. independent systems contract passes | true |
| 4. a universal bounded expansion rule can be stated | true |

**The headline number is not the interesting result either.** The interesting
result is §2: on every dataset and every structural family measured, R3's
entire recovered-gold mass was already inside R2's context universe. R3 did
not expose anything R2 could not already see; it made a subset of what R2
could see *scoreable*.

---

## 2. What R2/R3 overlap actually looks like

100 validation queries per dataset (`sampling.identical_to_m0a: true` — the
same 100 queries M0A scored), the frozen `Cq` pools, `U2 = TARGET_H1(Cq)`
built fresh per query, and `A_struct` from M0A's unchanged
`STRUCTURAL_NEIGHBOUR` construction at M0A's own headline budget
(`graph_expansion_cap = 128`). Every recovered gold instance is classified
into one of four cross-tabulation cells before being folded into the three
filed classes, so the "already in U2" and "still missing" readings can never
silently overlap without it being visible in the cells beneath them.

| dataset | family | IN_U2 & RECOVERED | IN_U2, not recovered | BEYOND_U2, RECOVERED | BEYOND_U2, not recovered | recovered total |
|---|---|---|---|---|---|---|
| squad_clean | structural_only | 0 | 0 | 0 | 0 | 0 |
| squad_clean | baseline_a_simple | 0 | 0 | 0 | 0 | 0 |
| 2wiki_clean | structural_only | 41 | 3 | **0** | 5 | 41 |
| 2wiki_clean | baseline_a_simple | 41 | 3 | **0** | 5 | 41 |
| metaqa | structural_only | 43 | 0 | **0** | 0 | 43 |
| metaqa | baseline_a_simple | 43 | 0 | **0** | 0 | 43 |

MEASURED. `BEYOND_U2_RECOVERED` — a gold recovered into the scored candidate
pool that R2's context universe did not already contain — is **zero in every
cell**. squad_clean recovers nothing because R1 already sits at ceiling there
(`recall_ceiling@1 = 1.0`; see M0A §7); it is listed for completeness, not
because it informs this question.

Restated as the three filed classes (gold-instance counts; `R2_CONTEXT_RECOVERABLE`
and `STILL_MISSING` share the `IN_U2_NOT_RECOVERED` cell by construction — see
`partition.the_three_classes_are_not_disjoint` in the declaration):

| dataset | family | R2_CONTEXT_RECOVERABLE | R3_BEYOND_R2 | STILL_MISSING |
|---|---|---|---|---|
| 2wiki_clean | structural_only | 44 | **0** | 8 |
| metaqa | structural_only | 43 | **0** | 0 |

`recovery_share`: of the golds structural expansion recovered,
`already_in_r2_context = 1.0` and `beyond_r2_context = 0.0`, on both families,
on both datasets with any recoverable gold. MEASURED.

**Matched-budget agreement.** The same partition, rebuilt against the
matched-budget pool (`|Cq'| == |Cq|`, tail-evicting rather than additive), is
byte-identical to the additive partition's cross-tabulation on all six
family/dataset cells (`matched_budget.agrees_with_additive: true`
everywhere). Structural expansion evicted nothing that mattered to this
question at this budget. MEASURED.

### Locked wording

The declaration filed two candidate sentences before this table existed, one
for each possible reading. The measured reading is unambiguous, so this is the
sentence that stands:

> **R2 already exposed the answer; R3 just makes it scoreable.** Every gold
> instance `STRUCTURAL_NEIGHBOUR` expansion recovered into the scored
> candidate pool, on every dataset and family where anything was recovered,
> was already reachable in `TARGET_H1(Cq)` — R2's context universe. R3's
> distinct contribution measured here is not exposure; it is making an
> already-visible node scoreable, which R2's wider-but-unscored context
> cannot do (§3 reconfirms R1's and R2's oracle ceilings are bit-exact).

The rejected sentence — "R3 extends the scored universe beyond
`TARGET_H1`" — does not hold on any cell measured. **Not claimed**: that this
holds beyond the three datasets, the two structural families, or the 100-query
sample measured here; that it would hold at a different budget; or that it
says anything about `L1_DIRECTIONAL`, which was not rerun.

---

## 3. R1 == R2 reconfirmed bit-exact

`r1_ceiling_equals_r2_ceiling_exactly: true` on all three datasets. R2's wider
context is not scoreable and moves no oracle number; this was reconfirmed
rather than assumed, exactly as filed. MEASURED.

---

## 4. Cross-validation against M0A's own headline numbers

M0A.1 reuses M0A's frozen `STRUCTURAL_NEIGHBOUR` construction unchanged, on
the same 100 queries, at the same `graph_expansion_cap = 128`. It is a
completely separate runner built in this phase, so its headline-budget numbers
should reproduce M0A's matched-budget table
([`M0A_PROBE_RESULTS.md` §4](M0A_PROBE_RESULTS.md)) if nothing drifted between
the two:

| dataset | family | M0A (execution 2) | M0A.1 (this run) |
|---|---|---|---|
| 2wiki_clean | structural_only, golds recovered | 41 | 41 |
| 2wiki_clean | structural_only, recall@5 movement | +15.25 pt | +15.25 pt |
| metaqa | structural_only, golds recovered | 43 | 43 |
| metaqa | structural_only, AnyGold / recall@5 movement | +30.00 / +34.83 pt | +30.00 / +34.83 pt |

MEASURED, to the decimal, from `scripts/analyze_m0a1_overlap.py`'s own
`curve_movement` at the saturating budget (§5) — an independent artifact
agreeing with M0A's is evidence the two runners are not silently diverging,
not a claim either result needed re-derivation to be trusted.

---

## 5. Structural saturation, and a universal budget

Deterministic cumulative `STRUCTURAL_NEIGHBOUR` expansion at
`+4, +8, +16, +32, +64`, then the uncapped one-hop frontier
(`full_bounded_n1_frontier`, which additionally lifts `per_seed_cap` — a
disclosed discontinuity, not a prefix of the numeric points' ordering).
Purpose is saturation analysis, not per-dataset fitting; no budget below was
chosen after looking at which dataset it applied to.

| dataset | family | +4 | +8 | +16 | +32 | +64 | full |
|---|---|---|---|---|---|---|---|
| 2wiki_clean | structural_only, golds recovered | 17 | 27 | 34 | 41 | 41 | 41 |
| metaqa | structural_only, golds recovered | 6 | 17 | 31 | 36 | **43** | 43 |

Marginal recovered-gold-per-added-node reaches zero on the `64 → full` step
for **every** cell with any recoverable gold — including metaqa, which needed
the full `+64` to finish saturating (2wiki plateaus earlier, at `+32`).
`scripts/analyze_m0a1_overlap.py::universal_budget` scans the fixed point list
and reports the smallest one that reaches each cell's own full-frontier
recovery, without reading which dataset it is looking at:

```
smallest_sufficient_universal_budget: 64
per_numeric_point_reaches_full_recovery_everywhere:
  4: false   8: false   16: false   32: false   64: true
```

MEASURED. **One universal `graph_expansion_cap = 64` reaches full observed
recovery on both structural regimes tested** — cheaper than M0A's own headline
budget of 128, at identical recovery, on this evidence.

---

## 6. Independent systems contract for the structural arm

Filed before results, and explicitly not inheriting `L1_DIRECTIONAL`'s M0A
abort: `latency_factor <= 4.0`, `peak_process_rss_bytes <= 15,032,385,536`
(14 GiB).

| dataset | expansion p95 (ms) | context-build p95 (ms) | ratio | peak RSS |
|---|---|---|---|---|
| squad_clean | 0.294–0.370 | 2.051–2.144 | 0.17–0.18 | 4.79 GB |
| 2wiki_clean | 0.288–0.354 | 1.902–2.057 | 0.17–0.18 | 3.63 GB |
| metaqa | 0.252–0.382 | 0.907–0.998 | 0.34–0.40 | 7.68 GB |

MEASURED. Every ratio sits at 4–24% of the 4.0 threshold; every peak RSS sits
at 24–51% of the 14 GiB bound. The bound was genuinely open before this run
(M0A.1 builds `U2` per query plus six curve points, which M0A did not) — it is
not being reported as "already known to pass" the way the latency factor was.

---

## 7. Leakage and invariants, composed from three sources

Condition 1 is not one dict lookup; it is three checks that are never allowed
to silently stand in for one another:

1. **Four invariants recorded on this run's own artifact** —
   `r1_ceiling_equals_r2_ceiling_exactly`, `u2_contains_cq`,
   `a_struct_is_disjoint_from_cq`, `cq_struct_contains_cq` — all `true` on all
   three datasets. MEASURED.
2. **The frozen candidate pool is bit-exact against the frozen artifact** —
   read from `result["candidate_contract"]["status"] ==
   "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"`. `validate_candidate_contract`
   raises rather than records a failure, so a `M0A1_OVERLAP_COMPLETE` status
   already implies this; it is not copied into the `invariants` dict by the
   runner, so the analysis script reads it from its actual location instead of
   assuming a five-key dict is really six. MEASURED.
3. **Gold identity cannot reach expansion.** Checked live, not recalled:
   `scripts/analyze_m0a1_overlap.py::structural_leakage_check` introspects
   `mp_retrieval.candidate_expansion_v2.expand`'s real signature —
   `{method, rowptr, col, node_embeddings, query_embedding, anchor, pool,
   seeds, budget, num_nodes}` — and confirms no parameter could carry gold
   identity. STRUCTURAL IDENTITY, verified fresh on every analysis run. The
   behavioural counterpart — permuting golds leaves `expand()`'s output byte
   identical — is proved once for both expansion methods by
   `tests/test_candidate_expansion_v2.py::test_permuting_gold_leaves_the_expansion_byte_identical`
   and disclosed by name rather than silently assumed to still be true.

All three hold, on all three datasets. `gates`-equivalent process checks
(`trained_anything: false`, `test_split_read: false`) also hold, unchanged
from the runner's own recorded fields.

---

## 8. What this does not authorise

- **Not M0B.** `advancement_to_m0b`'s regimes are filed as a proposal
  (`status: PROPOSAL_AT_M0A1_DO_NOT_LAUNCH`) and remain exactly that. Nothing
  here launches R1/R2/R3 training.
- **Not a directional re-evaluation.** `L1_DIRECTIONAL` was not rerun. M0A's
  wording discipline stands unchanged: *"Directional selection provided no
  incremental candidate-headroom benefit over bounded structural expansion in
  M0A."* This document adds nothing to and subtracts nothing from that
  sentence.
- **Not a claim about ranking.** No ranker has been trained on `Cq_struct`.
  §2's finding is about candidate exposure and scoreability, not about
  whether a trained model would use the newly-scoreable nodes.
- **Not a per-dataset budget.** §5's `graph_expansion_cap = 64` is read off
  the fixed, pre-declared point list on the two datasets with any
  recoverable gold in this sample; it is a candidate universal bound, not a
  fitted one, and not yet validated beyond these three datasets.

---

## 9. Compute

Declared before launch (`configs/m0a1_overlap.yaml#compute`): 3 jobs, CPU-only,
$0.07 estimated against a $0.30 ceiling, anchored to M0A execution 2's
*measured* per-query latencies rather than a fresh host timing. All three jobs
completed on the first attempt (`M0A1_OVERLAP_COMPLETE`, no retries, no
failures); the launch gate (`scripts/spawn_modal_jobs.py m0a1-overlap
--dry-run`) reported `expected_spend_usd: 0.07`, `verdict: "fits in one
window"`, matching the filed estimate exactly before a single container ran.
