# M0C — isolating structural candidate scoreability from graph re-expansion

Declaration: [`configs/m0c_bounded_r3.yaml`](../configs/m0c_bounded_r3.yaml),
filed before any M0C number exists. Status: `DECLARED_NOT_LAUNCHED`.

## 1. The question

Can structural candidate scoreability be isolated from additional graph
expansion? Zero training only.

M0B measured R3 as `scored=Cq_struct, context=TARGET_H1(Cq_struct)`. That
changes two things in the same step: which nodes are eligible to be scored
(candidate admission) and how large the surrounding graph context is (a
second, uncontrolled `TARGET_H1` walk seeded from the wider, post-admission
pool). Because both move together, a future model trained on R2 and again on
R3 cannot attribute a change in behaviour to either cause alone — the
comparison is causally confounded by construction, not by an accident of
sampling.

The evidence this is not a hypothetical concern is M0B's own real,
already-filed number: on `hotpotqa_clean`, median context size grew from
R2's 1,935 nodes to R3's 126,056 — roughly 65×. That is not a marginal
widening; it is a different regime of computation sitting on top of a
different eligibility rule, and the two are inseparable in M0B's data.

M0C's fix is narrow. It does not touch candidate admission (`A64`) at all —
that construction is exactly M0B's, reused. It changes only what R3's
*context* is defined to be:

```
C3         = stable_union(Cq, A64)          -- unchanged from M0B's Cq_struct
U3_bounded = stable_union(U2, A64)          -- NOT TARGET_H1(C3)
```

Admitting a node into the scored set does not recursively earn it another
neighbourhood expansion. `U3_bounded` folds every structurally-admitted node
into the model's domain — it is scoreable and it is in context — without
paying for, or being contaminated by, a second graph walk.

## 2. M0B is history

M0B's own findings are not recomputed, reinterpreted, or rerun. Its commits
(`acceffd`, `e769a1b`), regime definitions, and results stand exactly as
executed. Concretely, and unchanged:

- 600 queries (100/dataset × six datasets), zero invariant breaches, $0.088
  actual compute.
- `hotpotqa_clean`: R2→R3 context growth ≈65× (median 1,935 → 126,056);
  admitted-node containment in U2 = 0.9964 (15 of 4,167 admitted instances
  outside U2) — the only dataset where A64 recovered gold beyond U2's own
  context.
- `webqsp`: 86 gold instances beyond U2 that even A64 does not recover,
  `all_gold_at_pool = 0.31` (36/100 queries with zero gold in the pool) — the
  worst graph-availability ceiling of the six, alongside the worst candidate
  eligibility.
- `metaqa`, not `webqsp`, is the real peak-RSS worst case (7.69 GB).

M0C does not re-derive any of these. What M0B measured as `R3` is renamed
`R3_FULL_REEXPANSION` here — a real, valuable zero-training systems/
structural finding in its own right ("full graph repair after candidate
admission can be pathologically expensive on some graph regimes"), retained
and cited, never rerun. This correction does not invalidate M0B; it makes
M0B more valuable — the full-reexpansion result becomes a systems control,
while M0C gives the causally clean three-regime setup the QLS-vs-GNN paper
actually needs.

### The biggest implication, restated

`R2 → R3_BOUNDED` is now, by construction, a single-variable change:
candidate eligibility, and nothing else. Any future model-behaviour
difference measured across that step is attributable to eligibility alone.
`R2 → R3_FULL_REEXPANSION` (M0B's own object, cited not rerun) remains
available as a *separate*, disclosed comparison for whoever wants to study
the cost/benefit of full re-expansion specifically — but it is never
conflated with the bounded comparison in the same headline number.

## 3. Regimes and sets

| Regime | Scored | Context | Changed vs. M0B |
|---|---|---|---|
| R1 | `Cq` | `G[Cq]` | no |
| R2 | `Cq` | `U2 = TARGET_H1(Cq)` | no |
| R3_BOUNDED | `C3 = stable_union(Cq, A64)` | `U3_bounded = stable_union(U2, A64)` | **new mainline** |
| R3_FULL_REEXPANSION | `Cq_struct` (== `C3`) | `TARGET_H1(Cq_struct)` | M0B's object, cited only |

`A64` is exactly M0B's `STRUCTURAL_NEIGHBOUR` admission at
`graph_expansion_cap=64` over the `structural_only` edge family — re-derived
here (M0B persisted only aggregates, not per-query node-ID arrays, so there
is nothing to literally load from), not regenerated with a different
algorithm and not re-tuned per dataset.

### Why `U2 ∪ A64` rather than asserting `A64 ⊆ U2`

`U2 = TARGET_H1(Cq)` and M0B already established, as a per-query checked
invariant (`u2_contains_cq`), that `Cq ⊆ U2`. Since `C3 = Cq ∪ A64`:

```
U2 ∪ A64 = U2 ∪ (Cq ∪ A64) = U2 ∪ C3        (because Cq ⊆ U2)
```

so `U3_bounded = stable_union(U2, A64)` and `U2 ∪ C3` are the same set. This
is why the construction is a cheap `np.union1d`, not a graph traversal — no
new walk is needed to guarantee `C3 ⊆ U3_bounded`, because `U3_bounded` is
defined as exactly the union that makes it true. M0C checks this
per-query (`c3_subset_u3_bounded`, `u2_subset_u3_bounded` in the runner's own
gated invariants) rather than relying on the algebra alone.

Asserting `A64 ⊆ U2` outright (skipping the union) was considered and
rejected: M0B's own real hotpotqa_clean measurement is 0.9964, not 1.0 — 15
of 4,167 admitted instances sit outside U2, and hotpotqa_clean is the one
dataset where those extra nodes carried gold recovery. `U3_bounded = U2 ∪
A64` keeps every scoreable node inside context, on every dataset, without
depending on a containment rate that M0B already measured as imperfect.

### NODE_ROLE under R3_BOUNDED

Exactly one role per node, unchanged implementation
(`src/mp_retrieval/overlap_audit.py#node_roles`, not modified):

- `RETRIEVAL_CANDIDATE`: `v ∈ Cq`
- `STRUCTURAL_SCORED_CANDIDATE`: `v ∈ C3 \ Cq` (== `A64`)
- `CONTEXT_ONLY`: `v ∈ U3_bounded \ C3`

`node_roles` requires `cq ⊆ scored ⊆ context`, which the containment
argument above proves holds for `(Cq, C3, U3_bounded)` — no code change was
needed to reuse it for the bounded regime.

## 4. Why this does not reopen paper 1 or M0B's own precedent

`configs/candidate_headroom.yaml`'s `diagnostic_contract` (candidate
regeneration/admission/graph expansion all `prohibited_in_paper_1`) is
untouched, exactly as M0A and M0B left it — it governs `oracle(Cq)`; M0C, like
M0B, studies the separate, disclosed object `oracle(Cq_struct)` under the
`why_a_new_namespace` precedent `configs/m0a_probe.yaml` opened and
`configs/m0b_regime_map.yaml` already cited once. M0C cites the same chain a
third time rather than re-litigating it. Nothing about M0C reopens M0B's own
declaration either — `A64` and `C3`'s construction are identical; only the
*context* attached to that already-authorised scored set changes.

## 5. Datasets and sampling

The same six datasets, the same 600-query headline panel (100/dataset,
validation split, deterministic prefix of the split order) — no new
sampling. M0C's entire value is a controlled comparison against M0B on the
exact same queries; a different sample would reopen the question M0C exists
to close. A 5-query/dataset smoke sample (same selection rule) runs first,
matching M0A/M0A.1/M0B's own two-stage discipline, to prove reuse cheaply
before the headline container cost is incurred.

`webqsp` keeps its standing flags from M0B (worst at both candidate
eligibility and graph availability); M0C does not attempt `H2`/`H3` or
another candidate generator for it. Future model comparisons on webqsp must
report absolute R@K *and* regime-specific ceiling attainment (achieved
effectiveness / regime-specific oracle), so unreachable evidence is not
confused with ranking failure — this is the fairness requirement any later
MLP-vs-GNN comparison needs, and M0C's results doc states it explicitly
rather than leaving it implicit.

## 6. Reuse contract

M0B persisted only per-dataset aggregate statistics — never per-query
node-ID arrays. There is no literal cache of `Cq`, `A64`, or `U2` to load.
"Reuse" therefore means: re-derive the identical deterministic construction
(same frozen candidate pools, same seeds, same `A64` rule) and assert the
resulting aggregates equal M0B's own filed numbers bit-exactly. That
re-derivation *is* the reuse proof — not a shortcut around one.

Quantities whose meaning is identical between `R3_FULL_REEXPANSION` and
`R3_BOUNDED`, because neither references `U3`, and are therefore reused/
re-verified rather than newly measured: candidate counts, AnyGold/AllGold/
GoldFraction, oracle R@1/R@5/R@20, recovered missing golds, rescued queries,
provenance of admitted candidates, `A64` itself.

Quantities that are genuinely new and require rebuilding: `U3_bounded`'s
node/edge count, isolate fraction, radius-1 retention, boundary-cut
fraction, candidate-endpoint two-hop path retention, seed-relative reach,
and the nine catalogued feature families' distributions/correlations/
discrimination/latency/RSS, measured over `U3_bounded` instead of
`TARGET_H1(Cq_struct)`.

`R3_FULL_REEXPANSION`'s own context/feature numbers are **cited from M0B's
already-filed results, not recomputed** in M0C's headline run. Recomputing
them would pay for the same expensive `TARGET_H1(Cq_struct)` walk a second
time for no new information, and would work directly against "M0C must be
dramatically cheaper than M0B."

If exact scored-set/headroom reproduction against M0B's filed numbers fails,
the stage stops and reports rather than proceeding on a silently-diverged
construction. This check lives in `tests/test_m0c_reuse_against_m0b.py`
(conditionally skipped when the gitignored real outputs are not present
locally, mirroring `tests/test_m0b_declaration.py`'s own pattern), not
inside the runner — a fresh Modal container has no access to M0B's
downloaded outputs.

## 7. Feature catalog

The frozen nine-family catalog (`docs/GRAPH_CONTEXT_FEATURE_CATALOG.md`) is
reused verbatim, unchanged. Only context-dependent families need rebuilding
under R3_BOUNDED (see §6) — R1/R2 feature measurements are unchanged from
M0B and are re-derived only as part of the reuse proof, not as new
measurement in their own right.

M0C's step 6 does not reuse M0B's `R3_FULL_REEXPANSION` classification
labels as if they automatically apply to `R3_BOUNDED`. The five predeclared
labels (`NO_VARIATION`/`REDUNDANT`/`SIGNAL_PRESENT`/`HIGH_COST`/
`CANDIDATE_FOR_TRAINED_SCREEN`), unchanged rules from
`configs/m0b_regime_map.yaml#classification_labels`, are recomputed
mechanically against `R3_BOUNDED`'s own measured distributions.

## 8. Systems contract and invariants

Thresholds unchanged from M0B (`latency_factor: 4.0`,
`peak_rss_bytes_max: 14 GiB`). `U3_bounded`'s construction cost is expected
to be cheap — a plain array union over two already-materialised arrays, not
a graph walk — but is measured and reported on the systems contract's own
terms, not assumed cheap without measurement.

The eight required invariants (`configs/m0c_bounded_r3.yaml#invariants_required`):

1. `scored_R1 == scored_R2` — live, per query.
2. `oracle_R1 == oracle_R2`, bit-exact — live, per dataset.
3. `C3_M0C == C3_M0B` — post-hoc, aggregate-level, against M0B's filed JSON.
4. `U2 ⊆ U3_bounded` — live, per query.
5. `C3 ⊆ U3_bounded` — live, per query.
6. `|C3 \ Cq| ≤ 64` — live, per query (`overlap_audit.regime_set_invariants`,
   unmodified).
7. R3 candidate/headroom metrics match M0B's bit-exactly — post-hoc.
8. NODE_ROLE partition is exhaustive and mutually exclusive — inherited
   proof (`overlap_audit.node_roles`'s docstring argument plus the
   200-trial randomized sweep in `tests/test_overlap_audit.py`), not
   re-verified per query in M0C, exactly as M0B relied on the same proof
   unmodified.

A failure on any of the eight stops the stage. None is averaged away across
the sample.

## 9. Compute — filed from a real measurement

`configs/m0c_bounded_r3.yaml#compute.cost_ceiling_usd` is now `is_final:
true`, filed 2026-09-04 from the step-2 smoke run's real per-dataset
measurement, mirroring M0A/M0A.1/M0B's own two-stage discipline exactly:
smoke on 5 queries/dataset (done — all six datasets, including a retry of
`musique_clean` after the candidate-contract-flag fix) → derive a real
estimate → file a ceiling with a conservative margin → gate-check → launch
the 600-query headline run.

The point estimate is **$0.0553**, derived the same way M0B derived its own
$0.088: 6 datasets × 100 queries × (R1 feature + R2 build + R2 feature + A64
admission + U3_bounded build + R3_bounded feature) per-query cost at each
dataset's own real p99, plus each dataset's own real
`cold_start_compile_ms` once per container, plus the same 30s/container × 6
residual margin M0B's own updated estimate used. At the $0.634/h CPU rate
`docs/COMPUTE_LEDGER.md` documents, that is 0.087255 CPU-hours. The filed
ceiling is **$2.00**, ≈36× the point estimate — inside this repo's stated
30–40× convention, not carried forward from M0B's own $5.00 unchanged (full
derivation and rationale: `configs/m0c_bounded_r3.yaml#compute`).

M0C is confirmed dramatically cheaper than M0B for the structural reason
predicted: R1/R2/A64 construction is the same real work M0B already paid
for (re-derived here as the reuse proof, not skipped — so it is not free,
and its measured cost lands within noise of M0B's own smoke numbers), but
R3's expensive part — `TARGET_H1(Cq_struct)`'s context build and the feature
pass over it — is replaced by a cheap array union and a feature pass over a
strictly smaller (`U3_bounded ⊆ U3_full`, proved in §3) context. On
`hotpotqa_clean`, where M0B's full re-expansion was largest, the saving
shows up directly: `mainline_r3_bounded_feature` p99 is 115.0 ms against
M0B's filed `R3_FULL_REEXPANSION` feature p99 of 1,145.0 ms on the same
dataset — roughly a tenth of the cost, because the bounded context never
re-expands around the newly scoreable structural nodes.

Peak RSS across all six smoke containers: 3.66 GB (`2wiki_clean`) to 7.69 GB
(`metaqa`), comfortably under the 16 GB container ceiling (highest at 48%)
and closely tracking M0B's own filed figures for the same six datasets —
expected, since R1/R2/A64 are the identical work and `U3_bounded` is never
larger than M0B's own `U3_full`. No stop condition triggered; step 4
(headline launch) is cleared to proceed.

## 10. What this declaration does not authorise

Filing this document authorises step 1 only. It does not authorise: the
step-2 reuse-proof run, building `U3_bounded` for real, the context-
dependent diagnostic recomputation, invariant verification, the corrected
feature/regime map, or the trained-screen proposal. Each needs its own
artifact before the next runs, exactly as M0B required for its own steps
2–8.

Standing prohibitions, restated: no QLS fitting, no GNN fitting, no
`H2`/`H3`, no new candidate generator, no D-series work, E2 stays paused,
Package F stays sealed, no workspace migration, no test-split reads. Step 8
is a hard stop — no fitting or screening run is authorised without a
further, separate, explicit declaration picking up one of step 7's proposed
cells.

## 11. What actually happened

**Steps 1–3 complete.** Full results land in `docs/M0C_BOUNDED_R3_RESULTS.md`
once the headline run (step 4) returns; this section tracks progress as it
lands, matching `docs/M0B_REGIME_MAP_PROTOCOL.md`'s own convention.

**Step 1 (declare):** filed `configs/m0c_bounded_r3.yaml` and this protocol
doc, commit `4269f8b`.

**Step 2 (prove reuse) — one real bug, then verified clean.** The smoke run
(5 queries/dataset) surfaced a real defect on first Modal contact:
condensing M0B's per-dataset block into M0C's config silently dropped
`musique_clean`'s `pre_hop_metadata_v1` `candidate_contract_compatibility`
flag while keeping `2wiki_clean`'s, so `musique_clean`'s job failed
`validate_candidate_contract` with "Frozen baseline candidate contract does
not match." Fixed by restoring the flag and adding a permanent regression
test (`test_candidate_contract_compatibility_matches_m0b_for_every_dataset`
in `tests/test_m0c_declaration.py`) comparing every dataset's flag against
M0B's own YAML. `musique_clean`'s smoke job was re-spawned alone and
completed. Commit `a9f730f`.

With all six smoke outputs downloaded, `tests/test_m0c_reuse_against_m0b.py`
ran for real against M0B's own filed smoke data (not toy fixtures): 48/48
parametrized checks passed — bit-exact reproduction of R1 headroom and
candidate counts, R2 context-node counts, shared R1/R2 invariants, A64
admission (`C3_M0C == C3_M0B` on every family, admitted count ≤ 64 on
every query), R3 containment and gold-overlap classification, and
retrieval/structural node-role counts, across all six datasets. The guard
test (`test_at_least_one_stage_has_been_downloaded_and_checked`) confirms
this was a real check against real data, not a vacuous all-skip run. The
reuse claim in §6 is now verified, not just proved algebraically.

**Step 3 (cost) — filed.** See §9. Point estimate $0.0553, ceiling $2.00
(≈36×), `is_final: true`. No stop condition triggered.

**Steps 4–8:** not yet run.
