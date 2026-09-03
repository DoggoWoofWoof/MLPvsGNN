# M1A: one-seed trained feature screening -- protocol

**Status: `DECLARED_NOT_LAUNCHED`.** This document and `configs/m1a_feature_screen.yaml`
together authorise steps 1-2 of the eight-step plan below (file the
declaration, derive cells) and nothing else. No fitting happens on the
strength of this document.

## 1. What M0-M0C established, and what M1 adds

M0A through M0C were entirely zero-training. Their headline result, corrected
and committed as canonical in `docs/M0C_BOUNDED_R3_RESULTS.md` (commit
`c0f6402`): A64 admission determines which nodes become *scoreable*,
independent of how much graph context surrounds them; `R3_BOUNDED`
(`U3 = U2 ∪ A64`) isolates that eligibility repair from M0B's R3
re-expansion confound, bit-exact on recovered gold, statistically flat on
context size. GEOMETRY/SUPPORT/PATH/DIFFUSION/TOPOLOGY gain no *additional*
wide-context material moving R2 → `R3_BOUNDED` -- an availability finding,
not a learned-value finding, because M0C fits no model.

M1 is the first phase in this track that fits anything. It is not a rerun of
the D-series' catalog (`docs/GRAPH_CONTEXT_FEATURE_CATALOG.md`, frozen at
D10) -- that catalog was measured on exactly one dataset (`2wiki_clean`)
under exactly one regime (`CAND`, this track's R1). M1A is a **transfer
screen**: the same nine-family catalog, tested for the first time on
`hotpotqa_clean`/`metaqa`/`webqsp`, and for the first time at R2/`R3_BOUNDED`
on `2wiki_clean` itself. Nothing here compares against a GNN, and no GNN
result may influence QLS feature selection at any point.

## 2. The three frozen regimes

Canonical for all trained work from this point forward, verbatim:

```
R1 -- HISTORICAL                       scored = Cq   context = G[Cq]
R2 -- GRAPH REPAIR                     scored = Cq   context = U2 = TARGET_H1(Cq)
R3 -- BOUNDED STRUCTURAL SCOREABILITY  scored = C3 = Cq ∪ A64   context = U3 = U2 ∪ A64
```

`A64` is the exact, frozen M0B/M0C deterministic native-structural +64
admission rule -- never re-tuned, never regenerated with another algorithm,
in M1. `R3_FULL_REEXPANSION` (`TARGET_H1(C3)`) is cited from M0B only, as a
systems control. It is never trained on any dataset, in M1A or after.

R1 maps directly onto the existing `CAND` arm and R2 onto `TARGET_H1`
(`src/mp_retrieval/graph_context.py::ARMS`) -- both already wired into a
real trainer via `context_feature_store`. R3 does not: no existing code
builds `U2 ∪ A64` and feeds it into feature construction. That wiring is new
work (§8).

## 3. Frozen BASE

Identical composition and meaning in every cell:

- **seed identity** -- `distance_0`, `I[v ∈ Cq's seed set]`, col 0 of the
  frozen local block (`scripts/run_graph_context_d3.py::SEED_ID_COLUMN`).
- **dense reciprocal rank**, **SPLADE reciprocal rank** -- the RETRIEVAL
  family, reused from `src/mp_retrieval/linear_control.py::rank_feature_rows`
  / `RANK_FEATURE_NAMES` (exactly two entries: `dense_reciprocal_rank`,
  `splade_reciprocal_rank`), the frozen `(K+1)/(K+rank+1)` transform, `K=60`.
  **Not** `src/mp_retrieval/qls_v2_retrieval_prior.py` -- that module has
  the same nine-feature spirit but different column semantics and has never
  been wired into any trained run; using it here would silently substitute
  an unvalidated construction for the one the catalog's D4/D5 evidence is
  actually about.
- **retriever agreement** -- not part of `RANK_FEATURE_NAMES`. This is
  `scripts/run_graph_context_d4.py`'s own new-at-D4 column,
  `in_dense & in_splade` cast to float32 (`run_graph_context_d4.py:235-238`),
  reused verbatim from that construction, not from `linear_control.py`.
- **semantic branch** -- `src/mp_retrieval/qls_v2_semantic.py::SemanticHead`
  at rung **S3** (`cosine_qd`, `dot_qd_pct`, `mean_abs_diff`,
  `semantic_product`, `semantic_difference`; 1,536 learned parameters).

  **AMENDED 2026-09-04, by explicit user decision, before step 3's compute
  estimate.** This section originally proposed rung S2 (zero parameters) as
  BASE's primary semantic rung, flagged as an explicit judgement call for
  review. The user reviewed it and chose S3 instead, on causal rather than
  performance grounds: M1A asks whether a structural feature family adds
  value *once the ranker already has a competent semantic signal*. A
  zero-learned-parameter semantic rung under-constrains that question -- a
  structural family could look important partly because the semantic
  branch was artificially fixed, which would make a finding like "PATH
  matters on MetaQA" hard to interpret (PATH itself, or graph structure
  compensating for a deliberately weak semantic scorer?). Earlier work
  already showed semantic information can dominate on some datasets,
  MuSiQue especially -- screening structural families against a
  deliberately weak semantic baseline risks rediscovering that omitted
  semantic capacity as "graph value." S3 stays tiny relative to the
  historical ~213K-parameter QLS model, so this does not undermine the
  paper's efficiency thesis; it is a methodological choice for the
  *conditioning* baseline, not a promotion of S3 in an effectiveness
  experiment. S3 is shared identically across every arm and regime, so
  "frozen BASE, identical meaning across datasets" still holds -- BASE was
  already a learned ranker (the shared MLP head trains regardless), so
  there was no methodological purity being protected by forcing the
  semantic component specifically to stay parameter-free.

  **Both parameter counts are verified against the live implementation, not
  assumed from `RUNG_PARAMETERS`'s dict/docstring alone**: instantiating
  `SemanticHead(rung="S3")` and summing `p.numel()` over `.parameters()`
  gives exactly 1,536, all trainable; `SemanticHead(rung="S2")` gives
  exactly 0. See `SemanticHead.parameter_count()`
  (`qls_v2_semantic.py:178-179`) and the corresponding live-instantiation
  test in `tests/test_m1a_declaration.py`.

  **S2 is retained, demoted to a later minimality-ablation control** (not
  BASE, not multiplied through the 44-arm matrix -- that would double the
  matrix for a question M1A is not asking): once structural survivors are
  known, `final_survivor + S2` vs. `final_survivor + S3` on one or two
  representative cells tests whether the learned 1,536-parameter semantic
  diagonal/head earns its cost. Not authorised by this declaration.

For R3-only structural candidates (A64-admitted, never retrieval-scored):
`dense_reciprocal_rank = splade_reciprocal_rank = retriever_agreement = 0`,
and NODE_ROLE (§4) exists specifically so zero retrieval rank is never the
only signal distinguishing "structurally admitted" from "ranked last."

## 4. Feature catalog: reused verbatim, one family newly built

The nine families and their column ranges
(`docs/GRAPH_CONTEXT_FEATURE_CATALOG.md`, `FROZEN_AT_D10`) are reused
without modification. Per family, for M1A:

| family | columns | representation | status |
| --- | --- | --- | --- |
| RETRIEVAL, SEED | 10-12, 0 | -- | in BASE |
| GEOMETRY | 1-3 (col 0 already in BASE) | seed-distance one-hot | tested (§5) |
| SUPPORT | 4 | historical `seed_connections` | tested, historical form only (§6) |
| PATH | 5-7 | historical `paths_length_1/2/3` | tested, historical form only (§6) |
| DIFFUSION | 8 | `personalized_pagerank` | **excluded** -- NEGLIGIBLE on 2wiki/CAND (D7), no motivation elsewhere yet |
| TOPOLOGY | 9 | `common_out_neighbors...` | **excluded** -- NEGLIGIBLE, weakest family (D7) |
| PROVENANCE | -- | graph-construction axis | not a screening target |
| NODE_ROLE | new | `is_structurally_admitted` | tested, R3 only (below) |

**NODE_ROLE** has no surviving code -- D0b's one probe (`bridge_support`)
was a different, unrelated formula and was killed. M1A's encoding is one new
binary column: `1.0` for `v ∈ A64` (`STRUCTURAL_SCORED_CANDIDATE`), `0.0`
for `v ∈ Cq` (`RETRIEVAL_CANDIDATE`). `CONTEXT_ONLY` nodes are never scored
under any regime, so they never receive this feature -- a two-state, not
three-state, encoding from the ranker's point of view. This is not a
prohibited new feature: `src/mp_retrieval/overlap_audit.py::node_roles`
already proves the exhaustive, mutually-exclusive partition (200-trial
randomized sweep, reused unmodified through M0B and M0C); the column exposes
an already-proven set-membership fact, introducing no new formula and no new
graph computation. Under R1/R2 the scored set is always `Cq`, so this column
is identically zero -- NO_VARIATION, and NODE_ROLE arms are declared only
under R3.

## 5. SUPPORT/PATH: historical representation stays primary

D-series evidence is preserved, not reopened. D7 found SUPPORT and PATH
individually promising on `2wiki_clean`/`CAND` (+0.64/+0.82 R@5 over
`D6_BASE_13`); D8's corrected `distinct_support` only Pareto-matched
history; D9/D10's `branch_diversity` PATH replacement was **MIXED**
(R@5 −0.67, MRR +0.62 -- fails both `IMPROVES` and `PARETO-MATCHES`, per
`docs/GRAPH_CONTEXT_D10_RESULTS.md`'s own corrected verdict, not the
misleading `PATH DIVERSITY PARETO-MATCHES` string the code wrote into
`stage_d10.json`, which that document itself calls "the wrong reading...
in the direction that flatters the work"). M1A tests only the historical
representations. Neither replacement is promoted globally by D10's mixed
result, and neither is an arm here.

## 6. Cell/arm matrix -- step 2, derived here

Full per-dataset rationale lives in `configs/m1a_feature_screen.yaml`'s
`datasets:` block; summarised:

| dataset | validation queries | R1 | R2 | R3 |
| --- | ---: | --- | --- | --- |
| `squad_clean` | 26,063 | BASE only | -- | -- |
| `2wiki_clean` | 3,000 | BASE, +GEOM, +SUPPORT, +PATH | same | same + NODE_ROLE |
| `hotpotqa_clean` | 19,570 | BASE, +GEOM, +SUPPORT, +PATH | same | same + NODE_ROLE (**Priority 1**) |
| `metaqa` | 39,138 | BASE, +GEOM, +SUPPORT, +PATH | same | same + NODE_ROLE (Priority 2) |
| `webqsp` | 315 | BASE only | BASE only | BASE + NODE_ROLE (Priority 2) |

`musique_clean` is out of scope: its M0C 6b/6c pattern (1.00x `R3_BOUNDED`
growth, near-ceiling recovery, grouped with `squad_clean` as "deprioritized"
in M0C's own Step 7) does not supply any feature/regime condition
`squad_clean` doesn't already cover.

Validation-split sizes are the real canonical splits
(`src/mp_retrieval/complete_data.py::CompleteRetrievalDataset.split`,
confirmed against `outputs/sa_mlp_confirmation/*.json`), **not** M0A-M0C's
100-query diagnostic panel -- that panel is a deterministic-prefix slice of
this same split, not an alternate sample. `webqsp`'s 315 is the real
validation count; M0C's own "1,578 expected_queries" was the full
train+val+test manifest total, confirmed by direct read this session.

`squad_clean` gets one cell because its R2/R3_BOUNDED context growth is
1.00x over R1 in every M0C measurement -- there is no repair to isolate.
`webqsp` stays bounded ("cells justified by reachable headroom", per
instruction): M0C found `R3_BOUNDED` recovers 78 gold instances on
`webqsp`, all already reachable within `U2`, so it contributes zero
*additional ceiling lift* over R2 -- but promotion into the *scored* set
(`C3`) rather than mere context presence is a scoreability question, not a
ceiling question, which is exactly what the one `BASE+NODE_ROLE` arm tests.
`2wiki_clean`/`hotpotqa_clean`/`metaqa` get identical R1/R2 matrices
(GEOMETRY/SUPPORT/PATH) precisely because the catalog has never been tested
on the latter two, and never at R2 on any of the three -- this is new
transfer information, not a rerun of a settled result.

**44 arms are predeclared** (1 + 13 + 13 + 13 + 4), one seed each. Up to
**3 conditional interaction arms** are reserved (§7), for a ceiling of **47
seed-units**.

## 7. Conditional training, not power set

Individual family arms are fit first, always. An interaction arm
`BASE+X+Y` is added to a cell only if both `BASE+X` and `BASE+Y`
individually moved that cell's validation R@5 materially -- never
pre-declared before individual results exist. The only cells where this can
apply are `2wiki_clean`/`hotpotqa_clean`/`metaqa` at R3 (NODE_ROLE paired
against whichever of GEOMETRY/SUPPORT/PATH moved that same cell materially).
`webqsp`/R3 has only one non-BASE arm, so no interaction is possible there;
`squad_clean` and every R1/R2 cell have no NODE_ROLE arm to pair against.
The power set is never enumerated.

## 8. What is reused vs. newly written

Reused as-is: the trainer core
(`scripts/run_sa_mlp_confirmation.py::_build_model`, `_fit`, `_score_once`,
`validate_candidate_contract`), metrics aggregation
(`scripts/run_operator_screen.py::_metric_row`, `_aggregate_rows`), static
and local feature construction (`run_graph_context_d0b.py`,
`run_graph_context_d1.py`), seed identity (`run_graph_context_d3.py`),
retrieval columns (`linear_control.py`), the semantic branch
(`qls_v2_semantic.py`, code-complete but never before wired into a trained
run), R1/R2 regime arms (`graph_context.py`), and the NODE_ROLE partition
proof (`overlap_audit.py`).

Newly written, before step 4's smoke run: (a) a bounded R3 context builder
-- `candidate_expansion_v2.expand` for A64 plus `np.union1d` against `U2`,
matching M0C's own `U3_bounded` construction, wired into
`context_feature_store` where today only the six existing `ARMS` are
accepted; (b) a per-family column mask, since no existing code slices or
zeros specific catalog column ranges per declared arm
(`ExplicitFeatureMLP`'s `include_interactions/static/local` booleans are
SA_MLP's own three coarse arm-types, not this catalog's column ranges);
(c) the `is_structurally_admitted` column; (d) a runner looping
(dataset, regime, arm) for one seed, generalising
`run_sa_mlp_confirmation.py`'s own fit/score/telemetry loop from fixed model
identities to this file's column-masked arms; (e) reuse of the same timing
convention M0B/M0C's `feature_latency_ms.steady_state.p99` came from --
**resolved**: `_feature_ms()` in `scripts/run_m0b_webqsp_probe.py:125-138`
wraps the per-query `qls_local_features(...)` call in
`time.perf_counter()`; `_percentiles()` in `scripts/run_m0a_probe.py:52-62`
takes the resulting raw-millisecond list and reports `np.percentile` at
50/95/99 plus mean/max; "steady_state" means the raw list with index 0
(query-0's one-time Numba parallel-JIT compile) excluded, per
`R1_COLD_START_ATTRIBUTION_NOTE`/`STEADY_STATE_ONLY_ATTRIBUTION_NOTE` in
`scripts/run_m0c_bounded_r3.py`. The new M1A runner must time its own
per-arm feature-build calls the same way (perf_counter-wrapped, query-0
excluded from steady_state, `np.percentile` at the same three points) so
p50/p95/p99 stay comparable across M0B/M0C/M1A rather than silently
becoming a new, inconsistent measurement.

## 9. Pre-declared selection rule

Filed before any fit runs. Primary metric: validation R@5. Pareto
admissibility tolerance: ±0.25pp. Material regression threshold: >0.50pp,
unless uncertainty overlaps. R@1/R@20/MRR/FullCov@20 are secondary
diagnostics and never silently replace the primary metric. Among admissible
arms, prefer lexicographically: (1) lower uncached feature-build p95,
(2) fewer parameters, (3) lower peak train memory, (4) a deterministic tie
rule **not yet fixed** -- filed as an open item to resolve before step 5,
not to be decided post-hoc on an actual tie. GNN outcomes play no role in
this selection.

## 10. Metrics required per fit

R@1/R@5/R@20/MRR/FullCov@20; regime oracle/headroom and ceiling attainment
(mandatory on `webqsp`, so unreachable gold is never misread as a ranking
failure); parameters; train time; peak train VRAM/RSS; uncached
feature-build latency p50/p95/p99. Most of these map directly onto fields
`run_sa_mlp_confirmation.py` already writes per fit -- the feature-build
percentiles are the one new instrumentation requirement (§8).

## 11. Sampling and seeds

The entire canonical validation split per dataset, no subsampling -- the
same convention every prior trained phase (SA_MLP screen/confirmation,
D4-D10) has used, and distinct from M0A-M0C's 100-query diagnostic slice of
that same split. One seed (0) only, per explicit instruction; multi-seed
confirmation is reserved for whatever survives M1A/M1B as a Pareto
candidate, not for the screen itself.

## 12. Compute -- deferred to step 3

Not estimated in this document. A preliminary, non-binding order-of-magnitude
anchor: `docs/COMPUTE_LEDGER.md`'s E2 phase-confirmation measured 0.174
GPU-h per seed-unit *realized* (vs. 0.041 GPU-h pure-compute estimate,
≈4.2x) at a comparable full-validation-split scale; 47 seed-units against
that rate is ≈8.2 GPU-h, not a filed number, and does not account for this
file's much larger per-dataset validation splits (up to 39,138 queries on
`metaqa`) or the new R3-builder/column-mask code's uncosted overhead.

One correction to how this track has described the ceiling convention:
`configs/m0c_bounded_r3.yaml`'s compute section described "a ceiling
roughly 30-40x actual measured cost" as "this repo's own established
convention," citing Stage C's $5.15-ceiling-vs-$0.16-actual as the example.
Reading `docs/COMPUTE_LEDGER.md` directly this session shows that same
Stage C entry (L358-362) names its 32x cost ceiling a **mistake**
("approves more than it needs to"), not a convention -- the ledger's actual
practice, followed correctly by D0 onward, is to derive a ceiling from
measured-or-well-reasoned throughput with a *stated, reasoned* multiplier,
and to re-derive the cost ceiling from the same measured rate as the
CPU/GPU-hour ceiling rather than carrying an unrelated placeholder forward.
M1A's step-3 ceiling will follow the D0-onward practice, not the phrase
carried in M0C's own compute section.

## 13. The eight-step plan

```
1. file this declaration, derive cells        <- steps 1-2, this document
2. derive exact cells/arms from the M0C map    <- folded into §6 above
3. estimate compute before any launch
4. real smoke: one dataset x one cell x BASE, plus one feature arm
5. if gates pass, run the M1A one-seed screen
6. build the dataset x regime x feature-family trained-effect map
7. propose (not launch) M1B survivor expansion
8. hard stop
```

Step 6 classification definitions (predeclared): **PROMOTED** (admissible
and improvement exceeds the material threshold, uncertainty not overlapping
zero); **PARETO_MATCH** (admissible, improvement inside the tolerance band
either direction); **NO_EFFECT** (inside tolerance on every metric, no
systems advantage); **REGIME_SPECIFIC** (PROMOTED/PARETO_MATCH in exactly
one of R1/R2/R3 for a dataset, NO_EFFECT/HARMFUL in the others);
**DATASET_SPECIFIC** (PROMOTED/PARETO_MATCH on exactly one dataset at a
regime, NO_EFFECT/HARMFUL elsewhere); **HARMFUL** (regression exceeds the
material threshold, uncertainty not overlapping zero).

Step 7 proposes only -- mirrors M0C's own Step 7 discipline exactly. Step 8
is a hard stop: no GNN work, no further Modal launch beyond what step 5
itself authorises, begins without a new, separate, explicit authorisation.

## 14. Standing prohibitions (restated)

No GNN training. No tuning QLS against GNN results. No training
`R3_FULL_REEXPANSION`. No changing A64. No H2/H3. No new feature families or
new formulas beyond the frozen catalog. No reopening D-series conclusions.
No five-seed screen -- one seed only in M1A. No inspecting Package F. No
resuming E2. No workspace migration.

## 15. Open items before step 4

One item remains explicitly unresolved by this document and must be
settled before the smoke run, not silently defaulted inside runner code:

1. **The lexicographic tie-break rule's 4th tier** (§9) -- not yet fixed.
   Only matters if arms (1)-(3) tie exactly; not blocking for the smoke run
   itself (a single BASE + one feature-arm smoke has nothing to tie-break),
   so this must be fixed before step 5 (the real one-seed screen) rather
   than before step 4.

Resolved this session, no longer open: **which module M0B/M0C's
`feature_latency_ms.steady_state.p99` actually came from** (§8, §10) --
confirmed as `_feature_ms()` in `scripts/run_m0b_webqsp_probe.py:125-138`
(`time.perf_counter()` around `qls_local_features(...)`) reduced by
`_percentiles()` in `scripts/run_m0a_probe.py:52-62` (`np.percentile` at
50/95/99, query-0 excluded from steady_state). The new M1A runner reuses
this exact convention.

The semantic-branch rung choice (§3) was exactly this kind of flagged,
not-yet-open item -- reviewed by the user 2026-09-04, resolved as **S3
primary, S2 demoted to a later minimality-ablation control**, before step
3's compute estimate. No longer open.
