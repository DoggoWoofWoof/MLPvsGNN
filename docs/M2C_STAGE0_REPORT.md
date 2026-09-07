# M2C Stage 0 — structural conditioning for S4's blockers

**Status:** complete. Two verdicts, filed against the gate committed in `e637be6`
-- the amendment to the `a71a83f` declaration, and the commit that first put the
numeric thresholds in writing -- before any Modal job ran, and applied by
`scripts/evaluate_m2c_stage0_gate.py`, committed in `6f6f3d2` before any result
was fetched.

> ## `STOP_STRUCTURAL_M2C`
> ## `ADMISSION_CLOSED`

Nothing was trained. No test split was read. All four cells ran on the
development (fit) 80% of the validation split, so the 20% M2B's filed numbers
come from is untouched.

| | |
|---|---|
| declaration | `configs/m2c_s4_structural_conditioning.yaml` |
| protocol | `docs/M2C_S4_STRUCTURAL_CONDITIONING_PROTOCOL.md` |
| compute record | `docs/M2C_STAGE0_COMPUTE_RECORD.md` |
| results | `outputs/m2c_s4_structural_conditioning/stage0/*.json` |
| gate output | `outputs/m2c_s4_structural_conditioning/stage0_gate.json` |
| probe code | identical across all four cells (`c0afaa4` and `6f6f3d2` differ only in the gate applier) |

---

## The one-sentence result

Deterministic graph-derived directional geometry is defined for **0.99% to 9.09%**
of the candidates S4 scores, and where it is defined it does not order those
candidates usefully — so it cannot repair S4's blockers by ranking, and at a
matched +64 budget it does not move the K-aware ceiling either.

---

## 1. The exact residual × provenance matrix

Three residuals (`R0_RAW_QUERY_CONTROL`, `R1_LEGACY_DIRECTIONAL`,
`R2_SEED_SUBSPACE`) × three provenances (`G_STRUCT` = `structural_only`,
`G_KNN` = `knn_only`, `G_FULL` = `baseline_a_simple`) × three arms (`S4`,
`direction_only`, `S4_plus_direction_rrf`) × four cells. Zero training.

**S4 reference, on each cell's development panel:**

| cell | panel | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| `squad_clean`/R1 | 20 850 | 0.7164 | 0.9131 | 0.9654 | 0.8036 |
| `musique_clean`/R1 | 3 190 | 0.3758 | 0.7628 | 0.9073 | 0.8672 |
| `2wiki_clean`/R3 | 2 400 | 0.4390 | 0.8597 | 0.9689 | 0.9589 |
| `metaqa`/R3 | 31 310 | 0.4107 | 0.7566 | 0.8724 | 0.7241 |

**Directional coverage** — the fraction of scored candidates with at least one
eligible seed-edge displacement. It is a property of the graph and the seeds, so
it is identical across the three residuals within a cell, which is the arithmetic
check that the residuals are varying only the scoring rule:

| cell | `G_FULL` | `G_STRUCT` | `G_KNN` | queries with zero coverage (FULL / STRUCT / KNN) |
|---|---:|---:|---:|---|
| `squad_clean`/R1 | 0.0909 | 0.0691 | 0.0228 | 49 / 2 450 / 1 014 |
| `musique_clean`/R1 | 0.0568 | 0.0267 | 0.0315 | 3 / 313 / 11 |
| `2wiki_clean`/R3 | 0.0419 | 0.0215 | 0.0210 | 0 / 10 / 30 |
| `metaqa`/R3 | 0.0605 | 0.0511 | 0.0099 | 0 / 41 / 3 286 |

No residual was degenerate and none fell back to the raw query, on any cell —
`residual_states` is all-zero everywhere. The mechanism was fully exercised; it
simply has almost nothing to be exercised on.

## 2. Error-conditioned margin diagnostics

The load-bearing measurement. Population: queries where S4's top-1 is wrong and
at least one relevant candidate is scored. `directional_margin = dir(best
relevant) − dir(S4's top wrong)`.

**A margin exists only where both sides are covered.** At 1–9% coverage most
comparisons have an uncovered side, and those are counted, not averaged. (This
is the defect that the smoke run exposed: pre-masking uncovered candidates to
`−inf` made a two-uncovered comparison evaluate `−inf − (−inf)` and every mean
came back `NaN`. Fixed in `c0afaa4`.)

| cell | error population | best measurable fraction |
|---|---:|---:|
| `squad_clean`/R1 | 5 811 | 489 / 5 811 = **8.4%** |
| `musique_clean`/R1 | 699 | 72 / 699 = **10.3%** |
| `2wiki_clean`/R3 | 184 | 12 / 184 = **6.5%** |
| `metaqa`/R3 | 8 603 | 701 / 8 603 = **8.2%** |

**Mean margin over measurable comparisons, `G_STRUCT`:**

`f⁺` is the fraction of measurable comparisons with a positive margin.

| cell | R0 raw query | R1 legacy | R2 seed-subspace |
|---|---:|---:|---:|
| `squad_clean`/R1 | −0.0380 (f⁺ 0.336) | **+0.0932** (f⁺ 0.592) | −0.0261 (f⁺ 0.353) |
| `musique_clean`/R1 | +0.0257 (f⁺ 0.520) | −0.0214 (f⁺ 0.500) | **+0.0576** (f⁺ 0.660) |
| `2wiki_clean`/R3 | −0.0056 (f⁺ 0.667) | +0.0129 (f⁺ 0.556) | +0.0593 (f⁺ 0.667) |
| `metaqa`/R3 | +0.0021 (f⁺ 0.532) | −0.0414 (f⁺ 0.442) | +0.0032 (f⁺ 0.546) |

**The sign is not stable across cells.** The only residual with a positive margin
on `squad` is the legacy one; the only residuals with a positive margin on
`musique` are the raw-query control and the seed-subspace. **No residual is
positive on both blockers.** A mechanism whose direction of effect depends on
which cell it is measured in is not a mechanism.

Stratified by where the first relevant item sits — `squad_clean`/R1,
`R1_LEGACY|G_STRUCT`, the single most favourable combination in the whole probe:

| stratum | population | measurable | f⁺ | mean margin |
|---|---:|---:|---:|---:|
| rank 2–5 | 4 101 | 332 | 0.590 | +0.0861 |
| rank 6–20 | 1 091 | 68 | 0.618 | +0.1285 |
| beyond 20 | 619 | 14 | 0.500 | +0.0894 |

Positive in every stratum, and available on 8.1% / 6.2% / 2.3% of them.

## 3. Direction-only metrics

Ranking by directional score alone, uncovered candidates last:

| cell | best direction-only R@5 (combination) | S4 R@5 |
|---|---|---:|
| `squad_clean`/R1 | 0.1484 (R0 \| G_FULL) | 0.9131 |
| `musique_clean`/R1 | 0.0961 (R2 \| G_FULL) | 0.7628 |
| `2wiki_clean`/R3 | 0.0589 (R2 \| G_FULL) | 0.8597 |
| `metaqa`/R3 | 0.1109 (R2 \| G_STRUCT) | 0.7566 |

Worst across the board is `R1_LEGACY | G_STRUCT` on `squad` at R@5 = 0.0092.
The ordering of the three provenances tracks their coverage and nothing else:
`G_FULL` ≥ `G_STRUCT` ≫ `G_KNN` in 10 of the 12 cell × residual pairs, with
`G_STRUCT` edging ahead only twice and only marginally (`2wiki` under R1,
0.0203 vs 0.0201; `metaqa` under R2, 0.1109 vs 0.1058). Direction alone recovers
between 1% and 15% of what S4 recovers.

## 4. Fixed S4 + direction fusion

Reciprocal rank fusion at the project's frozen constant **60**
(`configs/candidate_budget.yaml`), equal weights, no sweep of either.

**All nine combinations damage both blocker cells catastrophically:**

| | ΔR@5 `squad` | ΔR@5 `musique` | ΔR@1 `squad` | ΔR@1 `musique` |
|---|---:|---:|---:|---:|
| best of nine | −56.25pp | −56.68pp | −56.40pp | −29.69pp |
| worst of nine | −65.92pp | −59.31pp | −66.88pp | −32.99pp |

Control cells regress by **45.80 to 66.87pp** on R@5, across all nine.

This is not a bug and it is not an artifact of the fusion constant. Equal-weight
RRF gives a near-uninformative list the same authority as S4's, and a list that
is arbitrary for ~93% of candidates is exactly that. The declared fusion is
unusable because the signal underneath it is unusable — and the protocol
forbids repairing that by tuning a weight, correctly, since a weight tuned until
the damage disappears is a weight tuned to zero.

- **Effectiveness: 0 of 9 combinations pass.** Neither condition (A) nor (B).
- **Protection: 0 of 9 combinations protected.**

## 5. The STRUCT / KNN / FULL result

For **ranking**, which is the new question — the M0A `STRUCT/KNN/FULL` result was
about admission ceilings and does not answer it.

`G_KNN` has the lowest coverage on three of four cells (0.99% on `metaqa`, with
3 286 of 31 310 queries at zero coverage) and the weakest direction-only metrics
almost everywhere. `G_FULL` has the highest coverage and the best direction-only
metrics. No provenance carries ranking signal the others lack; the ordering is
coverage, restated. This is consistent with — and independent of — M0A's finding
that `knn_only` recovered exactly zero golds at the ceiling.

## 6–7. A64 vs legacy-PF64 vs seed-subspace-PF64, and the exact ceiling deltas

Budget 64 on all three arms, on the symmetrised `structural_only` graph A64 is
defined on. 2 000-query admission panel per cell. **`recall_ceiling@5`** — the
K-aware ceiling that bounds recall@5, never `candidate_ceiling`.

| cell | Jaccard(legacy, subspace) | unique admitted A64 / legacy / subspace | unique **relevant** admissions | Δ`recall_ceiling@5` legacy | Δ subspace |
|---|---:|---|---|---:|---:|
| `squad_clean`/R1 | 0.779 | 4 102 / 7 808 / 7 728 | 0 / 0 / 0 | **−0.300pp** | **−0.500pp** |
| `musique_clean`/R1 | 0.832 | 3 417 / 4 199 / 4 197 | 9 / 10 / 11 | +0.071pp | +0.054pp |
| `2wiki_clean`/R3 | 0.839 | 8 430 / 10 028 / 10 161 | 1 / 1 / 1 | +0.013pp | 0.000pp |
| `metaqa`/R3 | 0.748 | 6 812 / 15 478 / 17 787 | 0 / 0 / 0 | **0.000pp** | **0.000pp** |

Two things are worth stating plainly.

**The directional arms admit far more distinct nodes and gain nothing.** On
`metaqa` the seed-subspace arm admits 17 787 distinct nodes against A64's 6 812 —
2.6× as many — and all three arms return `recall_ceiling@5 = 0.896575`, identical
to six decimal places. That is a direct replication of M0A's null under the
genuinely new residual, on a different cell, by a different code path.

**On `squad` the directional arms are worse than the blind control.** At a fixed
budget of 64, admitting different nodes means displacing A64's structural picks,
and the displaced picks were more valuable than the replacements.

## 8. The blocker required-repair calculation

`required_repair_to_guard = max(0, deficit_to_S3 − 0.50pp)` — the deficit less
M2B's per-cell tolerance. Neither the deficit nor the tolerance.

| cell | S4 − S3 deficit | must recover | best oracle Δceiling@5 | shortfall |
|---|---:|---:|---:|---:|
| `squad_clean`/R1 | −0.8824pp | **0.3824pp** | −0.3000pp (legacy) | −0.682pp |
| `musique_clean`/R1 | −2.5721pp | **2.0721pp** | +0.0708pp (legacy) | −2.001pp |

`musique`'s best arm delivers **3.4%** of what the guard requires. `squad`'s best
arm moves the ceiling the wrong way.

The amendment's correction stands and is now measured rather than argued: the
earlier claim that a perfect admission mechanism *could not* repair
`squad_clean`/R1 was too strong — 0.46pp of exposure exceeded the 0.3824pp
requirement, leaving a narrow path. That path is now closed by measurement, not
by arithmetic. **An oracle bound can rule a path out; it never rules one in**,
and both cells are ruled out.

## 9. Timing and storage

CPU only, 4 cores, 16 GB, no GPU — as the compute record declared.

| cell | predicted | actual | directional kernel p50 / p95 |
|---|---:|---:|---|
| `squad_clean`/R1 | 1 111.5 s | **314.2 s** | 7.25 / 11.14 ms per query |
| `musique_clean`/R1 | 851.1 s | **63.8 s** | 7.63 / 9.39 ms |
| `2wiki_clean`/R3 | 191.4 s | **43.6 s** | 6.98 / 7.85 ms |
| `metaqa`/R3 | 635.8 s | **471.4 s** | 6.97 / 8.09 ms |
| total | 2 789.8 s | **893.0 s** (32% of estimate) | |

The estimate was deliberately quoted at the larger of two possible pool sizes so
that a wrong assumption would overstate the bill; it did, by roughly 3×. Spend
was inside the $0.61 expectation and far inside the $2.00 ceiling. Two wasted
`2wiki` reruns (below) were each gated at an expected $0.04 and each actually ran
in under a minute.

Results: **131 504 bytes** across four files, against a 4 MB estimate. Written
only under `outputs/m2c_s4_structural_conditioning/stage0/`. Nothing under any
M2, M2B, M3, Package F or canonical-CRAG path was read for writing or modified.

**One incident, recorded because it nearly corrupted this report.** The first
`2wiki_clean`/R3 result was written by a smoke run before the NaN fix. Two
subsequent reruns each reported success, and each had its write to the result
volume **silently discarded** — the container saw its own bytes, committed, and
the old file survived, while brand-new files from the same batch persisted
normally. The stale artifact was detected only because its recorded
`source_commit` was the pre-fix commit. It was removed from the volume and the
cell rerun, and two guards were added: the runner unlinks the output path before
writing it, and `fetch` now reports each artifact's build commit and will refuse
a mismatch when the caller states which one it expects. Had the schema not
changed in the fix, this would have put a pre-fix result into the table above
wearing a fresh timestamp.

## 10. The two verdicts

They are independent by declaration, because ranking and admission are separate
mechanisms and Stage 0 could easily have settled one without the other.

### Ranking → `STOP_STRUCTURAL_M2C`

| condition | result |
|---|---|
| effectiveness (A or B, both blockers) | **FAIL** — 0 of 9 combinations |
| protection (controls within 0.50pp) | **FAIL** — 0 of 9 combinations |
| mechanistic 1: R2 ≠ R1 behaviourally | pass — max metric difference 0.019–0.137, admitted-set Jaccard 0.75–0.84 |
| mechanistic 2: margin meaningfully positive on ≥1 blocker | pass **on the letter only** |

All four are required. Two fail outright.

The second mechanistic condition deserves its qualifier. "Meaningfully positive"
was never quantified in the declaration, so the gate applier deliberately uses
the weakest defensible reading — more often positive than not, and positive on
average — and labels the pass as depending on a number nobody filed. Under that
reading it passes on both blockers, but by a *different residual on each*, and on
8.4% and 7.2% of their error populations respectively. Nothing in the verdict
turns on it: effectiveness and protection both fail, and the gate requires all
four.

Per the declaration: **do NOT train `S4-STRUCT-TRANSFORM`.** A semantic-only S4
repair may still be scientifically justified later; structural conditioning is
not. Code that runs is not evidence that a mechanism exists.

### Admission → `ADMISSION_CLOSED`

Closed on **both** blocker cells, each by comparing the arm's oracle
Δ`recall_ceiling@5` against that cell's `required_repair_to_guard`. `squad`'s
best arm is negative; `musique`'s delivers 3.4% of its requirement. The
seed-subspace residual — the one genuinely new variable in the phase — does not
reopen M0A's null, and on `metaqa` reproduces it to six decimal places while
admitting 2.6× the nodes.

---

## What Stage 1 would have been, and why it is not proposed

The declaration says: if advancement is justified, propose the minimum learned
Stage-1 matrix and its cost; do not run it. **Advancement is not justified**, so
no Stage-1 matrix is proposed. Proposing one anyway would be filing a plan for
work the evidence just closed.

The matched `S4-SEMANTIC-TRANSFORM` control remains mandatory if a learned
transform stage is ever opened on other grounds. That obligation is unaffected
by this result — it constrains how such a stage would have to be run, not
whether this one may be.

## What this does not say

- It does not say S4's blockers are unrepairable. It says **this** mechanism does
  not repair them. S4's deficit sits far inside ranking headroom that already
  exists; Track A was already the leading track and remains so.
- It does not say graph structure is useless to retrieval. It says
  *displacement-direction geometry over seed-incident edges*, at the coverage
  these graphs actually provide, carries no usable ranking signal for S4's
  mistakes.
- It does not touch M2, M2B, M3, Package F, canonical CRAG or E2. `SELECTED_S3`
  at 3 585 parameters remains M2B's frozen outcome; S4 remains a challenger that
  did not clear the guard, and this phase did not change that in either
  direction.

## `STOP_FOR_REVIEW`

All ten items the declaration asked for are above, and both verdicts are filed.
The phase stops here. Nothing further in M2C runs -- no learned transform, no
remaining ten cells, no additional seeds -- without a further dated amendment
that has read this report. `SELECTED_S3` stays M2B's frozen outcome, and the
gates in the declaration record exactly what was earned: Stage 0 ran, its gate
was evaluated, Stage 1 is not authorised.
