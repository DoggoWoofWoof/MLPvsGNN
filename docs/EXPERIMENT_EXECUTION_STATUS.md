# Experiment execution status

Last audited: 2026-09-01 (resumption in flight; see below).

This is the authoritative operational stopping point. It records artifact
completeness and integrity only. No partial-seed metric and no incomplete E1
validation outcome was read or interpreted.

## Execution state (2026-09-01)

The spend limit recorded on 2026-08-31 has been lifted. GPU allocation on
workspace `deepalimohapatra1973` was re-verified end to end by running the
registered launchers, not by probing: containers start, the data Volume mounts,
and training proceeds. The earlier stopping condition no longer holds and the
instruction not to relaunch B, C, and E1 is withdrawn.

B, C, and E1 have been resumed. All partial checkpoints on the
`message-passing-retrieval-data` Volume were preserved; the launchers resume
disjoint fingerprinted paths and skip verified model seeds rather than
retraining them.

Two execution hazards were found and fixed before resuming.

1. **The phase screen had no cell-level resume.** `scripts/run_phase_screen.py`
   would have retrained and overwritten already-complete, integrity-audited E1
   cells with fresh numbers. It now reuses a registered complete cell and
   refuses to overwrite a record written under a different frozen contract.
2. **`modal run --detach` does not survive client teardown.** It keeps only the
   last triggered function alive, so a package launched from an interactive
   session stops when that session ends; this silently killed four in-flight
   runs on 2026-09-01. Long packages are now submitted with
   `scripts/spawn_modal_jobs.py`, which deploys the app and spawns each job as a
   server-side call that outlives the client. This changes only submission, not
   what is computed.

Account rotation cannot substitute for quota here. The frozen data Volume exists
in a single workspace and the launchers open it with `create_if_missing=False`,
so rotating a volume-bound task replaces a quota error with a missing-volume
error. `experiments.py` now refuses to rotate those tasks and says so. Rotation
would only become useful if the Volume were replicated.

## Integrity audit

The read-only audit in `scripts/audit_modal_integrity.py` verified every
persisted model checkpoint that the result manifests register. For every
complete B/C condition it also verified the packed query-metric artifact. The
audit additionally checks:

- dataset fingerprint and selected model family;
- candidate-contract proof SHA-256 and equality with the sealed baseline;
- candidate ID/order, directly against the compatibility proof where one is
  required and through the frozen canonical contract otherwise;
- test-query order for B/C;
- seeds, loss/training configuration, graph family, and edge-set hash for B;
- budget, budget candidate-contract hash, and fixed RRF constant for C;
- axis, rate, perturbation seed, training seed, and the no-test-metrics screen
  contract for E1.

The local protocol-file hashes are recorded in the machine-readable audit, but
the remote result schema does not embed those file hashes. Scientific fields
from the frozen protocol are therefore checked field by field. The generated
audit JSON is operational output and remains git-ignored.

Audit result: **424/424 persisted model checkpoints verified; 0 invalid
conditions.** Transient Modal storage DNS errors are retried and are never
silently treated as artifact corruption.

### Status notation

- `C` = COMPLETE and integrity verified.
- `P Qx/Gy` = PARTIAL / RESUMABLE, with `x` verified QLS-MLP seeds and `y`
  verified seed-aware-GNN seeds.
- `M` = MISSING.
- Every B/C condition expects `Q5/G5`; every E1 cell expects seed 0 only,
  `Q1/G1`.

## Package B — edge provenance

The sealed A multigraph result is reused from the completed six-dataset
confirmation: **all six datasets are COMPLETE at Q5/G5 and require no GPU
work.** The matrix below covers the four newly trained families.

| Dataset | A-simple | Symbolic B | kNN-only | Full union C |
|---|---:|---:|---:|---:|
| 2Wiki | C | C | C | C |
| MuSiQue | C | C | C | C |
| WebQSP | C | C | C | C |
| HotpotQA | P Q5/G1 | P Q5/G0 | P Q5/G0 | P Q4/G0 |
| SQuAD | P Q3/G0 | P Q2/G0 | P Q4/G0 | P Q3/G0 |
| MetaQA | P Q1/G0 | P Q1/G0 | M | M |

Summary, re-derived 2026-09-01 with `scripts/audit_modal_integrity.py`:
**16 COMPLETE, 6 PARTIAL / RESUMABLE, 2 MISSING, 0 INVALID** out of 24
new-family conditions. There are 183/240 verified model-seed work units;
**57 remain**. At condition-launch granularity, 8 conditions need resumption or
first launch. The matrix above predates this re-audit and is kept as the
pre-resume checkpoint.

Package B is **not compilable** as a six-dataset package. Complete subsets must
not be promoted to a paper table while the registered package is incomplete.

## Package C — candidate budget

Candidate construction remains frozen to equal-weight dense/SPLADE RRF with the
registered candidate hashes and RRF constant. No budget is selected on test.

| Dataset | Budget 50 | Budget 100 | Budget 200 | Budget 400 |
|---|---:|---:|---:|---:|
| 2Wiki | C | C | C | C |
| MuSiQue | C | C | C | C |
| WebQSP | C | C | C | C |
| HotpotQA | C | P Q5/G1 | P Q5/G1 | P Q5/G1 |
| SQuAD | P Q5/G1 | P Q5/G1 | P Q5/G0 | M |
| MetaQA | M | M | M | M |

Summary, re-derived 2026-09-01 with `scripts/audit_modal_integrity.py`:
**21 COMPLETE, 3 PARTIAL / RESUMABLE, 0 MISSING, 0 INVALID** out of 24
conditions. There are 233/240 verified model-seed work units; **7 remain**.
Nothing is missing any more: every registered condition now exists on the
volume, and the three that are incomplete are mid-training rather than
unstarted. The matrix above predates this re-audit and is kept as the
pre-resume checkpoint.

Package C is the closest gate. When it closes, `scripts/check_package_d_gate.py`
decides whether Package D may launch.

Package C is **not compilable**. Package D is also locked: budget 400 is
complete only for 2Wiki, MuSiQue, and WebQSP; HotpotQA is partial and SQuAD and
MetaQA are missing. D may launch only when all six budget-400 checkpoint,
candidate-cache, and parity contracts pass.

## Package E1 — validation-only phase screen

No E1 test metrics have been computed. The screen uses the single registered
training seed 0. Rates for `degree_rewire`, `hub_injection`, and `random_add`
are 0.10/0.25/0.50/1.00; `feature_mask` rates are
0.25/0.50/0.75/1.00.

| Dataset / axis | Level 1 | Level 2 | Level 3 | Level 4 |
|---|---:|---:|---:|---:|
| 2Wiki / degree rewire | C | C | C | C |
| 2Wiki / feature mask | C | C | C | C |
| 2Wiki / hub injection | C | C | C | C |
| 2Wiki / random add | C | C | C | C |
| MuSiQue / degree rewire | C | C | C | C |
| MuSiQue / feature mask | C | C | C | C |
| MuSiQue / hub injection | C | C | C | C |
| MuSiQue / random add | C | C | C | C |
| WebQSP / degree rewire | C | C | C | C |
| WebQSP / feature mask | C | C | C | C |
| WebQSP / hub injection | C | C | C | C |
| WebQSP / random add | C | C | C | C |
| HotpotQA / degree rewire | C | C | C | P Q1/G0 |
| HotpotQA / feature mask | M | M | M | M |
| HotpotQA / hub injection | M | M | M | M |
| HotpotQA / random add | P Q1/G0 | P Q1/G0 | M | M |
| SQuAD / degree rewire | M | M | M | M |
| SQuAD / feature mask | M | M | M | M |
| SQuAD / hub injection | M | M | M | M |
| SQuAD / random add | M | M | C | M |
| MetaQA / degree rewire | M | M | M | M |
| MetaQA / feature mask | M | M | M | M |
| MetaQA / hub injection | M | M | M | M |
| MetaQA / random add | M | M | M | M |

For the first, third, and fourth axes, Levels 1–4 mean
0.10/0.25/0.50/1.00. For feature masking they mean
0.25/0.50/0.75/1.00.

Summary, re-derived 2026-09-01 with `scripts/audit_modal_integrity.py`:
**57 COMPLETE, 3 PARTIAL / RESUMABLE, 36 MISSING, 0 INVALID** out of 96 cells.
There are 117/192 verified seed-0 model work units; **75 remain**. At
cell-launch granularity, 39 cells need resumption or first launch.

E1 is **not compilable**. The selected crossover/end-point rates cannot be
frozen, so E2 five-seed test confirmation is not unlocked. Do not infer rates
from the completed datasets and do not inspect partial-cell validation values.

## Exact remaining GPU workload

Both columns were re-derived 2026-09-01 with
`scripts/audit_modal_integrity.py`, which verifies persisted checkpoints without
reading partial effectiveness metrics. **No condition is INVALID in any
package**, so nothing completed so far has to be recomputed.

| Package | Unfinished condition/cell launches | Remaining model-seed work units |
|---|---:|---:|
| B | 8 | 57 |
| C | 3 | 7 |
| E1 | 39 | 75 |
| **Total** | **50** | **139** |

These counts move while the jobs run and are a checkpoint, not a contract.

Resumed on 2026-09-01 as persistent deployed Modal calls: 12 Package B jobs,
12 Package C jobs, and 48 Package E1 cells, all restricted to the three
incomplete datasets (HotpotQA, SQuAD, MetaQA). Cells that are already COMPLETE
are reused by the idempotent runners rather than retrained.

A work unit is one model/seed training-and-evaluation record: QLS-MLP or the
selected seed-aware GNN. E1 has only seed 0. The 248 count is exact for the
registered protocols given the verified persisted state; it is not a claim
about equal wall-clock cost across datasets or models.

## What is compilable now

- The sealed six-dataset fairness confirmation and Packages A1/A2/A3 are
  complete and immutable.
- B, C, and E1 are not complete and must not be compiled into final paper
  claims.
- D remains gated on all six C budget-400 conditions.
- E2 remains gated on a complete E1 screen followed by deterministic rate
  selection, a committed configuration, and a new frozen protocol tag.
- Package F remains unopened.

## Candidate-generation headroom (companion layer)

Complete for all six datasets as of 2026-09-01. This is a diagnostic, not an
experimental condition: it modified no candidate pool and changed no frozen
hash. See `CANDIDATE_HEADROOM_PROTOCOL.md` and `CANDIDATE_HEADROOM_RESULTS.md`.

It changes how B, C, and the confirmation must be *read*, not what they compute:

- Every primary metric is now reported beside `min(p, K) / g`, its achievable
  ceiling. Pool coverage `p / g` is not an oracle Recall@K and must not be
  quoted as one.
- The Package C ceiling rises monotonically with budget on every dataset, so
  budget effects must be read against the per-budget ceiling rather than in
  absolute terms.
- Absolute metric levels are not comparable across datasets without their
  ceilings.

## Gate tooling verified before the gates open (2026-09-01)

Both downstream compilers were exercised while B/C/E1 were still running, so a
closing gate cannot open onto broken tooling. Neither check wrote into the
repository or touched a frozen artifact.

**Package D.** `scripts/analyze_online_systems.py` was run end to end on
contract-valid synthetic inputs for all six datasets: it compiles, renders, and
enforces its budget-400 fingerprint check. Cache break-even, which Package D
must report, had no implementation; it is now derived in the analysis layer from
measurements the frozen protocol already takes. It was registered before Package
D produced any timing, so its definition could not be chosen to suit a result.
The measurement contract is unchanged and still reads
`ONLINE_SYSTEMS_PROTOCOL_FROZEN_BEFORE_TIMING`.

**Package C.** `scripts/compile_package_c.py` implements the joint reporting the
budget protocol already required: it joins Package C effectiveness with the
headroom ceilings and decomposes each budget step into ceiling movement and
ranking improvement. It was verified against the real headroom outputs with
synthetic effectiveness numbers, which confirmed the join keys and, with
attainment held constant, attributed every gain to the ceiling and none to
ranking. The decomposition sums exactly to the observed change on all 18 steps.

`scripts/check_package_d_gate.py` decides whether D may launch. The Modal
launcher only checks the artifacts of the one dataset it is about to run, so on
a partially finished Package C it would start per dataset and fail per dataset
instead of refusing as a whole. The gate check reads all six budget-400
conditions and requires every one to be complete, five-seeded for both models,
not test-selected, and carrying a recorded checkpoint hash. It exposes contract
booleans only and never reads a metric, so it is safe to run while Package C is
still finishing. It exits non-zero while the gate is closed.

**Package E2.** `scripts/analyze_phase_screen.py` was run on a full synthetic
96-cell screen against the real sealed confirmations, read-only. All three
branches of the locked selection rule behave as specified, the analysis reports
`test_metrics_computed: false`, and it stops at
`RATES_SELECTED_REQUIRES_PROTOCOL_COMMIT_BEFORE_TEST`.

The gate that keeps E2 rate selection validation-only was previously untested:
nothing checked that the analyzer refuses a screen cell carrying test metrics.
It does refuse one, and that refusal is now a regression test, confirmed to fail
if the guard is removed. Cells trained on another seed and incomplete screens are
covered too, so an incomplete screen cannot be partially selected from.

## Resume order

1. Re-run `scripts/audit_modal_integrity.py` and retain the matrix above as the
   pre-resume checkpoint. **Done 2026-09-01** at condition granularity via
   `scripts/audit_modal_progress.py`.
2. Resume B, C, and E1 with the existing idempotent launchers. Do not delete or
   regenerate candidates, graphs, splits, feature caches, or partial records.
   **Done 2026-09-01** via `scripts/spawn_modal_jobs.py`.
3. Re-audit. Compile B/C/E1 only when their full registered matrices are
   COMPLETE and integrity-valid.
4. Launch D only after all six C budget-400 gates pass.
5. Generate the E2 selected-rate proposal from complete validation-only E1,
   review it, commit it, and tag it before any selected test evaluation.
6. Do not build a utility predictor unless E2 confirms reproducible help,
   neutral, and harm regimes.

## Repository boundary

The original C-RAG repository remains strictly read-only. This audit read only
the standalone repository, its sealed local confirmations, and the standalone
Modal result Volume. No C-RAG file was created, changed, staged, committed, or
pushed.

## Local verification

The full local suite passes **137 tests** with `PYTHONPATH=src`, up from 101 on
2026-08-31: 13 new tests guard the phase-screen resume path, 14 cover the
headroom arithmetic and reachability buckets, and 9 cover the headroom runner
end to end. `mp_retrieval` is not installed into the environment, so the suite
must be run with `PYTHONPATH` set; plain `pytest` fails at import. Both audit
scripts also pass scoped Ruff checks and Python byte-compilation. The only
warnings are existing Torch Geometric deprecation warnings.
