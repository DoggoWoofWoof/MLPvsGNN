# Experiment execution status

Last audited: 2026-08-31.

This is the authoritative operational stopping point. It records artifact
completeness and integrity only. No partial-seed metric and no incomplete E1
validation outcome was read or interpreted.

## Stopping condition

New Modal GPU allocation is blocked by the workspace spend limit:

```text
Workspace ac-1Zd8AkijYgSgLk37ju340f has exceeded its spend limit
```

The earlier ephemeral Modal apps are no longer active. Their completed and
partial checkpoints persist on the `message-passing-retrieval-data` Volume.
Do not relaunch B, C, or E1 until quota is restored. When quota returns, use the
existing idempotent launchers: they resume disjoint fingerprinted paths and skip
verified model seeds rather than retraining them.

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

Summary: **12 COMPLETE, 10 PARTIAL / RESUMABLE, 2 MISSING, 0 INVALID**
out of 24 new-family conditions. There are 154/240 verified model-seed
work units; **86 remain**. At condition-launch granularity, 12 conditions need
resumption or first launch.

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
| HotpotQA | P Q5/G3 | P Q5/G1 | P Q5/G1 | P Q5/G1 |
| SQuAD | P Q5/G1 | P Q5/G1 | P Q5/G0 | M |
| MetaQA | M | M | M | M |

Summary: **12 COMPLETE, 7 PARTIAL / RESUMABLE, 5 MISSING, 0 INVALID**
out of 24 conditions. There are 163/240 verified model-seed work units;
**77 remain**. At condition-launch granularity, 12 conditions need resumption
or first launch.

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

Summary: **52 COMPLETE, 3 PARTIAL / RESUMABLE, 41 MISSING, 0 INVALID**
out of 96 cells. There are 107/192 verified seed-0 model work units;
**85 remain**. At cell-launch granularity, 44 cells need resumption or first
launch.

E1 is **not compilable**. The selected crossover/end-point rates cannot be
frozen, so E2 five-seed test confirmation is not unlocked. Do not infer rates
from the completed datasets and do not inspect partial-cell validation values.

## Exact remaining GPU workload

| Package | Unfinished condition/cell launches | Remaining model-seed work units |
|---|---:|---:|
| B | 12 | 86 |
| C | 12 | 77 |
| E1 | 44 | 85 |
| **Total** | **68** | **248** |

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

## Resume order after quota restoration

1. Re-run `scripts/audit_modal_integrity.py` and retain this exact matrix as the
   pre-resume checkpoint.
2. Resume B, C, and E1 with the existing idempotent launchers. Do not delete or
   regenerate candidates, graphs, splits, feature caches, or partial records.
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

The full local suite passes **101 tests** with `PYTHONPATH=src`. Both audit
scripts also pass scoped Ruff checks and Python byte-compilation. The only
warnings are existing Torch Geometric deprecation warnings.
