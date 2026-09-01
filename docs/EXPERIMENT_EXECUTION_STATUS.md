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

## Package C — candidate budget — COMPLETE and FROZEN (2026-09-01)

Candidate construction remains frozen to equal-weight dense/SPLADE RRF with the
registered candidate hashes and RRF constant. No budget is selected on test.

**24 / 24 conditions COMPLETE, 240 / 240 model-seed work units verified, 0
PARTIAL, 0 MISSING, 0 INVALID** (`scripts/audit_modal_integrity.py`,
2026-09-01). All six datasets are complete at all four budgets, so the matrix
that previously tracked partial cells is retired.

Close-out performed in the registered order: integrity audit, fetch from the
volume, `analyze_candidate_budget.py` at the registered 1000 bootstrap
replicates and seed 20260831, `compile_package_c.py` for the joint
budget-and-ceiling report, then this freeze.

Reports: [CANDIDATE_BUDGET_RESULTS.md](CANDIDATE_BUDGET_RESULTS.md) and
[CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md](CANDIDATE_BUDGET_AND_HEADROOM_RESULTS.md).

### What a larger budget actually bought

Recall factors exactly as `attainment x ceiling`, so each budget step splits
into a ceiling term and a ranking term with no residual. The ordering is
ceiling-first and was fixed in code before any result was read.

Across the 36 dataset-step-model cells:

| Cut-off | Ceiling term > 0 | Ranking term < 0 | Ranking term > 0 |
|---|---:|---:|---:|
| R@1 | 30 / 36 | 24 / 36 | 12 / 36 |
| R@5 | 36 / 36 | 33 / 36 | 3 / 36 |
| R@20 | 36 / 36 | 36 / 36 | 0 / 36 |

Attainment at R@5 falls from budget 50 to budget 400 in **12 / 12**
dataset-model pairs, from -0.010 (SQuAD) to -0.056 (WebQSP).

Enlarging the candidate budget therefore did not improve ranking anywhere in
this grid. Every observed recall gain across budgets is a candidate-supply
effect, and both rankers convert a strictly smaller fraction of the reachable
ceiling as the pool grows. A raw recall increase across budgets must not be
reported as a reranking improvement.

### Where message passing separates from structure alone

Paired `GNN - QLS` R@5, five matched seeds, Holm-adjusted across datasets within
each budget. Significant at 0.05 in 7 of 24 cells:

| Dataset | Effect at budget 50 -> 400 | Holm-significant cells |
|---|---|---|
| 2Wiki | +0.16 -> +1.28 | 200, 400 |
| HotpotQA | +0.21 -> +0.70 | 200, 400 |
| MetaQA | +0.09 -> +0.02 | 50, 100, 200 |
| MuSiQue | +0.71 -> +0.60 | none |
| SQuAD | -0.08 -> +0.14 | none |
| WebQSP | -0.36 -> -0.34 | none |

The advantage is budget-dependent and dataset-dependent, not uniform. On the
multi-hop Wikipedia sets it appears only once the pool is large enough to
contain relational context and then grows with the budget. On MetaQA the
significant effects are real but an order of magnitude smaller than the
diagnostic gap between the ranker and its own ceiling. On WebQSP message
passing does not help at any budget. This is the ladder rung Paper 1 exists to
measure, and it is now measured rather than assumed.

These effect sizes are comparable across budgets within a dataset because both
models receive identical pools: the ceiling cancels in the paired difference.
Absolute levels remain incomparable across datasets without their ceilings.

## Package D — uncached online systems — RUNNING (launched 2026-09-01)

Gate justification, checked rather than assumed: `scripts/check_package_d_gate.py`
reported **OPEN (6/6 budget-400 conditions integrity-valid)** and exited 0. The
check validates, per dataset, the completion status, the dataset identity, that
the budget is 400, that no budget was test-selected, that all five seeds exist
for both models, and that each carries a checkpoint path and checkpoint hash. It
reads contract booleans only and never an effectiveness metric, so opening the
gate cannot be influenced by a result.

D depends on Package C alone. B and E1 are still running and D does not wait for
them.

Launched as six persistent server-side calls on
`message-passing-retrieval-online-systems` via `scripts/spawn_modal_jobs.py`,
one per dataset, not with `modal run --detach`.

D measures what the cached numbers elsewhere in this repository deliberately do
not. Three quantities stay distinct and must never be merged in reporting:

1. **Cached operator latency** — the model forward over an already-built
   candidate subgraph with the query-local summary already computed. This is
   what Package C reports.
2. **Uncached post-retrieval latency** — fusion, seed selection, graph
   induction, QLS structural computation, QLS scoring, GNN propagation and
   readout, charged on an unseen embedding with nothing precomputed.
3. **Raw-query end-to-end latency** — the above plus first-stage retrieval.

Per condition D records batch 1 and 16, p50/p95/p99, throughput, GPU and CPU
memory, cold start, cache and storage footprint, and the cache break-even point.
Break-even is compute-only: `build_ms <= repeats * (uncached_ms - cached_ms)`,
expressed in repeat servings of the same query, and it excludes per-query cache
storage because the frozen protocol does not measure that.

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
| B | 4 | 26 |
| C | 0 | 0 (CLOSED) |
| E1 | 16 | 34 |
| **Total** | **20** | **60** |

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
- **Package C is complete, compiled, and frozen.** Its budget-versus-ceiling
  decomposition and its paired `GNN - QLS` contrasts are final.
- B and E1 are not complete and must not be compiled into final paper claims.
- D is RUNNING: its gate was verified OPEN 6/6 and it was launched on
  2026-09-01. It is not compilable yet.
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

## Fetching results after a server-side run

Each package's Modal module downloads its own results, but only inside the
`local_entrypoint` that `modal run` invokes. Because long packages are now
submitted with `scripts/spawn_modal_jobs.py` so they survive client teardown,
that entrypoint never runs and no result reaches this machine. The analyzers
read local files, so a gate can close with every result safely on the volume and
still be uncompilable.

`scripts/fetch_modal_results.py <package>` closes that gap. It downloads only
conditions whose status is complete, skipping and naming the rest, so a partial
download cannot be mistaken for a finished package. It writes through a staging
file, so an interrupted download cannot leave a truncated file that later looks
like a result. Remote paths carry a per-dataset data-fingerprint directory that
the local layout does not, and the phase screen stores one directory per
axis-and-rate where the analyzer expects one directory per axis; both mappings
are derived from the discovered remote path. Use `--dry-run` to see what would
be fetched without writing.

## Package C close-out sequence

Run in this order once the C matrix reads 24/24. Nothing here interprets a
result until step 4, and no step selects anything on test.

```bash
python scripts/audit_modal_integrity.py
python scripts/fetch_modal_results.py candidate_budget
python scripts/analyze_candidate_budget.py
python scripts/compile_package_c.py
python scripts/check_package_d_gate.py
```

1. **Integrity.** Re-audit and require `INVALID: 0` for candidate_budget and 24
   COMPLETE conditions. Stop if either fails.
2. **Fetch.** Expect 48 files from 24 complete conditions and no skips.
3. **Compile.** `analyze_candidate_budget.py` produces the effectiveness table
   and the registered paired statistics: the paired optimizer-seed then shared
   query hierarchical bootstrap, the paired-seed t test, and Holm adjustment
   across datasets within each budget.
4. **Join.** `compile_package_c.py` reports every metric beside its per-budget
   ceiling and decomposes each budget step into ceiling movement and ranking
   improvement.
5. **Freeze.** Commit the results and tag before anything downstream reads them.
6. **Unlock D.** `check_package_d_gate.py` exits zero only when all six
   budget-400 conditions are complete and integrity-valid. Launch Package D
   through `scripts/spawn_modal_jobs.py`, not `modal run --detach`.

Package D does not wait on B or E1. Its only dependency is step 6.

## Package B and E1 close-out sequences

Package B, once its matrix reads 24/24:

```bash
python scripts/audit_modal_integrity.py
python scripts/fetch_modal_results.py edge_provenance
python scripts/analyze_edge_provenance.py
```

Compile and freeze before interpreting. All Package B graph families share the
frozen candidate set, so they share one ceiling per dataset: the absolute level
is an upstream constraint and only the *difference* between families is a
topology or ranking effect. The analyzer enforces the precondition for that
reading by requiring an identical `test_query_order_sha256` across every family.
The question to answer is whether QLS and the GNN benefit from genuine
relational topology or from embedding similarity reintroduced as kNN edges,
which is what separates `symbolic_b` from `knn_only`. No family may be preferred
because it reads better.

Package E1, once its matrix reads 96/96:

```bash
python scripts/audit_modal_integrity.py
python scripts/fetch_modal_results.py phase_screen
python scripts/analyze_phase_screen.py
```

The analyzer applies the already-frozen selection rule on validation only,
refuses any cell that computed test metrics, and stops at
`RATES_SELECTED_REQUIRES_PROTOCOL_COMMIT_BEFORE_TEST`. Write the selected rates
to the generated confirmation config, commit, and create a new freeze tag
**before** launching E2. Test outcomes must never be inspected while E2
conditions are still selectable.

## E2 explanatory covariates: what already exists

These are post-hoc explanatory covariates for interpreting E2, not
preregistered axes, so they are computed after E2 runs and none of them gates
its launch. Recorded now only so the work is known.

Already available from the headroom diagnostic and Package B/C outputs:
candidate ceiling, missing-gold rate, Dense/SPLADE disagreement
(`source_complementarity`), graph density and induced size
(`structural_context_all_queries`), and the shortest-hop distance buckets.

Not yet computed, and deliberately not built ahead of need: seed recall,
connected-seed fraction, path redundancy, PPR concentration, and hub exposure.

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
