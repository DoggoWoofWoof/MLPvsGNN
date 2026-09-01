# Experiment execution status

Last audited: 2026-09-02 (E2 in flight; QLS-v2 design phase opened).

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

Immediately after launch the workspace showed `edge-provenance` with 4 tasks and
`phase-screen` with 6, which saturates the roughly ten-container concurrency
cap, so D reported 0 running tasks. The six calls are queued server-side, not
dropped: that is the property spawned calls have and `modal run --detach` does
not. D will start as B and E1 free slots. An app sitting at 0 tasks while the
cap is full is expected and is not a reason to relaunch it.

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
**Superseded 2026-09-01.** E1 closed at 96/96 conditions, 192/192 seed-0
model work units, 0 MISSING and 0 INVALID under
`scripts/audit_modal_integrity.py`. The selected rates are frozen below.

## Package E1 — COMPLETE AND FROZEN

Integrity: 96 COMPLETE, 0 PARTIAL, 0 MISSING, 0 INVALID; 192/192 work units.
All 96 results were fetched (`fetch_modal_results.py phase_screen`, 96 files
from 96 complete conditions) before any analysis ran, because the local tree was
behind the Volume and the analyzer reads local files.

`scripts/analyze_phase_screen.py` applied the already-frozen rule and stopped at
`PHASE_SCREEN_VALIDATION_ONLY_ANALYZED`, writing `docs/PHASE_SCREEN_RESULTS.md`
and the generated `configs/phase_confirmation.yaml`. The rule reads
`best_validation_recall@5` from each cell's training record and nothing else; no
E1 cell computes a test metric at all, and the integrity auditor independently
refuses any cell whose `screen_contract.test_metrics_computed` is not `false`.

### Selected rates — the frozen E2 conditions

The rule keeps, per axis and per dataset, both endpoints plus any bracket where
the validation GNN−QLS gap changes sign (and any exact zero with its
neighbours). E2 runs the union across datasets, so every dataset runs every
selected level of every axis:

| Axis | Selected rates | Non-clean cells per dataset |
|---|---|---:|
| `degree_rewire` | 0.0, 0.10, 0.25, 0.50, 1.00 | 4 |
| `random_add` | 0.0, 0.10, 0.25, 0.50, 1.00 | 4 |
| `hub_injection` | 0.0, 0.10, 0.25, 0.50, 1.00 | 4 |
| `feature_mask` | 0.0, 0.25, 0.50, 0.75, 1.00 | 4 |

Every level of every axis survived selection. That is an outcome of the rule,
not a decision: sign changes occur at different rates in different datasets, and
the union of six datasets covers the grid. The clean rate is selected on every
axis but is **not** an E2 cell — the clean condition reuses the sealed five-seed
confirmation rather than being retrained, so E2 is 6 datasets x 16 non-clean
cells = **96 cells, 960 GPU work units** (2 models x 5 seeds).

### What the validation screen does and does not say

`docs/PHASE_SCREEN_RESULTS.md` holds the per-cell validation gaps. Two features
are worth recording now, both strictly as validation observations that E2 has
not yet adjudicated:

- **Sign changes are real and common.** Most dataset-axis rows cross zero at
  some rate. This is the registered crossover phenomenon rather than a nuisance,
  and it is why the rule selected so much of the grid.
- **`feature_mask` at 1.00 is strongly negative in all six datasets.** At total
  feature masking the seed-aware GNN loses heavily. Registered prediction P5
  expects feature masking with useful topology to move the gap *positive*, and
  several datasets do show positive gaps at intermediate masking before
  collapsing at 1.00. The shape is therefore non-monotone rather than simply
  contrary to P5, but nothing here may be reported as a result: these are
  single-seed validation numbers, and only the five-seed test confirmation can
  adjudicate P5.

No rate was chosen by looking at any of this. The rule is mechanical and was
frozen before E1 ran.

## E2 launch path — built 2026-09-01, gate opened and LAUNCHED 2026-09-01

E2 had no execution path at all. `scripts/run_phase_confirmation.py` ends with

```python
raise SystemExit("Use the frozen Modal phase-confirmation launcher")
```

and that launcher did not exist: there was no `modal_phase_confirmation.py`, no
spawn-registry entry, and the runner was referenced only by its own unit test.
`modal_confirmation.py` is a different package — it runs `run_confirmation.py`
for the sealed five-seed gate, not the phase-crossover cells. The gap would have
surfaced at the exact moment the screen closed.

`scripts/modal_phase_confirmation.py` now exists and is registered as
`phase-confirmation` in `scripts/spawn_modal_jobs.py`. Four properties matter
and each is covered by `tests/test_modal_phase_confirmation.py`:

- **It cannot launch before the gate.** `configs/phase_confirmation.yaml` is
  generated by `analyze_phase_screen.py` from the locked rule, never written by
  hand. The launcher loads it lazily and exits with an explanation while it is
  absent, so a premature launch cannot invent rates.
- **Rate zero is never a cell.** The merged rate set always contains 0.0, but
  the runner refuses `rate <= 0` because the clean condition reuses the sealed
  confirmation instead of being retrained. The launcher drops it.
- **Each cell points at its own screen result.** E2 reuses the seed-0 validation
  checkpoint from the matching screen cell, and the runner re-verifies dataset,
  axis, rate, fingerprint, `test_metrics_computed is False`, and training seed 0.
- **Every cell runs the registered five seeds.**

The container shape is read from `phase_screen.yaml`, the confirmation config's
ancestor, so the module imports before the gate; the app name is asserted
against the one the analysis generates.

### Pre-flight against real artifacts (2026-09-01)

The launcher was exercised end to end before the gate, using only the rates the
locked rule retains *unconditionally* -- clean plus each axis endpoint -- so
nothing in the check anticipates or depends on a screen outcome. A real screen
can only add bracketing rates to that floor. The simulated config was written to
a scratch directory and the module path was monkeypatched, so
`configs/phase_confirmation.yaml` was not created and the gate stayed closed.

- 24 cells built from the six datasets and four axes, with the clean rate
  correctly excluded from every one.
- All five registered seeds present on each cell.
- All six sealed `sa_mlp_confirmation` baselines carry the fields the launcher
  reads: fingerprint, baseline parameters, and the clean data, topology and
  feature cache paths.
- The data fingerprint each cell derives matched the fingerprint directory E1
  actually wrote on the volume, for all six datasets.
- All 96 E1 cell paths exist on the volume, including the four axis-endpoint
  cells per dataset that E2 will reuse for its seed-0 checkpoint.

The last two points matter most: they confirm the `screen_result` path E2
computes resolves to a real screen cell rather than merely looking plausible.

### Still missing downstream of E2 (not blocking the launch)

These are needed only after E2 produces results, and each is a known gap rather
than a discovered one:

1. ~~**No `scripts/analyze_phase_confirmation.py`**~~ — **closed 2026-09-01,
   written against the config while no E2 result existed**, which is the only
   order in which its choices cannot be steered by an outcome. It applies the
   registered contract rather than restating it: primary contrast
   `seed_aware_gnn_minus_sa_mlp_recall_at_5`, `_mean_std_ci` over the five
   seeds, `_hierarchical_paired_ci` over seed and query, and `_holm` across
   datasets within one axis and rate. Every cell is refused unless it asserts
   `test_selected_rate` false, the locked validation-only rule, and seed-0
   checkpoint provenance; a missing assertion is a refusal, not a pass. The
   packed metrics are checked against the recorded digest and query-order hash,
   because the result records a container-side path that does not survive the
   fetch and only the digest travels.

   Two further properties are asserted in the report itself. The clean rate is
   **not** an E2 cell: every axis reuses the one sealed five-seed confirmation
   as a shared origin rather than retraining a second, non-identical clean
   condition. And no axis touches candidate generation, so within a dataset the
   candidate pool and the Recall@K ceiling are identical at every rate — unlike
   Package C, **no E2 movement across rates can be a ceiling effect**, which is
   what makes the crossover claim a ranking claim.

   A sign change is only called a crossover when both endpoints are
   individually Holm-significant. Two cells that are each indistinguishable from
   zero can straddle zero by noise alone, and reporting that as a regime change
   would manufacture the paper's headline finding out of nothing. Fifteen tests
   in `tests/test_analyze_phase_confirmation.py` cover each refusal and that
   guard.
2. ~~**No `phase_confirmation` entry in `fetch_modal_results.py`**~~ — **closed
   2026-09-01.** E2 shares the screen's remote condition naming but also writes
   packed per-query metrics, which the paired statistics need, so a cell maps to
   `{dataset}/{axis}/rate_{key}/{filename}` — a directory — rather than to the
   screen's single flat file, which would have nowhere to put the second
   artifact.
3. ~~**No `phase_confirmation` entry in `audit_modal_integrity.py`**~~ —
   **closed 2026-09-01, before launch.** E2 is registered in the shared
   `PACKAGES` registry with a `gated_on` marker: while
   `configs/phase_confirmation.yaml` does not exist both auditors report
   `GATED / CONFIG NOT GENERATED` rather than inventing an expected matrix,
   which is the only behaviour compatible with the validation-only rule. Once
   generated, the expected matrix excludes the clean rate, so no permanent
   phantom shortfall is reported. E2 gets its own contract branch rather than
   reusing E1's: E1 must not compute test metrics, whereas E2 exists to compute
   them, so applying E1's clause would mark every valid cell INVALID. The E2
   branch instead verifies `test_selected_rate` is `false`,
   `selected_by_locked_validation_only_rule` is `true`, and
   `seed_zero_validation_checkpoint_reused_without_test_peeking` is `true`, and
   treats a missing assertion as a failure rather than as evidence of
   compliance. It also refuses any cell written at rate 0.0. Nine tests in
   `tests/test_audit_phase_confirmation.py` cover the gate and each clause.

**All three are now closed.** Package D remains the one gap: it is not yet in
the audit registry; its result is a
systems benchmark keyed by batch size with no per-seed checkpoint records, so it
needs a different verification shape rather than a registry line, and adding a
half-correct entry during a live gating audit was not worth the risk.

### E2 launched

`spawn_modal_jobs.py phase-confirmation` deployed
`message-passing-retrieval-phase-confirmation` and spawned 96 server-side calls
after, and only after, the selected rates were committed and tagged
`phase-confirmation-protocol-v1`. The dry run confirmed the same 96 cells, 16
per dataset, with no clean cell among them, and each cell's `screen_result`
resolving to a screen cell that exists. Modal showed the app saturating the
container cap alongside Package B's last condition.

### The first launch failed entirely and was relaunched

All 96 cells of that first launch died on live containers with

```
AttributeError: 'Namespace' object has no attribute 'projection_dim'
```

and wrote nothing at all: no results, no checkpoints, not even the output
directory. The launcher built a Namespace missing `projection_dim`, `layers`,
`dropout` and `temperature`.

Those four are read by `_build_model` in `run_sa_mlp_confirmation.py` rather
than by `run_phase_confirmation.py`'s own body, which is why the pre-flight
missed them: it verified that the arguments were built and that every path
resolved to a real screen cell, and that is not the same as verifying they
satisfy the runner's contract. The values were already in the frozen protocol;
the launcher simply did not pass them through, so no condition, rate or seed
changed and `phase-confirmation-protocol-v1` still describes the experiment
exactly.

It was found by querying a spawned call's status directly rather than by
waiting: the audit reported 96 MISSING with 0 PARTIAL and the app reported 0
running tasks, a combination that cannot mean "in progress".

The guard is now in place. `tests/test_modal_phase_confirmation.py` parses
`run_phase_confirmation.py` and `_build_model` for every `args.<name>` they read
and asserts the built Namespace supplies each one; removing `projection_dim`
again fails that test by name. Against the real generated protocol: 33 required
attributes, none missing across all 96 cells.

E2 was relaunched after the fix. That is not a relaunch of functioning work --
the first attempt left nothing behind, so the runners' idempotence had nothing
to resume and the second launch started clean. It is now confirmed running:
96 unique spawned calls and checkpoint directories appearing on the volume,
which means cells are past model construction, the exact point where the first
attempt died.

## Exact remaining GPU workload

Both columns were re-derived 2026-09-01 with
`scripts/audit_modal_integrity.py`, which verifies persisted checkpoints without
reading partial effectiveness metrics. **No condition is INVALID in any
package**, so nothing completed so far has to be recomputed.

| Package | Unfinished condition/cell launches | Remaining model-seed work units |
|---|---:|---:|
| B | 4 | 17 |
| C | 0 | 0 (CLOSED) |
| E1 | 11 | 21 |
| **Total** | **15** | **38** |

Re-derived 2026-09-01 after the Package C freeze: B is 20/24 conditions and
223/240 units; C is 24/24 and 240/240; E1 is 85 COMPLETE, 1 PARTIAL, 10 MISSING
and 171/192 units. Package D adds 6 dataset conditions, currently queued.

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

## Package D close-out sequence

Once all six datasets report `UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_COMPLETE`:

```bash
python scripts/fetch_modal_results.py online_systems
python scripts/analyze_online_systems.py
```

`online_systems` was missing from the fetch registry, the same gap that would
have left Package C uncompilable. It is registered now. Two shape differences
had to be handled: D writes one result per dataset rather than one per
condition, so discovery needed its own remote depth of three instead of the
four the other packages use, and the local layout is a flat
`outputs/online_systems/<dataset>.json`. A dry run against the volume was used
to confirm the existing depth-four packages still discover all 24 Package C
conditions and 48 files after the change.

Fetching a package that has not written anything yet now reports zero
conditions instead of raising: D's output root does not exist on the volume
while its calls are queued. Only that one case is tolerated, since reporting an
empty package when the real problem was an inability to look would be worse
than failing.

D was previously absent from `audit_modal_integrity.py`; it is **registered as
of 2026-09-01** with its own verification shape, described in the E2 section
above. That registration is what revealed D had already finished: without it the
package was running unmonitored and its completion was invisible.

## Package D — COMPLETE AND FROZEN

Integrity: 6 COMPLETE, 0 PARTIAL, 0 MISSING, 0 INVALID; 12/12 benchmarked model
paths. Fetched and analyzed to
`UNCACHED_UNSEEN_EMBEDDING_SYSTEMS_ALL_DATASETS_ANALYZED`; results in
`docs/ONLINE_SYSTEMS_RESULTS.md`.

### The efficiency claim inverts under the uncached boundary

This is the package's reason for existing and the result is not the convenient
one. Under warm caches, QLS-MLP's operator is 4x to 8x cheaper than the
seed-aware GNN's: 0.113-0.139 ms per query against 0.446-0.953 ms. Under the
uncached on-demand boundary, where the timed path must rebuild equal-RRF
candidates, seeds, induced topology and the query-local summaries themselves,
**QLS-MLP is slower than the GNN in 10 of the 12 dataset-batch cells** --
1.13x to 1.39x slower at batch 1, and at batch 16 either marginally slower or,
in two cells only (`squad_clean` 0.997 and `metaqa` 0.955), marginally faster.

The mechanism is visible in the break-even table's build column. QLS's build
prefix costs 2.2-3.0 ms per query at batch 1 against the GNN's 0.8-1.6 ms,
because the query-local summaries are themselves part of what must be
constructed on demand. What the cached comparison charges to a preprocessing
step, the uncached comparison charges to the request.

The three latencies must therefore stay separate, exactly as registered:
**cached operator latency is not uncached on-demand post-retrieval latency, and
neither is raw-query end-to-end latency.** Query encoding, Dense ANN lookup and
SPLADE index lookup are shared upstream and excluded from both arms, so none of
these numbers is an end-to-end system latency. No efficiency claim for QLS-MLP
may be stated without naming which boundary it holds at: on this evidence the
QLS advantage is a property of warm caching, not of the method.

The break-even figures are compute-only lower bounds. The frozen protocol
measures static-asset bytes but not per-query cache footprint, so storage is
excluded; a real cache would also pay for memory the protocol does not charge.

## Package B — COMPLETE AND FROZEN

Integrity: 24 COMPLETE, 0 PARTIAL, 0 MISSING, 0 INVALID; 240/240 work units.
Fetched (48 files from 24 conditions), analyzed to
`EDGE_PROVENANCE_ALL_DATASETS_ANALYZED`, and compiled against the frozen-union
ceiling into `docs/EDGE_PROVENANCE_AND_HEADROOM_RESULTS.md`. The compiler
verified that the candidate-contract hash Package B trained against equals the
hash the headroom diagnostic measured, for all six datasets, so the attached
ceiling describes these results rather than another pool.

The ceiling is identical across the families of a dataset by construction, since
only the edge family changes. It therefore explains none of the differences
below: the absolute level is an upstream candidate constraint no edge family can
move, and only the difference between families is a topology or ranking effect.

### The registered question: relations or re-injected embedding similarity?

`symbolic_b` is structural and NER edges -- genuine relational topology.
`knn_only` is the embedding-similarity edges alone. Relational minus similarity
on R@5, in points:

| Dataset | QLS-MLP | Seed-aware GNN |
|---|---:|---:|
| metaqa | +6.76 | +6.62 |
| hotpotqa_clean | +4.15 | +4.60 |
| webqsp | +2.66 | +5.37 |
| 2wiki_clean | +2.02 | +4.05 |
| musique_clean | +0.75 | +0.53 |
| squad_clean | -0.02 | +0.02 |

Relational topology beats embedding-similarity edges in five of six datasets for
both models. **Message passing here is not merely exploiting embedding
similarity reintroduced as edges.** On `squad_clean` the graph family makes no
difference at all -- every family lands within 0.1 points -- and that null is
reported rather than dropped.

This contrast is a difference of seed means. It carries **no p-value and no
confidence interval**, because the analyzer's paired statistics test QLS against
the GNN within a family, not one family against another. It must not be
described as statistically significant.

A quantity confound is possible: relational edge sets are larger than similarity
edge sets in every dataset. `full_union_c` rebuts it directly. It has the most
edges of any family and is nonetheless *worse* than `symbolic_b` on 2wiki
(-0.97/-0.20), hotpotqa (-1.23/-0.88) and metaqa (-0.42/-0.03). Adding kNN edges
to relational edges does not help and usually hurts, which is what registered
prediction P2 expects of unstructured neighbor noise, and is the opposite of
what a pure edge-count explanation predicts.

### Where message passing actually wins, and by how little

The GNN minus QLS contrast is Holm-significant in **11 of 30 dataset-family
cells**, and every significant effect is small -- between +0.2 and +2.3 points.

- 2wiki and hotpotqa: significant and positive in four of five families each.
- metaqa: significant in three, but at +0.19, +0.26 and **-0.14** points;
  `symbolic_b` is significantly *negative* there.
- webqsp, musique and squad: nothing significant in any family.

The pattern that matters is `knn_only`. It is the one family in which the GNN
never significantly beats QLS in any dataset, and in which its point estimate is
negative in three of six. The GNN's advantage, where it exists at all, depends
on relational edges being present; supplied only with embedding neighborhoods,
message passing does not beat a fixed query-local summary.

No family is preferred here for reading better. `symbolic_b` is not the best
family for the GNN everywhere -- on 2wiki and hotpotqa `full_union_c` and the
sealed multigraph produce larger significant gaps, and on metaqa `symbolic_b` is
the family where the GNN loses.

## Package B and E1 close-out sequences

Package B, once its matrix reads 24/24:

```bash
python scripts/audit_modal_integrity.py
python scripts/fetch_modal_results.py edge_provenance
python scripts/analyze_edge_provenance.py
python scripts/compile_package_b.py
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

`scripts/compile_package_b.py` performs that join and was verified against the
four already-complete datasets on 2026-09-01 with a shape-only check that
printed no metric value and wrote nothing into the repository. It reports every
family beside the shared ceiling with attainment at each cut-off, and puts
`symbolic_b` against `knn_only` in its own table so the provenance question is
answered rather than assumed. The ceiling is attached only after the Package B
candidate-contract hash is verified equal to the hash the headroom diagnostic
measured; a mismatch is refused, because a ceiling from another pool would
silently rescale every gap. All six datasets were confirmed to share that hash
between the two packages. The compiler also refuses a partial analysis, so a
report cannot be frozen with missing rows that would read as absent effects.

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

## E2 STALLED 2026-09-02 — workspace spend limit re-exceeded

**E2 is not corrupt and nothing is lost. It is blocked on billing, not on code
or data.**

Detected while checking E2 health during the QLS-v2 design phase: two audits
~40 minutes apart both reported 48/96 conditions with no movement, and
`modal app list` showed **0 running tasks** on
`message-passing-retrieval-phase-confirmation` (ap-cj2qvLjN99Vcr4ki5r22sU,
deployed 2026-09-01 22:03 IST). That combination — incomplete work plus zero
tasks — cannot mean "in progress", and is the same signature that exposed the
failed first E2 launch.

Confirmed by a minimal CPU-only probe at 2026-09-02 02:04 IST:

```
Workspace ac-1Zd8AkijYgSgLk37ju340f has exceeded its spend limit
```

E2 therefore ran roughly **four hours** after launch before container execution
was blocked. Volume reads still work; only compute is blocked.

### Exact state at the stall

| dataset | status |
|---|---|
| 2wiki_clean | **COMPLETE** 16/16 conditions |
| webqsp | **COMPLETE** 16/16 |
| musique_clean | **COMPLETE** 16/16 |
| hotpotqa_clean | 10 PARTIAL, 6 MISSING |
| metaqa | 16 MISSING (never started) |
| squad_clean | 16 MISSING (never started) |

Totals: **48/96 conditions complete, 568/960 model-seed cells, 38 missing,
10 partial.** The volume holds output directories for four datasets only;
`metaqa` and `squad_clean` have none.

The partial hotpotqa cells stopped mid-GNN: `sa_mlp` has all five seeds while
`seed_aware_gnn` has three or four. Example —
`hotpotqa_clean/degree_rewire/rate_0.10`:
`sa_mlp` seeds [0,1,2,3,4], `seed_aware_gnn` seeds [0,1,2].

### Why this is safely resumable

- The launcher resumes **disjoint fingerprinted paths** and **skips verified
  model seeds** rather than retraining them, so completed seeds are not redone.
- Every completed cell keeps its per-model-seed checkpoint on the Volume.
- **No protocol change is required or permitted.** The frozen
  `configs/phase_confirmation.yaml` and its tag `phase-confirmation-protocol-v1`
  still describe the experiment exactly; resuming runs the same rates, seeds,
  conditions and analyzer.
- No test outcome has been inspected. The analyzer has not been run.

### Integrity verified after the stall (2026-09-02)

`scripts/audit_modal_integrity.py` was run over the whole Volume after the block
was confirmed. E2:

```
COMPLETE            48
PARTIAL / RESUMABLE 10
MISSING             38
INVALID              0
completed_gpu_work_units 568 / 960     remaining 392
```

**Zero INVALID.** Every one of the 58 cells that has any output passed:

- `candidate_contract_verified: true` (58/58)
- `query_order_verified: true` (58/58)
- **exactly one** protocol SHA-256 across all 58 cells,
  `1f9b761bbad6fd0d6fccabee30defaba7c85904fa86beccd423b4d2464643976`, which
  matches the committed `configs/phase_confirmation.yaml` byte for byte.

That last check is the important one: it proves **no protocol drift**. Every cell
that ran, ran against the tagged configuration, so the completed work and the
work still to do belong to the same experiment.

The 10 partial cells are all HotpotQA, and all have the same shape — `sa_mlp`
complete at 5/5 seeds, `seed_aware_gnn` short by one or two:

```
hotpotqa_clean/degree_rewire/rate_0.10   qls 5/5   gnn 3/5
hotpotqa_clean/degree_rewire/rate_0.25   qls 5/5   gnn 4/5
hotpotqa_clean/degree_rewire/rate_0.50   qls 5/5   gnn 4/5
hotpotqa_clean/degree_rewire/rate_1.00   qls 5/5   gnn 4/5
hotpotqa_clean/hub_injection/rate_0.10   qls 5/5   gnn 4/5
hotpotqa_clean/hub_injection/rate_0.25   qls 5/5   gnn 4/5
hotpotqa_clean/random_add/rate_0.10      qls 5/5   gnn 4/5
hotpotqa_clean/random_add/rate_0.25      qls 5/5   gnn 3/5
hotpotqa_clean/random_add/rate_0.50      qls 5/5   gnn 4/5
hotpotqa_clean/random_add/rate_1.00      qls 5/5   gnn 4/5
```

Remaining work decomposes exactly: 38 missing conditions x 10 model-seed cells
= 380, plus 12 unfinished GNN seeds in the partials = **392**, which matches the
auditor's own count. There is no unexplained gap.

The other four packages re-verified COMPLETE with 0 INVALID at the same time:
edge_provenance 24/24, candidate_budget 24/24, phase_screen 96/96,
online_systems 6/6.

### Required action (external — cannot be resolved from this repository)

The workspace spend limit must be raised or reset before E2 can continue. Until
then **no relaunch is possible**, and none should be attempted: a relaunch under
the block would fail the same way and produce no artifacts.

Once execution is unblocked, resume with the registered launcher via
`scripts/spawn_modal_jobs.py` — **not** `modal run --detach` — and re-audit
before compiling anything.

### What must not happen

- Do not reduce the seed set, drop datasets, or trim rates to fit a budget.
  Those are frozen protocol parameters; changing them to fit a billing
  constraint would silently redefine the experiment.
- Do not compile or analyze a partial E2. `analyze_phase_confirmation.py`
  refuses an incomplete matrix by design.
- Do not treat the three complete datasets as a result. They are three of six.

## QLS-v2 design phase — opened 2026-09-02, GATED ON REVIEW

Packages A-E measure **QLS-v1**. They remain frozen and reported. A separate,
strictly read-only phase re-reads them and the v1 implementation to diagnose
where v1 loses information or costs too much, and specifies a v2 under a
revised thesis.

Nothing in this phase modified a frozen result, protocol, config, candidate
pool, candidate hash, or CRAG artifact. **No QLS-v2 model has been trained and
no v2 code has been written.**

### The thesis changed (2026-09-02)

The target is no longer "an MLP that beats a GNN". It is: most useful graph
information for retrieval ranking reduces to a small set of query-conditioned
structural statistics; expose them explicitly and a tiny feed-forward ranker
suffices. The scientific object is the **feature set**, not the architecture.

**Hard constraint: QLS-v2 uses no GNN in any part of the method** -- no teacher,
no distillation, no hidden representations, no GNN-generated labels or residual
targets, no learned message passing at training or inference. Frozen GNN results
are evaluation baselines only. This is to be enforced by an AST import-contract
test, not by documentation alone.

| Document | Contents |
|---|---|
| `docs/QLS_V1_WEAKNESS_AUDIT.md` | six defects W1-W6 with frozen evidence, plus the axes v1 already wins |
| `docs/QLS_V2_FEATURE_CATALOG.md` | 33 candidate features: formulas, costs, cacheability, failure modes, registered predictions |
| `docs/QLS_V2_DESIGN.md` | no-GNN constraint, seed-bitset computation, the ~1.4K-4.3K parameter learner |
| `docs/QLS_V2_DEVELOPMENT_PROTOCOL.md` | Phases 0-8, selection rule, mandatory LODO, freeze and confirmation |
| `docs/QLS_V2_SYSTEMS_PLAN.md` | tail bounding, bounded diffusion, sketch backend, Pareto table |

### The six defects

All are **information losses or unbounded costs -- none is a shortage of
capacity**:

```
W1  min seed distance, collapsed and bucketed
W2  seed_connections counts edges, not distinct supporting seeds
W3  path features are walks, not independent evidence paths
W4  graph provenance flattened
W5  max-normalization compresses useful mid-range values around hubs
W6  cold-query cost concentrated in the query_local_summary heavy tail
```

**Preserved narrower frozen findings, not to be generalized:**

- context-related GNN separation is primarily **2wiki + hotpotqa**, not universal;
- learned message passing does **not** obtain its advantage from similarity-only
  (kNN) topology -- the GNN never significantly beats QLS on `knn_only` anywhere.

### Three new groundings established this phase

1. **|S_q| <= 10** -- seeds are the union of dense top-5 and SPLADE top-5
   (`configs/sa_mlp_screen.yaml`). Per-seed statistics are cheap and a per-node
   seed bitmask fits one 16-bit word. No multi-word fallback is needed.
2. **Embedding width is 768**, so v1's q/x projection is 98,432 parameters --
   **46% of the 213,506-parameter model** -- before any structural feature is
   consumed. A scalar-only ranker deletes that mass outright.
3. **A3 is the decisive evidence that capacity is not the constraint.** The
   frozen 19-parameter linear model, with no embeddings and no adjacency,
   recovers 51.8% (WebQSP), 47.9% (HotpotQA), 65.6% (MetaQA) of the
   selected-RRF to QLS gap. Proposed v2 learner: **~1,395-4,291 parameters,
   50-153x smaller than the GNN.**

### Corrections carried explicitly rather than smoothed over

1. v1's features are already per-query max-normalized and `hub_degree_percentile`
   is already a percentile. **"QLS-v1 uses raw features" is false.** The defect
   is outlier compression (W5), so the fix is rank/percentile statistics.
2. The percentile decomposition **refines** Package D's frozen mean-based framing
   using the same artifacts. `ONLINE_SYSTEMS_RESULTS.md` is left exactly as
   frozen; both descriptions are true because the distribution is right-skewed.

### Naming hazard

The internal frozen key `sa_mlp` collides with published SA-MLP
(arXiv 2210.09609), which distills from a **GNN teacher** -- exactly what v2
forbids. The key must not be renamed (frozen hashes); the paper name is QLS-MLP
and must explicitly disambiguate. See `docs/TERMINOLOGY_AND_POSITIONING.md`.

### Declared leakage

The audit read test-set aggregates from Packages B, C and D. W1-W6 are
code-level findings and the A3 evidence is a Package A result, but W5's dataset
scoping used test aggregates. No hyperparameter, feature admission or
architecture choice has been made from test data, and the protocol forbids it.
The six-dataset test set is therefore a weakened confirmation surface; LODO
transfer and Package F carry the generalization claim.

### Gate

Implementation and training are blocked until
`QLS_V2_DEVELOPMENT_PROTOCOL.md` is reviewed, frozen and tagged. Stage-0
diagnostics (D1-D4, validation-only) are likewise deferred, because their
outcomes gate feature admission and running them before the gates are frozen
would defeat the discipline they enforce.

**E2 remains the QLS-v1 diagnosis and must not be used to choose any v2
hyperparameter.** Package F stays unopened until the v2 architecture, feature
set and implementation are frozen at Phase 7.

## Repository boundary

The original C-RAG repository remains strictly read-only. This audit read only
the standalone repository, its sealed local confirmations, and the standalone
Modal result Volume. No C-RAG file was created, changed, staged, committed, or
pushed.

## Local verification

The full local suite passes **237 tests** with `PYTHONPATH=src`, up from 137 on
2026-09-01: the additions guard the phase-screen resume path, the headroom
arithmetic and reachability buckets, the headroom runner, Package D's distinct
audit shape, E2's progress and integrity contracts, and E2's analyzer --
including the AST contract test that would have caught the failed first E2
launch. `mp_retrieval` is not installed into the environment, so the suite
must be run with `PYTHONPATH` set; plain `pytest` fails at import. Both audit
scripts also pass scoped Ruff checks and Python byte-compilation. The only
warnings are existing Torch Geometric deprecation warnings.
