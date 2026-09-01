# QLS-v2 Development Protocol

**Status:** `DRAFT_AWAITING_REVIEW_NOT_FROZEN`
**Date:** 2026-09-02
**Governs:** all QLS-v2 work from first measurement through final confirmation
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md),
[`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md)

> **No QLS-v2 training may begin until this document is reviewed, frozen and
> tagged.** Nothing below has been executed.

## 0. The problem this protocol exists to prevent

The QLS-v2 effort is, by construction, an attempt to beat a comparator whose
results we have already seen. Without a discipline, the natural process —
adjust, re-evaluate on the same test set, keep what wins — produces a method
that appears to dominate and an evaluation that means nothing.

The discipline is:

```
Freeze QLS-v1 and its results
  -> use them to identify weaknesses          (audit: DONE, read-only)
  -> build QLS-v2 on development evidence     (this protocol, sections 2-5)
  -> freeze QLS-v2 completely                 (section 6)
  -> evaluate once on untouched confirmation  (section 7)
```

Each arrow is one-way. Section 8 lists what would invalidate the result.

---

## 1. Declared leakage, and what follows from it

**This must be stated in the paper.** The weakness audit read *aggregate
test-set* contrasts from Packages B, C and D (family-level and budget-level
GNN−QLS differences, and test-set latency percentiles). Those aggregates
informed which weaknesses were prioritized. That is information flow from the
test set into the design of v2, and pretending otherwise would be the exact
failure this protocol exists to prevent.

Three facts bound its severity, and all three are stated rather than relied on
silently:

1. **W1 and W3 are code-level findings.** The min-distance collapse, the
   edge-count seed support, the walk-based path counts, the provenance-blind
   edge array and the ~16 edge passes are properties of
   [`structural_features.py`](../src/mp_retrieval/structural_features.py). They
   were identifiable without any test metric, and would have been found by
   reading the implementation alone.
2. **W2's scoping used test aggregates.** That QLS's deficit grows with budget
   *on 2wiki and hotpotqa specifically* came from frozen test numbers. This is
   the most leakage-exposed part of the design (change C5 in particular).
3. **No hyperparameter, feature admission, or architecture choice has been made
   from test data, and under this protocol none may be.**

**Consequence for reporting.** The six-dataset test set is a *weakened*
confirmation surface for v2: it is unbiased with respect to v2's *selection* but
not with respect to v2's *problem framing*. Therefore:

- The six-dataset confirmation is reported as the **primary** v2 result, with
  this leakage declared in the paper text.
- A **cross-corpus confirmation on data never read at any stage** is required
  before the Pareto-dominance claim is stated without qualification. Package F is
  the natural surface for this.

**Open decision for the user, not for this document.** Package F was scoped as
the fresh untouched confirmation for the *v1* claim. It cannot serve as an
untouched surface for both v1 and v2 without being consumed twice. Whether F is
spent on v1, on v2, or split, is a research-scope decision. **Package F remains
unopened and this protocol makes no use of it.** The decision is flagged here so
it is made deliberately rather than by default.

---

## 2. Data discipline

Canonical splits are `train` / `validation` / `test`, already fixed per dataset
and unchanged from Packages A–E.

| Split | Use during v2 development | Use during confirmation |
|---|---|---|
| `train` | model fitting | model fitting |
| `validation` | **all** selection, all measurement, all ablation reading | seed-0 checkpoint selection only, as in v1 |
| `test` | **prohibited — no read of any kind** | read once, after the freeze tag |

The prohibition on `test` during development is absolute and includes: metric
computation, plotting, "just checking", per-query inspection, and any script that
loads test-split query metrics. It applies to the six frozen A–E analyses as
well: those files are test-set artifacts.

**Enforcement.** Development scripts must load splits through a wrapper that
raises on a `test` request unless an explicit `--confirmation-run` flag is passed,
and that flag must be unavailable until the freeze tag exists. This is the same
gating pattern already used successfully for E2, where
`configs/phase_confirmation.yaml` did not exist until the validation-only rule
had been committed.

**Development datasets.** Feature and architecture development runs on the
`validation` splits of all six datasets. Reporting a v2 variant's validation
number on a subset chosen after seeing the results is prohibited; the
development report covers all six or declares the exclusion in advance.

---

## 3. Order of work

The audit's §8 lists four measurements deliberately not computed. They come
first, because the design's mechanisms are hypotheses until they are.

**Stage 0 — validation-set diagnostics (read-only, no training).**

| # | Measurement | Decides |
|---|---|---|
| D1 | Per-query correlates of the v1−GNN gap against inference-safe covariates (connected-seed fraction, path redundancy, PPR concentration, hub exposure, component size) | whether the gap is structural at all, and which features to prioritize |
| D2 | Fraction of candidates with a degenerate local signature, by candidate budget | whether W2's proposed mechanism is real (gates C5) |
| D3 | Significance of the relational extraction advantage on validation | whether W1's magnitude survives a test (gates C6's motivation) |
| D4 | Operation-level profile inside `query_local_summary_ms` | which traversal produces the tail (gates the systems plan) |

All four run on `validation` only. **If D2 shows no degeneracy growth, C5 is
dropped** rather than retained on intuition; the same conditional applies to each
gate above. Committing these gates in advance is what makes them meaningful.

**Stage 1 — feature implementation and unit correctness.** Each of C1–C7
implemented behind a flag, with tests that check the feature computes what it
claims on hand-built graphs (a candidate at distance 1 from one seed and 5 from
four others must produce a different C1 vector than one at distance 1 from all
five — the audit's own counterexample becomes a test case).

**Stage 2 — ablation ladder on validation.** Section 4.

**Stage 3 — systems profiling and variant selection.** Owned by
[`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md); accuracy-neutral variants are
chosen on cost alone.

**Stage 4 — freeze.** Section 6.

**Stage 5 — confirmation.** Section 7.

---

## 4. Ablation ladder (validation only)

Reported for every rung: Recall@1/5/20, MRR, FullCoverage@20, trainable
parameters, and p50/p95 uncached latency. A rung that improves recall while
regressing p95 is not an improvement under the Pareto objective and must not be
presented as one.

| Rung | Configuration | Isolates |
|---|---|---|
| R0 | QLS-v1 as frozen | reference |
| R1 | v1 + C7 (percentile transforms) | normalization alone |
| R2 | R1 + C1 (multi-scale distance) | W1a |
| R3 | R2 + C2 (multi-seed support) | W1b |
| R4 | R3 + C4 (diffusion distribution) | W1a/W2 diffusion |
| R5 | R4 + C5 (candidate-local structure) | W2 |
| R6 | R5 + C3.branching | W1c cheap half |
| R7 | R6 + C3.disjoint | W1c expensive half |
| R8 | R7 + C8 (gated head) | architecture |
| R9 | R8 + C6 (provenance split) | typing — **only if a typed GNN comparator is run** |
| X1 | v1 features + C8 | how much of any gain is architectural, not informational |
| X2 | full v2 − each C, one at a time | leave-one-out for the final configuration |

**Fixed in advance:** the ladder order, the metrics, and that X1 and X2 are
reported regardless of outcome. X1 exists because "the gain came from the gate,
not the features" is the most likely reviewer objection and the most likely
truth; discovering it late would be worse than designing for it now.

**Per-budget reporting.** C5's motivation is budget-dependent, so R5 and X2 are
reported at candidate budgets 50/100/200/400, not only at the default.

---

## 5. Development-only selection rule

Fixed before any variant is trained; not revisable after seeing results.

1. **Selection statistic:** mean `validation Recall@5` across development seeds
   `{0,1,2}`. No other metric selects; the others are reported.
2. **Admission threshold:** a rung is admitted only if its mean validation
   Recall@5 exceeds the previous rung by **≥ 0.20 points** *and* its 95% paired
   interval across development seeds excludes 0. A change that is merely
   non-negative is **not** admitted — this is what prevents accumulating a dozen
   neutral features that collectively overfit validation.
3. **Cost veto:** a rung whose measured p95 latency exceeds the previous rung's
   by more than 10% is rejected regardless of recall, unless the systems plan
   supplies a bounded implementation that removes the regression.
4. **Tie-breaking**, in order: fewer parameters, lower p95, fewer features.
5. **C9 compact subset:** greedy backward elimination, removing the feature whose
   removal costs least validation Recall@5, stopping when cumulative loss would
   exceed **0.30 points** from the full set. The tolerance is declared here, in
   advance, and applies whatever the resulting subset turns out to be.
6. **Hyperparameters** (`learning_rate`, `dropout`, `temperature`, widths): a
   pre-declared grid, searched once, selected on validation Recall@5, and the
   full grid reported. No second pass.
7. **Seeds:** development uses `{0,1,2}`. Confirmation uses `{0,1,2,3,4}`.
   Development seeds are a subset by design so that confirmation adds unseen
   seeds rather than reusing exactly the development draw.

**Prohibited during development, explicitly:** consulting any Package A–E test
metric; selecting datasets, budgets, or edge families for reporting after seeing
outcomes; re-running the ladder with a changed threshold because the first pass
admitted too little; and adjusting the admission threshold, the cost veto, or the
C9 tolerance after they have been applied.

---

## 6. Freeze

Before any confirmation run, a single commit must contain:

1. `configs/qls_v2.yaml` — complete feature list, architecture, hyperparameters,
   the C6 admit/exclude decision with its justification, and the per-dataset
   direction declaration sourced from
   [`DATASET_GRAPH_PROVENANCE.md`](DATASET_GRAPH_PROVENANCE.md).
2. The development report with the full ladder, including rejected rungs and X1.
3. Measured trainable parameter counts per dataset, each verified ≤ the selected
   GNN's count (213,568 / 213,440 / 209,280 as applicable).
4. The confirmation analyzer, written and tested **against the config while no
   confirmation result exists** — the same discipline used for
   `analyze_phase_confirmation.py`.
5. A statement of which comparator the GNN side runs (untyped, or typed if C6 was
   admitted).

Then a signed tag, e.g. `qls-v2-protocol-v1`. Confirmation may not launch before
the tag exists. As with E2, the launcher should refuse to run if the config is
absent.

**Immutability.** After the tag, the v2 specification is fixed. A bug discovered
in confirmation is fixed and the confirmation **restarted from scratch** under a
new tag, with both attempts disclosed — not patched mid-run. (This is what was
done when the E2 launch failed on a missing runner argument: the fix changed no
condition, rate or seed, and the incident is recorded rather than erased.)

---

## 7. Confirmation

- **Once.** One evaluation on `test`, seeds `{0,1,2,3,4}`, all six datasets.
- **Paired** by seed and query against the same selected GNN, using the existing
  hierarchical CI machinery in
  [`scripts/analyze_linear_rank_structure.py`](../scripts/analyze_linear_rank_structure.py)
  (`_hierarchical_paired_ci`, `_holm`).
- **Holm scope** declared in the config before the run.
- **Every Pareto axis reported**: Recall@1/5/20, MRR, FullCoverage@20, cached
  operator latency, uncached p50/p95/p99, throughput, peak GPU memory,
  trainable parameters — for both methods, on all six datasets.
- **Every ablation from §4 reported**, including rungs that did not help.
- **No re-selection.** If v2 loses on an axis, that is the result.

**The dominance claim is stated only if every axis holds.** Partial dominance is
reported as partial dominance, naming the axes lost. A v2 that improves Recall@5
on two datasets and regresses p95 everywhere is a negative result for the Pareto
objective and will be written as one.

---

## 8. What would invalidate the result

Any of the following, if it occurred, must be disclosed and the confirmation
treated as void:

- reading `test` metrics before the freeze tag;
- changing the design, threshold, tolerance or grid after a confirmation run;
- selecting reported datasets, budgets or families post hoc;
- admitting C6 without a typed GNN comparator;
- fabricating direction features on an undirected dataset;
- exceeding the parameter budget;
- running confirmation from an untagged or modified config;
- any candidate-pool, candidate-hash, or CRAG modification.

## 9. Standing constraints unaffected by this protocol

- **Package E2 continues untouched to completion.** Its rates, analyzer, seeds,
  selected conditions and test protocol are not modified by anything here. No v2
  work may share, reuse, or interfere with its Modal capacity in a way that
  perturbs it.
- **Package F stays unopened.**
- **All frozen A–E results stay exactly as they are** — not rewritten, retuned,
  reinterpreted or deleted. QLS-v1 remains a reported method, not an
  embarrassment to be minimized: the paper's contribution includes the finding
  that a fixed query-local summary already matches message passing on 3/6
  datasets and on similarity-only graphs.
- **CRAG remains strictly read-only.**
