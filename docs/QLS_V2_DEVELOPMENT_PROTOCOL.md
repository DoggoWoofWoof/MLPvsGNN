# QLS-v2 Development Protocol

**Status:** `DRAFT_AWAITING_REVIEW_NOT_FROZEN`
**Date:** 2026-09-02
**Governs:** all QLS-v2 work from first measurement through final confirmation
**Depends on:** [`QLS_V1_WEAKNESS_AUDIT.md`](QLS_V1_WEAKNESS_AUDIT.md),
[`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md),
[`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md)

> **No QLS-v2 implementation or training may begin until this document is
> reviewed, frozen and tagged.** Nothing below has been executed.

---

## 0. The two failure modes this protocol prevents

**Failure 1 — tuning against the comparator.** We are trying to beat a
comparator whose results we have seen. Without discipline, the natural process
(adjust, re-evaluate on the same test set, keep what wins) produces a method that
appears to dominate and an evaluation that means nothing.

**Failure 2 — accidentally compressing a GNN.** If any GNN-derived signal enters
the method — teacher, hidden states, generated labels, residual targets — the
thesis collapses into a distillation result. §1 makes this a hard gate.

The discipline:

```
Freeze QLS-v1 and its results          (done)
  -> identify weaknesses, read-only    (done: W1-W6)
  -> build QLS-v2 on development evidence   (Phases 0-6)
  -> freeze QLS-v2 completely               (Phase 7)
  -> evaluate once on untouched confirmation (Phase 8)
```

Each arrow is one-way.

---

## 1. Hard gate: no GNN in the method

Prohibited at every phase:

```
GNN teacher                          GNN-generated labels
GNN distillation                     GNN-generated residual targets
GNN hidden representations           learned message passing (train)
GNN logits or soft targets           learned message passing (inference)
```

Frozen GNN results are **evaluation baselines only**, read at Phase 8 for
comparison and never as a training or selection signal.

**Enforcement.** The v2 training entry point must not import
`build_seed_aware_message_passing` or any message-passing module, and a unit test
must assert this by AST inspection of the import graph — the same contract-test
pattern that now guards the E2 launcher's argument namespace. A prohibition that
is only documented is not enforced.

---

## 2. Data discipline

Canonical splits `train` / `validation` / `test`, unchanged from Packages A–E.

| Split | Phases 0–6 (development) | Phase 8 (confirmation) |
|---|---|---|
| `train` | model fitting | model fitting |
| `validation` | **all** selection, measurement, ablation | seed-0 checkpoint selection only |
| `test` | **prohibited — no read of any kind** | read once, after the freeze tag |

The prohibition is absolute: no metric computation, no plotting, no "just
checking", no per-query inspection. It covers the frozen A–E analysis files,
which are test-set artifacts.

**Enforcement.** Development scripts load splits through a wrapper that raises on
a `test` request unless `--confirmation-run` is passed, and that flag is
unavailable until the freeze tag exists — the same gating that kept
`configs/phase_confirmation.yaml` from existing until E1's rule was committed.

---

## 3. Declared leakage

**This must appear in the paper.** The weakness audit read *aggregate test-set*
contrasts from Packages B, C and D. That is information flow from the test set
into v2's problem framing.

Bounding it honestly:

1. **W1–W6 are code-level findings.** Every one is a property of
   [`structural_features.py`](../src/mp_retrieval/structural_features.py) and was
   identifiable by reading the implementation alone.
2. **The A3 evidence that motivates the whole strategy is a Package A result**,
   already published in our own frozen reports.
3. **W5's dataset scoping (2wiki/hotpotqa) used test aggregates.** This is the
   most leakage-exposed part of the design, and it is what Group E targets.
4. **No hyperparameter, feature admission, or architecture choice has been made
   from test data, and this protocol forbids it.**

**Consequence.** The six-dataset test set is unbiased with respect to v2's
*selection* but not with respect to its *problem framing*. It is therefore the
primary confirmation surface **with the leakage declared**, and the
leave-one-dataset-out transfer result (Phase 6) plus Package F (Phase 8) carry
the generalization claim.

---

## 4. The staged program

### Phase 0 — Measurement infrastructure

**No model improvement.** Build per-feature instrumentation so that every feature
family records:

```
build p50 / p95 / p99      storage bytes
CPU RSS                    GPU memory
feature dimension          cacheability class
```

Also complete the four diagnostics the audit deliberately left uncomputed
(validation only):

| # | Measurement | Gates |
|---|---|---|
| D1 | Per-query correlates of the v1−GNN gap vs inference-safe covariates | feature prioritization |
| D2 | Degenerate-signature fraction by candidate budget | Group E |
| D3 | Significance of the relational extraction gap | the §7 graph ablation |
| D4 | Operation-level profile inside `query_local_summary_ms` | the whole systems plan |

**These gates are committed in advance.** If D2 shows no degeneracy growth, Group
E is dropped rather than retained on intuition.

**Exit criterion:** every catalog feature has a measured build cost, and D1–D4
are reported.

### Phase 1 — Feature diagnostics

**No large model.** Use a linear scorer or a fixed tiny MLP, and evaluate each
feature *group* independently.

> **Goal: discover what information matters. Not: maximize score.**

This distinction is the phase's whole point. A group that adds little here is
information the retrieval task does not need — that is a finding, not a failure.

**Exit criterion:** per-group effectiveness and cost on validation, all six
datasets.

### Phase 2 — Incremental feature frontier

Staged additions, evaluated on validation only:

```
R0  retrieval features only          (Group A)
R1  + independent seed support       (C1-C5)
R2  + bounded seed geometry          (Group B)
R3  + path diversity                 (C6-C7)
R4  + fixed-depth diffusion          (Group D)
R5  + cheap static topology          (Group E)
```

Then **leave-one-group-out** ablations on the Pareto candidates, and backward
elimination to the minimum sufficient set.

For each rung record:

```
R@1 R@5 R@20 MRR FullCov@20
training time, training peak memory
inference p50 / p95 / p99
CPU memory, GPU memory
parameter count, feature-build time
```

**No arbitrary combinatorial feature search.** The ladder order is fixed here, in
advance, and R1 comes before R2 because the audit predicts distinct seed support
is the single highest-value addition — a registered prediction that the ladder
can falsify.

**Exit criterion:** a named minimum sufficient feature set with its frontier.

### Phase 3 — Bounded computation

For every expensive feature, compare exact against bounded:

```
exact iterative PPR   vs  H=1 / H=2 / H=3 diffusion / H=3 truncated PPR
walk counting         vs  bitset seed support
full traversal        vs  bounded 3-hop traversal
online computation    vs  precomputed BUDDY-style sketch
```

**Selection is cost-first among accuracy-neutral variants**: screen for
neutrality (within 0.10 R@5 of exact on validation), then take the lowest p95.
If nothing is neutral, the choice escalates to the Phase 2 admission rule. This
ordering is fixed now so a slow-but-slightly-better variant cannot be justified
after the fact.

**Exit criterion:** the `query_local_summary` tail is bounded, measured.

### Phase 4 — Minimal learner comparison

Only after the feature contract is known:

```
linear
tiny single MLP
two-branch residual MLP
two-branch + explicit crosses
```

**That is the entire search.** No 50-architecture sweep. Gating, attention and
transformers are excluded unless the four above all fail.

**Exit criterion:** the smallest learner within the effectiveness frontier.

### Phase 5 — Ranking objective (only if needed)

```
current loss  ->  pairwise  ->  listwise  ->  listwise + structural auxiliary
```

Any structural auxiliary loss must be **query-conditioned**. Do not assert
"connected nodes should have similar representations" — graph neighbours are not
necessarily relevant. A defensible positive relation is "same query, both
structurally supported by multiple high-confidence seeds."

**No GNN-derived supervision under any circumstance** (§1).

### Phase 6 — Universal transfer (LODO) — **mandatory**

True leave-one-dataset-out:

```
train on five datasets  ->  evaluate on the held-out sixth
```

repeated for all six folds. **No dataset ID.** Identical feature formulas,
transformations, architecture, loss and hyperparameters across folds. Development
and validation splits only until the Phase 7 freeze.

The question this answers is the one that makes the paper significant:

> Are these structural primitives **universal properties of retrieval**, or
> benchmark-specific statistics?

**A LODO failure is a reportable result**, not a reason to add dataset-specific
handling.

### Phase 7 — Freeze

One commit containing:

1. `configs/qls_v2.yaml` — final feature list, exact formulas, computation
   algorithm, architecture, loss, normalization, hyperparameters, and the
   per-dataset direction declaration sourced from
   [`DATASET_GRAPH_PROVENANCE.md`](DATASET_GRAPH_PROVENANCE.md).
2. The full development report: every rung, every rejected variant, LODO folds.
3. Measured trainable parameter counts.
4. The confirmation analyzer, written and tested **against the config while no
   confirmation result exists** (the `analyze_phase_confirmation.py` discipline).
5. The no-GNN import contract test (§1).

Then a signed tag, e.g. `qls-v2-protocol-v1`. **Development stops here.**

**Immutability.** A bug found during confirmation is fixed and confirmation
**restarted from scratch** under a new tag with both attempts disclosed — never
patched mid-run. (Precedent: the failed first E2 launch changed no condition,
rate or seed, and is recorded rather than erased.)

### Phase 8 — Untouched external confirmation

Only now open Package F. **One shot.**

---

## 5. Development-only selection rule

Fixed before any variant is trained; not revisable after seeing results.

1. **Selection statistic:** mean `validation Recall@5` across development seeds
   `{0,1,2}`. No other metric selects; all others are reported.
2. **Admission threshold:** a rung is admitted only if it exceeds the previous
   rung by **≥ 0.20 R@5 points** *and* its 95% paired interval across development
   seeds excludes 0. Merely non-negative is **not** admitted — this is what stops
   a dozen neutral features accumulating into validation overfit.
3. **Cost veto:** a rung whose measured p95 exceeds the previous rung's by more
   than 10% is rejected regardless of recall, unless Phase 3 supplies a bounded
   implementation that removes the regression.
4. **Tie-breaking**, in order: fewer features, fewer parameters, lower p95.
5. **Backward elimination tolerance:** stop when cumulative validation R@5 loss
   from the full set would exceed **0.30 points**. Declared here, in advance.
6. **Hyperparameters:** one pre-declared grid, searched once, selected on
   validation R@5, full grid reported. No second pass.
7. **Seeds:** development `{0,1,2}`; confirmation `{0,1,2,3,4}`. Development is a
   subset by design so confirmation adds unseen seeds.

**`ΔR@5 / Δp95_ms` is reported as an explanatory statistic** — it is expected to
make decisions obvious (if distinct seed support buys +2.0 R@5 for +0.02 ms and
PPR buys +0.15 for +1.3 ms, removing PPR is clearly right) — **but a single
scalar ratio is never the sole selection criterion.**

**Prohibited during development:** consulting any Package A–E test metric;
selecting datasets, budgets or families for reporting after seeing outcomes;
re-running a ladder with a changed threshold; adjusting the threshold, cost veto
or elimination tolerance after they have been applied.

---

## 6. Pareto evaluation

A candidate `m` is characterized by
`(effectiveness, latency, memory, training cost)`. `m` is **dominated** if
another candidate achieves:

```
>= R@5  AND  >= R@20  AND  >= MRR
AND
<= p95 latency  AND  <= memory  AND  <= parameters  AND  <= training time
```

Dominated points are **kept for audit** and excluded from final architecture
candidates.

Required frontier plots:

```
R@5 vs uncached p95              <- the primary systems plot
R@5 vs trainable parameters
R@5 vs peak training memory
R@5 vs training wall-clock
R@5 vs feature-build p95
ceiling-normalized attainment vs p95
```

The last is likely the fairest cross-dataset view, since candidate ceilings
differ substantially (webqsp 0.30 vs squad 0.98 at budget 50):

```
attainment = R@5 / R@5_ceiling
```

**Training efficiency is measured, never inferred from parameter count.** For
every final candidate record:

```
trainable parameters    training wall time    GPU-hours
peak GPU VRAM           peak CPU RSS          samples/sec
```

---

## 7. Confirmation rules

- **Once.** One evaluation on `test`, seeds `{0,1,2,3,4}`, all six datasets.
- **Paired** by seed and query against the frozen GNN baseline, using
  `_hierarchical_paired_ci` and `_holm` from
  [`scripts/analyze_linear_rank_structure.py`](../scripts/analyze_linear_rank_structure.py).
- **Holm scope** declared in the config before the run.
- **Every Pareto axis reported** for both methods on all six datasets.
- **Every ablation and LODO fold reported**, including those that do not favour v2.
- **No re-selection.** If v2 loses on an axis, that is the result.

**The dominance claim is stated only if every axis holds.** Partial dominance is
reported as partial, naming the axes lost. **Do not hide dimensions on which the
final method fails.**

---

## 8. What would invalidate the result

Disclose and treat the confirmation as void if any occurred:

- reading `test` metrics before the freeze tag;
- any GNN-derived signal entering the method (§1);
- changing design, threshold, tolerance or grid after a confirmation run;
- post-hoc selection of reported datasets, budgets or families;
- dataset identity or dataset-specific constants entering the feature map;
- fabricating direction features on an undirected dataset;
- running confirmation from an untagged or modified config;
- any candidate-pool, candidate-hash or CRAG modification.

---

## 9. Standing constraints

- **Package E2 continues untouched to completion.** It is the QLS-v1 diagnosis.
  Its rates, analyzer, seeds, conditions and test protocol are not modified, and
  **E2 must not be used to choose any v2 hyperparameter.** It is monitored
  separately.
- **Package F stays unopened** until the final v2 architecture, feature set and
  implementation are frozen at Phase 7.
- **All frozen A–E results stay exactly as they are.** QLS-v1 remains a reported
  method: the finding that a fixed query-local summary already matches message
  passing on 3/6 datasets, is never significantly beaten on similarity-only
  graphs, and already wins on GPU memory, is part of the contribution.
- **CRAG remains strictly read-only.**
