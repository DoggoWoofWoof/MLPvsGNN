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

### Phase 2 — Two crossed frontiers, structural and semantic

Two things are unknown and they are not the same question. How much *structure*
retrieval needs, and how cheaply the *semantic* comparison can be done. Phase 2
runs both, crossed, on validation only.

> **The 40-feature catalog is a superset and an audit record. It is NOT the
> default model.** The primary frontier below admits **12–15 structural
> dimensions**, not 24 and not 40. Anything outside R0–R5 enters only if the
> frontier demonstrably fails, and that failure must be reported.

#### 2a. Primary structural frontier — target 12–15 dimensions

The rung order follows the audit's priority ordering, highest expected value
first, so that if the frontier terminates early it terminates having already
bought the most.

| Rung | Adds | Dims | Cum. | Priority |
|---|---|---:|---:|---|
| **R0** | retrieval prior only (Group A) — **no structure at all** | 0 | 0 | baseline |
| **R1** | independent seed support: `support@1`, `support≤2`, `support≤3` (C1–C3) | 3 | 3 | **VERY HIGH** |
| **R2** | rank-weighted seed support: `rw@1`, `rw≤2`, `rw≤3` (C5a–C5c) | 3 | 6 | **VERY HIGH** |
| **R3** | seed geometry: `min_seed_distance`, `mean_reachable_distance`, `unreachable_seed_fraction` (B1, B6, B7) | 3 | 9 | HIGH |
| **R4** | bounded diffusion: `h1`, `h2`, `h3` (D1–D3 **or** D4–D6, one encoding, chosen in Phase 1) | 3 | **12** | HIGH |
| **R5** | path diversity + induced topology: `shortest_path_multiplicity`, `unique_predecessor_count`, `seed_component_fraction` (C6, C7, E3) | 3 | 15 | MEDIUM |

R0 is the honest floor. It asks how much of the result is topological *at all*,
and A3's frozen evidence — 19 parameters, no embeddings, no adjacency, recovering
51.8 / 47.9 / 65.6% of the RRF→QLS gap — says the answer is "more than a reviewer
would guess."

**R1 before R2 is deliberate.** The two rungs share hop budgets and differ only in
whether supporting seeds are weighted by retrieval confidence, so the R1→R2 delta
isolates exactly one question: *does confidence in the supporting seed carry
information beyond the count of supporting seeds?* The weighting rule (`w(s) =
1/r(s)`, no free constant) is frozen in the catalog **before** any R2 number
exists.

**Registered prediction: R5 will be unnecessary.** We expect R4→R5 to fall below
the admission threshold and R5 to be dropped, leaving a **12-dimensional**
structure vector. Two independent reasons: (i) A2's `structural_summary` — the
source of the only demonstrated fixed-structure wins, +5.20 R@5 on webqsp and
+4.41 on metaqa — is built from query-local features in the R1–R4 tier with zero
path-diversity or component terms (`structural_controls.py:120-141`); (ii) R1–R4
are all read-outs of a *single* fused traversal while R5 needs an extra
union-find pass, so R5 must clear the admission bar while also paying more.
Recording the prediction is what makes it falsifiable — if R5 is admitted, that is
a real finding about path diversity and gets reported as one.

**Not in the primary frontier:** the seven v1 global static features, edge
provenance typing, and everything in the catalog's exclusion list. They stay
audit-only.

#### 2b. Semantic frontier — target 0 to 1,536 parameters

QLS-v1 spends **98,304 parameters** — 46.0% of the model — on a learned `768→64`
projection of `q` and `d`, before reading a single structural feature. musique
says that semantic path is doing real work and cannot simply be deleted. It does
not say the *projection* is how that work must be done.

| Rung | Semantic input | Learned params |
|---|---|---:|
| **S0** | none — ranks and RRF only | 0 |
| **S1** | + `cosine(q,d)` | 0 |
| **S2** | + `cosine`, `dot_pct`, `mean_abs_diff` — fixed reductions of `q⊙d` and `\|q−d\|` | 0 |
| **S3** | + `semantic_product = Σ w_i q_i d_i` and `semantic_difference = Σ v_i \|q_i − d_i\|` | **1,536** |
| *v1* | learned `768→64` projection on both sides | 98,304 |

The diagonal pair costs `2 × embedding_dim`; the projection costs
`2 × embedding_dim × projection_dim`. The ratio is the identity
`2DP / 2D = P = projection_dim = 64`, exactly, not an estimate.

**Both semantic branches are carried through Phase 2 and Phase 3.** Do not
collapse to one early. The scalar/light-semantic branch (S0–S2, zero semantic
parameters) and the compact-embedding-interaction branch (S3) are separate Pareto
candidates with genuinely different cost profiles, and musique is the reason:
scalar-only is predicted to lose ~11 points there, which would be fatal to a
universality claim if S0–S2 were the only branch carried. Equally, S3 is not
assumed to win — the v1 projection is included as a fifth rung so the comparison
is against the real incumbent.

**Reported for every S rung** against the v1 projection: validation R@1/R@5/R@20/
MRR, trainable parameters, peak training VRAM, training wall-clock, and inference
p50/p95/p99. Parameters and training cost are expected to move; latency may not
(see [`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md) §2.1 — the semantic path
is predicted to be bandwidth-bound, and that prediction is registered *before*
measurement precisely so it cannot be quietly discarded).

#### 2c. The cross

The two frontiers are crossed, not run in sequence. Grid: `{R0..R5} × {S0..S3}`
plus the `{R0..R5} × v1-projection` column as incumbent reference. Each cell gets
the full marginal-efficiency row of
[`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md) §8.

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

**Mandatory per-transition report.** Every R and S transition reports the full
marginal-efficiency row — `dR@1 dR@5 dR@20 dMRR`, `dp50 dp95 dp99`, `dCPU_RSS`,
`dtraining_peak_VRAM`, `dparameters` — as specified in
[`QLS_V2_SYSTEMS_PLAN.md`](QLS_V2_SYSTEMS_PLAN.md) §8. A rung with no such row is
not admitted whatever its R@5. `dR@5 / dp95_ms` accompanies each row as an
explanatory statistic and is never the selection objective.

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

### Phase 8 — Untouched external confirmation on Package F

**DECIDED 2026-09-02: Package F is reserved exclusively for the final QLS-v2
confirmation.** This allocation is recorded now, before any v2 measurement, so
that it cannot be chosen later to suit a result.

What that means, explicitly:

```
DO NOT open F.                DO NOT inspect F.
DO NOT run QLS-v1 on F.       DO NOT use F for feature selection.
DO NOT use F for architecture selection, normalization constants,
       hyperparameter tuning, or any Pareto decision.
```

F is opened **exactly once**, after all of the following have completed and been
frozen, in this order:

```
1  feature frontier      (Phase 2, both R and S)      -> frozen feature contract
2  computation frontier  (Phase 3)                    -> frozen bounded backend
3  minimal learner       (Phase 4, and Phase 5 if it triggers)
4  LODO transfer         (Phase 6, mandatory)         -> universality established
5  final freeze          (Phase 7)                    -> tagged, hashes recorded
6  THEN, and only then, Package F.  One shot.
```

If the F result is worse than development suggested, **that is the finding** and
it is reported. There is no second F evaluation, no "F with a corrected
hyperparameter", and no re-freeze that re-opens F.

**QLS-v1's role changes accordingly.** v1 is now **diagnostic and historical
evidence** drawn from the six existing datasets. It is not a candidate for F and
will not be run there. The frozen v1 and GNN results remain evaluation baselines
on Packages A–E exactly as they stand.

**Why F rather than the six development datasets.** Every one of the six has been
looked at — the weakness audit, the Package A decomposition, the musique finding,
the noise estimates behind `tau` in §6.2 all read frozen numbers from them. That
is legitimate diagnostic use, but it means a strong v2 result on those six can
always be challenged as having been shaped by what we saw. F has never been
opened by anyone, so a v2 result there is the one number in this project that
carries no such objection. Spending it on anything less than the final claim
would waste the only untouched evidence we have.

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

### 6.1 The selection rule — lexicographic, no weighted scalar

**There is no objective of the form `R@5 − λ · latency`.** A weighted scalar
invites the obvious challenge — *why that λ?* — and any answer is either arbitrary
or was chosen after seeing which λ produced the preferred winner. The rule below
has no free weight.

Maintain the **nondominated frontier** across all seven axes (§6). Then, to pick
the one candidate that goes forward:

```
STEP 1  ADMIT   keep every candidate whose mean validation R@5 across the six
                development datasets is within tau of the best such mean.
STEP 2  ORDER   among admitted candidates, minimise in strict order:
                  (1) uncached p95 latency        <- deployability
                  (2) trainable parameter count   <- the thesis
                  (3) peak training memory        <- training cost
STEP 3  BREAK   remaining ties: fewer features, then fewer structural rungs,
                then the lexicographically smallest feature-set identifier.
                Deterministic; never a judgement call.
```

Step 3 exists so the rule always terminates on exactly one candidate without a
human choosing.

### 6.2 The tolerance — PROPOSED FOR REVIEW, not yet frozen

> **`tau = 0.25 Recall@5 points`, applied to the mean across the six development
> datasets, on validation.**

This value is **proposed for the user's review before freeze** and is written
here so it is fixed *before* any v2 result exists. It was chosen against two
measured quantities from frozen artifacts, not picked for convenience:

**1. It sits at the noise floor of the selection statistic.** From
`outputs/sa_mlp_confirmation_analysis.json`, QLS-v1's seed-to-seed sample
standard deviation of test R@5 is:

```
2wiki 0.421   musique 0.513   webqsp 1.144
hotpot 0.198  squad   0.115   metaqa 0.047     (R@5 points, n=5 seeds)
```

The seed-level standard deviation of the *six-dataset mean* therefore lies in
**[0.224, 0.406]** points — 0.224 if seed effects are independent across
datasets, 0.406 if perfectly correlated. **`tau = 0.25` is at the lower end of
that band.** A tighter tolerance would force the frontier to prefer candidates
whose apparent advantage is smaller than the noise in the statistic used to
choose them, which is how validation overfit happens.

**2. It cannot concede the effect we are contesting.** The smallest GNN
advantage over QLS-v1 that survives Holm correction is **0.531 R@5 points**
(hotpotqa, Holm p = 0.0027; the other significant one is 2wiki at 1.443, Holm
p = 0.0205). `tau = 0.25` is **47% of that**, so the tolerance alone can never
hand back a difference we are trying to close.

**The risk, stated plainly.** If the selected candidate finishes within 0.25
points of the best candidate, and the final result then lands within 0.25 points
of the GNN, the tolerance is part of the story and **must be reported as such** —
including what the untolerated best candidate would have scored. That reporting
obligation is fixed here.

**Alternatives considered and their consequences**, so the reviewer can move the
number with full information:

| tau | vs noise band [0.22, 0.41] | vs the 0.531 effect | consequence |
|---:|---|---|---|
| 0.15 | **below** the floor | 28% | chases differences smaller than measurement noise |
| **0.25** | at the lower edge | 47% | **proposed** |
| 0.30 | inside | 56% | buys more cost reduction; concedes over half the contested effect |
| 0.50 | above the upper bound | 94% | could give away the entire hotpotqa effect |

**tau is not revisable after any Phase 2 result is observed.** If the measured
noise of the v2 selection statistic turns out to exceed 0.25, that is **reported
as a limitation**, not used to move tau — moving it afterwards is exactly the
result-dependent flexibility this protocol exists to prevent.

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

## 9. Freeze scope — what is fixed now and what is deliberately not

Freezing too little permits result-shopping. Freezing too much invents details
whose right form genuinely depends on measurements not yet taken. This section
draws the line explicitly.

### FROZEN as of this document (not revisable after results are observed)

| # | Item | Where |
|---|---|---|
| 1 | The scientific thesis and the claim being tested | [`QLS_V2_DESIGN.md`](QLS_V2_DESIGN.md) §0 |
| 2 | The **no-GNN constraint** and its AST import-contract test | §1 |
| 3 | The candidate feature catalog (40 features, five-plus-one groups) | [`QLS_V2_FEATURE_CATALOG.md`](QLS_V2_FEATURE_CATALOG.md) |
| 4 | The **rank-weighting rule** `w(s) = 1/r(s)`, no free constant | catalog, Group C |
| 5 | Phase 0 instrumentation: what is measured, at what percentiles | §4 Phase 0, systems §2 |
| 6 | Phase 1 diagnostic protocol | §4 Phase 1 |
| 7 | Phase 2 staged frontiers **R0–R5 and S0–S3**, their order and contents | §4 Phase 2 |
| 8 | The marginal-efficiency report required per transition | systems §8 |
| 9 | The admission threshold, cost veto and elimination tolerance | §5 |
| 10 | The **lexicographic Pareto selection procedure** | §6.1 |
| 11 | The tolerance `tau = 0.25` — **proposed, pending the review checkpoint** | §6.2 |
| 12 | **Package F allocation** — v2 final confirmation only, one shot | §4 Phase 8 |
| 13 | Data discipline, declared leakage, confirmation rules | §2, §3, §7 |
| 14 | The registered predictions | catalog §3 |

Item 11 is the single open item and is flagged as such: the *procedure* is
frozen, the *number* awaits the reviewer. It must be settled before Phase 2 runs,
not after.

### PROSPECTIVE — depends on Phase 0–2 findings, deliberately not frozen

| Item | What legitimately decides it |
|---|---|
| Phase 3 bounded-backend choice (`H`, truncated PPR vs plain diffusion, sketch vs online) | the Pareto experiment in systems §4; the rule that selects is frozen, the winner is not |
| Phase 4 exact learner width `w` and the interaction form | the surviving feature dimensionality, which Phase 2 determines |
| Phase 5 ranking objective — whether a listwise loss is used at all | triggered only if pointwise scoring is shown to be the binding constraint |
| Phase 6 LODO reporting detail | mandatory that it runs; its per-dataset presentation follows the transfer results |
| Phase 7 freeze artifacts and tag names | mechanical, follows the winner |
| Whether the sketch backend ships as a second implementation | its measured Pareto position |
| Diffusion encoding: magnitude (D1–D3) vs percentile (D4–D6) | one Phase 1 decision, made on validation before Phase 2 begins |

**The distinction that matters:** what is prospective is *which variant wins*.
What is frozen is *the rule that decides*. No selection rule, threshold,
tolerance, weighting or admission criterion appears in the prospective column —
that is what "no result-dependent flexibility in the core feature-selection
rules" means in practice.

---

## 10. Standing constraints

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
