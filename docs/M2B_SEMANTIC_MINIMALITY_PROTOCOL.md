# M2B: semantic minimality -- protocol

**Status: `DECLARED_RECONNAISSANCE_COMPLETE_NO_FIT_AUTHORISED`.** This
document and `configs/m2b_semantic_minimality.yaml` together authorise
declaration and reconnaissance, both of which are complete. **No fit is
authorised, including the smoke cell.** Execution requires a further dated
amendment after the review this phase stops for.

## 1. What M2 settled, and what is left

M2 froze one structural schema and showed it clears the admissibility floor
in every declared cell. `configs/m2_qls_v2_freeze.yaml#selected` records the
freeze:

```
M2_QLS_UNIVERSAL_SELECTED

QLS-UNIVERSAL structural schema =
    RETRIEVAL + SEED + NODE_ROLE + SUPPORT + PATH

NODE_ROLE:  R1/R2 = constant zero
            R3    = structural-admission indicator

semantic rung entering M2B:  S3
current total trainable params:  3,585
```

The claim that result licenses, in the wording it must be written in: **the
universal schema was Pareto-admissible across all six development datasets
and all 14 declared regime cells.** Not "improves every dataset", and the
+1.776pp macro is not a robust universal gain -- `webqsp` contributes
+9.881pp of it against +0.065pp to +0.376pp from each of the other five. The
verdict survives its removal (five-dataset macro +0.155pp, worst cell
-0.026pp), which is what it actually rests on.

What M2 did not settle is the semantic half of that model. Of 3,585 trainable
parameters, **3,072 are semantic** -- 86% -- and of 14 input columns, 5 are.
The structural claim this track exists to make is about the other 14%. M2B
asks the obvious follow-up:

> With the M2 structural representation frozen, what is the minimum semantic
> representation that remains Pareto-admissible across all six development
> datasets?

Only the semantic representation varies. `R1/R2/R3`, `A64`, the scored and
context sets, the retrieval prior, seed identity, `NODE_ROLE`, `SUPPORT`,
`PATH`, candidate normalization, the loss, the optimizer and every training
hyperparameter are exactly M2's. **Feature selection is not reopened.** M2
answered that question, and reopening it here would make M2's verdict
retroactively conditional on a later phase.

## 2. The three candidates, and why they were counted rather than quoted

| rung | implementation | semantic cols | semantic params | scorer width | scorer params | **total** |
|------|----------------|--------------:|----------------:|-------------:|--------------:|----------:|
| S2 | `SemanticHead(rung=S2, dim=1536)` | 3 | 0 | 12 | 449 | **449** |
| S3 | `SemanticHead(rung=S3, dim=1536)` | 5 | 3,072 | 14 | 513 | **3,585** |
| S4 | `ProjectionSemanticHead(dim=1536)` | 258 | 196,608 | 267 | 8,609 | **205,217** |

Every number in that table came off a live `nn.Module` via `numel()`, from
`scripts/m2b_semantic_audit.py`, at the frozen 1536 width with
`precomputed_width=9` and M2's own dropout and temperature. None was
transcribed. The audit refuses to report anything at all if its S3 row does
not reproduce the 3,585 that M2's 15 completed fits actually ran under -- the
one row with fifteen fits of evidence behind it is the row that licenses
trusting the other two.

### 2.1 The 98,304 that is not this track's number

`qls_v2_semantic.V1_SEMANTIC_PARAMETERS` is 98,304, and planning prose around
this track has repeated "the historical projection is about 98K" for months.
It is computed as `2 * EMBEDDING_DIM * 64` from that module's
`EMBEDDING_DIM = 768` default.

**This track's payload is 1536-dimensional.** `run_m2_qls_v2_freeze.py`
refuses to start against anything else. Instantiated against the real payload,
the same projection holds **196,608** semantic parameters -- exactly twice --
and S4 totals 205,217, not ~98K.

Writing 98K into this declaration would have understated the control by a
factor of two, and every reduction ratio computed against it would have been
wrong in the direction that flatters the incumbent. Nothing about reading the
old constant would have revealed that. It is the reason no parameter total in
this phase is a literal.

### 2.2 What S4 is, and what it is not

S4 is labelled `CURRENT_DIMENSION_PROJECTION_CONTROL`, **not**
`HISTORICAL_EXACT`. The formulas are v1's: the same two bias-free projections,
the same GELU, the same post-GELU normalization, the same four blocks and two
scalars as `operator_models.ExplicitFeatureMLP.forward_explicit`. Three things
are not v1's:

1. the input width is the current payload's 1536, not 768;
2. what consumes the columns is M2's frozen scorer, not v1's
   parameter-matched head;
3. the structural columns beside it are M2's.

It is a control for *does a projection-style semantic branch buy anything over
two learned vectors* -- which is M2B's question. It is not a reconstruction of
QLS-v1, which would need v1's scorer too and would answer a different one.

S4 also does not live in `qls_v2_semantic.RUNG_FEATURES`. Those rungs form a
strictly nested chain that `tests/test_qls_v2_semantic.py` asserts, and a
projection block shares no column with S2 or S3; adding it there would break
that invariant and assert a monotone relationship that does not exist. It has
its own module, `src/mp_retrieval/m2b_semantic_control.py`.

S0 and S1 are excluded on purpose: no semantic signal at all, and a single
cosine. The question is minimality among usable representations, not the floor
of the ladder.

### 2.3 How the branch is swapped without touching anything else

`M1AScorer` takes an optional `semantic_head`. Passing `None` -- what every
M1A, M1B and M2 caller does -- builds `SemanticHead(rung, dim)` exactly as
before, so no completed fit's construction changes and the full suite is
unchanged at 6447 passed / 108 skipped. M2B passes its own head. The
constructor raises if an injected head's `rung` disagrees with the rung name
the model is being recorded under: a fit filed under the wrong rung is
unrecoverable after the fact.

## 3. The evaluation matrix

M2's 14 cells, read from `m2_qls_v2_freeze.yaml#m2_selection_matrix.cells`
rather than re-derived:

```
squad_clean      R1
musique_clean    R1
2wiki_clean      R1  R2  R3
hotpotqa_clean   R1  R2  R3
metaqa           R1  R2  R3
webqsp           R1  R2  R3
```

14 cells x 3 rungs = **42 logical fits**. The S3 row is already run: all 14
M2 QLS-UNIVERSAL fits are seed-0 S3 fits in exactly these cells under exactly
this schema.

| | count |
|---|---:|
| logical matrix | 42 |
| reused (S3) | 14 |
| new (S2) | 14 |
| new (S4) | 14 |
| **new total** | **28** |

`scripts/m2b_reuse_and_compute.py` derives this by matching each declared cell
against the fits M2's headline artifacts actually contain, and reports any
cell it cannot match rather than assuming coverage. Zero were unmatched.

One thing is *not* reused: **S3's timings**. The systems tie-break orders on
uncached inference p95, and an S3 latency measured on another day in another
container measures the container. S3 is re-benchmarked for inference alongside
the new rungs; only its fitted weights and effectiveness metrics carry over.

## 4. Feature-store reuse, and the one thing that blocks it

**M2's feature stores are immutable inputs to M2B. Graph and structural
features are not rebuilt per semantic rung.**

This holds by construction, and the proof is mechanical rather than a reading
of the code. A persisted cell's identity is `cell_build_key`, whose eight
fields are dataset, data fingerprint, regime, panel size, per-seed cap,
neighbour scan cap, A64 mainline family, and the declaration hash. The
semantic rung is not among them. `m2b_reuse_and_compute.py` calls that real
function with the rung set to S2, S3 and S4 on an otherwise identical
namespace and shows the key is byte-identical in every declared regime.

The store holds 12 structural columns -- `BASE 4, GEOMETRY 3, SUPPORT 1,
PATH 3, NODE_ROLE 1` -- of which QLS-UNIVERSAL reads 9; the surplus is the
excluded GEOMETRY family. **No column is semantic.**
`M1AScorer.forward_explicit` computes the semantic block live, per query, from
raw node and query embeddings, so there is nothing rung-specific in the store
to go stale.

That gives the equality the declaration requires across S2/S3/S4 in a cell --
query IDs, scored candidates, context, structural feature tensor, retrieval
features, seed, `NODE_ROLE`, `SUPPORT`, `PATH`, normalization -- **by identity
rather than by comparison.** All ten come from the one persisted store. A
comparison would be the weaker guarantee: it could pass on two objects that
merely happened to agree.

### 4.1 The obstruction is the paperwork, not the features

`cell_build_key`'s eighth field is `config_sha256`, a hash of the whole M2
declaration file. That file has since gained amendment 7 (M2's verdict), the
`result` block, amendment 8 and the `selected` block:

```
build time  d9518792075054612f70cd77d710c4af71f70407c287e8b883b87e68b84747a1
now         9c2790dddf6f2f8506ad891101906ba672530c56969ce0d6109c41248640b723
```

`load_cell_for_fit` recomputes the key and raises on any differing field, so
**all 14 stores would be refused** -- on a field describing the paperwork
rather than the features. The obvious workaround, a rebuild, would spend the
single largest line in the budget to reproduce byte-identical arrays.

**Resolution: `PIN_TO_RECORDED_BUILD_TIME_HASH`.** M2B passes the build-time
hash as the expected value, recovered from each M2 fit's own
`provenance.config_sha256` (identical across all six datasets). The check is
not relaxed and the field is not dropped: the other seven fields are still
compared exactly, so a store from the wrong dataset, regime, panel or A64
budget is still refused. The hash reverts to what it was always for -- a
record of *which declaration built this store*.

This licenses exactly one situation: the declaration file gained prose after
the stores were built. If any of the seven cell-identity fields had moved, the
store would be refused, and should be.

## 5. The selection rule, frozen before any M2B number exists

Primary metric **recall@5**. Anchoring is **symmetric and best-anchored, not
incumbent-anchored**:

```
cell_best(c)       = max recall@5 among S2/S3/S4 in cell c
dataset_score(r,d) = mean recall@5 of rung r over that dataset's declared regimes
dataset_best(d)    = max dataset_score among S2/S3/S4
macro_score(r)     = equal-weight mean of the six dataset_scores
macro_best         = max macro_score among S2/S3/S4
```

A rung is effectiveness-admissible **iff all three hold**:

```
macro_score(r)      >= macro_best     - 0.25pp
dataset_score(r,d)  >= dataset_best(d) - 0.50pp   for every dataset d
recall@5(r,c)       >= cell_best(c)    - 0.50pp   for every cell c
```

Why all three. The per-dataset clause stops a rung buying one giant gain by
sacrificing another dataset, which a macro-only rule would admit. The
equal-weight dataset macro stops the three-regime datasets carrying triple the
weight of `squad_clean` and `musique_clean`, which have one cell each. The
per-cell clause stops a rung failing badly in one regime and being covered by
its siblings inside the same dataset.

Best-anchoring is the point. S3 gets no privilege for being the incumbent and
S2 gets none for being smaller. Under an incumbent-anchored rule S3 would be
admissible by definition and the phase could only ever reject its challengers.

**If none survives: `SEMANTIC_PARETO_CONFLICT`, and stop for review.** Not a
widened tolerance, not a dropped clause, not a reweighted macro. A threshold
moved after seeing the numbers it failed is not a threshold.

**If several survive**, lexicographic systems Pareto ordering:

```
1. lower uncached inference p95
2. fewer total trainable parameters
3. lower peak memory
4. deterministic filed tie rule
```

The filed tie rule: smaller semantic output width, then the earlier of
S2/S3/S4 in that order. Filed now, before any number exists, so a tie is
resolved by a rule rather than by whichever rung is being argued for at the
time.

Parameter count is second on purpose. It is the headline this track cares
about, which is exactly why ordering on it first would let M2B select for its
own publication story. Latency is the property a reader of the systems table
can check on their own hardware.

Secondary diagnostics -- recall@1, recall@20, MRR, FullCov@20 -- are reported
for every cell and rung and read only *after* the recall@5 outcome is
computed. **They cannot replace recall@5 post-outcome.** They may explain a
result or raise a caution for a later phase; they may not overturn a verdict.

## 6. Seeds

Seed 0 only, one seed. **Five-seed confirmation remains prohibited anywhere in
development.** If selection is still decision-critical within the gray bands
after the one-seed result, the *smallest sufficient* 3-seed resolution may be
**proposed**, naming the specific cells it would resolve. It is not launched
with the one-seed run and is not launched automatically. Five seeds are not
the fallback.

## 7. Instrumentation

Every new fit records, at the time it runs: checkpoint; per-query rows;
aggregate metrics reconstructed from those rows; source commit; config
fingerprint; dataset and candidate fingerprints; feature-store fingerprint;
**semantic-rung fingerprint**; semantic, scorer and total parameter counts;
train time; uncached inference p50/p95/p99; peak VRAM and RSS.

**No recovery reruns later.** A field missing from a completed fit is not
recoverable, because the rerun would be a different fit on different hardware.
A run that cannot record these is not started.

The semantic-rung fingerprint is new in M2B and is what makes the phase
auditable: a hash over the rung name, implementing module and class, ordered
feature names, embedding width, projection width where one exists, and
semantic parameter count. Two fits differing only in semantic representation
are otherwise nearly indistinguishable in their provenance -- precisely the
confusion this phase could produce.

## 8. Compute

Rates are `docs/COMPUTE_LEDGER.md` L546-547, reused verbatim: $2.241/h GPU,
$0.634/h CPU.

| line | floor | conservative |
|------|------:|-------------:|
| feature-store build (0 new) | $0.00 | $0.00 |
| S2 fits (14) | $0.6922 | $0.6922 |
| S4 fits (14) | $0.6922 | $1.7306 |
| inference benchmarking (42 arms) | $0.7843 | $0.7843 |
| container overhead (28, upper bound) | $1.3272 | $1.3272 |
| **total** | **$3.4959** | **$4.5343** |

Proposed ceiling **$6.00** -- the conservative bound plus ~32%, headroom for
the S4 multiplier being wrong in the expensive direction. Below M2's $9.00
because M2B buys no feature builds, and M2 measured $4.3914 *with* them.

The feature-build line is zero **because of a proof, not an assumption**. M2's
own estimate put feature build at 5,431-5,782 s and called it 83% of that
phase; at GPU rates that is $3.60 avoided.

Fit costs are each cell's own measured M2 S3 `training_seconds`, scaled. S2 is
floored at parity: it runs the same per-query loop over the same candidates
and does strictly less arithmetic inside it, so it cannot be slower for a
compute reason. S4 is scaled by 2.5 for the conservative bound. It replaces
three reductions with two 1536x64 projections -- about 64x the arithmetic per
candidate -- and widens the scorer's first layer from 14 to 267 columns.
Neither is likely to appear at 64x in wall clock, because at this scale the
fits are dominated by per-query kernel-launch overhead in `M1AScorer`'s Python
loop rather than by the size of each kernel. **That is an argument, not a
measurement**, which is exactly why the smoke measures it before any fan-out.
2.5 is a loose bound to launch under, not a prediction.

Container overhead is priced at one container per fit, which is an upper
bound; M2's actual orchestration was one container per dataset, and the
equivalent here is 12 rather than 28. The larger number is kept because the
estimate should not depend on an orchestration choice nobody has made yet.

Storage: checkpoints total 11.7 MB across all cells and rungs. S4's is 57x the
incumbent's per cell -- under a megabyte each and not worth pricing, but the
ratio is the point of the phase and is not hidden inside a rounding.

## 9. Smoke before fan-out

**Exactly one cell, and only S2 and S4 in it.** S3 is already proven there by
a completed M2 fit; re-running it would spend money to re-answer a question
with a filed answer.

**Cell: `2wiki_clean / R3`.** It exercises a real R3 `NODE_ROLE` column --
non-constant structural admission -- so the smoke tests the branch under the
schema's most demanding regime rather than under R1, where `NODE_ROLE` is
identically zero and a broken structural path could pass unnoticed. It is also
the cell M2 smoked, so its numbers are comparable, and one of the cheapest in
the matrix at ~19 s of training.

The smoke must establish: both heads instantiate at the frozen width and
produce the declared column counts; the persisted M2 store loads under the
pinned build-time hash; every instrumentation field is populated; and the
measured S4 fit time, which replaces the 2.5 multiplier with a number.

**The smoke is a fit and is not authorised by this document.**

## 10. What this phase can and cannot do

If M2B returns `SEMANTIC_PARETO_CONFLICT`, or a reviewer rejects the candidate
set, the frozen object remains M2's: QLS-UNIVERSAL at S3, 3,585 parameters,
exactly as `m2_qls_v2_freeze.yaml#selected` records it. **M2B can only narrow
the semantic representation or fail to.** It cannot invalidate M2, because it
changes nothing M2 measured.

Standing prohibitions, unchanged in force: no five-seed confirmation anywhere
in development; no Package F; no E2 resume; no A64 changes; no structural
schema changes in this phase; no GNN training; no M3; no canonical CRAG read.

## 11. Stop

Declaration, live implementation audit, parameter accounting, exact workload,
reuse plan, compute estimate -- all six are filed. **`STOP_FOR_REVIEW`.**

Nothing happens until a further dated amendment authorises execution, and that
authorisation covers the smoke cell first, with the 28-fit fan-out only after
the smoke's numbers are reviewed.
