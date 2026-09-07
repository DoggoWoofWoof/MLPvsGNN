# M2C — S4 structural conditioning

Filed 2026-09-07, before any M2C arm is fit. The machine-readable declaration is
[`configs/m2c_s4_structural_conditioning.yaml`](../configs/m2c_s4_structural_conditioning.yaml);
this document is the argument behind it. Status:
`M2C_STAGE0_COMPLETE_STOP_STRUCTURAL_ADMISSION_CLOSED` — Stage 0 ran on
2026-09-08 and returned `STOP_STRUCTURAL_M2C` and `ADMISSION_CLOSED`; the
results are in [`M2C_STAGE0_REPORT.md`](M2C_STAGE0_REPORT.md). Everything below
was written before those results and is left as it was.

**This is a post-hoc development branch**, opened because of an observed S4
result rather than from a preregistered hypothesis, and labelled as such in
every artifact it produces. It does not reinterpret M2 or M2B. Those records are
frozen and are read here only as a starting point:

| record | state, unchanged by M2C |
|---|---|
| M2 | QLS-UNIVERSAL accepted |
| M2B | preregistered verdict `SELECTED_S3`, 3,585 parameters |
| S4 | a **challenger**: 205,217 parameters, best in 12/14 cells, fastest rung, blocked at `squad_clean/R1` and `musique_clean/R1` |

M2C touches no M2 or M2B file, no Package F, no canonical CRAG, no E2, and no
part of M3. **No GNN is used anywhere in M2C.**

## 1. The question

> Can deterministic graph-derived displacement geometry, and/or a small
> structure-conditioned residual transform, repair S4's localized generalization
> failures while retaining its effectiveness and systems advantages, without
> learned message passing?

The question is *not* "can we make S4 score higher". Four outcomes have to stay
distinguishable, and the arm ladder in §6 exists to keep them apart:

| | claim |
|---|---|
| **A** | useful graph information already exists as deterministic geometry |
| **B** | that geometry only needs a small learned interaction |
| **C** | a generic extra adapter helps regardless of graph structure |
| **D** | the apparent signal is embedding-kNN, not native topology |

C is why the matched-capacity control in §6 is mandatory rather than optional. D
is why every result is measured under three graph provenance views.

## 2. What the archaeology found, and why it reorders the phase

The mechanism this phase's specification describes — a query residual, real edge
displacements `normalize(x_v − x_u)`, a cosine compatibility against the
residual, per-seed top-k, dedup, a global cap, a matched-budget rule — **already
exists in this repository**, in
[`src/mp_retrieval/candidate_expansion_v2.py`](../src/mp_retrieval/candidate_expansion_v2.py),
as `L1_DIRECTIONAL`. It has leakage tests that permute gold labels and require
byte-identical output.

It has also already been measured, and the result was a null.

**The incumbent is the control.**
[`configs/m0c_bounded_r3.yaml`](../configs/m0c_bounded_r3.yaml) defines A64 as
the M0B `STRUCTURAL_NEIGHBOUR` admission at `graph_expansion_cap=64`,
re-derived through `candidate_expansion_v2.expand`. So the R3 candidate set that
every M2B R3 cell already uses **is** the blind-admission arm, and the proposed
budget-matched comparison "R3-A64 versus R3-PF64" is `STRUCTURAL_NEIGHBOUR`
versus `L1_DIRECTIONAL` under a different name.

**M0A ran that comparison.** From
[`docs/M0A_PROBE_RESULTS.md`](M0A_PROBE_RESULTS.md), 100 validation queries ×
3 datasets × 3 edge families, at matched, additive and unconstrained budgets:

1. `L1_DIRECTIONAL` and `STRUCTURAL_NEIGHBOUR` produced **identical headroom in
   every one of the nine family cells, to every recorded digit** — including
   cells where the two arms admitted almost disjoint sets
   (`squad_clean`/structural: mean Jaccard 0.292, minimum 0.013). The "the caps
   never bound, so no choice was forced" explanation was checked and rejected
   for seven of the nine cells.
2. `knn_only` recovered **exactly zero** golds on every dataset, under every
   method, at every budget including unconstrained. `structural_only` and
   `baseline_a_simple` recovered identically: `2wiki_clean` +15.25pp ceiling@5,
   `metaqa` +34.83pp, `squad_clean` +0.00pp.
3. The gain saturates. On `metaqa`, admitting seven times as many nodes per
   query (167 against 23) recovered zero additional gold.

Two consequences follow, and both are filed at the top of the declaration
because discovering them after spending compute would be the expensive way to
learn them.

**The STRUCT/KNN/FULL ceiling table is already filled in.** STRUCT == FULL,
KNN == 0. It is reported from M0A, not re-derived. A Track B result showing kNN
contributing to the ceiling is a bug hypothesis before it is a discovery.

**Exactly one variable is genuinely new.** M0A's residual is
`normalize(e_q − x_anchor)` for a **single** anchor at dense rank 1 — it discards
nine of the ten retrieval seeds and makes the direction a function of whichever
document dense retrieval happened to put first. This phase specifies

```
r_q = q − P_span(E_q) q
```

the component of the query orthogonal to the span of **all** the inference-safe
seeds: what no seed already explains. That is a different and better-motivated
residual, and it is the only thing that differs. So the admission track, if it
runs, is a **one-variable re-test of a measured null**, not a fresh exploration,
and is scoped that way.

Reporting a Track B win without saying that the same comparison under a
single-anchor residual returned a null would be wrong. If the seed-subspace
residual breaks the tie, the finding is *"the residual definition was what
mattered"* — a sharper claim than "directions help", and one that only exists
because the null is on the record.

### What is genuinely unmeasured

`query_residual` appears in exactly one module and one test in this repository,
**both about admission**. Directional compatibility has never been used to
*order* candidates. `dir_max`, `dir_mean`, the projected offset and the
structure-conditioned transform are all unmeasured, and M0A measured oracle
ceilings without ever measuring actual/ceiling.

**Ranking is the open question; admission is the narrow re-test.** The phase is
ordered accordingly, which is the reverse of the order in which the two ideas
arrived.

## 3. Ceiling and attainment

The decomposition that separates the two tracks:

```
recall@5  =  attainment@5  ×  recall_ceiling@5
```

A reranking intervention moves only the first factor. The second is fixed by
which candidates are **scored**.

### Which ceiling

M2B records **two** ceiling quantities, and only one of them bounds `recall@5`.
An earlier draft of this declaration divided by the wrong one; the definitions
are pinned here so it cannot recur.

| quantity | source | K-aware? |
|---|---|---|
| `recall_ceiling@5` | `regime_headroom["recall_ceiling@5"]` | **yes** — this is the bound on recall@5 |
| `candidate_pool_ceiling` | `metrics["candidate_ceiling"]` | no — macro fraction of a query's golds present in the scored pool |

They diverge wherever a query has more than five golds. `metaqa` reaches 246
golds per query (mean 7.69, p95 36) and `webqsp` 222 (mean 5.93, p95 27), so on
those two datasets K binds long before pool coverage does — `metaqa`/R1 has
`recall_ceiling@5` 0.325429 against a split-level pool coverage of 0.333843, and
`webqsp`/R1 0.460434 against 0.493853 — and dividing by pool coverage would
invent ranking headroom no ranker could reach. On the other four datasets no
query has more than four golds, K never binds, and the two agree up to the
held-out/split panel difference. **Both blocker cells are in that second
group** — `squad_clean` has exactly one gold per query, `musique_clean` at most
four — so their headroom numbers are unaffected by the distinction.

M2B also already computed attainment, as `ceiling_attainment_at_5`, against
`recall_ceiling@5`; verified to agree in all 50 rows. The baseline table copies
that field rather than deriving a second attainment that would quietly disagree
with the frozen record.

### What was measured

**R1 and R2 share a ceiling exactly.** `recall_ceiling@5` is bit-identical
between R1 and R2 on `2wiki_clean` (0.793333), `webqsp` (0.460434), `metaqa`
(0.325429) and `hotpotqa_clean` (0.934006). R2 widens *context* while still
scoring `Cq`, so it cannot change what R1 could have retrieved. R3 is the only
regime that moves the ceiling, because `C3 = Cq ∪ A64` changes the scored set.
R2 exposed some answers structurally; R3's distinctive contribution was making
them scoreable.

**Ranking headroom exists in every cell.** All 28 seed-0 S3/S4 rows have a
positive ceiling-minus-recall gap; the smallest is 7.77pp (`squad_clean`/R1,
S3).

**Exposure is bounded on the blockers.**
`recall_ceiling_perfect_retrieval@5 − recall_ceiling@5` is the part of the
ceiling candidate generation threw away — the only quantity an admission-side
intervention can move.

| cell (S4) | ranking headroom | exposure lost to generation |
|---|---:|---:|
| `squad_clean`/R1 | 8.52pp | **0.46pp** |
| `musique_clean`/R1 | 18.52pp | 6.39pp |
| `webqsp`/R3 | 36.68pp | 17.31pp |

### How much actually has to be recovered

An earlier draft of this section compared the 0.46pp of squad exposure against
the 0.50pp per-cell tolerance and concluded that admission could not repair that
blocker even in principle. **That comparison was wrong, and the conclusion drawn
from it is withdrawn.** The tolerance is not the amount of work.
`SYMMETRIC_BEST_ANCHORED` admits a rung that is no worse than 0.50pp below the
cell's best, so a challenger sitting at a deficit `d` has to recover
`d − 0.50pp`, not `0.50pp` and not the whole of `d`:

```
required_repair_to_guard = max(0, |robust deficit| − 0.50pp)
```

Recomputed from the frozen multi-seed margins by
`scripts/m2c_baseline_table.py`, never typed:

| cell (S4) | robust deficit | must recover | exposure available | slack | share of exposure that must convert |
|---|---:|---:|---:|---:|---:|
| `squad_clean`/R1 | −0.8824pp | **0.3824pp** | 0.4604pp | 0.0780pp | **83.1%** |
| `musique_clean`/R1 | −2.5721pp | **2.0721pp** | 6.3853pp | 4.3132pp | 32.5% |

Exposure exceeds the required repair on **both** blockers, so admission is not
ruled out arithmetically on either. The corrected reading:

> Squad's blocker is primarily an ordering defect. Candidate admission has only
> a narrow theoretical path to clear the M2B guard and cannot be assumed
> sufficient.

Note that the ordering of plausibility reverses under the correct arithmetic.
Squad, the cell the earlier draft called impossible, needs 83.1% of *all*
available exposure to convert into realised recall@5; musique needs 32.5%. On
this test musique is the *more* plausible admission target of the two, not the
less.

**An oracle bound is not an achieved metric.** Exposure is what a *perfect*
admission mechanism could win. Comparing it against `required_repair_to_guard`
can rule a path out; it can never rule one in. A real mechanism converting 83%
of an oracle bound is not something any evidence in this repository predicts.

So the repair target sits far inside ranking headroom that already exists, and
well inside exposure on musique. **Track A remains the leading track** — on
sufficiency of headroom, on M0A's null, and on the failure shape in §4 — but it
leads on weight of evidence, not because Track B has been eliminated. Keeping
the two separate stops a ceiling result being reported as though it had repaired
the blockers, or the reverse.

## 4. The shape of the failure

Derived from the frozen baseline rows, and it constrains what a repair can look
like.

**The damage is concentrated at the head of the ranking.** Over seeds 0/1/2,
monotone in the cutoff, consistent in every individual seed:

| cell | ΔR@1 | ΔR@5 | ΔR@20 | ΔMRR |
|---|---:|---:|---:|---:|
| `squad_clean`/R1 | −4.303pp | −0.882pp | −0.262pp | −2.615pp |
| `musique_clean`/R1 | −12.505pp | −2.572pp | −1.108pp | −14.922pp |

The gold is still inside S4's top 20 almost exactly as often as inside S3's.
**S4 is not a worse retriever here; it is a worse orderer.** Across all 14 cells
at seed 0, S4 is best at recall@5 in 12 but worse at recall@1 in 8 — and worse
at MRR in the same 8.

**It is a passage-graph phenomenon.** S4 loses recall@1 in exactly the eight
passage-graph cells (`2wiki_clean`, `hotpotqa_clean`, `musique_clean`,
`squad_clean`) and wins recall@1 in all six KB cells (`metaqa`, `webqsp`). The
split is clean, with no exceptions.

Two things follow.

**A second, independent reason Track A leads.** A candidate-admission
intervention cannot repair a rank-1 ordering error among candidates that were
*already scored*, and §4 shows the damage is concentrated exactly there. So:

> S4's blocker is dominated by top-of-ranking error; candidate exposure is a
> secondary possible contributor.

This is evidence about where the damage sits, and it is independent of M0A's
null. It is **not** proof that admission is irrelevant — the arithmetic in §3
leaves both blockers open to an admission repair in principle, and the two
statements have to be allowed to stand together.

**A constraint on what counts as a repair.** Because the damage decays
monotonically with K, an arm that closes recall@5 while leaving recall@1 and MRR
far below S3 has shuffled the top 5 rather than repaired the ordering. recall@1
and MRR are reported next to recall@5 for every arm on both blocker cells, and
that outcome is reported as what it is.

## 5. Inputs, and what is frozen

The QLS-universal block (`dense_rr`, `splade_rr`, `retriever_agreement`,
`is_seed`, `SUPPORT`, `PATH_1..3`, `NODE_ROLE`) plus S4's existing projected
semantic representation. Nothing is removed or silently redefined.

S4 as recovered from
[`src/mp_retrieval/m2b_semantic_control.py`](../src/mp_retrieval/m2b_semantic_control.py):

| item | value |
|---|---|
| class | `ProjectionSemanticHead` |
| `query_projection`, `node_projection` | `Linear(1536, 64, bias=False)` |
| `node_state` | `normalize(gelu(node_projection(e_v)))` |
| emitted columns | 4 × 64 + 2 = 258 |
| blocks | `query_state`, `node_state`, `state_product`, `state_absolute_difference` |
| scalars | `normalized_state_dot`, `raw_projection_dot_scaled` |
| parameters | 196,608 semantic + 8,609 scorer = 205,217 |

`node_projection` is the `W_d` the zero-new-parameter offset reuses.

### Graph provenance views

Mapped onto the families
[`src/mp_retrieval/edge_provenance.py`](../src/mp_retrieval/edge_provenance.py)
already reconstructs — in the frozen node-row coordinate system, validated
against the master documents, fingerprinted with a serialization-independent
sha256. M2C **selects among existing families and reconstructs nothing**.

| view | family |
|---|---|
| `G_STRUCT` | `structural_only` |
| `G_KNN` | `knn_only` |
| `G_FULL` | `baseline_a_simple` |

`G_FULL` is `baseline_a_simple`, not `full_union_c`. The module's own identity
line is `knn_only = baseline_a_simple MINUS structural_only`, so
`baseline_a_simple` is exactly STRUCT + KNN — which is what "native structural +
frozen kNN" means. `full_union_c` adds a third provenance (NER) that this phase
did not ask about and that is out of scope.

For the first provenance experiment the frozen S4/QLS feature store does not
change: only the new displacement branch varies between the three views, so a
difference between them is the provenance of the *new* information rather than a
simultaneous change to SUPPORT/PATH.

### The residual

Implemented in
[`src/mp_retrieval/m2c_structural_offset.py`](../src/mp_retrieval/m2c_structural_offset.py)
by rank-revealing SVD of the seed matrix with a relative singular-value
tolerance, then projection onto the retained basis — **never a normal-equations
inverse**, which is exactly the formulation that fails when two seeds are
near-collinear. Seeds are retrieved by similarity to one query, so they routinely
are.

Zero seeds, one seed, collinear seeds, rank-deficient `E` and a vanishing
residual are each handled and each **recorded**. When the query lies inside the
span of its own seeds there is no "what is missing" direction, so the fallback is
the normalized query with `degenerate_residual = true` on the row. A fallback
that is not recorded is a silent second method.

The two residuals are **not nested**: `q − x_a` subtracts the anchor *vector*,
while `q − P_span{x_a} q` subtracts only the component of `q` along it.
`single_anchor_residual` is provided so a probe can *measure* how far apart they
point instead of assuming the change matters.

## 6. The ladder

| arm | what | new trainable params |
|---|---|---|
| **A0** `S4` | the frozen current challenger | 0 |
| **A1** `S4-DIR-SCALAR` | S4 unchanged plus `dir_max`, `dir_mean` | 2 scorer weights per scalar admitted |
| **A2** `S4-PF-OFFSET` | deterministic displacement in S4's own 64-D state space | **0** |
| **A3** `S4-SEMANTIC-TRANSFORM` | matched-capacity control on `[q64, d64]` only | = A4 |
| **A4** `S4-STRUCT-TRANSFORM` | small residual transform conditioned on structural geometry | small, reported live |

A1 answers whether the useful information is a scalar rather than a vector
shift — the cheapest possible positive result, and the one that would make
everything above it unnecessary.

**A2 is a ZERO-NEW-PARAMETER STRUCTURAL OFFSET, never a "parameter-free
model."** S4 itself is learned. The offset introduces no new learned parameter,
and that is the precise claim; the other phrasing would assert something untrue
about the object it is bolted onto. Construction: `delta64 = W_d @ delta_star`
reusing S4's existing frozen `node_projection`, a bounded deterministic `alpha`
from the same directional geometry, output `normalize(d64 + alpha * delta64)`.

*The GELU wrinkle.* S4's `node_state` is `normalize(gelu(W_d e_v))`, so `d64` has
passed through a GELU and `W_d delta_star` has not; the sum mixes a
post-activation state with a pre-activation displacement. It is implemented as
specified and declared for what it is — the Jacobian-free first-order image of
the displacement, ignoring GELU's local slope. The pre-activation variant
`normalize(gelu(W_d(e_v + alpha·delta_star)))` is a legitimate alternative and is
**not** run as a second arm in the pilot; Stage 0 can compare the two
geometrically for free, since neither needs training. Silently substituting the
tidier form would be wrong.

**A3 is mandatory, not optional.** Without it, "A4 improves" is
indistinguishable from "we added another MLP". Same architecture, hidden width,
parameter budget, initialization and training as A4; inputs restricted to
`[q64, d64]`, with `delta64`, the directional scalars and any new
SUPPORT/PATH-derived input forbidden.

**A4 is FiLM, identity-initialized** (γ = 1, β = 0 at step 0). Both candidate
formulations are identity-preserving at initialization, which is the property
that matters; FiLM's γ additionally lets the transform *attenuate* the semantic
state where structure contradicts it, and §4 shows the failure cells are cases
where S4 is confidently wrong at the head of the list. A pure additive residual
can only add. **One formulation is implemented, not both.**

The cross-product — 5 arms × 3 graph views × 14 cells × multiple seeds — is not
run.

## 7. Tracks

**Track A, ranking offset — the primary track.** Candidate universe frozen (`Cq`
under R1/R2, `C3` under R3). Does structural displacement help S4 *order* what it
already has? Moves attainment only; the ceiling is reported unchanged as proof
that it did not move.

**Track B, offset admission — conditional.** `C4 = C3 ∪ A_offset`, reporting
Δceiling, Δactual recall@5 and attainment *separately*. Budget **64 only**: A64
is the incumbent admission at exactly that budget, so 64 is the only value at
which the comparison is budget-matched rather than budget-buying. 16 and 32
would answer "does a smaller budget do worse", which M0A already answered by
showing the gain saturates at a per-seed cap of 16. The frontier construction,
per-seed cap, scan cap, dedup rule, global cap and matched-budget rule are the
existing `candidate_expansion_v2` code paths, byte-identical between arms —
**only the residual definition changes**, or the result is uninterpretable
against M0A's null. Runs only if Stage 0 shows the new residual actually changes
which directions are chosen.

**Track C, transform — conditional** on A or B producing something to exploit.

## 8. Stage 0 — zero training

Trains nothing, reads no test split, runs on the validation/development split
only.

**Primary question.** Does graph-derived directional compatibility contain
useful *ranking* signal for S4's mistakes?

**Secondary question.** Does replacing the legacy single-anchor residual with
the seed-subspace residual change *admission* enough to reopen M0A's headroom
null?

The ordering is deliberate and it is the amendment's main structural change.
§3's arithmetic leaves admission open on both blockers, so the secondary
question is live — but §4 puts the damage at the head of the ranking, and M0A
already nulled the admission mechanism once. Ranking is asked first because it
is where the evidence points, not because admission has been eliminated.

### Cells

Four. Declared before any result, derived from the frozen graph audit and M0A,
both of which predate every M2C number.

| role | cell | why |
|---|---|---|
| failure | `squad_clean`/R1 | blocker, −0.882pp robust |
| failure | `musique_clean`/R1 | blocker, −2.572pp robust |
| passage R3 control | `2wiki_clean`/R3 | highest isolated-gold prevalence of the passage sets (20.0% of queries), largest measured one-hop ceiling recovery among them (+15.25pp) |
| KB R3 control | `metaqa`/R3 | largest recovery M0A measured anywhere (+34.83pp ceiling@5), most graph-damaged of the six (isolation 0.412, boundary cut 0.904) |

Both controls are R3 because R3 is the only regime whose scored set an offset
could ever change. **All 14 cells are not run.**

### The three residual controls

| id | form | role |
|---|---|---|
| **R0** `RAW_QUERY_CONTROL` | `r = normalize(q)` | subtracts nothing |
| **R1** `LEGACY_DIRECTIONAL` | `normalize(e_q − x_anchor)`, anchor = dense rank 1 | what M0A actually ran |
| **R2** `SEED_SUBSPACE` | `r = q − P_E q` over inference-safe seeds | the new variable |

**The central comparison is R1 against R2.** That is the one genuinely new
independent variable in this phase; everything else is a control on it.

**R1 is called, not reimplemented.** The arm invokes
`candidate_expansion_v2.query_residual` directly. An arm reconstructed from
prose would not be the function M0A ran, and the comparison against M0A's null
would not be a comparison.

**R0 exists so a positive result cannot be misattributed.** If a bare query
direction ranks as well as either residual, then "what the seeds do not explain"
is not the operative construct and the residual is decoration. Note that a
degenerate R2 falls back to `normalize(q)` and so *coincides* with R0 as a
vector — the `fell_back_to_query` flag keeps the two distinguishable in the
record, because a fallback row and a control row mean different things.

### The directional score

`delta(s, v) = normalize(e_v − e_s)`, compatibility `cosine(r_q, delta(s, v))`,
aggregated to a minimum of **`dir_max` and `dir_mean`**. Optionally `delta_star`
from a fixed aggregation rule with `tau` declared before results. **No large
feature catalogue is built** — a wide catalogue would turn a null into a search.

**Coverage is measured explicitly** and reported beside every margin: the
fraction of scored candidates with at least one eligible seed-edge displacement.
A signal that exists for a small minority of candidates cannot repair a ranking
even if it is perfectly informative where it exists, and a margin without its
coverage hides exactly that.

### The error-conditioned diagnostic

**This is the load-bearing measurement, and it matters more than any global
correlation.** A positive global correlation with relevance is compatible with
*zero* repair capacity, if the signal agrees with S4 wherever S4 is already
right.

Population: every query where S4's top-1 is wrong **and** a relevant candidate
exists in the scored universe. Queries with no relevant candidate scored are
excluded and counted separately — no reranker could win them, and leaving them
in would dilute the margin with unwinnable cases.

```
directional_margin = dir(best relevant) − dir(S4's top-ranked wrong candidate)
```

Reported: the margin, the fraction positive, mean and median. Stratified by
where the first relevant item currently sits — **rank 2–5**, **rank 6–20**,
**beyond 20 or absent** — because a signal that only helps when the answer is
already at rank 2 buys much less than one that reaches into the tail.

Implemented as `m2c_structural_offset.error_conditioned_margin`. It is the only
function in that module permitted to see a label, it is named in an explicit
allowlist that tests enforce in both directions, and the labels select *which
candidates to compare* — the directional scores are handed in already built.

### The zero-training ranking test

**Arm A, `direction_only`.** Rank the existing scored universe by directional
compatibility alone; report recall@1/5/20 and MRR. This is a **diagnostic, not
a candidate model** — nobody is proposing to ship it.

**Arm B, `S4_plus_direction_rrf`.** Reciprocal rank fusion, **not a tuned score
weight**. The constant is **60**, `REUSED_NOT_NEW`, taken from
`configs/candidate_budget.yaml#candidate_construction.rrf_constant`, with the
same `ascending_global_node_id` tie-break. **No weight sweep and no constant
sweep.** A fitted weight would make this a one-parameter trained model needing
its own control; RRF at the project's existing constant introduces nothing new
and is falsifiable exactly as it stands.

The third arm is `S4` itself, unmodified, as the reference every delta is taken
against.

Matrix: **3 arms × 3 residuals × 3 provenances × 4 cells, zero training.**

### Provenance, for ranking

Measured under `G_STRUCT`, `G_KNN` and `G_FULL`. Held byte-identical between
arms: scored candidates, S4's existing structural features, the S4 checkpoint,
candidate normalization. Only the new directional calculation changes.

**M0A does not answer this.** Its STRUCT/KNN/FULL result is about the admission
*ceiling*. This is whether provenance carries *ranking* signal — a different
quantity over a different operation. That the admission table is settled is not
a reason to skip the ranking version.

### The advance gate

**Filed before any Modal result exists.** `stage_0_probe_run: false` in the
declaration's gate block is what makes that checkable rather than asserted.

Advance requires **all four** of:

1. **Effectiveness** — either (A) S4 + direction RRF improves recall@5 on
   **both** failure cells, by at least **+0.25pp** on at least one; **or** (B)
   recall@1 improves by at least **+2.0pp** on **both**, with recall@5 no worse
   than **−0.25pp** on either.
2. **Protection** — neither positive control regresses by more than **0.50pp**
   recall@5.
3. **Mechanism** — the seed-subspace residual is **not behaviourally identical**
   to the legacy one.
4. **Mechanism** — the relevant-versus-top-wrong margin is meaningfully positive
   on **at least one** failure cell.

These are not adjustable after results are seen. **If the gate fails: do NOT
train `S4-STRUCT-TRANSFORM`.** S3 remains the final development semantic choice
and S4 remains a challenger with a filed, unrepaired blocker. That is a
publishable outcome, not a failure to be worked around. Code that runs is not
evidence that a mechanism exists, and a transform will not be trained on noise.

### The minimal admission diagnostic

Budget **64 only** — A64 is the incumbent admission at exactly that budget, so
64 is the only value at which the comparison is budget-matched rather than
budget-buying. Three arms:

| arm | what |
|---|---|
| `A64` | the existing `STRUCTURAL_NEIGHBOUR` blind-admission control |
| `legacy-PF64` | R1's residual, parameter-free, at the same budget |
| `seed-subspace-PF64` | R2's residual, parameter-free, at the same budget |

Reported: pairwise **Jaccard** of admitted sets, **unique admitted nodes**,
**unique relevant admissions**, and the exact **`recall_ceiling@5`** for each
arm plus deltas.

**`candidate_ceiling` is never used as the recall@5 bound.** It is macro pool
coverage with no K; `recall_ceiling@5` is K-aware and is the denominator M2B
itself used. They agree on the four passage sets and diverge materially on
metaqa and webqsp, and substituting one for the other invents headroom. §3 keeps
the distinction; the diagnostic must too.

### Admission feasibility

Compare the oracle `Δrecall_ceiling@5` each arm achieves against
`required_repair_to_guard` from §3 — **0.3824pp** on `squad_clean`/R1,
**2.0721pp** on `musique_clean`/R1.

- Oracle gain **below** required → that arm **cannot** clear the M2B guard by
  admission, whatever its ranking does.
- Oracle gain **at or above** required → admission is not excluded. Not
  excluded is **not the same as capable**: an oracle bound is not an achieved
  metric, and squad would need 83.1% of it to convert.

### Passage versus KB

The split from §4 is quantified exactly, and it has **two live explanations**:

- **structural** — passage graphs carry displacement geometry S4 ignores;
- **semantic** — S4's projection miscalibrates passage embeddings, and the graph
  is incidental.

Stage 0 has to distinguish them. **If directional graph signal does not explain
the passage errors, a graph-conditioned repair is not forced merely because that
story is more attractive** — the honest reading would then be a semantic
calibration path, and the matched `A3_S4_SEMANTIC_TRANSFORM` control stays
mandatory the moment any learned-transform stage opens.

## 9. Stage 1 — proposed, not authorized

**Withdrawn from scope by the amendment.** The committed declaration allowed a
tiny Stage-1 pilot to follow automatically if the Stage-0 gate passed. It no
longer does: Stage 0 stops for review whether it passes or fails, and a Stage-1
matrix is *proposed with a cost* rather than run.

The shape below is retained as the proposal, not as an authorization.

Seed 0 only. S3 and S4 reused without refitting. Cells: the two blockers, plus a
**positive control declared before its results are seen** — `webqsp`/R3, where
S4 is already far ahead of S3 (+9.908pp) and the structural evidence is
nontrivial (R3 lifts webqsp's `recall_ceiling@5` from 0.4604 to 0.7290, the
largest ceiling movement of any cell, and it still loses 17.31pp to candidate
generation). It is the cleanest place to detect a challenger that repairs the
blockers by damaging what S4 was already good at.

Learned arms would not be trained under all three graph views; the Stage-0
provenance result picks one. The zero-new-parameter offset may compare
provenance sources cheaply, because it costs no training.

Targets: both blockers within **0.50pp** of the frozen S3 incumbent; positive
control no regression greater than 0.50pp from S4; recall@1 and MRR reported
beside recall@5 per §4. These are development gates and do not retroactively
change M2B.

Systems target: total uncached p95. A challenger that repairs effectiveness but
destroys S4's latency advantage does not automatically advance — that advantage
is measured, 1.068 ms p95 against S3's 1.922 ms.

## 10. Systems

The timed span runs from the retrieval the regime already requires, through
structural-edge lookup, residual construction, direction lookup and aggregation,
projection of the structural vector, the transform if any, to the scorer.
Components are reported separately: `structural_offset_build_ms`, `semantic_ms`,
`transform_ms`, `scorer_ms`, then total uncached p50/p95/p99, alongside peak
VRAM and RSS, feature-store bytes, and both parameter counts.

**Precomputed-only scorer timing is not uncached latency**, and offline build
cost is reported separately from online lookup cost.

The precomputation study estimates edge count, fp32 and fp16 storage, read
bandwidth and online compute saved *before* choosing among: P0 online, P1 full
1536-D delta for eligible edges, P2 a compressed deterministic projection, P3
`W_d @ delta(u,v)` for a frozen checkpoint. **P3 is checkpoint-specific** and
must be invalidated by the model-config hash, or it silently serves one
checkpoint's projections to another's scorer.

## 11. Provenance interpretation

| observation | reading |
|---|---|
| STRUCT > KNN | genuine topology contributes |
| KNN ≈ STRUCT | semantic geometry is doing much of the work |
| FULL > both | complementary provenance |
| FULL < STRUCT | kNN adds noise or semantic redundancy |

No structural-reasoning claim is made without this control. The prior for the
ceiling is already known — M0A measured KNN = 0 recovered golds everywhere.

## 12. Firewall

**Absolutely prohibited:** gold nodes as seeds, gold relation types,
supporting-fact labels at inference, dataset IDs, target-test tuning, Package F,
GNN outputs, GNN hidden states, GNN teacher or distillation, canonical CRAG test
inspection.

**Only inference-safe inputs:** retrieval seeds, frozen embeddings, frozen graph
topology, frozen edge provenance, the current regime's context, the current QLS
features.

Enforced by test, not by prose. The existing `candidate_expansion_v2` leakage
test permutes gold labels and requires byte-identical output;
[`tests/test_m2c_structural_offset.py`](../tests/test_m2c_structural_offset.py)
holds the new mechanism to the same standard, and additionally checks that no
function in the module accepts a label-shaped argument at all.

## 13. What happens after a success

No winner is declared immediately. Order: seeds 1 and 2 on decision-critical
pilot cells only; then, if still successful, the winning challenger across the
remaining M2B cells at seed 0; then, if still Pareto-admissible, targeted extra
seeds only where necessary. **No five-seed development sweep.**

Full-matrix advancement compares prospectively against both S3 and S4, and
replaces S3 only if the challenger repairs the blocker behaviour enough to
satisfy the universal per-cell and per-dataset protection, preserves S4's broad
advantages rather than overfitting SQuAD and MuSiQue, remains
systems-competitive, and has a defensible mechanism attribution from the offset,
the matched semantic transform and the provenance controls.

**The M2B rule is not edited after results.** A new M2C rule is filed
prospectively; M2B's rule and verdict stand as they are.

Generalization does not stop at target-refit success: `LODO-SQUAD` and
`LODO-MUSIQUE` first, training the allowed learned component without that
dataset and evaluating without refit. A2 has no new structural parameters, so it
is naturally dataset-transferable except for the existing S4 weights — that
asymmetry is stated, not used as a claim that A2 "generalizes" in the sense a
transform would. Learned transforms must distinguish architecture generalization
from parameter generalization. No dataset ID input, no target-specific logic, no
target-test tuning. Six-way LODO only if the two initial tests show it could
matter.

## 14. Authorization

**Amended 2026-09-07.** The turn that accepted `a71a83f` authorised Stage-0
execution and withdrew the Stage-1 pilot from scope until Stage 0 reports.

Authorized by this declaration and its amendment: the declaration and the
correction itself, the archaeology, the immutable baseline export, the
parameter-free residual and direction implementation, **the compute record**,
**real Stage-0 Modal execution**, **the zero-training ranking and provenance
probe**, **the minimal +64 admission diagnostic**, **result fetch and
verification**, tests, and **the Stage-0 report**.

Not authorized: **learned transform training**, **any Stage-1 fit**, the full
14-cell M2C evaluation, extra seeds, six-way LODO, M3 or any GNN, canonical
CRAG, Package F, E2.

| gate | state |
|---|---|
| `baseline_table_exported` | earned 2026-09-07 |
| `protocol_document_filed` | earned 2026-09-07 |
| `declaration_tested` | earned 2026-09-07 |
| `archaeology_recorded` | earned 2026-09-07 |
| `prior_evidence_audited` | earned 2026-09-07 |
| `residual_implemented_and_tested` | earned 2026-09-07 |
| `stage_0_compute_record_filed` | not earned |
| `stage_0_probe_run` | not earned |
| `stage_0_advance_gate_evaluated` | not earned |
| `stage_1_authorised` | **withdrawn from scope by this amendment** |

**Stage 0 cannot run on this machine.** Audited rather than assumed: the only
local data is `data/processed/*_l2_pilot.pt` for `2wiki_clean`, `musique_clean`
and `webqsp`. Those payloads carry graph topology, candidate indices, the four
expert signals and gold labels — but **no node or query embeddings**, and both
the residual and the displacements are functions of embeddings. They are also
marked `pilot_test_only` with *"No training split is present; do not report as
paper evidence"*, and every query in them is split 2, so they could not carry a
scientific result even if they held the tensors. `squad_clean` and `metaqa` are
not present locally in any form.

Stage 0 is therefore a Modal launch, and needs the compute record filed before
it runs — predicted jobs, hardware choice, walltime, storage, expected spend,
ceiling, and abort conditions. That record is
`docs/M2C_STAGE0_COMPUTE_RECORD.md`, and it is the next gate.

Validate cheaply, then expand only when the current scale produces evidence that
more compute can change a scientific decision. Ledger line:
`m2c_s4_structural_conditioning`.

## 15. Hard stop

**After Stage 0** — not after Stage 1, which this amendment removed from scope.
Produce, in order:

1. the exact residual × provenance matrix
2. directional error-margin diagnostics
3. direction-only metrics
4. fixed S4 + direction fusion metrics
5. the `G_STRUCT` / `G_KNN` / `G_FULL` result
6. `A64` versus `legacy-PF64` versus `seed-subspace-PF64`
7. exact `recall_ceiling@5` deltas
8. the blocker required-repair calculation
9. timing and storage
10. the two verdicts

**Two verdicts, returned independently:**

| question | values |
|---|---|
| ranking | `STOP_STRUCTURAL_M2C` \| `ADVANCE_STRUCTURAL_RANKING_M2C` |
| admission | `ADMISSION_CLOSED` \| `ADMISSION_REMAINS_PLAUSIBLE` |

They are separate mechanisms answering separate questions, and Stage 0 can
easily settle one without settling the other. A single verdict would force them
to move together and would hide which of the two the evidence actually spoke to.

If advancement is justified, propose the exact minimum learned Stage-1 matrix
and its cost — **do not run it**. Then `STOP_FOR_REVIEW`.
