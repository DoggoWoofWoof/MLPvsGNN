# M2D — S4 semantic / top-rank repair

Filed 2026-09-08, before any M2D arm is fit. The machine-readable declaration is
[`configs/m2d_s4_semantic_repair.yaml`](../configs/m2d_s4_semantic_repair.yaml);
this document is the argument behind it. Status:
`M2D_STAGE2_RECORD_FILED_NOTHING_SUBMITTED` — Stage 1 ran and its verdict
stands. The eight fits were submitted, fetched and verified, and the gate
committed before any of them existed returned `STOP_S4_DEVELOPMENT` on case 3.
The numbers are in [`docs/M2D_STAGE1_REPORT.md`](M2D_STAGE1_REPORT.md) and the
rule's own working in [`docs/M2D_STAGE1_GATE.md`](M2D_STAGE1_GATE.md); neither
is altered by what follows.

One qualification the case label does not carry: A3-MINIMAL's single shortfall,
on musique_clean/R1, was RESOLVABLE rather than FAIL. It missed the 0.50pp
admissibility guard by **0.065pp**, against a seed variation already measured at
**0.816pp** on that same cell — an order of magnitude wider than the gap. That
is the one state extra seeds could change, section 15's trigger did not fire
(RESOLVABLE is not a pass), and the gate's own
`extra_seeds.authorised_by_this_gate` is `false`, so none were run.

Review has now read that report and authorised the minimum additional seeds, in
the dated amendment at section 15b of the declaration. The accurate description
is not "trying more seeds because seed 0 failed": seed 0 landed inside a
**pre-measured decision-uncertainty band**, and the four fits are the smallest
matrix that can resolve the decision — A3-MINIMAL only, the two blockers only,
seeds 1 and 2 only, with seed 0 and every S3/S4 comparison row reused rather
than refit. No threshold moves. What the amendment adds is the aggregation the
guard is applied to, fixed in advance as the **mean of three same-seed
differences**, so that no seed can be selected after the fact. Filed with it,
before the fits exist, is the observation that this stage is *not* expected to
pass on seed 0's strength: native S4 degrades monotonically with seed on
MuSiQue, so seeds 1 and 2 must average better than seed 0 managed.

The rule that will judge those fits is `scripts/m2d_stage2_gate.py`, committed
with its 31 tests and with no Stage-2 artifact anywhere on disk — which is why
those tests run on synthetic fits. It refuses a verdict on fewer than three
seeds, refuses a Stage-2 file carrying seed 0 (that row is Stage 1's and is
reused, not refit), pairs each seed against its own S3 and S4 rows, and reports
the per-seed signs and sample SD beside the mean without letting them decide.

Its price is filed with it, in
[`docs/M2D_STAGE2_COMPUTE_RECORD.md`](M2D_STAGE2_COMPUTE_RECORD.md): four jobs,
four fits, 385.8 GPU-seconds, an expected **$0.79** against a **$2.00** hard
ceiling. Stage 1's record predicted its seconds by scaling M2B's; this one reads
them out of the Stage-1 A3-MINIMAL artifacts for these same two cells, so the
safety factor is 1.25 rather than 2x and the record says why. Every Stage-2 gate
in the declaration is now `true` and **nothing has been submitted** — a claim
that rests on `outputs/m2d_s4_semantic_repair/stage2/` not existing, which
`tests/test_m2d_declaration.py` checks against the disk. Stage 0 is complete and
reported. Section 3 ran and its findings are below; the Stage-0 jobs were priced
in [`docs/M2D_STAGE0_COMPUTE_RECORD.md`](M2D_STAGE0_COMPUTE_RECORD.md) and ran on
all four declared cells; condition B was measured afterwards on the two failure
cells, because the Stage-0 probe ranks whole models and B asks about a single
primitive, and a stop resting on an unmeasured condition is not a stop; section
10's gate was applied to all of it in
[`docs/M2D_STAGE0_GATE.md`](M2D_STAGE0_GATE.md), unchanged from the version that
returned the pending verdict; and section 20's eleven items, the verdict and the
stop are in [`docs/M2D_STAGE0_REPORT.md`](M2D_STAGE0_REPORT.md).

It advanced on B alone. Conditions A and C both fail on their filed terms: no
fixed fusion arm gains on either blocker, and the S3+S4 diagnostic loses
recall@5 on MuSiQue. B holds on exactly one of the five primitives S4 is
missing — `semantic_difference`, S3's learned weighted L1 at raw 1536 — which
reorders 59.5% of SQuAD's and 74.8% of MuSiQue's top-1 errors ranking alone. A
pass by one of five is a selection over five and is reported as one.

**The last gate is earned and nothing has run.** Section 6's compute record is
filed in [`docs/M2D_STAGE1_COMPUTE_RECORD.md`](M2D_STAGE1_COMPUTE_RECORD.md),
derived by `scripts/m2d_stage1_compute_record.py` from M2B's own measured
per-cell S4 training seconds rather than carried over from M2B's $0.92 — the
basis differs, and the record says how. With that, every gate in the
declaration is `true`, so the gate block can no longer express "and still
nothing has run". That claim now rests on the results directory being empty,
and `tests/test_m2d_declaration.py` refuses this status the moment a Stage-1
artifact appears.

**Stage 1 is now authorised, with an amendment attached.** The stop returned an
operator authorisation of the eight-fit pilot section 20 specified, and section
8b of the declaration files the amendment it came with, dated and against commit
`6290e97`, before any Stage-1 fit exists. Two things are resolved there. Arm A3
was written about "the two learned S3 diagonal interactions" and the measurement
implicates one, so the arm runs as **A3-MINIMAL** — S4 plus `semantic_difference`
only, 1,536 parameters against an authorised ceiling of at most 2 × 1536, which
is a narrowing of the envelope and not an expansion. And arm A1's columns are
derived by applying this file's own frozen admission rule to the diagnostics:
`cosine_qd` and `mean_abs_diff`, with `dot_qd_pct` excluded by its own named rule
and raw `dot(q, d)` excluded because nothing measured it. No effectiveness,
systems or parameter threshold is altered.

The pair is chosen to be causal rather than to be two guesses. A1's
`mean_abs_diff` is |q − d| reduced by the uniform weight 1/dim; A3-MINIMAL's
`semantic_difference` is the same |q − d| reduced by a learned v. On that column
the arms differ only in whether the 1,536 weights are free, so a null result
means the extra degrees of freedom bought nothing over a fixed uniform
weighting. They are not nested overall — A1 also carries `cosine_qd`, which
A3-MINIMAL does not — and no report may describe A1 as a restriction of
A3-MINIMAL. Both are ablations of **one** universal S4 ranker: one semantic
branch, one scorer, one forward pass. Nothing here proposes an ensemble, a
second model at inference, or a dataset router.

**This is a post-hoc development branch.** It was opened because of an observed
S4 development result, not from a preregistered hypothesis, and it is labelled
so wherever it is reported. M2B's rule rejected S2 and S4 as universal rungs and
selected S3; that decision is frozen. M2D asks a different question — whether
the specific defect that rejected S4 is repairable — and its answer, either way,
does not reinterpret M2B's.

---

## The question

> Can S4 retain its projected-semantic effectiveness and latency advantages
> while recovering the raw semantic / top-rank signal it needs on passage
> datasets?

Three things that question is **not**:

- It is not "is S4 better than S3". M2B answered that under a filed rule.
- It is not another graph-feature phase. M2C closed that, twice measured.
- It is not a search for an architecture. The ladder below is a list of
  *removals* from a prospective maximum, not a space to sweep.

---

## Why this phase is not M2C again

M2C asked whether deterministic graph geometry could repair S4's blockers and
returned two independent STOPs. The part that matters here is *why* it failed,
because the same failure mode is available to M2D and has to be designed
against.

Directional geometry was **defined for 0.99%–9.09%** of the candidates S4
scores. A signal that is undefined for nine candidates in ten cannot reorder a
top-1 list, and no amount of learned transformation over it could have. The
mechanism was ruled out by *coverage*, before effectiveness was even reached.

The semantic branch has the opposite property. Every candidate has an
embedding, so every semantic primitive is defined for 100% of the scored set.
Whatever M2D finds, it will not be a coverage null — which also means M2D
cannot borrow M2C's excuse if it fails. A semantic primitive that is defined
everywhere and still does not separate S4's errors is a real negative result
about the primitive.

M2C also removed a possibility this phase depends on not needing: candidate
admission. `ADMISSION_CLOSED` means the gold is where it is going to be. The
repair has to be an *ordering* repair among candidates that were already
scored, which is exactly the shape of the defect.

---

## The failure shape

Every number here comes from
`outputs/m2c_s4_structural_conditioning/m2b_baseline_table.json`, the immutable
export of M2B's 50 fits. Nothing was rerun to produce this section.

### The two blockers, three-seed means, S4 − S3

| cell | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|
| `squad_clean/R1` | −4.303pp | −0.882pp | −0.262pp | −2.615pp |
| `musique_clean/R1` | −12.505pp | −2.572pp | −1.108pp | −14.922pp |

Every one of these eight figures is negative in **every** seed. This is not a
seed artifact.

### The family split, seed 0, S4 − S3

| family | cells | R@1 | R@5 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| passage-graph | 8 | **−6.459pp** | +0.740pp | +0.314pp | **−7.335pp** |
| KB | 6 | +3.653pp | +5.487pp | +4.179pp | +8.947pp |

S4 loses R@1 in **every** passage cell and wins **every** metric in **every** KB
cell. The split has no exceptions.

### What the shape says, and what it does not

Read the passage row across: −6.459pp at rank 1, +0.740pp at rank 5, +0.314pp at
rank 20. The deficit *vanishes as the cutoff grows*. On passage graphs S4 puts
the gold in the candidate list about as often as S3 does and then **orders it
worse**. MRR, which is sensitive to exactly that, moves −7.335pp.

So the prospective interpretation is:

> S4's remaining failure is dominated by top-of-ranking ordering.

That is a statement about **shape**, and the declaration says so explicitly. It
is not yet a claim about cause. "S4 orders the top worse on passage data" is
what the numbers show; *why* is what §3 and Stage 0 are for. A repair chosen
now would be chosen from a story about the numbers rather than from evidence
about the mechanism — which is the specific error M2C's `prior_measured_evidence`
section exists to prevent repeating.

---

## Archaeology before repair

The declaration forbids proposing any arm before the exact live formulas for
S2, S3 and S4 are recorded. This is not ceremony. This track has already been
bitten once by a quoted constant: `V1_SEMANTIC_PARAMETERS` is 98,304, a fact
about a 768-dimensional payload, and the real cost at the frozen 1536 width is
twice that. Nobody would have found it by reading the number.

The same trap is available here in a nastier form. S4 emits **258** semantic
columns. It is easy, and wrong, to reason backwards from 258 to a guess about
what those columns are. The declaration says: do not infer the formulas from
258. Instantiate the head, read its `feature_names`, and count its parameters
with `numel()`.

What the archaeology must produce is two sets:

- **`S3_NOT_IN_S4`** — the semantic primitives S3 has and S4 does not. This set,
  and not a hypothesis about it, is the source of every candidate repair.
- **`S4_NOT_IN_S3`** — so a proposed addition can be checked against what S4
  already holds.

And one rule about how to write them: where S4 holds a *restricted* form of an
S3 primitive — the same quantity computed in a different basis, say — the
archaeology must say restricted-in-what-way rather than record it as absent.
Re-adding something S4 already has in another basis would spend parameters and
latency on a duplicate and then attribute any movement to the wrong cause.

### What it found

Run by [`scripts/m2d_semantic_archaeology.py`](../scripts/m2d_semantic_archaeology.py)
against live heads at 1536. The live counts reproduce what the fits recorded —
0, 3,072 and 196,608 semantic parameters — so this describes the same models
that produced the blocker deltas and not a lookalike.

| | semantic params | columns | comparison space | rank-aware columns |
|---|---:|---:|---|---|
| S2 | 0 | 3 | raw 1536 | `dot_qd_pct` |
| S3 | 3,072 | 5 | raw 1536 | `dot_qd_pct` |
| S4 | 196,608 | 258 | projected 64 | **none** |

Three of the findings are measurements rather than readings of the source, and
each was obtained by running the head twice — once on the candidate set, once on
the set minus one candidate — and comparing the rows both runs share.

**S4 has no set-dependent column at all.** Not one of its 258 columns changes
when a *different* candidate leaves the set. `dot_qd_pct` — the within-query
percentile of ⟨q, d⟩ — is the only rank-aware and the only query-relative
quantity in any of the three rungs, and S4 has no form of it in any basis. It is
the single entry in `S3_NOT_IN_S4` that is genuinely **ABSENT**.

**64 of S4's 258 columns cannot reorder anything.** The `query_state` block is
expanded across candidate rows, so it takes the same value for every candidate
of a query and was measured to have zero spread. The scorer can only use it to
shift a query's candidates together, which no within-query ranking metric can
see. S4's column count is 258; its *reordering* column count is 194.

**The other four `S3_NOT_IN_S4` entries are RESTRICTED, not absent.** S4 has a
cosine (`normalized_state_dot`), an L1 term (`state_absolute_difference`) and a
learned product (`state_product`) — all after two rank-64 projections and a
GELU. Every S4 column factors through those projections, so no setting of them
makes a column equal a full-rank form on the raw vectors; but a rank-64 form is
not *less* than S3's full-rank diagonal either, it is restricted differently.
Neither class contains the other. That is why none of these four may be
re-added as though S4 were missing them.

**Honest cost, both halves.** The scorer's first layer is as wide as the column
count. Deriving the slope from S2 and S3 gives 32 parameters per semantic
column, and that figure predicts S4's recorded non-semantic remainder of 8,609
exactly. So S4's 258 columns cost 8,160 scorer parameters *beyond S2's three*,
on top of the 196,608 in the projections.

None of this says what caused the passage blockers. It says which primitives
are available to which rung; sections 4 to 6 are what may implicate one.

---

## Stage 0: everything that can be answered without a fit

Stage 0 trains nothing and fits nothing. Every diagnostic in it is derivable
from per-query rows M2B already produced and from the Dense and SPLADE rankings
already stored for each query. §18 of the declaration makes that a rule rather
than a preference: paying GPU time for a number that is on disk is a cost with
no information attached.

### The four cells, chosen mechanically and in advance

| role | cell | held-out queries | why |
|---|---|---:|---|
| blocker | `squad_clean/R1` | 5 213 | mandatory |
| blocker | `musique_clean/R1` | 797 | mandatory |
| passage control | `hotpotqa_clean/R1` | 3 914 | largest S4-winning passage panel |
| KB control | `metaqa/R1` | 7 828 | largest S4-winning KB panel |

The control rule is: among cells where S4 *currently wins* R@5 at seed 0, take
the largest held-out panel, once within each family. Six passage cells and six
KB cells qualify.

This rule is worth stating because of what it *rejects*. `webqsp` shows by far
the biggest S4 advantage anywhere — +8.6pp to +9.9pp on R@5 — on a panel of
**63** held-out queries. A control chosen by effect size would have picked it,
and a 0.50pp protection threshold on 63 queries is a threshold on less than one
query. Choosing by panel size picks `metaqa`, with 7 828.

All four cells are R1. Both blockers are R1, and `hotpotqa` and `metaqa` each
have three equally-sized regimes, so holding the regime constant costs nothing
and means an arm's effect cannot be confounded with a regime change.

### A. Complementarity

The question is blunt:

> Are S3 and S4 semantically complementary, or is S4 simply a noisier version
> of S3 on passage data?

These are very different worlds and the counts separate them cleanly. If
S4-wrong/S3-right is large while S3-wrong/S4-right is near zero, S4 is
*dominated* on the passage family and there is nothing to fuse — the honest
conclusion would be that S4's projection loses information S3 keeps, full stop.
If both counts are large, the two representations disagree productively, and an
integrated representation might capture that at one model's cost.

Reported separately for passage and KB, because the whole point of the family
split is that pooling them would average a −6.459pp failure against a +3.653pp
success and describe neither.

### B. Fixed rank fusion, at a constant nobody chose here

`configs/candidate_budget.yaml` already fixes the RRF constant at **60**. It is
reused, not selected. No weight and no constant is searched, because a fusion
whose constant was tuned on these cells would be a small trained model wearing
a control's name — and the arm's whole claim is to be parameter-free.

| arm | list | eligible as a final model |
|---|---|---|
| `Z0` | S4 alone | reference |
| `Z1` | RRF(S4, Dense) | yes |
| `Z2` | RRF(S4, SPLADE) | yes |
| `Z3` | RRF(S4, Dense, SPLADE) | yes |
| `Z4` | RRF(S4, S3) | **no — diagnostic only** |

`Z4` deserves its own paragraph. Running both semantic models at inference
would defeat the systems object this phase exists to protect: the reason to
repair S4 rather than accept S3 is S4's latency, and a model that runs S3 too
has spent it entirely. `Z4` answers exactly one question — *does S3 contain
ranking information complementary to S4* — and a `Z4` number may never be
reported as a candidate result. It is evidence about whether a cheaper
integrated representation is worth designing, not a design.

`RRF(S3, Dense)` and `RRF(S3, SPLADE)` are computed too if the stored ranks make
them cheap. They are context: they say whether the *incumbent* is improvable by
the same parameter-free move, which is the only way to tell an S4-specific gain
from a gain any model would collect.

There is a reason to expect the fusion arms to be informative here where M2C's
were not. M2C's fusion collapsed catastrophically — R@5 fell 56 to 66pp — and
that was not a bug: it fused S4's 0.9131 against a directional list scoring
0.0092 to 0.1484, which is close to random. Equal-weight RRF against a near-random
list destroys a good list, mechanically. Dense and SPLADE are not near-random;
they are the retrievers that produced the candidate set. If these fusions
collapse too, that will mean something different, and the Z0 reference is there
to measure it against.

### C. Error-conditioned rescue

For every query where S4's top-1 is wrong **and** a relevant candidate is
already in the scored set, classify what would have rescued it — Dense, SPLADE,
S3, or nothing — at ranks 1, 5 and 20, split by family.

Queries whose gold was never a candidate are excluded and counted separately.
They are not ranking errors, and M2C's `ADMISSION_CLOSED` means they are not
going to become candidates.

This table is what determines whether the repair should preserve raw dense
geometry, lexical signal, the S3 diagonal interaction, or something else — and
the declaration forbids choosing a model before it exists.

One discipline carried forward from M2C without waiting to be bitten again:
**every conditioned statistic is reported beside the fraction of the population
it was measurable on.** M2C's margins were measurable on 6.5%–10.3% of the error
population, and a clean-looking margin on 7% of the errors is not a repair for
the cell. Pooling the statistic with its coverage hides precisely that.

---

## The ladder, and why it is a maximum rather than a plan

| arm | what it adds | runs only if |
|---|---|---|
| `A0` | nothing — frozen S4 | always, as the reference |
| `A1` | the smallest missing **raw-space** semantic scalars | the diagnostic names them |
| `A2` | parameter-free Dense RRF at k=60 | Stage 0 shows fusion helps |
| `A3` | the **exact existing** S3 diagonal parameters | complementarity implicates *those two* specifically |
| `A4` | a tiny zero-initialised semantic residual | A1–A3 show signal exists but static inclusion cannot reach it |

**Do not run all five.** The archaeology and Stage 0 *remove* arms; nothing adds
one. Five arms run and one reported would be a selection over five presented as
a result.

`A3` has the tightest condition on purpose. "S3 wins somewhere" does not
implicate S3's two learned diagonals — S3 also carries three parameter-free raw
scalars, and those are `A1`'s territory at a fraction of the cost. `A3` is
justified only if the diagnostic points at the *learned* interactions
specifically, and it adds them **unmodified**. Redesigning them would make the
arm a new model rather than a transplant, and it would no longer answer whether
S3's actual capacity is what S4 lacks.

### On `dot_qd_pct`

The single most conspicuous thing S4 lacks is likely to be query-relative rank
structure. The temptation to add it immediately is strong and the declaration
blocks it, because `dot_qd_pct` was the dominant latency source in S2 and S3.
Adding it back would spend the systems advantage that is the entire reason for
repairing S4 rather than accepting S3 — a repair that wins effectiveness and
loses the latency argument has removed its own justification.

So the order is fixed: ask first whether **cheap raw geometry alone** is
sufficient. Only if the error-conditioned table shows that percentile or rank
structure *specifically* — and not raw geometry — separates the relevant
candidate from S4's wrong top item does percentile become admissible, and then
as a **separate expensive control** with its own measured latency, never folded
into a cheap arm.

---

## If a residual is ever justified

`A4` is not authorised by this file and would need its own amendment. Its shape
is filed now so that the amendment is a decision about a known object:

```
base       = the S4 projected representation, unchanged
correction = a small function of SEMANTIC INPUTS ONLY
combined   = base + gated correction      (correction initialised to zero)
```

Zero initialisation is the discipline, not a detail. An arm that starts as its
own base cannot win by being *different*; it can only win by learning. S3's own
design uses the same trick — `w = 1/dim` makes its learned product start as the
mean elementwise product, so an S3-over-S2 gain is attributable to the learned
weights rather than to the channels existing.

Forbidden in the correction: graph conditioning of any kind, dataset identity,
any target-specific branch, and any multimillion-parameter adapter.

The control it needs is **not** a matched structural control — there is no
structural input to match. It is the simpler raw-semantic skip from `A1`, so
that extra nonlinear capacity has to earn its place rather than merely occupy
it.

---

## The gates

Both are filed here, before any diagnostic has run, and neither is adjustable
afterwards. M2C's experience is why that last clause is now checked by git
ancestry rather than by a flag inside the file claiming it: a self-attested
"filed before results" establishes nothing, since it lives in the file it is
making a claim about.

### Stage-0 advance

Advance only if **at least one** holds:

- **A.** A fixed zero-training fusion improves R@5 on **both** failure cells,
  with at least one improvement ≥ **+0.25pp**, and regresses **neither** control
  cell by more than **0.50pp** on R@5.
- **B.** The complementarity analysis shows one missing S3 or raw semantic
  primitive systematically ranks the relevant item above S4's wrong top item on
  **both** failure cells.
- **C.** The S3+S4 diagnostic RRF substantially repairs **both** failure cells.

Otherwise: **`STOP_M2D`**, freeze S3, and move to M3 with S3. The declaration
adds the sentence that matters most: *do not train another adapter because S4's
macro number is attractive.* An attractive aggregate, driven by the KB family
S4 already wins by +5.487pp, is not evidence about the passage family it loses.

Condition A does not name *which* fusion arm must improve, so all of them are
evaluated and the count that passed out of the count tried is reported. A pass
by one of four is a selection over four, and narrowing the gate to one arm
afterwards would be picking the comparison from the results.

### Effectiveness

A challenger `C` must be within **0.50pp of S3** on R@5 at both blockers, and
must not regress either control by more than **0.50pp** relative to S4.

R@1 and MRR are reported not as supporting detail but as the metrics the defect
lives in. The passage family loses **6.459pp at rank 1** and **7.335pp of MRR**
against **+0.314pp at rank 20**. A challenger that fixes R@5 and leaves rank 1
where it is has not repaired the failure this phase was opened for.

### Systems

Measured on full uncached inference with nothing excluded — raw-space
reductions, rank fusion, the semantic skip and any residual transform all count.

| rung | uncached p95 (blocker fits) | uncached p95 (all 18 fits) |
|---|---:|---:|
| S4 | 1.0683 ms | 1.0728 ms |
| S3 | 1.9138 ms | 1.9466 ms |

The gate uses the blocker subset, because that is where a repair has to hold.
Both subsets are recorded so that neither can be quoted selectively later.

**A repaired S4 must keep total uncached p95 below S3's.** Above it, the systems
reason for replacing S3 is gone, and the phase should say so rather than
reporting an effectiveness win as if it stood alone.

---

## Parameter accounting, stated honestly

| rung | semantic params | total params | uncached p95 | fits averaged |
|---|---:|---:|---:|---:|
| S2 | 0 | 449 | 1.6837 ms | 14 |
| S3 **(incumbent)** | 3 072 | **3 585** | 1.9466 ms | 18 |
| S4 (challenger) | 196 608 | **205 217** | 1.0728 ms | 18 |

S4 is the **large** model in this comparison, by a factor of 57. Any sentence
implying otherwise is false, and the declaration forbids writing one.

The target story is therefore **not** "fewer parameters than everything". It is:

> A projection-heavy feed-forward representation maps efficiently to GPU kernels
> and, with a minimal semantic-preservation mechanism, can deliver high
> retrieval effectiveness without learned message passing.

That is a claim about *where the parameters go* and how they map to hardware,
not about how few there are — and it is falsifiable by the systems gate above.

---

## Infrastructure first

Before any M2D compute, the persistence failure M2C discovered is closed.

A write that **overwrites an existing file** on the Modal result volume was
silently discarded at commit: the container saw its own bytes, reported
success, and the old file survived, while brand-new files from the same batch
persisted normally. It happened twice, and the only reason it was caught is
that the stale artifact recorded a pre-fix `source_commit`.

[`src/mp_retrieval/run_artifacts.py`](../src/mp_retrieval/run_artifacts.py)
addresses every scientific artifact by phase, dataset, regime, seed, arm,
source commit and run id, where the run id is Modal's own function-call id.
Two runs — including a rerun after a bug fix — therefore *cannot* address one
path, so the store never sees a second write to an existing key and has nothing
to discard. A second write to an existing artifact path is **refused**, not
unlinked: unlinking makes an overwrite safe, refusing makes it impossible.

After writing, the artifact is reopened and its source commit, run id, config
fingerprint, row count and content hash are checked — the last two recomputed
rather than read back from fields that claim the answer.

That in-container check is recorded in the declaration as **not sufficient**,
because it is the trap: the copy a container reads is the copy that gets
discarded. The check with teeth is host-side after fetching, and only a
verified artifact may be exposed as the selected logical result. Where several
runs exist for one logical result, the caller *names the commit*; two verified
runs at the same commit is an ambiguity to resolve, never a tie broken by
timestamp.

[`tests/test_run_artifacts.py`](../tests/test_run_artifacts.py) reproduces the
failure class rather than describing it: it models the volume's measured
behaviour, shows the old fixed-path scheme reproducing the incident *including
the deceptive successful read-back*, and shows the new scheme unable to reach
that state.

---

## What this phase may not touch

M2, M2B, M2C, M3/GNN, Package F, canonical CRAG and E2 are all frozen or out of
scope. No test split is read. No GNN output, hidden state, teacher or
distillation target may enter M2D in any role.

The **scored universe is frozen**: `SUPPORT`, `PATH`, `NODE_ROLE`, `A64`, the
R1/R2/R3 definitions, graph context, graph provenance, candidate admission and
the candidate budget all stay exactly as M2/M2B left them. M2D changes the
semantic comparison and nothing else. If the candidate set moved, an
effectiveness change would be uninterpretable — it could belong to the
semantics or to the new candidates.

And the rule with the sharpest edge:

> If semantic repair fails, **stop**. Do not return to graph offsets looking for
> another variant.

A third directional residual would be the same experiment under a new name, and
that null has now been measured twice.

---

## Stop

After Stage 0 — or Stage 1, if it is legitimately reached — report the exact
S3-versus-S4 semantic difference, the passage-versus-KB error decomposition, the
S4-wrong / S3-right counts, the Dense/SPLADE/S3 rescue fractions, the fixed RRF
results, the selected minimal repair if any, the blocker deltas, R@1/R@5/R@20/
MRR, p50/p95/p99, the parameter count and the compute spend.

Then one verdict — **`STOP_S4_DEVELOPMENT`** or **`ADVANCE_TARGETED_M2D`**. If
STOP: freeze S3, and the next phase is M3 independent GNN development. If
ADVANCE: give the exact minimum next matrix and its cost, and do not run it.

Then `STOP_FOR_REVIEW`.
