# Stage D10: the replacement trades Recall@5 for MRR, and the rule that killed D9 was measuring the machine

D9 defined a bounded branch-diversity replacement for the historical PATHS
family, measured how far the two representations diverge, and then stopped at a
rule that raced its whole construction wall clock against the historical
feature build. D10 repairs that rule, rebuilds the block D9 discarded, and fits
the one arm D9 never reached.

Two results, and the second one is the one that changes how the next stage
should be run.

    V - H, validation, x100:  R@1 +0.48   R@5 -0.67   R@20 +0.04
                              MRR +0.62   FullCov@20 +0.03

    -> MIXED. Materially better on MRR, materially worse on R@5,
       at the +/-0.005 threshold filed before the run.

The replacement is not an improvement on the historical family. It is a
reallocation: it finds a first correct node more often and a correct top-5 set
less often.

The systems result: this container built the historical features in 50.7 s and
the whole branch construction in 49.5 s -- a ratio of **0.9779** -- so D9's
original rule would have passed here, on byte-identical work it aborted at
1.2449 on a slower host. Same code, same rule, two containers, opposite
answers.

2Wiki, seed 0, `G[Cq]`, `CAND`. 13,500 queries opened, 3,000 reported;
4,851,276 candidate rows rebuilt. One fitted arm, `D10_PATH_DIVERSITY_13`;
`D6_BASE_13` and `D7_PATHS_13` reused from D7's result file and never refitted.

## The downstream decision

What later stages read off D10 is one line, and it is neither band label:

    D10 = MIXED / NOT PROMOTED

Branch diversity is **not promoted** over historical PATHS. The R@5 loss
exceeds the +/-0.50 pp tolerance filed before the run, so the replacement does
not enter the Pareto-selected 2Wiki feature subset. Historical PATHS is not
promoted either -- it remains a feature-family control, since "the correction
does not beat it" is not the same as "admit the historical columns".

Both preregistered labels stay in the provenance below, because both were
filed. Neither is what feature selection reads.

## The label is contested, and the favourable reading is the wrong one

Two statements of the interpretation bands were both filed before the
measurement, and they disagree about this specific outcome.

| Source | Filed | Gives |
| --- | --- | --- |
| the D10 declaration's predicates | before this run | `PATH DIVERSITY FAILS` |
| `classify()`, inherited from D9 | before D9 ran | `PATH DIVERSITY PARETO-MATCHES` |

`PATH DIVERSITY PARETO-MATCHES` is the string inside `stage_d10.json`, because
that is what the runner computed and the result file is not edited afterwards.
It is also the wrong reading, and it is wrong in the direction that flatters
the work.

Applied to the measured increment, exactly one filed predicate is satisfied:

- `IMPROVES` requires materially positive on R@5 or MRR **and** not materially
  negative on any metric. MRR is +0.62; R@5 is -0.67. Not satisfied.
- `PARETO-MATCHES` requires that **no** metric move materially in either
  direction. Two did. Not satisfied.
- `FAILS` requires materially negative on R@5 or MRR. R@5 is. **Satisfied.**

D9's implementation routes anything mixed into a catch-all `else` branch
labelled with the Pareto string. That branch had never executed: D9 aborted
before training, so no D-stage had ever produced a mixed result at this
architecture. The band set is not exhaustive over the outcome space, and
nothing filed before D10 says what a mixed outcome is called.

The two labels carry different instructions. The Pareto one says *retain the
replacement on Pareto grounds: bounded, deterministic, intrinsic and cheaper,
an accuracy increase is not required.* `PATH DIVERSITY FAILS` says *do not add
walk counts back*, and treats the outcome as evidence about how much further
structural engineering is worth. Taking the runner's answer because the runner
emitted it would be selecting a label for what it licenses.

The filed predicates are taken. `classify()` is **not** repaired here, and
neither is `stage_d10.json`. Changing the classifier after seeing the outcome
it produced is the result-driven rewriting D9 refused when it found its own
cost rule defective, and the same answer applies to a defect found in my
favour. The band-set defect is recorded and left standing; whoever files the
next band set should make the bands exhaustive and name the mixed case.

One consequence worth stating plainly rather than leaving to be noticed. The
document test `tests/test_graph_context_d10_doc.py` was written before this run
and asserted that exactly one of the three band strings could appear in this
write-up. That assertion inherited the same non-exhaustiveness assumption the
band set has, and a contested outcome cannot be disclosed under it. It was
amended **after** the outcome was known, to require that a disagreement be
disclosed -- both labels named, the runner's answer attributed to
`classify()`, and the reading taken stated -- and to keep the single-label rule
whenever the two agree. The amendment makes the test stricter and is recorded
here because a test edited after a result is exactly the kind of change a
reader should be told about rather than left to find in the diff.

Both readings agree on the substance, which is what matters: the bounded
branch-diversity representation is not an accuracy win over the historical
normalized walk-count representation at this operating point.

## What the arms actually scored

Validation, 3,000 queries, x100.

| arm | R@1 | R@5 | R@20 | MRR | FullCov@20 |
| --- | --- | --- | --- | --- | --- |
| `D6_BASE_13` | 39.39 | 71.38 | 77.70 | 93.88 | 51.90 |
| `D7_PATHS_13` | 39.27 | 72.20 | 77.83 | 93.73 | 52.20 |
| `D10_PATH_DIVERSITY_13` | 39.75 | 71.53 | 77.88 | 94.35 | 52.23 |

Three increments, x100.

| increment | R@1 | R@5 | R@20 | MRR | FullCov@20 |
| --- | --- | --- | --- | --- | --- |
| `H - B` historical paths | -0.12 | +0.82 | +0.13 | -0.15 | +0.30 |
| `V - B` branch diversity | +0.36 | +0.14 | +0.18 | +0.47 | +0.33 |
| `V - H` the comparison | +0.48 | -0.67 | +0.04 | +0.62 | +0.03 |

`H - B` reproduces D7's recorded `delta_paths` at `max_abs_diff` 0.0, read from
D7's rows, without refitting `H`.

The two families move different metrics in opposite directions from the same
base. `H` buys R@5 and gives back MRR. `V` buys MRR and R@1 and barely moves
R@5. Both beat the base on four metrics of five. Whatever walk multiplicity
encodes, bounded branch diversity does not strictly subsume it, and D9's
representation divergence at hops 2 and 3 is now shown to be a divergence that
reaches ranking rather than one that washes out.

## Where the trade happens

`V - H` by gold stratum, x100.

| stratum | queries | R@1 | R@5 | R@20 | MRR | FullCov@20 |
| --- | --- | --- | --- | --- | --- | --- |
| isolated | 834 | +0.24 | +0.60 | +0.27 | +0.32 | +0.48 |
| degree_1 | 750 | +0.40 | -0.60 | +0.07 | +0.65 | +0.13 |
| low_degree | 1218 | +0.92 | -1.46 | -0.08 | +1.03 | -0.25 |
| ordinary | 198 | -1.01 | -1.52 | -0.25 | -0.80 | -0.51 |

Isolated queries are the only stratum where the replacement wins on all five.
The R@5 loss is concentrated in `low_degree`, which is also where the R@1 and
MRR gains are largest -- the same 1218 queries are being reordered, not
different ones. The 198 `ordinary` queries lose on all five.

## The systems rule was measuring the host

D9 aborted on this rule:

    stop before training if the branch-diversity construction time exceeds
    the historical local feature build time

D10 filed this one, before the runner was launchable:

    abort if the replacement kernel p95 per query exceeds the historical
    structural build p95 per query on the same container, or if the temporary
    workspace exceeds the declared bounded contract. Shared candidate and
    graph extraction is reported separately and is not charged.

Three containers, the same abort quantity, read from each stage's own result
file:

| | D8 | D9 | D10 |
| --- | --- | --- | --- |
| historical build | 51.4 s | 74.2 s | 50.7 s |
| whole construction | 42.0 s | 92.4 s | 49.5 s |
| construction / build | 0.8166 | 1.2449 | 0.9779 |
| D9's rule says | pass | **abort** | pass |

D9's container ran the same historical build 1.44 times slower than D8's, and
paid 79.24 s of shared induced-edge extraction against D8's 39.22 s. This
container is back at D8's speed -- build p95 3.943 ms against D8's 4.083 --
and the identical construction D9 aborted comes in at 97.79% of the build,
under it rather than over.

The repaired rule was filed with an argument rather than an outcome: a
total-seconds race is a ratio of one fixed cost to another, so a host that runs
the shared build slowly moves the threshold instead of the measurement. D10 did
not need the argument. The defect is now measured directly.

What the repaired rule saw:

| quantity | value |
| --- | --- |
| replacement kernel p95 | 0.611 ms per query |
| historical structural build p95 | 3.943 ms per query |
| ratio | 0.155 |
| kernel total | 7.07 s, 13.95% of the build |
| shared induced-edge extraction | 40.86 s, reported, not charged |
| temporary workspace | 6368 bytes against a 1 MiB backstop |
| fixed graph passes | 3, no convergence loop |

Two things about that rule need saying outright. First, it was repaired
**after** the abort it caused, and it permits exactly the work the old rule
blocked -- the shape of result-driven rewriting. The declaration says so in
those words, states in advance that D9's own recorded percentiles (0.877 ms
kernel against 8.432 ms build) would have passed it, and rests the repair on
the argument above rather than on D10's outcome. Second, the abort stayed live:
had this container reversed the percentiles, D10 would have stopped without
fitting, exactly as D9 did.

The 1 MiB workspace figure is a **new** backstop filed for D10. An earlier
draft of the declaration described it as something `tests/test_path_diversity.py`
already asserted. It does not -- that test asserts the exact allocation
identity, two words per node per 64 support elements, which is the guarantee
with teeth and which does predate D10. The absolute ceiling is a pathology
tripwire, and the observed 6368 bytes sits 164 times under it.

## The rebuild, and why there had to be one

D9 built the replacement block in memory and discarded it at the abort,
persisting only its JSON result. Checked before any D10 code was written, the
stage directory holds fourteen files, `stage_b.json` through `stage_d9.json`,
and no feature tensor. No durable artifact was reusable.

So D10 rebuilt all 4,851,276 rows. That is forced by D9's failure to persist,
not chosen for stage symmetry, and rebuilding unverified would have been worse
than not rebuilding at all. Every content-sensitive statistic D9 recorded was
read off `stage_d9.json` at full precision and required to match:

| condition | D10 | D9 |
| --- | --- | --- |
| candidate rows | 4,851,276 | 4,851,276 |
| max branch, hops 1/2/3 | 7 / 113 / 255 | 7 / 113 / 255 |
| pairs reordered, hop 1 | 0.0009358088563998761 | identical |
| pairs reordered, hop 2 | 0.00370229089220594 | identical |
| pairs reordered, hop 3 | 0.008446842308984613 | identical |
| nonzero agreement, hops 1/2/3 | 1.0 / 1.0 / 1.0 | identical |
| distinct scalars before / after cast | 138 / 78 | 138 / 78 |

All thirteen conditions matched, the ordering fractions to full recorded
precision. The rebuilt block is D9's block.

D10 records `sha256:400443606913a3b0b7931849e4b8108a1789b7227381446a3dc7a43e322783a0`
for the int16 branch counts and
`sha256:e6c3560e231b1d4bb5333f7f7a126a91671a26e421455f9dd71b6488693a4d34`
for the injected float16 columns. The tensor itself is not persisted -- D10 is
the last 2Wiki-only feature-development stage and no declared consumer exists
for the block -- but the recorded hash lets any later stage verify a rebuild in
one comparison instead of thirteen, which is the thing D9 lacked and D10 had to
pay for.

Two diagnostics D9 already reported were **not** recomputed: D8's distinct
support overlap, and the all-opened mechanistic pass. Their absence is by
design, not omission, and it is part of why this container cost less than
projected.

## What the float16 block cannot represent

The frozen 13-column block is float16. Above a branch count of 51 the stored
saturation is no longer distinguishable from that of the next branch count.
The boundary is not taken on trust from D9: it is derived from the stored dtype
in front of the diagnostic and then checked against D9's 51.

| quantity | value |
| --- | --- |
| total rows | 4,851,276 |
| rows the cast cannot separate | 134 |
| fraction of all rows | 0.003% |
| rows with any nonzero PATH value | 666,517 |
| fraction of nonzero PATH rows | 0.020% |

All 134 affected rows are affected at hop 3 and 33 of them also at hop 2; hop 1
never reaches the boundary. The historical columns are log1p ratios in the same
dtype and pay the same ceiling, so this advantages neither arm. It is recorded
as a documented limitation of the frozen block. The dtype was not changed here,
and this does not open a further stage.

(The hop maxima behind that table are all-opened maxima, 8 / 121 / 255. The
reproduction table's 7 / 113 / 255 are validation-slice maxima. Different
populations, not a contradiction.)

## No feature redesign was possible

The transform, its tensor guards, its injection check and its mechanistic
statistic are imported from `scripts/run_graph_context_d9.py` rather than
restated. There is no second copy of the saturation formula in the tree that
could have been retuned once D9's divergence statistics were known, and a test
asserts the imported objects are the same objects.

`V` differs from `D7_PATHS_13` in columns 5-7 and nowhere else, `max_abs_diff`
0.0 on every other checked block. Columns 0-3 carry `D6_BASE_13`'s exact
distance geometry, columns 10-12 the exact D4/D5/D6 graded retrieval prior,
columns 4, 8 and 9 are exactly zero. The injected scalars reach the learner as
the bounded saturation itself, not divided a second time by a per-query
maximum. All three arms carry 213,689 parameters at head width 61.

## Cost

| | |
| --- | --- |
| container | 149.0 s |
| GPU-hours | 0.041 |
| cost | $0.07 |
| authorised | 0.08 GPU-h and $0.20 |
| projected | 240 s |
| fitted runs | 1 |

Measured from the Modal call, not from the result file. Under both ceilings and
under the projection -- not because reuse saved anything, since there was
nothing to reuse, but because two already-reported diagnostics were dropped and
this container ran the shared build at D8's speed rather than D9's.

Splits 9450 / 1050 / 3000. Epoch selection on a deterministic tail of train,
one validation read per arm, test split unread.

## What this does not establish

The design does not separate the change in counting rule from the change in
numerical transform and the calibration the learner sees. Attribution to walk
multiplicity alone is not identified by this design, in either direction. It is
**not** concluded that removing duplicate or cyclic walks caused the MRR gain,
and it is **not** concluded that raw repeated walks are inherently useful
because they held R@5. Both claims would need an arm this stage does not have.
The sentence filed for a V-over-H outcome is not used here, because V does not
win overall.

One replacement, one historical family, one dataset and one seed. Nothing here
is evidence about weighted paths, disjoint paths, motif structure, or the
composition of bounded branch diversity with D8's distinct support -- that
composition was explicitly not trained. Development evidence on `G[Cq]`, not an
evaluation, and not a statistical significance claim.

## Verdict

**`D10 = MIXED / NOT PROMOTED`** is the downstream decision.

Under the filed predicates the band is `PATH DIVERSITY FAILS`, recorded as a
MIXED outcome the band set did not anticipate.
`PATH DIVERSITY PARETO-MATCHES` stands in `stage_d10.json` as what the
inherited `classify()` computed. Both are kept as provenance; the decision line
above is what selection reads.

The bounded branch-diversity representation does not win over the historical
normalized walk-count representation on 2Wiki. It trades -0.67 R@5 for +0.62
MRR and +0.48 R@1, concentrated in low-degree queries, and loses on every
metric for the 198 `ordinary` ones.

D9's abort was not evidence about the feature. It was evidence about the
container: the same construction reads 1.2449 of the build on one host and
0.9779 on another. That rule is repaired, the repair is filed with the
admission that it permits what it once blocked, and the abort stayed live.

This is the last 2Wiki-only feature-development stage.
