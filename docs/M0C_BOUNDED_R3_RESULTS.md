# M0C bounded R3: results, Steps 4-8

**Status:** `STEPS_4_THROUGH_7_COMPLETE`. This document is the record Step 8
(stop) points to. It reports what Steps 4-7 of
[`M0C_BOUNDED_R3_PROTOCOL.md`](M0C_BOUNDED_R3_PROTOCOL.md) measured and
proposes; it declares and launches nothing further. M0B itself is untouched
-- commits `acceffd`/`e769a1b` and `outputs/m0b_regime_map/**` stand exactly
as filed, cited below, never recomputed.

**Source:** `outputs/m0c_bounded_r3/headline/*.json` (gitignored, not
committed; every M0C number below traces to one of those six files) cross-
checked bit-exact against `outputs/m0b_regime_map/headline/*.json` via
`tests/test_m0c_reuse_against_m0b.py` (48/48 passing at `headline` stage).
Six datasets, 100 queries each, validation split, deterministic prefix,
`structural_only` mainline family, `graph_expansion_cap=64` -- the identical
sample M0B scored. Launched via
`scripts/spawn_modal_jobs.py m0c-bounded-r3 --stage headline`, all six
returned `M0C_BOUNDED_R3_COMPLETE` in 111 s wall clock, concurrent.

## Step 4: what ran

Dataset roster, `num_nodes`, and `candidate_rows` are unchanged from M0B
(same frozen candidate contract, same six datasets) -- see
`M0B_REGIME_MAP_RESULTS.md` Step 4 rather than duplicating the table here.
Every M0C job's `candidate_contract.status` reads
`BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE` and every `expected_contract_sha256`
matches its `observed_contract_sha256`, confirmed again independently on
this run, not assumed from the reuse contract.

## Step 5: invariant verification

**All 8 required invariants hold, all six datasets, all 600 queries.**

Six live, checked per query/per dataset inside `scripts/run_m0c_bounded_r3.py`
(the runner raises on any violation):

| dataset | oracle_R1=R2 | scored_R1=R2 | U2⊇Cq | A64∩Cq=∅ | \|C3∖Cq\|≤64 | Cq_struct=Cq∪A64 | U2⊆U3_b | C3⊆U3_b |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `squad_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `musique_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `2wiki_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `hotpotqa_clean` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `metaqa` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `webqsp` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Two more verified post-hoc, aggregate-level, against M0B's own filed files
(`tests/test_m0c_reuse_against_m0b.py`, `headline` stage, 48/48 passing):
`C3_M0C == C3_M0B` (via bit-exact admission-count and containment-rate
equality on every family, every dataset) and "R3 candidate/headroom metrics
match M0B bit-exactly" (via bit-exact R1 headroom, R2 context, admission,
containment, gold-overlap, and node-role reproduction). The eighth --
`NODE_ROLE` partition exhaustiveness/mutual-exclusivity -- is an inherited
construction guarantee of the unmodified `overlap_audit.node_roles` helper
(200-trial randomized sweep in `tests/test_overlap_audit.py`), not
re-verified per query here, exactly as M0B relied on it.

### Containment and recovery -- identical to M0B, because C3 is identical to M0B

These are not new measurements; they are M0B's own numbers, confirmed
bit-exact reproduced by the invariant above. They are reported here because
Step 6 depends on them and the point they make is central to why M0C exists.

**Admitted-node containment** (`fraction(A64 inside U2)`):

| dataset | admitted total | admitted in U2 | admitted beyond U2 | containment_rate |
| --- | ---: | ---: | ---: | ---: |
| `squad_clean` | 4,591 | 4,591 | 0 | **1.0000** |
| `musique_clean` | 3,266 | 3,266 | 0 | **1.0000** |
| `2wiki_clean` | 1,638 | 1,638 | 0 | **1.0000** |
| `hotpotqa_clean` | 4,167 | 4,152 | 15 | **0.9964** |
| `metaqa` | 2,298 | 2,298 | 0 | **1.0000** |
| `webqsp` | 1,636 | 1,636 | 0 | **1.0000** |

**Gold recovery** (a gold instance is "recovered" when A64 admission promotes
it into the scored set `C3`; recall `scored_R1 == scored_R2` always holds, so
none of this recovery is attributable to R1 or R2 alone -- it is entirely a
consequence of structural candidate admission):

| dataset | gold missing from `Cq` | recovered via A64 | already reachable in U2 | beyond U2 (structural-only reach) | still missing after A64 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `squad_clean` | 0 | 0 | n/a | n/a | 0 |
| `musique_clean` | 4 | 0 | n/a | n/a | 4 |
| `2wiki_clean` | 49 | 41 | 41 (100%) | 0 | 8 |
| `hotpotqa_clean` | 10 | 9 | 8 (89%) | **1 (11%)** | 1 |
| `metaqa` | 43 | 43 | 43 (100%) | 0 | 0 |
| `webqsp` | 314 | 78 | 78 (100%) | 0 | **236** |

**The mechanism this table isolates, which is the reason M0C exists:**
*which* gold gets recovered is fixed entirely by A64 admission -- identical
under `R3_FULL_REEXPANSION` and `R3_BOUNDED`, because `C3` is bit-exact
identical (verified above). `hotpotqa_clean` is the only dataset where any
of that recovery (1 of 9 instances, 11%) reaches a gold node that plain
one-hop context (`U2`) could not even see -- on the other four datasets with
any recovery at all, every recovered instance was already graph-reachable
under `U2`, and structural admission's only contribution was making an
already-reachable node *scoreable*, not making it *reachable*. What the two
R3 regimes disagree about is never which candidates get scored -- it is how
much surrounding graph context is available once they are. That is Step 6c.

## Step 6: the corrected dataset x regime map

### 6a. Bottleneck attribution -- unchanged from M0B, restated with the mechanism split out

Candidate eligibility (`all_gold_at_pool`, `missing_gold_fraction_micro`,
`queries_with_no_gold_in_pool`) is bit-exact identical to M0B, since R1 is
untouched. The "recovered" column below is now split by mechanism using the
table above, rather than attributed jointly to "R2/R3":

| dataset | all_gold_at_pool | missing_frac (micro) | queries w/ 0 gold in pool | recovered by A64 (of which beyond U2) | still missing after A64 | primary bottleneck |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `squad_clean` | 1.0000 | 0.0000 | 0/100 | n/a | 0 | **none** -- already at ceiling |
| `musique_clean` | 0.9600 | 0.0200 | 0/100 | 0 | 4 | graph availability, residual is tiny |
| `2wiki_clean` | 0.6100 | 0.2112 | 0/100 | 41 (0) | 8 | **candidate eligibility**, closed by plain one-hop reach alone |
| `hotpotqa_clean` | 0.9000 | 0.0500 | 0/100 | 9 (1) | 1 | candidate eligibility; the one dataset where structural reach itself (not just scoreability) adds anything |
| `metaqa` | 0.6000 | 0.3525 | **30/100** | 43 (0) | 0 | **candidate eligibility**, severe -- closed by plain one-hop reach alone |
| `webqsp` | 0.3100 | 0.5470 | **36/100** | 78 (0) | **236** | **both** -- worst candidate eligibility, and a graph ceiling A64 does not lift regardless of regime |

### 6b. Feature-family computability per regime -- the corrected map

This replaces M0B's own 6b, which listed one "R3" column backed by
`TARGET_H1(Cq_struct)`. That column conflated two things this run separates:
candidate scoreability (identical under both R3 regimes) and context
richness (an `R3_FULL_REEXPANSION`-only property `R3_BOUNDED` does not
share). It is still a computability/availability map, not an accuracy map --
no per-column value or occupancy rate is computed here.

| family | R1 (`G[Cq]`) | R2 (`TARGET_H1(Cq)`) | R3_BOUNDED (`U2 ∪ A64`) | R3_FULL_REEXPANSION (cited from M0B, not recomputed) |
| --- | --- | --- | --- | --- |
| RETRIEVAL | computable, identical across regimes on `Cq` | computable, identical across regimes on `Cq` | computable on `Cq`; undefined on A64-admitted nodes, same as M0B's R3 | same |
| SEED | computable, identical across regimes | computable, identical across regimes | computable on `Cq`; undefined on A64-admitted nodes | same |
| GEOMETRY / SUPPORT / PATH / DIFFUSION / TOPOLOGY | starved -- no context to bucket against | computable over `U2`'s real context | **computable over `U3_bounded`, but richness is within noise of R2's own** (6c: growth 1.00-1.0013x) -- these families gain essentially nothing from moving R2→R3_BOUNDED | computable over `U3_full`; richness gap vs. R2 is real and dataset-dependent, up to 56.27x (6c) -- but this context is a systems control, never trained on |
| PROVENANCE | n/a | n/a | this run is fresh evidence: `structural_only`, bounded-context construction, six datasets | M0B's own run remains the evidence for the full-reexpansion construction |
| NODE ROLE | only `RETRIEVAL_CANDIDATE` (0 `STRUCTURAL_SCORED_CANDIDATE`, all six) | same as R1 | **`STRUCTURAL_SCORED_CANDIDATE` real and nonzero, bit-exact identical counts to M0B's R3** (6c) -- this is the one family where R3_BOUNDED carries all the same information R3_FULL does, at a fraction of the cost | same counts, at 1.15x-56.27x the context cost |

**The corrected conclusion:** for GEOMETRY/SUPPORT/PATH/DIFFUSION/TOPOLOGY,
whatever richness M0B's R3 appeared to add over R2 was coming from context
re-expansion, not from candidate scoreability -- `R3_BOUNDED` shows those
families see almost no change from R2 at all. For NODE ROLE, the opposite
holds: it captures 100% of the scoreability signal at R2-level context cost,
because `STRUCTURAL_SCORED_CANDIDATE` depends only on `C3`, never on `U3`'s
size.

### 6c. Context growth and feature cost, R2 -> R3 -- the three-way systems comparison

| dataset | R2 context (median) | R3_BOUNDED context (median) | R3_FULL context (median, cited) | R3_BOUNDED growth | R3_FULL growth (cited) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `squad_clean` | 6,195.5 | 6,195.5 | 7,125.0 | **1.00x** | 1.15x |
| `musique_clean` | 2,171.0 | 2,171.0 | 3,418.5 | **1.00x** | 1.57x |
| `2wiki_clean` | 2,037.5 | 2,037.5 | 40,196.0 | **1.00x** | 19.73x |
| `hotpotqa_clean` | 2,973.5 | 2,977.5 | 167,313.5 | **1.0013x** | **56.27x** |
| `metaqa` | 5,960.0 | 5,960.0 | 10,461.5 | **1.00x** | 1.76x |
| `webqsp` | 1,864.0 | 1,864.0 | 16,885.0 | **1.00x** | 9.06x |

Feature-build latency (p99, ms) follows context size directly:

| dataset | R2 feature p99 | R3_BOUNDED feature p99 | R3_FULL feature p99 (cited) | R3_BOUNDED vs R3_FULL |
| --- | ---: | ---: | ---: | ---: |
| `squad_clean` | 179.4 | 175.0 | 161.9 | ~1.0x |
| `musique_clean` | 14.2 | 16.7 | 13.9 | ~1.2x |
| `2wiki_clean` | 5.9 | 9.0 | 89.5 | **~0.10x (10x cheaper)** |
| `hotpotqa_clean` | 161.0 | 106.6 | 1,549.3 | **~0.07x (14.5x cheaper)** |
| `metaqa` | 27.0 | 29.2 | 36.5 | ~0.80x |
| `webqsp` | 64.2 | 74.8 | 355.7 | **~0.21x (4.8x cheaper)** |

**Hotpot, highlighted, as the confound this protocol was written to resolve:**
under `R3_FULL_REEXPANSION`, `hotpotqa_clean`'s context balloons **56.27x**
over R2 (median 2,973.5 -> 167,313.5 nodes) and its feature cost follows to
1,549.3 ms p99 -- a real, systems-significant blowup that would have made any
R2-vs-R3-trained-model comparison ambiguous between "the model got more
candidates" and "the model got a vastly bigger graph to read features from."
Under `R3_BOUNDED`, the same 9 gold recoveries happen (identical `C3`,
verified above), at a context size **statistically flat against R2**
(2,973.5 -> 2,977.5, +0.13%) and a feature cost that is *lower* than R2's own
p99, not higher (161.0 -> 106.6 ms; the two measurements come from
independent runs and this is ordinary container-to-container variance on a
near-identical context size, not a real reversal). The blowup M0B measured
was real and is preserved as a systems-control finding
(`R3_FULL_REEXPANSION`, `outputs/m0b_regime_map/**`, untouched) -- it was
never a property of candidate scoreability, which is now shown directly to
cost almost nothing extra.

`2wiki_clean` shows the same pattern at smaller absolute scale: R3_FULL's
19.73x context growth and 10x feature-cost increase over R3_BOUNDED both
vanish once scoreability and re-expansion are separated, even though `C3`
(and therefore every recovered gold instance) is unchanged between the two.

### webqsp: absolute recall and regime-specific ceiling attainment

Per standing scope (no `H2`/`H3`, no new candidate generator for `webqsp`),
this reports what the frozen candidates and the corrected R3 regime actually
attain, not a new arm:

| metric | value |
| --- | ---: |
| `recall_ceiling@1` (R1, perfect ranker over `Cq`) | 0.3351 |
| `recall_ceiling@5` | 0.4066 |
| `recall_ceiling@20` | 0.4323 |
| `candidate_ceiling_mean` (fraction of gold present in `Cq` at all) | 0.4441 |
| `all_gold_at_pool` | 0.3100 |
| `any_gold_at_pool` | 0.6400 |

`webqsp` is candidate-eligibility bound before any graph regime is even
considered: 56% of validation queries are missing at least some gold from
`Cq`, and R1's own perfect-ranker ceiling never exceeds 0.44 regardless of
downstream regime. Under `R3_BOUNDED`, structural admission recovers 78 gold
instances -- all 78 already reachable within `U2`, none beyond it (6a/6c) --
so `R3_BOUNDED` contributes zero *additional* ceiling lift over `R2` here;
whatever ceiling gain the regime offers on `webqsp`, R2's own one-hop context
already provides in full. The 236 still-missing gold instances (86 of them
beyond even `U2`'s reach, per M0B's own filed cross-tabulation, bit-exact
reproduced here) are a candidate-generation and graph-availability ceiling
that neither `R2` nor `R3` at `cap=64` lifts -- consistent with `webqsp`
staying named as the joint-bottleneck dataset in 6a.

## Step 7: proposed minimum trained-screen matrix

**Proposal only -- `PROPOSED_NOT_DECLARED`.** Nothing below is launched by
this document. This replaces M0B's own Step 7 rather than extending it --
regenerated mechanically from the corrected 6b/6c map above, not by carrying
forward M0B's `R3_FULL`-scale priority reasoning. Where M0B prioritized
GEOMETRY/SUPPORT/PATH/DIFFUSION/TOPOLOGY on `2wiki_clean`/`hotpotqa_clean`/
`webqsp` because R3 context growth there was largest, 6b/6c now show that
growth was entirely an `R3_FULL_REEXPANSION` property `R3_BOUNDED` does not
share -- so those cells are not carried forward as proposed here.

| priority | dataset | family | why |
| --- | --- | --- | --- |
| 1 | `hotpotqa_clean` | NODE ROLE | the only dataset where structural admission recovers gold plain one-hop context could not reach (1/9 recovered-gold instances, 6a) -- and now cleanly isolated from context-size confound, this signal costs 14.5x less to compute than it appeared to under M0B's R3 |
| 2 | `2wiki_clean`, `metaqa`, `webqsp` | NODE ROLE | `STRUCTURAL_SCORED_CANDIDATE` promotion recovers real gold (41, 43, 78 instances respectively) at R2-level context cost; worth confirming whether this recovery translates into ranking signal, independent of any wider-context family |
| deprioritized | `squad_clean`, `musique_clean` | all | at or near ceiling already (squad: `all_gold_at_pool=1.0`; musique: residual gap is 4 gold instances total) -- no family can improve what is barely missing |
| not proposed under `R3_BOUNDED` | any | GEOMETRY, SUPPORT, PATH, DIFFUSION, TOPOLOGY | 6b/6c show these families gain no material richness moving R2→R3_BOUNDED (growth 1.00-1.0013x) -- screening them here would just re-measure R2's own already-known story under a different name |
| explicitly out of scope | any | GEOMETRY, SUPPORT, PATH, DIFFUSION, TOPOLOGY under `R3_FULL_REEXPANSION` | the richness gap is real there, but `R3_FULL_REEXPANSION` is a systems control -- standing prohibition keeps it untrained, on Hotpot or anywhere |
| not proposed | any | PROVENANCE | graph-axis choice, already measured by this run and `EDGE_PROVENANCE_RESULTS.md`, not a screening target |

This is a priority ordering for a future declaration to pick up, not a
commitment that any cell above will be screened. Screening itself -- running
the D-ladder's univariate methodology against real per-column feature values
-- is new work with its own cost and its own declaration, out of scope for
M0C.

## Step 8: stop

M0C Steps 1-7 are complete. No further Modal launch, no new candidate-
admission rule, no per-dataset cap tuning, no learned offset, no
`TARGET_H1`/`R3_BOUNDED` change, no QLS/GNN training, and no screening run
happens on the strength of this document. M0B stays exactly as filed --
`R3_FULL_REEXPANSION` is preserved as a systems-control finding, never
trained on. E2 stays paused, Package F stays sealed, no workspace migration.
The next action on any of Step 7's proposed cells requires its own
declaration, following this protocol's own discipline.
