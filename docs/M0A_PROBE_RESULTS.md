# M0A probe — results

**Status: M0A COMPLETE. No launch is authorised by this document.**

Declaration: [`configs/m0a_probe.yaml`](../configs/m0a_probe.yaml).
Protocol: [`M0A_PROBE_PROTOCOL.md`](M0A_PROBE_PROTOCOL.md).
Rule applied by [`scripts/analyze_m0a_probe.py`](../scripts/analyze_m0a_probe.py),
which reads its thresholds from the declaration and cannot choose them.

Every claim below is labelled. `MEASURED` is a number read off the probe
output. `STRUCTURAL IDENTITY` holds by construction and was asserted in the
runner rather than observed. `INFERENCE` is a reading of measured numbers.
`NOT MEASURED` is exactly that, and is not softened anywhere.

---

## 1. The verdict

**`MOVEMENT_UNDER_A_BREACHED_ABORT_RULE`** — MEASURED, by the filed rule.

All gates passed on all three datasets: every invariant held, the matched
budget held for every query, latency percentiles were recorded, the scan cap
was reported, nothing was trained, the test split was not read.

The movement branch fires. The decisive-null branch does not. But the filed
systems abort rule also fires on two of the three datasets, and the runner did
not implement the abort, so those datasets were reported when the declaration
says they should have been stopped before reporting. That breach is carried in
the verdict string rather than noted underneath it.

**The headline number is not the interesting result.** The interesting results
are §3 and §4: the method this probe was built to test is indistinguishable
from a trivial control, and the recovery it produces is not budget-limited.

---

## 2. What was run

100 validation queries per dataset, deterministic prefix of the split order,
three edge-provenance families, two expansion methods, matched and additive
budgets, plus an unconstrained diagnostic. Two executions; §10 covers why.

| dataset | nodes | stored directed edges | stored symmetric | queries without frozen seeds |
|---|---|---|---|---|
| squad_clean | 19,029 | 2,857,316 | no | 0 |
| 2wiki_clean | 65,865 | 855,146 | no | 0 |
| metaqa | 40,151 | 585,728 | no | 0 |

MEASURED. The frozen candidate contract verified bit-exact on all three
(`BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE`); 2wiki_clean under
`pre_hop_metadata_v1`, the other two under `current_with_hop_metadata`.

R1 candidate ceilings, the baseline every R3 cell is measured against:

| dataset | AnyGold | recall ceiling@5 | AllGold | queries with no gold in pool |
|---|---|---|---|---|
| squad_clean | 1.0000 | 1.0000 | 1.0000 | 0 |
| 2wiki_clean | 1.0000 | 0.8175 | 0.6100 | 0 |
| metaqa | 0.7000 | 0.6517 | 0.6000 | 30 |

MEASURED.

`r1_ceiling_equals_r2_ceiling_exactly` held on all three — STRUCTURAL IDENTITY,
asserted per cell in the runner, which raises rather than reports if R2 moves
the candidate ceiling. R2 widens context only; it cannot change what R1 could
have retrieved.

---

## 3. The primary method is indistinguishable from its control

L1_DIRECTIONAL — the parameter-free reconstruction this probe exists to test —
and STRUCTURAL_NEIGHBOUR — the simplest possible control, lowest global node id
first — produce **identical headroom in every one of the nine family cells on
all three datasets**, to every recorded digit. MEASURED.

The obvious explanation would be that the caps never bound, so both methods
admitted the whole neighbourhood and no comparison happened. That was checked,
and for most cells it is false:

| dataset | family | mean Jaccard of admitted sets | min | queries with a capped seed | comparison happened |
|---|---|---|---|---|---|
| squad_clean | structural_only | 0.292 | 0.013 | 99 / 100 | yes |
| squad_clean | baseline_a_simple | 0.301 | 0.013 | 99 / 100 | yes |
| squad_clean | knn_only | 0.998 | 0.800 | 0 / 100 | **no** |
| 2wiki_clean | structural_only | 0.973 | 0.429 | 11 / 100 | yes |
| 2wiki_clean | baseline_a_simple | 0.974 | 0.592 | 18 / 100 | yes |
| 2wiki_clean | knn_only | 0.999 | 0.892 | 1 / 100 | yes |
| metaqa | structural_only | 0.884 | 0.188 | 23 / 100 | yes |
| metaqa | baseline_a_simple | 0.891 | 0.222 | 29 / 100 | yes |
| metaqa | knn_only | 1.000 | 1.000 | 0 / 100 | **no** |

MEASURED.

On squad_clean the two arms admit sets that overlap by 29% on average, and on
one query by 1.3%. They are choosing almost entirely different nodes. The
headroom is the same anyway.

**INFERENCE:** where the comparison happened, the directional signal buys
nothing over taking the lowest-id neighbours. Whatever recovery one-hop
expansion produces is a property of *which neighbourhood is entered*, not of
*which neighbours are chosen inside it*. The query/entity residual and the
edge-displacement directions — the whole content of the reconstructed L1
method — do not select better nodes than an arbitrary deterministic order.

This is a null about the method, and it is a real one, not an artifact of a
budget too loose to force a choice. It does not become a null about one-hop
expansion itself: expansion moves the ceiling (§1, §5), the direction just does
not decide how.

Two cells are excluded from that reading. On squad_clean and metaqa,
`knn_only` never hit the per-seed cap on any query, so both arms took the whole
neighbourhood and the comparison genuinely did not happen there. Those two
cells are `the_comparison_happened: false` in the analysis output and are not
evidence either way.

---

## 4. The recovery is structural, and kNN edges contribute nothing

Matched-budget movement against R1, in points:

| dataset | family | AnyGold | recall ceiling@5 | golds recovered | golds lost |
|---|---|---|---|---|---|
| squad_clean | structural_only | +0.00 | +0.00 | 0 | 0 |
| squad_clean | knn_only | +0.00 | +0.00 | 0 | 0 |
| squad_clean | baseline_a_simple | +0.00 | +0.00 | 0 | 0 |
| 2wiki_clean | structural_only | +0.00 | **+15.25** | 41 | 0 |
| 2wiki_clean | knn_only | +0.00 | +0.00 | 0 | 0 |
| 2wiki_clean | baseline_a_simple | +0.00 | **+15.25** | 41 | 0 |
| metaqa | structural_only | **+30.00** | **+34.83** | 43 | 0 |
| metaqa | knn_only | +0.00 | +0.00 | 0 | 0 |
| metaqa | baseline_a_simple | **+30.00** | **+34.83** | 43 | 0 |

MEASURED, identical under both expansion methods (§3).

`knn_only` recovers **exactly zero** golds on every dataset, under every
method, at every budget including unconstrained. `structural_only` and
`baseline_a_simple` recover identically.

**INFERENCE:** the recovery comes from real graph topology. Embedding
similarity recycled through kNN edges adds nothing to the candidate oracle —
which is consistent, since the candidates were already selected by dense
similarity, so kNN neighbours are largely nodes the retriever had already
considered. This is the cleanest result in the probe and the one least
dependent on the choice of expansion method.

On metaqa the recovery is total: 43 of 43 missing golds, 40 of the 40 queries
that were missing any gold, and all 30 queries that had no gold in the pool at
all. MEASURED. A ceiling of exactly 1.0000 warrants suspicion, so the leakage
defence is stated explicitly: expansion output is byte-identical under
permutation of the gold labels
(`test_permuting_gold_leaves_the_expansion_byte_identical`, passing), so the
expansion cannot be reading golds. The plain reading is that MetaQA is a
knowledge-base QA set whose answer entities are direct neighbours of the topic
entity, and the topic entity is reliably among the frozen seeds.

---

## 5. The gain saturates far below the declared budget

The unconstrained diagnostic lifts every cap — per-seed 16 to unbounded, graph
expansion 128 to unbounded, scan cap to unbounded — and is reported separately,
never as the matched-budget headline.

| dataset | family | matched admitted/query (mean) | unconstrained admitted/query (mean) | matched R@5 points | unconstrained R@5 points |
|---|---|---|---|---|---|
| 2wiki_clean | structural_only | 16.38 | 18.17 | +15.25 | +15.25 |
| 2wiki_clean | baseline_a_simple | 33.78 | 36.04 | +15.25 | +15.25 |
| metaqa | structural_only | 23.05 | 167.03 | +34.83 | +34.83 |
| metaqa | baseline_a_simple | 30.32 | 174.94 | +34.83 | +34.83 |
| squad_clean | structural_only | 51.46 | 394.96 | +0.00 | +0.00 |

Matched counts are the STRUCTURAL_NEIGHBOUR arm; L1_DIRECTIONAL differs only
in the second decimal (16.33, 33.73, 22.95, 30.23, 55.42 respectively).

MEASURED.

On metaqa, admitting **seven times** as many nodes per query recovers **zero**
additional gold. The gain is fully realised at a per-seed cap of 16.

**INFERENCE:** the matched-budget headline is not an artifact of a generous
budget, and R3 is not winning because its candidate set is unbounded — the
concern the budget rules were written to prevent. Loosening the budget does
not help it. Whatever one-hop expansion recovers, it recovers immediately.

---

## 6. The matched budget never actually bound

`golds_lost: 0` in every cell on every dataset. MEASURED.

The matched rule requires `|Cq'| == |Cq|` exactly per query, so admitted nodes
displace the tail of the frozen candidate order. The rule was enforced — the
runner raises on any violation, and `matched_budget_pool_size_equals_cq` held
throughout — but the evicted tail never contained a gold in any cell of any
dataset. The ceiling could have fallen and did not.

**INFERENCE:** on these three datasets the matched constraint costs nothing, so
the matched and additive readings coincide. That is a fact about where golds
sit in the frozen candidate order, not a general guarantee; a dataset with
golds deep in the tail would separate them. The constraint should stay.

---

## 7. squad_clean cannot test recovery

R1 on squad_clean is already 1.0000 on AnyGold, recall ceiling@5 and @20,
AllGold, full-coverage@20 and gold fraction. MEASURED. There is no headroom to
recover, so every R3 cell reads +0.00 and no expansion method can be
distinguished from another by outcome.

squad_clean therefore contributes to this probe only as a contract test — that
expansion preserves the invariants, holds the matched budget, and does not
break the frozen candidate equivalence — and as the sharpest available test of
§3, since it is where the two arms diverge most (mean Jaccard 0.292) while
still producing identical results. It is not evidence about recovery, and it is
not counted as such.

One detail specific to squad_clean: the L1 method dropped 82 zero-norm
displacement edges on `structural_only` and 83 on `baseline_a_simple`
(MEASURED, 0 on every other dataset and 1 on `knn_only`). Zero-norm
displacements are dropped and counted by design; they are duplicate-position
node pairs and cannot yield a direction. `degenerate_residual_queries` was 0
everywhere — no query had a degenerate query/anchor residual.

---

## 8. What this probe does NOT show

**NOT MEASURED — and this is the first thing M0B must resolve.** The expansion
frontier is the *undirected* one-hop neighbourhood of the seeds: all three
graphs are stored asymmetric, and the runner symmetrises the CSR in memory
(never writing it back) so that a displacement direction is not decided by an
arbitrary storage orientation. R2's context, `TARGET_H1(Cq) = Cq ∪ N1_in(Cq)`,
is the *in-neighbour* one-hop neighbourhood.

These two sets are not the same. It is therefore **not** structurally
guaranteed that the golds R3 recovers were already inside R2's context, and
this probe did not measure the overlap. So the probe cannot say whether R3's
gain is:

- new information entering the system through out-edges R2 never saw, or
- a re-partition of what R2 already held in context, moving nodes across the
  scored/context boundary.

Those two readings imply very different things about what a candidate-side
change is worth, and nothing here distinguishes them. Any statement that R3
"finds" golds R2 could not reach is unsupported by this probe.

Also not shown: anything about ranking. Every number here is a candidate
*ceiling* — what a perfect scorer could reach. No scorer was run, nothing was
trained (`trained_anything: false` on all three), and the test split was not
read (`test_split_read: false` on all three). MEASURED.

---

## 9. Systems, and the abort rule that fired

The declaration files this condition: a dataset is aborted before reporting if
its expansion p95 per query exceeds its context-build p95 per query by more
than a factor of four on the same container. It fired.

| dataset | cell | expansion p95 | context p95 | ratio |
|---|---|---|---|---|
| squad_clean | structural_only / L1_DIRECTIONAL | 27.93 ms | 2.97 ms | **9.41** |
| squad_clean | baseline_a_simple / L1_DIRECTIONAL | 26.67 ms | 3.35 ms | **7.96** |
| metaqa | structural_only / L1_DIRECTIONAL | 17.76 ms | 1.99 ms | **8.92** |
| metaqa | baseline_a_simple / L1_DIRECTIONAL | 12.94 ms | 1.87 ms | **6.90** |

MEASURED, against a filed factor of 4.0. 2wiki_clean did not breach. Every
STRUCTURAL_NEIGHBOUR cell is far inside the rule; **every breach is in the
L1_DIRECTIONAL arm**, which is the arm that computes per-edge displacement
directions and cosines.

Two things about this must not be glossed:

1. **The runner did not implement the abort.** The rule was applied after the
   fact, in the analysis, on data that had already been read. That is weaker
   than aborting before reporting, and it is the reason the verdict carries the
   `_UNDER_A_BREACHED_ABORT_RULE` suffix rather than a footnote. The
   declaration also says thresholds are not moved after seeing M0A, so the
   factor of 4.0 was applied as filed and not softened to fit.
2. **INFERENCE:** the breach and §3 point the same way. The directional method
   costs 7–9× the context build in the arm where it does its distinctive work,
   and buys nothing measurable for it.

Resource use, MEASURED on a 4 CPU / 16 GiB container:

| dataset | peak process RSS | vs. the 3–4 GiB estimate |
|---|---|---|
| squad_clean | 5.13 GiB | over |
| 2wiki_clean | 3.87 GiB | at |
| metaqa | 8.36 GiB | over, by ~2.2× |

The compute estimate was wrong on memory. metaqa stayed inside the 16 GiB
container, but the margin was ~1.9×, not the ~4× the estimate implied. An
estimate this far off should not be reused for M0B without revision.

`temporary_workspace_bytes: 4096` is **a declared bound, not a measurement** —
it is `8 * graph_expansion_cap * 4` written down in advance. No workspace
figure in this probe was observed, and none should be quoted as if it were.

---

## 10. Process deviations

All three are recorded here because they happened, not because they change the
numbers. None is hidden.

**Diagnostics were added after execution 1's outcome was seen.** Execution 1
showed L1_DIRECTIONAL and STRUCTURAL_NEIGHBOUR producing identical headroom,
which could not be distinguished from a budget so loose that no comparison
happened. Execution 2 added the cap-firing counters, the admitted-set
recording, and the unconstrained frontier diagnostic that §3 and §5 rest on.
This is a post-outcome change to what is measured, marked
`PROCESS_DEVIATION_RECORDED_NOT_HIDDEN` in the declaration.

Both executions are preserved, and execution 2's headroom is **byte-identical
to execution 1 on all 14 shared cells of all three datasets** — verified, not
asserted. The only difference is the three added
`UNCONSTRAINED_FRONTIER/diagnostic` cells per dataset. The added diagnostics
changed nothing that was already measured.

**The abort rule was applied post hoc**, as §9 states.

**The filed advancement rule's two branches are not disjoint.** The movement
branch fires on AnyGold *or* recall@5 clearing its threshold; the decisive-null
branch reads recall@5 alone. A run can satisfy both. This was found by
inspecting the rule **before any M0A result was read**, is recorded in the
declaration as `the_two_branches_are_not_disjoint`, and the analysis reports
`CONTRADICTORY` rather than silently picking a branch if it ever happens. It
did not happen here — the null branch did not fire — so it did not affect this
verdict.

---

## 11. What M0A does not authorise

M0A was a candidate-oracle probe on 100 validation queries per dataset. It
authorises nothing on its own; advancement is manual
(`advancement_is_automatic: false`).

It does **not** authorise: training QLS, training a GNN, launching M0B,
resuming E2, opening F, migrating workspaces, expanding to six datasets, or
reading any test split.

**The reading that should carry forward, stated plainly.** One-hop structural
expansion moves the candidate ceiling substantially on two of three datasets,
entirely through real graph topology and not at all through kNN edges, and it
does so at a per-seed cap of 16 with no benefit from a larger budget. The
parameter-free directional method built to steer that expansion does not
outperform picking neighbours by ascending node id, in cells where the two
demonstrably chose different nodes, and it costs 7–9× the context build to do
it. And it is not established that any of the recovered golds were outside
what R2 already held in context (§8).

Anything M0B proposes should start from §8, because until that overlap is
measured, the size of the effect in §4 is known but its meaning is not.
