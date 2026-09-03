# M0A.1 — R2/R3 overlap and minimal structural expansion audit

Declaration: [`configs/m0a1_overlap.yaml`](../configs/m0a1_overlap.yaml).
Filed before any M0A.1 number existed.

A zero-training follow-up to M0A. It trains nothing, fits nothing, reads no
test split, opens no new stage, and does not authorise M0B.

## 1. The question

M0A measured that bounded structural expansion recovers missing gold
candidates — 41 on 2wiki_clean, 43 on metaqa, 0 lost anywhere. It did not
measure whether those golds were already sitting in R2's context.

That distinction decides what R3 contributes:

- If the recovered golds were already in `TARGET_H1(Cq)`, then R2 already
  exposed the answer as graph context and could not score it. R3's whole
  intervention is making selected structural context nodes **scoreable**.
- If many recovered golds lie outside `TARGET_H1(Cq)`, then R3 reaches
  genuinely new graph territory and deserves its own construction.

M0A's results document labelled this NOT MEASURED and named it as the first
thing to resolve. This stage resolves it, and nothing else.

**The wording is not chosen before the measurement.** Both readings are filed
verbatim in the declaration under
`interpretation_is_not_chosen_before_measuring`. The runner emits counts; the
sentence is selected from those two filed options afterwards.

## 2. M0A is history

The M0A verdict stands exactly as filed:

```text
MOVEMENT_UNDER_A_BREACHED_ABORT_RULE
```

M0A.1 does not rewrite, soften, or re-derive it. Nothing measured here may be
used to restate the M0A outcome.

The findings M0A.1 treats as settled and does not recompute:

1. Native structural expansion materially moves candidate headroom on 2Wiki
   and especially MetaQA.
2. kNN-only expansion recovers zero golds in every tested cell.
3. Native-only and union recover the same golds.
4. `L1_DIRECTIONAL` and `STRUCTURAL_NEIGHBOUR` produce identical observed
   candidate-headroom outcomes despite different admitted node sets.
5. The directional arm is materially more expensive and breached the filed
   latency abort.

### Wording that is allowed, and wording that is not

Allowed:

> Directional selection showed no incremental candidate-headroom benefit over
> bounded structural expansion in M0A.

Not allowed: "directional expansion is useless", or any claim that directional
selection can never affect downstream ranking. The two arms admitted different
non-gold distractors and no ranker has been trained on either pool, so
downstream difficulty is **untested, not equal**.

## 3. The four sets

| set | definition |
|---|---|
| `Cq` | the historical scored candidate set, frozen candidate order, bit-exact |
| `U2` | `TARGET_H1(Cq)` — R2's context universe, under the exact existing contract |
| `A_struct` | nodes admitted by the M0A `STRUCTURAL_NEIGHBOUR` construction, unchanged |
| `Cq_struct` | `stable_union(Cq, A_struct)` |

`U2` is `Cq` union the **in-neighbours** of `Cq`, via
`graph_context.context_nodes("TARGET_H1")`. `A_struct` comes from the
**undirected** one-hop frontier of the frozen family CSR, symmetrised in
memory and never written back.

Those are not the same neighbourhood, and that asymmetry is the entire reason
this stage exists. If they were the same set, `R3_BEYOND_R2` would be empty by
construction and there would be nothing to measure.

`Cq_struct` is the additive pool. M0A measured `golds_lost == 0` in every cell,
so the matched-budget pool recovered the same golds — that is re-measured here
rather than assumed, and both are reported.

## 4. The partition, and a defect in it found before any result was read

For every gold **absent from `Cq`**, the declaration files three classes:

```text
R2_CONTEXT_RECOVERABLE   gold ∈ U2
R3_BEYOND_R2             gold ∉ U2 and gold ∈ Cq_struct
STILL_MISSING            gold ∉ Cq_struct
```

**These three classes are not disjoint.** A gold in `U2` but not in
`Cq_struct` satisfies both `R2_CONTEXT_RECOVERABLE` and `STILL_MISSING` as
literally written. This was found by reading the rule before any M0A.1 number
existed, and is recorded in the declaration as
`the_three_classes_are_not_disjoint`.

No class definition was altered to fix it. Instead the primitive that is
measured and reported is the 2×2 cross-tabulation:

| | `∈ Cq_struct` | `∉ Cq_struct` |
|---|---|---|
| **`∈ U2`** | `IN_U2_AND_RECOVERED` | `IN_U2_NOT_RECOVERED` |
| **`∉ U2`** | `BEYOND_U2_RECOVERED` | `BEYOND_U2_NOT_RECOVERED` |

The three filed classes are derived from it unchanged:

```text
R2_CONTEXT_RECOVERABLE = IN_U2_AND_RECOVERED + IN_U2_NOT_RECOVERED
R3_BEYOND_R2           = BEYOND_U2_RECOVERED
STILL_MISSING          = IN_U2_NOT_RECOVERED + BEYOND_U2_NOT_RECOVERED
```

with `IN_U2_NOT_RECOVERED` — the overlapping cell — named and counted
separately, so neither reading is hidden. That cell is itself informative: it
is where R2 held the answer as context and bounded structural expansion still
failed to promote it.

Both gold-instance counts and query counts are reported, for every dataset and
every edge family.

## 5. Invariants, asserted rather than assumed

Each is checked in the runner, which raises rather than reporting a number
computed over a broken premise:

- `oracle(Cq under R1) == oracle(Cq under R2)`, **bit-exactly**. R2 may expose
  additional golds as context, but they are not scoreable, so this must never
  be described as a candidate-headroom improvement.
- `Cq` is bit-exact against the frozen artifact.
- `A_struct ∩ Cq = ∅`.
- `Cq ⊆ Cq_struct` and `Cq ⊆ U2`.
- Permuting the gold labels leaves the expansion byte-identical.

## 6. The saturation curve

Purpose: **saturation analysis, not hyperparameter fitting.**

M0A found that admitting 7× more nodes per query on metaqa recovered nothing
further. This curve locates where the gain actually arrives.

Using only the frozen `STRUCTURAL_NEIGHBOUR` construction and the M0A
deterministic ordering contract, `graph_expansion_cap` is swept over:

```text
+4  +8  +16  +32  +64  full bounded N1 frontier
```

with `per_seed_cap = 16`, `hop_cap = 1` and
`neighbour_scan_cap_per_seed = 4096` held fixed.

The numeric points are **strictly nested**: `STRUCTURAL` scores are constant,
so `_rank` breaks every tie by ascending node id and the cap-*k* admitted set
is the *k* lowest ids of the same frontier. The final point lifts
`per_seed_cap` as well, so it is a set-wise superset of every numeric point but
is **not** a prefix of the same ordering. That discontinuity is stated because
it is real, not smoothed over.

Reported at each point: added nodes per query (median / p95 / max), AnyGold,
AllGold, gold fraction, oracle recall ceiling @1 / @5 / @20, missing golds
recovered, and marginal recovered gold per added node.

**No dataset-specific budget is fitted, recommended, or carried forward.** M0B
needs one universal bounded rule, or one deterministic query-derived rule that
does not read dataset identity.

## 7. Edge provenance: reuse rather than recompute

The overlap audit runs on `structural_only` and `baseline_a_simple` — the two
families that actually recovered golds.

`knn_only` is **reproduced from the stored M0A artifacts and asserted**, not
recomputed. M0A settled it at zero recovered golds in every tested cell, at
every budget including unconstrained. Spending a container to recompute a
settled zero would multiply work for nothing.

## 8. The structural arm's own systems contract

The M0A latency abort fired only on `L1_DIRECTIONAL` cells. That verdict is
about that arm. The structural arm is measured here against its own threshold
and **neither inherits the failure nor is excused by it**.

Measured: expansion p50/p95/p99, context-build p50/p95/p99, the temporary
workspace bound, and peak process RSS.

| threshold | value | is its outcome open? |
|---|---|---|
| latency factor | 4.0 | **no — disclosed below** |
| peak RSS | 14 GiB | yes |
| temporary workspace | 1 MiB / query | it is a bound, not a measurement |

**Disclosure.** The latency factor of 4.0 is inherited unchanged from
`configs/m0a_probe.yaml`. It was not selected now to fit the structural arm —
but M0A execution 2 already measured every structural cell at 0.477–0.621 ms
expansion p95 against 1.884–3.818 ms context p95, so this threshold is already
known to pass. It is filed because the rule requires a filed threshold, not
because it is a test whose outcome is open. Saying so is the point.

The peak-RSS bound **is** open. M0A measured 5.13 / 3.87 / 8.36 GiB; M0A.1
holds one more per-query set and iterates six curve points. M0A's a-priori
3–4 GiB estimate was wrong by roughly 2.2× on metaqa, so the bound here is set
from M0A's measurements plus margin rather than from a fresh estimate.

## 9. Advancement to M0B

Advancement is not automatic, and this stage does not launch it.

The simple structural R3 construction advances to a six-dataset M0B **only if
all four hold**:

1. Leakage and invariants pass, including the bit-exact `R1 == R2` ceiling and
   gold-permutation byte-identity.
2. Structural expansion materially improves candidate headroom — at least 1.0
   point of matched-budget AnyGold or oracle recall@5 on at least one
   structural regime.
3. Its independent systems contract passes on its own numbers.
4. A universal bounded expansion rule can be stated without dataset identity.

No directional method is required for advancement. Thresholds are not moved
after seeing M0A.1.

If M0B is later authorised, the proposed regimes — **a proposal only, not a
launch** — are:

```text
R1:  scored = Cq          context = G[Cq]
R2:  scored = Cq          context = TARGET_H1(Cq)
R3:  scored = Cq_struct   context = TARGET_H1(Cq_struct)
```

## 10. Compute

CPU-only. Three jobs, one execution, 4 CPU / 16 GiB, estimated 320 work
seconds total (90 / 30 / 200 for squad_clean / 2wiki_clean / metaqa).

Priced by the project's own model — `container_rate_usd_per_hour` and
`expected_spend_usd` at **$0.3172/hour** for this shape with the standing 0.4
utilisation assumption: **$0.07**, or **$0.21** at three times the estimate,
against a filed ceiling of $0.30.

Cheaper than M0A because it runs one execution rather than two, two edge
families rather than three, and no `L1_DIRECTIONAL` arm — which cost 12–28 ms
per query against structural expansion's 0.4–0.6 ms.

The estimate is anchored to M0A execution 2's **measured** per-query latencies
on the same container shape, not to a fresh host timing. The peak-memory figure
is a **prediction, not a measurement**, and is labelled as such.

## 11. What M0A.1 does not authorise

No QLS training. No GNN training. No M0B. No feature training. No six-dataset
expansion. No new expansion algorithm. E2 stays paused. F stays sealed. No
workspace migration. The frozen `candidate_headroom` component is not touched.

The stage ends at a decision about whether simple structural expansion
advances, and then it stops.
