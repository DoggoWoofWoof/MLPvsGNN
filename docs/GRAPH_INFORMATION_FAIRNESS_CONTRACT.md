# Graph-information fairness contract

Filed before any QLS-v2 versus GNN comparison is run, and binding on all of
them. Nothing in this document depends on an outcome, because a fairness rule
written after seeing the numbers is not a fairness rule.

## 1. The failure this exists to prevent

The historical one-layer seed-aware GNN was trained and evaluated on the
**vertex-induced candidate graph** `G[Cq]`: an edge survives only when both its
endpoints are candidates. Phase -1 measured what that deletes, and Stage B of
the graph-context pilot measured it again on real validation queries:

| | 2wiki_clean | hotpotqa_clean |
|---|---:|---:|
| median radius-1 global-neighbourhood retention | 0.100 | 0.089 |
| boundary cut | 0.852 | 0.900 |
| candidates isolated inside `G[Cq]` | 39.0% | 34.1% |

Roughly nine in ten of a candidate's stored one-hop messages never reach it, and
about a third of candidates have no neighbour at all. A one-layer GNN on that
substrate is not merely operating on a sparse graph; it is **induction-starved**,
which is a different and stronger statement.

The pilot found a context that repairs it. `TARGET_H1 = Cq u N1_in(Cq)` restores
radius-1 retention to 1.000 with a boundary cut of 0.000 and no isolated
candidate, at 3.1% and 0.6% of the corpus graph.

The failure mode is now obvious, and it is the one this contract forbids:

> Give QLS-v2 the repaired context, leave the GNN on `G[Cq]`, and report that
> QLS-v2 beats the GNN.

That comparison measures the substrate, not the method. It would be indefensible
whether or not anyone noticed, and any reviewer who read Phase -1 would notice.

## 2. The contract

**The graph context is a property of the evaluation condition, not of the
method.** Every system compared within a condition sees the same `Uq`.

1. **Identical context.** Within a condition, QLS-v2 and every GNN baseline
   receive the same context node set `Uq` and the same induced edge set over it,
   built by the same code path from the same frozen CSR.
2. **Identical scoring universe.** Both score exactly `Cq`. Context nodes
   contribute structure and are never ranked, so the candidate ceiling is
   identical by construction and no comparison is a candidate-generation
   comparison in disguise.
3. **Identical candidate pools.** Frozen, untouched, and hash-checked on both
   sides.
4. **Identical splits and seeds.** Same train/validation partition, same seed
   protocol, same number of seeds.
5. **A condition is named by its context.** Results are reported as
   "QLS-v2 vs GNN **under `TARGET_H1`**", never as "QLS-v2 vs GNN". If more than
   one context is run, each is a separate condition with its own table.

## 3. What is allowed to differ, and it is the only thing

Given identical graph information, the two systems differ in what they do with
it, and that difference is the paper's actual claim:

| | how it uses `Uq` |
|---|---|
| GNN baseline | **learned** message passing: parameters on the edges, trained end to end |
| QLS-v2 | **fixed** query-conditioned structural descriptors, computed once, no learned propagation at train or inference |

The question is whether learned propagation is necessary once the graph
information is equalised. That question only means something when the graph
information *is* equalised, which is what §2 is for.

## 4. Obligations on the QLS-v2 side

Carried unchanged from the frozen protocol, and repeated because this document
is where a reader will look for them:

* No GNN anywhere in the method. No GNN teacher, no distillation, no
  hidden-state imitation, no GNN-generated labels, residual targets or feature
  supervision, no learned message passing at train or inference.
* No GNN outcome may be used to select a QLS feature. Feature selection is
  decided on QLS-side evidence or preregistration, never by trying a feature and
  keeping it because the gap to the GNN closed.
* GNNs are evaluation and reference baselines. They are not part of the method
  and they are not part of its development loop.

## 5. Obligations on the GNN side

The mirror image, and the one that is easier to violate by omission:

* **The GNN is not to be crippled.** It gets the same context, the same pools,
  the same splits, the same seed count, and a tuning budget at least equal to
  QLS-v2's. If QLS-v2 is tuned over a grid, the GNN is tuned over a comparable
  one, and the grids are reported.
* **Parameter count is not a handicap to impose.** QLS-v2's smallness is a
  finding to report, not a constraint to enforce on the comparator. A GNN is not
  shrunk to match it.
* **Depth is matched to context radius, and any mismatch is declared.** A GNN
  with `L` layers has an `L`-hop receptive field. Under `TARGET_H1` the context
  is radius 1, so a one-layer GNN sees its full natural receptive field and a
  two-layer GNN is truncated at the context boundary. Where a baseline is
  truncated, that is a limitation **of the condition** and is reported as such --
  it is not counted as evidence against the baseline, and a deeper baseline is
  not entered into a condition that cannot feed it without the truncation being
  stated in the same table.
* **The strongest fair baseline is the comparator.** Not the historical one, and
  not the most convenient one. If a stronger GNN exists within the condition, it
  is the one reported.

## 6. Cost accounting

Context construction is paid by **both** systems, so it cancels in any
like-for-like latency comparison and neither side may claim it as an advantage.
It is nonetheless reported separately rather than folded in, because it is a
real serving cost of the condition and a reader deciding whether to adopt any of
this needs to see it:

```text
context construction     shared        p50/p95/p99 per query, reported once
method cost              not shared    per system: features, or propagation
end-to-end               derived       shared + method
```

Measured under `TARGET_H1` at Stage B, p95 per query: 1.9 ms (2wiki) and 74.0 ms
(hotpotqa) to build the context; 6.0 ms and 157.2 ms for the QLS descriptor
kernel on top.

A Pareto claim must state which of these it is on. "QLS-v2 is cheaper" is not a
claim; "QLS-v2's method cost is cheaper under an identical context" is.

## 7. What this contract does not claim

It does not claim that any cited paper defines this procedure. Enclosing
subgraphs, seed-anchored expansions and radius-bounded contexts all appear in the
graph-retrieval and link-prediction literature, and the arms in the pilot were
shaped by that reading; none of those papers specifies `Cq u N1_in(Cq)` over a
frozen candidate pool with a fixed scoring universe, and this document does not
pretend otherwise. The contract is ours, it is stated in full here, and it is
falsifiable by inspection of the code paths it names.

It also does not settle which context is right. That is the pilot's business.
This document only guarantees that whichever context is chosen, both sides get
it.

## 8. Where it is enforced

| clause | enforced by |
|---|---|
| identical scoring universe | `scored_nodes == Cq` in the runner contract, and `Cq` is a subset of `Uq` asserted per arm in `tests/test_graph_context.py` |
| identical context construction | one code path, `mp_retrieval.graph_context.context_nodes`, called by every consumer |
| frozen candidate pools | contract hash compared before and after every run |
| no GNN in the method | no import of a model module in the QLS-v2 feature path |
| context named in results | condition key carried in the result JSON |

The clauses in §5 are not machine-checkable and are the ones most likely to be
broken by accident. They are checked at review time against this list, and a
comparison that cannot point at this document for each of them is not reported.

## 9. Amendment, 2026-09-02: two clarifications the pilot forced

Filed after the graph-context pilot returned and before any comparison is run,
so neither clause depends on an outcome.

### 9.1 The privilege is not the implementation

Two objects were being named by one phrase, and conflating them would smuggle an
engineering choice into a scientific claim:

| | |
|---|---|
| **graph-information privilege** | which structural information a system is permitted to see. `TARGET_H1` is the leading candidate. |
| **feature backend** | how the permitted statistics are computed. Materialising `G[Cq u N1_in(Cq)]` is one implementation. |

§2 binds the **privilege**: both systems see the same permitted information.
Neither side is required to compute it the same way. QLS-v2 may derive seed
support, distance at most 2, and bridge counts directly from the global CSR
without building the subgraph -- §8 of the pilot results proves those statistics
are determined by the privilege rather than by the subgraph object -- and a GNN
may propagate over the materialised graph. What is forbidden is a *difference in
permitted information*, not a difference in data structure.

Any implicit backend must be exactly equivalent to the explicit one for every
statistic it claims to compute, or must be declared as an approximation with its
own name and its own numbers. A cap on context size is such an approximation:
its rule is fixed from context-size distributions and systems constraints before
its ranking effect is measured, never tuned against outcomes.

### 9.2 The contract is depth-aware, and `TARGET_H1` is not a universal context

§5 requires depth to be matched to context radius. Made explicit, because the
convenient error is to fix one context and call it fair:

* A one-layer GNN's natural receptive field is radius 1, which `TARGET_H1`
  supplies in full.
* A two-layer GNN needs radius-2 computational context -- `TARGET_H2`, or
  literature-standard neighbour sampling -- and is truncated at `TARGET_H1`'s
  boundary.
* Deeper baselines need correspondingly more.

**`TARGET_H1` must not be declared the graph context for all GNN depths.** Doing
so would truncate every baseline deeper than one layer while leaving QLS-v2
whole, which is the exact failure §1 exists to prevent, wearing the opposite
costume. Each depth is its own condition with its own permitted context, the
strongest practical baseline is selected on validation, and any truncation is
reported in the same table as the result it affects.
