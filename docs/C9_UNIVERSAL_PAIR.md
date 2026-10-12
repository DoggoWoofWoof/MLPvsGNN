# C9: MLP against GNN within each model family, both at their fastest, cold

Declared 12 October 2026, about 13:45, before any timed number. It follows C1-C6 (docs/C1_COLD_COST.md to
docs/C6_FAST_CHAINS.md). C1-C6 compared the lean MLP (zrc) with the six GNN, which belongs to the other family.

## Why

The user, 12 October:
- There are two model families, **universal** and **universal-lean**. The MLP and the GNN are compared **within** each
  family, as a pair.
- Every model at its fastest. The MLP should be much faster than its GNN. **Cold latency first.**
- Everything else stays the same: the links, the pools and the edges.

## The two pairs

| family | MLP | GNN | inputs both read |
| --- | --- | --- | --- |
| universal | u_mlp_v2_mix, the six pair's twin, seed 0 (`twin0`) | u_gnn_v2_ef, seed 0 (`gnn0`) | the v2 compile's 197 columns, the 1,536-wide node embeddings, the pool edges by family, the seeds |
| universal-lean | zrc (J5, p@swa) | zsp (J5, p@swa) | the pick; on the typed graphs their chain entries; zsp adds the pool links |

- **What the universal MLP reads of the graph.** Its input block averages each family's in-neighbours' projected
  embeddings and the seeds' reach, once, before any row is scored. It reads no edge attribute and no relation, and
  passes nothing between rows after that.
- **What the universal GNN adds**, over the same edges: three relation-aware message-passing steps and the evidence
  flow.
- **The lean pair** is C4's (untyped graphs) and C6's (typed graphs) unchanged. zsp is zrc's inputs plus one
  propagation step over zrm's scores. C3 filed it at 1.08-1.14× zrc's cold cost.
  - On the typed graphs, each model builds its own chain entries: zrc's contribution rule, zsp's rmatch mass cap.
    Each build is a timed stage of its own path.
- **This lean pair is F1's C8** (docs/F1_ONE_INPUT_SET.md, amendment 3). F1's step 1 found every family used by both
  models, so the controlled set and both ceilings equal the current inputs. zgn is in neither pair and is not timed.

## What is new: FastTwin

`outputs/mp_unified/c9_universal.py` (new; frozen once run). FastTwin is the twin's serving form at batch 1:
- **The input block is FusedGNN's code** (C3): the numba per-family segment mean, in edge order per node. The twin's
  column gates are applied to it: x·g and z·g, with g the gate row of the question's projected query. The six GNN keeps
  that same input-block code, so neither model's input block is faster than the other's.
- **Each family's edges (u, v) are read straight from the compile.** The twin reads no edge attribute and no relation
  slot, so it has no edge pack.
- **The body and the readout are the module's.**
- C3's rules hold: no node store, no pruning, no quantisation, no caching across questions, and no change to any
  input the model reads.

The twin is loaded as mp_approx_six_base_score.pair_models loads it: its fit record's arm, its weights' state_sha256
and its parameter count (330,955) are checked.

## What is timed

C9's run is C4's run (2wiki, hotpotqa, musique, squad) or C6's run (metaqa, webqsp) with one more path, **umlp9**:

| graphs | paths, in an order rotating question by question |
| --- | --- |
| untyped | umlp9; zrc4, zsp4, gnn4; C4's references zrc3, zsp3, gnn3 |
| typed | umlp9; zrc6, zsp6, gnn5; C6's references zrc5, zsp5 |

- **umlp9** = the shared stages (pool, read and compile: c1_cold.shared, the same full v2 compile as the six GNN's, at
  C4's galloping form), the twin's inputs (pack), FastTwin and the top 5.
- Every other path is its filed code, unchanged.
- **Protocol (C6's):** the laptop, pinned, one thread, the session idle, the first 200 s1eval questions of each
  dataset, gc off, cold, batch 1. A warm repeat of each forward is timed beside.

## Checks (untimed, every question)

- C4's checks (untyped) and C6's checks (typed) on their paths, unchanged.
- umlp9's pool equals the look's.
- **FastTwin is within TOL = 1e-4 of the twin module's own forward** on the six GNN's pack of the same question, with
  the same top 5.
- FastTwin's top 5 equal those of the look's stored twin0 scores. The largest difference is reported.
- A question failing any check is kept out of the summaries. More than 2% failing stops a dataset.

**Development check** (30 questions each, not timed): see the end of this file, filled before any timed run.

## What is reported

`outputs/c9/<dataset>.json` and `outputs/c9/report.{json,md}`, all cold, batch 1:
- p50, p95 and p99 per path;
- **within each family, GNN / MLP**: the universal GNN over umlp9 (gnn4 or gnn5), and zsp over zrc (zsp4 / zrc4 or
  zsp6 / zrc6), each with a 2,000-resample bootstrap 95% CI;
- beside: the universal GNN over zrc (C1-C6's ratio), the universal GNN over zsp, and umlp9 over zrc (what the
  universal MLP's heavier inputs cost);
- the warm forward ratios within each family, labelled warm;
- every stage's p50.

There are no thresholds. Whatever the ratios are, they are filed.

## What comes after (each its own declaration)

- The largest shared stages (pool, read, compile), in exact forms, for both families.
- The universal input block, for the twin and the six GNN alike.

## What this does not do

- No training, no new fit, and no change to any pool, link, edge, column, chain entry or the encoder.
- No test split is read. Nothing is chosen per dataset.

## Development check

Run 12 October 2026, 13:46-13:58, before any timed number. `check --queries 30` on each dataset: the first 30 s1eval
questions, every path of the run, untimed, pinned.

| dataset | questions failing any check | FastTwin vs the twin module: largest difference | FastTwin vs the look's twin0: largest difference | same top 5 as the module / as the look |
| --- | ---: | ---: | ---: | ---: |
| 2wiki | 0 / 30 | 0.0 | 5.7e-6 | 30 / 30 |
| hotpotqa | 0 / 30 | 0.0 | 5.7e-6 | 30 / 30 |
| musique | 0 / 30 | 0.0 | 5.7e-6 | 30 / 30 |
| squad | 0 / 30 | 0.0 | 4.8e-6 | 30 / 30 |
| metaqa | 0 / 30 | 0.0 | 7.6e-6 | 30 / 30 |
| webqsp | 0 / 30 | 0.0 | 1.1e-5 | 30 / 30 |

- **FastTwin is bit-identical** to the twin module's forward on every question: the same per-family means, in the same
  order.
- **Against the look's stored twin0** (scored on the host, batched), the largest difference is 1.1e-5, under TOL.
- **C4's and C6's checks** pass on every question of every path.
