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

## Results

Run 12 October 2026, 14:03-14:30: the laptop, pinned, one thread, the first 200 s1eval questions of each dataset.
Records are in `outputs/c9/<dataset>.json` and `outputs/c9/report.{json,md}`. Every check passes on every question:
1,200 / 1,200, with FastTwin bit-identical to the twin module (largest difference 0.0).

**Conditions.** Other applications were running on the laptop during this run; the CPU load was 74% when checked
afterwards. Absolute times are 1.5-1.9× C4's and C6's filed stage times; for example, musique gnn4's compile took
106.5 ms here against 59.0 ms filed. Both members of each pair ran in the same rotating order on the same questions.
The cross-family ratio gnn4 / zrc4 (1.94-3.46×) matches C4's filed 2.0-3.4×. Read the ratios from this run. Its
absolute times are inflated and are not quoted.

Cold per question, p50 in ms (this run's), and GNN / MLP within each family with 95% CIs:

| dataset | universal MLP | universal GNN | **universal GNN / MLP** | zrc | zsp | **zsp / zrc** | warm forward: universal GNN / MLP | warm forward: zsp / zrc |
| --- | ---: | ---: | --- | ---: | ---: | --- | ---: | ---: |
| 2wiki | 12.0 | 15.2 | **1.27 [1.22, 1.32]** | 7.8 | 8.2 | **1.04 [0.97, 1.13]** | 2.03 | 1.35 |
| hotpotqa | 10.9 | 14.1 | **1.29 [1.24, 1.34]** | 6.5 | 7.8 | **1.21 [1.11, 1.28]** | 1.78 | 1.42 |
| squad | 7.8 | 10.9 | **1.40 [1.32, 1.53]** | 4.4 | 5.0 | **1.16 [1.06, 1.27]** | 2.21 | 1.50 |
| musique | 195.9 | 322.6 | **1.65 [1.60, 1.70]** | 93.3 | 105.8 | **1.13 [1.10, 1.16]** | 3.20 | 1.16 |
| metaqa | 125.3 | 175.6 | **1.40 [1.36, 1.43]** | 68.2 | 71.4 | **1.05 [1.03, 1.07]** | 2.10 | 1.06 |
| webqsp | 152.0 | 215.4 | **1.42 [1.33, 1.49]** | 163.0 | 153.8 | **0.94 [0.91, 0.98]** | 2.49 | 1.00 |

**Universal family.** The MLP is faster on all six datasets: 1.27-1.65× cold, 1.78-3.20× warm. Cold, that is not much
faster.
- The pool, read and compile, which both models run, are 53-66% of the MLP's cold time.
- The MLP's forward (32-45%) is mostly the input block, and the GNN runs the same input-block code.

**Lean family.** zsp costs 0.94-1.21× zrc cold.
- zsp's message passing is one propagation step over scores. Its GNN-only work is the link build (0.19-9.57 ms) and
  0.46-1.24 ms more forward.
- The stages both run (pool, read, compile, lean inputs, chains) are 68-81% of zrc's cold time.
- On webqsp zsp is the faster: its chain rule (rmatch's mass cap) builds in 52.2 ms, zrc's contribution rule in 61.7 ms.

**Beside.**
- The universal GNN over zrc is 1.32-3.46× (C1-C6's comparison).
- The universal MLP over zrc is 0.93-2.10×. On webqsp the universal MLP is faster than zrc, because zrc's chain build
  costs more than the twin's heavier inputs.

**What bounds each family's lead.** Suppose every stage both members of a family run cost nothing, and only the
GNN-only work and the forwards remained. The cold ratio would reach:

| dataset | universal | lean |
| --- | ---: | ---: |
| 2wiki | 1.76× | 1.43× |
| hotpotqa | 1.74× | 1.55× |
| squad | 2.00× | 1.50× |
| musique | 2.85× | 1.51× |
| metaqa | 1.92× | 1.21× |
| webqsp | 2.25× | 1.15× |

- **Universal.** The lead can grow past that only if the input block, which both models run, also gets cheaper. The next
  stage makes the shared stages and the input block faster, for both models alike.
- **Lean.** No fair change makes zrc much faster than zsp. zsp is zrc plus one cheap propagation step, so the ratio stays
  under about 1.6× whatever the shared stages cost. The lean family's finding is that its GNN's message passing costs
  4-21% more cold (webqsp aside, where zsp is the faster).
