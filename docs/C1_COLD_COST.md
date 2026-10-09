# C1: the cold cost table, MLP (zrc) against the GNNs (zsp and the six GNN)

Declared 9 October 2026, about 17:20, before any number. Stage C1 of docs/PROGRAM_2026_10_09.md. It is a
measurement only: no model is trained, chosen or changed. Claim 3 (docs/PAPER_CLAIMS_2026_10_09.md) is read from it.

## Why, and what the code already says

Claim 3 says the MLP is faster cold. The filed cold ratio, about 1.5×, is from older models; the 6–9× was warm.

Reading the code before timing anything shows where the cost sits:

- **zrc (the MLP base)** reads four of the compile's column blocks (rank, dense_cos, topo_STRUCT, depth_STRUCT). It
  also reads:
  - lean_mlp's walk lines (WALK, WALKF), which walk the question's **compiled pool edges**;
  - lean_mlp3's store lines (SEED, DISTS) and int8 store codes;
  - on the typed KBs, rmatch's chain entries with zrc's keep rule.
- **zsp (the GNN track's base)** reads zrm's inputs. Its edges are `zlink.chunk_edges` of the same compiled pool edges,
  deduplicated. It then adds one score propagation: a segment mean, a soft maximum and a degree per family, through a
  385-parameter head.
- So **zsp's extra work over the MLP is one deduplication and one aggregation over edges the MLP's features already
  needed.** Against zsp the cold ratio should be near 1× by construction.
- A cold lead exists only against a GNN with costs of its own. **The six GNN** (u_gnn_v2_ef: the full 197-column v2
  compile, then two message-passing layers) is timed beside, in its exact fast form (gnn_fast.FastGNN, as lean_time9
  serves it).

The table states all three. Rewording Claim 3 is the user's decision, not this stage's.

## What is timed

On the laptop, one process per dataset and path group:
- pinned to one logical processor, with EcoQoS opted out and ABOVE_NORMAL priority (own process only);
- one BLAS, numba and torch thread;
- the session otherwise idle.

Datasets: all six. Questions: the first 200 of each dataset's s1eval carve, in carve order. A carve with fewer is
taken whole.

**Cold** means each question is computed from scratch, starting from its first-stage dense and SPLADE lists:
- no feature, pool, walk or chain cache;
- no warm-up pass, and every question is reported.

What may exist before the first question is **index time**, timed and reported on its own rows:
- loading the graph's CSR and the node stores;
- the int8 store codes of the nodes;
- node projections;
- relation-name tables;
- numba's compile of the kernels at process start.

Per question, at batch 1, every stage is timed with `time.perf_counter`:

| stage | MLP zrc | GNN zsp | six GNN | notes |
| --- | --- | --- | --- | --- |
| pool | ✓ | ✓ | ✓ | pool and seeds from the lists (m3b's builder, as lean_time9) |
| compile | ✓ | ✓ | ✓ | fast_features.FastCompiler, v2. The reference compile is timed once beside it, labelled |
| lean inputs | ✓ | ✓ | – | projection, WALK, WALKF, SEED, DISTS and store codes, by lean_mlp's, lean_mlp2's and lean_mlp3's per-query functions |
| chains | ✓ (zrc rule) | ✓ (zrm rule) | – | typed KBs only: rmatch.question_entries; zero on untyped graphs |
| edges | – | ✓ | – | zlink.chunk_edges on the question's compiled edges |
| batch + forward | ✓ | ✓ | ✓ | the model's own forward on the CPU, then top 5. For the six GNN: pack, then FastGNN |
| **total** | ✓ | ✓ | ✓ | |

- **Batch 16:** the same 200 questions in 13 groups of 16 (the last group smaller). Each group's inputs are built
  question by question, then one forward. Its per-question cost is the group's time over its size.
- **Order:** the paths rotate question by question, so no path always runs first on a question.

## Checks (untimed, each question)

- The rebuilt pool and seeds equal the look's.
- The harness's inputs equal the carve's cached arrays. Float16 is compared as bits where the cache stores it; a
  stated tolerance applies only where the BLAS thread count differs from the cache's build (as recorded).
- On the read's first 64 questions, the harness's scores equal the filed read's stored scores within 1e-4.
- Everywhere, the top 5 matches the training form's.

A question failing a check is reported, and its timings are kept out of the summaries. More than 2% failing on a dataset
stops that dataset.

## Reported

- Per dataset, path and stage: p50, p95 and p99 at batch 1 and batch 16, with 95% bootstrap intervals (2,000
  resamples, seed 0).
- Peak RSS.
- The index-time rows.
- **The ratios**: the six GNN's and zsp's total over zrc's, as p50 ratios with bootstrap intervals. The headline is cold;
  nothing warm is reported.
- Where each model's time goes: the stage shares.

If the MLP's cold lead over the six GNN is under 6×, the next stage, in its own file, attacks the largest shared stage,
the compile.

## Files

    outputs/mp_unified/c1_cold.py                   the harness (new; frozen once run)
    outputs/c1/<dataset>.json                       per-question timings, checks, index rows
    outputs/c1/report.json, report.md               the table

    python outputs/mp_unified/c1_cold.py --selftest
    python outputs/mp_unified/c1_cold.py run --dataset 2wiki --queries 200 --out outputs/c1/2wiki.json
    python outputs/mp_unified/c1_cold.py report

## Results

(Filed after the run.)
