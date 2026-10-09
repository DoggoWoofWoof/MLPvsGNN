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
- **Order:** the stages every path shares (pool, embedding read, compile) run once per question and are counted in
  every path's total. The model-specific stages then run in an order that rotates question by question, so no path
  always runs first on a question.
- **The embedding read** (the pool's rows from the node table) is its own stage. On the laptop, 2wiki's and
  hotpotqa's tables do not fit in memory, so those reads come from disk. Totals are given with it and without it.
- *Amended about 17:25, before any number:* numba compiles its kernels on first use. One question outside the
  measured set (the carve's 201st) is run once at process start to trigger that compile. Its time is index time, and
  nothing from it is reused by a measured question beyond the compiled code.
- Typed graphs (metaqa, webqsp) need relation slots and chains in the cold path. They follow the four untyped
  datasets, in the same file.

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

Filed 9 October 2026, about 19:00. The four untyped datasets ran on the laptop, pinned (logical processor 2, EcoQoS off,
ABOVE_NORMAL; each record's `pin`), one thread, the first 200 s1eval questions, cold. Table: outputs/c1/report.md
(report.json and the per-question records beside it). The typed graphs (metaqa, webqsp) follow.

**Checks.** On every dataset, all 200 questions pass:
- pools, seeds and edges equal the look's on all 200;
- the top 5 of every path is the training form's on all 200;
- on 2wiki, hotpotqa and squad the rebuilt rows equal the step-1 cache bit for bit, and zrc's and zsp's scores equal
  the cache's exactly;
- on musique, 20 questions' rows differ from the cache by at most 7e-4. The compile's BLAS thread count differs from the
  host build's (2 threads). The scores differ by at most 0.008, with the top 5 unchanged;
- the six GNN is within 5e-6 of the look's stored scores everywhere.

No question was excluded.

**Batch 1, total per question, p50 ms (95% interval of the ratio):**

| dataset | zrc (MLP) | zsp (GNN) | six GNN | zsp / zrc | six GNN / zrc |
| --- | ---: | ---: | ---: | --- | --- |
| 2wiki | 31.0 | 31.4 | 30.3 | 1.01 [0.99, 1.05] | 0.98 [0.95, 1.01] |
| hotpotqa | 12.7 | 13.4 | 11.9 | 1.05 [1.01, 1.07] | 0.94 [0.91, 0.96] |
| squad | 7.9 | 8.6 | 7.2 | 1.08 [1.06, 1.11] | 0.90 [0.86, 0.93] |
| musique | 269.2 | 284.3 | 349.7 | 1.06 [1.04, 1.07] | 1.30 [1.29, 1.32] |

**Where the time goes (p50 ms, batch 1):**

| dataset | pool + read | compile (shared) | zrc lean inputs | zrc forward | six GNN pack + forward |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2wiki | 1.3 | 22.1 | 2.4 | 4.2 | 6.3 |
| hotpotqa | 1.2 | 4.2 | 2.3 | 4.2 | 6.1 |
| squad | 0.6 | 1.7 | 1.6 | 3.6 | 4.7 |
| musique | 16.5 | 62.5 | 165.1 | 19.5 | 268.8 |

**The model forward alone, batch 16, per question (ms):** zrc 1.1 / 0.96 / 0.59 / 17.1 against the six GNN 9.3 / 8.0 /
4.1 / 335.7 (2wiki, hotpotqa, squad, musique). That is 8×, 8×, 7× and 20×.

**Reading.**
- **Claim 3, as the models are served today, does not hold cold.** End to end at batch 1, the MLP is at the six GNN's
  cost on the three small-pool datasets (0.90–0.98×) and 1.3× faster on musique.
- The MLP's forward is 7–20× cheaper than the GNN's. That is the 6–9× filed earlier, measured warm on the forward
  alone. Cold, two things eat it:
  - the shared compile, which the MLP needs only four blocks of (rank, dense_cos, topo_STRUCT, depth_STRUCT);
  - the MLP's own walk inputs, which cost as much as message passing on large pools (musique: 165 ms against the
    GNN's 265 ms forward).
- zsp costs within 1–8% of zrc on every dataset, as the code predicted.
- Per this file's rule (lead under 6×), the next stage attacks the cost. In order of size:
  1. an exact partial compile for the MLP (only the blocks it reads);
  2. an exact fast form of the walk inputs (WALK, WALKF, SEED and DISTS in numba, as lean_fast did for the
     MP-free blocks);
  3. a fused batch-1 forward without torch's per-call overhead.

  Each is checked bit for bit against these records' scores, and each is declared in its own file.
