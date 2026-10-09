# C2: the MLP's exact cold serving path

Declared 10 October 2026, about 03:00, before any timed number. Stage C2 of docs/PROGRAM_2026_10_09.md. It follows
C1 (docs/C1_COLD_COST.md), whose rule sent it here: the MLP's cold lead over the six GNN was under 6×.

No model is trained, chosen or changed. The models are C1's (outputs/b1b/fits/{zrc,zsp}/J5, candidate p@swa). Only
the code that builds zrc's inputs changes, and it must produce **the same bits**.

## What C1 found

End to end at batch 1, the MLP (zrc) cost as much as the six GNN on 2wiki, hotpotqa and squad, and was 1.3× faster on
musique. Two things ate its forward's 7–20× lead:

- **The shared compile.** zrc was served through the six GNN's whole 197-column v2 compile, but it reads only four
  blocks of it: rank, dense_cos, topo_STRUCT and depth_STRUCT (plus rrf, which SEED and DISTS read).
- **Its own lean inputs**, built through lean_mlp.lean_query. That function also computes the SEM, SEED and NBR
  blocks and the node projection they need. zrc reads none of them: its SEMB comes from the int8 store codes.
  lean_cache.walkf_of then runs lean_query a second time for WALKF.

## What C2 changes

`outputs/mp_unified/c2_fast.py` (new; frozen once run) has three parts.

**`compile_mlp`.** This is fast_features.FastCompiler's compile for zrc's columns only. It uses the same kernels,
buffers, operands and order as `FastCompiler._compile` for what it computes:
- retrieval;
- the three families' pool edges;
- topology on the STRUCT view;
- the seed products E @ E_Sᵀ;
- the depth basis on the STRUCT view.

Every other column is left at 0. None feeds a column zrc reads.

**`walk_block` / `walkf_block`.** These are lean_query's WALK lines and walkf_of's lines, with nothing else. SEED and
DISTS still come from lean_cache.store_seed_dists, unchanged.

**The forward.** zrc's and zsp's forwards are unchanged (C1's `Rows` and `forward`). zsp's links are C1's `links_of`
on C2's edges.

The six GNN is served exactly as in C1: the full fast compile, `fast_pack`, then `FastGNN`. It needs all 197 columns,
so this stage gives it nothing to drop. That compile is already its exact fast form.

**Development check, before this file.** `c2_fast.py check` ran on 30 questions per dataset. On every question the
pool, zrc's compile columns, the edges, the float16 rows, the store codes and zrc's scores equal C1's path bit for bit.
The check's per-stage times are development numbers, not filed numbers.

## What is timed

Everything not stated here is C1's protocol:
- laptop, pinned to logical processor 2, EcoQoS off, ABOVE_NORMAL priority (own process only);
- one BLAS, numba and torch thread; session idle;
- the first 200 s1eval questions, cold;
- numba's compile on the carve's 201st question, reported as index time;
- batch 1, plus batch 16 in 13 groups.

There are four paths. **Each runs its whole chain on its own**: pool, embedding read, its compile, its own stages.
Nothing is shared between paths, and their order rotates question by question.

| path | what it is |
| --- | --- |
| **zrc2** | the MLP (zrc) in C2's form: `compile_mlp`, then C2's lean inputs, then zrc's forward |
| **zsp2** | the GNN track's base (zsp) on C2's inputs, plus its links |
| **zrc1** | zrc in C1's form, timed in the same run, so C2's speed-up is read without comparing across runs |
| **gnn6** | the six GNN (u_gnn_v2_ef), as in C1 |

Datasets: the four untyped ones (2wiki, hotpotqa, squad, musique), as C1 filed. Typed graphs follow, in their own file,
after C1's.

## Checks (untimed, every question)

- **pools:** every path's pool equals the look's;
- **seeds:** each MLP path's seeds and buckets equal the look's;
- **zrc2 = zrc1:** C2's edges, float16 rows and store codes equal C1's form bit for bit, and so do zrc's scores;
- **zsp2 = zsp1:** zsp's scores on C2's rows equal zsp's scores on C1's rows, exactly;
- **six GNN:** within C1's tolerance of the look's stored scores, with the same top 5;
- **step-1 cache:** zrc's and zsp's scores on the cache's rows, with the same top 5.

A question failing any check is reported, and its timings are kept out of the summaries. More than 2% failing on a
dataset stops that dataset.

## Reported, and how it is read

Per dataset and path, the totals' p50, p95 and p99 at batch 1, with and without the embedding read, and 95% bootstrap
intervals (2,000 resamples, seed 0). Also the stage p50s, the batch-16 forward per question, the index rows and peak
RSS.

The ratios, as p50 ratios with bootstrap intervals: six GNN / zrc2, zsp2 / zrc2, six GNN / zsp2, and zrc1 / zrc2.

- **Claim 3 ("the MLP is faster cold")** holds on a dataset when the six GNN / zrc2 interval lies above 1. Its size is
  stated as a multiple.
- **Against zsp** (the GNN track's base), the gap is what message passing adds on top of the same inputs. It is
  reported, not graded.
- **If the lead over the six GNN is under 6× on a dataset,** the remaining stages from C1's list come next, each in its
  own file:
  - SEED and DISTS in numba, exact;
  - a fused batch-1 forward without torch's per-call overhead.

  Before either is built, the stage p50s say which is larger.

## Files

    outputs/mp_unified/c2_fast.py              the exact path, the check and the harness
    outputs/c2/<dataset>.json                  per-question timings, checks, index rows
    outputs/c2/report.json, report.md          the table

    python outputs/mp_unified/c2_fast.py --selftest
    python outputs/mp_unified/c2_fast.py check --dataset musique --queries 30
    python outputs/mp_unified/c2_fast.py run --dataset musique --queries 200
    python outputs/mp_unified/c2_fast.py report

## Results

Filed 10 October 2026, about 02:45. The four untyped datasets ran on the laptop:
- pinned to logical processor 2, EcoQoS off, ABOVE_NORMAL (each record's `pin`);
- one thread; the first 200 s1eval questions; cold.

The table is outputs/c2/report.md (report.json and the per-question records beside it). The run log is
outputs/c2/run.log.

**Checks.** All 200 questions pass every check on every dataset. No question was excluded.
- Pools, seeds and edges equal the look's.
- C2's rows and store codes equal C1's form bit for bit.
- zrc's scores are equal on 200/200, and so are zsp's.
- The six GNN is within 5.3e-6 of the look's stored scores, with the same top 5.
- Against the step-1 cache:
  - on 2wiki, hotpotqa and squad, zrc's and zsp's scores are exact;
  - on musique, they are within 0.0085, top 5 unchanged. This is C1's BLAS-thread note: the host built that cache on
    2 threads.

**Batch 1, total per question, p50 ms (95% interval of the ratio):**

| dataset | zrc2 (MLP) | zsp2 (GNN track) | six GNN | zrc1 (C1 form) | six GNN / zrc2 | zsp2 / zrc2 | zrc1 / zrc2 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 8.8 | 9.5 | 11.5 | 12.2 | 1.30 [1.26, 1.33] | 1.08 [1.05, 1.09] | 1.38 [1.34, 1.40] |
| hotpotqa | 8.1 | 8.7 | 10.6 | 11.1 | 1.31 [1.25, 1.35] | 1.08 [1.05, 1.10] | 1.36 [1.32, 1.40] |
| squad | 5.1 | 5.6 | 6.3 | 6.9 | 1.22 [1.19, 1.26] | 1.09 [1.08, 1.10] | 1.36 [1.34, 1.37] |
| musique | 99.9 | 119.6 | 488.9 | 362.3 | 4.89 [4.69, 5.17] | 1.20 [1.16, 1.24] | 3.63 [3.39, 3.74] |

**Where the time goes now (p50 ms, batch 1):**

| dataset | path | pool + read | compile | lean inputs | links | forward |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2wiki | zrc2 | 1.4 | 2.3 | 0.7 | – | 4.2 |
| 2wiki | six GNN | 1.3 | 3.9 | – | – | 6.2 (pack + forward) |
| musique | zrc2 | 15.9 | 25.2 | 26.5 | – | 28.4 |
| musique | zsp2 | 15.8 | 24.9 | 27.2 | 18.4 | 29.6 |
| musique | six GNN | 15.9 | 88.8 | – | – | 378.2 (pack + forward) |

**Batch 16, forward per question (ms).** zrc2 against the six GNN:

| dataset | zrc2 | six GNN | multiple |
| --- | ---: | ---: | ---: |
| 2wiki | 0.96 | 8.0 | 8× |
| hotpotqa | 0.87 | 7.2 | 8× |
| squad | 0.54 | 3.6 | 7× |
| musique | 26.3 | 513 | 20× |

**Reading.**
- **The exact partial path makes the MLP 1.4× faster cold on the three small-pool datasets, and 3.6× faster on musique.**
  It changes no bit of any input or score.
- **Claim 3 now holds cold on all four datasets**, against the six GNN:
  - 1.2–1.3× on the small-pool datasets;
  - 4.9× on musique.
- **The GNN track's base (zsp) gets every one of these savings.** It reads the same inputs, so it stays within 8–20% of
  the MLP. Its extra cost is the links and its propagation step.
- **The lead is under 6× everywhere**, so per this file's rule the remaining stages come next. On the small-pool
  datasets, the batch-1 forward is now the largest stage for every path: about 4 ms of torch per-call overhead around
  a forward that costs about 1 ms per question at batch 16.
- **The six GNN has not yet had the same treatment.** Its compile and forward are its C1 forms. The next stage, C3,
  gives every model, the GNNs included, its fastest exact form under one rule. No path gets an optimisation the others
  are denied.
