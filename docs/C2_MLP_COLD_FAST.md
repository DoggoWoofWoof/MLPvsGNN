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

Not yet run.
