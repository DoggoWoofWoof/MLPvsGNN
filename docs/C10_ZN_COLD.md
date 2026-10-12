# C10: zn, the lean MLP that reads no edges, against zsp, cold, both at their fastest

Declared 12 October 2026, about 18:40, before any timed number. It follows C9 (docs/C9_UNIVERSAL_PAIR.md) and ZN
(docs/ZN_LEAN_NO_MP.md).

## Why

The user, 12 October:
- "mlp should not have any message passing"
- "i want the lean mlp to be 2-3x times faster"
- "overall, mlp should be 2-3x faster at the minimum from the fastest gnn"
- "make all of them as fast as possible, mlp should always be much faster than gnn and we are prioritizing cold
  latencies rest all the stuff should remain same including the links, pools edges etc"

C9 filed zsp at 0.94-1.21× zrc's cold time. zrc reads five families computed over the pool graph's edges, and the
chains on the KB graphs, and zsp needs the same inputs. ZN removed them: zn reads no edge of the graph. ZN measured
what that costs in accuracy (zn keeps 13-73% of zsp's R@5 gain over rrf). C10 measures what it buys in speed.

## The pair

| | zn | zsp |
| --- | --- | --- |
| fit | `outputs/screen/fits/scr-zn` (L-musique), p@swa | `outputs/b1b/fits/zsp/J5`, p@swa (C1-C9's) |
| model | lean_screen3.ZRet | zprop.ZProp |
| inputs | rank, dense_cos, SEMB, SEED | zrc's: also topo_STRUCT, depth_STRUCT, WALK, WALKF, DISTS; the chain entries on the typed graphs; the pool links |

zsp is the fastest GNN filed (C9: the universal GNN costs more). Its latency does not depend on which fit's weights it
carries. J5 keeps C1-C9's paths and checks unchanged. The checks do depend on the weights, so each model is checked
against its own module on its own fit.

## The paths

Four paths per question, in an order rotating question by question:

| path | what it is |
| --- | --- |
| **zn10** | pool (FastPool); the node rows (FastRead); the retrieval columns (FastCompiler's kernels); the seed products from the store codes decoded once (Decoder); zn's float16 rows; ZFast and the top 5 |
| **zsp10** | pool (FastPool); c2_fast's partial compile reading the node rows through FastCompiler's own float16 gather; the edges and lean inputs (c3_fast.lean_rows3 with the codes decoded once); C6's chain entries on the typed graphs; c3_fast's links; ZFast and the top 5 |
| **zn4** | zn at the forms C9 had: the frozen pool code, nodes.read, Store.decode of the pool's codes, c3_fast.FastZ |
| **zsp4 / zsp6** | zsp's filed fastest path, unchanged: C4's on the untyped graphs, C6's on the typed graphs |

## What is new, for both models alike

`outputs/mp_unified/c10_zn.py` (new; frozen once run).

- **FastPool.** m3b's frozen pool construction in one numba pass: the seeds, the equal-RRF or dense base, the capped hop
  expansion and the union. It gives the same pool and seeds as lean_time.PoolBuilder. The selftest compares them on 468
  random graphs and lists, over every base, regime and hop setting the six datasets use. Every question is also checked
  against the look's pool.
- **The node rows through FastCompiler's float16 table.** zn reads them with FastRead. zsp's compile reads them with its
  own gather (`embeddings=None`). Both are bit for bit nodes.read's.
- **Decoder.** The pool's store codes are decoded once, straight from the code table. The result is Store.decode's,
  bit for bit.
- **ZFast.** c3_fast.FastZ's folded forward, with these changes:
  - the raw columns and their z-scores go into the first layer as one GEMM over [raw | z];
  - the raw columns are read through the float16 table and z-scored in one numba pass, with sums in float64;
  - zsp's propagation step runs in one numba pass;
  - on the typed graphs, the chain term is chains_fast.FastChain's, as in C6.

  It is the same function on the same weights, checked against the module.
- **Buffers sized once.** Every reused buffer (FastCompiler's, FastRead's, the decoders', ZFast's) is sized to the
  carve's largest pool and touched once at setup.
- **zsp10's compile runs the serial kernel table at every pool size.** At one thread, a prange kernel only adds its
  scheduling. zn's retrieval kernels are the same in both tables. Every question checks zsp10's compile against zsp's
  filed path, bit for bit.

zsp keeps everything else in its filed form: the compile's columns, the edges, WALK, WALKF, SEED, DISTS, the chain
entries and the links.

## Diagnostics run before this declaration

Two diagnostics were run while designing this stage. They are not filed numbers.

- **zn at C9's forms against zsp4**, 60 questions on the untyped graphs, cold p50 totals: zsp4 / zn was 1.87
  (hotpotqa, 2wiki), 2.17 (squad) and 2.71 (musique). On musique, zn's pool and node read were 3.2 and 7.7 ms of its
  20.7 ms.
- **The node read through the float16 table**: bit-equal to nodes.read and 3.7-8× faster.

They are why this stage rewrites the pool, the read and the decode, for both models.

## Protocol

C9's:
- the laptop, pinned to one logical processor, EcoQoS off, above-normal priority (own process only);
- one thread, the session idle;
- the first 200 s1eval questions of each dataset;
- gc off, cold, batch 1;
- one untimed question first, for the JIT;
- a warm repeat of each forward timed beside.

## Checks (untimed, every question)

- Every path's pool and seeds equal the look's.
- zn10's node rows equal nodes.read's, bit for bit.
- zn10's and zsp10's decoded store rows equal Store.decode's, bit for bit.
- zn10's and zn4's rows equal the rank, dense_cos and SEED columns of zsp's filed path, bit for bit.
- zsp10's compile (the scalars and every family's edges), its rows and its links equal zsp's filed path's, bit for bit.
- On the typed graphs:
  - zsp10's chain entries equal zsp6's and the filed build's, bit for bit;
  - its e_rel equals the look's.
- Every path's scores are within TOL = 1e-4 of its module's forward on its own rows, with the same top 5:
  - zn: lean_screen3.ZRet on scr-zn;
  - zsp: zprop.ZProp on J5, with the links and, on the typed graphs, the chains.
- A question failing any check is kept out of the summaries. More than 2% failing stops a dataset.

**Development check** (30 questions each, untimed): see the end of this file, filled before any timed run.

## What is reported

`outputs/c10/<dataset>.json` and `outputs/c10/report.{json,md}`, all cold, batch 1:
- p50, p95 and p99 per path, and every stage's p50;
- **the headline: the faster GNN form (zsp10 or zsp's filed path, by cold p50) over zn10**, with a 2,000-resample
  bootstrap 95% CI. The user's target is at least 2×. The report marks whether the ratio reaches 2 and whether its CI
  lies above 2;
- beside:
  - zsp10 / zn10 and zsp's filed path / zn10;
  - zn4 / zn10, what this stage's forms give zn;
  - zsp's filed path / zsp10, what they give zsp;
  - zsp's filed path / zn4, the diagnostic's comparison;
- the warm forward ratio zsp10 / zn10, labelled warm.

There are no thresholds. Whatever the ratios are, they are filed.

## What this does not do

- No training, no new fit, and no change to any pool, link, edge, column, chain entry or the encoder.
- zn's accuracy is ZN's: it keeps 13-73% of zsp's gain, not the 90% that claim 2 needs. C10 measures speed only. That
  gap is the next stage's work, declared in its own file.
- No test split is read. Nothing is chosen per dataset.

## Files

    outputs/mp_unified/c10_zn.py              the paths, checks, run, report and selftest (new)
    outputs/c10/<dataset>.json                one dataset's run
    outputs/c10/report.{json,md}              the six datasets

    python outputs/mp_unified/c10_zn.py --selftest
    python outputs/mp_unified/c10_zn.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c10_zn.py run --dataset 2wiki
    python outputs/mp_unified/c10_zn.py report

Selftest, 12 October 18:30: FastPool equals PoolBuilder on 468 random cases and refuses unequal list widths as it does;
the float16 table and the decode bit for bit; ZFast within 1.0e-5 of FastZ (TOL 1e-4), the same top 5.

## Development check

Run 12 October 18:35-18:43, 30 questions per dataset, untimed (`check --dataset X --queries 30`), before any timed
run. No question failed any check on any dataset.

- **Bit for bit, 30/30 on every dataset:** the pool and seeds of every path; zn10's node rows; the decoded store rows;
  zn's rows against zsp's filed columns; zsp10's compile (the serial table at every pool size, including musique's,
  metaqa's and webqsp's pools of 256 rows or more), its rows and its links.
- **On metaqa and webqsp, 30/30:** zsp10's chain entries equal zsp6's and the filed build's, bit for bit; its e_rel
  equals the look's.
- **Every path's top 5 equals its module's, 30/30.** The largest score difference from the module, per path:

| dataset | zn10 | zn4 | zsp10 | zsp's filed path |
| --- | ---: | ---: | ---: | ---: |
| 2wiki | 3.8e-6 | 3.8e-6 | 3.8e-6 | 3.8e-6 (zsp4) |
| hotpotqa | 3.8e-6 | 3.3e-6 | 3.8e-6 | 3.8e-6 (zsp4) |
| squad | 3.6e-6 | 2.7e-6 | 1.9e-6 | 1.4e-6 (zsp4) |
| musique | 8.1e-6 | 7.9e-6 | 3.3e-5 | 3.3e-5 (zsp4) |
| metaqa | 4.8e-6 | 4.3e-6 | 6.3e-5 | 6.4e-5 (zsp6) |
| webqsp | 1.3e-5 | 1.3e-5 | 7.8e-5 | 7.7e-5 (zsp6) |

TOL is 1e-4. zsp10's largest differences match its filed path's to within 1e-6, so they come from zsp's own forward on
the large pools, not from this stage's forms. C9 kept 200/200 webqsp questions at the same TOL.

The check also prints p50s, but in a fixed order (zn10, zsp10, zn4, then zsp's filed path), so whichever path first
touches the graph's data pays for it. They are not the stage's timing and are not used. A development diagnostic on 2wiki
in rotated order (60 questions, twice; not filed) gave zn10 1.15, zsp10 3.33, zn4 3.06 and zsp4 5.66 ms.

## Results

Run 12 October 2026, 19:01-19:11, on the laptop under the protocol above: pinned to logical processor 2, EcoQoS off,
above-normal priority, one thread, the session idle. 200 questions per dataset. **Every question passed every check
on every dataset** (kept 200/200 everywhere). Files: outputs/c10/<dataset>.json and outputs/c10/report.{json,md}.

### The headline: the fastest GNN form over zn10, cold

Cold p50 per question in ms, 95% bootstrap intervals (2,000 resamples). The faster GNN form is zsp10 on all six.

| dataset | zn10 | zsp10 | zsp's filed path | **fastest GNN / zn10** | at least 2× | CI above 2 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 1.11 | 5.79 | 11.68 (zsp4) | **5.22 [3.50, 13.57]** | yes | yes |
| hotpotqa | 1.05 | 2.78 | 4.73 (zsp4) | **2.65 [2.58, 2.75]** | yes | yes |
| squad | 0.75 | 2.02 | 3.18 (zsp4) | **2.71 [2.62, 2.86]** | yes | yes |
| musique | 9.05 | 48.11 | 65.26 (zsp4) | **5.31 [5.20, 5.43]** | yes | yes |
| metaqa | 7.78 | 30.71 | 44.87 (zsp6) | **3.95 [3.88, 4.01]** | yes | yes |
| webqsp | 8.86 | 85.07 | 106.35 (zsp6) | **9.61 [9.05, 10.08]** | yes | yes |

Beside it:

| dataset | zsp filed / zn10 | zn4 / zn10 | zsp filed / zsp10 | zsp filed / zn4 | warm forward zsp10 / zn10 (warm) |
| --- | --- | --- | --- | --- | --- |
| 2wiki | 10.53 [6.09, 15.21] | 2.43 [2.36, 2.51] | 2.02 [0.45, 4.22] | 4.34 [2.48, 6.27] | 1.43 [1.38, 1.49] |
| hotpotqa | 4.50 [4.37, 4.76] | 2.40 [2.34, 2.49] | 1.70 [1.64, 1.79] | 1.88 [1.81, 1.99] | 1.44 [1.40, 1.49] |
| squad | 4.26 [4.12, 4.41] | 2.07 [2.02, 2.16] | 1.57 [1.50, 1.62] | 2.06 [1.98, 2.12] | 1.41 [1.38, 1.44] |
| musique | 7.21 [7.08, 7.38] | 2.61 [2.55, 2.66] | 1.36 [1.34, 1.38] | 2.77 [2.72, 2.83] | 1.55 [1.51, 1.59] |
| metaqa | 5.77 [5.69, 5.85] | 2.57 [2.53, 2.61] | 1.46 [1.44, 1.48] | 2.25 [2.21, 2.27] | 1.72 [1.69, 1.76] |
| webqsp | 12.01 [11.24, 12.66] | 3.09 [2.97, 3.16] | 1.25 [1.21, 1.28] | 3.89 [3.69, 4.08] | 4.12 [3.99, 4.26] |

- **This stage's forms** make zn 2.07-3.09× faster (zn4 / zn10) and zsp 1.25-2.02× faster (zsp filed / zsp10), with
  the same scores (the checks above).
- **Where zsp10's cold time goes** (stage p50s, report.md): on musique, the compile 20.3 ms, the lean inputs 13.7, the
  links 6.1 and the forward 7.1, against 9.05 ms for zn10's whole path. On webqsp, the chain entries 36.3, the compile
  20.6 and the lean inputs 7.0. The inputs computed over the pool graph's edges are most of the GNN's time.
- p95 and p99 per path are in report.md.

### 2wiki: two groups of GNN times, from the run's untimed checks

On 2wiki only, whichever GNN path (zsp10 or zsp4) ran first on a question took far longer in its compile than the one
run after it. Read by group (split after the run; not the declared headline):

| path | first GNN path on the question (100 questions) | after the other GNN path (100) |
| --- | --- | --- |
| zsp10 total, p50 | 24.63 ms; / zn10 22.21 [18.28, 28.26] | 2.94 ms; / zn10 2.65 [2.54, 2.84] |
| zsp4 total, p50 | 25.41 ms | 5.25 ms |
| zsp10 compile, p50 | 22.66 ms | 1.29 ms |

- The MLP paths do not depend on their position: zn10's p50 is 0.99-1.15 ms at every position.
- On the other five datasets, the gap between the groups is at most 0.4 ms on the passage graphs (hotpotqa 2.96 against
  2.64 ms) and 2.4 ms on webqsp.
- The all-question p50 sits between the two groups, which is why 2wiki's interval is wide.

**The cause is the run's own untimed checks.** A probe and a diagnostic were run after the run; neither is filed.
- **The probe.** The edge stores the compile reads are held in RAM. The node table is a 17.14 GiB memory map (150 shards
  of 40,000 rows); hotpotqa's is 14.97 GiB.
- **The diagnostic.** 40 questions, each running zn10 and then zsp10's compile twice, with nothing else between questions
  (no checks and no C9-era paths). The first compile's p50 is 2.80 ms and the second's 1.90 ms. The edge read accounts
  for most of the gap (1.12 against 0.54 ms). The 22 ms does not come back.
- **The explanation.** C10's run checks every question after its four paths, untimed: module forwards, nodes.read,
  Store.decode and the filed path's comparisons. On this 16.8 GB laptop that work pushes the edge stores out of the
  process's memory. The next question's first path that reads them pays the page faults. zn reads no edge, so it never
  pays.
- **What this means for zsp's time.** It is not zsp's own cold cost. A server that answers question after question
  keeps its stores resident. The headline stands as declared. The conservative reading of 2wiki is the second group:
  zsp10 / zn10 = 2.65 [2.54, 2.84], still above 2.
- **For later stages.** The checks move out of the timed loop: each timed path records its scores, and a separate
  untimed pass checks them.

### What it means

- **zn10 is at least 2× faster than the fastest GNN form on all six datasets, with every interval above 2**
  (2.65-9.61×).
- **zn is not a model the paper ships.** It keeps 13-73% of zsp's gain (ZN), not the 90% of Claim 1
  (docs/PAPER_CLAIMS_2026_10_09.md; this file's declaration calls it claim 2). It is the
  ceiling: how fast an MLP gets when it reads no edge. Together with ZN, it shows that the inputs computed over the
  edges are both where the GNN's time goes and where the accuracy comes from.
- **The user, 12 October, after this run:** "all these models should be the best models wrt to the recall coverage
  etc, we are not removing stuff to make it faster we are making faster keeping the wins". No model loses an input
  to gain speed.
- **The next stage, C11,** runs the best models of both families at this stage's forms, every input kept, with the
  same scores:
  - the lean pair, zrc and zsp;
  - the universal pair, the twin and the six GNN.
- **The lean pair is bounded by construction.** zsp is zrc's inputs plus one propagation step over scores. C9 bounds
  zsp / zrc under about 1.6×, however cheap the shared stages get. C11 files it as it is.
