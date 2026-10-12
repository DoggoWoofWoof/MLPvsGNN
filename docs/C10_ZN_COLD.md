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

To be filled before any timed run.

## Results

To be filled after the run.
