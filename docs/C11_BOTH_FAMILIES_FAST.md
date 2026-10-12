# C11: both families, every model at its fastest exact form, cold

Declared 12 October 2026, about 21:15, before any timed number. It follows C9 (docs/C9_UNIVERSAL_PAIR.md) and C10
(docs/C10_ZN_COLD.md).

## Why

The user, 12 October:
- "universal have same input features for both and lean can have different depending on what is required by the model
  but overall, mlp should be 2-3x faster at the minimum from the fastest gnn"
- "all these models should be the best models wrt to the recall coverage etc, we are not removing stuff to make it
  faster we are making faster keeping the wins, so we need to be better not make our models worse"
- "get me the best models that are the faster possible"
- Standing: "i dont want to give any unfair advantage to mlp make both of them the fastest"

C10 filed zn10 at 2.65-9.61× faster than the fastest GNN form. zn reads no edge, though, and keeps only 13-73% of zsp's
gain. No model loses an input to gain speed, so zn is not one of the models timed here.

C11 times the best models of both families, each at its fastest exact form. Every new form is given to both models of
a family alike.

## The pairs

| family | MLP | GNN | inputs |
| --- | --- | --- | --- |
| universal | u_mlp_v2_mix, the six pair's twin, seed 0 (`twin0`, C9's) | u_gnn_v2_ef, seed 0 (`gnn0`, C1-C9's six GNN) | the same for both: the v2 compile's 197 columns, the 1,536-wide node embeddings, the pool edges by family, the seeds |
| universal-lean | zrc (`outputs/b1b/fits/zrc/J5`, p@swa) | zsp (`outputs/b1b/fits/zsp/J5`, p@swa) | zrc's; zsp adds the pool links and one propagation step over scores |

- The fits, their records and their checks are C9's.
- Nothing is retrained.

## The paths

Eight paths per question, in an order rotating question by question:

| path | model | what it is |
| --- | --- | --- |
| **umlp11** | twin0 | FastPool; FastCompiler's full v2 compile on the serial kernel table at every pool size, reading the node rows through its own float16 gather; those rows then serve the forward; FusedTwin; the top 5 |
| **gnn11** | gnn0 | the same pool and compile; the six GNN's pack (lean_time.fast_pack; on the typed graphs c5_typed.fast_pack5, with the relation slots); FusedGNN11; the top 5 |
| **zrc11** | zrc | C10's forms, as zsp10 has them: FastPool; c2_fast's partial compile on the serial table through the compile's own gather; the lean rows with the store codes decoded once (Decoder); C6's chain entries on the typed graphs; ZFast; the top 5 |
| **zsp10** | zsp | C10's path, unchanged |
| umlp9 | twin0 | C9's path, unchanged |
| gnn4 / gnn5 | gnn0 | the filed path: C4's on the untyped graphs, C5's on the typed graphs |
| zrc4 / zrc6 | zrc | the filed path: C4's / C6's |
| zsp4 / zsp6 | zsp | the filed path: C4's / C6's |

**Each model's fastest form** is whichever of its two paths has the lower cold p50 on that dataset.

## What is new, for both models alike

`outputs/mp_unified/c11_fast.py` (new; frozen once run).

- **FusedInput: one input-block code for both universal models.**
  - It is m3b_models.InputBlock at batch 1: twin0 uses it with its column gates (universal_v2_models.GatedInputBlock),
    gnn0 without them.
  - The node and query projections and the input linear stay torch GEMMs.
  - Every column between them is written into one reused buffer by numba passes:
    - the scalars, raw and z-scored, with segment_zscore's float32 sums in row order, bit for bit;
    - the semantic head's 258 columns and the difference column (summed in float64);
    - the three family prototypes, summed in edge order per node as index_add_ sums them;
    - the reach prototype.
  - gnn0's cell steps, evidence flow and readout stay FusedGNN's (C3). twin0's body and readout stay the module's.
  - It replaces both C9's FastTwin input and C3's FusedGNN input, so the two models' input blocks are again the same
    code.
- **The compile's own gather serves the forward** (both universal models).
  - FastCompiler reads the node rows through its float16 table, their norms in the same pass. This is C10's form for
    zsp.
  - The forward reads those same rows, so nodes.read is no longer called.
  - The rows are nodes.read's, bit for bit (checked).
- **The full v2 compile runs the serial kernel table at every pool size** (both universal models). This is C10's rule
  for the partial compile.
- **FastPool** for both universal models.
- **zrc gets the forms zsp10 already has** (C10): FastPool, the serial table, the compile's own gather, Decoder, ZFast,
  and C6's chain entries on the typed graphs.
- **Every reused buffer is sized once**, to the carve's largest pool, and touched once at setup, for all four models.
- **Timing and checks are separate passes.**
  - Why: in C10, the untimed checks between questions pushed 2wiki's edge stores out of memory on this 16.8 GB laptop,
    and the next question's first path paid the page faults.
  - What C11 does: every question is timed first, each path in turn, with nothing between the paths. Then every path
    runs again, untimed, and is checked.
  - The re-run's scores must equal the timed pass's: within 1e-6, with the same top 5.

Nothing else changes: the pools, the edges, the links, every column, the chain entries, the fits and the encoder.

## Diagnostics run before this declaration

Two diagnostics were run while designing this stage. They are development numbers, not filed ones.

- **Where the universal pair's cold time goes** (musique, 40 questions, C9's forms, the laptop under load).
  - The full v2 compile took 99 ms. Its biggest stages were the fixed neighbour aggregation (C) at 38.1 ms
    (memory-bound), the depth basis at 24.8, the seed-conditioned aggregation (D) at 17.3, the edges at 7.8, the
    topology at 6.3 and the retrieval columns at 3.4.
  - C4 filed musique's compile at 59 ms unloaded, so these are proportions, not times.
  - The compile is shared by both universal models.
- **The fused input block** (40 questions per dataset): the top 5 was the module's on 40/40 questions, and base_z
  matched bit for bit on 40/40.

| dataset | twin0 forward, C9's → fused | gnn0 forward, C3's → fused | largest difference from the module (twin0 / gnn0) |
| --- | --- | --- | --- |
| hotpotqa | 5.56 → 2.66 ms | 9.36 → 6.34 ms | 5.7e-6 / 1.9e-6 |
| musique | 59.66 → 36.67 ms | 175.4 → 149.98 ms | 5.7e-6 / 3.8e-6 |

## Protocol

C10's:
- the laptop, pinned to one logical processor, EcoQoS off, above-normal priority (own process only);
- one thread, the session idle, and the laptop's CPU load checked before each dataset;
- the first 200 s1eval questions of each dataset;
- gc off, with a collection every 25 questions, between questions;
- cold, batch 1;
- one untimed question first, for the JIT;
- a warm repeat of each forward timed beside.

## Checks (untimed, in a separate pass after the timed one)

Every path runs again on every question. Then:

- **The re-run.** Each path's scores are within 1e-6 of its timed scores, with the same top 5.
- **The pools.** Every path's pool equals the look's.
- **The universal pair's inputs, bit for bit:**
  - umlp11's and gnn11's seeds equal umlp9's;
  - their node rows equal nodes.read's (umlp9's);
  - their compiles equal the GNN's filed path's: the scalars, every family's edges, the seeds and the seed weights;
  - gnn11's pack equals the filed path's in every field but the node rows, which are checked above;
  - their base_z equals segment_zscore's.
- **The universal pair's scores:**
  - umlp11 and umlp9 are within TOL of the twin module's forward on the same question's pack, with the same top 5. They
    also have the same top 5 as the look's stored twin0 scores.
  - gnn11 and the GNN's filed path are within TOL of FastGNN (C1's reference form), with the same top 5. They also have
    the same top 5 as the look's stored gnn0 scores.
- **The lean pair:**
  - every lean path's seeds equal the look's;
  - zrc11's and zsp10's decoded store rows equal Store.decode's, bit for bit;
  - zrc11's and zsp10's rows and compiles equal their filed paths', bit for bit;
  - zsp10's links equal c1_cold.links_of's, bit for bit;
  - on the typed graphs, zrc11's and zsp10's chain entries equal their filed paths' and the filed builds', bit for bit,
    and their e_rel equals the look's;
  - every lean path is within TOL of its module's forward on its own rows, with the same top 5. zsp's module gets the
    links, and on the typed graphs both modules get the chains.
- TOL is 1e-4.
- A question failing any check is kept out of the summaries. More than 2% failing stops a dataset.

**Development check** (30 questions each, untimed): see the end of this file, filled before any timed run.

## What is reported

`outputs/c11/<dataset>.json` and `outputs/c11/report.{json,md}`, all cold, batch 1:
- p50, p95 and p99 per path; every stage's p50; the warm forward p50 beside.
- Each model's fastest form.
- **The headlines**, each with a 2,000-resample bootstrap 95% CI:
  - **universal:** gnn0's fastest form over twin0's fastest form;
  - **lean:** zsp's fastest form over zrc's fastest form;
  - **across the families:**
    - the fastest GNN form of all (gnn0's or zsp's) over twin0's fastest form;
    - the same over zrc's fastest form;
    - gnn0's fastest form over zrc's fastest form.
- Each headline is marked: does the ratio reach 2, and does its CI lie above 2. The user's target is at least 2× over
  the fastest GNN.
- Per model, the filed path over the new one: what this stage's forms give each model.
- The warm forward ratio within each family, labelled warm.
- Each path's p50 by its position in the rotated order, so that an order effect like C10's 2wiki one shows.

There are no thresholds. Whatever the ratios are, they are filed.

## What this does not do

- No training and no new fit.
- No change to any pool, edge, link, column, chain entry or the encoder. No input is removed.
- No pruning, quantisation, caching across questions, node store or early exit.
- The shared stages (the pool and the compile) stay bit for bit the filed ones. Making the compile itself faster would
  be a separate stage, declared in its own file.
- zn is not timed again; C10 filed it.
- No test split is read. Nothing is chosen per dataset.
- **The lean ratio is bounded by construction.** zsp is zrc plus the links and one propagation step, so C9 bounds
  zsp / zrc under about 1.6×, however cheap the shared stages get. C11 files it as it is.

## Files

    outputs/mp_unified/c11_fast.py            the paths, checks, run, report and selftest (new)
    outputs/c11/<dataset>.json                one dataset's run
    outputs/c11/report.{json,md}              the six datasets

    python outputs/mp_unified/c11_fast.py --selftest
    python outputs/mp_unified/c11_fast.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c11_fast.py run --dataset 2wiki
    python outputs/mp_unified/c11_fast.py report

Selftest, 12 October 20:55. FusedInput was run against InputBlock and GatedInputBlock on 360 random cases. They cover
empty families, no edges, no seeds, more seeds than slots, and constant, near-constant and 1e4-scaled columns.
- The raw and z-scored scalars and base_z: bit for bit.
- The other columns of the linear's input: within 2.4e-7.
- h: within 3.4e-7 of the largest |h| (or of 1).

## Development check

Run 12 October 20:58-21:08, 30 questions per dataset, untimed (`check --dataset X --queries 30`), before any timed
run. No question failed any check on any dataset.

- **Bit for bit, 30/30 on every dataset:**
  - every path's pool;
  - umlp11's and gnn11's seeds, node rows and compiles (the full v2 compile on the serial table at every pool size,
    including musique's, metaqa's and webqsp's pools of 256 rows or more);
  - gnn11's pack, and both universal models' base_z;
  - every lean path's seeds; zrc11's and zsp10's decoded store rows, rows and compiles; zsp10's links.
- **On metaqa and webqsp, 30/30:** zrc11's and zsp10's chain entries equal their filed paths' and the filed builds',
  bit for bit; their e_rel equals the look's.
- **The re-run** gave every path the same scores on every question (largest difference 0).
- **Every path's top 5 equals its module's, 30/30.** The universal paths' top 5 also equal the look's stored scores,
  and their scores are within 1.1e-5 of them. The largest score difference from the module, per path:

| dataset | umlp11 | umlp9 | gnn11 | gnn0's filed path | zrc11 | zrc's filed path | zsp10 | zsp's filed path |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki | 3.8e-6 | 0 | 2.9e-6 | 1.9e-6 (gnn4) | 7.6e-6 | 7.6e-6 (zrc4) | 3.8e-6 | 3.8e-6 (zsp4) |
| hotpotqa | 5.7e-6 | 0 | 1.9e-6 | 1.9e-6 (gnn4) | 5.7e-6 | 4.8e-6 (zrc4) | 3.8e-6 | 3.8e-6 (zsp4) |
| squad | 3.8e-6 | 0 | 1.9e-6 | 1.9e-6 (gnn4) | 1.9e-6 | 2.9e-6 (zrc4) | 1.9e-6 | 1.4e-6 (zsp4) |
| musique | 5.7e-6 | 0 | 3.8e-6 | 3.8e-6 (gnn4) | 5.1e-5 | 5.0e-5 (zrc4) | 3.3e-5 | 3.3e-5 (zsp4) |
| metaqa | 7.6e-6 | 0 | 4.8e-6 | 3.8e-6 (gnn5) | 8.1e-5 | 8.1e-5 (zrc6) | 6.3e-5 | 6.4e-5 (zsp6) |
| webqsp | 7.6e-6 | 0 | 4.3e-6 | 2.9e-6 (gnn5) | 3.8e-5 | 4.0e-5 (zrc6) | 7.8e-5 | 7.7e-5 (zsp6) |

- TOL is 1e-4.
- umlp9's FastTwin equals the module bit for bit (C9).
- The lean paths' largest differences match their filed paths' to within 2e-6. They come from the lean models' own
  forwards on the large pools (C10), not from this stage's forms.

The check also prints p50s, but in a fixed order, so whichever path first touches the graph's data pays for it. They
are not the stage's timing and are not used. The stage p50s it prints say where the time goes:
- **The full v2 compile, which both universal models share,** is 60-67 ms on musique and webqsp and 36 ms on metaqa.
- **The universal forwards** on those datasets: twin0's (umlp11) 22-23 ms, gnn0's (gnn11) 50-92 ms.
- **zrc11's chain entries on webqsp** (its contribution rule) take longer than zsp10's (rmatch's mass cap): 56 against
  42 ms.

## Results

Run 12 October 2026, 21:11-21:45, on the laptop under the protocol above: pinned to logical processor 2, EcoQoS off,
above-normal priority, one thread, the session idle. The CPU load before each dataset was 7.8-27.7%, with 3.7-5.3 GB
free. 200 questions per dataset. Files: outputs/c11/<dataset>.json and outputs/c11/report.{json,md}.

- **Every question passed every check on every dataset** (kept 200/200 everywhere).
- The re-run gave every path exactly its timed scores (largest difference 0).
- The largest score difference from the module, per path:
  - umlp11 1.1e-5 and gnn11 6.7e-6;
  - zrc11 9.2e-5 and zsp10 7.8e-5, both on webqsp. Their filed paths reach 8.8e-5 and 7.7e-5 there.
  - TOL is 1e-4.

### Universal: gnn0 over twin0, cold

Cold p50 per question in ms. The ratios have 95% bootstrap intervals (2,000 resamples). The new path is each model's
fastest form on all six datasets.

| dataset | twin0: umlp11 | twin0: umlp9 (C9) | gnn0: gnn11 | gnn0: filed | **gnn0 / twin0** | at least 2× | warm forward, gnn0 / twin0 (warm) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 4.71 | 7.14 | 6.76 | 9.29 (gnn4) | **1.44 [1.36, 1.48]** | no | 2.27 [2.20, 2.33] |
| hotpotqa | 4.35 | 6.53 | 6.18 | 8.30 (gnn4) | **1.42 [1.36, 1.48]** | no | 2.29 [2.20, 2.39] |
| squad | 2.95 | 4.25 | 4.61 | 6.09 (gnn4) | **1.56 [1.52, 1.62]** | no | 2.73 [2.60, 2.87] |
| musique | 83.26 | 109.57 | 154.13 | 178.98 (gnn4) | **1.85 [1.81, 1.88]** | no | 4.14 [4.05, 4.19] |
| metaqa | 57.24 | 80.29 | 88.42 | 106.91 (gnn5) | **1.54 [1.52, 1.57]** | no | 2.40 [2.36, 2.45] |
| webqsp | 77.79 | 106.41 | 117.33 | 141.14 (gnn5) | **1.51 [1.49, 1.52]** | no | 2.73 [2.70, 2.78] |

C9 filed 1.27-1.65× for the same pair. Both models got the same forms. The ratio rose because those forms shrank the
work the two models share, which is a larger part of the MLP's time.

### Universal-lean: zsp over zrc, cold

| dataset | zrc: zrc11 | zrc: filed | zsp: zsp10 | zsp: filed | **zsp / zrc** | at least 2× | warm forward, zsp / zrc (warm) |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| 2wiki | 2.72 | 4.34 (zrc4) | 2.76 | 4.80 (zsp4) | **1.01 [0.99, 1.05]** | no | 1.11 [1.09, 1.14] |
| hotpotqa | 2.41 | 3.95 (zrc4) | 2.52 | 4.27 (zsp4) | **1.04 [1.02, 1.09]** | no | 1.14 [1.10, 1.17] |
| squad | 1.81 | 2.61 (zrc4) | 1.90 | 3.00 (zsp4) | **1.05 [1.02, 1.08]** | no | 1.15 [1.11, 1.17] |
| musique | 37.16 | 54.36 (zrc4) | 43.71 | 61.55 (zsp4) | **1.18 [1.16, 1.19]** | no | 1.12 [1.10, 1.13] |
| metaqa | 29.77 | 45.52 (zrc6) | 32.18 | 47.68 (zsp6) | **1.08 [1.06, 1.10]** | no | 1.06 [1.02, 1.10] |
| webqsp | 88.66 | 111.79 (zrc6) | 84.54 | 108.30 (zsp6) | **0.95 [0.93, 0.98]** | no | 1.02 [1.00, 1.04] |

### Across the families

The fastest GNN form of all is zsp10, on all six datasets.

| dataset | zsp10 / twin0 | zsp10 / zrc | gnn0 / zrc |
| --- | --- | --- | --- |
| 2wiki | 0.59 [0.56, 0.61] | 1.01 [0.99, 1.05] | 2.49 [2.38, 2.56] |
| hotpotqa | 0.58 [0.56, 0.61] | 1.04 [1.02, 1.09] | 2.56 [2.45, 2.66] |
| squad | 0.64 [0.62, 0.66] | 1.05 [1.02, 1.08] | 2.54 [2.45, 2.64] |
| musique | 0.52 [0.52, 0.53] | 1.18 [1.16, 1.19] | 4.15 [4.07, 4.19] |
| metaqa | 0.56 [0.55, 0.57] | 1.08 [1.06, 1.09] | 2.97 [2.92, 3.02] |
| webqsp | 1.09 [1.05, 1.18] | 0.95 [0.93, 0.98] | 1.32 [1.24, 1.36] |

### What this stage's forms give each model

Each model's filed path over its new path, cold:

| dataset | twin0: umlp9 / umlp11 | gnn0: filed / gnn11 | zrc: filed / zrc11 | zsp: filed / zsp10 |
| --- | --- | --- | --- | --- |
| 2wiki | 1.52 [1.43, 1.57] | 1.38 [1.32, 1.43] | 1.60 [1.55, 1.66] | 1.74 [1.66, 1.78] |
| hotpotqa | 1.50 [1.43, 1.56] | 1.34 [1.29, 1.41] | 1.64 [1.58, 1.71] | 1.70 [1.60, 1.77] |
| squad | 1.44 [1.41, 1.48] | 1.32 [1.25, 1.37] | 1.44 [1.40, 1.48] | 1.58 [1.52, 1.63] |
| musique | 1.32 [1.29, 1.33] | 1.16 [1.15, 1.18] | 1.46 [1.43, 1.48] | 1.41 [1.39, 1.43] |
| metaqa | 1.40 [1.38, 1.43] | 1.21 [1.18, 1.23] | 1.53 [1.49, 1.55] | 1.48 [1.45, 1.51] |
| webqsp | 1.37 [1.35, 1.39] | 1.20 [1.19, 1.22] | 1.26 [1.21, 1.29] | 1.28 [1.22, 1.31] |

- Every model is faster with the same scores: twin0 by 1.32-1.52×, gnn0 by 1.16-1.38×, zrc by 1.26-1.64× and zsp by
  1.28-1.74×. zsp10 is C10's path, unchanged.
- p95 and p99 per path are in report.md.

### Where the time goes

Stage p50s in ms. They do not add up exactly to the totals' p50s.

**Universal:**

| dataset | the compile both models run | twin0's forward | gnn0's pack + forward | the compile's share of twin0's p50 |
| --- | ---: | ---: | ---: | ---: |
| 2wiki | 2.86 | 1.72 | 0.20 + 3.48 | 61% |
| hotpotqa | 2.66 | 1.52 | 0.19 + 3.19 | 61% |
| squad | 1.76 | 1.08 | 0.18 + 2.61 | 60% |
| musique | 61.28 | 21.72 | 3.64 + 88.63 | 74% |
| metaqa | 35.98 | 20.79 | 1.74 + 50.99 | 63% |
| webqsp | 55.35 | 22.01 | 2.23 + 59.38 | 71% |

**Lean:**

| dataset | zrc: compile / lean inputs / chains / forward | zsp: compile / lean inputs / chains / links / forward |
| --- | --- | --- |
| 2wiki | 1.27 / 0.60 / – / 0.66 | 1.19 / 0.58 / – / 0.14 / 0.72 |
| hotpotqa | 1.08 / 0.56 / – / 0.60 | 1.00 / 0.56 / – / 0.13 / 0.66 |
| squad | 0.75 / 0.45 / – / 0.45 | 0.68 / 0.43 / – / 0.12 / 0.48 |
| musique | 18.40 / 12.51 / – / 5.64 | 18.37 / 12.53 / – / 5.76 / 6.27 |
| metaqa | 11.71 / 5.94 / 4.99 / 6.86 | 11.68 / 6.08 / 4.89 / 1.65 / 7.42 |
| webqsp | 19.97 / 7.24 / 44.54 / 16.30 | 20.16 / 7.10 / 37.30 / 2.31 / 16.73 |

- **In the universal family, the compile both models run is 60-74% of twin0's cold time.**
- **How far a faster compile could go.** These numbers are worked out from the stage p50s above, not timed.
  - With the compile at zero, gnn0 / twin0 would be 2.04 (2wiki), 2.12 (hotpotqa), 2.47 (squad), 4.20 (musique),
    2.52 (metaqa) and 2.77 (webqsp).
  - For 2×, the compile would have to fall to:
    - 48.5 ms on musique (1.26× faster);
    - 17.3 ms on webqsp (3.2×);
    - 10.9 ms on metaqa (3.3×);
    - 0.55 ms on squad (3.2×).
  - On 2wiki and hotpotqa it would have to fall to 0.07 and 0.19 ms, so the compile alone cannot get there. twin0's
    forward there is half of gnn0's.
- **In the lean family, zsp's message passing costs its links and its second forward:** 0.15-6.4 ms, which is 3-17% of
  zrc's time.
- **On webqsp, zrc is the slower model.** Its chain rule takes 44.5 ms, against 37.3 ms for zsp's.

### 2wiki: the first path on each question pays for its edge reads

- On 2wiki, whichever path runs first on a question takes 36-52 ms longer in its compile. This is the p50 over the 25
  questions each path is first on. On hotpotqa the extra is 2-3 ms. The paths after it do not pay.
- **It is the compile's edge reads, not the node rows.**
  - umlp9 reads its node rows in a stage of its own, and that stage does not pay: 0.91 ms first, 0.80 ms otherwise.
  - umlp9's compile, which reads the edge stores, does pay: 54.5 ms first, 2.5 ms otherwise.
- **Every model pays it alike.** Each path runs first on 25 of the 200 questions, so the p50s are each model's cost
  with the question's edges in memory.
- **This corrects C10.** C10 found the same effect on 2wiki and put it down to its untimed checks between questions.
  C11's timed pass has no checks between questions, and the first path still pays, so the checks were not the only
  cause. This laptop has 16.8 GB of memory, and 2wiki's node table is a 17.14 GiB memory map.

### What it means

- **The best models of both families now run at their fastest exact forms**, with every input kept and the same
  scores.
- **Neither family's MLP reaches 2× its GNN.**
  - **Universal: 1.42-1.85×.** The compile both models share is most of the MLP's time. Making it faster for both is
    the next lever. It can reach 2× on musique, webqsp, metaqa and squad (a compile 1.26-3.3× faster). On 2wiki and
    hotpotqa, the forwards keep the ratio near 2 even with a free compile.
  - **Lean: 0.95-1.18×.** zsp is zrc plus its links and one propagation step over scores, so it costs only 1-18% more in all.
    No fair change makes zrc 2× faster than zsp. The lean GNN's message passing is cheap, and the lean MLP keeps
    0.97-0.99 of its gain on B1 without it.
- **Against the universal GNN, the lean MLP (zrc) is 2.49-4.15× faster on five datasets**, and 1.32× on webqsp, where
  zrc's chain entries are its largest stage.
- **The lean GNN (zsp10) is faster than the universal MLP (twin0)** on five datasets (0.52-0.64×): the lean inputs
  cost less than the universal compile.
- **Where the shared compile's time goes.** A development profile (not filed; 40 questions, the serial table, pinned)
  covered the three datasets with about 2,000-row pools.

  | stage, ms | musique | metaqa | webqsp |
  | --- | ---: | ---: | ---: |
  | C: per-family neighbour prototypes over the 1,536-wide rows | 24.1 | 11.6 | 14.9 |
  | depth basis | 16.0 | 8.1 | 11.1 |
  | D: seed prototypes | 11.6 | 5.8 | 10.1 |
  | typed basis | – | 1.5 | 10.2 |
  | whole compile | 66.0 | 33.1 | 65.0 |

  At about 2,000 rows, the pool's float32 rows (13 MB) do not fit in the core's cache. C reads them once per edge.
- **The next stage, C12,** makes the shared compile faster, the same for both models of each family. It is declared in
  its own file, before any number.
