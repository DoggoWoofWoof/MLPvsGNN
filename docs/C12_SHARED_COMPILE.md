# C12: the compile both models of a family share, made faster for both alike

Declared 13 October 2026 at 01:22, before any timed number. It follows C11 (docs/C11_BOTH_FAMILIES_FAST.md).

## Why

- C11 filed every model at its fastest exact form, with every input kept. Neither family's MLP is 2× its GNN.
  - **Universal:** the compile both models run is 60-74% of twin0's time. However fast twin0's forward gets, the
    shared compile holds gnn0 / twin0 at 1.42-1.85.
  - **Lean:** zsp is zrc plus its links and one propagation step, so the ratio is bounded by construction.
- C11 left the compile alone: "The shared stages (the pool and the compile) stay bit for bit the filed ones. Making
  the compile itself faster would be a separate stage, declared in its own file." C12 is that stage.
- The user, 12 October: "we are not removing stuff to make it faster we are making faster keeping the wins", and
  standing: "i dont want to give any unfair advantage to mlp make both of them the fastest".

## What changes, for every model alike

`outputs/mp_unified/c12_fast.py` (new; frozen once run). Two exact kernels go into the compile's kernel table.
`C12K` wraps C11's serial table, and every other kernel is the table's own (on the typed graphs, its typed-edge stash
too).

- **D's seed-conditioned cosines, by weight pattern.**
  - _seed_cos3 builds row v's three seed prototypes (sum_s W[k, v, s] E_S[s]) in a row buffer for every row, then
    divides by den[k, v] inside one einsum pass for the dot with E[v] and the prototype's square.
  - Rows whose weight rows W[k, v] are the same have the same prototype, and their den is the same (it is a function
    of the weights). C12 makes each distinct pattern's y = prototype / den and its einsum square once. Each row is
    then one einsum dot with y.
  - It is bit for bit _seed_cos3's: the same float32 sums in the same order, the same division and the same einsum
    lanes (fast_features._ein_dot and _lane_div).
  - The cases it does not cover go back to _seed_cos3 itself: a weight outside {0, 0.5, 1}, more than 31 seeds, or
    two rows of one pattern with a different den.
- **The node rows widened by the F16C instruction.**
  - The compile's gather turns each node row's float16 codes into float32 through a 65,536-entry table. C12 uses the
    vcvtph2ps instruction instead.
  - It is bit for bit the table's on every code but the signalling NaNs (0x7C01-0x7DFF, either sign), which the
    instruction quiets. A row holding one is redone through the table.
- **Who calls what.** Every model's compile gathers its rows, so all four get the second kernel. The full v2 compile
  (twin0 and gnn0) also calls the first. The partial compile (zrc and zsp) does not.

Nothing else changes: the pools, the edges, the links, every column, the chain entries, the fits, the forwards and the
encoder. Every compile's output is C11's, bit for bit.

## Diagnostics run before this declaration

Development numbers on the laptop, not filed ones (scratch harness; 30 questions each).

**Each kernel beside the original, inside real compiles** (every call also ran the new form into separate outputs;
both compiles; bit for bit on every call):

| dataset | seed cosines | gather | prototypes from float16 codes | ring sums, dense |
| --- | --- | --- | --- | --- |
| musique | 8.31 → 3.21 ms | 2.78 → 2.66 ms | 21.05 → 17.16 ms | 1.25 → 12.68 ms |
| metaqa | 2.73 → 1.21 ms | 2.46 → 2.38 ms | 9.99 → 9.17 ms | 0.73 → 6.51 ms |
| webqsp | 8.99 → 3.34 ms | 4.63 → 4.27 ms | 16.79 → 17.02 ms | 1.27 → 12.98 ms |

(p50 per call; the seed cosines and the prototypes run once per full compile, the gather once per compile, the ring
sums nine times.)

**Whole compiles** with the seed cosines, the gather and the prototypes from codes swapped in, the gather keeping each
row's codes for the prototypes (bit for bit on 30/30 questions):

| dataset | full v2 compile | partial compile |
| --- | --- | --- |
| musique | 64.57 → 56.72 ms (1.14×) | 19.76 → 19.83 ms |
| webqsp | 56.49 → 52.55 ms (1.08×) | 19.68 → 20.21 ms (0.97×) |

**Kept out:**
- **The prototypes from float16 codes.** They are faster on musique and metaqa, not on webqsp. They also need the
  gather to keep each row's codes, which costs the partial compile (webqsp 0.97×). There is one form for all six
  datasets, and this one is not faster on all of them.
- **The dense ring sums.** They are 5 to 10 times slower at every ring density measured.
- **Node projections made at index time.** That is a node store: C11 kept it out, and the cold headline builds
  nothing ahead of the question.

## The paths

Eight per question, in an order rotating question by question:

| path | model | what it is |
| --- | --- | --- |
| **umlp12** | twin0 | C11's umlp11 on C12's table |
| **gnn12** | gnn0 | C11's gnn11 on C12's table |
| **zrc12** | zrc | C11's zrc11 on C12's table |
| **zsp12** | zsp | C11's zsp10 on C12's table |
| umlp11, gnn11, zrc11, zsp10 | the four | C11's paths, unchanged |

**Each model's fastest form** is whichever of its two paths has the lower cold p50 on that dataset.

## Protocol

C11's:
- the laptop, pinned to one logical processor, EcoQoS off, above-normal priority (own process only);
- one thread, the session idle, and the laptop's CPU load checked before each dataset;
- the first 200 s1eval questions of each dataset;
- gc off, with a collection every 25 questions, between questions;
- cold, batch 1;
- one untimed question first, for the JIT;
- a warm repeat of each forward timed beside;
- timing and checks in separate passes: every question timed first, each path in turn with nothing between them;
  then every path run again, untimed, each output copied as it returns, and checked.

## Checks (untimed, in a separate pass after the timed one)

- **The re-run.** Each path's scores are within 1e-6 of its timed scores, with the same top 5.
- **The pools.** Every path's pool equals the look's.
- **The universal pair, bit for bit:**
  - all four universal paths' seeds and gathered node rows equal umlp11's;
  - umlp12's and gnn12's compiles equal umlp11's and gnn11's: the scalars, every family's edges, the seeds and the
    seed weights;
  - umlp12's and gnn12's scores equal umlp11's and gnn11's exactly.
- **The lean pair, bit for bit:**
  - every lean path's seeds equal the look's;
  - zrc12's and zsp12's compiles and lean rows (the rows and the decoded store rows) equal zrc11's and zsp10's;
  - zsp12's links equal zsp10's;
  - on the typed graphs, their chain entries and e_rel equal zrc11's and zsp10's;
  - their scores equal zrc11's and zsp10's exactly.
- C11's checks against the modules and the look carry over through these equalities. C11 filed them on the same
  questions.
- A question failing any check is kept out of the summaries. More than 2% failing stops a dataset.

**Development check** (30 questions each, untimed): see the end of this file, filled before any timed run.

## What is reported

`outputs/c12/<dataset>.json` and `outputs/c12/report.{json,md}`, all cold, batch 1:
- p50, p95 and p99 per path; every stage's p50; the warm forward p50 beside;
- each model's fastest form;
- **the headlines**, each with a 2,000-resample bootstrap 95% CI:
  - universal: gnn0's fastest form over twin0's;
  - lean: zsp's fastest form over zrc's;
  - across the families: the fastest GNN form of all over twin0's and over zrc's, and gnn0's over zrc's;
  - each marked: does it reach 2, and does its CI lie above 2;
- the family ratios with C12's paths on both sides and with C11's on both sides, from the same run;
- per model, C11's path over C12's: the total and the compile alone;
- each path's p50 by its position in the rotated order.

There are no thresholds. Whatever the ratios are, they are filed.

## What to expect (written before the run)

- From the development numbers: the full compile about 1.1× faster on musique and webqsp, less on metaqa, little on
  2wiki, hotpotqa and squad. gnn0 / twin0 should rise by a few hundredths at most (musique to about 1.9).
- The lean ratio should not move. The partial compile gains only the gather.
- C12 does not bring any MLP to 2× the fastest GNN. That is zsp on every dataset, and zsp is zrc plus its links and
  one propagation step.

## What this does not do

- No training and no new fit.
- No change to any pool, edge, link, column, chain entry, forward or the encoder. No input is removed.
- No pruning, quantisation, caching across questions, node store or early exit.
- Nothing is chosen per dataset: one kernel table for all six.
- No test split is read.

## Files

    outputs/mp_unified/c12_fast.py            the kernels, C12K, the paths, checks, run, report and selftest (new)
    outputs/c12/<dataset>.json                one dataset's run
    outputs/c12/report.{json,md}              the six datasets

    python outputs/mp_unified/c12_fast.py --selftest
    python outputs/mp_unified/c12_fast.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c12_fast.py run --dataset 2wiki
    python outputs/mp_unified/c12_fast.py report

Selftest, 13 October 01:12, against fast_features' serial kernels:
- h2f equals the table on all 64,514 codes that are not signalling NaNs. Those come back NaN, and a row holding one
  takes the table.
- gather_rows and gather_shards are bit for bit on five row sets of up to 2,000 rows, covering every code class.
- seed_cos3 by pattern is bit for bit on 270 random compile-like cases. 175 of them are the uncovered kinds, handed
  back to _seed_cos3. No covered case was handed back.

## Development check

13 October, 01:15-01:21: `check --queries 30` on each dataset, untimed, all eight paths and every check above.

- **Every check passed on 30 of 30 questions on all six datasets.** No question failed.
- The re-run gave every path exactly its first scores (largest difference 0).
- On metaqa and webqsp, the typed chain entries and e_rel equal C11's on every question.
- seed_cos3 by pattern ran on every call and handed none back to _seed_cos3:
  - 122 calls on 2wiki, hotpotqa, musique and metaqa, 118 on squad and webqsp;
  - the median number of patterns per call runs from 54 (squad) to 216 (musique).
- A three-question `run` and `report` on squad and metaqa, written to a scratch folder, passed every check.
- The check's own timings ran in a fixed order beside other work. They are not reported.
