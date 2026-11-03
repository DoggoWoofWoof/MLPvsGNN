# COV: pick O's coverage and localization record

Declared 4 November 2026 (about 02:30 IST), before any COV number. Module: `outputs/mp_unified/covdiag.py`. Records:
`outputs/cov/report.json` and `report.md`. It writes nothing else.

Nothing trains, nothing is fit and nothing is chosen. This is a record: it reads filed pools and filed golds, and
decides nothing about pools or models.

## 1. Why

**U2w named this record** (docs/U2W_ONE_POOL_RULE_THREE_DATASETS.md, section 8): "Pick O's coverage on s1eval and B1
(section 5's measures, for pick O alone) is the coverage and localization record either way, so it is measured now from
PKO's and B1P's filed pools, in its own file." This is that file.

**No such record exists yet.** PKO filed rows, not recall: "No recall, R@5 or ceiling of PKO's pools is computed in
this stage" (docs/PKO_PICK_O_POOLS_UNDER_WQ.md, section 7).

**The question asked on 4 November**: at full scale, do our pools hold more golds than other rankers hold in the same
number of rows? Section 4 answers it from P3a's filed full-corpus orders, which keep every gold's rank.

## 2. The pools, the golds and the corpora (all read-only)

**s1eval, the full corpora.** The three datasets 2wiki, hotpotqa and musique.
- **Pools:** PKO's s1eval parts, `outputs/pko/pools/parts/<D>__s1eval__<i>of<n>.npz`. That is 5 parts on 2wiki
  (12,576 questions), 3 on hotpotqa (7,405) and 1 on musique (2,417).
  - Each part's sha256 must equal its record's `file_sha256`, and `zpko.check_filed` must pass on it.
  - A question's pool is `zpko.pools_of` (B and the kept expansion), and its rows are `zpko.pool_sizes`. Both are
    imported unchanged; they are what the fits read.
- **Golds:** P3a's filed orders, `outputs/p3a/orders_<D>.npz` (`gptr`, `golds`), matched to PKO's questions by id.
- **The corpus:** N_D is the node count of the pools' graph: the record's `graph_detail` npz, `len(indptr) - 1`. The
  file's sha256 must equal the record's `npz_sha256`, and every part of a dataset must name the same graph.

**B1, HippoRAG 2's settings** (1,000 questions each).
- **Pools:** B1P's `outputs/b1p/pools/<D>__b1w.npz`, the hyperlink-free pools of record on B1. The sha256 must equal
  the record's `sha256`, and `zpko.check_filed` must pass. Their rows are package rows.
- **Golds:** `outputs/bench/hipporag2_w/<D>/queries.json`, whose golds index the setting's `nodes.json`. They are
  mapped to package rows through `nodes.json`'s `package_row`.
- **The corpus:** the setting's distinct package rows (B1P section 6: 6,119 to 11,654 passages).

## 3. The measures (U2w's section 5, for pick O alone)

For each dataset, on s1eval and on B1:
- **ALL:** the share of questions with every gold in the pool;
- **gold recall:** the share of golds in the pool (golds held over golds, pooled over questions); the question mean
  of each question's share sits beside;
- **rows per question:** mean, median, 90th percentile and maximum;
- **rows as a share of the corpus:** mean rows / N_D, and median rows / N_D;
- **HippoRAG 2's candidate set** beside: every passage of the corpus (N_D rows), because its walk scores the whole
  graph. On s1eval, N_D is the full corpus; on B1, the setting.

A question with no gold is left out of every measure and counted.

**Intervals:** 95%, over 2,000 question resamples with seed 0, as `b1c_walks.boot_ci` draws them (HLW's estimator).
ALL uses `boot_ci` itself. Gold recall is a ratio of sums, so it is recomputed on the same resamples.

## 4. Beside, on s1eval: other rankers at the pool's own size

**The comparison.** For each question q, let k_q be its pool's rows. Each of P3a's filed orders X is read at depth
min(k_q, len_X(q)): a gold is held when its 0-based rank `pos_X` is at least 0 and below that depth. The orders are:
- **A1:** HippoRAG 2's restart rule walked on the corpus hyperlinks. On musique, which has none, A1 walks our three
  link families, and A1 = A1_all (P3a);
- **A1_all:** HippoRAG 2's rule walked on all three link families;
- **RRF:** our first stage, the RRF of the dense and SPLADE top-1,000 lists;
- **DENSE** and **SPLADE:** the first-stage lists, 1,000 deep.

**What each comparison reports:**
- each order's ALL and gold recall at k_q;
- the paired differences, pool − X, with 95% intervals on the same resamples.

**The call.** For each dataset, order and measure:
- MORE when the interval's low end is above 0;
- LESS when its high end is below 0;
- SAME otherwise.

These calls are fixed now.

**Depth.** P3a cut the walks at 7,000 nodes. RRF's list holds at most about 2,000 nodes, and DENSE's and SPLADE's
1,000.
- Where k_q exceeds an order's length, the order is read at its full length, because it holds nothing further. Those
  questions are counted.
- The paired differences are also reported on the questions where k_q fits inside the order (beside, no call).

**Inputs differ.** A1 walks the corpus hyperlinks, while pick O's pools use none: their graph is W2_S, the LLM-free
link rule (U2b). So the walk has an input the pools lack.

## 5. Checks

Any failure stops the run, and nothing is written.
- **Shas:** every pools file and graph file sha256 against its record.
- **Pools:** `zpko.check_filed` on every pools file. Each pool's size from `pools_of` must equal `pool_sizes`.
- **s1eval ids:** PKO's ids must equal P3a's as sets, with no duplicates, in the counts of section 2.
- **B1 ids:** the pools' ids must equal `queries.json`'s, in order.
- **Ranges:**
  - on s1eval, every gold and every pool row must lie in [0, N_D);
  - on B1, every pool row must be one of the setting's package rows.
- **Golds:**
  - on s1eval, each question's golds must be distinct, because P3a's ranks are kept per gold;
  - on B1, golds that map to the same package row count once, and the questions where that happens are counted.
- **The first stage agrees.**
  - The base B of pick O's pools is RRF's head, so P3a's RRF top 10 should lie inside the pool.
  - The share of s1eval questions where it does is reported.
  - A share below 0.99 stops the run, as a sign of an id or question mismatch.
- **The self-test** (`python outputs/mp_unified/covdiag.py selftest`) must pass before the run. It checks, on made-up
  arrays:
  - the measures against brute-force set counts;
  - the depth rule, with ranks of −1 and ranks past an order's length;
  - the ratio interval against a direct loop on `boot_ci`'s resamples;
  - the calls;
  - the B1 mapping through `package_row`, with a duplicate package row;
  - refusals of a sha mismatch, an id mismatch and an out-of-range row.

## 6. Running

```
python outputs/mp_unified/covdiag.py selftest
python outputs/mp_unified/covdiag.py run         (on the host, where PKO's and B1P's pools and the graphs are)
```

The run reads about 0.1 GB of pools and three graph files (up to about 0.6 GB each, hashed once), then counts. It
should take a few minutes on 1 CPU (estimate). Its paths are written relative to the repository.

## 7. What follows

Nothing is decided here. If U2x carries a pick, U2w's stage B measures the same on the new pools with this module's
functions, and this record is the comparison row. A record closes nothing; the next coverage question gets its own
file.

## 8. Filed (4 November 2026, about 02:35 IST)

The run took 25.5 s on the host at 02:27 IST, and every check passed. P3a's RRF top 10 lay inside the pool on 100.00% of
s1eval questions on all three datasets. No question lacked a gold, and no B1 question had two golds on one package row.
Records: `outputs/cov/report.json` and `report.md`.

**Pick O's pools.** ALL and gold recall in percent, with 95% intervals. HippoRAG 2's candidate set is every passage of
the corpus column.

| carve | dataset | ALL | gold recall | rows: mean / median / p90 | corpus | mean rows / corpus |
|---|---|---|---|---|---:|---:|
| s1eval | 2wiki | 96.9 [96.6, 97.2] | 98.7 [98.6, 98.8] | 1,081.2 / 1,090 / 1,383 | 5,989,847 | 0.018% |
| s1eval | hotpotqa | 96.0 [95.5, 96.4] | 97.9 [97.7, 98.1] | 706.0 / 688 / 1,017 | 5,233,329 | 0.013% |
| s1eval | musique | 94.8 [93.9, 95.6] | 97.8 [97.4, 98.1] | 5,263.6 / 4,980 / 6,490 | 117,534 | 4.48% |
| B1 | 2wiki | 91.9 [90.1, 93.6] | 96.1 [95.2, 96.9] | 88.5 / 77 / 145 | 6,119 | 1.45% |
| B1 | hotpotqa | 97.0 [95.9, 98.1] | 98.5 [98.0, 99.1] | 65.3 / 53 / 100 | 9,811 | 0.67% |
| B1 | musique | 99.8 [99.5, 100.0] | 99.9 [99.8, 100.0] | 3,433.6 / 3,455 / 3,532 | 11,654 | 29.46% |

**At each pool's own size, on s1eval.** Pool minus order in gold recall, in points, with 95% intervals and the call:

| order | 2wiki | hotpotqa | musique |
|---|---|---|---|
| A1 (HippoRAG 2's walk on the corpus hyperlinks) | +0.6 [+0.5, +0.8] MORE | -1.6 [-1.8, -1.4] LESS | +0.9 [+0.6, +1.2] MORE |
| A1_all (its rule on all three link families) | +0.6 [+0.4, +0.8] MORE | -1.5 [-1.7, -1.3] LESS | +0.9 [+0.6, +1.2] MORE |
| RRF (our first stage) | +22.3 [+21.9, +22.7] MORE | +6.3 [+5.9, +6.7] MORE | +9.2 [+8.4, +10.0] MORE |
| DENSE | +25.2 [+24.8, +25.6] MORE | +8.0 [+7.6, +8.4] MORE | +12.0 [+11.2, +12.9] MORE |
| SPLADE | +24.1 [+23.7, +24.5] MORE | +9.7 [+9.2, +10.1] MORE | +18.4 [+17.5, +19.3] MORE |

- The ALL calls go the same way everywhere. Against A1: 2wiki +1.5 [+1.1, +1.9], hotpotqa -3.3 [-3.7, -2.9],
  musique +1.7 [+1.0, +2.4].
- **Depth.**
  - On musique, every pool is longer than RRF's list (about 2,000 nodes), so that row compares the pool with RRF's
    whole list.
  - DENSE's and SPLADE's lists are 1,000 deep, shorter than most pools: 8,139 of 12,576 questions on 2wiki, 839 on
    hotpotqa, all on musique.
  - A1 is shorter than the pool on 157 musique questions. On the other 2,260 the difference is +0.95 [+0.64, +1.29].
  - RRF is shorter than the pool on 5 2wiki questions. Without them, the difference is +22.3 [+21.9, +22.8].

**Reading** (deciding nothing).
1. **Localization.** On the full corpora, pick O's pools keep 97.8% to 98.7% of golds in 706 to 5,264 rows per
   question. That is 0.013% of hotpotqa's corpus and 0.018% of 2wiki's; musique's 4.48% is of a 117,534-passage corpus.
   HippoRAG 2's walk scores every passage.
2. **Against our first stage.** In the same number of rows, the pools hold many more golds: +22.3 points on 2wiki and
   +6.3 on hotpotqa, where RRF's list is deep enough.
3. **Against HippoRAG 2's walk read to the same depth.**
   - The pools hold slightly more golds on 2wiki and musique, and fewer on hotpotqa.
   - The walk uses the corpus hyperlinks; the pools use none.
   - The walk covers well deep in its order (as P1a found). Its weak point is its head: at full scale its R@5 falls
     below RRF's (HLW, part B).
4. **On B1's small settings** the pools take a larger share: 0.67% to 1.45% of the setting on hotpotqa and 2wiki, and
   29.46% on musique.

**What follows.** Nothing is decided. If U2x carries a pick, its stage B's coverage is read against this record.
