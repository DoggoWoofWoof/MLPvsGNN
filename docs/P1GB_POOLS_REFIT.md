# P1g-b: the lean pair refit on P1g2's chosen pools

Declared 17 October 2026, after P1g2's verdict ([P1G2_GATED_EXTRA.md](P1G2_GATED_EXTRA.md), section 6; P1g's own is
[P1G_POOL_BEAM.md](P1G_POOL_BEAM.md), section 9) and before any number of this stage. Code: `scripts/p1gb_pools.py`, `outputs/mp_unified/look_p1gb.py`, `outputs/mp_unified/zp1gb.py`.
Numbers here are development numbers. Status: declared.

## 1. The question

The user's target (15 October): **ALL ≥ 0.95 on every dataset, coverage first, then the fewest rows.** P1g2 chose one
pool rule for all six datasets on s1sel, from P1g's 805 rules and their gated forms: `g:f1.5+cut_cos_5000` (1.5 times today's fill, plus the first 5,000
of B's one-link neighbours by the question's cosine, gated), TARGET_MET at 1,975.5 rows against today's 699.7
(P1g2's section 6). A pool is:
- **B:** the question's base and seeds, unchanged;
- **fill_f:** the first round(f × |today's fill|) nodes of KIND_T's order outside B (P1e's order, unchanged; above f = 1
  the fill goes on along the same order and stops at its end);
- **X:** the chosen extra set's nodes outside B and fill_f (P1g's `extra_sets`, unchanged);
- **the gate** (a gated rule only): on a question whose fill today is 0 (|I_q| = |B|), X is dropped and the pool is B
  alone. KIND_T's size rule already judged that question needs no expansion, so no label is read.

A pool that holds more golds raises what a ranker can reach. It also adds negatives, and the ranker must find the
golds among more rows. P1g-b asks whether the lean pair ranks better when trained and read on P1g2's chosen pools,
in-domain and zero-shot:
- **the MLP, zrckp:** zrc's model plus the pointwise KT head (161 weights), with no message passing;
- **the GNN, zspk:** zsp's propagation head plus the KT head.

Neither model changes, nor do their settings, seeds or batches. The graph, the encoder, the carves and the relation
tables are today's.

## 2. The pools (`scripts/p1gb_pools.py pools`)

For each of P1c2's eleven carves (each training dataset's fit carve and the six s1eval carves), per question:
1. KIND_T's order, through P1c2's `kind_t` and P1e's orders, unchanged;
2. full = the order's nodes outside B; today's fill = full's first |I_q| − |B| nodes;
3. fill_f and X as above (X only where the gate is open); the pool's expansion is fill_f, then X's nodes outside
   fill_f.

**Checks** (the stage stops on any failure):
- P1g2's verdict is TARGET_MET or CEILING_MET and was written by P1g2's filed script, naming P1g's filed script;
- on every question, B is P1c2's filed base and seeds, |I_q| is P1c2's, and today's fill recomputed here is P1c2's
  filed expansion;
- on each s1eval carve, the pools' ALL, gold recall and mean rows equal P1g2's record of the chosen rule (P1g2's
  `eval` record for the five training datasets, its verdict's webqsp record from P1g's `stats` arrays);
- no pool repeats a node, and no expansion holds a node of B.

Written to `outputs/p1gb/pools/<D>/<D>__<carve>.npz` (step 4c's arrays where they apply, plus `today_count`,
`fill_want`, `fill_k` and `x_count`) and `pools.json` (each file's sha256, its rows, and the ALL and recall of P1c2's
pools and of P1g-b's).

## 3. The features

Everything is the filed pipeline, on P1g-b's roots (`outputs/p1gb/`):
- **looks** (`look_p1gb.py`): look_x_six, unchanged, after `m3b_compile.prepare`; each question's pool becomes B plus
  the filed expansion (`p1gb_pools.replace_pools`). The replacement stops on a missing question, a base and seed set
  or a frozen pool that is not the filed one, or an expansion that is not the filed length, repeats a node or touches
  B;
- **caches:** `lean_cache.py --look-root outputs/p1gb/look --out-root outputs/p1gb/cache`;
- **chains and links:** rmatch's and zrc's chains (metaqa and webqsp) and zlink's pool edges, through `zp1gb.py build`;
- **the KT input** (`p1gb_pools.py kt`): P1c3's eight columns (`p1c3_kt.columns`, unchanged), the same orders
  recomputed per question. The fill column marks fill_f. Checks: B and fill_f recomputed are the filed ones, and each
  question's cache rows are its pool, each node once. Beside it, the same pass writes one side column, each row's
  place in P1d's RRF5 order scaled as P1c3's columns (`side_rrf5.npy`), for the ledger's item L2 (RRF5's position as
  a ninth KT column). No P1g-b fit reads it, and L2 is declared in its own file;
- **the gate** (`p1gb_pools.py gate`), before any fit, on every carve:
  - the pools file is the manifest's, and every carve's pools come from one rule;
  - the cache's questions are step 1's and the file's;
  - step 1's rows are the filed |I_q|, and P1g-b's are the filed pool sizes (|B| + the expansion);
  - every look shard filed a pools record naming the file and its counts;
  - the KT record is on the cache's parts and root.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and L-hotpotqa (`scr-p1gb-zrckp`, `-hp`; `scr-p1gb-zspk`, `-hp`), with
  settings, seeds and batches unchanged, each read on the six s1eval carves of P1g-b's pools.
- **Bases:** the same arm's card fit of the split on P1c2's pools:
  - zrckp: `scr-p1c3p-zrckp` and `scr-p1c3p-zrckp-hp`;
  - zspk: `scr-p1c3-zspk` and `scr-p1c3-zspk-hp`.

  The reads use the same questions and gold totals. R@5 is over the question's golds, so a gold a pool misses counts
  as a miss, and a gold it adds counts as a hit only when it is ranked in the top five. Step 1's fits are reported
  beside. The re-call's base screens are each base fit compared with step 1's fit of the split
  (`outputs/p1gb/base-<arm>-<split>.json`), as G2's were.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). Each arm is
  decided against its own base. GAIN, LOSS and WITHIN are docs/SCREENS.md's.
- **Reads too large for the card** may be re-run with `--rows-cap` (whole questions per batch; the scores do not
  change). Each one is recorded in its read.json, as G2's were.

### The rule, fixed before any number (P1c2's, unchanged)

- **ADOPT_FOR_FULL_RUN:** neither arm's re-call has a LOSS on any of its twelve reads, and the two arms together have
  at least one GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a re-call is missing, is not P1g-b's, or does not hold twelve reads.

Both families read the same pools, so a pool that costs either arm is not adopted.

**Reported beside, deciding nothing:**
- FC@5 and hit@1 per read, as the compares give them;
- each carve's ALL, recall and mean rows under P1c2's pools and P1g-b's (from `pools.json`);
- the rows per question each arm scores, against P1c2's.

## 5. What follows

- **ADOPT_FOR_FULL_RUN:** the full run, six splits per arm under grade-nullx, in its own file. P1g2's chosen pools
  are then the pools of P3's gap ledger, G2r, E1, P5 and the refits of screens adopted on P1c2's pools (L1, L3, T1).
  The new rows' cost (KIND_T's longer fill, the extra set, and the features and forwards of more rows) is timed cold
  beside P4's timings before any latency claim.
- **NOT_ADOPTED:** no full run. P1g2's coverage stands as a pool result. The record says, read by read, whether the
  pools' extra golds reached the top five, which is where P3's ledger starts. The lean pair stays on P1c2's pools.

Flagged, as for every pool so far: KIND_T's walks and the extra set read the graph. The pool is built before either
model runs and both models read the same pool, so the MLP's own scoring stays free of message passing.

## 6. How it runs

- **Self-tests (pass before this commit):**
  - `python scripts/p1gb_pools.py --selftest`: the expansion (fill_f, then X outside it, cut at the order's end); the
    gate (X only on questions today's fill expands); the pools file and its rows; the replacement of any length and its refusals; the KT columns with fill_f and their
    refusals; the rows check; the chosen rule and its refusals;
  - `python outputs/mp_unified/zp1gb.py --selftest`: the arms are the lean pair as filed; P1g-b's looks, caches,
    chains, links, gate and KT root in place, P1c2's and P1c3's restored, today's relation tables untouched; train
    refuses another arm, split, fit name, cache root or device and needs P1g-b's gate PASS; read refuses a fit that
    is not P1g-b's; relz's pair and re-call against each arm's card fits on P1c2's pools, stamped P1g-b's; the
    verdict's three outcomes.
- **Host items** (379; `p1g2-eval-*` and `p1gb-*`), from P1c2's and P1c3's filed items, with shards scaled by each
  dataset's rows (metaqa ×1.68, squad ×1.00, musique ×3.15, hotpotqa ×11.60, 2wiki ×14.14, webqsp ×2.24):
  - P1g2's five s1eval runs (beside; each dataset's pools wait for its run, which their check reads);
  - six pools and the base copy;
  - 177 look shards and 140 cache parts (P1c2's 51 and 39, scaled);
  - 17 builds, 11 KT passes and the gate;
  - the four fits and four reads on the card at `--frac 0.92`, share 0.94, one at a time (P1c3p's 0.30 scaled by
    the larger of metaqa's and musique's row ratios);
  - four base screens, four compares, two pairs, two re-calls and the verdict.
- **Disk:** P1c2's roots take 40.3 GB on the host (looks 17.8, caches 14.1, chains 4.7, links 3.5; measured
  17 October). Scaled per dataset by the rows, P1g-b's take about 120 GB: metaqa 34, musique 34, 2wiki 26,
  hotpotqa 14, webqsp 12 and squad 1. The host had 159 GB free at 03:00 on 17 October. The feeder's 30 GB floor
  holds the heavy writers if other projects take the drive. P1c2's roots stay until E1's and T1's reads are done.
- **ETA** (P1c2's measured runs, scaled by the rows):
  - evals and pools: about 1.5 hours;
  - looks: about 70 job-hours (P1c2's 24.5 × 2.9), about 5 hours at the feeder's caps;
  - caches, builds, KT and the gate: about 1.5 hours;
  - the four fits and reads, one at a time on the card: about 6 hours.

  About 14 hours in all.
- The encoder stays frozen. webqsp never trains. Test splits are never read.

## 7. Results

Not run yet.
