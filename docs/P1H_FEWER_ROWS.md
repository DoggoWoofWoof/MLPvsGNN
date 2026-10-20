# P1h: the same coverage in fewer rows

Declared 20 October 2026, before any number. The user's direction of 20 October (fewer pool rows at the same coverage). Code: `outputs/mp_unified/zp1h.py`. Not run yet.

Pools only: nothing trains, nothing is fit, and the cut reads no gold. CPU only, on the host.

## 1. Why

**The user's direction (20 October):** reduce the nodes in the pools where possible, to raise precision. The
standing rules hold: one rule for all six datasets, no parameter fit, nothing fit to golds, coverage first and then
the fewest rows.

**Where the pools stand.** P1g2 met the user's target (`outputs/p1g2/verdict.json`: TARGET_MET). Its rule,
`g:f1.5+cut_cos_5000`, keeps every gold of at least 0.95 of s1sel's questions on all five training datasets. It
costs 2.8 times today's rows: 1,975.5 against 699.7 (P1f's cost, the question-weighted mean rows over the five).

| dataset | ALL, P1g2's rule | mean rows | ALL, today's pool | mean rows |
| --- | ---: | ---: | ---: | ---: |
| metaqa | 0.9646 | 3,406.4 | 0.9519 | 2,021.7 |
| squad | 0.9793 | 50.0 | 0.9793 | 50.0 |
| musique | 0.9505 | 6,599.8 | 0.8846 | 2,093.5 |
| hotpotqa | 0.9914 | 1,091.2 | 0.9556 | 94.1 |
| 2wiki | 0.9693 | 1,486.3 | 0.9525 | 105.1 |
| webqsp (s1eval, beside) | 0.9328 | 4,739.5 | 0.8410 | 2,119.9 |

(s1sel unless marked; P1g2's verdict, `chosen_s1sel`, `today_s1sel` and `beside_webqsp_s1eval`.)

P1g-b built these pools on the fit and s1eval carves. On s1eval they hold 1.68 (metaqa), 1.00 (squad), 3.15
(musique), 11.60 (hotpotqa), 14.14 (2wiki) and 2.24 (webqsp) times today's rows (docs/P1GB_POOLS_REFIT.md,
section 6).

**What the extra rows did to the models.** P1g-b refit the lean pair on these pools: NOT_ADOPTED on 18 October
(zrckp 1 GAIN, 0 LOSS; zspk 0 GAIN, 1 LOSS; docs/P1GB_POOLS_REFIT.md, section 7). The new golds reached the top
five on webqsp zero-shot, but not in domain: metaqa fell on every read, and musique in domain fell on both models.
More rows in a pool are more rows to rank.

**Where there is room.** hotpotqa (0.9914) and 2wiki (0.9693) sit above 0.95 at 11.6 and 14.1 times today's rows.
musique sits on the line: 0.9505 of its 1,534 s1sel questions is 1,458, the fewest that meet 0.95 (1,457 would be
0.9498). One τ serves every dataset, so a cut that costs one of those musique questions one gold fails the choice.
Whether any τ below 1 passes is what P1h finds out.

**Which score.** P1g-b's pools files store no score (scripts/p1gb_pools.py, `pool_arrays`). Per question they hold
the base and seeds, |I_q|, today's fill size, the expansion's nodes and three counts (fill_k, fill_want, x_count).
The rule's nodes also come from three different orders: the first stage's RRF for the base, KIND_T's order (a walk,
or RRF5T) for the fill, and the cosine rank for the extra set. There is no common score to take a share of. So the
cut uses the rank mass 1/(60 + rank) over the pool's build order. 60 is the RRF constant of the first stage
(equal_rrf) and of RRF5T.

## 2. The cut

**The build order.** Each question's P1g-b pool, in the order its rule built it:
1. **B**, the base and seeds (the floor; their order inside B does not matter);
2. **fill_f**: KIND_T's first min(|full|, round(1.5 × today's fill)) nodes outside B, in KIND_T's order (full is
   KIND_T's order without B);
3. **the extra nodes**: the extra set X's nodes outside fill_f, by their rank in X's own order. X is the first 5,000
   of B's one-link neighbours by cos(q, v) (D5b's cos order), so this part runs down the cosine. X is computed only
   where today's fill is above 0 (P1g2's gate).

These are P1g-b's nodes, node for node (section 6 checks them). Only part 3's order differs from the file's: P1g-b
files X in node-id order, the order P1g's cut returns it in.

**The mass.** Position p of the build order (p = 1 is B's first node) carries 1/(60 + p). In a pool of n
positions, the first k carry the share S(k) / S(n), where S(k) = 1/61 + 1/62 + … + 1/(60 + k).

**The cut.** For one global τ, each question keeps the shortest prefix whose share reaches τ: the smallest k with
S(k) ≥ τ × S(n). Then:
- **the floor:** k is at least |B|, so base and seeds always stay;
- **the ceiling:** k is at most n, so no pool grows;
- τ = 1 keeps every pool unchanged.

**What the cut reads.** Two sizes per question, |B| and n, and the build order to name the nodes it keeps. It reads
no gold and no label, and nothing in it is fit. The mass depends on the position alone, so the kept length depends
only on |B|, n and τ. Long pools lose the most. With B empty, τ = 0.9 keeps 736 of 1,000 positions (74%) and 4,327
of 7,000 (62%); τ = 0.99 keeps 971 (97%) and 6,672 (95%). These figures describe the rule on round lengths; they are
not data.

**The floor is base and seeds.** Every model builds its pool from them: m3b_compile's `build_pool` returns base ∪
seeds ∪ expansion, and P1g-b's `replace_pools` stops when base and seeds are not the filed ones. No model reads a
minimum row count. The only row threshold in the feature code (256 rows, src/mp_retrieval/m3b_features.py) decides
threading, not rows. squad's s1sel pools are B alone (KIND_T gives all 1,498 questions an empty fill), and its
s1eval pools keep today's size (×1.00), so the cut leaves squad almost untouched.

## 3. Choosing τ (s1sel)

- **The grid:** 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99, 1.0. No other value is read, and the script
  refuses one.
- **ALL, on counts:** the share of a carve's questions whose every gold the cut pool keeps. A question with no gold
  counts 0, as in P1a to P1g. A τ meets a dataset when 100 × (questions with every gold kept) ≥ 95 × (questions):
  0.95 is read on whole questions, with no rounding.
- **τ\*:** the smallest grid value that meets all five training datasets (metaqa, squad, musique, hotpotqa, 2wiki)
  on s1sel. webqsp has no s1sel and is not read.
- **No τ below 1:** if only τ = 1 meets, or none does, there is no cut (NO_CUT, section 4).
- **The s1sel pools.** P1g-b built fit and s1eval pools only. So P1h rebuilds P1g-b's rule on s1sel, through
  P1g-b's own functions and P1g's cut, and checks every question against P1g's filed s1sel arrays and P1g2's verdict
  (section 6).

## 4. The grade (s1eval)

At τ\*, and at every grid value beside, per dataset:
- ALL (with its count) and recall;
- rows: mean, p50 and p90, against P1g-b's mean rows, and the share cut;
- the questions cut and the golds lost.

**Overall:** P1f's cost on s1eval at τ\* against τ = 1. The cost is the question-weighted mean of the five training
datasets' mean rows. The weights are fit + s1eval questions, from P1c2's manifests: metaqa 15,745, squad 17,729,
musique 7,018, hotpotqa 13,335, 2wiki 18,504. The fall in rows is 1 − cost(τ\*) / cost(1).

webqsp is graded beside. Only s1sel and s1eval are read; test splits never are.

### The rule, fixed before any number

- **CUT:** τ\* < 1; on s1eval, ALL ≥ 0.95 (on counts) at τ\* on all five training datasets; and P1f's cost falls
  by at least 10%.
- **NO_CUT:** otherwise. That is: only τ = 1 meets on s1sel, or s1eval falls below 0.95 on a training dataset at
  τ\*, or the cost falls by less than 10%.
- **INCOMPLETE:** an array is missing or is not this stage's. That covers an orders file, its gold positions or its
  record (stamp or sha256), and the select or grade record. It also covers select and grade not computed on the
  files on disk.

**Reported beside, deciding nothing:** webqsp at every τ; recall; p50 and p90; every grid τ's s1eval table, with the
questions cut and the golds lost.

The verdict closes only this form (docs/SCREENS.md, section 3b), and it names the next (section 5).

## 5. What each outcome leads to (fixed now)

| P1h's verdict | the next form |
| --- | --- |
| CUT | The cut pools at τ\* become the candidate pools for the next model refits: the free GNN G6 (declared in parallel) and the lean pair. Each is adopted only through that model's own declared screen; P1h adopts no model change. The fit carves' cut pools come from the same build order and τ\*, built in that stage's own file. |
| NO_CUT | A learned cascade: a cheap scorer's per-question cut, its one setting chosen the same way on s1sel (the smallest grid value keeping ALL ≥ 0.95 on all five), in its own declared file. |
| INCOMPLETE | The missing array is built; nothing else changes. |

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/zp1h.py --selftest` (about 3 s) checks, on
toy pools:
- **The cut:** on 157 toy pools at every grid τ, the kept prefix equals a rewrite without numpy (running sums and a
  bisection). It is the shortest prefix reaching τ, never below base and seeds and never above the pool, and it
  grows with τ. τ = 1 returns every pool unchanged; a τ off the grid is refused.
- **The build order:** on toy graphs, the extra set equals P1g's, set for set (cos, w, rrf and res cuts of 50 to
  300). The filed order is P1g-b's expansion, and the build order holds the same nodes, its extra part running down
  the cosine.
- **The checks:** on P1g-b's and P1g's own layouts, the per-question and per-carve checks pass. They stop on a
  changed node, |I_q|, fill, held gold or summary, and on a pools file of another rule.
- **No gold in the cut:** it opens no gold file, and gives the same pools with the gold positions permuted and with
  them removed.
- **The choice:** the smallest τ meeting all five (1,457 of 1,534 fails, 1,458 meets); webqsp's numbers never move
  it.
- **The verdict:** CUT end to end on a toy root. NO_CUT when only τ = 1 meets musique, on an s1eval miss and on a
  9.99% fall. INCOMPLETE on a missing orders file, gold file, record or grade, a foreign record, stale select and
  grade records and an off-grid τ.
- **Refusals:** test splits, carves outside s1sel and s1eval (webqsp's s1sel too) and foreign roots. Nothing is
  written under outputs/p1h, p1gb, p1g, p1g2, p1f or p1c2/pools.

**Development runs, before this commit.** On the laptop's copy of the frozen package (the same freeze record),
`orders --limit` ran KIND_T's order, the extra set's ranks and every per-question check against P1f's stats and
P1g's filed arrays on s1sel: metaqa (30 questions), squad (30) and musique (30, then 150). All passed. Nothing was
written, and no coverage or row number was logged. On 2 threads, musique took 30 s for 30 questions and 59 s for 150
(the walks' compilation is in both): about 0.24 s per added question. The s1eval checks read P1g-b's pools, which
only the host holds, so they first run there.

**Commands.**

    python outputs/host_ops/pylib_run.py outputs/mp_unified/zp1h.py orders DATASET --carve s1sel --threads 2 --host
    python outputs/host_ops/pylib_run.py outputs/mp_unified/zp1h.py orders DATASET --carve s1eval --threads T --host
    python outputs/mp_unified/zp1h.py select
    python outputs/mp_unified/zp1h.py grade
    python outputs/mp_unified/zp1h.py verdict

`orders` writes three files per carve under `outputs/p1h/orders/<D>/`:
- `<D>__<carve>.npz`: base and seeds and the build order, with no gold;
- `<D>__<carve>.golds.npz`: each gold's place in the build order, read by the grade only;
- `<D>__<carve>.json`: the record.

`select`, `grade` and `verdict` write `outputs/p1h/select`, `grade` and `verdict` (`.json` and `.md`). `--limit N`
runs the first N questions with every check and writes nothing.

**The orders pass's checks.** Any failure stops the item, and nothing is written.
- **s1eval, per question:** B, |I_q|, today's fill, the expansion as filed (node for node), fill_k, fill_want and
  x_count equal P1g-b's pools file.
- **s1eval, per carve:** ALL, recall and rows (mean, p90, max) equal P1g-b's manifest. The manifest's rule must be
  P1g2's chosen one, and the frozen substrate's record must be the one P1g-b built on.
- **s1sel, per question:** |B| and |I_q| equal P1f's stats, and today's fill equals P1g's. The rows and held golds
  equal P1g's filed arrays under the chosen rule, with P1g2's gate.
- **s1sel, per carve:** ALL, recall, rows and p90 equal P1g2's verdict.
- **Both:** the build order holds the filed expansion's nodes, with the same first fill_k.

**Host items** (`p1h-*`, 14, CPU only):
- **5 s1sel orders** (`p1h-orders-<D>-s1sel`): 2 CPUs and 2 threads each, and the memory of P1g's stats items:
  metaqa 3, squad 2, musique 3, hotpotqa 6 and 2wiki 8 GB.
- **6 s1eval orders** (`p1h-orders-<D>-s1eval`), with the CPUs, threads and memory of P1g-b's pools items:
  - metaqa 3 CPUs and 6.5 GB, under retry_cmd as P1g-b's was;
  - squad 2 and 3.5, musique 2 and 7.7;
  - hotpotqa 6 and 23.4, 2wiki 8 and 29.2;
  - webqsp 3 and 7.4.

  These are the CPU counts of the runs that filed the arrays, so the float sums under the cosine ranks are the same.
- **select** (after the five s1sel orders), **grade** (after select and the six s1eval orders) and **verdict**
  (after grade): 1 CPU and 2 GB each.
- **No earlier item.** P1g-b's pools items ended on 17 October, and P1f's and P1g's stats before that. Their files
  are the items' inputs: P1g-b's pools manifest and s1eval file for each s1eval item, and P1f's and P1g's s1sel
  stats for each s1sel item. So the feeder waits if one is not in the workspace.
- **Inputs:** P1g-b's pools items' lists without the fit carve's files and P1c2's pools, with the carve's own step
  4d, 4e and P1e files, plus zp1h.py.
- The sizes are copied from those items, not measured for P1h.

**Disk, an estimate:** about 0.41 GB in all, at 4 bytes per node position before compression (the files are
compressed, so less). Per item, the orders file; the gold file and the record add under 1 MB each:
- **s1sel** (P1g2's s1sel rows): metaqa 20 MB, squad 0.3 MB, musique 41 MB, hotpotqa 7 MB, 2wiki 9 MB;
- **s1eval** (P1c2's s1eval rows times P1g-b's ratios): metaqa 133 MB, squad 2 MB, musique 64 MB, hotpotqa 32 MB,
  2wiki 75 MB, webqsp 29 MB;
- **select, grade and verdict:** under 1 MB each.

**ETA, an estimate.** No full run of this pass is filed. Per question it costs P1f's walks (KIND_T's order) plus
one extra set. P1g's stats priced 77 extra sets per question: 0.51 to 2.41 s per s1sel question, metaqa to 2wiki
(outputs/p1g/stats_<D>.json). musique's development runs took about 0.24 s per added question on the laptop, about
an eighth of P1g's 1.885 s on the host. Per item, a central estimate at that eighth (never below P1f's walks), with the
upper bound at P1g's full rate in brackets, each plus a minute or two to open the dataset:
- **s1sel:** metaqa 2 min (13 min), squad 1 to 2 min (KIND_T alone: no squad question has a fill), musique 6 min
  (48), hotpotqa 8 min (59), 2wiki 8 min (60);
- **s1eval:** metaqa 12 min (1.4 h), squad 11 min (KIND_T alone), musique 10 min (1.3 h), hotpotqa 37 min (4.8 h),
  2wiki 1.1 h (8.4 h), webqsp 9 min (1.2 h);
- **select, grade and verdict:** a minute or two each.

From the start to the verdict: about 1.5 hours, at most about 9, set by 2wiki's s1eval item. The items reserve about
16 CPU-hours at the central estimate and at most about 115 (hotpotqa's and 2wiki's s1eval items hold 6 and 8 CPUs).

The encoder stays frozen. webqsp never trains. Test splits are never read. No LLM or transformer.

## 7. Results

Not run yet.
