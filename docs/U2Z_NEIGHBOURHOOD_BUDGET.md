# U2z: pool depth from the question's own neighbourhood

Declared 4 November 2026 (about 08:55 IST), before any U2z number. Module: `outputs/mp_unified/zu2z.py`, run on the
laptop. It imports `zu2s` and `zu2w` unchanged, and through them U2m, U2e, P1e, P1f and P1h. Records:
`outputs/u2z/choice.json` and `choice.md`.

Stage A is pools only: nothing trains, nothing is fit, no walk runs, and no statistic is computed that U2s's arrays do
not already hold. The one new thing is a budget read from a count they hold, the size of the base's one-link
neighbourhood (`n_n1`). The choice reads golds on s1sel, as every pool stage's choice does
(docs/U2S_WALK_MASS_BUDGET.md, section 6).

**Read before this declaration:** the filed results of U2s to U2y, the sizes in section 1's table (means of counts in
U2s's filed s1sel arrays), and `zu2z.py dev`, which checks the new tables on one construction of musique against U2e's
evaluate and prints no coverage. No coverage or rows of any new candidate has been computed or read.

## 1. Why

**U2y found no unified rule** (docs/U2Y_WALK_MASS_CUT.md, section 7; filed in 1d02163): NOT_UNIFIED and
KIND_FREE_TOO_DEAR, nothing carried. Its section 5 named this form: one rule that picks a construction for each corpus
before any gold is read, from the corpus's own pool sizes in U2s's filed arrays.

**The gap is depth.** The fewest mean rows that reach level 0.96 on each dataset alone, over the universal candidates
(U2w's `per_dataset_fewest_rows`), are 4,122.4 (musique), 467.5 (hotpotqa) and 441.3 (2wiki): 8.8 to 9.3 ×. Every
budget so far spends about the same on all three:
- U2x's AGREE, cheapest 0.96 point: 4,250.1, 4,074.6 and 4,366.9 rows (musique, hotpotqa, 2wiki);
- U2y's WMS, cheapest 0.96 point on C: 4,156.2, 3,937.4 and 4,009.5.

Pick O spends 5,522.2, 713.1 and 1,070.5. Its constructions differ by corpus: the 13 September knee rule, which reads
gold, gave musique rrf200+FULL:h2_c25 and hotpotqa and 2wiki rrf50+STRUCT:h1_c25, and ADA f16 then scales each corpus's
own fill.

**What the filed arrays show** (means over the s1sel questions; counts only):

| | musique | hotpotqa | 2wiki |
|---|---:|---:|---:|
| questions | 1,534 | 1,508 | 1,496 |
| mean \|B\|, rrf50 / rrf200 / dense200 | 50 / 200 / 201.0 | 50 / 200 / 201.1 | 50 / 200 / 201.6 |
| mean \|I\|, rrf50+FULL h1 / h2 / h3 | 231.7 / 1,954.2 / 2,023.3 | 206.5 / 1,949.6 / 2,026.1 | 279.0 / 2,027.0 / 2,032.4 |
| mean \|I\|, rrf200+FULL h1 / h2 / h3 | 367.6 / 2,067.2 / 2,136.0 | 345.8 / 2,074.3 / 2,150.6 | 421.9 / 2,163.5 / 2,168.9 |
| mean order length, rrf50+FULL:h2, kind_t, kt_cos | 9,025.4 | 8,626.9 | 9,164.4 |
| mean `n_n1`, rrf50 | 2,609.6 | 1,366.3 | 1,861.7 |
| mean `n_n1`, rrf200 | 7,472.6 | 4,884.4 | 6,864.3 |
| mean `n_n1`, dense200 | 6,855.4 | 4,470.8 | 6,658.5 |
| `n_n1` rrf50, 10th / 50th / 90th percentile | 1,013 / 1,871 / 4,699 | 641 / 1,050 / 1,841 | 1,176 / 1,802 / 2,349 |
| own construction (knee rule), mean \|I\| | rrf200+FULL:h2_c25, 2,067.2 | rrf50+STRUCT:h1_c25, 119.6 | rrf50+STRUCT:h1_c25, 162.2 |

**A cell chosen from its size cannot separate the corpora.** The universal constructions' sizes (|B|, |I|, order
lengths) are alike on the three, so a rule that picks a construction from them picks the same one for all three. That
is the universal case U2s to U2y already cover. Pick O's per-corpus constructions come from gold, not from sizes, and
a size threshold placed between the cells would be fitted to three corpora. So this file does not test that literal
reading of U2y's wording.

**The one size that differs is `n_n1`**, the number of nodes one link from the base's B, which the build's graph step
computes from the graph and B alone. Musique's is 1.9 × (rrf50), 1.5 × (rrf200) and 1.5 × (dense200) hotpotqa's, and
1.4 ×, 1.1 × and 1.0 × 2wiki's. It also varies more between questions than between corpora.

**The form tested:** depth proportional to the question's own neighbourhood, per question and per corpus. The question
is whether one construction with one depth rule holds pick O's level inside the cap on all three datasets.

**Expectation, stated before the run:** the corpora's `n_n1` spread (at most 1.9 ×) is far below the 9 × spread in need,
so the form is more likely to end at UNIFIED_AT_A_ROW_PRICE, or not at all, than inside the cap.

## 2. The rule

A pool is U2s's: the base B, then the first k_q nodes of one build order outside B, then P1h's cut at τ. Only k_q is
new. For a question with n1_q = the size of its one-link neighbourhood of B (U2s's `n_n1`, in the column of the
construction's base):

- **NBRQ(λ, k0):** k_q = max(k0, round(λ × n1_q)).
- **NBRC(λ):** k_q = round(λ × the mean of n1 over the corpus's questions), the same for every question of the corpus.
  This is a depth picked for each corpus from its own pool size.

**The grid:**
- λ in {0.25, 0.35, 0.5, 0.7, 1, 1.4, 2, 2.8, 4, 5.6, 8}, steps of about √2;
- k0 in {0, 50, 300} for NBRQ;
- so 33 NBRQ and 11 NBRC budgets, 44 in all;
- the 12 universal constructions × 3 recipes × 14 orders × 44 budgets × 10 τ give **221,760 new candidates**, beside
  U2s's 1,103,760, so 1,325,520 in all.

Candidates are named `<construction>/<recipe>/<order>:nbq_l<λ>_k<k0>@<τ>` and `...:nbc_l<λ>@<τ>`.

**No gold and no per-dataset constant:** λ and k0 are the same on every corpus, and the frontier picks among the new
candidates exactly as among U2s's. Rounding is numpy's `rint`.

**The construction stays universal.** The frontier picks one cell, as in U2s to U2y. What differs by corpus (NBRC) or by
question (NBRQ) is the depth. This departs from U2y's wording, "picks a construction for each corpus", for the reason in
section 1; the verdict below judges whether the departure delivers.

## 3. What stage A reads and checks

**Inputs:** U2s's filed stats for the three (`outputs/u2s/stats/<D>__s1sel.json` and `.npz`, each checked against its
sha256), P1f's weights, and U2w's `outputs/u2w/choice.json` (its ceilings, incumbent and filed frontiers).

**The choice is U2w's section 3 unchanged,** on the 1,325,520 candidates:
- levels 0.950 to 1.000 in steps of 0.005;
- on dataset d, the target at level t is min(t, ceiling_d − 0.005), with U2w's filed ceilings (musique 0.9811, hotpotqa
  0.9881, 2wiki 0.996), whether or not a new candidate exceeds them (recorded beside);
- cost is P1f's question-weighted mean rows over musique 7,018, hotpotqa 13,335 and 2wiki 18,504;
- the cap is 1,953.5 rows; ties go to the earlier candidate, U2s's first;
- frontier C is the kind_t candidates, E the rrf5t and both candidates, O pick O's own construction.

**Checks** (a miss stops the run and files nothing):
- U2w's: pick O's filed point (ALL and rows on each dataset) and U2s's 11 filed frontier C points reproduce;
- U2s's candidates alone give U2w's filed frontiers C and E, every point's pick, level and cost;
- pick O's level and cost are U2w's filed 0.96 and 1,751.9;
- U2s's ceilings are U2w's filed ones, and P1f's weights are U2s's filed ones;
- the new tables equal U2e's `evaluate` (the path U2s's `held_of` uses) on 24 random new candidates per dataset.

## 4. The verdict and what follows (fixed now)

**What must be achieved:** pick O's level, 0.96, on all three datasets within the 1,953.5-row cap, with U2w's `decide`
unchanged, so that UNIFIED_RAISED, UNIFIED_HOLDS or UNIFIED_AT_A_PRICE (or KIND_FREE_NO_PRICE or _AT_A_PRICE) carries a
point, and UNIFIED_AT_A_ROW_PRICE (or KIND_FREE_AT_A_ROW_PRICE) carries the cheapest point at pick O's level if it costs
at most 2 × pick O's 1,751.9, that is 3,503.8 rows. The carry order is U2w's: E, then C, then the cheaper row-priced
point (ties to C), else nothing.

Beside, deciding nothing:
- every level's point on C and E from 0.80 to 1.00, past the cap too, with each dataset's ALL and rows and the point's
  form (FIX, ADA, MASS, NBRQ or NBRC). This gives the level reached within the cap;
- the frontiers over the new candidates alone;
- the cheapest point at the incumbent's level for each of the 44 new budgets;
- per dataset, the fewest rows any new candidate needs at each level (an upper bound on what a per-corpus rule could
  need);
- the ceilings with and without the new candidates;
- the mean `n_n1` by base and the mean |I| by construction;
- the carried pick (or, if none, the cheapest new point at the incumbent's level) read on metaqa, squad and webqsp's
  filed s1sel arrays beside pick O, deciding nothing.

**A pick is carried.** This track does not build pools or queue refits. Stages B and C follow U2w's sections 5 and 6
unchanged: the pools are built by PKO's steps with the construction, the recipe, the order, the depth rule and τ
replaced; their coverage is measured on s1eval and B1; zg6r and zrckpn are refitted on them, with the same LOSS rule for
admission. The build's entry point and its checks go in a dated section of this file before it runs. Its checks include
that the carried pick's rows and held golds reproduce this stage's tables on s1sel, question for question. The main
session decides the timing (D1).

**Nothing is carried.** Pick O stays the pools of record. This closes depth read from the pool's own size (`n_n1`),
per question and per corpus. The next form is **U2aa**, in its own file: one construction, with the depth multiplier of
section 2's rule set for each corpus by one procedure on a small labelled dev sample of that corpus, judged by this
file's verdict table. It reads gold from the dev sample, as the knee rule that made pick O's constructions does, so
whether it counts as unified is the user's decision (D3). The only gold-free information left that is not a pool size
is the question's own text (its hop structure); it is not in U2s's arrays, so it needs a host pass, and it is parked.

A verdict closes only the form it tests (docs/SCREENS.md, section 3b).

**D1, the timing (default first).** As U2x's D1:
- stage B's build goes into the host queue ahead of the gated fits as soon as a pick is carried;
- if stage C cannot be read before the freeze (00:00 IST, 7 November), pick O stays for the models of record, and the
  unified rule's coverage is reported beside;
- the user decides at the freeze.

**D2, the row price (default first): keep U2w's declared 2 ×,** for U2x's reasons (its section 5). Build space scales
with rows: PKO's 73 GB × (the pick's rows ÷ pick O's) against about 91 GB free, so a row-priced pick needs clean-up
first, and refits on it would not finish before the freeze. The user may instead accept U2w's C point at 0.96 (4,215.6
rows). That is the user's decision, not this file's.

**D3, U2aa (default first):** declare it and run it on the laptop the same day, if this stage carries nothing. The user
may decline it and keep pick O.

## 5. Running

```
python outputs/mp_unified/zu2z.py --selftest
python outputs/mp_unified/zu2z.py dev
python outputs/mp_unified/zu2z.py choose --cache <dir>
```

**Where it runs:** on the laptop, one core, about 2 GB (estimates). The new tables should take about 7 minutes (the
development run took 11 seconds for one construction of musique). U2s's tables are rebuilt as U2w's choice did, about 25
to 30 minutes, and cached under `--cache` keyed by each stats file's sha256. `choose` writes `outputs/u2z/choice.json` and `choice.md`,
committed with a dated results section in this file.

**Amendments:** a change to `zu2z.py` after the commit that declares this file and before the run's numbers is recorded
here as an amendment, with the rule unchanged. None: the run's `zu2z.py` is the one declared in 361f7a8.

## 6. Results of stage A, 4 November 2026 (about 09:40 IST)

Filed in `outputs/u2z/choice.json` and `choice.md`. `zu2z.py choose` ran on the laptop in 2,145 s, one core.

**Verdicts: UNIFIED_AT_A_ROW_PRICE and KIND_FREE_AT_A_ROW_PRICE. E_ROW is carried:**
`rrf50+none/both/kt_cos:nbc_l2.8@0.9`.
- It reaches level 0.96 at 3,274.7 rows, 1.869 × pick O's 1,751.9, inside the 2 × bound (3,503.8) and outside the cap
  (1,953.5). U2w's universal point at 0.96 was 4,215.6 rows (2.41 ×), so this is 22% fewer rows.
- No point on C or E at 0.95 or above is within the cap: the cheapest at 0.95 is 2,508.0 rows (1.43 × pick O's). The
  level frontier C reaches within the cap (U2s's candidates and the new ones together) is 0.925, at 1,876.9 rows; pick
  O's own construction reaches 0.96 at 1,751.9.
- C's cheapest point at 0.96 is `rrf50+none/kind_t/kt_cos+n2_10:nbc_l2.8@0.9`, at 3,275.4 rows. E's is 0.7 rows
  cheaper, so section 4's carry order takes E.
- The ceilings do not move: musique 0.9811, hotpotqa 0.9881, 2wiki 0.996.

**ALL (rows) on s1sel:**

| pools | cost | musique | hotpotqa | 2wiki |
|---|---:|---|---|---|
| pick O `own/kind_t/kt_cos:ada_f16_k0@0.85` | 1,751.9 | 0.9668 (5,522.2) | 0.9602 (713.1) | 0.9699 (1,070.5) |
| carried `rrf50+none/both/kt_cos:nbc_l2.8@0.9` | 3,274.7 | 0.9615 (4,506.4) | 0.9775 (2,533.0) | 0.9860 (3,342.0) |

**What the carried pick is:**
- the construction is `rrf50+none`: B is RRF's top 50 plus the seeds and nothing is expanded in the graph (|I| = |B|);
- the recipe is `both` (RRF of ROUTEst's and RRF5T's orders), the order `kt_cos`, the cut τ = 0.9;
- the depth is NBRC(2.8), k_q = round(2.8 × the corpus's mean `n_n1` of rrf50): 7,307 on musique (mean 2,609.62),
  3,826 on hotpotqa (1,366.35) and 5,213 on 2wiki (1,861.70). 23.8% of musique's questions have an order shorter than
  7,307 and take all of it; none do on the other two.

**Checks passed (all five of section 3):** pick O's filed point and U2s's 11 filed frontier C points reproduce; U2s's
candidates alone give U2w's filed frontiers C and E; pick O's level and cost are U2w's; U2s's ceilings and P1f's weights
are U2w's and U2s's; and the new tables equal U2e's `evaluate` on 72 random new candidates, 24 per dataset.

**Beside, deciding nothing:**
- *Per question does not beat per corpus.* The cheapest NBRQ point at 0.96 costs 3,989.5 rows (λ = 4); NBRC's is 3,274.7
  (λ = 2.8). Only NBRC meets 0.96 below 3,700 rows. NBRC's cost at λ = 2 and λ = 4 is 3,332.8 and 3,444.2, so the optimum
  is flat, not a knife edge.
- *Where the price comes from.* The fewest rows any new candidate needs at 0.96 on each dataset alone are 4,265.2
  (musique), 479.0 (hotpotqa) and 477.5 (2wiki). The carried pick spends 4,506.4, 2,533.0 and 3,342.0: close to musique's
  need, and 5.3 × and 7.0 × the other two's. `n_n1` spreads 1.9 × across the corpora where need spreads 9 ×, as section 1
  expected.
- *The other corpora* (filed s1sel arrays; the depth follows each corpus's mean `n_n1` of 540.3, 1,928.1 and 2,234.1):
  - metaqa: ALL 0.8651 at 1,108.0 rows, against pick O's 0.9653 at 4,369.9;
  - squad: 0.9993 at 3,448.7 rows, against pick O's 0.9793 at 50.0;
  - webqsp: 0.9188 at 3,937.1 rows, against pick O's 0.9069 at 4,803.4.

  So a depth set from the mean `n_n1` alone does not carry to the other three: it falls short on metaqa (0.10 below
  pick O's ALL on a quarter of its rows), pays 69 × pick O's rows on squad for +0.020, and is above pick O's ALL on
  18% fewer rows on webqsp. They are not in the verdict.
- *The candidates were chosen over 1,325,520 on about 1,500 questions a dataset,* so the carried point's s1sel ALL is
  selected. Stage B's s1eval and B1 coverage is the read that counts.

**What follows (section 4, fixed before the run).** A pick is carried, so stages B and C follow U2w's sections 5 and 6.
This track built no pools and queued nothing. Section 7 is the build's entry point and checks. The row price is within
the declared 2 ×, so D2's default stands. The timing is the user's at D1.

## 7. Stage B's build: entry point and checks (written 4 November 2026, not run)

PKO's build (`outputs/mp_unified/zpko.py`, docs/PKO_PICK_O_POOLS_UNDER_WQ.md) is fixed to pick O: `pick_plan` parses the
name with U2m's `MB.parse`, `open_pko` and `rule_batch` read the own construction's base and I, and `kept_counts` takes
ADA's k_q from the fill. The carried pick needs a new module, `outputs/mp_unified/zu2zb.py`, that imports `zpko` and
U2s's functions unchanged and replaces these five things. Its commands are `zpko.py`'s (`plan`, `check`, `pools`,
`merge`, `chain`, `gate`).

1. **The plan** (`pick_plan`, `u2s_verdict`, `u2s_record`). `MB.parse` knows only U2e's 219 budgets (`S.budgets()`), so
   the name is parsed with this file's `zu2z.nbr_budgets`: cell `rrf50+none` (U2s's cell 0, base 0, base pool
   `equal_rrf_budget_50`, which is `M.BASES[0]`), recipe `both` (r = 2), order `kt_cos`, budget `nbc_l2.8` (NBRC, λ = 2.8),
   τ = 0.9. The check that U2s's verdict carries pick O becomes a check that `outputs/u2z/choice.json` carries this
   name as E_ROW, was written by the `zu2z.py` on disk (its `script_sha256`), and names the U2s records the build reads
   (its `stats_sha256`).
2. **The construction** (`open_pko`, `rule_batch`). The base pool is `M.BASES[0]`'s for all three datasets, B is that
   base plus the seeds, and I = B (nothing is expanded). The own-cell checks of `open_pko` are made for cell 0 and
   base 0. The check in `rule_batch` that B is the carve's step-4d B and lies inside the own construction's I is
   rewritten for this construction: B against U2s's filed |B| of base 0, and I = B.
3. **The recipe.** The outside list is `F.outside(D.rrf([p1["ROUTEst"], o1["RRF5T"]], constant), B)`, as `zu2s.py`
   forms its `rec[2]`, in place of `kt`'s.
4. **The depth** (`kept_counts`). k_q = round(2.8 × m), with m the mean of n1's size over the carve's questions
   (`zu2z.nbr_k`), in place of ADA's fill-based k_q; then k = min(|order|, k_q), keep = `P1H.keep_counts(|B|, k, 0.9)`,
   and the pool is B plus the order's first keep nodes. `rule_batch` already computes n1 per question
   (`D5B.n1_table`), but m needs the whole carve, so the build first runs a pass over the carve that records n1's size
   for every question (B, the steer and the one-link table; no walk), and the shards read k_q from it.
5. **The filed fields** (`u2s_filed`, `filed_rows_q`). They read U2s's arrays at cell 0, base 0 and recipe 2 in place of
   the own cell, and take k_q and the kept rows from the new rule.

**D4, the mean m on the other carves (default first).** Section 2's m is the mean over the corpus's questions, and stage
A took it over s1sel's. For the other carves:
- the default is the mean over the carve's own questions, so no per-dataset number is stored, and on s1sel it is stage
  A's value (2,609.62, 1,366.35 and 1,861.70);
- the alternative is s1sel's mean for every carve of that dataset, so a dataset's carves share one k_q, at the price of
  three stored numbers.

The two differ by the sampling error of a mean over a carve's questions; on s1sel they are the same.

**The carves** are U2w's section 5: the six fit and s1eval carves PKO built (musique s1fit 4,601 and s1eval 2,417,
hotpotqa 5,930 and 7,405, 2wiki 5,928 and 12,576, which are U2w's 7,018, 13,335 and 18,504 questions), s1sel for the
reproduction below (1,534, 1,508 and 1,496), and B1's three settings.

**Checks before any pool is kept** (PKO's section 5 (a), run inside the pools pass on s1sel of each of the three, U2s's
carve of each; the build never loads U2s's gold arrays):
- question for question against U2s's filed arrays: |B| (= |I|), the fill, the 14 order lengths of recipe `both`, U2e's
  union sizes, n1, r0 and the second link's counts, taken and hubs, the walk-mass counts and their totals, k_q and the
  kept rows (`zpko.differ` on `u2s_fields` and `u2s_filed`);
- the carve's mean n1 on s1sel is 2,609.62, 1,366.35 and 1,861.70, and k_q is 7,307, 3,826 and 5,213;
- the mean of |B| plus keep over the filed arrays is 4,506.4, 2,533.0 and 3,342.0 rows, and 3,274.7 over the three
  (section 6's table);
- the pools file's own invariants (`check_filed`): pointers, B and the expansion disjoint and each unique, count =
  fill_k + x_count, one KT row per pool row.

**The held golds, before any s1eval or fit carve is built.** After the s1sel pools exist, the coverage step (the only
step that reads golds; the build reads none) takes each s1sel question's golds from the carve's labels and counts those
inside its built pool. They equal, question for question, the held counts of U2e's `evaluate` on U2s's filed arrays for
the carried candidate (as `zu2z.eval_one` calls it), and ALL is 0.9615, 0.9775 and 0.9860. A miss stops the chain.

**Host and space (estimates, not measurements).** They scale PKO's table (its section 9) by each dataset's rows ratio at
the pick against pick O, on s1sel: musique 0.816 (4,506.4 ÷ 5,522.2), hotpotqa 3.552 (2,533.0 ÷ 713.1) and 2wiki 3.122
(3,342.0 ÷ 1,070.5).
- Rows: the six fit and s1eval carves hold about 127.2 million rows, against 68.1 million at pick O.
- Disk: about 142 GiB with the looks (53.6 GiB of it looks, freed by `--prune`) and about 89 GiB without, against about
  91 GB free (U2x's figure). The largest carve is 2wiki's s1eval, about 42.5 GiB; musique's s1fit is about 27.7 GiB. So
  the chain builds one carve at a time and ships and frees between runs, as PKO's guard requires. Section 4's D2 scaling
  (73 GB × 1.869 = 136 GB) is the same order. s1sel and B1 are not counted.
- Time: the pools pass takes at most 2.03 s per question (PKO's measure on webqsp, its upper bound): at most 21.9 h for
  the six carves' 38,857 questions on one lane, and 2.6 h more for s1sel's 4,538. The downstream steps (look, cache,
  rmatch, zrc and links, 1,666 µs per row at PKO's laptop rates) take about 58.9 h for the six carves on one lane (31.5 h
  at pick O's rows), before any fit. The pools pass and the coverage can be ready before the freeze (00:00 IST, 7
  November); a refit on these pools is unlikely to be read by then, and D1's default covers that case.
- The runner is PKOH's (`zpkoh.py`), with `u2z-` job names; the host's ledger is read for space before anything is
  queued.

A verdict closes only the form it tests (docs/SCREENS.md, section 3b): this one carries depth read from the corpus's own
`n_n1`, per corpus. Per-question depth from `n_n1` does no better. U2aa (section 4) is not needed unless stage B or C
undoes the pick.
