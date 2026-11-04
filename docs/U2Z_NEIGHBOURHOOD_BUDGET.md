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
here as an amendment, with the rule unchanged. None so far.
