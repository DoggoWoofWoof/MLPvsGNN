# U2y: P1h's cut taken over the walk's mass

Declared 4 November 2026 (about 05:45 IST), before any U2y number. Module: `outputs/mp_unified/zu2y.py`, run on the
host through `outputs/mp_unified/zu2yh.py`. It imports `zu2s` and `zu2w` unchanged, and through them U2m, U2e, P1e,
P1f and P1h. Records: `outputs/u2y/stats/<D>__s1sel.wm.npz` (uncommitted) and `.wm.json`, then
`outputs/u2y/choice.json` and `choice.md`.

Stage A is pools only: nothing trains and nothing is fit. The stats pass runs U2s's walks again, unchanged, and cuts
each question's pools two new ways. The choice reads golds on s1sel, as every pool stage's choice does
(docs/U2S_WALK_MASS_BUDGET.md, section 6).

## 1. Why

**U2x found no unified rule either** (docs/U2X_RRF5T_AGREEMENT_BUDGET.md, section 8; filed in 91d9ef2). Its verdicts
were NOT_UNIFIED and KIND_FREE_TOO_DEAR:
- no universal candidate is within the 1,953.5-row cap at any level, with or without its 302,400 AGREE candidates;
- at pick O's level, 0.96, the cheapest C and E point is still U2w's, at 4,215.6 rows, 2.41 × pick O's 1,751.9.

**Neither budget reads depth.** AGREE's cheapest 0.96 point spends 4,074.6 rows a question on hotpotqa and 4,366.9 on
2wiki, as many as on musique (4,250.1). Pick O spends 713.1 and 1,070.5 there (U2w). So neither one walk's mass (U2s's
MASS) nor RRF5T's agreement (U2x's AGREE), read as a budget, separates the questions that need depth from those that
do not.

**The cut never reads the question.** Every pool so far ends with P1h's cut (docs/P1H_FEWER_ROWS.md): of the pool's
build order, keep the shortest prefix whose rank mass, the sum of 1 / (60 + i) over places i from 1, reaches τ of the
whole pool's. Rank mass depends only on places. So the cut keeps a share of the pool that depends only on its length
and |B|, whatever the question.

**The form tested here** is U2s's section 10's second: P1h's cut with the walk's mass in place of rank mass.
- When a question's walk puts its mass on a few nodes near the head of the order, the cut keeps few rows.
- When the mass spreads down the order, as it should on a deep question, the cut keeps more.
- The budget still bounds the pool, so the cut can only shorten it.

The question is whether one construction cut this way can hold pick O's level on all three datasets within the cap.

## 2. The cuts

**The walks** are U2s's two MASS walks, A1all and KTall (docs/U2S_WALK_MASS_BUDGET.md, section 2), unchanged. Each
returns up to 7,000 nodes with their masses. A node's walk mass is its mass in the walk, and 0 when the walk does not
reach it. The walk's total is the fsum of all its masses.

**The pool** is U2e's (docs/U2E_POOLS_TO_THE_CEILING.md, section 5): B, then the first k_q nodes outside B of one
order, then the cut. Let kk = min(k_q, the order's length). The 12 constructions, 3 recipes, 14 orders and 219 budgets
are U2s's. Only the cut is new.

**WMS(τ, walk, total), the share cut.** It is P1h's `keep_counts` with walk masses in place of rank masses.
- The build order's masses are w0, for B as one block, then the walk masses of the order's first kk nodes.
- w0 is the fsum of B's walk masses ("pool"), or 0 ("exp", the expansion alone).
- The cut keeps the shortest prefix of the expansion whose float64 prefix sum, w0 included, reaches τ of the whole
  pool's: at least 0 and at most kk nodes. When B alone holds τ of the pool's mass, the pool is B.
- τ takes P1h's nine values below 1: 0.99, 0.975, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6 and 0.5. τ = 1 is the uncut pool,
  which U2s's candidates already hold.
- An expansion with no walk mass keeps B only.

**WMT(ε, walk), the threshold cut.** It keeps the shortest prefix of the kk expansion nodes that holds every one of
them whose walk mass is positive and at least ε × the walk's total, and none when there is none. ε is 1e-2, 1e-3,
1e-4, 1e-5 or 1e-6.

Both grids are fixed here, before any number. Nothing is fitted. The walks, the orders and the cuts read no gold;
golds enter only the counts on s1sel.

A pool's rows are |B| plus the nodes the cut keeps. A gold is held when it is in B or among the kept nodes, and ALL
counts the questions whose every gold is held, as in U2s.

## 3. The candidates and the choice

**The candidates:**
- U2s's 1,103,760 universal candidates, unchanged: 12 constructions × 3 recipes × 14 orders × 219 budgets × P1h's 10
  values of τ;
- 3,973,536 WMS candidates: the same 12 × 3 × 14 × 219, × 9 values of τ × 2 walks × 2 totals, named
  `<U2s's name at τ>~<walk>.<total>`, for example `rrf50+FULL:h2_c25/kind_t/kt_cos:ada_f16_k0@0.9~A1all.pool`;
- 1,103,760 WMT candidates: 12 × 3 × 14 × 219 × 5 values of ε × 2 walks, named
  `<construction>/<recipe>/<order>:<budget>@eps<ε>~<walk>`, for example
  `rrf50+FULL:h2_c25/kind_t/kt_cos:ada_f16_k0@eps0.0001~KTall`;
- the own construction's 91,980 candidates, which include pick O, read as in U2w for the incumbent.

Ties go to the earlier candidate: U2s's first, in U2s's order, then WMS's, then WMT's, in the layouts above.

**The choice is U2w's sections 3 and 4, unchanged**, through zu2w's own functions, as in U2x:
- the cost is P1f's question-weighted mean rows over the three datasets;
- t runs from 0.950 to 1.000 in steps of 0.005;
- the target on dataset d is min(t, C_d − 0.005), where C_d is the best ALL any universal candidate reaches on d,
  U2s's, WMS's or WMT's;
- the cap is 1,953.5 rows;
- frontier C runs over the kind_t candidates, and E over the rrf5t and both candidates;
- the verdict table, E's verdicts and the carry order are `zu2w.decide`'s, with the row price of 2 × the incumbent's
  cost.

## 4. Checks

Any failure stops the run, and nothing is written.

**The stats pass** runs `zu2s.stats_stage` unchanged, with seven in-process patches that are restored afterwards, also
on an exception. They record each question's calls without changing them:
1. P1e's `kind` opens a question;
2. U2m's `fill_of` records each (construction, recipe)'s fill;
3. U2e's `question_orders` records its 14 orders;
4. U2e's `places` records each gold's place in each order;
5. `zu2s.mass_counts` records each base's B and the two MASS walks' nodes and masses;
6. `zu2s.mq_ok` closes the question with its m_q;
7. `zu2s.OUT` is set to `outputs/u2y`, so nothing under `outputs/u2s` is written.

So every check of U2s's section 7 runs again. On top of those, for each question:
- the calls come in the pass's order: P1e's kind; for each of the 12 or 13 constructions and each of the 3 recipes, a
  fill, 14 orders and 14 places; 6 mass counts, with the same walk objects for every base and one B per base; then one
  m_q;
- each walk's nodes are distinct, and its masses finite and non-negative;
- no order holds a node of its B;
- the arrays the capture rebuilds (fill, lens, pos, m_q, the 12 constructions' |B| and the pointers) equal the ones
  the pass writes, shard by shard;
- every sum fits int32.

**The merge** checks that each part is this script's, matches its file's sha256 and is part i of k; that the parts
hold the merged stats' question count; and that the merged rerun's arrays equal U2s's filed arrays, array for array.

**The choice** checks, before any number is written:
- U2w's filed choice is zu2w.py's, and carried nothing;
- U2s's filed stats are the ones U2w read (sha256);
- the rerun's arrays equal U2s's filed arrays, and the walk-mass sums are on that rerun (sha256) and its questions;
- the control, P1h's rank-mass cut rebuilt from the captured places, gives U2s's table at all 1,103,760 universal
  candidates on each of the three datasets;
- pick O's filed point and U2w's filed frontier C points are reproduced from the same tables;
- P1f's weights are U2w's.

**Development runs** (`dev D --limit 5`, on the host for each of the three) must pass every check of the stats pass
before the full run. They also check the control against U2s's table on their 5 questions, and 2,000 drawn WMS and WMT
candidates against a plain-Python reference of section 2's cuts. They write nothing.

**The self-test** (`python outputs/mp_unified/zu2y.py --selftest`) passed on the laptop before this commit. It checks:
- the share cut equals P1h's `keep_counts` bit for bit when the masses are rank masses (300 cases × 9 values of τ × 7
  pool sizes);
- both cuts equal the plain-Python reference on made-up masses with zeros and ties;
- the walk lookup and totals, and the refusal of repeated nodes and of negative or non-finite masses;
- the capture on made-up questions in the pass's call order, with 12 and 13 constructions: the rebuilt arrays equal
  the pass's, the control equals U2s's table, and 300 drawn candidates equal the reference;
- the refusal of calls out of order and of a missing question;
- the patches reach the functions the pass calls and are put back, also on an exception;
- the sums of parts add to the whole, and an int32 overflow is refused;
- the names decode, and the verdict is `zu2w.decide`'s;
- stats writes each part, merge adds them and refuses a changed or missing part, and the merged record reads back.

`zu2yh.py --check` passed on the laptop before this commit.

## 5. The verdict and what follows (fixed now)

The verdict and what is carried follow U2w's section 4. Beside, deciding nothing:
- every level's point on C and E, past the cap too, with each dataset's ALL and rows, and each point's form (FIX, ADA,
  MASS, WMS or WMT);
- the frontiers over the WMS and WMT candidates alone;
- the cheapest point of each form, of each walk and of each total at the incumbent's level, with each dataset's rows.
  This answers the question of section 1: a cut that tracked depth would spend fewer rows on hotpotqa and 2wiki than
  on musique;
- the ceilings with and without the walk-mass candidates;
- P1e's kinds, counted per dataset;
- the mean share of each walk's mass outside each base's B, per dataset.

**A pick is carried.** Stages B and C follow U2w's sections 5 and 6 unchanged, as in U2x's section 5:
- the pools are built by PKO's steps with the construction and the cut replaced;
- their coverage is measured on s1eval and B1;
- zg6r and zrckpn are refitted on them, with the same LOSS rule for admission.

The build's entry point and its checks go in a dated section of this file before it runs.

**Nothing is carried.** Pick O stays the pools of record. This cut is the last form U2s's section 10 names. The next
form is one rule that picks a construction for each corpus before any gold is read, from the corpus's own pool sizes
in U2s's filed arrays, declared in its own file.

**Routing by P1e's kind is not the next form.** P1e's entity share is 0.0 on all three s1sel carves
(`outputs/p1e/pools_D.json`), so a kind route sends every question of the three the same way.

A verdict closes only the form it tests (docs/SCREENS.md, section 3b).

**D1, the timing (default first).** As U2x's D1:
- stage B's build goes into the host queue ahead of the gated fits as soon as a pick is carried;
- if stage C cannot be read before the freeze (00:00 IST, 7 November), pick O stays for the models of record, and the
  unified rule's coverage is reported beside;
- the user decides at the freeze.

**D2, the row price (default first): U2w's declared 2 ×**, for U2x's reasons (its section 5).

## 6. Running

```
python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py --selftest
python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py dev D --limit 5 --threads 2      (D = 2wiki, hotpotqa, musique: the gate)
python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py stats D --shard I 4 --threads 2  (I = 0, 1, 2, 3)
python outputs/host_ops/pylib_run.py outputs/mp_unified/zu2yh.py merge D 4
python outputs/mp_unified/zu2y.py choose                                                           (laptop, after fetching outputs/u2y/stats)
```

**The host runner.** `zu2yh.py` is U2x's amendment A1 runner (docs/U2X_RRF5T_AGREEMENT_BUDGET.md, section 7) in front
of zu2y. `zu2xh.mirror()` puts PKOH's package mirror in place and checks that the patch reaches the module
`zu2s.stats_stage` calls, then `zu2y.main` runs on the same arguments.

**Where it runs.** On the host CPU, as U2x's pass.
- Each shard asks for 2 CPUs and 8 GB (2wiki), 6 GB (hotpotqa) or 3 GB (musique). U2x's shards asked for 7, 5 and 2
  GB and finished. U2y's sums and per-question arrays add about 0.2 GB (int64 over 6.18 million candidates).
- The development runs ask for 2 CPUs and 12 GB, the self-test 1 CPU and 4 GB, each merge 1 CPU and 8 GB.
- The inputs are U2x's (U2s's inputs and its filed stats for the three datasets), with zu2y.py, zu2yh.py and this
  file in place of zu2x.py and its file.

**ETA (estimate).**
- U2x's longest shards took 5,108 s (2wiki), 4,522 s (hotpotqa) and 3,828 s (musique), with three a dataset
  (`seconds_longest_part` in `outputs/u2x/stats/<D>__s1sel.json`). With four a dataset, U2y's should take about 1 to
  1.5 hours each, after the self-test and the development runs, then the merges.
- Then the choice on the laptop: U2x's took 1,959 s over 1.41 million candidates. U2y's has 6.18 million, so roughly
  1 to 2 hours.

**Records.** The stats pass writes its rerun of U2s's stats and the walk-mass sums under `outputs/u2y/stats`
(uncommitted, as U2x's). The choice writes `outputs/u2y/choice.json` and `choice.md`, committed with a dated results
section in this file.
