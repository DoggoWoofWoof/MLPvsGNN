# WQP: U2s's pool pick, chosen again under WQ

Declared 28 October 2026, before any WQP run and before any number on webqsp's select carve. This is WQ section 9's
third item ([WQ_WEBQSP_TRAINS.md](WQ_WEBQSP_TRAINS.md)): "U2s's pool pick, chosen again under WQ: metaqa leaves the
training sets, and webqsp's select carve is the selection carve". U2s ([U2S_WALK_MASS_BUDGET.md](U2S_WALK_MASS_BUDGET.md),
section 11) carried pick O, `own/kind_t/kt_cos:ada_f16_k0@0.85` (ADA), chosen on the lean protocol's five training sets'
s1sel carves, metaqa among them, with webqsp's s1eval read beside. Code: `outputs/mp_unified/zwqp.py`.

WQP reads no fit and trains nothing, so it does not wait for WQ's or WQK's BASE_READY. Pools only: no construction,
recipe, order, budget or cut reads a gold.

## 1. What is computed

**webqsp's select carve**: M3B's select carve of webqsp (`outputs/m3b/carves.json`), 310 questions, of which 305 have a
gold in the graph. The 305 ids hash to `d3c3cc4ba8d4d03983e3713500b585b1fabd0e597a8e317cf267c286591d4a96`, the value
the carve's look recorded (`outputs/mp_unified/look/webqsp/select/record.json`). Its questions are not s1eval's
(`outputs/m3b/population_overlap.json`: select 305, fit 1,202, eval 1,503).

It goes through the chain that filed webqsp's s1eval stats for U2s, one stage at a time, in this order:

| # | stage | module (committed, imported unchanged) | writes under |
|---|---|---|---|
| 1 | step 4d | scripts/step4d_union_pools.py | outputs/wqp/step4d/pools |
| 2 | step 4e | scripts/step4e_paper_pools.py | outputs/wqp/step4e/pools |
| 3 | P1d | scripts/p1d_free_pool.py | outputs/wqp/p1d |
| 4 | P1e | scripts/p1e_anchor_kind.py | outputs/wqp/p1e |
| 5 | P1f | scripts/p1f_pool_size.py | outputs/wqp/p1f |
| 6 | P1g | scripts/p1g_pool_beam.py | outputs/wqp/p1g |
| 7 | U2c, today's graph (a KB has no chosen graph) | outputs/mp_unified/zu2c.py | outputs/wqp/u2c |
| 8 | U2e | outputs/mp_unified/zu2e.py | outputs/wqp/u2e |
| 9 | U2m (c) | outputs/mp_unified/zu2m.py | outputs/wqp/u2m |
| 10 | U2m (b) | outputs/mp_unified/zu2m_b.py | outputs/wqp/u2m/b |
| 11 | U2m (a) | outputs/mp_unified/zu2m_a.py | outputs/wqp/u2m/a |
| 12 | U2s | outputs/mp_unified/zu2s.py | outputs/wqp/u2s |

**The patches**, in this process only, in the manner of WQ's `with_fit`:
- **The carve.**
  - Step 4's opened carves gain webqsp's `mselect` entry: select, 305, the digest above. carves.json on disk is not
    touched, so its pinned sha256 holds.
  - Step 4c's and D5's `carves_of` give ("select",) for webqsp. Every U2 stage reads its carve through D5's.
  - P1d, P1e and P1f ask step 4e's `run_carves` for {"s1eval"} on webqsp in their own code. The patched `run_carves`
    passes {"select"} for webqsp instead, and every other dataset's request unchanged.
- **The roots.** Each stage's output root moves to the table's, P1f's own P1e root with it. Unchanged and read only:
  step 4e's choice, P1c2's manifests, (a)'s filed EM types and U2p's verdict.
- **The filed verdicts.** (c) and (e)'s, (b)'s and (a)'s verdict files are copied under outputs/wqp byte for byte, so
  every sha256 a stage checks or records is the filed one. They are copies, never links: a write through a link would
  change the filed file. A copy already there must be the same bytes; nothing is overwritten.
- **Three checks re-pointed.** (b), (a) and U2s each check that the stats they read on webqsp are the ones the filed
  verdict read. On select no filed verdict read them. Those three checks instead require the file WQP's ledger
  (section 5) filed for the stage before. That is (c)'s for (b), (b)'s for (a) and (a)'s for U2s. Only webqsp's entry is
  re-pointed.

**What stays as filed:**
- U2m (c) and (e)'s carried frontier, (b)'s G, (a)'s pick, U2p's beta and the carried walk.
- WQ section 9 re-chooses U2s's pick only. Re-choosing the upstream picks would need each upstream verdict run again
  under WQ. Section 7 says when that follows.
- No step-1 cache exists for select. Step 4d records none, and no stage of this chain reads one.

## 2. The check (before select's stages)

Stages 1 to 5 run again on webqsp's s1eval carve through WQP's patches, with WQP's thread setting, into
`outputs/wqp/check`. The carve switches and the re-pointed checks, which only select needs, are left out. The output is compared with the filed arrays of step 4d, step 4e, P1d, P1e and P1f on that carve.
- **REPRODUCED** needs every key, dtype, shape and byte of all five (WQ section 4's rule). File sha256s and the
  manifests' entries are reported beside.
- **NOT_REPRODUCED** stops WQP before select's stages. The failing stage is reported in section 9 and fixed in an
  amendment.

Record: `outputs/wqp/check.json`.

## 3. The choice (U2s section 6's rule, under WQ)

**The selection set S**
- squad, musique, hotpotqa and 2wiki on their s1sel carves: U2s's filed stats, each checked to be the file U2s's verdict
  read;
- webqsp on its select carve: WQP's stats.
- metaqa leaves S. Its s1sel is read beside.

**The cost** is P1f's question-weighted mean rows over S, with P1f's weights, fit plus s1eval questions:
- squad 17,729, musique 7,018, hotpotqa 13,335, 2wiki 18,504, from P1f's `weights()`;
- webqsp 2,705: WQ section 2's fit carve, 1,202 questions (zwq's `FIT_N`), plus its s1eval's 1,503 (P1c2's manifest).

**Levels and targets**
- Levels run from 0.950 to 1.000 in steps of 0.005.
- The targets are min(t, C_d − 0.005) on each d in S. C_d is the best ALL any universal candidate reaches on d.

**The cap** is restated under WQ, with the same definition:
- It is U2c's chosen pool's cost, `g:f3+cut_cos_7000@0.9`.
- The chosen pool's rows come from U2c's filed chosen-graph stats on the four s1sel carves, and from WQP's U2c stats on
  webqsp/select (today's graph, a KB's only graph).
- They are weighted as above. Under U2s's set and weights this cost is 1,953.5. The verdict recomputes it and stops if
  it is not.

**The incumbent** is U2s's carried pick O, read on S: its level and cost under the targets, weights and cap above.
- It stays the bar wherever its cost falls.
- The record states whether it is within WQ's cap.
- If it meets no level on S, its level counts as 0.940, two steps below the lowest level, so every frontier pick
  with a level beats it.
- The verdict stops if U2s's filed verdict does not carry pick O.

**The frontiers** are U2s's three, over the same 219 budgets:
- **C**: universal kind_t;
- **E**: universal rrf5t and both;
- **O**: the own construction's kind_t.

Each frontier's pick is its point at the highest level within WQ's cap.

**The verdicts** are U2s's, against the incumbent's level L and cost on S:

| verdict (c) | when |
| --- | --- |
| UNIFIED_RAISED | L_C > L |
| UNIFIED_HOLDS | L_C = L |
| UNIFIED_AT_A_PRICE | L_C = L − 0.005 |
| NOT_UNIFIED | otherwise |

| verdict (e) | when |
| --- | --- |
| KIND_FREE_NO_PRICE | L_E ≥ L − 0.005, and C has no pick or L_E ≥ L_C |
| KIND_FREE_AT_A_PRICE | L_E = L_C − 0.005 and L_E ≥ L − 0.005 |
| KIND_FREE_TOO_DEAR | otherwise |

| verdict (o) | when |
| --- | --- |
| OWN_RAISED | L_O > L |
| OWN_CHEAPER | L_O = L and pick O costs less than the incumbent |
| NO_GAIN | otherwise |

**What is carried**, the first that applies:
1. pick E, when (e) is not KIND_FREE_TOO_DEAR;
2. pick C, when (c) is not NOT_UNIFIED;
3. pick O, when (o) is not NO_GAIN;
4. otherwise U2s's pick stays.

## 4. Reported beside, deciding nothing

- **U2s's verdict, reproduced.** The same tables, under U2s's own set (the five s1sel carves), P1f's weights, its cap
  and U2p's incumbent, must give U2s's filed verdicts, picks, levels, costs and ceilings. Otherwise the verdict stops:
  a WQP table that cannot reproduce U2s's verdict decides nothing. `zwqp.py reproduce` runs the same check on the six
  filed tables before any select stage (section 8).
- **Each pick read off S.** Every pick, the carried one and the incumbent on metaqa's s1sel and webqsp's s1eval, at
  ALL and rows.
- **Leave one out.** WQ trains on two splits, each without one wiki set (WQ-musique without musique, WQ-hotpotqa
  without hotpotqa). For each split, the three frontiers' picks are computed again with that dataset left out of S, the
  ceilings and the cap restated on the smaller S. S without a split's zero-shot wiki set is that split's training
  sets (WQ section 2). This shows whether a split's zero-shot set moved the pick.
- **Per dataset:** each d in S's ceiling and fewest rows at each level, as U2s reports it.
- **The price:** the carried pick against the incumbent, ALL and rows, on S, metaqa and webqsp's s1eval.

## 5. Checks

**After every stage**, in the stage's own process, with the stage's own loader where it has one:
- the stage filed webqsp/select;
- 305 questions, carve digest `d3c3cc4b…` where the record names one, freeze `58958f33…` where it names one;
- the record's file is its sha256, and every check holds (rows_served_verified is set only by `--limit` runs, as in
  every filed stats record); a `--limit` record is refused;
- the stage's own loader, after the record checks, where it has one (P1f, P1g, U2c, U2e, U2m (c), (b), (a), U2s; step
  4d, step 4e, P1d and P1e have none);
- nothing under the chain's filed roots changed during the stage: outputs/step1, step4b, step4d, step4e, p1c2, p1d,
  p1e, p1f, p1g, p1g2, u2b, u2c, u2e, u2m, u2p, u2q, u2s, m3b, d5 and mp_unified/look. Every file's size and mtime is
  compared, before against after, `__pycache__` left out. Other roots are written by other work running at the same
  time (WQK's take, for one), so they are left out.

The stage, its record's sha256, its file's sha256, its seconds and the result go into `outputs/wqp/ledger.json`. A stage
that fails a check stops the chain. Nothing later runs.

**The verdict** requires:
- the ledger's twelve stages, each OK and its record and arrays unchanged, and the check REPRODUCED;
- U2s's verdict to be the committed zu2s.py's and to carry pick O; U2c's to be the committed zu2c.py's, choosing
  `g:f3+cut_cos_7000@0.9`; U2c's chosen pool to cost 1,953.5 under U2s's set, U2p's cap;
- the four passage stats, metaqa's and webqsp's s1eval stats to be the files U2s's verdict read, and select's to be the
  files WQP's ledger filed (U2s's and U2c's);
- every stats file to be on the carried walk U2p's verdict carries (pick, points, G, beta, U2p's verdict sha256), with
  U2m's questions and golds.

## 6. Where it runs, and the records

- **Where.** The laptop, CPU only, one stage at a time (RAM).
- **Threads.** numba runs 6 threads. Every parallel loop in the chain runs one question per thread at a time
  (step4_pool_reach's ppr_batch says so), so the arrays do not depend on the count. The check (section 2) runs the same
  setting and tests that. BLAS keeps each module's own 2, set at zwqp's top before numpy loads, as in the filed runs.

**Commands:**

    python outputs/mp_unified/zwqp.py chain [--threads 6] [--retry]
    python outputs/mp_unified/zwqp.py stage NAME [--threads 6] [--check]
    python outputs/mp_unified/zwqp.py compare-check
    python outputs/mp_unified/zwqp.py reproduce
    python outputs/mp_unified/zwqp.py verdict
    python outputs/mp_unified/zwqp.py --selftest

- **chain** runs the check's five stages, compare-check, the twelve select stages and then the verdict, each in its
  own process, one after another. A stage the ledger holds as OK, its record and arrays unchanged, is not run again.
  A FAILED stage stops the chain; `--retry` runs it again, and the ledger keeps the earlier entry.
- **stage** runs one stage in this process under the patches, then section 5's checks, and files its ledger entry.
- **compare-check** writes `outputs/wqp/check.json` (section 2). It exits 0 on REPRODUCED and 2 otherwise.
- **reproduce** is section 4's reproduction on the six filed tables. It writes nothing.
- **verdict** writes `outputs/wqp/verdict.json` and `.md`.
- **Committed:** this file, the code, `outputs/wqp/check.json`, `outputs/wqp/ledger.json` and the verdict.
- **Never committed:** the arrays, parts, manifests' npz files, the placed verdict copies and the logs.

## 7. What follows (fixed now)

- **Whatever the verdict:** U2n's zrckpt refit, which U2s section 10 put on U2s's pick, is on the pick WQP carries,
  under WQ, in its own file.
- **If WQP carries a pick other than U2s's:** the upstream choices are chosen again under WQ, each in its own file, in
  this order: U2m (c) and (e), (b), (a), U2p's beta. The selection set then moves the choice, so the upstream choices
  should see it too.
- **If U2s's pick stays:** the upstream choices stand. U2s section 10's other next forms follow, under WQ: RRF5T's
  agreement as the budget, and P1h's cut over the walk's mass.

The text encoder stays frozen. Test splits are never read. There is no LLM and no topic-entity help.

## 8. Self-test, passed before this commit

`python outputs/mp_unified/zwqp.py --selftest` passes on the laptop. It checks:
1. **The patches.**
   - Every switch and root is set inside the block and restored after, on a failure too.
   - Check mode switches no carve and re-points nothing.
   - The select entry is refused when another is there.
   - run_carves passes {"select"} for webqsp only.
2. **The re-pointed checks.** Only webqsp's file sha256 changes, and the filed verdict's object is not changed. The
   sha256 is taken from the ledger. A record that is not the ledger's, or a stage the ledger does not hold, stops.
3. **The after-stage checks**, on toy records. They refuse a wrong question count, digest, carve, freeze or file
   sha256, a failed check, a `--limit` run, and a write under a filed root. A write under outputs/wqp or `__pycache__`
   is not flagged. The placed verdicts are the filed bytes and are never overwritten.
4. **The choice**, on toy tables. It checks:
   - the levels and the targets;
   - the verdicts and what is carried in each case;
   - the incumbent over the cap stays the bar, and an incumbent with no level is beaten;
   - the cap's restatement;
   - leave one out.
5. **The repository.** The chain's modules load once each. The filed check arrays, the three verdicts to place, and
   U2s's and U2c's verdicts are where section 1 names them.

`python outputs/mp_unified/zwqp.py reproduce` passes on the laptop too: the six filed tables give U2s's filed verdicts,
picks, levels, costs and ceilings, and U2c's chosen pool costs 1,953.5 under U2s's set.

## 9. Results

Filed 28 October 2026. One chain on the laptop, `zwqp.py chain --threads 6`, exit 0; the ledger holds seventeen
entries, each OK with no problems, and no earlier entry (nothing was retried).

**The check (section 2): REPRODUCED.** On webqsp's s1eval carve (1,503 questions) the five stages' arrays equal the
filed ones in every key, dtype, shape and byte (12, 12, 11, 14 and 11 keys for step 4d, step 4e, P1d, P1e and P1f), and
each file's sha256 equals the filed file's; no manifest entry differs. Seconds: 136.6, 246.5, 340.5, 524.1 and 485.4,
at 6 threads. Record: `outputs/wqp/check.json`.

**Select's stages.** The twelve stages on webqsp's select carve (305 questions), each OK: s4d 33.4 s, s4e 67.7, P1d
81.9, P1e 123.5, P1f 115.7, P1g 998.4, U2c 1,003.3, U2e 195.0, U2m 1,611.9, U2m (b) 877.6, U2m (a) 687.1 and U2s
1,538.9 (7,334.4 s in all). The verdict took 2,193.3 s: tables of 1,103,760 universal and 91,980 own candidates over
219 budgets on seven carves.

**The verdict: NOT_UNIFIED, KIND_FREE_TOO_DEAR, NO_GAIN. U2s's pick stays** (section 3's fourth case).
`outputs/wqp/verdict.json` and `.md`.
- S is squad, musique, hotpotqa and 2wiki on s1sel and webqsp on select, weighted 17,729, 7,018, 13,335, 18,504 and
  2,705.
- The cap is 1,545.3 rows, U2c's chosen pool `g:f3+cut_cos_7000@0.9` on S (1,953.5 on U2s's set).
- The incumbent, `own/kind_t/kt_cos:ada_f16_k0@0.85`, costs 1,382.4 on S, within the cap, and meets no level:
  webqsp's select carve reads ALL 0.9082 at 4,806.7 rows. Its level is taken as 0.94 (section 3).
- No pick of C, E or O is within the cap at any level from 0.95 to 1.0. At 0.95 the cheapest are C
  `rrf50+none/kind_t/kt_n1:fix_10000@0.95` at 5,974.3 rows, E `rrf50+FULL:h3_c25_v2000/both/kt_n1:ada_f6_k0@0.99` at
  7,262.3, and O `own/kind_t/kt_n1:ada_f6_k0@0.99` at 1,853.3. Each reaches 0.95 on webqsp's select carve only at about
  5,800 to 7,800 rows there (0.9508 in all three).

**Beside (section 4), deciding nothing:**
- **U2s's verdict, reproduced** from the same tables: NOT_UNIFIED, KIND_FREE_TOO_DEAR, OWN_CHEAPER, carrying
  `own/kind_t/kt_cos:ada_f16_k0@0.85`.
- **Leave one out.** WQ-musique's training sets (musique left out): cap 994.1, NOT_UNIFIED, KIND_FREE_TOO_DEAR,
  OWN_RAISED; pick O `own/kind_t/kt_n1:ada_f6_k0@0.99` meets 0.95 at 786.9 rows and would be carried. WQ-hotpotqa's
  (hotpotqa left out): cap 1,714.0, NOT_UNIFIED, KIND_FREE_TOO_DEAR, NO_GAIN, U2s's pick. So musique, WQ-musique's
  zero-shot set, is what keeps that pick O above the cap on S: it needs 9,796.1 rows on musique.
- **The price** is nothing, since the carried pick is the incumbent. Its reads (ALL, rows): metaqa 0.9653, 4,369.9;
  squad 0.9793, 50.0; musique 0.9668, 5,522.2; hotpotqa 0.9602, 713.1; 2wiki 0.9699, 1,070.5; webqsp's s1eval
  0.9069, 4,803.4; webqsp's select 0.9082, 4,806.7.
- **Ceilings:** metaqa 0.9947, squad 0.9993, musique 0.9811, hotpotqa 0.9881, 2wiki 0.996, and webqsp 0.9607 on both
  its carves. Webqsp is the one carve the carried pick leaves below 0.95. Its fewest rows at 0.95 are 5,661.8 on s1eval
  and 5,680.6 on select.

**What follows (section 7, U2s's pick stays).** The upstream choices stand. Each of the next forms is declared in its
own file before its numbers: U2n's zrckpt refit on U2s's pick under WQ; then, under WQ, U2s section 10's other next
forms, RRF5T's agreement as the budget and P1h's cut over the walk's mass.
