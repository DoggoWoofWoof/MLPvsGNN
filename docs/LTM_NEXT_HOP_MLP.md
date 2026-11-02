# LTM: zrckp with LTW's learned next-hop score as seven input columns, on PKO's pools, on the host's CPU

Declared 2 November 2026, before any LTM scorer fit, column or MLP fit. Stage modules: `outputs/mp_unified/ltmcols.py`
(the columns) and `outputs/mp_unified/zltm.py` (the MLP's fits, compares and calls). They import `ltwdiag.py`,
`chndiag.py`, `zwq.py`, `zml1s.py`, `zdob.py`, `zp1c3p.py`, `zdep.py` and `zltg.py` and edit none of them. ltmcols.py's
lists, A cases, folds and census ran on the host's CPU before this file (4ea493e, deciding nothing). Section 3 discloses
them, with the one setting fixed from their counts.

## 1. The question

- **The cause, as filed** (LTG section 1). musique in-domain is the main target (the user, 2 November).
  - The knee, A3: zg6r reads musique flat at every depth while the pool's reach rises.
  - CHN, NO_SIGNAL: no fixed score of the frozen vectors singles out the next hop among a found passage's neighbours.
  - LTW, SIGNAL and STORE_KEEPS: a **learned** three-way score of (question, found passage, their residual) against the
    row does. It puts musique's missed next hop first or second for 0.530 of s1eval's targets, against 0.360 for `pair`
    (0.548 on the store's axes; 2wiki 0.852, hotpotqa 0.795).
  - LTW, LEARNED_PAYS: reranking rrf's top five with it (H1) lifts R@5 by +0.039, +0.155 and +0.108 on musique, 2wiki and
    hotpotqa.
- **What the MLP lacks.** zrckp reads each pool row's own columns and the KT walk positions:
  - the first stage's ranks and dense cosine;
  - the typed links' topology and depth;
  - the store's vector columns (SEMB);
  - the seeds, the walks and the distances.

  None of them scores a row as the next hop from a passage the first stage found, for what the question still needs. C1R
  puts the MLP at 0.84 to 0.89 of zg6r's gain on musique. Claim 1 does not hold on two in-domain reads: WQ-hotpotqa's
  musique and WQ-musique's hotpotqa.
- **What LTM does.** LTG puts LTW's term inside the GNN's message passing. LTM gives the MLP the same learned score
  without message passing:
  - the score is computed outside the MLP, from the first stage's top five rows, PKO's links and the frozen vectors;
  - it enters the MLP as seven cached columns per row;
  - every input is fixed before the MLP scores any row. The user ruled (2 November) that reading the frozen vectors of a
    fixed beam "isnt message passing it is more akin to beam search";
  - a row's column depends on the other candidates of the same anchor through a softmax under the fixed scorer, as a rank
    column depends on the other rows' first-stage scores. No input depends on the MLP's own scores.
- **So the question is:** as inputs of zrckp, do LTW's next-hop columns raise in-domain R@5, above all on musique and its
  3- and 4-hop questions?
- LTW's H1 gains are over rrf's top five (musique s1eval R@5 0.4734), not over zrckp, which already ranks above rrf. This
  screen measures what is left for the MLP.

## 2. The columns (`ltmcols.py`)

- **Anchors.** Each question's top five rows by rrf in PKO's pool (the cache's rank columns), fixed before any row is
  scored.
- **Candidates.** Anchor k's candidates are its neighbours by any of PKO's link families, outside the top five: LTW's T
  lists. A row can be a candidate of several anchors.
- **The scorer.** LTW's `bil` form (LTW section 10), on the full 1,536-D rows. It has two parts:
  - a small network (hidden 32) on LTW's per-candidate features. These are the row's cosines to the question, to the
    anchor and to their residual, the pair score, the anchor's cosine, the first stage's rank columns, the row's log rank,
    and its link families;
  - the three-way bilinear term of rank 32: (question, anchor, residual) against the row.

  It trains with LTW's loop and settings:
  - on A lists only. A question gives an A list when rrf's top five holds a gold row and that row's candidates hold a
    missed gold; the first such gold is the anchor;
  - at most 256 candidates a list;
  - learning rate 1e-3, weight decay 1e-4, batch 32, up to 60 epochs, patience 6;
  - early stopping on the macro T2 over the datasets, measured on 0.10 of the training components held out.
- **The columns.**
  - Column k (next_1 to next_5), for a row among anchor k's candidates: the log-softmax of the scorer over anchor k's
    candidates, plus the log of their number. That is the log-ratio to uniform, clipped to ±8.
  - A row that is not among anchor k's candidates gets −8 in column k.
  - next_max is the largest of the five. next_n is the number of anchors whose candidates hold the row (0 to 5).
  - rrf's top five rows hold −8 and 0.
  - The columns are stored as float16 per cache part (`outputs/ltm/cols/<split>/<dataset>/<carve>/<part>/next.npy`).
    Each part's record pins, by sha256, its cache part's record, its links record, the scorer's fit record and the five
    fold files.
- **One scorer per split, refit on the split's own training carves** (zwq's):
  - WQ-musique: webqsp fit, squad fit, hotpotqa fit and 2wiki fit;
  - WQ-hotpotqa: webqsp fit, squad fit, musique s1fit and 2wiki fit.

  A split's scorer never sees the dataset the split holds out. LTW's own fits saw musique's s1fit carve and hotpotqa's fit
  carve together, so neither is used here.
- **Cross-fitted.**
  - The training carves' questions fall into five folds. A fold is built from components: questions that share a gold
    node in their pools stay together, and components are dealt largest first to the emptiest fold.
  - Fold fit j trains without fold j.
  - A training question's columns come from the fit that never saw its component. So the MLP trains on columns as good as
    those it reads at eval, not on fitted ones.
  - An eval question's columns come from the fold fit its index draws (`default_rng([0, dataset, 7])`, uniform over the
    five).
- **Which datasets train a scorer: those with at least 200 A lists** (`MIN_LISTS`). This was fixed from the census's
  counts (section 3), before any scorer fit:
  - webqsp's fit carve holds 26 A lists, squad's none;
  - LTW's loss weighs each dataset alike, and its stop is the macro over them. So 26 lists of webqsp would weigh as much
    as 2wiki's 3,464 and steer the stop on a handful of held components;
  - WQ-musique's scorer learns from hotpotqa and 2wiki, WQ-hotpotqa's from musique and 2wiki;
  - every carve's rows still get columns.
- **Checked.**
  - For each list, the anchor's and every candidate's cosine is recomputed from the vectors and checked against the cache's
    dense cosine (LTW's identity check: at most 1% of the checks may fail). A carve that fails writes no columns.
  - The MLP's loader refuses columns of another split, another scorer, other cache parts, another file or another dtype.
  - A fit's train and read must name one scorer, the split's present one.

## 3. The census (it ran before this file and decided only `MIN_LISTS`)

`python outputs/mp_unified/ltmcols.py lists`, all eleven carves, on the host (1 to 36 s each):

| carve | questions | components | largest (share) | T lists | rows in lists | gold in lists | A lists | A targets | rrf R@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2wiki fit | 5,928 | 4,184 | 84 (0.014) | 29,167 | 0.089 | 0.0078 | 3,464 | 3,464 | 0.598 |
| 2wiki s1eval | 12,576 | 7,221 | 2,565 (0.204) | 61,921 | 0.093 | 0.0084 | 7,619 | 7,619 | 0.608 |
| hotpotqa fit | 5,930 | 5,294 | 9 (0.0015) | 29,362 | 0.115 | 0.0070 | 2,222 | 2,222 | 0.695 |
| hotpotqa s1eval | 7,405 | 6,503 | 9 (0.0012) | 36,668 | 0.114 | 0.0069 | 2,799 | 2,799 | 0.685 |
| metaqa s1eval | 9,785 | 1,231 | 8,366 (0.855) | 48,358 | 0.008 | 0.0276 | 14 | 19 | 0.005 |
| musique s1eval | 2,417 | 533 | 84 (0.035) | 12,084 | 0.047 | 0.0038 | 915 | 923 | 0.473 |
| musique s1fit | 4,601 | 731 | 455 (0.099) | 23,002 | 0.045 | 0.0030 | 1,253 | 1,262 | 0.511 |
| squad fit | 5,856 | 5,399 | 3 (0.0005) | 27,791 | 0.473 | 0.0018 | 0 | 0 | 0.911 |
| squad s1eval | 11,873 | 1,444 | 26 (0.0022) | 56,120 | 0.430 | 0.0020 | 0 | 0 | 0.905 |
| webqsp fit | 1,202 | 667 | 203 (0.169) | 5,995 | 0.050 | 0.0113 | 26 | 88 | 0.057 |
| webqsp s1eval | 1,503 | 825 | 218 (0.145) | 7,500 | 0.048 | 0.0117 | 31 | 99 | 0.054 |

The columns of this table:

- **rows in lists:** the share of pool rows that are some anchor's candidate;
- **gold in lists:** the share of gold among the lists' candidates;
- **A targets:** the missed golds among the A lists' candidates.

What the census shows:

- **The folds are even.** Every training carve's largest component is under a fifth of its questions (webqsp's 0.169 is
  the largest), so each fold holds about a fifth.
- **The giant components are on eval carves:** metaqa s1eval 0.855, 2wiki s1eval 0.204. Their columns come from drawn
  fits. metaqa's draws give the five fits 1,979, 1,981, 1,917, 1,947 and 1,961 questions.
- **On musique, an anchor's candidates hold only 0.045 to 0.047 of the rows.** 95% of musique's rows read −8 in all five
  columns. The block can move only the rows next to rrf's top five, which is where LTW's targets are.
- squad's lists hold 0.43 to 0.47 of its rows (its pools are small); metaqa's hold 0.008.

## 4. What runs

- **The arm, zrckpn.** zrckp is ML1's MLP of record: zrm's model with the KT pointwise head (zp1c3p's ZRCKP), on zp1c3's
  chain carve. zrckpn is zrckp with NEXT as one more block.
  - The trainer reads NEXT like any cached block: its seven columns raw, z-scored on the training rows, and a mask. That
    is 15 more inputs to the first layer.
  - The output layer starts at zero, as zrckp's does.
- **Everything else is DOB's base:**
  - the trainer, the card, PKO's roots and typed links, and the KT input;
  - the read cap (65,536) and the candidate p@swa;
  - the pick: DOB's nine blocks, with NEXT after them. A fit whose trainer drops NEXT as dead on its training rows is
    refused.
- **The device is DOB's:** the host's CPU at 6 threads, in DOB's environment (torch 2.8.0+cpu), with the GPU hidden.
  - The base is DOB's CPU zrckp at seeds 0, 1 and 2 on both splits (`scr-dob-zrckp-s<k>`, with `-hp` on WQ-hotpotqa).
    These are DOB's own fits.
  - No GPU fit is compared.
- **The fits.**
  - Screen: seed 0 on both splits, `scr-ltm-zrckpn-s0` (WQ-musique) and `scr-ltm-zrckpn-s0-hp` (WQ-hotpotqa).
  - Full run: seeds 1 and 2 on both splits, queued now behind the screen's gate.
  - Each fit's train and read are stamped under the key `ltm`. The stamp names the split's scorer (its fit record's
    sha256), every carve's column record (sha256), the pick, the threads and the device, beside zml1s's stamp (gate,
    code).
- **The compares,** through zwq's `run_compare`, as DOB's:
  - each LTM fit against DOB's base fit of its seed and split;
  - DOB's seeds 1 and 2 against its seed 0 (the null), filed again under LTM's names.
- **How it is set, in-process.**
  - zltm registers zrckpn beside zrckp in lean_screen's arm table: zrckp's model on the chain carve, with the split's
    columns beside it.
  - For the length of a fit, zml1s runs with LTM's names, stamp and stage, DOB's device and card, and lean_gpu's pick with
    NEXT.
  - Every patched name is put back afterwards, also on a failure.

## 5. The reads

DEP's section 3, unchanged:

- ML1's twelve reads: two splits by six datasets, p@swa on the s1eval carves;
- musique's 3-hop (760 of 2,417 questions) and 4-hop (405) slices on both splits, each from the fit's musique read file
  (`zdep.slice_r5`). A slice file's R@5 over all questions must equal its compare's within 0.00006.

## 6. The rule (fixed now)

LTG's in-domain rule (LTG section 4), per the user's "indomain for now", against DOB's CPU base.

- **Deciding (ten):**
  - the eight reads of datasets their split trains on. WQ-musique trains on webqsp, squad, hotpotqa and 2wiki;
    WQ-hotpotqa on webqsp, squad, musique and 2wiki;
  - the WQ-hotpotqa fits' musique 3-hop and 4-hop slices.
- **Beside (six):** the four zero-shot reads (musique and metaqa on WQ-musique; hotpotqa and metaqa on WQ-hotpotqa) and
  the WQ-musique fits' musique slices.

**The screen (seed 0).** Each deciding read is called on its own:

- the floor is max(0.0075, 2 x the base's spread over its three seeds);
- **LOSS:** the delta is at or below minus the floor. **GAIN:** the delta is at or above the floor;
- **NOT_ADOPTED:** any LOSS. **ADOPT_FOR_FULL_RUN:** no LOSS and at least one GAIN. **NO_GAIN:** every read WITHIN.

**The full run (seeds 1 and 2).** Each deciding read is called by `zml1.call` on its three seed deltas, with the pair
floor of both arms' spreads.

- **ADOPT:** no LOSS among the ten, and a GAIN on at least one of the eight dataset reads. A GAIN means d0 is at or above
  the floor and d1 and d2 are above 0. A slice GAIN alone does not adopt, and UNCONFIRMED_GAIN does not count.
- **NOT_ADOPTED** otherwise.

**INCOMPLETE** holds when any of these is true:

- a fit, compare or read file is missing, or a base seed is missing;
- a fit's train or read did not run on DOB's device at 6 threads with LTM's pick;
- a fit's gate is not PKO's present gate;
- a fit's train and read name different scorers, or a scorer that is not the split's present fit record;
- a training carve's columns are missing from the train, or an eval carve's from the read;
- a column record changed after the fit read it;
- the fit's trainer code differs from its base's (stage modules aside);
- the base's R@5 at a seed differs between two compares, or a read file's R@5 differs from its compare's.

A verdict closes only the form it tested.

## 7. Reported beside, deciding nothing

- The six beside reads, called the same way.
- DEP's sixteen-read rule on the same numbers (`zdep.decide_screen`, and `zdep.decide_full` once the full run is read),
  as `dep_rule_beside` in `outputs/ltm/calls.json`.
- **The columns per carve.** For each carve: its share of rows next to an anchor, and its A lists' T2 under bil, `pair` and
  rrf. Each A list is ranked by the fit its question's columns come from. On a training carve that is out of fold; on an
  eval carve it is the drawn fit.
- Seconds per fit.
- C1R's two reads (WQ-hotpotqa's musique, WQ-musique's hotpotqa), marked in `calls.md`.

## 8. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `ltm-gate`, then section 6's full-run rule. On ADOPT, zrckpn is the MLP candidate. Claim 1 (C1R's intervals, against the GNN of record) is measured again with it. Claim 3's cold timing counts the columns' cost: the scorer's forward over every anchor's candidates and reading their vectors, measured, not assumed. |
| NO_GAIN or NOT_ADOPTED | The columns as declared close. The next form, in its own file, follows the beside reads (section 7), in this order. (a) If a training carve's out-of-fold A T2 under bil is more than 0.05 above the same dataset's eval carve's, the folds leak, and the next form refits with folds over every pool node two questions share. (b) Otherwise, if the WQ-hotpotqa fit's musique 3- and 4-hop slices fall while its musique read does not, the next form is the cumulative residual as columns (LTG section 6's next form): each anchor's query takes out the span of all five anchors, not its own passage alone. (c) Otherwise, the next form is the term over every pool row against each anchor, linked or not, since musique's lists hold 0.045 of its rows. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

LTG (the term inside the GNN) is declared in its own file. LTM does not wait on it, and neither model is built on the
other.

## 9. Risks, stated before any number

- **Seed 0 only in the screen.** This is a one-fit screen; the full run decides any adoption.
- **Leakage across folds.** The component folds keep questions that share a gold node apart. Other shared structure, such
  as a shared non-gold passage or entity, can still cross folds. A training row's out-of-fold columns may then be better
  than an eval row's, and the MLP would trust them more than they deserve. The beside table puts each training carve's
  out-of-fold T2 next to its eval carve's; section 8 (a) acts on a gap.
- **Out of fold against drawn.** Both a training row's and an eval row's columns come from a fit on four fifths of the
  training lists. They are alike by construction, leakage aside.
- **One hop.** The columns score the anchors' direct neighbours. A 3- or 4-hop question's third passage may be two links
  from a found one. The WQ-hotpotqa slices are deciding reads; section 8 (b) acts on them.
- **All link families, NER included.** NER is LTW's weakest family (store's term 0.401 against `pair`'s 0.381;
  structural 0.681). The scorer has the families as features, but the columns do not separate them.
- **webqsp and squad do not train a scorer.** Their rows get columns from a scorer that learned on the split's two
  multi-hop datasets. squad has no A lists at all.
- **Coverage.** musique's lists hold 0.045 of its rows, so most rows read −8; section 8 (c) acts on this.
- **The start.** NEXT widens the first layer by 15 inputs, so the fit's random draws are not the base's. The floor
  (2 x the base's seed spread) is set for a change of draw.
- **The columns' cost at serving.** It enters Claim 3 only if LTM adopts, and then as measured.
- **Time.**
  - DOB's CPU zrckp fits took 1 h 25 min to 2 h 24 min (train and read). LTM's add 15 inputs and the columns' load: about
    the same.
  - LTW's joint fit took 2 min 24 s at 2 threads. Each split's five fold fits should take 10 to 20 min, and each carve's
    columns seconds to about 15 min.
- **Memory.**
  - DOB's CPU zrckp fits peaked at 27.2 GB (WQ-musique) and 30.1 to 30.8 GB (WQ-hotpotqa). The columns add under 1 GB per
    fit.
  - The scorer fits should peak near LTW's joint fit (2.5 GB). A column item holds its carve's rows' columns and its A
    lists' vectors: 2wiki s1eval, the largest, should stay under 4 GB.
  - The feeder resizes each arm from its measured peaks.
- **Disk.** The twenty column sets hold 218 million rows x 7 float16 columns: about 3.1 GB.

## 10. How it runs

Feeder items on the host's CPU. They do not touch the GPU.

1. `ltm-selftest` (`zltm.py --selftest`) and `ltm-selftest-cols2` (`ltmcols.py --selftest`, on the code committed with
   this file).
2. The scorer fits, after the self-tests (2 CPUs, 6 GB):
   - `ltm-fit-wqm`: `ltmcols.py fit --split WQ-musique --host`;
   - `ltm-fit-wqh`: `ltmcols.py fit --split WQ-hotpotqa --host`.
3. The twenty column sets, `ltm-cols-<wqm|wqh>-<dataset>-<carve>`, each after its split's fit (2 CPUs, 6 GB); then
   `ltm-cols-report`.
4. The screen's fits, after `pkoh-gate`, `ltm-selftest` and their split's ten column sets (6 CPUs):
   - `ltm-zrckpn-s0`: 34 GB;
   - `ltm-zrckpn-s0-hp`: 38 GB.
5. `ltm-compare` (after the two fits and DOB's six base fits), then `ltm-calls`, then `ltm-gate`
   (`outputs/ltm/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
6. The full run, after `pkoh-gate` and `ltm-gate`: `ltm-zrckpn-s1`, `-s2`, `-s1-hp` and `-s2-hp`, with the screen's
   sizes. Then `ltm-compare-full` and `ltm-calls-full`.

```
python outputs/mp_unified/ltmcols.py fit --split WQ-musique --host
python outputs/mp_unified/ltmcols.py cols --split WQ-musique --dataset musique --carve s1eval --host
python outputs/mp_unified/ltmcols.py report                         (outputs/ltm/report.json and .md)
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zltm.py fit --name scr-ltm-zrckpn-s0
python outputs/mp_unified/zltm.py compare-all
python outputs/mp_unified/zltm.py calls                             (outputs/ltm/calls.json and .md)
```

Only `outputs/ltm/calls.json`, `calls.md`, `report.json` and `report.md` are committed. The lists, fold fits and columns
stay on the host.

Both self-tests passed on the laptop before this commit.

`ltmcols.py --selftest` covers:

- the carves of each split;
- the folds on a toy component graph;
- the log-ratio: uniform scores give 0, the ratios normalise, and they are clipped;
- on a toy PKO tree of the WQ-hotpotqa carves:
  - the lists, with no top-five row among the candidates, and A lists equal to LTW's own;
  - a split's five fold fits, with the identity check and the left-out counts;
  - the columns: shape, dtype and records' shas. Each column equals its fold fit's log-ratio, and another fold's fit gives
    other columns. next_n counts the anchors, the top five hold −8 and 0, next_max is the largest, and rows off every list
    hold −8;
  - the report;
  - `MIN_LISTS` leaving a dataset out.

`zltm.py --selftest` covers:

- the names, the pairs and the pick, disjoint from ML1's, DOB's and LTG's fits;
- zrckpn's arm, and the swap and its restoring;
- the columns beside zp1c3's toy chain carve:
  - alignment with an independent row index, the widths, and the dead-block check;
  - every refusal: another split; an edited training flag, identity, split, column list, cache sha, scorer or row count;
    a changed file; float32 under matching shas; a preset scorer; a second carve on another scorer;
- the model: 15 more inputs to the first layer, and with drawn weights the output reads NEXT;
- a toy `fit_variant` with NEXT, and the trainer's set check;
- the patches, and that they are put back, also on a failure;
- the compares' and slices' reading on filed shapes;
- section 6's rule and DEP's beside on made-up reads.
