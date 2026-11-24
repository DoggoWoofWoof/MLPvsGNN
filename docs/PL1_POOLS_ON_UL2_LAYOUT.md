# PL1: the pools rebuilt on the adopted UL2 layout, graded on all seven datasets (the last pool stage)

Declared 25 November 2026, before any pool is built and before any PL1 number. **Nothing here trains, no model reads a
vector it did not read before, and nothing of the record is rewritten.** The pools of record (PKO, B1P, ABL's SRK) stay
where they are and are the comparison.

## 0. The user's words, and the ruling this page follows

- 25 Nov, first: "we have our proper encoders frozen and what form are we storing then use those do not waste time or
  memory if we dont have to, finalize pools step by step until we are completely finished and dont need to reopen it,
  check according to the after paper plan if wehave missed out something".
- 25 Nov, second (to the question whether to override the 11 Nov rule): "yeah do it we can reclaim space also after it
  and the plans arent dataset specific all the datasets can be compared with oracle and reach diagnosis and can be
  improved do for everything, link improvement with precision is all we need".
- **Amendment A6** of docs/CANONICAL_LINK_AND_POOL_LAYOUT.md (a9fb5f2) records the ruling: A2's pool standard is
  superseded for this one stage, and only once the universal call of docs/UL2_LAYOUT_ON_E3V.md (sections 12.6 and 13) is
  filed. Earlier rulings that bind: universal rules ("we need to have universal rules": every rule on all seven
  datasets), S only where the data has structure and both versions on 2Wiki and HotpotQA (10 and 21 Nov), pools first and
  parameter-free (no learning until the pools are final), one model per family, no LLM, the hub line closed (UL2 8.15:
  no hub step is proposed here; hub forms already in UL2's layout are labels like any other).

## 1. What is open, and what PL1 closes

| open item (the 5 Nov list, the canonical layout's section 6, the later rulings) | where it stands | PL1 |
| --- | --- | --- |
| pools built on the updated rules (spaCy's ner family still inside the pools of record) | pools of record keep the old walk graph (A2) | rebuilds them on the adopted layout |
| coverage at fewer rows, with S in | COV (25ce7d7) read 2Wiki, HotpotQA and MuSiQue on the old first stage only | COV's measures on all seven, on the new pools |
| reach diagnosis (why the ranked nodes do not reach the gold) | asked for MuSiQue and WebQSP (layout section 6 item 3); no filed record has the split | the split on all seven (section 4, M3) |
| oracle comparison | UL1 and UL2 read slots on the layout; the pools' own reach was never read against the oracle | M2 on all seven |
| link precision per dataset | AP4 (six datasets, older layout); the ul2p ledger (laptop session) on the e3v layout | M4, the same ledger on the pools' reach |
| MultiHop-RAG | has no pick-O pools | builds them with the same rule (section 5) |
| the refit | waits on the pools (UL2 section 9) | PL1's pools are the refit's pools |

**What PL1 decides:** whether the new pools replace the pools of record as the standard, and the end of the link and pool
work (section 7). It decides nothing about any model.

## 2. What is fixed (nothing below is chosen after a number)

- **Encoder and storage.** e3v is the encoder (D101); the frozen encoder stays as a comparison only. Document vectors are
  read as int8 (E5's scheme, FREE on six datasets); windows are not read at all. The first stage is the e3v dense list
  (ENC's E0 lists) with SPLADE unchanged, 1,000 deep, as the pool chain already reads it. No second encoder, no new
  encode, no new storage form.
- **The graph.** The layout UL2's universal call carries, per dataset and version: B0 = S | W2_S | F0T | K_e3v plus the
  forms the call carries (UL2 12.3, 12.6, 13). S > R > K dedup, every pair held by one family only, labels by the rule that
  made it. 2Wiki and HotpotQA carry both versions (RK and SRK); MuSiQue, SQuAD and MultiHop-RAG RK; MetaQA and WebQSP SRK.
  The three walk families keep the slots the record gives them (structural; ner, now R's other parts; knn, now K_e3v), as
  D83 section 4 maps them, with the carried forms inside R's slot.
- **The pick rule.** PKO's pick O exactly as filed (`own/kind_t/kt_cos:ada_f16_k0@0.85`, docs/PKO_PICK_O_POOLS_UNDER_WQ.md
  section 2). **This page adds no constant and changes none.**
- **Questions.** Every carve the refit will read: the fit carves, the dev carves (s1sel) and the s1eval carves of the six
  datasets, and MultiHop-RAG's train, dev and test. The measures below are read on the eval carves (s1eval; MultiHop-RAG
  test); s1sel is read beside.

### 2a. Universal (the user, 25 Nov: "the pools or the rules or links all should be universal")

- **One rule set, one call.** The link rules (S, W2_S, F0T, K_e3v, the carried forms, the S > R > K dedup, the labels), the
  pool rule (pick O with its constants) and the four measures are the same on all seven datasets and both versions. No
  dataset gets its own constant, form, label set, cut, depth or first stage, here or in PL2.
- **What differs between datasets is the data, not the rule.** S exists only where the data has structure (KB triples,
  corpus hyperlinks; ruling 10 Nov), and 2Wiki and HotpotQA show both versions (21 Nov). The node is the corpus's own unit
  (an article for MultiHop-RAG). Same rule, different inputs.
- **Carried everywhere or nowhere.** Any link form or pool change enters by UL2's universal rule (ABOVE somewhere, BELOW
  nowhere, one price bound), on all datasets at once. The call in section 6 is ONE call over all seven, never one per
  dataset. A form that helps one dataset and hurts another is not carried, and PL2 may not special-case a dataset.
- **Stand-ins are rules too.** Where a dataset lacks an input (no oracle pairs: SQuAD; no pool of record: MultiHop-RAG), the
  stand-in is declared once, in this page, for every dataset that lacks it.
- **Enforced in the grade.** `zpl1g` records one hash of the constants and the code path in every dataset's report, and
  `pl1-report` refuses to file when two datasets differ.

## 3. The steps (each step's record is filed, with its gates, before the next runs)

| step | what | gate (a failure stops that dataset; none of its numbers is read) |
| --- | --- | --- |
| S1 | **Materialize the layout as walk stores:** one CSR per family and per version, from B0 and each dataset's `outputs/ul2c/<d>/carried.npz` (the kept pairs beyond B0 with the carried forms' bit per pair, written by `zul2c.write_carried` when the call is filed). New code only; nothing of UL2 is edited. | G-L: every family's pair count equals UL2's grade counts; dedup exact (no pair in two families); labels' bit counts equal the grade's |
| S2 | **The e3v first stage** for every question: the E0 lists. | G-E: R@5 and R@1000 equal E4's filed values (E6 for MultiHop-RAG) |
| S3 | **The adapter reproduces the record.** The new world, fed the OLD graph and the frozen first stage, rebuilds one carve's pools of record (the identity check D83 used as G1). | G-I: B, |I_q|, the keep cut and I_q's sha256 equal the record's on one carve per dataset |
| S4 | **Build the pools** on the new graph and the e3v first stage, all carves of section 2. | the build's own checks (zpko's); no gold is read to build a pool |
| S5 | **Grade** M1 to M4 (section 4), on the eval carves. | G-C: the grade's pool-size and gold counts equal the build's |
| S6 | **The call and the closing rule** (sections 6 and 7). | |

Cost is not measured on the e3v graph. The pools of record cost about 30 to 39 lane-hours for all datasets (PKO's docs);
S4 is declared as host rows in the feeder's order, and its wall time is reported, not assumed.

## 4. The measures (the same code and constants on all seven datasets)

- **M1, coverage and rows (COV's).** ALL (every gold of a question in the pool), gold recall, mean pool rows, rows as a
  share of the corpus; new pools against the pools of record on the same questions, paired 95% bootstrap (COV's
  `ratio_ci`, `diff_block`). MultiHop-RAG has no pool of record: its comparison is the first stage's own list at the same
  rows (COV's equal-rows form).
- **M2, the oracle comparison.** Gold pairs (GP) as UL1 and UL2 define them: the share of a question's gold pairs with a
  link inside the pool graph and inside the pool; MuSiQue's chain slots (CH) beside, by hops (2, 3, 4); the KBs' two to ten
  golds as UL1. **A dataset with no oracle pair stands on M3:** SQuAD (one gold per question) has no pair, so its oracle
  comparison is M3 alone. The same rule applies to any dataset that lacks pairs.
- **M3, the reach diagnosis (new; built from zu2s's per-gold place `pos` and the build's `keep`).** Every gold outside the
  first stage's top 5 is placed in one class:
  - **IN_B**: already in the first-stage union B;
  - **KEPT**: not in B, in a walk's order inside the keep cut (the pool reaches it);
  - **BEYOND**: not in B, in a walk's order but beyond the keep cut (a link path exists; the row budget cut it);
  - **NOPATH**: in no walk's order (no link path within the walk's depth from any seed).
  
  Read per dataset, per hop on MuSiQue (2, 3, 4), and per label of the walk that placed the gold (KEPT and BEYOND). The
  counts are shares of golds, with paired intervals against the record's pools where the record has them.
- **M4, precision.** The label precision ledger of docs/UL2P_LABEL_PRECISION_E3V.md (AP4's measures: top-1 precision, lift,
  reach, rank1) read on the new pools' anchors, and the precision of the pairs that placed each KEPT gold. Calls POINTS /
  NO_LIFT / BELOW_BASE as AP4 defines them.

## 5. MultiHop-RAG

It has no pool of record and no pick-O build. PL1 builds its pools with the same rule on article nodes (D84's granularity),
the same S1-S5, and reports it as a seventh row of every table, not a special case. Its eval is the test split; dev is read
beside.

## 6. The calls (fixed now)

Each measure is read per dataset and version, on the eval carve, against the pools of record (MultiHop-RAG: against the
first stage at equal rows). The call itself is ONE call over all of them (section 2a):

- **Coverage.** Gold recall: ABOVE / LEVEL / BELOW by the paired interval, and ALL likewise.
- **Price.** Mean rows: the new pools' rows over the record's, with its interval.
- **ADOPT_POOLS** (the new pools become the standard) when, on every dataset and version, gold recall is not BELOW and the
  rows ratio is not above 1.00 at its upper end, OR any dataset above 1.00 is named as the price of a gold-recall gain
  that is ABOVE on that dataset and the others are not BELOW; otherwise **NOT_ADOPTED: the pools of record stay**. (The
  fixed default is the first clause; the second only keeps a dataset's gain from being judged as its cost.)
- **Where it fails** the diagnosis names why (M3, M4), and section 7 applies.

## 7. The closing rule: no reopening

The user: "finalize pools step by step until we are completely finished and dont need to reopen it". So the stage has a
fixed end, decided now:

1. **PL1's report** files M1 to M4 and the call for all seven datasets.
2. **At most ONE improvement round (PL2).** It runs only if the call is NOT_ADOPTED somewhere, or a dataset's NOPATH
   share is above 5% of its golds and M4 names a link form already declared in UL2 (sections 6 and 7), AP or MHB that
   would reach them at precision above the base rate (POINTS). Any form must pass UL2's universal rule (ABOVE somewhere,
   BELOW nowhere, one price bound, carried everywhere or nowhere). No hub step (UL2 8.15). PL2 is declared as its own page
   before it runs.
3. **After PL2, or at once if it is not needed, the pools are frozen** whatever the numbers: the layout, the pools and the
   residual (every NOPATH and BEYOND share, named by label) are filed as the standard, and the refit reads them. The link
   and pool work is then closed. A later reopening needs the user's explicit word.
4. **Space.** Only after the closing record is filed: a list of what can be reclaimed (the other encoders' vectors, the
   frozen-pool intermediates the new pools replace, arm worlds nothing reads) is written for the user, each delete on the
   user's OK, after outputs/host_cleanup/deleted_*.json is checked. The pools of record are kept until the user says
   otherwise.

## 8. Code, rows, outputs

- **New modules only** (they import, never edit): `outputs/mp_unified/zpl1m.py` (S1, materialize), `zpl1w.py` (the world
  adapter: `zenc1b2.install`'s patches generalized to all seven datasets and UL2's carried layout; S2 to S4),
  `zpl1g.py` (S5: COV's array-level functions, M2, M3, M4). Never edited here: zpko, zu2d_pools, zu2s, zu2c, zd83*,
  d83world, covdiag, zenc1b2, zul2*, zap4 and everything on the never-edit list.
- **Rows** (host, prefix `pl1-`), inserted in the feed after the UL2 chain's reports: materialize and the first-stage gate
  per dataset, the identity check per dataset, then the builds in a size order that keeps the memory cap (the old chain's
  lane sizes: zpko `LANE_GIB`, `SHARD_Q`), then the grades, then `pl1-report`.
- **Outputs:** `outputs/pl1/<dataset>/` and `outputs/pl1/report.{json,md}`. Arrays stay uncommitted.
- **Self-test on real-shape data** before any host row (a toy test passed once while eight real fits failed).

## 9. Disclosed

- Written after a read-only survey of the pool code (25 Nov) and after the UL2 grades of MuSiQue, SQuAD, MetaQA and
  MultiHop-RAG were filed; no pool number and no PL1 number was read. The universal call (UL2 12.6 and 13) is not yet
  filed; PL1's rows wait for it.
- No UL2 graph materializer exists yet. The one artifact S1 reads is `outputs/ul2c/<d>/carried.npz` (written by `zul2c.write_carried`; it does not exist before the call is filed); S1 writes the walk stores. An earlier line of this page called that file `carved.npz`, which was my wrong name; corrected 25 Nov before any number.
- The e3v E0 first-stage lists are on the host; whether every dataset's exists is checked by G-E.
- The closing rule's "one improvement round" and the 5% NOPATH line are mine, a default the user can change before the
  first PL1 number is read.
- This page decides nothing about the models, the paper's numbers, the encoder or the hub line.
