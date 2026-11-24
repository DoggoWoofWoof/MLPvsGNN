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
- 25 Nov, third (on the closing rule): "i dont want to freeze pools for the sake of it, make sure we freeze when we know
  it is the best for that specific arm etc". Section 7 follows it (amendment 2d).
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

**What PL1 decides:** whether the new pools replace the pools of record as the standard, and, arm by arm, when the link and
pool work for that arm is finished (section 7). It decides nothing about any model.

## 2. What is fixed (nothing below is chosen after a number)

- **Encoder and storage.** e3v is the encoder (D101); the frozen encoder stays as a comparison only. Document vectors are
  read as the host holds them (e3v's shards, fp16): no second copy is made for PL1. The int8 form (E5, FREE on six datasets)
  is the form to keep after the pools are final; converting the shards is on section 7's space list, each step on the
  user's OK. Windows are not read at all. The first stage is the e3v dense list
  (ENC's E0 lists) with SPLADE unchanged, 1,000 deep, as the pool chain already reads it. No second encoder, no new
  encode, no new storage form.
- **The graph.** The layout UL2's universal call carries, per dataset and version: B0 = S | W2_S | F0T | K_e3v plus the
  forms the call carries (UL2 12.3, 12.6, 13). S > R > K dedup, every pair held by one family only, labels by the rule that
  made it. 2Wiki and HotpotQA carry both versions (RK and SRK); MuSiQue, SQuAD and MultiHop-RAG RK; MetaQA and WebQSP SRK.
  The three walk families keep the slots the record gives them (structural; ner, now R's other parts; knn, now K_e3v), as
  D83 section 4 maps them, with the carried forms inside R's slot.
- **The pick rule.** PKO's pick O exactly as filed (`own/kind_t/kt_cos:ada_f16_k0@0.85`, docs/PKO_PICK_O_POOLS_UNDER_WQ.md
  section 2). **This page adds no constant and changes none.**
- **Questions.** The s1eval carves of the six datasets and MultiHop-RAG's test split (dev beside it): the carves E0's e3v
  first-stage lists exist for (E0 holds the s1eval rows only). Pools for the fit and s1sel carves are NOT built here: the
  e3v first stage on those questions does not exist, and the refit's own declaration builds them with this same code once
  the pools are adopted. The measures below are read on the eval carves.

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

### 2b. What each of the three walk families holds, per version (declared now; the survey found it was only implied)

| version | structural family | ner family | knn family |
| --- | --- | --- | --- |
| RK (MuSiQue, SQuAD, MultiHop-RAG; 2Wiki and HotpotQA without S) | W2_S | F0T, then the carried forms | K_e3v |
| SRK (MetaQA, WebQSP; 2Wiki and HotpotQA with S) | S | W2_S and F0T, then the carried forms | K_e3v |

The rule, one for all datasets: **the structural family holds S where the version has S, else W2_S; the ner family holds the
rest of R (W2_S when S holds the structural slot, F0T, the carried forms); the knn family holds K_e3v.** RK is D83 section 1's
slot; SRK is A1's (S in the slot, the rule graph inside R). The S > R > K dedup is D83's `d83stores.dedup`: the ner store
loses the pairs S holds where S exists, nothing is removed inside R, and the knn store loses every pair the other two hold.

**Row order, not weight.** The survey (zu2m_a, step4_pool_reach) found that the walk does not read a family's weight: a ner or
knn edge counts 1 times the steer, and the structural edge counts the EM and score type's weight. A store's weight only
fixes the order of each row, which matters where the code reads a row prefix (`capped`, `capped_multi`, `expand_hops`, which
build the incumbent pool I). So the order is declared, with no new constant: in the ner store each node's row lists the F0T
pairs by 1/df weight (D83's), then the W2_S-only pairs (SRK), then the carried pairs by the form's own score, ties by node id.
K_e3v's weight is the cosine, the larger of the two ends (zenc1b2's `k_edges`). A pair is held by one family only (dedup)
except inside R.

**Types.** The EM and score types are fitted on the structural CSR (zpko checks the entries equal the adjacency's). So the
types are refitted on every (dataset, version) structural CSR, on e3v's vectors, by `zu2m_a_types.types_stage` as ENC E1b
part 2 did for its arms. An unsupervised fit on the family's pairs; no question, no gold. One process serves one (dataset,
version).

## 3. The steps (each step's record is filed, with its gates, before the next runs)

| step | what | gate (a failure stops that dataset; none of its numbers is read) |
| --- | --- | --- |
| S1 | **Materialize the layout as walk stores:** one CSR per family and per version, from B0 and each dataset's `outputs/ul2c/<d>/carried.npz` (the kept pairs beyond B0 with the carried forms' bit per pair, written by `zul2c.write_carried` when the call is filed). New code only; nothing of UL2 is edited. | G-L: every family's pair count equals UL2's grade counts; dedup exact (no pair in two families); labels' bit counts equal the grade's |
| S2 | **The e3v first stage** for every question: the E0 lists. | G-E: R@5 and R@1000 equal E4's filed values (E6 for MultiHop-RAG) |
| S2b | **Refit the EM and score types** on each (dataset, version) structural CSR with e3v's vectors (`types_stage`). | G-T: the same code on the OLD structural CSR with the frozen vectors reproduces the filed types array for array (E1b2's GT) |
| S3 | **The adapter reproduces the record.** ENC E1b part 2's two gates, on every dataset: GV (the record view: PKO's own pass through the adapter equals PKO's filed part, array by array, on one part per dataset) and GS (the search view: the frozen first stage through the new world's code path holds the golds as the filed pools do, within 0.1 point). | G-I = GV and GS |
| S4 | **Build the pools** of the eval carves on the new graph and the e3v first stage, in one pass that also writes each gold's place (the reach read, section 4 M3) after the question's pool is final. | the build's own checks (zpko's); G-N: the pool arrays are identical with and without the placement code (no gold reaches the build) |
| S5 | **Grade** M1 to M4 (section 4), on the eval carves. | G-C: the grade's pool-size and gold counts equal the build's |
| S6 | **The call and the closing rule** (sections 6 and 7). | |

Cost is not measured on the e3v graph. PKO's 30 to 39 lane-hours were for all 11 carves; PL1 builds the eval carves only.
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
- **M3, the reach diagnosis (new; the per-gold place `pos` of zu2e.places, and the build's `keep`).** `pos` is 0 for a gold
  in B, p >= 1 for the p-th node of an order, and -1 for a gold in no order. Every gold outside the first stage's top 5 is
  placed in one class:
  - **IN_B**: already in the first-stage union B;
  - **KEPT**: not in B, 1 <= pos <= keep (the pool reaches it);
  - **BEYOND**: not in B, pos > keep (the walks reach it; the row budget cut it);
  - **OUTSIDE**: pos = -1, in no order: outside every source, the walks' depth and the NMAX = 7,000 cut included. This is
    **not** claimed as "no link path"; where a path test is wanted, the grade reads `one_link` and `in_I` beside.

  The orders are the whole of `question_orders`' (the survey: the kept prefix alone cannot separate BEYOND from OUTSIDE), so
  S4 computes them for the golds in the same pass; the pools file does not hold them.
  
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

## 7. The closing rule: an arm is frozen when it is known best, not before

The user, 25 Nov: "finalize pools step by step until we are completely finished and dont need to reopen it", and then: "i
dont want to freeze pools for the sake of it, make sure we freeze when we know it is the best for that specific arm etc".
So the stage ends arm by arm, on evidence, and has no cap on rounds. (The first draft of this section capped the work at one
improvement round and froze "whatever the numbers"; that was my default and the user's word replaces it.)

**An arm** is one (dataset, version) pair: MuSiQue RK, SQuAD RK, MultiHop-RAG RK, MetaQA SRK, WebQSP SRK, and 2Wiki and
HotpotQA each under RK and SRK. Nine arms.

1. **PL1's report** files M1 to M4 and the call (section 6) for all nine arms, with each arm's residual: every OUTSIDE and
   BEYOND share, by label (M3), and the M4 reading of the links that placed the KEPT golds.
2. **Improvement rounds PL2, PL3, ... with no cap on their number.** A round is declared on its own page before it runs. It
   names the arms whose residual it targets and the one change it tries: a link form or a pool change that M3 and M4 point
   to (a form with POINTS at a residual label; for BEYOND, a pool change). It enters by UL2's universal rule (ABOVE somewhere,
   BELOW nowhere, one price bound, carried everywhere or nowhere), so it is graded on all nine arms, never one. No hub step
   (UL2 8.15). A change that is not already a declared form (UL2, AP, MHB) is a design step: it goes to the user first.
3. **The freeze test, per arm.** An arm is FROZEN when its own record shows both:
   - PL1's grade and the residual named by label are filed for it, and
   - the candidate list for that arm is empty: every change M3 or M4 says could reach its residual has been graded under
     the universal rule and was carried (the arm was rebuilt with it and graded again) or NOT_CARRIED, or the residual holds
     no gold.
   The freeze record names the evidence files. It is not a share threshold: the 5% OUTSIDE line of the first draft is
   dropped as a trigger and stays in the report as a flag. An arm with a non-empty candidate list is OPEN, and the report
   says which candidate is next.
4. **Rules are universal, freezes are per arm.** A frozen arm never gets its own rule, form or constant (section 2a). When a
   later round carries a form, it carries to every arm at once; the frozen arms are rebuilt with it and graded again (a
   carried form is BELOW nowhere, so no arm is worse), and each arm's freeze record is filed again. So "frozen" means "no
   declared candidate is left that could improve this arm", and it holds until a carried form changes the layout for all.
5. **When the pools are final.** The refit reads the frozen arms. Its start is the user's call; the report says how many
   arms are frozen and what keeps any other open. A frozen arm is reopened only on the user's explicit word, or by item 4's
   rebuild.
6. **Space.** A list of what can be reclaimed (the other encoders' vectors, the frozen-pool intermediates the new pools
   replace, arm worlds nothing reads) is written for the user when the last arm is frozen, or sooner for items that only
   frozen arms used. Each delete is on the user's OK, after outputs/host_cleanup/deleted_*.json is checked. The pools of
   record are kept until the user says otherwise.

## 8. Code, rows, outputs

- **New modules only** (they import, never edit): `outputs/mp_unified/zpl1m.py` (S1, materialize), `zpl1w.py` (the world
  adapter: `zenc1b2.install`'s patches generalized to all seven datasets and UL2's carried layout; S2 to S4),
  `zpl1g.py` (S5: COV's array-level functions, M2, M3, M4). Never edited here: zpko, zu2d_pools, zu2s, zu2c, zd83*,
  d83world, covdiag, zenc1b2, zul2*, zap4 and everything on the never-edit list.
- **Rows** (host, prefix `pl1-`), inserted in the feed after the UL2 chain's reports: materialize and the first-stage gate
  per dataset, the identity check per dataset, then the builds in a size order that keeps the memory cap (the old chain's
  lane sizes: zpko `LANE_GIB`, `SHARD_Q`), then the grades, then `pl1-report`.
- **Outputs** (`<d>` = the dataset, `<v>` = RK or SRK; MultiHop-RAG is RK). Arrays stay uncommitted; the json heads are filed.
  - `outputs/pl1/<d>/<v>/layout.{npz,json}`: S1's pair tables and gate G-L (`zpl1m.py`, committed).
  - `outputs/pl1/<d>/<v>/types/<d>.{npz,json}`: S2b's EM and score types.
  - `outputs/pl1/<d>/<v>/pools/parts/<d>__s1eval__<i>of<k>.{npz,json}` and `.kt.npy`: zpko's own part format.
  - `outputs/pl1/<d>/<v>/places/<d>__s1eval__<i>of<k>.npz`: the reach read, one row per gold. Per question: `ids`, `keep`
    (the cut), `kt_len` (|kt_cos|), `g_ptr`. Per gold: `g_node`, `pos` (0 in B, p >= 1 the p-th node of kt_cos, -1 in no
    order), `in_I` (in the incumbent pool), `one_link` (one link from a B node).
  - `outputs/pl1/<d>/<v>/grade.json` and `outputs/pl1/report.{json,md}`: S5 and S6.
- **Self-test on real-shape data** before any host row (a toy test passed once while eight real fits failed).

## 9. Disclosed

- Written after a read-only survey of the pool code (25 Nov) and after the UL2 grades of MuSiQue, SQuAD, MetaQA and
  MultiHop-RAG were filed; no pool number and no PL1 number was read. The universal call (UL2 12.6 and 13) is not yet
  filed; PL1's rows wait for it.
- No UL2 graph materializer exists yet. The one artifact S1 reads is `outputs/ul2c/<d>/carried.npz` (written by `zul2c.write_carried`; it does not exist before the call is filed); S1 writes the walk stores. An earlier line of this page called that file `carved.npz`, which was my wrong name; corrected 25 Nov before any number.
- The e3v E0 first-stage lists are on the host; whether every dataset's exists is checked by G-E.
- The closing rule's first draft (one improvement round, a 5% OUTSIDE line, frozen "whatever the numbers") was my default;
  the user replaced it on 25 Nov (amendment 2d, section 7). What is still mine in the new rule: the unit of an arm
  (dataset and version), and that an arm's freeze test is an empty candidate list and not a share threshold.
- This page decides nothing about the models, the paper's numbers, the encoder or the hub line.
- **Amendment 2b (25 Nov, after the survey, before any number).** Five things the first draft left out or got wrong, now
  declared: which links each walk family holds per version (2b; the SRK structural slot is S, with W2_S moved into the ner
  family, which is A1's layout and the most arguable choice here); that the walk reads row order and not weight (the carried
  pairs' order is a rule with no constant); that the EM and score types are refitted per (dataset, version) (step S2b); that
  E0's e3v lists exist for the s1eval rows only, so the build is the eval carves (fit and s1sel pools are the refit's);
  and that `pos = -1` means "in no order", not "no link path" (OUTSIDE, not NOPATH).
- **Amendment 2c (25 Nov, before any number).** The first draft said the pools read the vectors as int8; the shards on the
  host are fp16 and the pool code reads shards, so the pools read those and the int8 conversion is a space step after the
  pools are final. The file names of every PL1 output are fixed in section 8.
- **Amendment 2d (25 Nov, before any number).** The user: "i dont want to freeze pools for the sake of it, make sure we
  freeze when we know it is the best for that specific arm". Section 7 now freezes per arm (nine arms), by an empty
  candidate list with its evidence named, with no cap on the number of improvement rounds. One tension is named, not
  hidden: the rules stay universal (section 2a), so an improvement round is graded and carried on all arms at once, and a
  frozen arm is rebuilt and graded again when one carries; a per-arm freeze is of the diagnosis, not a per-arm rule. The
  one-call ADOPT_POOLS of section 6 is unchanged (it asks whether the new layout's pools replace the pools of record).
