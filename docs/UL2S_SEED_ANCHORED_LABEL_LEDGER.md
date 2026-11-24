# UL2S: the seed-anchored label ledger on the e3v layout, all seven datasets (a scouting stage)

Declared 25 November 2026, before any UL2S number. Results are filed in section 10, after the runs.

It follows:
- UL2P ([UL2P_LABEL_PRECISION_E3V.md](UL2P_LABEL_PRECISION_E3V.md); filed 0eeeea6): the gold-anchored ledger, MultiHop-RAG read, SQuAD n/a (its D-P1), five datasets waiting for host rows;
- AP4 ([AP4_LABEL_PRECISION_LEDGER.md](AP4_LABEL_PRECISION_LEDGER.md); filed c72372d): found and top1 anchors on the first stage's top five;
- UL2 ([UL2_LAYOUT_ON_E3V.md](UL2_LAYOUT_ON_E3V.md)): the e3v layout, its labels and forms.

Code: `outputs/mp_unified/zul2s.py`, new. It imports `zul2p`, `zul2c`, `zul2u`, `zul2`, `zul1`, `zul2m`, `zap4` and `zenc` and edits none of them. Outputs go only under `outputs/ul2p/seed/`; arrays (npz) are never written. `outputs/pl1/` and the host feed are read, never written.

**What UL2S does not touch:**
- Nothing trains and no model reads anything. No LLM, no spaCy.
- No pool, link, column, fit or record of record is written. The pool-anchored ledger is PL1's (its M4); this page does not repeat it.
- **UL2S decides nothing:** no carry, no adoption, no weights. It measures, and names candidate rule changes (section 7).

## 1. Why

**The user, 25 Nov (verbatim, relayed by the main session):** "for all datasets find the gold links that directly give the answer and compare with those like we did for musique etc, we have to do the same for all datasets and improve the rules do it for all 7".

**Before that, about MultiHop-RAG's pools (verbatim):** "No coverage read of these pools has been filed yet / why not". That is PL1's. Its post-hoc read of the actual pools (relayed): dev any-gold 100.0% and all-gold 93.3% (15 of 224 questions miss a gold); test any-gold 99.8% and all-gold 94.9% (58 of 1,130).

**What UL2P left open.**
- It conditions on the anchor being a gold: a slot's two ends. A model does not know the golds; it has the first stage's rows.
- It has no cell for SQuAD (one gold per question, so no gold pair). The user's "all 7" is the yes to UL2P's D-P1 option: seed anchors.
- It says how precise each label is between golds, not which golds the rules never reach from what the first stage found.

**What this page measures:** from the first stage's top five rows (the seeds), which of the golds the first stage missed does each label, group, form and the any-link unit reach, at what cost in candidates, and which golds no rule reaches at all (section 4's classes). On all seven datasets.

## 2. The seeds, and how this differs from AP4, UL2P and PL1

- **Seeds** are the top five rows of the record's first stage: `zenc.rrf_top` of the e3v dense list and the served SPLADE list (the call E0's `h_rrf5` and PL1's pools use). Beside it, the top five of the e3v dense list alone. The rrf list is the one that decides; the dense list is read beside it with the core measures only (section 4).
- **Whole graph, no pool.** A candidate is any linked row of the e3v layout outside the top five, not a pool row. The base rate is over the n - 5 rows outside the top five.
- **AP4:** the same anchors (found, top1), the same idea; pools replaced by the whole graph, the layout and labels are e3v's (UL2), the forms are 12.6's.
- **UL2P:** the same units and the same label set; the anchor is a seed, not a slot end.
- **PL1's M4:** the same anchors on the pool rows; this page has no pool.
- **This is a different instrument from each of them.** Its numbers are compared with theirs only as the written expectations in section 7.

## 3. What is read

| dataset | version(s) | population (the s1eval carve; E0's questions) | where it runs |
|---|---|---|---|
| MultiHop-RAG | RK | test, 1,130 questions with a gold (UL2 13.2); dev beside | laptop |
| SQuAD | RK | 11,873 | laptop, after two input copies from the host (below) |
| MuSiQue | RK | 2,417 | host |
| MetaQA | SRK | E0's questions | host |
| WebQSP | SRK | E0's questions | host |
| HotpotQA | RK and SRK | E0's questions | host, after its e3v K |
| 2Wiki | RK and SRK | E0's questions | host, after its e3v K |

**Read from each dataset** (all by code that is already filed):
- the layout: UL2P's loaders (`zul2p.load_host`, `zul2p.load_mhr`): families, K_e3v, the forms' ranks (12.6's `ranks.npz`), all gated against 12.6's grade (UL2P's G-P1 to G-P3);
- the first stage: `outputs/enc/e0/e3v/<d>.npz` (E0's lists: ids, top_ids, gold_total, h_d5, h_rrf5) and the served SPLADE list (`retrieval_cache/splade_top1000.npz`); MultiHop-RAG: PL1 S0a's complete lists (`outputs/pl1/multihoprag/RK/s0/first_e3v.npz`) and the package's SPLADE list;
- the golds: `outputs/enc/e0/gold/<d>.npz` (E0's gold cache); MultiHop-RAG's `s0/gold.npz`, the same shape.

**Gates (a mismatch stops the run):**
- G-P1 to G-P3: UL2P's, unchanged: B0's size, each form's pairs beyond B0, the slots and covers equal the grade's;
- G-S1: the first stage's question ids equal the gold cache's, and `gold_total` equals the gold sets' sizes;
- G-S2: the dense top five's gold hits equal E0's `h_d5`, per question;
- G-S3: the rrf top five's gold hits equal E0's `h_rrf5`, per question (MultiHop-RAG: the lists' and the gold file's sha256 equal PL1 S0a's `first.json`, and the golds equal `zul2m.golds_splits`);
- G-S4: the population equals UL2's s1eval carve (`zul2.slot_kinds`' ids, in order);
- G-S5: every question has five distinct seed rows in [0, n).

**Two input copies for SQuAD on the laptop** (read-only, no number): `outputs/enc/e2/knn/e3v/squad.npz` (and its `.json`) and `outputs/ul2c/squad/ranks.npz`, copied from the host by `rx fetch`, checked by sha256 against 12.6's `grade.json`. If a copy is not possible, SQuAD runs on the host with the rest.

## 4. The measures

**Anchor kinds** (per list: the rrf top five, and beside it the dense top five):

| kind | anchors | targets |
|---|---|---|
| seed | each of the question's five seed rows | the question's golds outside the five (the missed golds) |
| top1 | the first seed row | the same |
| found | each gold inside the five (AP4's found) | the same |

**Candidates** of an anchor: the unit's pairs at its row, in either direction, with the other end outside the question's five. A question counts when it has a gold. A question whose golds are all inside the five has no target but its candidates count: a deployed link follower does not know that.

**Per unit, kind and list** (every measure is a ratio of sums over the questions that have an anchor of the kind):

| measure | definition | call |
|---|---|---|
| price per anchor | candidates / anchors | |
| precision | targets / candidates | |
| base | the expected precision if every candidate were a uniform row outside the five: sum of (candidates x missed golds / (n - 5)) over questions, divided by candidates | |
| lift | targets / expected targets (precision / base) | **POINTS** when the low end is above 1; **BELOW_BASE** when the high end is below 1; **NO_LIFT** otherwise (AP4's) |
| reach | missed golds that are a candidate of some anchor through the unit / missed golds | |
| question reach | questions with a missed gold in which at least one is reached / questions with a missed gold (and, beside, all of them) | |
| graph price | the unit's pairs / missed golds reached (12.6's price, in pairs per gold) | |
| vs any | the unit's precision minus the any-link unit's, paired | ABOVE, LEVEL or BELOW |

**Intervals:** 95% percentile intervals over 2,000 resamples of the questions with replacement (`zap4.boot`: `default_rng(0)`, chunks of 100). The paired differences use the same draw. **Core measures** (the dense list beside the rrf list): price, precision, reach, graph price and their intervals; no lift call, no vs any, no exact sets.

**Not measured:** rank1, top2 and ranks by cosine (a rank needs a score on every candidate; E0's lists hold the top 1,000), and any pool measure.

**The classes of the missed golds** (seed kind, each list; exclusive, in this order; a share of the missed golds, with an interval over questions):

| class | meaning |
|---|---|
| SEED_B0 | a B0 pair joins the gold to a seed row: the rules find it |
| SEED_FORM | not that, but a pair of one of 12.6's forms (beyond B0) joins it to a seed row |
| GOLD_LINKED | no seed link, but a B0 or form pair joins it to another gold of the question: **the link that gives the answer exists, and the first stage has no anchor** (both golds missed) |
| LINKED_ELSE | it has B0 or form pairs, none to a seed row or a gold |
| ISOLATED | it has no B0 pair and no form pair |

The last two are **the golds no rule reaches**. A dataset with one gold per question has no GOLD_LINKED.

**The cut frontier.** The any-link pairs split by their exact label set (the sets are disjoint). In descending precision, the cumulative candidates, targets and reach as sets are added: points only. It is the trade a tighter rule would make.

**MuSiQue by hops (2, 3, 4):** every unit, core measures.

## 5. The units

As UL2P section 5, unchanged: labels (each pair under every label it carries, the S relations as `S.<rel>` with the 12 most frequent at the seeds, the rest `S.other`), the groups by the dedup S > R > K, the any-link unit B0, the exact label sets, multi minus single, the seven forms beyond B0 and their rank quintiles. One unit added for the classes and the frontier: **all** = B0 plus every form's pairs.

## 6. What it can and cannot show

- **Graph-level reach is not R@5.** A link that reaches a gold adds nothing when a model already ranks it; a poor link can help in combination. UL2S names candidates; reads decide.
- **The seeds are the record's first stage.** The models' own top rows differ.
- **Precision has no pool ceiling.** A seed on a hub gives 10^4 candidates; that is the label's real price at that row, and the point of the measure.
- **SQuAD, plainly.** SQuAD has one gold passage per question and no multi-hop structure. A link there can show only whether a retrieved wrong passage is one rule-step (a shared entity, a name co-mention, a nearest neighbour) from the answer passage, and at what candidate price. It cannot show a hop between golds: there is no `found` target (a found anchor has no partner) and no GOLD_LINKED class. SQuAD's `found` kind and GOLD_LINKED are reported n/a. The `seed` and `top1` kinds are the whole read there.
- **Pool coverage is PL1's.** A gold reached here may be outside the pool; a gold missed here may be inside it.
- **The universal rule.** Every candidate named in section 7 is read on all seven datasets and is carried everywhere or nowhere.

## 7. The calls, the expectations and what follows (fixed before any number)

**Expectations.** Written before any UL2S number, **but not blind**: UL2P's MultiHop-RAG read and AP4's reads are known. They decide nothing; the report marks each HELD, NOT_HELD or NOT_READ. All are on the rrf list, seed kind, unless said.

| id | expectation |
|---|---|
| E1 | The any-link unit POINTS, and every B0 label with at least 20 targets POINTS, on every dataset and version. |
| E2 | Multi-label pairs are ABOVE single-label pairs in precision on every dataset except MetaQA, where they are not ABOVE. |
| E3 | Every form beyond B0 is BELOW the any-link unit in precision wherever it has at least 20 targets. |
| E4 | S has the highest precision of the labels on MetaQA and WebQSP. |
| E5 | On MuSiQue the any-link reach falls with the hops: 2 > 3 > 4. |
| E6 | On every dataset with more than one gold, the any-link precision at the found anchors is ABOVE its precision at the top1 anchor (the found interval's low end above the top1 interval's high end). |
| E7 | GOLD_LINKED is under 5% of the missed golds on every dataset with more than one gold: the first stage's gap is not a link gap. |

**What follows, named now (names only; each is read on all seven):**
- **Candidate label weights:** a label that POINTS and is ABOVE the any-link unit in precision. A label is **universal** only when it is ABOVE on every dataset where it has at least 20 targets; the report counts. AP4's D92 stays default no.
- **The agreement candidate:** multi minus single ABOVE on every dataset (count reported).
- **Candidate tighter cuts:** a form BELOW the any-link unit as a whole whose first quintile is ABOVE it.
- **Cost without precision (a flag, a drop or cap candidate):** a label holding at least 5% of the any-link candidates at the seeds that is BELOW the any-link unit in precision, or has no lift (NO_LIFT, BELOW_BASE).
- **The rules' gap:** the shares of ISOLATED plus LINKED_ELSE (golds no rule reaches) and of GOLD_LINKED, per dataset. Where the first is large, only a new rule family could reach those golds; where GOLD_LINKED is large, the gap is the first stage's. A new family would need no model, no LLM and no spaCy; none is named here.
- **Hubs are closed (the user, 24 Nov):** no hub step is named, whatever the numbers show.
- **No refit, no carry, no new form.**

## 8. Rows and runs

- **Laptop:** `python outputs/mp_unified/zul2s.py selftest`; then `ledger --ds multihoprag` and `ledger --ds squad`.
- **Host, one row per dataset** (MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki): `python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/mp_unified/zul2s.py ledger --ds D --host --threads 1`.
  - 1 CPU each. Memory is an estimate, re-asked at the measured peak.
  - Each row depends on the dataset's `ul2c-grade-D` row. HotpotQA and 2Wiki therefore wait for their e3v K.
  - The first stage's lists and the gold cache must be on the host; the served SPLADE lists are in the host's package.
- **`ul2s-report`** reads the ledgers and writes `outputs/ul2p/seed/report.json` and `report.md`.
- **No row is added to the feed by this stage's author;** the exact lines are given to the session that owns the feed. The module is committed before they are sent.

## 9. Decisions (the user's; my default first)

- **D-S1, which list decides.** The rrf top five (the record's first stage). **Default: rrf decides;** the dense top five is read beside it with core measures.
- **D-S2, the pool-anchored ledger on the same units.** PL1's M4 is its home. **Default: not repeated here.**
- **D-S3, a carried arm.** 12.6's carry-by-budget is unfiled. **Default: when it is filed, a ledger of its arm is a new section here.**

## 10. Results

**Filed 25 Nov 2026 for MultiHop-RAG and SQuAD only (laptop).** MuSiQue, MetaQA, WebQSP, HotpotQA and 2Wiki are not read: their rows run on the host and `ul2s-report` has not been run on them. **Nothing below is a carry, an adoption or a rule change.** Every candidate in 10.5 waits for the other five (the universal rule). Declaration `3641b2c`, code `9b067ea` (committed before any number). The ledgers are `outputs/ul2p/seed/multihoprag/ledger.json` and `outputs/ul2p/seed/squad/ledger.json`; `report.md` holds every table. It prints percentages to two decimals, which hides SQuAD's precisions (0.04 [0.04, 0.04]); the numbers below are the ledgers' to three. Seeds are the rrf top five; the version is RK on both (S is empty on both: no S pairs). The numbers are percent unless a count.

### 10.1 The runs

- Gates G-P1..G-P3 and G-S1..G-S5 passed on both; the self-test had passed before either run.
- **MultiHop-RAG** (n = 609): 1,130 test questions, 2,985 golds; dev (s1sel) 224 questions beside. The rrf top five finds 1,741 golds and misses 1,244 (783 questions); the dense top five finds 1,828 and misses 1,157. 55.8 s, 0.09 GB peak.
- **SQuAD** (n = 20,233): 11,873 questions, one gold each. The rrf top five finds 10,478 and misses 1,395; the dense top five finds 9,937 and misses 1,936. 677 s, 0.7 GB peak. `found` and GOLD_LINKED are n/a (section 6): the `seed` and `top1` kinds are the whole read.

### 10.2 The expectations (rrf list, seed kind; only the two datasets read)

| id | call | reads | what decided it |
|---|---|---|---|
| E1 | HELD | 2 of 2 | the any-link unit and every B0 label with at least 20 targets POINTS on both (labels under 20 targets are not read: MultiHop-RAG R.mention 2, R.title_comention 10; SQuAD R.title_comention 0) |
| E2 | HELD | 2 of 2 | multi minus single ABOVE: MultiHop-RAG +6.807 [6.248, 7.402] points; SQuAD +0.115 [0.099, 0.131]. MetaQA, the declared exception, is not read |
| E3 | **NOT_HELD** | 8 of 14 | MultiHop-RAG: all 7 forms BELOW any-link. SQuAD: only NTc BELOW; H5, NT1, HK2, H10 LEVEL; HM and HK1 ABOVE |
| E4 | NOT_READ | 0 | MetaQA and WebQSP are host rows |
| E5 | NOT_READ | 0 | MuSiQue is a host row |
| E6 | **NOT_HELD** | 0 of 1 | MultiHop-RAG: found 0.797 [0.739, 0.855] against top1 1.159 [1.084, 1.241]: found is BELOW top1, the opposite of AP4's pattern. The dev cell has the same direction (found 0.722 [0.603, 0.851], top1 1.067 [0.908, 1.245]). SQuAD n/a |
| E7 | HELD | 1 of 1 | MultiHop-RAG GOLD_LINKED 0.00% (nearly vacuous, 10.6). SQuAD n/a |

E3 was written from the MultiHop-RAG read and fails on SQuAD because SQuAD's any-link unit is dominated by R.mention at 0.042% precision, a low bar the forms clear or match. E6's miss is not explained by this stage.

### 10.3 MultiHop-RAG

- **Any-link:** 25,343 pairs, 478,732 candidates (84.7 per anchor), 5,274 targets, precision **1.102 [1.037, 1.169]**, lift 6.21 [5.99, 6.42], reach **97.83 [96.91, 98.66]** of the missed golds (question reach 97.8%).
- **Labels:** R.name_comention holds 99.04% of the candidates (1.106 [1.040, 1.174], vs any +0.004 [0.001, 0.007]). **K.knn** holds 4.29%: precision **6.521 [6.038, 7.063]** (lift 34.3), vs any **+5.420 [4.965, 5.916]**, reach 55.55 [53.06, 58.02] at 2.0 pairs per gold reached (R.name_comention: 20.6). R.mention has 2 pairs and 2 targets (not read); R.title_comention 75 pairs, 10 targets, LEVEL. K-only pairs (group K, beyond R) are BELOW any-link: 0.623 [0.330, 0.944] on 3,371 candidates.
- **Agreement:** the exact set R.name_comention + K.knn is 17,169 candidates (3.59% of the any-link candidates), 1,319 targets, precision **7.682 [7.114, 8.312]**; R.name_comention alone is 456,922 candidates at 0.858 [0.804, 0.916] (BELOW any-link). Agreement is about 9 times the single-label bulk. The first frontier tier alone reaches 54.58% of the missed golds; the single-label bulk adds the rest to 97.51%.
- **Forms:** all seven BELOW any-link as wholes (H5 0.436, HM 0.681, HK1 0.381, NTc 0.114, NT1 0.187, HK2 0.285, H10 0.356). **NTc** holds 230,851 candidates (48.2% of the any-link count) at lift BELOW_BASE, and NT1 has NO_LIFT. Within a form the first rank quintile is ABOVE the last on six of seven (NTc LEVEL). **HM's first quintile 1.831 [1.444, 2.239] (115 targets) is ABOVE any-link**; the other forms' first quintiles are not. The pseudo-unit **all** (B0 + every form) has 1.83 times the candidates, precision 0.666 (BELOW any-link by 0.436 [0.404, 0.469] points) and reach 99.60: +1.77 points of reach for 398,077 more candidates.
- **The classes of the 1,244 missed golds:** SEED_B0 97.83 (1,217), SEED_FORM 1.77 (22), LINKED_ELSE 0.40 (5), GOLD_LINKED 0, ISOLATED 0.
- **Beside:** the dense list gives any-link 1.046 [0.977, 1.115], reach 97.41, K.knn 5.323 (core measures, within the rrf list's intervals). The dev cell: any-link 0.981 [0.846, 1.134], K.knn 5.367, multi minus single +5.470; no call reverses in direction, and three differ by one step (R.name_comention vs any-link: ABOVE by 0.004 on the test, LEVEL on dev; HM: BELOW on the test, LEVEL on dev; R.mention, with 0 to 2 targets). It does not contradict UL2P's MultiHop-RAG read (K.knn ABOVE, name co-mention the bulk).

### 10.4 SQuAD, plainly

One gold per question: the read says whether a retrieved wrong passage is one rule step from the answer passage and at what candidate price. It is a meaningful read (base about 0.0007%, lift 57.8), not a hop.

- **Any-link:** 995,869 pairs, 4,985,869 candidates (84.0 per anchor, about 420 per question: 2% of the corpus before the five anchors' overlaps are removed), 2,138 targets, precision **0.043 [0.039, 0.046]**, lift 57.8 [53.7, 62.1], reach **51.83 [49.27, 54.43]** of the 1,395 missed golds, at 1,377 pairs per gold reached. At the top1 anchor the precision is 0.054 [0.049, 0.058].
- **Labels:** R.mention holds 85.67% of the candidates at 0.042 [0.038, 0.045] (LEVEL vs any-link, as the bulk) and reaches 42.72, at 1,473.5 pairs per gold. R.name_comention holds 20.37% (labels overlap), precision 0.073 [0.065, 0.082], **ABOVE +0.030 [0.024, 0.038]**, reach 25.52. **K.knn** holds 4.31%: precision **0.206 [0.184, 0.230]** (lift 323), **ABOVE +0.164 [0.142, 0.185]**, reach 24.88 [22.70, 27.16] at **124.9 pairs per gold**. R.title_comention: 12,619 candidates, **0 targets**, BELOW_BASE, BELOW any-link. K-only pairs (group K): 0.176 [0.131, 0.224], ABOVE any-link (on MultiHop-RAG they are BELOW).
- **Exact sets:** R.mention alone is 76.49% of the candidates (3,813,777) at 0.029 [0.027, 0.033], BELOW any-link, and 1,123 of the 2,138 targets. The multi-label sets: R.mention + R.name_comention + K.knn 59,757 candidates at **0.241 [0.198, 0.288]**; R.mention + K.knn 109,303 at 0.193; R.mention + R.name_comention 276,545 at 0.110; R.name_comention + K.knn 10,170 at 0.275. Keeping the two single-label bulk sets out (R.mention alone, R.name_comention alone) leaves 490,501 candidates (9.8%) at 0.152 (3.5 times any-link) with reach 31.61, i.e. 61% of the 51.83.
- **Forms:** as wholes only NTc is BELOW any-link (0.027 [0.021, 0.034]); HM (0.068) and HK1 (0.057) are ABOVE; H5, NT1, HK2, H10 are LEVEL. The first rank quintile of **every** form except NTc is ABOVE any-link (0.081 to 0.129 against 0.043). **all** has 1.29 times the candidates, precision 0.040 (BELOW any-link by 0.003 [0.001, 0.004]) and reach 61.79: **+9.96 points of reach (139 golds)** that only a form reaches.
- **The classes of the 1,395 missed golds:** SEED_B0 51.83 (723), SEED_FORM 9.96 (139), **LINKED_ELSE 38.21 [35.63, 40.79] (533)**, ISOLATED 0. Every gold has pairs; for 38% none of them touches the five seeds.
- **Beside:** the dense list (1,936 missed) gives any-link 0.059 [0.055, 0.063], reach 51.65, K.knn 0.309, classes 51.65 / 10.49 / 37.86.

### 10.5 What the two reads point to (named; none adopted; each waits for the other five)

1. **Label count as a weight or a tier (agreement).** Pairs carrying two or more labels are about 9 times (MultiHop-RAG, 7.682 against 0.858) and about 8 times (SQuAD, 0.241 against 0.029) as precise as the bulk single-label pairs, on 2 of 2 datasets read. The dedup S > R > K keeps one label per pair for the walk store; the ledger shows the agreement it discards is the strongest signal either dataset has. Whether the model's edge features keep the count is to be checked in the code before any claim. As a cut that drops the single-label bulk it trades reach: MultiHop-RAG keeps the two multi-label sets (17,230 candidates, 7.667, reach 54.74 of 97.83); SQuAD keeps every set but R.mention alone and R.name_comention alone (490,501 candidates, 0.152, reach 31.61 of 51.83): 28 and 10 times fewer candidates. As a weight or a tier it loses none. A model read decides (section 6). **E2 carries; MetaQA is the declared exception.**
2. **A K.knn weight** (ABOVE any-link and POINTS on 2 of 2 arms with at least 20 targets: UNIVERSAL so far). Points above any-link: +5.420 and +0.164. K is the cheapest label in graph price on both (2.0 and 124.9 pairs per gold, against 20.6 and 1,473.5 for R) and holds 4.3% of the candidates. K-only pairs are not universal (SQuAD ABOVE, MultiHop-RAG BELOW): on MultiHop-RAG K's precision is its agreement with R.name_comention.
3. **An R.name_comention weight** (ABOVE on 2 of 2, UNIVERSAL by the declared rule). Read it with care: on MultiHop-RAG it is 99% of the candidates and +0.004 points above the unit it dominates, which has no content. On SQuAD the margin is real (+0.030).
4. **A tighter cut: HM's first quintile** (HM BELOW any-link as a whole on MultiHop-RAG, first quintile 1.831 against 1.102). It meets the declared definition on MultiHop-RAG only; on SQuAD HM is ABOVE as a whole and its first quintile is ABOVE too (0.117 against 0.043). The first quintile of the other forms is ABOVE any-link on SQuAD and not on MultiHop-RAG. **No form tier is above on both except HM's.**
5. **Cost without precision (the declared 5% flag): none.** R.mention is 85.7% of SQuAD's candidates but LEVEL with POINTS, not BELOW. Beside the flag, as observations: **NTc** is BELOW any-link on both and BELOW_BASE on MultiHop-RAG, so if forms are ever added to B0 it is the one that costs without precision; **R.title_comention** has no target among its 12,619 candidates on SQuAD and a LEVEL 10 targets on MultiHop-RAG, at 0.25% of the candidates on both: no case to keep it or drop it from these reads.
6. **The rules' gap.** MultiHop-RAG: 0.40% of the missed golds are reached by no rule (5 golds): the rules reach what the five seeds' neighbourhoods contain; the gap is not a link gap. **SQuAD: 38.21% (533 golds) are linked to something but to no seed.** No rule family tested here reaches them from the five seeds; this stage does not measure a two-hop path (anchor to neighbour to gold), nor a deeper anchor set. No new family is named.

### 10.6 What to distrust

- **MultiHop-RAG's reach is a small-corpus effect.** Each anchor has 84.7 candidates in 604 other rows; five anchors cover up to 70% of the corpus before their overlaps are removed. 97.8% reach is therefore not an enrichment; the precision and lift (6.2 times base) are the reads. For the same reason E7 held almost vacuously: only 5 of the 1,244 missed golds have no seed link of any kind (B0 or form), and none of those is linked to another gold.
- **SQuAD's R.mention is the bulk of everything** (76.5% of the candidates as an exact set), so every SQuAD comparison against any-link is a comparison against R.mention.
- **Graph-level reach is not R@5** (section 6): a gold a model already ranks gains nothing from a link, and a weak link can help in combination.
- **Two datasets of seven.** The universal rule needs the other five: MetaQA is the declared E2 exception, and MetaQA, WebQSP, HotpotQA and 2Wiki carry S pairs, which neither dataset read here has (S is empty on both).
- **E6 is unexplained.** The found anchors are less precise than the top1 anchor on MultiHop-RAG, in both the test and the dev cell.

### 10.7 Still to read

The five host rows (`ul2s-musique`, `-metaqa`, `-webqsp`, `-hotpotqa`, `-2wiki`) and `ul2s-report`, which rewrites `report.json` and `report.md` over all seven (the laptop copies filed with this section cover two datasets and are replaced). E4, E5 and MuSiQue's hops are read there. HotpotQA and 2Wiki wait for their `ul2c-grade` rows. The universal test of 10.5 (every candidate carried everywhere or nowhere) is run on the seven, and nothing is adopted before it.

## 11. Development log (disclosed before any UL2S number)

- No `zul2s` code existed before this page. Before it, only file shapes and sizes were inspected (the keys of E0's lists and gold caches, PL1 S0a's files, grade.json fields), and a probe that loaded SQuAD's and MuSiQue's families on the laptop printed counts of pairs and stopped at the missing K_e3v. No UL2S measure was computed.
- UL2P's code (`zul2p.py`) and its filed numbers are unchanged.
- `zul2s.py` was committed (`9b067ea`) before any number. Its self-test passed on its first run; slips found while assembling it (a malformed conditional in `list_record`, a convoluted one-way evaluator, a tuple argument in `exact_record`, a loose assertion in `universal`, a d5 expression) were fixed before that run.
- SQuAD's inputs were copied from the host by `rx fetch` and checked by sha256 (`outputs/enc/e2/knn/e3v/squad.npz` and `.json`, `outputs/ul2c/squad/ranks.npz`); they are untracked local copies.
- MultiHop-RAG's first stage is PL1 S0a's `first_e3v.npz` and `gold.npz` (sha256 against `first.json`) with the package's SPLADE list; the dense top five's hits must equal the filed d5 of 64.29 (an added gate). Its dev cell (s1sel) exists beside the test cell only there: E0's first stage holds s1eval alone for the six host datasets.
- MuSiQue's hops are read on the rrf seed only.
- After the runs `report` printed percentages to two decimals; the section 10 numbers come from the ledgers to three. No code changed after the numbers.

## 12. Files

- `outputs/mp_unified/zul2s.py`
- `outputs/ul2p/seed/<dataset>/ledger.json`: the ledger and its record (shas of the grade, ranks, first-stage lists, gold cache; no array)
- `outputs/ul2p/seed/report.json` and `report.md`
