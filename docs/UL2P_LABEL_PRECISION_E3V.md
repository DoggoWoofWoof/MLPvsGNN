# UL2P: the label precision ledger on the e3v layout (a scouting stage)

Declared 25 November 2026, before any UL2P number. Results are filed in section 10, after the runs.

It follows:
- AP4 ([AP4_LABEL_PRECISION_LEDGER.md](AP4_LABEL_PRECISION_LEDGER.md); filed c72372d): per-label precision on the older D83 layout, six datasets;
- UL2 ([UL2_LAYOUT_ON_E3V.md](UL2_LAYOUT_ON_E3V.md); sections 2, 8, 12.1-12.6 and 13): the e3v layout, its labels and its slots, seven datasets.

Code: `outputs/mp_unified/zul2p.py`, new. It imports `zul2c`, `zul2u`, `zul2`, `zul1`, `zul2m` and `zap4` (the bootstrap, the ratio, the two calls) and edits none of them. Outputs go only under `outputs/ul2p/`; arrays (npz) are never written.

**What UL2P does not touch:**
- Nothing trains and no model reads anything.
- No pool, link, column, fit or record of record is written. 12.6's carry-by-budget is unfiled, so there is no carried arm here.
- **UL2P decides nothing:** no carry, no adoption, no weights. It measures, and names candidates (section 6).

## 1. Why

**The user, 25 Nov (verbatim):** "we have a proper pool or not, because that is the first thing i wanted, proper rules completed with dedupe, removal of spacy, comparing with gold or oracle with all the datasets and improvement of the link based rules and finishing up on that part with proper precision for each dataset".

**The user, 16 Nov (verbatim, AP4's reason):** "the edges should also have labels based on how it was created so that we can unify with KB labels and make the model train and understand better rather than just giving family names".

**What is open.**
- UL2 graded reach and price in pairs (slots covered, pairs added). It did not ask how precise each label's links are.
- AP4 asked that on the D83 layout (spaCy NER in R, the frozen K, no forms). It did not cover MultiHop-RAG and was not run on the e3v layout.
- This page measures it on the e3v layout, with the dedup, without spaCy, for all seven datasets, against the gold or oracle pairs.

## 2. What AP4's measures cannot carry over

AP4 anchored every measure on the record's pools: the top five of rrf, candidates outside the top five, the base rate from golds outside the top five, rank1 among an anchor's candidates by rrf. **The e3v layout has no pools or rrf columns.** The layout is graph level (UL2 section 2), and pools are a later step that gets its own declaration.

So UL2P changes the anchor, not the idea:
- **gold-anchored, pool-free:** an anchor is a slot endpoint, and a target is the slot's other endpoint;
- **not measured:** rank1, top2, rank by cosine, found reach, the top1 anchor. All of them need a first-stage ranking.

**This is a different instrument from AP4's, not a replication.** Its numbers are not compared with AP4's, except as the written expectations in section 6.

## 3. What is read

**The arm: B0, the e3v layout before any form** (UL2 12.6): S (SRK only) + W2_S + F0T + K_e3v. The forms beside it are the seven of 12.6, each alone beyond B0 (section 5).

| dataset | versions | cells (s1eval decides; s1sel is beside) | where it runs |
|---|---|---|---|
| MultiHop-RAG | RK | GP, s1eval = test, s1sel = dev (13.2) | laptop |
| MuSiQue | RK | GP and CH | host |
| SQuAD | RK | none: one gold per question, so no slot | **n/a** (section 9, D-P1) |
| MetaQA | SRK | GP, 2 to 10 golds | host |
| WebQSP | SRK | GP, 2 to 10 golds, s1eval only | host |
| HotpotQA | RK and SRK | GP | host, after e3v K |
| 2Wiki | RK and SRK | GP | host, after e3v K |

**Read from each dataset** (all by code that is already filed):
- the families (`zul1.families`: S with its relation labels, W2_S, CN_all, F0T) and K_e3v (`zul2.knn_keys`);
- the carve questions and slots (`zul2.slot_kinds`): GP's gold pairs and MuSiQue's chain hops;
- the forms' ranked pairs beyond B0, from 12.6's grade (`outputs/ul2c/<d>/ranks.npz`; MultiHop-RAG's `outputs/ul2m/multihoprag/ranks.npz`), checked by the sha256 in its `grade.json` and by 12.6's build code.

**Gates (a mismatch stops the run):**
- G-P1: B0's size equals the grade's filed B0 per version, and REC's covered slots equal the grade's filed counts;
- G-P2: for every form, the pairs beyond B0 equal the grade's `pairs_beyond_B0` count for that form;
- G-P3: the slots' counts and the questions' sha256 equal the grade's cells.

## 4. The measures

**Anchor occurrences.** One per (question, node), for each endpoint of each of the question's slots in the cell.

**Candidates.** A unit's pairs at the anchor's node, in either direction. This includes links to golds of other questions and, on MuSiQue's CH, to golds of the same question that are not that hop's partner.

**Targets.** A candidate that is the slot's other endpoint. A covered slot gives two targets, one per endpoint.

**Per unit, cell and version** (every measure is a ratio of sums over the questions that have a slot in the cell):

| measure | definition | call |
|---|---|---|
| pairs | the unit's size in the whole graph | |
| price per anchor | candidates / anchors | |
| precision | targets / candidates | |
| reach | slots covered / slots | |
| graph price | pairs / slots covered (12.6's price, compare P* = 4,997.1) | |
| base | 2 x slots / (anchors x (n - 1)): a uniform partner among all n nodes | |
| lift | precision / base | **POINTS** when the interval's low end is above 1; **BELOW_BASE** when its high end is below 1; **NO_LIFT** otherwise (AP4's) |
| vs any | the unit's precision minus the any-link unit's, paired | ABOVE, LEVEL or BELOW |

**Intervals:** 95% percentile intervals over 2,000 resamples of the questions with replacement (`zap4.boot`: `default_rng(0)`, chunks of 100). The paired differences use the same draw (same seed, same number of questions).

**What this base rate means.** The base is over all n nodes, not a pool. Lifts are large (hundreds to millions) and say little beyond POINTS; the lifts are compared across labels within a dataset only. Precision has no pool ceiling: a gold node with 10^4 neighbours through a unit contributes 10^4 candidates. That is the unit's real price at that node, and it is the point of the measure.

## 5. The units

**Labels, each pair under every label it carries** (the labels of UL2 section 8, no dedup):
- `S.<rel>` per store relation (SRK only): the 12 relations with most pairs at the cell's anchors, the rest pooled as `S.other`;
- `R.mention` (W2_S), `R.name_comention` (CN_all), `R.title_comention` (F0T not in CN_all), `K.knn` (K_e3v).

**Groups, by the dedup S > R > K:** S; R (W2_S and F0T, minus S); K (K_e3v, minus S and R).

**Any link:** B0's union (S in SRK).

**Exact label sets, on the any-link pairs.** The bits are S (any relation), R.mention, R.name_comention, R.title_comention and K.knn. The most frequent sets are listed, until 99% of the pairs at the anchors are listed (at most 12).

**Multi minus single.** The precision of pairs with two or more labels minus that of pairs with one, paired (ABOVE, LEVEL or BELOW).

**The forms, each alone beyond B0** (in SRK also beyond S): H5 (R.hub), HM (R.hub), HK1 (R.hub_key), NTc (R.written), NT1 (R.written), HK2 (R.hub_key), H10 (R.hub). Labels are 12.6's.

**Rank quintiles of each form.** Its pairs beyond B0 are taken in the form's own rank order (12.6's) and split into five parts of equal count, first = highest ranked. Each part is a unit. They show where along the build's order the precision sits, which is what cutting to a budget keeps.

**MuSiQue by hops (2, 3, 4):** every unit, GP and CH.

**A unit with no pair at the cell's anchors** is listed as such, with its pairs, and has no precision.

## 6. The calls, the expectations and what follows (fixed before any number)

**Expectations.** They are written before any UL2P number, **but not blind**: AP4's filed reads on the older layout are known (section 2). They decide nothing. The report marks each HELD, NOT_HELD or NOT_READ.

| id | expectation |
|---|---|
| E1 | On every dataset, every B0 label with at least 20 targets POINTS. (AP4 saw NO_LIFT for MetaQA's R.name_comention and K.knn, and for SQuAD's R.mention.) |
| E2 | R.title_comention has the highest precision of B0's three R labels on HotpotQA, 2Wiki and MuSiQue, in each version. |
| E3 | S has the highest precision of B0's labels on MetaQA and WebQSP. |
| E4 | Every form beyond B0 is BELOW the any-link unit in precision, on every dataset and version where it has at least 20 targets. The forms were chosen for reach, and their pairs are the tail. |
| E5 | In every form with at least 20 targets in both, the first rank quintile is ABOVE the last in precision (paired). The build's own order carries signal. |
| E6 | Multi-label pairs are ABOVE single-label pairs in precision on MultiHop-RAG, MuSiQue, HotpotQA and 2Wiki, and not ABOVE on MetaQA. (AP4: 8 of 9 reads; MetaQA -5.8 points.) |
| E7 | On MuSiQue the any-link reach on GP falls with the hops: 2 > 3 > 4. |

**What follows, named now:**
- **Candidate label weights.** A label that POINTS and is ABOVE any link is a candidate up-weight, or bias initialization, for a later refit. It is named, not adopted. AP4's D92 stays default no.
- **Candidate tighter cuts.** A form that is BELOW any link as a whole but has a first quintile ABOVE it is a candidate for a tighter cut than 12.6's budget. It is named; 12.6's rule and its carried forms do not change.
- **A flag.** A label BELOW the any-link unit in precision on a dataset where it holds at least 5% of the any-link pairs at the anchors is reported beside that dataset's grade. It changes no rule.
- **No refit, no carry, no new form.**

## 7. Rows and runs

- **Laptop:** `python outputs/mp_unified/zul2p.py selftest`, then `ledger --ds multihoprag`.
- **Host, one row per dataset:** `python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/mp_unified/zul2p.py ledger --ds D --host --threads 1`.
  - 1 CPU each. Memory is an estimate, 8 GB (WebQSP 16 GB), to be re-asked at the measured peak.
  - Each row depends on the dataset's `ul2c-grade-D` row (its `ranks.npz`). HotpotQA and 2Wiki therefore wait for their e3v K.
- **`ul2p-report`** (host or laptop): reads the ledgers, writes `outputs/ul2p/report.json` and `report.md`.
- **No row is added to the feed by this stage's author;** the exact lines are given to the session that owns the feed.

## 8. What it can and cannot show

- **Graph-level precision is not R@5.** A precise label can add nothing when a model already ranks its targets. An imprecise label can help in combination. UL2P names candidates; reads decide.
- **Gold-anchored precision conditions on the anchor being a gold.** A model does not know the golds; it has a first stage's rows. AP4's top1 anchor, a first-stage row, is not reproduced.
- **Pools are a later step.** The pool-anchored ledger (the first stage's rows as anchors) runs after pools are declared, on the same units.
- **The carried arm is not read.** 12.6's carry-by-budget is unfiled; when it is, a ledger of its arm is a new section here.
- **Hubs weigh in as candidates.** Both the label's precision and its graph price are shown, so a precise-per-anchor label that is dear in pairs shows both.

## 9. Decisions (the user's; my default first)

- **D-P1, SQuAD.** SQuAD has no gold pair, so the gold-anchored ledger has no cell there. **Default: n/a,** with its label sizes only (from UL2's grade). The option, not run: seed anchors. SEM2's R2a takes as seeds the top 5 of the dense list and the top 5 of SPLADE (P5b's), with no pool. Those seeds would give AP4's found and top1 anchors on all seven datasets, SQuAD included, but couple this ledger to SEM2's seed files. It needs its own declaration.
- **D-P2, the carried arm.** When 12.6's carry-by-budget is filed, run this ledger on its arm. **Default: yes,** as a new section of this page.

## 10. Results

Filed 25 November 2026. Commits: the declaration e51076c; the code ec07431, then 5fef6ff (the report's section-6 lists, added before filing; the engine and every ledger number unchanged). **One dataset is read of seven: MultiHop-RAG, on the laptop.** SQuAD is n/a. The other five need host rows (10.6). Numbers are from `outputs/ul2p/multihoprag/ledger.json` and `report.md`.

**Development log (disclosed).** The self-test on random graphs and on MultiHop-RAG's real shapes printed no measure. The MultiHop-RAG ledger was first run before the REC covered-slot gate and the section-6 lists were in the code; it was re-run with the committed code and its numbers are the same. Nothing else was read.

### 10.1 What was read, and the gates

| dataset | status | cells |
|---|---|---|
| MultiHop-RAG | read (RK) | GP: s1eval 1,130 questions and 2,747 slots decides; s1sel 224 questions and 504 slots beside |
| SQuAD | n/a | no gold pair (D-P1); only 12.6's label sizes |
| MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki | NOT READ | host rows, 10.6 |

**Gates pass on MultiHop-RAG.** G-P1: B0 = 25,343 pairs = S 0 + R 25,080 + K 263, the grade's. G-P2: every form's pairs beyond B0 equal the grade's (H5 2,387; HM 1,332; HK1 2,712; NTc 13,668; NT1 4,604; HK2 4,886; H10 4,588). G-P3: slots, questions and the cover arrays equal the grade's.

### 10.2 MultiHop-RAG, RK, GP, s1eval (precision, reach and "vs any" in points; intervals are 95%)

The graph has 609 nodes, so the any-link graph holds 13.7% of all node pairs. The any-link unit has 89 candidates per anchor, and the uniform-partner base is 0.30%.

| unit | pairs | precision % | reach % | graph price | lift call | vs any |
|---|---|---|---|---|---|---|
| label R.mention | 2 | 0.00 [0.00, 0.00] | 0.00 | - | BELOW_BASE | -1.77 BELOW |
| label R.name_comention | 25,005 | 1.77 [1.70, 1.84] | 84.89 [83.13, 86.73] | 10.7 | POINTS | 0.01 [-0.00, 0.01] LEVEL |
| label R.title_comention | 75 | 2.08 [0.80, 3.70] | 0.29 [0.11, 0.52] | 9.4 | POINTS | 0.32 [-0.97, 1.94] LEVEL |
| label K.knn | 1,366 | **9.96 [9.26, 10.72]** | 28.14 [26.25, 30.28] | **1.8** | POINTS | **8.19 [7.51, 8.93] ABOVE** |
| group R | 25,080 | 1.77 [1.71, 1.85] | 85.18 [83.46, 86.99] | 10.7 | POINTS | 0.01 [0.00, 0.01] ABOVE |
| group K | 263 | 0.98 [0.47, 1.62] | 0.36 [0.18, 0.61] | 26.3 | POINTS | -0.79 [-1.32, -0.14] BELOW |
| **any link** | 25,343 | 1.77 [1.70, 1.84] | 85.55 [83.86, 87.32] | 10.8 | POINTS | - |
| form H5 | 2,387 | 1.05 [0.87, 1.24] | 6.26 [5.20, 7.39] | 13.9 | POINTS | -0.72 [-0.92, -0.50] BELOW |
| form HM | 1,332 | 1.85 [1.53, 2.19] | 5.53 [4.50, 6.58] | 8.8 | POINTS | 0.09 [-0.27, 0.44] LEVEL |
| form HK1 | 2,712 | 0.75 [0.61, 0.90] | 4.95 [4.00, 5.89] | 19.9 | POINTS | -1.02 [-1.20, -0.84] BELOW |
| form NTc | 13,668 | 0.27 [0.22, 0.31] | 6.26 [5.24, 7.31] | 79.5 | NO_LIFT | -1.50 [-1.59, -1.41] BELOW |
| form NT1 | 4,604 | 0.29 [0.22, 0.36] | 2.33 [1.77, 2.93] | 71.9 | NO_LIFT | -1.48 [-1.59, -1.38] BELOW |
| form HK2 | 4,886 | 0.60 [0.50, 0.71] | 7.03 [5.87, 8.19] | 25.3 | POINTS | -1.17 [-1.30, -1.03] BELOW |
| form H10 | 4,588 | 0.71 [0.61, 0.83] | 7.83 [6.61, 9.08] | 21.3 | POINTS | -1.05 [-1.20, -0.90] BELOW |

Reach of a form is beyond B0 (the forms' pairs are the tail), so it is small beside the any-link unit's.

**Exact label sets** (the any-link pairs by the labels they carry):

| labels | share of candidates | precision % | reach % | vs any |
|---|---|---|---|---|
| R.name_comention only | 93.9% | 1.26 [1.19, 1.33] | 57.12 | BELOW |
| R.name_comention and K.knn | 5.1% | **11.32 [10.54, 12.17]** | 27.78 | ABOVE |
| K.knn only | 0.8% | 0.98 [0.47, 1.62] (20 targets) | 0.36 | BELOW |

**Multi minus single:** +10.03 [9.23, 10.90] points, ABOVE.

**Rank quintiles of each form** (precision %, first and last of five; the paired call is on the difference):

| form | first | last | call | first quintile vs any |
|---|---|---|---|---|
| H5 | 3.01 [2.45, 3.61] (266 targets) | 0.06 (2 targets) | ABOVE | ABOVE |
| HM | 4.58 [3.64, 5.67] (216) | 0.39 (6) | ABOVE | ABOVE |
| HK1 | 1.93 [1.52, 2.36] (190) | 0.04 (2) | ABOVE | LEVEL |
| NTc | 0.12 [0.06, 0.19] (28) | 0.08 (24) | LEVEL | BELOW |
| NT1 | 0.90 [0.67, 1.15] (112) | 0.03 (2) | ABOVE | BELOW |
| HK2 | 1.93 [1.61, 2.27] (354) | 0.00 (0) | ABOVE | LEVEL |
| H10 | 1.99 [1.63, 2.35] (326) | 0.00 (0) | ABOVE | LEVEL |

**s1sel beside** (224 questions, 504 slots) agrees on every main call: K.knn 9.41 [7.87, 11.02] ABOVE any; R.name_comention 1.72 LEVEL; any 1.71 [1.56, 1.86]; multi minus single +9.39 [7.56, 11.28] ABOVE. Five small-count units differ in a call: R.title_comention lift (NO_LIFT), group R vs any (LEVEL), group K lift (NO_LIFT) and vs any (LEVEL), form H5 vs any (LEVEL, not BELOW) and form NT1 lift (POINTS, not NO_LIFT).

### 10.3 The expectations (written before any number; decide nothing)

| id | call | reads | detail |
|---|---|---|---|
| E1 | HELD | 1 of 1 | MultiHop-RAG's two labels with 20 or more targets (R.name_comention, K.knn) both POINT |
| E2 | NOT_READ | 0 | HotpotQA, 2Wiki, MuSiQue are not read |
| E3 | NOT_READ | 0 | MetaQA, WebQSP are not read |
| E4 | NOT_HELD | 6 of 7 | HM is LEVEL with the any-link unit, not BELOW |
| E5 | NOT_HELD | 0 of 1 | only NTc has 20 or more targets in both its first and last quintile, and it is LEVEL; the other six forms' last quintiles have 0 to 6 targets, so the declared bar leaves them uncalled, though their paired calls above are ABOVE |
| E6 | HELD | 1 of 1 | multi minus single +10.03 ABOVE |
| E7 | NOT_READ | 0 | MuSiQue is not read |

### 10.4 What the MultiHop-RAG read shows (a reading, not a decision)

- **The any-link unit is R.name_comention.** It is 25,005 of 25,343 pairs and gives the any-link reach (85.55%) at 89 candidates per anchor and 1.77% precision, 5.84 [5.69, 5.99] times the uniform base. About one candidate in 56 is the partner.
- **K.knn is the most precise label and the cheapest to read: 9.96% at 5.2 candidates per anchor, 1.8 pairs per covered slot.** It reaches 28%. **Its precision lives where it agrees with the name co-mention:** R.name_comention and K.knn together give 11.32%, K.knn alone 0.98% (20 targets), R.name_comention alone 1.26%. So the label is a sign of agreement, and a weight on it would be a weight on agreement.
- **R.mention (2 pairs) and R.title_comention (75 pairs, 0.29% reach) are too small to read on this dataset.** Neither has 20 targets. E2 needs MuSiQue, HotpotQA and 2Wiki.
- **All seven forms are the tail, as E4 expected, except HM, which is level.** NTc and NT1 (the written forms) are NO_LIFT: 0.27% and 0.29% precision at 72 to 80 pairs per covered slot. The five hub forms POINT, at 0.60% to 1.85%.
- **The build's rank order carries signal in the hub forms and in NT1, and not in NTc.** Cutting H5 or HM to its first quintile gives 3.01% or 4.58% precision against the any-link unit's 1.77%, both ABOVE it (paired). NTc's first quintile is not above its last.

### 10.5 What follows (names only; no rule, carry or weight changes)

- **Candidate label weight:** K.knn on MultiHop-RAG (POINTS, ABOVE any). Named, not adopted; AP4's D92 stays default no. It is the only label that qualifies on this read.
- **Candidate tighter cut:** H5 (BELOW any as a whole, its first quintile ABOVE). HM's first quintile is also ABOVE any, but HM is LEVEL as a whole, so the declared rule does not name it.
- **Flags:** none (no label BELOW any link holds 5% of the any-link candidates; group K is BELOW at 1% of the pairs).
- **Not read yet:** the S labels (MetaQA, WebQSP, HotpotQA and 2Wiki SRK), E2, E3, E7, and every MuSiQue, MetaQA, WebQSP, HotpotQA and 2Wiki number.

### 10.6 Host rows needed (not in the feed; this stage's author does not edit it)

`ul2p_host_rows.txt` in the sessions' notes folder holds the exact lines (format `line|x|deps|name|cpus|mem|inputs|command`). One row per dataset, 1 CPU, no GPU:

| row | deps | mem (GB, estimate) | command |
|---|---|---|---|
| ul2p-musique | ul2c-grade-musique | 8 | `python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/mp_unified/zul2p.py ledger --ds musique --host --threads 1` |
| ul2p-metaqa | ul2c-grade-metaqa | 4 | the same with `--ds metaqa` |
| ul2p-webqsp | ul2c-grade-webqsp | 16 | the same with `--ds webqsp` |
| ul2p-hotpotqa | ul2c-grade-hotpotqa | 16 | the same with `--ds hotpotqa` |
| ul2p-2wiki | ul2c-grade-2wiki | 16 | the same with `--ds 2wiki` |
| ul2p-report | the five rows above | 2 | `python outputs/mp_unified/zul2p.py report` |

- **Inputs:** each ledger row takes its grade row's 29 inputs plus `zap4.py`, `zd83rep.py`, `zul2p.py` and this page. They are new files that exist only on the laptop, so `rx push --inputs` goes first (the feeder never pushes inputs). The report row also takes MultiHop-RAG's and SQuAD's ledger files from the laptop.
- **Memory is an estimate, to be re-asked at the first measured peak.** 12.6's grade measured 0.28 GB (MuSiQue), 0.36 GB (MetaQA) and 11.73 GB (WebQSP) for the same loads plus its builds.
- **HotpotQA and 2Wiki** wait behind their e3v K through their grade rows. Their ledgers read both versions (RK and SRK).
- The pool-anchored ledger on the same units runs after the pools are rebuilt (section 8).

## 11. Files

- `outputs/mp_unified/zul2p.py`
- `outputs/ul2p/<dataset>/ledger.json`: the ledger and its record (shas of the grade, ranks, slots and questions; no array)
- `outputs/ul2p/report.json` and `report.md`
