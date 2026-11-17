# AP4: the label precision ledger (AP's D80, a scouting stage)

Declared 16 November 2026, before any AP4 number. Results filed 17 November 2026 (section 10).

It follows:
- AP3 ([AP3_STRUCTURAL_SLOT_TITLE_FORMS.md](AP3_STRUCTURAL_SLOT_TITLE_FORMS.md); Part A filed ec19758, 12 Nov 03:44);
- UL1 ([UL1_UNIFIED_LINKS_ORACLE_CHECK.md](UL1_UNIFIED_LINKS_ORACLE_CHECK.md); filed c662a6b, CARRIED F0T);
- D83 ([D83_UNIFIED_R_REFIT.md](D83_UNIFIED_R_REFIT.md); declared and running).

Code: `outputs/mp_unified/zap4.py`, new. It imports `zd83rep.py`, `ltwdiag.py`, `chndiag.py` and `lean_cache.py` and edits none of them. Outputs go only under `outputs/ap4/`.

**What AP4 does not touch:**
- Nothing trains and no model is read.
- No pool, link, column or fit of record is written.
- AP4 decides nothing about D83: D83's rule is section 7 of its own page, fixed before its reads.

**D80** (AP3's page, section 9): "a later stage (AP4) on query-side typed rules and a triple family, starting with a scouting stage with its own declaration. Default: yes after AP3's Part A is filed, as a scouting stage only."
- The default's condition has been met since 12 Nov.
- This page is that scouting stage. It narrows it to the first question the typed rules need answered: how precisely each rule's labelled links point from what the first stage found to what it missed.
- The query-side typed rules and the triple family stay named for after it (section 6).

## 1. Why

**The user, 16 Nov (verbatim):**
- "i do not think we need to unnecessarily train models after every improvement, like if we created better links we can train a model after verifying those links improved connectivity etc by comparing with the oracle links we have and across datasets etc and unifying everything";
- "the edges should also have labels based on how it was created so that we can unify with KB labels and make the model train and understand better rather than just giving family names".

**What UL1 left open.** UL1 graded the unified links against the oracle: the reach of gold pairs, and the price in pairs. It did not ask two questions:
- which rule's links point at a missed gold from the passages the first stage already ranks high;
- where that gold sits among an anchor's linked rows.

**The filed diagnostics name that as the gap.**
- Reach is saturated: the knee (DEPTH_FLAT) and COV (pools keep 97.8-98.7% of the golds).
- Ranking is what is missing. In CHN's D1, MuSiQue's missed gold is first or second among a found gold's neighbours for only 0.22-0.28 of them.
- AP3's scouting found that hub links reach chain slots but rank their partners poorly.

**Why it matters for D83.** D83 gives the models five labels in place of three families.
- If the labels differ in precision, a model has something to learn from them.
- If they do not, the labels can add nothing beyond the families.

AP4 measures this at the graph level, before D83's reads exist. It also names label weights for a later refit and does not adopt them.

## 2. What is read (the host, read only)

| arm | root, carve | edge bits | datasets |
|---|---|---|---|
| D83E | `outputs/abl/D83E`, s1eval | D83's labels: S, R.mention, R.name_comention, R.title_comention, K.knn | the six |
| REC | `outputs/pko`, s1eval | the record's families: slot (S on a KB, W2_S on passages), NER.spacy, K.knn | the six |
| B1V | `outputs/b1v`, b1v | D83's labels | 2Wiki, HotpotQA, MuSiQue |
| B1W | `outputs/b1p`, b1w | the record's families | 2Wiki, HotpotQA, MuSiQue |

**What is read from each arm:**
- the cache and links parts (`lean_cache.part_dirs`; `ltwdiag.part_arrays`, which ties each links part to its cache part by the links record);
- the rrf and dense_cos columns, read from the cache's xc where `lean_cache.xc_columns` puts them (`ltwdiag`'s check).

**The paired arms must match.** D83E and B1V reuse the record's pools (A2). So D83E and REC, and B1V and B1W, must hold:
- the same questions;
- the same pool rows and golds;
- the same rrf top fives.

**G6** checks this by sha256 of the ids, of the pools (row counts, node ids, gold flags) and of the top fives, before any comparison. It stops the report on a mismatch.

## 3. The measures, the calls and the expectations (fixed before any number)

**Per question.** A question counts when it has a gold and a non-empty pool. rrf's order (`chndiag.order_of`: descending, NaN last, ties by pool position) gives the top five.

**Two anchors:**
- **found:** each gold in the top five, the diagnostic anchor;
- **top1:** rrf's first row, gold or not. This is what a model has without knowing the golds.

**Candidates and targets.**
- An anchor's candidates are its linked rows outside the top five, linked in either direction, each with its label bits.
- A target is a gold among the candidates.

**Units:**
- each label (or family) alone;
- the groups:
  - S (KB only);
  - R: D83's three R labels; the record's slot plus NER.spacy on passages, NER.spacy on a KB;
  - K;
- any link.

**Per unit, kind and dataset.** Every measure is a ratio of sums over the questions:

| measure | definition | call |
|---|---|---|
| price | candidate pairs per anchor | |
| precision | targets / pairs | |
| base | golds outside the top five / rows outside the top five | |
| lift | precision / base | **POINTS** when the interval's low end is above 1; **BELOW_BASE** when its high end is below 1; **NO_LIFT** otherwise |
| reach (found only) | share of in-pool golds outside the top five that are a candidate of some found anchor through the unit | |
| rank1 | among anchors with a target through the unit, the share whose best target is the unit's first candidate by rrf | |
| vs any | the unit's precision minus any link's, paired | ABOVE, LEVEL or BELOW |

- **Beside rank1, as points only:** top2, rank1 by dense_cos, and the median best position.
- **Exact label sets:** the share, precision and lift of the most frequent sets, until 99% of the pairs are listed (at most 12).
- **Multi minus single:** the precision of pairs with two or more labels minus that of pairs with one, paired: ABOVE, LEVEL or BELOW.
- **MuSiQue by hops (2, 3, 4):** every unit and kind.

**D83's arm minus the record's arm** (same questions, after G6).
- Measures: precision (both kinds), price, rank1, and found reach.
- Units: the S, R and K groups and any link.
- Each is a paired difference: ABOVE, LEVEL or BELOW.
- Beside them: the K pairs at the anchors, counted in both arms. They should be the same set, because D83's K.knn label is read before the dedup.

**Intervals:** 95% percentile intervals over 2,000 resamples of the questions with replacement (numpy `default_rng(0)`, drawn in chunks of 100).

**Expectations** (written before any number; they decide nothing; the report marks each HELD, NOT_HELD or NOT_READ):

| id | expectation |
|---|---|
| E1 | On 2Wiki and HotpotQA, R.mention has the highest top1 precision of D83's labels. |
| E2 | K.knn has the lowest top1 lift of D83's labels on every passage dataset. |
| E3 | Pairs with two or more labels are ABOVE single-label pairs in top1 precision, on every dataset. |
| E4 | On MetaQA and WebQSP, S has the highest top1 precision of D83's labels. |
| E5 | MuSiQue's found-anchored reach of any link falls with the hops (2 > 3 > 4). |
| E6 | D83's R is not BELOW the record's R in top1 precision on any paired dataset. |

**Label weights named, not adopted:** ln(top1 lift) per D83 label and dataset.

**What follows, named now:**
- **Candidate up-weights.** A label that POINTS and is ABOVE any link is a candidate up-weight, or a bias initialization, for a later refit (D92, section 9).
- **E3.** If it holds, it is graph-level support for reading label sets (D83's form) over families. If it does not hold, D91 (D83 without the labels) gains weight.
- **A flag for D83.** D83's R BELOW the record's R in precision on a dataset is a flag for D83's reads there. It is reported beside D83's report and does not change D83's rule.
- **No refit.** AP4 declares none.

## 4. Rows

- **18 ledger rows,** `python outputs/mp_unified/zap4.py ledger --arm A --dataset D` through `retry_cmd.py` (3 tries):
  - REC (6) and B1W (3) now, with no dependencies;
  - D83E (6) after `d83-chain-e-<D>`;
  - B1V (3) after `d83b-chain-<D>`.
- **Resources:** 1 CPU each. Memory as D83's labelstats rows: 6 GB, or 10 GB for MuSiQue and MetaQA.
- **`ap4-report`** runs after the 18 ledger rows (1 CPU, 8 GB).
- **Then the laptop:** `rx fetch` of `outputs/ap4/report.*` and `outputs/ap4/ledger/*/*.json`, then filing here.

## 5. What it can and cannot show

- **Graph-level precision is not R@5.** A precise label can add nothing when the model already ranks its targets. An imprecise label can help in combination. AP4 names candidates; reads decide.
- **The top1 anchor is rrf's.** The models' own top rows differ.
- **Base rates differ by dataset** (pool sizes). So lifts are compared across labels within a dataset, not across datasets.
- **The pools are the record's (A2).** Reach is bounded by the pool.

## 6. What AP4 does not do

- **Query-side typed rules** (a question's words choosing which labels to follow): a later scouting stage, named after this ledger is filed.
- **The triple family** (P2c's triples on the host): the same.
- **S on 2Wiki and HotpotQA:** the hyperlinks are D89's. D89's page will run this ledger on its own stores when they exist.
- **No refit, and no change to D83, its rows or its rule.**

## 7. Development log (disclosed before any AP4 number)

- **16 Nov, 14:27-14:29 IST: dry runs** of `zap4.py ledger` on the host.
  - Inputs: REC WebQSP and REC MuSiQue, the first 300 questions each (`--limit 300`), written to `outputs/ap4/dry`.
  - The report refuses limited ledgers and never reads that folder.
  - On the laptop, the report's code (`arm_block`, `pair_block`) ran on the two dry ledgers. It printed only array shapes, unit names and timing. No measure was looked at.
- **The self-test passes:**
  - `question_pairs` equals a brute force (5 and 3 labels; ties; duplicate links in both directions);
  - precision, reach and rank1 equal a brute force;
  - the exact sets and multi/single partition the pairs;
  - the bootstrap and the calls;
  - the units cover every bit;
  - the pair block of a ledger against itself is LEVEL everywhere, with equal K pairs;
  - G6 stops on a mismatch;
  - the report's text is written.

## 8. Files

- `outputs/mp_unified/zap4.py`
- `outputs/ap4/ledger/<arm>/<dataset>.npz` and `.json`: the ledger and its record, with the sha256 of the ids, pools, top fives and the npz.
- `outputs/ap4/report.json` and `report.md`.

## 9. Decisions (the user's; my default first)

- **D80:** answered by this page. The scouting stage is the label precision ledger; typed query rules and the triple family come after it.
- **D92** (new): adopt AP4's named weights as the label-bias initialization of a later refit. **Default: no,** unless the user says so after D83's report.

## 10. Results, filed 17 Nov 2026

**The report.**
- `outputs/ap4/report.json` and `report.md`, written on the host at 2026-11-16T21:57:33Z.
- Code: zap4.py as committed in 42e5e06. The report's `zap4_sha256` (377e6c57…) is that file with LF line ends; the laptop checkout has CRLF (35a3117f…).
- 18 ledgers:
  - D83E and REC on all six datasets: SQuAD 11,873 questions, HotpotQA 7,405, 2Wiki 12,576, MuSiQue 2,417, MetaQA 9,785, WebQSP 1,503;
  - B1V and B1W on the 1,000-question B1 carves of MuSiQue, HotpotQA and 2Wiki.
- G6 passed for every pair (same questions, pools and top fives).

**Expectations** (they decide nothing):

| id | call | what the reads show |
|---|---|---|
| E1 | NOT_HELD | Held on 1 of 4 reads (B1V 2Wiki). R.title_comention has the highest top1 precision on D83E HotpotQA (7.2%) and 2Wiki (9.5%) and on B1V HotpotQA. |
| E2 | NOT_HELD | 0 of 7. K.knn POINTS on every passage read. The lowest lift is R.name_comention on HotpotQA and 2Wiki, R.mention on MuSiQue and R.title_comention on SQuAD. |
| E3 | NOT_HELD | 8 of 9. Multi-label pairs are ABOVE single-label pairs everywhere except MetaQA: −5.8 points [−6.2, −5.3], BELOW. |
| E4 | HELD | S has the highest top1 precision on MetaQA (9.4%, lift 56) and WebQSP (1.8%, lift 15). |
| E5 | HELD | Any link's found reach on MuSiQue by hops (2 > 3 > 4): D83E 71.1 > 29.9 > 24.0 and B1V 84.4 > 57.7 > 43.1. |
| E6 | NOT_HELD | 5 of 9. D83's R is BELOW the record's R in top1 precision on B1 HotpotQA (−0.7 points), B1 2Wiki (−0.7), MetaQA (−0.3) and WebQSP (−1.3). |

**D83's labels at the top1 anchor** (D83E; precision in %, 95% intervals; "vs any" in points):

| dataset | label | pairs per anchor | precision | lift | call | vs any |
|---|---|---|---|---|---|---|
| SQuAD | R.mention | 12.84 | 0.2 [0.2, 0.2] | 1.1 [1.0, 1.2] | NO_LIFT | −0.0 BELOW |
| SQuAD | R.name_comention | 3.13 | 0.4 [0.3, 0.5] | 2.4 [2.1, 2.8] | POINTS | +0.2 ABOVE |
| SQuAD | R.title_comention | 0.06 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | BELOW_BASE | −0.2 BELOW |
| SQuAD | K.knn | 1.96 | 0.4 [0.3, 0.5] | 2.6 [2.1, 3.1] | POINTS | +0.2 ABOVE |
| HotpotQA | R.mention | 14.66 | 2.2 [2.1, 2.3] | 26.2 [25.0, 27.5] | POINTS | +0.8 ABOVE |
| HotpotQA | R.name_comention | 11.25 | 1.1 [1.1, 1.2] | 13.5 [12.7, 14.3] | POINTS | −0.2 BELOW |
| HotpotQA | R.title_comention | 0.77 | 7.2 [6.5, 8.0] | 85.8 [77.5, 94.8] | POINTS | +5.8 ABOVE |
| HotpotQA | K.knn | 4.37 | 1.8 [1.6, 1.9] | 20.9 [19.3, 22.6] | POINTS | +0.4 ABOVE |
| 2Wiki | R.mention | 15.93 | 3.2 [3.1, 3.3] | 34.7 [34.0, 35.4] | POINTS | +1.6 ABOVE |
| 2Wiki | R.name_comention | 19.18 | 1.5 [1.4, 1.5] | 16.1 [15.6, 16.6] | POINTS | −0.1 BELOW |
| 2Wiki | R.title_comention | 0.57 | 9.5 [8.7, 10.2] | 102.2 [94.5, 110.1] | POINTS | +7.9 ABOVE |
| 2Wiki | K.knn | 3.96 | 3.0 [2.9, 3.2] | 32.5 [31.0, 34.1] | POINTS | +1.4 ABOVE |
| MuSiQue | R.mention | 46.50 | 0.5 [0.5, 0.6] | 19.9 [17.6, 22.5] | POINTS | −0.2 BELOW |
| MuSiQue | R.name_comention | 18.30 | 1.6 [1.5, 1.8] | 60.6 [56.0, 65.1] | POINTS | +0.9 ABOVE |
| MuSiQue | R.title_comention | 0.66 | 6.5 [5.2, 7.9] | 239.8 [194.1, 293.2] | POINTS | +5.8 ABOVE |
| MuSiQue | K.knn | 4.10 | 2.9 [2.6, 3.3] | 109.3 [97.7, 121.4] | POINTS | +2.2 ABOVE |
| MetaQA | S | 6.39 | 9.4 [9.0, 9.8] | 56.0 [52.7, 59.6] | POINTS | +3.4 ABOVE |
| MetaQA | R.name_comention | 0.61 | 0.1 [0.0, 0.2] | 0.6 [0.1, 1.2] | NO_LIFT | −5.9 BELOW |
| MetaQA | K.knn | 3.16 | 0.2 [0.1, 0.3] | 1.2 [0.9, 1.6] | NO_LIFT | −5.8 BELOW |
| WebQSP | S | 58.10 | 1.8 [1.3, 2.4] | 15.3 [11.4, 20.1] | POINTS | +0.1 ABOVE |
| WebQSP | R.name_comention | 1.66 | 1.1 [0.7, 1.6] | 9.6 [6.0, 13.8] | POINTS | −0.6 LEVEL |
| WebQSP | R.title_comention | 0.00 | 0.0 [0.0, 0.0] | 0.0 [0.0, 0.0] | BELOW_BASE | −1.7 BELOW |
| WebQSP | K.knn | 5.57 | 0.9 [0.7, 1.2] | 8.0 [6.0, 10.4] | POINTS | −0.8 BELOW |

The KBs have no R.mention pairs (and MetaQA no R.title_comention pairs). Lifts compare labels within a dataset, not across datasets (section 5).

**D83 minus the record** (paired, same questions and pools):
- **Record carves (D83E − REC):**
  - On HotpotQA, 2Wiki and MuSiQue, D83's R is ABOVE in top1 precision (+0.1 points each) and in found reach (+1.0, +0.8 and +2.6 points).
  - On HotpotQA and 2Wiki it costs fewer pairs per anchor (−1.0 and −1.6).
  - Found rank1 is BELOW on all three (−1.5 to −4.1 points).
  - SQuAD is LEVEL. On the KBs, R is BELOW in precision (MetaQA −0.3, WebQSP −1.3).
- **B1 carves (B1V − B1W):**
  - D83's R pays more pairs per anchor (+2.1 MuSiQue, +0.9 HotpotQA, +0.2 2Wiki).
  - It reaches more on MuSiQue and HotpotQA (+3.1 and +5.7 points, ABOVE); 2Wiki is LEVEL.
  - It is BELOW in top1 precision on HotpotQA and 2Wiki (−0.7 each) and in found rank1 on all three (−3.8 to −4.9).
- **K pairs at the anchors** were expected to be the same set (section 3).
  - They are on 4 of 9 reads. On the other 5, D83's K pairs are a strict subset of the record's, short by 2 to 339; the largest is MuSiQue D83E, 21,292 against 21,631 (1.6%).
  - The cause is not established. It changes no call: group K is LEVEL in every pair except D83E MuSiQue top1 precision (+0.1, ABOVE).

**What follows** (section 3's named consequences):
- **A flag for D83.** D83's R is BELOW the record's R in top1 precision on B1 HotpotQA, B1 2Wiki, MetaQA and WebQSP. It goes beside D83's report and changes neither D83's rule nor its reads.
- **E3 held on 8 of 9 reads.** This is graph-level support for reading label sets (D83's form) over families. The exception is MetaQA, where S alone is the precise label. D91 (D83 without labels) runs as declared.
- **Candidate up-weights** (POINTS and ABOVE any link, D83E top1):
  - R.title_comention on HotpotQA, 2Wiki and MuSiQue;
  - R.mention on HotpotQA and 2Wiki;
  - K.knn on all four passage datasets;
  - R.name_comention on SQuAD and MuSiQue;
  - S on MetaQA and WebQSP.

  Their ln(top1 lift) is in the report. **D92 stays default no.**
- **Label precision is far from uniform.** R.title_comention's few pairs (0.6-0.8 per anchor) are 3x (HotpotQA, 2Wiki) to 13x (MuSiQue) as precise as R.mention's. K.knn is not the weakest label anywhere on passages. This is the case for the query-side typed rules (section 6), still named, not declared.
- **Open item:** the K-pair subset above. It is checked before D91's report reads the K family.
