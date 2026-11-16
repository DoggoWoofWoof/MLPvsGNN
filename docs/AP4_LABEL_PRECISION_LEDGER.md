# AP4: the label precision ledger (AP's D80, a scouting stage)

Declared 16 November 2026, before any AP4 number.

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
