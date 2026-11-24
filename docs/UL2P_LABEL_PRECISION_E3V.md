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

(not filed; the runs have not been made)

## 11. Files

- `outputs/mp_unified/zul2p.py`
- `outputs/ul2p/<dataset>/ledger.json`: the ledger and its record (shas of the grade, ranks, slots and questions; no array)
- `outputs/ul2p/report.json` and `report.md`
