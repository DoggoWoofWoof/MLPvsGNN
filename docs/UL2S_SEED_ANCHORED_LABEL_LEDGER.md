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

(not filed; the runs have not been made)

## 11. Development log (disclosed before any UL2S number)

- No `zul2s` code existed before this page. Before it, only file shapes and sizes were inspected (the keys of E0's lists and gold caches, PL1 S0a's files, grade.json fields), and a probe that loaded SQuAD's and MuSiQue's families on the laptop printed counts of pairs and stopped at the missing K_e3v. No UL2S measure was computed.
- UL2P's code (`zul2p.py`) and its filed numbers are unchanged.

## 12. Files

- `outputs/mp_unified/zul2s.py`
- `outputs/ul2p/seed/<dataset>/ledger.json`: the ledger and its record (shas of the grade, ranks, first-stage lists, gold cache; no array)
- `outputs/ul2p/seed/report.json` and `report.md`
