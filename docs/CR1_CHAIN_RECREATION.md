# CR1 — recreating the oracle's chains: how far the gold slots are from linked, and which order finds the next hop, on all seven datasets

Declared 26 Nov 2026, before any CR1 number exists. Code: `outputs/mp_unified/zcr1.py` (committed with this page, before the first row is queued). It imports `zul2p.py`, `zul2c.py`, `zul1.py`, `zap4.py` and `zenc.py`, read only; nothing earlier changes. Outputs only under `outputs/cr1/`. No pool is built, no model reads anything, nothing trains, no LLM, no spaCy, no GPU. Gold positions and MuSiQue's decompositions are the **grading only**. `s1eval` decides, `s1sel` is beside, as in UL2P.

## 1. Why this page exists

The user's words (26 Nov):

- **"we measured sending the oracle or direct gold links to the model getting 78 recall ... find that and check what are we missing and how can we restrict, can we do a per hop limit some scoring based on the questions or some PPR based technique"**
- **"yeah need to measure that and also need to find a way to recreate the chains and use that logic on other datasets"** (the reply to: pool survival of the links between golds is unmeasured)
- **"check the same logic across all the other datasets"**

What exists, and what it leaves open:

1. The 78 is SWP Part D (`docs/SWP_S_TO_R_SWAP.md` 5c and 5e, 8 Nov): MuSiQue's 1,090 oracle chain pairs added to the rule graph lift the two models of record in-domain R@5 from 69.0 to 79.6 and from 67.4 to 78.2; decoy and swapped partners lower it. The chain pairs are made from MuSiQue's own decomposition (a sub-question that names `#k` joins its supporting paragraph to that of hop k). That is a ceiling that uses labels.
2. UL2P (`docs/UL2P_LABEL_PRECISION_E3V.md`) reads, on the current layout, how many gold slots the whole any-link graph B0 holds directly: MuSiQue chain slots 49.76 percent, gold pairs 33.76; MetaQA 13.12; WebQSP 19.07; 2Wiki 43.76 (SRK 45.80); HotpotQA 65.31 (SRK 80.01); MultiHop-RAG 85.55; SQuAD has no pair. It says nothing about a slot with no direct link: how far apart its two ends are.
3. MHB (`docs/MHB_MULTIHOP_BRIDGES.md`, 18 Nov, old layout with spaCy) censused how MuSiQue's, 2Wiki's and HotpotQA's chains are made. Its unlinked MuSiQue slots are mostly a bridge both passages name as a hub (63 percent), or write but not as a rule name (22 percent). The hub line is closed; no degree, IDF or hub-key attribute is used here.
4. GP3 amendment A4 (a530804a, the GP1 session's) reads slot recall **inside a walk's pool** and the precision of the pool's own links on a fixed 40-question sample. It decides nothing and its numbers are not filed. CR1 is the whole-graph counterpart: it reads how far the slots are from linked, and how the next hop is best ordered. The two are read side by side once both exist.
5. GP1 and GP2 order a source's links by the pair's own dot (the two stored rows) and by the first-stage rank of the source. **None of them orders a source's links by the question.** R8 part B read a flat question cosine as the scorer of branches at a cap and found the routed scorer below it on four of five; its measure was completion at a cap, not the rank of a direct gold link among its source's links, and a path-aware order (the question minus what the source already says) was not read.

## 2. What the stage is

For every slot cell UL2P reads (the same questions, slots and layouts), CR1 gives two readings, both gold-anchored and pool-free:

- **Part A: distance.** Each slot is classed D (its two ends are linked in B0), T (not linked, but at least one node is linked to both ends), or N (neither). A question is chain-connected within one link (CC1: all its slots D) or within two (CC2: all D or T). The two-link count of intermediates and whether an intermediate is itself a gold are read. A control with the same machinery and a different question's gold gives the chance level.
- **Part B: order.** For the slots that are D, the target's rank among its source's B0 neighbours under three orders: the pair's own dot (GP1's and GP3's), the question's cosine to the neighbour (the flat question cosine), and the **residual question cosine** (the question with the source's own direction removed, then its cosine to the neighbour). The residual is the label-free form of the idea that the next hop answers what the question has not yet said; it has no parameter.

CR1 declares **no rule**. If Part B calls the residual order better on enough datasets, a label-free path rule (CR2) is declared on its own page, with its own gates, before any pool number; the pool grade of CR2 needs the GP3 walker (the GP1 session's), reached by message. If Part A shows that two-link paths hold most chains at a small count of intermediates, a composed-path form is declared the same way. Neither is promised.

## 3. Slots, versions, definitions

- **Slots** are UL2P's: `UL2P.load_host(d)` (and `load_mhr()` for MultiHop-RAG) returns the cells keyed (carve, kind): **GP** is every pair of distinct golds of a question (the KBs: 2 to 10 golds, `UL.KB_GOLDS`, the grade's), **CH** is MuSiQue's chain pairs (zul1's `chain_slots`, the ORC rule). Both carves, `s1eval` decides, `s1sel` beside. **SQuAD has one gold per question and no slot: n/a**, as in UL2P (its seed-anchored read stands in UL2S).
- **What a slot means per dataset.** MuSiQue CH: the oracle's chain pairs (2 to 4 hops). MuSiQue GP: all gold pairs of the same questions. 2Wiki, HotpotQA, MultiHop-RAG GP: the supporting passages' pairs; HotpotQA and 2Wiki bridge questions are chains of two, their comparison questions are two parallel golds with no chain (MHB Part C). **MetaQA and WebQSP golds are the answer set** (the KB's answer entities): their GP slots are answer to answer, not topic to answer; the topic-to-answer KB path is the S layer's own and is not a slot here (named, not read: the seed-to-gold hop distance is UL2H's and GP1's).
- **Versions** are the grade's: RK on every dataset; SRK where S exists (HotpotQA, 2Wiki hyperlinks; MetaQA, WebQSP KB triples). B0 is UL2P's: S (SRK only), R (W2_S mention, F0T title and name co-mention), K (the e3v kNN), each pair once.
- **N(x)** is the set of nodes joined to x by a B0 pair, either direction. **D**: b in N(a). **T**: not D, and N(a) and N(b) share a node (the intermediate m, m not a, not b). **N**: neither. `|M|` is the count of shared nodes. **Via gold** is a T slot with a shared node that is a gold of the same question.
- **CC1** (a question with at least one slot and every slot D) and **CC2** (every slot D or T), per carve and kind, and by hop on MuSiQue (2, 3, 4). Unit: the question; paired intervals bootstrap over questions as UL2P does (`zap4.boot`).
- **Control (chance):** for each slot (a, b) of question j, the control slot is (a, b') where b' is an end of the next question's (with a slot, in order, cyclic) first slot; skipped where b' = a. The control is read with the same classes. All its ends are anchors of the same cell, so no further neighbour list is needed.

## 4. Part B in detail

- **Question and node vectors:** the frozen e3v rows (`zenc.vec_shards(root, "e3v", d, "docs")`, `Rows`) and the question rows of the same encoder (`kind = "queries"`), looked up by the question id the carve names. Rows are unit norm (gate G-C5 checks the norm of the question rows and of a sample of node rows).
- **Source and target:** for a D slot (a, b), the **source** is the end with the larger question cosine (ties: the lower node id), the target the other. This uses no gold beyond the slot's two ends being gold, and no chain direction (MuSiQue's order of hops is not used: it is a label).
- **Scores of a neighbour x of the source s:** `pd(x) = row(s) . row(x)`; `fq(x) = q . row(x)`; `rq(x) = (q - (q . ŝ) ŝ) . row(x)`, with ŝ = row(s) / |row(s)|, which equals `fq(x) - (q . ŝ)(ŝ . row(x))`. Ranks do not depend on the residual's length, so it is not renormalised.
- **Rank of the target** = 1 + the number of neighbours of the source with a strictly larger score (ties share the rank, as GP1's A3). Per slot and per order: rank, and `|N(s)|`.
- **Reported per cell and version:** the number of D slots; the share of D slots with rank <= 10, <= 30 (GP1's cut) and <= 100; the median rank as a share of `|N(s)|`; the chance share (`mean(min(1, K/|N(s)|))`); the three paired differences **rq - fq**, **fq - pd** and **rq - pd** at <= 30, with intervals (questions resampled, UL2P's `boot`). Slots whose source has more than 200,000 neighbours are ranked in blocks (no neighbour is dropped); none is skipped.
- **Selection effect, stated:** Part B reads only direct slots, the easier part. The D share of the cell is printed beside, and a statement about order is a statement about the slots the links already reach.

## 5. Gates

- **G-C1:** each cell's D count equals UL2P's any-link covered count in `outputs/ul2p/<d>/ledger.json` (per carve, kind, version), exactly.
- **G-C2:** the neighbour lists of the anchors, built from the incident pairs, give each anchor the degree `zul2p.node_deg` gives it on the same keys.
- **G-C3 (self-test, no real data):** D, T, N, `|M|`, via-gold, CC1, CC2 and the three ranks against a brute-force version on toy graphs (random graphs, ties, hubs, a slot whose ends share a neighbour that is the other end's gold).
- **G-C4:** the control's slots have no self-pair and the same count as the cell's slots, up to skips; the skips are reported.
- **G-C5:** the question rows are aligned: over the first 200 questions of a carve, the question cosine of each gold is above that of 1,000 fixed random nodes in at least 0.9 of the (gold, random) comparisons (a mis-aligned question row would give 0.5). The question rows and a sample of 1,000 node rows have norm 1 within 1e-3.
- **G-C6:** on a sample of 1,000 D slots the vectorised ranks equal a per-slot loop's, exactly.
- **G-C7:** the version's `B0` size equals the grade's (UL2P's G-P1), as the loader checks.

A gate failure stops the row and is reported; nothing is relaxed after a number is read.

### 5.1 Amendment A1 (26 Nov ~19:30, committed before MetaQA's re-run; no MetaQA or WebQSP number of this stage has been read)

- **What happened.** The MetaQA row (`cr1-metaqa`, first run) stopped at G-C5: the golds' question cosine was above the random nodes' in 0.619 of the comparisons, below 0.9. The row stops before Part A, so no D, T, N count and no rank of MetaQA has been read. MuSiQue's row (golds 0.9 or above on its carves) and the MultiHop-RAG read passed the gate as declared.
- **Diagnosis (a read-only probe on the host, not a number of this stage).** The question rows are aligned: MetaQA has 407,513 e3v question rows for 407,513 question ids, unit norm, and the five first-ranked results of the stored first-stage dense list are above 1,000 random nodes in 0.9838 of the comparisons (WebQSP: 4,737 rows for 4,737 ids, 0.9998; its golds 0.9541). The declared check assumed that a gold is a text that answers the question. On MetaQA and WebQSP a gold is an **answer entity** (a name), whose cosine to the question is not above a random node's in the share a passage's is, so the gold-based share measures the wrong thing there, not the alignment.
- **The change.** On MetaQA and WebQSP only, G-C5's reference nodes are the **five first-ranked results of the stored first-stage dense list** (`retrieval_cache/dense_top1000.npz`, `ids` by question row), with the same 200 questions, the same 1,000 fixed random nodes, the same 0.9 threshold and the same unit-norm checks; the golds' share is recorded beside it. Every other dataset keeps the gold-based check. A mis-aligned question row would still give about 0.5 and stop the row.
- **What does not change.** The slots, classes, ranks, scores, expectations X1-X8 and calls A and B, and the other gates. A gate stops a row or passes it; it adds nothing to a number. `zcr1.py selftest` passes (1,305 checks), now including the gate on both bases (aligned passage golds pass; unaligned golds stop without a list; a dense list of other questions stops) and the memory-mapped read of a stored and of a compressed `.npz`. MuSiQue's and MultiHop-RAG's ledgers were written by the first form of the gate; the re-run rows record the new form in `gates.G-C5`.

### 5.2 Amendment A2 (26 Nov, report code only)

Section 3 declares CC1 and CC2 by hop on MuSiQue CH and the ledger stores them (`by_hop`), but the first `report` printed no table of them. `zcr1.py report` now prints that table (CC1 and CC2 per hop count, with the intervals the ledger holds). No ledger, no row, no definition, no call changes; the table is a print of stored values.

### 5.3 Amendment A3 (26 Nov, before any Part C number): a fused order, and why an order is preferred where

**Why.** The first reads disagree on which order is best: the question cosine `fq` is above the pair's own dot `pd` on MultiHop-RAG (+3.92 points at rank 30) and MuSiQue CH (+8.32) and far below it on MetaQA (-25.94, the golds there are answer entities). The user's idea of 26 Nov: "check and see how and why we prefer something depending on how the dataset we have and the other scores accordingly and take a weighted score or something". Part C reads that on the same slots, sources, targets and neighbour lists, with nothing fitted.

**Part C, the orders.** For every D slot's source `s`, with `pd` and `fq` the two scores of Part B over `N(s)`:
- `sum`: `pd + fq` (raw cosines; the score that spreads the neighbours more weighs more).
- `zs`: the sum of the two scores' z-scores over `N(s)` (equal weights; a zero spread gives zero).
- `rrf`: reciprocal rank fusion, `1/(60 + rank_pd) + 1/(60 + rank_fq)`, ranks as in Part B (ties share the rank), the usual constant 60.
- `z25` and `z75`: the z-score sum with the weight 0.25 and 0.75 on `fq` (the rest on `pd`). These two exist to show **which weight a dataset prefers**; they are descriptive, no rule picks between them here.
Ranks, shares at 10/30/100, chance and the bootstrap over questions are as in Part B. The paired differences added per cut: `F - pd`, `F - fq`, `F - best` and `F - worst`, where best and worst are the better and the worse of `pd` and `fq` in that cell by total hits at that cut (this uses the gold hits: descriptive, labelled as such).

**Part C, the label-free statistics (no gold is read).** Per cell, over the D slots' sources: the mean and median of the Pearson correlation between `pd` and `fq` across the source's neighbours, the median standard deviation of `pd`, the median standard deviation of `fq`, and the median of their ratio. They say how much the two scores agree and how much each separates the neighbours, so an order's preference can be tied to a number that exists without labels.

**Part C, the call.** **C: SAFE_FUSION**, for each of the three fixed fusions `sum`, `zs`, `rrf` (not `z25`, `z75`): at rank 30 on the decide cells of the six datasets (the picks of call B), the fusion is **SAFE** if its paired difference to the better single order is not BELOW on at least 5 of the 6 datasets **and** its difference to the worse single order is ABOVE on at least 5 of 6; otherwise **NOT_SAFE** (PENDING while fewer than 6 datasets are read). A SAFE fusion would be a universal order with no dataset-dependent weight; a NOT_SAFE call closes only these three fixed forms: it does not say a weight chosen from the label-free statistics cannot work (that rule, CR2, is declared on its own page after this read, if the statistics show a pattern that holds on at least 5 of the 6 datasets).

**Expectations XC1-XC5 (written before any Part C number).**
- **XC1:** on MetaQA (and WebQSP when read) every fusion is BELOW `pd` and ABOVE `fq` at rank 30: `fq` there is close to noise against `pd`, so the fused order lands between the two, below the best by more than 5 points. About 0.75.
- **XC2:** on MultiHop-RAG and MuSiQue CH `zs` and `sum` are within 3 points of `fq`'s share at rank 30 and above `pd`'s. About even odds.
- **XC3:** **no fusion is SAFE** (the fused order is BELOW the better single order on at least 2 of the 6 datasets). About 0.6.
- **XC4:** the correlation between `pd` and `fq` is lower on MetaQA than on MultiHop-RAG and MuSiQue, and the standard deviation of `fq` relative to `pd`'s is lower on MetaQA than on those two. About 0.65.
- **XC5:** among the five weights (`pd` alone, `z25`, `zs`, `z75`, `fq` alone) the best at rank 30 is `fq` or `z75` on the passage datasets (MultiHop-RAG, MuSiQue, HotpotQA, 2Wiki) and `pd` or `z25` on the KBs (MetaQA, WebQSP), so the preferred weight splits by node type. About 0.55.

**Gates and rows.** G-C3 gains toy checks of `rank_all`, the fusions and the statistics against a per-element loop (ties included). **G-C8:** the three first-run ledgers (MultiHop-RAG, MuSiQue, MetaQA; copies kept as `ledger_pre_a3.json` beside each) are re-run with this code and their Part A cells and the `pd`, `fq`, `rq` shares, medians and paired differences of Part B must equal the first run's exactly (the new orders add keys; nothing old changes); a difference stops the reading. MuSiQue and MetaQA re-run as rows `cr1a3-musique` and `cr1a3-metaqa`, MultiHop-RAG on the laptop; the queued HotpotQA, 2Wiki and WebQSP rows run this code from their first send (a ledger without Part C is re-run under `cr1a3-<d>`).

### 5.4 Amendment A4 (26 Nov ~20:05, a gate made tolerant to float rounding, disclosed after a failure; no Part B number of HotpotQA had been read)

- **What happened.** HotpotQA's row (`cr1-hotpotqa`, first run) stopped at G-C6: 1 of the 1,000 sampled D slots had a rank by the blocked product different from the per-neighbour loop's. 2Wiki's row passed G-C6 with the same code. The row stops before any Part B number is written, so no HotpotQA rank had been read; Part A of its first cell (D 65.307, equal to UL2P's) had been logged.
- **Cause (the reading, not yet a per-slot proof).** A float32 matrix-vector product over a block and a scalar dot product of one row round differently in the last bits. Two neighbours whose true scores lie within that rounding of the target's score can swap order between the two computations, which moves the target's strict rank by one. HotpotQA's neighbour lists are long and contain near-duplicate nodes, so such near-ties are likely there.
- **The change, made after seeing the failure and said so.** G-C6 accepts a rank difference in one order only when it is **at most the number of neighbours whose score in that order lies within 1e-5 of the target's** (float32 rounding of a 1,152-term dot product is of the order of 1e-7); a larger difference still stops the row. The number of accepted differences is written to the ledger (`gates.G-C6`: `slots`, `rank_differences_within_float_noise`), so a large count would show a problem the gate hides. The ranks reported are the blocked ones, unchanged; no number of the stage depends on the gate. The first runs of MultiHop-RAG, MuSiQue, MetaQA and 2Wiki passed the exact form; they are not re-run for this.
- `zcr1.py selftest` passes with a hand-counted check of the noise-tie count.
- **Disclosure (added ~20:10, after the commit of A4).** The failed first run had written its first cell (RK, s1eval, GP) to a partial ledger (`partial: true`) before G-C6 stopped it in the next cell, and the report printed that cell's Part A and Part B rows when 2Wiki's ledger was fetched (the fetch glob carries the shared outputs folder), so the statement "no HotpotQA Part B number had been read" held when the amendment was written but not at 20:08. Those numbers came from a row whose G-C6 had not yet run; they are superseded by the retry's complete ledger and are not quoted anywhere until G-C6 has passed on it.

## 6. Expectations (written before any number)

- **X1:** the D share equals UL2P's any-link recall on every cell (this is G-C1; any difference stops the row).
- **X2:** T is large on the dense graphs: **D + T is at least 0.9 of the slots on MultiHop-RAG, HotpotQA and 2Wiki RK**, which would mean a two-link path is a weak signal there unless `|M|` is small. Held at about even odds.
- **X3:** on MuSiQue CH, **D + T is below 0.9** (the unlinked chain slots share no neighbour in B0 in a fair share), because the MHB bridges that are neither a title nor a name are not a link in any family. About even odds.
- **X4:** the median `|M|` of a T slot is **at least 10** on every dataset except MetaQA's SRK, so that a path through an unnamed intermediate is not selective by itself. Likely.
- **X5:** the control's D + T share is **at least 0.5 of the cell's own** on MetaQA and WebQSP (hub nodes: any two nodes share a neighbour), so that their T class carries little. Held with some confidence.
- **X6:** `fq` is above `pd` at rank <= 30 on every cell (the question carries information the pair's own dot does not). Likely.
- **X7:** **`rq` is above `fq` at rank <= 30 on at least 4 of the 6 datasets read (MuSiQue CH, MultiHop-RAG, HotpotQA, 2Wiki, MetaQA, WebQSP).** Held at about 0.4: the residual removes what the source says, which helps a chain and hurts a comparison pair (both golds answer the question), and R8 found the routed scorer below the flat question cosine.
- **X8:** `rq` is **below** `fq` on HotpotQA's and 2Wiki's comparison-heavy cells (their GP slots are mostly parallel pairs). About even odds.

## 7. Calls and effect sizes

- **A: CHAIN_WITHIN_TWO**, per dataset: CC2 minus the control's CC2, paired intervals; the call is ABOVE / LEVEL / BELOW by `zap4.call`. The size of the intermediate set is printed beside (median and 90th percentile of `|M|`). A high T share with a large `|M|` is **not** a recreation of the chain.
- **B: RESIDUAL_ORDER**, per cell: the paired difference `rq - fq` in the share of D slots with rank <= 30, by `zap4.call` (ABOVE, LEVEL, BELOW), at least 20 D slots (UL2P's MIN_T). Across datasets: **UNIVERSAL** if ABOVE on at least 5 of the 6 datasets' decide cells and BELOW on none; **NOT_UNIVERSAL** otherwise. The same for `fq - pd`. A call closes only its form: it does not say a path-aware order cannot work (a trained or relation-specific one is not read).
- Effect size beside every call: the points of the share, the chance share and the D share. The control is read, not subtracted into a call, except in A.

## 8. Cost, rows, outputs

- One row per dataset on the host (`python outputs/host_ops/pylib_run.py outputs/mp_unified/zcr1.py ledger --ds D --host`), mostly small: B0's pairs are filtered to those incident to the anchors (UL2P's `incident`), the neighbour lists are of the anchors only; WebQSP's anchors have thousands of neighbours each, the ranking is blocked (no matrix larger than 2 GB). MultiHop-RAG runs on the laptop (its vectors are local).
- Outputs: `outputs/cr1/<d>/ledger.json` (aggregated, small, committed with `git add -f`), `outputs/cr1/report.json` and `report.md` from `zcr1.py report`. No array is written.
- Checkpoints: the ledger is written per (carve, kind, version) cell as it finishes, so a restart loses one cell.

## 9. Not read here (named)

- The pool survival of the slots (GP3 A4, the GP1 session) and the composed-path pool grade (CR2, after this page).
- Chains for the KBs from the topic entity (the path the question names); a decomposition-free chain definition for MultiHop-RAG beyond its evidence pair.
- A trained or relation-specific path scorer; the answer-string form of the MHB B4 bridge (it needs the answer).
- SQuAD (no slot).

## 10. Results (read 26 Nov 2026, 20:30; six datasets with a slot, SQuAD n/a)

Every number is in `outputs/cr1/<d>/ledger.json` and `outputs/cr1/report.md`. The decide cell is `s1eval` (MuSiQue CH; RK on HotpotQA and 2Wiki, SRK beside); intervals are 95 percent over questions.

### 10.1 Provenance and gates

- **Rows.** MultiHop-RAG on the laptop (5.7 s); `cr1a3-musique` (27.9 s) and `cr1a3-metaqa` (66.7 s) on the host; `cr1-webqsp` (job 261126-200057-cr1-webqsp-8105, 117.9 s, peak working set 11.73 GB); `cr1-2wiki` (319.9 s, 7.04 GB); `cr1a4-hotpotqa` (job 261126-201912-cr1a4-hotpotqa-95fe, 203.4 s, 5.25 GB). The ledgers record the code hash: 8da021fa (the A3 code) on five, **96f0b28f (A4) on HotpotQA**. The five A3-code ledgers passed G-C6 in its exact form, so A4 changes none of their numbers.
- **G-C1:** every cell's D count equals UL2P's (MultiHop-RAG 2,350 of 2,747; MuSiQue CH 1,984 of 3,987 and GP 2,013 of 5,962; MetaQA 4,912 of 37,446; WebQSP 833 of 4,369; HotpotQA RK 4,836 and SRK 5,925 of 7,405; 2Wiki RK 11,522 and SRK 12,061 of 26,331). **G-C2** passes on all. **G-C5:** MultiHop-RAG 0.9874, MuSiQue 0.9873, HotpotQA 0.9995, 2Wiki 0.9937 (gold basis); MetaQA 0.9838 and WebQSP 0.9998 (stored dense top-5 basis, A1; their golds' share 0.6194 and 0.9541). **G-C6:** exact on five; **HotpotQA accepted one rank difference within float noise among 1,000 slots** (A4's count, in the ledger). **G-C8 PASS** on the three re-runs (MultiHop-RAG, MuSiQue, MetaQA).
- **Disclosure (HotpotQA).** The first run stopped at G-C6 (A4). The feeder's own retry (job 261126-201152-cr1-hotpotqa-4276, 20:12) ran the host's 19:51 copy of `zcr1.py`, which had no A4 change (the feeder does not re-send a changed input on a retry), and stopped on the same slot. The A4 file was then pushed (host and local hash equal, 96f0b28f) and the row re-queued as `cr1a4-hotpotqa`. Every HotpotQA number below is from that run's complete ledger. The two failed runs left partial ledgers that fetches delivered into the shared folder, and `report` printed their lines between 20:10 and 20:22; none of those lines is quoted here.

### 10.2 Part A: how far the gold slots are from linked (percent; decide cell)

| dataset | cell | slots | D | T | N | CC1 | CC2 | control D+T | median \|M\| of T | CC2 minus control |
|---|---|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | RK GP | 2,747 | 85.55 | 14.20 | 0.26 | 74.96 | 99.47 | 98.64 | 18 | +1.51 [0.71, 2.41] |
| MuSiQue | RK CH | 3,987 | 49.76 | 34.44 | 15.80 | 42.37 | 76.00 | 33.02 | 3 | +45.44 [42.78, 48.04] |
| MuSiQue | RK GP | 5,962 | 33.76 | 37.07 | 29.17 | 36.66 | 62.64 | 34.62 | 3 | +33.63 [31.15, 36.17] |
| MetaQA | SRK GP | 37,446 | 13.12 | 64.79 | 22.09 | 7.43 | 77.91 | 55.58 | 4 | +37.81 [36.11, 39.57] |
| WebQSP | SRK GP | 4,369 | 19.07 | 80.16 | 0.78 | 16.78 | 98.94 | 69.71 | 15 | +39.18 [35.11, 43.09] |
| HotpotQA | RK GP | 7,405 | 65.31 | 16.73 | 17.96 | 65.31 | 82.04 | 9.79 | 2 | +72.25 [71.26, 73.31] |
| HotpotQA | SRK GP | 7,405 | 80.01 | 6.63 | 13.36 | 80.01 | 86.64 | 10.45 | 2 | +76.19 [75.22, 77.16] |
| 2Wiki | RK GP | 26,331 | 43.76 | 20.59 | 35.65 | 50.93 | 65.81 | 24.11 | 1 | +45.68 [44.72, 46.63] |
| 2Wiki | SRK GP | 26,331 | 45.80 | 20.15 | 34.05 | 52.76 | 66.21 | 25.40 | 1 | +45.59 [44.65, 46.56] |

MuSiQue CH by hop count (questions; CC1 / CC2): 2 hops (1,252) 70.29 / 90.81; 3 hops (760) 14.47 / 63.29; 4 hops (405) 8.39 / 54.07. The share of T slots with `|M|` of 10 or more: MultiHop-RAG 0.782, WebQSP 0.635, MetaQA 0.428, MuSiQue CH 0.122, 2Wiki 0.017, HotpotQA 0.013. Via-gold shares of T: MultiHop-RAG 0.264, MetaQA 0.112, WebQSP 0.094, MuSiQue CH 0.011, 2Wiki 0.001, HotpotQA 0.

**Call A (CHAIN_WITHIN_TWO): ABOVE on every cell.** What the effect means differs by dataset: on **MuSiQue, HotpotQA and 2Wiki a T slot has few intermediates** (median 3, 2 and 1), and the control's D + T share is 10 to 33 percent against 64 to 87 for the cell's own slots, so a two-link path is a selective signal there. On **MultiHop-RAG the control's D + T is 98.64** (the cell's own 99.75): a shared neighbour exists for nearly any pair, the call is ABOVE by 1.5 points and says nothing. On **MetaQA and WebQSP the control's D + T is 56 and 70 percent** (the cell's own 78 and 99; hubs): ABOVE by 38 and 39 points, but with median `|M|` of 4 and 15 the path is not selective. Reached at two links, MuSiQue's chain slots are 84.2 percent (D 49.76 plus T 34.44); 15.8 percent have no path of two links in B0 at all.

### 10.3 Part B: the order of a source's neighbours (D slots, share with the target at rank 30 or better; chance beside)

| dataset | cell | D slots | median degree | pd | fq | rq | chance | fq minus pd | rq minus fq |
|---|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | RK GP | 2,350 | 75 | 94.89 | 98.81 | 51.45 | 42.43 | **+3.92 ABOVE** [2.87, 5.01] | -47.36 BELOW |
| MuSiQue | RK CH | 1,984 | 42 | 79.99 | 88.31 | 75.61 | 65.88 | **+8.32 ABOVE** [6.61, 9.99] | -12.70 BELOW |
| MetaQA | SRK GP | 4,912 | 4,970 | 34.63 | 8.69 | 2.95 | 5.89 | **-25.94 BELOW** [-27.39, -24.45] | -5.74 BELOW |
| WebQSP | SRK GP | 833 | 6,747 | 37.58 | 40.58 | 19.21 | 16.50 | **+3.00 LEVEL** [-3.26, +9.02] | -21.37 BELOW |
| HotpotQA | RK GP | 4,836 | 31 | 91.67 | 98.53 | 93.51 | 79.60 | **+6.87 ABOVE** [6.12, 7.61] | -5.03 BELOW |
| 2Wiki | RK GP | 11,522 | 41 | 93.23 | 94.05 | 70.86 | 70.76 | **+0.82 ABOVE** [0.35, 1.26] | -23.18 BELOW |

SRK beside: HotpotQA fq minus pd +8.19 ABOVE, rq minus fq -4.46 BELOW; 2Wiki +0.85 ABOVE, -24.03 BELOW. The carve `s1sel` agrees in sign on every dataset that has one (WebQSP has no `s1sel` cell; 2Wiki RK +0.14 and SRK +0.41 are LEVEL).

- **The cut matters.** At rank 30 several cells are near their ceiling (HotpotQA 98.5, MultiHop-RAG 98.8, against median degrees of 31 and 75), which understates the order. At rank 10 the paired difference fq minus pd is **+23.35 on HotpotQA, +11.24 on MuSiQue CH, +7.53 on MultiHop-RAG**, -0.73 LEVEL on 2Wiki, **-17.26 on MetaQA** and **-7.92 BELOW on WebQSP**; at rank 100 WebQSP is **+11.04 ABOVE** (and MetaQA -48.11). WebQSP's question order is behind the pair dot in the first ten, level at thirty, ahead at a hundred.
- The median rank as a share of the source's neighbours (pd / fq): MultiHop-RAG 0.113 / 0.095, MuSiQue CH 0.286 / 0.185, HotpotQA 0.289 / 0.118, 2Wiki 0.220 / 0.217, WebQSP 0.026 / 0.027, MetaQA **0.023 / 0.381**.

**Call B, residual: `rq` minus `fq` is NOT_UNIVERSAL: BELOW on 6 of 6** (-4.5 to -47.4 points). **Call B, question order: `fq` minus `pd` is NOT_UNIVERSAL** (the rule needs ABOVE on at least 5 and BELOW on none): ABOVE on 4 (MultiHop-RAG, MuSiQue, HotpotQA, 2Wiki), LEVEL on WebQSP, BELOW on MetaQA.

### 10.4 Part C: fused orders and the label-free statistics (rank 30; decide cell)

| dataset | pd | fq | zs | sum | rrf | z25 | z75 | best single | zs minus best | z75 minus best | r(pd, fq) | sd fq / sd pd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MultiHop-RAG | 94.89 | 98.81 | 98.47 | 98.30 | 98.43 | 97.19 | 98.85 | fq | -0.34 LEVEL | +0.04 LEVEL | 0.895 | 0.758 |
| MuSiQue CH | 79.99 | 88.31 | 87.45 | 86.69 | 87.65 | 85.13 | 88.31 | fq | -0.86 LEVEL | 0.00 LEVEL | 0.656 | 0.770 |
| MetaQA | 34.63 | 8.69 | 26.51 | 26.36 | 23.66 | 33.86 | 19.16 | pd | **-8.12 BELOW** | **-15.47 BELOW** | 0.377 | 1.017 |
| WebQSP | 37.58 | 40.58 | 42.14 | 40.22 | 42.38 | 38.78 | 43.34 | fq | +1.56 LEVEL | +2.76 ABOVE | 0.637 | 0.734 |
| HotpotQA | 91.67 | 98.53 | 96.94 | 96.69 | 97.25 | 94.79 | 98.16 | fq | -1.59 BELOW | -0.37 BELOW | 0.835 | 0.912 |
| 2Wiki | 93.23 | 94.05 | 94.64 | 94.58 | 94.80 | 94.11 | 94.71 | fq | +0.60 ABOVE | +0.66 ABOVE | 0.899 | 0.860 |

**Call C (SAFE_FUSION): NOT_SAFE for `sum`, `zs` and `rrf`.** None is BELOW the better single order on 5 of 6 datasets (`sum` on 3, `zs` and `rrf` on 4); each is ABOVE the worse single order on 6 of 6. The z75 column is **descriptive, post hoc** (z75 is a declared order but not one of call C's three): at rank 30 it is within 0.4 points of the best single order on MultiHop-RAG, MuSiQue CH and HotpotQA, above it on 2Wiki and WebQSP, and 15.5 points below it on MetaQA.

### 10.5 The expectations, one by one

| | what was written | result |
|---|---|---|
| X1 | D equals UL2P's recall on every cell | **Held** (G-C1, exact) |
| X2 | D + T at least 0.9 on MultiHop-RAG, HotpotQA, 2Wiki RK | **Not held:** 99.75, 82.04, 64.35; true on one of three. T is small and selective on the other two (median `\|M\|` 2 and 1) |
| X3 | MuSiQue CH D + T below 0.9 | **Held:** 84.20 |
| X4 | median `\|M\|` of T at least 10 except on MetaQA | **Not held:** 18 and 15 on MultiHop-RAG and WebQSP; 3, 2, 1 on MuSiQue, HotpotQA, 2Wiki |
| X5 | control D + T at least 0.5 of the cell's on MetaQA, WebQSP | **Held:** 0.71 (55.58 of 77.91) and 0.70 (69.71 of 99.23) |
| X6 | `fq` above `pd` at rank 30 on every cell | **Not held:** ABOVE on four datasets, LEVEL on WebQSP, BELOW on MetaQA by 25.9 |
| X7 | `rq` above `fq` on at least 4 of 6 | **Not held, the opposite:** BELOW on 6 of 6; the residual is the worst of the three orders on five datasets (on HotpotQA it is above the pair dot, 93.51 against 91.67) |
| X8 | `rq` below `fq` on HotpotQA and 2Wiki | **Held**, but not specific: it is below on MuSiQue's chain cells (-12.70) and MultiHop-RAG (-47.36) too, so the residual hurts chains as much as comparisons |
| XC1 | on MetaQA and WebQSP every fusion is BELOW `pd` and ABOVE `fq`, more than 5 below the best | **Held on MetaQA** (fusions 8.1 to 11.0 below `pd`, `zs` 17.8 above `fq`); **not held on WebQSP**, where `fq` is the better single order and the fusions are level with it and 2.6 to 4.8 above `pd` |
| XC2 | on MultiHop-RAG and MuSiQue CH `zs` and `sum` within 3 points of `fq` and above `pd` | **Held** (-0.34 and -0.51; -0.86 and -1.61) |
| XC3 | no fusion is SAFE | **Held** (call C NOT_SAFE for all three) |
| XC4 | r(pd, fq) lower on MetaQA than on MultiHop-RAG and MuSiQue, and sd(fq) / sd(pd) lower | **Not held as written:** first half true (0.377 against 0.895 and 0.656); second half false (1.017 against 0.758 and 0.770: on MetaQA the question cosine spreads the neighbours as much as the pair dot, to no use) |
| XC5 | best weight is `fq` or z75 on the passage datasets and `pd` or z25 on the KBs | **Not held:** true on five of six (MultiHop-RAG z75, MuSiQue CH `fq` and z75, HotpotQA `fq`, 2Wiki z75, MetaQA `pd`); **WebQSP's best weight is z75 (43.34)**, the passage side, not the KB side |

### 10.6 What the stage says and does not say

- **Chains can be recreated from the links on the passage datasets, with a small intermediate set**: MuSiQue 84.2 percent of the oracle's chain slots within two links at a median of 3 intermediates (by hop 90.8, 63.3 and 54.1 percent of questions with every slot within two links), HotpotQA 82.0 (RK) and 86.6 (SRK), 2Wiki 65.8 and 66.2. This is the whole graph, not a pool: whether a walk's pool keeps these paths is GP3 amendment A4 (the GP1 session's), not read here. The 15.8 percent of MuSiQue's slots with no two-link path, and 2Wiki's 34 to 36 percent, are links no rule has made.
- **Order by the question**: on the four passage datasets the question cosine puts the next hop ahead of the pair's own dot (+3.9 to +8.3 at rank 30, +7.5 to +23.4 at rank 10 except 2Wiki), on WebQSP it is behind in the first ten and ahead at a hundred, and **on MetaQA it fails (-25.9 at rank 30, median rank share 0.38 against 0.02)**. The reading is that MetaQA's gold slots are answer to answer (names), a slot the question's words do not describe; the topic-to-answer path of a KB question, which is where the relation words matter, is **not a slot here** (section 3).
- **The residual order hurts everywhere** (X7): removing the source's direction from the question loses 4.5 to 47 points against the question cosine. No path-aware form of this kind is carried.
- **No weight is chosen.** Five of the six datasets prefer a question-heavy weight (`fq` or z75), MetaQA alone prefers `pd`. The label-free statistic r(pd, fq) is lowest on MetaQA (0.377) but does not order the other five (MuSiQue 0.656 has a +8.3 gain from `fq`, 2Wiki 0.899 has +0.8, WebQSP 0.637 has +3.0). A rule that picks the weight from it would rest on one dataset's reversal, not on a pattern holding on 5 of 6, so **CR2 (a label-free weight) is not declared on this evidence**. A fixed question-heavy order (z75) is within 0.4 of the best on three datasets at rank 30 and costs 15.5 on MetaQA; to become a rule it needs its own page, declared before any number, and a test on data it has not seen.
- **Not said:** nothing here is a pool grade, a trained scorer or a topic-to-answer KB walk; the pair dot is the pair's own row dot (GP1's), not a PPR or a graph score; rank 30 is the declared cut and understates the order where lists are short.

### 10.7 Named for the next pages (not declared here)

1. **KB topic-to-answer ordering with the relation labels** (MetaQA first): the S layer's edges carry relation labels (`S.<rel>`); the question's relation words may order a topic entity's neighbours where an answer's name cannot (the user, 26 Nov: "scoring would work so improve it, metaqa needs proper work").
2. **A redundancy census of the links** (duplicate and near-duplicate nodes, links that add no new document), before any rule that prunes or prioritises.
3. **A held-out semi-structured KB** (STaRK-Prime) for every rule frozen by then, so that MetaQA's reversal is read as an outlier or as a class from more than two KBs (the user, 26 Nov: CWQ shares WebQSP's KB and is not added; "only stark" will).
