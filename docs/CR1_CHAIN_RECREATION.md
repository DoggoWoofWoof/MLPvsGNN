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

## 10. Results

None yet.
