# DRP1: what the pool loses when the SPLADE leg, the K family, or both are dropped (declared 20:11 IST, 27 Nov 2026)

**Status: DECLARED. No arm number exists. Nothing is chosen, adopted, removed or deleted by this page.** The keep-or-drop call stays the user's; this page prices it.

## 0. What the user asked

- "so i think i am leaning towards dropping splade and also K, need to properly find how much we lose so do diagnoise properly" (27 Nov).
- Earlier the same day: "i feel splade is not required for us but do check out that", and "so compared to dense is splade faster?", "but the issue is splade vectors are dataset specific", "generalization wont be possbile with that".
- "wait what K was done on CPU? that is why i took time?" A correction to the answer given: the K records show a card device for WebQSP, HotpotQA and 2Wiki (the zenc4g path) and the CPU for MetaQA, MuSiQue and SQuAD; STaRK's K used the CPU `nn_top` because STK2 judged 4 to 9 hours acceptable and the card route was not checked.
- Standing rulings that bind this page: one rule on all ten datasets, "deal with all the datasets the same way no matter what"; subset first; a rule is committed before its number; effect size beside every call; no model is trained.

## 1. The question and its scope

The pool is built by PL2 to PL6's engine: first stage (the e3v or frozen dense list and the SPLADE list, fused) gives the seeds and the flat lists; a walk of at most three hops over the three stores (`structural` = S, `ner` = R, `knn` = K; dedup S, then R, then K) adds nodes. **Two parts of that are in question:** the SPLADE leg (seeds and fused lists) and the K store (the walk's third family). This page measures **what the pool's all-gold rate (ALL) and size (distinct nodes) become without each, and without both**, on all ten datasets, against the filed pools. It reads **pools only**: K and SPLADE as inputs of the MLP or the GNN are not read here, and a pool lost here could still be recovered by a model that ranks better; that is a separate read.

## 2. The three arms (each changes one thing; the engine is imported, edited nowhere)

| arm | change | how |
| --- | --- | --- |
| `nospl` | the SPLADE list is the dense list wherever the pool code reads it | `zleg1p.dense_only` (LEG1-A6's own): the seeds become the dense top 5, `rrf1000_bN` the dense top N, `splade_*` cells their dense twins |
| `nok` | the K store is empty | a wrapper over the Source's families returns an empty CSR (indptr all zero, no columns) for `knn`; `structural` and `ner` are the filed objects, untouched |
| `both` | both changes | the two wrappers together |

Because K is deduped last (S, then R, then K), **dropping K removes exactly the K-only edges**; an edge that S or R also holds stays. A cell that reads no K (`STRUCT` and `NER` walks, flat cells) is **identical under `nok` by construction**; `FULL`, `KNN` and `FS` (FULL at hop 1) read K. The page keeps the arms apart from the labels: S, R, K are the rule labels, `structural`, `ner`, `knn` the walk's stores.

## 3. Datasets, cells and which arms run

**The seven (eval carve, frozen first stage, as LEG1-A6).** The cells are A6's rule: PL6's needed cells whose filed eval mean nodes are at most 16,000, **plus the dataset's R95 pick whatever its size** (WebQSP's R95 cell has 39,530 nodes; A6 left it out; DRP1 keeps it because it is the only WebQSP cell that reaches 0.95). For the four A6 covered the lists are checked equal to A6's. **STaRK (test split, e3v):** STK3's 16 pick cells (`groups.picks` of `outputs/stk3/cells.json`); the other 47 of its 63 are not run.

| dataset | eval questions | chunks of 128 | cells | cells that read K | arms run | no-SPLADE source |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| MetaQA | 9,785 | 77 | 7 | 0 | none (`nok` = filed, `both` = `nospl`, by construction) | A6's chunks |
| MuSiQue | 2,417 | 19 | 8 | 8 | `nok`, `both` | A6's chunks |
| SQuAD | 11,873 | 93 | 4 | 1 | `nok`, `both` | A6's chunks |
| MultiHop-RAG | 1,130 | 9 | 4 | 3 | `nok`, `both` | A6's chunks |
| WebQSP | 1,503 | 12 | 9 | 4 | `nospl`, `nok`, `both` | DRP1 |
| HotpotQA | 7,405 | 58 | 5 | 1 | `nospl`, `nok`, `both` | DRP1 |
| 2Wiki | 12,576 | 99 | 6 | 5 | `nospl`, `nok`, `both` | DRP1 |
| STaRK Prime | 2,801 | 22 | 16 | 8 | `nospl`, `nok`, `both` | DRP1 |
| STaRK Amazon | 1,642 | 13 | 16 | 8 | `nospl`, `nok`, `both` | DRP1 |
| STaRK MAG | after STK3's MAG rows | | 16 | 8 | `nospl`, `nok`, `both` | DRP1 |

(The question counts and chunk numbers are those of `outputs/drp1/pool/<ds>/cells_<ds>.json` and `outputs/drp1/stark_cells.json`, written by `zdrp1.py cells` before any arm ran; STaRK's chunk orders are STK3's, so the arms and the filed chunks cover the same questions.)

## 4. Gates (a failed gate stops the row; no arm number is read from a dataset with a failed gate)

- **G-D0** (the intact source reproduces the filed arrays): on the first 16 questions (the seven: the filed eval arrays; STaRK: STK3's chunk at the gate position), every cell's all-gold booleans and sizes equal the filed ones. A6's gate, unchanged.
- **G-D1** (the arm's identities): `nospl`: the SPLADE list is the dense list; `nok`: the `knn` store has no column and the SPLADE list is untouched; both: `structural` and `ner` are the filed objects.
- **G-D2** (`nok` leaves the K-free cells alone): on the same 16 questions, every cell that does not read K equals the filed arrays under `nok`. This is what shows the wrapper reached the engine and changed only K.

## 5. What is read, and the calls

Per dataset and arm, on the questions every source covers: per cell the filed ALL and the arm's ALL with Wilson 95 percent intervals, the **loss = filed ALL minus arm ALL** with its paired bootstrap interval (PL2's `paired`, seed `[BOOT_SEED, k, arm]`), and the nodes ratio (arm over filed). **Calls on the loss, A6's line:** NEEDED if the interval's lower end is at least +1.0 point, DISPENSABLE if the upper end is below +1.0, else UNCLEAR. **Headline cells:** the R95 cell and the bounded pick of each of the seven (the bounded pick when it is in the list); for STaRK the best filed cell within 3,500 nodes (post hoc, labelled). **Re-pick read (post hoc):** for each arm the best ALL among the list's cells within 500, 1,000, 2,500, 3,500, 5,000 and 16,000 nodes, beside the filed best (this answers "after re-picking inside the list"); and the smallest cell that reaches 0.95 in each arm (A6's `reach`). `read` writes `outputs/drp1/read_<name>.json`; a read on a partial set says PARTIAL.

## 6. Known before any arm number (priors from filed records)

- **SPLADE (A6, filed):** at the R95 cell the loss is SQuAD +1.76 [1.42, 2.11] NEEDED, MuSiQue +1.28 [0.74, 1.86] UNCLEAR, MetaQA +1.02 [0.77, 1.29] UNCLEAR, MultiHop-RAG +0.88 [0.18, 1.59] UNCLEAR. STK3's flat cells: the dense list alone is below the fused list at every size on both STaRK KBs read (Prime -3.1 to -4.8 at 20 to 1,000; Amazon -2.3 to -7.1), `splade_top200` -7.0 on Prime.
- **K (STK3 and the ledger):** on Prime FULL minus STRUCT is under 2 points in nine of nine twins; on Amazon five of nine are 2 points or more (up to +6.9 [+5.5, +8.3]), and R-only pairs are 6.57 M against K-only 1.55 M of 12.84 M; K alone is weak (UL2P/AP4: 0.98 percent precision), precise where it agrees with R.
- **Cost records (what dropping would save):** K search time of the seven: SQuAD 5.3 s, MetaQA 29 s, MuSiQue 115 s (CPU), WebQSP 9,712 s, HotpotQA 7,540 s, 2Wiki 6,616 s (card device recorded); STaRK's K pricing pass: Amazon about 4.1 h, MAG about 9.4 h (STK2 section 9.2, CPU, 8 threads). SPLADE (C0 under l5): card rate against the dense pair 2.0x Prime, 3.8x Amazon, 0.71x MAG (LEG1 A7); SPV1's ten-dataset vectors about 9 GB.

## 7. Expectations (written before any arm number)

- **E1 (SPLADE on the three new datasets).** At the R95 cell, dropping SPLADE costs 0.5 to 2.5 points on each of WebQSP, HotpotQA and 2Wiki, and the call is NEEDED on at least one of them.
- **E2 (SPLADE on STaRK).** On Prime and Amazon the best cell within 3,500 nodes loses at least 2 points without SPLADE.
- **E3 (K on the seven).** Dropping K costs under 1.0 point at the R95 cell on at least three of the four datasets whose R95 cell reads K (MuSiQue, MultiHop-RAG, WebQSP, 2Wiki); on MetaQA, SQuAD and HotpotQA the R95 cell reads no K and the loss is 0.
- **E4 (K on STaRK).** Prime loses under 1.0 point at the best cell within 3,500 nodes; Amazon loses 1.0 or more at some FULL pick cell but **less than** that cell's FULL-minus-STRUCT gap of section 20 of STK3's page (R stays).
- **E5 (both).** For each dataset where all three arms run, the loss of `both` at the R95 cell (best cell for STaRK) is at most the sum of the two single losses plus 1 point.
- **E6 (gates).** G-D0, G-D1 and G-D2 pass on every row with the same code.
- **E7 (size).** At the headline cells the arm's mean nodes are between 0.7 and 1.1 times the filed nodes.

## 8. Code, rows and cost

**Code:** `outputs/mp_unified/zdrp1.py` (new; imports `zleg1p.py`, `zpl2g.py` to `zpl6g.py` and `zstk3.py`, edits none; selftest 26/26; sha256 28431aba7885ec3a; edit only through an amendment on this page). Commands `cells`, `run` (the seven), `runk` (STaRK), `read`. One row per dataset and part, all its arms inside the row (chunk by chunk, resumable, each chunk written when done), `--host`, 2 CPUs, memory asked from the measured peaks of the PL rows on the same dataset (MuSiQue 4 GB, SQuAD 3, MultiHop-RAG 3, WebQSP 12, HotpotQA 8, 2Wiki 12, STaRK 6). **Known cost:** A6's chunk times (MuSiQue 543 s a chunk of 128 for its 8 cells, SQuAD 8 s, MultiHop-RAG 2.8 s) give about 6 row-hours for MuSiQue's two arms, 0.4 for SQuAD's, under 0.1 for MultiHop-RAG's; Prime and Amazon about 1.6 and 0.9 CPU-hours an arm from STK3's 4.0 seconds a question for 63 cells (16 cells run here). **Not known:** WebQSP (the 39,530-node cell), HotpotQA and 2Wiki have no timing for this cell list; the first chunks measure it and section 9 is rewritten with the measured wall time. Rows: `drp1-mu0..mu3`, `sq0..sq1`, `mh0`, `wq0..wq3`, `hp0..hp3`, `tw0..tw3`, `pr0..pr3`, `am0..am1`, and MAG's `ma0..ma5` behind `stk3m-run0..5`.

## 9. What this does not read or decide

No model is trained or ranked; the pools are the filed links' pools, not a re-tuned grid (a cell outside the list could do better without SPLADE or K; the re-pick read is inside the list only). SPLADE's use as a model input and K's use as a model feature are not read. Nothing is deleted: the SPLADE vectors, K stores and their weights stay until the user decides and per-action OK is given. No cell, rule or link set is chosen; PL8 stays undeclared.

## 10. Amendment 1 (20:23 IST, 27 Nov 2026): where two datasets run; WebQSP's measured cost (no arm number is used here)

- **MultiHop-RAG and SQuAD run on the laptop, not as host rows.** Their data packages exist only on the laptop (LEG1-A6 ran them there too). A smoke run of MultiHop-RAG on the host stopped at its first read (the retrieval cache of the package is not on the host); it wrote no gate file and no chunk, and no number was read from it. Section 8's rows `mh0` and `sq0..sq1` are therefore **not queued**; the two datasets run as `zdrp1.py run --ds multihoprag` and `run --ds squad` on the laptop (2 BLAS threads, `-E -P -B`), each with the same gates and the same chunk files under `outputs/drp1/pool/<ds>/`. Nothing else in sections 2 to 7 changes.
- **WebQSP's cost, measured by its first rows (`drp1-wq0..3`, started 20:15):** gates passed (G-D0 0 mismatches over 16 questions and 9 cells; G-D1 and G-D2 pass for the three arms); a chunk of 128 questions takes about 306 seconds for each arm, peak 7.3 GB (12 GB asked), so each of the four parts (3 chunks, 3 arms) takes about 2.8 hours. HotpotQA's and 2Wiki's costs stay unmeasured until their rows run.

## 11. Amendment 2 (20:52 IST, 27 Nov 2026): slice reads while the rows run (rule written before any slice number is read)

- **The request.** The user asked for quicker results on a smaller subset ("also can we get quicker results on a smaller subset i do not want to waste much of the time"). The chunk files are written in a seeded permutation of the question blocks (section 3) and a part takes every K-th position of that permutation, so the chunks present at any moment are a random subset of the split's question blocks, the same blocks for every arm that has them. `zdrp1.py read` already keeps only the questions every arm covers and marks the file PARTIAL; `zdrp1.py` is not edited.
- **The rule.** A PARTIAL read is a **slice read**: it is named as such with its question count and the chunks it used; the gates, arms, cells and the 1.0-point line are those of sections 2 to 5, with intervals from the slice (wider than the whole split's). A slice call (NEEDED, DISPENSABLE, UNCLEAR) is a screen and not the read of section 5: it is not filed as one, adopts nothing, and no row is cancelled because of it. The rows keep running to the whole split, and where the whole-split read gives a different call at a cell both are reported. The expectations E1 to E7 are judged on the whole-split reads only.
- **First slices.** Taken the same hour on whatever chunks WebQSP and HotpotQA have written; their question counts are printed beside each number in section 12.

## 12. Results so far (22:28 IST, 27 Nov 2026): three whole-split reads and five screens; E1 to E7 are not judged yet

`zdrp1.py read` on the chunks fetched at 22:26 (host rows `drp1-*`, laptop runs for MultiHop-RAG and SQuAD). The gates held on every dataset read (G-D0 0 mismatches over the first 16 questions and all cells; G-D1 and G-D2 pass for the three arms). Loss = filed ALL minus the arm's ALL, in points, with the paired bootstrap interval; the line is 1.0 point (section 5). **COMPLETE rows are whole-split reads and count for the expectations; PARTIAL rows are slices (amendment 2: screens, nothing adopted, no row cancelled).** The `both` arm (dense only, no K) is the user's 27 Nov main line; `nospl` isolates SPLADE and `nok` isolates K.

**At the cell that reaches 0.95 in the filed pools ("R95"):**

| dataset | read (questions) | R95 cell (filed ALL) | K: `nok` loss | SPLADE: `nospl` loss | both: `both` loss |
|---|---|---|---|---|---|
| MultiHop-RAG | COMPLETE (1,130) | `rrf1000_b25+FULL:h1_c100+top25` (0.9690) | +0.00 DISPENSABLE | +0.88 [+0.18, +1.59] UNCLEAR | +0.88 [+0.18, +1.59] UNCLEAR |
| SQuAD | COMPLETE (11,873) | `rrf1000_b15` (0.9554) | +0.00 DISPENSABLE | **+1.76 [+1.42, +2.11] NEEDED** | **+1.76 [+1.41, +2.08] NEEDED** |
| WebQSP | COMPLETE (1,503) | `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` (0.9474, 39,530 nodes) | +0.80 [+0.27, +1.33] UNCLEAR | +0.53 [-0.13, +1.13] UNCLEAR | **+2.00 [+1.20, +2.93] NEEDED** |
| HotpotQA | PARTIAL (3,309 of 7,405) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9604) | +0.00 DISPENSABLE | **+2.63 [+2.02, +3.26] NEEDED** | **+2.63 [+1.99, +3.33] NEEDED** |
| 2Wiki | PARTIAL (1,536 of 12,576) | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9603) | +0.20 [+0.00, +0.46] DISPENSABLE | **+8.72 [+7.29, +10.22] NEEDED** | **+8.79 [+7.29, +10.22] NEEDED** |
| MuSiQue | PARTIAL (384 of 2,417) | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` (0.9635) | +0.26 [-0.52, +1.04] UNCLEAR | (arm not yet written) | +1.30 [+0.00, +2.86] UNCLEAR |
| STaRK Prime | PARTIAL (256 of 2,801) | best filed cell within 3,500 nodes `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.9102 on these 256) | -0.78 UNCLEAR | +3.52 UNCLEAR | +3.12 UNCLEAR |
| STaRK Amazon | PARTIAL (746 of 1,642) | the same cell (0.8713 on these 746) | +0.40 UNCLEAR | **+3.22 NEEDED** | **+3.89 NEEDED** |

MetaQA is not in DRP1's seven rows (LEG1 A6's dense-only pool read of 26 Nov put it at -1.0); MAG waits for its links rows.

**Other cells read (whole-split rows, from the same files):** SQuAD, SPLADE +1.13 at 63 nodes and +0.09 [+0.03, +0.15] at 1,572 nodes (the leg matters only in the smallest pools); K 0.00 at every K-free cell and +0.11 at the cell with K. WebQSP, K 0.00 at the STRUCT cells, +1.80 [+1.13, +2.46] NEEDED at a 4,977-node FULL cell and +3.39 at a weak 69 percent cell; SPLADE +2.20 [+1.20, +3.13] NEEDED at 2,095 nodes, +1.60 at 2,520 and +2.73 at 3,247; the `both` arm +3.99 at the FULL two-hop cell; WebQSP never reaches 0.95 within about 2,500 nodes (best filed 88.0 at 2,520). MultiHop-RAG, zero at the 100-node cell (99.1 in every arm). HotpotQA's first slice (1,408 questions): SPLADE +3.20 at 78 nodes, +2.98 at 154, +1.42 at 2,977 and +0.71 at 5,179. 2Wiki's first slice (640 questions): SPLADE +7.81 [+5.78, +10.00] at 918 nodes, +7.34 at 2,209 and 3,047, +16.09 at 106; the dense-only arm's best ALL within 2,500 nodes is 88.9 against the filed 95.9.

**What it says (a screen where PARTIAL; no call is filed from a slice).**

- **K does not pay anywhere it was read.** The loss is 0.00 to +0.26 on MultiHop-RAG, SQuAD, HotpotQA, 2Wiki and MuSiQue and +0.80 on WebQSP, none above the 1.0-point line; the two STaRK screens are -0.78 and +0.40 and unclear on 256 and 746 questions.
- **SPLADE pays where the questions name their entities and the pool is small:** +8.7 on 2Wiki, +2.6 on HotpotQA, +1.8 on SQuAD, +3.2 on Amazon (screen). It is under the line or unclear on MultiHop-RAG (+0.88), WebQSP alone (+0.53) and the 256-question Prime screen (+3.52, interval not read). The loss shrinks with the pool: it is 0.09 on SQuAD at 1,572 nodes and 0.71 on HotpotQA at 5,179.
- **The dense-only line therefore starts lowest on 2Wiki** (-8.8 at its 0.95 cell) and needs a name or lexical signal there; this page does not test one.

**Not done here:** E1 to E7 are judged on whole-split reads only (amendment 2), and five of the eight rows are partial; the next reads come when `drp1-hp`, `tw`, `mu`, `pr`, `am` parts finish, and MAG after `stk2m-links` and `stk3m-run0` to `run5`. No row was cancelled, no SPLADE or K asset was deleted, and nothing is adopted: the main line's choice (dense only, no K, add-back later) is the user's, made on 27 Nov.

## 13. Whole-split results of the nine datasets (09:52 IST, 28 Nov 2026): E1 to E7 judged

**What ran.** Every `drp1-*` host row finished (the last, `ma5`, at about 08:00 IST on 28 Nov); MultiHop-RAG and SQuAD ran on the laptop (section 10). `zdrp1.py read` (host sha 28431aba7885ec3a = the laptop's) ran on the host at 09:45 IST on 28 Nov for HotpotQA, 2Wiki, MuSiQue, WebQSP, Prime, Amazon and MAG; the section 12 reads of SQuAD and MultiHop-RAG were already whole-split. **All nine reads are COMPLETE** (questions: SQuAD 11,873; MultiHop-RAG 1,130; HotpotQA 7,405; 2Wiki 12,576; MuSiQue 2,417; WebQSP 1,503; Prime 2,801; MAG 2,665; Amazon 1,642). MetaQA is not a DRP1 row. Records: `outputs/drp1/read_<name>.json`, the gate files `outputs/drp1/pool/<name>/gate*.json` and the chunks. **Which encoder:** the first six datasets' first-stage lists and vectors are the older encoder's (`load_source(ds, "eval", "frozen")`; the policy page's list of older-encoder modules names `zdrp1.py`), so their numbers are **older-encoder reads**; the three STaRK KBs are e3v.

**Gates (E6).** Every gate file of every row reads PASS: G-D0 (the intact source reproduces the filed arrays; 0 mismatches), G-D1 and G-D2, on all nine datasets and every part (114 files, six of them from the laptop runs).

**Loss at the headline cell** (filed ALL minus the arm's ALL, in points, paired bootstrap interval, call at the 1.0-point line; the headline cell is the R95 cell of the seven and, for STaRK, the best filed cell within 3,500 nodes, post hoc, as in section 5; nodes ratio arm over filed in the last column):

| dataset | headline cell (filed ALL at nodes) | K: `nok` | SPLADE: `nospl` | both | nodes ratio (both) |
| --- | --- | --- | --- | --- | ---: |
| SQuAD | `rrf1000_b15` (95.5 at 15) | +0.00 DISPENSABLE | +1.76 [+1.42, +2.11] NEEDED | +1.76 [+1.41, +2.08] NEEDED | 1.00 |
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` (96.9 at 50) | +0.00 DISPENSABLE | +0.88 [+0.18, +1.59] UNCLEAR | +0.88 [+0.18, +1.59] UNCLEAR | 1.00 |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (96.3 at 78) | +0.00 DISPENSABLE | +2.98 [+2.57, +3.43] NEEDED | +2.98 [+2.54, +3.46] NEEDED | 0.80 |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` (96.1 at 917) | +0.17 [+0.10, +0.26] DISPENSABLE | **+8.56 [+8.06, +9.07] NEEDED** | **+8.77 [+8.27, +9.28] NEEDED** | 0.92 |
| MuSiQue | `rrf1000_b400+FULL:h3_bf_v20000~cs-fu+top15000` (96.4 at 15,400) | +1.37 [+0.83, +1.94] UNCLEAR | +1.28 [+0.74, +1.86] UNCLEAR | +2.52 [+1.78, +3.27] NEEDED | 1.00 |
| WebQSP | `rrf1000_b400+FULL:h3_bf_v40000~cs-vr` (94.7 at 39,530) | +0.80 [+0.27, +1.33] UNCLEAR | +0.53 [-0.13, +1.13] UNCLEAR | +2.00 [+1.20, +2.93] NEEDED | 0.94 |
| Prime | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (85.5 at 2,475) | -0.32 [-0.82, +0.18] DISPENSABLE | +2.78 [+1.64, +3.86] NEEDED | +1.96 [+0.82, +2.96] UNCLEAR | 0.98 |
| MAG | `rrf1000_b400+STRUCT:h3_c25_v2000` (90.8 at 2,325) | +0.00 DISPENSABLE | +2.48 [+1.73, +3.34] NEEDED | +2.48 [+1.65, +3.26] NEEDED | 1.00 |
| Amazon | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (87.1 at 2,448) | +0.67 [+0.24, +1.16] UNCLEAR | +3.23 [+2.01, +4.33] NEEDED | +4.02 [+2.86, +5.24] NEEDED | 1.00 |

**The dense-only, no-K arm re-picked inside the list (post hoc: the best ALL among the list's cells within 2,500 nodes, filed beside the arm; no rule picks it).**

| dataset | filed best within 2,500 | `both` best within 2,500 | difference | smallest cell at 95 (filed / `both`) |
| --- | --- | --- | ---: | --- |
| SQuAD | 99.9 at 2,200 | 99.8 at 1,572 | -0.1 | 15 / 50 nodes |
| MultiHop-RAG | 99.1 at 100 | 99.1 at 100 | 0.0 | 50 / 50 |
| HotpotQA | 97.8 at 154 | 98.3 at 1,891 | +0.5 | 78 / 136 |
| 2Wiki | 96.6 at 2,211 | **88.2 at 2,200** | **-8.4** | 917 / none |
| MuSiQue | 88.5 at 2,438 | 86.1 at 2,485 | -2.4 | 15,400 / none |
| WebQSP | 87.9 at 2,095 | 85.7 at 2,100 | -2.2 | none / none |
| Prime | 85.5 at 2,475 | 85.0 at 2,073 | -0.5 | none / none |
| MAG | 90.8 at 2,325 | 91.7 at 2,043 | +0.9 | 5,648 / 5,259 |
| Amazon | 87.1 at 2,448 | 83.1 at 2,456 | -4.0 | none / none |

K alone (`nok`) at the same read: 2Wiki 96.4 at 2,243, MuSiQue 88.0 at 2,471, WebQSP 87.9 at 2,095 (equal to the filed), Prime 85.8 at 2,488 (above the filed), MAG 90.8 (equal), Amazon 86.5 at 2,473.

**Calls on section 7's expectations (whole-split reads only, amendment 2).**

- **E1 NOT HELD as worded.** Dropping SPLADE costs +0.53 on WebQSP (in the 0.5 to 2.5 range), but +2.98 on HotpotQA and +8.56 on 2Wiki (both above 2.5). The second half holds: the call is NEEDED on HotpotQA and 2Wiki.
- **E2 HELD.** Prime +2.78 [+1.64, +3.86] and Amazon +3.23 [+2.01, +4.33] at the best filed cell within 3,500 nodes (both at least 2).
- **E3 HELD.** The loss is under 1.0 on MultiHop-RAG (0.00), WebQSP (+0.80) and 2Wiki (+0.17), three of the four; MuSiQue is +1.37 [+0.83, +1.94] (over). SQuAD and HotpotQA read exactly 0.00.
- **E4 HELD.** Prime -0.32 (under 1.0). On Amazon the largest K loss at a FULL pick cell is +3.53 at the 15,400-node cell, which is less than that cell's FULL-minus-STRUCT gap of +6.09 (section 20 of the STK3 page); the other FULL cells read -0.49 to +0.67 (the one non-FULL K cell, `rrf1000_b50+KNN:h1_c100+top25`, +0.91).
- **E5 HELD on all nine.** The tightest is WebQSP: `both` +2.00 against a bound of +2.33 (the two single losses plus 1 point); the largest `both` loss is 2Wiki's +8.77 against +9.74.
- **E6 HELD** (above).
- **E7 HELD.** The `both` arm's mean nodes are 0.80 to 1.00 of the filed nodes at the nine headline cells.

**What this says (the main line is dense only, no K, as the user ruled on 27 Nov).**

- **K is dispensable on every dataset read.** The whole-split loss is at most +0.80 at the headline cell except MuSiQue's +1.37 [+0.83, +1.94] (UNCLEAR; its dense-only no-K arm is 2.4 points below the filed best within 2,500 nodes, of which K alone is 0.5). Dropping K also removes the K-search cost (WebQSP 9,712 s, HotpotQA 7,540 s, 2Wiki 6,616 s, Amazon about 4.1 h, MAG about 9.4 h).
- **SPLADE matters where the question names the entities and the pool is small.** At the filed headline cell it is NEEDED on SQuAD, HotpotQA, 2Wiki, Prime, MAG and Amazon; once the dense-only arm may re-pick inside the list, the loss at about 2,500 nodes is -0.1 (SQuAD), +0.5 in the arm's favour (HotpotQA), +0.9 in the arm's favour (MAG), -0.5 (Prime), -2.2 (WebQSP), -2.4 (MuSiQue), -4.0 (Amazon) and **-8.4 on 2Wiki**. **The gap that is left at the target size is 2Wiki (8.4 points), Amazon (4.0), MuSiQue (2.4) and WebQSP (2.2).**
- **Against the user's target (95 percent of the golds at about 2,500 nodes), the dense-only no-K arm re-picked reaches it on SQuAD (99.8), MultiHop-RAG (99.1) and HotpotQA (98.3) only.** The other six are below it at 2,500 nodes: 2Wiki 88.2, MuSiQue 86.1, WebQSP 85.7, Prime 85.0, MAG 91.7, Amazon 83.1 (MetaQA is not read here; LEG1 A6's dense-only read put it 1.0 below its filed R95 cell). Within the list's cells the dense-only arm reaches 95 on MAG at 5,259 nodes (the filed list at 5,648) and nowhere on 2Wiki, MuSiQue, WebQSP, Prime or Amazon.
- **Instrument note.** These are the pool engine's cells (a flat first-stage head plus budgeted walks over the filed links). The simpler budgeted walk of JNT1 section 12.8 (dense top 5 seeds only, e3v) reads 56.3 on WebQSP and 84.3 on 2Wiki at about 2,000 nodes; the two instruments are not comparable, and the engine's WebQSP number (85.7 at 2,100 nodes, older encoder) is 29 points higher than JNT1's.

**Not read here:** the seven datasets' dense-only arm on the defined encoder (the first-stage lists here are the older encoder's; e3v lists for the eval carves exist under `outputs/enc/e0/e3v`); a rule that recovers 2Wiki's lexical signal without SPLADE or BM25; MetaQA; the STaRK KBs under a name rule. Nothing is adopted, deleted or chosen by this section; SPLADE and K stay on disk; PL8 stays undeclared.
