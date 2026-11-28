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

## 14. Amendment 3 (10:02 IST, 28 Nov 2026): the defined-encoder run and a name-seed arm for the SPLADE gap (declared before any number)

**Why.** Section 13 reads the dense-only, no-K arm (the user's main line, 27 Nov) at 2,500 nodes: it reaches 95 on SQuAD, MultiHop-RAG and HotpotQA only, and is furthest from the filed pools on 2Wiki (-8.4), Amazon (-4.0), MuSiQue (-2.4) and WebQSP (-2.2). Two things follow. (1) The first six datasets' reads are older-encoder reads; the user's 28 Nov ruling is that only the defined encoder (e3v) is used. (2) SPLADE's gap on 2Wiki is a lexical one (the questions name two or more article titles); JNT1 section 12.7 found that the question-name rule's seed gain tracks its precision (2Wiki 91.3 percent, ABOVE the dense-only seeds at 5 of 5 widths of the simple walk; HotpotQA 23.6 percent, BELOW at 3 of 5), and that SPLADE's own seeds are not above dense-only seeds on any of the six. BM25 stays out (the user's 27 Nov ruling).

**What.** One new module, `outputs/mp_unified/zdrp2.py` (declared here; it imports `zdrp1.py`, `zleg1p.py`, `zpl2g.py`, `zpl5g.py`, `zpl6g.py`, `zjnt2.py` (only `qn_extra` and `node_titles`) and `zenc.py`, and edits none of them; its sha256 prefix is `5e05e7a6f9d88a2c`, committed with this amendment; a later edit goes through a further amendment, a host sha check and new row names). The source is `load_source(ds, "eval", "e3v")`: the first-stage lists are ENC E0's e3v lists (`outputs/enc/e0/e3v/<ds>.npz`, depth 1,000) and the vectors are e3v's. **The cells, the eval questions and the seeded chunk order are DRP1's own** (`outputs/drp1/pool/<ds>/cells_<ds>.json`; the gate compares the question-id hash), so every number pairs with section 13's on the same questions and cells. Datasets: the six ENC E0 covers (SQuAD, MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki; host rows) and MultiHop-RAG, whose e3v first stage is PL1 S0a's record (`zpl1h.read_first`; laptop rows, as DRP1's own MultiHop-RAG rows were). The STaRK KBs are e3v already and do not run here (their name rule needs a loader of their own and is a later declaration).

**Two arms, both with the K store empty and no SPLADE list.**

- `both` (e3v): DRP1's `both` arm on the e3v source (the SPLADE slot is the dense list; the seeds are the dense top 5).
- `names` (e3v): the SPLADE slot is a list built per question: the question's name seeds (`zjnt2.qn_extra`, constants unchanged: a node whose whole title is a span of the question, longest span first, a title no other node has, at most 5 seeds, no group variant, no label, no model) placed first (those the dense list ranks in the dense list's order, the others by node id), followed by the dense list in its own order without those nodes, cut to the dense list's width. A question with no name seed gets the dense list itself (so its pool equals `both`'s). The engine's seeds are the union of the dense top 5 and the SPLADE slot's top 5, so a question's name seeds (at most 5) are always seeds, and its flat head `rrf1000_bN` sees them promoted by the RRF of the two lists. The pool code is unchanged: the seeds (top 5 of the fused list) and `rrf1000_bN` then see the name nodes promoted. The rule's frequency line (a span needs one token that at most 2 percent of the questions contain) is read over the eval carve's questions; the constants were set on a 300-question dev slice, so the line sits at a different count here (the page that ran them is JNT1 section 12.7; no constant is retuned).

**Gates (a failed gate stops the row; no arm number is read from a dataset with a failed gate).** **G-E0:** the source's first stage is e3v (for the six: E0's list, its file the one E0's record names by sha256), at least 1,000 deep or as deep as the KB, and the eval carve's question-id hash equals the cells file's. **G-D1:** `both` and `names` have an empty K store and the S and R stores are the filed objects; `both`'s SPLADE slot is the dense list; `names`' is neither the dense list nor the older SPLADE list. **G-E2:** on the first 16 questions every cell that reads no K, run on the intact e3v source with only its K store emptied, equals its value on the intact source (SPLADE and K present), which shows the wrapper reached the engine and changed only what it names (it does not apply where every cell reads K). **G-N1:** every `names` row has the dense list's width and dtype and no repeated id; its first entries are exactly the name seeds, the rest keep the dense order; a question with no name seed equals the dense row; every name seed is among the seeds.

**Reads.** Slice reads under amendment 2 (the first N chunks of the seeded order, named as slices, nothing adopted): N = 12 chunks (1,536 questions) for SQuAD, MetaQA, HotpotQA, WebQSP (all of its 12) and 2Wiki, 8 chunks (1,024 questions) for MuSiQue, and the whole split for MultiHop-RAG (1,130 questions). A whole-split run needs a call on the slice and its own line on this page. Per cell and arm: ALL with its Wilson interval and the loss against the filed arrays (as section 5); `names` minus `both` per cell with a paired bootstrap (2,000 resamples, seed 20261127) and the call ABOVE (interval lower end above +1.0), BELOW (upper end below -1.0), else LEVEL; `both` (e3v) against the older-encoder `both` arm of section 13 on the same questions (the encoder effect; not formed for MetaQA, whose `both` arm DRP1 did not run). **Re-pick read (post hoc, labelled):** each arm's best ALL among the cells within 500, 1,000, 2,500, 3,500 nodes. **The gap call:** gap = the filed best within 2,500 nodes minus `both`'s best within 2,500; recovered share = (`names`' best minus `both`'s best) over the gap; RECOVERED at 0.90 or more, PARTIAL from 0.25 to under 0.90, NOT_RECOVERED under 0.25, NO_GAP if the gap is under 1.0 point.

**Expectations (written before any number; priors in brackets).**

- **N1.** On 2Wiki `names` recovers at least 4.0 points of the 8.4: PARTIAL or RECOVERED [65 percent].
- **N2.** On HotpotQA `names` is LEVEL with `both` at the best within 2,500 nodes [60 percent].
- **N3.** On SQuAD, MetaQA, WebQSP and MultiHop-RAG `names` is LEVEL with `both` at the best within 2,500 nodes (SQuAD has no name seeds, MetaQA's name seed is the topic entity, WebQSP's name precision is 1.2 percent; MultiHop-RAG is untested: its prior is lower) [70 percent each, 55 for MultiHop-RAG].
- **N4.** On MuSiQue `names` is ABOVE `both` at one or more cells within 2,500 nodes [55 percent].
- **N5.** e3v's `both` is not below the older encoder's `both` by more than 1.0 point at the best within 2,500 nodes on any dataset [70 percent].
- **N6.** No dataset has `names` BELOW `both` at its best within 2,500 nodes: the rule is carried everywhere or it is not carried (the user's 21 Nov ruling) [55 percent].
- **N7.** The four gates pass on every row.

**What it decides.** Nothing is adopted. If N6 holds and N1 holds the name rule is the candidate lexical leg for the main line and its universality on the other datasets (the STaRK KBs, MultiHop-RAG) is the next declaration; if N6 fails the rule is not universal in the pool engine either and a precision-aware variant needs its own amendment with this failure on record. Dense only, no K stays the main line; SPLADE and K stay on disk until the user's per-action OK; PL8 stays undeclared.

**Cost (measured: the mean seconds of DRP1's own `both` chunks of 128 questions on the older encoder; e3v's 1,152-wide vectors may cost more, the first chunk measures it).** 2Wiki 117 s, HotpotQA 162 s, WebQSP 331 s, MuSiQue 711 s, SQuAD 6 s, MultiHop-RAG 1.4 s (MetaQA has no DRP1 chunk). Two arms over the slices: 2Wiki 12 chunks about 47 minutes, HotpotQA about 65, WebQSP about 132, MuSiQue 8 chunks about 190, SQuAD under 3, MultiHop-RAG under 1; each split into four parts of one row each, so the longest row is MuSiQue's at about 50 minutes. The name-seed build is one pass over a dataset's titles (one row each, ahead of its parts).

## 15. Slice results of five datasets on e3v (10:40 IST, 28 Nov 2026): N1 to N7 judged where the data is in; whole-split runs declared

**What ran.** Section 14's rows `d2n`, `d2r`, `d2q` (host sha of `zdrp2.py` 5e05e7a6f9d88a2c = the laptop's). All gates read PASS on every part read here (G-E0 and G-N1 in `gate_names.json`; G-D1 and G-E2 in `gate_run_p*.json`, 0 mismatches over the K-free cells). The reads are slices of DRP1's seeded chunk order (1,536 questions each: SQuAD, MetaQA, HotpotQA, 2Wiki) and the whole split of MultiHop-RAG (1,130, laptop); SQuAD's read ran on the laptop (the host has no filed SQuAD grid; `d2q-sq` stopped with "no filed eval grid here" and is not re-run). **MuSiQue and WebQSP are still running; their calls follow in a later section.** Records: `outputs/drp1e/read_<ds>.json`, the chunks and gate files under `outputs/drp1e/pool/<ds>/`. Name seeds found (`gate_names.json`): 2Wiki 12,509 of 12,576 questions (mean 1.58 per question), HotpotQA 7,363 of 7,405 (3.40), MetaQA 8,851 of 9,785 (0.98), MuSiQue 1,925 of 2,417 (1.22), WebQSP 1,397 of 1,503 (1.37), SQuAD 0 of 11,873 (85 titles are shared by several paragraphs), MultiHop-RAG 1 of 1,130.

**The best cell within 2,500 nodes, per dataset (post hoc, each arm's own best; ALL in percent, nodes in brackets; `names - both` with its paired interval; the older-encoder `both` arm is DRP1 section 13's, on the same questions).**

| dataset (questions) | filed | `both` (e3v) | `names` (e3v) | `names - both` | older `both` | e3v - older |
| --- | --- | --- | --- | --- | --- | --- |
| 2Wiki (1,536 of 12,576) | 96.3 [2,212] | **96.2 [2,230]** | 96.6 [2,230] | +0.46 [+0.20, +0.85] LEVEL | 88.7 [2,201] | **+7.42 [+6.05, +8.79]** |
| HotpotQA (1,536 of 7,405) | 98.1 [155] | 98.8 [1,891] | 99.1 [2,485] | +0.26 [+0.07, +0.52] LEVEL | 98.6 [1,893] | +0.26 [-0.26, +0.78] |
| MetaQA (1,536 of 9,785) | 91.3 [2,335] | 94.4 [2,457] | 94.4 [2,470] | +0.00 [+0.00, +0.00] LEVEL | 89.6 [2,483] | +4.75 [+3.71, +5.86] |
| SQuAD (1,536 of 11,873) | 99.9 [2,243] | 99.5 [1,637] | 99.5 [1,637] | +0.00 [+0.00, +0.00] LEVEL | 99.7 [1,623] | -0.26 [-0.65, +0.13] |
| MultiHop-RAG (1,130, whole) | 99.1 [100] | 98.8 [100] | 98.8 [100] | +0.00 [+0.00, +0.00] LEVEL | 99.1 [100] | -0.35 [-1.06, +0.35] |

Other reads. **2Wiki at 500 nodes or fewer:** `names` 92.3 at 82 nodes against `both` 89.1 at 80, +3.26 [+2.34, +4.23] ABOVE (the only cell of the five datasets called ABOVE; the other five of 2Wiki's six cells are LEVEL); the first cell at 95 is the same cell for both arms (96.0 and 96.4 at 830 nodes, the filed list 96.0 at 918). **MetaQA:** the first cell at 95 within 3,500 nodes is `dense_top200+STRUCT:h3_c25_v4000` (95.2 at 2,522 nodes, both arms). **SQuAD at 500 nodes or fewer:** e3v `both` 95.4 at 50 nodes against the older arm's 97.7 at 50, -2.28 [-3.12, -1.43] (BELOW; the only place e3v is below the older encoder by more than a point). **Cell calls `names - both` over all cells:** 2Wiki ABOVE 1, LEVEL 5; HotpotQA LEVEL 5 (range -0.39 to +0.46); MetaQA, SQuAD and MultiHop-RAG every cell exactly 0.00 (the pools are identical cell by cell; for MetaQA the likely reason is that the name seed is the topic entity the dense top 5 already holds, which was not measured; SQuAD has no name seeds).

**Calls on section 14's expectations (slices; MuSiQue and WebQSP not yet in).**

- **N1 NOT HELD as worded.** The call was written for a gap of 8.4 points. On e3v there is no such gap: `both` is 96.2 against the filed 96.3 at about 2,200 nodes (the declared gap call is NO_GAP at every limit from 1,000 nodes up). The gap of section 13 came from the older encoder (+7.42 [+6.05, +8.79] from the older `both` to the e3v one), not from the missing SPLADE list. The name rule adds +0.46 [+0.20, +0.85] at 2,230 nodes (LEVEL by the declared 1.0-point line).
- **N2 HELD.** HotpotQA: +0.26 [+0.07, +0.52] LEVEL at the best within 2,500 nodes.
- **N3 HELD on SQuAD, MetaQA and MultiHop-RAG** (all LEVEL, exactly 0.00); WebQSP is not in.
- **N4 and N5:** MuSiQue and WebQSP not in. N5 on the five datasets read: the e3v `both` best within 2,500 nodes is above the older one on 2Wiki (+7.42), MetaQA (+4.75) and HotpotQA (+0.26) and below it by 0.26 and 0.35 on SQuAD and MultiHop-RAG, so no dataset read is below by more than 1.0 point.
- **N6 so far HELD:** no dataset has `names` below `both` at its best within 2,500 nodes, and no cell of the five is called BELOW.
- **N7 so far HELD:** the gates pass on every part read.

**What this says (slices, not whole-split reads).** On the defined encoder the dense-only, no-K pool reaches 95 at about 2,500 nodes on 2Wiki (96.2), HotpotQA (98.8), MetaQA (94.4 at 2,457 nodes, 95.2 at 2,522), SQuAD (99.5) and MultiHop-RAG (98.8), each within 0.4 point of the filed pool (MetaQA 3.1 above it). The 2Wiki gap that section 13 read as SPLADE's is the older encoder's. The question-name rule is not harmful on any dataset read and is positive by under one point at about 2,500 nodes (2Wiki +0.46, HotpotQA +0.26), with a clear gain only at the smallest 2Wiki cell (+3.26 at about 80 nodes).

**Whole-split runs declared (the call on the slice that section 11 requires).** The slices are whole-split-sized in cost for MetaQA and SQuAD (1 and 6 seconds a chunk) and one to two hours on four to six parts for 2Wiki and HotpotQA, and the question of the lexical leg is decided on whole splits, so the same module, cells, arms, gates and seeded order run over every remaining chunk of 2Wiki, HotpotQA, MetaQA and SQuAD (the resumable `run` without `--slice`; chunks already made are kept). Rows `d2w-*` (parts) and `d2z-*` (the read; SQuAD's read runs on the laptop). WebQSP's slice is its whole split (12 of 12 chunks). MuSiQue follows when its slice is in. Nothing is adopted; SPLADE, K and PL8 stay as they are.

## 16. MuSiQue and WebQSP on e3v (11:31 IST, 28 Nov 2026): N3 to N7 judged on all seven datasets; MuSiQue's whole-split rows declared by section 15 queued

**What ran.** Rows `d2r-mu0..3`, `d2r-wq0..3`, `d2q-mu`, `d2q-wq` of section 14 (same module sha 5e05e7a6f9d88a2c). All gates read PASS on every part (`gate_names.json`, `gate_run_p*.json`; 0 mismatches over the K-free cells). WebQSP is the whole split (1,503 questions, 12 of 12 chunks); MuSiQue is a slice (1,024 of 2,417 questions, 8 of 19 chunks). Records: `outputs/drp1e/read_musique.json`, `read_webqsp.json`.

**The best cell within 2,500 nodes (post hoc, each arm's own best; ALL in percent, nodes in brackets).**

| dataset | filed | `both` (e3v) | `names` (e3v) | `names - both` | older `both` | e3v - older | gap call (filed - `both`) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue (1,024 of 2,417) | 86.0 [2,436] | 83.4 [2,485] | 84.4 [2,472] | +0.98 [+0.10, +1.86] LEVEL | 82.2 [2,484] | +1.17 [-0.59, +2.93] | 2.6 points, PARTIAL (share 0.37) |
| WebQSP (1,503, whole) | 87.9 [2,095] | 86.6 [2,099] | 86.0 [2,096] | -0.60 [-1.40, +0.20] LEVEL | 85.7 [2,100] | +0.93 [-0.27, +2.20] | 1.3 points, NOT_RECOVERED (share -0.47) |

Within 3,500 nodes: MuSiQue filed 87.0 [2,981], `both` and `names` unchanged (83.4 and 84.4), gap 3.6 PARTIAL (share 0.27); WebQSP filed 88.0 [2,520], `both` 87.3 [2,747], `names` 87.7 [2,762], `names - both` +0.40 [-0.13, +0.93] LEVEL, gap 0.7 NO_GAP.

**Cell calls `names - both` (all cells of each cells file).** MuSiQue: 7 LEVEL, 1 ABOVE; the ABOVE cell is `equal_rrf_budget_200+FULL:h2_c25` (`both` 70.6 at 1,804 nodes, `names` 73.2 at 1,912 nodes, +2.64 [+1.66, +3.71]; `names` uses 6 percent more nodes there, and the cell is 11 points under the best within 2,500). Range of the gain over the eight cells +0.29 to +2.64. WebQSP: 9 LEVEL, range -0.60 to +1.06; the two negative cells are the two STRUCT cells with visit cap 2,000 (-0.60 and -0.53).

**The target.** Neither arm has a cell at 95 within 2,500 nodes on either dataset, and nor has the filed pool. The cells that reach it are the wide walks: MuSiQue's filed 95.0 sits at 15,400 nodes (`both` 92.5, `names` 93.2 there), WebQSP's filed 94.7 at 39,530 nodes (`both` 93.0, `names` 93.7 at about 39,000). At the 2,500-node bound the best of the main line is 84.4 (MuSiQue) and 86.6 (WebQSP), 10.6 and 8.4 points under 95; the filed pool is 86.0 and 87.9. So on these two datasets the miss is the walk's coverage per node, not SPLADE, K or the encoder (e3v is +1.17 and +0.93 over the older arm, both intervals include 0).

**Calls on section 14's expectations (all seven datasets in; N1 and N2 as in section 15).**

- **N3 HELD on all four datasets named.** SQuAD, MetaQA and MultiHop-RAG exactly 0.00; WebQSP -0.60 [-1.40, +0.20] LEVEL at the best within 2,500 nodes.
- **N4 HELD, narrowly and weakly.** MuSiQue has one ABOVE cell within 2,500 nodes (+2.64 [+1.66, +3.71] at 1,912 nodes). At the best within 2,500 nodes the gain is +0.98 [+0.10, +1.86], LEVEL by the declared line (the lower end is above 0, the upper end above 1.0). The ABOVE cell is a small cell that does not set the best, and the slice is 1,024 of 2,417 questions; the whole-split read decides whether the gain at the best cell clears 1.0.
- **N5 HELD.** The e3v `both` best within 2,500 nodes is not below the older one by more than 1.0 on any of the seven: +7.42 (2Wiki), +4.75 (MetaQA), +1.17 (MuSiQue), +0.93 (WebQSP), +0.26 (HotpotQA), -0.26 (SQuAD), -0.35 (MultiHop-RAG). The wording is the best within 2,500 nodes; SQuAD at 50 nodes (-2.28, BELOW, section 15) is outside it and stays reported there.
- **N6 HELD.** No dataset has `names` BELOW `both` at its best within 2,500 nodes (lowest: WebQSP -0.60 [-1.40, +0.20]), and no cell of the seven is called BELOW (lowest cell gain -0.60).
- **N7 HELD with one stated exception.** The gates pass on every part of every read. The host row `d2q-sq` failed twice (the feeder's retry too) on "no filed eval grid here", which is a missing input and not a gate; SQuAD's read was run on the laptop and its gates pass.
- **N1, re-read against every value.** The text reads "recovers at least 4.0 points of the 8.4". There is no 8.4 on e3v (2Wiki `both` 96.2 against the filed 96.3 at 2,230 nodes), so the call stays NOT HELD as worded. At 500 nodes or fewer the declared gap call reads RECOVERED (gap 1.4, share 2.27), which is the one limit where the name rule closes more than the whole gap.

**What the seven reads say together (slices except WebQSP and MultiHop-RAG; nothing adopted).** Dense-only, no K, on e3v reaches 95 or more within about 2,500 nodes on 2Wiki (96.2), HotpotQA (98.8), SQuAD (99.5), MultiHop-RAG (98.8) and MetaQA (95.2 at 2,522 nodes; 94.4 within 2,500), and does not on MuSiQue (83.4) and WebQSP (86.6). The question-name rule is never BELOW `both` and is called ABOVE only in small cells (2Wiki at about 80 nodes, MuSiQue at about 1,900 nodes). The name rule is therefore not a replacement for SPLADE's lexical leg at 2,500 nodes on MuSiQue (gap 2.6, share 0.37) or WebQSP (share -0.47). The remaining distance on those two is walk coverage per node (and, on the STaRK sets, the same under their own cells), not a missing first-stage list.

**Rows.** `d2w-mu0..3` (parts of MuSiQue's whole split over the remaining 11 chunks; the eight chunks of the slice are kept) and `d2z-mu` (the read), declared by section 15's last paragraph, pushed to the feed before the first unsent held row (feed sha aba35e663e72, 12,460 items). Whole-split rows of 2Wiki (`d2w-tw0..5`), HotpotQA (`d2w-hp0..3`), MetaQA (`d2w-me0..1`) and SQuAD (`d2w-sq0..1`) are running or waiting as declared in section 15. SPLADE, K and PL8 stay as they are.

## 17. Whole-split reads of all seven datasets on e3v (12:43 IST, 28 Nov 2026): the target holds on four, not on MetaQA, MuSiQue and WebQSP

**What ran.** The whole-split rows declared in sections 15 and 16 (`d2w-tw0..5`, `d2w-hp0..3`, `d2w-me0..1`, `d2w-sq0..1`, `d2w-mu0..3`, same module sha 5e05e7a6f9d88a2c) and their reads. 2Wiki (12,576 questions), HotpotQA (7,405) and MuSiQue (2,417) were read on the host (`d2z-tw`, `d2z-hp`, `d2z-mu`); MetaQA (9,785) by the same command (`zdrp2.py read --ds metaqa`); SQuAD (11,873) on the laptop (the host has no filed eval grid for it, as in section 16). WebQSP (1,503) and MultiHop-RAG (1,130) were whole splits already. All 32 gate files (`gate_names.json`, `gate_run_p*.json`) read PASS and every read says COMPLETE. Records: `outputs/drp1e/read_<ds>.json` (the slice records of sections 15 and 16 stay in git history, commits 9c473c1c and e7d2c2e8).

**A trap found on the way.** `d2z-me` wrote the whole-split MetaQA record at 11:44 and the host file was then overwritten by a push that carried the laptop's tracked slice record (the files are tracked, so `rx push` replaces a newer host copy; a push after 11:44 did it). The row's log showed COMPLETE and the file said SLICE. The read command was run again on the host after the push and the record is the whole one. From this commit the laptop's copies are the whole-split records, so a later push no longer reverts them.

**The best cell within 2,500 nodes, whole splits (each arm's own best; ALL in percent, nodes in brackets; post hoc as in sections 15 and 16).**

| dataset | questions | filed | `both` (e3v) | `names` (e3v) | `names - both` | older `both` | e3v - older | gap call (filed - `both`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2Wiki | 12576 (COMPLETE) | 96.6 [2,211] | 96.7 [2,228] | 97.0 [2,228] | +0.30 [+0.21, +0.41] LEVEL | 88.2 | +8.55 [+8.04, +9.06] | -0.2 points NO_GAP |
| HotpotQA | 7405 (COMPLETE) | 97.8 [154] | 98.5 [1,885] | 98.8 [2,483] | +0.26 [+0.14, +0.36] LEVEL | 98.3 | +0.27 [+0.00, +0.54] | -0.7 points NO_GAP |
| MetaQA | 9785 (COMPLETE) | 89.7 [2,336] | 93.3 [2,485] | 93.3 [2,496] | +0.00 [+0.00, +0.00] LEVEL | 88.8 | +4.49 [+4.05, +4.95] | -3.6 points NO_GAP |
| SQuAD | 11873 (COMPLETE) | 99.9 [2,200] | 99.6 [1,574] | 99.6 [1,574] | +0.00 [+0.00, +0.00] LEVEL | 99.8 | -0.24 [-0.35, -0.13] | 0.3 points NO_GAP |
| MultiHop-RAG | 1130 (COMPLETE) | 99.1 [100] | 98.8 [100] | 98.8 [100] | +0.00 [+0.00, +0.00] LEVEL | 99.1 | -0.35 [-1.06, +0.35] | 0.4 points NO_GAP |
| MuSiQue | 2417 (COMPLETE) | 88.5 [2,438] | 86.2 [2,487] | 87.4 [2,476] | +1.16 [+0.66, +1.70] LEVEL | 86.1 | +0.17 [-0.91, +1.28] | 2.2 points PARTIAL (share 0.52) |
| WebQSP | 1503 (COMPLETE) | 87.9 [2,095] | 86.6 [2,099] | 86.0 [2,096] | -0.60 [-1.40, +0.20] LEVEL | 85.7 | +0.93 [-0.27, +2.20] | 1.3 points NOT_RECOVERED (share -0.47) |

Cells of each cells file (`names - both`): 2Wiki 5 LEVEL and 1 ABOVE (`equal_rrf_budget_50+STRUCT:h1_c25`, +2.78 at about 80 nodes), HotpotQA 5 LEVEL (-0.34 to +0.43), MetaQA 7 LEVEL (0.00), SQuAD 4 LEVEL (0.00), MultiHop-RAG 4 LEVEL (0.00), MuSiQue 7 LEVEL and 1 ABOVE (`equal_rrf_budget_200+FULL:h2_c25`, +2.15 [+1.61, +2.73], `names` at 1.049 times the nodes), WebQSP 9 LEVEL (-0.60 to +1.06).

**What changed against the slices.** The slice numbers move by up to 3 points where the slice was a small share of the split: MetaQA's `both` is 93.3 at 2,485 nodes (the slice had 94.4 at 2,457 and 95.2 at 2,522; on the whole split 94.1 is reached at 2,585 nodes, so the slice's "95 within about 2,500" does not carry), MuSiQue's `both` is 86.2 (slice 83.4) and its filed pool 88.5 (slice 86.0), so the gap is 2.2 points (slice 2.6).

**The target (all golds in the pool at 95 or more within about 2,500 nodes).** Met by the dense-only, no-K, no-name e3v pool on 2Wiki (96.7), HotpotQA (98.5), SQuAD (99.6 at 1,574 nodes) and MultiHop-RAG (98.8 at 100 nodes). Not met on MetaQA (93.3 within 2,500 nodes; the cell with 99.5 sits at 4,404 nodes, the filed pool's 95.8 at 4,973), MuSiQue (86.2; its best listed cell, 94.0, is at 15,400 nodes) and WebQSP (86.6; 93.0 at 39,142 nodes). On MetaQA the e3v pool is 3.6 points above the filed one, so the shortfall to 95 is the walk's coverage per node, as on the other two.

**Calls on section 14's expectations, whole splits.**

- **N1 NOT HELD as worded** (unchanged): there is no 8.4-point gap on e3v. 2Wiki `both` 96.7 against the filed 96.6.
- **N2 HELD.** HotpotQA `names - both` +0.26 [+0.14, +0.36], LEVEL.
- **N3 HELD on SQuAD, MetaQA, WebQSP and MultiHop-RAG** (0.00, 0.00, -0.60 [-1.40, +0.20], 0.00, all LEVEL).
- **N4 HELD, and this time at a cell.** MuSiQue has one ABOVE cell within 2,500 nodes (+2.15 [+1.61, +2.73], `names` using 4.9 percent more nodes there); at the best within 2,500 nodes the gain is +1.16 [+0.66, +1.70], LEVEL by the declared 1.0-point line (the interval's upper end is above 1.0).
- **N5 HELD.** e3v minus older at the best within 2,500 nodes: +8.55 (2Wiki), +4.49 (MetaQA), +0.93 (WebQSP), +0.27 (HotpotQA), +0.17 (MuSiQue), -0.24 (SQuAD), -0.35 (MultiHop-RAG); none below by more than 1.0. SQuAD at 50 nodes stays -1.64 [-1.97, -1.33] on the whole split (more than 1.0 below the older arm), outside the wording of N5, and is reported as in section 15.
- **N6 HELD.** No dataset has `names` BELOW `both` at its best within 2,500 nodes (lowest -0.60 on WebQSP) and no cell is called BELOW (lowest cell -0.60).
- **N7 HELD with the exceptions of section 16** (the host's `d2q-sq`/`d2z-sq` rows, a missing input, run on the laptop; and the overwritten MetaQA record above, re-read). The 32 gate files pass.

**What the seven whole-split reads say together.** On the defined encoder the dense-only, no-K pool needs no SPLADE and no K on four datasets, loses nothing against the filed pool within 2,500 nodes on five (2Wiki, HotpotQA, MetaQA, SQuAD, MultiHop-RAG; MetaQA is 3.6 above), and is 2.2 (MuSiQue) and 1.3 (WebQSP) points below it. The name rule is LEVEL everywhere within 2,500 nodes, ABOVE in two small cells, and stays out of the main line (user, 28 Nov): nothing from the `names` arm is carried. The remaining distance to 95 on MetaQA, MuSiQue and WebQSP is the walk's coverage per node; BRG1 reads where the missed golds are lost.

**Rows.** None new. SPLADE, K, PL8 stay as they are.
