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
