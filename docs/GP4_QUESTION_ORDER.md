# GP4 — question-aware orders on GP3's walk: does the question put the golds into a smaller pool?

Declared 26 Nov 2026 20:45, **before any GP4 number exists and before any GP3 coverage number is filed** (GP3's quarter is not solved; the host rows are running). Nothing of GP3 changes: `zgp3.py` is frozen (sha256 prefix `12895fff62e3835a`), its rows are queued, and GP3's score list stays K1, K2, K3, K4, PP, R0, R1. This page adds scores **in a new file** (`outputs/mp_unified/zgp4.py`, to be written after this page and committed before any row, importing `zgp3.py` read-only) and reads them **beside** GP3's. Nothing runs until GP3's quarter is filed.

The user's words (26 Nov, relayed): *"scoring would work so improve it, metaqa needs proper work"*; and (21-25 Nov, still binding) *"we need to have universal rules"*, *"the edges should also have labels based on how it was created"*, *"iterate on a dev slice, then the full run once per declared rule"*, *"do not block or slow down the gpu run of the other process"*.

## 1. Where this comes from

CR1 (`docs/CR1_CHAIN_RECREATION.md`, filed ee7ced71, section 10) read the order of a source's neighbours on the **oracle's direct gold links over the whole graph, no pool**: the question cosine `fq` of the neighbour puts the next hop ahead of the pair's own dot `pd` at rank 30 by +3.9 (MultiHop-RAG), +8.3 (MuSiQue CH), +6.9 (HotpotQA), +0.8 (2Wiki), +3.0 LEVEL (WebQSP; BELOW at rank 10 by 7.9, ABOVE at rank 100 by 11.0), and **-25.9 on MetaQA** (its golds are answer names). Call NOT_UNIVERSAL. The z75 fusion (0.75 z-score of `fq` + 0.25 z-score of `pd`, z over the source's neighbours) is within 0.4 of the best single order on three datasets, above it on 2Wiki (+0.66) and WebQSP (+2.76), 15.5 below on MetaQA. The residual order is below `fq` on 6 of 6 and is **not carried**. CR1 §10.6 says a question-heavy order needs its own page, declared before any number, and a test on data it has not seen: this is that page.

CR1 is a read of links **to the golds**. GP4 asks the pool question: on GP3's walk (same graph, levels, S whole, ladder, triples, distinct-node count), do the golds come into the pool at fewer distinct nodes when the candidates are ordered by the question?

## 2. The scores (named Q1-Q3 so that they cannot be confused with GP2's K5 and K6, which GP3 named and did not run)

For a candidate c at hop h, over its incoming links from the kept parents (GP3 section 5's definitions of `u`, `rk`, `vt`, `dt`), with `qv` the question's vector and `e(x)` the node rows the pair dot is read from (the e3v rows, the same vectors CR1's `fq` reads):

| score | order | what it is |
|---|---|---|
| Q1 | (fq(c)) | **the question cosine of the tail**: `qv . e(c)`, larger first (a function of the tail only, so no aggregation over links); ties by node id |
| Q2 | (z75(c)) | **CR1's z75**: for a link (a, c) of a kept parent a, `0.75 z_a(fq(c)) + 0.25 z_a(pd(a, c))`, each z-score taken over the level-h neighbours of a (the rows of a in the hop table; a zero spread gives z = 0); the node's key is the **largest over its links from kept parents** (as `dt` is for K4); float32 as GP3's `sortable_f32` |
| Q3 | (fr(a, c)) | **KB only (MetaQA, WebQSP), conditional:** the cosine of the question to the **relation name of the S edge** a to c, largest over the node's S links from kept parents; a candidate with no S link ranks after every one that has (then by Q1). Needs two things that do not exist yet in a form I have checked: the relation id of every S link, and the relation names' vectors in the encoder of the node rows. **Q3 is declared only if both are present when the first GP4 row is queued**; otherwise it is "not read", filed so |

The residual order is not run (CR1 X7). PP, K1, K2, K3, K4, R0, R1 are GP3's, read from GP3's stored chunks (item 4), not recomputed.

**What is requested from the CR1 session for Q3 (the parent asked for the format):** a file `outputs/gp4/rel_<d>.npz` per KB with `names` (the S relation names as strings, one per relation id, the labels the S edges carry as `S.<rel>`), `vec` (float32, shape R by the node-row width, L2-normalised, the relation names (underscores read as spaces) embedded by the encoder that made the node rows), and a statement of where each S link's relation id is read (a store and a key), so that the walker can attach it to a link without a new graph. It may not exist for WebQSP's Freebase labels in a usable form; then Q3 is read on MetaQA alone and the call says so.

## 3. The walker (new file, written after this page, gated before any row)

`zgp4.py` imports `zgp3.py` and changes **nothing** in it: it adds the node keys of Q1-Q3 to the same layers (the same `layer`, nested prefixes, S whole, distinct count, caps) and writes chunks in `zgp3.py`'s format. Gates, fixed now: **G4-1** self-test: the walker equals a brute-force walk (as `zgp3.py`'s) for Q1 and Q2 (and Q3 if declared) on toy graphs with random vectors; **G4-2** zgp4's K1, K2, K4 and PP chunks equal GP3's stored chunks bit for bit on the first 3 questions of every dataset (a new file may not move an old number), before any GP4 row; **G4-3** the question vector of every dataset is the one CR1's G-C5 gate checked aligned (the five first-ranked stored dense results are above random nodes' in at least 0.9 of the comparisons: CR1 read 0.9838 on MetaQA and 0.9998 on WebQSP). A failed gate stops the stage and is filed.

## 4. Population, reads, calls

**Subset first** (the user's rule): the questions of **chunk 1 and chunk 4 of GP3's quarter** (40 questions) in every light cell and WebQSP, **chunk 1** (20) in HotpotQA and 2Wiki, **all 56** in MultiHop-RAG. These chunks are in GP3's stored quarter for every dataset (the heavy cells' declared chunks are 1, 4, 7, 10, 13, 16), so every question is read **twice, with the same seeds, graph and caps, under old and new scores**: a paired comparison. The rest of dev and the full quarter are read **once**, only if the call below is made and only under the rule it fixes. No s1eval, no model.

**The read (the sample is too small for ALL at 0.95, a 20-question share moves in steps of 0.05; so the read is the gold's place, which is continuous):** at the three triples (100, 100, 100), (300, 300, 300) and (1000, 1000, 1000) of GP3's ladder, per cell and per score, **the share of non-free golds (levels 1 to 3) that are in the pool (hop recall), the mean distinct nodes of the pool, and ALL**, for Q1, Q2 (Q3) and for GP3's K1, K2, K3, K4, PP. The score's own gain is the **share of golds in the pool at the triple, minus the same for the best of K1-K4 and PP in that cell at that triple** (the best old score is chosen per cell and triple **after** the read, so the comparison favours the old scores), with a paired bootstrap interval over questions (2,000 resamples, seed 0).

**Calls (fixed now).** Per cell, at (300, 300, 300): ABOVE if the gain of the better of Q1 and Q2 is at least 3 points and its interval is above 0, BELOW if the gain is at most -3 and its interval is below 0, else LEVEL; mean nodes of the pool reported beside (a gain bought with a larger pool is flagged, not counted: the call is made over triples whose mean nodes are within 10 percent of the best old score's; otherwise "not comparable"). **Q-UNIVERSAL: ABOVE on at least 7 of the 9 cells and BELOW on none** (the user's universal rule; MetaQA is one cell and counts as one). Otherwise Q-NOT_UNIVERSAL, and the per-class reading is filed (passage datasets alone; KBs alone) with its cells named, as a finding and **not as a rule**. Q3 has its own line (KB cells only, ABOVE/LEVEL/BELOW against the best of Q1, Q2 and the old scores, at (300, 300, 300) and (1000, 1000, 1000)). A rule that fuses Q-scores with the old ones, or picks a weight, is **not declared here** (CR1 §10.6: no label-free weight was found; one would be a new page and a test on unseen data).

## 5. What I expect (written before any number)

- **E1.** Q2 is ABOVE on MultiHop-RAG, MuSiQue and HotpotQA RK at (300, 300, 300): 60 percent (CR1's +3.9, +8.3, +6.9 at rank 30 on direct links; a pool keeps S whole and the dot already orders the first stage, so the gain is smaller than on links).
- **E2.** MetaQA: Q1 and Q2 are BELOW the best old score at (300, 300, 300): 75 percent (CR1: -25.9 at rank 30 on answer-name slots; MetaQA's golds are answers).
- **E3.** WebQSP is LEVEL or BELOW for Q1 at (100, 100, 100) and ABOVE at (1000, 1000, 1000) for Q2: 40 percent (CR1's rank-10 and rank-100 signs).
- **E4.** Q-UNIVERSAL is **not** reached: 80 percent (MetaQA alone breaks it if E2 holds; 2Wiki's +0.8 is below the 3-point bar).
- **E5.** Q3 (if read) is ABOVE the best of Q1, Q2 and the old scores on MetaQA at (1000, 1000, 1000): 40 percent (a question's relation words order the topic entity's neighbours; the pool at hop 2 and 3 holds answers reached through many relations).
- **E6.** G4-2 passes (a new file reproduces the old chunks): 90 percent.

## 6. Cost and operations

Per question the walk costs GP3's seconds plus the question cosine of the distinct candidates (up to a few million tails per hop on the heavy cells; a gather of int8 or float rows and one matrix-vector product, which I expect to add under 10 percent) and, for Q2, a per-source z-score over the hop table. Rows: one per dataset (two chunk files for light cells), one CPU, the memory of GP3's rows, **queued only after GP3's quarter rows have ended so that they do not take the host's memory from them and from the other session's rows** (the user: no throttling, no blocking of the GPU session's work). The first row is a gate row (G4-2 on 3 questions per dataset).

## 7. What this page does not do

It does not change GP3, its final rule, the link set, or any gate or call of GP3. It is not a trained scorer, not a model read (the SWP lift is a model read) and not a new link rule. It reads orders at fixed triples of a fixed walk; whether a question-aware order is **carried** is decided only by the call in section 4 and only for the walk, never for the models.

## 8. Amendments

None yet.

## 9. Results

None yet.
