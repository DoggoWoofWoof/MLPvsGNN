# BRG1: why the budgeted walk does not bridge, and what Amazon's first stage lacks

Declared 28 Nov 2026 12:16 IST, before any number of this page was read. Read-only diagnostics: no pool, cell, rule or link is changed, nothing is adopted, and no model is trained.

## 1. Why

The user asked (28 Nov, after the name rule was ruled out of the main line): "why arent we able to bridge what are we actually missing and also do amazon specific diagnostics and wherever we are worse do check out".

State at the filing of DRP1 section 16 (dense-only, no K, e3v, best cell within 2,500 nodes; ALL in percent, filed pool in brackets): MuSiQue 83.4 (86.0), WebQSP 86.6 (87.9), STaRK-Prime 85.0, STaRK-MAG 91.7, STaRK-Amazon 83.1 (87.1). The target is 95. STK3 section 18 measured that on Amazon the first list alone (fused, 90.6 at 2,500) beats the best walk cell (87.1 at 2,448 nodes, -3.3 [-4.6, -2.1]) and left three questions unread: whether the names are in the seeds, how many of a gold's neighbours the walk visits before the budget ends, and where the gold sits in the walk's candidate order. The chunk files of every earlier read store counts only (`present`, `size` per cell), so none of that can be read from them. BRG1 re-runs the walk with every hop's candidate list kept.

## 2. What runs

Module `outputs/mp_unified/zbrg1.py` (sha256 prefix `fc4cdb5b5ba74a69`; selftest PASS: the traced walk equals `zpl5g.expand5` on 120 random walks of four walk settings (keys vr and fu, with and without rr), `in_edges` and `bfs_dist` equal brute force, all seven reasons occur and their identities hold on three toy graphs). It imports the PL2 to PL6 code, `zdrp1.py`, `zdrp2.py` and `zstk3.py` and edits none of them. Later edits go through an amendment on this page.

**`trace`** (one set at a time). The source is the dense-only, no-K arm on e3v (`zdrp1.variant(src, "both")`: the SPLADE slot is the dense list, the K store is empty; no name seeds). Pool = head of depth `b` of the cell's list, plus the walk's expansion, exactly as `zpl3g.grade3` forms it. Per question the walk (`zpl5g.expand5`, a budget-following walk) is run twice, once by the engine and once by the traced copy that keeps, per hop, the frontier, the budget and the ordered list of fresh candidates (before the budget cut). Per gold node of the question the module records:

| reason (first that applies, in this order) | meaning |
| --- | --- |
| `IN_HEAD` | in the first `b` entries of the cell's list |
| `IN_EXP` | in the walk's expansion, not in the head |
| `NO_IN_EDGE` | not in the pool; no node's row in the cell's families lists it (nothing links to it under the cell's regime) |
| `BUDGET_CUT` | a fresh candidate of some hop (inside the first 512 entries of a frontier node's row, not yet visited) that the hop's budget cut; the place among that hop's candidates and the budget are recorded (the hop with the smallest place over budget) |
| `WINDOW_CUT` | never a candidate; a frontier node's row lists it beyond the first 512 entries |
| `UPSTREAM_CUT` | never a candidate, not window-cut; an uncapped search from the five seeds over the cell's families reaches it within the cell's hop count, so a node on its path was cut at an earlier hop |
| `NO_PATH_IN_H` | not reached within the cell's hop count even without caps, window or budget |

Per missed gold also recorded: its place in the cell's list; the number of in-edges; whether another gold of its question links to it (`gany`), and the best state of those gold parents that are in the pool (1 = expanded, i.e. its row was read; 2 = in the head only, never expanded; 3 = kept at the last hop, not expanded) and of any parent in the pool; the smallest list place among its parents (what seed depth would put a parent in the frontier). Per question: pool size, golds present, golds among the five seeds, the frontier, candidate count, budget and kept count of each hop, and (`flat`) the golds in the e3v dense list's prefix of the pool's size (the flat pool at equal nodes; the STaRK sets read the 2,500-deep list of `first.npz`, the seven only the 1,000-deep one, so the comparison is made on the STaRK sets only).

**`links`** (all ten sets, whole split, questions with two or more golds): per family set (S, R = ner, K, S+R, S+R+K) the share of golds with a linked partner gold (either direction, direct edge), the share of questions whose golds form one connected component, by gold count, and the share of those edges that sit beyond the first 512 entries of the row (invisible to the walk). The K store is read as stored; K is not in the main line, it is the comparison.

**`lists`** (STaRK sets): the dense (e3v), SPLADE and fused (RRF) first-stage lists at depth 2,500: gold recall and ALL at 5, 10, 50, 100, 200, 500, 1,000, 2,500, by gold count; the overlap of dense and SPLADE golds at depth 1,000 and 2,500 (both, dense only, SPLADE only, neither); the share of questions with a gold among the top five of each list (the walk's seeds are the dense top five); with `--text` (Amazon, Prime) the gold's node-text length and the share of the question's terms of four letters or more that occur in the node's text, for four groups of golds (dense top 100; dense 100 to 1,000; SPLADE-only at 1,000; neither). The term overlap is a count read for the diagnosis; it is not a retrieval score and BM25 stays out (user ruling 27 Nov, STK1 A4).

**Gates.** G-B0: the declared cell below is the dense-only arm's best cell within 2,500 nodes in the filed read (`outputs/drp1e/read_<ds>.json`, `outputs/drp1/read_<kb>.json`). G-D1: the arm's identities (`zdrp2.arm_fails(.., "both")`). G-E0 (the seven): the source's first stage is e3v's list. **G-B1: on every question of every chunk, the traced expansion equals the engine's expansion node for node and in order, and the traced pool's golds and size equal `grade3`'s present count and size.** A mismatch stops the row and no number is read from that set. `read` refuses a set without a PASS `gate_trace.json`.

**Cells (declared).**

| set | cell | filed `both` ALL at nodes |
| --- | --- | --- |
| MuSiQue | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 83.4 at 2,485 |
| WebQSP | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 86.6 at 2,099 |
| STaRK-Amazon | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 83.1 at 2,456 |
| STaRK-Prime | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 85.0 at 2,073 |
| STaRK-MAG | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 91.7 at 2,043 |

2Wiki, HotpotQA, MetaQA, SQuAD and MultiHop-RAG reach 95 within about 2,500 nodes (DRP1 sections 15 and 16) with walks that are not budget-following (`h2_c100`, `h3_c25` caps) or with a flat head; they are read through `links` only (the control for the link statistics).

**Subset.** The first 3 chunks (128 questions each, 384 questions) of the seeded order: DRP1's `chunk_order` for the seven (`outputs/drp1/pool/<ds>/cells_<ds>.json`), STK3's `chunk_order(n_chunks, kb index)` for the STaRK sets. The subset is a spread sample by construction. `links` and `lists` are array reads and take the whole split. If a finding needs the whole split, the remaining chunks run under the same declaration.

**Instrument.** The pool engine's own (PL cells over the layouts the filed pools use). JNT1's simple walk is not read here and the two are never mixed in one table. Intervals at the gold level are Wilson and ignore that the golds of one question are not independent (they are too narrow); question-level shares carry a bootstrap where stated.

## 3. Priors (written before any number)

On the subset of 384 questions, dense-only e3v, the declared cells. Each is judged against every value it names; effect sizes stand beside the call.

- **T1.** MuSiQue: of the missed golds, `BUDGET_CUT` + `WINDOW_CUT` + `UPSTREAM_CUT` hold at least 50 percent and `NO_IN_EDGE` + `NO_PATH_IN_H` at most 35 percent (UL5: the oracle depth is 1,831 nodes, under the bound, so the stored links reach the golds and the order loses them).
- **T2.** WebQSP: `NO_IN_EDGE` + `NO_PATH_IN_H` hold at least 50 percent of the missed golds and `BUDGET_CUT` at most 25 percent (UL5: oracle depth 9,301 nodes, 3.7 times the bound; the lever is links).
- **T3.** On MuSiQue and WebQSP, a missed gold whose question has another gold already in the pool has a gold parent in the pool for fewer than 40 percent of such missed golds; where it has, the parent is expanded (state 1) for at least half of them (the walk follows the link and the budget loses the target).
- **T4.** MuSiQue `BUDGET_CUT` golds sit near the cut: the median place over budget is at most 3 (a threefold budget at the cutting hop would catch half of them).
- **T5.** At least 80 percent of the questions have a gold among the five seeds on MuSiQue and on WebQSP.
- **T6.** `links`, S+R: on MuSiQue at least 50 percent of the questions with two or more golds have their golds in one component; on WebQSP fewer than 50 percent; on at least one STaRK set fewer than 50 percent.
- **A1.** Amazon: the dense list's gold recall and ALL at 2,500 are below the fused list's by at least 2 points of ALL (STK3: dense alone 88.1, fused 90.6 at 2,500; DRP1 section 13: the dense-only gap is 4.0 points at about 2,500 nodes).
- **A2.** Amazon: golds found by SPLADE and not by the dense list at depth 1,000 are at least 3 percent of all golds.
- **A3.** Amazon: those SPLADE-only golds have a higher mean share of the question's terms in their text than the dense-top-100 golds (by at least 0.05), and a median text length at least 1.5 times that of the dense-top-100 golds (long documents lose in a fixed-length dense vector).
- **A4.** Amazon: the fused top five has a gold in at least 2 points more questions than the dense top five (SPLADE helps the seeds).
- **A5.** Amazon: the traced walk's ALL on the subset is below the flat pool's at the same node counts, and the loss sits on questions with three or more golds.

## 4. What the read may and may not say

It reports, per set, the table of reasons (counts, shares, Wilson), the bridge states, the budget-cut places, the sole-blocker shares (incomplete questions whose missed golds all have one reason), the seed share, the per-hop frontier, candidate and budget means, and for the STaRK sets the walk-minus-flat ALL at equal nodes with a question-level bootstrap (2,000 resamples, seed 20261128). It answers where the golds are lost. It adopts no rule: a lever it points at (order, budget, seed depth, window, links) is a new declaration with its own rule before its number. The name rule, K, SPLADE and BM25 stay out of the main line; where Amazon's remaining lexical gap shows in the lists it is reported to the user as a decision.

## 5. Rows

Host CPU rows, `retry_cmd` + `pylib_run` as the DRP1 rows run: `brg-mu`, `brg-wq`, `brg-pr`, `brg-mg`, `brg-am` (`trace --ds|--kb`), `brl-*` (`links`, ten sets), `brs-pr`, `brs-mg`, `brs-am` (`lists`; `--text` on Prime and Amazon). Records: `outputs/brg1/<set>/gate_trace.json`, `trace_c<chunk>.npz`, `outputs/brg1/links_<set>.json`, `lists_<kb>.json`; `read` writes `outputs/brg1/read_<set>.json`. The results go into section 6 below with the calls on the priors.

## 7. Amendment 1 (28 Nov 2026 12:22 IST, before any number of `lists --ds`)

**What changes.** `lists` also takes `--ds` (the seven datasets): the e3v dense list and the filed SPLADE list of the dataset's pool source (1,000 deep, the instrument's own), their reciprocal-rank fusion, and the same tables (gold recall and ALL by gold count at depths 5 to 1,000, the dense/SPLADE overlap, the share of questions with a gold in the top five). Why: the user asked to check "wherever we are worse", and the first-stage part of each gap to the filed pool is one table per set (SQuAD -2.28 at 50 nodes in DRP1 section 15; MuSiQue -2.6, WebQSP -1.3 and Amazon -4.0 at about 2,500 nodes). Module sha256 prefix `ef34bbc4a9ab1eb7`; selftest PASS. `trace`, `read`, `links` and `lists --kb` are unchanged in this amendment. Rows that ran before it (`trace` of the five sets, `links` of the ten, `lists --kb` of the three STaRK sets, from 12:21 IST) ran under `fc4cdb5b5ba74a69`.

**Priors for `lists --ds` (written before any of its numbers).**

- **S1.** SQuAD: the fused list's ALL exceeds the dense list's by at least 2 points at depth 50 and by less than 1 point at depth 200 (the 50-node loss of section 15 is the first stage's lexical leg; a deeper list absorbs it).
- **S2.** MuSiQue and WebQSP: the fused-minus-dense gap in ALL at depth 1,000 is under 1 point (the distance of those two sets to the filed pool is not the first stage).
- **S3.** Among the seven, no set other than SQuAD (at depth 50) has a fused-minus-dense gap in ALL above 2 points at any depth of 200 or more.

## 6. Results (28 Nov 2026 13:06 IST)

**What ran.** `trace` on five sets (MuSiQue, WebQSP, Amazon, Prime, MAG), three chunks of the seeded order each (384 questions; Prime 369), G-B1 PASS on all 15 chunks (0 mismatches between the traced and the engine's expansion, grade included); `links` on all ten sets over the whole split; `lists` on the three STaRK sets (with the text properties on Prime and Amazon) and, by Amendment 1, on the seven datasets. The rows ran as `rx exec` streams on the host (the laptop for MultiHop-RAG's `links` and `lists`, whose input is laptop-only), not as feed rows; G-B0, G-D1 and G-E0 passed on every run. Records: `outputs/brg1/<set>/gate_trace.json`, `trace_c*.npz`, `read_<set>.json`, `links_<set>.json` (ten), `lists_<set>.json` (ten). Every number below is a subset read (the first three chunks) unless it says whole split; the flat comparison is post hoc (declared for the STaRK sets only).

**Table 1. Where the subset's golds are, and why the missed ones are missed** (shares of all golds; the last five columns are shares of the missed golds, counts in brackets).

| set | questions | golds | ALL | mean nodes | in head | via expansion | missed | NO_IN_EDGE | BUDGET_CUT | WINDOW_CUT | UPSTREAM_CUT | NO_PATH_IN_H |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue | 384 | 1,280 | 86.2% (331/384) | 2,467 | 85.9% | 9.1% | 64 (5.0%) | 0.0% (0) | 21.9% (14) | 6.2% (4) | 0.0% (0) | 71.9% (46) |
| WebQSP | 384 | 2,784 | 87.8% (337/384) | 2,097 | 8.9% | 54.1% | 1,030 (37.0%) | 0.0% (0) | 37.5% (386) | 57.6% (593) | 3.1% (32) | 1.8% (19) |
| Amazon | 384 | 1,957 | 85.7% (329/384) | 2,440 | 89.0% | 4.1% | 134 (6.8%) | 0.7% (1) | 3.0% (4) | 61.2% (82) | 0.0% (0) | 35.1% (47) |
| Prime | 369 | 809 | 86.7% (320/369) | 2,076 | 50.7% | 38.8% | 85 (10.5%) | 0.0% (0) | 64.7% (55) | 7.1% (6) | 14.1% (12) | 14.1% (12) |
| MAG | 384 | 1,064 | 92.4% (355/384) | 2,045 | 61.6% | 31.7% | 72 (6.8%) | 0.0% (0) | 80.6% (58) | 19.4% (14) | 0.0% (0) | 0.0% (0) |

**Table 2. Bridge states of the missed golds and the budget cuts.**

| set | questions with a gold among the five seeds | missed golds with a gold parent in the pool | of those, the parent was expanded | missed golds with any parent expanded | budget cut: median place over budget [within 2x, 4x] | candidates at the cutting hop / budget (medians) | parent in the dense top 20 / 100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| MuSiQue | 92% (354/384) | 12/64 (19%) | 0/12 | 28% (18/64) | 1.29 [71%, 93%] (n 14) | 15,423 / 1,484 | 8% / 17% |
| WebQSP | 10% (40/384) | 33/1,030 (3%) | 33/33 | 95% (979/1,030) | 1.92 [54%, 88%] (n 386) | 9,713 / 773 | 15% / 23% |
| Amazon | 63% (241/384) | 19/134 (14%) | 0/19 | 64% (86/134) | 2.56 [50%, 75%] (n 4) | 10,808 / 1,816 | 1% / 12% |
| Prime | 29% (108/369) | 15/85 (18%) | 6/15 | 72% (61/85) | 2.58 [42%, 75%] (n 55) | 12,931 / 882 | 28% / 65% |
| MAG | 49% (190/384) | 10/72 (14%) | 10/10 | 100% (72/72) | 2.25 [50%, 71%] (n 58) | 16,856 / 810 | 14% / 26% |

**Table 3. ALL of the subset by number of golds** (questions in brackets).

| set | 1 gold | 2 golds | 3 golds | 4 or more golds |
| --- | --- | --- | --- | --- |
| MuSiQue | none | none | 93.0 (256) | 72.7 (128) |
| WebQSP | 96.8 (185) | 90.8 (65) | 86.5 (37) | 69.1 (97) |
| Amazon | 99.1 (110) | 96.8 (62) | 85.1 (47) | 72.7 (165) |
| Prime | 88.0 (241) | 86.4 (44) | 90.3 (31) | 79.2 (53) |
| MAG | 96.9 (195) | 93.1 (72) | 92.6 (27) | 82.2 (90) |

**Table 4. The walk against the flat pool at the same node count (post hoc; STaRK only; the flat list is the e3v dense list cut at the pool's size).**

| set | walk ALL | flat ALL | walk minus flat | walk / flat ALL by gold count (1 / 2 / 3 / 4+ golds) |
| --- | --- | --- | --- | --- |
| Prime | 86.7 | 65.0 | +21.68 [+17.07, +26.56] | 88 / 72, 86 / 61, 90 / 65, 79 / 36 |
| MAG | 92.4 | 82.0 | +10.42 [+7.29, +13.80] | 97 / 92, 93 / 82, 93 / 81, 82 / 61 |
| Amazon | 85.7 | 88.5 | -2.86 [-6.25, +0.52] (the 188 questions whose flat list is not capped: -4.79) | 99 / 96, 97 / 98, 85 / 87, 73 / 81 (differences +3.6, -1.6, -2.1, -7.9) |

(Amazon's whole-split STK3 figure is -3.3 [-4.6, -2.1]. MuSiQue's and WebQSP's dense lists are shallower than the pool for some questions, so no flat comparison is made.)

**Table 5. Gold-to-gold links on the whole split** (questions with two or more golds; S, R, K, S+R, S+R+K = share of those questions' golds that have a linked partner gold; the last three = share of questions whose golds form one connected set, and by gold count 2 / 3 / 4+ for S+R).

| set | questions | S | R | K | S+R | S+R+K | one component S+R | one component S+R+K | one component S+R by gold count (2 / 3 / 4+) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MetaQA | 5,919 | 2% | 0% | 17% | 2% | 18% | 0% | 1% | 1% / 0% / 0% |
| 2Wiki | 12,576 | 73% | 35% | 18% | 76% | 77% | 50% | 51% | 64% / - / 0% |
| HotpotQA | 7,405 | 80% | 22% | 20% | 80% | 80% | 80% | 80% | 80% / - / - |
| MuSiQue | 2,417 | 30% | 37% | 13% | 52% | 54% | 40% | 41% | 66% / 13% / 11% |
| WebQSP | 740 | 19% | 1% | 26% | 19% | 36% | 9% | 12% | 16% / 14% / 3% |
| MultiHop-RAG | 1,130 | 0% | 88% | 1% | 88% | 88% | 81% | 82% | 74% / 89% / 89% |
| SQuAD | none (one gold per question) | | | | | | | | |
| Amazon | 1,158 | 26% | 35% | 25% | 50% | 64% | 21% | 28% | 36% / 27% / 16% |
| Prime | 1,032 | 34% | 12% | 26% | 42% | 61% | 27% | 37% | 36% / 25% / 22% |
| MAG | 1,373 | 64% | 9% | 22% | 68% | 76% | 41% | 50% | 54% / 40% / 28% |

**Table 6. First-stage lists, whole split: fused (dense + SPLADE) minus dense, ALL in points, at depth 5 / 50 / 200 / 1,000 (2,500 for the STaRK sets); and the golds only SPLADE finds at depth 1,000.**

| set | dense ALL at 1,000 (2,500) | fused minus dense at 5 / 50 / 200 / 1,000 | at 2,500 | golds found by SPLADE only |
| --- | --- | --- | --- | --- |
| Amazon | 76.9 (88.1) | +2.31 / +5.79 / +5.91 / +4.87 | +2.50 | 5.3% (474/8,910) |
| Prime | 56.0 (61.9) | +1.79 / +5.14 / +3.96 / +3.14 | +2.82 | 6.8% (467/6,910) |
| MAG | 74.7 (82.0) | +3.64 / +4.32 / +3.41 / +2.59 | +1.95 | 6.0% (453/7,539) |
| SQuAD | 99.5 | +4.56 / +2.16 / +1.06 / +0.33 | | 0.4% (48/11,873) |
| MetaQA | 12.5 | -0.17 / -0.02 / +0.62 / +5.83 | | 11.3% (8,214/72,697) |
| MuSiQue | 67.0 | +0.46 / +0.74 / +0.08 / -0.50 | | 2.9% (188/6,404) |
| WebQSP | 26.8 | -2.06 / -1.66 / -2.00 / -2.53 | | 0.8% (88/10,583) |
| 2Wiki | 53.2 | +1.88 / +2.38 / +2.29 / +1.58 | | 3.2% (966/30,654) |
| HotpotQA | 81.1 | +3.86 / +5.79 / +4.97 / +2.65 | | 2.8% (419/14,810) |
| MultiHop-RAG | 100.0 (at 500) | -4.25 / -1.06 / -0.09 / +0.00 (at 500) | | 0.0% |

**Reading, set by set.**

- **MuSiQue.** The pool is the 800-deep list head: 85.9 percent of the golds are in it, the walk adds 9.1 percent, 5.0 percent are missed. Of the missed golds 71.9 percent have no path from the five seeds within the cell's two hops, 21.9 percent were candidates a little beyond the budget (median place 1.29 times the budget; 13 of 14 within 4 times) and 6.2 percent were beyond the window. 92 percent of the questions have a gold among the seeds, so the walk starts on the chain and cannot follow it: the incomplete questions are the three- and four-gold chains (3 golds 93.0, 4 or more 72.7), and S+R link all the golds of a question into one connected set for 66 percent of the two-gold, 13 percent of the three-gold and 11 percent of the four-or-more-gold questions. The 12 missed golds that have a gold parent in the pool have it in the head and never expanded (the walk expands the five seeds and what hop 1 keeps). The loss is a link problem; the order has at most the 14 budget-cut golds to recover (1.1 percent of the golds).
- **WebQSP.** 54.1 percent of the golds arrive by expansion and only 8.9 percent through the head; 10 percent of the questions have a gold among the seeds (the answers sit one to three hops from the entity). The 1,030 missed golds are 37.0 percent of all golds and sit in a few questions with very large answer sets (post hoc: 13 questions hold 985 of them); 57.6 percent are beyond the 512-entry window of a frontier node's row and 37.5 percent were candidates at about 1.9 times the budget among about 9,700 candidates; for 95 percent of the missed golds some parent was expanded. Questions with up to three golds reach 86 to 97; four or more reach 69.1. Hub normalisation stays closed (8.15); order and window are what the traces point at.
- **Amazon.** 89.0 percent of the golds are in the 800-deep head and the walk adds 4.1 percent (81 golds) for about 1,640 nodes; the dense list alone at the same node count has more: the walk is 2.86 below the flat pool (4.79 below on the uncapped questions), the loss on four-or-more-gold questions (-7.9). Of the missed golds 61.2 percent are beyond the window (the gold is at place 512 or later of a frontier node's row) and 35.1 percent have no path within two hops; only 3.0 percent are budget cuts, so an order has little to recover; S+R connect all the golds of 21 percent of the questions with two or more. The first stage: the fused list is 2.50 above the dense one at 2,500 and 5.79 above at 50, SPLADE alone finds 5.3 percent of the golds the dense list misses at depth 1,000, and the fused top five holds a gold in 69.8 percent of the questions (dense 63.1). The text is capped at 1,200 characters (the median is 1,200 in every group), so length does not separate the groups, and SPLADE-only golds share fewer question terms with their text (0.340) than the dense top-100 golds (0.414). This is the one remaining lexical gap on the defined encoder, and it is a result for the user's decision; BM25 stays out.
- **Prime.** The walk is 21.68 [17.07, 26.56] above the flat pool. The 85 missed golds are 64.7 percent budget cuts (median 2.58 times the budget among about 12,900 candidates for a budget of about 880; 41 of 55 within 4 times), 14.1 percent upstream cuts and 14.1 percent beyond two hops; for 72 percent of them a parent was expanded. The order inside the budget is the lever here.
- **MAG.** The walk is 10.42 [7.29, 13.80] above the flat pool; ALL 92.4 on the subset (91.7 on the whole split). 80.6 percent of the 72 missed golds are budget cuts (median 2.25 times the budget among about 16,900 candidates for about 810), 19.4 percent beyond the window, none without a path; every missed gold has an expanded parent. Four or more golds reach 82.2.
- **Links across the ten sets** (Table 5). A chain's golds are linked to each other by S and R for 76 to 80 percent of the golds on 2Wiki and HotpotQA, 52 percent on MuSiQue, 88 percent on MultiHop-RAG, 42 to 68 percent on the STaRK sets, 19 percent on WebQSP and 2 percent on MetaQA (where K, at 17 percent, is the only family that links them). The share of questions whose golds form one connected set falls with the number of golds on MuSiQue (66, 13, 11 percent for two, three, four or more golds), 2Wiki, WebQSP, MetaQA and the three STaRK sets, and does not on MultiHop-RAG (74, 89, 89 percent).

**Calls on the priors** (each judged against every value it names; effect sizes beside).

- **T1 NOT HELD.** MuSiQue budget, window and upstream cuts hold 28.1 percent of the missed golds (at least 50 was expected), no-in-edge plus no-path 71.9 percent (at most 35 was expected): the opposite split. The oracle depth of 1,831 nodes (UL5) says the golds are reachable; the cell's two budgeted hops do not reach them.
- **T2 NOT HELD.** WebQSP no-in-edge plus no-path hold 1.8 percent of the missed golds (at least 50 was expected) and budget cuts 37.5 percent (at most 25 was expected); the window (57.6 percent) is the larger cause.
- **T3 NOT HELD as worded.** The share of missed golds with a gold parent in the pool is under 40 percent on both sets (MuSiQue 12/64 = 18.8, WebQSP 33/1,020 = 3.2); the parent is expanded for at least half only on WebQSP (33/33), on MuSiQue for 0 of 12 (the parents sit in the head and are never expanded).
- **T4 HELD.** MuSiQue's median place over budget is 1.29 (at most 3 expected); 10 of 14 within 2 times and 13 of 14 within 4 times.
- **T5 NOT HELD.** MuSiQue 92.2 percent (at least 80 expected) holds; WebQSP 10.4 percent does not.
- **T6 NOT HELD as worded.** MuSiQue's S+R put 40.3 percent of the questions in one component (at least 50 expected); WebQSP 9 percent holds (fewer than 50) and so do Amazon (21), Prime (27) and MAG (41).
- **A1 HELD.** Fused minus dense at 2,500 is +2.50 ALL (90.6 against 88.1), 0.5 above the 2-point line.
- **A2 HELD.** 5.3 percent of the golds (474 of 8,910), against at least 3.
- **A3 NOT HELD.** The question-term share of the SPLADE-only golds is 0.340 against 0.414 for the dense top-100 golds (a difference of -0.074; at least +0.05 was expected) and the median text length is 1,200 for both (a ratio of 1.0; at least 1.5 was expected; the text is capped at 1,200).
- **A4 HELD.** The fused top five holds a gold in 69.8 percent of the questions, the dense top five in 63.1 (+6.7 points; at least +2 was expected).
- **A5 HELD in direction.** The subset walk is 2.86 below the flat pool (post hoc, interval [-6.25, +0.52] includes 0; 4.79 below on the uncapped questions; the whole-split STK3 read is -3.3 [-4.6, -2.1]); the loss is -7.9 on four or more golds and -2.1 on three, and also -1.6 on two (the one-gold questions gain +3.6).
- **S1 NOT HELD.** SQuAD fused minus dense is +2.16 at depth 50 (at least 2 expected, holds) and +1.06 at depth 200 (under 1 expected, misses by 0.06).
- **S2 NOT HELD as intended.** MuSiQue -0.50 and WebQSP -2.53 at depth 1,000 are under 1 only as signed values; by size WebQSP's gap is 2.53 (the fused list is below the dense one).
- **S3 NOT HELD.** HotpotQA +4.97 (depth 200), +3.47 (500), +2.65 (1,000), 2Wiki +2.29 (200), MetaQA +4.02 (500) and +5.83 (1,000) exceed 2 points at depths of 200 or more.

**What the traces say about bridging.**

1. **MuSiQue is a links problem.** The golds that the walk misses are mostly beyond two hops of the seeds, the chain continuation links are absent from S+R for 87 to 89 percent of the three- and four-gold questions, and the order has at most 1.1 percent of the golds to recover. No ordering lever applies; CR1 (filed, no CR2) is the record of the rule links.
2. **WebQSP is the budget and the window on very large answer sets.** About 95 percent of its missed golds are cut by the window or by the budget while a parent was expanded.
3. **Prime and MAG are the order inside the budget.** 65 and 81 percent of their missed golds are budget cuts at 2.3 to 2.6 times the budget, so a better order (or a wider budget) is the lever there.
4. **Amazon's walk loses to the flat list** because 89 percent of its golds are in the head, the walk's expansion adds 4 percent, and the misses are beyond the window or beyond two hops; the lexical gap of the first stage (+2.5 at 2,500) is a decision for the user.
5. The distances to the filed pool elsewhere (MetaQA 93.3 on the whole split, SQuAD's small cells) are not traced here; Table 6 gives their first-stage part (MetaQA: SPLADE finds 11.3 percent of the golds the dense list misses at depth 1,000, fused +5.83 at that depth; SQuAD: +4.56 at depth 5, +2.16 at 50).

Nothing is adopted. The screen of the order and the window on these traces is BRG2 (`docs/BRG2_ORDER_AND_WINDOW_SCREEN.md`, declared 13:01 IST, priors O1 to O9 written before its numbers).
