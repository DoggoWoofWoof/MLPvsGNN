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

## 6. Results

None yet.
