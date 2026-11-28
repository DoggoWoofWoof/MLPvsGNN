# BRG2: which candidate order and which scoring window recover the golds the walk cuts

Declared 28 Nov 2026 13:01 IST, after BRG1 (`docs/BRG1_BRIDGE_TRACE.md`) and before any number of this page. Read-only screen on the subsets of BRG1; nothing is adopted, no cell and no rule of the filed work changes.

## 1. Why

BRG1 gave each gold node that the dense-only, no-K, no-name e3v walk misses one reason. Two reasons are about the walk's own choices and not about the links: the gold is among a hop's candidates but ranked beyond that hop's budget (BUDGET_CUT), or a frontier node lists it beyond the scoring window of 512 entries (WINDOW_CUT). On the subsets they hold: WebQSP 37.5 and 57.6 percent of the missed golds, Prime 64.7 and 7.1, Amazon 3.0 and 61.2, MuSiQue 21.9 and 6.2 (MAG pending). The other reasons (no path within the hops, an upstream cut) are not the walk's order and are untouched here.

The walk orders the fresh candidates of a hop by the question's cosine (key `vr`, or `fu`, the reciprocal-rank fusion of `vr`, `sp`, `rk`) and keeps the first `budget`. Two ideas from the user's list were never run on the traced candidate lists: ordering by how many frontier nodes point at a candidate (a co-citation or personalised-PageRank-like signal, "some PPR based technique") and a per-hop limit guided by the question. The per-hop limit is already the walk's budget; this page tests the order inside it and the window in front of it, and bounds what any order could do (the ceiling).

## 2. What runs

`outputs/mp_unified/zbrg2.py` (new; imports `zbrg1.py` and the pool code, edits none of them). The budget-following walk of PL5 (`zpl5g.expand5`) is re-run hop by hop as in BRG1's `trace_walk`, with the order of a hop's fresh candidates and the scoring window replaced by one of six variants:

| variant | order of the fresh candidates | window |
| --- | --- | --- |
| `control@512` | the cell's own (`order_group5`, rr merge where the cell has it) | 512 |
| `sup@512` | the number of distinct frontier nodes whose row lists the candidate, most first; ties to the control order | 512 |
| `ppr@512` | the sum over those frontier nodes of 1 / (c + place of the frontier node in the frontier + 1), c = the first stage's fusion constant; ties to the control order | 512 |
| `fus@512` | the reciprocal-rank fusion (same c) of the control order and the `ppr` order; ties to the control order | 512 |
| `control@4096` | the cell's own | 4096 |
| `fus@4096` | as `fus@512` | 4096 |

Families keep their rr merge where the cell has it: the variant's order is applied inside each family's list and the lists are merged by place as in `order_fresh5`. The visited cap and the hop budgets are the cell's, so the variants differ in which nodes are kept, not in how many.

Per question and variant the record keeps: the golds in the pool (head and expansion), the pool size, and the golds in the head or among any hop's candidates (`u`: the pool a perfect order would give at this variant's own frontier, the **ceiling**); per hop the golds among the candidates and among the kept.

**Sets, cells, subset.** MuSiQue, WebQSP, Amazon, Prime, MAG, each at BRG1's declared cell (G-B0 of BRG1 re-read: MuSiQue and Amazon `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`, WebQSP, Prime and MAG `dense_top200+STRUCT:h3_bf_v2000~cs-fu`), on BRG1's three chunks of the seeded order (chunk positions 0 to 2). The other five datasets have no cut golds within 2,500 nodes that an order could recover (2Wiki, HotpotQA, SQuAD, MultiHop-RAG meet the target; MetaQA is read in a later declaration if this screen is not closed).

**Gates.** G-O0 = BRG1's G-B0, G-D1 and G-E0 (imported). **G-O1**: `control@512` reproduces the engine's expansion and its grade (present, size) on every question of every chunk; a mismatch stops the chunk and no number of that set is read. The self-test (`zbrg2.py selftest`) checks that `control` equals `expand5` on 192 toy walks (two windows, four settings), that every order keeps the candidate set and that the support and `ppr` counts match a hand-made list: PASS before this page was committed.

**The read** (`zbrg2.py read`): per variant ALL (all golds in the pool, share of questions), golds in the pool, mean nodes, the ceiling, kept share of the candidate golds by hop, ALL by gold count; per variant against `control@512` the paired difference with a 2,000-resample bootstrap over questions (seed 20261128) and the call of `zdrp2.gain_call` (ABOVE if the lower end is above +1.0 point, BELOW if the upper end is below -1.0, else LEVEL). Subset reads are screens: no verdict reaches a rule from them.

## 3. Priors (written before any number of this page)

- **O1 (MuSiQue).** No order at 512 (`sup`, `ppr`, `fus`) is ABOVE `control@512`, and the ceiling of `control@512` is at most 4.0 points above its ALL (BRG1: 72 percent of the missed golds have no path within two hops, 22 percent are budget cuts).
- **O2 (Prime).** `fus@512` is ABOVE `control@512`, and at hop 2 it keeps at least 3 points more of the candidate golds than `control@512` (BRG1: 65 percent of the missed golds are budget cuts at about 2.6 times the budget).
- **O3 (WebQSP, window).** `control@4096` has at least 20 percent more candidate golds (summed over the hops) than `control@512`, and its ALL minus `control@512` is LEVEL, not ABOVE (the candidate lists grow and the same budget cuts them; PL5's wide windows did not displace the best cell).
- **O4 (WebQSP, order).** No order at 512 is ABOVE `control@512` (13 of 384 questions hold 985 of the 1,030 missed golds; an order cannot fit answer sets of hundreds into a budget of about 800).
- **O5 (Amazon).** No variant is ABOVE `control@512` (61 percent of its misses are window cuts, 35 percent have no path, 3 percent are budget cuts).
- **O6 (Amazon, window).** The ceiling of `control@4096` is at least 2.0 points above the ceiling of `control@512` (the window cuts are candidates once the window is wide).
- **O7 (MAG).** No order at 512 is BELOW `control@512` (written before MAG's BRG1 read was opened).
- **O8 (all five).** No variant is BELOW `control@512` on any set, and every variant's mean pool size is within 2 percent of the control's.
- **O9 (G-O1).** `control@512` equals the engine on every question of every chunk (0 mismatches).

## 4. What the read may say

- Each prior is judged against every value of its sentence, not the headline one; a prior with one value outside its wording is NOT HELD as worded and says which.
- A lever is a candidate for one later declaration (a whole-split read on all ten datasets under one rule) only if a variant is ABOVE on at least three of the five sets and BELOW on none. If no variant is ABOVE anywhere, ordering by support or `ppr` is closed as a lever for this walk (this form only: the rule, the cells and the subset of this page) and the remaining distance is in the links (BRG1: MuSiQue's chain links, Amazon's hubs).
- The ceilings say how far any order or window could go at the walk's own frontier: if a set's ceiling is within 1.0 point of its ALL, the order has nothing to recover there.
- Nothing from SPLADE, K, name seeds or BM25 enters. The hub line stays closed (8.15): `sup` counts co-citation by the frontier, it is not a hub normalisation of the links or of the scores.

## 5. Rows

CPU only, on the host through `rx exec` streams (`pylib_run.py outputs/mp_unified/zbrg2.py order --ds D | --kb K --host`), one set at a time per stream, two streams (MuSiQue then WebQSP; Amazon then Prime), MAG after its BRG1 trace ends. A `read` per set on the laptop after the fetch. Cost: six walks per question against BRG1's one.

## 6. Results

None yet.
