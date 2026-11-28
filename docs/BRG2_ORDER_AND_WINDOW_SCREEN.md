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

## 6. Results (28 Nov 2026 13:40 IST)

**What ran.** `order` on the five sets of BRG1, three chunks each (384 questions; Prime 369), module sha256 prefix `30d9a5de5eea529b`, 30 chunk-variants per set. Streams on the host: MuSiQue then WebQSP; Amazon; Prime then MAG; and two extra MAG streams for chunk positions 1 and 2 (the first MAG stream skipped the chunks they had written). G-O1 PASS on every chunk of every set (0 mismatches between `control@512` and the engine's expansion and grade); `control@512`'s ALL equals BRG1's on all five sets (86.2, 87.8, 85.7, 86.7, 92.4). Records: `outputs/brg2/<set>/gate_order.json`, `ord_c*.npz`, `read_<set>.json`. All numbers are subset reads (screens), post hoc in the sense of the page's section 4.

**Table 1. ALL (percent) by variant; the paired difference to `control@512` in points with its interval, and the call.**

| set | `control@512` | `sup@512` | `ppr@512` | `fus@512` | `control@4096` | `fus@4096` |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue | 86.2 | 78.6 (-7.55 [-10.42, -4.95] BELOW) | 79.7 (-6.51 [-9.12, -3.91] BELOW) | 83.9 (-2.34 [-4.17, -0.52] LEVEL) | 85.9 (-0.26 [-1.56, +1.04] LEVEL) | 82.0 (-4.17 [-6.51, -2.08] BELOW) |
| WebQSP | 87.8 | 79.2 (-8.59 [-11.46, -5.99] BELOW) | 76.8 (-10.94 [-14.32, -7.81] BELOW) | 85.4 (-2.34 [-3.91, -1.04] BELOW) | 88.8 (+1.04 [-0.78, +3.12] LEVEL) | 87.5 (-0.26 [-2.86, +2.08] LEVEL) |
| Amazon | 85.7 | 84.6 (-1.04 [-2.08, -0.26] LEVEL) | 84.1 (-1.56 [-2.86, -0.52] LEVEL) | 85.2 (-0.52 [-1.30, +0.00] LEVEL) | 87.5 (+1.82 [+0.00, +3.91] LEVEL) | 86.5 (+0.78 [-0.78, +2.60] LEVEL) |
| Prime | 86.7 | 82.7 (-4.07 [-6.50, -1.90] BELOW) | 82.9 (-3.79 [-6.50, -1.36] BELOW) | 86.7 (+0.00 [-1.36, +1.63] LEVEL) | 86.4 (-0.27 [-1.90, +1.36] LEVEL) | 85.9 (-0.81 [-2.71, +0.81] LEVEL) |
| MAG | 92.4 | 87.8 (-4.69 [-6.77, -2.86] BELOW) | 87.0 (-5.47 [-8.07, -3.39] BELOW) | 91.4 (-1.04 [-2.34, +0.00] LEVEL) | 93.2 (+0.78 [+0.00, +1.82] LEVEL) | 91.7 (-0.78 [-2.34, +0.78] LEVEL) |

**Table 2. The ceiling: ALL of a perfect order at each variant's own frontier (golds in the head or among any hop's candidates), and what the wide window adds.**

| set | ALL (`control@512`) | ceiling `control@512` | ceiling `control@4096` | candidate golds, 512 to 4096 (summed over hops) | kept share of the candidate golds at hop 2 / hop 3 (`control@512`) | mean pool size of `control@4096` over `control@512` |
| --- | --- | --- | --- | --- | --- | --- |
| MuSiQue | 86.2 | 89.3 (+3.1) | 90.1 | 568 to 608 (+7.0%) | 95.2% / - | 0.961 |
| WebQSP | 87.8 | 93.2 (+5.4) | 97.7 | 2,166 to 2,425 (+12.0%) | 70.2% / 25.9% | 0.990 |
| Amazon | 85.7 | 86.5 (+0.8) | 90.4 | 983 to 1,191 (+21.2%) | 99.1% / - | 0.957 |
| Prime | 86.7 | 96.5 (+9.8) | 97.8 | 675 to 692 (+2.5%) | 83.5% / 41.3% | 0.995 |
| MAG | 92.4 | 97.4 (+5.0) | 99.5 | 841 to 881 (+4.8%) | 83.0% / 39.7% | 0.990 |

Pool sizes of the orders at 512 against the control: `sup` 1.014 to 1.058 times, `ppr` 1.021 to 1.063 times, `fus` 1.000 to 1.002 times.

**Calls on the priors** (each against every value of its sentence).

- **O1 HELD.** MuSiQue: `sup` -7.55 and `ppr` -6.51 are BELOW, `fus` -2.34 LEVEL, none ABOVE; the ceiling of `control@512` is 3.1 points above its ALL (at most 4.0 expected).
- **O2 NOT HELD.** Prime `fus@512` is +0.00 [-1.36, +1.63] LEVEL, not ABOVE, and at hop 2 it keeps 83.1 percent of the candidate golds against the control's 83.5 (3 points more were expected). (Prime's `control` key is already the fusion `fu`.)
- **O3 NOT HELD as worded.** WebQSP `control@4096` has 12.0 percent more candidate golds than `control@512` (at least 20 expected); its ALL is +1.04 [-0.78, +3.12], LEVEL (as expected).
- **O4 HELD.** WebQSP: `sup` -8.59, `ppr` -10.94 and `fus` -2.34 are BELOW; none ABOVE.
- **O5 HELD.** Amazon: all five variants LEVEL (-1.56 to +1.82); `control@4096` is the nearest to ABOVE (+1.82 [+0.00, +3.91]).
- **O6 HELD.** Amazon's ceiling rises from 86.5 to 90.4 with the wide window (+3.9; at least +2.0 expected).
- **O7 NOT HELD.** MAG: `sup` -4.69 [-6.77, -2.86] and `ppr` -5.47 [-8.07, -3.39] are BELOW (none BELOW expected); `fus` -1.04 LEVEL.
- **O8 NOT HELD.** BELOW appears on four of the five sets (`sup`, `ppr` on MuSiQue, WebQSP, Prime, MAG; `fus` on WebQSP; `fus@4096` on MuSiQue), and the pool sizes of `sup` and `ppr` are 1.4 to 6.3 percent above the control's, of `control@4096` up to 4.3 percent below (within 2 percent expected).
- **O9 HELD.** 0 mismatches on all 15 chunks of the five sets, 3 chunks each.

**What the screen says.**

1. **Ordering the candidates by how many frontier nodes list them, or by a reciprocal-rank sum of the frontier's places, is worse than the question's cosine on every set** (BELOW on four of five, LEVEL on Amazon at -1.04 and -1.56; the hop-2 kept share of the candidate golds falls by 8 to 42 points, Amazon 99.1 to 91.2 and 81.2, MuSiQue 95.2 to 54.0 and 52.9), and its fusion with the control order is level or below. The question's cosine is a stronger signal than the frontier's agreement. By the page's section 4 rule (a lever needs ABOVE on at least three of five sets and BELOW on none) **support, `ppr` and `fus` are closed as orders for this walk in this form**, and no variant is ABOVE on any set.
2. **The wide window raises the ceilings (+0.8 to +4.5 points; Amazon +3.9, WebQSP +4.5) but its ALL is level on all five sets** (WebQSP +1.04, Amazon +1.82, MAG +0.78 positive, MuSiQue and Prime -0.26 and -0.27): the candidates arrive and the same order does not keep them.
3. **The headroom is in the order and it is large where the walk already has the golds as candidates.** At the cell's own frontier a perfect order would give Prime 96.5 (+9.8), MAG 97.4 (+5.0), WebQSP 93.2 (+5.4), MuSiQue 89.3 (+3.1), Amazon 86.5 (+0.8); with the wide window 97.8, 99.5, 97.7, 90.1, 90.4. MuSiQue (+3.1) and Amazon (+0.8) have little at the cell's frontier (their misses are mostly beyond it: BRG1's no-path and window cuts), Prime, MAG and WebQSP have 5 to 10 points that no parameter-free key of PL4 to PL6 or this page realises. A scorer that sees the question and the edge is what is left for them; the learned stage is the place to measure it.

Nothing is adopted.
