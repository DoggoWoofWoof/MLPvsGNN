# PL5: the coverage target, round 4 (a cap that follows the budget, a wider scoring window, one fused order, on all seven)

Declared 25 November 2026, before any PL5 coverage number exists. What I have read are PL4's FILED records for the five laptop datasets
(`outputs/pl4/<d>/select.json`, `confirm_frozen.json`, `confirm_e3v.json`, `miss.json`, `outputs/pl4/gr_*.json`, `outputs/pl4/report.md`; PL4's page
sections 1 to 14.5; commits c71f0d7 declaration, c25eddf five of seven read, 9b5350e7 builder), PL3's records they rest on, the walk code of
`m3b_pools`, `zpl3g.py` and `zpl4g.py`, and the array names and shapes of the family stores (`outputs/m3b/csr/*.npz`). **One labelled diagnostic was run before this commit,
and it informs the design: D3 (section 1.2), 300 dev questions evenly spaced, on PL4's three FILED chosen cells of the SHORT datasets (MuSiQue, MetaQA, WebQSP);
it reads the labels, it evaluates no PL5 cell, and it is the same kind of post-hoc diagnostic as PL3's D1 and PL4's D2, which informed PL4 and PL5 in the same
way.** Every other run before this commit is label-free or an agreement check: the cell list and its sha256 (`zpl5g.py cells`), the cost of the walks
(`zpl5g.py time`: seconds and nodes scored), an agreement check of the engine's walk against a per-node loop on real questions of WebQSP and MuSiQue (2 questions each, the 54 walks of an earlier draft of the list: 0 differences;
the declared 46: 0 differences on the same two WebQSP questions after the final edit, and the self-test's G-X5 on MultiHop-RAG and SQuAD, 552 checks each), and the engine self-test (PASS in 606 s; agreement counts, no coverage). **Nothing here trains, no model is fitted, no LLM or spaCy is used,
and every pool function is gold-free.** The pools of record, PL1's pools and PL2's, PL3's and PL4's code and records are untouched. This page is the fourth round of the
coverage target (docs/PL1_POOLS_ON_UL2_LAYOUT.md section 7).

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how much ever is
  possible"; "dont wait on me work on the fixes to whatever comes up keep the cpu and gpu full dont waste time or compute".
- Earlier, still binding: "the pools or the rules or links all should be universal"; "i dont want to freeze pools for the sake of it, make sure we
  freeze when we know it is the best for that specific arm"; "link improvement with precision is all we need". Pools first and parameter-free (no
  learning, no model, no LLM, no spaCy); one model per family; the hub line is CLOSED (UL2 8.15): none of the changes below reads a node's degree or
  adds a hub step; every rule is universal on all seven datasets (MultiHop-RAG, MuSiQue, SQuAD, MetaQA, WebQSP, HotpotQA, 2Wiki), carried everywhere
  or nowhere. The 2,500 mean-node bound of PL2 to PL4 is not changed by this page.

## 1. What PL4 found, and what D3 adds (the records this round starts from)

### 1.1 PL4 (read 5 of 7; HotpotQA and 2Wiki are host rows still returning; the old call and the frontier reading are both INCOMPLETE)

PL4 chose, per dataset, the R95 cell of the union of PL2's 804, PL3's 1,470 and its own 1,932 cells (dev, frozen lists) and confirmed it on eval:

| dataset | PL4's chosen cell (the incumbent of this page) | eval ALL (frozen) at nodes | state |
|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 at 50.0 | past 0.95 |
| squad | `rrf1000_b15` | 0.9554 at 15.0 | past 0.95 |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 at 2,274.9 | **SHORT** |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 at 2,335.9 | **SHORT** (0.9296 on e3v) |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 at 2,468.1 | **SHORT** (0.7618 at 2,720.1 nodes on e3v) |
| hotpotqa, 2wiki | PL4's host rows (this page reads them as they land) | | |

PL4's D1 and D2 (its section 14.5; 300 dev questions): of the missing golds of the three SHORT cells, **none is R (a ranking cut) and almost none is U (outside the uncapped reach)**; they are C,
cut by a cap or by the visited budget. D2 attributes MuSiQue's 37 to the cap at hop 2 (13) and hop 3 (6), the budget at hop 2 (9) and hop 3 (7) and 2 upstream; MetaQA's 373 to the budget at hop 3 (264), the
cap (27) and upstream (82); WebQSP's 245 to **the per-seed cap at hop 1 (126) and the per-frontier cap at hop 2 (97)**, 22 upstream, **none to the budget**. PL4's four options (`sp`, `rk`, `eq`, `rr`) reorder
or re-share the visited budget and leave the caps alone, so on WebQSP they cannot move anything (identical ALL and sizes on every eval question) and on MetaQA `eq` (equal shares) bought nothing (its cells alone reach
0.8297 on dev against 0.8978).

### 1.2 D3 (new; labels read; post-hoc; 300 dev questions evenly spaced; PL4's three chosen cells; `zpl5r.py place`, committed with this page)

D3 asks, of each gold missing from PL4's chosen cell's pool, **where it sits** in what the walk listed:
- **Cap-cut golds** (D2's rule: a neighbour of a node the walk kept, not among the capped entries): is it inside the first 512 neighbours of the node's row (the `cs` scoring window), what is its cosine place
  among that node's window and what is its place among the hop's own fresh candidates under each key (cosine `vr`, support `sp`, row rank `rk`); if outside the window, the smallest row position any parent gives it.
- **Budget-cut golds** (in the capped entries, not kept): its place among the hop's capped fresh candidates under `vr`, `sp`, `rk`, and how many fit in what was left of the visited budget.

| dataset (cell) | cut | hop | golds | inside the 512 window | outside; within row position 1,024 / 2,048 / 4,096 / 16,384 |
|---|---|---|---|---|---|
| WebQSP (`...h2_c100_v5000~s10-cs-sp`) | cap | 1 | 126 | **123** (cosine place median 170 among a median 350 candidates) | 3; 2 / 2 / 3 / 3 |
| WebQSP | cap | 2 | 97 | **57** (of 4,305 fresh candidates, median) | 40; 19 / 28 / **30** / 34 |
| MuSiQue (`...h3_c25_v2000~cs-vr-rr-eq`) | cap | 2 | 13 | 0 | 13; 4 / 10 / **11** / 13 |
| MuSiQue | cap | 3 | 6 | 3 | 3; 3 / 3 / 3 / 3 |
| MetaQA (`...STRUCT:h3_c25_v2000`) | cap | 2 / 3 | 15 / 12 | 12 / 12 | 3 / 0; 1 / 3 / 3 / 3 at hop 2 |

Budget-cut golds: **MetaQA 264 at hop 3** (hop 3 has 4,390 fresh candidates, median, of which the budget left admits 1,150): under the cell's own order none fits by definition; **under cosine, 152 of the 264 would fit
in what was left; under support, 109; under row rank, 91** (within the first 1,000 candidates by cosine: 132; 2,000: 215). MuSiQue 16 (hop 2: 9, of which 6 fit under cosine in what was left, 1,728 median; hop 3: 7, of which none fits under cosine in the 880 left, 1 under support). WebQSP none. Read: for WebQSP, cap-cut golds at hop 1 are all inside the window and are cheap to reach (122 of the 123 sit within the first 500 fresh candidates), and at hop 2 a cap that keeps the best by cosine
until the budget is spent reaches 51 of the 57 inside-window golds within 1,500 candidates (45 within 1,000, 54 within 2,000); 30 more of the 40 outside the window sit in the first 4,096 entries of their row, which is a
wider window, not a different cap.

### 1.3 The lever, and what D3 says against using it alone

**The lever of this page is the one D2 named: a cap that follows the budget** (no fixed per-seed or per-frontier cap; every hop keeps as many as its share of what is left of the visited budget admits). It is aimed at WebQSP
(223 of 245 cap-cut) and MuSiQue (19 of 37). **I considered a different lever and did not make it the main one:** MetaQA's shortfall is the hop-3 budget (264 of 375 missing, 70 percent), which a budget-following cap does not
touch, and PL4's `eq` is the same equal-share schedule that this page's walk applies (to hop 1 as well); PL4 found it bought nothing on MetaQA. So **this round is expected to move WebQSP most, MuSiQue somewhat, and MetaQA
little or not at all; it may stay SHORT on all three**. What D3 shows for MetaQA is that the hop-3 order matters more than the share: cosine would fit 152 of the 264, row rank 91. That is a different lever (an order for the
last hop that sees the bridge, section 12) and it is not in this grid: the options here that touch order are `vr` (cosine, the lever PL3 and PL4 used) and `fu`, a parameter-free fusion of cosine, support and row rank, included as one
cheap test of whether the three orders together fit more than cosine alone (D3 says each alone fits fewer than cosine on MetaQA, so `fu` is a hedge, not an expectation). Two limits of the D3 reading: it is conditional on PL4's walk
(a different hop-2 frontier changes what hop 3 lists), and the 4,096-entry window is a cost trade, not a finding that 4,096 is the right width (PL4 section 5 left a wider window out for its cost).

## 2. What is fixed (PL2's, PL3's and PL4's pages, by reference)

Everything of PL2 sections 2, 3 and 4.1 to 4.2, PL3 sections 2 and 4 and PL4 sections 2, 4 and 6 (the walk, `cs`, `vr`, `FS`, `s10`, `s20`, `sp`, `rk`, `rr`, `eq`, the calls and their statistics) is unchanged and read by
the same code (`zpl2g.py`, `zpl3g.py`, `zpl4g.py` imported, not edited): the substrate (the served package for the six datasets, MultiHop-RAG's own B0 layout), the dev carves (`s1sel`; MultiHop-RAG's dev split;
WebQSP's `sorted(train ids)[1::2]`) and eval carves (`s1eval`; MultiHop-RAG's test), the frozen first stage for selection and the frozen and e3v first stages for confirmation, the seeds (dense top 5 U SPLADE top 5) unless a
cell says otherwise, the measures (ALL = `all_gold_at_pool`, ANY, mean nodes = `candidates_mean`, Wilson 95% on ALL, paired bootstrap of 1,000 resamples), the floors 0.95 and 0.98, the bound **2,500** mean nodes, the bases and sizes
(`rrf1000_b25`, `b100`, `b400`, `b800`, `b1600`, `dense_top200`; M = 50, 100, 200, 400, 800, 1,600). A walk with none of this page's new options is PL2's, PL3's or PL4's walk, bit for bit (gate G-R).

## 3. What this round tests, and what it does not

| name | here | walks / cells |
|---|---|---|
| a cap that follows the budget (PL4 14.8, PL3 14.8) | walk setting `h{H}_bf_v{V}` (`bf`): no per-seed or per-frontier cap but the scoring window (512 entries); every hop keeps the best of what the window lists until its share of the visited budget is spent; the budget is the only cut | 30 walks (G1 to G3) |
| a window wider than 512 (PL4 section 12) | setting `h{H}_bf_v{V}_w4096`: the window is the first 4,096 entries of a row, scored by cosine (D3: 30 of WebQSP's 40 and 11 of MuSiQue's 13 outside-window hop-2 golds sit there) | 8 walks (G4) |
| one fused order | key `fu` on the `bf` walks: reciprocal-rank fusion of the orders `vr`, `sp`, `rk` of the same candidate list | 12 walks of G1 |
| the one-hop walk with the whole window | setting `h1_cW` (1 hop, cap 512, no budget): the hop-1 golds D3 puts inside the window at their best place | 4 walks (G5) |
| a parent-conditioned rank, the weight values of NER and KNN rows, label-ordered caps, a hop-weighted schedule, a last-hop-only order | **not in this round** (section 12) | 0 |

## 4. The changes, exactly (the walk; section 8's gates hold the code to this text)

A **walk** is `{regime}:{setting}~{mods}`, mods joined by `-` in the order `s10`, `s20`, `cs`, `vr`, `sp`, `rk`, `fu`, `rr`, `eq`. A walk whose setting is not `bf` or `h1_cW` is PL4's, PL3's or PL2's walk (`zpl4g.expand4`), unchanged.
A **budget-following walk** (`h{H}_bf_v{V}` or `h{H}_bf_v{V}_w4096`, H = 2 or 3 hops, V = the visited budget in {1,000, 2,000, 5,000}, the window W = 512, or 4,096 for `_w4096`) takes `cs` and exactly one key, `vr` or `fu`, and takes no
`sp`, `rk` or `eq` (it is `eq`-shared by construction); `rr` only on regime FULL; `s10` and `s20` never together. For every hop h = 1..H, in order:

1. **The list.** For every node of the frontier (the seeds at hop 1; at hop h >= 2 the nodes hop h-1 kept, in the order it kept them) and every family of the hop (the regime's families at hop 1; its hop-2/3 families after, as in
   PL3 section 4), the node's first W entries of its row in that family are scored by their float64 inner product with the question vector and ordered best first (ties to the lower node id): the node's **list**; an **entry** is
   (node x, frontier index i, family, place r), r = 0 for the first of a list. There is no per-seed or per-frontier cap other than W.
2. **Fresh nodes.** Entries whose node is already visited (the seeds and every node kept at an earlier hop) are removed; the distinct nodes that remain are the **fresh nodes**.
3. **Order.** `vr`: by cosine to the question, highest first, ties to the lower id. `fu`: a node's score is the sum over the three orders `vr`, `sp`, `rk`, in that order, of `1 / (60 + p + 1)`, p = its 0-based place in that order of the same
   fresh list (`sp`: by the number of distinct frontier nodes that have an entry for the node, highest first, ties by cosine then id; `rk`: by the smallest place r any of its entries has, ties by cosine then id; 60 = the constant the
   first stage's fusion uses); ordered by -score, then id. `rr` (FULL, with `vr` or `fu`): each family's entries are ordered by the key on their own, the families' lists are merged by place (every family's first, then every family's second,
   ..., within a place STRUCT, NER, KNN), a node keeps its first appearance (PL4 section 4).
4. **Budget.** With `rem` = V minus the number of nodes visited so far (the seeds plus every node kept before) the hop keeps the first `ceil(rem / (H - h + 1))` fresh nodes in that order (the last hop keeps all that is left; what a hop
   cannot fill passes to the next). The walk stops when `rem` <= 0 or the frontier or the fresh list is empty. The kept nodes join the visited set and are the next frontier, in order.

The **one-hop window walk** `h1_cW~cs` (and `~s10-cs`): 1 hop from the seeds (the fused first stage's top 10 with `s10`), each seed's neighbours in the regime's families, the best 512 by cosine out of its first 512 entries, in
PL3's candidate order; no key, no budget (a PL3 `h1_c100~cs` walk with cap 512). **Seeds, regimes, `FS`, `s10`, `s20`, `cs`, `+top{M}` and the union cells are as in PL3 and PL4** (section 2); the ranked cells `+top{M}` rank the
walk's nodes by cosine to the question, as before. A budget-following walk has at most V nodes counting its seeds, so its ranked cells exist only for M <= V. Inner products are accumulated in float64 from the fp16 rows in numpy's pairwise
order (no BLAS), one score per node and question; ties go to the lower node id everywhere; nothing is random.

## 5. The cells (fixed at this commit; nothing is added after a number is read)

- Bases and sizes as PL3 (section 2): 6 bases; M in {50, 100, 200, 400, 800, 1,600}. Each walk x each base as a union (6) and as a ranked cell (6 x its sizes) = 42 cells (36 when V = 1,000), minus the names PL2, PL3 or PL4 already hold.
- **46 walks** (`zpl5g.py cells` prints them; every one has a new setting, none repeats a PL2, PL3 or PL4 walk):
  - **G1, the budget-following walk, two keys** (24): `cs-vr` and `cs-fu` on STRUCT and FULL x H in {2, 3} x V in {1,000, 2,000, 5,000}.
  - **G2, the families take turns** (6): FULL, H in {2, 3} x V in {1,000, 2,000, 5,000}, `cs-vr-rr`.
  - **G3, wider seeds** (4): FULL, H in {2, 3} x V in {2,000, 5,000}, `s10-cs-vr`.
  - **G4, the wide window** (8): STRUCT and FULL x H in {2, 3} x V in {2,000, 5,000} on `_w4096`, `cs-vr`.
  - **G5, one hop with the whole window** (4): `h1_cW` with `cs` and with `s10-cs`, on STRUCT and FULL.
- **1,872 cells**, all new. Cell list sha256 (`P2.cells_sha256`, the sorted names): `a5a759803992cc0524a4ebd4de21d21396ea599dd9f0597329be9842ab9baaeb` (1,872 distinct). PL2's 804:
  `76bdf3a822e6bb8caf21121f853c9a8383e7844f99425e004d18c75f2aededc1`; PL3's 1,470: `ff879881090993bca1e063b715724c0d89411b8cacc77f5df7bcccc2b29c642e`; PL4's 1,932:
  `32502a00db12e3afffd325170183f0508a7f108afecd505c73a516a4908eed96`; the union of the four, **6,078** distinct: `64da37db623e47348dae6626c47161e56c782f9fa511f24e56966ed79e38e7a0`.
- Tags (descriptive, for what each option bought): `bf`+`vr` 480, `bf`+`fu` 480, `bf`+`vr`+`rr` 240, `bf`+`vr`+`seeds` 168, `bf`+`vr`+`wide` 336, `u1` (the one-hop window walks) 84, `u1`+`seeds` 84 (they sum to 1,872).
- The walk list was fixed after two cost timings on MuSiQue (40 dev questions, no label read): the first draft of 54 walks (with `fu` on the wide walks and on `rr` and `s10`) read 9.4 s per question, about 2.4 times PL4's 3.9 s, and the declared 46
  read **4.94 s per question (73,814 nodes scored per question; 0.63 GB), about 1.27 times PL4's**; the `fu` walks and the 4,096 window are the costly ones, so `fu` is on W = 512 only and the wide window carries `vr` only.
- Cells above the bound are read, cannot be selected, and are shown in the any-size readings (section 6).

## 6. The rule, the confirmation and the call

**Selection** (dev, frozen lists, over the union of PL2's 804, PL3's 1,470, PL4's 1,932 and the 1,872): among cells with mean nodes <= 2,500, **R95** = the smallest mean nodes with dev ALL >= 0.95 (ties to the higher ALL, then the name);
if none, the highest ALL (ties to fewer nodes), and the dataset is flagged SHORT. **R98** likewise at 0.98. Readings (each by the same rule): `all` (the decider), `prev` (PL2's, PL3's and PL4's 4,206 cells, the incumbent grid: it must give PL4's
filed pick, checked against `outputs/pl4/<d>/select.json`), `pl3` (PL2's and PL3's cells), `new` (the 1,872 only), and `only:TAG` per tag. The whole curve and the frontier of each reading are filed.
**Beside the bound (they decide nothing)**: R95any and R98any = the smallest mean nodes with dev ALL >= the floor at **any size** (no bound), and the matched-budget picks at N = 3,000, 4,000, 5,000, 6,000, 7,000 (the price of coverage above the bound).

**Confirmation** (eval, nothing re-chosen): the cells R95, R98, R95any, R98any name in the readings `all`, `prev` and `new`, the matched-node-count picks (bounded and any-size), and PL2's filed own construction are re-run unchanged on the eval carve,
with the frozen lists and with e3v's lists.

**The call, as PL3 and PL4 state it (the old rule, unchanged).** On eval, frozen lists, per dataset: NEW = R95 of the reading `all`, OLD = R95 of the reading `prev` (PL4's own chosen cell, the incumbent). Paired bootstrap of the ALL difference (NEW - OLD) and of
the ratio of mean nodes (NEW / OLD); one generator per dataset, `numpy.random.default_rng([20261125, k])`, k the dataset's place in the order MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MuSiQue, MetaQA, WebQSP, consumed in the order R95, R98,
the new-only pick, the five budgets below, R95any, R98any, then the five any-size budgets.
- **BELOW**: NEW misses 0.95 where OLD reaches it, or the ALL difference's upper end is below 0 with the nodes ratio's lower end at or above 1.
- **ABOVE**: NEW reaches 0.95 where OLD does not, or the nodes ratio's upper end is below 1 with ALL not BELOW, or the ALL difference's lower end is above 0 with the nodes ratio's upper end at or below 1.
- **LEVEL**: anything else.
**CARRY_PL5** when all seven are read, no dataset is BELOW, at least one is ABOVE, and every chosen cell is within the 2,500 bound on the selection carve; INCOMPLETE while fewer than seven are read; otherwise NOT_CARRIED (PL4's grid stays the rule).

**The frontier reading (PL4's, unchanged).** For each budget N in {500, 1,000, 1,500, 2,000, 2,500} mean nodes: NEW_N = the cell of the reading `all` with the highest dev ALL among cells with dev mean nodes <= N (ties to fewer nodes, then the name); OLD_N the same over
the reading `prev`. On eval (frozen lists), the paired bootstrap of ALL(NEW_N) - ALL(OLD_N): the budget is **ABOVE** if the interval's lower end is above 0, **BELOW** if its upper end is below 0, else LEVEL. A dataset is BELOW if any budget is BELOW, ABOVE if
at least two budgets are ABOVE and none is BELOW, else LEVEL. **CARRY_PL5_F** when all seven are read, no dataset is BELOW by either reading, at least one is ABOVE by the frontier reading, and every chosen cell is within the bound. The two calls are filed
under their own names (`call` and `call_frontier` in `outputs/pl5/call.json`); the old reading is never hidden or replaced; which one governs is the user's decision.

**The price at any size (new; filed beside the calls, decides nothing).** The target is coverage as close to 1.00 as possible **at the smallest node counts**, and a bound of 2,500 can hide what a larger pool costs. So the call file also holds, per
dataset, R95any and R98any of NEW against OLD (the same paired statistics and `status_of`, `price` in `call.json`) and the five any-size budgets 3,000 to 7,000 (`frontier_any`: per budget the same interval and status; informational, with no BELOW/ABOVE
consequence). They say what the 0.95 floor costs in nodes when the bound is lifted and whether the new cells make it cheaper.

**How the bound is read on e3v lists.** As PL4 section 6: selection and both calls read the frozen dev lists; the e3v confirmation re-runs the chosen cells on the eval carve's e3v lists and reports its mean nodes beside the frozen reading with a flag `OVER_BOUND`
when it exceeds 2,500; the flag changes neither call.

## 7. Stop rule

The 46 walks, the bases, the sizes M, the settings, the windows, the bound, the floors, the readings, the budgets of the frontier reading and the any-size budgets, and the carves are fixed by this commit; every output records the sha256 of the cell list and the
code-path hash, and `zpl5g.py` refuses a grid whose cell list is not the declared one and a select whose heads carry another code-path hash. Nothing is added to the grid after a number is read. If a dataset is still SHORT after this round the report (a)
says in numbers which gold nodes remain unreached and which stage cut them (section 10), and (b) names the next change, which is declared on its own page.

## 8. Gates (each is a failure that stops that dataset's reading; none of its numbers is read)

- **G-R (PL2's, PL3's and PL4's cells are reproduced).** The PL5 engine, given the 2,274 cell names of PL2 and PL3, builds them on the first 400 dev questions of a dataset and, given PL4's 1,932, on the first 150, and every present-count and pool-size equals the
  filed dev arrays (PL2's `grid_dev_frozen`, PL3's and PL4's three parts) element for element. Required on every dataset that is read; the host's own G-R on MuSiQue is a precondition of every host select and confirm (`pl5-musiquegr`).
- **G-X5 (the walk).** With the PL5 options off the engine's walk equals `zpl3g.expand3` array for array, on 60 questions, for every regime and setting; every PL2, PL3, PL4 and PL5 walk (including the 46 above) equals an independent per-node Python loop written from
  section 4 (and PL4's for the PL2 to PL4 walks), on 3 questions per dataset.
- **Engine self-test.** The engine's present counts and pool sizes against a brute-force pool builder written from section 4 (explicit sets) on a sample of questions, 3 cells per walk (MultiHop-RAG and SQuAD).
- **G-F (gold-free).** No pool function takes or reads a label (checked on the signatures and the source of every function that builds a pool); the states are built, and hashed for G-D, from a source whose label arrays are withheld. Labels enter only in `zpl3g.grade3`.
- **G-D (determinism).** The pools of a sample of questions hash the same under 1 thread with chunk 64 and 4 threads with chunk 7.
- **G-U (universal).** One code-path sha256 (the pool code, the constants and PL4's code-path hash, which holds PL3's and PL2's) and the cell list's sha256 are recorded in every grid head; `select` refuses a head that differs, and `call` refuses to file when two
  datasets differ.

## 9. What each option can and cannot show

| option | what it can show | what it cannot show |
|---|---|---|
| budget-following cap (`bf`, W = 512) | whether keeping the best 512 of every list by cosine, and letting the budget alone cut, reaches the golds a cap of 25 or 100 dropped (WebQSP's hop-1 and hop-2 golds D3 puts inside the window) | a gold outside the first 512 entries of its row (WebQSP 43 of 223; MuSiQue 16 of 19); a hop-3 budget cut (MetaQA); it spends the budget that PL4's caps left unspent, so cells are larger (WebQSP's PL4 cell reads 2,468 nodes against the 2,500 bound) |
| wide window (`_w4096`) | whether scoring a node's first 4,096 entries reaches the outside-window golds (MuSiQue hop 2, WebQSP hop 2) | what a window of 16,384 would do (D3 says 34 of WebQSP's 40 sit within it); the cosine cost grows with the window (about +25 percent per walk in the timing) |
| fused order (`fu`) | whether cosine, support and row rank together keep more of a hop's golds than cosine alone | a gold no order reaches inside the budget; D3 says on MetaQA each of `sp` and `rk` alone fits fewer of the hop-3 golds than `vr`, so `fu` is not expected to win there |
| one hop with the whole window (`h1_cW`) | whether the hop-1 golds alone (WebQSP's 123 inside-window ones) reach a pool in few nodes | any gold that needs a second hop |

None of these reaches a gold the first stage and the graph together do not reach; the pools and links are rule-made, no label is read, and no semantic signal is added that the first stage's encoder did not have.

## 10. The short datasets: what remains unreached (D1, D2, D3; post-hoc and labelled so; they decide nothing)

For each dataset whose R95 of the reading `all` is SHORT, at that cell: **D1** (class R / C / U of each missing gold) and **D2** (which stage cut each C gold: cap, budget, upstream, by hop), as PL4 section 10, from a traced walk of the cell (for a budget-following walk the
trace is `zpl5g.ref_walk5`'s, checked equal to the engine's on every question; the stage "cap" for a `bf` walk is the scoring window); and **D3** (section 1.2: where the golds sit, and how many would fit in what was left under each order). Both read the labels,
are run after `select`, and are never an input to any rule. The code is `outputs/mp_unified/zpl5r.py` (`place` for D3, committed with this page; `miss` and `report` are added with the results).

## 11. Code, rows, outputs

- **New modules only:** `outputs/mp_unified/zpl5g.py` (engine, commands `cells`, `time`, `gr`, `grid`, `select`, `confirm`, `call`, `selftest`; every command except `cells` and `call` takes `--host`; committed with this page) and `outputs/mp_unified/zpl5r.py` (D3 now; the report, D1 and D2 with the
  results; it reads filed records and the labels for the diagnostics only, chooses nothing). They import `zpl2g.py`, `zpl3g.py`, `zpl4g.py`, `zpl4r.py`, `scripts/m3a_headroom.py` and `mp_retrieval.m3b_pools` and edit none of them.
- **Outputs:** `outputs/pl5/<d>/{grid_dev_frozen*.npz (arrays, uncommitted), grid_dev_frozen*.json (heads), select.json, confirm_frozen.json, confirm_e3v.json, miss.json, place.json}`, `outputs/pl5/{cells.json, selftest.json, gr_<d>.json, call.json, report.md, feed_rows.txt}`.
- **Laptop** (one heavy job at a time, three parts of the dev grid in parallel, frozen lists): MultiHop-RAG (with e3v), MetaQA, MuSiQue, WebQSP and SQuAD; expected cost per dataset about 1.27 times PL4's CPU time (PL4: 82 s, 2,392 s, 9,488 s, 8,308 s, 7,012 s, so about
  105 s, 3,040 s, 12,050 s, 10,550 s, 8,900 s summed over the parts, an upper bound, as PL4's were taken with several heavy processes sharing the CPUs).
- **Host rows** (`pl5-*`, CPU only, one class per row: the first two name tokens are unique per row; `pylib_run.py` convention) in `outputs/pl5/feed_rows.txt`, for **HotpotQA and 2Wiki only** (PL4's host rows for these two datasets are not repeated;
  PL5's rows run only the cells PL5 adds, and select reads PL4's grids as the incumbent): the host's G-R on MuSiQue (`pl5-musiquegr`, 2 cpus, 3 GB; it needs the MuSiQue grids of PL2, PL3 and PL4, pushed as inputs), G-R on HotpotQA (`pl5-hotpotqagr`, 2 cpus, 10 GB) and 2Wiki (`pl5-2wikigr`, 2 cpus, 11 GB), the dev grids in three
  parts (`pl5-hotpotqag0` to `g2`, 2 cpus, 17 GB each; `pl5-2wikig0` to `g2`, 2 cpus, 19 GB each), `pl5-hotpotqasel` and `pl5-2wikisel` (1 cpu, 3 GB), the frozen confirmations (`pl5-hotpotqacf` 2 cpus 4 GB, `pl5-2wikicf` 2 cpus 10 GB) and the e3v
  confirmations (`pl5-hotpotqace`, `pl5-2wikice`). The memory is PL4's measured peak for the same dataset (HotpotQA grid 14.95 GB, PL3's 2Wiki 17.1 GB) plus 1 to 2 GB of scratch for the wide window's score arrays (a frontier of 5,000 nodes of 4,096 entries is at most
  0.7 GB per array); the G-R rows and `sel`, `cf`, `ce` rows keep PL4's declared sizes. Expected cost: **HotpotQA's three grid parts about 5,350 s summed (PL4's 4,210 s x 1.27), 2Wiki's about 8,000 s summed (PL3's 4,819 s x 1.30, the PL4-to-PL3 ratio HotpotQA's rows measured, x 1.27; PL4's 2Wiki parts are not yet read)**;
  with three parts in parallel about 1,800 s and 2,700 s of wall time; the G-R rows take minutes. They depend on PL2's, PL3's and PL4's records, whose dev grids the union reads (PL4's host rows `pl4-hotpotqag*`, `pl4-2wikig*`, `pl4-*sel`).

## 12. What follows

1. The calls (section 6) over all seven; the results are filed here (section 14) with, per dataset, PL4's chosen cell against the new one, the frontier reading's five budgets, the price at any size, the smallest cell at 0.95 and 0.98 (dev and eval), what each option bought, what is
   still short, D1, D2 and D3.
2. **Next changes, names only** (none is adopted, none is in the grid): an order for the hop that sees the bridge (a parent-conditioned rank: a node's cosine plus its best parent's), which D3 puts at MetaQA's 264 hop-3 budget-cut golds; a hop-weighted schedule
   (a back-loaded share: PL4's `eq` is equal and bought nothing on MetaQA); a window wider than 4,096 if W = 4,096 pays on MuSiQue and WebQSP; the weight values of NER and KNN rows as a key; label-ordered caps once a UL2 layout exists on every dataset.
3. An e3v-bounded selection once e3v lists exist for the select carves; the same grids on the UL2 layout stores once S1 has materialized them.

## 13. Disclosed

- Written after reading PL4's filed records and page (sections 1 to 14.5), PL3's records, the walk code of `m3b_pools`, `zpl3g.py` and `zpl4g.py`, and the array names and shapes of the family stores; **D3 was computed before this commit on the labels of 300 dev questions of
  MuSiQue, MetaQA and WebQSP and is part of why the lever and the window are what they are**; no PL5 coverage number was computed before this commit (the cost timings printed seconds and nodes scored; the agreement checks printed counts).
- 6,078 cells are selected over on 1.5k-question dev carves (224 for MultiHop-RAG, about +/- 2 points): the winner's dev ALL is optimistic, more so than PL4's over 4,206. That is why the eval read decides and why both calls are paired intervals on eval against PL4's own chosen cell (or its
  matched-budget cell), not comparisons of dev numbers.
- The bound is not changed: the price-at-any-size readings are beside the calls, and a result that needs more than 2,500 nodes is filed as such, not carried.
- The budget-following walk lists and orders every node's first 512 (or 4,096) entries by cosine, so its cost is dominated by cosine scores (73,814 nodes scored per question in the timing); the cosine order inside a list is the only place the question's vector enters, as in PL3's `cs`.
  `fu` reads support and row rank through the same lists as PL4's `sp` and `rk`; the weight values and the typed relation labels are not read.
- The old call's price clause can credit a gain bought with fewer nodes at an unchanged ALL (PL4's MuSiQue ABOVE was of that kind); the frontier reading and the any-size price are filed beside it for that reason.
- The six datasets' graph is the old package graph (as PL2, PL3 and PL4 disclosed); MultiHop-RAG reads its B0 layout. The UL2 layout is not materialized on the six.
- Pre-commit runs on real data are the cost timings (MuSiQue, 40 dev questions), the agreement checks of the walks against a per-node loop (WebQSP and MuSiQue, 2 questions) and the self-test (MultiHop-RAG and SQuAD); G-R, which compares the engine to PL2's, PL3's and PL4's filed arrays and reads
  no PL5 number, runs after this commit, on every dataset that is read.

## 14. Results (26 Nov 2026; 7 of 7 datasets read)

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl5/<d>/{select,confirm_frozen,confirm_e3v,miss,place}.json`, `outputs/pl5/call.json`, `outputs/pl5/gr_*.json`) by
`outputs/mp_unified/zpl5g.py call` and `outputs/mp_unified/zpl5r.py report` / `miss`; the tables of section 14.10 are pasted unchanged from `outputs/pl5/report.md`. The laptop datasets (MultiHop-RAG, SQuAD,
MuSiQue, MetaQA, WebQSP) ran on the laptop (dev grid, select, frozen confirmation; SQuAD's frozen confirmation in four question slices through `zpl5r.py confirm-par`, as PL4's); HotpotQA and 2Wiki (their dev grids, frozen
and e3v confirmations) are host rows. The e3v confirmation ran for MultiHop-RAG, HotpotQA and 2Wiki only, as the declaration says (section 11); none was declared for SQuAD, MuSiQue, MetaQA or WebQSP. Nothing was changed in the rule, the cells or the engine after a number was read.

### 14.1 Gates and cost

The engine `zpl5g.py` is the declaration commit's (3168891e): script sha256 starts `c3e8221e4fb4`, the code-path hash `62a27dbb08c4` is equal in every select head and `call.json` (`code_path_equal_everywhere` true; G-U PASS), the cell list `a5a759803992` (1,872 cells; 6,078 with PL2's, PL3's and PL4's). **G-R** (PL2's 804, PL3's 1,470 and PL4's 1,932 cells rebuilt by this engine on the first 400 dev questions and PL4's on the first 150, 1,199,400 elements per dataset, 224 questions on MultiHop-RAG) PASSED with 0 present and 0 size mismatches on MultiHop-RAG, SQuAD, MuSiQue, MetaQA, WebQSP (laptop) and on MuSiQue, HotpotQA, 2Wiki (host rows: `gr_<d>_host.json`). The engine self-test PASSED (`selftest.json`: cell list, G-F, G-D, the engine against a brute-force pool builder, G-X5 the budget-following walk against a per-node loop). Every select re-read PL4's own pick from PL2's, PL3's and PL4's cells and got PL4's filed pick (`prev_pick_equals_pl4_filed` true on the 7 read). Cost of the dev grids (sum of the three parts): MultiHop-RAG 102 s, 0.28 GB peak (laptop); SQuAD 7,941 s, 0.12 GB peak (laptop); HotpotQA 4,376 s, 15.26 GB peak (host); 2Wiki 6,850 s, 17.48 GB peak (host); MuSiQue 9,740 s, 0.47 GB peak (laptop); MetaQA 1,982 s, 0.40 GB peak (laptop); WebQSP 9,847 s, 6.33 GB peak (laptop). Frozen confirmations: MultiHop-RAG 14 s, SQuAD 5,495 s, HotpotQA 9,385 s, 2Wiki 13,680 s, MuSiQue 5,923 s, MetaQA 2,136 s, WebQSP 7,967 s.

### 14.2 The calls (read 7 of 7: old rule CARRY_PL5; frontier reading CARRY_PL5_F)

**The old rule, as PL3 and PL4 state it** (unchanged; section 6): per dataset on eval, frozen lists, NEW = R95 of the reading `all` (all 6,078 cells), OLD = PL4's own chosen cell, paired bootstrap of the ALL difference and of the
ratio of mean nodes. Clause by clause:

| dataset | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | BELOW: (1) NEW misses where OLD reaches / (2) ALL diff upper end < 0 with ratio lower end >= 1 | ABOVE: (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (ALL not BELOW) / (3) ALL diff lower end > 0 with ratio upper end <= 1 | status | chosen cell inside 2,500 (eval nodes) |
|---|---|---|---|---|---|---|---|
| MultiHop-RAG | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (50.0) |
| SQuAD | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (15.0) |
| HotpotQA | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (77.9) |
| 2Wiki | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (917.0) |
| MuSiQue | no / no | +0.25 [-0.37, +0.83] | 1.0718 [1.0705, 1.0729] | no / no | no / no / no | **LEVEL** | yes (2,438.2) |
| MetaQA | no / no | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (2,335.9) |
| WebQSP | no / no | +9.85 [+8.18, +11.65] | 0.8486 [0.8288, 0.8702] | no / no | no / yes / yes | **ABOVE** | yes (2,094.6) |

- **BELOW: 0.** ABOVE: 1 (WebQSP); LEVEL: 6 of the 7 read.
- **The old call: CARRY_PL5** (BELOW 0, ABOVE 1, LEVEL 6; read 7 of 7). CARRY_PL5 needs all seven read, no BELOW, at least one ABOVE and every chosen cell inside the 2,500 bound.

- **What decides each status.** WebQSP is ABOVE through clauses 2 and 3: NEW (`dense_top200+STRUCT:h3_bf_v2000~cs-fu`, a budget-following walk of PL5's own cells) is +9.85 points [+8.18, +11.65] of ALL over PL4's pick and uses 0.85 times its nodes [0.83, 0.87]; neither reaches 0.95 on eval (0.8789 and 0.7804). MuSiQue is LEVEL: +0.25 points [-0.37, +0.83] with 1.07 times the nodes [1.0705, 1.0729], an interval that includes 0 bought with more nodes. On the other datasets read NEW is PL4's own cell (difference exactly 0, ratio exactly 1), SQuAD included (`rrf1000_b15`, the same cell on dev). **Every chosen cell is inside 2,500 on eval frozen lists** (the largest is MuSiQue's, 2,438.2); no clause fires BELOW. (all seven read: CARRY_PL5; frontier reading CARRY_PL5_F)


**The frontier reading** (section 6; filed beside the old one, never in its place): per budget N, the highest dev ALL among cells of at most N dev nodes, all cells against PL4's cells; paired
bootstrap of the eval ALL difference (points; `=` marks the same cell on both sides, difference exactly 0):

| dataset | 500 | 1,000 | 1,500 | 2,000 | 2,500 | frontier status |
|---|---|---|---|---|---|---|
| MultiHop-RAG | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| SQuAD | +0.00 [+0.00, +0.00] LEVEL = | +0.01 [+0.00, +0.03] LEVEL | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| HotpotQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | -0.07 [-0.24, +0.11] LEVEL | -0.14 [-0.30, +0.01] LEVEL | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| 2Wiki | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| MuSiQue | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +2.23 [+1.53, +2.98] ABOVE | +1.37 [+0.54, +2.23] ABOVE | +0.25 [-0.29, +0.83] LEVEL | **ABOVE** |
| MetaQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| WebQSP | +3.46 [+2.00, +4.99] ABOVE | +5.46 [+3.46, +7.58] ABOVE | +10.11 [+8.38, +12.04] ABOVE | +7.39 [+5.32, +9.38] ABOVE | +9.85 [+8.18, +11.78] ABOVE | **ABOVE** |

- **Frontier call: CARRY_PL5_F** (BELOW 0, ABOVE 2, LEVEL 5; read 7 of 7). The two ABOVE readings are MuSiQue (+2.23 points [+1.53, +2.98] at 1,500 nodes and +1.37 [+0.54, +2.23] at 2,000, LEVEL at 500, 1,000 and 2,500: +0.25 [-0.29, +0.83]) and WebQSP (ABOVE at all five budgets, +3.46 to +10.11). HotpotQA's two budgets with a different cell read -0.07 [-0.24, +0.11] and -0.14 [-0.30, +0.01]: LEVEL, both intervals include 0.
- **The two calls are filed under their own names and this page does not choose between them** (section 6): which one governs is the user's decision. The old rule is CARRY_PL5 and the frontier reading is CARRY_PL5_F (read 7 of 7).

The cells behind each frontier budget (dev ALL at dev nodes; eval ALL at eval nodes; the eval difference in points):

| dataset | budget | cells | dev | eval | difference |
|---|---|---|---|---|---|
| MultiHop-RAG | 500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 1000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 1500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 2000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 2500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| SQuAD | 500 | `rrf1000_b400+STRUCT:h1_c100~cs+top50` (both) | 0.9960 at 446.9 | 0.9965 at 446.7 | same cell |
| SQuAD | 1000 | NEW `rrf1000_b100+FULL:h3_bf_v5000_w4096~cs-vr+top800`; PL4 `rrf1000_b100+FULL:h3_c100_v5000~cs-vr-eq+top800` | 0.9980 at 900.0; 0.9980 at 900.0 | 0.9979 at 900.0; 0.9978 at 900.0 | +0.01 LEVEL |
| SQuAD | 1500 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` (both) | 0.9987 at 1,199.8 | 0.9982 at 1,200.0 | same cell |
| SQuAD | 2000 | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` (both) | 0.9993 at 1,600.0 | 0.9991 at 1,599.9 | same cell |
| SQuAD | 2500 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (both) | 1.0000 at 2,190.4 | 0.9992 at 2,200.0 | same cell |
| HotpotQA | 500 | `rrf1000_b100+FULL:h1_c100~s20+top800` (both) | 0.9887 at 460.5 | 0.9854 at 460.9 | same cell |
| HotpotQA | 1000 | `rrf1000_b800+FULL:h1_c100~cs+top400` (both) | 0.9927 at 932.5 | 0.9896 at 931.2 | same cell |
| HotpotQA | 1500 | NEW `rrf1000_b400+STRUCT:h3_bf_v1000~cs-fu`; PL4 `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.9940 at 1,276.9; 0.9934 at 1,111.7 | 0.9916 at 1,279.0; 0.9923 at 1,112.4 | -0.07 LEVEL |
| HotpotQA | 2000 | NEW `rrf1000_b800+STRUCT:h3_bf_v1000~cs-fu`; PL4 `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9954 at 1,605.4; 0.9940 at 1,880.9 | 0.9931 at 1,609.2; 0.9945 at 1,883.6 | -0.14 LEVEL |
| HotpotQA | 2500 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` (both) | 0.9960 at 2,246.8 | 0.9918 at 2,226.5 | same cell |
| 2Wiki | 500 | `rrf1000_b200+FULL:h1_c100+top400` (both) | 0.9325 at 419.2 | 0.9360 at 421.6 | same cell |
| 2Wiki | 1000 | `rrf1000_b400+FULL:h1_c100~s20` (both) | 0.9552 at 911.2 | 0.9614 at 920.4 | same cell |
| 2Wiki | 1500 | `rrf1000_b400+FULL:h1_c100~s20` (both) | 0.9552 at 911.2 | 0.9614 at 920.4 | same cell |
| 2Wiki | 2000 | `rrf1000_b400+FULL:h1_c100~s20` (both) | 0.9552 at 911.2 | 0.9614 at 920.4 | same cell |
| 2Wiki | 2500 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (both) | 0.9626 at 2,215.1 | 0.9656 at 2,211.2 | same cell |
| MuSiQue | 500 | `rrf1000_b400+FULL:h1_c100+brg100` (both) | 0.7373 at 498.2 | 0.7683 at 498.1 | same cell |
| MuSiQue | 1000 | `rrf1000_b800+FULL:h1_c100~s10+top200` (both) | 0.7947 at 988.6 | 0.8213 at 986.6 | same cell |
| MuSiQue | 1500 | NEW `rrf1000_b800+FULL:h2_bf_v1000~cs-vr-rr+top800`; PL4 `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.8455 at 1,494.0; 0.8246 at 1,388.8 | 0.8618 at 1,470.0; 0.8395 at 1,357.1 | +2.23 ABOVE |
| MuSiQue | 2000 | NEW `rrf1000_b400+FULL:h2_bf_v2000~cs-vr-rr+top1600`; PL4 `rrf1000_b400+FULL:h3_c25_v2000~cs-sp-eq+top1600` | 0.8677 at 1,999.7; 0.8559 at 2,000.0 | 0.8664 at 1,999.5; 0.8527 at 2,000.0 | +1.37 ABOVE |
| MuSiQue | 2500 | NEW `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr`; PL4 `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8898 at 2,464.3; 0.8781 at 2,298.7 | 0.8846 at 2,438.2; 0.8821 at 2,274.9 | +0.25 LEVEL |
| MetaQA | 500 | `rrf1000_b100+STRUCT:h2_c25+top400` (both) | 0.5638 at 480.9 | 0.5484 at 479.6 | same cell |
| MetaQA | 1000 | `equal_rrf_budget_200+STRUCT:h2_c25` (both) | 0.6927 at 764.8 | 0.6899 at 763.7 | same cell |
| MetaQA | 1500 | `rrf1000_b25+FS:h2_c100_v5000` (both) | 0.7308 at 1,473.1 | 0.7289 at 1,473.7 | same cell |
| MetaQA | 2000 | `rrf1000_b25+STRUCT:h3_c25_v2000` (both) | 0.8938 at 1,999.3 | 0.8949 at 1,995.3 | same cell |
| MetaQA | 2500 | `rrf1000_b400+STRUCT:h3_c25_v2000` (both) | 0.8978 at 2,339.7 | 0.8968 at 2,335.9 | same cell |
| WebQSP | 500 | NEW `dense_top200+FULL:h1_cW~s10-cs+top1600`; PL4 `dense_top200+FULL:h1_c100~s20+top400` | 0.7226 at 414.6; 0.6802 at 377.7 | 0.7319 at 425.7; 0.6973 at 384.7 | +3.46 ABOVE |
| WebQSP | 1000 | NEW `dense_top200+STRUCT:h3_bf_v1000~cs-fu+top800`; PL4 `rrf1000_b800+FULL:h1_c100~s20+top400` | 0.7857 at 1,000.0; 0.7147 at 940.5 | 0.7811 at 1,000.0; 0.7265 at 947.7 | +5.46 ABOVE |
| WebQSP | 1500 | NEW `dense_top200+STRUCT:h3_bf_v1000~cs-fu`; PL4 `rrf1000_b100+FS:h2_c100_v5000+top1600` | 0.8315 at 1,098.6; 0.7339 at 1,486.9 | 0.8310 at 1,099.0; 0.7299 at 1,486.4 | +10.11 ABOVE |
| WebQSP | 2000 | NEW `rrf1000_b25+FULL:h2_bf_v2000~s10-cs-vr`; PL4 `dense_top200+FS:h2_c100_v5000~cs-vr` | 0.8494 at 1,976.0; 0.7571 at 1,891.5 | 0.8443 at 1,971.1; 0.7705 at 1,966.1 | +7.39 ABOVE |
| WebQSP | 2500 | NEW `dense_top200+STRUCT:h3_bf_v2000~cs-fu`; PL4 `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.8759 at 2,094.2; 0.7731 at 2,442.4 | 0.8789 at 2,094.6; 0.7804 at 2,468.1 | +9.85 ABOVE |

**R98, beside the calls** (the same pairing at the 0.98 floor; decides nothing):

| dataset | NEW (R98 of all cells), eval | OLD (PL4's R98), eval | ALL difference, 95% CI | nodes ratio, 95% CI | status by the rule |
|---|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| SQuAD | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| HotpotQA | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| 2Wiki | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9656 at 2,211.2 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9656 at 2,211.2 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| MuSiQue | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8846 at 2,438.2 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8821 at 2,274.9 | +0.25 [-0.37, +0.83] | 1.0718 [1.0705, 1.0731] | LEVEL |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| WebQSP | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8789 at 2,094.6 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` 0.7804 at 2,468.1 | +9.85 [+7.98, +11.71] | 0.8486 [0.8283, 0.8683] | ABOVE |

### 14.3 What each dataset says (dev = selection carve, frozen lists; eval = confirmation)

The construction each dataset would get from R95 of the reading `all` (NEW), and PL4's (OLD), eval frozen ALL at mean nodes, and the e3v re-run of NEW (flagged OVER_BOUND above 2,500 nodes):

| dataset | NEW (eval frozen ALL at nodes) | OLD = PL4's (eval frozen ALL at nodes) | NEW meets 0.95 on eval, frozen | NEW on e3v (ALL at nodes) |
|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | yes | 0.9699 at 50.0 |
| SQuAD | `rrf1000_b15` 0.9554 at 15.0 | `rrf1000_b15` 0.9554 at 15.0 | yes | not read |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | yes | 0.9642 at 77.5 |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | yes | 0.9684 at 916.1 |
| MuSiQue | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8846 at 2,438.2 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8821 at 2,274.9 | no (short by 6.54 points) | not read |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | no (short by 5.32 points) | not read |
| WebQSP | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8789 at 2,094.6 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` 0.7804 at 2,468.1 | no (short by 7.11 points) | not read |

**Which datasets meet the target on eval, frozen lists (ALL >= 0.95), of the 7 read: 4**: MultiHop-RAG, SQuAD, HotpotQA, 2Wiki. **They do not:** MuSiQue (0.8846), MetaQA (0.8968), WebQSP (0.8789).
No chosen cell exceeds the bound on the e3v lists where those were read. The bound is read on the selection carve's frozen lists, as declared (section 6).

- **MultiHop-RAG** (224 dev questions; reaches 0.95 on dev.) R95 of all cells is PL4's own cell, `rrf1000_b25+FULL:h1_c100+top25` (dev 0.9777 at 50.0; eval frozen 0.9690 at 50.0); no PL5 cell is a smaller or higher-ALL R95 pick. The best of the PL5 cells alone is `rrf1000_b25+FULL:h1_cW~s10-cs+top50` (dev 0.9955 at 75.0). R98: `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 on eval (the same cell as PL4), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (the same cell at every budget). At any size, R95 is `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 nodes on eval (PL4 picks the same cell), status LEVEL.
- **SQuAD** (1,498 dev questions; reaches 0.95 on dev.) R95 of all cells is PL4's own cell, `rrf1000_b15` (dev 0.9519 at 15.0; eval frozen 0.9554 at 15.0); no PL5 cell is a smaller or higher-ALL R95 pick. The best of the PL5 cells alone is `rrf1000_b25+STRUCT:h1_cW~cs+top50` (dev 0.9713 at 73.7). R98: `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 on eval (the same cell as PL4), reaches 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (budgets with a different cell, eval ALL difference in points: 1000: +0.01). At any size, R95 is `rrf1000_b15` 0.9554 at 15.0 nodes on eval (PL4 picks the same cell), status LEVEL.
- **HotpotQA** (1,508 dev questions; reaches 0.95 on dev.) R95 of all cells is PL4's own cell, `rrf1000_b25+STRUCT:h1_c100~cs+top100` (dev 0.9622 at 77.3; eval frozen 0.9634 at 77.9); no PL5 cell is a smaller or higher-ALL R95 pick. The best of the PL5 cells alone is `rrf1000_b25+STRUCT:h1_cW~cs+top100` (dev 0.9602 at 77.3). R98: `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 on eval (the same cell as PL4), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (budgets with a different cell, eval ALL difference in points: 1500: -0.07; 2000: -0.14). At any size, R95 is `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 nodes on eval (PL4 picks the same cell), status LEVEL.
- **2Wiki** (1,496 dev questions; reaches 0.95 on dev.) R95 of all cells is PL4's own cell, `rrf1000_b400+FULL:h1_c100~s20+top800` (dev 0.9545 at 908.2; eval frozen 0.9609 at 917.0); no PL5 cell is a smaller or higher-ALL R95 pick. The best of the PL5 cells alone is `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` (dev 0.9525 at 1,269.5). R98: `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9656 at 2,211.2 on eval (the same cell as PL4), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (the same cell at every budget). At any size, R95 is `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 nodes on eval (PL4 picks the same cell), status LEVEL.
- **MuSiQue** (1,534 dev questions; SHORT on dev.) R95 of all cells: `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (dev 0.8898 at 2,464.3) against PL4's `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (dev 0.8781 at 2,298.7); on eval frozen 0.8846 at 2,438.2 against 0.8821 at 2,274.9: ALL +0.25 points [-0.37, +0.83], nodes ratio 1.0718 [1.0705, 1.0729], status **LEVEL**. R98: `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8846 at 2,438.2 on eval (PL4: `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8821 at 2,274.9), does not reach 0.98 on eval, status LEVEL. Frontier reading: **ABOVE** (budgets with a different cell, eval ALL difference in points: 1500: +2.23; 2000: +1.37; 2500: +0.25). At any size, R95 is `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 nodes on eval (PL4 picks the same cell; SHORT on dev), status LEVEL.
- **MetaQA** (1,497 dev questions; SHORT on dev.) R95 of all cells is PL4's own cell, `rrf1000_b400+STRUCT:h3_c25_v2000` (dev 0.8978 at 2,339.7; eval frozen 0.8968 at 2,335.9); no PL5 cell is a smaller or higher-ALL R95 pick. The best of the PL5 cells alone is `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (dev 0.7882 at 2,276.8). R98: `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 on eval (the same cell as PL4), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (the same cell at every budget). At any size, R95 is `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9582 at 4,972.8 nodes on eval (PL4 picks the same cell), status LEVEL.
- **WebQSP** (1,507 dev questions; SHORT on dev.) R95 of all cells: `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (dev 0.8759 at 2,094.2) against PL4's `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (dev 0.7731 at 2,442.4); on eval frozen 0.8789 at 2,094.6 against 0.7804 at 2,468.1: ALL +9.85 points [+8.18, +11.65], nodes ratio 0.8486 [0.8288, 0.8702], status **ABOVE**. R98: `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8789 at 2,094.6 on eval (PL4: `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` 0.7804 at 2,468.1), does not reach 0.98 on eval, status ABOVE. Frontier reading: **ABOVE** (budgets with a different cell, eval ALL difference in points: 500: +3.46; 1000: +5.46; 1500: +10.11; 2000: +7.39; 2500: +9.85). At any size, R95 is `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9175 at 6,079.5 nodes on eval (PL4: `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8390 at 6,029.5; SHORT on dev), status LEVEL.

### 14.4 What each option bought

Read from `outputs/pl5/report.md` ("What the PL5 cells bought" and "What each option bought", section 14.10). **The dev frontier of all cells differs from PL2 to PL4's on three datasets only.** WebQSP: +4.6 points at 200 nodes, +3.9 to +4.3 from 300 to 750, +7.1 at 1,000, then +9.8, +9.2 and +10.3 at 1,500, 2,000 and 2,500 nodes (0.7731 to 0.8759 at 2,500); above the bound +9.1, +8.0, +6.2 and +7.2 at 3,000 to 6,000. MuSiQue: +2.09 at 1,500, +1.17 at 2,000 and 2,500, +0.98 at 3,000, +0.52 at 4,000, +0.20 at 5,000 and nothing from 6,000. HotpotQA: +0.07 at 200, 300, 400 and 1,500 and +0.13 at 2,000, one or two questions of 1,508. MultiHop-RAG, SQuAD, 2Wiki and MetaQA: nothing at any of the 19 budgets. On eval the WebQSP gain reads +9.85 points [+8.18, +11.65] at 2,500 (ABOVE), MuSiQue's +2.23 and +1.37 at 1,500 and 2,000 (ABOVE) and +0.25 at 2,500 (LEVEL), HotpotQA's -0.07 and -0.14 (LEVEL).
**By option, on the three datasets whose PL4 cell is SHORT inside the bound** (R95 of the option's own cells, the highest ALL where none reaches; dev): every one of the seven options beats PL4's cell on WebQSP (0.7810 to 0.8759 against 0.7731; the budget-following cap with any key and the fused order give 0.8759 at 2,094.2, the cosine order 0.8666 at 2,499.5); four of seven beat it on MuSiQue (the budget-following cap with any key, the cosine order and the families-take-turns order 0.8898 at 2,464.3; seeds of 10 and 20 0.8827; PL4's is 0.8781; the fused order 0.8761 and the one-hop whole-window walks 0.8449 are lower, the 4,096-entry window ties); and none beats it on MetaQA (the best option cell is 0.7882 at 2,276.8, PL4's 0.8978). On the four datasets that reach 0.95 inside the bound no option makes a smaller cell reach it, so their picks stay PL4's.


### 14.5 The price of the floor at any size (beside the calls; decides nothing)

A dataset that stays SHORT inside the bound may not stay short without it; this table says what the 0.95 floor and the 0.98 floor cost when the bound is lifted, and whether the PL5 cells make that cheaper (PL6 reads the same question with a ladder of visited budgets). Eval, frozen lists;
NEW = the smallest cell (dev) at the floor over all cells, OLD = the same over PL2-PL4's cells; `SHORT on dev` marks a floor that no cell reaches on dev (then the cell is the highest-ALL cell).

| dataset | floor | NEW (R at any size, all cells), eval | OLD (PL2-PL4's cells), eval | ALL difference, 95% CI | nodes ratio, 95% CI | status |
|---|---|---|---|---|---|---|
| MultiHop-RAG | 0.95 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| MultiHop-RAG | 0.98 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| SQuAD | 0.95 | `rrf1000_b15` 0.9554 at 15.0 | `rrf1000_b15` 0.9554 at 15.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| SQuAD | 0.98 | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| HotpotQA | 0.95 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| HotpotQA | 0.98 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| 2Wiki | 0.95 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| 2Wiki | 0.98 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9785 at 6,275.0 (SHORT on dev) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9785 at 6,275.0 (SHORT on dev) | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| MuSiQue | 0.95 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 (SHORT on dev) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 (SHORT on dev) | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| MuSiQue | 0.98 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 (SHORT on dev) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9222 at 5,742.4 (SHORT on dev) | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| MetaQA | 0.95 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9582 at 4,972.8 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9582 at 4,972.8 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| MetaQA | 0.98 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9615 at 6,259.3 (SHORT on dev) | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9615 at 6,259.3 (SHORT on dev) | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL (the same cell) |
| WebQSP | 0.95 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9175 at 6,079.5 (SHORT on dev) | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8390 at 6,029.5 (SHORT on dev) | +7.85 [+6.39, +9.45] | 1.0083 [1.0065, 1.0102] | LEVEL |
| WebQSP | 0.98 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9175 at 6,079.5 (SHORT on dev) | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8390 at 6,029.5 (SHORT on dev) | +7.85 [+6.32, +9.25] | 1.0083 [1.0065, 1.0102] | LEVEL |

The any-size frontier (the highest dev ALL at most N dev nodes, N above the bound; eval ALL difference of all cells against PL2-PL4's, points):

| dataset | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 | any-size frontier status |
|---|---|---|---|---|---|---|
| MultiHop-RAG | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| SQuAD | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| HotpotQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | -0.05 [-0.18, +0.07] LEVEL | -0.05 [-0.18, +0.07] LEVEL | **LEVEL** |
| 2Wiki | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| MuSiQue | +0.17 [-0.50, +0.79] LEVEL | +0.62 [+0.04, +1.16] ABOVE | -0.12 [-0.99, +0.66] LEVEL | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| MetaQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| WebQSP | +8.45 [+6.85, +10.11] ABOVE | +6.52 [+4.66, +8.25] ABOVE | +6.99 [+5.26, +8.58] ABOVE | +7.85 [+6.25, +9.45] ABOVE | +7.85 [+6.39, +9.25] ABOVE | **ABOVE** |

**What the any-size reading says** (eval, frozen lists; section 14.5): the 0.95 floor's cell is PL4's own on six of seven datasets (difference exactly 0) and WebQSP's moves: the PL5 cells reach 0.9175 at 6,079.5 nodes where PL2 to PL4's reach 0.8390 at 6,029.5 (+7.85 points [+6.39, +9.45], nodes ratio 1.0083: LEVEL by the rule, ABOVE by the ALL interval), and neither reaches 0.95 on dev or eval. MetaQA reaches 0.95 only above the bound (0.9582 at 4,972.8 nodes, the same cell as PL4's); MuSiQue stays at 0.9222 at 5,742.4 (SHORT on dev, best dev 0.9355), 2Wiki's 0.98 floor is never reached (0.9785 at 6,275.0 on eval). PL6 (`docs/PL6_COVERAGE_ROUND5.md`) reads the same question with a ladder of visited budgets.


### 14.6 What is still short, in numbers (diagnostics D1 and D2, post-hoc, labels read; 300 dev questions evenly spaced over the carve)

Only the SHORT datasets are diagnosed (the chosen R95 cell of the reading `all`; budget-following walks are traced by `zpl5g.ref_walk5`, the stage "cap" of a `bf` walk being the scoring window). **D1** (each gold missing from the pool: R ranked past M, C cut by the caps, the window or the order but inside the uncapped three-hop FULL reach of its seeds, U beyond it):

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (bound 2,500) |
|---|---|---|---|---|---|---|---|
| MuSiQue | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 36 of 300 (0.120) | 39 (1.08 per such question) | 0 | 39 | 0 | 2,462.3 |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | 33 of 300 (0.110) | 375 (11.36 per such question) | 0 | 373 | 2 | 2,337.4 |
| WebQSP | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 31 of 300 (0.103) | 81 (2.61 per such question) | 0 | 81 | 0 | 2,093.5 |

**D2** (which stage cut each C gold, from a traced walk of the cell; the trace equals the engine's walk on every question):

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|
| MuSiQue | 39 | 0 | 18 | 0 | 9 | 0 | 12 | yes |
| MetaQA | 373 | 0 | 15 | 12 | 0 | 264 | 82 | yes |
| WebQSP | 81 | 2 | 35 | 5 | 19 | 16 | 4 | yes |

**Read** (the same 300 dev questions as PL4's D1 and D2, at PL5's chosen cells): in the three SHORT datasets no missing gold is ranked out (R 0) and only two lie beyond the uncapped three-hop reach (U: MetaQA's 2 of 375); every other missing gold is cut by a cap or a budget inside a reach the walk could take. **WebQSP**: 31 of 300 questions miss a gold, 81 golds (2.6 per such question; PL4's cell missed 245 C golds on the same questions); **42 are cut by a cap** (hop 1: 2, hop 2: 35, hop 3: 5), **35 by the visited budget** (hop 2: 19, hop 3: 16) and 4 are upstream. PL4's D2 had put 223 of its 245 at the per-seed and per-frontier caps and none at the budget: the budget-following cell has moved most of the cut from the caps to the budget and removed two thirds of the missing golds at 2,094 nodes. **MuSiQue**: 36 of 300 questions miss a gold, 39 golds (about one per such question; PL4's cell 37), 18 at the cap at hop 2, 9 at the budget at hop 2, 12 upstream: no one stage dominates, as at PL4's cell. **MetaQA** (unchanged cell): 33 questions, 375 golds (11.4 per such question), 264 of the 373 C golds cut by the budget at hop 3. Upstream means the gold is no neighbour of any node the walk kept, so every route to it passes through a node an earlier hop dropped (PL4 section 10).


### 14.7 Beside the calls: the e3v lists (post-hoc; decides nothing)

NEW (R95 of all cells) and OLD (PL4's) re-run on the eval carve's e3v lists. The mean nodes are flagged OVER_BOUND when above 2,500 (the bound is read on the frozen lists; the flag changes neither call):

| dataset | NEW, e3v (ALL at nodes) | OLD = PL4's, e3v (ALL at nodes) | ALL difference | NEW meets 0.95 |
|---|---|---|---|---|
| MultiHop-RAG | 0.9699 at 50.0 | 0.9699 at 50.0 | +0.00 | yes |
| SQuAD | not read | | | |
| HotpotQA | 0.9642 at 77.5 | 0.9642 at 77.5 | +0.00 | yes |
| 2Wiki | 0.9684 at 916.1 | 0.9684 at 916.1 | +0.00 | yes |
| MuSiQue | not read | | | |
| MetaQA | not read | | | |
| WebQSP | not read | | | |

### 14.8 What the calls mean, and what they leave unread

- **At 7 of 7 read the old rule is CARRY_PL5 and the frontier reading is CARRY_PL5_F**, so the two agree: neither finds a BELOW; the old rule's call rests on one ABOVE (WebQSP), the frontier reading's on two (MuSiQue at 1,500 and 2,000 nodes, WebQSP at all five budgets). The page still does not choose between them (section 6): which governs is the user's decision.
- **A reading of what the datasets say.** On five of the seven the PL5 pick is PL4's (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki and MetaQA: the same cell, difference exactly 0). MuSiQue's moves by +0.25 points with 7% more nodes (LEVEL), and **WebQSP's moves by +9.85 points with 15% fewer nodes**, from a budget-following walk (PL5's own cells, `bf`): the one change of this round that the old rule credits. It is also the change D2 pointed at (PL4's D2 put 223 of WebQSP's 245 attributed C golds at the per-seed and per-frontier caps). **Eval, frozen lists: four datasets meet 0.95 inside the bound (MultiHop-RAG 0.9690, SQuAD 0.9554, HotpotQA 0.9634, 2Wiki 0.9609); three do not (MuSiQue 0.8846, MetaQA 0.8968, WebQSP 0.8789), short by 6.54, 5.32 and 7.11 points.**
- **What it leaves unread.** (a) The e3v lists were read for MultiHop-RAG, HotpotQA and 2Wiki only (none declared for the other four); no chosen cell exceeds the bound there. (b) The three SHORT datasets are short *inside* 2,500 nodes: PL6 reads what lifting the bound costs, and PL7 what restricting back toward 2,500 costs. (c) Every number here is on the current links; the link-tightening stage and one pool re-run are declared after the link set is final.


### 14.9 The next changes (names only; none is adopted, none is in this round's grid; each is declared on its own page before any number)

From this page's section 12, unchanged by the results: a parent-conditioned rank (a node's cosine plus its best parent's), aimed at the hop-3 budget cut of MetaQA and MuSiQue; a hop-weighted schedule; a window wider than 4,096 if it paid on MuSiQue and WebQSP; the weight values of NER and KNN rows as a key; label-ordered caps once a UL2 layout exists on every dataset. **Declared separately, before any number: PL6 (`docs/PL6_COVERAGE_ROUND5.md`, 744811e6)** reads the user's ruling of 25 Nov ("i do not mind just rules should be universal") as one selection rule on all seven with no node bound and a ladder of visited budgets (10,000 to 80,000) for the datasets that stay SHORT; it reads this round's selects as its rung 0. The user's later ruling (relayed, 25 Nov) made that an analysis ("by lifting that bound i asked you to do analysis, we still need to start restricting it and bring towards 2.5k"); **PL7 (`docs/PL7_COVERAGE_RESTRICTION.md`, 24f4892e)**, declared before any number, restricts the pool back toward 2,500 nodes under one rule on all seven, on the current links (provisional until the link set is final).


### 14.10 The tables (the report, as of 7 of 7 read)

#### Gates and cost

| dataset | G-R (PL2's, PL3's and PL4's cells rebuilt: first 400 dev questions, PL4's 150) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q, 799,176 elements, mismatches 0/0) | 3, 102 s, 0.28 GB |
| squad | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 7,941 s, 0.12 GB |
| hotpotqa | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 4,376 s, 15.26 GB |
| 2wiki | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 6,850 s, 17.48 GB |
| musique | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 9,740 s, 0.47 GB |
| metaqa | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 1,982 s, 0.4 GB |
| webqsp | PASS (400 q, 1,199,400 elements, mismatches 0/0) | 3, 9,847 s, 6.33 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 552, squad PASS 552; G-X5 PASS, PASS); code-path sha256 `62a27dbb...`.

#### Dev reading (the rule's own numbers, dev carve, frozen lists): PL4's chosen cell (the incumbent, reading `prev`) against the grid of all 6,078 cells

| dataset | dev q | PL4 R95 (prev) | R95 (all cells) | PL4 R98 | R98 (all cells) | R95 of the PL5 cells only | prev pick re-read = PL4's filed pick |
|---|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` (0.9955 at 75.0) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b25+STRUCT:h1_cW~cs+top50` (0.9713 at 73.7) | True |
| hotpotqa | 1508 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b25+STRUCT:h1_cW~cs+top100` (0.9602 at 77.3) | True |
| 2wiki | 1496 | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (0.9626 at 2,215.1, SHORT) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (0.9626 at 2,215.1, SHORT) | `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` (0.9525 at 1,269.5) | True |
| musique | 1534 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.8898 at 2,464.3, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.8898 at 2,464.3, SHORT) | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (0.8898 at 2,464.3, SHORT) | True |
| metaqa | 1497 | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (0.7882 at 2,276.8, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (0.8759 at 2,094.2, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (0.8759 at 2,094.2, SHORT) | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (0.8759 at 2,094.2, SHORT) | True |

#### The chosen cell at 0.95 (R95) and its confirmation on eval: PL4's against the grid of all cells (and the PL5 cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL4 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag all | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag PL5 only | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` | 0.9955 | 75.0 | 0.9885 (75.0) | 0.9823 (75.0) |
| squad PL4 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| squad all | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | not read |
| squad PL5 only | `rrf1000_b25+STRUCT:h1_cW~cs+top50` | 0.9713 | 73.7 | 0.9721 (73.2) | not read |
| hotpotqa PL4 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa all | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa PL5 only | `rrf1000_b25+STRUCT:h1_cW~cs+top100` | 0.9602 | 77.3 | 0.9616 (77.9) | 0.9633 (77.6) |
| 2wiki PL4 | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki all | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki PL5 only | `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` | 0.9525 | 1,269.5 | 0.9519 (1,265.2) | 0.9719 (1,280.4) |
| musique PL4 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | not read |
| musique all | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| musique PL5 only | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| metaqa PL4 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa PL5 only | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.7882 | 2,276.8 | 0.7785 (2,277.0) SHORT | not read |
| webqsp PL4 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |
| webqsp all | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |
| webqsp PL5 only | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |

#### The chosen cell at 0.98 (R98) and its confirmation on eval: PL4's against the grid of all cells (and the PL5 cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL4 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag all | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag PL5 only | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` | 0.9955 | 75.0 | 0.9885 (75.0) | 0.9823 (75.0) |
| squad PL4 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| squad all | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | not read |
| squad PL5 only | `rrf1000_b25+FULL:h3_bf_v2000~s10-cs-vr+top50` | 0.9846 | 75.0 | 0.9816 (75.0) | not read |
| hotpotqa PL4 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa all | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa PL5 only | `rrf1000_b100+STRUCT:h1_cW~cs+top400` | 0.9801 | 162.2 | 0.9783 (166.4) SHORT | 0.9754 (167.3) SHORT |
| 2wiki PL4 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| 2wiki all | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| 2wiki PL5 only | `rrf1000_b1600+FULL:h2_bf_v1000~cs-fu` (SHORT) | 0.9572 | 2,412.8 | 0.9568 (2,405.5) SHORT | 0.9758 (2,435.7) SHORT |
| musique PL4 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | not read |
| musique all | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| musique PL5 only | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` (SHORT) | 0.8898 | 2,464.3 | 0.8846 (2,438.2) SHORT | not read |
| metaqa PL4 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | not read |
| metaqa PL5 only | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.7882 | 2,276.8 | 0.7785 (2,277.0) SHORT | not read |
| webqsp PL4 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | not read |
| webqsp all | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |
| webqsp PL5 only | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` (SHORT) | 0.8759 | 2,094.2 | 0.8789 (2,094.6) SHORT | not read |

#### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | not read |
| hotpotqa | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9684 (93.7) | 0.9646 (92.2) |
| 2wiki | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9058 (105.6) | 0.9310 (101.6) |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | not read |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | not read |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | not read |

#### The price of the floor at any size, dev (the bound is not applied): the smallest cell with ALL >= 0.95 and >= 0.98, PL4's grid / all cells

| dataset | 0.95: PL4's grid | 0.95: all cells | 0.98: PL4's grid | 0.98: all cells |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7) |
| musique | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) | none (best `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4) |
| metaqa | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2) | none (best `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2) |
| webqsp | none (best `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1) | none (best `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9257 at 6,070.5) | none (best `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1) | none (best `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9257 at 6,070.5) |

#### The highest ALL on dev (the bound is 2,500 mean nodes): PL4's grid / all cells, within the bound and at any size

| dataset | PL4's grid, within the bound | all cells, within the bound | PL4's grid, any size | all cells, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| hotpotqa | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 | `rrf1000_b400+FULL:h3_bf_v5000~cs-fu` 0.9980 at 5,173.9 |
| 2wiki | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9626 at 2,215.1 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9626 at 2,215.1 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7 |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8781 at 2,298.7 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` 0.8898 at 2,464.3 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` 0.8759 at 2,094.2 | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` 0.9257 at 6,070.5 |

#### What the PL5 cells bought (dev frontier of all cells minus the frontier of PL2-PL4's cells, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 | 3000 | 4000 | 5000 | 6000 | 7000 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9853 / 0.9853 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9927 / 0.9927 (+0.00) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9960 / 0.9960 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9980 / 0.9980 (+0.00) | 0.9987 / 0.9987 (+0.00) | 0.9993 / 0.9993 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| hotpotqa | 0.6127 / 0.6127 (+0.00) | 0.9224 / 0.9224 (+0.00) | 0.9496 / 0.9496 (+0.00) | 0.9682 / 0.9682 (+0.00) | 0.9788 / 0.9788 (+0.00) | 0.9814 / 0.9821 (+0.07) | 0.9834 / 0.9841 (+0.07) | 0.9834 / 0.9841 (+0.07) | 0.9887 / 0.9887 (+0.00) | 0.9901 / 0.9901 (+0.00) | 0.9927 / 0.9927 (+0.00) | 0.9934 / 0.9940 (+0.07) | 0.9940 / 0.9954 (+0.13) | 0.9960 / 0.9960 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9980 / 0.9980 (+0.00) | 0.9980 / 0.9980 (+0.00) |
| 2wiki | 0.3195 / 0.3195 (+0.00) | 0.7172 / 0.7172 (+0.00) | 0.8342 / 0.8342 (+0.00) | 0.8890 / 0.8890 (+0.00) | 0.8984 / 0.8984 (+0.00) | 0.9078 / 0.9078 (+0.00) | 0.9164 / 0.9164 (+0.00) | 0.9298 / 0.9298 (+0.00) | 0.9325 / 0.9325 (+0.00) | 0.9452 / 0.9452 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9626 / 0.9626 (+0.00) | 0.9626 / 0.9626 (+0.00) | 0.9646 / 0.9646 (+0.00) | 0.9646 / 0.9646 (+0.00) | 0.9766 / 0.9766 (+0.00) | 0.9786 / 0.9786 (+0.00) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6349 / 0.6349 (+0.00) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7373 / 0.7373 (+0.00) | 0.7673 / 0.7673 (+0.00) | 0.7947 / 0.7947 (+0.00) | 0.8246 / 0.8455 (+2.09) | 0.8559 / 0.8677 (+1.17) | 0.8781 / 0.8898 (+1.17) | 0.8846 / 0.8944 (+0.98) | 0.8957 / 0.9009 (+0.52) | 0.8990 / 0.9009 (+0.20) | 0.9355 / 0.9355 (+0.00) | 0.9355 / 0.9355 (+0.00) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7308 / 0.7308 (+0.00) | 0.8938 / 0.8938 (+0.00) | 0.8978 / 0.8978 (+0.00) | 0.8998 / 0.8998 (+0.00) | 0.9272 / 0.9272 (+0.00) | 0.9613 / 0.9613 (+0.00) | 0.9646 / 0.9646 (+0.00) | 0.9653 / 0.9653 (+0.00) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6058 / 0.6516 (+4.58) | 0.6603 / 0.6994 (+3.92) | 0.6802 / 0.7213 (+4.11) | 0.6802 / 0.7226 (+4.25) | 0.6855 / 0.7273 (+4.18) | 0.7147 / 0.7857 (+7.10) | 0.7339 / 0.8315 (+9.75) | 0.7571 / 0.8494 (+9.22) | 0.7731 / 0.8759 (+10.29) | 0.7883 / 0.8792 (+9.09) | 0.8056 / 0.8852 (+7.96) | 0.8401 / 0.9018 (+6.17) | 0.8507 / 0.9224 (+7.17) | 0.8540 / 0.9257 (+7.17) |

(each entry: the frontier of PL2-PL4's cells / the frontier of all cells (points bought); `-` = no cell is that small; 3,000 and above are beside the bound)

#### What each option bought: R95 of the option's own cells (dev, inside the bound) beside PL4's R95, and its best ALL inside the bound

| dataset | PL4 R95 | budget-following cap (any key) | cosine order (bf) | fused order (bf) | families take turns (bf) | seeds 10/20 | one-hop whole-window walks | 4,096-entry window |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | 0.9911 at 75.0 | 0.9911 at 75.0 | 0.9866 at 75.0 | 0.9866 at 75.0 | 0.9955 at 75.0 | 0.9955 at 75.0 | 0.9866 at 75.0 |
| squad | 0.9519 at 15.0 | 0.9800 at 74.5 | 0.9800 at 74.5 | 0.9793 at 74.5 | 0.9840 at 75.0 | 0.9713 at 74.4 | 0.9713 at 73.7 | 0.9800 at 74.5 |
| hotpotqa | 0.9622 at 77.3 | 0.9595 at 824.5 | 0.9602 at 899.4 | 0.9595 at 824.5 | 0.9602 at 899.4 | 0.9615 at 83.7 | 0.9602 at 77.3 | 0.9735 at 1,990.0 |
| 2wiki | 0.9545 at 908.2 | 0.9525 at 1,269.5 | 0.9512 at 2,314.7 | 0.9525 at 1,269.5 | 0.9539 at 2,315.2 | 0.9392 at 2,306.9 SHORT | 0.9432 at 1,804.3 SHORT | 0.9485 at 2,291.9 SHORT |
| musique | 0.8781 at 2,298.7 SHORT | 0.8898 at 2,464.3 SHORT | 0.8898 at 2,464.3 SHORT | 0.8761 at 2,476.5 SHORT | 0.8898 at 2,464.3 SHORT | 0.8827 at 2,439.4 SHORT | 0.8449 at 2,032.2 SHORT | 0.8781 at 2,372.9 SHORT |
| metaqa | 0.8978 at 2,339.7 SHORT | 0.7882 at 2,276.8 SHORT | 0.6152 at 2,491.0 SHORT | 0.7882 at 2,276.8 SHORT | 0.5845 at 2,237.5 SHORT | 0.6152 at 2,491.0 SHORT | 0.4549 at 1,656.5 SHORT | 0.5157 at 2,219.2 SHORT |
| webqsp | 0.7731 at 2,442.4 SHORT | 0.8759 at 2,094.2 SHORT | 0.8666 at 2,499.5 SHORT | 0.8759 at 2,094.2 SHORT | 0.8427 at 2,064.0 SHORT | 0.8666 at 2,499.5 SHORT | 0.7810 at 1,709.5 SHORT | 0.8374 at 2,459.8 SHORT |

#### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

#### The call, the old rule as PL3 and PL4 state it (R95 of all cells against R95 of PL2-PL4's cells): CARRY_PL5 (read 7 of 7: 2wiki, hotpotqa, metaqa, multihoprag, musique, squad, webqsp; BELOW 0, ABOVE 1, LEVEL 6)

| dataset | status | NEW (R95, all cells) | NEW ALL (nodes) | OLD (PL4 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | LEVEL (same cell) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | LEVEL (same cell) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | LEVEL | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 0.8846 (2,438.2) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 (2,274.9) | +0.25 [-0.37, +0.83] | 1.072 [1.071, 1.073] |
| metaqa | LEVEL (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| webqsp | ABOVE | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 0.8789 (2,094.6) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 (2,468.1) | +9.85 [+8.18, +11.65] | 0.849 [0.829, 0.870] |

Beside the call (decide nothing): R98 and the PL5-cells-only pick against PL4's R95 / R98.

| dataset | R98 status | R98 NEW ALL (nodes) vs OLD | PL5-only status | PL5-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | LEVEL | 0.9788 (58.0) vs 0.9788 (58.0) | LEVEL | `rrf1000_b25+FULL:h1_cW~s10-cs+top50` | 0.9885 (75.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h1_cW~cs+top50` | 0.9721 (73.2) |
| hotpotqa | LEVEL | 0.9783 (153.5) vs 0.9783 (153.5) | BELOW | `rrf1000_b25+STRUCT:h1_cW~cs+top100` | 0.9616 (77.9) |
| 2wiki | LEVEL | 0.9656 (2,211.2) vs 0.9656 (2,211.2) | BELOW | `rrf1000_b400+FULL:h3_bf_v1000~cs-fu` | 0.9519 (1,265.2) |
| musique | LEVEL | 0.8846 (2,438.2) vs 0.8821 (2,274.9) | LEVEL | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 0.8846 (2,438.2) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8968 (2,335.9) | ABOVE | `rrf1000_b400+STRUCT:h3_bf_v2000~cs-fu` | 0.7785 (2,277.0) |
| webqsp | ABOVE | 0.8789 (2,094.6) vs 0.7804 (2,468.1) | ABOVE | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 0.8789 (2,094.6) |

#### The call, the frontier reading (matched node counts inside the bound, declared before any number): CARRY_PL5_F (BELOW 0, ABOVE 2, LEVEL 5)

Per dataset and budget N: the cell of all cells with the highest dev ALL at most N dev nodes against the same for PL2-PL4's cells; eval, frozen lists; paired bootstrap of the ALL difference.

| dataset | frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL | 500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 500 | `rrf1000_b400+STRUCT:h1_c100~cs+top50` (same cell) | `rrf1000_b400+STRUCT:h1_c100~cs+top50` | 0.9965 (446.7) | 0.9965 (446.7) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 1000 | `rrf1000_b100+FULL:h3_bf_v5000_w4096~cs-vr+top800` | `rrf1000_b100+FULL:h3_c100_v5000~cs-vr-eq+top800` | 0.9979 (900.0) | 0.9978 (900.0) | +0.01 [+0.00, +0.03] | LEVEL |
| squad | LEVEL | 1500 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` (same cell) | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` | 0.9982 (1,200.0) | 0.9982 (1,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 2000 | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` (same cell) | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` | 0.9991 (1,599.9) | 0.9991 (1,599.9) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 2500 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 500 | `rrf1000_b100+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b100+FULL:h1_c100~s20+top800` | 0.9854 (460.9) | 0.9854 (460.9) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~cs+top400` (same cell) | `rrf1000_b800+FULL:h1_c100~cs+top400` | 0.9896 (931.2) | 0.9896 (931.2) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1500 | `rrf1000_b400+STRUCT:h3_bf_v1000~cs-fu` | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.9916 (1,279.0) | 0.9923 (1,112.4) | -0.07 [-0.24, +0.11] | LEVEL |
| hotpotqa | LEVEL | 2000 | `rrf1000_b800+STRUCT:h3_bf_v1000~cs-fu` | `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9931 (1,609.2) | 0.9945 (1,883.6) | -0.14 [-0.30, +0.01] | LEVEL |
| hotpotqa | LEVEL | 2500 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` | 0.9918 (2,226.5) | 0.9918 (2,226.5) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 500 | `rrf1000_b200+FULL:h1_c100+top400` (same cell) | `rrf1000_b200+FULL:h1_c100+top400` | 0.9360 (421.6) | 0.9360 (421.6) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1500 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2500 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9656 (2,211.2) | 0.9656 (2,211.2) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | ABOVE | 500 | `rrf1000_b400+FULL:h1_c100+brg100` (same cell) | `rrf1000_b400+FULL:h1_c100+brg100` | 0.7683 (498.1) | 0.7683 (498.1) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | ABOVE | 1000 | `rrf1000_b800+FULL:h1_c100~s10+top200` (same cell) | `rrf1000_b800+FULL:h1_c100~s10+top200` | 0.8213 (986.6) | 0.8213 (986.6) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | ABOVE | 1500 | `rrf1000_b800+FULL:h2_bf_v1000~cs-vr-rr+top800` | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.8618 (1,470.0) | 0.8395 (1,357.1) | +2.23 [+1.53, +2.98] | ABOVE |
| musique | ABOVE | 2000 | `rrf1000_b400+FULL:h2_bf_v2000~cs-vr-rr+top1600` | `rrf1000_b400+FULL:h3_c25_v2000~cs-sp-eq+top1600` | 0.8664 (1,999.5) | 0.8527 (2,000.0) | +1.37 [+0.54, +2.23] | ABOVE |
| musique | ABOVE | 2500 | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8846 (2,438.2) | 0.8821 (2,274.9) | +0.25 [-0.29, +0.83] | LEVEL |
| metaqa | LEVEL | 500 | `rrf1000_b100+STRUCT:h2_c25+top400` (same cell) | `rrf1000_b100+STRUCT:h2_c25+top400` | 0.5484 (479.6) | 0.5484 (479.6) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1000 | `equal_rrf_budget_200+STRUCT:h2_c25` (same cell) | `equal_rrf_budget_200+STRUCT:h2_c25` | 0.6899 (763.7) | 0.6899 (763.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1500 | `rrf1000_b25+FS:h2_c100_v5000` (same cell) | `rrf1000_b25+FS:h2_c100_v5000` | 0.7289 (1,473.7) | 0.7289 (1,473.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2000 | `rrf1000_b25+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b25+STRUCT:h3_c25_v2000` | 0.8949 (1,995.3) | 0.8949 (1,995.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2500 | `rrf1000_b400+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | ABOVE | 500 | `dense_top200+FULL:h1_cW~s10-cs+top1600` | `dense_top200+FULL:h1_c100~s20+top400` | 0.7319 (425.7) | 0.6973 (384.7) | +3.46 [+2.00, +4.99] | ABOVE |
| webqsp | ABOVE | 1000 | `dense_top200+STRUCT:h3_bf_v1000~cs-fu+top800` | `rrf1000_b800+FULL:h1_c100~s20+top400` | 0.7811 (1,000.0) | 0.7265 (947.7) | +5.46 [+3.46, +7.58] | ABOVE |
| webqsp | ABOVE | 1500 | `dense_top200+STRUCT:h3_bf_v1000~cs-fu` | `rrf1000_b100+FS:h2_c100_v5000+top1600` | 0.8310 (1,099.0) | 0.7299 (1,486.4) | +10.11 [+8.38, +12.04] | ABOVE |
| webqsp | ABOVE | 2000 | `rrf1000_b25+FULL:h2_bf_v2000~s10-cs-vr` | `dense_top200+FS:h2_c100_v5000~cs-vr` | 0.8443 (1,971.1) | 0.7705 (1,966.1) | +7.39 [+5.32, +9.38] | ABOVE |
| webqsp | ABOVE | 2500 | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.8789 (2,094.6) | 0.7804 (2,468.1) | +9.85 [+8.18, +11.78] | ABOVE |

#### Beside the calls: the price of the floor at any size (eval; the bound does not apply; decides nothing). Counts for R95any: {'BELOW': 0, 'ABOVE': 0, 'LEVEL': 7}; the any-size frontier (3,000-7,000): {'BELOW': 0, 'ABOVE': 1, 'LEVEL': 6}

| dataset | floor | status | NEW cell | NEW ALL (nodes) | OLD cell | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.95 | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| multihoprag | 0.98 | LEVEL (same cell) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9788 (58.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | 0.95 | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | 0.98 | LEVEL (same cell) | `rrf1000_b50+KNN:h1_c100+top25` | 0.9818 (63.0) | `rrf1000_b50+KNN:h1_c100+top25` | 0.9818 (63.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | 0.95 | LEVEL (same cell) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | 0.98 | LEVEL (same cell) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9783 (153.5) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9783 (153.5) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | 0.95 | LEVEL (same cell) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | 0.98 | LEVEL (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` | 0.9785 (6,275.0) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` | 0.9785 (6,275.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | 0.95 | LEVEL (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | 0.98 | LEVEL (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| metaqa | 0.95 | LEVEL (same cell) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 (4,972.8) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 (4,972.8) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| metaqa | 0.98 | LEVEL (same cell) | `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 (6,259.3) | `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 (6,259.3) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| webqsp | 0.95 | LEVEL | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9175 (6,079.5) | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` | 0.8390 (6,029.5) | +7.85 [+6.39, +9.45] | 1.008 [1.007, 1.010] |
| webqsp | 0.98 | LEVEL | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | 0.9175 (6,079.5) | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` | 0.8390 (6,029.5) | +7.85 [+6.32, +9.25] | 1.008 [1.007, 1.010] |

| dataset | any-size frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL | 3000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 4000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 5000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 6000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 7000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 3000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 4000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 5000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 6000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 7000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 3000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` | 0.9937 (2,963.6) | 0.9937 (2,963.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 4000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` | 0.9937 (2,963.6) | 0.9937 (2,963.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 5000 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr` | 0.9937 (2,963.6) | 0.9937 (2,963.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 6000 | `rrf1000_b400+FULL:h3_bf_v5000~cs-fu` | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` | 0.9946 (5,178.2) | 0.9951 (5,454.6) | -0.05 [-0.18, +0.07] | LEVEL |
| hotpotqa | LEVEL | 7000 | `rrf1000_b400+FULL:h3_bf_v5000~cs-fu` | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` | 0.9946 (5,178.2) | 0.9951 (5,454.6) | -0.05 [-0.18, +0.07] | LEVEL |
| 2wiki | LEVEL | 3000 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9656 (2,211.2) | 0.9656 (2,211.2) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 4000 | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9675 (3,055.3) | 0.9675 (3,055.3) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 5000 | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` (same cell) | `rrf1000_b1600+FULL:h3_c25_v2000~s20-cs-vr-eq` | 0.9675 (3,055.3) | 0.9675 (3,055.3) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 6000 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-rk` (same cell) | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-rk` | 0.9765 (5,264.5) | 0.9765 (5,264.5) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 7000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` | 0.9785 (6,275.0) | 0.9785 (6,275.0) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 3000 | `rrf1000_b1600+FULL:h2_bf_v2000~s10-cs-vr` | `rrf1000_b1600+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8904 (2,950.2) | 0.8887 (2,703.3) | +0.17 [-0.50, +0.79] | LEVEL |
| musique | LEVEL | 4000 | `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` | `rrf1000_b1600+FULL:h3_c25_v2000~cs-sp` | 0.8945 (2,980.2) | 0.8883 (3,103.9) | +0.62 [+0.04, +1.16] | ABOVE |
| musique | LEVEL | 5000 | `rrf1000_b1600+FULL:h2_bf_v2000~cs-vr-rr+top1600` | `rrf1000_b400+FULL:h2_c100_v5000~cs-vr-rr` | 0.8945 (2,980.2) | 0.8957 (4,956.7) | -0.12 [-0.99, +0.66] | LEVEL |
| musique | LEVEL | 6000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 7000 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` (same cell) | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` | 0.9222 (5,742.4) | 0.9222 (5,742.4) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 3000 | `rrf1000_b800+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b800+STRUCT:h3_c25_v2000` | 0.8999 (2,699.3) | 0.8999 (2,699.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 4000 | `dense_top200+STRUCT:h3_c25_v4000` (same cell) | `dense_top200+STRUCT:h3_c25_v4000` | 0.9244 (3,600.3) | 0.9244 (3,600.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 5000 | `rrf1000_b25+STRUCT:h3_c100_v5000` (same cell) | `rrf1000_b25+STRUCT:h3_c100_v5000` | 0.9582 (4,972.8) | 0.9582 (4,972.8) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 6000 | `rrf1000_b800+STRUCT:h3_c100_v5000` (same cell) | `rrf1000_b800+STRUCT:h3_c100_v5000` | 0.9599 (5,601.9) | 0.9599 (5,601.9) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 7000 | `rrf1000_b1600+STRUCT:h3_c100_v5000` (same cell) | `rrf1000_b1600+STRUCT:h3_c100_v5000` | 0.9615 (6,259.3) | 0.9615 (6,259.3) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | ABOVE | 3000 | `rrf1000_b800+STRUCT:h3_bf_v2000~cs-fu` | `rrf1000_b800+FS:h2_c100_v5000` | 0.8802 (2,520.2) | 0.7957 (2,948.2) | +8.45 [+6.85, +10.11] | ABOVE |
| webqsp | ABOVE | 4000 | `rrf1000_b1600+STRUCT:h2_bf_v2000~cs-fu` | `rrf1000_b800+FULL:h2_c100_v5000~s20-cs-sp` | 0.8736 (3,247.0) | 0.8084 (3,956.0) | +6.52 [+4.66, +8.25] | ABOVE |
| webqsp | ABOVE | 5000 | `rrf1000_b800+FULL:h2_bf_v5000~cs-fu` | `rrf1000_b25+FULL:h3_c100_v5000~cs-sp-eq` | 0.8975 (4,976.6) | 0.8277 (4,986.8) | +6.99 [+5.26, +8.58] | ABOVE |
| webqsp | ABOVE | 6000 | `rrf1000_b800+STRUCT:h3_bf_v5000~cs-fu` | `rrf1000_b800+FULL:h3_c100_v5000~cs-sp-eq` | 0.9168 (5,488.5) | 0.8383 (5,433.3) | +7.85 [+6.25, +9.45] | ABOVE |
| webqsp | ABOVE | 7000 | `rrf1000_b1600+STRUCT:h3_bf_v5000~cs-fu` | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` | 0.9175 (6,079.5) | 0.8390 (6,029.5) | +7.85 [+6.39, +9.25] | ABOVE |

#### The 2,500 bound on e3v lists (selection reads the frozen dev lists; this is the same cell re-run on the eval carve's e3v lists)

| dataset | cell (R95, all cells) | nodes: dev frozen | eval frozen | eval e3v | e3v over the bound |
|---|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 50.0 | 50.0 | 50.0 | within |
| squad | `rrf1000_b15` | 15.0 | 15.0 | not read | - |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 77.3 | 77.9 | 77.5 | within |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | 908.2 | 917.0 | 916.1 | within |
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 2,464.3 | 2,438.2 | not read | - |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 2,339.7 | 2,335.9 | not read | - |
| webqsp | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 2,094.2 | 2,094.6 | not read | - |

#### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps or the budget) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 300 | 36 (0.120) | 39 | 1.08 | 0 (q 0) | 39 (q 36) | 0 (q 0) | 0 | 2,462.3 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 300 | 31 (0.103) | 81 | 2.61 | 0 (q 0) | 81 (q 31) | 0 (q 0) | 0 | 2,093.5 |

#### D2: which stage cut the C golds (post-hoc; labels read; the cell's walk traced hop by hop)

| dataset | cell | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 1 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | 39 | 0 | 18 | 0 | 0 | 9 | 0 | 12 | True |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 373 | 0 | 15 | 12 | 0 | 0 | 264 | 82 | True |
| webqsp | `dense_top200+STRUCT:h3_bf_v2000~cs-fu` | 81 | 2 | 35 | 5 | 0 | 19 | 16 | 4 | True |

### 14.11 Surprises and contradictions with filed numbers

- **The WebQSP pick is smaller than the one it replaces** (2,094.6 nodes against 2,468.1) and 9.85 points higher; the old rule's price clause 2 (ratio upper end below 1) and clause 3 (ALL lower end above 0 with the ratio upper end at or below 1) both fire.
- **The two readings differ on MuSiQue**: the old rule says LEVEL (+0.25 points, ratio 1.07, an interval that includes 0) and the frontier reading says ABOVE (two budgets, 1,500 and 2,000, with the lower ends above 0); both are filed under their own names.
- **No PL5 cell is competitive on MetaQA**: the best option cell inside the bound reads 0.7882 on dev against PL4's 0.8978; MetaQA's D2 (section 14.6: 264 of its 373 attributed C golds cut by the hop-3 budget) is the account of what its pool still misses.
- **SQuAD's pick is 15 nodes and stands 0.54 points above the floor on eval** (0.9554 on 11,873 questions; dev 0.9519): the smallest cell that reaches 0.95 sits close to it by construction, so a 15-node pool is at the floor, not above it with room.


### 14.12 Disclosed and undone
- **The laptop's driver was re-ordered once, mid-run (25 Nov 15:39).** `run_chain.sh` (MuSiQue's G-R, then MultiHop-RAG, MetaQA, WebQSP, MuSiQue, SQuAD) was stopped while WebQSP's three grid parts were running (the parts were left running, not restarted) and replaced by `outputs/pl5/run_chain2.sh`, at the coordinator's request, so that MuSiQue's dev grid and select exist before WebQSP's frozen confirmation and the host's PL6 rows can start earlier. The commands, the rule, the cells, the seeds and the carves are the same; only their order changed (`outputs/pl5/run_chain.log` records both).
- **WebQSP's G-R ran twice**: PASSED at 15:01 (1,763 s) and, because `run_chain2.sh` called `run_ds.sh webqsp` without `no-gr`, again at 17:48 (3,265 s, the machine paging), PASS with 0 present and 0 size mismatches both times; `gr_webqsp.json` is the second record (same code, same cells, same 400 + 150 questions). The repeat cost about 55 minutes of laptop time and changed nothing.
- **SQuAD's G-R also ran twice**: PASSED at 14:03 (1,058.0 s; the dev grid and select of the first chain were filed at 14:47) and again at 22:32 (3,087.5 s), when `run_chain2.sh` called `run_ds.sh squad no-confirm` without `no-gr` (the dev grid parts existed and were not rebuilt); PASS with 0 present and 0 size mismatches both times, `gr_squad.json` is the second record. The repeat cost about 31 minutes of laptop time and changed nothing.
- **SQuAD's frozen confirmation ran as four question slices** (`zpl5r.py confirm-par --ds squad --slices 4`, 11,873 eval questions, PL4's method) from 22:34 and was merged at 00:05 on 26 Nov; the slices ran at roughly 5 to 10 minutes per 300 questions beside other work, so their seconds are upper bounds.
- **D1 and D2 for MuSiQue and WebQSP ran together** (`zpl5r.py miss`, one process each, after SQuAD's slices ended; MetaQA's ran earlier, at 13:57).
- The laptop's WebQSP grid ran at 7.8 to 8 s per question per part with the machine paging at about 4,000 pages a second (free memory under 1 GB beside three 4 to 5 GB processes): its seconds are upper bounds, as PL4's were.
- 6,078 cells are selected over on 1.5k-question dev carves (224 for MultiHop-RAG): a winner's dev ALL is optimistic, so every call is a paired interval on eval; the dev numbers of section 14.3 are the selection's, not a result.
- The bound is not changed by this page: the price at any size (14.5) is beside the calls, and a result that needs more than 2,500 nodes is filed as such, not carried. PL6 (744811e6) reads the user's ruling on the bound.
- D3 (section 1.2) was computed before the declaration commit on PL4's chosen cells; D1 and D2 here are run at PL5's chosen cells of the SHORT datasets, post-hoc, and decide nothing.
- The grid arrays (`*.npz`) are not committed (as PL2's to PL4's); the heads, selects, confirmations, the call, the diagnostics and the report are.

