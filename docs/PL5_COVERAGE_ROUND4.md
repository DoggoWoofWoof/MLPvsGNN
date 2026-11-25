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
