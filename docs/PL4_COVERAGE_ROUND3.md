# PL4: the coverage target, round 3 (four ways to spend a walk's node budget, on all seven)

Declared 25 November 2026, before any PL4 coverage number exists. What I have read are PL3's FILED records
(`outputs/pl3/<d>/select.json`, `confirm_frozen.json`, `confirm_e3v.json`, `miss.json` for MuSiQue, MetaQA and WebQSP, `outputs/pl3/call.json`,
`report.md`; PL3's page sections 1 to 14; commits 7a55929 declaration, 9c70978 gates, 8bf1acf five of seven, 2824c2e seven of seven), the walk
code of `m3b_pools` and `zpl3g.py`, and the array names and shapes of the family stores (`outputs/m3b/csr/*.npz`: which arrays each holds, not their
values). Nothing below was computed after the page's rule was chosen. The only runs before this commit are label-free or agreement checks: the
cell list and its sha256 (`zpl4g.py cells`), the cost of the walks (`zpl4g.py time`, which prints seconds and nodes scored; no label is read), an
agreement check of the engine's walk against a per-node loop on random synthetic graphs (no real data: 1,888 comparisons, 0 differences), and the
engine self-test (the engine against a brute-force pool builder; it prints agreement counts, not coverage). **Nothing here trains, no model is
fitted, no LLM or spaCy is used, and every pool function is gold-free.** The pools of record, PL1's pools and PL2's and PL3's code and records are
untouched. This page is the third round of the coverage target (docs/PL1_POOLS_ON_UL2_LAYOUT.md section 7).

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how much ever is
  possible"; "dont wait on me work on the fixes to whatever comes up keep the cpu and gpu full dont waste time or compute".
- Earlier, still binding: "the pools or the rules or links all should be universal"; "i dont want to freeze pools for the sake of it, make sure we
  freeze when we know it is the best for that specific arm"; "link improvement with precision is all we need". Pools first and parameter-free (no
  learning, no model, no LLM, no spaCy); one model per family; the hub line is CLOSED (UL2 8.15): none of the changes below reads a node's degree or
  adds a hub step; every rule is universal on all seven datasets (MultiHop-RAG, MuSiQue, SQuAD, MetaQA, WebQSP, HotpotQA, 2Wiki), carried everywhere
  or nowhere.

## 1. What PL3 found (the records this round starts from)

PL3 (CARRY_PL3: BELOW 0, ABOVE 2 (HotpotQA, 2Wiki), LEVEL 5, `outputs/pl3/call.json`) chose, per dataset, the R95 cell of the union of PL2's 804 and its
own 1,470 cells (dev, frozen lists) and confirmed it on eval:

| dataset | PL3's chosen cell | eval ALL (frozen) at nodes | state |
|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 at 50.0 | past 0.95 |
| squad | `rrf1000_b15` | 0.9554 at 15.0 | past 0.95 (0.9421 on e3v) |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 at 77.9 | past 0.95 |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 at 917.0 | past 0.95 |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 at 2,455.1 | **SHORT** |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 at 2,335.9 | **SHORT** (0.9296 on e3v) |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 at 2,468.1 | **SHORT** (0.7618 at 2,720.1 nodes on e3v) |

PL3's diagnostic D1 (section 14.5 there; 300 dev questions evenly spaced; labels read; post-hoc) is what this round is built on: of the 658 gold nodes
missing from the three SHORT cells' pools, **656 are class C** (not in the cell's own walk but in the uncapped three-hop FULL reach of its seeds) and 2 are
U; none is R (these cells carry no `+top{M}` cut, so no ranking cut can lose a node: the pool is the walk). The uncapped walk reaches 1.000 / 0.998 /
0.996 of the missing golds on MuSiQue / MetaQA / WebQSP. **The shortfall is the caps and the 2,500-node bound, not the reach.** The grid's own best ALL at
any size is 0.9355 / 0.9653 / 0.8480 only at 5,783 / 6,268 / 6,067 nodes (2.3 to 2.5 times the bound). Two more facts from PL3's tables that bear on what to
try: (a) the three SHORT cells spend their whole walk budget (their pools read 2,337 to 2,487 nodes against a visited cap of 2,000 plus a base of 400 to
800 nodes, or 5,000 on WebQSP's), so the **visited cap binds**: what survives it, and in which order, is the cell; (b) the order that survives is, in
PL3's best cells, arrival order (MetaQA) or cosine to the question (`vr`: MuSiQue, WebQSP), a signal that knows nothing about how many paths lead to a
node or how the families share the budget. PL3's section 14.7 also names a limit of the call's rule, which this page addresses in section 6.

## 2. What is fixed (PL2's and PL3's pages, by reference)

Everything of PL2 sections 2, 3 and 4.1 to 4.2 and PL3 sections 2 and 4 (the walk, `cs`, `vr`, `FS`, `s10`, `s20`) is unchanged and read by the same code
(`zpl2g.py` and `zpl3g.py` imported, not edited): the substrate (the served package for the six datasets, MultiHop-RAG's own B0 layout), the dev carves
(`s1sel`; MultiHop-RAG's dev split; WebQSP's `sorted(train ids)[1::2]`) and eval carves (`s1eval`; MultiHop-RAG's test), the frozen first stage for
selection and the frozen and e3v first stages for confirmation, the seeds (dense top 5 U SPLADE top 5) unless a cell says otherwise, the measures (ALL =
`all_gold_at_pool`, ANY, mean nodes = `candidates_mean`, Wilson 95% on ALL, paired bootstrap of 1,000 resamples), the floors 0.95 and 0.98, the bound
**2,500** mean nodes, the bases and sizes of PL3 section 5 (`rrf1000_b25`, `b100`, `b400`, `b800`, `b1600`, `dense_top200`; M = 50, 100, 200, 400, 800,
1,600), and the settings `h2_c100_v5000` (2 hops, caps 100 / 100, visited 5,000), `h3_c100_v5000` (3 hops, 100 / 100, 5,000) and `h3_c25_v2000` (3 hops,
25 / 25, 2,000). A walk with none of this page's new options is PL3's or PL2's walk, bit for bit (gate G-R).

## 3. What this round tests, and what it does not

| PL3 14.8 name | here | walks / cells |
|---|---|---|
| support-ordered expansion (the number of seeds that reach a node) | option `sp` | 17 walks |
| reading the per-link weights | option `rk` (row rank): NER and KNN rows are stored best weight first, so a node's place in a row IS its weight order; the weight values are not read | 17 walks |
| a hop-3 frontier restricted to the nodes the hop-2 ranking kept | option `eq`: each hop may keep only an equal share of what is left of the visited cap, so hop 3 expands only from the nodes hop 2 kept and always has budget | 22 walks |
| per-family visited budgets | option `rr`: the families take turns (every family's first, then every family's second, ...) | 12 walks |
| label-ordered caps (per-link rule labels ordered by the ledger's precision, on a UL2 layout) | **not in this round**: the six datasets' package graph carries no per-link rule label and the UL2 layout stores exist for MultiHop-RAG only (`outputs/ul2m/multihoprag`; the other datasets' `outputs/ul2*` folders hold grades and ranks, not link stores), so the order is not expressible on every dataset and a rule cannot be universal on it | 0 |

**What the four options can and cannot touch.** The caps keep, for each frontier node and family, its first `cap` neighbours (or the best by cosine, `cs`);
that set is PL3's and none of the four options changes it. The options change **which of the capped candidates survive the visited cap and in what
order the next hop expands them** (`sp`, `rk`, `rr`), and **how the cap is shared between the hops** (`eq`). A gold that a per-frontier cap or a per-seed cap
cuts before the visited cap sees it is out of reach of every option here; diagnostic D2 (section 10) says, in numbers and after the fact, how many of the C
golds are of that kind, which fixes what the next round has to be. No link-rule change is made here (it goes through UL2's gates).

## 4. The changes, exactly (the walk; section 8's gates hold the code to this text)

A **walk** is `{regime}:{setting}~{mods}`, mods joined by `-` in the order `s10`, `s20`, `cs`, `vr`, `sp`, `rk`, `rr`, `eq`. At most one of `vr`, `sp`, `rk`
(the order key); `s10` and `s20` never together; the key, `rr` and `eq` never on a one-hop setting; `eq` only on a three-hop setting; `rr` only on regime FULL
(it needs two or more families at hops 2 and 3). A walk without `sp`, `rk`, `rr`, `eq` is PL3's walk.

For each hop h >= 2 of a walk (the frontier at hop 2 is hop 1's nodes; at hop 3, the nodes hop 2 kept, in the order it kept them): for every frontier node (index i,
in frontier order) and every family of the hop's regime (in the order STRUCT, NER, KNN), its **list** is the first `cap` neighbours in the family's row, or with `cs`
the `cap` with the highest float64 inner product with the question vector out of its first 512 neighbours (ties to the lower id); an **entry** is (node x, frontier
index i, family, place r), r = 0 for the first of a list. The entries are in PL3's candidate order (frontier node, then family, then place). The entries whose node is
already visited (the seeds and every node kept at an earlier hop) are removed; the **fresh nodes** are the distinct nodes that remain. The cell's order key decides
which fresh nodes the hop keeps and the order of the frontier for the next hop:

- **no key:** the order of first appearance (PL2's walk). **vr:** by cosine to the question, highest first, ties to the lower id (PL3).
- **sp (support):** by the number of distinct frontier nodes that have an entry for the node (over the families together), highest first; ties by cosine, then id.
- **rk (row rank):** by the smallest place r any of its entries has (a node that is some frontier node's best neighbour comes before one that is nobody's), then by
  cosine, then id. With `cs` the place is the cosine place in a node's list; without it, the row's own order: best weight first for NER and KNN, node id for STRUCT.
- **rr (families take turns):** each family's entries are ordered by the key alone (its own support, its own places; with no key, first appearance), the families'
  ordered lists are merged by position (every family's first node, then every family's second, ...; within a position STRUCT, NER, KNN), and a node keeps its first
  appearance. The merged list is cut as above.
- **eq (equal share):** with `rem` = visited cap minus the nodes visited so far when hop h starts, the hop keeps at most `ceil(rem / (H - h + 1))` fresh nodes, H = 3
  (so hop 2 keeps at most half of `rem`, hop 3 keeps what is left). Without `eq` a hop keeps at most `rem`, as in PL2 and PL3.

The kept nodes (the first `budget` of the ordered fresh nodes) join the visited set and become the next frontier, in that order. Hop 1 is PL3's (the per-seed cap,
`cs` if on, first appearances) and is not changed by any of the four options. **Seeds, regimes, `FS`, `s10`, `s20`, `cs`, `+top{M}` and the union cells are as in
PL3** (section 2); the ranked cells `+top{M}` rank the walk's nodes by cosine to the question, as before. There is no `+brg` cell in this round (PL3 14.4: bridge-conditioned
ranking bought nothing on any of the seven). Inner products are accumulated in float64 from the fp16 rows in numpy's pairwise order (no BLAS), one score per node and
question, remembered for that question; ties go to the lower node id everywhere; nothing is random.

## 5. The cells (fixed at this commit; nothing is added after a number is read)

- Bases and sizes as PL3 (section 2): 6 bases; M in {50, 100, 200, 400, 800, 1,600}. Each walk x each base as a union (6) and as a ranked cell (6 x 6 = 36) = 42 cells.
- **46 walks** (`zpl4g.py cells` prints them; every one has at least one new option, so none repeats a PL2 or PL3 walk):
  - **G1, the order key** (14): `cs-sp` and `cs-rk` on STRUCT:`h3_c25_v2000` and STRUCT:`h3_c100_v5000` and on FULL:`h2_c100_v5000`, `h3_c25_v2000`, `h3_c100_v5000`
    (10); `sp` and `rk` with the caps in row order on FULL:`h3_c25_v2000` and FULL:`h3_c100_v5000` (4).
  - **G2, equal shares** (14): `cs-vr-eq`, `cs-sp-eq`, `cs-rk-eq` on STRUCT and FULL, each on `h3_c25_v2000` and `h3_c100_v5000` (12); `cs-vr-eq` on FS:`h3_c25_v2000` and
    FS:`h3_c100_v5000` (2).
  - **G3, the families take turns** (12): `cs-vr-rr`, `cs-sp-rr`, `cs-rk-rr` on FULL:`h2_c100_v5000` and FULL:`h3_c25_v2000` (6); `cs-vr-rr-eq`, `cs-sp-rr-eq`,
    `cs-rk-rr-eq` on FULL:`h3_c25_v2000` and FULL:`h3_c100_v5000` (6).
  - **G4, wider seeds with the new options** (6): FULL:`h2_c100_v5000` with `s10-cs-sp`, `s10-cs-rk`, `s20-cs-sp`, `s20-cs-rk`; FULL:`h3_c25_v2000` with `s10-cs-vr-eq`, `s20-cs-vr-eq`.
- 46 x 42 = **1,932 cells**, all new. Cell list sha256 (`P2.cells_sha256`, the sorted names):
  `32502a00db12e3afffd325170183f0508a7f108afecd505c73a516a4908eed96` (1,932 distinct). PL2's 804: `76bdf3a822e6bb8caf21121f853c9a8383e7844f99425e004d18c75f2aededc1`;
  PL3's 1,470: `ff879881090993bca1e063b715724c0d89411b8cacc77f5df7bcccc2b29c642e`; the union, **4,206** distinct:
  `4b076faebda8cac8e140c5c783baee6ebc252b510b85350c1f7644f811509b95`.
- Tags (descriptive, for what each option bought; cells with several options carry several): `sp` 294 alone, `rk` 294, `eq` 252 (the `vr` shares), `sp+eq` 168, `rk+eq` 168,
  `rr` 84, `sp+rr` 84, `rk+rr` 84, `rr+eq` 84, `sp+rr+eq` 84, `rk+rr+eq` 84, `sp+seeds` 84, `rk+seeds` 84, `eq+seeds` 84 (they sum to 1,932).
- The walk list was fixed after one cost timing on MuSiQue (3.9 s per question for all 46 walks on 40 dev questions, about 1.5 times PL3's), before any coverage was read.
  Options that were considered and left out for cost: a parent-conditioned rank (the cosine of a node to the question plus its best parent, a pair of vectors per entry)
  and a wider cosine window than 512 (millions of entries per hop); they are named in section 12.
- Cells above the bound are read, cannot be selected, and are shown in the any-size readings.

## 6. The rule, the confirmation and the call

**Selection** (dev, frozen lists, over the union of PL2's 804, PL3's 1,470 and the 1,932): among cells with mean nodes <= 2,500, **R95** = the smallest mean nodes with dev ALL
>= 0.95 (ties to the higher ALL, then the name); if none, the highest ALL (ties to fewer nodes), and the dataset is flagged SHORT. **R98** likewise at 0.98. Readings (each by
the same rule): `all` (the decider), `pl3` (PL2's and PL3's 2,274 cells, the incumbent grid: it must give PL3's filed pick, checked against `outputs/pl3/<d>/select.json`),
`new` (the 1,932 only), and per option `pl3+TAG` and `only:TAG`. The whole curve and the frontier of each reading are filed.

**Confirmation** (eval, nothing re-chosen): the cells R95 and R98 name in the readings `all`, `pl3` and `new`, the matched-node-count picks below, and PL2's filed own construction are re-run
unchanged on the eval carve, with the frozen lists and with e3v's lists.

**The call, as PL3 states it (the old rule, unchanged).** On eval, frozen lists, per dataset: NEW = R95 of the reading `all`, OLD = R95 of the reading `pl3` (PL3's own chosen cell, the
incumbent). Paired bootstrap of the ALL difference (NEW - OLD) and of the ratio of mean nodes (NEW / OLD); one generator per dataset, `numpy.random.default_rng([20261125, k])`, k the
dataset's place in the order MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MuSiQue, MetaQA, WebQSP, consumed in the order R95, R98, the new-only pick, then the five budgets below.
- **BELOW**: NEW misses 0.95 where OLD reaches it, or the ALL difference's upper end is below 0 with the nodes ratio's lower end at or above 1.
- **ABOVE**: NEW reaches 0.95 where OLD does not, or the nodes ratio's upper end is below 1 with ALL not BELOW, or the ALL difference's lower end is above 0 with the nodes ratio's upper end
  at or below 1.
- **LEVEL**: anything else.
**CARRY_PL4** when all seven are read, no dataset is BELOW, at least one is ABOVE, and every chosen cell is within the 2,500 bound on the selection carve; INCOMPLETE while fewer than seven are
read; otherwise NOT_CARRIED (PL3's grid stays the rule).

**A change of the rule, proposed here before any number: the frontier reading.** PL3 14.7 filed that the call cannot credit a gain bought with more nodes on a SHORT dataset (MuSiQue's +5.05
points at 2.31 times the nodes is LEVEL, because an ABOVE clause needs either the floor or a nodes ratio at or below 1). This is a limit of the rule, not a finding, and a coverage target that
asks for the highest coverage inside a fixed node bound has to be able to say when a grid holds more coverage inside that bound. So this page also files, beside the old call and without
replacing it, a second reading:
- For each budget N in {500, 1,000, 1,500, 2,000, 2,500} mean nodes: NEW_N = the cell of the reading `all` with the highest dev ALL among cells with dev mean nodes <= N (ties to fewer nodes,
  then the name); OLD_N the same over the reading `pl3`. Both are at most N nodes on dev by construction: this is a comparison at **matched node counts**, not at the same cell size.
- On eval (frozen lists), the paired bootstrap of ALL(NEW_N) - ALL(OLD_N). The budget is **ABOVE** if the interval's lower end is above 0, **BELOW** if its upper end is below 0, else LEVEL
  (the same cell on both sides is LEVEL with difference 0).
- A dataset's **frontier status**: BELOW if any of the five budgets is BELOW; ABOVE if at least two of the five are ABOVE and none is BELOW (two, because five budgets are five looks);
  otherwise LEVEL.
- **CARRY_PL4_F** when all seven are read, no dataset is BELOW by the old reading's clauses and none is BELOW by the frontier reading, at least one dataset is ABOVE by the frontier reading, and
  every chosen cell is within the 2,500 bound. It changes how the call credits a pool; it does **not** change what is carried: each dataset's construction is still its R95 cell of the
  reading `all` (a SHORT dataset's highest dev ALL inside the bound, which is NEW_2500). The matched-budget picks are readings, not constructions.
- **Both calls are filed under their own names (`call` and `call_frontier` in `outputs/pl4/call.json`) and the old reading is never hidden or replaced.** If they differ, this page does not choose;
  which one governs is the user's decision, and the page states what each would carry.

**How the bound is read on e3v lists.** The 2,500 bound is a property of the cell as chosen on the selection carve's frozen lists: selection, the within-bound clause of both calls and the
frontier budgets all use the dev carve's frozen lists, because e3v first-stage lists exist for the eval carve only (E0's `top_ids`; there are no e3v lists for the select carves), so no
selection can be made on them. The e3v confirmation re-runs the chosen cells on the eval carve's e3v lists and **reports its mean nodes beside the frozen reading with a flag `OVER_BOUND`
when it exceeds 2,500** (PL3's WebQSP cell reads 2,720 there against 2,468 frozen: the e3v fused lists differ, so the same walk spends different nodes). The flag does not change either call
(a call is read on the frozen lists as declared); it is filed in the confirm record, the call table and the report, and a flagged cell is not offered to PL1's e3v pool rows by this page. An
e3v-bounded selection needs e3v lists for the select carves and is named in section 12.

## 7. Stop rule

The 46 walks, the bases, the sizes M, the settings, the bound, the floors, the readings, the budgets of the frontier reading and the carves are fixed by this commit; every output records the sha256
of the cell list and the code-path hash, and `zpl4g.py` refuses a grid whose cell list is not the declared one and a select whose heads carry another code-path hash. Nothing is added to the grid
after a number is read. If a dataset is still SHORT after this round the report (a) says in numbers which gold nodes remain unreached and **which stage cut them** (section 10), and (b) names the
next change, which is declared on its own page.

## 8. Gates (each is a failure that stops that dataset's reading; none of its numbers is read)

- **G-R (PL2's and PL3's cells are reproduced).** The PL4 engine, given the 2,274 cell names of PL2 and PL3, builds them on the first 400 dev questions of a dataset and every present-count and
  pool-size equals the filed dev arrays (PL2's `grid_dev_frozen`, PL3's three parts) element for element. Required on every dataset that is read; the host's own G-R on MuSiQue is a
  precondition of every host select and confirm (`pl4-grmusique`).
- **G-X4 (the walk).** With the four options off the engine's walk equals `zpl3g.expand3` array for array, on 60 questions, for every regime and setting and for `cs` and `vr` in every combination;
  every PL3 walk and every one of the 46 walks equals an independent per-node Python loop written from section 4, on 3 questions per dataset.
- **Engine self-test.** The engine's present counts and pool sizes against a brute-force pool builder written from section 4 (explicit sets) on a sample of questions, 3 cells per walk.
- **G-F (gold-free).** No pool function takes or reads a label (checked on the signatures and the source of every function that builds a pool); the states are built, and hashed for G-D, from a
  source whose label arrays are withheld. Labels enter only in `zpl3g.grade3`.
- **G-D (determinism).** The pools of a sample of questions hash the same under 1 thread with chunk 64 and 4 threads with chunk 7.
- **G-U (universal).** One code-path sha256 (the pool code, the constants and PL3's code-path hash, which holds PL2's) and the cell list's sha256 are recorded in every grid head; `select` refuses a
  head that differs, and `call` refuses to file when two datasets differ.

## 9. What each option can and cannot show

| option | what it can show | what it cannot show |
|---|---|---|
| `sp` support | whether a node many frontier nodes lead to (a gold reached by several paths: a KB answer entity, a paragraph several entities link) is worth more of the budget than a node one path leads to | a gold reached by one path only; any gold the per-frontier cap already dropped; with cap 25 many nodes have support 1 and the cosine tie-break decides, as in `vr` |
| `rk` row rank | whether every frontier node getting its best neighbours in before any gets its second beats arrival order (the first frontier nodes take the whole budget now) and beats global cosine; without `cs` it reads the stored weight order of NER and KNN rows | the weight values themselves (not read); STRUCT rows have no weight, their place is node id without `cs` and cosine with it |
| `eq` equal shares | whether reserving budget for hop 3 (which a hop-2 that fills the visited cap starves) buys the third-hop golds D1 says exist (MuSiQue needs three hops: 0.892 at two, 1.000 at three) | anything on a two-hop setting; a gold at hop 2 loses budget to hop 3 when the share binds; the share is one rule (equal), not a tuned split |
| `rr` families take turns | whether NER and KNN get a share of the budget next to STRUCT (MuSiQue: NER reaches 0.627 and KNN 0.256 of the missing golds within 3 hops) | MetaQA and WebQSP (NER and KNN reach under 4% of their golds: `rr` can only take budget from STRUCT there); one fair-share rule, not a learned weight |

None of the four can reach a gold the walk's caps never list (section 3) or a gold the first stage and the graph together do not reach; the pools and links are rule-made, no label is read, and
no semantic signal is added that the first stage's encoder did not have.

## 10. The short datasets: what remains unreached (diagnostics D1 and D2; post-hoc and labelled so; they decide nothing)

**D1 as PL3 section 10** (class R / C / U of each gold missing from the chosen cell's pool; 300 dev questions evenly spaced over the carve), for each dataset whose R95 of the reading `all`
is SHORT, at that cell. **D2 (new): which stage cut each C gold.** For each C gold of the chosen cell, on the same questions, from a traced walk of that cell:
- **cap**: the gold is a neighbour (uncapped, in the families of the hop) of a node the walk kept at the previous hop (the seeds for hop 1), at hop h, and is not in that hop's capped entries
  (a per-seed or per-frontier cap dropped it);
- **budget**: it is in the capped entries of hop h (not visited) and not among the fresh nodes the hop kept (the visited cap, the hop's share, or the order dropped it);
- **upstream**: neither of the two above: it is not a neighbour (in the hop's families) of any node the walk kept at any hop it walks, so every route to it passes through a node an earlier hop
  dropped, or needs more hops than the cell walks (a two-hop cell cannot take a third-hop gold). A gold that is a neighbour of the frontier at several hops is counted at the first hop where it was
  in the capped entries and not kept (**budget**), else at the first hop where it was a neighbour and not in the capped entries (**cap**); hop 1 has no visited cap, so a hop-1 miss is always a cap.
The report gives, per dataset, the number of C golds in each stage (and by hop) and says which stage the next change has to attack. Both read the labels, are run after `select`, and are never an
input to any rule. The code is `outputs/mp_unified/zpl4r.py miss`.

## 11. Code, rows, outputs

- **New modules only:** `outputs/mp_unified/zpl4g.py` (engine, commands `cells`, `time`, `gr`, `grid`, `select`, `confirm`, `call`, `selftest`; every command except `cells` and `call` takes
  `--host`; committed with this page) and `outputs/mp_unified/zpl4r.py` (the report, D1 and D2; it reads filed records and the labels for D1/D2 only, chooses nothing; committed with the
  results). They import `zpl3g.py`, `zpl2g.py`, `scripts/m3a_headroom.py` and `mp_retrieval.m3b_pools` and edit none of them.
- **Outputs:** `outputs/pl4/<d>/{grid_dev_frozen*.npz (arrays, uncommitted), grid_dev_frozen*.json (heads), select.json, confirm_frozen.json, confirm_e3v.json, miss.json}`,
  `outputs/pl4/{cells.json, selftest.json, gr_<d>.json, call.json, report.md, feed_rows.txt}`.
- **Laptop** (one heavy job at a time, three parts of the dev grid in parallel, frozen lists): MultiHop-RAG (with e3v), MetaQA, MuSiQue, WebQSP and SQuAD; expected cost per dataset about 1.5
  times PL3's (PL3: 46 s, 1,982 s, 4,054 s, 4,122 s and 5.6 GB, 2,032 s). **Host rows** (`pl4-*`, CPU only, one class per row: the first two name tokens are unique per row; declared memory
  = the measured peak of the PL3 row of the same dataset until a PL4 peak is measured; `pylib_run.py` convention) in `outputs/pl4/feed_rows.txt`: the host's G-R on MuSiQue, HotpotQA and 2Wiki;
  the HotpotQA and 2Wiki dev grids (three parts each; PL3's 3,228 s / 14.8 GB and 4,819 s / 17.1 GB, so about 1.5 times that), select and confirm; the e3v confirmations of SQuAD, MuSiQue,
  MetaQA and WebQSP. They depend on PL3's and PL2's records, whose dev grids the union reads.

## 12. What follows

1. The calls (section 6) over all seven; the results are filed here (section 14) with, per dataset, PL3's chosen cell against the new one, the frontier reading's five budgets, the smallest cell
   at 0.95 and 0.98 (dev and eval), what each option bought, what is still short, D1 and D2.
2. **Next changes, names only** (none is adopted, none is in the grid): label-ordered caps (once a UL2 layout exists on every dataset); a parent-conditioned rank (a node's cosine to the
   question plus its best parent's, which scores the bridge the question text does not name); the weight values of NER and KNN rows as a key; a window wider than 512 for `cs`; whatever D2
   says dominates the cut (a per-frontier cap cut would need caps that follow the budget, not these options).
3. An e3v-bounded selection once e3v lists exist for the select carves; the same grids on the UL2 layout stores once S1 has materialized them.

## 13. Disclosed

- Written after reading PL3's filed records and page (sections 1 to 14), the walk code of `m3b_pools` and `zpl3g.py`, and the array names and shapes of the family stores; no PL4 coverage number
  was computed before this commit (the cost timing printed seconds and nodes scored; the self-test printed agreement counts).
- 4,206 cells are selected over on 1.5k-question dev carves (224 for MultiHop-RAG, about +/- 2 points): the winner's dev ALL is optimistic, more so than PL3's over 2,274. That is why the eval
  read decides and why both calls are paired intervals on eval against PL3's own chosen cell (or its matched-budget cell), not comparisons of dev numbers.
- The frontier reading is a **change of the rule**, proposed here and not before; it makes a node-bought gain count, and the old reading stays beside it. A matched-budget ABOVE can be a gain
  bought entirely with nodes (a cell at 2,500 against PL3's at 2,455, say), which is exactly what it is meant to credit and exactly what the old rule refuses; the page does not say which view
  is right.
- `rk` reads the weights only through the order the rows are stored in (best weight first for NER and KNN); the weight values and the typed relation labels are not read, and label-ordered
  caps are not tried (section 3). `sp` and `rk` use the question's cosine as the tie-break, so every PL4 cell needs the ranking vectors (on the laptop: MultiHop-RAG's e3v only; the other
  datasets' e3v vectors sit on the host).
- The four options do not change the per-seed or per-frontier caps (section 3): if D2 says the C golds are cut there, this round cannot reach them and the next must.
- PL3's `cs` still ranks inside the first 512 neighbours of a row; `eq` is one rule (equal shares over the remaining hops), `rr` one rule (turns), not tuned splits; the settings in the grid
  are PL2's three, so a family can look absent because its best setting is not in the grid.
- The six datasets' graph is the old package graph (as PL2 and PL3 disclosed); MultiHop-RAG reads its B0 layout. The UL2 layout is not materialized on the six.
- Pre-commit runs on real data are the cost timing (MuSiQue, 40 dev questions, 46 walks) and the self-test (MultiHop-RAG and SQuAD); G-R, which compares the engine to PL3's and PL2's filed
  arrays and reads no PL4 number, runs after this commit, on every dataset that is read.
