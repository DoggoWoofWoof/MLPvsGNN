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

## 14. Results (25 Nov 2026; 7 of 7 datasets read)

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl4/<d>/{select,confirm_frozen,confirm_e3v,miss}.json`, `outputs/pl4/call.json`, `outputs/pl4/gr_*.json`) by
`outputs/mp_unified/zpl4g.py call` and `outputs/mp_unified/zpl4r.py report` / `miss`; the tables of section 14.9 are pasted unchanged from `outputs/pl4/report.md`. The laptop datasets (MultiHop-RAG, SQuAD,
MuSiQue, MetaQA, WebQSP) ran on the laptop (dev grid, select, frozen confirmation; SQuAD's frozen confirmation in four question slices through `zpl4r.py confirm-par`, whose arrays were first checked equal to
the single-process confirmation on MultiHop-RAG); HotpotQA and 2Wiki, and every e3v confirmation but MultiHop-RAG's, are host rows. Nothing was changed in the rule, the cells or the engine after a number was read.

### 14.1 Gates and cost

The engine `zpl4g.py` is byte-identical to the declaration commit c71f0d7: script sha256 starts `a9aa6a03c493`, the code-path hash `aa7c9ac9ddd5` is equal in every select head and `call.json` (`code_path_equal_everywhere` true; G-U PASS), the cell list `32502a00db12` (1,932 cells; 4,206 with PL2's and PL3's). **G-R** (PL2's 804 and PL3's 1,470 cells rebuilt on the first 400 dev questions, 909,600 elements per dataset, 224 questions on MultiHop-RAG) PASSED with 0 present and 0 size mismatches on MultiHop-RAG, SQuAD, MuSiQue, MetaQA, WebQSP (laptop) and on MuSiQue, HotpotQA, 2Wiki (host rows `pl4-musiquegr`, `pl4-hotpotqagr`, `pl4-2wikigr`: `gr_<d>_host.json`). The engine self-test PASSED (`selftest.json`: cell list, G-F, G-X4, engine against a brute-force pool builder on MultiHop-RAG and SQuAD, G-D). Every select re-read PL3's own pick from the union of PL2's and PL3's cells and got PL3's filed pick (`pl3_pick_equals_filed` true on the 7 read). Cost of the dev grids (sum of the three parts): MultiHop-RAG 82 s, 0.28 GB peak (laptop); SQuAD 7,012 s, 0.12 GB peak (laptop); HotpotQA 4,210 s, 14.95 GB peak (host); 2Wiki 7,610 s, 17.35 GB peak (host); MuSiQue 9,488 s, 0.64 GB peak (laptop); MetaQA 2,392 s, 0.40 GB peak (laptop); WebQSP 8,308 s, 5.46 GB peak (laptop). Frozen confirmations: MultiHop-RAG 15 s, SQuAD 4,006 s, HotpotQA 1,773 s, 2Wiki 9,233 s, MuSiQue 2,094 s, MetaQA 2,219 s, WebQSP 605 s. (The laptop's numbers were taken with up to eight heavy processes sharing 12 logical CPUs, so they are upper bounds; SQuAD's 7,012 s ran beside the G-R of the next datasets; SQuAD's confirmation, in four parallel slices, is wall-clock.)

### 14.2 The calls (read 7 of 7: old rule CARRY_PL4; frontier reading NOT_CARRIED)

**The old rule, as PL3 states it** (unchanged; section 6): per dataset on eval, frozen lists, NEW = R95 of the reading `all` (all 4,206 cells), OLD = PL3's own chosen cell, paired bootstrap of the ALL difference and of the
ratio of mean nodes. Clause by clause:

| dataset | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | BELOW: (1) NEW misses where OLD reaches / (2) ALL diff upper end < 0 with ratio lower end >= 1 | ABOVE: (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (ALL not BELOW) / (3) ALL diff lower end > 0 with ratio upper end <= 1 | status | chosen cell inside 2,500 (eval nodes) |
|---|---|---|---|---|---|---|---|
| MultiHop-RAG | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (50.0) |
| SQuAD | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (15.0) |
| HotpotQA | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (77.9) |
| 2Wiki | yes / yes | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (917.0) |
| MuSiQue | no / no | +0.33 [-0.33, +0.99] | 0.9266 [0.9253, 0.9279] | no / no | no / yes / no | **ABOVE** | yes (2,274.9) |
| MetaQA | no / no | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** (the same cell) | yes (2,335.9) |
| WebQSP | no / no | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | no / no | no / no / no | **LEVEL** | yes (2,468.1) |

- **BELOW: 0.** ABOVE: 1 (MuSiQue); LEVEL: 6 of the 7 read.
- **The old call: CARRY_PL4** (BELOW 0, ABOVE 1, LEVEL 6; read 7 of 7). CARRY_PL4 needs all seven read, no BELOW, at least one ABOVE and every chosen cell inside the 2,500 bound.
- **MuSiQue's ABOVE is by the price clause alone** (clause 2: the nodes ratio's upper end is below 1 with ALL not BELOW); its ALL difference is +0.33 points [-0.33, +0.99], an interval that includes 0. The rule is applied as written.

**The frontier reading** (section 6; a proposed change of the rule, filed beside the old one and never in its place): per budget N, the highest dev ALL among cells of at most N dev nodes, all cells against PL3's cells; paired
bootstrap of the eval ALL difference (points; `=` marks the same cell on both sides, difference exactly 0):

| dataset | 500 | 1,000 | 1,500 | 2,000 | 2,500 | frontier status |
|---|---|---|---|---|---|---|
| MultiHop-RAG | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| SQuAD | +0.00 [+0.00, +0.00] LEVEL = | +0.04 [-0.05, +0.13] LEVEL | +0.00 [+0.00, +0.00] LEVEL = | -0.01 [-0.05, +0.03] LEVEL | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| HotpotQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| 2Wiki | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | -0.01 [-0.14, +0.13] LEVEL | **LEVEL** |
| MuSiQue | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | -0.58 [-1.57, +0.50] LEVEL | +0.33 [-0.29, +0.95] LEVEL | **LEVEL** |
| MetaQA | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | **LEVEL** |
| WebQSP | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL = | +0.00 [+0.00, +0.00] LEVEL | **LEVEL** |

- **Frontier call: NOT_CARRIED** (BELOW 0, ABOVE 0, LEVEL 7; read 7 of 7). On the 7 read no budget of any dataset is ABOVE by the frontier's rule
  (a lower end above 0) and none is BELOW; the cells at each budget, with dev and eval numbers, are in the table of section 14.9 (`What the PL4 cells bought`) and in `call.json`.
- **The two calls are filed under their own names and this page does not choose between them** (section 6). At 7 of 7 read: the old rule is CARRY_PL4; the frontier reading is NOT_CARRIED.

The cells behind each frontier budget (dev ALL at dev nodes; eval ALL at eval nodes; the eval difference in points):

| dataset | budget | cells | dev | eval | difference |
|---|---|---|---|---|---|
| MultiHop-RAG | 500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 1000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 1500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 2000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| MultiHop-RAG | 2500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (both) | 1.0000 at 100.0 | 0.9912 at 100.0 | same cell |
| SQuAD | 500 | `rrf1000_b400+STRUCT:h1_c100~cs+top50` (both) | 0.9960 at 446.9 | 0.9965 at 446.7 | same cell |
| SQuAD | 1000 | NEW `rrf1000_b100+FULL:h3_c100_v5000~cs-vr-eq+top800`; PL3 `rrf1000_b400+STRUCT:h3_c100_v5000~cs-vr+brg400` | 0.9980 at 900.0; 0.9973 at 795.7 | 0.9978 at 900.0; 0.9974 at 793.3 | +0.04 LEVEL |
| SQuAD | 1500 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` (both) | 0.9987 at 1,199.8 | 0.9982 at 1,200.0 | same cell |
| SQuAD | 2000 | NEW `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800`; PL3 `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top200` | 0.9993 at 1,600.0; 0.9993 at 1,737.0 | 0.9991 at 1,599.9; 0.9992 at 1,740.6 | -0.01 LEVEL |
| SQuAD | 2500 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (both) | 1.0000 at 2,190.4 | 0.9992 at 2,200.0 | same cell |
| HotpotQA | 500 | `rrf1000_b100+FULL:h1_c100~s20+top800` (both) | 0.9887 at 460.5 | 0.9854 at 460.9 | same cell |
| HotpotQA | 1000 | `rrf1000_b800+FULL:h1_c100~cs+top400` (both) | 0.9927 at 932.5 | 0.9896 at 931.2 | same cell |
| HotpotQA | 1500 | `rrf1000_b800+FULL:h1_c100~s20+top800` (both) | 0.9934 at 1,111.7 | 0.9923 at 1,112.4 | same cell |
| HotpotQA | 2000 | `rrf1000_b1600+FULL:h1_c100~s20+top800` (both) | 0.9940 at 1,880.9 | 0.9945 at 1,883.6 | same cell |
| HotpotQA | 2500 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` (both) | 0.9960 at 2,246.8 | 0.9918 at 2,226.5 | same cell |
| 2Wiki | 500 | `rrf1000_b200+FULL:h1_c100+top400` (both) | 0.9325 at 419.2 | 0.9360 at 421.6 | same cell |
| 2Wiki | 1000 | `rrf1000_b400+FULL:h1_c100~s20` (both) | 0.9552 at 911.2 | 0.9614 at 920.4 | same cell |
| 2Wiki | 1500 | `rrf1000_b400+FULL:h1_c100~s20` (both) | 0.9552 at 911.2 | 0.9614 at 920.4 | same cell |
| 2Wiki | 2000 | `rrf1000_b400+FULL:h1_c100~s20` (both) | 0.9552 at 911.2 | 0.9614 at 920.4 | same cell |
| 2Wiki | 2500 | NEW `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq`; PL3 `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9626 at 2,215.1; 0.9606 at 2,078.2 | 0.9656 at 2,211.2; 0.9656 at 2,085.7 | -0.01 LEVEL |
| MuSiQue | 500 | `rrf1000_b400+FULL:h1_c100+brg100` (both) | 0.7373 at 498.2 | 0.7683 at 498.1 | same cell |
| MuSiQue | 1000 | `rrf1000_b800+FULL:h1_c100~s10+top200` (both) | 0.7947 at 988.6 | 0.8213 at 986.6 | same cell |
| MuSiQue | 1500 | `rrf1000_b800+FULL:h1_c100~s20+top800` (both) | 0.8246 at 1,388.8 | 0.8395 at 1,357.1 | same cell |
| MuSiQue | 2000 | NEW `rrf1000_b400+FULL:h3_c25_v2000~cs-sp-eq+top1600`; PL3 `rrf1000_b1600+FULL:h1_c100~s20+top400` | 0.8559 at 2,000.0; 0.8449 at 1,963.8 | 0.8527 at 2,000.0; 0.8585 at 1,946.3 | -0.58 LEVEL |
| MuSiQue | 2500 | NEW `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq`; PL3 `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8781 at 2,298.7; 0.8735 at 2,488.9 | 0.8821 at 2,274.9; 0.8788 at 2,455.1 | +0.33 LEVEL |
| MetaQA | 500 | `rrf1000_b100+STRUCT:h2_c25+top400` (both) | 0.5638 at 480.9 | 0.5484 at 479.6 | same cell |
| MetaQA | 1000 | `equal_rrf_budget_200+STRUCT:h2_c25` (both) | 0.6927 at 764.8 | 0.6899 at 763.7 | same cell |
| MetaQA | 1500 | `rrf1000_b25+FS:h2_c100_v5000` (both) | 0.7308 at 1,473.1 | 0.7289 at 1,473.7 | same cell |
| MetaQA | 2000 | `rrf1000_b25+STRUCT:h3_c25_v2000` (both) | 0.8938 at 1,999.3 | 0.8949 at 1,995.3 | same cell |
| MetaQA | 2500 | `rrf1000_b400+STRUCT:h3_c25_v2000` (both) | 0.8978 at 2,339.7 | 0.8968 at 2,335.9 | same cell |
| WebQSP | 500 | `dense_top200+FULL:h1_c100~s20+top400` (both) | 0.6802 at 377.7 | 0.6973 at 384.7 | same cell |
| WebQSP | 1000 | `rrf1000_b800+FULL:h1_c100~s20+top400` (both) | 0.7147 at 940.5 | 0.7265 at 947.7 | same cell |
| WebQSP | 1500 | `rrf1000_b100+FS:h2_c100_v5000+top1600` (both) | 0.7339 at 1,486.9 | 0.7299 at 1,486.4 | same cell |
| WebQSP | 2000 | `dense_top200+FS:h2_c100_v5000~cs-vr` (both) | 0.7571 at 1,891.5 | 0.7705 at 1,966.1 | same cell |
| WebQSP | 2500 | NEW `dense_top200+FULL:h2_c100_v5000~s10-cs-sp`; PL3 `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7731 at 2,442.4; 0.7731 at 2,442.4 | 0.7804 at 2,468.1; 0.7804 at 2,468.1 | +0.00 LEVEL |

**R98, beside the calls** (the same pairing at the 0.98 floor; decides nothing):

| dataset | NEW (R98 of all cells), eval | OLD (PL3's R98), eval | ALL difference, 95% CI | nodes ratio, 95% CI | status by the rule |
|---|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| SQuAD | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| HotpotQA | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| 2Wiki | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9656 at 2,211.2 | `rrf1000_b1600+FULL:h1_c100~s20+top800` 0.9656 at 2,085.7 | -0.01 [-0.16, +0.14] | 1.0602 [1.0589, 1.0616] | LEVEL |
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8821 at 2,274.9 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8788 at 2,455.1 | +0.33 [-0.29, +1.03] | 0.9266 [0.9253, 0.9280] | ABOVE |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` 0.7804 at 2,468.1 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7804 at 2,468.1 | +0.00 [+0.00, +0.00] | 1.0000 [1.0000, 1.0000] | LEVEL |

### 14.3 What each dataset says (dev = selection carve, frozen lists; eval = confirmation)

The construction each dataset would get from R95 of the reading `all` (NEW), and PL3's (OLD), eval frozen ALL at mean nodes, and the e3v re-run of NEW (flagged OVER_BOUND above 2,500 nodes):

| dataset | NEW (eval frozen ALL at nodes) | OLD = PL3's (eval frozen ALL at nodes) | NEW meets 0.95 on eval, frozen | NEW on e3v (ALL at nodes) |
|---|---|---|---|---|
| MultiHop-RAG | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9690 at 50.0 | yes | 0.9699 at 50.0 |
| SQuAD | `rrf1000_b15` 0.9554 at 15.0 | `rrf1000_b15` 0.9554 at 15.0 | yes | 0.9421 at 15.0 |
| HotpotQA | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9634 at 77.9 | yes | 0.9642 at 77.5 |
| 2Wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9609 at 917.0 | yes | 0.9684 at 916.1 |
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8821 at 2,274.9 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8788 at 2,455.1 | no (short by 6.79 points) | 0.8813 at 2,274.8 |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 | no (short by 5.32 points) | 0.9296 at 2,325.6 |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` 0.7804 at 2,468.1 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7804 at 2,468.1 | no (short by 16.96 points) | 0.7618 at 2,720.1 **OVER_BOUND** |

**Which datasets meet the target on eval, frozen lists (ALL >= 0.95), of the 7 read: 4**: MultiHop-RAG, SQuAD, HotpotQA, 2Wiki. **They do not:** MuSiQue (0.8821), MetaQA (0.8968), WebQSP (0.7804).
The e3v re-run of a chosen cell exceeds the 2,500 bound on: WebQSP. The bound is read on the selection carve's frozen lists, as declared (section 6).

- **MultiHop-RAG** (224 dev questions; reaches 0.95 on dev.) R95 is PL3's own cell, `rrf1000_b25+FULL:h1_c100+top25` (dev 0.9777 at 50.0; eval frozen 0.9690 at 50.0); no PL4 cell is a smaller or higher-ALL R95 pick. R98: `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9788 at 58.0 on eval (the same cell as PL3), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (the same cell at every budget).
- **SQuAD** (1,498 dev questions; reaches 0.95 on dev.) R95 is PL3's own cell, `rrf1000_b15` (dev 0.9519 at 15.0; eval frozen 0.9554 at 15.0); no PL4 cell is a smaller or higher-ALL R95 pick. R98: `rrf1000_b50+KNN:h1_c100+top25` 0.9818 at 63.0 on eval (the same cell as PL3), reaches 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (budgets with a different cell, eval ALL difference in points: 1000: +0.04; 2000: -0.01).
- **HotpotQA** (1,508 dev questions; reaches 0.95 on dev.) R95 is PL3's own cell, `rrf1000_b25+STRUCT:h1_c100~cs+top100` (dev 0.9622 at 77.3; eval frozen 0.9634 at 77.9); no PL4 cell is a smaller or higher-ALL R95 pick. R98: `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9783 at 153.5 on eval (the same cell as PL3), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (the same cell at every budget).
- **2Wiki** (1,496 dev questions; reaches 0.95 on dev.) R95 is PL3's own cell, `rrf1000_b400+FULL:h1_c100~s20+top800` (dev 0.9545 at 908.2; eval frozen 0.9609 at 917.0); no PL4 cell is a smaller or higher-ALL R95 pick. R98: `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9656 at 2,211.2 on eval (PL3: `rrf1000_b1600+FULL:h1_c100~s20+top800` 0.9656 at 2,085.7), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (budgets with a different cell, eval ALL difference in points: 2500: -0.01).
- **MuSiQue** (1,534 dev questions; SHORT on dev.) R95 of all cells: `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (dev 0.8781 at 2,298.7) against PL3's `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (dev 0.8735 at 2,488.9); on eval frozen 0.8821 at 2,274.9 against 0.8788 at 2,455.1: ALL +0.33 points [-0.33, +0.99], nodes ratio 0.9266 [0.9253, 0.9279], status **ABOVE**. R98: `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8821 at 2,274.9 on eval (PL3: `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8788 at 2,455.1), does not reach 0.98 on eval, status ABOVE. Frontier reading: **LEVEL** (budgets with a different cell, eval ALL difference in points: 2000: -0.58; 2500: +0.33).
  What moved: the cells carrying `eq` (equal hop shares) or `rr` (families taking turns; the R95 cell carries both, with `cs-vr`) reach dev ALL 0.8781 at 2,298.7 nodes, and cells carrying `sp` (support order) the same 0.8781 at 2,469.5, against PL3's 0.8735 at 2,488.9: 190 fewer nodes and 0.46 points more on dev. The dev frontier reads +1.11 points at 2,000 nodes and +0.46 at 2,500, 0 below 2,000. Row rank `rk` and seeds with the new options add nothing at R95. On eval the pair reads +0.33 points [-0.33, +0.99], an interval that includes 0, at 0.9266 of the nodes: the old rule's ABOVE rests on the price clause alone, as HotpotQA's did in PL3, and the frontier reading, which asks for ALL at a matched node count, finds no budget whose interval excludes 0 (2,000: -0.58 [-1.57, +0.50]; 2,500: +0.33 [-0.29, +0.95]). Still short by 6.79 points; the highest ALL at any size is unchanged (0.9355 at 5,783.4 nodes).
- **MetaQA** (1,497 dev questions; SHORT on dev.) R95 is PL3's own cell, `rrf1000_b400+STRUCT:h3_c25_v2000` (dev 0.8978 at 2,339.7; eval frozen 0.8968 at 2,335.9); no PL4 cell is a smaller or higher-ALL R95 pick. R98: `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8968 at 2,335.9 on eval (the same cell as PL3), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (the same cell at every budget).
  Nothing moved: no PL4 cell beats PL3's at any dev node count (the dev frontier gain is 0.00 at every budget); `~cs-sp` alone reaches 0.8303 at 2,284.0 and the other options are lower. The three-hop STRUCT walk's visited cap (2,000) is what holds the cell at 2,340 nodes: the options reorder or re-share what the cap lets in, and the cap decides.
- **WebQSP** (1,507 dev questions; SHORT on dev.) R95 of all cells: `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (dev 0.7731 at 2,442.4) against PL3's `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (dev 0.7731 at 2,442.4); on eval frozen 0.7804 at 2,468.1 against 0.7804 at 2,468.1: ALL +0.00 points [+0.00, +0.00], nodes ratio 1.0000 [1.0000, 1.0000], status **LEVEL**. R98: `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` 0.7804 at 2,468.1 on eval (PL3: `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7804 at 2,468.1), does not reach 0.98 on eval, status LEVEL. Frontier reading: **LEVEL** (budgets with a different cell, eval ALL difference in points: 2500: +0.00).
  The R95 pick `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` is a tie with PL3's `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (the same dev ALL and nodes; ties go to the name, and `sp` sorts before `vr`). The two cells have identical present-gold counts and identical pool sizes on every one of the 1,503 eval questions (paired ALL difference exactly 0.00 [0.00, 0.00], nodes ratio 1.0000), although the visited cap binds on 121 of them (sizes above 5,000): there the two orders can keep different nodes, and on this data they keep the same number of golds. No option moves the dev frontier at any budget; the best ALL at any size rises from 0.8480 to 0.8540 (at 6,021.1 nodes, `~cs-sp-eq`, more than twice the bound).

### 14.4 What each option bought

Read from `outputs/pl4/report.md` ("What each option bought" and "What the PL4 cells bought", section 14.9): on the 7 read, the dev frontier of all cells differs from PL3's only on MuSiQue
(+1.11 points at 2,000 nodes and +0.46 at 2,500 on dev, from `eq` and `rr` with `cs-vr`, and `sp`; on eval the 2,000-node pair reads -0.58 [-1.57, +0.50], so that dev gain does not carry, and the 2,500-node pair +0.33 [-0.29, +0.95]), on SQuAD (+0.07 points at three budgets of 75 to 1,000 nodes: one question's worth in 1,498), and on 2Wiki at one budget (+0.20 points at 2,500 nodes on dev, three questions of 1,496, from the `eq` cell `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq`, 0.9626 at 2,215.1 against PL3's 0.9606 at 2,078.2; on eval -0.01 [-0.14, +0.13], LEVEL). `rk` moves no read dataset's R95 pick (its cells alone reach 0.8638 on MuSiQue, 0.7862 on MetaQA, 0.7731 on WebQSP). The one SHORT dataset whose R95 pick moves is MuSiQue (190 nodes fewer, through `eq`, `rr`
and `sp` cells). Nothing moves MetaQA or WebQSP at any budget, and nothing moves MultiHop-RAG or HotpotQA. On SQuAD the R95 and R98 picks are PL3's; the +0.07 points are one question and read LEVEL on eval
(1,000 nodes: +0.04 [-0.05, +0.13]); on 2Wiki and HotpotQA the R95 pick is PL3's, and 2Wiki's R98 cell differs from PL3's but does not reach 0.98 and reads the same eval ALL (0.9656 on both, LEVEL).

### 14.5 What is still short, in numbers (diagnostics D1 and D2, post-hoc, labels read; 300 dev questions evenly spaced over the carve)

Only the SHORT datasets are diagnosed (the chosen R95 cell of the reading `all`). **D1** (each gold missing from the pool: R ranked past M, C cut by the caps or the order but inside the uncapped three-hop FULL reach of its seeds, U beyond it):

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (bound 2,500) |
|---|---|---|---|---|---|---|---|
| MuSiQue | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 35 of 300 (0.117) | 37 (1.06 per such question) | 0 | 37 | 0 | 2,301.4 |
| MetaQA | `rrf1000_b400+STRUCT:h3_c25_v2000` | 33 of 300 (0.110) | 375 (11.36 per such question) | 0 | 373 | 2 | 2,337.4 |
| WebQSP | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 63 of 300 (0.210) | 245 (3.89 per such question) | 0 | 245 | 0 | 2,421.1 |

**D2** (which stage cut each C gold, from a traced walk of the cell; the trace equals the engine's walk on every question):

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|
| MuSiQue | 37 | 0 | 13 | 6 | 9 | 7 | 2 | yes |
| MetaQA | 373 | 0 | 15 | 12 | 0 | 264 | 82 | yes |
| WebQSP | 245 | 126 | 97 | 0 | 0 | 0 | 22 | yes |

- **The shortfall is not the ranking and not the reach**: R is 0 on every SHORT dataset (these cells carry no `+top{M}` cut) and U is at most 2 golds. It is C: the caps and the visited budget.
- **MetaQA** (STRUCT, three hops, visited cap 2,000): most C golds are cut by the **budget at hop 3** (264 of 373: the hop-3 candidates the visited cap does not admit), 82 are upstream
  (reached only through a node an earlier hop dropped) and 27 are cut by a per-frontier cap. `eq` (equal hop shares), the one option that changes the shares, did not help (its cells alone reach 0.8297 on dev against 0.8978).
- **WebQSP** (FULL, two hops, per-seed and per-frontier caps of 100, visited cap 5,000): D2 attributes none of its 245 C golds to the visited budget; they are cut by the **per-seed cap at hop 1** (126) and the
  **per-frontier cap at hop 2** (97), then upstream (22). The next change here is a cap that follows the budget (spend what the bound leaves on the next-ranked neighbours of each seed or frontier node), not an order.
- **MuSiQue** (FULL, three hops): the 37 C golds are spread over the cap at hop 2 (13) and hop 3 (6) and the budget at hop 2 (9) and hop 3 (7), with 2 upstream: no one stage dominates, so no single change is aimed at all of it.
- A cap that follows the budget spends more nodes: WebQSP's chosen cell is at 2,442 dev and 2,468 eval nodes against the 2,500 bound, so it has almost no room (the bound is the price of the next change there); MuSiQue's, at
  2,299 on dev and 2,275 on eval, has about 200.

### 14.6 Beside the calls: the e3v lists (post-hoc; decides nothing)

NEW (R95 of all cells) and OLD (PL3's) re-run on the eval carve's e3v lists. The mean nodes are flagged OVER_BOUND when above 2,500 (the bound is read on the frozen lists; the flag changes neither call):

| dataset | NEW, e3v (ALL at nodes) | OLD = PL3's, e3v (ALL at nodes) | ALL difference | NEW meets 0.95 |
|---|---|---|---|---|
| MultiHop-RAG | 0.9699 at 50.0 | 0.9699 at 50.0 | +0.00 | yes |
| SQuAD | 0.9421 at 15.0 | 0.9421 at 15.0 | +0.00 | no |
| HotpotQA | 0.9642 at 77.5 | 0.9642 at 77.5 | +0.00 | yes |
| 2Wiki | 0.9684 at 916.1 | 0.9684 at 916.1 | +0.00 | yes |
| MuSiQue | 0.8813 at 2,274.8 | 0.8775 at 2,457.0 | +0.37 | no |
| MetaQA | 0.9296 at 2,325.6 | 0.9296 at 2,325.6 | +0.00 | no |
| WebQSP | 0.7618 at 2,720.1 **OVER_BOUND** | 0.7618 at 2,720.1 **OVER_BOUND** | +0.00 | no |

On the e3v lists the chosen cell of WebQSP reads above 2,500 mean nodes (OVER_BOUND); under section 6 the flag changes neither call, and a flagged cell is not offered to PL1's e3v pool rows by this page. WebQSP's PL3 cell carries the same flag with the same sizes, as PL3 filed (2,720 nodes).

### 14.7 What the calls mean, and what they leave unread

- **At 7 of 7 read the old rule is CARRY_PL4 and the frontier reading is NOT_CARRIED.** The old rule's call rests on one ABOVE, MuSiQue's, which is by the price clause alone (the ALL interval includes 0) and on no BELOW; the frontier reading finds no budget of any dataset ABOVE (a lower end above 0) and none BELOW. The two are filed under their own names; this page does not say which governs (section 6): which one governs is the user's decision.
- **A reading of what the read datasets say.** On six of the seven datasets PL4's cells are PL3's cells or tie them at R95 (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki and MetaQA: the same R95 cell; WebQSP: identical ALL and sizes on every eval question); on MuSiQue they save 7.3 percent of the nodes for a gain whose interval includes 0. HotpotQA and 2Wiki, the two host rows, agree with the five laptop datasets: both reach 0.95 inside the bound with PL3's own cell (HotpotQA 0.9634 at 77.9 nodes, 2Wiki 0.9609 at 917.0), no PL4 cell moves HotpotQA's dev frontier at any of the 14 budgets, and 2Wiki's moves at 2,500 nodes only (+0.20 points on dev, three questions of 1,496; on eval -0.01 [-0.14, +0.13], LEVEL). None of the three SHORT datasets is closer to 0.95 in any way the rule can credit: MuSiQue 0.8821 (+0.33 points), MetaQA 0.8968 (0.00), WebQSP 0.7804 (0.00).
- **D2 against the four options.** `sp`, `rk`, `eq` and `rr` reorder or re-share the visited budget and leave the per-seed and per-frontier caps alone. D2 puts all of WebQSP's attributed C golds (223 of 245; 22 upstream) at those
  caps, which no option touches, and most of MetaQA's (264 of 373) at hop 3's budget, where `eq` bought nothing. That was the disclosed limit of this round (section 13, "the four options do not change the per-seed or
  per-frontier caps"); it is now measured.

### 14.8 The next changes (names only; none is adopted, none is in this round's grid; each is declared on its own page before any number)

Against the caps (WebQSP, MuSiQue): **a cap that follows the budget** (each seed and frontier node keeps its next-ranked neighbours until the visited budget is spent, instead of a fixed number); against the hop-3 budget
(MetaQA, MuSiQue): **a parent-conditioned rank** (a node's cosine to the question plus its best parent's, the bridge the question text does not name; section 12), and the **weight values of NER and KNN rows** as a key.
Label-ordered caps need a UL2 layout on all seven (section 3). The e3v-bounded selection needs e3v lists for the select carves (section 12).

### 14.9 The tables (the report, as of 7 of 7 read)

#### Gates and cost

| dataset | G-R (PL2's 804 + PL3's 1,470 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |
|---|---|---|
| multihoprag | PASS (224 q x 2274 cells = 509,376 elements, mismatches 0/0) | 3, 82 s, 0.28 GB |
| squad | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 7,012 s, 0.12 GB |
| hotpotqa | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 4,210 s, 14.95 GB |
| 2wiki | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 7,610 s, 17.35 GB |
| musique | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 9,488 s, 0.64 GB |
| metaqa | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 2,392 s, 0.4 GB |
| webqsp | PASS (400 q x 2274 cells = 909,600 elements, mismatches 0/0) | 3, 8,308 s, 5.46 GB |

Self-test: PASS (cells PASS, G-F PASS, G-D PASS; engine multihoprag PASS 552, squad PASS 552; G-X4 PASS, PASS); code-path sha256 `aa7c9ac9...`.

#### Dev reading (the rule's own numbers, dev carve, frozen lists): PL3's chosen cell (the incumbent) against the grid of all 4,206 cells

| dataset | dev q | PL3 R95 (PL2 U PL3 cells) | R95 (all cells) | PL3 R98 | R98 (all cells) | R95 of the new cells only | PL3's pick re-read = its filed pick |
|---|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` (0.9955 at 58.6) | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` (0.9955 at 58.6) | True |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b15` (0.9519 at 15.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-rk+top50` (0.9780 at 71.9) | True |
| hotpotqa | 1508 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` (0.9622 at 77.3) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b100+STRUCT:h1_c100~cs+top200` (0.9814 at 152.7) | `rrf1000_b25+FULL:h3_c25_v2000~rk+top800` (0.9503 at 825.0) | True |
| 2wiki | 1496 | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b400+FULL:h1_c100~s20+top800` (0.9545 at 908.2) | `rrf1000_b1600+FULL:h1_c100~s20+top800` (0.9606 at 2,078.2, SHORT) | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (0.9626 at 2,215.1, SHORT) | `rrf1000_b400+FULL:h3_c25_v2000~rk+top1600` (0.9505 at 2,000.0) | True |
| musique | 1534 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (0.8735 at 2,488.9, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (0.8781 at 2,298.7, SHORT) | True |
| metaqa | 1497 | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000` (0.8978 at 2,339.7, SHORT) | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` (0.8303 at 2,284.0, SHORT) | True |
| webqsp | 1507 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (0.7731 at 2,442.4, SHORT) | True |

#### The chosen cell at 0.95 (R95) and its confirmation on eval: PL3's against the grid of all cells (and the new cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL3 | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag all | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) |
| multihoprag new only | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9955 | 58.6 | 0.9788 (58.0) | 0.9779 (58.2) |
| squad PL3 | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT |
| squad all | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT |
| squad new only | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-rk+top50` | 0.9780 | 71.9 | 0.9773 (72.8) | 0.9661 (72.3) |
| hotpotqa PL3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa all | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9622 | 77.3 | 0.9634 (77.9) | 0.9642 (77.5) |
| hotpotqa new only | `rrf1000_b25+FULL:h3_c25_v2000~rk+top800` | 0.9503 | 825.0 | 0.9568 (825.0) | 0.9611 (825.0) |
| 2wiki PL3 | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki all | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9545 | 908.2 | 0.9609 (917.0) | 0.9684 (916.1) |
| 2wiki new only | `rrf1000_b400+FULL:h3_c25_v2000~rk+top1600` | 0.9505 | 2,000.0 | 0.9491 (2,000.0) SHORT | 0.9719 (2,000.0) |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | 0.8775 (2,457.0) SHORT |
| musique all | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| musique new only | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa new only | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` (SHORT) | 0.8303 | 2,284.0 | 0.8262 (2,281.6) SHORT | 0.8400 (2,238.7) SHORT |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp all | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp new only | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |

#### The chosen cell at 0.98 (R98) and its confirmation on eval: PL3's against the grid of all cells (and the new cells alone)

| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag PL3 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag all | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| multihoprag new only | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9955 | 58.6 | 0.9788 (58.0) SHORT | 0.9779 (58.2) SHORT |
| squad PL3 | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| squad all | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| squad new only | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9820 | 74.9 | 0.9800 (75.0) SHORT | 0.9698 (75.0) SHORT |
| hotpotqa PL3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa all | `rrf1000_b100+STRUCT:h1_c100~cs+top200` | 0.9814 | 152.7 | 0.9783 (153.5) SHORT | 0.9753 (153.3) SHORT |
| hotpotqa new only | `rrf1000_b800+FULL:h3_c25_v2000~rk+top800` | 0.9801 | 1,600.0 | 0.9780 (1,600.0) SHORT | 0.9792 (1,600.0) SHORT |
| 2wiki PL3 | `rrf1000_b1600+FULL:h1_c100~s20+top800` (SHORT) | 0.9606 | 2,078.2 | 0.9656 (2,085.7) SHORT | 0.9716 (2,088.6) SHORT |
| 2wiki all | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| 2wiki new only | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` (SHORT) | 0.9626 | 2,215.1 | 0.9656 (2,211.2) SHORT | 0.9721 (2,225.5) SHORT |
| musique PL3 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` (SHORT) | 0.8735 | 2,488.9 | 0.8788 (2,455.1) SHORT | 0.8775 (2,457.0) SHORT |
| musique all | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| musique new only | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` (SHORT) | 0.8781 | 2,298.7 | 0.8821 (2,274.9) SHORT | 0.8813 (2,274.8) SHORT |
| metaqa PL3 | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa all | `rrf1000_b400+STRUCT:h3_c25_v2000` (SHORT) | 0.8978 | 2,339.7 | 0.8968 (2,335.9) SHORT | 0.9296 (2,325.6) SHORT |
| metaqa new only | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` (SHORT) | 0.8303 | 2,284.0 | 0.8262 (2,281.6) SHORT | 0.8400 (2,238.7) SHORT |
| webqsp PL3 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp all | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |
| webqsp new only | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` (SHORT) | 0.7731 | 2,442.4 | 0.7804 (2,468.1) SHORT | 0.7618 (2,720.1) SHORT OVER_BOUND |

#### The filed own construction on eval (PL2's OLD cell, re-read by this engine)

| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|
| multihoprag | `equal_rrf_budget_50` | 0.9540 (50.0) | 0.9451 (50.0) |
| squad | `equal_rrf_budget_50` | 0.9798 (50.0) | 0.9736 (50.0) |
| hotpotqa | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9684 (93.7) | 0.9646 (92.2) |
| 2wiki | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9058 (105.6) | 0.9310 (101.6) |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | 0.7911 (2,087.2) |
| metaqa | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | 0.9278 (2,005.1) |
| webqsp | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | 0.6447 (2,124.5) |

#### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL3's grid / all cells

| dataset | 0.95: PL3's grid | 0.95: all cells | 0.98: PL3's grid | 0.98: all cells |
|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 | `rrf1000_b25+FS:h3_c25_v2000~cs-vr+top50` 0.9955 at 58.6 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b15` 0.9519 at 15.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b25+STRUCT:h1_c100~cs+top100` 0.9622 at 77.3 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 | `rrf1000_b100+STRUCT:h1_c100~cs+top200` 0.9814 at 152.7 |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | `rrf1000_b400+FULL:h1_c100~s20+top800` 0.9545 at 908.2 | none (best 0.9719 at 6,230.3) | none (best 0.9786 at 6,277.7) |
| musique | none (best 0.9355 at 5,783.4) | none (best 0.9355 at 5,783.4) | none (best 0.9355 at 5,783.4) | none (best 0.9355 at 5,783.4) |
| metaqa | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | `rrf1000_b25+STRUCT:h3_c100_v5000` 0.9613 at 4,983.6 | none (best 0.9653 at 6,268.2) | none (best 0.9653 at 6,268.2) |
| webqsp | none (best 0.8480 at 6,067.3) | none (best 0.8540 at 6,021.1) | none (best 0.8480 at 6,067.3) | none (best 0.8540 at 6,021.1) |

#### The highest ALL on dev (the bound is 2,500 mean nodes): PL3's grid / all cells, within the bound and at any size

| dataset | PL3's grid, within the bound | all cells, within the bound | PL3's grid, any size | all cells, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+NER:h3_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs-vr+top800` 1.0000 at 2,190.4 |
| hotpotqa | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` 0.9960 at 2,246.8 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 | `rrf1000_b800+STRUCT:h3_c100_v5000~cs-vr` 0.9980 at 5,441.6 |
| 2wiki | `rrf1000_b1600+FULL:h1_c100~s20+top800` 0.9606 at 2,078.2 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` 0.9626 at 2,215.1 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9719 at 6,230.3 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-rk` 0.9786 at 6,277.7 |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` 0.8735 at 2,488.9 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` 0.8781 at 2,298.7 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 | `rrf1000_b1600+FULL:h2_c100_v5000~s20-cs-vr` 0.9355 at 5,783.4 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b400+STRUCT:h3_c25_v2000` 0.8978 at 2,339.7 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 | `rrf1000_b1600+STRUCT:h3_c100_v5000` 0.9653 at 6,268.2 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` 0.7731 at 2,442.4 | `rrf1000_b1600+FULL:h3_c100_v5000~vr` 0.8480 at 6,067.3 | `rrf1000_b1600+FULL:h3_c100_v5000~cs-sp-eq` 0.8540 at 6,021.1 |

#### What the PL4 cells bought (dev frontier of all cells minus PL3's frontier, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 400 | 500 | 750 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / 0.8973 (+0.00) | 0.9777 / 0.9777 (+0.00) | 0.9955 / 0.9955 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / 0.9666 (+0.00) | 0.9793 / 0.9793 (+0.00) | 0.9846 / 0.9853 (+0.07) | 0.9887 / 0.9887 (+0.00) | 0.9920 / 0.9927 (+0.07) | 0.9933 / 0.9933 (+0.00) | 0.9947 / 0.9947 (+0.00) | 0.9953 / 0.9953 (+0.00) | 0.9960 / 0.9960 (+0.00) | 0.9967 / 0.9967 (+0.00) | 0.9973 / 0.9980 (+0.07) | 0.9987 / 0.9987 (+0.00) | 0.9993 / 0.9993 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| hotpotqa | 0.6127 / 0.6127 (+0.00) | 0.9224 / 0.9224 (+0.00) | 0.9496 / 0.9496 (+0.00) | 0.9682 / 0.9682 (+0.00) | 0.9788 / 0.9788 (+0.00) | 0.9814 / 0.9814 (+0.00) | 0.9834 / 0.9834 (+0.00) | 0.9834 / 0.9834 (+0.00) | 0.9887 / 0.9887 (+0.00) | 0.9901 / 0.9901 (+0.00) | 0.9927 / 0.9927 (+0.00) | 0.9934 / 0.9934 (+0.00) | 0.9940 / 0.9940 (+0.00) | 0.9960 / 0.9960 (+0.00) |
| 2wiki | 0.3195 / 0.3195 (+0.00) | 0.7172 / 0.7172 (+0.00) | 0.8342 / 0.8342 (+0.00) | 0.8890 / 0.8890 (+0.00) | 0.8984 / 0.8984 (+0.00) | 0.9078 / 0.9078 (+0.00) | 0.9164 / 0.9164 (+0.00) | 0.9298 / 0.9298 (+0.00) | 0.9325 / 0.9325 (+0.00) | 0.9452 / 0.9452 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9552 / 0.9552 (+0.00) | 0.9606 / 0.9626 (+0.20) |
| musique | 0.2979 / 0.2979 (+0.00) | 0.4146 / 0.4146 (+0.00) | 0.4733 / 0.4733 (+0.00) | 0.5261 / 0.5261 (+0.00) | 0.5906 / 0.5906 (+0.00) | 0.6349 / 0.6349 (+0.00) | 0.6734 / 0.6734 (+0.00) | 0.6956 / 0.6956 (+0.00) | 0.7373 / 0.7373 (+0.00) | 0.7673 / 0.7673 (+0.00) | 0.7947 / 0.7947 (+0.00) | 0.8246 / 0.8246 (+0.00) | 0.8449 / 0.8559 (+1.11) | 0.8735 / 0.8781 (+0.46) |
| metaqa | 0.0147 / 0.0147 (+0.00) | 0.2325 / 0.2325 (+0.00) | 0.3039 / 0.3039 (+0.00) | 0.3267 / 0.3267 (+0.00) | 0.3280 / 0.3280 (+0.00) | 0.3300 / 0.3300 (+0.00) | 0.3641 / 0.3641 (+0.00) | 0.3681 / 0.3681 (+0.00) | 0.5638 / 0.5638 (+0.00) | 0.6887 / 0.6887 (+0.00) | 0.6927 / 0.6927 (+0.00) | 0.7308 / 0.7308 (+0.00) | 0.8938 / 0.8938 (+0.00) | 0.8978 / 0.8978 (+0.00) |
| webqsp | 0.1526 / 0.1526 (+0.00) | 0.3676 / 0.3676 (+0.00) | 0.4625 / 0.4625 (+0.00) | 0.5601 / 0.5601 (+0.00) | 0.5886 / 0.5886 (+0.00) | 0.6058 / 0.6058 (+0.00) | 0.6603 / 0.6603 (+0.00) | 0.6802 / 0.6802 (+0.00) | 0.6802 / 0.6802 (+0.00) | 0.6855 / 0.6855 (+0.00) | 0.7147 / 0.7147 (+0.00) | 0.7339 / 0.7339 (+0.00) | 0.7571 / 0.7571 (+0.00) | 0.7731 / 0.7731 (+0.00) |

(each entry: PL3's frontier / the frontier of all cells (points bought); `-` = no cell is that small)

#### What each option bought: R95 of PL3's cells plus that option's cells (dev), and the best ALL inside the bound among that option's cells alone

| dataset | PL3 R95 | support order | row-rank order | equal hop shares | families take turns | seeds 10/20 with a new option |
|---|---|---|---|---|---|---|
| multihoprag | 0.9777 at 50.0 | = PL3; alone best 1.0000 at 123.2 | = PL3; alone best 1.0000 at 123.2 | = PL3; alone best 1.0000 at 111.5 | = PL3; alone best 1.0000 at 123.2 | = PL3; alone best 1.0000 at 123.4 |
| squad | 0.9519 at 15.0 | = PL3; alone best 0.9993 at 1,940.5 | = PL3; alone best 0.9993 at 1,600.0 | = PL3; alone best 0.9993 at 1,700.0 | = PL3; alone best 0.9993 at 1,700.0 | = PL3; alone best 0.9993 at 1,944.7 |
| hotpotqa | 0.9622 at 77.3 | = PL3; alone best 0.9927 at 2,172.6 | = PL3; alone best 0.9934 at 2,327.3 | = PL3; alone best 0.9947 at 2,464.0 | = PL3; alone best 0.9940 at 2,467.1 | = PL3; alone best 0.9934 at 2,408.3 |
| 2wiki | 0.9545 at 908.2 | = PL3; alone best 0.9579 at 2,288.7 | = PL3; alone best 0.9592 at 2,318.3 | = PL3; alone best 0.9626 at 2,215.1 | = PL3; alone best 0.9592 at 2,318.4 | = PL3; alone best 0.9626 at 2,215.1 |
| musique | 0.8735 at 2,488.9 SHORT | 0.8781 at 2,469.5 SHORT; alone best 0.8781 at 2,469.5 | = PL3; alone best 0.8638 at 2,393.8 | 0.8781 at 2,298.7 SHORT; alone best 0.8781 at 2,298.7 | 0.8781 at 2,298.7 SHORT; alone best 0.8781 at 2,298.7 | = PL3; alone best 0.8709 at 2,263.9 |
| metaqa | 0.8978 at 2,339.7 SHORT | = PL3; alone best 0.8303 at 2,284.0 | = PL3; alone best 0.7862 at 2,284.1 | = PL3; alone best 0.8297 at 2,284.0 | = PL3; alone best 0.8003 at 2,260.9 | = PL3; alone best 0.7428 at 2,286.6 |
| webqsp | 0.7731 at 2,442.4 SHORT | 0.7731 at 2,442.4 SHORT; alone best 0.7731 at 2,442.4 | = PL3; alone best 0.7731 at 2,442.5 | = PL3; alone best 0.7379 at 2,498.1 | = PL3; alone best 0.7624 at 2,288.5 | 0.7731 at 2,442.4 SHORT; alone best 0.7731 at 2,442.4 |

#### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

#### The call, the old rule as PL3 states it (R95 of all cells against PL3's R95): CARRY_PL4 (read 7 of 7: 2wiki, hotpotqa, metaqa, multihoprag, musique, squad, webqsp; BELOW 0, ABOVE 1, LEVEL 6)

| dataset | status | NEW (R95, all cells) | NEW ALL (nodes) | OLD (PL3 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL (same cell) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| squad | LEVEL (same cell) | `rrf1000_b15` | 0.9554 (15.0) | `rrf1000_b15` | 0.9554 (15.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| hotpotqa | LEVEL (same cell) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 0.9634 (77.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| 2wiki | LEVEL (same cell) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | `rrf1000_b400+FULL:h1_c100~s20+top800` | 0.9609 (917.0) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| musique | ABOVE | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 (2,274.9) | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8788 (2,455.1) | +0.33 [-0.33, +0.99] | 0.927 [0.925, 0.928] |
| metaqa | LEVEL (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |
| webqsp | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 (2,468.1) | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | +0.00 [+0.00, +0.00] | 1.000 [1.000, 1.000] |

Beside the call (decide nothing): R98 and the new-cells-only pick against PL3's R95.

| dataset | R98 status | R98 NEW ALL (nodes) vs PL3 | new-only status | new-only cell | ALL (nodes) |
|---|---|---|---|---|---|
| multihoprag | LEVEL | 0.9788 (58.0) vs 0.9788 (58.0) | LEVEL | `rrf1000_b25+FS:h3_c25_v2000~cs-vr-eq+top50` | 0.9788 (58.0) |
| squad | LEVEL | 0.9818 (63.0) vs 0.9818 (63.0) | LEVEL | `rrf1000_b25+STRUCT:h3_c25_v2000~cs-rk+top50` | 0.9773 (72.8) |
| hotpotqa | LEVEL | 0.9783 (153.5) vs 0.9783 (153.5) | BELOW | `rrf1000_b25+FULL:h3_c25_v2000~rk+top800` | 0.9568 (825.0) |
| 2wiki | LEVEL | 0.9656 (2,211.2) vs 0.9656 (2,085.7) | BELOW | `rrf1000_b400+FULL:h3_c25_v2000~rk+top1600` | 0.9491 (2,000.0) |
| musique | ABOVE | 0.8821 (2,274.9) vs 0.8788 (2,455.1) | ABOVE | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 0.8821 (2,274.9) |
| metaqa | LEVEL | 0.8968 (2,335.9) vs 0.8968 (2,335.9) | ABOVE | `rrf1000_b400+STRUCT:h3_c25_v2000~cs-sp` | 0.8262 (2,281.6) |
| webqsp | LEVEL | 0.7804 (2,468.1) vs 0.7804 (2,468.1) | LEVEL | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 0.7804 (2,468.1) |

#### The call, the frontier reading (matched node counts, proposed on the page before any number): NOT_CARRIED (BELOW 0, ABOVE 0, LEVEL 7)

Per dataset and budget N: the cell of all cells with the highest dev ALL at most N dev nodes against the same for PL3's grid; eval, frozen lists; paired bootstrap of the ALL difference.

| dataset | frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |
|---|---|---|---|---|---|---|---|---|
| multihoprag | LEVEL | 500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 1500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2000 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| multihoprag | LEVEL | 2500 | `rrf1000_b50+FULL:h2_c100_v5000+top50` (same cell) | `rrf1000_b50+FULL:h2_c100_v5000+top50` | 0.9912 (100.0) | 0.9912 (100.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 500 | `rrf1000_b400+STRUCT:h1_c100~cs+top50` (same cell) | `rrf1000_b400+STRUCT:h1_c100~cs+top50` | 0.9965 (446.7) | 0.9965 (446.7) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 1000 | `rrf1000_b100+FULL:h3_c100_v5000~cs-vr-eq+top800` | `rrf1000_b400+STRUCT:h3_c100_v5000~cs-vr+brg400` | 0.9978 (900.0) | 0.9974 (793.3) | +0.04 [-0.05, +0.13] | LEVEL |
| squad | LEVEL | 1500 | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` (same cell) | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` | 0.9982 (1,200.0) | 0.9982 (1,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| squad | LEVEL | 2000 | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top200` | 0.9991 (1,599.9) | 0.9992 (1,740.6) | -0.01 [-0.05, +0.03] | LEVEL |
| squad | LEVEL | 2500 | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` (same cell) | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | 0.9992 (2,200.0) | 0.9992 (2,200.0) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 500 | `rrf1000_b100+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b100+FULL:h1_c100~s20+top800` | 0.9854 (460.9) | 0.9854 (460.9) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~cs+top400` (same cell) | `rrf1000_b800+FULL:h1_c100~cs+top400` | 0.9896 (931.2) | 0.9896 (931.2) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 1500 | `rrf1000_b800+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.9923 (1,112.4) | 0.9923 (1,112.4) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 2000 | `rrf1000_b1600+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9945 (1,883.6) | 0.9945 (1,883.6) | +0.00 [+0.00, +0.00] | LEVEL |
| hotpotqa | LEVEL | 2500 | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` (same cell) | `rrf1000_b800+STRUCT:h2_c100_v5000~cs-vr` | 0.9918 (2,226.5) | 0.9918 (2,226.5) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 500 | `rrf1000_b200+FULL:h1_c100+top400` (same cell) | `rrf1000_b200+FULL:h1_c100+top400` | 0.9360 (421.6) | 0.9360 (421.6) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 1500 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2000 | `rrf1000_b400+FULL:h1_c100~s20` (same cell) | `rrf1000_b400+FULL:h1_c100~s20` | 0.9614 (920.4) | 0.9614 (920.4) | +0.00 [+0.00, +0.00] | LEVEL |
| 2wiki | LEVEL | 2500 | `rrf1000_b400+FULL:h3_c25_v2000~s20-cs-vr-eq` | `rrf1000_b1600+FULL:h1_c100~s20+top800` | 0.9656 (2,211.2) | 0.9656 (2,085.7) | -0.01 [-0.14, +0.13] | LEVEL |
| musique | LEVEL | 500 | `rrf1000_b400+FULL:h1_c100+brg100` (same cell) | `rrf1000_b400+FULL:h1_c100+brg100` | 0.7683 (498.1) | 0.7683 (498.1) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~s10+top200` (same cell) | `rrf1000_b800+FULL:h1_c100~s10+top200` | 0.8213 (986.6) | 0.8213 (986.6) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 1500 | `rrf1000_b800+FULL:h1_c100~s20+top800` (same cell) | `rrf1000_b800+FULL:h1_c100~s20+top800` | 0.8395 (1,357.1) | 0.8395 (1,357.1) | +0.00 [+0.00, +0.00] | LEVEL |
| musique | LEVEL | 2000 | `rrf1000_b400+FULL:h3_c25_v2000~cs-sp-eq+top1600` | `rrf1000_b1600+FULL:h1_c100~s20+top400` | 0.8527 (2,000.0) | 0.8585 (1,946.3) | -0.58 [-1.57, +0.50] | LEVEL |
| musique | LEVEL | 2500 | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr` | 0.8821 (2,274.9) | 0.8788 (2,455.1) | +0.33 [-0.29, +0.95] | LEVEL |
| metaqa | LEVEL | 500 | `rrf1000_b100+STRUCT:h2_c25+top400` (same cell) | `rrf1000_b100+STRUCT:h2_c25+top400` | 0.5484 (479.6) | 0.5484 (479.6) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1000 | `equal_rrf_budget_200+STRUCT:h2_c25` (same cell) | `equal_rrf_budget_200+STRUCT:h2_c25` | 0.6899 (763.7) | 0.6899 (763.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 1500 | `rrf1000_b25+FS:h2_c100_v5000` (same cell) | `rrf1000_b25+FS:h2_c100_v5000` | 0.7289 (1,473.7) | 0.7289 (1,473.7) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2000 | `rrf1000_b25+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b25+STRUCT:h3_c25_v2000` | 0.8949 (1,995.3) | 0.8949 (1,995.3) | +0.00 [+0.00, +0.00] | LEVEL |
| metaqa | LEVEL | 2500 | `rrf1000_b400+STRUCT:h3_c25_v2000` (same cell) | `rrf1000_b400+STRUCT:h3_c25_v2000` | 0.8968 (2,335.9) | 0.8968 (2,335.9) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 500 | `dense_top200+FULL:h1_c100~s20+top400` (same cell) | `dense_top200+FULL:h1_c100~s20+top400` | 0.6973 (384.7) | 0.6973 (384.7) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 1000 | `rrf1000_b800+FULL:h1_c100~s20+top400` (same cell) | `rrf1000_b800+FULL:h1_c100~s20+top400` | 0.7265 (947.7) | 0.7265 (947.7) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 1500 | `rrf1000_b100+FS:h2_c100_v5000+top1600` (same cell) | `rrf1000_b100+FS:h2_c100_v5000+top1600` | 0.7299 (1,486.4) | 0.7299 (1,486.4) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 2000 | `dense_top200+FS:h2_c100_v5000~cs-vr` (same cell) | `dense_top200+FS:h2_c100_v5000~cs-vr` | 0.7705 (1,966.1) | 0.7705 (1,966.1) | +0.00 [+0.00, +0.00] | LEVEL |
| webqsp | LEVEL | 2500 | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | `dense_top200+FULL:h2_c100_v5000~s10-cs-vr` | 0.7804 (2,468.1) | 0.7804 (2,468.1) | +0.00 [+0.00, +0.00] | LEVEL |

#### The 2,500 bound on e3v lists (selection reads the frozen dev lists; this is the same cell re-run on the eval carve's e3v lists)

| dataset | cell (R95, all cells) | nodes: dev frozen | eval frozen | eval e3v | e3v over the bound |
|---|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 50.0 | 50.0 | 50.0 | within |
| squad | `rrf1000_b15` | 15.0 | 15.0 | 15.0 | within |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100~cs+top100` | 77.3 | 77.9 | 77.5 | within |
| 2wiki | `rrf1000_b400+FULL:h1_c100~s20+top800` | 908.2 | 917.0 | 916.1 | within |
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 2,298.7 | 2,274.9 | 2,274.8 | within |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 2,339.7 | 2,335.9 | 2,325.6 | within |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 2,442.4 | 2,468.1 | 2,720.1 | OVER_BOUND |

#### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)

| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |
|---|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 300 | 35 (0.117) | 37 | 1.06 | 0 (q 0) | 37 (q 35) | 0 (q 0) | 0 | 2,301.4 |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 300 | 33 (0.110) | 375 | 11.36 | 0 (q 0) | 373 (q 32) | 2 (q 2) | 0 | 2,337.4 |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 300 | 63 (0.210) | 245 | 3.89 | 0 (q 0) | 245 (q 63) | 0 (q 0) | 0 | 2,421.1 |

#### D2: which stage cut the C golds (post-hoc; labels read; the cell's walk traced hop by hop)

| dataset | cell | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|---|
| musique | `rrf1000_b800+FULL:h3_c25_v2000~cs-vr-rr-eq` | 37 | 0 | 13 | 6 | 9 | 7 | 2 | True |
| metaqa | `rrf1000_b400+STRUCT:h3_c25_v2000` | 373 | 0 | 15 | 12 | 0 | 264 | 82 | True |
| webqsp | `dense_top200+FULL:h2_c100_v5000~s10-cs-sp` | 245 | 126 | 97 | 0 | 0 | 0 | 22 | True |

### 14.10 Surprises and contradictions with filed numbers

- WebQSP's R95 pick changed only by a name tie (`sp` against `vr` at the same dev ALL and nodes); the paired eval difference is exactly 0 and the nodes ratio exactly 1. A reader should not take `~s10-cs-sp` as a new finding.
- `rk` (row-rank order), which PL4 declared as the order most unlike PL3's, moves no dataset's R95 pick; the only frontier cell that carries it is SQuAD's 2,000-node pick (-0.01 points on eval, LEVEL).
- MuSiQue's old-rule ABOVE is the price clause with an ALL interval that includes 0: the rule says ABOVE, the frontier reading says LEVEL at every budget. This page reports both.

### 14.11 Disclosed and undone

- The laptop's SQuAD frozen confirmation was started as the single-process `zpl4g.py confirm` (the lane script's step), was projected at about three hours, and was stopped by hand at about 1,400 of 11,873 questions
  (it wrote nothing); it was replaced by `zpl4r.py confirm-par --slices 4`, which evaluates four question slices with the engine's own `evaluate4`, concatenates the arrays and runs `cmd_confirm`'s own tail. The
  concatenated arrays were checked equal to the single-process confirmation on MultiHop-RAG (`--verify`). `chain4.log` records the stop as `FAIL confirm squad` at 11:06.
- D1/D2 ran on 300 dev questions chosen evenly over the carve (as PL3's D1), post-hoc, and decide nothing. `zpl4r.py` (the report and the diagnostics, not the engine) is committed with this section.
- The frontier reading is a change of the rule proposed in this page's declaration; it is filed beside the old call, and neither is chosen here.
- This section was first filed at c25eddf with five of the seven read (the call INCOMPLETE on both readings) and is re-filed here from the same records plus HotpotQA's and 2Wiki's host rows (`pl4-hotpotqag*`, `pl4-hotpotqace`, `pl4-hotpotqacf`, `pl4-2wikig*`, `pl4-2wikisel`, `pl4-2wikicf`, `pl4-2wikice`); the five laptop datasets' numbers are unchanged from that filing, and the hand-written sentences of 14.4 and 14.7 were rewritten for seven.
- The laptop's grids and confirmations ran with up to eight heavy processes on 12 logical CPUs; seconds are upper bounds.
- The grid arrays (`*.npz`) are not committed (as PL2's and PL3's); the heads, selects, confirmations, the call, the diagnostics and the report are.
- Not done in this filing: PL1's rebuild of the pools on a carried construction (PL1's rows); the next changes of 14.8.
