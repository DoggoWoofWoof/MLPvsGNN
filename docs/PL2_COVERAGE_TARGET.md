# PL2: the coverage target, the smallest pools that hold every gold (round 1 of docs/PL1_POOLS_ON_UL2_LAYOUT.md section 7)

Declared 25 November 2026, before any PL2 number exists. The numbers I have read are the FILED ones (outputs/m3a/headroom,
outputs/m3b/contract, outputs/pl1/multihoprag/RK/s0/knee_*.json); nothing below was computed after the page's rule was
chosen. **Nothing here trains, no model is fitted, no LLM or spaCy is used, and every pool function is gold-free.** The
pools of record and PL1's pools are untouched. This page is a PL1 improvement round: it names the arms it targets (all nine),
the change it tries (new pool-construction families, section 5), and it enters by UL2's universal rule (section 4.4).

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links
  improve how much ever is possible".
- Earlier, still binding: "the pools or the rules or links all should be universal"; "i dont want to freeze pools for the
  sake of it, make sure we freeze when we know it is the best for that specific arm"; "link improvement with precision is all
  we need". Pools first and parameter-free (no learning, no model, no LLM, no spaCy); one model per family; the hub line is
  CLOSED (UL2 8.15): no hub step is proposed here, and hub forms already in UL2's layout are labels like any other; every
  rule is universal on all seven datasets (MultiHop-RAG, MuSiQue, SQuAD, MetaQA, WebQSP, HotpotQA, 2Wiki).

## 1. What the target is

Per question the pool is a set of corpus nodes (passages, articles, KB nodes). **ALL** = the share of questions whose pool
holds every gold node of the question; **ANY** = at least one (both are `headroom_metrics`' `all_gold_at_pool` and
`any_gold_at_pool`, unchanged). **Nodes** = the mean pool size (`candidates_mean`). The target: ALL >= 0.95 on every dataset,
as close to 1.00 as possible, at the smallest mean nodes; the links (the typed graph) are the lever that should make this
cheaper than retrieval alone. Two floors are read, 0.95 (the user's) and 0.98 (read beside it, not a second target).

Where the FILED grids stand (read, on the frozen first stage, the filed eval populations; 13 Sep, `outputs/m3a/headroom`,
31 eval cells per dataset; mean nodes in brackets): SQuAD 0.980 (50), retrieval alone; HotpotQA 0.968 (94) in a STRUCT
cell, best 0.987 (1,919); MultiHop-RAG, on its own layout and 224 dev questions, 0.987 (100) on e3v lists and 0.964 (50) on
frozen lists; 2Wiki 0.950 only at 2,140 (retrieval alone best 0.513 at 365); MuSiQue best 0.803 (2,092) (retrieval alone
0.585 at 341); MetaQA best 0.717 (1,290) (retrieval alone 0.076 at 378); WebQSP best 0.655 (1,175) (retrieval alone 0.285
at 327). So SQuAD, HotpotQA and MultiHop-RAG are past the floor at under 100 nodes, 2Wiki only at 2,140, and MuSiQue, MetaQA
and WebQSP never reach it in the 31 cells. The filed own constructions (M3B's contract, which added 3-hop cells for MetaQA
and WebQSP; its knee rule, `fraction_of_attainable@5 >= 0.90`, does not ask for ALL) hold ALL 0.906 / 0.968 / 0.894 / 0.803 /
0.980 / 0.688 on 2Wiki / HotpotQA / MetaQA / MuSiQue / SQuAD / WebQSP, and MultiHop-RAG's S0c arm 0.933 (dev). **The new
target replaces the knee with the ALL floor as the own-construction rule** (section 4).

## 2. What is fixed

### 2a. Substrate, questions, first stage

- **The six datasets** read the served package (CRAG, `configs/m3a_headroom.yaml`'s `package_root`, freeze
  `58958f33...`, read-only, the loader imported in place as the filed runs did): its three graph families (the code's
  `structural`, `ner`, `knn` slots), its SPLADE cache and its frozen dense cache and vectors. **MultiHop-RAG** has no package
  graph: it reads its own B0 layout (`outputs/pl1/multihoprag/RK/layout.npz`, the stores S0b filed), with PL1's S0 readers.
  The UL2 layout of the six is not materialized yet (no `outputs/pl1/<d>/<v>/layout.npz` exists on the host on 25 Nov), so
  **PL2's six-dataset numbers are on the package's old graph; the same code reads a layout when one exists** (`--graph layout`,
  section 7), and that run is the later step 8.2, not part of this round's calls.
- **Questions (the same rule on all seven; the stand-in is declared once, here).** *Selection* ("dev") = a carve of questions
  the grid never confirms on; *confirmation* ("eval") = the s1eval carve, read only for the cells the rule has already named.
  - dev: step 1's matched select carve `s1sel` (`outputs/step1/carves.json`, sha256 `53cfb89f...`, key `dselect`: 1,497
    MetaQA, 1,498 SQuAD, 1,534 MuSiQue, 1,508 HotpotQA, 1,496 2Wiki; train questions, zero-gold excluded); MultiHop-RAG's
    dev split (224 questions, PL1's S0c carve); **WebQSP has no select carve (it never trained, carves.json's note), so its dev
    is the complement of its eval holdout in train, `sorted(train ids)[1::2]` (M3B's `training_source_ids` rule), zero-gold
    excluded.**
  - eval: `s1eval` (MuSiQue 2,417, SQuAD 11,873, HotpotQA 7,405, 2Wiki 12,576, MetaQA 9,785 = every fourth dev question,
    WebQSP 1,503) and MultiHop-RAG's test split (1,130). These are the questions ENC's E0 e3v lists exist for.
- **First stages.** The selection read is on the FROZEN first stage (the package's dense and SPLADE top-1000 caches), because
  e3v's first-stage lists exist only for the eval rows (E0). The confirmation read is on BOTH the frozen lists and e3v's (E0's
  `outputs/enc/e0/e3v/<d>.npz` for the six; `outputs/pl1/multihoprag/RK/s0/first_e3v.npz` for MultiHop-RAG, which also has
  dev). SPLADE is the package's in every read. Seeds are the filed rule: dense top 5 U SPLADE top 5, first occurrence kept.
- **Disclosed:** the filed grids' "eval" population is the dev split (the same questions as s1eval, MetaQA excepted); M3B chose
  its own constructions on it. PL2 selects on a different carve and confirms on that one, so the selection never sees the
  confirmation questions. The filed 31 cells are re-computed on MuSiQue's filed population by the same code (section 6, G-K).

### 2b. The first-stage vectors for ranked expansion (family K)

The ranking vector is the first stage's own encoder: the package's frozen dense vectors for the frozen read, e3v's document
and query vectors (`outputs/enc/e3v/<d>/{docs,queries}`, fp16, L2-normalised, 1,152 wide) for the e3v read. Score = inner
product, accumulated in float64 from the fp16 rows (so the order does not depend on the BLAS thread count; gate G-D).
No second encoder, no new encode, no GPU; every read is a gather and a dot product.

## 3. The measures (the same code on all seven)

Per cell: ALL, ANY, gold recall (`coverage_micro`), the fields of `M3A.cell` (the filed `KEEP` set), mean / p50 / p95 / max
pool rows, and per-question present counts and pool sizes kept for the paired reads. ALL gets a Wilson 95% interval; every
difference of two cells is a paired bootstrap over questions (1,000 resamples, `numpy.random.default_rng(20261125)`). The
membership count is `candidate_headroom.present_counts`, the metrics are `headroom_metrics` and `full_coverage_ceiling`,
assembled exactly as `scripts/m3a_headroom.cell` assembles them, from per-chunk counts (gate G-K compares the two on the
filed rows).

## 4. The rule

### 4.1 Selection (on dev, universal, no constant of its own)

Among the declared cells (section 5) whose mean nodes are within the bound **2,500** (M3B's bound):
**R95** = the cell with the smallest mean nodes among those with dev ALL >= 0.95; ties to the higher ALL, then to the name in
sort order. If no cell reaches 0.95, R95 = the cell with the highest dev ALL, ties to fewer nodes (and the dataset is flagged
SHORT). **R98** likewise at 0.98. Each is read three ways, by the same rule: over all cells, over the retrieval-only cells
(families R0, R1), and over the graph cells (G, G2, K); "what the links bought" is the difference between the graph frontier
and the retrieval-only frontier at the same nodes (the frontier at N = the best ALL among cells with nodes <= N). The whole
curve (every cell's ALL against nodes, its Pareto frontier marked) is filed per dataset.

### 4.2 Confirmation (on eval, after selection, nothing re-chosen)

The cells R95 and R98 name (three readings each, so at most six) and the filed own construction (4.3) are re-run unchanged on
the eval carve, with the frozen lists and with e3v's lists. **CONFIRMED** = eval ALL >= the floor (0.95, 0.98); otherwise
**SHORT by x points**, with the dev-to-eval drop printed. No cell is re-selected on eval: a shortfall is a finding and the next
round (section 8) is declared on its own page.

### 4.3 The old rule, for the call

**OLD** = the filed own construction of the dataset (M3B's contract: MetaQA `equal_rrf_budget_50+STRUCT:h3_c25_v2000`, SQuAD
`equal_rrf_budget_50`, MuSiQue `equal_rrf_budget_200+FULL:h2_c25`, HotpotQA and 2Wiki `equal_rrf_budget_50+STRUCT:h1_c25`,
WebQSP `dense_top200+FULL:h3_c25_v2000`; MultiHop-RAG: PL1's S0c, `equal_rrf_budget_50`). Each is a cell of section 5's grid,
re-run by the same code on dev and on eval. **NEW** = R95 over every cell of section 5 (the all-cells reading).

### 4.4 The call (UL2's universal rule: ABOVE somewhere, BELOW nowhere, one price bound, one call over all seven)

On eval, per dataset, NEW against OLD (paired, frozen lists, because the filed constructions are frozen-list ones): the ALL
difference (NEW - OLD) and the ratio of mean nodes (NEW / OLD), each with its paired interval.
- **BELOW**: NEW misses the floor where OLD reaches it, or the ALL difference's upper end is below 0 with the nodes ratio's
  lower end at or above 1.
- **ABOVE**: NEW reaches the floor where OLD does not, or the nodes ratio's upper end is below 1 with ALL not BELOW, or the ALL
  difference's lower end is above 0 with the nodes ratio's upper end at or below 1.
- **LEVEL**: anything else.

**CARRY_PL2** (section 5's grid replaces the knee as the own-construction rule of all seven datasets) when no dataset is BELOW,
at least one is ABOVE, and every dataset's chosen cell is within the 2,500 bound. Otherwise **NOT_CARRIED: the knee rule stays
the own-construction rule** and the grid is still filed as the diagnosis. A grid that helps one dataset and hurts another is
not carried (PL1 section 2a): the call is on the whole of section 5's grid, and the per-family reads (which family supplied
each dataset's chosen cell) are filed beside it and decide nothing.

## 5. The grid (fixed at this commit; nothing is added after a number is read)

`regime` = the family set the expansion walks (the filed names): STRUCT, NER, KNN, FULL = all three. The walk is
`m3b_pools.expand_hops`, which is `scripts/m3a_headroom.expand` byte for byte for hops <= 2 (its docstring says so; gate G-X
checks it) and the same rule iterated for hop 3: node-major capped neighbours in CSR row order; later hops expand the previous
hop's fresh nodes in order of first appearance, each contributing its first per_frontier_cap neighbours per family, the visited
skipped, until visited_cap nodes (seeds included) are visited. Every cell below is a deterministic function of the questions,
the first-stage lists, the graph and (family K only) the first-stage vectors.

**Settings** (name: hops / per_seed_cap / per_frontier_cap / visited_cap), nine, the same on all datasets: the filed three
`h1_c25` 1/25, `h1_c100` 1/100, `h2_c25` 2/25/25/2000; M3B's two 3-hop ones (its MetaQA and WebQSP cells; they are here so every
filed construction is a cell of the grid) `h3_c25_v2000` 3/25/25/2000, `h3_c25_v4000` 3/25/25/4000; and four new
`h2_c25_v5000` 2/25/25/5000, `h2_c100_v2000` 2/100/100/2000, `h2_c100_v5000` 2/100/100/5000, `h3_c100_v5000` 3/100/100/5000.

| family | cells | what it is | what it can show / cannot show |
| --- | --- | --- | --- |
| R0 filed retrieval | 7: `dense_top200`, `splade_top200`, `frozen_union`, `equal_rrf_budget_{50,100,200,400}` | the filed first-stage pools (RRF of two top-200 lists) | the filed starting point; nothing new |
| R1 deeper retrieval (i) | 37: `rrf1000_b{10,15,20,25,30,40,50,60,75,100,125,150,200,250,300,400,500,600,800,1000,1200,1600,2000}` (23) and `dense_top{5,10,20,25,30,50,75,100,150,300,400,600,800,1000}` (14) | equal-weight RRF (constant 60) of the dense and SPLADE top-1000, a prefix of b rows (at most the union); the dense list's prefixes | **the retrieval-only frontier**, from the smallest sizes to the whole union: how far retrieval alone goes. It says nothing about links and cannot reach a gold that both lists rank past 1,000. The filed grids fused top-200 lists to match the inherited protocol's exposure; here exposure is the object, so the fusion is deeper (a pool of a given size differs slightly from R0's) |
| G graph union (ii) | 108: bases `equal_rrf_budget_50`, `equal_rrf_budget_200`, `dense_top200` x 4 regimes x 9 settings (24 of them are the filed G cells) | base U the whole bounded expansion, as filed | the reach of each regime at its size (a ceiling per node, not an order); it cannot rank the added nodes, so most of what it adds is not gold; the dense base is M3B's WebQSP base, here on every dataset |
| G2 bigger base (i) | 12: base `rrf1000_b800` x 4 regimes x `h1_c25`, `h1_c100`, `h2_c25` | a deep first-stage base plus links | whether links add golds to a base that already holds the first stage's deep tail; cells above 2,500 nodes are read but cannot be selected |
| K ranked expansion (iii) | 640: bases `rrf1000_b{25,50,100,200}` and `dense_top{25,50,100,200}` x M in {25,50,100,200,400} x 4 regimes x settings `h1_c100`, `h2_c25`, `h2_c100_v5000`, `h3_c100_v5000` | pool = base U the top M of (the expansion's nodes not in the base), ranked by the first stage's inner product with the question (ties to the lower node id); a ranked cell adds at most M nodes | whether ordering the walk's reach by the encoder's own geometry makes the added nodes cheap. It cannot reach a gold the walk does not reach (the reach ceilings of the filed `reachability` blocks) and cannot raise a gold the encoder ranks below M inside the walk's set; the ranking adds no semantic signal the first stage did not have |

804 cells per dataset in all (44 + 108 + 12 + 640). The walks run once per (query, regime, setting): 36.

**Not in this round (candidates for a later one, declared on its own page):** label-ordered caps (the walk's per-node caps
taken in the UL2P ledger's precision order, K.knn first, instead of CSR row order); support-ordered expansion (the number of
seeds that reach a node); reading the per-link weights; any link-rule change (it goes through UL2's gates, not through this
grid). **Stop rule:** the cell list, the settings, M, the bases, the bound, the floors and the dev and eval carves are fixed
by this commit; every output records the sha256 of the cell list and the code refuses to file when it differs. If a dataset is
SHORT, the report names what was tried and the next change, and that change is a new page.

## 6. Gates (each is a failure that stops that dataset's reading; none of its numbers is read)

- **G-K (the filed cells are reproduced).** The new code, on MuSiQue's filed eval population and frozen lists, builds the 31
  filed cells (R0 and the filed G cells) and each must equal the filed row of `outputs/m3a/headroom/musique.json` on every `KEEP`
  field to 1e-9 (the shape of PL1's `zpl1h.py gk`). Required before any new cell on any dataset is read. **G-K2:** the same on
  MultiHop-RAG's dev questions, with e3v and with frozen lists, on its B0 layout, against the 31 cells of
  `outputs/pl1/multihoprag/RK/s0/knee_e3v.json` and `knee_frozen.json` (this closes the layout-store path and the e3v-list
  path).
- **G-X (the walk).** For each regime and each setting of hops <= 2, `expand_hops` equals `M3A.expand` array for array on at
  least 200 questions; the 3-hop walk equals an independent per-node Python loop on a smaller sample.
- **G-F (gold-free).** No pool function takes or reads gold (checked on the signatures and the source of every function that
  builds a pool, and by a run whose gold argument is withheld: the pool arrays' hash is the same). Gold enters only in the
  grading that counts present nodes after the pool is built.
- **G-D (determinism).** The pools of a sample of questions hash the same under 1 and 4 BLAS/OpenMP threads and under a
  different chunking of the questions.
- **G-U (universal).** One hash of the code path and the constants is recorded in every dataset's output; the report refuses
  to file when two datasets differ (PL1's `zpl1g` rule).

## 7. Code, rows, outputs

- **New module only:** `outputs/mp_unified/zpl2g.py`. It imports `scripts/m3a_headroom.py` (`capped_multi`,
  `stable_union_rows`, `rrf_fused_rows`, `prefix_rows`, `FamilyCSR`, the population and gold loaders),
  `mp_retrieval.m3b_pools` (`expand_hops`, `seeds_of`), `scripts/m3b_contract.py` (`KEEP`), and `step4_pool_reach`'s opened
  package as PL1's `zpl1h.py` does, and PL1's S0 readers for MultiHop-RAG; it edits none of them and none of the PL1 modules.
  Commands: `gk`, `grid --ds D --carve dev|eval --first frozen|e3v [--cells FILE]`, `select --ds D`, `confirm --ds D`, `call`,
  `selftest`. Laptop-sized self-test on MultiHop-RAG and SQuAD first.
- **Outputs** (arrays uncommitted, json heads filed): `outputs/pl2/<d>/{grid_dev_frozen,select,confirm}.json`,
  `outputs/pl2/gk_musique.json`, `outputs/pl2/call.json`; the CSR cache under `outputs/pl2/csr`. Host rows are `pl2-*`, CPU
  only, small (1 to 8 cpus), no GPU. The query chunks are streamed (a chunk's pools are graded and discarded), so memory is
  the graph, the caches and one chunk.

## 8. What follows

1. The cells the call carries become each dataset's own construction for PL1's pool builds (docs/PL1_POOLS_ON_UL2_LAYOUT.md).
   The ranked family needs the first stage's vectors inside the pool code and a base that is not one of U2m's three bases
   (the S0d note of `zpl1h.py` already records the second); that is a request to PL1's owner, not an edit made here.
2. The same grid on the UL2 layout stores once S1 has materialized them (the link-reach of the new links is the measure the
   next round needs), and on the e3v first stage for dev once e3v lists exist for the select carves.
3. The diagnosis in the results section names the candidate link-rule changes that the reach ceilings point to. **Names only;
   adopting one is a design step that goes to the user first** (PL1 section 7).

## 9. Disclosed

- Written after reading, read-only, the filed grids' rows and `reachability` blocks, M3B's contract, PL1's S0 module and
  record, and the UL2P ledger's published precision order (K.knn the most precise label on every dataset; R.mention the
  costly one). No PL2 number was computed. The ledger session's files are not edited and are read, not decided.
- The filed own constructions were chosen by a knee rule that does not read ALL, on the eval population; the NEW rule is chosen
  on a different carve by a different objective, so the call compares two rules, not two cells of one rule.
- Selection is on 1.5k-question carves (about +/- 1 point at 0.95) over 804 cells, so the winner's cell is optimistic on dev;
  that is why confirmation exists and why the rule is not decided by significance. The eval read is the decider and is not
  re-selected.
- The six datasets' graph here is the old package graph (its ner family is the spaCy one). It is the graph the filed grids
  were measured on, which is what lets G-K close; PL1's layout (F0T, W2_S, carried forms, K_e3v) will differ, and section 8.2
  is where that is read.
