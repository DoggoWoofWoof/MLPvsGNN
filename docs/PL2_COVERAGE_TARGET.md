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

## 10. Results (25 Nov 2026; all seven datasets read, re-filed 07:18)

Read: all seven. MultiHop-RAG, SQuAD, MuSiQue, MetaQA and WebQSP on the laptop (first filing, 06:03); HotpotQA and 2Wiki on the host
(rows `pl2-*`, run 06:01 to 07:13 on the host's package mirror and the e3v vectors); the four e3v confirmations of SQuAD, MuSiQue, MetaQA
and WebQSP on the host (06:12 to 06:15). Every number below is a record's (`outputs/pl2/<d>/{select,confirm_frozen,confirm_e3v}.json`,
`outputs/pl2/call.json`); the tables are written from those records by `outputs/mp_unified/zpl2r.py` into `outputs/pl2/report.md` and are
pasted here unchanged (the five-dataset tables of the 06:03 filing are replaced by these; commit b251d67 holds them). The pool functions were not
edited after the code commit 656cc37; the host flag, the population cache's atomic write, the host gate check, the G-U check in `select` and the
once-at-start hashes were added around it (commits fb4b1a9, fbc83eb, a9810ac; the code-path sha256 `18c31653...` is the same before and
after each, and `call.json` records it equal on all seven datasets).

### 10.1 Gates and cost

- **G-K** MuSiQue: all 31 filed cells equal to 1e-9 (max difference 0), the three CSR stores equal the family CSRs (laptop,
  64 s; the host's own G-K is row `pl2-gk-musique`, and every host select and confirm refuses to run without it). **G-K2**
  MultiHop-RAG e3v and frozen: PASS against `knee_e3v.json` and `knee_frozen.json`. **Self-test** (the engine against a
  brute-force pool builder, G-X, G-F, G-D at 1 and 4 threads): PASS (`selftest.json`). **G-U**: every dev grid head of the five
  datasets carries code-path sha256 `18c31653...` and the declared cell list's `76bdf3a8...`; `select` now refuses a head that
  does not (the SQuAD and MuSiQue heads first written had been hashed after this file was edited under the running job, so their
  hash differed from the code that ran; both grids were re-run and are the ones read, section 10.7).
- **Cost** (one process, one core, no GPU; dev grid of 804 cells): MultiHop-RAG 12 s (224 questions), MetaQA 314 s (1,497; 0.43 GB),
  SQuAD 488 s (1,498; 0.35 GB), MuSiQue 505 s (1,534; 0.65 GB), WebQSP 534 s (1,507; 0.5 GB private, 4.9 GB resident as mapped
  vector pages). A host smoke of 14 to 16 dev questions with the host's package mirror ran on 2Wiki (32 s with the population
  built; 1.85 GB), HotpotQA (48 s; 1.70 GB), MetaQA (5.5 s; 0.39 GB) and WebQSP (21 s; 0.84 GB): the mirror opens, the e3v lists
  and vectors read, nothing was graded for the record.

- **HotpotQA and 2Wiki (host, 25 Nov)**: every grid head of both carries code-path sha256 `18c31653...` and cell list `76bdf3a8...` (G-U, checked
  by `select` before it reads), and the host's own G-K on MuSiQue (`pl2-gk-musique`, 0.4 GB) ran before every host select and confirm.
  Cost on the host (2 cores each, no GPU): HotpotQA dev grid 3 parts of about 2 minutes (10.2 GB resident, mostly mapped vector pages), 2Wiki 4, 4 and 3 minutes
  (10.5 GB); select 3 seconds; the frozen and e3v confirmations about 1 to 2 minutes each. The 2Wiki parts were renamed before they ran (below, 10.7).

### 10.2 Tables

### Dev reading (the rule's own numbers, dev carve, frozen lists)

| dataset | dev q | R95 retrieval-only | R95 with links | R98 retrieval-only | R98 with links | OLD (own construction) |
|---|---|---|---|---|---|---|
| multihoprag | 224 | `rrf1000_b50` (0.9732 at 50.0) | `rrf1000_b25+FULL:h1_c100+top25` (0.9777 at 50.0) | `rrf1000_b60` (0.9866 at 60.0) | `rrf1000_b50+FULL:h1_c100+top25` (0.9955 at 75.0) | `equal_rrf_budget_50` (0.9643 at 50.0) |
| squad | 1498 | `rrf1000_b15` (0.9519 at 15.0) | `dense_top25+KNN:h1_c100+top25` (0.9579 at 41.4) | `rrf1000_b75` (0.9833 at 75.0) | `rrf1000_b50+KNN:h1_c100+top25` (0.9806 at 63.3) | `equal_rrf_budget_50` (0.9793 at 50.0) |
| hotpotqa | 1508 | `rrf1000_b2000` (0.8687 at 1,764.6, SHORT) | `rrf1000_b25+STRUCT:h1_c100+top100` (0.9615 at 77.3) | `rrf1000_b2000` (0.8687 at 1,764.6, SHORT) | `rrf1000_b100+STRUCT:h1_c100+top200` (0.9808 at 153.5) | `equal_rrf_budget_50+STRUCT:h1_c25` (0.9682 at 94.1) |
| 2wiki | 1496 | `rrf1000_b2000` (0.5107 at 1,842.1, SHORT) | `equal_rrf_budget_200+FULL:h2_c25` (0.9492 at 2,129.4, SHORT) | `rrf1000_b2000` (0.5107 at 1,842.1, SHORT) | `equal_rrf_budget_200+FULL:h2_c25` (0.9492 at 2,129.4, SHORT) | `equal_rrf_budget_50+STRUCT:h1_c25` (0.8964 at 105.1) |
| musique | 1534 | `rrf1000_b2000` (0.7353 at 1,698.3, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `rrf1000_b2000` (0.7353 at 1,698.3, SHORT) | `rrf1000_b800+FULL:h1_c100` (0.8012 at 1,082.6, SHORT) | `equal_rrf_budget_200+FULL:h2_c25` (0.7679 at 2,093.5) |
| metaqa | 1497 | `rrf1000_b2000` (0.2712 at 1,894.9, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `rrf1000_b2000` (0.2712 at 1,894.9, SHORT) | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (0.8958 at 2,157.6, SHORT) | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` (0.8938 at 2,021.7) |
| webqsp | 1507 | `rrf1000_b2000` (0.4154 at 1,666.3, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `rrf1000_b2000` (0.4154 at 1,666.3, SHORT) | `dense_top200+FULL:h2_c100_v2000` (0.7505 at 1,888.2, SHORT) | `dense_top200+FULL:h3_c25_v2000` (0.6967 at 2,118.8) |

### The chosen cell (R95 over all cells) and its confirmation on eval

| dataset | chosen cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v | OLD eval ALL frozen | OLD eval ALL e3v |
|---|---|---|---|---|---|---|---|
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | 0.9777 | 50.0 | 0.9690 (50.0) | 0.9699 (50.0) | 0.9540 | 0.9451 |
| squad | `rrf1000_b15` | 0.9519 | 15.0 | 0.9554 (15.0) | 0.9421 (15.0) SHORT | 0.9798 | 0.9736 |
| hotpotqa | `rrf1000_b25+STRUCT:h1_c100+top100` | 0.9615 | 77.3 | 0.9627 (77.9) | 0.9646 (77.6) | 0.9684 | 0.9646 |
| 2wiki | `equal_rrf_budget_200+FULL:h2_c25` (SHORT) | 0.9492 | 2,129.4 | 0.9503 (2,140.1) | 0.9750 (2,139.0) | 0.9058 | 0.9310 |
| musique | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | 0.8010 (1,061.1) SHORT | 0.8026 | 0.7911 |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | 0.9281 (2,142.8) SHORT | 0.8949 | 0.9278 |
| webqsp | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | 0.7432 (1,989.4) SHORT | 0.6880 | 0.6447 |

### The same at 0.98 (R98 over all cells) on eval

| dataset | chosen cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |
|---|---|---|---|---|---|
| multihoprag | `rrf1000_b60` | 0.9866 | 60.0 | 0.9681 (60.0) SHORT | 0.9664 (60.0) SHORT |
| squad | `rrf1000_b50+KNN:h1_c100+top25` | 0.9806 | 63.3 | 0.9818 (63.0) | 0.9763 (63.0) SHORT |
| hotpotqa | `rrf1000_b100+STRUCT:h1_c100+top200` | 0.9808 | 153.5 | 0.9776 (154.5) SHORT | 0.9743 (154.4) SHORT |
| 2wiki | `equal_rrf_budget_200+FULL:h2_c25` (SHORT) | 0.9492 | 2,129.4 | 0.9503 (2,140.1) SHORT | 0.9750 (2,139.0) SHORT |
| musique | `rrf1000_b800+FULL:h1_c100` (SHORT) | 0.8012 | 1,082.6 | 0.8283 (1,064.2) SHORT | 0.8010 (1,061.1) SHORT |
| metaqa | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` (SHORT) | 0.8958 | 2,157.6 | 0.8952 (2,153.5) SHORT | 0.9281 (2,142.8) SHORT |
| webqsp | `dense_top200+FULL:h2_c100_v2000` (SHORT) | 0.7505 | 1,888.2 | 0.7638 (1,890.8) SHORT | 0.7432 (1,989.4) SHORT |

### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here), retrieval-only and with links

| dataset | retrieval-only 0.95 | with links 0.95 | retrieval-only 0.98 | with links 0.98 |
|---|---|---|---|---|
| multihoprag | `rrf1000_b50` 0.9732 at 50.0 | `rrf1000_b25+FULL:h1_c100+top25` 0.9777 at 50.0 | `rrf1000_b60` 0.9866 at 60.0 | `rrf1000_b50+FULL:h1_c100+top25` 0.9955 at 75.0 |
| squad | `rrf1000_b15` 0.9519 at 15.0 | `dense_top25+KNN:h1_c100+top25` 0.9579 at 41.4 | `rrf1000_b75` 0.9833 at 75.0 | `rrf1000_b50+KNN:h1_c100+top25` 0.9806 at 63.3 |
| hotpotqa | none on the grid (best 0.8687) | `rrf1000_b25+STRUCT:h1_c100+top100` 0.9615 at 77.3 | none on the grid (best 0.8687) | `rrf1000_b100+STRUCT:h1_c100+top200` 0.9808 at 153.5 |
| 2wiki | none on the grid (best 0.5107) | `rrf1000_b800+FULL:h2_c25` 0.9545 at 2,695.6 | none on the grid (best 0.5107) | none on the grid (best 0.9552) |
| musique | none on the grid (best 0.7353) | none on the grid (best 0.8449) | none on the grid (best 0.7353) | none on the grid (best 0.8449) |
| metaqa | none on the grid (best 0.2712) | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 | none on the grid (best 0.2712) | none on the grid (best 0.9613) |
| webqsp | none on the grid (best 0.4154) | none on the grid (best 0.8016) | none on the grid (best 0.4154) | none on the grid (best 0.8016) |

### The highest ALL on dev (any cell; the bound is 2,500 mean nodes), retrieval-only and with links

| dataset | retrieval-only, within the bound | retrieval-only, any size | with links, within the bound | with links, any size |
|---|---|---|---|---|
| multihoprag | `rrf1000_b125` 1.0000 at 125.0 | `rrf1000_b125` 1.0000 at 125.0 | `rrf1000_b50+FULL:h2_c100_v5000+top50` 1.0000 at 100.0 | `rrf1000_b50+FULL:h2_c100_v5000+top50` 1.0000 at 100.0 |
| squad | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b1600` 0.9980 at 1,544.8 | `rrf1000_b800+FULL:h2_c25` 0.9973 at 2,319.9 | `rrf1000_b800+FULL:h2_c25` 0.9973 at 2,319.9 |
| hotpotqa | `rrf1000_b2000` 0.8687 at 1,764.6 | `rrf1000_b2000` 0.8687 at 1,764.6 | `rrf1000_b800+FULL:h1_c100` 0.9927 at 935.0 | `rrf1000_b800+FULL:h1_c100` 0.9927 at 935.0 |
| 2wiki | `rrf1000_b2000` 0.5107 at 1,842.1 | `rrf1000_b2000` 0.5107 at 1,842.1 | `equal_rrf_budget_200+FULL:h2_c25` 0.9492 at 2,129.4 | `equal_rrf_budget_200+FULL:h2_c25_v5000` 0.9552 at 4,073.5 |
| musique | `rrf1000_b2000` 0.7353 at 1,698.3 | `rrf1000_b2000` 0.7353 at 1,698.3 | `rrf1000_b800+FULL:h1_c100` 0.8012 at 1,082.6 | `rrf1000_b800+FULL:h2_c25` 0.8449 at 2,611.7 |
| metaqa | `rrf1000_b2000` 0.2712 at 1,894.9 | `rrf1000_b2000` 0.2712 at 1,894.9 | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` 0.8958 at 2,157.6 | `equal_rrf_budget_50+STRUCT:h3_c100_v5000` 0.9613 at 5,003.7 |
| webqsp | `rrf1000_b2000` 0.4154 at 1,666.3 | `rrf1000_b2000` 0.4154 at 1,666.3 | `dense_top200+FULL:h2_c100_v2000` 0.7505 at 1,888.2 | `dense_top200+FULL:h3_c100_v5000` 0.8016 at 5,100.2 |

### What the links bought (dev frontier of the graph cells minus the retrieval cells, at equal mean nodes; ALL)

| dataset | 25 | 50 | 75 | 100 | 150 | 200 | 300 | 500 | 1000 | 2000 | 2500 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| multihoprag | 0.8973 / none | 0.9732 / 0.9777 (+0.45) | 0.9866 / 0.9955 (+0.89) | 0.9955 / 1.0000 (+0.45) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) | 1.0000 / 1.0000 (+0.00) |
| squad | 0.9666 / none | 0.9793 / 0.9773 (-0.20) | 0.9833 / 0.9846 (+0.13) | 0.9887 / 0.9853 (-0.33) | 0.9920 / 0.9907 (-0.13) | 0.9933 / 0.9913 (-0.20) | 0.9940 / 0.9947 (+0.07) | 0.9953 / 0.9953 (+0.00) | 0.9973 / 0.9967 (-0.07) | 0.9980 / 0.9967 (-0.13) | 0.9980 / 0.9973 (-0.07) |
| hotpotqa | 0.6127 / none | 0.6684 / 0.9224 (+25.40) | 0.6936 / 0.9496 (+25.60) | 0.7122 / 0.9682 (+25.60) | 0.7341 / 0.9775 (+24.34) | 0.7533 / 0.9808 (+22.75) | 0.7825 / 0.9828 (+20.03) | 0.8123 / 0.9834 (+17.11) | 0.8501 / 0.9927 (+14.26) | 0.8687 / 0.9927 (+12.40) | 0.8687 / 0.9927 (+12.40) |
| 2wiki | 0.3195 / none | 0.3523 / 0.7172 (+36.50) | 0.3663 / 0.8342 (+46.79) | 0.3803 / 0.8890 (+50.87) | 0.4024 / 0.8984 (+49.60) | 0.4138 / 0.9078 (+49.40) | 0.4285 / 0.9164 (+48.80) | 0.4432 / 0.9325 (+48.93) | 0.4706 / 0.9358 (+46.52) | 0.5107 / 0.9378 (+42.71) | 0.5107 / 0.9492 (+43.85) |
| musique | 0.2979 / none | 0.3696 / 0.4146 (+4.50) | 0.4153 / 0.4733 (+5.80) | 0.4407 / 0.5261 (+8.54) | 0.4778 / 0.5906 (+11.28) | 0.5111 / 0.6271 (+11.60) | 0.5600 / 0.6734 (+11.34) | 0.6206 / 0.7066 (+8.60) | 0.6812 / 0.7881 (+10.69) | 0.7353 / 0.8012 (+6.58) | 0.7353 / 0.8012 (+6.58) |
| metaqa | 0.0147 / none | 0.0200 / 0.2325 (+21.24) | 0.0234 / 0.3039 (+28.06) | 0.0261 / 0.3267 (+30.06) | 0.0321 / 0.3280 (+29.59) | 0.0588 / 0.3300 (+27.12) | 0.0675 / 0.3641 (+29.66) | 0.1102 / 0.5638 (+45.36) | 0.1864 / 0.6927 (+50.63) | 0.2712 / 0.7408 (+46.96) | 0.2712 / 0.8958 (+62.46) |
| webqsp | 0.1526 / none | 0.1772 / 0.3676 (+19.04) | 0.1977 / 0.4625 (+26.48) | 0.2150 / 0.5601 (+34.51) | 0.2455 / 0.5886 (+34.31) | 0.2694 / 0.6025 (+33.31) | 0.3026 / 0.6251 (+32.25) | 0.3278 / 0.6251 (+29.73) | 0.3988 / 0.6676 (+26.87) | 0.4154 / 0.7505 (+33.51) | 0.4154 / 0.7505 (+33.51) |

(each entry: retrieval-only frontier / graph frontier (links bought in points); `none` = no graph cell is that small)

### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)

| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |
|---|---|---|---|---|---|---|---|
| squad | 11873 | 54 | 0.074 / 0.519 / 0.741 | 0.037 / 0.222 / 0.685 | 0.000 / 0.074 / 0.093 | 0.111 / 0.722 / 0.981 | 53 of 54 |
| hotpotqa | 1852 | 378 | 0.914 / 0.945 / 0.977 | 0.203 / 0.278 / 0.364 | 0.101 / 0.184 / 0.252 | 0.927 / 0.963 / 0.992 | 373 of 378 |
| 2wiki | 2096 | 1038 | 0.878 / 0.930 / 0.987 | 0.352 / 0.445 / 0.568 | 0.089 / 0.164 / 0.213 | 0.921 / 0.982 / 0.998 | 991 of 1038 |
| musique | 2417 | 1003 | 0.262 / 0.770 / 0.935 | 0.347 / 0.485 / 0.627 | 0.111 / 0.178 / 0.256 | 0.464 / 0.892 / 1.000 | 1003 of 1003 |
| metaqa | 39138 | 36160 | 0.118 / 0.486 / 0.996 | 0.000 / 0.000 / 0.000 | 0.001 / 0.006 / 0.017 | 0.119 / 0.551 / 0.998 | 35989 of 36160 |
| webqsp | 1503 | 1074 | 0.366 / 0.898 / 0.989 | 0.022 / 0.024 / 0.025 | 0.005 / 0.014 / 0.032 | 0.364 / 0.927 / 0.996 | 1053 of 1074 |

### The call: CARRY_PL2 (read 7 of 7: 2wiki, hotpotqa, metaqa, multihoprag, musique, squad, webqsp)

| dataset | status | NEW | NEW ALL (nodes) | OLD | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |
|---|---|---|---|---|---|---|---|
| multihoprag | ABOVE | `rrf1000_b25+FULL:h1_c100+top25` | 0.9690 (50.0) | `equal_rrf_budget_50` | 0.9540 (50.0) | +1.50 [+0.53, +2.48] | 1.000 [1.000, 1.000] |
| squad | ABOVE | `rrf1000_b15` | 0.9554 (15.0) | `equal_rrf_budget_50` | 0.9798 (50.0) | -2.44 [-2.71, -2.20] | 0.300 [0.300, 0.300] |
| hotpotqa | ABOVE | `rrf1000_b25+STRUCT:h1_c100+top100` | 0.9627 (77.9) | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9684 (93.7) | -0.57 [-0.84, -0.28] | 0.832 [0.828, 0.836] |
| 2wiki | ABOVE | `equal_rrf_budget_200+FULL:h2_c25` | 0.9503 (2,140.1) | `equal_rrf_budget_50+STRUCT:h1_c25` | 0.9058 (105.6) | +4.45 [+4.10, +4.83] | 20.258 [20.195, 20.323] |
| musique | ABOVE | `rrf1000_b800+FULL:h1_c100` | 0.8283 (1,064.2) | `equal_rrf_budget_200+FULL:h2_c25` | 0.8026 (2,092.4) | +2.57 [+1.41, +3.72] | 0.509 [0.506, 0.511] |
| metaqa | LEVEL | `equal_rrf_budget_200+STRUCT:h3_c25_v2000` | 0.8952 (2,153.5) | `equal_rrf_budget_50+STRUCT:h3_c25_v2000` | 0.8949 (2,017.6) | +0.03 [+0.00, +0.07] | 1.067 [1.067, 1.067] |
| webqsp | ABOVE | `dense_top200+FULL:h2_c100_v2000` | 0.7638 (1,890.8) | `dense_top200+FULL:h3_c25_v2000` | 0.6880 (2,119.6) | +7.58 [+5.79, +9.45] | 0.892 [0.883, 0.901] |

### 10.3 What each dataset says (dev reading, eval confirmation, frozen lists)

- **MultiHop-RAG** (224 dev questions, so about +/- 2 points). 0.95 is reached at 50 mean nodes with and without links; links buy +0.45
  point at 50, +0.89 at 75 and +0.45 at 100 nodes. The chosen R95 cell (`rrf1000_b25+FULL:h1_c100+top25`, ranked expansion of a
  25-node base) holds 0.969 on eval (frozen) and 0.970 (e3v) at 50 nodes against the old own construction's 0.954 / 0.945: ABOVE,
  +1.50 points [+0.53, +2.48] at the same price. At 0.98 the retrieval-only cell (60 nodes) is SHORT on eval (0.968 / 0.966); the
  link cell (75 nodes) is confirmed (0.988 / 0.981). Dev is 224 questions: the 60-node dev read (0.987) was optimistic by 1.8 points.
- **SQuAD**. Retrieval depth is the lever, not links. The fused full lists give 0.952 dev / 0.955 eval at **15 nodes** (`rrf1000_b15`);
  the smallest link cell at 0.95 is larger (41 nodes). At equal nodes the graph frontier is below the retrieval frontier at 50, 100,
  150 and 200 nodes (-0.2, -0.33, -0.13, -0.20 points) and above it by 0.1 point at 75 and 300: links buy nothing measurable here (the retrieval ladder has 37 depths, the graph cells 12 bases, so a small negative is the finer ladder, not harm).
  At 0.98: retrieval-only 75 nodes (eval 0.985), with links 63 nodes (eval 0.982). The call is ABOVE on the price rule (15 nodes
  against 50 at a pool that still holds 0.95), with ALL 2.4 points below the old cell's 0.980: it is a smaller pool, not a better one.
- **MuSiQue: cannot reach 0.95.** Best within the bound 0.801 dev (0.828 eval) at 1,083 nodes (`rrf1000_b800+FULL:h1_c100`); best at
  any size on the grid 0.845 at 2,612 nodes (`rrf1000_b800+FULL:h2_c25`); retrieval-only best 0.735 at 1,698 (eval 0.757). Links buy
  +4.5 to +11.6 points at equal nodes from 50 to 1,000 nodes and +6.6 at 2,000. ANY is 0.999 at the chosen cell and micro-coverage
  0.918: every question has a gold, the second or third is the one missing. The old cell (`equal_rrf_budget_200+FULL:h2_c25`,
  2,092 nodes) holds 0.803 on eval; the new one 0.828 at 1,064 nodes (ABOVE, +2.57 points [+1.41, +3.72] at 0.51 of the nodes).
- **MetaQA: reaches 0.95 only past the bound.** Retrieval-only is 0.27 at 1,895 nodes; typed structure is the whole result. Within the
  bound the best is 0.896 dev / 0.895 eval at 2,154 nodes (`equal_rrf_budget_200+STRUCT:h3_c25_v2000`); the smallest cell at 0.95 on
  the grid is `equal_rrf_budget_50+STRUCT:h3_c100_v5000` at **5,004 nodes** (0.961). The best STRUCT cell (0.961) is above the best FULL cell (0.911): NER reaches 0.000 and KNN 0.017 of MetaQA's
  missing golds in 3 hops. LEVEL against the old cell (+0.03 point, nodes 1.07x).
- **WebQSP: cannot reach 0.95.** Best within the bound 0.751 dev (0.764 eval) at 1,888 nodes (`dense_top200+FULL:h2_c100_v2000`); best at
  any size 0.802 at 5,100 nodes. Retrieval-only best 0.415. The questions that miss a gold miss about 8 each; ANY is 0.942 and
  micro-coverage 0.592 at the chosen cell. ABOVE the old cell (+7.58 points [+5.92, +9.38] at 0.89 of the nodes).
- **HotpotQA: reaches 0.95 with links at 77 nodes.** Retrieval-only never does (best 0.869 dev at 1,765 nodes, 0.879 eval). With the structural
  walk the R95 cell `rrf1000_b25+STRUCT:h1_c100+top100` holds 0.9615 dev at 77.3 nodes and **0.9627 eval (frozen), 0.9646 (e3v)** at 78 nodes (ANY 0.988,
  micro-coverage 0.975). The links buy +25.4 points at 50 nodes, +25.6 at 75 and 100, +24.3 at 150 and +14.3 at 1,000. R98 is `rrf1000_b100+STRUCT:h1_c100+top200`
  (0.981 dev at 153.5 nodes) and is SHORT on eval (0.9776 frozen, 0.9743 e3v). The highest ALL inside the bound is 0.9927 at 935 nodes
  (`rrf1000_b800+FULL:h1_c100`). The old cell (`equal_rrf_budget_50+STRUCT:h1_c25`, 93.7 nodes) holds 0.9684 on eval: the call is ABOVE on the second clause
  (nodes ratio 0.832 [0.828, 0.836], ALL -0.57 points [-0.84, -0.28]): **a smaller pool at slightly lower ALL, a price read like SQuAD's, not a better pool.**
- **2Wiki: 0.95 only at about 2,100 nodes.** Retrieval-only is 0.511 dev at 1,842 nodes (0.591 eval). The links buy 36 to 51 points at equal nodes from 50 to 1,000 nodes, but the
  graph frontier then flattens: 0.889 at 100 nodes, 0.908 at 200, 0.933 at 500, 0.936 at 1,000, 0.938 at 2,000, so the cells that reach 0.95 are hop-2 FULL walks over a 200- to 800-node base
  (`equal_rrf_budget_200+FULL:h2_c25`: **0.9492 dev at 2,129 nodes, SHORT of the floor by 0.0008; 0.9503 eval (frozen) and 0.9750 (e3v) at 2,140 nodes**; ANY 1.000,
  micro-coverage 0.979). The smallest cell at 0.95 at any size is `rrf1000_b800+FULL:h2_c25` (0.9545 at 2,696 nodes) and the best on the grid is 0.9552 at 4,074. The old cell
  (`equal_rrf_budget_50+STRUCT:h1_c25`, 105.6 nodes) holds 0.906 on eval: the call is ABOVE on the first clause (NEW reaches the floor where OLD does not; ALL +4.45 points
  [+4.10, +4.83]) **at 20.3 times the nodes [20.2, 20.3]**, inside the bound but not cheap. FULL reaches 0.982 of the missing golds within 2 hops and 0.998 within 3 (991 of 1,038 questions with a
  missing gold have all of them within 3), so the golds are reachable and the caps cut them (10.5).
- **The four e3v confirmations (read 07:15; nothing re-chosen).** The e3v vectors move the confirmed ALL by dataset: SQuAD's 15-node cell holds **0.9421 (SHORT of 0.95; frozen 0.9554)**
  and its R98 cell 0.9763 (SHORT); MuSiQue 0.8010 (frozen 0.8283; OLD 0.7911 on e3v); MetaQA 0.9281 (frozen 0.8952; OLD 0.9278 on e3v: LEVEL on both); WebQSP 0.7432 at 1,989 nodes (frozen
  0.7638; OLD 0.6447 on e3v). MultiHop-RAG (0.9699), HotpotQA (0.9646) and 2Wiki (0.9750) hold or rise on e3v. The call is on the frozen lists, as declared; the e3v column is the confirmation the
  pools will meet when PL1 builds them on the e3v layout.

### 10.4 The call

`call.json`: **CARRY_PL2 (read 7 of 7): BELOW 0, ABOVE 6 (MultiHop-RAG, SQuAD, HotpotQA, 2Wiki, MuSiQue, WebQSP), LEVEL 1 (MetaQA); the code-path hash is equal on
all seven and every chosen cell is inside the 2,500-node bound** (the largest, MetaQA's, holds 2,154 nodes on eval). This is the declared rule's outcome and it is read as the
rule states it: section 5's grid replaces the knee as the own-construction rule of all seven datasets. What the six ABOVEs are, one by one:
- **a better pool at the same or a smaller price**: MultiHop-RAG (ALL +1.50 points, same 50 nodes), MuSiQue (+2.57 at 0.51 of the nodes), WebQSP (+7.58 at 0.89 of the nodes);
- **a smaller pool at a lower ALL** (the second clause, a price read): SQuAD (-2.44 points at 0.30 of the nodes) and HotpotQA (-0.57 at 0.83 of the nodes);
- **the floor reached where OLD misses it, at 20 times the nodes**: 2Wiki (0.9503 against 0.9058; 2,140 against 106 nodes).
MetaQA is LEVEL (+0.03 point at 1.07 times the nodes). **Which datasets meet the target (ALL >= 0.95 on eval, frozen lists):** MultiHop-RAG (0.969 at 50 nodes), SQuAD (0.955 at 15, but 0.942 on e3v), HotpotQA
(0.963 at 78) and 2Wiki (0.950 at 2,140, 0.975 on e3v; dev 0.9492 is SHORT by 0.0008). **They do not:** MuSiQue (0.828 at 1,064), MetaQA (0.895 at 2,154; 0.961 only at 5,004 nodes) and WebQSP (0.764
at 1,891; 0.802 at 5,100): the call does not read these three as meeting the target. The six candidate changes of 10.6 are tested on all seven datasets in round 2 (docs/PL3_COVERAGE_ROUND2.md) before PL1 builds pools on this construction.

### 10.5 Why the short datasets are short (the filed reach blocks, M3A's populations)

The filed `reachability` blocks say where the golds missing from the first-stage union sit. They are ceilings for an uncapped walk;
the walk here is capped (first `per_seed_cap` neighbours in CSR order, first `per_frontier_cap` per frontier node, `visited_cap` in
order of appearance), so a cell is far below its ceiling when the caps cut before the gold.

| dataset | questions with a missing gold | STRUCT within 1 / 2 / 3 hops | FULL within 1 / 2 / 3 hops | best cell on the grid |
|---|---|---|---|---|
| MuSiQue | 1,003 of 2,417 | 0.262 / 0.770 / 0.935 | 0.464 / 0.892 / 1.000 | 0.845 at 2,612 (h2, base 800) |
| MetaQA | 36,160 of 39,138 | 0.118 / 0.486 / 0.996 | 0.119 / 0.551 / 0.998 | 0.961 at 5,004 (STRUCT h3, cap 100, visited 5,000) |
| WebQSP | 1,074 of 1,503 | 0.366 / 0.898 / 0.989 | 0.364 / 0.927 / 0.996 | 0.802 at 5,100 (FULL h3, cap 100, visited 5,000) |
| SQuAD | 54 of 11,873 | 0.074 / 0.519 / 0.741 | 0.111 / 0.722 / 0.981 | 0.998 at 1,545 (retrieval) |
| HotpotQA | 378 of 1,852 | 0.914 / 0.945 / 0.977 | 0.927 / 0.963 / 0.992 | 0.9927 at 935 (FULL h1, base 800) |
| 2Wiki | 1,038 of 2,096 | 0.878 / 0.930 / 0.987 | 0.921 / 0.982 / 0.998 | 0.9552 at 4,074 (FULL h2, base 200, cap 25, visited 5,000) |

- **MuSiQue** needs three hops of FULL (hop 2 reaches 0.89 of the missing golds, hop 3 all of them), and FULL beats every single family
  on the grid (STRUCT 0.738, NER 0.777, KNN 0.709, FULL 0.845). At these node counts the walk's depth is bought by the base: the best
  cell is a deep base (800) with a hop-2 walk, better than a 200 base with the deepest walk (0.822 at 5,119 nodes). The walk stops at
  caps that were set for the old 2,000-node pools.
- **MetaQA** and **WebQSP** are answer-entity questions (a question with a gold missing from the first stage misses about 7.7 and 8.1 golds). STRUCT within 3 hops
  holds 99.6% / 98.9% of them, but ALL needs every one of about eight, so ALL is the product of eight caps' luck; the pool that
  holds them is 2 to 5 thousand nodes. NER and KNN reach nothing here (WebQSP NER 0.025, KNN 0.032 within 3 hops) and only add nodes.
- **SQuAD** is the contrast: 54 questions in 11,873 miss a gold from the union, so there is nothing for links to buy; the
  frontier is the retrieval depth.
- **HotpotQA** is the case the links solve: 378 of 1,852 questions miss a gold from the union, the structural links (hyperlinks) reach 0.914 of them in one hop, and one hop at
  a cap of 100 gives 0.96 at 77 nodes. **2Wiki** has the reach (FULL 0.982 within 2 hops) but not the pool: ALL sits at 0.89 to 0.94 from 100 to 2,000 nodes and only hop-2 FULL walks over a 200- to 800-node base cross 0.95,
  so the pool pays 20 times the old price for the last 4 points; the first candidate of 10.6 (similarity-ordered caps) is the change aimed at this.

### 10.6 Candidate link-rule changes the diagnosis points to (names only; none is adopted, none is in the grid)

1. **Similarity-ordered caps**: the per-seed and per-frontier caps keep the neighbours with the highest question cosine instead of the
   first in CSR order (the cut that now drops golds before ranking sees them).
2. **Regime per hop**: STRUCT at hop 3 with NER and KNN confined to hop 1 (MetaQA, WebQSP: NER and KNN reach under 4% of the missing
   golds in three hops, and the best STRUCT cell on each is above the best FULL cell or within 2 points of it).
3. **Deeper base before the walk**: `rrf1000_b1200` to `b1600` with the hop-2 walk (MuSiQue's best cell is the deepest base the walk grid
   has, b800).
4. **Seed widening**: seeds from the top 10 or 20 of the fused list, not dense top 5 union splade top 5 (more first golds to walk from;
   MuSiQue ANY is 0.999 and ALL 0.80).
5. **Bridge-conditioned ranking**: rank the expansion by cosine to the question and to the base's best documents, for the second gold
   the question text alone does not point at.
6. **Visited cap by budget, not by order of appearance** (a cap on the expansion's rank, not on its arrival).

### 10.7 Surprises and contradictions with filed numbers

- Some filed numbers quoted in the task statement differ from the filed eval-population rows (probably the legacy-continuity rows);
  section 1 holds the eval-population rows. Every OLD number above is the filed construction read again by this engine on its own
  questions (G-K ties the engine to the filed cells on MuSiQue exactly; the OLD ALL on eval is MetaQA 0.895 against the filed 0.894,
  WebQSP 0.688 against 0.688, MuSiQue 0.803 against 0.803, SQuAD 0.980 against 0.980).
- SQuAD's R95 cell is 15 nodes and the retrieval-only frontier is on or above the graph frontier from 50 to 200 nodes: the claim that links lift
  the pools at the small end holds on MultiHop-RAG, MuSiQue, MetaQA and WebQSP and not on SQuAD.
- MuSiQue's dev carve is harder than its eval population (+2.7 points for the chosen cell, +3.5 for the old one); MetaQA's and
  WebQSP's agree within 1.3 points.
- The two dev grids written first (SQuAD, MuSiQue) carried a code-path hash computed after an edit of this file under the running job. The
  code that ran was the committed pool code; the check was added to `select` and both grids were re-run. The numbers read are the
  re-runs'; they equal the first runs' arrays element for element (see the note filed with the re-run in `outputs/pl2/rerun_check.json`).
- WebQSP's mapped vector pages put its resident set at 4.9 GB while its private memory stays at 0.5 GB; the host rows ask for the mapped size.

- **The 2Wiki rows were renamed before they ran.** The feeder groups rows by the first two words of the name; after one HotpotQA part of `pl2-g` finished it sized every later `pl2-g-*` row as needing another 60 GB and none fitted
  (nothing is logged for a skipped row). The 2Wiki parts were resent under new names with the same commands and inputs: `pl2-g-2wiki-0/1/2` as `pl2-gw-2wiki-0`, `pl2-gw-2wiki-1`, `pl2-gz-2wiki` and `pl2-cf-2wiki` as `pl2-fw-2wiki`; `outputs/pl2/feed_rows.txt` keeps the original names. The run was 06:58 to 07:13.
- **The e3v vectors change the verdict of two cells**: SQuAD's 15-node R95 cell falls to 0.9421 (SHORT) on e3v, WebQSP's loses 2.1 points and MuSiQue's 2.7, while MetaQA's gains 3.3 and 2Wiki's 2.5 (and the old cells move with them: MetaQA OLD 0.9278, 2Wiki OLD 0.931).
  The call is on the frozen lists as declared; the e3v column is the one PL1's pools will meet.
- **2Wiki's chosen cell is SHORT on dev by 0.0008 and holds 0.9503 on eval**: the rule's SHORT branch (highest ALL inside the bound) picked it, so the table marks it SHORT while the eval confirms the floor on the frozen lists.

### 10.8 Undone

Nothing of PL2's rows is undone: the 17 rows of `outputs/pl2/feed_rows.txt` ran (the three 2Wiki parts and the 2Wiki fix-up under the names of 10.7), `outputs/pl2` was fetched from the host, `zpl2g.py call` and `zpl2r.py` ran
on the laptop, and this section was re-filed. Open: round 2 (PL3), which tests the six candidate changes of 10.6 on all seven datasets before PL1's real e3v pool rows are built.
