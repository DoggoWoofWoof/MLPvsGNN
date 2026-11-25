# UL2H: the hop-coverage ledger on the e3v layout, all seven datasets (a scouting stage)

Declared 25 November 2026, before any UL2H number exists. **Nothing here trains, no model reads anything, no LLM or spaCy is used, no filed
graph, pool, fit or record changes, and nothing is adopted.** It measures and names. It is UL2S's second step (docs/UL2S_SEED_ANCHORED_LABEL_LEDGER.md);
UL2S's numbers (filed `d80bc69`) are the only numbers I have read of this line, and they are one hop from five seeds.

## 0. The user's words, and the rulings that bind

- 25 Nov: "coverage target is 95+ and as close to 100 as possible for the gold pools at the smallest node counts, links improve how much ever is
  possible". The lead's reading: per dataset, ALL-gold coverage (every gold of a question in the pool) >= 0.95, as near 1.00 as possible, at the
  smallest mean pool size.
- Relayed with it: "for each question's gold nodes that are NOT reached within 1/2/3 hops of the seeds (dense top5 U SPLADE top5, e3v lists), name
  which rule/label would have linked them (or none), and the price (candidates per anchor) of each label, so the candidate rule changes you name
  point at the unreached golds on MuSiQue / MetaQA / WebQSP / 2Wiki first."
- Earlier and still binding: "we need to have universal rules" (every rule on all seven, carried everywhere or nowhere); the hub line is CLOSED
  (UL2 8.15: no hub step is named, whatever the numbers show); no LLM, no spaCy, no model; declare before any number; adopt nothing.
- PL2 (docs/PL2_COVERAGE_TARGET.md, `zpl2g.py`) builds the extended coverage grid of **capped, ranked pools** from `outputs/m3a/headroom`. Its files and
  `scripts/m3a_headroom.py` are not touched here.

## 1. Why, and what this adds to PL2

PL2's filed grids (read in its section 1): the 95% floor is reached only on SQuAD (50 nodes), HotpotQA (93-94) and MultiHop-RAG (50-100); on 2Wiki at
2,140; MuSiQue (best 0.803), MetaQA (0.717) and WebQSP (0.655) never reach it. Those pools are **capped** (a per-node cap, a vertex budget, a
ranking). When a gold is missing from a capped pool two things can be true: the cap or the ranking dropped it, or **no pool built from those links
could ever hold it**. UL2H separates them: it measures the **uncapped** graph, hop by hop from the same seeds.

- If a gold is within h hops of the seeds in a unit's graph, a pool of that unit at h hops *can* hold it; what is left is a cap and ranking problem (PL2's).
- If it is not within 3 hops of the seeds in any unit (B0, or B0 plus any form), **no rule here can put it in a pool built from these seeds**: a rule
  change must name a different link, or the seeds are the limit. That list is what this stage hands to rule work.
- The price of each unit and hop (mean pool nodes) is the uncapped size; a capped pool is a subset of it, so these are the upper bounds PL2's nodes
  should be read against.

## 2. Seeds and population (fixed)

- **Seeds** = the dense top 5 U the SPLADE top 5 of the question, first occurrence kept, dense first (PL2 2a; 5 to 10 distinct rows). The dense list is
  ENC E0's e3v list (`outputs/enc/e0/e3v/<d>.npz`; MultiHop-RAG: PL1 S0a's `first_e3v.npz`); SPLADE is the served package's `splade_top1000.npz`.
- **Questions:** UL2's s1eval carve (MuSiQue 2,417, SQuAD 11,873, MetaQA 9,785, WebQSP 1,503, HotpotQA 7,405, 2Wiki 12,576; zero-gold questions
  excluded, none were), MultiHop-RAG's test split (1,130) with its dev split (224) beside. Same population and gates as UL2S (G-S1 to G-S5, which
  stop the run), loaded by UL2S's own functions.
- **Golds** are the grading only. A gold in the seeds is **found** (distance 0); the others are the **missed golds**.
- **Gates added:** G-H1 every question's union is 5 to 10 distinct rows in [0, n), the dense five first. G-H2 (external): the engine, run on the dense
  top five alone at one hop in the any-link unit, reaches the same missed and reached gold **counts** as UL2S's filed dense-list `any` unit on s1eval (`missed` and `reached`, equal as integers)
  wherever UL2S's ledger exists and carries that list (UL2S's `dense` list, `any` unit, version and carve as read there); where it does not, the check is
  recorded as not run. G-H3: per missed gold and per question,
  distance and pool size are monotone over the subset order of the units (section 3); a violation stops the run.

## 3. The graphs (units), and the walk

The graph is the whole e3v layout's, never a pool, never capped, undirected; a path may pass through any node and any pair of the unit. The versions
are UL2's (RK or SRK per dataset, as UL2S read them). Units, per version:

| unit | pairs |
|---|---|
| `label:S` (SRK only), `label:R.mention`, `label:R.name_comention`, `label:R.title_comention`, `label:K.knn` | each label alone, as UL2P's section 5 defines it |
| `any` | B0 = S (SRK) U R.mention U R.title and name co-mention U K.knn |
| `plus:F` for F in H5, HM, HK1, NTc, NT1, HK2, H10 | B0 U form F (the form's pairs beyond B0) |
| `all` | B0 U every form (UL2S's pseudo-unit) |

The subset order: any label's graph is a subset of `any`, which is a subset of `plus:F`, which is a subset of `all`.

**The walk** (per question, per unit): level 0 = the seeds; level h = the nodes first reached at h hops. Distance of a gold g: 0 if a seed, else 1, 2, 3,
or **4 (more than 3)**. Levels 1 and 2 are walked outward from the seeds; "within 3" is read from the gold's side (a neighbour of g at level <= 2), so
no three-hop ball is needed for the coverage measures.

## 4. The measures

Per unit and hop h = 1, 2, 3 (and the seeds alone at h = 0), over the questions (every measure a ratio of sums over questions; 95% percentile intervals,
2,000 resamples of questions, `zap4.boot`, chunks of 100, `default_rng(0)`):

| measure | definition |
|---|---|
| ALL | the share of questions with every gold within h hops (the PL2 target) |
| ANY | with at least one |
| recall | golds within h hops / golds (micro) |
| missed reach | missed golds within h hops / missed golds |
| nodes | the mean pool size, nodes within h hops including the seeds. **Exact for h = 1, 2.** For h = 3 exact where the expansion costs at most 2^24 gathered entries (and, on graphs over 100,000 nodes, only on every 10th question: indices divisible by 10); `nodes3_lb` = the mean of the exact size where measured and of the two-hop size elsewhere (a lower bound), with the **measured share** printed |
| price per seed | (mean nodes at hop 1 - mean seeds) / mean seeds: the unit's new distinct nodes per seed (UL2S's price counted pairs; this counts nodes) |

**The graph floors (PL2's R95 and R98, uncapped).** Per dataset and version, among all (unit, hop) cells with hop 1 to 3, **G95** = the cell with the
smallest mean nodes among those whose ALL point is at least 0.95 (nodes = the exact size at hops 1 and 2, `nodes3_lb` at hop 3; a hop-3 cell whose
measured share is below 1 is marked `bound`; ties to the higher ALL, then name order); **G98** likewise at 0.98. If no cell
reaches the floor the dataset is **SHORT_GRAPH**: no pool of these units at three hops from these seeds can reach it, whatever the cap. The best ALL cell
is printed beside.

**Where the missed golds stand** (all shares are of the missed golds, intervals over questions), for h = 1, 2, 3:
- **B0 distance:** the shares at distance 1, 2, 3 and 4 (more than 3) in `any`.
- **Label attribution** (the golds within h hops in `any`): for each label graph alone, the share within h hops; **ONLY_L** = reached by label L alone and by
  no other label alone; **MULTI_ALONE** = reached by two or more labels alone; **MIXED** = reached in `any` but by no single label alone (a path that
  needs two labels).
- **Rescue** (the golds NOT within h hops in `any`): the share reached within h hops by `plus:F`, per form; **ONLY_F** = by F and by no other single `plus`;
  **ALL_ONLY** = by `all` but by no single `plus`; **NONE** = not within h hops in `all`.
- **NONE split:** **ISOLATED** = no pair of B0 or any form touches the gold; **FAR** = it has pairs but is more than h hops from the seeds in `all`.

Pool prices of each unit are read beside these (the table of nodes per unit and hop), so each rescue has its price: the difference of mean nodes between
`plus:F` and `any`.

## 5. What it can and cannot show

- **It is graph-level and uncapped.** A gold within h hops can still be lost by a cap or a ranking; that is PL2's. A gold beyond three hops in `all` is
  lost for any pool of these units at three hops.
- **Hop-3 sizes** are bounds on the large graphs (above); the coverage measures are exact.
- **Hubs:** a hub inflates every size it touches; that is the price shown, and no hub step is named (closed).
- **Not R@5:** a gold a model already ranks gains nothing from a link; a weak link can help in combination.
- **Seeds are the record's first stage**, not a model's top rows. Four hops and more are not measured.
- **SQuAD** has one gold per question: ALL = ANY, and the class split is the same read as UL2S's.
- **MultiHop-RAG** has 609 nodes: its balls cover most of the corpus within two hops; its precision-like readings are not made here, and its ALL at hop 2 or 3
  is a small-corpus fact (UL2S 10.6).

## 6. The expectations and what follows (fixed before any number)

**Expectations.** Written before any UL2H number, **but not blind**: UL2S's one-hop reads of MultiHop-RAG and SQuAD are known (any-link reach of the missed golds
97.8% and 51.8% from five rrf seeds; question reach 96.8% on MultiHop-RAG). They decide nothing; the report marks each HELD, NOT_HELD or NOT_READ.

| id | expectation |
|---|---|
| H1 | ALL in `any` at 3 hops is at least 0.95 on MultiHop-RAG and on SQuAD |
| H2 | on those two, G95 is neither `any` nor `all`: a single label, or K.knn, holds the floor with fewer nodes than the whole any-link graph |
| H3 | on every dataset read, K.knn has the smallest price per seed at hop 1 of the label units that have pairs |
| H4 | on every dataset read where `any` leaves a missed gold beyond 3 hops, `all` reaches at least one of them |
| H5 | on MultiHop-RAG, ALL in `any` at hop 1 is at least 0.95 (UL2S's question reach makes this nearly known) |

**What follows, named now (names only; each is read on all seven; the lead's four first: MuSiQue, MetaQA, WebQSP, 2Wiki):**
- **The graph floors:** G95, G98 or SHORT_GRAPH per dataset and version, with the cell, its nodes (and bound) and the best ALL.
- **The gap:** per dataset, the share of the missed golds beyond 1, 2 and 3 hops in `any`; of those, the share rescued by `all`, by each `plus:F`
  (interval's low end above 0 is listed) with its node price; the residual NONE split into ISOLATED and FAR.
- **Which label links them:** the ONLY_L, MULTI_ALONE and MIXED shares.
- **The prices:** price per seed at hop 1 and mean nodes at hop 2 per label and unit.
- **Candidate rule changes (a name, not a carry):** a form F whose rescue share's low end is above 0 (add F to the pool graph, at its price); the label with
  the largest ONLY share (its weight or priority); on a SHORT_GRAPH dataset, that the seeds and these rules, not the cap, limit coverage, and that a new link
  family or a deeper seed set is needed. **No new family is named here** (no LLM, no spaCy, no model).
- **Hubs are closed;** the universal rule applies: every candidate is read on all seven and carried everywhere or nowhere. **No refit, no carry, no new form.**

## 7. Rows and runs

- **Laptop:** `python outputs/mp_unified/zul2h.py selftest`; then `ledger --ds multihoprag` and `ledger --ds squad`; then `report`.
- **Host, one row per dataset** (MuSiQue, MetaQA, WebQSP, HotpotQA, 2Wiki), 1 CPU, no GPU:
  `python outputs/host_ops/retry_cmd.py --tries 2 --wait 90 -- python outputs/mp_unified/zul2h.py ledger --ds D --host --threads 1`.
  - Each row depends on the dataset's `ul2c-grade-D` row (the same inputs as `ul2s-D`). Memory and time are estimates until the first measured peak; the
    run **checkpoints each (version, unit, carve)** under `outputs/ul2p/hops/<d>/ckpt/` (JSON, not committed) and resumes on a retry.
  - The graphs are built as int32 adjacency (about 8 bytes per pair) one unit at a time; WebQSP's any-link graph has 507 million pairs (UL2P's ledger).
- **`ul2h-report`** reads the ledgers and writes `outputs/ul2p/hops/report.json` and `report.md`.
- **No row is added to the feed by this stage's author;** the exact lines are given to the lead. The module is committed before they are sent.

## 8. Decisions (the user's; my default first)

- **D-H1, the seeds.** The union the lead named (dense 5 U SPLADE 5). **Default: the union only;** UL2S's rrf five is not repeated.
- **D-H2, depth.** Three hops. **Default: three;** a fourth is not measured.
- **D-H3, capped pools.** PL2's. **Default: not repeated here.**

## 9. Results

Read 25 November 2026 on the laptop with `outputs/mp_unified/zul2h.py` (committed `082cedc` before any run; the declaration is `29bb293`). MultiHop-RAG
and SQuAD are read (RK; MultiHop-RAG's test split decides, its dev split is beside; SQuAD's s1eval); **MuSiQue, MetaQA, WebQSP, HotpotQA and 2Wiki are host
rows and are not read** (section 7), so nothing below speaks for the four datasets the lead named first. Points in percent with 95% intervals over questions
unless a count is given; nodes are the mean uncapped pool size per question.

### 9.1 Runs and gates

- MultiHop-RAG 40.8 s, peak working set 0.09 GB; SQuAD 2,949.7 s, 0.7 GB. Hop-3 sizes are exact on both
  (graphs under 100,000 nodes, no question over the 2^24 budget: every `bound` is false).
- Gates passed: the loader gates (G-P1 to G-P3, G-S1 to G-S5, UL2S's), G-H1 (every union 5 to 10 distinct rows, dense five first: mean 7.4398 rows
  on MultiHop-RAG, 7.5134 on SQuAD), G-H2 (the dense five alone, one hop, any-link: missed/reached golds 1157/1127 on
  MultiHop-RAG's s1eval, 222/214 on s1sel, 1936/1000 on SQuAD: equal to UL2S's filed counts), G-H3 (18 subset pairs per dataset, no violation), and G-H4
  (in the code beside G-H1 to G-H3, not named on the declaration page: R.name_comention is inside F0T, so every label's graph is inside the any-link graph; the run stops otherwise). `zul2h.py` sha256 `a0fea9a22bcfda1ec2d2e0d31a1140225e0e89ef6af570b01b5ab077613a0b8a` (both ledgers).
- Population: MultiHop-RAG 1130 questions, 2,985 golds, 2,134 in the seeds, 851 missed (mean 7.4398 seeds); SQuAD 11,873 questions,
  one gold each, 10,900 in the seeds, 973 missed (mean 7.5134 seeds). Seeds only: ALL = ANY = recall 91.81 [91.29, 92.28] on SQuAD; MultiHop-RAG ALL 44.51 [41.59, 47.43],
  ANY 97.97 [97.08, 98.76], recall 71.49 [70.00, 73.03].

### 9.2 The expectations

| id | call | read |
|---|---|---|
| H1 | **HELD** | multihoprag: ALL in any at 3 hops 1.0; squad: ALL in any at 3 hops 0.99874 |
| H2 | **HELD** | multihoprag: G95 ('label:R.name_comention', 1, 273.884); squad: G95 ('label:K.knn', 2, 77.934) |
| H3 | **NOT_HELD** | multihoprag: cheapest label at hop 1 label:R.mention (0.0055); squad: cheapest label at hop 1 label:R.title_comention (0.1714) |
| H4 | **HELD** | squad: 15 beyond 3 hops in any; all reaches 13 |
| H5 | **HELD** | multihoprag: ALL in any at 1 hop 0.98584 |

H3 is NOT_HELD as declared and the reason is plain: the label with the fewest pairs has the smallest price per seed (MultiHop-RAG's R.mention has 2 pairs,
SQuAD's R.title_comention 2,809); K.knn's price per seed is 2.3234 on MultiHop-RAG and 3.0256 on SQuAD. H2 is HELD on MultiHop-RAG
by a hair that means nothing (G95 is R.name_comention at hop 1 with 273.884 nodes against 275.576 for the whole any-link graph: the same graph in effect), and on SQuAD
clearly (K.knn at 2 hops, 77.934 nodes against 6,512 for the any-link graph at 2 hops).

### 9.3 MultiHop-RAG (609 nodes: a small-corpus read, UL2S 10.6 applies)

Graph floors: G95 and G98 are label:R.name_comention at 1 hop, ALL 98.58% at 273.884 nodes; the best ALL is 100% (R.name_comention at 2 hops, 607.816 nodes of 609: nearly the whole corpus).

| unit | hop | ALL | recall | mean nodes |
|---|---:|---|---|---:|
| label:R.name_comention | 1 | 98.58 [97.88, 99.20] | 99.46 [99.19, 99.70] | 273.9 |
| label:R.name_comention | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 607.8 |
| label:R.name_comention | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.0 |
| label:K.knn | 1 | 65.22 [62.48, 68.14] | 86.16 [85.13, 87.26] | 24.7 |
| label:K.knn | 2 | 72.21 [69.65, 74.87] | 89.38 [88.48, 90.35] | 49.0 |
| label:K.knn | 3 | 75.22 [72.74, 77.79] | 90.52 [89.65, 91.39] | 90.2 |
| any | 1 | 98.58 [97.88, 99.20] | 99.46 [99.19, 99.70] | 275.6 |
| any | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 608.7 |
| any | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |
| all | 1 | 99.82 [99.56, 100.00] | 99.93 [99.83, 100.00] | 415.1 |
| all | 2 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |
| all | 3 | 100.00 [100.00, 100.00] | 100.00 [100.00, 100.00] | 609.0 |

The missed golds' distance in the any-link graph: 1 hop(s): 835 (98.12 [97.18, 98.95] %) ; 2 hop(s): 16 (1.88 [1.05, 2.82] %) ; 3 hop(s): 0 (0.00 [0.00, 0.00] %) ; beyond 3: 0 (0.00 [0.00, 0.00] %); none is isolated, and at 2 hops every missed gold is reached. One hop of the any-link graph covers
45% of the corpus (275.576 of 609 nodes), which is why ALL is high at every unit that holds the name co-mention label: coverage here is a corpus-size fact
(UL2S 10.6), not evidence about the rules. K.knn alone reaches 65.22 [62.48, 68.14] at one hop for 24.726 nodes; R.mention, with 2 pairs, adds nothing.
The dev split (s1sel) reads the same way (any-link ALL at one hop 97.77 [95.54, 99.55], G95 label:R.name_comention at 1 hop).

### 9.4 SQuAD (20,233 nodes; one gold per question, so ALL = ANY = recall)

Graph floors: **G95 = label:K.knn at 2 hops, ALL 95.19% at 77.934 nodes; G98 = any at 2 hops, ALL 98.80% at 6,512 nodes**
(a third of the corpus); the best ALL is 99.98% (all at 3 hops, 19,814 nodes, 98% of the corpus).

| unit | hop | ALL | recall | mean nodes |
|---|---:|---|---|---:|
| label:K.knn | 1 | 94.24 [93.81, 94.65] | 94.24 [93.81, 94.65] | 30.2 |
| label:K.knn | 2 | 95.19 [94.78, 95.59] | 95.19 [94.78, 95.59] | 77.9 |
| label:K.knn | 3 | 95.92 [95.56, 96.29] | 95.92 [95.56, 96.29] | 171.3 |
| label:R.name_comention | 1 | 94.24 [93.82, 94.65] | 94.24 [93.82, 94.65] | 108.6 |
| label:R.name_comention | 2 | 95.81 [95.46, 96.15] | 95.81 [95.46, 96.15] | 1,475.9 |
| label:R.name_comention | 3 | 97.61 [97.34, 97.88] | 97.61 [97.34, 97.88] | 8,151.7 |
| label:R.mention | 1 | 95.44 [95.06, 95.81] | 95.44 [95.06, 95.81] | 344.0 |
| label:R.mention | 2 | 97.08 [96.77, 97.37] | 97.08 [96.77, 97.37] | 3,336.3 |
| label:R.mention | 3 | 98.75 [98.54, 98.95] | 98.75 [98.54, 98.95] | 11,053.4 |
| any | 1 | 96.40 [96.06, 96.75] | 96.40 [96.06, 96.75] | 413.4 |
| any | 2 | 98.80 [98.59, 98.99] | 98.80 [98.59, 98.99] | 6,511.6 |
| any | 3 | 99.87 [99.81, 99.93] | 99.87 [99.81, 99.93] | 17,963.4 |
| plus:H10 | 1 | 96.98 [96.66, 97.28] | 96.98 [96.66, 97.28] | 469.0 |
| plus:H10 | 2 | 99.33 [99.17, 99.46] | 99.33 [99.17, 99.46] | 8,206.4 |
| plus:H10 | 3 | 99.96 [99.92, 99.99] | 99.96 [99.92, 99.99] | 19,141.4 |
| all | 1 | 97.30 [97.00, 97.59] | 97.30 [97.00, 97.59] | 544.5 |
| all | 2 | 99.51 [99.39, 99.63] | 99.51 [99.39, 99.63] | 10,487.9 |
| all | 3 | 99.98 [99.96, 100.00] | 99.98 [99.96, 100.00] | 19,813.7 |

The 973 missed golds by distance in the any-link graph: 1 hop(s): 546 (56.12 [52.83, 59.30] %) ; 2 hop(s): 284 (29.19 [26.41, 32.07] %) ; 3 hop(s): 128 (13.15 [11.00, 15.31] %) ; beyond 3: 15 (1.54 [0.82, 2.36] %). None is isolated in the all graph (0); the 15 beyond three hops
of the any-link graph: the all graph reaches 13 of them at three hops and 2 stay out (both FAR, with pairs but too far).

Which label reaches them (the 546 missed golds the any-link graph reaches at one hop; the other 427 of the 973 are the rescue question below): R.mention 431, R.name_comention 289, K.knn 289, R.title_comention 0; reached by one label
alone: R.mention 112, R.name_comention 65, K.knn 29; by two or more alone 340. At one hop K.knn reaches the same number of missed golds as R.name_comention
(289) for 30.246 nodes against 108.606. At two hops 79 missed golds are reached in the any-link graph by no single label alone (a path needing two labels).

What the forms rescue (missed golds the any-link graph does not reach, 427 at one hop, 143 at two):

One hop:

| form | missed golds rescued | share of the missed golds (%) | mean nodes added to any |
|---|---:|---|---:|
| H5 | 51 | 5.24 [3.80, 6.68] | 28.5 |
| HM | 38 | 3.91 [2.67, 5.14] | 15.8 |
| HK1 | 35 | 3.60 [2.47, 4.73] | 18.9 |
| NTc | 24 | 2.47 [1.54, 3.49] | 44.0 |
| NT1 | 35 | 3.60 [2.47, 4.73] | 29.5 |
| HK2 | 47 | 4.83 [3.60, 6.27] | 35.8 |
| H10 | 68 | 6.99 [5.34, 8.63] | 55.6 |

Two hops:

| form | missed golds rescued | share of the missed golds (%) | mean nodes added to any |
|---|---:|---|---:|
| H5 | 48 | 4.93 [3.60, 6.37] | 906.8 |
| HM | 33 | 3.39 [2.26, 4.62] | 485.9 |
| HK1 | 33 | 3.39 [2.26, 4.73] | 611.7 |
| NTc | 37 | 3.80 [2.67, 5.04] | 1,815.6 |
| NT1 | 38 | 3.91 [2.77, 5.14] | 1,069.0 |
| HK2 | 41 | 4.21 [2.98, 5.55] | 1,094.1 |
| H10 | 63 | 6.48 [4.93, 8.12] | 1,694.8 |

### 9.5 Named, not adopted (each is read on all seven before any carry; the lead's four first)

- **A cheap graph floor exists on SQuAD:** K.knn alone at 2 hops holds ALL 95.19% for 77.934 nodes on average, against 6,512 for the whole any-link graph at 2 hops (ALL 98.80 [98.59, 98.99]).
  PL2's filed grid reaches 0.95 on SQuAD at 50 nodes with a ranked, capped pool; these balls are uncapped, so the two sizes are not like for like, and a ranking can hold the floor
  at fewer nodes than a ball does.
- **Forms rescue golds the any-link graph cannot reach, at a price:** every form's rescue share has its low end above 0 at hops 1, 2 and 3 on SQuAD (section 6's rule for a candidate); at one hop
  HM adds 15.786 nodes for 38 rescued golds, H10 adds 55.606 for 68. Only NTc, NT1, HK2 and H10 reach golds no other single form does (ONLY_F at one hop:
  NTc 14, NT1 16, HK2 4, H10 8). On MultiHop-RAG forms rescue 14 golds at one hop and none at two (the corpus is covered).
- **Label priority:** at one hop on SQuAD R.mention reaches the most missed golds (431, of which 112 no other label reaches alone) and K.knn the most per node
  (289 for 30.246 nodes); on MultiHop-RAG R.name_comention carries the corpus. No weight or priority is named from two datasets.
- **No new family is named:** no gold is isolated on either dataset, and the golds beyond three hops are few (15 on SQuAD, 0 on MultiHop-RAG); the open question is MuSiQue, MetaQA, WebQSP and 2Wiki, where PL2's grids fall short.

### 9.6 What to distrust

- These are the two datasets whose target PL2 already meets (SQuAD 50 nodes, MultiHop-RAG 50-100): the read says the graphs hold the golds there; it says nothing about MuSiQue, MetaQA, WebQSP and 2Wiki, the four the lead named, where PL2's grids fall short of the floor.
- Mean sizes are over all questions; a hub seed inflates a ball and no median is read. A ball is not a pool: coverage here is what a pool of that unit *could* hold.
- MultiHop-RAG's balls reach most of a 609-node corpus within two hops: its floors are a corpus-size fact. H2's margin there is 1.7 nodes.
- The hop-3 rows of SQuAD's large units are balls of 89% of the corpus: ALL near 100% there is again corpus size.
- The floor cells are chosen on points, not intervals (section 4); a cell whose interval crosses 0.95 is still a floor cell.
- The seeds are the record's first stage (dense five and SPLADE five), not a model's top rows; four hops and more were not walked (distance 4 = beyond three).

### 9.7 Still to read

MuSiQue, MetaQA, WebQSP, HotpotQA and 2Wiki: host rows `ul2h-<dataset>` (section 7), then `ul2h-report`. H1 to H5 are called again on all seven then; the graph floors, the gap and the candidate lists of section 6 are
filed for the four datasets the lead named first only when those rows are read.

## 10. Development log (disclosed before any UL2H number)

- No `zul2h` code existed before this page. Before it I read PL2's page (its section 1 for the filed grid, 2a for the seeds) and the filed pair counts in
  UL2P's ledgers (`b0` sizes: MuSiQue 3.39M, MetaQA 0.35M, WebQSP 507M, MultiHop-RAG 25K pairs); no hop measure was computed.
- UL2S's and UL2P's code and filed numbers are unchanged.
- The declaration was committed first (`29bb293`), then `zul2h.py` (`082cedc`) with its self-test PASS (set loops on random graphs and on MultiHop-RAG's real any-link and
  all-forms graphs, two wrong engines caught, hand-made measures, G-H2 against UL2S's filed MultiHop-RAG counts); no UL2H number was read before that commit. A review of the
  module before its first run fixed slips in the module and its self-test (three hand-made expected values, a toy population that could repeat a row, a pair count that was 0
  for a resumed checkpoint, one measure that differed on resume and was dropped, the MultiHop-RAG population selected with the decided cell); the self-test passed on the fixed code.
- MultiHop-RAG was read first (40.8 s) and the report code was run once on that ledger alone to check its format (it then called H4 NOT_APPLICABLE: no missed gold beyond
  three hops); SQuAD followed (2,949.7 s) and the report was rebuilt from both ledgers, which is the report filed here. No code changed after the first run. The checkpoints
  (`ckpt/`) are working files, not filed.
- The hop-3 sizes are exact on both datasets (no `bound`, measured share 1); the sampling and budget rules of section 4 did not fire.
- The code carries a fourth gate, G-H4 (R.name_comention inside F0T), that section 2 does not name; it only stops a run, it moves no number.

## 11. Files

- `outputs/mp_unified/zul2h.py`
- `outputs/ul2p/hops/<dataset>/ledger.json`: aggregates and the record (shas, gates); the per-unit checkpoints (`ckpt/`) are working files, not filed
- `outputs/ul2p/hops/report.json` and `report.md`
