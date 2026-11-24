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

(not filed; the runs have not been made)

## 10. Development log (disclosed before any UL2H number)

- No `zul2h` code existed before this page. Before it I read PL2's page (its section 1 for the filed grid, 2a for the seeds) and the filed pair counts in
  UL2P's ledgers (`b0` sizes: MuSiQue 3.39M, MetaQA 0.35M, WebQSP 507M, MultiHop-RAG 25K pairs); no hop measure was computed.
- UL2S's and UL2P's code and filed numbers are unchanged.

## 11. Files

- `outputs/mp_unified/zul2h.py`
- `outputs/ul2p/hops/<dataset>/ledger.json`: aggregates and the record (shas, gates); the per-unit checkpoints (`ckpt/`) are working files, not filed
- `outputs/ul2p/hops/report.json` and `report.md`
