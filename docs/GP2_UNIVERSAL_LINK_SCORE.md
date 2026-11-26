# GP2 — the best universal link-keeping rule: question windows, text windows, per-hop caps, one exact program

Declared 26 Nov 2026, before any GP2 number exists. Code: `outputs/mp_unified/zgp2.py` (committed with this page). Imports `zgp1.py` (sha256 `0a9c6bc0026f71849fc4cacee333f9665390ce2bb3616ad9e338d7abe9df2c14`) and `zgp1p.py` (`062986dcc0b6f5f5020c3094523228976f86839563dcf8e9d3fe851e80730839`) read-only. Neither is edited.

## 1. Why this page exists

The user, 26 Nov, answering whether the base layout is the final link set: **"no we need to find the best possible universal rule maybe use windows etc a better scoring idk"**, and, shown the readings below: **"try everything"**. Standing rulings that bind this page: universal rules only, one rule with the same parameters on every judged cell; coverage 95 percent or more at the smallest node count; S never removed (A.1); subset first, then the full run once; no learning until the pools are final; no LLM, no cross-encoder; the GPU is not ours; the declared call before the number; a result that fails an expectation is reported as failed.

GP1 (`docs/GP1_GOLD_LINK_PATTERNS.md`, quarter filed 838f199a) asked whether one keep-rule over the link's **labels and its own attributes** takes weight off the walks. On the quarter it priced 95 percent reach at 0.99997 of the candidates (K0, the old key, 1.000): the attributes of section 5 do not separate the links gold paths need. Two things GP1 did **not** test, and this page does:

1. **Question-conditioned scores.** The only question-conditioned attributes GP1 computed were `votes` and `qov`. The question's own first-stage rank of the tail, which GP1's first draft of section 15 wrongly credited to A3, was never extracted (GP1 section 15.8 records the correction: A3 `rank` is the tail's rank among its **source's** frontier links by the pair dot of two stored rows).
2. **The hop.** GP1's keys had no step. The frontier grows with each hop, so most candidate weight sits at steps 2 and 3, while the seeds' own links (step 1) are few. A rule that keeps all of step 1 and windows steps 2 and 3 was not expressible.

## 2. What "windows" and "a better scoring" could mean, and what is in

The words do not say. Readings, each marked:

| reading | what it is | status |
|---|---|---|
| **A. rank windows** | keep a link if its tail sits inside a window of the question's own first-stage list: the dense list, the SPLADE list, or the better of the two (windows 1-3, 4-10, 11-30, 31-100, 101-300, 301-1000, beyond); the same windows over the source's frontier ranked by pair dot (GP1's exact A3 rank, finer bins: a per-source top-W cap) | **in** (attributes `dr`, `sr`, `ur`, `pr`) |
| **B1. text windows** | words around the tail's mention in the head's text: how many distinct content tokens of the question lie within 80 characters either side of the first mention of the tail's title (no IDF) | **in** (attribute `tw`; CPU, laptop) |
| **B2. embedding windows** | e3v similarity between the question and the mention window (E6's 64-token windows) | **costed row only, not run**: no window vector is stored (windows were never stored; the package holds no e3v vector of a dev question), and encoding them needs the GPU, which is not ours. The cost is measured and written in the report, nothing is encoded |
| **C. per-hop windows** | the cap differs by hop: W_1, W_2, W_3 per group, i.e. the step is part of the key | **in** (the unordered key dimension `step`; tied-steps run beside) |
| D. the node-budget window | a window of 2,500 nodes around the seeds | not a rule; it is the pool bound the pools (PL) already carry |
| a learned scorer, a cross-encoder, an LLM scorer | a better score by training or by a model | **out** by the rulings (no learning until the pools are final; cross-encoder dropped 23 Nov; no LLM shortcuts) |
| SEM2 R8's scorer W2 | the question's own cosine to the tail as a walk-time router | **overlap, stated**: `dr`/`sr`/`ur` read the same first-stage lists R8's question-conditioned routing builds on. GP2 claims nothing R8 will read, and R8's Part B is not touched |

"A better scoring" is read as three things: a **question-conditioned** score (A, B1), a **rank-sum** of several ordered attributes kept above a threshold or inside a window (the T form), and the **exact optimal keep set** over a finite family (the M form); never as anything learned.

## 3. What can be read (checked by listing, before any number)

- **Package retrieval cache** (`CRAG/data/final_canonical/<d>/retrieval_cache/dense_top1000.npz`, `splade_top1000.npz`), all six non-MultiHop-RAG datasets: keys `ids` (int32) and `scores` (float16), K = 1000 per query, query rows in `queries/query_ids.json` order. The walk's seeds are the top 5 of each. These are the package's frozen lists (the lists the dev grids of PL read); the dense list is the package encoder's, not e3v.
- **MultiHop-RAG** (609 nodes): PL1 S0a's complete e3v dense lists (`outputs/pl1/multihoprag/RK/s0/first_e3v.npz`, sha-checked against `first.json`) and the package's SPLADE list; the dense list there is e3v.
- **Node texts**: `nodes.jsonl` per dataset (title, text; a KB node's title is its text).
- **GP1's link rows**: `outputs/gp1/<d>/work/<v>_<part>_<c0>.npz`, one row per (question, source, tail, step) for every needed unit and every sampled control, with `kind`, `w`, `step`, `grp`, `dot`, `rank` (exact within the source's full frontier), `votes`, `exact`, `off`, the gold arrays and incidences. The quarter's chunks are here; the rest-of-dev chunks arrive when the rows `gp1xr-*` finish. **GP2 runs entirely on these rows: no walker, no host row, no GPU.**
- **Not read, not available**: any e3v vector of a question or a window; the dense scores are not used (see `qg` below).

Facts looked at before this page (infrastructure only, no GP2 quantity): file keys and shapes above; that the laptop rebuilds each dataset's dev carve (MuSiQue 1,534, SQuAD 1,498, MetaQA 1,497, HotpotQA 1,508, 2Wiki 1,496 questions with a gold); the row counts of GP1's quarter extract records.

## 4. The program

**Data.** Nine judged cells j (MHR RK, MuSiQue RK, SQuAD RK, MetaQA SRK, WebQSP SRK, HotpotQA RK, HotpotQA SRK, 2Wiki RK, 2Wiki SRK). In cell j, rows i: needed units (kind 1, weight w_i = the sum of 1/|N(g)| over the golds the unit serves; price weight 1) and sampled controls (kind 0, weight w_i = exact/sampled, in both reach and price). Exactly GP1's tables.

**Keys.** For an attribute set σ of ordered attributes (section 5) the key of a row is k = (u, o): u = (group, step) with group ∈ {S, R, K} (S > R > K dedup, GP1's `grp`) and step ∈ {1, 2, 3}, nine unordered groups; o = the vector of σ's bins. The number of keys is 9 × Π bins.

**Per cell and key**: r_jk = (needed weight in k) / (needed weight in the cell); p_jk = (price weight in k) / (price weight in the cell) (GP1's `columns`).

**Variables**: x_k ∈ {0,1}; τ ≥ 0.

**Program** (one x for all nine cells; ρ = 0.95):

    minimise τ
    (C1) for every cell j:  Σ_k r_jk x_k ≥ ρ
    (C2) for every cell j:  Σ_k p_jk x_k ≤ τ
    (C3) x_k = 1 for every key of group S (A.1; S protected). S free is run beside.
    (C4) x_k ≤ x_k' for k' one bin better than k in one attribute, everything else equal (monotone: a better-ranked link is never dropped in favour of a worse one)
    (C5) no cell index on x: the rule is universal. Own optima per cell are run beside.

A second stage minimises Σ x at the optimal τ (fewest keys); a key no quarter row falls in is removed by that stage. **The rule applied to the rest of dev is exactly the kept set x**: C4 is written over every key, seen or not, so x is already an upper set (a key at least as good in every attribute, inside the same group and step, as a kept key is kept, whether or not the quarter ever had a row in it); a key the quarter never saw and that is above no kept key is removed by the second stage. The needed weight the rest of dev has in keys the quarter never saw is reported per cell as "needed weight in keys unseen on Q", so a reach lost there is visible, not hidden.

**Solvers** (all exact on the declared family; HiGHS, relative gap 0, 600 s):

- **L**: the LP relaxation (a bound; its duals show which cell's reach binds). **LP is never the result.**
- **U**: the integer program without C4 (the best any partition of the declared keys can do: a bound for M).
- **M**: the integer program with C4 (**the rule**).
- **T**: the **rank-sum** form: score s(row) = Σ over σ of the row's bin index (0 worst); keep when s ≥ θ_u, one integer threshold per non-S group u (six of them); exact by enumeration over all threshold vectors (vectorised, in blocks). T is a feasible point of M, so τ_M ≤ τ_T. A one-attribute σ = `pr` is exactly "keep the top W frontier links of each source, W per group and hop".
- **Chain check (G-C1)**: τ_L ≤ τ_U ≤ τ_M ≤ τ_T on every space, else the run stops.

**Beside the call, on the final space**: S free (cost of A.1); ρ 0.90 and 0.99; size caps of 12, 24, 48 keys; the 9 own optima (cost of universality = τ_M minus the largest own optimum); the frontier (the best reach the worst cell can have at price caps 0.30 to 0.80); the tied-steps program (the same x on steps 1, 2, 3 of a group: what per-hop caps buy); and GP1's reference value 0.99997.

## 5. The declared family

Ordered attributes (bin 0 is the worst, the highest index is kept first; every direction is + and fixed here, not read from data):

| id | name | definition | bins (worst to best) |
|---|---|---|---|
| A-1 | `dr` | rank of the tail in the question's **dense** list (the list whose top 5 seeds the walk; 0 = absent) | out / >1000, 301-1000, 101-300, 31-100, 11-30, 4-10, 1-3 |
| A-2 | `sr` | the same in the question's **SPLADE** list | same 7 |
| A-3 | `ur` | the better (smaller) of `dr` and `sr` | same 7 |
| A-4 | `pr` | GP1's A3 exact rank of the tail among its source's frontier links by pair dot (source-relative; question-independent) | >1000, 301-1000, 101-300, 31-100, 11-30, 4-10, 1-3 |
| A-5 | `vo` | GP1's A4 votes (question-conditioned: nodes of the previous ball that link to the tail) | 1, 2, 3+ |
| B-1 | `tw` | distinct content tokens of the question in the window of 80 characters either side of the first mention of the tail's title in the head's text, the title's own tokens removed (`mention_stats`' first offset; no IDF); no mention at all is the worst bin | no mention, 0, 1, 2+ |

Not declared, and why: **`qg`** (the dense score's gap to the question's top score): the dense list is the package encoder's in six datasets and e3v in MultiHop-RAG, so equal cut points would not mean the same thing; the rank is scale-free and carries the question-tail relation. **`dot`, `qov`, `dir`, `exact`, `ntok`, `caps`, `lead`, `off`, `cnt`, `rel`**: GP1 screened them SIGN_SPLIT, WEAK or NO_SIGNAL; GP2 does not re-enter them (`dir` and `exact` are what `tw`'s "no mention" bin encodes). **Degree, hub, IDF** attributes are excluded as in GP1.

**Spaces.** Phase 1 (A + C, no text): σ0 = no attribute (nine groups, keep or drop each group at each hop); the 5 singletons of {dr, sr, ur, pr, vo}; the 10 pairs: 16 spaces. Phase 2 (B1): the singleton {tw} and the 5 pairs {tw, x}: 6 spaces. **Forward triples**: the best pair of the whole family extended by each remaining attribute: at most 4 spaces. At most 26 spaces; every one gets L, U, M, T.

**The final rule** = the space with the smallest τ_M over the whole family (ties: fewer kept keys, then fewer attributes, then the order above); it gets the beside runs and the grade. Phase 1's best space and σ0 are graded beside it.

## 6. Procedure and order

1. **Columns** (`zgp2.py cols --part q`): `dr`, `sr`, `ur` for every row of the nine quarter cells, from the package lists (laptop, mmap of the stored `ids`; the question's row from the dev carve exactly as `zgp1.dev_slice` builds it); then `tw`. Aligned with the chunk rows, checked by a hash of (q, a, c).
2. **Phase 1 on the quarter**, filed as it stands (`outputs/gp2/rule_q_phase1.json`); **phase 2 on the quarter** (`tw`), filed (`rule_q_phase2.json`); the final space is chosen by the rule above (`rule_q.json`).
3. **Commit the rule record before any rest-of-dev row is read.** The rest of dev (GP1's rows `gp1xr-*`) is read **once**, for the final rule, phase 1's best and σ0.
4. No s1eval anywhere; no new host row; no GPU; nothing trains.

**Gates** (a failed gate stops the run and is reported, never worked around):
- **G-Q0** mapping: for every question, every step-1 source of its rows lies in its dense top 5 ∪ SPLADE top 5 as rebuilt here (zero exceptions), and the number of dev questions equals the extract record's.
- **G-Q1** lookups: 2,000 random rows per cell (all if fewer): `dr` and `sr` equal a naive scan of the stored list.
- **G-W1** windows: 1,000 mention rows per cell: `tw` equals a second implementation (regular-expression word search, token window) of the same definition.
- **G-C1** chain (section 4). **G-C2** self-test: M and T equal brute-force enumeration on toy programs (done before this page's first number). **G-C3** `zgp1.py` and `zgp1p.py` have the shas above.
- **G-C4** tables: each cell's needed weight and row counts equal the extract record's, and the cell's needed weight equals GP1's `screen_q.json`.
- Chunks flagged CAPPED are NOT_READ, as in GP1 (WebQSP capped golds 1.7 percent on the quarter).

## 7. Calls

SCREEN_IN (per cell, as GP1): reach ≥ 0.92, price ≤ 0.50, gold path retention ≥ 0.90. **UNIVERSAL_RULE** iff the final rule is SCREEN_IN in all nine cells on the rest of dev; **NOT_UNIVERSAL** otherwise. Reported beside the call (not calls): τ_M of every space, each cell's own optimum, the frontier, per-dataset needed-versus-control tables for each new attribute (share of needed and of control weight per bin; the stratified AUC with its interval as in GP1, as a diagnostic), the per-step split of needed and candidate weight, and which cells bind (duals). Passage cells and KB cells are reported separately; a **class** rule (one rule for passage datasets, another for KBs) is not a result of this page: the universal ruling excludes it, and I name it for the user if the data point there.

## 8. What I expect (written before any GP2 number)

The binding facts I know from GP1: the S links alone are 0.30 to 0.40 of the worst cell's candidates (the frontier at price cap 0.30 was infeasible with S protected), MetaQA's first stage finds 14 of 3,228 golds and its needed tails are unrelated to the question's words, GP1's best price was 0.99997. So:

1. **X1: τ_M of the final space is at most 0.90** (some universal gain over GP1's 0.99997): 55 percent. **At most 0.60: 15 percent. At most 0.50 (the price line): 5 percent.**
2. **X2: the reach rows that bind (positive duals) include MetaQA or WebQSP**: 80 percent. A question rank window cannot reach a KB answer node that shares no words with the question.
3. **X3: at least three of the five passage cells (MHR, MuSiQue, SQuAD, HotpotQA, 2Wiki) have an own optimum at or below 0.50** (rank windows work where the gold shares words with the question): 70 percent; the cost of universality is then 0.3 or more: 70 percent.
4. **X4: the best pair contains at least one of `dr`, `sr`, `ur`**: 70 percent; `pr` alone beats `vo` alone: 60 percent.
5. **X5: per-hop caps matter**: the tied-steps program is at least 0.02 worse than the per-hop one on the final space: 60 percent.
6. **X6: `tw` adds at least 0.02 over phase 1's best**: 25 percent (the mention window overlaps what `ur` already sees).
7. **X7 (the call)**: **NOT_UNIVERSAL on the rest of dev: 95 percent**. On the rest of dev reach is 0.92 or more in at least seven of nine cells: 55 percent (a program at its reach constraint on the quarter gives back a few points on new questions); price within 0.05 of the quarter's in at least seven of nine: 70 percent.
8. **X8: the best rule's gain is in the passage cells**: the prices of MuSiQue, SQuAD, HotpotQA and 2Wiki on the rest of dev are at least 0.20 below the KB cells'.

**What each outcome means for the pools.**
- **UNIVERSAL_RULE (5 percent)**: the rule becomes the link filter; PL8 may be declared with it, on the user's word that the link set is final.
- **NOT_UNIVERSAL with a clear gain (price 0.60 to 0.90)**: a universal rule exists but not at the 0.50 price; I give the frontier (best reach at 0.50) and the user decides whether to trade reach or price, or to treat the KB cells separately, before any pool is rebuilt.
- **NOT_UNIVERSAL with no gain (price 0.90 or more)**: no universal rule in this family either; the link set stays base, PL8 stays undeclared, and the lever is not a link filter at all; the next named step would be a walk-time order under the node budget (R8's line), not a rule here.
- In every case the per-dataset tables say what a rule would look like where it works; they are findings, not rules.

## 9. What this page does not do

No s1eval; no degree, hub or IDF attribute; no learning; no e3v window or question vector is encoded (B2 is costed only); no edit of `zgp1.py`, `zgp1p.py`, `zsem2*`, `zpl*g`, `zul2*`, `zul3a`, `zul4lp`, D83 code, `zenc*`, `chaincov`, `chaindiag`, `hopdiag`; no host row; no change to SEM2 R8's files or rows. Link and pool layouts stay base. Aggregated records go under `outputs/gp2/` and are committed; link-level columns are not.

## 10. Amendments (dated; none changes a rule, a gate or an expectation)

- **A1, 26 Nov, before any GP2 number was read.** The first run of `cols --part q` passed G-Q0 and G-Q1 on MultiHop-RAG (20,614 rows) and then stopped on MuSiQue: loading MultiHop-RAG makes `zmhr` replace `zenc.served_root`, so a process that has loaded MultiHop-RAG cannot read the package of another dataset. `zgp2.py cols` and `text` now re-invoke themselves once per dataset (one process, one dataset). The MultiHop-RAG column file written by the first run is rewritten by the second run from the same code path. No program was solved and no table was read.

## 11. Quarter result, family 1 (A rank windows, B1 text windows, C per-hop caps), filed 26 Nov

Gates: G-Q0 and G-Q1 (the question's lists, 2,000 random rows per cell against a naive scan) and G-W1 (the text window against a second implementation, 1,000 level-3 rows per cell) pass on all nine cells; G-C4 (needed units and weight equal the extract records and GP1's screen) passes. Records: `outputs/gp2/rule_q_phase1.json`, `rule_q_phase2.json`, `rule_q.json`, `diag_q.json`, `<dataset>/gates_q.json`. Every space solved exactly (M: OPTIMAL, gap 0; chain τ_L ≤ τ_U ≤ τ_M ≤ τ_T verified).

**τ_M by space** (the smallest worst-cell price at reach 0.95 in all nine cells, S protected, monotone): groups only 0.9996 (S0, tied nowhere: 9 keys, per hop). Singletons: dr 0.9954, ur 0.9956, pr 0.9961, sr 0.9965, vo 0.9993, tw 0.9962. Pairs: 0.9953 for dr+sr, dr+ur, dr+pr, tw+dr; 0.9954 for dr+vo, ur+pr, tw+pr; 0.9955 sr+pr, tw+ur; 0.9956 sr+ur, ur+vo; 0.9958 tw+sr; 0.9961 pr+vo; 0.9962 tw+vo; 0.9965 sr+vo. Forward triples on the best pair (tw+dr): +sr, +ur, +pr, +vo all 0.9953 (tw+dr+pr 0.99529, the smallest, 1,140 kept keys of 1,764, is the **final space** by the page's rule). GP1's reference was 0.99997. The LP relaxations sit at 0.77 to 0.95: the gap between the relaxation and the integer, monotone program is the whole story below.

**Final rule on the quarter** (reach / price per cell): MultiHop-RAG 0.996 / 0.995; MuSiQue 0.950 / 0.962; SQuAD 0.957 / 0.982; MetaQA 0.965 / 0.864; WebQSP 0.986 / 0.992; HotpotQA RK 0.983 / 0.943 and SRK 0.990 / 0.916; 2Wiki RK 0.980 / 0.918 and SRK 0.986 / 0.894.

**Beside the call.** Own optima (each cell alone, same family and constraints): MultiHop-RAG 0.296, MuSiQue 0.147, SQuAD 0.188, HotpotQA RK 0.0018 and SRK 0.089, 2Wiki RK 0.0007 and SRK 0.033, **MetaQA 0.718, WebQSP 0.987**. The cost of universality as the page defines it (τ_M minus the largest own optimum) is 0.0084 because WebQSP's own optimum already sits at 0.987. S free: 0.99529, the same (protecting S costs nothing here). ρ 0.90: 0.99525; ρ 0.99: 0.99782; LP relaxation at ρ 0.94 and 0.96: 0.727 and 0.799. Size caps of 12, 24 and 48 keys: infeasible, an artifact of this key space (the protected S keys alone number 588). Frontier (best reach the worst cell can have under a price cap): cap 0.30 infeasible, 0.40: 0.694, 0.50: 0.751, 0.60: 0.755, 0.70: 0.758, 0.80: 0.766. Tied steps (one x for the three hops of a group): 0.99993, so per-hop caps buy 0.0046; groups only with tied steps: 1.0000. Duals of the LP's reach rows: MetaQA 3.16, MuSiQue 0.54, SQuAD 0.001, the rest 0.

**What the tables say, per dataset** (needed versus control share per bin; stratified AUC with its interval, in `diag_q.json`). Passage cells separate under ordinary attributes: MultiHop-RAG dr 0.87, ur 0.87; SQuAD dr 0.83, ur 0.83; HotpotQA ur 0.73 to 0.75 and tw 0.67 to 0.68; MuSiQue ur 0.68, dr 0.66; 2Wiki tw 0.76, pr 0.65 to 0.66, ur 0.59. KB cells do not: MetaQA dr 0.51, sr 0.47, ur 0.50, pr 0.43 (below one half), tw 0.46; WebQSP pr 0.69, dr 0.59, ur 0.59, vo 0.61. Where the weight sits: 53 to 99 percent of each cell's needed weight is at hop 1 (MetaQA 53 percent, the rest at hops 2 and 3), but in eight of the nine cells 80 to 99 percent of the candidate weight is at hop 2 (MultiHop-RAG: 92 percent at hop 1), and group R holds 91 to 100 percent of it in eight cells (MetaQA: R 59 percent, S 39 percent): the walk's cost is the hop-2 R frontier, and WebQSP and MetaQA need part of it that no declared attribute marks.

**Expectations, scored.** X1 (τ_M at most 0.90): **failed** (0.9953; I gave it 55 percent). X2 (a KB cell binds): true in the relaxation (MetaQA dual 3.16) and in the integer program through WebQSP's own optimum. X3 (three of five passage cells have an own optimum at most 0.50): true, all five do (0.0007 to 0.296); the second clause (cost of universality at least 0.3): **failed** as defined (0.0084), true against the passage optima (gap 0.70 to 0.99). X4 (the best pair contains a question rank): true (tw+dr; dr+x for every x within 0.0001); pr alone beats vo alone: true. X5 (per-hop caps buy at least 0.02): **failed** (0.0046). X6 (tw adds at least 0.02 over phase 1's best): false, as expected at 75 percent (0.0000). X7 and X8 wait for the rest of dev (X8 already looks failed: on the quarter MuSiQue 0.962 and SQuAD 0.982 are not 0.20 below MetaQA 0.864).

**Reading.** In this family there is no universal rule: the program's answer is the page's second no-gain branch (price 0.90 or more), the link set stays base, PL8 stays undeclared. The finding is where the program is stuck: two KB cells, WebQSP (own optimum 0.987) and MetaQA (0.718), whose needed hop-2 links the question's words, the windows, the source-relative rank and the hop do not mark, while every passage cell has an own optimum at or below 0.30. A class rule (passage versus KB) would sit far below 0.50 for the passage cells; the universal ruling excludes it from this page, the data point is the own optima above.

## 12. Amendment A2: rule labels, two stored attributes, and the class split (26 Nov, declared after the family-1 quarter numbers of section 11 were read and before any number of this amendment)

**Why.** Section 11 leaves the program stuck on WebQSP and MetaQA, and shows 91 to 100 percent of the candidate weight in group R at hop 2 in eight cells. The group dimension lumps three different rules into R; the user's ruling is that an edge is labelled by the rule that made it, and "try everything" says the labelled keys have not been tried. GP1's per-row columns `qov` and `dir` were also never in a GP2 space. This amendment adds exactly these and nothing else. It is a second family chosen after the first family's numbers were read, on the quarter; the choice between families is made on the quarter alone and the rest of dev is read once (item 6), so the amendment cannot leak the rest of dev, but it is a wider search on the quarter than section 4 declared, and section 11's final is kept as the reference.

1. **Label dimension `lab`** replaces the group dimension `grp` in phase 3: S, W (R.mention), CN (R.name_comention), TI (R.title_comention), K, where a link that several rules made takes the first of that order (S, W, CN, TI, K: the extract's profile bits, zul2p.base_units' order). `lab` is unordered and is crossed with the hop: 15 groups. S is protected as before. **Gate G-L1**: `lab` maps onto the stored `grp` on every row of every cell (S to 0; W, CN, TI to 1; K to 2).
2. **Two more ordered attributes**, from GP1's stored columns, monotone (more is better, bin 0 worst): `qov` (question content tokens in the tail's title: 0, 1, 2 or more) and `dir` (the link's direction in the any-link graph: none, one-way, mutual).
3. **Spaces.** Phase 3 (`lab`, ordered pool dr, sr, ur, pr, vo, tw, qov, dir): no attribute, the 8 singletons, the 28 pairs: 37 spaces, no triples (a triple in 15 groups is 5,145 keys). Phase 4 (`grp`, the new attributes only): qov, dir, qov+dir, and each of them with dr, pr and tw: 9 spaces. The rank-sum solver T is enumerated only where at most 8 non-S groups exist; the 15-group spaces have 12, so there T is not run and the chain checked is τ_L ≤ τ_U ≤ τ_M.
4. **The final rule of GP2** is the smallest τ_M over all spaces of section 4 and this amendment (ties: fewer kept keys, fewer attributes, declaration order). If it is a space of section 4, it is section 11's final and nothing changes.
5. **Beside runs** on the final (as in section 7) and the **class split**, data for the user and not a rule or a call of this page: the same program (same family, S protected, ρ 0.95, monotone) restricted to the seven passage cells (MultiHop-RAG, MuSiQue, SQuAD, HotpotQA RK and SRK, 2Wiki RK and SRK) and to the two KB cells (MetaQA, WebQSP), on section 11's final space and on the best space of each of phases 3 and 4.
6. **Rest of dev, once**, as in section 7: the final, section 11's final, phase 1's best and S0 (the groups-only rule).

**What I expect (written before any number of this amendment).**
- **X9**: the labels lower the final τ_M by at least 0.02 below section 11's 0.9953: 30 percent. τ_M at most 0.90: 10 percent. At most 0.50: 2 percent.
- **X10**: `qov` or `dir` is in the best space of phase 3: 40 percent.
- **X11**: the passage-only program's τ_M is at most 0.50 on the quarter: 65 percent (the largest own optimum among the seven cells is 0.296; the one-rule cost is the rest). The KB-only program's τ_M is at most 0.90: 25 percent (WebQSP's own optimum was 0.987).
- **X12**: NOT_UNIVERSAL on the rest of dev: 97 percent.

**What each outcome means.** If X9 holds without reaching 0.50, the labels help but not enough; I give the frontier and the user decides. If the passage-only τ_M is at or below 0.50 and the KB-only is not, the data point is that the two classes need different rules and the universal ruling is what keeps them one; I name that to the user as a choice and build nothing. If nothing moves, the link set stays base, PL8 stays undeclared, and the stuck cells are KB cells whose needed hop-2 links nothing stored marks.

## 13. Quarter result, amendment A2, filed 26 Nov (rest of dev still unread)

Gate G-L1 (labels map onto the stored groups on every row) passes on all nine cells. Records: `outputs/gp2/rule_q_phase3.json`, `rule_q_phase4.json`, `rule_q_a2.json`. Every space solved exactly (M: OPTIMAL, relative gap at most 1e-6 as HiGHS reports it, for the final 6.2e-7; chain τ_L ≤ τ_U ≤ τ_M; T not run on the 15-group spaces, as declared).

**Phase 3 (label dimension, 37 spaces), τ_M.** Labels and hops alone (no ordered attribute): 0.9907 (groups only was 0.9996). Singletons: pr 0.9869; dr, sr, ur, vo, tw, qov, dir all 0.9907 (they add nothing to the labels alone). Pairs: **pr+vo 0.9683** (138 kept keys, the global best), ur+vo 0.9804, dr+vo and sr+vo 0.9826, dr+pr, sr+pr, ur+pr, pr+tw, pr+qov, pr+dir 0.9869, every other pair 0.9907. **Phase 4 (groups, the new attributes):** qov 0.9959, dir 0.9957, qov+dir 0.9956, qov+dr and dir+dr 0.9953, qov+pr 0.9954, dir+pr 0.9955, qov+tw 0.9955, dir+tw 0.9957: no gain over section 11.

**Global best (the final rule of GP2): labels × hops × pr × vo, τ_M 0.9683** (section 11's final was 0.9953; GP1's 0.99997). Kept keys by label: S 63, W 35, CN 21, K 19. On the quarter, reach / price per cell: MultiHop-RAG 0.990 / 0.842; MuSiQue 0.957 / 0.941; SQuAD 0.952 / 0.928; MetaQA 0.950 / 0.691; WebQSP 0.977 / 0.968; HotpotQA RK 0.950 / 0.148 and SRK 0.975 / 0.217; 2Wiki RK 0.961 / 0.311 and SRK 0.978 / 0.340.

**Beside the call, on the final.** Own optima: MultiHop-RAG 0.573, MuSiQue 0.324, SQuAD 0.342, MetaQA 0.688, **WebQSP 0.968**, HotpotQA RK 0.145 and SRK 0.089, 2Wiki RK 0.003 and SRK 0.033; cost of universality (τ_M minus the largest own optimum) 0.0006: **WebQSP alone sets the price**. S free: 0.9537 (protecting S costs 0.0146 here). ρ 0.90: 0.875; ρ 0.99: 0.993; LP relaxation at ρ 0.94 and 0.96: 0.712 and 0.791. Size caps: infeasible (the protected S keys alone exceed the caps). Frontier: cap 0.30 infeasible, 0.40: 0.734, **0.50: 0.824**, 0.60: 0.830, 0.70: 0.864, 0.80: 0.871 (section 11's final: 0.75 at 0.50). Tied steps: 0.9872, so per-hop caps buy 0.019. Duals: MetaQA 3.78, MuSiQue 0.36, SQuAD 0.08, the rest 0.

**The class split (data for the user, not a rule of this page): the same program on the seven passage cells and on the two KB cells, τ_M on the quarter.**

| space | passage cells (7) | KB cells (2) |
|---|---|---|
| section 11's final (groups, tw+dr+pr) | 0.9545 | 0.9869 |
| labels, pr+vo | 0.9359 | 0.9683 |
| groups, dir+dr | 0.9528 | 0.9907 |

**Section 11's last sentence is withdrawn.** It said a class rule "would sit far below 0.50 for the passage cells"; that was a statement about the own optima, which I read as if it were the passage-only program. The passage-only program is 0.94 to 0.95, not far below 0.50, so a class rule is not the way out either. Why one x over the seven passage cells costs this much when each alone is at most 0.57 is not measured here: the passage-only program's per-cell prices show MultiHop-RAG 0.911, MuSiQue 0.936 and SQuAD 0.916 held together while HotpotQA and 2Wiki sit at 0.15 to 0.34, so those three conflict on shared keys (that they conflict is the data; which keys, and whether a different key space would separate them, is not). Under the labels rule, HotpotQA and 2Wiki are already at prices 0.15 to 0.34, MetaQA 0.69, MultiHop-RAG 0.84, SQuAD 0.93, MuSiQue 0.94, WebQSP 0.97: the binding passage cell is MuSiQue (price 0.936 even without the KB cells), the binding cell overall is WebQSP.

**Expectations, scored.** X9: the labels lower τ_M by at least 0.02 below section 11's (0.9953 to 0.9683, 0.027): **true** (30 percent); τ_M at most 0.90: **false**. X10 (`qov` or `dir` in phase 3's best space): **false** (40 percent): both add nothing to the labels, and in the group dimension nothing either. X11 (passage-only at most 0.50; KB-only at most 0.90): **both failed** (0.936 to 0.955; 0.968 to 0.991). X12 and X7, X8 wait for the rest of dev.

**Reading.** The labelled keys are the first thing in two pages that moves the price (0.9953 to 0.9683) and the only reason is the rule labels: `lab` alone is 0.9907, and R.mention / R.name_comention / R.title_comention / K carry different needed-to-control ratios per hop. The family is now complete as declared (A, B1, C, A2; B2 costed only, not run). No rule in it reaches the 0.50 price line on all nine cells, and the program names the cell that stops it, WebQSP at own optimum 0.968, then MetaQA at 0.688.

## 14. Amendment A3: the text-window ladder, the co-mention distance, the score windows, per-question caps (C2), the node bound and the grade set (26 Nov, declared after the quarter numbers of sections 11 and 13 were read and before any number of this amendment; the rest of dev is still unread)

**Why.** The user's "try everything" was relayed again with the three readings spelled out. Checked against section 2, three things the user asked for were narrower here than asked: **B1** had one window width and no co-mention distance; **A** had the ranks but not the question-tail score; **C** was a threshold per (group, hop), not a cap on how many links a question carries to the next hop. B2's cost was promised "in the report" and is not written, the node bound (2,500) was never put beside the prices, and the rest-of-dev grade did not hold the winner of every family. This amendment adds exactly these and nothing else. It is a **third** family chosen on the quarter after the quarter's numbers were read, so the search on the quarter is wider than section 4 declared; the rest of dev is read once, for the rules named in item 7, and it cannot leak the choice.

**1. New ordered attributes** (bin 0 worst; the highest index is kept first; every direction fixed here).

| id | name | definition | bins (worst to best) |
|---|---|---|---|
| B-2 | `tw40`, `tw160`, `tw320` | `tw` with the window 40, 160, 320 characters either side of the first mention (`tw` itself is 80; the ladder in tokens is about 7, 13, 27, 53 either side); no IDF | no mention, 0, 1, 2+ |
| B-3 | `cd` | co-mention distance: characters from the first mention's span of the tail's title to the nearest occurrence, in the head's text, of a question content token that is not a token of the title (`\w+` tokens, lower case, `STOP_WORDS` out, as `content_tokens`) | no mention; no question token in the head's text; more than 320; 161 to 320; 81 to 160; 41 to 80; 40 or less |
| A-6 | `qn` | the tail's position in the question's **dense** list as a share of the list's own score range: z = (s - s_last) / (s_first - s_last) with s the stored score of the tail, s_first and s_last the list's first and last stored scores (z = 1 when they are equal); scale-free, so the package encoder and e3v are comparable | not in the list, z < 0.1, 0.1 to 0.2, 0.2 to 0.4, 0.4 to 0.6, 0.6 to 0.8, 0.8 or more |
| A-7 | `qs` | the same over the **SPLADE** list | same 7 |

`qn` and `qs` are the "question-tail cosine" of reading A in a form that means the same thing on every dataset (section 5's `qg` was left out for the opposite reason: raw cut points do not carry across encoders).

**2. Key-space spaces, phase 5** (dimension `lab`, section 12's winning dimension; every space gets L, U, M; T is not enumerable at 12 non-S groups): each new attribute x in {tw40, tw160, tw320, cd, qn, qs} alone, with `pr`, with `vo`, and with `pr` and `vo` (24 spaces); and the **forward triples on section 13's best space** (pr+vo) with each remaining attribute of the pool (dr, sr, ur, tw, qov, dir; 6 spaces): **30 spaces**. A space of 15 groups with a triple has at most 15 x 7 x 7 x 3 = 2,205 keys.

**3. Family C2: per-question caps on the question-ranked fan-out (variables of the program).** Section 4's C was a threshold per (group, hop). C2 caps the **number of links** a question carries at each hop:

- Candidates of a question q at hop h: its non-S link rows at step h (S is protected as before: S links are kept whole and not counted against the cap). A link score orders them; the cap keeps the top W_h.
- **Scores** (lexicographic, best first; every key of a row is a stored column; the last key of each is GP1's `dot`, used only as a tie-break, never as a bin): **K1** (`ur`, `dot`); **K2** (`rank` ascending, `dot`), the per-source rank taken across sources, so rank 1 of every source comes before rank 2 of any; **K3** (`ur`, `rank` ascending, `votes`, `dot`); **K4** (`dot`), the question-independent baseline; **K5** (`tw`, `ur`, `rank` ascending, `dot`); **K6** (`cd`, `ur`, `rank` ascending, `dot`).
- **Variables**: W_1 >= W_2 >= W_3 (non-increasing in hop) from the grid {10, 30, 100, 300, 1,000, 3,000, 10,000, 30,000, 100,000, 300,000, no cap}: 286 triples. The same triple in all nine cells (C5, universal).
- **Program** (per score, exact by enumeration of the 286 triples): minimise the worst cell's price subject to every cell's reach >= 0.95; price and reach as in section 4 (needed rows count 1 in the price and their weight in the reach; a sampled control counts its weight). Reach and price are sums over the three hops, so the 286 triples are scored from 11 values per (cell, hop).
- **Ranks are estimated from the rows**: the number of candidates of (q, h) ahead of a row is its needed rows ahead (1 each) plus the weights of the controls ahead (a control stands for exact/sampled candidates). Ties (identical full key) keep the fraction (W - ahead) / (tie total), clipped to [0, 1]: the expectation over a random order of the tie. The estimate has sampling noise; the rest of dev applies the same rule to new questions with new draws, so the noise is part of the graded number.
- **How this differs from PL5 to PL7's pool caps.** Those picked **nodes** for a pool by a pool rule and restricted the pool toward 2,500 nodes under one rule; C2 ranks **links** by one of six declared link scores and caps links per (question, hop). Links are an upper bound on nodes (a node reached by two links counts twice), so a cap of W links carries at most W nodes.

**4. The final rule of GP2** is now the smallest worst-cell price over every key-space of sections 4, 12 and 14 (tau_M) and every C2 score (tau_C2); ties: fewer kept keys (a C2 score counts as more than any key space), fewer attributes, declaration order (section 12's best, then phase 5's best, then C2). It gets the beside runs that exist for its form (a key-space: section 7's; a C2 score: per-cell reach, price and links per question, the required uniform cap of each cell alone).

**5. B2 (embedding windows): costed here, decided by a rule.** Declared on the CPU only if the CPU estimate for the quarter's windows is **at most 3 hours**; otherwise the exact GPU row is written on the page, queued for the GPU and **not run** (the GPU is not ours). Facts measured before this amendment, on the host GPU (`outputs/enc/e3b/win/*.json`, E3B): e3v is the fusion of gte-modernbert-base (149M) and granite-embedding-small (47M), 1,152 wide; encoding SQuAD's 86,113 windows of 64 tokens (5,779,892 tokens) took 48.6 s for gtemodern and 48.7 s for granitesm, 1.13 ms per window for the pair. The laptop has no copy of either model (downloading about 0.8 GB needs the user's word), so **no CPU measurement is made**; the CPU number on the page is an estimate from parameters (196M x 2 x 64 = 25 GFLOP per window) and is labelled so. The windows to encode come from `outputs/mp_unified/zgp2b2.py` (counts only).

**6. The node bound, beside every result.** The user's target is 95 percent or more at about 2,500 nodes. For every graded rule and every cell: candidate links per question (the weighted estimate), the **kept links per question** (price x candidates per question, an upper bound on kept nodes), the price that 2,500 links per question would be (2,500 x questions / candidates), and the S links per question alone (S is protected, so S above 2,500 per question makes the target incompatible with A.1 on that cell). WebQSP first (oracle depth 9,301: its lever is ordering).

**7. Rest of dev, once.** The rules graded together, one read of each cell: the **final** (item 4); one **winner per family**: A (phase 1's best), B1 (phase 2's best, tw with one partner), C (sigma 0: groups x hop, the per-hop threshold alone), C2 (its best score and triple), phase 5's best space (the ladder, `cd`, the scores and the triples); section 12's best (labels, pr+vo) and section 11's final (tw+dr+pr). UNIVERSAL_RULE is called only for the final; the rest are reported beside it. The R grade of `zgp1p.py` (GP1) continues as declared there.

**8. Gates (new).** **G-W2** the ladder and `cd` against second implementations (1,000 mention rows per cell: a regular-expression word search for the windows, a per-token boundary search for the distance); **G-Q2** `qn` and `qs` against a naive lookup (2,000 random rows per cell, all if fewer); **G-C5** the C2 function against exhaustive enumeration of the tie orders on toy programs (self-test), and on every cell the cap "no cap" gives reach 1 and price 1 while a cap of 0 keeps exactly S; **G-C6** every rule of item 7, evaluated link by link on the quarter, gives its program's own number (the worst-cell price equals tau within 1e-4 for a key space and 1e-6 for C2; every reach is at least 0.95 less the same tolerance), or the run stops; G-Q3 (columns aligned with the chunk rows) and G-C1 as before.

**9. Code and records.** New file `outputs/mp_unified/zgp2a3.py` (commands `text2`, `score`, `solve5`, `cap`, `a3`, `grade`, `report`, `selftest`); it imports `zgp2.py`, `zgp1.py` and `zgp1p.py` unchanged (their pinned shas stand in every record). `text2` and `score` add six arrays (`tw40`, `tw160`, `tw320`, `cd`, `qn`, `qs`) to the link-level `outputs/gp2/<d>/cols_<part>_<v>.npz` files (never committed); the arrays `dr`, `sr`, `tw` are untouched, so every earlier record's number is unchanged, but the files' sha256 in earlier records no longer matches the files (they gained arrays). Records: `rule_q_phase5.json`, `rule_q_c2.json`, `rule_q_a3.json`, `b2_cost.json` (quarter); `grade_r_a3.json` (rest of dev). C2's reach and price use the expectation over a random order inside each tie; the path retention of the grade applies one fixed random thinning (seed 20261126) of those fractions. The toy self-test passed before this section was written; no number of this amendment existed then.

**10. What I expect (written before any number of this amendment).**
- **X13**: phase 5's best space lowers the quarter's tau_M by at least 0.01 below section 13's 0.9683: 25 percent; at most 0.90: 5 percent; at most 0.50: 1 percent.
- **X14**: `cd` or one of the three widths is in phase 5's best space: 40 percent; `qn` or `qs`: 15 percent.
- **X15**: C2's best score reaches a worst-cell price at most 0.90 with every cell's reach >= 0.95: 10 percent; at most 0.50: 2 percent; below section 13's 0.9683: 30 percent.
- **X16**: C2's chosen W_2 is at least 10,000 links per question (nowhere near 2,500): 80 percent.
- **X17 (the node bound)**: the final rule's kept links per question exceed 2,500 in at least eight of the nine cells: 90 percent; in WebQSP: 99 percent (WebQSP's candidates are about 391,000 links per question on the quarter, so 2,500 links is a price of 0.0064).
- **X18**: NOT_UNIVERSAL for the final on the rest of dev: 98 percent.

**What each outcome means.** If nothing in this amendment lowers the price below section 13's, the answer of this page stands: no universal rule in any of the three readings of windows, the link set stays base, PL8 stays undeclared, and the node bound is a statement about walk-time ordering and depth, not a link filter. If C2 gets a cell to its reach inside 2,500 links, that cell's cap goes beside the price as the first node-count evidence of this page.
