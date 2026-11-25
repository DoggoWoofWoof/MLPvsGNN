# UL3: tightening the link set against the gold, part A (a one-hop screen on the e3v layout)

Declared 25 November 2026, before any UL3 number exists. **Nothing here trains, no model reads anything, no LLM or spaCy is used, no filed graph, pool,
fit or record changes, and no link is removed from any store.** Part A is a pool-free SCREEN on gold pairs: it says whether some subset of the B0
layout's pairs keeps almost all of the one-hop gold coverage at a fraction of the candidates. It decides nothing about the layout; parts B and C (section 3)
are declared on their own pages, and only they can change a link set.

## 0. The user's words, and the rulings that bind

- 25 Nov, ~23:00: "we currently do not have proper gpu access do we to train and get our models, so till then we need to properly finish the pools, encoding
  finish for all datasets, improving the R links, so that we can target the coverage budget and precision and eliminate the links we might not need, and
  keep the links we do actually need based on the gold links and compare how we can get ... once that is done we will have corpus and the pools we can target
  the vector improvement and models after that with SEM2 etc ... once we find the correct links only then we have to do it".
- 25 Nov, ~23:25: "i think we would also speed up instead of using the full pools we should use smaller subsets and once we know our links have improved or they
  work and target the failures etc then we should do on the full datasets". Section 8 is the answer to this.
- Earlier and still binding: "we need to have universal rules" (one rule on all seven, carried everywhere or nowhere); S is the structural links
  (hyperlinks, KB triplets), R the rule-based links, K the kNN, never conflated, every edge labelled by the rule that made it, and "we will keep all weights";
  the hub line is CLOSED (UL2 8.15: no degree, IDF or hub-key attribute appears anywhere below); no LLM, no spaCy, no learning; declare before any number.
- The PL agent's two conditions, accepted: (1) tightening selects by LABEL or CLASS (never by single pairs) against gold on the DEV carves only and is graded on
  s1eval; (2) one pool re-run (PL8) on the final link set and nothing extra from the pool side before it.

## 1. What I had read before writing this page

UL2P's filed ledger (docs/UL2P_LABEL_PRECISION_E3V.md) and UL2H's, read as tables on the six slot datasets: the precision and reach of each B0 label on
s1eval at one hop (every label is between 0.01% and about 7% precise; B0 is 69 to 99% `R.mention` on MuSiQue, MetaQA and WebQSP); the exact label-profile sets of
MuSiQue, WebQSP and MetaQA (a pair that several label families agree on is 13 to 30 times more precise than a pair of one family: MuSiQue mention-only 0.25%,
adding `name_comention` 3.2%, adding `knn` 7.3%); and from UL2H that MetaQA's `S` has 0.005% one-hop precision yet is the only unit with ALL-gold coverage 1.0 at three
hops, and that MuSiQue's `R.mention` alone reaches 28% of the golds the other units miss at two hops. **I have read no UL3 quantity: the cells below, their dot
quintiles, the keep sets and the reach and price retained do not exist yet.** The self-test (section 9) ran on the laptop before this page was written;
it asserts structure on small random graphs and on MultiHop-RAG's real shapes, checks gates G3-2 and G3-3 there (UL2P's filed MultiHop-RAG ledger: 265,876
candidates, 2,350 covered slots; dots within 1.1e-7 of a float64 dot), and prints no table of the screen.

## 2. Prior art (searched 25 Nov; self-reported numbers not re-verified)

No work I found reports the precision of its graph edges against gold evidence links, or prunes a link set to a gold-coverage budget. The closest: NeuroGraphDB
(gold-pair linkage of its graph), Democratizing GraphRAG (hub pruning, reported as the share of gold titles kept), TRIAGE (a gold path coverage measure), and the
graph retrievers G3RAG, KGP, GRIEVER, LinearRAG and PathRAG, which build links by rule or by an LLM and judge them only through end retrieval. So the
instrument here is ours (UL2P/UL2S/UL2H), and none of those works is a baseline for it.

## 3. The three parts, and why part A cannot decide alone

- **A (this page):** the one-hop screen on gold pairs. Cheap (one CPU, no pool), run on a small subset first (section 8).
- **B (declared after A's read):** the path-level guard: the UL2H walker (uncapped, hops 1 to 3 from the seeds) on the kept link set against the full one, with the
  graph floors G95 and G98 of UL2H. It is the guard against dropping a link that no one hop pays for but that paths need.
- **C (the PL agent's PL8):** the capped, ranked pool grid on the link set that survived B. The pools consume links through caps and ranking, so only this grid says
  what a tightening is worth to a pool.

A one-hop screen cannot grade a link family alone, and UL2H shows it: MetaQA's `S` has 0.005% one-hop precision and is nevertheless the only unit that reaches
ALL-gold coverage 1.0 at three hops. A SCREEN_IN here is therefore a necessary condition for part B and never a decision; a SCREEN_OUT closes the variant.

## 4. Population, anchors, candidates (UL2P's, by reference)

The six slot datasets and versions of UL2P's ledger: MultiHop-RAG RK; MuSiQue RK; MetaQA SRK; WebQSP SRK; HotpotQA RK and SRK; 2Wiki RK and SRK. Kinds: GP (gold
pairs: every pair of the question's golds) on all, and CH (MuSiQue's chain pairs) beside. SQuAD has one gold per question, no slot, and is n/a. An **anchor** is a
slot's endpoint, a **candidate** a B0 pair at that node, a **target** the slot's other endpoint, a slot **covered** when its pair is in the unit (the
columns of UL2P's `unit_cols`, loaded by its own functions after its gates G-P1 to G-P3). The any-link graph is B0 = S (SRK) | W2_S | F0T | K_e3v. B0's layout
is the filed one (UL2 section 12, a70ec8cc: NONE_CARRIED, layout = base); nothing below changes it.

**Carves.** Selection uses the DEV carve only: s1sel (MuSiQue, MetaQA, HotpotQA, 2Wiki, MultiHop-RAG's dev split); WebQSP has no s1sel (it is read zero-shot), so its
dev carve is the PL agent's: the non-holdout half of train (`outputs/pl2/webqsp/pop_dev.npz`). **Grading is on s1eval** (MultiHop-RAG: its test split). s1eval
selects nothing and is never read by the subset stage.

## 5. Cells: the attribute, the three variants

Every any-link pair at the anchors falls in exactly one cell of a partition. Two attributes, both fixed by a rule (no learning):

- the pair's **label profile**: the exact set of label families that made it (the pair's bits over S, R.mention, R.name_comention, R.title_comention, K.knn, and for
  SRK the dataset's top S relations as UL2P's section 5 names them);
- the **dot** of the pair's two stored e3v document rows (float32, the score the filed H/NT forms use).

| variant | cell |
|---|---|
| `P` | the label profile |
| `C` | the S > R > K group of the pair x the quintile of its dot inside the group |
| `PC` | the label profile x the quintile of its dot inside the profile |

Quintile cut points: five equal-count parts of the dots of the class's pairs at the DEV anchors; a class with fewer than five such pairs has one bin. Per cell
the columns (candidates, targets, covered slots) are counted per question on each carve.

## 6. The keep rule and the call (fixed)

On the dev carve, cells are ordered by their **precision** (covered slots per candidate; ties: fewer candidates, then the lower cell id) and taken, highest first,
until the kept cells' covered slots reach RHO of all cells' covered slots; a cell with no candidate is never kept. RHO = 0.95 is the call (the PL target); 0.99,
0.90 and 0.80 are the frontier beside, never the call. On s1eval the kept cells (the same cell ids) give:

- **reach retained** = covered slots of the kept cells / covered slots of the any-link graph;
- **price retained** = candidates of the kept cells / candidates of the any-link graph;
- precision of the kept set and of the any-link graph, and their paired difference;

all with the 95% bootstrap interval over questions (zap4's). **Call, per judged cell (dataset, version, kind) and variant, at RHO = 0.95:** SCREEN_IN iff reach
retained >= 0.92 and price retained <= 0.50 (point values); else SCREEN_OUT; NOT_READ when undefined. Both thresholds are chosen now, before any number: 0.92 is the 0.95 budget
less the width an s1eval/dev shift may cost, 0.50 is the smallest saving worth carrying to the pool grid. **Universal read: a variant PASSES_SCREEN iff all six
datasets are read and every judged cell is SCREEN_IN; otherwise FAILS_SCREEN.** The per-cell calls are reported either way; they decide nothing alone, and a rule that
passes on some datasets only is not a universal rule (the user's ruling) and is carried nowhere.

The judged cells are the ones the filed ledger holds on s1eval: MultiHop-RAG RK GP; MuSiQue RK GP and CH; MetaQA SRK GP; WebQSP SRK GP; HotpotQA RK GP and SRK GP; 2Wiki
RK GP and SRK GP (nine).

## 7. What I expect (stated before the number)

**(The KB-dataset sentence below is withdrawn by section 12; the rest stands.)** Most likely FAILS_SCREEN for every variant: on the KB datasets the structural links carry nearly all the one-hop reach at a price near the whole graph, so a keep set
at 95% reach cannot be much cheaper than B0 there. On MuSiQue and the passage datasets, where B0 is mostly one low-precision label, agreement between families (P) and the
dot (C, PC) may well keep 92% of reach at well under half the candidates. If that holds, A reads as a split between structured and passage datasets and the question
for part B becomes whether one rule (this one, with the whole graph kept when nothing can be cut) can be carried everywhere, which the universal ruling decides on its
own page. If no variant passes anywhere, the layout stays and the link line ends at UL2's base with the screen as its record.

## 8. The iteration subset (the user's ~23:25 ruling)

Before the full run of any rule, a **sub run**: a quarter of the dev carve (the dev questions whose index mod 4 is 0) selects; the other dev questions
(index mod 4 is not 0) grade; the anchors are the dev carve's only, so the dots and cells are computed over a fraction of the pairs; **s1eval is never read**.
Its outputs (`outputs/ul3/<d>/sub.json`, `outputs/ul3/sub_report.md`) are diagnostic and carry NO call: they show what each variant keeps, its reach and price on the
rest of dev, and the classes it drops that hold the most covered slots, so the next rule targets the failures. (At SRK the top-relation vocabulary is chosen over the
pairs at the anchors read, so a sub run's S relation classes may differ from the full run's; it is a diagnostic for that reason too.) The full run (`cells`: dev selects, s1eval
grades) is read **once per declared rule**. A rule changed after a sub run is a new declared rule on a new page or section, committed before its next run. The same
staging is the plan for parts B and C: a subset of dev questions first (pools, walks), the full carves once the rule has survived it.

## 9. Gates (each failure stops that dataset's reading; none of its numbers is read)

- **UL2P's own G-P1 to G-P3** on every load (B0's size, each form's pairs beyond B0, each cell's slots and questions equal the grade's).
- **G3-1** per variant and carve, the cells partition the any-link pairs: per-question candidates and covered slots summed over the cells equal the any-link unit's (array equality).
- **G3-2** the any-link unit's candidates, targets and covered slots on each whole carve equal UL2P's filed ledger (s1eval and s1sel; WebQSP's dev carve is not a UL2P carve and is exempt).
- **G3-3** the blocked float32 dots equal a float64 dot of the same two rows on a fixed 512-pair sample within 2e-5.
- **G3-4** WebQSP's dev carve shares no question with s1eval.
- **Self-test** (`zul3a.py selftest`, laptop): the dots, classes, bins, partition and keep rule equal brute force on small random graphs (a deliberately wrong keep order is caught);
  MultiHop-RAG's real shapes pass G3-2 and G3-3; the subset is a partition of the dev slots; the report code runs on the real record's shape.

## 10. Code, rows, outputs

`outputs/mp_unified/zul3a.py` (new; imports zul2p, zul2c, zul2u, zul2, zul1, zul2m and zap4 and edits none of them). Host rows: one `sub` row and one `cells` row per
dataset (twelve), one CPU each, placed in the feed before the SEM2 rows; the reports run on the laptop after `rx fetch`. Outputs only under `outputs/ul3/`; arrays
are never written.

## 11. What follows

A read of the sub rows, then of the full rows, filed as a results section of this page. If a variant PASSES_SCREEN, part B (path-level guard, subset first) is declared;
if none does, I name which cells and classes failed and whether one different rule is worth a page, on the dev subset first. The PL agent is told when a link set is final; its
PL8 re-run starts only then, and the int8 world, the vectors and the models follow on that pool.

## 12. Amendment A.1, 25 November, before any host number (MultiHop-RAG had been read)

- **The user, ~23:40:** "the issue with that is R has much much more links than S, we cant afford to have multiple useless links draining our walks and need to
  have much smarter walks too which can be done two ways eliminate useless links and use the question to guide, so we are at first step here and if the vectors look
  good we can have a semantic walk as well".
- **What the filed UL2P ledgers say** (their B0 headers, read before this amendment; B0 pairs at the anchors, S > R > K groups): WebQSP S 1.2% / R 97.7% / K 1.1%;
  MuSiQue R 94.7% / K 5.3%; HotpotQA SRK S 22.6% / R 63.6% / K 13.8%; 2Wiki SRK S 22.3% / R 69.9% / K 7.9%; MetaQA S 36% / R 38% / K 26%; MultiHop-RAG R 99.0% / K 1.0%.
  R is the bulk everywhere, and S is a small part of B0 on every dataset that has it.
- **A correction:** section 7 expected the KB datasets to price near the whole graph because S carries their reach. That was wrong: S is a small share of the candidates,
  so keeping all of S is cheap where S carries the reach, and the screen can still find a large saving in R and K there. The sentence is withdrawn.
- **The rule change (A.1):** the S cells (the pairs in the structural store: the profile's S bit; the S group of variant C) are **never removed**. Selection runs over the
  other cells in precision order as in section 6; the S cells' covered slots count toward the RHO budget and their candidates toward the price. **The call is A.1's**;
  the section 6 rule with S not protected is computed beside and reported, never the call. RHO, the thresholds (0.92, 0.50), the variants, the carves, the judged cells and the
  pass rule are unchanged. In an RK version (MultiHop-RAG, MuSiQue, HotpotQA RK, 2Wiki RK) there is no S, so A.1 and the first rule coincide.
- **What I had read:** both MultiHop-RAG stages on the laptop (23:30), before this amendment: at RHO 0.95, C and PC SCREEN_IN, P SCREEN_OUT. The amendment follows from the user's
  words and the filed shares; it cannot change those readings (no S there), and the MultiHop-RAG records are re-run with the amended module so that every record carries its sha.
  No host row had started; none of the five host datasets has been read.
- **The two levers, and where each lives:** (1) drop useless links before the walk: this page (R and K are what it can cut; S stays), then parts B and C; (2) let the question guide
  the walk: expansion filtered or ranked by the question vector (the pool builder already ranks expansion by the first-stage inner product); a walk-time rule, declared on the pool
  side after the link set is final, subset first; (3) a semantic walk (SEM2/R4's operators and regions): only if the vectors look good. Nothing for (2) or (3) is run here.
- A pass here still decides nothing about the layout: part B (paths) and part C (pools) must keep the gold coverage on the tightened set.

## 13. Disclosed

- I chose the cell partition, the 0.92 and 0.50 thresholds, RHO = 0.95 and the subset fraction (a quarter) before any UL3 number, and from reading UL2P/UL2H's
  tables (section 1) rather than from the grid itself.
- The MultiHop-RAG self-test read UL2P's filed any-link counts and 512 dots as gates; it printed no screen number.
- A screen that passes proves nothing about pools; a screen that fails on one dataset says nothing about the pool grid there.
