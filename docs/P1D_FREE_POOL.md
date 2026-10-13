# P1d: one parameter-free pool rule for all six datasets

Declared 13 October 2026, before any number of the stage. P1's fourth stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), under the user's rule of about 10:40 (section 1). It reads the pools
only: no model trains or reads here, and nothing is fit.

## 1. The user's rule, and what it changes

The user, 13 October, about 10:40: "no no until the pools are finalized there should be no learning the offset etc was
for reranking the links and pool finalization are dataset agnostic and parameter free and should have the most golds
in least nodes then we go with the node space EM etc and learned offsets what we discussed".

So the pool is a fixed rule, the same on all six datasets:
- **No weight is fit to golds.** P1b's mixture (MX), P1b2's rank gate (RC) and P2b's family gate (MXU) are fit by EM
  to the golds, so they are learning. They are reported beside as learned upper lines, never as the pool.
- **Nothing is set per dataset.** M3A's regimes pick a graph per dataset (the structural family alone on metaqa,
  hotpotqa and 2wiki; all three families on musique and webqsp; no walk on squad). The rules here never read them.
- **The constants are fixed here, before any number,** and all come from earlier stages:
  - step 4e's walk (alpha 0.5, push threshold 1e-7, top 7,000 nodes);
  - the first stage's RRF constant (60);
  - P1b's head of the first-stage lists (the dense and SPLADE top 100).
- **The rule is judged by golds per node:** coverage at every pool size N, against the best LLM-free SOTA pool order
  (P1a's bar).

Node-space EM and the learned offsets come after the pools, as rerankers, with the models retrained on the final
pools (P1c is held until then, f8da4b0).

## 2. What the filed records ask of a fixed rule

HippoRAG 2's restart (P1a's A1) walked on the structural family alone is one rule for all six. P2a part 1 filed it on
every dataset (s1eval, ALL at each question's size of today's pools, |I_q|):

| dataset | A1 on the structural family | P1a's bar | the order that leads at \|I_q\| |
| --- | ---: | ---: | --- |
| metaqa | 0.812 | 0.961 | LINKED, the walk from our linked entities |
| squad | 0.965 | 0.981 | the first-stage lists, unwalked |
| musique | 0.832 | 0.883 | A2 on all three families |
| hotpotqa | 0.973 | 0.973 | A1 on the structural family |
| 2wiki | 0.911 | 0.911 | A1 on the structural family |
| webqsp | 0.851 | 0.790 | A1 on all three families |

It holds the bar on hotpotqa, 2wiki and webqsp. Three needs remain:
1. **metaqa needs the walk from the linked entities.** Our exact-match linker is precise there and loose on the
   passage corpora. Measured by the share of a question's linked nodes that the first-stage lists' top 100 also hold
   (P1b's linked_in_lists, on the fit carves):

   | | metaqa | webqsp | musique | hotpotqa | 2wiki |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | linked nodes per question (median) | 2 | 3 | 3 | 24 | 16 |
   | share held by the lists' top 100 (mean) | 0.98 | 0.58 | 0.51 | 0.155 | 0.14 |

   webqsp's numbers are a label-free count on its s1eval questions (no gold read), since webqsp has no fit carve.
   squad has no linked node.
2. **squad needs the lists' head.** Every walk is below the unwalked lists at N = 25 and 50 (P2a part 1).
3. **musique needs the ner and knn families.** They cost hotpotqa and 2wiki (P2a part 1).

The rules below meet these needs from the question's own signals (the corroborated links), or fuse every source
with fixed weights. Neither reads a gold or names a dataset.

## 3. The sources

Per question, on every dataset. squad's families are walked as P2b walks them.
- **Two graphs, the same rule everywhere:**
  - st, the structural family alone;
  - all, the three families together.

  Both use M3B's stores and P2a's unions.
- **A1(g):** HippoRAG 2's restart (P1a's A1: today's seeds at half the mass, the RRF-weighted dense and SPLADE
  top-1,000 lists at the other half), walked on g.
- **K(g):** the corroborated anchors, walked on g. L* is the set of linked nodes (step 4e's exact-match linking) that
  the union of the dense and SPLADE top 100 also holds. K(g) restarts uniformly over L*, and is empty when L* is.
- **R0:** the RRF lists, unwalked: P1a's RRF order, with P1b's zero-step masses (the RRF scores scaled to sum to 1).
  P1b's zero step sorts the scaled scores, and scaling can tie two nodes the raw scores do not, so its order can
  differ from P1a's at a tie. The development runs found such ties on musique. R0 keeps P1a's order.

Every walk uses step 4e's kernel and excludes nothing. It keeps its top 7,000 nodes, with their masses scaled to sum
to 1 (P1b's walk with masses).

## 4. The six rules, fixed here

| rule | the order |
| --- | --- |
| A1st | A1(st) |
| A1both | CombSUM of A1(st) and A1(all) |
| ROUTEst | when the question has linked nodes and L* holds every one of them: K(st), then A1(st)'s other nodes in A1(st)'s order; A1(st) otherwise |
| ROUTEboth | the same with CombSUM of K(st) and K(all), then A1both's other nodes; A1both otherwise |
| SUM5 | CombSUM of the five sources A1(st), A1(all), K(st), K(all) and R0 |
| RRF5 | RRF over the same five orders, with the first stage's constant (60) |

- **CombSUM** adds the sources' masses. Each source's masses sum to 1, so this is P1b's mixture with equal weights.
- **RRF** scores a node by the sum over the sources of 1/(60 + its rank), with ranks starting at 1. A source that does
  not hold the node adds nothing.
- An empty source is skipped. Every order sorts by score, breaks ties by node position, and is cut at 7,000.
- **ROUTE's condition** is all or nothing, so it adds no threshold: the links are trusted only when the first stage
  retrieves every one of them.
- **ROUTE's backfill:** a walk from a few anchors can stop short of 7,000 nodes, so A1's order fills the rest. ROUTE
  therefore never holds fewer nodes than its A1 branch.

## 5. The choice and the call, fixed before any number

**The checks come first.** Any failure stops the stage.
- On every carve, A1(st) and A1(all) equal P2a part 1's filed A1 on its subsets S and SNK:
  - at every N of P1a's grid, on both measures;
  - and at both matched sizes, |I_q| and |P4e|.
- R0 equals P1a's RRF order at the same points.

**The choice, on s1sel.** One rule is chosen for all six, among the six above, on the five training datasets' s1sel
carves (webqsp has none, so the choice is zero-shot there):
1. the fewest points (dataset, N, measure) where the rule is more than 0.002 below P1a's bar on s1sel;
2. ties go to the smaller largest gap;
3. then to the higher mean ALL at |I_q|;
4. then to the table's order.

This is the only choice in the stage. Nothing else is selected.

**The call, on s1eval.** The chosen rule is compared with P1a's bar on each dataset, by P1a's rule. At every N of
the grid (25 to 7,000) and on both measures (ALL and gold recall), the bar is the best of the LLM-free SOTA orders A1,
A2, LINKED, DENSE, SPLADE and RRF, on the dataset's regime graph:
- **AT_OR_ABOVE:** the rule is never more than 0.002 below the bar.
- **BELOW:** anything else. The record names every point where the rule is below, the order that leads there, and
  the largest gap.

The stage's call is **ALL_SIX** when all six datasets are AT_OR_ABOVE, and **NOT_ALL_SIX** otherwise.

**Reported beside, deciding nothing:**
- every rule on both carves, against the bar, and each source's own curve;
- **the like-for-like bar:** the same SOTA rules on the two graphs every dataset has: A1 and LINKED on st and on all,
  and the lists. LINKED on both graphs is P2b's filed L_st and L_all. P1a's bar runs them on each dataset's regime,
  a graph picked per dataset;
- at |I_q| and |P4e|: today's pools, the rule, the best SOTA order, and the learned gates MX, RC (P1b2) and MXU (P2b),
  from their filed records;
- **golds per node:** the smallest N of the grid at which the rule's ALL comes within the tie (0.002) of today's
  pools' ALL at |I_q|, beside the same N for the bar;
- the share of questions with any corroborated link, and the share on ROUTE's K branch.

## 6. What follows

- **ALL_SIX:** the chosen rule is the pool order. The steps that follow:
  1. P2c part 2 reads it with the LLM-free family added to the all graph, in its own file;
  2. the models are retrained on its top |I_q| (P1c, re-declared on this pool), so the look cost is unchanged;
  3. P5's node-space EM and the learned offsets follow, as rerankers.
- **NOT_ALL_SIX:** the chosen rule is still the parameter-free pool, under the user's rule.
  - The claim that our pools are at or above the SOTA pools on all six is not made, and the record names where.
  - The next pool change, a source or a rule with nothing fit, is declared in its own file.
  - The fitted gates stay beside and are never the pool.

Flagged, as every pool so far: the walks are propagation over the graph. The pool is built before either model runs,
and both models of a family read the same pool, so the MLP's own scoring stays free of message passing.

## 7. How it runs

- **Code:** `scripts/p1d_free_pool.py`. It imports, unchanged:
  - P1b's walk with masses and its zero-step sources;
  - P2b's two unions over P2a's family stores;
  - P1a's restarts, curves and matched sizes;
  - step 4e's carve runner.
- **The self-test** covers:
  - CombSUM, RRF and ROUTE's backfill on toy sources: a lone source keeps its order, empty sources are skipped, and
    ties go by node position;
  - the corroborated links;
  - ROUTE's branches: every link corroborated, some, none, and no link;
  - the sources on a toy graph equal step 4e's walk, on both unions and on the regime's own union;
  - the choice's order and tie-breaks, and the call (at the tie, and just past it), on made-up records;
  - the verdict refuses a source that differs from its filed record, and a part run with `--limit`;
  - no order is listed twice.
- **Development runs, no number.** `--limit Q` runs the stage on the first Q questions of each carve and writes
  nothing. It logs only that every step ran and that the walks equal step 4e's own walk on the same restarts, order
  for order, on P2b's unions and on the regime's own union, and that R0 equals P1a's RRF order. Before this file was
  committed, the self-test passed, and `--limit 64` runs on musique, metaqa and squad on the laptop ran every step
  with these checks held.
- **Host jobs** (`p1d-pools-<dataset>`, light lane), sized from P2b's apply, which walked the same unions:

  | dataset | CPUs | memory |
  | --- | ---: | ---: |
  | 2wiki | 8 | 6 GB |
  | hotpotqa | 6 | 5 GB |
  | metaqa | 4 | 3 GB |
  | musique, squad, webqsp | 3 | 2, 2 and 3 GB |

  Each writes `outputs/p1d/pools_<dataset>.json` (curves and shares, a few hundred KB) and each carve's gold positions
  under every order, in `pos_<dataset>__<carve>.npz` (a few hundred KB). Then `verdict`, run locally, writes
  `outputs/p1d/verdict.{json,md}`. It refuses parts from different scripts or freezes, and parts run with `--limit`.
- **ETA:** four walks per question against P2b's six. 2wiki is the longest, at about 20 minutes on 8 CPUs. The call
  should land about 30 minutes after the items start.
- The encoder stays frozen. No model trains or reads, nothing is fit, and test splits are never read.

## 8. Results (filed 13 October 2026)

The six host items ran from 11:32 to 11:50. The verdict ran locally at 12:26 and wrote `outputs/p1d/verdict.json` and
`verdict.md`, with each dataset's curves, shares and gold positions beside them in `outputs/p1d/`.

**The checks held on all eleven carves.** A1(st) and A1(all) equal P2a part 1's filed A1 on S and SNK, and R0 equals
P1a's RRF order. They agree at every N of the grid, on both measures, and at both matched sizes.

### The choice (s1sel): RRF5

| rule | points below the bar | largest gap | mean ALL at \|I_q\| | metaqa | squad | musique | hotpotqa | 2wiki |
| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| A1st | 76 | -0.317 | 0.893 | BELOW (30) | BELOW (10) | BELOW (30) | AT_OR_ABOVE | BELOW (6) |
| A1both | 72 | -0.334 | 0.896 | BELOW (30) | BELOW (8) | BELOW (16) | BELOW (10) | BELOW (8) |
| ROUTEst | 53 | -0.149 | 0.905 | BELOW (7) | BELOW (10) | BELOW (30) | AT_OR_ABOVE | BELOW (6) |
| ROUTEboth | 82 | -0.089 | 0.910 | BELOW (27) | BELOW (8) | BELOW (30) | BELOW (9) | BELOW (8) |
| SUM5 | 42 | -0.174 | 0.924 | BELOW (30) | BELOW (2) | AT_OR_ABOVE | BELOW (10) | AT_OR_ABOVE |
| **RRF5** | **38** | -0.112 | **0.938** | BELOW (29) | BELOW (2) | AT_OR_ABOVE | BELOW (7) | AT_OR_ABOVE |

### The call (s1eval): NOT_ALL_SIX

RRF5 is AT_OR_ABOVE on squad, musique and 2wiki, and BELOW on three datasets:
- **metaqa:** 29 points, all against LINKED (the walk from our linked entities). The largest gap is -0.114 (N = 150,
  ALL), and the gaps are -0.09 to -0.11 from N = 25 to 300.
- **hotpotqa:** 4 points against A1 on the structural family, the largest -0.008:
  - N = 50, ALL -0.004;
  - N = 100, ALL -0.008 and gold recall -0.004;
  - N = 750, ALL -0.002.
- **webqsp, the head:** 3 points against A1 on all three families: N = 25, ALL -0.054 and gold recall -0.058; N = 50,
  ALL -0.003. Against the like-for-like bar, where A1 on the structural family leads, it is 8 points and -0.144.

Every rule against the bar on s1eval (points below and the largest gap), deciding nothing:

| rule | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | --- | --- | --- | --- | --- |
| A1st | BELOW (30, -0.316) | BELOW (14, -0.019) | BELOW (30, -0.084) | AT_OR_ABOVE | AT_OR_ABOVE | AT_OR_ABOVE |
| A1both | BELOW (30, -0.335) | BELOW (10, -0.015) | BELOW (21, -0.025) | BELOW (5, -0.011) | BELOW (4, -0.018) | AT_OR_ABOVE |
| ROUTEst | BELOW (4, -0.005) | BELOW (14, -0.019) | BELOW (30, -0.132) | AT_OR_ABOVE | AT_OR_ABOVE | AT_OR_ABOVE |
| ROUTEboth | BELOW (27, -0.068) | BELOW (10, -0.015) | BELOW (30, -0.057) | BELOW (4, -0.010) | BELOW (4, -0.018) | BELOW (1, -0.005) |
| SUM5 | BELOW (30, -0.179) | BELOW (4, -0.003) | BELOW (1, -0.003) | BELOW (6, -0.008) | AT_OR_ABOVE | BELOW (1, -0.003) |
| **RRF5** | BELOW (29, -0.114) | AT_OR_ABOVE | AT_OR_ABOVE | BELOW (4, -0.008) | AT_OR_ABOVE | BELOW (3, -0.058) |

No rule is AT_OR_ABOVE on all six. ROUTEst comes closest on the KB datasets and loses musique and squad.

### At the models' sizes (ALL, s1eval)

At each question's size of today's pools, |I_q|:

| dataset (mean rows) | today's pools | RRF5 | the best SOTA order | MX (learned) | RC (learned) | MXU (learned) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| metaqa (2,018) | 0.895 | 0.910 | LINKED 0.961 | 0.923 | 0.963 | 0.920 |
| squad (50) | 0.980 | 0.979 | A1 0.981 | 0.981 | 0.981 | 0.978 |
| musique (2,092) | 0.803 | **0.899** | A2 0.883 | 0.892 | 0.844 | 0.887 |
| hotpotqa (94) | 0.968 | 0.963 | A1 0.973 | 0.969 | 0.971 | 0.969 |
| 2wiki (106) | 0.906 | **0.961** | A1 0.911 | 0.914 | 0.913 | 0.917 |
| webqsp (2,120) | 0.688 | **0.876** | A1 0.790 | 0.806 | 0.782 | 0.857 |

At step 4e's pool sizes, |P4e|:

| dataset (mean rows) | today's pools | RRF5 | the best SOTA order | MX (learned) | RC (learned) | MXU (learned) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| metaqa (7,082) | 0.953 | 0.971 | LINKED 0.992 | 0.979 | 0.994 | 0.976 |
| squad (150) | 0.992 | 0.991 | A1 0.992 | 0.992 | 0.992 | 0.991 |
| musique (6,899) | 0.937 | 0.944 | A2 0.939 | 0.947 | 0.905 | 0.945 |
| hotpotqa (188) | 0.979 | 0.981 | A1 0.982 | 0.981 | 0.980 | 0.980 |
| 2wiki (221) | 0.945 | 0.970 | A1 0.926 | 0.928 | 0.927 | 0.932 |
| webqsp (7,213) | 0.909 | 0.925 | A1 0.862 | 0.890 | 0.864 | 0.927 |

- **Against today's pools at |I_q|,** RRF5 gains on four datasets: musique +0.096, 2wiki +0.055, webqsp +0.188 and
  metaqa +0.015. It is level on squad (-0.001) and below on hotpotqa (-0.005).
- **Against the learned gates,** RRF5, with nothing fit, is above all three on musique, 2wiki and webqsp at
  |I_q|. It is below them on metaqa (RC 0.963) and hotpotqa (RC 0.971), and on squad by 0.002.

**Golds per node:** the smallest N at which an order reaches today's pools' ALL at |I_q|.

| dataset | today's rows | RRF5 | the bar |
| --- | ---: | ---: | ---: |
| metaqa | 2,018 | 2,000 | 750 |
| squad | 50 | 50 | 50 |
| musique | 2,092 | **500** | 750 |
| hotpotqa | 94 | 150 | 100 |
| 2wiki | 106 | **50** | 100 |
| webqsp | 2,120 | **500** | 1,000 |

So RRF5 holds today's golds in about a quarter of the nodes on musique and webqsp, and in half on 2wiki.

**Shares of questions** (s1eval): with a linked node, with a corroborated one, and on ROUTE's K branch.
- metaqa 0.997, 0.996 and 0.949.
- squad 0 for all three.
- musique 0.943, 0.792 and 0.266.
- hotpotqa 1.000, 0.904 and 0.019.
- 2wiki 1.000, 0.998 and 0.004.
- webqsp 0.997, 0.954 and 0.293.

### Where RRF5 falls short (diagnostics, deciding nothing)

- **metaqa needs the walk from the links to lead.** K(st) alone is the bar's LINKED: 0.961 at |I_q| and 0.546 at
  N = 25, against RRF5's 0.910 and 0.439.
  - RRF5 gives three of its five votes to the sources led by the first stage: A1 on both graphs, and the lists.
  - On metaqa's name-only nodes, the lists hold names that share the question's words, not its answers. R0 holds
    every gold for 0.261 of questions at |I_q|.
- **ROUTE's K branch is right on metaqa and wrong on musique.** On s1sel, ALL at N = 25, split by the branch:

  | questions | share | K(st) | RRF5 | A1st |
  | --- | ---: | ---: | ---: | ---: |
  | metaqa, where the K branch changed the order (1,418) | 0.947 | 0.559 | 0.445 | 0.228 |
  | metaqa, the others (79) | 0.053 | 0.316 | 0.291 | 0.241 |
  | musique, where the K branch changed the order (346) | 0.226 | 0.118 | 0.425 | 0.277 |

  On musique's K-branch questions, K(st) holds every gold for 0.405 at N = 2,000, against RRF5's 0.890.
- **Why K fails on musique.** We looked at route questions of musique's s1sel. The walk from a linked passage keeps
  the anchor first and then floods its neighbours.
  - The neighbours that lead are passages sharing one title: seven 'Paris', six 'Canada', seven 'Education' and
    seven 'World War II'. musique splits an article into paragraphs, and a title mention links a passage to every
    paragraph with that title, which also link to each other.
  - The second gold is found by its words, two or more hops away behind a hub. Its positions in K(st) were 132 to
    2,612 ('Nicolae Ceaușescu', 'Public Health Agency of Canada', 'Sri Lanka Matha'), or it was absent.
  - A walk cannot rank such a gold above the anchor's own neighbours, so on these questions the lists must lead.
- **hotpotqa:** RRF5 leads A1st at N = 25 (0.885 against 0.830 on s1eval) and trails it from N = 50 to 100.
- **webqsp's head:** A1st has the best head (0.376 at N = 25 on s1eval). The other sources are weak there (Kst 0.180,
  Kall 0.062 and R0 0.085), and RRF5 gives them three of five votes.

### What follows

By section 6:
- **RRF5 is the parameter-free pool for now.** The claim that our pools are at or above the SOTA pools on all six is
  not made. It falls short on metaqa (against LINKED), hotpotqa (N = 50 to 100, against A1) and webqsp's head
  (against A1).
- **The next pool change, with nothing fit, is declared in its own file (P1e).** Its target is the split above:
  - the walk from linked entities leads on metaqa;
  - the lists lead where a linked passage's walk floods;
  - A1's head stays whole on hotpotqa and webqsp.
- The learned gates stay beside and are never the pool. P1c stays held, and the models are retrained once the pool
  is final.
