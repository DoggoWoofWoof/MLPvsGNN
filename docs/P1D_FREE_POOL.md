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
