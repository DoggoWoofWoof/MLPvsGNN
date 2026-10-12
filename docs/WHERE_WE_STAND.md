# Where we stand, 13 October 2026 (05:11)

This page is the place to start. It says what the project claims, what the numbers say today, and what is running.
Each stage's own document stays the record, and every number here is copied from one.

Every number here is a development number: chosen on development carves, often after many arms. The paper's numbers
come from one declared confirmation run of the chosen MLP and GNN on questions never used for selection.

## The three claims

The claims are fixed in [PAPER_CLAIMS_2026_10_09.md](PAPER_CLAIMS_2026_10_09.md), and the stages that test them are in
[PROGRAM_2026_10_09.md](PROGRAM_2026_10_09.md). Since 13 October the target is our own full corpora on all six datasets,
with B1 a calibration line, in the order of [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md): pools, links, the SOTA
models and the gap ledger, speed, and one model with its own representation.

| claim | what it says | where it stands |
| --- | --- | --- |
| 1 | The MLP keeps at least 90% of the GNN's R@5 gain over the first stage, with no message passing. | Holds on B1's three settings (0.97-0.99). The six-dataset test is the confirmation run's. |
| 2 | A GNN built without any LLM is at the state of the art (above HippoRAG 2 on B1). | Above on 2wiki and hotpotqa; below on musique. Without the corpus hyperlinks, 2wiki stays above. On the full corpora the pools come first: ours are below the best LLM-free SOTA pool order on five datasets (P1a, section 5). |
| 3 | The MLP is faster, cold first. | Both MLPs are faster than the universal GNN on all six datasets (twin0 1.46-1.89×; zrc 2.38-3.99×, webqsp 1.30×). The target is at least 2× faster than the fastest GNN, which is zsp on all six, and no MLP is there: zrc is 0.96-1.18× faster, and twin0 is faster than zsp only on webqsp. With every input kept, each MLP stays below 2× zsp on five datasets even if every shared stage cost nothing (section 3). |

## The models

| name | family | what it is | message passing |
| --- | --- | --- | --- |
| `zrc` | lean | The MLP base: zrm's model over zrc's chain entries. | no |
| `zsp` | lean | The GNN base: zrc's inputs plus one propagation step over each row's neighbours' scores. | yes, one step |
| `u_mlp_v2_mix` (twin0) | universal | The universal MLP: the v2 compile's 197 columns and the 1,536-wide embeddings. | no |
| `u_gnn_v2_ef` (gnn0) | universal | The universal GNN: the twin's inputs plus three relation-aware steps and the evidence flow. | yes, three steps |
| `zn` | diagnostic | The lean MLP with every edge-derived input removed. A speed ceiling, not a model. | no |
| `rrf` | baseline | Reciprocal-rank fusion of the dense encoder and SPLADE. No training. | no |

Within a family, both models read the same pools, carves and seeds. In the universal family they also read the same
inputs. In the lean family, each model reads what it uses ([F1_ONE_INPUT_SET.md](F1_ONE_INPUT_SET.md)), and today that
is the same set: every input family is used by both models.

## 1. Against the state of the art (B1)

R@5 ×100 on HippoRAG 2's three settings, 1,000 questions each, with 95% intervals
([B1B_MODELS_ON_HIPPORAG2.md](B1B_MODELS_ON_HIPPORAG2.md), [U1A_B1_WITHOUT_HYPERLINKS.md](U1A_B1_WITHOUT_HYPERLINKS.md)).
J5 is the fit trained on five datasets. Nothing was chosen on B1.

| setting | HippoRAG 2 | our RRF | zsp J5 (GNN) | zrc J5 (MLP) | MLP share of the GNN's gain | zsp J5, U1a's link rule in place of the hyperlinks |
| --- | ---: | ---: | --- | --- | ---: | --- |
| musique | 74.7 | 56.4 | 66.3 [64.5, 68.1] BELOW | 66.1 BELOW | 0.98 | 66.3 BELOW (the same graph) |
| 2wiki | 90.4 | 71.2 | **97.2 [96.5, 97.8] ABOVE** | 96.4 ABOVE | 0.97 | **95.9 ABOVE** |
| hotpotqa | 96.3 | 85.0 | **98.2 [97.6, 98.9] ABOVE** | 98.2 ABOVE | 0.99 | 95.0 BELOW |

- **The lead comes from the features and models, not the encoder.** Our first stage is below HippoRAG 2's
  NV-Embed-v2 on all three settings (56.4 vs 69.7, 71.2 vs 76.5, 85.0 vs 94.5), and our lift over it is larger
  (+9.9 vs +5.0, +25.9 vs +13.9, +13.2 vs +1.8).
- **The hyperlink caveat.** On 2wiki and hotpotqa, the structural family is the documents' own Wikipedia hyperlinks,
  which HippoRAG 2 does not use. One LLM-free link rule built from the passages alone keeps 2wiki 5.5 points above
  HippoRAG 2. On hotpotqa it falls 1.3 points short. Without any links the models fall well below it (77.3 on 2wiki,
  90.0 on hotpotqa), so a few precise edges are what the models need, not an LLM-built graph.
- **musique: the gap is ranking, not reach.** The pools hold 98.2% of its golds and every gold for 95.3% of questions,
  but R@5 is 66.3 ([MUSIQUE_DIAGNOSIS.md](MUSIQUE_DIAGNOSIS.md), [DIAG_LINK_WORDS.md](DIAG_LINK_WORDS.md)).
  - **Where the 8.4 points come from.** Our first stage starts 13.3 points below NV-Embed-v2 (56.4 against 69.7).
    Our models add 9.9 over it, twice HippoRAG 2's lift over its own encoder (5.0), which wins back 4.9 of the 13.3.
  - **By hop count** (zsp J5): R@5 is 77.0 on two-hop, 61.8 on three-hop and 41.3 on four-hop questions. 752 of the
    990 golds it misses are in three- and four-hop questions.
  - **The missed golds are next to found ones** (D1): 0.38 are direct neighbours of a found gold and 0.77 are within
    two hops, but among about 45 such neighbours the model ranks the missed gold 5th to 9th.
  - **The words around a link do not pick that neighbour** (D4, NO_SIGNAL). Ranking the neighbours by the question's
    words near each link (title mentions, shared entities, KB relation names) puts the missed gold in the top two
    0.261 of the time against zrc's 0.225, and 0.308 against 0.291 in the second fit. The declared bar was +0.05 and
    1.25× in both. As declared, no ranking screen follows, and musique's gap is reported as the first stage's.
- **A training-free rule** (B1d: two of the five slots go to the neighbours of the top passages) is already ABOVE
  on 2wiki (92.8) and close on hotpotqa (95.2). It is not universal: it costs squad and comparison questions (G1a).

On the six datasets' own eval carves (s1eval, full corpora), zsp J5's R@5 is metaqa 77.9, squad 91.0, musique 56.6,
hotpotqa 90.5, 2wiki 87.3, and webqsp 26.8 zero-shot (webqsp never trains) ([G1A_UNIVERSAL_BRIDGE_SIX.md](G1A_UNIVERSAL_BRIDGE_SIX.md)).

## 2. The MLP against the GNN (Claim 1)

- **On B1, the MLP keeps 0.97-0.99 of the GNN's gain** and is within 0.8 points of it in every in-domain row.
- **On the six eval carves**, zrc and zsp are within 0.015 R@5 of each other on every read of two splits but one
  ([ZN_LEAN_NO_MP.md](ZN_LEAN_NO_MP.md)). The exception is webqsp zero-shot on L-hotpotqa, where the MLP is higher
  (0.287 against 0.226).
- **The edges are what the MLP needs.** zn, the same MLP without any edge-derived input, keeps only 13-73% of zsp's
  gain (2wiki 0.71-0.73, hotpotqa 0.35-0.53, musique 0.13-0.49, metaqa 0.28, webqsp 0.15-0.25). Topology, depth,
  walks, distances and the KB chains carry most of the gain, and the MLP reads them without message passing: each is
  computed from the graph and the question before any row is scored.
- **Training longer moves nothing** (round 33, [FULL_ROUND33.md](FULL_ROUND33.md)). Doubling the steps gives no read
  past what reseeding the same fit moves, for either model, so both keep 8 epochs.
- **More training questions are not the lever.** Halving zrc's training data loses on 4 of 12 reads under the seed
  null ([DIAG_DATA_SCALE.md](DIAG_DATA_SCALE.md)). At the same number of optimiser steps, the half loses only one
  (round 33's step control, [SCREENS.md](SCREENS.md)), so D3's loss was mostly the steps', and by its rule no second
  fit carve is built.

## 3. Speed (Claim 3)

Cold means one question at a time from scratch: no prebuilt per-question caches and no warm-up. Every model runs at its
fastest form that keeps its scores (within 1e-4 and the same top 5) and reads every one of its inputs. Laptop, pinned,
one thread. Records: [C12_SHARED_COMPILE.md](C12_SHARED_COMPILE.md) (intervals there),
[C11_BOTH_FAMILIES_FAST.md](C11_BOTH_FAMILIES_FAST.md) and [C10_ZN_COLD.md](C10_ZN_COLD.md).

**Cold p50 per question, ms (C12, each model at its fastest form):**

| dataset | twin0 (universal MLP) | gnn0 (universal GNN) | zrc (lean MLP) | zsp (lean GNN) |
| --- | ---: | ---: | ---: | ---: |
| 2wiki | 4.11 | 6.00 | 2.52 | 2.61 |
| hotpotqa | 3.91 | 5.88 | 2.36 | 2.48 |
| squad | 2.67 | 4.24 | 1.65 | 1.73 |
| musique | 75.66 | 143.26 | 35.87 | 42.19 |
| metaqa | 49.47 | 77.45 | 26.42 | 28.07 |
| webqsp | 69.87 | 107.26 | 82.65 | 79.62 |

**GNN time over MLP time, cold:**

| dataset | universal: gnn0 / twin0 | lean: zsp / zrc | gnn0 / zrc | zsp / zn (C10) |
| --- | --- | --- | --- | --- |
| 2wiki | 1.46 | 1.04 | 2.38 | 5.22 |
| hotpotqa | 1.50 | 1.05 | 2.49 | 2.65 |
| squad | 1.59 | 1.05 | 2.57 | 2.71 |
| musique | 1.89 | 1.18 | 3.99 | 5.31 |
| metaqa | 1.57 | 1.06 | 2.93 | 3.95 |
| webqsp | 1.54 | 0.96 | 1.30 | 9.61 |

- **Both MLPs are faster than the universal GNN on all six datasets.**
  - The universal MLP is 1.46-1.89× faster.
  - The lean MLP is 2.38-3.99× faster on five datasets. On webqsp it is 1.30× faster, because its chain entries are its
    largest stage.
- **No MLP is 2× faster than the fastest GNN.** The fastest GNN is zsp on all six datasets. zrc is 1.04-1.18×
  faster than it on five datasets and slower on webqsp (0.96). twin0 is slower than it on five (zsp / twin0 0.56-0.65)
  and faster only on webqsp (1.14).
- **Why, with every input kept.** These figures are worked out from C12's stage p50s, not timed.
  - **Universal.** The compile both models run is 58-71% of twin0's time. Even if it cost nothing, gnn0 / twin0 would
    top out at 2.17 (2wiki, hotpotqa) to 4.08 (musique). For 2×, the compile would have to get 1.18× faster on
    musique, 3.0-3.7× on squad, metaqa and webqsp, and 8-9× on 2wiki and hotpotqa.
  - **Lean.** zsp is zrc plus its links and one propagation step, which cost 3-17% of zrc's time. Even if everything
    before the forwards cost nothing, zsp / zrc would be 1.16-1.34 on five datasets and 2.12 on musique.
  - **Across families.** On five datasets, zsp's whole time is only 1.47-1.95× twin0's forward alone, so no compile,
    however fast, brings twin0 to 2× zsp there. On webqsp it would take twin0's compile 2.6× faster.
- **zn**, the lean MLP with every edge-derived input removed, is 2.65-9.61× faster than zsp (C10). It keeps only
  13-73% of the gain, so it is a ceiling, not a model.
- **What made each model faster at the same scores:**
  - C11's forms: twin0 by 1.32-1.52×, gnn0 by 1.16-1.38×, zrc by 1.26-1.64× and zsp by 1.28-1.74×.
  - C12's kernels (the seed cosines made once per weight pattern; the node rows widened in one instruction): the full
    compile both universal models share got 1.02-1.10× faster (1.10 on musique). gnn0 / twin0 on musique went from
    1.84 to 1.89 with one form on both sides. The lean pair did not move (0.97-1.04×, every interval includes 1).
  - C12's compiles, lean rows, links, chain entries and scores equal C11's bit for bit on all 1,200 questions.
- **What is left in the universal compile.** A development profile of musique's compile put the per-family
  neighbour prototypes (about 24 ms) and the depth basis (about 16 ms) at the top. C12 did not touch either.

## 4. Links and edges (U1, G1a)

The aim is one LLM-free link rule for all six datasets, so the headline does not need the 2wiki and hotpotqa
hyperlinks.

| stage | what it did | result |
| --- | --- | --- |
| U1a | Title-mention rule on B1's settings | Recovers the hyperlinks at P 0.76 / R 0.81 (2wiki) and P 0.66 / R 0.79 (hotpotqa); musique's graph exactly. Keeps 94% (2wiki) and 61% (hotpotqa) of the hyperlinks' gain. |
| U1b | The same rule on the full corpora | Unusable on 2wiki, hotpotqa and webqsp: precision 0.10-0.14, hubs on common words ("From", "With", "County"), 40× webqsp's KB. |
| U1d | Case and corpus-statistic filters (V2 chosen) | Hyperlink F1 0.325 (2wiki), 0.433 (hotpotqa): precision more than doubled, recall 0.57 / 0.68. webqsp still gets 311M mention edges beside 8.3M triples. |
| U1e | Cap each target's in-degree (c = 1,024 chosen) | F1 rises to 0.395 / 0.467. 97.5% of webqsp's mention links go, every KB triple stays, pools are 0.735 of step 4e's size. |
| G1a | A fixed bridge-slot rule on all six | NOT universal: ABOVE on four datasets, SAME on musique, BELOW on squad. The trained models are 2.7-66 points above it. |
| U1c | zrc and zsp refit on U1d's graph with step 4h's pools ([U1C_RETRAIN_ON_U.md](U1C_RETRAIN_ON_U.md)) | NOT_ADOPTED: no GAIN in 24 reads, under the seed null too; zsp loses webqsp zero-shot (−0.056). |
| U1f | The same on U1e's damped graph, GU_1024 ([U1F_RETRAIN_ON_GU1024.md](U1F_RETRAIN_ON_GU1024.md)) | NOT_ADOPTED: no GAIN in 24 reads; LOSSes on musique (zrc, −0.010) and webqsp zero-shot (zsp, −0.043). |

The rule's links are still imprecise: on 2wiki about three in four are not hyperlinks. That is in the gap ledger.

**P2a, 13 October: what each link family is worth on the full corpora** ([P2A_LINK_FAMILIES.md](P2A_LINK_FAMILIES.md)).
Each family was left out of, or added to, the pools (part 1), and masked in today's fits' inputs with no retraining
(part 2).

| family | in the pools (part 1, ALL at \|I_q\|) | in today's fits (part 2, R@5 lost when masked) |
| --- | --- | --- |
| structural (KB triples, hyperlinks, musique's own) | Carries the reach on every graph: metaqa +0.540, 2wiki +0.310, webqsp +0.265, hotpotqa +0.165, musique +0.015 | Both fits rely on it: 0.17 to 0.77 on the linked datasets, 0.026 to 0.029 on musique. On 2wiki, masking it raises hit@1 by 0.010 to 0.016. |
| ner | Adds reach only on musique (+0.032); pushes golds out elsewhere | Carries musique for both fits (0.009 / 0.014); elsewhere only hit@1, for the MLP |
| knn | Adds reach only on musique; pushes golds out elsewhere | Mostly hit@1 (0.003 to 0.016). The GNN relies on it on all six (webqsp zero-shot R@5 0.013), the MLP on three. |

No family is a drop candidate, and no subset of families suits all six datasets, so no graph changes. The universal
form of what P2a found is a rule over the families (P2b): the structural family first, then ner and knn, which keeps
the head and adds musique's tail.

By U1f's rule, no full run uses either graph, and both bases keep today's graph and pools. The hyperlink price, what
the trained models lose without the hyperlinks on the eval carves, goes to the gap ledger. For both arms it is
0.035-0.065 R@5 on hotpotqa and 0.010-0.020 on 2wiki. The rule's links recover the hyperlinks as edges, but the
models do not get the hyperlinks' value back from them.

## 5. Pools (step 4, P1)

A pool is the candidate set a model reranks. ALL is the share of questions with every gold in the pool.

| step | what it did | verdict |
| --- | --- | --- |
| 4 | Union walk over all families | NOT_ADOPTED: the union dilutes the structural hop. |
| 4b | Walk on each dataset's own frozen graph | ADOPT: musique +0.032, webqsp +0.094, the rest level. |
| 4c | Step 1's pipeline retrained on 4b's pools | Mixed early read; its fits and grade never ran (a one-fit screen replaced them). |
| 4d | Union of the first-stage lists (U_q) | Its later stages moved into 4e. |
| 4e | HippoRAG 2's restart walk plus PullNet-style linking (A3, k = 2) | CHOSEN. ALL on s1eval: metaqa 0.953, squad 0.992, musique 0.937, hotpotqa 0.979, 2wiki 0.945, webqsp 0.909. Pools about 2× the first-stage union (3× on squad). |
| 4f | A ranked cut at half the size | NO_SHRINK: the coverage lives in the union, not near the top of any one list. |
| 4g | Where the misses are | One hop away on the KBs and musique (78-89%); the first stage on hotpotqa and squad. |
| 4h | A second expansion from the top-ranked nodes (W1, on U1d's graph) | CHOSEN: mean ALL 0.917 at 0.787 of 4e's size. On U1d's graph, hotpotqa (0.896) and 2wiki (0.873) lose reach the hyperlinks gave. |

Today's models were trained and read on step 1's frozen pools and today's graph. U1c and U1f tried both bases on the
universal graph with step 4h's pools. Neither was adopted, so the bases stay where they were.

**P1a, 13 October: our pools against the SOTA pool constructions, on the full corpora**
([P1A_POOL_SCOREBOARD.md](P1A_POOL_SCOREBOARD.md)). Every order walks the same graph with step 4e's kernel; only the
restart differs. On s1eval, our construction (A3, step 4e's chosen restart) is BELOW the best LLM-free SOTA order on
five datasets. squad has no graph, so every order there is the same.
- **HippoRAG 2's restart rule (A1) leads on the passage graphs and webqsp.** hotpotqa up to N = 750 (0.830 against
  0.512 at N = 25), 2wiki up to N = 300 (0.838 against 0.578 at N = 50; ours leads at 500 to 2,000), musique up to
  N = 300 (by up to 0.027), webqsp at every N (by 0.005 to 0.043).
- **The walk from our linked entities alone (LINKED) leads on metaqa** at every N (0.959 against 0.800 at N = 2,000).
  It comes within 0.036 of the walk from the assigned topic entities at ReaRev's m = 500 (0.868 against 0.903).
- **Headroom at no extra look cost.** Cut at each question's own pool size today, a walk's top N holds more golds than
  today's pools: metaqa +0.066, musique +0.082, webqsp +0.102, hotpotqa and 2wiki +0.005.

**P1b and P1b2: one restart for all six** ([P1B_RESTART_MIXTURE.md](P1B_RESTART_MIXTURE.md),
[P1B2_RANK_GATE.md](P1B2_RANK_GATE.md)). No fixed restart is best on all six, so both fit one per-question rule on the
training datasets' fit carves.
- **P1b's mixture (MX)** weights the seed, linked-entity and first-stage walks from fifteen label-free features of the
  question, fit by EM to the golds' mass. It is BELOW the best SOTA order on five datasets, because the call reads
  rank and the fit moved weight to the peaked seed walk.
- **P1b2's rank gate (RC)** adds HippoRAG 2's own walk and fits a hard choice among eight fixed blends to the golds'
  ranks.
  - It is AT_OR_ABOVE the best SOTA order on metaqa, squad, hotpotqa and 2wiki.
  - It is BELOW on musique (up to −0.070) and on webqsp's head (up to −0.083).
  - The per-question oracle over the same blends is AT_OR_ABOVE on all six, so the gap is the gate's. Its soft read
    is AT_OR_ABOVE on musique.
- **At each question's own size of today's pools, both are above today's pools on all six** (ALL, s1eval):

  | order | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | today's pools | 0.895 | 0.980 | 0.803 | 0.968 | 0.906 | 0.688 |
  | MX | 0.923 | 0.981 | 0.892 | 0.969 | 0.914 | 0.806 |
  | RC | 0.963 | 0.981 | 0.844 | 0.971 | 0.913 | 0.782 |
  | best SOTA order | 0.961 | 0.981 | 0.883 | 0.973 | 0.911 | 0.790 |

- **P1c reads MX,** by P1b2's declared rule: its mean over the five s1sel carves is 0.936, against RC's 0.926.
  - [P1C_POOLS_RETRAINED.md](P1C_POOLS_RETRAINED.md) (6996aea) rebuilds the models' pools from MX's order at each
    question's own size, so the look cost is unchanged.
  - It runs one screen fit per family, zrc and zsp.

## 6. Running now (13 October, 05:11)

- **P1c on the host** ([P1C_POOLS_RETRAINED.md](P1C_POOLS_RETRAINED.md)).
  - The six pool builds run first; they are light items.
  - Then the looks, caches and builds, about 45 GB, then the identity gate.
  - Then the four screen fits and their reads on the card, then the call.
- **Done since the last update:**
  - P1b and P1b2, declared, run and filed (section 5);
  - P2a, both parts, declared, run and filed (section 4).
- **Host disk: 28.6 GB free at 05:10.** That is below the 30 GB line, where the feeder sends only light items, so
  P1c's looks wait for room.
  - Clean-up 15 (U1c's and U1f's arrays) is listed and dry-run. It frees about 143 GB once both names of each
    hard-linked pair go.
  - It is the user's command.

## 7. Open gaps

| gap | the number | what works on it |
| --- | --- | --- |
| Pools against SOTA (full corpora) | P1b2's rank gate: at or above the best SOTA order on four datasets; below on musique (up to −0.070) and webqsp's head (up to −0.083). The oracle over the same blends is above on all six, so the gap is the gate's. | P1c screens the models on MX's pools (above today's on all six at the same size). A gate change goes in its own file. |
| musique ranking | 66.3 vs HippoRAG 2's 74.7; the pools hold 98.2% of golds; 752 of 990 misses are on three- and four-hop questions | D4 (link words): NO_SIGNAL, so the gap is reported as the first stage's. Rounds 29-32, U1c and D4 moved nothing here. |
| Hyperlink-free hotpotqa | 95.0 vs 96.3 | U1c and U1f NOT_ADOPTED: retraining on the universal graph costs hotpotqa 0.035-0.065 R@5. Open. |
| Link precision | Hyperlink F1 0.40 (2wiki), 0.47 (hotpotqa) | A universal rule; one file per change. |
| Pool misses | One hop on the KBs and musique; first stage on hotpotqa and squad | Step 4h covers the one-hop misses; the first-stage misses are open. |
| Zero-shot to webqsp | zsp 26.8 zero-shot; a model trained on webqsp reached about 60 (S6) | Longer training lifts the MLP's transfer, but under seed noise (round 33). |
| Training data | At equal steps, half the data loses 1 read of 12 (round 33's step control) | Not the lever: no second fit carve, by D3's rule. |
| MLP vs the fastest GNN | zsp is the fastest GNN on all six; zrc is 0.96-1.18× faster, twin0 slower except on webqsp (1.14× faster) | Not met. With every input kept, the forwards keep each MLP below 2× on five datasets (section 3). How Claim 3 is worded is open. |
| Universal speed | gnn0 / twin0 1.46-1.89× (C12) | P4, after P5a: per-question structural work moved to index time. C13 (exact per-family prototypes and depth basis) is parked: it reaches 2× on musique only; squad, metaqa and webqsp need the compile 3.0-3.7× faster. |
| Lean speed | zsp / zrc 0.96-1.18× (C12) | Bounded by the lean design: 1.16-1.34 even with free shared stages, 2.12 on musique. |

## How we got here

- **3 to 19 September: the regime map, the MLP selection and the controlled comparison (M0 to M3B).** Message passing
  helps multi-hop coverage on the three readable passage datasets, with a causal control (0bbb3c4).
- **19 September to 1 October: one model for everything, and MLP approximations.** The universal GNN matches or beats
  the per-dataset GATs. Its MLP twin fails its replication gate on 2wiki. The best non-MP approximation stays below the
  GNN.
- **1 to 7 October: the lean MLP and GNN tracks, and transfer.** The lean MLP comes close to the GNN in-domain. Joint
  training keeps most of the gain on a held-out passage graph. For unseen KB relations, only parts of the gap close.
- **7 to 9 October: one-fit screens and the bases.** Each idea gets one screen fit and is decided under the seed null.
  zrc became the MLP base and zsp the GNN base. B1b put both against HippoRAG 2.
- **9 to 12 October: links, pools and speed.** U1a to U1e built one LLM-free link rule. Steps 4e to 4h rebuilt the
  pools. C1 to C11 made every model cold-fast at identical scores. Round 33 found that longer training does not help.
- **12 to 13 October: the universal graph tried on the bases.** U1c and U1f refit zrc and zsp on it, and neither was
  adopted. D4 found that a link's words do not pick the neighbour a musique question follows (NO_SIGNAL). C12 made
  the full compile both universal models share faster, bit for bit (gnn0 / twin0 up to 1.89×).
- **13 October: the full corpora.** The target moved to our own corpora on all six datasets (2wiki 6.0M and hotpotqa
  5.2M passages).
  - P1a put our pools against the SOTA pool constructions there: below on five datasets.
  - P1b and P1b2 fit one restart rule for all six. The rank gate is at or above the SOTA order on four datasets, and
    both rules are above today's pools on all six at the same size.
  - P2a measured each link family: the structural family carries the reach and the ranking, and no family can be
    dropped.

## What is next

The order is [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md)'s. Each stage gets its own file, committed before its
numbers.
1. **P1c: the models on MX's pools** (running). One screen fit per family. A full run follows only if no read loses.
   The rank gate's gap on musique and webqsp is a gate change, in its own file.
2. **P2b: one link rule over the families.** The structural family first, then ner and knn, keeping the head and
   adding musique's tail. It is graded on all six, the pools first and then a one-fit screen.
3. **P3: the SOTA rankers beside our four models**, and the gap ledger for SOTA, GNN and MLP by hop slice.
4. **P5a: the model's own representation, read as a retriever first.** Per-node vectors and relation offsets, fit by
   EM and composed for multi-hop targets without walking the graph, with offsets for unseen relations from their
   names, all inside one model.
5. **P4: speed, after P5a.** 2-3× in every family with every win kept.
6. **Claim 3's wording and the confirmation run** of the chosen MLP and GNN on held-out questions (B1 read beside it,
   never used to choose).
