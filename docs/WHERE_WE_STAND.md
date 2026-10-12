# Where we stand, 12 October 2026

This page is the place to start. It says what the project claims, what the numbers say today, and what is running.
Each stage's own document stays the record, and every number here is copied from one.

Every number here is a development number: chosen on development carves, often after many arms. The paper's numbers
come from one declared confirmation run of the chosen MLP and GNN on questions never used for selection.

## The three claims

The claims are fixed in [PAPER_CLAIMS_2026_10_09.md](PAPER_CLAIMS_2026_10_09.md), and the stages that test them are in
[PROGRAM_2026_10_09.md](PROGRAM_2026_10_09.md).

| claim | what it says | where it stands |
| --- | --- | --- |
| 1 | The MLP keeps at least 90% of the GNN's R@5 gain over the first stage, with no message passing. | Holds on B1's three settings (0.97-0.99). The six-dataset test is the confirmation run's. |
| 2 | A GNN built without any LLM is at the state of the art (above HippoRAG 2 on B1). | Above on 2wiki and hotpotqa; below on musique. Without the corpus hyperlinks, 2wiki stays above. |
| 3 | The MLP is faster, cold first. | Both MLPs are faster than the universal GNN on all six datasets (zrc: 2.00-3.43×, webqsp 1.33×). Against the lean GNN, whose message passing is one cheap step, zrc is at most 1.21× (section 3). |

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
  but R@5 is 66.3. The gain is on two-hop questions; three- and four-hop questions get almost none
  ([MUSIQUE_DIAGNOSIS.md](MUSIQUE_DIAGNOSIS.md)).
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
- **More data would help.** Halving zrc's training data loses on 4 of 12 reads under the seed null, and quartering it
  loses on 10 ([DIAG_DATA_SCALE.md](DIAG_DATA_SCALE.md): DATA_LIMITED).

## 3. Speed (Claim 3)

Cold means one question at a time from scratch: no prebuilt per-question caches and no warm-up. Every model runs at its
fastest form that keeps its scores (within 1e-4 and the same top 5). Laptop, pinned, one thread.

**GNN time over MLP time, cold, p50:**

| dataset | universal: gnn0 / twin0 (C9) | lean: zsp / zrc (C9) | gnn0 / zrc (C4, C6) | fastest zsp / zn (C10) |
| --- | --- | --- | --- | --- |
| 2wiki | 1.27 | 1.04 | 2.05 | 5.22 |
| hotpotqa | 1.29 | 1.21 | 2.00 | 2.65 |
| squad | 1.40 | 1.16 | 2.27 | 2.71 |
| musique | 1.65 | 1.13 | 3.43 | 5.31 |
| metaqa | 1.40 | 1.05 | 2.42 | 3.95 |
| webqsp | 1.42 | 0.94 | 1.33 | 9.61 |

Records: [C9_UNIVERSAL_PAIR.md](C9_UNIVERSAL_PAIR.md), [C4_GALLOP_EDGES.md](C4_GALLOP_EDGES.md),
[C6_FAST_CHAINS.md](C6_FAST_CHAINS.md), [C10_ZN_COLD.md](C10_ZN_COLD.md). On 2wiki, C10's conservative reading is 2.65.

- **Both MLPs are faster than the universal GNN on all six datasets.** Against the lean GNN zsp, zrc is 1.04-1.21×
  (2wiki's interval includes 1), and 0.94× on webqsp, where zsp's chain rule builds faster than zrc's.
- **Within a family, the shared stages bound the lead.** The stages both models run are 53-66% of the universal MLP's
  cold time (pool, node read, compile; most of the rest is the input block, which the GNN runs too) and 68-81% of
  zrc's (pool, node read, compile, lean inputs, chains). Even if they cost nothing, the lead would top out at
  1.74-2.85× (universal) and 1.15-1.55× (lean).
- **The lean GNN's message passing is cheap.** zsp is zrc plus one propagation step over scores. It costs 4-21% more
  cold, and on the screens' reads it adds at most 0.015 R@5. So no fair change makes zrc 2× faster than zsp.
- **2× or more** holds against a GNN whose message passing does real work: zrc against the universal GNN is 2.00-3.43×
  on five datasets, but 1.33× on webqsp. zn reaches 2.65-9.61× against zsp's fastest form, but it loses most of the
  gain, so it is the ceiling, not a model.
- **Next, C11:** all four models at C10's fastest forms, with identical scores, and the shared input layer made
  faster for both models of a family alike. Its checks run outside the timed loop (C10 found that checks inside it
  slow 2wiki's next GNN compile by evicting the edge stores).

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

The rule's links are still imprecise: on 2wiki about three in four are not hyperlinks. That is in the gap ledger.

## 5. Pools (step 4)

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

Today's models were trained and read on step 1's frozen pools and today's graph. U1c and U1f move both bases onto the
universal graph and step 4h's pools.

## 6. Running now (12 October, 20:00)

- **U1c screen** ([U1C_RETRAIN_ON_U.md](U1C_RETRAIN_ON_U.md)): zrc and zsp refit on U1d's graph (GU) with step 4h's
  pools, on L-musique and L-hotpotqa. The two zrc fits are training on the host's card. The zsp fits follow, then the
  reads and calls. Result expected around 23:00 to 01:00.
- **U1f screen** ([U1F_RETRAIN_ON_GU1024.md](U1F_RETRAIN_ON_GU1024.md)): the same screen on U1e's graph (GU_1024).
  The pool edges are building. Result expected on 13 October.
- **Which graph goes to the full run is fixed by rule.** If both pass, GU_1024. If one passes, that one. If neither
  passes, there is no full run, and the reads go to the gap ledger.
- **Host disk:** 69 GB free after clean-ups 13 and 14. The next clean-up list follows U1c's and U1f's chain builds.

## 7. Open gaps

| gap | the number | what works on it |
| --- | --- | --- |
| musique ranking | 66.3 vs HippoRAG 2's 74.7; the pools hold 98.2% of golds | Ranking, not reach. Three- and four-hop questions get almost nothing. |
| Hyperlink-free hotpotqa | 95.0 vs 96.3 | U1c / U1f: retrain on the universal graph. |
| Link precision | Hyperlink F1 0.40 (2wiki), 0.47 (hotpotqa) | A universal rule; one file per change. |
| Pool misses | One hop on the KBs and musique; first stage on hotpotqa and squad | Step 4h covers the one-hop misses; the first-stage misses are open. |
| Zero-shot to webqsp | zsp 26.8 zero-shot; a model trained on webqsp reached about 60 (S6) | Longer training lifts the MLP's transfer, but under seed noise (round 33). |
| Training data | DATA_LIMITED | No new data source declared. |
| Lean speed | zsp / zrc 0.94-1.21×, bounded at 1.55× | C11 makes both faster. The bound is the lean design's. |

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
  pools. C1 to C10 made every model cold-fast at identical scores. Round 33 found that longer training does not help.

## What is next

1. **U1c and U1f**, then the full run on the chosen graph, with B1 read beside it (never used to choose).
2. **C11**: all four models at their fastest forms, with identical scores, and the shared input layer made faster for
   both models of a family alike.
3. **The musique ranking gap**: three- and four-hop questions, as one universal change graded on all six datasets.
4. **The confirmation run** of the chosen MLP and GNN on held-out questions, for the paper.
