# Where we stand, 13 October 2026 (16:45)

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
| 1 | The MLP keeps at least 90% of the GNN's R@5 gain over the first stage, with no message passing. | Holds on B1's three settings (0.97-0.99). On the full corpora (P3a, beside): the lean MLP keeps at least 0.968 of zsp's gain over RRF on R@5 and FC@5 on all six; the universal MLP keeps at least 0.9 on metaqa, hotpotqa and webqsp, 0.852 on 2wiki and 0.596 on musique. Where the MLP loses to its GNN, it is mostly golds one link from a found gold on hotpotqa and 2wiki, and ranking elsewhere (P3b). The six-dataset test is the confirmation run's. |
| 2 | A GNN built without any LLM is at the state of the art (above HippoRAG 2 on B1). | Above on 2wiki and hotpotqa; below on musique. Without the corpus hyperlinks, 2wiki stays above. On the full corpora both GNNs are at or above the best LLM-free SOTA ranking on all six datasets, on R@5, FC@5 and hit@1 (P3a), and still at or above on metaqa and webqsp with the seeds removed, on a thin read (P3b). The final pool, P1e's parameter-free KIND_T, is at or above the best LLM-free SOTA pool order on four datasets, and within 0.005 and 0.007 of it on metaqa and hotpotqa (section 5). |
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

**P3a and P3b, 13 October: the SOTA rankings on our full corpora** ([P3A_SOTA_RANKERS.md](P3A_SOTA_RANKERS.md),
[P3B_GAP_LEDGER.md](P3B_GAP_LEDGER.md)). The bar is the best of eight LLM-free SOTA rankings per measure, run on our
encoder and graph:
- HippoRAG 2's restart walk, in two forms and each on all three families;
- the walk from our linked entities;
- the dense list, SPLADE and RRF.

Each is cut at the models' pool size. R@5 on s1eval:

| dataset | the bar (R@5) | zrc (lean MLP) | zsp (lean GNN) | twin0 (universal MLP) | gnn0 (universal GNN) |
| --- | --- | ---: | ---: | ---: | ---: |
| metaqa | LINKED 0.241 | 0.780 | 0.779 | 0.721 | 0.784 |
| squad | A1 0.905 | 0.911 | 0.910 | 0.897 | 0.904 |
| musique | RRF 0.473 | 0.567 | 0.566 | 0.512 | 0.538 |
| hotpotqa | RRF 0.685 | 0.905 | 0.905 | 0.879 | 0.899 |
| 2wiki | RRF 0.608 | 0.870 | 0.873 | 0.849 | 0.891 |
| webqsp | DENSE 0.110 | 0.307 | 0.268 | 0.602 | 0.629 |

webqsp is zero-shot for the lean pair and trained on for the universal pair.

- **Both GNNs are AT_OR_ABOVE the bar on all six datasets,** on R@5, FC@5 and hit@1 (paired bootstrap, tie 0.002):
  SOTA_GNN for zsp and for gnn0.
  - **Thin places:** squad (gnn0's R@5 is 0.0015 below, within the tie; twin0 is the only model BELOW anywhere),
    musique's hit@1 (gnn0 +0.015, the interval crosses 0), and twin0's hit@1 on hotpotqa and 2wiki (ties).
- **The caveat.** The bar runs on our LLM-free graphs. It is not HippoRAG 2 as released (an LLM-built graph and
  NV-Embed-v2; on B1 musique our GNN is below it, 66.3 against 74.7), and not the trained KB-QA reasoners.
- **With the seeds removed** (metaqa and webqsp, whose walk rows put the seeds first): all four models are AT_OR_ABOVE.
  The read is thin, though. The walk rows spend their first ten on the seeds, so it decides only 17 of 9,785 metaqa
  questions and 12 of 1,503 webqsp questions.
- **Where the bar wins, the GNN had the gold in its pool and ranked it out** (P3b). The bar's golds outside our pools
  are 0.0006 of R@5 on metaqa and 0 elsewhere. Every bar>GNN gap is mostly RANKED (lean's hotpotqa splits ranked and
  bridge). These leads are small: 0.004 to 0.063 of R@5 lost, against 0.02 to 0.55 won back.
- **Each model's own misses** (R@5 shares, P3b):
  - musique: ranked out 0.23-0.28, one link from a found gold 0.11-0.13, outside the pool 0.079 (0.198 on 4-hop);
  - webqsp, lean pair zero-shot: ranked out 0.43-0.47, against 0.12-0.15 for the universal pair trained on it;
  - outside the pool: webqsp 0.13, 2wiki 0.036, metaqa, squad and hotpotqa 0.020.

## 2. The MLP against the GNN (Claim 1)

- **On B1, the MLP keeps 0.97-0.99 of the GNN's gain** and is within 0.8 points of it in every in-domain row.
- **On the full corpora (P3a, P3b),** the lean MLP keeps at least 0.968 of zsp's gain over RRF on R@5 and FC@5 on all
  six datasets (hit@1 0.73-0.84 on hotpotqa, 2wiki and webqsp). The universal MLP keeps at least 0.9 of gnn0's gain on
  metaqa, hotpotqa and webqsp, 0.852 on 2wiki and 0.596 on musique.
- **What message passing buys over the MLP** (P3b's GNN>MLP ledger):
  - **hotpotqa and 2wiki:** golds one link from a gold the model found (BRIDGE), on their bridge-type questions. On
    the comparison questions it is ranking.
  - **musique:** split, with BRIDGE on 2-hop questions and ranking on 3- and 4-hop.
  - **metaqa, squad and webqsp:** ranking.

  The BRIDGE part is the MP-free target of P5's offsets (section 6).
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

**P2b, 13 October: one gate over the families, with no per-dataset regime** ([P2B_FAMILY_GATE.md](P2B_FAMILY_GATE.md)).
REGIME_STAYS. The gate over nine sources (MXU) is below P1b's MX by more than the tie on metaqa, squad and musique, and
above it on webqsp (0.857 against 0.806, zero-shot for the gate) and 2wiki. Beside, it is at or above the best SOTA
order on musique, 2wiki and webqsp, the three where P1b2's rank gate is not, so the two gates cover all six but neither
alone. The regimes stay, and P1b3 (its own file) fits a gate to the ranks over the nine sources.

**P2c part 1, 13 October: an LLM's graph without the LLM** ([P2C_LLM_FREE_TRIPLES.md](P2C_LLM_FREE_TRIPLES.md)
section 9, the user's direction of 08:50). Our triples come from a parser and fixed rules: spaCy's small CNN pipeline,
no transformer. They were compared on B1's musique and 2wiki passages with LLM-built extractions of the same passages:
HippoRAG 2's OpenIE (gpt-4o-mini; Llama-3.3-70B beside) and SiReRAG's propositions.
- **In the pool walk, ours adds more than every LLM graph.** Over the same base: musique +8.4 R@5 points against +3.7
  to +4.8, and 2wiki +3.8 against +1.5. MATCHED on both, at 1.73 to 2.55 times the reference's gain.
- **In HippoRAG 2's search with a string link:**
  - musique: ours adds more (+11.6 against +8.2 to +9.6), MATCHED;
  - 2wiki: 0.77 of SiReRAG's gain (+12.7 against +16.6), PARTIAL. SiReRAG's gain there is in its phrases, not its
    relation pairs.
- **Agreement.** Our phrases hold 73 to 87% of the references' phrases at 26 to 44% precision. Relation pairs match
  7 to 15%.
- **So on B1 musique, the LLM graph is not what our walks lack.**

**P2c part 2, 13 October: our triples on our own corpora** ([P2C2_FOURTH_FAMILY.md](P2C2_FOURTH_FAMILY.md), a screen on
the fit carves). NO_GAIN, so no full run follows.
- **What was tested:** the P form of our parse of every passage of the four passage corpora (up to 84.8M edges on
  2wiki), as a fourth family in the pool walks under P1e's rule.
- **The result:** it moves the pool's ALL by -0.007 to +0.002 at N = 25 and by at most 0.001 at the models' sizes.
  The walks from the first stage's seeds gain at the head on squad and musique. The walks from the links lose on every
  corpus with links, and the two cancel.
- **Why:** most of its edges are new (66% on musique and 74% on squad are in none of the three families), but they move
  no golds into the pool's head.
- **So the links stage closes with the three families.** Part 1's gain on B1's small corpora does not carry to our full
  corpora, where the three families and a rule that fuses the first stage's lists already hold what it adds.

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

- **P1c is held** (f8da4b0) and superseded by P1c2. It would have retrained the models on MX's pools
  ([P1C_POOLS_RETRAINED.md](P1C_POOLS_RETRAINED.md)), and MX is fit to the golds. Under the user's rule (below), MX,
  RC and MXU stay beside as learned upper lines and are never the pool. P1c2
  ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)) runs the same stage on the final pool, KIND_T.
- **P2b's gate (MXU) without regimes** is the complement of RC: at or above the best SOTA order on musique, 2wiki and
  webqsp (webqsp 0.857 at |I_q|). The regimes stay (section 4).
- **What the pools still miss** (P3b): no bar>GNN gap is the pool's, since the bar's golds are inside our pools. Each
  model's own POOL share is largest on musique's 4-hop questions (0.198) and webqsp (0.13). That is what the later
  pool stages target.

**P1c2's pools, 13 October 15:48: the final pools, written per question** ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)).
Each question keeps base and seeds, then takes KIND_T's first nodes outside them up to today's size. All six jobs pass
their checks, and no question is short. All golds in the pool (ALL, s1eval):

| pool | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| today's pools | 0.895 | 0.980 | 0.803 | 0.968 | 0.906 | 0.688 |
| P1c2's final pools | **0.961** | 0.980 | **0.900** | 0.959 | **0.959** | **0.841** |

- Above today's pools on four datasets, the same on squad (its 50-node pools are base and seeds), and 0.009 below on
  hotpotqa.
- Keeping base and seeds moves KIND_T's own top |I_q| by at most 0.009: hotpotqa 0.965 to 0.959, webqsp 0.832 to
  0.841.
- The pools hold about 2,000 nodes on metaqa, musique and webqsp, about 100 on hotpotqa and 2wiki, and 50 on squad.

**The user's rule, 13 October, about 10:40: the pools first, with nothing learned**
([PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md)).
- The links and the final pool are one rule for all six datasets: nothing fit to golds, nothing set per dataset, and
  fixed published constants.
- The rule is judged by golds per node, against the best SOTA pool order on all six.
- The node-space EM and the learned offsets come after, as rerankers, with the models retrained on the final pools.

**P1d: one parameter-free pool rule** ([P1D_FREE_POOL.md](P1D_FREE_POOL.md) section 8, filed 8d6bdc7). RRF5 is
chosen, and the call is NOT_ALL_SIX.
- **The rule:** RRF over five sources with the first stage's constant (60), and nothing fit. The sources:
  - HippoRAG 2's restart (A1) and the walk from our corroborated links (the linked nodes the first stage's top 100
    also holds), each on the structural family alone and on all three families;
  - the lists, unwalked.
- **Against P1a's bar on s1eval:**
  - AT_OR_ABOVE on squad, musique and 2wiki.
  - BELOW on metaqa: 29 points, up to -0.114, against the walk from our linked entities.
  - BELOW on hotpotqa: 4 points, up to -0.008, at N = 50 to 100.
  - BELOW on webqsp's head: -0.054 at N = 25.
- **At each question's size of today's pools** (ALL, s1eval):

  | order | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | today's pools | 0.895 | 0.980 | 0.803 | 0.968 | 0.906 | 0.688 |
  | RRF5 | 0.910 | 0.979 | 0.899 | 0.963 | 0.961 | 0.876 |
  | best SOTA order | 0.961 | 0.981 | 0.883 | 0.973 | 0.911 | 0.790 |

  - With nothing fit, RRF5 is above the learned gates on musique, 2wiki and webqsp.
  - It holds today's golds in about a quarter of the nodes on musique and webqsp, and in half on 2wiki.
- **Why it falls short.**
  - On metaqa's KB, the walk from the links must lead: 0.559 against RRF5's 0.445 at N = 25 (s1sel).
  - On musique, that walk fails: a linked passage's walk floods its neighbours (paragraphs sharing a title, such as
    seven 'Paris'), while the second gold is found by its words.
  - P1e, the next pool change with nothing fit, targets this split.

**P1e: what the links land on decides how they are used** ([P1E_ANCHOR_KIND.md](P1E_ANCHOR_KIND.md) section 8,
filed db5ce0a). KIND_T is chosen, and the call is NOT_ALL_SIX. **It is the final pool.**
- **The rule.** A question is an entity question when the nodes it starts from (its corroborated links, or its seeds)
  are all name-only: KB nodes whose text is only a name.
  - Entity questions take ROUTEst. When every link is corroborated, that is the walk from the links on the structural
    family, backfilled by A1. Otherwise it is A1.
  - Passage questions take RRF5T. That is P1d's RRF5, with the two walks from the links steered by the first stage's
    lists: within a family, a neighbour's share of the mass follows its RRF weight.
  - Nothing is fit. The constants are RRF's 60 and step 4e's walk.
- **Against P1a's bar on s1eval:**
  - AT_OR_ABOVE on squad, musique, 2wiki and webqsp.
  - BELOW on metaqa: 4 points, at most -0.005, against the walk from our linked entities.
  - BELOW on hotpotqa: 3 points, at most -0.007, against A1 at N = 50 to 100.
  - From P1d's RRF5, metaqa goes from 29 points to 4, and webqsp from 3 to none.
- **At each question's size of today's pools** (ALL, s1eval):

  | order | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
  | --- | ---: | ---: | ---: | ---: | ---: | ---: |
  | today's pools | 0.895 | 0.980 | 0.803 | 0.968 | 0.906 | 0.688 |
  | KIND_T | **0.962** | 0.979 | **0.900** | 0.965 | **0.960** | **0.832** |
  | best SOTA order | 0.961 | 0.981 | 0.883 | 0.973 | 0.911 | 0.790 |

- **Golds per node.** KIND_T reaches today's pools' ALL in 750 of 2,018 rows on metaqa. It needs about a quarter of
  the rows on musique and webqsp (500) and half on 2wiki (50 of 106). hotpotqa needs 150 rows against today's 94.
- **What is left, and why no further pool stage is declared:**
  - **metaqa:** on the 5% of questions with an uncorroborated link, the walk from all their links leads.
  - **webqsp:** on its K branch (29% of questions), the walk from the corroborated links is behind A1 on the structural
    family: 0.227 against 0.355 at N = 25. Like-for-like, webqsp is below by up to 0.051.
  - **So the two KBs pull the entity branch in opposite directions.** No label-free signal we have separates them, and
    a fixed change that helps one costs the other.
  - **hotpotqa:** A1 leads the fusion from N = 50 to 100 by up to 0.007. A rule with more A1 votes closes most of it,
    but loses squad and 2wiki's head.

## 6. P5: the model's own representation

**P5a, 13 October: the node space read as a retriever on its own** ([P5A_NODE_SPACE.md](P5A_NODE_SPACE.md)). STOPS.
- **What was built.** Each node got its own vector, fit to its corpus's graph from the frozen vector (TransE-like, with
  latent relation offsets by EM). A map from the question to a chain of one or two offsets was trained on the fit
  carves.
- **The result.** From the s1eval seeds, the space found more of the golds that today's pools miss, and of those zsp
  ranks below 5, than the graph's own two hops only on metaqa. It found fewer on the other five datasets.
- **The map collapsed.** One chain took all 2,000 sampled questions on squad, musique, hotpotqa and 2wiki. With the best
  chain per question (an oracle, not a result), the space beats the hops on the low read on five of six datasets, and
  beyond two hops it finds golds the hops cannot (metaqa 0.148 against 0 at 1,000 candidates). So the space holds more
  than the map reads.
- **Next, after the final pools: a learned offset g(question, seed, graph features),** from the user's direction of
  08:50, as a reranker.
  - It is a function of all three, with the seed's graph features computed at index time, so the MLP stays MP-free.
  - Its target is P3b's BRIDGE part of GNN>MLP.
  - It goes in its own file.

**P5b, declared 16:05 (55c3bbb): the learned offset as a reranker on the final pools**
([P5B_LEARNED_OFFSETS.md](P5B_LEARNED_OFFSETS.md)).
- **g:** one network for all six corpora (200,465 weights). For each seed it reads the question, the seed's vector in
  P5a's space, the seed's index-time neighbour mean and degrees, and its first-stage ranks. It gives four offsets and
  their weights; a candidate's score is its nearness to the seeds' targets.
- **The fit:** EM over which seed and offset explain each gold, on P5a's fit samples over P1c2's final pools. webqsp
  never fits.
- **The call:** H2 (ring 1, ring 2, then the rest, each by cosine) fused with g's order, against H2 alone, on the golds
  zsp leaves below its top 5. GOES_FURTHER if it finds more on some dataset and fewer on none; then P5c gives the
  models g's scores as inputs.

**P5b's result, 16:45: STOPS** ([P5B_LEARNED_OFFSETS.md](P5B_LEARNED_OFFSETS.md), section 8).
- **The call.** Fused with H2, g finds more of the low golds on metaqa and 2wiki, ties on squad and hotpotqa, finds
  fewer on musique, and is mixed on webqsp (zero-shot).
- **Where it works: 2wiki and metaqa.** At 10 candidates, g alone finds 0.322 of 2wiki's low golds against H2's
  0.136, and 0.076 of metaqa's against 0.009. It finds golds two or more links from the seeds, which the rings cannot
  order (2wiki at 25: 0.394 of the golds beyond ring 2, H2 none).
- **Where it fails.** musique is near chance (0.007 at 10, against H2's 0.159), and webqsp, which g never saw, is
  below H2 (0.012 against 0.026).
- **Why (beside the call): the fit memorized its questions.** Its own loss on musique's fit questions fell to 0.88
  nats per gold, with 200,465 weights, 30 epochs and no held-out check. On squad's 104 fit questions it fell to 0.004.
- **Next: P5b2.** The same g, stopped at the epoch that best predicts held-out golds no training question had, with
  every number split into golds the fit questions had and new ones. It gets its own file before any number.

## 7. Running now (13 October, 16:45)

- **P5b** filed: STOPS (section 6). It ran in about 10 minutes on the host. **P5b2** is next, declared in its own
  file before any number; it reuses P5b's bases on the host.
- **P1c2** ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)) retrains the two lean bases, zrc and zsp, on the final pools,
  with one screen fit per family on L-musique and L-hotpotqa. P1c's rule decides: a full run follows only if neither
  arm loses a read and one gains.
  - **Done:** its pools, on all six (section 5).
  - **Waiting for clean-up 15:** its looks, caches and fits need about 45 GB. The host's drive has about 22 GB free.
- **Clean-up 15** (U1c's and U1f's arrays, about 143 GB freed) is the user's command. Until it runs, the feeder sends
  only light items.
- **Done since the last update:**
  - P1e filed (db5ce0a): KIND_T, the final pool;
  - P2c part 2 closed as a screen, NO_GAIN (6a5aedc);
  - P1c2's pools written on all six;
  - P5b declared (55c3bbb) and filed: STOPS.

## 8. Open gaps

| gap | the number | what works on it |
| --- | --- | --- |
| Pools against SOTA (full corpora) | P1e's KIND_T, parameter-free, the final pool: at or above the best SOTA order on squad, musique, 2wiki and webqsp; below on metaqa (up to −0.005) and hotpotqa (up to −0.007). Like-for-like, webqsp is below A1 on the structural family (up to −0.051). | Closed. metaqa and webqsp pull the entity branch in opposite directions, so no further fixed rule is declared. The learned gates stay beside. |
| SOTA rankers (full corpora) | Both GNNs at or above the best LLM-free SOTA ranking on all six (P3a). Where the bar wins, the gold was in our pool and ranked out (P3b). | P5's representation. A seeds-removed read on more than 17 and 12 questions needs P3a's walk rows kept past ten nodes. |
| What message passing buys | GNN>MLP is golds one link from a found gold on hotpotqa and 2wiki (P3b): lean 0.009-0.016 of R@5, universal 0.034-0.051 | P5b STOPS: the learned offset g(question, seed, graph features) finds more of these golds on 2wiki and metaqa, but its fit memorized (musique near chance). P5b2 next: the same g with a held-out stop (section 6). P5a's fixed map STOPS. |
| musique ranking | 66.3 vs HippoRAG 2's 74.7; the pools hold 98.2% of golds; 752 of 990 misses are on three- and four-hop questions. On the full corpus (P3b): ranked out 0.23-0.28 of R@5, one link from a found gold 0.11-0.13, outside the pool 0.079 (0.198 on 4-hop). | D4 (link words): NO_SIGNAL. P2c part 1: on B1 musique our LLM-free graph adds more than the LLM-built graphs, so the LLM graph is not what our walks lack. |
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
    dropped. P2b found that one gate without the per-dataset regimes is not enough (REGIME_STAYS).
  - P3a put both GNNs at or above the best LLM-free SOTA ranking on all six datasets. P3b's ledger found that the
    bar's wins are golds our GNNs had and ranked out, and that message passing buys golds one link from a found one.
  - P5a's node space, read as a retriever, STOPS: its question-to-chain map collapsed to one chain per corpus.
  - The user's direction at 08:50: LLM-free triples in place of an LLM-built graph (P2c), and offsets learned as a
    function of the question, the seed and the graph.
  - P2c part 1: our LLM-free graph adds more than the LLM-built graphs in the pool walk on B1's musique and 2wiki.
  - The user's rule at 10:40. The links and the final pool are parameter-free and the same for all six. Learning
    comes after, as reranking: the node-space EM and the learned offsets. P1c is held, and P1d tests one fixed rule.
  - P1d chose RRF over five sources. It is at or above the SOTA pools on three datasets, and holds today's golds in a
    quarter to half of the nodes on three.
  - P1e split the rule by what the links land on: the walk from the links for entity questions, a steered fusion for
    passage questions. It is at or above the SOTA pools on four datasets and within 0.007 on the other two. It is the
    final pool.
  - P2c part 2: our triples add no golds per node to it on our corpora, so the links stay the three families.
  - P1c2 wrote the final pools per question: above today's pools on four datasets at the same size.
  - P5b read a learned offset as a reranker on them. It finds more of the golds zsp leaves low on 2wiki and metaqa, but
    its fit memorized its questions, and it STOPS.

## What is next

The order is [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md)'s, under the user's rule of 10:40. Each stage gets its own
file, committed before its numbers.
1. **The pool, with nothing learned: final.** P1e's KIND_T (NOT_ALL_SIX: metaqa up to −0.005, hotpotqa up to
   −0.007). The links stay the three families: P2c part 2's fourth family is NO_GAIN.
2. **The models on the final pools.** P1c2 ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)) retrains both bases on
   KIND_T's pools at today's sizes, with one screen fit per family. A full run follows only if no read loses. Its
   pools are written; its looks, caches and fits wait for clean-up 15.
3. **Rerankers on the final pools:** the learned offset g(question, seed, graph features), fit by EM over the latent
   seed and offset, aimed at the BRIDGE part of GNN>MLP and at the RANKED misses. P5b STOPS: its fit memorized. P5b2
   stops the fit on held-out new golds. If it goes further, P5c gives the models its scores as inputs.
4. **P4: speed.** 2-3× in every family with every win kept.
5. **Claim 3's wording and the confirmation run** of the chosen MLP and GNN on held-out questions. B1 is read beside
   it and never used to choose.
