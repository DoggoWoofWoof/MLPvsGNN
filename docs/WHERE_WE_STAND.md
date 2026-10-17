# Where we stand, 15 October 2026 (20:30)

This page is the place to start. It says what the project claims, what the numbers say today, and what is running.
Each stage's own document stays the record, and every number here is copied from one. To learn the project from
scratch, with worked examples, read [PROJECT_STORY.md](PROJECT_STORY.md); the dated record of what was tried and why
is [PROJECT_TIMELINE.md](PROJECT_TIMELINE.md).

Every number here is a development number: chosen on development carves, often after many arms. The paper's numbers
come from one declared confirmation run of the chosen MLP and GNN on questions never used for selection.

## The three claims

The claims are fixed in [PAPER_CLAIMS_2026_10_09.md](PAPER_CLAIMS_2026_10_09.md), and the stages that test them are in
[PROGRAM_2026_10_09.md](PROGRAM_2026_10_09.md). Since 13 October the target is our own full corpora on all six datasets,
with B1 a calibration line, in the order of [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md): pools, links, the SOTA
models and the gap ledger, speed, and one model with its own representation.

| claim | what it says | where it stands |
| --- | --- | --- |
| 1 | The MLP keeps at least 90% of the GNN's R@5 gain over the first stage, with no message passing. | Holds on B1's three settings (0.97-0.99). The lean pair since 14 October, (zrckp, zspk), with 95% question-bootstrap intervals (P3c, section 2): **CLAIM1_NOT_ALL**. In-domain it holds on metaqa, hotpotqa and 2wiki (lower bounds 0.996, 0.999, 0.981); on squad (1.302 [0.859, 2.824], a GNN gain of only +0.0045) and musique (0.933 [0.881, 0.983]) the share is above 0.9 but not shown. Zero-shot it holds on all six. On the full corpora (P3a, beside), the earlier pair: the lean MLP keeps at least 0.968 of zsp's gain over RRF on R@5 and FC@5 on all six; the universal MLP keeps at least 0.9 on metaqa, hotpotqa and webqsp, 0.852 on 2wiki and 0.596 on musique. Where the MLP loses to its GNN, it is mostly golds one link from a found gold on hotpotqa and 2wiki, and ranking elsewhere (P3b). The six-dataset test is the confirmation run's. |
| 2 | A GNN built without any LLM is at the state of the art (above HippoRAG 2 on B1). | Above on 2wiki and hotpotqa; below on musique. Without the corpus hyperlinks, 2wiki stays above. On the full corpora both GNNs are at or above the best LLM-free SOTA ranking on all six datasets, on R@5, FC@5 and hit@1 (P3a), and still at or above on metaqa and webqsp with the seeds removed, on a thin read (P3b). The final pool, P1e's parameter-free KIND_T, is at or above the best LLM-free SOTA pool order on four datasets, and within 0.005 and 0.007 of it on metaqa and hotpotqa (section 5). |
| 3 | The MLP is faster, cold first. | Both MLPs are faster than the universal GNN on all six datasets (twin0 1.46-1.89×; zrc 2.38-3.99×, webqsp 1.30×). The target is at least 2× faster than the fastest GNN, which is zsp on all six, and no MLP is there: zrc is 0.96-1.18× faster, and twin0 is faster than zsp only on webqsp. With every input kept, each MLP stays below 2× zsp on five datasets even if every shared stage cost nothing (section 3). The final pool's build is shared by every model (P4a): on hotpotqa and 2wiki it is 99% of every model's cold end to end, and every ratio there is 1.00-1.02 end to end. P4b's exact walks build the same pools 1.14-3.75× faster (hotpotqa 171 → 48 ms, 2wiki 188 → 50 ms); the build is still about 20 times zrc's end to end there, and every ratio there is 1.00-1.08. The KT input the lean pair now reads (P4c) adds 4-7 ms per question on musique, hotpotqa and 2wiki, 22 ms on squad, 32 ms on webqsp and 39 ms on metaqa, mostly walks. zspk / zrck is 0.98-1.08, and zrck is slower than gnn0 on four datasets. |

## The models

| name | family | what it is | message passing |
| --- | --- | --- | --- |
| `zrckp` | lean, the MLP since 14 October | zrc plus a pointwise head over the row's own eight KT columns (161 parameters, no score input), on KIND_T's pools. Adopted by P1c3p's full run (section 5). | no |
| `zspk` | lean, the GNN since 14 October | zsp plus P1c3's KT head (the eight KT columns and the row's score z-scored over its question's rows), on KIND_T's pools. Adopted by P1c3's full run, paired with zrckp. | yes, one step |
| `zrc` | lean, base | The MLP base: zrm's model over zrc's chain entries, on today's pools. | no |
| `zsp` | lean, base | The GNN base: zrc's inputs plus one propagation step over each row's neighbours' scores. | yes, one step |
| `u_mlp_v2_mix` (twin0) | universal | The universal MLP: 129 columns of the v2 compile's 197 (the frozen contract) and the 1,536-wide embeddings. Trains on all six datasets, webqsp included. | no |
| `u_gnn_v2_ef` (gnn0) | universal | The universal GNN: the twin's inputs plus three relation-aware steps and the evidence flow. | yes, three steps |
| `zn` | diagnostic | The lean MLP with every edge-derived input removed. A speed ceiling, not a model. | no |
| `rrf` | baseline | Reciprocal-rank fusion of the dense encoder and SPLADE. No training. | no |

Within a family, both models read the same pools, carves and seeds. In the universal family they also read the same
inputs. In the lean family, each model reads what it uses ([F1_ONE_INPUT_SET.md](F1_ONE_INPUT_SET.md)), and today that
is the same set: every input family is used by both models.

## 1. Against the state of the art (B1)

R@5 ×100 on HippoRAG 2's three settings, 1,000 questions each, with 95% intervals
([B1B_MODELS_ON_HIPPORAG2.md](B1B_MODELS_ON_HIPPORAG2.md), [U1A_B1_WITHOUT_HYPERLINKS.md](U1A_B1_WITHOUT_HYPERLINKS.md)).
J5 is the fit trained on five datasets. Nothing was chosen on B1's runs, but B1's questions are rows of s1eval, which
every adoption decision read (section 8; a re-call without them is proposed).

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
    words near each link (title mentions, shared entities, KB relation names), fused with zrc's own order, puts the
    missed gold in the top two 0.261 of the time against zrc's 0.225, and 0.308 against 0.291 in the second fit. The
    declared bar was +0.05 and 1.25× in both. As declared, no ranking screen follows, and musique's gap is reported
    as the first stage's.
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
- **The lean pair since 14 October, (zrckp, zspk), with intervals** (P3c,
  [P3C_CLAIM1_INTERVALS.md](P3C_CLAIM1_INTERVALS.md)): the MLP's share of its GNN's gain over RRF on the J5 reads,
  with 95% intervals from 2,000 question resamples. **Call: CLAIM1_NOT_ALL.**

  | measure | metaqa | squad | musique | hotpotqa | 2wiki | webqsp (zs) |
  | --- | --- | --- | --- | --- | --- | --- |
  | R@5 | 0.997 [0.996, 0.998] | 1.302 [0.859, 2.824] | 0.933 [0.881, 0.983] | 1.010 [0.999, 1.020] | 0.986 [0.981, 0.991] | 1.294 [1.211, 1.385] |
  | FC@5 | 0.998 [0.996, 1.001] | 1.302 [0.859, 2.824] | 0.990 [0.918, 1.065] | 1.011 [1.001, 1.021] | 0.984 [0.977, 0.990] | 1.351 [1.243, 1.472] |
  | hit@1 | 0.986 [0.983, 0.990] | 1.577 [1.262, 2.126] | 0.936 [0.822, 1.054] | 0.844 [0.480, 1.227] | 0.915 [0.893, 0.936] | 0.744 [0.600, 0.894] |
  | in-domain clause (R@5) | HOLDS | NOT_SHOWN | NOT_SHOWN | HOLDS | HOLDS | (never trains) |
  | zero-shot clause (R@5) | HOLDS | HOLDS | HOLDS | HOLDS | HOLDS | HOLDS |

  - squad's gain over RRF is small (zspk +0.0045 [+0.0012, +0.0077]), so its share is a ratio of two small numbers.
    zrckp's mean is above zspk's there (0.9112 against 0.9099).
  - musique is the one the models can move: zspk gains more there than zrckp (0.580 against 0.573), and the share's
    lower bound needs to rise by 0.019.
  - Zero-shot (each dataset's leave-out fit; webqsp on J5), the MLP is never below the GNN by more than its in-domain
    gap. It is above it on metaqa held out (by 0.037) and on webqsp (by 0.070).
  - hit@1 is below 0.9 on hotpotqa and on webqsp zero-shot, as for the earlier pair.
  - The point estimates filed earlier (squad 1.289, musique 0.932) came from four-decimal means; these use the exact
    means.
- **On the full corpora (P3a, P3b),** the earlier pair: the lean MLP keeps at least 0.968 of zsp's gain over RRF on R@5 and FC@5 on all
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

**The final pool's build (P4a, [P4A_POOL_COLD.md](P4A_POOL_COLD.md)).** C12 timed the models on today's pool, which
FastPool builds in 0.07-0.33 ms. KIND_T's pool walks the graph first.
- **What it costs.** P4a timed the build cold, batch 1, one thread, on C12's 200 questions per dataset. Every one of
  the 1,200 pools equals P1c2's, bit for bit. The p50 is 0.19 ms on squad (no walk), 10.4 ms on metaqa, 61.3 ms on
  webqsp, 76.0 ms on musique, 198.9 ms on 2wiki and 286.2 ms on hotpotqa. The walks are 88-97% of it.
- **What it does to the ratios.** The build is the same for every model, so it pulls every MLP/GNN ratio toward 1. On
  hotpotqa and 2wiki it is 99% of each model's cold end to end, and every ratio is 1.00-1.02 there. gnn0 / zrc falls
  from 3.99 to 1.97 on musique, from 2.93 to 2.40 on metaqa and from 1.30 to 1.14 on webqsp. squad does not move.
- **Why the walks are slow.** An A1 walk on hotpotqa or 2wiki touches about 270,000-330,000 nodes and reads about a
  million union entries, at 67-142 ns each. The time goes to memory: random reads of each neighbour's state, and the
  reset after the walk.
- **The call is WALKS_DOMINATE, so next is P4b:** faster exact walks, the same pools bit for bit. Several-fold faster
  walks would still leave the build far above the models' 1.7-6 ms on hotpotqa and 2wiki. Whether Claim 3 is stated on
  the model stages, with the build beside them, or end to end, is the user's call.

**Faster exact walks (P4b, [P4B_FAST_WALKS.md](P4B_FAST_WALKS.md), filed 14 October 16:53): FASTER.** The same walks,
step for step, with their state laid out for the memory: one record per node for what each visit reads, the neighbour
lists merged, no reset pass, the nodes numbered breadth-first, and reads issued ahead.
- **Exact.** Both forms built every pool on 200 fresh questions per dataset, back to back. 5,322 walks were compared
  one by one, with the same order and counts every time, and both forms gave P1c2's pools bit for bit.
- **KIND_T's build, cold p50, P4a's form → P4b's:** metaqa 7.2 → 6.4 ms (1.14×), musique 48.9 → 34.4 (1.42×),
  webqsp 34.6 → 13.2 (2.62×), hotpotqa 171 → 48 (3.58×), 2wiki 188 → 50 (3.75×). The gain grows with the graph.
- **End to end, with C12's model stages:** on hotpotqa and 2wiki the build is still about 20 times zrc's whole end to
  end, and every ratio there is 1.00-1.08. gnn0 / zrc is 2.51 on musique, 2.58 on metaqa and 1.24 on webqsp.
- **A session difference:** P4a's form ran at 0.58-0.95 of its filed p50 in P4b's session. The ratio does not depend
  on that, because both forms ran side by side.
- **No exact form left cuts the walks' own work.** A question's walks are independent, so they could run on separate
  threads exactly. That changes C12's one-thread protocol, so it is the user's call, as is Claim 3's wording.

**The KT input's serving cost (P4c, [P4C_KT_SERVING.md](P4C_KT_SERVING.md), filed 14 October 20:56): TIMED.** The
lean pair now reads KIND_T's walk positions. Serving them means the walks the build skipped, the eight columns and the
head, timed cold beside the build on 200 fresh questions per dataset. Every check held on every question.
- **The cost, p50 per question:**
  - musique, hotpotqa and 2wiki: +4-7 ms (the build already runs four of the five walks, so only Kst is extra);
  - webqsp +32 ms and metaqa +39 ms (three to five extra walks);
  - squad +22 ms (no links, so the build walks nothing, but the columns need two walks).
  The columns and the head together are under 1.4 ms.
- **End to end with C12's model stages,** the MLP with the KT input takes 16× zrc's time on squad, 2.1× on metaqa,
  1.34× on webqsp and 1.06-1.16× on the other three.
- **The ratios:** zspk / zrck is 0.98-1.08, because both pay the same serve path. zrck is slower than gnn0 on squad,
  webqsp, hotpotqa and 2wiki. zrck's head is an upper bound for zrckp's.
- **A re-time:** the first runs were made while this session was still running commands. They were timed again with
  the session idle, by a rule committed before any number was read (8f7856f). The two agree within 3.3%.

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

**P1c2's screen, filed 14 October 15:05: NOT_ADOPTED** ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md) section 7). Both
bases were retrained on the final pools, one fit per family on L-musique and L-hotpotqa, and read against their own fits
on today's pools.
- zrc has 5 GAINs and no LOSS in 12 reads. zsp has 5 GAINs and one LOSS: webqsp zero-shot on L-musique's fit, −0.0167.
  The rule needs no LOSS on either arm, so there is no full run.
- metaqa (+0.020 to +0.021) and 2wiki (+0.039 to +0.045) gain on both arms and both fits. On 2wiki R@5 rises about
  twice as much as the pools' gold recall (+0.020): the models also rank better on these pools.
- On musique and webqsp the extra golds do not reach the top five. musique's pools add 0.042 of gold recall, and R@5
  moves by +0.001 to +0.006 in-domain and −0.032 to −0.045 zero-shot (WITHIN its wide seed null). webqsp's add 0.108,
  and R@5 moves by −0.017 to +0.034.
- So those two are RANKED gaps: the golds are in the pool and ranked out. Today's models on today's pools stay the
  claims' models until a model on the final pool beats them.

**P1c3's screen, filed 14 October 17:22: ADOPT_FOR_FULL_RUN** ([P1C3_KT_INPUT.md](P1C3_KT_INPUT.md) section 7). Both
lean models get one more input on KIND_T's pools: where KIND_T's construction put each pool row. That is eight columns
computed when the pool is built, from no gold, read through a 177-parameter head. **Correction (18:40):** the head's
ninth input is the row's score z-scored over its question's rows, so zrck's head reads the other rows' scores. That
breaks the MLP's 8 October input rule, so zrck serves no claim; P1c3p's pointwise head (zrckp) is the fix (below).
zspk is the GNN and keeps it.
- zrck (the MLP) and zspk (the GNN) each have 6 GAINs and no LOSS in 12 reads after the re-call.
- metaqa (+0.021 to +0.026) and 2wiki (+0.040 to +0.066) gain on both arms and both fits. webqsp zero-shot: zrck
  +0.037 and +0.082, zspk −0.002 and +0.042.
- Against P1c2's fits, the input wins back 0.61 and 0.69 of what the pools cost on musique zero-shot. zrck now reads
  above KIND_T's own order there (0.5059 against 0.5025). On webqsp zrck keeps 0.34 and 0.76 of the pools' extra
  golds (P1c2: 0.08 and 0.26).
- Beside, deciding nothing: hotpotqa's hit@1 falls on all four fits (−0.008 to −0.043, each interval below 0) while its
  R@5 rises a little, and musique zero-shot's FC@5 falls (−0.033 and −0.021).

**P1c3's full run, filed 14 October 19:41: ADOPT** ([P1C3_FULL_RUN.md](P1C3_FULL_RUN.md), Results). Four more fits per
arm, 36 reads each, re-graded under the seed null over every split.
- Neither arm has a LOSS. Each has 5 of 11 primary GAINs (J5 metaqa, 2wiki and webqsp; L-metaqa's metaqa; L-2wiki's
  2wiki): zrck 18 GAINs in all, zspk 16.
- 2wiki rises +0.040 to +0.066 on every split, and metaqa on every split. webqsp zero-shot rises on six splits for zrck
  and four for zspk.
- The filed grades' one LOSS each (L-musique's held-out musique read, −0.0176 and −0.0100) is WITHIN under its null
  floor of 0.0720.
- Beside: hotpotqa's held-out hit@1 falls about 0.04 for both; musique's hit@1 rises on all six splits.
- zrck serves no claim, so the lean pair waits on zrckp.

**P1c3p's screen, filed 14 October 19:55: ADOPT_FOR_FULL_RUN**
([P1C3P_POINTWISE_HEAD.md](P1C3P_POINTWISE_HEAD.md) section 7). zrckp is zrc plus a head that reads only the row's own
eight KT columns.
- 7 GAINs and no LOSS in 12 reads after the re-call: metaqa +0.024 and +0.025, 2wiki +0.061 and +0.062, webqsp
  zero-shot +0.024 and +0.061, musique in-domain +0.010.
- Beside, against zrck: level on ten of twelve reads, +0.022 on 2wiki in L-hotpotqa's fit, and below on webqsp
  zero-shot (it keeps 0.65 and 0.74 of zrck's gain there). hotpotqa's hit@1 still falls (−0.034 zero-shot).

**P1c3p's full run, filed 14 October 20:56: ADOPT** ([P1C3P_FULL_RUN.md](P1C3P_FULL_RUN.md), Results). Four more fits,
36 reads against zrc, re-graded under the seed null over every split.
- 5 of 11 primary GAINs (J5 metaqa, 2wiki and webqsp; L-metaqa's metaqa; L-2wiki's 2wiki), 20 GAINs in all, no LOSS.
- 2wiki rises +0.060 to +0.064 on every split. metaqa rises on every split, +0.108 held out (0.142 to 0.249). webqsp
  zero-shot rises on all six splits (+0.014 to +0.066). musique rises +0.006 to +0.010 in-domain and is −0.011 held out
  (WITHIN). hotpotqa and squad are within 0.008.
- Beside: hotpotqa's hit@1 falls on all six splits (−0.002 to −0.034), and musique's held-out FC@5 falls 0.026. Against
  zrck, zrckp is level on 31 of 36 reads.
- **So the lean pair is (zrckp, zspk),** both on KIND_T's pools with the KT input. Its serving cost is P4c's (section
  3). The universal pair (twin0, gnn0) is unchanged.

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
**P5b2, declared 16:45 (f8e9e41), filed 14 October 06:55: STOPS** ([P5B2_HELD_OUT_STOP.md](P5B2_HELD_OUT_STOP.md),
section 7). The same g, with its fit stopped at the epoch that best predicts held-out golds no training question had,
and every number split into golds the fit questions had (seen) and new ones.
- **The stop worked:** both fits kept an early epoch (4 and 3 of 30), where the loss on the fit's own questions equals
  its held-out loss. P5b's own g, read again beside, reproduces P5b's orders on every question.
- **The call.** Fused with H2, g finds more of the low golds on metaqa, hotpotqa (one point) and 2wiki, ties on squad,
  finds fewer on musique, and is mixed on webqsp (zero-shot).
- **Where g works, it is not memory.** On 2wiki and metaqa it finds about twice H2's share of the golds no fit
  question had (2wiki at 10: 0.316 against 0.156). They are golds two or more links from the seeds, which the rings
  cannot order.
- **The fit was not what held g back.** Stopped early, g reads the new golds about as P5b's 30-epoch g did. musique
  stays near chance (0.011 at 10, against H2's 0.159): its golds sit one link from the seeds, where H2's rings and
  cosine order them best, and a fixed fusion cannot back off where g knows nothing.
- **The learned offsets close in this form,** and P5c does not start. Any further reranker form gets its own file.

**P5b3, declared 07:45 (25eab88), filed 14 October 09:50: STOPS** ([P5B3_BACKOFF_GATE.md](P5B3_BACKOFF_GATE.md),
section 8). A learned back-off gate, for P5b2's diagnosis that a fixed fusion cannot back off.
- **The gate:** for each question, a weight λ between H2 and P5b2's g, with λ = sigmoid(b + w · z). Its inputs z are
  ten numbers known before any gold:
  - the pool's size and its ring shares;
  - g's entropy, and its probability mass by ring;
  - g's largest weight and its offset size;
  - how far g's top 25 and H2's top 25 agree.

  That makes 11 weights; g itself is unchanged.
- **The fit:** on P5b2's held-out fit questions (which g never trained on), and only on golds that no training question
  had. It maximizes the mean log of the gated rank score.
- **The call is P5b's,** for the gated order against H2 on s1eval: GOES_FURTHER if it finds more of the low golds on
  some dataset and fewer on none.
- **The result.** At weights ½, 1 and 0, the gated order gives P5b2's F, H2 and G on every question.
  - The gate finds more of the low golds than H2 on metaqa, hotpotqa, 2wiki and webqsp (zero-shot, where P5b2's fixed
    fusion was mixed). It ties on squad.
  - It still finds fewer on musique: 0.005 to 0.011 fewer, against F's 0.037 to 0.092.
- **The gate works as built.** It puts 0.32 on H2 on metaqa, where g reaches; there it finds about what g alone finds
  (0.076 at 10, against H2's 0.009). On musique, where g knows nothing, it puts 0.90 on H2.
- **Why musique still loses:** below a weight of 1, g's rank still moves nodes in H2's order. musique's low golds sit
  one link from the seeds, where H2's cosine orders them best and g's rank is noise.
- **This closed only the forms tested:** a fixed fusion and a gate, read without a ranking model. The audit of
  17 October ([VERDICT_AUDIT_2026_10_17.md](VERDICT_AUDIT_2026_10_17.md)) reopens the idea. Its next form is P5c:
  g's scores as a fixed input to both models, where a trained head can learn to ignore them on musique.

## 7. Running now (17 October, 14:10)

- **The pools: the user's target is met.** P1g and P1g2 are TARGET_MET (4eb23ff). P1g2's rule (1.5 times today's
  KIND_T fill plus the first 5,000 one-link neighbours by cosine, gated to the questions KIND_T expands) gives ALL ≥
  0.95 on every training dataset on s1sel at 1,976 rows on average (musique 0.885 → 0.951). webqsp's s1eval, beside,
  goes from 0.841 to 0.933.
- **P1g-b** ([P1GB_POOLS_REFIT.md](P1GB_POOLS_REFIT.md)) on the host: those pools built with their looks, caches,
  chains, links and KT input, then the lean pair refit on them. Five datasets' pools are built. metaqa's pool build
  failed at 04:27 on a Windows file lock (WinError 32 on its last rename). The feeder dropped its 128 dependent items,
  and they were re-queued at 09:09 with the build under retries. Its full run
  ([P1GB_FULL_RUN.md](P1GB_FULL_RUN.md), 9bc6e41) is queued behind its screen and gated on it.
- **On the card:** A2 ([A2_RANKS_BESIDE_Z.md](A2_RANKS_BESIDE_Z.md)), pool ranks and top-50 z-scores beside zret's
  z-scores in both models. Queued behind it, A3 ([A3_VSHARE_LEAN.md](A3_VSHARE_LEAN.md)): vshare's share
  augmentation of the pool statistics, moved into zret's reference, in both models' training; then A4
  ([A4_BLOCK_DROPOUT_LEAN.md](A4_BLOCK_DROPOUT_LEAN.md)), bdrop20's block dropout at the top of both models; then A5
  ([A5_KIND_HEAD_BACKOFF.md](A5_KIND_HEAD_BACKOFF.md)), zck's kind head on typed graphs with its gradient kept off the
  shared weights; then A6 ([A6_LINK_WORDS_INPUT.md](A6_LINK_WORDS_INPUT.md)), D4's link words as six fixed columns
  in both models, after spaCy's mentions are built on all six corpora on the CPU; then A7
  ([A7_S6_SCORER_INPUT.md](A7_S6_SCORER_INPUT.md)), S6's chain scorer refit per split and cross-fitted, as four
  fixed columns in both models; then A8 ([A8_QDEPTH_INPUT.md](A8_QDEPTH_INPUT.md)), round 11's depth prior fitted on
  its own per split and cross-fitted, as six fixed columns in both models; then A9
  ([A9_EDGE_FAMILY_DROPOUT.md](A9_EDGE_FAMILY_DROPOUT.md)), the MP-unified track's edge-family dropout on zspx alone
  (GNN only); then A10 ([A10_G2_ON_ZSPX.md](A10_G2_ON_ZSPX.md)), G2's hidden-state layers over zspx's score
  (GNN only); then A11 ([A11_FOURTH_FAMILY_LINKS.md](A11_FOURTH_FAMILY_LINKS.md)), our LLM-free phrase links as a
  fourth link family in zspk's step (GNN only); then A12 ([A12_GSURG_GNN.md](A12_GSURG_GNN.md)), zspk trained with
  gradient surgery (GNN only). P5c's builds
  ([P5C_OFFSET_INPUT.md](P5C_OFFSET_INPUT.md)) are on the CPU.
- **Filed on 17 October:**
  - **R1: zrckpr ADOPT_FOR_FULL_RUN, zspkr NOT_ADOPTED** ([R1_RESIDUAL_INPUT.md](R1_RESIDUAL_INPUT.md)): the bridge
    residual as nine fixed columns. The MLP gains webqsp zero-shot (+0.0135, L-musique) with no LOSS. The GNN moves
    nothing past the floor. musique in domain: MLP +0.0034, GNN +0.0011, both within the floor. Next forms:
    zrckpr's refit and full run on P1g-b's pools, with the columns beside zspx's inputs too; the audit's item 13
    (G1b's bridge columns on zspk).
  - **G5 ADOPT_FOR_FULL_RUN** ([G5_COMBINED_GNN.md](G5_COMBINED_GNN.md)): zspx (zspk with its step end to end, G3's
    residual message and G4's second step) has 2 GAIN and no LOSS. musique in-domain rises +0.0147 (L-hotpotqa),
    more than any single change. webqsp zero-shot rises +0.0235 (L-hotpotqa), where no single change was a GAIN.
    zspx is the GNN that goes on to P1g-b's pools, in its own declared full run.
  - **G3 ADOPT_FOR_FULL_RUN** ([G3_BRIDGE_RESIDUAL.md](G3_BRIDGE_RESIDUAL.md)): the bridge-residual message lifts
    musique in-domain +0.0096 (L-hotpotqa) and zero-shot +0.0114 (L-musique, within its wide floor), with no LOSS.
  - **G4: both arms ADOPT_FOR_FULL_RUN** ([G4_DEEPER_PROPAGATION.md](G4_DEEPER_PROPAGATION.md)): a second
    propagation step (+0.0115) and the step trained end to end (+0.0093), both on musique in-domain, with no LOSS.
  - **E1 NOT_ADOPTED** ([E1_LATENT_CHAIN_EM.md](E1_LATENT_CHAIN_EM.md)): EM over the chain head's latent chains moves
    nothing on the MLP and costs the GNN 2wiki (−0.019).
  - **T1 NOT_ADOPTED** ([T1_GRANULARITY_TRANSFORMS.md](T1_GRANULARITY_TRANSFORMS.md)): S6's transforms on metaqa's
    fit carve lower the MLP's webqsp zero-shot (−0.015, −0.034), the read they were meant to lift.
  - **L1 and L3 NOT_ADOPTED** (4eb23ff): zsep's loss on the lean pair moves nothing; zck's offsets on a frozen base
    cost webqsp zero-shot once.
- **What the GNN results say:** musique's ranking moves when the GNN's messages carry more than one hop's scores or
  what a neighbour adds to the question, not when it has more layers of the same message (G2). The MLP has no
  messages, so musique's Claim 1 share will fall unless the MLP's inputs catch up.
- **Filed on 15 October:**
  - **G2 NOT_ADOPTED** ([G2_HIDDEN_STATE_GNN.md](G2_HIDDEN_STATE_GNN.md)): three gated hidden-state message-passing
    layers on zspk add nothing once training luck is counted (0 GAIN, 0 LOSS of 12).
  - **M1 RUN** (059d090): musique's missed golds sit one link from found ones, and zspk lifts them no more than the MLP.
  - **P1f NONE_MEETS** ([P1F_POOL_SIZE_RULE.md](P1F_POOL_SIZE_RULE.md)): no size rule over KIND_T keeps today's rows
    and every dataset's floor; metaqa and musique bind.
  - **The B1 re-call STANDS** (b19edb5): all 60 adoption decisions reproduce exactly without B1's 3,000 questions.
  - **D5 RUN** ([D5_GOLD_REACH.md](D5_GOLD_REACH.md)): musique's missed golds are close but buried (99.5% of
    questions have every gold within two links, but a missed one-link gold ranks about 2,350th of 6,456 neighbours by
    first-stage weight); metaqa's are its 3-hop answers.
- **Done since the last update:**
  - P1c3p's full run filed (section 5): ADOPT. The lean pair is (zrckp, zspk).
  - P4c filed (section 3): TIMED, after a re-time with the session idle.
  - P3c filed (section 2): Claim 1 for the new pair with intervals, CLAIM1_NOT_ALL (squad and musique NOT_SHOWN
    in-domain; zero-shot HOLDS on all six).

## 8. Open gaps

| gap | the number | what works on it |
| --- | --- | --- |
| Pools against SOTA (full corpora) | P1e's KIND_T, parameter-free, the final pool: at or above the best SOTA order on squad, musique, 2wiki and webqsp; below on metaqa (up to −0.005) and hotpotqa (up to −0.007). Like-for-like, webqsp is below A1 on the structural family (up to −0.051). | Closed. metaqa and webqsp pull the entity branch in opposite directions, so no further fixed rule is declared. The learned gates stay beside. |
| SOTA rankers (full corpora) | Both GNNs at or above the best LLM-free SOTA ranking on all six (P3a). Where the bar wins, the gold was in our pool and ranked out (P3b). | P5's representation. A seeds-removed read on more than 17 and 12 questions needs P3a's walk rows kept past ten nodes. |
| What message passing buys | GNN>MLP is golds one link from a found gold on hotpotqa and 2wiki (P3b): lean 0.009-0.016 of R@5, universal 0.034-0.051 | P5b and P5b2 STOP: the learned offset g(question, seed, graph features) finds more of these golds on 2wiki, metaqa and hotpotqa, golds no fit question had among them, but fewer on musique (near chance, with or without a held-out stop) and mixed on webqsp zero-shot (section 6). P5a's fixed map STOPS. P5b3's per-question back-off between H2 and g STOPS too: more on four datasets, still fewer on musique (0.005 to 0.011). The learned offsets close. |
| Golds the final pool adds | P1c2: about two thirds reach the top five on metaqa, and 2wiki's R@5 rises twice the pools' recall gain; on musique and webqsp little or none does (zsp −0.017 on webqsp zero-shot) | P1c3's walk positions as an input, adopted for the lean pair (zrckp, zspk): webqsp zero-shot rises on all six splits (+0.014 to +0.066 for zrckp), metaqa +0.108 held out, 2wiki +0.060 to +0.064 everywhere. Still open: musique's held-out read (−0.011, WITHIN; FC@5 −0.026) and hotpotqa's hit@1 (−0.002 to −0.034 on all six splits). |
| musique ranking | 66.3 vs HippoRAG 2's 74.7; the pools hold 98.2% of golds; 752 of 990 misses are on three- and four-hop questions. On the full corpus (P3b): ranked out 0.23-0.28 of R@5, one link from a found gold 0.11-0.13, outside the pool 0.079 (0.198 on 4-hop). | D4 (link words): NO_SIGNAL. P2c part 1: on B1 musique our LLM-free graph adds more than the LLM-built graphs, so the LLM graph is not what our walks lack. |
| B1's questions inside s1eval | 1,000 each of musique's 2,417, hotpotqa's 7,405 and 2wiki's 12,576 s1eval questions are B1's (found 14 October). No fit trained on them, but every screen and full run read them when it decided, as did earlier gates on the whole dev splits; the claims doc's even/odd split was never applied. | **Re-called 15 October (b19edb5): STANDS.** All 60 adoption decisions reproduce exactly without B1's 3,000 questions, and none of the 17 lineage verdicts changes. B1's ids stay out of the confirmation run's held-out questions. |
| Hyperlink-free hotpotqa | 95.0 vs 96.3 | U1c and U1f NOT_ADOPTED: retraining on the universal graph costs hotpotqa 0.035-0.065 R@5. Open. |
| Link precision | Hyperlink F1 0.40 (2wiki), 0.47 (hotpotqa) | A universal rule; one file per change. |
| Pool misses | One hop on the KBs and musique; first stage on hotpotqa and squad | Step 4h covers the one-hop misses; the first-stage misses are open. |
| Zero-shot to webqsp | zsp 26.8 zero-shot; a model trained on webqsp reached about 60 (S6) | Longer training lifts the MLP's transfer, but under seed noise (round 33). |
| Training data | At equal steps, half the data loses 1 read of 12 (round 33's step control) | Not the lever: no second fit carve, by D3's rule. |
| MLP vs the fastest GNN | zsp is the fastest GNN on all six; zrc is 0.96-1.18× faster, twin0 slower except on webqsp (1.14× faster) | Not met. With every input kept, the forwards keep each MLP below 2× on five datasets (section 3). With the final pool's build, every ratio on hotpotqa and 2wiki is 1.00-1.02 end to end (P4a), and 1.00-1.08 with P4b's faster walks. How Claim 3 is worded is open. |
| Universal speed | gnn0 / twin0 1.46-1.89× (C12) | P4a: the final pool's build (0.19-286 ms) is shared and pulls every ratio toward 1. P4b FASTER: the build 1.14-3.75× faster, the same pools. C13 (exact per-family prototypes and depth basis) is parked: it reaches 2× on musique only; squad, metaqa and webqsp need the compile 3.0-3.7× faster. |
| Lean speed | zsp / zrc 0.96-1.18× (C12); zspk / zrck 0.98-1.08× with the KT input (P4c) | Bounded by the lean design: 1.16-1.34 even with free shared stages, 2.12 on musique. The KT input's extra walks (4-39 ms) are shared by both models. |

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
  - P5b2 stopped g's fit on held-out golds no fit question had. g's gains on 2wiki and metaqa hold on those new golds,
    musique stays near chance, and it STOPS: the learned offsets close in this form.
- **14 October: one more reranker form.** P5b3 lets a learned gate set, per question, how much the fusion trusts g
  and how much H2. It keeps g's gains and finds more on webqsp too, but musique still loses a little, and it STOPS:
  the learned offsets close.
- **14 October: the final pool's cost.** P4a timed KIND_T's build cold. The walks cost 10-286 ms per question, and on
  hotpotqa and 2wiki the build is 99% of every model's end to end. Next is P4b: the same walks, exact, and faster.
- **14 October: the models on the final pool.** P1c2 retrained zrc and zsp on KIND_T's pools: NOT_ADOPTED. Both gain on
  metaqa and 2wiki, but zsp loses webqsp zero-shot, and on musique and webqsp the extra golds are ranked out.
- **14 October: faster walks, and the walk positions as an input.** P4b made KIND_T's walks 1.14-3.75× faster,
  exactly. P1c3 gave both lean models KIND_T's walk positions as one more input. Its screen passes with 6 GAINs and no
  LOSS on each, and its full run decides.
- **14 October, evening: the walk positions adopted, and the MLP rule kept.** P1c3's full run is ADOPT: no LOSS in 36
  reads on either arm. The user caught that zrck's head reads other rows' scores through a per-question z-score, so
  zrck serves no claim. P1c3p's pointwise head (zrckp) passed its screen with 7 GAINs and no LOSS. Its full run is
  ADOPT (no LOSS in 36 reads), so the lean pair is (zrckp, zspk). P4c timed the input's serving cost: 4-7 ms per
  question where KIND_T already walks, 22-39 ms where it does not.

- **15 October: deeper message passing, and where the pools lose golds.** G2 gave zspk three gated layers that
  pass hidden states: NOT_ADOPTED, no read moves beyond training luck. M1 and D5 found why the models' misses resist:
  on musique the missing golds are one or two links from the base but buried among thousands of neighbours, and in
  neither first-stage list. P1f found no size rule that keeps today's rows. The B1 re-call stands. D5b measures
  neighbour orders that read the parent passage, for one rule that lifts every dataset's ALL to 0.95 (P1g).

## What is next

The order is [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md)'s, under the user's rule of 10:40. Each stage gets its own
file, committed before its numbers.
1. **The pool, with nothing learned: final.** P1e's KIND_T (NOT_ALL_SIX: metaqa up to −0.005, hotpotqa up to
   −0.007). The links stay the three families: P2c part 2's fourth family is NO_GAIN.
2. **The models on the final pools.** P1c2 ([P1C2_POOLS_KIND_T.md](P1C2_POOLS_KIND_T.md)) retrained both bases on
   KIND_T's pools at today's sizes: **NOT_ADOPTED**, with no full run. Today's models stay the claims' models. Its
   record starts P3's ledger: on musique and webqsp the pools' extra golds are ranked out. P1c3
   ([P1C3_KT_INPUT.md](P1C3_KT_INPUT.md)) adds KIND_T's walk positions as one input to both lean models. Its full run
   ([P1C3_FULL_RUN.md](P1C3_FULL_RUN.md)) is **ADOPT**, but zrck's head breaks the MLP rule. P1c3p's pointwise head
   (zrckp) is **ADOPT** ([P1C3P_FULL_RUN.md](P1C3P_FULL_RUN.md)), so the lean pair is (zrckp, zspk). P4c timed its
   serving cost: **TIMED**. **Next here, each in its own file:**
   - done: Claim 1's share for (zrckp, zspk) with question-bootstrap intervals (P3c): **CLAIM1_NOT_ALL**, squad and
     musique NOT_SHOWN in-domain, zero-shot HOLDS on all six (section 2);
   - P3's ledger on the new pair: hotpotqa's hit@1 (a fresh hit@1-term screen), RRF5's position as a ninth KT
     column, and zck stacked on a frozen base.
3. **Rerankers on the final pools:** the learned offset g(question, seed, graph features), fit by EM over the latent
   seed and offset, aimed at the BRIDGE part of GNN>MLP and at the RANKED misses. P5b and P5b2 STOP: g finds more on
   2wiki and metaqa, golds no fit question had among them, but adds nothing on musique, and a fixed fusion cannot back
   off there. P5b3 ([P5B3_BACKOFF_GATE.md](P5B3_BACKOFF_GATE.md)), a learned per-question back-off between H2 and
   g, STOPS too: it finds more on four datasets, but still fewer on musique. **Closed:** the learned offsets close,
   and P5c does not start.
4. **P4: speed.** 2-3× in every family with every win kept. P4a ([P4A_POOL_COLD.md](P4A_POOL_COLD.md)) timed the
   final pool's build: WALKS_DOMINATE. P4b ([P4B_FAST_WALKS.md](P4B_FAST_WALKS.md)) made the walks 1.14-3.75×
   faster, exactly: FASTER. P4c ([P4C_KT_SERVING.md](P4C_KT_SERVING.md)) timed the KT input's serving cost: TIMED
   (4-39 ms per question, mostly walks). Running a question's walks on separate threads would change C12's
   one-thread protocol, so it is the user's call.
5. **Claim 3's wording and the confirmation run** of the chosen MLP and GNN on held-out questions. B1 is read beside
   it and never used to choose. B1's questions sit inside s1eval (section 8), so the held-out questions should leave
   them out, and the adoption decisions should first be re-called without them.
