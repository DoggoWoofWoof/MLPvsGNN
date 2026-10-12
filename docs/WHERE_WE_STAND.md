# Where we stand, 13 October 2026 (01:00)

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
| 3 | The MLP is faster, cold first. | Both MLPs are faster than the universal GNN on all six datasets (twin0 1.42-1.85×; zrc 2.49-4.15×, webqsp 1.32×). The target is at least 2× faster than the fastest GNN, which is zsp on all six, and no MLP is there yet: zrc is 0.95-1.18× (section 3). |

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
fastest form that keeps its scores (within 1e-4 and the same top 5) and reads every one of its inputs. Laptop, pinned,
one thread. Records: [C11_BOTH_FAMILIES_FAST.md](C11_BOTH_FAMILIES_FAST.md) (intervals there) and
[C10_ZN_COLD.md](C10_ZN_COLD.md).

**Cold p50 per question, ms (C11):**

| dataset | twin0 (universal MLP) | gnn0 (universal GNN) | zrc (lean MLP) | zsp (lean GNN) |
| --- | ---: | ---: | ---: | ---: |
| 2wiki | 4.71 | 6.76 | 2.72 | 2.76 |
| hotpotqa | 4.35 | 6.18 | 2.41 | 2.52 |
| squad | 2.95 | 4.61 | 1.81 | 1.90 |
| musique | 83.26 | 154.13 | 37.16 | 43.71 |
| metaqa | 57.24 | 88.42 | 29.77 | 32.18 |
| webqsp | 77.79 | 117.33 | 88.66 | 84.54 |

**GNN time over MLP time, cold:**

| dataset | universal: gnn0 / twin0 | lean: zsp / zrc | gnn0 / zrc | zsp / zn (C10) |
| --- | --- | --- | --- | --- |
| 2wiki | 1.44 | 1.01 | 2.49 | 5.22 |
| hotpotqa | 1.42 | 1.04 | 2.56 | 2.65 |
| squad | 1.56 | 1.05 | 2.54 | 2.71 |
| musique | 1.85 | 1.18 | 4.15 | 5.31 |
| metaqa | 1.54 | 1.08 | 2.97 | 3.95 |
| webqsp | 1.51 | 0.95 | 1.32 | 9.61 |

- **Both MLPs are faster than the universal GNN on all six datasets.**
  - The universal MLP is 1.42-1.85× faster.
  - The lean MLP is 2.49-4.15× faster on five datasets. On webqsp it is 1.32× faster, because its chain entries are its
    largest stage.
- **No MLP is yet 2× faster than the fastest GNN.** The fastest GNN form is zsp's (zsp10) on all six datasets. zrc is
  0.95-1.18× faster than it. twin0 is slower than it (zsp10 / twin0 0.52-0.64, 1.09 on webqsp).
- **Within a family, the MLP does not reach 2×.**
  - **Universal.** The compile both models run is 60-74% of twin0's time. Even if it cost nothing, the ratio would top
    out at 2.04 (2wiki) to 4.20 (musique).
  - **Lean.** zsp is zrc plus its links and one propagation step over scores, which cost 3-17% of zrc's time. Even if
    everything before the forwards cost nothing (the compile, the lean inputs and the chain entries), zsp / zrc would
    be 1.17-1.33 on five datasets and 2.13 on musique. These figures are worked out from C11's stage p50s, not timed.
- **zn**, the lean MLP with every edge-derived input removed, is 2.65-9.61× faster than zsp. It keeps only 13-73% of
  the gain, so it is a ceiling, not a model.
- **C11's forms made every model faster at the same scores:** twin0 by 1.32-1.52×, gnn0 by 1.16-1.38×, zrc by
  1.26-1.64× and zsp by 1.28-1.74×.
- **Next, C12: the shared compile made faster for both models of each family.**
  - A development profile on the 2,000-row pools puts most of musique's 66 ms compile in three stages: the per-family
    neighbour prototypes (24 ms), the depth basis (16) and the seed prototypes (12).
  - For 2× in the universal family, the compile must get 1.26× faster on musique and about 3.2× faster on squad,
    metaqa and webqsp.
  - On 2wiki and hotpotqa, the forwards keep the ratio near 2 even with a free compile.

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

By U1f's rule, no full run uses either graph, and both bases keep today's graph and pools. The hyperlink price, what
the trained models lose without the hyperlinks on the eval carves, goes to the gap ledger. For both arms it is
0.035-0.065 R@5 on hotpotqa and 0.010-0.020 on 2wiki. The rule's links recover the hyperlinks as edges, but the
models do not get the hyperlinks' value back from them.

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

Today's models were trained and read on step 1's frozen pools and today's graph. U1c and U1f tried both bases on the
universal graph with step 4h's pools. Neither was adopted, so the bases stay where they were.

## 6. Running now (13 October, 01:00)

- **D4, the words of a link** ([DIAG_LINK_WORDS.md](DIAG_LINK_WORDS.md)), a diagnostic declared before its numbers.
  - The question: on musique, do the words around a link (a title mention, a KB relation name, a shared entity) pick
    out the neighbour a question follows, where zrc's score ranks it 5th to 9th?
  - It reads zrc's two card fits on CPU. The NER mentions were built on the laptop and rebuild every NER edge of
    musique and squad exactly.
  - The first run stopped at musique's read, after metaqa and squad had passed identity. Clean-up 13 had removed step
    1's musique look chunks, which look_step1.py rebuilds bit for bit.
  - They are being rebuilt now, about 20 minutes. D4 then runs again unchanged, about 25 minutes.
- **What D4 decides, fixed before any number.**
  - SIGNAL declares round 34: the link words as an input of both families, screened on two fits under the seed null.
    For the MLP they are computed before any row is scored, so there is no message passing.
  - NO_SIGNAL reports musique's gap as the first stage's: 13.3 points below NV-Embed-v2 at the start.
- **Host disk: 30.5 GB free.** Clean-up 15, U1c's and U1f's arrays, is listed and dry-run: about 143 GB once both
  names of each hard-linked pair go. It is the user's command.

## 7. Open gaps

| gap | the number | what works on it |
| --- | --- | --- |
| musique ranking | 66.3 vs HippoRAG 2's 74.7; the pools hold 98.2% of golds; 752 of 990 misses are on three- and four-hop questions | D4 (link words); round 34 if SIGNAL. Rounds 29-32 and U1c moved nothing here. |
| Hyperlink-free hotpotqa | 95.0 vs 96.3 | U1c and U1f NOT_ADOPTED: retraining on the universal graph costs hotpotqa 0.035-0.065 R@5. Open. |
| Link precision | Hyperlink F1 0.40 (2wiki), 0.47 (hotpotqa) | A universal rule; one file per change. |
| Pool misses | One hop on the KBs and musique; first stage on hotpotqa and squad | Step 4h covers the one-hop misses; the first-stage misses are open. |
| Zero-shot to webqsp | zsp 26.8 zero-shot; a model trained on webqsp reached about 60 (S6) | Longer training lifts the MLP's transfer, but under seed noise (round 33). |
| Training data | DATA_LIMITED | No new data source declared. |
| MLP vs the fastest GNN | zsp10 is the fastest GNN on all six; zrc is 0.95-1.18× faster, twin0 slower | The 2× target is not met. Every input stays, so this is open. |
| Universal speed | gnn0 / twin0 1.42-1.85× (C11) | C12: a faster compile for both models; it is 60-74% of twin0's time. |
| Lean speed | zsp / zrc 0.95-1.18× (C11) | Bounded by the lean design: 1.17-1.33 even with free shared stages, 2.13 on musique. |

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
  adopted. D4 asks whether a link's words can close musique's ranking gap.

## What is next

1. **D4's verdict.** Round 34 is declared if it is SIGNAL: the link words for both families, one universal change
   graded on all six datasets.
2. **C12**: the compile both models of a family share, made faster for both alike, then timed as C11 was.
3. **The confirmation run** of the chosen MLP and GNN on held-out questions, for the paper, with B1 read beside it
   (never used to choose).
