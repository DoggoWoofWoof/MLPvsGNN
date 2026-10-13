# P3a: the state-of-the-art rankers on our full corpora, beside our four models (a diagnostic)

Declared 13 October 2026, about 08:10, before any number of the stage. P3's first stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md). It reads rankings only: no model trains, and nothing is adopted.

## 1. Question

On all six full corpora, does each family's GNN rank at or above the best LLM-free state-of-the-art ranker? The
comparison uses the models' own measures (R@5, FC@5, hit@1), on the same questions and the same gold sets. That is
Claim 2 read on our corpora.

- **Where Claim 2 has been read so far.** Only on B1, HippoRAG 2's 6-12k-passage corpora (B1b, 3b033f6). Since
  13 October, B1 is a calibration line, not the target.
- **What P1a did not read.** P1a compared the same SOTA constructions with our pools, as coverage by pool size. It never
  read them as rankers at the top five, where the models are graded.
- **What P3b needs.** P3b's gap ledger needs a SOTA column on the same questions.

## 2. The rows

**The SOTA rows (the bar)** are P1a's LLM-free orders read as rankings, plus HippoRAG 2's two restarts walked on all
three link families:

| row | what | whose |
| --- | --- | --- |
| A1 | HippoRAG 2's restart rule on the regime's graph: seeds at half the mass, the RRF-weighted dense and SPLADE top-1000 lists at the other half | SOTA (HippoRAG 2) |
| A2 | A1 with HippoRAG's node specificity | SOTA (HippoRAG) |
| A1_all, A2_all | A1 and A2 walked on all three link families (structural, ner, knn), with that graph's degrees | SOTA (HippoRAG 2's rule on our fullest graph) |
| LINKED | uniform over our exact-match linked nodes (step 4e's linking, PullNet's rule) | SOTA (the KB-QA line, from our linking) |
| DENSE, SPLADE | the first-stage top-1000 lists | first stage |
| RRF | the equal-RRF order of the two lists | first stage |

**Beside, not the bar:**

| row | what |
| --- | --- |
| ORACLE | uniform over the **assigned** topic entities, on metaqa and webqsp only: the KB-QA line's exposure, never our input |
| A3, A0 | our pools' own walks (step 4e's restart; today's seeds only), read as rankers |

**Our four models, as filed:**

| row | family | where its scores come from | webqsp |
| --- | --- | --- | --- |
| zrc | lean MLP | the J5 fit, read by P2a part 2 with every edge kept (equal to the filed J5 read) | zero-shot |
| zsp | lean GNN | as zrc | zero-shot |
| twin0 | universal MLP | step 1's s1eval cache, score column 0 (its means equal step 1's grade) | in-domain: the pair was fit on all six fit carves |
| gnn0 | universal GNN | step 1's s1eval cache, score column 1 | in-domain |

- **How each row ranks.** A walk ranks nodes by mass, with step 4e's kernel (its alpha and eps), cut at 7,000 rows,
  nothing excluded. A list ranks in its own order. A model ranks the rows of today's pools (step 1's s1eval looks) by its
  score, in lean_gpu's order: descending, NaN last, ties by pool position. A gold outside a model's pool, or outside an
  order's rows, is missed.
- **Why A1_all and A2_all.** HippoRAG 2 links passages through the entities they mention and through synonym edges, not
  through hyperlinks.
  - On 2wiki and hotpotqa, the regime graph P1a walks is the structural family alone, and there that family is the
    hyperlinks.
  - ner (shared named entities) and knn (nearest neighbours under the frozen encoder) are the closest LLM-free
    counterparts we have.
  - P2a part 1 found that adding them costs reach on 2wiki and hotpotqa, but the bar should not rest on that: it takes
    the better of the two graphs.
  - On musique and webqsp the regime already walks all three families, so A1_all and A2_all equal A1 and A2 there. They
    are recomputed, not assumed.
  - On squad, P1a's walks are its RRF order. A1_all and A2_all walk squad's three families too.
- **squad and the KB.** LINKED is empty on squad (no linked node), as in P1a. A question with no linked node gets an
  empty LINKED order and misses every gold there.
- **Not run.** This is the user's decision 1 in [PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), and its default stands:
  - the published systems as released (HippoRAG 2 with its LLM-built graph and its encoder, GFM-RAG, G-reasoner);
  - the KB-QA line's trained reasoners (ReaRev, NuTrea, GNN-RAG).

  Their published numbers stay calibration lines, with their exposure named.

## 3. Questions and measures

- **Questions:** every s1eval question of the six datasets. These are the questions P1a, step 1's grade and P2a part 2
  read:

  | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
  | ---: | ---: | ---: | ---: | ---: | ---: |
  | 9,785 | 11,873 | 2,417 | 7,405 | 12,576 | 1,503 |

- **Golds:** the carve's gold set per question, the same for every row. Its count equals the models' gold_total.
- **Measures (lean_gpu's metrics_of, the models' grade):**
  - R@5: the share of the question's golds among the ranking's first five;
  - FC@5: 1 when all of them are there;
  - hit@1: 1 when the first is a gold.

  Each is a mean over questions.

## 4. The checks and the call, fixed before any number

**The checks come first.** The verdict refuses, and nothing is compared, unless:
- every P1a order recomputed here equals P1a's filed s1eval curve at every N (ALL, recall and the mean pool);
- A1_all equals P2a part 1's filed A1 on the three families (subset SNK) at every N;
- the universal pair's means (R@5, FC@5, hit@1, to six decimals) equal step 1's grade (the J5 read's refs);
- the lean pair's R@5 (to four decimals) equals the filed J5 reads;
- every row reads the same questions, with the same gold counts;
- the six parts ran on one script and one freeze, none with `--limit`.

**The bar.** Per dataset and measure, the bar is the best SOTA row: the highest mean of the eight (A1, A2, LINKED,
DENSE, SPLADE, RRF, A1_all, A2_all), with ties going to the earlier row in that list. Because the bar takes the best of
eight rows per measure, it is a little stricter than any one system, which errs against our claim.

**Per model and dataset.** For each measure, take the per-question difference to the bar and a paired bootstrap of its
mean (2,000 resamples, seed 0, the 95% interval). Each measure scores:
- **+1** when the mean is more than 0.002 above and the interval is above 0;
- **−1** when the mean is more than 0.002 below and the interval is below 0;
- **0** otherwise.

The model is **AT_OR_ABOVE** on the dataset if no measure scores −1, and **BELOW** otherwise. The label is ABOVE, BELOW,
MIXED or TIE, as in P2a part 2.

**The call (Claim 2 on our corpora),** for each family's GNN (zsp in the lean family, gnn0 in the universal):
- **SOTA_GNN** if it is AT_OR_ABOVE on all six datasets;
- **NOT_YET** otherwise. The record names each dataset and measure where it is below the bar, and by how much.

**Reported beside, deciding nothing:**
- the MLPs (zrc, twin0) against the bar, scored the same way;
- per family, dataset and measure, the MLP's share of the GNN's gain over RRF, where that gain is more than 0.002. This
  is Claim 1's quantity on our corpora;
- ORACLE on metaqa and webqsp (the exposure), and A3 and A0;
- every row's means on all three measures.

## 5. What follows

- **P3b, the ledger,** in its own file. Per dataset and hop slice, it takes each gold the bar ranks in its top five that a
  model does not, and the reverse, and sorts it into PROGRAM_2026_10_09's four buckets:
  - not in the pool;
  - in the pool and reachable from a found gold, but not ranked;
  - beyond the features;
  - top rank only.

  P3b reads this stage's records: each gold's position in every order, and every order's first ten nodes. Nothing is
  walked again.
- **NOT_YET:** the shortfalls set the bar that the next model stages (P5) must clear, on these rows.
- **SOTA_GNN:** Claim 2 holds on our corpora for that family, as a development read. The paper's confirmation run
  follows.
- **No model changes here.**

Flagged: the SOTA walks propagate over the graph. They are rows to beat, never inputs to either model.

## 6. How it runs

- **Code:** `scripts/p3a_sota_rankers.py`. It imports, unchanged:
  - P1a's orders (its walks, lists, topic entities, positions and curves);
  - P2a's family stores, step 4b's subset union and step 4e's carve runner.

  P1a's orders are built by the same calls P1a's coverage stage makes.
- **The self-test** covers:
  - the measures from gold positions, against hand-worked questions;
  - the numpy ranking of the universal pair's scores against lean_gpu.top_hit, on 200 made-up questions with ties, NaN
    and signed zeros;
  - the comparison's labels, calls and shares on made-up rows;
  - the verdict's checks on made-up records. It refuses a P1a curve that differs, a P2a A1 curve that differs, means that
    differ from step 1's grade or the J5 reads, and a gold count that differs;
  - a part written and read back gives the same rows.
- **Development runs, no number.** `--limit Q` reads the first Q questions and writes nothing. It logs only that every
  order was built, and that the questions and gold counts are those of step 1's cache and P2a's reads. Before this file
  was committed:
  - the self-test passed;
  - `read --limit 300` ran on musique, metaqa and squad on the laptop. Every order was built, the questions and gold
    counts matched both, and nothing was written.
- **Host jobs** (light lane). They are sized from P2b's gathers, which open the same two unions (2wiki peaked at 5.5 GB on
  8 CPUs, hotpotqa at 4.5 GB), plus about 0.6 GB for one batch of eleven 7,000-node orders:

  | item | what | where |
  | --- | --- | --- |
  | `p3a-read-sota-<dataset>` (six) | the orders, each gold's positions, each order's first ten, and the universal pair's counts, on s1eval | 2wiki 8 CPUs, 7 GB; hotpotqa 6, 6 GB; metaqa 4, 3 GB; musique 3, 3 GB; squad 3, 2 GB; webqsp 3, 3 GB |
  | `verdict` | the checks and the call | locally, on the fetched records |

  Records go to `outputs/p3a/`: `orders_<dataset>.json` (curves and timings) and `orders_<dataset>.npz` (compressed:
  positions, first-ten lists and lengths, about 10 MB on 2wiki).
- **ETA:**
  - P1a's nine orders took 863 s on 2wiki's s1eval at 5 threads;
  - P2a part 1's A1 and A3 on the three families took 498 s at 8 threads;
  - so 2wiki should take about 20-25 minutes with loading.

  The six run side by side, so the call should land about half an hour after the start, if the host's CPUs are free.
- The encoder stays frozen. No model trains or reads, nothing is adopted, and test splits are never read.

## 7. Results (filed 13 October, about 09:00)

Records: `outputs/p3a/`. These are orders_<dataset>.json and .npz, verdict.json and verdict.md.
- **The checks held on all six:**
  - P1a's curves;
  - P2a part 1's A1 on the three families;
  - step 1's grade for twin0 and gnn0;
  - the J5 reads for zrc and zsp;
  - the questions and gold counts;
  - one script and one freeze, and no part ran with `--limit`.

**The call: zsp SOTA_GNN, gnn0 SOTA_GNN.** Both GNNs are AT_OR_ABOVE the best LLM-free SOTA row on all six full
corpora, on R@5, FC@5 and hit@1. On our corpora, against these rows, Claim 2 holds as a development read.

**Means on every s1eval question (R@5 / FC@5 / hit@1).** The bar is the best of the eight SOTA rows per measure:

| dataset | bar's row | bar | zrc (lean MLP) | zsp (lean GNN) | twin0 (universal MLP) | gnn0 (universal GNN) |
| --- | --- | --- | --- | --- | --- | --- |
| metaqa | LINKED / LINKED / A1 | 0.241 / 0.208 / 0.007 | 0.780 / 0.667 / 0.892 | 0.779 / 0.667 / 0.889 | 0.721 / 0.596 / 0.771 | 0.784 / 0.667 / 0.889 |
| squad | A1 (= RRF) | 0.905 / 0.905 / 0.735 | 0.911 / 0.911 / 0.751 | 0.910 / 0.910 / 0.749 | 0.897 / 0.897 / 0.736 | 0.904 / 0.904 / 0.743 |
| musique | RRF / DENSE / RRF | 0.473 / 0.136 / 0.635 | 0.567 / 0.259 / 0.705 | 0.566 / 0.259 / 0.695 | 0.512 / 0.199 / 0.650 | 0.538 / 0.249 / 0.650 |
| hotpotqa | RRF / A1 / RRF | 0.685 / 0.437 / 0.829 | 0.905 / 0.848 / 0.852 | 0.905 / 0.849 / 0.856 | 0.879 / 0.795 / 0.833 | 0.899 / 0.836 / 0.866 |
| 2wiki | RRF / SPLADE / RRF | 0.608 / 0.256 / 0.875 | 0.870 / 0.706 / 0.894 | 0.873 / 0.721 / 0.898 | 0.849 / 0.666 / 0.871 | 0.891 / 0.759 / 0.919 |
| webqsp | DENSE | 0.110 / 0.088 / 0.071 | 0.307 / 0.242 / 0.110 | 0.268 / 0.209 / 0.134 | 0.602 / 0.488 / 0.563 | 0.629 / 0.511 / 0.637 |

webqsp is zero-shot for zrc and zsp, and in-domain for twin0 and gnn0.

**Where the margins are thin.** Everywhere else every model is above the bar by at least 0.018 on every measure.
- **squad.** The bar there is the first stage's RRF order. gnn0's R@5 is −0.0015 [−0.0052, +0.0019], within the tie,
  and its hit@1 is +0.0082. zsp is +0.0045 [+0.0014, +0.0077]. twin0 is BELOW (R@5 −0.0083), the only call below the
  bar among the four models.
- **hit@1 on musique.** The universal pair is +0.015, and the interval crosses 0.
- **hit@1 on hotpotqa and 2wiki.** twin0 is within the tie (+0.0035 and −0.0045).

**Claim 1's quantity, beside: the MLP's share of its GNN's gain over RRF.** It is left out where the GNN's gain is
within 0.002.

| family | measure | metaqa | squad | musique | hotpotqa | 2wiki | webqsp |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lean (zrc of zsp) | R@5 | 1.000 | 1.174 | 1.013 | 0.999 | 0.989 | 1.184 |
| lean | FC@5 | 0.999 | 1.174 | 1.000 | 0.998 | 0.968 | 1.191 |
| lean | hit@1 | 1.004 | 1.083 | 1.171 | 0.840 | 0.795 | 0.731 |
| universal (twin0 of gnn0) | R@5 | 0.920 | n/a | 0.596 | 0.904 | 0.852 | 0.952 |
| universal | FC@5 | 0.893 | n/a | 0.560 | 0.899 | 0.816 | 0.951 |
| universal | hit@1 | 0.867 | 0.171 | 1.027 | 0.094 | −0.103 | 0.875 |

- **The lean MLP** keeps at least 0.968 of the GNN's gain on R@5 and FC@5 on all six. On hit@1 it is below 0.9 on
  hotpotqa, 2wiki and webqsp (zero-shot).
- **The universal MLP** keeps at least 0.9 on R@5 on metaqa, hotpotqa and webqsp. It falls short on musique (0.596) and
  2wiki (0.852), and on hit@1 everywhere but metaqa and musique.

**What the bar is, and what it is not.**
- **Every bar row runs on our LLM-free graphs**, from our linking. That covers HippoRAG 2's two restart rules (on the
  regime graph and on all three families), the first-stage lists and RRF, and the KB-QA line's walk from the linked
  nodes.
  - It is not HippoRAG 2 as released, with its LLM-extracted triples, its LLM fact filter and its encoder.
  - It is not the KB-QA line's trained reasoners (ReaRev, NuTrea, GNN-RAG).
  - Both stay calibration lines, as decision 1's default says. On B1, HippoRAG 2's own corpora, our GNN was above
    HippoRAG 2's published R@5 on 2wiki and hotpotqa and below it on musique: 66.3 against 74.7 (B1b, 3b033f6).
- **On the passage datasets the bar is mostly the first stage.** RRF, SPLADE or DENSE sets every cell but hotpotqa's
  FC@5, where A1 leads. On our graphs, HippoRAG 2's walk adds coverage deep in the order (P1a), not the top five.
- **On metaqa and webqsp, the walk rows put the seeds first.** Those are the linked or topic entities, and they are
  rarely the answers (A1's hit@1 on metaqa: 0.007). Taking the seeds out of those orders was not declared here.
  Removing them moves any other node up by at most the number of seeds. P3b's file declares that read beside the bar.

**What follows (section 5).**
- **Claim 2.** It holds on our corpora against these rows, as a development read; the paper's confirmation run
  follows.
- **P3b's ledger** reads these records, in its own file.
- **The comparison with an LLM-built graph** is the 13 October link direction: how much of an LLM graph's gain an
  LLM-free construction can recover. It gets its own file.
- **No model changes here.**
