# B1c: what a training-free walk on our graphs buys on HippoRAG 2's settings, and where the misses are

Declared 9 October 2026, before any number. B1a (docs/B1_HIPPORAG2_ALIGNMENT.md) built the three settings.

## The base stays fixed

The user's instruction of 9 October: close the gaps, find more issues and keep closing them, without changing the
base. The base is:
- the frozen encoder and the package's vectors;
- the graphs built in B1a;
- the current models (the zrc MLP and the zsp GNN).

This stage changes none of them. It trains nothing. It measures what personalised PageRank over our three edge
families adds to the first stage, which is the core of HippoRAG 2 without its LLM-built graph. It then places every
miss.

## Arms

Each arm is a personalised PageRank on one setting's graph:
- edges as undirected, weighted by family (structural 1, ner Σ1/df, knn cosine);
- each node's outgoing weights normalised to 1;
- restart probability 0.5, as in our step 4 walks;
- 50 power iterations;
- passages ranked by their PageRank score, ties broken by RRF order.

Two settings vary across arms, giving 10 walk arms; with R0 that is 11 arms in all. The walk arms are named
`<restart>/<families>`, e.g. `T5/all`.

**Restart** (two choices):

| name | restart mass |
| --- | --- |
| T5 | uniform on the RRF top 5 |
| RW | on the RRF top 1000, in proportion to 1/(60 + rank) |

**Families** (five choices): all, structural, ner, knn, ner+structural.

**R0** is the base for comparison: the RRF first stage, unchanged.

## How an arm is chosen, and what is filed

- **Choice.** The arm with the best R@5 on each setting's even-numbered questions (0, 2, 4, ...) is chosen. Ties go to
  the earlier arm in the order above. The odd-numbered questions are never used to choose.
- **Filed numbers.** R@2, R@5 and R@10 of every arm on the even half, the odd half and all 1,000 questions, next to
  HippoRAG 2's published R@5.
- **The decision.** The chosen arm is ABOVE R0 when its odd-half R@5 gain over R0 has a 95% bootstrap interval
  (2,000 resamples of questions, seed 0) wholly above 0. It is BELOW when the interval is wholly below 0, and SAME
  otherwise.

## The ledger

For the chosen arm on all 1,000 questions, every gold outside the top 5 is placed, with:
- its rank in the arm and in R0;
- its hop distance from the arm's top 5, over all families undirected, and per family (to 3 hops);
- the question's kind: hop count on musique; question type on 2wiki and hotpotqa;
- whether the question names the gold. This is either the gold's title, normalised, appearing in the normalised
  question, or the question and the gold sharing a named entity. Entities come from spaCy en_core_web_sm with the
  9-label whitelist of the B1a graphs, on the question and on the gold passage;
- whether another gold of the same question is in the top 5. If so, this miss is the second step of a chain.

Buckets (first that fits):

| bucket | the miss |
| --- | --- |
| N1 | is named by the question but ranked below 5 |
| C1 | is one hop from a found gold of the same question |
| T1 | is one hop from the top 5 (not via a found gold) |
| R10 | is in the top 10 |
| FAR | everything else |

Each bucket points to a different fix, and the next stage (declared in its own file) takes the largest.

## Running it

    python scripts/b1c_walks.py run [--datasets musique 2wiki hotpotqa]   -> outputs/bench/hipporag2/b1c.json, b1c.md
    python scripts/b1c_walks.py --selftest

## Results

Run 9 October 2026, 15:00–15:08, on the laptop. Records: `outputs/bench/hipporag2/b1c.json` and `b1c.md`.

**Walks: SAME on all three settings.** No training-free walk moves R@5 off the first stage.

| setting | R0 (RRF) R@5 | chosen arm | its R@5 (all) | odd-half delta [95% CI] | call | HippoRAG 2 |
| --- | --- | --- | --- | --- | --- | --- |
| musique | 56.4 | R0 | 56.4 | +0.0 [+0.0, +0.0] | SAME | 74.7 |
| 2wiki | 71.2 | T5/structural | 71.7 | +0.3 [-0.2, +0.8] | SAME | 90.4 |
| hotpotqa | 85.0 | T5/structural | 85.7 | +0.1 [-0.3, +0.5] | SAME | 96.3 |

R@5 (all 1,000) of every arm:

| arm | musique | 2wiki | hotpotqa |
| --- | --- | --- | --- |
| R0 | 56.4 | 71.2 | 85.0 |
| T5/all | 56.4 | 71.2 | 85.0 |
| T5/structural | 55.9 | 71.7 | 85.7 |
| T5/ner | 56.4 | 71.2 | 85.0 |
| T5/knn | 56.4 | 71.2 | 85.0 |
| T5/ner+structural | 56.4 | 71.2 | 85.2 |
| RW/all | 28.8 | 47.4 | 64.8 |
| RW/structural | 17.5 | 55.2 | 62.0 |
| RW/ner | 24.3 | 32.8 | 43.0 |
| RW/knn | 28.8 | 36.0 | 49.5 |
| RW/ner+structural | 16.1 | 39.9 | 61.5 |

Ledger of the chosen arm's misses:

| setting | misses / golds | N1 | C1 | T1 | R10 | FAR | named | second of a chain | median rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| musique | 1234 / 2648 | 0.190 | 0.423 | 0.150 | 0.024 | 0.212 | 0.190 | 0.859 | 35.5 |
| 2wiki | 787 / 2470 | 0.066 | 0.909 | 0.001 | 0.001 | 0.023 | 0.066 | 0.995 | 7 |
| hotpotqa | 287 / 2000 | 0.355 | 0.585 | 0.035 | 0.000 | 0.024 | 0.355 | 0.916 | 6 |

- musique: hops from top 5 {'1': 859, '2': 261, '3': 107, '-1': 7}; one hop by family {'structural': 0.315, 'ner': 0.554, 'knn': 0.327}; R@5 by kind {'2hop': 0.6525, '3hop1': 0.5254, '3hop2': 0.5388, '4hop1': 0.3333, '4hop2': 0.3611, '4hop3': 0.4355}
- 2wiki: hops from top 5 {'1': 751, '2': 13, '3': 13, '-1': 10}; one hop by family {'structural': 0.892, 'ner': 0.766, 'knn': 0.518}; R@5 by kind {'bridge_comparison': 0.5309, 'comparison': 0.9754, 'compositional': 0.6659, 'inference': 0.7315}
- hotpotqa: hops from top 5 {'1': 267, '3': 6, '2': 10, '-1': 4}; one hop by family {'structural': 0.906, 'ner': 0.62, 'knn': 0.648}; R@5 by kind {'bridge': 0.8298, 'comparison': 0.9709}

Why the walks did nothing, read from the arms rather than tuned after the fact:
- **T5 cannot change the top 5.** With half of the mass restarting on the RRF top 5 (0.1 each per step), those five
  nodes keep the five highest scores. The walk can only reorder ranks 6 and below, so the T5 arms equal R0 or move
  R@5 by under a point.
- **RW spreads its restart over 1,000 passages and smears them.** Every RW arm falls 20–40 points.

A walk that can lift a neighbour into the top 5 needs restart mass that falls off steeply with rank, or passages
scored by their neighbours. That is a design question for the next stage, not a reason to tune this one on the
benchmark's questions.

**The ledger is the finding.** The misses are second steps of chains:
- **The share of misses whose question already has another gold in the top 5:**

| setting | share |
| --- | ---: |
| 2wiki | 99.5% |
| hotpotqa | 92% |
| musique | 86% |

- **The share of misses one hop from that found gold:**

| setting | share | carried mostly by |
| --- | ---: | --- |
| 2wiki | 91% | the hyperlink family (89% of misses) |
| hotpotqa | 58% | — |
| musique | 42% | ner (55% of misses one hop from the top 5) |

- **hotpotqa:** a further 35% of misses are named by the question (title or shared entity) but still ranked below 5.
- **R@5 by question kind:**
  - musique falls with hops: 0.65 at 2 hops, 0.53–0.54 at 3, 0.33–0.44 at 4;
  - 2wiki bridge-comparison: 0.53;
  - comparison questions: about 0.97 on both 2wiki and hotpotqa.

The gap to HippoRAG 2 on these settings is the bridge: the first passage is found, and the second is one edge from it
but not ranked. This is the same "reachable, not ranked" result that D1 found on our own six datasets. The next stage
scores a passage by the passages it links to (the bridge), with the base unchanged, and is declared in its own file.
