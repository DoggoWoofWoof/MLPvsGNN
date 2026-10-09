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

(Filed after the run.)
