# B1d: does a training-free bridge signal exist in our vectors and graphs? (chain slots with query offsets)

Declared 9 October 2026, before any number. B1a built HippoRAG 2's three settings; B1c found that personalised
PageRank does nothing (SAME on all three) and that the misses are second steps of chains. 86% to 99.5% of the misses
have a sibling gold in the top 5, and most are one edge from it.

## The question

B1c's walks could not lift a neighbour into the top 5. This stage asks a narrower question, with nothing trained:
- given the first stage's leading passage(s), is the missing second passage the best-scoring neighbour of a found one;
- and is that neighbour found better by a query offset (the question vector plus the found passage's vector) than by
  the question alone?

If yes, the bridge is a signal our frozen vectors and LLM-free graphs already carry, and G1 learns it as a model input
for both models. If no, G1 has to find it elsewhere. This is also the first test of the offset-space direction: a
parallelogram-style composition of existing vectors, with nothing re-encoded.

## The base stays fixed

The encoder and the package's vectors, B1a's graphs and first-stage lists, and our models are all unchanged. Nothing
is trained or re-encoded.

## Arms

Every arm keeps the first stage's top *m* passages and fills slots *m*+1 to 5 with bridge candidates:
1. **Sources:** the first stage's (RRF) top *s* passages.
2. **Candidates:** passages joined to a source by an edge of the arm's families (undirected), not already kept.
   Family set `none` instead takes the first stage's top 100 with no edge needed.
3. **Bridge score** of a candidate *p*: the maximum over its sources *r* of
   - `Q`: cos(q, p), the question alone; or
   - `QR`: cos(normalise(q̂ + r̂), p̂), the question plus the source, where x̂ is x normalised to length 1.
4. **The list:**
   - the kept top *m*;
   - then candidates by bridge score, until 5 are filled (ties by first-stage order);
   - then the rest of the first-stage list in order.
   If there are too few candidates, the first stage fills the gap.

Four settings vary, giving 40 arms; with R0 that is 41.

| setting | choices |
| --- | --- |
| *s* | 1, 2 |
| *m* | 3, 4 |
| families | all, structural, ner, knn, none |
| score | Q, QR |

Arms are named `s<s>m<m>/<families>/<score>`. **R0** is the RRF first stage, unchanged.

## Choice and call

These are the same rules as B1c:
- **Choice.** The arm with the best R@5 on the even questions is chosen, ties going to the earlier arm in the order
  *s*, *m*, families, score as listed.
- **Call.** The odd-half R@5 gain over R0 is ABOVE, SAME or BELOW by a 95% bootstrap interval (2,000 resamples,
  seed 0).
- **Filed numbers.** R@2, R@5 and R@10 of every arm on even, odd and all.
- **The offset question.** For each *s*, *m* and families, the QR arm's R@5 minus the Q arm's on all questions,
  filed.

## Encoder check (E2 of docs/PAPER_CLAIMS_2026_10_09.md)

The chosen arm's rule is rerun with SPLADE's list in place of RRF, both as the sources and as the kept list; Q and QR
still use the dense vectors. Its gain over the SPLADE list is filed beside the RRF gain. This is reported, never used
to choose.

## What it decides

- **ABOVE on at least one setting:** the bridge is in the vectors. G1 (declared in its own file) turns the
  chosen arm's quantities into per-row inputs for both models:
  - the best bridge score;
  - the source rank;
  - the family of the edge;
  - the QR-minus-Q margin.

  These are computed from the first stage, before any row is scored. The MLP may therefore read them without message
  passing.
- **SAME everywhere:** the signal is not in a fixed composition, and G1 must learn the composition.

## Running it

    python scripts/b1d_bridge_slots.py run   -> outputs/bench/hipporag2/b1d.json, b1d.md
    python scripts/b1d_bridge_slots.py --selftest

## Results

(filled after the run)
