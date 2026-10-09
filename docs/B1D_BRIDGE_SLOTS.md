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

Run 9 October 2026, 15:27–15:33, on the laptop. Records: `outputs/bench/hipporag2/b1d.json` and `b1d.md`.

**The call: ABOVE on all three settings.** The bridge is in our vectors and graphs, and nothing is trained.

| setting | R0 (RRF) R@5 | chosen arm | R@5 (all) | odd-half gain [95% CI] | call |
| --- | ---: | --- | ---: | --- | --- |
| musique | 56.4 | s1m4/ner/Q | 60.4 | +4.0 [+2.5, +5.7] | ABOVE |
| 2wiki | 71.2 | s2m3/structural/Q | 92.8 | +21.2 [+18.9, +23.2] | ABOVE |
| hotpotqa | 85.0 | s2m3/structural/Q | 95.2 | +10.0 [+8.0, +11.9] | ABOVE |

**Against HippoRAG 2**, scored by Claim 2's rule (the bootstrap interval of our R@5 against the published number):

| setting | ours, all 1,000 [95% CI] | ours, odd half [95% CI] | HippoRAG 2 | call |
| --- | --- | --- | ---: | --- |
| musique | 60.4 [58.6, 62.4] | 60.3 [57.7, 62.9] | 74.7 | BELOW |
| 2wiki | 92.8 [91.7, 93.8] | 92.3 [90.7, 93.9] | 90.4 | **ABOVE** |
| hotpotqa | 95.2 [94.2, 96.2] | 94.9 [93.4, 96.2] | 96.3 | BELOW (the interval's top is 0.1 below) |

**E1, the method's lift.** This compares our gain over our own dense list with HippoRAG 2's gain over NV-Embed-v2's
dense list:

| setting | ours | HippoRAG 2 |
| --- | ---: | ---: |
| musique | +3.7 | +5.0 |
| 2wiki | +22.0 | +13.9 |
| hotpotqa | +10.7 | +1.8 |

**E2, a weaker first stage.** The same rule on SPLADE's list gains as much or more:

| setting | SPLADE R@5 | with the rule | gain |
| --- | ---: | ---: | ---: |
| musique | 50.8 | 56.5 | +5.7 |
| 2wiki | 70.5 | 92.6 | +22.1 |
| hotpotqa | 80.3 | 94.1 | +13.8 |

The gain does not come from the dense encoder.

**The offset question: no.** Adding the source to the question (QR) does not beat the question alone (Q) where it
matters. On the structural and ner rules it is 0 to −3 points. It helps only without edges (`none`: +2 to +5 on 2wiki),
or with knn. A fixed vector sum is not the bridge; the edge is. Any offset must be learned (G1).

**By question kind.** The gain sits where the ledger said it would:

| setting | kind | R@5 before | R@5 after |
| --- | --- | ---: | ---: |
| 2wiki | compositional | 0.654 | 0.925 |
| 2wiki | bridge-comparison | 0.531 | 0.879 |
| 2wiki | inference | 0.727 | 0.935 |
| hotpotqa | bridge | 0.822 | 0.953 |
| musique | 2-hop | 0.652 | 0.715 |
| musique | 3-hop | 0.525–0.539 | 0.531–0.543 |
| musique | 4-hop | 0.333–0.435 | 0.370–0.444 |

The fixed rule gives up a little on comparison questions: hotpotqa 0.971 to 0.947, and 2wiki 0.978 to 0.975. A
comparison question's two golds are both near the question, so spending two slots on neighbours costs it.

**The caveat to state in the paper.** On 2wiki and hotpotqa, the winning family is `structural`: the documents' own
Wikipedia hyperlinks.
- **Provenance.** These come from the source corpora (2Wiki's `para_with_hyperlink`, HotpotQA's `text_with_links`).
  They are question-independent and LLM-free, and the package's provenance rules hold them as node-local source facts.
- **But HippoRAG 2 does not use them**, and both datasets built their bridge questions along hyperlinks.

So the comparison must also be shown with a text-only graph (ner, knn and `none`; musique's structural family is title
mentions, already text-derived). Text-only, best arm, R@5 (all):

| setting | R@5 | gain over R0 |
| --- | ---: | ---: |
| musique | 60.4 | +4.0 (unchanged) |
| 2wiki | 81.8 | +10.6 |
| hotpotqa | 86.7 | +1.7 |

These are read from the arm table, not chosen. Text-only, 2wiki and hotpotqa stay below HippoRAG 2. The gap to close
without hyperlinks is the entity-level link, which is what the LLM-free entity-node graph is for.

**What this decides.**
- The bridge is a real, LLM-free signal: ABOVE on all three settings, with nothing trained.
- G1 (declared next, in its own file) makes the slot rule's quantities per-row inputs to both models. These are the
  best bridge score, the source rank, the edge family and a slot indicator. G1 learns:
  - when to spend the slots, which fixes the comparison questions;
  - how far to chain, which is musique's 3–4 hops.
- Musique is the setting that needs the most. Its misses are 14 points from HippoRAG 2, and they are 3–4 hop chains,
  where one fixed slot step adds almost nothing.

## Addendum (9 October, 15:45, before its numbers): one universal rule

The user's ruling: every fix must be universal and dataset-agnostic. The choice above picked a different arm per
setting (musique `s1m4/ner/Q`, 2wiki and hotpotqa `s2m3/structural/Q`). That is a per-dataset choice, and it is not
the result.

**The universal rule.**
- **Choice.** One arm for all three settings: the arm with the highest mean, over the three settings, of its
  even-half R@5. Ties go to the earlier arm.
- **Call.** It is called on each setting's odd half against R0, as above.
- **HippoRAG 2.** It is scored against HippoRAG 2 by Claim 2's rule.

**Families.** Family sets are kinds of edge, and each kind is built by one rule wherever it exists:
- `structural` means the corpus's own document links. These are hyperlinks where the source has them; on musique,
  title mentions.
- `ner` is shared spaCy entities.
- `knn` is the 3 nearest neighbours.

A family absent from a dataset is empty there, never replaced.

Run with `python scripts/b1d_universal.py`, which writes `outputs/bench/hipporag2/b1d_universal.json`. The per-setting
table above stays as the upper bound of a per-dataset choice. From now on, the universal arm is the number that is
cited.

### Addendum results (9 October, 15:50; `outputs/bench/hipporag2/b1d_universal.json`)

**The universal arm is `s2m3/structural/Q`**, with a mean even-half R@5 of 0.818. The next four arms are all
`structural` variants, 0.805–0.813. One rule for all three settings: keep the top 3, then fill slots 4 and 5 with the
best document-link neighbours (by cos with the question) of the top 2.

| setting | R0 R@5 | universal R@5 (all) [95% CI] | odd-half gain [CI] | vs R0 | HippoRAG 2 | vs HippoRAG 2 | per-setting best |
| --- | ---: | --- | --- | --- | ---: | --- | ---: |
| musique | 56.4 | 57.0 [55.1, 59.0] | +1.1 [-0.7, +2.9] | SAME | 74.7 | BELOW | 60.4 (ner) |
| 2wiki | 71.2 | 92.8 [91.7, 93.8] | +21.2 [+19.0, +23.3] | ABOVE | 90.4 | **ABOVE** | 92.8 |
| hotpotqa | 85.0 | 95.2 [94.2, 96.2] | +10.0 [+8.0, +11.9] | ABOVE | 96.3 | BELOW | 95.2 |

**Reading.**
- **musique's 4-point gain was a per-dataset choice.** It came from picking the ner family for musique. Under one
  rule, musique gets +1.1 (SAME).
- **The universal fix needs a graph whose links mean the same thing on every dataset.** Today, `structural` is
  hyperlinks on two datasets and title mentions on the third, and `ner` helps where `structural` does not.

  The next stage is one LLM-free graph representation, built by one rule on all six datasets and the B1 settings.
  Its link kinds, the rule's quantities and the model that combines them are all fixed once and are never chosen
  per dataset:
  - document links where the source has them;
  - entity nodes from spaCy entities;
  - kNN.
