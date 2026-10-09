# U1d: precise universal links, one rule for all six

Declared 10 October 2026, about 01:15, before any full-corpus number of the variants below. It follows U1b
(docs/U1B_UNIVERSAL_LINKS_SIX.md). U1c, the retrain on the chosen graphs, gets its own file.

## Why

U1b's title-mention rule finds 70% of the hyperlinks on the 2wiki and hotpotqa full corpora, but only 10-14% of its
edges are hyperlinks. Its hubs are common words that are also titles ("From", "With", "That", "It Was") and, on webqsp,
Freebase schema words ("Record", "Topic", "Notable for"): 322 million mention edges against 8.3 million KB triples.

The user, 10 October: links first, then pools, then the models; recreate the hyperlinks universally "as much as
possible". U1d keeps the reach and removes the noise with one rule, chosen once for all six datasets.

## What the design look showed (not a result)

On the first 30,000 passages of hotpotqa and 2wiki, against their hyperlinks only (no question was read):

| hotpotqa sample | edges | P | R | F1 |
| --- | ---: | ---: | ---: | ---: |
| U1b's rule | 464,592 | 0.138 | 0.676 | 0.229 |
| + case-sensitive | 410,422 | 0.156 | 0.674 | 0.254 |
| + capitalization statistic | 222,972 | 0.285 | 0.669 | 0.400 |
| + aliases, no disambiguation | 1,160,589 | 0.059 | 0.715 | 0.108 |

On the 2wiki sample, the statistic estimated on 30,000 passages alone lost recall (0.694 -> 0.553). U1d estimates it on
the whole corpus.

Of the hyperlinks the rule misses (hotpotqa sample, 30,833), about 6,200 are titles with a parenthetical whose bare
name is in the text ("Paris (band)"), 2,500 are titles whose part before a comma is in the text ("Springfield,
Illinois"), and 8,200 are multi-token titles whose last token is in the text (a surname). Linking these aliases to every
owner floods the graph. So U1d links each one to one owner, chosen by the passage's context.

## The rule (one for all six)

**Surfaces of a title** (a node's `title` field; a KB node's text):
- kind 0: the title itself;
- kind 1 (alias): the title without a trailing parenthetical, and the text before its first comma;
- kind 2 (last): the last token of a multi-token title (parenthetical removed) that starts with a capital.

Surfaces shorter than 4 characters never match (c3_derived's floor).

**A match:**
1. The surface occurs word-bounded in the passage (U1b's matcher and bounded test).
2. **Case:** at that occurrence the passage writes the surface as the surface is written, except that the first
   letter may take either case.
3. **The capitalization statistic:** over the whole corpus, every word-bounded occurrence of the lowercased surface
   that does not start a sentence is counted as capitalized (first letter upper) or lowercase (all lower). A surface
   whose lowercase count exceeds its capitalized count is a common word, and it never links. A sentence starts at the
   text's start or after `. ! ? : ; " '`.

**Owners.** The nodes whose surface it is. Nodes with the same title form one title group: a title can span several
passages, as on musique. An edge goes to every node of the chosen group, never to the passage itself.
- One group: link to it.
- More than 50 groups (step 4e's linking guard): no link.
- Otherwise, link to the group whose frozen dense vector is closest to the passage's: the highest cosine over the
  group's nodes. The vectors are the package's `embeddings/dense/docs`. The encoder is unchanged; no model and no LLM
  is involved.

**Variants:**

| variant | surfaces | when an exact title owns the surface |
| --- | --- | --- |
| V2 | kind 0 | (only exact titles) |
| V5e | kinds 0, 1 | the exact titles only |
| V5 | kinds 0, 1 | all owners compete on context |
| V6e | kinds 0, 1, 2 | the exact titles only |
| V6 | kinds 0, 1, 2 | all owners compete on context |

**Relations.** As U1b: relation 0 on the passage corpora; a new relation id after the KB vocabulary on metaqa and
webqsp, whose KB triples are kept. 2wiki's and hotpotqa's hyperlinks are dropped.

## The choice, fixed before any number

- **Criterion:** the highest mean directed F1 of hyperlink recovery over the 2wiki and hotpotqa full corpora.
- **Tie rule:** a variant within 0.005 of the best loses to an earlier one in the table's order (the simpler rule).
- **Scope:** the chosen variant builds `structural_U` on all six datasets. No dataset gets its own choice.
- **Data read:** no question, gold or B1 file is read. The hyperlinks are the only labels.

F1 weighs the two sides equally. The pools and models then judge the graph in U1c and step 4h; U1d does not claim that
the F1 winner is the best graph for retrieval.

## Checks

- Inputs: the package freeze's RECORD_SHA256; nodes.jsonl against DATASET.json; structural.npz against
  GRAPH_MANIFEST; the dense manifest's row count and shard size, and each shard file's size against the manifest.
- Every link shard checks that the corpus statistic was merged from this code's stat shards on the same nodes
  (sha256s).
- Merges check coverage of [0, n), equal input and code shas, each shard's npz sha256, and no repeated pair or
  self-loop.
- A self-test covers the surfaces, the case test, the sentence-start rule, the statistic and the owner choice.
- An end-to-end run on musique on the laptop precedes the host run.

## What U1d reports

1. Hyperlink recovery per variant on 2wiki and hotpotqa (directed and unordered P, R; directed F1), beside U1b's rule
   (V0).
2. The choice.
3. Per dataset and variant: edges, isolated nodes, the largest in-degree, the most-linked titles; the KB share.
4. The chosen graph per dataset, against today's structural family and U1b's.

## How it runs

1. Host, per dataset: `stat` shards -> `statmerge` -> `link` shards -> `score`.
   - Shards: 2wiki 12, hotpotqa 8, webqsp 4, metaqa, musique and squad 1 each.
2. `choose`, once 2wiki and hotpotqa are scored.
3. `build` on all six, then `report`.

Code: `outputs/mp_unified/u1d.py`. Records: `outputs/u1d/<dataset>/{capstat,score,build}.json`,
`outputs/u1d/choice.json`, `outputs/u1d/report.{json,md}` (committed). Arrays are untracked.

## Results

Not yet run.
