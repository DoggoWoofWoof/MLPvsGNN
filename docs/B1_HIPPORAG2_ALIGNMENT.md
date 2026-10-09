# B1: HippoRAG 2's benchmark settings, with our graphs (stage B1a: the build)

Declared 9 October 2026, before any number. This is program stage B1 (docs/PROGRAM_2026_10_09.md). The user approved
downloading the files and building the graphs on 9 October.

## Why

The published multi-hop retrievers (HippoRAG 2, GFM-RAG, G-reasoner, HGRAG, PropRAG) report R@5 on the same three
small settings: 1,000 questions each from MuSiQue, 2WikiMultiHopQA and HotpotQA, each retrieving from its own corpus
of the questions' passages. HippoRAG 2 releases these files. Our numbers are on other populations and on
open-corpus graphs (5–6 million passages on 2wiki and hotpotqa), so they cannot sit beside the published tables.
B1 rebuilds the published settings with our substrate and our graph method, so our models' R@5 can be put next to the
published numbers on the same questions and corpora.

B1a (this file) builds the settings and files only the first-stage R@5 as a sanity anchor. B1b, the models' R@5,
is declared in its own file before it runs.

## Source

HuggingFace dataset `osunlp/HippoRAG_2`, main branch: `musique.json`, `musique_corpus.json`,
`2wikimultihopqa.json`, `2wikimultihopqa_corpus.json`, `hotpotqa.json`, `hotpotqa_corpus.json`. These are fetched to
`outputs/bench/hipporag2/raw/`, with bytes and sha256 in `fetch.json`. The raw files are third-party data and are
not committed.

Sizes:

| setting | questions | passages |
| --- | ---: | ---: |
| musique | 1,000 | 11,656 |
| 2wiki | 1,000 | 6,119 |
| hotpotqa | 1,000 | 9,811 |

## The build: nothing is re-encoded

**1. Passages → package rows.** Every passage maps to one row of the frozen CRAG package's dataset of the same name:
- the title must be equal;
- the text must be equal with all whitespace removed.

  The probe of 9 October found every passage this way:
- musique and 2wiki match exactly;
- hotpotqa matches exactly on 9,606 passages. The other 205 differ only by one space after an opening quote (our row
  reads `" Decibel`, HippoRAG 2's `"Decibel`).

  Where several rows match, the lowest position is taken. The setting's node *i* is the *i*-th passage of the
  corpus file. Its dense and SPLADE vectors are those of its package row, from the frozen encoder; nothing is
  re-encoded.

  A passage that matches no row stops the build. Re-encoding would change the encoder's inputs, and that would need
  its own declaration.

**2. Questions → package query rows.** Each question is matched by its id, and its question text must be equal:
- musique and 2wiki: the dev split;
- hotpotqa: the validation split.

  Its dense and SPLADE query vectors are the package's. These are dev-side questions, so none of our models has
  trained on them; our training carves draw from the train splits. A question that does not match stops the build.

**3. Golds.** These are HippoRAG 2's supporting passages, mapped to setting nodes:
- musique: the paragraphs with `is_supporting`, by title and text;
- 2wiki and hotpotqa: the `supporting_facts` titles.

  The agreement with the package's own gold rows for the same questions is recorded, not used.

**4. Graphs.** These follow the package's method (CRAG HANDOFF §7), on the setting's nodes:
- **structural:** the subgraph of the package's structural family induced on the setting's rows, with direction and
  relation kept. This is title-mention links for musique and Wikipedia hyperlinks for 2wiki and hotpotqa. A
  title-mention edge depends only on its two passages, so the induced subgraph is what a rebuild would give.
- **ner:** rebuilt on the setting's passages by CRAG's `build_ner_edges` rule:
  - spaCy en_core_web_sm, with the parser and lemmatizer disabled;
  - the 9-label whitelist;
  - entity text longer than 2 characters, lowercased and stripped;
  - 2 ≤ df ≤ 25 within the setting;
  - weight = Σ 1/df;
  - one row per unordered pair.

  It is rebuilt, not induced, because df depends on the corpus.
- **knn:** rebuilt as exact 3-nearest-neighbour cosine over the setting's dense vectors (float32 compute from the
  float16 rows), self excluded, one row per unordered pair, weight = cosine. It is rebuilt because the neighbours
  depend on the corpus.

**5. First stage.** The dense and SPLADE top-1000 lists of each question over the setting's nodes, by exact inner
product on the package's vectors. Ties are broken by node index.

## Filed numbers (B1a)

- **R@5** (the published metric: the share of a question's supporting passages in the top 5, averaged over
  questions) of:
  - the dense list;
  - the SPLADE list;
  - their RRF, with the package's constant.

  These are first-stage numbers only, a sanity anchor for B1b. They are not a result about our models.
- **Graph statistics** per family: edges, mean degree, isolated nodes.
- **Gold coverage of the graph:** the share of gold passages within 1 and 2 hops of the dense top 5.

## Running it

    python scripts/b1_build.py build [--datasets musique 2wiki hotpotqa]   -> outputs/bench/hipporag2/<dataset>/
    python scripts/b1_build.py --selftest

Outputs per setting:
- `nodes.json`: each node's package row, title and match kind;
- `queries.json`: ids, package query rows, gold nodes;
- `graph_<family>.npz`;
- `first_stage.npz`;
- `build.json`, with every sha256 and statistic.

## Results

(Filed after the run.)
