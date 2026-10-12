# D4: do the words of a link say which neighbour the question follows?

Declared 12 October 2026, about 23:55, before any of its numbers. This is a diagnostic. It decides one thing: whether
round thirty-four (the words of a link as an input of both families) is declared.

It trains nothing and changes no fit, graph, encoder or feature. It reads zrc's fits (the MLP's base) on CPU through
zrct's read, which is zrc's. As D1 did with zrm (docs/DIAG_BRIDGE.md), it reads a copy of each fit's `models.pt` and
`screen.json` in a work folder, so the sealed fit folders are never written. A diagnostic is not a result, and no
number from it is cited as one.

## Why

- **musique is the one B1 setting below HippoRAG 2** (66.3 against 74.7; docs/B1B_MODELS_ON_HIPPORAG2.md). B1 is read
  here only to say where the gap is. D4 runs on the development carves, and nothing is chosen on B1.
  - Our first stage starts 13.3 points below NV-Embed-v2 alone (56.4 against 69.7). Our models add +9.9 over it,
    twice HippoRAG 2's lift over its own encoder (+5.0).
  - The pools hold 98.2% of the golds. By hop count (zsp J5), R@5 is 77.0 on 2-hop, 61.8 on 3-hop and 41.3 on 4-hop
    questions. 752 of the 990 golds it misses are in 3- and 4-hop questions.
- **D1 (REACHABLE_NOT_RANKED):** the golds the model misses sit next to golds it finds. On musique, 0.38 of them are
  direct neighbours and 0.77 are within two hops; on 2wiki 0.77, on hotpotqa 0.92. Among about 45 such neighbours on
  musique, its score ranks the missed gold 5th to 9th. It ranks it 1st or 2nd for only 0.22 to 0.28 of them.
- **D2 (FEATURE_LIMIT):** a stronger scorer on the same per-row inputs gains nothing.
- **Rounds 29 to 32 and U1c moved nothing on musique.** They tried:
  - the first stage's bridge rows as inputs (zbr);
  - more propagation (zdp);
  - a learned four-layer GNN (zg1);
  - an offset space from the leading rows (zof);
  - new links with new pools (U1c).
- **None of them reads which link a question follows.**
  - A KB link carries a relation name, and the KB models read it (relz).
  - A passage link carries none. A structural link is a title that one passage mentions. A NER link is an entity
    that both passages mention.
  - The words around that mention say what the link means. In "Green is the fourth studio album by English musician
    Steve Hillage", the words around "Steve Hillage" say who made the album. HippoRAG 2's LLM-built triples hold such
    words.
  - D4 asks whether plain words, with no LLM, single out the neighbour the question follows where the model's score
    does not.

## What it measures

For each of the two fits, zrc's card fits on L-musique and L-hotpotqa (`scr-zrct`, `scr-zrct-hp`), on the six
s1eval carves:

- **The base score:** zrc's p@swa score per row, kept as D1 kept zrm's.
  - Identity: zrc's R@5 here must lie within 0.002 of the fit's filed read.
  - Each question's count of golds in the top five must equal the read's.
  - Otherwise the run exits 1 and gives no verdict.
- **The question's stems:** lowercased runs of letters and digits, less a fixed English stopword list and single
  characters. Each is cut to its first five characters (a fixed truncation stemmer).
- **A link's words:** one window per mention, the mention's own tokens left out.

  | link | its windows |
  | --- | --- |
  | structural, passage graphs (musique and squad's title mentions; 2wiki and hotpotqa's hyperlinks) | the 10 tokens on each side of each occurrence of one end's title in the other end's text, in both directions |
  | structural, KB graphs (metaqa, webqsp) | the names of the relations in the edge's slots |
  | NER (musique and squad) | the 10 tokens on each side of each mention, in both passages, of each entity both passages mention, if its corpus df is within the graph's [2, 25] |
  | kNN | none |

  - The hyperlink graphs store no anchor text. A hyperlink whose target's title does not occur in the source text has
    no words.
  - D4 gives no words to NER links on 2wiki, hotpotqa or metaqa. 2wiki and hotpotqa hold millions of passages
    (hotpotqa 5.23M), and spaCy over them would take about 18 CPU-hours each for reads that decide nothing.
- **The link score** of a candidate c given a found row a: the largest, over the links between a and c and their
  windows, of the summed IDF of the question's stems found in the window.
  - The stems of a's and c's titles are left out, so the score speaks for the link, not for its ends.
  - IDF is over every node's title and text: log((N + 1) / (df + 1)).

Over the questions zrc finds partly (at least one gold in its top five, at least one in-pool gold outside it), each
missed gold next to a found gold is ranked among the found golds' neighbours outside the top five (D1's nb rank),
three ways:

| order | how |
| --- | --- |
| base | by zrc's score |
| words | by the link score (the largest over the found golds), ties by zrc's score |
| fused | by the sum of the two ranks, ties by zrc's score |

Reported for each read:
- the share ranked 1st or 2nd (`le2`) and the median rank, for each order;
- coverage: the shares of missed golds and of all candidates with a window, and with a non-zero link score;
- for the missed golds with a non-zero score, which family gave it.

Reported as R@5 against zrc over every question with a gold, D1's O1 and C1 and the same with the fused order:
- **O1 and O1w (oracle anchor):** zrc's top three, then the two best neighbours of the highest-ranked gold in zrc's top
  five (by zrc's score in O1, by the fused order in O1w), then zrc's order;
- **C1 and C1w (label-free):** the same, anchored on zrc's top-1.

## The rule, fixed before any number

The rule is read on musique, the setting the gap is in. 2wiki and hotpotqa are reported beside it.

- **SIGNAL:** on musique, in both fits, the fused order ranks the missed gold 1st or 2nd at least 0.05 more often than
  zrc's score does, and at least 1.25 times as often.
- **WEAK:** that holds in one fit only.
- **NO_SIGNAL:** otherwise.
- **INCOMPLETE** in any of these cases:
  - a fit fails its identity;
  - the spaCy mentions rebuild less than 0.9 of musique's NER edges (the words could not be recovered);
  - musique ranks no missed gold.

| verdict | what follows |
| --- | --- |
| SIGNAL | Round thirty-four is declared in docs/SCREENS.md. The link score becomes an input of both families on all six datasets. The MLP's version is anchored on the first stage's top rows, so it is computed before any row is scored (no message passing). The GNN's is also anchored on its own top rows. Each family gets a two-fit screen decided under the seed null. NER mentions are first built for every corpus (spaCy on the host). |
| WEAK | Reported and weighed. No screen is declared from D4 alone. |
| NO_SIGNAL | The words around a link do not say which neighbour a question follows. No ranking screen follows from D4. musique's gap is reported as the first stage's: our list starts 13.3 points below NV-Embed-v2, and our models add twice HippoRAG 2's lift. |
| INCOMPLETE | The cause is fixed and the run repeated. No other number from it is read. |

O1w, C1w, 2wiki, hotpotqa, squad and the KB reads are reported, not decided.

## Where the words come from

- **Texts and titles:** the served package's `nodes.jsonl`. Line i is node i, the position a pool holds. A KB node's
  title is its text, as in u1b. The file's sha256 is checked against `DATASET.json` (on the host, its verified
  mirror).
- **Questions:** the eval split files named in `configs/m3b_controlled_comparison.yaml` (`dev`; hotpotqa
  `validation`; webqsp's `train_holdout`, taken from `train`). Every s1eval id is in them. No test split is read.
- **Relation names:** `outputs/m3b/relations/<ds>_rel_vocab.json`, indexed by the look's relation slots.
- **NER mentions:** spaCy 3.8.15 with en_core_web_sm 3.8.0, set up as the graph was built (CRAG
  `src/pipeline/ner_edges.py`):
  - the parser and lemmatizer are off;
  - the nine labels are kept;
  - keys are the lowercased, stripped entity text, more than 2 characters;
  - df is over the whole corpus.
  - On the host, spaCy runs from the laptop's installed files: `outputs/host_ops/pylib/spacy3815_cp313_win_amd64.zip`
    (sha256 `2eda7570…`, never committed). It is unpacked beside the numba bundle and put at the end of `sys.path`.
    The host's pip cannot reach the index past the lab's TLS-inspecting firewall. The host and the laptop run the
    same Python (3.13.5) and numpy (2.3.2).
  - Before any number, 300 of musique's NER edges were sampled, and both ends of every one share an entity key.
  - `ents` rebuilds the graph's NER edges from the mentions and files the share it recovers (the gate above).

## How it runs

- **Code:** `outputs/mp_unified/linkwords.py` (new, frozen once run):
  - `ents`: the NER mentions of one corpus;
  - `run`: both fits in one job, with one pass over each corpus for the IDF and the texts, then the verdict;
  - `verdict` and `--selftest`.
  - It reuses D1's capture of the per-row scores (bridgediag.py) with zrc's read in place of zrm's.
- **On the host's CPU lane:**
  - `ents` for musique (8 CPUs), then squad;
  - then the run (8 CPUs). D1's read took about 4 minutes per fit. The corpus passes add about 10 minutes, mostly
    2wiki's 5.3 GB.
- **Outputs:**
  - `outputs/linkwords/<ds>_ents.{npz,json}` (the npz is never committed);
  - `outputs/diag/linkwords-scr-zrct{,-hp}.{json,md}`;
  - `outputs/diag/linkwords-zrc.{json,md}` (the verdict).

## Amendment 1 (13 October, 00:25, before the run)

The rule is unchanged. Three things about how it runs changed.

- **The NER mentions were built on the laptop, not on the host.** The laptop runs the same spaCy (3.8.15),
  en_core_web_sm (3.8.0), Python (3.13.5) and numpy (2.3.2) over the same served nodes, with nodes.jsonl's sha256
  checked. The files reach the host by `rx push --inputs`. The run checks each file's content digest and the nodes'
  sha256 when it loads them. The host needs no spaCy, and the zip stays only as a fallback.
- **The rebuild gate is met in full.** The mentions rebuild every NER edge of both graphs, with none extra:
  - musique: 800,278 of 800,278 edges;
  - squad: 156,130 of 156,130 edges;
  - weights within 7.1e-7 of the graph's (float32 rounding);
  - records: `outputs/linkwords/<ds>_ents.json`.

  They were built twice, the second time by the committed script. Both builds give the same content digest.
- **Two guards were added to the run:**
  - the words are read only on the s1eval carves, while identity is still checked on every carve the read covers;
  - each fit's record is written after each dataset, so a later failure keeps the reads already done.
- **The job:** one item on the host's CPU lane, 4 threads.
  - The two reads take about 4 minutes each.
  - One pass over each corpus: about 6 minutes for 2wiki (6.0M nodes) and 5 for hotpotqa (5.2M nodes), timed on the
    laptop at 58 µs a node.
