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

## Results (13 October, 01:19)

**NO_SIGNAL.** Neither fit passes the rule on musique.

| fit | le2, zrc's score | le2, fused | difference | ratio | the rule's bar | passes |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| scr-zrct | 0.225 | 0.261 | +0.036 | 1.16 | +0.050 and 1.25 | no |
| scr-zrct-hp | 0.291 | 0.308 | +0.018 | 1.06 | +0.050 and 1.25 | no |

What follows, as declared:
- The words around a link do not say which neighbour a question follows.
- No ranking screen follows from D4. Round thirty-four is not declared.
- musique's gap is reported as the first stage's. Our list starts 13.3 points below NV-Embed-v2 (56.4 against
  69.7), and our models add 9.9 over it, twice HippoRAG 2's lift over its own encoder (5.0).

**Both runs are complete, and identity passed on every carve read.** zrc's R@5 equals each fit's filed read:
- scr-zrct: 2wiki 0.8733, hotpotqa 0.9041, metaqa 0.7782, musique 0.5235, squad 0.9105, webqsp 0.3384.
- scr-zrct-hp: 2wiki 0.8731, hotpotqa 0.8451, metaqa 0.7793, musique 0.5645, squad 0.9112, webqsp 0.2865.
- The NER mentions rebuilt 800,278 of musique's 800,278 NER edges and 156,130 of squad's 156,130, with none extra.

**Every read.** The missed golds are those next to a found gold, ranked among the found golds' neighbours
outside the top five. Rank 1 is the best. O1, O1w, C1 and C1w are R@5 against zrc's, over every question with a
gold.

| fit | read | partly found | missed golds ranked | le2 base / words / fused | median rank base / words / fused | missed with a window | missed with words | O1 / O1w | C1 / C1w |
| --- | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- |
| scr-zrct | musique | 1,471 | 770 | 0.225 / 0.294 / 0.261 | 9 / 8 / 8 | 0.956 | 0.600 | -0.0135 / -0.0101 | -0.0099 / -0.0079 |
| scr-zrct | 2wiki | 2,523 | 2,182 | 0.706 / 0.667 / 0.733 | 2 / 2 / 1 | 0.855 | 0.650 | -0.0118 / -0.0115 | -0.0195 / -0.0199 |
| scr-zrct | hotpotqa | 739 | 681 | 0.583 / 0.526 / 0.599 | 2 / 2 / 2 | 0.705 | 0.548 | +0.0027 / +0.0023 | -0.0068 / -0.0076 |
| scr-zrct | metaqa | 2,713 | 2,739 | 0.447 / 0.117 / 0.127 | 3 / 64 / 31 | 0.093 | 0.081 | -0.0646 / -0.0684 | -0.0628 / -0.0666 |
| scr-zrct | webqsp | 319 | 338 | 0.237 / 0.192 / 0.210 | 8 / 10 / 9 | 0.713 | 0.216 | -0.0529 / -0.0552 | -0.0076 / +0.0016 |
| scr-zrct | squad | 0 | 0 | | | | | -0.0264 / -0.0264 | -0.0059 / -0.0062 |
| scr-zrct-hp | musique | 1,411 | 736 | 0.291 / 0.300 / 0.308 | 5 / 6 / 5.5 | 0.969 | 0.594 | -0.0235 / -0.0225 | -0.0143 / -0.0174 |
| scr-zrct-hp | 2wiki | 2,472 | 2,022 | 0.697 / 0.682 / 0.737 | 2 / 2 / 1 | 0.854 | 0.648 | -0.0149 / -0.0146 | -0.0228 / -0.0233 |
| scr-zrct-hp | hotpotqa | 1,573 | 1,444 | 0.549 / 0.564 / 0.578 | 2 / 2 / 2 | 0.701 | 0.533 | +0.0176 / +0.0201 | +0.0102 / +0.0128 |
| scr-zrct-hp | metaqa | 2,695 | 2,692 | 0.454 / 0.111 / 0.123 | 3 / 61 / 30 | 0.091 | 0.081 | -0.0654 / -0.0691 | -0.0624 / -0.0665 |
| scr-zrct-hp | webqsp | 275 | 285 | 0.225 / 0.193 / 0.204 | 8 / 9 / 8 | 0.649 | 0.260 | -0.0594 / -0.0626 | -0.0111 / -0.0038 |
| scr-zrct-hp | squad | 0 | 0 | | | | | -0.0270 / -0.0270 | -0.0076 / -0.0083 |

squad has no partly found question in either fit, so it ranks nothing.

**Reported, not decided:**
- **musique, the words alone.** The words alone rank the missed gold 1st or 2nd more often than zrc's score in
  scr-zrct (0.294 against 0.225), but barely in scr-zrct-hp (0.300 against 0.291). The rule reads the fused order.
- **musique, coverage.** 96% of the missed golds have a window and 60% have a non-zero score, against 41% of all
  candidates. Of the missed golds with words in scr-zrct, 343 take them from NER links and 119 from title links.
- **The two hyperlink datasets.** The fused order is above zrc's score in both fits:
  - 2wiki: 0.733 and 0.737, against 0.706 and 0.697;
  - hotpotqa: 0.599 and 0.578, against 0.583 and 0.549.
- **The KB graphs.** The relation names lead away from the gold. In scr-zrct, le2 falls from 0.447 to 0.127 on
  metaqa and from 0.237 to 0.210 on webqsp. On metaqa, 75% of all candidates carry a question stem in their
  relation names, but only 8% of the missed golds do.
- **The re-orders as retrieval.** Each swaps the top five's last two rows for an anchor's two best neighbours (the
  highest-ranked gold in O1, zrc's top-1 in C1).
  - On musique, all four lose R@5 in both fits, and the fused order moves them by at most 0.0034.
  - The only gains: hotpotqa in scr-zrct (O1 +0.0027, O1w +0.0023), webqsp in scr-zrct (C1w +0.0016), and hotpotqa
    in scr-zrct-hp (O1 +0.0176, O1w +0.0201, C1 +0.0102, C1w +0.0128).

**The first run stopped.**
- The first run (lw-read-zrc) stopped at musique's read. Clean-up 13 had removed step 1's musique s1eval look
  chunks. Their records were kept, and the chunks were marked to be made again.
- It read no words. metaqa and squad passed identity first (R@5 0.7782 and 0.9105, as filed).
- The chunks were made again on the host (lk1r-musique-s1eval-0..3, step 1's lines under new names). The 20
  chunks the laptop still held matched the rebuilt ones array for array.
- D4 then ran again unchanged (lw-read-zrc2), and its records are the ones above.
