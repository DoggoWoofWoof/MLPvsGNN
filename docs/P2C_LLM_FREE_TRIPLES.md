# P2c: LLM-free triples in place of an LLM-built graph (part 1, the calibration on B1)

Declared 13 October 2026, about 10:15, before any number of the stage. P2's third stage in
[PROGRAM_2026_10_13.md](PROGRAM_2026_10_13.md), from the user's direction of 13 October (about 08:50): "the links shud
try to get gains for creating llm based graphs or edges or triplets without actually using llm, so try and see how much
can we approx". It builds graphs and reads walks: no model trains, nothing is adopted.

## 1. Question

HippoRAG 2 builds its graph with an LLM. For each passage, the LLM lists named entities, then (subject, relation,
object) triples. The graph's nodes are the passages and the triples' phrases. Its edges are:
- the triples;
- passage-phrase mentions;
- synonym edges between phrases, from phrase embeddings.

Our three link families (structural, ner, knn) use no LLM.

The questions:
- Can a parser and fixed rules build the triples an LLM builds?
- How much of the LLM graph's gain does that graph recover, in the same walks?

**Part 1 (this file) is the calibration.** It runs on B1's musique and 2wiki passages, where LLM-built extractions of
the same passages exist. B1 is a calibration line, so no rule here is chosen on B1's questions or golds.

**Part 2 (its own file, next)** puts the same construction on our four passage corpora in full as a fourth link
family, read by the pool walks as P2a part 1 read the three families. The corpora are 2wiki 6.0M passages, hotpotqa
5.2M, musique 118k and squad 20k. Their parse runs on the host during part 1 (section 8) and gives no number.

On metaqa and webqsp the graph is already a curated triple graph (the KB), and node texts are names only. There is
nothing to parse, so they are not in part 2.

## 2. The references, built with LLMs and read as released

| setting | reference | what it holds per passage |
| --- | --- | --- |
| musique | **H4o**: HippoRAG 2's OpenIE with gpt-4o-mini, the released graph's extraction | named entities, triples |
| musique | **H70**: the same with Llama-3.3-70B-Instruct (beside) | named entities, triples |
| musique, 2wiki | **SR**: SiReRAG's propositions (beside on musique, the only reference on 2wiki) | propositions, each with its entities |

- **Where they are.** The CRAG repo's `baselines/repos` (HippoRAG `outputs/musique/openie_results_ner_*.json`,
  SiReRAG `*_kg.json`). That repo is read-only. The files are never committed, and neither is anything that copies
  their phrases.
- **Each reference is matched to B1's passages, position by position:**
  - the titles are equal in normal form;
  - the texts are equal in normal form.

  The files carry U+FFFD where the source had non-ASCII characters. B1's corpus is the same release file, and the normal
  form treats U+FFFD as a separator.
- **H4o and H70 as graphs,** as HippoRAG 2 builds them (`add_fact_edges`, `add_passage_edges`):
  - the phrases are the triples' subjects and objects;
  - each passage mentions its triples' phrases;
  - the relation edges are the triples.

  A triple without exactly three non-empty fields is skipped, as HippoRAG 2 skips it. A triple whose two ends are one
  phrase adds the phrase and no edge. The counts (triples kept, skipped, one-phrase; SR's propositions) are recorded.
- **SR as a graph:**
  - the phrases are the propositions' entities;
  - each passage mentions them;
  - a relation edge joins each pair of entities in one proposition.

## 3. Our construction (version 1, fixed here)

spaCy 3.8.15 with en_core_web_sm 3.8.0: a CNN pipeline (tagger, dependency parser, NER, lemmatizer), with no
transformer. It is run over each passage's text, and the passage's title is one of the passage's phrases.

**Phrases.**
- the title, without a trailing parenthesis ("Green (band)" becomes "Green");
- named entities of every label except the numeric ones (CARDINAL, ORDINAL, QUANTITY, PERCENT, MONEY);
- noun chunks whose root is a noun or proper noun, at most 8 tokens, without leading determiners and possessive
  pronouns.

**The normal form.** It applies to every phrase, ours and the references', and to the questions:
- lower case, with "'s" removed;
- every non-word character becomes a space, and whitespace is collapsed;
- a leading "the", "a" or "an" is removed;
- phrases shorter than 2 characters are dropped.

**Within a passage (string rules, no embedding):**
- **Pronouns.** A pronoun argument (he, she, it, they, him, her, them, his, its, their) is the title.
- **Surnames.** A one-word phrase equal to the last word of exactly one longer PERSON entity of the passage is that
  entity. The title counts when it is itself a PERSON entity of the passage.
- **Acronyms.** An all-capital token in parentheses right after a phrase, spelling the initials of that phrase's
  capitalized words, is that phrase. A leading "the", "a" or "an" is not part of the phrase (the normal form), so "The
  National Basketball Association (NBA)" spells NBA.

**Triples, from the dependency parse:**

| pattern | triple |
| --- | --- |
| a verb with a subject (nsubj, nsubjpass and their conjuncts) and an object (dobj, attr, oprd, dative and conjuncts) | (subject, verb lemma + particle, object) |
| the same verb with a prepositional object (prep or agent, then pobj, and conjuncts) | (subject, lemma + preposition, object) |
| a verb heading a relative clause or a participle (relcl, acl) whose subject is missing or a relative pronoun (who, whom, which, that) | the noun it modifies is the subject |
| an apposition | (head, "be", apposition) |
| a noun with a prepositional object, when the two are different phrases | (noun, preposition, object) |
| a possessive that is not a pronoun | (possessor, "poss", possessed) |

- An argument is the phrase that holds its head token; entities come before noun chunks.
- If no phrase holds the head token, there is no triple.
- Self-loops and repeats within a passage are dropped.

**Our graph.** The phrase nodes are every phrase above, not only the triples' arguments. HippoRAG 2 keeps only the
triples' phrases. A rule extractor finds fewer triples, so ours keeps the entities and chunks too.

**Later versions.** Version 1 is fixed here. A later version may change these rules. It is tuned only on agreement
with the references' phrases and pairs (section 5), never on B1's questions or golds. It is read once, in its own file.

## 4. Two forms of each graph

| form | what | who reads it |
| --- | --- | --- |
| **P**, a passage family | passages that share a phrase are linked when the phrase is in 2 to 25 passages; the weight is the sum of 1/df over the shared phrases | our pools and models take families in this form. It is the ner family's rule (`b1_build.ner_edges`), with the graph's phrases in place of spaCy's entity keys |
| **H**, HippoRAG 2's graph | passages and phrases are nodes, undirected; a passage-phrase mention weighs 1; two phrases are joined with twice their number of triples (HippoRAG 2 stores each triple in both directions and each direction is an edge); self-loops are not edges | HippoRAG 2's search (WH below) |

The H form has no synonym edges: HippoRAG 2's come from phrase embeddings, and this program uses none.

**The graphs read** (X below):
- the references (H4o, H70 and SR on musique; SR on 2wiki);
- **ours**;
- **NER**: today's ner keys (the ner family's nine labels, as it was built), as phrases with mentions only;
- **ours-noREL** and **X-noREL** (beside): a graph without its relation edges, so mentions only.

## 5. The reads

### Agreement

Read per setting and reference, on normal forms, summed over passages (micro):

| measure | recall | precision |
| --- | --- | --- |
| phrases | the reference's phrases found among ours | ours found among the reference's (its phrases and, for H4o and H70, its entity list) |
| pairs | the reference's relation pairs (unordered, both ends phrases) found among ours | ours found among the reference's |
| P edges | the reference's P-form passage pairs found among ours | ours found among the reference's |

Beside: the same for NER's keys, and for ours without the triples (pairs empty).

### Walk gains

The measure is R@5; R@2 and R@10 go beside. Every walk runs on B1's 1,000 questions per setting.

**WP, B1c's walk.**
- B1c's PageRank: alpha 0.5, 50 steps, the T5 and RW restarts from the RRF list.
- It runs over today's three families, and over today's three plus X's P form.
- gain(X) = R@5(today + X) − R@5(today). It is read per restart.

**WH, HippoRAG 2's search with an LLM-free link.**
- **The walk:** PageRank on X's H form, damping 0.5 (a restart of 0.5 each step), 50 steps.
- **The reset, before it is normalized:**
  - every passage: its dense cosine, min-max normalized over the corpus, times 0.05 (HippoRAG 2's
    `passage_node_weight`);
  - X's phrases whose normal form is a word n-gram (n up to 8) of the question's normal form, each weighing 1/(the
    number of passages it is in);
  - only the five largest phrase weights (HippoRAG 2's `linking_top_k`). Ties go to the longer phrase, then the earlier
    one in the question.
- **No phrase matched:** the ranking is the dense list. HippoRAG 2 does the same when no fact survives.
- **The ranking:** passages by score, ties by RRF rank (B1c's `rank_by`).
- **The base** is the same search on NER's graph. gain(X) = R@5(X) − R@5(NER).
- **Beside:** each X-noREL, and the dense list.
- **What WH replaces in HippoRAG 2's search:**
  - its match of the question to triples by triple embeddings;
  - the LLM's filter of those triples.

  The link here is the string match. HippoRAG's first version linked the question's entities to phrases the same way,
  with phrase embeddings where this uses string equality.

### The share

For each walk (WP-T5, WP-RW, WH), setting and reference:
- share = gain(ours) / gain(reference);
- its interval comes from a paired bootstrap over questions (2,000 resamples, seed 0, 95%), as the ratio of the
  resampled mean gains.

| call | when |
| --- | --- |
| NO_REFERENCE_GAIN | the reference's gain is within the tie (0.002), or its interval reaches 0. The share is not read: in that walk, the LLM graph adds nothing over today's links |
| MATCHED | ours − reference is not −1 by P3a's rule (below the tie with the interval below 0) |
| PARTIAL | ours is below the reference, and our own gain is above the tie with its interval above 0 |
| NONE | otherwise |

## 6. What it decides

Part 1 reads, and adopts nothing.
- **Part 2 runs whatever part 1 reads.** Our corpora are the target and B1 the calibration. Part 2 (its own file)
  reads version 1's P form as a fourth family in the pool walks.
- **A reference gains and ours is PARTIAL or NONE:** the agreement table says what to imitate next (phrases or pairs).
  Version 2 is tuned on agreement only.
- **No walk shows a reference gain:** on B1 musique, the 74.7 against 66.3 gap to HippoRAG 2 is not its graph's, in
  our walks. What remains is outside the graph:
  - HippoRAG 2's question-to-fact link, which embeds every triple (finer than a node, so not added here);
  - its encoder: NV-Embed-v2's dense R@5 is 69.7 against our 56.7, a named exposure difference.

## 7. The checks, first

Nothing is read unless all of these hold:
- **References:** each matches B1's passages position by position (titles and texts in normal form). Its passage and
  triple counts are recorded.
- **Questions:** B1's `queries.json` golds; the raw file's questions in the same order (ids equal).
- **The dense scores:** the package's vectors for B1's rows give B1's `first_stage.npz` dense list exactly (top 1,000,
  same order). So WH's reset scores are B1's.
- **WP's base:** today's three families give B1c's filed R@5 for T5/all and RW/all (`b1c.json`, 4 decimals).
- **The P rule:** NER's keys, through the P rule, give `graph_ner.npz` exactly (the same edges; weights within 1e-6).
- **The walks:** every reset sums to 1, and every ranking holds at least 10 distinct passages.
- **Provenance:** one script sha256 across parse, refs and run, and no `--limit` in a filed record.

## 8. How it runs

**Code:** `scripts/p2c_triples.py`. It imports, unchanged, B1c's walk functions (`b1c_walks`) and B1's vector readers
(`b1_build`).

| command | what | writes |
| --- | --- | --- |
| `parse <setting>` | our construction and NER's keys on a B1 setting | `outputs/p2c/parse_b1_<setting>.npz` and `.json` |
| `parse <dataset> --part k/n --host` | the construction on one part of a full corpus (part 2's input, no number) | `outputs/p2c/parse_<dataset>_<k>of<n>.npz` and `.json` |
| `refs <setting>` | the references in the same format | `outputs/p2c/ref_<setting>_<ref>.npz` (never committed) |
| `run <setting>` | the checks, agreement and walks | `outputs/p2c/b1_<setting>.json` |
| `verdict` | the shares and calls | `outputs/p2c/verdict.json` and `.md` |
| `--selftest` | below | nothing |

**Each parse file holds:**
- a phrase vocabulary;
- each passage's phrase ids;
- its triples (subject, relation, object ids);
- a relation vocabulary.

Part 2 merges the parts' vocabularies. The relation names are also P5's: relation names on graphs whose edges have
none.

**The self-test covers:**
- the normal form;
- every triple pattern, the three string rules and the argument rule, on sentences parsed by spaCy (the expected
  triples worked by hand from the parse);
- the P rule against a hand-worked case, and against `b1_build.ner_edges` on made-up passages;
- the H walk against a dense power iteration on a small graph;
- the reset (top five, ties, no match falls back to the dense list);
- the reference readers (skipped triples, pairs within a proposition);
- the share's calls on made-up gains;
- a parse part written and read back.

**Development runs, no number.** `--limit N` parses or reads the first N passages or questions, writes nothing, and
logs only that every step ran and the checks held. Before this file was committed:
- the self-test passed;
- `parse musique --limit 300`, `refs musique --limit 300` and `refs 2wiki --limit 300` ran (every reference matched
  B1's passages);
- `parse 2wiki --part 0/60 --limit 20000` ran (a full corpus's part, on the laptop's copy);
- `run musique --limit 50` and `run 2wiki --limit 50` ran, and their checks held. WP's walks always run on all 1,000
  questions, since B1c's check needs them; WH's ran on the 50.

**Laptop.** spaCy parses about 5,900 to 6,800 words a second on one laptop process: 20,000 passages of 2wiki took 299
and 260 seconds in two runs beside other jobs, with a peak of 0.86 GB. B1 musique's parse, with NER's keys, takes about
4 minutes, and 2wiki's about 2. In the development runs, WP's walks took about 4 minutes on musique and 1 on 2wiki, and
WH's nine graphs took 35 and 9 seconds on 50 questions. So a setting's full run takes about 15 minutes, and the verdict
should land about 45 minutes after the commit.

**Host (part 2's input, no number):** items `p2c-read-parse-<dataset>-<k>of<n>`, light lane, 1 CPU each, from the
host's verified mirror. The host runs spaCy from `outputs/host_ops/pylib/spacy3815_cp313_win_amd64.zip` (never
committed). One item, `p2c-copy-spacy`, unpacks it once (`unpack-spacy`, which then parses a test sentence), and the
parse items wait for it. A parse refuses to run when the unpack is missing or came from another zip. The names put the
items in the feeder's light lane (its `--light-re`, for items that write only MBs), which keeps running while the
host's drive is below the 30 GB floor (24 GB free when they were queued): from B1's parse files (about 3 bytes a
word), a part writes at most about 28 MB and the 103 parts about 2.5 GB in all; the unpacked spaCy takes 147 MB.

The sizes come from the laptop's slower measured speed and the mean words per passage (one line in 50 of each
corpus). Each part takes 25 minutes or less at that speed. The memory is the 0.86 GB peak of a 20,000-passage part,
with room for parts five to six times its size:

| dataset | passages | mean words | CPU-hours | parts | minutes per part | memory per part |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| musique | 117,534 | 73.4 | 0.4 | 2 | 12 | 2 GB |
| squad | 20,233 | 115.5 | 0.1 | 1 | 7 | 2 GB |
| hotpotqa | 5,233,329 | 46.0 | 11.3 | 40 | 17 | 3 GB |
| 2wiki | 5,989,847 | 88.8 | 25.0 | 60 | 25 | 3 GB |

About 37 CPU-hours in all: about 2 hours on 20 of the host's CPUs, if they run at the laptop's speed.

The parse's records hold counts and seconds only. Part 2 measures before it decides anything.

The encoder stays frozen. No LLM or transformer is called, no model trains, and test splits are never read.
