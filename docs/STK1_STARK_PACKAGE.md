# STK1: the three STaRK knowledge bases as the eighth, ninth and tenth datasets, packaged and read the same way as the seven

Declared 26 Nov 2026, 22:25 IST, before any STK1 number exists. Code: `outputs/mp_unified/zstk1.py` (a new module, committed with this page before any row is queued; it imports `zenc.py`, `zenc3.py` and `zenc3v.py` and edits none of them). Safe readers for the downloaded files: `scripts/stark/` (`stark_inspect.py`, `stark_extract.py`, `stark_nodes.py`, `stark_sample.py`, `stark_keys.py`, `stark_peek.py`, `stark_find.py`).

## 1. Why this page exists

The user's words (26 Nov):

- "it fits your 'no real KB is pure entity' point best" (on STaRK), "only stark will also solve our problems", "fine do it we will have 10 datasets then", "if those are relevant like stark then do download it is fine" (MAG and Amazon).
- "yeah these sound good just see what all we can do and finish".
- **"deal with all the datasets the same way no matter what"** and then **"all 10"**.

What follows from the last two lines:

1. There is **one** builder, **one** node-text rule, **one** encoder, **one** first-stage read and **one** set of gates for Prime, MAG and Amazon. No dataset gets a lighter read, a different text rule or a different metric. Order of the rows is size only (Prime, then MAG, then Amazon); the code and the arguments other than the dataset name are identical.
2. The three enter the repository's machinery the way MultiHop-RAG did (`docs/MHR1_MULTIHOPRAG_PACKAGE.md`): as a package in the CRAG datasets' layout, so that the links, the pools and the grades that run on the seven run on them unchanged. No walk or pool rule is invented here for STaRK; the first read that would have needed one is STK3 below, which uses the seven's own pool rule.
3. The seven (SQuAD, HotpotQA, 2Wiki, MuSiQue, MetaQA, WebQSP, MultiHop-RAG) are not re-read on this page. Their first-stage and pool rows exist and stay the comparison. What this page adds is the first three rows of the same table for the other three.

What exists and what it leaves open:

- The benchmark (arXiv 2404.13207, v3) has three knowledge bases whose nodes carry text and whose edges carry typed relations: **Prime** (a biomedical KB), **MAG** (a scholarly KB), **Amazon** (a product KB). Its own tables report Hit@1, Hit@5, Recall@20 and MRR (nothing deeper than 20 items); **no published number at a pool of hundreds or thousands of nodes was found**, so the repository's headline (all golds inside the pool at about 2,500 distinct nodes) has no comparator there that this page can cite and is computed here for the first stage itself.
- The downloads are in `outputs/stark/` (gitignored): `prime_dl/`, `mag_dl/`, `amazon_dl/` (the dataset's files from the Hugging Face dataset snap-stanford/stark, CC-BY-4.0). They are untrusted data (section 3).

## 2. What STK1 is and what it is not

STK1 declares and runs, for each of the three, in this order: **A** the package, **B** the e3v encode on the host CPU, **C** the first-stage read (dense and BM25 legs now; the SPLADE leg and the fusion as C2, section 7).

STK1 declares **no link rule, no pool rule and no model**. The next stages are named here and are **not** declared and **not** run:

- **STK2** (links): S = the benchmark's own edges, labelled `S.<relation name>` (the 18 or so relation types), stored by A in `graph/s_edges.npz`; R = the seven's universal rule set where it applies to text nodes; K = the e3v kNN; every edge labelled by its rule, deduplicated S > R > K, as on the seven. Its page declares which of the seven's rules need what from a node (a title, a name, a passage) before any link number.
- **STK3** (pools): the seven's pool rule (PL7's) on the ten, at the same node counts, graded against the first stage's flat top-N at equal distinct nodes (this page's C gives that baseline).
- **STK4** (models): in-domain, one MLP and one GNN per the standing rule, only after STK2's links are verified against the benchmark's answers and STK3 has pools.

## 3. The data and how it is read

| KB | queries (csv rows) | splits (train / val / test) | test-0.1 | files used |
|---|---|---|---|---|
| Prime | 11,204 | 6,162 / 2,241 / 2,801 | 280 | `processed.zip`, `stark_qa.csv`, five index files |
| MAG | 13,323 | 7,993 / 2,665 / 2,665 | 266 | same |
| Amazon | 9,100 | 5,910 / 1,548 / 1,642 | 164 | same |

(Counts are the downloaded files', read by `stark_inspect.py` before this page; gate P1 repeats them from the package build.) The human-written query files (98, 84, 81 questions) are **not** in the package; they are test-only extras and are named for a later page if wanted.

Prime's pinned files (sha256, first 16 hex): `processed.zip` f6f265f60b761784, `raw.zip` bf37a99ff5f3618c (not used by STK1), `stark_qa.csv` a27e5d63da6044e5, `test.index` 424d99e7d0ec209a, `train.index` b828de166d641cc7, `val.index` df7fce5fde2a96bf, `test-0.1.index` 05a94ab75cc2359d. The builder writes the full sha256 of every file it reads into `DATASET.json` (`raw`), for all three; those are the pins.

**Which folder.** Prime and MAG: `processed/` of `processed.zip`. Amazon's zip holds two variants (`processed/` and `processed/cache/brand-category-color/`); only the second holds all six files the builder reads (including `node_types.pt`), so Amazon reads that one. P1 reports both variants' node counts.

**Untrusted files.** Each download is in its own empty directory; every `node_info.pkl` is read only through `Safe` in `zstk1.py` (builtin containers and `OrderedDict`, plus, because Amazon's records hold numpy scalars, exactly `numpy.dtype` and the numpy scalar rebuild; any other global raises); every `.pt` file only with `torch.load(weights_only=True)`; zip members are extracted only by `stark_extract.py` after a member-name check; every Python run that reads them uses `-I`; the scripts live outside the data directories.

**Design look before this page (disclosed).** The key lists of Prime's `node_info.pkl` per node type (`stark_sample.py`, `stark_keys.py`: which keys, how many nodes carry text, mean lengths) and the first records of MAG's and Amazon's pickles (`stark_peek.py`, `stark_find.py`: the key names of one author record, one paper record, one product record, read without unpickling) were looked at to write T1. No query, no answer id and no retrieval number was read.

## 4. The node text: one rule, T1, for every node of every KB

A node's record is the dictionary in `node_info.pkl` (Prime: `id`, `type`, `name`, `source`, `details`; MAG and Amazon: flat dictionaries). Its type name is the benchmark's own (`node_types.pt` through `node_type_dict.pkl`).

**T1.** The text is `<type>: ` followed by the record's fields, flattened, cut at **1,200 characters**:

1. The record's own `type` key is left out (the head carries it).
2. A string is kept if, after collapsing whitespace, it is not empty, is not `nan`, holds at least one letter, and has not already been kept in this node (first in stored order wins). Numbers, booleans and strings without a letter (prices, dates, ids made of digits) are dropped.
3. A list keeps its items' order, joined by `; `.
4. A dictionary's children are deduplicated in stored order and then **ordered by their flattened length, shortest first** (ties in stored order), joined by `; `. This applies at every level, so names, titles, brands and categories come before abstracts, descriptions and reviews, whatever order a dataset stored them in.
5. The result is `<type>: <body>`, or `<type>:` if nothing is kept, cut at 1,200 characters.

No field name, no per-type list and no per-dataset choice appears in the rule. It is cheap on purpose: with e3v's window of 8,192 tokens no text is cut by the encoder, and the same cap bounds the cost of Amazon's long reviews.

**Beside, not a rule.** The benchmark's own documents very likely also carry relation text (its paper's Fig. 6 caption suggests it; the paper does not say). T1 does not: the S edges are STK2's. So T1's first-stage numbers are not a reproduction of the published rows (section 7 reads the difference rather than assuming it).

## 5. The package (A)

Root `outputs/stark/package/`, one folder per KB (`stark_prime`, `stark_mag`, `stark_amazon`), gitignored, in the CRAG datasets' layout:

- `nodes.jsonl`: line i is the benchmark's node i. Fields `node_id` = `<ds>:<i>` (the hash ids of the six cannot be used: many KB nodes share a text), `text` (T1), `type`, `kb_index`, `content_hash`.
- `queries/<split>.jsonl` and `queries/query_ids.json`, splits `train`, `dev` (the benchmark's val), `test`, **the benchmark's own index files, each in file order**; `query_id` = `<ds>_q<csv id, five digits>`; `question` = `query` verbatim; `gold_node_ids` = the answer ids as node ids, in the file's order; `n_golds`; `source_position`. Every read here is zero-shot: no model of record has seen any of these questions, and nothing trains on the benchmark's train split on this page.
- `embeddings/dense/{docs,queries}/manifest.json` (rows only: STaRK has no frozen-arm vectors; e3v's parts and fusion live where zenc puts every encoder's vectors, `outputs/enc/<arm>/<ds>/`).
- `graph/s_edges.npz` (`src`, `dst`, `rel` per edge of the benchmark's `edge_index`) and `graph/s_rel.json` (relation id to name). Nothing reads them in STK1; STK2 does.
- `DATASET.json` (record `STK1_PACKAGE`, counts, files' sha256, the split rule, `raw` pins) and `outputs/stark/package/PACKAGE_FREEZE.json` (one `dataset_json_sha256` per KB). A package is built once; the builder refuses an existing one.

**Gates (each stops the build):**

- **P1** (consistency): `node_info` has keys 0..N-1; `node_types` has N rows and every type id has a name; `edge_index` is 2 x E, `edge_types` has E rows, every endpoint is in 0..N-1, every relation id has a name; the three split index files are disjoint and their union is exactly the csv's rows; `test-0.1` is inside `test`; every question has at least one answer and its answers are in range (a node named twice in one question's list counts once: amendment A1, section 5.1).
- **P2** (the text rule): `zstk1.py selftest` passes on the rule's toy cases and on `Safe`'s allow and block cases (it must pass on the host as on the laptop before the build); the share of nodes whose text is the head alone, and the share at the cap, are recorded.
- **P3** (recorded, no gate): the types of the answer nodes.
- **P4** (the readers): zenc's `query_texts` returns every question row and `doc_chunks` returns the node texts in order, equal to the builder's.
- **P5** (the S store, recorded, no gate): edge count, distinct pairs, self loops, share of edges whose reverse is also present, edges per relation.

### 5.1 Amendment A1 (26 Nov, 22:26 IST, after P1 stopped Prime's build and before any package, vector or retrieval number)

- **What happened.** P1, as first written, stopped on "36 questions with an answer out of range or repeated". The diagnosis (`answer_ids` read, nothing else): none is out of range; 36 of Prime's 11,204 questions (train 23, dev 7, test 6) name the same node more than once in their answer list (for example a list of six ids with three distinct). No package file was written.
- **Why the rule changes.** A question's gold set is the set of **distinct** answer nodes, on the seven as here (R@k and ALL@k divide by distinct golds). A repeat inside a list is the benchmark's data entry, not a second answer. Treating it as damage would drop 36 questions from Prime and, on the other two, an unknown number, for a reason unrelated to retrieval; and counting the repeat twice would change R@20's divisor.
- **What changes.** `read_qa` collapses an answer list to its distinct ids in first-seen order. `n_golds` is the distinct count. The out-of-range check stays a stop. `package.json` and `DATASET.json` record, per split, the questions with a repeat and the answer ids dropped (`answer_repeats_a1`), so the effect is visible for all three KBs. Nothing else in the rule, the gates or the legs changes; no question is dropped.
- **Beside, not a rule.** The benchmark's own evaluator may divide by the listed length. The published rows are therefore read with the caveat that, on at most 36 of 11,204 Prime questions, the two divisors differ.

## 6. The encode (B)

e3v only (D101: gte-modernbert-base and granite-embedding-small-english-r2 at `zenc3.NEW3`'s pins, fused by `zenc3v.fuse` at weight 0.5, 1,152 wide, float16 unit rows, no prompts, window 8,192). The frozen arm is the six's comparison arm and has no STaRK rows.

- **Device**: the host CPU, float32 weights, 8 threads per row, a token budget of 8,192 per batch (as MHR1's section 3.1). The card is hidden before torch loads. The user's rule (21 and 25 Nov): the GPU is not ours; "cant we just encode with the cpu we have on the host?" (26 Nov). No GPU row is queued.
- **Checkpoints** (the user: a loss is minutes): rows are encoded in blocks of 4,096 and each block is saved as soon as it is done (`shard_<s>.blk<b>.npy`); a restart skips finished blocks. A shard of 40,000 rows (zenc's size) is assembled from its blocks, checksummed, and its blocks removed (this process's own scratch). The manifests are zenc's format, so `zenc3v.cmd_vectors` and the readers accept them unchanged.
- **Gates**: no damaged row (zenc's rule), no text cut at the window, rows equal nodes and questions, the parts' manifests pin the revisions.
- **Smoke first** (the standing subset-first rule): `encode --smoke 2048` times 2,048 nodes of shard 0 per part and writes nothing; it prices the full rows and decides nothing else.
- Each row records seconds and tokens; plans are re-priced from the smoke, not from the estimate below.

## 7. The first stage (C): legs, metrics, comparators, calls

For every test question, exact float32 top **2,500** over all nodes for each leg:

- **dense_e3v**: inner product of e3v's question and node vectors, ties to the lower id (zenc's `search`).
- **bm25**: Okapi BM25 over T1's texts, lowercase `[a-z0-9]+` tokens, k1 = 1.5, b = 0.75, Lucene-style idf, positive scores only, ties to the lower id. This leg exists to check T1 against the published BM25 row (G-B); it is not part of the seven's first stage.
- **C2, the seven's own first stage**: the record's SPLADE contract (naver/splade-cocondenser-ensembledistil revision 49cf4c7b, 256 tokens, batches of 64, log1p-relu max-pooled), then the fusion of dense and SPLADE by the record's reciprocal-rank rule (1/(60 + rank), float32). The model is on the laptop only, so the leg runs on the laptop CPU as MHR1's did; its code is added to `zstk1.py` as a `splade` subcommand under a dated amendment on this page before the first SPLADE number. Its question and node counts are the same per KB as the other legs.

**Metrics (per leg, all questions of `test`, mean in percent with a 95 percent bootstrap interval over questions, 1,000 resamples, seed 0):**

- the benchmark's four: **Hit@1**, **Hit@5**, **R@20** (answers found in the top 20 divided by the answers, averaged over questions), **MRR** (reciprocal rank of the first answer, read inside the top 2,500; the benchmark reads it over the whole ranking, a difference below 1/2,500 per question);
- the repository's: **R@k** and **ALL@k** (every answer inside the top k) at k = 1, 5, 20, 100, 1,000 and **2,500**. ALL@2,500 is the flat baseline STK3's pools must beat at equal distinct nodes; the target there is 0.95.

**Published comparators (beside).** The benchmark's Table 6, "Full Testing Dataset", arXiv v3, Hit@1 / Hit@5 / R@20 / MRR:

| | Prime | MAG | Amazon |
|---|---|---|---|
| BM25 | 12.75, 27.92, 31.25, 19.84 | 25.85, 45.25, 45.69, 34.91 | 44.94, 67.42, 53.77, 55.30 |
| ada-002 | 12.63, 31.49, 36.00, 21.41 | 29.08, 49.61, 48.36, 38.62 | 39.16, 62.73, 53.29, 50.35 |
| multi-ada-002 | 15.10, 33.56, 38.05, 23.49 | 25.92, 50.43, 50.80, 36.94 | 40.07, 64.98, 55.12, 51.55 |

(Prime's rows were read by this session from the paper's HTML; MAG's and Amazon's were read from the same table by a delegated reader and not re-read by this session; every comparison below says so.) Trained systems (mFAR, MoR, GraphRAFT, GRASP, KAR and others) are a different class and are not compared here; none of these rows uses the benchmark's train split and none uses a language model.

**Calls.** For each leg and each published row, each of the four metrics: ABOVE if the leg's 95 percent interval lies above the published point, BELOW if below, LEVEL otherwise, with the difference in points beside. The published point's own noise is not modelled, and the published documents differ from T1 (section 4): these calls are readings beside, **not** claims of reproduction or of superiority.

**G-B (T1 against the published BM25 row).** With d = the larger of |Hit@1 - published| and |R@20 - published| of the bm25 leg: **REPRODUCES** if d is at most 3 points, **CLOSE** if at most 8, **DIFFERENT** otherwise. G-B stops nothing: it says how far T1's documents are from the benchmark's, and the reason is read from the text statistics, not assumed.

## 8. Expectations, written before any number

- **X1** P1, P2, P4 pass on all three; P5 shows many edges per node on Prime (the benchmark's own statistic is about 125 per node) and reciprocal edges for most of them.
- **X2** T1's cap binds on a minority of Prime's nodes (more than 5 percent, fewer than half: six of ten node types have a name and no text) and on most of MAG's papers and Amazon's products.
- **X3** G-B on bm25: Prime CLOSE or REPRODUCES; MAG CLOSE or REPRODUCES (title and abstract fit the cap); **Amazon DIFFERENT** (reviews and questions beyond the cap, relation text in the published documents). A wrong guess here is a finding about T1, not a reason to change it.
- **X4** dense_e3v is above bm25 on R@20 on Prime and MAG (the published ada-002 is above BM25 there by 4.75 and 2.67 points) and level or below on Amazon (published: 53.29 against 53.77).
- **X5** ALL@2,500 of every leg is **below 0.95 on Prime and MAG**; nothing is predicted for Amazon. If a leg reaches 0.95 at 2,500 on a KB, STK3's pool has nothing to add there at that size, and that is reported plainly.
- **X6** Cost: the smoke re-prices the encode. The planning ceiling at 8 threads is **Prime within 4 hours, MAG within 3 days, Amazon within 4 days**; if the smoke says a KB exceeds its ceiling, this page gets an amendment on the thread count or the shard order **before** that row is queued. The rule (T1, the encoder, the legs) does not change with the cost.
- **X7** The user's standing point (no real KB is pure entity) is what Prime, MAG and Amazon test: nodes that carry text. If dense_e3v beats BM25 by more than it does on MetaQA's pure-entity graph, that is the first evidence in the repository for it; it is not a call on this page.

## 9. Code and compute

- `zstk1.py selftest | package | encode | vectors | first`; every subcommand takes `--kb`; `package` takes the two folders as arguments (no path is written in the code).
- Prime's package is built on the laptop (the pickle is 70 MB). MAG's and Amazon's pickles are 1.2 GB and 5.4 GB, so their packages are built on the host from the pushed zips (each extracted into a new empty directory by `stark_extract.py`, only the members the builder reads for Amazon).
- Rows go through the feeder with per-row class names, 8 CPUs each, and the host's admission rule; the encode rows are one KB at a time unless the host is otherwise idle. No delete is made: the scratch blocks are the process's own, removed after their shard is checksummed.
- Outputs: `outputs/stark/<kb>/package.json`, `first.json`, `first.npz` (gitignored); this page's section 10 is filled from them.

## 10. Result

(empty until the rows are read)
