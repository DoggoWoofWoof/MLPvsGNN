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

### 5.2 Amendment A2 (26 Nov, 22:39 IST, after Prime's package and smoke, before any encode, vector or retrieval number)

- **Prime's package is built** (P1, P2, P4 pass; P3 and P5 recorded in `outputs/stark/prime/package.json`): 129,375 nodes, 11,204 questions (train 6,162, dev 2,241, test 2,801), 8,100,498 S edges in 18 relations, 36 questions with a repeated answer node (A1). The text statistics are in section 10 when the rows are read. The package files are on the host and its hash of `zstk1.py` equals the laptop's.
- **The smoke** (2,048 nodes, the first of shard 0, 8 threads, other rows running): gte-modernbert 271.7 s (7.5 nodes/s, 1,180 tokens/s), granite-small 83.0 s (24.7 nodes/s, 3,866 tokens/s), 320,779 tokens (157 per node: these first nodes are gene/protein records, among the longest), no text cut at the window, longest 445 tokens. At that rate a node costs 0.174 s over both parts; 129,375 nodes would take 6.2 h **if every node were this long**. Prime's median text is 73 characters, so the real time is lower; the row's own block timings price it, and the 4 hour ceiling of X6 is read from them, not from the smoke.
- **MAG's and Amazon's rows are queued behind their packages without a separate smoke.** The encode is resumable by 4,096-row block and a block is the same whatever the thread count, so a wrong guess costs nothing already done: after MAG's first blocks the row's own per-block seconds give the projection. If that projection is above X6's ceiling, a further amendment changes the thread count or splits the shards across rows **before** the next block is encoded, and the finished blocks stay valid. The rule (T1, the encoder, the legs) does not change with the cost.
- **`scripts/stark/stark_extract.py` takes an optional member prefix** (only the members under it are extracted; the name check still covers every member of the zip). Amazon's zip holds the same six files twice (`processed/` and `processed/cache/brand-category-color/`, 5.2 GB each pickle); only the second holds all six (section 3), so only it is extracted, from the zip on the host, into a new empty directory. No other behaviour of the script changes.
- **Queueing.** One chain per KB in the host's feeder, rows `stk1p-*`, `stk1m-*`, `stk1a-*`: package (MAG and Amazon only; Prime's was built on the laptop), encode, vectors, first stage. Size only orders them (Prime, MAG, Amazon); the arguments differ in the KB name and the memory ask.

### 5.3 Amendment A3 (26 Nov, 23:09 IST, after the SPLADE module was written and self-tested, before any SPLADE number)

- **What changes.** Section 7 said the SPLADE leg would be a `splade` subcommand of `zstk1.py`. It is instead its own declared module, `outputs/mp_unified/zstk1s.py` (`selftest`, `splade --kb prime|mag|amazon [--threads N] [--smoke N]`), because the host's rows run `zstk1.py` and a file that rows run is not edited while they run. `zstk1.py first --splade-top FILE` is unchanged: it already reads the file this module writes and adds the legs `splade` and `rrf_e3v_splade` (1/(60 + rank), float32, ties to the lower id, as in the record). Nothing else in section 7 changes.
- **The contract is the record's** (`zmhr.py`'s constants): naver/splade-cocondenser-ensembledistil at revision 49cf4c7b0db5b870a401ddf5e2669993ef3699c7, 256 tokens, batches of 64 taken in file order and padded to the longest in the batch, `log1p(relu(logits))` max-pooled over every position, padding included. The selftest checks the merge (ties, empty ranks), the sparse-row conversion and that `zmhr`'s constants are the ones used; it passed.
- **How it scores.** The test questions only (the `test` rows of `query_ids.json`). Nodes are encoded in blocks of 2,048; each block's sparse vectors are multiplied against the questions' SPLADE matrix and only the running top 2,500 per question is kept (positive scores only, ties to the lower node id, by a stable merge). No per-node vector is stored. The state (about 56 MB for Prime) is written atomically after every block and a restart resumes there, so a loss is one block, about 25 minutes at the laptop's rate. Outputs: `outputs/stark/<kb>/splade_top.npz` (`ids`: one row per question of `query_ids.json`, the test rows filled, the others -1) and `splade_top.json` (revision, tokens, seconds, code and file hashes).
- **Where it runs.** On the laptop CPU, 6 threads, lowered priority, started as `python -E -P` and not `-I`: the library `packaging` lives in the user site, which `-I` hides. `-E -P` still ignores the environment and puts neither the working directory nor the script's folder on the path; the module reads only the package's own JSON and the pinned model's files with `local_files_only`. The model is the laptop's existing copy; it is never downloaded again and never copied out of the hub cache.
- **What it costs, measured.** The smoke on Prime's first nodes (the longest ones) gave 1.4 nodes/s and 234 real tokens/s at 6 threads. Prime's 129,375 nodes are therefore **at most about 26 hours**, and MAG's 1,872,968 nodes about **15 days** at the same rate: MAG and Amazon cannot run on the laptop. The host has no copy of the model. Putting one there is a 440 MB download of a public model at the pinned revision and waits for the user's explicit OK; even with it, each of the two would be a multi-day CPU row on a host that is already running the encodes and the other session's rows.
- **Decision on the legs.** The rule is the user's ("deal with all the datasets the same way no matter what", "all 10"): the same first stage on all three, order following size only. Prime's SPLADE row starts on the laptop now. MAG's and Amazon's SPLADE rows are named and priced here and **not queued**; until the user decides, their first stage is read on dense_e3v and bm25 only, and every table says that the splade and rrf columns are absent for them and why (cost). That is a gap in what is read, not a different rule.

### 5.4 Amendment A4 (26 Nov, 23:45 IST, after the BM25 leg was removed from `zstk1.py` and its selftest passed, before any first-stage number)

The user, 26 Nov: **"why do we need bm25 etc our best are splade and dense do not do any useless work"**.

- **What changes.** There is no BM25 leg and no G-B. The legs are dense_e3v on all three KBs, and splade and rrf_e3v_splade where the SPLADE leg exists (Prime now). In `zstk1.py` the function `bm25_top`, the bm25 leg, the G-B block and the BM25 check of the selftest are deleted (one insertion, 42 deletions) and `first` no longer reads the nodes' texts; nothing else in it changes. The selftest passes (13 checks). No row has run `first` yet, so no number is affected.
- **What it supersedes.** Section 2's "(dense and BM25 legs now ...)", the last bullet of section 5.3 ("dense_e3v and bm25 only" becomes "dense_e3v only"), section 7's bm25 bullet and the G-B paragraph, X3, the bm25 half of X4, and X7's comparison with BM25 (where SPLADE exists, the comparison X7 wants is dense_e3v against splade). The earlier text stays as the record of what was declared.
- **What stays.** The benchmark's published rows, its BM25 row among them, stay as readings beside: each leg against each printed row with the interval call of section 7. They are numbers from the paper; nothing is run for them.
- **What is lost, said plainly.** G-B was the only check of how far T1's documents are from the benchmark's. Without it, a gap to a published row cannot be split into T1 and the encoder. The first-stage metric code is covered by the selftest on toy rankings only. The 100-question BM25 look made on Prime before this amendment (for plumbing, disclosed in section 10's note) stays disclosed and is not used.
- **What it saves.** BM25 builds a word-count matrix over every node's text and scores it against every test question, per KB; its cost on MAG and Amazon was never measured, so no hours are claimed.
- **What it leaves.** Until the SPLADE decision for MAG and Amazon (a 440 MB download of the public model on the host, waiting for the user's OK, and a multi-day CPU row for each), their first stage is the single dense_e3v leg, and every table says that splade and rrf are absent for them and why.

### 5.5 Amendment A5 (27 Nov, 02:58 IST, after Prime's SPLADE leg stopped on its first checkpoint after a resume, before any SPLADE number)

- **What happened.** The leg was stopped by the harness for low memory (its last save, 20,480 of 129,375 nodes, is from 01:57) and restarted at 02:36 from that checkpoint. It ended at 02:52 with `PermissionError: [WinError 5]` on `os.replace` of the checkpoint, at block 22,528.
- **Cause.** The resume path of `zstk1s.py` opened the checkpoint with `np.load` and kept the file object open for the whole run; Windows refuses to replace an open file. A first run never resumes and has no open file, which is why the saves before the restart worked. It is the same fault SPL1's A6 found in `zspl1.py`.
- **Code only.** The resume path reads the checkpoint inside a `with` block and copies `top_s`; nothing else in the module changes (selftest passes). The contract, the lists and every pass line of section 5.3 are as before. No SPLADE number has been read.
- **The saved state.** The failed replace left the newer checkpoint complete in `splade_state.tmp.npz` (22,528 nodes; sha256 starts 4ebcfaf3900dd4c2; the same `n_nodes` 129,375 and `nq` 2,801 as the older file, the same array shapes, every first-rank score at least the older file's). It was renamed over `splade_state.npz` once, by hand, so the restart resumes at 22,528 and not at 20,480.

### 5.6 Amendment A6 (27 Nov, 03:19 IST, after the original's weights passed SPL1's gate G-C0 on the host, before any SPLADE number of a STaRK KB)

The user, 26 Nov: **"deal with all the datasets the same way no matter what"** and **"all 10"**; 27 Nov: **"we can run whatever is required not an issue"** (the host card), and, on the original's weights for SPL1's cost reading, **"yes do it"**.

- **What changes.** Sections 5.3 and 5.4 left the SPLADE legs of MAG and Amazon "named and priced, not queued" because the host had no copy of the model and a CPU row would take days. Both reasons are gone: SPL1's A7 put the pinned model (revision 49cf4c7b0db5b870a401ddf5e2669993ef3699c7) on the host as `outputs/spl1/models/c0/model.safetensors`, and its gate G-C0 passed (the mean document non-zeros of the converted weights on all 609 MultiHop-RAG nodes are 213.2, the record's stored value). SPL1's rule is that if no smaller encoder passes, the record's leg stays; all three candidates are already WORSE on SQuAD (SPL1 A7), so the record's leg is the leg. The SPLADE leg is therefore run for **all three KBs the same way**, on the host card, order following size only: Prime (129,375 nodes, 2,801 test questions), Amazon (1,035,542 nodes, 1,642), MAG (1,872,968 nodes, 2,665).
- **New module, nothing else edited.** `outputs/mp_unified/zstk1g.py` (`selftest`, `splade --kb K [--device cuda|cpu] [--chunk 16] [--smoke N]`). It imports the merge, the sparse-row conversion and the constants of `zstk1s.py`, which is not edited (the laptop's Prime leg is running it). Same contract as 5.3: 256 tokens, batches of 64 in file order padded to the longest in the batch, `log1p(relu(logits))` max-pooled over every position, the running top 2,500 per question (ties to the lower id, positive scores only), a checkpoint after every block of 2,048, the same `splade_top.npz` for `zstk1.py first --splade-top`. Different: the weights come from the safetensors file above, the forward pass runs on the card in chunks of 16 rows of the same padded batch (the vectors are the unsplit batch's), the module does not hide the card (zstk1.py does at import), a checkpoint replace retries on a Windows lock (SPL1 A5), and `splade_blocks.jsonl` records each block's encode seconds so the rate is the median of the blocks (the guard suspends the card's rows while the other account computes: SPL1 A4). Selftest passes.
- **Gate G-D (device).** The first 512 nodes of Prime, once on the card and once on 4 CPU threads (smoke rows): the mean non-zeros per node differ by at most 1 percent. If not, the card rows are cancelled and the leg stays on the CPU.
- **Rows.** `stk1p-spc` (CPU smoke) and `stk1p-spg` (card smoke), then `stk1p-spl` (Prime), `stk1p-first2` (`zstk1.py first --kb prime --splade-top`), `stk1a-spl`, `stk1a-first2`, `stk1m-spl`, `stk1m-first2`, the card rows chained one after the other and after SPL1's last C0 row so no row of ours shares the card with that timing. `first2` runs `first` again with the SPLADE leg; the earlier dense-only `first.json` of a KB (Prime's was read at 02:47) is superseded and only the later file is cited; both are kept in the logs.
- **Cost, read from the first row.** Hours = nodes divided by the median block rate of Prime's row, on the guard's terms (a suspended row keeps its checkpoint). X6's planning ceilings stand (MAG within 3 days, Amazon within 4 days); if the rate prices a KB above its ceiling, this page gets an amendment before that row runs. The laptop's Prime leg (zstk1s.py, now at 22,528 nodes) keeps running as an independent cross-check, read as **G-T** when it ends: the share of Prime's test questions whose top five differ between the two files is reported, and any difference on the lists of record is explained or the card file is not used.
- **What does not change.** Section 7's legs, metrics and calls, the declared comparisons, the expectations, the BM25 removal. A SPLADE number may be read only from the files these rows write.

### 5.7 Amendment A7 (27 Nov, 12:29 IST, after the host card was found free and idle, before any dense vector of a STaRK KB was made on it)

The user, 27 Nov: **"so finish up the dense encodes of all stark datasets"**, with the host card reported free (0 percent, no other user since the host restart) and "we can run whatever is required not an issue" (the card). Section 6 chose the host CPU because the card was the other user's; that reason is gone.

- **What is left, at the CPU's rate.** Prime's two parts are assembled (`stk1p-vec` and `stk1p-first` read them). MAG's gte-modernbert documents have 31 of 47 shards done, granite-small's none; Amazon's gte has 2 of 26, granite-small's none. Priced from section 6's block seconds (0.82 s per mean text character, per block of 4,096 rows, at 8 threads), MAG needs about 26 more hours (about 14 of gte, about 12 of granite-small) and Amazon about 3 to 4 days, against section 8's ceilings (3 and 4 days) and with the CPU shared with the GP rows.
- **What changes.** The remaining document blocks of both KBs are made on the card. The device line of section 6 is amended for the blocks only; every other rule of section 6 stands: the node text (T1), the parts and pins, the fusion weight, the blocks of 4,096 rows and their file names, the shards of 40,000, the manifests and `zenc3v.cmd_vectors`.
- **New module, nothing else edited.** `outputs/mp_unified/zstk1d.py` (`selftest`, `gate`, `fill --kb mag|amazon`). `fill` takes the shards last first and, within a shard, the blocks last first, and makes each block that does not exist: the same texts (read from `nodes.jsonl` by byte offset), zenc's `encode_texts` (longest first, batches under the part's own token budget as `zenc3.NEW3` sets it, 65,536 for gte-modernbert and 131,072 for granite-small, an out-of-memory halves it), float16 weights as E0N's encodes of the seven datasets, zenc's damaged-row rule, a stop on any text cut at the window. It writes `<block>.gtmp.npy` and moves it into place with `os.replace` (the CPU row's scratch name is `.tmp.npy`, so the two never share a file). **It never assembles a shard, writes a manifest or removes a file.** The two CPU rows (`stk1m-enc`, `stk1a-enc`, zstk1.py, not edited) stay the assemblers: they walk the shards forward, skip every block that exists, make any block still missing on the CPU, assemble, checksum and write zenc's manifests. The card works from the end and the CPU from the front, so the two meet in the middle and the CPU rows then only assemble. A block the card writes into a shard the CPU has already assembled is an orphan scratch file; it is listed in the fill's record and left until the user's OK to remove it.
- **Gate G-V (the vectors).** The first 512 rows of Prime's documents in shard 0 and in shard 3, encoded on the card by each part and compared with the stored CPU rows of the same positions (`outputs/enc/<part>/stark_prime/docs`): the minimum cosine at least 0.99 (zenc's floor, `G0_MIN_COS`), no damaged row, no text cut. The 99th percentile, the median and the card's nodes per second and tokens per second are written beside it to `outputs/stark/dense_gpu/gate_v.json`. A fail stops the card rows (`fill` refuses to run without a passed gate) and the CPU rows go on alone.
- **Disclosed.** The shard json and manifest of an assembled shard say `device: cpu`, `weights: float32`, because they describe the assembling process; some blocks of that shard were made on the card in float16. The record of which blocks are the card's is `outputs/stark/dense_gpu/fill_<kb>.json` (part, shard, block, rows, seconds, tokens). Within a shard the two kinds of block are held to G-V's floor, not to bit equality.
- **Rows.** `stk1d-gate` (card, 2 CPUs, 8 GB), then `stk1d-m` (MAG) and `stk1d-a` (Amazon), each 2 CPUs, 8 GB and a share of the card, after the gate. `stk1m-enc` is running and `stk1a-enc` is queued, both unchanged; `stk1m-vec`, `stk1m-first` and the Amazon rows after them wait on those two as before. No delete and no overwrite of a finished shard.
- **Expectations, written before any number.** **X14:** G-V holds (the minimum cosine at least 0.99; the median at least 0.999, since float16 weights against float32 differ in the third decimal). **X15:** the card's rate on the gate's rows is at least 5 times the CPU smoke's (7.5 nodes/s for gte-modernbert and 24.7 for granite-small on Prime's first 2,048 nodes at 8 threads with other rows running, section 6), that is at least 37 and at least 123 nodes/s; the smoke's nodes are the longest, the gate's are the first 512 rows of two shards. **X16:** both KBs' remaining blocks are made within 12 hours of card time together. A read of any of these is a reading of the gate and fill records only; no retrieval number comes from this amendment.

### 5.8 Amendment A8 (27 Nov, 12:38 IST, after SPL1's speed-lever screen picked l5 (D2 ADOPT_LEVER(l5), docs/SPL1_SPARSE_LEG_SCREEN.md section 10.7), before any SPLADE number of Amazon or MAG)

The user, 27 Nov: **"check and compare properly and if any other model also we should compare then add those and finally decide"** (the record's model with speed levers against other models), and the standing **"deal with all the datasets the same way no matter what"**.

- **What the screen found.** The record's model (the original, C0) with half-precision weights, forward passes of 32 rows and the next batch tokenized in a second thread (SPL1's arm l5) encodes 4.31 times faster than the record's contract on the card (941.6 against 218.1 nodes/s, SQuAD's first 4,096 nodes), and its first-stage reads against the record's lists move by 0.033 point at most (18 reads, all LEVEL). D2 adopts it for the STaRK legs that have not run. The whole-run gain is smaller than the block gain (SQuAD 2.31 times, MultiHop-RAG 1.84 times by wall clock).
- **What changes (code only).** `zstk1g.py` gets `--lever none|l5` (default `none`, A6's path unchanged, byte for byte) and `--gate N`. With `--lever l5` the weights are half precision, a batch is still padded to the longest in the batch, the forward pass runs in chunks of 32 rows, the pooling is `log1p(relu(max over positions of the raw logits))` (the same numbers as A6's pooling in real arithmetic, because the function is non-decreasing; padding positions stay in the maximum, as the record has it), and the next batch of 64 is tokenized in a second thread while the card works on this one. It uses `zspl1l.py`'s `tokenize`, `pool_batch` and arm table, not edited. The question vectors are made the same way as the node vectors. The 256-token cut, the batches of 64 in file order, the top 2,500 per question, the blocks of 2,048 and the checkpoints are A6's. The checkpoint records the lever and a resume under another lever stops; `splade_top.json` records the lever, the dtype, the chunk and the sha256 of `zspl1l.py`.
- **Gate G-L.** `--gate N` encodes the first N nodes and the first N test questions of the KB twice, with A6's path on float32 weights and with the lever on a half-precision copy, and applies A11's E2 to each set: the first percentile of the per-vector cosine at least 0.999 and the number of non-zeros within 1.0 percent. It writes `outputs/stark/<kb>/gate_lever_l5.json` and exits 1 on a fail. The leg's row depends on the gate's row, so a fail drops the leg's row; the leg is then re-queued under a new row name on A6's path (`--lever none`), and this page is told.
- **Which KBs.** Amazon and MAG run at l5. Prime's leg was made before A11 (A6's path, float32) and stays as it is; it is not re-run, so the three KBs do not share one lever, and the difference is bounded by A11's reads on three datasets (at most 0.033 point on any read), not by a read on Prime. The seven datasets keep the record's lists (A13 reads the pool cells with l5's lists in place of them, SPL1 section 8.14).
- **Rows.** `stk1a-gl` and `stk1m-gl` (the gate, 512 nodes and 512 questions, a share of the card), then `stk1a-spl2` and `stk1m-spl2` with `--lever l5`, each after its gate and after SPL1's last candidate row as before.
- **Expectations, written before any number.** **X17:** G-L passes on both KBs (the first percentile at least 0.999, non-zeros within 1.0 percent); the texts of Amazon's products are longer than SQuAD's passages, so more of them reach the 256-token cut and the lever's gain per node is smaller than SQuAD's block gain (closer to MultiHop-RAG's 2.03 times). **X18:** the median block rate of Amazon's leg is at least 2 times the A6 path's median block rate on Prime (read from `splade_blocks.jsonl`); a miss is reported, and changes nothing. A read of either is a reading of the gate and block records only.

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
