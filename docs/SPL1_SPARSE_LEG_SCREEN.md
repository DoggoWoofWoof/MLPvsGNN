# SPL1: which sparse encoder carries the first stage's second leg (smaller, and not worse)

Declared 27 Nov 2026, 00:01 IST, before any SPL1 number exists. Follows `docs/ING1_INGESTION_AND_STAGE_REGISTRY.md` (registry rule set RS1, stage `first`, status FINAL: "e3v dense and SPLADE fused by reciprocal rank (C=60), depth 2,500"), `docs/ENC_ENCODER_COMPARISON.md` (the dense side, D101) and `docs/STK1_STARK_PACKAGE.md` section 5.4 (A4, BM25 removed). Code: `outputs/mp_unified/zspl1.py` (a new module; written and self-tested after this page is committed and committed again before its first row). Nothing is downloaded, encoded or read for this page.

## 1. Why this page exists

The user's words (27 Nov): **"why do we need bm25 etc our best are splade and dense do not do any useless work"**, then **"is the host gpu free, we can run whatever is required not an issue, are there smaller splade models etc can we make things faster and better on that front as well?"**, then **"first find the best models for our job whichever splade works best and is smaller and has even better recall we will use that and rebuild all the datasets, then our all 10 datasets will have the smallest possible encoders for both dense and splade and univerasal processing links etc"**.

The sparse leg of the first stage is `naver/splade-cocondenser-ensembledistil` (109M, revision 49cf4c7b, a non-commercial licence), the record's contract on the seven. It is the slow part of a rebuild: on the laptop CPU it ran at 1.4 nodes per second, so Prime alone is a day, MAG weeks. This page chooses, before any number, how a smaller sparse encoder is allowed to replace it.

## 2. What is chosen, and what is not

- **Only the sparse leg.** The dense side is on record: E3 screened smaller dense arms and the user took `e3v` (gte-modernbert-base + granite-embedding-small-english-r2, fused at weight 0.5, 197M, 1,152 wide) as D101; the small arm alone (`granitesm`, 48M) came out SHORT of the bars (`docs/ENC_ENCODER_COMPARISON.md`, the E3 tables). No dense run is made here. If the user wants the dense side opened again, that is a separate page.
- **No BM25** (STK1 A4). The legs are dense and sparse.

## 3. The candidates

Read from the models' own cards and file listings on 27 Nov 2026 by a delegated reader (public pages; no file was downloaded). The BEIR column is what the sentence-transformers table prints (BEIR-13 average nDCG@10); the OpenSearch cards print their own 13-dataset average (0.528, 0.517, 0.497). The families were scored on slightly different subsets, so these columns say who is worth screening, not who wins on this repository's data. That is what section 5 is for.

| id | repo and revision | parameters | BEIR-13 printed | licence | weights to fetch | query side |
| --- | --- | --- | --- | --- | --- | --- |
| C0 (the record) | naver/splade-cocondenser-ensembledistil @ 49cf4c7b | 109M | 50.5 | cc-by-nc-sa-4.0 | not fetched: the record's own top lists are read | the model |
| C1 | ibm-granite/granite-embedding-30m-sparse @ ad82b1fd09541c998c8d45045d601c51fdb8a9b7 | 30M (6 layers, 384 wide, vocabulary 50,265) | 50.8 | apache-2.0 | `model.safetensors` 60.7 MB | the model |
| C2 | opensearch-project/opensearch-neural-sparse-encoding-v2-distill @ 269e6638b2c4f648996691f6d751495285d8f330 | 67M | 52.8 | apache-2.0 | `model.safetensors` 268 MB | the model |
| C3 | opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini @ 4af867a426867dfdd744097531046f4289a32fdd | 23M | 49.7 | apache-2.0 | `model.safetensors` 91 MB and `idf.json` | none: a weight lookup (inference-free) |

**Not screened, and why.** `naver/splade-v3` (109M, 51.7): the same size, gated, a non-commercial licence. `naver/splade-v3-distilbert` (67M, 50.0): below C2 on the printed number at the same size, same licence terms as C0. `opensearch ... doc-v3-distill` (67M, 51.7, inference-free): the same size as C2 and lower. `opensearch ... doc-v3-gte` (133M, 54.6): larger than C2 and its card runs custom code (`trust_remote_code`), which this repository does not run on files it fetched. `rasyosef/splade-mini` and `splade-tiny` (11M and 4M; 44.5 and 40.6 printed): 6 and 10 points under C0 on the printed scale, further than the pass line of section 5 can forgive. `Linkup-Platform/linkup-sparseup-embed-v1` (about 150M, 56.4): larger than C0, so not a smaller encoder; kept as the named upper reference if no smaller candidate passes, and its newer version is not read. SlideOn (174M), KURE-Sparse and MILCO (593M): larger.

The weights are the repositories' `model.safetensors` only. The `.bin` pickles are never fetched.

## 4. The contract of the new leg (the same for every candidate)

- **Model's own recipe, nothing tuned.** Activation and pooling as the repository's own pooling configuration says (read from the file after the fetch and written into the record), over the positions the attention mask keeps. The record's contract (C0) pools over every position, padding included, which makes a vector depend on the other rows of its batch; that is not copied, and it is said here so the comparison is read for what it is: C0 is the leg as it shipped, the candidates are the leg as the card writes it.
- **The same text and the same cut.** The node text of the package, truncated at 256 tokens (C0's cut), batches of 64 in file order; fp32; score = sparse dot product; top 1,000 per question, ties to the lower node id. C3's question side is its idf weight lookup over the question's tokens.
- **Where the list is cut.** The question set of each dataset is the one the dense read used (section 5), so a list of the candidate and C0's list are over the same questions and the same nodes.

## 5. The screen

**Datasets (the user's earlier ruling: "check only the small and hard datasets"):** SQuAD (20,233 nodes; the 11,873 questions E0 read), MuSiQue (117,534 nodes; the 2,417 questions E0 read), MultiHop-RAG (609 articles; its 2,255 questions, where SPLADE reads only an article's lead). The other seven datasets are not run in the screen: they are rebuilt only if a candidate is picked (section 6).

**Lists compared.** C0's list is the record's own `splade_top1000` for the same questions (not re-run). The dense list is `e3v`'s top 1,000 as E0 filed it (`outputs/enc/e0/e3v/<dataset>.npz` for SQuAD and MuSiQue; MultiHop-RAG's from its package). Fusion is `zenc.rrf_top` (constant 60, float32, ties to the lower id).

**Reads.** Per dataset and candidate, set recall of the golds (golds found over golds, averaged over questions, as E0):
- **H1: RRF R@5** of e3v and the leg (the first stage as it stands).
- **H2: seed recall**: the golds inside e3v's top 5 together with the leg's top 5 (the seeds the walk starts from, `m3b_pools.seeds_of`).
- Guards, read and not called: the leg alone R@20 and R@1,000; RRF R@20.
Paired bootstrap over questions, 1,000 resamples, seed 0, on each difference (candidate minus C0).

**The call, per candidate and dataset, on each of H1 and H2** (the interval of the difference, in points):
- **BETTER**: the interval's low end is above 0.
- **LEVEL**: the low end is at least -1.0 (the user's "smallest encoder which doesn't lose much", used as 1 point in every encoder screen so far).
- **WORSE**: the low end is below -1.0.

**A candidate PASSES** if it is BETTER or LEVEL on both reads on all three datasets. **The pick is the passing candidate with the fewest parameters** (C3 23M, then C1 30M, then C2 67M). If a passing candidate is larger but BETTER on all six cells where the pick is only LEVEL, it is named beside the pick with its extra parameters, and the user chooses; nothing overrides the smaller one automatically. **If none passes**, the screen says which cell failed for each candidate and by how much, C0 stays, and the user is told the price of staying (section 7).

**Cost, read beside the call:** encode throughput of each candidate on the host GPU (nodes per second at 256 tokens, measured on SQuAD), and from it the hours of each of the ten datasets' nodes. No cost number enters the call.

## 6. What a pick triggers

Nothing automatically. A pick is followed by **SPL2**, declared before it runs: the leg's lists on all ten datasets' questions (the seven's test carves and the three STaRK KBs), one rule, each dataset's nodes priced from the token counts and the measured throughput; the registry's rule set RS1 gets an amendment on its `first` rule (the sparse leg's model) and the seeds, the UL2H seed readings and the first-stage reads of STK1 are re-read beside the filed ones; no filed number is edited. The laptop's Prime SPLADE run (C0, one 2,048-node block per checkpoint) is stopped when a pick is declared, since the host GPU would replace it, and not before.

## 7. Compute, downloads and what needs the user's word

- **GPU.** The user (27 Nov): "we can run whatever is required not an issue". The host's card is shared with another account's virtual machine (about 17 GB of its 24.6 GB held; about 140 W drawn with nothing of this repository on it). The screen's rows are short (minutes; 4 candidates by 3 datasets) and are sized to the card's free memory. They are not started if the card's free memory is below the row's ask. This does not extend to long rows: the SPL2 rebuild is priced and declared on its own.
- **Downloads.** Each is a public file at a pinned revision, fetched on the host with `huggingface_hub` without a token into its own new empty directory, `allow_patterns` limited to the model's `model.safetensors`, `config.json`, tokenizer files, the pooling configuration (and `idf.json` for C3), `local_files_only` afterwards, and read by Python run with `-I`. Size: C1 about 62 MB, C2 about 270 MB, C3 about 92 MB, small files included: **about 425 MB, three repositories, no token, no custom code**. Each fetch waits for the user's explicit yes.
- **The price of staying with C0 is read, not assumed.** The record's leg costs about 1.4 nodes per second on the laptop and has not been timed on the host's GPU; SPL1's cost line gives its comparison.

## 8. Code, rows and outputs

`outputs/mp_unified/zspl1.py` (new): `fetch --repo ID --rev SHA` (allow-listed files, own directory, records every file's sha256), `encode --model C --dataset D` (nodes in blocks, sparse matrix to `outputs/spl1/<model>/<dataset>/`, checkpoint after every block, a restart resumes), `read --dataset D` (E0's measures and the paired bootstrap against C0), `selftest` (a toy matrix and a toy ranking by hand; the pooling check against the repository's own pooling file). Rows go through the feeder in the classes `spl1c`, `spl1r`. Every row is a short GPU row (0.29 of the card, one at a time) with a checkpoint; a loss is one block. Outputs under `outputs/spl1/`, records `*.json`, a page section 10 when the first read lands.

### 8.1 Amendment A1 (27 Nov, 00:10 IST, after `zspl1.py` was written and its selftest passed, before any fetch, check, encode or read)

The user approved the three downloads (27 Nov: "Yes, all three"). What the code settled, stated before a number exists:

- **The recipe is the library's own, not a copy of it.** Each model is loaded as `sentence_transformers.SparseEncoder` from its fetched folder (`trust_remote_code=False`, `local_files_only`), nodes go through `encode_document`, questions through `encode_query` (C3's inference-free route), and every module that cuts has its `max_seq_length` set to 256. This replaces section 8's "pooling check against the repository's own pooling file": the pooling is the file's, run by the library. Micro-batches of 32 (the logits of a 50,000-word vocabulary at 64 would not fit beside the other account's job), blocks of 2,048 nodes.
- **`check` (before any encode).** On 16 real SQuAD node texts: a row encoded alone and inside its batch differ by at most 1e-3 and share their non-zero support on at least 90 percent of rows; on up to 4 texts longer than 256 tokens, appending words past the cut changes nothing (at most 1e-4); every value is non-negative; a question gives a non-empty vector. A model that fails is not encoded.
- **Lists hold positive scores only.** A question's list can be shorter than 1,000 (empty slots are -1 and RRF skips them). C0's record list for MultiHop-RAG holds all 609 articles, the candidate's those with a positive score, so **R@1000 is not read on MultiHop-RAG** (609 nodes: it would be 100 percent for C0 by construction). The leg's R@20 and the fused reads are read there.
- **Gate G-S1 (before any candidate is read).** C0's own reads must reproduce what is filed: on SQuAD and MuSiQue the question rows and gold counts equal E0's and C0's RRF R@5 with e3v equals E0's filed value to 5e-6 (0.882507 and 0.477658); on MultiHop-RAG the top 5 of the fused list equal MHR1's filed `e3v_splade` list on at least 99.9 percent of questions and the gold sets are equal. A failure stops the read and nothing is called. MultiHop-RAG's e3v list is recomputed exact in float32 over all 609 articles (the filed lists keep 20 columns).
- **What is bootstrapped.** The calls use RRF R@5 (H1) and seed recall (H2). RRF R@20, the leg's R@20 and (where n >= 1,000) R@1,000 are given with their intervals and are not called.
- **The cost line** is nodes per second from each candidate's SQuAD encode record and nodes / rate for the nine datasets that have a package: an estimate at SQuAD's passage length (KB entity names are shorter, so for them it is high), not a measurement of those datasets.

### 8.2 Amendment A2 (27 Nov, 00:17 IST, after the user's second message, before any candidate encode or read)

The user, 27 Nov: **"by smaller i also mean smaller vectors and faster encoding times"**.

- **"Smaller" has three measured parts.** (1) Parameters. (2) **Vector size**: mean non-zeros per node, taken as the mean of the three screen datasets' means from each candidate's own encode record; bytes at 8 per non-zero (an int32 id and a float32 value) and the question side's non-zeros are given beside. (3) **Encode time**: nodes per second on the host GPU over SQuAD's 20,233 nodes, the three candidates run one after the other in one chain with the card's free memory written down at each start; a CPU reading (host CPU, the first 512 SQuAD nodes, `encode --smoke 512 --device cpu`) is given beside and is not ranked.
- **The pick (replaces section 5's "fewest parameters").** The pass line is unchanged: BETTER or LEVEL on both reads on all three datasets, and no cost number enters it. Among the passing candidates each cost is ranked (1 = best: fewer parameters, fewer non-zeros, more nodes per second; equal values share the lower rank). **The pick is the passing candidate with the lowest sum of the three ranks; a tie goes to the fewer parameters.** A passing candidate that is BETTER on all six cells where the pick is not is named beside with its three costs, and the user chooses.
- **C0 is not ranked; what is known of its costs.** C0's vectors are kept only for MultiHop-RAG (609 articles, all cut at 256 tokens): 213.2 non-zeros per article, read from its stored shard, which is the figure the candidates' MultiHop-RAG means are compared with. C0's weights are not on the host (the repository's only weights file is a 438 MB pickle, `pytorch_model.bin`; there is no safetensors file), so C0 has no encode time on the host card. Its only figure is the laptop's 1.4 nodes per second at 6 threads on Prime's longest nodes, which is not comparable and ranks nothing. Putting C0 on the host card for a like-for-like time needs that file on the host and the user's word; it is not requested here.
- **Disclosed.** `check` (plumbing, 16 SQuAD texts) printed mean non-zeros of about 262 (C1), 339 (C3) and 252 (C2) before this rule was written. They are not the screen's reads and are not used.
- **The dense side (on record, not run here).** e3v is 1,152 wide. The storage stage (`docs/ENC_ENCODER_COMPARISON.md` section 21) found int8 document vectors free and 768-wide PCA with int8 near free. The small dense arm alone (`granitesm`, 48M, 384 wide; its four sets encoded in 67 s against gtemodern's 319 s on the host GPU) reads 76.12 and 43.89 R@5 on SQuAD and MuSiQue against the fused e3v's 83.68 and 48.15, which is why it was SHORT. A dense encoder or vector size beyond that is a separate page.

### 8.3 Amendment A3 (27 Nov, 00:21 IST, after the nine GPU rows were queued, before any of them started)

- **Micro-batch 16, not 32.** The card has about 6 GB free beside the other account's job, and the sparse head's logits are batch x 256 x vocabulary floats (50,265 for C1, 30,522 for C2 and C3). Sixteen texts per forward pass keeps the head's logits to about 0.8 GB for C1, so a pass does not fail when the other account's memory grows. The recipe's other parts are unchanged (fp32, 256-token cut, positive scores only, blocks of 2,048). `check` already showed that a text alone and in a batch differ by at most 1.4e-5, so the batch size moves no list; the value is written in each encode record (`batch`).
- **Options only.** `encode` gains `--batch` (default 16) and `--threads` (CPU rows only: torch threads, recorded). Nothing else in the module changes. The CPU readings of section 8.2 are queued as three chained rows (`--smoke 512 --device cpu --threads 4`, the first 512 SQuAD nodes, one candidate after the other) and are read beside the GPU readings, not ranked.
- **Order.** The GPU rows run MultiHop-RAG first (609 nodes, a minute each), then SQuAD, then MuSiQue, one row after the other; a row that fails is re-sent once by the feeder.

### 8.4 Amendment A4 (27 Nov, 00:39 IST, after the first GPU row was suspended, before any candidate encode finished or was read)

- **What happened.** The first GPU row (C1 on MultiHop-RAG) started at 00:24:57 and the standing guard (`outputs/host_ops/gpu_guard.py`, the user's 17 Nov rule "if someone else has use for gpu let them finish") suspended it at 00:25:53 for another account's virtual-machine worker (18.0 GB, 63% of the engines). A suspended process keeps its wall clock running, so a run's wall-clock rate (section 8.2's nodes per second) can be a fraction of what the models take, and it differs from candidate to candidate by when the other account's work comes and goes. The guard is not touched and the row is not resumed by hand; it takes the card back when the other work has been absent for ten polls.
- **The speed reading changes.** Each encode row now writes the seconds of every block's encode call (the models' forward passes for 2,048 nodes; the question encode, the scoring against the questions and the checkpoint are outside it) to `blocks.jsonl`, and the record carries the per-block rates and their **median**. **The speed of a candidate is the median block rate on SQuAD (10 blocks)**, replacing the whole-run wall rate; the wall rate stays in the record and is printed beside. The three candidates see the same blocks in the same order, so the blocks are comparable one by one; the median ignores the few blocks a suspension or the other account's work stretches. The CPU readings are read the same way and stay unranked.
- **What is not changed.** The pass line, the rank-sum pick, the parameter and non-zero costs, the batch size and every list. No candidate number existed when this was written: the first row has not finished.
- **Disclosed.** A median over ten blocks is still moved by contention that lasts most of a run; the block rates are all printed, and a candidate whose blocks sit visibly below the others' in time order is named in the result.

### 8.5 Amendment A5 (27 Nov, 01:25 IST, after the first four encode rows and one failed row; the only candidate numbers read so far are MultiHop-RAG's)

- **What happened.** C2 on SQuAD failed twice (01:00 and 01:22) on the checkpoint step: `os.replace` of the state file raised a Windows PermissionError (access denied), a short lock by another process on the target. It failed after block 6 and again at the first block of the resume; the card was suspended and resumed by the guard around the first failure. Nothing in the candidate's lists was affected: the checkpoint at 10,240 nodes is the last one written.
- **Code only, no rule.** Every file replace in `zspl1.py` now retries on a PermissionError (40 tries, 3 s apart) before it raises, and the block records are de-duplicated by block end with the last record kept, so a block redone after a resume counts once in the median. Selftest 14 checks pass. The failed row is re-sent by the feeder; the other rows are unchanged.
- **Disclosed.** The first read of the MultiHop-RAG cells (three candidates, all calls) was made before this amendment, on rows that had finished; no later row changes them. The reader's rule, the pass line and the pick are not touched.

### 8.6 Amendment A6 (27 Nov, 02:28 IST, a correction to A5, before any SQuAD or MuSiQue candidate number was read)

- **A5's reading was wrong for two of the three failures.** The re-sent C2 SQuAD row failed at 02:2x exactly as the first retry had (01:22): on `os.replace` of the checkpoint, after "resume at 10,240", with the 40 retries of A5 spent. The cause is in `zspl1.py`: the resume path opened the checkpoint with `np.load` and kept the file object open for the whole run, and Windows refuses to replace an open file (reproduced in a scratch folder: PermissionError 5 with the file open, success after `close()`). So the resumed runs could never save a checkpoint; the first failure (01:00, a fresh run, after block 6) is not explained by this and is still taken as a short lock by another process.
- **Code only.** The resume path reads the checkpoint inside a `with` block and copies the arrays; the selftest gains a check that a checkpoint read for a resume does not block its own replacement (15 checks pass). Rule, pass line, pick, batch size, A4's median speed and every list are as before.
- **The chain.** The feeder dropped the dependent rows after the second failure (C3 on SQuAD and the three MuSiQue rows were never run). They are re-queued under new names (`spl1h-*`), the C2 SQuAD row resuming from its checkpoint at 10,240 nodes (its blocks are written once per block end, A5). No SQuAD or MuSiQue candidate number has been read: C1's SQuAD row finished but was not read.

### 8.7 Amendment A7 (27 Nov, 02:58 IST, after the SQuAD cells of all three candidates were read and before any number of C0 was measured on the host)

The user, 27 Nov, on the first report: **"and C0? the original?"**, then **"speed i meant"** and **"yes do it"** (the download of the original's 438 MB weights file onto the host).

- **What is read so far, and what it means for the call.** On SQuAD all three candidates are WORSE than C0 on H1 (RRF R@5: C3 -1.04 [-1.39, -0.72], C2 -2.04 [-2.39, -1.71], C1 -3.13 [-3.55, -2.73]) and on H2 (C3 -1.65, C2 -2.38, C1 -3.61, every interval below zero). The pass line of section 5 needs BETTER or LEVEL on both reads on all three datasets, so no candidate can pass whatever MuSiQue shows. The MuSiQue rows still run (about 20 minutes of card time in all) so that every declared cell is filled, and the page's section 10 says that the verdict rests on SQuAD. The calls are not changed.
- **Why C0 is timed.** Section 5 says that if none passes C0 stays and the user is told the price of staying, and the first report could not give C0's side of the cost table: C0 had never been timed on the host card and its weights are only a pickle that was not on the host. Without it, the speed and vector-size comparison reads one side only.
- **The download (the user's OK of 27 Nov).** `naver/splade-cocondenser-ensembledistil` at revision 49cf4c7b0db5b870a401ddf5e2669993ef3699c7, from the public hub with no token, the files `config.json`, `vocab.txt`, `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` and `pytorch_model.bin` (the only weights file; a pickle, about 438 MB; licence cc-by-nc-sa-4.0), into the new empty folder `outputs/spl1/models/c0` on the host. The fetch stops over 460 MB for the pickle or 480 MB in all. Nothing is downloaded to the laptop, whose copy stays read through `local_files_only`.
- **How the pickle is handled.** It is read once, on the host, with `torch.load(..., weights_only=True)` (the restricted reader: tensors and plain containers only); the module stops unless the result is a plain dict of tensors, writes it as `model.safetensors` beside it, and the timing rows load the model from that file alone (`use_safetensors=True`, `local_files_only`, no remote code). The module runs with `python -I`.
- **The recipe is the record's** (`zmhr.py`, as `zstk1s.py` uses it): 256 tokens, batches of 64 in file order padded to the longest in the batch, `log1p(relu(logits))` max-pooled over every position, padding included. The forward pass of a batch is split in chunks of 16 rows of the same padded tensor, as A3's micro-batch does for the candidates, so the vectors are the unsplit batch's (selftest: 3 checks pass) and the card's peak matches the candidates'. Vectors are not stored; the non-zeros are counted.
- **New module and rows.** `outputs/mp_unified/zspl1c.py` (`selftest`, `fetch`, `convert`, `encode --ds D [--smoke N --device cpu --threads T]`, one more declared module; `zspl1.py`, which rows are running, is not touched). Rows: `spl1c-fetch`, `spl1c-conv` (CPU), `spl1c-cpusq` (the first 512 SQuAD nodes on 4 CPU threads, the candidates' timing row), then the three GPU rows `spl1c-gsq`, `spl1c-gmh`, `spl1c-gmq` (SQuAD, MultiHop-RAG, MuSiQue), chained after `spl1h-c3mq` so that no row of this screen shares the card with C0's timing.
- **Reads.** Per dataset: nodes per second as the median of the blocks' rates (A4) with the wall rate beside, document and question non-zeros (and bytes per node at 8 per non-zero), the CPU rate. **Gate G-C0:** the mean document non-zeros of C0 on MultiHop-RAG is within 1.0 of the record's stored vectors (213.2); a model loaded wrongly or a recipe that differs would be far off. If it fails, C0's speed is reported as not comparable and is used nowhere.
- **What does not change.** The pass line, the calls, the pick rule (C0 stays unranked: it is the reference, not a candidate; its measured values are printed beside the candidates' and are the price of staying), the lists, the batch size, A4's median rate. Nothing in section 9 is edited.
- **Disclosed.** The speeds are not taken under identical card conditions: the other account's work and the guard's suspensions differ from row to row (A4), which is why the median block rate leads and the wall rate stays beside. The CPU rows share the host's cores with rows of other sessions that happen to run. Before this amendment the MultiHop-RAG cells (before A5) and the SQuAD cells of C1, C2 and C3 were read.

### 8.8 Amendment A8 (27 Nov, 04:00 IST, after the user's next message and before any pool-coverage number of any leg was read)

The user, 27 Nov: **"its not about the size but more of the speed to recall loss atp, if there are any other alternatives then check else do pool cov for c0 vs c3 and whatever else is worth and compare and finalize autonomusly, once done start redoing the encodings for all 10, 6x faster with barely a 1 point recall loss seems worth it see if we can do better than this"**.

- **What changes in the question.** Section 5 asks whether a candidate keeps the first-stage reads (H1, H2) within one point; C3 does not on SQuAD (H1 -1.04 [-1.39, -0.72]), so by section 5 C0 stays. The user's measure is the price per unit of speed, and the walk that follows the first stage can repair part of what a weaker leg loses (the leg alone at rank 1,000 is level: C3 -0.56 on SQuAD). A8 adds the read that decides the pools: the all-gold coverage of the filed pool cells with C3's list in place of the record's. Section 5's calls are not edited; A8 adds a second rule (D1) that the decision rests on, and the page says so.
- **Alternatives (checked, nothing downloaded).** The search found C3's family (inference-free sparse retrievers: a query is a bag of tokens with learned weights, no encoder pass) and no published model that is faster than C3 at similar recall; the larger ones named in section 3 and the unscreened ones (the gated v3 models, the doc-v3 pair, custom-code models) are heavier than C3 or lose more recall than C2. No new download is proposed. One would need the user's explicit OK with file, source and size.
- **New module.** `outputs/mp_unified/zspl1p.py` (one more declared module; `selftest`, `pool --ds D`, `call`). It imports `zpl2g`, `zpl5g` and `zpl6g` and changes none of them; in its own process it replaces only the function that reads `retrieval_cache/splade_top1000.npz` by one that returns C3's lists for the same questions. The engine, the cells, the dense list (the record's frozen top 1,000, the same for both arms), the graph, the vectors and the carve are the filed ones.
- **Cells (read from the filed files at run time, their sha256 recorded).** Per dataset: PL5's `frontier_picks.all` at 500, 1,000, 1,500, 2,000 and 2,500 nodes, PL6's `R95_bounded` pick, and the old cell (`old` of PL5/PL6); duplicates removed. Eval carve (s1eval; MultiHop-RAG's 1,130 questions, SQuAD's 11,873, MuSiQue's 2,417). The picks were made on the dev carve with C0's list, so they favour C0; disclosed, and not corrected.
- **Arms.** C0's arm is the per-question arrays of the filed PL5 and PL6 eval confirmation grids (no re-run). C3's arm is this module's run on the same questions. **Gate G-P1:** through the module's replaced reader with C0's own list, the old cell's `present` and `size` arrays equal the filed ones on every question (this checks the row mapping and the replaced reader); on a miss nothing is read. **Padding:** C3's list ends in -1 where a node has score zero (MultiHop-RAG; SQuAD and MuSiQue lists are full); the missing ids are appended in ascending order. C0's own tail order cannot be reproduced; the effect is limited to ranks past the positive scores (about 550 of 609 on the first MultiHop-RAG question).
- **Reads.** Per dataset and cell: the all-gold fraction of both arms, the paired difference (C3 minus C0) with a bootstrap over questions (1,000 resamples, seed 0), both arms' mean nodes and their difference. **Per-cell call** as in section 5: BETTER (low end above 0), LEVEL (low end at least -1.0 point), WORSE; a cell where C3's mean nodes exceed C0's by more than 5 percent is called MORE_NODES whatever its coverage.
- **Decision rule D1 (declared here, before any number).** The two cells on each dataset that sit at the user's node bound are the deciding ones: the 2,500-node frontier cell and PL6's `R95_bounded`; the smaller cells and the old cell are read and reported, not called.
  - **ADOPT_C3** if all six deciding cells are BETTER or LEVEL and the card speed ratio (C3 over C0, median block rate on SQuAD, from `spl1h-c3sq` and `spl1c-gsq`) is at least 3.0. If C0's card row fails, the CPU ratio of the first 512 SQuAD nodes (C3 15.21, C0 2.36 nodes per second) stands in.
  - **ADOPT_C3_AT_PRICE** if at least one deciding cell is WORSE but every one of the six has a point difference of -1.5 points or better (the user's "barely a 1 point recall loss", read as a point estimate with half a point of room) and the speed ratio is at least 3.0. The price (each cell's point difference and interval) is printed beside the decision and goes in the report.
  - **C0_STAYS** otherwise. The next amendment (A9) then declares the speed levers of C0 itself (half precision, batches sorted by length, the pooling over real tokens only), each read for the recall it changes against the record's lists; it is not declared before it is needed.
  - Nothing is decided on two datasets: MuSiQue's C3 list must be in before D1 is applied.
- **Rows.** The pool reads run on the laptop for SQuAD and MultiHop-RAG now (their C3 lists are filed) and for MuSiQue when `spl1h-c3mq` lands (laptop, or the host if memory stops it). **Removed from the feed (reason: they price C0's weights on the card, about four days for Amazon and MAG, before D1 is applied):** `spl1c-gmq` (C0 on MuSiQue; its number decides nothing and costs about 40 minutes of card time that a rebuild needs) and the A6 chain `stk1p-spg`, `stk1p-spl`, `stk1p-first2`, `stk1a-spl`, `stk1a-first2`, `stk1m-spl`, `stk1m-first2`; they come back, with the leg D1 names, in SPL2. `spl1c-gsq` and `spl1c-gmh` stay. The laptop's Prime run (C0, zstk1s) is not stopped; if C3 is adopted it is redundant and the user is told.
- **Disclosed.** The SQuAD cells of the first-stage screen were read before A8; no pool number of C3 was read before it (C0's filed pool numbers of PL5 and PL6 were known and are the C0 arm). The dense list is the record's frozen one, not e3v's (the filed pool grids exist on it for the three datasets; E0's e3v lists cover the six eval carves only); the first-stage effect on e3v is the H1 and H2 of section 10.

### 8.9 Amendment A9 (27 Nov, 04:14 IST, after the MultiHop-RAG pool read (the first version of `zspl1p.py`, commit 73ce8355) and before any SQuAD or MuSiQue pool number)

- **The SQuAD read is split in parts.** One core runs the filed cells at about 0.9 seconds per question (300 questions in 260 seconds), which is about three hours for SQuAD's 11,873 eval questions. The questions are independent, so `zspl1p.py` gains `pool --ds D --part I --of K` (questions `[n*I/K, n*(I+1)/K)`, each part written to its own file with its own G-P1 on its questions) and `merge --ds D --of K` (the parts must tile the questions and every G-P1 pass; then the same reading as a whole run). SQuAD runs in four parts on the laptop's free cores; MuSiQue is sized when its list lands. No cell, call, bootstrap, rule or D1 changes; the result of a split run equals a whole run's by construction (nothing is shared between questions).
- **The first version patched the file reader; the module now sets `src.splade` of the loaded source** (a list array of the same shape and dtype). The effect is the same; the change is made so that the module patches nothing in the pinned modules. The MultiHop-RAG read taken with the first version is re-run through the new code and must give the same numbers, and both files are kept (the first: all cells `LEVEL` or better at +0.27 point, the old cell `WORSE` at -0.97 point; read before this amendment).
- **Disclosed.** The single-process SQuAD run started with the first version was 600 questions in when it was set aside; its numbers are not read.

### 8.10 Amendment A10 (27 Nov, 05:08 IST, after the four SQuAD parts were written and before the merge produced any number)

- **A path correction in `merge`.** The merge step looked for each part's gate file under the name of the part's arrays (`squad_c3_part0of4.json`); the gate files are written as `squad_part0of4.json`. The merge stopped with a file-not-found before reading any cell, so no SQuAD pool number was read. `cmd_merge` now reads the gate file of the same part under its real name. Nothing else changes: the cells, the paired bootstrap, the calls and D1 are those of section 8.8, and the four parts' own G-P1 results are read from the files they were written to.

### 8.11 Amendment A11 (27 Nov, 06:27 IST, after the user's reply to the report of section 10.6 and before any number of a speed lever or of a further candidate)

The user, 27 Nov, on the report that C0 stays and that its speed levers were not run: **"check and compare properly and if any other model also we should compare then add those and finally decide"**. A8 said that a C0_STAYS would be followed by an amendment declaring the speed levers of C0 itself; A9 and A10 were used for the split of the pool read and its merge fix, so this is A11. It has two parts: the levers of C0 (declared here in full, run now) and further candidates (named here with file, source and size, not downloaded; their screen is declared in A12 after the user's OK).

**Part 1: C0's speed levers.**

- **Why they can matter (reasoned, not measured).** C3 has about a fifth of C0's weights and is only 1.72 times faster on the card, against 6.44 times on the CPU: the card is not limited by the weight count. C0's forward pass ends in a vocabulary head (768 to 30,522) at every one of the 256 positions, in fp32, and the record's pooling then makes three element-wise passes (`relu`, `log1p`, `max`) over the `[rows, 256, 30,522]` logits. The tokenizer and the copy of the dense vectors to the CPU run between the card's calls.
- **Arms** (C0's weights, the record's 256 tokens, batches of 64 in file order padded to the longest in the batch, unless the row says otherwise). All arms that pool as the record does return the same vector in exact arithmetic; the first one tests the identity, the others change the arithmetic only.

| arm | what changes against the record's contract (L0 = `zspl1c.py`, fp32, chunks of 16) | vector |
|---|---|---|
| l0 | nothing (the reference, timed in the same process as the others) | the record's |
| l1 | the maximum over positions is taken on the raw logits and `log1p(relu(.))` is applied to the pooled row: `max_t f(x_t) = f(max_t x_t)` for the non-decreasing `f`, so no element-wise pass over the logits | equal in exact arithmetic |
| l2 | l1 with TF32 matrix products | rounding only |
| l3 | l1 with the model in fp16 (logits in fp16, the pooled row in fp32) | rounding only |
| l4 | l1 with the model in bf16 | rounding only |
| l5 | l3 with chunks of 32 and the next batch tokenized in a second thread while the card runs the current one | as l3 |
| l6 (information only) | l5 with the pooling over real tokens only and the nodes of a block sorted by length before batching (the vector then no longer depends on its batch; it is not the record's) | differs from the record's |

- **Bench (rule fixed here).** The first 4,096 SQuAD nodes in blocks of 512, and the first 256 eval questions, on the host card in one process. Per block every arm is run on the same texts, the order of the arms rotated from block to block, after one untimed warm-up block per arm; the card's peak memory per arm is read. The rate of an arm is the median over the 8 blocks of nodes per second (tokenizer and the copy to the CPU inside the time, as in A7), and **E1** is the paired ratio: the median over blocks of (arm's rate over l0's rate on the same block), so the other account's load, which changes from minute to minute, is common to both.
- **E2 (equivalence on the bench sample, nodes and questions each).** The first percentile of the per-vector cosine to l0's vector is at least 0.999 and the total non-zeros are within 1.0 percent of l0's. l1 is also reported as the share of bit-equal vectors; it is not required to be 1 (the identity is exact in real arithmetic; the `log1p` kernel is not guaranteed monotone to the last bit).
- **Pick.** The arm among l1 to l5 with the largest E1 ratio among those with E1 at least 1.5 and E2 passed. If none, **LEVERS_NOT_ADOPTED** and nothing below is run. l6 is never picked: it is read for what it would buy.
- **Full reads (only if a pick exists).** SQuAD (20,233 nodes; 11,873 eval questions), MultiHop-RAG (609; 2,255) and MuSiQue (117,534; 2,417) with the pick, and with l6; l0 as well on SQuAD and MultiHop-RAG (not MuSiQue: about ten minutes of card for a reference that the record's list already is). The rate is the median block rate of a full run (A4). Per dataset the lists are read as in section 5 against the record's list (H1: RRF R@5, H2: seeds, and the leg's R@20 and R@1000 where defined), paired bootstrap (1,000 resamples, seed 0), and the pick against l0 on SQuAD and MultiHop-RAG. **Calls for a lever**: BETTER (low end above 0), LEVEL (low end at least **-0.20** point, tighter than section 5's -1.0 because a numerical lever should move nothing), WORSE. The list agreement (the share of questions whose top 1,000 is the same set, and whose top 20 is the same sequence) is printed beside.
- **D2 (declared here, before any number).** **ADOPT_LEVER(pick)** if E1 and E2 hold and every read of the pick (four reads on each of the three datasets against the record's list, and the same four reads against l0 on SQuAD and MultiHop-RAG) is BETTER or LEVEL. Otherwise **LEVERS_NOT_ADOPTED**. If adopted: the three STaRK legs (`stk1p-spl`, `stk1a-spl`, `stk1m-spl`; Prime's runs at l0 in any case, since it started before this amendment) run with the pick through an STK1 amendment that adds a flag to `zstk1g.py` whose default is the record's path; the seven datasets keep the record's lists, which the pick reproduces to within E2 and the calls above. No pool-coverage read is made for an equivalent lever unless the list agreement is below 99 percent of the questions; the user is told in that case and the pool cells are read as in A8. **If l6 is BETTER than l0 by the interval on H1 on all three datasets, the user is told**; adopting it would change every list of the ten datasets and is a separate decision.
- **New module and rows.** `outputs/mp_unified/zspl1l.py` (one more declared module; `selftest`, `bench`, `call`, `encode --arm A --ds D`, `read`; it imports `zspl1`, `zspl1c` and `zworld` and edits none of them; the host runs it with `python -I -B`, the laptop with `-E -P -B`). Rows (host, GPU, the card to ourselves): `spl1l-bench`, `spl1l-call` (CPU), then the encodes `spl1l-*` named by arm and dataset; the reads run on the laptop. The Prime leg already running (`stk1p-spl`) is not touched; `stk1a-spl` and `stk1m-spl` wait for `spl1l-call` so that no lever is in doubt when they start.

**Part 2: further candidates (named, not downloaded).** Sizes from the hub's file listing (no file fetched):

| key | repo | weights file | listed size (all files) | licence |
|---|---|---|---|---|
| c4 | opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill | model.safetensors 268.0 MB | 271.6 MB | Apache-2.0 |
| c5 | naver/splade-v3-distilbert | pytorch_model.bin 268.0 MB (a pickle) | 268.9 MB | CC BY-NC-SA 4.0 |
| c6 | naver/splade-v3 | pytorch_model.bin 438.1 MB (a pickle) | 439.1 MB | CC BY-NC-SA 4.0 |
| c7 | opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte | model.safetensors 549.6 MB | 553.2 MB | Apache-2.0 |

The pickles would be read as C0's was (`torch.load(weights_only=True)`, written as safetensors, loaded from that alone). Each needs the user's OK with this file, source and size. Their screen (A12) is declared when the OK comes, before any candidate number.

**Expectations, written before any number (marked held or not in section 10).**

- **X8.** l1 passes E2 and is at most 1.3 times l0's rate: the matrix products, not the element-wise passes, dominate.
- **X9.** l3 and l5 pass E2 and reach at least 2.0 times l0's rate; l5 is the pick.
- **X10.** l6 is at least 1.3 times l5's rate on SQuAD, and its cosine to l0's vectors has a first percentile below 0.99 (the pad positions matter), without a loss of H1 beyond -1.0 point.

**Disclosed.** The user's message repeats the text of the previous report; it is read as "run the screen, compare, add models, decide". The levers are chosen from what the record's contract allows without changing C0's weights; further levers (a custom kernel, pruning the vocabulary head to the tokens that fire) are not declared.

### 8.12 Amendment A12 (27 Nov, 06:37 IST, after the user's OK of the four further models and before any number of any of them)

The user, 27 Nov, answering the question that listed the four models of A11 with file, source and size: all four chosen (**c4 opensearch doc-v3-distill, c5 naver splade-v3-distilbert, c7 opensearch doc-v3-gte, c6 naver splade-v3**). This is the OK that section 7 and the download rule need.

- **What is downloaded** (public hub, no token, to the host only, each model into its own new empty folder `outputs/spl1/models/cK`; nothing goes to the laptop). Revisions are pinned:

| key | repo @ revision | weights (listed) | files taken | stop if over |
|---|---|---|---|---|
| c4 | opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill @ babf71f3c48695e2e53a978208e8aba48335e3c0 | model.safetensors 268.0 MB | every json, txt, md and safetensors file of the listing (271.6 MB; no hidden file, no onnx or openvino folder) | 300 MB |
| c5 | naver/splade-v3-distilbert @ 2db06b86d65e316e2ca9907aa1aa8be6f8c4e739 | pytorch_model.bin 268.0 MB (a pickle) | the json, txt and md files and the pickle (268.9 MB) | 300 MB |
| c6 | naver/splade-v3 @ fdfeceb91d7b9de7985b38addd3ba9f53a59a355 | pytorch_model.bin 438.1 MB (a pickle) | the same kinds (439.1 MB) | 480 MB |
| c7 | opensearch-project/opensearch-neural-sparse-encoding-doc-v3-gte @ 1646fef40807937e8e130c66d327a26421c408d5 | model.safetensors 549.6 MB | every json, txt, md and safetensors file (553.2 MB) | 600 MB |

- **The pickles (c5, c6)** are read once, on the host, with `torch.load(weights_only=True)` and written as `model.safetensors` beside it (as A7 did for C0); the module stops unless the result is a plain dict of tensors; the pickle is then renamed `pytorch_model.bin.unused`, so the loader finds only the safetensors file. No remote code is used (`trust_remote_code=False`). Licences: c5 and c6 are CC BY-NC-SA 4.0 (non-commercial; the weights are used for research measurements and not redistributed); if either were adopted the user is told, since the licence would enter the pipeline.
- **The screen** is section 5's, unchanged: each candidate in its own native contract (the library's own pooling and query route, 256 tokens, fp32, micro-batch 16 as in A3), nothing stored but the top-1,000 lists, on SQuAD, MuSiQue and MultiHop-RAG; H1 (RRF R@5) and H2 (seeds) and the leg's R@20 and R@1,000 against the record's lists; paired bootstrap (1,000 resamples, seed 0); calls BETTER, LEVEL (low end at least -1.0 point), WORSE. **FIRST-STAGE-OK** is section 5's pass line: BETTER or LEVEL on H1 and H2 on all three datasets. Per candidate the card rate (median block rate on SQuAD, A4) and the non-zeros per node and question are read.
- **Decision rule D3 (declared here, before any number).** R is C0 at the contract D2 leaves in force: the lever's SQuAD median block rate if D2 adopts one, A7's 199.61 nodes per second if not. A candidate replaces C0 for all ten datasets only through a pool read, because a replacement means rebuilding every first stage; it gets the pool read (declared in A13, which also generalizes `zspl1p.py`, hard-wired to C3 today) if it is FIRST-STAGE-OK and one of these holds:
  - **(a)** its speed ratio to R is at least 3.0;
  - **(b)** it is BETTER on H1 and H2 on all three datasets, its H1 difference is at least +1.0 point on each, and its speed ratio is at least 1.0;
  - **(c)** it is not FIRST-STAGE-OK, but every H1 and H2 difference is -1.5 points or better and its speed ratio is at least 3.0 (the price rule of D1).
  If a candidate would meet (a) or (c) except for its ratio, it is re-timed at fp16 (the lever arm l3 on the same weights; the same systems setting that C0 may get) before the call, and the ratio of that re-time is the one used. If no candidate gets a pool read, **C0 stays**, with D2's lever if one was adopted, and the user is told each candidate's price. Nothing is decided before all three datasets of a candidate are read. A pool read of a candidate is adopted only if its six deciding cells (A8) are BETTER or LEVEL.
- **Module and rows.** `outputs/mp_unified/zspl1m.py` (one more declared module; `selftest`, `fetch`, `convert`, `check`, `encode`, `read`; it sets the four keys in `zspl1.py`'s tables in its own process, calls `zspl1.py`'s functions and edits nothing there; the host runs it with `python -I -B`, the laptop's `read` with `-E -P -B`). Rows: `spl1m-fc4` to `spl1m-fc7` (fetch, CPU and network, now), `spl1m-vc5` and `spl1m-vc6` (convert), `spl1m-kc4` to `spl1m-kc7` (the section-4 check on the CPU), then twelve GPU rows `spl1m-g*` (four models, three datasets) chained after the last row of A11's chain and before the gate that holds the STaRK legs, so that no candidate shares the card with another of our rows.
- **Expectations, written before any number.** **X11** c4 (the next generation of C3's family) is WORSE than C0 on SQuAD's H1 (queries without expansion) and LEVEL on MuSiQue and MultiHop-RAG. **X12** c5 is LEVEL on all six reads and at most 2.0 times R's rate (the vocabulary head is as large as C0's). **X13** c6 is LEVEL or BETTER on all six, with a ratio between 0.8 and 1.2, and does not meet (b) (a gain of 1.0 point on H1 on each dataset is more than the filed gaps between SPLADE recipes). **X14** c7 is BETTER on SQuAD's H1 and slower than R (ratio below 0.8). **X15** no candidate meets (a), (b) or (c): C0 stays.
- **Disclosed.** Candidates are timed natively (fp32, micro-batch 16) while C0's lever arms run at their own settings; the fp16 re-time above is the parity step. C3's pool read (section 10) is not repeated for any of these. The expectations X11 to X15 and A11's X8 to X10 are marked held or not in section 10.

## 9. Expectations, written before any number

- **X1.** C2 (the highest printed score) is BETTER or LEVEL on all six cells; it is the largest of the three, so it is the pick only if C1 and C3 do not pass.
- **X2.** C1 (30M, the printed score level with C0) is LEVEL on SQuAD and MuSiQue; the risk is MultiHop-RAG, where 256 tokens of a long article is all any of them reads.
- **X3.** C3 (inference-free: no question expansion) is WORSE on MuSiQue at H2, because its multi-hop questions are the ones that need expansion most.
- **X4.** No candidate is BETTER than C0 by more than 2 points on H1: the leg is one of two in a fusion that already holds e3v, so a better leg moves the fused read less than it moves the leg alone.

- **X5 (A8).** On SQuAD C3's deciding cells are LEVEL: the pools reach down to the leg's rank 1,000, where C3 is level (-0.56), so the walk repairs most of the first-stage loss.
- **X6 (A8).** MuSiQue is where C3 loses most (X3): at least one of its two deciding cells is WORSE, and the decision is ADOPT_C3_AT_PRICE or C0_STAYS on that dataset.
- **X7 (A8).** MultiHop-RAG's cells are LEVEL or BETTER: its pools hold a tenth of a 609-article corpus, and the reads there differ by a handful of questions.

Expectations are not calls; each is read against the table and marked held or not held in section 10.

## 10. Result

Written 27 Nov 2026, 06:05 IST, after the MuSiQue pool read. Section 5's rule and D1 of section 8.8 are applied as declared; the changes made after a first number was read are A9 (the parts and the `src.splade` change, after the MultiHop-RAG pool read) and A10 (a path correction, before any SQuAD pool number); both are in sections 8.9 and 8.10.

### 10.1 The first-stage screen (section 5): no candidate passes

All-gold and recall points against C0 with the paired 95% bootstrap interval (SQuAD 11,873 questions, MuSiQue 2,417, MultiHop-RAG 2,255). A cell is WORSE when the interval's low end is below -1.0 point.

**squad** (11,873 questions, 20,233 nodes). C0's own values: RRF R@5 88.25, seed recall 91.81, RRF R@20 95.28, leg R@20 94.50, leg R@1000 99.50.

| read | C3 opensearch doc-v2-mini 23M | C1 granite 30M | C2 opensearch v2-distill 67M |
|---|---|---|---|
| RRF R@5 (points vs C0) | -1.04 [-1.39, -0.72] WORSE | -3.13 [-3.55, -2.73] WORSE | -2.04 [-2.39, -1.71] WORSE |
| seed recall (points vs C0) | -1.65 [-1.98, -1.31] WORSE | -3.61 [-4.06, -3.15] WORSE | -2.38 [-2.69, -2.06] WORSE |
| RRF R@20 (points vs C0) | -0.84 [-1.08, -0.58] WORSE | -2.29 [-2.59, -2.00] WORSE | -1.43 [-1.68, -1.17] WORSE |
| leg R@20 (points vs C0) | -2.81 [-3.17, -2.47] WORSE | -3.04 [-3.45, -2.61] WORSE | -2.54 [-2.89, -2.19] WORSE |
| leg R@1000 (points vs C0) | -0.56 [-0.72, -0.41] LEVEL | -0.06 [-0.19, +0.08] LEVEL | -0.14 [-0.25, -0.01] LEVEL |

**musique** (2,417 questions, 117,534 nodes). C0's own values: RRF R@5 47.77, seed recall 53.00, RRF R@20 60.42, leg R@20 55.27, leg R@1000 80.88.

| read | C3 opensearch doc-v2-mini 23M | C1 granite 30M | C2 opensearch v2-distill 67M |
|---|---|---|---|
| RRF R@5 (points vs C0) | -0.03 [-0.66, +0.62] LEVEL | +0.04 [-0.59, +0.73] LEVEL | +0.12 [-0.45, +0.67] LEVEL |
| seed recall (points vs C0) | -0.48 [-0.95, +0.05] LEVEL | -0.77 [-1.30, -0.19] WORSE | -0.32 [-0.76, +0.12] LEVEL |
| RRF R@20 (points vs C0) | +0.35 [-0.19, +0.87] LEVEL | -0.16 [-0.74, +0.43] LEVEL | +0.69 [+0.19, +1.20] BETTER |
| leg R@20 (points vs C0) | -2.35 [-3.03, -1.61] WORSE | +1.41 [+0.66, +2.22] BETTER | +1.52 [+0.91, +2.14] BETTER |
| leg R@1000 (points vs C0) | -2.11 [-2.76, -1.47] WORSE | +3.26 [+2.56, +3.95] BETTER | +2.53 [+1.95, +3.09] BETTER |

**multihoprag** (2,255 questions, 609 nodes). C0's own values: RRF R@5 61.87, seed recall 73.74, RRF R@20 91.28, leg R@20 83.92.

| read | C3 opensearch doc-v2-mini 23M | C1 granite 30M | C2 opensearch v2-distill 67M |
|---|---|---|---|
| RRF R@5 (points vs C0) | -0.34 [-1.03, +0.32] WORSE | +0.26 [-0.55, +1.06] LEVEL | +0.27 [-0.35, +0.86] LEVEL |
| seed recall (points vs C0) | -0.47 [-0.93, +0.01] LEVEL | -1.80 [-2.40, -1.25] WORSE | -0.68 [-1.18, -0.18] WORSE |
| RRF R@20 (points vs C0) | -1.18 [-1.70, -0.72] WORSE | -1.74 [-2.31, -1.25] WORSE | +0.05 [-0.40, +0.49] LEVEL |
| leg R@20 (points vs C0) | -2.57 [-3.30, -1.91] WORSE | +0.65 [-0.16, +1.42] LEVEL | +1.48 [+0.89, +2.03] BETTER |
| leg R@1000 (points vs C0) | n/a | n/a | n/a |

Section 5's pass needs "not WORSE" on all six cells (RRF R@5 and seed recall, three datasets). C3 is WORSE on SQuAD (-1.04 and -1.65), and on MultiHop-RAG's RRF R@5 (-0.34, interval low end -1.03); C1 and C2 are WORSE on SQuAD and on a seed cell. Verdict of the screen: **none passes; no pick on the first-stage reads**. The reads were decided on the pools (A8), which is the next subsection.

### 10.2 The pool-coverage read of C0 against C3 (A8 to A10, rule D1)

The filed PL5/PL6 cells of the eval carves, C3's list in place of the record's SPLADE list, everything else as filed. Gate G-P1 (the replaced list reproduces the filed old-cell numbers exactly) passed on all questions of the three datasets (SQuAD 11,873 in four parts, MuSiQue 2,417 in four parts, MultiHop-RAG 1,130).

| dataset | cell | role | C0 all-gold | C3 all-gold | diff pt [95% interval] | call | nodes C0 -> C3 |
|---|---|---|---|---|---|---|---|
| squad | `rrf1000_b400+STRUCT:h1_c100~cs+top50` | frontier_500 | 0.9965 | 0.9960 | -0.05 [-0.13, +0.03] | LEVEL | 446.7 -> 447.4 |
| squad | `rrf1000_b100+FULL:h3_bf_v5000_w4096~cs-vr+top800` | frontier_1000 | 0.9979 | 0.9977 | -0.02 [-0.07, +0.03] | LEVEL | 900.0 -> 900.0 |
| squad | `rrf1000_b400+FULL:h2_c100_v5000~s20-cs-vr+top800` | frontier_1500 | 0.9982 | 0.9985 | +0.03 [-0.04, +0.09] | LEVEL | 1200.0 -> 1200.0 |
| squad | `rrf1000_b800+FULL:h3_c100_v5000~cs-rk+top800` | frontier_2000 | 0.9991 | 0.9987 | -0.03 [-0.08, +0.01] | LEVEL | 1599.9 -> 1599.8 |
| squad | `rrf1000_b1600+STRUCT:h2_c100_v5000~cs+top800` | frontier_2500 **(deciding)** | 0.9992 | 0.9992 | +0.00 [-0.03, +0.03] | LEVEL | 2200.0 -> 2254.1 |
| squad | `rrf1000_b15` | R95_bounded **(deciding)** | 0.9554 | 0.9463 | -0.91 [-1.16, -0.65] | WORSE | 15.0 -> 15.0 |
| squad | `equal_rrf_budget_50` | old | 0.9798 | 0.9768 | -0.29 [-0.45, -0.13] | LEVEL | 50.0 -> 50.0 |
| multihoprag | `rrf1000_b50+FULL:h2_c100_v5000+top50` | frontier_500 | 0.9912 | 0.9938 | +0.27 [+0.00, +0.62] | LEVEL | 100.0 -> 100.0 |
| multihoprag | `rrf1000_b50+FULL:h2_c100_v5000+top50` | frontier_1000 | 0.9912 | 0.9938 | +0.27 [+0.00, +0.62] | LEVEL | 100.0 -> 100.0 |
| multihoprag | `rrf1000_b50+FULL:h2_c100_v5000+top50` | frontier_1500 | 0.9912 | 0.9938 | +0.27 [+0.00, +0.62] | LEVEL | 100.0 -> 100.0 |
| multihoprag | `rrf1000_b50+FULL:h2_c100_v5000+top50` | frontier_2000 | 0.9912 | 0.9938 | +0.27 [+0.00, +0.62] | LEVEL | 100.0 -> 100.0 |
| multihoprag | `rrf1000_b50+FULL:h2_c100_v5000+top50` | frontier_2500 **(deciding)** | 0.9912 | 0.9938 | +0.27 [+0.00, +0.62] | LEVEL | 100.0 -> 100.0 |
| multihoprag | `rrf1000_b25+FULL:h1_c100+top25` | R95_bounded **(deciding)** | 0.9690 | 0.9717 | +0.27 [-0.18, +0.80] | LEVEL | 50.0 -> 50.0 |
| multihoprag | `equal_rrf_budget_50` | old | 0.9540 | 0.9442 | -0.97 [-2.04, +0.00] | WORSE | 50.0 -> 50.0 |
| musique | `rrf1000_b400+FULL:h1_c100+brg100` | frontier_500 | 0.7683 | 0.7724 | +0.41 [-0.54, +1.37] | LEVEL | 498.1 -> 498.4 |
| musique | `rrf1000_b800+FULL:h1_c100~s10+top200` | frontier_1000 | 0.8213 | 0.8312 | +0.99 [+0.25, +1.78] | BETTER | 986.6 -> 988.1 |
| musique | `rrf1000_b800+FULL:h2_bf_v1000~cs-vr-rr+top800` | frontier_1500 | 0.8618 | 0.8672 | +0.54 [-0.00, +1.12] | LEVEL | 1470.0 -> 1485.4 |
| musique | `rrf1000_b400+FULL:h2_bf_v2000~cs-vr-rr+top1600` | frontier_2000 | 0.8664 | 0.8676 | +0.12 [-0.54, +0.74] | LEVEL | 1999.5 -> 2000.0 |
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | frontier_2500 **(deciding)** | 0.8846 | 0.8862 | +0.17 [-0.37, +0.74] | LEVEL | 2438.2 -> 2456.3 |
| musique | `rrf1000_b800+FULL:h2_bf_v2000~cs-vr-rr` | R95_bounded **(deciding)** | 0.8846 | 0.8862 | +0.17 [-0.37, +0.74] | LEVEL | 2438.2 -> 2456.3 |
| musique | `equal_rrf_budget_200+FULL:h2_c25` | old | 0.8026 | 0.8113 | +0.87 [+0.08, +1.61] | BETTER | 2092.4 -> 2101.2 |

**D1 on the CPU basis: ADOPT_C3_AT_PRICE (superseded: section 10.6 applies D1 with C0's card rate, which landed three minutes after this section was committed, and gives C0_STAYS).** Five of the six deciding cells are LEVEL (SQuAD frontier_2500 +0.00; MultiHop-RAG +0.27 on both; MuSiQue +0.17 on both, one cell serves both roles there), one is WORSE: SQuAD's R95_bounded pool of 15 nodes, -0.91 point [-1.16, -0.65], which is a first-stage-only pool (no walk). Every point difference is above -1.5. No cell has MORE_NODES (largest +2.5% of nodes, SQuAD frontier_2500). The speed ratio is 6.44 (CPU, first 512 SQuAD nodes, 4 threads: C3 15.21, C0 2.36 nodes per second), above 3.0.

**The price, in numbers.** The loss sits in the smallest pools: 15 nodes on SQuAD, -0.91 point of all-gold, and MultiHop-RAG's old 50-node cell, -0.97 [-2.04, +0.00] (not a deciding cell). Every other cell, from SQuAD's 50-node old cell up to 2,500 nodes, is LEVEL or BETTER; MuSiQue is BETTER on two cells (+0.99 at 1,000 nodes, +0.87 on the old cell). The first-stage screen's SQuAD losses (RRF R@5 -1.04, seeds -1.65) show up in a pool that has no walk and are gone from the pools that walk (read as a pattern of these cells, not a tested mechanism).

**Expectations.** X5 (SQuAD's deciding cells LEVEL): **not held**, the 15-node cell is WORSE (the walk does not repair a pool that is only the first stage). X6 (MuSiQue loses most): **not held**, MuSiQue is LEVEL or BETTER on all seven cells although C3's own leg R@1000 there is -2.11 WORSE (the pools repair it; the screen's first-stage read mispredicted the pool read on this dataset). X7 (MultiHop-RAG LEVEL or BETTER): **held** on all deciding cells (the old cell, which decides nothing, is -0.97 WORSE).

### 10.3 Costs

| measure | C0 original 109M | C3 23M | C1 30M | C2 67M |
|---|---|---|---|---|
| card nodes/s, block median, squad | - | 343.8 | 287.9 | 266.4 |
| card nodes/s, block median, musique | - | 27.2 | 337.3 | 321.1 |
| card nodes/s, block median, multihoprag | - | 127.8 | - | 109.7 |
| card nodes/s, whole run (wall), squad | - | 36.3 | 33.6 | 189.3 |
| card nodes/s, whole run (wall), musique | - | 25.2 | 53.8 | 24.4 |
| card nodes/s, whole run (wall), multihoprag | - | 83.6 | 74.8 | 72.1 |
| doc non-zeros per node, squad | - | 340.4 | 269.2 | 245.4 |
| doc non-zeros per node, musique | - | 315.4 | 200.8 | 211.0 |
| doc non-zeros per node, multihoprag | - | 336.9 | 368.2 | 265.9 |
| question non-zeros, squad | - | 11.9 | 57.2 | 134.5 |
| question non-zeros, musique | - | 18.4 | 64.2 | 142.4 |
| question non-zeros, multihoprag | - | 45.2 | 120.3 | 199.9 |
| CPU nodes/s, first 512 SQuAD nodes, 4 threads | 2.36 | 15.21 | 10.81 | 6.90 |
| doc non-zeros, the same 512 nodes | 193.6 | 337.0 | 262.3 | 245.5 |
| doc non-zeros, MultiHop-RAG on CPU (all 609) | 213.2 | - | - | - |

G-C0 (CPU row): this 213.2 vs record 213.2, difference +0.00: PASS

C3's MuSiQue row ran on the host CPU (6 threads, 25.2 nodes per second wall) because the other account held the card from 03:45; its "block median" in the table is a CPU rate there. C0's card speed was not measured: `spl1c-gsq` and `spl1c-gmh` stay in the feed and their card rates are filed beside this table when the card lets them run. **C3's vectors are not smaller in non-zeros:** 315 to 340 document non-zeros per node against C0's 213.2 (MultiHop-RAG) and 193.6 (the 512 SQuAD nodes), about 1.6 times more; its question side is smaller (11.9 to 45.2 non-zeros against C0's 128.1 on MultiHop-RAG). SPL2 keeps the top-2,500 lists per question and not the document vectors, so this costs no storage there.

### 10.4 Disclosures

- D1's text says the CPU ratio stands in "if C0's card row fails". The card row has not failed: it has not run, because the card has been held since 03:45. The module's coded fallback ("the card row of C0 is not here") was used as committed in A8 before any pool number was read. The card ratio is filed when the row runs; if it falls below 3.0 the user is told, since that would change a premise of this decision. (It did: section 10.6.)
- The pool read uses the record's frozen dense list (the filed grids exist on it), not e3v's; the first-stage effect on e3v is the H1 and H2 reads of 10.1.
- C1 and C2 were not read on the pools (they decide nothing under D1: slower than C3 and not better on the first stage).
- C3's MuSiQue list was encoded on CPU, the others on the card; the lists are device-independent up to float noise (SPL2's dry run: scores within 4.2e-5 between CPU and card on MultiHop-RAG).
- The laptop's Prime SPLADE leg (C0) is redundant once C3 serves Prime; it is not stopped by this page (it was reaped by the system's memory guard on 27 Nov at about 04:50 with its checkpoint at 49,152 of 129,375 nodes; it was not restarted).

### 10.5 What followed (superseded by 10.6)

The rebuild of the sparse leg over all ten datasets with C3 was declared on `docs/SPL2_SPARSE_LEG_REBUILD.md` (0be8ccce) and its nine rows were put in the feed at 06:06. One of them ran (`spl2-mhr`, MultiHop-RAG, 17 seconds on the card, 609 nodes) before section 10.6 withdrew them.

### 10.6 C0's card rate landed: D1 gives C0_STAYS

`spl1c-gsq` (C0 on the host card, SQuAD, 20,233 nodes) finished at about 06:08, after section 10.2 was committed at 06:05 on the CPU basis (A8's coded fallback, "the card row of C0 is not here"). D1 names the card rate as the speed ratio and the CPU ratio only as a stand-in, so the card rate decides:

| measure (SQuAD, card, micro-batch 16, fp32) | C0 | C3 | ratio |
| --- | --- | --- | --- |
| median per-block rate, nodes per second | 199.61 | 343.83 | **1.72** |
| document non-zeros per node | 195.2 | 340.4 | 1.74 more |
| whole run (wall; the guard suspends rows) | 19.74 | 36.26 | 1.84 |

The CPU ratio of the first 512 nodes (6.44) overstated the card's: a small model on the card is probably limited by per-call overhead more than by arithmetic (not tested here). **D1: C0_STAYS (the card speed ratio is below 3.0)**, `outputs/spl1/pool/call.json` (the CPU-basis call is kept as `call_cpu_basis.json`). The decision of 10.2 is not carried; its pool reads (the six deciding cells and the price) stand as read.

- **What this means.** C3 is 1.72 times faster than C0 on the card, with 1.7 times more document non-zeros per node and a loss of 0.91 point on SQuAD's 15-node pool. The user's premise ("6x faster with barely a 1 point recall loss seems worth it") was the CPU ratio; on the card it does not hold, and D1's pre-declared threshold (3.0) is not met. C3 is not adopted.
- **SPL2 is withdrawn.** `docs/SPL2_SPARSE_LEG_REBUILD.md` and `zspl2.py` stay in the repository as written, no row of them is in the feed, and the one row that ran (`spl2-mhr`, C3 on MultiHop-RAG) is not used by anything. The seven datasets keep the record's SPLADE lists (C0); only the three STaRK KBs need a sparse leg.
- **The A6 rows return as declared.** The STK1 A6 rows that A8 took out of the feed (C0 on the host card for Prime, Amazon and MAG, then the first-stage rows that read them: `stk1p-spg`, `stk1p-spl`, `stk1p-first2`, `stk1a-spl`, `stk1a-first2`, `stk1m-spl`, `stk1m-first2`) are put back unchanged, except that `stk1p-spg` no longer waits for `spl1c-gmq` (C0's MuSiQue card timing decides nothing now and is not returned). At C0's card rate the three legs are about 11 minutes (Prime, 129,375 nodes), 1.4 hours (Amazon, 1,035,542) and 2.6 hours (MAG, 1,872,968) of an exclusive card.
- **Speed levers (A8's declared follow-up) are not run.** fp16, larger micro-batches and a masked pooling are options only for C0's three STaRK legs; fp16 and the micro-batch change values (C0's contract pools over padding, so a vector depends on its batch), and the 4-hour card cost does not justify a screen. If the user wants them, they are a new amendment.
- **The laptop's Prime SPLADE leg (C0 on CPU)** is redundant: the card does Prime in minutes. It was reaped by the memory guard at about 04:50 and is not restarted.
