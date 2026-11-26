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

## 9. Expectations, written before any number

- **X1.** C2 (the highest printed score) is BETTER or LEVEL on all six cells; it is the largest of the three, so it is the pick only if C1 and C3 do not pass.
- **X2.** C1 (30M, the printed score level with C0) is LEVEL on SQuAD and MuSiQue; the risk is MultiHop-RAG, where 256 tokens of a long article is all any of them reads.
- **X3.** C3 (inference-free: no question expansion) is WORSE on MuSiQue at H2, because its multi-hop questions are the ones that need expansion most.
- **X4.** No candidate is BETTER than C0 by more than 2 points on H1: the leg is one of two in a fusion that already holds e3v, so a better leg moves the fused read less than it moves the leg alone.

Expectations are not calls; each is read against the table and marked held or not held in section 10.

## 10. Result

Empty. Nothing has run.
