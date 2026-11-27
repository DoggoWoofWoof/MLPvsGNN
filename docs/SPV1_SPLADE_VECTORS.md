# SPV1: the best SPLADE leg's vectors, for every node and every question of all ten datasets

Declared 19:24 IST, 27 Nov 2026, before any row of this page has run. No retrieval number is read here: this page stores vectors and reads their fidelity.

## 0. The user's words, and the rulings that bind

- 27 Nov: "if we finish dense encodes quick put out best splade on queue we will atleast have the vectors and cleanup everything later we if removal is better or doesnt matter we will delete splade later".
- 27 Nov, after the dense vectors were found complete: "i told you to finish the best splade we found on all datasets right? make sure all the link level and pool and encoding work is fnished" and "and we will get a time estimate too from this".
- Standing: one rule for all ten datasets; no BM25 (STK1 A4); the SPLADE leg's fate (keep or remove) stays the user's and is not decided here; nothing is deleted by this page; every new number needs a declared rule first.

## 1. What is stored, and why this leg

The leg is **C0 under SPL1 A11's lever l5** (`naver/splade-cocondenser-ensembledistil` at the pinned weights in `outputs/spl1/models/c0`, half-precision weights, forward passes of 32 rows, `log1p(relu(max over positions of the raw logits))`): the leg SPL1 kept (C0_STAYS), the lever STK1 A8 gated (G-L) and the first stage of the three STaRK KBs used. The candidates c1 to c5 were screened in SPL1 and lost the speed-to-recall trade; none is encoded here.

For each of the ten datasets (`DATASETS` in `zspv1.py`): **every node** (the world's `nodes.jsonl`, line i is node i) and **every question of every split** (the world's query rows, in row order). Per dataset folder `outputs/spv1/<ds>/`: `shard_NNNNN.npz` (65,536 nodes each), `queries.npz`, `gate.json`, `manifest.json`, and for the three STaRK KBs `replay.json`.

**Format.** Per shard: `indptr` int64, `indices` uint16 (the vocabulary is 30,522), `data` float16, `n_cols`. A stored vector is the l5 vector after a float16 round trip; no cap, no pruning, no mixing, no other change (LEG1's document and question caps stay open reads and are not applied). **Batching.** C0 pools over padding, so a vector depends on its batch: batches are the record's 64 rows from index 0, the shard size (65,536) is a multiple of 64 and of the 2,048-row block, so a resumed run reproduces the vectors of an uninterrupted one.

## 2. Gates and checks

- **G-V (per dataset, before anything is stored).** On the first 2,048 nodes and the first 2,048 questions, the stored form (l5, uint16 and float16 round trip) against the record's own pooling (arm l0, float32): the first percentile of the per-vector cosine at least 0.999 and the non-zero count within 1.0 percent (SPL1 A11's E2; STK1 A8's G-L). A failure stores nothing for that dataset and stops it.
- **check (per dataset, after).** Every shard's sha256 against the manifest, row counts add to the node count, every stored index below the vocabulary.
- **replay (information, the three STaRK KBs).** The first 64 test questions scored against the stored shards, top 5, 20, 100 and 1,000 as sets against the filed first stage's SPLADE list (`first.npz`, `top_splade`). No gate: it says how much the half-precision storage moves a list.
- **Free space.** An encode stops before a shard if the host volume has under 25 GB free (the run resumes from its last shard).

## 3. Cost: estimate, written before any row runs

The card is idle and ours. Rates, measured on an exclusive card: A7's timing of l5 (Prime 1,687, Amazon 1,065, MAG 4,481 nodes a second) and SPL1 A11's blocks (SQuAD 897, MuSiQue 1,022, MultiHop-RAG 293 nodes a second); the other four datasets are priced from their node text (mean characters: MetaQA 13, WebQSP 29, HotpotQA 272, 2Wiki 498) against those rates, so **their estimates are the loosest** (a factor of two either way). An encode here does no top-k merge, so the SPL1 wall rates (which include it) are lower bounds.

| dataset | nodes | questions | rate used (nodes/s) | estimate |
|---|---|---|---|---|
| MultiHop-RAG | 609 | 2,255 | 293 | under 1 min |
| SQuAD | 20,233 | 142,192 | 900 | about 2 min |
| MetaQA | 43,234 | 407,513 | 4,000 | about 4 min (the questions dominate) |
| MuSiQue | 117,534 | 24,814 | 1,000 | about 2.5 min |
| STaRK Prime | 129,375 | 11,204 | 1,687 | about 1.5 min |
| STaRK Amazon | 1,035,542 | 9,100 | 1,065 | about 16 min |
| STaRK MAG | 1,872,968 | 13,323 | 4,481 | about 7 min |
| WebQSP | 2,592,894 | 4,737 | 4,000 | about 11 min |
| HotpotQA | 5,233,329 | 105,257 | 1,000 to 2,000 | 45 to 90 min |
| 2Wiki | 5,989,847 | 192,606 | 1,000 to 1,800 | 55 to 100 min |

**Total: about 2.5 to 4 hours of card time**, plus 1 to 3 minutes per large dataset to stream the node file. Rows run side by side on one card without adding throughput; if another user takes the card, `gpu_guard.py` suspends the rows and they resume from their last shard (a few minutes lost at most). **Storage:** 4 bytes a non-zero and 8 a row; with the A11 non-zero means (SQuAD 195, MuSiQue 156, MultiHop-RAG 213) and the short-node KBs near 15 to 40, the ten datasets come to about **9 GB** (2Wiki 4 GB, HotpotQA 2.6 GB, Amazon 0.8 GB), against 66 GB free on the host.

## 4. Expectations, written before any number

- **V1.** G-V passes on all ten datasets.
- **V2.** The stored non-zero mean of SQuAD, MuSiQue and MultiHop-RAG matches A11's record (195.1, 155.7, 213.1) within 1.0 percent (the same vectors, so the same counts).
- **V3.** Every dataset finishes within twice its estimate in section 3, and the whole job within 8 hours, on an uncontended card.
- **V4.** The ten datasets together store at most 12 GB.
- **V5.** On each STaRK KB the replay's top-20 set agreement with the filed list is at least 0.95 (the storage moves only the sixth significant digit of a value).

## 5. Code, rows, outputs

New module `outputs/mp_unified/zspv1.py` (declared here; edit only through an amendment on this page; it imports `zspl1`, `zspl1c`, `zspl1l` and `zworld` and edits none of them). Commands: `selftest` (13 checks, laptop, no model), `encode --ds D [--smoke N] [--min-free-gb 25]`, `check --ds D`, `replay --ds K`. Rows (host, classes `spv1-<dataset>`, `spv1r-<kb>`; 2 CPUs, 8 GB, 0.25 of the card; no row depends on another, so a failure on one dataset stops only that dataset; each encode is wrapped in `retry_cmd.py --tries 2`): `spv1-multihoprag`, `spv1-squad`, `spv1-metaqa`, `spv1-musique`, `spv1-stark_prime`, `spv1-stark_amazon`, `spv1-stark_mag`, `spv1-webqsp`, `spv1-hotpotqa`, `spv1-2wiki`, then `spv1r-prime`, `spv1r-amazon`, `spv1r-mag` after their encodes. The vectors stay on the host (about 9 GB); the manifests, gate and replay records come to the laptop and are filed on this page as section 7.

## 6. Not read, and what follows

Not read: any retrieval number from these vectors; any cap, mixing or modified-vector read (LEG1's open parts); the decision to keep or remove SPLADE (the user's; these shards would be deleted only on the user's per-action OK); ING1's registry entry for the sparse stage (an ING1 amendment follows the filing). **Nothing is adopted; no delete.**
