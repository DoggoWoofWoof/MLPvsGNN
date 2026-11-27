# SPL2: the sparse leg of the first stage, rebuilt over all ten datasets with the leg SPL1 chose

**WITHDRAWN 27 Nov 2026, 06:11 IST (before any list of it was used).** The page was declared on D1's CPU-basis call (ADOPT_C3_AT_PRICE). C0's card rate landed at about 06:08 (SQuAD: C0 199.61, C3 343.83 nodes per second, ratio 1.72) and D1, which names the card rate, gives C0_STAYS (`docs/SPL1_SPARSE_LEG_SCREEN.md` section 10.6). The nine rows were taken out of the feed at 06:09; one ran first (`spl2-mhr`, MultiHop-RAG, 17 seconds on the card) and its output (`outputs/spl2/multihoprag` on the host) is not used. No other row ran. The page and `zspl2.py` stay as written; a later rebuild with another leg is a new page.

Declared 27 Nov 2026, 06:06 IST, after the decision of rule D1 (`docs/SPL1_SPARSE_LEG_SCREEN.md` sections 8.8 to 8.10, result in section 10) and before any SPL2 row has run. Follows `docs/ING1_INGESTION_AND_STAGE_REGISTRY.md` (registry rule set RS1, stage `first`: "e3v dense and SPLADE fused by reciprocal rank (C=60), depth 2,500"). Code: `outputs/mp_unified/zspl2.py` (a new module, written and checked before this page; committed with it).

## 1. Why this page exists

The user's words (27 Nov): **"first find the best models for our job whichever splade works best and is smaller and has even better recall we will use that and rebuild all the datasets, then our all 10 datasets will have the smallest possible encoders for both dense and splade and univerasal processing links etc"**, then **"its not about the size but more of the speed to recall loss atp ... do pool cov for c0 vs c3 and whatever else is worth and compare and finalize autonomusly, once done start redoing the encodings for all 10, 6x faster with barely a 1 point recall loss seems worth it see if we can do better than this"**.

SPL1 decided the leg on the pools (rule D1): **ADOPT_C3_AT_PRICE**: five of the six deciding cells LEVEL (SQuAD at 2,500 nodes +0.00; MultiHop-RAG +0.27 on both; MuSiQue +0.17), one WORSE (SQuAD's 15-node pool, -0.91 point [-1.16, -0.65]), the speed ratio 6.44 (CPU basis; C0's card row has not run) (`docs/SPL1_SPARSE_LEG_SCREEN.md` section 10.2). This page is the rebuild: one rule, every dataset, the new leg's top lists over the questions the later stages read.

## 2. The rule (the same for the ten)

- **The leg.** C3 (`opensearch-project/opensearch-neural-sparse-encoding-doc-v2-mini` at revision 4af867a4, 23M, inference-free question side), loaded from the host's own folder with `local_files_only`, the library's own recipe, cut at 256 tokens, micro-batch 16, fp32. Nothing is tuned; this is SPL1's contract unchanged.
- **The nodes.** Every node of the dataset's package, in file order (`zworld` for the seven, the STaRK package for Prime, MAG and Amazon). Node texts are streamed in blocks, never held whole.
- **The questions.** The seven: the dev and eval carves of PL2 (`outputs/pl2/<dataset>/pop_dev.npz` and `pop_eval.npz`, union), MultiHop-RAG: all 2,255 questions. STaRK: the test questions. Train questions are not listed here (SPL3, before any model).
- **The list.** The best K = 2,500 nodes per question over all blocks (the registry's depth for `first`; a superset of the top 1,000 that the pools read), scores that are not positive are not candidates, ties go to the lower node id. Blocks of 2,048 nodes; the search of block t runs in a second thread while block t+1 is encoded; the running top-K is checkpointed at least every 150 seconds (a loss is minutes).
- **Order.** Smallest first: MultiHop-RAG, SQuAD, MetaQA, MuSiQue, Prime, WebQSP, MAG, HotpotQA, 2Wiki; Amazon when its package passes the STK1 checks. Rows are GPU rows (the card is shared; `gpu_guard` yields to the other account and takes the card back).
- **Outputs.** `outputs/spl2/<dataset>/top.npz` (`ids` int32 (questions, K), `scores` float32, `rows` int64), `top.json` (provenance: model revision and file hashes, device, batch, block-rate median, non-zeros, nodes cut at 256, code and list hashes).

## 3. Gates

- **G-S0 (code):** `zspl2.py selftest` passes (the running top-K equals a one-shot search; a restart resumes and gives the same lists; a state of another run is not resumed; the node stream; the rows rule). Passed before this page.
- **G-S2 (the module against SPL1):** on MultiHop-RAG the module's lists equal SPL1's C3 lists (set overlap 1.0, scores within 1e-3). Read before this page as a dry run on the host CPU: set overlap 1.0 over 2,255 questions, same-slot share 0.99986, largest score difference 4.2e-5. (Disclosure, section 6.)
- **G-S3 (speed, recorded not gated):** the median per-block rate of each row's encode calls is written into `top.json` (SPL1's rule: speed is the block median, never wall time, because the guard suspends rows).

## 4. What this page does not do

- No pool number and no first-stage number is read here. The RRF and the pool rows that read these lists are a later declared page (the STK1 first-stage rows and PL8; PL8 is not declared until the user says the link set is final).
- The laptop's Prime SPLADE leg (C0) is not stopped by this page. It is redundant once C3 serves Prime; the user decides.
- No train-question lists (SPL3).
- No speed screen (fp16, a larger micro-batch on a free card) on this page; that is an amendment, read on SQuAD first, if the user wants it.

## 5. Rows

`spl2-<dataset>` per dataset in the feed (GPU 0.15, 4 cores, memory by question count: 6 GB, 12 GB for 2Wiki and HotpotQA), command `python -I -B outputs/mp_unified/zspl2.py encode --ds <dataset>`, resumable by the same command. Estimated card time at SPL1's 343 nodes per second: MultiHop-RAG, SQuAD, MetaQA and MuSiQue minutes; Prime 6 minutes; WebQSP 2.1 hours; MAG 1.5 hours; HotpotQA 4.2 hours; 2Wiki 4.9 hours; Amazon 50 minutes at 1.0 M nodes (its package is not here yet). About 13 hours of exclusive card in all; the other account's use stretches it.

## 6. Disclosures

- Three dry runs of the module were made on the host CPU before this page, as code checks and not as results: MultiHop-RAG (all 609 nodes; the comparison with SPL1's lists above), a 512-node Prime sample and a 2,048-node 2Wiki sample (the loaders, the stream and the checkpoint). No coverage or recall number was read from them.
- C0's card speed on SQuAD was not measured (the card was held); D1 used SPL1's CPU ratio, as its rule allows.

## 7. Result

Empty until the rows finish.
