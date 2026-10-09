# The paper's three claims and what makes each one bulletproof

Written 9 October 2026, from the user's direction of the same day. The direction: a bulletproof paper showing that
the MLP is close enough to the GNN, that we can build a SOTA GNN without relying on any LLM, and that the MLP is
faster. Each claim's definition is fixed here, before the numbers that test it. The stages that produce the evidence
are those of docs/PROGRAM_2026_10_09.md. The M3 firewall still holds: no MLP-against-GNN number is cited before M4.

## What stays fixed (the base)

- The frozen encoder (gte-Qwen2-1.5B-instruct) and SPLADE, and the package's vectors. Nothing is re-encoded or
  fine-tuned.
- Graphs built without any LLM: the datasets' own links, title mentions, spaCy entities, and exact kNN. A new
  LLM-free graph element (for example, entity nodes from the spaCy entities) is allowed. It is built by a declared
  rule, the same on every dataset.
- The current models (zrc MLP, zsp GNN) as the starting point. Every change is an addition on top of them, declared in
  its own file, the same on all six datasets, and graded on all six, zero-shot included.

## Claim 1: the MLP is close enough to the GNN

- **Definition.** The MLP uses no message passing at inference: every input is computed from the graph and the
  question before any row is scored, and no row's score or hidden state reaches another row.

  "Close" means both of the following:
  - the MLP keeps at least 90% of the GNN's R@5 gain over the first stage, with a 95% bootstrap interval, on each of
    the six datasets in-domain;
  - zero-shot, the MLP is not BELOW the GNN by more than the in-domain gap.

  The 90% bar is fixed now. If a dataset misses it, that dataset is reported as missing it.
- **Evidence.**
  - Both models read the same features from the same pools, carves and seeds.
  - Results are given on all six datasets, on leave-one-dataset-out zero-shot, and on HippoRAG 2's three settings
    (B1).
  - There are confirmation runs on held-out questions never used for selection.
- **Attacks answered in advance:**
  - *"The features are message passing in disguise."* The answer is the definition above, plus a feature-by-feature
    table of what each input reads.
  - *"Close only because the GNN is weak."* Claim 2 answers it: the GNN is the strong one.
  - *"Cherry-picked datasets."* All six are reported, plus the B1 settings.

## Claim 2: a SOTA GNN without any LLM

The user's ruling of 9 October: we aim higher than "best LLM-free". The claim is that **better features and models
beat a better encoder**. HippoRAG 2 has a 7B encoder and an LLM-built graph; we keep our frozen 1.5B encoder and win
on features and models. The encoder risk is answered by measurement (below), not by changing our encoder, which stays
frozen.

**What "without any LLM" means.** No generative LLM call (no text generation) at indexing or at query time.
Embedding models are encoders, even when they are built on a language-model backbone. Our gte-Qwen2-1.5B is built on
Qwen2-1.5B, and NV-Embed-v2 on Mistral-7B, so the definition has to say this. By it:
- **Using a generative LLM:** HippoRAG 2 (OpenIE triples and the recognition filter), GFM-RAG and G-reasoner (an
  LLM-built graph at indexing), and RAPTOR (LLM summaries).
- **LLM-free:** the dense retrievers, and our system.

**Definition (the headline).** On HippoRAG 2's MuSiQue, 2WikiMultiHopQA and HotpotQA settings (B1: the same 1,000
questions and corpora, passage R@5), our GNN's R@5 is compared with HippoRAG 2's published R@5 (74.7 / 90.4 / 96.3,
Table 3):

| call | the 95% bootstrap interval of our R@5 (2,000 resamples of the 1,000 questions, seed 0) |
| --- | --- |
| ABOVE | lies wholly above the published number |
| AT | contains it |
| BELOW | lies wholly below it |

The claim holds on a setting when the call is ABOVE or AT, and is reported setting by setting. A setting that is
BELOW is reported as BELOW. Over our first stage (RRF 56.4 / 71.2 / 85.0), reaching HippoRAG 2 needs
+18.3 / +19.2 / +11.3 points.

**Encoder risk, answered by measurement.** A reviewer will say "you are behind or ahead because of the encoder". Each
answer below keeps our encoder fixed.
- **E1, the method's lift against the method's lift.** Our GNN's R@5 gain over our own dense list, set beside each
  published system's gain over its own encoder's dense list. HippoRAG 2 over NV-Embed-v2 dense gains
  +5.0 / +13.9 / +1.8. If our lift is larger, our features and models add more than theirs, on any encoder. This
  comparison is fixed now as the paper's main encoder-free statement.
- **E2, a weaker first stage.** The same trained models, fed the SPLADE-only first stage (50.8 / 70.5 / 80.3, the
  package's own vectors), and with the dense list removed from the inputs where the model allows it. If the lift holds
  on a weaker first stage, the gain is not the dense encoder's.
- **E3, matched encoder.** The other graph methods on our encoder and our graphs:
  - HippoRAG 2's personalised-PageRank core (B1c: SAME on all three);
  - GFM-RAG's and G-reasoner's released GNNs (B2).
- **E4, the cost.** Encoder size (1.5B against 7B), LLM calls at indexing (0 against one per passage), and indexing
  time, beside accuracy.

The best LLM-free published number (NV-Embed-v2 dense, 69.7 / 76.5 / 94.5) stays in the table as a second line. It is
no longer the target.

- **Evidence.**
  - B1 settings with golds verified (agreement 1.0).
  - Selection never touches the benchmark questions as a whole. Choices are made on our own training carves, or on
    the even half, and reported on the odd half and on all of them.
  - Published numbers are quoted from the papers' tables, with table numbers given.
- **Attacks answered in advance:**
  - *"Smaller encoder."* E1–E3. And with a smaller encoder, a win is stronger, not weaker.
  - *"Tuned on the test questions."* Answered by the even/odd rule and the dev-split provenance.
  - *"An LLM-free graph is weaker by construction."* Show what each LLM-free graph element buys, family by family.
- **The gap, as measured on 9 October** (B1a, B1c):
  - Our first stage is 18.0, 19.6 and 11.8 points below HippoRAG 2.
  - Of that, 13.0, 5.7 and 10.0 points are the encoder (NV-Embed-v2 dense against our dense).
  - The misses are second steps of chains: 86% to 99.5% have a sibling gold already in the top 5, and 1 hop from the
    dense top 5 holds 87% / 97% / 99% of all golds. The fix is bridge scoring and entity-level links (B1d, G1, an
    LLM-free entity-node graph).

## Claim 3: the MLP is faster

- **Definition.** Cold end-to-end latency for a new question:
  - from the first-stage lists to the top 5, with no feature cache and no warm-up;
  - both models in their fastest exact serving forms;
  - p50, p95 and p99 at batch 1 and batch 16, with 95% bootstrap intervals;
  - the laptop pinned as in docs, and the host CPU.

  The cost is split into graph index, query-local feature build and forward (C1). The headline is the cold ratio. A
  warm ratio, if shown, sits beside it and is labelled as warm.
- **Where it stands:** the filed cold ratio is about 1.5× (2wiki, older models). The 6–9× was warm. C1 measures the
  current models and then attacks the shared feature build, which dominates the MLP's time.
- **Attacks answered in advance:**
  - *"Warm caches."* The headline is cold.
  - *"The GNN was not optimised."* The GNN runs in its exact fast form.
  - *"Speed traded for accuracy."* Claim 1's numbers come from the same models that are timed.

## Novelty to keep

- **The analysis:** message passing buys full coverage at 5, not hit@1; misses are reachable but not ranked; the
  bridge ledger. No published system reports these.
- **The setting:** set retrieval on unseen questions and unseen graphs, passage and KB, zero-shot. SGC, SIGN, GLNN,
  SA-MLP and RTA study node classification.
- **The construction:** an LLM-free graph and pipeline with a frozen mid-size encoder, with its cost stated.
