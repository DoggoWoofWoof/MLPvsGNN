# Dataset, Graph, and CRAG-Reuse Guide

This document explains what the standalone message-passing retrieval project
reuses from the original CRAG repository, what it deliberately does not reuse,
what the `clean` dataset label means, and what graph is supplied to each model.

Publication prose uses **QLS-MLP (Query-Local Structure MLP)**. The frozen
implementation key remains `sa_mlp`, and existing SA-named files/tags remain
unchanged; see [the terminology and positioning note](TERMINOLOGY_AND_POSITIONING.md).

## Executive summary

The project reuses a substantial amount of CRAG's **frozen experimental data**,
but almost none of the C-RAG **system or runtime code**.

The primary reused source is:

```text
CRAG/data/ukb_storage/<dataset>/gte_qwen/
```

From this storage, the standalone project consumes frozen node and query
embeddings, Dense and SPLADE candidate identities, query manifests and gold
labels, and global graph adjacency. It does not execute the C-RAG L1/L2/L3
architecture, router, expert fusion, partition routing, graph traversal, or
universal-KB retrieval system.

The most accurate description is:

> A standalone, controlled candidate-reranking study using frozen retrieval
> outputs and graph substrates originally constructed in CRAG.

Data reuse is substantial. Method and runtime-code reuse is minimal.

## What is reused from CRAG

Each complete frozen dataset normally supplies the following artifacts:

| Artifact | Meaning | Role in this project |
| --- | --- | --- |
| `nodes.npy` | Frozen GTE-Qwen embedding for every document or entity | Identical raw node features for all models |
| `queries_all.npy` | Frozen query embeddings | Query conditioning |
| `dense_top200_all.npy` | Dense retriever's top-200 global node IDs | Candidate construction |
| `splade_top200_all.npy` | SPLADE's top-200 global node IDs | Candidate construction |
| `query_ids_all.json` | Query IDs, split assignments, gold nodes, and metadata | Labels, splits, and evaluation |
| `graph.pt` | Global document or entity adjacency | Fixed structural summaries and GNN message passing |
| `node_ids.json`, where available | Original KB entity-to-row mapping | Alignment validation |

For each query, the candidate pool is the stable union:

```text
Dense top-200 followed by previously unseen SPLADE top-200 candidates
```

The resulting pool contains at most approximately 400 nodes. Candidate
identity and order are frozen and shared by all compared models. The Dense and
SPLADE *outputs* are reused; their online retrieval runtimes are not part of the
standalone experiment.

The frozen arrays contain ranked IDs, not raw Dense/SPLADE scores. The current
stable union is not Reciprocal Rank Fusion. Standard RRF can nevertheless be
computed exactly from the retained rank positions and is registered as future
work. RRF over the unchanged union can alter the initial ranking or structural
seeds but cannot recover a gold node absent from both top-200 lists.

## What UKB means here

Two distinct ideas can otherwise be confused.

### `data/ukb_storage`

This is CRAG's persistent per-dataset storage and index layout. It contains
embeddings, graph tensors, candidate arrays, manifests, indexes, partitions,
checkpoints, results, and other cached products.

The standalone project relies heavily on selected frozen data products stored
there.

### The C-RAG universal-KB system

This is the larger systems architecture: the unified substrate, L1/L2/L3
retrieval stages, routing, expert selection and fusion, partition search, and
graph traversal.

The standalone project does not run this system. Using files from
`ukb_storage` does not mean that the UKB/C-RAG retrieval architecture is being
used.

For example, the following may be present in CRAG storage but are not inputs to
the final candidate-reranking comparison:

- `partition_map.json` and METIS partitions;
- centroids and partition-routing state;
- the BM25 index and online BM25 runtime;
- C-RAG routers and expert fusion;
- L3 graph traversal;
- CRAG checkpoints and system results.

## Post-retrieval experimental flow

```text
Read-only CRAG UKB storage
    |
    +-- frozen node/query embeddings
    +-- frozen Dense/SPLADE candidate IDs
    +-- frozen query splits and gold labels
    +-- frozen global graph adjacency
                 |
                 v
Standalone data contract
                 |
                 +-- candidate pool of at most approximately 400 nodes
                 +-- graph induced over exactly those candidates
                              |
                              +-- Plain MLP: embeddings only
                              +-- Seed-only MLP: embeddings + seed indicator
                              +-- QLS-MLP: embeddings + fixed graph summaries
                              +-- GNN: embeddings + seed indicator + adjacency
```

The runtime-facing interface is therefore a query embedding, upstream Dense and
SPLADE ranked IDs, and a frozen corpus graph. The standalone project owns
candidate fusion, candidate-induced topology, fixed summaries or learned
message passing, and candidate ranking. It is not a raw-text search engine,
UKB runtime, C-RAG system, or full RAG pipeline.

The GNN does not normally propagate over the complete 781,485-node WebQSP
graph for every query. Given a query candidate set \(C_q\), the loader extracts
the exact candidate-induced subgraph:

\[
E_q = \{(u,v) \in E : u \in C_q, v \in C_q\}.
\]

Every comparison therefore uses the same query, raw embeddings, candidate
identities and ordering, gold labels, split, loss, and seed protocol. The
controlled difference is how graph information is made available to the
learner.

## What `clean` means

`Clean` does not mean that the text was merely spell-checked, that all
benchmark limitations were eliminated, or that the graph has no synthetic
edges.

In CRAG, `clean` primarily means that the document graph was rebuilt to remove
**label leakage**.

Earlier 2Wiki/MuSiQue-style construction could connect documents because they
appeared together as gold supporting documents for a question. Such co-gold
edges leak evaluation structure directly into graph topology. The clean
builders instead:

1. deduplicate passages using title/content identity;
2. construct label-free document relationships, primarily title mentions;
3. keep question-to-gold associations only as labels;
4. exclude question nodes from `graph.pt`;
5. exclude document-to-question backedges;
6. exclude co-gold or co-supporting-document bridge edges; and
7. add embedding-nearest-neighbor edges during indexing.

Therefore:

> `Clean` means that answer and supporting-document labels were not used to
> construct document-to-document topology.

It does not mean that the graph is purely human-authored or that it contains no
embedding-based kNN edges.

## What graph is actually used

Each dataset has one frozen global graph whose nodes are retrievable documents,
passages, or KB entities. Its edges generally originate from two sources.

### Structural edges

For passage datasets, these commonly encode label-free title mentions. If
passage A mentions the title of passage B, a structural connection can be
created.

For MetaQA and WebQSP, structural connectivity originates from native KB
triples.

### Embedding kNN edges

CRAG's indexer adds approximately three nearest-neighbor connections per node
using the frozen GTE-Qwen node embeddings. These edges make sparse regions more
connected and relate semantically similar nodes.

The graph is made effectively undirected by adding reverse adjacency entries.
The reported graph edge counts are therefore adjacency entries, not
necessarily unique undirected relations or KB facts.

### Important edge-type limitation

The final `graph.pt` stores flattened, untyped adjacency. The standalone study
cannot reliably determine from that tensor whether an individual edge was:

- a title-mention edge;
- a native KB-relation edge; or
- a synthetic embedding-kNN edge.

This is now a submission-critical confound rather than a minor metadata gap.
Because the kNN edges were constructed from the same frozen embedding family
used as node features, a reviewer can reasonably ask whether graph gains recycle
semantic geometry already available to every model. Before a final paper, the
standalone export must recover source sidecars, prove that their union exactly
reconstructs the frozen adjacency, and compare **native/title/KB-only**,
**embedding-kNN-only**, and **union** graphs for both QLS-MLP and the seed-aware
GNN. No change may be made to the read-only CRAG repository.

This is **Package B** in the canonical future plan and is mandatory before
submission.

Likewise, the MetaQA and WebQSP GNNs receive entity connectivity but not raw
relation labels such as `directed_by` or `acted_in`. The current paper should
therefore make claims about topology or adjacency, not typed relation
semantics.

## Dataset-by-dataset guide

| Dataset | Nodes | Frozen adjacency entries | What a node represents | Main graph source | Important caveat |
| --- | ---: | ---: | --- | --- | --- |
| 2Wiki | 65,865 | 855,146 | Deduplicated Wikipedia passage | Label-free title links plus embedding kNN | Meaningful distractor corpus |
| MuSiQue | 13,672 | 280,108 | Deduplicated passage | At minimum embedding kNN; any recovered title links are flattened | Current source is effectively gold-heavy |
| HotpotQA | 507,494 | 16,223,058 | Wikipedia article or passage | Label-free title links plus embedding kNN | Principal large passage/scaling dataset |
| SQuAD | 19,029 | 2,857,316 | Passage or article | Label-free structural/semantic links plus kNN | Mostly single-gold and effectively gold-heavy |
| MetaQA | 40,151 | 585,728 | Movie-domain KB entity | Native KB triples plus embedding kNN | Relation types are flattened; includes 1/2/3-hop queries |
| WebQSP | 781,485 | 13,379,166 | Freebase entity | Native Freebase triples plus embedding kNN | Small query set and limited candidate ceiling |

### 2Wiki

Each node is a canonical, deduplicated Wikipedia passage. The clean graph
excludes connections inserted because two documents jointly support a gold
answer. It contains label-free document relationships and embedding kNN edges.

2Wiki has a meaningful distractor corpus. It is useful for testing whether
neighborhood aggregation improves multi-document coverage or instead
introduces irrelevant context.

### MuSiQue

Nodes are passages used for compositional multi-hop questions. The safest
statement about the frozen graph is that it contains embedding-kNN topology.
Title-recovery and structural-processing utilities existed in CRAG, but the
final adjacency is untyped, so this study should not claim a precise final
edge-type composition without reconstructing provenance from source assets.

The present frozen source is relatively gold-heavy. Its absolute recall should
not be described as representative of unconstrained open-corpus retrieval, but
matched comparisons between models remain informative.

### HotpotQA

HotpotQA is the primary large-scale passage experiment. The final frozen
artifact contains 507,494 nodes and 97,852 queries. This is larger than the
66,573-node/6,162-query count in an older CRAG README, which referred to an
earlier substrate rather than the final complete artifact.

Its topology contains label-free document relationships plus embedding kNN.
Hotpot is particularly valuable for measuring scaling, latency, memory,
multi-answer coverage, and the behavior of message passing as graph and corpus
size increase.

### SQuAD

SQuAD is predominantly a single-answer passage-retrieval regime. It is a useful
control: if broad propagation is most useful for multi-document structural
coverage, its incremental value should be smaller here than on genuinely
multi-hop or multi-gold tasks.

The available frozen source is relatively gold-heavy, so absolute open-corpus
recall also requires careful wording.

### MetaQA

MetaQA is a native movie-domain KB rather than a Wikipedia passage graph. Each
node is an entity from a triple:

\[
(\text{subject}, \text{relation}, \text{object}).
\]

Entities are connected according to the KB triples. Node text verbalizes a
bounded number of incident facts and is embedded with GTE-Qwen; the common
indexer then adds kNN edges.

Questions contain answer entities as labels. Question nodes and
question-to-answer edges are not included in `graph.pt`. The final adjacency
also does not retain relation types. MetaQA is nevertheless a valuable test of
hop dependence because it has explicit 1-hop, 2-hop, and 3-hop question groups.

### WebQSP

WebQSP uses a large Freebase-derived entity graph. Original Freebase triples
provide structural adjacency, and entity text verbalizes incident relations
before embedding. The common indexer adds embedding-kNN edges.

As with MetaQA, answer entities are labels, questions are not graph nodes, and
relation types are flattened. Its global graph is the largest by node count,
but the frozen query set contains only 1,578 questions.

WebQSP's candidate pool misses gold entities for many queries. A reranker cannot
retrieve a gold node that is absent from its frozen candidate pool, so WebQSP
model results must always be interpreted alongside candidate coverage and the
candidate ceiling.

## What `complete dataset` means

`Complete` is an internal data-contract status, not a claim that the benchmark
is philosophically complete or that its candidate pool contains every answer.

A complete frozen source contains:

- all query embeddings expected by its manifest;
- aligned query identifiers;
- canonical train, validation, and test membership;
- Dense and SPLADE candidates;
- gold global node IDs;
- global node embeddings; and
- a graph whose node rows align with those embeddings.

Every registered query must occur in exactly one split, and candidate, query,
label, embedding, and graph identities must pass validation.

Completeness does **not** guarantee:

- that every gold appears in the candidate pool;
- that the source is a fully open-world corpus;
- that an effectively all-gold source provides realistic absolute recall; or
- that the original public benchmark has no omissions outside the frozen
  experimental source.

The candidate ceiling is particularly important. If a gold is absent from the
Dense/SPLADE union, no MLP, QLS-MLP, or GNN reranker can recover it.

## What each final model receives

| Model | Raw query/node embeddings | Retriever seed membership | Fixed graph summaries | Candidate adjacency in learned forward pass |
| --- | :---: | :---: | :---: | :---: |
| Plain MLP | Yes | No | No | No |
| Seed-only MLP | Yes | Yes | No | No |
| QLS-MLP | Yes | Yes | Yes | No |
| Seed-aware selected GNN | Yes | Yes | No | Yes |

The retrieval seed set is the stable union of the Dense top-5 and SPLADE top-5
candidate identities. QLS-MLP computes deterministic query-local summaries such
as distance/connectivity, path-count, and PPR-style signals from those seeds.
Its learned forward pass does not aggregate neighboring embeddings.

This yields the intended decomposition:

```text
seed-only - plain = value of the frozen retrieval prior
QLS-MLP - seed-only = value of fixed graph computation
QLS-MLP - GNN = fixed structural summaries versus learned message passing
```

QLS-MLP is therefore **non-message-passing**, not topology-free.

## What is deliberately not reused

The standalone project does not reuse the following as method contributions or
runtime dependencies:

- the C-RAG L1/L2/L3 architecture;
- the KL advisor/router or expert-fusion logic;
- `full_fd2` or edge-semantic fusion;
- partition routing and METIS search;
- C-RAG graph traversal;
- the universal C-RAG system runtime;
- GraphRAG/HippoRAG comparison machinery;
- online BM25, SPLADE, ColBERT, or FAISS retrieval runtimes;
- CRAG checkpoints;
- old C-RAG paper numbers as final standalone evidence;
- question nodes or question-to-gold graph edges;
- gold-derived co-support bridge edges; or
- typed-relation modeling.

A small snapshot of six CRAG source files and three historical result JSON files
exists under `legacy/crag_snapshot/` for provenance only. The standalone Python
package does not import or execute this snapshot. Data schemas, input
construction, models, losses, training, fixed graph summaries, statistics, and
analysis were implemented and validated in the standalone repository.

## Correct scientific framing

This is not simply a comparison between an MLP that ignores graphs and a GNN
that uses graphs. The controlled comparison separates four sources of value:

\[
\text{embedding-only learning}
\]

versus

\[
\text{retrieval-prior conditioning}
\]

versus

\[
\text{fixed graph computation + MLP}
\]

versus

\[
\text{learned graph message passing}.
\]

That separation is what permits the study to ask whether retrieval gains come
from the initial retriever, from graph topology itself, or specifically from
learned neighborhood aggregation.

## Related provenance and contract documents

- [CRAG extraction audit](CRAG_EXTRACTION_AUDIT.md)
- [Standalone data contract](../data/README.md)
- [Complete-data loader](../src/mp_retrieval/complete_data.py)
- [Six-dataset results](SIX_DATASET_RESULTS.md)
- [QLS-MLP confirmation protocol](SA_MLP_CONFIRMATION_PROTOCOL.md)
- [QLS-MLP confirmation results](SA_MLP_CONFIRMATION_RESULTS.md)
- [QLS feature leakage audit](SA_FEATURE_LEAKAGE_AUDIT.md)
- [Legacy candidate compatibility audit](LEGACY_CANDIDATE_COMPATIBILITY.md)
- [RRF and unseen-embedding evaluation plan](RRF_AND_ONLINE_EVALUATION_FUTURE_WORK.md)
- [Paper-readiness and real-world audit](PAPER_READINESS_AND_REAL_WORLD_FUTURE_WORK.md)
- [Terminology, overlap, and submission positioning](TERMINOLOGY_AND_POSITIONING.md)

The original CRAG repository is a read-only provenance source. It must not be
edited as part of this standalone project.
