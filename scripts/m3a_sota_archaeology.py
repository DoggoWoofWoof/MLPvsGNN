"""M3A item 2: what modern graph-retrieval systems actually receive at inference.

Every load-bearing claim here was verified against the current primary artifact --
arXiv version and date recorded, official repository where one exists -- and not
carried from a summary. Where a source does not state something, the record says
NOT STATED rather than guessing, and where an implementation could not be
unambiguously located the system is marked PAPER_ONLY_REFERENCE.

The output is not a union of everything these systems use. It is a classification:
which of their information can be compiled into fixed or query-local features
ahead of ranking, which is intrinsically a message-passing mechanism, and which is
a privilege we cannot obtain under our own retrieval setup. That classification is
what M3B needs; the union would be a shopping list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = ROOT / "docs" / "M3A_SOTA_ARCHAEOLOGY.md"
JSON_PATH = ROOT / "outputs" / "m3a" / "sota_archaeology.json"

VERIFIED_ON = "2026-09-08"

# --------------------------------------------------------------------------
# Feature classification buckets. Only the first two may enter QLS-U.
# --------------------------------------------------------------------------

SHARED_BASE = "SHARED_BASE_FEATURE"
FIXED_SUMMARY = "FIXED_STRUCTURAL_SUMMARY"
MP_ONLY = "MESSAGE_PASSING_ONLY_MECHANISM"
PRIVILEGED = "PRIVILEGED_UNAVAILABLE"
NOT_UNIVERSAL = "DATASET_SPECIFIC_NOT_UNIVERSAL"
TOO_COSTLY = "TOO_EXPENSIVE_OR_REDUNDANT"
EXTERNAL = "EXTERNAL_TRAINING_RESOURCE"

ADMISSIBLE_TO_QLS_U = (SHARED_BASE, FIXED_SUMMARY)

SYSTEMS: tuple[dict[str, Any], ...] = (
    {
        "system": "GraphER",
        "paper_version": "arXiv 2603.24925v3",
        "paper_date": "2026-08-02",
        "verified_from": "arXiv abstract page and full v3 PDF text",
        "repository": None,
        "repository_revision_if_resolved": None,
        "implementation_status": "PAPER_ONLY_REFERENCE",
        "datasets": [
            "Spider1_dev", "Spider1_test", "Bird_dev", "Beaver_dev",
            "HotpotQA", "2WikiMultihopQA", "MuSiQue", "BEIR-NQ",
        ],
        "node_type": "data object (passage / table)",
        "edge_type": "conceptual proximity: two objects share one or more named entities",
        "edge_weight": "A_ij = (unique named entities shared) / (total unique entities in object j)",
        "typed_edges": False,
        "direction_preserved": False,
        "query_embedding": True,
        "node_embedding": True,
        "relation_embedding": False,
        "seed_entity_required": False,
        "seed_entity_source": "n/a -- reranks a base retriever's candidates",
        "oracle_or_inference_safe": "inference_safe",
        "retrieval_score_input": "yes -- the GCS score, which is a function of base retriever scores",
        "semantic_score_input": "yes -- query embedding and object embedding",
        "degree_feature": "implicit via the row-normalized adjacency inside GCS",
        "rf_ief": False,
        "relation_query_compatibility": False,
        "entity_overlap": True,
        "graph_cohesion": "GCS -- Graph Cohesive Smoothing",
        "hop_distance": False,
        "operator": "GATv2",
        "depth": "five GATv2 layers followed by two fully connected layers",
        "hidden_width": "NOT STATED",
        "residual": "NOT STATED",
        "attention": True,
        "pretraining": None,
        "extra_supervision": None,
        "candidate_scope": "reranks the base retriever's candidate list only",
        "candidate_count": 200,
        "graph_scope": "candidate-induced, built at query time",
        "metric": "PR@K = 1 if ALL relevant objects are in the top-K, else 0 (set coverage)",
        "metric_is_recall": False,
        "corpus_protocol": (
            "HotpotQA and 2WikiMultihopQA: 2000 randomly sampled dev queries, with the "
            "retrieval corpus built from the passages corresponding to those sampled "
            "queries. MuSiQue: the full musique_ans_v1.0_dev branch (2417 queries). "
            "Table 5's base retriever is Llama-Embed-Nemotron-8B+BM25."
        ),
        "classification": "MECHANISTIC_EXTERNAL_EVIDENCE",
        "not_a_direct_comparator_because": (
            "sampled 2000-query dev subsets with a corpus induced from those samples, a "
            "200-candidate rerank scope, a much stronger base retriever, and a set-coverage "
            "metric rather than recall. The paper itself notes GraphER 'cannot recover "
            "relevant objects that are absent from the initial retrieval results'."
        ),
    },
    {
        "system": "NuTrea",
        "paper_version": "arXiv 2310.15484 (NeurIPS 2023)",
        "paper_date": "2023-10-24",
        "verified_from": "arXiv PDF full text and official repository README",
        "repository": "https://github.com/mlvlab/NuTrea",
        "repository_revision_if_resolved": "initial code release 2023.12 (no tag stated)",
        "implementation_status": "OFFICIAL_CODE_AVAILABLE",
        "datasets": ["WebQuestionsSP", "ComplexWebQuestions", "MetaQA-1hop", "MetaQA-2hop", "MetaQA-3hop"],
        "node_type": "KG entity",
        "edge_type": "typed KG triple",
        "typed_edges": True,
        "direction_preserved": True,
        "query_embedding": True,
        "node_embedding": "RF-IEF, plus learned layers",
        "relation_embedding": True,
        "seed_entity_required": True,
        "seed_entity_source": (
            "given: 'For each question, at least one topic entity is assigned and a "
            "subgraph within 2-hops from the topic entities is extracted'"
        ),
        "oracle_or_inference_safe": "ORACLE -- topic entities are assigned, not retrieved",
        "retrieval_score_input": False,
        "semantic_score_input": True,
        "degree_feature": "yes -- RF uses raw counts specifically to reflect local degree",
        "rf_ief": True,
        "relation_query_compatibility": True,
        "entity_overlap": False,
        "graph_cohesion": False,
        "hop_distance": "implicit -- 2-hop subgraph extraction",
        "operator": "tree-search message passing: Expansion -> Backup -> Node Ranking",
        "depth": "NuTrea layers; the Backup step probes unreached subtree regions",
        "hidden_width": "NOT STATED in the sections read",
        "residual": "NOT STATED",
        "attention": "NOT STATED",
        "pretraining": None,
        "extra_supervision": "weakly supervised; no ground-truth logical queries",
        "candidate_scope": "2-hop subgraph around the given topic entities",
        "candidate_count": "NOT STATED",
        "graph_scope": "question-specific KG subgraph",
        "metric": "Hit@1 and F1",
        "metric_is_recall": False,
        "reported": {
            "WebQuestionsSP": {"Hit@1": 77.43, "F1": 72.70},
            "ComplexWebQuestions": {"Hit@1": 53.61, "F1": 49.53},
            "MetaQA-1hop": {"Hit@1": 97.40, "F1": 97.62},
            "MetaQA-2hop": {"Hit@1": 99.99, "F1": 99.82},
            "MetaQA-3hop": {"Hit@1": 98.89, "F1": 87.06},
        },
        "corpus_protocol": "standard KGQA subgraph protocol with assigned topic entities",
        "classification": "MECHANISTIC_EXTERNAL_EVIDENCE",
        "not_a_direct_comparator_because": (
            "topic entities are assigned rather than retrieved, and the candidate universe "
            "is a 2-hop subgraph around them. That is a different and stronger information "
            "privilege than our retrieval-derived candidate pool."
        ),
    },
    {
        "system": "ReaRev",
        "paper_version": "EMNLP Findings 2022",
        "paper_date": "2022-12-07",
        "verified_from": "official repository README (training commands and reported values)",
        "repository": "https://github.com/cmavro/ReaRev_KGQA",
        "repository_revision_if_resolved": "no tag stated",
        "implementation_status": "OFFICIAL_CODE_AVAILABLE",
        "datasets": ["WebQSP", "CWQ", "MetaQA-3hop"],
        "node_type": "KG entity",
        "edge_type": "typed KG triple",
        "typed_edges": True,
        "direction_preserved": True,
        "query_embedding": "decomposed into instruction vectors (num_ins 2-3)",
        "node_embedding": "entity_dim 50",
        "relation_embedding": "relation word embeddings (--relation_word_emb True for WebQSP and CWQ, False for MetaQA-3hop)",
        "seed_entity_required": True,
        "seed_entity_source": "assumed provided in the preprocessed data",
        "oracle_or_inference_safe": "ORACLE -- topic entities assumed given",
        "retrieval_score_input": False,
        "semantic_score_input": True,
        "degree_feature": "NOT STATED",
        "rf_ief": False,
        "relation_query_compatibility": True,
        "entity_overlap": False,
        "graph_cohesion": False,
        "hop_distance": False,
        "operator": "query-conditioned GNN with repeated reasoning iterations",
        "depth": "num_gnn 2 (WebQSP) or 3 (CWQ, MetaQA-3); num_iter 3 (WebQSP) or 2 (CWQ, MetaQA-3)",
        "hidden_width": "entity_dim 50",
        "residual": "NOT STATED",
        "attention": "NOT STATED",
        "pretraining": None,
        "extra_supervision": "weakly supervised",
        "candidate_scope": "question-specific KG subgraph",
        "candidate_count": "NOT STATED in the README",
        "graph_scope": "question-specific KG subgraph",
        "metric": "Hit@1 (README reports a single headline number per dataset)",
        "metric_is_recall": False,
        "reported": {"WebQSP": 76.4, "CWQ": 52.9, "MetaQA-3hop": 98.9},
        "released_configurations": {
            "WebQSP": "--entity_dim 50 --num_iter 3 --num_ins 2 --num_gnn 2 --lm sbert --relation_word_emb True",
            "CWQ": "--entity_dim 50 --num_iter 2 --num_ins 3 --num_gnn 3 --lm sbert --relation_word_emb True",
            "MetaQA-3hop": "--entity_dim 50 --num_iter 2 --num_ins 3 --num_gnn 3 --lm lstm --relation_word_emb False",
        },
        "corpus_protocol": "standard KGQA subgraph protocol with given topic entities",
        "classification": "MECHANISTIC_EXTERNAL_EVIDENCE",
        "not_a_direct_comparator_because": "same topic-entity privilege as NuTrea",
    },
    {
        "system": "GNN-RAG",
        "paper_version": "ACL Findings 2025",
        "paper_date": "2025",
        "verified_from": "ACL Anthology page and official repository README",
        "repository": "https://github.com/cmavro/GNN-RAG",
        "repository_revision_if_resolved": "no tag stated",
        "implementation_status": "OFFICIAL_CODE_AVAILABLE",
        "datasets": ["WebQSP", "CWQ"],
        "pipeline_stages": [
            "1. GNN reasons over a dense subgraph and retrieves candidate answer nodes",
            "2. shortest paths from question entities to those candidates are extracted",
            "3. the paths are verbalized and given to an LLM, which produces the answer",
        ],
        "stage_relevant_to_us": "stage 1 only",
        "node_type": "KG entity",
        "edge_type": "typed KG triple",
        "typed_edges": True,
        "direction_preserved": True,
        "seed_entity_required": True,
        "seed_entity_source": "question entities assumed given",
        "oracle_or_inference_safe": "ORACLE",
        "operator": "lightweight GNN assigning node importance weights; the repository ships a `gnn/` folder of KGQA GNNs including ReaRev configurations",
        "reuses_rearev": "stated by the authors; the README excerpt read does not itself name ReaRev, so recorded as REPORTED_NOT_CONFIRMED_FROM_README",
        "metric": "answer F1 and Hit, measured END-TO-END after an LLM reads the retrieved paths",
        "metric_is_recall": False,
        "reported": {
            "note": (
                "the headline numbers are end-to-end QA metrics after LLM generation, e.g. "
                "outperforming LLM-based retrieval approaches by 8.9-15.5 points at answer "
                "F1 on multi-hop/multi-entity questions. No isolated GNN-stage retrieval "
                "metric was found in the sources read."
            )
        },
        "classification": "PIPELINE_REFERENCE",
        "not_a_direct_comparator_because": (
            "its headline metrics are produced after path extraction and LLM answer "
            "generation. Importing them as if they were retriever recall@K would compare "
            "our ranker against a whole QA system."
        ),
    },
    {
        "system": "GFM-RAG",
        "paper_version": "NeurIPS 2025 / ICLR 2026",
        "paper_date": "2025",
        "verified_from": "official repository and OpenReview listing",
        "repository": "https://github.com/RManLuo/gfm-rag",
        "repository_revision_if_resolved": "no tag stated",
        "implementation_status": "OFFICIAL_CODE_AVAILABLE",
        "model_scale": "8M parameter graph foundation model",
        "pretraining": "two-stage training over 60 knowledge graphs, 14M+ triples, 700k documents",
        "classification": "EXTERNAL_SOTA_REFERENCE",
        "confound": "large-scale graph pretraining",
        "do_not_conflate_with": ["G-reasoner", "GFM-Retriever"],
    },
    {
        "system": "G-reasoner",
        "paper_version": "arXiv 2509.24276 (ICLR 2026)",
        "paper_date": "2025-09",
        "verified_from": "arXiv listing and repository description",
        "repository": "https://github.com/RManLuo/gfm-rag",
        "repository_revision_if_resolved": "34M pretrained model released 2026-04",
        "implementation_status": "OFFICIAL_CODE_AVAILABLE",
        "model_scale": "34M parameter graph foundation model",
        "graph_input_format": "QuadGraph, a four-layer abstraction unifying heterogeneous knowledge sources",
        "classification": "EXTERNAL_SOTA_REFERENCE",
        "confound": "large-scale graph pretraining",
        "do_not_conflate_with": ["GFM-RAG", "GFM-Retriever"],
    },
    {
        "system": "GFM-Retriever",
        "paper_version": "arXiv 2603.07179v1",
        "paper_date": "2026-03-07",
        "verified_from": "arXiv HTML v1",
        "full_title": (
            "Retrieving Minimal and Sufficient Reasoning Subgraphs with Graph Foundation "
            "Models for Path-aware GraphRAG"
        ),
        "authors": "Haonan Yuan, Qingyun Sun, Junhua Shi, Mingjun Liu, Jiaqi Yuan, Ziwei Zhang, Xingcheng Fu, Jianxin Li",
        "repository": None,
        "repository_revision_if_resolved": None,
        "implementation_status": "PAPER_ONLY_REFERENCE",
        "implementation_note": "the paper states codes will be made public after formal publication",
        "operator": "query-conditioned GNN",
        "depth": "6 layers",
        "hidden_width": 512,
        "node_embedding": "initialized with the all-mpnet-v2 sentence encoder",
        "method": "label-free Information Bottleneck formulation for subgraph selection",
        "metric": "Document Recall (R@2D, R@5D)",
        "metric_is_recall": True,
        "reported": {
            "HotpotQA": {"R@2D": 83.4, "R@5D": 90.5},
            "2WikiMultiHopQA": {"R@2D": 91.3, "R@5D": 93.8},
            "MuSiQue": {"R@2D": 51.3, "R@5D": 59.8},
        },
        "classification": "EXTERNAL_SOTA_REFERENCE",
        "confound": "graph foundation-model pretraining plus a learned subgraph selector",
        "do_not_conflate_with": ["GFM-RAG", "G-reasoner"],
        "authorship_note": (
            "different authors from GFM-RAG and G-reasoner; the RManLuo/gfm-rag repository "
            "is NOT this paper's implementation"
        ),
    },
)

# --------------------------------------------------------------------------
# GraphER's three-arm ablation, transcribed verbatim from the v3 PDF, Table 5.
# This is the single most decision-relevant external result, so it is recorded
# in full rather than as the two-arm delta that motivated GAT-NO-MP.
# --------------------------------------------------------------------------

GRAPHER_TABLE_5 = {
    "source": "arXiv 2603.24925v3, Table 5",
    "base_retriever": "Llama-Embed-Nemotron-8B+BM25",
    "metric_definition": "PR@K is 1 if all relevant objects are in the top-K search results, else 0",
    "arms": {
        "GraphER-GCS": "Graph Cohesive Smoothing only; the paper states GCS is a linear transformation of the initial retrieval scores, so this arm learns nothing",
        "GraphER-GAT": "five GATv2 layers then two fully connected layers",
        "GraphER-MLP": "the same network with message passing disabled; layers, hidden dimensions and input features unchanged",
    },
    "gcs_update_rule": "p^(t+1) = alpha * s + (1 - alpha) * W * p^(t), W row-normalized adjacency, s the seed retriever scores, final score max(p^(t), s)",
    "gcs_is_learned": False,
    "mlp_receives_gcs_as_input_feature": True,
    "PR@5": {
        "Spider1_dev": {"GCS": 47.6, "GAT": 49.7, "MLP": 45.2},
        "Spider1_test": {"GCS": 29.2, "GAT": 40.9, "MLP": 36.7},
        "Bird_dev": {"GCS": 29.4, "GAT": 36.9, "MLP": 35.2},
        "Beaver_dev": {"GCS": 17.0, "GAT": 15.0, "MLP": 15.0},
        "HotpotQA": {"GCS": 78.8, "GAT": 78.0, "MLP": 78.9},
        "2WikiMultihopQA": {"GCS": 43.8, "GAT": 44.1, "MLP": 42.5},
        "MuSiQue": {"GCS": 25.4, "GAT": 25.6, "MLP": 21.6},
        "BEIR-NQ": {"GCS": 52.3, "GAT": 54.4, "MLP": 50.3},
    },
    "PR@10": {
        "Spider1_dev": {"GCS": 69.0, "GAT": 70.1, "MLP": 67.2},
        "Spider1_test": {"GCS": 50.4, "GAT": 59.1, "MLP": 55.1},
        "Bird_dev": {"GCS": 72.8, "GAT": 77.0, "MLP": 79.0},
        "Beaver_dev": {"GCS": 31.1, "GAT": 29.6, "MLP": 28.2},
        "HotpotQA": {"GCS": 88.7, "GAT": 88.9, "MLP": 87.2},
        "2WikiMultihopQA": {"GCS": 51.1, "GAT": 53.0, "MLP": 51.1},
        "MuSiQue": {"GCS": 36.9, "GAT": 37.4, "MLP": 32.4},
        "BEIR-NQ": {"GCS": 73.4, "GAT": 73.0, "MLP": 71.2},
    },
}

OUR_DATASETS = ("HotpotQA", "2WikiMultihopQA", "MuSiQue")

# --------------------------------------------------------------------------
# The classification. This is the actual deliverable: not a union of SOTA
# features, but a judgement about which of them can be compiled.
# --------------------------------------------------------------------------

FEATURE_CLASSIFICATION: tuple[dict[str, Any], ...] = (
    {
        "feature": "query embedding, candidate embedding",
        "motivated_by": ["GraphER", "GFM-Retriever"],
        "bucket": SHARED_BASE,
        "contributes": "the semantic match every system starts from",
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "already computed and frozen",
        "missing_data_behaviour": "always defined on all six datasets",
    },
    {
        "feature": "base retriever score and rank (Dense, SPLADE), retriever agreement",
        "motivated_by": ["GraphER"],
        "bucket": SHARED_BASE,
        "contributes": "the seed signal GCS propagates; GraphER feeds a retrieval-derived score directly to its GAT",
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "already computed",
        "missing_data_behaviour": "always defined",
    },
    {
        "feature": "graph cohesion score (GCS-style fixed propagation of retriever scores)",
        "motivated_by": ["GraphER"],
        "bucket": FIXED_SUMMARY,
        "contributes": (
            "the largest single graph signal in GraphER's own ablation; a parameter-free "
            "linear propagation that on our three overlapping datasets lands within "
            "0.2-1.9pp of the five-layer GAT at PR@10"
        ),
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "a few sparse mat-vec products per query over the candidate-induced graph",
        "missing_data_behaviour": "degenerates to the raw retriever score when the graph is empty; needs an availability mask",
    },
    {
        "feature": "shared named-entity overlap count and specificity between candidates",
        "motivated_by": ["GraphER"],
        "bucket": FIXED_SUMMARY,
        "contributes": "the edge definition itself, usable as a candidate-level statistic",
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "offline NER, then per-query counting",
        "missing_data_behaviour": "masked where NER is unavailable",
    },
    {
        "feature": "RF-IEF relation signature of a node",
        "motivated_by": ["NuTrea"],
        "bucket": FIXED_SUMMARY,
        "contributes": (
            "characterises an uninformative node by the rarity of its incident relation "
            "types; RF(v,r) is a raw incident-relation count and IEF(r) = log(|V| / (1 + EF(r))). "
            "Both are corpus statistics and neither requires propagation."
        ),
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "one offline pass over the typed graph; |R|-dimensional per node, reducible",
        "missing_data_behaviour": "undefined without typed relations -- masked on passage datasets unless an edge_type proxy is used",
        "requires": "TYPED_GRAPH_CONTRACT",
    },
    {
        "feature": "query-to-relation compatibility, e.g. max/mean cos(q, e_r) over seed->candidate relations",
        "motivated_by": ["NuTrea", "ReaRev"],
        "bucket": FIXED_SUMMARY,
        "contributes": (
            "ReaRev matches decomposed question instructions against relation word "
            "embeddings during propagation. The matching itself is a similarity, not a "
            "propagation, and can be computed per candidate ahead of ranking."
        ),
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "embed the relation vocabulary once; per-query it is a small dot product per incident relation",
        "missing_data_behaviour": "masked where relations are untyped",
        "requires": "TYPED_GRAPH_CONTRACT",
    },
    {
        "feature": "hop distance from an inference-safe seed, distinct supporting seeds, path statistics",
        "motivated_by": ["NuTrea", "GNN-RAG"],
        "bucket": FIXED_SUMMARY,
        "contributes": "the traversal geometry a path-search GNN is implicitly recovering",
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "bounded BFS from the seed set per query",
        "missing_data_behaviour": "masked when a candidate is unreachable; distance needs an explicit unreachable code, not a large sentinel",
    },
    {
        "feature": "fixed (non-learned) neighbour aggregation of frozen embeddings, per hop and per edge family",
        "motivated_by": ["ReaRev", "GFM-Retriever", "GraphER"],
        "bucket": FIXED_SUMMARY,
        "contributes": (
            "the aggregation half of message passing without the learned half. This is the "
            "sharpest available test: fixed aggregation plus a learned interaction, against "
            "learned aggregation at every step."
        ),
        "qls_u_can_consume_without_message_passing": True,
        "gat_receives_identical": True,
        "cost": "one sparse mean per hop per edge family; the dominant new storage cost in M3A",
        "missing_data_behaviour": "zero vector plus mask when a candidate has no neighbours in that family",
    },
    {
        "feature": "learned, query-conditioned, iterated neighbour aggregation",
        "motivated_by": ["ReaRev", "NuTrea", "GraphER", "GFM-Retriever"],
        "bucket": MP_ONLY,
        "contributes": "the mechanism under test; it is the treatment, not a feature",
        "qls_u_can_consume_without_message_passing": False,
        "gat_receives_identical": "this is exactly what only the GAT receives",
        "cost": "n/a",
        "missing_data_behaviour": "n/a",
    },
    {
        "feature": "assigned topic/seed entities and the 2-hop subgraph extracted around them",
        "motivated_by": ["NuTrea", "ReaRev", "GNN-RAG"],
        "bucket": PRIVILEGED,
        "contributes": "a candidate universe anchored on entities the system did not have to find",
        "qls_u_can_consume_without_message_passing": "not admissible to either model",
        "gat_receives_identical": "neither model receives it",
        "cost": "n/a",
        "missing_data_behaviour": "n/a",
        "why_refused": (
            "NuTrea states that for each question at least one topic entity is assigned. "
            "Our seeds come from retrieval. Importing an assigned topic entity would give "
            "both models an oracle and would not be reproducible under our own setup."
        ),
    },
    {
        "feature": "large-scale graph pretraining (8M/34M parameter graph foundation models)",
        "motivated_by": ["GFM-RAG", "G-reasoner", "GFM-Retriever"],
        "bucket": EXTERNAL,
        "contributes": "cross-domain transfer, not a query-time information advantage",
        "qls_u_can_consume_without_message_passing": False,
        "gat_receives_identical": False,
        "cost": "prohibitive and confounding",
        "missing_data_behaviour": "n/a",
        "why_refused": "it changes what the model has learned, not what it is told at inference; it would confound the message-passing increment",
    },
    {
        "feature": "shortest-path verbalization consumed by an LLM",
        "motivated_by": ["GNN-RAG"],
        "bucket": TOO_COSTLY,
        "contributes": "answer generation, downstream of ranking",
        "qls_u_can_consume_without_message_passing": False,
        "gat_receives_identical": False,
        "cost": "an LLM call per query",
        "missing_data_behaviour": "n/a",
        "why_refused": "out of scope: it is a QA stage, and its metrics are not retriever metrics",
    },
    {
        "feature": "table structural proximity (Spider/Bird/Beaver schema links)",
        "motivated_by": ["GraphER"],
        "bucket": NOT_UNIVERSAL,
        "contributes": "nothing on our six datasets",
        "qls_u_can_consume_without_message_passing": False,
        "gat_receives_identical": False,
        "cost": "n/a",
        "missing_data_behaviour": "n/a",
        "why_refused": "no table-retrieval dataset is in our matrix",
    },
)


# --------------------------------------------------------------------------
# The compilation targets. These are NOT a minimal union of SOTA inputs. The
# hypothesis under test is that the useful graph computation can be compiled
# ahead of ranking, so the compiled features ARE the method, and it is expected
# that some of them appear in no SOTA paper's input list. The GNN receives all
# of them as well, so the comparison stays asymmetric in the GNN's favour.
# --------------------------------------------------------------------------

COMPILATION_TARGETS: tuple[dict[str, Any], ...] = (
    {
        "direction": "A",
        "name": "query-local topology",
        "content": (
            "shortest hop distance to any seed, distinct supporting seeds, distinct "
            "supporting paths, supporting seeds at 1/2/3 hops, unique neighbours reached, "
            "seed-normalised support, candidate degree, local clustering, number of "
            "seed-connected components, fraction of seeds connected"
        ),
        "sota_motivation": "the traversal geometry NuTrea's tree search and GNN-RAG's path extraction recover by propagating",
        "sota_provides_explicitly": False,
        "why_ours": (
            "distinct supporting seeds rather than an edge count, a real shortest "
            "distance rather than collapsed buckets, distinct-path statistics rather "
            "than walk counts, and a seed-count normalisation -- the four specific "
            "weaknesses of the historical QLS block"
        ),
        "needs": ["graph", "inference-safe seeds"],
        "masking": "unreachable is a distinct code, never a large sentinel distance",
    },
    {
        "direction": "B",
        "name": "typed-relation compatibility",
        "content": (
            "max and mean query-to-relation similarity over incident relations, top-3 "
            "relation match, relation frequency, RF-IEF, matching seed-to-candidate "
            "relations, two-hop relation chains, direction match, per-hop relation-pattern "
            "scores"
        ),
        "sota_motivation": "NuTrea's RF-IEF node signature and ReaRev's instruction-to-relation matching",
        "sota_provides_explicitly": True,
        "why_ours": (
            "the ideas are theirs; what is ours is computing them as features ahead of "
            "ranking instead of inside a propagation step"
        ),
        "needs": ["TYPED_GRAPH_CONTRACT"],
        "masking": "an availability mask per relation group, because zero must not conflate 'no typed relation exists here' with 'the relation matched poorly'",
    },
    {
        "direction": "C",
        "name": "deterministic neighbourhood embeddings",
        "content": (
            "mu_v^(1) = mean of e_u over u in N(v), and mu^(2), mu^(3); kept separate per "
            "provenance and per relation family; projected 1536 -> 64"
        ),
        "sota_motivation": "the aggregation half of every GNN in the table",
        "sota_provides_explicitly": False,
        "why_ours": (
            "fixed aggregation plus a learned interaction, set against learned aggregation "
            "at every layer. This isolates the learned part of message passing rather than "
            "the fact of neighbourhood access."
        ),
        "needs": ["graph", "frozen embeddings"],
        "masking": "zero vector plus an explicit mask when a candidate has no neighbour in that family",
    },
    {
        "direction": "D",
        "name": "seed-conditioned deterministic aggregation",
        "content": (
            "p_v^(1) = mean of e_s over seeds s that reach v in one hop, and p^(2), p^(3); "
            "ranker input becomes [q, d, p^(1), p^(2), p^(3)]"
        ),
        "sota_motivation": "query conditioning is what ReaRev, NuTrea and GFM-Retriever all use their message passing to achieve",
        "sota_provides_explicitly": False,
        "why_ours": (
            "it makes the aggregation query-dependent without making it learned, which is "
            "the closest deterministic analogue of a query-conditioned GNN"
        ),
        "needs": ["graph", "inference-safe seeds", "frozen embeddings"],
        "masking": "zero vector plus mask when no seed reaches the candidate at that hop",
    },
)

ASYMMETRIC_DESIGN = {
    "principle": (
        "the two models are NOT given identical raw inputs. The compiled features are the "
        "method under test, so QLS-U must have them; and the GNN is given all of them too, "
        "plus the raw typed graph and message passing, so no reviewer can argue it was "
        "handicapped."
    ),
    "qls_u_receives": "F(q, v) -- the compiled features, with availability masks",
    "gnn_receives": "F(q, v) + G -- the same features, plus the graph and learned message passing",
    "delta_definition": "Delta = GNN(F, G) - QLS_U(F)",
    "delta_reads_as": (
        "the effectiveness that remains attributable specifically to learned message "
        "passing AFTER the graph computation has been compiled -- which is the registered "
        "question, and is not the same quantity as the value of graph access itself"
    ),
    "delta_sign_is_not_predicted": True,
    "gat_no_mp_control": (
        "the same network with propagation disabled and everything else unchanged, which "
        "is exactly GraphER's own MLP ablation; it separates the architecture from the "
        "mechanism"
    ),
    "second_table_is_separate": (
        "native external SOTA numbers are reported in their own protocols in a separate "
        "reference table, never merged into the controlled comparison"
    ),
}


def grapher_deltas() -> dict[str, Any]:
    """Both contrasts the table supports, not only the one that was quoted."""

    out: dict[str, Any] = {}
    for metric in ("PR@5", "PR@10"):
        rows = GRAPHER_TABLE_5[metric]
        out[metric] = {
            dataset: {
                "gat_minus_mlp": round(values["GAT"] - values["MLP"], 4),
                "gat_minus_gcs": round(values["GAT"] - values["GCS"], 4),
                "mlp_minus_gcs": round(values["MLP"] - values["GCS"], 4),
            }
            for dataset, values in rows.items()
        }
    return out


def record() -> dict[str, Any]:
    admissible = [
        row for row in FEATURE_CLASSIFICATION if row["bucket"] in ADMISSIBLE_TO_QLS_U
    ]
    return {
        "format": "m3a_sota_archaeology_v1",
        "phase": "M3A",
        "item": "2 -- SOTA literature and code archaeology",
        "generated_by": "scripts/m3a_sota_archaeology.py",
        "verified_on": VERIFIED_ON,
        "verification_rule": (
            "every load-bearing claim checked against the current primary paper version "
            "and the official repository where one exists; NOT STATED where a source is "
            "silent; PAPER_ONLY_REFERENCE where no implementation could be located"
        ),
        "decides": "nothing; it classifies information so M3B can be designed",
        "systems": list(SYSTEMS),
        "grapher_ablation": GRAPHER_TABLE_5,
        "grapher_deltas": grapher_deltas(),
        "feature_classification": list(FEATURE_CLASSIFICATION),
        "compilation_targets": list(COMPILATION_TARGETS),
        "asymmetric_design": ASYMMETRIC_DESIGN,
        "contract_is_a_union": False,
        "contract_is_a_compilation_target_list": True,
        "admissible_to_qls_u": [row["feature"] for row in admissible],
        "buckets": {
            bucket: [row["feature"] for row in FEATURE_CLASSIFICATION if row["bucket"] == bucket]
            for bucket in (SHARED_BASE, FIXED_SUMMARY, MP_ONLY, PRIVILEGED, NOT_UNIVERSAL, TOO_COSTLY, EXTERNAL)
        },
    }


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def render(data: dict[str, Any]) -> str:
    deltas = data["grapher_deltas"]
    table5 = data["grapher_ablation"]

    ablation_rows = []
    for dataset in OUR_DATASETS:
        pr10 = table5["PR@10"][dataset]
        d = deltas["PR@10"][dataset]
        ablation_rows.append([
            dataset,
            f"{pr10['GCS']:.1f}",
            f"{pr10['GAT']:.1f}",
            f"{pr10['MLP']:.1f}",
            f"{d['gat_minus_mlp']:+.1f}",
            f"**{d['gat_minus_gcs']:+.1f}**",
        ])

    pr5_rows = []
    for dataset in OUR_DATASETS:
        pr5 = table5["PR@5"][dataset]
        d = deltas["PR@5"][dataset]
        pr5_rows.append([
            dataset,
            f"{pr5['GCS']:.1f}",
            f"{pr5['GAT']:.1f}",
            f"{pr5['MLP']:.1f}",
            f"{d['gat_minus_mlp']:+.1f}",
            f"**{d['gat_minus_gcs']:+.1f}**",
        ])

    system_rows = [
        [
            system["system"],
            system.get("paper_version", "?"),
            system.get("implementation_status", "?"),
            "yes" if system.get("typed_edges") else "no" if "typed_edges" in system else "-",
            str(system.get("seed_entity_required", "-")),
            system.get("oracle_or_inference_safe", "-"),
            system.get("classification", "-"),
        ]
        for system in data["systems"]
    ]

    bucket_rows = [
        [bucket, str(len(features)), "<br>".join(features) or "-"]
        for bucket, features in data["buckets"].items()
    ]

    target_rows = [
        [
            target["direction"],
            target["name"],
            target["sota_motivation"],
            "yes" if target["sota_provides_explicitly"] else "**no -- ours**",
        ]
        for target in data["compilation_targets"]
    ]

    admissible = "\n".join(f"- {feature}" for feature in data["admissible_to_qls_u"])

    gat_over_gcs = ", ".join(
        f"`{deltas['PR@10'][dataset]['gat_minus_gcs']:+.1f}`" for dataset in OUR_DATASETS
    )
    musique = table5["PR@10"]["MuSiQue"]
    musique_over_mlp = deltas["PR@10"]["MuSiQue"]["gat_minus_mlp"]
    musique_honest = musique["GAT"] - max(musique["GCS"], musique["MLP"])

    return f"""# M3A SOTA archaeology

Generated by `{data['generated_by']}`. Verified {data['verified_on']}. This
document decides {data['decides']}.

Verification rule: {data['verification_rule']}.

## The systems

{_table(["system", "version", "code", "typed edges", "needs seed entity", "privilege", "role"], system_rows)}

## GraphER already ran our experiment, and it has three arms

This is the most decision-relevant external result, because GraphER's own
ablation is the causal control we were planning to build: message passing
disabled with *"the number of layers, hidden dimensions, and input features
remain unchanged"*, reducing the GAT to a per-node MLP.

The contrast that has been quoted is GAT minus MLP. The table supports a second
contrast that was not quoted, and it points the other way. `GraphER-GCS` is
**not a learned model**: the paper states that GCS is a linear transformation of
the initial retrieval scores. It is a fixed, parameter-free propagation
`{table5['gcs_update_rule']}`.

PR@10, the metric that was quoted:

{_table(["dataset", "GCS (fixed)", "GAT", "MLP", "GAT-MLP", "GAT-GCS"], ablation_rows)}

PR@5, from the same table:

{_table(["dataset", "GCS (fixed)", "GAT", "MLP", "GAT-MLP", "GAT-GCS"], pr5_rows)}

Three things follow, and all three matter for M3B.

**A parameter-free graph propagation captures nearly all of it.** On the three
datasets that overlap ours, five layers of GATv2 buy {gat_over_gcs} points at
PR@10 over a fixed linear smoothing of retriever scores. At PR@5 on HotpotQA the
GAT is the *worst* of the three arms.

**The quoted `{musique_over_mlp:+.1f}` on MuSiQue is mostly a weak MLP arm.**
GraphER-MLP scores {musique['MLP']:.1f} there while the fixed GCS it receives *as
an input feature* scores {musique['GCS']:.1f}. An MLP that is handed a signal and
then ranks {musique['GCS'] - musique['MLP']:.1f} points below that signal is
underfitting, not demonstrating that message passing is required. The honest
increment against the best non-message-passing arm is `{musique_honest:+.1f}`.

**PR@K is not recall.** The paper defines PR@K as 1 only if *all* relevant
objects are in the top K. That is set coverage -- our `full_coverage@K` -- not
`recall@K`. No GraphER number may be compared to a recall row of ours.

GraphER is therefore recorded as `MECHANISTIC_EXTERNAL_EVIDENCE`, not a direct
comparator: {[s for s in data['systems'] if s['system'] == 'GraphER'][0]['not_a_direct_comparator_because']}

## The KB systems buy their strength partly with an oracle

NuTrea reports Hit@1 of 77.43 on WebQSP, 99.99 on MetaQA-2hop and 98.89 on
MetaQA-3hop, and its RF-IEF embedding is a genuine, importable idea: `RF(v, r)`
counts incident edges of each relation type at a node and
`IEF(r) = log(|V| / (1 + EF(r)))` discounts relation types that occur at many
nodes. Both are corpus statistics. Neither needs a GNN.

But the candidate universe is not comparable to ours. NuTrea states that for each
question *at least one topic entity is assigned* and a 2-hop subgraph is
extracted around it. ReaRev and GNN-RAG assume the same. That is an oracle, and
the feature idea has to be separated from the information privilege: we take
RF-IEF and query-to-relation matching, and we refuse assigned topic entities.

ReaRev's released configurations are also smaller than the framing suggests:
`entity_dim 50`, `num_gnn 2` and `num_iter 3` on WebQSP. The KB state of the art
is not deep or wide; it is *typed and query-conditioned*.

GNN-RAG's headline numbers are end-to-end answer F1 after an LLM reads extracted
shortest paths. Only its first stage is comparable to a ranker, and no isolated
retrieval metric for that stage was found in the sources read.

## The GFM family: three different papers

`GFM-RAG` (8M parameters) and `G-reasoner` (34M, QuadGraph) share the
`RManLuo/gfm-rag` repository. `GFM-Retriever` is a **different paper by different
authors** -- arXiv 2603.07179v1, Haonan Yuan et al., 7 March 2026 -- and that
repository is not its implementation. Its architecture *is* stated in the paper
(6-layer query-conditioned GNN, hidden 512, all-mpnet-v2 initialisation) and its
document recall is R@5D 90.5 / 93.8 / 59.8 on HotpotQA / 2Wiki / MuSiQue, but no
code is released yet, so it is marked `PAPER_ONLY_REFERENCE` and its architecture
is not made load-bearing.

All three carry large-scale graph pretraining, which is a training-resource
confound rather than an inference-time information advantage. They belong in the
external reference table, not in the controlled comparison.

## Classification, not union

{_table(["bucket", "n", "features"], bucket_rows)}

Only `{SHARED_BASE}` and `{FIXED_SUMMARY}` are admissible to QLS-U:

{admissible}

The rule that produced this list is not "SOTA uses it, so add it". It is: *this
system's message passing is trying to discover X; can X be computed ahead of
ranking?* Where the answer is yes, X becomes a feature and both models get it.
Where the answer is no, X stays the treatment and only the GAT gets it.

## This is a compilation target list, not a minimal union

The distinction matters enough to state explicitly, because it changes what the
result would mean. The hypothesis is not that an MLP given a GNN's raw inputs
matches the GNN. It is that **learned message passing may become unnecessary
once the useful graph computation has been compiled into sufficiently expressive
fixed or query-local structural features before ranking**.

Under that hypothesis the compiled features *are* the method. Restricting QLS-U
to inputs a GNN paper explicitly lists would remove the thing being tested. So
some of these targets deliberately appear in no SOTA input list:

{_table(["", "target", "what SOTA uses message passing to obtain", "SOTA states it as an input?"], target_rows)}

The fairness is then restored on the other side, and asymmetrically:

- QLS-U receives `{ASYMMETRIC_DESIGN['qls_u_receives']}`.
- The GNN receives `{ASYMMETRIC_DESIGN['gnn_receives']}`.

The GNN is *strictly advantaged* by construction, so
`{ASYMMETRIC_DESIGN['delta_definition']}` reads as
{ASYMMETRIC_DESIGN['delta_reads_as']}. Its sign is not predicted. A positive
Delta is a real finding about message passing; a Delta near zero is a real
finding about compilation. `GAT-NO-MP` sits alongside as the control:
{ASYMMETRIC_DESIGN['gat_no_mp_control']}.

Native external numbers stay out of that comparison: {ASYMMETRIC_DESIGN['second_table_is_separate']}.

Machine-readable form: `outputs/m3a/sota_archaeology.json`.
"""


def main() -> None:
    data = record()
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    DOC_PATH.write_text(render(data), encoding="utf-8", newline="\n")
    print(f"wrote {JSON_PATH.relative_to(ROOT)}")
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
