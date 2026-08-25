#!/usr/bin/env python
"""Export a read-only CRAG substrate to the neutral global retrieval format."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.data import GraphRetrievalData, QuerySplit  # noqa: E402
from mp_retrieval.protocol import sha256_file  # noqa: E402


def _hash_split(query_id: str) -> int:
    bucket = int(hashlib.sha256(query_id.encode("utf-8")).hexdigest()[:8], 16) % 10_000
    if bucket < 7_000:
        return int(QuerySplit.TRAIN)
    if bucket < 8_500:
        return int(QuerySplit.VALIDATION)
    return int(QuerySplit.TEST)


def _node_features(index_dir: Path) -> tuple[torch.Tensor, Path]:
    array_path = index_dir / "nodes.npy"
    if array_path.exists():
        array = np.load(array_path, mmap_mode="r")
        return torch.from_numpy(np.asarray(array, dtype=np.float32).copy()), array_path
    faiss_path = index_dir / "nodes.index"
    if not faiss_path.exists():
        raise FileNotFoundError(f"No nodes.npy or nodes.index under {index_dir}")
    try:
        import faiss
    except ImportError as exc:
        raise ImportError("Install the 'crag' extra to reconstruct a FAISS index") from exc
    index = faiss.read_index(str(faiss_path))
    array = index.reconstruct_n(0, index.ntotal).astype("float32")
    return torch.from_numpy(array), faiss_path


def _query_features(index_dir: Path, query_nodes: list[dict], model_name: str | None) -> tuple[torch.Tensor, str]:
    array_path = index_dir / "queries_all.npy"
    ids_path = index_dir / "query_ids_all.json"
    if array_path.exists() and ids_path.exists():
        metadata = json.loads(ids_path.read_text(encoding="utf-8"))
        ids = metadata["ids"] if isinstance(metadata, dict) else metadata
        id_to_row = {str(query_id): row for row, query_id in enumerate(ids)}
        array = np.load(array_path, mmap_mode="r")
        missing = [query["node_id"] for query in query_nodes if query["node_id"] not in id_to_row]
        if not missing:
            rows = [id_to_row[query["node_id"]] for query in query_nodes]
            return torch.from_numpy(np.asarray(array[rows], dtype=np.float32).copy()), str(array_path)
    if not model_name:
        raise FileNotFoundError(
            "Aligned query arrays were unavailable. Pass --query-model to encode query text."
        )
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Install the 'crag' extra to encode missing query features") from exc
    model = SentenceTransformer(model_name)
    encoded = model.encode(
        [query["content"] for query in query_nodes],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return torch.from_numpy(np.asarray(encoded, dtype=np.float32)), f"sentence-transformers:{model_name}"


def export(args) -> None:
    crag_root = args.crag_root.resolve()
    master = crag_root / "data" / "processed" / f"master_nodes_{args.dataset}.json"
    storage = crag_root / "data" / "ukb_storage" / args.dataset
    index_dir = storage / args.encoder_subdir if args.encoder_subdir else storage
    if not (index_dir / "graph.pt").exists():
        index_dir = storage
    records = json.loads(master.read_text(encoding="utf-8"))
    records = [row for row in records if row.get("metadata", {}).get("source") == args.dataset]
    documents = [row for row in records if row.get("metadata", {}).get("type") != "question"]
    raw_queries = [row for row in records if row.get("metadata", {}).get("type") == "question"]
    node_to_index = {row["node_id"]: idx for idx, row in enumerate(documents)}
    query_nodes: list[dict] = []
    relevance: list[torch.Tensor] = []
    for query in raw_queries:
        positives = sorted({node_to_index[node] for node in query.get("neighbors", []) if node in node_to_index})
        if positives:
            query_nodes.append(query)
            relevance.append(torch.tensor(positives, dtype=torch.long))
    node_features, node_feature_source = _node_features(index_dir)
    if node_features.shape[0] != len(documents):
        raise ValueError("Node feature order/count does not match the document master")
    query_features, query_feature_source = _query_features(index_dir, query_nodes, args.query_model)
    graph_path = index_dir / "graph.pt"
    graph = torch.load(graph_path, map_location="cpu", weights_only=False)
    data = GraphRetrievalData(
        node_features=node_features,
        edge_index=graph.edge_index.cpu().long(),
        query_features=query_features,
        relevance=relevance,
        query_split=torch.tensor([_hash_split(row["node_id"]) for row in query_nodes], dtype=torch.long),
        node_ids=[row["node_id"] for row in documents],
        query_ids=[row["node_id"] for row in query_nodes],
        metadata={
            "status": "development_hash_split",
            "warning": "Replace with canonical/official splits before paper evaluation.",
            "dataset": args.dataset,
            "crag_root_read_only": str(crag_root),
            "master_source": str(master),
            "master_sha256": sha256_file(master),
            "graph_source": str(graph_path),
            "graph_sha256": sha256_file(graph_path),
            "node_feature_source": str(node_feature_source),
            "query_feature_source": query_feature_source,
            "edge_semantics": "untyped CRAG graph; canonical typed-edge export required",
        },
    )
    data.save(args.output)
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "nodes": data.num_nodes,
                "edges": data.num_edges,
                "queries": data.num_queries,
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crag-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder-subdir", default="gte_qwen")
    parser.add_argument("--query-model", default=None)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
