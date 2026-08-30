"""Rank-only controls for frozen Dense and SPLADE candidate arrays."""

from __future__ import annotations

import hashlib
import json
import pickle
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class FrozenRankContract:
    """Lightweight view of a complete ranking contract without graph loading."""

    root: Path
    dataset: str
    dense: np.ndarray
    splade: np.ndarray
    query_ids: list[str]
    golds: list[tuple[int, ...]]
    split_indices: dict[str, np.ndarray]
    source_sha256: dict[str, str]
    identity_source: str

    @property
    def query_count(self) -> int:
        return int(self.dense.shape[0])

    @property
    def source_width(self) -> int:
        return int(self.dense.shape[1])


def sha256_file(path: str | Path, *, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _node_mapping(
    root: Path,
    *,
    dataset: str,
) -> tuple[dict[str, int] | None, Path | None, str]:
    path = root / "node_ids.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            if len(set(map(str, payload))) != len(payload):
                raise ValueError("node_ids.json contains duplicate IDs")
            mapping = {str(node_id): index for index, node_id in enumerate(payload)}
        elif isinstance(payload, dict):
            mapping = {str(node_id): int(index) for node_id, index in payload.items()}
            if len(set(mapping.values())) != len(mapping):
                raise ValueError("node_ids.json maps multiple IDs to the same row")
        else:
            raise ValueError("node_ids.json must be a row-ordered list or ID-to-row mapping")
        return mapping, path, "explicit_node_ids_json"

    if dataset != "metaqa":
        return None, None, "numeric_suffix"

    identity_path = root.parent / "splade_doc_embs.pkl"
    if not identity_path.is_file():
        return None, None, "numeric_suffix"
    with identity_path.open("rb") as stream:
        # This is a frozen, trusted local CRAG artifact, never an uploaded pickle.
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or not isinstance(payload.get("id_to_idx"), dict):
        raise TypeError("MetaQA SPLADE identity source has no id_to_idx mapping")
    mapping = {str(node_id): int(index) for node_id, index in payload["id_to_idx"].items()}
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("MetaQA SPLADE identity mapping is not one-to-one")
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("MetaQA SPLADE identity mapping is not onto contiguous node rows")
    return mapping, identity_path, "frozen_splade_id_to_idx"


def _gold_index(node_id: Any, node_mapping: dict[str, int] | None) -> int:
    text = str(node_id)
    if node_mapping is not None:
        try:
            return int(node_mapping[text])
        except KeyError as exc:
            raise ValueError(f"Gold node ID is absent from node_ids.json: {text!r}") from exc
    try:
        return int(text.rsplit("_", 1)[-1])
    except ValueError as exc:
        raise ValueError(f"Gold node ID does not end in a numeric row index: {text!r}") from exc


def load_frozen_rank_contract(
    root: str | Path,
    *,
    dataset: str | None = None,
    hash_sources: bool = True,
) -> FrozenRankContract:
    """Load only rankings, splits, and gold IDs from a complete frozen source."""

    root = Path(root)
    required = ("dense_top200_all.npy", "splade_top200_all.npy", "query_ids_all.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete frozen ranking source {root}: missing {missing}")

    dense = np.load(root / required[0], mmap_mode="r")
    splade = np.load(root / required[1], mmap_mode="r")
    manifest = json.loads((root / required[2]).read_text(encoding="utf-8"))
    query_ids = [str(value) for value in manifest["ids"]]
    raw_golds = manifest["golds"]
    if dense.ndim != 2 or dense.shape != splade.shape:
        raise ValueError("Dense and SPLADE rankings must have the same two-dimensional shape")
    if dense.shape[0] != len(query_ids) or len(raw_golds) != len(query_ids):
        raise ValueError("Rankings, query IDs, and gold rows are misaligned")
    if not np.issubdtype(dense.dtype, np.integer) or not np.issubdtype(splade.dtype, np.integer):
        raise ValueError("Ranked candidate arrays must contain integer global node IDs")

    resolved_dataset = str(dataset or manifest.get("dataset", root.parent.name))
    mapping, identity_path, identity_source = _node_mapping(root, dataset=resolved_dataset)
    if identity_source == "frozen_splade_id_to_idx":
        maximum_ranked_row = max(int(np.max(dense)), int(np.max(splade)))
        if mapping is None or maximum_ranked_row >= len(mapping):
            raise ValueError("MetaQA identity mapping does not cover every ranked node row")
    golds = [
        tuple(sorted({_gold_index(node_id, mapping) for node_id in row}))
        for row in raw_golds
    ]
    if any(not row for row in golds):
        raise ValueError("Every query must have at least one gold node")

    raw_splits = manifest["split_indices"]
    aliases = {"train": "train", "val": "validation", "validation": "validation", "test": "test"}
    split_lists: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    assigned: set[int] = set()
    for source_name, canonical_name in aliases.items():
        for raw_index in raw_splits.get(source_name, []):
            index = int(raw_index)
            if index in assigned:
                raise ValueError(f"Query {index} occurs in multiple canonical splits")
            if index < 0 or index >= len(query_ids):
                raise ValueError(f"Split contains invalid query index {index}")
            assigned.add(index)
            split_lists[canonical_name].append(index)
    if assigned != set(range(len(query_ids))):
        raise ValueError("Canonical splits do not cover every query exactly once")
    split_indices = {
        name: np.asarray(indices, dtype=np.int64) for name, indices in split_lists.items()
    }

    source_sha256 = {}
    if hash_sources:
        for name in required:
            source_sha256[name] = sha256_file(root / name)
        if identity_path is not None:
            source_sha256[identity_path.name] = sha256_file(identity_path)

    return FrozenRankContract(
        root=root,
        dataset=resolved_dataset,
        dense=dense,
        splade=splade,
        query_ids=query_ids,
        golds=golds,
        split_indices=split_indices,
        source_sha256=source_sha256,
        identity_source=identity_source,
    )


def rrf_rankings(
    dense: np.ndarray,
    splade: np.ndarray,
    *,
    dense_weights: Iterable[float],
    constant: int = 60,
    top_k: int = 20,
) -> dict[float, np.ndarray]:
    """Return deterministic weighted-RRF rankings for a batch.

    Candidate IDs are sorted once. Equal score ties inherit ascending global
    node ID order. Each source row must itself contain unique IDs.
    """

    dense = np.asarray(dense, dtype=np.int64)
    splade = np.asarray(splade, dtype=np.int64)
    if dense.ndim != 2 or dense.shape != splade.shape:
        raise ValueError("Dense and SPLADE batches must have the same two-dimensional shape")
    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    if not 0 < top_k <= dense.shape[1] + splade.shape[1]:
        raise ValueError("top_k must fit within the concatenated candidate width")

    rows, width = dense.shape
    ids = np.concatenate((dense, splade), axis=1)
    ranks = 1.0 / (constant + np.arange(1, width + 1, dtype=np.float64))
    dense_component = np.broadcast_to(
        np.concatenate((ranks, np.zeros(width, dtype=np.float64))), ids.shape
    )
    splade_component = np.broadcast_to(
        np.concatenate((np.zeros(width, dtype=np.float64), ranks)), ids.shape
    )
    source = np.broadcast_to(
        np.concatenate((np.zeros(width, dtype=np.int8), np.ones(width, dtype=np.int8))),
        ids.shape,
    )

    by_id = np.argsort(ids, axis=1, kind="stable")
    sorted_ids = np.take_along_axis(ids, by_id, axis=1)
    sorted_dense = np.take_along_axis(dense_component, by_id, axis=1).copy()
    sorted_splade = np.take_along_axis(splade_component, by_id, axis=1).copy()
    sorted_source = np.take_along_axis(source, by_id, axis=1)

    same_next = sorted_ids[:, :-1] == sorted_ids[:, 1:]
    same_source = sorted_source[:, :-1] == sorted_source[:, 1:]
    if np.any(same_next & same_source):
        raise ValueError("A source ranking contains duplicate candidate IDs")

    sorted_dense[:, :-1] += np.where(same_next, sorted_dense[:, 1:], 0.0)
    sorted_splade[:, :-1] += np.where(same_next, sorted_splade[:, 1:], 0.0)
    valid = np.ones((rows, width * 2), dtype=bool)
    valid[:, 1:] = ~same_next

    output: dict[float, np.ndarray] = {}
    for raw_weight in dense_weights:
        weight = float(raw_weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("Dense RRF weights must lie in [0, 1]")
        scores = weight * sorted_dense + (1.0 - weight) * sorted_splade
        scores[~valid] = -np.inf
        by_score = np.argsort(-scores, axis=1, kind="stable")[:, :top_k]
        output[weight] = np.take_along_axis(sorted_ids, by_score, axis=1)
    return output


def ranking_metrics(
    ranking: Sequence[int],
    golds: Sequence[int],
    candidate_ids: Sequence[int],
    *,
    ks: Sequence[int] = (1, 5, 20),
) -> dict[str, float | None]:
    """Metrics matching the sealed all-gold and in-pool conditional semantics."""

    gold = set(map(int, golds))
    if not gold:
        raise ValueError("Every query must have at least one gold node")
    candidates = set(map(int, candidate_ids))
    available = gold & candidates
    ordered = list(map(int, ranking))
    first = next((index + 1 for index, node in enumerate(ordered) if node in gold), None)
    row: dict[str, float | None] = {
        "candidate_ceiling": len(available) / len(gold),
        "candidate_available": float(bool(available)),
        "mrr": 0.0 if first is None else 1.0 / first,
        "conditional_mrr": None if not available else (0.0 if first is None else 1.0 / first),
    }
    for k in ks:
        hits = len(gold & set(ordered[:k]))
        row[f"recall@{k}"] = hits / len(gold)
        row[f"full_coverage@{k}"] = float(hits == len(gold))
        row[f"conditional_recall@{k}"] = None if not available else hits / len(available)
        row[f"conditional_hit@{k}"] = None if not available else float(hits > 0)
        row[f"conditional_full_coverage@{k}"] = (
            None if not available else float(hits == len(available))
        )
    return row


def aggregate_metric_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, float | int]:
    output: dict[str, float | int] = {}
    for name, raw_values in arrays.items():
        values = np.asarray(raw_values, dtype=np.float64)
        finite = np.isfinite(values)
        output[name] = float(values[finite].mean()) if finite.any() else float("nan")
        if name.startswith("conditional_"):
            output[f"{name}_queries"] = int(finite.sum())
    return output


def select_dense_weight(
    validation_metrics: Mapping[float, Mapping[str, float | int]],
    *,
    metric: str = "recall@5",
) -> float:
    """Validation-only selection with the frozen equal-weight proximity tie rule."""

    if not validation_metrics:
        raise ValueError("At least one validation candidate is required")
    candidates = []
    for raw_weight, metrics in validation_metrics.items():
        weight = float(raw_weight)
        candidates.append((-float(metrics[metric]), abs(weight - 0.5), weight))
    return float(min(candidates)[2])
