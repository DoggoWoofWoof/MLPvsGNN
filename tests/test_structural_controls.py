import hashlib

import numpy as np

from mp_retrieval.structural_controls import (
    candidate_contract_hashes,
    candidate_order_sha256,
    equal_rrf_fusion,
    fixed_structural_scores,
    rank_scores,
    selected_rrf_ranking,
    stable_candidate_union,
)


def test_stable_union_and_selected_rrf_preserve_full_candidate_set():
    dense = np.array([5, 3, 9], dtype=np.int64)
    splade = np.array([3, 4, 9], dtype=np.int64)
    candidates = stable_candidate_union(dense, splade)
    ranking = selected_rrf_ranking(dense, splade, dense_weight=0.5, constant=60)
    assert candidates.tolist() == [5, 3, 9, 4]
    assert ranking.tolist() == [3, 9, 5, 4]


def test_rank_scores_and_equal_rrf_use_global_id_ties():
    candidate_ids = np.array([9, 2, 5], dtype=np.int64)
    assert rank_scores(candidate_ids, np.ones(3)).tolist() == [2, 5, 9]
    fused = equal_rrf_fusion(
        np.array([9, 2, 5], dtype=np.int64),
        np.array([2, 9, 5], dtype=np.int64),
        constant=60,
    )
    assert fused.tolist() == [2, 9, 5]


def test_fixed_structural_score_formulas_are_locked():
    local = np.zeros((2, 10), dtype=np.float32)
    local[0, 0] = 1.0
    local[0, 4:8] = [1.0, 0.5, 0.25, 0.0]
    local[0, 8] = 0.75
    local[0, 9] = 0.5
    local[1, 2] = 1.0
    scores = fixed_structural_scores(local)
    assert scores["distance"].tolist() == [1.0, np.float32(1.0 / 3.0)]
    assert scores["ppr"].tolist() == [0.75, 0.0]
    assert np.isclose(scores["path_connectivity"][0], 0.45)
    assert np.isclose(scores["structural_summary"][0], 4.0 / 7.0)


def test_candidate_order_hash_covers_query_ids_counts_and_ids():
    dense = np.array([[5, 3], [4, 1]], dtype=np.int64)
    splade = np.array([[3, 2], [1, 0]], dtype=np.int64)
    ptr = np.array([0, 3, 6], dtype=np.int64)
    observed = candidate_order_sha256(["q0", "q1"], dense, splade, ptr)
    digest = hashlib.sha256()
    for query_id, candidates in (("q0", [5, 3, 2]), ("q1", [4, 1, 0])):
        values = np.asarray(candidates, dtype=np.int64)
        digest.update(query_id.encode())
        digest.update(len(candidates).to_bytes(4, "little"))
        digest.update(values.tobytes())
    assert observed == digest.hexdigest()

    tensor = hashlib.sha256()
    for query_id, candidates in (("q0", [5, 3, 2]), ("q1", [4, 1, 0])):
        tensor.update(query_id.encode())
        tensor.update(np.asarray(candidates, dtype=np.int64).tobytes())
    hashes = candidate_contract_hashes(["q0", "q1"], dense, splade, ptr)
    assert hashes["candidate_tensor_sha256"] == tensor.hexdigest()
