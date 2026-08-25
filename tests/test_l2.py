import torch

from mp_retrieval.data import QuerySplit
from mp_retrieval.l2_data import CandidateQuery, L2CandidateDataset, edge_index_to_csr
from mp_retrieval.l2_features import build_candidate_features, present_only_stats
from mp_retrieval.l2_protocol import comparison_contract
from mp_retrieval.l2_stats import candidate_graph_statistics


def make_query(name, scores, mask, relevant, split):
    return CandidateQuery(
        query_id=name,
        candidate_index=torch.tensor([0, 1, 2], dtype=torch.long),
        expert_scores=torch.tensor(scores, dtype=torch.float32),
        expert_mask=torch.tensor(mask, dtype=torch.bool),
        relevant_local=torch.tensor(relevant, dtype=torch.long),
        relevant_global=torch.tensor(relevant, dtype=torch.long),
        split=int(split),
    )


def test_present_only_statistics_ignore_missing_values():
    query = make_query(
        "q",
        [[1.0, 1000.0], [3.0, 2.0], [5.0, 4.0]],
        [[1, 0], [1, 1], [1, 1]],
        [0],
        QuerySplit.TRAIN,
    )
    mean, std = present_only_stats([query])
    assert mean.tolist() == [3.0, 3.0]
    features = build_candidate_features(query, mean, std)
    assert torch.isfinite(features).all()
    assert features[0, 1] == 0.0


def test_induced_candidate_graph_uses_only_pool_nodes(tmp_path):
    edge_index = torch.tensor([[0, 0, 1, 2, 3], [1, 3, 2, 3, 0]], dtype=torch.long)
    rowptr, col, _ = edge_index_to_csr(edge_index, 4)
    query = make_query(
        "q",
        [[1, 0], [0, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [0],
        QuerySplit.TEST,
    )
    artifact = L2CandidateDataset("toy", 4, ["a", "b"], rowptr, col, [query]).validate()
    local_edges, _ = artifact.induced_subgraph(query)
    assert set(zip(local_edges[0].tolist(), local_edges[1].tolist())) == {(0, 1), (1, 2)}
    path = tmp_path / "l2.pt"
    artifact.save(path)
    assert L2CandidateDataset.load(path).queries[0].candidate_ceiling == 1.0


def test_l2_contract_hashes_shared_inputs_and_stats_keep_types_unavailable():
    query = make_query(
        "q",
        [[1, 0], [0, 1], [1, 1]],
        [[1, 1], [1, 1], [1, 1]],
        [0],
        QuerySplit.TRAIN,
    )
    mean, std = present_only_stats([query])
    features = {query: build_candidate_features(query, mean, std)}
    edges = {query: torch.tensor([[0, 1], [1, 2]], dtype=torch.long)}
    contract = comparison_contract([query], features, edges, seeds=[0, 1])
    assert contract["topology_is_only_extra_gnn_information"]
    assert len(contract["sha256"]["shared_model_inputs"]) == 64
    stats = candidate_graph_statistics(query, features[query], edges[query], None)
    assert stats["edge_types_available"] is False
    assert stats["edge_type_entropy"] is None
