import torch

from mp_retrieval.data import GraphRetrievalData, QuerySplit


def tiny_data() -> GraphRetrievalData:
    return GraphRetrievalData(
        node_features=torch.eye(4),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        query_features=torch.eye(2, 4),
        relevance=[torch.tensor([0, 1]), torch.tensor([3])],
        query_split=torch.tensor([QuerySplit.TRAIN, QuerySplit.TEST]),
        node_ids=["a", "b", "c", "d"],
        query_ids=["q0", "q1"],
    )


def test_round_trip(tmp_path):
    original = tiny_data().validate()
    path = tmp_path / "tiny.pt"
    original.save(path)
    loaded = GraphRetrievalData.load(path)
    assert loaded.num_nodes == 4
    assert loaded.num_queries == 2
    assert loaded.relevance[0].tolist() == [0, 1]
    assert loaded.subset_queries(QuerySplit.TEST).tolist() == [1]


def test_invalid_edge_rejected():
    data = tiny_data()
    data.edge_index[1, -1] = 10
    try:
        data.validate()
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("invalid edge was accepted")

