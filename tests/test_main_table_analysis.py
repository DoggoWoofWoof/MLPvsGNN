import pytest

from scripts.analyze_main_table import _metaqa_hops


def _hop_seed(value: float):
    metrics = {
        "recall@1": value,
        "recall@5": value,
        "recall@20": value,
        "mrr": value,
        "full_coverage@20": value,
    }
    return {
        "by_hop": {
            str(hop): {"queries": 10 * hop, "metrics": metrics}
            for hop in (1, 2, 3)
        }
    }


def test_metaqa_hop_analysis_is_paired_within_seed() -> None:
    item = {
        "dataset": "metaqa",
        "selected_gnn": "gcn",
        "models": {
            "mlp": {"seeds": {str(seed): _hop_seed(0.6 + seed / 100) for seed in range(5)}},
            "gnn": {"seeds": {str(seed): _hop_seed(0.5 + seed / 100) for seed in range(5)}},
        },
    }

    rows = _metaqa_hops(item)

    assert [row["queries"] for row in rows] == [10, 20, 30]
    assert rows[0]["mlp_minus_gnn"]["recall@5"]["mean"] == pytest.approx(0.1)
