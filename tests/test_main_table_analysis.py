import pytest

from scripts.analyze_main_table import _conclusion, _metaqa_hops


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


def test_conclusion_classifies_paired_intervals_and_system_costs() -> None:
    def row(dataset: str, low: float, high: float):
        return {
            "dataset": dataset,
            "mlp_minus_gnn": {"recall@5": {"ci95_low": low, "ci95_high": high}},
        }

    systems = [
        {
            "gnn_over_mlp_latency": 4.0,
            "gnn_minus_mlp_peak_gpu_memory_mb_incremental": 100.0,
            "gnn_over_mlp_parameters": 1.02,
        },
        {
            "gnn_over_mlp_latency": 9.0,
            "gnn_minus_mlp_peak_gpu_memory_mb_incremental": 300.0,
            "gnn_over_mlp_parameters": 1.04,
        },
    ]

    result = _conclusion(
        [row("mlp", 0.01, 0.03), row("gnn", -0.03, -0.01), row("tie", -0.01, 0.01)],
        systems,
    )

    assert result["mlp_wins"] == ["mlp"]
    assert result["gnn_wins"] == ["gnn"]
    assert result["neutral"] == ["tie"]
    assert result["gnn_over_mlp_latency_min"] == 4.0
    assert result["gnn_minus_mlp_incremental_memory_mib_max"] == 300.0
