import numpy as np

from scripts.analyze_sa_mlp_confirmation import (
    _hierarchical_paired_ci,
    _holm,
    _interpretation_gates,
    _mean_std_ci,
)


def test_holm_adjustment_is_monotone_and_capped() -> None:
    adjusted = _holm({"a": 0.01, "b": 0.02, "c": 0.9})

    assert adjusted == {"a": 0.03, "b": 0.04, "c": 0.9}


def test_paired_hierarchical_bootstrap_preserves_constant_effect() -> None:
    right = np.zeros((5, 20, 3), dtype=np.float32)
    left = right + np.asarray([0.1, -0.2, 0.3], dtype=np.float32)

    low, high = _hierarchical_paired_ci(left, right, replicates=100, seed=7)

    assert np.allclose(low, [0.1, -0.2, 0.3])
    assert np.allclose(high, [0.1, -0.2, 0.3])
    assert _mean_std_ci([0.1] * 5)["sample_std"] == 0.0


def _gate_row(
    dataset: str,
    *,
    recovery: float,
    substitution_seed_low: float,
    substitution_query_low: float,
) -> dict:
    plain = 0.10
    sa = 0.20
    return {
        "dataset": dataset,
        "models": {
            "plain_mlp": {"recall@5": {"mean": plain}},
            "seed_only": {"recall@5": {"mean": plain + recovery * (sa - plain)}},
            "sa_mlp": {"recall@5": {"mean": sa}},
        },
        "contrasts": {
            "sa_mlp_minus_seed_only": {
                "recall@5": {
                    "seed_effect": {"mean": 0.04, "ci95_low": 0.02},
                    "paired_hierarchical_query_ci95_low": 0.01,
                    "holm_significant_0.05": True,
                }
            },
            "sa_mlp_minus_seed_aware_gnn": {
                "recall@5": {
                    "seed_effect": {"mean": -0.005, "ci95_low": substitution_seed_low},
                    "paired_hierarchical_query_ci95_low": substitution_query_low,
                }
            },
        },
    }


def test_interpretation_gates_apply_registered_thresholds_conservatively() -> None:
    rows = [
        _gate_row(
            "metaqa", recovery=0.40, substitution_seed_low=-0.009, substitution_query_low=-0.008
        ),
        _gate_row(
            "webqsp", recovery=0.50, substitution_seed_low=-0.009, substitution_query_low=-0.02
        ),
        _gate_row(
            "hotpotqa_clean",
            recovery=0.81,
            substitution_seed_low=-0.009,
            substitution_query_low=-0.009,
        ),
        {"dataset": "2wiki_clean"},
        {"dataset": "musique_clean"},
        {"dataset": "squad_clean"},
    ]

    gates = _interpretation_gates(rows)

    assert gates["graph_summary_signal"]["observed"] == 3
    assert gates["graph_summary_signal"]["supported"] is True
    assert gates["seed_prior_explanation"]["observed"] == 1
    assert gates["seed_prior_explanation"]["supported"] is False
    assert gates["fixed_summary_substitution"]["observed"] == 2
    assert gates["fixed_summary_substitution"]["supported"] is True
    assert gates["universal_mlp_claim_supported"] is False
