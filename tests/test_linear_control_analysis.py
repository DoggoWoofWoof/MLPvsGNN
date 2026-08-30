import numpy as np

from scripts.analyze_linear_rank_structure import (
    _hierarchical_paired_ci,
    _holm,
    _mean_std_ci,
    _paired_t_pvalue,
)


def test_seed_summary_and_constant_paired_pvalue_are_explicit():
    summary = _mean_std_ci([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary["n"] == 5
    assert np.isclose(summary["mean"], 0.3)
    assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]
    assert _paired_t_pvalue([0.2] * 5) == 0.0
    assert _paired_t_pvalue([0.0] * 5) == 1.0


def test_holm_is_monotone_and_capped():
    adjusted = _holm({"a": 0.001, "b": 0.01, "c": 0.2})
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"] <= 1.0
    assert np.isclose(adjusted["a"], 0.003)


def test_hierarchical_paired_interval_preserves_strict_positive_effect():
    left = np.ones((5, 20, 2), dtype=np.float32)
    right = np.zeros_like(left)
    low, high = _hierarchical_paired_ci(left, right, replicates=100, seed=7)
    np.testing.assert_allclose(low, np.ones(2))
    np.testing.assert_allclose(high, np.ones(2))
