import numpy as np

from scripts.analyze_sa_mlp_confirmation import (
    _hierarchical_paired_ci,
    _holm,
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
