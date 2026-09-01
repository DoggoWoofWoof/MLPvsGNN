import pytest

from scripts.analyze_online_systems import _cache_break_even, _ratio


def _condition(total_per_query: float) -> dict:
    return {
        "batch_latency_ms": {
            "fusion_and_seed_ms": {"mean": 4.0},
            "topology_induction_ms": {"mean": 6.0},
            "query_local_summary_ms": {"mean": 2.0},
            "gather_transfer_forward_topk_ms": {"mean": 8.0},
            "total_ms": {"mean": 20.0},
        },
        "total_latency_ms_per_query": {"mean": total_per_query},
    }


def test_online_ratio_is_directionally_explicit() -> None:
    assert _ratio(2.0, 4.0) == 0.5


def test_cache_break_even_repays_after_measured_prefix_is_recovered() -> None:
    # Cacheable prefix is 4+6+2 = 12 ms per batch of 4, so 3 ms per query.
    # Each cached serving saves 5 - 1 = 4 ms, so 0.75 further servings repay it.
    entry = _cache_break_even(_condition(5.0), cached_ms_per_query=1.0, batch_size=4)
    assert entry["cache_build_ms_per_query"] == pytest.approx(3.0)
    assert entry["saving_ms_per_served_query"] == pytest.approx(4.0)
    assert entry["cache_ever_repays"] is True
    assert entry["break_even_additional_servings"] == pytest.approx(0.75)
    assert entry["break_even_total_servings"] == pytest.approx(1.75)


def test_cache_build_cost_is_normalized_per_query_not_per_batch() -> None:
    # batch_latency_ms is a per-batch measurement; the same stage totals must
    # cost four times as much per query at batch 1 as at batch 4.
    at_one = _cache_break_even(_condition(5.0), cached_ms_per_query=1.0, batch_size=1)
    at_four = _cache_break_even(_condition(5.0), cached_ms_per_query=1.0, batch_size=4)
    assert at_one["cache_build_ms_per_query"] == pytest.approx(12.0)
    assert at_four["cache_build_ms_per_query"] == pytest.approx(3.0)


def test_cache_never_repays_when_the_cached_path_is_not_faster() -> None:
    entry = _cache_break_even(_condition(5.0), cached_ms_per_query=6.0, batch_size=1)
    assert entry["saving_ms_per_served_query"] == pytest.approx(-1.0)
    assert entry["cache_ever_repays"] is False
    assert entry["break_even_additional_servings"] is None
    assert entry["break_even_total_servings"] is None


def test_cache_break_even_declares_that_storage_is_excluded() -> None:
    entry = _cache_break_even(_condition(5.0), cached_ms_per_query=1.0, batch_size=1)
    assert entry["definition"] == "compute_only_per_query_cache_excluding_storage"
