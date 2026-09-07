"""The latency measurement M2B's tie-break rests on, and the way it could lie.

If the timed span excluded the semantic branch, S4's two ``1536 -> 64``
projections would cost nothing, the projection control would win every
efficiency comparison it entered, and the phase's Pareto tie-break would be
decided by a measurement error. So the central test here does not inspect the
code: it puts a deliberately slow semantic head into a real model and requires
the reported total to grow by roughly what that head costs. A benchmark that
started from an already-computed semantic tensor could not pass it.

The same treatment is applied to the other two things inside the span -- the
raw-embedding gather and the frozen structural block fetch -- and to the claim
that the measurement is uncached, which is tested by counting how many queries
the store is asked about.
"""

from __future__ import annotations

import pathlib
import sys
import time

import pytest
import torch

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from mp_retrieval.m1a_screen import M1AScorer  # noqa: E402
from mp_retrieval.m2b_inference_timing import (  # noqa: E402
    measure_uncached_inference,
    percentiles,
)
from mp_retrieval.m2b_semantic_control import ProjectionSemanticHead  # noqa: E402
from mp_retrieval.qls_v2_semantic import SemanticHead  # noqa: E402

DIM = 32
PRECOMPUTED_WIDTH = 9
CANDIDATES = 7
QUERIES = 4
DELAY_SECONDS = 0.004


class _Query:
    """The two attributes the timing path reads off a widened query."""

    def __init__(self, query_index: int, candidates: int) -> None:
        self.query_index = query_index
        self.candidate_index = torch.arange(candidates, dtype=torch.long)


class _Store:
    """A structural block source that counts what it was asked for."""

    def __init__(self, width: int) -> None:
        self.width = width
        self.calls: list[int] = []

    def batch_features(self, queries, *, include_static, include_local, device):
        assert include_static and include_local
        self.calls.append(len(queries))
        rows = sum(int(query.candidate_index.numel()) for query in queries)
        return torch.zeros((rows, self.width), device=device)


class _SlowHead(torch.nn.Module):
    """A semantic head whose only distinguishing property is that it is slow."""

    def __init__(self, inner: torch.nn.Module, delay: float = DELAY_SECONDS) -> None:
        super().__init__()
        self.inner = inner
        self.rung = inner.rung
        self.feature_names = inner.feature_names
        self.delay = delay

    def forward(self, query, candidates):
        time.sleep(self.delay)
        return self.inner(query, candidates)


def _model(head: torch.nn.Module | None = None, rung: str = "S3") -> M1AScorer:
    return M1AScorer(
        precomputed_width=PRECOMPUTED_WIDTH,
        semantic_rung=rung,
        embedding_dim=DIM,
        dropout=0.2,
        temperature=0.07,
        semantic_head=head,
    )


@pytest.fixture
def environment():
    torch.manual_seed(0)
    queries = [_Query(index, CANDIDATES) for index in range(QUERIES)]
    return {
        "queries": queries,
        "node_embeddings": torch.randn(CANDIDATES, DIM),
        "query_embeddings": torch.randn(QUERIES, DIM),
        "store": _Store(PRECOMPUTED_WIDTH),
        "device": torch.device("cpu"),
    }


def _measure(model, environment, **overrides) -> dict:
    return measure_uncached_inference(
        model=model, **environment, **{"repeats": 2, "warmup": 1, **overrides}
    )


# --------------------------------------------------------------------------
# The semantic branch is inside the clock
# --------------------------------------------------------------------------


def test_a_slow_semantic_head_shows_up_in_the_total(environment) -> None:
    """The test that stops S4's projections being measured as free."""

    fast = _measure(_model(), environment)
    slow = _measure(_model(_SlowHead(SemanticHead(rung="S3", dim=DIM))), environment)

    added_ms = DELAY_SECONDS * 1000
    growth = slow["total_model_ms"]["p50"] - fast["total_model_ms"]["p50"]
    assert growth > added_ms * 0.5, (
        f"a semantic head {added_ms} ms slower moved the reported total by only "
        f"{growth:.3f} ms, so the semantic branch is not inside the timed span"
    )


def test_a_slow_semantic_head_shows_up_in_the_semantic_attribution(environment) -> None:
    fast = _measure(_model(), environment)
    slow = _measure(_model(_SlowHead(SemanticHead(rung="S3", dim=DIM))), environment)
    assert (
        slow["semantic_ms"]["p50"] - fast["semantic_ms"]["p50"] > DELAY_SECONDS * 1000 * 0.5
    )


def test_parameter_count_does_not_predict_semantic_latency(environment) -> None:
    """Deliberately not an ordering assertion, because the ordering is not fixed.

    S4 holds 196,608 semantic parameters to S2's zero, and it is tempting to
    write a test asserting S4 is therefore slower. It is not reliably slower.
    S4 spends its time in two BLAS matmuls; S2 spends its in
    ``within_query_percentile``'s ``argsort``/``unique``/``scatter_add_`` chain
    and a full-width elementwise mean, which are many small kernels rather than
    one large one. Measured at the real 1536 width over 200 candidates, S4's
    semantic branch came out CHEAPER than both S2's and S3's.

    That is exactly why M2B's fit multiplier has to be measured rather than
    argued from parameter counts, and why this suite asserts only that each
    rung is measured -- the direction is a finding, not an invariant.
    """

    measured = {
        rung: _measure(_model(head, rung=rung), environment, repeats=4)
        for rung, head in (
            ("S2", SemanticHead(rung="S2", dim=DIM)),
            ("S3", SemanticHead(rung="S3", dim=DIM)),
            ("S4", ProjectionSemanticHead(dim=DIM)),
        )
    }
    for rung, result in measured.items():
        assert result["semantic_ms"]["mean"] > 0, rung
        assert result["total_model_ms"]["p95"] >= result["total_model_ms"]["p50"], rung


def test_the_structural_block_fetch_is_inside_the_span(environment) -> None:
    """The frozen precomputed block is part of what a served query pays."""

    class _SlowStore(_Store):
        def batch_features(self, queries, **kwargs):
            time.sleep(DELAY_SECONDS)
            return super().batch_features(queries, **kwargs)

    fast = _measure(_model(), environment)
    slow = _measure(_model(), {**environment, "store": _SlowStore(PRECOMPUTED_WIDTH)})
    assert (
        slow["total_model_ms"]["p50"] - fast["total_model_ms"]["p50"]
        > DELAY_SECONDS * 1000 * 0.5
    )


def test_the_span_is_recorded_as_starting_at_raw_embeddings(environment) -> None:
    result = _measure(_model(), environment)
    assert result["measured_span"].startswith("raw query and candidate embeddings")
    assert "1536 -> 64" in result["why_the_span_starts_there"]


# --------------------------------------------------------------------------
# Uncached means one query at a time
# --------------------------------------------------------------------------


def test_every_measurement_asks_the_store_about_exactly_one_query(environment) -> None:
    _measure(_model(), environment)
    assert environment["store"].calls, "the store was never consulted"
    assert set(environment["store"].calls) == {1}, (
        "a measurement batched queries together, which would amortise exactly the "
        "per-query cost the p95 is supposed to capture"
    )


def test_the_sample_size_is_repeats_times_queries(environment) -> None:
    result = _measure(_model(), environment, repeats=3, warmup=1)
    assert result["total_model_ms"]["n"] == 3 * QUERIES
    assert result["queries"] == QUERIES
    assert result["repeats"] == 3


def test_warmup_queries_are_not_counted(environment) -> None:
    result = _measure(_model(), environment, repeats=1, warmup=2)
    assert result["warmup_queries"] == 2
    assert result["total_model_ms"]["n"] == QUERIES


def test_warmup_larger_than_the_query_set_is_reported_honestly(environment) -> None:
    result = _measure(_model(), environment, repeats=1, warmup=99)
    assert result["warmup_queries"] == QUERIES


# --------------------------------------------------------------------------
# The attributions are labelled as attributions
# --------------------------------------------------------------------------


def test_the_attributions_do_not_claim_to_partition_the_total(environment) -> None:
    result = _measure(_model(), environment)
    assert "do not sum to" in result["attribution_note"]
    assert "total_model_ms.p95" in result["attribution_note"]


def test_both_attributions_are_measured(environment) -> None:
    result = _measure(_model(), environment)
    assert result["semantic_ms"]["n"] == QUERIES
    assert result["scorer_ms"]["n"] == QUERIES
    assert result["semantic_ms"]["p50"] > 0
    assert result["scorer_ms"]["p50"] > 0


def test_the_hooks_are_removed_afterwards(environment) -> None:
    """A left-behind hook would slow every later fit in the same container."""

    model = _model()
    _measure(model, environment)
    assert not model.semantic_head._forward_hooks
    assert not model.semantic_head._forward_pre_hooks
    assert not model.scorer._forward_hooks
    assert not model.scorer._forward_pre_hooks


# --------------------------------------------------------------------------
# Housekeeping the caller depends on
# --------------------------------------------------------------------------


def test_the_models_training_mode_is_restored(environment) -> None:
    model = _model()
    model.train()
    _measure(model, environment)
    assert model.training is True

    model.eval()
    _measure(model, environment)
    assert model.training is False


def test_measuring_with_no_queries_is_refused(environment) -> None:
    with pytest.raises(ValueError, match="no queries"):
        _measure(_model(), {**environment, "queries": []})


def test_peak_memory_is_reported(environment) -> None:
    result = _measure(_model(), environment)
    assert result["peak_inference_gpu_memory_mb"] == 0.0  # cpu
    assert result["device"] == "cpu"


# --------------------------------------------------------------------------
# The percentile helper
# --------------------------------------------------------------------------


def test_percentiles_report_observed_values_not_interpolations() -> None:
    values = [float(index) for index in range(1, 101)]
    result = percentiles(values)
    assert result["p50"] in values and result["p95"] in values and result["p99"] in values
    assert result["p50"] == 50.0
    assert result["p95"] == 95.0
    assert result["p99"] == 99.0
    assert result["max"] == 100.0
    assert result["n"] == 100


def test_percentiles_are_order_independent() -> None:
    import random

    values = [float(index) for index in range(1, 51)]
    shuffled = values[:]
    random.Random(0).shuffle(shuffled)
    assert percentiles(shuffled) == percentiles(values)


def test_percentiles_of_a_single_value() -> None:
    result = percentiles([2.5])
    assert result["p50"] == result["p95"] == result["p99"] == result["max"] == 2.5
    assert result["n"] == 1


def test_percentiles_of_nothing_are_zero_not_an_error() -> None:
    assert percentiles([])["n"] == 0
