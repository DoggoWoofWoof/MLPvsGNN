"""Uncached inference latency for the whole model, semantic branch included.

The measurement M2B's Pareto tie-break rests on is easy to get wrong in a way
that would silently decide the phase. If the benchmark starts from an
already-computed semantic tensor and times only the scorer, then S2, S3 and S4
are being compared on a ``Linear(W, 32) -> GELU -> Linear(32, 1)`` whose widths
differ by a factor of twenty -- while the thing that actually separates them,
S4's two ``1536 -> 64`` projections, is never charged for at all. A projection
control that costs nothing would win every tie-break it entered.

So the span timed here is the whole path a served query walks:

    raw query and candidate embeddings
      -> the rung's semantic computation
      -> the frozen precomputed structural block
      -> the scorer
      -> a score per candidate

One query at a time, with nothing carried between queries, because that is what
"uncached" means for a retrieval system: the p95 the paper reports is the
latency of a query nobody has asked before.

An early measurement with this harness already contradicts the intuition the
phase started from. At the real 1536 width over 200 candidates on CPU, S4's
semantic branch is CHEAPER than S2's and S3's -- two BLAS matmuls against
``within_query_percentile``'s ``argsort``/``unique``/``scatter_add_`` chain plus
a full-width elementwise mean. 196,608 parameters do not imply a slower query.
Nothing here assumes an ordering between the rungs; the ordering is a result the
smoke measures, which is precisely why M2B's fit multiplier may not be guessed.

``semantic_ms`` and ``scorer_ms`` come from a second pass with forward hooks
attached, and are attributions rather than a partition of the total: the hooks
and the synchronisations they force cost time the clean pass does not pay.
Their sum will therefore not equal ``total_model_ms`` and is not meant to. The
number that decides anything is the clean total's p95.
"""

from __future__ import annotations

import time
from typing import Any

import torch

#: Queries run before the clock starts, so CUDA kernel selection and any lazy
#: allocation are not charged to the first measured query.
DEFAULT_WARMUP_QUERIES = 8

#: How many times the whole query set is walked. More passes tighten the tail
#: estimate; the reported n is repeats * queries and travels with the result.
DEFAULT_REPEATS = 3


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentiles(values: list[float]) -> dict[str, float]:
    """p50/p95/p99 by nearest-rank on the sorted sample, plus mean and max.

    Nearest-rank rather than interpolation so every reported percentile is a
    latency that was actually observed.
    """

    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0, "n": 0}
    ordered = sorted(values)
    count = len(ordered)

    def at(fraction: float) -> float:
        rank = max(1, min(count, int(-(-fraction * count // 1))))
        return ordered[rank - 1]

    return {
        "p50": at(0.50),
        "p95": at(0.95),
        "p99": at(0.99),
        "mean": sum(ordered) / count,
        "max": ordered[-1],
        "n": count,
    }


def _one_query_total_ms(
    model: torch.nn.Module,
    query: Any,
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    store: Any,
    device: torch.device,
) -> float:
    """The full stack for one query, timed end to end.

    The index gather, the structural block fetch, the semantic computation, the
    concatenation and the scorer are all inside the clock, because all of them
    happen between a query arriving and a score leaving.
    """

    _synchronize(device)
    started = time.perf_counter()
    candidate_index = query.candidate_index.to(device)
    nodes = node_embeddings[candidate_index]
    queries = query_embeddings[
        torch.tensor([query.query_index], dtype=torch.long, device=device)
    ]
    structural = store.batch_features(
        [query], include_static=True, include_local=True, device=device
    )
    batch_index = torch.zeros(nodes.shape[0], dtype=torch.long, device=device)
    model.forward_explicit(nodes, queries, batch_index, structural)
    _synchronize(device)
    return (time.perf_counter() - started) * 1000.0


class _SpanRecorder:
    """Wall time inside one submodule's forward, via pre- and post-hooks."""

    def __init__(self, module: torch.nn.Module, device: torch.device) -> None:
        self.device = device
        self.spans: list[float] = []
        self._started: float | None = None
        self._handles = [
            module.register_forward_pre_hook(self._enter),
            module.register_forward_hook(self._exit),
        ]

    def _enter(self, _module, _inputs) -> None:
        _synchronize(self.device)
        self._started = time.perf_counter()

    def _exit(self, _module, _inputs, _output) -> None:
        _synchronize(self.device)
        if self._started is not None:
            self.spans.append((time.perf_counter() - self._started) * 1000.0)
            self._started = None

    def total_ms(self) -> float:
        """This query's time in the submodule; forward_explicit may call it more than once."""

        total = sum(self.spans)
        self.spans.clear()
        return total

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()


def measure_uncached_inference(
    *,
    model: torch.nn.Module,
    queries: list[Any],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    store: Any,
    device: torch.device,
    repeats: int = DEFAULT_REPEATS,
    warmup: int = DEFAULT_WARMUP_QUERIES,
) -> dict[str, Any]:
    """Latency percentiles for the whole model, and the peak memory it needed.

    Returns the clean totals under ``total_model_ms``, the hooked attributions
    under ``semantic_ms`` and ``scorer_ms``, and the peak allocation observed
    while the measurement ran.
    """

    if not queries:
        raise ValueError("cannot measure inference latency with no queries")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for query in queries[: max(0, warmup)]:
                _one_query_total_ms(
                    model, query, node_embeddings, query_embeddings, store, device
                )

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            totals: list[float] = []
            for _pass in range(repeats):
                for query in queries:
                    totals.append(
                        _one_query_total_ms(
                            model, query, node_embeddings, query_embeddings, store, device
                        )
                    )

            peak_gpu = (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            )

            # Attribution pass. Separate because the hooks' own synchronisations
            # inflate what they measure and would contaminate the clean total.
            semantic = _SpanRecorder(model.semantic_head, device)
            scorer = _SpanRecorder(model.scorer, device)
            semantic_ms: list[float] = []
            scorer_ms: list[float] = []
            try:
                for query in queries:
                    _one_query_total_ms(
                        model, query, node_embeddings, query_embeddings, store, device
                    )
                    semantic_ms.append(semantic.total_ms())
                    scorer_ms.append(scorer.total_ms())
            finally:
                semantic.remove()
                scorer.remove()
    finally:
        model.train(was_training)

    return {
        "measured_span": (
            "raw query and candidate embeddings -> semantic rung -> frozen precomputed "
            "structural block -> scorer -> score"
        ),
        "why_the_span_starts_there": (
            "S4's two 1536 -> 64 projections are the expensive thing under test; a "
            "benchmark starting from an already-computed semantic tensor would charge "
            "S4 nothing for them"
        ),
        "uncached": "one query per measurement, nothing reused between queries",
        "queries": len(queries),
        "repeats": repeats,
        "warmup_queries": min(max(0, warmup), len(queries)),
        "total_model_ms": percentiles(totals),
        "semantic_ms": percentiles(semantic_ms),
        "scorer_ms": percentiles(scorer_ms),
        "attribution_note": (
            "semantic_ms and scorer_ms come from a hooked pass whose synchronisations "
            "cost time the clean pass does not pay, so they do not sum to "
            "total_model_ms and are not a partition of it. The tie-break uses "
            "total_model_ms.p95."
        ),
        "peak_inference_gpu_memory_mb": peak_gpu / 2**20,
        "device": str(device),
    }
