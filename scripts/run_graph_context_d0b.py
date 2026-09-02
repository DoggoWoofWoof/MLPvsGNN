"""Stage D0b: does the D0 bridge signal survive contact with an actual ranker?

D0 scored four structural quantities one at a time by rank AUC and found that
``bridge_support`` -- distinct one-hop-from-a-seed intermediates with an edge
into the candidate -- separates relevant from irrelevant candidates on the
stratum historical induction stranded, at 0.87-0.95 mean AUC where the
historical substrate scores exactly 0.5000 by identity.

A univariate AUC is a weak claim. Retrieval scores already rank these candidates
well, and a quantity can be individually informative and add nothing once the
ordinary features are present. This stage asks the multivariate question, and it
is deliberately the cheapest experiment that can answer it: CPU only, no GNN, no
architecture search, no test split, one learner, one seed.

    Does restoring the context help a ranker that already has the retrieval and
    static graph features -- and does the specific bridge quantity help beyond
    the restored descriptor itself?

**The learner is the frozen A3 linear control**, not a new model: a bias-free
linear scorer over the frozen feature block, trained with the frozen
``segmented_listwise_loss``, AdamW, weight decay 0.0, gradient clip 1.0, three
epochs, 512-query batches, at the learning rate A3's protocol already selected
for this dataset. Nothing here is tuned. Using a learner the project already
froze is what keeps this a measurement of the features rather than of a search.

**Arms.** Every arm shares nine arm-invariant columns -- two frozen reciprocal
rank features and the seven frozen static graph features. What differs is the
block appended to them::

    RETRIEVAL_ONLY    nothing.        A floor. Not one of the three arms the
                                      question asks about; it is here so that
                                      "all three are equal" can be told apart
                                      from "the local block does nothing".
    CAND              10 local        the frozen QLS-v1 descriptor over the
                                      historical candidate-induced context.
                                      This arm reproduces A3's feature set.
    TARGET_H1         10 local        the same descriptor over Cq u N1_in(Cq).
    TARGET_H1_BRIDGE  10 local + 1    and log1p(bridge_support) beside it.

Every local block is read out under candidate normalisation, so the scoring set
and the normalising set are both exactly ``Cq`` in every arm and no candidate is
rescaled by a node nobody scores. ``CAND`` is bit-identical under that setting,
which is what makes it the control.

**Splits.** Train fits, validation reports, the test split is refused. Epochs are
fixed at three and the final epoch is reported: no checkpoint is selected on the
split the result is read from. The per-epoch validation trajectory is recorded
so a reader can see whether a different epoch would reorder the arms.

**Strata.** R@k is a per-query quantity, so a candidate stratum has to be lifted
to a query stratum. A query is assigned the *minimum* induced degree over its
in-pool gold candidates -- the hardest gold it contains -- which makes the four
strata a partition and makes ``gold_isolated`` exactly the population D0
measured its 0.9513 on: queries whose gold had no neighbour at all in ``G[Cq]``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import (  # noqa: E402
    build_operators,
    candidate_structure,
    context_nodes,
    qls_local_features,
)
from mp_retrieval.linear_control import (  # noqa: E402
    LOCAL_FEATURE_NAMES,
    RANK_FEATURE_NAMES,
    STATIC_FEATURE_NAMES,
    rank_feature_rows,
    segmented_listwise_loss,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from mp_retrieval.rank_fusion import aggregate_metric_arrays, ranking_metrics  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0 import seed_support  # noqa: E402
from scripts.run_graph_context_pilot import DEGREE_BUCKETS, SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import validate_candidate_contract  # noqa: E402

COMPLETE_STATUS = "GRAPH_CONTEXT_D0B_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D0B_IN_PROGRESS"

#: The nine arm-invariant columns, in the frozen A3 order, so an arm's weight
#: vector can be read beside A3's without re-deriving which column is which.
BASE_FEATURE_NAMES = RANK_FEATURE_NAMES + STATIC_FEATURE_NAMES

#: The one added column, and why it is shaped this way. The frozen kernel scales
#: every count column as ``log1p(count) / max log1p(count)``; a raw bridge count
#: would be the only unbounded column in the block and would carry a different
#: implicit prior. Matching the schema is not cosmetic -- it is what makes "the
#: bridge feature helped" a statement about the quantity and not about its units.
BRIDGE_FEATURE_NAME = "bridge_support_log1p_candidate_normalised"

#: Arm name -> the blocks appended to ``BASE_FEATURE_NAMES``.
ARMS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("RETRIEVAL_ONLY", ()),
    ("CAND", ("local_cand",)),
    ("TARGET_H1", ("local_h1",)),
    ("TARGET_H1_BRIDGE", ("local_h1", "bridge")),
)

#: The three arms the scientific question is about. ``RETRIEVAL_ONLY`` is a
#: floor, reported but not part of the comparison, and named separately so a
#: later reader cannot mistake it for a fourth candidate design.
QUESTION_ARMS = ("CAND", "TARGET_H1", "TARGET_H1_BRIDGE")

BLOCK_WIDTH = {"local_cand": 10, "local_h1": 10, "bridge": 1}

#: What Section 3 asks for, plus the one coverage metric A3 reports, so the two
#: results can be read against each other without a second run.
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")

#: Frozen A3 training settings. Not tuned here and not tunable: the learning rate
#: is the one A3's protocol selected for this dataset on validation R@5 at seed
#: 0, reused rather than re-selected, so this stage performs no search.
TRAINING = {
    "optimizer": "AdamW",
    "epochs": 3,
    "query_batch_size": 512,
    "weight_decay": 0.0,
    "gradient_clip_norm": 1.0,
    "loss": "multi_positive_listwise_cross_entropy",
    "bias": False,
    "epoch_selection": "none -- the final epoch is reported",
}

STRATA = tuple(name for name, _, _ in DEGREE_BUCKETS) + ("no_gold_in_pool",)


def arm_feature_names(blocks: Sequence[str]) -> list[str]:
    names = list(BASE_FEATURE_NAMES)
    for block in blocks:
        if block == "bridge":
            names.append(BRIDGE_FEATURE_NAME)
        else:
            suffix = "cand" if block == "local_cand" else "h1"
            names.extend(f"{name}__{suffix}" for name in LOCAL_FEATURE_NAMES)
    return names


def query_stratum(degree: np.ndarray, positive: np.ndarray) -> str:
    """The hardest in-pool gold's degree bucket, or the no-gold case.

    Minimum rather than maximum: a query holding one stranded gold and one
    well-connected gold is a query the repair could help, and calling it
    ``ordinary`` would hide the population D0 was measuring.
    """
    if not positive.any():
        return "no_gold_in_pool"
    worst = int(degree[positive].min())
    for name, low, high in DEGREE_BUCKETS:
        if low <= worst <= high:
            return name
    raise ValueError(f"Induced degree {worst} falls outside the frozen buckets")


def log1p_candidate_normalised(values: np.ndarray) -> np.ndarray:
    """The frozen kernel's count treatment, applied to one extra column.

    ``log1p`` then divide by the maximum over the scored candidates, with the
    kernel's own zero guard: a column that is zero everywhere stays zero rather
    than becoming a division by zero.
    """
    scaled = np.log1p(np.asarray(values, dtype=np.float64))
    maximum = scaled.max() if scaled.size else 0.0
    return (scaled / (maximum if maximum > 0 else 1.0)).astype(np.float32)


class PackedFeatures:
    """Per-query feature blocks for one split, packed end to end.

    The blocks are stored apart rather than pre-concatenated because three of
    the four arms share the same nine base columns and two share the same local
    block. Assembling per batch costs a copy of 512 queries' rows; storing four
    full matrices would cost four times the memory for the same numbers.
    """

    def __init__(self) -> None:
        self.ptr: list[int] = [0]
        self.base: list[np.ndarray] = []
        self.local_cand: list[np.ndarray] = []
        self.local_h1: list[np.ndarray] = []
        self.bridge: list[np.ndarray] = []
        self.candidate_ids: list[np.ndarray] = []
        self.positive: list[np.ndarray] = []
        self.golds: list[np.ndarray] = []
        self.strata: list[str] = []
        self.query_ids: list[int] = []

    def append(self, *, base, local_cand, local_h1, bridge, ids, positive, golds, stratum, query_index) -> None:
        self.ptr.append(self.ptr[-1] + int(ids.size))
        self.base.append(base)
        self.local_cand.append(local_cand)
        self.local_h1.append(local_h1)
        self.bridge.append(bridge)
        self.candidate_ids.append(ids)
        self.positive.append(positive)
        self.golds.append(golds)
        self.strata.append(stratum)
        self.query_ids.append(int(query_index))

    def finalise(self) -> None:
        self.ptr = np.asarray(self.ptr, dtype=np.int64)
        for name in ("base", "local_cand", "local_h1", "bridge"):
            stack = getattr(self, name)
            setattr(self, name, np.concatenate(stack, axis=0) if stack else np.zeros((0, 1), np.float32))
        self.candidate_ids = np.concatenate(self.candidate_ids) if self.candidate_ids else np.zeros(0, np.int64)
        self.positive = np.concatenate(self.positive) if self.positive else np.zeros(0, bool)
        self.strata = np.asarray(self.strata)
        self.query_ids = np.asarray(self.query_ids, dtype=np.int64)

    @property
    def query_count(self) -> int:
        return int(self.ptr.size - 1)

    def rows(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = self.ptr[queries + 1] - self.ptr[queries]
        offsets = np.empty(queries.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(lengths, out=offsets[1:])
        rows = np.concatenate(
            [np.arange(self.ptr[q], self.ptr[q + 1]) for q in queries]
        ) if queries.size else np.zeros(0, dtype=np.int64)
        return rows, lengths, offsets

    def matrix(self, blocks: Sequence[str], rows: np.ndarray) -> np.ndarray:
        parts = [self.base[rows]]
        for block in blocks:
            parts.append(getattr(self, block)[rows])
        return np.concatenate(parts, axis=1, dtype=np.float32)


def build_split(
    queries,
    rowptr,
    col,
    size,
    operators,
    static,
    dense,
    splade,
    *,
    rrf_constant: int,
    query_cap: int,
    latency: dict[str, list[float]],
) -> PackedFeatures:
    """Compute every arm's features for one split.

    Context construction reads the frozen pool and the frozen retrieval seeds
    and nothing else. Labels are joined afterwards, to mark positives and to
    assign a stratum; they never reach ``context_nodes`` or the feature kernel.
    """
    packed = PackedFeatures()
    for query in queries if query_cap <= 0 else queries[:query_cap]:
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        seed_local = (
            None
            if query.retrieval_seed_local is None
            else query.retrieval_seed_local.numpy().astype(np.int64, copy=False)
        )
        if seed_local is None or seed_local.size == 0:
            continue
        pool = np.unique(candidates)
        if pool.size != candidates.size:
            raise ValueError("The frozen candidate pool contains a duplicate ID")
        seeds = np.unique(candidates[seed_local])
        # Back to the frozen candidate order, so every row of every block lines
        # up with `candidate_index` and therefore with `relevant_local`.
        order = np.searchsorted(pool, candidates)

        started = time.perf_counter()
        cand_nodes = context_nodes("CAND", operators=operators, pool=pool, seeds=seeds)
        local_cand = qls_local_features(
            rowptr=rowptr, col=col, nodes=cand_nodes, pool=pool, seeds=seeds,
            size=size, normalisation="candidate",
        )[order]
        latency["CAND"].append((time.perf_counter() - started) * 1000.0)

        started = time.perf_counter()
        h1_nodes = context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds)
        local_h1 = qls_local_features(
            rowptr=rowptr, col=col, nodes=h1_nodes, pool=pool, seeds=seeds,
            size=size, normalisation="candidate",
        )[order]
        support = seed_support(
            rowptr, col, h1_nodes, pool, seeds, size,
            edge_source=operators.edge_source,
        )
        bridge = log1p_candidate_normalised(support["bridge_support"])[order, None]
        latency["TARGET_H1"].append((time.perf_counter() - started) * 1000.0)

        base = np.concatenate(
            (
                rank_feature_rows(
                    dense[query.query_index], splade[query.query_index], candidates,
                    constant=rrf_constant,
                ),
                np.asarray(static[candidates], dtype=np.float32),
            ),
            axis=1,
            dtype=np.float32,
        )
        degree = candidate_structure(rowptr, col, pool, pool, size)["induced_degree"][order]

        positive = np.zeros(candidates.size, dtype=bool)
        relevant_local = query.relevant_local.numpy().astype(np.int64, copy=False)
        if relevant_local.size:
            positive[relevant_local] = True

        packed.append(
            base=base,
            local_cand=local_cand.astype(np.float32, copy=False),
            local_h1=local_h1.astype(np.float32, copy=False),
            bridge=bridge.astype(np.float32, copy=False),
            ids=candidates,
            positive=positive,
            golds=query.relevant_global.numpy().astype(np.int64, copy=False),
            stratum=query_stratum(degree, positive),
            query_index=query.query_index,
        )
    packed.finalise()
    return packed


def evaluate(weight: np.ndarray, packed: PackedFeatures, blocks: Sequence[str]) -> dict[str, Any]:
    """Validation metrics overall and per query stratum.

    Ranking ties break on ascending global node ID, the same convention every
    sealed result in this project uses, so a metric here is comparable with one
    from A3 rather than merely similar to it.
    """
    total = int(packed.ptr[-1])
    scores = packed.matrix(blocks, np.arange(total)) @ weight
    rows_per_query = []
    for query in range(packed.query_count):
        start, end = int(packed.ptr[query]), int(packed.ptr[query + 1])
        ids = packed.candidate_ids[start:end]
        ranking = ids[np.lexsort((ids, -scores[start:end]))]
        rows_per_query.append(ranking_metrics(ranking, packed.golds[query], ids))

    def _aggregate(selected: Sequence[int]) -> dict[str, float | int]:
        arrays = {
            metric: np.asarray(
                [
                    np.nan if rows_per_query[index][metric] is None
                    else float(rows_per_query[index][metric])
                    for index in selected
                ],
                dtype=np.float64,
            )
            for metric in METRICS
        }
        summary = {
            key: value for key, value in aggregate_metric_arrays(arrays).items()
            if key in METRICS
        }
        summary["queries"] = int(len(selected))
        return summary

    everything = list(range(packed.query_count))
    return {
        "overall": _aggregate(everything),
        "by_gold_stratum": {
            stratum: _aggregate(np.flatnonzero(packed.strata == stratum).tolist())
            for stratum in STRATA
        },
    }


def train_arm(
    blocks: Sequence[str],
    train: PackedFeatures,
    validation: PackedFeatures,
    *,
    seed: int,
    learning_rate: float,
) -> dict[str, Any]:
    """Fit the frozen A3 linear control on train and read it out on validation.

    Weights start at zero, as A3's do, so the arms differ by their features
    rather than by their initialisation.
    """
    seed_everything(seed)
    width = len(BASE_FEATURE_NAMES) + sum(BLOCK_WIDTH[block] for block in blocks)
    weight = torch.zeros(width, requires_grad=True)
    optimizer = torch.optim.AdamW(
        [weight], lr=learning_rate, weight_decay=float(TRAINING["weight_decay"])
    )
    eligible = np.flatnonzero(
        np.asarray([
            train.positive[int(train.ptr[q]) : int(train.ptr[q + 1])].any()
            for q in range(train.query_count)
        ])
    )
    batch_size = int(TRAINING["query_batch_size"])
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(int(TRAINING["epochs"])):
        order = eligible.tolist()
        random.Random(seed + epoch * 1_000_003).shuffle(order)
        losses: list[float] = []
        for start in range(0, len(order), batch_size):
            batch = np.asarray(order[start : start + batch_size], dtype=np.int64)
            rows, lengths, offsets = train.rows(batch)
            features = torch.from_numpy(train.matrix(blocks, rows))
            segments = torch.from_numpy(
                np.repeat(np.arange(batch.size, dtype=np.int64), lengths)
            )
            positives = torch.from_numpy(np.flatnonzero(train.positive[rows]))
            optimizer.zero_grad(set_to_none=True)
            loss = segmented_listwise_loss(
                features @ weight, segments, positives, num_queries=int(batch.size)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_([weight], float(TRAINING["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach()))
        snapshot = weight.detach().numpy().copy()
        history.append(
            {
                "epoch": epoch + 1,
                "mean_loss": float(np.mean(losses)) if losses else float("nan"),
                "validation": evaluate(snapshot, validation, blocks)["overall"],
            }
        )
    seconds = time.perf_counter() - started
    final = weight.detach().numpy().copy()
    return {
        "parameters": int(width),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "train_queries": int(train.query_count),
        "train_queries_with_in_pool_gold": int(eligible.size),
        "training_seconds": round(seconds, 2),
        "history": history,
        "weights": [float(value) for value in final],
        "validation": evaluate(final, validation, blocks),
    }


def _percentiles(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {key: float("nan") for key in ("p50", "p95", "p99", "mean", "max")}
    return {
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }


def load_or_build_static(
    feature_cache: Path,
    graph_path: Path,
    size: int,
    *,
    pagerank_damping: float,
    pagerank_iterations: int,
    clustering_max_wedges: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The seven frozen static graph features, sealed if present and built if not.

    The active Modal workspace holds the topology-only slice of this dataset --
    graph, candidates, seeds, golds and splits, but neither the embeddings nor
    the sealed `derived/` caches, which is why every graph-context stage so far
    has opened the dataset with ``require_embeddings=False``. The static block is
    a deterministic function of ``graph.pt``, so it can be rebuilt rather than
    replicated, through the *shipped* builder that produced the sealed file and
    at the frozen parameters, so the two agree by construction rather than by
    resemblance.

    Which path was taken is returned and recorded. A result that quietly does
    not say whether it read a sealed artifact or rebuilt one is a result whose
    provenance cannot be checked later.
    """
    sealed = Path(feature_cache) / "static.npy"
    if sealed.is_file():
        static = np.load(sealed, mmap_mode="r")
        provenance: dict[str, Any] = {"source": "sealed", "path": str(sealed)}
    else:
        from mp_retrieval.structural_features import build_static_features

        started = time.perf_counter()
        static = build_static_features(
            graph_path,
            pagerank_damping=pagerank_damping,
            pagerank_iterations=pagerank_iterations,
            clustering_max_wedges=clustering_max_wedges,
        )
        provenance = {
            "source": "rebuilt_from_graph",
            "why": f"{sealed} is absent in this workspace",
            "builder": "mp_retrieval.structural_features.build_static_features",
            "pagerank_damping": pagerank_damping,
            "pagerank_iterations": pagerank_iterations,
            "clustering_max_wedges_per_node": clustering_max_wedges,
            "seconds": round(time.perf_counter() - started, 1),
        }
    if static.ndim != 2 or static.shape[1] != len(STATIC_FEATURE_NAMES):
        raise ValueError("The static feature matrix has an unexpected shape")
    if int(static.shape[0]) != size:
        raise ValueError("The static feature matrix does not cover the graph")
    return static, provenance


def _alignment_against_sealed_cache(
    packed: PackedFeatures, feature_cache: Path | None
) -> dict[str, Any]:
    """Check the recomputed CAND block against the sealed A3 structural cache.

    The point of the check is that this stage's ``CAND`` arm is A3's feature set
    and not merely a feature set that resembles it. The sealed cache stores
    float16, so the comparison is made in float16 -- an exact match there is the
    strongest statement the stored precision can support.
    """
    if feature_cache is None:
        return {"checked": False, "why": "no sealed structural cache was supplied"}
    root = Path(feature_cache)
    if not (root / "metadata.json").is_file():
        return {"checked": False, "why": f"no sealed structural cache at {root}"}
    from mp_retrieval.structural_controls import FrozenStructuralCache

    cache = FrozenStructuralCache.load(root)
    differences: list[float] = []
    exact = 0
    total = 0
    for position, query_index in enumerate(packed.query_ids.tolist()):
        start, end = int(packed.ptr[position]), int(packed.ptr[position + 1])
        sealed_start = int(cache.candidate_ptr[query_index])
        sealed_end = int(cache.candidate_ptr[query_index + 1])
        if sealed_end - sealed_start != end - start:
            return {
                "checked": True,
                "aligned": False,
                "why": f"candidate count differs at query {query_index}",
            }
        sealed = np.asarray(cache.local[sealed_start:sealed_end], dtype=np.float16)
        mine = packed.local_cand[start:end].astype(np.float16)
        exact += int((sealed == mine).all(axis=1).sum())
        total += end - start
        differences.append(
            float(np.abs(sealed.astype(np.float32) - mine.astype(np.float32)).max())
        )
    return {
        "checked": True,
        "aligned": exact == total,
        "rows": int(total),
        "rows_bit_identical_in_float16": int(exact),
        "max_absolute_difference": float(max(differences)) if differences else 0.0,
        "precision": "float16, the sealed cache's stored dtype",
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D0b is a development diagnostic; the test split is not read")
    if "train" not in args.splits or "validation" not in args.splits:
        raise ValueError("Stage D0b fits on train and reports on validation; both are required")

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=False)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )

    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    size = int(dataset.num_nodes)
    operators = build_operators(rowptr, col, size)

    static, static_provenance = load_or_build_static(
        Path(args.feature_cache), Path(args.data) / "graph.pt", size,
        pagerank_damping=float(args.static_pagerank_damping),
        pagerank_iterations=int(args.static_pagerank_iterations),
        clustering_max_wedges=int(args.static_clustering_max_wedges),
    )
    dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d0b",
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "arms": {name: arm_feature_names(blocks) for name, blocks in ARMS},
        "question_arms": list(QUESTION_ARMS),
        "floor_arm": "RETRIEVAL_ONLY",
        "training": dict(TRAINING) | {"learning_rate": float(args.learning_rate), "seed": int(args.seed)},
        "contract": {
            "gpu_used": False,
            "gnn_trained": False,
            "message_passing": False,
            "scored_nodes": "exactly Cq in every arm",
            "normalisation": "candidate-readout in every arm",
            "normalization_nodes": "exactly Cq in every arm",
            "candidate_pools_modified": False,
            "gold_ids_used_in_context_construction": False,
            "test_split_read": False,
            "learning_rate_selected_here": False,
            "learning_rate_provenance": "the value A3's frozen protocol selected for this dataset",
            "epoch_selected_on_validation": False,
            "evidence_class": "development diagnostic, not an evaluation",
        },
        "static_features": static_provenance,
    }

    latency: dict[str, list[float]] = {"CAND": [], "TARGET_H1": []}
    built = {}
    for split_name in ("train", "validation"):
        queries = dataset.split(SPLITS[split_name])
        if not queries:
            raise ValueError(f"Stage D0b needs a non-empty {split_name} split")
        started = time.perf_counter()
        built[split_name] = build_split(
            queries, rowptr, col, size, operators, static, dense, splade,
            rrf_constant=int(args.rrf_constant),
            query_cap=int(args.query_cap),
            latency=latency,
        )
        result.setdefault("feature_build", {})[split_name] = {
            "queries": built[split_name].query_count,
            "candidate_rows": int(built[split_name].ptr[-1]),
            "seconds": round(time.perf_counter() - started, 1),
        }
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    result["feature_build"]["latency_ms_per_query"] = {
        arm: _percentiles(values) for arm, values in latency.items()
    }
    result["gold_stratum_counts"] = {
        split_name: {
            stratum: int((built[split_name].strata == stratum).sum())
            for stratum in STRATA
        }
        for split_name in built
    }
    result["cand_arm_alignment"] = _alignment_against_sealed_cache(
        built["validation"], args.feature_cache
    )
    _atomic_json(args.output, result)

    result["results"] = {}
    for name, blocks in ARMS:
        result["results"][name] = train_arm(
            blocks, built["train"], built["validation"],
            seed=int(args.seed), learning_rate=float(args.learning_rate),
        )
        _atomic_json(args.output, result)
        if checkpoint_hook is not None:
            checkpoint_hook()

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while computing a read-only diagnostic")
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--query-cap", type=int, default=0, help="0 means the whole split")
    parser.add_argument("--rrf-constant", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    # The frozen static-feature parameters, carried so a rebuild in a workspace
    # without the sealed cache produces the same matrix rather than a similar one.
    parser.add_argument("--static-pagerank-damping", type=float, default=0.85)
    parser.add_argument("--static-pagerank-iterations", type=int, default=30)
    parser.add_argument("--static-clustering-max-wedges", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
