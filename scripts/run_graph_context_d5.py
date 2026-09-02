"""Stage D5: do the graded retrieval prior and seed-distance geometry compose?

D3 decomposed QLS-v1's local block and found seed identity dominant, distance
geometry worth its place on depth, and the remaining six columns lower-priority.
D4 then added A3's graded retrieval evidence on top of the bare seed bit and got
+4.13 R@5, +9.18 R@1 and +13.03 MRR -- larger than any structural increment --
but *lost* R@20 and FullCov@20 against the full historical block.

So the two signals look complementary: the prior owns the head of the ranking,
geometry owns the depth. Nothing yet says whether they add.

    Z1   SEED_ID_ONLY                     the bare bit.            D3, reused.
    Z2   DISTANCE_ONLY                    the frozen bucket group. D3, reused.
    Z3   FULL_LOCAL                       all ten columns.         D3, reused.
    Z1R  SEED_PLUS_GRADED_RETRIEVAL       bit + prior.             D4, reused.
    Z2R  DISTANCE_PLUS_GRADED_RETRIEVAL   geometry + prior.            NEW.

    delta_prior_given_geometry  = Z2R - Z2    does D4's gain survive geometry?
    delta_geometry_given_prior  = Z2R - Z1R   does geometry survive the prior?
    interaction                 = (Z2R - Z1R) - (Z2 - Z1)

Five reference points for one new run.

**The retrieval prior must be bit-identical to D4's.** `candidate_readout`
rescales columns 4-9 and leaves 0-3 alone, so a prior written into columns 4-6
*before* that normalisation would not be the prior D4 measured, and `Z2R - Z1R`
would confound adding geometry with changing the retrieval transform. The
shipped `build_local_features` already applies `candidate_readout` before it
returns, so injecting afterwards is safe -- but that is a property of call
ordering, not of the column indices, and D4 proved nothing about it. Here it is
proven: the prior columns are compared elementwise against the block D4's own
`build_graded_retrieval_block` produces, and `max_abs_diff` must be exactly 0.

The distance group is proven the same way, against `masked_local` over the same
`build_local_features` output D3 masked. So Z2R is exactly D3's geometry plus
exactly D4's prior, and nothing else.

Both contexts are `G[Cq]`. No GNN. No test split. One dataset, one seed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.graph_context import NORMALISED_COLUMNS, build_operators  # noqa: E402
from mp_retrieval.linear_control import (  # noqa: E402
    LOCAL_FEATURE_NAMES,
    rank_feature_rows,
)
from mp_retrieval.protocol import seed_everything  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_graph_context_d0b import load_or_build_static  # noqa: E402
from scripts.run_graph_context_d1 import (  # noqa: E402
    METRICS,
    MODEL_NAME,
    _percentiles,
    build_local_features,
    context_feature_store,
    holdout_split,
    stratify,
)
from scripts.run_graph_context_d3 import (  # noqa: E402
    ALL_COLUMNS,
    DISTANCE_COLUMNS,
    SEED_ID_COLUMN,
    assert_seed_identity_column,
    masked_local,
)
from scripts.run_graph_context_d4 import (  # noqa: E402
    AGREEMENT_COLUMN as D4_AGREEMENT_COLUMN,
    DENSE_RANK_COLUMN as D4_DENSE_RANK_COLUMN,
    RETRIEVAL_COLUMNS as D4_RETRIEVAL_COLUMNS,
    SPLADE_RANK_COLUMN as D4_SPLADE_RANK_COLUMN,
    a3_rank_feature_audit,
    build_graded_retrieval_block,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _build_model,
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D5_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D5_IN_PROGRESS"

#: Z2R's layout. The distance group keeps its historical slots, so the geometry
#: is byte-for-byte D3's. The prior moves to the first three slots the group
#: leaves free -- which `candidate_readout` *would* rescale if anything were
#: written there before it ran. Nothing is: the injection happens after
#: `build_local_features` returns, and `assert_prior_matches_d4` proves it.
PRIOR_COLUMNS = (4, 5, 6)
DENSE_RANK_COLUMN, SPLADE_RANK_COLUMN, AGREEMENT_COLUMN = PRIOR_COLUMNS
ZERO_COLUMNS = (7, 8, 9)

COLUMN_SEMANTICS: dict[int, str] = {
    0: "distance_0, unchanged: exactly I[d in Sq]",
    1: "distance_1, unchanged",
    2: "distance_2, unchanged",
    3: "distance_3_plus_or_unreachable, unchanged",
    DENSE_RANK_COLUMN: "dense_reciprocal_rank (A3), injected after candidate_readout",
    SPLADE_RANK_COLUMN: "splade_reciprocal_rank (A3), injected after candidate_readout",
    AGREEMENT_COLUMN: "retriever_agreement, injected after candidate_readout",
}

NEW_ARM = "DISTANCE_PLUS_GRADED_RETRIEVAL"
D5_ARMS = (
    "SEED_ID_ONLY",
    "DISTANCE_ONLY",
    "FULL_LOCAL",
    "SEED_PLUS_GRADED_RETRIEVAL",
    NEW_ARM,
)
REUSED_FROM_D3 = ("SEED_ID_ONLY", "DISTANCE_ONLY", "FULL_LOCAL")
REUSED_FROM_D4 = ("SEED_PLUS_GRADED_RETRIEVAL",)

#: D3 trained two of the three rows D5 takes from it. `FULL_LOCAL` it reused
#: from D2, where the same ten-column arm ran as `FULL_CAND`. That row is
#: therefore a two-hop reuse here, and is recorded as measured in D2 rather
#: than relabelled to the file D5 happens to read it out of.
D3_TRAINED_THERE = ("SEED_ID_ONLY", "DISTANCE_ONLY")
D3_REUSED_FROM_D2 = {"FULL_LOCAL": ("stage_d2", "FULL_CAND")}

#: The two complementary conditionals, and the interaction they bracket.
INCREMENTS: tuple[tuple[str, str, str], ...] = (
    ("delta_prior_given_geometry", NEW_ARM, "DISTANCE_ONLY"),
    ("delta_geometry_given_prior", NEW_ARM, "SEED_PLUS_GRADED_RETRIEVAL"),
)
INTERACTION = ("delta_geometry_given_prior", ("DISTANCE_ONLY", "SEED_ID_ONLY"))

CONTEXT = "CAND"
HOLDOUT_FRACTION = 0.1

_INCREMENT_MEANING = {
    "delta_prior_given_geometry": (
        "incremental value of the graded retrieval prior once seed-distance "
        "geometry is already present"
    ),
    "delta_geometry_given_prior": (
        "incremental value of seed-distance geometry once the graded retrieval "
        "prior is already present"
    ),
}


def prior_value_columns(
    queries, candidate_ptr: np.ndarray, dense, splade, *, constant: int, dtype
) -> tuple[np.ndarray, dict[str, Any]]:
    """The three retrieval columns as values, independent of where they land.

    Returned in the local block's own dtype so that the float16 rounding is
    applied once, here, rather than differing between the arm that stores them
    at columns 1-3 and the arm that stores them at 4-6.
    """
    rows = int(candidate_ptr[-1])
    values = np.zeros((rows, len(PRIOR_COLUMNS)), dtype=dtype)
    agreed = 0
    for position, query in enumerate(queries):
        start = int(candidate_ptr[position])
        end = int(candidate_ptr[position + 1])
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        if candidates.size != end - start:
            raise RuntimeError(
                f"Query {position} spans {end - start} rows but has {candidates.size} candidates"
            )
        ranks = rank_feature_rows(
            np.asarray(dense[query.query_index]),
            np.asarray(splade[query.query_index]),
            candidates,
            constant=constant,
        )
        in_dense = ranks[:, 0] > 0.0
        in_splade = ranks[:, 1] > 0.0
        if not np.all(in_dense | in_splade):
            raise RuntimeError(
                f"Query {position} has a candidate ranked by neither retriever"
            )
        both = in_dense & in_splade
        values[start:end, 0] = ranks[:, 0]
        values[start:end, 1] = ranks[:, 1]
        values[start:end, 2] = both.astype(np.float32)
        agreed += int(both.sum())
    return values, {
        "definition": "B, the full candidate retrieval prior",
        "columns": {str(index): COLUMN_SEMANTICS[index] for index in PRIOR_COLUMNS},
        "candidate_rows": rows,
        "rows_ranked_by_both": agreed,
        "rows_ranked_by_neither": 0,
        "constant_K": int(constant),
        "dtype": str(np.dtype(dtype)),
        "injected_after_candidate_readout": True,
    }


def assert_prior_matches_d4(
    queries, local: np.ndarray, candidate_ptr: np.ndarray, dense, splade,
    values: np.ndarray, *, constant: int,
) -> dict[str, Any]:
    """The prior D5 injects must be the prior D4 measured, to the last bit.

    Built by calling D4's own `build_graded_retrieval_block` rather than a
    shared helper, so the comparison is against the shipped code that produced
    D4's numbers and not against a second implementation that might agree with
    D5 while both drift from D4.

    This is the check that makes `Z2R - Z1R` a geometry comparison. Columns 4-9
    are the ones `candidate_readout` rescales; if the injection ever moved ahead
    of that normalisation, the values would differ here and the stage would
    refuse rather than quietly measuring "geometry plus a different transform".
    """
    d4_block, _provenance = build_graded_retrieval_block(
        queries, local, candidate_ptr, dense, splade, constant=constant
    )
    theirs = d4_block[:, list(D4_RETRIEVAL_COLUMNS)]
    difference = np.abs(
        np.asarray(theirs, dtype=np.float64) - np.asarray(values, dtype=np.float64)
    )
    max_abs_diff = float(difference.max()) if difference.size else 0.0
    if max_abs_diff != 0.0:
        raise RuntimeError(
            "D5's retrieval prior is not D4's: max_abs_diff "
            f"{max_abs_diff!r} over {values.shape[0]} candidate rows. Z2R - Z1R "
            "would confound adding geometry with changing the retrieval transform."
        )
    if not np.array_equal(theirs, values):
        raise RuntimeError("D5's retrieval prior differs from D4's in representation")
    return {
        "compared_against": "scripts.run_graph_context_d4.build_graded_retrieval_block",
        "d4_columns": [D4_DENSE_RANK_COLUMN, D4_SPLADE_RANK_COLUMN, D4_AGREEMENT_COLUMN],
        "d5_columns": list(PRIOR_COLUMNS),
        "candidate_rows_compared": int(values.shape[0]),
        "max_abs_diff": 0.0,
        "elementwise_identical": True,
        "why_it_matters": (
            "candidate_readout rescales columns 4-9 and leaves 0-3 untouched, so a "
            "prior written into 4-6 before normalisation would not be D4's prior; "
            "the injection happens after build_local_features returns, and this "
            "proves the values survived unchanged"
        ),
        "normalised_columns": list(NORMALISED_COLUMNS),
        "prior_lands_in_normalised_slots": bool(
            set(PRIOR_COLUMNS) & set(NORMALISED_COLUMNS)
        ),
    }


def assert_geometry_matches_d3(block: np.ndarray, local: np.ndarray) -> dict[str, Any]:
    """Z2R's distance group must be exactly the group D3 masked for `DISTANCE_ONLY`."""

    theirs = masked_local(local, DISTANCE_COLUMNS)
    columns = list(DISTANCE_COLUMNS)
    if not np.array_equal(block[:, columns], theirs[:, columns]):
        raise RuntimeError("Z2R's distance columns are not D3's DISTANCE_ONLY geometry")
    buckets = np.asarray(block[:, columns], dtype=np.float32)
    if not np.array_equal(buckets.sum(axis=1), np.ones(buckets.shape[0], dtype=np.float32)):
        raise RuntimeError("The seed-distance buckets are not a complete one-hot")
    if block[:, list(ZERO_COLUMNS)].any():
        raise RuntimeError("Z2R carries structure outside its declared slots")
    return {
        "compared_against": "scripts.run_graph_context_d3.masked_local over DISTANCE_COLUMNS",
        "columns": [LOCAL_FEATURE_NAMES[index] for index in DISTANCE_COLUMNS],
        "candidate_rows_compared": int(block.shape[0]),
        "elementwise_identical": True,
        "complete_one_hot": True,
        "columns_left_zero": [LOCAL_FEATURE_NAMES[index] for index in ZERO_COLUMNS],
        "not_rescaled": "columns 0-3 are outside NORMALISED_COLUMNS by design",
    }


def _condition_rows(source: dict[str, Any], result: dict[str, Any], args, label: str):
    """The conditions that must agree for any earlier arm to be reusable here."""

    yield ("dataset", args.dataset, source.get("dataset"))
    yield ("data_fingerprint", args.data_fingerprint_sha256,
           source.get("data_fingerprint_sha256"))
    yield ("context", CONTEXT, source.get("context"))
    yield ("model", MODEL_NAME, source.get("model"))
    yield ("num_nodes", result["num_nodes"], source.get("num_nodes"))
    yield ("candidate_contract",
           result["candidate_contract"].get("observed_contract_sha256"),
           source.get("candidate_contract", {}).get("observed_contract_sha256"))
    yield ("local_feature_schema", list(LOCAL_FEATURE_NAMES),
           source.get("local_feature_schema"))
    yield ("candidate_rows", result["feature_build"]["candidate_rows"],
           source.get("feature_build", {}).get("candidate_rows"))
    yield ("static_feature_source", result["static_features"].get("source"),
           source.get("static_features", {}).get("source"))
    yield ("architecture_changed", False,
           source.get("ablation", {}).get("architecture_changed"))
    yield ("same_epoch_budget_for_every_arm", True,
           source.get("ablation", {}).get("same_epoch_budget_for_every_arm"))
    proof = source.get("seed_identity_proof", {})
    ours = result["seed_identity_proof"]
    yield (f"{label}_seed_identity_column", ours["column"], proof.get("column"))
    yield (f"{label}_seed_identity_rows", ours["candidate_rows_compared"],
           proof.get("candidate_rows_compared"))
    yield (f"{label}_seed_identity_seed_rows", ours["seed_rows"], proof.get("seed_rows"))
    yield (f"{label}_seed_identity_mismatches", 0, proof.get("mismatches"))
    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        yield (f"split_{key}", result["splits"][key], source.get("splits", {}).get(key))
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed"):
        yield (f"training_{key}", result["training"][key],
               source.get("training", {}).get(key))


def verify_reuse(d3: dict[str, Any], d4: dict[str, Any], result: dict[str, Any], args):
    """Refuse to reuse any of the four earlier arms across a condition that differs.

    Four reused rows are most of this stage's value, and every one of them was
    measured in a different container from this one. Each check names a way two
    stages could silently diverge; a mismatch raises rather than degrading into
    a five-row table assembled from different experiments.
    """
    if d3.get("status") != "GRAPH_CONTEXT_D3_COMPLETE":
        raise RuntimeError(f"D3 result is not complete: {d3.get('status')!r}")
    if d4.get("status") != "GRAPH_CONTEXT_D4_COMPLETE":
        raise RuntimeError(f"D4 result is not complete: {d4.get('status')!r}")

    checks: dict[str, Any] = {}
    for source, label in ((d3, "d3"), (d4, "d4")):
        for name, mine, theirs in _condition_rows(source, result, args, label):
            key = name if name.startswith(label) else f"{label}_{name}"
            checks[key] = {"d5": mine, "source": theirs, "match": mine == theirs}

    # The reused rows must be the arms they are named after, and must have been
    # trained where they claim, rather than themselves reused from further back.
    for arm in D3_TRAINED_THERE:
        checks[f"d3_{arm}_trained_there"] = {
            "d5": True,
            "source": d3.get("results", {}).get(arm, {}).get("retrained_here"),
            "match": d3.get("results", {}).get(arm, {}).get("retrained_here") is True,
        }
    # The one row D3 did not train. Following it back to D2 rather than trusting
    # the label on D3's copy is the whole point of checking provenance at all.
    for arm, (stage, name) in D3_REUSED_FROM_D2.items():
        row = d3.get("results", {}).get(arm, {})
        observed = [row.get("measured_in"), row.get("measured_as"),
                    row.get("retrained_here")]
        checks[f"d3_{arm}_reused_from_d2"] = {
            "d5": [stage, name, False],
            "source": observed,
            "match": observed == [stage, name, False],
        }
    checks["d3_arms_trained_here"] = {
        "d5": ["SEED_ID_ONLY", "DISTANCE_ONLY"],
        "source": d3.get("arms_trained_here"),
        "match": d3.get("arms_trained_here") == ["SEED_ID_ONLY", "DISTANCE_ONLY"],
    }
    checks["d4_arms_trained_here"] = {
        "d5": ["SEED_PLUS_GRADED_RETRIEVAL"],
        "source": d4.get("arms_trained_here"),
        "match": d4.get("arms_trained_here") == ["SEED_PLUS_GRADED_RETRIEVAL"],
    }
    checks["d4_seed_bit_not_replaced"] = {
        "d5": False,
        "source": d4.get("ablation", {}).get("seed_bit_replaced"),
        "match": d4.get("ablation", {}).get("seed_bit_replaced") is False,
    }
    checks["d4_rrf_constant"] = {
        "d5": int(args.rrf_constant),
        "source": d4.get("a3_rank_feature_audit", {}).get("constant_K"),
        "match": int(args.rrf_constant)
        == d4.get("a3_rank_feature_audit", {}).get("constant_K"),
    }
    # D4 reused D3's SEED_ID_ONLY. If the two files disagree about it, the chain
    # this table is assembled from is broken somewhere behind us.
    checks["chain_seed_id_only_agrees"] = {
        "d5": d3.get("ladder", {}).get("SEED_ID_ONLY"),
        "source": d4.get("ladder", {}).get("SEED_ID_ONLY"),
        "match": d3.get("ladder", {}).get("SEED_ID_ONLY")
        == d4.get("ladder", {}).get("SEED_ID_ONLY"),
    }

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "Earlier arms cannot be reused; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d5']!r} vs {checks[name]['source']!r})"
                        for name in failed)
        )
    return {
        "reused_arms": {
            **{arm: "stage_d3" for arm in D3_TRAINED_THERE},
            **{arm: f"{stage} via stage_d3" for arm, (stage, _name)
               in D3_REUSED_FROM_D2.items()},
            **{arm: "stage_d4" for arm in REUSED_FROM_D4},
        },
        "sources": ["stage_d3.json", "stage_d4.json"],
        "why": (
            "Four of the five rows were measured under identical conditions in "
            "earlier stages. Retraining them would pay four times over for the "
            "same numbers and would give some arms a different initialisation."
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "new_runs": 1,
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D5 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D5 fits on train and reports on validation; both and only both")

    sources = {}
    for name, path in (("d3", args.d3_result), ("d4", args.d4_result)):
        resolved = Path(path)
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Stage D5 reuses four earlier arms but {resolved} is absent"
            )
        sources[name] = json.loads(resolved.read_text(encoding="utf-8"))
    d3, d4 = sources["d3"], sources["d4"]

    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    contract_before = dataset.metadata["candidate_contract_sha256"]
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    dense = np.load(Path(args.data) / "dense_top200_all.npy", mmap_mode="r")
    splade = np.load(Path(args.data) / "splade_top200_all.npy", mmap_mode="r")

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

    train_all = dataset.split(SPLITS["train"])
    validation = dataset.split(SPLITS["validation"])
    if not train_all or not validation:
        raise RuntimeError("Stage D5 requires non-empty train and validation splits")
    train, holdout = holdout_split(train_all, float(args.holdout_fraction))
    opened = train + holdout + validation
    strata = stratify(validation, rowptr, col, size)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    node_embeddings = torch.from_numpy(np.asarray(dataset.node_array)).to(
        device=device, dtype=torch.float32
    )
    query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(
        device=device, dtype=torch.float32
    )

    latency: list[float] = []
    started = time.perf_counter()
    # The same call D3 and D4 made. `candidate_readout` runs inside it, so
    # everything injected below lands after normalisation.
    local, candidate_ptr = build_local_features(
        opened, rowptr, col, size, operators, CONTEXT,
        damping=float(args.damping), ppr_iterations=int(args.ppr_iterations),
        latency=latency,
    )
    build_seconds = time.perf_counter() - started

    seed_identity = assert_seed_identity_column(opened, local)
    rank_audit = a3_rank_feature_audit(int(args.rrf_constant), int(dense.shape[1]))

    started = time.perf_counter()
    values, prior = prior_value_columns(
        opened, candidate_ptr, dense, splade,
        constant=int(args.rrf_constant), dtype=local.dtype,
    )
    prior_equivalence = assert_prior_matches_d4(
        opened, local, candidate_ptr, dense, splade, values, constant=int(args.rrf_constant)
    )
    treatment = masked_local(local, DISTANCE_COLUMNS)
    treatment[:, list(PRIOR_COLUMNS)] = values
    geometry_equivalence = assert_geometry_matches_d3(treatment, local)
    prior_seconds = time.perf_counter() - started

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d5",
        "question": (
            "after graded retrieval evidence is available, does seed-distance "
            "geometry still provide incremental ranking value?"
        ),
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "local_feature_schema": list(LOCAL_FEATURE_NAMES),
        "column_semantics": {str(key): value for key, value in COLUMN_SEMANTICS.items()},
        "arms": {
            "SEED_ID_ONLY": {"carries": ["distance_0"], "measured_in": "stage_d3"},
            "DISTANCE_ONLY": {
                "carries": [LOCAL_FEATURE_NAMES[i] for i in DISTANCE_COLUMNS],
                "measured_in": "stage_d3",
            },
            "FULL_LOCAL": {
                "carries": [LOCAL_FEATURE_NAMES[i] for i in ALL_COLUMNS],
                "measured_in": "stage_d2",
                "measured_as": "FULL_CAND",
                "reused_via": "stage_d3",
            },
            "SEED_PLUS_GRADED_RETRIEVAL": {
                "carries": ["distance_0", "the A3 prior at columns 1-3"],
                "measured_in": "stage_d4",
            },
            NEW_ARM: {
                "carries": [COLUMN_SEMANTICS[index] for index in (*DISTANCE_COLUMNS,
                                                                  *PRIOR_COLUMNS)],
                "measured_in": "stage_d5",
            },
        },
        "seed_identity_proof": seed_identity,
        "a3_rank_feature_audit": rank_audit,
        "graded_retrieval_prior": prior | {"seconds": round(prior_seconds, 1)},
        "prior_equivalence_with_d4": prior_equivalence,
        "geometry_equivalence_with_d3": geometry_equivalence,
        "ablation": {
            "method": "injecting the prior into free slots after candidate_readout",
            "why": (
                "the distance group keeps its historical slots so the geometry is "
                "D3's exactly, and the prior lands in three columns that are zero "
                "in DISTANCE_ONLY, so local_dim, head width and parameter count "
                "are unchanged"
            ),
            "injection_is_post_normalisation": True,
            "injection_note": (
                "candidate_readout rescales columns 4-9; build_local_features applies "
                "it before returning, so values written afterwards are not rescaled. "
                "Proven against D4's own block rather than assumed."
            ),
            "architecture_changed": False,
            "same_epoch_budget_for_every_arm": True,
        },
        "static_features": static_provenance,
        "splits": {
            "train_fit": len(train),
            "train_holdout_for_epoch_selection": len(holdout),
            "validation_reported": len(validation),
            "test_read": False,
        },
        "feature_build": {
            "seconds": round(build_seconds, 1),
            "candidate_rows": int(candidate_ptr[-1]),
            "shared_by_every_arm": True,
            "latency_ms_per_query": _percentiles(latency),
        },
        "training": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "epoch_selection": "a deterministic tail of train, never validation",
            "validation_reads_per_arm": 1,
        },
        "contract": {
            "gnn_trained": False,
            "message_passing": False,
            "scored_nodes": "exactly Cq in every arm",
            "context_restored": False,
            "candidate_pools_modified": False,
            "test_split_read": False,
            "epoch_selected_on_validation": False,
            "hyperparameters_searched": False,
            "architecture_changed": False,
            "evidence_class": "development experiment, not an evaluation",
        },
        "results": {},
    }
    result["reuse"] = verify_reuse(d3, d4, result, args)
    _atomic_json(args.output, result)

    # `measured_in` names where the numbers were produced, not the file they
    # were read out of, so a two-hop row keeps pointing at D2.
    for arm in REUSED_FROM_D3:
        source = dict(d3["results"][arm])
        result["results"][arm] = source | {
            "measured_in": source.get("measured_in", "stage_d3"),
            "reused_via": "stage_d3",
            "retrained_here": False,
        }
    for arm in REUSED_FROM_D4:
        source = dict(d4["results"][arm])
        result["results"][arm] = source | {
            "measured_in": source.get("measured_in", "stage_d4"),
            "reused_via": "stage_d4",
            "retrained_here": False,
        }
    _atomic_json(args.output, result)

    target_parameters = int(args.baseline["selected_gnn"]["parameters"]["parameters"])
    features = context_feature_store(
        opened, static, treatment, candidate_ptr, len(dataset.queries), arm=CONTEXT
    )
    seed_everything(int(args.seed))
    model = _build_model(MODEL_NAME, dataset, features, args.selected_gnn, target_parameters, args)
    model, telemetry = _fit(
        MODEL_NAME, model, train, holdout,
        node_embeddings, query_embeddings, None, features, device,
        epochs=int(args.epochs), batch_size=int(args.batch_size),
        learning_rate=float(args.learning_rate), weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    metrics, _rows, inference = _score_once(
        MODEL_NAME, model, validation,
        node_embeddings, query_embeddings, None, features, device,
        batch_size=int(args.batch_size), ks=tuple(args.ks), timed=True,
    )
    by_stratum = {}
    for name, subset in strata.items():
        if not subset:
            continue
        scores, _rows, _timing = _score_once(
            MODEL_NAME, model, subset,
            node_embeddings, query_embeddings, None, features, device,
            batch_size=int(args.batch_size), ks=tuple(args.ks), timed=False,
        )
        by_stratum[name] = {"queries": len(subset)} | {
            key: scores[key] for key in METRICS if key in scores
        }

    result["results"][NEW_ARM] = {
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "query_local_block": (
            "distance_0..3 (D3 geometry) + dense_reciprocal_rank, "
            "splade_reciprocal_rank, retriever_agreement (D4 prior)"
        ),
        "measured_in": "stage_d5",
        "retrained_here": True,
        "training": telemetry,
        "validation": {key: metrics[key] for key in METRICS if key in metrics},
        "validation_all_metrics": metrics,
        "validation_by_stratum": by_stratum,
        "inference": inference,
    }
    _atomic_json(args.output, result)
    if checkpoint_hook is not None:
        checkpoint_hook()
    del features, model

    if dataset.metadata["candidate_contract_sha256"] != contract_before:
        raise RuntimeError("Candidate contract changed while running a read-only experiment")

    counts = {arm: result["results"][arm]["parameters"] for arm in D5_ARMS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The table changed the parameter count: {counts}")

    ladder = {arm: result["results"][arm]["validation"] for arm in D5_ARMS}
    result["ladder"] = ladder
    result["increments"] = {
        name: {
            "from": lower,
            "to": upper,
            "measures": _INCREMENT_MEANING[name],
            **{key: float(ladder[upper][key] - ladder[lower][key]) for key in ladder[upper]},
        }
        for name, upper, lower in INCREMENTS
    }
    conditional, (plain_upper, plain_lower) = INTERACTION
    result["interaction"] = {
        "definition": "(Z2R - Z1R) - (Z2 - Z1)",
        "measures": (
            "whether seed-distance geometry is worth the same with the retrieval "
            "prior present as without it"
        ),
        "conditional": conditional,
        "unconditional": f"{plain_upper} - {plain_lower}",
        "descriptive_only": True,
        "descriptive_note": (
            "one dataset, one seed, one operating point; a sign, not a statistical "
            "interaction claim"
        ),
        **{
            key: float(
                result["increments"][conditional][key]
                - (ladder[plain_upper][key] - ladder[plain_lower][key])
            )
            for key in ladder[NEW_ARM]
        },
    }
    result["increments_by_stratum"] = {
        name: {
            stratum: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"][lower]["validation_by_stratum"][stratum][key]
                )
                for key in METRICS
                if key in block
            }
            for stratum, block in result["results"][upper]["validation_by_stratum"].items()
        }
        for name, upper, lower in INCREMENTS
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["arms_trained_here"] = [NEW_ARM]
    result["comparators"] = {
        "delta_seed_d3": d3["increments"]["delta_seed"],
        "delta_distance_d3": d3["increments"]["delta_distance"],
        "delta_remaining_structure_d3": d3["increments"]["delta_remaining_structure"],
        "delta_retrieval_quality_d4": d4["increments"]["delta_retrieval_quality"],
        "note": "the earlier increments, carried forward so D5's can be read against them",
    }
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage D5: retrieval prior and geometry")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--d3-result", type=Path, required=True)
    parser.add_argument("--d4-result", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default=None)
    # A3's frozen fusion constant, and the one D4 measured its prior at. D5 is a
    # composition experiment, so this is not a D5 choice.
    parser.add_argument("--rrf-constant", type=int, default=60)
    # The frozen QLS-v1 confirmation values, as D1-D4 default them. D5 reuses
    # four arms from those stages and must not disagree about the operating
    # point they were measured at.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--damping", type=float, default=0.85)
    parser.add_argument("--ppr-iterations", type=int, default=30)
    parser.add_argument("--static-pagerank-damping", type=float, default=0.85)
    parser.add_argument("--static-pagerank-iterations", type=int, default=30)
    parser.add_argument("--static-clustering-max-wedges", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
