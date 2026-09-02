"""Stage D4: given the seed bit, is graded retrieval evidence worth anything more?

D3 settled that QLS-v1's ten-column query-local block is, to first order, one
binary column: `distance_0`, proven elementwise equal to `I[d in Sq]` on
4,851,276 real candidate rows. `SEED_ID_ONLY` reached 96.1% of the full block's
R@5 and beat it outright on R@1 and MRR.

That bit is coarse. It says *whether* a candidate was in the dense-top-5 union
SPLADE-top-5 seed set and nothing else -- not where the retrievers ranked it,
not whether both retrievers found it, and nothing at all about the ~350
candidates outside the seed set. D4 asks the one question that follows:

    after the model already knows bare seed membership, does graded retrieval
    evidence provide substantial additional ranking value?

    Z1   SEED_ID_ONLY                only `I[d in Sq]`.        Reused from D3.
    Z1R  SEED_PLUS_GRADED_RETRIEVAL  that same bit, plus the frozen A3 retrieval
                                     prior for every candidate.        NEW.

    delta_retrieval_quality = Z1R - Z1

**The bit is preserved in both arms**, bit-identical, taken from the same
`build_local_features` call. D4 is not "graded evidence instead of the bit"; it
is "graded evidence on top of the bit". Replacing the bit would have confounded
*is graded evidence better than the bit?* with *what happens when the bit is
removed?*, and only the second of those is already answered.

The prior is **definition B, the full candidate retrieval prior**: dense and
SPLADE evidence for every candidate in `Cq`, not just for seeds. That is what
makes the arm `GRADED_RETRIEVAL_PRIOR` rather than "seed quality" -- most of its
information is about candidates the seed bit says nothing about. It is available
because `Cq` *is* the top-200 dense union SPLADE union while `Sq` is only the
top-5 union, so every candidate carries a rank from at least one retriever.

The columns are **A3's**, not new ones. `rank_feature_rows` is called with the
same arguments D0B calls it with, so the transform is the frozen reciprocal-rank
one, `(K+1)/(K + rank + 1)` at `K = 60`, which is universal: rank 0 is exactly
1.0 and an absent candidate is exactly 0.0 whatever the list length. Only the
agreement column is new, and it is a deterministic function of the other two.

Three currently-zeroed slots carry them, so `local_dim` stays 10, the head stays
61 wide and both arms are 213,506 parameters. Same three-epoch frozen QLS-v1
budget, same seed, same split, same initialisation: D4 must not give its new arm
a convergence advantage over the control it reuses.

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
    RANK_FEATURE_NAMES,
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
    SEED_ID_COLUMN,
    assert_seed_identity_column,
    masked_local,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from scripts.run_sa_mlp_confirmation import (  # noqa: E402
    _build_model,
    _fit,
    _score_once,
    validate_candidate_contract,
)

COMPLETE_STATUS = "GRAPH_CONTEXT_D4_COMPLETE"
IN_PROGRESS_STATUS = "GRAPH_CONTEXT_D4_IN_PROGRESS"

#: The three zeroed slots the retrieval prior is carried in. They are slots, not
#: distance buckets: in `SEED_ID_ONLY` these columns are exactly zero, so filling
#: them adds information without touching the seed bit, the architecture or the
#: parameter count. Every result field names them by content, never by the
#: historical `LOCAL_FEATURE_NAMES` entry that occupies the same index.
DENSE_RANK_COLUMN = 1
SPLADE_RANK_COLUMN = 2
AGREEMENT_COLUMN = 3
RETRIEVAL_COLUMNS = (DENSE_RANK_COLUMN, SPLADE_RANK_COLUMN, AGREEMENT_COLUMN)

#: Slot index -> what D4 actually puts there. `distance_0` keeps its historical
#: meaning; the other three are overwritten.
COLUMN_SEMANTICS: dict[int, str] = {
    SEED_ID_COLUMN: "distance_0, unchanged: exactly I[d in Sq]",
    DENSE_RANK_COLUMN: "dense_reciprocal_rank (A3), overwrites the zeroed distance_1 slot",
    SPLADE_RANK_COLUMN: "splade_reciprocal_rank (A3), overwrites the zeroed distance_2 slot",
    AGREEMENT_COLUMN: "retriever_agreement (new), overwrites the zeroed distance_3 slot",
}

AGREEMENT_FEATURE_NAME = "retriever_agreement"

D4_ARMS = ("SEED_ID_ONLY", "SEED_PLUS_GRADED_RETRIEVAL")
NEW_ARM = "SEED_PLUS_GRADED_RETRIEVAL"
REUSED_FROM_D3: dict[str, str] = {"SEED_ID_ONLY": "SEED_ID_ONLY"}

CONTEXT = "CAND"
HOLDOUT_FRACTION = 0.1


def a3_rank_feature_audit(constant: int, list_length: int) -> dict[str, Any]:
    """What A3's two rank features are, checked rather than quoted from memory.

    The requirement on any retrieval feature entering D4 is that it be universal
    -- that its value not depend on how long the ranked list happened to be, or
    on how many candidates a query drew. A raw rank fails that; a reciprocal rank
    with a fixed constant does not. Rather than assert it, this evaluates the
    shipped function on a synthetic list and reports what came back.
    """
    candidates = np.arange(list_length, dtype=np.int64)
    values = rank_feature_rows(candidates, candidates[::-1].copy(), candidates, constant=constant)
    top = float(values[0, 0])
    tail = float(values[list_length - 1, 0])
    if top != 1.0:
        raise RuntimeError(f"A3's rank transform no longer maps rank 0 to 1.0 (got {top})")

    # Halving the list must not move any value: this is the universality claim.
    half = list_length // 2
    short = rank_feature_rows(
        candidates[:half], candidates[:half][::-1].copy(), candidates[:half], constant=constant
    )
    if not np.array_equal(short[:, 0], values[:half, 0]):
        raise RuntimeError("A3's rank transform depends on the length of the ranked list")

    # And distinct ranks must survive the float16 cast the local block uses, or
    # the graded column would silently collapse into a coarser one.
    exact = np.asarray(
        [float(constant + 1) / (constant + rank + 1) for rank in range(list_length)],
        dtype=np.float32,
    )
    if np.unique(exact.astype(np.float16)).size != list_length:
        raise RuntimeError("Distinct ranks collide once stored at the local block's float16")

    return {
        "source": "mp_retrieval.linear_control.rank_feature_rows",
        "also_used_by": "scripts/run_graph_context_d0b.py, the A3 linear controls",
        "feature_names": list(RANK_FEATURE_NAMES),
        "formula": "(K + 1) / (K + zero_based_rank + 1)",
        "constant_K": int(constant),
        "missing_retriever_handling": "exactly 0.0 for a candidate absent from that ranking",
        "normalisation": (
            "universal: rank 0 is exactly 1.0 and rank r is the same value whatever "
            "the list length or pool size, so no pool-dependent scale enters"
        ),
        "rank_0_value": top,
        "worst_rank": list_length - 1,
        "worst_rank_value": tail,
        "invariant_to_list_length": True,
        "distinct_ranks_survive_float16": True,
        "seeds_special_cased": False,
        "seeds_note": (
            "the value depends only on rank position; a seed is not treated "
            "differently from any other candidate, which is what makes this a "
            "prior over all of Cq rather than a seed-quality score"
        ),
        "reused_verbatim": True,
    }


def build_graded_retrieval_block(
    queries, local: np.ndarray, candidate_ptr: np.ndarray, dense, splade, *, constant: int
) -> tuple[np.ndarray, dict[str, Any]]:
    """`Z1R`'s local block: the bare seed bit, plus A3's prior over all of `Cq`.

    Built from the same `local` array `SEED_ID_ONLY` is masked out of, so the
    seed column the two arms see is the same array element, not a recomputation
    of it. Everything outside the four named slots stays exactly zero.
    """
    if local.shape[1] != len(LOCAL_FEATURE_NAMES):
        raise RuntimeError(f"Local block has {local.shape[1]} columns, expected 10")
    if set(RETRIEVAL_COLUMNS) & set(NORMALISED_COLUMNS):
        raise RuntimeError(
            "A retrieval column landed in a slot `candidate_readout` rescales; the "
            "stored values would no longer be A3's transform"
        )

    block = np.zeros_like(local)
    block[:, SEED_ID_COLUMN] = local[:, SEED_ID_COLUMN]

    covered = 0
    agreed = 0
    dense_only = 0
    splade_only = 0
    for position, query in enumerate(queries):
        start = int(candidate_ptr[position])
        end = int(candidate_ptr[position + 1])
        candidates = query.candidate_index.numpy().astype(np.int64, copy=False)
        if candidates.size != end - start:
            raise RuntimeError(
                f"Query {position} spans {end - start} rows but has {candidates.size} candidates"
            )
        rows = rank_feature_rows(
            np.asarray(dense[query.query_index]),
            np.asarray(splade[query.query_index]),
            candidates,
            constant=constant,
        )
        in_dense = rows[:, 0] > 0.0
        in_splade = rows[:, 1] > 0.0
        if not np.all(in_dense | in_splade):
            raise RuntimeError(
                f"Query {position} has a candidate ranked by neither retriever; the "
                "graded prior would not be defined for every candidate in Cq"
            )
        both = in_dense & in_splade
        block[start:end, DENSE_RANK_COLUMN] = rows[:, 0]
        block[start:end, SPLADE_RANK_COLUMN] = rows[:, 1]
        block[start:end, AGREEMENT_COLUMN] = both.astype(np.float32)
        covered += int(candidates.size)
        agreed += int(both.sum())
        dense_only += int(np.count_nonzero(in_dense & ~in_splade))
        splade_only += int(np.count_nonzero(in_splade & ~in_dense))

    if covered != int(candidate_ptr[-1]):
        raise RuntimeError("The graded prior does not cover every candidate row")

    provenance = {
        "definition": "B, the full candidate retrieval prior",
        "why_b": (
            "graded evidence is defined for every candidate in Cq, not only for "
            "seeds, so the arm measures a retrieval prior rather than a seed-quality "
            "score; Cq is the top-200 dense union SPLADE union while Sq is the top-5 "
            "union, so no candidate is left without a rank"
        ),
        "not_called_seed_quality": True,
        "columns": {
            str(index): COLUMN_SEMANTICS[index]
            for index in (SEED_ID_COLUMN, *RETRIEVAL_COLUMNS)
        },
        "agreement_definition": (
            "1.0 if the candidate is ranked by both retrievers, else 0.0; a "
            "deterministic function of the two A3 columns, each strictly positive "
            "exactly when that retriever ranked the candidate"
        ),
        "agreement_is_new_not_from_a3": True,
        "never_rescaled_by_candidate_readout": True,
        "candidate_rows": covered,
        "rows_ranked_by_both": agreed,
        "rows_dense_only": dense_only,
        "rows_splade_only": splade_only,
        "rows_ranked_by_neither": 0,
        "every_candidate_has_graded_evidence": True,
        "zeroed_columns": [
            LOCAL_FEATURE_NAMES[index]
            for index in range(len(LOCAL_FEATURE_NAMES))
            if index not in (SEED_ID_COLUMN, *RETRIEVAL_COLUMNS)
        ],
    }
    return block, provenance


def assert_seed_bit_is_shared(control: np.ndarray, treatment: np.ndarray) -> dict[str, Any]:
    """Both arms must carry the same bit, and only the treatment may carry more.

    The whole causal reading of `delta_retrieval_quality` is that one thing
    changed. If the seed column differed between the arms by even one row, the
    increment would be measuring the graded prior *and* a perturbed bit.
    """
    if control.shape != treatment.shape:
        raise RuntimeError("The two arms' local blocks differ in shape")
    if not np.array_equal(control[:, SEED_ID_COLUMN], treatment[:, SEED_ID_COLUMN]):
        raise RuntimeError("The seed bit is not identical across the two arms")
    if control[:, 1:].any():
        raise RuntimeError("The control arm carries more than the bare seed bit")
    untouched = [
        index for index in range(len(LOCAL_FEATURE_NAMES))
        if index not in (SEED_ID_COLUMN, *RETRIEVAL_COLUMNS)
    ]
    if treatment[:, untouched].any():
        raise RuntimeError("The treatment arm carries structure outside its declared slots")
    if not treatment[:, list(RETRIEVAL_COLUMNS)].any():
        raise RuntimeError("The treatment arm's retrieval columns are empty")
    return {
        "seed_bit_identical_across_arms": True,
        "seed_rows": int(np.count_nonzero(control[:, SEED_ID_COLUMN])),
        "control_carries_only_the_bit": True,
        "treatment_adds_only": [COLUMN_SEMANTICS[index] for index in RETRIEVAL_COLUMNS],
        "columns_zero_in_both": [LOCAL_FEATURE_NAMES[index] for index in untouched],
    }


def verify_reuse(d3: dict[str, Any], result: dict[str, Any], args) -> dict[str, Any]:
    """Refuse to reuse D3's `SEED_ID_ONLY` across any condition that could matter.

    Reuse is the whole reason D4 costs one run instead of two, but it is only
    legitimate if the reused row is the same experiment under a different name.
    Each check names a way the two stages could silently diverge; a mismatch
    raises rather than degrading into a comparison of different worlds.
    """
    if d3.get("status") != "GRAPH_CONTEXT_D3_COMPLETE":
        raise RuntimeError(f"D3 result is not complete: {d3.get('status')!r}")

    control = d3.get("results", {}).get("SEED_ID_ONLY", {})
    proof = d3.get("seed_identity_proof", {})
    ours = result["seed_identity_proof"]

    checks: dict[str, Any] = {}
    for label, mine, theirs in (
        ("dataset", args.dataset, d3.get("dataset")),
        ("data_fingerprint", args.data_fingerprint_sha256, d3.get("data_fingerprint_sha256")),
        ("context", CONTEXT, d3.get("context")),
        ("model", MODEL_NAME, d3.get("model")),
        ("num_nodes", result["num_nodes"], d3.get("num_nodes")),
        (
            "candidate_contract",
            result["candidate_contract"].get("observed_contract_sha256"),
            d3.get("candidate_contract", {}).get("observed_contract_sha256"),
        ),
        ("local_feature_schema", list(LOCAL_FEATURE_NAMES), d3.get("local_feature_schema")),
        ("candidate_rows", result["feature_build"]["candidate_rows"],
         d3.get("feature_build", {}).get("candidate_rows")),
        ("static_feature_source", result["static_features"].get("source"),
         d3.get("static_features", {}).get("source")),
        ("architecture_changed", False, d3.get("ablation", {}).get("architecture_changed")),
        ("same_epoch_budget_for_every_arm", True,
         d3.get("ablation", {}).get("same_epoch_budget_for_every_arm")),
        # The reused row must be the bare-bit arm, and it must have been trained
        # in D3 rather than itself reused from somewhere further back.
        ("control_columns_kept", ["distance_0"],
         d3.get("arms", {}).get("SEED_ID_ONLY", {}).get("columns_kept")),
        ("control_retrained_in_d3", True, control.get("retrained_here")),
        # And the bit it was trained on must be the bit measured here.
        ("seed_identity_column", ours["column"], proof.get("column")),
        ("seed_identity_rows", ours["candidate_rows_compared"],
         proof.get("candidate_rows_compared")),
        ("seed_identity_seed_rows", ours["seed_rows"], proof.get("seed_rows")),
        ("seed_identity_mismatches", 0, proof.get("mismatches")),
    ):
        checks[label] = {"d4": mine, "d3": theirs, "match": mine == theirs}

    for key in ("train_fit", "train_holdout_for_epoch_selection", "validation_reported"):
        checks[f"split_{key}"] = {
            "d4": result["splits"][key],
            "d3": d3.get("splits", {}).get(key),
            "match": result["splits"][key] == d3.get("splits", {}).get(key),
        }

    for key in ("epochs", "batch_size", "learning_rate", "weight_decay", "seed"):
        mine = result["training"][key]
        theirs = d3.get("training", {}).get(key)
        checks[f"training_{key}"] = {"d4": mine, "d3": theirs, "match": mine == theirs}

    failed = sorted(name for name, block in checks.items() if not block["match"])
    if failed:
        raise RuntimeError(
            "D3's SEED_ID_ONLY cannot be reused; these conditions differ: "
            + ", ".join(f"{name} ({checks[name]['d4']!r} vs {checks[name]['d3']!r})"
                        for name in failed)
        )
    return {
        "reused_arms": dict(REUSED_FROM_D3),
        "source": "stage_d3.json",
        "why": (
            "Z1 is D3's SEED_ID_ONLY under identical conditions. Retraining it for "
            "cosmetic symmetry would pay twice for the same numbers and would give "
            "one of the two arms a different initialisation."
        ),
        "conditions_checked": checks,
        "all_conditions_match": True,
        "new_runs": 1,
    }


def run(args: argparse.Namespace, checkpoint_hook: Callable[[], None] | None = None):
    if "test" in args.splits:
        raise ValueError("Stage D4 is a development experiment; the test split is not read")
    if sorted(args.splits) != ["train", "validation"]:
        raise ValueError("Stage D4 fits on train and reports on validation; both and only both")

    d3_path = Path(args.d3_result)
    if not d3_path.is_file():
        raise FileNotFoundError(f"Stage D4 reuses D3's SEED_ID_ONLY arm but {d3_path} is absent")
    d3 = json.loads(d3_path.read_text(encoding="utf-8"))

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
        raise RuntimeError("Stage D4 requires non-empty train and validation splits")
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
    # The same call D3 made, so `Z1R`'s seed column is the array element `Z1`
    # was trained on rather than a recomputation that happens to agree.
    local, candidate_ptr = build_local_features(
        opened, rowptr, col, size, operators, CONTEXT,
        damping=float(args.damping), ppr_iterations=int(args.ppr_iterations),
        latency=latency,
    )
    build_seconds = time.perf_counter() - started

    seed_identity = assert_seed_identity_column(opened, local)
    rank_audit = a3_rank_feature_audit(int(args.rrf_constant), int(dense.shape[1]))

    started = time.perf_counter()
    treatment, prior = build_graded_retrieval_block(
        opened, local, candidate_ptr, dense, splade, constant=int(args.rrf_constant)
    )
    prior_seconds = time.perf_counter() - started
    control = masked_local(local, (SEED_ID_COLUMN,))
    shared_bit = assert_seed_bit_is_shared(control, treatment)

    result: dict[str, Any] = {
        "status": IN_PROGRESS_STATUS,
        "dataset": args.dataset,
        "stage": "stage_d4",
        "question": (
            "after the model already knows bare seed membership, does graded "
            "retrieval evidence provide substantial additional ranking value?"
        ),
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "num_nodes": size,
        "model": MODEL_NAME,
        "context": CONTEXT,
        "local_feature_schema": list(LOCAL_FEATURE_NAMES),
        "column_semantics": {str(key): value for key, value in COLUMN_SEMANTICS.items()},
        "arms": {
            "SEED_ID_ONLY": {
                "carries": ["distance_0 == I[d in Sq]"],
                "measured_in": "stage_d3",
            },
            NEW_ARM: {
                "carries": ["distance_0 == I[d in Sq]"]
                + [COLUMN_SEMANTICS[index] for index in RETRIEVAL_COLUMNS],
                "measured_in": "stage_d4",
            },
        },
        "seed_identity_proof": seed_identity,
        "seed_bit_preserved": shared_bit,
        "a3_rank_feature_audit": rank_audit,
        "graded_retrieval_prior": prior | {"seconds": round(prior_seconds, 1)},
        "ablation": {
            "method": "filling zeroed slots, not widening the block",
            "why": (
                "the retrieval prior occupies three columns that are exactly zero in "
                "SEED_ID_ONLY, so local_dim, head width and parameter count are "
                "identical and the increment measures information, not capacity"
            ),
            "architecture_changed": False,
            "same_epoch_budget_for_every_arm": True,
            "seed_bit_replaced": False,
            "seed_bit_replaced_note": (
                "the bare bit is preserved in both arms; replacing it would confound "
                "'is graded evidence better than the bit' with 'what happens when the "
                "bit is removed'"
            ),
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
    result["reuse"] = verify_reuse(d3, result, args)
    _atomic_json(args.output, result)

    for arm, source_name in REUSED_FROM_D3.items():
        source = d3["results"][source_name]
        result["results"][arm] = dict(source) | {
            "measured_in": "stage_d3",
            "measured_as": source_name,
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
            "seed bit + dense_reciprocal_rank + splade_reciprocal_rank + retriever_agreement"
        ),
        "measured_in": "stage_d4",
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

    counts = {arm: result["results"][arm]["parameters"] for arm in D4_ARMS}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f"The comparison changed the parameter count: {counts}")

    ladder = {arm: result["results"][arm]["validation"] for arm in D4_ARMS}
    result["ladder"] = ladder
    result["increments"] = {
        "delta_retrieval_quality": {
            "from": "SEED_ID_ONLY",
            "to": NEW_ARM,
            "measures": (
                "incremental value of graded dense and SPLADE evidence over every "
                "candidate, given bare retrieval-seed membership"
            ),
            **{
                key: float(ladder[NEW_ARM][key] - ladder["SEED_ID_ONLY"][key])
                for key in ladder[NEW_ARM]
            },
        }
    }
    result["increments_by_stratum"] = {
        "delta_retrieval_quality": {
            stratum: {"queries": block["queries"]}
            | {
                key: float(
                    block[key]
                    - result["results"]["SEED_ID_ONLY"]["validation_by_stratum"][stratum][key]
                )
                for key in METRICS
                if key in block
            }
            for stratum, block in result["results"][NEW_ARM]["validation_by_stratum"].items()
        }
    }
    result["gold_stratum_counts"] = {
        "validation": {name: len(subset) for name, subset in strata.items() if subset}
    }
    result["arms_trained_here"] = [NEW_ARM]
    result["comparators"] = {
        "delta_seed_d3": d3["increments"]["delta_seed"],
        "delta_distance_d3": d3["increments"]["delta_distance"],
        "delta_remaining_structure_d3": d3["increments"]["delta_remaining_structure"],
        "note": "D3's increments, carried forward so D4's size can be read against them",
    }
    result["status"] = COMPLETE_STATUS
    _atomic_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage D4: the graded retrieval prior")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--d3-result", type=Path, required=True)
    parser.add_argument("--dataset", default="2wiki_clean")
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--baseline", type=json.loads, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--holdout-fraction", type=float, default=HOLDOUT_FRACTION)
    parser.add_argument("--selected-gnn", default=None)
    # A3's frozen fusion constant. Not a D4 choice: changing it would mean the
    # stage was no longer reusing A3's validated transform.
    parser.add_argument("--rrf-constant", type=int, default=60)
    # The frozen QLS-v1 confirmation values, as D1, D2 and D3 default them. D4
    # reuses a D3 arm as its control, so it must not disagree about the
    # operating point that control was measured at.
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
