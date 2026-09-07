#!/usr/bin/env python
"""M2C Stage-0 probe: does graph direction carry RANKING signal for S4's mistakes?

Trains nothing. Reads no test split. Fits no parameter, sweeps no weight, and
introduces no tunable constant -- the one constant it uses, the RRF 60, is the
project's own, declared in ``configs/candidate_budget.yaml`` long before M2C
existed.

What it measures, per declared cell:

  1. the residual x provenance matrix -- R0/R1/R2 crossed with
     G_STRUCT/G_KNN/G_FULL, under three arms: S4 alone, direction alone, and
     their reciprocal-rank fusion;
  2. the error-conditioned margin, which is the load-bearing one: on the
     queries S4 gets wrong, does the direction prefer a relevant candidate over
     the wrong one S4 currently ranks first?
  3. the minimal +64 admission diagnostic: A64 against the two parameter-free
     residual admissions at the same budget.

Everything except the directional calculation is held byte-identical between
arms: the scored candidates, S4's existing structural features, the S4
checkpoint and the candidate normalization all come from the sealed artifacts
and are loaded, never rebuilt. If the directional calculation is not the only
thing that changes, the comparison against M0A's null is not a comparison.

The panel is the development (fit) portion of each cell's validation split --
the same ``holdout_split`` division M2B made, so M2B's held-out 20% stays
unexamined and remains a clean surface for any later stage. The dataset test
split is never opened by either.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

import scripts.run_m1a_feature_screen as _m1a  # noqa: E402
import scripts.run_m2b_semantic_minimality as _m2b  # noqa: E402
from mp_retrieval import m2c_structural_offset as offset  # noqa: E402
from mp_retrieval.candidate_expansion_v2 import (  # noqa: E402
    STRUCTURAL,
    ExpansionBudget,
    expand,
    query_residual,
)
from mp_retrieval.candidate_headroom import ragged_from_rows  # noqa: E402
from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.edge_provenance import graph_payload  # noqa: E402
from mp_retrieval.headroom_v2 import regime_headroom  # noqa: E402
from mp_retrieval.l2_data import edge_index_to_csr  # noqa: E402

COMPLETE_STATUS = "M2C_STAGE0_COMPLETE"
IN_PROGRESS_STATUS = "M2C_STAGE0_IN_PROGRESS"
KS = (1, 5, 20)

#: The declared provenance views, and the families they resolve to. Audited
#: against src/mp_retrieval/edge_provenance.py rather than reconstructed:
#: knn_only = baseline_a_simple MINUS structural_only, and full_union_c adds
#: NER on top, which is why G_FULL is baseline_a_simple and NOT full_union_c.
PROVENANCE_FAMILIES = {
    "G_STRUCT": "structural_only",
    "G_KNN": "knn_only",
    "G_FULL": "baseline_a_simple",
}
RESIDUALS = ("R0_RAW_QUERY_CONTROL", "R1_LEGACY_DIRECTIONAL", "R2_SEED_SUBSPACE")
ARMS = ("S4", "direction_only", "S4_plus_direction_rrf")

#: Which aggregate the ranking arms use. dir_mean is reported beside it but is
#: not the ranking signal; declaring one before results is what keeps this from
#: becoming a two-way choice made after seeing which won.
RANKING_SIGNAL = "dir_max"


def _metrics(ranks: np.ndarray, relevant: np.ndarray) -> dict[str, float]:
    """recall@k and MRR for one query, from ranks over the scored pool."""

    if not relevant.any():
        return {**{f"recall@{k}": 0.0 for k in KS}, "mrr": 0.0, "scored": 0.0}
    relevant_ranks = ranks[relevant]
    best = int(relevant_ranks.min())
    total = int(relevant.sum())
    return {
        **{
            f"recall@{k}": float((relevant_ranks <= k).sum()) / total
            for k in KS
        },
        "mrr": 1.0 / best,
        "scored": 1.0,
    }


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {**{f"recall@{k}": 0.0 for k in KS}, "mrr": 0.0, "queries": 0}
    return {
        **{
            f"recall@{k}": float(np.mean([r[f"recall@{k}"] for r in rows])) for k in KS
        },
        "mrr": float(np.mean([r["mrr"] for r in rows])),
        "queries": len(rows),
    }


def _load_family_csr(path: Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """One provenance family, symmetrised before anything reads it.

    Symmetrised because M0A and M1A both symmetrise these graphs before use,
    and for M0A's stated reason: the family graphs are stored asymmetric, and
    choosing an orientation is a choice a probe is not entitled to make. It
    matters twice here. The +64 admission arms have to expand over the same
    adjacency A64 expanded over or they are not budget-matched to it; and the
    directional scores have to be computed over the same adjacency M0A's null
    was measured on, or the comparison against that null compares two graphs
    as much as two mechanisms.

    ``_m1a._undirected`` rather than a local copy: it is the exact function M2
    applied to this family when it built the R3 cell masters this probe reads.
    """

    edge_index, stored_nodes = graph_payload(path)
    if int(stored_nodes) != int(num_nodes):
        raise ValueError(f"{path} declares {stored_nodes} nodes, dataset has {num_nodes}")
    rowptr, col, _ = edge_index_to_csr(torch.from_numpy(edge_index), num_nodes)
    rowptr, col, was_symmetric = _m1a._undirected(
        np.asarray(rowptr, dtype=np.int64), np.asarray(col, dtype=np.int64), num_nodes
    )
    return rowptr, col, bool(was_symmetric)


def _residual_vectors(
    query_vector: np.ndarray,
    seed_rows: np.ndarray,
    anchor_row: np.ndarray,
) -> dict[str, dict[str, Any]]:
    """The three controls, built from embeddings and seeds only.

    R1 CALLS ``candidate_expansion_v2.query_residual`` -- the exact function
    M0A ran. Reimplementing it from its description would produce an arm that
    is not the thing M0A nulled, and the comparison would be against a
    lookalike.
    """

    raw = offset.raw_query_direction(query_vector)
    legacy = query_residual(query_vector, anchor_row)
    subspace = offset.seed_subspace_residual(query_vector, seed_rows)
    legacy_norm = float(np.linalg.norm(legacy))
    return {
        "R0_RAW_QUERY_CONTROL": {
            "vector": raw.vector,
            "degenerate": bool(raw.degenerate),
            "fell_back_to_query": False,
        },
        "R1_LEGACY_DIRECTIONAL": {
            "vector": legacy,
            "degenerate": bool(legacy_norm <= 0.0),
            "fell_back_to_query": False,
        },
        "R2_SEED_SUBSPACE": {
            "vector": subspace.vector,
            "degenerate": bool(subspace.degenerate),
            "fell_back_to_query": bool(subspace.fell_back_to_query),
        },
    }


#: The three arms of the +64 admission diagnostic. A64 is the blind control R3
#: already uses, so 64 is the only budget at which this comparison is
#: budget-matched rather than budget-buying.
ADMISSION_ARMS = ("A64", "legacy_PF64", "seed_subspace_PF64")


def _admission_diagnostic(
    panel: list[Any],
    *,
    node_array: np.ndarray,
    query_array: np.ndarray,
    rowptr: np.ndarray,
    col: np.ndarray,
    num_nodes: int,
    budget: ExpansionBudget,
    cap: int,
    mainline_family: str,
    stored_was_symmetric: bool,
) -> dict[str, Any]:
    """A64 against the two parameter-free residual admissions, at one budget.

    Reports set overlap FIRST and the ceiling second, because overlap is what
    the evidence actually turns on: M0A's null was that the directional method
    and its blind control admitted sets whose difference moved no ceiling. If
    the arms here select the same nodes, that null transfers directly and no
    ceiling arithmetic is needed to know it.

    The ceiling reported is ``recall_ceiling@5`` -- K-aware, the quantity that
    actually bounds recall@5 and the denominator M2B itself used. NOT
    ``candidate_ceiling``, which is pool coverage with no K and which diverges
    from it materially on the KB graphs.
    """

    subset = panel[:cap] if cap else panel
    pools: dict[str, list[np.ndarray]] = {arm: [] for arm in ADMISSION_ARMS}
    admitted: dict[str, list[np.ndarray]] = {arm: [] for arm in ADMISSION_ARMS}
    relevant_admissions: dict[str, int] = dict.fromkeys(ADMISSION_ARMS, 0)
    golds_rows: list[np.ndarray] = []
    degenerate = dict.fromkeys(ADMISSION_ARMS, 0)

    for query in subset:
        pool = query.candidate_index.numpy().astype(np.int64, copy=False)
        golds = query.relevant_global.numpy().astype(np.int64, copy=False)
        golds_rows.append(golds)
        seed_local = query.retrieval_seed_local
        seeds = (
            np.unique(pool[seed_local.numpy().astype(np.int64, copy=False)])
            if seed_local is not None and seed_local.numel()
            else pool[:0]
        )
        anchor = int(query.anchor_global)
        query_vector = query_array[query.query_index]

        results = {
            "A64": expand(
                STRUCTURAL,
                rowptr=rowptr, col=col, node_embeddings=node_array,
                query_embedding=None, anchor=anchor, pool=pool, seeds=seeds,
                budget=budget, num_nodes=num_nodes,
            ),
            "legacy_PF64": offset.expand_with_residual(
                query_residual(query_vector, node_array[anchor]),
                rowptr=rowptr, col=col, node_embeddings=node_array,
                anchor=anchor, pool=pool, seeds=seeds, budget=budget,
                num_nodes=num_nodes,
            ),
            "seed_subspace_PF64": offset.expand_with_residual(
                offset.seed_subspace_residual(query_vector, node_array[seeds]).vector,
                rowptr=rowptr, col=col, node_embeddings=node_array,
                anchor=anchor, pool=pool, seeds=seeds, budget=budget,
                num_nodes=num_nodes,
            ),
        }
        for arm, expansion in results.items():
            pools[arm].append(expansion.matched_pool)
            admitted[arm].append(expansion.admitted)
            relevant_admissions[arm] += int(np.isin(expansion.admitted, golds).sum())
            degenerate[arm] += int(expansion.degenerate_residual)

    golds = ragged_from_rows(golds_rows)
    ceilings = {
        arm: regime_headroom(
            ragged_from_rows(rows), golds, num_nodes=num_nodes, ks=KS
        )[0]
        for arm, rows in pools.items()
    }
    baseline = ceilings["A64"]["recall_ceiling@5"]
    overlaps = {
        f"{left}|{right}": offset.admission_overlap(
            np.concatenate(admitted[left]) if admitted[left] else np.zeros(0, dtype=np.int64),
            np.concatenate(admitted[right]) if admitted[right] else np.zeros(0, dtype=np.int64),
        )
        for left, right in (
            ("A64", "legacy_PF64"),
            ("A64", "seed_subspace_PF64"),
            ("legacy_PF64", "seed_subspace_PF64"),
        )
    }
    return {
        "queries": len(subset),
        "budget": budget.graph_expansion_cap,
        # Which graph all three arms expanded over, stated rather than implied.
        # A64 is the incumbent blind admission and it is defined on ONE graph:
        # the symmetrised mainline provenance family, which is also the family
        # M2 recorded in the R3 cell master's build key. Running the residual
        # arms on anything else would make them differ from A64 in two things
        # at once -- the scoring rule and the adjacency -- and the diagnostic
        # exists to isolate the first.
        "graph": {
            "family": mainline_family,
            "edges": int(col.size),
            "symmetrised": True,
            "stored_already_symmetric": bool(stored_was_symmetric),
            "why": (
                "the family A64 is defined on, symmetrised the way M0A and M1A "
                "symmetrise it, so the arms differ only in how the frontier is scored"
            ),
        },
        "why_64_only": (
            "A64 is the incumbent blind admission at exactly this budget, so 64 "
            "is the only value at which the comparison is budget-matched rather "
            "than budget-buying."
        ),
        "arms": list(ADMISSION_ARMS),
        "pairwise_overlap": overlaps,
        "unique_admitted_nodes": {
            arm: int(np.unique(np.concatenate(rows)).size) if rows and rows[0].size else 0
            for arm, rows in admitted.items()
        },
        "unique_relevant_admissions": relevant_admissions,
        "degenerate_residual_queries": degenerate,
        "recall_ceiling_at_5": {
            arm: float(metrics["recall_ceiling@5"]) for arm, metrics in ceilings.items()
        },
        "delta_recall_ceiling_at_5_pp_versus_a64": {
            arm: float(metrics["recall_ceiling@5"] - baseline) * 100
            for arm, metrics in ceilings.items()
        },
        "candidate_ceiling_is_not_used_as_the_r5_bound": (
            "recall_ceiling@5 is K-aware and is what bounds recall@5. "
            "candidate_ceiling is macro pool coverage with no K; the two agree on "
            "the passage sets and diverge on metaqa and webqsp, so substituting "
            "one for the other invents headroom."
        ),
    }


def _baseline_block(baseline: Any) -> dict[str, Any]:
    """The frozen baseline block, however the caller happens to hold it.

    A launcher passes the dict it already read from the SA-MLP confirmation
    artifact, as M0A's and M2B's launchers do; a CLI run passes the path to that
    artifact. Accepting both keeps the launcher on the established convention
    without making the command line carry a JSON blob.
    """

    if isinstance(baseline, dict):
        return baseline
    path = Path(baseline)
    confirmation = json.loads(path.read_text(encoding="utf-8"))
    if confirmation.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
        raise ValueError(f"{path} is not a complete confirmation artifact")
    return confirmation["baseline"]


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.output.is_file():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if existing.get("status") == COMPLETE_STATUS:
            return existing

    started = time.perf_counter()
    dataset = load_complete_dataset(args.data, dataset=args.dataset)
    if dataset.node_array is None or dataset.query_array is None:
        raise ValueError(
            f"{args.data} was opened topology-only. The residual and the "
            "displacements are both functions of embeddings, so this probe "
            "cannot run without them."
        )
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")

    candidate_contract = _m2b.validate_candidate_contract(
        _baseline_block(args.baseline), dataset, args.candidate_contract_compatibility
    )
    validation = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(validation) < args.queries:
        raise ValueError(
            f"validation split holds {len(validation)} queries, needed {args.queries}"
        )
    if any(query.split == int(QuerySplit.TEST) for query in validation):
        raise RuntimeError("a test-split query reached the Stage-0 panel; refusing")

    # The sealed cell master: scored sets and the structural feature block M2
    # built. Loaded under its contract, never rebuilt -- a rebuilt block would
    # not be the input S4 was fit on.
    scored_sets, master_blocks, _latency, _fp, _decision = _m2b.load_cell_under_contract(
        args.cell_features, args, args.regime, candidate_contract=candidate_contract
    )
    widened = [
        _m1a._widen_query(query, scored)
        for query, scored in zip(validation, scored_sets, strict=True)
    ]
    # The SAME division M2B made. The development portion is what Stage 0 reads;
    # M2B's holdout is left untouched so a later stage can still report on it.
    panel, holdout = _m2b.holdout_split(widened, args.holdout_fraction)
    if args.panel_cap:
        panel = panel[: args.panel_cap]

    store, precomputed_width = _m1a._arm_store(
        arm=_m2b.RUNNER_UNIVERSAL_ARM,
        regime=args.regime,
        master_blocks=master_blocks,
        queries=widened,
        # The whole dataset's query count, not the panel's. The store is
        # indexed by ``query.query_index``, which is a position in the dataset's
        # query array and not in the split -- sizing it to the split makes every
        # query past the split's length an IndexError.
        query_count=len(dataset.queries),
        num_nodes=int(dataset.num_nodes),
    )

    device = torch.device("cpu")
    node_embeddings = torch.from_numpy(np.asarray(dataset.node_array)).to(device).float()
    query_embeddings = torch.from_numpy(np.asarray(dataset.query_array)).to(device).float()
    model = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung="S4",
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=int(node_embeddings.shape[1]),
        semantic_head=_m2b.build_semantic_head("S4", int(node_embeddings.shape[1])),
    ).to(device)
    model.load_state_dict(
        torch.load(args.s4_checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()

    loaded = {
        view: _load_family_csr(
            args.edge_provenance_root / family / "graph.pt", int(dataset.num_nodes)
        )
        for view, family in PROVENANCE_FAMILIES.items()
    }
    families = {view: (rowptr, col) for view, (rowptr, col, _sym) in loaded.items()}
    stored_symmetry = {
        PROVENANCE_FAMILIES[view]: sym for view, (_r, _c, sym) in loaded.items()
    }

    # A64's own graph. It is the mainline provenance family -- the same value
    # M2 recorded in this cell's build key -- and it must be loaded here even
    # though G_STRUCT happens to name the same family today, because the
    # diagnostic's correctness depends on the A64 identity and not on that
    # coincidence continuing to hold.
    if args.a64_mainline_family not in stored_symmetry:
        raise ValueError(
            f"the mainline family {args.a64_mainline_family!r} is not among the loaded "
            f"provenance views {sorted(stored_symmetry)}; the +64 diagnostic cannot "
            "expand over a graph it did not load"
        )
    mainline_view = next(
        view
        for view, family in PROVENANCE_FAMILIES.items()
        if family == args.a64_mainline_family
    )
    a64_rowptr, a64_col = families[mainline_view]

    node_array = np.asarray(dataset.node_array, dtype=np.float64)
    query_array = np.asarray(dataset.query_array, dtype=np.float64)

    # (arm, residual, provenance) -> per-query metric rows.
    collected: dict[tuple[str, str, str], list[dict[str, float]]] = {}
    margins: dict[tuple[str, str], list[offset.ErrorMargin]] = {}
    coverage: dict[tuple[str, str], list[float]] = {}
    residual_states: dict[str, dict[str, int]] = {
        name: {"degenerate": 0, "fell_back_to_query": 0} for name in RESIDUALS
    }
    excluded = {"no_relevant_in_pool": 0, "s4_top1_already_right": 0}
    s4_rows: list[dict[str, float]] = []
    kernel_ms: list[float] = []

    with torch.no_grad():
        for query in panel:
            pool = query.candidate_index.numpy().astype(np.int64, copy=False)
            if pool.size == 0:
                continue
            relevant = np.isin(pool, query.relevant_global.numpy())
            seed_local = query.retrieval_seed_local
            seeds = (
                np.unique(pool[seed_local.numpy().astype(np.int64, copy=False)])
                if seed_local is not None and seed_local.numel()
                else pool[:0]
            )

            s4_scores = (
                _m2b._one_query_scores(
                    model, query, node_embeddings, query_embeddings, store, device
                )
                .cpu()
                .numpy()
                .astype(np.float64)
                .reshape(-1)
            )
            s4_ranks = offset.rank_positions(s4_scores, pool)
            s4_row = _metrics(s4_ranks, relevant)

            query_vector = query_array[query.query_index]
            residuals = _residual_vectors(
                query_vector, node_array[seeds], node_array[int(query.anchor_global)]
            )
            for name, state in residuals.items():
                residual_states[name]["degenerate"] += int(state["degenerate"])
                residual_states[name]["fell_back_to_query"] += int(state["fell_back_to_query"])

            pool_rows = node_array[pool]
            seed_rows = node_array[seeds]
            tick = time.perf_counter()
            for view, (rowptr, col) in families.items():
                seed_pos, pool_pos = offset.seed_incident_pairs(pool, seeds, rowptr, col)
                for residual_name in RESIDUALS:
                    direction = offset.pool_directional_scores(
                        residuals[residual_name]["vector"],
                        pool_rows,
                        seed_rows,
                        seed_pos,
                        pool_pos,
                    )
                    signal = getattr(direction, RANKING_SIGNAL)
                    coverage.setdefault((residual_name, view), []).append(direction.coverage)

                    direction_ranks = offset.rank_with_coverage(signal, direction.covered, pool)
                    collected.setdefault(
                        ("direction_only", residual_name, view), []
                    ).append(_metrics(direction_ranks, relevant))

                    fused = offset.reciprocal_rank_fusion(
                        [s4_ranks, direction_ranks], pool
                    )
                    collected.setdefault(
                        ("S4_plus_direction_rrf", residual_name, view), []
                    ).append(_metrics(offset.rank_positions(fused, pool), relevant))

                    # The load-bearing diagnostic. The coverage mask goes in
                    # beside the scores rather than folded into them: an
                    # uncovered candidate has no direction to compare, and
                    # masking it to a sentinel first made a comparison between
                    # two uncovered candidates come out as -inf minus -inf.
                    margin = offset.error_conditioned_margin(
                        query.query_id,
                        s4_scores,
                        signal,
                        pool,
                        relevant,
                        covered=direction.covered,
                    )
                    if margin is not None:
                        margins.setdefault((residual_name, view), []).append(margin)
            kernel_ms.append((time.perf_counter() - tick) * 1000.0)

            if not relevant.any():
                excluded["no_relevant_in_pool"] += 1
            elif relevant[int(np.argmin(s4_ranks))]:
                excluded["s4_top1_already_right"] += 1
            s4_rows.append(s4_row)

    for residual_name in RESIDUALS:
        for view in PROVENANCE_FAMILIES:
            collected[("S4", residual_name, view)] = s4_rows

    matrix = [
        {
            "arm": arm,
            "residual": residual_name,
            "provenance": view,
            **_mean_metrics(rows),
            "coverage_mean": (
                None
                if arm == "S4"
                else float(np.mean(coverage[(residual_name, view)]))
            ),
        }
        for (arm, residual_name, view), rows in sorted(collected.items())
    ]

    admission = _admission_diagnostic(
        panel,
        node_array=node_array,
        query_array=query_array,
        rowptr=a64_rowptr,
        col=a64_col,
        num_nodes=int(dataset.num_nodes),
        # A64's own budget object, built by the function that defines it, so
        # the control cannot drift from the incumbent by a constant typed here.
        budget=_m1a._a64_budget(
            per_seed_cap=args.per_seed_cap,
            neighbour_scan_cap_per_seed=args.neighbour_scan_cap_per_seed,
        ),
        cap=args.admission_cap,
        mainline_family=args.a64_mainline_family,
        stored_was_symmetric=stored_symmetry[args.a64_mainline_family],
    )

    result: dict[str, Any] = {
        "status": COMPLETE_STATUS,
        "stage": "m2c_stage0_probe",
        "dataset": args.dataset,
        "regime": args.regime,
        "cell": f"{args.dataset}/{args.regime}",
        "declaration": "configs/m2c_s4_structural_conditioning.yaml",
        "protocol": "docs/M2C_S4_STRUCTURAL_CONDITIONING_PROTOCOL.md",
        "compute_record": "docs/M2C_STAGE0_COMPUTE_RECORD.md",
        "data_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract": candidate_contract,
        "source_commit": args.source_commit,
        "trained_anything": False,
        "test_split_read": False,
        "split": "validation",
        "panel": {
            "queries": len(panel),
            "portion": "development (fit) portion of the validation split",
            "holdout_left_unexamined": len(holdout),
            "holdout_fraction": args.holdout_fraction,
            "why": (
                "M2B divided each cell's validation split 80/20 and reported on "
                "the 20%. Stage 0 reads only the 80% that fitting already spent, "
                "so the surface M2B's filed numbers come from stays clean."
            ),
        },
        "ranking_signal": RANKING_SIGNAL,
        "fusion": {
            "method": "reciprocal_rank_fusion",
            "constant": offset.RRF_CONSTANT,
            "constant_status": "REUSED_NOT_NEW",
            "constant_source": "configs/candidate_budget.yaml#candidate_construction.rrf_constant",
            "weights": "equal",
            "swept": False,
        },
        "provenance_families": PROVENANCE_FAMILIES,
        "matrix": matrix,
        "s4_reference": _mean_metrics(s4_rows),
        "residual_states": residual_states,
        "error_conditioned": {
            f"{residual_name}|{view}": offset.summarise_margins(rows)
            for (residual_name, view), rows in sorted(margins.items())
        },
        "excluded_from_the_error_population": excluded,
        "admission_diagnostic": admission,
        "coverage": {
            f"{residual_name}|{view}": {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "queries_with_zero_coverage": int(sum(1 for v in values if v == 0.0)),
            }
            for (residual_name, view), values in sorted(coverage.items())
        },
        "systems": {
            "directional_kernel_ms_per_query_p50": float(np.percentile(kernel_ms, 50))
            if kernel_ms
            else None,
            "directional_kernel_ms_per_query_p95": float(np.percentile(kernel_ms, 95))
            if kernel_ms
            else None,
            "wall_seconds": round(time.perf_counter() - started, 1),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--regime", required=True, choices=["R1", "R2", "R3"])
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-contract-compatibility", required=True)
    parser.add_argument("--cell-features", type=Path, required=True)
    parser.add_argument("--s4-checkpoint", type=Path, required=True)
    parser.add_argument("--edge-provenance-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--per-seed-cap", type=int, default=16)
    parser.add_argument("--neighbour-scan-cap-per-seed", type=int, default=4096)
    # The family A64 is defined on, and the value M2 recorded in the R3 cell
    # master's build key. Imported, not typed: a different string here does not
    # merely mislabel the arm, it makes the sealed master refuse to load.
    parser.add_argument("--a64-mainline-family", default=_m1a.MAINLINE_FAMILY)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--panel-cap", type=int, default=0)
    parser.add_argument("--admission-cap", type=int, default=2000)
    parser.add_argument("--source-commit", default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = run(args)
    print(f"{result['cell']}: {result['panel']['queries']} queries, {len(result['matrix'])} rows")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
