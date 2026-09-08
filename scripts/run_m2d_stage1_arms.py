#!/usr/bin/env python
"""Fit M2D Stage 1's two arms on one cell, and measure them against native S4.

Section 5 authorises eight fits: A1 and A3-MINIMAL, on two blockers and two
controls, at seed 0. This runner does one cell of that -- both arms -- and
writes one immutable artifact per arm, which is the unit
``scripts/m2d_stage1_gate.py`` reads.

Nothing here re-derives S4, S3 or the scorer. The arms reach the fit through
the injection point M2B already built for varying the semantic branch, so the
scorer, the loss, the optimizer, the splits, the store and the candidate
contract are the same objects M2B fit, and section 12's list of things not to
reopen is satisfied by construction rather than by promise.

Why a comparison to filed rows needs earning
--------------------------------------------
The gate compares these arms against M2B's *filed* S3 and S4 rows rather than
against anything refit here, which is section 5's reuse rule and is the only
affordable design. But a delta against a filed row means nothing unless this
container scored the same queries with the same inputs. So three identities
are checked before any arm is judged, and each refuses rather than warns:

**The panel.** ``holdout_split`` is deterministic, so the same data under the
same fraction yields the same held-out queries -- but "should" is not
"did". The digest of this container's panel must equal the digest of the
panel M2B recorded for this cell.

**The structural inputs.** The cell master is loaded, never rebuilt, and the
resulting shared-inputs hash must equal M2B's for this cell. A different store
would make every delta a difference in structure wearing a semantic label.

**Native S4 itself.** M2B's S4 checkpoint is loaded and re-scored here, on this
panel, with no training. Its metrics must reproduce M2B's filed S4 row. This
is the check that can actually fail if either of the two above is subtly
wrong, and it is also what section 8's error-conditioned analysis needs: the
per-query rank-1 outcome of native S4 on exactly the queries the arms scored.

The re-score is not a fit. It trains nothing and it produces no comparison row
of its own -- M2B's filed numbers remain the baseline. It is the same
"reused checkpoint, re-benchmarked here" pattern M2B used to bring M2's S3
into its own containers.

What each artifact carries
--------------------------
Section 15's list, per arm: the metrics, the deltas' ingredients, p50/p95/p99
uncached, the exact parameter counts including what the frozen scorer costs to
widen, the training time, and section 8's top-1 correction analysis in both
directions. Written through :mod:`mp_retrieval.run_artifacts`, so the path
carries the run id, the source commit and the config fingerprint, the write
refuses to overwrite, and the file is read back and verified before this
function returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval import run_artifacts
from mp_retrieval.complete_data import load_complete_dataset
from mp_retrieval.data import QuerySplit
from mp_retrieval.headroom_v2 import ragged_from_rows, regime_headroom
from mp_retrieval.m2b_inference_timing import (
    DEFAULT_REPEATS,
    DEFAULT_WARMUP_QUERIES,
    measure_uncached_inference,
)
from mp_retrieval.m2d_stage1_arms import build_arm_head
from scripts import feature_build_contract as fbc
from scripts import run_m1a_feature_screen as _m1a
from scripts import run_m2_qls_v2_freeze as _m2
from scripts import run_m2b_semantic_minimality as _m2b
from scripts.m2d_stage1_gate import check_payload
from scripts.run_edge_provenance import _atomic_json
from scripts.run_graph_context_d1 import holdout_split
from scripts.run_operator_screen import _aggregate_rows
from scripts.run_sa_mlp_confirmation import (
    _fit,
    _score_once,
    validate_candidate_contract,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"

PHASE = "m2d"
STAGE = "stage1"

#: The status the gate admits. A file that does not carry it is refused there
#: rather than skipped, so this string is part of the contract between the two.
COMPLETE_STATUS = "M2D_STAGE1_ARM_COMPLETE"

#: Stage 2's, for the same reason. The two stages write different strings
#: because they are different experiments bought under different authority,
#: and an artifact that could not say which one paid for it would let a
#: three-seed mean be assembled out of rows nobody authorised.
STAGE_2 = "stage2"
STAGE_2_COMPLETE_STATUS = "M2D_STAGE2_SEED_COMPLETE"

#: Section 8b's arms, in the order they are fit. The control first: if it is
#: going to fail on this cell for a systems or data reason, that is better
#: learned before the candidate's fit is paid for.
ARMS = ("A1", "A3_MINIMAL")

#: The rung whose checkpoint is reused and re-scored, and against which the
#: error-conditioned analysis is conditioned. Never refit.
NATIVE_RUNG = "S4"

KS = _m1a.KS
DECLARED_SEED = 0


def authorised_seeds() -> dict[int, dict[str, Any]]:
    """Which seeds this runner may fit, and on whose authority.

    Read out of the declaration rather than typed here. Stage 1 ran at seed 0
    under `stage_1`; section 15b later authorised seeds 1 and 2 for A3-MINIMAL
    alone. Widening the runner by hand would have made the seed gate a comment
    -- this way, running a new seed requires the file that has to justify it to
    say so first, and the refusal below quotes what the file actually permits.

    Stage 1's entry is not rewritten by Stage 2's presence: seed 0 keeps both
    arms and keeps writing Stage 1's status, so a re-run of seed 0 cannot
    silently become a Stage-2 row.
    """

    config = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    table = {
        int(seed): {"stage": STAGE, "status": COMPLETE_STATUS, "arms": tuple(ARMS)}
        for seed in config["stage_1"]["seeds"]
    }
    stage_2 = config.get("stage_2") or {}
    for seed in stage_2.get("seeds", ()):
        table[int(seed)] = {
            "stage": STAGE_2,
            "status": STAGE_2_COMPLETE_STATUS,
            "arms": tuple(stage_2["arms"]),
        }
    return table


def panel_digest(query_ids: list[str]) -> str:
    """A hash of the held-out panel, in order.

    Order is part of the identity, not an incidental detail: section 8's
    analysis pairs this container's per-query rows against native S4's by
    position, and two identical sets in different orders would pair every
    query with the wrong one while agreeing on every aggregate.
    """

    canonical = json.dumps(list(query_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def check_identity(
    *,
    observed: str,
    expected: str | None,
    what: str,
    consequence: str,
) -> dict[str, Any]:
    """Refuse a mismatch; record the check either way.

    ``expected`` may be ``None`` only for a local smoke run that has no filed
    cell to match. That is recorded in the artifact as an unchecked identity
    rather than passed over silently, because an artifact that cannot say
    which panel it scored cannot support a delta against anything.
    """

    if expected is not None and observed != expected:
        raise ValueError(
            f"{what} does not match what M2B filed for this cell "
            f"(observed {observed[:16]}, expected {expected[:16]}). {consequence} "
            "Refusing rather than reporting a delta against rows this container did "
            "not reproduce."
        )
    return {
        "what": what,
        "observed": observed,
        "expected": expected,
        "checked": expected is not None,
    }


def integration_analysis(
    native_rows: list[dict[str, Any]], arm_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Section 8: among the queries native S4 got wrong at rank 1, what changed.

    Both directions are mandatory. A corrections count on its own is not a
    measurement -- an arm that repairs 200 top-1 errors and breaks 300
    previously correct ones has made rank 1 worse, and a report that showed
    only the 200 would say the opposite.

    "Wrong at rank 1" is ``recall@1 == 0``: nothing relevant was ranked first.
    On a single-gold cell that is the whole story; on a multi-gold cell
    recall@1 is fractional, and a query that goes from 0 to any positive value
    has had a relevant node lifted to rank 1, which is the event this analysis
    is about.
    """

    if len(native_rows) != len(arm_rows):
        raise ValueError(
            f"the arm scored {len(arm_rows)} queries and native S4 scored "
            f"{len(native_rows)}; nothing can be paired between them"
        )

    native = [float(row["recall@1"]) for row in native_rows]
    arm = [float(row["recall@1"]) for row in arm_rows]

    errors = [index for index, value in enumerate(native) if value == 0.0]
    corrected = sum(1 for index in errors if arm[index] > 0.0)
    newly_broken = sum(
        1 for index, value in enumerate(native) if value > 0.0 and arm[index] == 0.0
    )
    return {
        "queries": len(native),
        "s4_top1_errors": len(errors),
        "s4_top1_correct": len(native) - len(errors),
        "corrected": corrected,
        "newly_broken": newly_broken,
        "net_top1_corrections": corrected - newly_broken,
        "corrected_fraction_of_s4_errors": (
            corrected / len(errors) if errors else None
        ),
        "newly_broken_fraction_of_s4_correct": (
            newly_broken / (len(native) - len(errors))
            if len(native) - len(errors)
            else None
        ),
        "definition": (
            "conditioned on native S4's per-query recall@1 on this same panel, "
            "re-scored in this container from M2B's checkpoint. A correction is a "
            "query with S4 recall@1 == 0 and arm recall@1 > 0; a break is the "
            "reverse. Net is corrections minus breaks."
        ),
    }


def _build(arm: str, *, precomputed_width: int, embedding_dim: int, args, device):
    """One arm's model: M2's scorer, this arm's semantic branch.

    ``semantic_rung`` and the head's own ``rung`` must agree or ``M1AScorer``
    refuses, which is what keeps a fit from being recorded under another arm's
    name.
    """

    return _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=arm,
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=embedding_dim,
        semantic_head=build_arm_head(arm, embedding_dim),
    ).to(device)


def checkpoint_round_trips(checkpoint: Path, model, *, arm, precomputed_width, query,
                           node_embeddings, query_embeddings, store, device,
                           dropout, temperature) -> dict[str, Any]:
    """M2B's check, rebuilt through ``build_arm_head`` instead of a rung name.

    Same claim, and the same reason for making it: "a checkpoint was written"
    is a file-existence statement, while what any later reuse depends on is
    that loading it back reproduces the scores this fit reported.
    """

    embedding_dim = int(node_embeddings.shape[1])
    reloaded = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=arm,
        dropout=dropout,
        temperature=temperature,
        embedding_dim=embedding_dim,
        semantic_head=build_arm_head(arm, embedding_dim),
    ).to(device)
    reloaded.load_state_dict(
        torch.load(checkpoint, map_location=device, weights_only=True), strict=True
    )

    was_training = model.training
    model.eval()
    reloaded.eval()
    try:
        with torch.no_grad():
            original = _m2b._one_query_scores(
                model, query, node_embeddings, query_embeddings, store, device
            )
            restored = _m2b._one_query_scores(
                reloaded, query, node_embeddings, query_embeddings, store, device
            )
    finally:
        model.train(was_training)
    difference = float((original - restored).abs().max())
    if difference != 0.0:
        raise ValueError(
            f"reloading {checkpoint} does not reproduce this fit's scores (max absolute "
            f"difference {difference}); the checkpoint is not the model that was measured"
        )
    return {
        "loaded_strict": True,
        "max_absolute_score_difference": difference,
        "compared_on_query": query.query_id,
    }


def _score(model, held_out, node_embeddings, query_embeddings, store, device, args):
    """Score, then prove the aggregate is reconstructible from the rows kept."""

    metrics, rows, inference = _score_once(
        "sa_mlp", model, held_out, node_embeddings, query_embeddings,
        None, store, device, batch_size=args.batch_size, ks=KS, timed=True,
    )
    if not rows:
        raise ValueError("scoring produced no per-query rows, so nothing could reproduce "
                         "the aggregate")
    reconstructed = _aggregate_rows(rows)
    mismatched = _m2._aggregate_mismatch(metrics, reconstructed)
    if mismatched:
        raise ValueError(
            f"re-aggregating the stored per-query rows does not reproduce {mismatched}"
        )
    non_finite = sorted(
        name for name, value in metrics.items()
        if isinstance(value, (int, float)) and not bool(np.isfinite(float(value)))
    )
    if non_finite:
        raise ValueError(f"metrics {non_finite} are not finite")
    return metrics, rows, inference


def rescore_native_s4(
    *, args, precomputed_width, node_embeddings, query_embeddings, store, held_out, device
) -> dict[str, Any]:
    """M2B's S4, loaded and re-scored here. No training, no new baseline.

    ``strict=True`` is load-bearing. A checkpoint whose semantic branch did not
    match would otherwise load with the mismatched tensors silently dropped,
    and every number below would describe a partly randomised model while
    reporting it as M2B's.
    """

    if not args.s4_checkpoint.is_file():
        raise FileNotFoundError(
            f"{args.s4_checkpoint} does not exist, so native S4 cannot be re-scored on "
            "this panel. Refusing rather than proceeding: without it there is no "
            "per-query rank-1 baseline, and section 8's error-conditioned analysis -- "
            "the mechanism-specific evidence this stage exists to produce -- would be "
            "silently dropped from every artifact."
        )
    embedding_dim = int(node_embeddings.shape[1])
    model = _m1a.build_m1a_model(
        precomputed_width=precomputed_width,
        semantic_rung=NATIVE_RUNG,
        dropout=args.dropout,
        temperature=args.temperature,
        embedding_dim=embedding_dim,
        semantic_head=_m2b.build_semantic_head(NATIVE_RUNG, embedding_dim),
    ).to(device)
    model.load_state_dict(
        torch.load(args.s4_checkpoint, map_location=device, weights_only=True), strict=True
    )
    model.eval()

    started = time.perf_counter()
    metrics, rows, _inference = _score(
        model, held_out, node_embeddings, query_embeddings, store, device, args
    )
    # Timed HERE, on this container's clock, with the same harness and the same
    # panel the arms are timed on. M2B's filed p95 for S4 was measured in
    # another container on another day; a same-container reference is what makes
    # "this arm costs X% more than native S4" a measurement rather than a
    # comparison of two machines. The filed values stay in the record too, and
    # the gate still applies section 13's bound against the filed S3 number.
    native_latency = measure_uncached_inference(
        model=model,
        queries=held_out[: args.latency_queries],
        node_embeddings=node_embeddings,
        query_embeddings=query_embeddings,
        store=store,
        device=device,
        repeats=args.latency_repeats,
        warmup=args.latency_warmup,
    )
    reproduction = {
        "filed_recall_at_5": args.expected_s4_recall_at_5,
        "rescored_recall_at_5": float(metrics["recall@5"]),
        "difference_pp": (
            (float(metrics["recall@5"]) - args.expected_s4_recall_at_5) * 100.0
            if args.expected_s4_recall_at_5 is not None
            else None
        ),
        "bound_pp": args.s4_reproduction_bound_pp,
        "why_this_bound": (
            "the cell's own measured S4 seed-to-seed spread, from M2B's filed table. "
            "Deliberately generous: this check exists to catch a wrong panel, a wrong "
            "store or a wrong checkpoint, not to certify float reproducibility across "
            "devices. A re-score of one checkpoint that moved further than three "
            "different seeds do is not that checkpoint on that panel."
        ),
    }
    if (
        reproduction["difference_pp"] is not None
        and abs(reproduction["difference_pp"]) > args.s4_reproduction_bound_pp
    ):
        raise ValueError(
            f"re-scoring M2B's S4 checkpoint on this panel gives recall@5 "
            f"{metrics['recall@5']:.4f} against the filed "
            f"{args.expected_s4_recall_at_5:.4f} -- "
            f"{reproduction['difference_pp']:+.3f}pp, beyond the "
            f"{args.s4_reproduction_bound_pp:.3f}pp bound. Something differs between "
            "this container and the one that filed the row, so no delta computed here "
            "would be against native S4."
        )
    return {
        "rung": NATIVE_RUNG,
        "reused_from": str(args.s4_checkpoint),
        "trained_here": False,
        "why_not_a_fit": (
            "M2B paid for this fit. It is loaded and re-scored here so the arms have a "
            "per-query rank-1 baseline on their own panel; M2B's filed row remains the "
            "baseline every reported delta is taken against."
        ),
        "metrics": metrics,
        "rows": rows,
        "reproduction": reproduction,
        "uncached_inference": native_latency,
        "systems": {
            "uncached_p50_ms": native_latency["total_model_ms"]["p50"],
            "uncached_p95_ms": native_latency["total_model_ms"]["p95"],
            "uncached_p99_ms": native_latency["total_model_ms"]["p99"],
            "semantic_p95_ms": native_latency["semantic_ms"]["p95"],
            "scorer_p95_ms": native_latency["scorer_ms"]["p95"],
        },
        "rescore_seconds": round(time.perf_counter() - started, 3),
    }


def fit_one_arm(
    *,
    arm: str,
    args,
    store,
    precomputed_width: int,
    train_queries: list[Any],
    held_out: list[Any],
    node_embeddings: torch.Tensor,
    query_embeddings: torch.Tensor,
    device: torch.device,
    fit_root: Path,
    provenance: dict[str, Any],
    shared_inputs: dict[str, Any],
    native: dict[str, Any],
    identities: list[dict[str, Any]],
    headroom: dict[str, Any],
    feature_build_latency_ms: dict[str, float],
) -> dict[str, Any]:
    """One (cell, arm) fit, scored, timed, checkpointed and analysed."""

    _m1a.seed_everything(args.seed)
    embedding_dim = int(node_embeddings.shape[1])
    model = _build(
        arm, precomputed_width=precomputed_width, embedding_dim=embedding_dim,
        args=args, device=device,
    )

    fitted, training = _fit(
        "sa_mlp", model, train_queries, held_out,
        node_embeddings, query_embeddings, None, store, device,
        epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        seed=args.seed,
    )
    losses = [epoch["loss"] for epoch in training["history"]]
    if not losses or not all(bool(np.isfinite(loss)) for loss in losses):
        raise ValueError(
            f"{args.dataset}/{args.regime}/{arm}: training losses {losses} are not all "
            "finite, so the fit diverged and its metrics mean nothing"
        )

    metrics, rows, inference = _score(
        fitted, held_out, node_embeddings, query_embeddings, store, device, args
    )
    scores = _m2b.finite_scores(
        fitted, held_out, node_embeddings, query_embeddings, store, device
    )

    parameters = {
        "total": fitted.trainable_parameter_count(),
        "semantic": fitted.semantic_parameter_count(),
        "scorer": fitted.scorer_parameter_count(),
    }
    fingerprint = _m2b.semantic_rung_fingerprint(fitted.semantic_head)
    if fingerprint["semantic_parameters"] != parameters["semantic"]:  # pragma: no cover
        raise ValueError("the arm fingerprint disagrees with the model it fingerprints")

    # Section 10: the full uncached cost of this model, including the added
    # column, its reduction and the widened scorer. Not a cached or
    # precomputed variant -- the same harness M2B timed every rung with.
    latency = measure_uncached_inference(
        model=fitted,
        queries=held_out[: args.latency_queries],
        node_embeddings=node_embeddings,
        query_embeddings=query_embeddings,
        store=store,
        device=device,
        repeats=args.latency_repeats,
        warmup=args.latency_warmup,
    )

    fit_root.mkdir(parents=True, exist_ok=True)
    checkpoint = fit_root / "checkpoint.pt"
    torch.save(fitted.state_dict(), checkpoint)
    round_trip = checkpoint_round_trips(
        checkpoint, fitted, arm=arm, precomputed_width=precomputed_width,
        query=held_out[0], node_embeddings=node_embeddings,
        query_embeddings=query_embeddings, store=store, device=device,
        dropout=args.dropout, temperature=args.temperature,
    )
    query_ids = [query.query_id for query in held_out]
    _atomic_json(
        fit_root / "per_query_rows.json",
        {
            "dataset": args.dataset,
            "regime": args.regime,
            "arm": arm,
            "seed": args.seed,
            "query_ids": query_ids,
            "rows": rows,
            "aggregate_from_rows": _aggregate_rows(rows),
            "row_key_order": list(rows[0]),
        },
    )

    head = fitted.semantic_head
    achieved = metrics.get("recall@5")
    ceiling = headroom.get("recall_ceiling@5")
    authority = authorised_seeds()[args.seed]
    return {
        "status": authority["status"],
        "phase": PHASE,
        "stage": authority["stage"],
        "cell": f"{args.dataset}/{args.regime}",
        "dataset": args.dataset,
        "regime": args.regime,
        "arm": arm,
        "seed": args.seed,
        "test_split_read": False,
        "split": fbc.QUERY_SPLIT,
        "selection": fbc.QUERY_SELECTION,
        "declared_arm": _m2.DECLARED_UNIVERSAL_ARM,
        "runner_arm": _m2.RUNNER_UNIVERSAL_ARM,
        "reused_from_m2b": False,
        "precomputed_width": precomputed_width,
        "semantic_rung_fingerprint": fingerprint,
        "shared_inputs_sha256": shared_inputs["sha256"],
        "identity_checks": identities,
        "held_out_queries": len(held_out),
        "held_out_query_ids": query_ids,
        "train_queries": len(train_queries),
        "regime_headroom": headroom,
        "ceiling_attainment_at_5": (
            float(achieved) / float(ceiling) if achieved is not None and ceiling else None
        ),
        "metrics": metrics,
        "per_query_recall_at_5": [float(row["recall@5"]) for row in rows],
        "per_query_recall_at_1": [float(row["recall@1"]) for row in rows],
        "held_out_scores": scores,
        "training": training,
        "batched_inference": inference,
        "uncached_inference": latency,
        "checkpoint_round_trip": round_trip,
        "native_s4_rescore": {
            key: value for key, value in native.items() if key != "rows"
        },
        # Section 8, in both directions, against native S4 on this same panel.
        "integration": integration_analysis(native["rows"], rows),
        # Section 11. The added semantic parameters are the arm's own; the
        # added TOTAL is larger, because M2B's scorer width is derived from
        # len(feature_names) and every added column widens its first layer.
        # Reporting only the first number would be the "tiny model" claim
        # section 11 forbids.
        "parameters": {
            **parameters,
            "semantic_columns": len(head.feature_names),
            "added_semantic_columns": len(head.added_names),
            "added_semantic_parameters": head.added_parameter_count(),
            "native_s4_semantic_parameters": head.projection.parameter_count(),
        },
        "systems": {
            "train_time_seconds": training["training_seconds"],
            "peak_train_vram_mb": training["peak_training_gpu_memory_mb_total"],
            "peak_inference_vram_mb": latency["peak_inference_gpu_memory_mb"],
            "peak_rss_mb": inference["peak_cpu_rss_mb_total"],
            "uncached_p50_ms": latency["total_model_ms"]["p50"],
            "uncached_p95_ms": latency["total_model_ms"]["p95"],
            "uncached_p99_ms": latency["total_model_ms"]["p99"],
            "semantic_p50_ms": latency["semantic_ms"]["p50"],
            "semantic_p95_ms": latency["semantic_ms"]["p95"],
            "scorer_p50_ms": latency["scorer_ms"]["p50"],
            "scorer_p95_ms": latency["scorer_ms"]["p95"],
            # The attribution is the timing module's own, and its caveat travels
            # with it: the hooked pass pays for synchronisations the clean pass
            # does not, so these do not partition the total and must not be
            # presented as the added column's exact cost.
            "attribution_note": latency["attribution_note"],
            # Native S4, same container, same clock, same panel. This is the
            # honest "what did the added column cost" number; the filed S3 and
            # S4 p95 values were measured elsewhere and the gate uses them for
            # its bound, not for this attribution.
            "native_s4_same_container_p95_ms": native["systems"]["uncached_p95_ms"],
            "increase_over_native_s4_same_container_pct": (
                latency["total_model_ms"]["p95"]
                / native["systems"]["uncached_p95_ms"] - 1.0
            ) * 100.0 if native["systems"]["uncached_p95_ms"] else None,
            "added_semantic_p95_ms_same_container": (
                latency["semantic_ms"]["p95"] - native["systems"]["semantic_p95_ms"]
            ),
            "cached_or_precomputed_semantic_difference": False,
            "what_is_timed": (
                "the whole uncached forward: the structural columns, all of native S4's "
                "projection work, this arm's added column and its reduction, and the "
                "widened scorer. Section 10's primary number, cold, with nothing "
                "precomputed between queries."
            ),
        },
        "instrumentation": {
            "checkpoint": str(checkpoint),
            "per_query_rows": str(fit_root / "per_query_rows.json"),
            "query_ids": len(query_ids),
            "aggregate_metrics_reconstructed_from_rows": True,
            "source_commit": provenance["source_commit"],
            "config_fingerprint": provenance["config_sha256"],
            "dataset_fingerprint": provenance["dataset_fingerprint_sha256"],
            "candidate_fingerprint": provenance["candidate_id_order_sha256"],
            "candidate_contract_sha256": provenance["candidate_contract_sha256"],
            "feature_store_fingerprint": shared_inputs["arm_store_sha256"],
            "cell_features_fingerprint_sha256": shared_inputs["cell_features_sha256"],
            "feature_build_contract_sha256": shared_inputs["feature_build_contract_sha256"],
            "semantic_rung_fingerprint": fingerprint["sha256"],
            "semantic_parameter_count": parameters["semantic"],
            "scorer_parameter_count": parameters["scorer"],
            "total_parameter_count": parameters["total"],
            "train_time_seconds": training["training_seconds"],
            "uncached_inference_p50_ms": latency["total_model_ms"]["p50"],
            "uncached_inference_p95_ms": latency["total_model_ms"]["p95"],
            "uncached_inference_p99_ms": latency["total_model_ms"]["p99"],
            "uncached_feature_build_latency_ms": feature_build_latency_ms,
            "peak_vram_mb": max(
                float(training["peak_training_gpu_memory_mb_total"]),
                float(latency["peak_inference_gpu_memory_mb"]),
            ),
            "peak_rss_mb": inference["peak_cpu_rss_mb_total"],
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    table = authorised_seeds()
    if args.seed not in table:
        raise ValueError(
            f"seed {args.seed} is not authorised by the declaration, which permits "
            f"{sorted(table)}. Extra seeds are added by amending that file, not by "
            "passing a flag."
        )
    authority = table[args.seed]
    arms = list(args.arms) if args.arms else list(authority["arms"])
    unknown = sorted(set(arms) - set(authority["arms"]))
    if unknown:
        raise ValueError(
            f"--arms {unknown} are not authorised at seed {args.seed}, which permits "
            f"{list(authority['arms'])}. Stage 2 runs A3-MINIMAL alone: A1 was Stage "
            "1's attribution control and seeds cannot reopen what it answered."
        )

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=True)
    if len(dataset.queries) != args.expected_queries:
        raise ValueError("Complete dataset query count differs from the registered protocol")
    if dataset.node_array.shape[1] != args.frozen_embedding_dim:
        raise ValueError(
            f"{args.dataset}: node embeddings are {dataset.node_array.shape[1]}-dimensional, "
            f"--frozen-embedding-dim says {args.frozen_embedding_dim} -- every semantic head "
            "would build at the wrong width and every parameter count would be wrong"
        )
    candidate_contract = validate_candidate_contract(
        args.baseline, dataset, args.candidate_contract_compatibility
    )
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    if len(queries) < args.queries:
        raise ValueError(f"validation split holds {len(queries)} queries, needed {args.queries}")

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    node_embeddings = torch.from_numpy(
        np.array(dataset.node_array, dtype=np.float32, copy=True)
    ).to(device)
    query_embeddings = torch.from_numpy(
        np.array(dataset.query_array, dtype=np.float32, copy=True)
    ).to(device)
    num_nodes = int(dataset.num_nodes)
    query_count = len(dataset.queries)

    provenance = {
        "source_commit": _m2._source_commit(args.source_commit),
        "config_sha256": hashlib.sha256(DECLARATION_PATH.read_bytes()).hexdigest(),
        "dataset_fingerprint_sha256": args.data_fingerprint_sha256,
        "candidate_contract_sha256": candidate_contract["observed_contract_sha256"],
        "candidate_id_order_sha256": candidate_contract["candidate_id_order_sha256"],
    }

    scored_sets, master_blocks, build_latency, cell_fingerprint, decision = (
        _m2b.load_cell_under_contract(
            args.cell_features, args, args.regime, candidate_contract=candidate_contract,
        )
    )
    golds = ragged_from_rows([_m1a.QueryView(query).golds for query in queries])
    headroom, _present, _counts = regime_headroom(
        ragged_from_rows(scored_sets), golds, num_nodes=num_nodes, ks=KS
    )
    widened = [
        _m1a._widen_query(query, scored)
        for query, scored in zip(queries, scored_sets, strict=True)
    ]
    train_queries, held_out = holdout_split(widened, args.holdout_fraction)

    identities = [
        check_identity(
            observed=panel_digest([query.query_id for query in held_out]),
            expected=args.expected_panel_sha256,
            what="the held-out panel",
            consequence=(
                "The arms would be scored on different queries than the filed S3 and S4 "
                "rows they are compared against, and every delta would be unpaired."
            ),
        )
    ]

    # Section 4: the seed is part of the path, not just of the payload. Two
    # seeds of the same cell in one container would otherwise write the same
    # checkpoint, and the second would silently be scored against the first.
    fit_root = (
        Path(args.fit_root or (args.output_root / "fits"))
        / args.regime
        / f"seed_{args.seed}"
    )
    store, precomputed_width = _m1a._arm_store(
        arm=_m2.RUNNER_UNIVERSAL_ARM, regime=args.regime, master_blocks=master_blocks,
        queries=widened, query_count=query_count, num_nodes=num_nodes,
    )
    arm_store_sha256 = _m2.save_feature_store(
        store,
        fit_root / "arm_store",
        extra={
            "dataset": args.dataset,
            "regime": args.regime,
            "declared_arm": _m2.DECLARED_UNIVERSAL_ARM,
            "runner_arm": _m2.RUNNER_UNIVERSAL_ARM,
            "precomputed_width": precomputed_width,
            "built_by": "scripts/run_m2d_stage1_arms.py",
            "shared_by_arms": arms,
        },
    )
    shared = _m2b.shared_inputs_of_cell(
        cell_features_sha256=cell_fingerprint,
        arm_store_sha256=arm_store_sha256,
        feature_build_contract_sha256=decision["feature_build_contract_sha256"],
        precomputed_width=precomputed_width,
        query_ids=[query.query_id for query in held_out],
        family_digests=_m2b.structural_family_digests(master_blocks),
        candidate_contract=candidate_contract,
    )
    identities.append(
        check_identity(
            observed=shared["sha256"],
            expected=args.expected_shared_inputs_sha256,
            what="the cell's shared structural inputs",
            consequence=(
                "A difference between these arms and the filed rungs would be a "
                "difference in structure wearing a semantic label."
            ),
        )
    )

    native = rescore_native_s4(
        args=args, precomputed_width=precomputed_width, node_embeddings=node_embeddings,
        query_embeddings=query_embeddings, store=store, held_out=held_out, device=device,
    )

    receipts: dict[str, Any] = {}
    for arm in arms:
        payload = fit_one_arm(
            arm=arm,
            args=args,
            store=store,
            precomputed_width=precomputed_width,
            train_queries=train_queries,
            held_out=held_out,
            node_embeddings=node_embeddings,
            query_embeddings=query_embeddings,
            device=device,
            fit_root=fit_root / arm.lower(),
            provenance=provenance,
            shared_inputs=shared,
            native=native,
            identities=identities,
            headroom=headroom,
            feature_build_latency_ms=build_latency,
        )
        payload["provenance"] = provenance
        payload["device"] = str(device)
        payload["data_fingerprint_sha256"] = args.data_fingerprint_sha256
        payload["candidate_contract"] = candidate_contract
        # Before the write, not after: a payload the gate cannot judge is a
        # wasted fit, and the container that produced it is the only place
        # that can still say so cheaply.
        check_payload(payload, f"{args.dataset}/{args.regime}/{arm}")

        identity = run_artifacts.ArtifactIdentity(
            phase=PHASE,
            dataset=args.dataset,
            regime=args.regime,
            arm=arm,
            source_commit=args.source_commit,
            run_id=args.run_id or run_artifacts.current_run_id(),
            seed=args.seed,
        )
        receipt = run_artifacts.write_artifact(
            args.output_root,
            identity,
            payload,
            config_fingerprint=args.config_fingerprint,
            rows_at="per_query_recall_at_5",
        )
        payload["artifact"] = receipt.as_dict()
        receipts[arm] = receipt.as_dict()

    return {
        "status": (
            "M2D_STAGE1_CELL_COMPLETE"
            if authority["stage"] == STAGE
            else "M2D_STAGE2_CELL_COMPLETE"
        ),
        "phase": PHASE,
        "stage": authority["stage"],
        "cell": f"{args.dataset}/{args.regime}",
        "seed": args.seed,
        "arms": arms,
        "test_split_read": False,
        "identity_checks": identities,
        "native_s4_reproduction": native["reproduction"],
        "held_out_queries": len(held_out),
        "artifacts": receipts,
        "wall_seconds": round(time.perf_counter() - started, 1),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--regime", required=True, choices=["R1", "R2", "R3"])
    parser.add_argument("--data-fingerprint-sha256", required=True)
    parser.add_argument("--expected-queries", type=int, required=True)
    parser.add_argument("--frozen-embedding-dim", type=int, required=True)
    parser.add_argument("--queries", type=int, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate-contract-compatibility", default=None)
    parser.add_argument("--cell-features", type=Path, required=True)
    parser.add_argument("--s4-checkpoint", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=DECLARED_SEED)
    # --- the identities that make a delta against a filed row legitimate ----
    parser.add_argument("--expected-panel-sha256", default=None)
    parser.add_argument("--expected-shared-inputs-sha256", default=None)
    parser.add_argument("--expected-s4-recall-at-5", type=float, default=None)
    parser.add_argument("--s4-reproduction-bound-pp", type=float, required=True)
    # --- M2B's training setup, which section 12 says not to reopen ----------
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--latency-queries", type=int, default=200)
    parser.add_argument("--latency-repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--latency-warmup", type=int, default=DEFAULT_WARMUP_QUERIES)
    # --- M2's frozen build key, transcribed so the sealed master loads ------
    parser.add_argument("--per-seed-cap", type=int, default=16)
    parser.add_argument("--neighbour-scan-cap-per-seed", type=int, default=4096)
    parser.add_argument("--a64-mainline-family", default=_m2.MAINLINE_FAMILY)
    parser.add_argument("--device", default=None)
    parser.add_argument("--fit-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-fingerprint", required=True)
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run(args)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
