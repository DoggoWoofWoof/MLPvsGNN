#!/usr/bin/env python
"""Did the M2B smoke establish what the declaration said it must establish?

``launch_authorization.gates.engineering_smoke_passes`` is flipped by this
script and by nothing else. The declaration says so in as many words: a smoke
that ran is not a smoke that passed, and the gate turns on "a verification
artifact against smoke_before_fanout.what_the_smoke_must_establish, item by
item, not by the job exiting zero". Reading the items off a run log and
asserting them in conversation is the failure that sentence exists to prevent.

So every declared item is matched to a check here by a phrase unique to that
item, and an item this script cannot settle STOPS the verification rather than
being quietly dropped from the count. That is the only way the artifact can
honestly say "11 of 11": the denominator has to come from the declaration, not
from the length of a list in this file.

Two of the eleven get more than a field lookup:

*   The controlled comparison. The runner hands one store to every rung, so the
    property holds by construction -- but a claim that load-bearing should not
    be checked only by the process that benefits from it being true. The
    digests are recompared here, and the structural families are checked to be
    real digests over real columns, because a digest over an empty NODE_ROLE
    column would agree across rungs and establish nothing.
*   The parameter accounting. 449 and 205,217 are the phase's headline numbers,
    and the second is the one the earlier planning prose had at 98,304. They
    are read from the DECLARATION rather than from a constant here, so this
    script cannot agree with a wrong number by carrying its own copy of it.

Reads only. Writes outputs/m2b_semantic_minimality/smoke_verification.json and
spends no compute.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.feature_build_contract import SCIENTIFIC_BUILD_KEY_FIELDS  # noqa: E402
from scripts.run_m1a_feature_screen import ARM_FAMILIES  # noqa: E402
from scripts.run_m2b_semantic_minimality import (  # noqa: E402
    DECLARED_UNIVERSAL_ARM,
    RUNNER_UNIVERSAL_ARM,
)

DECLARATION_PATH = REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml"
SMOKE_RESULT_PATH = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "smoke" / "2wiki_clean.json"
)
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "smoke_verification.json"

HEX64 = re.compile(r"^[0-9a-f]{64}$")

PASSED = "PASSED"
FAILED = "FAILED"

#: The structural families the smoke's arm actually carries, read from the arm
#: vocabulary rather than transcribed, so a change to the arm cannot leave this
#: script checking families the fits no longer have. Keyed on the runner's
#: composition string (BASE+NODE_ROLE+SUPPORT+PATH) rather than on the
#: declaration's name for it (QLS-UNIVERSAL), which names no families.
ARM_STRUCTURAL_FAMILIES = ("BASE",) + ARM_FAMILIES[RUNNER_UNIVERSAL_ARM]

#: Each declared item, matched to the check that settles it by a phrase unique
#: to that item's text. Matching on a phrase rather than on list position means
#: reordering the declaration cannot silently repoint a check at another item;
#: a phrase that stops matching stops the verification.
ITEM_KEYS: dict[str, str] = {
    "both heads instantiate": "heads_instantiate_at_the_frozen_width",
    "feature_build_contract_sha256": "store_loads_under_its_contract_and_is_not_rebuilt",
    "byte-identical shared inputs": "only_the_semantic_rung_differs",
    "S2 reports semantic parameters 0": "s2_parameter_accounting",
    "S4 reports semantic parameters 196,608": "s4_parameter_accounting",
    "a checkpoint is written": "checkpoint_round_trips",
    "per-query rows are written": "aggregate_rebuilt_from_rows",
    "every training-epoch loss is finite": "losses_and_scores_are_finite",
    "every instrumentation field": "instrumentation_is_populated",
    "full uncached inference p50": "uncached_latency_and_peak_vram",
    "the measured S4 fit time": "s4_fit_time_measured",
}


def _flat(text: Any) -> str:
    return " ".join(str(text).split())


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _is_digest(value: Any) -> bool:
    return bool(HEX64.match(str(value or "")))


def declared_items(declaration: dict[str, Any]) -> dict[str, str]:
    """The declaration's own list, keyed by the check that settles each item.

    The denominator of "n of n passed" comes from here. An item this script
    cannot match refuses, because a verification that silently checked ten of
    eleven items would report a pass it had not earned.
    """

    items = declaration["smoke_before_fanout"]["what_the_smoke_must_establish"]
    matched: dict[str, str] = {}
    for item in items:
        flat = _flat(item)
        keys = [key for phrase, key in ITEM_KEYS.items() if phrase in flat]
        if len(keys) != 1:
            raise SystemExit(
                f"the declared item {flat!r} matches {keys or 'no'} check(s) in this "
                "script. Every declared item must be settled by exactly one check; "
                "refusing to report a verification that skipped one."
            )
        if keys[0] in matched:
            raise SystemExit(f"two declared items both map to {keys[0]!r}")
        matched[keys[0]] = flat
    return matched


def _rung(cell: dict[str, Any], rung: str) -> dict[str, Any]:
    fit = (cell.get("rungs") or {}).get(rung)
    if not isinstance(fit, dict):
        raise SystemExit(f"the smoke result has no {rung} fit; there is nothing to verify")
    return fit


def _parameters_check(
    declaration: dict[str, Any], fit: dict[str, Any], rung: str
) -> dict[str, Any]:
    """Against the declaration's counts, never against a constant in this file."""

    declared = declaration["semantic_candidates"][rung]
    expected = {
        "semantic": int(declared["semantic_trainable_parameters"]),
        "scorer": int(declared["scorer_trainable_parameters"]),
        "total": int(declared["total_trainable_parameters"]),
        "semantic_columns": int(declared["semantic_output_width"]),
    }
    parameters = fit.get("parameters") or {}
    fingerprint = fit.get("semantic_rung_fingerprint") or {}
    observed = {
        "semantic": parameters.get("semantic"),
        "scorer": parameters.get("scorer"),
        "total": parameters.get("total"),
        # The width lives on the fingerprint rather than in `parameters`.
        # Reading it there also ties the count to the head that was hashed.
        "semantic_columns": fingerprint.get("semantic_columns"),
    }
    return {
        "passed": observed == expected,
        "expected": expected,
        "observed": observed,
        "read_from": "configs/m2b_semantic_minimality.yaml#semantic_candidates",
    }


def check(
    declaration: dict[str, Any], result: dict[str, Any], regime: str
) -> dict[str, dict[str, Any]]:
    cell = (result.get("cells") or {}).get(regime)
    if not isinstance(cell, dict):
        raise SystemExit(
            f"the smoke result has no {regime!r} cell; the declaration smokes "
            f"{declaration['smoke_before_fanout']['cell']}"
        )
    smoked = list(declaration["smoke_before_fanout"]["rungs"])
    fits = {rung: _rung(cell, rung) for rung in smoked}
    s2, s4 = fits["S2"], fits["S4"]

    frozen = int(declaration["semantic_formulas"]["frozen_embedding_dim"])
    fingerprints = {
        rung: fit.get("semantic_rung_fingerprint") or {} for rung, fit in fits.items()
    }
    declared_columns = {
        rung: int(declaration["semantic_candidates"][rung]["semantic_output_width"])
        for rung in smoked
    }

    admission = cell.get("store_admission") or {}
    shared_of_cell = cell.get("shared_inputs") or {}
    families = shared_of_cell.get("structural_family_digests") or {}
    empty_families = sorted(
        family for family in ARM_STRUCTURAL_FAMILIES if not _is_digest(families.get(family))
    )
    shared_digests = {rung: fit.get("shared_inputs_sha256") for rung, fit in fits.items()}
    rung_digests = {
        rung: fingerprint.get("sha256") for rung, fingerprint in fingerprints.items()
    }
    panel = cell.get("held_out_query_ids") or []

    declared_fields = list(declaration["instrumentation_requirement"]["fields"])
    missing_fields = {
        rung: sorted(
            field
            for field in declared_fields
            if (fit.get("instrumentation") or {}).get(field) is None
        )
        for rung, fit in fits.items()
    }

    losses = {
        rung: [epoch.get("loss") for epoch in (fit.get("training") or {}).get("history", [])]
        for rung, fit in fits.items()
    }
    scores = {rung: fit.get("held_out_scores") or {} for rung, fit in fits.items()}
    latency = {
        rung: {
            key: (fit.get("systems") or {}).get(key)
            for key in (
                "uncached_inference_p50_ms",
                "uncached_inference_p95_ms",
                "uncached_inference_p99_ms",
            )
        }
        for rung, fit in fits.items()
    }
    seconds = {
        rung: (fit.get("systems") or {}).get("train_time_seconds") for rung, fit in fits.items()
    }

    return {
        "heads_instantiate_at_the_frozen_width": {
            "passed": all(
                fingerprint.get("embedding_dim") == frozen
                for fingerprint in fingerprints.values()
            )
            and {rung: f.get("semantic_columns") for rung, f in fingerprints.items()}
            == declared_columns,
            "expected": {"embedding_dim": frozen, "semantic_columns": declared_columns},
            "observed": {
                rung: {
                    "embedding_dim": fingerprint.get("embedding_dim"),
                    "semantic_columns": fingerprint.get("semantic_columns"),
                    "projection_dim": fingerprint.get("projection_dim"),
                    "class": fingerprint.get("class"),
                }
                for rung, fingerprint in fingerprints.items()
            },
        },
        "store_loads_under_its_contract_and_is_not_rebuilt": {
            # Admitted on the seven scientific identity fields, with the
            # declaration-hash drift carried as provenance rather than as a
            # refusal -- which is the whole reason the contract exists.
            "passed": admission.get("admitted") is True
            and admission.get("scientific_key_differences") == []
            and admission.get("evidence") in {"recorded", "reconstructed"}
            and _is_digest(admission.get("feature_build_contract_sha256"))
            and bool(result.get("features_were_loaded_not_rebuilt")),
            "identity_fields_compared": len(SCIENTIFIC_BUILD_KEY_FIELDS),
            "observed": {
                "admitted": admission.get("admitted"),
                "evidence": admission.get("evidence"),
                "scientific_key_differences": admission.get("scientific_key_differences"),
                "feature_build_contract_sha256": admission.get("feature_build_contract_sha256"),
                "original_full_config_sha256": admission.get("original_full_config_sha256"),
                "features_were_loaded_not_rebuilt": result.get(
                    "features_were_loaded_not_rebuilt"
                ),
            },
        },
        "only_the_semantic_rung_differs": {
            "passed": len(set(shared_digests.values())) == 1
            and all(_is_digest(digest) for digest in shared_digests.values())
            and len(set(rung_digests.values())) == len(fits)
            and all(_is_digest(digest) for digest in rung_digests.values())
            and not empty_families
            and len(panel) > 0
            and int(shared_of_cell.get("held_out_query_count") or -1) == len(panel),
            "observed": {
                "shared_inputs_sha256": shared_digests,
                "semantic_rung_fingerprint_sha256": rung_digests,
                "structural_family_digests": {
                    family: families.get(family) for family in ARM_STRUCTURAL_FAMILIES
                },
                "held_out_queries": len(panel),
                "candidate_normalisation": shared_of_cell.get("candidate_normalisation"),
                "candidate_contract_sha256": shared_of_cell.get("candidate_contract_sha256"),
            },
            "families_without_a_real_digest": empty_families,
            "what_the_shared_digest_covers": sorted(
                key for key in shared_of_cell if key != "sha256"
            ),
        },
        "s2_parameter_accounting": _parameters_check(declaration, s2, "S2"),
        "s4_parameter_accounting": _parameters_check(declaration, s4, "S4"),
        "checkpoint_round_trips": {
            "passed": all(
                (fit.get("checkpoint_round_trip") or {}).get("loaded_strict") is True
                and (fit.get("checkpoint_round_trip") or {}).get(
                    "max_absolute_score_difference"
                )
                == 0.0
                and bool((fit.get("instrumentation") or {}).get("checkpoint"))
                for fit in fits.values()
            ),
            "observed": {rung: fit.get("checkpoint_round_trip") for rung, fit in fits.items()},
        },
        "aggregate_rebuilt_from_rows": {
            "passed": len(panel) > 0
            and all(
                (fit.get("instrumentation") or {}).get(
                    "aggregate_metrics_reconstructed_from_rows"
                )
                is True
                and int((fit.get("instrumentation") or {}).get("query_ids") or 0) == len(panel)
                and len(fit.get("per_query_recall_at_5") or []) == len(panel)
                for fit in fits.values()
            ),
            "observed": {
                rung: {
                    "reconstructed": (fit.get("instrumentation") or {}).get(
                        "aggregate_metrics_reconstructed_from_rows"
                    ),
                    "query_ids": (fit.get("instrumentation") or {}).get("query_ids"),
                    "per_query_recall_at_5": len(fit.get("per_query_recall_at_5") or []),
                    "per_query_rows": (fit.get("instrumentation") or {}).get("per_query_rows"),
                }
                for rung, fit in fits.items()
            },
            "cell_panel": len(panel),
        },
        "losses_and_scores_are_finite": {
            # Argsorting a vector containing NaN still returns a permutation,
            # so a model emitting NaN produces a ranking and healthy-looking
            # aggregates. A reused rung is exempt from the loss half only
            # because it trains no epochs; it is not exempt from the score half.
            "passed": all(
                bool(values) and all(_finite(loss) for loss in values)
                for rung, values in losses.items()
                if not fits[rung].get("reused_from_m2")
            )
            and all(entry.get("all_finite") is True for entry in scores.values())
            and all(int(entry.get("scored_values") or 0) > 0 for entry in scores.values()),
            "observed": {
                "epoch_losses": losses,
                "held_out_scores": scores,
                "reused_from_m2": {rung: fit.get("reused_from_m2") for rung, fit in fits.items()},
            },
        },
        "instrumentation_is_populated": {
            "passed": not any(missing_fields.values()),
            "declared_fields": len(declared_fields),
            "missing": missing_fields,
        },
        "uncached_latency_and_peak_vram": {
            "passed": all(
                all(_positive(value) for value in per_rung.values())
                for per_rung in latency.values()
            )
            and all(
                _finite((fit.get("instrumentation") or {}).get("peak_vram_mb"))
                and _finite((fit.get("instrumentation") or {}).get("peak_rss_mb"))
                for fit in fits.values()
            ),
            "observed": {
                "uncached_inference_ms": latency,
                "peak_vram_mb": {
                    rung: (fit.get("instrumentation") or {}).get("peak_vram_mb")
                    for rung, fit in fits.items()
                },
                "peak_rss_mb": {
                    rung: (fit.get("instrumentation") or {}).get("peak_rss_mb")
                    for rung, fit in fits.items()
                },
                "measured_span": (s4.get("uncached_inference") or {}).get("measured_span"),
                "tie_break_orders_on": (s4.get("systems") or {}).get("tie_break_orders_on"),
            },
        },
        "s4_fit_time_measured": {
            # The number that replaces the guessed 2.5x multiplier. S4 must
            # actually have trained: a reused S4 reports zero seconds and would
            # measure nothing while still producing a ratio that looks like an
            # answer.
            "passed": s4.get("reused_from_m2") is False
            and _positive(seconds["S4"])
            and _positive(seconds["S2"]),
            "observed": {
                "train_time_seconds": seconds,
                "measured_s4_over_s2": (
                    float(seconds["S4"]) / float(seconds["S2"])
                    if _positive(seconds["S2"]) and _finite(seconds["S4"])
                    else None
                ),
                "the_multiplier_this_replaces": 2.5,
                "reused_from_m2": {rung: fit.get("reused_from_m2") for rung, fit in fits.items()},
            },
        },
    }


def build(
    result_path: Path = SMOKE_RESULT_PATH, declaration_path: Path = DECLARATION_PATH
) -> dict[str, Any]:
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    if not result_path.is_file():
        raise SystemExit(
            f"{result_path} is missing; there is no smoke to verify. The gate stays false."
        )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    dataset, _, regime = str(declaration["smoke_before_fanout"]["cell"]).partition("/")
    dataset, regime = dataset.strip(), regime.strip()
    if result.get("dataset") != dataset:
        raise SystemExit(
            f"the declaration smokes {dataset} but this result is for {result.get('dataset')!r}"
        )
    if result.get("seed") != declaration["smoke_before_fanout"]["seed"]:
        raise SystemExit(
            f"the declaration smokes seed {declaration['smoke_before_fanout']['seed']} but "
            f"this result is seed {result.get('seed')!r}"
        )

    items = declared_items(declaration)
    checks = check(declaration, result, regime)
    unchecked = sorted(set(items) - set(checks))
    if unchecked:
        raise SystemExit(f"declared items {unchecked} have no check; refusing to report a pass")

    failed = sorted(key for key, entry in checks.items() if not entry["passed"])
    return {
        "status": "M2B_SMOKE_VERIFICATION_COMPLETE",
        "verdict": FAILED if failed else PASSED,
        "cell": declaration["smoke_before_fanout"]["cell"],
        "rungs": list(declaration["smoke_before_fanout"]["rungs"]),
        "seed": declaration["smoke_before_fanout"]["seed"],
        "arm": DECLARED_UNIVERSAL_ARM,
        "runner_arm": RUNNER_UNIVERSAL_ARM,
        "structural_families_checked": list(ARM_STRUCTURAL_FAMILIES),
        "declared_items": len(items),
        "checked_items": len(checks),
        "passed_items": len(checks) - len(failed),
        "failed_items": failed,
        "the_denominator_comes_from_the_declaration": (
            "smoke_before_fanout.what_the_smoke_must_establish, matched item by item. An item "
            "this script cannot settle stops the verification rather than being dropped from "
            "the count, so 'n of n' cannot be reached by checking fewer things."
        ),
        "what_this_earns": (
            "launch_authorization.gates.engineering_smoke_passes and "
            "gates.parameter_accounting_matches, and nothing else. The cost gate is earned "
            "separately, once the measured S4 seconds have replaced the 2.5x multiplier and "
            "the projection has been recomputed against the filed ceiling."
        ),
        "items": {key: {"declared_as": items[key], **checks[key]} for key in items},
        "source_result": str(result_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify the M2B smoke against its declaration")
    parser.add_argument("--result", type=Path, default=SMOKE_RESULT_PATH)
    parser.add_argument("--declaration", type=Path, default=DECLARATION_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    verification = build(args.result, args.declaration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": verification["verdict"],
                "passed_items": verification["passed_items"],
                "declared_items": verification["declared_items"],
                "failed_items": verification["failed_items"],
                "measured_s4_over_s2": verification["items"]["s4_fit_time_measured"]["observed"][
                    "measured_s4_over_s2"
                ],
                "written": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if verification["verdict"] == PASSED else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
