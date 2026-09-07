#!/usr/bin/env python
"""Apply M2's frozen selection rule to the finished screen, mechanically.

configs/m2_qls_v2_freeze.yaml#universal_selection_rule was closed on 2026-09-07
BEFORE any M2 fit produced a number, and it says why this file exists: "The rule
is applied mechanically by a committed script (scripts/m2_selection_report.py)
whose tests assert the arithmetic on synthetic cases that exercise each clause
independently, so the verdict cannot be reached by reading a table and choosing
a framing."

So nothing here is a judgement call. Every threshold, every reference arm and
every aggregation weight is read out of the declaration; the only thing this
script contributes is arithmetic and a set of refusals.

What it refuses, and why each one is a way the verdict could go wrong quietly:

*   A missing cell. Fourteen cells were declared; a verdict computed on twelve
    is a different rule, not a partial answer, so an incomplete screen produces
    no verdict at all.
*   A reference arm derived from one source only. The rule names the QLS-CELL
    incumbent, and the matrix separately marks it with an
    ``_incumbent_`` status. Both are read and a disagreement stops the report --
    "the better of the two" is exactly the adaptive comparison the rule forbids,
    and a silently wrong reference is how it would arrive.
*   A reference number this script resolved for itself. Reused references come
    through scripts/m2_reuse_audit's own record resolver and are cross-checked
    against the recall the audit recorded, so the number the verdict rests on is
    the number the audit passed, not a second reading of the same file.
*   An unpaired comparison. The rule's per-cell clause claims both sides are
    seed-0 fits on the same validation prefix, the same holdout split and the
    same candidate contract. That is checked here per cell rather than trusted.
*   A candidate that is not the declared object: wrong arm name, wrong
    precomputed width, wrong parameter count, or a fit the runner did not mark
    new.

Reads only. Writes outputs/m2_qls_v2_freeze/selection_report.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from statistics import fmean
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.m2_reuse_audit import _m1a_record, _m1b_record  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
REUSE_AUDIT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "reuse_audit.json"
HEADLINE_DIR = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "selection_report.json"

CANDIDATE_ARM = "QLS-UNIVERSAL"
BASE_ARM = "BASE"
PP = 100.0


def _metric_key(declared: str) -> str:
    """``recall_at_5`` in the declaration is ``recall@5`` in a result's metrics.

    The declaration writes metric names in the form YAML keys take; the runner
    writes them in the form M1A has always written them. Mapping one onto the
    other here keeps the declared list authoritative -- the alternative is a
    second list in this file that can drift from it.
    """

    return declared.replace("_at_", "@")


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


def incumbent_arm(declaration: dict[str, Any], dataset: str, regime: str) -> str:
    """The reference arm for one cell, from qls_cell.map and nothing else.

    "The QLS-CELL incumbent arm for that cell if one exists, otherwise BASE."
    The map stores FEATURES, and an arm name is BASE plus those features in the
    catalog's own naming -- which is why a multi-feature entry stops the report
    rather than being composed here: no such arm was ever run, so the name this
    would invent would resolve to nothing.
    """

    entry = declaration["qls_cell"]["map"].get(dataset, {})
    cell = entry.get(regime) if isinstance(entry, dict) else None
    if not isinstance(cell, dict):
        # musique_clean carries a dataset-level status and no regime cells: it
        # has no trained evidence, so it has no incumbent and references BASE.
        return BASE_ARM
    features = list(cell.get("features") or [])
    if not features:
        return BASE_ARM
    if len(features) > 1:
        raise SystemExit(
            f"{dataset}/{regime}: qls_cell.map lists {features} as the incumbent, but no "
            "multi-feature arm was ever run -- the reference cannot be named"
        )
    return f"{BASE_ARM}+{features[0]}"


def _check_matrix_agrees(declaration: dict[str, Any], dataset: str, regime: str,
                         reference: str) -> str:
    """The matrix's own ``_incumbent_`` marker must name the same arm."""

    arms = declaration["m2_selection_matrix"]["cells"][dataset][regime]
    if reference not in arms:
        raise SystemExit(
            f"{dataset}/{regime}: the rule references {reference!r} but the matrix declares "
            f"{sorted(arms)} -- one of the two is wrong and the delta would be meaningless"
        )
    marked = sorted(arm for arm, status in arms.items() if "_incumbent_" in str(status))
    if marked and marked != [reference]:
        raise SystemExit(
            f"{dataset}/{regime}: qls_cell.map gives the incumbent as {reference!r} but the "
            f"matrix marks {marked} -- refusing rather than picking one"
        )
    if not marked and reference != BASE_ARM:
        raise SystemExit(
            f"{dataset}/{regime}: qls_cell.map gives an incumbent {reference!r} that the "
            "matrix marks nowhere"
        )
    return str(arms[reference])


def _panel(result: dict[str, Any], cell: dict[str, Any]) -> dict[str, Any]:
    """What has to match on both sides for the pair to be a paired comparison."""

    return {
        "data_fingerprint_sha256": result["data_fingerprint_sha256"],
        "queries": result["queries"],
        "split": result["split"],
        "holdout_fraction": result["holdout_fraction"],
        "train_queries": cell["train_queries"],
        "held_out_queries": cell["held_out_queries"],
        "candidate_contract_sha256": result["candidate_contract"]["observed_contract_sha256"],
    }


def _metrics(record: dict[str, Any], keys: list[str], where: str) -> dict[str, float]:
    metrics = record.get("metrics") or {}
    missing = [key for key in keys if key not in metrics]
    if missing:
        raise SystemExit(f"{where}: no {missing} in its metrics block")
    return {key: float(metrics[key]) for key in keys}


def _headline(dataset: str, headline_dir: Path) -> dict[str, Any] | None:
    path = headline_dir / f"{dataset}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "M2_QLS_V2_FREEZE_DATASET_COMPLETE":
        raise SystemExit(f"{path} carries status {payload.get('status')!r}, not a complete screen")
    if payload.get("dataset") != dataset:
        raise SystemExit(f"{path} records dataset {payload.get('dataset')!r}")
    return payload


def _reused_reference(dataset: str, regime: str, arm: str, audit: dict[str, Any],
                      keys: list[str]) -> tuple[dict[str, float], dict[str, Any], str]:
    """The reference the reuse audit already accepted, read through its resolver."""

    row = next(
        (fit for fit in audit["fits"]
         if (fit["dataset"], fit["regime"], fit["arm"]) == (dataset, regime, arm)),
        None,
    )
    if row is None:
        raise SystemExit(
            f"{dataset}/{regime}/{arm} is the declared reference but the reuse audit did not "
            "audit it; the verdict may not rest on an unaudited fit"
        )
    if not row.get("reusable"):
        raise SystemExit(
            f"{dataset}/{regime}/{arm} was refused by the reuse audit ({row.get('verdict')}); "
            "it has to be re-run before it can serve as a reference"
        )
    resolved = _m1b_record(dataset, regime, arm) or _m1a_record(dataset, regime, arm)
    result, record, source = resolved
    if source != row["source"]:
        raise SystemExit(
            f"{dataset}/{regime}/{arm}: the audit read {row['source']} but the resolver now "
            f"returns {source} -- the reference moved between the audit and this report"
        )
    metrics = _metrics(record, keys, source)
    primary = metrics[_metric_key(audit_primary := "recall_at_5")]
    if abs(primary - float(row["recall_at_5"])) > 1e-12:
        raise SystemExit(
            f"{dataset}/{regime}/{arm}: {audit_primary} is {primary} here but the reuse audit "
            f"recorded {row['recall_at_5']}"
        )
    cell = result["cells"][regime]
    return metrics, _panel(result, cell), source


def _new_reference(dataset: str, regime: str, arm: str, headline: dict[str, Any],
                   keys: list[str]) -> tuple[dict[str, float], dict[str, Any], str]:
    """A reference the matrix marks new -- musique_clean's BASE, which has no
    M1A fit to reuse because M1A never ran the dataset."""

    cell = headline["cells"][regime]
    record = cell["arms"].get(arm)
    if record is None:
        raise SystemExit(f"{dataset}/{regime}: the screen did not run its reference arm {arm!r}")
    source = f"outputs/m2_qls_v2_freeze/headline/{dataset}.json#cells.{regime}.arms.{arm}"
    return _metrics(record, keys, source), _panel(headline, cell), source


def _candidate(dataset: str, regime: str, headline: dict[str, Any], declaration: dict[str, Any],
               keys: list[str]) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    cell = headline["cells"][regime]
    record = cell["arms"].get(CANDIDATE_ARM)
    if record is None:
        raise SystemExit(f"{dataset}/{regime}: the screen has no {CANDIDATE_ARM} arm")
    schema = declaration["qls_universal"]["feature_schema"]
    counts = declaration["qls_universal"]["parameter_count"]
    identity = {
        "runner_arm": record.get("runner_arm"),
        "precomputed_width": record.get("precomputed_width"),
        "total_parameters": (record.get("parameters") or {}).get("total"),
        "seed": record.get("seed"),
        "matrix_status": record.get("matrix_status"),
    }
    expected = {
        "runner_arm": schema["arm_name"],
        "precomputed_width": schema["precomputed_width"],
        "total_parameters": counts["total"],
        "seed": 0,
        "matrix_status": "new",
    }
    wrong = {k: (identity[k], expected[k]) for k in expected if identity[k] != expected[k]}
    if wrong:
        raise SystemExit(
            f"{dataset}/{regime}: the fitted arm is not the declared candidate -- "
            f"{ {k: f'got {g!r}, declared {e!r}' for k, (g, e) in wrong.items()} }"
        )
    source = (
        f"outputs/m2_qls_v2_freeze/headline/{dataset}.json"
        f"#cells.{regime}.arms.{CANDIDATE_ARM}"
    )
    return _metrics(record, keys, source), _panel(headline, cell), identity


def cell_rows(declaration: dict[str, Any], audit: dict[str, Any], headlines: dict[str, dict],
              keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for dataset, regimes in declaration["m2_selection_matrix"]["cells"].items():
        headline = headlines.get(dataset)
        for regime in regimes:
            reference = incumbent_arm(declaration, dataset, regime)
            status = _check_matrix_agrees(declaration, dataset, regime, reference)
            if headline is None:
                rows.append({"dataset": dataset, "regime": regime, "reference_arm": reference,
                             "present": False, "why": "no completed screen for this dataset"})
                continue
            if str(status).startswith("reuse_"):
                ref_metrics, ref_panel, ref_source = _reused_reference(
                    dataset, regime, reference, audit, keys
                )
            else:
                ref_metrics, ref_panel, ref_source = _new_reference(
                    dataset, regime, reference, headline, keys
                )
            cand_metrics, cand_panel, identity = _candidate(
                dataset, regime, headline, declaration, keys
            )
            unpaired = {
                field: (ref_panel[field], cand_panel[field])
                for field in ref_panel
                if ref_panel[field] != cand_panel[field]
            }
            if unpaired:
                raise SystemExit(
                    f"{dataset}/{regime}: reference and candidate were not fit on the same "
                    f"panel -- {unpaired}. The rule's per-cell quantity is a paired delta."
                )
            rows.append({
                "dataset": dataset,
                "regime": regime,
                "present": True,
                "reference_arm": reference,
                "reference_matrix_status": status,
                "reference_source": ref_source,
                "candidate_arm": CANDIDATE_ARM,
                "candidate_identity": identity,
                "panel": ref_panel,
                "reference_metrics": ref_metrics,
                "candidate_metrics": cand_metrics,
                "deltas_pp": {
                    key: (cand_metrics[key] - ref_metrics[key]) * PP for key in keys
                },
            })
    return rows


def verdict(declaration: dict[str, Any], rows: list[dict[str, Any]],
            primary: str) -> dict[str, Any]:
    """The three clauses, then the label the declaration defines for the result."""

    rule = declaration["universal_selection_rule"]
    thresholds = rule["thresholds"]
    macro_tolerance = float(thresholds["macro_tolerance_pp"])
    dataset_floor = float(thresholds["dataset_material_regression_pp"])
    cell_floor = float(thresholds["cell_material_regression_pp"])

    per_dataset: dict[str, list[float]] = {}
    for row in rows:
        per_dataset.setdefault(row["dataset"], []).append(row["deltas_pp"][primary])
    dataset_delta = {name: fmean(values) for name, values in per_dataset.items()}
    macro_delta = fmean(dataset_delta.values())

    declared_counts = rule["per_dataset"]["cells_per_dataset"]
    wrong = {
        name: (len(values), declared_counts.get(name))
        for name, values in per_dataset.items()
        if len(values) != declared_counts.get(name)
    }
    if wrong:
        raise SystemExit(
            f"the rule declares cells_per_dataset {declared_counts} but this report has "
            f"{wrong} -- the per-dataset mean would be over a different set of cells"
        )

    failing_cells = sorted(
        ({"dataset": row["dataset"], "regime": row["regime"],
          "delta_pp": row["deltas_pp"][primary]}
         for row in rows if row["deltas_pp"][primary] < cell_floor),
        key=lambda item: item["delta_pp"],
    )
    failing_datasets = sorted(
        ({"dataset": name, "delta_pp": value}
         for name, value in dataset_delta.items() if value < dataset_floor),
        key=lambda item: item["delta_pp"],
    )
    clauses = {
        "1_macro_within_tolerance": macro_delta >= macro_tolerance,
        "2_no_dataset_material_regression": not failing_datasets,
        "3_no_cell_material_regression": not failing_cells,
    }

    def _in_band(value: float) -> bool:
        # The gray band the declaration names: at or above the material floor,
        # but below the macro tolerance.
        return cell_floor <= value < macro_tolerance

    gray_reasons = []
    if _in_band(macro_delta):
        gray_reasons.append(f"macro_delta {macro_delta:.4f}pp sits in the band")
    gray_reasons.extend(
        f"{name} dataset_delta {value:.4f}pp sits in the band"
        for name, value in sorted(dataset_delta.items()) if _in_band(value)
    )
    gray_reasons.extend(
        f"{row['dataset']}/{row['regime']} cell_delta {row['deltas_pp'][primary]:.4f}pp "
        "sits in the band"
        for row in rows if _in_band(row["deltas_pp"][primary])
    )

    if not (clauses["2_no_dataset_material_regression"]
            and clauses["3_no_cell_material_regression"]):
        label = "NOT_ADVANCED_AS_FILED"
    elif gray_reasons:
        label = "GRAY_PENDING_THREE_SEED"
    elif all(clauses.values()):
        label = "ADVANCE_QLS_UNIVERSAL"
    else:  # pragma: no cover -- unreachable while the floors bound the mean
        raise SystemExit(
            f"clauses {clauses} with macro_delta {macro_delta} match no declared outcome "
            "label; refusing to invent one"
        )

    return {
        "outcome": label,
        "outcome_definition": rule["outcome_labels"][label],
        "primary_metric": rule["primary_metric"],
        "macro_delta_pp": macro_delta,
        "dataset_delta_pp": dict(sorted(dataset_delta.items())),
        "cell_delta_pp": {
            f"{row['dataset']}/{row['regime']}": row["deltas_pp"][primary] for row in rows
        },
        "clauses": clauses,
        "advancement_condition": rule["advancement_condition"],
        "thresholds_pp": {
            "macro_tolerance": macro_tolerance,
            "dataset_material_regression": dataset_floor,
            "cell_material_regression": cell_floor,
        },
        "failing_cells": failing_cells,
        "failing_datasets": failing_datasets,
        "gray_band_members": gray_reasons,
        "advances": label == "ADVANCE_QLS_UNIVERSAL",
    }


def secondary_summary(rows: list[dict[str, Any]], keys: list[str],
                      primary: str) -> dict[str, Any]:
    """Every declared diagnostic at every level, and never a deciding number."""

    summary: dict[str, Any] = {}
    for key in keys:
        if key == primary:
            continue
        per_dataset: dict[str, list[float]] = {}
        for row in rows:
            per_dataset.setdefault(row["dataset"], []).append(row["deltas_pp"][key])
        dataset_delta = {name: fmean(v) for name, v in per_dataset.items()}
        summary[key] = {
            "macro_delta_pp": fmean(dataset_delta.values()),
            "dataset_delta_pp": dict(sorted(dataset_delta.items())),
            "cell_delta_pp": {
                f"{row['dataset']}/{row['regime']}": row["deltas_pp"][key] for row in rows
            },
        }
    return summary


def systems_and_parameter_table(headlines: dict[str, dict], rows: list[dict[str, Any]],
                                ) -> dict[str, Any]:
    """What the selected object costs, per cell.

    m2_output.contents names this table alongside the deltas: params, train
    seconds, feature-build p50/p95/p99, peak VRAM/RSS. M2's claim is a
    parameter-efficiency claim as much as a recall claim, and M4's Pareto table
    is built from exactly these columns, so the numbers belong in the report
    rather than only in the six per-dataset result files.

    Candidate side only, deliberately. The reference fits were run in M1A and
    M1B, in other launches on other days; putting their seconds and megabytes
    in the same table would read as a cost comparison that nothing here
    controls for. Parameter counts are the exception -- they are a property of
    the specification, not of the run -- so the declared reference-arm width is
    carried and the timings are not.
    """

    table: dict[str, Any] = {
        "measured_for": CANDIDATE_ARM,
        "why_candidate_only": (
            "Reference fits were run in M1A/M1B on other days and other launches; their "
            "timings are not controlled against these and are not reported here as if "
            "they were. Parameter counts are a property of the specification and are "
            "reported for both."
        ),
        "cells": {},
    }
    for row in rows:
        if not row["present"]:
            continue
        dataset, regime = row["dataset"], row["regime"]
        cell = headlines[dataset]["cells"][regime]
        record = cell["arms"][CANDIDATE_ARM]
        parameters = record.get("parameters") or {}
        training = record.get("training") or {}
        inference = record.get("inference") or {}
        build_latency = cell.get("uncached_feature_build_latency_ms") or {}
        table["cells"][f"{dataset}/{regime}"] = {
            "parameters": {
                "total": parameters.get("total"),
                "semantic": parameters.get("semantic"),
                "scorer": parameters.get("scorer"),
            },
            "train_seconds": training.get("training_seconds"),
            "uncached_feature_build_ms": {
                level: build_latency.get(level) for level in ("p50", "p95", "p99")
            },
            "peak_train_vram_mb": training.get("peak_training_gpu_memory_mb_total"),
            "peak_train_rss_mb": (record.get("systems") or {}).get("peak_train_rss_mb"),
            "peak_train_rss_mb_provenance": (record.get("systems") or {}).get(
                "peak_train_rss_mb_provenance"
            ),
            "inference_latency_ms_per_query": inference.get("latency_ms_per_query"),
            "train_queries": cell.get("train_queries"),
            "held_out_queries": cell.get("held_out_queries"),
        }
    totals = [c["parameters"]["total"] for c in table["cells"].values()]
    table["parameters_identical_in_every_cell"] = len(set(totals)) == 1
    table["total_parameters"] = totals[0] if table["parameters_identical_in_every_cell"] else None
    table["total_train_seconds"] = sum(
        c["train_seconds"] for c in table["cells"].values() if c["train_seconds"] is not None
    )
    return table


def build(headline_dir: Path = HEADLINE_DIR) -> dict[str, Any]:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    audit = json.loads(REUSE_AUDIT_PATH.read_text(encoding="utf-8"))
    rule = declaration["universal_selection_rule"]

    primary = _metric_key(rule["primary_metric"])
    keys = [primary] + [
        _metric_key(name) for name in rule["secondary_diagnostics"]["reported_always"]
        if _metric_key(name) != primary
    ]

    cells = declaration["m2_selection_matrix"]["cells"]
    headlines = {dataset: _headline(dataset, headline_dir) for dataset in cells}
    rows = cell_rows(declaration, audit, headlines, keys)
    declared_cells = sum(len(regimes) for regimes in cells.values())
    complete = [row for row in rows if row["present"]]
    missing = [f"{row['dataset']}/{row['regime']}" for row in rows if not row["present"]]

    report: dict[str, Any] = {
        "status": "M2_SELECTION_COMPLETE" if not missing else "M2_SELECTION_INCOMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml#universal_selection_rule",
        "rule_frozen": rule["status"],
        "rule_ruled_by": rule["ruled_by"],
        "git_head_now": _git_head(),
        "seeds": "seed 0 on both sides; this is a one-seed screening scale",
        "declared_cells": declared_cells,
        "cells_reported": len(complete),
        "cells_missing": missing,
        # Carried into the report unchanged, per m2_output.contents. It is the
        # map every cell's reference arm was resolved from, so a reader can see
        # what each delta was measured against without opening the declaration.
        "qls_cell_incumbent_map": {
            f"{row['dataset']}/{row['regime']}": {
                "reference_arm": row["reference_arm"],
                "matrix_status": row.get("reference_matrix_status"),
            }
            for row in rows
        },
        "cells": rows,
    }
    if missing:
        report["verdict"] = None
        report["why_no_verdict"] = (
            f"{len(missing)} of {declared_cells} declared cells have no result. The rule "
            "aggregates over the declared cells; a verdict on a subset is a different rule, "
            "not a partial answer."
        )
        return report

    report["verdict"] = verdict(declaration, complete, primary)
    report["secondary_diagnostics"] = secondary_summary(complete, keys, primary)
    report["systems_and_parameter_table"] = systems_and_parameter_table(headlines, complete)
    report["secondary_diagnostics_never_decide"] = (
        rule["secondary_diagnostics"]["never_deciding"]
    )
    report["source_commits"] = sorted({
        payload["provenance"]["source_commit"]
        for payload in headlines.values() if payload
    })
    report["what_this_is_not"] = (
        "A confirmation. Every number here is one seed on the development substrate the "
        "2026-09-07 course decision set aside for method selection; nothing in it is a final "
        "result and no canonical CRAG data was read."
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headline-dir", type=Path, default=HEADLINE_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)

    report = build(args.headline_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    summary = {
        "status": report["status"],
        "cells_reported": report["cells_reported"],
        "cells_missing": report["cells_missing"],
    }
    if report["verdict"] is not None:
        summary.update({
            "outcome": report["verdict"]["outcome"],
            "macro_delta_pp": round(report["verdict"]["macro_delta_pp"], 4),
            "dataset_delta_pp": {
                k: round(v, 4) for k, v in report["verdict"]["dataset_delta_pp"].items()
            },
            "clauses": report["verdict"]["clauses"],
            "failing_cells": report["verdict"]["failing_cells"],
            "gray_band_members": report["verdict"]["gray_band_members"],
        })
    print(json.dumps(summary, indent=2))
    return 0 if report["status"] == "M2_SELECTION_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
