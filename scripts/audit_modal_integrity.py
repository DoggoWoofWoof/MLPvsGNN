#!/usr/bin/env python
"""Verify persisted Modal checkpoints without reading partial effectiveness metrics."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import modal
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_modal_progress import (
    GATED,
    PACKAGES,
    REPO_ROOT,
    _condition_key,
    _expected_keys,
    _is_gated,
)

MODELS = ("sa_mlp", "seed_aware_gnn")
STORAGE_PREFIX = "/root/message-passing-retrieval/storage/"


def _load_context(package: str) -> dict[str, Any]:
    config_path = REPO_ROOT / "configs" / f"{package}.yaml"
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    confirmations = {
        dataset: json.loads((REPO_ROOT / spec["confirmation"]).read_text(encoding="utf-8"))
        for dataset, spec in config["datasets"].items()
    }
    return {
        "config": config,
        "confirmations": confirmations,
        "protocol_file": str(config_path.relative_to(REPO_ROOT)),
        "protocol_file_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


async def _read_bytes(volume: modal.Volume, path: str) -> bytes:
    return b"".join([chunk async for chunk in volume.read_file.aio(path)])


async def _remote_sha256(
    volume: modal.Volume,
    path: str,
    semaphore: asyncio.Semaphore,
) -> str:
    if not path.startswith(STORAGE_PREFIX):
        raise ValueError(f"Artifact path is outside the registered storage root: {path}")
    last_error: Exception | None = None
    for attempt in range(5):
        digest = hashlib.sha256()
        try:
            async with semaphore:
                async for chunk in volume.read_file.aio(path.removeprefix(STORAGE_PREFIX)):
                    digest.update(chunk)
            return digest.hexdigest()
        except Exception as exc:  # noqa: BLE001 - retry transient Modal storage errors
            last_error = exc
            if attempt < 4:
                await asyncio.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def _verify_candidate_proof(
    payload: dict[str, Any],
    confirmation: dict[str, Any],
    package: str,
) -> list[str]:
    errors = []
    key = "source_candidate_contract" if package == "candidate_budget" else "candidate_contract"
    proof = payload.get(key)
    if not isinstance(proof, dict):
        return [f"missing {key}"]
    expected_proof_hash = proof.get("proof_sha256")
    unhashed = copy.deepcopy(proof)
    unhashed.pop("proof_sha256", None)
    observed_proof_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if expected_proof_hash != observed_proof_hash:
        errors.append("candidate proof SHA-256 mismatch")
    if proof.get("status") != "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE":
        errors.append("candidate proof status is not bit-exact")
    if proof.get("expected_contract_sha256") != confirmation["baseline"][
        "candidate_contract_sha256"
    ]:
        errors.append("candidate contract differs from sealed baseline")
    if proof.get("observed_contract_sha256") != proof.get("expected_contract_sha256"):
        errors.append("observed candidate contract differs from sealed baseline")

    # 2Wiki and MuSiQue require a compatibility reconstruction and therefore
    # carry an independently sealed candidate-order digest. The other four
    # datasets are canonical already: their observed frozen contract is the
    # independently sealed baseline contract, which includes candidate order.
    confirmation_proof = confirmation.get("comparison_contract", {}).get(
        "candidate_compatibility_proof"
    )
    if confirmation_proof is not None and proof.get(
        "candidate_id_order_sha256"
    ) != confirmation_proof.get("candidate_id_order_sha256"):
        errors.append("candidate ID/order hash differs from sealed confirmation")
    return errors


def _verify_payload_contract(
    package: str,
    payload: dict[str, Any],
    context: dict[str, Any],
) -> list[str]:
    errors = []
    dataset = payload.get("dataset")
    config = context["config"]
    if dataset not in config["datasets"]:
        return ["unregistered dataset"]
    confirmation = context["confirmations"][dataset]
    if payload.get("data_fingerprint_sha256") != confirmation.get(
        "data_fingerprint_sha256"
    ):
        errors.append("dataset fingerprint mismatch")
    errors.extend(_verify_candidate_proof(payload, confirmation, package))
    result_config = payload.get("config", {})
    spec = config["datasets"][dataset]
    if result_config.get("selected_gnn") != spec["selected_gnn"]:
        errors.append("selected GNN differs from protocol")
    if package in {"edge_provenance", "candidate_budget"}:
        training = config["training"]
        for field in (
            "seeds",
            "epochs",
            "batch_size",
            "layers",
            "dropout",
            "temperature",
            "learning_rate",
            "weight_decay",
            "ks",
            "inference_repeats",
        ):
            if result_config.get(field) != training[field]:
                errors.append(f"training field differs: {field}")
        if payload.get("data", {}).get("test_query_order_sha256") != confirmation[
            "data"
        ]["test_query_order_sha256"]:
            errors.append("test query-order hash mismatch")
    if package == "edge_provenance":
        family = payload.get("family")
        if family not in config["trained_families"]:
            errors.append("unregistered graph family")
        provenance = payload.get("edge_provenance", {})
        if provenance.get("selected_family") != family:
            errors.append("edge-provenance family mismatch")
        if provenance.get("selected_edge_key_sha256") != payload.get("edge_key_sha256"):
            errors.append("edge-set SHA-256 mismatch")
    elif package == "candidate_budget":
        budget = int(payload.get("budget", -1))
        if budget not in config["candidate_contract"]["budgets"]:
            errors.append("unregistered candidate budget")
        if result_config.get("budget") != budget:
            errors.append("candidate budget differs from run config")
        if result_config.get("rrf_constant") != config["candidate_contract"][
            "rrf_constant"
        ]:
            errors.append("RRF constant differs from protocol")
        if not isinstance(payload.get("budget_candidate_contract_sha256"), str):
            errors.append("missing budget candidate-contract hash")
    elif package == "phase_confirmation":
        training = config["training"]
        if sorted(result_config.get("seeds") or []) != sorted(training["seeds"]):
            errors.append("confirmation seed set differs")
        for field in ("epochs", "batch_size", "learning_rate", "weight_decay", "ks"):
            if result_config.get(field) != training[field]:
                errors.append(f"confirmation training field differs: {field}")
        axis = payload.get("axis")
        rate = float(payload.get("rate", -1))
        registered = config["axes"].get(axis, {}).get("rates", [])
        if axis not in config["axes"] or rate not in map(float, registered):
            errors.append("unregistered confirmation cell")
        if rate <= 0.0:
            errors.append("clean rate is not a confirmation cell")
        if result_config.get("perturbation_seed") != config["axes"].get(axis, {}).get(
            "perturbation_seed"
        ):
            errors.append("perturbation seed differs")
        if payload.get("data", {}).get("test_query_order_sha256") != confirmation[
            "data"
        ]["test_query_order_sha256"]:
            errors.append("test query-order hash mismatch")
        contract = payload.get("confirmation_contract", {})
        # The gate this package exists to respect: the rate under test was
        # chosen from validation alone, and seed 0 reused the screen checkpoint
        # without any test metric having been consulted to select it.
        if contract.get("test_selected_rate") is not False:
            errors.append("confirmation rate was selected using test outcomes")
        if contract.get("selected_by_locked_validation_only_rule") is not True:
            errors.append("confirmation rate did not come from the locked rule")
        if (
            contract.get("seed_zero_validation_checkpoint_reused_without_test_peeking")
            is not True
        ):
            errors.append("seed-0 checkpoint provenance not asserted")
    else:
        training = config["training"]
        if result_config.get("training_seed") != training["seed"]:
            errors.append("phase training seed differs")
        for field in (
            "epochs",
            "batch_size",
            "layers",
            "dropout",
            "temperature",
            "learning_rate",
            "weight_decay",
            "ks",
        ):
            if result_config.get(field) != training[field]:
                errors.append(f"phase training field differs: {field}")
        axis = payload.get("axis")
        rate = float(payload.get("rate", -1))
        if axis not in config["axes"] or rate not in map(float, config["axes"][axis]["rates"]):
            errors.append("unregistered phase cell")
        if result_config.get("perturbation_seed") != config["perturbation_seeds"].get(axis):
            errors.append("perturbation seed differs")
        screen = payload.get("screen_contract", {})
        if screen.get("test_metrics_computed") is not False:
            errors.append("validation screen accessed test metrics")
    return errors


def _checkpoint_records(package: str, payload: dict[str, Any]) -> list[tuple[str, int, dict[str, Any]]]:
    records = []
    for model in MODELS:
        model_record = payload.get("models", {}).get(model, {})
        if package == "phase_screen":
            if model_record:
                records.append((model, 0, model_record))
        else:
            for seed, record in model_record.get("seeds", {}).items():
                records.append((model, int(seed), record))
    return records


async def _verify_condition(
    volume: modal.Volume,
    package: str,
    path: str,
    payload: dict[str, Any],
    context: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    errors = _verify_payload_contract(package, payload, context)
    expected_seeds = [0] if package == "phase_screen" else [0, 1, 2, 3, 4]
    seed_status = {
        model: {str(seed): "MISSING" for seed in expected_seeds} for model in MODELS
    }
    checkpoint_checks = []
    for model, seed, record in _checkpoint_records(package, payload):
        checkpoint_path = record.get("checkpoint_path")
        expected_hash = record.get("checkpoint_file_sha256")
        check = {
            "model": model,
            "seed": seed,
            "path": checkpoint_path,
            "expected_sha256": expected_hash,
        }
        if not checkpoint_path or not expected_hash:
            check["verified"] = False
            check["error"] = "missing checkpoint path or SHA-256"
            seed_status[model][str(seed)] = "INVALID"
            errors.append(f"{model}/seed_{seed} checkpoint metadata missing")
        else:
            try:
                observed = await _remote_sha256(volume, checkpoint_path, semaphore)
                check["observed_sha256"] = observed
                check["verified"] = observed == expected_hash
            except Exception as exc:  # noqa: BLE001 - audit must classify every corrupt path
                check["verified"] = False
                check["error"] = f"{type(exc).__name__}: {exc}"
            if check["verified"]:
                seed_status[model][str(seed)] = "COMPLETE"
            else:
                seed_status[model][str(seed)] = "INVALID"
                errors.append(f"{model}/seed_{seed} checkpoint failed integrity")
        checkpoint_checks.append(check)
    query_metrics = {"applicable": package != "phase_screen", "verified": None}
    if payload.get("status") == PACKAGES[package]["complete_status"] and package != "phase_screen":
        packed = payload.get("query_metrics", {})
        try:
            observed = await _remote_sha256(volume, packed["path"], semaphore)
            query_metrics.update(
                {
                    "path": packed["path"],
                    "expected_sha256": packed["sha256"],
                    "observed_sha256": observed,
                    "verified": observed == packed["sha256"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - classify incomplete remote artifact
            query_metrics.update(
                {"verified": False, "error": f"{type(exc).__name__}: {exc}"}
            )
        if not query_metrics["verified"]:
            errors.append("packed query metrics failed integrity")
    all_seed_cells = all(
        status == "COMPLETE"
        for model_status in seed_status.values()
        for status in model_status.values()
    )
    final_status = payload.get("status") == PACKAGES[package]["complete_status"]
    if errors:
        classification = "INVALID"
    elif final_status and all_seed_cells and query_metrics["verified"] is not False:
        classification = "COMPLETE"
    else:
        classification = "PARTIAL / RESUMABLE"
    return {
        "condition": _condition_key(package, payload),
        "classification": classification,
        "result_path": path,
        "result_status": payload.get("status"),
        "protocol_file": context["protocol_file"],
        "protocol_file_sha256": context["protocol_file_sha256"],
        "result_embedded_protocol_file_sha256": False,
        "candidate_contract_verified": not any("candidate" in error for error in errors),
        "query_order_verified": not any("query-order" in error for error in errors),
        "seed_status": seed_status,
        "checkpoint_checks": checkpoint_checks,
        "query_metrics": query_metrics,
        "errors": errors,
    }


async def audit_integrity(volume_name: str) -> dict[str, Any]:
    volume = modal.Volume.from_name(volume_name, create_if_missing=False)
    semaphore = asyncio.Semaphore(8)
    output = {}
    for package, spec in PACKAGES.items():
        if _is_gated(package):
            output[package] = GATED
            continue
        context = _load_context(package)
        paths = []
        try:
            async for entry in volume.iterdir.aio(spec["path"], recursive=True):
                if entry.path.endswith("/result.json"):
                    paths.append(entry.path)
        except modal.exception.NotFoundError:
            paths = []
        payloads = await asyncio.gather(*(_read_bytes(volume, path) for path in sorted(paths)))
        decoded = [json.loads(value) for value in payloads]
        conditions = await asyncio.gather(
            *(
                _verify_condition(
                    volume,
                    package,
                    path,
                    payload,
                    context,
                    semaphore,
                )
                for path, payload in zip(sorted(paths), decoded)
            )
        )
        found = {row["condition"] for row in conditions}
        expected = _expected_keys(package)
        conditions.extend(
            {
                "condition": key,
                "classification": "MISSING",
                "seed_status": {
                    model: {
                        str(seed): "MISSING"
                        for seed in ([0] if package == "phase_screen" else range(5))
                    }
                    for model in MODELS
                },
                "errors": [],
            }
            for key in sorted(expected - found)
        )
        conditions.sort(key=lambda row: row["condition"])
        counts = {
            status: sum(row["classification"] == status for row in conditions)
            for status in ("COMPLETE", "PARTIAL / RESUMABLE", "MISSING", "INVALID")
        }
        completed_units = sum(
            status == "COMPLETE"
            for row in conditions
            for model in MODELS
            for status in row["seed_status"][model].values()
        )
        expected_units = len(expected) * len(MODELS) * (1 if package == "phase_screen" else 5)
        output[package] = {
            "condition_counts": counts,
            "completed_gpu_work_units": completed_units,
            "expected_gpu_work_units": expected_units,
            "remaining_gpu_work_units": expected_units - completed_units,
            "conditions": conditions,
        }
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--volume", default="message-passing-retrieval-data")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "modal_integrity_audit.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(audit_integrity(args.volume))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {
        package: {
            "condition_counts": record["condition_counts"],
            "completed_gpu_work_units": record["completed_gpu_work_units"],
            "expected_gpu_work_units": record["expected_gpu_work_units"],
            "remaining_gpu_work_units": record["remaining_gpu_work_units"],
        }
        for package, record in result.items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
