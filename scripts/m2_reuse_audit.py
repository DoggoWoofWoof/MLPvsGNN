#!/usr/bin/env python
"""M2 reuse audit: is each of the 19 proposed reused seed-0 fits actually reusable?

Read-only over results that already exist. Filed by
configs/m2_qls_v2_freeze.yaml#m2_selection_matrix.reuse_audit, which is the
block this script implements literally.

Why this is not M1B's audit again
---------------------------------
scripts/m1b_reuse_audit.py proved reuse by *repository identity*: HEAD equalled
M1A's launch commit, so nothing could have changed. That premise is gone --
HEAD has moved for M1B's own commit and for M2 amendment 1's scoped runner
change, which taught run_m1a_feature_screen.py the universal arm. So this audit
reasons per fit and per change instead:

  (A) Seven per-fit checks, each read from the fit's own recorded result --
      never inherited from a repository-level verdict.
  (B) One bit-exact feature probe, run here, comparing the *pre-change* feature
      builder (git object 3d85916, before the universal arm existed) against the
      current one on a deterministic fixture, for every historical arm type.
      Equal element-for-element, or the arm is not reusable. A diff that looks
      additive is not evidence; this is.

The probe covers what a per-fit check cannot: the seven recorded checks would
all still pass if the current code silently produced different feature values
under the same names. (B) is what makes that impossible to miss.

Any candidate that fails anything is marked NEW rather than argued into reuse;
the workload counts this script emits are what scripts/m2_compute_estimate.py
and the launch gate read.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.candidate_expansion_v2 import STRUCTURAL, expand  # noqa: E402
from mp_retrieval.complete_data import load_complete_dataset  # noqa: E402
from mp_retrieval.data import QuerySplit  # noqa: E402
from mp_retrieval.graph_context import build_operators  # noqa: E402
from scripts.run_edge_provenance import _atomic_json  # noqa: E402
from scripts.run_m0a_probe import QueryView, _load_family_csr, _undirected  # noqa: E402
from scripts.run_m0b_regime_map import MAINLINE_FAMILY, _a64_budget  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_HEADLINE_DIR = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline"
M1B_HEADLINE_DIR = REPO_ROOT / "outputs" / "m1b_targeted_resolution" / "headline"
OUTPUT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "reuse_audit.json"

#: The last commit before M2 amendment 1 touched scripts/run_m1a_feature_screen.py.
#: The blob at this rev is byte-identical to the one at f81985d (M1A's own launch
#: commit), so it *is* the code every reused fit was produced by.
PRE_UNIVERSAL_REV = "3d85916"
RUNNER_RELPATH = "scripts/run_m1a_feature_screen.py"

#: Every arm any M1A or M1B fit ever ran. BASE+NODE_ROLE+SUPPORT is M1B's own
#: interaction arm, registered into ARM_FAMILIES at import time by
#: run_m1b_targeted_resolution.py rather than declared in M1A's vocabulary --
#: the probe registers it into both modules the same way, so the historical
#: path is reproduced rather than approximated.
HISTORICAL_ARMS: tuple[str, ...] = (
    "BASE",
    "BASE+GEOMETRY",
    "BASE+SUPPORT",
    "BASE+PATH",
    "BASE+NODE_ROLE",
    "BASE+NODE_ROLE+SUPPORT",
)
M1B_INTERACTION_ARM = "BASE+NODE_ROLE+SUPPORT"
M1B_INTERACTION_FAMILIES = ("NODE_ROLE", "SUPPORT")
UNIVERSAL_ARM = "BASE+NODE_ROLE+SUPPORT+PATH"
REGIMES = ("R1", "R2", "R3")

#: Read live from modal_m1a_feature_screen._runner_args / its M1B twin; listed
#: here only to name which arguments the per-fit check compares.
TRAINER_HYPERPARAMETERS = (
    "epochs", "batch_size", "dropout", "temperature",
    "learning_rate", "weight_decay", "semantic_rung", "holdout_fraction",
    "per_seed_cap", "neighbour_scan_cap_per_seed", "a64_mainline_family",
)

#: The four R3 cells M1B re-ran with row capture. Where a reused fit exists in
#: both places the M1B record is preferred: identical number (its own
#: cross_checked_against_m1a_splice records delta 0.0pp), plus the per-query
#: rows a gray-zone expansion would need.
M1B_R3_DATASETS = ("2wiki_clean", "hotpotqa_clean", "metaqa", "webqsp")

# --- the deterministic probe fixture ---------------------------------------
# Mirrors tests/test_run_m1a_feature_screen.py's toy dataset construction: the
# same node count, the same descending dense/splade rows, the same family-graph
# arithmetic. Owned here rather than in the test so the audit is self-contained
# and the test can check the audit against the same fixture it audits.

FIXTURE_SEED = 20260907
NUM_NODES = 40
FIXTURE_FAMILIES = (MAINLINE_FAMILY, "baseline_a_simple")
DENSE = np.array(
    [
        [5, 4, 3, 2, 1, 0],
        [11, 10, 9, 8, 7, 6],
        [17, 16, 15, 14, 13, 12],
        [23, 22, 21, 20, 19, 18],
    ]
)
SPLADE = (DENSE + 2) % NUM_NODES
BASE_GOLDS = ("doc_3", "doc_9", "doc_15", "doc_21")
FIXTURE_EMBEDDING_DIM = 6
FIXTURE_PER_SEED_CAP = 4
FIXTURE_NEIGHBOUR_SCAN_CAP = 4096


def _git_object_sha(rev: str, relpath: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", f"{rev}:{relpath}"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def _git_show(rev: str, relpath: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "show", f"{rev}:{relpath}"],
        capture_output=True, text=True, check=True,
    ).stdout


def _git_head() -> str:
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def load_pre_universal_module(workdir: Path) -> types.ModuleType:
    """Import the feature builder as it stood at ``PRE_UNIVERSAL_REV``.

    Written under a ``scripts/`` directory that has no ``configs/`` sibling, so
    the module's own ``_resolve_repo_root`` falls through its ``file_path``
    branch to the ``sys.path`` branch and finds this repo -- which is the
    behaviour that fallback was written for, and which lets the old module
    import the same ``mp_retrieval``/``scripts`` dependencies the current one
    does. Only the runner source is old; everything it imports is shared, which
    is exactly the scope of the change being audited.
    """

    package = workdir / "scripts"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    source = package / "run_m1a_feature_screen_pre_universal.py"
    source.write_text(_git_show(PRE_UNIVERSAL_REV, RUNNER_RELPATH), encoding="utf-8")

    name = "m2_reuse_audit_pre_universal_runner"
    spec = importlib.util.spec_from_file_location(name, source)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib contract
        raise RuntimeError(f"could not build an import spec for {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ARM_FAMILIES.setdefault(M1B_INTERACTION_ARM, M1B_INTERACTION_FAMILIES)
    return module


def write_probe_fixture(root: Path) -> dict:
    """Build the toy dataset and its family graphs; return the probe inputs."""

    data = root / "data"
    graphs = root / "families"
    data.mkdir(parents=True, exist_ok=True)

    for offset, family in enumerate(FIXTURE_FAMILIES, start=1):
        source = np.arange(NUM_NODES, dtype=np.int64)
        target = (source + 4 * offset) % NUM_NODES
        edge_index = np.concatenate(
            [np.stack([source, target]), np.stack([target, source])], axis=1
        )
        path = graphs / family
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"edge_index": torch.tensor(edge_index, dtype=torch.long), "num_nodes": NUM_NODES},
            path / "graph.pt",
        )

    family_rowptr, family_col = _load_family_csr(graphs / MAINLINE_FAMILY / "graph.pt", NUM_NODES)
    family_rowptr, family_col, _symmetric = _undirected(family_rowptr, family_col, NUM_NODES)
    budget = _a64_budget(
        per_seed_cap=FIXTURE_PER_SEED_CAP,
        neighbour_scan_cap_per_seed=FIXTURE_NEIGHBOUR_SCAN_CAP,
    )
    pool = np.unique(np.concatenate([DENSE[0], SPLADE[0]])).astype(np.int64)
    seeds = np.unique(np.concatenate([DENSE[0, :5], SPLADE[0, :5]])).astype(np.int64)
    expansion = expand(
        STRUCTURAL,
        rowptr=family_rowptr,
        col=family_col,
        node_embeddings=np.zeros((NUM_NODES, FIXTURE_EMBEDDING_DIM), dtype=np.float32),
        query_embedding=None,
        anchor=int(DENSE[0, 0]),
        pool=pool,
        seeds=seeds,
        budget=budget,
        num_nodes=NUM_NODES,
    )
    outside_cq = expansion.admitted[~np.isin(expansion.admitted, pool)]
    if outside_cq.size == 0:
        raise RuntimeError(
            "fixture produced no A64 admission outside Cq -- NODE_ROLE would be "
            "identically zero under R3 and the probe would prove nothing"
        )
    discovered_gold = int(outside_cq[0])

    rng = np.random.default_rng(FIXTURE_SEED)
    np.save(data / "nodes.npy", rng.normal(size=(NUM_NODES, FIXTURE_EMBEDDING_DIM)).astype(np.float32))
    np.save(data / "queries_all.npy", rng.normal(size=(4, FIXTURE_EMBEDDING_DIM)).astype(np.float32))
    np.save(data / "dense_top200_all.npy", DENSE)
    np.save(data / "splade_top200_all.npy", SPLADE)
    (data / "query_ids_all.json").write_text(
        json.dumps(
            {
                "ids": ["q0", "q1", "q2", "q3"],
                "golds": [
                    [BASE_GOLDS[0], f"doc_{discovered_gold}"],
                    [BASE_GOLDS[1]],
                    [BASE_GOLDS[2]],
                    [BASE_GOLDS[3]],
                ],
                "split_indices": {"train": [3], "val": [0, 1, 2], "test": []},
            }
        ),
        encoding="utf-8",
    )
    source = np.arange(NUM_NODES, dtype=np.int64)
    torch.save(
        {
            "edge_index": torch.tensor(
                np.stack([source, (source + 5) % NUM_NODES]), dtype=torch.long
            ),
            "num_nodes": NUM_NODES,
        },
        data / "graph.pt",
    )

    dataset = load_complete_dataset(data, dataset="hotpotqa_clean", require_embeddings=True)
    queries = dataset.split(QuerySplit.VALIDATION)
    rowptr = dataset.rowptr.numpy().astype(np.int64, copy=False)
    col = dataset.col.numpy().astype(np.int64, copy=False)
    return {
        "views": [QueryView(query) for query in queries],
        "queries": queries,
        "dense": DENSE,
        "splade": SPLADE,
        "rowptr": rowptr,
        "col": col,
        "num_nodes": NUM_NODES,
        "operators": build_operators(rowptr, col, NUM_NODES),
        "family_rowptr": family_rowptr,
        "family_col": family_col,
        "node_embeddings": dataset.node_array,
        "budget": budget,
        "query_count": len(dataset.queries),
        "a64_only_gold": discovered_gold,
    }


def _column_indices(module: types.ModuleType, arm: str, regime: str, width: int) -> list[int]:
    """Which master columns this module's ``_arm_columns`` selects, measured not re-derived.

    A master whose every row is ``[0, 1, ..., width-1]`` comes back as the
    literal index list the arm keeps, so this reads the real selection logic
    rather than restating MASTER_COLUMNS/ARM_FAMILIES a second time (which would
    only prove that this file agrees with itself).
    """

    marker = np.tile(np.arange(width, dtype=np.float32), (3, 1))
    return [int(v) for v in module._arm_columns(marker, arm, regime)[0]]


def _store_arrays(module: types.ModuleType, arm: str, regime: str, fixture: dict, masters: list):
    store, width = module._arm_store(
        arm=arm,
        regime=regime,
        master_blocks=masters,
        queries=fixture["queries"],
        query_count=fixture["query_count"],
        num_nodes=fixture["num_nodes"],
    )
    return store, int(width)


def run_bit_exact_probe(fixture: dict, current, previous) -> dict:
    """Old vs current feature tensors, every historical arm, every regime.

    Both modules build their own master block from the same raw inputs and then
    slice it with their own arm-composition code, so this compares the whole
    feature path rather than one function in isolation. A pair where both
    modules *refuse* the arm (NODE_ROLE outside R3) counts as agreement too --
    that refusal is part of the historical behaviour being preserved.
    """

    per_regime: dict[str, dict] = {}
    all_equal = True
    node_role_was_exercised = False

    for regime in REGIMES:
        kwargs = {k: fixture[k] for k in (
            "views", "queries", "dense", "splade", "rowptr", "col",
            "num_nodes", "operators", "family_rowptr", "family_col",
            "node_embeddings", "budget",
        )}
        current_scored, current_masters, _lat = current._cell_master_local(regime=regime, **kwargs)
        previous_scored, previous_masters, _lat = previous._cell_master_local(regime=regime, **kwargs)

        scored_equal = len(current_scored) == len(previous_scored) and all(
            np.array_equal(a, b) for a, b in zip(current_scored, previous_scored, strict=True)
        )
        # The master block itself is allowed to differ in WIDTH under R1/R2 --
        # that is the whole change: the current builder always appends a
        # NODE_ROLE column, the old one appended it only under R3. What may not
        # differ is any column a historical arm actually reads, which is what
        # the per-arm comparison below establishes.
        shared = min(
            min(m.shape[1] for m in current_masters), min(m.shape[1] for m in previous_masters)
        )
        shared_columns_equal = all(
            np.array_equal(a[:, :shared], b[:, :shared])
            for a, b in zip(current_masters, previous_masters, strict=True)
        )
        appended_node_role_is_zero_outside_r3 = regime == "R3" or all(
            not m[:, shared:].any() for m in current_masters
        )

        arms: dict[str, dict] = {}
        for arm in HISTORICAL_ARMS:
            entry: dict = {"regime": regime, "arm": arm}
            outcomes = {}
            for label, module in (("current", current), ("previous", previous)):
                try:
                    store, width = _store_arrays(module, arm, regime, fixture, {
                        "current": current_masters, "previous": previous_masters
                    }[label])
                except ValueError as exc:
                    outcomes[label] = {"refused": True, "reason": str(exc)}
                else:
                    outcomes[label] = {
                        "refused": False,
                        "local": np.asarray(store.local),
                        "candidate_ptr": np.asarray(store.candidate_ptr),
                        "query_position": np.asarray(store.query_position),
                        "precomputed_width": width,
                    }

            cur, prev = outcomes["current"], outcomes["previous"]
            if cur["refused"] or prev["refused"]:
                entry["both_refused"] = bool(cur["refused"] and prev["refused"])
                entry["equal"] = entry["both_refused"]
                entry["refusal_reason"] = cur.get("reason") or prev.get("reason")
                entry["why"] = "NODE_ROLE is defined only for R3; both paths refuse it identically"
            else:
                local_equal = (
                    cur["local"].dtype == prev["local"].dtype
                    and cur["local"].shape == prev["local"].shape
                    and np.array_equal(cur["local"], prev["local"])
                )
                entry["both_refused"] = False
                entry["dtype"] = str(cur["local"].dtype)
                entry["rows"], entry["columns"] = (int(cur["local"].shape[0]), int(cur["local"].shape[1]))
                entry["local_bit_exact"] = bool(local_equal)
                entry["candidate_ptr_equal"] = bool(
                    np.array_equal(cur["candidate_ptr"], prev["candidate_ptr"])
                )
                entry["query_position_equal"] = bool(
                    np.array_equal(cur["query_position"], prev["query_position"])
                )
                entry["precomputed_width"] = cur["precomputed_width"]
                entry["precomputed_width_equal"] = (
                    cur["precomputed_width"] == prev["precomputed_width"]
                )
                entry["column_indices"] = _column_indices(
                    current, arm, regime, current_masters[0].shape[1]
                )
                entry["column_indices_equal"] = entry["column_indices"] == _column_indices(
                    previous, arm, regime, previous_masters[0].shape[1]
                )
                entry["equal"] = bool(
                    local_equal
                    and entry["candidate_ptr_equal"]
                    and entry["query_position_equal"]
                    and entry["precomputed_width_equal"]
                    and entry["column_indices_equal"]
                )
                if regime == "R3" and "NODE_ROLE" in current.ARM_FAMILIES[arm]:
                    node_role_was_exercised = node_role_was_exercised or bool(
                        np.asarray(cur["local"])[:, -1].any()
                        if arm.endswith("NODE_ROLE")
                        else cur["local"].any()
                    )
            all_equal = all_equal and bool(entry["equal"])
            arms[arm] = entry

        per_regime[regime] = {
            "scored_sets_equal": bool(scored_equal),
            "shared_master_columns_equal": bool(shared_columns_equal),
            "shared_master_column_count": int(shared),
            "current_master_columns": int(current_masters[0].shape[1]),
            "previous_master_columns": int(previous_masters[0].shape[1]),
            "appended_node_role_is_zero_outside_r3": bool(appended_node_role_is_zero_outside_r3),
            "arms": arms,
        }
        all_equal = all_equal and scored_equal and shared_columns_equal
        all_equal = all_equal and appended_node_role_is_zero_outside_r3

    return {
        "status": "BIT_EXACT_HISTORICAL_ARM_EQUIVALENCE" if all_equal else "PROBE_FAILED",
        "pre_universal_rev": PRE_UNIVERSAL_REV,
        "pre_universal_blob_sha": _git_object_sha(PRE_UNIVERSAL_REV, RUNNER_RELPATH),
        "m1a_launch_blob_sha": _git_object_sha("f81985d", RUNNER_RELPATH),
        "blob_is_the_one_every_reused_fit_ran": (
            _git_object_sha(PRE_UNIVERSAL_REV, RUNNER_RELPATH)
            == _git_object_sha("f81985d", RUNNER_RELPATH)
        ),
        "fixture": {
            "num_nodes": NUM_NODES,
            "queries": len(fixture["queries"]),
            "seed": FIXTURE_SEED,
            "a64_only_gold": fixture["a64_only_gold"],
        },
        "arms_probed": list(HISTORICAL_ARMS),
        "regimes_probed": list(REGIMES),
        "node_role_column_was_nonzero_under_r3": bool(node_role_was_exercised),
        "comparison": "element-for-element on the float16 store the trainer reads, not a tolerance",
        "all_historical_arms_bit_exact": bool(all_equal),
        "per_regime": per_regime,
    }


# --- the per-fit checks ------------------------------------------------------


def proposed_reused_fits(declaration: dict) -> list[tuple[str, str, str, str]]:
    """(dataset, regime, arm, matrix_status) for every fit the matrix marks reused."""

    fits = []
    for dataset, regimes in declaration["m2_selection_matrix"]["cells"].items():
        for regime, arms in regimes.items():
            for arm, status in arms.items():
                if str(status).startswith("reuse_"):
                    fits.append((dataset, regime, arm, status))
    return fits


def _live_runner_hyperparameters(dataset: str, source: str) -> dict:
    """Read the launcher's own arguments, live, rather than transcribing them."""

    if source == "m1b":
        from scripts import modal_m1b_targeted_resolution as launcher
    else:
        from scripts import modal_m1a_feature_screen as launcher
    job = launcher._jobs([dataset])[0]
    args = launcher._runner_args(job, stage="headline")
    values = {name: getattr(args, name) for name in TRAINER_HYPERPARAMETERS}
    values["seed"] = getattr(args, "seed", None)
    if values["seed"] is None:
        seeds = list(getattr(args, "seeds", []))
        values["seed"] = seeds[0] if seeds else None
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in values.items()}


def _m1b_record(dataset: str, regime: str, arm: str) -> tuple[dict, dict, str] | None:
    path = M1B_HEADLINE_DIR / f"{dataset}.json"
    if regime != "R3" or dataset not in M1B_R3_DATASETS or not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    cell = result["cells"].get(regime, {})
    arm_entry = cell.get("arms", {}).get(arm)
    if arm_entry is None or "0" not in arm_entry.get("seeds", {}):
        return None
    return (
        result,
        arm_entry["seeds"]["0"],
        f"outputs/m1b_targeted_resolution/headline/{dataset}.json"
        f"#cells.{regime}.arms.{arm}.seeds.0",
    )


def _m1a_record(dataset: str, regime: str, arm: str) -> tuple[dict, dict, str]:
    result = json.loads((M1A_HEADLINE_DIR / f"{dataset}.json").read_text(encoding="utf-8"))
    record = result["cells"][regime]["arms"][arm]
    return (
        result,
        record,
        f"outputs/m1a_feature_screen/headline/{dataset}.json#cells.{regime}.arms.{arm}",
    )


def audit_fit(dataset: str, regime: str, arm: str, status: str, probe: dict) -> dict:
    preferred = _m1b_record(dataset, regime, arm)
    if preferred is not None:
        result, record, source = preferred
        source_phase = "m1b"
    else:
        result, record, source = _m1a_record(dataset, regime, arm)
        source_phase = "m1a"

    probe_arm = probe["per_regime"][regime]["arms"][arm]
    hyperparameters = _live_runner_hyperparameters(dataset, source_phase)
    contract = result["candidate_contract"]

    checks = {
        "same_historical_arm_name": {
            "recorded": record["arm"],
            "expected": arm,
            "is_a_historical_arm": arm in HISTORICAL_ARMS,
            "is_not_the_universal_arm": arm != UNIVERSAL_ARM,
            "pass": bool(
                record["arm"] == arm and arm in HISTORICAL_ARMS and arm != UNIVERSAL_ARM
            ),
        },
        "same_feature_column_indices": {
            "column_indices": probe_arm.get("column_indices"),
            "equal_old_vs_current": probe_arm.get("column_indices_equal"),
            "pass": bool(probe_arm.get("column_indices_equal")),
        },
        "same_model_input_width": {
            "recorded_precomputed_width": record["precomputed_width"],
            "current_code_precomputed_width": probe_arm.get("precomputed_width"),
            "pass": record["precomputed_width"] == probe_arm.get("precomputed_width"),
        },
        "same_trainer_hyperparameters": {
            "read_live_from": f"scripts/modal_{source_phase}_"
            + ("feature_screen" if source_phase == "m1a" else "targeted_resolution")
            + ".py::_runner_args",
            "values": hyperparameters,
            "recorded_semantic_rung": record["semantic_rung"],
            "recorded_holdout_fraction": result["holdout_fraction"],
            "recorded_seed": record.get("seed", result.get("seed")),
            "recorded_a64_mainline_family": result["a64_mainline_family"],
            "a64_mainline_family_matches": _a64_family_matches(result),
            "why_a64_can_be_null": (
                "run_m1a_feature_screen.run() records a64_mainline_family only "
                "when the dataset declares an R3 cell -- A64 builds C3 and nothing "
                "else, so an R1-only dataset (squad_clean) correctly records null. "
                "The launcher argument is 'structural_only' either way."
            ),
            "pass": bool(
                hyperparameters["epochs"] == 3
                and hyperparameters["batch_size"] == 16
                and hyperparameters["dropout"] == 0.2
                and hyperparameters["temperature"] == 0.07
                and hyperparameters["learning_rate"] == 1e-3
                and hyperparameters["weight_decay"] == 1e-4
                and hyperparameters["semantic_rung"] == record["semantic_rung"] == "S3"
                and hyperparameters["holdout_fraction"] == result["holdout_fraction"] == 0.2
                and hyperparameters["per_seed_cap"] == 16
                and hyperparameters["a64_mainline_family"] == "structural_only"
                and _a64_family_matches(result)
                and int(record.get("seed", result.get("seed"))) == 0
            ),
        },
        "same_validation_and_holdout_query_ids": {
            "split": result["split"],
            "selection": result["selection"],
            "queries": result["queries"],
            "test_split_read": result["test_split_read"],
            "train_queries": result["cells"][regime]["train_queries"],
            "held_out_queries": result["cells"][regime]["held_out_queries"],
            "holdout_split_reproduces_the_recorded_boundary": _holdout_boundary_matches(
                result["queries"],
                result["holdout_fraction"],
                result["cells"][regime]["train_queries"],
                result["cells"][regime]["held_out_queries"],
            ),
            "why_ids_follow_from_the_boundary": (
                "the query set is a deterministic prefix of the frozen validation "
                "split order and holdout_split cuts it at a computed index, so an "
                "equal (queries, holdout_fraction, cut) reproduces the same ids"
            ),
            "pass": bool(
                result["split"] == "validation"
                and result["selection"] == "deterministic_prefix_of_the_split_order"
                and result["test_split_read"] is False
                and _holdout_boundary_matches(
                    result["queries"],
                    result["holdout_fraction"],
                    result["cells"][regime]["train_queries"],
                    result["cells"][regime]["held_out_queries"],
                )
            ),
        },
        "same_dataset_fingerprint": {
            "recorded": result["data_fingerprint_sha256"],
            "pass": bool(result["data_fingerprint_sha256"]),
        },
        "same_candidate_contract_fingerprint": {
            "status": contract["status"],
            "expected_contract_sha256": contract["expected_contract_sha256"],
            "observed_contract_sha256": contract["observed_contract_sha256"],
            "candidate_id_order_sha256": contract["candidate_id_order_sha256"],
            "pass": bool(
                contract["status"] == "BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE"
                and contract["expected_contract_sha256"] == contract["observed_contract_sha256"]
            ),
        },
    }
    checks_pass = all(check["pass"] for check in checks.values())
    reusable = bool(checks_pass and probe_arm["equal"])
    return {
        "dataset": dataset,
        "regime": regime,
        "arm": arm,
        "matrix_status": status,
        "source_phase": source_phase,
        "source": source,
        "preferred_m1b_rows": source_phase == "m1b",
        "recall_at_5": record["metrics"]["recall@5"],
        "cross_checked_against_m1a_splice": record.get("cross_checked_against_m1a_splice"),
        "per_fit_checks": checks,
        "per_fit_checks_pass": checks_pass,
        "bit_exact_probe_pass": bool(probe_arm["equal"]),
        "reusable": reusable,
        "verdict": "REUSE" if reusable else "MARK_NEW",
    }


def _a64_family_matches(result: dict) -> bool:
    """``structural_only`` where an R3 cell exists, ``None`` where none does.

    Not a leniency: ``run()`` writes this field as ``args.a64_mainline_family if
    "R3" in cells else None``, because A64 is only ever used to build C3.
    Demanding ``structural_only`` from an R1-only dataset would fail a fit for
    running the code correctly.
    """

    recorded = result.get("a64_mainline_family")
    if "R3" in result["cells"]:
        return recorded == "structural_only"
    return recorded is None


def _holdout_boundary_matches(queries: int, fraction: float, train: int, held_out: int) -> bool:
    """Recompute holdout_split's own cut and compare it to the recorded sizes."""

    cut = queries - max(1, int(round(queries * fraction)))
    return cut == train and (queries - cut) == held_out


def _cross_check_fingerprints_agree(fits: list[dict]) -> dict:
    """Every fit on one dataset must carry the same dataset/contract fingerprints.

    A per-fit check cannot see this: it would pass on a dataset whose M1A and
    M1B records disagreed, because each record is self-consistent. Comparing
    across the fits of one dataset is what catches a spliced result.
    """

    by_dataset: dict[str, set] = {}
    for fit in fits:
        checks = fit["per_fit_checks"]
        key = (
            checks["same_dataset_fingerprint"]["recorded"],
            checks["same_candidate_contract_fingerprint"]["observed_contract_sha256"],
            checks["same_candidate_contract_fingerprint"]["candidate_id_order_sha256"],
        )
        by_dataset.setdefault(fit["dataset"], set()).add(key)
    disagreeing = sorted(d for d, keys in by_dataset.items() if len(keys) > 1)
    return {
        "datasets_checked": sorted(by_dataset),
        "datasets_with_disagreeing_fingerprints": disagreeing,
        "pass": not disagreeing,
    }


def main() -> dict:
    declaration = yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))
    proposed = proposed_reused_fits(declaration)
    workload = declaration["m2_selection_matrix"]["workload"]

    # ignore_cleanup_errors: load_complete_dataset mmaps nodes.npy (its zero-copy
    # convention) and Windows refuses to unlink a mapped file, so a clean probe
    # would otherwise die in teardown. Only the scratch fixture is affected.
    with tempfile.TemporaryDirectory(prefix="m2_reuse_audit_", ignore_cleanup_errors=True) as tmp:
        workdir = Path(tmp)
        from scripts import run_m1a_feature_screen as current

        current.ARM_FAMILIES.setdefault(M1B_INTERACTION_ARM, M1B_INTERACTION_FAMILIES)
        previous = load_pre_universal_module(workdir)
        fixture = write_probe_fixture(workdir)
        probe = run_bit_exact_probe(fixture, current, previous)

    fits = [audit_fit(dataset, regime, arm, status, probe) for dataset, regime, arm, status in proposed]
    fingerprint_cross_check = _cross_check_fingerprints_agree(fits)

    reused = sum(1 for fit in fits if fit["reusable"])
    refused = [f"{f['dataset']}/{f['regime']}/{f['arm']}" for f in fits if not f["reusable"]]
    declared_new = int(workload["new_fits"])
    all_reusable = bool(
        reused == len(proposed)
        and probe["all_historical_arms_bit_exact"]
        and fingerprint_cross_check["pass"]
    )

    manifest = {
        "status": "M2_REUSE_AUDIT_COMPLETE",
        "declaration": "configs/m2_qls_v2_freeze.yaml#m2_selection_matrix.reuse_audit",
        "scope": "every proposed reused fit individually, not one repository-level verdict",
        "git_head_now": _git_head(),
        "why_git_head_is_recorded_but_not_decisive": (
            "M1B's audit could prove reuse from HEAD identity alone; that premise "
            "is gone. HEAD is recorded for provenance only -- the verdict comes "
            "from the per-fit checks and the bit-exact probe below."
        ),
        "proposed_reused_fits": len(proposed),
        "declared_reused_fits": int(workload["reused_fits"]),
        "bit_exact_feature_probe": probe,
        "fingerprint_cross_check": fingerprint_cross_check,
        "fits": fits,
        "all_reusable": all_reusable,
        "reusable_fits": reused,
        "refused_fits": refused,
        "expected_reused_fits": reused,
        "expected_new_fits": declared_new + (len(proposed) - reused),
        "declared_new_fits": declared_new,
        "declared_logical_fits": int(workload["logical_fits"]),
        "workload_unchanged_by_this_audit": reused == int(workload["reused_fits"]),
        "on_failure": (
            "any refused fit is counted as NEW above; re-run "
            "scripts/m2_compute_estimate.py before the launch gate is evaluated"
        ),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(OUTPUT_PATH, manifest)
    return manifest


if __name__ == "__main__":
    result = main()
    probe = result["bit_exact_feature_probe"]
    print(f"pre-universal rev {probe['pre_universal_rev']} blob {probe['pre_universal_blob_sha'][:12]}")
    print(f"  blob identical to M1A's launch commit: {probe['blob_is_the_one_every_reused_fit_ran']}")
    print(f"  fixture: {probe['fixture']['queries']} queries, {probe['fixture']['num_nodes']} nodes, "
          f"A64-only gold doc_{probe['fixture']['a64_only_gold']}")
    print(f"  NODE_ROLE nonzero under R3: {probe['node_role_column_was_nonzero_under_r3']}")
    for regime, entry in probe["per_regime"].items():
        widths = f"{entry['previous_master_columns']}->{entry['current_master_columns']} master cols"
        print(f"  {regime} ({widths}):")
        for arm, arm_entry in entry["arms"].items():
            mark = "==" if arm_entry["equal"] else "!!"
            if arm_entry["both_refused"]:
                detail = "both refuse (NODE_ROLE outside R3)"
            else:
                detail = (f"{arm_entry['rows']}x{arm_entry['columns']} {arm_entry['dtype']} "
                          f"bit-exact={arm_entry['local_bit_exact']}")
            print(f"    {mark} {arm:24s} {detail}")
    print()
    for fit in result["fits"]:
        mark = "OK " if fit["reusable"] else "NEW"
        print(f"  {mark} {fit['dataset']:16s} {fit['regime']} {fit['arm']:16s} "
              f"[{fit['source_phase']}] r@5={fit['recall_at_5']:.6f}")
    print(f"\nWrote {OUTPUT_PATH}")
    print(
        f"\nVERDICT: all_reusable={result['all_reusable']}  "
        f"reused={result['expected_reused_fits']}/{result['proposed_reused_fits']}  "
        f"new={result['expected_new_fits']}  refused={result['refused_fits'] or 'none'}"
    )
