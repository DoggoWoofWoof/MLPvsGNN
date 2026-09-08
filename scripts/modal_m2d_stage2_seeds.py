"""Modal launcher for M2D Stage 2: A3-MINIMAL at two more seeds, two cells.

Four spawned jobs -- one per (cell, seed) -- each fitting A3-MINIMAL once and
re-scoring M2B's native S4 on the same panel without training it. Four new fits
in total, which is exactly what section 15b authorises and no more.

Submit through scripts/spawn_modal_jobs.py so the run is server-side; a
detached local run is not a registered execution.

What differs from Stage 1's launcher
------------------------------------
Stage 1 spawned one job per cell and fit BOTH arms inside it, because section
10 asked what A3-MINIMAL's added column costs against its own control and two
containers cannot answer that. Stage 2 asks a different question -- whether the
seed-0 shortfall survives averaging over three seeds -- and A1 is not part of
it. So the unit is one fit, and the seed is what distinguishes two jobs on one
cell.

That makes the seed load-bearing in three places Stage 1 never had to think
about, and all three are wrong in the same silent way if they are missed:

*   **The checkpoint.** Native S4 must be M2B's S4 AT THIS SEED. M2B's headline
    tree holds seed 0 only; amendment 3's resolution tree holds seeds 1 and 2,
    under a different prefix. Re-scoring seed 0's checkpoint at seed 1 would
    produce a perfectly well-formed artifact whose S4 column is a constant
    across the three seeds it is supposed to vary.
*   **The expectation.** The recall@5 that re-score must reproduce is that
    seed's, read from that seed's filed M2B artifact.
*   **The path.** run_artifacts already carries a seed segment, and the runner
    now puts the seed in the local checkpoint path too, so two seeds of one
    cell in one store cannot address one file.

The panel is NOT one of them. `holdout_split` is a deterministic tail of a
frozen-order split with no seed in it, and M2B's filed resolution artifacts
carry the same held-out ids and the same shared-inputs digest as its headline.
That is checked here rather than assumed, because it is the assumption that
makes a same-seed comparison legal.

Everything else is Stage 1's, imported rather than restated: the container
shape comes from the filed Stage-2 compute record, the placement map comes from
M2's declaration, the identity expectations are derived host-side from M2B's
own filed artifacts, and the matrix -- arms, cells and seeds -- comes from
configs/m2d_s4_semantic_repair.yaml. A launcher holding its own copy of the
matrix can only ever disagree with the thing that declared it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import modal
import yaml

REMOTE_ROOT = "/root/message-passing-retrieval"
HOST_REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_REPO_ROOT = (
    HOST_REPO_ROOT
    if (HOST_REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml").is_file()
    else Path(REMOTE_ROOT)
)
M2D_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m2d_s4_semantic_repair.yaml"
M2_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
CONFIRMATION_CONFIG_PATH = RUNTIME_REPO_ROOT / "configs" / "sa_mlp_confirmation.yaml"
COMPUTE_RECORD_PATH = (
    RUNTIME_REPO_ROOT / "outputs" / "m2d_s4_semantic_repair" / "stage2_compute_record.json"
)

#: M2B's filed artifacts, host-side only. The headline tree is seed 0; the
#: resolution tree is amendment 3's seeds 1 and 2, and it is the one this
#: stage reads. Both are named because the panel identity is checked across
#: them: a resolution artifact whose panel had drifted from the headline's
#: would break every same-seed comparison the gate makes.
M2B_HEADLINE = HOST_REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "headline"
M2B_RESOLUTION = HOST_REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "resolution"
BASELINE_TABLE = (
    HOST_REPO_ROOT / "outputs" / "m2c_s4_structural_conditioning" / "m2b_baseline_table.json"
)

M2D_CONFIG_TEXT = M2D_CONFIG_PATH.read_text(encoding="utf-8")
CONFIG = yaml.safe_load(M2D_CONFIG_TEXT)
M2_CONFIG = yaml.safe_load(M2_CONFIG_PATH.read_text(encoding="utf-8"))
M1A_CONFIG = yaml.safe_load(M1A_CONFIG_PATH.read_text(encoding="utf-8"))
CONFIRMATION_CONFIG = yaml.safe_load(CONFIRMATION_CONFIG_PATH.read_text(encoding="utf-8"))

#: Over the decoded text, for the reason Stage 1 gives: the host reads this
#: file with CRLF and the container with LF, and one declaration must not have
#: two fingerprints depending on which side asked.
CONFIG_FINGERPRINT = hashlib.sha256(M2D_CONFIG_TEXT.encode("utf-8")).hexdigest()

#: The two declared Stage-2 cells, read from the declaration. One dataset each,
#: which is what lets a job be addressed by dataset the way every other
#: launcher in this track is.
CELLS: dict[str, str] = {}
for _cell in CONFIG["stage_2"]["cells"]:
    _dataset, _regime = _cell.split("/")
    if _dataset in CELLS:
        raise RuntimeError(
            f"{_dataset} carries two declared Stage-2 cells; this launcher addresses "
            "jobs by dataset and the second would be unreachable"
        )
    CELLS[_dataset] = _regime

#: The one arm and the two seeds, from the declaration rather than from here.
ARMS: tuple[str, ...] = tuple(CONFIG["stage_2"]["arms"])
SEEDS: list[int] = [int(seed) for seed in CONFIG["stage_2"]["seeds"]]

#: Stage 1's seed, which this stage REUSES and must never refit. It is named
#: here only so the refusal below can be written; nothing else reads it.
REUSED_SEED = int(CONFIG["stage_1"]["seeds"][0])
if REUSED_SEED in SEEDS:
    raise RuntimeError(
        f"the declaration lists seed {REUSED_SEED} among Stage 2's new seeds, but that "
        "row is Stage 1's and is reused rather than refit. Refitting it would replace "
        "the artifact Stage 1's own verdict was computed from."
    )

# spawn_modal_jobs.py's shared package protocol validates --datasets against
# module.CONFIG["datasets"]. The alias satisfies that contract without adding a
# datasets key to the declaration's own schema.
CONFIG["datasets"] = CELLS

#: The container shape the Stage-2 compute record filed, taken from the
#: declaration's own copy of it -- the copy that exists inside the container,
#: because the image carries src/, scripts/ and configs/ and no outputs/ tree.
SHAPE = CONFIG["launch_authorization"]["stage_2_compute_record"]
if not SHAPE["gpu"]:
    raise RuntimeError(
        "M2D Stage 2 fits four models; a record that priced no accelerator has not "
        "priced this stage."
    )

# Copied, not aliased: M2D must not be able to change M1A's or M2's infra view.
MODAL_CONFIG = dict(M1A_CONFIG["modal"])
GPU = SHAPE["gpu"]
CPU = int(SHAPE["cpu"])
MEMORY_MB = int(SHAPE["memory_mb"])
TIMEOUT_SECONDS = int(SHAPE["timeout_seconds"])
MODAL_CONFIG.update(
    {"gpu": GPU, "cpu": CPU, "memory_mb": MEMORY_MB, "timeout_seconds": TIMEOUT_SECONDS}
)

STORAGE_ROOT = MODAL_CONFIG["storage_root"]
APP_NAME = "message-passing-retrieval-m2d-stage2-seeds"
OUTPUT_PREFIX = "m2d_s4_semantic_repair"
STAGE_PREFIX = "stage2"
M2_OUTPUT_PREFIX = "m2_qls_v2_freeze"
M2B_OUTPUT_PREFIX = "m2b_semantic_minimality"

#: The SAME runner Stage 1 used. Section 3 freezes the architecture, so a
#: second runner would be a second implementation of a model that is not
#: allowed to change; the runner reads the declaration for which seeds it may
#: fit and stamps the stage into what it writes.
RUNNER_MODULE = "scripts.run_m2d_stage1_arms"

#: What the runner writes at a Stage-2 seed. Held equal to the runner's own
#: constant by a test: the gate refuses any artifact that does not carry it,
#: so a drift between the two is a stage whose results the gate cannot read.
STAGE_2_COMPLETE_STATUS = "M2D_STAGE2_SEED_COMPLETE"

#: The frozen build key of the cell stores M2 sealed. M2D reads it; never sets it.
BUILD_KEY = M2_CONFIG["qls_universal"]["hyperparameters"]

#: The rung whose weights this stage re-scores, and the directory M2B stored it
#: in. M2B writes ``fit_root = cell_root / rung.lower()``, so the two differ in
#: case; on the container's filesystem that difference is a missing file.
NATIVE_RUNG = "S4"

#: M2B's subtree for amendment 3's extra seeds. Its headline tree is seed 0
#: only, so this is where a seed-1 or seed-2 S4 checkpoint actually lives.
M2B_RESOLUTION_SUBTREE = "resolution"

#: One spawned call is one cell at one seed and writes its artifact at the end,
#: so a restart redoes exactly one fit. spawn_modal_jobs.py reads this to
#: decide what a restart costs; silence there is costed as no resumption.
RESUME_GRANULARITY = "cell_seed"

app = modal.App(APP_NAME)
result_volume = modal.Volume.from_name(MODAL_CONFIG["result_volume"], create_if_missing=False)
image = (
    modal.Image.micromamba(python_version="3.11")
    .env(
        {
            "CONDA_OVERRIDE_CUDA": "12.1",
            "CUDA_HOME": "/opt/conda",
            "TORCH_CUDA_ARCH_LIST": "8.6",
            "PYTHONPATH": f"{REMOTE_ROOT}:{REMOTE_ROOT}/src",
        }
    )
    .apt_install("git", "build-essential", "ninja-build")
    .pip_install(
        "torch==2.2.1",
        "numpy<2.0",
        "scipy<1.14",
        "numba==0.60.0",
        "psutil==6.1.1",
        "pyyaml==6.0.2",
    )
    .pip_install(
        "torch-geometric==2.5.2",
        "torch-scatter==2.1.2",
        "torch-sparse==0.6.18",
        find_links="https://data.pyg.org/whl/torch-2.2.1+cu121.html",
    )
    .add_local_dir(str(RUNTIME_REPO_ROOT / "src"), remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir(str(RUNTIME_REPO_ROOT / "scripts"), remote_path=f"{REMOTE_ROOT}/scripts")
    .add_local_dir(str(RUNTIME_REPO_ROOT / "configs"), remote_path=f"{REMOTE_ROOT}/configs")
)


def compute_record() -> dict[str, Any]:
    """The filed Stage-2 compute record. Host-side only, and read on demand.

    On demand for Stage 1's reasons: this module is imported inside the
    container too, the image carries no ``outputs/`` tree, and the registry
    test imports every launcher on a tree where the record may not have been
    generated. Absence is refused in ``require_authorisation``, which is where
    a launch is actually decided.
    """

    if not COMPUTE_RECORD_PATH.is_file():
        raise SystemExit(
            f"{COMPUTE_RECORD_PATH} does not exist. A filed compute record is a "
            "precondition of this launch; regenerate it with "
            "scripts/m2d_stage2_compute_record.py before submitting."
        )
    record = json.loads(COMPUTE_RECORD_PATH.read_text(encoding="utf-8"))
    if not record.get("filed_before_any_job_was_submitted"):
        raise SystemExit(f"{COMPUTE_RECORD_PATH} is not a pre-launch record")
    if record["container"]["gpu"] is None or record.get("authorises_no_gpu"):
        raise SystemExit(
            "the filed record authorises no GPU, but Stage 2 fits four models. "
            "Regenerate it rather than launching against a prediction for a "
            "different kind of job."
        )
    if record.get("trains_nothing"):
        raise SystemExit("the filed record says this stage trains nothing; it fits 4 models")
    container = record["container"]
    declared = (GPU, CPU, MEMORY_MB, TIMEOUT_SECONDS)
    filed = (
        container["gpu"],
        int(container["cpu_cores"]),
        int(container["memory_mb"]),
        int(container["timeout_seconds"]),
    )
    if declared != filed:
        raise SystemExit(
            f"the declaration's container shape {declared} and the filed record's "
            f"{filed} disagree. They are written by one script; regenerate rather "
            "than reconcile them by hand."
        )
    # Per (cell, seed), not per cell: this stage prices four containers on two
    # cells, and a record checked only by cell would admit a launch that ran a
    # seed nobody priced.
    priced = {(item["cell"], int(item["seed"])) for item in record["workload"]["cells"]}
    declared_units = {
        (f"{dataset}/{regime}", seed)
        for dataset, regime in CELLS.items()
        for seed in SEEDS
    }
    if priced != declared_units:
        raise SystemExit(
            f"the filed record prices {sorted(priced)} and the declaration names "
            f"{sorted(declared_units)}. A job nobody priced is a job nobody authorised."
        )
    if int(record["workload"]["fits"]) != len(declared_units):
        raise SystemExit(
            f"the filed record prices {record['workload']['fits']} fits and this "
            f"launcher spawns {len(declared_units)}"
        )
    return record


def _host_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(HOST_REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):  # pragma: no cover
        return None


def _stage_1_launcher() -> Any:
    """Stage 1's launcher, imported host-side for the helpers it already has.

    Imported inside the function rather than at module scope because this
    module is loaded in the container as well, and nothing remote needs Stage
    1's app, image or volume handle. Imported at all -- rather than copied --
    because the two helpers below are derivations, and two spellings of one
    derivation is how a same-seed comparison silently stops being one.
    """

    import importlib

    for path in (HOST_REPO_ROOT, HOST_REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return importlib.import_module("scripts.modal_m2d_stage1_arms")


def panel_digest(query_ids: list[str]) -> str:
    """The runner's digest, imported rather than reimplemented.

    Two spellings of one hash only ever show up as an identity mismatch on a
    paid container, which is the failure the identity check exists to prevent.
    """

    for path in (HOST_REPO_ROOT, HOST_REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from scripts.run_m2d_stage1_arms import panel_digest as digest

    return digest(query_ids)


def _s4_seed_spread_pp(dataset: str, regime: str) -> tuple[float, str]:
    """The bound the native-S4 re-score must reproduce M2B's recall@5 within.

    Stage 1's derivation, imported. Both Stage-2 cells carry three filed S4
    seeds, so both take the per-cell branch of it -- which is the tighter one,
    and is the same band the gate calls resolvable.
    """

    return _stage_1_launcher()._s4_seed_spread_pp(dataset, regime)


def _validation_split_queries(dataset: str, confirmation: dict[str, Any]) -> int:
    """This dataset's validation split size, derived then cross-checked.

    Stage 1's, imported. Stage 2 loads the same sealed cell master under the
    same build key, so it has to ask for the identical panel M2 scored.
    """

    return _stage_1_launcher()._validation_split_queries(dataset, confirmation)


def _m2b_resolution_artifact(dataset: str, seed: int) -> dict[str, Any]:
    path = M2B_RESOLUTION / f"seed{seed}" / f"{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist, so there is no filed M2B row at seed {seed} for "
            f"{dataset}. Every Stage-2 delta is taken against M2B's own same-seed "
            "rows; a seed whose row cannot be read cannot be launched."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def m2b_expectations(dataset: str, regime: str, seed: int) -> dict[str, Any]:
    """What this container must reproduce, from M2B's own filed same-seed row.

    Computed host-side and passed in, for Stage 1's two reasons -- the volume
    holds M2B's fits but not its artifacts, and an expectation the container
    derives for itself is one it can derive wrongly and then satisfy.

    The panel is taken from the resolution artifact and then checked against
    the headline's, rather than taken from the headline and assumed to hold.
    `holdout_split` has no seed in it so the two SHOULD agree; if they ever did
    not, every same-seed comparison in this stage would be comparing panels,
    and the right place to find that out is before a container starts.
    """

    resolution = _m2b_resolution_artifact(dataset, seed)
    block = resolution["cells"][regime]
    rung = block["rungs"][NATIVE_RUNG]
    if int(rung["seed"]) != seed:
        raise ValueError(
            f"{dataset}/{regime}: M2B's seed-{seed} resolution artifact carries a "
            f"{NATIVE_RUNG} row at seed {rung['seed']}"
        )

    headline_path = M2B_HEADLINE / f"{dataset}.json"
    if not headline_path.is_file():
        raise FileNotFoundError(
            f"{headline_path} does not exist, so the panel this stage scores cannot be "
            "checked against the one Stage 1 scored"
        )
    headline = json.loads(headline_path.read_text(encoding="utf-8"))["cells"][regime]
    if block["held_out_query_ids"] != headline["held_out_query_ids"]:
        raise ValueError(
            f"{dataset}/{regime}: M2B's seed-{seed} panel differs from its headline "
            "panel. The holdout split carries no seed, so this cannot happen without "
            "something upstream having changed, and every same-seed delta this stage "
            "reports would be a comparison across panels."
        )
    if block["shared_inputs"]["sha256"] != headline["shared_inputs"]["sha256"]:
        raise ValueError(
            f"{dataset}/{regime}: M2B's seed-{seed} structural inputs differ from its "
            "headline's. The structural block is frozen across seeds by construction."
        )

    bound_pp, bound_source = _s4_seed_spread_pp(dataset, regime)
    return {
        "panel_sha256": panel_digest(list(block["held_out_query_ids"])),
        "shared_inputs_sha256": block["shared_inputs"]["sha256"],
        "s4_recall_at_5": float(rung["metrics"]["recall@5"]),
        "s4_reproduction_bound_pp": bound_pp,
        "s4_reproduction_bound_source": bound_source,
        "held_out_queries": int(block["held_out_queries"]),
        "precomputed_width": int(block["precomputed_width"]),
        "filed_by": str(
            (M2B_RESOLUTION / f"seed{seed}" / f"{dataset}.json").relative_to(HOST_REPO_ROOT)
        ).replace("\\", "/"),
    }


def _m2_fits_root(dataset: str, fingerprint: str) -> PurePosixPath:
    """Where M2's headline run left this dataset's cell masters. Read-only.

    Seed-independent: M2 sealed one universe and every M2D seed scores it.
    """

    return (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / M2_OUTPUT_PREFIX
        / dataset
        / fingerprint[:16]
        / MODAL_CONFIG["execution_label"]
        / "headline"
        / "fits"
    )


def _m2b_fits_root(dataset: str, fingerprint: str, seed: int) -> PurePosixPath:
    """Where M2B left this dataset's semantic checkpoints AT THIS SEED. Read-only.

    Not the headline tree Stage 1 read. M2B's headline is seed 0; amendment 3's
    two extra seeds were written under ``resolution/seed<N>``, which is the one
    place a seed-1 or seed-2 S4 checkpoint exists. Pointing this at the
    headline would re-score seed 0's weights under a seed-1 label and produce
    an artifact that looks correct and is not.
    """

    if seed not in SEEDS:
        raise ValueError(
            f"seed {seed} is not one of Stage 2's declared seeds {SEEDS}, and M2B "
            f"filed no resolution tree for it"
        )
    return (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / M2B_OUTPUT_PREFIX
        / dataset
        / fingerprint[:16]
        / MODAL_CONFIG["execution_label"]
        / M2B_RESOLUTION_SUBTREE
        / f"seed{seed}"
        / "fits"
    )


def _jobs(datasets: list[str]) -> list[dict[str, Any]]:
    """One job per (dataset, seed). Four in total when both cells are asked for."""

    unknown = set(datasets) - set(CELLS)
    if unknown:
        raise ValueError(
            f"{sorted(unknown)} are not M2D Stage-2 cells; the declaration names "
            f"{sorted(CELLS)} and no others"
        )
    commit = _host_commit()
    jobs = []
    for dataset in datasets:
        regime = CELLS[dataset]
        settings = CONFIRMATION_CONFIG["datasets"][dataset]
        confirmation_path = HOST_REPO_ROOT / f"outputs/sa_mlp_confirmation/{dataset}.json"
        confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
        if confirmation.get("status") != "SA_MLP_CONFIRMATION_DATASET_COMPLETE":
            raise ValueError(f"{confirmation_path} is not a complete confirmation artifact")
        fingerprint = confirmation["data_fingerprint_sha256"]
        for seed in SEEDS:
            jobs.append(
                {
                    "dataset": dataset,
                    "regime": regime,
                    "seed": seed,
                    "fingerprint": fingerprint,
                    "data_remote": confirmation["config"]["data"],
                    "baseline": confirmation["baseline"],
                    "expected_queries": int(confirmation["config"]["expected_queries"]),
                    "frozen_embedding_dim": int(
                        M1A_CONFIG["base"]["semantic_rung"]["frozen_embedding_dim"]
                    ),
                    "validation_split_queries": _validation_split_queries(
                        dataset, confirmation
                    ),
                    "candidate_contract_compatibility": settings.get(
                        "candidate_contract_compatibility"
                    ),
                    "m2_fits_remote": str(_m2_fits_root(dataset, fingerprint)),
                    "m2b_fits_remote": str(_m2b_fits_root(dataset, fingerprint, seed)),
                    "expects": m2b_expectations(dataset, regime, seed),
                    "arms": list(ARMS),
                    "config_fingerprint": CONFIG_FINGERPRINT,
                    "source_commit": commit,
                }
            )
    return jobs


def _active_modal_profile() -> str | None:
    """The profile this client will actually submit under, or None if unreadable."""

    from_env = os.environ.get("MODAL_PROFILE")
    if from_env:
        return from_env
    try:
        from modal.config import _profile
    except ImportError:  # pragma: no cover - modal always ships this
        return None
    return _profile or None


def execution_placement() -> dict[str, str]:
    """Which workspace each dataset runs on, from M2's declaration and only there.

    Stage 2 reads M2's cell masters and M2B's checkpoints, so a cell can only
    be fit from the workspace whose volume holds them. The map moves as
    workspaces reach their spend limits, so a second copy could only ever go
    stale.
    """

    copied = CONFIG.get("launch_authorization", {}).get("execution_placement")
    if copied:
        raise SystemExit(
            "configs/m2d_s4_semantic_repair.yaml has grown its own execution_placement. "
            "The map is declared once, in configs/m2_qls_v2_freeze.yaml, and read at "
            "submit time; delete the copy."
        )
    inherited = M2_CONFIG["launch_authorization"].get("execution_placement")
    if not inherited:
        raise SystemExit(
            "configs/m2_qls_v2_freeze.yaml declares no workspace for these datasets; "
            "a dataset with no declared workspace cannot be submitted to one"
        )
    return dict(inherited)


def check_execution_placement(datasets: list[str]) -> dict[str, Any]:
    """Refuse to submit a dataset under a workspace its sealed artifacts are not on.

    scripts/spawn_modal_jobs.py calls this by name if a launcher defines it, so
    the check runs before the app is deployed and before anything is spawned.
    It is also where the declaration's gates are checked, because this is the
    only hook the server-side submission path calls before it deploys.
    """

    require_authorisation()
    placement = execution_placement()
    active = _active_modal_profile()
    if active is None:
        raise SystemExit(
            "cannot read the active Modal profile, so the workspace a job would land "
            "in is unknown. Set MODAL_PROFILE before importing modal."
        )
    undeclared = sorted(set(datasets) - set(placement))
    if undeclared:
        raise SystemExit(
            f"{undeclared} have no declared workspace in "
            "configs/m2_qls_v2_freeze.yaml#launch_authorization.execution_placement"
        )
    misplaced = {
        dataset: placement[dataset] for dataset in datasets if placement[dataset] != active
    }
    if misplaced:
        raise SystemExit(
            f"active profile {active!r} is not where these datasets' sealed artifacts "
            f"live: {misplaced}. Submit each dataset under its declared workspace."
        )
    return {"active_profile": active, "datasets": {d: placement[d] for d in datasets}}


def require_authorisation() -> None:
    """The declaration's gates, checked at call time rather than at import.

    At call time so this module stays importable -- and testable -- whatever
    the gates currently say, which is this track's standing order: commit the
    mechanism before the authorisation.
    """

    authorisation = CONFIG["launch_authorization"]
    gates = authorisation["gates"]
    for gate in (
        "stage_1_reported",
        "stage_2_amendment_filed",
        "stage_2_authorised",
        "stage_2_gate_committed",
        "stage_2_compute_record_filed",
    ):
        if not gates.get(gate):
            raise SystemExit(
                f"configs/m2d_s4_semantic_repair.yaml has not earned {gate}. "
                "Every one of these is a precondition of this launch."
            )
    # Stage 1's report is a precondition rather than a formality: section 15b
    # authorises these seeds BECAUSE of what that report measured, so a launch
    # that preceded it would be resolving a decision nobody had stated.
    if CONFIG["stage_2"]["seeds"] != SEEDS or tuple(CONFIG["stage_2"]["arms"]) != ARMS:
        raise SystemExit(
            f"the declaration now names seeds {CONFIG['stage_2']['seeds']} and arms "
            f"{CONFIG['stage_2']['arms']}; this launcher spawns {ARMS} at {SEEDS}"
        )
    if int(CONFIG["stage_2"]["new_fits"]) != len(CELLS) * len(SEEDS):
        raise SystemExit(
            f"the declaration authorises {CONFIG['stage_2']['new_fits']} new fits and "
            f"this launcher spawns {len(CELLS) * len(SEEDS)}"
        )
    forbidden = {item.lower() for item in authorisation["not_authorised"]}
    for clause in (
        "the full 14-cell m2d screen",
        "seeds 1 and 2 on any other arm, cell or regime",
        "a fourth seed",
    ):
        if clause not in forbidden:
            raise SystemExit(
                f"the declaration no longer forbids {clause!r}, so this launcher can no "
                "longer assume the four fits it spawns are the whole authorised workload"
            )
    # Last, because it is the expensive one and because its failure message is
    # about regenerating an artifact rather than about authorisation.
    compute_record()


def _output_root(job: dict[str, Any]) -> PurePosixPath:
    """The run store this job's artifact is written INTO, not its path.

    No seed segment here on purpose: run_artifacts derives phase, dataset,
    regime, SEED, arm, commit and run id beneath this root, so two seeds of one
    cell are separated by the layout that already exists rather than by a
    second convention invented here.
    """

    return (
        PurePosixPath(STORAGE_ROOT)
        / "outputs"
        / OUTPUT_PREFIX
        / STAGE_PREFIX
        / job["dataset"]
        / job["fingerprint"][:16]
        / MODAL_CONFIG["execution_label"]
    )


def _runner_args(job: dict[str, Any]) -> argparse.Namespace:
    from scripts.run_m0b_regime_map import MAINLINE_FAMILY

    regime = job["regime"]
    expects = job["expects"]
    return argparse.Namespace(
        data=Path(job["data_remote"]),
        dataset=job["dataset"],
        regime=regime,
        data_fingerprint_sha256=job["fingerprint"],
        expected_queries=int(job["expected_queries"]),
        frozen_embedding_dim=int(job["frozen_embedding_dim"]),
        queries=int(job["validation_split_queries"]),
        baseline=job["baseline"],
        candidate_contract_compatibility=job["candidate_contract_compatibility"],
        cell_features=Path(job["m2_fits_remote"]) / regime / "cell_features",
        # This seed's M2B checkpoint, from this seed's resolution tree.
        s4_checkpoint=(
            Path(job["m2b_fits_remote"]) / regime / NATIVE_RUNG.lower() / "checkpoint.pt"
        ),
        arms=list(job["arms"]),
        seed=int(job["seed"]),
        expected_panel_sha256=expects["panel_sha256"],
        expected_shared_inputs_sha256=expects["shared_inputs_sha256"],
        expected_s4_recall_at_5=expects["s4_recall_at_5"],
        s4_reproduction_bound_pp=expects["s4_reproduction_bound_pp"],
        output_root=Path(_output_root(job)),
        config_fingerprint=job["config_fingerprint"],
        source_commit=job["source_commit"],
        run_id=None,
        fit_root=None,
        device=None,
        # Section 3 freezes the architecture and section 12 says not to reopen
        # M2B's training setup. Identical to Stage 1's, which is the only way
        # a seed-to-seed difference means the seed.
        holdout_fraction=0.2,
        epochs=3,
        batch_size=16,
        dropout=0.2,
        temperature=0.07,
        learning_rate=1e-3,
        weight_decay=1e-4,
        latency_queries=200,
        latency_repeats=3,
        latency_warmup=8,
        per_seed_cap=int(BUILD_KEY["per_seed_cap"]),
        neighbour_scan_cap_per_seed=int(BUILD_KEY["neighbour_scan_cap_per_seed"]),
        a64_mainline_family=MAINLINE_FAMILY,
    )


@app.function(
    image=image,
    volumes={STORAGE_ROOT: result_volume},
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    cpu=CPU,
    memory=MEMORY_MB,
)
def run_stage2(job: dict[str, Any]) -> dict[str, Any]:
    import importlib

    os.chdir(REMOTE_ROOT)
    # Load-bearing: M2's masters and M2B's checkpoints were written by other
    # containers in other phases, so without this the volume view is whatever
    # it was when this container started.
    result_volume.reload()
    run = importlib.import_module(RUNNER_MODULE).run

    result = run(_runner_args(job))
    result_volume.commit()
    return {
        "status": result["status"],
        "dataset": job["dataset"],
        "cell": result["cell"],
        "seed": result["seed"],
        "arms": result["arms"],
        "held_out_queries": result["held_out_queries"],
        "test_split_read": result["test_split_read"],
        "identity_checks": result["identity_checks"],
        "native_s4_reproduction": result["native_s4_reproduction"],
        "artifacts": {arm: receipt["path"] for arm, receipt in result["artifacts"].items()},
        "wall_seconds": result["wall_seconds"],
    }


def _staging_suffix(job: dict[str, Any], remote_path: str) -> PurePosixPath:
    """Where one physical run lands under the local staging directory.

    The identity segments and the filename, and nothing above them -- which is
    exactly what verify_artifact_file compares, and short enough that the
    mirrored path does not run past Windows' 260-character limit.
    """

    return PurePosixPath(remote_path).relative_to(_output_root(job))


def _download(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    relative = remote_path.removeprefix(f"{STORAGE_ROOT}/")
    with local_path.open("wb") as stream:
        for chunk in result_volume.read_file(relative):
            stream.write(chunk)


def _artifact_names() -> tuple[str, str]:
    """The phase and filename run_artifacts will have used.

    Imported here rather than at module scope: this module is imported inside
    the container too, and only the fetch path needs the runner's constants.
    """

    import importlib

    for path in (HOST_REPO_ROOT, HOST_REPO_ROOT / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from mp_retrieval.run_artifacts import ARTIFACT_FILENAME

    return importlib.import_module(RUNNER_MODULE).PHASE, ARTIFACT_FILENAME


def _remote_prefix(job: dict[str, Any], arm: str) -> PurePosixPath:
    """One fit's logical result directory on the volume, above the physical runs.

    ``<phase>/<dataset>/<regime>/<seed>/<arm>``. The segments are asked for
    rather than spelled: run_artifacts renders the seed as ``seed_1``, and a
    launcher that wrote ``seed1`` would list an empty prefix and report that a
    completed fit produced nothing.
    """

    from mp_retrieval.run_artifacts import logical_segments

    phase, _ = _artifact_names()
    return _output_root(job).joinpath(
        *logical_segments(
            phase=phase,
            dataset=job["dataset"],
            regime=job["regime"],
            arm=arm,
            seed=int(job["seed"]),
        )
    )


def _remote_artifacts(prefix: PurePosixPath) -> list[str]:
    _, filename = _artifact_names()
    relative = str(prefix).removeprefix(f"{STORAGE_ROOT}/")
    found = []
    for entry in result_volume.listdir(relative, recursive=True):
        if PurePosixPath(entry.path).name == filename:
            found.append(f"{STORAGE_ROOT}/{entry.path}")
    return sorted(found)


def fetch(
    datasets: str = ",".join(CELLS), expect_source_commit: str = ""
) -> list[dict[str, Any]]:
    """Copy the selected result for each cell, seed and arm down beside the gate.

    Separate from the spawn because the spawn is server-side and returns as
    soon as the calls are registered. Reads the volume; writes nothing to it.

    Every physical run found is downloaded and verified, and then ONE is
    selected as the logical result -- by commit, stated by the caller, never by
    whichever file is newest. A run that did not land leaves an older run's
    numbers in place, and a fetcher that took the newest file would report them
    without noticing.

    The local filename carries the seed. Two seeds of one cell are two rows of
    the same mean, and a name that dropped the seed would let the second
    overwrite the first and the gate report a two-seed mean as a three-seed one.
    """

    if not expect_source_commit:
        raise SystemExit(
            "fetch needs the commit whose run is being reported. Selecting between "
            "physical runs is a decision, and a fetch that picked for itself would "
            "be free to report a run that was never meant to stand."
        )
    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    fetched = []
    local_root = HOST_REPO_ROOT / "outputs" / OUTPUT_PREFIX / STAGE_PREFIX
    staging = local_root / "runs"
    from mp_retrieval import run_artifacts

    for job in _jobs(requested):
        cell = f"{job['dataset']}/{job['regime']}"
        seed = int(job["seed"])
        for arm in job["arms"]:
            remotes = _remote_artifacts(_remote_prefix(job, arm))
            if not remotes:
                raise RuntimeError(
                    f"{cell} {arm} seed {seed}: the volume holds no artifact under its "
                    "prefix"
                )
            receipts = []
            for remote in remotes:
                local = staging / _staging_suffix(job, remote)
                _download(remote, local)
                receipts.append(run_artifacts.verify_artifact_file(local))
            selected = run_artifacts.select_logical_result(
                receipts, expect_source_commit=expect_source_commit
            )
            destination = (
                local_root / f"{job['dataset']}_{job['regime']}_{arm}_seed{seed}.json"
            )
            destination.write_bytes(Path(selected.path).read_bytes())
            fetched.append(
                {
                    "dataset": job["dataset"],
                    "cell": cell,
                    "seed": seed,
                    "arm": arm,
                    "local": str(destination),
                    "source_commit": selected.identity.source_commit,
                    "run_id": selected.identity.run_id,
                    "rows": selected.row_count,
                    "physical_runs_found": len(receipts),
                }
            )
    return fetched


@app.local_entrypoint()
def main(datasets: str = ",".join(CELLS)) -> None:
    """A local fallback only. The declared submission path is server-side.

    scripts/spawn_modal_jobs.py deploys the app and spawns each job so the run
    outlives the client; this entrypoint exists so the app is complete and
    exercisable, and it refuses under the same gates.
    """

    requested = [name.strip() for name in datasets.split(",") if name.strip()]
    check_execution_placement(requested)
    results = list(run_stage2.map(_jobs(requested), return_exceptions=True))
    failures = [item for item in results if isinstance(item, BaseException)]
    if failures:
        raise RuntimeError(f"{len(failures)} M2D Stage-2 job(s) failed: {failures}")
    print(json.dumps(results, indent=2))
