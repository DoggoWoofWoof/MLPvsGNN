"""What the step-D equivalence probe has to get right to be worth believing.

The probe is the evidence for one launch gate
(launch_authorization.gates.feature_build_equivalence_proved) and for one
decision (moving 83% of M2's estimated billed seconds off the A10G). Both rest
entirely on its comparison being strict and on its verdict being a conjunction
of what it actually measured -- a probe that reports EQUIVALENT because its
comparison is loose is worse than no probe, because it looks like evidence.

So these tests do not re-run the probe (it is minutes of real feature building
and is run once, on demand, writing its artifact). They pin the comparison
primitives, the verdict's derivation, and the committed artifact's agreement
with the declaration it is filed against.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import numpy as np
import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2_feature_build_equivalence as probe  # noqa: E402

DECLARATION_PATH = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
ARTIFACT_PATH = REPO_ROOT / "outputs" / "m2_qls_v2_freeze" / "feature_build_equivalence.json"
PROBE_PATH = REPO_ROOT / "scripts" / "m2_feature_build_equivalence.py"

#: The stage keys run_probe's verdict is a conjunction over, and the field in
#: each that carries that stage's own answer.
VERDICT_INPUTS = {
    "stage_1_real_call_site": ("no_torch_object_reaches_the_build",
                              "every_persisted_object_reloads_identical"),
    "stage_2_every_arm_from_a_reloaded_master": ("all_identical",),
    "stage_3_separate_process_no_cuda_one_numba_thread": ("all_identical",),
    "stage_4_training_from_a_reloaded_store": ("all_identical",),
}


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.is_file():
        pytest.skip("run scripts/m2_feature_build_equivalence.py first")
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


# --- the comparison primitive -------------------------------------------------


def test_identical_is_bit_exact_not_merely_equal_in_value() -> None:
    """The declaration's bar is "element-for-element, under the exact dtype and
    normalisation semantics already in force ... Not 'close'". np.array_equal
    calls float16 0.5 equal to float32 0.5; the store's local block is float16,
    so a value-only comparison would pass a store built at the wrong width."""

    half = np.array([0.5, 0.25], dtype=np.float16)
    assert probe._identical(half, half.copy())
    assert np.array_equal(half, half.astype(np.float32))
    assert not probe._identical(half, half.astype(np.float32))


def test_identical_rejects_a_shape_a_reshape_would_hide() -> None:
    flat = np.arange(6, dtype=np.float16)
    assert not probe._identical(flat, flat.reshape(2, 3))


def test_identical_rejects_a_single_changed_element() -> None:
    left = np.zeros((4, 12), dtype=np.float16)
    right = left.copy()
    right[3, 11] = np.float16(2.0 ** -14)
    assert not probe._identical(left, right)


def test_identical_ignores_only_memory_layout() -> None:
    """A C-ordered and an F-ordered view of the same values ARE the same
    features; the probe compares contents, not strides."""

    values = np.arange(12, dtype=np.float16).reshape(3, 4)
    assert probe._identical(values, np.asfortranarray(values))


def test_blocks_identical_requires_the_same_number_of_queries() -> None:
    block = [np.zeros((2, 12), dtype=np.float16)]
    assert probe._blocks_identical(block, [block[0].copy()])
    assert not probe._blocks_identical(block, block + [block[0].copy()])


# --- the verdict --------------------------------------------------------------


def test_the_verdict_is_a_conjunction_over_every_stage(artifact) -> None:
    measured = [
        artifact[stage][field] for stage, fields in VERDICT_INPUTS.items() for field in fields
    ]
    assert measured, "the artifact must carry the stage answers, not only the verdict"
    assert artifact["equivalent"] is all(measured)
    assert artifact["verdict"] == ("EQUIVALENT" if artifact["equivalent"] else "NOT_EQUIVALENT")


def test_run_probe_cannot_report_equivalent_while_a_stage_disagrees() -> None:
    """Read off the source rather than by running it: every stage answer named
    above has to appear in the conjunction, so a stage cannot be added to the
    report and silently left out of the verdict."""

    tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"), filename=str(PROBE_PATH))
    node = next(
        item
        for item in ast.walk(tree)
        if isinstance(item, ast.FunctionDef) and item.name == "run_probe"
    )
    assignment = next(
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "equivalent" for t in item.targets)
    )
    assert isinstance(assignment.value, ast.BoolOp)
    assert isinstance(assignment.value.op, ast.And)
    keys = {
        item.slice.value
        for item in ast.walk(assignment.value)
        if isinstance(item, ast.Subscript) and isinstance(item.slice, ast.Constant)
    }
    expected = {field for fields in VERDICT_INPUTS.values() for field in fields}
    assert expected <= keys
    assert len(assignment.value.values) == sum(len(f) for f in VERDICT_INPUTS.values())


def test_the_probe_exits_nonzero_when_the_split_is_not_equivalent() -> None:
    """A probe whose failure mode is a zero exit and a JSON file nobody reads is
    not a gate."""

    source = PROBE_PATH.read_text(encoding="utf-8")
    assert 'return 0 if report["equivalent"] else 1' in source


# --- what it actually compared ------------------------------------------------


def test_the_probe_checked_the_arm_the_headline_will_fit(artifact) -> None:
    """A probe that only proved BASE separable would prove nothing about
    QLS-UNIVERSAL, which is the arm every one of the 15 new fits trains."""

    checked = artifact["stage_2_every_arm_from_a_reloaded_master"]["arms_checked"]
    assert "BASE+NODE_ROLE+SUPPORT+PATH" in checked
    assert set(probe.declared_runner_arms()) <= set(checked)


def test_the_probe_covered_every_regime_the_matrix_declares(declaration, artifact) -> None:
    declared = {
        regime
        for cells in declaration["m2_selection_matrix"]["cells"].values()
        for regime in cells
    }
    assert declared <= set(probe.REGIMES)
    assert declared <= set(artifact["stage_1_real_call_site"]["regimes_built"])


def test_the_build_receives_no_torch_object_which_is_why_a_cpu_container_is_safe(
    artifact,
) -> None:
    """The load-bearing finding: a build that never receives a torch object or a
    device cannot observe whether an accelerator is attached, which is what makes
    a CPU-vs-CPU cross-process comparison sufficient evidence for a CPU-vs-GPU
    container split."""

    stage = artifact["stage_1_real_call_site"]
    assert stage["no_torch_object_reaches_the_build"] is True
    assert stage["torch_objects_reaching_the_build"] == []
    by_regime = stage["build_argument_types"]
    assert set(by_regime) == set(stage["regimes_built"])
    for regime, types in by_regime.items():
        assert set(probe.BUILD_KWARGS) <= set(types), regime
        assert not any("torch" in str(name).lower() for name in types.values()), (regime, types)
        assert "Tensor" not in set(types.values()), regime


def test_the_thread_count_check_is_there_because_the_kernels_are_parallel(artifact) -> None:
    """A CPU container does not have the GPU container's core count. Several
    kernels on this path are @njit(parallel=True), so a reduction whose order
    depends on the thread count would silently change features across the
    split -- and would not be visible in any single-process comparison."""

    stage = artifact["stage_3_separate_process_no_cuda_one_numba_thread"]
    assert stage["child_environment"]["NUMBA_NUM_THREADS"] == "1"
    assert stage["child_environment"]["CUDA_VISIBLE_DEVICES"] == ""
    parent = stage["parent_numba_threads"]
    assert parent == "default" or int(parent) >= 1
    assert parent != stage["child_environment"]["NUMBA_NUM_THREADS"], (
        "the child is pinned to one thread and the parent is not pinned at all; if both "
        "sides carried the same setting the comparison would say nothing about threads"
    )
    assert stage["fixture_bytes_identical"] is True, (
        "otherwise a master-block difference could be blamed on a different input"
    )
    assert stage["all_identical"] is True


def test_the_artifact_states_the_limit_of_its_own_evidence(artifact) -> None:
    """Local torch is a CPU build, so no A10G-built tensor is obtainable here.
    The report says so rather than letting the reader assume the comparison was
    against real GPU hardware."""

    cannot = artifact["what_this_cannot_prove"]
    assert "A10G" in cannot and "CPU against CPU" in cannot
    assert artifact["what_this_does_not_decide"].startswith("Whether to adopt the split")


# --- agreement with the declaration this is filed against ---------------------


def test_the_gate_is_true_only_while_the_artifact_backs_it(declaration) -> None:
    gate = declaration["launch_authorization"]["gates"]["feature_build_equivalence_proved"]
    if not gate:
        return
    assert ARTIFACT_PATH.is_file(), (
        "the gate claims proof; the proof has to be on disk to be checkable"
    )
    report = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert report["status"] == "M2_FEATURE_BUILD_EQUIVALENCE_COMPLETE"
    assert report["equivalent"] is True, (
        "or the gate must instead carry a filed decision to retain the proven "
        "single-container path -- feature_build_compute_check.adopt_only_if"
    )


def test_the_probe_answers_the_question_the_declaration_asked(declaration) -> None:
    check = declaration["feature_build_compute_check"]
    asked = " ".join(check["what_must_be_tested_before_launch"].split())
    assert "CPU container" in asked and "bit-exact" in asked
    requirement = " ".join(check["equivalence_requirement"].split())
    for named in ("local", "candidate_ptr", "query_position", "precomputed_width"):
        assert named in requirement
    # Every array the requirement names is one _store_identical compares, and
    # the width is compared alongside it in stage 2.
    source = PROBE_PATH.read_text(encoding="utf-8")
    for named in ("local", "candidate_ptr", "query_position", "static"):
        assert f"left.{named}, right.{named}" in source


def test_the_probe_spends_no_modal_compute() -> None:
    """Step D is a local measurement. A probe that quietly billed a container
    would be compute spent outside the amendment that authorised it."""

    tree = ast.parse(PROBE_PATH.read_text(encoding="utf-8"), filename=str(PROBE_PATH))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name == "modal" or name.startswith("modal.") for name in imported), imported
    # It does import the M2 launcher -- that is how stage 1 builds the arguments
    # the real Modal job would pass, instead of a second hand-written set that
    # could drift from it. Importing an app defines functions; it runs none. What
    # would actually bill is an invocation.
    invocations = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    } & {"remote", "spawn", "spawn_map", "remote_gen", "starmap"}
    assert invocations == set(), invocations
