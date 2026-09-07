"""The feature-store contract hash: what it must ignore, and what it must catch.

The contract exists because M2 identified a persisted tensor by the SHA-256 of
a mutable YAML file, and the first amendment to that file invalidated fourteen
stores that had not changed. Replacing one hash with another only helps if the
new one has both properties, so both are tested here as pairs:

* an amendment about authorisation, statistics, compute or wording must leave
  the hash where it is, and
* a changed feature formula, constant, column layout, candidate set, split or
  dtype must move it.

A test suite that only checked the first would pass for a constant function.

The formula half of the contract is a version string, so it gets the same
treatment: the pinned behaviour digest is measured from the real
``_cell_master_local`` over a deterministic toy cell, and a mutated formula is
required to break the pin. Otherwise a formula could change while the version,
and therefore every contract hash, stayed still.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import feature_build_contract as fbc  # noqa: E402
from scripts import m2_reuse_audit as audit  # noqa: E402
from scripts import m2b_feature_store_identity as identity  # noqa: E402
from scripts import run_m1a_feature_screen as m1a  # noqa: E402
from scripts import run_m2_qls_v2_freeze as runner  # noqa: E402

M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"

#: A contract that is complete and internally consistent, used wherever a test
#: needs to perturb exactly one thing.
BASE_KWARGS = {
    "dataset": "2wiki_clean",
    "data_fingerprint_sha256": "d7c2da85e2b656805246e98ae7ac6e29cdc3355a2a8a6a3b9e554f2801caeb04",
    "regime": "R3",
    "queries": 3000,
    "query_split": fbc.QUERY_SPLIT,
    "query_selection": fbc.QUERY_SELECTION,
    "candidate_contract_sha256": "11609cd75c40925bfe30810bc55703148cc7626eebe14e3b92fa2acfa67f9121",
    "candidate_id_order_sha256": "c6ea17c51c53a01d616c0a6dcec6239fdcd70f536bb60bfec5e50b3183988daa",
    "per_seed_cap": 16,
    "neighbour_scan_cap_per_seed": 4096,
    "a64_mainline_family": "structural_only",
    "store_format": "m2_cell_master_features_v1",
    "master_columns": 12,
    "master_dtype": "float32",
}


def contract(**overrides) -> dict:
    return fbc.feature_build_contract(**{**BASE_KWARGS, **overrides})


def digest(**overrides) -> str:
    return fbc.contract_sha256(contract(**overrides))


@pytest.fixture(scope="module")
def toy_cell(tmp_path_factory):
    """The deterministic fixture the reuse audit already builds for its probe."""

    return audit.write_probe_fixture(tmp_path_factory.mktemp("contract_probe"))


@pytest.fixture(scope="module")
def report() -> dict:
    return identity.build_report()


# --------------------------------------------------------------------------
# What the hash must ignore
# --------------------------------------------------------------------------


def test_paperwork_does_not_appear_in_the_contract() -> None:
    fields = set(contract())
    forbidden = {
        "config_sha256", "source_commit", "seed", "epochs", "learning_rate",
        "batch_size", "dropout", "temperature", "weight_decay", "holdout_fraction",
        "semantic_rung", "device", "artifact_root", "status", "verdict",
    }
    assert not (fields & forbidden), sorted(fields & forbidden)


def test_the_semantic_rung_is_absent_so_one_store_serves_every_rung() -> None:
    """M2B's whole reuse claim: the master block is structural."""

    assert not any("rung" in field or "semantic" in field for field in contract())


def test_amending_the_declaration_cannot_move_the_hash(tmp_path: pathlib.Path) -> None:
    """The exact failure that started this: an amendment invalidating a tensor."""

    declaration = yaml.safe_load(M2_DECLARATION.read_text(encoding="utf-8"))
    before = digest()

    declaration["amendments"].append(
        {"amendment": 99, "date": "2026-09-07", "change": "recorded a selection"}
    )
    declaration["compute"] = {"ceiling_usd": 6.0}
    amended = tmp_path / "amended.yaml"
    amended.write_text(yaml.safe_dump(declaration), encoding="utf-8")

    assert amended.read_bytes() != M2_DECLARATION.read_bytes(), "the fixture did not amend"
    assert digest() == before


def test_the_declaration_hash_really_has_drifted(report: dict) -> None:
    """Otherwise the problem this module solves would be hypothetical."""

    problem = report["the_problem"]
    assert problem["drifted"] is True
    assert problem["recorded_in_every_store"] != problem["the_same_file_now"]


def test_key_order_in_the_caller_cannot_move_the_hash() -> None:
    forward = contract()
    reversed_order = dict(reversed(list(forward.items())))
    assert list(reversed_order) != list(forward)
    assert fbc.contract_sha256(reversed_order) == fbc.contract_sha256(forward)


# --------------------------------------------------------------------------
# What the hash must catch
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "hotpotqa_clean"),
        ("data_fingerprint_sha256", "0" * 64),
        ("regime", "R2"),
        ("queries", 2999),
        ("query_split", "test"),
        ("query_selection", "random_sample"),
        ("candidate_contract_sha256", "1" * 64),
        ("candidate_id_order_sha256", "2" * 64),
        ("per_seed_cap", 8),
        ("neighbour_scan_cap_per_seed", 2048),
        ("a64_mainline_family", "some_other_family"),
        ("store_format", "m2_cell_master_features_v2"),
        ("master_columns", 11),
        ("master_dtype", "float16"),
    ],
)
def test_changing_any_load_bearing_input_moves_the_hash(field: str, value) -> None:
    assert digest(**{field: value}) != digest(), field


def test_every_contract_field_is_covered_by_one_of_the_two_tests_above() -> None:
    """No field may be added that neither test exercises."""

    ignored = {"contract_version", "feature_builder", "feature_formula_version"}
    formula = {
        "context_arm", "feature_damping", "feature_ppr_iterations",
        "feature_normalisation", "master_column_layout", "master_column_count",
    }
    perturbed = {
        "dataset", "data_fingerprint_sha256", "regime", "queries", "query_split",
        "query_selection", "candidate_contract_sha256", "candidate_id_order_sha256",
        "per_seed_cap", "neighbour_scan_cap_per_seed", "a64_mainline_family",
        "store_format", "master_columns", "master_dtype",
    }
    uncovered = set(contract()) - ignored - formula - perturbed
    assert not uncovered, (
        f"{sorted(uncovered)} entered the contract without a test proving the hash "
        "responds to it"
    )


@pytest.mark.parametrize(
    ("constant", "value"),
    [("FEATURE_DAMPING", 0.9), ("FEATURE_PPR_ITERATIONS", 9), ("CONTEXT_ARM", "TARGET_H2")],
)
def test_changing_a_live_formula_constant_moves_the_hash(
    monkeypatch: pytest.MonkeyPatch, constant: str, value
) -> None:
    """The constants are read from the live module, so this is not a restatement."""

    before = digest()
    monkeypatch.setattr(m1a, constant, value)
    assert digest() != before


def test_changing_the_master_column_layout_moves_the_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = digest()
    monkeypatch.setattr(
        m1a, "MASTER_COLUMNS", {**m1a.MASTER_COLUMNS, "PATH": slice(8, 10)}
    )
    assert digest() != before


def test_bumping_the_formula_version_moves_every_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = digest()
    monkeypatch.setattr(fbc, "FEATURE_FORMULA_VERSION", "m1a_master_block_v2")
    assert digest() != before


def test_an_unknown_regime_is_refused_rather_than_hashed() -> None:
    with pytest.raises(ValueError, match="unknown regime"):
        contract(regime="R4")


# --------------------------------------------------------------------------
# The formula version is defended by behaviour, not by discipline
# --------------------------------------------------------------------------


def test_the_live_formulas_reproduce_the_pinned_behaviour_digest(toy_cell: dict) -> None:
    measured = fbc.measure_formula_behaviour(toy_cell)
    assert measured["rounded"] == fbc.pinned_formula_behaviour(), (
        "the master block the real builder emits no longer matches the digest pinned "
        f"for {fbc.FEATURE_FORMULA_VERSION!r}. If a formula genuinely changed, add a new "
        "FEATURE_FORMULA_VERSION and its measured digest in the same commit -- every "
        "store built under the old formulas must stop validating."
    )


def test_the_behaviour_digest_is_reproducible(toy_cell: dict) -> None:
    assert fbc.measure_formula_behaviour(toy_cell) == fbc.measure_formula_behaviour(toy_cell)


def test_the_behaviour_digest_catches_a_changed_feature_value(
    toy_cell: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pin is worth nothing unless a real formula edit breaks it."""

    import numpy as np

    before = fbc.measure_formula_behaviour(toy_cell)
    original = m1a._cell_master_local

    def nudged(**kwargs):
        scored, masters, latencies = original(**kwargs)
        masters = [block.copy() for block in masters]
        masters[0][0, 0] = np.float32(masters[0][0, 0]) + np.float32(1.0)
        return scored, masters, latencies

    monkeypatch.setattr(m1a, "_cell_master_local", nudged)
    assert fbc.measure_formula_behaviour(toy_cell)["rounded"] != before["rounded"]


def test_the_behaviour_digest_catches_a_changed_candidate_set(
    toy_cell: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Which rows exist is as much a formula as what is in them."""

    before = fbc.measure_formula_behaviour(toy_cell)
    original = m1a._cell_master_local

    def truncated(**kwargs):
        scored, masters, latencies = original(**kwargs)
        return [block[:-1] for block in scored], [block[:-1] for block in masters], latencies

    monkeypatch.setattr(m1a, "_cell_master_local", truncated)
    assert fbc.measure_formula_behaviour(toy_cell)["rounded"] != before["rounded"]


def test_the_digest_covers_all_three_regimes() -> None:
    """R2 and R3 differ from R1 only in how the context is built."""

    assert fbc.BEHAVIOUR_REGIMES == ("R1", "R2", "R3")


def test_an_unpinned_version_raises_rather_than_passing_silently() -> None:
    with pytest.raises(KeyError, match="no behaviour digest is pinned"):
        fbc.pinned_formula_behaviour("m1a_master_block_v999")


# --------------------------------------------------------------------------
# The fourteen existing stores
# --------------------------------------------------------------------------


def test_every_declared_cell_was_audited(report: dict) -> None:
    assert report["cells_declared"] == report["cells_audited"] == 14


def test_the_reconstruction_matches_what_current_code_would_build(report: dict) -> None:
    """The user's condition for accepting a reconstructed key at all."""

    for entry in report["stores"]:
        cell = f"{entry['dataset']}/{entry['regime']}"
        assert entry["differing_fields"] == [], cell
        assert entry["directions_agree"] is True, cell
        assert entry["feature_build_contract_sha256"] == entry["forward_contract_sha256"]


def test_the_forward_direction_runs_the_real_launcher() -> None:
    """A transcription of the launcher's arguments would prove nothing."""

    import inspect

    source = inspect.getsource(identity.audit_store)
    assert "launcher._jobs" in source and "launcher._runner_args" in source


def test_each_store_keeps_its_original_declaration_hash_verbatim(report: dict) -> None:
    for entry in report["stores"]:
        assert entry["original_full_config_sha256"] == identity.M2_BUILD_TIME_CONFIG_SHA256
        assert entry["original_full_config_sha256"] != entry["feature_build_contract_sha256"]


def test_the_audit_leaves_every_downloaded_store_metadata_byte_identical(
    tmp_path: pathlib.Path,
) -> None:
    """Provenance is read and carried forward, never rewritten."""

    import hashlib

    paths = [path for _dataset, _regime, path in identity.store_metadata_paths()]
    assert len(paths) == 14
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    identity.main(["--out", str(tmp_path / "identity.json")])
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    assert after == before


def test_the_stores_are_distinguishable_from_each_other(report: dict) -> None:
    """A contract that collapsed two cells would let a fit load the wrong tensor."""

    hashes = [entry["feature_build_contract_sha256"] for entry in report["stores"]]
    assert len(set(hashes)) == len(hashes) == 14


def test_every_store_agrees_about_the_cell_independent_fields(report: dict) -> None:
    shared = report["shared_contract_fields"]
    assert shared["feature_formula_version"] == fbc.FEATURE_FORMULA_VERSION
    assert shared["master_columns"] == 12
    assert shared["master_dtype"] == "float32"
    assert shared["query_split"] == fbc.QUERY_SPLIT
    # build_report records a failure for any shared field that took more than
    # one value; asserting on that is the real check, rather than guessing from
    # the reported value's type which fields disagreed.
    assert not [
        failure for failure in report["failed_checks"] if failure.startswith("stores disagree")
    ]


def test_the_formula_constants_were_the_same_at_every_build_commit(report: dict) -> None:
    for entry in report["stores"]:
        assert entry["formula_constants_unchanged_since_build"] is True, entry["dataset"]
        for name, check in entry["formula_constants"].items():
            assert check["at_build_commit"] == check["in_the_tree_now"], (entry["dataset"], name)


def test_the_verdict_is_permitted_and_the_exit_code_agrees(
    report: dict, tmp_path: pathlib.Path
) -> None:
    assert report["failed_checks"] == []
    assert report["verdict"] == identity.VERDICT_PERMITTED
    assert identity.main(["--out", str(tmp_path / "identity.json")]) == 0


def test_a_disagreeing_store_forbids_reuse_and_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """End to end: if a store were genuinely foreign, this gate would say so."""

    original = identity.audit_store

    def drifted(dataset, regime, path):
        entry = original(dataset, regime, path)
        if (dataset, regime) == ("webqsp", "R3"):
            entry["directions_agree"] = False
            entry["differing_fields"] = ["data_fingerprint_sha256"]
        return entry

    monkeypatch.setattr(identity, "audit_store", drifted)
    report = identity.build_report()
    assert report["verdict"] == identity.VERDICT_FORBIDDEN
    assert any("webqsp/R3" in failure for failure in report["failed_checks"])
    assert "recomputed before anything launches" in report["what_the_verdict_means"]
    assert identity.main(["--out", str(tmp_path / "identity.json")]) == 1


# --------------------------------------------------------------------------
# The runner emits the strings the contract hashes
# --------------------------------------------------------------------------


def test_the_runner_emits_the_contracts_split_constants() -> None:
    assert runner.QUERY_SPLIT is fbc.QUERY_SPLIT
    assert runner.QUERY_SELECTION is fbc.QUERY_SELECTION


def test_the_hoist_did_not_change_what_m2_actually_wrote() -> None:
    """Every build artifact M2 produced must still read the same as the constants."""

    artifacts = sorted(identity.BUILD_ARTIFACT_DIR.glob("*.json"))
    assert len(artifacts) == 6
    for path in artifacts:
        written = json.loads(path.read_text(encoding="utf-8"))
        assert written["split"] == fbc.QUERY_SPLIT, path.name
        assert written["selection"] == fbc.QUERY_SELECTION, path.name


# --------------------------------------------------------------------------
# The admission decision a fit actually makes
# --------------------------------------------------------------------------


def _store(**overrides) -> dict:
    """A legacy M2 store's metadata, in the shape M2 really wrote it."""

    build_key = {
        "dataset": "2wiki_clean",
        "data_fingerprint_sha256": BASE_KWARGS["data_fingerprint_sha256"],
        "regime": "R3",
        "queries": 3000,
        "per_seed_cap": 16,
        "neighbour_scan_cap_per_seed": 4096,
        "a64_mainline_family": "structural_only",
        "config_sha256": identity.M2_BUILD_TIME_CONFIG_SHA256,
    }
    metadata = {
        "format": "m2_cell_master_features_v1",
        "master_columns": 12,
        "master_dtype": "float32",
        "fingerprint_sha256": "f" * 64,
        "build_key": build_key,
        "source_commit": "3489f4cee74c159cb52f7c5a6927fa5c09cf4f8b",
    }
    metadata.update(overrides)
    return metadata


def _expected_key(**overrides) -> dict:
    key = {field: _store()["build_key"][field] for field in fbc.SCIENTIFIC_BUILD_KEY_FIELDS}
    key.update(overrides)
    return key


def test_a_legacy_store_is_admitted_on_reconstructed_evidence() -> None:
    decision = fbc.verify_store_identity(
        store_metadata=_store(),
        expected_build_key=_expected_key(),
        forward_contract=contract(),
    )
    assert decision["admitted"] is True
    assert decision["evidence"] == "reconstructed"
    assert decision["why"] == ""


def test_a_stale_declaration_hash_is_never_a_refusal() -> None:
    """The failure this whole module exists to remove."""

    store = _store()
    store["build_key"]["config_sha256"] = "0" * 64
    decision = fbc.verify_store_identity(
        store_metadata=store,
        expected_build_key=_expected_key(),
        forward_contract=contract(),
    )
    assert decision["admitted"] is True
    assert decision["original_full_config_sha256"] == "0" * 64


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset", "metaqa"),
        ("data_fingerprint_sha256", "0" * 64),
        ("regime", "R2"),
        ("queries", 100),
        ("per_seed_cap", 8),
        ("neighbour_scan_cap_per_seed", 2048),
        ("a64_mainline_family", "another_family"),
    ],
)
def test_a_store_built_for_a_different_cell_is_refused(field: str, value) -> None:
    decision = fbc.verify_store_identity(
        store_metadata=_store(),
        expected_build_key=_expected_key(**{field: value}),
        forward_contract=contract(),
    )
    assert decision["admitted"] is False
    assert decision["evidence"] == "refused"
    assert field in decision["scientific_key_differences"]


def test_every_scientific_key_field_is_exercised_by_that_test() -> None:
    """A field nothing perturbs is a field the loader does not really check."""

    assert set(fbc.SCIENTIFIC_BUILD_KEY_FIELDS) == {
        "dataset", "data_fingerprint_sha256", "regime", "queries",
        "per_seed_cap", "neighbour_scan_cap_per_seed", "a64_mainline_family",
    }
    assert "config_sha256" not in fbc.SCIENTIFIC_BUILD_KEY_FIELDS


def test_a_legacy_store_whose_formulas_have_moved_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reconstruction substitutes today's formulas, so this is the load-bearing check."""

    monkeypatch.setattr(m1a, "FEATURE_DAMPING", 0.9)
    decision = fbc.verify_store_identity(
        store_metadata=_store(),
        expected_build_key=_expected_key(),
        forward_contract=contract(),
    )
    assert decision["admitted"] is False
    assert "FEATURE_DAMPING" in decision["why"]


def test_a_legacy_store_with_no_build_commit_is_refused() -> None:
    decision = fbc.verify_store_identity(
        store_metadata=_store(source_commit=None),
        expected_build_key=_expected_key(),
        forward_contract=contract(),
    )
    assert decision["admitted"] is False
    assert "which formulas built it" in decision["why"]


def test_a_contract_bearing_store_is_checked_against_its_recorded_hash() -> None:
    forward = contract()
    store = _store(feature_build_contract_sha256=fbc.contract_sha256(forward))
    decision = fbc.verify_store_identity(
        store_metadata=store, expected_build_key=_expected_key(), forward_contract=forward
    )
    assert decision["admitted"] is True
    assert decision["evidence"] == "recorded"


def test_a_contract_bearing_store_with_the_wrong_hash_is_refused() -> None:
    store = _store(feature_build_contract_sha256="9" * 64)
    decision = fbc.verify_store_identity(
        store_metadata=store, expected_build_key=_expected_key(), forward_contract=contract()
    )
    assert decision["admitted"] is False
    assert "is not the contract this process would build" in decision["why"]


def test_the_reconstructed_path_does_not_claim_the_contract_hash_as_evidence() -> None:
    """It would be circular, and the docstring must not pretend otherwise."""

    import inspect

    doc = inspect.getdoc(fbc.verify_store_identity)
    assert "circular" in doc


# --------------------------------------------------------------------------
# The formula-constant snapshot: git on the host, comparison in the container
# --------------------------------------------------------------------------


def test_the_snapshot_matches_what_git_says_at_every_recorded_commit():
    """The one place the snapshot is held to the history it claims to record.

    A store's build contract is reconstructed with today's formula identity,
    and that substitution is only honest if those constants were the same when
    the store was built. A container cannot check that itself -- it has no
    repository -- so it compares against recorded values instead. Which means
    a wrong recording would admit a store that should be refused, and this is
    the test standing between those two things.
    """

    snapshot = json.loads(
        fbc.FORMULA_CONSTANT_SNAPSHOT_PATH.read_text(encoding="utf-8")
    )
    assert snapshot["commits"], "a snapshot recording no commits proves nothing"
    for commit, entry in snapshot["commits"].items():
        assert entry["constants"] == fbc.historical_formula_constants(commit), commit


def test_the_snapshot_covers_every_commit_an_m2_store_was_built_at():
    """Coverage read from the artifacts, not from the snapshot's own list.

    Checking the snapshot against itself would pass at any coverage, including
    none.
    """

    from scripts.m2b_formula_constants_snapshot import build_commits

    snapshot = json.loads(
        fbc.FORMULA_CONSTANT_SNAPSHOT_PATH.read_text(encoding="utf-8")
    )
    assert set(build_commits()) <= set(snapshot["commits"])


def test_an_unrecorded_commit_is_refused_not_waved_through(tmp_path):
    """"No evidence" must not resolve to "therefore fine".

    That reading would make the check a formality the moment a new build
    commit appeared -- which is exactly when it matters.
    """

    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"commits": {}}), encoding="utf-8")
    with pytest.raises(KeyError, match="records no formula constants"):
        fbc.formula_constants_unchanged_since("deadbeef", snapshot_path=path)


def test_a_missing_snapshot_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no record of what the formula constants"):
        fbc.formula_constants_unchanged_since(
            "deadbeef", snapshot_path=tmp_path / "absent.json"
        )


def test_a_partial_record_cannot_establish_the_claim(tmp_path):
    # Three of four constants matching is not "the formulas are unchanged".
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"commits": {"c0ffee": {"constants": {"CONTEXT_ARM": "TARGET_H1"}}}}),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="do not cover"):
        fbc.formula_constants_unchanged_since("c0ffee", snapshot_path=path)


def test_a_changed_constant_is_caught_against_this_processs_own_imports(tmp_path):
    """The half the container computes for itself.

    The snapshot supplies history; the running process supplies what it
    actually imported. A snapshot claiming a different damping factor must
    fail here, because the comparison is against a live value that no snapshot
    can influence.
    """

    live = fbc.live_formula_constants()
    path = tmp_path / "snapshot.json"
    path.write_text(
        json.dumps({"commits": {"c0ffee": {"constants": {**live, "FEATURE_DAMPING": 0.5}}}}),
        encoding="utf-8",
    )
    result = fbc.formula_constants_unchanged_since("c0ffee", snapshot_path=path)
    assert result["all_unchanged"] is False
    assert result["per_constant"]["FEATURE_DAMPING"]["unchanged"] is False
    assert result["per_constant"]["FEATURE_DAMPING"]["in_the_tree_now"] == live["FEATURE_DAMPING"]
    assert result["per_constant"]["CONTEXT_ARM"]["unchanged"] is True


def test_a_matching_snapshot_passes_and_says_where_each_half_came_from(tmp_path):
    live = fbc.live_formula_constants()
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps({"commits": {"c0ffee": {"constants": live}}}), encoding="utf-8")
    result = fbc.formula_constants_unchanged_since("c0ffee", snapshot_path=path)
    assert result["all_unchanged"] is True
    assert result["live_values_from"] == "this process's own imports"
    assert str(path) == result["historical_values_from"]


def test_the_real_build_commit_still_passes_through_the_snapshot_path():
    """The end-to-end case: M2's fourteen stores stay loadable.

    This is what the container will evaluate, and it now runs the same way
    here as it does there.
    """

    from scripts.m2b_formula_constants_snapshot import build_commits

    for commit in build_commits():
        result = fbc.formula_constants_unchanged_since(commit)
        assert result["all_unchanged"] is True, commit
        assert set(result["per_constant"]) == set(fbc.FORMULA_CONSTANT_SOURCES)


def test_the_check_needs_no_git_at_runtime(monkeypatch):
    """A container has src/ and scripts/ and no repository.

    The first M2B smoke died on `git show` after being billed for a GPU, so
    the absence of git is asserted rather than hoped for: this fails loudly if
    the historical half ever creeps back into the runtime path.
    """

    def _refuse(*args, **kwargs):
        raise AssertionError("the runtime path must not shell out to git")

    monkeypatch.setattr(fbc.subprocess, "run", _refuse)
    from scripts.m2b_formula_constants_snapshot import build_commits

    for commit in build_commits():
        assert fbc.formula_constants_unchanged_since(commit)["all_unchanged"] is True
