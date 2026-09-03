"""D10 fits the arm D9 declared and never reached, under a repaired systems rule.

Three things need attacking here, and only the first is inherited.

The first is the usual tensor discipline: `V` must differ from `B` in columns
5-7 and nowhere else, and the injected scalars must arrive as
`branch/(branch+1)` rather than divided a second time by a per-query maximum.
D10 imports those guards from D9 rather than restating them, so the tests that
matter are the ones proving the import is real -- that there is no second copy
of the transform which could have drifted after D9's divergence statistics
were known.

The second is the repaired rule. D9's abort compared two wall clocks and so
measured the host: D8 passed it at 82% of its build, D9 failed it at 124% on a
container running the same build 1.44 times slower. D10 charges the branch
kernel only, at p95 per query, against the historical build's p95 on the same
container. A rule rewritten after an abort, which then permits the aborted
work, is exactly the shape of result-driven rewriting -- so the gate's logic is
pinned here against its filed threshold in both directions, and the abort is
exercised end to end rather than assumed to be live.

The third is new. D9 persisted no feature artifact, so D10 rebuilds 4.85M rows
and has to prove the rebuild is D9's block before it is allowed to reach a
model. That check reads D9's recorded statistics; it is not a gate, and a test
asserts it raises rather than reporting when the rebuild disagrees.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import CompleteRetrievalDataset  # noqa: E402
from mp_retrieval.path_diversity import (  # noqa: E402
    HISTORICAL_NAMES,
    HOPS,
    REPLACEMENT_NAMES,
)
from scripts.run_graph_context_d6 import LOCAL_DIM  # noqa: E402
from scripts.run_graph_context_d10 import (  # noqa: E402
    BASE,
    CAUSAL_WORDING_RESTRICTION,
    COMPLETE_STATUS,
    D9_FINGERPRINT,
    DELTA_H_MINUS_B,
    DELTA_V_MINUS_B,
    DELTA_V_MINUS_H,
    HISTORICAL_ARM,
    REPLACEMENT_ARM,
    REUSED_ARMS,
    STORAGE_RESOLVABLE_BRANCH_COUNT,
    SYSTEMS_ABORT_RULE,
    SYSTEMS_STATUS,
    WORKSPACE_CONTRACT_BYTES,
    block_fingerprints,
    build_parser,
    run,
    storage_collapse_diagnostic,
    systems_gate,
    verify_reproduces_d9,
)
from test_graph_context_d1 import args_for  # noqa: E402
from test_graph_context_d1 import dataset as build_dataset  # noqa: E402
from test_graph_context_d4 import RRF_CONSTANT, write_rank_lists  # noqa: E402
from test_graph_context_d7 import d6_result  # noqa: E402, F401
from test_graph_context_d9 import synthetic as d9_synthetic  # noqa: E402

METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")


@contextmanager
def synthetic(data):
    """D9's patch set, extended to D10's own module."""
    import scripts.run_graph_context_d10 as module

    static = np.linspace(0.0, 1.0, int(data.num_nodes) * 7, dtype=np.float32).reshape(
        int(data.num_nodes), 7
    )
    original = (module.load_complete_dataset, module.load_or_build_static)
    module.load_complete_dataset = lambda *a, **k: data
    module.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        with d9_synthetic(data):
            yield module
    finally:
        module.load_complete_dataset, module.load_or_build_static = original


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


def d10_args(tmp_path: Path, data, d7_result: Path, **overrides):
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d10.json")
    parsed["d7_result"] = d7_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d7_result(tmp_path_factory, data, d6_result) -> Path:  # noqa: F811
    """A real D7, so the historical arm D10 reuses is genuine output."""
    import scripts.run_graph_context_d7 as d7_module
    from test_graph_context_d7 import d7_args

    with synthetic(data):
        seven = d7_args(tmp_path_factory.mktemp("d10_d7"), data, d6_result)
        d7_module.run(seven)
    return Path(seven.output)


@contextmanager
def _patched(name, value):
    """Swap one module-level name on D10 for the duration of a run."""
    import scripts.run_graph_context_d10 as module

    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


@contextmanager
def _forced(name, decide):
    """Force the systems gate to a fixed answer, keeping the real numbers.

    At synthetic scale the branch kernel and the historical build are within
    microseconds of each other -- the pure-Python `njit` fallback is used here,
    numba is not installed -- so which side of the gate a run lands on is a
    race. The branch is chosen explicitly in the fixtures, and the gate's own
    arithmetic is tested directly against its filed threshold instead.
    """
    import scripts.run_graph_context_d10 as module

    original = getattr(module, name)

    def patched(*a, **k):
        return decide(original(*a, **k))

    setattr(module, name, patched)
    try:
        yield
    finally:
        setattr(module, name, original)


def _fingerprint_of(result) -> dict:
    """What D9 would have recorded, had D9 been run on this substrate."""
    entries = list(result["mechanistic_comparison"]["columns"].values())
    precision = result["tensor_equivalence"]["storage_precision"]
    return {
        "candidate_rows": int(result["feature_build"]["candidate_rows"]),
        "maximum_branch_count": [int(e["maximum_branch_count"]) for e in entries],
        "distinct_scalar_values_before_cast": int(
            precision["distinct_scalar_values_before_cast"]
        ),
        "distinct_scalar_values_after_cast": int(
            precision["distinct_scalar_values_after_cast"]
        ),
        "pairs_reordered": [
            float(e["ordering"]["fraction_ordered_differently"]) for e in entries
        ],
        "nonzero_agreement": [float(e["nonzero_agreement"]) for e in entries],
    }


def _passthrough(*_a, **_k):
    return {"all_conditions_match": True, "why": "patched for the first pass"}


@pytest.fixture(scope="module")
def synthetic_fingerprint(tmp_path_factory, data, d7_result):
    """One pass with the reproduction check disabled, to learn this substrate.

    The real D9_FINGERPRINT describes 2Wiki. Running D10 against it here would
    only prove that a five-query synthetic graph is not 2Wiki. Measuring the
    substrate first and then requiring D10 to land on it exercises the check
    against genuine values, which is the property under test.
    """
    with synthetic(data), _patched("verify_reproduces_d9", _passthrough), _forced(
        "systems_gate", lambda report: report | {"fired": False}
    ):
        result = run(d10_args(tmp_path_factory.mktemp("d10_probe"), data, d7_result))
    return _fingerprint_of(result)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d7_result, synthetic_fingerprint):
    with synthetic(data), _patched("D9_FINGERPRINT", synthetic_fingerprint), _forced(
        "systems_gate", lambda report: report | {"fired": False}
    ):
        return run(d10_args(tmp_path_factory.mktemp("d10"), data, d7_result))


@pytest.fixture(scope="module")
def aborted_by_systems(tmp_path_factory, data, d7_result, synthetic_fingerprint):
    with synthetic(data), _patched("D9_FINGERPRINT", synthetic_fingerprint), _forced(
        "systems_gate", lambda report: report | {"fired": True}
    ):
        return run(d10_args(tmp_path_factory.mktemp("d10_abort"), data, d7_result))


# --- the transform is D9's, not a second copy ---


def test_the_transform_is_imported_from_d9_not_restated():
    """The one property that makes "no feature redesign" enforceable."""
    import scripts.run_graph_context_d9 as d9
    import scripts.run_graph_context_d10 as d10

    for name in (
        "build_path_diversity",
        "d9_local_block",
        "assert_d9_block_is_exact",
        "assert_the_injection_was_not_rescaled",
        "mechanistic_gate",
        "replacement_definition",
        "historical_paths_audit",
        "classify",
    ):
        assert getattr(d10, name) is getattr(d9, name), name


def test_the_bands_and_the_material_threshold_are_d9s():
    import scripts.run_graph_context_d9 as d9

    delta = dict.fromkeys(METRICS, 0.0)
    assert d9.classify(delta)["thresholds"]["material"] == d9.MATERIAL
    assert d9.MATERIAL == 0.005


def test_the_replacement_arm_is_renamed_but_defined_identically(completed):
    import scripts.run_graph_context_d9 as d9

    assert REPLACEMENT_ARM == "D10_PATH_DIVERSITY_13"
    assert REPLACEMENT_ARM != d9.REPLACEMENT_ARM
    arm = completed["arms"][REPLACEMENT_ARM]
    assert arm["identical_in_definition_to"].startswith(d9.REPLACEMENT_ARM)
    assert completed["replacement_definition"] == d9.replacement_definition()


# --- the repaired systems rule ---


def test_the_gate_charges_the_kernel_and_not_the_shared_extraction():
    """A kernel far faster than the build passes, however slow the sharing is."""
    report = systems_gate([1.0] * 20, [10.0] * 20, 4096)
    assert not report["fired"]
    assert report["replacement_kernel_p95_ms"] < report[
        "historical_structural_build_p95_ms"
    ]
    assert "shared candidate and graph extraction" in report["what_is_not_charged"]


def test_the_gate_fires_when_the_kernel_p95_exceeds_the_build_p95():
    report = systems_gate([100.0] * 20, [10.0] * 20, 4096)
    assert report["fired"]
    assert report["kernel_is_slower_than_the_build"]
    assert not report["workspace_exceeds_the_contract"]


def test_the_gate_fires_on_workspace_alone():
    report = systems_gate([1.0] * 20, [10.0] * 20, WORKSPACE_CONTRACT_BYTES + 1)
    assert report["fired"]
    assert report["workspace_exceeds_the_contract"]
    assert not report["kernel_is_slower_than_the_build"]


def test_the_workspace_backstop_is_new_and_the_shape_guarantee_is_not():
    """The 1 MiB number is D10's. The thing it backs up predates D10.

    tests/test_path_diversity.py:355 asserts the exact allocation identity --
    2 * n * ceil(|S|/64) words -- and has since D9. That is the guarantee with
    teeth. The absolute ceiling is a tripwire filed for D10, and calling it
    pre-existing would have been the easy false claim to make.
    """
    assert WORKSPACE_CONTRACT_BYTES == 1 << 20
    source = (REPO_ROOT / "tests" / "test_path_diversity.py").read_text(
        encoding="utf-8"
    )
    assert "2 * n * expected_words * 8" in source
    assert "1 << 20" not in source and "1048576" not in source


def test_the_rule_is_a_percentile_comparison_not_a_wall_clock_race():
    assert "p95" in SYSTEMS_ABORT_RULE
    assert "not charged" in SYSTEMS_ABORT_RULE
    report = systems_gate([1.0] * 20, [10.0] * 20, 4096)
    assert "same container" in report["why_p95_not_totals"] or "same clock" in report[
        "why_p95_not_totals"
    ]


def test_a_uniformly_slower_host_does_not_move_the_verdict():
    """The property D9's rule lacked: scale both clocks, keep the answer."""
    fast = systems_gate([1.0] * 20, [10.0] * 20, 4096)
    slow = systems_gate([3.0] * 20, [30.0] * 20, 4096)
    assert fast["fired"] == slow["fired"] is False
    assert fast["ratio"] == slow["ratio"]


def test_d9s_recorded_percentiles_would_not_have_fired_this_rule():
    """Filed in the declaration, so it cannot be presented as a discovery."""
    report = systems_gate([0.877] * 20, [8.432] * 20, 6368)
    assert not report["fired"]
    assert report["ratio"] < 1.0


# --- the rebuild has to be D9's block ---


def _comparison(maxima, reordered, agreement):
    return {
        "columns": {
            f"{HISTORICAL_NAMES[i]}_vs_{REPLACEMENT_NAMES[i]}": {
                "maximum_branch_count": maxima[i],
                "nonzero_agreement": agreement[i],
                "ordering": {"fraction_ordered_differently": reordered[i]},
            }
            for i in range(HOPS)
        }
    }


def _equivalence(before, after):
    return {
        "storage_precision": {
            "distinct_scalar_values_before_cast": before,
            "distinct_scalar_values_after_cast": after,
        }
    }


def _quantities(rows):
    return {"branch": np.zeros((rows, HOPS), dtype=np.int16)}


def _d9_shaped():
    return (
        _comparison(
            D9_FINGERPRINT["maximum_branch_count"],
            D9_FINGERPRINT["pairs_reordered"],
            D9_FINGERPRINT["nonzero_agreement"],
        ),
        _equivalence(
            D9_FINGERPRINT["distinct_scalar_values_before_cast"],
            D9_FINGERPRINT["distinct_scalar_values_after_cast"],
        ),
        _quantities(D9_FINGERPRINT["candidate_rows"]),
        D9_FINGERPRINT["candidate_rows"],
    )


def test_a_rebuild_matching_d9_verifies():
    report = verify_reproduces_d9(*_d9_shaped())
    assert report["all_conditions_match"]
    assert len(report["conditions_checked"]) == 2 + 3 * 3 + 2
    assert all(block["match"] for block in report["conditions_checked"].values())


@pytest.mark.parametrize(
    "hop,maximum", [(0, 8), (1, 114), (2, 254)]
)
def test_a_rebuild_with_a_different_branch_maximum_stops_the_stage(hop, maximum):
    _comp, equivalence, quantities, rows = _d9_shaped()
    maxima = list(D9_FINGERPRINT["maximum_branch_count"])
    maxima[hop] = maximum
    comparison = _comparison(
        maxima, D9_FINGERPRINT["pairs_reordered"], D9_FINGERPRINT["nonzero_agreement"]
    )
    with pytest.raises(RuntimeError, match="does not reproduce D9"):
        verify_reproduces_d9(comparison, equivalence, quantities, rows)


def test_a_rebuild_with_the_wrong_row_count_stops_the_stage():
    comparison, equivalence, quantities, _rows = _d9_shaped()
    with pytest.raises(RuntimeError, match="candidate_rows"):
        verify_reproduces_d9(comparison, equivalence, quantities, 4851275)


def test_a_rebuild_that_loses_the_storage_collapse_stops_the_stage():
    comparison, _equiv, quantities, rows = _d9_shaped()
    with pytest.raises(RuntimeError, match="distinct_scalar_values_after_cast"):
        verify_reproduces_d9(comparison, _equivalence(138, 79), quantities, rows)


def test_the_ordering_tolerance_is_tight_enough_to_catch_a_real_change():
    """The fractions are integer ratios, so the band is for accumulation order."""
    _comp, equivalence, quantities, rows = _d9_shaped()
    nudged = [value + 5e-10 for value in D9_FINGERPRINT["pairs_reordered"]]
    report = verify_reproduces_d9(
        _comparison(
            D9_FINGERPRINT["maximum_branch_count"],
            nudged,
            D9_FINGERPRINT["nonzero_agreement"],
        ),
        equivalence, quantities, rows,
    )
    assert report["all_conditions_match"]
    broken = [value + 1e-6 for value in D9_FINGERPRINT["pairs_reordered"]]
    with pytest.raises(RuntimeError, match="pairs_reordered"):
        verify_reproduces_d9(
            _comparison(
                D9_FINGERPRINT["maximum_branch_count"],
                broken,
                D9_FINGERPRINT["nonzero_agreement"],
            ),
            equivalence, quantities, rows,
        )


def test_the_check_says_it_is_not_a_gate():
    report = verify_reproduces_d9(*_d9_shaped())
    assert "not" in report["is_not_a_gate"].lower()
    assert "selection criterion" in report["is_not_a_gate"]
    assert report["on_failure"] == "stop and inspect, never fit around it"


def test_the_filed_fingerprint_is_the_one_d9_recorded():
    """Pinned, so a later edit cannot quietly loosen what a rebuild must match."""
    assert D9_FINGERPRINT["candidate_rows"] == 4851276
    assert D9_FINGERPRINT["maximum_branch_count"] == [7, 113, 255]
    assert D9_FINGERPRINT["distinct_scalar_values_before_cast"] == 138
    assert D9_FINGERPRINT["distinct_scalar_values_after_cast"] == 78
    assert D9_FINGERPRINT["pairs_reordered"] == [
        0.0009358088563998761,
        0.00370229089220594,
        0.008446842308984613,
    ]
    assert D9_FINGERPRINT["nonzero_agreement"] == [1.0, 1.0, 1.0]


# --- the float16 diagnostic, reported and not repaired ---


def test_the_storage_diagnostic_counts_rows_the_cast_cannot_separate():
    branch = np.array(
        [[0, 0, 0], [1, 2, 3], [STORAGE_RESOLVABLE_BRANCH_COUNT, 0, 0], [0, 0, 900]],
        dtype=np.int16,
    )
    report = storage_collapse_diagnostic(branch, branch.astype(np.float16))
    assert report["total_rows"] == 4
    assert report["total_affected_rows"] == 2
    assert report["fraction_of_all_rows"] == 0.5
    assert report["rows_with_any_nonzero_path"] == 3
    assert report["fraction_of_nonzero_path_rows"] == pytest.approx(2 / 3)
    assert report["by_hop"][0]["rows_affected"] == 1
    assert report["by_hop"][2]["rows_affected"] == 1


def test_a_block_below_the_boundary_reports_nothing_affected():
    branch = np.full((10, HOPS), STORAGE_RESOLVABLE_BRANCH_COUNT - 1, dtype=np.int16)
    report = storage_collapse_diagnostic(branch, branch.astype(np.float16))
    assert report["total_affected_rows"] == 0
    assert report["fraction_of_all_rows"] == 0.0
    assert report["fraction_of_nonzero_path_rows"] == 0.0


def test_the_diagnostic_reports_the_three_quantities_that_were_asked_for(completed):
    report = completed["float16_storage_diagnostic"]
    for key in (
        "total_affected_rows",
        "fraction_of_all_rows",
        "fraction_of_nonzero_path_rows",
    ):
        assert key in report, key
    assert report["dtype_was_not_changed_here"]
    # The synthetic block is float64, where the saturations stay separable; the
    # real 2Wiki block is float16. What is pinned here is that the diagnostic
    # reads the dtype in front of it and derives the boundary from that, which
    # is the bug a hardcoded 51 would have hidden.
    assert report["first_unresolvable_branch_count"] > 0
    assert report["boundary_matches_d9"] == (report["dtype"] == "float16")


def test_the_boundary_is_the_one_d9_measured():
    """51 is not a filed constant to trust; it falls out of float16."""
    from scripts.run_graph_context_d10 import _first_unresolvable_branch_count

    assert STORAGE_RESOLVABLE_BRANCH_COUNT == 51
    assert _first_unresolvable_branch_count(np.dtype("float16")) == 51
    assert _first_unresolvable_branch_count(np.dtype("float32")) > 51


def test_the_diagnostic_does_not_advantage_either_arm(completed):
    assert "same dtype" in completed["float16_storage_diagnostic"]["both_arms_pay_it"]


# --- hashes instead of a persisted tensor ---


def test_the_fingerprint_is_a_hash_of_the_block_not_of_its_repr():
    branch = np.arange(12, dtype=np.int16).reshape(4, 3)
    first = block_fingerprints(branch, branch.astype(np.float16))
    same = block_fingerprints(branch.copy(), branch.astype(np.float16))
    assert first["branch_counts_sha256"] == same["branch_counts_sha256"]
    changed = branch.copy()
    changed[2, 1] += 1
    assert (
        block_fingerprints(changed, changed.astype(np.float16))["branch_counts_sha256"]
        != first["branch_counts_sha256"]
    )
    assert first["rows"] == 4 and first["hops"] == 3


def test_the_run_records_a_hash_of_what_it_built(completed):
    block = completed["block_fingerprints"]
    assert len(block["branch_counts_sha256"]) == 64
    assert len(block["injected_columns_sha256"]) == 64
    assert block["branch_counts_dtype"] == "int16"
    assert block["rows"] == completed["feature_build"]["candidate_rows"]
    assert "no declared consumer" in block["the_tensor_is_not_persisted"]


# --- the tensor discipline, end to end ---


def test_exactly_one_arm_is_fitted(completed):
    assert completed["arms_trained_here"] == [REPLACEMENT_ARM]
    assert completed["arms_reused"] == list(REUSED_ARMS)
    assert completed["results"][REPLACEMENT_ARM]["retrained_here"]
    for arm in REUSED_ARMS:
        assert not completed["results"][arm]["retrained_here"]
        assert completed["results"][arm]["reused_via"] == "stage_d10"


def test_the_replacement_differs_from_the_historical_arm_in_the_path_columns_only(
    completed,
):
    equivalence = completed["tensor_equivalence"]
    assert equivalence["differs_from_historical_in"] == list(HISTORICAL_NAMES)
    assert equivalence["other_residual_columns_are_zero"]
    assert equivalence["max_abs_diff"] == 0.0
    for name, check in equivalence["columns"].items():
        assert check["elementwise_identical"], name


def test_the_injected_scalars_were_not_rescaled_a_second_time(completed):
    injection = completed["injection_discipline"]
    assert injection["elementwise_identical"]
    assert injection["the_replacement_bypasses_the_normaliser"]
    assert injection["normalised_columns_extended_to_the_replacement"] is False
    for name, column in injection["columns"].items():
        assert column["max_abs_diff_against_branch_over_branch_plus_one"] == 0.0, name
        assert 0.0 <= column["column_minimum"] <= column["column_maximum"] < 1.0, name
    assert completed["normalisation"][
        "normalised_columns_extended_to_the_replacement"
    ] is False


def test_the_two_collapse_boundaries_agree(completed):
    """D9 computes this inline in its tensor guard; D10 computes it again.

    They are separate implementations reading the same dtype, so a
    disagreement would mean one of them is describing storage that is not
    there.
    """
    guard = completed["tensor_equivalence"]["storage_precision"]
    diagnostic = completed["float16_storage_diagnostic"]
    assert (
        guard["largest_branch_count_still_distinguishable_from_its_successor"]
        == diagnostic["first_unresolvable_branch_count"]
    )
    assert guard["block_dtype"] == diagnostic["dtype"]


def test_the_three_arms_share_one_architecture(completed):
    counts = {
        arm: completed["results"][arm]["parameters"]
        for arm in (BASE, HISTORICAL_ARM, REPLACEMENT_ARM)
    }
    assert len(set(counts.values())) == 1
    widths = {
        arm: completed["results"][arm]["local_dim"]
        for arm in (BASE, HISTORICAL_ARM, REPLACEMENT_ARM)
    }
    assert set(widths.values()) == {LOCAL_DIM}
    assert completed["ablation"]["columns_added"] == 0
    assert not completed["ablation"]["architecture_changed_between_arms"]


def test_all_three_increments_are_reported(completed):
    for name in (DELTA_H_MINUS_B, DELTA_V_MINUS_B, DELTA_V_MINUS_H):
        assert name in completed["increments"], name
        for metric in METRICS:
            assert metric in completed["increments"][name], (name, metric)


def test_h_minus_b_reproduces_d7_without_refitting_h(completed):
    report = completed["h_minus_b_reproduces_d7"]
    assert not report["refitted_h"]
    assert report["max_abs_diff"] <= report["tolerance"]
    assert report["d7_delta_paths"].keys() == report["d10_h_minus_b"].keys()


def test_the_verdict_carries_the_causal_wording_restriction(completed):
    restriction = completed["verdict"]["wording_restriction"]
    assert restriction == CAUSAL_WORDING_RESTRICTION
    assert "outperforms" in restriction["if_v_beats_h_say"]
    assert "caused" in restriction["if_v_beats_h_do_not_say"]
    assert "inherently useful" in restriction["if_h_beats_v_do_not_say"]
    assert "transform" in restriction["why"]


def test_the_verdict_is_one_of_the_three_filed_bands(completed):
    import scripts.run_graph_context_d9 as d9

    assert completed["verdict"]["label"] in {d9.IMPROVES, d9.PARETO_MATCHES, d9.FAILS}
    assert completed["verdict"]["trained"]
    assert completed["verdict"]["thresholds"]["material"] == d9.MATERIAL


def test_the_stage_records_that_it_is_the_last_2wiki_only_feature_stage(completed):
    assert completed["this_is_the_last_2wiki_only_feature_stage"]
    assert completed["status"] == COMPLETE_STATUS


def test_the_run_reports_that_no_reusable_artifact_existed(completed):
    rule = completed["reuse_rule"]
    assert rule["a_durable_d9_artifact_was_looked_for"]
    assert "none" in rule["what_was_found"].lower()
    assert "4,851,276" in rule["consequence"]
    assert "forced by D9's failure to persist" in rule["consequence"]


def test_the_overlap_diagnostic_is_absent_by_design_not_by_omission(completed):
    construction = completed["branch_diversity_construction"]
    assert construction["overlap_diagnostic_computed"] is False
    assert construction["overlap_diagnostic_seconds"] is None
    assert "by design" in completed["mechanistic_comparison"][
        "the_overlap_diagnostic_was_not_recomputed"
    ]


def test_the_comparison_is_the_validation_block_and_reads_as_a_check(completed):
    comparison = completed["mechanistic_comparison"]
    assert comparison["measured_on"] == "validation feature construction"
    assert "NOT a selection criterion" in comparison["read_as"]
    assert "also_over_every_opened_query" not in comparison


# --- the abort is live, not decorative ---


def test_the_systems_abort_stops_before_any_arm_is_fitted(aborted_by_systems):
    assert aborted_by_systems["status"] == SYSTEMS_STATUS
    assert aborted_by_systems["arms_trained_here"] == []
    assert aborted_by_systems["results"] == {}
    assert "ladder" not in aborted_by_systems
    assert "increments" not in aborted_by_systems


def test_the_abort_reports_the_same_label_d9_now_carries(aborted_by_systems):
    label = aborted_by_systems["verdict"]["label"]
    assert label == (
        "PATH REPLACEMENT EFFECTIVENESS UNTESTED -- ABORTED BY "
        "PRE-REGISTERED SYSTEMS GATE"
    )
    assert not aborted_by_systems["verdict"]["trained"]
    assert "No effectiveness claim" in aborted_by_systems["not_established"]


def test_the_abort_still_reports_everything_it_measured(aborted_by_systems):
    """Stopping is not the same as reporting nothing."""
    for key in (
        "mechanistic_comparison",
        "reproduces_d9",
        "block_fingerprints",
        "float16_storage_diagnostic",
        "systems_gate",
        "tensor_equivalence",
        "reuse",
        "deterministic_reproduction",
    ):
        assert key in aborted_by_systems, key
    assert aborted_by_systems["reproduces_d9"]["all_conditions_match"]


def test_the_completed_run_records_the_gate_it_passed(completed):
    gate = completed["systems_gate"]
    assert gate["rule"] == SYSTEMS_ABORT_RULE
    assert not gate["fired"]
    assert gate["workspace_contract_bytes"] == WORKSPACE_CONTRACT_BYTES
    assert gate["temporary_workspace_bytes"] <= WORKSPACE_CONTRACT_BYTES


def test_d9s_old_abort_quantity_is_still_recorded_but_is_not_the_rule(completed):
    construction = completed["branch_diversity_construction"]
    assert construction["whole_construction_share_of_the_historical_build"] is not None
    assert "not the rule any more" in construction["what_that_last_number_is_for"]
    assert construction["abort_rule"] == SYSTEMS_ABORT_RULE


def test_the_stage_refuses_the_test_split(tmp_path, data, d7_result):
    args = d10_args(tmp_path, data, d7_result)
    args.splits = ["train", "test"]
    with pytest.raises(ValueError, match="test split is not read"):
        run(args)


def test_the_stage_refuses_a_missing_d7(tmp_path, data, d7_result):
    args = d10_args(tmp_path, data, d7_result)
    args.d7_result = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError, match="absent"):
        run(args)


def test_the_declaration_was_filed_before_this_module_existed():
    """The systems rule lives in the config, and says why it changed."""
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "graph_context_pilot.yaml").read_text(encoding="utf-8")
    )
    rule = config["stages"]["D10"]["corrected_systems_rule_filed_before_fitting"]
    assert "p95" in rule["abort_condition"]
    assert rule["workspace_contract_bytes"] == WORKSPACE_CONTRACT_BYTES
    assert "would NOT have aborted D9" in rule[
        "stated_in_advance_so_it_cannot_be_presented_as_a_discovery"
    ]
    assert "result-driven rewriting" in rule[
        "stated_in_advance_so_it_cannot_be_presented_as_a_discovery"
    ]


def test_the_declaration_admits_no_artifact_was_reusable():
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "graph_context_pilot.yaml").read_text(encoding="utf-8")
    )
    reuse = config["stages"]["D10"]["reuse_rule"]
    assert reuse["what_was_found"].startswith("NONE")
    assert "not chosen for stage symmetry" in reuse["consequence"]


def test_the_declaration_forbids_the_follow_ups():
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "graph_context_pilot.yaml").read_text(encoding="utf-8")
    )
    forbidden = config["stages"]["D10"]["does_not_authorise"]
    for banned in (
        "weighted support",
        "path variants",
        "diffusion replacement",
        "neighbourhood replacement",
        "another 2Wiki seed",
        "resuming E2",
        "opening F",
        "GNN",
        "TARGET_H1",
        "workspace migration",
    ):
        assert banned in forbidden, banned
    assert config["stages"]["D10"]["is_the_last_2wiki_only_feature_stage"]


def test_the_filed_fingerprint_and_the_declared_one_are_the_same_numbers():
    """The declaration has to be machine-readable, not prose that looks filed.

    The first version of this block was indented one level too deep and was
    swallowed into the folded scalar above it, so every number in it parsed as
    part of a sentence. It read correctly and asserted nothing.
    """
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "graph_context_pilot.yaml").read_text(encoding="utf-8")
    )
    declared = config["stages"]["D10"]["d9_recorded_for_comparison"]
    assert isinstance(declared, dict)
    assert declared["candidate_rows"] == D9_FINGERPRINT["candidate_rows"]
    assert declared["pairs_reordered_by_hop"] == D9_FINGERPRINT["pairs_reordered"]
    assert declared["nonzero_agreement_by_hop"] == D9_FINGERPRINT["nonzero_agreement"]
    assert (
        declared["maximum_branch_count_by_hop"]
        == D9_FINGERPRINT["maximum_branch_count"]
    )
    assert (
        declared["distinct_scalar_values_before_cast"]
        == D9_FINGERPRINT["distinct_scalar_values_before_cast"]
    )
    assert (
        declared["distinct_scalar_values_after_cast"]
        == D9_FINGERPRINT["distinct_scalar_values_after_cast"]
    )


def test_every_declared_d10_field_is_a_real_mapping_not_swallowed_prose():
    """The same failure mode, swept across the whole D10 block."""
    import yaml

    config = yaml.safe_load(
        (REPO_ROOT / "configs" / "graph_context_pilot.yaml").read_text(encoding="utf-8")
    )
    d10 = config["stages"]["D10"]
    for name in (
        "the_frozen_replacement",
        "reuse_rule",
        "corrected_systems_rule_filed_before_fitting",
        "d9_recorded_for_comparison",
        "arms",
        "frozen_operating_point",
        "column_layout",
        "comparisons",
        "interpretation_fixed_in_advance",
        "causal_wording_restriction",
        "float16_diagnostic",
        "projected_cost",
        "execution_order",
    ):
        assert isinstance(d10[name], dict), name
    assert isinstance(d10["tensor_guards_required_before_fitting"], list)
    for value in d10.values():
        if isinstance(value, str):
            assert ": >-" not in value, value[:60]


def test_a_reproduction_failure_still_writes_its_evidence(
    tmp_path, data, d7_result, synthetic_fingerprint
):
    """Stopping is right. Stopping with an empty volume is not."""
    broken = dict(synthetic_fingerprint)
    broken["candidate_rows"] = synthetic_fingerprint["candidate_rows"] + 1
    args = d10_args(tmp_path, data, d7_result)
    with synthetic(data), _patched("D9_FINGERPRINT", broken), _forced(
        "systems_gate", lambda report: report | {"fired": False}
    ), pytest.raises(RuntimeError, match="does not reproduce D9"):
        run(args)
    written = json.loads(Path(args.output).read_text(encoding="utf-8"))
    assert written["status"] == "GRAPH_CONTEXT_D10_STOPPED_AT_REPRODUCTION_FAILURE"
    assert written["reproduces_d9"]["all_conditions_match"] is False
    assert "candidate_rows" in written["reproduces_d9"]["failure"]
    assert written["arms_trained_here"] == []
    assert written["results"] == {}
    # The evidence someone would need to diagnose it is on disk, not lost.
    for key in ("mechanistic_comparison", "tensor_equivalence", "block_fingerprints"):
        assert key in written, key


def test_the_completed_run_records_the_check_it_passed(completed):
    assert completed["reproduces_d9"]["all_conditions_match"]
    assert completed["reproduces_d9"]["conditions_checked"]
