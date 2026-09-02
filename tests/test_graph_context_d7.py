"""D7 attributes a ~0.9 R@5 effect to one of four families, so the things that
could quietly misattribute it are what this file attacks.

Three failure modes matter. An arm that leaks a second family's columns would
credit the wrong family, so every arm is checked against D6's full block inside
its own family and against D6's base everywhere else. The four families must
tile the six columns exactly once, or the additivity residual measures the
bookkeeping rather than the interaction. And because `B` and `F` are reused from
another container rather than refitted, the guards that stand in for a
reproduction run are tested by breaking them: a substrate that does not
reproduce D6's recorded statistics must stop the stage, not be trained around.

The pre-registered family thresholds are exercised as a table. They were fixed
before any arm was fitted and the tests pin the boundaries so that a later edit
to the verdict rule shows up as a failing test rather than a nicer result.
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
from mp_retrieval.graph_context import NORMALISED_COLUMNS  # noqa: E402
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from scripts.run_graph_context_d3 import DISTANCE_COLUMNS  # noqa: E402
from scripts.run_graph_context_d5 import prior_value_columns  # noqa: E402
from scripts.run_graph_context_d6 import (  # noqa: E402
    BASE_ARM,
    FULL_ARM,
    HISTORICAL_COLUMNS,
    LOCAL_DIM,
    PRIOR_COLUMNS,
    RESIDUAL_COLUMNS,
    d6_local_block,
)
from scripts.run_graph_context_d7 import (  # noqa: E402
    COMPLETE_STATUS,
    CONTEXT,
    D7_ARMS,
    DOMINANT_SHARE,
    FAMILIES,
    JOINT_INCREMENT,
    NEGLIGIBLE,
    NEGLIGIBLE_BAND,
    PROMISING,
    PROMISING_R5,
    REUSED_ARMS,
    TRADEOFF,
    assert_families_partition_the_residual,
    assert_family_blocks_are_exact,
    build_parser,
    classify_family,
    family_by_arm,
    residual_column_occupancy,
    run,
    verify_reuse,
    verify_substrate_reproduces_d6,
)
from test_graph_context_d1 import args_for, dataset as build_dataset, local_block  # noqa: E402
from test_graph_context_d4 import (  # noqa: E402
    DENSE_SLICE,
    RRF_CONSTANT,
    SPLADE_SLICE,
    write_rank_lists,
)

METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")
NODES = 48


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


@contextmanager
def synthetic(data):
    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    import scripts.run_graph_context_d4 as d4_module
    import scripts.run_graph_context_d5 as d5_module
    import scripts.run_graph_context_d6 as d6_module
    import scripts.run_graph_context_d7 as module

    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    patched = (module, d6_module, d5_module, d4_module, d3_module, d2_module)
    original = [(m.load_complete_dataset, m.load_or_build_static) for m in patched]
    for m in patched:
        m.load_complete_dataset = lambda *a, **k: data
        m.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        yield module
    finally:
        for m, (loader, builder) in zip(patched, original, strict=True):
            m.load_complete_dataset, m.load_or_build_static = loader, builder


def d7_args(tmp_path: Path, data, d6_result: Path, **overrides):
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d7.json")
    parsed["d6_result"] = d6_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d6_result(tmp_path_factory, data) -> Path:
    """A real D2 through D6, so the control D7 reuses is genuine output."""

    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    import scripts.run_graph_context_d4 as d4_module
    import scripts.run_graph_context_d5 as d5_module
    import scripts.run_graph_context_d6 as d6_module
    from test_graph_context_d2 import d2_args
    from test_graph_context_d3 import d3_args
    from test_graph_context_d4 import d4_args
    from test_graph_context_d5 import d5_args
    from test_graph_context_d6 import d6_args

    with synthetic(data):
        two = d2_args(tmp_path_factory.mktemp("d7_d2"), data)
        d2_module.run(two)
        three = d3_args(tmp_path_factory.mktemp("d7_d3"), data, Path(two.output))
        d3_module.run(three)
        four = d4_args(tmp_path_factory.mktemp("d7_d4"), data, Path(three.output))
        d4_module.run(four)
        five = d5_args(
            tmp_path_factory.mktemp("d7_d5"), data, Path(three.output), Path(four.output)
        )
        d5_module.run(five)
        six = d6_args(tmp_path_factory.mktemp("d7_d6"), data, Path(five.output))
        d6_module.run(six)
    return Path(six.output)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d6_result):
    with synthetic(data):
        return run(d7_args(tmp_path_factory.mktemp("d7"), data, d6_result))


@pytest.fixture(scope="module")
def blocks(data):
    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    values, _prior = prior_value_columns(
        data.queries, ptr, dense, splade, constant=RRF_CONSTANT, dtype=local.dtype
    )
    base = d6_local_block(local, values, keep=DISTANCE_COLUMNS)
    full = d6_local_block(local, values, keep=HISTORICAL_COLUMNS)
    arms = {f.arm: d6_local_block(local, values, keep=f.keep()) for f in FAMILIES}
    return local, values, base, full, arms


# --- the partition -----------------------------------------------------------


def test_the_four_families_tile_the_six_columns_exactly_once():
    partition = assert_families_partition_the_residual()
    covered: list[int] = []
    for family in FAMILIES:
        covered.extend(family.columns)
    assert sorted(covered) == sorted(RESIDUAL_COLUMNS)
    assert len(covered) == len(set(covered)) == 6
    assert partition["disjoint"] and partition["covers_every_residual_column"]


def test_the_families_are_the_declared_ones():
    assert {f.key: f.column_names for f in FAMILIES} == {
        "SUPPORT": ["seed_connections"],
        "PATHS": ["paths_length_1", "paths_length_2", "paths_length_3"],
        "DIFFUSION": ["personalized_pagerank"],
        "NEIGHBOURHOOD": ["common_out_neighbors_with_seed_neighborhood"],
    }
    assert D7_ARMS == (
        "D7_SUPPORT_13", "D7_PATHS_13", "D7_DIFFUSION_13", "D7_NEIGHBOURHOOD_13"
    )


def test_no_family_claims_a_distance_or_prior_column():
    for family in FAMILIES:
        assert set(family.columns).isdisjoint(DISTANCE_COLUMNS), family.key
        assert set(family.columns).isdisjoint(PRIOR_COLUMNS), family.key
        assert set(family.keep()) >= set(DISTANCE_COLUMNS), family.key


def test_an_overlapping_partition_would_be_refused(monkeypatch):
    """Two families sharing a column would double-count it in the sum."""

    import scripts.run_graph_context_d7 as module

    greedy = module.Family("SUPPORT", "D7_SUPPORT_13", (4, 5), "…")
    monkeypatch.setattr(module, "FAMILIES", (greedy,) + module.FAMILIES[1:])
    with pytest.raises(RuntimeError, match="overlap"):
        module.assert_families_partition_the_residual()


def test_an_incomplete_partition_would_be_refused(monkeypatch):
    """A column no family carries would hide inside the additivity residual."""

    import scripts.run_graph_context_d7 as module

    monkeypatch.setattr(module, "FAMILIES", module.FAMILIES[:3])
    with pytest.raises(RuntimeError, match="not the six residual columns"):
        module.assert_families_partition_the_residual()


def test_every_family_says_what_surviving_would_mean():
    """Reading "PATHS survived" as "keep the walk counts" is the failure mode."""

    for family in FAMILIES:
        assert "NOT" in family.means, family.key
        assert len(family.means) > 60, family.key
    assert "edge count" in family_by_arm("D7_SUPPORT_13").means
    assert "walk counts" in family_by_arm("D7_PATHS_13").means
    assert "iterative PPR" in family_by_arm("D7_DIFFUSION_13").means
    assert "common-neighbour" in family_by_arm("D7_NEIGHBOURHOOD_13").means


# --- the blocks --------------------------------------------------------------


def test_every_arm_is_the_base_plus_exactly_its_own_family(blocks):
    local, values, base, full, arms = blocks
    for family in FAMILIES:
        block = arms[family.arm]
        assert block.shape[1] == LOCAL_DIM
        mine = list(family.columns)
        others = [c for c in RESIDUAL_COLUMNS if c not in family.columns]
        assert np.array_equal(block[:, list(DISTANCE_COLUMNS)], base[:, list(DISTANCE_COLUMNS)])
        assert np.array_equal(block[:, list(PRIOR_COLUMNS)], values)
        assert np.array_equal(block[:, mine], full[:, mine])
        assert not block[:, others].any()


def test_the_invariants_pass_on_honest_blocks(blocks):
    local, values, base, full, arms = blocks
    occupancy = residual_column_occupancy(full)
    proof = assert_family_blocks_are_exact(arms, base, full, values, occupancy)
    assert proof["every_arm_max_abs_diff"] == 0.0
    for arm, entry in proof["arms"].items():
        assert entry["other_residual_columns_are_zero"] is True
        for name, check in entry["columns"].items():
            assert check["elementwise_identical"] is True, (arm, name)
            assert check["max_abs_diff"] == 0.0, (arm, name)


def test_an_arm_that_leaks_another_family_is_refused(blocks):
    """The misattribution this stage exists to avoid."""

    local, values, base, full, arms = blocks
    occupancy = residual_column_occupancy(full)
    leaky = dict(arms)
    contaminated = np.array(arms["D7_SUPPORT_13"], copy=True)
    contaminated[:, 8] = full[:, 8]  # PPR smuggled into the SUPPORT arm
    leaky["D7_SUPPORT_13"] = contaminated
    with pytest.raises(RuntimeError, match="not exact"):
        assert_family_blocks_are_exact(leaky, base, full, values, occupancy)


def test_an_arm_missing_its_own_family_is_refused(blocks):
    local, values, base, full, arms = blocks
    occupancy = residual_column_occupancy(full)
    empty = dict(arms)
    hollow = np.array(arms["D7_PATHS_13"], copy=True)
    hollow[:, [5, 6, 7]] = 0.0
    empty["D7_PATHS_13"] = hollow
    with pytest.raises(RuntimeError, match="not exact"):
        assert_family_blocks_are_exact(empty, base, full, values, occupancy)


def test_perturbed_distance_geometry_is_refused(blocks):
    local, values, base, full, arms = blocks
    occupancy = residual_column_occupancy(full)
    broken = dict(arms)
    rotated = np.array(arms["D7_DIFFUSION_13"], copy=True)
    rotated[:, list(DISTANCE_COLUMNS)] = rotated[:, [3, 0, 1, 2]]
    broken["D7_DIFFUSION_13"] = rotated
    with pytest.raises(RuntimeError, match="not exact"):
        assert_family_blocks_are_exact(broken, base, full, values, occupancy)


def test_a_perturbed_prior_is_refused(blocks):
    local, values, base, full, arms = blocks
    occupancy = residual_column_occupancy(full)
    broken = dict(arms)
    shifted = np.array(arms["D7_NEIGHBOURHOOD_13"], copy=True)
    shifted[0, PRIOR_COLUMNS[0]] = shifted[0, PRIOR_COLUMNS[0]] + 1
    broken["D7_NEIGHBOURHOOD_13"] = shifted
    with pytest.raises(RuntimeError, match="not exact"):
        assert_family_blocks_are_exact(broken, base, full, values, occupancy)


def test_the_occupancy_count_is_the_one_d6_recorded(blocks):
    """The statistic that stands in for a reproduction run."""

    local, values, base, full, arms = blocks
    occupancy = residual_column_occupancy(full)
    expected = int((np.asarray(full[:, list(RESIDUAL_COLUMNS)], dtype=np.float64) != 0.0).sum())
    assert occupancy["total_nonzero_entries"] == expected
    assert set(occupancy["nonzero_rows_by_column"]) == {
        LOCAL_FEATURE_NAMES[c] for c in RESIDUAL_COLUMNS
    }


# --- reuse instead of a fifth run --------------------------------------------


def test_exactly_four_arms_are_fitted(completed):
    assert completed["arms_trained_here"] == list(D7_ARMS)
    assert len(completed["arms_trained_here"]) == 4
    assert completed["arms_reused"] == list(REUSED_ARMS)
    for arm in D7_ARMS:
        assert completed["results"][arm]["retrained_here"] is True
        assert completed["results"][arm]["measured_in"] == "stage_d7"
    for arm in REUSED_ARMS:
        assert completed["results"][arm]["retrained_here"] is False
        assert completed["results"][arm]["measured_in"] == "stage_d6"
        assert completed["results"][arm]["reused_via"] == "stage_d7"


def test_the_reused_rows_are_d6s_rows(completed, d6_result):
    d6 = json.loads(Path(d6_result).read_text(encoding="utf-8"))
    for arm in REUSED_ARMS:
        assert completed["results"][arm]["validation"] == d6["results"][arm]["validation"]
        assert completed["ladder"][arm] == d6["ladder"][arm]


def test_the_reuse_is_called_a_causal_control(completed):
    assert completed["reuse"]["role"] == "EXACT MATCHED CAUSAL CONTROL, REUSED NOT REFITTED"
    assert completed["reuse"]["new_runs"] == 4
    assert completed["reuse"]["all_conditions_match"] is True
    assert "could not change any" in completed["reuse"]["why_not_five"]


def test_an_incomplete_d6_is_refused(completed, d6_result, tmp_path, data):
    d6 = json.loads(Path(d6_result).read_text(encoding="utf-8"))
    d6["status"] = "GRAPH_CONTEXT_D6_IN_PROGRESS"
    with pytest.raises(RuntimeError, match="not complete"):
        verify_reuse(d6, completed, argparse.Namespace())


def test_a_d6_arm_that_was_itself_reused_is_refused(completed, d6_result):
    d6 = json.loads(Path(d6_result).read_text(encoding="utf-8"))
    d6["results"][BASE_ARM]["retrained_here"] = False
    with pytest.raises(RuntimeError, match="cannot anchor D7"):
        verify_reuse(d6, completed, argparse.Namespace())


@pytest.mark.parametrize(
    "path, replacement",
    [
        (("local_dim",), 10),
        (("parameter_accounting", "head_width"), 60),
        (("parameter_accounting", "d6_parameters"), 213506),
        (("training", "seed"), 7),
        (("splits", "validation_reported"), 1),
        (("normalisation", "extended_to_the_new_columns"), True),
        (("results", BASE_ARM, "local_dim"), 10),
    ],
)
def test_a_materially_different_d6_is_refused(completed, d6_result, path, replacement):
    """Every one of these would make `arm - D6_BASE_13` mean something else."""

    d6 = json.loads(Path(d6_result).read_text(encoding="utf-8"))
    node = d6
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = replacement
    args = argparse.Namespace(
        dataset=completed["dataset"],
        seed=completed["training"]["seed"],
        data_fingerprint_sha256=completed["data_fingerprint_sha256"],
        rrf_constant=RRF_CONSTANT,
    )
    with pytest.raises(RuntimeError, match="cannot reuse D6"):
        verify_reuse(d6, completed, args)


@pytest.mark.parametrize(
    "key, replacement",
    [
        ("full_residual_nonzero_entries", 1),
        ("candidate_rows", 1),
        ("rows_ranked_by_both", 1),
        ("seed_identity_proof", {"tampered": True}),
    ],
)
def test_a_substrate_that_does_not_reproduce_d6_stops_the_stage(
    completed, d6_result, blocks, key, replacement
):
    """§14: stop and inspect, never refit around a failed reuse guard."""

    _local, _values, _base, full, _arms = blocks
    d6 = json.loads(Path(d6_result).read_text(encoding="utf-8"))
    for holder in (
        d6["tensor_equivalence"], d6["feature_build"], d6["graded_retrieval_prior"], d6
    ):
        if key in holder:
            holder[key] = replacement
    occupancy = residual_column_occupancy(full)
    with pytest.raises(RuntimeError, match="STOP AND INSPECT"):
        verify_substrate_reproduces_d6(d6, completed, occupancy)


def test_the_reproduction_guard_passes_on_the_real_run(completed):
    reproduction = completed["deterministic_reproduction"]
    assert reproduction["all_conditions_match"] is True
    assert reproduction["on_failure"] == "stop and inspect, never refit around it"
    for name, check in reproduction["conditions_checked"].items():
        assert check["match"] is True, name


# --- the increments ----------------------------------------------------------


def test_every_family_increment_is_arithmetic(completed):
    base = completed["ladder"][BASE_ARM]
    for family in FAMILIES:
        block = completed["increments"][family.increment]
        assert block["from"] == BASE_ARM and block["to"] == family.arm
        for metric in base:
            expected = completed["ladder"][family.arm][metric] - base[metric]
            assert block[metric] == pytest.approx(expected, abs=1e-12), (family.key, metric)


def test_the_joint_increment_is_d6s_own_number(completed, d6_result):
    d6 = json.loads(Path(d6_result).read_text(encoding="utf-8"))
    joint = completed["joint_increment"]
    assert joint["measured_in"] == "stage_d6"
    for metric in METRICS:
        if metric in d6["increments"][JOINT_INCREMENT]:
            assert joint[metric] == pytest.approx(
                d6["increments"][JOINT_INCREMENT][metric], abs=1e-12
            )


def test_the_additivity_residual_is_arithmetic(completed):
    additivity = completed["additivity"]
    for metric, joint in additivity["joint"].items():
        summed = sum(
            completed["increments"][f.increment][metric] for f in FAMILIES
        )
        assert additivity["sum_of_individual_families"][metric] == pytest.approx(
            summed, abs=1e-12
        )
        assert additivity["additivity_residual"][metric] == pytest.approx(
            joint - summed, abs=1e-12
        )


def test_the_additivity_residual_is_descriptive_only(completed):
    assert completed["additivity"]["descriptive_only"] is True
    assert "not" in completed["additivity"]["not_an_interaction_claim"].lower()
    assert set(completed["additivity"]["how_to_read"]) == {
        "residual_near_zero", "large_positive_residual", "large_negative_residual"
    }


def test_the_attribution_picks_the_real_extremes(completed):
    increments = {f.key: completed["increments"][f.increment] for f in FAMILIES}
    attribution = completed["attribution"]
    assert attribution["most_recall@5_gain"]["family"] == max(
        increments, key=lambda key: increments[key]["recall@5"]
    )
    assert attribution["most_recall@1_loss"]["family"] == min(
        increments, key=lambda key: increments[key]["recall@1"]
    )
    assert attribution["most_mrr_loss"]["family"] == min(
        increments, key=lambda key: increments[key]["mrr"]
    )
    for entry in attribution.values():
        if isinstance(entry, dict):
            assert len(entry["ranking"]) == 4


# --- the pre-registered verdicts ---------------------------------------------


def delta(**overrides) -> dict[str, float]:
    row = dict.fromkeys(METRICS, 0.0)
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    "row, expected",
    [
        (delta(), NEGLIGIBLE),
        (delta(**{"recall@5": 0.0001}), NEGLIGIBLE),
        (delta(**{"recall@5": 0.004}), NEGLIGIBLE),
        (delta(**{"recall@5": PROMISING_R5}), PROMISING),
        (delta(**{"recall@5": 0.009}), PROMISING),
        (delta(**{"recall@20": 0.006, "full_coverage@20": 0.008}), PROMISING),
        (delta(**{"recall@20": 0.006, "recall@5": -0.02}), NEGLIGIBLE),
        (delta(**{"recall@5": 0.009, "recall@1": -0.02}), TRADEOFF),
        (delta(**{"recall@5": 0.009, "mrr": -0.03}), TRADEOFF),
        (delta(**{"recall@5": 0.009, "recall@1": -0.009}), TRADEOFF),
        (delta(**{"recall@5": 0.009, "recall@1": -0.004}), PROMISING),
        (delta(**{"recall@5": -0.02}), NEGLIGIBLE),
    ],
)
def test_the_family_verdict_follows_the_registered_thresholds(row, expected):
    assert classify_family(row)["label"] == expected


def test_a_family_can_qualify_on_depth_alone():
    verdict = classify_family(delta(**{"full_coverage@20": 0.02}))
    assert verdict["qualified_on_depth"] is True
    assert verdict["qualified_on_r5"] is False
    assert verdict["label"] == PROMISING


def test_near_zero_is_recorded_even_when_it_is_not_the_verdict():
    assert classify_family(delta())["near_zero_effectiveness"] is True
    assert classify_family(delta(**{"recall@5": 0.02}))["near_zero_effectiveness"] is False
    assert classify_family(delta(**{"recall@5": -0.02}))["harmful"] is True


def test_the_verdict_never_claims_significance(completed):
    for key, verdict in completed["family_classification"].items():
        assert verdict["label"] in {PROMISING, TRADEOFF, NEGLIGIBLE}, key
        assert "not a statistical significance claim" in verdict["not_a_significance_claim"]
        assert verdict["means_if_it_survives"], key
        assert "NOT" in verdict["means_if_it_survives"], key


def test_every_family_is_classified(completed):
    assert set(completed["family_classification"]) == {f.key for f in FAMILIES}


# --- the two branches D7 may land in -----------------------------------------


def test_dominance_needs_both_a_real_gain_and_most_of_the_joint(completed):
    flags = completed["outcome_flags"]
    r5 = flags["recall@5_by_family"]
    best = max(r5, key=lambda key: r5[key])
    if flags["dominant_family"] is not None:
        assert flags["dominant_family"] == best
        assert r5[best] >= PROMISING_R5
        assert r5[best] >= DOMINANT_SHARE * flags["joint_recall@5"]
    else:
        assert (
            r5[best] < PROMISING_R5
            or flags["joint_recall@5"] <= 0.0
            or r5[best] < DOMINANT_SHARE * flags["joint_recall@5"]
        )


def test_complementarity_is_flagged_only_when_no_family_moves(completed):
    flags = completed["outcome_flags"]
    r5 = flags["recall@5_by_family"]
    quiet = all(abs(value) < NEGLIGIBLE_BAND for value in r5.values())
    expected = quiet and flags["joint_recall@5"] >= PROMISING_R5
    assert flags["joint_complementarity_suspected"] is expected


def test_the_complementarity_branch_forbids_a_sweep(completed):
    guidance = completed["outcome_flags"]["if_complementarity"]
    assert "own declaration" in guidance
    assert "No combinatorial family sweep" in guidance


def test_the_dominant_branch_points_at_a_replacement_not_more_ablations(completed):
    guidance = completed["outcome_flags"]["if_dominant"]
    assert "replacement" in guidance
    assert "not more historical ablations" in guidance


# --- the frozen operating point ----------------------------------------------


def test_the_normaliser_and_definitions_are_untouched(completed):
    normalisation = completed["normalisation"]
    assert tuple(NORMALISED_COLUMNS) == (4, 5, 6, 7, 8, 9)
    assert normalisation["normalised_columns"] == [4, 5, 6, 7, 8, 9]
    assert normalisation["extended_to_the_new_columns"] is False
    assert normalisation["path_and_ppr_definitions_changed"] is False


def test_every_arm_has_the_same_architecture(completed):
    counts = {
        arm: completed["results"][arm]["parameters"] for arm in D7_ARMS + REUSED_ARMS
    }
    assert len(set(counts.values())) == 1, counts
    widths = {arm: completed["results"][arm]["local_dim"] for arm in D7_ARMS + REUSED_ARMS}
    assert set(widths.values()) == {LOCAL_DIM}
    assert completed["ablation"]["architecture_changed_between_arms"] is False
    assert completed["ablation"]["architecture_changed_vs_d6"] is False


def test_the_stage_reads_no_test_split(completed):
    assert completed["splits"]["test_read"] is False
    assert completed["contract"]["test_split_read"] is False
    assert completed["contract"]["epoch_selected_on_validation"] is False
    assert completed["contract"]["gnn_trained"] is False
    assert completed["contract"]["message_passing"] is False


def test_the_test_split_is_refused_outright(tmp_path, data, d6_result):
    with synthetic(data):
        args = d7_args(tmp_path, data, d6_result)
        args.splits = ["train", "validation", "test"]
        with pytest.raises(ValueError, match="test split is not read"):
            run(args)


def test_a_missing_d6_result_is_refused(tmp_path, data):
    with synthetic(data):
        args = d7_args(tmp_path, data, tmp_path / "absent.json")
        with pytest.raises(FileNotFoundError, match="reuses D6"):
            run(args)


def test_the_ladder_carries_six_rows(completed):
    assert set(completed["ladder"]) == set(D7_ARMS) | set(REUSED_ARMS)
    assert completed["status"] == COMPLETE_STATUS
    for row in completed["ladder"].values():
        assert set(METRICS) <= set(row)


def test_the_stratified_increments_are_arithmetic(completed):
    for family in FAMILIES:
        rows = completed["increments_by_stratum"][family.increment]
        for stratum, row in rows.items():
            upper = completed["results"][family.arm]["validation_by_stratum"][stratum]
            lower = completed["results"][BASE_ARM]["validation_by_stratum"][stratum]
            for metric in METRICS:
                if metric in row and metric in upper and metric in lower:
                    assert row[metric] == pytest.approx(
                        upper[metric] - lower[metric], abs=1e-12
                    ), (family.key, stratum, metric)


def test_the_stage_says_what_it_has_not_established(completed):
    text = completed["not_established"].lower()
    assert "one dataset" in text and "one seed" in text
    assert "not" in text
    assert "qls-v2" in text
