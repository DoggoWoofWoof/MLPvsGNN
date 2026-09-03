"""The advancement rule is applied by code that cannot choose its own numbers.

Every threshold comes from configs/m0a_probe.yaml. These tests drive synthetic
probe payloads across each side of each filed boundary, including the two cases
that must not silently resolve: a run whose gates failed, and a run where both
outcomes appear to hold at once.
"""

from __future__ import annotations

import copy

import pytest
import yaml

from scripts.analyze_m0a_probe import CONFIG_PATH, _thresholds, abort_rule, decide

BASE_HEADROOM = {"any_gold_at_pool": 0.60, "recall_ceiling@5": 0.50}
COUNTS = {"median": 300.0, "p95": 360.0, "max": 400, "mean": 320.0}
LATENCY = {"p50": 1.0, "p95": 2.0, "p99": 3.0, "mean": 1.5, "max": 4.0}
FAMILY = "structural_only"
METHOD = "L1_DIRECTIONAL"


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def _cell(delta_any: float, delta_r5: float) -> dict:
    return {
        "headroom": {
            "any_gold_at_pool": BASE_HEADROOM["any_gold_at_pool"] + delta_any / 100.0,
            "recall_ceiling@5": BASE_HEADROOM["recall_ceiling@5"] + delta_r5 / 100.0,
        },
        "candidate_count": dict(COUNTS),
        "expansion_latency_ms": dict(LATENCY),
        "context_build_latency_ms": dict(LATENCY),
        "admitted_per_query": dict(COUNTS),
        "neighbour_scan_cap_fired_queries": 0,
        "is_the_headline": True,
    }


def _probe(*, matched=(0.0, 0.0), additive=(0.0, 0.0)) -> dict:
    matched_cell = _cell(*matched)
    additive_cell = _cell(*additive)
    additive_cell["is_the_headline"] = False
    additive_cell["candidate_count"] = {**COUNTS, "mean": COUNTS["mean"] + 40.0}
    return {
        "trained_anything": False,
        "test_split_read": False,
        "invariants": {
            "r1_ceiling_equals_r2_ceiling_exactly": True,
            "matched_budget_pool_size_equals_cq": True,
        },
        "regimes": {
            "R1": {"headroom": dict(BASE_HEADROOM), "candidate_count": dict(COUNTS)},
            "R2": {"headroom": dict(BASE_HEADROOM), "candidate_count": dict(COUNTS)},
            f"R3/{FAMILY}/{METHOD}/matched": matched_cell,
            f"R3/{FAMILY}/{METHOD}/additive": additive_cell,
        },
    }


def _three(**kwargs) -> dict:
    return {name: _probe(**kwargs) for name in ("squad_clean", "2wiki_clean", "metaqa")}


def test_the_thresholds_are_read_from_the_declaration(config):
    filed = _thresholds(config)
    assert filed == {
        "movement_points": 1.0,
        "decisive_null_matched_points": 0.25,
        "decisive_null_additive_points": 1.0,
    }


def test_the_numbers_and_the_sentences_agree(config):
    """Either half can be read; neither can be changed without the other."""
    rule = config["advancement_to_m0b"]["and_one_of"]
    assert f"at least {rule['movement_points']} point" in " ".join(rule["movement"].split())
    null = " ".join(rule["decisive_null"].split())
    assert f"less than {rule['decisive_null_matched_points']} points" in null
    assert f"less than {rule['decisive_null_additive_points']} point" in null


def test_a_movement_at_the_threshold_counts(config):
    probes = _three()
    probes["metaqa"] = _probe(matched=(0.0, 1.0))
    verdict = decide(probes, config)
    assert verdict["verdict"] == "MOVEMENT"
    assert verdict["outcomes"]["movement_evidence"][0]["dataset"] == "metaqa"
    assert verdict["outcomes"]["movement_evidence"][0]["metric"] == "recall_ceiling@5"


def test_a_movement_just_under_the_threshold_does_not(config):
    probes = _three()
    probes["metaqa"] = _probe(matched=(0.0, 0.99))
    assert decide(probes, config)["verdict"] == "NEITHER"


def test_a_fall_counts_as_movement_as_much_as_a_rise(config):
    """Expansion that costs a point of ceiling is a finding, not a null."""
    probes = _three()
    probes["squad_clean"] = _probe(matched=(0.0, -1.5))
    verdict = decide(probes, config)
    assert verdict["verdict"] == "MOVEMENT"
    assert verdict["outcomes"]["movement_evidence"][0]["points"] == pytest.approx(-1.5)


def test_the_filed_branches_overlap_on_anygold_and_the_overlap_is_reported(config):
    """A defect in the rule as filed, found before any result was read.

    ``movement`` fires on AnyGold *or* recall@5. ``decisive_null`` reads only
    recall@5. So AnyGold moving a point while recall@5 stays inert satisfies
    both, and the branches are not disjoint. The analysis reports the overlap
    instead of picking the branch that suits the outcome; resolving it is a
    decision on the rule, not a computation over the data.
    """

    probes = _three()
    probes["squad_clean"] = _probe(matched=(-1.5, 0.0))
    verdict = decide(probes, config)
    assert verdict["verdict"] == "CONTRADICTORY"
    assert verdict["outcomes"]["movement"] is True
    assert verdict["outcomes"]["decisive_null"] is True


def test_everything_inert_reads_as_the_decisive_null(config):
    probes = _three(matched=(0.0, 0.1), additive=(0.0, 0.5))
    verdict = decide(probes, config)
    assert verdict["verdict"] == "DECISIVE_NULL"
    assert verdict["outcomes"]["decisive_null"] is True


def test_an_additive_pool_that_moves_blocks_the_null(config):
    """A matched null with an additive move is the in-between case, not a null."""
    probes = _three(matched=(0.0, 0.1), additive=(0.0, 1.5))
    assert decide(probes, config)["verdict"] == "NEITHER"


def test_the_in_between_case_refuses_to_advance(config):
    probes = _three(matched=(0.0, 0.5))
    verdict = decide(probes, config)
    assert verdict["verdict"] == "NEITHER"
    assert "is NOT launched" in verdict["why"]


def test_a_failed_gate_stops_the_reading_entirely(config):
    probes = _three(matched=(0.0, 5.0))
    probes["squad_clean"]["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"] = False
    verdict = decide(probes, config)
    assert verdict["verdict"] == "GATES_FAILED"
    assert verdict["gates_all_passed"] is False


def test_a_matched_pool_that_grew_fails_its_gate(config):
    probes = _three()
    grown = probes["metaqa"]["regimes"][f"R3/{FAMILY}/{METHOD}/matched"]
    grown["candidate_count"] = {**COUNTS, "mean": COUNTS["mean"] + 1.0}
    verdict = decide(probes, config)
    assert verdict["verdict"] == "GATES_FAILED"
    assert verdict["gates"]["metaqa"]["matched_budget_held_for_every_query"] is False


def test_a_missing_latency_percentile_fails_its_gate(config):
    probes = _three()
    cell = probes["squad_clean"]["regimes"][f"R3/{FAMILY}/{METHOD}/matched"]
    cell["expansion_latency_ms"] = {"p50": 1.0}
    assert decide(probes, config)["gates"]["squad_clean"]["latency_percentiles_recorded"] is False


def test_both_outcomes_at_once_is_reported_rather_than_resolved(config):
    """Disjoint by construction; if they ever overlap that needs diagnosis."""
    loosened = copy.deepcopy(config)
    loosened["advancement_to_m0b"]["and_one_of"]["decisive_null_matched_points"] = 5.0
    probes = _three(matched=(2.0, 0.0))
    verdict = decide(probes, loosened)
    assert verdict["verdict"] == "CONTRADICTORY"
    assert "needs diagnosis" in verdict["why"]


def test_the_analysis_never_claims_to_authorise_a_launch(config):
    verdict = decide(_three(matched=(0.0, 9.0)), config)
    assert verdict["advancement_is_automatic"] is False
    assert verdict["thresholds_were_filed_before_results"] is True


# --- context that is reported beside the rule, and cannot change it ---


def _with_sets(probe: dict, left: list[list[int]], right: list[list[int]],
               *, capped_queries: int) -> dict:
    matched = probe["regimes"][f"R3/{FAMILY}/{METHOD}/matched"]
    matched["admitted_nodes_per_query"] = left
    matched["seeds_at_the_per_seed_cap"] = capped_queries
    matched["queries_with_a_capped_seed"] = capped_queries
    control = copy.deepcopy(matched)
    control["admitted_nodes_per_query"] = right
    probe["regimes"][f"R3/{FAMILY}/STRUCTURAL_NEIGHBOUR/matched"] = control
    return probe


def test_a_null_where_the_cap_never_bound_is_marked_as_no_comparison(config):
    """Both arms took the whole neighbourhood, so neither ever chose anything."""
    probe = _with_sets(_probe(), [[1, 2]], [[1, 2]], capped_queries=0)
    agreement = decide({"squad_clean": probe}, config)["not_part_of_the_rule"][
        "arm_agreement"
    ]["squad_clean"][FAMILY]
    assert agreement["the_comparison_happened"] is False
    assert agreement["mean_jaccard"] == pytest.approx(1.0)
    assert agreement["headroom_is_identical"] is True


def test_a_null_where_the_arms_differed_is_marked_as_a_real_comparison(config):
    probe = _with_sets(_probe(), [[1, 2]], [[1, 3]], capped_queries=1)
    agreement = decide({"squad_clean": probe}, config)["not_part_of_the_rule"][
        "arm_agreement"
    ]["squad_clean"][FAMILY]
    assert agreement["the_comparison_happened"] is True
    assert agreement["queries_where_the_arms_admitted_identical_sets"] == 0
    assert agreement["mean_jaccard"] == pytest.approx(1 / 3)


def test_the_context_blocks_cannot_reach_the_verdict(config):
    plain = decide(_three(), config)
    noisy = decide({"squad_clean": _with_sets(_probe(), [[9]], [[8]], capped_queries=5),
                    "2wiki_clean": _probe(), "metaqa": _probe()}, config)
    assert plain["verdict"] == noisy["verdict"] == "DECISIVE_NULL"
    assert plain["outcomes"] == noisy["outcomes"]


# --- the filed abort rule ---


def _breach(probes: dict, name: str, *, ratio: float) -> dict:
    """Make one matched cell's expansion p95 exceed its context p95 by `ratio`."""
    cell = probes[name]["regimes"][f"R3/{FAMILY}/{METHOD}/matched"]
    cell["expansion_latency_ms"] = {
        **LATENCY,
        "p95": LATENCY["p95"] * ratio,
    }
    return probes


def test_the_abort_factor_is_read_from_the_declaration_not_hard_coded(config):
    lenient = copy.deepcopy(config)
    lenient["compute"]["abort_rule_latency_factor"] = 100.0
    probes = _breach(_three(), "metaqa", ratio=10.0)
    assert abort_rule(probes["metaqa"], config)["breached"] is True
    assert abort_rule(probes["metaqa"], lenient)["breached"] is False


def test_a_cell_inside_the_factor_does_not_breach(config):
    """Equal p95s are a ratio of one; the filed factor is four."""
    inside = abort_rule(_three()["squad_clean"], config)
    assert inside["breached"] is False
    assert inside["breaches"] == []
    assert inside["factor"] == 4.0


def test_the_boundary_is_strictly_greater_than_the_factor(config):
    exactly = abort_rule(_breach(_three(), "metaqa", ratio=4.0)["metaqa"], config)
    just_over = abort_rule(_breach(_three(), "metaqa", ratio=4.01)["metaqa"], config)
    assert exactly["breached"] is False
    assert just_over["breached"] is True


def test_a_breach_names_the_cell_the_arm_and_the_ratio(config):
    breached = abort_rule(_breach(_three(), "metaqa", ratio=6.0)["metaqa"], config)
    assert breached["arms_that_breached"] == [METHOD]
    row = breached["breaches"][0]
    assert row["cell"] == f"R3/{FAMILY}/{METHOD}/matched"
    assert row["ratio"] == pytest.approx(6.0)
    assert row["expansion_p95_ms"] == pytest.approx(LATENCY["p95"] * 6.0)
    assert row["context_p95_ms"] == pytest.approx(LATENCY["p95"])


def test_a_breach_marks_the_verdict_without_moving_the_outcome(config):
    """The reading is unchanged; whether it may be acted on is not."""
    clean = decide(_three(matched=(0.0, 9.0)), config)
    dirty = decide(_breach(_three(matched=(0.0, 9.0)), "metaqa", ratio=6.0), config)
    assert clean["verdict"] == "MOVEMENT"
    assert dirty["verdict"] == "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"
    assert dirty["outcomes"] == clean["outcomes"]
    assert dirty["movement_points_against_r1"] == clean["movement_points_against_r1"]
    assert dirty["abort_rule_breached"] is True
    assert "metaqa" in dirty["why"]
    assert "did not implement the abort" in dirty["why"]


def test_a_breach_does_not_manufacture_an_outcome(config):
    """A null under a breach stays a null; the suffix is not a second verdict."""
    dirty = decide(_breach(_three(matched=(0.0, 0.0)), "squad_clean", ratio=9.0), config)
    assert dirty["verdict"] == "DECISIVE_NULL_UNDER_A_BREACHED_ABORT_RULE"
    assert dirty["outcomes"]["movement"] is False


def test_a_failed_gate_still_outranks_a_breach(config):
    probes = _breach(_three(), "metaqa", ratio=9.0)
    probes["squad_clean"]["trained_anything"] = True
    assert decide(probes, config)["verdict"] == "GATES_FAILED_UNDER_A_BREACHED_ABORT_RULE"


def test_a_cell_missing_a_percentile_is_not_read_as_compliance(config):
    """An unevaluable rule must not be indistinguishable from a passing one."""
    probes = _three()
    cell = probes["metaqa"]["regimes"][f"R3/{FAMILY}/{METHOD}/matched"]
    del cell["context_build_latency_ms"]["p95"]
    rule = abort_rule(probes["metaqa"], config)
    assert rule["breached"] is False
    assert rule["cells_where_the_rule_could_not_be_evaluated"] == [
        f"R3/{FAMILY}/{METHOD}/matched"
    ]
    verdict = decide(probes, config)
    assert verdict["gates"]["metaqa"]["latency_percentiles_recorded"] is False
    assert verdict["verdict"] == "GATES_FAILED"


def test_a_zero_context_p95_is_a_breach_not_a_division_error(config):
    probes = _three()
    cell = probes["metaqa"]["regimes"][f"R3/{FAMILY}/{METHOD}/matched"]
    cell["context_build_latency_ms"] = {**LATENCY, "p95": 0.0}
    assert abort_rule(probes["metaqa"], config)["breached"] is True
