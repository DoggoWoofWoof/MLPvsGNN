"""The Phase -1 analyzer reports what the audit measured, and nothing else.

The failures worth guarding against here are not crashes. They are an analyzer
that quietly averages the two connectivity notions together, drops the
denominator that says how many queries a number came from, or grows a verdict
about whether a substrate is "good enough" -- each of which turns a measurement
into a claim the audit never made.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import analyze_graph_substrate as gs


def _split(**overrides) -> dict:
    split = {
        "queries": 100,
        "queries_without_retrieval_seeds": 2,
        "queries_without_gold_in_pool": 1,
        "node_level": {
            "retention_mean": 0.4,
            "retention_p10": 0.0,
            "retention_median": 0.33,
            "retention_p90": 0.8,
            "pooled_from_queries": 64,
            "global_degree_mean": 12.0,
        },
        "query_level": {
            "connectivity": {
                "candidates": 300.0,
                "candidates__queries_reporting": 100,
                "isolated_fraction": 0.25,
                "degree_1_fraction": 0.3,
                "degree_ge_2_fraction": 0.45,
            },
            "retention": {"retention_mean": 0.41, "retention_zero_fraction": 0.2},
            "receptive_field": {
                "R1_median": 2.0,
                "R2_median": 6.0,
                "R3_median": 11.0,
                "R1_mean": 2.5,
                "R1_median__queries_reporting": 100,
            },
            "message_flow_receptive_field": {
                "flow_R1_median": 1.0,
                "flow_R2_median": 3.0,
                "flow_R3_median": 4.0,
                "flow_R1_mean": 1.5,
            },
            "operator_edge_load": {
                "unique_non_self_edges": 500.0,
                "stored_non_self_messages": 900.0,
                "duplicate_message_fraction": 0.44,
                "messages_consumed_by_operator": 1200.0,
            },
            "seed_reachability_induced": {
                "reachable_at_1": 0.06,
                "reachable_at_2": 0.10,
                "reachable_at_3": 0.14,
            },
            "seed_reachability_induced_message_flow": {
                "reachable_at_1": 0.06,
                "reachable_at_2": 0.08,
                "reachable_at_3": 0.09,
            },
            "seed_reachability_global": {
                "reachable_at_1": 0.06,
                "reachable_at_2": 0.76,
                "reachable_at_3": 0.97,
            },
            "gold_path_preservation": {
                "connected_globally_fraction": 0.99,
                "connected_induced_fraction": 0.90,
                "globally_connected_but_induced_disconnected": 0.09,
            },
            "gold_bridge_loss": {"bridge_loss_at_3": 0.02, "eligible_at_3": 90.0},
        },
        "expansion": {
            "queries_measured": 512,
            "symmetric": {
                "candidates": 300.0,
                "U_seed_1_nodes": 330.0,
                "U_seed_1_expansion": 1.10,
                "U_seed_3_expansion": 167.0,
                "U_target_1_expansion": 5.5,
            },
            "message_flow": {"U_seed_1_expansion": 1.02},
        },
    }
    split.update(overrides)
    return split


def _audit(graphs: dict | None = None, status: str | None = None) -> dict:
    graphs = graphs or {
        "dataset_default": {
            "provenance": "the graph carried by the frozen corpus",
            "stored_directed_edges": 855146,
            "undirected_edges": 260807,
            "stored_graph_was_symmetric": False,
            "undirected_edge_key_sha256": "aa" * 32,
            "splits": {"validation": _split()},
        },
        "baseline_a_simple": {
            "provenance": "Package B edge family baseline_a_simple",
            "stored_directed_edges": 521614,
            "undirected_edges": 260807,
            "stored_graph_was_symmetric": True,
            "undirected_edge_key_sha256": "aa" * 32,
            "splits": {"validation": _split()},
        },
        "knn_only": {
            "provenance": "Package B edge family knn_only",
            "stored_directed_edges": 269474,
            "undirected_edges": 134737,
            "stored_graph_was_symmetric": True,
            "undirected_edge_key_sha256": "bb" * 32,
            "splits": {"validation": _split()},
        },
    }
    return {
        "status": status or gs.COMPLETE_STATUS,
        "dataset": "tiny",
        "queries": 100,
        "num_nodes": 5000,
        "num_stored_directed_edges": 855146,
        "diagnostic_contract": {
            "operator_kind": "gat",
            "max_hops": 3,
            "read_only": True,
            "pooled_query_cap": 4000,
            "expansion_query_cap": 512,
        },
        "graphs": graphs,
    }


def _write(root: Path, name: str, payload: dict, nested: bool = True) -> Path:
    base = root / gs.AUDIT_DIR
    path = (base / name / "abc123" / "substrate.json") if nested else (base / f"{name}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# The two connectivity notions
# ---------------------------------------------------------------------------


def test_the_two_connectivity_notions_are_reported_separately() -> None:
    """Protocol section 2: weak/symmetrised and exact directed message flow are
    two different questions and may never be collapsed into one number."""

    field = gs.receptive_field(_split())
    assert field["symmetrised"]["R3_median"] == 11.0
    assert field["message_flow"]["R3_median"] == 4.0
    assert field["message_flow_minus_symmetrised"]["R3_median"] == -7.0
    assert field["notions_coincide"] is False


def test_a_stored_symmetric_graph_shows_the_notions_coinciding() -> None:
    """Coincidence is a measured outcome, not an assumption -- so it has to be
    reachable, and reported as such rather than as an absent comparison."""

    split = _split()
    split["query_level"]["message_flow_receptive_field"] = {
        "flow_R1_median": 2.0,
        "flow_R2_median": 6.0,
        "flow_R3_median": 11.0,
        "flow_R1_mean": 2.5,
    }
    field = gs.receptive_field(split)
    assert field["notions_coincide"] is True
    assert set(field["message_flow_minus_symmetrised"].values()) == {0.0}


def test_seed_reachability_keeps_induced_and_global_apart() -> None:
    reach = gs.seed_reach(_split())
    assert reach["induced_symmetrised"]["reachable_at_3"] == 0.14
    assert reach["global"]["reachable_at_3"] == 0.97
    assert reach["global_minus_induced"]["reachable_at_3"] == pytest.approx(0.83)
    # The message-flow view is carried too, and is not folded into the other.
    assert reach["induced_message_flow"]["reachable_at_3"] == 0.09


def test_retention_does_not_mix_its_two_aggregation_levels() -> None:
    """One is pooled over candidates, the other is a mean across queries. They
    answer different questions and happen to share key names."""

    block = gs.retention(_split())
    assert block["node_level_pooled_over_candidates"]["retention_mean"] == 0.4
    assert block["query_level_mean_across_queries"]["retention_mean"] == 0.41
    assert block["node_level_pooled_over_candidates"]["pooled_from_queries"] == 64


# ---------------------------------------------------------------------------
# Denominators
# ---------------------------------------------------------------------------


def test_the_queries_reporting_denominator_is_carried_through() -> None:
    """A mean over 512 capped queries and a mean over 97,852 are not the same
    claim. The audit records which; dropping it here would upgrade one to the
    other."""

    block = gs.connectivity(_split())
    assert block["candidates__queries_reporting"] == 100
    assert gs.receptive_field(_split())["symmetrised"]["R1_median__queries_reporting"] == 100


def test_a_missing_denominator_is_absent_rather_than_invented() -> None:
    split = _split()
    del split["query_level"]["connectivity"]["candidates__queries_reporting"]
    block = gs.connectivity(split)
    assert "candidates" in block
    assert "candidates__queries_reporting" not in block


def test_a_metric_the_audit_did_not_record_is_omitted_not_zeroed() -> None:
    """A zero would read as a measured zero."""

    split = _split()
    split["query_level"]["connectivity"] = {"candidates": 300.0}
    block = gs.connectivity(split)
    assert block == {"candidates": 300.0}


# ---------------------------------------------------------------------------
# Provenance aliasing
# ---------------------------------------------------------------------------


def test_families_sharing_an_undirected_edge_set_are_reported_as_one_graph() -> None:
    """Two families with the same undirected edge key are the same substrate
    wearing two labels, and every downstream comparison between them is
    therefore vacuous. Reported rather than left in two identical rows."""

    aliases = gs.provenance_aliases(_audit())
    assert aliases["families_audited"] == 3
    assert aliases["distinct_undirected_graphs"] == 2
    assert len(aliases["aliased_groups"]) == 1
    group = aliases["aliased_groups"][0]
    assert group["families"] == ["baseline_a_simple", "dataset_default"]
    assert group["stored_directed_edges"]["dataset_default"] == 855146
    assert group["stored_graph_was_symmetric"]["dataset_default"] is False


def test_distinct_graphs_produce_no_alias_group() -> None:
    audit = _audit()
    audit["graphs"]["baseline_a_simple"]["undirected_edge_key_sha256"] = "cc" * 32
    aliases = gs.provenance_aliases(audit)
    assert aliases["aliased_groups"] == []
    assert aliases["distinct_undirected_graphs"] == 3


def test_a_family_without_an_edge_key_is_not_grouped_with_the_others() -> None:
    """An absent key is unknown, not equal to another absent key."""

    audit = _audit()
    del audit["graphs"]["baseline_a_simple"]["undirected_edge_key_sha256"]
    del audit["graphs"]["knn_only"]["undirected_edge_key_sha256"]
    aliases = gs.provenance_aliases(audit)
    assert aliases["families_audited"] == 1
    assert aliases["aliased_groups"] == []


# ---------------------------------------------------------------------------
# Discovery and completion
# ---------------------------------------------------------------------------


def test_both_output_layouts_are_discovered(tmp_path: Path) -> None:
    _write(tmp_path, "musique_clean", _audit())
    _write(tmp_path, "webqsp", _audit(), nested=False)
    found = gs.discover(tmp_path)
    assert sorted(found) == ["musique_clean", "webqsp"]


def test_the_analyzers_own_summary_is_not_read_back_as_an_audit(tmp_path: Path) -> None:
    _write(tmp_path, "musique_clean", _audit())
    (tmp_path / gs.AUDIT_DIR / "summary.json").write_text("{}", encoding="utf-8")
    assert sorted(gs.discover(tmp_path)) == ["musique_clean"]


def test_a_root_already_pointing_at_the_audit_directory_works(tmp_path: Path) -> None:
    _write(tmp_path, "musique_clean", _audit())
    assert sorted(gs.discover(tmp_path / gs.AUDIT_DIR)) == ["musique_clean"]


def test_an_in_progress_audit_is_reported_as_in_progress(tmp_path: Path) -> None:
    """Partial numbers are worth reading while a job runs, but a partial audit
    must never be presented as the finished measurement."""

    _write(tmp_path, "tiny", _audit(status="GRAPH_SUBSTRATE_AUDIT_IN_PROGRESS"))
    summary = gs.summarize_audit(gs.load_audit(gs.discover(tmp_path)["tiny"]))
    assert summary["complete"] is False


def test_require_complete_exits_non_zero_while_a_job_is_still_running(
    tmp_path: Path, capsys
) -> None:
    _write(tmp_path, "tiny", _audit(status="GRAPH_SUBSTRATE_AUDIT_IN_PROGRESS"))
    code = gs.main(
        [
            "--results-root",
            str(tmp_path),
            "--require-complete",
            "--output",
            str(tmp_path / "summary.json"),
        ]
    )
    assert code == 1


def test_a_complete_audit_passes_the_same_check(tmp_path: Path, capsys) -> None:
    _write(tmp_path, "tiny", _audit())
    code = gs.main(
        [
            "--results-root",
            str(tmp_path),
            "--require-complete",
            "--output",
            str(tmp_path / "summary.json"),
        ]
    )
    assert code == 0
    written = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert written["complete"] == ["tiny"]
    assert written["in_progress"] == []


def test_an_audit_filed_under_the_wrong_dataset_is_refused(tmp_path: Path) -> None:
    """A fetch that wrote squad's numbers into metaqa's directory would
    otherwise be reported as metaqa's substrate."""

    _write(tmp_path, "metaqa", _audit())  # the payload records dataset "tiny"
    with pytest.raises(SystemExit) as raised:
        gs.main(["--results-root", str(tmp_path), "--output", str(tmp_path / "s.json")])
    assert "wrong dataset" in str(raised.value)
    assert "'tiny'" in str(raised.value) and "'metaqa'" in str(raised.value)


def test_an_empty_root_is_refused_rather_than_reported_as_nothing_to_see(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        gs.main(["--results-root", str(tmp_path)])


# ---------------------------------------------------------------------------
# What the analyzer must not do
# ---------------------------------------------------------------------------


def test_the_emitted_summary_carries_no_adequacy_judgement(tmp_path: Path) -> None:
    """Protocol section 8 of the refinement: characterise continuously. A
    threshold here would smuggle in the binary judgement it rules out.

    Checked on the emitted structure rather than on the source text, so it
    tests what a reader receives instead of which words were used to say it.
    ``complete`` is about whether the job finished, not whether the substrate
    is any good, and ``notions_coincide`` reports a measured equality.
    """

    _write(tmp_path, "tiny", _audit())
    gs.main(["--results-root", str(tmp_path), "--output", str(tmp_path / "s.json")])
    written = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))

    # Each of these is a fact about the run or the graph, not a judgement about
    # the substrate: whether the job finished, whether the two notions came out
    # equal, whether the stored graph was symmetric, whether the audit wrote
    # anything, and the standing statement that section 8 admits nothing.
    allowed = {
        "complete",
        "notions_coincide",
        "admits_nothing_to_the_pool",
        "stored_graph_was_symmetric",
        "read_only",
    }
    verdict_words = ("adequate", "sufficient", "shallow", "pass", "fail", "ok", "good")

    def walk(node, path="", under_allowed=False):
        if isinstance(node, dict):
            for key, child in node.items():
                permitted = under_allowed or key in allowed
                if isinstance(child, bool) and not permitted:
                    assert False, f"boolean verdict at {path}/{key}"
                assert not any(
                    word in key.lower() for word in verdict_words
                ), f"{path}/{key} names a judgement"
                walk(child, f"{path}/{key}", permitted)
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]", under_allowed)

    walk(written["audits"])


def test_expansion_headroom_states_that_it_admits_nothing() -> None:
    """Section 8 is oracle-only. Nothing it measures may reach a pool, and the
    output says so rather than relying on the reader knowing."""

    block = gs.expansion(_split())
    assert block["admits_nothing_to_the_pool"] is True
    assert block["queries_measured"] == 512
    assert block["symmetric"]["U_seed_3_expansion"] == 167.0
    assert block["message_flow"]["U_seed_1_expansion"] == 1.02


def test_a_split_without_an_expansion_block_yields_nothing_rather_than_zeros() -> None:
    split = _split()
    del split["expansion"]
    assert gs.expansion(split) == {}


def test_an_unexpected_graph_family_is_named_not_dropped() -> None:
    """Silently ignoring a family the audit measured would hide it."""

    audit = _audit()
    audit["graphs"]["something_new"] = {
        "provenance": "x",
        "undirected_edge_key_sha256": "dd" * 32,
        "splits": {},
    }
    summary = gs.summarize_audit(audit)
    assert summary["graphs_not_in_the_reported_order"] == ["something_new"]


def test_a_file_that_is_not_an_audit_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "notanaudit.json"
    path.write_text(json.dumps({"dataset": "tiny"}), encoding="utf-8")
    with pytest.raises(ValueError):
        gs.load_audit(path)


def test_an_alias_group_carries_the_one_axis_the_pair_differs_on() -> None:
    """`baseline_a_simple` is defined as the deduplicated bidirectional
    projection of the multigraph, so every structural quantity is identical
    between them and message multiplicity is the whole difference. A group
    without the load is unreadable: it says two families are the same graph
    and gives no way to see what separates them."""

    audit = _audit()
    # Give the pair the multiplicity the real audit measures.
    audit["graphs"]["dataset_default"]["splits"]["validation"]["query_level"][
        "operator_edge_load"
    ] = {
        "unique_non_self_edges": 464.1,
        "messages_consumed_by_operator": 1065.0,
        "duplicate_message_fraction": 0.345,
    }
    audit["graphs"]["baseline_a_simple"]["splits"]["validation"]["query_level"][
        "operator_edge_load"
    ] = {
        "unique_non_self_edges": 464.1,
        "messages_consumed_by_operator": 823.8,
        "duplicate_message_fraction": 0.0,
    }

    group = gs.provenance_aliases(audit)["aliased_groups"][0]
    load = group["operator_message_load"]
    assert (
        load["dataset_default"]["validation"]["unique_non_self_edges"]
        == load["baseline_a_simple"]["validation"]["unique_non_self_edges"]
    )
    assert load["dataset_default"]["validation"]["duplicate_message_fraction"] == 0.345
    assert load["baseline_a_simple"]["validation"]["duplicate_message_fraction"] == 0.0


def test_the_zero_fraction_identity_is_checked_not_assumed() -> None:
    """A candidate with no induced edge has an empty receptive field at every
    hop, so the zero-fraction is constant in H and equals the independently
    measured isolated fraction. Three identical numbers read as a copy-paste
    error; the identity is checked so it can be reported as a consequence."""

    split = _split()
    split["query_level"]["receptive_field"].update(
        R1_zero_fraction=0.25, R2_zero_fraction=0.25, R3_zero_fraction=0.25
    )
    field = gs.receptive_field(split)
    assert field["zero_fraction_constant_in_hops"] is True
    assert field["zero_fraction_equals_isolated_fraction"] is True
    assert field["isolated_fraction"] == 0.25


def test_a_zero_fraction_that_moves_with_depth_is_reported_as_such() -> None:
    """If it ever does move, that is a finding about the audit, not something
    to normalise away."""

    split = _split()
    split["query_level"]["receptive_field"].update(
        R1_zero_fraction=0.25, R2_zero_fraction=0.10, R3_zero_fraction=0.05
    )
    field = gs.receptive_field(split)
    assert field["zero_fraction_constant_in_hops"] is False


def test_a_zero_fraction_disagreeing_with_the_isolated_fraction_is_flagged() -> None:
    split = _split()
    split["query_level"]["receptive_field"].update(
        R1_zero_fraction=0.40, R2_zero_fraction=0.40, R3_zero_fraction=0.40
    )
    field = gs.receptive_field(split)
    assert field["zero_fraction_constant_in_hops"] is True
    assert field["zero_fraction_equals_isolated_fraction"] is False
