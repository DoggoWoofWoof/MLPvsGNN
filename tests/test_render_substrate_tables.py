"""The rendered tables must not say something the summary does not.

The renderer is presentation, so the risk it carries is mislabelling rather
than miscomputation: a per-query count printed under a header that reads like a
rate, two retention aggregation levels collapsed into one column, or a partial
audit rendered as though it were whole. Those are the things tested here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import render_substrate_tables as rst


def _split(**overrides):
    payload = {
        "connectivity": {
            "candidates": 300.0,
            "edges_directed_non_self": 400.0,
            "isolated_fraction": 0.25,
            "degree_1_fraction": 0.3,
            "degree_ge_2_fraction": 0.45,
        },
        "retention": {
            "node_level_pooled_over_candidates": {
                "retention_mean": 0.2,
                "retention_median": 0.1,
                "global_degree_median": 7.0,
            },
            "query_level_mean_across_queries": {
                "retention_zero_fraction": 0.25,
                "retention_below_10pct_fraction": 0.4,
                "retention_below_25pct_fraction": 0.6,
                "boundary_cut_ratio": 0.8,
            },
        },
        "receptive_field": {
            "symmetrised": {
                "R1_median": 1.0,
                "R2_median": 2.0,
                "R3_median": 3.0,
                "R1_zero_fraction": 0.25,
            },
            "message_flow": {"R1_median": 1.0, "R2_median": 2.0, "R3_median": 3.0},
            "notions_coincide": True,
        },
        "operator_message_load": {
            "unique_non_self_edges": 400.0,
            "stored_non_self_messages": 600.0,
            "duplicate_message_fraction": 0.33,
            "stored_self_loops": 0.0,
            "operator_inserted_self_loops": 300.0,
            "messages_consumed_by_operator": 900.0,
        },
        "seed_reachability": {
            "induced_symmetrised": {
                "reachable_at_1": 0.06,
                "reachable_at_2": 0.10,
                "reachable_at_3": 0.14,
            },
            "global": {
                "reachable_at_1": 0.06,
                "reachable_at_2": 0.76,
                "reachable_at_3": 0.97,
            },
        },
        "path_preservation": {
            "gold_path_preservation": {
                "targets": 2.1,
                "connected_globally_fraction": 1.0,
                "connected_induced_fraction": 0.9,
                "globally_connected_but_induced_disconnected": 0.252,
                "distance_inflated_fraction": 0.02,
            },
            "gold_bridge_loss": {
                "bridge_loss_at_1": 0.0,
                "bridge_loss_at_2": 0.078,
                "bridge_loss_at_3": 0.099,
            },
        },
        "expansion_headroom": {
            "queries_measured": 512,
            "admits_nothing_to_the_pool": True,
            "symmetric": {
                "candidates": 359.2,
                "U_seed_1_expansion": 1.1,
                "U_seed_2_expansion": 110.6,
                "U_seed_3_expansion": 167.2,
                "U_target_1_expansion": 5.5,
                "U_target_2_expansion": 129.7,
                "U_target_3_expansion": 175.5,
            },
        },
    }
    payload.update(overrides)
    return payload


def _audit(dataset="tiny", complete=True, graphs=None, aliases=None):
    graphs = graphs or {
        name: {
            "stored_directed_edges": 855146 if name == "dataset_default" else 521614,
            "undirected_edges": 260807,
            "stored_graph_was_symmetric": name != "dataset_default",
            "splits": {"validation": _split()},
        }
        for name in rst.GRAPH_ORDER
    }
    return {
        "dataset": dataset,
        "complete": complete,
        "operator_kind": "gat",
        "message_flow": "source_to_target",
        "operator_edge_semantics": {
            "adds_self_loops": True,
            "coalesces_duplicates": False,
            "duplicate_sensitive": True,
            "aggregation": "attention_weighted_sum",
            "root_term": "inserted_self_loop",
            "isolated_node_still_scored": True,
        },
        "graphs": graphs,
        "provenance_aliases": aliases
        or {
            "distinct_undirected_graphs": 3,
            "families_audited": 4,
            "aliased_groups": [
                {
                    "undirected_edge_key_sha256": "a" * 64,
                    "families": ["baseline_a_simple", "dataset_default"],
                    "stored_directed_edges": {
                        "baseline_a_simple": 100,
                        "dataset_default": 180,
                    },
                    "stored_graph_was_symmetric": {
                        "baseline_a_simple": True,
                        "dataset_default": False,
                    },
                    "operator_message_load": {
                        "baseline_a_simple": {
                            "validation": {"messages_consumed_by_operator": 800.0}
                        },
                        "dataset_default": {
                            "validation": {"messages_consumed_by_operator": 1065.4}
                        },
                    },
                }
            ],
        },
    }


def _render(*audits, split="validation"):
    return rst.render({"audits": list(audits)}, split)


def _table(text, title_fragment):
    """The block of pipe rows under the heading whose title contains the fragment."""

    blocks = text.split("### ")
    for block in blocks:
        if title_fragment in block.split("\n", 1)[0]:
            return [line for line in block.splitlines() if line.startswith("|")]
    raise AssertionError(f"no table titled with {title_fragment!r}")


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_a_per_query_count_is_labelled_as_a_count():
    rows = _table(_render(_audit()), "Gold path preservation")
    header = rows[0]
    assert "lost by induction (count/query)" in header
    assert "targets (count/query)" in header


def test_the_count_column_carries_the_count_not_a_derived_rate():
    rows = _table(_render(_audit()), "Gold path preservation")
    cells = [c.strip() for c in rows[2].split("|")]
    # 0.252 is the recorded mean count of lost gold targets per query. A
    # renderer that "helpfully" divided by `targets` would print 0.120.
    assert "0.252" in cells
    assert "0.120" not in cells


def test_the_conditional_rate_is_reported_separately_from_the_count():
    rows = _table(_render(_audit()), "Gold path preservation")
    assert "bridge loss @1" in rows[0]
    assert "0.099" in rows[2]


def test_the_two_retention_levels_get_their_own_named_columns():
    header = _table(_render(_audit()), "Global-neighbourhood retention")[0]
    assert "mean (node-pooled)" in header
    assert "median (node-pooled)" in header
    assert "ret = 0 (query-mean)" in header


def test_both_connectivity_notions_appear_and_are_not_merged():
    header = _table(_render(_audit()), "Effective receptive field")[0]
    for column in ("sym R1", "sym R2", "sym R3", "flow R1", "flow R2", "flow R3"):
        assert column in header


def test_the_two_self_loop_sources_are_reported_separately():
    # Protocol 4.4 names both; 4.2 flags that `gcn` and `gat` insert their own,
    # so a stored self-loop would be consumed twice. Collapsing the two into one
    # column, or omitting the stored count, hides whether that case arises.
    rows = _table(_render(_audit()), "Operator message load")
    header = rows[0]
    assert "stored self-loops" in header
    assert "operator self-loops" in header
    cells = [c.strip() for c in rows[2].split("|")]
    assert cells.count("300.0") == 1  # operator-inserted, not doubled into both
    assert "0.0" in cells  # the stored count, recorded and rendered as zero


def test_a_stored_self_loop_count_is_rendered_not_swallowed():
    graphs = {
        "dataset_default": {
            "splits": {
                "validation": _split(
                    operator_message_load={
                        "unique_non_self_edges": 400.0,
                        "stored_non_self_messages": 600.0,
                        "duplicate_message_fraction": 0.33,
                        "stored_self_loops": 12.5,
                        "operator_inserted_self_loops": 300.0,
                        "messages_consumed_by_operator": 900.0,
                    }
                )
            }
        }
    }
    body = _table(_render(_audit(graphs=graphs)), "Operator message load")[2]
    assert "12.5" in body


def test_the_expansion_table_says_it_admits_nothing():
    text = _render(_audit())
    assert "ORACLE ONLY, admits nothing to any pool" in text


# ---------------------------------------------------------------------------
# What gets rendered at all
# ---------------------------------------------------------------------------


def test_an_incomplete_audit_is_not_rendered():
    text = _render(_audit(dataset="done"), _audit(dataset="running", complete=False))
    assert "done" in text
    assert "running" not in text


def test_a_summary_with_no_complete_audit_is_refused():
    with pytest.raises(SystemExit) as excinfo:
        _render(_audit(complete=False))
    assert "No complete audit" in str(excinfo.value)


def test_a_graph_family_the_audit_lacks_is_skipped_not_zeroed():
    partial = _audit(
        graphs={"dataset_default": {"splits": {"validation": _split()}}}
    )
    rows = _table(_render(partial), "Candidate-induced connectivity")
    assert len(rows) == 3  # header, separator, one family
    assert "structural" not in "\n".join(rows)


def test_a_split_the_audit_lacks_is_refused_not_rendered_empty():
    # Bare headers would read as "measured, and empty" rather than "never
    # audited". The refusal names what was actually audited.
    with pytest.raises(SystemExit) as excinfo:
        _render(_audit(), split="test")
    message = str(excinfo.value)
    assert "'test'" in message
    assert "validation" in message


def test_a_metric_the_audit_did_not_record_prints_as_a_dash():
    graphs = {
        "dataset_default": {
            "splits": {
                "validation": _split(
                    connectivity={
                        "candidates": 300.0,
                        "isolated_fraction": 0.25,
                        "degree_1_fraction": 0.3,
                        "degree_ge_2_fraction": 0.45,
                    }
                )
            }
        }
    }
    rows = _table(_render(_audit(graphs=graphs)), "Candidate-induced connectivity")
    assert "| -- |" in rows[2]
    assert "0.000" not in rows[2]


# ---------------------------------------------------------------------------
# Operator semantics
# ---------------------------------------------------------------------------


def test_the_two_semantics_that_license_readings_elsewhere_are_rendered():
    # `coalesces duplicates: no` is why duplicate edges are counted as real
    # messages; `isolated still scored: yes` is why an isolated candidate is
    # called scored rather than dropped. Both claims appear in the prose.
    rows = _table(_render(_audit()), "Operator edge semantics")
    header, body = rows[0], rows[2]
    assert "coalesces duplicates" in header
    assert "isolated still scored" in header
    cells = [c.strip() for c in body.split("|")]
    assert cells[header.split("|").index(" coalesces duplicates ")] == "no"
    assert cells[header.split("|").index(" isolated still scored ")] == "yes"


def test_operator_semantics_is_one_row_per_dataset_not_per_graph():
    rows = _table(_render(_audit(dataset="a"), _audit(dataset="b")), "Operator edge semantics")
    assert [r.split("|")[1].strip() for r in rows[2:]] == ["a", "b"]


def test_an_audit_without_traced_semantics_renders_dashes():
    audit = _audit()
    del audit["operator_edge_semantics"]
    del audit["operator_kind"]
    body = _table(_render(audit), "Operator edge semantics")[2]
    cells = [c.strip() for c in body.split("|")]
    # Every semantic column unknown -- and not silently rendered as "no".
    assert cells.count("--") == 7
    assert "no" not in cells


# ---------------------------------------------------------------------------
# Storage orientation
# ---------------------------------------------------------------------------


def test_the_orientation_table_reports_stored_multiplicity_as_a_ratio():
    # 2.0 exactly means symmetric with no duplicate edges. The sealed graph is
    # above it, and that gap is the whole duplicate-message finding.
    rows = _table(_render(_audit()), "Storage orientation")
    sealed = rows[2]
    assert "3.2788" in sealed
    assert "no" in [c.strip() for c in sealed.split("|")]
    dedup = rows[5]
    assert "2.0000" in dedup


def test_orientation_is_rendered_even_though_it_has_no_split():
    # It is a property of the stored artifact, so asking for a split the audit
    # does have must not make it vanish, and it must not gain a split column.
    header = _table(_render(_audit()), "Storage orientation")[0]
    assert "split" not in header.lower()


def test_a_graph_missing_its_edge_counts_prints_dashes_not_a_ratio():
    graphs = {"dataset_default": {"splits": {"validation": _split()}}}
    body = _table(_render(_audit(graphs=graphs)), "Storage orientation")[2]
    cells = [c.strip() for c in body.split("|")]
    assert cells.count("--") == 4
    assert "0.0000" not in cells


# ---------------------------------------------------------------------------
# Aliasing
# ---------------------------------------------------------------------------


def test_an_alias_group_names_both_families_and_their_shared_key():
    rows = _table(_render(_audit()), "Provenance aliasing")
    body = rows[2]
    assert "`baseline_a_simple` + `dataset_default`" in body
    assert "`" + "a" * 16 + "`" in body


def test_the_alias_row_carries_the_one_axis_the_pair_differs_on():
    body = _table(_render(_audit()), "Provenance aliasing")[2]
    # Same undirected graph, different stored multiplicity, different work.
    assert "100 / 180" in body
    assert "yes / no" in body
    assert "800.0 / 1065.4" in body


def test_an_audit_with_no_aliased_families_renders_an_empty_table():
    audit = _audit(aliases={"distinct_undirected_graphs": 4, "aliased_groups": []})
    rows = _table(_render(audit), "Provenance aliasing")
    assert len(rows) == 2  # header and separator only


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


def test_every_row_has_as_many_cells_as_its_header():
    text = _render(_audit(dataset="one"), _audit(dataset="two"))
    for block in text.split("### ")[1:]:
        rows = [line for line in block.splitlines() if line.startswith("|")]
        widths = {len(row.split("|")) for row in rows}
        assert len(widths) == 1, block.split("\n", 1)[0]


def test_datasets_are_rendered_in_summary_order_with_families_in_fixed_order():
    rows = _table(_render(_audit(dataset="b"), _audit(dataset="a")), "Operator message load")
    datasets = [row.split("|")[1].strip() for row in rows[2:]]
    assert datasets[:4] == ["b"] * 4
    assert datasets[4:] == ["a"] * 4
    families = [row.split("|")[2].strip() for row in rows[2:6]]
    assert families == [rst.SHORT[name] for name in rst.GRAPH_ORDER]


def test_the_output_ends_with_exactly_one_newline():
    text = _render(_audit())
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_the_cli_writes_the_same_text_it_would_print(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"audits": [_audit()]}), encoding="utf-8")
    out = tmp_path / "tables.md"
    assert rst.main(["--summary", str(summary), "--output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == _render(_audit())
