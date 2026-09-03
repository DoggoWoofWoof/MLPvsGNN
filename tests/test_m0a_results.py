"""The results document may not say more than the probe output supports.

Two kinds of test live here. The prose tests always run: they check that the
report labels its claims, records the deviations, and never claims to
authorise anything. The numeric tests cross-check every figure quoted in the
report against the probe JSON, and skip when the gitignored outputs are absent
rather than passing vacuously.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest
import yaml

from scripts.analyze_m0a_probe import CONFIG_PATH, decide

DOC = pathlib.Path("docs/M0A_PROBE_RESULTS.md")
PROBES = pathlib.Path("outputs/m0a_probe")
DATASETS = ("squad_clean", "2wiki_clean", "metaqa")


@pytest.fixture(scope="module")
def report() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def probes() -> dict:
    loaded = {}
    for name in DATASETS:
        path = PROBES / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{path} is gitignored and absent; numeric checks skipped")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# --- the report may not authorise anything ---


def test_the_report_opens_by_refusing_to_authorise_a_launch(report):
    head = report[: report.index("## 2.")]
    assert "No launch is authorised by this document." in head
    assert "M0A COMPLETE" in head


def test_the_report_names_everything_m0a_does_not_authorise(report):
    section = report[report.index("## 11.") :]
    assert "advancement_is_automatic: false" in section
    for forbidden in (
        "training QLS",
        "training a GNN",
        "launching M0B",
        "resuming E2",
        "opening F",
        "migrating workspaces",
        "reading any test split",
    ):
        assert forbidden in section, forbidden


def test_the_report_claims_no_training_and_no_test_split(report, probes):
    assert "`trained_anything: false` on all three" in report
    assert "`test_split_read: false` on all three" in report
    for name, probe in probes.items():
        assert probe["trained_anything"] is False, name
        assert probe["test_split_read"] is False, name


# --- the verdict in the prose is the verdict the rule computes ---


def test_the_quoted_verdict_is_the_one_the_filed_rule_produces(report, probes, config):
    computed = decide(probes, config)["verdict"]
    assert f"**`{computed}`**" in report
    assert computed == "MOVEMENT_UNDER_A_BREACHED_ABORT_RULE"


# --- every quoted number is the measured one ---


def _cells(section: str, dataset: str, family: str) -> list[str]:
    """The cells of the one table row for this dataset and family."""
    prefix = f"| {dataset} | {family} |"
    lines = [ln for ln in section.splitlines() if ln.startswith(prefix)]
    assert len(lines) == 1, f"expected one row for {dataset}/{family}, got {len(lines)}"
    return [part.strip() for part in lines[0].strip().strip("|").split("|")]


def test_the_movement_table_matches_the_probe_output(report, probes):
    """Recovered and lost gold counts are read off the runs, not retyped."""
    section = report[report.index("## 4.") : report.index("## 5.")]
    for name, probe in probes.items():
        for family in ("structural_only", "knn_only", "baseline_a_simple"):
            cell = f"R3/{family}/STRUCTURAL_NEIGHBOUR/matched"
            move = probe["movement"][cell]
            fields = _cells(section, name, family)
            assert fields[-1] == str(move["golds_lost"]), (name, family, "lost")
            assert fields[-2] == str(move["missing_golds_recovered"]), (name, family)


def test_no_gold_was_lost_anywhere_as_the_report_claims(report, probes):
    assert "`golds_lost: 0` in every cell on every dataset" in report
    for name, probe in probes.items():
        for cell, move in probe["movement"].items():
            assert move["golds_lost"] == 0, (name, cell)


def test_the_r1_baseline_table_matches_the_probe_output(report, probes):
    whole = report[report.index("## 2.") : report.index("## 3.")]
    section = whole[whole.index("R1 candidate ceilings") :]
    for name, probe in probes.items():
        headroom = probe["regimes"]["R1"]["headroom"]
        line = next(ln for ln in section.splitlines() if ln.startswith(f"| {name} |"))
        fields = [part.strip() for part in line.strip().strip("|").split("|")]
        assert fields[1] == f"{headroom['any_gold_at_pool']:.4f}", name
        assert fields[2] == f"{headroom['recall_ceiling@5']:.4f}", name
        assert fields[3] == f"{headroom['all_gold_at_pool']:.4f}", name
        assert fields[4] == str(headroom["queries_with_no_gold_in_pool"]), name


def test_the_knn_family_recovers_nothing_as_the_report_claims(report, probes):
    assert "recovers **exactly zero** golds on every dataset" in report
    for name, probe in probes.items():
        for cell, move in probe["movement"].items():
            if "/knn_only/" in cell:
                assert move["missing_golds_recovered"] == 0, (name, cell)


def test_the_abort_ratios_are_the_measured_ratios(report, probes, config):
    section = report[report.index("## 9.") : report.index("## 10.")]
    factor = float(config["compute"]["abort_rule_latency_factor"])
    assert f"filed factor of {factor}" in section
    quoted = {
        float(value) for value in re.findall(r"\*\*(\d+\.\d\d)\*\*", section)
    }
    measured = set()
    for probe in probes.values():
        for key, cell in probe["regimes"].items():
            if not (key.startswith("R3") and key.endswith("matched")):
                continue
            ratio = (
                cell["expansion_latency_ms"]["p95"]
                / cell["context_build_latency_ms"]["p95"]
            )
            if ratio > factor:
                measured.add(round(ratio, 2))
    assert quoted == measured
    assert measured, "the report describes a breach; the data must contain one"


def test_the_peak_memory_figures_are_the_measured_ones(report, probes):
    section = report[report.index("## 9.") : report.index("## 10.")]
    for name, probe in probes.items():
        gib = probe["systems"]["peak_process_rss_bytes"] / 2**30
        assert f"| {name} | {gib:.2f} GiB |" in section, name


def test_the_arm_agreement_table_matches_the_probe_output(report, probes):
    """Section 3 rests on these; a retyped Jaccard would make the null unfalsifiable."""
    section = report[report.index("## 3.") : report.index("## 4.")]
    checked = 0
    for name, probe in probes.items():
        for family in ("structural_only", "knn_only", "baseline_a_simple"):
            matched = {
                method: probe["regimes"][f"R3/{family}/{method}/matched"]
                for method in ("L1_DIRECTIONAL", "STRUCTURAL_NEIGHBOUR")
            }
            capped = matched["L1_DIRECTIONAL"]["queries_with_a_capped_seed"]
            fields = _cells(section, name, family)
            assert fields[4] == f"{capped} / {probe['queries']}", (name, family)
            checked += 1
    assert checked == 9


def test_a_family_where_no_cap_bound_is_marked_as_no_comparison(report, probes):
    section = report[report.index("## 3.") : report.index("## 4.")]
    for name, probe in probes.items():
        for family in ("structural_only", "knn_only", "baseline_a_simple"):
            cell = probe["regimes"][f"R3/{family}/L1_DIRECTIONAL/matched"]
            expected = "yes" if cell["queries_with_a_capped_seed"] else "**no**"
            assert _cells(section, name, family)[-1] == expected, (name, family)


# --- the report labels what it does not know ---


def test_the_unmeasured_overlap_is_labelled_not_measured(report):
    section = " ".join(
        report[report.index("## 8.") : report.index("## 9.")].split()
    )
    assert "**NOT MEASURED" in section
    assert "It is therefore **not** structurally guaranteed" in section
    assert "this probe did not measure the overlap" in section


def test_the_report_never_claims_r3_reached_what_r2_could_not(report):
    """The claim section 8 forbids must not appear anywhere else in the doc."""
    assert '"finds" golds R2 could not reach is unsupported' in report
    body = report.replace('"finds" golds R2 could not reach is unsupported', "")
    for overclaim in (
        "golds R2 could not reach",
        "outside R2's context",
        "new information R2",
    ):
        assert overclaim not in body, overclaim


def test_the_workspace_figure_is_declared_a_bound_not_a_measurement(report):
    section = report[report.index("## 9.") : report.index("## 10.")]
    assert "**a declared bound, not a measurement**" in section
    assert "`8 * graph_expansion_cap * 4`" in section
    assert "none should be quoted as if it were" in section


def test_the_memory_estimate_is_reported_as_wrong_rather_than_excused(report):
    section = report[report.index("## 9.") : report.index("## 10.")]
    assert "The compute estimate was wrong on memory." in section
    assert "should not be reused for M0B without revision" in section


# --- deviations are recorded, not smoothed over ---


def test_all_three_process_deviations_are_recorded(report):
    section = report[report.index("## 10.") : report.index("## 11.")]
    assert "None is hidden." in section
    assert "PROCESS_DEVIATION_RECORDED_NOT_HIDDEN" in section
    assert "The abort rule was applied post hoc" in section
    assert "the_two_branches_are_not_disjoint" in section
    assert "before any M0A result was read" in section


def test_the_two_executions_are_reported_as_verified_not_asserted(report):
    section = report[report.index("## 10.") : report.index("## 11.")]
    assert "byte-identical\nto execution 1 on all 14 shared cells" in section
    assert "verified, not\nasserted" in section


def test_the_report_does_not_soften_the_breached_abort_rule(report):
    section = report[report.index("## 9.") : report.index("## 10.")]
    assert "The runner did not implement the abort." in section
    assert "not softened to fit" in section


def test_the_null_about_the_method_is_not_widened_into_a_null_about_expansion(report):
    section = report[report.index("## 3.") : report.index("## 4.")]
    assert "It does not become a null about one-hop\nexpansion itself" in section
