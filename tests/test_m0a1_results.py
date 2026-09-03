"""The M0A.1 results document may not say more than the run output supports.

Prose tests always run: they check the report labels its claims, states the
rejected wording alongside the locked wording, and never claims to authorise
M0B. Numeric tests cross-check every figure quoted in the report against the
actual run and analysis JSON, and skip when the gitignored outputs are absent
rather than passing vacuously.
"""

from __future__ import annotations

import inspect
import json
import pathlib

import pytest
import yaml

from mp_retrieval.candidate_expansion_v2 import expand
from scripts.analyze_m0a1_overlap import CONFIG_PATH as ANALYSIS_CONFIG_PATH
from scripts.analyze_m0a1_overlap import decide, universal_budget

DOC = pathlib.Path("docs/M0A1_OVERLAP_RESULTS.md")
RESULTS_ROOT = pathlib.Path("outputs/m0a1_overlap")
ANALYSIS_PATH = pathlib.Path("outputs/m0a1_overlap_analysis.json")
DATASETS = ("squad_clean", "2wiki_clean", "metaqa")


@pytest.fixture(scope="module")
def report() -> str:
    return DOC.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    """Markdown source word-wraps; a quoted phrase may cross a line break."""
    return " ".join(text.split())


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(ANALYSIS_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results() -> dict:
    loaded = {}
    for name in DATASETS:
        path = RESULTS_ROOT / f"{name}.json"
        if not path.exists():
            pytest.skip(f"{path} is gitignored and absent; numeric checks skipped")
        loaded[name] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@pytest.fixture(scope="module")
def analysis(results, config) -> dict:
    if not ANALYSIS_PATH.exists():
        pytest.skip(f"{ANALYSIS_PATH} is gitignored and absent; numeric checks skipped")
    return decide(results, config)


def _row(section: str, dataset: str, family: str) -> list[str]:
    prefix = f"| {dataset} | {family} |"
    lines = [ln for ln in section.splitlines() if ln.startswith(prefix)]
    assert len(lines) == 1, f"expected one row for {dataset}/{family}, got {len(lines)}"
    return [part.strip().strip("*") for part in lines[0].strip().strip("|").split("|")]


# --- the report may not authorise anything, and states both wordings ---


def test_the_report_opens_by_refusing_to_authorise_a_launch(report):
    head = report[: report.index("## 2.")]
    assert "No launch is authorised by this document." in head
    assert "M0A.1 COMPLETE" in head


def test_the_report_names_everything_it_does_not_authorise(report):
    section = report[report.index("## 8.") : report.index("## 9.")]
    for forbidden in (
        "Not M0B",
        "Not a directional re-evaluation",
        "Not a claim about ranking",
        "Not a per-dataset budget",
    ):
        assert forbidden in section, forbidden


def test_m0a_wording_discipline_is_repeated_unchanged_not_reworded(report):
    assert (
        "Directional selection provided no incremental candidate-headroom "
        "benefit over bounded structural expansion in M0A." in _flat(report)
    )


def test_the_locked_wording_and_the_rejected_sentence_are_both_present(report):
    assert "R2 already exposed the answer; R3 just makes it scoreable." in _flat(report)
    assert "R3 extends the scored universe beyond" in _flat(report)
    assert "does not hold on any cell measured" in _flat(report)


def test_scope_limits_are_stated_not_implied(report):
    section = _flat(report[report.index("Not claimed") :])
    for limit in ("three datasets", "100-query sample", "L1_DIRECTIONAL"):
        assert limit in section, limit


# --- the verdict in the prose is the verdict the rule computes ---


def test_the_quoted_verdict_is_the_one_the_filed_rule_produces(report, analysis):
    assert f"**`{analysis['verdict']}`**" in report
    assert analysis["verdict"] == "ADVANCE"


def test_advancement_was_never_read_as_automatic(report, analysis):
    assert analysis["advancement_is_automatic"] is False
    assert "is not automatic" in _flat(report).lower()


# --- every quoted number is the measured one ---


def test_the_cross_tabulation_table_matches_the_run_output(report, results):
    section = report[report.index("## 2.") : report.index("Restated as the three")]
    for name, result in results.items():
        for family, row in result["overlap"].items():
            fields = _row(section, name, family)
            cells = row["additive"]["cross_tabulation"]["gold_instances"]
            assert fields[2] == str(cells["IN_U2_AND_RECOVERED"]), (name, family)
            assert fields[3] == str(cells["IN_U2_NOT_RECOVERED"]), (name, family)
            assert fields[4] == str(cells["BEYOND_U2_RECOVERED"]), (name, family)
            assert fields[5] == str(cells["BEYOND_U2_NOT_RECOVERED"]), (name, family)


def test_beyond_u2_recovered_is_actually_zero_everywhere_the_report_claims(results):
    for result in results.values():
        for row in result["overlap"].values():
            cells = row["additive"]["cross_tabulation"]["gold_instances"]
            assert cells["BEYOND_U2_RECOVERED"] == 0


def test_r1_equals_r2_is_reconfirmed_on_every_dataset(results):
    for name, result in results.items():
        assert result["invariants"]["r1_ceiling_equals_r2_ceiling_exactly"] is True, name


def test_the_cross_validation_table_matches_m0a1s_own_curve_movement(report, results):
    section = report[report.index("## 4.") : report.index("## 5.")]
    checks = {
        ("2wiki_clean", "structural_only", 64): ("recall_ceiling@5", 15.25),
        ("metaqa", "structural_only", 64): ("any_gold_at_pool", 30.00),
    }
    for (dataset, family, budget), (metric, expected_points) in checks.items():
        baseline = results[dataset]["r1"]["headroom"][metric]
        point = next(
            p
            for p in results[dataset]["curve"][family]["points"]
            if p["budget"] == budget
        )
        points_moved = (point[metric] - baseline) * 100.0
        assert points_moved == pytest.approx(expected_points, abs=0.01)
        assert f"{expected_points:.2f}" in section or f"+{expected_points:.2f}" in section


def test_the_universal_budget_matches_the_recomputed_one(report, results):
    section = report[report.index("## 5.") : report.index("## 6.")]
    computed = universal_budget(results)
    assert computed["smallest_sufficient_universal_budget"] == 64
    assert "smallest_sufficient_universal_budget: 64" in section


def test_the_systems_numbers_match_the_run_output(report, results):
    section = report[report.index("## 6.") : report.index("## 7.")]
    for name, result in results.items():
        rss_gb = result["systems"]["peak_process_rss_bytes"] / 1e9
        assert f"{rss_gb:.2f} GB" in section, name


def test_the_leakage_signature_check_matches_the_live_function(report):
    section = report[report.index("## 7.") :]
    parameters = set(inspect.signature(expand).parameters)
    assert parameters == {
        "method", "rowptr", "col", "node_embeddings", "query_embedding",
        "anchor", "pool", "seeds", "budget", "num_nodes",
    }
    for name in sorted(parameters):
        assert name in section, name


def test_no_ranker_claim_is_ever_made(report):
    assert "no ranker has been trained" in report.lower()
