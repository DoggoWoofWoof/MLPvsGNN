"""Every number in the D0 report, checked against the files that produced it.

Written because a number typed from memory into a results document already
happened once in this pilot and read as plausible. A prose claim that no test
can reach is a claim nobody is checking.

Skips when the result files are absent -- `outputs/` is gitignored, so a clone
without them should not fail, but a clone *with* them must agree.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_D0_RESULTS.md"
RESULTS = REPO_ROOT / "outputs" / "graph_context_pilot" / "stage_d0"
DATASETS = (
    "2wiki_clean", "hotpotqa_clean", "metaqa",
    "musique_clean", "squad_clean", "webqsp",
)
SUPPORTS = ("distinct_seed_support", "two_hop_seed_support", "bridge_support")


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    payloads = {}
    for dataset in DATASETS:
        path = RESULTS / f"{dataset}_stage_d0.json"
        if not path.is_file():
            pytest.skip(f"no D0 result for {dataset}")
        payloads[dataset] = json.loads(path.read_text(encoding="utf-8"))["splits"]["validation"]
    return payloads


@pytest.fixture(scope="module")
def text() -> str:
    return DOC.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def prose(text) -> str:
    """The document with its hard wrapping collapsed.

    A claim must not become uncheckable because it landed across a line break.
    """
    return re.sub(r"\s+", " ", text)


def row(text: str, first_cell: str, *, after: str) -> list[str]:
    """The first row with this leading cell *after* an anchor, split into cells.

    Every table in the report is keyed by dataset, so an unanchored search finds
    whichever one comes first and compares the wrong column. The anchor is a
    section heading, and a typo in it fails loudly rather than matching nothing.
    """
    assert after in text, f"anchor {after!r} is not in the document"
    for line in text[text.index(after) :].splitlines():
        if line.startswith(f"| {first_cell} |"):
            return [cell.replace("*", "").strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"no row starting {first_cell!r} after {after!r}")


COVERAGE = "## 2. Coverage"
INVARIANT = "## 3. The two-path invariant"
DELTAS = "## 5. Discrimination over all candidates"
ISOLATED = "## 7. The isolated stratum"
COST = "## 9. Cost"


def auc(results, dataset, arm, feature, bucket=None) -> float:
    arms = results[dataset]["arms"][arm]
    if bucket is None:
        return arms["auc"][feature]["mean"]
    return arms["auc_by_prior_induced_degree"][feature][bucket]["mean"]


# --- coverage ------------------------------------------------------------


def test_the_coverage_table_is_the_measured_one(results, text):
    for dataset in DATASETS:
        split = results[dataset]
        cells = row(text, dataset, after=COVERAGE)
        assert cells[1] == str(split["queries_measured"]), dataset
        assert cells[2] == str(split["queries_skipped_without_both_classes"]), dataset
        assert cells[3] == f"{split['positives_per_query_median']:.1f}", dataset


def test_the_isolated_gold_column_is_derived_from_the_stratum_coverage(results, text):
    """The report's most decision-relevant number is not stored anywhere.

    It is `queries_scored` on the isolated stratum: a stratum scores only with
    both a positive and a negative inside it, and isolated candidates are 17-37%
    of every pool, so it scores exactly when a gold candidate was isolated.
    """
    for dataset in DATASETS:
        split = results[dataset]
        scored = split["arms"]["TARGET_H1"]["auc_by_prior_induced_degree"][
            "bridge_support"
        ]["isolated"]["queries_scored"]
        share = scored / split["queries_measured"]
        quoted = row(text, dataset, after=COVERAGE)[4]
        assert f"{scored} ({share:.1%})" in quoted, (dataset, quoted, scored, share)


def test_the_two_datasets_that_lose_a_third_of_their_queries_are_named(results, text):
    thin = {
        d for d in DATASETS
        if results[d]["queries_skipped_without_both_classes"] > 0.3 * 300
    }
    assert thin == {"metaqa", "webqsp"}
    assert "105 of 300 metaqa queries" in text
    assert "102 of 300 webqsp queries" in text


def test_the_thin_isolated_strata_are_disclaimed_not_averaged_in(results, prose):
    """squad, webqsp and metaqa cannot carry the headline and the doc says so."""
    for dataset in ("squad_clean", "webqsp"):
        scored = results[dataset]["arms"]["TARGET_H1"][
            "auc_by_prior_induced_degree"
        ]["bridge_support"]["isolated"]["queries_scored"]
        assert scored == 3, dataset
    assert results["metaqa"]["arms"]["TARGET_H1"]["auc_by_prior_induced_degree"][
        "bridge_support"
    ]["isolated"]["queries_scored"] == 0
    assert "rest on three queries each" in prose
    assert "Only the first three rows carry weight" in prose


# --- the invariant -------------------------------------------------------


def test_the_two_path_table_is_the_measured_one(results, text):
    for dataset in DATASETS:
        cells = row(text, dataset, after=INVARIANT)
        for offset, arm in ((1, "CAND"), (2, "TARGET_H1")):
            preservation = results[dataset]["arms"][arm]["two_path_preservation"]
            for index, key in ((offset, "directed_bridges"), (offset + 2, "seed_bridges")):
                kept = preservation[f"{key}_in_context"]
                total = preservation[key]
                expected = f"{kept:,}/{total:,} ({kept / total:.3f})"
                assert expected in cells[index], (dataset, arm, key, cells[index])


def test_target_h1_preserved_every_directed_two_path(results):
    for dataset in DATASETS:
        preservation = results[dataset]["arms"]["TARGET_H1"]["two_path_preservation"]
        assert preservation["directed_bridges_in_context"] == preservation["directed_bridges"]
        assert preservation["seed_bridges_in_context"] == preservation["seed_bridges"]


def test_the_common_successor_exception_did_not_occur(results, text):
    for dataset in DATASETS:
        preservation = results[dataset]["arms"]["TARGET_H1"]["two_path_preservation"]
        assert preservation["common_successor_bridges"] == 0, dataset
    assert "The count is 0 on all six, MEASURED, not" in text


def test_the_destruction_ranges_are_the_measured_extremes(results, text):
    def destroyed(key):
        return [
            1 - p[f"{key}_in_context"] / p[key]
            for p in (results[d]["arms"]["CAND"]["two_path_preservation"] for d in DATASETS)
        ]

    candidate, seed = destroyed("directed_bridges"), destroyed("seed_bridges")
    assert f"**{min(candidate):.0%} to {max(candidate):.0%}**" in text
    assert f"**{min(seed):.0%} to {max(seed):.0%}**" in text


# --- the control ---------------------------------------------------------


def test_the_negative_control_is_zero_everywhere_including_every_stratum(results, prose):
    for dataset in DATASETS:
        assert auc(results, dataset, "CAND", "distinct_seed_support") == pytest.approx(
            auc(results, dataset, "TARGET_H1", "distinct_seed_support"), abs=5e-5
        ), dataset
        for bucket in ("isolated", "degree_1", "low_degree", "ordinary"):
            cand = auc(results, dataset, "CAND", "distinct_seed_support", bucket)
            target = auc(results, dataset, "TARGET_H1", "distinct_seed_support", bucket)
            if cand == cand:  # a stratum nobody scored is NaN in both arms
                assert cand == pytest.approx(target, abs=5e-5), (dataset, bucket)
    assert "+0.0000 on all six datasets, and in every one of the four degree strata" in prose


# --- the deltas ----------------------------------------------------------


def test_the_overall_delta_table_is_the_measured_one(results, text):
    features = (
        "seed_distance", "distinct_seed_support",
        "two_hop_seed_support", "bridge_support",
    )
    for dataset in DATASETS:
        cells = row(text, dataset, after=DELTAS)
        for index, feature in enumerate(features, start=1):
            delta = auc(results, dataset, "TARGET_H1", feature) - auc(
                results, dataset, "CAND", feature
            )
            assert f"{delta:+.4f}" in cells[index], (dataset, feature, cells[index])


def test_the_headline_ranges_match_the_deltas(results, text):
    deltas = {
        feature: {
            d: auc(results, d, "TARGET_H1", feature) - auc(results, d, "CAND", feature)
            for d in DATASETS
        }
        for feature in ("bridge_support", "two_hop_seed_support")
    }
    up = sorted(v for v in deltas["bridge_support"].values() if v > 0)
    assert len(up) == 5, "the doc says bridge_support improves on 5 of 6"
    assert f"5 of 6 datasets by {min(up):.3f}-{max(up):.3f} AUC" in text

    down = [v for v in deltas["two_hop_seed_support"].values() if v < 0]
    assert len(down) == 5, "the doc says two_hop_seed_support degrades on 5 of 6"
    assert f"*degrades* on 5 of 6, by as much as {-min(down):.3f}" in text


def test_no_claim_that_the_contexts_are_close_on_every_quantity(text):
    """A 0.2742 delta is not "within 0.07". The first draft said it was."""
    assert "within 0.07 AUC on every quantity" not in text


def test_the_absent_error_bar_is_stated_not_implied(results, text):
    summary = results["2wiki_clean"]["arms"]["CAND"]["auc"]["bridge_support"]
    assert set(summary) == {"mean", "median", "queries_scored", "queries_unscorable"}
    assert "These are point estimates with no error bar." in text
    assert "no conclusion in this document rests on the sign of an overall" in text


# --- the result ----------------------------------------------------------


def test_cand_scores_exactly_one_half_on_every_isolated_support(results):
    """The claim §7 rests on, in the data rather than only in the theorem."""
    scored = 0
    for dataset in DATASETS:
        stratified = results[dataset]["arms"]["CAND"]["auc_by_prior_induced_degree"]
        for feature in SUPPORTS:
            stat = stratified[feature]["isolated"]
            if stat["queries_scored"] == 0:
                continue
            scored += 1
            assert stat["mean"] == 0.5, (dataset, feature, stat["mean"])
            assert stat["median"] == 0.5, (dataset, feature, stat["mean"])
    assert scored == 15, "five datasets with an isolated stratum, three supports each"


def test_the_isolated_stratum_table_is_the_measured_one(results, text):
    for dataset in DATASETS:
        stat = results[dataset]["arms"]["TARGET_H1"]["auc_by_prior_induced_degree"][
            "bridge_support"
        ]["isolated"]
        cells = row(text, dataset, after=ISOLATED)
        if stat["queries_scored"] == 0:
            continue
        assert f"0.5000 -> {stat['mean']:.4f}" in cells[3], (dataset, cells[3])
        two_hop = results[dataset]["arms"]["TARGET_H1"][
            "auc_by_prior_induced_degree"
        ]["two_hop_seed_support"]["isolated"]
        assert f"0.5000 -> {two_hop['mean']:.4f}" in cells[2], (dataset, cells[2])


def test_the_weighted_range_covers_only_the_three_measurable_datasets(results, text):
    weighted = ("2wiki_clean", "musique_clean", "hotpotqa_clean")
    values = [auc(results, d, "TARGET_H1", "bridge_support", "isolated") for d in weighted]
    assert f"{min(values):.2f}-{max(values):.2f} mean AUC" in text


# --- the reading -----------------------------------------------------------


def test_the_saturation_figures_quoted_for_two_hop_support_are_real(results, text):
    stratified = results["2wiki_clean"]["arms"]
    for bucket, label in (("degree_1", "degree 1"), ("low_degree", "degree 2-4")):
        cand = stratified["CAND"]["auc_by_prior_induced_degree"]["two_hop_seed_support"][bucket]
        target = stratified["TARGET_H1"]["auc_by_prior_induced_degree"]["two_hop_seed_support"][bucket]
        assert f"{cand['mean']:.4f} to {target['mean']:.4f} at {label}" in text


def test_the_squad_counter_example_is_reported_not_buried(results, prose):
    delta = auc(results, "squad_clean", "TARGET_H1", "bridge_support") - auc(
        results, "squad_clean", "CAND", "bridge_support"
    )
    assert delta < 0
    cand = auc(results, "squad_clean", "CAND", "bridge_support", "low_degree")
    target = auc(results, "squad_clean", "TARGET_H1", "bridge_support", "low_degree")
    assert f"drops from {cand:.4f} to {target:.4f} at degree 2-4" in prose
    total = results["squad_clean"]["arms"]["CAND"]["two_path_preservation"]["directed_bridges"]
    other = results["2wiki_clean"]["arms"]["CAND"]["two_path_preservation"]["directed_bridges"]
    assert f"{total:,} candidate two-paths against 2wiki's {other:,}" in prose


def test_the_seed_distance_disclaimer_quotes_its_own_numbers(results, prose):
    isolated = auc(results, "2wiki_clean", "CAND", "seed_distance", "isolated")
    assert f"it reads {isolated:.4f} on 2wiki" in prose
    overall = [
        auc(results, d, arm, "seed_distance")
        for d in DATASETS for arm in ("CAND", "TARGET_H1")
    ]
    assert f"of {min(overall):.2f}-{max(overall):.2f}" in prose
    ties = [
        results[d]["arms"][arm]["tie_fraction"]["seed_distance"]
        for d in DATASETS for arm in ("CAND", "TARGET_H1")
    ]
    assert f"tie fraction of {min(ties):.3f}-{max(ties):.3f}" in prose
    webqsp = results["webqsp"]["arms"]
    assert "moves {:.4f} -> {:.4f}".format(
        webqsp["CAND"]["auc_by_prior_induced_degree"]["seed_distance"]["isolated"]["mean"],
        webqsp["TARGET_H1"]["auc_by_prior_induced_degree"]["seed_distance"]["isolated"]["mean"],
    ) in prose


# --- cost ----------------------------------------------------------------


def test_the_latency_table_is_the_measured_one(results, text):
    for dataset in DATASETS:
        cells = row(text, dataset, after=COST)
        arms = results[dataset]["arms"]
        marginal = arms["TARGET_H1"]["latency_ms"]["median"] - arms["CAND"]["latency_ms"]["median"]
        for index, arm in ((1, "CAND"), (2, "TARGET_H1")):
            latency = arms[arm]["latency_ms"]
            expected = f"{latency['median']:.1f} / {latency['p95']:.1f}"
            assert expected in cells[index], (dataset, arm, cells[index])
        assert f"+{marginal:.1f}" in cells[3], (dataset, cells[3])


def test_the_marginal_reading_is_the_one_the_doc_leads_with(results, text):
    """The 338 ms figure is the diagnostic's cost, and CAND pays most of it."""
    marginal = sorted(
        results[d]["arms"]["TARGET_H1"]["latency_ms"]["median"]
        - results[d]["arms"]["CAND"]["latency_ms"]["median"]
        for d in DATASETS
    )
    assert f"+{marginal[0]:.1f} to +{marginal[-1]:.1f} ms p50" in text
    historical = sorted(results[d]["arms"]["CAND"]["latency_ms"]["median"] for d in DATASETS)
    assert f"already costs {historical[-2]:.0f}-{historical[-1]:.0f} ms p50" in text


def test_the_compute_figure_is_labelled_a_bound_not_a_measurement(prose):
    assert "**BOUND**" in prose
    assert "<= 0.75 CPU-hours and <= $0.48" in prose
    assert "declared ceiling of <= 1.2 CPU-hours and <= $0.80" in prose


# --- the decision --------------------------------------------------------


def test_the_decision_is_stated_once_and_is_the_scoped_one(text):
    assert text.count("ADVANCE TO THE SMALL D1 PILOT") == 2  # headline table and §10
    assert "2wiki, one seed, `CAND` vs `TARGET_H1`" in text
    assert "Nothing larger." in text


def test_the_report_does_not_claim_a_ranking_result(text, prose):
    assert "There is no R@k, no MRR" in text
    assert "No model was trained." in text
    assert "we fixed the graph" not in text.lower()
    # The claim is nameable -- it is what the next stage has to earn. What
    # the document must not do is assert it, so the negation is pinned.
    assert 'does **not** support "restoring the graph improves ranking"' in prose


def test_the_deflationary_reading_is_addressed_with_its_own_numbers(results, prose):
    """The gap quoted against the shared zero set must be the measured gap."""
    stratified = results["2wiki_clean"]["arms"]["TARGET_H1"]["auc_by_prior_induced_degree"]
    two_hop = stratified["two_hop_seed_support"]["isolated"]["mean"]
    bridges = stratified["bridge_support"]["isolated"]["mean"]
    assert f"they score {two_hop:.4f} and {bridges:.4f} on that stratum" in prose
    assert f"A gap of {bridges - two_hop:.4f} between" in prose
