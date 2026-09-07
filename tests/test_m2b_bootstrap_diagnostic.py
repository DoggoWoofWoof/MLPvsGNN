"""M2B's paired bootstrap, tested on screens whose answers are known in advance.

The diagnostic decides nothing, which makes it easy to write carelessly. These
tests hold it to the two things it must not get wrong: the resampling must be
PAIRED (the rungs share the drawn queries, so a difference common to both
cancels), and the artifact must not overclaim -- one seed cannot speak to seed
stability no matter how narrow an interval gets.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from test_m2b_selection_report import (  # noqa: E402
    DECLARATION_PATH,
    RUNGS,
    screen,
    write,
)

from scripts import m2b_bootstrap_diagnostic as diagnostic  # noqa: E402
from scripts import m2b_selection_report as selection  # noqa: E402

M1B_DECLARATION_PATH = REPO_ROOT / "configs" / "m1b_targeted_resolution.yaml"


@pytest.fixture(scope="module")
def declaration() -> dict[str, Any]:
    return yaml.safe_load(DECLARATION_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cells(declaration: dict[str, Any]) -> dict[str, list[str]]:
    return selection.declared_cells(declaration)


def _panel(dataset: str) -> int:
    # webqsp's real held-out panel; everything else large enough that a single
    # query does not swamp the tolerances.
    return 63 if dataset == "webqsp" else 400


def with_rows(headlines: dict[str, dict[str, Any]],
              outcomes: dict[str, dict[str, np.ndarray]] | None = None,
              ) -> dict[str, dict[str, Any]]:
    """Attach a per-query outcome vector to every fit, and a panel to every cell.

    By default each rung's vector is a deterministic Bernoulli pattern whose
    mean is that fit's recorded recall@5, so the artifact stays self-consistent
    the way a real one is.
    """

    for dataset, headline in headlines.items():
        size = _panel(dataset)
        for regime, cell in headline["cells"].items():
            cell["held_out_query_ids"] = [f"{dataset}-{regime}-q{i}" for i in range(size)]
            for rung, fit in cell["rungs"].items():
                override = (outcomes or {}).get(f"{dataset}/{regime}", {}).get(rung)
                if override is not None:
                    vector = np.asarray(override, dtype=np.float64)
                else:
                    hits = round(fit["metrics"]["recall@5"] * size)
                    vector = np.array([1.0] * hits + [0.0] * (size - hits))
                fit["per_query_recall_at_5"] = [float(value) for value in vector]
                fit["metrics"]["recall@5"] = float(vector.mean())
    return headlines


def build(tmp_path: Path, headlines: dict[str, dict[str, Any]],
          report: dict[str, Any] | None = None) -> dict[str, Any]:
    directory = write(tmp_path, headlines)
    report_path = tmp_path / "selection_report.json"
    if report is not None:
        report_path.write_text(json.dumps(report), encoding="utf-8")
    return diagnostic.build(directory, DECLARATION_PATH, report_path)


def selection_report(tmp_path: Path, headlines: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return selection.build(write(tmp_path / "screen", headlines), DECLARATION_PATH)


# --- the parameters are M1B's, not this phase's --------------------------------------


def test_the_bootstrap_parameters_are_read_from_m1bs_declaration() -> None:
    filed = yaml.safe_load(M1B_DECLARATION_PATH.read_text(encoding="utf-8"))[
        "uncertainty_procedure"
    ]
    assert diagnostic.REPLICATES == int(filed["bootstrap_resample_count"]) == 10000
    assert diagnostic.CONFIDENCE_LEVEL == float(filed["confidence_level"]) == 0.95
    assert diagnostic.RNG_SEED == int(filed["bootstrap_rng_seed"])


def test_a_phase_with_more_than_one_seed_would_need_a_different_framing(
    tmp_path: Path, cells: dict[str, list[str]], declaration: dict[str, Any]
) -> None:
    # The artifact claims it cannot measure seed stability BECAUSE there is one
    # seed. Under a 3-seed policy that sentence is false, so the script refuses
    # rather than emitting it.
    drifted = copy.deepcopy(declaration)
    drifted["seed_policy"]["count"] = 3
    path = tmp_path / "drifted.yaml"
    path.write_text(yaml.safe_dump(drifted), encoding="utf-8")
    directory = write(tmp_path, with_rows(screen(cells)))
    with pytest.raises(SystemExit) as raised:
        diagnostic.build(directory, path, tmp_path / "missing.json")
    assert "written for a one-seed phase" in str(raised.value)


# --- the resampling is paired --------------------------------------------------------


def test_two_rungs_that_agree_on_every_query_get_an_interval_of_exactly_zero(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # The sharpest statement of pairing. If the rungs were resampled
    # independently, two identical vectors would still produce a nonzero
    # interval from the two different draws; paired, the difference is
    # identically zero in every replicate.
    rng = np.random.default_rng(7)
    outcomes = {
        f"{dataset}/{regime}": dict.fromkeys(
            RUNGS, rng.integers(0, 2, size=_panel(dataset)).astype(float)
        )
        for dataset, regimes in cells.items()
        for regime in regimes
    }
    built = build(tmp_path, with_rows(screen(cells), outcomes))
    pair = built["cells"]["webqsp/R1"]["pairs"]["S2_minus_S3"]
    assert pair["ci_lower_pp"] == 0.0
    assert pair["ci_upper_pp"] == 0.0
    assert pair["point_delta_pp"] == 0.0
    assert pair["straddles_zero"] is True


def test_a_constant_offset_on_every_query_gives_an_interval_that_excludes_zero(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # S2 beats S3 on exactly the same 10% of queries in every replicate, so the
    # paired delta is 10pp with no spread at all.
    outcomes = {}
    for dataset, regimes in cells.items():
        size = _panel(dataset)
        better = np.array([1.0] * size)
        worse = np.array([1.0] * (size - size // 10) + [0.0] * (size // 10))
        for regime in regimes:
            outcomes[f"{dataset}/{regime}"] = {"S2": better, "S3": worse, "S4": worse}
    built = build(tmp_path, with_rows(screen(cells), outcomes))
    pair = built["cells"]["hotpotqa_clean/R1"]["pairs"]["S2_minus_S3"]
    assert pair["point_delta_pp"] == pytest.approx(10.0)
    assert pair["ci_lower_pp"] > 0.0
    assert pair["straddles_zero"] is False


def test_noise_that_cancels_in_the_point_estimate_still_widens_the_interval(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # Same mean, disagreeing query by query: the point delta is ~0 and the
    # interval is wide. A point estimate alone cannot tell this apart from the
    # identical-vectors case above, which is the reason the diagnostic exists.
    rng = np.random.default_rng(11)
    outcomes = {}
    for dataset, regimes in cells.items():
        size = _panel(dataset)
        for regime in regimes:
            left = rng.integers(0, 2, size=size).astype(float)
            outcomes[f"{dataset}/{regime}"] = {
                "S2": left,
                "S3": 1.0 - left,
                "S4": left,
            }
    built = build(tmp_path, with_rows(screen(cells), outcomes))
    pair = built["cells"]["webqsp/R2"]["pairs"]["S2_minus_S3"]
    assert pair["ci_width_pp"] > 50.0
    assert pair["straddles_zero"] is True


def test_the_same_cell_reproduces_exactly_when_analysed_twice(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = with_rows(screen(cells))
    first = build(tmp_path / "a", copy.deepcopy(headlines))
    second = build(tmp_path / "b", copy.deepcopy(headlines))
    assert first["cells"] == second["cells"]


def test_a_cell_reproduces_independently_of_the_cells_around_it(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # The per-cell reseeding M1B chose. Changing metaqa's numbers must not move
    # webqsp's interval, which it would if one RNG stream walked all 14 cells.
    headlines = with_rows(screen(cells))
    baseline = build(tmp_path / "a", copy.deepcopy(headlines))
    perturbed = copy.deepcopy(headlines)
    size = _panel("metaqa")
    for regime in cells["metaqa"]:
        perturbed["metaqa"]["cells"][regime]["rungs"]["S2"]["per_query_recall_at_5"] = [
            float(i % 2) for i in range(size)
        ]
    after = build(tmp_path / "b", perturbed)
    assert after["cells"]["webqsp/R1"] == baseline["cells"]["webqsp/R1"]
    assert after["cells"]["metaqa/R1"] != baseline["cells"]["metaqa/R1"]


def test_every_declared_cell_and_every_rung_pair_is_analysed(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, with_rows(screen(cells)))
    assert len(built["cells"]) == 14
    for analysis in built["cells"].values():
        assert sorted(analysis["pairs"]) == [
            "S2_minus_S3", "S2_minus_S4", "S3_minus_S4"
        ]


# --- refusals ------------------------------------------------------------------------


def test_a_fit_without_per_query_outcomes_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = with_rows(screen(cells))
    del headlines["squad_clean"]["cells"]["R1"]["rungs"]["S4"]["per_query_recall_at_5"]
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "not computable from aggregates" in str(raised.value)


def test_a_cell_without_a_shared_panel_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = with_rows(screen(cells))
    headlines["musique_clean"]["cells"]["R1"]["held_out_query_ids"] = []
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "nothing here would be paired" in str(raised.value)


def test_a_per_query_vector_of_the_wrong_length_is_refused(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = with_rows(screen(cells))
    fit = headlines["2wiki_clean"]["cells"]["R3"]["rungs"]["S2"]
    fit["per_query_recall_at_5"] = fit["per_query_recall_at_5"][:-1]
    with pytest.raises(SystemExit) as raised:
        build(tmp_path, headlines)
    assert "one outcome per query per rung" in str(raised.value)


# --- what it may and may not claim ----------------------------------------------------


def test_the_artifact_says_in_a_field_that_it_cannot_measure_seed_stability(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, with_rows(screen(cells)))
    assert built["role"] == "DIAGNOSTIC_ONLY"
    assert "training-seed stability" in built["what_this_cannot_measure"]
    assert "Nothing in this file is one of them" in built[
        "is_not_an_advancement_condition"
    ]
    assert "amending it after seeing the numbers" in built[
        "is_not_an_advancement_condition"
    ]


def test_webqsps_panel_is_named_as_the_smallest_and_priced_per_query(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, with_rows(screen(cells)))
    smallest = built["smallest_panel"]
    assert smallest["held_out_queries"] == 63
    assert smallest["cell"].startswith("webqsp/")
    # One query is worth more than three times the per-cell tolerance.
    assert smallest["one_query_moves_recall_pp"] == pytest.approx(100 / 63)
    assert smallest["one_query_moves_recall_pp"] > 3 * 0.50


def test_without_a_selection_report_no_resolution_candidates_are_invented(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    built = build(tmp_path, with_rows(screen(cells)))
    assert built["three_seed_resolution"]["status"] == "NO_SELECTION_REPORT"
    assert built["three_seed_resolution"]["cells"] == []


def test_a_candidate_needs_both_the_decision_band_and_a_straddling_interval(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # hotpotqa/R1: S4 (the rung the latency ordering selects) loses the cell by
    # a hair, so it is admitted by the tolerance, and its per-query outcomes
    # differ from the winners' only by noise, so the interval straddles zero.
    # Both conditions -- the cell qualifies.
    rng = np.random.default_rng(3)
    size = _panel("hotpotqa_clean")
    winner = rng.integers(0, 2, size=size).astype(float)
    loser = winner.copy()
    flip = np.flatnonzero(winner == 1.0)[:1]
    loser[flip] = 0.0
    outcomes = {
        "hotpotqa_clean/R1": {"S2": winner, "S3": winner, "S4": loser},
    }
    headlines = with_rows(screen(cells), outcomes)
    report = selection_report(tmp_path, copy.deepcopy(headlines))
    assert report["verdict"]["selected_rung"] == "S4"
    assert [item["cell"] for item in report["decision_band"]["cells"]] == [
        "hotpotqa_clean/R1"
    ]
    built = build(tmp_path, headlines, report)
    resolution = built["three_seed_resolution"]
    assert resolution["status"] == "CANDIDATES_IDENTIFIED"
    assert [item["cell"] for item in resolution["cells"]] == ["hotpotqa_clean/R1"]
    assert resolution["selected_rung"] == "S4"


def test_a_wide_interval_outside_the_decision_band_is_not_a_candidate(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # Every interval here straddles zero -- all three rungs tie exactly. None of
    # it matters, because the selected rung won every cell outright and there is
    # nothing for a resolution to resolve.
    headlines = with_rows(screen(cells, {"S2": 0.50, "S3": 0.50, "S4": 0.52}))
    report = selection_report(tmp_path, copy.deepcopy(headlines))
    built = build(tmp_path, headlines, report)
    assert report["decision_band"]["status"] == "DECISIVE"
    assert built["three_seed_resolution"]["status"] == "NONE"
    assert built["three_seed_resolution"]["cells"] == []


def test_a_conflict_verdict_proposes_no_resolution(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    overrides = {
        ("squad_clean", "R1"): {"S2": 0.90, "S3": 0.50, "S4": 0.50},
        ("musique_clean", "R1"): {"S2": 0.50, "S3": 0.90, "S4": 0.50},
        ("2wiki_clean", "R1"): {"S2": 0.50, "S3": 0.50, "S4": 0.90},
    }
    headlines = with_rows(screen(cells, per_cell=overrides))
    report = selection_report(tmp_path, copy.deepcopy(headlines))
    assert report["verdict"]["outcome"] == "SEMANTIC_PARETO_CONFLICT"
    built = build(tmp_path, headlines, report)
    assert built["three_seed_resolution"]["status"] == "NO_RUNG_SELECTED"
    assert built["three_seed_resolution"]["cells"] == []


def test_the_resolution_is_a_proposal_and_says_so(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    headlines = with_rows(screen(cells))
    report = selection_report(tmp_path, copy.deepcopy(headlines))
    built = build(tmp_path, headlines, report)
    text = built["three_seed_resolution"].get("is_a_proposal_not_a_launch", "")
    assert "not launched by this script" in text
    assert "Five seeds remain prohibited" in text


def test_the_diagnostic_cannot_change_the_selection_reports_verdict(
    tmp_path: Path, cells: dict[str, list[str]]
) -> None:
    # Belt and braces: running the diagnostic leaves the report byte-identical.
    headlines = with_rows(screen(cells))
    report = selection_report(tmp_path, copy.deepcopy(headlines))
    before = json.dumps(report, sort_keys=True)
    build(tmp_path, headlines, report)
    assert json.dumps(report, sort_keys=True) == before


def test_main_writes_the_diagnostic(
    tmp_path: Path, cells: dict[str, list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    directory = write(tmp_path, with_rows(screen(cells)))
    output = tmp_path / "bootstrap_diagnostic.json"
    assert diagnostic.main([
        "--headline-dir", str(directory),
        "--declaration", str(DECLARATION_PATH),
        "--selection-report", str(tmp_path / "absent.json"),
        "--output", str(output),
    ]) == 0
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["status"] == "M2B_BOOTSTRAP_DIAGNOSTIC_COMPLETE"
    assert len(written["cells"]) == 14
    printed = json.loads(capsys.readouterr().out)
    assert printed["role"] == "DIAGNOSTIC_ONLY"
