"""Amendment 3's resolution, exercised before its fits exist.

The script cannot be checked against real seed-1 numbers yet -- that is the
point of committing it first. What CAN be checked, and is checked here, is every
property that must hold whatever those numbers turn out to be:

    the substitution is faithful (three identical seeds must leave the verdict
    bit-for-bit unchanged); the rule is the screen's own, not a copy; the
    thresholds do not move; the pairing is real; and a fit that reuses seed 0's
    weights under a new seed label is refused rather than averaged.

The synthetic seeds below are copies of the screen's seed 0. That is a
degenerate case on purpose: it is the one input whose correct output is known in
advance without knowing anything about S3 or S4.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts import m2b_resolution_report as resolution  # noqa: E402
from scripts import m2b_selection_report as selection  # noqa: E402

DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml").read_text(encoding="utf-8")
)
HEADLINES_EXIST = resolution.HEADLINE_DIR.is_dir() and bool(
    list(resolution.HEADLINE_DIR.glob("*.json"))
)
needs_headlines = pytest.mark.skipif(
    not HEADLINES_EXIST,
    reason="run the M2B screen; outputs/ is gitignored",
)


# --------------------------------------------------------------------------
# What the script reads, and refuses to assume
# --------------------------------------------------------------------------


def test_the_scope_comes_from_the_declaration_not_from_this_script() -> None:
    filed = resolution.scope(DECLARATION)
    assert filed["cells"] == ["squad_clean/R1", "musique_clean/R1"]
    assert filed["rungs"] == ["S3", "S4"]
    # Seed 0 is reused; 1 and 2 are the new fits.
    assert filed["seeds"] == [0, 1, 2]


def test_a_declaration_without_the_amendment_is_refused() -> None:
    """This script applies an amendment; it must not stand in for one."""

    without = {key: value for key, value in DECLARATION.items() if key != "resolution_amendment"}
    with pytest.raises(SystemExit, match="files no resolution_amendment"):
        resolution.scope(without)


def test_a_scope_whose_arithmetic_does_not_close_is_refused() -> None:
    """8 fits is 2 cells x 2 rungs x 2 new seeds. If the scope grows, so must the count."""

    widened = copy.deepcopy(DECLARATION)
    widened["resolution_amendment"]["scope"]["cells"].append("2wiki_clean/R3")
    with pytest.raises(SystemExit, match="but declares 8"):
        resolution.scope(widened)


def test_the_procedure_is_the_diagnostics_own() -> None:
    """Reused, not restated, so the two cannot drift apart."""

    from scripts import m2b_bootstrap_diagnostic as diagnostic

    assert resolution.REPLICATES == diagnostic.REPLICATES == 10000
    assert resolution.CONFIDENCE_LEVEL == diagnostic.CONFIDENCE_LEVEL == 0.95
    assert resolution.RNG_SEED == diagnostic.RNG_SEED


def test_the_interval_says_what_it_is_not() -> None:
    assert "NOT a random-effects" in resolution.WHAT_THE_INTERVAL_IS
    assert "resamples queries, not seeds" in resolution.WHAT_THE_INTERVAL_IS


# --------------------------------------------------------------------------
# The seed-wise statistic
# --------------------------------------------------------------------------


def test_the_seedwise_block_reports_every_field_the_amendment_asks_for() -> None:
    recalls = {0: {"S4": 0.90, "S3": 0.92}, 1: {"S4": 0.91, "S3": 0.90}, 2: {"S4": 0.93, "S3": 0.92}}
    block = resolution.seedwise(recalls, [0, 1, 2], "S4", "S3")

    assert block["per_seed_delta_pp"]["0"] == pytest.approx(-2.0)
    assert block["per_seed_delta_pp"]["1"] == pytest.approx(1.0)
    assert block["per_seed_delta_pp"]["2"] == pytest.approx(1.0)
    assert block["mean_delta_pp"] == pytest.approx(0.0)
    assert block["min_delta_pp"] == pytest.approx(-2.0)
    assert block["max_delta_pp"] == pytest.approx(1.0)
    assert block["sample_sd_pp"] == pytest.approx(np.std([-2.0, 1.0, 1.0], ddof=1))
    # A mean of zero with a sign pattern of -++ is exactly the case a reader
    # must be able to see, and a mean alone would hide it.
    assert block["sign_pattern"] == "-++"
    assert block["signs_agree"] is False


def test_signs_agree_when_they_do() -> None:
    recalls = {seed: {"S4": 0.9, "S3": 0.8} for seed in (0, 1, 2)}
    block = resolution.seedwise(recalls, [0, 1, 2], "S4", "S3")
    assert block["sign_pattern"] == "+++"
    assert block["signs_agree"] is True


# --------------------------------------------------------------------------
# The bootstrap's pairing
# --------------------------------------------------------------------------


def test_the_same_queries_are_used_for_both_rungs_and_every_seed() -> None:
    """A constant per-query gap must produce a zero-width interval.

    If the resample were drawn independently per rung or per seed, the two
    means would differ by sampling noise and the interval would open up. It
    staying shut is the evidence that one index set reaches all six arrays.
    """

    n = 200
    rng = np.random.default_rng(0)
    # Scaled so base + 0.1 never reaches 1.0: a clipped tail would make the
    # per-query gap non-constant and this test would be checking arithmetic
    # rather than pairing.
    base = rng.random(n) * 0.5
    arrays = {seed: {"S3": base, "S4": base + 0.1} for seed in (0, 1, 2)}
    block = resolution.paired_bootstrap(arrays, [0, 1, 2], "S4", "S3", n)

    assert block["ci_width_pp"] == pytest.approx(0.0, abs=1e-9)
    assert block["ci_lower_pp"] == pytest.approx(
        (arrays[0]["S4"] - arrays[0]["S3"]).mean() * 100, abs=1e-9
    )
    assert block["straddles_zero"] is False
    assert block["replicates"] == 10000


def test_the_statistic_averages_the_seeds_rather_than_pooling_them() -> None:
    """Two seeds ahead and one behind average to the mean, not to the majority."""

    n = 300
    rng = np.random.default_rng(1)
    base = rng.random(n)
    arrays = {
        0: {"S3": base, "S4": base - 0.30},
        1: {"S3": base, "S4": base + 0.15},
        2: {"S3": base, "S4": base + 0.15},
    }
    block = resolution.paired_bootstrap(arrays, [0, 1, 2], "S4", "S3", n)
    # mean(-30, +15, +15) = 0
    assert block["ci_lower_pp"] == pytest.approx(0.0, abs=1e-9)
    assert block["straddles_zero"] is True


def test_the_bootstrap_is_reproducible() -> None:
    n = 120
    rng = np.random.default_rng(7)
    arrays = {
        seed: {"S3": rng.random(n), "S4": rng.random(n)} for seed in (0, 1, 2)
    }
    first = resolution.paired_bootstrap(arrays, [0, 1, 2], "S4", "S3", n)
    second = resolution.paired_bootstrap(arrays, [0, 1, 2], "S4", "S3", n)
    assert first == second


# --------------------------------------------------------------------------
# The refusals that protect the pairing
# --------------------------------------------------------------------------


def test_a_reused_seed_0_checkpoint_at_a_new_seed_is_refused() -> None:
    """The trap this whole design has to avoid, asserted directly.

    M2 fit seed 0. A fit carrying those weights under a seed-1 label would give
    S3 the same recall three times, so all three deltas would share an
    identical term and the comparison would be S4's variance against a
    constant -- reported by the artifact as a clean reuse.
    """

    fit = {"reused_from_m2": True, "seed": 1}
    with pytest.raises(SystemExit, match="M2 fit seed 0 only"):
        resolution._refuse_a_reused_fit_at_a_new_seed(fit, "squad_clean/R1", "S3", 1)


def test_reuse_at_seed_0_is_exactly_how_the_screen_ran() -> None:
    assert (
        resolution._refuse_a_reused_fit_at_a_new_seed(
            {"reused_from_m2": True, "seed": 0}, "squad_clean/R1", "S3", 0
        )
        is None
    )


def test_a_fit_filed_under_the_wrong_seed_is_refused() -> None:
    artifact = {"cells": {"R1": {"rungs": {"S4": {"seed": 0}}}}}
    with pytest.raises(SystemExit, match="records seed 0"):
        resolution._fit(artifact, "squad_clean/R1", "R1", "S4", 1)


# --------------------------------------------------------------------------
# The substitution, against the one input whose answer is known in advance
# --------------------------------------------------------------------------


@pytest.fixture
def three_identical_seeds(tmp_path: Path) -> Path:
    """Seeds 1 and 2 as copies of the screen's seed 0, relabelled.

    Degenerate on purpose. When all three seeds carry the same numbers their
    mean is that number, so the substitution must be a no-op and the resolved
    verdict must equal the original exactly -- an expectation that holds without
    knowing anything about what S3 and S4 actually scored.
    """

    filed = resolution.scope(DECLARATION)
    root = tmp_path / "resolution"
    for seed in filed["seeds"][1:]:
        for cell in filed["cells"]:
            dataset = cell.partition("/")[0]
            artifact = json.loads(
                (resolution.HEADLINE_DIR / f"{dataset}.json").read_text(encoding="utf-8")
            )
            artifact["seed"] = seed
            # The screen reused M2's S3 at seed 0; a real seed-1 container
            # refits it, so the copy has to drop the flag or the reuse refusal
            # fires -- which is itself the behaviour under test above.
            artifact["reused_rung"] = None
            for regime in artifact["cells"].values():
                for rung, fit in regime["rungs"].items():
                    fit["seed"] = seed
                    if rung in filed["rungs"]:
                        fit["reused_from_m2"] = False
            path = root / f"seed{seed}" / f"{dataset}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return root


@needs_headlines
def test_three_identical_seeds_leave_the_verdict_untouched(three_identical_seeds: Path) -> None:
    """The strongest available check that the substitution changes nothing else.

    Any leakage -- a rung swapped, a cell mis-keyed, a mean taken over the wrong
    axis, a threshold quietly re-read -- would move a number here, because the
    correct answer is "exactly what the screen already said".
    """

    report = resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)
    original = selection.build(resolution.HEADLINE_DIR)

    assert report["verdict_before"]["outcome"] == "SEMANTIC_PARETO_CONFLICT"
    assert report["verdict_after"]["outcome"] == original["verdict"]["outcome"]
    assert report["verdict_after"]["selected_rung"] == original["verdict"]["selected_rung"]
    # Exact on everything the rule actually decides on...
    assert report["effectiveness_after"]["admissible"] == original["effectiveness"]["admissible"]
    for rung, after in report["effectiveness_after"]["by_rung"].items():
        before = original["effectiveness"]["by_rung"][rung]
        assert after["admissible"] == before["admissible"]
        # ...and equal to within an ulp on the numbers themselves. See the test
        # below for why that is not exact equality.
        for key in ("cell_margin_pp", "dataset_margin_pp"):
            assert after[key] == pytest.approx(before[key], abs=1e-9)
        assert after["macro_score"] == pytest.approx(before["macro_score"], abs=1e-12)
        assert after["macro_margin_pp"] == pytest.approx(before["macro_margin_pp"], abs=1e-9)

    for entry in report["substitution"]["changed"].values():
        assert entry["moved_pp"] == pytest.approx(0.0, abs=1e-12)


@needs_headlines
def test_the_substitution_perturbs_margins_by_an_ulp_and_that_is_all(
    three_identical_seeds: Path,
) -> None:
    """Averaging three copies of x is x mathematically, not bit-for-bit.

    ``fmean([x, x, x])`` can land one ulp off x, so even this degenerate
    substitution moves a margin by ~1e-13pp. Recorded rather than smoothed
    over, because the reason it does not matter is quantitative and worth
    stating: the smallest tolerance in the rule is 0.25pp, which is twelve
    orders of magnitude larger, and the two blocking margins are 0.748pp and
    2.258pp. Nothing at this scale can move an admissibility decision. If a
    verdict ever did turn on a difference this size, the verdict would be the
    thing to distrust.
    """

    report = resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)
    original = selection.build(resolution.HEADLINE_DIR)

    drift = max(
        abs(after[key][cell] - original["effectiveness"]["by_rung"][rung][key][cell])
        for rung, after in report["effectiveness_after"]["by_rung"].items()
        for key in ("cell_margin_pp", "dataset_margin_pp")
        for cell in after[key]
    )
    assert drift < 1e-9
    assert drift < 0.25 / 1e6


@needs_headlines
def test_only_the_named_cells_and_rungs_are_substituted(three_identical_seeds: Path) -> None:
    report = resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)

    assert sorted(report["substitution"]["changed"]) == [
        "musique_clean/R1/S3",
        "musique_clean/R1/S4",
        "squad_clean/R1/S3",
        "squad_clean/R1/S4",
    ]
    # 14 cells in the screen, 2 resolved, 12 left exactly as measured.
    assert len(report["substitution"]["cells_left_untouched"]) == 12
    assert "squad_clean/R1" not in report["substitution"]["cells_left_untouched"]
    assert report["substitution"]["thresholds_unchanged"] is True


@needs_headlines
def test_the_rule_that_judges_the_result_is_the_screens_own(
    three_identical_seeds: Path,
) -> None:
    """Not a reimplementation: the same module, called on substituted artifacts.

    If the resolution re-derived admissibility itself, a difference in verdict
    could be a difference in rule rather than in numbers, and the whole
    substitution design would prove nothing.
    """

    source = (REPO_ROOT / "scripts" / "m2b_resolution_report.py").read_text(encoding="utf-8")
    assert "from scripts import m2b_selection_report as selection" in source
    assert "resolved = selection.build(substituted_dir, declaration_path)" in source

    report = resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)
    tolerance = selection.build(resolution.HEADLINE_DIR)["rule"]["tolerance_pp"]
    assert tolerance == {"macro": 0.25, "per_dataset": 0.5, "per_cell": 0.5}
    assert report["systems"]["unchanged"] is True


@needs_headlines
def test_an_unmoved_verdict_maps_to_the_incumbent(three_identical_seeds: Path) -> None:
    """Identical seeds cannot admit S4, so the filed fallback branch is taken."""

    report = resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)
    assert report["outcome"]["status"] == "SELECTED_S3"
    assert report["outcome"]["selected_total_parameters"] == 3585
    assert "STOP_FOR_REVIEW" in report["outcome"]["then"]


@needs_headlines
def test_a_missing_resolution_fit_is_named_rather_than_skipped(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="has not been fitted"):
        resolution.build(resolution.HEADLINE_DIR, tmp_path / "absent")


@needs_headlines
def test_a_panel_that_moved_with_the_seed_is_refused(three_identical_seeds: Path) -> None:
    """The design is paired. A seed-dependent panel would silently break it."""

    path = three_identical_seeds / "seed1" / "squad_clean.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["cells"]["R1"]["held_out_query_ids"][0] = "a-query-from-somewhere-else"
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit, match="paired design requires one panel"):
        resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)


@needs_headlines
def test_an_aggregate_that_is_not_its_rows_is_refused(three_identical_seeds: Path) -> None:
    path = three_identical_seeds / "seed1" / "squad_clean.json"
    artifact = json.loads(path.read_text(encoding="utf-8"))
    artifact["cells"]["R1"]["rungs"]["S4"]["metrics"]["recall@5"] = 0.999
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(SystemExit, match="per-query rows average"):
        resolution.build(resolution.HEADLINE_DIR, three_identical_seeds)
