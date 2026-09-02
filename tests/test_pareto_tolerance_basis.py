"""The evidence behind the Pareto tolerance, re-derived from the frozen results.

``docs/EXPERIMENT_EXECUTION_STATUS.md`` proposes ``tau = 0.25`` R@5 points and is
explicit that the number was "chosen against two measured quantities, not for
convenience". It then states four figures: a seed-to-seed spread of 0.047-1.144,
a six-dataset mean-noise band of [0.224, 0.406], a smallest Holm-significant GNN
advantage of 0.531 points at Holm p = 0.0027, and the observation that tau is 47%
of that advantage.

A tolerance is exactly the kind of parameter that gets quietly widened later to
admit a result that missed, so its justification should not rest on four numbers
nobody can reproduce. Every one of them is recomputed here from the sealed
confirmation artifacts.

The band's endpoints were the one part the prose does not spell out. They are the
two extremes of how seed noise can combine across datasets: ``sqrt(sum sd^2)/6``
if the six datasets' seed noise is independent, and ``sum(sd)/6`` if it is
perfectly correlated. The same five seeds are shared by every dataset, so the
correlated end is the conservative one and the truth lies between.

These artifacts are the *clean-rate* confirmation, frozen and already reported.
Nothing here reads E2, whose outcomes must not influence any selection rule.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path

import pytest

stats = pytest.importorskip("scipy.stats")

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = REPO_ROOT / "outputs" / "sa_mlp_confirmation"
STATUS_DOC = REPO_ROOT / "docs" / "EXPERIMENT_EXECUTION_STATUS.md"

CONTRAST = "sa_mlp_minus_seed_aware_gnn"
METRIC = "recall@5"
DATASETS = 6
PROPOSED_TAU = 0.25

# The figures the status document states. Changing one there without re-deriving
# it here is what these tests exist to catch.
SEED_SD_MIN, SEED_SD_MAX = 0.047, 1.144
BAND_LOW, BAND_HIGH = 0.224, 0.406
SMALLEST_SIGNIFICANT = 0.531
SMALLEST_SIGNIFICANT_DATASET = "hotpotqa_clean"
SMALLEST_SIGNIFICANT_HOLM_P = 0.0027
TAU_AS_FRACTION_PERCENT = 47


def _payloads() -> dict[str, dict]:
    paths = sorted(CONFIRMATION.glob("*.json"))
    if len(paths) < DATASETS:
        pytest.skip(
            f"{CONFIRMATION} holds {len(paths)} of {DATASETS} sealed confirmation "
            "results; fetch them before re-deriving the tolerance basis."
        )
    return {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in paths}


def _seed_standard_deviations(payloads: dict[str, dict]) -> dict[str, float]:
    """QLS-v1 seed-to-seed std of R@5, in percentage points, per dataset."""
    out = {}
    for name, payload in payloads.items():
        seeds = payload["models"]["sa_mlp"]["seeds"]
        values = [seeds[key]["metrics"][METRIC] for key in sorted(seeds)]
        out[name] = statistics.stdev(values) * 100
    return out


def _gnn_advantages(payloads: dict[str, dict]) -> dict[str, tuple[float, float]]:
    """Per dataset: the GNN's R@5 lead in points, and the paired raw p-value."""
    out = {}
    for name, payload in payloads.items():
        contrast = payload["paired_contrasts"][CONTRAST][METRIC]
        differences = [contrast["by_seed"][k] for k in sorted(contrast["by_seed"], key=int)]
        # The stored contrast is sa_mlp minus GNN, so the GNN's lead is its negation.
        out[name] = (
            -contrast["mean"] * 100,
            float(stats.ttest_1samp(differences, 0.0).pvalue),
        )
    return out


def _holm(raw: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down, the correction the protocol already uses."""
    ordered = sorted(raw, key=lambda name: raw[name])
    adjusted, running = {}, 0.0
    for rank, name in enumerate(ordered):
        running = max(running, min(1.0, (len(raw) - rank) * raw[name]))
        adjusted[name] = running
    return adjusted


# --------------------------------------------------------------------------
# The first measured quantity: seed noise
# --------------------------------------------------------------------------


def test_the_seed_to_seed_spread_is_the_stated_range():
    deviations = _seed_standard_deviations(_payloads())
    assert min(deviations.values()) == pytest.approx(SEED_SD_MIN, abs=5e-4)
    assert max(deviations.values()) == pytest.approx(SEED_SD_MAX, abs=5e-4)


def test_the_quietest_and_noisiest_datasets_are_the_ones_that_set_the_range():
    """A range is only interpretable if its endpoints are attributable.

    metaqa is near-deterministic across seeds and webqsp is the smallest test
    split at 159 queries, so the spread is a property of the datasets rather
    than of an unstable training procedure.
    """
    deviations = _seed_standard_deviations(_payloads())
    assert min(deviations, key=deviations.get) == "metaqa"
    assert max(deviations, key=deviations.get) == "webqsp"


def test_the_mean_noise_band_spans_independent_to_perfectly_correlated_seeds():
    """The band is not an interval estimate; it is a pair of extremes.

    Neither end is a safe default on its own: assuming independence understates
    the noise of the six-dataset mean, and assuming perfect correlation
    overstates it. Quoting both is what makes tau defensible.
    """
    deviations = list(_seed_standard_deviations(_payloads()).values())
    independent = math.sqrt(sum(sd * sd for sd in deviations)) / DATASETS
    correlated = sum(deviations) / DATASETS

    assert independent == pytest.approx(BAND_LOW, abs=5e-4)
    assert correlated == pytest.approx(BAND_HIGH, abs=5e-4)
    assert independent < correlated


def test_the_proposed_tolerance_sits_just_inside_the_band():
    """tau below the band would be noise; far above it would admit real losses."""
    assert BAND_LOW < PROPOSED_TAU < BAND_HIGH
    assert PROPOSED_TAU - BAND_LOW < 0.05, "tau is meant to sit at the lower edge"


# --------------------------------------------------------------------------
# The second measured quantity: the smallest effect tau must not swallow
# --------------------------------------------------------------------------


def test_the_smallest_holm_significant_gnn_advantage_is_the_stated_effect():
    advantages = _gnn_advantages(_payloads())
    holm = _holm({name: p for name, (_, p) in advantages.items()})

    significant = {
        name: advantages[name][0]
        for name in advantages
        if holm[name] < 0.05 and advantages[name][0] > 0
    }
    assert significant, "the tolerance is calibrated against a real effect"

    smallest = min(significant, key=significant.get)
    assert smallest == SMALLEST_SIGNIFICANT_DATASET
    assert significant[smallest] == pytest.approx(SMALLEST_SIGNIFICANT, abs=5e-4)
    assert holm[smallest] == pytest.approx(SMALLEST_SIGNIFICANT_HOLM_P, abs=5e-5)


def test_the_tolerance_alone_cannot_hand_back_a_contested_effect():
    """The load-bearing property: tau must be a fraction of the smallest effect.

    If tau exceeded it, a variant could lose the one comparison the GNN
    demonstrably wins and still be admitted, which would make the frontier's
    conclusion an artifact of the tolerance.
    """
    assert PROPOSED_TAU < SMALLEST_SIGNIFICANT
    fraction = PROPOSED_TAU / SMALLEST_SIGNIFICANT * 100
    assert round(fraction) == TAU_AS_FRACTION_PERCENT


def test_only_two_datasets_show_a_holm_significant_gnn_advantage():
    """Context the tolerance is read against, and a negative result worth keeping.

    On four of six datasets the GNN's R@5 lead does not survive correction, and
    on webqsp the sign is against it. That is the honest backdrop for a
    no-message-passing thesis and must not quietly become "the GNN wins".
    """
    advantages = _gnn_advantages(_payloads())
    holm = _holm({name: p for name, (_, p) in advantages.items()})

    significant = {name for name in holm if holm[name] < 0.05}
    assert significant == {"2wiki_clean", "hotpotqa_clean"}
    assert advantages["webqsp"][0] < 0, "webqsp favours the fixed-feature model"


# --------------------------------------------------------------------------
# The document and the derivation must not drift apart
# --------------------------------------------------------------------------


def test_the_status_document_still_states_the_figures_derived_here():
    """Editing a figure in the prose without re-deriving it fails here.

    The tolerance's whole defence is that its numbers are measured. A tolerance
    quietly relaxed after seeing a result it excluded is the failure this guards
    against, so prose and derivation are pinned together.
    """
    if not STATUS_DOC.exists():
        pytest.skip(f"{STATUS_DOC} is absent")
    prose = STATUS_DOC.read_text(encoding="utf-8", errors="replace")

    for figure in (
        "0.047-1.144",
        "[0.224, 0.406]",
        "0.531",
        "0.0027",
        "47%",
        "tau = 0.25",
    ):
        assert figure in prose, f"the status document no longer states {figure!r}"
