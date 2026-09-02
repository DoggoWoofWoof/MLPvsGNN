"""The analyzer must hold up on the matrix it will actually be handed.

Every other check drives ``compile_analysis`` through a reduced config -- one
dataset, one axis, two rates. That is the right shape for testing refusals, and
it leaves two things unexercised that only exist at full scale.

Holm across a single dataset is the identity, so a correction applied to the
wrong grouping, or not applied at all, looks correct there and only starts
adjusting at six datasets. And the per-cell bootstrap seed is built from three
positional indices, so whether any two of the 120 cells can land on the same
seed is a question a two-cell config cannot ask.

The numbers below are synthetic. Nothing here reports, predicts, or depends on
an E2 outcome; only the shape of the compiled result is checked.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import scripts.analyze_phase_confirmation as apc
from scripts.analyze_phase_confirmation import (
    CLEAN_RATE,
    METRICS,
    MODEL_NAMES,
    PRIMARY_METRIC,
    _crossings,
    compile_analysis,
    render_markdown,
)

CONFIG = yaml.safe_load(
    (Path(apc.REPO_ROOT) / "configs" / "phase_confirmation.yaml").read_text(
        encoding="utf-8"
    )
)
DATASETS = sorted(CONFIG["datasets"])
AXES = sorted(CONFIG["axes"])
SEEDS = list(map(int, CONFIG["training"]["seeds"]))
ORDER = "f" * 64
QUERIES = 4


def _write_npz(path: Path, gap: float, rng: np.random.Generator) -> str:
    arrays = {}
    for index, model in enumerate(MODEL_NAMES):
        for seed in SEEDS:
            block = 0.4 + index * gap + rng.normal(0.0, 0.002, size=(QUERIES, len(METRICS)))
            arrays[f"{model}_seed_{seed}"] = block.astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metric_names=np.asarray(METRICS),
        query_order_sha256=np.asarray(ORDER),
        **arrays,
    )
    return apc._sha256(path)


def _models(gap: float, rng: np.random.Generator) -> dict:
    return {
        model: {
            "seeds": {
                str(seed): {
                    "metrics": {
                        metric: float(0.4 + index * gap + rng.normal(0.0, 0.002))
                        for metric in METRICS
                    }
                }
                for seed in SEEDS
            }
        }
        for index, model in enumerate(MODEL_NAMES)
    }


@pytest.fixture(scope="module")
def full_matrix(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """All 120 cells the registered config demands, built once."""
    tmp = tmp_path_factory.mktemp("full_matrix")
    root, sealed = tmp / "phase_confirmation", tmp / "sa_mlp_confirmation"
    sealed.mkdir(parents=True)
    rng = np.random.default_rng(20260902)

    for position, dataset in enumerate(DATASETS):
        # A different true gap per dataset, so the six p-values Holm sees in
        # each group are spread rather than identical.
        gap = (position - 2.5) * 0.01
        sha = _write_npz(sealed / f"{dataset}.query_metrics.npz", gap, rng)
        (sealed / f"{dataset}.json").write_text(
            json.dumps(
                {
                    "status": apc.SEALED_COMPLETE,
                    "dataset": dataset,
                    "data": {"test_query_order_sha256": ORDER},
                    "query_metrics": {"sha256": sha},
                    "models": _models(gap, rng),
                }
            ),
            encoding="utf-8",
        )
        for axis in AXES:
            for rate in sorted(map(float, CONFIG["axes"][axis]["rates"])):
                if rate == CLEAN_RATE:
                    continue
                cell = root / dataset / axis / f"rate_{apc._rate_key(rate)}"
                # Let the gap drift with rate so a sign change exists somewhere.
                cell_gap = gap + (rate - 0.5) * 0.02
                cell_sha = _write_npz(cell / "query_metrics.npz", cell_gap, rng)
                (cell / "result.json").write_text(
                    json.dumps(
                        {
                            "status": apc.CELL_COMPLETE,
                            "dataset": dataset,
                            "axis": axis,
                            "rate": rate,
                            "data": {"test_query_order_sha256": ORDER},
                            "query_metrics": {"sha256": cell_sha},
                            "confirmation_contract": {
                                "test_selected_rate": False,
                                "selected_by_locked_validation_only_rule": True,
                                "seed_zero_validation_checkpoint_reused_without_test_peeking": True,
                            },
                            "models": _models(cell_gap, rng),
                        }
                    ),
                    encoding="utf-8",
                )
    return root, sealed


@pytest.fixture(scope="module")
def analysis(full_matrix: tuple[Path, Path]) -> dict:
    root, sealed = full_matrix
    return compile_analysis(root, sealed_root=sealed, bootstrap_replicates=8)


def test_the_registered_matrix_is_one_hundred_and_twenty_cells(analysis: dict) -> None:
    expected = sum(len(CONFIG["axes"][axis]["rates"]) for axis in AXES) * len(DATASETS)
    assert len(analysis["rows"]) == expected == 120
    keys = {(row["dataset"], row["axis"], row["rate"]) for row in analysis["rows"]}
    assert len(keys) == expected, "a cell was compiled twice"


def test_every_axis_shares_one_clean_origin_per_dataset(analysis: dict) -> None:
    clean = [row for row in analysis["rows"] if row["is_clean_origin"]]
    assert len(clean) == len(DATASETS) * len(AXES) == 24
    # The clean cell is read from the sealed confirmation, so within a dataset
    # all four axes must carry byte-identical clean numbers -- retraining it per
    # axis is exactly what _cell_source exists to prevent.
    for dataset in DATASETS:
        values = {
            json.dumps(row["models"], sort_keys=True)
            for row in clean
            if row["dataset"] == dataset
        }
        assert len(values) == 1, f"{dataset} has more than one clean origin"


def test_holm_is_applied_across_all_six_datasets_in_each_group(analysis: dict) -> None:
    groups: dict[tuple[str, float], list[dict]] = {}
    for row in analysis["rows"]:
        groups.setdefault((row["axis"], row["rate"]), []).append(row)
    assert len(groups) == sum(len(CONFIG["axes"][axis]["rates"]) for axis in AXES)
    for (axis, rate), rows in groups.items():
        assert len(rows) == len(DATASETS), f"{axis}/{rate} spans {len(rows)} datasets"
        block = [row["seed_aware_gnn_minus_sa_mlp"][PRIMARY_METRIC] for row in rows]
        raw = [entry["paired_seed_t_pvalue"] for entry in block]
        adjusted = [entry["holm_pvalue"] for entry in block]
        for r, a in zip(raw, adjusted):
            assert a >= r - 1e-12, f"{axis}/{rate}: Holm lowered a p-value"
            assert 0.0 <= a <= 1.0
        # With six spread p-values the correction has to bite somewhere; if it
        # never does, the grouping collapsed to one test per group.
        assert any(a > r + 1e-12 for r, a in zip(raw, adjusted)), (
            f"{axis}/{rate}: Holm changed nothing across six datasets"
        )


def test_no_two_cells_share_a_bootstrap_seed() -> None:
    base = 20260901
    widest = max(len(CONFIG["axes"][axis]["rates"]) for axis in AXES)
    seeds = [
        base + d * 1009 + a * 101 + r
        for d in range(len(DATASETS))
        for a in range(len(AXES))
        for r in range(widest)
    ]
    assert len(set(seeds)) == len(seeds), (
        "two cells derive the same bootstrap seed, so their paired CIs would "
        "share a resampling draw"
    )


def test_crossings_on_a_full_rate_ladder_stay_adjacent_and_real(analysis: dict) -> None:
    """A five-rate ladder gives four intervals per dataset-axis to get wrong.

    The reduced config has one interval, so it cannot catch a crossing reported
    between rates that are not neighbours -- which is what a mis-sorted series
    would produce, and which would read as a phase transition spanning the
    whole ladder.
    """
    rows = analysis["rows"]
    for crossing in _crossings(analysis):
        rates = sorted(
            row["rate"] for row in rows
            if row["dataset"] == crossing["dataset"] and row["axis"] == crossing["axis"]
        )
        left, right = crossing["interval"]
        assert left in rates and right in rates
        assert rates.index(left) + 1 == rates.index(right), (
            "a crossing was reported between non-adjacent rates"
        )
        # The interval is only a crossing if the gap actually changes sign.
        assert crossing["gap_before"] * crossing["gap_after"] < 0

        holm = {
            row["rate"]: row["seed_aware_gnn_minus_sa_mlp"][PRIMARY_METRIC]["holm_pvalue"]
            for row in rows
            if row["dataset"] == crossing["dataset"] and row["axis"] == crossing["axis"]
        }
        assert crossing["both_endpoints_holm_significant"] == (
            holm[left] < 0.05 and holm[right] < 0.05
        ), "a crossing was called significant on something other than both endpoints"


def test_the_markdown_renders_every_dataset_at_full_size(analysis: dict) -> None:
    markdown = render_markdown(analysis)
    missing = [dataset for dataset in DATASETS if dataset not in markdown]
    assert not missing, "absent from the rendered report: " + ", ".join(missing)
    assert markdown.count("|") > 120, "the rendered report carries no full table"


def test_the_synthetic_ladder_actually_produces_crossings(analysis: dict) -> None:
    """Guard against the check above passing on an empty list.

    ``test_crossings_on_a_full_rate_ladder_stay_adjacent_and_real`` iterates the
    crossings, so it is vacuous if the fixture never generates one. The gaps in
    this fixture are built to change sign; if that stops being true the fixture
    has drifted and the adjacency check is no longer testing anything.
    """
    assert _crossings(analysis), (
        "the synthetic matrix produced no crossings at all, so the adjacency "
        "check above is vacuous"
    )
