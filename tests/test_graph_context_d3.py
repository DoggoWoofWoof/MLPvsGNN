"""D3 is a four-point ladder whose value depends entirely on what it holds fixed.

Two things can silently invalidate it. The masks could fail to mean what their
names say -- `SEED_ID_ONLY` leaking geometry is the exact error the D3 design
was written to avoid. And the two reused D2 arms could be reused across a
condition that actually differs, turning a decomposition into a comparison of
two experiments. Both are tested here harder than the arithmetic is.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import CompleteRetrievalDataset  # noqa: E402
from mp_retrieval.linear_control import LOCAL_FEATURE_NAMES  # noqa: E402
from scripts.run_graph_context_d3 import (  # noqa: E402
    ALL_COLUMNS,
    COMPLETE_STATUS,
    CONTEXT,
    D3_ARMS,
    DISTANCE_COLUMNS,
    INCREMENTS,
    REUSED_FROM_D2,
    SEED_ID_COLUMN,
    assert_seed_identity_column,
    build_parser,
    masked_local,
    run,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from test_graph_context_d1 import args_for, dataset as build_dataset  # noqa: E402

METRICS = ("recall@1", "recall@5", "recall@20", "mrr")
NODES = 48


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


@contextmanager
def synthetic(data):
    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as module

    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    patched = (module, d2_module)
    original = [(m.load_complete_dataset, m.load_or_build_static) for m in patched]
    for m in patched:
        m.load_complete_dataset = lambda *a, **k: data
        m.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        yield module
    finally:
        for m, (loader, builder) in zip(patched, original, strict=True):
            m.load_complete_dataset, m.load_or_build_static = loader, builder


def d3_args(tmp_path: Path, data, d2_result: Path, **overrides) -> argparse.Namespace:
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d3.json")
    parsed["d2_result"] = d2_result
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d2_result(tmp_path_factory, data) -> Path:
    """A real D2 run, so the reuse path is exercised against genuine output."""

    import scripts.run_graph_context_d2 as d2_module
    from test_graph_context_d2 import d2_args

    with synthetic(data):
        args = d2_args(tmp_path_factory.mktemp("d3_d2"), data)
        d2_module.run(args)
    return Path(args.output)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d2_result):
    with synthetic(data):
        return run(d3_args(tmp_path_factory.mktemp("d3"), data, d2_result))


# --- the masks mean what they are named -----------------------------------------


def test_the_arms_are_the_declared_ladder():
    assert [name for name, _ in D3_ARMS] == [
        "ZERO_LOCAL", "SEED_ID_ONLY", "DISTANCE_ONLY", "FULL_LOCAL"
    ]
    keeps = dict(D3_ARMS)
    assert keeps["ZERO_LOCAL"] == ()
    assert keeps["SEED_ID_ONLY"] == (0,)
    assert keeps["DISTANCE_ONLY"] == (0, 1, 2, 3)
    assert keeps["FULL_LOCAL"] == ALL_COLUMNS


def test_seed_id_only_is_one_column_and_it_is_the_membership_column():
    """The correction this stage exists for: it must not expose distance buckets."""

    keep = dict(D3_ARMS)["SEED_ID_ONLY"]
    assert len(keep) == 1
    assert LOCAL_FEATURE_NAMES[keep[0]] == "distance_0"
    assert set(keep) < set(DISTANCE_COLUMNS), "SEED_ID_ONLY must be strictly inside the group"
    for name in ("distance_1", "distance_2", "distance_3_plus_or_unreachable"):
        assert LOCAL_FEATURE_NAMES.index(name) not in keep


def test_distance_only_keeps_the_whole_bucket_group_and_nothing_else():
    keep = dict(D3_ARMS)["DISTANCE_ONLY"]
    assert tuple(LOCAL_FEATURE_NAMES[i] for i in keep) == LOCAL_FEATURE_NAMES[:4]
    for name in ("seed_connections", "personalized_pagerank", "paths_length_1"):
        assert LOCAL_FEATURE_NAMES.index(name) not in keep


@pytest.mark.parametrize("arm,keep", D3_ARMS)
def test_masking_zeroes_exactly_the_columns_outside_keep(arm, keep):
    block = np.random.default_rng(1).normal(size=(31, 10)).astype(np.float16) + 1.0
    masked = masked_local(block, keep)
    assert masked.shape == block.shape and masked.dtype == block.dtype
    for column in ALL_COLUMNS:
        if column in keep:
            assert np.array_equal(masked[:, column], block[:, column]), column
        else:
            assert not masked[:, column].any(), column


def test_full_local_masking_is_the_identity():
    block = np.random.default_rng(2).normal(size=(9, 10)).astype(np.float16)
    assert np.array_equal(masked_local(block, ALL_COLUMNS), block)


def test_the_seed_identity_assertion_rejects_a_column_that_is_not_the_indicator(data):
    """The guard is real: corrupt one row and the stage must refuse."""

    import scripts.run_graph_context_d3 as module
    from test_graph_context_d1 import local_block

    local, _ = local_block(data, "CAND")
    local = np.asarray(local).copy()
    assert module.assert_seed_identity_column(data.queries, local)["mismatches"] == 0
    local[:, SEED_ID_COLUMN] = 1.0 - local[:, SEED_ID_COLUMN]
    with pytest.raises(RuntimeError, match="not the frozen seed indicator"):
        module.assert_seed_identity_column(data.queries, local)


def test_the_run_proves_the_seed_column_on_its_own_queries(completed):
    proof = completed["seed_identity_proof"]
    assert proof["elementwise_identical"] is True
    assert proof["mismatches"] == 0
    assert proof["column"] == "distance_0"
    assert 0 < proof["seed_rows"] < proof["candidate_rows_compared"]
    assert completed["distance_group_proof"]["complete_one_hot"] is True


# --- the reuse ------------------------------------------------------------------


def test_only_the_two_new_arms_are_trained(completed):
    assert completed["arms_trained_here"] == ["SEED_ID_ONLY", "DISTANCE_ONLY"]
    for arm in ("ZERO_LOCAL", "FULL_LOCAL"):
        assert completed["results"][arm]["retrained_here"] is False
        assert completed["results"][arm]["measured_in"] == "stage_d2"


def test_the_reused_rows_are_d2s_rows_unchanged(completed, d2_result):
    d2 = json.loads(Path(d2_result).read_text(encoding="utf-8"))
    for ours, theirs in REUSED_FROM_D2.items():
        assert completed["results"][ours]["validation"] == d2["results"][theirs]["validation"]


def test_every_reuse_condition_is_recorded_and_matches(completed):
    reuse = completed["reuse"]
    assert reuse["all_conditions_match"] is True
    for name, block in reuse["conditions_checked"].items():
        assert block["match"] is True, name
    for expected in (
        "dataset", "data_fingerprint", "context", "model", "candidate_contract",
        "num_nodes", "candidate_rows", "architecture_changed",
        "split_train_fit", "split_validation_reported",
        "training_epochs", "training_seed", "training_learning_rate",
    ):
        assert expected in reuse["conditions_checked"], expected


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset", "somewhere_else"),
        # Not "0" * 64: that is the synthetic fixture's own fingerprint, so it
        # would corrupt nothing and the test would pass without the guard.
        ("data_fingerprint_sha256", "f" * 64),
        ("context", "TARGET_H1"),
        ("model", "plain_mlp"),
        ("num_nodes", 1),
    ],
)
def test_reuse_is_refused_when_a_condition_differs(tmp_path, data, d2_result, field, value):
    corrupted = json.loads(Path(d2_result).read_text(encoding="utf-8"))
    assert corrupted[field] != value, "the fixture already holds this value"
    corrupted[field] = value
    path = tmp_path / "stage_d2.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="cannot be reused"):
            run(d3_args(tmp_path, data, path))


@pytest.mark.parametrize("key", ["epochs", "seed", "learning_rate"])
def test_reuse_is_refused_when_the_training_recipe_differs(tmp_path, data, d2_result, key):
    """D3 decomposes D2 at D2's operating point; a different recipe is a different one."""

    corrupted = json.loads(Path(d2_result).read_text(encoding="utf-8"))
    corrupted["training"][key] = corrupted["training"][key] + 1
    path = tmp_path / "stage_d2.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="cannot be reused"):
            run(d3_args(tmp_path, data, path))


@pytest.mark.parametrize("key", ["train_fit", "validation_reported"])
def test_reuse_is_refused_when_the_splits_differ(tmp_path, data, d2_result, key):
    corrupted = json.loads(Path(d2_result).read_text(encoding="utf-8"))
    corrupted["splits"][key] = corrupted["splits"][key] + 1
    path = tmp_path / "stage_d2.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="cannot be reused"):
            run(d3_args(tmp_path, data, path))


def test_reuse_is_refused_when_d2_did_not_finish(tmp_path, data, d2_result):
    corrupted = json.loads(Path(d2_result).read_text(encoding="utf-8"))
    corrupted["status"] = "GRAPH_CONTEXT_D2_IN_PROGRESS"
    path = tmp_path / "stage_d2.json"
    path.write_text(json.dumps(corrupted), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="not complete"):
            run(d3_args(tmp_path, data, path))


def test_a_missing_d2_result_is_refused_rather_than_silently_retrained(tmp_path, data):
    with synthetic(data):
        with pytest.raises(FileNotFoundError, match="stage_d2|reuses D2"):
            run(d3_args(tmp_path, data, tmp_path / "absent.json"))


# --- the ladder -----------------------------------------------------------------


def test_the_ladder_has_all_four_arms_at_one_parameter_count(completed):
    counts = {arm: completed["results"][arm]["parameters"] for arm, _ in D3_ARMS}
    assert len(set(counts.values())) == 1, counts
    assert set(completed["ladder"]) == {name for name, _ in D3_ARMS}


def test_the_increments_are_the_differences_they_claim(completed):
    for name, upper, lower in INCREMENTS:
        block = completed["increments"][name]
        assert (block["from"], block["to"]) == (lower, upper)
        for metric in METRICS:
            expected = completed["ladder"][upper][metric] - completed["ladder"][lower][metric]
            assert block[metric] == pytest.approx(expected), (name, metric)


def test_the_increments_telescope_to_the_d2_result(completed):
    """seed + distance + remaining must equal FULL_LOCAL - ZERO_LOCAL exactly."""

    for metric in METRICS:
        total = sum(completed["increments"][name][metric] for name, _, _ in INCREMENTS)
        d2_gap = (
            completed["ladder"]["FULL_LOCAL"][metric]
            - completed["ladder"]["ZERO_LOCAL"][metric]
        )
        assert total == pytest.approx(d2_gap), metric


def test_the_stratified_increments_match_the_arms(completed):
    for name, upper, lower in INCREMENTS:
        for stratum, row in completed["increments_by_stratum"][name].items():
            for metric in METRICS:
                if metric not in row:
                    continue
                expected = (
                    completed["results"][upper]["validation_by_stratum"][stratum][metric]
                    - completed["results"][lower]["validation_by_stratum"][stratum][metric]
                )
                assert row[metric] == pytest.approx(expected), (name, stratum, metric)


def test_the_arms_are_not_all_the_same_experiment(completed):
    seen = [tuple(completed["ladder"][arm].items()) for arm, _ in D3_ARMS]
    assert len(set(seen)) > 1, "the ladder collapsed; the masks did nothing"


# --- the protocol ---------------------------------------------------------------


def test_the_test_split_is_refused(tmp_path, data, d2_result):
    args = d3_args(tmp_path, data, d2_result)
    args.splits = ["train", "validation", "test"]
    with pytest.raises(ValueError, match="test split is not read"):
        run(args)


@pytest.mark.parametrize("splits", [["train"], ["validation"], ["train", "train"]])
def test_only_the_two_development_splits_are_accepted(tmp_path, data, d2_result, splits):
    args = d3_args(tmp_path, data, d2_result)
    args.splits = splits
    with pytest.raises(ValueError, match="both and only both"):
        run(args)


def test_the_reported_split_is_never_handed_to_the_fitter(tmp_path, data, d2_result, monkeypatch):
    import scripts.run_graph_context_d3 as module

    seen: list[set[int]] = []
    real = module._fit

    def spy(name, model, train, holdout, *args, **kwargs):
        seen.append({int(query.query_index) for query in train + holdout})
        return real(name, model, train, holdout, *args, **kwargs)

    with synthetic(data):
        monkeypatch.setattr(module, "_fit", spy)
        run(d3_args(tmp_path, data, d2_result))

    validation = {int(query.query_index) for query in data.split(SPLITS["validation"])}
    assert seen and all(not (fitted & validation) for fitted in seen)
    assert len(seen) == 2, "only the two new arms should be fitted"


def test_the_contract_is_declared(completed):
    assert completed["contract"]["gnn_trained"] is False
    assert completed["contract"]["message_passing"] is False
    assert completed["contract"]["test_split_read"] is False
    assert completed["contract"]["epoch_selected_on_validation"] is False
    assert completed["contract"]["architecture_changed"] is False
    assert completed["context"] == CONTEXT == "CAND"
    assert completed["ablation"]["same_epoch_budget_for_every_arm"] is True


def test_the_run_completes_and_publishes_the_schema(completed):
    assert completed["status"] == COMPLETE_STATUS
    assert completed["local_feature_schema"] == list(LOCAL_FEATURE_NAMES)
    assert set(completed["increments"]) == {name for name, _, _ in INCREMENTS}


def test_the_parser_defaults_match_d2(completed=None):
    """D3 decomposes D2, so it must train at D2's operating point exactly."""

    from scripts.run_graph_context_d2 import build_parser as d2_parser

    shared = ("epochs", "batch_size", "learning_rate", "weight_decay",
              "hidden_dim", "projection_dim", "dropout", "temperature",
              "damping", "ppr_iterations", "holdout_fraction", "seed")
    ours = {action.dest: action.default for action in build_parser()._actions}
    theirs = {action.dest: action.default for action in d2_parser()._actions}
    for name in shared:
        assert ours[name] == theirs[name], name
