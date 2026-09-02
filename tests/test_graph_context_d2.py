"""D2 is a surgical ablation, and these tests are what makes "surgical" checkable.

The whole value of the stage is that exactly one thing differs between the two
arms. Every confound the historical `seed_only -> sa_mlp` contrast carries --
different head width, different parameter count, an added seed indicator, two
structural blocks moving at once -- is a confound this stage must not reproduce.
So the tests check the *sameness* at least as hard as they check the difference:
identical parameter counts, identical architecture, one shared feature build,
and a `FULL_CAND` arm that has to reproduce D1's `CAND` arm rather than merely
resemble it.
"""

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
# `tests` is not a package here, so D1's fixtures are reached by path rather than
# by `tests.`. Reusing them is the point: D2's synthetic dataset must be D1's, or
# the FULL_CAND control is comparing against a different world.
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import CompleteRetrievalDataset  # noqa: E402
from scripts.run_graph_context_d2 import (  # noqa: E402
    COMPLETE_STATUS,
    CONTEXT,
    D2_ARMS,
    build_parser,
    run,
    zeroed_local,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from test_graph_context_d1 import args_for, dataset as build_dataset  # noqa: E402

METRICS = ("recall@1", "recall@5", "recall@20", "mrr")


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


NODES = 48


def d2_args(tmp_path: Path, data, **overrides) -> argparse.Namespace:
    """D1's synthetic arguments, retargeted at D2's parser and output name."""

    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d2.json")
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@contextmanager
def synthetic(data):
    """Drive the real run() against the in-memory dataset, as D1's tests do."""

    import scripts.run_graph_context_d2 as module

    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    original = module.load_complete_dataset, module.load_or_build_static
    module.load_complete_dataset = lambda *a, **k: data
    module.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        yield module
    finally:
        module.load_complete_dataset, module.load_or_build_static = original


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data):
    """One real end-to-end run, driving the frozen model path on both arms."""

    with synthetic(data):
        return run(d2_args(tmp_path_factory.mktemp("d2"), data))


# --- the ablation itself --------------------------------------------------------


def test_zeroing_preserves_shape_and_dtype(data):
    """The head width follows the input dimension, so the shape has to survive."""

    block = np.random.default_rng(0).normal(size=(97, 10)).astype(np.float16)
    zeroed = zeroed_local(block)
    assert zeroed.shape == block.shape
    assert zeroed.dtype == block.dtype
    assert not zeroed.any()
    assert block.any(), "the fixture block was already zero, so this proved nothing"


def test_the_ablation_does_not_change_the_parameter_count(completed):
    """The confound that separates seed_only from sa_mlp. It must not recur here."""

    counts = {arm: completed["results"][arm]["parameters"] for arm, _ in D2_ARMS}
    assert len(set(counts.values())) == 1, counts
    assert completed["ablation"]["architecture_changed"] is False


def test_the_run_refuses_an_ablation_that_changed_the_parameter_count(
    tmp_path, data, monkeypatch
):
    """The guard is real, not decorative."""

    import scripts.run_graph_context_d2 as module

    real = module._build_model
    calls = {"n": 0}

    def widen(name, dataset, features, selected_gnn, target, args):
        model = real(name, dataset, features, selected_gnn, target, args)
        calls["n"] += 1
        if calls["n"] == 2:
            model.register_parameter("spurious", torch.nn.Parameter(torch.zeros(3)))
        return model

    with synthetic(data):
        monkeypatch.setattr(module, "_build_model", widen)
        with pytest.raises(RuntimeError, match="changed the parameter count"):
            run(d2_args(tmp_path, data))


def test_both_arms_share_one_feature_build(completed):
    """Two builds is two chances to differ by accident, and twice the price."""

    assert completed["feature_build"]["shared_by_both_arms"] is True
    assert completed["feature_build"]["candidate_rows"] > 0


def test_the_ablated_arm_records_that_its_weights_are_dead_but_trainable(completed):
    assert completed["ablation"]["dead_but_trainable_weights"] is True
    assert "zero gradient" in completed["ablation"]["dead_weight_note"]
    assert completed["results"]["NO_QUERY_LOCAL"]["query_local_block"] == "zeroed"
    assert completed["results"]["FULL_CAND"]["query_local_block"] == "present"


def test_zeroed_columns_cannot_reach_the_score(data):
    """The claim the dead-weight note rests on, checked on the actual module.

    A weight multiplying an exactly-zero input contributes exactly zero however
    it is initialised or decayed, so two models differing only in those weights
    must score identically. If some normalisation or bias path let the columns
    matter, this fails.
    """

    from mp_retrieval.operator_models import build_explicit_feature_mlp

    torch.manual_seed(0)
    model = build_explicit_feature_mlp(
        "sa_mlp", data.feature_dim, 8, static_dim=7, local_dim=10,
        target_parameters=5000, dropout=0.0, temperature=0.07,
    )
    model.eval()
    nodes = torch.randn(11, data.feature_dim)
    queries = torch.randn(2, data.feature_dim)
    batch_index = torch.tensor([0] * 6 + [1] * 5)
    static = torch.randn(11, 7)
    zeros = torch.zeros(11, 10)

    with torch.no_grad():
        before = model.forward_explicit(
            nodes, queries, batch_index, torch.cat([static, zeros], dim=1)
        )
        for name, parameter in model.named_parameters():
            if name == "scorer.0.weight":
                parameter[:, -10:] += torch.randn_like(parameter[:, -10:]) * 10.0
        after = model.forward_explicit(
            nodes, queries, batch_index, torch.cat([static, zeros], dim=1)
        )
    assert torch.equal(before, after)


# --- the arms differ, and differ only here ---------------------------------------


def test_the_two_arms_are_not_the_same_experiment(completed):
    """A vacuity check: if the ablation did nothing, the stage measured nothing."""

    full = completed["results"]["FULL_CAND"]["validation"]
    ablated = completed["results"]["NO_QUERY_LOCAL"]["validation"]
    assert full != ablated


def test_both_arms_run_the_historical_context(completed):
    """D2 is about the block, not the context. The closed branch stays closed."""

    assert completed["context"] == CONTEXT == "CAND"
    assert completed["contract"]["context_restored"] is False


def test_the_delta_is_reported_against_the_full_model(completed):
    delta = completed["delta_against_full"]["NO_QUERY_LOCAL"]
    for metric, value in delta.items():
        expected = (
            completed["results"]["NO_QUERY_LOCAL"]["validation"][metric]
            - completed["results"]["FULL_CAND"]["validation"][metric]
        )
        assert value == pytest.approx(expected)


def test_the_stratified_delta_matches_the_arms(completed):
    for name, row in completed["delta_against_full_by_stratum"]["NO_QUERY_LOCAL"].items():
        for metric in METRICS:
            if metric not in row:
                continue
            expected = (
                completed["results"]["NO_QUERY_LOCAL"]["validation_by_stratum"][name][metric]
                - completed["results"]["FULL_CAND"]["validation_by_stratum"][name][metric]
            )
            assert row[metric] == pytest.approx(expected), (name, metric)


def test_the_strata_reconstitute_the_overall_metric(completed):
    for arm, _ in D2_ARMS:
        by_stratum = completed["results"][arm]["validation_by_stratum"]
        total = sum(block["queries"] for block in by_stratum.values())
        for metric in METRICS:
            pooled = (
                sum(block["queries"] * block[metric] for block in by_stratum.values()) / total
            )
            assert pooled == pytest.approx(
                completed["results"][arm]["validation"][metric], abs=1e-9
            ), (arm, metric)


# --- the protocol ---------------------------------------------------------------


def test_the_test_split_is_refused(tmp_path, data):
    args = d2_args(tmp_path, data)
    args.splits = ["train", "validation", "test"]
    with pytest.raises(ValueError, match="test split is not read"):
        run(args)


@pytest.mark.parametrize("splits", [["train"], ["validation"], ["train", "train"]])
def test_only_the_two_development_splits_are_accepted(tmp_path, data, splits):
    args = d2_args(tmp_path, data)
    args.splits = splits
    with pytest.raises(ValueError, match="both and only both"):
        run(args)


def test_the_reported_split_is_never_handed_to_the_fitter(tmp_path, data, monkeypatch):
    """Epoch selection reads a tail of train; validation is read once, at the end."""

    import scripts.run_graph_context_d2 as module

    seen: list[set[int]] = []
    real = module._fit

    def spy(name, model, train, holdout, *args, **kwargs):
        seen.append({int(query.query_index) for query in train + holdout})
        return real(name, model, train, holdout, *args, **kwargs)

    with synthetic(data):
        monkeypatch.setattr(module, "_fit", spy)
        run(d2_args(tmp_path, data))

    validation = {int(query.query_index) for query in data.split(SPLITS["validation"])}
    assert seen and all(not (fitted & validation) for fitted in seen)


def test_the_declared_contract_says_no_gnn_and_no_message_passing(completed):
    assert completed["contract"]["gnn_trained"] is False
    assert completed["contract"]["message_passing"] is False
    assert completed["contract"]["test_split_read"] is False
    assert completed["contract"]["epoch_selected_on_validation"] is False
    assert completed["contract"]["hyperparameters_searched"] is False
    assert completed["splits"]["test_read"] is False


def test_the_run_completes_and_names_what_it_asked(completed):
    assert completed["status"] == COMPLETE_STATUS
    assert "incremental value" in completed["question"]
    assert set(completed["results"]) == {name for name, _ in D2_ARMS}
    assert completed["training"]["validation_reads_per_arm"] == 1


def test_the_parser_defaults_are_the_frozen_qls_v1_hyperparameters():
    """D2 and D1 must train identically, so they must not disagree on the values.

    Checked against D1's own parser rather than against numbers typed twice: if
    the frozen configuration ever moves, this fails instead of silently letting
    the two stages drift apart.
    """

    from scripts.run_graph_context_d1 import build_parser as d1_parser

    shared = ("epochs", "batch_size", "learning_rate", "weight_decay",
              "hidden_dim", "projection_dim", "dropout", "temperature")
    ours = {action.dest: action.default for action in build_parser()._actions}
    theirs = {action.dest: action.default for action in d1_parser()._actions}
    for name in shared:
        assert ours[name] == theirs[name], name
    assert (ours["epochs"], ours["batch_size"], ours["learning_rate"]) == (3, 16, 0.001)
    assert (ours["projection_dim"], ours["dropout"]) == (64, 0.2)


def test_the_arm_names_describe_what_is_removed():
    """Not `RETRIEVAL_ONLY`: this model keeps the static block and the embeddings."""

    names = {name for name, _ in D2_ARMS}
    assert names == {"FULL_CAND", "NO_QUERY_LOCAL"}
    assert "RETRIEVAL_ONLY" not in names
