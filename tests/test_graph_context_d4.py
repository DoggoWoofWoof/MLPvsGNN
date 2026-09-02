"""D4 measures one increment, and three things could quietly make it the wrong one.

The seed bit could stop being preserved, in which case
`delta_retrieval_quality` would be measuring "graded evidence *instead of* the
bit" rather than "on top of it" -- the exact confound the stage was redesigned
to avoid. The retrieval prior could turn out to be a seed-quality score rather
than a prior over all of `Cq`, which would make definition B a label instead of
a fact. And the reused D3 control could be reused across a condition that
actually differs, turning a causal increment into a comparison of two
experiments.

Those three are tested harder than the arithmetic is.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import CompleteRetrievalDataset  # noqa: E402
from mp_retrieval.graph_context import NORMALISED_COLUMNS  # noqa: E402
from mp_retrieval.linear_control import (  # noqa: E402
    LOCAL_FEATURE_NAMES,
    RANK_FEATURE_NAMES,
    rank_feature_rows,
)
from scripts.run_graph_context_d3 import SEED_ID_COLUMN, masked_local  # noqa: E402
from scripts.run_graph_context_d4 import (  # noqa: E402
    AGREEMENT_COLUMN,
    COLUMN_SEMANTICS,
    COMPLETE_STATUS,
    CONTEXT,
    D4_ARMS,
    DENSE_RANK_COLUMN,
    NEW_ARM,
    RETRIEVAL_COLUMNS,
    REUSED_FROM_D3,
    SPLADE_RANK_COLUMN,
    a3_rank_feature_audit,
    assert_seed_bit_is_shared,
    build_graded_retrieval_block,
    build_parser,
    run,
)
from scripts.run_graph_context_pilot import SPLITS  # noqa: E402
from test_graph_context_d1 import args_for, dataset as build_dataset, local_block  # noqa: E402

METRICS = ("recall@1", "recall@5", "recall@20", "mrr")
NODES = 48
RRF_CONSTANT = 60

#: The synthetic pools hold ten candidates. Splitting them 0-6 and 3-9 gives a
#: list every candidate appears in, an overlap that is neither empty nor total,
#: and candidates found by exactly one retriever -- the three cases the prior has
#: to represent.
DENSE_SLICE = slice(0, 7)
SPLADE_SLICE = slice(3, 10)


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return build_dataset()


def write_rank_lists(root: Path, data: CompleteRetrievalDataset) -> tuple[Path, Path]:
    """Frozen top-k lists whose union is exactly the frozen candidate pool."""

    root.mkdir(parents=True, exist_ok=True)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    dense_path = root / "dense_top200_all.npy"
    splade_path = root / "splade_top200_all.npy"
    np.save(dense_path, dense.astype(np.int64))
    np.save(splade_path, splade.astype(np.int64))
    return dense_path, splade_path


@contextmanager
def synthetic(data):
    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    import scripts.run_graph_context_d4 as module

    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    patched = (module, d3_module, d2_module)
    original = [(m.load_complete_dataset, m.load_or_build_static) for m in patched]
    for m in patched:
        m.load_complete_dataset = lambda *a, **k: data
        m.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        yield module
    finally:
        for m, (loader, builder) in zip(patched, original, strict=True):
            m.load_complete_dataset, m.load_or_build_static = loader, builder


def d4_args(tmp_path: Path, data, d3_result: Path, **overrides) -> argparse.Namespace:
    write_rank_lists(tmp_path, data)
    parsed = vars(args_for(tmp_path, data, **overrides))
    parsed["output"] = Path(parsed["output"]).with_name("stage_d4.json")
    parsed["d3_result"] = d3_result
    parsed["rrf_constant"] = RRF_CONSTANT
    known = {action.dest for action in build_parser()._actions}
    return argparse.Namespace(**{k: v for k, v in parsed.items() if k in known})


@pytest.fixture(scope="module")
def d3_result(tmp_path_factory, data) -> Path:
    """A real D2 then a real D3, so D4's reuse path meets genuine output."""

    import scripts.run_graph_context_d2 as d2_module
    import scripts.run_graph_context_d3 as d3_module
    from test_graph_context_d2 import d2_args
    from test_graph_context_d3 import d3_args

    with synthetic(data):
        two = d2_args(tmp_path_factory.mktemp("d4_d2"), data)
        d2_module.run(two)
        three = d3_args(tmp_path_factory.mktemp("d4_d3"), data, Path(two.output))
        d3_module.run(three)
    return Path(three.output)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data, d3_result):
    with synthetic(data):
        return run(d4_args(tmp_path_factory.mktemp("d4"), data, d3_result))


@pytest.fixture(scope="module")
def blocks(data):
    """The two arms' local blocks, built exactly as the runner builds them."""

    local, ptr = local_block(data, CONTEXT)
    local = np.asarray(local)
    dense = np.stack([q.candidate_index.numpy()[DENSE_SLICE] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[SPLADE_SLICE] for q in data.queries])
    treatment, prior = build_graded_retrieval_block(
        data.queries, local, ptr, dense, splade, constant=RRF_CONSTANT
    )
    return local, ptr, masked_local(local, (SEED_ID_COLUMN,)), treatment, prior


# --- the correction: the bit is kept, not replaced -------------------------------


def test_the_two_arms_are_the_declared_pair():
    assert D4_ARMS == ("SEED_ID_ONLY", "SEED_PLUS_GRADED_RETRIEVAL")
    assert NEW_ARM == "SEED_PLUS_GRADED_RETRIEVAL"
    assert REUSED_FROM_D3 == {"SEED_ID_ONLY": "SEED_ID_ONLY"}


def test_the_seed_bit_is_identical_in_both_arms(blocks):
    """Replacing the bit would confound two questions. It is preserved instead."""

    local, _ptr, control, treatment, _prior = blocks
    assert np.array_equal(control[:, SEED_ID_COLUMN], local[:, SEED_ID_COLUMN])
    assert np.array_equal(treatment[:, SEED_ID_COLUMN], local[:, SEED_ID_COLUMN])
    assert control[:, SEED_ID_COLUMN].any(), "no seeds, so this proved nothing"


def test_the_stage_records_that_the_bit_was_not_replaced(completed):
    assert completed["ablation"]["seed_bit_replaced"] is False
    assert completed["seed_bit_preserved"]["seed_bit_identical_across_arms"] is True
    assert completed["seed_bit_preserved"]["control_carries_only_the_bit"] is True
    assert completed["seed_bit_preserved"]["seed_rows"] > 0


def test_a_perturbed_bit_is_refused(blocks):
    local, _ptr, control, treatment, _prior = blocks
    broken = treatment.copy()
    row = int(np.flatnonzero(control[:, SEED_ID_COLUMN])[0])
    broken[row, SEED_ID_COLUMN] = 0
    with pytest.raises(RuntimeError, match="seed bit is not identical"):
        assert_seed_bit_is_shared(control, broken)


def test_a_control_carrying_more_than_the_bit_is_refused(blocks):
    local, _ptr, control, treatment, _prior = blocks
    leaky = control.copy()
    leaky[:, 4] = local[:, 4]
    with pytest.raises(RuntimeError, match="more than the bare seed bit"):
        assert_seed_bit_is_shared(leaky, treatment)


def test_a_treatment_carrying_structure_outside_its_slots_is_refused(blocks):
    local, _ptr, control, treatment, _prior = blocks
    leaky = treatment.copy()
    leaky[:, 8] = local[:, 8]
    with pytest.raises(RuntimeError, match="outside its declared slots"):
        assert_seed_bit_is_shared(control, leaky)


# --- definition B: a prior over Cq, not a seed-quality score ---------------------


def test_the_prior_is_declared_and_named_as_definition_b(completed):
    prior = completed["graded_retrieval_prior"]
    assert prior["definition"].startswith("B,")
    assert prior["not_called_seed_quality"] is True
    assert NEW_ARM == "SEED_PLUS_GRADED_RETRIEVAL"
    assert "seed_quality" not in json.dumps(completed["arms"])


def test_every_candidate_carries_graded_evidence_not_only_seeds(blocks):
    """The claim that makes it a retrieval prior rather than a seed score."""

    _local, _ptr, control, treatment, prior = blocks
    ranked = (treatment[:, DENSE_RANK_COLUMN] > 0) | (treatment[:, SPLADE_RANK_COLUMN] > 0)
    assert ranked.all()
    assert prior["rows_ranked_by_neither"] == 0
    assert prior["every_candidate_has_graded_evidence"] is True

    non_seed = control[:, SEED_ID_COLUMN] == 0
    assert non_seed.any()
    carried = (treatment[non_seed][:, DENSE_RANK_COLUMN] > 0) | (
        treatment[non_seed][:, SPLADE_RANK_COLUMN] > 0
    )
    assert carried.all(), "non-seed candidates carry no evidence; this is a seed score"


def test_a_candidate_ranked_by_neither_retriever_is_refused(data):
    local, ptr = local_block(data, CONTEXT)
    dense = np.stack([q.candidate_index.numpy()[:4] for q in data.queries])
    splade = np.stack([q.candidate_index.numpy()[:4] for q in data.queries])
    with pytest.raises(RuntimeError, match="ranked by neither retriever"):
        build_graded_retrieval_block(
            data.queries, np.asarray(local), ptr, dense, splade, constant=RRF_CONSTANT
        )


def test_the_prior_counts_both_the_overlap_and_the_singletons(blocks, data):
    _local, _ptr, _control, _treatment, prior = blocks
    overlap = len(range(*DENSE_SLICE.indices(10))) + len(range(*SPLADE_SLICE.indices(10))) - 10
    assert prior["rows_ranked_by_both"] == overlap * len(data.queries)
    assert prior["rows_dense_only"] > 0 and prior["rows_splade_only"] > 0
    assert (
        prior["rows_ranked_by_both"] + prior["rows_dense_only"] + prior["rows_splade_only"]
        == prior["candidate_rows"]
    )


# --- the columns are A3's, reused rather than reinvented -------------------------


def test_the_audit_reports_a3s_own_formula():
    audit = a3_rank_feature_audit(RRF_CONSTANT, 200)
    assert audit["source"] == "mp_retrieval.linear_control.rank_feature_rows"
    assert audit["feature_names"] == list(RANK_FEATURE_NAMES)
    assert audit["formula"] == "(K + 1) / (K + zero_based_rank + 1)"
    assert audit["constant_K"] == RRF_CONSTANT
    assert audit["reused_verbatim"] is True
    assert audit["seeds_special_cased"] is False


def test_the_transform_is_universal_not_pool_dependent():
    """A raw rank would fail this; the frozen reciprocal rank does not."""

    audit = a3_rank_feature_audit(RRF_CONSTANT, 200)
    assert audit["rank_0_value"] == 1.0
    assert audit["invariant_to_list_length"] is True
    long = a3_rank_feature_audit(RRF_CONSTANT, 200)["rank_0_value"]
    short = a3_rank_feature_audit(RRF_CONSTANT, 20)["rank_0_value"]
    assert long == short == 1.0
    expected = (RRF_CONSTANT + 1) / (RRF_CONSTANT + 199 + 1)
    assert audit["worst_rank_value"] == pytest.approx(expected, rel=1e-6)


def test_distinct_ranks_survive_the_blocks_float16():
    assert a3_rank_feature_audit(RRF_CONSTANT, 200)["distinct_ranks_survive_float16"] is True


def test_the_stored_columns_are_exactly_rank_feature_rows(blocks, data):
    """Not a reimplementation: the same function, checked value by value."""

    _local, ptr, _control, treatment, _prior = blocks
    for position, query in enumerate(data.queries):
        candidates = query.candidate_index.numpy()
        expected = rank_feature_rows(
            candidates[DENSE_SLICE], candidates[SPLADE_SLICE], candidates,
            constant=RRF_CONSTANT,
        ).astype(np.float16)
        rows = treatment[int(ptr[position]) : int(ptr[position + 1])]
        assert np.array_equal(rows[:, DENSE_RANK_COLUMN], expected[:, 0]), position
        assert np.array_equal(rows[:, SPLADE_RANK_COLUMN], expected[:, 1]), position


def test_an_absent_retriever_is_exactly_zero(blocks, data):
    _local, ptr, _control, treatment, _prior = blocks
    query = data.queries[0]
    rows = treatment[int(ptr[0]) : int(ptr[1])]
    absent_from_splade = np.arange(query.candidate_index.numel()) < SPLADE_SLICE.start
    assert absent_from_splade.any()
    assert not rows[absent_from_splade][:, SPLADE_RANK_COLUMN].any()


def test_seeds_are_not_special_cased(blocks, data):
    """A seed and a non-seed at the same rank must get the same value."""

    _local, ptr, control, treatment, _prior = blocks
    rows = treatment[int(ptr[0]) : int(ptr[1])]
    seeds = control[int(ptr[0]) : int(ptr[1]), SEED_ID_COLUMN] > 0
    assert seeds.any() and not seeds.all()
    for rank in range(DENSE_SLICE.start, DENSE_SLICE.stop):
        expected = np.float16((RRF_CONSTANT + 1) / (RRF_CONSTANT + rank + 1))
        assert rows[rank, DENSE_RANK_COLUMN] == expected, rank


def test_the_agreement_column_is_exactly_membership_of_both_lists(blocks):
    _local, _ptr, _control, treatment, prior = blocks
    both = (treatment[:, DENSE_RANK_COLUMN] > 0) & (treatment[:, SPLADE_RANK_COLUMN] > 0)
    assert np.array_equal(treatment[:, AGREEMENT_COLUMN] > 0, both)
    assert set(np.unique(treatment[:, AGREEMENT_COLUMN]).tolist()) <= {0.0, 1.0}
    assert prior["agreement_is_new_not_from_a3"] is True


def test_the_agreement_column_is_disclosed_as_new(completed):
    prior = completed["graded_retrieval_prior"]
    assert prior["agreement_is_new_not_from_a3"] is True
    assert "retriever_agreement (new)" in COLUMN_SEMANTICS[AGREEMENT_COLUMN]
    assert completed["a3_rank_feature_audit"]["feature_names"] == list(RANK_FEATURE_NAMES)


# --- the slots are free, so the architecture does not move -----------------------


def test_the_retrieval_columns_land_in_slots_the_control_leaves_empty(blocks):
    _local, _ptr, control, _treatment, _prior = blocks
    assert not control[:, list(RETRIEVAL_COLUMNS)].any()
    assert SEED_ID_COLUMN not in RETRIEVAL_COLUMNS


def test_the_retrieval_columns_are_never_rescaled(blocks):
    """`candidate_readout` touches 4-9; a rescaled column would not be A3's."""

    assert not set(RETRIEVAL_COLUMNS) & set(NORMALISED_COLUMNS)
    _local, _ptr, _control, _treatment, prior = blocks
    assert prior["never_rescaled_by_candidate_readout"] is True


def test_the_structural_columns_stay_zero_in_both_arms(blocks):
    _local, _ptr, control, treatment, prior = blocks
    untouched = [4, 5, 6, 7, 8, 9]
    assert not control[:, untouched].any()
    assert not treatment[:, untouched].any()
    assert prior["zeroed_columns"] == [LOCAL_FEATURE_NAMES[i] for i in untouched]


def test_the_two_arms_have_the_same_parameter_count(completed):
    counts = {arm: completed["results"][arm]["parameters"] for arm in D4_ARMS}
    assert len(set(counts.values())) == 1, counts
    assert completed["ablation"]["architecture_changed"] is False


def test_the_arms_are_not_vacuously_identical(blocks):
    _local, _ptr, control, treatment, _prior = blocks
    assert not np.array_equal(control, treatment), "nothing was added; there is no increment"


# --- one new run, and the control is genuinely reused ----------------------------


def test_only_the_new_arm_was_trained(completed):
    assert completed["arms_trained_here"] == [NEW_ARM]
    assert completed["results"]["SEED_ID_ONLY"]["retrained_here"] is False
    assert completed["results"]["SEED_ID_ONLY"]["measured_in"] == "stage_d3"
    assert completed["results"][NEW_ARM]["retrained_here"] is True
    assert completed["reuse"]["new_runs"] == 1


def test_the_reused_row_is_d3s_own_numbers(completed, d3_result):
    d3 = json.loads(d3_result.read_text(encoding="utf-8"))
    assert completed["results"]["SEED_ID_ONLY"]["validation"] == (
        d3["results"]["SEED_ID_ONLY"]["validation"]
    )
    assert completed["ladder"]["SEED_ID_ONLY"] == d3["ladder"]["SEED_ID_ONLY"]


def test_the_reuse_guard_checks_every_declared_condition(completed):
    checks = completed["reuse"]["conditions_checked"]
    assert completed["reuse"]["all_conditions_match"] is True
    assert all(block["match"] for block in checks.values())
    for required in (
        "dataset", "data_fingerprint", "context", "model", "num_nodes",
        "candidate_contract", "local_feature_schema", "candidate_rows",
        "static_feature_source", "architecture_changed", "control_columns_kept",
        "control_retrained_in_d3", "seed_identity_column", "seed_identity_rows",
        "seed_identity_seed_rows", "seed_identity_mismatches",
        "split_train_fit", "split_validation_reported",
        "training_epochs", "training_seed", "training_learning_rate",
    ):
        assert required in checks, required


@pytest.mark.parametrize(
    "field, value",
    [
        ("status", "GRAPH_CONTEXT_D3_IN_PROGRESS"),
        ("dataset", "some_other_dataset"),
        ("data_fingerprint_sha256", "f" * 64),
        ("context", "TARGET_H1"),
        ("model", "some_other_model"),
        ("num_nodes", 1),
        ("local_feature_schema", ["a", "b"]),
        ("splits.validation_reported", 1),
        ("splits.train_fit", 1),
        ("training.epochs", 99),
        ("training.seed", 7),
        ("training.learning_rate", 0.5),
        ("ablation.architecture_changed", True),
        ("arms.SEED_ID_ONLY.columns_kept", ["distance_0", "distance_1"]),
        ("results.SEED_ID_ONLY.retrained_here", False),
        ("seed_identity_proof.candidate_rows_compared", 1),
        ("seed_identity_proof.seed_rows", 1),
        ("seed_identity_proof.mismatches", 3),
        ("seed_identity_proof.column", "distance_1"),
    ],
)
def test_reuse_is_refused_when_a_condition_differs(
    tmp_path, data, d3_result, field, value
):
    """Every one of these would make the control a different experiment."""

    payload = json.loads(d3_result.read_text(encoding="utf-8"))
    node = payload
    *parents, leaf = field.split(".")
    for key in parents:
        node = node[key]
    assert node[leaf] != value, f"{field} was already {value!r}; this proves nothing"
    node[leaf] = value

    corrupted = tmp_path / "stage_d3.json"
    corrupted.write_text(json.dumps(payload), encoding="utf-8")
    with synthetic(data):
        with pytest.raises(RuntimeError, match="cannot be reused|is not complete"):
            run(d4_args(tmp_path, data, corrupted))


# --- the increment, and the contract --------------------------------------------


def test_the_increment_is_the_ladder_difference(completed):
    delta = completed["increments"]["delta_retrieval_quality"]
    assert delta["from"] == "SEED_ID_ONLY" and delta["to"] == NEW_ARM
    for metric in completed["ladder"][NEW_ARM]:
        expected = (
            completed["ladder"][NEW_ARM][metric]
            - completed["ladder"]["SEED_ID_ONLY"][metric]
        )
        assert delta[metric] == pytest.approx(expected, abs=1e-12), metric


def test_the_increment_is_reported_per_stratum(completed):
    by_stratum = completed["increments_by_stratum"]["delta_retrieval_quality"]
    assert by_stratum
    for stratum, row in by_stratum.items():
        assert row["queries"] > 0, stratum
        for metric in METRICS:
            if metric in row:
                expected = (
                    completed["results"][NEW_ARM]["validation_by_stratum"][stratum][metric]
                    - completed["results"]["SEED_ID_ONLY"]["validation_by_stratum"][stratum][
                        metric
                    ]
                )
                assert row[metric] == pytest.approx(expected, abs=1e-12), (stratum, metric)


def test_d3s_increments_are_carried_forward_for_scale(completed, d3_result):
    d3 = json.loads(d3_result.read_text(encoding="utf-8"))
    for name in ("delta_seed", "delta_distance", "delta_remaining_structure"):
        assert completed["comparators"][f"{name}_d3"] == d3["increments"][name]


def test_the_contract_holds(completed):
    contract = completed["contract"]
    assert contract["gnn_trained"] is False
    assert contract["message_passing"] is False
    assert contract["test_split_read"] is False
    assert contract["epoch_selected_on_validation"] is False
    assert contract["architecture_changed"] is False
    assert contract["scored_nodes"] == "exactly Cq in every arm"
    assert completed["splits"]["test_read"] is False
    assert completed["status"] == COMPLETE_STATUS


def test_the_test_split_is_refused_outright(tmp_path, data, d3_result):
    with synthetic(data):
        args = d4_args(tmp_path, data, d3_result)
        args.splits = ["train", "validation", "test"]
        with pytest.raises(ValueError, match="test split is not read"):
            run(args)


def test_an_absent_d3_result_is_refused(tmp_path, data):
    with synthetic(data):
        with pytest.raises(FileNotFoundError, match="SEED_ID_ONLY"):
            run(d4_args(tmp_path, data, tmp_path / "nothing.json"))


def test_the_operating_point_is_the_frozen_one():
    defaults = {a.dest: a.default for a in build_parser()._actions}
    assert defaults["epochs"] == 3
    assert defaults["batch_size"] == 16
    assert defaults["learning_rate"] == 0.001
    assert defaults["weight_decay"] == 0.0001
    assert defaults["seed"] == 0
    assert defaults["rrf_constant"] == 60
    assert defaults["holdout_fraction"] == 0.1


def test_the_reported_split_is_validation(completed, data):
    assert completed["splits"]["validation_reported"] == len(
        [q for q in data.queries if q.split == SPLITS["validation"]]
    )
    assert completed["training"]["epoch_selection"].endswith("never validation")
