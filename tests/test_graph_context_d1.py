"""Stage D1 is a controlled comparison, and these tests are what makes it one.

D1 claims that the only thing separating its two arms is the node space the
frozen feature kernel runs on. That claim is cheap to make and easy to break: an
epoch selected on the reporting split, a validation query reachable through the
feature store, a local block silently misaligned against the candidate order, or
a historical arm that is not bit-for-bit the historical arm would each turn a
context experiment into an uninterpretable one.

So the tests here are mostly refusals and identities rather than assertions about
outcomes. The one behavioural test that remains is the vacuity check: if the two
arms produced the same features there would be nothing to measure, and a green
suite would be hiding that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from mp_retrieval.complete_data import CompleteQuery, CompleteRetrievalDataset  # noqa: E402
from mp_retrieval.graph_context import (  # noqa: E402
    build_operators,
    context_nodes,
    qls_local_features,
)
from scripts.run_graph_context_d1 import (  # noqa: E402
    COMPLETE_STATUS,
    D1_ARMS,
    METRICS,
    MODEL_NAME,
    bucket_distribution,
    build_local_features,
    build_parser,
    context_feature_store,
    holdout_split,
    run,
)

NODES = 48
FEATURE_DIM = 8
QUERIES = 32
CONTRACT = "d1-synthetic-contract"


def graph() -> tuple[np.ndarray, np.ndarray]:
    """A graph whose bridges live *outside* the pools that need them.

    Every second node is a hub that no pool contains: reaching it takes the
    ``TARGET_H1`` context, which is the whole point of the arm. Without such
    nodes the two arms would coincide and the experiment would have no contrast
    to measure.
    """
    rng = np.random.default_rng(11)
    src = np.repeat(np.arange(NODES), 3)
    dst = (src + rng.integers(1, 7, size=src.size)) % NODES
    keep = src != dst
    return src[keep], dst[keep]


def dataset() -> CompleteRetrievalDataset:
    rng = np.random.default_rng(3)
    src, dst = graph()
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    rowptr = np.zeros(NODES + 1, dtype=np.int64)
    np.add.at(rowptr, src + 1, 1)
    rowptr = np.cumsum(rowptr)

    queries: list[CompleteQuery] = []
    for index in range(QUERIES):
        pool = rng.choice(NODES, size=10, replace=False).astype(np.int64)
        golds = pool[:2]
        queries.append(
            CompleteQuery(
                query_index=index,
                query_id=f"q{index}",
                candidate_index=torch.from_numpy(pool),
                relevant_local=torch.tensor([0, 1], dtype=torch.long),
                relevant_global=torch.from_numpy(golds.copy()),
                anchor_global=int(pool[0]),
                split=0 if index < 24 else 1,
                retrieval_seed_local=torch.tensor([2, 3], dtype=torch.long),
            )
        )
    return CompleteRetrievalDataset(
        root=Path("<synthetic>"),
        dataset="synthetic",
        node_array=rng.normal(size=(NODES, FEATURE_DIM)).astype(np.float32),
        query_array=rng.normal(size=(QUERIES, FEATURE_DIM)).astype(np.float32),
        rowptr=torch.from_numpy(rowptr),
        col=torch.from_numpy(dst),
        queries=queries,
        metadata={"candidate_contract_sha256": CONTRACT, "graph_nodes": NODES},
    )


@pytest.fixture(scope="module")
def data() -> CompleteRetrievalDataset:
    return dataset()


def local_block(data: CompleteRetrievalDataset, arm: str, queries=None):
    rowptr = data.rowptr.numpy()
    col = data.col.numpy()
    operators = build_operators(rowptr, col, NODES)
    return build_local_features(
        data.queries if queries is None else queries,
        rowptr, col, NODES, operators, arm,
        damping=0.85, ppr_iterations=8, latency=[],
    )


def args_for(tmp_path: Path, data: CompleteRetrievalDataset, **overrides):
    baseline = {
        "candidate_contract_sha256": CONTRACT,
        "selected_gnn": {"parameters": {"parameters": 4000}},
    }
    argv = [
        "--data", str(tmp_path),
        "--feature-cache", str(tmp_path / "absent"),
        "--dataset", "synthetic",
        "--expected-queries", str(len(data.queries)),
        "--baseline", json.dumps(baseline),
        "--data-fingerprint-sha256", "0" * 64,
        "--output", str(tmp_path / "stage_d1.json"),
        "--epochs", "1",
        "--batch-size", "8",
        "--device", "cpu",
    ]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", *[str(item) for item in value]]
    return build_parser().parse_args(argv)


@pytest.fixture(scope="module")
def completed(tmp_path_factory, data):
    """One real end-to-end run, driving the frozen model path on both arms."""

    tmp_path = tmp_path_factory.mktemp("d1")
    static = np.linspace(0.0, 1.0, NODES * 7, dtype=np.float32).reshape(NODES, 7)
    import scripts.run_graph_context_d1 as module

    original_load, original_static = module.load_complete_dataset, module.load_or_build_static
    module.load_complete_dataset = lambda *a, **k: data
    module.load_or_build_static = lambda *a, **k: (static, {"source": "synthetic"})
    try:
        return run(args_for(tmp_path, data))
    finally:
        module.load_complete_dataset = original_load
        module.load_or_build_static = original_static


# --- The historical arm must remain the historical arm ------------------------


def test_the_candidate_arm_is_bit_identical_to_the_frozen_kernel(data):
    """CAND under candidate readout is not an approximation of CAND before it.

    The readout divides by a maximum taken over the candidates, and under CAND
    the kernel's nodes *are* the candidates, so that maximum is the one the
    frozen pipeline already used. Anything less than bit-equality here would mean
    the control arm had quietly become a third variant.
    """
    rowptr, col = data.rowptr.numpy(), data.col.numpy()
    operators = build_operators(rowptr, col, NODES)
    local, ptr = local_block(data, "CAND")
    for position, query in enumerate(data.queries):
        candidates = query.candidate_index.numpy()
        pool = np.unique(candidates)
        seeds = np.unique(candidates[query.retrieval_seed_local.numpy()])
        nodes = context_nodes("CAND", operators=operators, pool=pool, seeds=seeds)
        frozen = qls_local_features(
            rowptr=rowptr, col=col, nodes=nodes, pool=pool, seeds=seeds, size=NODES,
            damping=0.85, ppr_iterations=8, normalisation="context",
        )[np.searchsorted(pool, candidates)].astype(np.float16)
        assert np.array_equal(local[int(ptr[position]) : int(ptr[position + 1])], frozen)


def test_the_two_arms_do_not_produce_the_same_features(data):
    """A vacuity check: with identical features D1 would measure nothing."""

    cand, _ = local_block(data, "CAND")
    target, _ = local_block(data, "TARGET_H1")
    assert cand.shape == target.shape
    assert not np.array_equal(cand, target)


def test_the_restored_context_can_only_pull_candidates_closer_to_the_seeds(data):
    """Widening the kernel's node space adds paths; it never removes them.

    Columns 0-3 are the one-hot distance bucket, so this is checkable exactly:
    no candidate may land in a *later* bucket under the wider context.
    """
    cand, _ = local_block(data, "CAND")
    target, _ = local_block(data, "TARGET_H1")
    assert (np.argmax(target[:, :4], axis=1) <= np.argmax(cand[:, :4], axis=1)).all()


# --- Alignment and addressability ---------------------------------------------


def test_features_are_packed_in_the_frozen_candidate_order(data):
    """Sorting the pool for the kernel must not reorder what the ranker reads."""

    rowptr, col = data.rowptr.numpy(), data.col.numpy()
    operators = build_operators(rowptr, col, NODES)
    local, ptr = local_block(data, "TARGET_H1")
    query = data.queries[0]
    candidates = query.candidate_index.numpy()
    assert not np.array_equal(candidates, np.sort(candidates)), "fixture must be unsorted"
    pool = np.unique(candidates)
    seeds = np.unique(candidates[query.retrieval_seed_local.numpy()])
    sorted_rows = qls_local_features(
        rowptr=rowptr, col=col,
        nodes=context_nodes("TARGET_H1", operators=operators, pool=pool, seeds=seeds),
        pool=pool, seeds=seeds, size=NODES,
        damping=0.85, ppr_iterations=8, normalisation="candidate",
    ).astype(np.float16)
    packed = local[: int(ptr[1])]
    for row, candidate in enumerate(candidates):
        assert np.array_equal(packed[row], sorted_rows[np.searchsorted(pool, candidate)])


def test_the_pointer_matches_every_pool_it_claims_to_cover(data):
    _local, ptr = local_block(data, "CAND")
    assert ptr[0] == 0 and len(ptr) == len(data.queries) + 1
    widths = np.diff(ptr)
    assert list(widths) == [int(q.candidate_index.numel()) for q in data.queries]


def test_a_query_the_stage_never_opened_is_not_addressable(data):
    """The test split is refused twice: unread, and also unreachable.

    A store that returned silent zeros for an unopened query would let a later
    edit read the reporting split by accident. This one raises.
    """
    opened = data.queries[:4]
    local, ptr = local_block(data, "CAND", queries=opened)
    static = np.zeros((NODES, 7), dtype=np.float32)
    store = context_feature_store(opened, static, local, ptr, len(data.queries), arm="CAND")
    assert store.local_for_query(opened[2]).shape[0] == opened[2].candidate_index.numel()
    with pytest.raises(KeyError):
        store.local_for_query(data.queries[-1])


def test_the_store_hands_the_ranker_the_static_and_local_blocks_it_expects(data):
    opened = data.queries[:4]
    local, ptr = local_block(data, "CAND", queries=opened)
    static = np.arange(NODES * 7, dtype=np.float32).reshape(NODES, 7)
    store = context_feature_store(opened, static, local, ptr, len(data.queries), arm="CAND")
    batch = store.batch_features(
        opened[:2], include_static=True, include_local=True, device=torch.device("cpu")
    )
    assert batch.shape == (int(ptr[2]), 17)
    assert torch.equal(
        batch[:, :7],
        torch.from_numpy(static[torch.cat([q.candidate_index for q in opened[:2]]).numpy()]),
    )


def test_the_store_is_not_written_to_the_volume(data):
    opened = data.queries[:2]
    local, ptr = local_block(data, "CAND", queries=opened)
    store = context_feature_store(
        opened, np.zeros((NODES, 7), np.float32), local, ptr, len(data.queries), arm="CAND"
    )
    assert store.metadata["persisted"] is False
    assert not store.root.exists()


# --- Epoch selection never sees the reporting split ---------------------------


def test_the_epoch_selection_holdout_comes_out_of_train(data):
    train = data.queries[:24]
    fit, holdout = holdout_split(train, 0.25)
    assert len(fit) + len(holdout) == len(train)
    assert {q.query_id for q in fit}.isdisjoint({q.query_id for q in holdout})
    assert fit + holdout == train


def test_the_holdout_is_the_same_slice_every_time(data):
    train = data.queries[:24]
    assert holdout_split(train, 0.1) == holdout_split(train, 0.1)


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.5])
def test_a_holdout_that_is_not_a_proper_fraction_is_refused(data, fraction):
    with pytest.raises(ValueError):
        holdout_split(data.queries[:24], fraction)


def test_a_training_split_too_small_to_hold_out_is_an_error_not_an_empty_set(data):
    with pytest.raises(ValueError):
        holdout_split(data.queries[:1], 0.9)


def test_the_reported_split_is_never_handed_to_the_fitter(tmp_path, data, monkeypatch):
    """The frozen ``_fit`` selects on whatever it is given. It is not given validation."""

    import scripts.run_graph_context_d1 as module

    seen: list[set[str]] = []
    original = module._fit

    def record(model_name, model, train_queries, validation_queries, *a, **k):
        seen.append({q.query_id for q in validation_queries})
        return original(model_name, model, train_queries, validation_queries, *a, **k)

    static = np.zeros((NODES, 7), dtype=np.float32)
    monkeypatch.setattr(module, "load_complete_dataset", lambda *a, **k: data)
    monkeypatch.setattr(module, "load_or_build_static", lambda *a, **k: (static, {}))
    monkeypatch.setattr(module, "_fit", record)
    run(args_for(tmp_path, data))

    reported = {q.query_id for q in data.queries if q.split == 1}
    assert len(seen) == len(D1_ARMS)
    for selection in seen:
        assert selection and selection.isdisjoint(reported)


# --- Refusals -----------------------------------------------------------------


@pytest.mark.parametrize(
    "splits", [["train", "validation", "test"], ["test"], ["train"], ["validation"]]
)
def test_only_the_two_development_splits_are_accepted(tmp_path, data, splits):
    with pytest.raises(ValueError):
        run(args_for(tmp_path, data, splits=splits))


def test_a_query_count_that_differs_from_the_protocol_is_refused(tmp_path, data, monkeypatch):
    import scripts.run_graph_context_d1 as module

    monkeypatch.setattr(module, "load_complete_dataset", lambda *a, **k: data)
    with pytest.raises(ValueError):
        run(args_for(tmp_path, data, expected_queries=[len(data.queries) + 1]))


def test_a_candidate_contract_that_does_not_match_the_baseline_is_refused(
    tmp_path, data, monkeypatch
):
    import scripts.run_graph_context_d1 as module

    monkeypatch.setattr(module, "load_complete_dataset", lambda *a, **k: data)
    args = args_for(tmp_path, data)
    args.baseline = dict(args.baseline, candidate_contract_sha256="something-else")
    with pytest.raises(ValueError):
        run(args, None)


# --- The distance-bucket channel ----------------------------------------------


def test_the_bucket_shares_are_a_distribution(data):
    local, ptr = local_block(data, "TARGET_H1")
    shares = bucket_distribution(local, ptr, data.queries, golds_only=False)
    assert shares["candidates"] == int(ptr[-1])
    assert sum(value for key, value in shares.items() if key != "candidates") == pytest.approx(1.0)


def test_the_gold_view_counts_only_the_golds(data):
    local, ptr = local_block(data, "CAND")
    shares = bucket_distribution(local, ptr, data.queries, golds_only=True)
    assert shares["candidates"] == sum(int(q.relevant_local.numel()) for q in data.queries)


# --- What the run reports ------------------------------------------------------


def test_the_run_reports_both_arms_and_the_metrics_it_declared(completed):
    assert completed["status"] == COMPLETE_STATUS
    assert list(completed["results"]) == list(D1_ARMS)
    for arm in D1_ARMS:
        assert set(completed["results"][arm]["validation"]) == set(METRICS)


def test_the_delta_is_reported_against_the_historical_arm(completed):
    delta = completed["delta_against_cand"]["TARGET_H1"]
    for key in METRICS:
        assert delta[key] == pytest.approx(
            completed["results"]["TARGET_H1"]["validation"][key]
            - completed["results"]["CAND"]["validation"][key]
        )


def test_both_arms_have_the_same_number_of_parameters(completed):
    """The architecture is held fixed; only the values entering it move."""

    counts = {completed["results"][arm]["parameters"] for arm in D1_ARMS}
    assert len(counts) == 1


def test_the_run_records_that_it_used_no_message_passing_and_no_bridge(completed):
    contract = completed["contract"]
    assert contract["gnn_trained"] is False
    assert contract["gnn_output_read"] is False
    assert contract["message_passing"] is False
    assert contract["bridge_feature_used"] is False
    assert contract["architecture_changed"] is False
    assert contract["test_split_read"] is False
    assert completed["model"] == MODEL_NAME


def test_the_run_records_the_normalisation_that_removed_the_declared_confound(completed):
    assert completed["contract"]["normalisation"] == "candidate-readout in every arm"
    assert completed["contract"]["normalization_nodes"] == "exactly Cq in every arm"


def test_the_feature_build_cost_is_measured_per_arm(completed):
    for arm in D1_ARMS:
        latency = completed["results"][arm]["feature_build"]["latency_ms_per_query"]
        assert latency["p50"] <= latency["p95"] <= latency["p99"] <= latency["max"]
        assert completed["results"][arm]["feature_build"]["candidate_rows"] > 0


def test_the_bucket_distributions_are_reported_for_both_arms(completed):
    for arm in D1_ARMS:
        buckets = completed["results"][arm]["seed_distance_buckets"]
        assert set(buckets) == {"all_candidates", "gold_candidates"}


def test_the_result_file_is_written_before_the_second_arm_starts(tmp_path, data, monkeypatch):
    """A crash in arm two must leave arm one on disk rather than nothing."""

    import scripts.run_graph_context_d1 as module

    static = np.zeros((NODES, 7), dtype=np.float32)
    monkeypatch.setattr(module, "load_complete_dataset", lambda *a, **k: data)
    monkeypatch.setattr(module, "load_or_build_static", lambda *a, **k: (static, {}))
    args = args_for(tmp_path, data)
    seen: list[dict] = []
    run(args, lambda: seen.append(json.loads(args.output.read_text(encoding="utf-8"))))
    assert seen[0]["status"] != COMPLETE_STATUS
    assert list(seen[0]["results"]) == ["CAND"]


# --- The launcher and the gate --------------------------------------------------


@pytest.fixture
def launcher(monkeypatch):
    """The launcher module, re-imported under a chosen stage and put back after.

    The stage and the query cap are read from the environment at import, because
    Modal fixes a function's timeout and shape when the decorator runs. Reloading
    is therefore the only way to test another stage -- and reloading back at
    teardown is the only way to keep that from leaking into the tests that import
    this module without reloading it, which would silently reprice a CPU stage as
    a GPU one.
    """
    import importlib

    import scripts.modal_graph_context_pilot as module

    def load(stage="stage_d1", query_cap="0"):
        monkeypatch.setenv("GRAPH_CONTEXT_STAGE", stage)
        monkeypatch.setenv("GRAPH_CONTEXT_QUERY_CAP", query_cap)
        return importlib.reload(module)

    try:
        yield load
    finally:
        monkeypatch.undo()
        importlib.reload(module)


def test_the_launcher_gives_d1_both_development_splits_and_neither_more(launcher):
    module = launcher()
    args = module._d1_runner_args(module._jobs(["2wiki_clean"])[0])
    assert args.splits == ["train", "validation"]


def test_the_launcher_carries_the_frozen_hyperparameters_rather_than_its_own(launcher):
    """QLS-v1's settings are read from the sealed confirmation, not retyped."""

    module = launcher()
    args = module._d1_runner_args(module._jobs(["2wiki_clean"])[0])
    frozen = json.loads(
        (Path(module.HOST_REPO_ROOT) / "outputs/sa_mlp_confirmation/2wiki_clean.json").read_text(
            encoding="utf-8"
        )
    )["config"]
    assert args.epochs == frozen["epochs"]
    assert args.batch_size == frozen["batch_size"]
    assert args.learning_rate == frozen["learning_rate"]
    assert args.weight_decay == frozen["weight_decay"]
    assert args.projection_dim == frozen["projection_dim"]
    assert args.dropout == frozen["dropout"]
    assert args.temperature == frozen["temperature"]


def test_d1_is_the_only_stage_that_asks_for_an_accelerator(launcher):
    assert launcher("stage_d1").GPU == "A10G"
    for stage in ("stage_b", "stage_c", "stage_d0", "stage_d0b"):
        assert launcher(stage).GPU is None, stage


def test_the_window_the_gate_reads_is_the_window_modal_will_enforce(launcher):
    module = launcher()
    assert module.TIMEOUT_SECONDS == module.MODAL_CONFIG["stage_timeout_seconds"]["stage_d1"]


def test_d1s_cost_model_does_not_collapse_to_the_load_term(launcher):
    """`query_cap` 0 means the whole split, not zero queries."""

    from scripts.spawn_modal_jobs import (
        GRAPH_CONTEXT_LOAD_SECONDS,
        _graph_context_d1_seconds,
    )

    module = launcher()
    job = module._jobs(["2wiki_clean"])[0]
    assert int(job["query_cap"]) == 0
    assert _graph_context_d1_seconds(module, job) > GRAPH_CONTEXT_LOAD_SECONDS * 1.5


def test_the_declared_d1_ceiling_is_not_below_what_the_gate_projects(launcher):
    """A declaration under its own gate's projection would block its own launch."""

    from scripts.spawn_modal_jobs import gate_launch

    module = launcher()
    jobs = module._jobs(["2wiki_clean"])
    report = gate_launch("graph-context", module, jobs)
    declared = module.CONFIG["stages"]["D1"]["projected_cost"]
    assert report["total_hours"] <= declared["gpu_hours_ceiling"]
    assert report["largest_unit_hours"] * 60.0 <= declared["wall_time_ceiling_minutes"]
    # The gate's own spend figure, not hours x rate: it divides by the standing
    # assumption about how much of billed time is useful work, and a ceiling that
    # ignored that would be under the number the gate actually reports.
    assert report["expected_spend_usd"] <= declared["cost_ceiling_usd"]


def test_the_declaration_records_the_prerequisite_it_cannot_run_without(launcher):
    """D1 needs embeddings the active workspace did not hold. Declared, not assumed."""

    declared = launcher().CONFIG["stages"]["D1"]["prerequisite"]
    assert declared["status"] == "SATISFIED"
    assert declared["is_a_workspace_migration"] is False
    assert declared["required_megabytes"] == pytest.approx(
        sum(entry["megabytes"] for entry in declared["files"])
    )


def test_the_replication_stayed_inside_what_was_approved(launcher):
    """The approval was for two files and 473.8 MB, and for nothing else.

    A replication that quietly grows is the failure mode this checks: the record
    has to say what was copied, that the optional payload was not, and that the
    hashes were compared rather than assumed. It is also what keeps the stage
    honest about *why* the optional payload was skipped -- the approval allowed
    it if a correctness check needed it, so the record must claim none did.
    """

    declared = launcher().CONFIG["stages"]["D1"]["prerequisite"]
    done = declared["executed"]
    assert done["files_copied"] == len(declared["files"]) == 2
    assert done["megabytes"] == declared["required_megabytes"] == pytest.approx(473.8)
    assert done["optional_derived_copied"] is False
    assert done["optional_derived_reason"].strip()

    verification = done["verification"]
    assert verification["mismatches"] == 0
    assert "sha256" in verification["before_transfer"]
    assert "sha256" in verification["after_transfer"]

    names = {entry["path"].rsplit("/", 1)[-1] for entry in done["files_verified"]}
    assert names == {entry["path"] for entry in declared["files"]}
    for entry in done["files_verified"]:
        assert len(entry["sha256"]) == 64 and int(entry["sha256"], 16) >= 0
        assert entry["bytes"] > 0
    # The two files are distinct, so two identical digests would mean one file
    # was hashed twice and the other never checked.
    assert len({entry["sha256"] for entry in done["files_verified"]}) == 2


def test_the_resolved_confound_still_records_that_it_existed(launcher):
    """Fixing a declared confound must not erase the record that it was there."""

    resolved = launcher().CONFIG["stages"]["D1"]["resolved_confound"]
    assert set(resolved) == {"was", "fix", "residual", "confound_free_channel"}
    assert "personalised PageRank" in resolved["residual"]


def test_adding_a_gpu_stage_did_not_reprice_the_cpu_stages(launcher):
    """D0b runs no accelerator, so its declaration must not be judged against one.

    The gate reads a shape off the launcher module. Once one stage in that module
    asked for a GPU, reading the raw config key would have priced every stage at
    the GPU rate and made D0b's sealed ceiling look breached by threefold.
    """
    from scripts.spawn_modal_jobs import gate_launch

    module = launcher("stage_d0b")
    report = gate_launch("graph-context", module, module._jobs(["2wiki_clean"]))
    declared = module.CONFIG["stages"]["D0B"]["projected_cost"]
    assert report["expected_spend_usd"] <= declared["cost_ceiling_usd"]
    assert report["container_usd_per_hour"] < 1.0
