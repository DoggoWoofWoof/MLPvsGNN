"""M2B's runner, end to end on a toy cell M2 itself built.

The claim this phase makes is comparative: S2, S3 and S4 differ in semantic
representation and in nothing else. The ways that claim could quietly be false
are what this file tests.

The fixture is deliberately two runners in sequence. M2's own runner builds the
cell master on the shared toy dataset; M2B's runner then loads it. That is the
real reuse path -- a store persisted by one phase, admitted by another on its
feature build contract rather than on the hash of a declaration file -- rather
than a store this file wrote for its own convenience.

What is asserted:

- the structural inputs are identical across rungs, by digest, per family,
  including the NODE_ROLE / SUPPORT / PATH values the smoke has to establish;
- the semantic branch is the only thing that differs, and the rungs really are
  distinguishable from each other in the artifact;
- the rung weights match what the declaration says they weigh, at the real
  1536 width, measured rather than transcribed;
- a checkpoint reloads into the scores it reported;
- a fabricated reuse is refused: S3 without M2's fits is an error, never a
  silent refit reported in the reused column;
- and a store whose declaration hash has drifted is still admitted, which is
  the whole reason feature_build_contract exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from test_run_m1a_feature_screen import _args as _m1a_args  # noqa: E402

from scripts import feature_build_contract as fbc  # noqa: E402
from scripts import m2b_semantic_formula_freeze as _freeze  # noqa: E402
from scripts import run_m1a_feature_screen as _m1a  # noqa: E402
from scripts import run_m2_qls_v2_freeze as _m2  # noqa: E402
from scripts import run_m2b_semantic_minimality as runner  # noqa: E402

M2B_DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2b_semantic_minimality.yaml").read_text(encoding="utf-8")
)
M2_DECLARATION = yaml.safe_load(
    (REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml").read_text(encoding="utf-8")
)

#: The toy fixture is hotpotqa_clean at width 6; R2 keeps the fixture cheap and
#: needs no A64 family graph in the M2B half.
TOY_DATASET = "hotpotqa_clean"
TOY_REGIME = "R2"

#: The rungs the fan-out actually fits. S3 is exercised separately, against a
#: real M2 fit, because reusing it is a different code path from fitting it.
SMOKE_RUNGS = ["S2", "S4"]


def _head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()


def _all_reusable_manifest() -> dict:
    fits = []
    for regime, arms in M2_DECLARATION["m2_selection_matrix"]["cells"][TOY_DATASET].items():
        for arm, status in arms.items():
            if str(status).startswith("reuse_"):
                fits.append({"dataset": TOY_DATASET, "regime": regime, "arm": arm, "reusable": True})
    return {
        "status": "M2_REUSE_AUDIT_COMPLETE",
        "bit_exact_feature_probe": {
            "status": "BIT_EXACT_HISTORICAL_ARM_EQUIVALENCE",
            "all_historical_arms_bit_exact": True,
        },
        "fits": fits,
    }


def _m2b_args(tmp_path: Path, **overrides) -> argparse.Namespace:
    """M2B's arguments, from the same toy dataset M1A and M2 use."""

    base = vars(_m1a_args(tmp_path, dataset=TOY_DATASET))
    for gone in ("edge_provenance_root", "edge_families", "arms", "semantic_rung"):
        base.pop(gone, None)
    base.update(
        {
            "seed": 0,
            "regimes": [TOY_REGIME],
            "rungs": list(SMOKE_RUNGS),
            "source_commit": _head_commit(),
            "artifact_root": tmp_path / "artifacts",
            "cell_master_root": None,
            "m2_fits_root": None,
            "latency_queries": 2,
            "latency_repeats": 1,
            "latency_warmup": 1,
            "output": tmp_path / "out" / "m2b.json",
        }
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _build_the_cell(tmp_path: Path, commit: str) -> None:
    """M2 persists the cell master M2B is forbidden to rebuild."""

    monkeypatch = pytest.MonkeyPatch()
    manifest = tmp_path / "reuse_audit.json"
    manifest.write_text(json.dumps(_all_reusable_manifest()), encoding="utf-8")
    monkeypatch.setattr(_m2, "REUSE_MANIFEST_PATH", manifest)
    m2_args = argparse.Namespace(
        **{
            **vars(_m1a_args(tmp_path, dataset=TOY_DATASET)),
            "seed": 0,
            "regimes": [TOY_REGIME],
            "arms": None,
            "stage": "build",
            "source_commit": commit,
            "artifact_root": tmp_path / "artifacts",
            "output": tmp_path / "out" / "m2_build.json",
        }
    )
    try:
        _m2.run(m2_args)
    finally:
        monkeypatch.undo()


@pytest.fixture(scope="module")
def _constants_snapshot(tmp_path_factory):
    """A snapshot that also covers HEAD, so a store built here can be dated.

    The committed snapshot records the commits M2's stores were built at. These
    tests build their own store at whatever HEAD happens to be, and dating it
    is refused unless that commit is recorded -- which is the refusal working,
    not a problem with it.

    The entry is resolved from git exactly the way the committed one was, so
    the tests still run the production path against a real historical value
    rather than one asserted into place.
    """

    committed = json.loads(fbc.FORMULA_CONSTANT_SNAPSHOT_PATH.read_text(encoding="utf-8"))
    commit = _head_commit()
    path = tmp_path_factory.mktemp("constants") / "snapshot.json"
    path.write_text(
        json.dumps(
            {
                **committed,
                "commits": {
                    **committed["commits"],
                    commit: {
                        "built_datasets": ["<built by this test>"],
                        "constants": fbc.historical_formula_constants(commit),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(fbc, "FORMULA_CONSTANT_SNAPSHOT_PATH", path)
        yield path


@pytest.fixture(scope="module")
def smoke(tmp_path_factory, _constants_snapshot):
    """One real S2 fit and one real S4 fit, on the same loaded cell."""

    tmp_path = tmp_path_factory.mktemp("m2b_smoke")
    commit = _head_commit()
    _build_the_cell(tmp_path, commit)
    args = _m2b_args(tmp_path, source_commit=commit)
    return runner.run(args), args


@pytest.fixture(scope="module")
def cell(smoke):
    return smoke[0]["cells"][TOY_REGIME]


# --------------------------------------------------------------------------
# The comparison is controlled
# --------------------------------------------------------------------------


def test_the_run_completes_and_names_its_declaration(smoke) -> None:
    result, _args = smoke
    assert result["status"] == runner.STATUS_COMPLETE
    assert result["declaration"] == "configs/m2b_semantic_minimality.yaml"
    assert result["arm"] == "QLS-UNIVERSAL"
    assert result["seed"] == 0
    assert result["test_split_read"] is False


def test_every_rung_records_the_same_shared_inputs(cell) -> None:
    """The check that makes "only the semantic rung differs" falsifiable."""

    recorded = {rung: fit["shared_inputs_sha256"] for rung, fit in cell["rungs"].items()}
    assert len(set(recorded.values())) == 1, recorded
    assert set(recorded) == set(SMOKE_RUNGS)


def test_the_structural_families_are_digested_one_by_one(cell) -> None:
    """NODE_ROLE, SUPPORT and PATH are separately checkable, not just "the store"."""

    digests = cell["shared_inputs"]["structural_family_digests"]
    assert set(digests) == set(_m1a.MASTER_COLUMNS)
    for family in ("NODE_ROLE", "SUPPORT", "PATH", "BASE"):
        assert len(digests[family]) == 64


def test_a_changed_structural_family_changes_its_digest_and_no_other() -> None:
    """A digest that moved for an untouched family would prove nothing."""

    rng = np.random.default_rng(0)
    blocks = [rng.normal(size=(5, 12)).astype(np.float32) for _ in range(2)]
    before = runner.structural_family_digests(blocks)

    mutated = [block.copy() for block in blocks]
    mutated[0][:, _m1a.MASTER_COLUMNS["NODE_ROLE"]] += 1.0
    after = runner.structural_family_digests(mutated)

    assert after["NODE_ROLE"] != before["NODE_ROLE"]
    for family in set(before) - {"NODE_ROLE"}:
        assert after[family] == before[family], family


def test_the_rungs_are_distinguishable_from_one_another(cell) -> None:
    fingerprints = {
        rung: fit["semantic_rung_fingerprint"]["sha256"] for rung, fit in cell["rungs"].items()
    }
    assert len(set(fingerprints.values())) == len(fingerprints), fingerprints


def test_the_rungs_differ_in_scorer_width_by_exactly_their_semantic_columns(cell) -> None:
    for rung, fit in cell["rungs"].items():
        columns = fit["semantic_rung_fingerprint"]["semantic_columns"]
        width = fit["precomputed_width"] + columns
        assert fit["parameters"]["scorer"] == 32 * width + 65, rung


def test_the_held_out_queries_are_the_same_queries_for_every_rung(smoke, cell) -> None:
    _result, args = smoke
    rows = {
        rung: json.loads(
            (Path(args.artifact_root) / TOY_REGIME / rung.lower() / "per_query_rows.json")
            .read_text(encoding="utf-8")
        )["query_ids"]
        for rung in cell["rungs"]
    }
    first = rows[SMOKE_RUNGS[0]]
    assert first, "the fixture must hold out at least one query"
    for rung, ids in rows.items():
        assert ids == first, rung


def test_one_store_is_persisted_for_the_cell_not_one_per_rung(smoke, cell) -> None:
    """The sharing is the scientific claim, so it is also the on-disk shape."""

    _result, args = smoke
    cell_root = Path(args.artifact_root) / TOY_REGIME
    assert (cell_root / "arm_store" / "metadata.json").is_file()
    metadata = json.loads((cell_root / "arm_store" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["shared_by_rungs"] == SMOKE_RUNGS
    assert metadata["fingerprint_sha256"] == cell["shared_inputs"]["arm_store_sha256"]
    for rung in cell["rungs"]:
        assert not (cell_root / rung.lower() / "feature_store").exists(), (
            "a per-rung store would let two rungs train on different structural tensors"
        )


def test_no_cell_master_is_rebuilt(smoke, monkeypatch, tmp_path) -> None:
    """M2B reads M2's features or it fails; it never recomputes them.

    Poisoning the builder and then running the whole cell loop again -- to a
    fresh output path, so the resume shortcut cannot hide the answer -- is the
    only way to establish this. A source-level check would pass on code that
    imported the builder and called it under another name.
    """

    _result, args = smoke

    def _forbidden(**_kwargs):
        raise AssertionError("M2B rebuilt a cell master instead of loading M2's")

    monkeypatch.setattr(_m1a, "_cell_master_local", _forbidden)
    # Its own artifact root, holding a copy of the same persisted master, so the
    # rerun cannot overwrite the fixture other tests read.
    elsewhere = tmp_path / "artifacts" / TOY_REGIME / "cell_features"
    elsewhere.mkdir(parents=True, exist_ok=True)
    for item in (Path(args.artifact_root) / TOY_REGIME / "cell_features").iterdir():
        (elsewhere / item.name).write_bytes(item.read_bytes())

    rerun = _m2b_args(tmp_path, source_commit=args.source_commit)
    rerun.data = args.data
    rerun.baseline = args.baseline
    rerun.expected_queries = args.expected_queries
    rerun.artifact_root = tmp_path / "artifacts"
    rerun.output = tmp_path / "again.json"
    again = runner.run(rerun)
    assert again["status"] == runner.STATUS_COMPLETE
    assert again["features_were_loaded_not_rebuilt"].startswith(
        "M2B never calls _cell_master_local"
    )
    # And the loaded cell is bit-identical to the one the first run used.
    assert (
        again["cells"][TOY_REGIME]["shared_inputs"]["cell_features_sha256"]
        == _result["cells"][TOY_REGIME]["shared_inputs"]["cell_features_sha256"]
    )


# --------------------------------------------------------------------------
# The parameter counts are the phase's claim
# --------------------------------------------------------------------------


def test_the_declared_parameter_counts_are_what_the_rungs_actually_weigh() -> None:
    """Measured at the real 1536 width, not transcribed from the declaration."""

    for rung, declared in runner.DECLARED_PARAMETERS.items():
        head = runner.build_semantic_head(rung, runner.FROZEN_EMBEDDING_DIM)
        model = _m1a.build_m1a_model(
            precomputed_width=_freeze.PRECOMPUTED_WIDTH,
            semantic_rung=rung,
            dropout=_freeze.SCORER_DROPOUT,
            temperature=_freeze.SCORER_TEMPERATURE,
            embedding_dim=runner.FROZEN_EMBEDDING_DIM,
            semantic_head=head,
        )
        assert model.semantic_parameter_count() == declared["semantic"], rung
        assert model.trainable_parameter_count() == declared["total"], rung
        assert len(head.feature_names) == declared["semantic_columns"], rung


def test_the_frozen_width_agrees_with_the_formula_freeze() -> None:
    assert runner.FROZEN_EMBEDDING_DIM == _freeze.FROZEN_EMBEDDING_DIM


def test_s4_is_two_projections_of_the_real_width_not_of_768() -> None:
    """The 98,304 trap: 2 * 768 * 64 is half of what this rung actually holds."""

    assert runner.DECLARED_PARAMETERS["S4"]["semantic"] == 2 * 1536 * 64
    assert runner.DECLARED_PARAMETERS["S4"]["semantic"] == 2 * (2 * 768 * 64)


def test_a_rung_that_changed_width_is_refused() -> None:
    with pytest.raises(ValueError, match="unrecoverable after the fact"):
        runner.check_declared_parameters("S2", {"semantic": 0, "total": 450}, 3)


def test_the_declared_counts_are_accepted_when_they_hold() -> None:
    declared = runner.DECLARED_PARAMETERS["S4"]
    runner.check_declared_parameters(
        "S4",
        {"semantic": declared["semantic"], "total": declared["total"]},
        declared["semantic_columns"],
    )


def test_a_toy_width_fit_is_not_held_to_the_frozen_counts(cell) -> None:
    """Otherwise this whole fixture could not run, and the guard would be untested."""

    for fit in cell["rungs"].values():
        assert fit["semantic_rung_fingerprint"]["embedding_dim"] != runner.FROZEN_EMBEDDING_DIM
    assert cell["rungs"]["S4"]["parameters"]["total"] != runner.DECLARED_PARAMETERS["S4"]["total"]


def test_s2s_parameter_count_does_not_depend_on_the_embedding_width(cell) -> None:
    """A fact worth stating plainly, because it is half of what S2 is.

    S2 holds no semantic parameters, so its 449 are the scorer's alone and the
    embedding width never enters them. The toy fixture runs at width 6 and
    reproduces the frozen total exactly. S4's 196,608 are two projections OF
    that width, and at width 6 it weighs almost nothing -- which is why the
    declared-count guard fires only at 1536.
    """

    assert cell["rungs"]["S2"]["parameters"]["semantic"] == 0
    assert cell["rungs"]["S2"]["parameters"]["total"] == runner.DECLARED_PARAMETERS["S2"]["total"]
    assert cell["rungs"]["S4"]["parameters"]["semantic"] == 2 * 6 * 64


# --------------------------------------------------------------------------
# The rung factory
# --------------------------------------------------------------------------


def test_each_rung_name_builds_the_module_the_declaration_names() -> None:
    assert type(runner.build_semantic_head("S2", 8)).__name__ == "SemanticHead"
    assert type(runner.build_semantic_head("S3", 8)).__name__ == "SemanticHead"
    assert type(runner.build_semantic_head("S4", 8)).__name__ == "ProjectionSemanticHead"
    assert runner.build_semantic_head("S2", 8).rung == "S2"
    assert runner.build_semantic_head("S4", 8).rung == "S4"


def test_an_unknown_rung_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown semantic rung"):
        runner.build_semantic_head("S5", 8)


def test_the_fingerprint_separates_rungs_and_widths() -> None:
    s2 = runner.semantic_rung_fingerprint(runner.build_semantic_head("S2", 64))
    s4_small = runner.semantic_rung_fingerprint(runner.build_semantic_head("S4", 64))
    s4_large = runner.semantic_rung_fingerprint(runner.build_semantic_head("S4", 128))
    assert s2["sha256"] != s4_small["sha256"]
    assert s4_small["sha256"] != s4_large["sha256"]
    assert s4_small["sha256"] == runner.semantic_rung_fingerprint(
        runner.build_semantic_head("S4", 64)
    )["sha256"]


def test_the_fingerprint_names_the_module_it_came_from() -> None:
    fingerprint = runner.semantic_rung_fingerprint(runner.build_semantic_head("S4", 64))
    assert fingerprint["module"] == "mp_retrieval.m2b_semantic_control"
    assert fingerprint["projection_dim"] == 64


# --------------------------------------------------------------------------
# Instrumentation the declaration requires
# --------------------------------------------------------------------------


def test_every_fit_is_addressable_on_its_own(smoke, cell) -> None:
    """28 fits sharing 6 containers must not become 6 results."""

    _result, args = smoke
    for rung in cell["rungs"]:
        fit_root = Path(args.artifact_root) / TOY_REGIME / rung.lower()
        assert (fit_root / "fit.json").is_file()
        assert (fit_root / "checkpoint.pt").is_file()
        assert (fit_root / "per_query_rows.json").is_file()
        standalone = json.loads((fit_root / "fit.json").read_text(encoding="utf-8"))
        assert standalone["semantic_rung"] == rung
        assert standalone["metrics"] == cell["rungs"][rung]["metrics"]


def test_the_rows_reproduce_the_aggregate_for_every_metric(smoke, cell) -> None:
    from scripts.run_operator_screen import _aggregate_rows

    _result, args = smoke
    for rung, fit in cell["rungs"].items():
        rows = json.loads(
            (Path(args.artifact_root) / TOY_REGIME / rung.lower() / "per_query_rows.json")
            .read_text(encoding="utf-8")
        )["rows"]
        assert _m2._aggregate_mismatch(fit["metrics"], _aggregate_rows(rows)) == [], rung


def test_every_declared_instrumentation_field_is_present_and_non_null(cell) -> None:
    """The declaration's list, checked against the artifact one name at a time.

    A rerun cannot recover a field a completed fit did not write, because the
    rerun would be a different fit on different hardware. So the names have to
    match literally, not after a mental translation.
    """

    declared = M2B_DECLARATION["instrumentation_requirement"]["fields"]
    for rung, fit in cell["rungs"].items():
        for field in declared:
            assert field in fit["instrumentation"], (rung, field)
            assert fit["instrumentation"][field] is not None, (rung, field)


def test_the_declared_field_list_is_not_silently_shrinking() -> None:
    declared = M2B_DECLARATION["instrumentation_requirement"]["fields"]
    assert len(declared) == 18
    assert "semantic_rung_fingerprint" in declared
    assert "uncached_inference_p95_ms" in declared


def test_peak_vram_covers_training_and_inference(cell) -> None:
    for rung, fit in cell["rungs"].items():
        peaks = fit["instrumentation"]["peak_gpu_memory_mb"]
        assert fit["instrumentation"]["peak_vram_mb"] == max(
            float(peaks["training_total"]), float(peaks["uncached_inference_total"])
        ), rung


def test_the_uncached_timing_covers_the_whole_model(cell) -> None:
    for rung, fit in cell["rungs"].items():
        latency = fit["uncached_inference"]
        assert latency["measured_span"].startswith("raw query and candidate embeddings"), rung
        assert latency["total_model_ms"]["n"] > 0, rung
        assert fit["systems"]["tie_break_orders_on"] == "uncached_inference_p95_ms"
        assert (
            fit["systems"]["uncached_inference_p99_ms"]
            >= fit["systems"]["uncached_inference_p95_ms"]
            >= fit["systems"]["uncached_inference_p50_ms"]
        ), rung


def test_the_feature_build_latency_the_fit_did_not_measure_is_carried(cell) -> None:
    for rung, fit in cell["rungs"].items():
        carried = fit["instrumentation"]["uncached_feature_build_latency_ms"]
        assert carried == cell["uncached_feature_build_latency_ms"], rung
        for percentile in ("p50", "p95", "p99"):
            assert percentile in carried, rung


def test_peak_memory_is_recorded_for_training_and_for_inference(cell) -> None:
    for rung, fit in cell["rungs"].items():
        peaks = fit["instrumentation"]["peak_gpu_memory_mb"]
        assert set(peaks) == {
            "training_total", "uncached_inference_total", "batched_inference_total"
        }, rung
        assert fit["systems"]["peak_rss_mb"] > 0, rung


def test_every_score_was_checked_for_nan(cell) -> None:
    for rung, fit in cell["rungs"].items():
        assert fit["held_out_scores"]["all_finite"] is True, rung
        assert fit["held_out_scores"]["scored_values"] > 0, rung


def test_a_model_that_emits_nan_is_caught_even_though_ranking_would_survive(smoke) -> None:
    """Metrics alone would not catch it: argsort of a NaN vector is a permutation."""

    _result, args = smoke
    store = _load_store(args)
    queries, node_embeddings, query_embeddings = _load_queries(args)

    class _NaNModel(torch.nn.Module):
        training = False

        def eval(self):
            return self

        def train(self, mode=True):
            return self

        def forward_explicit(self, nodes, *_rest):
            return torch.full((nodes.shape[0],), float("nan"))

    with pytest.raises(ValueError, match="non-finite score"):
        runner.finite_scores(
            _NaNModel(), queries[:1], node_embeddings, query_embeddings,
            store, torch.device("cpu"),
        )


def test_the_checkpoint_reloads_into_the_scores_it_reported(cell) -> None:
    for rung, fit in cell["rungs"].items():
        round_trip = fit["checkpoint_round_trip"]
        assert round_trip["loaded_strict"] is True, rung
        assert round_trip["max_absolute_score_difference"] == 0.0, rung


def test_a_checkpoint_that_is_not_this_model_is_caught(smoke, cell, tmp_path) -> None:
    """The round-trip check has to be able to fail, or it certifies nothing."""

    _result, args = smoke
    store = _load_store(args)
    queries, node_embeddings, query_embeddings = _load_queries(args)
    width = cell["precomputed_width"]
    dim = int(node_embeddings.shape[1])

    model = _m1a.build_m1a_model(
        precomputed_width=width, semantic_rung="S2", dropout=0.0, temperature=1.0,
        embedding_dim=dim, semantic_head=runner.build_semantic_head("S2", dim),
    )
    wrong = {key: torch.randn_like(value) for key, value in model.state_dict().items()}
    path = tmp_path / "wrong.pt"
    torch.save(wrong, path)

    with pytest.raises(ValueError, match="not the model that was measured"):
        runner.checkpoint_round_trips(
            path, model, rung="S2", precomputed_width=width, query=queries[0],
            node_embeddings=node_embeddings, query_embeddings=query_embeddings,
            store=store, device=torch.device("cpu"), dropout=0.0, temperature=1.0,
        )


# --------------------------------------------------------------------------
# Reuse is real reuse or it is refused
# --------------------------------------------------------------------------


def test_s3_without_m2s_fits_is_refused_not_refit(tmp_path) -> None:
    """A fresh fit reported in the reused column would be a fabricated reuse."""

    args = _m2b_args(tmp_path, rungs=["S3"], m2_fits_root=None)
    with pytest.raises(ValueError, match="fabricated reuse"):
        runner.reused_checkpoint_path(args, TOY_REGIME)


def test_s3_points_at_the_slug_m2_actually_wrote(tmp_path) -> None:
    args = _m2b_args(tmp_path, m2_fits_root=tmp_path / "m2_fits")
    path = runner.reused_checkpoint_path(args, "R3")
    assert path == tmp_path / "m2_fits" / "R3" / "qls_universal" / "checkpoint.pt"
    assert _m2.arm_slug(runner.DECLARED_UNIVERSAL_ARM) == "qls_universal"


def test_a_missing_m2_checkpoint_is_refused_rather_than_refit(smoke, cell, tmp_path) -> None:
    _result, args = smoke
    store = _load_store(args)
    queries, node_embeddings, query_embeddings = _load_queries(args)
    with pytest.raises(FileNotFoundError, match="fabricated reuse"):
        runner.fit_one_rung(
            rung="S3", regime=TOY_REGIME, store=store,
            precomputed_width=cell["precomputed_width"],
            train_queries=queries, validation_queries=queries,
            node_embeddings=node_embeddings, query_embeddings=query_embeddings,
            device=torch.device("cpu"), args=args, fit_root=tmp_path / "s3",
            provenance={"source_commit": "0" * 40, "config_sha256": "x",
                        "dataset_fingerprint_sha256": "y", "candidate_contract_sha256": "z",
                        "candidate_id_order_sha256": "w"},
            feature_build_latency_ms={"p50": 1.0, "p95": 1.0, "p99": 1.0},
            shared_inputs={"sha256": "s", "arm_store_sha256": "a",
                           "cell_features_sha256": "c", "feature_build_contract_sha256": "f"},
            reused_checkpoint=tmp_path / "absent" / "checkpoint.pt",
        )


def test_a_reused_fit_reports_zero_training_and_says_why(smoke, cell, tmp_path) -> None:
    """S3's weights are M2's; its latency is this container's. Both are stated."""

    _result, args = smoke
    store = _load_store(args)
    queries, node_embeddings, query_embeddings = _load_queries(args)
    dim = int(node_embeddings.shape[1])
    donor = _m1a.build_m1a_model(
        precomputed_width=cell["precomputed_width"], semantic_rung="S3",
        dropout=args.dropout, temperature=args.temperature, embedding_dim=dim,
        semantic_head=runner.build_semantic_head("S3", dim),
    )
    checkpoint = tmp_path / "m2" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(donor.state_dict(), checkpoint)

    fit = runner.fit_one_rung(
        rung="S3", regime=TOY_REGIME, store=store,
        precomputed_width=cell["precomputed_width"],
        train_queries=queries, validation_queries=queries,
        node_embeddings=node_embeddings, query_embeddings=query_embeddings,
        device=torch.device("cpu"), args=args, fit_root=tmp_path / "s3",
        provenance={"source_commit": "0" * 40, "config_sha256": "x",
                    "dataset_fingerprint_sha256": "y", "candidate_contract_sha256": "z",
                    "candidate_id_order_sha256": "w"},
        feature_build_latency_ms={"p50": 1.0, "p95": 1.0, "p99": 1.0},
        shared_inputs={"sha256": "s", "arm_store_sha256": "a",
                       "cell_features_sha256": "c", "feature_build_contract_sha256": "f"},
        reused_checkpoint=checkpoint,
    )
    assert fit["reused_from_m2"] is True
    assert fit["training"]["reused"] is True
    assert fit["training"]["training_seconds"] == 0.0
    assert "M2 paid for this fit" in fit["training"]["why_zero"]
    # Reused weights, freshly measured latency: the tie-break number is ours.
    assert fit["uncached_inference"]["total_model_ms"]["n"] > 0
    assert fit["systems"]["uncached_inference_p95_ms"] > 0


def test_the_reused_weights_are_loaded_not_retrained(smoke, cell, tmp_path) -> None:
    """The donor's scores must come back out, or the reuse is a refit in disguise."""

    _result, args = smoke
    store = _load_store(args)
    queries, node_embeddings, query_embeddings = _load_queries(args)
    dim = int(node_embeddings.shape[1])
    donor = _m1a.build_m1a_model(
        precomputed_width=cell["precomputed_width"], semantic_rung="S3",
        dropout=args.dropout, temperature=args.temperature, embedding_dim=dim,
        semantic_head=runner.build_semantic_head("S3", dim),
    )
    checkpoint = tmp_path / "m2" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(donor.state_dict(), checkpoint)

    runner.fit_one_rung(
        rung="S3", regime=TOY_REGIME, store=store,
        precomputed_width=cell["precomputed_width"],
        train_queries=queries, validation_queries=queries,
        node_embeddings=node_embeddings, query_embeddings=query_embeddings,
        device=torch.device("cpu"), args=args, fit_root=tmp_path / "s3",
        provenance={"source_commit": "0" * 40, "config_sha256": "x",
                    "dataset_fingerprint_sha256": "y", "candidate_contract_sha256": "z",
                    "candidate_id_order_sha256": "w"},
        feature_build_latency_ms={"p50": 1.0, "p95": 1.0, "p99": 1.0},
        shared_inputs={"sha256": "s", "arm_store_sha256": "a",
                       "cell_features_sha256": "c", "feature_build_contract_sha256": "f"},
        reused_checkpoint=checkpoint,
    )
    saved = torch.load(tmp_path / "s3" / "checkpoint.pt", map_location="cpu", weights_only=True)
    for key, value in donor.state_dict().items():
        assert torch.equal(saved[key], value), key


# --------------------------------------------------------------------------
# Admitting M2's stores
# --------------------------------------------------------------------------


def test_a_drifted_declaration_hash_does_not_stop_the_load(smoke) -> None:
    """The reason feature_build_contract exists, asserted on a real store."""

    _result, args = smoke
    metadata_path = Path(args.artifact_root) / TOY_REGIME / "cell_features" / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    original = metadata["build_key"]["config_sha256"]
    metadata["build_key"]["config_sha256"] = "d" * 64
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        scored, master, latency, fingerprint, decision = runner.load_cell_under_contract(
            metadata_path.parent, args, TOY_REGIME,
            candidate_contract={
                "observed_contract_sha256": args.baseline["candidate_contract_sha256"],
                "candidate_id_order_sha256": "unused-by-the-scientific-key",
            },
        )
    finally:
        metadata["build_key"]["config_sha256"] = original
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    assert decision["admitted"] is True
    assert decision["original_full_config_sha256"] == "d" * 64
    assert scored and master and latency and fingerprint


def test_m2s_sealed_fits_are_read_and_never_written_into(smoke, tmp_path) -> None:
    """M2's artifacts are immutable inputs, so M2B writes to its own root.

    Pointing --cell-master-root at a copy of M2's tree and --artifact-root
    somewhere else must produce the same fits and leave the master tree
    byte-identical.
    """

    _result, args = smoke
    sealed = tmp_path / "m2_sealed" / TOY_REGIME / "cell_features"
    sealed.mkdir(parents=True, exist_ok=True)
    for item in (Path(args.artifact_root) / TOY_REGIME / "cell_features").iterdir():
        (sealed / item.name).write_bytes(item.read_bytes())
    before = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
              for item in sorted(sealed.iterdir())}

    apart = _m2b_args(tmp_path, source_commit=args.source_commit)
    apart.data = args.data
    apart.baseline = args.baseline
    apart.expected_queries = args.expected_queries
    apart.cell_master_root = tmp_path / "m2_sealed"
    apart.artifact_root = tmp_path / "m2b_writes"
    apart.output = tmp_path / "apart.json"
    result = runner.run(apart)

    after = {item.name: hashlib.sha256(item.read_bytes()).hexdigest()
             for item in sorted(sealed.iterdir())}
    assert after == before, "M2B wrote into the cell-master tree it was only meant to read"
    assert not (tmp_path / "m2_sealed" / TOY_REGIME / "arm_store").exists()
    assert (tmp_path / "m2b_writes" / TOY_REGIME / "s4" / "fit.json").is_file()
    assert result["cell_master_root"] == str(tmp_path / "m2_sealed")
    for rung, fit in result["cells"][TOY_REGIME]["rungs"].items():
        assert fit["metrics"] == _result["cells"][TOY_REGIME]["rungs"][rung]["metrics"], rung


def test_a_store_built_for_another_cell_is_refused(smoke) -> None:
    _result, args = smoke
    root = Path(args.artifact_root) / TOY_REGIME / "cell_features"
    other = _m2b_args(Path(args.output).parent.parent, queries=args.queries + 1)
    other.artifact_root = args.artifact_root
    with pytest.raises(ValueError, match="different cell"):
        runner.load_cell_under_contract(
            root, other, TOY_REGIME,
            candidate_contract={
                "observed_contract_sha256": args.baseline["candidate_contract_sha256"],
                "candidate_id_order_sha256": "x",
            },
        )


def test_a_missing_store_is_refused_with_a_readable_error(tmp_path) -> None:
    args = _m2b_args(tmp_path)
    with pytest.raises(FileNotFoundError, match="never builds features"):
        runner.load_cell_under_contract(
            tmp_path / "nowhere", args, TOY_REGIME,
            candidate_contract={"observed_contract_sha256": "a", "candidate_id_order_sha256": "b"},
        )


def test_a_store_whose_arrays_moved_is_refused(smoke, tmp_path) -> None:
    _result, args = smoke
    source = Path(args.artifact_root) / TOY_REGIME / "cell_features"
    copy = tmp_path / "cell_features"
    copy.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        (copy / item.name).write_bytes(item.read_bytes())
    master = np.load(copy / "master_flat.npy")
    master[0, 0] += 1.0
    np.save(copy / "master_flat.npy", master)
    with pytest.raises(ValueError, match="do not hash to the fingerprint"):
        runner.load_cell_under_contract(
            copy, args, TOY_REGIME,
            candidate_contract={
                "observed_contract_sha256": args.baseline["candidate_contract_sha256"],
                "candidate_id_order_sha256": "x",
            },
        )


def test_the_admission_decision_travels_with_the_cell(cell) -> None:
    decision = cell["store_admission"]
    assert decision["admitted"] is True
    assert decision["evidence"] in ("recorded", "reconstructed")
    assert len(decision["feature_build_contract_sha256"]) == 64
    assert "cannot move a float" in decision["declaration_hash_drift_is_not_a_refusal"]


# --------------------------------------------------------------------------
# The shared-inputs digest is complete
# --------------------------------------------------------------------------


def _shared(**overrides) -> dict:
    base = {
        "cell_features_sha256": "a" * 64,
        "arm_store_sha256": "b" * 64,
        "feature_build_contract_sha256": "c" * 64,
        "precomputed_width": 9,
        "query_ids": ["q1", "q2"],
        "family_digests": {"BASE": "d" * 64, "NODE_ROLE": "e" * 64},
        "candidate_contract": {
            "observed_contract_sha256": "f" * 64,
            "candidate_id_order_sha256": "0" * 64,
        },
    }
    base.update(overrides)
    return runner.shared_inputs_of_cell(**base)


@pytest.mark.parametrize(
    "field,changed",
    [
        ("cell_features_sha256", "z" * 64),
        ("arm_store_sha256", "z" * 64),
        ("feature_build_contract_sha256", "z" * 64),
        ("precomputed_width", 14),
        ("query_ids", ["q1", "q3"]),
        ("family_digests", {"BASE": "d" * 64, "NODE_ROLE": "z" * 64}),
        (
            "candidate_contract",
            {"observed_contract_sha256": "z" * 64, "candidate_id_order_sha256": "0" * 64},
        ),
    ],
)
def test_changing_any_shared_input_changes_the_digest(field, changed) -> None:
    assert _shared(**{field: changed})["sha256"] != _shared()["sha256"], field


def test_the_same_inputs_give_the_same_digest() -> None:
    assert _shared()["sha256"] == _shared()["sha256"]


def test_reordering_the_held_out_queries_is_a_different_cell() -> None:
    """Order matters: the rows are compared position by position downstream."""

    assert _shared(query_ids=["q2", "q1"])["sha256"] != _shared(query_ids=["q1", "q2"])["sha256"]


def test_no_field_enters_the_shared_digest_untested() -> None:
    """A field added without a paired must-catch test would silently not be checked."""

    covered = {
        "cell_features_sha256", "arm_store_sha256", "feature_build_contract_sha256",
        "precomputed_width", "held_out_query_ids_sha256", "held_out_query_count",
        "structural_family_digests", "candidate_contract_sha256",
        "candidate_id_order_sha256", "candidate_normalisation", "sha256",
    }
    assert set(_shared()) == covered, (
        "a field was added to or removed from the shared-inputs digest; give it a "
        "must-catch test above before updating this set"
    )


def test_the_candidate_normalisation_comes_from_the_contract_not_a_literal() -> None:
    assert (
        _shared()["candidate_normalisation"]
        == fbc.formula_identity()["feature_normalisation"]
        == "candidate"
    )


# --------------------------------------------------------------------------
# The matrix is M2's, and the guards hold
# --------------------------------------------------------------------------


def test_the_matrix_is_exactly_m2s_cells() -> None:
    """The declaration says cells_are_m2s; this is that sentence as a check."""

    m2_cells = {
        dataset: sorted(regimes)
        for dataset, regimes in M2_DECLARATION["m2_selection_matrix"]["cells"].items()
    }
    m2b_cells = {
        dataset: sorted(regimes)
        for dataset, regimes in M2B_DECLARATION["evaluation_matrix"]["cells"].items()
    }
    assert m2b_cells == m2_cells
    assert sum(len(regimes) for regimes in m2b_cells.values()) == 14


def test_the_runner_reads_the_matrix_rather_than_trusting_a_flag() -> None:
    assert runner.declared_cells("2wiki_clean") == ["R1", "R2", "R3"]
    assert runner.declared_cells("squad_clean") == ["R1"]
    assert runner.declared_cells("musique_clean") == ["R1"]


def test_an_undeclared_dataset_is_refused() -> None:
    with pytest.raises(ValueError, match="not in M2B's evaluation matrix"):
        runner.declared_cells("nq_open")


def test_an_undeclared_regime_is_refused(tmp_path) -> None:
    args = _m2b_args(tmp_path, dataset="squad_clean", regimes=["R3"])
    with pytest.raises(ValueError, match="not declared"):
        runner.run(args)


def test_an_unknown_rung_is_refused_before_any_work(tmp_path) -> None:
    args = _m2b_args(tmp_path, rungs=["S9"])
    with pytest.raises(ValueError, match="not M2B's rungs"):
        runner.run(args)


def test_a_non_zero_seed_is_refused(tmp_path) -> None:
    args = _m2b_args(tmp_path, seed=1)
    with pytest.raises(ValueError, match="seed-0 screen"):
        runner.run(args)


def test_the_parser_refuses_a_non_zero_seed_too(tmp_path) -> None:
    with pytest.raises(SystemExit):
        runner.main([
            "--data", str(tmp_path), "--dataset", TOY_DATASET,
            "--data-fingerprint-sha256", "0" * 64, "--expected-queries", "1",
            "--frozen-embedding-dim", "1536", "--baseline", str(tmp_path / "b.json"),
            "--seed", "1", "--output", str(tmp_path / "o.json"),
        ])


def test_run_persists_its_own_result_because_main_is_not_the_caller(smoke) -> None:
    """The Modal container calls run() directly and never calls main().

    A smoke that fitted three rungs on a GPU, returned them, and left nothing
    on the volume is what this test exists to stop: the write lived in main(),
    so the fits were real and the aggregate was unrecoverable.
    """

    result, args = smoke
    assert args.output.is_file(), "run() returned a result it did not write down"
    assert json.loads(args.output.read_text(encoding="utf-8")) == result


def test_a_completed_run_is_not_repeated(smoke, monkeypatch) -> None:
    """Idempotence has to hold for run(), not just for the command line.

    The check reads the file run() writes, so with the write in main() a
    container could never resume its own completed work either.
    """

    result, args = smoke

    def _refuse(*fn_args, **fn_kwargs):
        raise AssertionError("a completed cell must not be loaded again, let alone refit")

    monkeypatch.setattr(runner, "load_cell_under_contract", _refuse)
    again = runner.run(args)
    assert again["status"] == runner.STATUS_COMPLETE
    assert again["cells"].keys() == result["cells"].keys()
    assert again == result


def test_the_rungs_the_declaration_calls_new_are_the_ones_fit(smoke) -> None:
    result, _args = smoke
    declared_new = M2B_DECLARATION["workload"]["new_by_rung"]
    assert sorted(runner.NEW_RUNGS) == sorted(declared_new)
    assert runner.REUSED_RUNG not in declared_new
    assert result["new_rungs"] == SMOKE_RUNGS


def test_the_split_and_selection_come_from_the_contract(smoke) -> None:
    result, _args = smoke
    assert result["split"] == fbc.QUERY_SPLIT == "validation"
    assert result["selection"] == fbc.QUERY_SELECTION


# --------------------------------------------------------------------------
# Shared helpers for the tests that need the fixture's own objects
# --------------------------------------------------------------------------


def _load_store(args):
    """Rebuild the fixture's arm store from what the run persisted."""

    from mp_retrieval.structural_features import StructuralFeatureStore

    return StructuralFeatureStore.load(Path(args.artifact_root) / TOY_REGIME / "arm_store")


def _load_queries(args):
    """The widened held-out queries and embedding tables the fixture fit on."""

    from mp_retrieval.complete_data import load_complete_dataset
    from mp_retrieval.data import QuerySplit
    from scripts.run_graph_context_d1 import holdout_split

    dataset = load_complete_dataset(args.data, dataset=args.dataset, require_embeddings=True)
    queries = dataset.split(QuerySplit.VALIDATION)[: args.queries]
    scored_sets, _master = _m2.load_cell_features(
        Path(args.artifact_root) / TOY_REGIME / "cell_features"
    )
    widened = [
        _m1a._widen_query(query, scored)
        for query, scored in zip(queries, scored_sets, strict=True)
    ]
    _train, held_out = holdout_split(widened, args.holdout_fraction)
    node_embeddings = torch.from_numpy(np.array(dataset.node_array, dtype=np.float32, copy=True))
    query_embeddings = torch.from_numpy(np.array(dataset.query_array, dtype=np.float32, copy=True))
    return held_out, node_embeddings, query_embeddings


# --------------------------------------------------------------------------
# The per-query outcomes the paired bootstrap needs
# --------------------------------------------------------------------------


def test_every_rung_records_a_per_query_outcome_aligned_to_the_cell_panel(cell):
    # M1B needed an amendment because no per-query outcome had been persisted
    # anywhere and its bootstrap was not computable after the fact. M2B pays
    # that cost up front, while the fit is running.
    panel = cell["held_out_query_ids"]
    assert len(panel) == cell["held_out_queries"]
    for rung, fit in cell["rungs"].items():
        assert len(fit["per_query_recall_at_5"]) == len(panel), rung


def test_the_per_query_vector_reproduces_the_reported_recall(cell):
    for rung, fit in cell["rungs"].items():
        vector = fit["per_query_recall_at_5"]
        assert sum(vector) / len(vector) == pytest.approx(
            fit["metrics"]["recall@5"], abs=1e-9
        ), rung


def test_the_panel_is_written_once_and_is_the_same_list_for_every_rung(cell, tmp_path):
    # The paired premise: one query id order, shared. Each rung's own rows file
    # is checked against it rather than the cell's copy being trusted.
    panel = cell["held_out_query_ids"]
    for rung, fit in cell["rungs"].items():
        rows = json.loads(
            Path(fit["instrumentation"]["per_query_rows"]).read_text(encoding="utf-8")
        )
        assert rows["query_ids"] == panel, rung
        assert [row["recall@5"] for row in rows["rows"]] == fit["per_query_recall_at_5"]
