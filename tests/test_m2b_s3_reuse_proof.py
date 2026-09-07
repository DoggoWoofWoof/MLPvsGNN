"""Tests for the gate that decides whether M2's 14 S3 fits may be reused.

M2B reuses fourteen already-paid S3 results rather than refitting them, which
is only legitimate if the code that would build an S3 model today is the same
function as the code whose containers produced those numbers. Between then and
now ``M1AScorer.__init__`` gained a ``semantic_head`` injection point so S4 can
be passed in. That edit is exactly the kind of change that silently invalidates
reuse, so ``scripts/m2b_s3_reuse_proof.py`` proves behaviour equivalence
against the pre-injection source pulled straight from git.

Two failure modes matter here and neither is caught by running the proof:

* the proof could pass because it is comparing the current file to itself, and
* the proof could pass because its equality checks cannot fail.

So this module checks the baseline really is a different file, mutates the
current implementation and requires the verdict to flip to
``S3_REUSE_FORBIDDEN`` with a non-zero exit, and asserts the proof's
transcribed ``FROZEN`` construction against ``qls_universal`` in the M2
declaration rather than trusting that the transcription was faithful.
"""

from __future__ import annotations

import hashlib
import inspect
import pathlib
import re
import subprocess
import sys

import pytest
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import m2b_s3_reuse_proof as proof  # noqa: E402

from mp_retrieval.m1a_screen import M1AScorer  # noqa: E402

M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"

#: Where a real M2 checkpoint is kept if one has been pulled off the volume.
#: outputs/ is gitignored, so the trained-weights tests skip when it is absent
#: rather than pretending a fresh clone can run them.
REFERENCE_CHECKPOINT = (
    REPO_ROOT / "outputs" / "m2b_semantic_minimality" / "reference"
    / "m2_2wiki_R3_s3_checkpoint.pt"
)


@pytest.fixture(scope="module")
def declaration() -> dict:
    return yaml.safe_load(M2_DECLARATION.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def universal(declaration: dict) -> dict:
    return declaration["qls_universal"]


@pytest.fixture(scope="module")
def baseline_class():
    module, _source = proof.load_baseline_module(proof.M2_FIT_COMMIT)
    return module.M1AScorer


# --------------------------------------------------------------------------
# The transcribed construction is checked against the declaration, not trusted
# --------------------------------------------------------------------------


def test_frozen_precomputed_width_is_the_declared_one(universal: dict) -> None:
    assert proof.FROZEN["precomputed_width"] == universal["feature_schema"][
        "precomputed_width"
    ]


def test_frozen_dropout_and_temperature_are_the_declared_ones(universal: dict) -> None:
    assert proof.FROZEN["dropout"] == universal["hyperparameters"]["dropout"]
    assert proof.FROZEN["temperature"] == universal["hyperparameters"]["temperature"]


def test_frozen_rung_and_dim_come_from_the_declared_semantic_head(universal: dict) -> None:
    """The rung and width are only written down inside one prose string."""

    spec = universal["architecture"]["semantic_head"]
    rung = re.search(r"rung=(\w+)", spec)
    dim = re.search(r"dim=(\d+)", spec)
    assert rung and dim, f"cannot read rung/dim out of {spec!r}"
    assert proof.FROZEN["semantic_rung"] == rung.group(1)
    assert proof.FROZEN["embedding_dim"] == int(dim.group(1))


def test_the_frozen_construction_reproduces_the_declared_parameter_count(
    universal: dict,
) -> None:
    """If FROZEN were wrong in any field, the count would move off 3,585."""

    model = proof.build(M1AScorer)
    assert sum(p.numel() for p in model.parameters()) == universal["parameter_count"][
        "total"
    ]


def test_frozen_has_no_field_the_scorer_does_not_accept() -> None:
    accepted = set(inspect.signature(M1AScorer.__init__).parameters)
    assert set(proof.FROZEN) <= accepted, (
        f"FROZEN carries {sorted(set(proof.FROZEN) - accepted)}, which M1AScorer would "
        "reject or ignore"
    )


# --------------------------------------------------------------------------
# The baseline is a real, different, pre-injection file
# --------------------------------------------------------------------------


def test_the_baseline_commit_is_an_ancestor_of_head() -> None:
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", proof.M2_FIT_COMMIT, "HEAD"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, (
        f"{proof.M2_FIT_COMMIT} is not in this history; the proof would be comparing "
        "against a commit this working tree never produced"
    )


def test_the_baseline_source_really_differs_from_the_current_one() -> None:
    """Otherwise every equality below passes for the trivial reason."""

    _module, source = proof.load_baseline_module(proof.M2_FIT_COMMIT)
    current = (REPO_ROOT / proof.BASELINE_SOURCE).read_text(encoding="utf-8")
    baseline_digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    current_digest = hashlib.sha256(current.encode("utf-8")).hexdigest()
    assert baseline_digest != current_digest, (
        "the pre-injection baseline and the current file are byte-identical, so this "
        "proof establishes nothing"
    )


def test_the_baseline_predates_the_injection(baseline_class) -> None:
    """Asked of the live class: the string appears in both files either way."""

    assert "semantic_head" not in inspect.signature(baseline_class.__init__).parameters
    assert "semantic_head" in inspect.signature(M1AScorer.__init__).parameters, (
        "the current scorer has no injection point, so there is nothing to prove "
        "equivalence across and S4 cannot be built"
    )


def test_the_baseline_loads_the_real_semantic_module(baseline_class) -> None:
    """A baseline carrying its own copy of SemanticHead would prove nothing."""

    from mp_retrieval.qls_v2_semantic import SemanticHead

    model = proof.build(baseline_class)
    assert isinstance(model.semantic_head, SemanticHead)


# --------------------------------------------------------------------------
# The probe exercises the thing that could actually break
# --------------------------------------------------------------------------


def test_the_probe_uses_more_than_one_query() -> None:
    """forward_explicit loops over batch positions; one query never concatenates."""

    assert proof.PROBE_QUERIES > 1


def test_the_captured_scorer_input_is_the_full_concatenated_width(universal: dict) -> None:
    model = proof.build(M1AScorer)
    nodes, queries, batch_index, structural = proof.probe_inputs(
        proof.FROZEN["embedding_dim"], proof.FROZEN["precomputed_width"]
    )
    scores, scorer_input, semantic = proof.capture(
        model, nodes, queries, batch_index, structural
    )
    rows = proof.PROBE_QUERIES * proof.PROBE_CANDIDATES
    assert scorer_input.shape == (rows, universal["feature_schema"]["width"])
    assert semantic.shape == (rows, universal["feature_schema"]["semantic_feature_count"])
    assert scores.shape == (rows,)


def test_the_scorer_input_is_hooked_not_recomputed() -> None:
    """A recomputed input could agree while the real one differs."""

    source = inspect.getsource(proof.capture)
    assert "register_forward_hook" in source


# --------------------------------------------------------------------------
# The equality machinery can fail
# --------------------------------------------------------------------------


class _DriftedScorer(M1AScorer):
    """A scorer that is a different function by exactly one constant."""

    def forward_explicit(self, *args, **kwargs):  # type: ignore[override]
        return super().forward_explicit(*args, **kwargs) + 1.0


def test_compare_detects_a_changed_forward() -> None:
    equalities = proof.compare(proof.build(M1AScorer), proof.build(_DriftedScorer))
    assert equalities["state_dict_keys_identical"] is True
    assert equalities["forward_scores_identical"] is False
    assert equalities["max_absolute_score_difference"] == pytest.approx(1.0)


def test_compare_detects_a_changed_semantic_block() -> None:
    """Same scorer, different semantic head: the concatenated input must move."""

    from mp_retrieval.qls_v2_semantic import SemanticHead

    class _DriftedSemantic(SemanticHead):
        def forward(self, *args, **kwargs):  # type: ignore[override]
            return super().forward(*args, **kwargs) + 1.0

    head = _DriftedSemantic(rung="S3", dim=proof.FROZEN["embedding_dim"])
    drifted = proof.build(M1AScorer, semantic_head=head)
    equalities = proof.compare(proof.build(M1AScorer), drifted)
    assert equalities["semantic_output_identical"] is False
    assert equalities["scorer_input_identical"] is False


def test_a_drifted_implementation_flips_the_verdict_and_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: pathlib.Path
) -> None:
    """End to end: if reuse were unsafe, this gate would say so and exit non-zero."""

    monkeypatch.setattr(proof, "M1AScorer", _DriftedScorer)
    report = proof.build_report(proof.M2_FIT_COMMIT, None)
    assert report["verdict"] == proof.VERDICT_FORBIDDEN
    assert "forward_scores_identical" in report["failed_checks"]
    assert "42 new fits" in report["what_the_verdict_means"]

    exit_code = proof.main(["--out", str(tmp_path / "report.json")])
    assert exit_code == 1


def test_the_undrifted_verdict_is_permitted(tmp_path: pathlib.Path) -> None:
    report = proof.build_report(proof.M2_FIT_COMMIT, None)
    assert report["failed_checks"] == []
    assert report["verdict"] == proof.VERDICT_PERMITTED
    assert proof.main(["--out", str(tmp_path / "report.json")]) == 0


def test_a_run_without_a_checkpoint_says_it_used_invented_weights() -> None:
    """The weaker run must not read as though it touched the reused numbers."""

    report = proof.build_report(proof.M2_FIT_COMMIT, None)
    assert report["trained_checkpoint"]["checked"] is False
    assert "invented" in report["trained_checkpoint"]["why_not"]


# --------------------------------------------------------------------------
# Finding the weights inside whatever shape the checkpoint was saved in
# --------------------------------------------------------------------------


@pytest.mark.parametrize("wrapper", ["model_state_dict", "state_dict", "model"])
def test_extract_state_unwraps_the_known_containers(wrapper: str) -> None:
    inner = {"scorer.0.bias": torch.zeros(3)}
    assert proof.extract_state({wrapper: inner, "epoch": 3}) is inner


def test_extract_state_accepts_a_bare_tensor_mapping() -> None:
    """This is the shape M2 actually wrote."""

    bare = {"scorer.0.bias": torch.zeros(3), "scorer.0.weight": torch.zeros(3, 4)}
    assert proof.extract_state(bare) is bare


def test_extract_state_refuses_a_mapping_it_cannot_read() -> None:
    with pytest.raises(SystemExit):
        proof.extract_state({"metrics": {"mrr": 0.5}, "epoch": 3})


def test_extract_state_refuses_a_non_mapping() -> None:
    with pytest.raises(SystemExit):
        proof.extract_state([1, 2, 3])


# --------------------------------------------------------------------------
# Real trained weights, when a checkpoint has been pulled off the volume
# --------------------------------------------------------------------------


requires_checkpoint = pytest.mark.skipif(
    not REFERENCE_CHECKPOINT.is_file(),
    reason=f"no M2 checkpoint at {REFERENCE_CHECKPOINT}; pull one off the volume to run",
)


@requires_checkpoint
def test_the_real_checkpoint_carries_the_declared_parameter_count(universal: dict) -> None:
    state = proof.extract_state(
        torch.load(REFERENCE_CHECKPOINT, map_location="cpu", weights_only=True)
    )
    assert sum(int(v.numel()) for v in state.values()) == universal["parameter_count"][
        "total"
    ]


@requires_checkpoint
def test_the_real_checkpoint_loads_strictly_into_both_implementations(
    baseline_class,
) -> None:
    trained = proof.check_checkpoint(REFERENCE_CHECKPOINT, baseline_class)
    assert trained["loaded_strict"] is True
    assert trained["semantic_output_identical"] is True
    assert trained["scorer_input_identical"] is True
    assert trained["forward_scores_identical"] is True
    assert trained["max_absolute_score_difference"] == 0.0


@requires_checkpoint
def test_a_missing_checkpoint_path_is_refused_rather_than_skipped(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(SystemExit):
        proof.build_report(proof.M2_FIT_COMMIT, tmp_path / "absent.pt")
