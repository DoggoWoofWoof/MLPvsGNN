"""The formula freeze must catch a wrong formula, not merely produce one.

``scripts/m2b_semantic_formula_freeze.py`` states a formula for every semantic
column M2B varies and then checks it by implementing it a second time. The
danger is a check that cannot fail: a tolerance wide enough to admit anything,
a column whose value is zero at initialisation so every formula reproduces it,
or a described width that is derived from the emitted one rather than from the
blocks. Each of those is tested here by breaking the formula on purpose and
requiring the verdict to flip.

The transcribed scorer settings are asserted against ``qls_universal`` rather
than trusted, on the same principle as the rest of this track: a freeze that
describes a model M2 did not fit is a freeze of nothing.
"""

from __future__ import annotations

import pathlib
import re
import sys

import pytest
import torch
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import m2b_semantic_formula_freeze as freeze  # noqa: E402

from mp_retrieval.m2b_semantic_control import ProjectionSemanticHead  # noqa: E402
from mp_retrieval.qls_v2_semantic import RUNG_FEATURES, SemanticHead  # noqa: E402

M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"


@pytest.fixture(scope="module")
def report() -> dict:
    return freeze.build_report()


@pytest.fixture(scope="module")
def universal() -> dict:
    return yaml.safe_load(M2_DECLARATION.read_text(encoding="utf-8"))["qls_universal"]


# --------------------------------------------------------------------------
# The transcribed constants are M2's
# --------------------------------------------------------------------------


def test_the_frozen_width_is_the_one_the_declaration_names(universal: dict) -> None:
    stated = re.search(r"dim=(\d+)", universal["architecture"]["semantic_head"])
    assert stated and freeze.FROZEN_EMBEDDING_DIM == int(stated.group(1))


def test_the_frozen_width_is_not_the_modules_768_default() -> None:
    """The error this whole rung exists to have caught."""

    from mp_retrieval.qls_v2_semantic import EMBEDDING_DIM, V1_SEMANTIC_PARAMETERS

    assert freeze.FROZEN_EMBEDDING_DIM != EMBEDDING_DIM
    assert V1_SEMANTIC_PARAMETERS == 2 * EMBEDDING_DIM * 64 == 98_304
    measured = ProjectionSemanticHead(dim=freeze.FROZEN_EMBEDDING_DIM).parameter_count()
    assert measured == 2 * V1_SEMANTIC_PARAMETERS == 196_608


def test_the_scorer_settings_are_the_declared_ones(universal: dict) -> None:
    assert freeze.SCORER_DROPOUT == universal["hyperparameters"]["dropout"]
    assert freeze.SCORER_TEMPERATURE == universal["hyperparameters"]["temperature"]
    assert freeze.PRECOMPUTED_WIDTH == universal["feature_schema"]["precomputed_width"]


def test_s3_reproduces_the_model_m2_actually_fitted(report: dict, universal: dict) -> None:
    """The row with fifteen completed fits behind it."""

    s3 = report["measured_ladder"]["S3"]
    assert s3["total_parameters"] == universal["parameter_count"]["total"] == 3585
    assert s3["semantic_parameters"] == universal["parameter_count"]["semantic"] == 3072
    assert s3["scorer_input_width"] == universal["feature_schema"]["width"] == 14


# --------------------------------------------------------------------------
# Every column is described, and described correctly
# --------------------------------------------------------------------------


def test_the_verdict_is_frozen_and_nothing_failed(report: dict, tmp_path: pathlib.Path) -> None:
    assert report["failed_checks"] == []
    assert report["verdict"] == "FORMULAS_FROZEN"
    assert freeze.main(["--out", str(tmp_path / "freeze.json")]) == 0


@pytest.mark.parametrize("rung", ["S2", "S3"])
def test_the_described_columns_are_the_modules_own_names(report: dict, rung: str) -> None:
    described = tuple(column["name"] for column in report["rungs"][rung]["columns"])
    assert described == RUNG_FEATURES[rung]


def test_every_scalar_column_carries_a_formula_and_a_parameter_count(report: dict) -> None:
    for rung in ("S2", "S3"):
        for column in report["rungs"][rung]["columns"]:
            assert column["formula"].strip(), (rung, column["name"])
            assert isinstance(column["parameters"], int)
            assert column["formula_reproduces_the_module"] is True


def test_s2_carries_no_learned_parameter_anywhere(report: dict) -> None:
    assert all(column["parameters"] == 0 for column in report["rungs"]["S2"]["columns"])
    assert report["measured_ladder"]["S2"]["semantic_parameters"] == 0


def test_s3s_two_learned_vectors_span_its_whole_parameter_count(report: dict) -> None:
    learned = sum(column["parameters"] for column in report["rungs"]["S3"]["columns"])
    assert learned == report["measured_ladder"]["S3"]["semantic_parameters"] == 3072
    assert learned == 2 * freeze.FROZEN_EMBEDDING_DIM


# --------------------------------------------------------------------------
# 258 is derived, not asserted
# --------------------------------------------------------------------------


def test_the_s4_width_comes_from_the_blocks_not_from_the_number(report: dict) -> None:
    blocks = report["rungs"]["S4"]["blocks"]
    derivation = blocks["width_derivation"]
    assert derivation["blocks"] == len(blocks["output_blocks"]) == 4
    assert derivation["scalars"] == len(blocks["output_scalars"]) == 2
    assert derivation["projection_dim"] == 64
    assert derivation["width"] == 4 * 64 + 2 == 258


def test_the_described_blocks_span_exactly_what_s4_emits(report: dict) -> None:
    verification = report["rungs"]["S4"]["verification"]
    assert verification["widths_agree"] is True
    assert verification["emitted_width"] == verification["described_width"] == 258

    covered: list[int] = []
    for block in verification["per_block"].values():
        start, stop = block["columns"]
        covered.extend(range(start, stop))
    assert covered == list(range(258)), "the blocks overlap or leave a gap"


def test_every_s4_block_reproduces_its_stated_formula(report: dict) -> None:
    per_block = report["rungs"]["S4"]["verification"]["per_block"]
    assert set(per_block) == {
        "query_state", "node_state", "state_product", "state_absolute_difference",
        "normalized_state_dot", "raw_projection_dot_scaled",
    }
    for name, block in per_block.items():
        assert block["formula_reproduces_the_module"] is True, name


def test_the_two_s4_scalars_are_stated_to_use_different_states(report: dict) -> None:
    """One is the normalised dot, the other the raw dot; conflating them is easy."""

    scalars = {entry["name"]: entry for entry in report["rungs"]["S4"]["blocks"]["output_scalars"]}
    assert "qhat_k * dhat_k" in scalars["normalized_state_dot"]["formula"]
    assert "g_k * h_k" in scalars["raw_projection_dot_scaled"]["formula"]
    assert "sqrt(P)" in scalars["raw_projection_dot_scaled"]["formula"]
    assert "before normalisation" in scalars["raw_projection_dot_scaled"]["note"]


def test_the_projections_are_recorded_as_bias_free(report: dict) -> None:
    blocks = report["rungs"]["S4"]["blocks"]
    assert "bias=False" in blocks["step_1_query_projection"]["formula"]
    assert "bias=False" in blocks["step_2_candidate_projection"]["formula"]
    head = ProjectionSemanticHead(dim=freeze.FROZEN_EMBEDDING_DIM)
    assert head.query_projection.bias is None and head.node_projection.bias is None


# --------------------------------------------------------------------------
# The check can fail
# --------------------------------------------------------------------------


def test_s3s_learned_columns_are_checked_with_non_zero_weights() -> None:
    """difference_weight starts at zero; any formula reproduces zero."""

    head = freeze._with_nonzero_weights(
        SemanticHead(rung="S3", dim=freeze.FROZEN_EMBEDDING_DIM)
    )
    assert float(head.difference_weight.detach().abs().max()) > 0
    assert float(head.product_weight.detach().abs().max()) > 0


def test_a_wrong_scalar_formula_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Swap the sign inside the weighted L1 -- still plausible, still wrong."""

    def wrong(weight):
        def compute(query, candidates):
            return torch.stack([((row - query) * weight).sum() for row in candidates])

        return compute

    monkeypatch.setattr(freeze, "_weighted_absolute_difference", wrong)
    report = freeze.build_report()
    assert "S3.semantic_difference does not match its stated formula" in report["failed_checks"]
    assert report["verdict"] == "FORMULA_TRANSCRIPTION_WRONG"


def test_a_wrong_projection_formula_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """Normalise before the GELU instead of after: a real, easy transcription slip."""

    def wrong(head, query, candidates):
        raw_query = head.query_projection.weight @ query
        raw_nodes = torch.stack([head.node_projection.weight @ row for row in candidates])
        unit_query = torch.nn.functional.gelu(raw_query / raw_query.norm().clamp_min(1e-12))
        unit_nodes = torch.nn.functional.gelu(
            raw_nodes / raw_nodes.norm(dim=1, keepdim=True).clamp_min(1e-12)
        )
        return raw_query, raw_nodes, unit_query, unit_nodes

    monkeypatch.setattr(freeze, "_projection_states", wrong)
    report = freeze.build_report()
    assert report["verdict"] == "FORMULA_TRANSCRIPTION_WRONG"
    assert any("S4." in failure for failure in report["failed_checks"])


def test_a_mis_described_width_is_caught(monkeypatch: pytest.MonkeyPatch) -> None:
    """A block list that does not span the emitted tensor must be refused."""

    original = freeze.verify_projection_blocks

    def truncated(head, dim):
        result = original(head, dim)
        dropped = result["per_block"].pop("raw_projection_dot_scaled")
        result["described_width"] -= dropped["columns"][1] - dropped["columns"][0]
        result["widths_agree"] = result["emitted_width"] == result["described_width"]
        return result

    monkeypatch.setattr(freeze, "verify_projection_blocks", truncated)
    report = freeze.build_report()
    assert "S4's described blocks do not span the width it emits" in report["failed_checks"]


def test_a_column_count_mismatch_is_refused_outright() -> None:
    """Describing fewer columns than a rung emits must raise, not report."""

    head = SemanticHead(rung="S3", dim=freeze.FROZEN_EMBEDDING_DIM)
    with pytest.raises(SystemExit, match="columns"):
        freeze.verify_scalar_columns(head, freeze.s2_columns(), freeze.FROZEN_EMBEDDING_DIM)


def test_renaming_a_column_is_refused_outright() -> None:
    head = SemanticHead(rung="S2", dim=freeze.FROZEN_EMBEDDING_DIM)
    renamed = freeze.s2_columns()
    renamed[0] = {**renamed[0], "name": "cosine_similarity"}
    with pytest.raises(SystemExit, match="names its columns"):
        freeze.verify_scalar_columns(head, renamed, freeze.FROZEN_EMBEDDING_DIM)


def test_the_tolerance_is_tight_enough_to_matter(report: dict) -> None:
    """Every measured agreement is orders of magnitude inside the bound."""

    measured = [
        column["relative_difference"]
        for column in report["rungs"]["S2"]["columns"] + report["rungs"]["S3"]["columns"]
    ] + [
        block["relative_difference"]
        for block in report["rungs"]["S4"]["verification"]["per_block"].values()
    ]
    assert max(measured) < freeze.TOLERANCE / 10, (
        "the formulas only just pass, which means the tolerance is doing the work"
    )


# --------------------------------------------------------------------------
# The ladder the freeze establishes
# --------------------------------------------------------------------------


def test_the_ladder_is_strictly_increasing_in_capacity(report: dict) -> None:
    ladder = report["measured_ladder"]
    columns = [ladder[rung]["semantic_columns"] for rung in ("S2", "S3", "S4")]
    parameters = [ladder[rung]["semantic_parameters"] for rung in ("S2", "S3", "S4")]
    totals = [ladder[rung]["total_parameters"] for rung in ("S2", "S3", "S4")]
    assert columns == sorted(columns) and len(set(columns)) == 3
    assert parameters == sorted(parameters) and len(set(parameters)) == 3
    assert totals == [449, 3585, 205217]


def test_the_scorer_size_formula_holds_at_every_rung(report: dict) -> None:
    for rung, entry in report["rungs"].items():
        scorer = entry["scorer"]
        assert scorer["scorer_parameters"] == 32 * scorer["input_width"] + 65, rung
        assert scorer["input_width"] == 9 + scorer["semantic_width"], rung


def test_the_question_the_ladder_makes_askable_is_recorded(report: dict) -> None:
    ladder = report["the_ladder"]
    assert "fixed" in ladder["S2"]
    assert "learned" in ladder["S3"] and "learned" in ladder["S4"]
    assert "graph structure" in ladder["the_question_this_makes_askable"]
