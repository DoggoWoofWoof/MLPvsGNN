"""M2's universal schema: NODE_ROLE is one column everywhere, zero where it does not apply.

configs/m2_qls_v2_freeze.yaml amendment 1 (universal_node_role_policy =
ZERO_WHEN_NOT_APPLICABLE) rests on four claims, each pinned here:

1. Under R1 and R2 the scored set is Cq, so NODE_ROLE's own definition
   (1 iff v in C3 \\ Cq) evaluates to a constant zero -- shape (n, 1), all
   zero, for every query. Under R3 nothing changed: Cq -> 0, A64 -> 1.
2. The runner change is scoped. Only the universal arm may carry NODE_ROLE
   outside R3; every historical M1A/M1B arm still raises there, and no
   historical arm reads the new column (its stores are column-for-column
   what they were), which is what M2's reuse of 19 seed-0 fits relies on.
3. The 14-column model is the 13-column model on any R1/R2 input: with
   the first-layer weights transplanted minus the NODE_ROLE column, the
   scores agree to float precision, and those 32 weights receive zero
   gradient. Feature/input correctness and parameter accounting -- not
   metric agreement after training, which the ruling does not require.
4. Parameter counts are measured by live instantiation, never asserted:
   3,585 / 3,072 / 513 at width 14; 3,553 / 3,072 / 481 at width 13; and
   the same formula reproduces the real M1A fits' recorded counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from mp_retrieval.m1a_screen import HEAD_WIDTH, M1AScorer, node_role_column  # noqa: E402
from scripts import run_m1a_feature_screen as runner  # noqa: E402
from scripts import run_m1b_targeted_resolution  # noqa: E402, F401  registers M1B's BASE+NODE_ROLE+SUPPORT arm
from scripts.run_m1a_feature_screen import (  # noqa: E402
    ARM_FAMILIES,
    MASTER_COLUMNS,
    UNIVERSAL_ARM,
    ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE,
    _arm_columns,
    run,
)
from test_run_m1a_feature_screen import _args  # noqa: E402  same cross-module precedent as test_graph_context_d*.py

M1A_DECLARATION = REPO_ROOT / "configs" / "m1a_feature_screen.yaml"
M2_DECLARATION = REPO_ROOT / "configs" / "m2_qls_v2_freeze.yaml"
M1A_HEADLINE = REPO_ROOT / "outputs" / "m1a_feature_screen" / "headline" / "hotpotqa_clean.json"
HISTORICAL_ARMS = ("BASE", "BASE+GEOMETRY", "BASE+SUPPORT", "BASE+PATH", "BASE+NODE_ROLE", "BASE+NODE_ROLE+SUPPORT")
NODE_ROLE_POSITION = 4  # inside the universal arm's 9-column precomputed block: BASE[0:4], NODE_ROLE[4]
UNIVERSAL_WIDTH = 14
FROZEN_DIM = 1536
SENTINEL = 999.0  # a value no feature column ever takes; surfaces any historical arm that reads NODE_ROLE


# --- 1. the feature's value per regime ------------------------------------------


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_node_role_is_shape_correct_and_identically_zero_when_the_scored_set_is_cq(regime):
    cq = np.array([3, 8, 12, 21, 40], dtype=np.int64)
    scored = cq  # exactly what _cell_master_local scores under R1 and R2
    # R2 widens the context (TARGET_H1), never the scored set; R1 widens neither.
    context = cq if regime == "R1" else np.union1d(cq, np.array([1, 2, 99], dtype=np.int64))
    column = node_role_column(cq=cq, scored=scored, context=context)
    assert column.shape == (len(scored), 1)
    assert column.dtype == np.float32
    assert np.all(column == 0)


def test_node_role_under_r3_is_zero_on_cq_and_one_on_structurally_admitted_candidates():
    cq = np.array([3, 8, 12], dtype=np.int64)
    a64 = np.array([5, 30], dtype=np.int64)
    c3 = np.union1d(cq, a64)
    u3_bounded = np.union1d(c3, np.array([77], dtype=np.int64))  # a context-only node, never scored
    column = node_role_column(cq=cq, scored=c3, context=u3_bounded)
    assert column.shape == (len(c3), 1)
    value = {int(node): float(v) for node, v in zip(c3.tolist(), column[:, 0].tolist(), strict=True)}
    for candidate_from_cq in cq.tolist():
        assert value[candidate_from_cq] == 0.0
    for structurally_admitted_candidate in a64.tolist():
        assert value[structurally_admitted_candidate] == 1.0


# --- 2. the runner change is scoped to the universal arm --------------------------


def test_universal_arm_is_registered_in_the_frozen_family_order_and_nowhere_in_m1a():
    assert UNIVERSAL_ARM == "BASE+NODE_ROLE+SUPPORT+PATH"
    assert ARM_FAMILIES[UNIVERSAL_ARM] == ("NODE_ROLE", "SUPPORT", "PATH")
    assert ZERO_NODE_ROLE_WHEN_NOT_APPLICABLE == frozenset({UNIVERSAL_ARM})
    assert ARM_FAMILIES["BASE+NODE_ROLE"] == ("NODE_ROLE",)
    assert ARM_FAMILIES["BASE+NODE_ROLE+SUPPORT"] == ("NODE_ROLE", "SUPPORT")
    # M1A's declaration is frozen: registering the arm must not put it in any M1A cell.
    declaration = yaml.safe_load(M1A_DECLARATION.read_text(encoding="utf-8"))
    for dataset, spec in declaration["datasets"].items():
        for regime, cell in spec.get("cells", {}).items():
            assert UNIVERSAL_ARM not in cell["arms"], (dataset, regime)


def test_universal_arm_columns_follow_the_frozen_precomputed_order():
    master = np.arange(12, dtype=np.float32).reshape(1, 12)
    # R3: BASE, then NODE_ROLE (master col 11), SUPPORT (7), PATH (8-10) -- the
    # declaration's frozen_column_order positions 0-8.
    assert _arm_columns(master, UNIVERSAL_ARM, "R3").tolist() == [[0, 1, 2, 3, 11, 7, 8, 9, 10]]
    master[:, MASTER_COLUMNS["NODE_ROLE"]] = 0.0
    for regime in ("R1", "R2"):
        out = _arm_columns(master, UNIVERSAL_ARM, regime)
        assert out.shape == (1, 9)
        assert out.tolist() == [[0, 1, 2, 3, 0, 7, 8, 9, 10]]


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_a_nonzero_node_role_outside_r3_is_refused_even_for_the_universal_arm(regime):
    master = np.zeros((3, 12), dtype=np.float32)
    master[1, MASTER_COLUMNS["NODE_ROLE"]] = 1.0
    with pytest.raises(ValueError, match="identically zero"):
        _arm_columns(master, UNIVERSAL_ARM, regime)


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_historical_arms_still_refuse_node_role_outside_r3_and_never_read_its_column(regime):
    master = np.arange(12, dtype=np.float32).reshape(1, 12)
    master[:, MASTER_COLUMNS["NODE_ROLE"]] = SENTINEL
    assert _arm_columns(master, "BASE", regime).tolist() == [[0, 1, 2, 3]]
    assert _arm_columns(master, "BASE+GEOMETRY", regime).tolist() == [[0, 1, 2, 3, 4, 5, 6]]
    assert _arm_columns(master, "BASE+SUPPORT", regime).tolist() == [[0, 1, 2, 3, 7]]
    assert _arm_columns(master, "BASE+PATH", regime).tolist() == [[0, 1, 2, 3, 8, 9, 10]]
    for arm in ("BASE+NODE_ROLE", "BASE+NODE_ROLE+SUPPORT"):
        with pytest.raises(ValueError, match="only defined for R3"):
            _arm_columns(master, arm, regime)


def test_historical_r3_arms_are_unchanged():
    master = np.arange(12, dtype=np.float32).reshape(1, 12)
    for arm in HISTORICAL_ARMS:
        expected = [master[0, sl].tolist() for sl in (MASTER_COLUMNS["BASE"], *(MASTER_COLUMNS[f] for f in ARM_FAMILIES[arm]))]
        assert _arm_columns(master, arm, "R3").tolist() == [[v for block in expected for v in block]], arm


# --- the real runner, end to end on the toy dataset -----------------------------


@pytest.fixture(scope="module")
def universal_toy_run(tmp_path_factory):
    """BASE and the universal arm under R1/R2/R3, recording each cell's master blocks.

    _declared_cells is patched to append the universal arm to hotpotqa_clean's
    declared cells because M1A's declaration is frozen and must not list it;
    _arm_columns is wrapped (not replaced) to capture the master block each
    query produced, once per (regime, query).
    """

    args = _args(tmp_path_factory.mktemp("m2_universal"), dataset="hotpotqa_clean", arms=["BASE", UNIVERSAL_ARM])
    seen: dict[str, list[np.ndarray]] = {}
    real_arm_columns = runner._arm_columns
    real_declared_cells = runner._declared_cells

    def _recording(master, arm, regime):
        if arm == "BASE":
            seen.setdefault(regime, []).append(np.array(master, copy=True))
        return real_arm_columns(master, arm, regime)

    def _with_universal(declaration, dataset):
        return {regime: [*arms, UNIVERSAL_ARM] for regime, arms in real_declared_cells(declaration, dataset).items()}

    with (
        patch("scripts.run_m1a_feature_screen._arm_columns", side_effect=_recording),
        patch("scripts.run_m1a_feature_screen._declared_cells", side_effect=_with_universal),
    ):
        result = run(args)
    return result, seen


@pytest.mark.parametrize("regime", ["R1", "R2"])
def test_runner_emits_a_zero_node_role_column_for_every_r1_r2_query(universal_toy_run, regime):
    result, seen = universal_toy_run
    assert len(seen[regime]) == result["queries"]
    for master in seen[regime]:
        node_role = master[:, MASTER_COLUMNS["NODE_ROLE"]]
        assert master.shape[1] == 12
        assert node_role.shape == (master.shape[0], 1)
        assert np.all(node_role == 0)


def test_runner_r3_node_role_column_is_binary_and_marks_a_real_structural_admission(universal_toy_run):
    result, seen = universal_toy_run
    assert len(seen["R3"]) == result["queries"]
    stacked = np.concatenate([master[:, MASTER_COLUMNS["NODE_ROLE"]] for master in seen["R3"]])
    # The fixture plants an A64-only gold for q0, so C3 \ Cq is non-empty: a 1 must exist.
    assert set(np.unique(stacked).tolist()) == {0.0, 1.0}


def test_universal_arm_fits_at_width_nine_in_every_regime_while_base_stays_at_four(universal_toy_run):
    result, _seen = universal_toy_run
    for regime in ("R1", "R2", "R3"):
        arms = result["cells"][regime]["arms"]
        assert set(arms) == {"BASE", UNIVERSAL_ARM}
        assert arms[UNIVERSAL_ARM]["precomputed_width"] == 9
        assert arms["BASE"]["precomputed_width"] == 4
        # scorer params = 32*in_dim + 65, embedding-dim independent, so the real 513 is verified even on the toy
        assert arms[UNIVERSAL_ARM]["parameters"]["scorer"] == HEAD_WIDTH * UNIVERSAL_WIDTH + 2 * HEAD_WIDTH + 1 == 513
        assert arms["BASE"]["parameters"]["scorer"] == 353


# --- 3. the 14-column model is the 13-column model wherever NODE_ROLE is zero ----


def _scorer(precomputed_width: int) -> M1AScorer:
    return M1AScorer(
        precomputed_width=precomputed_width,
        semantic_rung="S3",
        dropout=0.2,
        temperature=0.07,
        embedding_dim=FROZEN_DIM,
    )


def _transplanted_pair() -> tuple[M1AScorer, M1AScorer, list[int]]:
    """A universal (14-column) scorer and a 13-column scorer sharing every weight but NODE_ROLE's.

    Returns the precomputed-column indices the 13-column model reads (the
    universal block minus NODE_ROLE); the first-layer weight is sliced over
    the full 14 scorer inputs, where NODE_ROLE sits at the same position
    because the precomputed block precedes the semantic block.
    """

    torch.manual_seed(20260907)
    universal = _scorer(9)
    thirteen = _scorer(8)
    keep_weights = [column for column in range(UNIVERSAL_WIDTH) if column != NODE_ROLE_POSITION]
    keep_inputs = [column for column in range(9) if column != NODE_ROLE_POSITION]
    with torch.no_grad():
        thirteen.semantic_head.load_state_dict(universal.semantic_head.state_dict())
        thirteen.scorer[0].weight.copy_(universal.scorer[0].weight[:, keep_weights])
        thirteen.scorer[0].bias.copy_(universal.scorer[0].bias)
        thirteen.scorer[3].load_state_dict(universal.scorer[3].state_dict())
    return universal.eval(), thirteen.eval(), keep_inputs


def _batch(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    nodes = torch.randn(7, FROZEN_DIM, generator=generator)
    queries = torch.randn(2, FROZEN_DIM, generator=generator)
    batch_index = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    structural = torch.randn(7, 9, generator=generator)
    return nodes, queries, batch_index, structural


def test_fourteen_column_model_equals_the_thirteen_column_model_on_any_r1_r2_input():
    universal, thirteen, keep = _transplanted_pair()
    for seed in (1, 2, 3):
        nodes, queries, batch_index, structural = _batch(seed)
        structural[:, NODE_ROLE_POSITION] = 0.0  # the R1/R2 value, for every scored candidate
        with torch.no_grad():
            scores_14 = universal.forward_explicit(nodes, queries, batch_index, structural)
            scores_13 = thirteen.forward_explicit(nodes, queries, batch_index, structural[:, keep])
        assert torch.allclose(scores_14, scores_13, atol=1e-5, rtol=1e-5)


def test_the_column_is_live_under_r3_with_the_same_weights():
    universal, _thirteen, _keep = _transplanted_pair()
    nodes, queries, batch_index, structural = _batch(4)
    zero = structural.clone()
    zero[:, NODE_ROLE_POSITION] = 0.0
    admitted = structural.clone()
    admitted[:, NODE_ROLE_POSITION] = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    with torch.no_grad():
        scores_zero = universal.forward_explicit(nodes, queries, batch_index, zero)
        scores_admitted = universal.forward_explicit(nodes, queries, batch_index, admitted)
    changed = ~torch.isclose(scores_zero, scores_admitted)
    assert changed.tolist() == [True, False, True, False, False, True, False]


def test_with_node_role_weights_zeroed_the_models_agree_on_any_input_at_all():
    # The ruling's literal statement: the 14-column model reproduces the
    # 13-column one when the NODE_ROLE-connected weights are zero -- whatever
    # the column holds, R3 ones included.
    universal, thirteen, keep = _transplanted_pair()
    with torch.no_grad():
        universal.scorer[0].weight[:, NODE_ROLE_POSITION] = 0.0
    nodes, queries, batch_index, structural = _batch(5)  # NODE_ROLE column left at random values
    with torch.no_grad():
        scores_14 = universal.forward_explicit(nodes, queries, batch_index, structural)
        scores_13 = thirteen.forward_explicit(nodes, queries, batch_index, structural[:, keep])
    assert torch.allclose(scores_14, scores_13, atol=1e-5, rtol=1e-5)


def test_node_role_weights_receive_zero_gradient_under_r1_r2_and_nonzero_under_r3():
    for node_role_input, expect_dead in ((0.0, True), (1.0, False)):
        universal = _scorer(9).train()
        nodes, queries, batch_index, structural = _batch(6)
        structural[:, NODE_ROLE_POSITION] = node_role_input
        universal.forward_explicit(nodes, queries, batch_index, structural).sum().backward()
        grad = universal.scorer[0].weight.grad
        assert grad.shape == (HEAD_WIDTH, UNIVERSAL_WIDTH)
        node_role_grad = grad[:, NODE_ROLE_POSITION]
        assert bool(torch.all(node_role_grad == 0)) is expect_dead
        other = grad[:, [c for c in range(UNIVERSAL_WIDTH) if c != NODE_ROLE_POSITION]]
        assert bool(torch.any(other != 0))


# --- 4. parameter accounting, live -----------------------------------------------


def test_parameter_counts_are_the_declared_ones_by_live_instantiation():
    universal = _scorer(9)
    thirteen = _scorer(8)
    assert universal.trainable_parameter_count() == 3585
    assert universal.semantic_parameter_count() == 3072 == 2 * FROZEN_DIM
    assert universal.scorer_parameter_count() == 513
    assert thirteen.trainable_parameter_count() == 3553
    assert thirteen.semantic_parameter_count() == 3072
    assert thirteen.scorer_parameter_count() == 481
    # exactly the 32 first-layer weights on the NODE_ROLE input, nothing else
    assert universal.trainable_parameter_count() - thirteen.trainable_parameter_count() == HEAD_WIDTH


def test_declaration_transcribes_the_live_counts():
    declared = yaml.safe_load(M2_DECLARATION.read_text(encoding="utf-8"))
    counts = declared["qls_universal"]["parameter_count"]
    assert counts["total"] == _scorer(9).trainable_parameter_count()
    assert counts["thirteen_column_comparison_model"] == _scorer(8).trainable_parameter_count()
    assert declared["qls_universal"]["feature_schema"]["width"] == UNIVERSAL_WIDTH
    assert declared["qls_universal"]["feature_schema"]["frozen_column_order"][NODE_ROLE_POSITION] == "node_role_is_structurally_admitted"


@pytest.mark.skipif(not M1A_HEADLINE.exists(), reason="real M1A headline result not present")
def test_the_same_formula_reproduces_the_real_m1a_fits_recorded_counts():
    headline = json.loads(M1A_HEADLINE.read_text(encoding="utf-8"))
    arms = headline["cells"]["R1"]["arms"]
    for arm, width in (("BASE", 4), ("BASE+SUPPORT", 5), ("BASE+PATH", 7)):
        recorded = arms[arm]
        assert recorded["precomputed_width"] == width
        assert recorded["parameters"]["total"] == _scorer(width).trainable_parameter_count()
        assert recorded["parameters"]["scorer"] == HEAD_WIDTH * (width + 5) + 2 * HEAD_WIDTH + 1
