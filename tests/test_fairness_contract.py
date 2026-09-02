"""The fairness contract has to point at code that exists.

A contract whose enforcement column names a function nobody wrote is worse than
no contract: it reads as a guarantee and is a wish. These tests check the
mechanical clauses -- the ones the document claims are enforced by code -- and
say plainly which clauses they cannot check.

The clauses in section 5 are obligations on the GNN side (equal tuning budget,
depth matched to context radius, the strongest fair baseline). None of them is
machine-checkable, all of them are the ones most easily broken by omission, and
the document says so itself. They are checked at review time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

CONTRACT_PATH = REPO_ROOT / "docs" / "GRAPH_INFORMATION_FAIRNESS_CONTRACT.md"


@pytest.fixture(scope="module")
def contract() -> str:
    return CONTRACT_PATH.read_text(encoding="utf-8")


def test_the_contract_exists_before_any_comparison(contract):
    """Required to be filed BEFORE a QLS-v2 versus GNN comparison is run."""
    assert "before any QLS-v2 versus GNN comparison" in " ".join(contract.split())


def test_the_single_context_code_path_it_names_is_real():
    """One code path, so the two systems cannot silently diverge on context."""
    from mp_retrieval.graph_context import context_nodes

    assert callable(context_nodes)


def test_the_runner_declares_the_scoring_universe_the_contract_requires():
    source = (REPO_ROOT / "scripts" / "run_graph_context_pilot.py").read_text(
        encoding="utf-8"
    )
    assert '"scored_nodes": "exactly Cq in every arm"' in source


def test_the_runner_checks_the_candidate_contract_hash_on_both_sides_of_a_run():
    """Frozen pools, verified rather than assumed, before and after."""
    source = (REPO_ROOT / "scripts" / "run_graph_context_pilot.py").read_text(
        encoding="utf-8"
    )
    assert "contract_before" in source
    assert "Candidate contract changed while computing a read-only diagnostic" in source


def test_the_qls_feature_path_imports_no_model_module():
    """No GNN anywhere in the method -- checked at the import graph, not asserted.

    `graph_context` is what QLS-v2's context and descriptors are built from. The
    model modules are where every GNN in this repository lives.
    """
    source = (REPO_ROOT / "src" / "mp_retrieval" / "graph_context.py").read_text(
        encoding="utf-8"
    )
    for banned in ("operator_models", "l2_models", "torch_geometric", "import models"):
        assert banned not in source, banned


@pytest.mark.parametrize(
    "clause",
    [
        "The graph context is a property of the evaluation condition",
        "The GNN is not to be crippled",
        "Depth is matched to context radius",
        "The strongest fair baseline is the comparator",
        "Parameter count is not a handicap to impose",
    ],
)
def test_the_contract_states_each_obligation_it_is_relied_on_for(contract, clause):
    assert clause in contract


def test_the_contract_does_not_claim_a_paper_defines_the_procedure(contract):
    """Standing rule: never claim one cited paper defines our exact procedure."""
    flat = " ".join(contract.split())
    assert "It does not claim that any cited paper defines this procedure" in flat


def test_the_contract_keeps_shared_cost_out_of_the_method_comparison(contract):
    flat = " ".join(contract.split())
    assert "paid by **both** systems, so it cancels" in flat
    assert '"QLS-v2 is cheaper" is not a claim' in flat
