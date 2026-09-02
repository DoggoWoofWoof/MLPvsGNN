"""The results document has to agree with the result files it reports.

Written because it did not, twice. A headline figure was typed from memory as
"every arm moves 1.3% of candidates" when the measured value was 66.7%/65.6%,
and a seed count was asserted that no result file records. Both were caught by
reading the JSON back; neither would have been caught by anything else.

So the numeric claims in `docs/GRAPH_CONTEXT_PILOT_RESULTS.md` are pinned here
against `outputs/graph_context_pilot/`. Those outputs are gitignored, so every
test that needs them skips when they are absent -- a skip means "not checked
here", never "checked and fine", and the prose claims that do not depend on the
outputs are checked unconditionally below.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

DOC_PATH = REPO_ROOT / "docs" / "GRAPH_CONTEXT_PILOT_RESULTS.md"
CONFIG_PATH = REPO_ROOT / "configs" / "graph_context_pilot.yaml"
STAGE_C = REPO_ROOT / "outputs" / "graph_context_pilot" / "stage_c"

DATASETS = (
    "2wiki_clean",
    "hotpotqa_clean",
    "metaqa",
    "musique_clean",
    "squad_clean",
    "webqsp",
)


@pytest.fixture(scope="module")
def doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def flat(doc: str) -> str:
    """Whitespace-normalised, so a markdown line wrap cannot fail an assertion."""
    return " ".join(doc.split())


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def stage_c() -> dict:
    if not STAGE_C.is_dir():
        pytest.skip(f"{STAGE_C} not present -- gitignored outputs, nothing to check")
    results = {}
    for dataset in DATASETS:
        path = STAGE_C / f"{dataset}_stage_c.json"
        if not path.is_file():
            pytest.skip(f"{path.name} missing")
        results[dataset] = json.loads(path.read_text(encoding="utf-8"))
    return results


def arm(stage_c: dict, dataset: str, name: str) -> dict:
    return stage_c[dataset]["splits"]["validation"]["arms"][name]


# --- claims that stand on the result files -------------------------------


def test_the_repair_claim_is_what_the_result_files_say(stage_c):
    """Retention 1.0000 and isolation 0.0000 on all six, or the headline is wrong."""
    for dataset in DATASETS:
        recovery = arm(stage_c, dataset, "TARGET_H1")["recovery"]
        assert recovery["retention_median"] == pytest.approx(1.0, abs=5e-5), dataset
        assert recovery["isolated_fraction"] == pytest.approx(0.0, abs=5e-5), dataset


def test_hotpotqa_is_the_only_dataset_with_a_residual_boundary_cut(stage_c):
    """The exception the document reports, and it falls where direction matters."""
    cuts = {
        dataset: arm(stage_c, dataset, "TARGET_H1")["recovery"]["boundary_cut"]
        for dataset in DATASETS
    }
    nonzero = {d: c for d, c in cuts.items() if c > 5e-5}
    assert set(nonzero) == {"hotpotqa_clean"}, cuts
    assert nonzero["hotpotqa_clean"] == pytest.approx(0.0015, abs=5e-5)


def test_the_advancement_criterion_holds_on_every_dataset(stage_c):
    """Monotone decreasing in prior induced degree -- repair, not rescaling.

    A survivor that flattens across strata is moving every candidate a little,
    which is what the filed declaration said would disqualify it.
    """
    order = ("isolated", "degree_1", "low_degree", "ordinary")
    for dataset in DATASETS:
        strata = arm(stage_c, dataset, "TARGET_H1")["reach"][
            "seed_distance_improved_by_prior_induced_degree"
        ]
        values = [strata[key] for key in order]
        assert values == sorted(values, reverse=True), (dataset, strata)


def test_the_document_quotes_the_measured_context_sizes(doc, stage_c):
    """Every p50/p95 in the size table is the number in the result file."""
    for dataset in DATASETS:
        size = arm(stage_c, dataset, "TARGET_H1")["size"]["context_nodes"]
        corpus = f"{stage_c[dataset]['num_nodes']:,}"
        rows = [
            line for line in doc.splitlines()
            if line.startswith(f"| {dataset} | {corpus} |")
        ]
        assert len(rows) == 1, (dataset, rows)
        cells = [cell.strip().replace(",", "") for cell in rows[0].split("|")]
        assert str(round(size["median"])) in cells, (dataset, rows[0])
        assert str(round(size["p95"])) in cells, (dataset, rows[0])
        assert str(round(size["max"])) in cells, (dataset, rows[0])


def test_the_saturation_figures_are_the_measured_ones(doc):
    """The claim that was typed wrong once. `distance_bucket_changed` is
    identical across every non-CAND arm; if Stage B is on disk, check it."""
    stage_b = REPO_ROOT / "outputs" / "graph_context_pilot" / "stage_b"
    if not stage_b.is_dir():
        pytest.skip("stage_b outputs absent")
    for path in sorted(stage_b.glob("*.json")):
        arms = json.loads(path.read_text(encoding="utf-8"))["splits"]["validation"][
            "arms"
        ]
        moved = {
            name: value["movement"]["distance_bucket_changed"]
            for name, value in arms.items()
            if name != "CAND"
        }
        assert len(set(round(v, 4) for v in moved.values())) == 1, (path.name, moved)
        assert f"{next(iter(moved.values())):.4f}" in doc, (path.name, moved)


# --- claims that stand on their own -------------------------------------


def test_the_document_does_not_call_the_context_a_small_share_of_the_graph(flat):
    """On squad the median context is 30% of the corpus. Saying otherwise is false."""
    assert "bounded in absolute node count" in flat or (
        "bounded in nodes, not in share of the graph" in flat
    )
    assert "30.23%" in flat and "50.24%" in flat


def test_the_document_states_what_it_does_not_establish(flat):
    for clause in (
        "No effectiveness claim",
        "No QLS-v2 versus GNN comparison",
        "No claim that a cited paper defines this procedure",
        "The test split was not read",
    ):
        assert clause in flat, clause


def test_the_document_and_the_config_recommend_the_same_arm(flat, config):
    assert config["stage_c_outcome"]["recommended_arm"] == "TARGET_H1"
    assert "`TARGET_H1 = Cq u N1_in(Cq)`" in flat


def test_the_config_points_at_the_results_document(config):
    assert config["results"] == "docs/GRAPH_CONTEXT_PILOT_RESULTS.md"
    assert config["stage_c_outcome"]["results"] == "docs/GRAPH_CONTEXT_PILOT_RESULTS.md"


def test_the_config_carries_the_size_qualification_beside_the_result(config):
    """The qualification has to travel with the number, not sit in one document."""
    qualification = config["stage_c_outcome"]["size_qualification"]
    assert "ABSOLUTE NODE COUNT" in qualification
    assert "NOT in share of the graph" in qualification


def test_no_seed_count_is_asserted_anywhere(flat, config):
    """The pilot does not record seeds per query, so nothing may claim one."""
    assert "seeds-per-query count is not recorded" in flat
    assert "NOT asserted here" in " ".join(config["definitions"]["Sq"].split())
    assert not re.search(r"~\s*\d+\s+seeds", flat)
