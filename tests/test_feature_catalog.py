"""The frozen feature catalog, checked against the stages that produced it.

D10 froze a catalog rather than a winner. The failure mode this module exists
to prevent is a later stage reading the catalog as a ranking: quoting a family
as "the best one", or as tested when it was never fitted, or as universal when
every number in it is 2Wiki, seed 0, one architecture and one split.

So the checks here are mostly refusals. Every family must carry a status drawn
from a closed vocabulary. Every increment quoted must be the one the stage
actually recorded, to the digit. Families with no fitted arm must say so and
must claim no columns. And the block must keep saying, in its own text, that
the subset the ladder selected is a 2Wiki result.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "graph_context_pilot.yaml"
DOC = REPO_ROOT / "docs" / "GRAPH_CONTEXT_FEATURE_CATALOG.md"
STAGE_DIR = (
    REPO_ROOT / "outputs" / "graph_context_pilot" / "2wiki_clean" / "d7c2da85e2b65680"
)

FAMILIES = (
    "RETRIEVAL",
    "SEED",
    "GEOMETRY",
    "SUPPORT",
    "PATH",
    "DIFFUSION",
    "TOPOLOGY",
    "PROVENANCE",
    "NODE_ROLE",
)
REQUIRED_KEYS = frozenset(
    {
        "columns",
        "members",
        "status_on_2wiki",
        "evidence",
        "increment",
        "strength",
        "a_later_stage_may",
        "it_does_not_authorise",
    }
)
FITTED_STATUSES = frozenset(
    {
        "ADMITTED",
        "CARRIED_NOT_ON_ACCURACY",
        "PROMISING_REPLACEMENT_FAILED",
        "NEGLIGIBLE",
    }
)
UNFITTED_STATUSES = frozenset({"MEASURED_AS_A_GRAPH_AXIS", "NOT_TESTED"})
METRICS = ("recall@1", "recall@5", "recall@20", "mrr", "full_coverage@20")

# The base arm every family after D6 was tested against: distance geometry plus
# the graded retrieval prior, with columns 4 through 9 held at zero.
BASE_COLUMNS = (0, 1, 2, 3, 10, 11, 12)
LOCAL_DIM = 13


@pytest.fixture(scope="module")
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalog(config) -> dict:
    return config["feature_catalog"]


def _stage(name: str) -> dict:
    path = STAGE_DIR / f"stage_{name}.json"
    if not path.exists():
        pytest.skip(f"stage_{name}.json is not present locally")
    return json.loads(path.read_text(encoding="utf-8"))


def signed(value: float) -> str:
    return f"{'+' if value >= 0 else '-'}{abs(value) * 100:.2f}"


# --- shape -------------------------------------------------------------------


def test_the_catalog_is_frozen_and_names_its_stage(catalog):
    assert catalog["status"] == "FROZEN_AT_D10"
    assert "D10" in catalog["frozen_by"]


def test_exactly_the_nine_declared_families_are_present(catalog):
    assert tuple(catalog["families"]) == FAMILIES


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_carries_every_required_field(catalog, family):
    block = catalog["families"][family]
    assert REQUIRED_KEYS <= set(block), REQUIRED_KEYS - set(block)
    for key in REQUIRED_KEYS - {"columns"}:
        assert isinstance(block[key], str) and block[key].strip(), key


@pytest.mark.parametrize("family", FAMILIES)
def test_every_status_is_drawn_from_the_closed_vocabulary(catalog, family):
    status = catalog["families"][family]["status_on_2wiki"]
    assert status in catalog["status_vocabulary"], status
    assert status in FITTED_STATUSES | UNFITTED_STATUSES, status


@pytest.mark.parametrize("family", FAMILIES)
def test_columns_are_real_columns_of_the_frozen_block(catalog, family):
    for column in catalog["families"][family]["columns"]:
        assert 0 <= column < LOCAL_DIM, column


def test_a_family_with_no_fitted_arm_claims_no_columns(catalog):
    for name, block in catalog["families"].items():
        if block["status_on_2wiki"] in UNFITTED_STATUSES:
            assert block["columns"] == [], name


def test_seed_is_the_only_overlap_and_the_catalog_says_so(catalog):
    families = catalog["families"]
    assert set(families["SEED"]["columns"]) < set(families["GEOMETRY"]["columns"])
    # Every other pair is disjoint, or the contributions would double-count.
    named = [name for name in FAMILIES if name != "SEED"]
    for index, left in enumerate(named):
        for right in named[index + 1 :]:
            overlap = set(families[left]["columns"]) & set(families[right]["columns"])
            assert not overlap, (left, right, overlap)
    assert "overlap" in families["SEED"]["it_does_not_authorise"].lower()


def test_every_column_of_the_frozen_block_is_claimed(catalog):
    claimed = set()
    for block in catalog["families"].values():
        claimed |= set(block["columns"])
    assert claimed == set(range(LOCAL_DIM)), set(range(LOCAL_DIM)) - claimed


# --- the increments are the recorded ones ------------------------------------


def _quoted(text: str) -> list[str]:
    """Every signed two-decimal figure in a catalog increment string."""
    return re.findall(r"[+-]\d+\.\d{2}", text)


@pytest.mark.parametrize(
    "family,stage,key",
    [
        ("RETRIEVAL", "d4", "delta_retrieval_quality"),
        ("RETRIEVAL", "d5", "delta_prior_given_geometry"),
        ("GEOMETRY", "d5", "delta_geometry_given_prior"),
        ("SUPPORT", "d7", "delta_support"),
        ("PATH", "d7", "delta_paths"),
        ("DIFFUSION", "d7", "delta_diffusion"),
        ("TOPOLOGY", "d7", "delta_neighbourhood"),
    ],
)
def test_the_quoted_increment_is_the_recorded_one(catalog, family, stage, key):
    recorded = _stage(stage)["increments"][key]
    quoted = _quoted(catalog["families"][family]["increment"])
    for metric in METRICS:
        assert signed(recorded[metric]) in quoted, (family, stage, metric)


def test_the_path_replacement_increment_is_d10s(catalog):
    increments = _stage("d10")["increments"]
    delta = increments["delta_bounded_diversity_over_historical_walk_proxy"]
    quoted = _quoted(catalog["families"]["PATH"]["increment"])
    for metric in METRICS:
        assert signed(delta[metric]) in quoted, metric


def test_the_support_replacement_increment_is_d8s(catalog):
    increments = _stage("d8")["increments"]
    delta = increments["delta_corrected_representation_over_historical_proxy"]
    quoted = _quoted(catalog["families"]["SUPPORT"]["increment"])
    for metric in METRICS:
        assert signed(delta[metric]) in quoted, metric


def test_the_seed_and_distance_figures_are_d3s(catalog):
    increments = _stage("d3")["increments"]
    seed = catalog["families"]["SEED"]["increment"]
    geometry = catalog["families"]["GEOMETRY"]["increment"]
    assert f"{increments['delta_seed']['recall@5'] * 100:.2f}" in seed
    assert f"{increments['delta_distance']['recall@5'] * 100:.2f}" in geometry


def test_the_two_carried_families_are_carried_on_different_grounds(catalog):
    support = catalog["families"]["SUPPORT"]
    path = catalog["families"]["PATH"]
    assert support["status_on_2wiki"] == "CARRIED_NOT_ON_ACCURACY"
    assert "not on accuracy" in support["strength"].lower()
    assert path["status_on_2wiki"] == "PROMISING_REPLACEMENT_FAILED"
    assert "contested" in path["a_later_stage_may"].lower()


# --- the refusals -------------------------------------------------------------


def test_the_catalog_refuses_to_name_a_winner(catalog):
    assert "NOT a declaration of one global winner" in catalog["what_this_is_not"]
    assert "not a ranking" in catalog["what_this_is"].lower()
    blob = yaml.safe_dump(catalog).lower()
    for banned in ("the best family", "the winning family", "winner is"):
        assert banned not in blob, banned


def test_the_selected_subset_is_not_declared_universal(catalog):
    claim = catalog["the_2wiki_selected_subset_is_not_universal"]
    lowered = claim.lower()
    assert "not declared universal" in lowered
    for token in ("2wiki", "seed 0", "cand", "validation"):
        assert token in lowered, token


def test_the_selected_subset_is_the_base_arm_the_stages_actually_used(catalog):
    claim = catalog["what_the_ladder_actually_selected"]
    assert "0-3" in claim and "10-12" in claim
    assert "seven of thirteen" in claim.lower()
    assert len(BASE_COLUMNS) == 7
    admitted = set()
    for block in catalog["families"].values():
        if block["status_on_2wiki"] == "ADMITTED":
            admitted |= set(block["columns"])
    assert admitted == set(BASE_COLUMNS)


def test_the_untested_families_are_not_written_up_as_failures(catalog):
    for name, block in catalog["families"].items():
        if block["status_on_2wiki"] != "NOT_TESTED":
            continue
        assert "not tested" in block["it_does_not_authorise"].lower(), name


def test_the_negligible_families_are_marked_as_one_encoding_not_a_verdict(catalog):
    for name, block in catalog["families"].items():
        if block["status_on_2wiki"] != "NEGLIGIBLE":
            continue
        refusal = block["it_does_not_authorise"].lower()
        assert "no signal" in refusal or "worthless" in refusal, name


def test_provenance_is_kept_separate_from_the_feature_question(catalog):
    block = catalog["families"]["PROVENANCE"]
    assert block["status_on_2wiki"] == "MEASURED_AS_A_GRAPH_AXIS"
    refusal = block["it_does_not_authorise"].lower()
    assert "gnn" in refusal, "the no-GNN-selects-features rule has to be restated"


def test_the_inference_about_remaining_structure_is_labelled(catalog):
    claim = catalog["what_that_pattern_is_evidence_for"].lower()
    assert claim.startswith("stated as inference, not measurement")
    assert "not evidence that structure is exhausted" in claim


# --- the write-up -------------------------------------------------------------


@pytest.fixture(scope="module")
def text() -> str:
    if not DOC.exists():
        pytest.skip("the catalog write-up is not present locally")
    return DOC.read_text(encoding="utf-8")


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_appears_in_the_write_up(text, catalog, family):
    assert family.replace("_", " ") in text or family in text
    assert catalog["families"][family]["status_on_2wiki"] in text


@pytest.mark.parametrize("family", FAMILIES)
def test_the_write_up_carries_each_familys_refusal(text, catalog, family):
    """A status without its limit is the half a later stage would misread."""
    refusal = catalog["families"][family]["it_does_not_authorise"]
    stem = " ".join(refusal.split()[:4]).rstrip(",.").lower()
    assert stem in " ".join(text.split()).lower(), (family, stem)


def test_the_write_up_does_not_crown_a_winner(text):
    lowered = text.lower()
    for banned in ("the best family", "the winning family", "the winner"):
        assert banned not in lowered, banned
    assert "not a ranking" in lowered


def test_the_write_up_says_the_subset_is_a_2wiki_result(text):
    lowered = text.lower()
    assert "not declared universal" in lowered or "is a 2wiki result" in lowered
    assert "one dataset and one seed" in lowered
