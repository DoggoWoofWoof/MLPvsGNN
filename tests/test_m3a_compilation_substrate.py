"""Tests for the M3A-COMPILATION substrate measurements.

Three artifacts, one job each: what the graphs are (substrate stats), whether
the two blocked datasets can be given queries (query view feasibility), and
what retrieval would cost (retrieval budget). None of them builds anything.

The assertions worth having here are the ones that would catch a number that
looks fine. A gold resolution rate of 1.0000 is exactly as plausible as one of
0.4152 -- the first version of the rewrite produced the second, and the data
was not the problem. So the tests check the rewrite against the package rather
than against the report that the rewrite produced.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PACKAGE = Path("C:/Users/Swastik/Desktop/CRAG")
SUBSTRATE = ROOT / "outputs" / "m3a" / "graph_substrate_stats.json"
FEASIBILITY = ROOT / "outputs" / "m3a" / "query_view_feasibility.json"
RETRIEVAL = ROOT / "outputs" / "m3a" / "retrieval_budget.json"

needs_package = pytest.mark.skipif(
    not PACKAGE.exists(), reason="CRAG package not present on this host"
)
# The query-view feasibility measurement read the pre-freeze tree
# CRAG/data/canonical/, which the package owner deleted on 2026-09-12 after the
# served freeze (data/final_canonical/) superseded it; the blocker it priced is
# WITHDRAWN_WRONG_AT_FILING in configs/m3a_compilation.yaml#amendment_1_2026_09_13.
# The three tests that re-derive it from that tree therefore skip, with the
# reason, wherever the tree is absent. Nothing they assert has changed.
QUERY_VIEW_TREE = PACKAGE / "data" / "canonical" / "metaqa" / "documents.jsonl"
needs_query_view_tree = pytest.mark.skipif(
    not QUERY_VIEW_TREE.exists(),
    reason="CRAG/data/canonical/ was deleted upstream on 2026-09-12 (superseded by data/final_canonical/); "
    "the measurement these verify is withdrawn in m3a_compilation.yaml amendment 1",
)
needs_artifacts = pytest.mark.skipif(
    not (SUBSTRATE.exists() and FEASIBILITY.exists() and RETRIEVAL.exists()),
    reason="M3A compilation artifacts not generated on this host",
)

pytestmark = needs_artifacts


@pytest.fixture(scope="module")
def substrate() -> dict:
    return json.loads(SUBSTRATE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def feasibility() -> dict:
    return json.loads(FEASIBILITY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def retrieval() -> dict:
    return json.loads(RETRIEVAL.read_text(encoding="utf-8"))


# ── nothing was built, moved or spent ────────────────────────────────────────


def test_no_gpu_was_spent_anywhere(substrate, feasibility, retrieval) -> None:
    for report in (substrate, feasibility, retrieval):
        assert report["measured_cost"]["gpu_seconds"] == 0.0
        assert report["access_mode"] == "READ_ONLY"


def test_the_budgets_say_they_built_nothing(feasibility, retrieval) -> None:
    assert feasibility["builds_nothing"] is True
    assert retrieval["retrieves_nothing"] is True


def test_the_substrate_disclaims_being_a_headroom_number(substrate) -> None:
    """It is graph statistics with no query and no candidate pool behind it.
    Reading a ceiling off it would be inventing one."""
    note = substrate["not_a_headroom_number"].lower()
    assert "no queries" in note
    assert "no retrieval" in note


# ── the substrate agrees with the frozen adoption record ─────────────────────


def test_metaqa_matches_the_frozen_adoption_numbers(substrate) -> None:
    """Three quantities re-derived by a different script on a different pass.
    If the reader is wrong, these are what catch it."""
    metaqa = substrate["datasets"]["metaqa"]["families"]["structural"]
    assert metaqa["edge_rows"] == 133_582
    relations = metaqa["relations"]
    assert relations["distinct"] == 9
    assert relations["singletons"] == 0
    assert relations["entropy"]["normalised"] == pytest.approx(0.8349, abs=5e-5)


def test_metaqa_rows_are_all_distinct_triples(substrate) -> None:
    """8,902 repeated (u, v) pairs but zero exact duplicates: MetaQA stores
    parallel typed edges. Counting those as duplicates would file the structure
    Direction B depends on as a defect."""
    pairs = substrate["datasets"]["metaqa"]["families"]["structural"]["pairs"]
    assert pairs["repeated_node_pairs"] == 8_902
    assert pairs["repeat_kind"] == "parallel_typed_edges"
    assert pairs["exact_duplicate_rows"] == 0


def test_no_family_on_any_dataset_has_exact_duplicate_rows(substrate) -> None:
    for name, entry in substrate["datasets"].items():
        for family, row in entry["families"].items():
            pairs = row["pairs"]
            if pairs.get("status") != "MEASURED":
                continue
            assert pairs["exact_duplicate_rows"] == 0, f"{name}/{family}"


def test_the_edge_family_matrix_is_what_the_declaration_recorded(substrate) -> None:
    absent = {
        name: set(entry["families_absent"]) for name, entry in substrate["datasets"].items()
    }
    assert absent["metaqa"] == {"ner"}
    assert absent["hotpotqa"] == {"knn"}
    assert absent["2wiki"] == {"knn"}
    assert absent["squad"] == set()
    assert absent["musique"] == set()


def test_only_metaqa_has_more_than_one_relation(substrate) -> None:
    """The finding that makes Direction B metaqa-only by necessity rather than
    by authorisation, and makes a metaqa null impossible to generalise."""
    counts = {
        name: entry["families"]["structural"]["relations"]["distinct"]
        for name, entry in substrate["datasets"].items()
    }
    assert counts["metaqa"] == 9
    assert all(v == 1 for k, v in counts.items() if k != "metaqa"), counts


# ── the knee Directions C and D have to respect ──────────────────────────────


def test_the_two_hop_walk_count_is_consistent_with_the_max_degree(substrate) -> None:
    """The worst single node is max_degree squared. If these ever disagree, one
    of the two was computed from a different degree vector."""
    for name, entry in substrate["datasets"].items():
        for family, row in entry["families"].items():
            worst = row["two_hop"]["max_single_node_undirected"]
            assert worst == row["undirected_degree"]["max"] ** 2, f"{name}/{family}"


def test_the_large_graphs_have_hubs_that_make_unbounded_two_hop_impossible(
    substrate,
) -> None:
    """A hub with 189,410 neighbours yields 3.6e10 two-hop walks from one node.
    An unbounded 2-hop prototype is not a budget question there, it is
    impossible -- which is a design constraint on Direction C, not a tuning
    detail."""
    for name in ("hotpotqa", "2wiki"):
        structural = substrate["datasets"][name]["families"]["structural"]
        assert structural["undirected_degree"]["max"] > 100_000, name
        assert structural["two_hop"]["max_single_node_undirected"] > 1e10, name


def test_degrees_are_heavy_tailed_so_the_mean_does_not_describe_the_cost(
    substrate,
) -> None:
    """Guards the reason the tail is reported at all. If p99.9 were near the
    mean, a mean-sized neighbourhood budget would be sound."""
    for name in ("hotpotqa", "2wiki", "musique"):
        d = substrate["datasets"][name]["families"]["structural"]["undirected_degree"]
        assert d["max"] > 100 * d["mean"], name


def test_no_family_is_stored_reciprocally(substrate) -> None:
    """Every graph here is directed. 'The neighbours of v' is therefore a real
    choice Direction C has to declare rather than inherit."""
    for name, entry in substrate["datasets"].items():
        for family, row in entry["families"].items():
            pairs = row["pairs"]
            if pairs.get("status") != "MEASURED":
                continue
            assert pairs["reciprocity"] < 0.5, f"{name}/{family}"


def test_ner_and_knn_weights_are_not_on_the_same_scale(substrate) -> None:
    """A single threshold across both families would be meaningless: KNN
    weights are cosines, NER weights run past 1."""
    knn_max = substrate["datasets"]["metaqa"]["families"]["knn"]["weights"]["max"]
    ner_max = substrate["datasets"]["2wiki"]["families"]["ner"]["weights"]["max"]
    assert knn_max <= 1.0
    assert ner_max > 1.0


# ── the rewrite, checked against the package and not against its own report ──


@needs_query_view_tree
def test_the_metaqa_rewrite_is_verified_against_real_canonical_ids() -> None:
    """The first version of this rewrite scored 0.4152 and made metaqa look
    unusable. Re-deriving it here from the package means a regression shows up
    as a test failure rather than as a plausible-looking rate."""
    import m3a_query_view_feasibility as feas

    spec = feas.BLOCKED["metaqa"]
    prefix, replacement = spec["rewrite"]
    node_ids = feas.canonical_node_ids(PACKAGE / "data" / "canonical" / "metaqa")

    # Zero-padded upstream, unpadded canonically. This is the exact case that
    # broke, so it is asserted in both directions.
    assert "metaqa_ent_4210" in node_ids
    assert "metaqa_ent_04210" not in node_ids

    padded = "metaqa:e04210"
    naive = replacement + padded[len(prefix) :]
    assert naive not in node_ids, "the naive prefix swap must still be wrong"

    body = padded[len(prefix) :]
    corrected = replacement + str(int(body))
    assert corrected in node_ids


@needs_query_view_tree
def test_measure_split_actually_resolves_a_padded_gold_id(tmp_path: Path) -> None:
    """Exercises the real code path rather than re-deriving the answer beside
    it. Flipping strip_zero_padding off has to fail here -- the assertions
    above would not notice, because they compute the corrected form
    themselves."""
    import m3a_query_view_feasibility as feas

    spec = feas.BLOCKED["metaqa"]
    node_ids = feas.canonical_node_ids(PACKAGE / "data" / "canonical" / "metaqa")

    path = tmp_path / "padded.jsonl"
    path.write_text(
        json.dumps(
            {
                "question": "what movies are about ginger rogers",
                # zero-padded, as upstream writes it
                "gold_node_ids": ["metaqa:e04210", "metaqa:e36865"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = feas.measure_split(path, spec, node_ids)
    assert row["gold_ids_total"] == 2
    assert row["gold_ids_resolved"] == 2
    assert row["queries_fully_resolved"] == 1
    assert row["queries_with_no_resolved_gold"] == 0


@needs_query_view_tree
def test_stripping_padding_never_empties_a_legitimate_zero_index() -> None:
    """lstrip('0') would turn `metaqa:e0` into `metaqa_ent_`. Node 0 exists."""
    import m3a_query_view_feasibility as feas

    node_ids = feas.canonical_node_ids(PACKAGE / "data" / "canonical" / "metaqa")
    assert "metaqa_ent_0" in node_ids
    assert "metaqa:e0"[len("metaqa:e") :].lstrip("0") == ""
    assert str(int("0")) == "0"


def test_every_gold_id_resolves_on_both_blocked_datasets(feasibility) -> None:
    for name, entry in feasibility["blocked"].items():
        totals = entry["totals"]
        assert totals["gold_resolution_rate"] == 1.0, name
        assert totals["gold_ids_unresolved"] == 0, name
        assert totals["queries_with_no_resolved_gold"] == 0, name


def test_no_gold_id_had_an_unexpected_prefix(feasibility) -> None:
    """A nonzero count would mean the id space has a second form the rewrite
    does not model, and the resolution rate would be measuring the wrong set."""
    for name, entry in feasibility["blocked"].items():
        assert entry["totals"]["gold_ids_with_unexpected_prefix"] == 0, name


def test_the_split_counts_match_the_declaration(feasibility) -> None:
    assert feasibility["blocked"]["metaqa"]["totals"]["rows"] == 407_513
    assert feasibility["blocked"]["2wiki"]["totals"]["rows"] == 192_606


def test_the_three_present_datasets_still_match_their_encodings(feasibility) -> None:
    for name, row in feasibility["present"].items():
        assert row["queries"] == row["dense"] == row["splade"], name


# ── the compute records ──────────────────────────────────────────────────────


def test_the_query_encode_stays_under_its_declared_ceiling(feasibility) -> None:
    total = sum(
        entry["encode_budget"]["gpu_hours_total"]["high"]
        for entry in feasibility["blocked"].values()
    )
    assert total < feasibility["hard_ceiling_gpu_hours"]


def test_retrieval_stays_under_its_declared_ceiling(retrieval) -> None:
    total = sum(
        row["dense_scoring"]["gpu_hours"]["high"] for row in retrieval["datasets"].values()
    )
    assert total < retrieval["hard_ceiling_gpu_hours"]


def test_retrieval_cost_is_not_priced_as_matmul_alone(retrieval) -> None:
    """On the large corpora the top-K and shard-read terms are a large share of
    the total. Pricing the matmul alone would understate them."""
    for name in ("hotpotqa", "2wiki"):
        d = retrieval["datasets"][name]["dense_scoring"]
        other = d["seconds_topk_bandwidth"]["high"] + d["seconds_shard_read"]["high"]
        assert other > 0.1 * d["seconds_matmul"]["high"], name


def test_document_encodings_are_complete_on_all_five(retrieval) -> None:
    """The reason blocker 2 is cheap. If this ever went false the budget would
    be missing a corpus encode worth hours, not minutes."""
    for name, row in retrieval["datasets"].items():
        assert row["encodings"]["complete"] is True, name
        assert row["encodings"]["dense"] == row["documents"], name


def test_projected_query_counts_are_labelled_as_projected(retrieval) -> None:
    """Two rows price queries that do not exist yet. A built count and a
    hoped-for count must never sit in one column unlabelled."""
    sources = {name: row["queries_source"] for name, row in retrieval["datasets"].items()}
    assert sources["metaqa"] == "projected"
    assert sources["2wiki"] == "projected"
    assert sources["squad"] == "on disk"
    assert sources["musique"] == "on disk"
    assert sources["hotpotqa"] == "on disk"


def test_the_projected_counts_agree_with_what_feasibility_measured(
    retrieval, feasibility
) -> None:
    for name in ("metaqa", "2wiki"):
        assert retrieval["datasets"][name]["queries"] == (
            feasibility["blocked"][name]["totals"]["rows"]
        ), name


# ── the batch simulation this reuses ─────────────────────────────────────────


def test_the_encode_budget_prices_padding_not_rows() -> None:
    """Reused from the WebQSP budget. Rows-based pricing is what made two very
    different WebQSP representations look identical, so the property is pinned
    here too: one long row in a batch pads every other row up to it."""
    import m3a_query_view_feasibility as feas

    uniform = np.full(64, 10, dtype=np.int64)
    with_outlier = uniform.copy()
    with_outlier[0] = 1000

    a = feas.sorted_batch_slots(uniform, feas.DENSE_MAX_TOKENS, feas.BATCH)
    b = feas.sorted_batch_slots(with_outlier, feas.DENSE_MAX_TOKENS, feas.BATCH)
    assert b["padded_token_slots"] > a["padded_token_slots"]
    assert b["peak_batch_slots"] > a["peak_batch_slots"]
