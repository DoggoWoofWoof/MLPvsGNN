"""Tests for the M3A headroom declaration and the pure parts of its runner.

configs/m3a_headroom.yaml is committed before it runs. These tests hold the
shape of that commitment against the amendment it is declared under: the
served freeze is pinned and a mismatch refuses to run, every population is a
labelled non-test split taken whole, the retrieval pools are the inherited ones
and not a wider set, the graph expansion is a fixed function of seeds and graph,
and the oracle topic-entity column is a diagnostic and never an input.

The runner tests exercise only the helpers that carry protocol meaning --
neighbour order, the two-hop cap, the fused-list prefix identity -- on toy
graphs, so they run without the package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
COMPILATION = ROOT / "configs" / "m3a_compilation.yaml"
INHERITED = ROOT / "configs" / "candidate_headroom.yaml"
RUNNER = ROOT / "scripts" / "m3a_headroom.py"


@pytest.fixture(scope="module")
def headroom() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def amendment() -> dict:
    return yaml.safe_load(COMPILATION.read_text(encoding="utf-8"))["amendment_1_2026_09_13"]


@pytest.fixture(scope="module")
def inherited() -> dict:
    return yaml.safe_load(INHERITED.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def runner():
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("m3a_headroom", RUNNER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


# ── the declaration ──────────────────────────────────────────────────────────


def test_it_is_declared_under_the_amendment_and_defines_no_ceiling(headroom: dict) -> None:
    assert headroom["phase"] == "M3A-COMPILATION"
    assert headroom["declared_under"].startswith("configs/m3a_compilation.yaml#amendment_1_2026_09_13")
    assert headroom["defines_no_new_ceiling"] is True
    assert headroom["zero_gpu"] is True
    for name in ("present_counts", "headroom_metrics", "regime_headroom", "full_coverage_ceiling", "pool_movement", "rrf_rankings"):
        assert any(entry.endswith("." + name) for entry in headroom["reuses"]), name


def test_the_status_is_one_of_the_two_honest_values(headroom: dict) -> None:
    assert headroom["status"] in {"DECLARED_NOT_RUN", "RUN"}
    if headroom["status"] == "RUN":
        assert "run_record" in headroom, "a RUN status must carry the output sha256s and cost"


def test_the_served_freeze_is_pinned_and_a_mismatch_refuses_to_run(headroom: dict, amendment: dict) -> None:
    sub = headroom["substrate"]
    assert sub["access_mode"] == "READ_ONLY"
    assert sub["on_freeze_mismatch"] == "REFUSE_TO_RUN"
    assert sub["freeze_RECORD_SHA256_expected"] == amendment["served_substrate"]["freeze_RECORD_SHA256"]
    assert len(sub["freeze_RECORD_SHA256_expected"]) == 64


def test_every_population_is_the_amended_split_taken_whole(headroom: dict, amendment: dict) -> None:
    declared = amendment["evaluation_population"]["populations"]
    assert set(headroom["populations"]["eval_splits"]) == set(declared) == {"metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"}
    for name, split in headroom["populations"]["eval_splits"].items():
        assert split == declared[name]["split"], name
        assert split != "test"
    assert headroom["populations"]["eval_splits"]["webqsp"] == "train_holdout"


def test_the_webqsp_carve_is_pinned_by_hash_and_count(headroom: dict) -> None:
    carve = headroom["populations"]["webqsp_train_holdout"]
    assert carve["expected_queries"] == 1549
    assert len(carve["expected_carve_ids_sha256"]) == 64
    assert carve["on_mismatch"] == "REFUSE_TO_RUN"
    assert "every second id from index 0" in carve["carve"]


def test_legacy_continuity_is_a_paired_column_on_the_five_text_datasets(headroom: dict) -> None:
    paired = headroom["populations"]["paired_column"]
    assert paired["label"] == "NOT_AN_EVALUATION_POPULATION"
    assert set(paired["present_on"]) == {"metaqa", "squad", "musique", "hotpotqa", "2wiki"}
    assert "webqsp" not in paired["present_on"]


def test_membership_is_by_position_and_webqsp_gold_positions_are_a_cross_check(headroom: dict) -> None:
    gold = headroom["gold"]
    assert gold["field"] == "gold_node_ids"
    assert gold["membership"].startswith("by canonical position")
    assert "never used as the source" in gold["webqsp_gold_positions"]


def test_the_retrieval_pools_are_the_inherited_ones_and_no_wider(headroom: dict, inherited: dict) -> None:
    rp = headroom["retrieval_pools"]
    fusion = inherited["fusion"]
    assert rp["equal_rrf"]["budgets"] == fusion["budgets"] == [50, 100, 200, 400]
    assert rp["equal_rrf"]["constant"] == fusion["rrf_constant"] == 60
    assert rp["equal_rrf"]["dense_weight"] == fusion["dense_weight"] == 0.5
    assert rp["equal_rrf"]["splade_weight"] == fusion["splade_weight"] == 0.5
    assert rp["equal_rrf"]["fused_lists"] == ["dense_top200", "splade_top200"]
    assert rp["equal_rrf"]["top_k"] == 400
    assert rp["ks"] == inherited["reporting"]["ks"] == [1, 5, 20]
    assert rp["prefix_depths_for_R4"] == [1, 5, 10, 20, 50, 100, 200, 400, 1000]
    assert rp["seeds"].startswith("dense_top5 union splade_top5")
    assert "gold" not in rp["seeds"]


def test_the_five_regimes_match_the_amended_live_list(headroom: dict) -> None:
    live = yaml.safe_load(COMPILATION.read_text(encoding="utf-8"))["headroom"]["regimes"]
    regimes = {k: v for k, v in headroom["graph_regimes"].items() if k.isupper()}
    assert list(regimes) == live == ["RETRIEVAL", "STRUCT", "NER", "KNN", "FULL"]
    assert regimes["RETRIEVAL"] == []
    assert regimes["FULL"] == ["structural", "ner", "knn"]
    assert regimes["STRUCT"] == ["structural"] and regimes["NER"] == ["ner"] and regimes["KNN"] == ["knn"]


def test_the_expansion_is_a_fixed_function_of_seeds_and_graph(headroom: dict) -> None:
    gr = headroom["graph_regimes"]
    names = [s["name"] for s in gr["expansion_settings"]]
    assert names == ["h1_c25", "h1_c100", "h2_c25"]
    for setting in gr["expansion_settings"]:
        assert setting["hops"] in (1, 2)
        assert setting["per_seed_cap"] > 0
        if setting["hops"] == 2:
            assert setting["per_frontier_cap"] > 0 and setting["visited_cap"] > 0
        for forbidden in ("learned", "weight", "train", "model"):
            assert forbidden not in setting, forbidden
    assert gr["base_pools_for_exposure"] == ["equal_rrf_budget_50", "equal_rrf_budget_200"]
    assert "function of the seeds and the graph alone" in gr["expansion_is_independent_of_the_base_pool"]
    assert "0.90" in gr["knee_rule"] and "read, not chosen here" in gr["knee_rule"]


def test_the_reachability_diagnostic_is_the_inherited_one_with_declared_strides(headroom: dict) -> None:
    rd = headroom["reachability_diagnostic"]
    assert rd["definition"].startswith("mp_retrieval.candidate_headroom.missing_gold_reachability")
    assert rd["max_hops"] == 3
    assert rd["missing_relative_to"] == "frozen_union"
    assert "every 4th query" in rd["population"]["hotpotqa"]
    assert "every 6th query" in rd["population"]["2wiki"]
    for name in ("metaqa", "squad", "musique", "webqsp"):
        assert rd["population"][name] == "full eval split"


def test_the_oracle_topic_entity_column_is_a_diagnostic_and_never_an_input(headroom: dict) -> None:
    oracle = headroom["oracle_topic_entity_exposure"]
    assert oracle["status"] == "DIAGNOSTIC_COLUMN_ONLY"
    assert "seed" in oracle["never"] and "candidate pool" in oracle["never"]
    assert set(oracle["datasets"]) == {"metaqa", "webqsp"}
    assert oracle["graph"].startswith("structural")
    assert oracle["hops"] == [1, 2]


def test_provenance_and_execution_discipline_are_declared(headroom: dict) -> None:
    prov = headroom["outputs"]["provenance_on_every_file"]
    for field in ("freeze_RECORD_SHA256", "config_sha256", "git_commit", "runner_sha256"):
        assert field in prov
    ex = headroom["execution"]
    assert ex["order"].startswith("validate on squad and musique first")
    assert "one dataset in memory at a time" in ex["memory_discipline"]


# ── the pure helpers of the runner ───────────────────────────────────────────


def _toy_graph(src, dst, weight=None, n=8):
    return SimpleNamespace(
        src=np.asarray(src, dtype=np.int32),
        dst=np.asarray(dst, dtype=np.int32),
        weight=None if weight is None else np.asarray(weight, dtype=np.float32),
        n_nodes=n,
        directed=weight is None,
    )


def test_family_csr_is_symmetric_and_orders_weighted_neighbours_by_weight_then_position(runner) -> None:
    # 0-1 (w .2), 0-2 (w .9), 0-3 (w .9), 2-3 (w .5); stored one row per pair
    csr = runner.FamilyCSR(_toy_graph([0, 0, 0, 2], [1, 2, 3, 3], [0.2, 0.9, 0.9, 0.5]), "knn")
    assert csr.col[csr.indptr[0] : csr.indptr[1]].tolist() == [2, 3, 1]  # .9@2, .9@3, .2@1
    assert csr.col[csr.indptr[1] : csr.indptr[2]].tolist() == [0]  # reverse direction present
    assert csr.col[csr.indptr[3] : csr.indptr[4]].tolist() == [0, 2]  # .9 then .5
    assert csr.capped(np.asarray([0]), 2).tolist() == [2, 3]


def test_family_csr_collapses_parallel_typed_edges_and_orders_by_position(runner) -> None:
    # 0->1 twice (two relation types), 1->0 once, 2->0
    csr = runner.FamilyCSR(_toy_graph([0, 0, 1, 2], [1, 1, 0, 0]), "structural")
    assert csr.col[csr.indptr[0] : csr.indptr[1]].tolist() == [1, 2]
    assert csr.col[csr.indptr[1] : csr.indptr[2]].tolist() == [0]
    assert int(csr.indptr[-1]) == 4


def test_capped_multi_is_node_major_then_family_then_rank(runner) -> None:
    struct = runner.FamilyCSR(_toy_graph([0, 0, 1], [4, 5, 6]), "structural")
    knn = runner.FamilyCSR(_toy_graph([0, 1], [7, 5], [0.5, 0.5]), "knn")
    out = runner.capped_multi(np.asarray([1, 0]), [struct, knn], 25)
    # node 1: struct [6], knn [5]; node 0: struct [4, 5], knn [7]
    assert out.tolist() == [6, 5, 4, 5, 7]


def test_expand_two_hops_matches_the_per_node_reference_and_respects_the_visited_cap(runner) -> None:
    rng = np.random.default_rng(0)
    n = 60
    src = rng.integers(0, n, 400)
    dst = rng.integers(0, n, 400)
    keep = src != dst
    struct = runner.FamilyCSR(_toy_graph(src[keep], dst[keep], n=n), "structural")
    w = rng.random(int(keep.sum())).astype(np.float32)
    knn = runner.FamilyCSR(_toy_graph(src[keep], dst[keep], w, n=n), "knn")
    families = [struct, knn]

    def reference(seeds, setting):
        cap = setting["per_seed_cap"]
        hop1 = runner.first_occurrence_unique(np.concatenate([np.concatenate([f.capped(np.asarray([s]), cap) for f in families]) for s in seeds]))
        if setting["hops"] == 1:
            return hop1
        seen = set(np.concatenate((seeds, hop1)).tolist())
        out, budget = [hop1], setting["visited_cap"] - len(seen)
        for node in hop1:
            if budget <= 0:
                break
            nbrs = runner.first_occurrence_unique(np.concatenate([f.capped(np.asarray([node]), setting["per_frontier_cap"]) for f in families]))
            fresh = [v for v in nbrs.tolist() if v not in seen][:budget]
            if fresh:
                seen.update(fresh)
                budget -= len(fresh)
                out.append(np.asarray(fresh, dtype=np.int64))
        return np.concatenate(out)

    for cap_visited in (12, 25, 60, 1000):
        setting = {"name": "t", "hops": 2, "per_seed_cap": 3, "per_frontier_cap": 2, "visited_cap": cap_visited}
        for _ in range(20):
            seeds = rng.choice(n, size=4, replace=False).astype(np.int64)
            fast = runner.expand(seeds, families, setting)
            assert np.array_equal(fast, reference(seeds, setting))
            hop1_only = np.union1d(seeds, runner.expand(seeds, families, {**setting, "hops": 1})).size
            assert np.union1d(seeds, fast).size <= max(cap_visited, hop1_only)
            assert np.unique(fast).size == fast.size


def test_rrf_fused_rows_prefix_equals_the_inherited_budget_pool(runner) -> None:
    from mp_retrieval.rank_fusion import rrf_rankings

    rng = np.random.default_rng(1)
    dense = np.stack([rng.permutation(300)[:200] for _ in range(7)])
    splade = np.stack([rng.permutation(300)[:200] for _ in range(7)])
    fused = runner.rrf_fused_rows(dense, splade, 60)
    ranked = rrf_rankings(dense, splade, dense_weights=[0.5], constant=60, top_k=400)[0.5]
    for q in range(7):
        union = np.union1d(dense[q], splade[q]).size
        assert fused[q].size == union
        for budget in (50, 100, 200, 400):
            assert np.array_equal(fused[q][:budget], ranked[q, : min(budget, union)])
        assert np.unique(fused[q]).size == fused[q].size


def test_the_seed_rule_is_dense_top5_then_splade_top5_first_occurrence(runner) -> None:
    dense = np.asarray([[1, 2, 3, 4, 5]])
    splade = np.asarray([[3, 9, 1, 8, 7]])
    assert runner.stable_union_rows(dense, splade)[0].tolist() == [1, 2, 3, 4, 5, 9, 8, 7]
