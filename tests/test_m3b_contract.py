"""The contract script's rule and the two readings (rule as filed; ruled family)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load():
    spec = importlib.util.spec_from_file_location("m3b_contract", ROOT / "scripts" / "m3b_contract.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["m3b_contract"] = module
    spec.loader.exec_module(module)
    return module


CFG = yaml.safe_load((ROOT / "configs" / "m3b_controlled_comparison.yaml").read_text(encoding="utf-8"))
CFG_H = yaml.safe_load((ROOT / "configs" / "m3a_headroom.yaml").read_text(encoding="utf-8"))


def cell(pool, regime, setting, base, frac, cand, spec=None):
    c = {"pool": pool, "regime": regime, "setting": setting, "base_pool": base, "fraction_of_attainable@5": frac,
         "candidates_mean": cand, "recall_ceiling@5": frac * 0.8, "source": "test"}
    if spec:
        c["setting_spec"] = spec
    return c


H3 = {"name": "h3_c25_v2000", "hops": 3, "per_seed_cap": 25, "per_frontier_cap": 25, "visited_cap": 2000}
CELLS = [
    cell("equal_rrf_budget_50", "RETRIEVAL", "-", "-", 0.03, 50),
    cell("equal_rrf_budget_200+FULL:h2_c25", "FULL", "h2_c25", "equal_rrf_budget_200", 0.902, 1290),
    cell("equal_rrf_budget_200+STRUCT:h2_c25", "STRUCT", "h2_c25", "equal_rrf_budget_200", 0.86, 762),
    cell("equal_rrf_budget_50+STRUCT:h3_c25_v2000", "STRUCT", "h3_c25_v2000", "equal_rrf_budget_50", 0.978, 2017, H3),
    cell("equal_rrf_budget_200+STRUCT:h3_c25_v2000", "STRUCT", "h3_c25_v2000", "equal_rrf_budget_200", 0.978, 2153, H3),
    cell("equal_rrf_budget_50+STRUCT:h3_c25_v4000", "STRUCT", "h3_c25_v4000", "equal_rrf_budget_50", 0.988, 3468, {**H3, "name": "h3_c25_v4000", "visited_cap": 4000}),
]


def test_the_rule_as_filed_takes_the_smallest_pool_at_the_knee_within_the_bound():
    m = load()
    v = m.apply_rule(CELLS, 0.90, 2500.0)
    assert v["chosen"]["pool"] == "equal_rrf_budget_200+FULL:h2_c25"
    assert v["cells_within_bound"] == 5 and v["cells_reaching"] == 3


def test_the_ruled_reading_applies_the_same_rule_within_the_named_family():
    m = load()
    r = m.readings(CELLS, "metaqa", CFG, CFG_H)
    assert r["rule_as_filed"]["chosen"]["pool"] == "equal_rrf_budget_200+FULL:h2_c25"
    assert r["ruled_family"] == {"regime": "STRUCT", "hops": 3}
    assert r["ruled_reading"]["chosen"]["pool"] == "equal_rrf_budget_50+STRUCT:h3_c25_v2000"
    assert r["ruled_reading"]["cells_in_family"] == 3


def test_datasets_without_a_ruled_family_keep_the_rule_as_filed():
    m = load()
    r = m.readings(CELLS, "squad", CFG, CFG_H)
    assert r["ruled_reading"] is None and r["ruled_family"] is None
    record = m.finish_record({}, CELLS, "squad", CFG, CFG_H)
    assert record["reading"] == "rule_as_filed"
    assert record["construction"] == {"base_pool": "equal_rrf_budget_200", "regime": "FULL", "setting": next(s for s in CFG_H["graph_regimes"]["expansion_settings"] if s["name"] == "h2_c25")}


def test_webqsp_family_is_a_base_pool_prefix_predicate():
    m = load()
    fam = CFG["amendment_1_2026_09_13"]["candidate_contract_amended"]["ruled_reading"]["webqsp"]
    assert m.in_ruled_family(cell("dense_top100", "RETRIEVAL", "-", "-", 0.4, 100), fam, CFG_H)
    assert m.in_ruled_family(cell("wrrf0.75_budget_100+STRUCT:h1_c100", "STRUCT", "h1_c100", "wrrf0.75_budget_100", 0.7, 300), fam, CFG_H)
    assert not m.in_ruled_family(cell("equal_rrf_budget_200+STRUCT:h1_c100", "STRUCT", "h1_c100", "equal_rrf_budget_200", 0.79, 275), fam, CFG_H)
    assert not m.in_ruled_family(cell("splade_top50", "RETRIEVAL", "-", "-", 0.1, 50), fam, CFG_H)
