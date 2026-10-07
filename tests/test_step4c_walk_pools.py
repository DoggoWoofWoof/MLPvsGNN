"""Step 4c's pools (scripts/step4c_walk_pools.py, docs/STEP4C_WALK_POOLS_RETRAINED.md sections 2 to 4): a question's
P_F pool is base and seeds and the first B_q nodes of its walk order, at its frozen pool's size unless the walk offers
fewer; the replacement stops on any filed value it does not find; the gate's row counts, the pools file's manifest
check, the cache tiling, the pick rule and the labels."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


S4C = load("step4c_walk_pools", ROOT / "scripts" / "step4c_walk_pools.py")
M3C = load("m3b_compile", ROOT / "scripts" / "m3b_compile.py")
G1 = load("step1_grade", ROOT / "scripts" / "step1_grade.py")
CFG_H = {"retrieval_pools": {"equal_rrf": {"constant": 60}}}

IDS = ["q0", "q1", "q2"]
BASES = [np.array([1, 2, 3]), np.array([7, 8]), np.array([10])]
SEEDS = [np.array([3, 9]), np.array([8]), np.array([11])]
FROZEN_EXP = [np.array([4, 5, 20]), None, np.array([12, 13, 14, 15])]
ORDERS = [np.array([30, 5, 31, 40]), np.array([50, 51]), np.array([16])]


def frozen():
    inc = [M3C.build_pool(b, s, e)[0] for b, s, e in zip(BASES, SEEDS, FROZEN_EXP)]
    bs = [np.union1d(b, s) for b, s in zip(BASES, SEEDS)]
    return inc, bs


class Prep:
    def __init__(self, inc, ids=IDS):
        self.pop = types.SimpleNamespace(ids=list(ids))
        self.dense_ids = np.zeros((len(ids), 5), dtype=np.int64)
        self.splade_ids = np.zeros((len(ids), 5), dtype=np.int64)
        self.seeds = [s.copy() for s in SEEDS]
        self.pools = [p.copy() for p in inc]
        self.seeds_added = np.full(len(ids), -1)


def stub(bases=BASES):
    return types.SimpleNamespace(base_rows=lambda construction, d, s, m3a, contract, constant: [b.copy() for b in bases],
                                 build_pool=M3C.build_pool)


def filed_now():
    inc, bs = frozen()
    return {k: v.copy() for k, v in S4C.filed_arrays(IDS, bs, inc, ORDERS).items()}, inc


def test_filed_arrays_cut_each_order_at_the_frozen_size():
    filed, inc = filed_now()
    assert filed["inc_size"].tolist() == [7, 2, 6]
    assert filed["bs_size"].tolist() == [4, 2, 2]
    assert filed["B_q"].tolist() == [3, 0, 4]
    assert filed["count"].tolist() == [3, 0, 1]                   # q2's walk offers one node of four: short
    assert filed["nodes"].tolist() == [30, 5, 31, 16] and filed["ptr"].tolist() == [0, 3, 3, 4]
    assert filed["changed"].tolist() == [True, False, True]
    assert [x.decode() for x in filed["inc_sha"]] == [S4C.pool_sha(p) for p in inc]
    assert S4C.expected_rows(filed).tolist() == [7, 2, 3]


def test_replace_pools_puts_base_seeds_and_the_expansion_in_place():
    filed, inc = filed_now()
    prep = Prep(inc)
    stats = S4C.replace_pools(prep, filed, stub(), {}, CFG_H, None, None)
    assert stats == {"questions": 3, "changed": 2, "short": 1}
    assert prep.pools[0].tolist() == [1, 2, 3, 5, 9, 30, 31] and prep.pools[0].size == inc[0].size
    assert prep.pools[1].tolist() == inc[1].tolist() == [7, 8]
    assert prep.pools[2].tolist() == [10, 11, 16]
    assert prep.seeds_added.tolist() == [1, 0, 1]                 # m3b_compile.build_pool's count, as prepare files it
    assert all(np.array_equal(p, np.unique(p)) for p in prep.pools)


def test_a_question_order_other_than_the_files_is_found_by_id():
    filed, inc = filed_now()
    order = [2, 0, 1]
    prep = Prep([inc[k] for k in order], [IDS[k] for k in order])
    prep.seeds = [SEEDS[k].copy() for k in order]
    stats = S4C.replace_pools(prep, filed, stub([BASES[k] for k in order]), {}, CFG_H, None, None)
    assert stats["changed"] == 2 and prep.pools[0].tolist() == [10, 11, 16]


@pytest.mark.parametrize("breakage", ["missing id", "other base", "other frozen pool", "node in base and seeds",
                                      "repeated node", "other length"])
def test_replace_pools_stops_on_anything_not_filed(breakage):
    filed, inc = filed_now()
    prep, m3c = Prep(inc), stub()
    if breakage == "missing id":
        filed["ids"] = np.asarray(["q0", "qX", "q2"])
    elif breakage == "other base":
        m3c = stub([np.array([1, 2, 4]), BASES[1], BASES[2]])
    elif breakage == "other frozen pool":
        prep.pools[0] = np.array([1, 2, 3, 4, 5, 9, 21])
    elif breakage == "node in base and seeds":
        filed["nodes"][0] = 9
    elif breakage == "repeated node":
        filed["nodes"][1] = 30
    else:
        filed["count"][0] = 2
    with pytest.raises(SystemExit):
        S4C.replace_pools(prep, filed, m3c, {}, CFG_H, None, None)


def test_expansions_and_pool_sha():
    ptr, flat, count = S4C.expansions([np.array([5, 6, 7]), np.array([], dtype=np.int64), np.array([9])], [2, 3, 0])
    assert ptr.tolist() == [0, 2, 2, 2] and flat.tolist() == [5, 6] and count.tolist() == [2, 0, 0]
    a = np.array([1, 4, 9], dtype=np.int32)
    assert S4C.pool_sha(a) == S4C.pool_sha(a.astype(np.int64)) != S4C.pool_sha(np.array([1, 4, 10]))


def test_load_filed_checks_the_manifest(tmp_path):
    filed, _inc = filed_now()
    np.savez_compressed(tmp_path / "metaqa__s1eval.npz", **filed)
    S4C.write_json(tmp_path / "pools.json", {"files": {"metaqa__s1eval": {
        "file": "metaqa__s1eval.npz", "sha256": S4C.sha_file(tmp_path / "metaqa__s1eval.npz")}}})
    got, ent, msha = S4C.load_filed("metaqa", "s1eval", tmp_path)
    assert all(np.array_equal(got[k], filed[k]) for k in filed) and msha == S4C.sha_file(tmp_path / "pools.json")
    with pytest.raises(SystemExit):
        S4C.load_filed("metaqa", "s1sel", tmp_path)
    filed["nodes"][0] = 31
    np.savez_compressed(tmp_path / "metaqa__s1eval.npz", **filed)
    with pytest.raises(SystemExit):
        S4C.load_filed("metaqa", "s1eval", tmp_path)


def write_part(root, i, n, qr, ids, rows):
    d = root / "metaqa" / "s1eval" / f"part_{i}of{n}"
    d.mkdir(parents=True)
    np.save(d / "n.npy", np.asarray(rows, np.int32))
    (d / "record.json").write_text(json.dumps({"query_range": qr, "ids": ids, "look": {"carve_queries": 3}}), encoding="utf-8")


def test_carve_rows_join_the_parts_in_order(tmp_path):
    write_part(tmp_path, 1, 2, [2, 3], ["q2"], [3])
    write_part(tmp_path, 0, 2, [0, 2], ["q0", "q1"], [7, 2])
    ids, n, parts, _recs = S4C.carve_rows(tmp_path, "metaqa", "s1eval")
    assert ids == IDS and n.tolist() == [7, 2, 3] and [p.name for p in parts] == ["part_0of2", "part_1of2"]


def test_carve_rows_stop_on_a_gap(tmp_path):
    write_part(tmp_path, 0, 2, [0, 1], ["q0"], [7])
    write_part(tmp_path, 1, 2, [2, 3], ["q2"], [3])
    with pytest.raises(SystemExit):
        S4C.carve_rows(tmp_path, "metaqa", "s1eval")


def test_every_look_shard_files_its_pools_record(tmp_path, monkeypatch):
    monkeypatch.setattr(S4C, "LOOK", tmp_path)
    d = tmp_path / "metaqa" / "s1eval"
    d.mkdir(parents=True)
    ent = {"sha256": "a" * 64, "changed": 2, "short": 1}
    for tag in ("_0of2", "_1of2"):
        (d / f"record{tag}.json").write_text("{}", encoding="utf-8")
    (d / "pools_0of2.json").write_text(json.dumps({"pools_sha256": "a" * 64, "manifest_sha256": "m", "changed": 2,
                                                   "short": 1}), encoding="utf-8")
    assert S4C.look_pools_records("metaqa", "s1eval", ent)                 # shard 1 has no pools record
    (d / "pools_1of2.json").write_text(json.dumps({"pools_sha256": "a" * 64, "manifest_sha256": "m", "changed": 2,
                                                   "short": 1}), encoding="utf-8")
    assert S4C.look_pools_records("metaqa", "s1eval", ent) == []
    assert S4C.look_pools_records("metaqa", "s1eval", dict(ent, sha256="b" * 64))
    assert S4C.look_pools_records("metaqa", "s1eval", dict(ent, short=0))


@pytest.mark.parametrize("verdict, rule", [("ADOPT", "D"), ("NOT_ADOPTED", "M"), ("NO_EFFECT", "M")])
def test_the_pick_rule_is_the_one_step1_adopts(tmp_path, monkeypatch, verdict, rule):
    monkeypatch.setattr(S4C, "STEP1_GRADE", tmp_path / "grade.json")
    assert S4C.pick_rule() == (None, None)
    (tmp_path / "grade.json").write_text(json.dumps({"verdict": verdict}), encoding="utf-8")
    assert S4C.pick_rule() == (rule, verdict)


def test_labels_are_step1s():
    rng = np.random.default_rng(0)
    b = rng.random((400, 3))
    up = b.copy()
    up[:, 0] += 0.1
    assert S4C.compare(G1, up, b)[1] == "ABOVE"
    assert S4C.compare(G1, b, up)[1] == "BELOW"
    diff, lab = S4C.compare(G1, b, b.copy())
    assert lab == "AT" and diff["R@5"]["mean"] == 0.0 and set(diff) == set(G1.METRICS)
