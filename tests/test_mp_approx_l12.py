"""MP-Approx level 12, population module (configs/mp_approx_l12.yaml#tests): the declaration, its pins and its constants;
the population rule on toy ids; the carve rule, the train-branch replica and the carve checks on a toy dataset; the
scoring pass's rebinding; the loopcheck's comparison and the training-cache check; the check and the carve check on
synthetic combined sidecars and their refusals; the stages' machine; and no held, earlier level's or train-split query
in the dev sidecar, and only pinned train-split ids in the carve sidecars."""

from __future__ import annotations

import copy
import dataclasses
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l0 as L0  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
import mp_approx_l9 as L9  # noqa: E402
import mp_approx_l10 as L10  # noqa: E402
import mp_approx_l11 as L11  # noqa: E402
import mp_approx_l12 as L12  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)
import universal_v2_run as U  # noqa: E402  (the frozen runner, for m3b_compile)
from mp_retrieval import m3b_pools  # noqa: E402

_quiet = T8._quiet


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, L12):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])


def _stopped(stops) -> str:
    return (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the declaration, its pins and its constants ──────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256(stops):
    decl = L12.load_declaration()
    assert decl["phase"] == "MP_APPROX_L12" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L8.load_declaration()["registered_question"]
    assert list(decl["inputs"]["relations"]["order"]) == list(L8.REL_ORDER)
    L12.verify_inputs(decl)   # level 11's check on this copy, level 11's own files and rows, and the training caches
    assert not (stops / "hard_stops.json").exists()


@pytest.mark.parametrize("key", ["script_lf", "tests_lf", "declaration_lf", "fit_script_lf", "fit_tests_lf"])
def test_a_changed_level11_pin_stops(stops, monkeypatch, key):
    monkeypatch.setattr(L12, "_L11_VERIFY", lambda decl: None)   # the copied pins are the test above's
    decl = L12.load_declaration()
    decl["inputs"]["level11"][key]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.verify_inputs(decl)
    assert decl["inputs"]["level11"][key]["path"] in _stopped(stops)


def test_a_changed_rows_pin_a_changed_cache_pin_and_a_changed_copied_pin_stop(stops, monkeypatch):
    decl = L12.load_declaration()
    decl["inputs"]["level10"]["script_lf"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.verify_inputs(decl)
    assert "scripts/mp_approx_l10.py" in _stopped(stops)
    monkeypatch.setattr(L12, "_L11_VERIFY", lambda decl: None)
    decl = L12.load_declaration()
    decl["inputs"]["level11"]["excluded_rows"]["qids"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.verify_inputs(decl)
    assert "outputs/mp_approx_l11/metaqa/qids.json" in _stopped(stops)
    for carve, f in (("select", "qemb.npy"), ("fit", "query_ids.json")):
        decl = L12.load_declaration()
        decl["inputs"]["carve_cache"][carve][f] = "0" * 64
        with pytest.raises(SystemExit, match="HARD STOP"):
            L12.verify_inputs(decl)
        assert f"outputs/universal_v2/cache/metaqa/{carve}/{f}" in _stopped(stops)


def test_the_declared_constants_are_the_files():
    decl = L12.load_declaration()
    assert (L12.PER_HOP, L12.SALT, L12.NAME) == ({1: 221, 2: 1000, 3: 1000}, "mp_approx_l12|", "metaqa")
    assert {int(h): int(v) for h, v in decl["population"]["per_hop"].items()} == L12.PER_HOP
    assert {int(h): int(v) for h, v in decl["population"]["available_per_hop"].items()} == L12.AVAILABLE_PER_HOP
    assert all(L11.AVAILABLE_PER_HOP[h] - L11.PER_HOP == L12.AVAILABLE_PER_HOP[h] for h in (1, 2, 3))
    assert L12.PER_HOP[1] == L12.AVAILABLE_PER_HOP[1] and all(L12.PER_HOP[h] <= L12.AVAILABLE_PER_HOP[h] for h in (1, 2, 3))
    assert L12.FAMILIES == ("std", "nb") and L12.SCRIPT_REL == "scripts/mp_approx_l12.py"
    assert L12.LEVELS == ("level0", "level8", "level9", "level10", "level11")
    # the carves, the fits and the shards
    pins = decl["carves"]["pins"]
    assert tuple(k for k, v in pins.items() if isinstance(v, dict)) == L12.CARVES and L12.X_CARVES == tuple(f"x{j}" for j in range(1, 8))
    assert (pins["N"], pins["s_sel"], pins["s_fit"], pins["remaining"]) == (329282, 220, 55, 327785)
    assert decl["carves"]["fits"] == {"TW-1x": ["fit"], "TW-2x": ["fit", "x1"], "TW-4x": ["fit", "x1", "x2", "x3"],
                                      "TW-8x": ["fit", *L12.X_CARVES]}
    assert sum(L12.CARVE_SHARDS.values()) == 34 and L12.CARVE_SHARDS["select"] == 2 and L12.DEV_SHARDS == 2 and L12.LOOP_LIMIT == 24
    # the training caches: the fit and select carves, twelve files each, never the scalars or the seed weights
    cc = decl["inputs"]["carve_cache"]
    assert cc["dir"] == "outputs/universal_v2/cache/metaqa/" and tuple(k for k in cc if k != "dir") == L12.CACHE_CARVES
    assert all(tuple(cc[c]) == L12.CACHE_FILES for c in L12.CACHE_CARVES)
    assert L12.CACHE_ARRAYS == tuple(f[:-4] for f in L12.CACHE_FILES if f.endswith(".npy"))
    assert not {"scalars.npy", "seedw.npy"} & set(L12.CACHE_FILES)
    # the level 11 pins name level 11's committed files and rows
    lv = decl["inputs"]["level11"]
    assert (lv["declaration_lf"]["path"], lv["script_lf"]["path"], lv["tests_lf"]["path"], lv["fit_script_lf"]["path"],
            lv["fit_tests_lf"]["path"]) == ("configs/mp_approx_l11.yaml", "scripts/mp_approx_l11.py", "tests/test_mp_approx_l11.py",
                                            "scripts/mp_approx_l11_fit.py", "tests/test_mp_approx_l11_fit.py")
    assert lv["excluded_rows"]["qids"]["path"] == "outputs/mp_approx_l11/metaqa/qids.json"
    # every pin level 11 declares is copied unchanged (level 11's training sidecars are not an input here)
    l11 = L11.load_declaration()["inputs"]
    assert all(decl["inputs"][k] == l11[k] for k in l11 if k not in ("training_sidecars", "frozen_code_lf"))
    assert {k: v for k, v in decl["inputs"]["frozen_code_lf"].items() if k != "via_level0"} == \
        {k: v for k, v in l11["frozen_code_lf"].items() if k != "via_level0"}


# ── the population rule ──────────────────────────────────────────────────────


def _toy_population(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(500)]
    gate = rng.random(len(ids)) < 0.7
    excluded = L0.metaqa_subsample(ids, gate, per_hop=40)
    level8, _ = L8.l8_rows(ids, gate, excluded, per_hop=50)
    l8_ids = [ids[i] for i in level8]
    level9, _ = L9.l9_rows(ids, gate, excluded, per_hop=50, level8_ids=l8_ids, level8_per_hop=50)
    l9_ids = [ids[i] for i in level9]
    level10, _ = L10.l10_rows(ids, gate, excluded, per_hop=50, level8_ids=l8_ids, level9_ids=l9_ids, level8_per_hop=50,
                              level9_per_hop=50)
    level11, _ = L11.l11_rows(ids, gate, excluded, per_hop=50, level8_ids=l8_ids, level9_ids=l9_ids,
                              level10_ids=[ids[i] for i in level10], level8_per_hop=50, level9_per_hop=50, level10_per_hop=50)
    return ids, gate, excluded, level8, level9, level10, level11


def _rows(ids, gate, excluded, level8, level9, level10, level11, per_hop=None, **kw):
    return L12.l12_rows(ids, gate, excluded, per_hop or {1: 30, 2: 60, 3: 60}, level8_ids=[ids[i] for i in level8],
                        level9_ids=[ids[i] for i in level9], level10_ids=[ids[i] for i in level10],
                        level11_ids=[ids[i] for i in level11], level8_per_hop=50, level9_per_hop=50, level10_per_hop=50,
                        level11_per_hop=50, **kw)


def test_the_population_rule_leaves_out_levels_0_and_8_to_11_and_sorts_by_its_own_salt(stops):
    ids, gate, excluded, level8, level9, level10, level11 = _toy_population()
    rows, available = _rows(ids, gate, excluded, level8, level9, level10, level11)
    assert rows.size == 150 and np.all(np.diff(rows) > 0) and gate[rows].all()
    for earlier in (excluded, level8, level9, level10, level11):
        assert not np.isin(rows, earlier).any()
    out = set(excluded.tolist()) | set(level8.tolist()) | set(level9.tolist()) | set(level10.tolist()) | set(level11.tolist())
    for h, keep in ((1, 30), (2, 60), (3, 60)):
        pool = [int(i) for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == h and int(i) not in out]
        assert available[h] == len(pool) > keep
        want = sorted(pool, key=lambda i: hashlib.sha256(("mp_approx_l12|" + ids[i]).encode("utf-8")).hexdigest())[:keep]
        assert sorted(want) == [int(r) for r in rows if L0.hop_from_id(ids[r]) == h]
    again, _ = L12.l12_rows(list(ids), gate.copy(), excluded.copy(), {1: 30, 2: 60, 3: 60}, level8_per_hop=50, level9_per_hop=50,
                            level10_per_hop=50, level11_per_hop=50)
    assert np.array_equal(rows, again)
    # hop 1 takes every remaining row when its count is what is available
    every, _ = _rows(ids, gate, excluded, level8, level9, level10, level11, per_hop={1: available[1], 2: 60, 3: 60})
    hop1 = [int(r) for r in every if L0.hop_from_id(ids[r]) == 1]
    assert hop1 == sorted(i for i in range(len(ids)) if gate[i] and L0.hop_from_id(ids[i]) == 1 and i not in out)
    with pytest.raises(SystemExit, match="fewer than"):
        _rows(ids, gate, excluded, level8, level9, level10, level11, per_hop={1: available[1] + 1, 2: 60, 3: 60})
    assert not (stops / "hard_stops.json").exists()


def test_the_earlier_levels_recomputed_rows_must_be_their_pinned_ids(stops):
    ids, gate, excluded, level8, level9, level10, level11 = _toy_population(1)
    for cut, level in ((lambda r: (level8, level9, level10, r[1:]), "level 11's"), (lambda r: (level8, level9, r[1:], level11), "level 10's"),
                       (lambda r: (level8, r[1:], level10, level11), "level 9's"), (lambda r: (r[1:], level9, level10, level11), "level 8's")):
        rows = {"level 11's": level11, "level 10's": level10, "level 9's": level9, "level 8's": level8}[level]
        with pytest.raises(SystemExit, match="HARD STOP"):
            _rows(ids, gate, excluded, *cut(rows))
        assert f"{level} recomputed rows" in _stopped(stops)


# ── the carves ───────────────────────────────────────────────────────────────


def test_the_carve_rule_is_carve_ids_and_its_offsets_are_disjoint():
    source = sorted(f"metaqa:{h}hop:train:{i}" for h in (1, 2, 3) for i in range(333))[:997]
    for fit_cap in (20, 60, 100):
        rule = L12.carve_rule(source, 1500, 5, fit_cap)
        fit, select = m3b_pools.carve_ids(source, select_cap=1500, select_fraction=5, fit_cap=fit_cap)
        assert L12.carve_ids_of("fit", rule) == fit and L12.carve_ids_of("select", rule) == select
        assert rule["s_fit"] == max(1, -(-len(rule["remaining"]) // fit_cap)) and len(rule["remaining"]) + len(select) == len(source)
        offsets = [rule["remaining"][j::rule["s_fit"]] for j in range(rule["s_fit"])]
        assert sorted(q for o in offsets for q in o) == sorted(rule["remaining"])
        carves = {c: set(L12.carve_ids_of(c, rule)) for c in L12.CARVES if c in ("fit", "select") or int(c[1:]) < rule["s_fit"]}
        for a in carves:
            for b in carves:
                assert a == b or not (carves[a] & carves[b])
        with pytest.raises(ValueError):
            L12.carve_ids_of(f"x{rule['s_fit']}", rule)   # past the stride, an offset would reach another carve
    # the rule's arithmetic on metaqa's train-split size gives the pinned strides and sizes
    pins = L12.load_declaration()["carves"]["pins"]
    rule = L12.carve_rule([f"{i:07d}" for i in range(int(pins["N"]))], 1500, 5, 6000)
    assert (rule["s_sel"], rule["s_fit"], len(rule["remaining"])) == (pins["s_sel"], pins["s_fit"], pins["remaining"])
    assert all(len(L12.carve_ids_of(c, rule)) == pins[c]["queries"] + pins[c]["zero_gold_excluded"] for c in L12.CARVES)


class _ToyDataset:
    """The two things the train branch reads from a dataset: query_ids (the dataset rows) and queries(split)."""

    def __init__(self, train: list[str], other: list[str], seed: int = 0):
        order = list(train) + list(other)
        np.random.default_rng(seed).shuffle(order)
        self.query_ids = order
        self._rows = {"train": [{"query_id": q, "qtype": "movie_to_actor"} for q in train],
                      "dev": [{"query_id": q, "qtype": "movie_to_actor"} for q in other]}

    def queries(self, split: str):
        return iter(self._rows[split])


class _ToyM3a:
    """resolve_gold reads the query's golds from the positions map, so a query can be given none."""

    @staticmethod
    def resolve_gold(rows, positions, dataset):
        return [np.asarray(positions[r["query_id"]], dtype=np.int64) for r in rows]


def _toy_carves(seed=0, fit_cap=60, bad_hop1=False):
    rng = np.random.default_rng(seed)
    train = sorted(f"metaqa:{h}hop:{'trn' if bad_hop1 and h == 1 else 'train'}:{i}" for h in (1, 2, 3) for i in range(333))[:997]
    ds = _ToyDataset(train, [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(20)], seed)
    positions = {q: [] if rng.random() < 0.1 else rng.integers(0, 100, size=int(rng.integers(1, 4))).tolist() for q in train}
    m3b_compile = U.M3B_RUN.load_script("m3b_compile")
    cfg_m3b = {"populations": {"training_carves": {"fit": {"size_cap": fit_cap}}}}
    source, fit, select = m3b_compile.carves_for(ds, "metaqa", cfg_m3b)
    rule = L12.carve_rule(source, m3b_compile.SELECT_CAP, m3b_compile.SELECT_FRACTION, fit_cap)
    pins = {"N": len(source), "source_sha256": m3b_pools.ids_digest(source), "s_sel": rule["s_sel"], "s_fit": rule["s_fit"],
            "remaining": len(rule["remaining"])}
    for c in L12.CARVES:
        ids = L12.carve_ids_of(c, rule)
        kept = [q for q in ids if positions[q]]
        pins[c] = {"queries": len(kept), "zero_gold_excluded": len(ids) - len(kept), "ids_sha256": m3b_pools.ids_digest(kept),
                   "hops": L12.hop_counts(kept)}
    cfg = {"m3b_incumbents": {"training_carves_reused_here": {"metaqa": {
        "N": len(source), "fit": len(fit), "select": len(select), "fit_sha256": m3b_pools.ids_digest(fit),
        "select_sha256": m3b_pools.ids_digest(select)}}}}
    S = SimpleNamespace(m3b_compile=m3b_compile, ds=ds, cfg=cfg, cfg_m3b=cfg_m3b, cfg_h={}, m3a=_ToyM3a())
    return S, {"carves": {"pins": pins}}, positions, rule


def test_the_train_branch_replica_gives_m3b_compile_populations_carves():
    S, _decl, positions, rule = _toy_carves()
    m3b_compile, ds = S.m3b_compile, S.ds
    for kind in ("fit", "select"):
        ref = m3b_compile.population(ds, "metaqa", kind, S.cfg_m3b, {}, S.m3a, positions)
        mine = L12.carve_population(m3b_compile, ds, L12.carve_ids_of(kind, rule), kind, S.m3a, positions)
        assert L12.same_population(mine, ref) and ref.zero_gold_excluded > 0
    x1_ids = L12.carve_ids_of("x1", rule)
    x1 = L12.carve_population(m3b_compile, ds, x1_ids, "x1", S.m3a, positions)
    kept = [q for q in x1_ids if positions[q]]
    assert x1.ids == kept and x1.digest == m3b_pools.ids_digest(kept) and x1.n_before == len(x1_ids)
    assert x1.zero_gold_excluded == len(x1_ids) - len(kept) and x1.kind == "x1" and x1.dataset == "metaqa"
    assert np.array_equal(x1.idx, [ds.query_ids.index(q) for q in kept])
    assert all(np.array_equal(g, positions[q]) for g, q in zip(x1.golds, kept))
    # same_population sees every field
    for changed in (dataclasses.replace(x1, kind="x2"), dataclasses.replace(x1, idx=x1.idx + 1), dataclasses.replace(x1, ids=x1.ids[::-1]),
                    dataclasses.replace(x1, golds=[g + 1 for g in x1.golds]), dataclasses.replace(x1, n_before=x1.n_before + 1),
                    dataclasses.replace(x1, zero_gold_excluded=0), dataclasses.replace(x1, digest="0" * 64)):
        assert not L12.same_population(changed, x1)
    assert L12.same_population(dataclasses.replace(x1), x1)


def test_the_carve_populations_are_checked_against_the_pins(stops):
    S, decl, positions, _rule = _toy_carves(seed=2)
    for c in L12.CARVES:
        pop = L12.carve_population_checked(decl, S, positions, c)
        assert pop.kind == c and pop.digest == decl["carves"]["pins"][c]["ids_sha256"] and all(L12.is_train_id(q) for q in pop.ids)
    assert not (stops / "hard_stops.json").exists()
    refusals = (("N", lambda p: p.update(N=p["N"] + 1), "are not the pinned source"),
                ("s_fit", lambda p: p.update(s_fit=p["s_fit"] + 1), "strides are not the pinned ones"),
                ("x3 digest", lambda p: p["x3"].update(ids_sha256="0" * 64), "carve x3 is not its pin"),
                ("x3 hops", lambda p: p["x3"]["hops"].update({1: p["x3"]["hops"][1] + 1}), "carve x3 is not its pin"),
                ("x3 zero", lambda p: p["x3"].update(zero_gold_excluded=p["x3"]["zero_gold_excluded"] + 1), "carve x3 is not its pin"))
    for _name, change, text in refusals:
        bad = copy.deepcopy(decl)
        change(bad["carves"]["pins"])
        with pytest.raises(SystemExit, match="HARD STOP"):
            L12.carve_population_checked(bad, S, positions, "x3")
        assert text in _stopped(stops)
    S.cfg["m3b_incumbents"]["training_carves_reused_here"]["metaqa"]["fit_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.carve_population_checked(decl, S, positions, "select")
    assert "configs/universal_v2.yaml's digests" in _stopped(stops)


def test_x0_must_be_the_fit_carve_and_every_carve_id_a_train_split_id(stops, monkeypatch):
    S, decl, positions, _rule = _toy_carves(seed=3)
    real = L12.carve_population
    monkeypatch.setattr(L12, "carve_population", lambda *a: dataclasses.replace(real(*a), digest="0" * 64))
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.carve_population_checked(decl, S, positions, "x2")
    assert "x_0 through the train-branch lines" in _stopped(stops)
    monkeypatch.setattr(L12, "carve_population", real)
    S, decl, positions, _rule = _toy_carves(seed=3, bad_hop1=True)   # hop-1 ids are metaqa:1hop:trn:<n>
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.carve_population_checked(decl, S, positions, "x5")
    assert "is not a metaqa train-split id" in _stopped(stops)
    assert L12.is_train_id("metaqa:2hop:train:17") and not L12.is_train_id("metaqa:2hop:dev:17")
    assert not L12.is_train_id("webqsp:2hop:train:17") and not L12.is_train_id("metaqa:2hop:train")


# ── the scoring pass's rebinding ─────────────────────────────────────────────


def test_the_scoring_pass_rebinds_level_8s_names_only_inside_the_pass(tmp_path):
    names = ("l8_rows", "walk_entries", "verify_inputs", "CONFIG", "OUT", "DATA", "AVAILABLE_PER_HOP", "PER_HOP")
    before = {k: getattr(L8, k) for k in names}

    def rule(ids, gate, excluded, per_hop=None):
        return None

    with L12.level8_rebound(tmp_path / "x", rule):
        assert L8.l8_rows is rule and L8.walk_entries is L9.walk_entries_both and L8.verify_inputs is L12.verify_inputs
        assert (L8.CONFIG, L8.OUT, L8.DATA) == (L12.CONFIG, L12.OUT, tmp_path / "x") and L8.AVAILABLE_PER_HOP == L12.AVAILABLE_PER_HOP
        assert L8.PER_HOP == L12.PER_HOP and L9._L8_WALKS is before["walk_entries"]
    assert all(getattr(L8, k) is v for k, v in before.items())
    with pytest.raises(RuntimeError):
        with L12.level8_rebound(tmp_path / "x", rule):
            raise RuntimeError("inside the pass")
    assert all(getattr(L8, k) is v for k, v in before.items()) and L8.PER_HOP == 1000


def test_the_scoring_pass_reads_this_files_rule_per_hop_pins_and_paths_inside_it(tmp_path, monkeypatch, stops):
    seen = {}

    def fake_stage_score(decl, log, shard, limit, out_dir):
        seen.update(rows=L8.l8_rows, walks=L8.walk_entries, verify=L8.verify_inputs, config=L8.CONFIG, data=L8.DATA,
                    available=dict(L8.AVAILABLE_PER_HOP), per_hop=L8.PER_HOP, args=(shard, limit, out_dir))

    monkeypatch.setattr(L8, "stage_score", fake_stage_score)
    L12.stage_score(L12.load_declaration(), _quiet, (1, 2), None, tmp_path / "o")
    assert seen["walks"] is L9.walk_entries_both and seen["verify"] is L12.verify_inputs and seen["config"] == L12.CONFIG
    assert seen["data"] == tmp_path / "o" and seen["available"] == L12.AVAILABLE_PER_HOP and seen["per_hop"] == L12.PER_HOP
    assert seen["args"] == ((1, 2), None, tmp_path / "o") and seen["rows"].__qualname__.startswith("stage_score")
    assert L8.l8_rows is L9._L8_ROWS and L8.PER_HOP == 1000


# ── the loopcheck's comparison and the training-cache check ──────────────────


def test_the_loopcheck_comparison_finds_a_changed_byte_dtype_or_shape(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    T9._sidecar(a, monkeypatch, L9.walk_entries_both, n_q=8, seed=11)
    shutil.copytree(a, b)
    assert L12.array_differences(a, b) == []
    e = np.load(b / "e_node.npy")
    e[3] += 1
    np.save(b / "e_node.npy", e)
    np.save(b / "q_hop.npy", np.load(a / "q_hop.npy").astype(np.int32))
    np.save(b / "t_gold.npy", np.load(a / "t_gold.npy")[:-1])
    diffs = {d["array"]: d["differs_in"] for d in L12.array_differences(a, b)}
    assert diffs == {"e_node": ["bytes"], "q_hop": ["dtype", "bytes"], "t_gold": ["shape", "bytes"]}


def _toy_cache():
    rng = np.random.default_rng(5)
    pools = [np.sort(rng.choice(5000, size=n, replace=False)) for n in (9, 6, 11)]
    seeds = [np.array([0, 4]), np.array([2]), np.array([1, 3, 7])]
    golds = [np.array([1, 5]), np.array([0]), np.array([2, 8, 10])]
    totals = [2, 3, 4]
    edges = [[5, 7, 2], [1, 0, 3], [9, 9, 9]]
    qemb = rng.normal(size=(3, 16)).astype(np.float32)

    def ptr(xs):
        return np.r_[0, np.cumsum([x.size for x in xs])].astype(np.int64)

    cache = {"query_ids": ["metaqa:1hop:train:5", "metaqa:2hop:train:9", "metaqa:3hop:train:2"], "qrow": np.array([40, 7, 13]),
             "pool": np.concatenate(pools).astype(np.int32), "pool_ptr": ptr(pools), "seeds": np.concatenate(seeds).astype(np.int16),
             "seeds_ptr": ptr(seeds), "gold": np.concatenate(golds).astype(np.int16), "gold_ptr": ptr(golds),
             "gold_total": np.asarray(totals, dtype=np.int32), "edge_counts": np.asarray(edges, dtype=np.int32),
             "qemb": qemb.astype(np.float16)}
    return cache, pools, seeds, golds, totals, edges, qemb


def test_the_cache_check_refuses_a_changed_pool_seed_gold_total_edge_count_row_embedding_or_order(stops):
    cache, pools, seeds, golds, totals, edges, qemb = _toy_cache()
    fams = L8.FAMILIES

    def n_edges(i):
        return {f: edges[i][k] for k, f in enumerate(fams)}

    for i in range(3):
        assert L12.cache_query_problems(cache, i, pools[i], seeds[i], golds[i], totals[i], n_edges(i), qemb[i]) == []
    assert L12.cache_carve_problems(cache, cache["query_ids"], cache["qrow"]) == []
    i = 2
    other = pools[i].copy()
    other[4] += 1
    changed = {"pool": (other, seeds[i], golds[i], totals[i], n_edges(i), qemb[i]),
               "seeds": (pools[i], seeds[i][::-1], golds[i], totals[i], n_edges(i), qemb[i]),
               "gold": (pools[i], seeds[i], golds[i][:-1], totals[i], n_edges(i), qemb[i]),
               "gold_total": (pools[i], seeds[i], golds[i], totals[i] + 1, n_edges(i), qemb[i]),
               "edge_counts": (pools[i], seeds[i], golds[i], totals[i], {**n_edges(i), fams[1]: 10}, qemb[i]),
               "qemb": (pools[i], seeds[i], golds[i], totals[i], n_edges(i), qemb[i] + np.float32(0.5) * (np.arange(16) == 3))}
    for what, args in changed.items():
        assert L12.cache_query_problems(cache, i, *args) == [what]
    assert L12.cache_carve_problems(cache, cache["query_ids"][::-1], cache["qrow"]) == ["query_ids"]
    assert L12.cache_carve_problems(cache, cache["query_ids"], cache["qrow"] + np.array([0, 1, 0])) == ["qrow"]
    assert "pool_ptr rows" in L12.cache_carve_problems(cache, cache["query_ids"][:2], cache["qrow"][:2])
    compiled = SimpleNamespace(pool=pools[1], seeds_local=seeds[1], n_edges={**n_edges(1), fams[0]: 99})
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.check_cache_query(cache, 1, cache["query_ids"][1], compiled, golds[1], totals[1], qemb[1])
    assert "cache_equality" in _stopped(stops) and "edge_counts" in _stopped(stops)


def test_the_cache_files_are_opened_only_after_their_pins_and_never_for_an_x_carve(tmp_path, stops):
    cache = _toy_cache()[0]
    d = tmp_path / "cache" / "select"
    d.mkdir(parents=True)
    for key in L12.CACHE_ARRAYS:
        np.save(d / f"{key}.npy", cache[key])
    (d / "query_ids.json").write_text(json.dumps(cache["query_ids"]), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"n_queries": 3}), encoding="utf-8")
    pins = {f: L0.sha256_file(d / f) for f in L12.CACHE_FILES}
    shutil.copytree(tmp_path / "cache", tmp_path / "other")
    p = np.load(tmp_path / "other" / "select" / "pool.npy")
    p[0] += 1
    np.save(tmp_path / "other" / "select" / "pool.npy", p)
    decl = {"inputs": {"carve_cache": {"dir": str(tmp_path / "cache"), "select": pins}}}
    assert L12.open_cache(decl, "x1") is None
    got = L12.open_cache(decl, "select")
    assert got["query_ids"] == cache["query_ids"] and all(np.array_equal(got[k], cache[k]) for k in L12.CACHE_ARRAYS)
    del got
    gc.collect()
    decl["inputs"]["carve_cache"]["dir"] = str(tmp_path / "other")
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.open_cache(decl, "select")
    assert "pool.npy is not its pinned sha256" in _stopped(stops)


def test_the_loopcheck_refuses_a_smoke_under_the_outputs_or_of_another_limit(tmp_path, monkeypatch, stops, one_thread):
    decl = L12.load_declaration()
    with pytest.raises(SystemExit, match="never lies under"):
        L12.stage_loopcheck(decl, _quiet, L12.OUT / "smoke" / "metaqa")
    monkeypatch.setattr(L12, "LOOPCHECK", tmp_path / "loopcheck.json")
    monkeypatch.setattr(L12, "verify_inputs", lambda decl: None)
    d = tmp_path / "smoke" / "metaqa"
    T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=4, seed=12)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for extra in ({"limit": 12, "declaration_lf_sha256": L0.lf_sha256(L12.CONFIG)}, {"limit": 24, "declaration_lf_sha256": "0" * 64}):
        (d / "meta.json").write_text(json.dumps({**meta, **extra}), encoding="utf-8")
        with pytest.raises(SystemExit, match="not this file's dev smoke"):
            L12.stage_loopcheck(decl, _quiet, d)
    assert not (tmp_path / "loopcheck.json").exists()


# ── the check and the carve check on synthetic combined sidecars ─────────────


def _rewrite_ids(d: Path, new_ids: list[str], **meta_extra) -> None:
    """A toy sidecar's ids replaced: q_fold and q_inner recomputed by level 0's rules, and meta.json's sha256 values updated."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "qids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    np.save(d / "q_fold.npy", np.asarray([L0.fold_of(q) for q in new_ids], dtype=np.int64))
    np.save(d / "q_inner.npy", np.asarray([int(L0.is_inner(q)) for q in new_ids], dtype=np.int64))
    for key in ("q_fold", "q_inner"):
        meta["arrays_sha256"][f"{key}.npy"] = L0.sha256_file(d / f"{key}.npy")
    meta["qids_sha256"] = L0.sha256_file(d / "qids.json")
    meta.update(meta_extra)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _check_env(tmp_path, monkeypatch, n_q, seed):
    d = tmp_path / "l12" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    monkeypatch.setattr(L12, "DATA", d)
    monkeypatch.setattr(L12, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    return d, qids


def test_the_check_runs_level_9s_check_family_on_both_views(tmp_path, monkeypatch, stops, one_thread):
    d, qids = _check_env(tmp_path, monkeypatch, n_q=30, seed=4)
    check = L12.stage_check(L12.load_declaration(), _quiet)
    fam = check["families"]
    assert set(fam) == {"std", "nb"} and all(fam[f]["direction_check"]["passes"] for f in L12.FAMILIES)
    assert fam["std"]["chain_fit"]["recall"]["all"] == 1.0 and fam["nb"]["sizes"]["entries_total"] < fam["std"]["sizes"]["entries_total"]
    assert 0.0 <= check["gold_unreached_nb"]["all"] <= 1.0 and 0.0 <= check["nb_trim"]["gold_share_removed"]["all"] <= 1.0
    assert check["hops"] == L12.hop_counts(qids) and check["rows_with_gold_in_pool"]["all"] == len(qids)
    on_disk = json.loads((d / "check.json").read_text(encoding="utf-8"))
    assert on_disk["meta_sha256"] == L0.sha256_file(d / "meta.json") and "module_sha256" in on_disk
    assert not (stops / "hard_stops.json").exists()


def test_the_check_refuses_a_smoke_an_earlier_levels_id_a_held_row_a_train_id_and_a_failed_direction_check(tmp_path, monkeypatch, stops,
                                                                                                            one_thread):
    d, qids = _check_env(tmp_path, monkeypatch, n_q=12, seed=8)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "meta.json").write_text(json.dumps({**meta, "limit": 12}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke"):
        L12.stage_check(L12.load_declaration(), _quiet)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for level in ("level11", "level10", "level9", "level8", "level0"):
        decl = L12.load_declaration()
        fake = tmp_path / f"{level}_qids.json"
        fake.write_text(json.dumps([qids[3]]), encoding="utf-8")
        decl["inputs"][level]["excluded_rows"]["qids"]["path"] = str(fake)
        with pytest.raises(SystemExit, match="HARD STOP"):
            L12.stage_check(decl, _quiet)
        assert "level 0, level 8, level 9, level 10 or level 11 query id" in _stopped(stops)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.r_[np.ones(5, dtype=bool), False, np.ones(len(qids) - 6, dtype=bool)],
                                                               None, None))
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_check(L12.load_declaration(), _quiet)
    assert "a held row is in the sidecar" in _stopped(stops)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    _rewrite_ids(d, [q.replace(":toy:", ":train:") if i == 2 else q for i, q in enumerate(qids)])
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_check(L12.load_declaration(), _quiet)
    assert "a train-split query id is in the dev sidecar" in _stopped(stops)
    _rewrite_ids(d, qids)
    real = L8.chain_tokens
    monkeypatch.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))   # the map, turned around
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_check(L12.load_declaration(), _quiet)
    assert "direction_check (std)" in _stopped(stops)


def _carve_sidecar(root: Path, carve: str, monkeypatch, n_q: int, seed: int, tag: str) -> list[str]:
    d = root / "carves" / carve
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    ids = [q.replace(":toy:", f":train:{tag}") for q in qids]
    _rewrite_ids(d, ids, carve=carve)
    return ids


def _carve_env(tmp_path, monkeypatch):
    root = tmp_path / "l12"
    dev = T9._sidecar(root / "metaqa", monkeypatch, L9.walk_entries_both, n_q=10, seed=3)
    fit = _carve_sidecar(root, "fit", monkeypatch, 12, 5, "f")
    sel = _carve_sidecar(root, "select", monkeypatch, 10, 6, "s")
    monkeypatch.setattr(L12, "DATA", root / "metaqa")
    monkeypatch.setattr(L12, "CARVES_DIR", root / "carves")
    monkeypatch.setattr(L12, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    decl = L12.load_declaration()
    for c, ids in (("fit", fit), ("select", sel)):
        decl["carves"]["pins"][c] = {"queries": len(ids), "zero_gold_excluded": 0, "ids_sha256": m3b_pools.ids_digest(ids),
                                     "hops": L12.hop_counts(ids)}
    return root, decl, dev, fit, sel


def test_the_carve_check_runs_on_a_carve_sidecar_and_compares_it_with_the_dev_rows_and_the_other_carves(tmp_path, monkeypatch, stops,
                                                                                                        one_thread):
    root, decl, _dev, fit, _sel = _carve_env(tmp_path, monkeypatch)
    check = L12.stage_carve_check(decl, "fit", _quiet)
    assert check["carve"] == "fit" and check["queries"] == len(fit) and check["carves_compared"] == {"select": 10}
    assert check["ids_sha256"] == m3b_pools.ids_digest(fit) and check["hops"] == L12.hop_counts(fit) and check["dev_ids_shared"] == 0
    assert all(check["families"][f]["direction_check"]["passes"] for f in L12.FAMILIES)
    data = L8.Data(root / "carves" / "fit")
    for fam, k0 in (("twin", 0), ("gnn", 3)):
        for m in L8.RETRIEVAL:
            col = data.q_metrics[:, k0:k0 + 3, list(L12.METRIC_NAMES).index(m)]
            assert check["carve_metrics"][fam][m]["all"] == pytest.approx(float(col.mean()), abs=1e-12)
    on_disk = json.loads((root / "carves" / "fit" / "check.json").read_text(encoding="utf-8"))
    assert on_disk["meta_sha256"] == L0.sha256_file(root / "carves" / "fit" / "meta.json")
    assert not (stops / "hard_stops.json").exists()


def test_the_carve_check_refuses_a_dev_id_shared_ids_a_non_train_id_another_pin_and_a_missing_dev_sidecar(tmp_path, monkeypatch, stops,
                                                                                                          one_thread):
    root, decl, dev, fit, _sel = _carve_env(tmp_path, monkeypatch)
    _rewrite_ids(root / "metaqa", [fit[4]] + dev[1:])   # a carve id among the dev rows
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_carve_check(decl, "fit", _quiet)
    assert "a carve fit id is in the dev sidecar" in _stopped(stops)
    _rewrite_ids(root / "metaqa", dev)
    _carve_sidecar(root, "x1", monkeypatch, 12, 5, "f")   # the fit carve's ids again
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_carve_check(decl, "fit", _quiet)
    assert "carve fit and carve x1 share ids" in _stopped(stops)
    shutil.rmtree(root / "carves" / "x1")
    bad = copy.deepcopy(decl)
    bad["carves"]["pins"]["fit"]["ids_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_carve_check(bad, "fit", _quiet)
    assert "carve fit's ids are not its pin" in _stopped(stops)
    _rewrite_ids(root / "carves" / "fit", [q.replace(":train:", ":trn:") if i == 1 else q for i, q in enumerate(fit)], carve="fit")
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_carve_check(decl, "fit", _quiet)
    assert "a carve fit id is not a metaqa train-split id" in _stopped(stops)
    _rewrite_ids(root / "carves" / "fit", fit, carve="fit")
    _rewrite_ids(root / "carves" / "select", json.loads((root / "carves" / "select" / "qids.json").read_text(encoding="utf-8")),
                 carve="fit")
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_carve_check(decl, "select", _quiet)
    assert "is not carve select's" in _stopped(stops)
    shutil.rmtree(root / "metaqa")
    with pytest.raises(SystemExit, match="not assembled"):
        L12.stage_carve_check(decl, "fit", _quiet)


def test_the_carve_check_refuses_a_smoke_and_a_failed_direction_check(tmp_path, monkeypatch, stops, one_thread):
    root, decl, _dev, _fit, _sel = _carve_env(tmp_path, monkeypatch)
    meta = json.loads((root / "carves" / "fit" / "meta.json").read_text(encoding="utf-8"))
    (root / "carves" / "fit" / "meta.json").write_text(json.dumps({**meta, "limit": 12}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke"):
        L12.stage_carve_check(decl, "fit", _quiet)
    (root / "carves" / "fit" / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    real = L8.chain_tokens
    monkeypatch.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.stage_carve_check(decl, "fit", _quiet)
    assert "direction_check (std)" in _stopped(stops)


# ── the stages' machine ──────────────────────────────────────────────────────


def test_the_stages_refuse_the_wrong_machine_and_every_wrong_combination(stops):
    for argv in (["--stage", "check"], ["--stage", "score"], ["--stage", "fit", "--host"], ["--stage", "carve", "--host"],
                 ["--stage", "carve_check", "--host"], ["--stage", "check", "--host", "--carve", "fit"],
                 ["--stage", "carve", "--host", "--carve", "x8"], ["--stage", "check", "--host", "--shard", "0/3"],
                 ["--stage", "carve_check", "--host", "--carve", "fit", "--shard", "0/2"], ["--stage", "loopcheck", "--host"],
                 ["--stage", "loopcheck", "--host", "--out", "x", "--limit", "24"],
                 ["--stage", "loopcheck", "--host", "--out", str(L12.OUT / "x")],
                 ["--stage", "score", "--host", "--limit", "4"], ["--stage", "check", "--host", "--limit", "4", "--out", "x"],
                 ["--stage", "carve_check", "--host", "--carve", "fit", "--limit", "4", "--out", "x"],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L12.OUT / "x")],
                 ["--stage", "carve", "--host", "--carve", "x1", "--limit", "4", "--out", str(L12.OUT)],
                 ["--stage", "carve", "--host", "--carve", "x1", "--limit", "4", "--out", "x", "--shard", "0/2"]):
        with pytest.raises(SystemExit):
            L12.main(argv)


# ── no held, earlier level's or train-split query in the dev sidecar ─────────


def test_no_held_earlier_levels_or_train_split_query_id_appears_in_the_dev_sidecar():
    side = L12.DATA / "qids.json"
    if not side.exists():
        pytest.skip("this file's sidecar is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"]["metaqa"]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    got = json.loads(side.read_text(encoding="utf-8"))
    inp = L12.load_declaration()["inputs"]
    for level in L12.LEVELS:
        earlier = set(json.loads((ROOT / inp[level]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
        assert not (set(got) & earlier), level
    assert got and set(got) <= gate and len(got) == len(set(got)) == sum(L12.PER_HOP.values())
    assert not any(L12.is_train_id(q) for q in got) and L12.hop_counts(got) == L12.PER_HOP
    meta = L12.DATA / "meta.json"
    if meta.exists():
        assert json.loads(meta.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)


def test_every_carve_sidecar_holds_its_pinned_train_split_ids_and_no_dev_id():
    found = [c for c in L12.CARVES if (L12.CARVES_DIR / c / "qids.json").exists()]
    if not found:
        pytest.skip("no carve sidecar is on this machine")
    pins = L12.load_declaration()["carves"]["pins"]
    side = L12.DATA / "qids.json"
    dev = set(json.loads(side.read_text(encoding="utf-8"))) if side.exists() else set()
    seen = {}
    for c in found:
        got = json.loads((L12.CARVES_DIR / c / "qids.json").read_text(encoding="utf-8"))
        assert m3b_pools.ids_digest(got) == pins[c]["ids_sha256"] and all(L12.is_train_id(q) for q in got) and not (set(got) & dev)
        assert all(not (set(got) & ids) for ids in seen.values())
        seen[c] = set(got)
