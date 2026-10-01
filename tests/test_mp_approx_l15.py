"""MP-Approx level 15, population module (configs/mp_approx_l15.yaml#tests): the declaration, its pins and its constants;
r2's rule on toy strides and level 12's carve population check under the rebound names; the rebinding's restore; the
check on synthetic sidecars and its refusals; the stages' machine; and only r2's pinned train-split ids, none of an
earlier level's (level 14's r included) or of level 12's carves, in r2's sidecar."""

from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

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
import mp_approx_l13 as L13  # noqa: E402
import mp_approx_l14 as L14  # noqa: E402
import mp_approx_l15 as L15  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)
import test_mp_approx_l12 as T12  # noqa: E402  (level 12's toy dataset and carves)
from mp_retrieval import m3b_pools  # noqa: E402

_quiet = T8._quiet

DESIGN_LOOKS = ("outputs/mp_approx_l14_diag/diag_route.py", "outputs/mp_approx_l14_diag/diag_route.json",
                "outputs/mp_approx_l14_diag/diag_gate.py", "outputs/mp_approx_l14_diag/diag_gate.json",
                "outputs/mp_approx_l14_diag/diag_chain.py", "outputs/mp_approx_l14_diag/diag_chain.json",
                "outputs/mp_approx_l14_diag/diag_probe.py", "outputs/mp_approx_l14_diag/diag_probe.json",
                "outputs/mp_approx_l14_diag/diag_fz.py", "outputs/mp_approx_l14_diag/diag_fz.json",
                "outputs/mp_approx_l14_diag/diag_route2.py", "outputs/mp_approx_l14_diag/diag_route2.json",
                "outputs/mp_approx_l15_design/l15_read_carve_pins.py", "outputs/mp_approx_l15_design/l15_read_carve_pins.json")


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, L12, L13, L14, L15):
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
    decl = L15.load_declaration()
    assert decl["phase"] == "MP_APPROX_L15" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L8.load_declaration()["registered_question"]
    assert list(decl["inputs"]["relations"]["order"]) == list(L8.REL_ORDER)
    cv = decl["inputs"]["level12_carves_x4_x7"]
    if not all((ROOT / cv["dir"] / c / "meta.json").exists() for c in (*L15.L12_CARVES, *L15.L12_OTHER_CARVES)):
        pytest.skip("level 12's carve sidecars are not on this machine")
    L15.verify_inputs(decl)   # level 14's check on this copy, then level 14's own files and rows
    assert not (stops / "hard_stops.json").exists()


def test_the_design_looks_named_by_the_declaration_are_pinned_by_their_raw_sha256():
    text = L15.load_declaration()["design_informed_by"]
    pinned = set(re.findall(r"\(([0-9a-f]{64})\)", text))
    assert len(pinned) == len(DESIGN_LOOKS) == 14
    for rel in DESIGN_LOOKS:
        assert Path(rel).name in text
    present = [rel for rel in DESIGN_LOOKS if (ROOT / rel).exists()]
    if len(present) < len(DESIGN_LOOKS):
        pytest.skip("the design looks are not all on this machine")
    assert {L0.sha256_file(ROOT / rel) for rel in DESIGN_LOOKS} == pinned


@pytest.mark.parametrize("key", ["script_lf", "tests_lf", "declaration_lf", "fit_script_lf", "fit_tests_lf"])
def test_a_changed_level14_pin_stops(stops, monkeypatch, key):
    monkeypatch.setattr(L15, "_L14_VERIFY", lambda decl: None)   # the copied pins are the test above's
    decl = L15.load_declaration()
    decl["inputs"]["level14"][key]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.verify_inputs(decl)
    assert decl["inputs"]["level14"][key]["path"] in _stopped(stops)


def test_a_changed_copied_pin_level_14s_rows_or_level_14s_read_stops(stops, monkeypatch):
    decl = L15.load_declaration()
    decl["inputs"]["level13"]["script_lf"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.verify_inputs(decl)
    assert "scripts/mp_approx_l13.py" in _stopped(stops)
    with monkeypatch.context() as m:
        m.setattr(L14, "_L13_VERIFY", lambda decl: None)   # level 14's own part of the check, on this file's copy
        decl = L15.load_declaration()
        decl["inputs"]["earlier_reads"]["level14"]["sha256"] = "0" * 64
        with pytest.raises(SystemExit, match="HARD STOP"):
            L15.verify_inputs(decl)
        assert "outputs/mp_approx_l14/metaqa/read.json" in _stopped(stops)
    monkeypatch.setattr(L15, "_L14_VERIFY", lambda decl: None)
    decl = L15.load_declaration()
    decl["inputs"]["level14"]["excluded_rows"]["qids"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.verify_inputs(decl)
    assert "outputs/mp_approx_l14/metaqa/qids.json" in _stopped(stops)


def test_the_declared_constants_are_the_files():
    decl = L15.load_declaration()
    assert (L15.READ_CARVE, L15.R2_OFFSETS, L15.SHARDS, L15.NAME, L15.L14_CARVE) == ("r2", (10, 11), 12, "metaqa", "r")
    assert tuple(decl["population"]["offsets"]) == L15.R2_OFFSETS and decl["population"]["pin"] == "carves.pins.r2"
    pin = decl["carves"]["pins"]["r2"]
    assert {int(h): int(v) for h, v in decl["population"]["hops"].items()} == {int(h): int(v) for h, v in pin["hops"].items()}
    assert (pin["queries"], pin["zero_gold_excluded"], sum(int(v) for v in pin["hops"].values())) == (11920, 0, 11920)
    assert pin["ids_sha256"] != decl["carves"]["pins"]["r"]["ids_sha256"]
    assert L15.FAMILIES == ("std", "nb") and L15.SCRIPT_REL == "scripts/mp_approx_l15.py"
    assert L15.LEVELS == ("level0", "level8", "level9", "level10", "level11", "level12", "level13", "level14")
    assert L15.CARVES == ("fit", "select", "x1", "x2", "x3", "x4", "x5", "x6", "x7", "r", "r2") and L15._L12_CARVES == L12.CARVES
    assert L15.L12_CARVES == ("fit", "select", "x1", "x2", "x3") and L15.L12_OTHER_CARVES == ("x4", "x5", "x6", "x7")
    # the level 14 pins name level 14's committed files and rows, and its read is the third earlier read
    lv = decl["inputs"]["level14"]
    assert (lv["declaration_lf"]["path"], lv["script_lf"]["path"], lv["tests_lf"]["path"], lv["fit_script_lf"]["path"],
            lv["fit_tests_lf"]["path"]) == ("configs/mp_approx_l14.yaml", "scripts/mp_approx_l14.py", "tests/test_mp_approx_l14.py",
                                            "scripts/mp_approx_l14_fit.py", "tests/test_mp_approx_l14_fit.py")
    assert lv["excluded_rows"]["qids"]["path"] == "outputs/mp_approx_l14/metaqa/qids.json"
    assert {k: v["path"] for k, v in decl["inputs"]["earlier_reads"].items()} == {
        "level12": "outputs/mp_approx_l12/metaqa/read.json", "level13": "outputs/mp_approx_l13/metaqa/read.json",
        "level14": "outputs/mp_approx_l14/metaqa/read.json"}
    # every pin level 14 declares is copied unchanged (its earlier_reads extended by level 14's read), and its carve pins and fits too
    l14 = L14.load_declaration()
    assert all(decl["inputs"][k] == l14["inputs"][k] for k in l14["inputs"] if k != "earlier_reads")
    assert all(decl["inputs"]["earlier_reads"][k] == v for k, v in l14["inputs"]["earlier_reads"].items())
    pins = dict(decl["carves"]["pins"])
    pins.pop("r2")
    assert pins == l14["carves"]["pins"] and decl["carves"]["fits"] == l14["carves"]["fits"]
    # placement.identical_code: the score, assemble and check jobs never import the fit module
    src = (ROOT / L15.SCRIPT_REL).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+mp_approx_l15_fit", src, re.M)


# ── r2's rule and the rebound carve population check ─────────────────────────


def _pin_of(ids: list[str], positions: dict) -> dict:
    kept = [q for q in ids if positions[q]]
    return {"queries": len(kept), "zero_gold_excluded": len(ids) - len(kept), "ids_sha256": m3b_pools.ids_digest(kept),
            "hops": L15.hop_counts(kept)}


def _toy_with_r2(seed=0, fit_cap=60):
    S, decl, positions, rule = T12._toy_carves(seed, fit_cap=fit_cap)
    decl["carves"]["pins"]["r"] = _pin_of(L14.r_ids_of(rule), positions)
    decl["carves"]["pins"]["r2"] = _pin_of(L15.r2_ids_of(rule), positions)
    return S, decl, positions, rule


def test_r2_is_the_union_of_offsets_10_and_11_in_position_order_and_disjoint_from_every_other_carve():
    for fit_cap in (20, 60, 72):
        _S, _decl, _positions, rule = T12._toy_carves(0, fit_cap=fit_cap)
        s, rem = rule["s_fit"], rule["remaining"]
        assert s > 11
        r2 = L15.r2_ids_of(rule)
        assert r2 == [rem[i] for i in range(len(rem)) if i % s in (10, 11)] == L15.carve_ids_of("r2", rule)
        assert set(r2) == set(rem[10::s]) | set(rem[11::s]) and len(r2) == len(rem[10::s]) + len(rem[11::s]) == len(set(r2))
        pos = {q: i for i, q in enumerate(rem)}
        assert all(pos[a] < pos[b] for a, b in zip(r2, r2[1:]))
        assert L15.carve_ids_of("r", rule) == L14.r_ids_of(rule) and not set(r2) & set(L14.r_ids_of(rule))
        for c in L12.CARVES:
            assert L15.carve_ids_of(c, rule) == L12.carve_ids_of(c, rule)
            assert not set(r2) & set(L12.carve_ids_of(c, rule))
    for fit_cap in (79, 89):   # strides of 11 and 9 have no offset 11
        _S, _decl, _positions, rule = T12._toy_carves(0, fit_cap=fit_cap)
        assert rule["s_fit"] <= 11
        with pytest.raises(ValueError):
            L15.r2_ids_of(rule)
    # on metaqa's train-split size the rule gives r2's pinned count before the zero-gold rows leave
    pins = L15.load_declaration()["carves"]["pins"]
    rule = L12.carve_rule([f"{i:07d}" for i in range(int(pins["N"]))], 1500, 5, 6000)
    assert (rule["s_sel"], rule["s_fit"], len(rule["remaining"])) == (pins["s_sel"], pins["s_fit"], pins["remaining"])
    assert len(L15.r2_ids_of(rule)) == pins["r2"]["queries"] + pins["r2"]["zero_gold_excluded"] == 11920


def _last_stop(stops) -> dict:
    return json.loads((stops / "hard_stops.json").read_text(encoding="utf-8"))[-1]


def test_r2s_population_is_level_12s_carve_population_check_under_the_rebound_names(stops, monkeypatch):
    S, decl, positions, rule = _toy_with_r2(seed=2)
    with L15.level12_rebound():
        pop = L12.carve_population_checked(decl, S, positions, "r2")
        assert pop.kind == "r2" and pop.digest == decl["carves"]["pins"]["r2"]["ids_sha256"] and all(L15.is_train_id(q) for q in pop.ids)
        assert pop.ids == [q for q in L15.r2_ids_of(rule) if positions[q]] and pop.zero_gold_excluded > 0
        for c in (*L15._L12_CARVES, "r"):   # level 12's nine and level 14's r still pass beside r2
            assert L12.carve_population_checked(decl, S, positions, c).digest == decl["carves"]["pins"][c]["ids_sha256"]
    assert not (stops / "hard_stops.json").exists()
    for change in (lambda p: p["r2"].update(ids_sha256="0" * 64), lambda p: p["r2"]["hops"].update({2: p["r2"]["hops"][2] + 1}),
                   lambda p: p["r2"].update(zero_gold_excluded=0)):
        bad = copy.deepcopy(decl)
        change(bad["carves"]["pins"])
        with L15.level12_rebound(), pytest.raises(SystemExit, match="HARD STOP"):
            L12.carve_population_checked(bad, S, positions, "r2")
        assert _last_stop(stops)["message"] == "carves: carve r2 is not its pin"
    for offset, other in ((3, "x3"), (8, "r")):   # r2 on x3's offset, or on r's
        with monkeypatch.context() as m:
            m.setattr(L15, "r2_ids_of", lambda rule, j=offset: rule["remaining"][j::rule["s_fit"]])
            with L15.level12_rebound(), pytest.raises(SystemExit, match="HARD STOP"):
                L12.carve_population_checked(decl, S, positions, "r2")
        assert _last_stop(stops)["message"] == f"carves: carve {other} and carve r2 share ids"
    assert L15.r2_ids_of(rule) == [q for i, q in enumerate(rule["remaining"]) if i % rule["s_fit"] in (10, 11)]
    # outside the rebound, level 12's own function reads r2 as the stride's offset 2, so the pin check stops it
    assert L12.carve_ids_of("r2", rule) == L12.carve_ids_of("x2", rule)
    with pytest.raises(SystemExit, match="HARD STOP"):
        L12.carve_population_checked(decl, S, positions, "r2")
    assert _last_stop(stops)["message"] == "carves: carve r2 is not its pin"


def test_a_non_train_id_in_r2_stops_under_the_rebound_names(stops):
    S, _decl, positions, rule = T12._toy_carves(seed=3, bad_hop1=True)   # hop-1 ids are metaqa:1hop:trn:<n>
    ids = L15.r2_ids_of(rule)
    assert any(not L15.is_train_id(q) for q in ids if positions[q])
    decl = {"carves": {"pins": {**_decl["carves"]["pins"], "r": _pin_of(L14.r_ids_of(rule), positions), "r2": _pin_of(ids, positions)}}}
    with L15.level12_rebound(), pytest.raises(SystemExit, match="HARD STOP"):
        L12.carve_population_checked(decl, S, positions, "r2")
    assert "a carve r2 id is not a metaqa train-split id" in _stopped(stops)


def test_the_rebound_restores_level_12s_names_after_the_pass_and_on_an_error():
    names = ("CARVES", "carve_ids_of", "verify_inputs", "CONFIG")
    before = {k: getattr(L12, k) for k in names}
    l14_before = {k: getattr(L14, k) for k in ("CARVES", "carve_ids_of", "verify_inputs", "CONFIG", "r_ids_of")}
    with L15.level12_rebound():
        assert L12.CARVES == (*before["CARVES"], "r", "r2") and L12.carve_ids_of is L15.carve_ids_of
        assert L12.verify_inputs is L15.verify_inputs and L12.CONFIG == L15.CONFIG != before["CONFIG"]
        assert L15._L14_CARVE_IDS is l14_before["carve_ids_of"] and L14._L12_CARVE_IDS is before["carve_ids_of"]
        assert all(getattr(L14, k) is v for k, v in l14_before.items())   # level 14's module is not rebound
    assert all(getattr(L12, k) is v for k, v in before.items())
    with pytest.raises(RuntimeError):
        with L15.level12_rebound():
            raise RuntimeError("inside the pass")
    assert all(getattr(L12, k) is v for k, v in before.items())
    with pytest.raises(SystemExit, match="not one of"):   # outside the rebound, level 12's carve pass refuses r2
        L12.stage_carve({}, "r2", _quiet)


def test_the_scoring_pass_runs_level_12s_carve_pass_on_r2_with_this_files_names(tmp_path, monkeypatch, stops):
    seen = {}

    def fake_stage_carve(decl, carve, log, shard, limit, out_dir):
        seen.update(carves=L12.CARVES, ids=L12.carve_ids_of, verify=L12.verify_inputs, config=L12.CONFIG,
                    args=(carve, shard, limit, out_dir))

    monkeypatch.setattr(L12, "stage_carve", fake_stage_carve)
    L15.stage_score(L15.load_declaration(), _quiet, (3, 12), None, tmp_path / "o")
    assert seen["carves"] == L15.CARVES and seen["ids"] is L15.carve_ids_of and seen["verify"] is L15.verify_inputs
    assert seen["config"] == L15.CONFIG and seen["args"] == ("r2", (3, 12), None, tmp_path / "o")
    L15.stage_score(L15.load_declaration(), _quiet)
    assert seen["args"] == ("r2", None, None, L15.DATA)
    L15.stage_score(L15.load_declaration(), _quiet, None, 4, tmp_path / "smoke")
    assert seen["args"] == ("r2", None, 4, tmp_path / "smoke")
    assert L12.CARVES == L15._L12_CARVES and L12.carve_ids_of is L14._L12_CARVE_IDS and L12.CONFIG != L15.CONFIG


# ── the check ────────────────────────────────────────────────────────────────


def _fake_carve(d: Path, ids: list[str]) -> None:
    """An assembled carve as level 12's assembled_ids reads it: its qids.json and the sha256 its meta.json records."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "qids.json").write_text(json.dumps(ids), encoding="utf-8")
    (d / "meta.json").write_text(json.dumps({"qids_sha256": L0.sha256_file(d / "qids.json")}), encoding="utf-8")


def _check_env(tmp_path, monkeypatch, n_q=14, seed=4):
    d = tmp_path / "l15" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    ids = [q.replace(":toy:", ":train:s") for q in qids]
    T12._rewrite_ids(d, ids, carve="r2", carve_ids_sha256=m3b_pools.ids_digest(ids))
    carves = tmp_path / "l12" / "carves"
    for j, c in enumerate(L15._L12_CARVES):
        _fake_carve(carves / c, [f"metaqa:{1 + i % 3}hop:train:c{j}_{i}" for i in range(5)])
    monkeypatch.setattr(L15, "DATA", d)
    monkeypatch.setattr(L12, "CARVES_DIR", carves)
    monkeypatch.setattr(L15, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    decl = L15.load_declaration()
    decl["carves"]["pins"]["r2"] = {"queries": len(ids), "zero_gold_excluded": 0, "ids_sha256": m3b_pools.ids_digest(ids),
                                    "hops": L15.hop_counts(ids)}
    return d, carves, decl, ids


def _levels_on_file(decl) -> bool:
    return all((ROOT / decl["inputs"][lv]["excluded_rows"]["qids"]["path"]).exists() for lv in L15.LEVELS)


def test_the_check_runs_level_9s_check_family_on_both_views_and_compares_r2_with_level_12s_carves_and_the_earlier_rows(
        tmp_path, monkeypatch, stops, one_thread):
    d, _carves, decl, ids = _check_env(tmp_path, monkeypatch)
    if not _levels_on_file(decl):
        pytest.skip("the earlier levels' qids.json files are not on this machine")
    check = L15.stage_check(decl, _quiet)
    assert check["stage"] == "check" and check["carve"] == "r2" and check["queries"] == len(ids) == check["train_split_ids"]
    assert check["carves_compared"] == {c: 5 for c in L12.CARVES} and check["loopcheck_equal"] is True
    assert tuple(check["levels_compared"]) == L15.LEVELS and check["levels_compared"]["level14"] == 11920
    assert check["ids_sha256"] == m3b_pools.ids_digest(ids) and check["hops"] == L15.hop_counts(ids)
    assert all(check["families"][f]["direction_check"]["passes"] for f in L15.FAMILIES)
    assert check["rows_with_gold_in_pool"]["all"] == len(ids)
    data = L8.Data(d)
    for fam, k0 in (("twin", 0), ("gnn", 3)):
        for m in L8.RETRIEVAL:
            col = data.q_metrics[:, k0:k0 + 3, list(L12.METRIC_NAMES).index(m)]
            assert check["carve_metrics"][fam][m]["all"] == pytest.approx(float(col.mean()), abs=1e-12)
    on_disk = json.loads((d / "check.json").read_text(encoding="utf-8"))
    assert on_disk["meta_sha256"] == L0.sha256_file(d / "meta.json") and "module_sha256" in on_disk and on_disk["carve"] == "r2"
    assert not (stops / "hard_stops.json").exists()


def test_the_check_refuses_a_smoke_another_sidecar_a_non_train_id_another_pin_and_a_q_row_out_of_order(tmp_path, monkeypatch, stops,
                                                                                                       one_thread):
    d, _carves, decl, ids = _check_env(tmp_path, monkeypatch, n_q=12, seed=8)
    if not _levels_on_file(decl):
        pytest.skip("the earlier levels' qids.json files are not on this machine")
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "meta.json").write_text(json.dumps({**meta, "limit": 12}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke"):
        L15.stage_check(decl, _quiet)
    for extra in ({"carve": "r"}, {"carve_ids_sha256": "0" * 64}, {"carve": None}):
        (d / "meta.json").write_text(json.dumps({**meta, **extra}), encoding="utf-8")
        with pytest.raises(SystemExit, match="HARD STOP"):
            L15.stage_check(decl, _quiet)
        assert "is not r2's" in _stopped(stops)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    bad = copy.deepcopy(decl)
    bad["carves"]["pins"]["r2"]["hops"] = {1: 0, 2: 0, 3: len(ids)}
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.stage_check(bad, _quiet)
    assert "r2's ids are not its pin" in _stopped(stops)
    digest = m3b_pools.ids_digest(ids)
    T12._rewrite_ids(d, [q.replace(":train:", ":trn:") if i == 1 else q for i, q in enumerate(ids)], carve="r2", carve_ids_sha256=digest)
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.stage_check(decl, _quiet)
    assert "is not a metaqa train-split id" in _stopped(stops)
    T12._rewrite_ids(d, ids, carve="r2", carve_ids_sha256=digest)
    q_row = np.load(d / "q_row.npy")
    np.save(d / "q_row.npy", q_row[::-1].copy())
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    meta["arrays_sha256"]["q_row.npy"] = L0.sha256_file(d / "q_row.npy")
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.stage_check(decl, _quiet)
    assert "q_row is not each query's position in r2" in _stopped(stops)


def test_the_check_refuses_an_earlier_levels_id_rs_id_a_carves_id_an_unassembled_carve_and_a_failed_direction_check(
        tmp_path, monkeypatch, stops, one_thread):
    d, carves, decl, ids = _check_env(tmp_path, monkeypatch, n_q=12, seed=9)
    if not _levels_on_file(decl):
        pytest.skip("the earlier levels' qids.json files are not on this machine")
    for level in L15.LEVELS:   # level 14's rows are r
        bad = copy.deepcopy(decl)
        fake = tmp_path / f"{level}_qids.json"
        fake.write_text(json.dumps([ids[3]]), encoding="utf-8")
        bad["inputs"][level]["excluded_rows"]["qids"]["path"] = str(fake)
        with pytest.raises(SystemExit, match="HARD STOP"):
            L15.stage_check(bad, _quiet)
        last = _last_stop(stops)
        assert (last["message"], last["level"]) == ("check: a query id of level 0 or levels 8 to 14 is in r2", level)
    for c in ("fit", "select", "x5"):
        theirs = json.loads((carves / c / "qids.json").read_text(encoding="utf-8"))
        _fake_carve(carves / c, theirs + [ids[5]])
        with pytest.raises(SystemExit, match="HARD STOP"):
            L15.stage_check(decl, _quiet)
        assert f"r2 and level 12's carve {c} share ids" in _stopped(stops)
        _fake_carve(carves / c, theirs)
    theirs = json.loads((carves / "x7" / "qids.json").read_text(encoding="utf-8"))
    (carves / "x7" / "meta.json").unlink()
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.stage_check(decl, _quiet)
    assert "carve x7 is not assembled" in _stopped(stops)
    _fake_carve(carves / "x7", theirs)
    x2 = json.loads((carves / "x2" / "qids.json").read_text(encoding="utf-8"))
    (carves / "x2" / "qids.json").write_text(json.dumps(x2[:-1]), encoding="utf-8")   # not the sha256 its meta.json records
    with pytest.raises(SystemExit, match="not the sha256 its meta.json records"):
        L15.stage_check(decl, _quiet)
    _fake_carve(carves / "x2", x2)
    assert L15.stage_check(decl, _quiet)["carves_compared"] == {c: 5 for c in L12.CARVES}
    real = L8.chain_tokens
    monkeypatch.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))   # the map, turned around
    with pytest.raises(SystemExit, match="HARD STOP"):
        L15.stage_check(decl, _quiet)
    assert "direction_check (std)" in _stopped(stops)


# ── the stages' machine ──────────────────────────────────────────────────────


def test_the_stages_refuse_the_wrong_machine_and_every_wrong_combination(stops):
    for argv in (["--stage", "check"], ["--stage", "score"], ["--stage", "assemble"], ["--stage", "fit", "--host"],
                 ["--stage", "carve", "--host"], ["--stage", "check", "--host", "--shard", "0/12"],
                 ["--stage", "assemble", "--host", "--shard", "0/12"], ["--stage", "score", "--host", "--limit", "4"],
                 ["--stage", "check", "--host", "--limit", "4", "--out", "x"],
                 ["--stage", "assemble", "--host", "--limit", "4", "--out", "x"],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L15.OUT / "x")],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L15.OUT)],
                 ["--stage", "score", "--host", "--limit", "4", "--out", "x", "--shard", "0/12"]):
        with pytest.raises(SystemExit):
            L15.main(argv)


# ── r2's sidecar: only its pinned train-split ids ────────────────────────────


def test_r2s_sidecar_holds_its_pinned_train_split_ids_and_none_of_an_earlier_levels_or_of_level_12s_carves():
    side = L15.DATA / "qids.json"
    if not side.exists():
        pytest.skip("this file's sidecar is not on this machine")
    got = json.loads(side.read_text(encoding="utf-8"))
    decl = L15.load_declaration()
    pin = decl["carves"]["pins"]["r2"]
    assert m3b_pools.ids_digest(got) == pin["ids_sha256"] and len(got) == len(set(got)) == pin["queries"]
    assert all(L15.is_train_id(q) for q in got) and L15.hop_counts(got) == {int(h): int(v) for h, v in pin["hops"].items()}
    inp = decl["inputs"]
    for level in L15.LEVELS:
        earlier = set(json.loads((ROOT / inp[level]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
        assert not (set(got) & earlier), level
    for c in L12.CARVES:
        p = L12.CARVES_DIR / c / "qids.json"
        if p.exists():
            assert not (set(got) & set(json.loads(p.read_text(encoding="utf-8")))), c
    meta = L15.DATA / "meta.json"
    if meta.exists():
        m = json.loads(meta.read_text(encoding="utf-8"))
        assert m["qids_sha256"] == L0.sha256_file(side) and m["carve"] == "r2" and m["carve_ids_sha256"] == pin["ids_sha256"]
        assert m["limit"] is None and m["declaration_lf_sha256"] == L0.lf_sha256(L15.CONFIG)
