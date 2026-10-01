"""MP-Approx level 12, fit module (configs/mp_approx_l12.yaml#tests): the declared constants; a deploy view's roles on
synthetic sidecars (every dev row scored and in no fit or inner set, every select row with an in-pool gold inner, every
fit or x row with an in-pool gold fit, no carve row scored, per-query reads the part's) and its refusals (a part before
its check, after a change, as a smoke, off its pins, or twice) and a row off its role's set; a unit against level 11's
fit_and_capture plus score_dsh bit for bit; the scored rows' golds, the other seeds' twins and the stored metrics never
reaching a unit's scores; the bands, rho, reaches_gnn, the interpretation map and unseen_sequences on toy cases; the
code check; the check, carve check, fit, repeat, read, doc and file stages end to end on synthetic sidecars; and the
stages' machine."""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l0 as L0  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
import mp_approx_l9 as L9  # noqa: E402
import mp_approx_l10 as L10  # noqa: E402
import mp_approx_l10_fit as F10  # noqa: E402
import mp_approx_l11 as L11  # noqa: E402
import mp_approx_l11_fit as F11  # noqa: E402
import mp_approx_l12 as P  # noqa: E402
import mp_approx_l12_fit as F  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar and relation text)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)
import test_mp_approx_l12 as T12  # noqa: E402  (the population module's id rewrite)
from mp_retrieval import m3b_pools  # noqa: E402

_quiet = T8._quiet
TAGS = {"fit": "f", "select": "s", "x1": "xa", "x2": "xb", "x3": "xc", "x4": "xd", "x5": "xe", "x6": "xf", "x7": "xg"}
CHOICES = ("beta_dsh", "kappa_dsh", "eta_dsh", "kappa_cov", "eta_cov")
FORBIDDEN = ("message passing is unnecessary", "we do not need message passing", "the mlp wins")


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, P):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])   # the fit stages turn it on


def _light_fit_process():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _stopped(stops: Path) -> str:
    return (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── synthetic sidecars: the dev rows and the carves, checked ─────────────────


def _carve(root: Path, carve: str, monkeypatch, n_q: int, seed: int) -> list[str]:
    """Level 9's combined toy sidecar as a carve: train-split ids under the carve's tag, folds and inner flags by level
    0's rules, and meta.json's carve and carve_ids_sha256 as the carve stage files them."""
    d = root / "carves" / carve
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    ids = [q.replace(":toy:", f":train:{TAGS[carve]}") for q in qids]
    T12._rewrite_ids(d, ids, carve=carve, carve_ids_sha256=m3b_pools.ids_digest(ids))
    return ids


def _no_gold_in_pool(d: Path, qs: list[int]) -> None:
    """The queries' golds taken out of their pools (is_gold, t_gold and q_gold_in_pool zeroed, the gold total kept), with
    meta.json's sha256 values updated."""
    pool, types = np.load(d / "q_pool_size.npy"), np.load(d / "q_types.npy")
    node_ptr, type_ptr = np.r_[0, np.cumsum(pool)], np.r_[0, np.cumsum(types)]
    arrays = {key: np.load(d / f"{key}.npy") for key in ("is_gold", "t_gold", "q_gold_in_pool")}
    for q in qs:
        arrays["is_gold"][node_ptr[q]:node_ptr[q + 1]] = False
        arrays["t_gold"][type_ptr[q]:type_ptr[q + 1]] = 0
        arrays["q_gold_in_pool"][q] = 0
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    for key, arr in arrays.items():
        np.save(d / f"{key}.npy", arr)
        meta["arrays_sha256"][f"{key}.npy"] = L0.sha256_file(d / f"{key}.npy")
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _env(tmp_path, monkeypatch, carves: dict, n_dev: int = 24, zero: dict | None = None):
    """The dev sidecar and the named carves, every path of both modules pointed at them, the pins set to them, and the
    check and every carve check run."""
    root = tmp_path / "l12"
    dev = T9._sidecar(root / "metaqa", monkeypatch, L9.walk_entries_both, n_q=n_dev, seed=3)
    ids = {"eval": dev}
    for i, (c, n) in enumerate(carves.items()):
        ids[c] = _carve(root, c, monkeypatch, n, 20 + i)
    for c, qs in (zero or {}).items():
        _no_gold_in_pool(root / "carves" / c, qs)
    for mod in (P, F):
        monkeypatch.setattr(mod, "DATA", root / "metaqa")
        monkeypatch.setattr(mod, "CARVES_DIR", root / "carves")
        monkeypatch.setattr(mod, "LOOPCHECK", root / "loopcheck.json")
    monkeypatch.setattr(P, "verify_inputs", lambda decl: None)   # the population module's pins are its own tests'
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(dev), dtype=bool), None, None))
    monkeypatch.setattr(L8, "fit_process", _light_fit_process)
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    decl = P.load_declaration()
    decl["population"]["per_hop"] = P.hop_counts(dev)
    for c in carves:
        decl["carves"]["pins"][c] = {"queries": len(ids[c]), "zero_gold_excluded": 0, "ids_sha256": m3b_pools.ids_digest(ids[c]),
                                     "hops": P.hop_counts(ids[c])}
    P.stage_check(decl, _quiet)
    for c in carves:
        P.stage_carve_check(decl, c, _quiet)
    return root, decl, ids


# ── the declared constants ───────────────────────────────────────────────────


def test_the_declared_constants_are_the_files():
    decl = P.load_declaration()
    assert {k: tuple(v) for k, v in decl["carves"]["fits"].items()} == F.FITS and list(decl["carves"]["fits"]) == list(F.FITS)
    assert {k: tuple(v) for k, v in decl["units"]["roles"].items()} == F.ROLES
    arms = decl["arms"]
    assert {k: tuple(v) for k, v in arms["read_arms"].items()} == F.READ_ARMS and list(arms["read_arms"]) == list(F.READ_ARMS)
    assert set(arms["references"]) == set(F.REFERENCES) and set(arms["scores"]) == set(F.SCORES) == set(F11.SCORES)
    assert decl["readings"]["primary"] == F.PRIMARY
    assert {k: tuple(s.strip() for s in v.split(" - ")) for k, v in decl["statistics"]["contrasts"].items()} == F.CONTRASTS
    assert set(decl["readings"]["bands"]) == set(F.BANDS) and set(decl["readings"]["flags"]) == set(F.FLAGS)
    assert tuple(decl["readings"]["interpretation_map"]) == F.INTERPRETATION
    assert F.REPEAT_UNIT == ("TW-1x", 0) and "The unit (TW-1x, k = 0) runs again" in " ".join(decl["units"]["repeat"].split())
    assert F.GAP_SPLIT_ARMS == ("TW-1x-dsh", "TW-8x-dsh") and "For TW-1x-dsh and TW-8x-dsh" in decl["quantities"]["anchors"]["gap_split"]
    assert F.FOLD == 0 and "One unit per (fit, k) at fold 0" in decl["units"]["per_fit"] and tuple(F.SEEDS) == (0, 1, 2)
    assert "units/<fit>/k<k>_f0" in decl["units"]["unit_files"]
    assert F.unit_paths("TW-4x", 2) == (F.DATA / "units" / "TW-4x" / "k2_f0.npz", F.DATA / "units" / "TW-4x" / "k2_f0.json")
    assert F.part_names("TW-8x") == ("eval", "select", "fit", *P.X_CARVES) and set(F.part_names("TW-8x")[1:]) == set(P.CARVES)
    assert [F.role_of(p) for p in F.part_names("TW-2x")] == ["eval", "select", "fit_and_x", "fit_and_x"]
    assert F.SCRIPT_REL in decl["outputs"]["scripts"] and P.SCRIPT_REL in decl["outputs"]["scripts"]
    assert F.DOC == ROOT / decl["outputs"]["document"]
    assert (F.CONFIG, F.OUT, F.DATA, F.NAME, F.CARVES_DIR, F.LOOPCHECK) == (P.CONFIG, P.OUT, P.DATA, P.NAME, P.CARVES_DIR, P.LOOPCHECK)
    assert F.BETAS == F11.BETAS and F11.LEVEL10_FIT == "NB-set" and "level 10's fit_unit for NB-set" in arms["procedure"]
    pins, per_hop = decl["carves"]["pins"], decl["population"]["per_hop"]
    views = [sum(per_hop.values()) + pins["select"]["queries"] + sum(pins[c]["queries"] for c in F.FITS[fit]) for fit in F.FITS]
    assert views == [9678, 15638, 27558, 51398]   # the views the reservations scale by
    text = " ".join(decl["placement"]["reservations"].split())
    assert "TW-1x's view has 9,678" in text and "(15,638, 27,558 and 51,398 against 9,678)" in text


# ── the deploy view ──────────────────────────────────────────────────────────


def test_a_deploy_view_sets_each_parts_role_and_reads_each_query_from_its_part(tmp_path, monkeypatch, stops, one_thread):
    root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 14, "x1": 12}, zero={"select": [2], "fit": [0, 5]})
    view = F.deploy_view(decl, "TW-2x")
    names = ("eval", "select", "fit", "x1")
    assert view.sources == names and view.family == "nb"
    assert [Path(v.dir) for v in view.parts] == [root / "metaqa", *(root / "carves" / c for c in names[1:])]
    sizes = [len(ids[p]) for p in names]
    part = np.repeat(np.arange(len(names)), sizes)
    assert np.array_equal(view.q_part, part) and view.n_q == sum(sizes)
    assert np.array_equal(view.q_fold, np.where(part == 0, 0, -1)) and view.q_fold.dtype == view.parts[0].q_fold.dtype
    assert np.array_equal(view.q_inner, (part == 1).astype(view.q_inner.dtype))
    fit_q, inner_q, score_q = L8.unit_queries(view, F.FOLD)
    gold = view.q_gold_in_pool > 0
    assert np.array_equal(score_q, np.flatnonzero(part == 0))           # every dev row scored, no carve row
    assert np.array_equal(inner_q, np.flatnonzero((part == 1) & gold))  # the select rows with an in-pool gold
    assert np.array_equal(fit_q, np.flatnonzero((part >= 2) & gold))    # the fit and x rows with an in-pool gold
    off = view.q_offset
    assert not gold[[off[1] + 2, off[2], off[2] + 5]].any() and gold[part == 0].all()
    assert F.rows_by_part(view) == {"fit": {"eval": 0, "select": 0, "fit": 12, "x1": 12}, "inner": {"eval": 0, "select": 11, "fit": 0, "x1": 0}}
    own = [L9.View(Path(v.dir), "nb") for v in view.parts]
    for q in range(view.n_q):
        v, i = own[int(part[q])], int(view.q_local[q])
        assert all(np.array_equal(x, y) for x, y in zip(view.entries(q), v.entries(i)))
        assert np.array_equal(view.gold_local(q), v.gold_local(i)) and all(np.array_equal(view.z(q, k), v.z(i, k)) for k in F.SEEDS)
        for key in ("t_code", "t_size", "t_gold"):
            assert np.array_equal(getattr(view, key)[view.type_rows(q)], getattr(v, key)[v.type_rows(i)]), key
        assert view.meta["qtypes"][view.q_qtype[q]] == v.meta["qtypes"][v.q_qtype[i]] and np.array_equal(view.q_emb[q], v.q_emb[i])
        for key in ("q_hop", "q_pool_size", "q_gold_total", "q_gold_in_pool", "q_types", "q_entries"):
            assert getattr(view, key)[q] == getattr(v, key)[i], key
    assert F.deploy_view(decl, "TW-1x").sources == ("eval", "select", "fit")
    assert not (stops / "hard_stops.json").exists()


def test_a_part_is_refused_before_its_check_after_a_change_as_a_smoke_off_its_pins_or_twice(tmp_path, monkeypatch, stops, one_thread):
    root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 10, "fit": 10})
    F.deploy_view(decl, "TW-1x")
    with pytest.raises(SystemExit, match="has not passed"):
        F.deploy_view(decl, "TW-2x")   # x1 has no sidecar and no carve check
    d = root / "carves" / "fit"
    check_path, meta_path = d / "check.json", d / "meta.json"
    check_bytes, meta_bytes, hop_bytes = check_path.read_bytes(), meta_path.read_bytes(), (d / "q_hop.npy").read_bytes()
    check, meta = json.loads(check_bytes), json.loads(meta_bytes)
    failed = copy.deepcopy(check)
    failed["families"]["nb"]["direction_check"]["passes"] = False
    check_path.write_text(json.dumps(failed), encoding="utf-8")
    with pytest.raises(SystemExit, match="has not passed"):
        F.part_view(decl, "fit")
    check_path.write_bytes(check_bytes)
    meta_path.write_text(json.dumps({**meta, "note": "changed after the check"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.part_view(decl, "fit")
    assert "part fit's meta.json is not the one its carve_check read" in _stopped(stops)
    meta_path.write_text(json.dumps({**meta, "limit": 5}), encoding="utf-8")
    check_path.write_text(json.dumps({**check, "meta_sha256": L0.sha256_file(meta_path)}), encoding="utf-8")
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.part_view(decl, "fit")
    assert "part fit is a smoke run" in _stopped(stops)
    meta_path.write_bytes(meta_bytes)
    check_path.write_bytes(check_bytes)
    np.save(d / "q_hop.npy", np.load(d / "q_hop.npy") + 1)   # an array that is not what meta.json records
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.part_view(decl, "fit")
    assert "deploy view: part fit:" in _stopped(stops)
    (d / "q_hop.npy").write_bytes(hop_bytes)
    F.part_view(decl, "fit")
    for key, value in (("ids_sha256", "0" * 64), ("queries", 11), ("hops", {1: 10, 2: 0, 3: 0})):
        bad = copy.deepcopy(decl)
        bad["carves"]["pins"]["fit"][key] = value
        with pytest.raises(SystemExit, match="HARD STOP"):
            F.part_view(bad, "fit")
        assert "part fit is not what its pins record" in _stopped(stops)
    bad = copy.deepcopy(decl)
    bad["population"]["per_hop"] = {1: 221, 2: 1000, 3: 1000}
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.part_view(bad, "eval")
    assert "part eval is not what its pins record" in _stopped(stops)
    monkeypatch.setitem(F.FITS, "TW-1x", ("fit", "fit"))
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.deploy_view(decl, "TW-1x")   # one carve twice
    assert "a query id is in two parts" in _stopped(stops)


def test_a_row_off_its_roles_set_stops(tmp_path, monkeypatch, stops, one_thread):
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 10, "fit": 10})
    view = F.deploy_view(decl, "TW-1x")
    n_dev = len(ids["eval"])
    for key, q in (("q_fold", n_dev + 3), ("q_fold", 2), ("q_inner", n_dev + 12), ("q_inner", n_dev + 1)):
        arr = getattr(view, key)
        before = arr[q]
        arr[q] = {"q_fold": 0 if q >= n_dev else -1, "q_inner": 1 - before}[key]   # a select row scored, a dev row fitted, ...
        with pytest.raises(SystemExit, match="HARD STOP"):
            F.rows_by_part(view)
        assert "a row's set is not its role's" in _stopped(stops)
        arr[q] = before
    assert F.rows_by_part(view) == {"fit": {"eval": 0, "select": 0, "fit": 10}, "inner": {"eval": 0, "select": 10, "fit": 0}}


# ── a unit ───────────────────────────────────────────────────────────────────


def test_a_unit_is_level_11s_fit_and_capture_plus_score_dsh_bit_for_bit(tmp_path, monkeypatch, stops, one_thread):
    torch.use_deterministic_algorithms(True)
    root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    rel = T8._rel()
    out, flog = F.run_unit(F10.make_fitter(F.deploy_view(decl, "TW-1x"), rel, F11.LEVEL10_FIT), "TW-1x", 0, _quiet)
    fx = F10.make_fitter(F.deploy_view(decl, "TW-1x"), rel, F11.LEVEL10_FIT)
    arrays, f11, cap = F11.fit_and_capture(fx, 0, 0, _quiet)
    d = F11.score_dsh(fx, 0, cap)
    assert set(out) == set(arrays) | {"metrics_dsh", "score_dsh"}
    for key in arrays:
        assert out[key].dtype == arrays[key].dtype and np.array_equal(out[key], arrays[key]), key
    assert np.array_equal(out["metrics_dsh"], d["metrics"]) and np.array_equal(out["score_dsh"], np.concatenate(d["scores"]))
    assert {key: v for key, v in flog.items() if key in f11 and key != "timing"} == {key: v for key, v in f11.items() if key != "timing"}
    assert set(flog) - set(f11) == {"l12_fit", "parts", "parts_meta_sha256", "rows_by_part", "unseen_sequences", "beta_dsh", "kappa_dsh",
                                    "eta_dsh", "inner_mean3_dsh", "grid_inner_mean3_dsh"}
    assert (flog["beta_dsh"], flog["kappa_dsh"], flog["eta_dsh"]) == (F11.beta_label(d["beta"]), d["kappa"], d["eta"])
    assert flog["inner_mean3_dsh"] == d["inner_mean3"] and flog["grid_inner_mean3_dsh"] == d["grid"]
    n_dev = len(ids["eval"])
    assert np.array_equal(out["q"], np.arange(n_dev)) and (flog["fit_queries"], flog["inner_queries"], flog["scored_queries"]) == (16, 12, n_dev)
    assert flog["parts"] == ["eval", "select", "fit"] and flog["parts_meta_sha256"] == {
        "eval": L0.sha256_file(root / "metaqa" / "meta.json"), **{c: L0.sha256_file(root / "carves" / c / "meta.json") for c in ("select", "fit")}}
    with pytest.raises(SystemExit, match="are not the fit's"):
        F.run_unit(F10.make_fitter(F.deploy_view(decl, "TW-1x"), rel, F11.LEVEL10_FIT), "TW-2x", 0, _quiet)


def test_the_scored_rows_golds_the_other_seeds_twins_and_the_stored_metrics_never_reach_a_units_scores(tmp_path, monkeypatch, stops, one_thread):
    """arms: each arm reads z(T_k), the query embedding, the relation text and the nb family's compiled types and counts,
    nothing else. The dev rows' golds are read only by their metrics."""
    torch.use_deterministic_algorithms(True)
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    rel = T8._rel()
    first, f1 = F.run_unit(F10.make_fitter(F.deploy_view(decl, "TW-1x"), rel, F11.LEVEL10_FIT), "TW-1x", 0, _quiet)
    names = F.part_names("TW-1x")
    parts = [F.part_view(decl, p) for p in names]
    ev = parts[0]
    is_gold, t_gold = np.array(ev.is_gold, copy=True), np.array(ev.t_gold, copy=True)
    rng = np.random.default_rng(0)
    for q in range(ev.n_q):
        sl = ev.nodes(q)
        is_gold[sl] = rng.permutation(is_gold[sl])   # the same count, other nodes
        rows = ev.type_rows(q)
        t_gold[rows] = rng.integers(0, 3, size=t_gold[rows].size)
    ev.is_gold, ev.t_gold = is_gold, t_gold
    for v in parts:
        twin = np.array(v.twin_score, copy=True)
        twin[:, 1:] = rng.normal(size=twin[:, 1:].shape).astype(twin.dtype)   # seeds 1 and 2: unit k = 0 never reads them
        v.twin_score = twin
        v.q_metrics = np.full(np.shape(v.q_metrics), np.nan)                   # the stored T and G metrics
    again, f2 = F.run_unit(F10.make_fitter(F.with_roles(F11.MultiView(parts, names)), rel, F11.LEVEL10_FIT), "TW-1x", 0, _quiet)
    for key in ("q", "score_dens", "score_cov", "score_dsh", "score_ptr", "argmax"):
        assert np.array_equal(first[key], again[key]), key
    assert [f1[key] for key in CHOICES] == [f2[key] for key in CHOICES] and f1["kept_round"] == f2["kept_round"]
    assert not np.array_equal(first["metrics_dsh"], again["metrics_dsh"])   # the metrics do read the golds


# ── the statistics and the readings on toy cases ─────────────────────────────


def test_the_bands_come_in_the_declared_order_and_rho_is_the_share_of_the_gain():
    assert F.band(1.3, [1.05, 1.5], True) == "L12_ABOVE_GNN"
    assert F.band(0.8, [0.6, 1.0], True) == "L12_HIGH"
    assert F.band(0.8, [0.4, 1.2], True) == "L12_MID"
    assert F.band(0.2, [0.0, 0.45], True) == "L12_LOW"
    assert F.band(0.2, [0.0, 0.55], True) == "L12_MID"
    assert F.band(None, None, False) == "NOT_READ" and F.band(0.9, [0.8, 1.0], False) == "NOT_READ"
    rng = np.random.default_rng(0)
    T = rng.random((40, 3, 3))
    G = T + 0.5
    W = L0.boot_weights(40)
    _d, dens, readable = L8.denominators(T, G, W)
    assert readable == list(L8.RETRIEVAL)
    half = F.read_arm(T + 0.25, T, G, dens, readable, W)
    assert abs(half["rho_bar"]["point"] - 0.5) < 1e-12 and all(abs(half["rho"][m]["point"] - 0.5) < 1e-12 for m in L8.RETRIEVAL)
    assert half["band"] == "L12_MID" and all(half["gap_to_gnn"][m]["flag"] == "BELOW_GNN" for m in L8.RETRIEVAL)
    assert F.read_arm(T + 0.45, T, G, dens, readable, W)["band"] == "L12_HIGH"
    assert F.read_arm(T + 0.6, T, G, dens, readable, W)["band"] == "L12_ABOVE_GNN"
    assert F.read_arm(T + 0.05, T, G, dens, readable, W)["band"] == "L12_LOW"
    _d, dens0, none = L8.denominators(T, T, W)   # no gain: nothing is readable
    assert none == [] and F.read_arm(T + 0.1, T, T, dens0, none, W)["band"] == "NOT_READ"


def _arm(flags) -> dict:
    return {"gap_to_gnn": {m: {"flag": f} for m, f in zip(L8.RETRIEVAL, flags)}}


def test_reaches_gnn_is_the_first_fit_with_no_readable_metric_below_the_gnn():
    tr = {fit: {"fit": {"total": 100 * (i + 1)}, "inner": {"total": 10}} for i, fit in enumerate(F.FITS)}
    arms = {"TW-1x-dsh": _arm(["BELOW_GNN", None, None]), "TW-2x-dsh": _arm([None, "BELOW_GNN", None]),
            "TW-4x-dsh": _arm([None, None, "BEATS_GNN"]), "TW-8x-dsh": _arm([None, None, None])}
    every = list(L8.RETRIEVAL)
    assert F.reaches_gnn(arms, every, tr) == {"fit": "TW-4x", "arm": "TW-4x-dsh", "fit_rows": 300, "inner_rows": 10}
    assert F.reaches_gnn(arms, ["recall@5"], tr)["fit"] == "TW-2x"   # TW-2x's flag is on a metric that is not read
    below = _arm(["BELOW_GNN"] * 3)
    assert F.reaches_gnn({**arms, "TW-4x-dsh": below, "TW-8x-dsh": below}, every, tr) is None
    assert F.reaches_gnn(arms, [], tr) is None


def test_the_interpretation_lists_every_entry_that_applies_in_the_declared_order():
    every = list(L8.RETRIEVAL)
    flat = {c: {"ci": [-0.1, 0.1]} for c in F.CONTRASTS}

    def ci(**kw):
        return {**flat, **{c: {"ci": v} for c, v in kw.items()}}

    cases = [
        (("L12_HIGH", _arm([None] * 3), every, flat, 0.9), ["l12_high", "matched_not_below_gnn", "last_doubling_flat"]),
        (("L12_MID", _arm(["BELOW_GNN", None, None]), every,
          ci(data_2x=[0.01, 0.2], data_4x=[0.02, 0.3], data_8x=[0.03, 0.4], step_8x=[0.01, 0.1], dsh_adds=[-0.2, -0.01],
             ceiling_gap=[0.1, 0.3]), 0.4),
         ["l12_mid", "matched_below_gnn", "data_adds_2x", "data_adds_4x", "data_adds_8x", "last_doubling_adds", "dsh_hurts",
          "chain_not_identified"]),
        (("L12_LOW", _arm([None] * 3), every, ci(data_8x=[-0.3, -0.01], step_8x=[-0.3, -0.2], dsh_adds=[0.01, 0.02]), 0.6),
         ["l12_low", "matched_not_below_gnn", "data_hurts_8x", "dsh_adds"]),
        (("L12_ABOVE_GNN", _arm(["BEATS_GNN"] * 3), every, ci(ceiling_gap=[0.1, 0.3]), 0.7),
         ["l12_above_gnn", "matched_not_below_gnn", "last_doubling_flat"]),
        (("NOT_READ", _arm([None] * 3), [], {c: {"ci": None} for c in F.CONTRASTS}, float("nan")), []),
    ]
    for args, want in cases:
        got = F.interpretation(*args)
        assert got == want and [x for x in F.INTERPRETATION if x in got] == got


def test_unseen_sequences_counts_the_scored_rows_sequences_on_no_fit_row():
    qt1, qt2 = "movie_to_actor", "writer_to_movie_to_year"
    c1, c2 = (L8.type_code(0, L8.chain_tokens(L8.true_chain(qt))) for qt in (qt1, qt2))
    cx, cy, cz = L8.type_code(0, [26]), L8.type_code(0, [25, 24]), L8.type_code(0, [23, 22, 21])
    assert len({c1, c2, cx, cy, cz}) == 5
    b1 = L8.TB ** L8.MAX_L   # bucket 1: the same sequence
    data = SimpleNamespace(n_q=4, meta={"qtypes": [qt1, qt2]}, q_qtype=np.array([0, 1, 0, 1]), q_types=np.array([3, 1, 2, 1]),
                           t_code=np.array([c1, c1 + b1, cx, cy, c1, cz, c2 + b1]))
    got = F.unseen_sequences(data, np.array([0, 1]), np.array([2, 3]))
    assert got == {"scored_sequences": 3, "fit_sequences": 3, "unseen": 2, "true_chain_unseen_share": 0.5}
    assert F.unseen_sequences(data, np.array([0, 1, 3]), np.array([2]))["true_chain_unseen_share"] == 0.0
    assert F.unseen_sequences(data, np.array([0]), np.array([], dtype=np.int64))["true_chain_unseen_share"] is None


def test_the_identical_code_check_needs_every_script_in_one_committed_set_and_this_module_in_no_score_or_carve_job(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    a = "a" * 64
    score = {L8.SCRIPT_REL: a, L9.SCRIPT_REL: a, L10.SCRIPT_REL: a, L11.SCRIPT_REL: a, P.SCRIPT_REL: a}
    fit = {**score, F10.SCRIPT_REL: a, F11.SCRIPT_REL: a, F.SCRIPT_REL: a}
    good = {"score/dev/shard_0of2.json": score, "meta/dev": score, "check/dev": score, "score/fit/shard_0of4.json": score,
            "meta/fit": score, "check/fit": score, "loopcheck": score, "fit/TW-1x/k0": fit, "fit/TW-8x/k2": fit, "repeat": fit,
            "read": fit}
    assert F.code_problems(good, "HEAD") == []
    assert F.code_problems({**good, "meta/x1": {}}, "HEAD") == []   # a toy sidecar's meta records no module
    no_f11 = {rel: sha for rel, sha in fit.items() if rel != F11.SCRIPT_REL}
    assert any("are not in every fit" in p for p in F.code_problems({**good, "repeat": no_f11}, "HEAD"))
    assert any("are not in every fit" in p for p in F.code_problems({**good, "read": {**score, F10.SCRIPT_REL: a, F11.SCRIPT_REL: a}}, "HEAD"))
    assert any(p.startswith("check/dev: scripts/mp_approx_l12.py") for p in F.code_problems({**good, "check/dev": {L8.SCRIPT_REL: a}}, "HEAD"))
    assert any("in a score or carve job's record" in p for p in F.code_problems({**good, "score/x3/shard_1of4.json": fit}, "HEAD"))
    assert any("in a score or carve job's record" in p for p in F.code_problems({**good, "meta/select": fit}, "HEAD"))
    assert any("2 different" in p for p in F.code_problems({**good, "read": {**fit, F.SCRIPT_REL: "b" * 64}}, "HEAD"))
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "c" * 64)
    assert any("not the file at" in p for p in F.code_problems(good, "HEAD"))


# ── the stages end to end on synthetic sidecars ──────────────────────────────


def test_check_carve_check_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16, **{c: 10 for c in P.X_CARVES}}, n_dev=30)
    cfg = tmp_path / "mp_approx_l12.yaml"
    shutil.copyfile(P.CONFIG, cfg)   # the file stage appends its run record to a copy
    monkeypatch.setattr(F, "CONFIG", cfg)
    monkeypatch.setattr(F, "RECORD", root / "record.json")
    monkeypatch.setattr(F, "DOC", tmp_path / "MP_APPROX_L12.md")
    monkeypatch.setattr(F, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    loop = {"stage": "loopcheck", "queries": 24, "limit": 24, "rows_equal": True, "qtypes_equal": True, "chunk_queries": 8, "integrity": "toy",
            "arrays_compared": list(L8.ARRAY_KEYS), "differences": [], "equal": False, "module_sha256": {}}
    (root / "loopcheck.json").write_text(json.dumps(loop), encoding="utf-8")
    for fit in F.FITS:
        F.stage_fit(decl, fit, 0, _quiet)
    said = []
    F.stage_fit(decl, "TW-1x", 0, said.append)   # a unit already written is skipped on a restart
    assert any("not refitted" in s for s in said)
    F.stage_repeat(decl, _quiet)
    with pytest.raises(SystemExit, match="loopcheck"):
        F.stage_read(decl, _quiet)
    (root / "loopcheck.json").write_text(json.dumps({**loop, "equal": True}), encoding="utf-8")
    rd = F.stage_read(decl, _quiet)
    assert rd["primary"] == "TW-1x-dsh" and rd["reading"] == rd["arms"]["TW-1x-dsh"]["band"] and rd["reading"] in F.BANDS
    assert rd["readable_metrics"] and all(v["point"] is not None for v in rd["contrasts"].values())   # the toy is read, not skipped
    assert set(rd["arms"]) == set(F.READ_ARMS) | set(F.REFERENCES) and set(rd["contrasts"]) == set(F.CONTRASTS)
    for c_name, (a, b) in F.CONTRASTS.items():
        point = rd["contrasts"][c_name]["point"]
        assert point is None or abs(point - (rd["arms"][a]["rho_bar"]["point"] - rd["arms"][b]["rho_bar"]["point"])) < 1e-12
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"] and set(rd["flags"]) <= set(F.FLAGS)
    assert [x for x in F.INTERPRETATION if x in rd["interpretation"]] == rd["interpretation"]
    an = rd["anchors"]
    assert set(an) == set(decl["quantities"]["anchors"]) and set(rd["strata"]) == {"hop=1", "hop=2", "hop=3"}
    assert set(an["level9_anchors"]) == set(an["carve_metrics"]) == {"dev", *P.CARVES}
    checks = {c: json.loads((root / "carves" / c / "check.json").read_text(encoding="utf-8")) for c in P.CARVES}
    tr = an["training_rows"]
    for fit, carves in F.FITS.items():
        assert tr[fit]["fit"]["total"] == sum(checks[c]["rows_with_gold_in_pool"]["all"] for c in carves)
        assert tr[fit]["inner"]["total"] == tr[fit]["inner"]["select"] == checks["select"]["rows_with_gold_in_pool"]["all"]
        assert tr[fit]["fit"]["eval"] == tr[fit]["fit"]["select"] == tr[fit]["inner"]["eval"] == 0
    un = an["unseen_sequences"]
    assert un["TW-8x"]["unseen"] <= un["TW-4x"]["unseen"] <= un["TW-2x"]["unseen"] <= un["TW-1x"]["unseen"]
    for a, c_name in zip(F.GAP_SPLIT_ARMS, ("ceiling_gap", "ceiling_gap_8x")):
        g = an["gap_split"][a]
        assert g["right_pairs"] + g["wrong_pairs"] == len(ids["eval"])
        if g["right"] is not None:   # the split adds up to the ceiling gap
            assert abs(g["right"]["all"] + g["wrong"]["all"] - rd["contrasts"][c_name]["point"]) < 1e-9
    assert set(an["beta_choices"]) == {a for a, (_f, sc) in F.READ_ARMS.items() if sc == "dsh"}
    assert all(sum(v.values()) == 1 for v in an["beta_choices"].values()) and set(an["grid_edges"]) == set(F.READ_ARMS)
    n_dev = len(ids["eval"])
    for fit in F.FITS:
        npz, js = F.unit_paths(fit, 0)
        with np.load(npz) as z:
            assert np.array_equal(z["q"], np.arange(n_dev)) and z["metrics_dsh"].shape == z["metrics_cov"].shape
        flog = json.loads(js.read_text(encoding="utf-8"))
        assert (flog["l12_fit"], flog["fit"], flog["parts"], flog["k"], flog["fold"]) == (fit, "NB-set", list(F.part_names(fit)), 0, 0)
        assert flog["scored_queries"] == n_dev and len(flog["rounds"]) == 4 and flog["deterministic_algorithms"] is True
        assert {F.SCRIPT_REL, F10.SCRIPT_REL, F11.SCRIPT_REL, P.SCRIPT_REL} <= set(flog["module_sha256"])
    x3 = root / "carves" / "x3" / "check.json"
    saved = x3.read_bytes()
    x3.write_text(json.dumps({**json.loads(saved), "note": "changed after the read"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was not made from"):
        F.stage_doc(_quiet)
    x3.write_bytes(saved)
    F.stage_doc(_quiet)
    body = F.DOC.read_text(encoding="utf-8")
    for phrase in ("TW-1x-dsh", "label-matched", "none is a matched comparison", "It is not the within-U_q oracle rho",
                   "It is never a deployable or selected model", "reaches_gnn"):
        assert phrase in body, phrase
    assert not any(p in body.lower() for p in FORBIDDEN)
    rec = json.loads(F.RECORD.read_text(encoding="utf-8"))
    assert rec["phase"] == "MP_APPROX_L12" and rec["sources_sha256"]["read.json"] == L0.sha256_file(root / "metaqa" / "read.json")
    assert set(rec["carves"]) == set(P.CARVES) and rec["loopcheck"]["equal"] is True
    F.stage_file("2026_10_01", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l12_2026_10_01"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert set(run["contrasts"]) == set(F.CONTRASTS) and set(run["training_rows"]) == set(F.FITS) and run["primary_label_matched"] is True
    assert run["train_split_rows_scored"] == 0 and run["held_rows_read"] is False
    with pytest.raises(SystemExit, match="exists"):
        F.stage_file("2026_10_01", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_the_stages_refuse_the_wrong_machine_and_wrong_arguments(stops):
    for argv in (["--stage", "fit", "--fit", "TW-1x", "--k", "0"], ["--stage", "repeat"], ["--stage", "read"],
                 ["--stage", "doc", "--host"], ["--stage", "file", "--host"], ["--stage", "fit", "--host"],
                 ["--stage", "fit", "--host", "--fit", "TW-2x"], ["--stage", "fit", "--host", "--k", "0"],
                 ["--stage", "repeat", "--host", "--fit", "TW-1x"], ["--stage", "read", "--host", "--k", "0"],
                 ["--stage", "fit", "--host", "--fit", "TW-3x", "--k", "0"], ["--stage", "fit", "--host", "--fit", "TW-1x", "--k", "3"],
                 ["--stage", "doc", "--date", "2026_10_01"], ["--stage", "read", "--host", "--commit", "abc"], ["--stage", "file"],
                 ["--stage", "score", "--host"], ["--stage", "check", "--host"]):
        with pytest.raises(SystemExit):
            F.main(argv)
    assert not (stops / "hard_stops.json").exists()
    assert math.isinf(F.BETAS[-1])
