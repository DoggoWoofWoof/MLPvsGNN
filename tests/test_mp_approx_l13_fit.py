"""MP-Approx level 13, fit module (configs/mp_approx_l13.yaml#tests): the declared constants; b1d on toy posteriors (the
bucket-1 mass goes to the null type, the bucket-0 types keep theirs, a query without a bucket-1 type keeps its
log-probabilities bit for bit) and on a toy capture; a unit against level 12's run_unit plus level 11's score_dsh on the
b1d capture, bit for bit; the scored rows' golds, the other seeds' twins and the stored metrics never reaching a unit's
dsh or b1d scores; the bands, rho, reaches_gnn and the interpretation map on toy cases; the code check; the fit, repeat,
read, doc and file stages end to end on synthetic sidecars; and the stages' machine."""

from __future__ import annotations

import json
import math
import shutil
import sys
from fractions import Fraction
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
import mp_approx_l12 as P12  # noqa: E402
import mp_approx_l12_fit as F12  # noqa: E402
import mp_approx_l13 as P  # noqa: E402
import mp_approx_l13_fit as F  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar and relation text)
import test_mp_approx_l12_fit as T12F  # noqa: E402  (level 12's synthetic dev sidecar and carves, with level 12's checks run)

_quiet = T8._quiet
TBL = L8.TB ** L8.MAX_L
CHOICES = ("beta_dsh", "kappa_dsh", "eta_dsh", "kappa_cov", "eta_cov", "beta_b1d", "kappa_b1d", "eta_b1d")
B1D_KEYS = {"l13_fit", "beta_b1d", "kappa_b1d", "eta_b1d", "inner_mean3_b1d", "grid_inner_mean3_b1d", "b1d_changed", "b1d_mass_moved"}
FORBIDDEN = ("message passing is unnecessary", "we do not need message passing", "the mlp wins")


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, P12, P):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])   # the fit stages turn it on


def _env(tmp_path, monkeypatch, carves: dict, n_dev: int = 24, zero: dict | None = None):
    """Level 12's synthetic dev sidecar and carves (its _env: level 12's paths pointed at them and level 12's checks run),
    then this file's paths pointed at them, this file's per_hop and carve pins set to them, and this file's check run on
    the dev sidecar."""
    root, decl12, ids = T12F._env(tmp_path, monkeypatch, carves, n_dev=n_dev, zero=zero)
    for mod in (P, F):
        monkeypatch.setattr(mod, "DATA", root / "metaqa")
    monkeypatch.setattr(F, "CARVES_DIR", root / "carves")
    monkeypatch.setattr(P, "verify_inputs", lambda decl: None)   # the population module's pins are its own tests'
    decl = P.load_declaration()
    decl["population"]["per_hop"] = decl12["population"]["per_hop"]
    for c in carves:
        decl["carves"]["pins"][c] = decl12["carves"]["pins"][c]
    P.stage_check(decl, _quiet)
    return root, decl, ids


def _fitter(decl, fit="TW-1x"):
    return F10.make_fitter(F.deploy_view(decl, fit), T8._rel(), F11.LEVEL10_FIT)


# ── the declared constants ───────────────────────────────────────────────────


def test_the_declared_constants_are_the_files():
    decl = P.load_declaration()
    assert {k: tuple(v) for k, v in decl["carves"]["fits"].items()} == F.FITS and list(decl["carves"]["fits"]) == list(F.FITS)
    assert all(F12.FITS[fit] == carves for fit, carves in F.FITS.items())   # level 12's fits of these names
    assert set(P.L12_CARVES) == {"select", *(c for cs in F.FITS.values() for c in cs)}
    assert {k: tuple(v) for k, v in decl["units"]["roles"].items()} == F12.ROLES
    arms = decl["arms"]
    assert {k: tuple(v) for k, v in arms["read_arms"].items()} == F.READ_ARMS and list(arms["read_arms"]) == list(F.READ_ARMS)
    assert set(arms["references"]) == set(F.REFERENCES) and tuple(arms["scores"]) == F.SCORES
    assert decl["readings"]["primary"] == F.PRIMARY == "TW-1x-b1d"
    assert {k: tuple(s.strip() for s in v.split(" - ")) for k, v in decl["statistics"]["contrasts"].items()} == F.CONTRASTS
    assert set(decl["readings"]["bands"]) == set(F.BANDS) and set(decl["readings"]["flags"]) == set(F.FLAGS)
    assert tuple(decl["readings"]["interpretation_map"]) == F.INTERPRETATION
    assert F.REPEAT_UNIT == ("TW-1x", 0) and "The unit (TW-1x, k = 0) runs again" in " ".join(decl["units"]["repeat"].split())
    assert F.GAP_SPLIT_ARMS == ("TW-1x-b1d", "TW-4x-b1d") and "For TW-1x-b1d and TW-4x-b1d" in decl["quantities"]["anchors"]["gap_split"]
    assert set(decl["quantities"]["anchors"]) >= {"b1d_moved", "beta_choices", "reaches_gnn"}
    assert F.FOLD == 0 and "One unit per (fit, k) at fold 0" in decl["units"]["per_fit"] and tuple(F.SEEDS) == (0, 1, 2)
    assert "units/<fit>/k<k>_f0" in decl["units"]["unit_files"]
    assert F.unit_paths("TW-4x", 2) == (F.DATA / "units" / "TW-4x" / "k2_f0.npz", F.DATA / "units" / "TW-4x" / "k2_f0.json")
    assert F.SCRIPT_REL in decl["outputs"]["scripts"] and P.SCRIPT_REL in decl["outputs"]["scripts"]
    assert F.DOC == ROOT / decl["outputs"]["document"]
    assert (F.CONFIG, F.OUT, F.DATA, F.NAME) == (P.CONFIG, P.OUT, P.DATA, P.NAME)
    assert F.CARVES_DIR == F12.CARVES_DIR == ROOT / decl["inputs"]["level12_carves"]["dir"]
    assert F.HOPS == (2, 3) and {h for h, n in decl["population"]["per_hop"].items() if n} == set(F.HOPS)
    assert (F.BETAS, F.KAPPAS, F.ETAS) == (F11.BETAS, F10.KAPPAS, F10.ETAS) and F11.LEVEL10_FIT == "NB-set"
    sel = " ".join(arms["selection"].split())
    assert "beta in {1, 1.5, 2, 3, 4, 8, infinity}" in sel and "eta in {0.01, 0.1, 1, 10, 100}" in sel
    assert [float(Fraction(x)) for x in sel.split("kappa in {")[1].split("}")[0].split(", ")] == list(F.KAPPAS)
    assert F.TBL == 28 ** 3 == 21952 and "code // 28^3 = 1" in " ".join(arms["scores"]["b1d"].split())
    assert set(F12.UNIT_KEYS) < set(F.UNIT_KEYS)
    pins, per_hop = decl["carves"]["pins"], decl["population"]["per_hop"]
    views = [sum(per_hop.values()) + pins["select"]["queries"] + sum(pins[c]["queries"] for c in F.FITS[fit]) for fit in F.FITS]
    assert views == [9457, 27337]   # the views the reservations name
    text = " ".join(decl["placement"]["reservations"].split())
    assert "TW-1x's view here has 9,457" in text and "here 27,337" in text


# ── b1d ──────────────────────────────────────────────────────────────────────


def test_b1d_moves_every_bucket_1_types_mass_to_the_null_type_and_keeps_a_query_without_one_bit_for_bit():
    p = np.array([0.1, 0.2, 0.3, 0.15, 0.25])        # four types, then the null type
    codes = np.array([5, TBL + 5, 7, TBL + 9])       # buckets 0, 1, 0, 1
    lp = np.log(p)
    out, moved, changed = F.b1drop(lp, codes)
    assert changed and moved == float(np.exp(lp)[[1, 3]].sum())
    q = np.exp(out)
    assert np.isneginf(out[[1, 3]]).all() and q[1] == q[3] == 0.0
    assert np.allclose(q[[0, 2]], p[[0, 2]], rtol=1e-15, atol=0) and abs(q[-1] - (p[-1] + p[1] + p[3])) < 1e-15
    assert abs(q.sum() - 1.0) < 1e-12
    for beta in F.BETAS:   # the tempered posterior gives the dropped types nothing, at every beta
        t = np.exp(F11.tempered(out, beta))
        assert t[1] == t[3] == 0.0 and abs(t.sum() - 1.0) < 1e-12
    lp0 = np.log(np.array([0.5, 0.3, 0.2]))
    same, moved0, changed0 = F.b1drop(lp0, np.array([5, 2 * 28 + 3]))   # bucket 0 only
    assert same is lp0 and moved0 == 0.0 and not changed0
    assert F.b1drop(np.log(np.array([0.6, 0.4])), np.array([TBL + 1]))[0][0] == -np.inf   # all of it to the null type


def test_the_b1d_capture_changes_every_inner_and_scored_query_with_a_bucket_1_type_and_no_other():
    codes = np.array([3, TBL + 3, 4, 6, TBL + 6, TBL + 8])
    ptr = np.array([0, 2, 3, 6])     # query 0: types 0 and 1; query 1: type 2 (bucket 0 only); query 2: types 3 to 5
    data = SimpleNamespace(t_code=codes, type_rows=lambda q: slice(int(ptr[q]), int(ptr[q + 1])))
    rng = np.random.default_rng(0)

    def lp(n):
        x = rng.random(n + 1)
        return np.log(x / x.sum())

    cap = {"inner_q": np.array([1, 0]), "score_q": np.array([2, 1, 0]), "lp_inner": [lp(1), lp(2)], "lp_score": [lp(3), lp(1), lp(2)]}
    capb, info = F.b1d_capture(data, cap)
    assert capb["inner_q"] is cap["inner_q"] and capb["score_q"] is cap["score_q"]
    for qkey, key in (("inner_q", "lp_inner"), ("score_q", "lp_score")):
        for q, a, b in zip(cap[qkey], cap[key], capb[key]):
            assert np.array_equal(b, F.b1drop(a, codes[data.type_rows(int(q))])[0])
            if q == 1:
                assert b is a     # no bucket-1 type: the same array
    assert info["changed"] == {"inner": 1, "scored": 2}
    moved = [F.b1drop(a, codes[data.type_rows(int(q))])[1] for q, a in zip(cap["score_q"], cap["lp_score"])]
    assert info["mass_moved"] == float(np.mean(moved)) and moved[1] == 0.0 and min(moved[0], moved[2]) > 0


# ── a unit ───────────────────────────────────────────────────────────────────


def test_a_unit_is_level_12s_run_unit_plus_level_11s_score_dsh_on_the_b1d_capture_bit_for_bit(tmp_path, monkeypatch, stops, one_thread):
    torch.use_deterministic_algorithms(True)
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    out, flog = F.run_unit(_fitter(decl), "TW-1x", 0, _quiet)
    ref, rlog = F12.run_unit(_fitter(decl), "TW-1x", 0, _quiet)
    assert set(out) == set(ref) | {"metrics_b1d", "score_b1d"}
    for key in ref:
        assert out[key].dtype == ref[key].dtype and np.array_equal(out[key], ref[key]), key
    assert set(flog) - set(rlog) == B1D_KEYS
    assert {k: v for k, v in flog.items() if k in rlog and k != "timing"} == {k: v for k, v in rlog.items() if k != "timing"}
    fx = _fitter(decl)
    _arrays, _f11, cap = F11.fit_and_capture(fx, 0, 0, _quiet)
    capb, info = F.b1d_capture(fx.data, cap)
    d = F11.score_dsh(fx, 0, capb)
    assert np.array_equal(out["metrics_b1d"], d["metrics"]) and np.array_equal(out["score_b1d"], np.concatenate(d["scores"]))
    assert (flog["beta_b1d"], flog["kappa_b1d"], flog["eta_b1d"]) == (F11.beta_label(d["beta"]), d["kappa"], d["eta"])
    assert flog["inner_mean3_b1d"] == d["inner_mean3"] and flog["grid_inner_mean3_b1d"] == d["grid"]
    assert flog["b1d_changed"] == info["changed"] and flog["b1d_mass_moved"] == info["mass_moved"] and flog["l13_fit"] == "TW-1x"
    assert info["changed"]["scored"] > 0 and info["changed"]["inner"] > 0   # the toy has bucket-1 types, so b1d is exercised
    assert np.array_equal(out["q"], np.arange(len(ids["eval"]))) and out["score_b1d"].shape == out["score_dsh"].shape
    for fit in ("TW-2x", "TW-8x"):
        with pytest.raises(SystemExit, match="not one of this file's fits"):
            F.deploy_view(decl, fit)
    assert not (stops / "hard_stops.json").exists()


def test_the_scored_rows_golds_the_other_seeds_twins_and_the_stored_metrics_never_reach_a_units_scores(tmp_path, monkeypatch, stops, one_thread):
    """arms: each arm reads z(T_k), the query embedding, the relation text and the nb family's compiled types and counts,
    nothing else. The dev rows' golds are read only by their metrics."""
    torch.use_deterministic_algorithms(True)
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    first, f1 = F.run_unit(_fitter(decl), "TW-1x", 0, _quiet)
    names = F12.part_names("TW-1x")
    parts = [F12.part_view(decl, p) for p in names]
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
    again, f2 = F.run_unit(F10.make_fitter(F12.with_roles(F11.MultiView(parts, names)), T8._rel(), F11.LEVEL10_FIT), "TW-1x", 0, _quiet)
    for key in ("q", "score_dens", "score_cov", "score_dsh", "score_b1d", "score_ptr", "argmax"):
        assert np.array_equal(first[key], again[key]), key
    assert [f1[key] for key in CHOICES] == [f2[key] for key in CHOICES] and f1["kept_round"] == f2["kept_round"]
    assert (f1["b1d_changed"], f1["b1d_mass_moved"]) == (f2["b1d_changed"], f2["b1d_mass_moved"])
    assert not np.array_equal(first["metrics_b1d"], again["metrics_b1d"])   # the metrics do read the golds


# ── the statistics and the readings on toy cases ─────────────────────────────


def test_the_bands_come_in_the_declared_order_and_rho_is_the_share_of_the_gain():
    assert F.band(1.3, [1.05, 1.5], True) == "L13_ABOVE_GNN"
    assert F.band(0.8, [0.6, 1.0], True) == "L13_HIGH"
    assert F.band(0.8, [0.4, 1.2], True) == "L13_MID"
    assert F.band(0.2, [0.0, 0.45], True) == "L13_LOW"
    assert F.band(0.2, [0.0, 0.55], True) == "L13_MID"
    assert F.band(None, None, False) == "NOT_READ" and F.band(0.9, [0.8, 1.0], False) == "NOT_READ"
    rng = np.random.default_rng(0)
    T = rng.random((40, 3, 3))
    G = T + 0.5
    W = L0.boot_weights(40)
    _d, dens, readable = L8.denominators(T, G, W)
    assert readable == list(L8.RETRIEVAL)
    half = F.read_arm(T + 0.25, T, G, dens, readable, W)
    assert abs(half["rho_bar"]["point"] - 0.5) < 1e-12 and all(abs(half["rho"][m]["point"] - 0.5) < 1e-12 for m in L8.RETRIEVAL)
    assert half["band"] == "L13_MID" and all(half["gap_to_gnn"][m]["flag"] == "BELOW_GNN" for m in L8.RETRIEVAL)
    assert F.read_arm(T + 0.45, T, G, dens, readable, W)["band"] == "L13_HIGH"
    assert F.read_arm(T + 0.6, T, G, dens, readable, W)["band"] == "L13_ABOVE_GNN"
    assert F.read_arm(T + 0.05, T, G, dens, readable, W)["band"] == "L13_LOW"
    _d, dens0, none = L8.denominators(T, T, W)   # no gain: nothing is readable
    assert none == [] and F.read_arm(T + 0.1, T, T, dens0, none, W)["band"] == "NOT_READ"


def _arm(flags) -> dict:
    return {"gap_to_gnn": {m: {"flag": f} for m, f in zip(L8.RETRIEVAL, flags)}}


def test_reaches_gnn_is_the_first_read_arm_with_no_readable_metric_below_the_gnn():
    tr = {"TW-1x": {"fit": {"total": 100}, "inner": {"total": 10}}, "TW-4x": {"fit": {"total": 400}, "inner": {"total": 10}}}
    every = list(L8.RETRIEVAL)
    arms = {"TW-1x-dsh": _arm(["BELOW_GNN", None, None]), "TW-1x-b1d": _arm([None, "BELOW_GNN", None]),
            "TW-4x-dsh": _arm([None, None, "BEATS_GNN"]), "TW-4x-b1d": _arm([None, None, None])}
    assert F.reaches_gnn(arms, every, tr) == {"arm": "TW-4x-dsh", "fit": "TW-4x", "fit_rows": 400, "inner_rows": 10}
    assert F.reaches_gnn(arms, ["recall@5"], tr) == {"arm": "TW-1x-b1d", "fit": "TW-1x", "fit_rows": 100, "inner_rows": 10}
    below = _arm(["BELOW_GNN"] * 3)
    assert F.reaches_gnn({a: below for a in arms}, every, tr) is None
    assert F.reaches_gnn(arms, [], tr) is None


def test_the_interpretation_lists_every_entry_that_applies_in_the_declared_order():
    every = list(L8.RETRIEVAL)
    flat = {c: {"ci": [-0.1, 0.1]} for c in F.CONTRASTS}

    def ci(**kw):
        return {**flat, **{c: {"ci": v} for c, v in kw.items()}}

    cases = [
        (("L13_HIGH", _arm([None] * 3), every, flat, 0.9), ["l13_high", "matched_not_below_gnn"]),
        (("L13_MID", _arm(["BELOW_GNN", None, None]), every,
          ci(b1d_adds=[0.01, 0.1], b1d_adds_4x=[-0.2, -0.01], data_4x=[0.01, 0.2], data_4x_b1d=[0.02, 0.3], ceiling_gap=[0.1, 0.3]), 0.4),
         ["l13_mid", "matched_below_gnn", "b1d_adds", "b1d_hurts_4x", "data_adds_4x", "data_adds_4x_b1d", "chain_not_identified"]),
        (("L13_LOW", _arm([None] * 3), every, ci(b1d_adds=[-0.3, -0.01], b1d_adds_4x=[0.01, 0.02], ceiling_gap=[0.1, 0.3]), 0.6),
         ["l13_low", "matched_not_below_gnn", "b1d_hurts", "b1d_adds_4x"]),
        (("L13_ABOVE_GNN", _arm(["BEATS_GNN"] * 3), every, ci(ceiling_gap=[0.1, 0.3]), 0.7), ["l13_above_gnn", "matched_not_below_gnn"]),
        (("NOT_READ", _arm([None] * 3), [], {c: {"ci": None} for c in F.CONTRASTS}, float("nan")), []),
    ]
    for args, want in cases:
        got = F.interpretation(*args)
        assert got == want and [x for x in F.INTERPRETATION if x in got] == got


def test_the_identical_code_check_needs_every_script_in_one_committed_set_and_this_module_in_no_score_job(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    a = "a" * 64
    score = {L8.SCRIPT_REL: a, L9.SCRIPT_REL: a, L10.SCRIPT_REL: a, L11.SCRIPT_REL: a, P12.SCRIPT_REL: a, P.SCRIPT_REL: a}
    fit = {**score, F10.SCRIPT_REL: a, F11.SCRIPT_REL: a, F12.SCRIPT_REL: a, F.SCRIPT_REL: a}
    good = {"score/dev/shard_0of4.json": score, "meta/dev": score, "check/dev": score, "fit/TW-1x/k0": fit, "fit/TW-4x/k2": fit,
            "repeat": fit, "read": fit}
    assert F.code_problems(good, "HEAD") == []
    assert F.code_problems({**good, "meta/dev": {}}, "HEAD") == []   # a toy sidecar's meta records no module
    no_f12 = {rel: sha for rel, sha in fit.items() if rel != F12.SCRIPT_REL}
    assert any("are not in every fit" in p for p in F.code_problems({**good, "repeat": no_f12}, "HEAD"))
    assert any("are not in every fit" in p for p in F.code_problems({**good, "read": score}, "HEAD"))
    assert any(p.startswith("check/dev: scripts/mp_approx_l13.py") for p in F.code_problems({**good, "check/dev": {L8.SCRIPT_REL: a}}, "HEAD"))
    assert any("is in a score job's record" in p for p in F.code_problems({**good, "score/dev/shard_1of4.json": fit}, "HEAD"))
    assert any("is in a score job's record" in p for p in F.code_problems({**good, "meta/dev": fit}, "HEAD"))
    assert any("2 different" in p for p in F.code_problems({**good, "read": {**fit, F.SCRIPT_REL: "b" * 64}}, "HEAD"))
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "c" * 64)
    assert any("not the file at" in p for p in F.code_problems(good, "HEAD"))


# ── the stages end to end on synthetic sidecars ──────────────────────────────


def test_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16, "x1": 10, "x2": 10, "x3": 10}, n_dev=30)
    cfg = tmp_path / "mp_approx_l13.yaml"
    shutil.copyfile(P.CONFIG, cfg)   # the file stage appends its run record to a copy
    monkeypatch.setattr(F, "CONFIG", cfg)
    monkeypatch.setattr(F, "RECORD", root / "record.json")
    monkeypatch.setattr(F, "DOC", tmp_path / "MP_APPROX_L13.md")
    monkeypatch.setattr(F, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    for fit in F.FITS:
        F.stage_fit(decl, fit, 0, _quiet)
    said = []
    F.stage_fit(decl, "TW-1x", 0, said.append)   # a unit already written is skipped on a restart
    assert any("not refitted" in s for s in said)
    F.stage_repeat(decl, _quiet)
    rd = F.stage_read(decl, _quiet)
    assert rd["primary"] == "TW-1x-b1d" and rd["reading"] == rd["arms"]["TW-1x-b1d"]["band"] and rd["reading"] in F.BANDS
    assert rd["readable_metrics"] and all(v["point"] is not None for v in rd["contrasts"].values())   # the toy is read, not skipped
    assert set(rd["arms"]) == set(F.READ_ARMS) | set(F.REFERENCES) and set(rd["contrasts"]) == set(F.CONTRASTS)
    for c_name, (a, b) in F.CONTRASTS.items():
        assert abs(rd["contrasts"][c_name]["point"] - (rd["arms"][a]["rho_bar"]["point"] - rd["arms"][b]["rho_bar"]["point"])) < 1e-12
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"] and set(rd["flags"]) <= set(F.FLAGS)
    assert [x for x in F.INTERPRETATION if x in rd["interpretation"]] == rd["interpretation"]
    an = rd["anchors"]
    assert set(an) == set(decl["quantities"]["anchors"]) and set(rd["strata"]) == {"hop=2", "hop=3"}
    assert set(an["level9_anchors"]) == set(an["carve_metrics"]) == {"dev", *P.L12_CARVES}
    checks = {c: json.loads((root / "carves" / c / "check.json").read_text(encoding="utf-8")) for c in P.L12_CARVES}
    tr = an["training_rows"]
    for fit, carves in F.FITS.items():
        assert tr[fit]["fit"]["total"] == sum(checks[c]["rows_with_gold_in_pool"]["all"] for c in carves)
        assert tr[fit]["inner"]["total"] == tr[fit]["inner"]["select"] == checks["select"]["rows_with_gold_in_pool"]["all"]
        assert tr[fit]["fit"]["eval"] == tr[fit]["fit"]["select"] == tr[fit]["inner"]["eval"] == 0
    assert an["unseen_sequences"]["TW-4x"]["unseen"] <= an["unseen_sequences"]["TW-1x"]["unseen"]
    for a, c_name in zip(F.GAP_SPLIT_ARMS, ("ceiling_gap", "ceiling_gap_4x")):
        g = an["gap_split"][a]
        assert g["right_pairs"] + g["wrong_pairs"] == len(ids["eval"])
        if g["right"] is not None:   # the split adds up to the ceiling gap
            assert abs(g["right"]["all"] + g["wrong"]["all"] - rd["contrasts"][c_name]["point"]) < 1e-9
    assert set(an["beta_choices"]) == set(an["grid_edges"]) == set(F.READ_ARMS)
    assert all(sum(v.values()) == 1 for v in an["beta_choices"].values())
    assert set(an["b1d_moved"]) == set(F.FITS) and all(v["changed_scored_mean"] > 0 for v in an["b1d_moved"].values())
    n_dev = len(ids["eval"])
    for fit in F.FITS:
        npz, js = F.unit_paths(fit, 0)
        with np.load(npz) as z:
            assert np.array_equal(z["q"], np.arange(n_dev)) and z["metrics_b1d"].shape == z["metrics_dsh"].shape
        flog = json.loads(js.read_text(encoding="utf-8"))
        assert (flog["l13_fit"], flog["l12_fit"], flog["fit"], flog["parts"], flog["k"], flog["fold"]) == (
            fit, fit, "NB-set", list(F12.part_names(fit)), 0, 0)
        assert flog["scored_queries"] == n_dev and len(flog["rounds"]) == 4 and flog["deterministic_algorithms"] is True
        assert {F.SCRIPT_REL, F12.SCRIPT_REL, F11.SCRIPT_REL, F10.SCRIPT_REL, P.SCRIPT_REL} <= set(flog["module_sha256"])
    x3 = root / "carves" / "x3" / "check.json"
    saved = x3.read_bytes()
    x3.write_text(json.dumps({**json.loads(saved), "note": "changed after the read"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was not made from"):
        F.stage_doc(_quiet)
    x3.write_bytes(saved)
    F.stage_doc(_quiet)
    body = F.DOC.read_text(encoding="utf-8")
    for phrase in ("TW-1x-b1d", "label-matched", "It is never a deployable or selected model", "reaches_gnn",
                   "It is also not the within-U_q oracle rho", "says nothing about hop 1"):
        assert phrase in body, phrase
    assert not any(p in body.lower() for p in FORBIDDEN)
    rec = json.loads(F.RECORD.read_text(encoding="utf-8"))
    assert rec["phase"] == "MP_APPROX_L13" and rec["sources_sha256"]["read.json"] == L0.sha256_file(root / "metaqa" / "read.json")
    F.stage_file("2026_10_01", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l13_2026_10_01"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert run["primary"] == "TW-1x-b1d" and set(run["contrasts"]) == set(F.CONTRASTS) and set(run["training_rows"]) == set(F.FITS)
    assert run["train_split_rows_scored"] == 0 and run["held_rows_read"] is False and set(run["b1d_moved"]) == set(F.FITS)
    with pytest.raises(SystemExit, match="exists"):
        F.stage_file("2026_10_01", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_the_stages_refuse_the_wrong_machine_and_wrong_arguments(stops):
    for argv in (["--stage", "fit", "--fit", "TW-1x", "--k", "0"], ["--stage", "repeat"], ["--stage", "read"],
                 ["--stage", "doc", "--host"], ["--stage", "file", "--host"], ["--stage", "fit", "--host"],
                 ["--stage", "fit", "--host", "--fit", "TW-4x"], ["--stage", "fit", "--host", "--k", "0"],
                 ["--stage", "repeat", "--host", "--fit", "TW-1x"], ["--stage", "read", "--host", "--k", "0"],
                 ["--stage", "fit", "--host", "--fit", "TW-2x", "--k", "0"], ["--stage", "fit", "--host", "--fit", "TW-8x", "--k", "0"],
                 ["--stage", "fit", "--host", "--fit", "TW-1x", "--k", "3"],
                 ["--stage", "doc", "--date", "2026_10_01"], ["--stage", "read", "--host", "--commit", "abc"], ["--stage", "file"],
                 ["--stage", "score", "--host"], ["--stage", "check", "--host"]):
        with pytest.raises(SystemExit):
            F.main(argv)
    assert not (stops / "hard_stops.json").exists()
    assert math.isinf(F.BETAS[-1])
