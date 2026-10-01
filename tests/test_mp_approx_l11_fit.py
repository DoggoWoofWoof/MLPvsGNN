"""MP-Approx level 11, fit module (configs/mp_approx_l11.yaml#tests): the declared constants and the training sidecars'
pins; NB-set's unit against level 10's fit_unit bit for bit, with the captured log-probabilities the ones level 10's
score_fold received, and dsh at beta = 1 against level 10's dens; the tempered posterior and the dsh choice; a merged
view of one sidecar against that sidecar's own view; a merged view of synthetic sidecars (every added row in the fit or
inner set by its inner flag, none scored, per-query reads the part's) and its refusals; the scored fold's golds never
reaching its scores; the bands, the training-rows anchor and the code check; the check, fit, repeat, read, doc and file
stages end to end on synthetic sidecars; and the stages' machine."""

from __future__ import annotations

import copy
import json
import math
import shutil
import sys
from pathlib import Path

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
import mp_approx_l11 as P  # noqa: E402
import mp_approx_l11_fit as F  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar and relation text)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)

_quiet = T8._quiet
EXTRA_QTYPE = "director_to_movie"   # named by a training sidecar's qtype list, asked by none of its queries


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, P):
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


def _view(tmp_path, monkeypatch, n_q: int, seed: int, family: str = "nb"):
    d = tmp_path / f"s{seed}"
    T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    return L9.View(d, family)


def _training_sidecar(d: Path, monkeypatch, tag: str, n_q: int, seed: int, reorder: bool = False) -> list[str]:
    """Level 9's combined toy sidecar with its ids renamed under the tag, so that they meet no other sidecar's, its folds
    and inner flags recomputed by level 0's rules from the new ids, and, with reorder, its qtype list reversed with one
    unasked name added. Every changed array's sha256 goes into its meta.json."""
    T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    qids = [q.replace(":toy:", f":{tag}:") for q in json.loads((d / "qids.json").read_text(encoding="utf-8"))]
    (d / "qids.json").write_text(json.dumps(qids), encoding="utf-8")
    np.save(d / "q_fold.npy", np.asarray([L0.fold_of(q) for q in qids], dtype=np.int64))
    np.save(d / "q_inner.npy", np.asarray([int(L0.is_inner(q)) for q in qids], dtype=np.int64))
    if reorder:
        old = meta["qtypes"]
        new = old[::-1] + [EXTRA_QTYPE]
        qt = np.load(d / "q_qtype.npy")
        np.save(d / "q_qtype.npy", np.asarray([new.index(old[int(i)]) for i in qt], dtype=np.int64))
        meta["qtypes"] = new
    for key in ("q_fold", "q_inner", "q_qtype"):
        meta["arrays_sha256"][f"{key}.npy"] = L0.sha256_file(d / f"{key}.npy")
    meta["qids_sha256"] = L0.sha256_file(d / "qids.json")
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return qids


def _training(tmp_path, monkeypatch, n_q: int = 40) -> tuple[Path, Path]:
    d9, d10 = tmp_path / "t9" / "metaqa", tmp_path / "t10" / "metaqa"
    _training_sidecar(d9, monkeypatch, "l9toy", n_q, seed=11)
    _training_sidecar(d10, monkeypatch, "l10toy", n_q, seed=12, reorder=True)
    return d9, d10


def _pin(d: Path) -> dict:
    return {"path": (d / "meta.json").as_posix(), "sha256": L0.sha256_file(d / "meta.json")}


def _decl(d9: Path, d10: Path) -> dict:
    decl = P.load_declaration()
    decl["inputs"]["training_sidecars"] = {"level9": _pin(d9), "level10": _pin(d10)}
    return decl


def _hard_stops(stops: Path) -> str:
    return (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the declared constants and the training sidecars' pins ──────────────────


def test_the_declared_constants_are_the_files():
    decl = P.load_declaration()
    arms = decl["arms"]
    assert set(arms["fits"]) == set(F.FITS) and F.FITS["NB-set"] == ("own",)
    assert F.FITS["NB-setX"] == ("own",) + tuple(decl["inputs"]["training_sidecars"])
    assert {k: v["path"] for k, v in decl["inputs"]["training_sidecars"].items()} == {
        "level9": "outputs/mp_approx_l9/metaqa/meta.json", "level10": "outputs/mp_approx_l10/metaqa/meta.json"}
    assert {k: tuple(v) for k, v in arms["read_arms"].items()} == F.READ_ARMS
    assert set(arms["references"]) == set(F.REFERENCES) and set(F.SCORES) == set(arms["scores"])
    assert decl["readings"]["primary"] == F.PRIMARY
    assert {k: tuple(s.strip() for s in v.split(" - ")) for k, v in decl["statistics"]["contrasts"].items()} == F.CONTRASTS
    assert set(decl["readings"]["bands"]) == {"L11_ABOVE_GNN", "L11_HIGH", "L11_LOW", "L11_MID", "NOT_READ"}
    assert set(decl["readings"]["flags"]) == {"CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE", "THETA_AT_CLIP"}
    mapped = {x for pair in F.INTERPRET_CONTRAST.values() for x in pair if x}
    assert set(decl["readings"]["interpretation_map"]) == {"l11_above_gnn", "l11_high", "l11_mid", "l11_low", "chain_not_identified"} | mapped
    selection = " ".join(arms["selection"].split())
    assert list(F.BETAS) == [1, 1.5, 2, 3, 4, 8, math.inf] and "beta in {1, 1.5, 2, 3, 4, 8, infinity}" in selection
    assert list(F.KAPPAS) == [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16] and list(F.ETAS) == [0.01, 0.1, 1, 10, 100]
    assert "{1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16}" in selection and "{0.01, 0.1, 1, 10, 100}" in selection
    assert "Ties go to the smaller beta, then the smaller kappa, then the larger eta" in selection
    assert F.EM_ROUNDS == 10 and "10 rounds after round 0" in " ".join(decl["em"]["e_step_m_step"].split())
    assert F.LEVEL10_FIT == "NB-set" and "level 10's fit_unit for NB-set" in arms["procedure"]
    assert F.REPEAT_UNIT == ("NB-setX", 0, 0) and "(NB-setX, k = 0, fold 0)" in decl["cross_fitting"]["repeat"]
    assert set(F.GAP_SPLIT_ARMS) == {"NB-setX-dsh", "NB-set-cov"} and "For NB-setX-dsh and NB-set-cov" in decl["quantities"]["anchors"]["gap_split"]
    assert F.SCRIPT_REL in decl["outputs"]["scripts"] and P.SCRIPT_REL in decl["outputs"]["scripts"]
    assert (F.CONFIG, F.OUT, F.DATA, F.NAME) == (P.CONFIG, P.OUT, P.DATA, P.NAME) and F.DOC == ROOT / decl["outputs"]["document"]


def test_the_training_sidecar_pins_verify_and_a_changed_pin_stops(stops, monkeypatch):
    monkeypatch.setattr(P, "verify_inputs", lambda decl: None)   # the population module's pins are its own tests'
    decl = P.load_declaration()
    if not all((ROOT / pin["path"]).exists() for pin in decl["inputs"]["training_sidecars"].values()):
        pytest.skip("the training sidecars' meta.json files are not on this machine")
    F.verify_inputs(decl)
    assert not (stops / "hard_stops.json").exists()
    decl["inputs"]["training_sidecars"]["level10"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.verify_inputs(decl)
    assert "outputs/mp_approx_l10/metaqa/meta.json" in _hard_stops(stops)


# ── level 10's unit, the capture and the sharpened mixture ──────────────────


def test_nb_sets_unit_is_level_10s_unit_bit_for_bit_and_the_capture_is_what_score_fold_received(tmp_path, monkeypatch, stops, one_thread):
    torch.use_deterministic_algorithms(True)
    view = _view(tmp_path, monkeypatch, n_q=160, seed=3)
    monkeypatch.setattr(L8, "M_EPOCHS", 3)
    monkeypatch.setattr(L8, "LR", 3e-2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 4)
    rel = T8._rel()
    a10, f10 = F10.fit_unit(F10.make_fitter(view, rel, "NB-set"), "NB-set", 0, 0, _quiet)
    received, original = [], F10.score_fold

    def recorder(fx, score, k, lp_inner, lp_score, inner_q, score_q):
        received.append((score, [np.array(x) for x in lp_inner], [np.array(x) for x in lp_score], np.array(inner_q), np.array(score_q)))
        return original(fx, score, k, lp_inner, lp_score, inner_q, score_q)

    monkeypatch.setattr(F10, "score_fold", recorder)
    a11, f11 = F.run_unit(F10.make_fitter(view, rel, "NB-set"), "NB-set", 0, 0, _quiet)
    assert F10.score_fold is recorder   # the wrapping is undone after the unit
    assert f10["kept_round"] > 0        # a fitted round is kept, so the reloaded state matters
    for key in a10:
        assert a11[key].dtype == a10[key].dtype and np.array_equal(a11[key], a10[key]), key
    assert set(a11) - set(a10) == {"metrics_dsh", "score_dsh"}
    assert a11["score_dsh"].shape == a11["score_dens"].shape and a11["metrics_dsh"].shape == a11["metrics_dens"].shape
    assert {key: v for key, v in f11.items() if key in f10 and key != "timing"} == {key: v for key, v in f10.items() if key != "timing"}
    assert set(f11) - set(f10) == {"l11_fit", "sources", "rows_by_source", "beta_dsh", "kappa_dsh", "eta_dsh", "inner_mean3_dsh",
                                   "grid_inner_mean3_dsh"}
    assert f11["rows_by_source"] == {"fit": {"own": f10["fit_queries"]}, "inner": {"own": f10["inner_queries"]}}
    assert len(f11["grid_inner_mean3_dsh"]) == len(F.BETAS) * 50 and f11["beta_dsh"] in [F.beta_label(b) for b in F.BETAS]
    assert [r[0] for r in received] == list(F10.SCORES)
    received.clear()
    _a, _f, cap = F.fit_and_capture(F10.make_fitter(view, rel, "NB-set"), 0, 0, _quiet)
    assert len(received) == 2
    for _score, lp_inner, lp_score, inner_q, score_q in received:   # the capture is what each call received
        assert np.array_equal(inner_q, cap["inner_q"]) and np.array_equal(score_q, cap["score_q"])
        assert len(lp_inner) == len(cap["lp_inner"]) and all(np.array_equal(x, y) for x, y in zip(lp_inner, cap["lp_inner"]))
        assert len(lp_score) == len(cap["lp_score"]) and all(np.array_equal(x, y) for x, y in zip(lp_score, cap["lp_score"]))
    monkeypatch.setattr(F, "BETAS", (1.0,))   # beta = 1 alone: dsh is level 10's dens, choice, grid and scores
    a1, f1 = F.run_unit(F10.make_fitter(view, rel, "NB-set"), "NB-set", 0, 0, _quiet)
    assert np.array_equal(a1["score_dsh"], a1["score_dens"]) and np.array_equal(a1["metrics_dsh"], a1["metrics_dens"])
    assert (f1["beta_dsh"], f1["kappa_dsh"], f1["eta_dsh"]) == ("1.0", f1["kappa_dens"], f1["eta_dens"])
    assert {key.split("|", 1)[1]: v for key, v in f1["grid_inner_mean3_dsh"].items()} == f1["grid_inner_mean3_dens"]
    assert f1["inner_mean3_dsh"] == f1["grid_inner_mean3_dens"][f"{f1['kappa_dens']}|{f1['eta_dens']}"]


def test_the_capture_stops_on_different_log_probabilities_or_a_missing_call_and_restores_score_fold(monkeypatch, stops):
    monkeypatch.setattr(F10, "score_fold", lambda fx, score, k, lp_inner, lp_score, inner_q, score_q: {"score": score})
    stub = F10.score_fold
    lp, q = [np.log(np.full(3, 1 / 3))], np.array([4])

    def unit_calling(*lps):   # level 10's unit reduced to its score_fold calls, one per log-probability list given
        def unit(fx, fit, k, fold, log=print):
            for score, lp_inner in zip(F10.SCORES, lps):
                F10.score_fold(fx, score, k, lp_inner, lp, q, q)
            return {"q": q}, {"fit": fit}
        return unit

    monkeypatch.setattr(F10, "fit_unit", unit_calling(lp, lp))
    arrays, flog, cap = F.fit_and_capture(None, 0, 0, _quiet)
    assert flog == {"fit": "NB-set"} and np.array_equal(cap["inner_q"], q) and np.array_equal(cap["lp_inner"][0], lp[0])
    assert F10.score_fold is stub
    monkeypatch.setattr(F10, "fit_unit", unit_calling(lp, [x - 1.0 for x in lp]))
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.fit_and_capture(None, 0, 0, _quiet)
    assert "did not receive the same log-probabilities" in _hard_stops(stops) and F10.score_fold is stub
    monkeypatch.setattr(F10, "fit_unit", unit_calling(lp))
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.fit_and_capture(None, 0, 0, _quiet)
    assert "was not called once per score" in _hard_stops(stops) and F10.score_fold is stub


def test_the_tempered_posterior_is_p_to_the_beta_renormalised_with_its_two_ends():
    rng = np.random.default_rng(0)
    for T in (1, 4, 9):
        logit = rng.normal(size=T + 1)
        lp = logit - np.log(np.exp(logit).sum())
        p = np.exp(lp)
        assert F.tempered(lp, 1.0) is lp
        for beta in (1.5, 2.0, 3.0, 4.0, 8.0):
            got = np.exp(F.tempered(lp, beta))
            assert np.allclose(got, p ** beta / (p ** beta).sum(), rtol=1e-12, atol=1e-15) and abs(got.sum() - 1.0) < 1e-12
        top = F.tempered(lp, math.inf)
        assert np.exp(top).sum() == 1.0 and top[int(np.argmax(p))] == 0.0
    assert F.tempered(np.log([0.1, 0.4, 0.4, 0.1]), math.inf).tolist() == [-math.inf, 0.0, -math.inf, -math.inf]   # the first largest p
    assert F.tempered(np.log([0.2, 0.4, 0.4]), math.inf).tolist() == [-math.inf, 0.0, -math.inf]   # a type before the null type
    sharp = np.exp(F.tempered(np.log([0.5, 0.3, 0.2]), 8.0))
    assert sharp[0] > 0.98 and np.all(np.diff(sharp) < 0)


def test_the_dsh_choice_follows_the_grid_and_the_tie_order():
    K, E = F.KAPPAS, F.ETAS
    flat = {F.grid_key(b, k, e): 0.5 for b in F.BETAS for k in K for e in E}
    assert len(flat) == 7 * 10 * 5
    assert F.choose(flat) == (1.0, K[0], E[-1])   # all tied: the smallest beta, the smallest kappa, the largest eta
    best = {**flat, F.grid_key(3.0, K[6], E[2]): 0.6, F.grid_key(math.inf, K[-1], E[0]): 0.6}
    assert F.choose(best) == (3.0, K[6], E[2])
    order = {**flat, F.grid_key(2.0, K[7], E[3]): 0.7, F.grid_key(2.0, K[7], E[4]): 0.7, F.grid_key(2.0, K[8], E[4]): 0.7}
    assert F.choose(order) == (2.0, K[7], E[4])
    nan = {**flat, F.grid_key(1.0, K[0], E[-1]): float("nan")}
    assert F.choose(nan) == (1.0, K[0], E[-2])   # an undefined grid point is never chosen
    assert F.choose({**flat, F.grid_key(math.inf, K[0], E[0]): 0.9}) == (math.inf, K[0], E[0])
    assert F.grid_key(math.inf, 1.0, 0.01) == "inf|1.0|0.01" and F.beta_label(1.5) == "1.5" and F.beta_label(1.0) == "1.0"


# ── the merged view ──────────────────────────────────────────────────────────


def test_a_merged_view_of_one_sidecar_gives_that_sidecars_own_unit_bit_for_bit(tmp_path, monkeypatch, stops, one_thread):
    torch.use_deterministic_algorithms(True)
    view = _view(tmp_path, monkeypatch, n_q=90, seed=6)
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    rel = T8._rel()
    one = F.MultiView([view], ("own",))
    assert np.array_equal(one.type_ptr, view.type_ptr) and one.meta["qtypes"] == view.meta["qtypes"]
    a, fa = F.run_unit(F10.make_fitter(view, rel, "NB-set"), "NB-set", 0, 2, _quiet)
    b, fb = F.run_unit(F10.make_fitter(one, rel, "NB-set"), "NB-set", 0, 2, _quiet)
    assert set(a) == set(b) and all(a[key].dtype == b[key].dtype and np.array_equal(a[key], b[key]) for key in a)
    assert {key: v for key, v in fa.items() if key != "timing"} == {key: v for key, v in fb.items() if key != "timing"}


def test_a_merged_view_trains_on_every_added_row_by_its_inner_flag_never_scores_one_and_reads_each_from_its_part(tmp_path, monkeypatch, stops, one_thread):
    own = _view(tmp_path, monkeypatch, n_q=40, seed=3)
    d9, d10 = _training(tmp_path, monkeypatch)
    parts = [L9.View(d9, "nb"), L9.View(d10, "nb")]
    mv = F.MultiView([own, *parts], F.FITS["NB-setX"])
    assert mv.n_q == own.n_q + parts[0].n_q + parts[1].n_q and mv.family == "nb"
    assert mv.meta["qtypes"] == own.meta["qtypes"] + [EXTRA_QTYPE] and parts[1].meta["qtypes"] != own.meta["qtypes"]
    assert np.array_equal(mv.q_fold[:own.n_q], own.q_fold) and (mv.q_fold[own.n_q:] == -1).all()
    added = np.flatnonzero(mv.q_part > 0)
    gold = added[mv.q_gold_in_pool[added] > 0]
    inner = mv.q_inner[gold].astype(bool)
    assert gold.size == added.size and inner.any() and (~inner).any()
    for fold in range(L8.FOLDS):
        fit_q, inner_q, score_q = L8.unit_queries(mv, fold)
        own_fit, own_inner, own_score = L8.unit_queries(own, fold)
        assert np.array_equal(score_q, own_score) and not np.isin(added, score_q).any()
        assert np.isin(gold[inner], inner_q).all() and np.isin(gold[~inner], fit_q).all()
        assert np.array_equal(fit_q[fit_q < own.n_q], own_fit) and np.array_equal(inner_q[inner_q < own.n_q], own_inner)
    views = (own, *parts)
    for q in range(mv.n_q):
        v, i = views[int(mv.q_part[q])], int(mv.q_local[q])
        assert all(np.array_equal(x, y) for x, y in zip(mv.entries(q), v.entries(i)))
        assert np.array_equal(mv.gold_local(q), v.gold_local(i)) and np.array_equal(mv.z(q, 1), v.z(i, 1))
        assert np.array_equal(mv.t_code[mv.type_rows(q)], v.t_code[v.type_rows(i)])
        assert np.array_equal(mv.t_size[mv.type_rows(q)], v.t_size[v.type_rows(i)])
        assert np.array_equal(mv.t_gold[mv.type_rows(q)], v.t_gold[v.type_rows(i)])
        assert all(np.array_equal(mv.reach(q, int(c)), v.reach(i, int(c))) for c in v.t_code[v.type_rows(i)])
        assert mv.meta["qtypes"][mv.q_qtype[q]] == v.meta["qtypes"][v.q_qtype[i]] and np.array_equal(mv.q_emb[q], v.q_emb[i])
        for key in ("q_hop", "q_inner", "q_pool_size", "q_gold_total", "q_gold_in_pool", "q_types", "q_entries"):
            assert getattr(mv, key)[q] == getattr(v, key)[i], key
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    arrays, flog = F.run_unit(F10.make_fitter(mv, T8._rel(), "NB-set"), "NB-setX", 0, 1, _quiet)
    rows = flog["rows_by_source"]
    assert np.array_equal(arrays["q"], np.flatnonzero(own.q_fold == 1)) and flog["sources"] == list(F.FITS["NB-setX"])
    assert (sum(rows["fit"].values()), sum(rows["inner"].values())) == (flog["fit_queries"], flog["inner_queries"])
    for i, src in enumerate(("level9", "level10")):
        assert (rows["fit"][src], rows["inner"][src]) == (int((~parts[i].q_inner.astype(bool)).sum()), int(parts[i].q_inner.astype(bool).sum()))
    with pytest.raises(SystemExit, match="not the fit's"):
        F.run_unit(F10.make_fitter(mv, T8._rel(), "NB-set"), "NB-set", 0, 1, _quiet)   # NB-set never reads added rows


def test_the_merged_view_refuses_shared_ids_std_views_broken_pins_smoke_and_altered_sidecars_and_misaligned_parts(tmp_path, monkeypatch, stops):
    own = _view(tmp_path, monkeypatch, n_q=30, seed=8)
    d9, d10 = _training(tmp_path, monkeypatch, n_q=20)
    decl = _decl(d9, d10)
    mv = F.merged_view(decl, own)
    assert mv.sources == F.FITS["NB-setX"] and [v.dir for v in mv.parts] == [own.dir, d9, d10]
    with pytest.raises(SystemExit, match="nb views only"):
        F.MultiView([L9.View(own.dir, "std")], ("own",))
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.MultiView([own, L9.View(own.dir, "nb")], ("own", "level9"))
    assert "a query id is in two parts" in _hard_stops(stops)
    cases = []
    bad = copy.deepcopy(decl)
    bad["inputs"]["training_sidecars"]["level9"]["sha256"] = "0" * 64
    cases.append((bad, "is not its pinned sha256"))
    bad = copy.deepcopy(decl)
    bad["inputs"]["training_sidecars"]["level9"] = _pin(own.dir)
    cases.append((bad, "a level9 training row is in this file's population"))
    bad = copy.deepcopy(decl)
    bad["inputs"]["training_sidecars"]["level10"] = _pin(d9)
    cases.append((bad, "share a query id"))
    smoke = tmp_path / "smoke" / "metaqa"
    shutil.copytree(d10, smoke)
    meta = json.loads((smoke / "meta.json").read_text(encoding="utf-8"))
    (smoke / "meta.json").write_text(json.dumps({**meta, "limit": 5}), encoding="utf-8")
    bad = copy.deepcopy(decl)
    bad["inputs"]["training_sidecars"]["level10"] = _pin(smoke)
    cases.append((bad, "is a smoke run"))
    altered = tmp_path / "altered" / "metaqa"
    shutil.copytree(d9, altered)
    np.save(altered / "q_hop.npy", np.load(altered / "q_hop.npy") + 1)
    bad = copy.deepcopy(decl)
    bad["inputs"]["training_sidecars"]["level9"] = _pin(altered)
    cases.append((bad, "training sidecar level9"))
    for bad, phrase in cases:
        with pytest.raises(SystemExit, match="HARD STOP"):
            F.merged_view(bad, own)
        assert phrase in _hard_stops(stops), phrase
    shifted = L9.View(d9, "nb")
    shifted.type_ptr = shifted.type_ptr + 1
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.MultiView([own, shifted], ("own", "level9"))
    assert "do not line up with its parts" in _hard_stops(stops)


def test_the_scored_folds_golds_never_reach_its_scores(tmp_path, monkeypatch, stops, one_thread):
    """arms: each arm's inputs are z(T_k), q, the relation text and the view's types and counts. The scored fold's golds
    are read only by its metrics, under every score and with the added rows."""
    own = _view(tmp_path, monkeypatch, n_q=60, seed=7)
    d9, d10 = _training(tmp_path, monkeypatch)
    parts = [L9.View(d9, "nb"), L9.View(d10, "nb")]
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    rel = T8._rel()
    first, f1 = F.run_unit(F10.make_fitter(F.MultiView([own, *parts], F.FITS["NB-setX"]), rel, "NB-set"), "NB-setX", 0, 0, _quiet)
    other = L9.View(own.dir, "nb")
    is_gold, t_gold = np.array(other.is_gold, copy=True), np.array(other.t_gold, copy=True)
    rng = np.random.default_rng(0)
    for q in np.flatnonzero(other.q_fold == 0):
        sl = other.nodes(q)
        is_gold[sl] = rng.permutation(is_gold[sl])   # the same count, other nodes
        rows = other.type_rows(q)
        t_gold[rows] = rng.integers(0, 3, size=t_gold[rows].size)
    other.is_gold, other.t_gold = is_gold, t_gold
    again, f2 = F.run_unit(F10.make_fitter(F.MultiView([other, *parts], F.FITS["NB-setX"]), rel, "NB-set"), "NB-setX", 0, 0, _quiet)
    for key in ("q", "score_dens", "score_cov", "score_dsh", "score_ptr", "argmax"):
        assert np.array_equal(first[key], again[key]), key
    assert [f1[key] for key in ("beta_dsh", "kappa_dsh", "eta_dsh", "kappa_cov", "eta_cov")] == [
        f2[key] for key in ("beta_dsh", "kappa_dsh", "eta_dsh", "kappa_cov", "eta_cov")]
    assert not np.array_equal(first["metrics_dsh"], again["metrics_dsh"])   # the metrics do read them


# ── the statistics and the readings ──────────────────────────────────────────


def test_the_bands_come_in_the_declared_order_and_read_arm_carries_them():
    assert F.band(1.3, [1.05, 1.5], True) == "L11_ABOVE_GNN"
    assert F.band(0.8, [0.6, 1.0], True) == "L11_HIGH"
    assert F.band(0.8, [0.4, 1.2], True) == "L11_MID"
    assert F.band(0.2, [0.0, 0.45], True) == "L11_LOW"
    assert F.band(0.2, [0.0, 0.55], True) == "L11_MID"
    assert F.band(None, None, False) == "NOT_READ" and F.band(0.9, [0.8, 1.0], False) == "NOT_READ"
    rng = np.random.default_rng(0)
    T = rng.random((40, 3, 3))
    G = T + 0.5
    W = L0.boot_weights(40)
    _d, dens, readable = L8.denominators(T, G, W)
    assert F.read_arm(T + 0.45, T, G, dens, readable, W)["band"] == "L11_HIGH"
    assert F.read_arm(T + 0.6, T, G, dens, readable, W)["band"] == "L11_ABOVE_GNN"
    assert F.read_arm(T + 0.05, T, G, dens, readable, W)["band"] == "L11_LOW"


def test_the_training_rows_anchor_counts_each_source_per_unit():
    units = {"k0_f0": {"rows_by_source": {"fit": {"own": 10, "level9": 30, "level10": 31}, "inner": {"own": 2, "level9": 3, "level10": 4}}},
             "k0_f1": {"rows_by_source": {"fit": {"own": 12, "level9": 30, "level10": 31}, "inner": {"own": 1, "level9": 3, "level10": 4}}}}
    a = F.training_rows_anchor(units)
    assert a["fit"]["own"] == {"mean": 11.0, "min": 10, "max": 12} and a["fit"]["level9"] == {"mean": 30.0, "min": 30, "max": 30}
    assert a["fit"]["total"] == {"mean": 72.0, "min": 71, "max": 73} and a["inner"]["total"] == {"mean": 8.5, "min": 8, "max": 9}


def test_the_identical_code_check_needs_every_script_in_one_committed_set(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    a = "a" * 64
    score = {L8.SCRIPT_REL: a, L9.SCRIPT_REL: a, L10.SCRIPT_REL: a, P.SCRIPT_REL: a}
    fit = {**score, F10.SCRIPT_REL: a, F.SCRIPT_REL: a}
    good = {"score/shard_0": score, "check": score, "meta": score, "fit/NB-setX/k0_f0": fit, "repeat": fit, "read": fit}
    assert F.code_problems(good, "HEAD") == []
    assert F.code_problems({**good, "meta": {}}, "HEAD") == []   # a toy sidecar's meta records no module
    assert any("are not in every fit" in p for p in F.code_problems({**good, "repeat": {**score, F.SCRIPT_REL: a}}, "HEAD"))
    assert any("are not in every fit" in p for p in F.code_problems({**good, "read": {**score, F10.SCRIPT_REL: a}}, "HEAD"))
    assert any(p.startswith("check: scripts/mp_approx_l11.py") for p in F.code_problems({**good, "check": {L8.SCRIPT_REL: a}}, "HEAD"))
    assert any("2 different" in p for p in F.code_problems({**good, "read": {**fit, F.SCRIPT_REL: "b" * 64}}, "HEAD"))
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "c" * 64)
    assert any("not the file at" in p for p in F.code_problems(good, "HEAD"))


# ── the stages end to end on synthetic sidecars ──────────────────────────────


def _stage_env(tmp_path, monkeypatch, n_q, seed):
    d = tmp_path / "l11" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    cfg = tmp_path / "mp_approx_l11.yaml"
    shutil.copyfile(P.CONFIG, cfg)   # the file stage appends its run record to a copy
    for mod in (P, F):
        monkeypatch.setattr(mod, "DATA", d)
    monkeypatch.setattr(P, "verify_inputs", lambda decl: None)   # the population module's pins are its own tests'
    monkeypatch.setattr(F, "CONFIG", cfg)
    monkeypatch.setattr(F, "RECORD", tmp_path / "l11" / "record.json")
    monkeypatch.setattr(F, "DOC", tmp_path / "MP_APPROX_L11.md")
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "fit_process", _light_fit_process)
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    return d, cfg


def test_check_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    d, cfg = _stage_env(tmp_path, monkeypatch, n_q=90, seed=4)
    d9, d10 = _training(tmp_path, monkeypatch)
    monkeypatch.setattr(F, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    decl = _decl(d9, d10)
    with pytest.raises(SystemExit, match="check stage has not passed"):
        F.stage_fit(decl, "NB-set", 0, None, _quiet)
    check = P.stage_check(decl, _quiet)
    assert all(check["families"][f]["direction_check"]["passes"] for f in P.FAMILIES)
    F.stage_fit(decl, "NB-set", 0, None, _quiet)
    for fold in range(L8.FOLDS):
        F.stage_fit(decl, "NB-setX", 0, fold, _quiet)
    F.stage_fit(decl, "NB-setX", 0, None, _quiet)   # every unit is logged: a restart skips them all
    F.stage_repeat(decl, _quiet)
    rd = F.stage_read(decl, _quiet)
    assert rd["primary"] == "NB-setX-dsh" and rd["reading"] == rd["arms"]["NB-setX-dsh"]["band"]
    assert set(rd["arms"]) == set(F.READ_ARMS) | set(F.REFERENCES) and set(rd["contrasts"]) == set(F.CONTRASTS)
    for c_name, (a, b) in F.CONTRASTS.items():
        point = rd["contrasts"][c_name]["point"]
        assert point is None or abs(point - (rd["arms"][a]["rho_bar"]["point"] - rd["arms"][b]["rho_bar"]["point"])) < 1e-12
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"]
    an = rd["anchors"]
    assert set(rd["strata"]) == {"hop=1", "hop=2", "hop=3"} and set(an["agreement"]) == set(F.FITS) == set(an["theta"])
    assert set(an["grid_edges"]) == set(F.READ_ARMS) and set(an["gap_split"]) == set(F.GAP_SPLIT_ARMS) and "gold_unreached_nb" in an["check"]
    assert set(an["beta_choices"]) == {"NB-setX-dsh", "NB-set-dsh"} and all(sum(v.values()) == 5 for v in an["beta_choices"].values())
    tr = an["training_rows"]
    assert set(tr["NB-set"]["fit"]) == {"own", "total"} and set(tr["NB-setX"]["fit"]) == {"own", "level9", "level10", "total"}
    assert tr["NB-setX"]["fit"]["own"] == tr["NB-set"]["fit"]["own"] and tr["NB-setX"]["inner"]["own"] == tr["NB-set"]["inner"]["own"]
    for src, dd in (("level9", d9), ("level10", d10)):   # every added row trains every unit
        assert tr["NB-setX"]["fit"][src]["mean"] + tr["NB-setX"]["inner"][src]["mean"] == len(json.loads((dd / "qids.json").read_text(encoding="utf-8")))
    base = L8.Data(d, check=False)
    for fit in F.FITS:   # every query exactly once, out of fold, under all three scores
        for fold in range(L8.FOLDS):
            with np.load(d / "units" / fit / f"k0_f{fold}.npz") as z:
                assert np.array_equal(z["q"], np.flatnonzero(base.q_fold == fold))
                assert z["metrics_dsh"].shape == z["metrics_cov"].shape and z["score_dsh"].shape == z["score_cov"].shape
            flog = json.loads((d / "units" / fit / f"k0_f{fold}.json").read_text(encoding="utf-8"))
            assert (flog["l11_fit"], flog["fit"], flog["sources"]) == (fit, "NB-set", list(F.FITS[fit])) and len(flog["rounds"]) == 4
            assert flog["deterministic_algorithms"] is True and {F.SCRIPT_REL, F10.SCRIPT_REL, P.SCRIPT_REL} <= set(flog["module_sha256"])
    F.stage_doc(_quiet)
    body = F.DOC.read_text(encoding="utf-8")
    assert "NB-setX-dsh" in body and "NB-set-cov" in body and "It is not the within-U_q oracle rho" in body
    assert "gold-labelled queries" in body and "beyond the declared list" in body
    assert not any(p in body.lower() for p in ("message passing is unnecessary", "we do not need message passing", "the mlp wins"))
    rec = json.loads(F.RECORD.read_text(encoding="utf-8"))
    assert rec["sources_sha256"]["read.json"] == L0.sha256_file(d / "read.json") and rec["phase"] == "MP_APPROX_L11"
    F.stage_file("2026_10_01", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l11_2026_10_01"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert set(run["contrasts"]) == set(F.CONTRASTS) and set(run["fit_queries_per_unit_mean"]) == set(F.FITS)
    with pytest.raises(SystemExit, match="exists"):
        F.stage_file("2026_10_01", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_the_stages_refuse_the_wrong_machine_and_wrong_arguments(stops):
    for argv in (["--stage", "fit", "--fit", "NB-set", "--k", "0"], ["--stage", "repeat"], ["--stage", "read"],
                 ["--stage", "doc", "--host"], ["--stage", "file", "--host"], ["--stage", "fit", "--host"],
                 ["--stage", "fit", "--host", "--fit", "NB-setX"], ["--stage", "repeat", "--host", "--fit", "NB-set"],
                 ["--stage", "read", "--host", "--k", "0"], ["--stage", "repeat", "--host", "--fold", "0"],
                 ["--stage", "fit", "--host", "--fit", "XX", "--k", "0"], ["--stage", "fit", "--host", "--fit", "NB-set", "--k", "7"],
                 ["--stage", "fit", "--host", "--fit", "NB-setX", "--k", "0", "--fold", "5"], ["--stage", "score", "--host"]):
        with pytest.raises(SystemExit):
            F.main(argv)
