"""MP-Approx level 15, fit module (configs/mp_approx_l15.yaml#tests): the declared constants; FZ, the design look's, with
the declared checks added; FZ's log-probabilities on toy tables against the formula of arms.norms.fz (both buckets,
partners, absent and present types, an empty row), summing to one with finite gradients, and level 8's softmax when V
is empty; the FZ model's initial state against level 9's ResidualModel's; FZ's vocabulary (the fit set's types only,
unchanged when the inner and scored rows' types change) and partners, and its hard stops; the posterior-sum check;
the eval part's refusals (no check, a failed direction check, a changed meta.json or array, a smoke, ids other than
r2's pin); an sm unit against level 14's run_unit, bit for bit, and an FZ unit (its log, its sums, its hard stops and
level 9's make_model restored); the scored rows' golds, the other seeds' twins and the stored metrics never reaching an
sm or FZ unit's scores; the bands, rho, reaches_gnn, the where strata and the interpretation map (per hop included) on
toy cases; the code check; the fit, repeat, read, doc and file stages end to end on synthetic sidecars; and the stages'
machine."""

from __future__ import annotations

import copy
import inspect
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
import mp_approx_l12 as P12  # noqa: E402
import mp_approx_l12_fit as F12  # noqa: E402
import mp_approx_l13 as P13  # noqa: E402
import mp_approx_l13_fit as F13  # noqa: E402
import mp_approx_l14 as P14  # noqa: E402
import mp_approx_l14_fit as F14  # noqa: E402
import mp_approx_l15 as P  # noqa: E402
import mp_approx_l15_fit as F  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar and relation text)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)
import test_mp_approx_l12 as T12  # noqa: E402  (the population module's id rewrite)
import test_mp_approx_l12_fit as T12F  # noqa: E402  (level 12's synthetic carves, with level 12's checks run)
import test_mp_approx_l14 as T14  # noqa: E402  (an assembled carve's stand-in)
import test_mp_approx_l14_fit as T14F  # noqa: E402  (level 14's toy walk programme without some bucket-0 nb types)
import test_mp_approx_l15 as T15  # noqa: E402  (the earlier levels' rows on file)
from mp_retrieval import m3b_pools  # noqa: E402

_quiet = T8._quiet
TBL = L8.TB ** L8.MAX_L
CHOICES = ("beta_dsh", "kappa_dsh", "eta_dsh", "kappa_cov", "eta_cov")
B1D_CHOICES = ("beta_b1d", "kappa_b1d", "eta_b1d")
FORBIDDEN = ("message passing is unnecessary", "we do not need message passing", "the mlp wins")
DESIGN_LOOK = ROOT / "outputs" / "mp_approx_l14_diag" / "diag_fz.py"
DESIGN_LOOK_SHA256 = "699858d32152da05a81752708f2c87d667d7024b0f840e0099060bb568c816f2"


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, P12, P13, P14, P):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])   # the fit stages turn it on


def _stopped(stops: Path) -> str:
    return (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── synthetic sidecars: level 12's carves and r2, checked ────────────────────


def _own_process(path: Path) -> None:
    """A check record as its own host process files it: this test's process has imported the fit module, which a host
    check job never imports (placement.identical_code)."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["module_sha256"] = {rel: sha for rel, sha in rec["module_sha256"].items() if rel != F.SCRIPT_REL}
    path.write_text(json.dumps(rec), encoding="utf-8")


def _env(tmp_path, monkeypatch, carves: dict, n_r: int = 24, seed: int = 5, every: int | None = None):
    """Level 12's synthetic carves (its _env: level 12's paths pointed at them and level 12's checks run), assembled
    stand-ins for level 12's other carves (r2's disjointness check reads their ids), r2's sidecar (level 9's combined toy
    sidecar under train-split ids, naming carve r2 and its digest), this file's paths pointed at them, the carve pins and
    r2's pin set to them, and the population module's check run on r2."""
    if not T15._levels_on_file(P.load_declaration()):
        pytest.skip("the earlier levels' qids.json files are not on this machine")
    root, decl12, ids = T12F._env(tmp_path, monkeypatch, carves, n_dev=12)
    for c in P12.CARVES:
        if c not in carves:
            T14._fake_carve(root / "carves" / c, [f"metaqa:{1 + i % 3}hop:train:z{c}_{i}" for i in range(5)])
    d = tmp_path / "l15" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both if every is None else T14F._nb_b0_dropped(every), n_q=n_r, seed=seed)
    r2 = [q.replace(":toy:", ":train:r2_") for q in qids]   # not level 12's toy carve ids, which r2's check refuses to share
    T12._rewrite_ids(d, r2, carve="r2", carve_ids_sha256=m3b_pools.ids_digest(r2))
    for mod in (P, F):
        monkeypatch.setattr(mod, "DATA", d)
    monkeypatch.setattr(F, "CARVES_DIR", root / "carves")
    monkeypatch.setattr(P, "verify_inputs", lambda decl: None)   # the population module's pins are its own tests'
    decl = P.load_declaration()
    for c in carves:
        decl["carves"]["pins"][c] = decl12["carves"]["pins"][c]
    decl["carves"]["pins"]["r2"] = {"queries": len(r2), "zero_gold_excluded": 0, "ids_sha256": m3b_pools.ids_digest(r2),
                                    "hops": P.hop_counts(r2)}
    P.stage_check(decl, _quiet)
    _own_process(d / "check.json")
    ids["eval"] = r2
    return root, decl, ids


def _fitter(decl, norm="sm", view="full", fit="TW-1x"):
    return F.make_fitter(norm, F.deploy_view(decl, view, fit), T8._rel())


def _light(monkeypatch, rounds: int = 2) -> None:
    torch.use_deterministic_algorithms(True)
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", rounds)


# ── the declared constants ───────────────────────────────────────────────────


def test_the_declared_constants_are_the_files():
    decl = P.load_declaration()
    assert {k: tuple(v) for k, v in decl["carves"]["fits"].items()} == F.FITS and list(decl["carves"]["fits"]) == list(F.FITS)
    assert F.FITS == F14.FITS and all(F12.FITS[fit] == carves for fit, carves in F.FITS.items())
    assert set(P.L12_CARVES) == {"select", *(c for cs in F.FITS.values() for c in cs)}
    assert {k: tuple(v) for k, v in decl["units"]["roles"].items()} == F12.ROLES
    arms = decl["arms"]
    assert tuple(arms["norms"]) == F.NORMS == ("sm", "fz") and tuple(arms["views"]) == F.VIEWS == F14.VIEWS == ("full", "b0")
    assert {k: tuple(v) for k, v in arms["read_arms"].items()} == F.READ_ARMS and list(arms["read_arms"]) == list(F.READ_ARMS)
    assert tuple(arms["references"]) == F.REFERENCES and tuple(arms["scores"]) == tuple(F.SCORES)
    assert F.SCORES == F14.SCORES and F.UNIT_SCORES == F14.UNIT_SCORES and F.UNIT_KEYS == F14.UNIT_KEYS
    assert decl["readings"]["primary"] == F.PRIMARY == "FZ-TW-1x-b1d" and F.READ_ARMS[F.PRIMARY] == ("fz", "TW-1x", "b1d")
    contrasts = decl["statistics"]["contrasts"]
    assert {k: tuple(s.strip() for s in v.split(" - ")) for k, v in contrasts.items()} == F.CONTRASTS and list(contrasts) == list(F.CONTRASTS)
    assert " ".join(decl["statistics"]["hop_contrasts"].split()).startswith("fz_adds, fz_adds_b0 and over_l14_primary per hop (1, 2, 3)")
    assert decl["statistics"]["where_contrasts"].startswith("fz_adds and fz_adds_b0 per where stratum")
    assert F.HOP_CONTRASTS == ("fz_adds", "fz_adds_b0", "over_l14_primary") and F.WHERE_CONTRASTS == ("fz_adds", "fz_adds_b0")
    strata = " ".join(decl["quantities"]["strata"].split())
    assert F.HOPS == (1, 2, 3) and "Per hop (1, 2, 3)" in strata and all(w in strata for w in F.WHERE)
    assert F.WHERE == ("true_from_b0", "true_from_b1_only", "true_nowhere")
    assert set(decl["readings"]["bands"]) == set(F.BANDS) and set(decl["readings"]["flags"]) == set(F.FLAGS)
    assert tuple(decl["readings"]["interpretation_map"]) == F.INTERPRETATION
    assert F.REPEAT_UNIT == ("fz", "full", "TW-1x", 0)
    assert "The unit (fz, full, TW-1x, k = 0), which the primary reads, runs again" in " ".join(decl["units"]["repeat"].split())
    assert ("For FZ-TW-1x-b1d, TW-1x-b1d and FZ-TW-4x-b1d against NB-oracle, and for FZ-TW-1x-b0 and TW-1x-b0 against "
            "NB-oracle-b0" in " ".join(decl["quantities"]["anchors"]["gap_split"].split()))
    assert F.GAP_SPLIT == {"FZ-TW-1x-b1d": "NB-oracle", "TW-1x-b1d": "NB-oracle", "FZ-TW-4x-b1d": "NB-oracle",
                           "FZ-TW-1x-b0": "NB-oracle-b0", "TW-1x-b0": "NB-oracle-b0"}
    assert set(decl["quantities"]["anchors"]) >= {"fz_vocabulary", "null_mode", "b0_view", "b1d_moved", "exchangeability", "reaches_gnn"}
    assert "NB-oracle's rho_bar" in decl["readings"]["flags"]["CEILING_LOW"]
    per_fit = " ".join(decl["units"]["per_fit"].split())
    assert F.FOLD == 0 and tuple(F.SEEDS) == (0, 1, 2) and len(F.NORMS) * len(F.VIEWS) * len(F.FITS) * len(F.SEEDS) == 24
    assert ("One unit per (norm, view, fit, k) at fold 0, for the norms sm and fz, the views full and b0, the fits TW-1x and "
            "TW-4x, and k = 0, 1 and 2: 24 units" in per_fit)
    assert "units/<norm>/<view>/<fit>/k<k>_f0" in decl["units"]["unit_files"]
    assert F.unit_paths("fz", "b0", "TW-4x", 2) == (F.DATA / "units" / "fz" / "b0" / "TW-4x" / "k2_f0.npz",
                                                    F.DATA / "units" / "fz" / "b0" / "TW-4x" / "k2_f0.json")
    assert F.SCRIPT_REL in decl["outputs"]["scripts"] and P.SCRIPT_REL in decl["outputs"]["scripts"]
    assert F.DOC == ROOT / decl["outputs"]["document"] and F.RECORD == F.OUT / "record.json"
    assert "outputs/mp_approx_l15/record.json" in decl["outputs"]["record"]
    assert (F.CONFIG, F.OUT, F.DATA, F.NAME) == (P.CONFIG, P.OUT, P.DATA, P.NAME)
    assert F.CARVES_DIR == F12.CARVES_DIR == ROOT / decl["inputs"]["level12_carves"]["dir"]
    assert (F.BETAS, F.KAPPAS, F.ETAS) == (F11.BETAS, F10.KAPPAS, F10.ETAS) and F11.LEVEL10_FIT == "NB-set"
    assert "A unit chooses on its own view and norm" in " ".join(arms["selection"].split())
    assert F.TBL == L8.TB ** 3 and L8.MAX_L == 3 and F.SUM_TOL == 1e-4
    assert "(within 1e-4)" in " ".join(" ".join(decl["hard_stops"]).split())
    assert set(F.FZ_KEYS) == {"fz_vocab", "fz_partners", "fz_scored_rows", "fz_scored_entries_outside_v", "fz_scored_rows_outside_v"}
    text = " ".join(decl["placement"]["reservations"].split())
    for phrase in ("full TW-1x 1.15 CPUs and 2.8 GB", "full TW-4x 1.3 CPUs and 3.35 GB", "b0 TW-1x 1.1 CPUs and 1.9 GB",
                   "b0 TW-4x 1.2 CPUs and 2.25 GB", "The FZ units reserve the design look's: full TW-1x 1.3 CPUs and 2.8 GB",
                   "full TW-4x 1.5 CPUs and 3.4 GB", "b0 TW-1x 1.15 CPUs and 1.95 GB", "b0 TW-4x 1.3 CPUs and 2.3 GB",
                   "The repeat reserves its unit's figures", "The read reserves 1 CPU and 1.6 GB", "the fit module's 0.9 CPU and 0.45 GB"):
        assert phrase in text, phrase


def test_fz_is_the_design_looks_with_the_declared_checks_added():
    """arms.norms.fz: the design look's FZ, copied; its log-probabilities are the look's method, unchanged. The design
    look is untracked, so the comparison runs where it is; the pin in the declaration's text is checked everywhere."""
    assert f"diag_fz.py ({DESIGN_LOOK_SHA256})" in " ".join(P.CONFIG.read_text(encoding="utf-8").split())
    assert F.FZModel.__mro__[1] is L9.ResidualModel and F.FZSetFitter.__mro__[1] is F10.SetFitter
    assert not set(vars(F.FZModel)) & {"__init__", "forward", "weights", "compose"}   # level 9's, inherited
    if not DESIGN_LOOK.exists():
        pytest.skip("the design look is untracked and is not on this machine")
    assert L0.sha256_file(DESIGN_LOOK) == DESIGN_LOOK_SHA256
    assert inspect.getsource(F.FZModel.log_probs_fz) in DESIGN_LOOK.read_text(encoding="utf-8")


# ── FZ on toy tables ─────────────────────────────────────────────────────────


def _reference(w: np.ndarray, null: float, present: np.ndarray, vocab: np.ndarray, partner: np.ndarray) -> tuple[dict, float]:
    """arms.norms.fz by its formula, for one row: p at each present type (by table index) and p(null)."""
    U = vocab | present
    ew, en = np.exp(np.asarray(w, dtype=np.float64)), math.exp(float(null))
    Z = en + ew[U].sum()
    out = {}
    for i in np.flatnonzero(present):
        same = [int(i)] + ([int(partner[i])] if partner[i] >= 0 else [])
        out[int(i)] = sum(ew[j] for j in same if U[j]) / Z * ew[i] / sum(ew[j] for j in same if present[j])
    seq_present = present | np.asarray([partner[i] >= 0 and bool(present[partner[i]]) for i in range(w.size)], dtype=bool)
    return out, float((en + ew[U & ~seq_present].sum()) / Z)


def _batch(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, int]:
    tmax = max(max((len(r) for r in rows), default=1), 1)
    idx = np.zeros((len(rows), tmax), dtype=np.int64)
    mask = np.zeros((len(rows), tmax), dtype=bool)
    for i, r in enumerate(rows):
        idx[i, :len(r)], mask[i, :len(r)] = r, True
    return torch.as_tensor(idx), torch.as_tensor(mask), tmax


def _toy_fz(tmp_path, monkeypatch):
    """An FZ fitter on a toy deploy view, and an FZ model on its table with every parameter moved off its start."""
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=14)
    fx = _fitter(decl, "fz")
    torch.manual_seed(3)
    model = F.fz_make_model_for(fx)("NB-hyb", fx.rel_emb, fx.table)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.5 * torch.randn_like(p))
    return decl, fx, model


def test_fz_log_probs_on_toy_tables_equal_the_formula_sum_to_one_and_have_finite_gradients(tmp_path, monkeypatch, stops, one_thread):
    _decl, fx, model = _toy_fz(tmp_path, monkeypatch)
    n = int(fx.table.codes.size)
    partner = np.asarray(fx.fz_partner)
    pairs = [(i, int(p)) for i, p in enumerate(partner) if p > i]
    lonely = [i for i in range(n) if partner[i] < 0]
    assert len(pairs) >= 2 and len(lonely) >= 3   # the toy table holds sequences in both buckets, and sequences in one
    (i1, p1), (i2, p2) = pairs[0], pairs[1]
    vocab = np.zeros(n, dtype=bool)
    vocab[[i1, p2, lonely[0], lonely[1]]] = True
    rows = [[i1, p1],               # both buckets present
            [i2],                   # its partner absent and in V: the sequence's mass comes to it
            [p1],                   # its partner absent and in V
            [i1],                   # its partner absent and outside V: outside U
            [],                     # an empty row: the null type holds everything
            [lonely[2], p1, i2]]    # a present type outside V, beside V's absent types whose sequence the row lacks
    rng = np.random.default_rng(1)
    rows += [sorted(rng.choice(n, size=int(rng.integers(0, min(n, 7))), replace=False).tolist()) for _ in range(14)]
    idx, mask, tmax = _batch(rows)
    qemb = fx.qemb[:len(rows)]
    for vocab_now, partner_now in ((vocab, partner), (rng.random(n) < 0.5, partner), (vocab, np.full(n, -1))):
        model.fz_vocab, model.fz_partner = torch.as_tensor(vocab_now), torch.as_tensor(partner_now)
        with torch.no_grad():
            lp = model.log_probs_fz(qemb, idx, mask).numpy().astype(np.float64)
            w_all, null = model.weights(qemb, torch.arange(n)[None, :].expand(len(rows), n))
        assert lp.shape == (len(rows), tmax + 1) and np.allclose(np.exp(lp).sum(1), 1.0, atol=1e-5)
        for r, row in enumerate(rows):
            present = np.zeros(n, dtype=bool)
            present[row] = True
            want, want_null = _reference(w_all[r].numpy(), float(null[r]), present, vocab_now, partner_now)
            got = {t: math.exp(lp[r, j]) for j, t in enumerate(row)}
            assert set(got) == set(want) and all(abs(got[t] - want[t]) <= 1e-5 + 1e-4 * want[t] for t in want), r
            assert abs(math.exp(lp[r, tmax]) - want_null) <= 1e-5 + 1e-4 * want_null, r
            assert np.all(np.isneginf(lp[r, len(row):tmax]))   # padding
        assert lp[4, tmax] == 0.0   # the empty row
    model.fz_vocab, model.fz_partner = torch.as_tensor(vocab), torch.as_tensor(partner)
    model.zero_grad()
    q = qemb.clone().requires_grad_(True)
    lp = model.log_probs_fz(q, idx, mask)
    keep = torch.cat([mask, torch.ones(len(rows), 1, dtype=torch.bool)], 1)
    (-torch.where(keep, lp, torch.zeros_like(lp)).sum()).backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None] + [q.grad]
    assert grads and all(torch.isfinite(g).all() for g in grads) and any(g.abs().sum() > 0 for g in grads)
    assert not (stops / "hard_stops.json").exists()


def test_with_v_empty_fz_is_level_8s_softmax_and_on_one_bucket_a_softmax_over_u(tmp_path, monkeypatch, stops, one_thread):
    _decl, fx, model = _toy_fz(tmp_path, monkeypatch)
    n = int(fx.table.codes.size)
    rng = np.random.default_rng(2)
    rows = [sorted(rng.choice(n, size=int(rng.integers(0, min(n, 8))), replace=False).tolist()) for _ in range(16)] + [[]]
    idx, mask, tmax = _batch(rows)
    qemb = fx.qemb[:len(rows)]
    with torch.no_grad():
        w, null = model.weights(qemb, idx)
        soft = L8.Fitter.log_probs(w, null, mask).numpy()
        for partner in (np.asarray(fx.fz_partner), np.full(n, -1)):
            model.fz_vocab, model.fz_partner = torch.zeros(n, dtype=torch.bool), torch.as_tensor(partner)
            lp = model.log_probs_fz(qemb, idx, mask).numpy()
            assert np.array_equal(np.isneginf(lp), np.isneginf(soft)) and np.allclose(lp[np.isfinite(soft)], soft[np.isfinite(soft)], atol=2e-6)
        vocab = rng.random(n) < 0.4   # one bucket (no partner): a softmax over U, the absent types' mass on the null type
        model.fz_vocab, model.fz_partner = torch.as_tensor(vocab), torch.full((n,), -1, dtype=torch.int64)
        lp = model.log_probs_fz(qemb, idx, mask).numpy().astype(np.float64)
        w_all, nul = model.weights(qemb, torch.arange(n)[None, :].expand(len(rows), n))
    for r, row in enumerate(rows):
        u = vocab.copy()
        u[row] = True
        z = np.logaddexp.reduce(np.r_[w_all[r].numpy()[u], float(nul[r])].astype(np.float64))
        assert np.allclose(lp[r, :len(row)], w_all[r].numpy()[row] - z, atol=2e-5)
        absent = vocab.copy()
        absent[row] = False
        assert abs(lp[r, tmax] - (np.logaddexp.reduce(np.r_[w_all[r].numpy()[absent], float(nul[r])].astype(np.float64)) - z)) < 2e-5
    assert not (stops / "hard_stops.json").exists()


def test_the_fz_models_initial_state_is_level_9s_residual_models_for_the_same_seed(tmp_path, monkeypatch, stops, one_thread):
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=14)
    fx = _fitter(decl, "fz")
    torch.manual_seed(11)
    ref = L9.make_model("NB-hyb", fx.rel_emb, fx.table)
    after_ref = torch.rand(4)
    torch.manual_seed(11)
    fz = F.fz_make_model_for(fx)("NB-hyb", fx.rel_emb, fx.table)
    after_fz = torch.rand(4)
    assert type(ref) is L9.ResidualModel and type(fz) is F.FZModel and torch.equal(after_ref, after_fz)
    a, b = ref.state_dict(), fz.state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a) and not any(k.startswith("fz_") for k in b)
    assert [n for n, _p in ref.named_parameters()] == [n for n, _p in fz.named_parameters()]
    assert np.array_equal(fz.fz_vocab.numpy(), fx.fz_vocab) and np.array_equal(fz.fz_partner.numpy(), fx.fz_partner)
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.fz_make_model_for(fx)("NB-text", fx.rel_emb, fx.table)
    assert "an FZ unit's model is not level 9's NB-hyb" in _stopped(stops)
    (stops / "hard_stops.json").unlink()
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.fz_make_model_for(fx)("NB-hyb", fx.rel_emb, L8.TypeTable(fx.data.t_code))
    assert "not made on its fitter's type table" in _stopped(stops)
    (stops / "hard_stops.json").unlink()
    with pytest.raises(SystemExit, match="HARD STOP"):
        fx.logp(ref, np.arange(4))   # a softmax model in an FZ fitter
    assert "is not level 9's NB-hyb under FZ" in _stopped(stops)


def test_fzs_vocabulary_is_the_fit_sets_types_only_and_each_partner_the_same_sequence_in_the_other_bucket(tmp_path, monkeypatch, stops,
                                                                                                         one_thread):
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=14)
    for view in F.VIEWS:
        fx = _fitter(decl, "fz", view)
        data = fx.data
        codes = np.asarray(fx.table.codes, dtype=np.int64)
        fit_q, inner_q, score_q = L8.unit_queries(data, 0)
        owner = np.repeat(np.arange(data.n_q), data.q_types)
        on_fit = np.isin(owner, fit_q)
        assert set(codes[fx.fz_vocab].tolist()) == set(np.asarray(data.t_code)[on_fit].tolist())
        assert fit_q.size and inner_q.size and np.array_equal(score_q, np.arange(len(ids["eval"])))
        for i, p in enumerate(fx.fz_partner):
            other = (1 - codes[i] // TBL) * TBL + codes[i] % TBL
            if p >= 0:
                assert codes[p] == other
            else:
                assert other not in set(codes.tolist())
        assert (view == "b0") == (not (fx.fz_partner >= 0).any())   # one bucket: no partner
        # the inner and scored rows' types replaced by other types: V is unchanged
        t_code = np.asarray(data.t_code, dtype=np.int64).copy()
        moved = ~on_fit
        t_code[moved] = (t_code[moved] % TBL) + TBL * (1 - t_code[moved] // TBL) if view == "full" else t_code[moved][::-1]
        stand_in = SimpleNamespace(**{k: getattr(data, k) for k in ("n_q", "q_types", "q_fold", "q_inner", "q_gold_in_pool")}, t_code=t_code)
        table = L8.TypeTable(t_code)
        vocab, partner = F.fz_tables(SimpleNamespace(data=stand_in, table=table, gidx=table.index(t_code)))
        assert set(np.asarray(table.codes)[vocab].tolist()) == set(codes[fx.fz_vocab].tolist())
        tc = np.asarray(table.codes, dtype=np.int64)
        assert all(tc[p] == (1 - tc[i] // TBL) * TBL + tc[i] % TBL for i, p in enumerate(partner) if p >= 0)
    assert not (stops / "hard_stops.json").exists()
    bad = SimpleNamespace(data=fx.data, table=fx.table, gidx=np.zeros_like(fx.gidx))   # every row read as table type 0
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.fz_tables(bad)
    assert "vocabulary holds a type of no fit-set row or misses one" in _stopped(stops)
    (stops / "hard_stops.json").unlink()
    three = SimpleNamespace(data=fx.data, table=SimpleNamespace(codes=np.r_[np.asarray(fx.table.codes), 2 * TBL + 5]), gidx=fx.gidx)
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.fz_tables(three)
    assert "not under buckets 0 and 1" in _stopped(stops)


def test_the_sum_check_passes_a_posterior_that_sums_to_one_and_stops_on_one_that_does_not(stops):
    with np.errstate(divide="ignore"):
        F.check_sums(np.log(np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])), "toy")
        F.check_sums(np.log(np.array([[0.2, 0.3, 0.5 + 5e-5]])), "toy")   # within 1e-4
        for bad in ([[0.2, 0.3, 0.5 + 2e-4]], [[0.2, 0.3, np.nan]], [[0.0, 0.0, 0.0]]):
            (stops / "hard_stops.json").unlink(missing_ok=True)
            with pytest.raises(SystemExit, match="HARD STOP"):
                F.check_sums(np.log(np.array(bad)), "toy")
            assert "does not sum to one (within 1e-4)" in _stopped(stops)


# ── the eval part ────────────────────────────────────────────────────────────


def test_the_eval_part_refuses_no_check_a_failed_direction_check_a_changed_meta_json_or_array_a_smoke_and_ids_not_r2s(
        tmp_path, monkeypatch, stops, one_thread):
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=14)
    d = F.DATA
    v = F.eval_view(decl)
    assert list(v.qids) == ids["eval"] and type(v) is L9.View and v.family == "nb"
    check_p, meta_p = d / "check.json", d / "meta.json"
    check, meta = check_p.read_text(encoding="utf-8"), meta_p.read_text(encoding="utf-8")
    ck = json.loads(check)
    aside = d / "check.json.aside"
    check_p.rename(aside)
    with pytest.raises(SystemExit, match="r2's check has not passed"):
        F.eval_view(decl)
    aside.rename(check_p)
    nb = ck["families"]["nb"]
    failed = {**ck, "families": {**ck["families"], "nb": {**nb, "direction_check": {**nb["direction_check"], "passes": False}}}}
    for bad in ({**ck, "stage": "carve_check"}, {**ck, "carve": "r"}, failed):
        check_p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit, match="r2's check has not passed"):
            F.eval_view(decl)
    check_p.write_text(check, encoding="utf-8")

    def stops_with(text: str, dcl=decl) -> None:
        (stops / "hard_stops.json").unlink(missing_ok=True)
        with pytest.raises(SystemExit, match="HARD STOP"):
            F.eval_view(dcl)
        assert text in _stopped(stops), text

    def with_meta(**extra) -> None:
        """meta.json changed and the check's record of it set to match, so that the later refusals are reached."""
        meta_p.write_text(json.dumps({**json.loads(meta), **extra}), encoding="utf-8")
        check_p.write_text(json.dumps({**ck, "meta_sha256": L0.sha256_file(meta_p)}), encoding="utf-8")

    meta_p.write_text(json.dumps({**json.loads(meta), "note": "changed after the check"}), encoding="utf-8")
    stops_with("part eval's meta.json is not the one r2's check read")
    with_meta(limit=5)
    stops_with("part eval is a smoke run")
    for extra in ({"carve": "r"}, {"carve_ids_sha256": "0" * 64}):
        with_meta(**extra)
        stops_with("part eval is not what r2's pin records")
    with_meta()
    assert list(F.eval_view(decl).qids) == ids["eval"]
    for key, val in (("ids_sha256", "0" * 64), ("queries", 99), ("hops", {1: 0, 2: 0, 3: len(ids["eval"])})):
        bad = copy.deepcopy(decl)
        bad["carves"]["pins"]["r2"][key] = val
        stops_with("part eval is not what r2's pin records", bad)
    bad = copy.deepcopy(decl)
    bad["carves"]["pins"]["r2"] = copy.deepcopy(decl["carves"]["pins"]["r"])   # level 14's r is not r2
    stops_with("part eval is not what r2's pin records", bad)
    raw = (d / "t_gold.npy").read_bytes()
    np.save(d / "t_gold.npy", np.load(d / "t_gold.npy") + 1)   # an array that is not the one meta.json records
    stops_with("deploy view: part eval:")
    (d / "t_gold.npy").write_bytes(raw)
    assert list(F.eval_view(decl).qids) == ids["eval"]
    off = [q.replace(":train:", ":dev:") if i == 2 else q for i, q in enumerate(ids["eval"])]
    T12._rewrite_ids(d, off, carve="r2", carve_ids_sha256=m3b_pools.ids_digest(ids["eval"]))
    check_p.write_text(json.dumps({**ck, "meta_sha256": L0.sha256_file(meta_p)}), encoding="utf-8")
    stops_with("part eval is not what r2's pin records")   # a row that is not a train-split id


def test_the_deploy_views_are_level_14s_with_r2_as_the_eval_part(tmp_path, monkeypatch, stops, one_thread):
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=20, every=4)
    fv, dv = F.deploy_view(decl, "full", "TW-1x"), F.deploy_view(decl, "b0", "TW-1x")
    assert L9.View is F14.ORIGINAL_VIEW and all(isinstance(v, F14.B0View) for v in dv.parts) and not any(isinstance(v, F14.B0View) for v in fv.parts)
    assert tuple(fv.sources) == tuple(dv.sources) == F12.part_names("TW-1x") and list(fv.parts[0].qids) == ids["eval"]
    assert np.all(np.asarray(dv.t_code, dtype=np.int64) // TBL == 0) and np.any(np.asarray(fv.t_code, dtype=np.int64) // TBL == 1)
    for key in ("q_fold", "q_inner", "q_gold_in_pool", "q_qtype"):
        assert np.array_equal(getattr(dv, key), getattr(fv, key)), key
    with pytest.raises(SystemExit, match="not one of this file's views"):
        F.deploy_view(decl, "b1", "TW-1x")
    with pytest.raises(SystemExit, match="not one of this file's fits"):
        F.deploy_view(decl, "b0", "TW-2x")
    monkeypatch.setattr(F14, "B0View", L9.View)   # a filter that keeps every type
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.deploy_view(decl, "b0", "TW-1x")
    assert "a b0 view holds a type outside bucket 0" in _stopped(stops) and L9.View is F14.ORIGINAL_VIEW


# ── a unit ───────────────────────────────────────────────────────────────────


def test_an_sm_unit_is_level_14s_run_unit_bit_for_bit(tmp_path, monkeypatch, stops, one_thread):
    _light(monkeypatch)
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, every=4)
    for view in F.VIEWS:
        fx = _fitter(decl, "sm", view)
        assert type(fx) is F10.SetFitter
        out, flog = F.run_unit(fx, "sm", view, "TW-1x", 0, _quiet)
        ref, rlog = F14.run_unit(F10.make_fitter(F.deploy_view(decl, view, "TW-1x"), T8._rel(), F11.LEVEL10_FIT), view, "TW-1x", 0, _quiet)
        assert set(out) == set(ref)
        for key in ref:
            assert out[key].dtype == ref[key].dtype and np.array_equal(out[key], ref[key]), (view, key)
        assert set(flog) - set(rlog) == {"l15_norm"} and flog["l15_norm"] == "sm" and (flog["l14_view"], flog["l14_fit"]) == (view, "TW-1x")
        same = CHOICES + (B1D_CHOICES if view == "full" else ()) + ("kept_round", "kept_theta", "parameters", "walk_types", "fit_queries")
        assert [flog[key] for key in same] == [rlog[key] for key in same]
        assert np.array_equal(out["q"], np.arange(len(ids["eval"])))
    with pytest.raises(SystemExit, match="the fitter is not the norm's"):
        F.run_unit(_fitter(decl, "sm"), "fz", "full", "TW-1x", 0, _quiet)
    with pytest.raises(SystemExit, match="the fitter is not the norm's"):
        F.run_unit(_fitter(decl, "fz"), "sm", "full", "TW-1x", 0, _quiet)
    with pytest.raises(SystemExit, match="not one of this file's norms"):
        F.make_fitter("softmax", F.deploy_view(decl, "full", "TW-1x"), T8._rel())
    assert not (stops / "hard_stops.json").exists()


def test_an_fz_unit_fits_and_scores_under_fz_with_its_vocabulary_logged_and_level_9s_make_model_restored(tmp_path, monkeypatch, stops,
                                                                                                         one_thread):
    _light(monkeypatch)
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, every=4)
    original = L9.make_model
    n_r = len(ids["eval"])
    for view in F.VIEWS:
        sm, slog = F.run_unit(_fitter(decl, "sm", view), "sm", view, "TW-1x", 0, _quiet)
        fx = _fitter(decl, "fz", view)
        out, flog = F.run_unit(fx, "fz", view, "TW-1x", 0, _quiet)
        assert L9.make_model is original and set(out) == set(sm) and np.array_equal(out["q"], sm["q"])
        assert set(flog) - set(slog) == set(F.FZ_KEYS) and flog["l15_norm"] == "fz" and flog["fit"] == "NB-set"
        assert {k: flog[k] for k in F.FZ_KEYS} == F.fz_counts(fx) and flog["fz_vocab"] == int(fx.fz_vocab.sum()) > 0
        assert flog["fz_scored_rows"] == n_r and (flog["fz_partners"] == 0) == (view == "b0")
        assert flog["parameters"] == slog["parameters"] and flog["walk_types"] == slog["walk_types"]
        if view == "full":
            assert not np.array_equal(out["score_dsh"], sm["score_dsh"])   # FZ moves the posterior
        with F11.rebound(L9, make_model=F.fz_make_model_for(fx)):
            _arrays, _flog, cap = F11.fit_and_capture(fx, 0, 0, _quiet)
        for key, qkey in (("lp_score", "score_q"), ("lp_inner", "inner_q")):
            for q, lp in zip(cap[qkey], cap[key]):
                lp = np.asarray(lp)
                assert lp.size == int(fx.data.q_types[q]) + 1 and abs(np.exp(lp).sum() - 1) < 1e-5
                if int(fx.data.q_types[q]) == 0:
                    assert np.array_equal(lp, [0.0])   # a row without a type: the null type holds everything
        assert L9.make_model is original
    assert not (stops / "hard_stops.json").exists()
    monkeypatch.setattr(F, "SUM_TOL", -1.0)   # every sum fails: the check runs where the unit fits, selects and scores
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.run_unit(_fitter(decl, "fz"), "fz", "full", "TW-1x", 0, _quiet)
    assert "does not sum to one" in _stopped(stops) and L9.make_model is original


def test_the_scored_rows_golds_the_other_seeds_twins_and_the_stored_metrics_never_reach_a_units_scores(tmp_path, monkeypatch, stops, one_thread):
    """arms: each arm reads z(T_k), the query embedding, the relation text and the nb family's compiled types and counts,
    nothing else. r2's golds are read only by its metrics; FZ's vocabulary reads the fit rows' types only."""
    _light(monkeypatch)
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    names = F12.part_names("TW-1x")
    rng = np.random.default_rng(0)
    for norm in F.NORMS:
        for view in F.VIEWS:
            first, f1 = F.run_unit(_fitter(decl, norm, view), norm, view, "TW-1x", 0, _quiet)
            with F11.rebound(L9, View=F14.B0View if view == "b0" else F14.ORIGINAL_VIEW):
                parts = [F.eval_view(decl), *(F12.part_view(decl, p) for p in names[1:])]
            ev = parts[0]
            is_gold, t_gold = np.array(ev.is_gold, copy=True), np.array(ev.t_gold, copy=True)
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
            fx = F.make_fitter(norm, F12.with_roles(F11.MultiView(parts, names)), T8._rel())
            again, f2 = F.run_unit(fx, norm, view, "TW-1x", 0, _quiet)
            keys = ("q", "score_dens", "score_cov", "score_dsh", "score_ptr", "argmax") + (("score_b1d",) if view == "full" else ())
            for key in keys:
                assert np.array_equal(first[key], again[key]), (norm, view, key)
            choices = CHOICES + (B1D_CHOICES if view == "full" else ()) + (F.FZ_KEYS if norm == "fz" else ())
            assert [f1[key] for key in choices] == [f2[key] for key in choices] and f1["kept_round"] == f2["kept_round"]
            assert not np.array_equal(first["metrics_dsh"], again["metrics_dsh"])   # the metrics do read the golds
    assert not (stops / "hard_stops.json").exists()


# ── the statistics and the readings on toy cases ─────────────────────────────


def test_the_bands_come_in_the_declared_order_and_rho_is_the_share_of_the_gain():
    assert F.band(1.3, [1.05, 1.5], True) == "L15_ABOVE_GNN"
    assert F.band(0.8, [0.6, 1.0], True) == "L15_HIGH"
    assert F.band(0.8, [0.4, 1.2], True) == "L15_MID"
    assert F.band(0.2, [0.0, 0.45], True) == "L15_LOW"
    assert F.band(0.2, [0.0, 0.55], True) == "L15_MID"
    assert F.band(None, None, False) == "NOT_READ" and F.band(0.9, [0.8, 1.0], False) == "NOT_READ"
    rng = np.random.default_rng(0)
    T = rng.random((40, 3, 3))
    G = T + 0.5
    W = L0.boot_weights(40)
    _d, dens, readable = L8.denominators(T, G, W)
    assert readable == list(L8.RETRIEVAL)
    half = F.read_arm(T + 0.25, T, G, dens, readable, W)
    assert abs(half["rho_bar"]["point"] - 0.5) < 1e-12 and all(abs(half["rho"][m]["point"] - 0.5) < 1e-12 for m in L8.RETRIEVAL)
    assert half["band"] == "L15_MID" and all(half["gap_to_gnn"][m]["flag"] == "BELOW_GNN" for m in L8.RETRIEVAL)
    assert F.read_arm(T + 0.45, T, G, dens, readable, W)["band"] == "L15_HIGH"
    assert F.read_arm(T + 0.6, T, G, dens, readable, W)["band"] == "L15_ABOVE_GNN"
    assert F.read_arm(T + 0.05, T, G, dens, readable, W)["band"] == "L15_LOW"
    _d, dens0, none = L8.denominators(T, T, W)   # no gain: nothing is readable
    assert none == [] and F.read_arm(T + 0.1, T, T, dens0, none, W)["band"] == "NOT_READ"
    arms = {"a": F.read_arm(T + 0.25, T, G, dens, readable, W), "b": F.read_arm(T + 0.45, T, G, dens, readable, W)}
    c = F.paired(arms, "b", "a", readable)
    assert c["of"] == "rho_bar(b) - rho_bar(a)" and abs(c["point"] - 0.4) < 1e-12 and c["ci"][0] <= c["point"] <= c["ci"][1]
    assert F.paired(arms, "b", "a", []) == {"of": "rho_bar(b) - rho_bar(a)", "point": None, "ci": None}


def _arm(flags) -> dict:
    return {"gap_to_gnn": {m: {"flag": f} for m, f in zip(L8.RETRIEVAL, flags)}}


def test_reaches_gnn_is_the_first_read_arm_with_no_readable_metric_below_the_gnn_with_its_units_rows():
    tr = {f"{n}/{v}/{f}": {"fit": {"total": rows - (v == "b0") - 2 * (n == "fz")}, "inner": {"total": 10}}
          for n in F.NORMS for v in F.VIEWS for f, rows in (("TW-1x", 100), ("TW-4x", 400))}
    every = list(L8.RETRIEVAL)
    below = _arm(["BELOW_GNN"] * 3)
    arms = {a: below for a in F.READ_ARMS}
    arms["FZ-TW-1x-b0"] = _arm([None, "BEATS_GNN", None])
    assert F.reaches_gnn(arms, every, tr) == {"arm": "FZ-TW-1x-b0", "norm": "fz", "fit": "TW-1x", "view": "b0", "fit_rows": 97, "inner_rows": 10}
    arms["TW-4x-dsh"] = _arm([None] * 3)
    assert F.reaches_gnn(arms, every, tr) == {"arm": "TW-4x-dsh", "norm": "sm", "fit": "TW-4x", "view": "full", "fit_rows": 400, "inner_rows": 10}
    arms["TW-1x-b1d"] = _arm(["BELOW_GNN", None, None])
    assert F.reaches_gnn(arms, [L8.RETRIEVAL[1]], tr)["arm"] == "TW-1x-b1d" and F.reaches_gnn(arms, every, tr)["arm"] == "TW-4x-dsh"
    assert F.reaches_gnn({a: below for a in arms}, every, tr) is None and F.reaches_gnn(arms, [], tr) is None
    assert [F.unit_key(a) for a in F.READ_ARMS] == ["sm/full/TW-1x", "sm/full/TW-1x", "sm/b0/TW-1x", "sm/full/TW-4x", "sm/full/TW-4x",
                                                    "sm/b0/TW-4x", "fz/full/TW-1x", "fz/full/TW-1x", "fz/b0/TW-1x", "fz/full/TW-4x",
                                                    "fz/full/TW-4x", "fz/b0/TW-4x"]


def test_the_interpretation_lists_every_entry_that_applies_in_the_declared_order():
    every = list(L8.RETRIEVAL)
    flat = {c: {"ci": [-0.1, 0.1]} for c in F.CONTRASTS}
    hflat = {f"hop={h}": {c: {"ci": [-0.1, 0.1]} for c in F.HOP_CONTRASTS} for h in F.HOPS}
    up, down = [0.01, 0.1], [-0.2, -0.01]

    def ci(**kw):
        return {**flat, **{c: {"ci": v} for c, v in kw.items()}}

    def hci(**kw):   # hop1=[lo, hi]: the hop's fz_adds interval
        out = copy.deepcopy(hflat)
        for key, v in kw.items():
            out[f"hop={key[-1]}"]["fz_adds"]["ci"] = v
        return out

    cases = [
        (("L15_HIGH", _arm([None] * 3), every, flat, hflat, 0.9), ["l15_high", "matched_not_below_gnn"]),
        (("L15_MID", _arm(["BELOW_GNN", None, None]), every,
          ci(fz_adds=up, fz_adds_b0=down, fz_adds_dsh=down, fz_adds_4x=up, fz_adds_4x_b0=up, fz_adds_4x_dsh=down, over_l14_primary=up,
             b1d_over_b0_fz=down, b1d_over_b0_fz_4x=up, data_4x_fz=up, data_4x_fz_b0=up, bucket_ceiling=up, ceiling_gap=[0.1, 0.3]),
          hci(hop1=down, hop2=up, hop3=up), 0.4),
         ["l15_mid", "matched_below_gnn", "fz_adds", "fz_hurts_b0", "fz_hurts_dsh", "fz_adds_4x", "fz_adds_4x_b0", "fz_hurts_4x_dsh",
          "over_l14_primary", "b0_over_b1d_fz", "b1d_over_b0_fz_4x", "data_adds_4x_fz", "data_adds_4x_fz_b0", "b0_ceiling_binds",
          "chain_not_identified", "fz_hurts_hop1", "fz_adds_hop2", "fz_adds_hop3"]),
        (("L15_LOW", _arm([None] * 3), every,
          ci(fz_adds=down, fz_adds_b0=up, fz_adds_dsh=up, fz_adds_4x=down, fz_adds_4x_b0=down, fz_adds_4x_dsh=up, over_l14_primary=down,
             b1d_over_b0_fz=up, b1d_over_b0_fz_4x=down, data_4x_fz=down, ceiling_gap=[0.1, 0.3], ceiling_gap_b0=[0.1, 0.3]),
          hci(hop1=up, hop3=down), 0.6),
         ["l15_low", "matched_not_below_gnn", "fz_hurts", "fz_adds_b0", "fz_adds_dsh", "fz_hurts_4x", "fz_hurts_4x_b0", "fz_adds_4x_dsh",
          "under_l14_primary", "b1d_over_b0_fz", "b0_over_b1d_fz_4x", "fz_adds_hop1", "fz_hurts_hop3"]),
        (("L15_ABOVE_GNN", _arm(["BEATS_GNN"] * 3), every, ci(ceiling_gap_b0=[0.1, 0.3]), hflat, 0.3),   # ceiling_gap is not the b0 one
         ["l15_above_gnn", "matched_not_below_gnn"]),
        (("NOT_READ", _arm([None] * 3), [], {c: {"ci": None} for c in F.CONTRASTS},
          {h: {c: {"ci": None} for c in F.HOP_CONTRASTS} for h in hflat}, float("nan")), []),
    ]
    for args, want in cases:
        got = F.interpretation(*args)
        assert got == want and [x for x in F.INTERPRETATION if x in got] == got


def test_where_is_level_8s_chain_reach_of_the_rows_true_chain_on_the_nb_view(monkeypatch):
    reach = {0: (np.array([3]), np.array([], dtype=np.int64)), 1: (np.array([], dtype=np.int64), np.array([2, 4])),
             2: (np.array([], dtype=np.int64), np.array([], dtype=np.int64)), 3: (np.array([1]), np.array([1]))}
    seen = []

    def chain_reach(nb, q, steps):
        seen.append((q, steps))
        return reach[q]

    monkeypatch.setattr(L8, "true_chain", lambda qt: ("chain", qt))
    monkeypatch.setattr(L8, "chain_reach", chain_reach)
    nb = SimpleNamespace(n_q=4)
    base = SimpleNamespace(meta={"qtypes": ["a_to_b", "c_to_d"]}, q_qtype=np.array([0, 1, 1, 0]))
    assert F.where_of(nb, base).tolist() == [0, 1, 2, 0]
    assert seen == [(0, ("chain", "a_to_b")), (1, ("chain", "c_to_d")), (2, ("chain", "c_to_d")), (3, ("chain", "a_to_b"))]


def test_the_identical_code_check_needs_every_script_in_one_committed_set_and_this_module_in_no_score_or_check_job(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    a = "a" * 64
    score = {L8.SCRIPT_REL: a, L9.SCRIPT_REL: a, L10.SCRIPT_REL: a, L11.SCRIPT_REL: a, P12.SCRIPT_REL: a, P13.SCRIPT_REL: a,
             P14.SCRIPT_REL: a, P.SCRIPT_REL: a}
    fit = {**score, F10.SCRIPT_REL: a, F11.SCRIPT_REL: a, F12.SCRIPT_REL: a, F13.SCRIPT_REL: a, F14.SCRIPT_REL: a, F.SCRIPT_REL: a}
    good = {"score/r2/shard_0of12.json": score, "meta/r2": score, "check/r2": score, "fit/fz/full/TW-1x/k0": fit,
            "fit/sm/b0/TW-4x/k2": fit, "repeat": fit, "read": fit}
    assert F.code_problems(good, "HEAD") == []
    assert F.code_problems({**good, "meta/r2": {}}, "HEAD") == []   # a toy sidecar's meta records no module
    for rel in (F14.SCRIPT_REL, F13.SCRIPT_REL, F12.SCRIPT_REL, F.SCRIPT_REL):
        less = {k: v for k, v in fit.items() if k != rel}
        assert any("are not in every fit" in p for p in F.code_problems({**good, "fit/sm/b0/TW-4x/k2": less}, "HEAD")), rel
    assert any("are not in every fit" in p for p in F.code_problems({**good, "read": score}, "HEAD"))
    assert any(p.startswith(f"check/r2: {P.SCRIPT_REL}") for p in F.code_problems({**good, "check/r2": {L8.SCRIPT_REL: a}}, "HEAD"))
    for job in ("score/r2/shard_1of12.json", "meta/r2", "check/r2"):
        assert any("is in a score, assemble or check job's record" in p for p in F.code_problems({**good, job: fit}, "HEAD")), job
    assert any("2 different" in p for p in F.code_problems({**good, "read": {**fit, F.SCRIPT_REL: "b" * 64}}, "HEAD"))
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "c" * 64)
    assert any("not the file at" in p for p in F.code_problems(good, "HEAD"))


# ── the stages end to end on synthetic sidecars ──────────────────────────────


def test_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16, "x1": 10, "x2": 10, "x3": 10}, n_r=30, every=5)
    cfg = tmp_path / "mp_approx_l15.yaml"
    shutil.copyfile(P.CONFIG, cfg)   # the file stage appends its run record to a copy
    monkeypatch.setattr(F, "CONFIG", cfg)
    monkeypatch.setattr(F, "RECORD", tmp_path / "l15" / "record.json")
    monkeypatch.setattr(F, "DOC", tmp_path / "MP_APPROX_L15.md")
    monkeypatch.setattr(F, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    for norm in F.NORMS:
        for view in F.VIEWS:
            for fit in F.FITS:
                F.stage_fit(decl, norm, view, fit, 0, _quiet)
    said = []
    F.stage_fit(decl, "fz", "b0", "TW-1x", 0, said.append)   # a unit already written is skipped on a restart
    assert any("not refitted" in s for s in said)
    F.stage_repeat(decl, _quiet)
    rd = F.stage_read(decl, _quiet)
    n_r = len(ids["eval"])
    assert rd["primary"] == "FZ-TW-1x-b1d" and rd["reading"] == rd["arms"]["FZ-TW-1x-b1d"]["band"] and rd["reading"] in F.BANDS
    assert rd["carve"] == "r2" and rd["queries"] == n_r
    assert rd["readable_metrics"] and all(v["point"] is not None for v in rd["contrasts"].values())   # the toy is read, not skipped
    assert list(rd["arms"]) == [*F.READ_ARMS, *F.REFERENCES] and list(rd["contrasts"]) == list(F.CONTRASTS)
    for c_name, (a, b) in F.CONTRASTS.items():
        assert abs(rd["contrasts"][c_name]["point"] - (rd["arms"][a]["rho_bar"]["point"] - rd["arms"][b]["rho_bar"]["point"])) < 1e-12
    assert set(rd["strata"]) == set(rd["hop_contrasts"]) == {"hop=1", "hop=2", "hop=3"}
    toy_hops = set(np.unique(L9.View(F.DATA, "nb").q_hop).tolist())
    assert toy_hops == {1, 2}   # level 8's toy qtypes have one and two hops, so hop 3 is an empty stratum here
    assert rd["strata"]["hop=3"]["readable_metrics"] == [] and not any(x.endswith("hop3") for x in rd["interpretation"])
    for strata, contrasts, names in ((rd["strata"], rd["hop_contrasts"], F.HOP_CONTRASTS), (rd["where_strata"], rd["where_contrasts"], F.WHERE_CONTRASTS)):
        for s_name, sc in contrasts.items():
            s = strata[s_name]
            assert list(sc) == list(names)
            for c in names:
                a, b = F.CONTRASTS[c]
                if s["readable_metrics"]:
                    assert abs(sc[c]["point"] - (s["arms"][a]["rho_bar"]["point"] - s["arms"][b]["rho_bar"]["point"])) < 1e-12
                else:
                    assert sc[c]["point"] is None and sc[c]["ci"] is None
    assert list(rd["where_strata"]) == list(rd["where_contrasts"]) == [f"where={w}" for w in F.WHERE]
    assert sum(s["queries"] for s in rd["where_strata"].values()) == n_r
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"] and set(rd["flags"]) <= set(F.FLAGS)
    assert [x for x in F.INTERPRETATION if x in rd["interpretation"]] == rd["interpretation"]
    an = rd["anchors"]
    assert set(an) == set(decl["quantities"]["anchors"])
    assert set(an["level9_anchors"]) == set(an["carve_metrics"]) == {"r2", *P.L12_CARVES}
    ex = an["exchangeability"]
    l14 = json.loads((ROOT / decl["inputs"]["earlier_reads"]["level14"]["path"]).read_text(encoding="utf-8"))
    assert ex["r2"] == an["carve_metrics"]["r2"] and ex["r"] == l14["anchors"]["carve_metrics"]["r"]
    assert set(ex["level12"]) == {"dev", *P12.CARVES} and set(ex["level13_dev"]) == {"twin", "gnn"}
    checks = {c: json.loads((root / "carves" / c / "check.json").read_text(encoding="utf-8")) for c in P.L12_CARVES}
    tr = an["training_rows"]
    assert set(tr) == {f"{n}/{v}/{f}" for n in F.NORMS for v in F.VIEWS for f in F.FITS}
    for fit, carves in F.FITS.items():
        t = tr[f"sm/full/{fit}"]
        assert all(tr[f"{n}/{v}/{fit}"] == t for n in F.NORMS for v in F.VIEWS)   # the same rows fitted and selected in every unit of a fit
        assert t["fit"]["total"] == sum(checks[c]["rows_with_gold_in_pool"]["all"] for c in carves)
        assert t["inner"]["total"] == t["inner"]["select"] == checks["select"]["rows_with_gold_in_pool"]["all"]
        assert t["fit"]["eval"] == t["fit"]["select"] == t["inner"]["eval"] == 0
    assert set(an["agreement"]) == set(an["null_mode"]) == set(an["theta"]) == set(tr)
    assert set(an["unseen_sequences"]) == set(an["fz_vocabulary"]) == {f"{v}/{f}" for v in F.VIEWS for f in F.FITS}
    for key, v in an["fz_vocabulary"].items():
        assert v["vocab"] > 0 and v["scored_rows"] == n_r and (v["partners"] == 0) == key.startswith("b0/")
    for uk, v in an["agreement"].items():
        assert set(v["per_where_mean"]) == set(F.WHERE) and set(an["null_mode"][uk]["per_hop_mean"]) == {"hop=1", "hop=2", "hop=3"}
    for a, ref in F.GAP_SPLIT.items():
        g = an["gap_split"][a]
        assert g["reference"] == ref and g["right_pairs"] + g["wrong_pairs"] == n_r
    for a, c_name in (("FZ-TW-1x-b1d", "ceiling_gap"), ("FZ-TW-4x-b1d", "ceiling_gap_4x"), ("FZ-TW-1x-b0", "ceiling_gap_b0")):
        g = an["gap_split"][a]
        if g["right"] is not None:   # the split adds up to the ceiling gap
            assert abs(g["right"]["all"] + g["wrong"]["all"] - rd["contrasts"][c_name]["point"]) < 1e-9
    assert set(an["beta_choices"]) == set(an["grid_edges"]) == set(F.READ_ARMS)
    assert all(sum(v.values()) == 1 for v in an["beta_choices"].values())
    assert set(an["b1d_moved"]) == {f"{n}/{f}" for n in F.NORMS for f in F.FITS} and all(v["changed_scored_mean"] > 0 for v in an["b1d_moved"].values())
    b0v = an["b0_view"]
    assert set(b0v) == {"all", "hop=1", "hop=2", "hop=3"} and b0v["all"]["rows"] == n_r and 0 < b0v["all"]["no_b0_type_share"] < 1
    reach = an["reaches_gnn"]
    assert reach is None or (reach["arm"] in F.READ_ARMS and reach["fit_rows"] == tr[F.unit_key(reach["arm"])]["fit"]["total"])
    for norm in F.NORMS:
        for view in F.VIEWS:
            for fit in F.FITS:
                npz, js = F.unit_paths(norm, view, fit, 0)
                with np.load(npz) as z:
                    assert np.array_equal(z["q"], np.arange(n_r)) and ("metrics_b1d" in z.files) == (view == "full")
                flog = json.loads(js.read_text(encoding="utf-8"))
                assert (flog["l15_norm"], flog["l14_view"], flog["l14_fit"], flog["l12_fit"], flog["fit"], flog["parts"], flog["k"],
                        flog["fold"]) == (norm, view, fit, fit, "NB-set", list(F12.part_names(fit)), 0, 0)
                assert (flog.get("l13_fit") == fit) == (view == "full") and all((k in flog) == (norm == "fz") for k in F.FZ_KEYS)
                assert flog["scored_queries"] == n_r and len(flog["rounds"]) == 4 and flog["deterministic_algorithms"] is True
                assert {F.SCRIPT_REL, F14.SCRIPT_REL, F13.SCRIPT_REL, F12.SCRIPT_REL, F11.SCRIPT_REL, F10.SCRIPT_REL,
                        P.SCRIPT_REL} <= set(flog["module_sha256"])
    assert set(rd["code"]) >= {f"fit/{n}/{v}/{f}/k0" for n in F.NORMS for v in F.VIEWS for f in F.FITS} | {"repeat", "check/r2", "meta/r2"}
    x3 = root / "carves" / "x3" / "check.json"
    saved = x3.read_bytes()
    x3.write_text(json.dumps({**json.loads(saved), "note": "changed after the read"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was not made from"):
        F.stage_doc(_quiet)
    x3.write_bytes(saved)
    F.stage_doc(_quiet)
    body = F.DOC.read_text(encoding="utf-8")
    for phrase in ("FZ-TW-1x-b1d", "label-matched", "It is never a deployable or selected model", "reaches_gnn", "NB-oracle-b0",
                   "It is also not the within-U_q oracle rho", "train-split rows", "fz_adds", "bucket_ceiling", "true_from_b1_only",
                   "FZ vocabulary", "nor beside level 14's numbers on r as one quantity", "+0.043 [0.031, 0.056]"):
        assert phrase in body, phrase
    assert not any(p in body.lower() for p in FORBIDDEN)
    rec = json.loads(F.RECORD.read_text(encoding="utf-8"))
    assert rec["phase"] == "MP_APPROX_L15" and rec["sources_sha256"]["read.json"] == L0.sha256_file(F.DATA / "read.json")
    F.stage_file("2026_10_02", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l15_2026_10_02"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert run["primary"] == "FZ-TW-1x-b1d" and set(run["contrasts"]) == set(F.CONTRASTS) and set(run["training_rows"]) == set(tr)
    assert run["train_split_rows_scored"] == n_r and run["dev_rows_read"] is False and run["stored_metrics_checked"] is False
    assert set(run["hop_contrasts"]) == {"hop=1", "hop=2", "hop=3"} and set(run["where_contrasts"]) == set(rd["where_contrasts"])
    assert run["by_hop"]["hop=3"] is None and set(run["by_hop"]["hop=1"]) == set(rd["arms"])   # an empty stratum reads nothing
    assert run["b0_view"] == pytest.approx(b0v["all"]) and run["norms"] == list(F.NORMS)
    with pytest.raises(SystemExit, match="exists"):
        F.stage_file("2026_10_02", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_the_stages_refuse_the_wrong_machine_and_wrong_arguments(stops):
    unit = ["--norm", "fz", "--view", "b0", "--fit", "TW-1x", "--k", "0"]
    for argv in (["--stage", "fit", *unit], ["--stage", "repeat"], ["--stage", "read"],
                 ["--stage", "doc", "--host"], ["--stage", "file", "--host"], ["--stage", "fit", "--host"],
                 ["--stage", "fit", "--host", "--view", "b0", "--fit", "TW-1x", "--k", "0"],
                 ["--stage", "fit", "--host", "--norm", "fz", "--fit", "TW-4x", "--k", "0"],
                 ["--stage", "fit", "--host", "--norm", "fz", "--view", "b0", "--k", "0"],
                 ["--stage", "fit", "--host", "--norm", "fz", "--view", "full", "--fit", "TW-1x"],
                 ["--stage", "fit", "--host", "--norm", "softmax", "--view", "b0", "--fit", "TW-1x", "--k", "0"],
                 ["--stage", "fit", "--host", "--norm", "fz", "--view", "b1", "--fit", "TW-1x", "--k", "0"],
                 ["--stage", "fit", "--host", "--norm", "fz", "--view", "b0", "--fit", "TW-2x", "--k", "0"],
                 ["--stage", "fit", "--host", "--norm", "fz", "--view", "b0", "--fit", "TW-1x", "--k", "3"],
                 ["--stage", "repeat", "--host", "--norm", "fz"], ["--stage", "repeat", "--host", "--view", "b0"],
                 ["--stage", "read", "--host", "--k", "0"], ["--stage", "doc", "--date", "2026_10_02"],
                 ["--stage", "read", "--host", "--commit", "abc"], ["--stage", "file"],
                 ["--stage", "score", "--host"], ["--stage", "check", "--host"], ["--stage", "assemble", "--host"]):
        with pytest.raises(SystemExit):
            F.main(argv)
    assert not (stops / "hard_stops.json").exists()
