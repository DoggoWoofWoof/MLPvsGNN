"""MP-Approx level 10, fit module (configs/mp_approx_l10.yaml#tests): the declared constants; the set likelihood against a
brute-force Bernoulli log-likelihood, its closed-form theta as the maximiser of the gamma-weighted log-likelihood, and its
initial theta as the pooled ratio; the coverage against a brute-force sum; NB-draw's dens unit against level 9's NB-hyb
unit bit for bit; EM with the set likelihood on a planted chain beside a planted catch-all type; the scored fold's golds
never reaching its scores; the bands, grid edges, gap split and code check; the check, fit, repeat, read, doc and file
stages end to end on a synthetic combined sidecar; and the stages' machine."""

from __future__ import annotations

import json
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
import mp_approx_l10 as P  # noqa: E402
import mp_approx_l10_fit as F  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar, random multigraphs and relation text)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)

_quiet = T8._quiet
rank_metrics = L8.rank_metrics


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, P):
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


def _view(tmp_path, monkeypatch, family: str, n_q: int, seed: int):
    d = tmp_path / f"s{seed}"
    T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    return L9.View(d, family)


def _length(code: int) -> int:
    return len(L8.token_sequence(int(code)))


# ── the declared constants ───────────────────────────────────────────────────


def test_the_declared_constants_are_the_files():
    decl = P.load_declaration()
    arms = decl["arms"]
    assert {k: tuple(v) for k, v in arms["fits"].items()} == F.FITS
    assert {k: tuple(v) for k, v in arms["read_arms"].items()} == F.READ_ARMS
    assert set(arms["references"]) == set(F.REFERENCES) and set(F.SCORES) == set(arms["scores"])
    assert decl["readings"]["primary"] == F.PRIMARY
    assert {k: tuple(s.strip() for s in v.split(" - ")) for k, v in decl["statistics"]["contrasts"].items()} == F.CONTRASTS
    assert set(decl["readings"]["bands"]) == {"L10_ABOVE_GNN", "L10_HIGH", "L10_LOW", "L10_MID", "NOT_READ"}
    assert set(decl["readings"]["flags"]) == {"CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE", "THETA_AT_CLIP"}
    mapped = {x for pair in F.INTERPRET_CONTRAST.values() for x in pair if x}
    assert set(decl["readings"]["interpretation_map"]) == {"l10_above_gnn", "l10_high", "l10_mid", "l10_low", "chain_not_identified"} | mapped
    assert list(F.KAPPAS) == [1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8, 16] and list(F.ETAS) == [0.01, 0.1, 1, 10, 100]
    assert "{1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16}" in arms["selection"] and "{0.01, 0.1, 1, 10, 100}" in arms["selection"]
    assert F.EM_ROUNDS == 10 and "10 (E, M) rounds" in decl["em"]["rounds"]
    assert F.THETA_CLIP == 1e-6 and "[1e-6, 1 - 1e-6]" in arms["likelihood"]["set"]
    assert F.REPEAT_UNIT == ("NB-set", 0, 0) and "(NB-set, k = 0, fold 0)" in decl["cross_fitting"]["repeat"]
    assert set(F.GAP_SPLIT_ARMS) == {"NB-set-cov", "NB-draw-dens"} and "For NB-set-cov and NB-draw-dens" in decl["quantities"]["anchors"]["gap_split"]
    assert F.SCRIPT_REL in decl["outputs"]["scripts"] and P.SCRIPT_REL in decl["outputs"]["scripts"]
    assert (F.CONFIG, F.OUT, F.DATA, F.NAME) == (P.CONFIG, P.OUT, P.DATA, P.NAME) and F.DOC == ROOT / decl["outputs"]["document"]


# ── the set likelihood ───────────────────────────────────────────────────────


def test_the_set_likelihood_rows_are_a_bernoulli_log_likelihood_over_the_pool(tmp_path, monkeypatch):
    view = _view(tmp_path, monkeypatch, "nb", n_q=14, seed=1)
    fx = F.SetFitter(view, T8._rel())
    theta = np.array([0.7, 0.4, 0.2, 0.05])
    fx.set_theta(theta)
    for q in range(view.n_q):
        n = int(view.q_pool_size[q])
        gold = np.zeros(n, dtype=bool)
        gold[view.gold_local(q)] = True
        want = []
        for c in view.t_code[view.type_rows(q)]:
            inside = np.zeros(n, dtype=bool)
            inside[view.reach(q, int(c))] = True
            rho, eps = theta[_length(c) - 1], theta[3]
            want.append(sum(np.log(rho) if (i and g) else np.log(1 - rho) if i else np.log(eps) if g else np.log(1 - eps)
                            for i, g in zip(inside, gold)))
        want.append(sum(np.log(theta[3]) if g else np.log(1 - theta[3]) for g in gold))   # the null type reaches nothing
        assert np.allclose(fx.loglik(q), want, rtol=1e-12, atol=1e-9)


def test_the_closed_form_theta_is_the_pooled_weighted_ratio_and_maximises_the_weighted_log_likelihood(tmp_path, monkeypatch):
    view = _view(tmp_path, monkeypatch, "nb", n_q=24, seed=2)
    fx = F.SetFitter(view, T8._rel())
    fx.set_theta([0.5, 0.5, 0.5, 0.1])
    rng = np.random.default_rng(0)
    qs = np.arange(view.n_q)
    gammas = [rng.dirichlet(np.ones(int(view.q_types[q]) + 1)) for q in qs]
    best = fx.update_theta(gammas, qs)
    num, den, e_num, e_den = np.zeros(3), np.zeros(3), 0.0, 0.0
    for g, q in zip(gammas, qs):
        rows = view.type_rows(q)
        length = np.asarray([_length(c) for c in view.t_code[rows]])
        h, r = view.t_gold[rows].astype(float), view.t_size[rows].astype(float)
        for L in (1, 2, 3):
            num[L - 1] += (g[:-1] * h)[length == L].sum()
            den[L - 1] += (g[:-1] * r)[length == L].sum()
        G, N = float(view.q_gold_in_pool[q]), float(view.q_pool_size[q])
        e_num += (g[:-1] * (G - h)).sum() + g[-1] * G
        e_den += (g[:-1] * (N - r)).sum() + g[-1] * N
    assert (den > 0).all() and np.allclose(best, np.r_[num / den, e_num / e_den], rtol=1e-12, atol=0)

    def weighted(theta) -> float:
        fx.set_theta(theta)
        return sum(float(g @ fx.loglik(int(q))) for g, q in zip(gammas, qs))

    top = weighted(best)
    for i in range(4):
        for step in (1e-3, -1e-3, 1e-1, -1e-1):
            t = best.copy()
            t[i] = min(t[i] * (1 + step), 1 - 1e-6)
            assert weighted(t) <= top + 1e-9, (i, step)
    for _ in range(20):
        assert weighted(rng.uniform(0.01, 0.99, 4)) <= top


def test_the_initial_theta_is_the_pooled_ratio_and_a_zero_denominator_keeps_its_value(tmp_path, monkeypatch):
    view = _view(tmp_path, monkeypatch, "nb", n_q=40, seed=5)
    fx = F.SetFitter(view, T8._rel())
    fit_q, _inner, _score = L8.unit_queries(view, 0)
    h, r = np.zeros(3), np.zeros(3)
    for q in fit_q:
        rows = view.type_rows(q)
        for c, hh, rr in zip(view.t_code[rows], view.t_gold[rows], view.t_size[rows]):
            h[_length(c) - 1] += hh
            r[_length(c) - 1] += rr
    eps = view.q_gold_in_pool[fit_q].sum() / view.q_pool_size[fit_q].sum()
    assert np.allclose(fx.initial_theta(fit_q), np.r_[h / r, eps], rtol=1e-12, atol=0)
    real = fx.row_len.copy()
    fx.row_len = np.minimum(real, 2)   # no type of length 3: rho_3 starts at eps_s
    t0 = fx.initial_theta(fit_q)
    assert t0[2] == t0[3]
    fx.row_len = real
    fx.set_theta([0.3, 0.3, 0.123, 0.05])
    rng = np.random.default_rng(1)
    gammas = []
    for q in fit_q:
        g = rng.dirichlet(np.ones(int(view.q_types[q]) + 1))
        g[:-1][real[view.type_rows(q)] == 3] = 0.0   # no weight on a type of length 3: rho_3 keeps its value
        gammas.append(g / g.sum())
    new = fx.update_theta(gammas, fit_q)
    assert new[2] == 0.123 and new[0] != 0.3 and new[3] != 0.05


def test_theta_is_clipped_a_non_finite_theta_stops_and_the_clip_flag_reads_the_bounds(tmp_path, monkeypatch, stops):
    view = _view(tmp_path, monkeypatch, "nb", n_q=10, seed=6)
    fx = F.SetFitter(view, T8._rel())
    fx.set_theta([0.0, 1.0, 0.5, 0.0])
    assert fx.theta.tolist() == [1e-6, 1 - 1e-6, 0.5, 1e-6] and np.isfinite(fx.ll_rows).all()
    assert F.theta_at_clip(fx.theta) and not F.theta_at_clip([0.5, 0.4, 0.3, 0.01])
    with pytest.raises(SystemExit, match="HARD STOP"):
        fx.set_theta([np.nan, 0.5, 0.5, 0.1])
    assert "not finite" in (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the coverage ─────────────────────────────────────────────────────────────


def test_the_coverage_sums_the_types_that_reach_the_node_has_pool_mean_1_and_is_the_density_at_a_point_mass(tmp_path, monkeypatch):
    view = _view(tmp_path, monkeypatch, "nb", n_q=12, seed=1)
    fx = L8.Fitter(view, T8._rel())
    rng = np.random.default_rng(0)
    for q in range(view.n_q):
        T, n = int(view.q_types[q]), int(view.q_pool_size[q])
        logit = rng.normal(size=T + 1)
        lp = logit - np.log(np.exp(logit).sum())
        want = np.zeros(n)
        codes = view.t_code[view.type_rows(q)]
        for t, c in enumerate(codes):
            want[view.reach(q, int(c))] += np.exp(lp[t])
        a = F.coverage(view, q, lp)
        assert np.allclose(a, n * want / want.sum(), rtol=1e-12, atol=1e-12) and abs(a.mean() - 1.0) < 1e-12
        point = np.full(T + 1, -np.inf)
        point[int(rng.integers(T))] = 0.0
        assert np.allclose(F.coverage(view, q, point), fx.mixture(q, point), rtol=1e-12, atol=1e-12)
        null = np.r_[np.full(T, -np.inf), 0.0]   # all mass on the null type: no node is covered
        assert np.array_equal(F.coverage(view, q, null), np.ones(n))
    assert F.a_map(fx, "dens") == fx.mixture


# ── EM ───────────────────────────────────────────────────────────────────────


def test_nb_draws_dens_unit_is_level_9s_nb_hyb_unit_bit_for_bit(tmp_path, monkeypatch, one_thread):
    torch.use_deterministic_algorithms(True)
    view = _view(tmp_path, monkeypatch, "nb", n_q=160, seed=3)
    monkeypatch.setattr(L8, "M_EPOCHS", 3)
    monkeypatch.setattr(L8, "EM_ROUNDS", 2)
    monkeypatch.setattr(L8, "LR", 3e-2)
    monkeypatch.setattr(L9, "EM_ROUNDS", 4)
    monkeypatch.setattr(F, "EM_ROUNDS", 4)
    rel = T8._rel()
    a9, f9 = L9.fit_unit(L8.Fitter(view, rel), "NB-hyb", 0, 0, _quiet)
    a10, f10 = F.fit_unit(F.make_fitter(view, rel, "NB-draw"), "NB-draw", 0, 0, _quiet)
    assert f9["kept_round"] > 0   # a fitted round is kept, so the reloaded state matters
    for k9, k10 in (("q", "q"), ("metrics", "metrics_dens"), ("score", "score_dens"), ("score_ptr", "score_ptr"), ("argmax", "argmax")):
        assert a9[k9].dtype == a10[k10].dtype and np.array_equal(a9[k9], a10[k10]), k10
    assert (f10["kept_round"], f10["kappa_dens"], f10["eta_dens"]) == (f9["kept_round"], f9["kappa"], f9["eta"])
    assert f10["grid_inner_mean3_dens"] == f9["grid_inner_mean3"] and f10["rounds"] == f9["rounds"]
    assert f10["parameters"] == f9["parameters"] and "kept_theta" not in f10 and len(f10["grid_inner_mean3_cov"]) == 50
    assert a10["score_cov"].shape == a10["score_dens"].shape and a10["metrics_cov"].shape == a10["metrics_dens"].shape


CATCH = [3 * L8.REL_ORDER.index("has_genre"), 3 * L8.REL_ORDER.index("has_genre") + 1]   # has_genre fwd, then bwd


def _catchall_sidecar(d: Path, n_q: int, seed: int) -> list[str]:
    """Level 8's toy layout on the combined programme, with the golds set on the nb view. The golds are the true chain's
    nb reach set from the first seed plus one pool node outside it, so the chain loses one gold per query. The golds'
    kind is the golds and a fifth of the pool's other nodes. A catch-all type (CATCH, the same in every query, whose
    tokens are on no other edge) reaches every node of that kind."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    parts = {k: [] for k in L8.ARRAY_KEYS}
    qids, qtypes = [], sorted(T8.QT_TOY)
    i = 0
    while len(qids) < n_q:
        i += 1
        qt = T8.QT_TOY[int(rng.integers(len(T8.QT_TOY)))]
        steps = L8.true_chain(qt)
        chain = L8.chain_tokens(steps)
        n = int(rng.integers(50, 80))
        src, dst, tok = T8._random_multigraph(rng, n, int(rng.integers(2 * n, 3 * n)))
        keep = ~np.isin(tok, CATCH)
        src, dst, tok = src[keep], dst[keep], tok[keep]
        seeds = rng.choice(n, size=int(rng.integers(3, 6)), replace=False)
        bucket = np.r_[0, 0, np.ones(seeds.size - 2, dtype=np.int64)]
        for tgt in rng.choice(n, size=3, replace=False):
            at = int(seeds[0])
            for pos, t in enumerate(chain):
                nxt = int(tgt) if pos == len(chain) - 1 else int(rng.integers(n))
                src, dst, tok = np.r_[src, at], np.r_[dst, nxt], np.r_[tok, t]
                at = nxt
        buckets = [seeds[bucket == 0], seeds[bucket == 1]]
        code, node, _count = L9.walk_entries_both(src, dst, tok, n, buckets)
        chain_code = L8.type_code(0, chain) + L9.NB_SHIFT
        reach = node[code == chain_code]
        outside = np.setdiff1d(np.arange(n), np.r_[reach, seeds])
        if reach.size < 2 or reach.size > n // 4 or np.isin(seeds, reach).any() or outside.size < n // 5 + 1:
            continue
        pick = rng.choice(outside, size=n // 5 + 1, replace=False)
        golds = np.r_[reach, pick[0]]
        for tgt in np.r_[golds, pick[1:]]:
            mid = int(rng.choice(np.setdiff1d(np.arange(n), [seeds[0], tgt])))
            src, dst, tok = np.r_[src, seeds[0], mid], np.r_[dst, mid, tgt], np.r_[tok, CATCH]
        code, node, count = L9.walk_entries_both(src, dst, tok, n, buckets)
        if not np.array_equal(node[code == chain_code], reach) or not np.isin(golds, node[code == L8.type_code(0, CATCH) + L9.NB_SHIFT]).all():
            continue
        is_gold = np.zeros(n, dtype=bool)
        is_gold[golds] = True
        t_code, t_size, t_gold = L8.type_table(code, node, is_gold)
        qid = f"metaqa:{len(steps)}hop:toy:{i}"
        gl = np.flatnonzero(is_gold)
        twin = rng.normal(size=(n, 3)).astype(np.float32)
        gnn = twin + 2.0 * is_gold[:, None]
        metrics = [[rank_metrics(s[:, k].astype(np.float64), gl, gl.size)[mm] for mm in L8.METRIC_NAMES] for s in (twin, gnn) for k in range(3)]
        seed_row, bucket_row = np.full(10, -1), np.full(10, -1)
        seed_row[:seeds.size], bucket_row[:seeds.size] = seeds, bucket
        emb = rng.normal(scale=0.01, size=1536).astype(np.float32)
        emb[qtypes.index(qt)] += 5.0
        for key, val in (("q_row", len(qids)), ("q_hop", len(steps)), ("q_qtype", qtypes.index(qt)), ("q_fold", L0.fold_of(qid)),
                         ("q_inner", int(L0.is_inner(qid))), ("q_pool_size", n), ("q_gold_total", int(gl.size)),
                         ("q_gold_in_pool", int(gl.size)), ("q_te_local", int(seeds[0])), ("q_seed_local", seed_row),
                         ("q_seed_bucket", bucket_row), ("q_metrics", metrics), ("q_types", int(t_code.size)),
                         ("q_entries", int(code.size)), ("q_emb", emb), ("q_struct_edges", int(src.size)),
                         ("q_token_edges", int(src.size)), ("q_dir_class", [int(src.size), 0, 0])):
            parts[key].append(val)
        parts["twin_score"].append(twin)
        parts["is_gold"].append(is_gold)
        for key, val in (("t_code", t_code.astype(np.int32)), ("t_size", t_size.astype(np.int32)), ("t_gold", t_gold.astype(np.int32)),
                         ("e_code", code.astype(np.int32)), ("e_node", node.astype(np.int32)), ("e_count", count.astype(np.uint32))):
            parts[key].append(val)
        qids.append(qid)
    shas = {}
    for key in L8.Q_KEYS:
        dtype = np.float64 if key == "q_metrics" else np.float32 if key == "q_emb" else np.int64
        np.save(d / f"{key}.npy", np.asarray(parts[key], dtype=dtype))
    for key in L8.NODE_KEYS + L8.TYPE_KEYS + L8.ENTRY_KEYS:
        np.save(d / f"{key}.npy", np.concatenate(parts[key]))
    for key in L8.ARRAY_KEYS:
        shas[f"{key}.npy"] = L0.sha256_file(d / f"{key}.npy")
    (d / "qids.json").write_text(json.dumps(qids), encoding="utf-8")
    meta = {"arrays_sha256": shas, "qids_sha256": L0.sha256_file(d / "qids.json"), "qtypes": qtypes, "limit": None,
            "queries": len(qids), "mismatches": 0, "module_sha256": {}, "shards": {}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return qids


def test_em_with_the_set_likelihood_puts_its_argmax_on_a_planted_chain_beside_a_planted_catch_all(tmp_path, monkeypatch, one_thread):
    _catchall_sidecar(tmp_path / "s", n_q=240, seed=3)
    view = L9.View(tmp_path / "s", "nb")
    monkeypatch.setattr(F, "EM_ROUNDS", 5)
    rel = T8._rel()
    share = {}
    for fit in ("NB-set", "NB-draw"):
        arrays, flog = F.fit_unit(F.make_fitter(view, rel, fit), fit, 0, 0, _quiet)
        qt = [view.meta["qtypes"][i] for i in view.q_qtype[arrays["q"]]]
        seqs = [L8.token_sequence(int(c)) for c in arrays["argmax"]]
        share[fit] = {"chain": float(np.mean([s == tuple(L8.chain_tokens(L8.true_chain(t))) for s, t in zip(seqs, qt)])),
                      "catch_all": float(np.mean([s == tuple(CATCH) for s in seqs])), "kept_round": flog["kept_round"],
                      "theta": flog.get("kept_theta")}
    print("catch-all toy shares:", json.dumps(share))   # NB-draw's shares are descriptive only
    assert share["NB-set"]["chain"] >= 0.8, share


def test_the_scored_folds_golds_never_reach_its_scores(tmp_path, monkeypatch, one_thread):
    """arms: each arm's inputs are z(T_k), q, the relation text and the view's types and counts. The scored fold's golds
    are read only by its metrics."""
    view = _view(tmp_path, monkeypatch, "nb", n_q=60, seed=7)
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F, "EM_ROUNDS", 2)
    rel = T8._rel()
    first, _f = F.fit_unit(F.make_fitter(view, rel, "NB-set"), "NB-set", 0, 0, _quiet)
    other = L9.View(view.dir, "nb")
    is_gold, t_gold = np.array(other.is_gold, copy=True), np.array(other.t_gold, copy=True)
    rng = np.random.default_rng(0)
    for q in np.flatnonzero(other.q_fold == 0):
        sl = other.nodes(q)
        is_gold[sl] = rng.permutation(is_gold[sl])   # the same count, other nodes
        rows = other.type_rows(q)
        t_gold[rows] = rng.integers(0, 3, size=t_gold[rows].size)
    other.is_gold, other.t_gold = is_gold, t_gold
    again, _f = F.fit_unit(F.make_fitter(other, rel, "NB-set"), "NB-set", 0, 0, _quiet)
    for key in ("q", "score_dens", "score_cov", "score_ptr", "argmax"):
        assert np.array_equal(first[key], again[key]), key
    assert not np.array_equal(first["metrics_cov"], again["metrics_cov"])   # the metrics do read them


def test_a_fit_on_the_wrong_view_or_fitter_is_refused(tmp_path, monkeypatch):
    view = _view(tmp_path, monkeypatch, "std", n_q=20, seed=9)
    with pytest.raises(SystemExit, match="not the fit's"):
        F.fit_unit(F.make_fitter(view, T8._rel(), "NB-set"), "NB-set", 0, 0, _quiet)   # an std view for an nb fit
    with pytest.raises(SystemExit, match="not the fit's"):
        F.fit_unit(L8.Fitter(view, T8._rel()), "STD-set", 0, 0, _quiet)   # the draw fitter for a set fit


# ── the statistics and the readings ──────────────────────────────────────────


def test_the_bands_come_in_the_declared_order_and_read_arm_carries_them():
    assert F.band(1.3, [1.05, 1.5], True) == "L10_ABOVE_GNN"
    assert F.band(0.8, [0.6, 1.0], True) == "L10_HIGH"
    assert F.band(0.8, [0.4, 1.2], True) == "L10_MID"
    assert F.band(0.2, [0.0, 0.45], True) == "L10_LOW"
    assert F.band(0.2, [0.0, 0.55], True) == "L10_MID"
    assert F.band(None, None, False) == "NOT_READ" and F.band(0.9, [0.8, 1.0], False) == "NOT_READ"
    rng = np.random.default_rng(0)
    T = rng.random((40, 3, 3))
    G = T + 0.5
    W = L0.boot_weights(40)
    _d, dens, readable = L8.denominators(T, G, W)
    assert F.read_arm(T + 0.45, T, G, dens, readable, W)["band"] == "L10_HIGH"
    assert F.read_arm(T + 0.6, T, G, dens, readable, W)["band"] == "L10_ABOVE_GNN"
    assert F.read_arm(T + 0.05, T, G, dens, readable, W)["band"] == "L10_LOW"


def test_the_gap_split_sums_to_the_ceiling_gap_and_its_hops_to_the_whole():
    rng = np.random.default_rng(0)
    n, S = 30, 3
    T = rng.random((n, S, 3))
    G = T + rng.uniform(0.3, 0.6, (n, S, 3))
    W = L0.boot_weights(n)
    _d, dens, readable = L8.denominators(T, G, W)
    assert readable == list(F.RETRIEVAL)
    Mo, Mx = T + rng.uniform(0.2, 0.9, (n, S, 3)), T + rng.uniform(0.0, 0.5, (n, S, 3))
    right, hop = rng.random((n, S)) < 0.6, np.repeat([1, 2, 3], 10)
    gs = F.gap_split(Mo, Mx, dens, readable, right, hop)
    gap = L8.read_arm(Mo, T, G, dens, readable, W)["rho_bar"]["point"] - L8.read_arm(Mx, T, G, dens, readable, W)["rho_bar"]["point"]
    assert abs(gs["right"]["all"] + gs["wrong"]["all"] - gap) < 1e-12
    for tag in ("right", "wrong"):
        assert abs(sum(gs[tag][f"hop={h}"] for h in (1, 2, 3)) - gs[tag]["all"]) < 1e-12
    assert gs["right_pairs"] + gs["wrong_pairs"] == n * S and gs["right_pairs"] == int(right.sum())
    whole = F.gap_split(Mo, Mx, dens, readable, np.ones((n, S), dtype=bool), hop)
    assert abs(whole["right"]["all"] - gap) < 1e-12 and whole["wrong"]["all"] == 0.0
    assert F.gap_split(Mo, Mx, dens, [], right, hop)["right"] is None


def test_grid_edges_count_the_ends_of_the_scores_grid_and_the_last_round():
    units = {"a": {"kappa_cov": F.KAPPAS[0], "eta_cov": 1, "kept_round": F.EM_ROUNDS, "kappa_dens": 1, "eta_dens": 1},
             "b": {"kappa_cov": 1, "eta_cov": F.ETAS[-1], "kept_round": 3, "kappa_dens": 1, "eta_dens": 1},
             "c": {"kappa_cov": 1, "eta_cov": 1, "kept_round": 0, "kappa_dens": F.KAPPAS[-1], "eta_dens": 1},
             "d": {"kappa_cov": F.KAPPAS[-1], "eta_cov": F.ETAS[0], "kept_round": 2, "kappa_dens": 1, "eta_dens": 1}}
    assert F.grid_edges(units, "cov") == {"units": 4, "kappa_at_edge": 2, "eta_at_edge": 2, "either_at_edge": 3, "kept_last_round": 1,
                                          "kappa_at_low_end": 1, "kappa_at_high_end": 1, "eta_at_low_end": 1, "eta_at_high_end": 1}
    assert F.grid_edges(units, "dens")["either_at_edge"] == 1


def test_the_identical_code_check_needs_both_scripts_in_one_committed_set(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    a = "a" * 64
    score = {L8.SCRIPT_REL: a, L9.SCRIPT_REL: a, P.SCRIPT_REL: a}
    fit = {**score, F.SCRIPT_REL: a}
    good = {"score/shard_0": score, "check": score, "meta": score, "fit/NB-set/k0_f0": fit, "repeat": fit, "read": fit}
    assert F.code_problems(good, "HEAD") == []
    assert F.code_problems({**good, "meta": {}}, "HEAD") == []   # a toy sidecar's meta records no module
    assert any("mp_approx_l10_fit.py is not in every" in p for p in F.code_problems({**good, "repeat": score}, "HEAD"))
    assert any(p.startswith("check: scripts/mp_approx_l10.py") for p in F.code_problems({**good, "check": {L8.SCRIPT_REL: a}}, "HEAD"))
    assert any("2 different" in p for p in F.code_problems({**good, "read": {**fit, F.SCRIPT_REL: "b" * 64}}, "HEAD"))
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "c" * 64)
    assert any("not the file at" in p for p in F.code_problems(good, "HEAD"))


# ── the stages end to end on the synthetic combined sidecar ──────────────────


def _stage_env(tmp_path, monkeypatch, n_q, seed):
    d = tmp_path / "l10" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    cfg = tmp_path / "mp_approx_l10.yaml"
    shutil.copyfile(P.CONFIG, cfg)   # the file stage appends its run record to a copy
    for mod in (P, F):
        monkeypatch.setattr(mod, "DATA", d)
        monkeypatch.setattr(mod, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(F, "CONFIG", cfg)
    monkeypatch.setattr(F, "RECORD", tmp_path / "l10" / "record.json")
    monkeypatch.setattr(F, "DOC", tmp_path / "MP_APPROX_L10.md")
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "fit_process", _light_fit_process)
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    return d, cfg


def test_check_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    d, cfg = _stage_env(tmp_path, monkeypatch, n_q=90, seed=4)
    monkeypatch.setattr(F, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    decl = P.load_declaration()
    with pytest.raises(SystemExit, match="check stage has not passed"):
        F.stage_fit(decl, "NB-set", 0, _quiet)
    check = P.stage_check(decl, _quiet)
    assert all(check["families"][f]["direction_check"]["passes"] for f in P.FAMILIES)
    for fit in F.FITS:
        F.stage_fit(decl, fit, 0, _quiet)
    F.stage_fit(decl, "NB-set", 0, _quiet)   # every unit is logged: a restart skips them all
    F.stage_repeat(decl, _quiet)
    rd = F.stage_read(decl, _quiet)
    assert rd["primary"] == "NB-set-cov" and rd["reading"] == rd["arms"]["NB-set-cov"]["band"]
    assert set(rd["arms"]) == set(F.READ_ARMS) | set(F.REFERENCES) and set(rd["contrasts"]) == set(F.CONTRASTS)
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"]
    assert set(rd["strata"]) == {"hop=1", "hop=2", "hop=3"} and set(rd["anchors"]["agreement"]) == set(F.FITS)
    assert set(rd["anchors"]["theta"]) == {"NB-set", "STD-set"} and set(rd["anchors"]["grid_edges"]) == set(F.READ_ARMS)
    assert set(rd["anchors"]["gap_split"]) == set(F.GAP_SPLIT_ARMS) and "gold_unreached_nb" in rd["anchors"]["check"]
    c = rd["contrasts"]["over_level9_primary"]
    assert abs(c["point"] - (rd["arms"]["NB-set-cov"]["rho_bar"]["point"] - rd["arms"]["NB-draw-dens"]["rho_bar"]["point"])) < 1e-12
    assert rd["arms"]["NB-oracle"]["mean"]["recall@5"]["arm"] >= rd["arms"]["NB-oracle"]["mean"]["recall@5"]["twin"]
    base = L8.Data(d, check=False)
    for fit in F.FITS:   # every query exactly once, out of fold, under both scores
        for fold in range(5):
            with np.load(d / "units" / fit / f"k0_f{fold}.npz") as z:
                assert np.array_equal(z["q"], np.flatnonzero(base.q_fold == fold))
                assert z["metrics_cov"].shape == z["metrics_dens"].shape and z["score_cov"].shape == z["score_dens"].shape
            flog = json.loads((d / "units" / fit / f"k0_f{fold}.json").read_text(encoding="utf-8"))
            assert flog["family"] == F.FITS[fit][0] and flog["likelihood"] == F.FITS[fit][1] and len(flog["rounds"]) == 4
            assert flog["deterministic_algorithms"] is True and F.SCRIPT_REL in flog["module_sha256"]
            assert ("kept_theta" in flog) == (F.FITS[fit][1] == "set") and all(("theta" in r) == ("kept_theta" in flog) for r in flog["rounds"])
    F.stage_doc(_quiet)
    body = F.DOC.read_text(encoding="utf-8")
    assert "NB-set-cov" in body and "NB-draw-dens" in body and "It is not the within-U_q oracle rho" in body
    assert "beyond the declared list" in body
    assert not any(p in body.lower() for p in ("message passing is unnecessary", "we do not need message passing", "the mlp wins"))
    rec = json.loads(F.RECORD.read_text(encoding="utf-8"))
    assert rec["sources_sha256"]["read.json"] == L0.sha256_file(d / "read.json") and rec["phase"] == "MP_APPROX_L10"
    F.stage_file("2026_10_01", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l10_2026_10_01"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert set(run["contrasts"]) == set(F.CONTRASTS)
    with pytest.raises(SystemExit, match="exists"):
        F.stage_file("2026_10_01", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_the_stages_refuse_the_wrong_machine_and_wrong_arguments(stops):
    for argv in (["--stage", "fit", "--fit", "NB-set", "--k", "0"], ["--stage", "repeat"], ["--stage", "read"],
                 ["--stage", "doc", "--host"], ["--stage", "file", "--host"], ["--stage", "fit", "--host"],
                 ["--stage", "fit", "--host", "--fit", "NB-set"], ["--stage", "repeat", "--host", "--fit", "NB-set"],
                 ["--stage", "read", "--host", "--k", "0"], ["--stage", "fit", "--host", "--fit", "XX", "--k", "0"],
                 ["--stage", "fit", "--host", "--fit", "NB-set", "--k", "7"], ["--stage", "score", "--host"]):
        with pytest.raises(SystemExit):
            F.main(argv)
