"""MP-Approx level 7 (configs/mp_approx_l7.yaml#tests): the declaration and its pins; the placement spec; the query-free
columns against level 3's constants; the kernel with every column kept against level 5's L5Kern; the query-freeness of
the neighbour side and its moments; the weights, the mean arm and the compiled form; the bands, contrasts, shares and
grid; the bitwise refit comparison against level 3's and level 5's filed units; the probe, repeat, read and doc stages
end to end on the CPU over a synthetic level 3 sidecar with a level 5 kernel unit; and no held query in any sidecar."""

from __future__ import annotations

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
import mp_approx_l1 as L1  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l5 as L5  # noqa: E402
import mp_approx_l7 as L7  # noqa: E402
from mp_retrieval.m3b_features import EDGE_ATTR  # noqa: E402
from test_mp_approx_l0 import _synthetic_sidecar  # noqa: E402  (level 0's synthetic sidecar)
from test_mp_approx_l3 import _synthetic_l3  # noqa: E402  (level 3's synthetic compile)


def _quiet(_s: str) -> None:
    pass


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L1, L3, L5, L7):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def det_request():
    """host_gpu_det's determinism request on the CPU, restored afterwards (as in level 3's tests)."""
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1], warn_only=before[2])


# ── the declaration, its pins, the placement and the query-free columns ─────


def test_the_declaration_parses_its_pins_are_the_files_current_sha256_and_the_placement_is_level_3s(stops):
    decl = L7.load_declaration()
    assert decl["phase"] == "MP_APPROX_L7" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L3.load_declaration()["registered_question"]
    assert L7.SPEC == L3.SPEC and L7.SPEC["mode"] == "det" and L7.SPEC["threads"] == 8
    assert set(L7.HOST_STAGES) == {"probe", "repeat", "read", "run"}
    here = all((ROOT / decl["inputs"]["level0"]["sidecars"][n]["dir"] / "meta.json").exists()
               and (ROOT / decl["inputs"]["level3"]["sidecars"][n]["dir"] / "probes" / "probe_meta.json").exists()
               and (ROOT / decl["inputs"]["level5"]["probes"][n]["dir"] / "probe_meta.json").exists() for n in L7.DATASETS)
    if not here:
        pytest.skip("level 0's, level 3's or level 5's outputs are not on this machine")
    L7.verify_inputs(decl)


def test_the_query_free_columns_are_level_3s_and_the_declared_ones():
    assert L3.EPS_FAMILIES == ("structural", "ner", "knn", "self") and EDGE_ATTR == ("weight", "rel_compat", "rel_mask", "dir_fwd", "dir_bwd")
    assert L3.EPS_WIDTH == len(L3.EPS_FAMILIES) + len(EDGE_ATTR) + L3.JL_DIM == 73
    assert (L7.A0, L7.WEIGHT_COL, L7.RC_COL, L7.SELF_COL) == (4, 4, 5, 3)
    assert L7.QF_HALO == tuple(range(L3.B0_WIDTH, L3.B0_WIDTH + L3.JL_DIM)) == tuple(range(258, 322)) and L3.HALO_WIDTH == 322
    assert L7.QF_EPS == tuple(range(0, 4)) + tuple(range(6, 73)) and len(L7.QF_EPS) == 71
    assert L7.QW_EPS == tuple(range(0, 5)) + tuple(range(6, 73)) and len(L7.QW_EPS) == 72
    decl = L7.load_declaration()
    text = " ".join(decl["inputs"]["query_free_columns"].split())
    assert "columns 0-3 and 6-72 (71)" in text and "adds column 4 (72)" in text and "columns 258-321 (X_u R)" in text
    arms = decl["probe"]["arms"]
    assert set(arms) == set(L7.NEW)
    for p, (form, side, objective) in L7.NEW.items():
        n = len(L7.SIDES[side][0]) + len(L7.SIDES[side][1])
        assert side in ("qi", "qw") and ("uniform" in arms[p]) == (form == "MEAN") and ("LIST" in arms[p]) == (objective == "LIST")
        if form == "KERN" and objective == "MSE":
            assert f"({n} inputs)" in arms[p]
    assert L7.parameter_count(L7.build("L7-qi")) == L7.parameter_count(L7.build("L7-qi-list"))
    assert L7.parameter_count(L7.L7Kern("full", "KERN")) == L7.parameter_count(L5.L5Kern())


# ── the kernel ───────────────────────────────────────────────────────────────


def _toy(seed: int = 0, n_r: int = 7, n_h: int = 12, m: int = 40, lonely: int = 3):
    """A batch: n_r rows, n_h halo rows, m entries of random families; row `lonely` has no entry."""
    g = np.random.default_rng(seed)
    v = g.integers(0, n_r, m)
    v[v == lonely] = (lonely + 1) % n_r
    fam = g.integers(0, len(L3.FAMILIES), m)
    eps = np.zeros((m, L3.EPS_WIDTH))
    eps[np.arange(m), fam] = 1.0
    a0 = len(L3.EPS_FAMILIES)
    eps[:, a0:a0 + L3.N_ATTR] = g.random((m, L3.N_ATTR))
    eps[:, a0 + L3.N_ATTR:] = g.normal(size=(m, L3.JL_DIM)) * (fam == 0)[:, None]

    def t(a, dt=torch.float64):
        return torch.as_tensor(a, dtype=dt)
    inp = {"bv": t(g.normal(size=(n_r, L3.B0_WIDTH))), "nh": t(g.normal(size=(n_h, L3.HALO_WIDTH))),
           "q_rows": t(np.repeat(g.normal(size=(1, L3.JL_DIM)), n_r, 0)), "ent_u": t(g.integers(0, n_h, m), torch.long),
           "ent_v": t(v, torch.long), "ent_eps": t(eps), "row_h": t(g.permutation(n_h)[:n_r], torch.long)}
    return inp, v, lonely


def _net(side="qi", form="KERN", seed=0):
    torch.manual_seed(seed)
    return L7.L7Kern(side, form).double()


def test_with_every_column_kept_the_l7_kernel_is_level_5s_l5kern_exactly():
    inp, *_ = _toy(4)
    torch.manual_seed(3)
    l5 = L5.L5Kern().double()
    l7 = L7.L7Kern("full", "KERN").double()
    l7.load_state_dict(l5.state_dict())
    with torch.no_grad():
        out5, a5 = l5(**inp, return_alpha=True)
        out7, a7 = l7(**inp, return_alpha=True)
        assert torch.equal(out5, out7) and torch.equal(a5, a7)
        torch.testing.assert_close(l5.compiled(**inp), l7.compiled(**inp), atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError):
        L7.L7Kern("all", "KERN")


def _edge_parts(net, inp):
    n_r = inp["bv"].shape[0]
    with torch.no_grad():
        _v, _phi, psi, val = net.parts(inp["nh"], inp["q_rows"], inp["ent_u"], inp["ent_v"], inp["ent_eps"], inp["row_h"], n_r)
        moments = net.moments(inp["nh"], inp["ent_u"], inp["ent_v"], inp["ent_eps"], inp["row_h"], n_r)
    return [t for t in (psi, val) + tuple(moments) if t is not None]


def _same(a, b) -> bool:
    return all(torch.equal(x, y) for x, y in zip(a, b))


def _perturbed(inp, halo_cols=(), eps_cols=(), q=False, seed=9):
    g = torch.Generator().manual_seed(seed)
    out = dict(inp)
    if halo_cols:
        nh = inp["nh"].clone()
        nh[:, list(halo_cols)] += torch.randn(nh.shape[0], len(halo_cols), generator=g, dtype=nh.dtype) + 0.5
        out["nh"] = nh
    if eps_cols:
        eps = inp["ent_eps"].clone()
        eps[:, list(eps_cols)] += torch.rand(eps.shape[0], len(eps_cols), generator=g, dtype=eps.dtype) + 0.5
        out["ent_eps"] = eps
    if q:
        out["q_rows"] = inp["q_rows"] + 1.0
    return out


@pytest.mark.parametrize("form", L7.FORMS)
def test_l7_qis_neighbour_side_and_moments_are_query_free(form):
    inp, *_ = _toy(5)
    net = _net("qi", form)
    base = _edge_parts(net, inp)
    query_dependent = tuple(range(L3.B0_WIDTH))
    assert _same(base, _edge_parts(net, _perturbed(inp, halo_cols=query_dependent, eps_cols=(L7.WEIGHT_COL, L7.RC_COL), q=True)))
    for c in (257,):
        assert _same(base, _edge_parts(net, _perturbed(inp, halo_cols=(c,))))
    for cols in (L7.QF_HALO, (258,), (321,)):
        assert not _same(base, _edge_parts(net, _perturbed(inp, halo_cols=cols)))
    for c in L7.QF_EPS:
        assert not _same(base, _edge_parts(net, _perturbed(inp, eps_cols=(c,)))), c
    if form == "KERN":
        n_r = inp["bv"].shape[0]
        with torch.no_grad():
            phi = net.parts(inp["nh"], inp["q_rows"], inp["ent_u"], inp["ent_v"], inp["ent_eps"], inp["row_h"], n_r)[1]
            q2 = _perturbed(inp, q=True)
            phi2 = net.parts(q2["nh"], q2["q_rows"], q2["ent_u"], q2["ent_v"], q2["ent_eps"], q2["row_h"], n_r)[1]
        assert not torch.equal(phi, phi2)   # the query enters through phi only


def test_l7_qws_moments_change_with_the_weight_and_not_with_the_halos_query_dependent_columns_or_rel_compat():
    inp, *_ = _toy(6)
    net = _net("qw")
    base = _edge_parts(net, inp)
    assert _same(base, _edge_parts(net, _perturbed(inp, halo_cols=tuple(range(L3.B0_WIDTH)), eps_cols=(L7.RC_COL,), q=True)))
    assert not _same(base, _edge_parts(net, _perturbed(inp, eps_cols=(L7.WEIGHT_COL,))))


@pytest.mark.parametrize("arm", ["L7-qi", "L7-qw", "L7-qi-mean"])
def test_the_weights_are_positive_and_sum_to_one_and_the_mean_arms_are_one_over_the_entries_plus_one(arm):
    inp, v, lonely = _toy(1)
    n_r = inp["bv"].shape[0]
    torch.manual_seed(0)
    net = L7.build(arm).double()
    with torch.no_grad():
        _out, alpha = net(**inp, return_alpha=True)
    v_all = torch.as_tensor(np.concatenate([v, np.arange(n_r)]))
    assert alpha.shape == (v_all.numel(), L7.HEADS) and bool((alpha > 0).all())
    torch.testing.assert_close(torch.zeros(n_r, L7.HEADS, dtype=alpha.dtype).index_add_(0, v_all, alpha),
                               torch.ones(n_r, L7.HEADS, dtype=alpha.dtype), atol=1e-12, rtol=0)
    assert torch.equal(alpha[v.size + lonely], torch.ones(L7.HEADS, dtype=alpha.dtype))   # the self entry alone
    if L7.NEW[arm][0] == "MEAN":
        want = 1.0 / (np.bincount(v, minlength=n_r) + 1.0)
        np.testing.assert_allclose(alpha.numpy(), np.repeat(want[v_all.numpy()][:, None], L7.HEADS, 1), rtol=0, atol=1e-15)


@pytest.mark.parametrize("side,form", [("qi", "KERN"), ("qw", "KERN"), ("full", "KERN"), ("qi", "MEAN")])
def test_the_compiled_form_equals_the_forward_output_and_the_output_is_invariant_to_entry_order(side, form):
    inp, v, _l = _toy(2)
    net = _net(side, form)
    with torch.no_grad():
        out = net(**inp)
        torch.testing.assert_close(out, net.compiled(**inp), atol=1e-10, rtol=1e-10)
        perm = torch.as_tensor(np.random.default_rng(5).permutation(v.size))
        shuffled = net(**dict(inp, ent_u=inp["ent_u"][perm], ent_v=inp["ent_v"][perm], ent_eps=inp["ent_eps"][perm]))
    torch.testing.assert_close(out, shuffled, atol=1e-10, rtol=1e-10)


# ── the read ─────────────────────────────────────────────────────────────────


def test_the_bands_contrasts_shares_and_grid_are_the_declared_ones():
    assert L7.band_l7(0.80, [0.55, 0.95], True) == "L7_HIGH"
    assert L7.band_l7(0.80, [0.45, 0.95], True) == "L7_MID"
    assert L7.band_l7(0.20, [0.00, 0.45], True) == "L7_LOW"
    assert L7.band_l7(0.90, [0.80, 0.99], False) == "NOT_READ"
    decl = L7.load_declaration()
    contrasts = {}
    for c_name, text in decl["statistics"]["contrasts"].items():
        a, b, fam = re.match(r"rho_bar\((\S+)\) - rho_bar\((\S+)\) on (r|e)\b", text).groups()
        contrasts[c_name] = (fam, a, b)
    assert L7.CONTRASTS == contrasts
    shares = {}
    for s_name, text in decl["statistics"]["shares"].items():
        a, b, c, e, fam = re.match(r"\(rho_bar\((\S+)\) - rho_bar\((\S+)\)\) / \(rho_bar\((\S+)\) - rho_bar\((\S+)\)\) on (r|e)\b", text).groups()
        shares[s_name] = (fam, a, b, c, e)
    assert L7.SHARES == shares
    grid = decl["probe"]["grid"]
    assert list(L7.GRID["r"]) == grid["r_k"] and list(L7.GRID["e"]) == grid["e_k"]
    assert all(a in L7.GRID[f] and b in L7.GRID[f] for f, a, b in contrasts.values())
    assert all(all(x in L7.GRID[f] for x in (a, b, c, e)) for f, a, b, c, e in shares.values())
    assert L7.UNIT_ORDER[0] == ("r", "L7-qi") and len(L7.UNIT_ORDER) == len(grid["r_k"]) + len(grid["e_k"])
    assert L7.PRIMARY == "L7-qi" and decl["readings"]["primary_probe"].startswith("L7-qi on r_k")
    assert L7.REPEAT == ("2wiki", "r", "L7-qi") and decl["probe"]["repeat"].startswith("2wiki's (r, L7-qi)")
    assert set(L7.REFIT3) == {"B0-mlp", "L3-att"} and all(p in L3.PROBES for p in L7.REFIT3)
    assert L7.REFIT5 == ("L5-kern",) and L5.NEW["L5-kern"] == ("KERN", "MSE")
    assert set(L7.cross_dataset({})) == set(decl["readings"]["cross_dataset"])
    assert set(L7.STRATA_CONTRASTS) <= set(L7.CONTRASTS) and all(c in decl["statistics"]["strata_contrasts"] for c in L7.STRATA_CONTRASTS)
    assert set(decl["readings"]["interpretation_map"]) == {
        "l7_high", "qi_not_below_kernel", "qi_below_kernel", "qi_above_node_local", "halo_carries", "weight_carries",
        "conditioning_adds", "qi_not_below_attention", "objective_adds", "edge_effect_l7"}
    assert set(decl["readings"]["flags"]) == {"FIT_NOT_RANK", "SEED_BOUND", "REPEAT_DIFFERS", "L3_REFIT_DIFFERS", "L5_REFIT_DIFFERS"}


def _read_stub(points: dict, boots: dict, readable=True) -> dict:
    probes = {p: {"rho_bar": {"point": points[p], "ci": L0.ci(boots[p])}, "_rho_bar_boot": boots[p]} for p in points}
    return {"r": {"probes": probes, "readable_metrics": ["hit@1"] if readable else []},
            "e": {"probes": probes, "readable_metrics": ["hit@1"] if readable else []}}


def test_a_share_is_read_only_where_its_denominator_lies_above_0_and_the_contrasts_are_paired():
    g = np.random.default_rng(0)
    noise = g.normal(size=1000) * 0.01
    points = {"B0-mlp": 0.2, "L3-att": 0.8, "L5-kern": 0.7, "L7-qi": 0.6, "L7-qw": 0.65, "L7-qi-mean": 0.5, "L7-qi-list": 0.75}
    boots = {p: v + noise for p, v in points.items()}
    stub = _read_stub(points, boots)
    s = L7.shares_of(stub)
    assert s["qi_share"]["readable"] and abs(s["qi_share"]["point"] - 0.8) < 1e-12 and np.allclose(s["qi_share"]["ci"], [0.8, 0.8])
    assert s["qi_attention_share"]["readable"] and abs(s["qi_attention_share"]["point"] - 4 / 6) < 1e-12
    c = L7.contrasts_of(stub)
    assert abs(c["deploy_cost"]["point"] + 0.1) < 1e-12 and np.allclose(c["deploy_cost"]["ci"], [-0.1, -0.1])
    assert abs(c["weight_cost"]["point"] + 0.05) < 1e-12 and abs(c["query_conditioning"]["point"] - 0.1) < 1e-12
    assert set(L7.contrasts_of(stub, L7.STRATA_CONTRASTS)) == set(L7.STRATA_CONTRASTS)
    flat = dict(points, **{"L5-kern": 0.2})
    s = L7.shares_of(_read_stub(flat, {p: v + noise for p, v in flat.items()}))
    assert not s["qi_share"]["readable"] and s["qi_share"]["point"] is None and s["qi_share"]["denominator"] is not None
    s = L7.shares_of(_read_stub(points, boots, readable=False))
    assert s["qi_share"]["denominator"] is None and L7.contrasts_of(_read_stub(points, boots, readable=False))["deploy_cost"]["ci"] is None


def _readings_stub(contrast_cis: dict, primary_band="L7_MID", e_band="L7_MID", refit=None) -> dict:
    probes = {p: {"R2": {"point": 0.1}, "band": primary_band if p == L7.PRIMARY else "L7_MID"} for p in L7.GRID["r"]}
    e = {p: {"R2": {"point": 0.1}, "band": e_band if p == L7.PRIMARY else "L7_MID"} for p in L7.GRID["e"]}
    con = {c: {"point": None, "ci": contrast_cis.get(c)} for c in L7.CONTRASTS}
    return {"r": {"probes": probes}, "e": {"probes": e}, "reproducibility": {"r": {"mean": {"point": 0.9}}}, "contrasts": con,
            "refit": refit or {"r/L5-kern": {"level": 5, "bit_identical": True, "max_abs_diff": 0.0}}, "repeat": None}


def _cross_stub(interpretation: list, reading="L7_MID", den_ci=(0.5, 0.7)) -> dict:
    return {"interpretation": interpretation, "reading": reading,
            "shares": {"qi_share": {"denominator": None if den_ci is None else {"point": 0.6, "ci": list(den_ci)}}}}


def test_the_interpretation_entries_follow_the_declared_intervals():
    got = L7.readings(_readings_stub({"deploy_cost": [-0.05, 0.02], "qi_over_node_local": [0.1, 0.3], "halo_cost": [-0.2, -0.05],
                                      "weight_cost": [-0.1, 0.05], "query_conditioning": [0.02, 0.1], "qi_vs_attention": [-0.1, 0.01],
                                      "objective": [-0.1, 0.1], "edge_qi_vs_attention": [-0.3, -0.1]}, primary_band="L7_HIGH"))
    assert got["reading"] == "L7_HIGH" and got["flags"] == []
    assert set(got["interpretation"]) == {"l7_high", "qi_not_below_kernel", "qi_above_node_local", "halo_carries", "conditioning_adds",
                                          "qi_not_below_attention"}
    got = L7.readings(_readings_stub({"deploy_cost": [-0.3, -0.1], "qi_over_node_local": [-0.05, 0.05], "weight_cost": [-0.2, -0.01],
                                      "objective": [0.01, 0.1], "qi_vs_attention": [-0.4, -0.1], "edge_qi_vs_attention": [-0.1, 0.0]},
                                     refit={"r/L5-kern": {"level": 5, "bit_identical": False, "max_abs_diff": 1e-7},
                                            "r/L3-att": {"level": 3, "bit_identical": False, "max_abs_diff": 2e-7}}))
    assert set(got["interpretation"]) == {"qi_below_kernel", "weight_carries", "objective_adds", "edge_effect_l7"}
    assert any(f.startswith("L5_REFIT_DIFFERS (r/L5-kern") for f in got["flags"]) and any(f.startswith("L3_REFIT_DIFFERS (r/L3-att") for f in got["flags"])
    assert L7.readings(_readings_stub({}, e_band="L7_HIGH"))["interpretation"] == ["edge_effect_l7"]
    assert L7.cross_dataset({"2wiki": _cross_stub(["qi_not_below_kernel", "qi_above_node_local"], "L7_HIGH"),
                             "squad": _cross_stub([], "NOT_READ", None)}) == {
        "compile_once_supported": True, "compile_once_partial": False, "compile_once_not_supported": False, "weight_needed": False,
        "kernel_reference_held": True, "negative_control_as_expected": True}
    got = L7.cross_dataset({"2wiki": _cross_stub(["qi_below_kernel", "qi_above_node_local", "weight_carries"], den_ci=(-0.1, 0.3))})
    assert got["compile_once_partial"] is True and got["compile_once_supported"] is False and got["weight_needed"] is True
    assert got["kernel_reference_held"] is False and got["negative_control_as_expected"] is None
    assert L7.cross_dataset({"2wiki": _cross_stub(["qi_not_below_kernel"])})["compile_once_not_supported"] is True
    assert all(v is None for k, v in L7.cross_dataset({"metaqa": _cross_stub(["qi_above_node_local"])}).items())


def test_units_compare_is_bitwise(tmp_path):
    a = np.random.default_rng(0).normal(size=50)
    np.savez(tmp_path / "a.npz", **{"r|B0-mlp|0": a})
    np.savez(tmp_path / "b.npz", **{"r|B0-mlp|0": a.copy()})
    b = a.copy()
    b[7] = np.nextafter(b[7], np.inf)
    np.savez(tmp_path / "c.npz", **{"r|B0-mlp|0": b})
    assert L7.units_compare(tmp_path / "a.npz", tmp_path / "b.npz") == (True, 0.0)
    same, diff = L7.units_compare(tmp_path / "a.npz", tmp_path / "c.npz")
    assert same is False and 0 < diff < 1e-15


def test_the_refit_comparison_checks_each_filed_unit_against_its_probe_meta_and_compares_bitwise(tmp_path, stops):
    g = np.random.default_rng(1)
    d, d3, d5 = tmp_path / "l7", tmp_path / "l3", tmp_path / "l5" / "probes"
    for p in (d / "probes", d3 / "probes", d5):
        p.mkdir(parents=True)
    filed3, filed5 = {}, {}
    for t, p in L7.UNIT_ORDER:
        if p not in L7.REFIT:
            continue
        a = {f"{t}|{p}|{k}": g.normal(size=30) for k in L0.SEEDS}
        f = f"unit_{t}_{p}.npz"
        src = d3 / "probes" if p in L7.REFIT3 else d5
        np.savez(src / f, **a)
        (filed3 if p in L7.REFIT3 else filed5)[f] = L0.sha256_file(src / f)
        if p == "L5-kern":
            a = {**a, f"{t}|{p}|0": np.nextafter(a[f"{t}|{p}|0"], np.inf)}
        np.savez(d / "probes" / f, **a)
    L3.atomic_json(d3 / "probes" / "probe_meta.json", {"files_sha256": filed3})
    L3.atomic_json(d5 / "probe_meta.json", {"files_sha256": filed5})
    out = L7.refit_comparison(d, d3, d5)
    assert set(out) == {"r/B0-mlp", "r/L3-att", "r/L5-kern", "e/B0-mlp", "e/L3-att"}
    assert all(v["bit_identical"] and v["level"] == 3 for k, v in out.items() if k != "r/L5-kern")
    assert out["r/L5-kern"]["level"] == 5 and out["r/L5-kern"]["bit_identical"] is False and 0 < out["r/L5-kern"]["max_abs_diff"] < 1e-14
    np.savez(d5 / "unit_r_L5-kern.npz", x=np.zeros(3))   # the filed unit is no longer the one its probe_meta.json files
    with pytest.raises(SystemExit):
        L7.refit_comparison(d, d3, d5)
    assert (stops / "hard_stops.json").exists()


# ── probe, repeat, read and doc, end to end ─────────────────────────────────


def test_probe_repeat_read_and_doc_run_end_to_end_on_the_cpu(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 2)
    name = L7.REPEAT[0]
    d0 = _synthetic_sidecar(tmp_path / "l0" / name, n_q=120, sizes=(6, 14))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / name
    _synthetic_l3(d3, sc, n_rel=5)
    L3.stage_probe(name, log=_quiet, host=False, device="cpu", d=d3, l0_dir=d0)   # level 3's filed units, for the refit comparison
    d5 = tmp_path / "l5" / name / "probes"                                         # level 5's filed kernel unit, by level 5's code
    L5.fit_units(L3.ProbeData(sc, d3, "cpu"), None, [("r", "L5-kern")], d5, _quiet)
    L3.atomic_json(d5 / "probe_meta.json", {"files_sha256": {f.name: L0.sha256_file(f) for f in sorted(d5.glob("*_r_L5-kern.*"))}})
    d7 = tmp_path / "l7" / name
    L7.stage_probe(name, log=_quiet, host=False, device="cpu", d=d7, d3=d3, l0_dir=d0)
    meta = json.loads((d7 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    want = sorted([f"unit_{t}_{p}.npz" for t, p in L7.UNIT_ORDER] + [f"fitlog_{t}_{p}.json" for t, p in L7.UNIT_ORDER])
    assert sorted(meta["files_sha256"]) == want and meta["unit_order"][0] == "r/L7-qi"
    assert {"scripts/mp_approx_l7.py", "scripts/mp_approx_l5.py", "scripts/mp_approx_l3.py", "scripts/mp_approx_l0.py"} <= set(meta["module_sha256"])
    assert meta["fitting"]["arms"]["L7-qi"]["inputs"] == 135 and meta["fitting"]["arms"]["L7-qw"]["inputs"] == 136
    fl = json.loads((d7 / "probes" / "fitlog_r_L7-qi.json").read_text(encoding="utf-8"))
    assert fl["parameters"] == L7.parameter_count(L7.build("L7-qi")) < L7.parameter_count(L5.L5Kern())
    probes = L0.load_probes(d7)
    assert set(probes) == {f"{t}/{p}/{k}" for t, p in L7.UNIT_ORDER for k in L0.SEEDS}
    for v in probes.values():
        assert v.shape == (sc.n_rows,) and np.isfinite(v).all()
        assert np.abs(L0.centre_rows(v, sc.ptr) - v).max() < 1e-5
    before = L0.sha256_file(d7 / "probes" / "unit_r_L7-qi.npz")
    L7.stage_probe(name, log=_quiet, host=False, device="cpu", d=d7, d3=d3, l0_dir=d0)   # a restart skips every unit
    assert L0.sha256_file(d7 / "probes" / "unit_r_L7-qi.npz") == before
    L7.stage_probe(name, log=_quiet, host=False, device="cpu", d=d7, d3=d3, l0_dir=d0, repeat=True)
    rep = json.loads((d7 / "repeat" / "repeat.json").read_text(encoding="utf-8"))
    assert rep["unit"] == "r_L7-qi" and rep["bit_identical"] is True and rep["max_abs_diff"] == 0.0
    with pytest.raises(SystemExit):
        L7.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d7, d3=d3, l0_dir=d0, repeat=True)
    out = L7.stage_read(name, log=_quiet, host=False, d=d7, d3=d3, d5=d5, l0_dir=d0)
    assert set(out["r"]["probes"]) == set(L7.GRID["r"]) | {"ref:twin", "ref:other_seed"}
    assert set(out["e"]["probes"]) == set(L7.GRID["e"]) | {"ref:no_edge", "ref:other_seed"}
    assert set(out["contrasts"]) == set(L7.CONTRASTS) and set(out["shares"]) == set(L7.SHARES)
    assert set(out["refit"]) == {f"{t}/{p}" for t, p in L7.UNIT_ORDER if p in L7.REFIT} and out["refit"]["r/L5-kern"]["level"] == 5
    assert all(v["bit_identical"] for v in out["refit"].values())   # level 3's and level 5's code on the same inputs: their units exactly
    assert out["repeat"]["bit_identical"] is True
    assert not any(f.startswith(("L3_REFIT_DIFFERS", "L5_REFIT_DIFFERS", "REPEAT_DIFFERS")) for f in out["flags"])
    assert all(set(s["contrasts"]) == set(L7.STRATA_CONTRASTS) for s in out["strata"].values())
    assert {"gold_total=2", "gold_total>=3"} <= set(out["strata"])
    bands = [v["band"] for fam_ in ("r", "e") for v in out[fam_]["probes"].values()]
    assert all(b in ("L7_HIGH", "L7_MID", "L7_LOW", "NOT_READ") for b in bands)
    assert out["reading"] == out["r"]["probes"]["L7-qi"]["band"]
    assert set(out["interpretation"]) <= set(L7.load_declaration()["readings"]["interpretation_map"])
    L7.stage_doc(log=_quiet, out_root=tmp_path / "l7", doc=tmp_path / "doc.md", datasets=(name,), extra_deviations=["--gpus 0.16"])
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "## Readings" in text and "L7-qi-mean" in text and "host_gpu_det" in text and "--gpus 0.16" in text and "in-pool" in text
    assert "Level 5, as filed" in text
    rec = json.loads((tmp_path / "l7" / "record.json").read_text(encoding="utf-8"))
    assert rec["datasets"][name]["read_sha256"] == L0.sha256_file(d7 / "read.json")
    cd = rec["cross_dataset"]
    assert cd["negative_control_as_expected"] is None and isinstance(cd["compile_once_supported"], bool)
    assert sum(cd[k] for k in ("compile_once_supported", "compile_once_partial", "compile_once_not_supported")) <= 1
    assert not (L7.OUT / "hard_stops.json").exists() and not (L5.OUT / "hard_stops.json").exists() and not (L3.OUT / "hard_stops.json").exists()


def test_a_non_finite_prediction_is_a_hard_stop(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 1)
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=60, sizes=(4, 8))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    _synthetic_l3(d3, sc, n_rel=3)
    pd = L3.ProbeData(sc, d3, "cpu")

    def poisoned(pd_, probe, y, teacher, base, bv, stats, fit_q, val_q, test_q, seed):
        return np.full(int(sc.rows_of(test_q).sum()), np.inf), {"seed": seed}
    monkeypatch.setattr(L7, "fit_new", poisoned)
    with pytest.raises(SystemExit):
        L7.fit_units(pd, [("r", "L7-qi")], tmp_path / "p", _quiet)
    assert not (tmp_path / "p" / "unit_r_L7-qi.npz").exists()


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L7.DATASETS)
def test_no_held_query_id_appears_in_any_sidecar(name):
    side = L3.OUT / name / "qids.json"
    if not side.exists():
        pytest.skip(f"{name}: level 3's sidecar is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"][name]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    compiled = json.loads(side.read_text(encoding="utf-8"))
    assert compiled and set(compiled) <= gate
