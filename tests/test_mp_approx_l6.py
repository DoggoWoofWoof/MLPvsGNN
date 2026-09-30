"""MP-Approx level 6 (configs/mp_approx_l6.yaml#tests): the declaration and its pins; the placement spec; the structural
column; parameters shared across depth and no dataset input; the weights, the two messages and the compiled form; the
fixed variant; the bands, contrasts, shares and grid; the bitwise refit comparison; the probe, repeat, read and doc
stages end to end on the CPU over a synthetic level 3 sidecar; and no held query in any sidecar."""

from __future__ import annotations

import inspect
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
import mp_approx_l6 as L6  # noqa: E402
from test_mp_approx_l0 import _synthetic_sidecar  # noqa: E402  (level 0's synthetic sidecar)
from test_mp_approx_l3 import _synthetic_l3  # noqa: E402  (level 3's synthetic compile)


def _quiet(_s: str) -> None:
    pass


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L1, L3, L6):
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


# ── the declaration, its pins and the placement ─────────────────────────────


def test_the_declaration_parses_its_pins_are_the_files_current_sha256_and_the_placement_is_level_3s(stops):
    decl = L6.load_declaration()
    assert decl["phase"] == "MP_APPROX_L6" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L3.load_declaration()["registered_question"]
    assert L6.SPEC == L3.SPEC and L6.SPEC["mode"] == "det" and L6.SPEC["threads"] == 8
    assert set(L6.HOST_STAGES) == {"probe", "repeat", "read", "run"}
    sidecars = {n: (decl["inputs"]["level0"]["sidecars"][n], decl["inputs"]["level3"]["sidecars"][n]) for n in L6.DATASETS}
    here = all((ROOT / p0["dir"] / "meta.json").exists() and (ROOT / p3["dir"] / "probes" / "probe_meta.json").exists() for p0, p3 in sidecars.values())
    if not here:
        pytest.skip("level 0's or level 3's sidecars are not on this machine")
    L6.verify_inputs(decl)


def test_the_structural_column_is_eps_column_0_in_level_3s_family_order():
    assert L3.EPS_FAMILIES == ("structural", "ner", "knn", "self") and L6.STRUCT_COL == 0 and L6.SELF_COL == 3
    assert "column 0" in " ".join(L6.load_declaration()["inputs"]["structural_column"].split())


# ── the recurrence ───────────────────────────────────────────────────────────


def _toy(seed: int = 0, n_r: int = 7, n_h: int = 12, m: int = 40, lonely: int = 3, no_struct: int = 5):
    """A batch: n_r rows, n_h halo rows, m entries of random families; row `lonely` has no entry, and row `no_struct`
    has entries but none structural."""
    g = np.random.default_rng(seed)
    v = g.integers(0, n_r, m)
    v[v == lonely] = (lonely + 1) % n_r
    fam = g.integers(0, len(L3.FAMILIES), m)
    fam[v == no_struct] = g.integers(1, len(L3.FAMILIES), int((v == no_struct).sum()))
    if not (v == no_struct).any():
        v[0], fam[0] = no_struct, 1
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
    return inp, v, fam, lonely, no_struct


def _net(steps, form="KERN", fixed=False, seed=0):
    torch.manual_seed(seed)
    return L6.L6Rec(steps, form, fixed).double()


@pytest.mark.parametrize("form", L6.FORMS)
def test_parameters_are_shared_across_depth_and_no_input_names_a_dataset(form):
    counts = {(s, x): L6.parameter_count(L6.L6Rec(s, form, x)) for s in (1, 2, 3) for x in (False, True)}
    assert len(set(counts.values())) == 1
    names = {n for n, _ in L6.L6Rec(3, form).named_parameters()}
    assert not any("hop" in n or "step" in n or "dataset" in n for n in names)
    assert list(inspect.signature(L6.L6Rec.forward).parameters) == ["self", "bv", "nh", "q_rows", "ent_u", "ent_v", "ent_eps", "row_h", "return_weights"]
    arms = {p for p, a in L6.NEW.items() if a[0] == form}
    assert len({L6.parameter_count(L6.build(p)) for p in arms}) == 1


@pytest.mark.parametrize("form", L6.FORMS)
def test_the_weights_are_positive_and_sum_to_one_per_row_head_and_step_and_m_struct_is_the_renormalised_structural_part(form):
    inp, v, fam, lonely, no_struct = _toy()
    n_r = inp["bv"].shape[0]
    net = _net(3, form)
    with torch.no_grad():
        _out, per_step = net(**inp, return_weights=True)
        v_all = torch.as_tensor(np.concatenate([v, np.arange(n_r)]))
        for w in per_step:
            assert w.shape == (v_all.numel(), L6.HEADS) and bool((w > 0).all())
            torch.testing.assert_close(torch.zeros(n_r, L6.HEADS, dtype=w.dtype).index_add_(0, v_all, w),
                                       torch.ones(n_r, L6.HEADS, dtype=w.dtype), atol=1e-12, rtol=0)
            assert torch.equal(w[v.size + lonely], torch.ones(L6.HEADS, dtype=w.dtype))   # the self entry alone
        vv, struct, val, _fixed = net.edges(inp["nh"], inp["q_rows"], inp["ent_u"], inp["ent_v"], inp["ent_eps"], inp["row_h"])
        assert torch.equal(struct[:v.size], torch.as_tensor((fam == 0).astype(np.float64))) and not struct[v.size:].any()
        w = per_step[1]
        m, m_s = net.messages(w, val, vv, struct, n_r)
    for row in range(n_r):
        sel = (v_all == row).numpy()
        ws, vals, st = w.numpy()[sel], val.numpy()[sel], struct.numpy()[sel] == 1
        want = (ws[:, :, None] * vals).sum(0).reshape(-1)
        np.testing.assert_allclose(m[row].numpy(), want, rtol=1e-12, atol=1e-12)
        if st.any():
            want_s = ((ws[st][:, :, None] * vals[st]).sum(0) / ws[st].sum(0)[:, None]).reshape(-1)
            np.testing.assert_allclose(m_s[row].numpy(), want_s, rtol=1e-12, atol=1e-12)
        else:
            assert not m_s[row].any()
    assert not m_s[no_struct].any() and not m_s[lonely].any()


@pytest.mark.parametrize("steps", [1, 2, 3])
@pytest.mark.parametrize("fixed", [False, True])
def test_the_kernel_forms_step_messages_equal_its_compiled_form_and_the_output_is_invariant_to_entry_order(steps, fixed):
    inp, v, _fam, _l, _n = _toy(1)
    net = _net(steps, "KERN", fixed)
    with torch.no_grad():
        out = net(**inp)
        torch.testing.assert_close(out, net.compiled(**inp), atol=1e-10, rtol=1e-10)
        perm = torch.as_tensor(np.random.default_rng(5).permutation(v.size))
        shuffled = net(**dict(inp, ent_u=inp["ent_u"][perm], ent_v=inp["ent_v"][perm], ent_eps=inp["ent_eps"][perm]))
    torch.testing.assert_close(out, shuffled, atol=1e-10, rtol=1e-10)
    with pytest.raises(ValueError):
        _net(steps, "ATT").compiled(**inp)


@pytest.mark.parametrize("form", L6.FORMS)
def test_the_fixed_variant_uses_step_0s_weights_at_every_step_and_equals_re_weighting_at_one_step(form):
    inp, *_ = _toy(2)
    net_fix, net_rec = _net(3, form, True), _net(3, form, False)
    net_rec.load_state_dict(net_fix.state_dict())
    with torch.no_grad():
        out_f, w_f = net_fix(**inp, return_weights=True)
        out_r, w_r = net_rec(**inp, return_weights=True)
    assert len(w_f) == 3 and all(w is w_f[0] for w in w_f)
    assert torch.equal(w_f[0], w_r[0]) and not torch.equal(w_r[0], w_r[2])   # the state moves the re-weighting variant's kernel
    assert not torch.equal(out_f, out_r)
    one_f, one_r = _net(1, form, True), _net(1, form, False)
    one_r.load_state_dict(one_f.state_dict())
    with torch.no_grad():
        assert torch.equal(one_f(**inp), one_r(**inp))


# ── the read ─────────────────────────────────────────────────────────────────


def test_the_bands_contrasts_shares_and_grid_are_the_declared_ones():
    assert L6.band_l6(0.80, [0.55, 0.95], True) == "L6_HIGH"
    assert L6.band_l6(0.80, [0.45, 0.95], True) == "L6_MID"
    assert L6.band_l6(0.20, [0.00, 0.45], True) == "L6_LOW"
    assert L6.band_l6(0.90, [0.80, 0.99], False) == "NOT_READ"
    decl = L6.load_declaration()
    contrasts = {}
    for c_name, text in decl["statistics"]["contrasts"].items():
        a, b, fam = re.match(r"rho_bar\((\S+)\) - rho_bar\((\S+)\) on (r|e)\b", text).groups()
        contrasts[c_name] = (fam, a, b)
    assert L6.CONTRASTS == contrasts
    shares = {}
    for s_name, text in decl["statistics"]["shares"].items():
        a, b, c, e, fam = re.match(r"\(rho_bar\((\S+)\) - rho_bar\((\S+)\)\) / \(rho_bar\((\S+)\) - rho_bar\((\S+)\)\) on (r|e)\b", text).groups()
        shares[s_name] = (fam, a, b, c, e)
    assert L6.SHARES == shares
    grid = decl["probe"]["grid"]
    assert list(L6.GRID["r"]) == grid["r_k"] and list(L6.GRID["e"]) == grid["e_k"]
    refs = {"ref:other_seed"}
    assert all(a in L6.GRID[f] and b in L6.GRID[f] for f, a, b in contrasts.values())
    assert all(all(x in L6.GRID[f] or x in refs for x in (a, b, c, e)) for f, a, b, c, e in shares.values())
    assert L6.UNIT_ORDER[0] == ("r", "L6-3") and len(L6.UNIT_ORDER) == len(grid["r_k"]) + len(grid["e_k"])
    assert L6.PRIMARY == "L6-3" and decl["readings"]["primary_probe"].startswith("L6-3 on r_k")
    assert set(L6.REFIT) == {"B0-mlp", "L3-att"} and all(p in L3.PROBES for p in L6.REFIT)
    arms = decl["probe"]["arms"]
    assert set(arms) == set(L6.NEW)
    for p, (form, steps, fixed, objective) in L6.NEW.items():
        text = arms[p]
        assert text.startswith(f"{form}, T = {steps}") and (("step 0" in text) == fixed) and (("LIST" in text) == (objective == "LIST"))
    imap = set(decl["readings"]["interpretation_map"])
    assert set(L6.cross_dataset({})) == set(decl["readings"]["cross_dataset"])
    assert set(L6.STRATA_CONTRASTS) <= set(L6.CONTRASTS)
    assert {"l6_high", "composition_recovers", "composition_flat", "state_matters", "l6_above_attention", "edge_effect_l6"} <= imap


def _read_stub(points: dict, boots: dict, readable=True) -> dict:
    probes = {p: {"rho_bar": {"point": points[p], "ci": L0.ci(boots[p])}, "_rho_bar_boot": boots[p]} for p in points}
    return {"r": {"probes": probes, "readable_metrics": ["hit@1"] if readable else []},
            "e": {"probes": probes, "readable_metrics": ["hit@1"] if readable else []}}


def test_a_share_is_read_only_where_its_denominator_lies_above_0_and_the_contrasts_are_paired():
    g = np.random.default_rng(0)
    noise = g.normal(size=1000) * 0.01
    points = {"B0-mlp": 0.2, "L3-att": 0.6, "L6-1": 0.4, "L6-2": 0.5, "L6-3": 0.7, "L6-fix3": 0.45, "L6-list": 0.75,
              "L6-att1": 0.55, "L6-att3": 0.65, "ref:other_seed": 1.0}
    boots = {p: v + noise for p, v in points.items()}
    stub = _read_stub(points, boots)
    s = L6.shares_of(stub)
    assert s["l6_share"]["readable"] and abs(s["l6_share"]["point"] - 1.25) < 1e-12 and np.allclose(s["l6_share"]["ci"], [1.25, 1.25])
    assert s["gap_closed"]["readable"] and abs(s["gap_closed"]["point"] - 0.25) < 1e-12
    c = L6.contrasts_of(stub)
    assert abs(c["composition_3"]["point"] - 0.3) < 1e-12 and np.allclose(c["composition_3"]["ci"], [0.3, 0.3])
    assert set(L6.contrasts_of(stub, L6.STRATA_CONTRASTS)) == set(L6.STRATA_CONTRASTS)
    flat = dict(points, **{"L3-att": 0.2})
    s = L6.shares_of(_read_stub(flat, {p: v + noise for p, v in flat.items()}))
    assert not s["l6_share"]["readable"] and s["l6_share"]["point"] is None and s["l6_share"]["denominator"] is not None
    s = L6.shares_of(_read_stub(points, boots, readable=False))
    assert s["gap_closed"]["denominator"] is None and L6.contrasts_of(_read_stub(points, boots, readable=False))["state_kernel"]["ci"] is None


def _readings_stub(contrast_cis: dict, primary_band="L6_MID", e_band="L6_MID") -> dict:
    probes = {p: {"R2": {"point": 0.1}, "band": primary_band if p == L6.PRIMARY else "L6_MID"} for p in L6.GRID["r"]}
    e = {p: {"R2": {"point": 0.1}, "band": e_band if p == L6.PRIMARY else "L6_MID"} for p in L6.GRID["e"]}
    con = {c: {"point": None, "ci": contrast_cis.get(c)} for c in L6.CONTRASTS}
    return {"r": {"probes": probes}, "e": {"probes": e}, "reproducibility": {"r": {"mean": {"point": 0.9}}},
            "contrasts": con, "refit": {"r/B0-mlp": {"bit_identical": True, "max_abs_diff": 0.0}}, "repeat": None}


def test_the_interpretation_entries_follow_the_declared_intervals():
    got = L6.readings(_readings_stub({"composition_3": [0.05, 0.2], "l6_vs_attention": [0.01, 0.3], "state_kernel": [-0.1, 0.1],
                                      "edge_l6_vs_attention": [-0.2, 0.0]}, primary_band="L6_HIGH"))
    assert got["reading"] == "L6_HIGH"
    assert set(got["interpretation"]) == {"l6_high", "composition_recovers", "l6_above_attention", "l6_not_below_attention", "edge_effect_l6"}
    got = L6.readings(_readings_stub({"composition_3": [-0.1, 0.1], "l6_vs_attention": [-0.3, -0.01], "objective": [0.02, 0.1]}))
    assert set(got["interpretation"]) == {"composition_flat", "l6_below_attention", "objective_adds"} and got["flags"] == []
    assert L6.cross_dataset({"metaqa": {"interpretation": ["l6_above_attention"], "reading": "L6_MID"},
                             "2wiki": {"interpretation": ["l6_not_below_attention"], "reading": "L6_HIGH"},
                             "squad": {"interpretation": [], "reading": "NOT_READ"}}) == {
        "advisor_ideal": True, "positive_control_held": True, "negative_control_as_expected": True}
    assert L6.cross_dataset({"metaqa": {"interpretation": [], "reading": "L6_LOW"}})["advisor_ideal"] is None


def test_the_refit_comparison_is_bitwise(tmp_path):
    a = np.random.default_rng(0).normal(size=50)
    np.savez(tmp_path / "a.npz", **{"r|B0-mlp|0": a})
    np.savez(tmp_path / "b.npz", **{"r|B0-mlp|0": a.copy()})
    b = a.copy()
    b[7] = np.nextafter(b[7], np.inf)
    np.savez(tmp_path / "c.npz", **{"r|B0-mlp|0": b})
    assert L6.units_compare(tmp_path / "a.npz", tmp_path / "b.npz") == (True, 0.0)
    same, diff = L6.units_compare(tmp_path / "a.npz", tmp_path / "c.npz")
    assert same is False and 0 < diff < 1e-15


# ── probe, repeat, read and doc, end to end ─────────────────────────────────


def test_probe_repeat_read_and_doc_run_end_to_end_on_the_cpu(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 2)
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=120, sizes=(6, 14))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    _synthetic_l3(d3, sc, n_rel=5)
    L3.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d3, l0_dir=d0)   # level 3's filed units, for the refit comparison
    d6 = tmp_path / "l6" / "metaqa"
    L6.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d6, d3=d3, l0_dir=d0)
    meta = json.loads((d6 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    want = sorted([f"unit_{t}_{p}.npz" for t, p in L6.UNIT_ORDER] + [f"fitlog_{t}_{p}.json" for t, p in L6.UNIT_ORDER])
    assert sorted(meta["files_sha256"]) == want and meta["unit_order"][0] == "r/L6-3"
    assert {"scripts/mp_approx_l6.py", "scripts/mp_approx_l3.py", "scripts/mp_approx_l0.py"} <= set(meta["module_sha256"])
    assert "scripts/mp_approx_l5.py" not in meta["module_sha256"]
    probes = L0.load_probes(d6)
    assert set(probes) == {f"{t}/{p}/{k}" for t, p in L6.UNIT_ORDER for k in L0.SEEDS}
    for v in probes.values():
        assert v.shape == (sc.n_rows,) and np.isfinite(v).all()
        assert np.abs(L0.centre_rows(v, sc.ptr) - v).max() < 1e-5
    before = L0.sha256_file(d6 / "probes" / "unit_r_L6-3.npz")
    L6.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d6, d3=d3, l0_dir=d0)   # a restart skips every unit
    assert L0.sha256_file(d6 / "probes" / "unit_r_L6-3.npz") == before
    L6.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d6, d3=d3, l0_dir=d0, repeat=True)
    rep = json.loads((d6 / "repeat" / "repeat.json").read_text(encoding="utf-8"))
    assert rep["unit"] == "r_L6-3" and rep["bit_identical"] is True and rep["max_abs_diff"] == 0.0
    out = L6.stage_read("metaqa", log=_quiet, host=False, d=d6, d3=d3, l0_dir=d0)
    assert set(out["r"]["probes"]) == set(L6.GRID["r"]) | {"ref:twin", "ref:other_seed"}
    assert set(out["e"]["probes"]) == set(L6.GRID["e"]) | {"ref:no_edge", "ref:other_seed"}
    assert set(out["contrasts"]) == set(L6.CONTRASTS) and set(out["shares"]) == set(L6.SHARES)
    assert set(out["refit"]) == {f"{t}/{p}" for t, p in L6.UNIT_ORDER if p in L6.REFIT}
    assert all(v["bit_identical"] for v in out["refit"].values())   # level 3's code on the same inputs: its units exactly
    assert not any(f.startswith("L3_REFIT_DIFFERS") or f.startswith("REPEAT_DIFFERS") for f in out["flags"])
    assert all(set(s["contrasts"]) == set(L6.STRATA_CONTRASTS) for s in out["strata"].values())
    bands = [v["band"] for fam_ in ("r", "e") for v in out[fam_]["probes"].values()]
    assert all(b in ("L6_HIGH", "L6_MID", "L6_LOW", "NOT_READ") for b in bands)
    assert out["reading"] == out["r"]["probes"]["L6-3"]["band"]
    assert set(out["interpretation"]) <= set(L6.load_declaration()["readings"]["interpretation_map"])
    L6.stage_doc(log=_quiet, out_root=tmp_path / "l6", doc=tmp_path / "doc.md", datasets=("metaqa",), extra_deviations=["--gpus 0.16"])
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "## Readings" in text and "L6-fix3" in text and "host_gpu_det" in text and "--gpus 0.16" in text and "one hop" in text
    rec = json.loads((tmp_path / "l6" / "record.json").read_text(encoding="utf-8"))
    assert rec["datasets"]["metaqa"]["read_sha256"] == L0.sha256_file(d6 / "read.json")
    assert rec["cross_dataset"]["advisor_ideal"] is None and rec["cross_dataset"]["negative_control_as_expected"] is None
    assert not (L6.OUT / "hard_stops.json").exists() and not (L3.OUT / "hard_stops.json").exists()


def test_a_non_finite_prediction_is_a_hard_stop(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 1)
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=60, sizes=(4, 8))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    _synthetic_l3(d3, sc, n_rel=3)
    pd = L3.ProbeData(sc, d3, "cpu")

    def poisoned(pd_, probe, y, teacher, base, bv, stats, fit_q, val_q, test_q, seed):
        return np.full(int(sc.rows_of(test_q).sum()), np.inf), {"seed": seed}
    monkeypatch.setattr(L6, "fit_new", poisoned)
    with pytest.raises(SystemExit):
        L6.fit_units(pd, [("r", "L6-1")], tmp_path / "p", _quiet)
    assert not (tmp_path / "p" / "unit_r_L6-1.npz").exists()


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L6.DATASETS)
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
