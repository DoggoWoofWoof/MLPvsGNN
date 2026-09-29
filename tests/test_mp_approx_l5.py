"""MP-Approx level 5 (configs/mp_approx_l5.yaml#tests): the declaration and its pins; the placement spec; contract column
0; every moment family against a direct per-row computation, the empty row, entry-order invariance and the tie rule, and
the 3366 names; the factorised kernel's positivity, normalisation and compiled form; the bands, contrasts and shares; the
bitwise refit comparison; the moments, probe, repeat, read and doc stages end to end on the CPU over a synthetic level 3
sidecar; and no held query in any sidecar."""

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
from test_mp_approx_l0 import _synthetic_sidecar  # noqa: E402  (level 0's synthetic sidecar)
from test_mp_approx_l3 import _synthetic_l3  # noqa: E402  (level 3's synthetic compile)


def _quiet(_s: str) -> None:
    pass


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L1, L3, L5):
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
    decl = L5.load_declaration()
    assert decl["phase"] == "MP_APPROX_L5" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L3.load_declaration()["registered_question"]
    assert L5.SPEC == L3.SPEC and L5.SPEC["mode"] == "det" and L5.SPEC["threads"] == 8
    assert set(L5.HOST_STAGES) == {"moments", "probe", "repeat", "read", "run"}
    sidecars = {n: (decl["inputs"]["level0"]["sidecars"][n], decl["inputs"]["level3"]["sidecars"][n]) for n in L5.DATASETS}
    here = all((ROOT / p0["dir"] / "meta.json").exists() and (ROOT / p3["dir"] / "probes" / "probe_meta.json").exists() for p0, p3 in sidecars.values())
    if not here:
        pytest.skip("level 0's or level 3's sidecars are not on this machine")
    L5.verify_inputs(decl)


def test_contract_column_0_is_dense_cos_and_halo_column_129_is_its_pool_z_score():
    assert L5.ZCOS == L3.B0_WIDTH // 2 == 129 and L5.HALO_NAMES[L5.ZCOS] == "z000"
    assert "halo column 129" in decl_text("zcos_column")
    for name in L5.DATASETS:
        meta = L0.OUT / name / "meta.json"
        if not meta.exists():
            pytest.skip(f"{name}: level 0's sidecar is not on this machine")
        columns = json.loads(meta.read_text(encoding="utf-8"))["columns"]
        assert len(columns) == 129 and columns[0] == "dense_cos"


def decl_text(key: str) -> str:
    return " ".join(str(L5.load_declaration()["inputs"][key]).split())


# ── the moments ──────────────────────────────────────────────────────────────


def _chunk(seed: int = 0, n: int = 9, m: int = 60, empty=(2, 7), ties: bool = False):
    """One chunk's inputs: random halo rows (a few repeated zc values when ties), families, attributes with rel_mask on
    some structural entries, rel_text, and rows in `empty` without an entry."""
    g = np.random.default_rng(seed)
    row = g.integers(0, n, m)
    for e in empty:
        row[row == e] = (e + 1) % n
    H = g.normal(size=(m, L5.HALO_WIDTH))
    if ties:
        H[:, L5.ZCOS] = g.integers(0, 3, m).astype(np.float64)
    fam = g.integers(0, len(L5.FAMILIES), m)
    attr = g.random((m, L5.N_ATTR))
    attr[:, L5.RM] = ((fam == 0) & (g.random(m) < 0.6)).astype(np.float64)
    rel_text = g.normal(size=(m, L5.JL_DIM)) * (fam == 0)[:, None]
    self_xr = g.normal(size=(n, L5.JL_DIM))
    return H, row, fam, attr, rel_text, self_xr


def _direct(H, row, fam, attr, rel_text, self_xr) -> np.ndarray:
    """The declared moments, one row at a time, in entry order."""
    n = self_xr.shape[0]
    out = np.zeros((n, L5.N_MOMENTS))
    for v in range(n):
        out[v, L5.SL["self_xr"]] = self_xr[v]
        idx = np.flatnonzero(row == v)
        if idx.size == 0:
            continue
        X = H[idx]
        out[v, L5.SL["full_mean"]] = X.mean(0)
        out[v, L5.SL["full_std"]] = X.std(0)
        out[v, L5.SL["full_max"]] = X.max(0)
        for i, f in enumerate(L5.FAMILIES):
            if (fam[idx] == i).any():
                out[v, L5.SL[f"family_mean_{f}"]] = X[fam[idx] == i].mean(0)
        zc = X[:, L5.ZCOS]
        for tau, block in zip(L5.TAUS, ("query_kernel_tau1", "query_kernel_tau0.25")):
            w = np.exp((zc - zc.max()) / tau)
            out[v, L5.SL[block]] = (w[:, None] * X).sum(0) / w.sum()
        out[v, L5.SL["top"]] = X[int(np.argmax(zc))]   # argmax returns the first maximum: entry order on a tie
        rel = (fam[idx] == 0) & (attr[idx, L5.RM] == 1)
        if rel.any():
            rc = attr[idx[rel], L5.RC]
            w = np.exp((rc - rc.max()) / L5.REL_TAU)
            out[v, L5.SL["relation_kernel"]] = (w[:, None] * X[rel]).sum(0) / w.sum()
        eps = np.concatenate([np.eye(4)[fam[idx]], attr[idx], rel_text[idx]], axis=1)
        out[v, L5.SL["eps_mean"]] = eps.mean(0)
        out[v, L5.SL["attribute_max"]] = attr[idx].max(0)
        out[v, L5.SL["counts"]] = np.log1p([idx.size] + [(fam[idx] == i).sum() for i in range(3)])
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_every_moment_family_equals_a_direct_per_row_computation_and_an_empty_row_keeps_only_its_xr(seed):
    args = _chunk(seed, ties=seed == 2)
    got = L5.chunk_moments(*args)
    np.testing.assert_allclose(got, _direct(*args), rtol=1e-12, atol=1e-12)
    for e in (2, 7):
        assert np.array_equal(got[e, L5.SL["self_xr"]], args[5][e])
        assert not got[e, L5.SL["self_xr"].stop:].any()
    empty = L5.chunk_moments(args[0][:0], args[1][:0], args[2][:0], args[3][:0], args[4][:0], args[5])
    assert np.array_equal(empty[:, L5.SL["self_xr"]], args[5]) and not empty[:, 64:].any()


def test_the_moments_are_invariant_to_entry_order_and_the_top_row_takes_the_first_on_a_tie():
    H, row, fam, attr, rel_text, self_xr = _chunk(3)
    perm = np.random.default_rng(9).permutation(row.size)
    a = L5.chunk_moments(H, row, fam, attr, rel_text, self_xr)
    b = L5.chunk_moments(H[perm], row[perm], fam[perm], attr[perm], rel_text[perm], self_xr)
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    # a tie: two entries of row 0 share the largest zc; the first in entry order is the top row
    H2 = np.zeros((3, L5.HALO_WIDTH))
    H2[:, L5.ZCOS] = [1.0, 2.0, 2.0]
    H2[:, 0] = [10.0, 20.0, 30.0]
    got = L5.chunk_moments(H2, np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.zeros((3, 5)), np.zeros((3, 64)), np.zeros((1, 64)))
    assert got[0, L5.SL["top"].start] == 20.0
    got = L5.chunk_moments(H2[::-1].copy(), np.zeros(3, dtype=np.int64), np.zeros(3, dtype=np.int64), np.zeros((3, 5)), np.zeros((3, 64)), np.zeros((1, 64)))
    assert got[0, L5.SL["top"].start] == 30.0


def test_there_are_3366_moment_columns_with_the_declared_blocks_and_widths():
    assert L5.N_MOMENTS == 3366 == len(set(L5.MOMENT_NAMES))
    widths = {b: len(c) for b, c in L5.MOMENT_BLOCKS}
    declared = L5.load_declaration()["moments"]
    assert widths["self_xr"] == 64 and widths["eps_mean"] == L3.EPS_WIDTH == 73 and widths["attribute_max"] == 5 and widths["counts"] == 4
    assert all(widths[b] == L5.HALO_WIDTH for b in ("full_mean", "full_std", "full_max", "family_mean_structural", "family_mean_ner",
                                                     "family_mean_knn", "query_kernel_tau1", "query_kernel_tau0.25", "top", "relation_kernel"))
    assert [b for b, _ in L5.MOMENT_BLOCKS][0] == "self_xr" and "(258 + 3366)" in L5.load_declaration()["probe"]["L5_mom"]
    assert set(declared) >= {"self", "full", "family", "query_kernels", "top", "relation_kernel", "eps_mean", "attribute_max", "counts"}


def test_rel_text_is_level_3s_mean_of_the_valid_slots_times_8():
    g = np.random.default_rng(0)
    rel_jl = g.normal(size=(6, L5.JL_DIM)).astype(np.float32)
    slots = np.array([[0, 3, -1, -1], [-1, -1, -1, -1], [5, 5, 1, 2]])
    got = L5.rel_text_of(slots, rel_jl)
    want = np.stack([(rel_jl[0].astype(np.float64) + rel_jl[3]) / 2 * 8, np.zeros(L5.JL_DIM),
                     (2 * rel_jl[5].astype(np.float64) + rel_jl[1] + rel_jl[2]) / 4 * 8])
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert not L5.rel_text_of(slots, np.zeros((0, L5.JL_DIM), dtype=np.float32)).any()


# ── the factorised kernel ────────────────────────────────────────────────────


def _toy(seed: int = 0, n_r: int = 7, n_h: int = 12, m: int = 30, lonely: int = 3):
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
    return {"bv": t(g.normal(size=(n_r, L3.B0_WIDTH))), "nh": t(g.normal(size=(n_h, L3.HALO_WIDTH))),
            "q_rows": t(np.repeat(g.normal(size=(1, L3.JL_DIM)), n_r, 0)), "ent_u": t(g.integers(0, n_h, m), torch.long),
            "ent_v": t(v, torch.long), "ent_eps": t(eps), "row_h": t(g.permutation(n_h)[:n_r], torch.long)}, v, lonely


def test_the_kernel_is_positive_sums_to_one_per_row_and_head_and_equals_its_compiled_form():
    inp, v, lonely = _toy()
    n_r = inp["bv"].shape[0]
    torch.manual_seed(0)
    net = L5.L5Kern().double()
    with torch.no_grad():
        out, alpha = net(**inp, return_alpha=True)
        compiled = net.compiled(**inp)
    v_all = torch.as_tensor(np.concatenate([v, np.arange(n_r)]))
    assert alpha.shape == (v_all.numel(), L5.HEADS) and bool((alpha > 0).all())
    torch.testing.assert_close(torch.zeros(n_r, L5.HEADS, dtype=alpha.dtype).index_add_(0, v_all, alpha),
                               torch.ones(n_r, L5.HEADS, dtype=alpha.dtype), atol=1e-12, rtol=0)
    assert torch.equal(alpha[v.size + lonely], torch.ones(L5.HEADS, dtype=alpha.dtype))
    torch.testing.assert_close(out, compiled, atol=1e-10, rtol=1e-10)
    perm = torch.as_tensor(np.random.default_rng(5).permutation(v.size))
    with torch.no_grad():
        shuffled = net(**dict(inp, ent_u=inp["ent_u"][perm], ent_v=inp["ent_v"][perm], ent_eps=inp["ent_eps"][perm]))
    torch.testing.assert_close(out, shuffled, atol=1e-10, rtol=1e-10)


# ── the read ─────────────────────────────────────────────────────────────────


def test_the_bands_contrasts_and_shares_are_the_declared_ones():
    assert L5.band_l5(0.80, [0.55, 0.95], True) == "L5_HIGH"
    assert L5.band_l5(0.80, [0.45, 0.95], True) == "L5_MID"
    assert L5.band_l5(0.20, [0.00, 0.45], True) == "L5_LOW"
    assert L5.band_l5(0.90, [0.80, 0.99], False) == "NOT_READ"
    decl = L5.load_declaration()
    contrasts = {}
    for c_name, text in decl["statistics"]["contrasts"].items():
        a, b, fam = re.match(r"rho_bar\((\S+)\) - rho_bar\((\S+)\) on (r|e)\b", text).groups()
        contrasts[c_name] = (fam, a, b)
    assert L5.CONTRASTS == contrasts
    shares = {}
    for s_name, text in decl["statistics"]["shares"].items():
        a, b, c, e, fam = re.match(r"\(rho_bar\((\S+)\) - rho_bar\((\S+)\)\) / \(rho_bar\((\S+)\) - rho_bar\((\S+)\)\) on (r|e)\b", text).groups()
        shares[s_name] = (fam, a, b, c, e)
    assert L5.SHARES == shares
    grid = decl["probe"]["grid"]
    assert list(L5.GRID["r"]) == grid["r_k"] and list(L5.GRID["e"]) == grid["e_k"]
    assert all(a in L5.GRID[f] and b in L5.GRID[f] for f, a, b in contrasts.values())
    assert L5.UNIT_ORDER[0] == ("r", "L5-mom") and len(L5.UNIT_ORDER) == len(grid["r_k"]) + len(grid["e_k"])
    assert L5.PRIMARY == "L5-mom" and decl["readings"]["primary_probe"].startswith("L5-mom on r_k")
    assert set(L5.REFIT) == {"B0-mlp", "L3-mean", "L3-att"} and all(p in L3.PROBES for p in L5.REFIT)


def _read_stub(points: dict, boots: dict, readable=True) -> dict:
    probes = {p: {"rho_bar": {"point": points[p], "ci": L0.ci(boots[p])}, "_rho_bar_boot": boots[p]} for p in points}
    return {"r": {"probes": probes, "readable_metrics": ["hit@1"] if readable else []},
            "e": {"probes": probes, "readable_metrics": ["hit@1"] if readable else []}}


def test_a_share_is_read_only_where_its_denominator_lies_above_0_and_the_contrasts_are_paired():
    g = np.random.default_rng(0)
    noise = g.normal(size=1000) * 0.01
    points = {"B0-mlp": 0.2, "L3-mean": 0.3, "L3-att": 0.6, "L5-mom": 0.5, "L5-list": 0.55, "L5-kern": 0.58}
    boots = {p: v + noise for p, v in points.items()}
    stub = _read_stub(points, boots)
    s = L5.shares_of(stub)
    assert s["moment_share"]["readable"] and abs(s["moment_share"]["point"] - 0.75) < 1e-12
    assert np.allclose(s["moment_share"]["ci"], [0.75, 0.75])   # a shared noise cancels in both differences
    c = L5.contrasts_of(stub)
    assert abs(c["moments_vs_attention"]["point"] + 0.1) < 1e-12 and np.allclose(c["moments_vs_attention"]["ci"], [-0.1, -0.1])
    flat = dict(points, **{"L3-att": 0.2})
    s = L5.shares_of(_read_stub(flat, {p: v + noise for p, v in flat.items()}))
    assert not s["moment_share"]["readable"] and s["moment_share"]["point"] is None and s["moment_share"]["denominator"] is not None
    s = L5.shares_of(_read_stub(points, boots, readable=False))
    assert s["kernel_share"]["denominator"] is None and L5.contrasts_of(_read_stub(points, boots, readable=False))["kernel_vs_attention"]["ci"] is None


def test_the_refit_comparison_is_bitwise(tmp_path):
    a = np.random.default_rng(0).normal(size=50)
    np.savez(tmp_path / "a.npz", **{"r|B0-mlp|0": a})
    np.savez(tmp_path / "b.npz", **{"r|B0-mlp|0": a.copy()})
    b = a.copy()
    b[7] = np.nextafter(b[7], np.inf)
    np.savez(tmp_path / "c.npz", **{"r|B0-mlp|0": b})
    assert L5.units_compare(tmp_path / "a.npz", tmp_path / "b.npz") == (True, 0.0)
    same, diff = L5.units_compare(tmp_path / "a.npz", tmp_path / "c.npz")
    assert same is False and 0 < diff < 1e-15


# ── moments, probe, repeat, read and doc, end to end ─────────────────────────


def test_moments_probe_repeat_read_and_doc_run_end_to_end_on_the_cpu(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 3)
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=120, sizes=(6, 14))
    meta0 = json.loads((d0 / "meta.json").read_text(encoding="utf-8"))
    meta0["columns"] = ["dense_cos"] + [f"c{j}" for j in range(1, 129)]
    (d0 / "meta.json").write_text(json.dumps(meta0), encoding="utf-8")
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    _synthetic_l3(d3, sc, n_rel=5)
    L3.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d3, l0_dir=d0)   # level 3's filed units, for the refit comparison
    d5 = tmp_path / "l5" / "metaqa"
    mmeta = L5.stage_moments("metaqa", log=_quiet, host=False, d=d5, d3=d3, l0_dir=d0)
    M = np.load(d5 / "moments.npy")
    assert M.shape == (sc.n_rows, L5.N_MOMENTS) and M.dtype == np.float16 and mmeta["moments_sha256"] == L0.sha256_file(d5 / "moments.npy")
    pd = L3.ProbeData(sc, d3, "cpu")
    halo = np.load(d3 / "halo.npy").astype(np.float64)
    fam, attr = np.load(d3 / "ent_family.npy").astype(np.int64), np.load(d3 / "ent_attr.npy").astype(np.float64)
    slots, rel_jl = np.load(d3 / "ent_slots.npy"), np.load(d3 / "rel_jl.npy")
    for q in (0, 57, sc.n_q - 1):   # each query's stored moments are the direct computation, through to_f16
        es = slice(int(pd.ent_ptr[q]), int(pd.ent_ptr[q + 1]))
        hq = int(pd.halo_ptr[q])
        want = _direct(halo[hq + pd.ent_halo_in_q[es]], pd.ent_row_in_q[es], fam[es], attr[es], L5.rel_text_of(slots[es], rel_jl),
                       halo[hq + pd.row_halo_in_q[sc.ptr[q]:sc.ptr[q + 1]], L5.B0_WIDTH:])
        np.testing.assert_allclose(M[sc.ptr[q]:sc.ptr[q + 1]].astype(np.float64), want.astype(np.float32).astype(np.float16), rtol=2e-3, atol=1e-4)
    assert L5.stage_moments("metaqa", log=_quiet, host=False, d=d5, d3=d3, l0_dir=d0)["utc"] == mmeta["utc"]   # skipped on a restart
    L5.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d5, d3=d3, l0_dir=d0)
    meta = json.loads((d5 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    want = sorted([f"unit_{t}_{p}.npz" for t, p in L5.UNIT_ORDER] + [f"fitlog_{t}_{p}.json" for t, p in L5.UNIT_ORDER])
    assert sorted(meta["files_sha256"]) == want and meta["moments_sha256"] == mmeta["moments_sha256"]
    assert {"scripts/mp_approx_l5.py", "scripts/mp_approx_l3.py", "scripts/mp_approx_l0.py"} <= set(meta["module_sha256"])
    probes = L0.load_probes(d5)
    assert set(probes) == {f"{t}/{p}/{k}" for t, p in L5.UNIT_ORDER for k in L0.SEEDS}
    for v in probes.values():
        assert v.shape == (sc.n_rows,) and np.isfinite(v).all()
        assert np.abs(L0.centre_rows(v, sc.ptr) - v).max() < 1e-5
    L5.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d5, d3=d3, l0_dir=d0, repeat=True)
    rep = json.loads((d5 / "repeat" / "repeat.json").read_text(encoding="utf-8"))
    assert rep["unit"] == "r_L5-mom" and rep["bit_identical"] is True and rep["moments_identical"] is True
    assert not (d5 / "repeat" / "moments.npy").exists()
    out = L5.stage_read("metaqa", log=_quiet, host=False, d=d5, d3=d3, l0_dir=d0)
    assert set(out["r"]["probes"]) == set(L5.GRID["r"]) | {"ref:twin", "ref:other_seed"}
    assert set(out["e"]["probes"]) == set(L5.GRID["e"]) | {"ref:no_edge", "ref:other_seed"}
    assert set(out["contrasts"]) == set(L5.CONTRASTS) and set(out["shares"]) == set(L5.SHARES)
    assert set(out["refit"]) == {f"{t}/{p}" for t, p in L5.UNIT_ORDER if p in L5.REFIT}
    assert all(v["bit_identical"] for v in out["refit"].values())   # level 3's code on the same inputs: its units exactly
    assert not any(f.startswith("L3_REFIT_DIFFERS") or f.startswith("MOMENTS_DIFFER") for f in out["flags"])
    bands = [v["band"] for fam_ in ("r", "e") for v in out[fam_]["probes"].values()]
    assert all(b in ("L5_HIGH", "L5_MID", "L5_LOW", "NOT_READ") for b in bands)
    assert out["reading"] == out["r"]["probes"]["L5-mom"]["band"]
    assert set(out["interpretation"]) <= set(L5.load_declaration()["readings"]["interpretation_map"])
    L5.stage_doc(log=_quiet, out_root=tmp_path / "l5", doc=tmp_path / "doc.md", datasets=("metaqa",), extra_deviations=["--gpus 0.16"])
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "## Readings" in text and "L5-kern" in text and "host_gpu_det" in text and "--gpus 0.16" in text
    rec = json.loads((tmp_path / "l5" / "record.json").read_text(encoding="utf-8"))
    assert rec["datasets"]["metaqa"]["read_sha256"] == L0.sha256_file(d5 / "read.json")
    assert not (L5.OUT / "hard_stops.json").exists() and not (L3.OUT / "hard_stops.json").exists()


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L5.DATASETS)
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
    mm = L5.OUT / name / "moments_meta.json"
    if mm.exists():
        assert json.loads(mm.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)
