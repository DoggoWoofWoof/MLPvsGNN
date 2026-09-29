"""MP-Approx level 3 (configs/mp_approx_l3.yaml#tests): the declaration, its pins (level 0's included) and the JL
digest; the placement spec against the qualification's host_gpu_det; the entries of packed batches and the halo rows;
the attention's normalisation, its uniform arm, a row without in-edges, scores at 0 and entry order; the listwise loss;
the bands and the contrasts; the mirror manifest and its stop; the identical-code check; the probe, repeat, read and doc
stages end to end on the CPU over a synthetic compile; and no held query in any sidecar."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
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
from test_mp_approx_l0 import _synthetic_sidecar  # noqa: E402  (level 0's synthetic sidecar)
from test_universal_v2_models import world  # noqa: E402,F401  (the toy substrate)


def _quiet(_s: str) -> None:
    pass


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L1, L3):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def det_request():
    """host_gpu_det's determinism request on the CPU, restored afterwards. Without it the CPU's index_put_ with
    accumulate (the backward of every gather here) adds floats atomically across threads once a batch passes 32768
    elements, and a refit can differ from its first fit in the last bit."""
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled())
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1], warn_only=before[2])


# ── the declaration, its pins and the placement ─────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256_level_0s_included(stops):
    decl = L3.load_declaration()
    assert decl["phase"] == "MP_APPROX_L3" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"].startswith("After matching candidate exposure and inference-time graph information")
    assert hashlib.sha256(L1.jl_matrix().tobytes()).hexdigest() == decl["inputs"]["jl_matrix_sha256"]
    sidecars = decl["inputs"]["level0"]["sidecars"]
    assert set(sidecars) == set(L3.DATASETS) == set(decl["inputs"]["level1"]["q_tilde_from"])
    here = all((ROOT / p["dir"] / "meta.json").exists() for p in sidecars.values()) and all((L1.OUT / n / "meta.json").exists() for n in L3.DATASETS)
    if not here:
        pytest.skip("level 0's or level 1's sidecars are not on this machine")
    L3.verify_inputs(decl, laptop=True)   # code pins, level 0's sidecars, level 1's metas, level 0's own inputs, the JL digest


def test_the_placement_spec_is_the_qualifications_host_gpu_det():
    import gpu_task_qualification as Q

    arm = Q.ARMS["host_gpu_det"]
    assert L3.SPEC == {k: arm[k] for k in ("host", "device", "threads", "mode", "tf32", "env")}
    assert L3.SPEC["mode"] == "det" and L3.SPEC["tf32"] is False and L3.SPEC["threads"] == 8
    decl = L3.load_declaration()
    assert decl["placement"]["probes_and_read"] == "host_gpu_det"
    assert set(L3.HOST_STAGES) == {"probe", "repeat", "read", "run"}


# ── the compile: entries and halo rows ──────────────────────────────────────


def _corrupt(ent: dict, key: str, halo: int, rows: int) -> dict:
    bad = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ent.items()}
    a = bad[key]
    if key == "ent_attr":
        a[0, 0] = np.float16(float(a[0, 0]) + 1.0)
    elif key == "ent_slots":
        a[0, 0] = 1 if a[0, 0] < 0 else -1
    elif key == "ent_row":
        a[0] = (a[0] + 1) % rows
    elif key == "ent_halo":
        a[0] = (a[0] + 1) % halo
    elif key == "ent_family":
        a[0] = (a[0] + 1) % len(L3.FAMILIES)
    return bad


@pytest.mark.parametrize("name", ["typed", "untyped"])
def test_the_entries_of_a_random_packed_batch_reproduce_its_kept_rows_and_the_halo_row_reproduces_b0(world, name):
    rng = np.random.default_rng(3)
    offset = int(world.ctx2[name].rel_offset)
    assert (offset >= 0) == (name == "typed")
    R = L1.jl_matrix()
    live_slots, seen = 0, 0
    for i in range(4):
        batch = world.fits[name].pack(np.array([i]))
        ei, ea, n = batch.edge_index.numpy(), batch.edge_attr.numpy(), int(batch.x.shape[0])
        assert ea.shape[1] == L3.SLOT0 + L3.K_REL and n > 3
        loc = np.sort(rng.choice(n, size=max(2, n // 3), replace=False))
        ent = L3.query_entries(ei, ea, loc, offset)
        keep = np.isin(ei[1], loc)
        assert ent["ent_row"].size == int(keep.sum())
        assert L3.check_attributes(ent, ei, ea, loc, offset, n) is None
        assert np.array_equal(ent["halo_loc"], np.union1d(loc, ei[0][keep]))
        assert np.array_equal(ent["halo_loc"][ent["row_halo"]], loc)
        live_slots += int((ent["ent_slots"] >= 0).sum())
        if keep.any():
            seen += 1
            for key, part in (("ent_attr", "attributes"), ("ent_slots", "slots"), ("ent_row", "v"), ("ent_halo", "u"),
                              ("ent_family", "family")):
                bad = _corrupt(ent, key, ent["halo_loc"].size, loc.size)
                assert L3.check_attributes(bad, ei, ea, loc, offset, n) == part
        x = (rng.normal(size=(n, 129)) * 3).astype(np.float32)
        zx = L0.column_z(x).astype(np.float32)
        E = rng.normal(size=(n, 1536)).astype(np.float32)
        halo = L3.halo_rows(x, zx, E, ent["halo_loc"], R)
        assert halo.shape == (ent["halo_loc"].size, L3.HALO_WIDTH) and halo.dtype == np.float16
        b0 = np.concatenate([np.sign(x) * np.log1p(np.abs(x)), zx], axis=1)[loc]
        assert np.array_equal(halo[ent["row_halo"], :L3.B0_WIDTH], L0.to_f16(b0, "b0"))
        assert np.array_equal(halo[:, L3.B0_WIDTH:], L0.to_f16(E[ent["halo_loc"]].astype(np.float64) @ R, "xr"))
    assert seen > 0
    assert (live_slots > 0) == (name == "typed")   # relation slots only where the dataset has typed relations


def test_a_malformed_family_or_slot_row_is_refused():
    with pytest.raises(ValueError):
        L3.family_of(np.array([[1.0, 1.0, 0.0]], dtype=np.float32))
    with pytest.raises(ValueError):
        L3.family_of(np.array([[0.0, 0.0, 0.0]], dtype=np.float32))
    assert L3.family_of(np.zeros((0, 3), dtype=np.float32)).size == 0
    with pytest.raises(ValueError):
        L3.slots_local(np.array([[0.5, -1, -1, -1]], dtype=np.float32), 0)
    with pytest.raises(ValueError):
        L3.slots_local(np.array([[2.0, -1, -1, -1]], dtype=np.float32), -1)
    with pytest.raises(ValueError):
        L3.slots_local(np.array([[2.0, -1, -1, -1]], dtype=np.float32), 3)
    assert L3.slots_local(np.array([[5.0, 3.0, -1, -1]], dtype=np.float32), 3).tolist() == [[2, 0, -1, -1]]
    with pytest.raises(ValueError):
        L3.query_entries(np.zeros((2, 0), dtype=np.int64), np.zeros((0, 12), dtype=np.float32), np.array([2, 1]), 0)


# ── the probe ────────────────────────────────────────────────────────────────


def _toy(seed: int = 0, n_r: int = 7, n_h: int = 12, m: int = 30, lonely: int = 3):
    """One query's rows: entries with random sources and targets (row `lonely` has none), fixed inputs of the declared widths."""
    g = np.random.default_rng(seed)
    v = g.integers(0, n_r, m)
    v[v == lonely] = (lonely + 1) % n_r
    u = g.integers(0, n_h, m)
    fam = g.integers(0, len(L3.FAMILIES), m)
    eps = np.zeros((m, L3.EPS_WIDTH), dtype=np.float32)
    eps[np.arange(m), fam] = 1.0
    a0 = len(L3.EPS_FAMILIES)
    eps[:, a0:a0 + L3.N_ATTR] = g.random((m, L3.N_ATTR))
    eps[:, a0 + L3.N_ATTR:] = g.normal(size=(m, L3.JL_DIM)) * (fam == 0)[:, None]

    def t(a, dt=torch.float32):
        return torch.as_tensor(a, dtype=dt)
    inp = {"bv": t(g.normal(size=(n_r, L3.B0_WIDTH))), "nh": t(g.normal(size=(n_h, L3.HALO_WIDTH))),
           "q_rows": t(np.repeat(g.normal(size=(1, L3.JL_DIM)), n_r, 0)), "ent_u": t(u, torch.long), "ent_v": t(v, torch.long),
           "ent_eps": t(eps), "row_h": t(g.permutation(n_h)[:n_r], torch.long)}
    return inp, v, lonely


def test_attention_sums_to_one_per_row_and_head_the_mean_is_uniform_and_a_row_without_in_edges_reads_only_itself():
    inp, v, lonely = _toy()
    n_r = inp["bv"].shape[0]
    v_all = np.concatenate([v, np.arange(n_r)])
    deg = np.bincount(v_all, minlength=n_r)
    uniform = torch.as_tensor(1.0 / deg[v_all], dtype=torch.float32)[:, None].expand(-1, L3.HEADS)
    for is_uniform in (False, True):
        torch.manual_seed(0)
        net = L3.L3Probe(is_uniform)
        with torch.no_grad():
            out, alpha = net(**inp, return_alpha=True)
        assert out.shape == (n_r,) and alpha.shape == (v_all.size, L3.HEADS)
        sums = torch.zeros(n_r, L3.HEADS).index_add_(0, torch.as_tensor(v_all), alpha)
        torch.testing.assert_close(sums, torch.ones(n_r, L3.HEADS), atol=1e-6, rtol=0)
        assert torch.equal(alpha[v.size + lonely], torch.ones(L3.HEADS))   # its only entry is its self entry
        if is_uniform:
            torch.testing.assert_close(alpha, uniform, atol=1e-7, rtol=0)
        else:
            assert not torch.allclose(alpha, uniform, atol=1e-4)            # the learned weighting is not uniform at init


def test_l3_att_with_every_score_at_zero_equals_l3_mean_on_the_same_weights():
    inp, _v, _ = _toy(seed=1)
    torch.manual_seed(1)
    att = L3.L3Probe(False)
    mean = L3.L3Probe(True)
    mean.load_state_dict(att.state_dict())
    with torch.no_grad():
        before = att(**inp)
        att.att.zero_()
        assert torch.equal(att(**inp), mean(**inp))
        assert not torch.equal(before, mean(**inp))


def test_the_forward_is_invariant_to_the_order_of_a_rows_entries():
    inp, v, _ = _toy(seed=2, m=40)
    torch.manual_seed(2)
    net = L3.L3Probe(False)
    perm = torch.as_tensor(np.random.default_rng(5).permutation(v.size))
    shuffled = dict(inp, ent_u=inp["ent_u"][perm], ent_v=inp["ent_v"][perm], ent_eps=inp["ent_eps"][perm])
    with torch.no_grad():
        torch.testing.assert_close(net(**inp), net(**shuffled), atol=1e-6, rtol=1e-6)


def test_the_list_loss_is_zero_at_r_plus_a_constant_and_positive_otherwise():
    g = np.random.default_rng(4)
    sizes = np.array([5, 1, 9, 3])
    seg = torch.as_tensor(np.repeat(np.arange(sizes.size), sizes))
    n = int(sizes.sum())
    teacher, base = torch.as_tensor(g.normal(size=n)), torch.as_tensor(g.normal(size=n))
    const = torch.as_tensor(g.normal(size=sizes.size) * 5)[seg]
    kl = L3.list_kl(teacher - base + const, teacher, base, seg, sizes.size)
    assert kl.shape == (sizes.size,) and float(kl.abs().max()) < 1e-12
    kl = L3.list_kl(torch.as_tensor(g.normal(size=n)), teacher, base, seg, sizes.size)
    assert bool((kl[torch.as_tensor(sizes > 1)] > 0).all()) and abs(float(kl[1])) < 1e-12   # one row has nothing to order


# ── the read ─────────────────────────────────────────────────────────────────


def test_the_bands_carry_the_l3_labels_and_the_contrasts_are_the_declared_pairs():
    assert L3.band_l3(0.80, [0.55, 0.95], True) == "L3_HIGH"
    assert L3.band_l3(0.80, [0.45, 0.95], True) == "L3_MID"
    assert L3.band_l3(0.75, [0.50, 0.90], True) == "L3_HIGH"
    assert L3.band_l3(0.20, [0.00, 0.45], True) == "L3_LOW"
    assert L3.band_l3(0.25, [0.00, 0.50], True) == "L3_LOW"
    assert L3.band_l3(0.20, [0.00, 0.55], True) == "L3_MID"
    assert L3.band_l3(0.90, [0.80, 0.99], False) == "NOT_READ"
    assert L3.band_l3(float("nan"), [0.0, 0.0], True) == "NOT_READ"
    decl = L3.load_declaration()
    declared = {}
    for c_name, text in decl["statistics"]["contrasts"].items():
        a, b, fam = re.match(r"rho_bar\((\S+)\) - rho_bar\((\S+)\) on (r|e)\b", text).groups()
        declared[c_name] = (fam, a, b)
    assert L3.CONTRASTS == declared
    assert all(a in L3.GRID[f] and b in L3.GRID[f] for f, a, b in declared.values())
    grid = decl["probe"]["grid"]
    assert list(L3.GRID["r"]) == grid["r_k"] and list(L3.GRID["e"]) == grid["e_k"]
    assert L3.UNIT_ORDER[0] == ("r", "L3-att") and len(L3.UNIT_ORDER) == len(grid["r_k"]) + len(grid["e_k"])
    assert L3.PRIMARY == "L3-att" and decl["readings"]["primary_probe"].startswith("L3-att on r_k")


# ── the mirror and the code check ────────────────────────────────────────────


def _synthetic_l3(d: Path, sc, seed: int = 0, n_rel: int = 5, chunk: int = 37) -> dict:
    """A compiled sidecar over a synthetic level 0 sidecar, made by the compile's own helpers: each query's pool is its
    U_q rows (level 0's x and zx) plus up to four outside nodes, with random packed edges of the three families and
    relation slots on structural edges; written in chunks and assembled as the compile assembles them. Returns each
    query's local indices for comparison."""
    g = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    chunks = d / "chunks"
    chunks.mkdir()
    R = L1.jl_matrix()
    x_all, zx_all, local = np.load(sc.dir / "x.npy"), np.load(sc.dir / "zx.npy"), np.load(sc.dir / "local.npy")
    offset = 0 if n_rel else -1
    n_chunks = math.ceil(sc.n_q / chunk)
    kept = {}
    for ci in range(n_chunks):
        idx = np.arange(ci * chunk, min((ci + 1) * chunk, sc.n_q))
        per = {k: [] for k in ("halo", "row_halo", "ent_query", "ent_row", "ent_halo", "ent_family", "ent_attr", "ent_slots")}
        pq = {k: [] for k in ("q_index", "q_uq", "q_halo", "q_entries", "q_edges", "q_tilde")}
        for j in idx:
            a, b = int(sc.ptr[j]), int(sc.ptr[j + 1])
            loc = local[a:b].astype(np.int64)
            n_pool = int(loc.max()) + 1 + int(g.integers(0, 5))
            x = g.normal(size=(n_pool, 129)).astype(np.float32)
            zx = g.normal(size=(n_pool, 129)).astype(np.float32)
            x[loc] = x_all[a:b]
            zx[loc] = zx_all[a:b]
            E = g.normal(size=(n_pool, 1536)).astype(np.float32)
            m = int(g.integers(0, 3 * n_pool))
            fam = np.sort(g.integers(0, len(L3.FAMILIES), m))
            ei = np.stack([g.integers(0, n_pool, m), g.integers(0, n_pool, m)]).astype(np.int64)
            ea = np.full((m, L3.SLOT0 + L3.K_REL), -1.0, dtype=np.float32)
            ea[:, :L3.SLOT0] = 0.0
            ea[np.arange(m), fam] = 1.0
            ea[:, len(L3.FAMILIES):L3.SLOT0] = g.random((m, L3.N_ATTR))
            if n_rel:
                for e in np.flatnonzero(fam == 0):
                    k = int(g.integers(0, L3.K_REL + 1))
                    ea[e, L3.SLOT0:L3.SLOT0 + k] = g.integers(0, n_rel, k) + offset
            ent = L3.query_entries(ei, ea, loc, offset)
            assert L3.check_attributes(ent, ei, ea, loc, offset, n_pool) is None
            per["halo"].append(L3.halo_rows(x, zx, E, ent["halo_loc"], R))
            per["row_halo"].append(ent["row_halo"].astype(np.int32))
            per["ent_query"].append(np.full(ent["ent_row"].size, j, dtype=np.int32))
            for key in ("ent_row", "ent_halo"):
                per[key].append(ent[key].astype(np.int32))
            for key in ("ent_family", "ent_attr", "ent_slots"):
                per[key].append(ent[key])
            pq["q_index"].append(int(j))
            pq["q_uq"].append(int(loc.size))
            pq["q_halo"].append(int(ent["halo_loc"].size))
            pq["q_entries"].append(np.bincount(ent["ent_family"].astype(np.int64), minlength=len(L3.FAMILIES)))
            pq["q_edges"].append(np.bincount(fam, minlength=len(L3.FAMILIES)))
            pq["q_tilde"].append(g.normal(size=L3.JL_DIM).astype(np.float32))
            kept[int(j)] = {key: ent[key] for key in ("row_halo", "ent_row", "ent_halo")}
        arrays = {k: np.concatenate(v) for k, v in per.items()}
        arrays.update({"q_index": np.asarray(pq["q_index"], dtype=np.int64), "q_uq": np.asarray(pq["q_uq"], dtype=np.int64),
                       "q_halo": np.asarray(pq["q_halo"], dtype=np.int64), "q_entries": np.stack(pq["q_entries"]).astype(np.int64),
                       "q_edges": np.stack(pq["q_edges"]).astype(np.int64), "q_tilde": np.stack(pq["q_tilde"]),
                       "q_proto_max_diff": np.zeros(idx.size), "q_proto_max_ratio": np.zeros(idx.size)})
        np.savez(chunks / f"c{ci:05d}.npz", **arrays)
    rel_jl = g.normal(size=(n_rel, L3.JL_DIM)).astype(np.float32)
    meta = L3.assemble_l3(chunks, d, n_chunks, sc, sc.n_q, rel_jl)
    (d / "qids.json").write_text(json.dumps(sc.qids), encoding="utf-8")
    meta["halo_b0_rows_compared"] = L3.check_halo_b0(d, sc, sc.n_q)
    meta.update({"dataset": d.name, "queries": sc.n_q, "n_relations": n_rel, "rel_offset": offset, "seconds_this_process": 0.0,
                 "proto_entries_compared": 0, "mismatches": 0, "script_lf_sha256": L0.lf_sha256(ROOT / "scripts" / "mp_approx_l3.py"),
                 "qids_sha256": L0.sha256_file(d / "qids.json")})
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    shutil.rmtree(chunks)
    return kept


def test_the_mirror_is_written_and_verified_and_a_changed_byte_stops(tmp_path, stops):
    l0 = tmp_path / "outputs" / "mp_approx_l0" / "squad"
    d3 = tmp_path / "outputs" / "mp_approx_l3" / "squad"
    _synthetic_sidecar(l0, n_q=40, sizes=(4, 8))
    _synthetic_l3(d3, L0.Sidecar(l0), n_rel=0)
    path = L3.write_mirror("squad", tmp_path, l0, d3)
    mirror = json.loads(path.read_text(encoding="utf-8"))
    assert len(mirror["files"]) == len(L3.L0_TRANSFER) + len(L3.L3_TRANSFER)
    assert all(k.startswith("outputs/mp_approx_l") and chr(92) not in k for k in mirror["files"])
    assert L3.verify_mirror("squad", tmp_path, l0, d3) == L0.sha256_file(path)
    z = np.load(l0 / "z.npy")
    z[0, 0] += 1.0
    np.save(l0 / "z.npy", z)
    with pytest.raises(SystemExit, match="HARD STOP"):
        L3.verify_mirror("squad", tmp_path, l0, d3)
    last = json.loads((stops / "hard_stops.json").read_text(encoding="utf-8"))[-1]
    assert list(last["files"]) == ["outputs/mp_approx_l0/squad/z.npy"]
    assert not (L3.OUT / "hard_stops.json").exists()


def test_identical_code_needs_one_sha256_per_module_equal_to_the_committed_file():
    committed = {"a.py": "1", "b.py": "2"}.get
    assert L3.code_problems({"x/probe": {"a.py": "1", "b.py": "2"}, "x/read": {"a.py": "1"}}, committed) == {}
    assert set(L3.code_problems({"x/probe": {"a.py": "1"}, "y/probe": {"a.py": "9"}}, committed)) == {"a.py"}
    assert L3.code_problems({"x/probe": {"a.py": "9"}}, committed)["a.py"]["committed"] == "1"
    assert L3.code_problems({"x/probe": {"c.py": "3"}}, committed)["c.py"]["committed"] is None


# ── probe, repeat, read and doc, end to end ──────────────────────────────────


def test_probe_repeat_read_and_doc_run_end_to_end_on_the_cpu(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 3)
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=120, sizes=(6, 14))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    kept = _synthetic_l3(d3, sc, n_rel=5)
    pd = L3.ProbeData(sc, d3, "cpu")
    for q in (0, 57, sc.n_q - 1):
        assert np.array_equal(pd.row_halo_in_q[sc.ptr[q]:sc.ptr[q + 1]], kept[q]["row_halo"])
        assert np.array_equal(pd.ent_row_in_q[pd.ent_ptr[q]:pd.ent_ptr[q + 1]], kept[q]["ent_row"])
        assert np.array_equal(pd.ent_halo_in_q[pd.ent_ptr[q]:pd.ent_ptr[q + 1]], kept[q]["ent_halo"])
    qs = np.array([57, 0, 3])
    B = pd.batch(qs, True)
    assert torch.equal(B["rows"], torch.as_tensor(sc.rows_of_queries(qs)))
    assert int(B["ent_v"].max()) < B["rows"].numel() and int(B["ent_u"].max()) < B["hs"].numel()
    assert torch.equal(pd.halo[B["hs"][B["row_h"]]].float()[:, :L3.B0_WIDTH], torch.as_tensor(L0.to_f16(L0.base_raw(sc, "B0", 0, sc.rows_of_queries(qs)), "b0")).float())
    L3.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d3, l0_dir=d0)
    meta = json.loads((d3 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    want = sorted([f"unit_{t}_{p}.npz" for t, p in L3.UNIT_ORDER] + [f"fitlog_{t}_{p}.json" for t, p in L3.UNIT_ORDER])
    assert sorted(meta["files_sha256"]) == want
    assert {"scripts/mp_approx_l3.py", "scripts/mp_approx_l0.py", "scripts/mp_approx_l1.py"} <= set(meta["module_sha256"])
    assert meta["module_sha256"]["scripts/mp_approx_l3.py"] == L0.lf_sha256(ROOT / "scripts" / "mp_approx_l3.py")
    probes = L0.load_probes(d3)
    assert set(probes) == {f"{t}/{p}/{k}" for t, p in L3.UNIT_ORDER for k in L0.SEEDS}
    for v in probes.values():
        assert v.shape == (sc.n_rows,) and np.isfinite(v).all()
        assert np.abs(L0.centre_rows(v, sc.ptr) - v).max() < 1e-5   # within-query centred
    L3.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d3, l0_dir=d0, repeat=True)
    rep = json.loads((d3 / "repeat" / "repeat.json").read_text(encoding="utf-8"))
    assert rep["unit"] == "r_L3-att" and rep["bit_identical"] is True and rep["max_abs_diff"] == 0.0
    out = L3.stage_read("metaqa", log=_quiet, host=False, d=d3, l0_dir=d0)
    assert set(out["r"]["probes"]) == set(L3.GRID["r"]) | {"ref:twin", "ref:other_seed"}
    assert set(out["e"]["probes"]) == set(L3.GRID["e"]) | {"ref:no_edge", "ref:other_seed"}
    assert set(out["contrasts"]) == set(L3.CONTRASTS)
    bands = [v["band"] for fam in ("r", "e") for v in out[fam]["probes"].values()]
    bands += [v["band"] for s in out["strata"].values() for v in s["probes"].values()]
    assert all(b in ("L3_HIGH", "L3_MID", "L3_LOW", "NOT_READ") for b in bands)
    assert all(set(s["probes"]) == set(L3.STRATA_PROBES) for s in out["strata"].values())
    assert out["reading"] == out["r"]["probes"]["L3-att"]["band"] and out["repeat"]["bit_identical"] is True
    assert set(out["interpretation"]) <= set(L3.load_declaration()["readings"]["interpretation_map"])
    if out["r"]["readable_metrics"]:
        rp, c = out["r"]["probes"], out["contrasts"]["query_weighting"]
        assert abs(c["point"] - (rp["L3-att"]["rho_bar"]["point"] - rp["L3-mean"]["rho_bar"]["point"])) < 1e-12
    L3.stage_doc(log=_quiet, out_root=tmp_path / "l3", doc=tmp_path / "doc.md", datasets=("metaqa",), extra_deviations=["--gpus 0.33"])
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "## Readings" in text and "L3-att" in text and "host_gpu_det" in text and "--gpus 0.33" in text
    rec = json.loads((tmp_path / "l3" / "record.json").read_text(encoding="utf-8"))
    assert rec["datasets"]["metaqa"]["read_sha256"] == L0.sha256_file(d3 / "read.json")
    assert not (L3.OUT / "hard_stops.json").exists() and not (L0.OUT / "hard_stops.json").exists()


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L3.DATASETS)
def test_no_held_query_id_appears_in_any_sidecar(name):
    side = L3.OUT / name / "qids.json"
    if not side.exists():
        pytest.skip(f"{name}: not compiled yet")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"][name]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    compiled = json.loads(side.read_text(encoding="utf-8"))
    assert compiled and set(compiled) <= gate
    assert compiled == json.loads((L0.OUT / name / "qids.json").read_text(encoding="utf-8"))
