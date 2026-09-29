"""MP-Approx level 4 (configs/mp_approx_l4.yaml#tests): the declaration and its pins; the placement spec; the seed-walk and
subtree sketches, the walk totals and the path-query mean and max against a brute-force enumeration of the walks, on
small random typed graphs and on level 3's synthetic packed batches; edge-order and slot-order invariance; a reversed
two-edge path; the 267 names and the unreached row; L4-att's initialisation and its forward against L3-att's; the graph
integrity against level 3's stored entries; the bands, contrasts, readings and the bitwise refit comparison; the mirror;
the probe, repeat, read and doc stages end to end on the CPU over a synthetic sketch; and no held query in any sidecar."""

from __future__ import annotations

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
import mp_approx_l4 as L4  # noqa: E402
from test_mp_approx_l0 import _synthetic_sidecar  # noqa: E402  (level 0's synthetic sidecar)
from test_mp_approx_l3 import _synthetic_l3  # noqa: E402  (level 3's synthetic compile)


def _quiet(_s: str) -> None:
    pass


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L1, L3, L4):
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
    decl = L4.load_declaration()
    assert decl["phase"] == "MP_APPROX_L4" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L3.load_declaration()["registered_question"]
    assert L4.SPEC == L3.SPEC and L4.SPEC["mode"] == "det" and L4.SPEC["threads"] == 8
    assert set(L4.HOST_STAGES) == {"probe", "repeat", "read", "run"} and decl["placement"]["probes_and_read"] == "host_gpu_det"
    assert set(L4.TRANSFER) == {"paths.npy", "qids.json", "meta.json"}
    sidecars = {n: (decl["inputs"]["level0"]["sidecars"][n], decl["inputs"]["level3"]["sidecars"][n]) for n in L4.DATASETS}
    here = all((ROOT / p0["dir"] / "meta.json").exists() and (ROOT / p3["dir"] / "probes" / "probe_meta.json").exists() for p0, p3 in sidecars.values())
    if not here:
        pytest.skip("level 0's or level 3's sidecars are not on this machine")
    L4.verify_inputs(decl)


# ── the sketches against a brute-force enumeration ──────────────────────────


def _graph(seed: int, n: int = 8, m: int = 24, n_rel: int = 4, seeds=(0, 3, 3)):
    """A small random typed graph with a duplicated edge and a self-loop; slots on structural edges only, rel_compat on
    structural edges only; the seeds listed with a duplicate."""
    g = np.random.default_rng(seed)
    u, v = g.integers(0, n, m), g.integers(0, n, m)
    u, v = np.concatenate([u, u[:1], [1]]), np.concatenate([v, v[:1], [1]])
    m = u.size
    fam = g.integers(0, len(L4.FAMILIES), m)
    fam[-2] = fam[0]
    fwd = (g.random(m) < 0.5).astype(np.float64)
    bwd = (g.random(m) < 0.5).astype(np.float64)
    slots = np.full((m, L4.K_REL), -1, dtype=np.int64)
    for e in np.flatnonzero(fam == 0):
        k = int(g.integers(0, L4.K_REL + 1))
        slots[e, :k] = g.integers(0, n_rel, k)
    slots[-2] = slots[0]
    fwd[-2], bwd[-2] = fwd[0], bwd[0]
    rc = np.where(fam == 0, g.random(m), 0.0)
    rc[-2] = rc[0]
    h, s = L4.position_hashes(L4.edge_tokens(fam, fwd, bwd, slots))
    return u, v, h, s, rc, n, np.asarray(seeds, dtype=np.int64)


def _walks(u, v, starts: set, length: int) -> list:
    """Every walk of `length` edges (a list of edge indices) whose first edge leaves a node in `starts`."""
    out = [[e] for e in range(u.size) if int(u[e]) in starts]
    for _ in range(length - 1):
        out = [w + [e] for w in out for e in range(u.size) if u[e] == v[w[-1]]]
    return out


def _slog(a):
    return np.sign(a) * np.log1p(np.abs(a))


def _brute(u, v, h, s, rc, n, seeds) -> np.ndarray:
    """The declared block, walk by walk: each walk adds its sign to the bucket of its summed position hashes."""
    out = np.zeros((n, L4.N_PATHS))
    walk, tree = out[:, L4.PSL["seed_walk_sketch"]], out[:, L4.PSL["subtree_sketch"]]
    tot, pm = out[:, L4.PSL["walk_totals"]], out[:, L4.PSL["path_query_match"]]
    seedset = set(np.unique(seeds).tolist())
    for i, length in enumerate(L4.WALK_L):
        S, N, C, M = np.zeros((n, L4.WALK_B)), np.zeros(n), np.zeros(n), np.full(n, -np.inf)
        for w in _walks(u, v, seedset, length):
            end = int(v[w[-1]])
            S[end, sum(int(h[p, e]) for p, e in enumerate(w)) % L4.WALK_B] += np.prod([s[p, e] for p, e in enumerate(w)])
            N[end] += 1
            r = sum(float(rc[e]) for e in w)
            C[end] += r / length
            M[end] = max(M[end], r / length)
        walk[:, i * L4.WALK_B:(i + 1) * L4.WALK_B] = _slog(S)
        tot[:, i] = np.log1p(N)
        pm[:, i] = np.where(N > 0, C / np.maximum(N, 1), 0.0)
        pm[:, len(L4.WALK_L) + i] = np.where(N > 0, M, 0.0)
    for i, length in enumerate(L4.TREE_L):
        T, A = np.zeros((n, L4.TREE_B)), np.zeros(n)
        for w in _walks(u, v, set(range(n)), length):
            end = int(v[w[-1]])
            T[end, sum(int(h[p, e]) for p, e in enumerate(w)) % L4.TREE_B] += np.prod([s[p, e] for p, e in enumerate(w)])
            A[end] += 1
        tree[:, i * L4.TREE_B:(i + 1) * L4.TREE_B] = _slog(T)
        tot[:, len(L4.WALK_L) + i] = np.log1p(A)
    return out


COUNTS = slice(0, L4.PSL["walk_totals"].stop)   # the sketches and the totals: integers, so exactly equal


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_the_sketches_totals_and_path_query_match_equal_a_brute_force_enumeration_of_the_walks(seed):
    u, v, h, s, rc, n, seeds = _graph(seed)
    rows = np.arange(n)
    got, facts = L4.sketch_block(u, v, h, s, rc, n, seeds, rows)
    want = _brute(u, v, h, s, rc, n, seeds)
    assert np.array_equal(got[:, COUNTS], want[:, COUNTS])
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)
    assert facts["seeds"] == 2 and facts["seed_duplicates"] == 1 and facts["edges"] == u.size
    ends = np.bincount([int(v[w[-1]]) for w in _walks(u, v, {0, 3}, 3)], minlength=n)
    assert facts["max_seed_walks"] == ends.max() and facts["unreached"] == int(sum(
        not any(int(v[w[-1]]) == r for L in L4.WALK_L for w in _walks(u, v, {0, 3}, L)) for r in rows))
    sub = np.array([6, 1, 4])
    assert np.array_equal(L4.sketch_block(u, v, h, s, rc, n, seeds, sub)[0], got[sub])


def test_the_sketches_are_invariant_to_the_order_of_the_edges_and_a_token_to_the_order_of_its_slots():
    u, v, h, s, rc, n, seeds = _graph(4, m=30)
    perm = np.random.default_rng(1).permutation(u.size)
    a, _ = L4.sketch_block(u, v, h, s, rc, n, seeds, np.arange(n))
    b, _ = L4.sketch_block(u[perm], v[perm], h[:, perm], s[:, perm], rc[perm], n, seeds, np.arange(n))
    maxima = L4.PSL["path_query_match"].start + len(L4.WALK_L)
    assert np.array_equal(a[:, COUNTS], b[:, COUNTS]) and np.array_equal(a[:, maxima:], b[:, maxima:])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
    slots = np.array([[3, 1, -1, 2], [2, -1, 3, 1], [-1, 1, 2, 3]])
    tok = L4.edge_tokens([0, 0, 0], [1, 1, 1], [0, 0, 0], slots)
    assert tok[0] == tok[1] == tok[2]
    others = L4.edge_tokens([1, 0, 0, 0], [1, 0, 1, 1], [0, 0, 1, 0], [[3, 1, -1, 2], [3, 1, -1, 2], [3, 1, -1, 2], [3, 1, -1, 1]])
    assert len({int(t) for t in others} | {int(tok[0])}) == 5   # family, either direction flag or a slot changes the token
    assert np.array_equal(L4.edge_tokens([2], [0.9], [0.1], [[-1] * 4]), L4.edge_tokens([2], [0.6], [0.4], [[-1] * 4]))


def test_reversing_a_two_edge_typed_path_changes_its_bucket_or_sign():
    tok = L4.edge_tokens([0, 0], [1, 1], [0, 0], [[1, -1, -1, -1], [2, -1, -1, -1]])
    h, s = L4.position_hashes(tok)
    forward = ((h[0, 0] + h[1, 1]) % L4.WALK_B, s[0, 0] * s[1, 1])
    reverse = ((h[0, 1] + h[1, 0]) % L4.WALK_B, s[0, 1] * s[1, 0])
    assert forward != reverse
    # through the recursion: 0 -> 1 -> 2 typed (r1, r2) against (r2, r1), seed 0
    u, v = np.array([0, 1]), np.array([1, 2])
    a, _ = L4.sketch_block(u, v, h, s, np.zeros(2), 3, [0], [2])
    b, _ = L4.sketch_block(u, v, h[:, ::-1], s[:, ::-1], np.zeros(2), 3, [0], [2])
    two = slice(L4.WALK_B, 2 * L4.WALK_B)
    assert not np.array_equal(a[:, two], b[:, two]) and a[0, L4.PSL["walk_totals"]][1] == b[0, L4.PSL["walk_totals"]][1] == math.log1p(1)
    # over many random typed pairs a reversal keeps its bucket and sign about 1 time in 128
    g = np.random.default_rng(0)
    t = L4.edge_tokens(g.integers(0, 3, 4000), g.random(4000), g.random(4000), g.integers(-1, 50, (4000, L4.K_REL)))
    h, s = L4.position_hashes(t)
    i, j = np.arange(0, 4000, 2), np.arange(1, 4000, 2)
    same = (((h[0, i] + h[1, j]) % 64 == (h[0, j] + h[1, i]) % 64) & (s[0, i] * s[1, j] == s[0, j] * s[1, i])) & (t[i] != t[j])
    assert same.mean() < 0.03
    assert (h[0] != h[1]).mean() > 0.9   # the same token at two positions lands in different buckets


def test_the_block_has_267_named_columns_and_a_row_no_walk_reaches_is_zero():
    assert L4.N_PATHS == 267 == len(set(L4.PATH_NAMES)) and all("|" in c for c in L4.PATH_NAMES)
    widths = {b: len(c) for b, c in L4.PATH_BLOCKS}
    assert widths == {"seed_walk_sketch": 192, "subtree_sketch": 64, "walk_totals": 5, "path_query_match": 6}
    declared = L4.load_declaration()["compile"]["block"]
    assert set(declared) == set(widths) | {"arithmetic"}
    assert "(258 + 267)" in L4.load_declaration()["probe"]["L4_mlp"] and "(258 + 267 + 64)" in L4.load_declaration()["probe"]["L4_att"]
    # node 4 has no in-edge; node 3 is reached only from a node that no seed walk reaches
    u, v = np.array([0, 1, 2, 5]), np.array([1, 2, 0, 3])
    h, s = L4.position_hashes(L4.edge_tokens([0, 1, 2, 0], [1, 0, 1, 0], [0, 1, 0, 1], np.full((4, L4.K_REL), -1)))
    got, facts = L4.sketch_block(u, v, h, s, np.array([0.5, 0.0, 0.0, 0.25]), 6, [0], np.arange(6))
    assert not got[4].any()
    seedside = np.r_[L4.PSL["seed_walk_sketch"], L4.PSL["path_query_match"], np.arange(L4.PSL["walk_totals"].start, L4.PSL["walk_totals"].start + 3)]
    assert not got[3, seedside].any() and got[3, L4.PSL["subtree_sketch"]].any()
    assert facts["unreached"] == 3   # nodes 3, 4 and 5; the seed 0 is reached by its length-3 cycle


def test_a_walk_count_at_the_exactness_limit_stops(monkeypatch):
    """The limit lowered to 64 so that a small complete graph reaches it: 4 nodes, every ordered pair joined twice, all
    four seeds -- 8, 64 and 512 walks of length 1, 2 and 3 into each node, and 8, 64 from anywhere."""
    n = 4
    u = np.repeat(np.repeat(np.arange(n), n), 2)
    v = np.repeat(np.tile(np.arange(n), n), 2)
    h, s = np.zeros((3, u.size), dtype=np.int64), np.ones((3, u.size))
    monkeypatch.setattr(L4, "EXACT", 512.0)
    with pytest.raises(ValueError, match="length 3"):
        L4.sketch_block(u, v, h, s, np.zeros(u.size), n, [0, 1, 2, 3], [0])
    monkeypatch.setattr(L4, "EXACT", 513.0)
    got, facts = L4.sketch_block(u, v, h, s, np.zeros(u.size), n, [0, 1, 2, 3], [0])
    assert facts["max_seed_walks"] == 512 and facts["max_any_walks"] == 64
    assert got[0, L4.PSL["seed_walk_sketch"].start + 2 * L4.WALK_B] == math.log1p(512)   # every hash 0, every sign +1: bucket 0
    monkeypatch.setattr(L4, "EXACT", 16.0)   # one seed: 2, 16 and 128 walks into each node
    with pytest.raises(ValueError, match="seed-walk count of length 2"):
        L4.sketch_block(u, v, h, s, np.zeros(u.size), n, [0], [0])
    out0 = u != 0   # the seed without out-edges: no seed walk; 6 and 36 walks from anywhere into each node
    monkeypatch.setattr(L4, "EXACT", 36.0)
    with pytest.raises(ValueError, match="length 2 from anywhere"):
        L4.sketch_block(u[out0], v[out0], h[:, out0], s[:, out0], np.zeros(int(out0.sum())), n, [0], [0])


# ── L4-att ───────────────────────────────────────────────────────────────────


def _toy(seed: int = 0, n_r: int = 7, n_h: int = 12, m: int = 30):
    g = np.random.default_rng(seed)
    fam = g.integers(0, len(L3.FAMILIES), m)
    eps = np.zeros((m, L3.EPS_WIDTH), dtype=np.float32)
    eps[np.arange(m), fam] = 1.0
    a0 = len(L3.EPS_FAMILIES)
    eps[:, a0:a0 + L3.N_ATTR] = g.random((m, L3.N_ATTR))
    eps[:, a0 + L3.N_ATTR:] = g.normal(size=(m, L3.JL_DIM)) * (fam == 0)[:, None]

    def t(a, dt=torch.float32):
        return torch.as_tensor(np.asarray(a), dtype=dt)
    return {"nh": t(g.normal(size=(n_h, L3.HALO_WIDTH))), "q_rows": t(np.repeat(g.normal(size=(1, L3.JL_DIM)), n_r, 0)),
            "ent_u": t(g.integers(0, n_h, m), torch.long), "ent_v": t(g.integers(0, n_r, m), torch.long), "ent_eps": t(eps),
            "row_h": t(g.permutation(n_h)[:n_r], torch.long)}, t(g.normal(size=(n_r, L3.B0_WIDTH))), t(g.normal(size=(n_r, L4.N_PATHS)))


def test_l4_att_initialises_as_l3_att_and_equals_it_with_the_extra_readout_columns_zeroed():
    torch.manual_seed(7)
    l3 = L3.L3Probe(uniform=False)
    torch.manual_seed(7)
    l4 = L4.L4Att()
    s3, s4 = l3.state_dict(), l4.state_dict()
    assert set(s3) == set(s4)
    for key in s3:
        if not key.startswith("readout."):
            assert torch.equal(s3[key], s4[key]), key
    assert s4["readout.0.weight"].shape == (L0.MLP_HIDDEN, L3.B0_WIDTH + L4.N_PATHS + L3.ATT_WIDTH)
    with torch.no_grad():
        W = torch.zeros_like(l4.readout[0].weight)
        W[:, :L3.B0_WIDTH] = l3.readout[0].weight[:, :L3.B0_WIDTH]
        W[:, L3.B0_WIDTH + L4.N_PATHS:] = l3.readout[0].weight[:, L3.B0_WIDTH:]
        l4.readout[0].weight.copy_(W)
        l4.readout[0].bias.copy_(l3.readout[0].bias)
        for i in (2, 4):
            l4.readout[i].weight.copy_(l3.readout[i].weight)
            l4.readout[i].bias.copy_(l3.readout[i].bias)
    inp, bv, pv = _toy()
    with torch.no_grad():
        want, alpha3 = l3(bv, **inp, return_alpha=True)
        for p in (torch.zeros_like(pv), pv):
            got, alpha4 = l4(bv, p, **inp, return_alpha=True)
            assert torch.equal(alpha3, alpha4)
            torch.testing.assert_close(got, want, rtol=1e-6, atol=1e-6)


# ── the graph integrity against level 3's stored entries ────────────────────


def _captured_l3(monkeypatch, d3: Path, sc, n_rel: int = 5) -> list:
    """Level 3's synthetic compile, with every query's packed batch (edge_index, edge_attr, U_q rows, offset, pool size)
    caught at level 3's check_attributes, in query order."""
    batches, real = [], L3.check_attributes

    def spy(ent, ei, ea, loc, offset, pool_size):
        batches.append((np.array(ei), np.array(ea), np.array(loc, dtype=np.int64), int(offset), int(pool_size)))
        return real(ent, ei, ea, loc, offset, pool_size)
    monkeypatch.setattr(L3, "check_attributes", spy)
    _synthetic_l3(d3, sc, n_rel=n_rel)
    monkeypatch.setattr(L3, "check_attributes", real)
    assert len(batches) == sc.n_q
    return batches


def _seeds_of(j: int, n_pool: int) -> np.ndarray:
    g = np.random.default_rng(100 + j)
    return np.sort(g.choice(n_pool, size=min(n_pool, 1 + int(g.integers(0, 3))), replace=False))


def test_the_batch_gives_level_3s_stored_entries_and_a_changed_or_dropped_edge_stops(tmp_path, monkeypatch, stops):
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=50, sizes=(5, 12))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    batches = _captured_l3(monkeypatch, d3, sc)
    l3e = L4.Level3Entries(sc, d3)
    for j, (ei, ea, loc, offset, _n) in enumerate(batches):
        assert L4.graph_mismatch(L3.query_entries(ei, ea, loc, offset), l3e.query(j)) is None, j
    j = next(j for j, b in enumerate(batches) if L3.query_entries(*b[:4])["keep"].sum() >= 2)
    ei, ea, loc, offset, _n = batches[j]
    kept = np.flatnonzero(L3.query_entries(ei, ea, loc, offset)["keep"])
    ea2 = ea.copy()
    ea2[kept[0], L4.RC] += 0.5
    assert L4.graph_mismatch(L3.query_entries(ei, ea2, loc, offset), l3e.query(j)) == "ent_attr"
    dropped = L3.query_entries(np.delete(ei, kept[0], axis=1), np.delete(ea, kept[0], axis=0), loc, offset)
    assert L4.graph_mismatch(dropped, l3e.query(j)) is not None
    assert L4.graph_mismatch(L3.query_entries(ei, ea, loc, offset), l3e.query(j + 1 if j + 1 < sc.n_q else j - 1)) is not None


def test_query_paths_on_level_3s_packed_batches_equal_the_brute_force_block(tmp_path, monkeypatch, stops):
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=12, sizes=(4, 8))
    sc = L0.Sidecar(d0)
    batches = _captured_l3(monkeypatch, tmp_path / "l3" / "metaqa", sc)
    for j in (0, 5, sc.n_q - 1):
        ei, ea, loc, offset, n_pool = batches[j]
        seeds = _seeds_of(j, n_pool)
        fam = L3.family_of(ea[:, :L4.A0])
        slots = L3.slots_local(ea[:, L4.SLOT0:L4.SLOT0 + L4.K_REL], offset)
        h, s = L4.position_hashes(L4.edge_tokens(fam, ea[:, L4.FWD], ea[:, L4.BWD], slots))
        want = _brute(ei[0], ei[1], h, s, ea[:, L4.RC].astype(np.float64), n_pool, seeds)[loc]
        f64, _ = L4.sketch_block(ei[0], ei[1], h, s, ea[:, L4.RC].astype(np.float64), n_pool, seeds, loc)
        np.testing.assert_allclose(f64, want, rtol=1e-12, atol=1e-12)
        block, facts = L4.query_paths(ei, ea, n_pool, seeds, loc, offset)
        assert block.dtype == np.float16 and block.shape == (loc.size, 267) and np.array_equal(block, L0.to_f16(f64, "paths"))


# ── the bands, contrasts, readings and the refit comparison ─────────────────


def test_the_bands_contrasts_grid_and_flags_are_the_declared_ones():
    assert L4.band_l4(0.80, [0.55, 0.95], True) == "L4_HIGH"
    assert L4.band_l4(0.80, [0.45, 0.95], True) == "L4_MID"
    assert L4.band_l4(0.20, [0.00, 0.45], True) == "L4_LOW"
    assert L4.band_l4(0.90, [0.80, 0.99], False) == "NOT_READ"
    decl = L4.load_declaration()
    contrasts = {}
    for c_name, text in decl["statistics"]["contrasts"].items():
        a, b, fam = re.match(r"rho_bar\((\S+)\) - rho_bar\((\S+)\) on (r|e)\b", text).groups()
        contrasts[c_name] = (fam, a, b)
    assert L4.CONTRASTS == contrasts
    grid = decl["probe"]["grid"]
    assert list(L4.GRID["r"]) == grid["r_k"] and list(L4.GRID["e"]) == grid["e_k"]
    assert all(a in L4.GRID[f] and b in L4.GRID[f] for f, a, b in contrasts.values())
    assert L4.UNIT_ORDER[0] == ("r", "L4-att") and len(L4.UNIT_ORDER) == len(grid["r_k"]) + len(grid["e_k"]) == 7
    assert L4.PRIMARY == "L4-att" and decl["readings"]["primary_probe"].startswith("L4-att on r_k")
    assert set(L4.REFIT) == {"B0-mlp", "L3-att"} and all(p in L3.PROBES for p in L4.REFIT)
    assert set(L4.NEW) | set(L4.REFIT) == set(grid["r_k"]) | set(grid["e_k"]) and L4.NEW["L4-list"][1] == "LIST"
    assert set(decl["readings"]["flags"]) == {"FIT_NOT_RANK", "SEED_BOUND", "REPEAT_DIFFERS", "L3_REFIT_DIFFERS"}
    assert L4.REPEAT == ("metaqa", "r", "L4-att")


def _readings_stub(points: dict, boots: dict, primary_band="L4_MID", e_band="L4_MID", r2=0.1, repro=0.8, repeat=True, refit=True) -> dict:
    probes = {p: {"rho_bar": {"point": points[p], "ci": L0.ci(boots[p])}, "_rho_bar_boot": boots[p], "band": "L4_MID",
                  "R2": {"point": r2}} for p in points}
    out = {"r": {"probes": {p: dict(v) for p, v in probes.items()}, "readable_metrics": ["hit@1"]},
           "e": {"probes": {p: dict(v) for p, v in probes.items()}, "readable_metrics": ["hit@1"]},
           "reproducibility": {"r": {"mean": {"point": repro}}}, "repeat": {"bit_identical": repeat, "max_abs_diff": 0.0 if repeat else 1e-7},
           "refit": {"r/B0-mlp": {"bit_identical": refit, "max_abs_diff": 0.0 if refit else 2e-7}}}
    out["r"]["probes"]["L4-att"]["band"] = primary_band
    out["e"]["probes"]["L4-att"]["band"] = e_band
    out["contrasts"] = L4.contrasts_of(out)
    return out


def test_the_readings_list_every_declared_entry_that_applies_and_the_contrasts_are_paired():
    g = np.random.default_rng(0)
    noise = g.normal(size=1000) * 0.01
    points = {"B0-mlp": 0.2, "L4-mlp": 0.5, "L3-att": 0.6, "L4-att": 0.8, "L4-list": 0.9}
    stub = _readings_stub(points, {p: v + noise for p, v in points.items()}, primary_band="L4_HIGH")
    c = stub["contrasts"]
    assert abs(c["paths_over_attention"]["point"] - 0.2) < 1e-12 and np.allclose(c["paths_over_attention"]["ci"], [0.2, 0.2])
    r = L4.readings(stub)
    assert r["reading"] == "L4_HIGH" and r["flags"] == []
    assert set(r["interpretation"]) == {"l4_high", "l4_paths_add", "paths_node_local_adds", "objective_adds", "edge_effect_paths"}
    flat = {"B0-mlp": 0.2, "L4-mlp": 0.2, "L3-att": 0.1, "L4-att": 0.1, "L4-list": 0.1}
    stub = _readings_stub(flat, {p: v + noise for p, v in flat.items()}, primary_band="L4_LOW", r2=0.6, repro=0.4, repeat=False, refit=False)
    r = L4.readings(stub)
    assert set(r["interpretation"]) == {"l4_paths_flat", "paths_not_below_attention"}
    assert r["flags"][-3:] == ["SEED_BOUND", "REPEAT_DIFFERS (max |diff| 1.000e-07)", "L3_REFIT_DIFFERS (r/B0-mlp, max |diff| 2.000e-07)"]
    assert "FIT_NOT_RANK (r, L4-att)" in r["flags"] and "FIT_NOT_RANK (r, B0-mlp)" not in r["flags"]
    assert set(L4.readings(stub)["interpretation"]) <= set(L4.load_declaration()["readings"]["interpretation_map"])
    unread = _readings_stub(points, {p: v + noise for p, v in points.items()})
    unread["r"]["readable_metrics"] = []
    assert L4.contrasts_of(unread)["paths_over_attention"]["ci"] is None


def test_the_refit_comparison_is_bitwise(tmp_path):
    a = np.random.default_rng(0).normal(size=50)
    np.savez(tmp_path / "a.npz", **{"r|B0-mlp|0": a})
    np.savez(tmp_path / "b.npz", **{"r|B0-mlp|0": a.copy()})
    b = a.copy()
    b[7] = np.nextafter(b[7], np.inf)
    np.savez(tmp_path / "c.npz", **{"r|B0-mlp|0": b})
    assert L4.units_compare(tmp_path / "a.npz", tmp_path / "b.npz") == (True, 0.0)
    same, diff = L4.units_compare(tmp_path / "a.npz", tmp_path / "c.npz")
    assert same is False and 0 < diff < 1e-15


# ── a synthetic sketch sidecar, written by the compile's own helpers ────────


def _synthetic_l4(d4: Path, sc, batches: list, d3: Path, d0: Path, chunk: int = 17) -> dict:
    """paths.npy over level 3's synthetic packed batches, each query's seeds drawn from its pool, written in chunks and
    assembled as the compile assembles them."""
    chunks = d4 / "chunks"
    chunks.mkdir(parents=True)
    n_chunks = math.ceil(sc.n_q / chunk)
    for ci in range(n_chunks):
        idx = np.arange(ci * chunk, min((ci + 1) * chunk, sc.n_q))
        blocks, facts, q_entries = [], [], []
        for j in idx:
            ei, ea, loc, offset, n_pool = batches[j]
            block, f = L4.query_paths(ei, ea, n_pool, _seeds_of(int(j), n_pool), loc, offset)
            blocks.append(block)
            facts.append(f)
            q_entries.append(int(L3.query_entries(ei, ea, loc, offset)["ent_row"].size))
        L4.save_chunk(chunks / f"c{ci:05d}.npz", blocks, facts, idx, q_entries, idx)
    meta = L4.assemble_paths(chunks, d4, n_chunks, sc, sc.n_q)
    meta.update({"dataset": d4.name, "queries": sc.n_q, "seconds_this_process": 0.0, "threads": 1, "mismatches": 0,
                 "script_lf_sha256": L0.lf_sha256(ROOT / "scripts" / "mp_approx_l4.py"),
                 "level0_sidecar_meta_sha256": L0.sha256_file(d0 / "meta.json"), "level3_meta_sha256": L0.sha256_file(d3 / "meta.json")})
    meta = L4.finish_compile(d4, sc.qids, meta)
    shutil.rmtree(chunks)
    return meta


def test_the_mirror_is_written_and_verified_and_a_changed_byte_stops(tmp_path, monkeypatch, stops):
    l0 = tmp_path / "outputs" / "mp_approx_l0" / "squad"
    d3 = tmp_path / "outputs" / "mp_approx_l3" / "squad"
    d4 = tmp_path / "outputs" / "mp_approx_l4" / "squad"
    _synthetic_sidecar(l0, n_q=30, sizes=(4, 8))
    sc = L0.Sidecar(l0)
    meta = _synthetic_l4(d4, sc, _captured_l3(monkeypatch, d3, sc, n_rel=0), d3, l0)
    assert meta["arrays_shape"]["paths"] == [sc.n_rows, 267, "float16"] and meta["entries_compared"] > 0
    path = L4.write_mirror("squad", tmp_path, d4)
    files = json.loads(path.read_text(encoding="utf-8"))["files"]
    assert set(files) == {f"outputs/mp_approx_l4/squad/{f}" for f in L4.TRANSFER}
    assert L4.verify_mirror("squad", tmp_path, d4, d3, l0) == L0.sha256_file(path)
    raw = bytearray((d4 / "paths.npy").read_bytes())
    raw[-1] ^= 1
    (d4 / "paths.npy").write_bytes(bytes(raw))
    with pytest.raises(SystemExit, match="HARD STOP"):
        L4.verify_mirror("squad", tmp_path, d4, d3, l0)
    last = json.loads((stops / "hard_stops.json").read_text(encoding="utf-8"))[-1]
    assert "differ from mirror.json" in last["message"] and "outputs/mp_approx_l4/squad/paths.npy" in last["files"]
    assert not (L4.OUT / "hard_stops.json").exists()


# ── probe, repeat, read and doc, end to end ─────────────────────────────────


def test_probe_repeat_read_and_doc_run_end_to_end_on_the_cpu(tmp_path, monkeypatch, stops, det_request):
    monkeypatch.setattr(L0, "MLP_EPOCHS", 3)
    d0 = _synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=120, sizes=(6, 14))
    sc = L0.Sidecar(d0)
    d3 = tmp_path / "l3" / "metaqa"
    batches = _captured_l3(monkeypatch, d3, sc)
    L3.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d3, l0_dir=d0)   # level 3's filed units, for the refit comparison
    d4 = tmp_path / "l4" / "metaqa"
    cmeta = _synthetic_l4(d4, sc, batches, d3, d0)
    P = np.load(d4 / "paths.npy")
    assert P.shape == (sc.n_rows, 267) and P.dtype == np.float16 and np.isfinite(P).all() and P.any()
    assert cmeta["qids_sha256"] == L0.sha256_file(d4 / "qids.json") and json.loads((d4 / "qids.json").read_text(encoding="utf-8")) == sc.qids
    L4.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d4, d3=d3, l0_dir=d0)
    meta = json.loads((d4 / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    want = sorted([f"unit_{t}_{p}.npz" for t, p in L4.UNIT_ORDER] + [f"fitlog_{t}_{p}.json" for t, p in L4.UNIT_ORDER])
    assert sorted(meta["files_sha256"]) == want and meta["compile_meta_sha256"] == L0.sha256_file(d4 / "meta.json")
    assert {"scripts/mp_approx_l4.py", "scripts/mp_approx_l3.py", "scripts/mp_approx_l0.py"} <= set(meta["module_sha256"])
    probes = L0.load_probes(d4)
    assert set(probes) == {f"{t}/{p}/{k}" for t, p in L4.UNIT_ORDER for k in L0.SEEDS}
    for v in probes.values():
        assert v.shape == (sc.n_rows,) and np.isfinite(v).all()
        assert np.abs(L0.centre_rows(v, sc.ptr) - v).max() < 1e-5
    flog = json.loads((d4 / "probes" / "fitlog_r_L4-att.json").read_text(encoding="utf-8"))
    assert flog["folds"][0]["path_standardisation"]["columns"] == 267
    L4.stage_probe("metaqa", log=_quiet, host=False, device="cpu", d=d4, d3=d3, l0_dir=d0, repeat=True)
    rep = json.loads((d4 / "repeat" / "repeat.json").read_text(encoding="utf-8"))
    assert rep["unit"] == "r_L4-att" and rep["bit_identical"] is True
    out = L4.stage_read("metaqa", log=_quiet, host=False, d=d4, d3=d3, l0_dir=d0)
    assert set(out["r"]["probes"]) == set(L4.GRID["r"]) | {"ref:twin", "ref:other_seed"}
    assert set(out["e"]["probes"]) == set(L4.GRID["e"]) | {"ref:no_edge", "ref:other_seed"}
    assert set(out["contrasts"]) == set(L4.CONTRASTS)
    assert set(out["refit"]) == {"r/B0-mlp", "r/L3-att", "e/L3-att"}
    assert all(v["bit_identical"] for v in out["refit"].values())   # level 3's code on the same inputs: its units exactly
    assert not any(f.startswith("L3_REFIT_DIFFERS") or f.startswith("REPEAT_DIFFERS") for f in out["flags"])
    bands = [v["band"] for fam_ in ("r", "e") for v in out[fam_]["probes"].values()]
    assert all(b in ("L4_HIGH", "L4_MID", "L4_LOW", "NOT_READ") for b in bands)
    assert out["reading"] == out["r"]["probes"]["L4-att"]["band"]
    assert set(out["interpretation"]) <= set(L4.load_declaration()["readings"]["interpretation_map"])
    assert out["paths"]["paths_sha256"] == cmeta["arrays_sha256"]["paths.npy"] and out["paths"]["uq_rows"] == sc.n_rows
    L4.stage_doc(log=_quiet, out_root=tmp_path / "l4", doc=tmp_path / "doc.md", datasets=("metaqa",), extra_deviations=["--gpus 0.16"])
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "## Readings" in text and "L4-list" in text and "host_gpu_det" in text and "--gpus 0.16" in text and "Level 3, as filed" in text
    body = text.split("## What this does not say")[0].lower()   # the forbidden framings appear only where the document rules them out
    assert not any(p in body for p in ("message passing is unnecessary", "we do not need message passing", "the mlp wins"))
    rec = json.loads((tmp_path / "l4" / "record.json").read_text(encoding="utf-8"))
    assert rec["datasets"]["metaqa"]["read_sha256"] == L0.sha256_file(d4 / "read.json")
    assert not (L4.OUT / "hard_stops.json").exists() and not (L3.OUT / "hard_stops.json").exists()


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L4.DATASETS)
def test_no_held_query_id_appears_in_any_sidecar(name):
    side = L4.OUT / name / "qids.json"
    if not side.exists():
        pytest.skip(f"{name}: this file's sketch is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"][name]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    compiled = json.loads(side.read_text(encoding="utf-8"))
    assert compiled and set(compiled) <= gate
    assert compiled == json.loads((L0.OUT / name / "qids.json").read_text(encoding="utf-8"))
    meta = L4.OUT / name / "meta.json"
    if meta.exists():
        assert json.loads(meta.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)
