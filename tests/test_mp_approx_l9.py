"""MP-Approx level 9 (configs/mp_approx_l9.yaml#tests): the declaration, its pins and its constants; the population rule on
toy ids; the non-backtracking walk counts against a brute-force enumeration, their edge-order and block invariance and
their place inside the standard counts; the combined programme and its stops; the two views of a combined sidecar
against sidecars of one family; the residual models' parameters, inputs and start at level 8's TP; the nested level-8
protocol against level 8's fit_unit bit for bit, and EM on a planted chain; the bands, grid edges and contrasts; the
check, fit, repeat, read, doc and file stages end to end on a synthetic combined sidecar; and no held, level 0 or level 8
query in any sidecar."""

from __future__ import annotations

import hashlib
import inspect
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
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's random multigraphs, toy sidecar and relation text)

_quiet = T8._quiet


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9):
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
    """The fit stages' deterministic algorithms, at one thread, so the tests stay light on the laptop."""
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def _sidecar(d: Path, monkeypatch, programme, **kwargs) -> list[str]:
    """Level 8's toy sidecar, compiled by the given walk programme."""
    monkeypatch.setattr(L8, "walk_entries", programme)
    try:
        return T8._toy_sidecar(d, **kwargs)
    finally:
        monkeypatch.setattr(L8, "walk_entries", L9._L8_WALKS)


# ── the declaration, its pins and its constants ──────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256(stops):
    decl = L9.load_declaration()
    assert decl["phase"] == "MP_APPROX_L9" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L8.load_declaration()["registered_question"]
    assert list(decl["inputs"]["relations"]["order"]) == list(L8.REL_ORDER)
    L9.verify_inputs(decl)   # level 8's pins in this file's copy, level 0's verify_pins, and level 8's own files and rows
    assert not (stops / "hard_stops.json").exists()


@pytest.mark.parametrize("key", ["script_lf", "tests_lf", "declaration_lf"])
def test_a_changed_level8_pin_stops(stops, monkeypatch, key):
    monkeypatch.setattr(L9, "_L8_VERIFY", lambda decl: None)   # level 8's own pins are the test above's
    decl = L9.load_declaration()
    decl["inputs"]["level8"][key]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L9.verify_inputs(decl)
    assert decl["inputs"]["level8"][key]["path"] in (stops / "hard_stops.json").read_text(encoding="utf-8")


def test_a_changed_copied_pin_stops(stops):
    decl = L9.load_declaration()
    decl["inputs"]["frozen_code_lf"]["scripts/m3a_headroom.py"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L9.verify_inputs(decl)
    assert "m3a_headroom.py" in (stops / "hard_stops.json").read_text(encoding="utf-8")


def test_the_declared_constants_are_the_files():
    decl = L9.load_declaration()
    assert {a: tuple(v) for a, v in decl["arms"]["fitted"].items()} == L9.ARM_SPEC
    assert L9.ARMS == tuple(decl["arms"]["fitted"]) and L9.PRIMARY == decl["readings"]["primary"] == "NB-hyb"
    assert set(L9.REFERENCES) == set(decl["arms"]["references"]) and L9.REFERENCES == {"NB-oracle": "nb", "STD-oracle": "std"}
    assert {c: f"{a} - {b}" for c, (a, b) in L9.CONTRASTS.items()} == decl["statistics"]["contrasts"]
    assert L9.KAPPAS == (1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, 2.0, 4.0, 8.0, 16.0) and L9.ETAS == (0.01, 0.1, 1.0, 10.0, 100.0)
    assert (L9.EM_ROUNDS, L9.PER_HOP, L9.SALT, L9.HIDDEN, L9.NB_SHIFT) == (10, 1000, "mp_approx_l9|", 256, 43904)
    assert L9.NB_SHIFT == 2 * 28 ** 3 and L9.REPEAT_UNIT == ("NB-hyb", 0, 0) and L9.NESTED == "@L8"
    assert {int(h): v for h, v in decl["population"]["available_per_hop"].items()} == L9.AVAILABLE_PER_HOP
    assert all(L8.AVAILABLE_PER_HOP[h] - L8.PER_HOP == L9.AVAILABLE_PER_HOP[h] for h in (1, 2, 3))
    # the nested protocol reads level 8's own values
    assert (L8.EM_ROUNDS, L8.KAPPAS, L8.ETAS) == (5, (0.25, 0.5, 1.0, 2.0, 4.0, 8.0), (0.01, 0.1, 1.0))
    assert set(decl["readings"]["bands"]) == {"L9_ABOVE_GNN", "L9_HIGH", "L9_LOW", "L9_MID", "NOT_READ"}
    assert set(decl["readings"]["flags"]) == {"CEILING_LOW", "REPEAT_DIFFERS", "EM_NOT_MONOTONE", "GRID_EDGE"}
    named = {x for pair in L9.INTERPRET_CONTRAST.values() for x in pair if x}
    assert named | {"l9_above_gnn", "l9_high", "l9_mid", "l9_low", "chain_not_identified"} == set(decl["readings"]["interpretation_map"])
    assert set(L9.INTERPRET_CONTRAST) <= set(L9.CONTRASTS)
    read = set(L9.ARMS) | {a + L9.NESTED for a in L9.ARMS} | set(L9.REFERENCES)
    assert all(a in read and b in read for a, b in L9.CONTRASTS.values())


# ── the population rule ──────────────────────────────────────────────────────


def _toy_population(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(400)]
    gate = rng.random(len(ids)) < 0.7
    excluded = L0.metaqa_subsample(ids, gate, per_hop=40)
    level8, _ = L8.l8_rows(ids, gate, excluded, per_hop=50)
    return ids, gate, excluded, level8


def test_the_population_rule_leaves_out_level0s_and_level8s_rows_and_sorts_by_its_own_salt():
    ids, gate, excluded, level8 = _toy_population()
    rows, available = L9.l9_rows(ids, gate, excluded, per_hop=60, level8_ids=[ids[i] for i in level8], level8_per_hop=50)
    assert rows.size == 180 and np.all(np.diff(rows) > 0) and gate[rows].all()
    assert not np.isin(rows, excluded).any() and not np.isin(rows, level8).any()
    out = set(excluded.tolist()) | set(level8.tolist())
    for h in (1, 2, 3):
        pool = [int(i) for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == h and int(i) not in out]
        assert available[h] == len(pool)
        want = sorted(pool, key=lambda i: hashlib.sha256(("mp_approx_l9|" + ids[i]).encode("utf-8")).hexdigest())[:60]
        assert sorted(want) == [int(r) for r in rows if L0.hop_from_id(ids[r]) == h]
    again, _ = L9.l9_rows(list(ids), gate.copy(), excluded.copy(), per_hop=60, level8_per_hop=50)
    assert np.array_equal(rows, again)
    # rows_by_salt is level 8's rule with the salt as an argument
    a, av = L9.rows_by_salt(ids, gate, excluded, "mp_approx_l8|", 50)
    b, bv = L8.l8_rows(ids, gate, excluded, per_hop=50)
    assert np.array_equal(a, b) and av == bv
    with pytest.raises(SystemExit):
        L9.l9_rows(ids, gate, excluded, per_hop=10_000, level8_per_hop=50)


def test_level8s_recomputed_rows_must_be_its_pinned_ids(stops):
    ids, gate, excluded, level8 = _toy_population(1)
    with pytest.raises(SystemExit, match="HARD STOP"):
        L9.l9_rows(ids, gate, excluded, per_hop=60, level8_ids=[ids[i] for i in level8[1:]], level8_per_hop=50)
    assert "level 8's recomputed rows" in (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the non-backtracking walks ───────────────────────────────────────────────


def _multigraph(rng, n: int, m: int):
    """Level 8's random typed multigraph (parallel edges with the same or another token), with explicit self-loops (one
    of them doubled) and a two-cycle whose return edge is doubled."""
    src, dst, tok = T8._random_multigraph(rng, n, m)
    loops = rng.choice(n, size=min(3, n), replace=False)
    a, b = rng.choice(n, size=2, replace=False)
    src = np.r_[src, loops, loops[:1], a, b, b]
    dst = np.r_[dst, loops, loops[:1], b, a, a]
    tok = np.r_[tok, rng.integers(0, L8.N_TOK, loops.size + 1), rng.integers(0, L8.N_TOK, 3)]
    return src, dst, tok


def _brute_force_nb(src, dst, tok, n, buckets) -> dict:
    """Every walk v_0 .. v_L of length 1 to 3 from each bucket's seeds with v_{i+1} != v_{i-1}, enumerated edge by edge."""
    out = {}
    out_edges = [[] for _ in range(n)]
    for e in range(src.size):
        out_edges[int(src[e])].append(e)
    for b, seeds in enumerate(buckets):
        frontier = [(int(s), None, ()) for s in np.unique(seeds)]
        for _length in range(1, 4):
            nxt = []
            for node, prev, toks in frontier:
                for e in out_edges[node]:
                    if prev is not None and int(dst[e]) == prev:
                        continue
                    t = toks + (int(tok[e]),)
                    nxt.append((int(dst[e]), node, t))
                    key = (L8.type_code(b, t), int(dst[e]))
                    out[key] = out.get(key, 0) + 1
            frontier = nxt
    return out


def _entries(code, node, count) -> dict:
    b, toks, lengths = L8.decode_types(code)
    return {(int(b[i]), tuple(int(t) for t in toks[i, :int(lengths[i])]), int(node[i])): int(count[i]) for i in range(len(code))}


@pytest.mark.parametrize("seed", range(6))
def test_the_nb_counts_of_both_buckets_equal_a_brute_force_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 13))
    src, dst, tok = _multigraph(rng, n, int(rng.integers(10, 35)))
    seeds = rng.choice(n, size=4, replace=False)
    buckets = [seeds[:2], seeds[2:]]
    code, node, count = L9.walk_entries_nb(src, dst, tok, n, buckets, block=7)
    got = {(int(c), int(v)): int(k) for c, v, k in zip(code, node, count)}
    assert got == _brute_force_nb(src, dst, tok, n, buckets)
    assert np.array_equal(np.lexsort((node, code)), np.arange(code.size)) and len(got) == code.size and (count > 0).all()
    b, _toks, lengths = L8.decode_types(code)
    assert set(np.unique(b)) <= {0, 1} and set(np.unique(lengths)) <= {1, 2, 3}


def test_a_step_back_is_excluded_whichever_relation_it_uses_and_a_self_loop_only_when_it_repeats():
    # 0 -> 1 (token 3); 1 -> 0 under tokens 4 and 5 (an edge with two relations); 1 -> 2 (token 6); 2 -> 2 (token 7)
    src, dst, tok = np.array([0, 1, 1, 1, 2]), np.array([1, 0, 0, 2, 2]), np.array([3, 4, 5, 6, 7])
    got = _entries(*L9.walk_entries_nb(src, dst, tok, 3, [np.array([0]), np.array([2])]))
    assert got == {(0, (3,), 1): 1, (0, (3, 6), 2): 1, (0, (3, 6, 7), 2): 1, (1, (7,), 2): 1}
    std = _entries(*L8.walk_entries(src, dst, tok, 3, [np.array([0]), np.array([2])]))
    assert std[(0, (3, 4), 0)] == std[(0, (3, 5), 0)] == std[(1, (7, 7), 2)] == std[(1, (7, 7, 7), 2)] == 1


def test_the_nb_counts_do_not_depend_on_the_order_of_the_edges_or_the_block():
    rng = np.random.default_rng(5)
    n = 12
    src, dst, tok = _multigraph(rng, n, 40)
    buckets = [np.array([0, 3]), np.array([7])]
    ref = L9.walk_entries_nb(src, dst, tok, n, buckets)
    for trial in range(3):
        perm = np.random.default_rng(trial).permutation(src.size)
        for block in (1, 5, 10**6):
            got = L9.walk_entries_nb(src[perm], dst[perm], tok[perm], n, buckets, block=block)
            assert all(np.array_equal(a, b) for a, b in zip(ref, got))
    dup = L9.walk_entries_nb(src, dst, tok, n, [np.array([0, 3, 3, 0]), np.array([7, 7])])   # repeated seeds count once
    assert all(np.array_equal(a, b) for a, b in zip(ref, dup))


@pytest.mark.parametrize("seed", range(4))
def test_the_nb_counts_lie_within_the_standard_counts_and_equal_them_at_length_1(seed):
    rng = np.random.default_rng(10 + seed)
    n = int(rng.integers(6, 15))
    src, dst, tok = _multigraph(rng, n, int(rng.integers(15, 40)))
    buckets = [rng.choice(n, 2, replace=False), rng.choice(n, 3, replace=False)]
    s, u = int(buckets[0][0]), int((buckets[0][0] + 1) % n)   # a step back from a seed, so the families differ
    src, dst, tok = np.r_[src, s, u], np.r_[dst, u, s], np.r_[tok, 2, 9]
    std_arrays, nb_arrays = L8.walk_entries(src, dst, tok, n, buckets), L9.walk_entries_nb(src, dst, tok, n, buckets)
    std, nb = _entries(*std_arrays), _entries(*nb_arrays)
    assert set(nb) <= set(std) and all(nb[key] <= std[key] for key in nb)
    assert {k: v for k, v in std.items() if len(k[1]) == 1} == {k: v for k, v in nb.items() if len(k[1]) == 1}
    assert sum(nb.values()) < sum(std.values())
    L9.nb_within_std(*std_arrays, *nb_arrays, n)   # the per-query check passes


def test_a_walk_ceiling_a_large_count_and_a_seed_outside_the_pool_stop():
    rng = np.random.default_rng(1)
    src, dst, tok = T8._random_multigraph(rng, 10, 60)
    with pytest.raises(L8.WalkCeiling):
        L9.walk_entries_nb(src, dst, tok, 10, [np.array([0, 1])], cap=5)
    with pytest.raises(ValueError, match="outside the pool"):
        L9.walk_entries_nb(src, dst, tok, 10, [np.array([10])])
    c1 = L8.walk_entries(src, dst, tok, 10, [np.array([0, 1])])[0].size
    c2 = L9.walk_entries_nb(src, dst, tok, 10, [np.array([0, 1])])[0].size
    with pytest.raises(L8.WalkCeiling):   # the ceiling counts both families
        L9.walk_entries_both(src, dst, tok, 10, [np.array([0, 1])], cap=c1 + c2 - 1)
    assert L9.walk_entries_both(src, dst, tok, 10, [np.array([0, 1])], cap=c1 + c2)[0].size == c1 + c2
    big = 2000   # 2000^3 non-backtracking walks 0 -> 1 -> 2 -> 0 of one type
    src, dst = np.repeat([0, 1, 2], big), np.repeat([1, 2, 0], big)
    with pytest.raises(ValueError, match=r"2\^32"):
        L9.walk_entries_nb(src, dst, np.zeros(3 * big, dtype=np.int64), 3, [np.array([0])])


@pytest.mark.parametrize("seed", range(3))
def test_the_combined_programme_is_level_8s_entries_then_the_nb_entries_under_buckets_2_and_3(seed):
    rng = np.random.default_rng(20 + seed)
    n = int(rng.integers(6, 14))
    src, dst, tok = _multigraph(rng, n, int(rng.integers(12, 30)))
    buckets = [rng.choice(n, 2, replace=False), rng.choice(n, 2, replace=False)]
    c, v, k = L9.walk_entries_both(src, dst, tok, n, buckets, block=5)
    c1, v1, k1 = L8.walk_entries(src, dst, tok, n, buckets)
    c2, v2, k2 = L9.walk_entries_nb(src, dst, tok, n, buckets)
    s = c1.size
    assert np.array_equal(c[:s], c1) and np.array_equal(v[:s], v1) and np.array_equal(k[:s], k1)
    assert np.array_equal(c[s:], c2 + L9.NB_SHIFT) and np.array_equal(v[s:], v2) and np.array_equal(k[s:], k2)
    b = c // L8.TB ** L8.MAX_L
    assert np.all(np.diff(c) >= 0) and b[:s].max() <= 1 and b[s:].min() >= 2 and int(c.max()) < 4 * L8.TB ** L8.MAX_L


def test_a_broken_nb_entry_stops(monkeypatch):
    rng = np.random.default_rng(3)
    n = 10
    src, dst, tok = T8._random_multigraph(rng, n, 30)
    buckets = [np.array([0, 1]), np.array([2])]
    real = L9.walk_entries_nb
    std_codes = set(L8.walk_entries(src, dst, tok, n, buckets)[0].tolist())

    def larger(*args, **kwargs):   # an NB count above the standard one
        c, v, k = real(*args, **kwargs)
        return c, v, np.r_[k[:-1], 2 ** 31]

    def foreign(*args, **kwargs):   # an NB entry that is no standard entry
        c, v, k = real(*args, **kwargs)
        absent = next(x for x in (L8.type_code(b, [t, t, t]) for b in (0, 1) for t in range(27)) if x not in std_codes)
        return np.r_[c, absent], np.r_[v, 0], np.r_[k, 1]

    def short(*args, **kwargs):   # a length-1 entry missing
        c, v, k = real(*args, **kwargs)
        i = int(np.flatnonzero(L8.decode_types(c)[2] == 1)[0])
        keep = np.arange(c.size) != i
        return c[keep], v[keep], k[keep]

    for broken, message in ((larger, "exceeds"), (foreign, "not standard entries"), (short, "length-1")):
        monkeypatch.setattr(L9, "walk_entries_nb", broken)
        with pytest.raises(ValueError, match=message):
            L9.walk_entries_both(src, dst, tok, n, buckets)


# ── the views of a combined sidecar ──────────────────────────────────────────


def test_the_combined_toy_sidecar_holds_both_families_one_block_per_query(tmp_path, monkeypatch):
    _sidecar(tmp_path / "s", monkeypatch, L9.walk_entries_both, n_q=10, seed=7)
    raw = L8.Data(tmp_path / "s")
    b, _toks, lengths = L8.decode_types(raw.t_code)
    assert set(b.tolist()) == {0, 1, 2, 3} and set(lengths.tolist()) == {1, 2, 3}
    for q in range(raw.n_q):
        codes = np.asarray(raw.t_code[raw.type_rows(q)], dtype=np.int64)
        assert np.all(np.diff(codes) > 0) and (codes // L8.TB ** L8.MAX_L >= 2).any()
    with pytest.raises(ValueError):
        L9.View(tmp_path / "s", "both")


def test_each_view_of_a_combined_sidecar_is_the_sidecar_of_its_family(tmp_path, monkeypatch):
    paths = {}
    for tag, programme in (("both", L9.walk_entries_both), ("std", L9._L8_WALKS), ("nb", L9.walk_entries_nb)):
        paths[tag] = tmp_path / tag
        _sidecar(paths[tag], monkeypatch, programme, n_q=16, seed=11, planted=False)   # the same queries and golds
    vs = L9.views(paths["both"])
    rel = T8._rel()
    for fam in L9.FAMILIES:
        v, ref = vs[fam], L8.Data(paths[fam])
        assert v.family == fam and v.n_q == ref.n_q == 16
        for key in ("t_code", "t_size", "t_gold", "q_types", "q_entries", "type_ptr"):
            assert np.array_equal(np.asarray(getattr(v, key), dtype=np.int64), np.asarray(getattr(ref, key), dtype=np.int64)), (fam, key)
        assert set(np.unique(L8.decode_types(v.t_code)[0]).tolist()) <= {0, 1}
        fx, fr = L8.Fitter(v, rel), L8.Fitter(ref, rel)
        assert np.array_equal(fx.table.codes, fr.table.codes) and np.array_equal(fx.gidx, fr.gidx)
        assert np.array_equal(fx.ll_rows, fr.ll_rows) and np.array_equal(fx.ll_null, fr.ll_null)
        rng = np.random.default_rng(0)
        logps = []
        for q in range(v.n_q):
            assert all(np.array_equal(a, b) for a, b in zip(v.entries(q), ref.entries(q)))
            for code in v.t_code[v.type_rows(q)]:
                assert np.array_equal(v.reach(q, int(code)), ref.reach(q, int(code)))
            a = rng.normal(size=int(v.q_types[q]) + 1)
            logps.append(a - np.log(np.exp(a).sum()))
            assert np.array_equal(fx.mixture(q, logps[-1]), fr.mixture(q, logps[-1]))
        g1, m1 = fx.e_step(logps, np.arange(v.n_q))
        g2, m2 = fr.e_step(logps, np.arange(v.n_q))
        assert np.array_equal(m1, m2) and all(np.array_equal(a, b) for a, b in zip(g1, g2))
    assert np.array_equal(vs["std"].q_emb, vs["nb"].q_emb) and np.array_equal(vs["std"].is_gold, vs["nb"].is_gold)
    assert int(vs["nb"].q_entries.sum()) < int(vs["std"].q_entries.sum())


# ── the models ───────────────────────────────────────────────────────────────


def _codes(seed=0, m=40):
    rng = np.random.default_rng(seed)
    return sorted({L8.type_code(int(b), list(rng.integers(0, 27, int(L)))) for b, L in zip(rng.integers(0, 2, m), rng.integers(1, 4, m))})


def test_hyb_starts_exactly_at_level_8s_tp_under_the_same_seed():
    codes = _codes()
    table, rel = T8._table(codes), T8._rel()
    torch.manual_seed(7)
    tp = L8.TypeModel("TP", rel, table)
    torch.manual_seed(7)
    hyb = L9.ResidualModel("hyb", "linear", rel, table)
    q = torch.randn(5, 1536)
    idx = torch.as_tensor(np.random.default_rng(1).integers(0, len(codes), (5, 9)))
    with torch.no_grad():
        w1, n1 = tp.weights(q, idx)
        w2, n2 = hyb.weights(q, idx)
    assert torch.equal(w1, w2) and torch.equal(n1, n2)
    assert torch.equal(tp.A.weight, hyb.A.weight) and torch.equal(tp.P.weight, hyb.P.weight) and torch.equal(tp.theta, hyb.theta)
    own = [k for k, _ in hyb.named_parameters(recurse=False)]   # level 8's TP order, E last
    assert own == [k for k, _ in tp.named_parameters(recurse=False)] + ["E"] == ["delta", "theta", "beta", "lam", "c", "nu", "c_null", "E"]


def test_the_residual_arms_read_only_the_query_the_type_table_and_for_hyb_the_relation_text():
    table = T8._table([L8.type_code(0, [1]), L8.type_code(1, [1]), L8.type_code(1, [2, 5]), L8.type_code(0, [26, 0, 7]),
                       L8.type_code(1, [26, 0, 7])])
    rel = T8._rel()
    idm = L9.ResidualModel("id", "linear", rel, table)
    assert {k for k, _ in idm.named_parameters()} == {"A.weight", "beta", "lam", "c", "nu", "c_null", "E"}
    assert {k for k, _ in idm.named_buffers()} == {"tok", "tb", "tl", "seq"}
    hyb = L9.ResidualModel("hyb", "linear", rel, table)
    assert {k for k, _ in hyb.named_parameters()} == {"A.weight", "P.weight", "delta", "theta", "beta", "lam", "c", "nu", "c_null", "E"}
    assert {k for k, _ in hyb.named_buffers()} == {"rel_emb", "tok", "tb", "tl", "seq"}
    # E holds one row per token sequence, at 0, shared by the two buckets of a sequence
    seqs = table.codes % L8.TB ** L8.MAX_L
    assert idm.E.shape == (np.unique(seqs).size, L8.D) == (3, L8.D) and float(idm.E.detach().abs().sum()) == 0.0
    assert all((int(idm.seq[i]) == int(idm.seq[j])) == (seqs[i] == seqs[j]) for i in range(5) for j in range(5))
    with torch.no_grad():
        idm.E[idm.seq[0]] = 1.0
        e = idm.compose()
    both = [int(i) for i in np.flatnonzero(seqs == seqs[0])]
    assert len(both) == 2 and torch.equal(e[both[0]], e[both[1]]) and float(e[both[0]].sum()) == L8.D
    assert float(e[torch.as_tensor(np.flatnonzero(seqs != seqs[0]))].abs().sum()) == 0.0
    mlp = L9.ResidualModel("hyb", "mlp", rel, table)
    assert [tuple(p.shape) for p in mlp.A.parameters()] == [(256, 1536), (256,), (64, 256), (64,)]
    assert isinstance(mlp.A[1], torch.nn.GELU)
    for arm in L9.ARMS:
        m = L9.make_model(arm, rel, table)
        assert {k for k, _ in m.named_buffers()} <= {"rel_emb", "tok", "tb", "tl", "thb", "seq"}
        assert list(inspect.signature(m.weights).parameters) == ["qemb", "idx"]
        comp = L9.ARM_SPEC[arm][1]
        assert isinstance(m, L8.TypeModel) == (comp in ("text", "hash"))
        if comp == "text":
            assert m.arm == "TP"
        if comp == "hash":
            assert m.arm == "TP-hash"
    with pytest.raises(ValueError):
        L9.ResidualModel("text", "linear", rel, table)


# ── the nested protocol and EM ───────────────────────────────────────────────


def test_the_nested_level8_protocol_of_std_text_is_level_8s_fit_unit_bit_for_bit(tmp_path, monkeypatch, one_thread):
    torch.use_deterministic_algorithms(True)
    _sidecar(tmp_path / "both", monkeypatch, L9.walk_entries_both, n_q=160, seed=3)
    _sidecar(tmp_path / "std", monkeypatch, L9._L8_WALKS, n_q=160, seed=3)
    monkeypatch.setattr(L8, "M_EPOCHS", 3)
    monkeypatch.setattr(L8, "EM_ROUNDS", 2)
    monkeypatch.setattr(L8, "LR", 3e-2)
    monkeypatch.setattr(L9, "EM_ROUNDS", 4)
    rel = T8._rel()
    a9, f9 = L9.fit_unit(L8.Fitter(L9.View(tmp_path / "both", "std"), rel), "STD-text", 0, 0, _quiet)
    a8, f8 = L8.fit_unit(L8.Fitter(L8.Data(tmp_path / "std"), rel), "TP", 0, 0, _quiet)
    assert f8["kept_round"] > 0   # a fitted round is kept, so the state the nested protocol reloads matters
    assert np.array_equal(a9["q"], a8["q"])
    assert np.array_equal(a9["metrics_l8"], a8["metrics"]) and np.array_equal(a9["argmax_l8"], a8["argmax"])
    assert (f9["kept_round_l8"], f9["kappa_l8"], f9["eta_l8"]) == (f8["kept_round"], f8["kappa"], f8["eta"])
    assert f9["grid_inner_mean3_l8"] == f8["grid_inner_mean3"]
    assert f9["rounds"][:3] == f8["rounds"] and len(f9["rounds"]) == 5
    assert len(f9["grid_inner_mean3"]) == 50 and len(f9["grid_inner_mean3_l8"]) == 18


def test_em_with_nb_hyb_recovers_a_planted_chain_from_the_query(tmp_path, monkeypatch, one_thread):
    _sidecar(tmp_path / "s", monkeypatch, L9.walk_entries_both, n_q=240, seed=3)
    view = L9.View(tmp_path / "s", "nb")
    monkeypatch.setattr(L9, "EM_ROUNDS", 5)
    arrays, flog = L9.fit_unit(L8.Fitter(view, T8._rel()), "NB-hyb", 0, 0, _quiet)
    assert flog["kept_round"] > 0 and flog["rounds"][flog["kept_round"]]["inner_mll"] > flog["rounds"][0]["inner_mll"]
    qt = [view.meta["qtypes"][i] for i in view.q_qtype[arrays["q"]]]
    agree = np.mean([L8.token_sequence(int(c)) == tuple(L8.chain_tokens(L8.true_chain(t))) for c, t in zip(arrays["argmax"], qt)])
    assert agree >= 0.8, agree


# ── the statistics and the readings ──────────────────────────────────────────


def test_the_bands_come_in_the_declared_order():
    assert L9.band(1.3, [1.05, 1.5], True) == "L9_ABOVE_GNN"   # before L9_HIGH
    assert L9.band(0.9, [0.6, 1.2], True) == "L9_HIGH"
    assert L9.band(0.8, [0.45, 1.0], True) == "L9_MID"
    assert L9.band(0.2, [0.0, 0.45], True) == "L9_LOW"
    assert L9.band(0.2, [0.0, 0.55], True) == "L9_MID"
    assert L9.band(0.9, [0.6, 1.2], False) == "NOT_READ"
    assert L9.band(float("nan"), [0, 0], True) == "NOT_READ"


def test_read_arm_carries_this_files_band():
    rng = np.random.default_rng(0)
    n = 300
    T = rng.random((n, 3, 3)) * 0.5
    G = T + 0.2 + 0.01 * rng.random((n, 3, 3))
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    assert L9.read_arm(T + 0.5 * (G - T), T, G, dens, readable, W)["band"] == "L9_MID"
    assert L9.read_arm(T, T, G, dens, readable, W)["band"] == "L9_LOW"
    assert L9.read_arm(G + 0.3, T, G, dens, readable, W)["band"] == "L9_ABOVE_GNN"
    assert L9.read_arm(T, T, T, *L8.denominators(T, T, W)[1:], W)["band"] == "NOT_READ"


def test_grid_edges_count_the_ends_of_both_protocols_grids():
    units = {"a": {"kappa": 1 / 32, "eta": 1.0, "kept_round": 10, "kappa_l8": 0.25, "eta_l8": 1.0, "kept_round_l8": 5},
             "b": {"kappa": 1.0, "eta": 100.0, "kept_round": 3, "kappa_l8": 1.0, "eta_l8": 0.1, "kept_round_l8": 3},
             "c": {"kappa": 2.0, "eta": 0.1, "kept_round": 0, "kappa_l8": 8.0, "eta_l8": 0.01, "kept_round_l8": 0}}
    g = L9.grid_edges(units)
    assert g["wide"] == {"units": 3, "kappa_at_edge": 1, "eta_at_edge": 1, "either_at_edge": 2, "kept_last_round": 1,
                         "kappa_at_low_end": 1, "kappa_at_high_end": 0, "eta_at_low_end": 0, "eta_at_high_end": 1}
    assert g["level8"] == {"units": 3, "kappa_at_edge": 2, "eta_at_edge": 2, "either_at_edge": 2, "kept_last_round": 1,
                           "kappa_at_low_end": 1, "kappa_at_high_end": 1, "eta_at_low_end": 1, "eta_at_high_end": 1}


# ── the scoring pass's rebinding ─────────────────────────────────────────────


def test_the_scoring_pass_rebinds_level_8s_names_only_inside_the_pass(tmp_path):
    names = ("l8_rows", "walk_entries", "verify_inputs", "CONFIG", "OUT", "DATA", "AVAILABLE_PER_HOP")
    before = {k: getattr(L8, k) for k in names}

    def rule(ids, gate, excluded, per_hop=1000):
        return None

    with L9.level8_rebound(tmp_path / "x", rule):
        assert L8.l8_rows is rule and L8.walk_entries is L9.walk_entries_both and L8.verify_inputs is L9.verify_inputs
        assert (L8.CONFIG, L8.OUT, L8.DATA) == (L9.CONFIG, L9.OUT, tmp_path / "x") and L8.AVAILABLE_PER_HOP == L9.AVAILABLE_PER_HOP
    assert all(getattr(L8, k) is v for k, v in before.items())
    with pytest.raises(RuntimeError):
        with L9.level8_rebound(tmp_path / "x", rule):
            raise RuntimeError("inside the pass")
    assert all(getattr(L8, k) is v for k, v in before.items())


# ── the stages end to end on the synthetic combined sidecar ──────────────────


def _stage_env(tmp_path, monkeypatch, n_q, seed):
    d = tmp_path / "l9" / "metaqa"
    qids = _sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    cfg = tmp_path / "mp_approx_l9.yaml"
    shutil.copyfile(L9.CONFIG, cfg)   # the file stage appends its run record to a copy
    monkeypatch.setattr(L9, "CONFIG", cfg)
    monkeypatch.setattr(L9, "DATA", d)
    monkeypatch.setattr(L9, "RECORD", tmp_path / "l9" / "record.json")
    monkeypatch.setattr(L9, "DOC", tmp_path / "MP_APPROX_L9.md")
    monkeypatch.setattr(L9, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "fit_process", _light_fit_process)
    monkeypatch.setattr(L8, "FIT_THREADS", 1)   # the check and read stages set it; light on the laptop
    return d, cfg


def test_check_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    d, cfg = _stage_env(tmp_path, monkeypatch, n_q=90, seed=4)
    monkeypatch.setattr(L9, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(L8, "EM_ROUNDS", 2)
    monkeypatch.setattr(L9, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    decl = L9.load_declaration()
    check = L9.stage_check(decl, _quiet)
    fam = check["families"]
    assert all(fam[f]["direction_check"]["passes"] for f in L9.FAMILIES) and fam["std"]["chain_reach"]["b0"]["all"] == 1.0
    assert fam["std"]["chain_fit"]["recall"]["all"] == 1.0 and fam["std"]["chain_fit"]["precision"]["all"] == 1.0
    assert fam["nb"]["chain_fit"]["recall"]["all"] <= 1.0
    assert fam["nb"]["sizes"]["entries_total"] < fam["std"]["sizes"]["entries_total"]
    assert 0.0 <= check["nb_trim"]["r_star_share_removed"]["all"] <= 1.0
    for arm in L9.ARMS:
        L9.stage_fit(decl, arm, _quiet)
    L9.stage_repeat(decl, _quiet)
    rd = L9.stage_read(decl, _quiet)
    assert rd["reading"] == rd["arms"]["NB-hyb"]["band"] and rd["primary"] == "NB-hyb"
    assert set(rd["arms"]) == set(L9.ARMS) | {a + "@L8" for a in L9.ARMS} | {"NB-oracle", "STD-oracle"}
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"]
    assert set(rd["contrasts"]) == set(L9.CONTRASTS) and set(rd["strata"]) == {"hop=1", "hop=2", "hop=3"}
    assert set(rd["anchors"]["agreement"]) == set(L9.ARMS) and set(rd["anchors"]["grid_edges"]) == set(L9.ARMS)
    assert rd["arms"]["STD-oracle"]["mean"]["recall@5"]["arm"] >= rd["arms"]["STD-oracle"]["mean"]["recall@5"]["twin"]
    data = L8.Data(d, check=False)
    for arm in L9.ARMS:   # every query exactly once, out of fold, under both protocols
        for fold in range(5):
            with np.load(d / "units" / arm / f"k0_f{fold}.npz") as z:
                assert np.array_equal(z["q"], np.flatnonzero(data.q_fold == fold))
                assert z["metrics_l8"].shape == z["metrics"].shape and z["argmax_l8"].shape == z["argmax"].shape
            flog = json.loads((d / "units" / arm / f"k0_f{fold}.json").read_text(encoding="utf-8"))
            assert flog["family"] == L9.ARM_SPEC[arm][0] and flog["deterministic_algorithms"] is True and len(flog["rounds"]) == 4
    L9.stage_doc(_quiet)
    body = L9.DOC.read_text(encoding="utf-8")
    assert "NB-oracle" in body and "STD-text@L8" in body and "It is not the within-U_q oracle rho" in body
    assert not any(p in body.lower() for p in ("message passing is unnecessary", "we do not need message passing", "the mlp wins"))
    rec = json.loads(L9.RECORD.read_text(encoding="utf-8"))
    assert rec["sources_sha256"]["read.json"] == L0.sha256_file(d / "read.json")
    L9.stage_file("2026_10_01", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l9_2026_10_01"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert set(run["contrasts"]) == set(L9.CONTRASTS)
    with pytest.raises(SystemExit, match="exists"):
        L9.stage_file("2026_10_01", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_a_failed_direction_check_in_either_family_stops_before_any_fit(tmp_path, monkeypatch, stops, one_thread):
    d, _cfg = _stage_env(tmp_path, monkeypatch, n_q=20, seed=6)
    real = L8.chain_tokens
    with monkeypatch.context() as m:
        m.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))   # the map, turned around
        with pytest.raises(SystemExit, match="HARD STOP"):
            L9.stage_check(L9.load_declaration(), _quiet)
    assert "direction_check (std)" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="check stage has not passed"):
        L9.stage_fit(L9.load_declaration(), "NB-hyb", _quiet)
    real_family = L9.check_family

    def nb_fails(view, chains, qt_of):
        out, rstars = real_family(view, chains, qt_of)
        if view.family == "nb":
            out["direction_check"]["passes"] = False
        return out, rstars

    monkeypatch.setattr(L9, "check_family", nb_fails)
    with pytest.raises(SystemExit, match="HARD STOP"):
        L9.stage_check(L9.load_declaration(), _quiet)
    assert "direction_check (nb)" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="check stage has not passed"):
        L9.stage_repeat(L9.load_declaration(), _quiet)


def test_the_check_refuses_a_smoke_sidecar_and_a_level8_id(tmp_path, monkeypatch, stops, one_thread):
    d, _cfg = _stage_env(tmp_path, monkeypatch, n_q=12, seed=8)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "meta.json").write_text(json.dumps({**meta, "limit": 12}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke"):
        L9.stage_check(L9.load_declaration(), _quiet)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    decl = L9.load_declaration()
    fake = tmp_path / "level8_qids.json"
    fake.write_text(json.dumps([json.loads((d / "qids.json").read_text(encoding="utf-8"))[3]]), encoding="utf-8")
    decl["inputs"]["level8"]["excluded_rows"]["qids"]["path"] = str(fake)
    with pytest.raises(SystemExit, match="HARD STOP"):
        L9.stage_check(decl, _quiet)
    assert "level 8 query id" in (stops / "hard_stops.json").read_text(encoding="utf-8")


def test_the_identical_code_check_needs_this_script_in_one_committed_set(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    good = {"job1": {L8.SCRIPT_REL: "a" * 64, L9.SCRIPT_REL: "a" * 64}, "job2": {L8.SCRIPT_REL: "a" * 64, L9.SCRIPT_REL: "a" * 64}}
    assert L9.code_problems(good, "HEAD") == []
    assert any("mp_approx_l9.py is in no job's record" in p for p in L9.code_problems({"j": {L8.SCRIPT_REL: "a" * 64}}, "HEAD"))
    bad = {"job1": {L8.SCRIPT_REL: "a" * 64, L9.SCRIPT_REL: "a" * 64}, "job2": {L8.SCRIPT_REL: "a" * 64, L9.SCRIPT_REL: "b" * 64}}
    assert any("2 different" in p for p in L9.code_problems(bad, "HEAD"))


def test_the_stages_refuse_the_wrong_machine_and_a_smoke_under_the_outputs(stops):
    for argv in (["--stage", "check"], ["--stage", "fit", "--arm", "NB-hyb"], ["--stage", "read"], ["--stage", "doc", "--host"],
                 ["--stage", "file", "--host"], ["--stage", "fit", "--host"], ["--stage", "check", "--host", "--shard", "0/3"],
                 ["--stage", "score", "--host", "--limit", "4"], ["--stage", "check", "--host", "--limit", "4", "--out", "x"],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L9.OUT / "x")],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L9.OUT)]):
        with pytest.raises(SystemExit):
            L9.main(argv)


# ── no held, level 0 or level 8 query in any sidecar ─────────────────────────


def test_no_held_level0_or_level8_query_id_appears_in_any_sidecar():
    side = L9.DATA / "qids.json"
    if not side.exists():
        pytest.skip("this file's sidecar is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"]["metaqa"]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    got = json.loads(side.read_text(encoding="utf-8"))
    inp = L9.load_declaration()["inputs"]
    l0 = set(json.loads((ROOT / inp["level0"]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
    l8 = set(json.loads((ROOT / inp["level8"]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
    assert got and set(got) <= gate and not (set(got) & l0) and not (set(got) & l8) and len(got) == len(set(got))
    assert len(got) == 3 * L9.PER_HOP
    meta = L9.DATA / "meta.json"
    if meta.exists():
        assert json.loads(meta.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)
