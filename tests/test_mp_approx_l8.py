"""MP-Approx level 8 (configs/mp_approx_l8.yaml#tests): the declaration and its pins; the population rule on toy ids; the
walk counts against a brute-force enumeration and their edge-order invariance; the tokens and their stops; the 49 metaqa
qtypes' chains and the direction swap; the arms' inputs and the mixture; rotary, bag, untied and hashed composition; the
E-step against Bayes' rule and EM on a planted chain; the metrics, rho, readability, bands and contrasts; the check, fit,
repeat, read and doc stages end to end on a synthetic sidecar; and no held or level 0 query in any sidecar."""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import math
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
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l4 as L4  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402

QTYPES = {
    1: ["actor_to_movie", "director_to_movie", "writer_to_movie", "tag_to_movie", "movie_to_actor", "movie_to_director",
        "movie_to_writer", "movie_to_genre", "movie_to_imdbrating", "movie_to_imdbvotes", "movie_to_language", "movie_to_tags",
        "movie_to_year"],
    2: [f"{a}_to_movie_to_{x}" for a in ("actor", "director", "writer") for x in ("actor", "director", "genre", "language", "writer", "year")]
       + [f"movie_to_{a}_to_movie" for a in ("actor", "director", "writer")],
    3: [f"movie_to_{a}_to_movie_to_{x}" for a, xs in (("actor", ("director", "genre", "language", "writer", "year")),
                                                      ("director", ("actor", "genre", "language", "writer", "year")),
                                                      ("writer", ("actor", "director", "genre", "language", "year"))) for x in xs],
}


def _quiet(_s: str) -> None:
    pass


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])   # the fit stages turn it on


# ── the declaration and its pins ─────────────────────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256(stops):
    decl = L8.load_declaration()
    assert decl["phase"] == "MP_APPROX_L8" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"].startswith("After matching candidate exposure and inference-time graph information")
    assert list(decl["inputs"]["relations"]["order"]) == list(L8.REL_ORDER)
    assert {int(h): v for h, v in decl["population"]["available_per_hop"].items()} == L8.AVAILABLE_PER_HOP
    assert decl["inputs"]["frozen_code_lf"]["via_level0"].startswith("configs/mp_approx_l0.yaml#inputs.frozen_code (16 files")
    assert len(L0.load_declaration()["inputs"]["frozen_code"]) == 16
    L8.verify_inputs(decl)   # every pin, raw or LF, and level 0's verify_pins on metaqa; raises on any mismatch
    assert not (stops / "hard_stops.json").exists()


def test_a_changed_pin_stops(stops, monkeypatch):
    decl = L8.load_declaration()
    decl["inputs"]["frozen_code_lf"]["scripts/m3a_headroom.py"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L8.verify_inputs(decl)
    assert "m3a_headroom.py" in (stops / "hard_stops.json").read_text(encoding="utf-8")


def test_the_declared_constants_are_the_files():
    assert L8.KAPPAS == (0.25, 0.5, 1.0, 2.0, 4.0, 8.0) and L8.ETAS == (0.01, 0.1, 1.0)
    assert (L8.EPS, L8.D, L8.PLANES, L8.THETA_BASE, L8.EM_ROUNDS, L8.M_EPOCHS, L8.BATCH_Q) == (0.01, 64, 32, 100.0, 5, 10, 64)
    assert (L8.LR, L8.WD, L8.DIRECT_EPOCHS, L8.PATIENCE, L8.PER_HOP, L8.SALT) == (1e-3, 1e-4, 30, 3, 1000, "mp_approx_l8|")
    assert (L8.COL_FWD, L8.COL_BWD, L8.SLOT0, L8.K_REL, L8.A0) == (6, 7, 8, 4, 3)
    assert (L8.MAX_Q_ENTRIES, L8.MAX_ALL_ENTRIES, L8.N_TOK, L8.TB) == (20_000_000, 2_000_000_000, 27, 28)
    assert L8.ARMS == ("TP", "TP-bag", "TP-pos", "TP-hash", "TP-direct") and L8.PRIMARY == "TP" and L8.REPEAT_UNIT == ("TP", 0, 0)


# ── the population rule ──────────────────────────────────────────────────────


def test_the_population_rule_is_disjoint_from_the_excluded_rows_sorts_by_the_salted_sha256_and_keeps_the_first_n():
    rng = np.random.default_rng(0)
    ids = [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(300)]
    gate = rng.random(len(ids)) < 0.7
    excluded = L0.metaqa_subsample(ids, gate, per_hop=40)
    rows, available = L8.l8_rows(ids, gate, excluded, per_hop=50)
    assert rows.size == 150 and np.all(np.diff(rows) > 0) and gate[rows].all() and not np.isin(rows, excluded).any()
    for h in (1, 2, 3):
        pool = [i for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == h and i not in set(excluded.tolist())]
        assert available[h] == len(pool)
        want = sorted(pool, key=lambda i: hashlib.sha256(("mp_approx_l8|" + ids[i]).encode("utf-8")).hexdigest())[:50]
        assert sorted(want) == [int(r) for r in rows if L0.hop_from_id(ids[r]) == h]
    again, _ = L8.l8_rows(list(ids), gate.copy(), excluded.copy(), per_hop=50)
    assert np.array_equal(rows, again)
    with pytest.raises(SystemExit):
        L8.l8_rows(ids, gate, excluded, per_hop=10_000)


# ── the walks ────────────────────────────────────────────────────────────────


def _random_multigraph(rng, n: int, m: int):
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    tok = rng.integers(0, L8.N_TOK, m)
    extra = rng.integers(0, m, m // 4)   # parallel edges with the same or another token
    src, dst = np.r_[src, src[extra]], np.r_[dst, dst[extra]]
    tok = np.r_[tok, rng.integers(0, L8.N_TOK, extra.size)]
    return src, dst, tok


def _brute_force(src, dst, tok, n, buckets) -> dict:
    """Every walk of length 1 to 3 from each bucket's seeds, enumerated edge by edge."""
    out = {}
    out_edges = [[] for _ in range(n)]
    for e in range(src.size):
        out_edges[int(src[e])].append(e)
    for b, seeds in enumerate(buckets):
        frontier = [(int(s), ()) for s in seeds]
        for _length in range(1, 4):
            nxt = []
            for node, toks in frontier:
                for e in out_edges[node]:
                    t = toks + (int(tok[e]),)
                    nxt.append((int(dst[e]), t))
                    key = (L8.type_code(b, t), int(dst[e]))
                    out[key] = out.get(key, 0) + 1
            frontier = nxt
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_the_walk_counts_of_both_buckets_equal_a_brute_force_enumeration(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 14))
    src, dst, tok = _random_multigraph(rng, n, int(rng.integers(10, 30)))
    seeds = rng.choice(n, size=4, replace=False)
    buckets = [seeds[:2], seeds[2:]]
    code, node, count = L8.walk_entries(src, dst, tok, n, buckets, block=7)
    got = {(int(c), int(v)): int(k) for c, v, k in zip(code, node, count)}
    assert got == _brute_force(src, dst, tok, n, buckets)
    assert np.all(np.diff(code) >= 0) and len(got) == code.size and (count > 0).all()
    b, toks, lengths = L8.decode_types(code)
    assert set(np.unique(b)) <= {0, 1} and set(np.unique(lengths)) <= {1, 2, 3}


def test_the_walk_counts_do_not_depend_on_the_order_of_the_edges_or_the_block():
    rng = np.random.default_rng(5)
    n = 12
    src, dst, tok = _random_multigraph(rng, n, 40)
    buckets = [np.array([0, 3]), np.array([7])]
    ref = L8.walk_entries(src, dst, tok, n, buckets)
    for trial in range(3):
        perm = np.random.default_rng(trial).permutation(src.size)
        for block in (1, 5, 10**6):
            got = L8.walk_entries(src[perm], dst[perm], tok[perm], n, buckets, block=block)
            assert all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_a_walk_ceiling_and_a_seed_outside_the_pool_stop():
    rng = np.random.default_rng(1)
    src, dst, tok = _random_multigraph(rng, 10, 60)
    with pytest.raises(L8.WalkCeiling):
        L8.walk_entries(src, dst, tok, 10, [np.array([0, 1])], cap=5)
    with pytest.raises(ValueError, match="outside the pool"):
        L8.walk_entries(src, dst, tok, 10, [np.array([10])])


def test_the_type_code_is_the_declared_one_and_decodes():
    assert L8.type_code(0, [4]) == ((0 * 28 + 5) * 28 + 0) * 28 + 0
    assert L8.type_code(1, [0, 26, 3]) == ((1 * 28 + 1) * 28 + 27) * 28 + 4
    codes = [L8.type_code(b, t) for b in (0, 1) for t in ([2], [5, 7], [26, 0, 13])]
    b, toks, lengths = L8.decode_types(codes)
    assert b.tolist() == [0, 0, 0, 1, 1, 1] and lengths.tolist() == [1, 2, 3] * 2
    assert toks[2].tolist() == [26, 0, 13] and toks[1].tolist() == [5, 7, -1]


def test_the_tokens_take_the_distinct_relations_of_the_slots_and_the_direction_follows_the_flags():
    slots = np.array([[3, -1, -1, -1], [8, 2, 8, -1], [0, 0, 0, 0], [5, -1, 1, -1]])
    fwd = np.array([1.0, 0.0, 1.0, 1.0])
    bwd = np.array([0.0, 1.0, 1.0, 0.0])
    e, t = L8.edge_tokens(slots, fwd, bwd)
    assert e.tolist() == [0, 1, 1, 2, 3, 3]
    assert t.tolist() == [3 * 3 + 0, 3 * 2 + 1, 3 * 8 + 1, 3 * 0 + 2, 3 * 1 + 0, 3 * 5 + 0]
    e2, t2 = L8.edge_tokens(slots[:, ::-1], fwd, bwd)   # the order of the slots does not matter
    assert e2.tolist() == e.tolist() and t2.tolist() == t.tolist()
    with pytest.raises(ValueError, match="without a relation slot"):
        L8.edge_tokens(np.array([[-1, -1, -1, -1]]), np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="without a direction flag"):
        L8.edge_tokens(np.array([[2, -1, -1, -1]]), np.array([0.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="outside the relation vocabulary"):
        L8.edge_tokens(np.array([[9, -1, -1, -1]]), np.array([1.0]), np.array([0.0]))
    loc = L3.slots_local(np.array([[4.0, -1.0, 13.0, -1.0]]), 4)   # level 3's reading of bank rows
    assert loc.tolist() == [[0, -1, 9, -1]]


# ── the chain map ────────────────────────────────────────────────────────────


def test_the_chain_map_parses_all_49_qtypes_to_their_hops_length_and_the_swap_is_an_involution():
    assert [len(QTYPES[h]) for h in (1, 2, 3)] == [13, 21, 15]
    R = {r: i for i, r in enumerate(L8.REL_ORDER)}
    for h, qts in QTYPES.items():
        for qt in qts:
            steps = L8.true_chain(qt)
            assert len(steps) == h
            toks = L8.chain_tokens(steps)
            assert L8.chain_tokens([(t // 3, t % 3) for t in L8.chain_tokens(steps, swap=True)], swap=True) == toks
            assert all(0 <= t < L8.N_TOK and t % 3 in (0, 1) for t in toks)
    assert L8.true_chain("movie_to_actor") == [(R["starred_actors"], 0)]
    assert L8.true_chain("actor_to_movie") == [(R["starred_actors"], 1)]
    assert L8.true_chain("tag_to_movie") == [(R["has_tags"], 1)]
    assert L8.true_chain("movie_to_imdbvotes") == [(R["has_imdb_votes"], 0)]
    assert L8.true_chain("director_to_movie_to_year") == [(R["directed_by"], 1), (R["release_year"], 0)]
    assert L8.true_chain("movie_to_writer_to_movie") == [(R["written_by"], 0), (R["written_by"], 1)]
    assert L8.true_chain("movie_to_actor_to_movie_to_language") == [(R["starred_actors"], 0), (R["starred_actors"], 1), (R["in_language"], 0)]
    for bad in ("movie_to_movie", "actor_to_director", "movie", "movie_to_budget"):
        with pytest.raises(ValueError):
            L8.true_chain(bad)


# ── the model ────────────────────────────────────────────────────────────────


def _table(codes):
    return L8.TypeTable(np.asarray(codes, dtype=np.int64))


def _rel(seed=0):
    r = np.random.default_rng(seed).normal(size=(L8.N_REL, 1536)).astype(np.float32)
    return r / np.linalg.norm(r, axis=1, keepdims=True)


def test_rotary_composition_is_order_aware_and_norm_preserving_the_bag_is_not_and_tp_pos_at_the_identity_is_the_bag():
    torch.manual_seed(0)
    t1, t2 = 4, 19
    table = _table([L8.type_code(0, [t1, t2]), L8.type_code(0, [t2, t1])])
    tp, bag, pos = (L8.TypeModel(a, _rel(), table) for a in ("TP", "TP-bag", "TP-pos"))
    for m in (bag, pos):
        m.load_state_dict({k: v for k, v in tp.state_dict().items() if k in m.state_dict()}, strict=False)
    with torch.no_grad():
        e_tp = tp.compose(tp.tok, tp.tl, tp.thb)
        e_bag = bag.compose(bag.tok, bag.tl, bag.thb)
        e_pos = pos.compose(pos.tok, pos.tl, pos.thb)
        assert not torch.allclose(e_tp[0], e_tp[1], atol=1e-5)
        assert torch.allclose(e_bag[0], e_bag[1], atol=1e-6)
        assert torch.allclose(e_pos, e_bag, atol=1e-6)
        x = torch.randn(5, L8.D)
        ang = torch.rand(L8.PLANES) * 3
        assert torch.allclose(L8.rotate(x, ang).norm(dim=1), x.norm(dim=1), atol=1e-5)
        assert torch.allclose(L8.rotate(x, torch.zeros(L8.PLANES)), x)
    assert torch.allclose(tp.theta, L8.THETA_BASE ** (-torch.arange(32, dtype=torch.float32) / 32))
    assert torch.equal(pos.M, torch.eye(L8.D).repeat(3, 1, 1))


def test_tp_hash_buckets_are_level_4s_mix64_and_salt_h_rule():
    rng = np.random.default_rng(3)
    toks = np.full((50, 3), -1, dtype=np.int64)
    lengths = rng.integers(1, 4, 50)
    for i, L in enumerate(lengths):
        toks[i, :L] = rng.integers(0, 27, L)
    got = L8.hash_buckets(toks)
    for i in range(50):
        want = 0
        for pos in range(3):
            if toks[i, pos] >= 0:
                z = L4.mix64(np.array([toks[i, pos]], dtype=np.uint64) ^ np.uint64(L4.SALT_H[pos]))
                want += int(z[0] % np.uint64(64))
        assert got[i] == want % 64
    table = _table([L8.type_code(0, [3, 4]), L8.type_code(1, [3, 4])])
    m = L8.TypeModel("TP-hash", _rel(), table)
    assert not hasattr(m, "P") and not hasattr(m, "delta") and m.H.shape == (3, 64, L8.D)


def test_each_arm_reads_only_the_query_the_relation_text_and_the_type_table():
    table = _table([L8.type_code(0, [1]), L8.type_code(1, [2, 5]), L8.type_code(0, [26, 0, 7])])
    allowed_buffers = {"rel_emb", "tok", "tb", "tl", "thb"}
    params = {"TP": {"A.weight", "P.weight", "delta", "theta", "beta", "lam", "c", "nu", "c_null"},
              "TP-bag": {"A.weight", "P.weight", "delta", "beta", "lam", "c", "nu", "c_null"},
              "TP-pos": {"A.weight", "P.weight", "delta", "M", "beta", "lam", "c", "nu", "c_null"},
              "TP-hash": {"A.weight", "H", "beta", "lam", "c", "nu", "c_null"},
              "TP-direct": {"A.weight", "P.weight", "delta", "theta", "beta", "lam", "c", "alpha"}}
    for arm in L8.ARMS:
        m = L8.TypeModel(arm, _rel(), table)
        assert {k for k, _ in m.named_buffers()} == allowed_buffers
        assert {k for k, _ in m.named_parameters()} == params[arm]
        assert list(inspect.signature(m.weights).parameters) == ["qemb", "idx"]
        assert 190_000 < L8.n_params(m) < 215_000 or arm == "TP-hash"   # about 200,000 (compute.fits)
        for k in ("beta", "lam", "c"):
            assert float(getattr(m, k).abs().sum()) == 0.0
    m = L8.TypeModel("TP-direct", _rel(), table)
    assert abs(float(torch.nn.functional.softplus(m.alpha)) - 1.0) < 1e-6
    for key in L8.ARRAY_KEYS:   # the sidecar keeps no edge list and no neighbour array
        assert key.startswith(("q_", "t_", "e_")) or key in ("twin_score", "is_gold")
    assert "edge_index" not in L8.ARRAY_KEYS and "edge_attr" not in L8.ARRAY_KEYS


# ── a synthetic sidecar ──────────────────────────────────────────────────────


QT_TOY = ("movie_to_actor", "actor_to_movie", "movie_to_director_to_movie", "writer_to_movie_to_year")


def _toy_sidecar(d: Path, n_q: int = 80, seed: int = 0, planted: bool = True) -> list[str]:
    """A sidecar in the scoring pass's layout: random typed pools whose golds are the true chain's b0 reach set (planted),
    the query embedding naming the qtype, and stored T and G metrics in which G lifts the golds."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    parts = {k: [] for k in L8.ARRAY_KEYS}
    qids = []
    qtypes = sorted(QT_TOY)
    i = 0
    while len(qids) < n_q:
        i += 1
        qt = QT_TOY[int(rng.integers(len(QT_TOY)))]
        steps = L8.true_chain(qt)
        hop = len(steps)
        n = int(rng.integers(25, 45))
        m = int(rng.integers(3 * n, 5 * n))
        src, dst, tok = _random_multigraph(rng, n, m)
        seeds = rng.choice(n, size=int(rng.integers(3, 6)), replace=False)
        bucket = np.r_[0, 0, np.ones(seeds.size - 2, dtype=np.int64)]
        chain = L8.chain_tokens(steps)
        # plant the chain from the first seed to a handful of targets
        for tgt in rng.choice(n, size=3, replace=False):
            at = int(seeds[0])
            for pos, t in enumerate(chain):
                nxt = int(tgt) if pos == len(chain) - 1 else int(rng.integers(n))
                src, dst, tok = np.r_[src, at], np.r_[dst, nxt], np.r_[tok, t]
                at = nxt
        code, node, count = L8.walk_entries(src, dst, tok, n, [seeds[bucket == 0], seeds[bucket == 1]])
        is_gold = np.zeros(n, dtype=bool)
        rows = code == L8.type_code(0, chain)
        reach = node[rows]
        if planted:
            if reach.size < 2 or reach.size > n // 2:
                continue
            is_gold[reach] = True
        else:
            is_gold[rng.choice(n, size=2, replace=False)] = True
        t_code, t_size, t_gold = L8.type_table(code, node, is_gold)
        qid = f"metaqa:{hop}hop:toy:{i}"   # never a real id, so never a level 0 row
        gl = np.flatnonzero(is_gold)
        twin = rng.normal(size=(n, 3)).astype(np.float32)
        gnn = twin + 2.0 * is_gold[:, None]
        metrics = [[rank_metrics(s[:, k].astype(np.float64), gl, gl.size)[mm] for mm in METRIC_NAMES] for s in (twin, gnn) for k in range(3)]
        seed_row, bucket_row = np.full(10, -1), np.full(10, -1)
        seed_row[:seeds.size], bucket_row[:seeds.size] = seeds, bucket
        emb = rng.normal(scale=0.01, size=1536).astype(np.float32)
        emb[qtypes.index(qt)] += 5.0
        for key, val in (("q_row", len(qids)), ("q_hop", hop), ("q_qtype", qtypes.index(qt)), ("q_fold", L0.fold_of(qid)),
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


def test_the_mixture_equals_a_direct_sum_over_the_types(tmp_path):
    _toy_sidecar(tmp_path / "s", n_q=12, seed=1)
    data = L8.Data(tmp_path / "s")
    fx = L8.Fitter(data, _rel())
    rng = np.random.default_rng(0)
    for q in range(data.n_q):
        T = int(data.q_types[q])
        logit = rng.normal(size=T + 1)
        lp = logit - np.log(np.exp(logit).sum())
        n = int(data.q_pool_size[q])
        want = np.full(n, np.exp(lp[-1]) / n)
        codes = data.t_code[data.type_rows(q)]
        for t in range(T):
            reach = data.reach(q, int(codes[t]))
            want[reach] += np.exp(lp[t]) / reach.size
        assert np.allclose(fx.mixture(q, lp), n * want, rtol=1e-12, atol=1e-12)
        assert abs(fx.mixture(q, lp).sum() - n) < 1e-9


def test_the_e_step_is_bayes_rule_on_a_toy(tmp_path):
    _toy_sidecar(tmp_path / "s", n_q=6, seed=2)
    data = L8.Data(tmp_path / "s")
    fx = L8.Fitter(data, _rel())
    qs = np.arange(data.n_q)
    rng = np.random.default_rng(1)
    logps = []
    for q in qs:
        a = rng.normal(size=int(data.q_types[q]) + 1)
        logps.append(a - np.log(np.exp(a).sum()))
    gammas, mll = fx.e_step(logps, qs)
    for q in qs:
        n, g = int(data.q_pool_size[q]), data.gold_local(q)
        codes = data.t_code[data.type_rows(q)]
        lik = []
        for c in codes:
            R = set(data.reach(q, int(c)).tolist())
            lik.append(np.prod([(1 - 0.01) * (x in R) / len(R) + 0.01 / n for x in g]))
        lik.append(float(n) ** -g.size)
        joint = np.exp(logps[q]) * np.asarray(lik)
        assert np.allclose(gammas[q], joint / joint.sum(), rtol=1e-9, atol=1e-12)
        assert math.isclose(mll[q], math.log(joint.sum()), rel_tol=1e-9)


def test_em_recovers_a_planted_chain_from_the_query(tmp_path, one_thread):
    _toy_sidecar(tmp_path / "s", n_q=240, seed=3)
    data = L8.Data(tmp_path / "s")
    fx = L8.Fitter(data, _rel())
    arrays, flog = L8.fit_unit(fx, "TP", 0, 0, _quiet)
    assert flog["kept_round"] > 0 and flog["rounds"][flog["kept_round"]]["inner_mll"] > flog["rounds"][0]["inner_mll"]
    qt = [data.meta["qtypes"][i] for i in data.q_qtype[arrays["q"]]]
    agree = np.mean([L8.token_sequence(int(c)) == tuple(L8.chain_tokens(L8.true_chain(t))) for c, t in zip(arrays["argmax"], qt)])
    assert agree >= 0.8, agree


# ── the statistics and the readings ──────────────────────────────────────────


def test_the_bands_come_in_the_declared_order():
    assert L8.band(1.3, [1.05, 1.5], True) == "L8_ABOVE_GNN"   # before L8_HIGH
    assert L8.band(0.9, [0.6, 1.2], True) == "L8_HIGH"
    assert L8.band(0.8, [0.45, 1.0], True) == "L8_MID"
    assert L8.band(0.2, [0.0, 0.45], True) == "L8_LOW"
    assert L8.band(0.2, [0.0, 0.55], True) == "L8_MID"
    assert L8.band(0.9, [0.6, 1.2], False) == "NOT_READ"
    assert L8.band(float("nan"), [0, 0], True) == "NOT_READ"


def test_rho_readability_and_the_gap_to_the_gnn_on_toy_values():
    rng = np.random.default_rng(0)
    n = 400
    T = rng.random((n, 3, 3)) * 0.5
    G = T + 0.2 + 0.01 * rng.random((n, 3, 3))
    G[:, :, 2] = T[:, :, 2]   # hit@1: no gain, so it is not readable
    W = L0.boot_weights(n)
    den, dens, readable = L8.denominators(T, G, W)
    assert readable == ["recall@5", "full_coverage@5"] and not den["hit@1"]["readable"]
    for frac in (0.0, 0.5, 1.0):
        arm = T + frac * (G - T)
        out = L8.read_arm(arm, T, G, dens, readable, W)
        assert abs(out["rho_bar"]["point"] - frac) < 1e-9
        assert all(abs(out["rho"][m]["point"] - frac) < 1e-9 for m in readable)
    top = L8.read_arm(G + 0.05, T, G, dens, readable, W)
    assert top["gap_to_gnn"]["recall@5"]["flag"] == "BEATS_GNN" and top["band"] in ("L8_HIGH", "L8_ABOVE_GNN")
    low = L8.read_arm(T, T, G, dens, readable, W)
    assert low["gap_to_gnn"]["recall@5"]["flag"] == "BELOW_GNN" and low["band"] == "L8_LOW"
    none_read = L8.read_arm(T, T, T, *L8.denominators(T, T, W)[1:], W)
    assert none_read["band"] == "NOT_READ" and none_read["rho_bar"]["point"] is None


def test_the_contrasts_are_the_declared_pairs():
    assert L8.CONTRASTS == {"order": ("TP", "TP-bag"), "untied_positions": ("TP-pos", "TP"), "text_vs_hash": ("TP", "TP-hash"),
                            "latent_vs_direct": ("TP", "TP-direct"), "ceiling_gap": ("TP-oracle", "TP")}
    assert L8.RETRIEVAL == ("recall@5", "full_coverage@5", "hit@1")


def test_nmi_is_one_for_a_relabelling_and_zero_for_independence():
    x = ["a", "b", "c", "a", "b", "c"] * 5
    assert abs(L8.nmi(x, [{"a": 1, "b": 2, "c": 3}[v] for v in x]) - 1.0) < 1e-12
    assert abs(L8.nmi(["a", "a", "b", "b"], ["x", "y", "x", "y"])) < 1e-12
    seqs = [(1, 2), (3,), ("null",), (1, 2)]   # argmax token sequences are tuples of different lengths
    assert abs(L8.nmi(seqs, ["p", "q", "r", "p"]) - 1.0) < 1e-12


# ── the stages end to end on the synthetic sidecar ───────────────────────────


def test_check_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    d = tmp_path / "l8" / "metaqa"
    qids = _toy_sidecar(d, n_q=90, seed=4)
    monkeypatch.setattr(L8, "DATA", d)
    monkeypatch.setattr(L8, "RECORD", tmp_path / "l8" / "record.json")
    monkeypatch.setattr(L8, "DOC", tmp_path / "MP_APPROX_L8.md")
    monkeypatch.setattr(L8, "SEEDS", (0,))
    monkeypatch.setattr(L8, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: _rel())
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(L8, "EM_ROUNDS", 2)
    monkeypatch.setattr(L8, "DIRECT_EPOCHS", 3)
    decl = L8.load_declaration()
    check = L8.stage_check(decl, _quiet)
    assert check["direction_check"]["passes"] and check["anchors"]["chain_reach"]["b0"]["all"] == 1.0
    assert check["anchors"]["chain_fit"]["recall"]["all"] == 1.0 and check["anchors"]["chain_fit"]["precision"]["all"] == 1.0
    for arm in L8.ARMS:
        L8.stage_fit(decl, arm, _quiet)
    L8.stage_repeat(decl, _quiet)
    rd = L8.stage_read(decl, _quiet)
    assert rd["reading"] == rd["arms"]["TP"]["band"] and set(rd["arms"]) == set(L8.ARMS) | {"TP-oracle"}
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"]
    assert json.loads((d / "units" / "TP" / "k0_f0.json").read_text(encoding="utf-8"))["deterministic_algorithms"] is True
    assert rd["arms"]["TP-oracle"]["mean"]["recall@5"]["arm"] >= rd["arms"]["TP-oracle"]["mean"]["recall@5"]["twin"]
    assert set(rd["contrasts"]) == set(L8.CONTRASTS) and set(rd["strata"]) == {"hop=1", "hop=2", "hop=3"}
    for arm in L8.ARMS:   # every query exactly once, out of fold
        for fold in range(5):
            with np.load(d / "units" / arm / f"k0_f{fold}.npz") as z:
                assert np.array_equal(z["q"], np.flatnonzero(L8.Data(d, check=False).q_fold == fold))
    L8.stage_doc(_quiet)
    body = L8.DOC.read_text(encoding="utf-8")
    assert "TP-oracle" in body and "rho here is not the within-U_q oracle rho" in body
    assert not any(p in body.lower() for p in ("message passing is unnecessary", "we do not need message passing", "the mlp wins"))
    rec = json.loads(L8.RECORD.read_text(encoding="utf-8"))
    assert rec["sources_sha256"]["read.json"] == L0.sha256_file(d / "read.json")
    assert not (stops / "hard_stops.json").exists()


def test_a_failed_direction_check_stops_before_any_fit(tmp_path, monkeypatch, stops):
    d = tmp_path / "l8" / "metaqa"
    qids = _toy_sidecar(d, n_q=20, seed=6)
    monkeypatch.setattr(L8, "DATA", d)
    monkeypatch.setattr(L8, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    real = L8.chain_tokens
    monkeypatch.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))   # the map, turned around
    with pytest.raises(SystemExit, match="HARD STOP"):
        L8.stage_check(L8.load_declaration(), _quiet)
    assert "direction_check" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="check stage has not passed"):
        L8.stage_fit(L8.load_declaration(), "TP", _quiet)


def test_the_identical_code_check_needs_one_committed_set(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    good = {"job1": {L8.SCRIPT_REL: "a" * 64}, "job2": {L8.SCRIPT_REL: "a" * 64}}
    assert L8.code_problems(good, "HEAD") == []
    bad = {"job1": {L8.SCRIPT_REL: "a" * 64}, "job2": {L8.SCRIPT_REL: "b" * 64}}
    assert any("2 different" in p for p in L8.code_problems(bad, "HEAD"))
    assert any("not the file" in p for p in L8.code_problems({"j": {L8.SCRIPT_REL: "c" * 64}}, "HEAD"))


def test_the_stages_refuse_the_wrong_machine(stops):
    with pytest.raises(SystemExit):
        L8.main(["--stage", "check"])
    with pytest.raises(SystemExit):
        L8.main(["--stage", "doc", "--host"])
    with pytest.raises(SystemExit):
        L8.main(["--stage", "score", "--host", "--limit", "4", "--out", str(L8.OUT / "x")])


# ── no held or level 0 query in any sidecar ──────────────────────────────────


def test_no_held_or_level0_query_id_appears_in_any_sidecar():
    side = L8.DATA / "qids.json"
    if not side.exists():
        pytest.skip("this file's sidecar is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"]["metaqa"]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    got = json.loads(side.read_text(encoding="utf-8"))
    l0 = set(json.loads((ROOT / L8.load_declaration()["inputs"]["level0"]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
    assert got and set(got) <= gate and not (set(got) & l0) and len(got) == len(set(got))
    meta = L8.DATA / "meta.json"
    if meta.exists():
        assert json.loads(meta.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)


def test_the_toy_types_cover_every_length_and_bucket(tmp_path):
    _toy_sidecar(tmp_path / "s", n_q=10, seed=7)
    data = L8.Data(tmp_path / "s")
    b, _toks, lengths = L8.decode_types(data.t_code)
    assert set(b.tolist()) == {0, 1} and set(lengths.tolist()) == {1, 2, 3}
    assert all(itertools.starmap(lambda s, e: s <= e, zip(data.entry_ptr[:-1], data.entry_ptr[1:])))
