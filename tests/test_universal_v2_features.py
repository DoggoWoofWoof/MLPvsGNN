"""The v2 depth basis on hand-built graphs (configs/universal_v2.yaml
#authorization_2026_09_19.step_2_authorised.tests_required_beyond_shapes):
exact propagation at t = 1, 2, 3 on a chain; rings that exclude every earlier
hop; seed-conditioned prototypes over deterministically selected nodes;
direction reversal changing the direction descriptor and the ordered
direction columns and nothing else; relation order r1 -> r2 vs r2 -> r1
distinguished by the query-modulated propagation and, at the endpoint itself,
by the ordered relation-path channel of amendment 2 (the invariant relpath_*
summaries stay invariant); the ordered channel traced against the invariant
maximum, its tie rule, its absence; missing STRUCT / NER / KNN channels
setting the masks; the first 111 raw columns equal to the M3B module; the
contract counts (73 + 13 = 86 v2 columns, 78 + 86 = 164 screen input); the
cache round trip."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mp_retrieval import m3b_features as F  # noqa: E402
from mp_retrieval import m3b_pools as P  # noqa: E402
from mp_retrieval import m3b_train as T  # noqa: E402
from mp_retrieval import universal_v2_features as V  # noqa: E402


class _Nodes:
    def __init__(self, matrix):
        self.matrix = matrix

    def read(self, rows, dtype=np.float32):
        return self.matrix[np.asarray(rows)].astype(dtype)


def _graph(rng, n, m, weighted, typed, n_rel=4):
    src = rng.integers(0, n, size=m).astype(np.int32)
    dst = rng.integers(0, n, size=m).astype(np.int32)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    return SimpleNamespace(src=src, dst=dst, weight=rng.random(src.size).astype(np.float32) if weighted else None,
                           rel=rng.integers(0, n_rel, size=src.size).astype(np.int16) if typed else None, n_nodes=n, directed=not weighted)


def _typed(n, src, dst, rel):
    return SimpleNamespace(src=np.asarray(src, np.int32), dst=np.asarray(dst, np.int32), weight=None, rel=np.asarray(rel, np.int16),
                           n_nodes=n, directed=True)


def _weighted(n, src=(), dst=(), w=()):
    return SimpleNamespace(src=np.asarray(src, np.int32), dst=np.asarray(dst, np.int32), weight=np.asarray(w, np.float32), rel=None,
                           n_nodes=n, directed=False)


def _unit(rng, shape):
    x = rng.normal(size=shape).astype(np.float32)
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _world(rng, n, struct, ner=None, knn=None, rel_emb=None, q=None, emb=None):
    stores = {"structural": P.FamilyStore.from_graph(struct, "structural"),
              "ner": P.FamilyStore.from_graph(ner if ner is not None else _weighted(n), "ner"),
              "knn": P.FamilyStore.from_graph(knn if knn is not None else _weighted(n), "knn")}
    emb = _unit(rng, (n, F.DIM)) if emb is None else emb
    rel = None if rel_emb is None else F.RelationTable.from_arrays(rel_emb, stores["structural"].rel_count, n)
    q = _unit(rng, F.DIM) if q is None else q
    return SimpleNamespace(stores=stores, nodes=_Nodes(emb), emb=emb, rel=rel, q=q, n=n)


def _compile(w, seed_nodes, pool=None, timings=None):
    """One seed list (dense = splade = the given nodes), the whole graph as the pool unless given."""
    ids = np.asarray(seed_nodes, dtype=np.int64)
    scores = np.linspace(1.0, 0.5, ids.size).astype(np.float32)
    inp = F.QueryInputs(q=w.q, dense_ids=ids, dense_scores=scores, splade_ids=ids, splade_scores=scores * 10)
    seeds = P.seeds_of(ids, ids)
    pool = np.arange(w.n, dtype=np.int64) if pool is None else np.asarray(pool, dtype=np.int64)
    return V.compile_query_v2(inp, pool, seeds, w.stores, w.nodes, w.rel, timings=timings)


def col(c, name):
    return np.asarray(c.scalars[:, V.IDX[name]], dtype=np.float64)


# ── the chain 0 - 1 - 2 - 3 - 4, seed 0 ──────────────────────────────────────


@pytest.fixture(scope="module")
def chain():
    rng = np.random.default_rng(7)
    rel_emb = _unit(rng, (3, F.DIM))
    w = _world(rng, 5, _typed(5, [0, 1, 2, 3], [1, 2, 3, 4], [0, 1, 2, 2]), rel_emb=rel_emb)
    return SimpleNamespace(w=w, c=_compile(w, [0]), rel_emb=rel_emb)


def test_fixed_propagation_is_exact_at_every_depth_on_the_chain(chain):
    """W row-normalised (degrees 1, 2, 2, 2, 1), s = e_0: W s, W^2 s, W^3 s by hand."""
    c = chain.c
    assert c.seeds_local.tolist() == [0]
    for view in V.VIEWS:
        np.testing.assert_allclose(col(c, f"support_h1_{view}"), [0, 0.5, 0, 0, 0], atol=1e-6)
        np.testing.assert_allclose(col(c, f"support_h2_{view}"), [0.5, 0, 0.25, 0, 0], atol=1e-6)
        np.testing.assert_allclose(col(c, f"support_h3_{view}"), [0, 0.375, 0, 0.125, 0], atol=1e-6)
        np.testing.assert_allclose(col(c, f"paths_h1_{view}"), np.log1p([0, 1, 0, 0, 0]), atol=1e-6)
        np.testing.assert_allclose(col(c, f"paths_h2_{view}"), np.log1p([1, 0, 1, 0, 0]), atol=1e-6)
        np.testing.assert_allclose(col(c, f"paths_h3_{view}"), np.log1p([0, 2, 0, 1, 0]), atol=1e-6)
        np.testing.assert_allclose(col(c, f"branch_h2_{view}"), np.log1p([1, 0, 1, 0, 0]), atol=1e-6)
        np.testing.assert_allclose(col(c, f"branch_h3_{view}"), np.log1p([0, 2, 0, 1, 0]), atol=1e-6)


def test_rings_exclude_every_earlier_hop_and_the_centre(chain):
    c = chain.c
    deg = [1, 2, 2, 2, 1]
    ring2 = [1, 1, 2, 1, 1]      # 0:{2} 1:{3} 2:{0,4} 3:{1} 4:{2}
    ring3 = [1, 1, 0, 1, 1]      # 0:{3} 1:{4} 2:{} 3:{0} 4:{1}
    for view in V.VIEWS:
        np.testing.assert_allclose(col(c, f"ring_n_h1_{view}"), np.log1p(deg), atol=1e-6)
        np.testing.assert_allclose(col(c, f"ring_n_h2_{view}"), np.log1p(ring2), atol=1e-6)
        np.testing.assert_allclose(col(c, f"ring_n_h3_{view}"), np.log1p(ring3), atol=1e-6)
        # the seed (node 0) is at exact distance t from node t only
        for t in V.DEPTHS:
            expect = np.zeros(5)
            expect[t] = 1.0
            np.testing.assert_allclose(col(c, f"has_h{t}_{view}"), expect)
            np.testing.assert_allclose(col(c, f"seeds_at_h{t}_{view}"), np.log1p(expect), atol=1e-6)
            np.testing.assert_allclose(col(c, f"seedmass_h{t}_{view}"), expect, atol=1e-6)     # s_0 = rrf_0 / max rrf = 1
    # ring statistics of node 2 at depth 2: nodes 0 and 4
    cos = np.asarray(c.scalars[:, F.IDX["dense_cos"]], dtype=np.float64)
    assert col(c, "ring_qmean_h2_STRUCT")[2] == pytest.approx((cos[0] + cos[4]) / 2, abs=1e-5)
    assert col(c, "ring_qmax_h2_STRUCT")[2] == pytest.approx(max(cos[0], cos[4]), abs=1e-5)
    assert col(c, "ring_qmean_h3_STRUCT")[2] == 0.0 and col(c, "ring_qmax_h3_STRUCT")[2] == 0.0


def test_seed_conditioned_prototypes_use_only_the_seeds_at_that_exact_depth():
    """Two seeds at different distances from a node: the prototype at depth 1
    is the cosine with the seed at distance 1 alone, at depth 3 with the other
    alone, and the depth-2 prototype (no seed there) is 0 with has = 0."""
    rng = np.random.default_rng(11)
    w = _world(rng, 4, _typed(4, [0, 1, 2], [1, 2, 3], [0, 0, 0]), rel_emb=_unit(rng, (1, F.DIM)))     # path 0-1-2-3, seeds 1 and 3
    c = _compile(w, [1, 3])
    assert sorted(c.seeds_local.tolist()) == [1, 3]
    e = w.emb
    for view in V.VIEWS:
        proto = (e[1] + e[3]) / 2                        # node 2: both seeds at distance 1
        assert col(c, f"seedproto_h1_{view}")[2] == pytest.approx(float(e[2] @ proto / np.linalg.norm(proto)), abs=1e-5)
        assert col(c, f"seedproto_h1_{view}")[0] == pytest.approx(float(e[0] @ e[1]), abs=1e-5)     # node 0: seed 1 at 1
        assert col(c, f"has_h2_{view}")[0] == 0.0 and col(c, f"seedproto_h2_{view}")[0] == 0.0
        assert col(c, f"seedproto_h3_{view}")[0] == pytest.approx(float(e[0] @ e[3]), abs=1e-5)     # seed 3 at 3
        assert col(c, f"seeds_at_h3_{view}")[0] == pytest.approx(np.log1p(1))


def test_reversing_a_relation_changes_the_direction_descriptor_and_the_ordered_dirs_and_nothing_else(chain):
    """Stored 1 -b-> 2 becomes 2 -b-> 1: every depth-basis column, every
    ordered q / adj column and every other ordered dir column is unchanged;
    the dir columns that read the b step flip sign, and node 2 at depth 2
    (the walk 0 -a-> 1 -b-> 2, step 2 forward before, against the stored
    relation after) is one of them."""
    rng = np.random.default_rng(7)
    w2 = _world(rng, 5, _typed(5, [0, 2, 2, 3], [1, 1, 3, 4], [0, 1, 2, 2]), rel_emb=chain.rel_emb, q=chain.w.q, emb=chain.w.emb)
    c2 = _compile(w2, [0])
    dirs = [c for c in V.ORDERED_COLUMNS if "_dir" in c]
    for name in V.V2_COLUMNS:
        if name not in dirs:
            np.testing.assert_array_equal(col(chain.c, name), col(c2, name), err_msg=name)
    flipped = 0
    for name in dirs:
        a, b = col(chain.c, name), col(c2, name)
        diff = a != b
        np.testing.assert_array_equal(a[diff], -b[diff], err_msg=name)     # a flip is a sign change, never an appearance
        flipped += int(diff.sum())
    assert flipped > 0
    assert col(chain.c, "opath_h2_dir2")[2] == 1.0 and col(c2, "opath_h2_dir2")[2] == -1.0
    assert col(chain.c, "opath_h2_dir1")[2] == 1.0 and col(c2, "opath_h2_dir1")[2] == 1.0
    u1, v1, attr1 = chain.c.edges["structural"]
    u2, v2, attr2 = c2.edges["structural"]
    np.testing.assert_array_equal(u1, u2)
    np.testing.assert_array_equal(v1, v2)
    fwd, bwd = F.EDGE_ATTR.index("dir_fwd"), F.EDGE_ATTR.index("dir_bwd")
    other = [k for k in range(len(F.EDGE_ATTR)) if k not in (fwd, bwd)]
    np.testing.assert_array_equal(attr1[:, other], attr2[:, other])
    pair = (u1 == 1) & (v1 == 2)                       # message edge 1 -> 2: stored 1 -> 2 before, 2 -> 1 after
    assert pair.sum() == 1
    assert attr1[pair, fwd] == 1.0 and attr1[pair, bwd] == 0.0
    assert attr2[pair, fwd] == 0.0 and attr2[pair, bwd] == 1.0


# ── the typed basis: relation order ──────────────────────────────────────────


def _softmax_pair(a, b, tau=V.QSUPPORT_TEMPERATURE):
    return np.exp(a / tau) / (np.exp(a / tau) + np.exp(b / tau))


def test_relation_order_r1_r2_versus_r2_r1_is_distinguished_by_the_query_modulated_propagation():
    """Chain 0 -a-> 1 -b-> 2 -c-> 3 versus 0 -b-> 1 -a-> 2 -c-> 3, seed 0. The
    relation-path summaries of the endpoint (max / mean / min over the walk)
    are order-invariant by their declared formula and the test states that;
    the order is carried by qsupport_h1 / qsupport_h2 (the per-node
    normalisation at the intermediate node depends on which relation leads
    into it), by the intermediate node's own relpath_h1 and, at the endpoint
    itself, by the ordered channel (the next test)."""
    rng = np.random.default_rng(5)
    rel_emb = _unit(rng, (3, F.DIM))
    q = _unit(rng, F.DIM)
    ca, cb, cc = (rel_emb @ q).astype(np.float64)
    assert abs(ca - cb) > 1e-3
    emb = _unit(rng, (4, F.DIM))
    w1 = _world(rng, 4, _typed(4, [0, 1, 2], [1, 2, 3], [0, 1, 2]), rel_emb=rel_emb, q=q, emb=emb)
    w2 = _world(rng, 4, _typed(4, [0, 1, 2], [1, 2, 3], [1, 0, 2]), rel_emb=rel_emb, q=q, emb=emb)
    c1, c2 = _compile(w1, [0]), _compile(w2, [0])
    # qsupport by hand: owner 1 normalises over {a (from 0), b (from 2)}; owner 2 over {b (from 1), c (from 3)}
    q1_1 = _softmax_pair(ca, cb)                          # W_q[1, 0] with a leading into 1
    q2_2 = _softmax_pair(cb, cc) * q1_1                   # W_q[2, 1] * qsupport_h1[1]
    assert col(c1, "qsupport_h1")[1] == pytest.approx(q1_1, abs=1e-5)
    assert col(c1, "qsupport_h2")[2] == pytest.approx(q2_2, abs=1e-5)
    assert col(c1, "qsupport_h2")[0] == pytest.approx(q1_1, abs=1e-5)        # owner 0 has one typed in-edge: weight 1
    q1_1s = _softmax_pair(cb, ca)
    q2_2s = _softmax_pair(ca, cc) * q1_1s
    assert col(c2, "qsupport_h1")[1] == pytest.approx(q1_1s, abs=1e-5)
    assert col(c2, "qsupport_h2")[2] == pytest.approx(q2_2s, abs=1e-5)
    assert abs(q2_2 - q2_2s) > 1e-6 and abs(col(c1, "qsupport_h2")[2] - col(c2, "qsupport_h2")[2]) > 1e-6
    # the intermediate node reads the first relation
    assert col(c1, "relpath_max_h1")[1] == pytest.approx(ca, abs=1e-5)
    assert col(c2, "relpath_max_h1")[1] == pytest.approx(cb, abs=1e-5)
    # the endpoint statistics are order-invariant: one walk 0 -> 1 -> 2 in both graphs
    for name in ("relpath_max_h2", "relpath_mean_h2"):
        assert col(c1, name)[2] == pytest.approx((ca + cb) / 2, abs=1e-5)
        assert col(c2, name)[2] == pytest.approx((ca + cb) / 2, abs=1e-5)
    assert col(c1, "relpath_min_h2")[2] == pytest.approx(min(ca, cb), abs=1e-5)
    assert col(c2, "relpath_min_h2")[2] == pytest.approx(min(ca, cb), abs=1e-5)
    assert col(c1, "typed_walks_h2")[2] == pytest.approx(np.log1p(1)) and col(c1, "typed_walks_h2")[0] == pytest.approx(np.log1p(1))
    # depth 3 at node 3: the single walk a, b, c; at node 1: two walks (a, a, a) and (a, b, b)
    assert col(c1, "relpath_mean_h3")[3] == pytest.approx((ca + cb + cc) / 3, abs=1e-5)
    assert col(c1, "relpath_min_h3")[3] == pytest.approx(min(ca, cb, cc), abs=1e-5)
    assert col(c1, "typed_walks_h3")[1] == pytest.approx(np.log1p(2))
    assert col(c1, "relpath_max_h3")[1] == pytest.approx(max(ca, (ca + 2 * cb) / 3), abs=1e-5)
    assert col(c1, "relpath_mean_h3")[1] == pytest.approx((ca + (ca + 2 * cb) / 3) / 2, abs=1e-5)
    assert col(c1, "relpath_min_h3")[1] == pytest.approx(max(ca, min(ca, cb)), abs=1e-5)


# ── the ordered relation-path channel (amendment 2) ──────────────────────────


def test_the_ordered_channel_distinguishes_r1_r2_from_r2_r1_at_the_endpoint_itself():
    """The same two chains: node 2 reads (a, b) in one graph and (b, a) in the
    other in its ordered q columns, node 3 reads (a, b, c) versus (b, a, c) and
    its adjacent cosine adj23 changes with it; the chain walks follow their
    stored relations (dir +1) while the walk 0 -> 1 -> 0 retraces a against
    it (dir -1 at step 2); the mean of the ordered q columns is the invariant
    relpath_max at every node a walk reaches; no walk -> zeros and mask 0."""
    rng = np.random.default_rng(5)
    rel_emb = _unit(rng, (3, F.DIM))
    q = _unit(rng, F.DIM)
    ca, cb, cc = (rel_emb @ q).astype(np.float64)
    emb = _unit(rng, (4, F.DIM))
    w1 = _world(rng, 4, _typed(4, [0, 1, 2], [1, 2, 3], [0, 1, 2]), rel_emb=rel_emb, q=q, emb=emb)
    w2 = _world(rng, 4, _typed(4, [0, 1, 2], [1, 2, 3], [1, 0, 2]), rel_emb=rel_emb, q=q, emb=emb)
    c1, c2 = _compile(w1, [0]), _compile(w2, [0])
    assert col(c1, "opath_h2_q1")[2] == pytest.approx(ca, abs=1e-5) and col(c1, "opath_h2_q2")[2] == pytest.approx(cb, abs=1e-5)
    assert col(c2, "opath_h2_q1")[2] == pytest.approx(cb, abs=1e-5) and col(c2, "opath_h2_q2")[2] == pytest.approx(ca, abs=1e-5)
    assert col(c1, "relpath_max_h2")[2] == pytest.approx(col(c2, "relpath_max_h2")[2], abs=1e-6)     # the invariant summary cannot tell
    cab, cbc, cac = (float(rel_emb[0] @ rel_emb[1]), float(rel_emb[1] @ rel_emb[2]), float(rel_emb[0] @ rel_emb[2]))
    assert col(c1, "opath_h2_adj12")[2] == pytest.approx(cab, abs=1e-5) and col(c2, "opath_h2_adj12")[2] == pytest.approx(cab, abs=1e-5)
    for k, want in enumerate((ca, cb, cc), start=1):
        assert col(c1, f"opath_h3_q{k}")[3] == pytest.approx(want, abs=1e-5)
    for k, want in enumerate((cb, ca, cc), start=1):
        assert col(c2, f"opath_h3_q{k}")[3] == pytest.approx(want, abs=1e-5)
    assert col(c1, "opath_h3_adj12")[3] == pytest.approx(cab, abs=1e-5) and col(c1, "opath_h3_adj23")[3] == pytest.approx(cbc, abs=1e-5)
    assert col(c2, "opath_h3_adj12")[3] == pytest.approx(cab, abs=1e-5) and col(c2, "opath_h3_adj23")[3] == pytest.approx(cac, abs=1e-5)
    for c in (c1, c2):
        assert [col(c, f"opath_h2_dir{k}")[2] for k in (1, 2)] == [1.0, 1.0] and [col(c, f"opath_h3_dir{k}")[3] for k in (1, 2, 3)] == [1.0] * 3
        assert [col(c, f"opath_h2_dir{k}")[0] for k in (1, 2)] == [1.0, -1.0]                 # 0 -a-> 1 -a-> 0: the second step is against a
        assert col(c, "opath_h2_adj12")[0] == pytest.approx(1.0, abs=1e-5)                    # a composed with itself
        for t in V.ORDERED_DEPTHS:
            has = col(c, f"typed_walks_h{t}") > 0
            for k in range(1, t + 1):
                assert set(np.abs(col(c, f"opath_h{t}_dir{k}")[has]).tolist()) == {1.0}
                assert not col(c, f"opath_h{t}_dir{k}")[~has].any() and not col(c, f"opath_h{t}_q{k}")[~has].any()
            mean_q = np.mean([col(c, f"opath_h{t}_q{k}") for k in range(1, t + 1)], axis=0)
            np.testing.assert_allclose(mean_q[has], col(c, f"relpath_max_h{t}")[has], atol=1e-5)
            assert not mean_q[~has].any()
    assert col(c1, "typed_walks_h2")[3] == 0.0 and not any(col(c1, f"opath_h2_{x}")[3] for x in ("q1", "q2", "dir1", "dir2", "adj12"))


def test_the_ordered_channel_takes_the_argmax_walk_and_breaks_ties_by_entry_order():
    """Node 3 is reached at depth 2 by 0 -a-> 1 -a-> 3 and by 0 -a-> 2 <-a- 3
    (stored 3 -> 2): the two walks score the same, so the tie rule -- the
    first entry in (owner, neighbour, relation) order -- picks the walk
    through neighbour 1 (dir +1 at step 2). Renumbering the two middle nodes
    puts the inverse walk first and the chosen walk changes with it. With a
    better relation on one side there is no tie and that walk is taken
    whatever its position."""
    rng = np.random.default_rng(13)
    rel_emb = _unit(rng, (2, F.DIM))
    q = _unit(rng, F.DIM)
    ca, cb = (rel_emb @ q).astype(np.float64)
    emb = _unit(rng, (4, F.DIM))
    w = _world(rng, 4, _typed(4, [0, 1, 0, 3], [1, 3, 2, 2], [0, 0, 0, 0]), rel_emb=rel_emb, q=q, emb=emb)
    c = _compile(w, [0])
    assert col(c, "typed_walks_h2")[3] == pytest.approx(np.log1p(2))
    assert col(c, "opath_h2_dir2")[3] == 1.0 and col(c, "opath_h2_dir1")[3] == 1.0
    w_swapped = _world(rng, 4, _typed(4, [0, 2, 0, 3], [2, 3, 1, 1], [0, 0, 0, 0]), rel_emb=rel_emb, q=q, emb=emb)   # inverse walk via node 1
    cs = _compile(w_swapped, [0])
    assert col(cs, "opath_h2_dir2")[3] == -1.0 and col(cs, "opath_h2_dir1")[3] == 1.0
    better, worse = (0, 1) if ca > cb else (1, 0)
    hi, lo = max(ca, cb), min(ca, cb)
    w_b = _world(rng, 4, _typed(4, [0, 1, 0, 3], [1, 3, 2, 2], [worse, worse, worse, better]), rel_emb=rel_emb, q=q, emb=emb)
    cb_ = _compile(w_b, [0])                          # the inverse walk through node 2 scores higher: no tie, it is taken
    assert col(cb_, "opath_h2_dir2")[3] == -1.0 and col(cb_, "opath_h2_dir1")[3] == 1.0
    assert col(cb_, "opath_h2_q1")[3] == pytest.approx(lo, abs=1e-5) and col(cb_, "opath_h2_q2")[3] == pytest.approx(hi, abs=1e-5)
    assert col(cb_, "relpath_max_h2")[3] == pytest.approx((lo + hi) / 2, abs=1e-5)


# ── missing channels set the masks ───────────────────────────────────────────


def test_missing_typed_relations_and_missing_families_set_the_masks():
    rng = np.random.default_rng(3)
    n = 30
    struct = _graph(rng, n, 60, False, True)
    knn = _graph(rng, n, 80, True, False)
    # (a) structural edges without a relation table: the typed basis is unavailable everywhere
    w = _world(rng, n, struct, knn=knn)
    c = _compile(w, [0, 1])
    for t in V.DEPTHS:
        for name in (f"typed_walks_h{t}", f"qsupport_h{t}", f"relpath_max_h{t}", f"relpath_mean_h{t}", f"relpath_min_h{t}"):
            assert not col(c, name).any(), name
    for name in V.ORDERED_COLUMNS:
        assert not col(c, name).any(), name
    assert col(c, "ring_n_h1_FULL").sum() > col(c, "ring_n_h1_STRUCT").sum()       # KNN edges enter FULL only
    # (b) no NER and no KNN: FULL equals STRUCT column for column
    w2 = _world(rng, n, struct, rel_emb=_unit(rng, (4, F.DIM)))
    c2 = _compile(w2, [0, 1])
    for name in V.DEPTH_COLUMNS:
        if name.endswith("_STRUCT"):
            np.testing.assert_array_equal(col(c2, name), col(c2, name[: -len("_STRUCT")] + "_FULL"))
    assert col(c2, "typed_walks_h1").any() and col(c2, "opath_h2_dir1").any()
    # (c) no structural edges at all (a text graph with KNN only): every STRUCT and typed column is 0, FULL is not
    w3 = _world(rng, n, _typed(n, [], [], []), knn=knn)
    c3 = _compile(w3, [0, 1])
    for name in V.V2_COLUMNS:
        if name.endswith("_STRUCT") or name.startswith(("qsupport", "typed_walks", "relpath", "opath")):
            assert not col(c3, name).any(), name
    assert col(c3, "ring_n_h1_FULL").any() and col(c3, "support_h1_FULL").any()
    # (d) an isolated pool: every depth column is 0 and every mask is 0
    w4 = _world(rng, n, _typed(n, [], [], []))
    c4 = _compile(w4, [0, 1])
    assert not np.asarray(c4.scalars[:, V.M3B_N_COLUMNS:]).any()
    for name in V.MASK_COLUMNS:
        assert not col(c4, name).any()


# ── the ring construction, the M3B prefix, the contract, the cache ───────────


@pytest.mark.parametrize("n,m", [(12, 8), (40, 60), (80, 400), (60, 1200)])
def test_dense_bfs_rings_equal_the_sparse_reference_bands(n, m):
    rng = np.random.default_rng(n + m)
    u = rng.integers(0, n, size=m)
    v = rng.integers(0, n, size=m)
    keep = u != v
    A = F._local_graph(n, u[keep], v[keep])
    bands = V.exact_distance_bands(A)
    for (t, D), ref in zip(V.distance_rings(A), bands):
        dense = D > 0
        assert dense.dtype == bool and D.dtype == np.float32
        np.testing.assert_array_equal(dense, np.asarray(ref.todense()) > 0)
        assert not dense.diagonal().any()
        np.testing.assert_array_equal(dense, dense.T)
    # rings are disjoint and together are the nodes within three hops
    rings = [D > 0 for _, D in V.distance_rings(A)]
    assert not (rings[0] & rings[1]).any() and not (rings[1] & rings[2]).any() and not (rings[0] & rings[2]).any()
    reach = np.eye(n, dtype=bool)
    for _ in range(3):
        reach = reach | (np.asarray(A.todense()) @ reach.astype(np.float32) > 0)
    np.testing.assert_array_equal(rings[0] | rings[1] | rings[2] | np.eye(n, dtype=bool), reach)


def test_the_first_111_raw_columns_edges_and_seed_weights_are_the_m3b_compile(chain):
    rng = np.random.default_rng(9)
    n = 50
    w = _world(rng, n, _graph(rng, n, 150, False, True), _graph(rng, n, 60, True, False), _graph(rng, n, 90, True, False), rel_emb=_unit(rng, (4, F.DIM)))
    ids = np.arange(8, dtype=np.int64)
    scores = np.linspace(1.0, 0.5, 8).astype(np.float32)
    inp = F.QueryInputs(q=w.q, dense_ids=ids, dense_scores=scores, splade_ids=ids[::-1].copy(), splade_scores=scores * 10)
    seeds = P.seeds_of(ids, ids[::-1])
    pool = np.arange(n, dtype=np.int64)
    base = F.compile_query(inp, pool, seeds, w.stores, w.nodes, w.rel)
    c = V.compile_query_v2(inp, pool, seeds, w.stores, w.nodes, w.rel)
    assert c.scalars.shape == (n, V.N_COLUMNS) and base.scalars.shape == (n, F.N_COLUMNS)
    np.testing.assert_array_equal(c.scalars[:, : F.N_COLUMNS], base.scalars)
    np.testing.assert_array_equal(c.seeds_local, base.seeds_local)
    np.testing.assert_array_equal(c.seedw, base.seedw)
    np.testing.assert_array_equal(c.pool, base.pool)
    for fam in F.FAMILIES:
        for a, b in zip(c.edges[fam], base.edges[fam]):
            np.testing.assert_array_equal(a, b)
    assert np.isfinite(c.scalars).all()
    # the depth basis reproduces the M3B columns it generalises, exactly where the formulas coincide
    np.testing.assert_allclose(col(c, "seeds_at_h1_FULL"), np.asarray(base.scalars[:, F.IDX["seeds_1hop_FULL"]], np.float64), atol=1e-6)
    np.testing.assert_allclose(col(c, "paths_h2_STRUCT"), np.asarray(base.scalars[:, F.IDX["walks2_STRUCT"]], np.float64), atol=1e-6)
    np.testing.assert_allclose(col(c, "seedproto_h1_FULL"), np.asarray(base.scalars[:, F.IDX["cos_v_seedproto_h1"]], np.float64), atol=1e-5)
    np.testing.assert_allclose(col(c, "relpath_max_h1"), np.asarray(base.scalars[:, F.IDX["relmax_seed"]], np.float64), atol=1e-6)


def test_the_contract_counts_and_the_screen_input():
    assert V.N_DEPTH == 73 and V.N_ORDERED == 13 and V.N_V2 == 86 and V.N_COLUMNS == F.N_COLUMNS + 86 == 197
    assert V.COLUMNS[: F.N_COLUMNS] == list(F.COLUMNS) and V.COLUMNS[F.N_COLUMNS:] == V.V2_COLUMNS == V.DEPTH_COLUMNS + V.ORDERED_COLUMNS
    assert len(V.BLOCKS) == 9 and V.BLOCKS == [f"{f}_h{t}" for f in V.VIEWS for t in V.DEPTHS] + [f"TYPED_h{t}" for t in V.DEPTHS]
    assert len(V.MASK_COLUMNS) == 15                    # has (6) + ring_n (6) + typed_walks (3)
    assert sum(c.startswith("ring_n_") for c in V.MASK_COLUMNS) == 6 and sum(c.startswith("typed_walks_") for c in V.MASK_COLUMNS) == 3
    assert len(set(V.COLUMNS)) == V.N_COLUMNS
    assert V.ORDERED_COLUMNS == ["opath_h2_q1", "opath_h2_q2", "opath_h2_dir1", "opath_h2_dir2", "opath_h2_adj12",
                                 "opath_h3_q1", "opath_h3_q2", "opath_h3_q3", "opath_h3_dir1", "opath_h3_dir2", "opath_h3_dir3",
                                 "opath_h3_adj12", "opath_h3_adj23"]
    for name in V.ORDERED_COLUMNS:
        t = int(name[len("opath_h")])
        assert V.BLOCK_OF[name] == f"TYPED_h{t}" and V.MASKED_BY[name] == f"typed_walks_h{t}" and V.GROUP_OF[name] == "ordered_relation_path"
    core78 = [c for c in F.COLUMNS][:78]                # any 78 M3B names: the count is what is held here
    assert len(V.screen_input_columns(core78)) == 164 and V.screen_input_columns(core78) == core78 + V.V2_COLUMNS
    with pytest.raises(ValueError):
        V.screen_input_columns(["not_a_column"])
    contract = V.contract_json_v2(core78)
    assert contract["name"] == "UNIVERSAL_V2_FEATURE_CONTRACT" and contract["n_columns"] == 197 and contract["depth_basis_columns"] == 73
    assert contract["ordered_relation_path_columns"] == 13 and contract["v2_columns"] == 86 and contract["v2_column_names"] == V.V2_COLUMNS
    assert contract["m3b_raw_columns"] == 111 and contract["screen_input"]["columns"] == 164
    assert contract["screen_input"]["names"] == V.screen_input_columns(core78) and contract["ordered_depths"] == [2, 3]
    assert contract["masked_by"] == V.MASKED_BY and len(V.MASKED_BY) == 30 + 12 + 13
    unmasked = {c for c in V.V2_COLUMNS if c.startswith(("paths_", "branch_", "support_"))}
    assert set(V.MASKED_BY) == set(V.V2_COLUMNS) - set(V.MASK_COLUMNS) - unmasked and len(unmasked) == 16
    for name, mask in V.MASKED_BY.items():
        assert mask in V.MASK_COLUMNS and V.BLOCK_OF[mask] == V.BLOCK_OF[name], name
    assert [c["name"] for c in contract["columns"]] == V.COLUMNS
    assert all(c["formula"] for c in contract["columns"]) and contract["qsupport_temperature"] == 0.1
    assert sum(c["block"] is not None for c in contract["columns"]) == 86


def test_the_v2_cache_round_trips_through_the_m3b_reader(tmp_path):
    rng = np.random.default_rng(21)
    n = 40
    w = _world(rng, n, _graph(rng, n, 120, False, True), knn=_graph(rng, n, 60, True, False), rel_emb=_unit(rng, (4, F.DIM)))
    writer = V.CacheWriterV2(tmp_path / "fit", {"dataset": "toy", "kind": "fit", "feature_contract": V.CONTRACT_NAME, "n_columns": V.N_COLUMNS})
    compiled = []
    for i in range(3):
        w.q = _unit(rng, F.DIM)
        c = _compile(w, [i, i + 1, i + 2])
        compiled.append(c)
        writer.add(f"q{i}", i, w.q, c, np.asarray([0, 5]), 2)
    meta = writer.write()
    assert meta["n_columns"] == V.N_COLUMNS and meta["n_queries"] == 3
    ctx = T.DatasetContext("toy", w.stores, w.nodes, w.rel)
    data = T.CarveData(tmp_path / "fit", ctx)              # the pinned reader over the v2 layout
    assert data.scalars.shape == (3 * n, V.N_COLUMNS)
    for i, c in enumerate(compiled):
        qd = data.query(i)
        np.testing.assert_array_equal(qd["x"].astype(np.float32), c.scalars.astype(np.float16).astype(np.float32))
        np.testing.assert_array_equal(qd["pool"], c.pool)
    idx = np.asarray([V.IDX[name] for name in ("rrf", "support_h2_STRUCT", "qsupport_h3")])
    sub = T.CarveData(tmp_path / "fit", ctx, columns=idx)
    np.testing.assert_array_equal(sub.query(1)["x"], data.query(1)["x"][:, idx])
