"""MP-Approx level 1 (configs/mp_approx_l1.yaml#tests): the declaration, its pins (level 0's included) and the JL digest;
the mean operator against a dense reference, FULL as the concatenation; the projection's linearity; the prototype
recomputation against GatedInputBlock; the bases' widths and order; zero tokens and the band labels; the probe and read
stages end to end on a synthetic sidecar, with the cited-value stop; and no held query in any sidecar."""

from __future__ import annotations

import hashlib
import json
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
from test_mp_approx_l0 import _synthetic_sidecar  # noqa: E402  (level 0's synthetic sidecar)
from test_universal_v2_models import _arm, _batch, world  # noqa: E402,F401  (the toy substrate)


def _quiet(_s: str) -> None:
    pass


# ── the declaration and its pins ─────────────────────────────────────────────


def test_the_jl_matrix_is_the_declared_draw_and_has_the_declared_digest():
    decl = L1.load_declaration()
    R = L1.jl_matrix()
    assert R.shape == (1536, 64) and R.dtype == np.float64 and R.flags["C_CONTIGUOUS"]
    assert hashlib.sha256(R.tobytes()).hexdigest() == decl["inputs"]["jl_matrix_sha256"]
    assert abs(float(R.std()) * 8.0 - 1.0) < 0.02                       # entries N(0, 1/64)


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256_level_0s_included():
    decl = L1.load_declaration()
    assert decl["phase"] == "MP_APPROX_L1" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"].startswith("After matching candidate exposure and inference-time graph information")
    if not all((ROOT / p["dir"] / "meta.json").exists() for p in decl["inputs"]["level0"]["sidecars"].values()):
        pytest.skip("level 0's sidecars are not on this machine")
    L1.verify_inputs(decl)   # level 0's declaration, script, tests, sidecars, its own inputs, the JL digest; raises on any mismatch


# ── the operator ─────────────────────────────────────────────────────────────


def _dense_mean(u, v, n):
    A = np.zeros((n, n))
    for a, b in zip(np.asarray(u), np.asarray(v)):
        A[b, a] += 1.0
    deg = A.sum(1, keepdims=True)
    return np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)


def _edges(rng, n, dead=4, empty=()):
    out = {}
    for f in L1.FAMILIES:
        if f in empty:
            out[f] = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64))
            continue
        m = int(rng.integers(20, 60))
        u, v = rng.integers(0, n, m), rng.integers(0, n - dead, m)      # the last `dead` nodes have no in-edge
        out[f] = (np.r_[u, u[:5]], np.r_[v, v[:5]])                      # five edges listed twice
    return out


def test_the_mean_operator_equals_a_dense_reference_and_full_is_the_concatenation():
    rng = np.random.default_rng(0)
    n = 30
    edges = _edges(rng, n)
    X = rng.normal(size=(n, 7))
    ops = L1.operators(edges, n)
    Xt = torch.from_numpy(X)
    for f in L1.FAMILIES:
        np.testing.assert_allclose((ops[f] @ Xt).numpy(), _dense_mean(*edges[f], n) @ X, rtol=1e-12, atol=1e-12)
    fu, fv = L1.full_edges(edges)
    assert np.array_equal(fu, np.concatenate([edges[f][0] for f in L1.FAMILIES]))
    D_full = _dense_mean(fu, fv, n)
    np.testing.assert_allclose((ops["FULL"] @ Xt).numpy(), D_full @ X, rtol=1e-12, atol=1e-12)
    assert not np.allclose(D_full, np.mean([_dense_mean(*edges[f], n) for f in L1.FAMILIES], 0))   # not the mean of the three means
    P = _dense_mean([0, 0, 1], [2, 2, 2], 3)                            # an edge listed twice counts twice
    assert P[2].tolist() == [2 / 3, 1 / 3, 0.0] and P[:2].sum() == 0.0
    V = L1.hop_vectors(X, edges, np.arange(n))                          # the depths are powers, X at 0
    assert V.shape == (n, L1.N_TOK, 7) and np.array_equal(V[:, 0], X)
    for f_i, f in enumerate(L1.TOKEN_FAMILIES):
        D = D_full if f == "FULL" else _dense_mean(*edges[f], n)
        for k in L1.DEPTHS:
            np.testing.assert_allclose(V[:, 1 + 3 * f_i + (k - 1)], np.linalg.matrix_power(D, k) @ X, rtol=1e-10, atol=1e-12)
    assert np.all(V[n - 4:, 1:] == 0.0)                                 # no in-edge: a zero token at every depth


def test_a_family_without_edges_has_no_operator_and_zero_tokens():
    rng = np.random.default_rng(5)
    n = 20
    edges = _edges(rng, n, empty=("ner",))
    assert L1.mean_operator(*edges["ner"], n) is None
    V = L1.hop_vectors(rng.normal(size=(n, 5)), edges, np.arange(n))
    assert np.all(V[:, 4:7] == 0.0) and np.abs(V[:, 1:4]).max() > 0


def test_projecting_the_propagated_vectors_equals_propagating_the_projected_ones():
    rng = np.random.default_rng(1)
    n = 40
    edges = _edges(rng, n)
    X = rng.normal(size=(n, 1536))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    R = L1.jl_matrix()
    loc = np.sort(rng.choice(n, 15, replace=False))
    tok, cos_q, cos_self = L1.hop_tokens(X, rng.normal(size=1536), edges, loc, R)
    assert tok.shape == (15, L1.N_TOK, 64) and cos_q.shape == (15, L1.N_TOK) and cos_self.shape == (15, L1.N_SELF)
    np.testing.assert_allclose(tok, L1.hop_vectors(X @ R, edges, loc), rtol=1e-9, atol=1e-12)


# ── the prototypes of integrity.edges ────────────────────────────────────────


def test_the_prototype_recomputation_equals_the_gated_input_blocks_prototype_columns(world):
    twin = _arm(world, "u_mlp_v2_mix")
    batch = _batch(world, "typed", k=4)
    seen = []
    handle = twin.input.linear.register_forward_pre_hook(lambda _m, a: seen.append(a[0].detach().clone()))
    with torch.no_grad():
        twin(batch)
        channels = seen[0][:, 2 * len(world.core):]
        assert channels.shape[1] == L0.N_VECTOR
        q_state = L1.query_state(batch.qemb.numpy(), twin.input.semantic.query_projection.weight)[batch.node_query]
        fam = batch.edge_attr[:, : len(L1.FAMILIES)].argmax(1).numpy()
        ei = batch.edge_index.numpy()
        edges = {f: (ei[0, fam == i], ei[1, fam == i]) for i, f in enumerate(L1.FAMILIES)}
        ours = L1.twin_prototypes(batch.emb.numpy(), q_state, edges, twin.input.semantic.node_projection.weight)
    handle.remove()
    assert all(edges[f][0].size for f in L1.FAMILIES)
    torch.testing.assert_close(ours, channels[:, L1.PROTO], atol=1e-6, rtol=1e-5)
    blocks = ours.view(ours.shape[0], len(L1.FAMILIES), L1.PROTO_WIDTH).abs().sum(-1)
    assert float(ours.abs().max()) > 0 and bool((blocks == 0).any())   # some node lacks an in-neighbour in some family: zero
    diff, tol = L1.prototype_diff(ours.numpy(), channels[:, L1.PROTO].numpy().astype(np.float16))
    assert (diff <= tol).all()


def test_the_prototype_tolerance_is_two_float16_ulps_plus_1e_6():
    stored = np.array([0.5, 0.1, 0.0, -0.3], dtype=np.float16)
    ulp = np.spacing(np.abs(stored)).astype(np.float64)
    ok = stored.astype(np.float64) + 1.9 * ulp
    bad = stored.astype(np.float64) + 2.0 * ulp + 2e-6
    diff, tol = L1.prototype_diff(ok, stored)
    assert (diff <= tol).all()
    diff, tol = L1.prototype_diff(bad, stored)
    assert (diff > tol).all()


# ── the bases ────────────────────────────────────────────────────────────────


def _synthetic_tokens(d: Path, sc, seed: int = 0) -> Path:
    """A token sidecar with the token pass's array names, dtypes and shapes over a level 0 sidecar's rows."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    arrays = {"query": sc.query.astype(np.int32), "local": np.asarray(sc.arr("local")),
              "tokens": (rng.normal(size=(sc.n_rows, L1.N_TOK, 64)) * 0.1).astype(np.float16),
              "cos_q": rng.uniform(-1, 1, size=(sc.n_rows, L1.N_TOK)).astype(np.float32),
              "cos_self": rng.uniform(-1, 1, size=(sc.n_rows, L1.N_SELF)).astype(np.float32),
              "q_tilde": rng.normal(size=(sc.n_q, 64)).astype(np.float32),
              "q_edges": rng.integers(0, 50, size=(sc.n_q, 3)), "q_proto_max_diff": np.zeros(sc.n_q), "q_proto_max_ratio": np.zeros(sc.n_q)}
    arrays["tokens"][: sc.n_rows // 10, 1:] = 0.0                     # some rows without in-neighbours
    for key, v in arrays.items():
        np.save(d / f"{key}.npy", v)
    (d / "qids.json").write_text(json.dumps(sc.qids), encoding="utf-8")
    meta = {"dataset": d.name, "queries": sc.n_q, "uq_rows": sc.n_rows, "seconds_this_process": 0.0, "mismatches": 0,
            "edges_per_query_mean": {f: 1.0 for f in L1.FAMILIES}, "zero_token_share": dict.fromkeys(L1.token_names(), 0.1),
            "proto_entries_compared": sc.n_rows * 576, "proto_max_abs_diff": 0.0, "proto_max_ratio": 0.0,
            "arrays_sha256": {f"{key}.npy": L0.sha256_file(d / f"{key}.npy") for key in arrays}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


def test_the_bases_have_the_declared_widths_and_order(tmp_path):
    assert len(L1.base_columns("L1")) == 1947 and len(L1.base_columns("L1d1")) == 907
    assert L1.D1_TOKENS == (0, 1, 4, 7, 10) and L1.D1_SELF == tuple(3 * f for f in range(len(L1.TOKEN_FAMILIES)))
    names = L1.token_names()
    assert names == ["X"] + [f"{f}{k}" for f in ("structural", "ner", "knn", "FULL") for k in (1, 2, 3)]
    assert [names[t] for t in L1.D1_TOKENS] == ["X", "structural1", "ner1", "knn1", "FULL1"]
    assert [names[1 + s] for s in L1.D1_SELF] == ["structural1", "ner1", "knn1", "FULL1"]
    sc = L0.Sidecar(_synthetic_sidecar(tmp_path / "l0" / "metaqa", n_q=30))
    tk = L1.Tokens(_synthetic_tokens(tmp_path / "l1" / "metaqa", sc), sc)
    everything = slice(0, sc.n_rows)
    tokens = np.asarray(tk.tokens, dtype=np.float32)
    q = tk.q_tilde[sc.query]
    for base in L1.BASES:
        X = L1.base_raw_l1(sc, tk, base, everything)
        cols = L1.base_columns(base)
        assert X.shape == (sc.n_rows, len(cols)) and X.dtype == np.float32
        np.testing.assert_array_equal(X[:, :L1.B0_WIDTH], L0.base_raw(sc, "B0", 0, everything))
        for j, c in enumerate(cols[L1.B0_WIDTH:], start=L1.B0_WIDTH):
            kind, name, *rest = c.split(":")
            t = names.index(name)
            want = {"cos_q": lambda: tk.cos_q[:, t], "cos_self": lambda: tk.cos_self[:, t - 1],
                    "token": lambda: tokens[:, t, int(rest[0])], "q_tilde*token": lambda: tokens[:, t, int(rest[0])] * q[:, int(rest[0])]}[kind]()
            np.testing.assert_array_equal(X[:, j], want, err_msg=c)
        rows = np.flatnonzero(np.arange(sc.n_rows) % 3 == 0)            # sorted row indices give the same rows
        np.testing.assert_array_equal(L1.base_raw_l1(sc, tk, base, rows), X[rows])


# ── zero tokens and the bands ────────────────────────────────────────────────


def test_a_zero_token_has_cosine_zero_and_the_cosines_are_exact():
    rng = np.random.default_rng(2)
    n = 6
    edges = {"structural": (np.array([0, 1]), np.array([2, 2])), "ner": (np.empty(0, np.int64), np.empty(0, np.int64)),
             "knn": (np.array([2]), np.array([3]))}
    X = rng.normal(size=(n, 1536))
    q = rng.normal(size=1536)
    tok, cos_q, cos_self = L1.hop_tokens(X, q, edges, np.arange(n), L1.jl_matrix())
    assert np.all(tok[0, 1:] == 0) and np.all(cos_q[0, 1:] == 0) and np.all(cos_self[0] == 0)   # no in-edge anywhere
    ner = [1 + 3 * 1 + k for k in range(3)]
    assert np.all(tok[:, ner] == 0) and np.all(cos_q[:, ner] == 0)                               # no ner edge at all
    z = (X[0] + X[1]) / 2                                                                         # node 2's structural1
    assert abs(cos_q[2, 1] - z @ q / (np.linalg.norm(z) * np.linalg.norm(q))) < 1e-12
    assert abs(cos_self[2, 0] - z @ X[2] / (np.linalg.norm(z) * np.linalg.norm(X[2]))) < 1e-12
    assert abs(cos_q[4, 0] - X[4] @ q / (np.linalg.norm(X[4]) * np.linalg.norm(q))) < 1e-12
    knn2 = 1 + 3 * 2 + 1                                                                          # node 3's knn2 = node 2's knn1 = 0
    assert np.all(tok[3, knn2] == 0) and cos_q[3, knn2] == 0 and np.abs(tok[3, knn2 - 1]).max() > 0


def test_the_bands_carry_the_l1_labels_at_level_0s_thresholds():
    assert L1.band_l1(0.80, [0.55, 0.95], True) == "L1_HIGH"
    assert L1.band_l1(0.80, [0.45, 0.95], True) == "L1_MID"
    assert L1.band_l1(0.75, [0.50, 0.90], True) == "L1_HIGH"
    assert L1.band_l1(0.20, [0.00, 0.45], True) == "L1_LOW"
    assert L1.band_l1(0.25, [0.00, 0.50], True) == "L1_LOW"
    assert L1.band_l1(0.20, [0.00, 0.55], True) == "L1_MID"
    assert L1.band_l1(0.90, [0.80, 0.99], False) == "NOT_READ"
    assert L1.band_l1(float("nan"), [0.0, 0.0], True) == "NOT_READ"


# ── the probe and read stages, end to end ────────────────────────────────────


def test_probe_and_read_run_end_to_end_and_a_cited_value_off_by_more_than_1e_12_stops_the_read(tmp_path, monkeypatch):
    torch.set_num_threads(L0.PROBE_THREADS)
    l0_root, l1_root = tmp_path / "l0", tmp_path / "l1"
    d0 = _synthetic_sidecar(l0_root / "metaqa", n_q=160, sizes=(8, 20))
    monkeypatch.setattr(L0, "HARD_STOP_DIR", [tmp_path / "stops"])
    monkeypatch.setattr(L1, "HARD_STOP_DIR", [tmp_path / "stops"])
    L0.stage_probe("metaqa", log=_quiet, out_dir=d0)
    filed = L0.stage_read(log=_quiet, datasets=("metaqa",), out_root=l0_root)
    _synthetic_tokens(l1_root / "metaqa", L0.Sidecar(d0))
    L1.stage_probe("metaqa", log=_quiet, out_dir=l1_root / "metaqa", l0_dir=d0)
    meta = json.loads((l1_root / "metaqa" / "probes" / "probe_meta.json").read_text(encoding="utf-8"))
    assert sorted(meta["files_sha256"]) == ["fitlog_L1.json", "fitlog_L1d1.json", "unit_L1.npz", "unit_L1d1.npz"]
    assert meta["bases"] == {"L1": 1947, "L1d1": 907}
    rec = L1.stage_read(log=_quiet, datasets=("metaqa",), out_root=l1_root, l0_root=l0_root, filed=filed)
    ds = rec["datasets"]["metaqa"]
    assert set(ds["r"]["probes"]) == set(L1.R_NEW) | set(L1.R_CITED) | {"ref:twin", "ref:other_seed"}
    assert set(ds["e"]["probes"]) == set(L1.E_NEW) | set(L1.E_CITED) | {"ref:no_edge", "ref:other_seed"}
    assert set(ds["messages"]) == set(L1.M_NEW) | set(L1.M_CITED)
    assert set(ds["contrasts"]) == set(L1.CONTRASTS) | set(L1.MESSAGE_CONTRASTS)
    bands = [v["band"] for fam in ("r", "e") for v in ds[fam]["probes"].values()]
    assert all(b in ("L1_HIGH", "L1_MID", "L1_LOW", "NOT_READ") for b in bands)
    assert ds["reading"] == ds["r"]["probes"]["L1-mlp"]["band"]
    filed_ds = filed["datasets"]["metaqa"]
    for fam, cited in (("r", L1.R_CITED), ("e", L1.E_CITED)):
        for p in cited:
            assert ds[fam]["probes"][p]["rho_bar"] == filed_ds[fam]["probes"][p]["rho_bar"]   # recomputed, the same numbers
            assert ds[fam]["probes"][p]["R2"] == filed_ds[fam]["probes"][p]["R2"]
    for p in L1.M_CITED:
        assert ds["messages"][p] == filed_ds["messages"][p]                                    # level 0's read_messages, unchanged
    c = ds["contrasts"]["hop_tokens"]
    if ds["r"]["readable_metrics"]:
        assert abs(c["point"] - (ds["r"]["probes"]["L1-mlp"]["rho_bar"]["point"] - ds["r"]["probes"]["B0-mlp"]["rho_bar"]["point"])) < 1e-12
    m = ds["contrasts"]["message_tokens"]
    assert abs(m["point"] - (ds["messages"]["L1-ridge"]["R2_mean"]["point"] - ds["messages"]["B1-ridge"]["R2_mean"]["point"])) < 1e-12
    L1.stage_doc(log=_quiet, out_root=l1_root, doc=tmp_path / "doc.md")
    text = (tmp_path / "doc.md").read_text(encoding="utf-8")
    assert "## Readings" in text and "L1-mlp" in text and "(cited)" in text
    bad = json.loads(json.dumps(filed))
    point = bad["datasets"]["metaqa"]["r"]["probes"]["B0-mlp"]["rho_bar"]["point"]
    if point is None:
        pytest.skip("the synthetic gap was not readable")
    bad["datasets"]["metaqa"]["r"]["probes"]["B0-mlp"]["rho_bar"]["point"] = point + 1e-9
    with pytest.raises(SystemExit, match="HARD STOP"):
        L1.stage_read(log=_quiet, datasets=("metaqa",), out_root=l1_root, l0_root=l0_root, filed=bad)
    stops = json.loads((tmp_path / "stops" / "hard_stops.json").read_text(encoding="utf-8"))
    assert stops[-1]["probe"] == "B0-mlp" and not (L0.OUT / "hard_stops.json").exists()


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L1.DATASETS)
def test_no_held_query_id_appears_in_any_sidecar(name):
    side = L1.OUT / name / "qids.json"
    if not side.exists():
        pytest.skip(f"{name}: no tokens yet")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"][name]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    scored = json.loads(side.read_text(encoding="utf-8"))
    assert scored and set(scored) <= gate
    assert scored == json.loads((L0.OUT / name / "qids.json").read_text(encoding="utf-8"))
