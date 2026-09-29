"""MP-Approx level 0 (configs/mp_approx_l0.yaml#tests): the declaration and its pins, the id-only rules, the copied
attention against the cell, the taps and the no-edge forward on the real arms, the analysis set's exactness, the probes
and the measures on synthetic sidecars and hand cases, and no held query in any sidecar."""

from __future__ import annotations

import hashlib
import importlib.util
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
from mp_retrieval import universal_v2_models as M  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES, rank_metrics  # noqa: E402
from test_universal_v2_models import _arm, _batch, _randomise_readout, world  # noqa: E402,F401  (the toy substrate)


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── the declaration and its pins ─────────────────────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256():
    decl = L0.load_declaration()
    assert decl["phase"] == "MP_APPROX_L0" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"].startswith("After matching candidate exposure and inference-time graph information")
    L0.verify_pins(decl)   # eval arrays, ids, contract, selection, weights, records, frozen code; raises on any mismatch


def test_the_frozen_code_is_hashed_with_crlf_folded_to_lf(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"x\r\ny\r\n")
    b.write_bytes(b"x\ny\n")
    assert L0.lf_sha256(a) == L0.lf_sha256(b) == hashlib.sha256(b"x\ny\n").hexdigest()


# ── the rules that read ids only ─────────────────────────────────────────────


def test_the_metaqa_subsample_is_800_per_hop_deterministic_and_reads_ids_only():
    rng = np.random.default_rng(0)
    ids = [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(1500)]
    gate = rng.random(len(ids)) < 0.7
    rows = L0.metaqa_subsample(ids, gate)
    assert rows.size == 2400 and np.all(np.diff(rows) > 0) and gate[rows].all()
    assert [sum(L0.hop_from_id(ids[i]) == h for i in rows) for h in (1, 2, 3)] == [800, 800, 800]
    assert np.array_equal(rows, L0.metaqa_subsample(ids, gate))
    perm = rng.permutation(len(ids))                      # the chosen ids do not depend on the order they arrive in
    chosen = {ids[i] for i in rows}
    assert {[ids[i] for i in perm][r] for r in L0.metaqa_subsample([ids[i] for i in perm], gate[perm])} == chosen
    for h in (1, 2, 3):                                   # and they are the smallest salted hashes of their hop
        hop_ids = sorted((hashlib.sha256(("mp_approx_l0|" + ids[i]).encode()).hexdigest(), ids[i])
                         for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == h)
        assert {q for _, q in hop_ids[:800]} == {q for q in chosen if L0.hop_from_id(q) == h}


def test_the_metaqa_subsample_on_the_stored_population_is_2400_v2_gate_queries():
    decl = L0.load_declaration()
    arrays = decl["inputs"]["eval_arrays"]["metaqa"]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half, hop = z["half"].astype(bool), z["hop"]
    rows = L0.metaqa_subsample(ids, half)
    assert rows.size == 2400 and half[rows].all()
    assert np.bincount(hop[rows], minlength=4)[1:].tolist() == [800, 800, 800]


def test_the_fold_and_inner_split_rules_are_d02s_and_the_declared_hash():
    d02 = _load("umlp_d02", "umlp_d02_joint_separability.py")
    ids = json.loads((ROOT / "outputs/universal_v2/eval/2wiki_query_ids.json").read_text(encoding="utf-8"))[:400]
    assert [L0.fold_of(q) for q in ids] == [d02.fold_of(q) for q in ids]
    assert [L0.is_inner(q) for q in ids] == [int(hashlib.sha256((q + "|inner").encode()).hexdigest(), 16) % 10 == 0 for q in ids]
    assert 0.05 < np.mean([L0.is_inner(q) for q in ids]) < 0.2


# ── the copied attention, the taps, the no-edge forward ──────────────────────


def _cell_inputs(seed=0, n=40, e=160, hidden=32, d_r=8):
    g = torch.Generator().manual_seed(seed)
    cell = M.QueryRelationCell(hidden, d_r, heads=4, q_dim=M.PROJECTION_DIM).eval()
    src = torch.randint(0, n, (e,), generator=g)
    dst = torch.randint(0, n - 5, (e,), generator=g)        # the last five nodes have no in-edge
    h = torch.randn(n, hidden, generator=g)
    q = torch.randn(n, M.PROJECTION_DIM, generator=g)
    r = torch.randn(e, d_r, generator=g)
    self_r = torch.randn(d_r, generator=g)
    return cell, (h, q, torch.stack([src, dst]), r, self_r)


def test_the_copied_attention_reproduces_the_cells_message_and_splits_it_into_self_and_neighbour_parts():
    cell, args = _cell_inputs()
    seen = []
    handle = cell.dropout.register_forward_pre_hook(lambda _m, a: seen.append(a[0].detach().clone()))
    with torch.no_grad():
        cell(*args)
        m, m_nbr, m_self = L0.split_message(cell, *args)
    handle.remove()
    torch.testing.assert_close(m, seen[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(m_nbr + m_self, m, atol=1e-6, rtol=1e-6)
    assert float(m_nbr[-5:].abs().max()) == 0.0                     # no in-edge, no neighbour message
    torch.testing.assert_close(m_self[-5:], m[-5:], atol=0, rtol=0)  # there the self-loop is the whole message


def test_the_taps_record_the_twin_channels_body_and_the_gnn_messages_on_the_real_arms(world):
    twin, gnn = _arm(world, "u_mlp_v2_mix"), _arm(world, "u_gnn_v2_ef")
    _randomise_readout(twin)
    _randomise_readout(gnn)
    n_sc = 2 * len(world.core)
    taps = L0.Taps(twin, gnn, n_sc)
    batch = _batch(world, "typed", k=4)
    taps.on = True
    with torch.no_grad():
        s_twin, s_gnn = twin(batch), gnn(batch)
    taps.on = False
    assert taps.channels.shape == (batch.x.shape[0], twin.input.input_width - n_sc)
    torch.testing.assert_close(twin.readout(taps.body, taps.base_z), s_twin, atol=1e-6, rtol=1e-6)
    assert len(taps.copies) == len(taps.actual) == gnn.steps == 3
    for c, a, nb, sf in zip(taps.copies, taps.actual, taps.nbr, taps.selfs):
        torch.testing.assert_close(c, a, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(nb + sf, c, atol=1e-5, rtol=1e-5)
    assert float(torch.stack(taps.nbr).abs().max()) > 0
    taps.reset()
    with torch.no_grad():
        twin(batch)                                                  # off: nothing is recorded
    assert taps.channels is None and taps.copies == []
    taps.remove()


def test_a_no_edge_forward_restores_message_passing_and_leaves_the_state_dict_unchanged(world):
    gnn = _arm(world, "u_gnn_v2_ef")
    _randomise_readout(gnn)
    off = M.UGNNv2(len(world.core), 64, world.inputs["base_local"], evidence_index=world.inputs["evidence_local"], message_passing=False)
    off.load_state_dict(gnn.state_dict())
    off.set_relation_bank(world.bank)
    off.eval()
    batch = _batch(world, "typed", k=4)
    before = L0.state_digest(gnn)
    with torch.no_grad():
        s_on = gnn(batch)
        s_noedge = L0.noedge_forward(gnn, batch)
        s_off = off(batch)
        s_on_again = gnn(batch)
    assert gnn.message_passing is True and L0.state_digest(gnn) == before
    torch.testing.assert_close(s_noedge, s_off, atol=0, rtol=0)
    torch.testing.assert_close(s_on_again, s_on, atol=0, rtol=0)
    assert float((s_on - s_noedge).abs().max()) > 1e-5


# ── the analysis set ─────────────────────────────────────────────────────────


def test_u_q_metrics_equal_full_pool_metrics_for_the_functions_that_built_it_ties_included():
    rng = np.random.default_rng(3)
    for trial in range(300):
        n = int(rng.integers(3, 400))
        scores = [rng.integers(0, 6, size=n).astype(np.float64) if trial % 2 else rng.normal(size=n) for _ in range(9)]
        gold = rng.choice(n, size=int(rng.integers(0, min(n, 6) + 1)), replace=False)
        total = gold.size + int(rng.integers(0, 3))                  # some golds may lie outside the pool
        loc = L0.analysis_set(scores, gold)
        assert np.all(np.diff(loc) > 0) and np.isin(gold, loc).all()
        g_u = np.searchsorted(loc, gold)
        for s in scores:
            full, sub = rank_metrics(s, gold, total), rank_metrics(s[loc], g_u, total)
            assert all(full[m] == sub[m] for m in L0.UQ_EXACT)


def test_the_within_query_z_scores_zero_a_constant_and_ranks_follow_the_tie_rule():
    assert np.array_equal(L0.pool_z(np.full(5, 2.0)), np.zeros(5))
    z = L0.pool_z(np.array([1.0, 2.0, 3.0]))
    assert abs(z.mean()) < 1e-12 and abs(z.std() - 1) < 1e-12
    x = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    zx = L0.column_z(x)
    assert np.array_equal(zx[:, 1], np.zeros(3)) and abs(zx[:, 0].std() - 1) < 1e-12
    assert L0.pool_rank(np.array([0.5, 0.9, 0.5, 0.1])).tolist() == [2, 1, 3, 4]


# ── the probes and the read, on a synthetic sidecar ──────────────────────────


def _synthetic_sidecar(d: Path, n_q: int = 240, seed: int = 0, sizes=(12, 40), gold_signal: float = 1.5) -> Path:
    """A sidecar with the scoring pass's array names, dtypes and shapes, U_q the whole synthetic pool. The GNN's residual
    over the twin is planted as a function of three contract scalars (x0 knows the gold), its edge part as a function of
    another, and the messages as a linear map of eight; a tenth of the queries have no gold in the pool."""
    rng = np.random.default_rng(seed)
    d.mkdir(parents=True, exist_ok=True)
    qids = [f"synthetic:{seed}:{i}" for i in range(n_q)]
    size = rng.integers(sizes[0], sizes[1] + 1, size=n_q)
    ptr = np.concatenate([[0], np.cumsum(size)])
    n = int(ptr[-1])
    query = np.repeat(np.arange(n_q), size).astype(np.int32)
    local = np.concatenate([np.arange(s) for s in size]).astype(np.int32)
    is_gold = np.zeros(n, dtype=bool)
    gold_total = np.zeros(n_q, dtype=np.int64)
    for q in range(n_q):
        g = 0 if q % 10 == 9 else int(rng.integers(1, 4))
        is_gold[ptr[q] + rng.choice(size[q], size=g, replace=False)] = True
        gold_total[q] = max(g, 1) + int(rng.integers(0, 2))
    x = rng.normal(size=(n, 129)).astype(np.float32)
    x[:, 0] += gold_signal * is_gold
    zx = np.concatenate([L0.column_z(x[a:b]) for a, b in zip(ptr[:-1], ptr[1:])]).astype(np.float32)
    x = x.astype(np.float64)
    resid = 1.2 * np.tanh(2 * x[:, 0]) + 0.5 * x[:, 1] * x[:, 2]
    edge = 0.8 * x[:, 3]
    A = rng.normal(size=(8, 128)) / 3
    z = np.zeros((n, 9))
    mnbr = {}
    for k in L0.SEEDS:
        t = rng.normal(size=n) + 0.3 * is_gold
        gs = t + resid + 0.2 * rng.normal(size=n)
        g0 = gs - edge - 0.1 * rng.normal(size=n)
        for col, s in ((k, t), (3 + k, gs), (6 + k, g0)):
            z[:, col] = np.concatenate([L0.pool_z(s[a:b]) for a, b in zip(ptr[:-1], ptr[1:])])
        mnbr[k] = np.stack([x[:, :8] @ A * (t_ + 1) + 0.1 * rng.normal(size=(n, 128)) for t_ in range(L0.STEPS)], 1)
    rank = np.zeros((n, 9), dtype=np.int32)
    qm = np.zeros((n_q, 9, len(METRIC_NAMES)))
    for q in range(n_q):
        a, b = ptr[q], ptr[q + 1]
        gl = np.flatnonzero(is_gold[a:b])
        for f in range(9):
            rank[a:b, f] = L0.pool_rank(z[a:b, f])
            m = rank_metrics(z[a:b, f], gl, int(gold_total[q]))
            qm[q, f] = [m[k] for k in METRIC_NAMES]
    first = np.array([-1 if not is_gold[a:b].any() else int(rng.integers(0, 4)) for a, b in zip(ptr[:-1], ptr[1:])])
    arrays = {"query": query, "local": local, "is_gold": is_gold, "x": x.astype(np.float32), "zx": zx, "z": z, "rank": rank}
    for k in L0.SEEDS:
        arrays[f"channels_{k}"] = rng.normal(size=(n, L0.N_VECTOR)).astype(np.float16)
        arrays[f"body_{k}"] = rng.normal(size=(n, 128)).astype(np.float16)
        arrays[f"base_z_{k}"] = z[:, k].astype(np.float32)
        arrays[f"mnbr_{k}"] = mnbr[k].astype(np.float16)
    arrays.update({"q_row": np.arange(n_q, dtype=np.int64), "q_hop": np.arange(n_q, dtype=np.int64) % 3 + 1,
                   "q_gold_total": gold_total, "q_pool_size": size.astype(np.int64), "q_uq_size": size.astype(np.int64),
                   "q_fold": np.array([L0.fold_of(q) for q in qids], dtype=np.int64),
                   "q_first_support_STRUCT": first.astype(np.int64), "q_metrics": qm})
    for key, v in arrays.items():
        np.save(d / f"{key}.npy", v)
    (d / "qids.json").write_text(json.dumps(qids), encoding="utf-8")
    meta = {"dataset": d.name, "queries": n_q, "uq_rows": n, "chunk_queries": n_q, "chunks": 1, "threads": 1, "mismatches": 0,
            "seconds_this_process": 0.0,
            "arrays_sha256": {f"{key}.npy": L0.sha256_file(d / f"{key}.npy") for key in arrays}}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    return L0.Sidecar(_synthetic_sidecar(tmp_path_factory.mktemp("sidecar") / "metaqa"))


# ── the probes ───────────────────────────────────────────────────────────────


def test_ridge_on_centred_features_recovers_a_planted_linear_residual_and_ignores_a_query_constant_shift(synth):
    rng = np.random.default_rng(1)
    raw = L0.base_raw(synth, "B0", 0, slice(0, synth.n_rows)).astype(np.float64)
    beta = rng.normal(size=raw.shape[1]) * (rng.random(raw.shape[1]) < 0.1)
    y = L0.centre_rows(raw @ beta, synth.ptr)
    shift = np.repeat(rng.normal(scale=50, size=synth.n_q), synth.sizes)    # the same for every candidate of a query
    fold = 0
    Xs, _ = L0.standardised(synth, "B0", 0, synth.fold != fold)
    L0.centre_inplace(Xs, synth)
    b_plain, log_plain = L0.fit_ridge(Xs, synth, fold, lambda sel: y[sel, None], [("r", slice(0, 1))])
    y_shift = L0.centre_rows(y + shift, synth.ptr)
    b_shift, _ = L0.fit_ridge(Xs, synth, fold, lambda sel: y_shift[sel, None], [("r", slice(0, 1))])
    np.testing.assert_allclose(b_shift, b_plain, rtol=1e-9, atol=1e-9)
    test = synth.rows_of(synth.fold == fold)
    pred = Xs[test].astype(np.float64) @ b_plain[:, 0]
    assert 1 - ((pred - y[test]) ** 2).sum() / (y[test] ** 2).sum() > 0.99
    assert log_plain["lambda"]["r"] <= 1e-2                               # a noiseless target takes a small penalty


def test_the_ridge_solution_is_the_closed_form_and_each_output_group_takes_its_own_lambda(synth):
    rng = np.random.default_rng(2)
    G = np.eye(3) * 10.0
    C = rng.normal(size=(3, 4))
    betas = L0.ridge_solve(G, C, 5, (1e-3, 1.0))
    np.testing.assert_allclose(betas[1.0], C / (10.0 + 5.0), rtol=1e-12)
    np.testing.assert_allclose(betas[1e-3], C / (10.0 + 5e-3), rtol=1e-12)
    raw = L0.base_raw(synth, "B0", 0, slice(0, synth.n_rows)).astype(np.float64)
    y = np.stack([L0.centre_rows(raw[:, 0] + raw[:, 5], synth.ptr), L0.centre_rows(rng.normal(size=synth.n_rows), synth.ptr)], 1)
    Xs, _ = L0.standardised(synth, "B0", 0, synth.fold != 2)
    L0.centre_inplace(Xs, synth)
    _, log = L0.fit_ridge(Xs, synth, 2, lambda sel: y[sel], [("clean", slice(0, 1)), ("noise", slice(1, 2))])
    assert log["lambda"]["clean"] <= 1e-2 and log["lambda"]["noise"] >= 10     # a pure-noise output is shrunk hard


def test_the_mlp_probe_is_deterministic_at_a_fixed_seed_and_thread_count(synth):
    torch.set_num_threads(L0.PROBE_THREADS)
    fold = 1
    Xs, _ = L0.standardised(synth, "B0", 0, synth.fold != fold)
    r_c, _ = L0.probe_targets(synth)
    train = synth.fold != fold
    args = (Xs, r_c[:, 0], synth, train & ~synth.inner, train & synth.inner, ~train)
    p1, log1 = L0.fit_mlp(*args, seed=1000 + fold)
    p2, log2 = L0.fit_mlp(*args, seed=1000 + fold)
    assert np.array_equal(p1, p2) and log1 == log2
    p3, _ = L0.fit_mlp(*args, seed=1001 + fold)
    assert not np.array_equal(p1, p3)
    starts = np.concatenate([[0], np.cumsum(synth.sizes[~train])[:-1]])   # every query's prediction is centred
    assert p1.size == synth.sizes[~train].sum() and np.abs(np.add.reduceat(p1, starts)).max() < 1e-4


# ── the measures, on cases whose values are known ────────────────────────────


def test_the_bands_follow_the_declared_thresholds():
    assert L0.band(0.80, [0.55, 0.95], True) == "L0_HIGH"
    assert L0.band(0.80, [0.45, 0.95], True) == "L0_MID"
    assert L0.band(0.75, [0.50, 0.90], True) == "L0_HIGH"
    assert L0.band(0.20, [0.00, 0.45], True) == "L0_LOW"
    assert L0.band(0.25, [0.00, 0.50], True) == "L0_LOW"
    assert L0.band(0.20, [0.00, 0.55], True) == "L0_MID"
    assert L0.band(0.90, [0.80, 0.99], False) == "NOT_READ"
    assert L0.band(float("nan"), [0.0, 0.0], True) == "NOT_READ"


def test_discordant_pair_recovery_and_the_top5_overlap_on_a_hand_case():
    t, g = np.array([3.0, 2.0, 1.0, 0.0]), np.array([1.0, 2.0, 3.0, 0.0])
    gold = np.array([True, False, False, False])
    # T and G disagree on (0,1), (0,2), (1,2); pair (x,3) is concordant. One-gold discordant pairs: (0,1), (0,2).
    a_good = np.array([1.0, 3.0, 2.0, -1.0])        # orders (0,1) and (0,2) as G does, (1,2) as T does
    a_tie = np.array([1.0, 1.0, 1.0, 1.0])          # a tie is not recovered
    counts = L0.pair_counts(t, g, [a_good, a_tie, g, t], gold)
    assert counts.tolist() == [[3, 2, 2, 2], [3, 0, 2, 0], [3, 3, 2, 2], [3, 0, 2, 0]]
    assert L0.top_overlap(np.arange(10.0), np.arange(10.0)[::-1]) == 0.0
    assert L0.top_overlap(np.arange(10.0), np.arange(10.0)) == 1.0
    assert L0.top_overlap(np.array([5.0, 4.0, 3.0, 2.0, 1.0, 0.0]), np.array([5.0, 4.0, 3.0, 2.0, 0.0, 1.0])) == 0.8
    assert L0.top_overlap(np.array([1.0, 2.0, 3.0]), np.array([3.0, 2.0, 1.0])) == 1.0   # k = min(5, rows)


def test_the_twin_recovers_nothing_and_the_gnns_own_residual_recovers_everything(synth):
    """rho, R2, Spearman, DPR, gold DPR and the top-5 overlap at their two ends: shat = z(T_k) (zero residual) and
    shat = z(T_k) + r_k (the GNN's own residual, which ranks U_q as G_k does)."""
    z = np.load(synth.dir / "z.npy")
    r_c, _ = L0.probe_targets(synth)
    preds = {"zero": np.zeros_like(r_c), "own": r_c.copy()}
    meas = L0.per_query_measures(synth, z, np.load(synth.dir / "is_gold.npy"), np.load(synth.dir / "q_gold_total.npy"), preds, r_c, "r")
    qm = np.load(synth.dir / "q_metrics.npy")
    W = L0.boot_weights(synth.n_q)
    out = L0.read_family("r", meas, qm, W)
    assert out["readable_metrics"], "the planted residual must widen the gap"
    zero, own = out["probes"]["zero"], out["probes"]["own"]
    for m in L0.RETRIEVAL:
        if out["denominators"][m]["readable"]:
            assert zero["rho"][m]["point"] == 0.0 and own["rho"][m]["point"] == 1.0
    assert zero["rho_bar"]["point"] == 0.0 and own["rho_bar"]["point"] == 1.0
    assert zero["band"] == "L0_LOW" and own["band"] == "L0_HIGH"
    assert zero["R2"]["point"] == 0.0 and abs(own["R2"]["point"] - 1.0) < 1e-12
    assert zero["DPR"]["point"] == 0.0 and own["DPR"]["point"] == 1.0
    assert zero["gold_DPR"]["point"] == 0.0 and own["gold_DPR"]["point"] == 1.0
    assert own["top5_overlap"]["point"] == 1.0 and zero["top5_overlap"] == out["reference_top5_overlap"]
    assert abs(own["spearman"]["point"] - 1.0) < 1e-12
    for m in L0.RETRIEVAL:                                   # the denominators are the stored metrics' seed-mean gaps
        i = METRIC_NAMES.index(m)
        assert abs(out["denominators"][m]["gap"] - (qm[:, 3:6, i] - qm[:, 0:3, i]).mean()) < 1e-12


def test_the_message_measures_on_a_hand_case():
    target = np.zeros((3, L0.STEPS, 4))
    target[0, :, 0], target[1, :, 1] = 2.0, 1.0      # row 2 has a zero message: out of the cosine
    pred = target.copy()
    pred[1] *= -1.0
    s = L0.message_stats(pred, target, np.array([0, 3]))
    assert s.shape == (1, L0.STEPS, 4)
    assert s[0, 0].tolist() == [4.0, 5.0, 0.0, 2.0]   # error 0 + 4, energy 4 + 1, cosines 1 - 1, two live rows


def test_ratio_and_the_seed_mean_on_a_hand_case():
    W = np.ones((1, 3))
    point, boot = L0.ratio(np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0]), W)
    assert point == 1.0 and boot.tolist() == [1.0]
    num = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 3.0]])
    den = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    point, _ = L0.seed_mean_ratio(num, den, W)
    assert point == 1.0                                 # seed 0: 3/3, seed 1: 3/3


# ── no held query in any sidecar ─────────────────────────────────────────────


@pytest.mark.parametrize("name", L0.DATASETS)
def test_no_held_query_id_appears_in_any_sidecar(name):
    side = L0.OUT / name / "qids.json"
    if not side.exists():
        pytest.skip(f"{name}: not scored yet")
    decl = L0.load_declaration()
    arrays = decl["inputs"]["eval_arrays"][name]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    scored = json.loads(side.read_text(encoding="utf-8"))
    assert scored and set(scored) <= gate
    rows = np.load(L0.OUT / name / "q_row.npy")
    assert half[rows].all() and [ids[r] for r in rows] == scored
