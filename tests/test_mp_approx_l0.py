"""MP-Approx level 0 (configs/mp_approx_l0.yaml#tests): the declaration and its pins, the id-only rules, the copied
attention against the cell, the taps and the no-edge forward on the real arms, the analysis set's exactness, and no
held query in any sidecar."""

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
from mp_retrieval.m3b_train import rank_metrics  # noqa: E402
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
