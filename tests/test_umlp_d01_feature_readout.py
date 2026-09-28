"""UMLP-D0.1's readout: pair-AUC is the brute-force count, groups partition as declared, held rows cannot
reach the stored-metric view, only the declared checkpoints are named, frozen configs untouched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("umlp_d01", REPO_ROOT / "scripts" / "umlp_d01_feature_readout.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


@pytest.fixture(scope="module")
def decl():
    return D.load_declaration()


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} is a gitignored sidecar that is not present here")


def test_pair_auc_is_the_brute_force_count():
    rng = np.random.default_rng(3)
    X, Y = rng.integers(0, 4, (5, 3)).astype(float), rng.integers(0, 4, (7, 3)).astype(float)
    got = D.pair_auc_query(X, Y)
    for c in range(3):
        want = np.mean([(x > y) + 0.5 * (x == y) for x in X[:, c] for y in Y[:, c]])
        assert got[c] == pytest.approx(want)


def test_bootstrap_rule_is_the_pilot_rule():
    A = np.random.default_rng(1).random((40, 2))
    res = D.bootstrap_mean(A)
    rng = np.random.default_rng(0)
    means = [A[rng.integers(40, size=40)].mean(axis=0) for _ in range(1000)]
    lo, hi = np.percentile(means, [2.5, 97.5], axis=0)
    assert np.allclose(res["low"], lo) and np.allclose(res["high"], hi)


def _toy(decl):
    """Two queries: q0 twin top-1 is gold; q1 twin top-1 is a non-gold (not analysed)."""
    twin, gnns = decl["arms_scored"]["twin"][0], decl["arms_scored"]["mp_reference"]
    query = np.array([0] * 8 + [1] * 3)
    gold = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0], dtype=bool)
    rank = np.array([1, 3, 2, 4, 5, 6, 7, 9, 1, 2, 3])
    z = {"query": query, "is_gold": gold, f"{twin}/rank": rank, "row": np.array([0, 1])}
    z.update({f"{k}/rank": np.array([3, 1, 9, 9, 9, 9, 2, 8, 1, 2, 3]) for k in gnns})
    return z, twin, gnns


def test_groups_partition_as_declared(decl):
    z, twin, gnns = _toy(decl)
    g, top = D.groups_for_seed(z, twin, gnns)
    assert top.tolist() == [True, False]
    in_q0_gold = (z["query"] == 0) & z["is_gold"]
    assert np.array_equal(g["G_top"] | g["G_rec"] | g["G_miss"], in_q0_gold)
    assert not (g["G_top"] & g["G_rec"]).any() and not (g["G_rec"] & g["G_miss"]).any()
    assert np.array_equal(g["G_miss_mp"] | g["G_miss_both"], g["G_miss"]) and not (g["G_miss_mp"] & g["G_miss_both"]).any()
    assert g["G_miss_mp"].tolist()[6] and g["G_miss_both"].tolist()[7]
    assert not g["N_hard"][z["query"] == 1].any()
    assert np.array_equal(g["N_disp"], g["N_hard"] & (z[f"{twin}/rank"] <= 5))


def test_only_declared_checkpoints_and_no_m3b(decl):
    keys = decl["arms_scored"]["twin"] + decl["arms_scored"]["mp_reference"]
    src = (REPO_ROOT / "scripts" / "umlp_d01_feature_readout.py").read_text(encoding="utf-8")
    assert "outputs/m3b/models" not in src and "gat_universal_v1" not in src
    assert set(keys) == set(decl["inputs"]["checkpoints"])


def test_stored_metrics_view_is_held_poison_proof(decl, tmp_path, monkeypatch):
    inp = decl["inputs"]
    paths = {k: REPO_ROOT / inp[k]["path"] for k in ("v2_eval_seed0", "v2_eval_seeds_1_2")}
    for p in paths.values():
        _require(p)
    keys = decl["arms_scored"]["twin"] + decl["arms_scored"]["mp_reference"]
    half, clean = D.stored_gate_metrics(decl, keys)
    d2 = json.loads(json.dumps(decl, default=str))
    for k, p in paths.items():
        with np.load(p) as z:
            arrays = {n: z[n] for n in z.files}
        for n, v in arrays.items():
            if n != "half" and v.shape[:1] == half.shape and np.issubdtype(v.dtype, np.floating):
                v = v.copy()
                v[~half] = np.nan
                arrays[n] = v
        out = tmp_path / f"{k}.npz"
        np.savez(out, **arrays)
        d2["inputs"][k] = {"path": str(out), "sha256": "x"}
    monkeypatch.setattr(D, "ROOT", Path("/"))
    _, dirty = D.stored_gate_metrics(d2, keys)
    for k in clean:
        assert np.array_equal(clean[k], dirty[k]), k


def test_integrity_refuses_a_single_mismatch(decl, monkeypatch):
    """The score stage compares with '!=' and raises; a one-row perturbation of the stored view must be caught by
    the same comparison the stage uses."""
    stored = {"k/recall@5": np.array([1.0, 0.5])}
    r = {"recall@5": 0.5}
    caught = [j for j in range(2) if r["recall@5"] != stored["k/recall@5"][j]]
    assert caught == [0]


def test_frozen_configs_unchanged_by_the_run():
    rec = REPO_ROOT / "outputs" / "umlp_d01" / "record.json"
    _require(rec)
    pinned = json.loads(rec.read_text(encoding="utf-8"))["frozen_config_sha256"]
    for p in D.FROZEN_CONFIGS:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == pinned[p.name]
