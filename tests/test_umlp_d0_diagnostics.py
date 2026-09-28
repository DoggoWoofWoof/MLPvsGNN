"""UMLP-D0's diagnostics: the held half cannot reach the record, the slices partition V2_GATE,
only declared columns are loaded, and the frozen v2 file is untouched."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("umlp_d0", REPO_ROOT / "scripts" / "umlp_d0_diagnostics.py")
D = importlib.util.module_from_spec(spec)
spec.loader.exec_module(D)


def _require(path: Path):
    if not path.exists():
        pytest.skip(f"{path} is a gitignored sidecar that is not present here")


@pytest.fixture(scope="module")
def decl():
    return D.load_declaration()


def _synthetic(decl, n=400, seed=1):
    """A synthetic V2_GATE population in the loader's output shape."""
    rng = np.random.default_rng(seed)
    cols = {}
    for keys in D.arm_keys(decl).values():
        for k in keys:
            for m in D.METRICS:
                cols[f"{k}/{m}"] = rng.random(n).round(2)
    for s in decl["arms_read"]["twin"]:
        for b in range(9):
            cols[f"{s}/block_gate{b}"] = rng.random(n).astype(np.float32)
    total = rng.integers(1, 4, n).astype(float)
    g = {"half_n": n, "cols": cols, "gold_dist_struct": rng.integers(-1, 5, n), "pool_size": rng.integers(50, 300, n),
         "hop": np.zeros(n, dtype=int), "gold_total": total, "gold_in_pool": np.minimum(total, rng.integers(0, 4, n)).astype(float)}
    masks = {f"first_support_{v}": rng.integers(-1, 4, n) for v in ("STRUCT", "FULL")}
    ctx = {ds: {s: rng.random((50, 9)) for s in decl["arms_read"]["twin"]} for ds in ("metaqa", "squad")}
    return g, masks, ctx


def test_slices_partition_the_population(decl):
    g, masks, _ = _synthetic(decl)
    for name, parts in D.build_slices(g, masks).items():
        cover = sum(m.astype(int) for m in parts.values())
        assert (cover == 1).all(), name


def test_diagnose_is_deterministic(decl):
    g, masks, ctx = _synthetic(decl, n=120)
    a = json.dumps(D.diagnose(decl, g, masks, ctx), sort_keys=True)
    b = json.dumps(D.diagnose(decl, g, masks, ctx), sort_keys=True)
    assert a == b


def test_held_half_nan_poisoning_leaves_the_record_unchanged(decl, tmp_path, monkeypatch):
    """Poison every held row of every column in copies of the eval files; the loader's gate-only view must equal
    the clean one exactly, so nothing downstream can move."""
    inp = decl["inputs"]
    paths = [REPO_ROOT / inp[k]["path"] for k in ("v2_eval_seed0", "v2_eval_seeds_1_2", "m3b_eval")]
    for p in paths:
        _require(p)
    clean = D.load_gate_arrays(decl)
    poisoned = {}
    for key, p in zip(("v2_eval_seed0", "v2_eval_seeds_1_2", "m3b_eval"), paths):
        with np.load(p) as z:
            half = z["half"].astype(bool) if "half" in z.files else None
            arrays = {k: z[k] for k in z.files}
        if half is None:
            with np.load(paths[0]) as z0:
                half = z0["half"].astype(bool)
        for k, v in arrays.items():
            if k != "half" and v.shape[:1] == half.shape and np.issubdtype(v.dtype, np.floating):
                v = v.copy()
                v[~half] = np.nan
                arrays[k] = v
        out = tmp_path / f"{key}.npz"   # v2 and M3B records share the basename 2wiki.npz
        np.savez(out, **arrays)
        poisoned[key] = {"path": str(out), "sha256": "x"}
    d2 = json.loads(json.dumps(decl, default=str))
    for key, v in poisoned.items():
        d2["inputs"][key] = v
    monkeypatch.setattr(D, "ROOT", Path("/"))
    dirty = D.load_gate_arrays(d2)
    assert set(dirty["cols"]) == set(clean["cols"])
    for k in clean["cols"]:
        assert np.array_equal(clean["cols"][k], dirty["cols"][k]), k
    masks = D.load_masks()
    if masks is not None:
        ctx = {ds: {s: np.full((10, 9), 0.5) for s in decl["arms_read"]["twin"]} for ds in ("metaqa", "squad")}
        monkeypatch.setattr(D, "RESAMPLES", 50)
        a = D.diagnose(decl, clean, masks, ctx)
        b = D.diagnose(decl, dirty, masks, ctx)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_only_declared_columns_are_loaded(decl):
    inp = decl["inputs"]
    _require(REPO_ROOT / inp["v2_eval_seed0"]["path"])
    g = D.load_gate_arrays(decl)
    arms = {k for ks in D.arm_keys(decl).values() for k in ks}
    for key in g["cols"]:
        assert key.split("/")[0] in arms, key


def test_context_datasets_open_no_metric_column(decl, monkeypatch):
    ctx = decl["inputs"]["gate_value_context"]
    _require(REPO_ROOT / ctx["metaqa_seed0"]["path"])
    touched = []
    real = np.load

    class Spy:
        def __init__(self, z):
            self.z = z
            self.files = z.files

        def __getitem__(self, k):
            touched.append(k)
            return self.z[k]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.z.close()

    monkeypatch.setattr(D.np, "load", lambda p, *a, **k: Spy(real(p, *a, **k)))
    D.load_context_gates(decl)
    assert touched and all(k == "half" or "/block_gate" in k for k in touched), sorted(set(touched))[:5]


def test_universal_v2_yaml_is_the_frozen_one():
    rec = REPO_ROOT / "outputs" / "umlp_d0" / "record.json"
    _require(rec)
    pinned = json.loads(rec.read_text(encoding="utf-8"))["universal_v2_yaml_sha256_unchanged"]
    assert hashlib.sha256((REPO_ROOT / "configs" / "universal_v2.yaml").read_bytes()).hexdigest() == pinned


def test_record_says_the_held_half_was_not_read():
    rec = REPO_ROOT / "outputs" / "umlp_d0" / "record.json"
    _require(rec)
    r = json.loads(rec.read_text(encoding="utf-8"))
    assert r["held_half_read"] is False and r["whole_population"]["queries"] == 6290 and r["masks"]["mismatches"] == 0
