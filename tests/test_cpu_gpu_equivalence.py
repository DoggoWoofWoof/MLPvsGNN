"""configs/cpu_gpu_equivalence.yaml: the tolerance checks and the floor logic on toy arrays, the bundle round trip on a
synthetic batch, the frozen_code pins, and the device path (src/mp_retrieval/device_placement.py) against the pinned
originals -- on CPU everywhere, and on CUDA where there is one (device_path.tests). The host GPU runs this file in
mpr-cu128 under the host_gpu_det settings before any candidate arm runs (device_path.gate).

With CGE_DEVICE_PATH_REPORT set, the module writes what its device-path tests saw (determinism warnings, near ties, the
placement) to outputs/cpu_gpu_equivalence/device_path_tests/<host>__<device>.json; pytest's --junitxml carries the
outcomes."""

from __future__ import annotations

import os

# the determinism request needs the cuBLAS workspace fixed before CUDA starts in this process (placement_settings refuses
# "det" without it); nothing on the CPU reads it
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cpu_gpu_equivalence as E  # noqa: E402
from mp_retrieval import device_placement as D  # noqa: E402
from mp_retrieval import m3b_features as F  # noqa: E402
from mp_retrieval import m3b_models as M  # noqa: E402
from mp_retrieval import m3b_train as T  # noqa: E402
from test_m3b_models import world  # noqa: E402,F401  (the synthetic world: a module-scoped fixture)

DECL = yaml.safe_load((ROOT / "configs" / "cpu_gpu_equivalence.yaml").read_text(encoding="utf-8"))
CUDA = torch.cuda.is_available()
REPORT: dict = {}


@pytest.fixture(scope="module", autouse=True)
def _report():
    yield
    if os.environ.get("CGE_DEVICE_PATH_REPORT"):
        out = ROOT / "outputs" / "cpu_gpu_equivalence" / "device_path_tests"
        out.mkdir(parents=True, exist_ok=True)
        dev = "cuda" if CUDA else "cpu"
        body = {"utc": E.utc(), "placement": D.placement_block(dev), "cuda_available": CUDA, "seen": REPORT}
        (out / f"{platform.node()}__{dev}.json").write_text(json.dumps(E.jsonable(body), indent=1, default=str), encoding="utf-8")


@pytest.fixture
def one_thread():
    """Bit-for-bit comparisons of two CPU runs in one process are made at one thread; restored afterwards."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def _flags() -> dict:
    return {"det": torch.are_deterministic_algorithms_enabled(), "warn": torch.is_deterministic_algorithms_warn_only_enabled(),
            "tf32_mm": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "precision": torch.get_float32_matmul_precision(), "benchmark": torch.backends.cudnn.benchmark, "threads": torch.get_num_threads()}


def _restore(f: dict) -> None:
    torch.use_deterministic_algorithms(f["det"], warn_only=f["warn"])
    torch.backends.cuda.matmul.allow_tf32 = f["tf32_mm"]
    torch.backends.cudnn.allow_tf32 = f["tf32_cudnn"]
    torch.set_float32_matmul_precision(f["precision"])
    torch.backends.cudnn.benchmark = f["benchmark"]
    torch.set_num_threads(f["threads"])


@pytest.fixture
def det_cuda():
    """The host_gpu_det settings for one test, the process's own settings restored afterwards."""
    before = _flags()
    D.placement_settings("det", 8)
    yield
    _restore(before)


# ── the pins and the constants ───────────────────────────────────────────────


def test_the_frozen_code_pins_hold():
    assert E.frozen_code_problems(DECL) == {}


def test_the_checkpoints_match_their_pins_where_they_are():
    specs = E.model_specs(DECL)
    assert [s["key"] for s in specs] == list(DECL["object_under_test"]["models"])
    if not all(s["pt"].exists() and s["json"].exists() for s in specs):
        pytest.skip("the checkpoints are not in this workspace")
    assert E.checkpoint_problems(DECL) == {}


def test_the_script_uses_the_declared_numbers():
    text = yaml.safe_dump(DECL, width=10_000)
    tol = DECL["tolerances"]
    assert "atol=1e-5, rtol=1e-5" in tol["forward_inherited"] and (E.ATOL, E.RTOL) == (1e-5, 1e-5)
    assert tol["gradient"]["loss"] == "|L - L_ref| <= 1e-5 * |L_ref| + 1e-6" and (E.LOSS_REL, E.LOSS_ABS) == (1e-5, 1e-6)
    assert tol["gradient"]["global"].startswith("||g - g_ref||_2 <= 1e-4 * ||g_ref||_2") and E.GRAD_REL == 1e-4
    assert "+ 1e-6 * ||g_ref||_2" in tol["gradient"]["per_tensor"] and E.GRAD_ABS == 1e-6
    probe = DECL["probe"]
    assert "first 16 batches" in probe["batches"] and E.N_BATCHES == 16 and "batch 16" in probe["batches"] and E.BATCH_SIZE == 16
    assert "default_rng(0)" in probe["batches"] and E.SEED == 0 and "per_query" in probe["batches"] and E.DATASET_DRAW == "per_query"
    assert ", ".join(f"{n} {c}" for n, c in E.DRAW_COUNTS.items()) in probe["batches"] and sum(E.DRAW_COUNTS.values()) == 256
    assert "exceeds 1.0 GB" in probe["size_cap"] and "value >= 8" in probe["size_cap"] and (E.BUNDLE_CAP_BYTES, E.MIN_BATCHES) == (1.0e9, 8)
    assert dict(sorted(probe["fit_carves_sha256"].items())) == probe["fit_carves_sha256"]
    oc = DECL["object_under_test"]
    assert "7,067 rows" in oc["relation_bank"] and E.BANK_SHA256 in oc["relation_bank"] and E.BANK_ROWS == 7_067
    assert E.CORE_SHA256 in oc["contract"] and "129 columns" in oc["contract"] and E.N_COLUMNS == 129
    assert "420,932 (GNN) and 330,955 (twin)" in oc["contract"] and E.PARAMETERS == {E.GNN_ARM: 420_932, E.TWIN_ARM: 330_955}
    assert "steps 1, 2, 4, 8 and 16" in DECL["tiers"]["T3_trajectory"]["what"] and E.T3_AFTER == (1, 2, 4, 8, 16)
    assert "lr 1e-3, weight decay 1e-4 and clip 1.0" in DECL["tiers"]["T3_trajectory"]["what"] and (E.LR, E.WEIGHT_DECAY, E.CLIP) == (1e-3, 1e-4, 1.0)
    pl = DECL["placements"]
    assert set(pl["reference"]) == {E.REFERENCE} and set(pl["floor"]) == {E.FLOOR} and tuple(pl["candidates"]) == E.CANDIDATES
    env_text = {"host_cpu_t8": pl["candidates"]["host_cpu_t8"], "host_gpu_det": pl["candidates"]["host_gpu_det"],
                "host_gpu_default": pl["candidates"]["host_gpu_det"]}          # "the same GPU and settings" as host_gpu_det
    for arm in E.CANDIDATES:
        assert E.ARMS[arm]["env"] in env_text[arm]
    assert E.GPU_NAME in pl["candidates"]["host_gpu_det"] and f"driver {E.GPU_DRIVER}" in pl["candidates"]["host_gpu_det"]
    assert "8 threads" in pl["reference"][E.REFERENCE] and E.ARMS[E.REFERENCE]["threads"] == 8
    assert "4 threads" in pl["floor"][E.FLOOR] and E.ARMS[E.FLOOR]["threads"] == 4
    repeats = sorted(a for a, s in E.ARMS.items() if s["role"] == "repeat")
    assert repeats == sorted(f"{a}_r2" for a in ("laptop_cpu_t8", "host_gpu_det", "host_gpu_default"))
    assert all(E.ARMS[a]["tiers"] == ("T1",) for a in repeats)
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text and D.CUBLAS_WORKSPACE == ":4096:8"
    assert E.FIELDS == tuple(f.name for f in dataclasses.fields(M.PackedBatch))


# ── the cells, the tolerances and the verdicts ───────────────────────────────


def test_a_forward_cell_is_assert_close_per_query_draw():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=40).astype(np.float32)
    tol = E.ATOL + E.RTOL * np.abs(ref.astype(np.float64))
    ptr = np.array([0, 10, 25, 40])
    near = (ref + 0.5 * tol).astype(np.float32)          # inside the tolerance everywhere
    far = near.copy()
    far[12] = np.float32(ref[12] + 3 * tol[12])           # one element of draw 1 outside it
    for cand, want in ((ref.copy(), [True, True, True]), (near, [True, True, True]), (far, [True, False, True])):
        cells = E.forward_cells(cand, ref, ptr)
        assert cells["ok"].tolist() == want
        for q in range(3):
            s, e = ptr[q], ptr[q + 1]
            try:
                torch.testing.assert_close(torch.from_numpy(cand[s:e]).double(), torch.from_numpy(ref[s:e]).double(), atol=E.ATOL, rtol=E.RTOL)
                closed = True
            except AssertionError:
                closed = False
            assert closed == want[q]
    assert E.forward_cells(ref.copy(), ref, ptr)["bit"].all()
    assert E.forward_cells(near, ref, ptr)["bit"].tolist() == [bool(np.array_equal(near[s:e], ref[s:e])) for s, e in zip(ptr[:-1], ptr[1:])]
    nan = ref.copy()
    nan[30] = np.nan
    assert E.forward_cells(nan, ref, ptr)["ok"].tolist() == [True, True, False]
    # the tolerance scales with the reference's magnitude
    assert E.forward_cells(np.array([1000.009]), np.array([1000.0]), np.array([0, 1]))["ok"][0]
    assert not E.forward_cells(np.array([0.009]), np.array([0.0]), np.array([0, 1]))["ok"][0]


def _grads(loss, tensors):
    names = list(tensors)
    return {"loss": torch.tensor(loss, dtype=torch.float32), "names": names, "grads": {n: torch.tensor(v, dtype=torch.float32) for n, v in tensors.items()}}


def test_the_gradient_cells_on_a_hand_case():
    # two batches; tensor "w" carries the gradient (norm 5 in each batch), "bias" is near-null (the readout bias)
    ref = _grads([2.0, 3.0], {"w": [[3.0, 4.0], [0.0, 5.0]], "bias": [[0.0], [0.0]]})
    same = E.gradient_cells(copy.deepcopy(ref), ref)
    assert same["bit"].all() and same["loss_ok"].all() and same["global_ok"].all() and same["tensor_ok"].all()
    np.testing.assert_allclose(same["g_ref"], [5.0, 5.0])
    np.testing.assert_allclose(same["tensor_rhs"][:, 0], [1e-4 * 5 + 1e-6 * 5, 1e-6 * 5])


def test_the_gradient_cells_read_each_formula():
    ref = _grads([2.0, 3.0], {"w": [[3.0, 4.0], [0.0, 5.0]], "bias": [[0.0], [0.0]]})
    # the loss: |L - L_ref| <= 1e-5 |L_ref| + 1e-6 = 2.1e-5 for batch 0
    c = E.gradient_cells(_grads([2.0 + 2.0e-5, 3.0 + 1e-3], {"w": [[3.0, 4.0], [0.0, 5.0]], "bias": [[0.0], [0.0]]}), ref)
    assert c["loss_ok"].tolist() == [True, False]
    # the global norm: ||g - g_ref|| <= 1e-4 * 5 = 5e-4; the tensor: <= 1e-4 * ||g_ref_p|| + 1e-6 * ||g_ref|| = 5.05e-4 for "w"
    c = E.gradient_cells(_grads([2.0, 3.0], {"w": [[3.0, 4.0 + 4e-4], [0.0, 5.0 + 6e-4]], "bias": [[0.0], [0.0]]}), ref)
    assert c["global_ok"].tolist() == [True, False]
    assert c["tensor_ok"][0].tolist() == [True, False]
    # the null tensor: an error of 4e-6 passes (absolute term 5e-6), 6e-6 fails
    c = E.gradient_cells(_grads([2.0, 3.0], {"w": [[3.0, 4.0], [0.0, 5.0]], "bias": [[4e-6], [6e-6]]}), ref)
    assert c["tensor_ok"][1].tolist() == [True, False] and c["global_ok"].all() and c["loss_ok"].all()
    with pytest.raises(SystemExit, match="different parameters"):
        E.gradient_cells(_grads([2.0, 3.0], {"bias": [[0.0], [0.0]], "w": [[3.0, 4.0], [0.0, 5.0]]}), ref)


def test_the_verdict_is_the_first_reading_that_holds_and_the_floor_is_read_first():
    a, b, c = ("T1", "m", 0), ("T2", "m", 1, "loss"), ("T2", "m", 1, "tensor:w")
    assert E.verdict(set(), set(), True) == "EQUIVALENT_BIT_IDENTICAL"
    assert E.verdict(set(), {a}, True) == "EQUIVALENT_BIT_IDENTICAL"
    assert E.verdict(set(), set(), False) == "EQUIVALENT_WITHIN_TOLERANCE"
    assert E.verdict({a}, {a, b}, False) == "INCONCLUSIVE_FLOOR"
    assert E.verdict({a, b}, {a, b}, False) == "INCONCLUSIVE_FLOOR"
    assert E.verdict({a, c}, {a, b}, False) == "NOT_EQUIVALENT"
    assert E.verdict({c}, set(), False) == "NOT_EQUIVALENT"


def test_a_near_tie_is_two_candidates_within_twice_the_tolerance():
    assert E.has_near_tie(np.array([0.5, 0.5 + 1.5e-5, 2.0]))
    assert not E.has_near_tie(np.array([0.5, 0.5 + 5e-5, 2.0]))
    assert not E.has_near_tie(np.array([1.0]))


def test_the_draw_pointer_concatenates_the_batches():
    draws = [{"qptr": [0, 3, 7]}, {"qptr": [0, 2]}, {"qptr": [0, 1, 4, 9]}]
    assert E.draw_pointer(draws).tolist() == [0, 3, 7, 9, 10, 13, 18]


# ── the bundle ───────────────────────────────────────────────────────────────


def test_the_bundle_round_trip_is_bit_identical(world, tmp_path):
    fits = {"a": world.fit, "b": world.select}
    parts = E.replay_draw(fits, 2)
    for k, batch_parts in enumerate(parts):
        batch = T.pack_parts(fits, batch_parts, F.FAMILIES)
        path = tmp_path / f"batch_{k:02d}.pt"
        E.save_batch(batch, path)
        back = E.read_batch(path)
        for f in E.FIELDS:
            assert torch.equal(getattr(back, f), getattr(batch, f)) and getattr(back, f).dtype == getattr(batch, f).dtype, f
        assert E.field_pins(back) == E.field_pins(batch)
    x = torch.arange(6, dtype=torch.int32)
    assert E.tensor_sha256(x) != E.tensor_sha256(x.view(torch.float32))          # same bytes, another dtype
    assert E.tensor_sha256(x) != E.tensor_sha256(x.reshape(2, 3))                # same bytes, another shape
    assert E.tensor_sha256(x[::2]) == E.tensor_sha256(x[::2].contiguous())


def test_the_replayed_draw_is_the_draw_of_the_fit_loop(world, monkeypatch, one_thread):
    """The bundle's batches are fit_model's first batches for seed 0: the pinned fit loop's own calls to draw_indices,
    recorded, are the replay's draws in order."""
    fits = {"a": world.fit, "b": world.select}
    seen = []
    original = T.draw_indices

    def spy(*args, **kw):
        out = original(*args, **kw)
        seen.append([(n, np.array(i, copy=True)) for n, i in out])
        return out

    monkeypatch.setattr(T, "draw_indices", spy)
    torch.manual_seed(3)
    T.fit_model(M.QLSU(F.N_COLUMNS, 16, F.IDX["rrf"]), fits, {"a": world.select}, seed=E.SEED, arm="qls_u_sota_v1", config={}, max_epochs=1,
                batches_per_epoch=5, batch_size=E.BATCH_SIZE, dataset_draw=E.DATASET_DRAW, log=lambda *_: None)
    monkeypatch.setattr(T, "draw_indices", original)
    replay = E.replay_draw(fits, 5)
    assert len(seen) == len(replay) == 5
    for got, want in zip(seen, replay):
        assert [n for n, _ in got] == [n for n, _ in want]
        assert all(np.array_equal(g, w) for (_, g), (_, w) in zip(got, want))
    assert sum(len(i) for _, i in replay[0]) == E.BATCH_SIZE
    assert E.draw_counts(replay[:1]) == {n: sum(len(i) for m, i in replay[0] if m == n) for n in sorted({m for m, _ in replay[0]})}


# ── the device path on the CPU ───────────────────────────────────────────────


def _gat() -> torch.nn.Module:
    return M.UniversalGAT(F.N_COLUMNS, 16, F.IDX["rrf"], layers=2)


def _record_without_clock(rec) -> dict:
    d = dataclasses.asdict(rec)
    d.pop("seconds")
    d["history"] = [{k: v for k, v in e.items() if k != "seconds"} for e in d["history"]]
    return d


def _equal_metrics(a: dict, b: dict) -> bool:
    return set(a) == set(b) and all(np.array_equal(a[k], b[k], equal_nan=True) for k in a)


@pytest.mark.parametrize("device", [None, "cpu"])
def test_the_copies_on_the_cpu_are_the_pinned_fit_and_evaluation(world, device, one_thread):
    fits = {"a": world.fit, "b": world.select}
    kw = dict(seed=4, arm="gat_toy", config={"H": 16}, max_epochs=2, batches_per_epoch=6, batch_size=6, patience=5, log=lambda *_: None)
    torch.manual_seed(11)
    pinned, rec_p = T.fit_model(_gat(), fits, {"a": world.select, "b": world.fit}, **kw)
    torch.manual_seed(11)
    copied, rec_c = D.fit_model_placed(_gat(), fits, {"a": world.select, "b": world.fit}, device=device, **kw)
    sp, sc = pinned.state_dict(), copied.state_dict()
    assert list(sp) == list(sc) and all(torch.equal(sp[k], sc[k]) for k in sp)
    assert _record_without_clock(rec_c) == _record_without_clock(rec_p)
    assert rec_p.steps == 12 and rec_p.epochs_run == 2
    for data in (world.select, world.fit):
        assert _equal_metrics(D.evaluate_carve_placed(pinned, data, device=device), T.evaluate_carve(pinned, data))


def test_a_checkpoint_of_the_copy_is_a_checkpoint_of_the_pinned_fit(world, tmp_path, one_thread):
    """On the CPU the copy writes the pinned checkpoint: a fit begun by the copy and resumed by the pinned fit_model is
    the pinned uninterrupted fit, and the copy refuses a checkpoint written on another device type."""
    fits = {"a": world.fit, "b": world.select}
    kw = dict(seed=4, arm="gat_toy", config={"H": 16}, batches_per_epoch=5, batch_size=6, patience=9, log=lambda *_: None)
    torch.manual_seed(11)
    whole, rec_w = T.fit_model(_gat(), fits, {"a": world.select}, max_epochs=3, **kw)
    ck = tmp_path / "fit.ckpt"
    torch.manual_seed(11)
    D.fit_model_placed(_gat(), fits, {"a": world.select}, max_epochs=1, checkpoint=ck, **kw)
    assert "cuda_rng_state" not in torch.load(ck, weights_only=False)
    torch.manual_seed(999)
    resumed, rec_r = T.fit_model(_gat(), fits, {"a": world.select}, max_epochs=3, checkpoint=ck, **kw)
    assert rec_r.steps == rec_w.steps and [e["train_loss"] for e in rec_r.history] == [e["train_loss"] for e in rec_w.history]
    assert all(torch.equal(a, b) for a, b in zip(resumed.state_dict().values(), whole.state_dict().values()))
    state = torch.load(ck, weights_only=False)
    state["cuda_rng_state"] = torch.zeros(16, dtype=torch.uint8)
    torch.save(state, ck)
    with pytest.raises(ValueError, match="another device type"):
        D.fit_model_placed(_gat(), fits, {"a": world.select}, max_epochs=4, checkpoint=ck, **kw)


def test_the_settings_are_applied_read_back_and_refused_when_incomplete(monkeypatch):
    before = _flags()
    try:
        assert D.placement_settings(None, 3)["threads"] == 3
        with pytest.raises(ValueError):
            D.placement_settings("fast", 2)
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        with pytest.raises(RuntimeError, match="must be unset"):
            D.placement_settings("default", 2)
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG")
        with pytest.raises(RuntimeError, match="before CUDA starts"):
            D.placement_settings("det", 2)
        got = D.placement_settings("default", 2)
        assert got["cuda_matmul_allow_tf32"] is False and got["cudnn_allow_tf32"] is False and got["float32_matmul_precision"] == "highest"
        assert got["cudnn_benchmark"] is False and got["deterministic_algorithms"] is False and got["cublas_workspace_config"] is None
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        got = D.placement_settings("det", 2)
        assert got["deterministic_algorithms"] is True and got["deterministic_warn_only"] is True and got["cublas_workspace_config"] == ":4096:8"
    finally:
        _restore(before)


def test_a_batch_and_a_model_move_whole():
    b = M.PackedBatch(**{f: torch.zeros(2) for f in E.FIELDS})
    moved = D.batch_to(b, "cpu")
    assert moved is not b and all(getattr(moved, f).device.type == "cpu" for f in E.FIELDS)
    model = _gat()
    assert D.model_to(model, "cpu") is model
    assert D.cpu_copy({"a": [torch.ones(2), (torch.zeros(1), 3)]})["a"][1][1] == 3
    assert not D.on_device(None) and not D.on_device("cpu") and D.on_device("cuda") and D.on_device("cuda:0")


# ── the device path on CUDA (the host GPU) ───────────────────────────────────


def _centred(model, batch) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        s = model(D.batch_to(batch, next(model.parameters()).device)).detach().to("cpu")
    return E.centre(s, batch)


@pytest.mark.skipif(not CUDA, reason="no CUDA device here; the host GPU runs this")
def test_a_cuda_fit_checkpoints_cpu_tensors_and_resumes_to_the_uninterrupted_fit(world, tmp_path, det_cuda):
    fits = {"a": world.fit, "b": world.select}
    kw = dict(seed=4, arm="gat_toy", config={"H": 16}, batches_per_epoch=7, batch_size=6, patience=9, device="cuda", log=lambda *_: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.manual_seed(11)
        whole, rec_w = D.fit_model_placed(_gat(), fits, {"a": world.select}, max_epochs=3, **kw)
        ck = tmp_path / "fit.ckpt"
        torch.manual_seed(11)
        D.fit_model_placed(_gat(), fits, {"a": world.select}, max_epochs=1, checkpoint=ck, **kw)
        state = torch.load(ck, weights_only=False)
        tensors = []

        def walk(o):
            if torch.is_tensor(o):
                tensors.append(o)
            elif isinstance(o, dict):
                [walk(v) for v in o.values()]
            elif isinstance(o, (list, tuple)):
                [walk(v) for v in o]

        walk({k: state[k] for k in ("model", "optimiser", "best_state", "torch_rng_state", "cuda_rng_state")})
        assert tensors and all(t.device.type == "cpu" for t in tensors)
        torch.manual_seed(999)
        resumed, rec_r = D.fit_model_placed(_gat(), fits, {"a": world.select}, max_epochs=3, checkpoint=ck, **kw)
    assert next(whole.parameters()).device.type == "cuda" and next(resumed.parameters()).device.type == "cuda"
    assert rec_r.steps == rec_w.steps
    determinism = sorted({str(w.message) for w in caught if "determinis" in str(w.message).lower()})
    exact = all(torch.equal(a, b) for a, b in zip(resumed.state_dict().values(), whole.state_dict().values()))
    REPORT["cuda_fit_resume"] = {"determinism_warnings": determinism, "weights_torch_equal": exact,
                                 "train_loss": [[e["train_loss"] for e in r.history] for r in (rec_w, rec_r)]}
    if not determinism:
        assert exact and rec_r.best_epoch == rec_w.best_epoch
        assert [e["train_loss"] for e in rec_r.history] == [e["train_loss"] for e in rec_w.history]
    probe = world.select.pack(np.arange(6))
    torch.testing.assert_close(_centred(resumed, probe), _centred(whole, probe), atol=E.ATOL, rtol=E.RTOL)
    with pytest.raises(ValueError, match="another device type"):
        D.fit_model_placed(_gat(), fits, {"a": world.select}, max_epochs=4, checkpoint=ck, **{**kw, "device": None})


@pytest.mark.skipif(not CUDA, reason="no CUDA device here; the host GPU runs this")
def test_the_cuda_evaluation_is_the_cpu_evaluation_away_from_near_ties(world, det_cuda):
    torch.manual_seed(5)
    cpu_model = _gat().eval()
    gpu_model = D.model_to(copy.deepcopy(cpu_model), "cuda")
    seen = {}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for name, data in (("select", world.select), ("fit", world.fit)):
            cpu = T.evaluate_carve(cpu_model, data)
            gpu = D.evaluate_carve_placed(gpu_model, data, device="cuda")
            ties, differing = [], []
            for i in range(data.n_queries):
                batch = data.pack(np.array([i]))
                with torch.no_grad():
                    s = cpu_model(batch).numpy()
                tie = E.has_near_tie(s)
                same = all(np.array_equal(cpu[m][i:i + 1], gpu[m][i:i + 1], equal_nan=True) for m in T.METRIC_NAMES)
                if tie:
                    ties.append(i)
                elif not same:
                    differing.append(i)
            seen[name] = {"queries": int(data.n_queries), "near_ties": len(ties), "differing_without_a_near_tie": differing}
            assert differing == [], (name, differing)
    seen["determinism_warnings"] = sorted({str(w.message) for w in caught if "determinis" in str(w.message).lower()})
    REPORT["cuda_evaluation"] = seen
