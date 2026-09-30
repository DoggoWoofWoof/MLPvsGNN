"""scripts/deploy_ck_full_2wiki.py against configs/deploy_ck_full_2wiki.yaml: the declaration and its pins, the rebinding
of deploy_ck_2wiki's four globals, the checked compile, the latency sample, the cluster bootstrap with the speed bands and
the flags, the retention bands, module_shas under torch.ops, the module clock on the three arms, and (DEPLOY_CK_FULL_SMOKE=1,
once the six-base fits have ended) the smoke of every stage on the real 2wiki data, laptop CPU, read-only."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import deploy_ck_2wiki as DK  # noqa: E402
import deploy_ck_full_2wiki as M  # noqa: E402
import universal_v2_run as V2  # noqa: E402


def quiet(*_a, **_k):
    return None


@pytest.fixture(scope="module")
def decl():
    return M.load_declaration()


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    """Where this file's hard stops write; nothing is written to the repository."""
    out = tmp_path / "out"
    monkeypatch.setattr(M, "OUT", out)
    return out


# ── the declaration and its pins ─────────────────────────────────────────────


def test_the_declaration_parses_and_is_this_scripts(decl):
    assert decl["phase"] == M.PHASE and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"].startswith("After matching candidate exposure and inference-time graph information")
    assert tuple(decl["arms"]) == M.ARMS
    for arm, spec in decl["arms"].items():
        assert int(spec["parameters"]) == DK.ARM_SPECS[arm]["parameters"]
        assert list(spec["fits"]) == [DK.fit_key(arm, s) for s in M.SEEDS]
    r = decl["readings"]
    assert tuple(r["retention"]["bands"]) == M.READINGS and tuple(r["speed"]["bands"]) == M.SPEED_BANDS and tuple(r["flags"]) == M.FLAGS
    assert r["primary_half"] == M.HALVES[0] == DK.HALVES[0] and M.READ_METRICS == DK.READ_METRICS and M.SEEDS == DK.SEEDS
    assert tuple(decl["scoring"]["rebinding"]) == M.REBOUND and all(hasattr(DK, k) for k in M.REBOUND)
    same = DK.load_declaration()["population"]   # "same held population": deploy_ck_2wiki's, block for block
    assert {k: decl["population"][k] for k in ("split", "queries", "ids_sha256", "halves")} == {k: same[k] for k in ("split", "queries", "ids_sha256", "halves")}


def test_a_declaration_that_is_not_this_scripts_is_refused(tmp_path):
    text = M.CONFIG.read_text(encoding="utf-8")
    cases = {"parameter pin": text.replace("parameters: 390475", "parameters: 390476", 1),
             "sample sizes": text.replace("timed: 1024 queries", "timed: 1000 queries", 1)}
    d = yaml.safe_load(text)
    d["readings"]["flags"] = dict(reversed(list(d["readings"]["flags"].items())))
    cases["bands or the flags"] = yaml.safe_dump(d, sort_keys=False)
    d = yaml.safe_load(text)
    d["statistics"]["contrasts"]["full_gain"] = "twin - ck_full"
    cases["contrasts"] = yaml.safe_dump(d, sort_keys=False)
    for message, bad in cases.items():
        p = tmp_path / "bad.yaml"
        p.write_text(bad, encoding="utf-8")
        with pytest.raises(SystemExit, match=message):
            M.load_declaration(p)


def test_the_frozen_code_the_pinned_files_and_the_fits_equal_their_pins(decl):
    needed = [ROOT / p for p in decl["inputs"]["files_sha256"]] + [ROOT / M.DK_FITS / f"{k}{s}" for spec in decl["arms"].values()
                                                                   for k in spec["fits"] for s in (".pt", ".json")]
    missing = [p for p in needed if not p.exists()]
    if missing:
        pytest.skip(f"{len(missing)} pinned files are not on this machine")
    assert M.pin_problems(decl) == {}


def test_every_pin_is_checked(decl, tmp_path):
    bad = M.pin_problems(decl, root=tmp_path)   # an empty root: every pin is missing
    fits = {f"{k}{s}" for spec in decl["arms"].values() for k in spec["fits"] for s in (".pt", ".json")}
    assert set(bad) == set(decl["inputs"]["frozen_code_lf"]) | set(decl["inputs"]["files_sha256"]) | fits
    assert len(fits) == 18 and all(v["found"] is None for v in bad.values())


# ── the rebinding ────────────────────────────────────────────────────────────


def test_the_rebinding_sets_exactly_the_four_globals_and_is_undone(tmp_path, out_dir):
    before = dict(vars(DK))

    def fn(prep, context, i):
        return None

    with M.rebound(fn, tmp_path / "eval", tmp_path / "elsewhere"):
        inside = dict(vars(DK))
        assert {k for k in set(before) | set(inside) if before.get(k) is not inside.get(k)} == set(M.REBOUND)
        assert DK.compile_eval_query is fn and DK.EVAL == tmp_path / "eval" and DK.OUT == tmp_path / "elsewhere" and DK.LATENCY_QUERIES == 0
    assert set(vars(DK)) == set(before) and all(vars(DK)[k] is before[k] for k in before)
    with M.rebound(fn, tmp_path / "eval"):   # OUT defaults to this file's (looked up at the call)
        assert DK.OUT == out_dir
    with pytest.raises(RuntimeError):
        with M.rebound(fn, tmp_path / "eval"):
            raise RuntimeError
    assert all(vars(DK)[k] is before[k] for k in M.REBOUND)


def test_a_hard_stop_of_the_imported_code_writes_here_not_into_the_sealed_outputs(tmp_path, out_dir):
    sealed = DK.OUT / "hard_stop.json"
    stat = sealed.stat().st_mtime_ns if sealed.exists() else None
    with M.rebound(None, tmp_path / "eval"):
        with pytest.raises(SystemExit, match="HARD STOP: probe"):
            DK.hard_stop("probe", n=1)
    assert json.loads((out_dir / "hard_stop.json").read_text(encoding="utf-8"))["message"] == "probe"
    assert (sealed.stat().st_mtime_ns if sealed.exists() else None) == stat


# ── the checked compile ──────────────────────────────────────────────────────


def compiled(n: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    return SimpleNamespace(pool=np.arange(10, 10 + n, dtype=np.int64), seeds_local=np.array([0, 2], dtype=np.int64),
                           scalars=rng.random((n, 4), dtype=np.float32), seedw=rng.random((n, 3), dtype=np.float32),
                           edges={"knn": (np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32), rng.random((2, 5), dtype=np.float32)),
                                  "structural": (np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros((0, 5), np.float32))})


def flip(a: np.ndarray, at: int = 0) -> np.ndarray:
    """One bit of one value changed: the lowest bit of the value's bytes."""
    b = a.copy()
    width = {4: np.uint32, 8: np.uint64}[b.dtype.itemsize]
    b.reshape(-1).view(width)[at] ^= 1
    return b


def with_(c, **kw):
    return SimpleNamespace(**{**vars(c), **kw})


def test_the_compile_check_finds_a_one_bit_change_in_every_array():
    ref = compiled()
    assert M.compiled_problems(copy.deepcopy(ref), ref) == {}
    src, dst, attr = ref.edges["knn"]
    cases = {"scalars": with_(ref, scalars=flip(ref.scalars, 5)), "seedw": with_(ref, seedw=flip(ref.seedw)),
             "pool": with_(ref, pool=flip(ref.pool, 3)), "seeds_local": with_(ref, seeds_local=np.array([0, 3], dtype=np.int64)),
             "edges/knn/2": with_(ref, edges={**ref.edges, "knn": (src, dst, flip(attr, 7))}),
             "edges/knn/0": with_(ref, edges={**ref.edges, "knn": (flip(src), dst, attr)}),
             "edge_families": with_(ref, edges={"knn": ref.edges["knn"]})}
    for key, fast in cases.items():
        assert list(M.compiled_problems(fast, ref)) == [key], key
    assert M.compiled_problems(with_(ref, scalars=flip(ref.scalars, 5)), ref)["scalars"]["columns"] == [M.COLUMNS[1]]
    assert list(M.compiled_problems(with_(ref, pool=ref.pool.astype(np.int32)), ref)) == ["pool"]
    assert list(M.compiled_problems(with_(ref, seedw=ref.seedw.astype(np.float64)), ref)) == ["seedw"]


class FakeCompiler:
    def __init__(self, result):
        self.result, self.calls = result, 0

    def compile(self, inp, pool, seeds, embeddings=None, v2=True):
        assert v2 is True and embeddings is not None
        self.calls += 1
        return self.result


def test_the_checked_compile_returns_the_fast_compile_and_stops_on_a_difference(out_dir, monkeypatch):
    ref = compiled()
    fake = FakeCompiler(copy.deepcopy(ref))
    asked = []
    monkeypatch.setattr(M.FF, "compiler_for", lambda stores, nodes=None, rel_table=None, cap=None, threads=None: asked.append(threads) or fake)
    monkeypatch.setattr(M, "compile_query_v2", lambda inp, pool, seeds, stores, nodes, rel, embeddings=None: ref)
    ctx = SimpleNamespace(stores={}, rel_table=None, nodes=SimpleNamespace(read=lambda pool: np.ones((pool.size, 2), dtype=np.float32)))
    prep = SimpleNamespace(pools=[ref.pool], seeds=[ref.seeds_local], qemb=np.zeros((1, 4), np.float32), dense_ids=np.zeros((1, 3), np.int64),
                           dense_scores=np.zeros((1, 3), np.float32), splade_ids=np.zeros((1, 3), np.int64), splade_scores=np.zeros((1, 3), np.float32))
    cc = M.CheckedCompile()
    E, got = cc(prep, ctx, 0)
    assert got is fake.result and E.shape == (ref.pool.size, 2) and cc.queries == 1 and fake.calls == 1 and asked == [M.THREADS]
    fake.result = with_(ref, seedw=flip(ref.seedw, 4))
    with pytest.raises(SystemExit, match="HARD STOP: query 0"):
        cc(prep, ctx, 0)
    stop = json.loads((out_dir / "hard_stop.json").read_text(encoding="utf-8"))
    assert list(stop["problems"]) == ["seedw"] and stop["query"] == 0 and cc.queries == 1
    monkeypatch.setattr(M.FF, "compiler_for", lambda *a, **k: None)
    with pytest.raises(SystemExit, match="does not serve"):
        cc(prep, ctx, 0)


# ── the latency sample ───────────────────────────────────────────────────────


def half_labels() -> np.ndarray:
    """deploy_ck_2wiki's half labels in population order where its pinned eval arrays are here, else a stand-in."""
    p = ROOT / "outputs" / "deploy_ck_2wiki" / "eval" / "2wiki.npz"
    if p.exists():
        with np.load(p) as z:
            return z["half"].astype(bool)
    h = np.zeros(12576, dtype=bool)
    h[np.random.default_rng(5).permutation(12576)[:6290]] = True
    return h


def test_the_latency_sample_is_evenly_spaced_distinct_and_deterministic(decl):
    half = half_labels()
    assert int(half.sum()) == decl["population"]["halves"]["V2_GATE"] and int((~half).sum()) == decl["population"]["halves"]["V2_HELD_CONFIRMATION"]
    n, w = M.SIZES[False]["sample"], M.SIZES[False]["warm"]
    assert (n, w) == (1024, 64)
    timed, warm = M.sample_positions(half, n, gate=False), M.sample_positions(half, w, gate=True)
    assert np.unique(timed).size == timed.size == n and np.unique(warm).size == warm.size == w
    assert not half[timed].any() and half[warm].all() and np.all(np.diff(timed) > 0)
    for pos, size in ((timed, int((~half).sum())), (warm, int(half.sum()))):
        ranks = np.searchsorted(np.flatnonzero(half if pos is warm else ~half), pos)
        assert ranks[0] == 0 and ranks[-1] == size - 1
        assert np.all(np.abs(ranks - np.arange(pos.size) * (size - 1) / (pos.size - 1)) <= 0.5 + 1e-9)
    assert np.array_equal(timed, M.sample_positions(half.copy(), n, gate=False))
    with pytest.raises(ValueError):
        M.sample_positions(half, int((~half).sum()) + 1, gate=False)


def test_the_memory_batches_are_the_largest_pools():
    positions = np.array([3, 5, 8, 9, 12])
    sizes = np.zeros(20, dtype=np.int64)
    sizes[positions] = [40, 90, 90, 10, 70]
    assert M.top_memory_positions(positions, sizes, 3).tolist() == [5, 8, 12]


# ── the latency statistics, the speed bands, the flags ───────────────────────


def test_the_cluster_bootstrap_resamples_queries_with_their_rounds():
    rng = np.random.default_rng(3)
    X = rng.lognormal(0.0, 0.4, size=(40, 3))
    draws = M.cluster_draws(40)
    assert draws.shape == (1000, 40) and np.array_equal(draws[0], np.random.default_rng(0).integers(40, size=40))
    assert np.allclose(M.pooled_percentiles(X), np.percentile(X.ravel(), M.QS))
    boot = M.pooled_percentiles(X, draws)
    assert boot.shape == (1000, 3) and np.allclose(boot[17], np.percentile(X[draws[17]].ravel(), M.QS))
    for c in (0.37, 0.95, 1.6):   # a known ratio: every percentile scales with the values
        r = M.ratio_stats(c * X, X, draws)
        assert all(abs(r[q]["point"] - c) < 1e-4 and abs(r[q]["ci"][0] - c) < 1e-4 and abs(r[q]["ci"][1] - c) < 1e-4 for q in ("p50", "p95", "p99"))


@pytest.mark.parametrize("hi,band", [(0.5, "E2E_FASTER"), (0.90, "E2E_FASTER"), (0.9001, "E2E_MARGINAL"), (0.9999, "E2E_MARGINAL"),
                                     (1.0, "E2E_NOT_FASTER"), (1.7, "E2E_NOT_FASTER")])
def test_the_speed_bands_read_the_intervals_upper_end(hi, band):
    assert M.speed_band([0.1, hi]) == band


MODULES = {"twin": ("input", "body", "readout"), "ck_full": ("input", "body", "readout", "kernel"), "gnn": ("input", "readout")}


def fake_runs(q: int, ratio_total: float, ratio_forward: float, tail: float = 1.0, differs: int = 0, idle: bool = True) -> dict:
    """Timed records as stage_latency_run writes them: the GNN's totals per query and round, ck_full's at a fixed ratio of
    them, the twin's at 0.9 of ck_full's, then the GNN's slowest 5% times ``tail``; forwards at their own ratio."""
    rng = np.random.default_rng(11)
    runs = {}
    for r in range(3):
        g_total, g_fwd = rng.lognormal(0.0, 0.3, q) * 0.02, rng.lognormal(0.0, 0.3, q) * 0.01
        c_total = ratio_total * g_total
        g_total = np.where(g_total >= np.quantile(g_total, 0.95), tail * g_total, g_total)
        totals = {"gnn": g_total, "ck_full": c_total, "twin": 0.9 * c_total}
        fwds = {"gnn": g_fwd, "ck_full": ratio_forward * g_fwd, "twin": 0.5 * ratio_forward * g_fwd}
        for a in M.ARMS:
            stages = {s: np.full(q, 1e-4) for s in M.STAGES}
            stages.update(forward=fwds[a], total=totals[a])
            mods = {n: fwds[a] / 10 for n in MODULES[a]}
            runs[(a, r)] = {"arm": a, "round": r, "idle_at_start": idle,
                            "passes": {p: {"stages": {s: v.tolist() for s, v in stages.items()}, "modules": mods if p == "breakdown" else None,
                                           "top5_differs": differs if (p, a, r) == ("e2e_fast", "ck_full", 1) else 0} for p in M.PASSES}}
    return runs


def retention(full_vs_gnn_hi: float, gate: str = "FULL_KEEPS", fc5: str = "FULL_KEEPS") -> tuple[dict, dict]:
    halves = {h: {m: {"contrasts": {"full_vs_gnn": {"point": -0.01, "ci": [-0.02, full_vs_gnn_hi]}}} for m in M.READ_METRICS} for h in M.HALVES}
    bands = {h: {m: "FULL_KEEPS" for m in M.READ_METRICS} for h in M.HALVES}
    bands["V2_GATE"]["recall@5"] = gate
    bands[M.HALVES[0]]["full_coverage@5"] = fc5
    return halves, bands


def memory(ck: float, gnn: float) -> dict:
    return {"ck_full": {"median_increase_bytes": ck}, "gnn": {"median_increase_bytes": gnn}}


def test_a_clean_run_raises_no_flag():
    runs = fake_runs(200, 0.6, 0.55)
    mems = {(a, r): {"idle_at_start": True} for a in M.ARMS for r in range(3)}
    latency = M.read_latency(runs, 3)
    assert abs(latency["ratios"]["e2e_fast"]["ck_full/gnn"]["p50"]["point"] - 0.6) < 1e-4
    assert abs(latency["ratios"]["e2e_fast"]["twin/gnn"]["p50"]["point"] - 0.54) < 1e-4
    assert latency["kernel"] is not None and latency["queries"] == 200 and latency["rounds"] == 3
    forwards = np.concatenate([np.asarray(runs[("ck_full", r)]["passes"]["breakdown"]["stages"]["forward"]) for r in range(3)])
    assert abs(latency["graph_part"]["ck_full"]["p50_ms"]["point"] - 1000 * np.median(0.7 * forwards)) < 1e-2   # forward minus 3 of 4 tenths
    speed, flags, other = M.flags_of(*retention(-0.001), latency, memory(1.0, 2.0), runs, mems)
    assert speed == "E2E_FASTER" and flags == [] and other == {"tails": {"p95": "E2E_FASTER", "p99": "E2E_FASTER"}, "forward_only": "E2E_FASTER"}


def test_every_flag_lands_where_its_rule_says():
    runs = fake_runs(200, 1.3, 0.5, tail=3.0, differs=2)   # slower end to end, the GNN's heavier tail, a faster forward
    mems = {(a, r): {"idle_at_start": (a, r) != ("gnn", 2)} for a in M.ARMS for r in range(3)}
    latency = M.read_latency(runs, 3)
    halves, bands = retention(0.0, gate="FULL_PARTIAL", fc5="FULL_PARTIAL")
    speed, flags, other = M.flags_of(halves, bands, latency, memory(3.0, 2.0), runs, mems)
    assert speed == "E2E_NOT_FASTER" and other["forward_only"] == "E2E_FASTER" and other["tails"]["p99"] == "E2E_FASTER"
    assert flags == list(M.FLAGS)
    assert M.flags_of(*retention(-0.001), latency, memory(2.0, 2.0), runs, mems)[1] == [
        "E2E_SLOWER", "TAIL_DIFFERS", "FORWARD_ONLY_DIFFERS", "E2E_PATH_DIFFERS", "LAPTOP_NOT_IDLE"]


# ── the retention bands ──────────────────────────────────────────────────────


def synthetic_arrays(values: dict) -> dict:
    """arrays as the eval record holds them: every fit's per-query metric, the same values for every seed."""
    return {f"{DK.fit_key(a, s)}/{m}": np.asarray(v, dtype=np.float64) for a, v in values.items() for s in M.SEEDS for m in M.READ_METRICS}


@pytest.mark.parametrize("fraction,band", [(1.0, "FULL_KEEPS"), (0.8, "FULL_KEEPS"), (0.5, "FULL_PARTIAL"), (0.0, "FULL_NO_GAIN"),
                                           (-0.2, "FULL_NO_GAIN")])
def test_the_retention_bands_land_where_the_rules_say(fraction, band):
    n = 300
    rng = np.random.default_rng(1)
    twin, gain = rng.uniform(0.0, 0.5, n), rng.uniform(0.1, 0.3, n)
    h = M.read_half(synthetic_arrays({"twin": twin, "gnn": twin + gain, "ck_full": twin + fraction * gain}), np.ones(n, dtype=bool), "recall@5")
    assert M.band_of(h) == band
    s = h["shares"]["full_share"]
    assert s["readable"] and abs(s["point"] - fraction) < 1e-4 and h["contrasts"]["full_vs_gnn"]["of"] == ["ck_full", "gnn"]


def test_no_gnn_gain_is_its_own_band_and_leaves_the_share_unread():
    n = 200
    twin = np.random.default_rng(2).uniform(0.0, 1.0, n)
    h = M.read_half(synthetic_arrays({"twin": twin, "gnn": twin, "ck_full": twin + 0.1}), np.ones(n, dtype=bool), "recall@5")
    assert M.band_of(h) == "GNN_GAIN_ABSENT"
    s = h["shares"]["full_share"]
    assert not s["readable"] and s["ci"] is None and "not read" in M.fmt_share(s)


def test_a_share_above_the_threshold_with_a_wide_interval_is_not_kept():
    n = 300
    rng = np.random.default_rng(3)
    twin, gain, noise = rng.uniform(0.0, 0.5, n), rng.uniform(0.1, 0.3, n), rng.normal(0.0, 1.5, n)
    noisy = twin + 0.78 * gain + (noise - noise.mean())
    h = M.read_half(synthetic_arrays({"twin": twin, "gnn": twin + gain, "ck_full": noisy}), np.ones(n, dtype=bool), "recall@5")
    s = h["shares"]["full_share"]
    assert s["point"] >= M.KEEP["share"] and s["ci"][0] < M.KEEP["share_low"] and M.band_of(h) != "FULL_KEEPS"


def test_the_read_is_deploy_ck_2wikis_procedure():
    """The same arrays read by this file and by deploy_ck_2wiki's read_half give the same seed means and gnn_gain."""
    n = 150
    rng = np.random.default_rng(4)
    twin, gain = rng.uniform(0.0, 0.5, n), rng.uniform(0.0, 0.3, n)
    values = {"twin": twin, "gnn": twin + gain, "ck_full": twin + 0.9 * gain, "ck_qi": twin, "ck_self": twin}
    arrays, mask = synthetic_arrays(values), np.ones(n, dtype=bool)
    dk_decl = DK.load_declaration()
    ours, theirs = M.read_half(arrays, mask, "recall@5"), DK.read_half(dk_decl, arrays, mask, "recall@5")
    assert ours["contrasts"]["gnn_gain"] == theirs["contrasts"]["gnn_gain"]
    assert ours["shares"]["full_share"]["ci"] == theirs["shares"]["ref_share"]["ci"]
    assert all(ours["arms"][a] == theirs["arms"][a] for a in M.ARMS)


# ── placement, code identity, the module clock ─────────────────────────────


def test_module_shas_survives_torch_ops_relative_file():
    _ = torch.ops.aten   # its module's __file__ is the relative '_ops.py'
    shas = DK.module_shas()
    assert "scripts/deploy_ck_full_2wiki.py" in shas and "scripts/deploy_ck_2wiki.py" in shas
    assert shas["scripts/deploy_ck_full_2wiki.py"] == V2.lf_sha256(ROOT / "scripts" / "deploy_ck_full_2wiki.py")


class Tiny(torch.nn.Module):
    def __init__(self, kernel: bool = False):
        super().__init__()
        self.input, self.body, self.readout = torch.nn.Linear(3, 4), torch.nn.Linear(4, 4), torch.nn.Linear(4, 1)
        self.kernel = torch.nn.Linear(4, 4) if kernel else None

    def forward(self, x):
        h = self.input(x)
        if self.kernel is not None:
            h = h + self.kernel(h)
        return self.readout(self.body(h))


def test_the_module_clock_times_each_module_and_comes_off():
    m = Tiny(kernel=True)
    clock = M.ModuleClock(m)
    assert clock.names == ["input", "body", "readout", "kernel"]
    m(torch.zeros(2, 3))
    assert all(v > 0 for v in clock.spent.values())
    clock.reset()
    assert all(v == 0 for v in clock.spent.values())
    clock.remove()
    m(torch.zeros(2, 3))
    assert all(v == 0 for v in clock.spent.values())
    assert M.ModuleClock(Tiny()).names == ["input", "body", "readout"]


def test_the_breakdown_sees_each_arms_node_local_modules_and_the_kernel(decl):
    cfg, cfg_m3b, _ = V2.load_configs()
    inputs = V2.model_inputs(cfg, cfg_m3b)
    bank, _ = V2.build_relation_bank({})
    for arm, names in MODULES.items():
        model = DK.build_model(arm, inputs, bank)
        clock = M.ModuleClock(model)
        assert clock.names == list(names), arm
        clock.remove()


# ── the smoke on the real 2wiki data ─────────────────────────────────────────


@pytest.mark.skipif(os.environ.get("DEPLOY_CK_FULL_SMOKE") != "1",
                    reason="the smoke reads the real 2wiki data once the six-base fits have ended (DEPLOY_CK_FULL_SMOKE=1)")
def test_smoke_every_stage_on_real_2wiki(decl):
    if M.SMOKE.out.exists():
        pytest.skip(f"{M.SMOKE.out} exists; move it aside for a fresh smoke")
    rc = subprocess.run([sys.executable, str(ROOT / "scripts" / "deploy_ck_full_2wiki.py"), "--stage", "smoke"], cwd=ROOT).returncode
    assert rc == 0
    ev = json.loads((M.SMOKE.eval / f"{M.NAME}.json").read_text(encoding="utf-8"))
    n = len(range(0, decl["population"]["queries"], 64))
    assert ev["file"] == "configs/deploy_ck_full_2wiki.yaml" and ev["queries"] == n and ev["fast_compile"]["queries_checked"] == n
    assert ev["m3b_fixed_rrf_agreement"]["ok"] and all(v["ok"] for v in ev["mrr_audit"].values())
    assert sorted(ev["compiled_form"]) == [DK.fit_key("ck_full", s) for s in M.SEEDS]
    assert ev["rebinding"]["LATENCY_QUERIES"] == 0 and ev["latency"]["compile"] == {}
    sample = json.loads((M.SMOKE.latency / "sample.json").read_text(encoding="utf-8"))
    assert len(sample["timed"]) == 16 and len(sample["warm"]) == 4 and sample["pools_rebuilt_equal"] == 20
    assert sample["fast_compile"]["queries_checked"] == 20
    for a in M.ARMS:
        run = json.loads((M.SMOKE.latency / f"{a}__r0.json").read_text(encoding="utf-8"))
        assert set(run["passes"]) == set(M.PASSES) and all(p["top5_differs"] == 0 for p in run["passes"].values())
        assert run["placement"]["torch_threads"] == M.THREADS and set(run["placement"]["blas_env"].values()) == {str(M.THREADS)}
        mem = json.loads((M.SMOKE.memory / f"{a}__r0.json").read_text(encoding="utf-8"))
        assert mem["batches"] == 8 and mem["parameter_bytes"] >= 4 * decl["arms"][a]["parameters"]
    read = json.loads(M.SMOKE.read.read_text(encoding="utf-8"))
    assert read["smoke"] and read["reading"] in M.READINGS and read["speed"] in M.SPEED_BANDS and set(read["flags"]) <= set(M.FLAGS)
    assert "E2E_PATH_DIFFERS" not in read["flags"] and read["pareto_statement"].startswith("ck_full keeps")
    assert "SMOKE" in M.SMOKE.doc.read_text(encoding="utf-8")
