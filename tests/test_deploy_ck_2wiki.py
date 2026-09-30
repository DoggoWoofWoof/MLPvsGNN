"""scripts/deploy_ck_2wiki.py against configs/deploy_ck_2wiki.yaml: the declaration and its pins, the arms at their
parameter counts, the store digest, the resamples, the bands, the guards, and a smoke of every stage on the real 2wiki
data (fit, repeat, eval on a slice of the population with all 16 fits, read, doc). The smoke opens the package where it
is: the host mirror when the declared root exists (and then runs on host_gpu_det), else the laptop's package, read-only."""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # before CUDA starts (host_gpu_det); inert without a GPU

import copy  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import deploy_ck_2wiki as D  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import universal_v2_run as V2  # noqa: E402
from mp_retrieval.m3b_pools import FamilyStore  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES  # noqa: E402
from mp_retrieval.universal_v2_models import FAMILIES  # noqa: E402

DECLARED_ARMS = ("twin", "ck_qi", "ck_full", "ck_self", "gnn")


def quiet(*_a, **_k):
    return None


@pytest.fixture(scope="module")
def decl():
    return D.load_declaration()


@pytest.fixture
def out_dirs(tmp_path, monkeypatch):
    """Every output of the script under tmp_path; nothing is written to the repository."""
    out = tmp_path / "out"
    monkeypatch.setattr(D, "OUT", out)
    monkeypatch.setattr(D, "FITS", out / "fits")
    monkeypatch.setattr(D, "EVAL", out / "eval")
    monkeypatch.setattr(D, "RECORD", out / "record.json")
    monkeypatch.setattr(D, "DOC", tmp_path / "DEPLOY_CK_2WIKI.md")
    return out


# ── the declaration ──────────────────────────────────────────────────────────


def test_the_declaration_parses_and_is_this_scripts(decl):
    assert decl["phase"] == "DEPLOY_CK_2WIKI" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"].startswith("After matching candidate exposure and inference-time graph information")
    assert tuple(decl["arms"]) == DECLARED_ARMS
    for key, spec in decl["arms"].items():
        assert D.arm_spec(spec) == D.arm_spec(D.ARM_SPECS[key]) and int(spec["parameters"]) == D.ARM_SPECS[key]["parameters"]
    r = decl["readings"]
    assert tuple(r["bands"]) == D.READINGS and tuple(r["flags"]) == D.FLAGS
    assert (r["primary_kernel"], r["per_query_reference"]) == ("ck_qi", "ck_full")
    assert r["primary_half"] == D.HALVES[0] == "V2_HELD_CONFIRMATION"
    assert (r["primary_metric"], r["co_read_metric"]) == D.READ_METRICS[:2]
    assert list(decl["fitting"]["seeds"]) == list(D.SEEDS)
    fits = D.declared_fits(decl)
    assert len(fits) == 16 and fits[-1] == ("ck_qi", 0, True) and len({D.fit_key(*f) for f in fits}) == 16
    assert D.SPEC == L3.SPEC
    assert D.mirror_root(decl).as_posix() == decl["inputs"]["mirror"]["root"]
    assert decl["population"]["halves"] == {"V2_GATE": 6290, "V2_HELD_CONFIRMATION": 6286}
    assert sum(decl["population"]["halves"].values()) == decl["population"]["queries"] == 12576


def test_a_declaration_that_is_not_this_scripts_is_refused(decl, tmp_path):
    text = D.CONFIG.read_text(encoding="utf-8")
    bad_count = tmp_path / "count.yaml"
    bad_count.write_text(text.replace("parameters: 377803", "parameters: 377804", 1), encoding="utf-8")
    with pytest.raises(SystemExit, match="parameter pin"):
        D.load_declaration(bad_count)
    d = yaml.safe_load(text)
    d["readings"]["bands"] = dict(reversed(list(d["readings"]["bands"].items())))
    bad_bands = tmp_path / "bands.yaml"
    bad_bands.write_text(yaml.safe_dump(d, sort_keys=False), encoding="utf-8")
    with pytest.raises(SystemExit, match="bands or the flags"):
        D.load_declaration(bad_bands)


def test_the_training_rule_is_the_frozen_one(decl):
    cfg, cfg_m3b, _ = V2.load_configs()
    rule = D.training_rule(decl, cfg, cfg_m3b)
    assert {k: rule[k] for k in decl["fitting"]["rule"]} == decl["fitting"]["rule"]


def test_the_frozen_code_and_the_pushed_files_equal_their_pins(decl):
    missing = [p for p in D.pushed_paths(decl) if not (ROOT / p).exists()]
    if missing:
        pytest.skip(f"{len(missing)} pushed files are not on this machine")
    assert D.pin_problems(decl) == {}


def test_the_family_stores_equal_their_content_pins(decl):
    if not all((V2.M3B_OUT / "csr" / f"{D.NAME}_{f}.npz").exists() for f in FAMILIES):
        pytest.skip("the 2wiki family stores are not on this machine")
    assert D.store_problems(decl) == {}


# ── digests, guards ──────────────────────────────────────────────────────────


def test_the_store_digest_is_stable_and_reads_every_field():
    def store(**kw):
        base = dict(family="knn", n_nodes=4, indptr=np.array([0, 1, 2, 3, 4], dtype=np.int64), col=np.array([1, 0, 3, 2], dtype=np.int32),
                    weight=np.array([0.5, 0.5, 0.25, 0.25], dtype=np.float32))
        base.update(kw)
        return FamilyStore(**base)

    ref = D.store_digest(store())
    assert ref == D.store_digest(store())
    variants = [store(col=np.array([1, 0, 3, 3], dtype=np.int32)), store(col=np.array([1, 0, 3, 2], dtype=np.int64)),
                store(weight=None), store(n_nodes=5), store(family="ner"), store(indptr=np.array([[0, 1, 2, 3, 4]], dtype=np.int64))]
    assert len({D.store_digest(v) for v in variants} | {ref}) == len(variants) + 1
    assert [f.name for f in dataclasses.fields(FamilyStore)][:4] == ["family", "n_nodes", "indptr", "col"]


def test_the_state_digest_reads_content_not_objects():
    a = {"w": torch.arange(6, dtype=torch.float32).view(2, 3), "b": torch.zeros(3)}
    b = {k: v.clone() for k, v in a.items()}
    assert D.state_digest(a) == D.state_digest(b)
    b["w"][1, 2] += 1e-6
    assert D.state_digest(a) != D.state_digest(b)


def test_code_identity_needs_one_sha_per_module_equal_to_the_committed_file():
    committed = {"scripts/a.py": "1", "src/b.py": "2"}
    ok = {"verify": {"scripts/a.py": "1", "src/b.py": "2"}, "fit/x": {"scripts/a.py": "1"}}
    assert D.code_problems(ok, committed.get) == {}
    split = {"verify": {"scripts/a.py": "1"}, "fit/x": {"scripts/a.py": "3"}}
    assert set(D.code_problems(split, committed.get)) == {"scripts/a.py"}
    stale = {"verify": {"src/b.py": "9"}}
    assert set(D.code_problems(stale, committed.get)) == {"src/b.py"}


def test_the_ceiling_guard_stops_before_the_ceiling(decl, out_dirs):
    D.FITS.mkdir(parents=True)
    assert D.ceiling_guard(decl, "k")["within_ceiling"] and D.ceiling_guard(decl, "k")["projected_fit_hours"] == 1.5
    for i in range(2):   # 50 h filed and a 25 h projection: over the 60 fit-hour ceiling
        (D.FITS / f"f{i}.json").write_text(json.dumps({"seconds": 25 * 3600.0}), encoding="utf-8")
    with pytest.raises(SystemExit, match="compute ceiling"):
        D.ceiling_guard(decl, "k")
    assert json.loads((D.FITS / "ceiling_breach.json").read_text(encoding="utf-8"))["within_ceiling"] is False


def test_the_mirror_record_must_be_verified_at_the_declared_root(decl, out_dirs, tmp_path):
    good = {"status": "VERIFIED", "freeze_matches_declared": True, "loader_imported_from_mirror": True, "datasets": ["2wiki"],
            "mirror": decl["inputs"]["mirror"]["root"] + "/data/final_canonical"}
    d = copy.deepcopy(decl)
    path = tmp_path / "verify_2wiki.json"
    d["inputs"]["mirror"]["verify_record"] = str(path)
    for rec, ok in ((good, True), ({**good, "status": "PARTIAL"}, False), ({**good, "datasets": ["hotpotqa"]}, False),
                    ({**good, "mirror": "C:/elsewhere/data/final_canonical"}, False), ({**good, "freeze_matches_declared": False}, False)):
        path.write_text(json.dumps(rec), encoding="utf-8")
        if ok:
            assert D.mirror_record(d)[0] == rec
        else:
            with pytest.raises(SystemExit, match="HARD STOP"):
                D.mirror_record(d)
    path.unlink()
    with pytest.raises(SystemExit, match="no verify record"):
        D.mirror_record(d)


def test_a_fit_that_is_not_declared_is_refused(decl, out_dirs):
    for arm, seed, rep in (("ck_qw", 0, False), ("ck_qi", 3, False), ("twin", 0, True), ("ck_qi", 1, True)):
        with pytest.raises(SystemExit):
            D.stage_fit(decl, arm, seed, rep, log=quiet)
    assert not D.FITS.exists()


def test_every_arm_builds_at_its_pinned_count_within_the_gnn_budget(decl, out_dirs):
    cfg, cfg_m3b, _ = V2.load_configs()
    inputs = V2.model_inputs(cfg, cfg_m3b)
    bank, _ = V2.build_relation_bank({})   # 2wiki has no relation table: the bank of 2wiki alone has no rows
    assert bank.n_rows == 0
    for key in D.ARM_SPECS:
        model = D.build_model(key, inputs, bank)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == D.ARM_SPECS[key]["parameters"] <= V2.PARAMETER_BUDGET["gnn"]
    assert D.ARM_SPECS["twin"]["parameters"] <= V2.PARAMETER_BUDGET["twin"]


# ── the resamples and the bands ──────────────────────────────────────────────


def test_the_resample_matrix_reproduces_paired_bootstrap():
    rng = np.random.default_rng(7)
    for n in (1, 57, 400):
        a, b = rng.random(n), (rng.random(n) > 0.4).astype(np.float64)
        R = D.resample_matrix(n)
        assert R.shape == (D.BOOT["resamples"], n)
        assert np.array_equal(R[0], np.random.default_rng(0).integers(n, size=n))
        ref = V2.paired_bootstrap(a, b)
        got = D.interval(a[R].mean(axis=1) - b[R].mean(axis=1))
        assert abs(got[0] - ref["low"]) <= 1e-4 and abs(got[1] - ref["high"]) <= 1e-4


def synthetic_arrays(values: dict, n: int) -> dict:
    """arrays as the eval record holds them: every fit's per-query metric, the same values for every seed."""
    arrays = {}
    for arm, v in values.items():
        for s in D.SEEDS:
            for m in D.READ_METRICS:
                arrays[f"{D.fit_key(arm, s)}/{m}"] = np.asarray(v, dtype=np.float64)
    arrays["half"] = np.zeros(n, dtype=bool)
    return arrays


@pytest.mark.parametrize("fraction,band", [(1.0, "CK_KEEPS"), (0.8, "CK_KEEPS"), (0.5, "CK_PARTIAL"), (0.0, "CK_NO_GAIN"),
                                           (-0.2, "CK_NO_GAIN")])
def test_the_bands_land_where_the_rules_say(decl, fraction, band):
    n = 300
    rng = np.random.default_rng(1)
    twin = rng.uniform(0.0, 0.5, n)
    gain = rng.uniform(0.1, 0.3, n)
    values = {"twin": twin, "gnn": twin + gain, "ck_qi": twin + fraction * gain, "ck_self": twin, "ck_full": twin + gain}
    h = D.read_half(decl, synthetic_arrays(values, n), np.ones(n, dtype=bool), "recall@5")
    assert D.band_of(h) == band
    assert h["shares"]["ck_share"]["readable"] and abs(h["shares"]["ck_share"]["point"] - fraction) < 1e-9
    assert D.band_of(h, "ref_gain", "ref_share") == "CK_KEEPS"
    assert h["contrasts"]["compiled_cost"]["of"] == ["ck_qi", "ck_full"]


def test_no_gnn_gain_is_its_own_band_and_leaves_every_share_unread(decl):
    n = 200
    twin = np.random.default_rng(2).uniform(0.0, 1.0, n)
    values = {"twin": twin, "gnn": twin, "ck_qi": twin + 0.1, "ck_self": twin, "ck_full": twin}
    h = D.read_half(decl, synthetic_arrays(values, n), np.ones(n, dtype=bool), "recall@5")
    assert D.band_of(h) == "GNN_GAIN_ABSENT"
    assert all(not s["readable"] and s["ci"] is None for s in h["shares"].values())


def test_a_share_below_the_keep_threshold_with_a_wide_interval_is_not_kept(decl):
    n = 300
    rng = np.random.default_rng(3)
    twin = rng.uniform(0.0, 0.5, n)
    gain = rng.uniform(0.1, 0.3, n)
    noise = rng.normal(0.0, 1.5, n)
    noisy = twin + 0.78 * gain + (noise - noise.mean())   # the point at 0.78, the interval's lower end below 0.50
    values = {"twin": twin, "gnn": twin + gain, "ck_qi": noisy, "ck_self": twin, "ck_full": twin + gain}
    h = D.read_half(decl, synthetic_arrays(values, n), np.ones(n, dtype=bool), "recall@5")
    s = h["shares"]["ck_share"]
    assert s["point"] >= D.KEEP["share"] and s["ci"][0] < D.KEEP["share_low"] and D.band_of(h) != "CK_KEEPS"


# ── the smoke: every stage on the real 2wiki data ────────────────────────────


def package_place(decl) -> tuple[bool, bool]:
    """(host, reachable): the host mirror when the declared root exists here, else the laptop's package."""
    if Path(decl["inputs"]["mirror"]["root"]).exists():
        return True, True
    _, cfg_m3b, _ = V2.load_configs()
    return False, Path(cfg_m3b["substrate"]["package_root"]).exists()


@pytest.fixture(scope="module")
def opened(decl):
    host, reachable = package_place(decl)
    if not reachable:
        pytest.skip("neither the host mirror nor the laptop's package is on this machine")
    if not all((ROOT / p).exists() for p in D.pushed_paths(decl)):
        pytest.skip("the pinned 2wiki caches are not on this machine")
    return D.open_2wiki(decl, host=host, log=quiet)


def write_stand_in_fit(key: str, arm: str, seed: int, inputs, bank) -> None:
    """An untrained fit with the record fields the eval, read and doc stages read (the smoke fits two for real)."""
    torch.manual_seed(1000 + seed)
    model = D.build_model(arm, inputs, bank)
    state = {k: v.detach().to("cpu") for k, v in model.state_dict().items()}
    torch.save(state, D.FITS / f"{key}.pt")
    rec = {"key": key, "arm_key": arm, "seed": seed, "parameters": D.ARM_SPECS[arm]["parameters"], "best_epoch": 0, "epochs_run": 0,
           "best_select_macro_recall5": 0.0, "seconds": 0.0, "history": [], "cuda_peak_bytes": None, "warnings": [], "deviations": [],
           "module_sha256": {}, "state_sha256": D.sha256_file(D.FITS / f"{key}.pt"), "state_content_sha256": D.state_digest(state)}
    (D.FITS / f"{key}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_smoke_fit_repeat_eval_read_doc_on_real_2wiki(decl, opened, out_dirs, monkeypatch):
    cfg, cfg_m3b, cfg_h, inputs, contexts, handles, pkg, bank, m3b_compile = opened
    assert pkg[3]["RECORD_SHA256"] == decl["inputs"]["freeze_RECORD_SHA256"] and bank.n_rows == 0
    cuda = torch.cuda.is_available()
    if cuda:
        place, deviations = D.host_placement()   # host_gpu_det applied and read back, as the fit and eval stages do
    else:
        monkeypatch.setitem(D.SPEC, "device", "cpu")
        place, deviations = {"host": "laptop", "env": None, "device_name": "cpu", "driver": None, "torch": torch.__version__, "cuda": None}, []
    monkeypatch.setattr(D, "host_open", lambda d, log=None: (place, deviations, opened, {"smoke": True}))
    rule = D.training_rule(decl, cfg, cfg_m3b)
    monkeypatch.setattr(D, "training_rule", lambda d, c, m: {**rule, "max_epochs": 1, "batches_per_epoch": 2, "batch_size": 4, "patience": 1})
    monkeypatch.setattr(D, "LATENCY_QUERIES", 6)

    # one fit of the primary kernel and its repeat in the same process: bit-identical, both at the pinned count
    first = D.stage_fit(decl, "ck_qi", 0, log=quiet)
    again = D.stage_fit(decl, "ck_qi", 0, repeat=True, log=quiet)
    assert first["parameters"] == again["parameters"] == 377_803 and first["steps"] >= 1
    assert first["state_content_sha256"] == again["state_content_sha256"]
    assert D.trace(first["history"]) == D.trace(again["history"]) and first["history"][0]["seconds"] >= 0
    assert first["fit_queries"] == 5928 and first["select_queries"] == 1496 and first["relation_bank"]["rows"] == 0
    assert D.stage_fit(decl, "ck_qi", 0, log=quiet)["utc"] == first["utc"]   # a filed fit is not repeated
    for arm, seed, rep in D.declared_fits(decl):
        key = D.fit_key(arm, seed, rep)
        if not (D.FITS / f"{key}.json").exists():
            write_stand_in_fit(key, arm, seed, inputs, bank)

    # the eval pass on every 64th query of the population, every fit, the compiled form beside each kernel
    original = D.eval_placed
    monkeypatch.setattr(D, "eval_placed", lambda *a, **k: original(*a, shard=(0, 64), **k))
    ev = D.stage_eval(decl, log=quiet)
    n = len(range(0, 12576, 64))
    assert ev["queries"] == n and ev["shard"] == {"k": 0, "N": 64, "population_queries": 12576}
    assert ev["ids_sha256"] == decl["population"]["ids_sha256"]
    assert ev["halves"]["V2_GATE"] + ev["halves"]["V2_HELD_CONFIRMATION"] == n
    assert ev["m3b_fixed_rrf_agreement"]["ok"] and all(v["ok"] for v in ev["mrr_audit"].values())
    kernels = [D.fit_key(a, s, r) for a, s, r in D.declared_fits(decl) if a.startswith("ck_")]
    assert sorted(ev["compiled_form"]) == sorted(kernels) and len(kernels) == 10
    assert max(v["max_abs_score_difference"] for v in ev["compiled_form"].values()) <= 1e-4
    assert ev["latency"]["compile"]["n"] == 6 and ev["latency"][f"cpu:{D.fit_key('ck_qi', 0)}"]["n"] == 6
    with np.load(D.EVAL / f"{D.NAME}.npz") as z:
        assert all(z[f"{k}/{m}"].shape == (n,) for k in ev["scorers"] for m in METRIC_NAMES)

    # the read: the band, the flags, the repeat check; then the doc from the records
    read = D.stage_read(decl, log=quiet)
    assert read["reading"] in D.READINGS and set(read["flags"]) <= set(D.FLAGS)
    assert read["repeat"]["weights_bit_identical"] and read["repeat"]["metrics_identical"] and "REPEAT_DIFFERS" not in read["flags"]
    assert set(read["bands"]) == set(D.HALVES) and set(read["reference_bands"]) == set(D.HALVES)
    assert read["halves"]["whole"]["recall@5"]["queries"] == n
    ver = {"status": "VERIFIED", "mirror_verify_record_sha256": "0" * 64, "freeze_RECORD_SHA256": decl["inputs"]["freeze_RECORD_SHA256"],
           "stores_built_here": {f: False for f in FAMILIES}, "csr_content_sha256": decl["inputs"]["csr_content_sha256"],
           "fit_queries": 5928, "select_queries": 1496, "bank_rows": 0}
    (D.OUT / "verify.json").write_text(json.dumps(ver), encoding="utf-8")
    doc = D.stage_doc(decl, log=quiet)
    text = doc.read_text(encoding="utf-8")
    assert f"**Reading: `{read['reading']}`**" in text and "per-query reference `ck_full`" in text
    assert json.loads(D.RECORD.read_text(encoding="utf-8"))["reading"] == read["reading"]
