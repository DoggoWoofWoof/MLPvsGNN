"""Tests for scripts/universal_v2_six.py: the stage-2 tooling refuses what
authorization_stage_2_2026_09_22 and amendment_5_2026_09_23 forbid, and computes what they declare.

No fit, no compile and no eval pass runs here: the stages are exercised through their refusals and their
pure functions on synthetic arrays, as the pilot run tests do.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import universal_v2_six as SIXMOD          # noqa: E402
import universal_v2_run as V2              # noqa: E402

CONFIG = ROOT / "configs" / "universal_v2.yaml"
CARVES = ROOT / "outputs" / "m3b" / "carves.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def block(cfg) -> dict:
    return cfg[SIXMOD.STAGE2]


@pytest.fixture(scope="module")
def m3b() -> dict:
    return yaml.safe_load(V2.M3B_CONFIG.read_text(encoding="utf-8"))


def test_the_module_writes_beside_the_pilot_and_never_into_it():
    assert SIXMOD.SIX == V2.OUT / "six" and SIXMOD.SIX_FITS == SIXMOD.SIX / "fits" and SIXMOD.SIX_EVAL == SIXMOD.SIX / "eval"
    assert SIXMOD.SIX_FITS != V2.FITS and SIXMOD.SIX_EVAL != V2.EVAL
    assert SIXMOD.DOC == ROOT / "docs" / "UNIVERSAL_GNN_SIX.md" and SIXMOD.DOC != V2.DOC
    keys = {SIXMOD.fit_key(s) for s in SIXMOD.SEEDS}
    assert keys.isdisjoint({V2.fit_key(SIXMOD.ARM, s) for s in (0, 1, 2)})    # no stage-2 weight lands beside the trio fits
    assert all("__six__" in k for k in keys)


def test_the_constants_are_the_filed_ones(cfg, block):
    assert SIXMOD.ARM == block["the_freeze"]["arm"] == "u_gnn_v2_ef"
    assert sorted(SIXMOD.DATASETS) == sorted(k for k in block["populations_and_splits"] if k not in ("rule", "halves"))
    assert sorted(SIXMOD.ADDED) == ["hotpotqa", "musique", "webqsp"] and set(SIXMOD.TRIO) == set(SIXMOD.DATASETS) - set(SIXMOD.ADDED)
    assert list(SIXMOD.SEEDS) == list(block["training"]["seeds"]) == [0, 1, 2]
    assert SIXMOD.BAND == block["calibration_against_published_systems"]["verdicts"]
    assert SIXMOD.PARAMETERS == 420932 and SIXMOD.COLUMNS == 129
    head = block["evaluation_and_reading"]["headline_per_dataset"]
    for name, metrics in SIXMOD.HEADLINE.items():
        for m in metrics:
            assert m in head[name], (name, m)


def test_the_stage_blocks_are_required(cfg):
    assert SIXMOD.stage2_block(cfg)["stage_2_status"] in ("DECLARED_NOT_RUN", "RUN")
    assert SIXMOD.go_ahead_block(cfg)["filed_before"].startswith("any stage-2 cache")
    for missing in (SIXMOD.STAGE2, SIXMOD.AMD5):
        stripped = {k: v for k, v in cfg.items() if k != missing}
        with pytest.raises(SystemExit, match="declaration"):
            (SIXMOD.stage2_block if missing == SIXMOD.STAGE2 else SIXMOD.go_ahead_block)(stripped)


def test_the_carve_check_reads_the_m3b_file_itself(cfg, block):
    m3b = json.loads(CARVES.read_text(encoding="utf-8"))["per_dataset"]
    for name in SIXMOD.ADDED:
        for kind in ("fit", "select"):
            d = SIXMOD.declared_carve(block, name, kind)
            assert d["ids"] == int(m3b[name][kind]) and d["sha256"] == m3b[name][f"{kind}_sha256"]
    bad = copy.deepcopy(block)
    bad["training_carves"]["musique"]["fit"] = 1
    with pytest.raises(SystemExit, match="differs from outputs/m3b/carves.json"):
        SIXMOD.declared_carve(bad, "musique", "fit")


def test_the_disk_guard_takes_the_raised_bound_from_the_amendment_and_the_floor_from_compute(cfg):
    guard = SIXMOD.SixDiskGuard(cfg)
    assert guard.bound == 20e9 and guard.halt_below == 8e9
    stripped = copy.deepcopy({k: v for k, v in cfg.items() if k != "compute"})
    stripped["compute"] = {"abort_criteria": ["no floor here"]}
    with pytest.raises(SystemExit, match="disk floor"):
        SIXMOD.SixDiskGuard(stripped)
    no_bound = copy.deepcopy(cfg)
    no_bound[SIXMOD.AMD5]["systems_only_change"]["what"] = "nothing is raised"
    with pytest.raises(SystemExit, match="raised cache bound"):
        SIXMOD.SixDiskGuard(no_bound)


def test_compile_refuses_the_trio_and_anything_but_the_training_carves(cfg, m3b):
    with pytest.raises(SystemExit, match="the trio caches are not recompiled"):
        SIXMOD.stage_compile_six(cfg, m3b, {}, ["metaqa"], ("fit", "select"), log=lambda *a: None)
    with pytest.raises(SystemExit, match="the eval populations are compiled at eval time"):
        SIXMOD.stage_compile_six(cfg, m3b, {}, ["musique"], ("fit", "eval"), log=lambda *a: None)


def test_the_fit_refuses_a_seed_outside_the_declaration_and_a_warm_start(cfg, m3b, tmp_path, monkeypatch):
    # an empty fits directory: once the live seed-0 record exists the stage returns it before reaching the guard
    monkeypatch.setattr(SIXMOD, "SIX_FITS", tmp_path / "fits")
    with pytest.raises(SystemExit, match=r"seed 3: the stage fits seeds \[0, 1, 2\]"):
        SIXMOD.stage_fit_six(cfg, m3b, 3, log=lambda *a: None)
    loosened = copy.deepcopy(cfg)
    loosened[SIXMOD.STAGE2]["the_freeze"]["no_warm_start"] = "warm starting is fine"
    with pytest.raises(SystemExit, match="no longer bars a warm start"):
        SIXMOD.stage_fit_six(loosened, m3b, 0, log=lambda *a: None)


def test_the_no_warm_start_clause_as_filed_satisfies_the_guard(cfg):
    """The case above loosens the clause and expects a refusal; this is the other half -- the clause as the
    declaration actually writes it must PASS. It did not: the sentence opens with a capital N and the guard
    compared case-sensitively, so every fit refused in seven seconds before any weight was touched."""
    clause = " ".join(str(SIXMOD.stage2_block(cfg)["the_freeze"]["no_warm_start"]).split())
    assert SIXMOD.NO_WARM_START in clause.lower(), clause
    assert "fine-tuned or used as an initialization" in clause


def test_the_eval_pass_refuses_a_dataset_outside_the_six(cfg, m3b):
    with pytest.raises(SystemExit, match=r"\['nq'\]: not datasets of the stage"):
        SIXMOD.stage_eval_six(cfg, m3b, {}, ["nq"], 4096, log=lambda *a: None)


def test_the_populations_are_the_filed_ones_and_webqsp_stays_on_train_holdout(block):
    pops = {n: SIXMOD.declared_population(block, n) for n in SIXMOD.DATASETS}
    assert pops["webqsp"]["split"] == "train_holdout" and pops["webqsp"]["queries"] == 1503
    assert {n: p["queries"] for n, p in pops.items()} == {"metaqa": 39138, "2wiki": 12576, "squad": 11873,
                                                          "hotpotqa": 7405, "musique": 2417, "webqsp": 1503}
    assert all(p["split"] != "test" for p in pops.values()) and all(len(p["ids_sha256"]) == 64 for p in pops.values())


def test_the_training_rule_is_the_frozen_one_and_a_changed_value_refuses(cfg, m3b, monkeypatch):
    rule = SIXMOD.training_rule_six(cfg, m3b)
    assert rule["max_epochs"] == 6 and rule["batches_per_epoch"] == 2000 and rule["dataset_draw"] == "per_query"
    monkeypatch.setattr(V2, "training_rule_v2", lambda *a: {**rule, "max_epochs": 8})
    with pytest.raises(SystemExit, match="differ from the frozen rule"):
        SIXMOD.training_rule_six(cfg, m3b)


def test_the_fit_hours_guard_needs_the_measured_epoch_and_stops_at_the_ceiling(tmp_path, monkeypatch):
    monkeypatch.setattr(SIXMOD, "SIX", tmp_path)
    monkeypatch.setattr(SIXMOD, "SIX_FITS", tmp_path / "fits")
    with pytest.raises(SystemExit, match="the measured joint epoch precedes every stage-2 fit"):
        SIXMOD.fit_hours_guard_six({"max_epochs": 6}, log=lambda *a: None)
    (tmp_path / "timing.json").write_text(json.dumps({"epoch_seconds": 3600.0}), encoding="utf-8")
    out = SIXMOD.fit_hours_guard_six({"max_epochs": 6}, log=lambda *a: None)
    assert out["projected_fit_hours"] == 6.0 and out["ceiling_fit_hours"] == V2.FIT_HOURS_CEILING and out["within_ceiling"]
    (tmp_path / "timing.json").write_text(json.dumps({"epoch_seconds": 3600.0 * 40}), encoding="utf-8")
    with pytest.raises(SystemExit, match="compute ceiling"):
        SIXMOD.fit_hours_guard_six({"max_epochs": 6}, log=lambda *a: None)
    assert json.loads((tmp_path / "ceiling_breach.json").read_text(encoding="utf-8"))["within_ceiling"] is False


def _arrays(keys, n=40, rng=None):
    rng = rng or np.random.default_rng(0)
    out = {}
    for k in keys:
        for m in SIXMOD.REPORTED_METRICS:
            out[f"{k}/{m}"] = rng.random(n)
        out[f"{k}/gold_total"] = rng.integers(1, 4, n).astype(np.int64)
        for m in ("gate_step", "gate_evidence", "delta_ratio", "top1_changed"):
            out[f"{k}/{m}"] = rng.random(n)
    out["gold_dist_struct"] = rng.integers(-1, 4, n).astype(np.int64)
    out["hop"] = rng.integers(1, 4, n).astype(np.int64)
    return out


def test_a_cell_is_three_seeds_and_the_paired_delta_of_a_scope_against_itself_is_zero():
    ours_keys = [SIXMOD.fit_key(s) for s in SIXMOD.SEEDS]
    ours = _arrays(ours_keys)
    for k in SIXMOD.M3B_SEEDED.values():
        ours.update({f"{k.format(s=s)}/{m}": ours[f"{ours_keys[s]}/{m}"] for s in (0, 1, 2) for m in SIXMOD.REPORTED_METRICS})
    ours["fixed:rrf/recall@5"] = ours[f"{ours_keys[0]}/recall@5"]
    mask = np.ones(40, dtype=bool)
    c = SIXMOD.cell(ours, ours_keys, "recall@5", mask)
    assert c["seeds"] == 3 and len(c["per_seed"]) == 3
    assert abs(c["mean"] - float(np.mean([ours[f"{k}/recall@5"].mean() for k in ours_keys]))) < 5e-5
    mean = SIXMOD.seed_mean_per_query(ours, ours_keys, "recall@5", mask)
    assert mean.shape == (40,) and abs(float(mean.mean()) - c["mean"]) < 5e-5
    paired = V2.paired_bootstrap(mean, SIXMOD.seed_mean_per_query(ours, [SIXMOD.M3B_SEEDED["gat_universal_v1"].format(s=s) for s in (0, 1, 2)], "recall@5", mask))
    assert abs(paired["mean"]) < 1e-12 and abs(paired["low"]) < 1e-12 and abs(paired["high"]) < 1e-12


def test_the_slices_are_masks_over_the_same_scope():
    keys = [SIXMOD.fit_key(0)]
    ours = _arrays(keys)
    mask = np.zeros(40, dtype=bool)
    mask[:25] = True
    sl = SIXMOD.slices_for("metaqa", ours, mask)
    assert {"1hop", "2hop", "3hop"} <= set(sl) and all(m.shape == (25,) for m in sl.values())
    assert sum(int(sl[f"{h}hop"].sum()) for h in (1, 2, 3)) == 25
    assert "1hop" not in SIXMOD.slices_for("squad", ours, mask)
    buckets = [b for b in ("gold_at_seed", "gold_1_hop", "gold_2_hops", "gold_3_or_more", "no_gold_in_pool") if b in sl]
    assert sum(int(sl[b].sum()) for b in buckets) == 25


def test_the_reading_of_one_scope_carries_every_reported_metric_and_no_threshold():
    ours_keys = [SIXMOD.fit_key(s) for s in SIXMOD.SEEDS]
    ours = _arrays(ours_keys)
    ours["fixed:rrf/recall@5"] = ours[f"{ours_keys[0]}/recall@5"]
    for m in SIXMOD.REPORTED_METRICS:
        ours[f"fixed:rrf/{m}"] = ours[f"{ours_keys[0]}/{m}"]
    theirs = _arrays([p.format(s=s) for p in SIXMOD.M3B_SEEDED.values() for s in (0, 1, 2)], rng=np.random.default_rng(1))
    refs = SIXMOD.reference_keys(theirs, "2wiki")
    assert sorted(refs) == ["fixed:rrf", "gat_no_mp_v1", "gat_universal_v1", "qls_u_sota_v1"]
    scope = SIXMOD.read_scope("2wiki", ours, theirs, ours_keys, refs, np.ones(40, dtype=bool))
    assert scope["queries"] == 40 and sorted(scope["ours"]) == sorted(SIXMOD.REPORTED_METRICS)
    assert sorted(scope["paired"]) == sorted(refs) and set(scope["paired"]["gat_universal_v1"]) == set(SIXMOD.REPORTED_METRICS)
    assert set(scope["mechanism"]) == set(ours_keys)
    text = json.dumps(scope)
    assert "PASS" not in text and "FAIL" not in text and "verdict" not in text    # the stage has no gate


def test_no_test_split_is_scored(cfg, m3b):
    splits = m3b["populations"]["eval_splits"]
    assert all("test" not in str(splits[n]) for n in SIXMOD.DATASETS), splits
    assert splits["webqsp"] == "train_holdout"
    stub = types.SimpleNamespace(frozen_contract=lambda c: ("key", {"per_dataset": {"2wiki": {"construction": {}}}}))
    poisoned = copy.deepcopy(m3b)
    poisoned["populations"]["eval_splits"]["2wiki"] = "test"
    with pytest.raises(SystemExit, match="is a test split; no test split is authorised"):
        SIXMOD.eval_dataset_six("2wiki", cfg, poisoned, {}, {}, {}, None, None, (None, None, None, None), stub, None, 4096,
                                log=lambda *a: None)


def test_only_a_dated_stage_2_key_is_appended_and_only_once(tmp_path, cfg):
    config = tmp_path / "universal_v2.yaml"
    with open(config, "w", encoding="utf-8", newline=V2.LF) as f:
        f.write("status: RUN_PILOT_FAILED" + V2.LF)
    live = {"status": "RUN_PILOT_FAILED"}
    for bad in ("run_record_2026_09_21", "authorization_stage_2_2026_09_22", "run_record_stage_2", "notes_stage_2_2026_09_23"):
        with pytest.raises(SystemExit, match="not a dated stage-2 block key"):
            SIXMOD.append_block_six(live, bad, {"a": 1}, "h", config=config)
    key = "run_record_stage_2_2026_09_23"
    SIXMOD.append_block_six(live, key, {"stage_2_status": "RUN", "n": 1}, "the terminal state", config=config)
    reloaded = yaml.safe_load(config.read_text(encoding="utf-8"))
    assert reloaded[key] == {"stage_2_status": "RUN", "n": 1} and reloaded["status"] == "RUN_PILOT_FAILED"
    assert "\r" not in config.read_text(encoding="utf-8", newline="") and "# -- the terminal state --" in config.read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="never a re-file"):
        SIXMOD.append_block_six(live, "run_record_stage_2_2026_09_24", {"n": 2}, "h", config=config)


def test_a_moved_status_line_refuses_the_block(tmp_path):
    config = tmp_path / "universal_v2.yaml"
    with open(config, "w", encoding="utf-8", newline=V2.LF) as f:
        f.write("status: RUN_PILOT_FAILED" + V2.LF)
    live = {"status": "RUN_PILOT_FAILED"}
    original = SIXMOD.yaml.safe_load

    def moved(text):
        out = original(text)
        if "timing_stage_2" in text and isinstance(out, dict) and "timing_stage_2_2026_09_23" in out:
            out["status"] = "RUN_SIX"
        return out

    SIXMOD.yaml.safe_load = moved
    try:
        with pytest.raises(SystemExit, match="the status line moved"):
            SIXMOD.append_block_six(live, "timing_stage_2_2026_09_23", {"n": 1}, "h", config=config)
    finally:
        SIXMOD.yaml.safe_load = original


def test_every_block_is_filed_from_its_sidecar_and_the_document_from_the_reading(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(SIXMOD, "SIX", tmp_path)
    monkeypatch.setattr(SIXMOD, "DOC", tmp_path / "UNIVERSAL_GNN_SIX.md")
    for builder, match in ((SIXMOD.compile_block, "no six/compile_record.json"), (SIXMOD.timing_block, "no six/timing.json"),
                           (SIXMOD.record_block, "no six/read_record.json")):
        with pytest.raises(SystemExit, match=match):
            builder(cfg)
    with pytest.raises(SystemExit, match="the document is rendered from the reading"):
        SIXMOD.stage_doc_six(cfg, log=lambda *a: None)
    (tmp_path / "read_record.json").write_text(json.dumps({"per_dataset": {}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="the document precedes the run record"):
        SIXMOD.record_block(cfg)


def test_the_formatters_carry_the_spread_and_the_interval():
    assert SIXMOD.fmt({"mean": 0.5213, "sd": 0.0041}) == "0.5213 +/- 0.0041"
    assert SIXMOD.fmt_paired({"mean": 0.018, "low": -0.001, "high": 0.037}) == "+0.0180 [-0.0010, +0.0370]"


def test_a_shard_is_a_slice_of_one_population_and_the_merge_waits_for_all_of_them(tmp_path, monkeypatch):
    assert SIXMOD.parse_shard(None) is None and SIXMOD.parse_shard("0/4") == (0, 4) and SIXMOD.parse_shard("3/4") == (3, 4)
    for bad in ("4/4", "5/4"):
        with pytest.raises(SystemExit, match=r"k must be in \[0, N\)"):
            SIXMOD.parse_shard(bad)
    monkeypatch.setattr(SIXMOD, "SIX_EVAL", tmp_path)
    assert SIXMOD.merge_shards_six("musique", log=lambda *a: None) is None
    seen = []
    (tmp_path / "musique__shard0of2.json").write_text(json.dumps({"shard": {"k": 0, "N": 2}}), encoding="utf-8")
    assert SIXMOD.merge_shards_six("musique", log=seen.append) is None and "shards [0] of 2 present" in seen[-1]


def test_a_column_that_is_constant_or_unavailable_on_an_added_dataset_is_reported_and_kept(monkeypatch):
    names = [f"c{i}" for i in range(5)]
    monkeypatch.setattr(V2, "column_stats_v2", lambda d, i: {"rows": 10, "availability": np.array([1.0, 0.0, 1.0, 1.0, 0.5]),
                                                             "variance": np.array([1.0, 0.0, 0.0, 2.0, 1.0])})
    out = SIXMOD.column_behaviour(Path("."), {"columns": names, "column_indices": None})
    assert out["columns"] == 5 and out["rows"] == 10
    assert out["unavailable_kept"] == ["c1"] and out["constant_kept"] == ["c2"]
    assert out["availability_min"] == 0.0 and "kept" in out["action"] and "does not move" in out["action"]


def test_the_freeze_check_refuses_a_moved_contract(cfg):
    good = {"contract_block": V2.frozen_contract_v2(cfg)[0], "n_scalars": 129, "core_sha256": "8d1da88b14df" + "0" * 52,
            "base": "rrf", "evidence": ["rrf", "dense_cos", "splade_rr", "is_seed"]}
    out = SIXMOD.check_freeze(cfg, good)
    assert out["columns"] == 129 and out["declared_parameters"] == 420932 and out["hidden"] == 128 and out["arm"] == SIXMOD.ARM
    for field, value in (("n_scalars", 130), ("base", "dense_cos"), ("core_sha256", "0" * 64),
                         ("evidence", ["rrf", "dense_cos", "is_seed"])):
        with pytest.raises(SystemExit, match="hard stop"):
            SIXMOD.check_freeze(cfg, {**good, field: value})
    loosened = copy.deepcopy(cfg)
    loosened[SIXMOD.STAGE2]["the_freeze"]["what_is_frozen"] = "the architecture is frozen"
    with pytest.raises(SystemExit, match="no longer states the frozen architecture"):
        SIXMOD.check_freeze(loosened, good)


def test_the_cli_offers_the_filed_stages_only(tmp_path, monkeypatch):
    # an empty fits directory: on the live tree, once the eval pass exists, --stage read would run the reading
    # and write outputs/universal_v2/six/read_record.json from inside a test
    monkeypatch.setattr(SIXMOD, "SIX_FITS", tmp_path / "fits")
    with pytest.raises(SystemExit):
        SIXMOD.main(["--stage", "screen"])
    with pytest.raises(SystemExit, match="the reading follows the three fits"):
        SIXMOD.main(["--stage", "read"])


def test_the_stage_matches_the_trio_fit_on_the_architecture_and_only_the_served_bank_may_grow(cfg):
    """stage_fit_six compares its frozen dict against the trio fit of the same arm. Every field of that
    comparison is checked here against the committed trio records; the served relation bank is excluded from it
    because it is the concatenated relation embeddings of whichever datasets are open -- an input, not a weight."""
    _, contract = V2.frozen_contract_v2(cfg)
    for seed in SIXMOD.SEEDS:
        trio = V2.read_json(V2.FITS / f"{V2.fit_key(SIXMOD.ARM, seed)}.json")
        assert trio is not None, f"the replicated trio fit of seed {seed} is the thing this stage refits"
        assert trio["parameters"] == SIXMOD.PARAMETERS and trio["hidden"] == 128 and trio["columns"] == SIXMOD.COLUMNS
        assert trio["contract_block"] == V2.frozen_contract_v2(cfg)[0] and trio["core_sha256"].startswith("8d1da88b14df")
        assert trio["evidence"] == ["rrf", "dense_cos", "splade_rr", "is_seed"]
        assert trio["evidence_substitutions"] == {"splade_score_norm": "splade_rr"}
        assert int(trio["relation_bank"]["k_rel"]) == 4
    src = (ROOT / "scripts" / "universal_v2_six.py").read_text(encoding="utf-8")
    assert 'if k not in ("training", "relation_bank")' in src and "never a parameter" in src


def test_the_contract_is_reported_per_substrate_and_nothing_is_dropped_anywhere(cfg):
    """contract_transfer_to_the_added_datasets is reported, not acted on: the same 129 columns on every dataset,
    with the availability each substrate can give them."""
    out = SIXMOD.contract_on_datasets(cfg)
    assert list(out) == [n for n in SIXMOD.DATASETS if n in out]          # the filed dataset order, never sorted
    for name, c in out.items():
        assert c["columns"] == SIXMOD.COLUMNS and 0.0 <= c["availability_mean"] <= 1.0 and c["rows"] > 0
        assert c["unavailable"] == len(c["unavailable_columns"])
        assert "sampled" in c["source"] or "full scan" in c["source"]
    for name in SIXMOD.TRIO:
        assert name in out and "pilot screen" in out[name]["source"] and out[name]["constant"] is None
    record = V2.read_json(SIXMOD.SIX / "compile_record.json") or {"per_dataset": {}}
    for name in SIXMOD.ADDED:
        if "column_behaviour" in record["per_dataset"].get(name, {}):
            cb = record["per_dataset"][name]["column_behaviour"]
            assert out[name]["unavailable"] == len(cb["unavailable_kept"]) and out[name]["rows"] == cb["rows"]
            assert "reported and kept" in cb["action"]


def test_every_mechanism_readout_the_pass_produced_reaches_the_document():
    read = {"per_dataset": {
        "metaqa": {"scopes": {"whole": {"mechanism": {"k0": {"delta_ratio": 1.0, "top1_changed": 0.5, "gate_step1": 0.2,
                                                             "gate_step2": 0.3, "gate2_step1": 0.4, "gate2_step2": 0.1}}}}},
        "squad": {"scopes": {"whole": {"mechanism": {"k0": {"delta_ratio": 0.0, "top1_changed": 0.0}}}}}}}
    assert SIXMOD.mechanism_columns(read) == ["delta_ratio", "top1_changed", "gate_step1", "gate_step2",
                                              "gate2_step1", "gate2_step2"]
    assert SIXMOD.mechanism_columns({"per_dataset": {"x": {"scopes": {"whole": {"mechanism": {}}}}}}) == []


def test_the_trio_context_is_the_pilot_arrays_themselves_and_only_on_the_trio(cfg):
    """amendment_6: the column is read from the pinned pilot file, on the three datasets that have a trio
    checkpoint, and it is a paired delta against it -- nothing is rescored and no threshold appears."""
    added = SIXMOD.context_block(cfg)["what_is_added"]
    ids = json.loads((V2.EVAL / "2wiki_query_ids.json").read_text(encoding="utf-8"))
    with np.load(V2.EVAL / "2wiki.npz") as z:
        arm = added["context_arm"]
        trio = {m: z[f"{arm}/{m}"] for m in SIXMOD.REPORTED_METRICS}
    n = len(ids)
    our_keys = [SIXMOD.fit_key(s) for s in SIXMOD.SEEDS]
    ours = {f"{k}/{m}": trio[m] for k in our_keys for m in SIXMOD.REPORTED_METRICS}
    mask = np.ones(n, dtype=bool)
    out = SIXMOD.trio_context(cfg, "2wiki", ours, our_keys, {"whole": mask}, ids)
    assert out["arm"] == arm and out["seeds"] == 1 and out["declared_in"] == SIXMOD.AMD6
    assert out["trained_on"] == sorted(SIXMOD.TRIO) and out["we_are_trained_on"] == sorted(SIXMOD.DATASETS)
    scope = out["scopes"]["whole"]
    assert scope["queries"] == n and sorted(scope["trio"]) == sorted(SIXMOD.REPORTED_METRICS)
    for m in SIXMOD.REPORTED_METRICS:                      # ours is the trio itself here: every delta is zero
        p = scope["paired_six_minus_trio"][m]
        assert abs(p["mean"]) < 1e-12 and abs(p["low"]) < 1e-12 and abs(p["high"]) < 1e-12
        assert abs(scope["trio"][m]["mean"] - float(trio[m].mean())) < 5e-5
    assert "PASS" not in json.dumps(out) and "verdict" not in json.dumps(out)
    for name in SIXMOD.ADDED:                              # no trio checkpoint was ever fitted on these
        assert SIXMOD.trio_context(cfg, name, ours, our_keys, {"whole": mask}, ids) is None


def test_the_trio_context_refuses_a_moved_file_a_different_query_list_and_a_missing_declaration(cfg):
    ids = json.loads((V2.EVAL / "2wiki_query_ids.json").read_text(encoding="utf-8"))
    our_keys = [SIXMOD.fit_key(s) for s in SIXMOD.SEEDS]
    mask = np.ones(len(ids), dtype=bool)
    with pytest.raises(SystemExit, match="refusing to print it"):
        SIXMOD.context_block({k: v for k, v in cfg.items() if k != SIXMOD.AMD6})
    moved = copy.deepcopy(cfg)
    moved[SIXMOD.AMD6]["what_is_added"]["source_files"]["2wiki"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="hard stop"):
        SIXMOD.trio_context(moved, "2wiki", {}, our_keys, {"whole": mask}, ids)
    with pytest.raises(SystemExit, match="query lists differ"):
        SIXMOD.trio_context(cfg, "2wiki", {}, our_keys, {"whole": mask}, ids[:-1] + ["not the same query"])


def test_the_graph_er_pr_at_k_is_full_coverage_and_never_recall():
    from mp_retrieval.m3b_train import rank_metrics
    scores = np.arange(10, 0, -1, dtype=np.float64)          # candidate i is ranked i + 1
    both = rank_metrics(scores, np.array([0, 1, 7]), 3)      # golds at ranks 1, 2 and 8
    assert abs(both["recall@5"] - 2 / 3) < 1e-12 and both["full_coverage@5"] == 0.0
    assert both["recall@20"] == 1.0 and both["full_coverage@20"] == 1.0
    missing = rank_metrics(scores, np.array([0, 1, 7]), 4)   # a fourth gold the pool never reached counts against it
    assert missing["recall@20"] == 0.75 and missing["full_coverage@20"] == 0.0
    text = " ".join(SIXMOD.PR_AT_K_EQUIVALENCE)
    assert "`full_coverage@K` is the GraphER-compatible PR@K" in text and "never merged" in text
    assert "full_coverage@5" in SIXMOD.REPORTED_METRICS and "full_coverage@20" in SIXMOD.REPORTED_METRICS
    src = (ROOT / "scripts" / "universal_v2_six.py").read_text(encoding="utf-8")
    assert 'cal["care"], "", *PR_AT_K_EQUIVALENCE, "",' in src


def test_the_cost_section_labels_every_number_with_the_threads_it_ran_at():
    c = {"compile_rows": 123, "compile_hours": 1.5, "measured_epoch_seconds": 20282.9, "timing_peak_rss_gb": 9.1,
         "threads": 6, "fit_hours_this_stage": 84.67, "fit_peak_rss_gb": 10.2, "fit_hours_spent_total": 131.03,
         "ceiling_fit_hours": 150.0, "eval_hours": 7.25, "eval_threads": [4], "eval_peak_rss_gb": 6.3,
         "placement": "the laptop, one fit lane at the declared threads"}
    checkpoints = {SIXMOD.fit_key(s): {"threads": 8} for s in SIXMOD.SEEDS}
    lines = SIXMOD.cost_lines(c, checkpoints)
    assert "20282.9 s at 6 threads" in lines[1]              # amendment 7: a 6-thread number, labelled wherever it is read
    assert "84.67 fit-hours at 8 threads" in lines[2] and "131.03 of the 150 fit-hour ceiling" in lines[2]
    assert "7.25 hours at 4 threads, one dataset at a time" in lines[3]
    mixed = {**checkpoints, SIXMOD.fit_key(0): {"threads": 6}}
    assert "at 6, 8 threads" in SIXMOD.cost_lines(c, mixed)[2]  # a fit off the placement shows; it is not averaged away
    src = (ROOT / "scripts" / "universal_v2_six.py").read_text(encoding="utf-8")
    assert '*cost_lines(c, read["checkpoints"])' in src


def _checkpoint(seed, best, path):
    return {"seed": seed, "best_epoch": best, "epochs_run": len(path), "max_epochs": 6, "select_macro_by_epoch": path,
            "seconds": 3600.0, "state_sha256": "ab" * 32, "threads": 8}


def test_the_fitted_table_bolds_the_selected_epoch_and_names_a_seed_the_cap_ended():
    cks = {SIXMOD.fit_key(0): _checkpoint(0, 3, [0.1, 0.2, 0.3, 0.4, 0.35, 0.3]),
           SIXMOD.fit_key(1): _checkpoint(1, 5, [0.1, 0.2, 0.3, 0.4, 0.45, 0.5])}
    assert not SIXMOD.at_the_cap(cks[SIXMOD.fit_key(0)]) and SIXMOD.at_the_cap(cks[SIXMOD.fit_key(1)])
    lines = SIXMOD.fitted_lines(cks)
    assert "| 3 | 6 | 0.1000 / 0.2000 / 0.3000 / **0.4000** / 0.3500 / 0.3000 | 1.00 |" in lines[2]
    assert "| 5 | 6 | 0.1000 / 0.2000 / 0.3000 / 0.4000 / 0.4500 / **0.5000** | 1.00 |" in lines[3]
    assert "never a result" in " ".join(lines) and "epoch 5 is the last of the 6" in " ".join(lines)
    assert any(line.startswith("Seed 1 was selected at that last epoch") for line in lines)
    both = {k: _checkpoint(v["seed"], 5, [0.1, 0.2, 0.3, 0.4, 0.45, 0.5]) for k, v in cks.items()}
    assert any(line.startswith("Seeds 0, 1 were selected") for line in SIXMOD.fitted_lines(both))
    neither = {SIXMOD.fit_key(0): cks[SIXMOD.fit_key(0)]}
    assert SIXMOD.fitted_lines(neither)[-1] == "No checkpoint was selected at the last epoch the rule allows."


def test_the_per_seed_table_heads_each_seed_with_the_epoch_it_was_selected_at():
    read = {"checkpoints": {SIXMOD.fit_key(s): _checkpoint(s, b, [0.5] * 6) for s, b in zip(SIXMOD.SEEDS, (3, 5, 4))},
            "per_dataset": {}}
    for name in SIXMOD.DATASETS:
        ours = {m: {"per_seed": [0.5, 0.52, 0.51], "mean": 0.51, "sd": 0.0082, "seeds": 3} for m in SIXMOD.HEADLINE[name]}
        read["per_dataset"][name] = {"headline": list(SIXMOD.HEADLINE[name]), "scopes": {"whole": {"ours": ours}}}
    lines = SIXMOD.per_seed_lines(read)
    assert "| seed 0, epoch 3 | seed 1, epoch 5, at the cap | seed 2, epoch 4 | mean +/- sd | max - min |" in lines[3]
    assert "| musique | recall@5 | 0.5000 | 0.5200 | 0.5100 | 0.5100 +/- 0.0082 | 0.0200 |" in lines
    assert len(lines) - 5 == sum(len(SIXMOD.HEADLINE[n]) for n in SIXMOD.DATASETS)
    src = (ROOT / "scripts" / "universal_v2_six.py").read_text(encoding="utf-8")
    assert 'L += ["## 1. What was fitted", "", *fitted_lines(read["checkpoints"])]' in src
    assert 'L += ["", *per_seed_lines(read)]' in src
    assert '"select_macro_by_epoch": [round(float(h["select_macro_recall@5"]), 4) for h in rec["history"]]' in src
