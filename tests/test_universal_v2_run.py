"""The run and report tooling of universal-v2 on a synthetic sidecar world, in
the declared order (configs/universal_v2.yaml#order_of_operations,
#amendment_1_2026_09_19.check_3_firewalls_made_explicit.refusals_in_code):
the M3B pins; the screen reading fit caches only and refusing a select cache,
a non-pilot dataset and a second run after contract_frozen; every later stage
refusing before contract_frozen; the timing run before any fit; the seed-0
fits; u_gnn_v2_core78 refused before selection.json; the selection written
once and refused after an eval record; the gate refused before the selection,
read once, never reading V2_HELD_CONFIRMATION (held rows set to NaN leave every
gate number finite); seeds 1-2 refused without a gate pass; the supplement
eval pass; the held column produced once by the report and reused thereafter;
the dated-block filing, the status moves and the run record. The M3B files
are never touched; the real caches are never compiled; no real fit runs."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_universal_v2_models as TOY  # noqa: E402  (the toy substrate and caches of the models test)
from mp_retrieval import m3b_train as T  # noqa: E402
from mp_retrieval import universal_v2_features as V  # noqa: E402
from mp_retrieval import universal_v2_models as M  # noqa: E402


def _load(name: str):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


R = _load("universal_v2_run")
REPORT = _load("universal_v2_report")
PILOT = list(R.PILOT)
DATE = "2026_09_20"
LF = chr(10)
S = SimpleNamespace()   # the state of the sequential tests below, in declared order


# ── the synthetic world ──────────────────────────────────────────────────────


def _quiet(msg: str) -> None:
    pass


def _toy_diagnostics(n: int, dataset: str) -> dict:
    """The shape CompileDiagnostics.summary writes (the real values come from the compile); metaqa by hop."""
    def slot(q):
        return {"queries": q, "rows": 50 * q, "gold_rows": 2 * q, "typed_queries": q if dataset == "metaqa" else 0,
                "relation_slots": {"pairs": 100, "entries": 130, "pairs_truncated": 3, "fraction_of_pairs_truncated": 0.03, "queries_with_any_truncated_pair": 1,
                                   "fraction_of_queries_with_any_truncated_pair": 0.1, "relations_per_pair_histogram": {"1": 80, "2": 17, "5": 3}, "max_relations_per_pair": 5},
                "typed_walk_availability": {f"h{t}": {"rows": 0.5, "gold_rows": 0.9, "queries": 1.0, "queries_gold": 0.9} for t in (1, 2, 3)},
                "ordered_path": {f"h{t}": {"rows_with_walk": 25 * q, "fraction_of_walks_with_an_inverse_step": 0.4, "fraction_of_walks_composing_different_relations": 0.7} for t in (2, 3)}}
    by_hop = {"1hop": slot(n // 3), "2hop": slot(n // 3), "3hop": slot(n - 2 * (n // 3)), "all": slot(n)} if dataset == "metaqa" else {"all": slot(n)}
    return {"k_rel": 4, "by_hop": by_hop}


def _extend_meta(path: Path, kind: str, ids_sha: str) -> None:
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta.update({"population": {"ids": meta["n_queries"], "zero_gold_excluded": 0, "kept": meta["n_queries"], "ids_sha256": ids_sha},
                 "carve_sha256_declared": ids_sha, "ms_per_query": 12.5, "queries_per_second": 80.0, "peak_rss_bytes": 1 << 30, "compile_seconds": 3.0,
                 "group_seconds_per_1000_queries": {"depth_basis": 4.0, "typed_basis": 1.0}, "seeds_added_mean": 0.0,
                 "relation_slots": {"k_rel": 4, "queries": meta["n_queries"], "pairs": 100, "truncated": 3, "entries": 130, "fraction_of_pairs_truncated": 0.03,
                                    "fraction_of_queries_with_any_truncated_pair": 0.1},
                 "diagnostics": _toy_diagnostics(meta["n_queries"], path.parent.parent.name),
                 "relation_table": None, "m3b_contract_block": "candidate_contract_frozen_2026_09_13", "pool": "toy", "kind": kind})
    path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


@pytest.fixture(scope="module")
def sandbox(tmp_path_factory):
    """Every path of the run and report tooling redirected into a temporary tree: the declaration copied, the
    caches, fits, evals and the synthetic M3B eval arrays; the real outputs/, docs/ and configs/ are untouched."""
    base = tmp_path_factory.mktemp("v2run")
    mp = pytest.MonkeyPatch()
    out = base / "outputs" / "universal_v2"
    m3b_out = base / "outputs" / "m3b"
    config = base / "universal_v2.yaml"
    config.write_bytes(R.CONFIG.read_bytes())
    for name, value in (("OUT", out), ("CACHE", out / "cache"), ("FITS", out / "fits"), ("EVAL", out / "eval"), ("CONFIG", config),
                        ("M3B_OUT", m3b_out), ("DOC", base / "docs" / "UNIVERSAL_V2_PILOT.md")):
        mp.setattr(R, name, value)
    out.mkdir(parents=True)
    (m3b_out / "eval").mkdir(parents=True)
    rng = np.random.default_rng(7)
    contexts = {"metaqa": TOY._dataset(rng, "metaqa", 300, True, out / "cache"), "2wiki": TOY._dataset(rng, "2wiki", 260, False, out / "cache"),
                "squad": TOY._dataset(rng, "squad", 240, False, out / "cache")}
    for name in PILOT:
        for kind in ("fit", "select"):
            _extend_meta(out / "cache" / name / kind / "meta.json", kind, hashlib.sha256(f"{name}:{kind}".encode()).hexdigest())
    bank, ctx2 = M.build_relation_bank(contexts)
    cfg, cfg_m3b, cfg_h = R.load_configs(config=config)
    yield SimpleNamespace(base=base, out=out, m3b_out=m3b_out, config=config, cfg=cfg, cfg_m3b=cfg_m3b, cfg_h=cfg_h, contexts=contexts, ctx2=ctx2,
                          bank=bank, rng=rng)
    mp.undo()


def _config_text(sb) -> str:
    return sb.config.read_text(encoding="utf-8")


def _status_line(sb) -> str:
    return [l for l in _config_text(sb).splitlines() if l.startswith("status: ")][0]


# ── the declaration, the pins, the rule ──────────────────────────────────────


def test_pins_verified_and_a_changed_pin_refused(sandbox):
    pins = R.verify_pins(sandbox.cfg)
    assert len(pins) == 7
    cfg = copy.deepcopy(sandbox.cfg)
    block = cfg["m3b_incumbents"]["pinned_files_sha256_lf_normalised"]
    first = sorted(block)[0]
    block[first] = "0" * 64
    with pytest.raises(SystemExit, match="M3B pin mismatch"):
        R.verify_pins(cfg)


def test_m3b_core_from_the_committed_block_and_its_sha(sandbox):
    core, pairs, sha = R.m3b_core(sandbox.cfg_m3b)
    assert len(core) == 78 and sha == hashlib.sha256(",".join(core).encode("utf-8")).hexdigest()
    assert all({"earlier", "later"} <= set(p) for p in pairs)
    broken = copy.deepcopy(sandbox.cfg_m3b)
    broken[R.M3B_CORE_BLOCK]["surviving"] = core[:77]
    with pytest.raises(SystemExit):
        R.m3b_core(broken)


def test_dated_keys_and_append_block(sandbox, tmp_path):
    cfg = {"status": "DECLARED_NOT_RUN"}
    config = tmp_path / "c.yaml"
    config.write_text("status: DECLARED_NOT_RUN" + LF, encoding="utf-8")
    with pytest.raises(SystemExit, match="not a dated block key"):
        R.append_block(cfg, "contract_frozen_v1", {"a": 1}, "x", config)
    with pytest.raises(SystemExit, match="not a dated block key"):
        R.append_block(cfg, "timing_2026_9_20", {"a": 1}, "x", config)
    R.append_block(cfg, "timing_2026_09_20", {"a": 1, "b": [1, 2]}, "the timing", config)
    assert yaml.safe_load(config.read_text(encoding="utf-8"))["timing_2026_09_20"] == {"a": 1, "b": [1, 2]}
    assert R.dated_blocks(cfg, "timing") == ["timing_2026_09_20"]
    with pytest.raises(SystemExit, match="already filed"):
        R.append_block(cfg, "timing_2026_09_21", {"a": 2}, "again", config)
    assert R.DATED.match("authorization_stage_2_2026_10_01") and R.DATED.match("amendment_12_2026_10_01")
    assert not R.DATED.match("note_1_2026_10_01") and not R.DATED.match("amendment_2026_10_01")


def test_set_status_moves_once_and_refuses_the_wrong_origin(tmp_path):
    config = tmp_path / "c.yaml"
    config.write_bytes(("experiment: x" + chr(13) + LF + "status: DECLARED_NOT_RUN" + chr(13) + LF + "other:" + chr(13) + LF + "  status: not top level" + chr(13) + LF).encode("utf-8"))
    with pytest.raises(SystemExit, match="does not move"):
        R.set_status(config, "RUN", ("PILOT_GATE_READ",))
    assert R.set_status(config, "PILOT_GATE_READ", ("DECLARED_NOT_RUN",)) == "DECLARED_NOT_RUN"
    text = config.read_bytes()
    assert (b"status: PILOT_GATE_READ" + (chr(13) + LF).encode()) in text   # CRLF kept as found
    assert text.count(b"status: ") == 2 and b"  status: not top level" in text
    config.write_bytes(("status: A" + LF + "status: B" + LF).encode("utf-8"))
    with pytest.raises(SystemExit, match="exactly one"):
        R.set_status(config, "RUN", ("A",))


def test_training_rule_is_m3b_with_a_dated_override_only(sandbox):
    rule = R.training_rule_v2(sandbox.cfg, sandbox.cfg_m3b)
    assert rule["max_epochs"] == 6 and rule["batches_per_epoch"] == 2000 and rule["batch_size"] == 16 and rule["patience"] == 2
    assert rule["dataset_draw"] == "per_query" and rule["epoch_limit_s"] == 28800.0 and rule["override_block"] is None
    cfg = copy.deepcopy(sandbox.cfg)
    cfg["amendment_2_2026_09_21"] = {"training_override": {"batches_per_epoch": 1000, "max_epochs": 12, "patience": 4, "lr": 0.01}}
    over = R.training_rule_v2(cfg, sandbox.cfg_m3b)
    assert (over["batches_per_epoch"], over["max_epochs"], over["patience"]) == (1000, 12, 4)
    assert over["lr"] == 1e-3 and over["override_block"] == "amendment_2_2026_09_21"


def test_gate_thresholds_are_the_amended_cells(sandbox):
    gnn, twin, source = R.gate_thresholds(sandbox.cfg)
    assert source == "amendment_1_2026_09_19.pilot_gate_amended"
    cells = R.gate_cells(gnn, twin)
    assert [c[:3] for c in cells["gnn"]] == [("metaqa", "hit@1", "all"), ("metaqa", "hit@1", "3hop"), ("2wiki", "recall@5", "all"),
                                            ("2wiki", "full_coverage@5", "all"), ("squad", "recall@5", "all")]
    assert [c[4] for c in cells["gnn"]] == ["gat_universal_v1", "gat_universal_v1", None, None, None]
    assert [c[4] for c in cells["twin"]] == ["qls_u_sota_v1", None, None]
    assert R.declared_half_counts(sandbox.cfg, "squad") == {"V2_GATE": 5841, "V2_HELD_CONFIRMATION": 6032}


def test_disk_guard_reads_the_declared_floor_and_bound(sandbox):
    g = R.DiskGuardV2(sandbox.cfg)
    assert g.halt_below == 8e9 and g.bound == 12e9
    cfg = copy.deepcopy(sandbox.cfg)
    cfg["compute"]["abort_criteria"] = ["nothing about disk"]
    with pytest.raises(SystemExit):
        R.DiskGuardV2(cfg)


def test_paired_bootstrap_is_the_declared_procedure():
    a = np.asarray([1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    b = np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0])
    d = R.paired_bootstrap(a, b)
    assert d["mean"] == 0.375 and d["low"] <= d["mean"] <= d["high"] and d["resamples"] == 1000 and d["seed"] == 0 and d["n"] == 8
    assert R.paired_bootstrap(a, b) == d   # default_rng(0): reproducible


# ── compile: the diagnostics over every query ────────────────────────────────


def _toy_prep(rng, name: str, n: int, typed: bool, count: int):
    """A Prepared-shaped namespace over a toy substrate: metaqa-style ids carry the hop, untyped datasets do not."""
    stores = {"structural": TOY.P.FamilyStore.from_graph(TOY._graph(rng, n, 3 * n, False, typed, n_rel=6 if typed else 1), "structural"),
              "ner": TOY.P.FamilyStore.from_graph(TOY._graph(rng, n, n, True, False), "ner"),
              "knn": TOY.P.FamilyStore.from_graph(TOY._graph(rng, n, 2 * n, True, False), "knn")}
    emb = TOY._unit(rng, (n, V.M3B.DIM))
    rel = V.RelationTable.from_arrays(TOY._unit(rng, (6, V.M3B.DIM)), stores["structural"].rel_count, n) if typed else None
    ids, qemb, dense_ids, dense_scores, splade_ids, splade_scores, seeds, pools, golds = [], [], [], [], [], [], [], [], []
    for i in range(count):
        q = TOY._unit(rng, V.M3B.DIM)
        d = np.argsort(-(emb @ q))[:40].astype(np.int64)
        sp = rng.permutation(n)[:40].astype(np.int64)
        sd = TOY.P.seeds_of(d, sp)
        pool = np.union1d(np.union1d(d[:20], sp[:20]), TOY.P.expand_hops(sd, list(stores.values()), {"hops": 1, "per_seed_cap": 6}))
        pool = np.union1d(pool, sd)
        ids.append(f"metaqa:{1 + i % 3}hop:train:{i}" if name == "metaqa" else f"{name}-q{i}")
        qemb.append(q); dense_ids.append(d); dense_scores.append(np.sort(emb[d] @ q)[::-1].astype(np.float32))
        splade_ids.append(sp); splade_scores.append(np.sort(rng.random(40) * 10)[::-1].astype(np.float32))
        seeds.append(sd); pools.append(pool); golds.append(np.sort(d[[0, 2, 5]]))
    pop = SimpleNamespace(dataset=name, kind="fit", ids=ids, idx=np.arange(count), golds=golds, n_before=count, zero_gold_excluded=0, digest="toy")
    prep = SimpleNamespace(pop=pop, qemb=np.stack(qemb), dense_ids=np.stack(dense_ids), dense_scores=np.stack(dense_scores), splade_ids=np.stack(splade_ids),
                           splade_scores=np.stack(splade_scores), seeds=seeds, pools=pools, seeds_added=np.zeros(count), expansion_seconds=0.0)
    return prep, stores, TOY._Nodes(emb), rel


def test_compile_population_v2_records_the_diagnostics_over_every_query_by_hop(tmp_path):
    """amendment 2 k_rel_truncation_report_required: the truncation and the ordered-path availability are read
    from every query (no sampling), sliced by the hop in the metaqa id, and agree with a direct recount."""
    rng = np.random.default_rng(3)
    m3b_compile = R.M3B_RUN.load_script("m3b_compile")
    prep, stores, nodes, rel = _toy_prep(rng, "metaqa", 240, True, 30)
    meta = {"dataset": "metaqa", "kind": "fit", "feature_contract": V.CONTRACT_NAME, "n_columns": V.N_COLUMNS}
    written = R.compile_population_v2(prep, stores, nodes, rel, tmp_path / "metaqa" / "fit", meta, m3b_compile.gold_local_of, log=_quiet)
    d = written["diagnostics"]
    assert d["k_rel"] == M.K_REL == 4 and set(d["by_hop"]) == {"1hop", "2hop", "3hop", "all"}
    assert sum(d["by_hop"][h]["queries"] for h in ("1hop", "2hop", "3hop")) == d["by_hop"]["all"]["queries"] == 30
    assert written["relation_slots"]["queries"] == 30 and written["relation_slots"]["pairs"] == d["by_hop"]["all"]["relation_slots"]["pairs"]
    assert written["peak_rss_bytes"] > 0 and written["queries_per_second"] > 0 and written["compile_seconds"] >= 0
    # the direct recount: typed_pool_edges pair ids per query, the cache scalars for the availability
    pairs = truncated = 0
    for i in range(30):
        (_, _, _, _), pair_id, _ = V.typed_pool_edges(stores["structural"], np.asarray(prep.pools[i], dtype=np.int64), V.IN_POOL_CAP)
        st = M.relation_slot_stats(pair_id)
        pairs += st["pairs"]
        truncated += st["truncated"]
    allh = d["by_hop"]["all"]["relation_slots"]
    assert allh["pairs"] == pairs and allh["pairs_truncated"] == truncated and allh["fraction_of_pairs_truncated"] == round(truncated / max(pairs, 1), 6)
    assert sum(allh["relations_per_pair_histogram"].values()) == pairs
    assert sum(int(c) for k, c in allh["relations_per_pair_histogram"].items() if int(k) > 4) == truncated
    scalars = np.load(tmp_path / "metaqa" / "fit" / "scalars.npy", mmap_mode="r")
    ptr = np.load(tmp_path / "metaqa" / "fit" / "pool_ptr.npy")
    walk2 = np.asarray(scalars[:, V.IDX["typed_walks_h2"]]) > 0
    assert d["by_hop"]["all"]["rows"] == scalars.shape[0] and d["by_hop"]["all"]["typed_walk_availability"]["h2"]["rows"] == round(float(walk2.mean()), 6)
    per_query = [walk2[ptr[i]:ptr[i + 1]].any() for i in range(30)]
    assert d["by_hop"]["all"]["typed_walk_availability"]["h2"]["queries"] == round(sum(per_query) / 30, 6)
    inv = (np.asarray(scalars[:, [V.IDX["opath_h2_dir1"], V.IDX["opath_h2_dir2"]]]) < 0).any(axis=1)
    assert d["by_hop"]["all"]["ordered_path"]["h2"]["fraction_of_walks_with_an_inverse_step"] == round(float(inv.sum() / max(walk2.sum(), 1)), 6)
    hop1 = [i for i in range(30) if i % 3 == 0]
    assert d["by_hop"]["1hop"]["queries"] == len(hop1) and d["by_hop"]["1hop"]["rows"] == sum(int(ptr[i + 1] - ptr[i]) for i in hop1)
    # an untyped dataset: one slice, no typed query, every ordered column absent
    prep_u, stores_u, nodes_u, _ = _toy_prep(rng, "squad", 200, False, 12)
    written_u = R.compile_population_v2(prep_u, stores_u, nodes_u, None, tmp_path / "squad" / "fit", {**meta, "dataset": "squad"}, m3b_compile.gold_local_of, log=_quiet)
    du = written_u["diagnostics"]["by_hop"]
    assert set(du) == {"all"} and du["all"]["typed_queries"] == 0 and du["all"]["relation_slots"]["pairs"] == 0
    assert du["all"]["typed_walk_availability"]["h2"]["rows"] == 0.0 and du["all"]["ordered_path"]["h3"]["rows_with_walk"] == 0
    assert written_u["relation_slots"]["fraction_of_pairs_truncated"] == 0.0


# ── the pipeline in declared order (the tests below share S and run in file order) ──


def test_10_every_later_stage_refuses_before_contract_frozen(sandbox):
    with pytest.raises(SystemExit, match="contract_frozen"):
        R.model_inputs(sandbox.cfg, sandbox.cfg_m3b)
    with pytest.raises(SystemExit, match="contract_frozen"):
        R.frozen_contract_v2(sandbox.cfg)


def test_11_screen_refuses_a_select_cache_a_trimmed_cache_and_a_foreign_dataset(sandbox):
    fit_meta = sandbox.out / "cache" / "metaqa" / "fit" / "meta.json"
    meta = json.loads(fit_meta.read_text(encoding="utf-8"))
    swapped = {**meta, "kind": "select"}
    fit_meta.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(SystemExit, match="refuses any other cache"):
        R.screen_cache_dir("metaqa")
    fit_meta.write_text(json.dumps({**meta, "columns_stored": [0, 1, 2]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="trimmed"):
        R.screen_cache_dir("metaqa")
    fit_meta.write_text(json.dumps(meta), encoding="utf-8")
    assert R.screen_cache_dir("metaqa") == sandbox.out / "cache" / "metaqa" / "fit"
    with pytest.raises(SystemExit, match="three pilot fit carves"):
        R.stage_screen(sandbox.cfg, sandbox.cfg_m3b, ["metaqa", "2wiki", "hotpotqa"], log=_quiet)
    with pytest.raises(SystemExit, match="three pilot fit carves"):
        R.stage_screen(sandbox.cfg, sandbox.cfg_m3b, ["metaqa"], log=_quiet)


def test_12_screen_reads_fit_caches_only_and_protects_the_m3b_78(sandbox):
    screen = R.stage_screen(sandbox.cfg, sandbox.cfg_m3b, PILOT, log=_quiet)
    core78, _, sha78 = R.m3b_core(sandbox.cfg_m3b)
    assert len(screen["columns"]) == 164 and screen["columns"][:78] == core78 and screen["columns"][78:] == list(V.V2_COLUMNS)
    assert screen["raw_contract_pinned_in"] == "amendment_2_2026_09_19" and screen["raw_contract_sha256"] == R.sha_of_names(screen["columns"])
    assert screen["screen_rule_sha256"] == hashlib.sha256(screen["rule"].encode("utf-8")).hexdigest() and "|Spearman| >= 0.98" in screen["rule"]
    assert screen["v2_columns"]["screened"] == 86 and screen["v2_columns"]["depth_basis"] == 73 and screen["v2_columns"]["ordered_relation_path"] == 13
    assert screen["v2_columns"]["surviving"] == screen["v2_columns"]["depth_basis_surviving"] + screen["v2_columns"]["ordered_relation_path_surviving"]
    for name in PILOT:
        per = screen["per_dataset"][name]
        assert set(per["block_availability"]) == set(V.BLOCKS) and all(set(v) <= set(V.MASK_COLUMNS) for v in per["block_availability"].values())
        assert set(per["constant_columns"]) <= set(screen["columns"])
    assert set(screen["constant_columns_everywhere"]) <= set(screen["dropped"]) | set(core78)
    assert screen["surviving"][:78] == core78 and screen["m3b_core"] == {"size": 78, "sha256": sha78, "protected": True,
                                                                          "would_have_dropped_under_the_trio_statistics": screen["m3b_core"]["would_have_dropped_under_the_trio_statistics"]}
    assert not any(c in screen["dropped"] for c in core78)
    assert screen["read"].startswith("outputs/universal_v2/cache/<dataset>/fit only")
    assert set(screen["per_dataset"]) == set(PILOT)
    block = yaml.safe_load((sandbox.out / "universal_v2_core_contract_block.yaml").read_text(encoding="utf-8"))
    assert block["name"] == "UNIVERSAL_V2_CORE_CONTRACT" and block["surviving"] == screen["surviving"]
    assert block["sha256_of_comma_joined_surviving_names"] == hashlib.sha256(",".join(block["surviving"]).encode("utf-8")).hexdigest()
    assert block["screened_columns"] == 164 and block["surviving_columns"] == len(block["surviving"]) and block["m3b_core_first"]["always_survive"]
    assert block["v2_columns_screened"] == 86 and block["v2_columns_surviving"] == block["depth_basis_surviving"] + block["ordered_relation_path_surviving"]
    assert block["raw_contract_sha256"] == screen["raw_contract_sha256"] and block["screen_rule_sha256"] == screen["screen_rule_sha256"]
    assert [c for c in screen["columns"] if c not in block["surviving"]] == list(block["dropped"])
    S.screen = screen


def test_13_file_contract_freezes_the_block_and_the_screen_is_not_repeated(sandbox):
    with pytest.raises(SystemExit, match="YYYY_MM_DD"):
        R.stage_file(sandbox.cfg, "2026-09-20", ["contract"], config=sandbox.config, log=_quiet)
    with pytest.raises(SystemExit, match="--which"):
        R.stage_file(sandbox.cfg, DATE, ["screen"], config=sandbox.config, log=_quiet)
    (sandbox.out / "feature_contract.json").write_text(json.dumps(V.contract_json_v2(R.m3b_core(sandbox.cfg_m3b)[0])), encoding="utf-8")
    R.stage_file(sandbox.cfg, DATE, ["contract"], config=sandbox.config, log=_quiet)
    key, frozen = R.frozen_contract_v2(sandbox.cfg)
    assert key == f"contract_frozen_{DATE}" and frozen["status"] == "FROZEN" and frozen["surviving"] == S.screen["surviving"]
    assert frozen["screen_file_sha256"] == R.sha256_file(sandbox.out / "feature_screen.json")
    assert set(frozen["compile_record"]) == set(PILOT) and frozen["compile_record"]["metaqa"]["fit"]["relation_slots"]["k_rel"] == 4
    rec = frozen["compile_record"]["metaqa"]["fit"]
    assert set(rec["diagnostics"]["by_hop"]) == {"1hop", "2hop", "3hop", "all"} and rec["peak_rss_bytes"] == 1 << 30 and rec["queries_per_second"] == 80.0
    assert set(rec["cache_files_sha256"]) == {p.name for p in (sandbox.out / "cache" / "metaqa" / "fit").iterdir() if p.name != "meta.json"}
    assert rec["cache_files_sha256"]["scalars.npy"]["sha256"] == R.sha256_file(sandbox.out / "cache" / "metaqa" / "fit" / "scalars.npy")
    assert rec["cache_combined_sha256"] == R.cache_hashes(sandbox.out / "cache" / "metaqa" / "fit")["combined_sha256"]
    assert frozen["hashes"] == {"raw_contract_sha256": frozen["raw_contract_sha256"], "screen_rule_sha256": frozen["screen_rule_sha256"],
                                "surviving_columns_sha256": frozen["sha256_of_comma_joined_surviving_names"],
                                "six_caches_combined_sha256": frozen["hashes"]["six_caches_combined_sha256"]}
    assert len(frozen["hashes"]["six_caches_combined_sha256"]) == 64
    assert yaml.safe_load(_config_text(sandbox))[key] == frozen     # read back from the appended file
    assert _status_line(sandbox) == "status: DECLARED_NOT_RUN"
    with pytest.raises(SystemExit, match="not repeated"):
        R.stage_screen(sandbox.cfg, sandbox.cfg_m3b, PILOT, log=_quiet)
    with pytest.raises(SystemExit, match="already filed"):
        R.stage_file(sandbox.cfg, "2026_09_21", ["contract"], config=sandbox.config, log=_quiet)


def test_14_model_inputs_read_the_frozen_block_and_refuse_a_wrong_core(sandbox):
    inputs = R.model_inputs(sandbox.cfg, sandbox.cfg_m3b)
    core78, pairs, sha78 = R.m3b_core(sandbox.cfg_m3b)
    assert inputs["columns"][:78] == core78 and inputs["core78_columns"] == core78 and inputs["core78_sha256"] == sha78
    assert inputs["n_scalars"] == len(S.screen["surviving"]) and inputs["base"] == "rrf" and inputs["columns"][inputs["base_local"]] == "rrf"
    assert inputs["evidence_substitutions"] == {"splade_score_norm": "splade_rr"} or inputs["evidence_substitutions"] == {}
    assert inputs["core78_positions"] == list(range(78)) and inputs["column_indices"].tolist()[:78] == inputs["core78_indices"].tolist()
    broken = copy.deepcopy(sandbox.cfg)
    key = f"contract_frozen_{DATE}"
    broken[key]["surviving"] = [broken[key]["surviving"][1], broken[key]["surviving"][0]] + broken[key]["surviving"][2:]
    broken[key]["sha256_of_comma_joined_surviving_names"] = hashlib.sha256(",".join(broken[key]["surviving"]).encode("utf-8")).hexdigest()
    with pytest.raises(SystemExit, match="M3B core in its order"):
        R.model_inputs(broken, sandbox.cfg_m3b)
    broken[key]["surviving"] = list(sandbox.cfg[key]["surviving"]) + ["no_such_column"]
    broken[key]["sha256_of_comma_joined_surviving_names"] = hashlib.sha256(",".join(broken[key]["surviving"]).encode("utf-8")).hexdigest()
    with pytest.raises(SystemExit, match="outside the v2 layout"):
        R.model_inputs(broken, sandbox.cfg_m3b)
    S.inputs = inputs
    S.carves = R.open_carves_v2(sandbox.ctx2, inputs)
    assert set(S.carves) == {"v2", "core78", "control"} and set(S.carves["v2"]["fit"]) == set(PILOT)
    S.training = {**R.training_rule_v2(sandbox.cfg, sandbox.cfg_m3b), "max_epochs": 1, "batches_per_epoch": 2, "batch_size": 4}


def test_15_timing_run_precedes_every_fit_and_keeps_no_select_number(sandbox):
    lines = []
    timing = R.stage_timing(sandbox.cfg, S.inputs, S.carves, sandbox.bank, S.training, log=lines.append)
    assert set(timing["arms"]) == set(M.ARMS) and timing["arms"]["u_gnn_v2_core78"]["architecture"] == "u_gnn_v2_ef"
    assert timing["full_epoch_u_gnn_v2_ef"]["weights"] == "discarded" and timing["full_epoch_u_gnn_v2_ef"]["steps"] == 2
    assert timing["fallback"]["fires"] is False and "3 hours" in timing["fallback"]["rule"]
    assert not any("select macro" in l for l in lines) and any("select number not kept" in l for l in lines)
    assert not (sandbox.out / "fits").exists() or not any((sandbox.out / "fits").glob("*.pt"))
    assert R.stage_timing(sandbox.cfg, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet) == timing   # exists, not repeated
    R.stage_file(sandbox.cfg, DATE, ["timing"], config=sandbox.config, log=_quiet)
    assert sandbox.cfg[f"timing_{DATE}"]["status"] == "MEASURED_BEFORE_ANY_FIT" and sandbox.cfg[f"timing_{DATE}"]["output_sha256"] == R.sha256_file(sandbox.out / "timing.json")
    S.timing = timing


def test_16_seed0_fits_and_their_refusals(sandbox):
    with pytest.raises(SystemExit, match="not an arm"):
        R.run_fit("gat_v3", 0, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    with pytest.raises(SystemExit, match="not a declared seed"):
        R.run_fit("u_gnn_v2", 3, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    with pytest.raises(SystemExit, match="selection.json is written first"):
        R.run_fit("u_gnn_v2_core78", 0, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    with pytest.raises(SystemExit, match="passed its gate"):
        R.run_fit("u_gnn_v2", 1, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    with pytest.raises(SystemExit, match="no seed-0 fit record"):
        R.stage_select(sandbox.cfg, S.inputs, log=_quiet)
    S.fits = {}
    for arm in list(M.GNN_CANDIDATES) + list(M.TWIN_CANDIDATES) + ["gat_universal_v1_trio"]:
        rec = R.run_fit(arm, 0, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
        assert rec["key"] == f"{arm}__H128__s0" and rec["parameters"] <= rec["parameter_budget"] and rec["epochs_run"] == 1
        assert (sandbox.out / "fits" / f"{rec['key']}.pt").exists() and not (sandbox.out / "fits" / f"{rec['key']}.ckpt").exists()
        assert rec["state_sha256"] == R.sha256_file(sandbox.out / "fits" / f"{rec['key']}.pt")
        assert rec["relation_bank"]["rows"] == sandbox.bank.n_rows and rec["selected_gnn_architecture"] is None
        assert rec["columns"] == (78 if arm == "gat_universal_v1_trio" else S.inputs["n_scalars"])
        assert (rec["evidence"] is not None) == (arm == "u_gnn_v2_ef")
        S.fits[arm] = rec
    again = R.run_fit("u_gnn_v2", 0, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    assert again == S.fits["u_gnn_v2"]     # exists, not repeated
    (sandbox.out / "timing.json").rename(sandbox.out / "timing.json.moved")
    with pytest.raises(SystemExit, match="timing run precedes"):
        R.stage_timing(sandbox.cfg, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    (sandbox.out / "timing.json.moved").rename(sandbox.out / "timing.json")


def test_17_selection_written_once_from_select_carves_only(sandbox):
    sel = R.stage_select(sandbox.cfg, S.inputs, log=_quiet)
    for family, candidates in (("gnn", M.GNN_CANDIDATES), ("twin", M.TWIN_CANDIDATES)):
        best = max(candidates, key=lambda a: (S.fits[a]["best_select_macro_recall5"], -S.fits[a]["parameters"]))
        assert sel[family]["arm"] == best and set(sel[family]["candidates"]) == set(candidates)
    assert sel["eval_populations_scored_before_this_file"] is False and "gat_universal_v1_trio" not in json.dumps(sel["gnn"])
    assert not any(k in json.dumps(sel) for k in R.M3B_REFERENCES.values())
    with pytest.raises(SystemExit, match="written once"):
        R.stage_select(sandbox.cfg, S.inputs, log=_quiet)
    S.selection = sel
    rec = R.run_fit("u_gnn_v2_core78", 0, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    assert rec["selected_gnn_architecture"] == sel["gnn"]["arm"] and rec["columns"] == 78 and rec["core_sha256"] == S.inputs["core78_sha256"]
    S.fits["u_gnn_v2_core78"] = rec
    (sandbox.out / "selection.json").rename(sandbox.out / "selection.json.moved")
    with pytest.raises(SystemExit, match="no selection.json"):
        R.stage_gate(sandbox.cfg, S.inputs, log=_quiet)
    (sandbox.out / "selection.json.moved").rename(sandbox.out / "selection.json")


# ── the synthetic eval world: per-query metrics from rank_metrics, both halves labelled ──

N_EVAL = {"metaqa": 480, "2wiki": 360, "squad": 360}
POOL = 20


def _eval_ids(name: str) -> list[str]:
    n = N_EVAL[name]
    return [f"metaqa:{1 + i % 3}hop:dev:{i}" if name == "metaqa" else f"{name}-e{i}" for i in range(n)]


def _scorer_arrays(rng, n: int, two_gold: np.ndarray, p_top: float, p_5: float, p_cov: float = 0.9) -> dict:
    """One scorer's metric arrays: the first gold's rank is 1 with probability p_top, within 2-5 with p_5 - p_top,
    beyond 5 otherwise; a second gold (two_gold rows) lands within the top 5 with probability p_cov."""
    rows = []
    for i in range(n):
        order = rng.permutation(POOL)
        scores = np.zeros(POOL)
        scores[order] = np.arange(POOL, 0, -1, dtype=np.float64)
        u = rng.random()
        r = 1 if u < p_top else int(rng.integers(2, 6)) if u < p_5 else int(rng.integers(6, POOL + 1))
        golds = [order[r - 1]]
        if two_gold[i]:
            r2 = int(rng.integers(1, 6)) if rng.random() < p_cov else int(rng.integers(6, POOL + 1))
            while r2 == r:
                r2 = int(rng.integers(1, POOL + 1))
            golds.append(order[r2 - 1])
        rows.append(T.rank_metrics(scores, np.asarray(sorted(golds)), len(golds)))
    return {m: np.asarray([row[m] for row in rows], dtype=np.float64) for m in T.METRIC_NAMES}


def _mechanism_arrays(rng, key: str, n: int) -> dict:
    out = {"delta_ratio": rng.random(n) * 0.5, "top1_changed": (rng.random(n) < 0.3).astype(np.float64)}
    arm = key.split("__")[0]
    if arm in ("u_gnn_v2", "u_gnn_v2_ef", "u_gnn_v2_core78"):
        for t in (1, 2, 3):
            out[f"gate_step{t}"] = rng.random(n)
    if arm == "u_gnn_v2_ef":
        for t in (1, 2, 3):
            out[f"gate2_step{t}"] = rng.random(n) * 0.5
    if arm == "u_mlp_v2_mix":
        for b in range(4):
            out[f"block_gate{b}"] = rng.random(n)
    return out


def _biases(name: str, key: str, selection: dict) -> tuple[float, float]:
    """The gate outcome of the synthetic world: the selected GNN passes every cell, the selected twin fails metaqa."""
    arm = key.split("__")[0]
    if key == R.M3B_REFERENCES["gat_universal_v1"]:
        return (0.80, 0.90) if name == "metaqa" else (0.75, 0.90)
    if key in (R.M3B_REFERENCES["gat_no_mp_v1"], R.M3B_REFERENCES["qls_u_sota_v1"]):
        return (0.60, 0.85)
    if key.startswith("fixed:"):
        return (0.40, 0.80) if key != "fixed:rrf" or name != "squad" else (0.70, 0.92)
    if arm == selection["gnn"]["arm"]:
        return (0.97, 0.99)
    if arm == selection["twin"]["arm"]:
        return (0.50, 0.95)
    return (0.70, 0.90)


def _write_population(sb, name: str, keys: list[str], stub_records: bool = False) -> dict:
    """The eval record of one population through the run script's own write_eval_record, on synthetic per-query
    metrics, with the frozen M3B arrays of the same queries beside it under outputs/m3b/eval/."""
    rng = np.random.default_rng({"metaqa": 11, "2wiki": 12, "squad": 13}[name])
    n = N_EVAL[name]
    ids = _eval_ids(name)
    two_gold = rng.random(n) < 0.25
    half = np.asarray([i % 2 == 0 for i in range(n)])
    scorers = list(keys) + [f"fixed:{c}" for c in R.FIXED_SCORERS]
    arrays = {}
    for s in scorers:
        p_top, p_5 = _biases(name, s, S.selection)
        for m, a in _scorer_arrays(rng, n, two_gold, p_top, p_5).items():
            arrays[f"{s}/{m}"] = a
        if not s.startswith("fixed:"):
            for m, a in _mechanism_arrays(rng, s, n).items():
                arrays[f"{s}/{m}"] = a
    arrays["gold_dist_struct"] = rng.integers(0, 5, size=n).astype(np.int64)
    arrays["hop"] = np.asarray([1 + i % 3 for i in range(n)], dtype=np.int64) if name == "metaqa" else np.zeros(n, dtype=np.int64)
    arrays["pool_size"] = np.full(n, POOL, dtype=np.int64)
    arrays["half"] = half
    m3b = {"pool_size": arrays["pool_size"]}
    for label, key in R.M3B_REFERENCES.items():
        if key == "fixed:rrf":
            for m in T.METRIC_NAMES:
                m3b[f"{key}/{m}"] = arrays[f"fixed:rrf/{m}"]
        else:
            p_top, p_5 = _biases(name, key, S.selection)
            for m, a in _scorer_arrays(rng, n, two_gold, p_top, p_5).items():
                m3b[f"{key}/{m}"] = a
    np.savez_compressed(sb.m3b_out / "eval" / f"{name}.npz", **m3b)
    (sb.m3b_out / "eval" / f"{name}_query_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    return {"ids": ids, "arrays": arrays, "scorers": scorers, "half": half}


def _record(sb, name: str, world: dict, shard=None, file_base=None, supplement=None, keys=None) -> dict:
    ids, arrays, scorers = world["ids"], world["arrays"], world["scorers"]
    n_full = len(ids)
    if keys is not None:
        scorers = list(keys) + [s for s in scorers if s.startswith("fixed:")]
        arrays = {k: v for k, v in arrays.items() if "/" not in k or k.split("/")[0] in scorers}
    if shard is not None:
        k, N = shard
        ids = ids[k::N]
        arrays = {a: v[k::N] for a, v in arrays.items()}
    n = len(ids)
    pop = SimpleNamespace(idx=np.arange(n), zero_gold_excluded=0, digest=R.m3b_pools.ids_digest(ids), ids=ids)
    first = scorers[0]
    ceiling = R.M3B_RUN.ceiling_from_arrays(arrays[f"{first}/gold_in_pool"], arrays[f"{first}/gold_total"], arrays["pool_size"])
    latency = {"compile": [0.01, 0.02], "pack": [0.001, 0.002], **{s: [0.003, 0.004] for s in scorers if not s.startswith("fixed:")}}
    return R.write_eval_record(name, arrays, scorers, pop, SimpleNamespace(seeds_added=np.zeros(n)), ceiling, latency, shard, n_full,
                               R.m3b_pools.ids_digest(world["ids"]), "candidate_contract_frozen_2026_09_13", {"per_dataset": {name: {"pool": "toy"}}}, {},
                               10, time.time() - 1.0, time.time() - 1.0, {"RECORD_SHA256": "f" * 64}, S.inputs, log=_quiet, file_base=file_base, supplement=supplement)


def test_18_eval_records_written_through_write_eval_record(sandbox):
    keys = sorted(S.fits[a]["key"] for a in S.fits)
    S.world, S.records = {}, {}
    for name in PILOT:
        S.world[name] = _write_population(sandbox, name, keys)
        rec = _record(sandbox, name, S.world[name])
        assert rec["halves"]["V2_GATE"] == int(S.world[name]["half"].sum()) and rec["summary_half"].startswith("V2_GATE only")
        assert all(a["ok"] for a in rec["mrr_audit"].values()) and rec["m3b_fixed_rrf_agreement"]["ok"]
        assert set(rec["summary_V2_GATE"]) == set(S.world[name]["scorers"])
        gate_rows = S.world[name]["half"]
        for s in S.world[name]["scorers"]:
            assert rec["summary_V2_GATE"][s]["recall@5"] == round(float(S.world[name]["arrays"][f"{s}/recall@5"][gate_rows].mean()), 4)
        assert (sandbox.out / "eval" / f"{name}.npz").exists() and json.loads((sandbox.out / "eval" / f"{name}_query_ids.json").read_text()) == S.world[name]["ids"]
        S.records[name] = rec
    with pytest.raises(SystemExit, match="disagrees with the frozen M3B arrays"):
        bad = {m: S.world["squad"]["arrays"][f"fixed:rrf/{m}"].copy() for m in ("recall@5", "hit@1", "gold_in_pool", "gold_total", "pool_size")}
        bad["recall@5"][3] += 0.5
        R.m3b_fixed_rrf_agreement("squad", bad, None, N_EVAL["squad"])
    with pytest.raises(SystemExit, match="MRR audit failed"):
        arrays = {k: v.copy() for k, v in S.world["squad"]["arrays"].items()}
        arrays["fixed:rrf/mrr"][0] = 0.123
        R.audit_scorers(arrays, ["fixed:rrf"], "squad")
    with pytest.raises(SystemExit, match="an eval record exists"):
        (sandbox.out / "selection.json").rename(sandbox.out / "selection.json.moved")
        try:
            R.stage_select(sandbox.cfg, S.inputs, log=_quiet)
        finally:
            (sandbox.out / "selection.json.moved").rename(sandbox.out / "selection.json")


def test_19_merge_shards_reproduces_the_unsharded_arrays(sandbox):
    world = S.world["squad"]
    for k in (0, 1):
        _record(sandbox, "squad", world, shard=(k, 2), file_base="squadshards")
    assert R.merge_shards("squadshards", log=_quiet) is not None
    merged = R.read_json(sandbox.out / "eval" / "squadshards.json")
    assert merged["dataset"] == "squad" and merged["shard"] is None and merged["queries"] == N_EVAL["squad"] and merged["ids_sha256"] == S.records["squad"]["ids_sha256"]
    assert merged["summary_V2_GATE"] == S.records["squad"]["summary_V2_GATE"] and merged["halves"]["V2_GATE"] == S.records["squad"]["halves"]["V2_GATE"]
    assert abs(merged["ceiling_as_compiled"]["recall_ceiling@5"] - S.records["squad"]["ceiling_as_compiled"]["recall_ceiling@5"]) < 1e-9
    with np.load(sandbox.out / "eval" / "squadshards.npz") as z, np.load(sandbox.out / "eval" / "squad.npz") as ref:
        assert set(z.files) == set(ref.files) and all(np.array_equal(z[a], ref[a]) for a in ref.files)
    assert json.loads((sandbox.out / "eval" / "squadshards_query_ids.json").read_text()) == world["ids"]
    for p in sandbox.out.glob("eval/squadshards*"):
        p.unlink()


def _nan_the_held_rows(sb, stem: str, dataset: str) -> None:
    """Every scorer array of a record with its V2_HELD_CONFIRMATION rows set to NaN: a stage that reads a held
    row now produces a NaN (or refuses on it)."""
    path = sb.out / "eval" / f"{stem}.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    held = ~arrays["half"].astype(bool)
    for k, v in arrays.items():
        if "/" in k and v.dtype.kind == "f":
            v = v.copy()
            v[held] = np.nan
            arrays[k] = v
    np.savez_compressed(path, **arrays)


def _restore_finite(sb, stem: str, world: dict, keys=None) -> None:
    path = sb.out / "eval" / f"{stem}.npz"
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    for k in arrays:
        if "/" in k:
            arrays[k] = world["arrays"][k]
    np.savez_compressed(path, **arrays)


def test_20_gate_reads_v2_gate_only_once_for_the_selected_arms(sandbox):
    for name in PILOT:
        _nan_the_held_rows(sandbox, name, name)
    gate = R.stage_gate(sandbox.cfg, S.inputs, log=_quiet)
    assert gate["half"] == "V2_GATE" and gate["held_half_read"] is False and gate["thresholds_from"] == "amendment_1_2026_09_19.pilot_gate_amended"
    assert set(gate["verdict"]) == {S.selection["gnn"]["arm"], S.selection["twin"]["arm"]}
    text = json.dumps(gate)
    assert "NaN" not in text and "nan" not in text.replace("nan", "nan") or "NaN" not in text
    assert "NaN" not in text
    g, t = S.selection["gnn"]["arm"], S.selection["twin"]["arm"]
    assert gate["outcome"] == {g: "PASS", t: "FAIL"}
    assert gate["family_outcome"] == {"GNN_GATE": "PASS", "TWIN_GATE": "FAIL", "overall": "GNN_ONLY_PASS", "selected": {"GNN_GATE": g, "TWIN_GATE": t}}
    assert R.family_outcome({"gnn": "a", "twin": "b"}, {"a": "PASS", "b": "PASS"})["overall"] == "BOTH_PASS"
    assert R.family_outcome({"gnn": "a", "twin": "b"}, {"a": "FAIL", "b": "PASS"})["overall"] == "TWIN_ONLY_PASS"
    assert R.family_outcome({"gnn": "a", "twin": "b"}, {"a": "FAIL", "b": "FAIL"})["overall"] == "PILOT_FAILED"
    cells = {(c["dataset"], c["metric"], c["slice"]): c for c in gate["verdict"][g]["cells"]}
    half = S.world["metaqa"]["half"]
    arr = S.world["metaqa"]["arrays"][f"{R.fit_key(g, 0)}/hit@1"]
    assert cells[("metaqa", "hit@1", "all")]["value"] == round(float(arr[half].mean()), 4)
    hop3 = S.world["metaqa"]["arrays"]["hop"] == 3
    assert cells[("metaqa", "hit@1", "3hop")]["value"] == round(float(arr[half & hop3].mean()), 4)
    assert cells[("metaqa", "hit@1", "all")]["interval"]["low"] > 0 and cells[("metaqa", "hit@1", "all")]["paired_vs"] == "gat_universal_v1"
    assert all(c["holds"] for c in gate["verdict"][g]["cells"])
    failed = [c for c in gate["verdict"][t]["cells"] if not c["holds"]]
    assert failed and failed[0]["dataset"] == "metaqa"
    assert set(gate["reported_not_advancing"]) == ({a for a in M.GNN_CANDIDATES + M.TWIN_CANDIDATES} - {g, t})
    assert all(v["cannot_advance"] for v in gate["reported_not_advancing"].values())
    assert set(gate["paired_on_V2_GATE"]) == {"selected_gnn_minus_selected_twin", "selected_gnn_minus_u_gnn_v2_core78", "selected_gnn_minus_gat_universal_v1_trio",
                                            "u_gnn_v2_core78_minus_gat_universal_v1_trio", "selected_gnn_minus_gat_universal_v1_frozen",
                                            "selected_twin_minus_qls_u_sota_v1_frozen", "selected_twin_minus_gat_no_mp_v1_frozen"}
    assert gate["queries_V2_GATE"] == {name: int(S.world[name]["half"].sum()) for name in PILOT}
    assert set(gate["slices_V2_GATE"]["metaqa_by_hop"]) == {"1hop", "2hop", "3hop"} and gate["mechanism_readouts_V2_GATE"]["squad"][R.fit_key(g, 0)]["delta_ratio"] > 0
    with pytest.raises(SystemExit, match="read once"):
        R.stage_gate(sandbox.cfg, S.inputs, log=_quiet)
    for name in PILOT:
        _restore_finite(sandbox, name, S.world[name])
    S.gate = gate


def test_21_file_gate_moves_the_status_and_seeds_follow_the_pass(sandbox):
    with pytest.raises(SystemExit, match="follows the gate record and the held record"):
        R.stage_file(sandbox.cfg, DATE, ["run"], config=sandbox.config, log=_quiet)
    with pytest.raises(SystemExit):
        REPORT.stage_held(sandbox.cfg, log=_quiet)      # the gate is not filed in the declaration yet
    R.stage_file(sandbox.cfg, DATE, ["gate"], config=sandbox.config, log=_quiet)
    block = sandbox.cfg[f"pilot_gate_record_{DATE}"]
    assert block["status"] == "READ_ONCE_ON_V2_GATE" and block["output_sha256"] == R.sha256_file(sandbox.out / "gate_record.json") and block["held_half_read"] is False
    assert _status_line(sandbox) == "status: PILOT_GATE_READ"
    sandbox.cfg["status"] = "PILOT_GATE_READ"
    with pytest.raises(SystemExit, match="already filed"):
        R.stage_file(sandbox.cfg, "2026_09_21", ["gate"], config=sandbox.config, log=_quiet)
    g, t = S.selection["gnn"]["arm"], S.selection["twin"]["arm"]
    with pytest.raises(SystemExit, match="passed its gate"):
        R.run_fit(t, 1, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    with pytest.raises(SystemExit, match="passed its gate"):
        R.run_fit("gat_universal_v1_trio", 1, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
    for seed in (1, 2):
        rec = R.run_fit(g, seed, S.inputs, S.carves, sandbox.bank, S.training, log=_quiet)
        assert rec["seed"] == seed and rec["key"] == R.fit_key(g, seed)
        S.fits[f"{g}__s{seed}"] = rec
    with pytest.raises(SystemExit, match="seeds 1-2 of a passing arm"):
        REPORT.stage_held(sandbox.cfg, log=_quiet)      # fitted but not yet scored on the populations


def test_23_supplement_eval_pass_refusals_and_tag(sandbox):
    """stage_eval with --models: the contexts and the per-population scoring are stubbed (they need the served
    package); the refusals, the tag and the skip of an existing record are the object under test."""
    calls = []

    def stub_eval(name, cfg, cfg_m3b, cfg_h, inputs, models, context, ds, pkg, m3b_compile, m3b_contract, chunk_nodes, shard=None, only_keys=None,
                  tag=None, supplement=None, log=print):
        calls.append({"name": name, "only_keys": only_keys, "tag": tag, "supplement": supplement, "models": sorted(models)})

    g = S.selection["gnn"]["arm"]
    keys = [R.fit_key(g, 1), R.fit_key(g, 2)]
    expected_tag = "more_" + hashlib.sha256(",".join(sorted(keys)).encode("utf-8")).hexdigest()[:8]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "open_contexts_v2", lambda cfg_m3b, datasets, m3b_compile: (sandbox.ctx2, {n: None for n in datasets}, None, sandbox.bank))
        mp.setattr(R, "eval_dataset", stub_eval)
        with pytest.raises(SystemExit, match="pilot scores the three declared populations"):
            R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, ["hotpotqa"], 24000, log=_quiet)
        with pytest.raises(SystemExit, match="no fitted model"):
            R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, PILOT, 24000, models_only=[R.fit_key(g, 1), "gat_universal_v1_trio__H128__s2"], log=_quiet)
        with pytest.raises(SystemExit, match="already scored"):
            R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, PILOT, 24000, models_only=[R.fit_key(g, 0)], log=_quiet)
        (sandbox.out / "gate_record.json").rename(sandbox.out / "gate_record.json.moved")
        try:
            with pytest.raises(SystemExit, match="follows the gate"):
                R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, PILOT, 24000, models_only=keys, log=_quiet)
        finally:
            (sandbox.out / "gate_record.json.moved").rename(sandbox.out / "gate_record.json")
        R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, PILOT, 24000, log=_quiet)     # seed-0 records exist: nothing scored
        assert calls == []
        R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, PILOT, 24000, models_only=keys, log=_quiet)
        assert [c["name"] for c in calls] == PILOT and all(c["only_keys"] == keys and c["tag"] == expected_tag for c in calls)
        assert all(c["supplement"] == {"tag": expected_tag, "models": sorted(keys), "seed0_record_sha256": R.sha256_file(sandbox.out / "eval" / f"{c['name']}.json")} for c in calls)
        assert all(set(keys) <= set(c["models"]) and R.fit_key(g, 0) in c["models"] for c in calls)
    S.supplement_tag = expected_tag
    S.supplement_keys = keys


def test_24_supplement_records_beside_the_seed0_records(sandbox):
    """The records the stubbed pass would have written: the same populations scored by seeds 1-2 of the passing
    arm, through write_eval_record, pinned to the seed-0 record."""
    g = S.selection["gnn"]["arm"]
    S.supplement = {}
    for name in PILOT:
        world = S.world[name]
        rng = np.random.default_rng({"metaqa": 21, "2wiki": 22, "squad": 23}[name])
        two_gold = world["arrays"][f"{R.fit_key(g, 0)}/gold_total"] >= 2
        extra = {}
        for key in S.supplement_keys:
            for m, a in _scorer_arrays(rng, N_EVAL[name], two_gold, 0.96, 0.99).items():
                extra[f"{key}/{m}"] = a
            for m, a in _mechanism_arrays(rng, key, N_EVAL[name]).items():
                extra[f"{key}/{m}"] = a
        sup_world = {"ids": world["ids"], "half": world["half"], "scorers": S.supplement_keys + [s for s in world["scorers"] if s.startswith("fixed:")],
                     "arrays": {**{k: v for k, v in world["arrays"].items() if "/" not in k or k.split("/")[0].startswith("fixed:")}, **extra}}
        supplement = {"tag": S.supplement_tag, "models": sorted(S.supplement_keys), "seed0_record_sha256": R.sha256_file(sandbox.out / "eval" / f"{name}.json")}
        rec = _record(sandbox, name, sup_world, file_base=f"{name}__{S.supplement_tag}", supplement=supplement, keys=S.supplement_keys)
        assert rec["supplement"] == supplement and set(rec["scorers"]) == set(sup_world["scorers"])
        S.supplement[name] = sup_world
    with pytest.raises(SystemExit, match="already scored"), pytest.MonkeyPatch.context() as mp:
        mp.setattr(R, "open_contexts_v2", lambda cfg_m3b, datasets, m3b_compile: (sandbox.ctx2, {n: None for n in datasets}, None, sandbox.bank))
        R.stage_eval(sandbox.cfg, sandbox.cfg_m3b, sandbox.cfg_h, S.inputs, PILOT, 24000, models_only=[S.supplement_keys[0]], log=_quiet)
    rows, mask, meta = REPORT.pooled_rows("metaqa", "V2_GATE")
    assert set(meta) == {"metaqa", f"metaqa__{S.supplement_tag}"} and all(f"{k}/recall@5" in rows for k in S.supplement_keys)
    assert mask.sum() == S.world["metaqa"]["half"].sum() and rows[f"{S.supplement_keys[0]}/recall@5"].shape[0] == mask.sum()


def test_25_held_half_read_once_by_the_report_after_the_gate_is_filed(sandbox):
    g, t = S.selection["gnn"]["arm"], S.selection["twin"]["arm"]
    cfg_bad = copy.deepcopy(sandbox.cfg)
    cfg_bad[f"pilot_gate_record_{DATE}"]["output_sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="not the record filed"):
        REPORT.stage_held(cfg_bad, log=_quiet)
    cfg_bad = copy.deepcopy(sandbox.cfg)
    cfg_bad["status"] = "DECLARED_NOT_RUN"
    with pytest.raises(SystemExit, match="PILOT_GATE_READ or later"):
        REPORT.stage_held(cfg_bad, log=_quiet)
    with pytest.raises(SystemExit, match="produced once"):
        REPORT.stage_doc(sandbox.cfg, sandbox.cfg_m3b, log=_quiet)      # no held record yet
    held = REPORT.stage_held(sandbox.cfg, log=_quiet)
    assert held["half"] == "V2_HELD_CONFIRMATION" and held["read_once"] is True and held["status_at_read"] == "PILOT_GATE_READ"
    assert held["passing_arms"] == [g] and held["outcome_on_V2_GATE"] == S.gate["outcome"]
    assert held["queries"] == {name: int((~S.world[name]["half"]).sum()) for name in PILOT}
    for name in PILOT:
        heldmask = ~S.world[name]["half"]
        for key in S.world[name]["scorers"]:
            assert held["summary"][name][key]["recall@5"] == round(float(S.world[name]["arrays"][f"{key}/recall@5"][heldmask].mean()), 4)
        for key in S.supplement_keys:
            assert held["summary"][name][key]["hit@1"] == round(float(S.supplement[name]["arrays"][f"{key}/hit@1"][heldmask].mean()), 4)
        assert set(held["records_read"][name]) == {name, f"{name}__{S.supplement_tag}"}
        assert held["seeds"][name][g]["seeds"] == [0, 1, 2] and held["seeds"][name][g]["recall@5"]["n"] == 3 and held["seeds"][name][t]["seeds"] == [0]
    conf = held["gate_cells_confirmatory"]
    assert set(conf) == {g, t} and all(v["confirmatory_not_a_gate"] for v in conf.values())
    assert "u_gnn_v2_core78_minus_gat_universal_v1_trio" in held["paired"] and held["paired"]["selected_gnn_minus_selected_twin"]["metaqa"]["hit@1"]["mean"] > 0
    assert held["frozen_references"]["metaqa"]["gat_universal_v1"]["hit@1"] == round(float(_m3b_array(sandbox, "metaqa", "gat_universal_v1__H128_L2__s0/hit@1")[~S.world["metaqa"]["half"]].mean()), 4)
    assert held["gate_record_sha256"] == R.sha256_file(sandbox.out / "gate_record.json")
    with pytest.raises(SystemExit, match="read once"):
        REPORT.stage_held(sandbox.cfg, log=_quiet)
    S.held = held


def _m3b_array(sb, name: str, key: str) -> np.ndarray:
    with np.load(sb.m3b_out / "eval" / f"{name}.npz") as z:
        return z[key]


def test_26_doc_rendered_from_the_sidecars_reads_held_through_the_record_only(sandbox):
    doc = REPORT.stage_doc(sandbox.cfg, sandbox.cfg_m3b, log=_quiet)
    text = doc.read_text(encoding="utf-8")
    assert doc == R.DOC and "# Universal-v2 pilot" in text and "## 12. Run record" in text
    for section in ("## 1. The frozen contract", "## 2. Cost", "## 3. Selection behind the firewall", "## 4. The pilot gate (V2_GATE, seed 0, read once)",
                    "## 5. Seed confirmation (V2_GATE, seeds 0-2)", "## 6. V2_HELD_CONFIRMATION", "## 7. Slices", "## 8. Mechanism readouts",
                    "## 9. Calibration", "## 10. Audit", "## 11. Reading"):
        assert section in text, section
    g = S.selection["gnn"]["arm"]
    assert f"`{g}` — PASS" in text and "CONFIRMED" in text and "nan" not in text.lower().replace("nan)", "") or "NaN" not in text
    for f in sandbox.cfg["forbidden_framings"][:3]:
        assert f not in text.replace("Forbidden framings are not used: ", "").split("'prove message passing is unnecessary'")[0]
    # amendment 2 family_status_vocabulary: the family labels at a glance and in the reading, and the one-family rule
    assert "**GNN_GATE PASS** / **TWIN_GATE FAIL** / overall **GNN_ONLY_PASS**" in text.split("## 1. The frozen contract")[0]
    reading = text.split("## 11. Reading")[1].split("## 12. Run record")[0]
    assert "overall **GNN_ONLY_PASS**" in reading and "only GNN_GATE passed at seed 0" in reading and "the proposed universal pair has NOT passed" in reading
    assert "has passed the pilot gate" not in reading
    # the compile diagnostics of section 1: every carve and (metaqa) every hop, the hashes of contract_frozen
    contract = text.split("## 1. The frozen contract")[1].split("## 2. Cost")[0]
    for k in ("raw_contract_sha256", "screen_rule_sha256", "surviving_columns_sha256", "six_caches_combined_sha256"):
        assert k in contract, k
    assert "| metaqa | fit | 1hop |" in contract and "| metaqa | select | all |" in contract and "| squad | fit | all |" in contract
    assert "K_REL = 4 relation-text slots" in contract and "queries / s | wall s | peak RSS GB | cache GB" in contract
    rec = R.read_json(sandbox.out / "report_run_record.json")
    assert rec["doc_sha256_lf"] == R.lf_sha256(doc) and set(rec["seed_confirmation"]) == {g}
    conf = rec["seed_confirmation"][g]
    assert conf["complete"] and conf["seeds_present"] == [0, 1, 2] and conf["half"] == "V2_GATE"
    assert all(set(c["per_seed"]) == {"0", "1", "2"} for c in conf["cells"])
    assert rec["hypotheses"]["H_pilot"]["holds"] is False      # the twin failed its gate: H_pilot claims both pass
    assert any("universal_v2.yaml" in k for k in rec["files"]) and any(k.endswith("held_record.json") for k in rec["files"])
    held_section = text.split("## 6. V2_HELD_CONFIRMATION")[1].split("## 7. Slices")[0]
    # the held half is read through held_record.json only: NaN held rows leave the document identical
    for name in PILOT:
        _nan_the_held_rows(sandbox, name, name)
        _nan_the_held_rows(sandbox, f"{name}__{S.supplement_tag}", name)
    doc2 = REPORT.stage_doc(sandbox.cfg, sandbox.cfg_m3b, log=_quiet)
    text2 = doc2.read_text(encoding="utf-8")
    assert text2.split("## 6. V2_HELD_CONFIRMATION")[1].split("## 7. Slices")[0] == held_section
    assert text2.split("## 5. Seed confirmation")[1].split("## 6.")[0] == text.split("## 5. Seed confirmation")[1].split("## 6.")[0]
    assert "nan" not in text2.split("## 12. Run record")[0].lower()
    for name in PILOT:
        _restore_finite(sandbox, name, S.world[name])
        _restore_finite(sandbox, f"{name}__{S.supplement_tag}", S.supplement[name])
    REPORT.stage_doc(sandbox.cfg, sandbox.cfg_m3b, log=_quiet)


def test_27_run_record_filed_last_and_the_status_names_the_family_that_passed(sandbox):
    """The toy world is GNN_ONLY_PASS: the status says so (amendment 2 family_status_vocabulary), never RUN."""
    R.stage_file(sandbox.cfg, DATE, ["run"], config=sandbox.config, log=_quiet)
    block = sandbox.cfg[f"run_record_{DATE}"]
    assert block["outcome"] == S.gate["outcome"] and block["held_record_sha256"] == R.sha256_file(sandbox.out / "held_record.json")
    assert block["family_outcome"] == S.gate["family_outcome"] and block["family_outcome"]["overall"] == "GNN_ONLY_PASS"
    assert block["doc_sha256"] == R.lf_sha256(R.DOC) and block["incidents"] == []
    assert set(block["evals"]) == set(PILOT) | {f"{n}__{S.supplement_tag}" for n in PILOT}
    assert len(block["fits"]) == 8 and all(v["record_sha256"] for v in block["fits"].values())
    assert _status_line(sandbox) == "status: RUN_GNN_ONLY_PASS"
    reloaded = yaml.safe_load(_config_text(sandbox))
    assert [k for k in reloaded if R.DATED.match(k)] == ["amendment_1_2026_09_19", "amendment_2_2026_09_19", f"contract_frozen_{DATE}", f"timing_{DATE}",
                                                          f"pilot_gate_record_{DATE}", f"run_record_{DATE}"]
    # nothing above the dated blocks changed except the status line
    original = (R.ROOT / "configs" / "universal_v2.yaml").read_text(encoding="utf-8").split(LF)   # the repository declaration, not the sandbox copy
    now = _config_text(sandbox).split(LF)
    cut = next(i for i, l in enumerate(now) if l.startswith(f"# -- UNIVERSAL_V2_CORE_CONTRACT"))
    diff = [(a, b) for a, b in zip(original, now[:cut]) if a != b]
    assert diff == [("status: DECLARED_NOT_RUN", "status: RUN_GNN_ONLY_PASS")]
