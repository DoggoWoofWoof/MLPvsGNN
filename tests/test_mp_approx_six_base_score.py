"""configs/mp_approx_six_base.yaml, work 2 (scripts/mp_approx_six_base_score.py): the two pairs, work 2's pins and
preconditions, the six-pair rows and stored arrays, the checkpoint check, the equivalence comparison, the fits record,
the appended blocks, the descriptive gap and no held row in a sidecar. The copy against level 0 on real data runs only
with MPR_SIX_BASE_EQUIVALENCE=1 (it opens the CRAG package and takes minutes); the scoring pass refuses until that
equivalence is filed by the equivalence stage."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_six_base_score as W2  # noqa: E402
from mp_retrieval.m3b_train import METRIC_NAMES  # noqa: E402

SB, L0 = W2.SB, W2.L0


def _raise(msg: str, **_evidence):
    raise SystemExit(msg)


# ── the pairs ────────────────────────────────────────────────────────────────


def test_the_six_pair_is_work_1s_twin_and_stage_2s_gnn():
    keys = W2.SIX.keys()
    assert keys == {**{f"twin{k}": SB.fit_key(k) for k in W2.SEEDS}, **{f"gnn{k}": W2.U6.fit_key(k) for k in W2.SEEDS}}
    assert keys["twin0"] == "u_mlp_v2_mix__H128__six__s0" and keys["gnn2"] == "u_gnn_v2_ef__H128__six__s2"
    assert W2.SIX.stored == ("gnn0", "gnn1", "gnn2")   # T_k has no stored counterpart (work_2.integrity.twin)
    assert W2.SIX.threads == 4


def test_the_trio_pair_is_level_0s():
    keys = W2.TRIO_PAIR.keys()
    assert keys["twin1"] == "u_mlp_v2_mix__H128__s1" and keys["gnn1"] == "u_gnn_v2_ef__H128__s1"
    assert W2.TRIO_PAIR.stored == tuple(L0.STORED_FUNCS) and W2.TRIO_PAIR.threads == L0.SCORE_THREADS == 6


def test_the_scoring_threads_are_the_stage_2_eval_records():
    for name in W2.DATASETS:
        assert json.loads((W2.STAGE2_EVAL / f"{name}.json").read_text(encoding="utf-8"))["threads"] == W2.SCORE_THREADS


def test_a_level_0_hard_stop_never_lands_in_level_0s_directory():
    assert L0.HARD_STOP_DIR[0] == W2.OUT
    assert "mp_approx_l0" not in W2.OUT.as_posix() and W2.OUT == ROOT / "outputs" / "mp_approx_six_base"


def test_the_run_order_scores_each_dataset_once_and_the_equivalence_covers_the_trio():
    assert sorted(W2.RUN_ORDER) == sorted(W2.DATASETS) and len(set(W2.RUN_ORDER)) == 6
    assert sorted(W2.EQUIVALENCE_LIMITS) == sorted(W2.TRIO)


# ── pins and preconditions ───────────────────────────────────────────────────


def test_work_2s_pins_are_filed_and_hold():
    decl = SB.load_declaration()
    s2 = decl["inputs"]["stage_2_eval"]
    assert sorted(s2["pins"]) == sorted(W2.DATASETS)
    assert list(s2["level_0_code_lf"]) == ["scripts/mp_approx_l0.py"]
    assert W2.pin_differences(decl) == []


def test_a_changed_pin_is_named(tmp_path):
    decl = SB.load_declaration()
    decl["inputs"]["stage_2_eval"]["pins"]["musique"]["npz"] = "0" * 64
    decl["inputs"]["stage_2_eval"]["level_0_code_lf"] = {"scripts/mp_approx_l0.py": "0" * 64}
    decl["fits_record_2099_01_01"] = {"fits": {SB.fit_key(0): {"state_sha256": "0" * 64}}}
    assert W2.pin_differences(decl, fits_dir=tmp_path) == ["outputs/universal_v2/six/eval/musique.npz", "scripts/mp_approx_l0.py",
                                                           str(tmp_path / f"{SB.fit_key(0)}.pt")]


def test_the_preconditions_name_what_is_not_filed(tmp_path):
    decl = {k: v for k, v in SB.load_declaration().items() if not k.startswith("fits_record_")}
    missing = W2.preconditions(decl, equiv_dir=tmp_path)
    assert len(missing) == 4 and missing[0].startswith("fits_record_<date>")
    assert missing[1:] == [f"equivalence/{n}.json with equal true" for n in W2.TRIO]
    assert W2.preconditions(decl, equiv_dir=tmp_path, need_equivalence=False) == missing[:1]
    for n in W2.TRIO:
        (tmp_path / f"{n}.json").write_text(json.dumps({"equal": True}), encoding="utf-8")
    decl["fits_record_2099_01_01"] = {"fits": {SB.fit_key(k): {} for k in W2.SEEDS}}
    assert W2.preconditions(decl, equiv_dir=tmp_path) == []
    (tmp_path / "squad.json").write_text(json.dumps({"equal": False}), encoding="utf-8")
    assert W2.preconditions(decl, equiv_dir=tmp_path) == ["equivalence/squad.json with equal true"]


def test_the_scoring_pass_refuses_before_it_opens_anything(tmp_path, monkeypatch):
    decl = {k: v for k, v in SB.load_declaration().items() if not k.startswith("fits_record_")}
    monkeypatch.setattr(SB, "load_declaration", lambda path=None: decl)
    monkeypatch.setattr(W2, "pair_open", lambda *a, **k: (_ for _ in ()).throw(AssertionError("opened")))
    n = torch.get_num_threads()
    try:
        with pytest.raises(SystemExit, match="work 2 does not start"):
            W2.score_pair(W2.SIX, "squad", log=lambda s: None, out_dir=tmp_path / "squad")
    finally:
        torch.set_num_threads(n)
    assert not (tmp_path / "squad").exists()


def test_a_scored_dataset_is_not_rescored(tmp_path, monkeypatch):
    (tmp_path / "meta.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(W2, "pair_verify", lambda pair: (_ for _ in ()).throw(AssertionError("must not run")))
    n = torch.get_num_threads()
    try:
        assert W2.score_pair(W2.SIX, "squad", log=lambda s: None, out_dir=tmp_path) is None
    finally:
        torch.set_num_threads(n)


# ── the six pair's inputs ────────────────────────────────────────────────────


def test_the_six_rows_are_level_0s_on_the_trio_and_every_query_elsewhere():
    want = {"metaqa": 2400, "2wiki": 6290, "squad": 5841, "hotpotqa": 7405, "musique": 2417, "webqsp": 1503}
    for name, n in want.items():
        ids = json.loads((W2.STAGE2_EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))
        half, hop, stored = W2.six_stored(name)
        rows = W2.six_rows(name, ids, half)
        assert rows.size == n, name
        assert sorted(stored) == ["gnn0", "gnn1", "gnn2"] and all(v.shape == (len(ids), len(METRIC_NAMES)) for v in stored.values())
        if name in W2.TRIO:
            assert half is not None and half[rows].all()
            level0 = ROOT / "outputs" / "mp_approx_l0" / name / "q_row.npy"
            if level0.exists():
                assert np.array_equal(rows, np.load(level0))   # the same rows as level 0's sidecar
        else:
            assert half is None and np.array_equal(rows, np.arange(len(ids))) and not hop.any()
    ids = json.loads((W2.STAGE2_EVAL / "metaqa_query_ids.json").read_text(encoding="utf-8"))
    assert np.array_equal(W2.six_stored("metaqa")[1], [L0.hop_from_id(q) for q in ids])


def test_the_six_halves_hops_and_ids_are_the_pilots_on_the_trio():
    d0 = L0.load_declaration()
    for name in W2.TRIO:
        half6, hop6, _ = W2.six_stored(name)
        half0, hop0, _ = L0.load_stored(d0, name)
        assert np.array_equal(half6, half0) and np.array_equal(hop6, hop0)
        ids0 = json.loads((ROOT / d0["inputs"]["eval_arrays"][name]["query_ids"]["path"]).read_text(encoding="utf-8"))
        assert ids0 == json.loads((W2.STAGE2_EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))


def test_a_checkpoint_is_its_records_arm_and_weights(tmp_path):
    (tmp_path / "k.pt").write_bytes(b"weights")
    sha = SB.sha256_file(tmp_path / "k.pt")
    (tmp_path / "k.json").write_text(json.dumps({"arm": "u_gnn_v2_ef", "state_sha256": sha}), encoding="utf-8")
    assert W2.checkpoint_problems(tmp_path, "k", "u_gnn_v2_ef") == []
    assert W2.checkpoint_problems(tmp_path, "k", "u_mlp_v2_mix") == ["k: fit record arm u_gnn_v2_ef is not u_mlp_v2_mix"]
    (tmp_path / "k.pt").write_bytes(b"other")
    assert W2.checkpoint_problems(tmp_path, "k", "u_gnn_v2_ef") == ["k: the weights are not the state_sha256 of their record"]
    assert W2.checkpoint_problems(tmp_path, "missing", "u_gnn_v2_ef")[0].startswith("missing: no fit record or weights")


def test_the_stage_2_checkpoints_pass_the_check():
    for k in W2.SEEDS:
        assert W2.checkpoint_problems(W2.U6.SIX_FITS, W2.U6.fit_key(k), "u_gnn_v2_ef") == []


# ── the equivalence ──────────────────────────────────────────────────────────


def test_the_comparison_reads_every_array_the_ids_and_the_chunking():
    a = {k: 1 for k in W2.COMPARED}
    a["arrays_sha256"] = {"z.npy": "x", "q_metrics.npy": "y"}
    b = json.loads(json.dumps(a))
    assert W2.compare_sidecars(a, b) == []
    b["arrays_sha256"]["z.npy"] = "other"
    b["chunks"] = 2
    assert W2.compare_sidecars(a, b) == ["arrays_sha256", "chunks"]
    assert {"arrays_sha256", "arrays_shape", "qids_sha256", "uq_rows", "chunk_queries"} <= set(W2.COMPARED)


def test_the_equivalence_runs_on_the_trio_only(tmp_path):
    with pytest.raises(SystemExit):
        W2.stage_equivalence("hotpotqa", log=lambda s: None, root=tmp_path)


@pytest.mark.skipif(os.environ.get("MPR_SIX_BASE_EQUIVALENCE") != "1",
                    reason="opens the CRAG package and takes minutes; set MPR_SIX_BASE_EQUIVALENCE=1")
def test_the_copy_equals_level_0_on_the_trio_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(SB, "OUT", tmp_path)   # a hard stop of the copy lands in the test's directory
    n = torch.get_num_threads()
    try:
        rec = W2.stage_equivalence("metaqa", log=lambda s: None, limit=13, root=tmp_path / "equivalence")
    finally:
        torch.set_num_threads(n)
    assert rec["equal"] and rec["differ"] == [] and rec["chunks"] >= 2
    assert rec["arrays_sha256"] == rec["copy_arrays_sha256"]


# ── the fits record and the appended blocks ──────────────────────────────────


def _fit(d: Path, seed: int, seconds: float, best: int, epochs: int) -> None:
    key = SB.fit_key(seed)
    (d / f"{key}.pt").write_bytes(b"weights" + bytes([seed]))
    hist = [{"epoch": e, "seconds": 10.0 * (e + 1), "select_macro_recall@5": 0.5 + 0.01 * e, "select_recall@5": {}} for e in range(epochs)]
    rec = {"arm": SB.ARM, "parameters": SB.PARAMETERS, "seed": seed, "epochs_run": epochs, "best_epoch": best, "seconds": seconds,
           "history": hist, "state_sha256": SB.sha256_file(d / f"{key}.pt"), "threads": 8, "pack_workers": 2,
           "blas_threads": {"OPENBLAS_NUM_THREADS": "2"}, "peak_rss_bytes": 1, "placement": {"host": "laptop"}, "git_head": "g",
           "module_sha256": {"scripts/mp_approx_six_base.py": "s"}, "utc": "u"}
    (d / f"{key}.json").write_text(json.dumps(rec), encoding="utf-8")


def test_the_fits_record_transcribes_the_three_records(tmp_path):
    for s, (sec, best, epochs) in enumerate(((3600.0, 2, 4), (7200.0, 5, 6), (1800.0, 1, 3))):
        _fit(tmp_path, s, sec, best, epochs)
    block = W2.fits_record(tmp_path)
    assert block["fit_hours"] == 3.5 and block["within_ceiling"] and block["one_script"]
    f1 = block["fits"][SB.fit_key(1)]
    assert f1["best_epoch"] == 5 and f1["best_epoch_at_the_cap"] and f1["epochs_run"] == 6
    assert f1["select_macro_recall5_trace"] == [0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
    assert f1["epoch_seconds"] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    assert not block["fits"][SB.fit_key(0)]["best_epoch_at_the_cap"]
    assert f1["record_sha256"] == SB.sha256_file(tmp_path / f"{SB.fit_key(1)}.json")


def test_the_fits_record_waits_for_all_three_and_refuses_a_changed_weight(tmp_path, monkeypatch):
    _fit(tmp_path, 0, 1.0, 1, 3)
    _fit(tmp_path, 1, 1.0, 1, 3)
    with pytest.raises(SystemExit, match="fits_record follows all three"):
        W2.fits_record(tmp_path)
    _fit(tmp_path, 2, 1.0, 1, 3)
    (tmp_path / f"{SB.fit_key(2)}.ckpt").write_bytes(b"")
    with pytest.raises(SystemExit, match="not finished"):
        W2.fits_record(tmp_path)
    (tmp_path / f"{SB.fit_key(2)}.ckpt").unlink()
    (tmp_path / f"{SB.fit_key(2)}.pt").write_bytes(b"changed")
    monkeypatch.setattr(W2, "hard_stop", _raise)
    with pytest.raises(SystemExit, match="state_sha256"):
        W2.fits_record(tmp_path)


def test_an_appended_block_reads_back_and_moves_the_status_only_when_asked(tmp_path):
    cfg = tmp_path / "decl.yaml"
    cfg.write_text("phase: X\nstatus: DECLARED_NOT_RUN\nwork: 1\n", encoding="utf-8")
    W2.append_block("fits_record_2026_10_01", {"a": 1.5, "b": [1, 2]}, config=cfg)
    d = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert d["status"] == "DECLARED_NOT_RUN" and d["fits_record_2026_10_01"] == {"a": 1.5, "b": [1, 2]}
    W2.append_block("run_record_2026_10_02", {"c": np.float64(0.25), "d": float("nan")}, config=cfg, status_to="RUN")
    d = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert d["status"] == "RUN" and d["run_record_2026_10_02"] == {"c": 0.25, "d": None} and d["work"] == 1
    with pytest.raises(SystemExit):
        W2.append_block("run_record_2026_10_02", {}, config=cfg)
    with pytest.raises(SystemExit):
        W2.append_block("run_record_2026_10_03", {}, config=cfg, status_to="RUN")   # the status is no longer DECLARED_NOT_RUN


# ── the record's quantities ──────────────────────────────────────────────────


def test_the_descriptive_gap_is_the_seed_mean_of_g_and_t_on_a_hand_case():
    qm = np.zeros((4, len(L0.FUNCS), len(METRIC_NAMES)))
    fi = {f: i for i, f in enumerate(L0.FUNCS)}
    mi = {m: i for i, m in enumerate(METRIC_NAMES)}
    for k in W2.SEEDS:
        qm[:, fi[f"gnn{k}"], mi["recall@5"]] = [1, 1, 0, 0]
        qm[:, fi[f"twin{k}"], mi["recall@5"]] = [1, 0, 0, 0]
        qm[:, fi[f"gnn{k}"], mi["hit@1"]] = k / 2            # seeds 0, 0.5, 1: seed mean 0.5
        qm[:, fi[f"noedge{k}"], mi["recall@5"]] = 1           # G0_k is not part of the gap
    g = W2.descriptive_gap(qm)
    assert g["G"]["recall@5"] == 0.5 and g["T"]["recall@5"] == 0.25 and g["G_minus_T"]["recall@5"] == 0.25
    assert g["G"]["hit@1"] == 0.5 and g["T"]["hit@1"] == 0.0 and g["G"]["full_coverage@5"] == 0.0
    assert set(g) == {"G", "T", "G_minus_T"} and all(set(v) == set(W2.GAP_METRICS) for v in g.values())


@pytest.mark.parametrize("name", W2.DATASETS)
def test_no_held_query_in_a_six_sidecar(name):
    d = W2.OUT / name
    if not (d / "meta.json").exists():
        pytest.skip(f"{name}: not scored yet")
    q_row = np.load(d / "q_row.npy")
    assert W2.held_rows_scored(name, q_row) == 0
    ids = json.loads((W2.STAGE2_EVAL / f"{name}_query_ids.json").read_text(encoding="utf-8"))
    assert json.loads((d / "qids.json").read_text(encoding="utf-8")) == [ids[i] for i in q_row]


def test_the_held_row_count_reads_stage_2s_halves():
    half, _hop, _stored = W2.six_stored("2wiki")
    assert W2.held_rows_scored("2wiki", np.flatnonzero(half)) == 0
    assert W2.held_rows_scored("2wiki", np.flatnonzero(~half)[:7]) == 7
    assert W2.held_rows_scored("musique", np.arange(5)) == 0


def test_main_needs_its_arguments():
    for argv in (["--stage", "equivalence"], ["--stage", "score"], ["--stage", "fits-record"], ["--stage", "file", "--date", "2026_10_01"],
                 ["--stage", "fits-record", "--date", "2026_10_01", "--host"],
                 ["--stage", "file", "--date", "2026_10_01", "--commit", "abc", "--host"], ["--stage", "run", "--host"],
                 ["--stage", "verify"], ["--stage", "status", "--shard", "0/2"],
                 ["--stage", "score", "--dataset", "squad", "--shard", "2/2"]):
        with pytest.raises(SystemExit):
            W2.main(argv)


# ── amendment 2: the host, the shards, the sidecars that stay there ─────────


def test_the_host_amendment_pins_the_six_verify_records():
    block = SB.load_declaration()[W2.HOST_BLOCK]
    m = block["mirror"]
    assert m["root"] == yaml.safe_load(W2.MIRROR_CONFIG.read_text(encoding="utf-8"))["host"]["mirror_root"]
    covered = []
    for rel, want in m["verify_records"].items():
        path = ROOT / rel
        if not path.exists():
            pytest.skip(f"{rel}: the fetched verify record is not on this machine")
        assert SB.sha256_file(path) == want
        rec = json.loads(path.read_text(encoding="utf-8"))
        assert rec["status"] == "VERIFIED" and rec["freeze_RECORD_SHA256"] == m["freeze_RECORD_SHA256"]
        covered += rec["datasets"]
    assert sorted(covered) == sorted(W2.DATASETS)
    assert "4 torch threads" in " ".join(block["host_placement"]["threads"].split())


def test_host_mode_puts_the_mirror_in_memory_only(monkeypatch):
    decl = SB.load_declaration()
    if not all((ROOT / rel).exists() for rel in decl[W2.HOST_BLOCK]["mirror"]["verify_records"]):
        pytest.skip("the fetched verify records are not on this machine")
    monkeypatch.setattr(W2.V2, "load_configs", W2.V2.load_configs)   # restored after the test
    monkeypatch.setattr(W2, "PLACEMENT", {"where": "laptop"})
    m3b_config = ROOT / "configs" / "m3b_controlled_comparison.yaml"
    before = SB.sha256_file(m3b_config)
    placement = W2.host_mode(decl, log=lambda s: None)
    _cfg, cfg_m3b, _cfg_h = W2.V2.load_configs()
    root = decl[W2.HOST_BLOCK]["mirror"]["root"]
    assert cfg_m3b["substrate"]["package_root"] == str(Path(root))
    assert placement["where"] == "host" and placement["mirror_root"] == root and W2.PLACEMENT is placement
    assert SB.sha256_file(m3b_config) == before
    assert yaml.safe_load(m3b_config.read_text(encoding="utf-8"))["substrate"]["package_root"] != root


def _host_decl(tmp_path: Path, records: dict, root: str | None = None) -> dict:
    real = SB.load_declaration()[W2.HOST_BLOCK]["mirror"]
    root = real["root"] if root is None else root
    pins = {}
    for i, rec in enumerate(records.values()):
        p = tmp_path / "outputs" / "host_mirror_six" / f"verify_{i}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rec), encoding="utf-8")
        pins[p.relative_to(tmp_path).as_posix()] = SB.sha256_file(p)
    return {W2.HOST_BLOCK: {"mirror": {"root": root, "freeze_RECORD_SHA256": real["freeze_RECORD_SHA256"], "verify_records": pins}}}


def _verified(datasets: list[str], **changes) -> dict:
    real = SB.load_declaration()[W2.HOST_BLOCK]["mirror"]
    rec = {"datasets": datasets, "mirror": f"{real['root']}/data/final_canonical", "status": "VERIFIED",
           "freeze_matches_declared": True, "loader_imported_from_mirror": True, "freeze_RECORD_SHA256": real["freeze_RECORD_SHA256"]}
    rec.update(changes)
    return rec


def test_host_mode_refuses_what_is_not_the_verified_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(W2, "hard_stop", _raise)
    monkeypatch.setattr(W2, "ROOT", tmp_path)
    monkeypatch.setattr(W2.V2, "load_configs", W2.V2.load_configs)
    monkeypatch.setattr(W2, "PLACEMENT", {"where": "laptop"})
    with pytest.raises(SystemExit, match="not filed"):
        W2.host_mode({}, log=lambda s: None)
    six = {"a": _verified(["metaqa", "2wiki", "squad"]), "b": _verified(["hotpotqa", "musique", "webqsp"])}
    decl = _host_decl(tmp_path, six)
    W2.host_mode(decl, log=lambda s: None)   # the fake records pass
    assert W2.PLACEMENT["where"] == "host"
    with pytest.raises(SystemExit, match="root differs"):
        W2.host_mode(_host_decl(tmp_path, six, root="D:/elsewhere/CRAG"), log=lambda s: None)
    for bad, match in ((_verified(["metaqa", "2wiki", "squad"], status="MISMATCH"), "not VERIFIED"),
                       (_verified(["metaqa", "2wiki", "squad"], loader_imported_from_mirror=False), "not VERIFIED"),
                       (_verified(["metaqa", "2wiki", "squad"], mirror="C:/other/data/final_canonical"), "not VERIFIED"),
                       (_verified(["metaqa", "2wiki", "squad"], freeze_RECORD_SHA256="0" * 64), "not VERIFIED"),
                       (_verified(["metaqa", "2wiki"]), "cover the six")):
        with pytest.raises(SystemExit, match=match):
            W2.host_mode(_host_decl(tmp_path, {"a": bad, "b": six["b"]}), log=lambda s: None)
    decl = _host_decl(tmp_path, six)
    first = next(iter(decl[W2.HOST_BLOCK]["mirror"]["verify_records"]))
    (tmp_path / first).write_text("{}", encoding="utf-8")   # not the pinned bytes
    with pytest.raises(SystemExit, match="pinned verify record"):
        W2.host_mode(decl, log=lambda s: None)


def test_a_shard_is_i_of_n_and_the_shards_partition_the_chunks():
    assert W2.parse_shard(None) is None and W2.parse_shard("0/4") == (0, 4) and W2.parse_shard("3/4") == (3, 4)
    for text in ("4/4", "1/1", "-1/3", "a/b", "3", "1/2/3"):
        with pytest.raises(SystemExit):
            W2.parse_shard(text)
    for n_chunks in (1, 2, 7, 43):
        assert W2.shard_chunks(n_chunks, None) == list(range(n_chunks))
        for n in (2, 3, 4):
            parts = [W2.shard_chunks(n_chunks, (i, n)) for i in range(n)]
            assert sorted(c for p in parts for c in p) == list(range(n_chunks))
            assert all(c % n == i for i, p in enumerate(parts) for c in p)


def _mini_sidecar(d: Path) -> None:
    """Level 0's layout on two queries: the rows grouped by query, the fold rule, one array that will stay on the host."""
    d.mkdir(parents=True)
    qids = ["metaqa:1hop:x0", "metaqa:2hop:x1"]
    (d / "qids.json").write_text(json.dumps(qids), encoding="utf-8")
    arrays = {"query": np.asarray([0, 0, 1], dtype=np.int32), "q_row": np.asarray([3, 9], dtype=np.int64),
              "q_fold": np.asarray([L0.fold_of(q) for q in qids], dtype=np.int64),
              "q_metrics": np.zeros((2, len(L0.FUNCS), len(METRIC_NAMES))), "x": np.arange(6, dtype=np.float32).reshape(3, 2)}
    shas = {}
    for key, arr in arrays.items():
        np.save(d / f"{key}.npy", arr)
        shas[f"{key}.npy"] = SB.sha256_file(d / f"{key}.npy")
    meta = {"arrays_sha256": shas, "qids_sha256": SB.sha256_file(d / "qids.json"), "pair": "six", "limit": None, "mismatches": 0}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def test_the_host_verify_and_the_file_stage_take_a_sidecar_that_stays_on_the_host(tmp_path, monkeypatch):
    monkeypatch.setattr(W2, "hard_stop", _raise)
    monkeypatch.setattr(W2, "PLACEMENT", {"where": "host"})
    d = tmp_path / "metaqa"
    _mini_sidecar(d)
    sc, where = W2.filed_sidecar("metaqa", d, None)
    assert where == "laptop" and sc.n_q == 2 and sc.n_rows == 3
    rec = W2.stage_verify(log=lambda s: None, out=tmp_path, datasets=("metaqa",))
    assert rec["datasets"]["metaqa"] == {"meta_sha256": SB.sha256_file(d / "meta.json"), "arrays_checked": 5, "queries": 2, "uq_rows": 3}
    assert json.loads((tmp_path / W2.HOST_VERIFY).read_text(encoding="utf-8"))["placement"] == {"where": "host"}
    x = (d / "x.npy").read_bytes()
    (d / "x.npy").unlink()   # the array that stays on the host
    with pytest.raises(SystemExit, match="does not vouch"):
        W2.filed_sidecar("metaqa", d, None)
    sc, where = W2.filed_sidecar("metaqa", d, rec)
    assert where == "host" and sc.n_q == 2
    stale = {"datasets": {"metaqa": dict(rec["datasets"]["metaqa"], meta_sha256="0" * 64)}}
    with pytest.raises(SystemExit, match="does not vouch"):
        W2.filed_sidecar("metaqa", d, stale)
    q_metrics = (d / "q_metrics.npy").read_bytes()
    np.save(d / "q_metrics.npy", np.ones((2, len(L0.FUNCS), len(METRIC_NAMES))))
    with pytest.raises(SystemExit, match="not the sha256"):
        W2.filed_sidecar("metaqa", d, rec)
    (d / "q_metrics.npy").write_bytes(q_metrics)
    (d / "q_fold.npy").unlink()
    with pytest.raises(SystemExit, match="fetch"):
        W2.filed_sidecar("metaqa", d, rec)
    np.save(d / "q_fold.npy", np.asarray([L0.fold_of(q) for q in ["metaqa:1hop:x0", "metaqa:2hop:x1"]], dtype=np.int64))
    (d / "x.npy").write_bytes(x)
    (d / "qids.json").write_text(json.dumps(["metaqa:1hop:x0", "metaqa:2hop:other"]), encoding="utf-8")
    with pytest.raises(SystemExit, match="qids.json"):
        W2.filed_sidecar("metaqa", d, rec)


def test_the_verify_stage_waits_for_the_six_passes(tmp_path):
    with pytest.raises(SystemExit, match="not scored"):
        W2.stage_verify(log=lambda s: None, out=tmp_path, datasets=("squad",))
