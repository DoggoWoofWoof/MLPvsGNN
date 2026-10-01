"""MP-Approx level 13, population module (configs/mp_approx_l13.yaml#tests): the declaration, its pins and its constants;
the population rule on toy ids; the scoring pass's rebinding; the check on synthetic combined sidecars and its
refusals; the stages' machine; and no held, earlier level's or train-split query in the dev sidecar."""

from __future__ import annotations

import hashlib
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
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
import mp_approx_l9 as L9  # noqa: E402
import mp_approx_l10 as L10  # noqa: E402
import mp_approx_l11 as L11  # noqa: E402
import mp_approx_l12 as L12  # noqa: E402
import mp_approx_l13 as L13  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)

_quiet = T8._quiet


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, L12, L13):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])


def _stopped(stops) -> str:
    return (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the declaration, its pins and its constants ──────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256(stops):
    decl = L13.load_declaration()
    assert decl["phase"] == "MP_APPROX_L13" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L8.load_declaration()["registered_question"]
    assert list(decl["inputs"]["relations"]["order"]) == list(L8.REL_ORDER)
    if not all((ROOT / decl["inputs"]["level12_carves"]["dir"] / c / "meta.json").exists() for c in L13.L12_CARVES):
        pytest.skip("level 12's carve sidecars are not on this machine")
    L13.verify_inputs(decl)   # level 12's check on this copy, level 12's own files and rows, and its carve sidecars' records
    assert not (stops / "hard_stops.json").exists()


@pytest.mark.parametrize("key", ["script_lf", "tests_lf", "declaration_lf", "fit_script_lf", "fit_tests_lf"])
def test_a_changed_level12_pin_stops(stops, monkeypatch, key):
    monkeypatch.setattr(L13, "_L12_VERIFY", lambda decl: None)   # the copied pins are the test above's
    decl = L13.load_declaration()
    decl["inputs"]["level12"][key]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L13.verify_inputs(decl)
    assert decl["inputs"]["level12"][key]["path"] in _stopped(stops)


def test_a_changed_rows_pin_a_changed_carve_pin_and_a_changed_copied_pin_stop(stops, monkeypatch):
    decl = L13.load_declaration()
    decl["inputs"]["level11"]["script_lf"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L13.verify_inputs(decl)
    assert "scripts/mp_approx_l11.py" in _stopped(stops)
    monkeypatch.setattr(L13, "_L12_VERIFY", lambda decl: None)
    decl = L13.load_declaration()
    decl["inputs"]["level12"]["excluded_rows"]["qids"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L13.verify_inputs(decl)
    assert "outputs/mp_approx_l12/metaqa/qids.json" in _stopped(stops)
    for carve, f in (("select", "check.json"), ("fit", "meta.json")):
        decl = L13.load_declaration()
        decl["inputs"]["level12_carves"][carve][f] = "0" * 64
        with pytest.raises(SystemExit, match="HARD STOP"):
            L13.verify_inputs(decl)
        assert f"outputs/mp_approx_l12/carves/{carve}/{f}" in _stopped(stops)


def test_the_declared_constants_are_the_files():
    decl = L13.load_declaration()
    assert (L13.PER_HOP, L13.SALT, L13.NAME) == ({1: 0, 2: 1000, 3: 1000}, "mp_approx_l13|", "metaqa")
    assert {int(h): int(v) for h, v in decl["population"]["per_hop"].items()} == L13.PER_HOP
    assert {int(h): int(v) for h, v in decl["population"]["available_per_hop"].items()} == L13.AVAILABLE_PER_HOP
    assert all(L12.AVAILABLE_PER_HOP[h] - L12.PER_HOP[h] == L13.AVAILABLE_PER_HOP[h] for h in (1, 2, 3))
    assert all(L13.PER_HOP[h] <= L13.AVAILABLE_PER_HOP[h] for h in (1, 2, 3)) and L13.PER_HOP[1] == L13.AVAILABLE_PER_HOP[1] == 0
    assert L13.FAMILIES == ("std", "nb") and L13.SCRIPT_REL == "scripts/mp_approx_l13.py" and L13.DEV_SHARDS == 4
    assert L13.LEVELS == ("level0", "level8", "level9", "level10", "level11", "level12")
    # the level 12 pins name level 12's committed files and rows, and the carve sidecars it trains on
    lv = decl["inputs"]["level12"]
    assert (lv["declaration_lf"]["path"], lv["script_lf"]["path"], lv["tests_lf"]["path"], lv["fit_script_lf"]["path"],
            lv["fit_tests_lf"]["path"]) == ("configs/mp_approx_l12.yaml", "scripts/mp_approx_l12.py", "tests/test_mp_approx_l12.py",
                                            "scripts/mp_approx_l12_fit.py", "tests/test_mp_approx_l12_fit.py")
    assert lv["excluded_rows"]["qids"]["path"] == "outputs/mp_approx_l12/metaqa/qids.json"
    cv = decl["inputs"]["level12_carves"]
    assert cv["dir"] == "outputs/mp_approx_l12/carves/" and tuple(k for k in cv if k != "dir") == L13.L12_CARVES
    assert all(tuple(cv[c]) == ("meta.json", "check.json") for c in L13.L12_CARVES)
    # every pin level 12 declares is copied unchanged, and its carve pins too
    l12 = L12.load_declaration()
    assert all(decl["inputs"][k] == l12["inputs"][k] for k in l12["inputs"])
    assert decl["carves"]["pins"] == l12["carves"]["pins"]


# ── the population rule ──────────────────────────────────────────────────────


def _toy_population(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(600)]
    gate = rng.random(len(ids)) < 0.7
    excluded = L0.metaqa_subsample(ids, gate, per_hop=40)
    level8, _ = L8.l8_rows(ids, gate, excluded, per_hop=50)
    l8_ids = [ids[i] for i in level8]
    level9, _ = L9.l9_rows(ids, gate, excluded, per_hop=50, level8_ids=l8_ids, level8_per_hop=50)
    l9_ids = [ids[i] for i in level9]
    level10, _ = L10.l10_rows(ids, gate, excluded, per_hop=50, level8_ids=l8_ids, level9_ids=l9_ids, level8_per_hop=50,
                              level9_per_hop=50)
    l10_ids = [ids[i] for i in level10]
    level11, _ = L11.l11_rows(ids, gate, excluded, per_hop=50, level8_ids=l8_ids, level9_ids=l9_ids, level10_ids=l10_ids,
                              level8_per_hop=50, level9_per_hop=50, level10_per_hop=50)
    level12, _ = L12.l12_rows(ids, gate, excluded, {1: 30, 2: 60, 3: 60}, level8_ids=l8_ids, level9_ids=l9_ids, level10_ids=l10_ids,
                              level11_ids=[ids[i] for i in level11], level8_per_hop=50, level9_per_hop=50, level10_per_hop=50,
                              level11_per_hop=50)
    return ids, gate, excluded, level8, level9, level10, level11, level12


def _rows(ids, gate, excluded, level8, level9, level10, level11, level12, per_hop=None, **kw):
    return L13.l13_rows(ids, gate, excluded, per_hop or {1: 0, 2: 70, 3: 70}, level8_ids=[ids[i] for i in level8],
                        level9_ids=[ids[i] for i in level9], level10_ids=[ids[i] for i in level10],
                        level11_ids=[ids[i] for i in level11], level12_ids=[ids[i] for i in level12], level8_per_hop=50,
                        level9_per_hop=50, level10_per_hop=50, level11_per_hop=50, level12_per_hop={1: 30, 2: 60, 3: 60}, **kw)


def test_the_population_rule_leaves_out_levels_0_and_8_to_12_and_sorts_by_its_own_salt(stops):
    ids, gate, excluded, *earlier = _toy_population()
    rows, available = _rows(ids, gate, excluded, *earlier)
    assert rows.size == 140 and np.all(np.diff(rows) > 0) and gate[rows].all()
    assert not any(L0.hop_from_id(ids[r]) == 1 for r in rows)
    for e in (excluded, *earlier):
        assert not np.isin(rows, e).any()
    out = set(excluded.tolist()).union(*(set(e.tolist()) for e in earlier))
    for h, keep in ((2, 70), (3, 70)):
        pool = [int(i) for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == h and int(i) not in out]
        assert available[h] == len(pool) > keep
        want = sorted(pool, key=lambda i: hashlib.sha256(("mp_approx_l13|" + ids[i]).encode("utf-8")).hexdigest())[:keep]
        assert sorted(want) == [int(r) for r in rows if L0.hop_from_id(ids[r]) == h]
    assert available[1] == len([i for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == 1 and int(i) not in out])
    again, _ = L13.l13_rows(list(ids), gate.copy(), excluded.copy(), {1: 0, 2: 70, 3: 70}, level8_per_hop=50, level9_per_hop=50,
                            level10_per_hop=50, level11_per_hop=50, level12_per_hop={1: 30, 2: 60, 3: 60})
    assert np.array_equal(rows, again)
    with pytest.raises(SystemExit, match="fewer than"):
        _rows(ids, gate, excluded, *earlier, per_hop={1: 0, 2: available[2] + 1, 3: 70})
    assert not (stops / "hard_stops.json").exists()


def test_the_earlier_levels_recomputed_rows_must_be_their_pinned_ids(stops):
    ids, gate, excluded, *earlier = _toy_population(1)
    names = ("level 8's", "level 9's", "level 10's", "level 11's", "level 12's")
    for j, level in enumerate(names):
        cut = [e[1:] if i == j else e for i, e in enumerate(earlier)]
        with pytest.raises(SystemExit, match="HARD STOP"):
            _rows(ids, gate, excluded, *cut)
        assert f"{level} recomputed rows" in _stopped(stops)


# ── the scoring pass ─────────────────────────────────────────────────────────


def test_the_scoring_pass_rebinds_level_8s_names_only_inside_the_pass(tmp_path):
    names = ("l8_rows", "walk_entries", "verify_inputs", "CONFIG", "OUT", "DATA", "AVAILABLE_PER_HOP", "PER_HOP")
    before = {k: getattr(L8, k) for k in names}

    def rule(ids, gate, excluded, per_hop=None):
        return None

    with L13.level8_rebound(tmp_path / "x", rule):
        assert L8.l8_rows is rule and L8.walk_entries is L9.walk_entries_both and L8.verify_inputs is L13.verify_inputs
        assert (L8.CONFIG, L8.OUT, L8.DATA) == (L13.CONFIG, L13.OUT, tmp_path / "x") and L8.AVAILABLE_PER_HOP == L13.AVAILABLE_PER_HOP
        assert L8.PER_HOP == L13.PER_HOP and L9._L8_WALKS is before["walk_entries"]
    assert all(getattr(L8, k) is v for k, v in before.items())
    with pytest.raises(RuntimeError):
        with L13.level8_rebound(tmp_path / "x", rule):
            raise RuntimeError("inside the pass")
    assert all(getattr(L8, k) is v for k, v in before.items()) and L8.PER_HOP == 1000


def test_the_scoring_pass_reads_this_files_rule_per_hop_pins_and_paths_inside_it(tmp_path, monkeypatch, stops):
    seen = {}

    def fake_stage_score(decl, log, shard, limit, out_dir):
        seen.update(rows=L8.l8_rows, walks=L8.walk_entries, verify=L8.verify_inputs, config=L8.CONFIG, data=L8.DATA,
                    available=dict(L8.AVAILABLE_PER_HOP), per_hop=L8.PER_HOP, args=(shard, limit, out_dir))

    monkeypatch.setattr(L8, "stage_score", fake_stage_score)
    L13.stage_score(L13.load_declaration(), _quiet, (1, 2), None, tmp_path / "o")
    assert seen["walks"] is L9.walk_entries_both and seen["verify"] is L13.verify_inputs and seen["config"] == L13.CONFIG
    assert seen["data"] == tmp_path / "o" and seen["available"] == L13.AVAILABLE_PER_HOP and seen["per_hop"] == L13.PER_HOP
    assert seen["args"] == ((1, 2), None, tmp_path / "o") and seen["rows"].__qualname__.startswith("stage_score")
    assert L8.l8_rows is L9._L8_ROWS and L8.PER_HOP == 1000 and L12.l12_rows is L13._L12_ROWS


# ── the check ────────────────────────────────────────────────────────────────


def _rewrite_ids(d: Path, new_ids: list[str]) -> None:
    """A toy sidecar's ids replaced: q_fold and q_inner recomputed by level 0's rules, and meta.json's sha256 values updated."""
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "qids.json").write_text(json.dumps(new_ids), encoding="utf-8")
    np.save(d / "q_fold.npy", np.asarray([L0.fold_of(q) for q in new_ids], dtype=np.int64))
    np.save(d / "q_inner.npy", np.asarray([int(L0.is_inner(q)) for q in new_ids], dtype=np.int64))
    for key in ("q_fold", "q_inner"):
        meta["arrays_sha256"][f"{key}.npy"] = L0.sha256_file(d / f"{key}.npy")
    meta["qids_sha256"] = L0.sha256_file(d / "qids.json")
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")


def _check_env(tmp_path, monkeypatch, n_q, seed):
    d = tmp_path / "l13" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    monkeypatch.setattr(L13, "DATA", d)
    monkeypatch.setattr(L13, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    return d, qids


def test_the_check_runs_level_9s_check_family_on_both_views(tmp_path, monkeypatch, stops, one_thread):
    d, qids = _check_env(tmp_path, monkeypatch, n_q=30, seed=4)
    check = L13.stage_check(L13.load_declaration(), _quiet)
    fam = check["families"]
    assert set(fam) == {"std", "nb"} and all(fam[f]["direction_check"]["passes"] for f in L13.FAMILIES)
    assert fam["std"]["chain_fit"]["recall"]["all"] == 1.0 and fam["nb"]["sizes"]["entries_total"] < fam["std"]["sizes"]["entries_total"]
    assert check["hops"] == L13.hop_counts(qids) and check["rows_with_gold_in_pool"]["all"] == len(qids)
    on_disk = json.loads((d / "check.json").read_text(encoding="utf-8"))
    assert on_disk["meta_sha256"] == L0.sha256_file(d / "meta.json") and "module_sha256" in on_disk
    assert not (stops / "hard_stops.json").exists()


def test_the_check_refuses_a_smoke_an_earlier_levels_id_a_held_row_a_train_id_and_a_failed_direction_check(tmp_path, monkeypatch, stops,
                                                                                                            one_thread):
    d, qids = _check_env(tmp_path, monkeypatch, n_q=12, seed=8)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "meta.json").write_text(json.dumps({**meta, "limit": 12}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke"):
        L13.stage_check(L13.load_declaration(), _quiet)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for level in ("level12", "level11", "level10", "level9", "level8", "level0"):
        decl = L13.load_declaration()
        fake = tmp_path / f"{level}_qids.json"
        fake.write_text(json.dumps([qids[3]]), encoding="utf-8")
        decl["inputs"][level]["excluded_rows"]["qids"]["path"] = str(fake)
        with pytest.raises(SystemExit, match="HARD STOP"):
            L13.stage_check(decl, _quiet)
        assert "a level 0 or level 8 to 12 query id" in _stopped(stops)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.r_[np.ones(5, dtype=bool), False, np.ones(len(qids) - 6, dtype=bool)],
                                                               None, None))
    with pytest.raises(SystemExit, match="HARD STOP"):
        L13.stage_check(L13.load_declaration(), _quiet)
    assert "a held row is in the sidecar" in _stopped(stops)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    _rewrite_ids(d, [q.replace(":toy:", ":train:") if i == 2 else q for i, q in enumerate(qids)])
    with pytest.raises(SystemExit, match="HARD STOP"):
        L13.stage_check(L13.load_declaration(), _quiet)
    assert "a train-split query id is in the dev sidecar" in _stopped(stops)
    _rewrite_ids(d, qids)
    real = L8.chain_tokens
    monkeypatch.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))   # the map, turned around
    with pytest.raises(SystemExit, match="HARD STOP"):
        L13.stage_check(L13.load_declaration(), _quiet)
    assert "direction_check (std)" in _stopped(stops)


# ── the stages' machine ──────────────────────────────────────────────────────


def test_the_stages_refuse_the_wrong_machine_and_every_wrong_combination(stops):
    for argv in (["--stage", "check"], ["--stage", "score"], ["--stage", "fit", "--host"], ["--stage", "carve", "--host"],
                 ["--stage", "check", "--host", "--shard", "0/2"], ["--stage", "assemble", "--host", "--shard", "0/2"],
                 ["--stage", "score", "--host", "--limit", "4"], ["--stage", "check", "--host", "--limit", "4", "--out", "x"],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L13.OUT / "x")],
                 ["--stage", "score", "--host", "--limit", "4", "--out", "x", "--shard", "0/2"]):
        with pytest.raises(SystemExit):
            L13.main(argv)


# ── no held, earlier level's or train-split query in the dev sidecar ─────────


def test_no_held_earlier_levels_or_train_split_query_id_appears_in_the_dev_sidecar():
    side = L13.DATA / "qids.json"
    if not side.exists():
        pytest.skip("this file's sidecar is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"]["metaqa"]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    got = json.loads(side.read_text(encoding="utf-8"))
    inp = L13.load_declaration()["inputs"]
    for level in L13.LEVELS:
        earlier = set(json.loads((ROOT / inp[level]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
        assert not (set(got) & earlier), level
    assert got and set(got) <= gate and len(got) == len(set(got)) == sum(L13.PER_HOP.values())
    assert not any(L13.is_train_id(q) for q in got) and L13.hop_counts(got) == L13.PER_HOP
    meta = L13.DATA / "meta.json"
    if meta.exists():
        assert json.loads(meta.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)
