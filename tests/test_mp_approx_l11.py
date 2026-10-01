"""MP-Approx level 11, population module (configs/mp_approx_l11.yaml#tests): the declaration, its pins and its constants;
the population rule on toy ids; the scoring pass's rebinding; the check stage on a synthetic combined sidecar and its
refusals; the stages' machine; and no held, level 0, level 8, level 9 or level 10 query in any sidecar."""

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
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar and relation text)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)

_quiet = T8._quiet


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])


# ── the declaration, its pins and its constants ──────────────────────────────


def test_the_declaration_parses_and_every_pin_is_the_files_current_sha256(stops):
    decl = L11.load_declaration()
    assert decl["phase"] == "MP_APPROX_L11" and decl["status"] in ("DECLARED_NOT_RUN", "RUN")
    assert decl["registered_question"] == L8.load_declaration()["registered_question"]
    assert list(decl["inputs"]["relations"]["order"]) == list(L8.REL_ORDER)
    L11.verify_inputs(decl)   # level 10's check on this copy (level 9's, level 8's and level 0's pins), then level 10's
    assert not (stops / "hard_stops.json").exists()


@pytest.mark.parametrize("key", ["script_lf", "tests_lf", "declaration_lf", "fit_script_lf", "fit_tests_lf"])
def test_a_changed_level10_pin_stops(stops, monkeypatch, key):
    monkeypatch.setattr(L11, "_L10_VERIFY", lambda decl: None)   # the copied pins are the test above's
    decl = L11.load_declaration()
    decl["inputs"]["level10"][key]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L11.verify_inputs(decl)
    assert decl["inputs"]["level10"][key]["path"] in (stops / "hard_stops.json").read_text(encoding="utf-8")


def test_a_changed_level10_rows_pin_and_a_changed_copied_pin_stop(stops, monkeypatch):
    decl = L11.load_declaration()
    decl["inputs"]["level9"]["script_lf"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L11.verify_inputs(decl)
    assert "scripts/mp_approx_l9.py" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    monkeypatch.setattr(L11, "_L10_VERIFY", lambda decl: None)
    decl = L11.load_declaration()
    decl["inputs"]["level10"]["excluded_rows"]["qids"]["sha256"] = "0" * 64
    with pytest.raises(SystemExit, match="HARD STOP"):
        L11.verify_inputs(decl)
    assert "outputs/mp_approx_l10/metaqa/qids.json" in (stops / "hard_stops.json").read_text(encoding="utf-8")


def test_the_declared_constants_are_the_files():
    decl = L11.load_declaration()
    assert (L11.PER_HOP, L11.SALT, L11.NAME) == (1000, "mp_approx_l11|", "metaqa")
    assert {int(h): v for h, v in decl["population"]["available_per_hop"].items()} == L11.AVAILABLE_PER_HOP
    assert all(L10.AVAILABLE_PER_HOP[h] - L10.PER_HOP == L11.AVAILABLE_PER_HOP[h] for h in (1, 2, 3))
    assert L11.FAMILIES == ("std", "nb") and L11.SCRIPT_REL == "scripts/mp_approx_l11.py"
    assert L11.LEVELS == ("level0", "level8", "level9", "level10")
    # the level 10 pins name level 10's committed files and rows
    lv = decl["inputs"]["level10"]
    assert (lv["declaration_lf"]["path"], lv["script_lf"]["path"], lv["tests_lf"]["path"], lv["fit_script_lf"]["path"],
            lv["fit_tests_lf"]["path"]) == ("configs/mp_approx_l10.yaml", "scripts/mp_approx_l10.py", "tests/test_mp_approx_l10.py",
                                            "scripts/mp_approx_l10_fit.py", "tests/test_mp_approx_l10_fit.py")
    assert lv["excluded_rows"]["qids"]["path"] == "outputs/mp_approx_l10/metaqa/qids.json"
    # every pin level 10 declares is copied unchanged
    l10 = L10.load_declaration()["inputs"]
    assert all(decl["inputs"][k] == l10[k] for k in l10)


# ── the population rule ──────────────────────────────────────────────────────


def _toy_population(seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"metaqa:{h}hop:dev:{i}" for h in (1, 2, 3) for i in range(500)]
    gate = rng.random(len(ids)) < 0.7
    excluded = L0.metaqa_subsample(ids, gate, per_hop=40)
    level8, _ = L8.l8_rows(ids, gate, excluded, per_hop=50)
    level9, _ = L9.l9_rows(ids, gate, excluded, per_hop=50, level8_ids=[ids[i] for i in level8], level8_per_hop=50)
    level10, _ = L10.l10_rows(ids, gate, excluded, per_hop=50, level8_ids=[ids[i] for i in level8],
                              level9_ids=[ids[i] for i in level9], level8_per_hop=50, level9_per_hop=50)
    return ids, gate, excluded, level8, level9, level10


def _rows(ids, gate, excluded, level8, level9, level10, per_hop=60, **kw):
    return L11.l11_rows(ids, gate, excluded, per_hop, level8_ids=[ids[i] for i in level8], level9_ids=[ids[i] for i in level9],
                        level10_ids=[ids[i] for i in level10], level8_per_hop=50, level9_per_hop=50, level10_per_hop=50, **kw)


def test_the_population_rule_leaves_out_levels_0_8_9_and_10_and_sorts_by_its_own_salt():
    ids, gate, excluded, level8, level9, level10 = _toy_population()
    rows, available = _rows(ids, gate, excluded, level8, level9, level10)
    assert rows.size == 180 and np.all(np.diff(rows) > 0) and gate[rows].all()
    for earlier in (excluded, level8, level9, level10):
        assert not np.isin(rows, earlier).any()
    assert not np.isin(level10, level9).any() and not np.isin(level10, level8).any() and not np.isin(level10, excluded).any()
    out = set(excluded.tolist()) | set(level8.tolist()) | set(level9.tolist()) | set(level10.tolist())
    for h in (1, 2, 3):
        pool = [int(i) for i in np.flatnonzero(gate) if L0.hop_from_id(ids[i]) == h and int(i) not in out]
        assert available[h] == len(pool)
        want = sorted(pool, key=lambda i: hashlib.sha256(("mp_approx_l11|" + ids[i]).encode("utf-8")).hexdigest())[:60]
        assert sorted(want) == [int(r) for r in rows if L0.hop_from_id(ids[r]) == h]
    again, _ = L11.l11_rows(list(ids), gate.copy(), excluded.copy(), per_hop=60, level8_per_hop=50, level9_per_hop=50,
                            level10_per_hop=50)
    assert np.array_equal(rows, again)
    with pytest.raises(SystemExit):
        L11.l11_rows(ids, gate, excluded, per_hop=10_000, level8_per_hop=50, level9_per_hop=50, level10_per_hop=50)


def test_the_earlier_levels_recomputed_rows_must_be_their_pinned_ids(stops):
    ids, gate, excluded, level8, level9, level10 = _toy_population(1)
    with pytest.raises(SystemExit, match="HARD STOP"):
        _rows(ids, gate, excluded, level8, level9, level10[1:])
    assert "level 10's recomputed rows" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="HARD STOP"):
        _rows(ids, gate, excluded, level8, level9[1:], level10)
    assert "level 9's recomputed rows" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    with pytest.raises(SystemExit, match="HARD STOP"):
        _rows(ids, gate, excluded, level8[1:], level9, level10)
    assert "level 8's recomputed rows" in (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the scoring pass's rebinding ─────────────────────────────────────────────


def test_the_scoring_pass_rebinds_level_8s_names_only_inside_the_pass(tmp_path):
    names = ("l8_rows", "walk_entries", "verify_inputs", "CONFIG", "OUT", "DATA", "AVAILABLE_PER_HOP")
    before = {k: getattr(L8, k) for k in names}

    def rule(ids, gate, excluded, per_hop=1000):
        return None

    with L11.level8_rebound(tmp_path / "x", rule):
        assert L8.l8_rows is rule and L8.walk_entries is L9.walk_entries_both and L8.verify_inputs is L11.verify_inputs
        assert (L8.CONFIG, L8.OUT, L8.DATA) == (L11.CONFIG, L11.OUT, tmp_path / "x") and L8.AVAILABLE_PER_HOP == L11.AVAILABLE_PER_HOP
        assert L9._L8_WALKS is before["walk_entries"]
    assert all(getattr(L8, k) is v for k, v in before.items())
    with pytest.raises(RuntimeError):
        with L11.level8_rebound(tmp_path / "x", rule):
            raise RuntimeError("inside the pass")
    assert all(getattr(L8, k) is v for k, v in before.items())


def test_the_scoring_pass_reads_this_files_rule_pins_and_paths_inside_it(tmp_path, monkeypatch, stops):
    seen = {}

    def fake_stage_score(decl, log, shard, limit, out_dir):
        seen.update(rows=L8.l8_rows, walks=L8.walk_entries, verify=L8.verify_inputs, config=L8.CONFIG, data=L8.DATA,
                    available=dict(L8.AVAILABLE_PER_HOP), args=(shard, limit, out_dir))

    monkeypatch.setattr(L8, "stage_score", fake_stage_score)
    L11.stage_score(L11.load_declaration(), _quiet, (1, 3), None, tmp_path / "o")
    assert seen["walks"] is L9.walk_entries_both and seen["verify"] is L11.verify_inputs and seen["config"] == L11.CONFIG
    assert seen["data"] == tmp_path / "o" and seen["available"] == L11.AVAILABLE_PER_HOP and seen["args"] == ((1, 3), None, tmp_path / "o")
    assert seen["rows"].__qualname__.startswith("stage_score") and L8.l8_rows is L9._L8_ROWS


# ── the check stage on a synthetic combined sidecar ──────────────────────────


def _check_env(tmp_path, monkeypatch, n_q, seed):
    d = tmp_path / "l11" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both, n_q=n_q, seed=seed)
    monkeypatch.setattr(L11, "DATA", d)
    monkeypatch.setattr(L11, "verify_inputs", lambda decl: None)
    monkeypatch.setattr(L0, "load_stored", lambda decl, name: (np.ones(len(qids), dtype=bool), None, None))
    monkeypatch.setattr(L8, "FIT_THREADS", 1)
    return d, qids


def test_the_check_runs_level_9s_check_family_on_both_views(tmp_path, monkeypatch, stops, one_thread):
    d, _qids = _check_env(tmp_path, monkeypatch, n_q=30, seed=4)
    check = L11.stage_check(L11.load_declaration(), _quiet)
    fam = check["families"]
    assert set(fam) == {"std", "nb"} and all(fam[f]["direction_check"]["passes"] for f in L11.FAMILIES)
    assert fam["std"]["chain_fit"]["recall"]["all"] == 1.0 and fam["nb"]["sizes"]["entries_total"] < fam["std"]["sizes"]["entries_total"]
    assert 0.0 <= check["gold_unreached_nb"]["all"] <= 1.0 and 0.0 <= check["nb_trim"]["gold_share_removed"]["all"] <= 1.0
    on_disk = json.loads((d / "check.json").read_text(encoding="utf-8"))
    assert on_disk["meta_sha256"] == L0.sha256_file(d / "meta.json") and "module_sha256" in on_disk
    assert not (stops / "hard_stops.json").exists()


def test_the_check_refuses_a_smoke_sidecar_an_earlier_levels_id_and_a_failed_direction_check(tmp_path, monkeypatch, stops, one_thread):
    d, qids = _check_env(tmp_path, monkeypatch, n_q=12, seed=8)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    (d / "meta.json").write_text(json.dumps({**meta, "limit": 12}), encoding="utf-8")
    with pytest.raises(SystemExit, match="smoke"):
        L11.stage_check(L11.load_declaration(), _quiet)
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    for level in ("level10", "level9", "level8", "level0"):
        decl = L11.load_declaration()
        fake = tmp_path / f"{level}_qids.json"
        fake.write_text(json.dumps([qids[3]]), encoding="utf-8")
        decl["inputs"][level]["excluded_rows"]["qids"]["path"] = str(fake)
        with pytest.raises(SystemExit, match="HARD STOP"):
            L11.stage_check(decl, _quiet)
        assert "level 0, level 8, level 9 or level 10 query id" in (stops / "hard_stops.json").read_text(encoding="utf-8")
    real = L8.chain_tokens
    monkeypatch.setattr(L8, "chain_tokens", lambda steps, swap=False: real(steps, not swap))   # the map, turned around
    with pytest.raises(SystemExit, match="HARD STOP"):
        L11.stage_check(L11.load_declaration(), _quiet)
    assert "direction_check (std)" in (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── the stages' machine ──────────────────────────────────────────────────────


def test_the_stages_refuse_the_wrong_machine_and_a_smoke_under_the_outputs(stops):
    for argv in (["--stage", "check"], ["--stage", "score"], ["--stage", "fit", "--host"], ["--stage", "check", "--host", "--shard", "0/3"],
                 ["--stage", "score", "--host", "--limit", "4"], ["--stage", "check", "--host", "--limit", "4", "--out", "x"],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L11.OUT / "x")],
                 ["--stage", "score", "--host", "--limit", "4", "--out", str(L11.OUT)]):
        with pytest.raises(SystemExit):
            L11.main(argv)


# ── no held, level 0, level 8, level 9 or level 10 query in any sidecar ──────


def test_no_held_or_earlier_levels_query_id_appears_in_any_sidecar():
    side = L11.DATA / "qids.json"
    if not side.exists():
        pytest.skip("this file's sidecar is not on this machine")
    decl0 = L0.load_declaration()
    arrays = decl0["inputs"]["eval_arrays"]["metaqa"]
    ids = json.loads((ROOT / arrays["query_ids"]["path"]).read_text(encoding="utf-8"))
    with np.load(ROOT / arrays["seed0"]["path"]) as z:
        half = z["half"].astype(bool)
    gate = {q for q, h in zip(ids, half) if h}
    got = json.loads(side.read_text(encoding="utf-8"))
    inp = L11.load_declaration()["inputs"]
    for level in L11.LEVELS:
        earlier = set(json.loads((ROOT / inp[level]["excluded_rows"]["qids"]["path"]).read_text(encoding="utf-8")))
        assert not (set(got) & earlier), level
    assert got and set(got) <= gate and len(got) == len(set(got)) == 3 * L11.PER_HOP
    meta = L11.DATA / "meta.json"
    if meta.exists():
        assert json.loads(meta.read_text(encoding="utf-8"))["qids_sha256"] == L0.sha256_file(side)
