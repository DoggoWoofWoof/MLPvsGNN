"""MP-Approx level 14, fit module (configs/mp_approx_l14.yaml#tests): the declared constants; B0View, the design look's,
copied unchanged; the eval part's refusals (no check, a failed direction check, a changed meta.json or array, a smoke, a
sidecar not r's and ids other than r's pin); the b0 view on toy sidecars (only bucket-0 types kept, the per-query
pointers and entry blocks agreeing with level 9's View, level 9's View restored, a b0 view holding another type refused)
and on a handmade layout (a query with no bucket-0 type, a non-block layout refused); a query with no bucket-0 type
keeping the null type only in a b0 unit; a full unit against level 13's run_unit and a b0 unit against level 12's
run_unit on the b0 view, bit for bit; the scored rows' golds, the other seeds' twins and the stored metrics never
reaching a unit's dsh, b1d or b0 scores; the bands, rho, reaches_gnn, the b0_view anchor and the interpretation map (per
hop included) on toy cases; the code check; the fit, repeat, read, doc and file stages end to end on synthetic
sidecars; and the stages' machine."""

from __future__ import annotations

import copy
import inspect
import json
import math
import shutil
import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l0 as L0  # noqa: E402
import mp_approx_l3 as L3  # noqa: E402
import mp_approx_l8 as L8  # noqa: E402
import mp_approx_l9 as L9  # noqa: E402
import mp_approx_l10 as L10  # noqa: E402
import mp_approx_l10_fit as F10  # noqa: E402
import mp_approx_l11 as L11  # noqa: E402
import mp_approx_l11_fit as F11  # noqa: E402
import mp_approx_l12 as P12  # noqa: E402
import mp_approx_l12_fit as F12  # noqa: E402
import mp_approx_l13 as P13  # noqa: E402
import mp_approx_l13_fit as F13  # noqa: E402
import mp_approx_l14 as P  # noqa: E402
import mp_approx_l14_fit as F  # noqa: E402
import test_mp_approx_l8 as T8  # noqa: E402  (level 8's toy sidecar and relation text)
import test_mp_approx_l9 as T9  # noqa: E402  (level 9's combined toy sidecar)
import test_mp_approx_l12 as T12  # noqa: E402  (the population module's id rewrite)
import test_mp_approx_l12_fit as T12F  # noqa: E402  (level 12's synthetic carves, with level 12's checks run)
import test_mp_approx_l14 as T14  # noqa: E402  (an assembled carve's stand-in)
from mp_retrieval import m3b_pools  # noqa: E402

_quiet = T8._quiet
TBL = L8.TB ** L8.MAX_L
CHOICES = ("beta_dsh", "kappa_dsh", "eta_dsh", "kappa_cov", "eta_cov")
B1D_CHOICES = ("beta_b1d", "kappa_b1d", "eta_b1d")
FORBIDDEN = ("message passing is unnecessary", "we do not need message passing", "the mlp wins")
DESIGN_LOOK = ROOT / "outputs" / "mp_approx_l12_diag" / "diag_b0.py"
DESIGN_LOOK_SHA256 = "fbefc98a917b13de268e591bcdcb385fee9a6e9118a77cfc91b6163cb2ea01f5"


@pytest.fixture
def stops(tmp_path, monkeypatch):
    """Every hard stop of this test goes to its own directory, never under outputs/."""
    d = tmp_path / "stops"
    for mod in (L0, L3, L8, L9, L10, L11, P12, P13, P):
        monkeypatch.setattr(mod, "HARD_STOP_DIR", [d])
    return d


@pytest.fixture
def one_thread():
    before = (torch.get_num_threads(), torch.are_deterministic_algorithms_enabled())
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before[0])
    torch.use_deterministic_algorithms(before[1])   # the fit stages turn it on


def _stopped(stops: Path) -> str:
    return (stops / "hard_stops.json").read_text(encoding="utf-8")


# ── synthetic sidecars: level 12's carves and r, checked ─────────────────────


def _nb_b0_dropped(every: int):
    """Level 9's combined walk programme, with the nb family's bucket-0 entries (stored bucket 2) dropped on every
    every-th call, so that some toy queries hold no bucket-0 nb type. Their std entries, which plant the golds, are kept."""
    calls = [0]

    def programme(*args, **kwargs):
        code, node, count = L9.walk_entries_both(*args, **kwargs)
        calls[0] += 1
        if calls[0] % every:
            return code, node, count
        keep = np.asarray(code, dtype=np.int64) // TBL != 2
        return code[keep], node[keep], count[keep]

    return programme


def _own_process(path: Path) -> None:
    """A check record as its own host process files it: this test's process has imported the fit module, which a host
    check job never imports (placement.identical_code)."""
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec["module_sha256"] = {rel: sha for rel, sha in rec["module_sha256"].items() if rel != F.SCRIPT_REL}
    path.write_text(json.dumps(rec), encoding="utf-8")


def _env(tmp_path, monkeypatch, carves: dict, n_r: int = 24, seed: int = 5, every: int | None = None):
    """Level 12's synthetic carves (its _env: level 12's paths pointed at them and level 12's checks run), assembled
    stand-ins for level 12's other carves (r's disjointness check reads their ids), r's sidecar (level 9's combined toy
    sidecar under train-split ids, naming carve r and its digest), this file's paths pointed at them, the carve pins and
    r's pin set to them, and this file's check run on r."""
    root, decl12, ids = T12F._env(tmp_path, monkeypatch, carves, n_dev=12)
    for c in P12.CARVES:
        if c not in carves:
            T14._fake_carve(root / "carves" / c, [f"metaqa:{1 + i % 3}hop:train:z{c}_{i}" for i in range(5)])
    d = tmp_path / "l14" / "metaqa"
    qids = T9._sidecar(d, monkeypatch, L9.walk_entries_both if every is None else _nb_b0_dropped(every), n_q=n_r, seed=seed)
    r = [q.replace(":toy:", ":train:r") for q in qids]
    T12._rewrite_ids(d, r, carve="r", carve_ids_sha256=m3b_pools.ids_digest(r))
    for mod in (P, F):
        monkeypatch.setattr(mod, "DATA", d)
    monkeypatch.setattr(F, "CARVES_DIR", root / "carves")
    monkeypatch.setattr(P, "verify_inputs", lambda decl: None)   # the population module's pins are its own tests'
    decl = P.load_declaration()
    for c in carves:
        decl["carves"]["pins"][c] = decl12["carves"]["pins"][c]
    decl["carves"]["pins"]["r"] = {"queries": len(r), "zero_gold_excluded": 0, "ids_sha256": m3b_pools.ids_digest(r),
                                   "hops": P.hop_counts(r)}
    P.stage_check(decl, _quiet)
    _own_process(d / "check.json")
    ids["eval"] = r
    return root, decl, ids


def _fitter(decl, view="full", fit="TW-1x"):
    return F10.make_fitter(F.deploy_view(decl, view, fit), T8._rel(), F11.LEVEL10_FIT)


# ── the declared constants ───────────────────────────────────────────────────


def test_the_declared_constants_are_the_files():
    decl = P.load_declaration()
    assert {k: tuple(v) for k, v in decl["carves"]["fits"].items()} == F.FITS and list(decl["carves"]["fits"]) == list(F.FITS)
    assert all(F12.FITS[fit] == carves for fit, carves in F.FITS.items()) and F13.FITS == F.FITS
    assert set(P.L12_CARVES) == {"select", *(c for cs in F.FITS.values() for c in cs)}
    assert {k: tuple(v) for k, v in decl["units"]["roles"].items()} == F12.ROLES
    arms = decl["arms"]
    assert tuple(arms["views"]) == F.VIEWS == ("full", "b0")
    assert {k: tuple(v) for k, v in arms["read_arms"].items()} == F.READ_ARMS and list(arms["read_arms"]) == list(F.READ_ARMS)
    assert tuple(arms["references"]) == F.REFERENCES and tuple(arms["scores"]) == tuple(F.SCORES)
    assert {sc: view for sc, (view, _m) in F.SCORES.items()} == {"dsh": "full", "b1d": "full", "b0": "b0"}
    assert F.UNIT_SCORES == {"full": F13.SCORES, "b0": ("dsh",)} and all(m in F.UNIT_SCORES[v] for v, m in F.SCORES.values())
    assert decl["readings"]["primary"] == F.PRIMARY == "TW-1x-b0"
    contrasts = decl["statistics"]["contrasts"]
    assert {k: tuple(s.strip() for s in v.split(" - ")) for k, v in contrasts.items()} == F.CONTRASTS and list(contrasts) == list(F.CONTRASTS)
    assert " ".join(decl["statistics"]["hop_contrasts"].split()).startswith("b0_adds and b1d_adds per hop (1, 2, 3)")
    assert F.HOP_CONTRASTS == ("b0_adds", "b1d_adds") and F.HOPS == (1, 2, 3) and "per hop (1, 2, 3)" in decl["quantities"]["strata"]
    assert set(decl["readings"]["bands"]) == set(F.BANDS) and set(decl["readings"]["flags"]) == set(F.FLAGS)
    assert tuple(decl["readings"]["interpretation_map"]) == F.INTERPRETATION
    assert F.REPEAT_UNIT == ("b0", "TW-1x", 0)
    assert "The unit (b0, TW-1x, k = 0), which the primary reads, runs again" in " ".join(decl["units"]["repeat"].split())
    assert ("For TW-1x-b1d and TW-4x-b1d against NB-oracle, and for TW-1x-b0 and TW-4x-b0 against NB-oracle-b0"
            in " ".join(decl["quantities"]["anchors"]["gap_split"].split()))
    assert F.GAP_SPLIT == {"TW-1x-b1d": "NB-oracle", "TW-4x-b1d": "NB-oracle", "TW-1x-b0": "NB-oracle-b0", "TW-4x-b0": "NB-oracle-b0"}
    assert set(decl["quantities"]["anchors"]) >= {"b0_view", "b1d_moved", "exchangeability", "beta_choices", "reaches_gnn"}
    per_fit = " ".join(decl["units"]["per_fit"].split())
    assert F.FOLD == 0 and tuple(F.SEEDS) == (0, 1, 2) and len(F.VIEWS) * len(F.FITS) * len(F.SEEDS) == 12
    assert "One unit per (view, fit, k) at fold 0, for the views full and b0, the fits TW-1x and TW-4x, and k = 0, 1 and 2: 12" in per_fit
    assert "units/<view>/<fit>/k<k>_f0" in decl["units"]["unit_files"]
    assert F.unit_paths("b0", "TW-4x", 2) == (F.DATA / "units" / "b0" / "TW-4x" / "k2_f0.npz", F.DATA / "units" / "b0" / "TW-4x" / "k2_f0.json")
    assert F.SCRIPT_REL in decl["outputs"]["scripts"] and P.SCRIPT_REL in decl["outputs"]["scripts"]
    assert F.DOC == ROOT / decl["outputs"]["document"] and F.RECORD == F.OUT / "record.json"
    assert "outputs/mp_approx_l14/record.json" in decl["outputs"]["record"]
    assert (F.CONFIG, F.OUT, F.DATA, F.NAME) == (P.CONFIG, P.OUT, P.DATA, P.NAME)
    assert F.CARVES_DIR == F12.CARVES_DIR == ROOT / decl["inputs"]["level12_carves"]["dir"]
    assert (F.BETAS, F.KAPPAS, F.ETAS) == (F11.BETAS, F10.KAPPAS, F10.ETAS) and F11.LEVEL10_FIT == "NB-set"
    sel = " ".join(arms["selection"].split())
    assert "beta in {1, 1.5, 2, 3, 4, 8, infinity}" in sel and "eta in {0.01, 0.1, 1, 10, 100}" in sel
    assert "A b0 unit chooses on its own view" in sel
    assert [float(Fraction(x)) for x in sel.split("kappa in {")[1].split("}")[0].split(", ")] == list(F.KAPPAS)
    assert F.TBL == 28 ** 3 == 21952 and "code // 28^3 = 0" in " ".join(arms["views"]["b0"].split())
    assert F.UNIT_KEYS == {"full": F13.UNIT_KEYS, "b0": F12.UNIT_KEYS} and set(F12.UNIT_KEYS) < set(F13.UNIT_KEYS)
    pins = decl["carves"]["pins"]
    views = [pins["r"]["queries"] + pins["select"]["queries"] + sum(pins[c]["queries"] for c in F.FITS[fit]) for fit in F.FITS]
    assert views == [19377, 37257]   # the views the reservations name
    text = " ".join(decl["placement"]["reservations"].split())
    for phrase in ("here the views hold 19,377 and 37,257 queries with 11,920 scored",
                   "The full TW-1x units reserve 1.15 CPUs and 2.25 GB, and the full TW-4x units 1.35 CPUs and 2.75 GB",
                   "The b0 TW-1x units and the repeat reserve 1.1 CPUs and 1.6 GB, and the b0 TW-4x units 1.25 CPUs and 1.9 GB",
                   "The read reserves 1 CPU and 1.1 GB", "Each test job reserves 1 CPU and 0.4 GB"):
        assert phrase in text, phrase


def test_b0view_is_the_design_looks_copied_unchanged():
    """arms.views.b0: the design look's B0View, copied unchanged. The design look is untracked, so the comparison runs
    where it is (the laptop); the pin in the declaration's text is checked everywhere."""
    assert f"diag_b0.py ({DESIGN_LOOK_SHA256})" in " ".join(P.CONFIG.read_text(encoding="utf-8").split())
    assert F.ORIGINAL_VIEW is L9.View and issubclass(F.B0View, L9.View) and F.B0View.__mro__[1] is L9.View
    if not DESIGN_LOOK.exists():
        pytest.skip("the design look is untracked and lives on the laptop only")
    assert L0.sha256_file(DESIGN_LOOK) == DESIGN_LOOK_SHA256
    src = DESIGN_LOOK.read_text(encoding="utf-8")
    assert inspect.getsource(F.B0View) in src and "ORIGINAL_VIEW = L9.View\n" in src


# ── the eval part ────────────────────────────────────────────────────────────


def test_the_eval_part_refuses_no_check_a_failed_direction_check_a_changed_meta_json_or_array_a_smoke_and_ids_not_rs(
        tmp_path, monkeypatch, stops, one_thread):
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=14)
    d = F.DATA
    v = F.eval_view(decl)
    assert list(v.qids) == ids["eval"] and type(v) is L9.View and v.family == "nb"
    check_p, meta_p = d / "check.json", d / "meta.json"
    check, meta = check_p.read_text(encoding="utf-8"), meta_p.read_text(encoding="utf-8")
    ck = json.loads(check)
    aside = d / "check.json.aside"
    check_p.rename(aside)
    with pytest.raises(SystemExit, match="r's check has not passed"):
        F.eval_view(decl)
    aside.rename(check_p)
    nb = ck["families"]["nb"]
    failed = {**ck, "families": {**ck["families"], "nb": {**nb, "direction_check": {**nb["direction_check"], "passes": False}}}}
    for bad in ({**ck, "stage": "carve_check"}, {**ck, "carve": "fit"}, failed):
        check_p.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(SystemExit, match="r's check has not passed"):
            F.eval_view(decl)
    check_p.write_text(check, encoding="utf-8")

    def stops_with(text: str, dcl=decl) -> None:
        (stops / "hard_stops.json").unlink(missing_ok=True)
        with pytest.raises(SystemExit, match="HARD STOP"):
            F.eval_view(dcl)
        assert text in _stopped(stops), text

    def with_meta(**extra) -> None:
        """meta.json changed and the check's record of it set to match, so that the later refusals are reached."""
        meta_p.write_text(json.dumps({**json.loads(meta), **extra}), encoding="utf-8")
        check_p.write_text(json.dumps({**ck, "meta_sha256": L0.sha256_file(meta_p)}), encoding="utf-8")

    meta_p.write_text(json.dumps({**json.loads(meta), "note": "changed after the check"}), encoding="utf-8")
    stops_with("part eval's meta.json is not the one r's check read")
    with_meta(limit=5)
    stops_with("part eval is a smoke run")
    for extra in ({"carve": "x8"}, {"carve_ids_sha256": "0" * 64}):
        with_meta(**extra)
        stops_with("part eval is not what r's pin records")
    with_meta()
    assert list(F.eval_view(decl).qids) == ids["eval"]
    for key, val in (("ids_sha256", "0" * 64), ("queries", 99), ("hops", {1: 0, 2: 0, 3: len(ids["eval"])})):
        bad = copy.deepcopy(decl)
        bad["carves"]["pins"]["r"][key] = val
        stops_with("part eval is not what r's pin records", bad)
    raw = (d / "t_gold.npy").read_bytes()
    np.save(d / "t_gold.npy", np.load(d / "t_gold.npy") + 1)   # an array that is not the one meta.json records
    stops_with("deploy view: part eval:")
    (d / "t_gold.npy").write_bytes(raw)
    assert list(F.eval_view(decl).qids) == ids["eval"]
    off = [q.replace(":train:", ":dev:") if i == 2 else q for i, q in enumerate(ids["eval"])]
    T12._rewrite_ids(d, off, carve="r", carve_ids_sha256=m3b_pools.ids_digest(ids["eval"]))
    check_p.write_text(json.dumps({**ck, "meta_sha256": L0.sha256_file(meta_p)}), encoding="utf-8")
    stops_with("part eval is not what r's pin records")   # a row that is not a train-split id


# ── the b0 view ──────────────────────────────────────────────────────────────


def test_the_b0_view_keeps_only_bucket_0_types_with_pointers_and_entry_blocks_that_agree_with_level_9s_view(
        tmp_path, monkeypatch, stops, one_thread):
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=20, every=4)
    full, b0 = L9.View(F.DATA, "nb"), F.B0View(F.DATA, "nb")
    code = np.asarray(full.t_code, dtype=np.int64)
    assert np.all(np.asarray(b0.t_code, dtype=np.int64) // TBL == 0) and b0.n_q == full.n_q and list(b0.qids) == list(full.qids)
    assert np.array_equal(b0.type_ptr, np.r_[0, np.cumsum(b0.q_types)])
    none = []
    for q in range(full.n_q):
        rows = full.type_rows(q)
        keep = np.flatnonzero(code[rows] // TBL == 0) + rows.start
        assert np.array_equal(keep, np.arange(rows.start, rows.start + keep.size))   # a prefix of the query's types
        r0 = b0.type_rows(q)
        assert r0.stop - r0.start == keep.size == int(b0.q_types[q])
        for key in ("t_code", "t_size", "t_gold", "t_first"):
            assert np.array_equal(np.asarray(getattr(b0, key))[r0], np.asarray(getattr(full, key))[keep]), key
        node, cnt, tl = b0.entries(q)
        fnode, fcnt, ftl = full.entries(q)
        m = int(b0.q_entries[q])
        assert m == int(np.asarray(full.t_size)[keep].sum()) == node.size
        assert np.array_equal(node, fnode[:m]) and np.array_equal(cnt, fcnt[:m]) and np.array_equal(tl, ftl[:m])
        for c in code[keep]:
            assert np.array_equal(b0.reach(q, int(c)), full.reach(q, int(c)))
        for c in code[rows][code[rows] // TBL == 1][:3]:
            assert b0.reach(q, int(c)).size == 0 < full.reach(q, int(c)).size
        if keep.size:
            assert int(b0.entry_start[q]) == int(full.entry_start[q])
        else:
            none.append(q)
    assert none and len(none) < full.n_q   # the toy holds rows without a bucket-0 type, and rows with one
    dv = F.deploy_view(decl, "b0", "TW-1x")
    assert L9.View is F.ORIGINAL_VIEW and all(isinstance(v, F.B0View) for v in dv.parts)   # level 9's View restored
    assert np.all(np.asarray(dv.t_code, dtype=np.int64) // TBL == 0) and tuple(dv.sources) == F12.part_names("TW-1x")
    fv = F.deploy_view(decl, "full", "TW-1x")
    assert not any(isinstance(v, F.B0View) for v in fv.parts) and np.any(np.asarray(fv.t_code, dtype=np.int64) // TBL == 1)
    for key in ("q_fold", "q_inner", "q_gold_in_pool", "q_hop"):   # the same rows fitted, selected and scored
        assert np.array_equal(getattr(dv, key), getattr(fv, key)), key
    assert dv.qids == fv.qids and int(dv.t_code.size) < int(fv.t_code.size)
    with pytest.raises(SystemExit, match="not one of this file's views"):
        F.deploy_view(decl, "b1", "TW-1x")
    for fit in ("TW-2x", "TW-8x"):
        with pytest.raises(SystemExit, match="not one of this file's fits"):
            F.deploy_view(decl, "b0", fit)
    assert not (stops / "hard_stops.json").exists()
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.run_unit(_fitter(decl, "full"), "b0", "TW-1x", 0, _quiet)   # a full view is not a b0 unit's
    assert "a b0 view holds a type outside bucket 0" in _stopped(stops)
    (stops / "hard_stops.json").unlink()
    monkeypatch.setattr(F, "B0View", L9.View)   # a filter that keeps every type
    with pytest.raises(SystemExit, match="HARD STOP"):
        F.deploy_view(decl, "b0", "TW-1x")
    assert "a b0 view holds a type outside bucket 0" in _stopped(stops) and L9.View is F.ORIGINAL_VIEW


def _handmade(arrays: dict):
    """level 9's View's __init__ replaced by one that sets a handmade nb layout, for B0View's own filter."""
    def init(self, d, family, check=True):
        self.dir, self.family = d, family
        for key, val in arrays.items():
            setattr(self, key, np.asarray(val, dtype=np.int64))
        self.n_q = int(self.q_types.size)
    return init


def test_the_b0_view_on_a_handmade_layout_keeps_a_query_with_no_bucket_0_type_empty_and_refuses_a_non_block_layout(tmp_path, monkeypatch):
    T = TBL
    good = {"t_code": [5, 9, T + 2, T + 7, T + 1, T + 4, 3, T + 3], "t_size": [2, 1, 3, 1, 2, 2, 1, 4],
            "t_gold": [1, 0, 2, 0, 0, 1, 1, 0], "t_first": [0, 2, 3, 6, 7, 9, 11, 12], "q_types": [4, 2, 2, 0],
            "e_node": np.arange(16) * 10, "e_count": np.arange(16) + 1}
    monkeypatch.setattr(L9.View, "__init__", _handmade(good))
    v = F.B0View(tmp_path, "nb")
    assert v.t_code.tolist() == [5, 9, 3] and v.t_size.tolist() == [2, 1, 1] and v.t_gold.tolist() == [1, 0, 1]
    assert v.t_first.tolist() == [0, 2, 11] and v.q_types.tolist() == [2, 0, 1, 0] and v.q_entries.tolist() == [3, 0, 1, 0]
    assert v.type_ptr.tolist() == [0, 2, 2, 3, 3] and v.entry_start.tolist() == [0, 0, 11, 0]
    node, cnt, tl = v.entries(0)
    assert node.tolist() == [0, 10, 20] and cnt.tolist() == [1, 2, 3] and tl.tolist() == [0, 0, 1]
    assert [x.size for x in v.entries(1)] == [0, 0, 0] and [x.size for x in v.entries(3)] == [0, 0, 0]   # no bucket-0 type
    assert v.entries(2)[0].tolist() == [110] and v.reach(2, 3).tolist() == [110]
    assert v.reach(0, 9).tolist() == [20] and v.reach(0, T + 2).size == 0 and v.reach(1, T + 1).size == 0
    gap = {**good, "t_first": [0, 5, 2, 6, 7, 9, 11, 12]}   # query 0's two bucket-0 types are not one block
    monkeypatch.setattr(L9.View, "__init__", _handmade(gap))
    with pytest.raises(SystemExit, match="the bucket-0 entries of a query are not one block"):
        F.B0View(tmp_path, "nb")


def test_a_query_with_no_bucket_0_type_keeps_the_null_type_only_in_a_b0_unit(tmp_path, monkeypatch, stops, one_thread):
    torch.use_deterministic_algorithms(True)
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16}, n_r=20, every=4)
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    fx = _fitter(decl, "b0")
    n_r = len(ids["eval"])
    none = set(np.flatnonzero(np.asarray(fx.data.q_types[:n_r]) == 0).tolist())
    assert none and len(none) < n_r
    _arrays, _flog, cap = F11.fit_and_capture(fx, 0, 0, _quiet)
    assert set(np.asarray(cap["score_q"]).tolist()) == set(range(n_r))
    for q, lp in zip(cap["score_q"], cap["lp_score"]):
        assert np.asarray(lp).size == int(fx.data.q_types[q]) + 1   # the query's types, then the null type
        if int(q) in none:
            assert np.array_equal(np.asarray(lp), [0.0])                 # the null type only, with all the mass
    out, _flog = F.run_unit(_fitter(decl, "b0"), "b0", "TW-1x", 0, _quiet)
    ptr = out["score_ptr"]
    for i, q in enumerate(out["q"]):
        if int(q) in none:
            shift = np.asarray(out["score_dsh"][ptr[i]:ptr[i + 1]], dtype=np.float64) - fx.data.z(int(q), 0)
            assert np.ptp(shift) < 1e-6 and int(out["argmax"][i]) == -1   # the twin's order: the null type reaches no node
    assert not (stops / "hard_stops.json").exists()


# ── a unit ───────────────────────────────────────────────────────────────────


def test_a_full_unit_is_level_13s_run_unit_and_a_b0_unit_level_12s_run_unit_on_the_b0_view_bit_for_bit(tmp_path, monkeypatch, stops, one_thread):
    torch.use_deterministic_algorithms(True)
    _root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    units = {}
    for view, ref_run in (("full", F13.run_unit), ("b0", F12.run_unit)):
        out, flog = F.run_unit(_fitter(decl, view), view, "TW-1x", 0, _quiet)
        ref, rlog = ref_run(_fitter(decl, view), "TW-1x", 0, _quiet)
        assert set(out) == set(ref)
        for key in ref:
            assert out[key].dtype == ref[key].dtype and np.array_equal(out[key], ref[key]), (view, key)
        assert set(flog) - set(rlog) == {"l14_view", "l14_fit"} and (flog["l14_view"], flog["l14_fit"]) == (view, "TW-1x")
        assert {k: v for k, v in flog.items() if k in rlog and k != "timing"} == {k: v for k, v in rlog.items() if k != "timing"}
        units[view] = (out, flog)
    (full, flog_f), (b0, flog_b) = units["full"], units["b0"]
    assert {"metrics_b1d", "score_b1d"} <= set(full) and not {"metrics_b1d", "score_b1d"} & set(b0)
    assert flog_f["l13_fit"] == "TW-1x" and "l13_fit" not in flog_b and flog_b["l12_fit"] == flog_f["l12_fit"] == "TW-1x"
    assert np.array_equal(full["q"], b0["q"]) and np.array_equal(full["q"], np.arange(len(ids["eval"])))
    assert flog_b["walk_types"] < flog_f["walk_types"] and flog_b["rows_by_part"] == flog_f["rows_by_part"]
    assert not np.array_equal(b0["score_dsh"], full["score_dsh"])
    assert not (stops / "hard_stops.json").exists()


def test_the_scored_rows_golds_the_other_seeds_twins_and_the_stored_metrics_never_reach_a_units_scores(tmp_path, monkeypatch, stops, one_thread):
    """arms: each arm reads z(T_k), the query embedding, the relation text and the nb family's compiled types and counts,
    nothing else. r's golds are read only by its metrics."""
    torch.use_deterministic_algorithms(True)
    _root, decl, _ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16})
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 2)
    names = F12.part_names("TW-1x")
    rng = np.random.default_rng(0)
    for view in F.VIEWS:
        first, f1 = F.run_unit(_fitter(decl, view), view, "TW-1x", 0, _quiet)
        with F11.rebound(L9, View=F.B0View if view == "b0" else F.ORIGINAL_VIEW):
            parts = [F.eval_view(decl), *(F12.part_view(decl, p) for p in names[1:])]
        ev = parts[0]
        is_gold, t_gold = np.array(ev.is_gold, copy=True), np.array(ev.t_gold, copy=True)
        for q in range(ev.n_q):
            sl = ev.nodes(q)
            is_gold[sl] = rng.permutation(is_gold[sl])   # the same count, other nodes
            rows = ev.type_rows(q)
            t_gold[rows] = rng.integers(0, 3, size=t_gold[rows].size)
        ev.is_gold, ev.t_gold = is_gold, t_gold
        for v in parts:
            twin = np.array(v.twin_score, copy=True)
            twin[:, 1:] = rng.normal(size=twin[:, 1:].shape).astype(twin.dtype)   # seeds 1 and 2: unit k = 0 never reads them
            v.twin_score = twin
            v.q_metrics = np.full(np.shape(v.q_metrics), np.nan)                   # the stored T and G metrics
        fx = F10.make_fitter(F12.with_roles(F11.MultiView(parts, names)), T8._rel(), F11.LEVEL10_FIT)
        again, f2 = F.run_unit(fx, view, "TW-1x", 0, _quiet)
        keys = ("q", "score_dens", "score_cov", "score_dsh", "score_ptr", "argmax") + (("score_b1d",) if view == "full" else ())
        for key in keys:
            assert np.array_equal(first[key], again[key]), (view, key)
        choices = CHOICES + (B1D_CHOICES if view == "full" else ())
        assert [f1[key] for key in choices] == [f2[key] for key in choices] and f1["kept_round"] == f2["kept_round"]
        assert not np.array_equal(first["metrics_dsh"], again["metrics_dsh"])   # the metrics do read the golds
    assert not (stops / "hard_stops.json").exists()


# ── the statistics and the readings on toy cases ─────────────────────────────


def test_the_bands_come_in_the_declared_order_and_rho_is_the_share_of_the_gain():
    assert F.band(1.3, [1.05, 1.5], True) == "L14_ABOVE_GNN"
    assert F.band(0.8, [0.6, 1.0], True) == "L14_HIGH"
    assert F.band(0.8, [0.4, 1.2], True) == "L14_MID"
    assert F.band(0.2, [0.0, 0.45], True) == "L14_LOW"
    assert F.band(0.2, [0.0, 0.55], True) == "L14_MID"
    assert F.band(None, None, False) == "NOT_READ" and F.band(0.9, [0.8, 1.0], False) == "NOT_READ"
    rng = np.random.default_rng(0)
    T = rng.random((40, 3, 3))
    G = T + 0.5
    W = L0.boot_weights(40)
    _d, dens, readable = L8.denominators(T, G, W)
    assert readable == list(L8.RETRIEVAL)
    half = F.read_arm(T + 0.25, T, G, dens, readable, W)
    assert abs(half["rho_bar"]["point"] - 0.5) < 1e-12 and all(abs(half["rho"][m]["point"] - 0.5) < 1e-12 for m in L8.RETRIEVAL)
    assert half["band"] == "L14_MID" and all(half["gap_to_gnn"][m]["flag"] == "BELOW_GNN" for m in L8.RETRIEVAL)
    assert F.read_arm(T + 0.45, T, G, dens, readable, W)["band"] == "L14_HIGH"
    assert F.read_arm(T + 0.6, T, G, dens, readable, W)["band"] == "L14_ABOVE_GNN"
    assert F.read_arm(T + 0.05, T, G, dens, readable, W)["band"] == "L14_LOW"
    _d, dens0, none = L8.denominators(T, T, W)   # no gain: nothing is readable
    assert none == [] and F.read_arm(T + 0.1, T, T, dens0, none, W)["band"] == "NOT_READ"
    arms = {"a": F.read_arm(T + 0.25, T, G, dens, readable, W), "b": F.read_arm(T + 0.45, T, G, dens, readable, W)}
    c = F.paired(arms, "b", "a", readable)
    assert c["of"] == "rho_bar(b) - rho_bar(a)" and abs(c["point"] - 0.4) < 1e-12 and c["ci"][0] <= c["point"] <= c["ci"][1]
    assert F.paired(arms, "b", "a", []) == {"of": "rho_bar(b) - rho_bar(a)", "point": None, "ci": None}


def _arm(flags) -> dict:
    return {"gap_to_gnn": {m: {"flag": f} for m, f in zip(L8.RETRIEVAL, flags)}}


def test_reaches_gnn_is_the_first_read_arm_with_no_readable_metric_below_the_gnn_with_its_views_rows():
    tr = {f"{v}/{f}": {"fit": {"total": n - (v == "b0")}, "inner": {"total": 10}} for v in F.VIEWS for f, n in (("TW-1x", 100), ("TW-4x", 400))}
    every = list(L8.RETRIEVAL)
    below = _arm(["BELOW_GNN"] * 3)
    arms = {"TW-1x-dsh": _arm(["BELOW_GNN", None, None]), "TW-1x-b1d": _arm([None, "BELOW_GNN", None]),
            "TW-1x-b0": _arm(["BELOW_GNN", None, "BELOW_GNN"]), "TW-4x-dsh": _arm([None, None, "BEATS_GNN"]),
            "TW-4x-b1d": _arm([None] * 3), "TW-4x-b0": _arm([None] * 3)}
    assert F.reaches_gnn(arms, every, tr) == {"arm": "TW-4x-dsh", "fit": "TW-4x", "view": "full", "fit_rows": 400, "inner_rows": 10}
    assert F.reaches_gnn(arms, ["recall@5"], tr) == {"arm": "TW-1x-b1d", "fit": "TW-1x", "view": "full", "fit_rows": 100, "inner_rows": 10}
    first_b0 = {**arms, "TW-1x-dsh": below, "TW-1x-b1d": below, "TW-1x-b0": _arm([None] * 3)}
    assert F.reaches_gnn(first_b0, every, tr) == {"arm": "TW-1x-b0", "fit": "TW-1x", "view": "b0", "fit_rows": 99, "inner_rows": 10}
    assert F.reaches_gnn({a: below for a in arms}, every, tr) is None
    assert F.reaches_gnn(arms, [], tr) is None
    assert [F.unit_key(a) for a in F.READ_ARMS] == ["full/TW-1x", "full/TW-1x", "b0/TW-1x", "full/TW-4x", "full/TW-4x", "b0/TW-4x"]


def test_the_interpretation_lists_every_entry_that_applies_in_the_declared_order():
    every = list(L8.RETRIEVAL)
    flat = {c: {"ci": [-0.1, 0.1]} for c in F.CONTRASTS}
    hflat = {f"hop={h}": {c: {"ci": [-0.1, 0.1]} for c in F.HOP_CONTRASTS} for h in F.HOPS}

    def ci(**kw):
        return {**flat, **{c: {"ci": v} for c, v in kw.items()}}

    def hci(**kw):   # hop1=[lo, hi]: the hop's b0_adds interval
        out = copy.deepcopy(hflat)
        for key, v in kw.items():
            out[f"hop={key[-1]}"]["b0_adds"]["ci"] = v
        return out

    cases = [
        (("L14_HIGH", _arm([None] * 3), every, flat, hflat, 0.9), ["l14_high", "matched_not_below_gnn"]),
        (("L14_MID", _arm(["BELOW_GNN", None, None]), every,
          ci(b0_adds=[0.01, 0.1], b0_adds_4x=[-0.2, -0.01], b1d_adds=[0.01, 0.1], b1d_adds_4x=[-0.2, -0.01], data_4x=[0.01, 0.2],
             data_4x_b1d=[0.02, 0.3], data_4x_b0=[0.01, 0.1], bucket_ceiling=[0.01, 0.05], ceiling_gap_b0=[0.1, 0.3]),
          hci(hop1=[-0.2, -0.01], hop2=[0.01, 0.2], hop3=[0.02, 0.3]), 0.4),
         ["l14_mid", "matched_below_gnn", "b0_adds", "b0_hurts_4x", "b1d_adds", "b1d_hurts_4x", "data_adds_4x", "data_adds_4x_b1d",
          "data_adds_4x_b0", "b0_ceiling_binds", "chain_not_identified", "b0_hurts_hop1", "b0_adds_hop2", "b0_adds_hop3"]),
        (("L14_LOW", _arm([None] * 3), every,
          ci(b0_adds=[-0.3, -0.01], b0_adds_4x=[0.01, 0.02], b1d_adds=[-0.3, -0.01], b1d_adds_4x=[0.01, 0.02],
             ceiling_gap=[0.1, 0.3], ceiling_gap_b0=[0.1, 0.3]),
          hci(hop1=[0.01, 0.2], hop3=[-0.3, -0.1]), 0.6),
         ["l14_low", "matched_not_below_gnn", "b0_hurts", "b0_adds_4x", "b1d_hurts", "b1d_adds_4x", "b0_adds_hop1", "b0_hurts_hop3"]),
        (("L14_ABOVE_GNN", _arm(["BEATS_GNN"] * 3), every, ci(ceiling_gap=[0.1, 0.3]), hflat, 0.3),   # ceiling_gap is not the b0 one
         ["l14_above_gnn", "matched_not_below_gnn"]),
        (("NOT_READ", _arm([None] * 3), [], {c: {"ci": None} for c in F.CONTRASTS},
          {h: {c: {"ci": None} for c in F.HOP_CONTRASTS} for h in hflat}, float("nan")), []),
    ]
    for args, want in cases:
        got = F.interpretation(*args)
        assert got == want and [x for x in F.INTERPRETATION if x in got] == got


def test_the_b0_view_anchor_counts_the_bucket_0_types_and_entries_per_row():
    T = TBL
    nb = SimpleNamespace(n_q=3, q_types=np.array([3, 2, 1]), q_entries=np.array([6, 5, 2]),
                         t_code=np.array([1, 2, T + 1, T + 3, T + 4, 7]), t_size=np.array([1, 2, 3, 2, 3, 2]))
    a = F.b0_view_anchor(nb, np.array([1, 2, 2]))
    assert a["all"] == pytest.approx({"rows": 3, "types": 2.0, "entries": 13 / 3, "types_b0": 1.0, "entries_b0": 5 / 3,
                                      "no_b0_type_share": 1 / 3})
    assert a["hop=1"] == {"rows": 1, "types": 3.0, "entries": 6.0, "types_b0": 2.0, "entries_b0": 3.0, "no_b0_type_share": 0.0}
    assert a["hop=2"] == {"rows": 2, "types": 1.5, "entries": 3.5, "types_b0": 0.5, "entries_b0": 1.0, "no_b0_type_share": 0.5}
    assert a["hop=3"] == {"rows": 0, "types": None, "entries": None, "types_b0": None, "entries_b0": None, "no_b0_type_share": None}


def test_the_identical_code_check_needs_every_script_in_one_committed_set_and_this_module_in_no_score_or_check_job(monkeypatch):
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "a" * 64)
    a = "a" * 64
    score = {L8.SCRIPT_REL: a, L9.SCRIPT_REL: a, L10.SCRIPT_REL: a, L11.SCRIPT_REL: a, P12.SCRIPT_REL: a, P13.SCRIPT_REL: a, P.SCRIPT_REL: a}
    fit = {**score, F10.SCRIPT_REL: a, F11.SCRIPT_REL: a, F12.SCRIPT_REL: a, F13.SCRIPT_REL: a, F.SCRIPT_REL: a}
    good = {"score/r/shard_0of12.json": score, "meta/r": score, "check/r": score, "fit/full/TW-1x/k0": fit, "fit/b0/TW-4x/k2": fit,
            "repeat": fit, "read": fit}
    assert F.code_problems(good, "HEAD") == []
    assert F.code_problems({**good, "meta/r": {}}, "HEAD") == []   # a toy sidecar's meta records no module
    for rel in (F13.SCRIPT_REL, F12.SCRIPT_REL, F.SCRIPT_REL):
        less = {k: v for k, v in fit.items() if k != rel}
        assert any("are not in every fit" in p for p in F.code_problems({**good, "fit/b0/TW-4x/k2": less}, "HEAD")), rel
    assert any("are not in every fit" in p for p in F.code_problems({**good, "read": score}, "HEAD"))
    assert any(p.startswith(f"check/r: {P.SCRIPT_REL}") for p in F.code_problems({**good, "check/r": {L8.SCRIPT_REL: a}}, "HEAD"))
    for job in ("score/r/shard_1of12.json", "meta/r", "check/r"):
        assert any("is in a score, assemble or check job's record" in p for p in F.code_problems({**good, job: fit}, "HEAD")), job
    assert any("2 different" in p for p in F.code_problems({**good, "read": {**fit, F.SCRIPT_REL: "b" * 64}}, "HEAD"))
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: "c" * 64)
    assert any("not the file at" in p for p in F.code_problems(good, "HEAD"))


# ── the stages end to end on synthetic sidecars ──────────────────────────────


def test_fit_repeat_read_doc_and_file_run_end_to_end(tmp_path, monkeypatch, stops, one_thread):
    root, decl, ids = _env(tmp_path, monkeypatch, {"select": 12, "fit": 16, "x1": 10, "x2": 10, "x3": 10}, n_r=30, every=5)
    cfg = tmp_path / "mp_approx_l14.yaml"
    shutil.copyfile(P.CONFIG, cfg)   # the file stage appends its run record to a copy
    monkeypatch.setattr(F, "CONFIG", cfg)
    monkeypatch.setattr(F, "RECORD", tmp_path / "l14" / "record.json")
    monkeypatch.setattr(F, "DOC", tmp_path / "MP_APPROX_L14.md")
    monkeypatch.setattr(F, "SEEDS", (0,))
    monkeypatch.setattr(L8, "load_rel_emb", lambda decl: T8._rel())
    monkeypatch.setattr(L8, "M_EPOCHS", 2)
    monkeypatch.setattr(F10, "EM_ROUNDS", 3)
    monkeypatch.setattr(L8, "committed_lf_sha", lambda commit, rel: L0.lf_sha256(ROOT / rel))
    for view in F.VIEWS:
        for fit in F.FITS:
            F.stage_fit(decl, view, fit, 0, _quiet)
    said = []
    F.stage_fit(decl, "b0", "TW-1x", 0, said.append)   # a unit already written is skipped on a restart
    assert any("not refitted" in s for s in said)
    F.stage_repeat(decl, _quiet)
    rd = F.stage_read(decl, _quiet)
    n_r = len(ids["eval"])
    assert rd["primary"] == "TW-1x-b0" and rd["reading"] == rd["arms"]["TW-1x-b0"]["band"] and rd["reading"] in F.BANDS
    assert rd["carve"] == "r" and rd["queries"] == n_r
    assert rd["readable_metrics"] and all(v["point"] is not None for v in rd["contrasts"].values())   # the toy is read, not skipped
    assert list(rd["arms"]) == [*F.READ_ARMS, *F.REFERENCES] and list(rd["contrasts"]) == list(F.CONTRASTS)
    for c_name, (a, b) in F.CONTRASTS.items():
        assert abs(rd["contrasts"][c_name]["point"] - (rd["arms"][a]["rho_bar"]["point"] - rd["arms"][b]["rho_bar"]["point"])) < 1e-12
    assert set(rd["strata"]) == set(rd["hop_contrasts"]) == {"hop=1", "hop=2", "hop=3"}
    toy_hops = set(np.unique(L9.View(F.DATA, "nb").q_hop).tolist())
    assert toy_hops == {1, 2}   # level 8's toy qtypes have one and two hops, so hop 3 is an empty stratum here
    assert rd["strata"]["hop=3"]["readable_metrics"] == [] and not any(x.endswith("hop3") for x in rd["interpretation"])
    for h, hc in rd["hop_contrasts"].items():
        s = rd["strata"][h]
        assert list(hc) == list(F.HOP_CONTRASTS) and (s["queries"] > 0) == (int(h[-1]) in toy_hops)
        for c in F.HOP_CONTRASTS:
            a, b = F.CONTRASTS[c]
            if s["readable_metrics"]:
                assert abs(hc[c]["point"] - (s["arms"][a]["rho_bar"]["point"] - s["arms"][b]["rho_bar"]["point"])) < 1e-12
            else:
                assert hc[c]["point"] is None and hc[c]["ci"] is None
    assert rd["repeat"]["bit_identical"] and "REPEAT_DIFFERS" not in rd["flags"] and set(rd["flags"]) <= set(F.FLAGS)
    assert [x for x in F.INTERPRETATION if x in rd["interpretation"]] == rd["interpretation"]
    an = rd["anchors"]
    assert set(an) == set(decl["quantities"]["anchors"])
    assert set(an["level9_anchors"]) == set(an["carve_metrics"]) == {"r", *P.L12_CARVES}
    ex = an["exchangeability"]
    assert ex["r"] == an["carve_metrics"]["r"] and set(ex["level12"]) == {"dev", *P12.CARVES} and set(ex["level13_dev"]) == {"twin", "gnn"}
    checks = {c: json.loads((root / "carves" / c / "check.json").read_text(encoding="utf-8")) for c in P.L12_CARVES}
    tr = an["training_rows"]
    assert set(tr) == {f"{v}/{f}" for v in F.VIEWS for f in F.FITS}
    for fit, carves in F.FITS.items():
        assert tr[f"b0/{fit}"] == tr[f"full/{fit}"]   # the same rows fitted and selected in both views
        t = tr[f"full/{fit}"]
        assert t["fit"]["total"] == sum(checks[c]["rows_with_gold_in_pool"]["all"] for c in carves)
        assert t["inner"]["total"] == t["inner"]["select"] == checks["select"]["rows_with_gold_in_pool"]["all"]
        assert t["fit"]["eval"] == t["fit"]["select"] == t["inner"]["eval"] == 0
    assert set(an["agreement"]) == set(an["theta"]) == set(an["unseen_sequences"]) == set(tr)
    for a, ref in F.GAP_SPLIT.items():
        g = an["gap_split"][a]
        assert g["reference"] == ref and g["right_pairs"] + g["wrong_pairs"] == n_r
    for a, c_name in (("TW-1x-b1d", "ceiling_gap"), ("TW-1x-b0", "ceiling_gap_b0"), ("TW-4x-b0", "ceiling_gap_b0_4x")):
        g = an["gap_split"][a]
        if g["right"] is not None:   # the split adds up to the ceiling gap
            assert abs(g["right"]["all"] + g["wrong"]["all"] - rd["contrasts"][c_name]["point"]) < 1e-9
    assert set(an["beta_choices"]) == set(an["grid_edges"]) == set(F.READ_ARMS)
    assert all(sum(v.values()) == 1 for v in an["beta_choices"].values())
    assert set(an["b1d_moved"]) == set(F.FITS) and all(v["changed_scored_mean"] > 0 for v in an["b1d_moved"].values())
    b0v = an["b0_view"]
    nb = L9.View(F.DATA, "nb")
    assert set(b0v) == {"all", "hop=1", "hop=2", "hop=3"} and b0v["all"]["rows"] == n_r
    assert 0 < b0v["all"]["no_b0_type_share"] < 1 and b0v["all"]["types_b0"] < b0v["all"]["types"] == pytest.approx(float(np.mean(nb.q_types)))
    reach = an["reaches_gnn"]
    assert reach is None or (reach["arm"] in F.READ_ARMS and reach["fit_rows"] == tr[F.unit_key(reach["arm"])]["fit"]["total"])
    for view in F.VIEWS:
        for fit in F.FITS:
            npz, js = F.unit_paths(view, fit, 0)
            with np.load(npz) as z:
                assert np.array_equal(z["q"], np.arange(n_r)) and ("metrics_b1d" in z.files) == (view == "full")
            flog = json.loads(js.read_text(encoding="utf-8"))
            assert (flog["l14_view"], flog["l14_fit"], flog["l12_fit"], flog["fit"], flog["parts"], flog["k"], flog["fold"]) == (
                view, fit, fit, "NB-set", list(F12.part_names(fit)), 0, 0)
            assert (flog.get("l13_fit") == fit) == (view == "full")
            assert flog["scored_queries"] == n_r and len(flog["rounds"]) == 4 and flog["deterministic_algorithms"] is True
            assert {F.SCRIPT_REL, F13.SCRIPT_REL, F12.SCRIPT_REL, F11.SCRIPT_REL, F10.SCRIPT_REL, P.SCRIPT_REL} <= set(flog["module_sha256"])
    assert set(rd["code"]) >= {f"fit/{v}/{f}/k0" for v in F.VIEWS for f in F.FITS} | {"repeat", "check/r", "meta/r"}
    x3 = root / "carves" / "x3" / "check.json"
    saved = x3.read_bytes()
    x3.write_text(json.dumps({**json.loads(saved), "note": "changed after the read"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="was not made from"):
        F.stage_doc(_quiet)
    x3.write_bytes(saved)
    F.stage_doc(_quiet)
    body = F.DOC.read_text(encoding="utf-8")
    for phrase in ("TW-1x-b0", "label-matched", "It is never a deployable or selected model", "reaches_gnn", "NB-oracle-b0",
                   "It is also not the within-U_q oracle rho", "train-split rows", "b0_adds", "bucket_ceiling"):
        assert phrase in body, phrase
    assert not any(p in body.lower() for p in FORBIDDEN)
    rec = json.loads(F.RECORD.read_text(encoding="utf-8"))
    assert rec["phase"] == "MP_APPROX_L14" and rec["sources_sha256"]["read.json"] == L0.sha256_file(F.DATA / "read.json")
    F.stage_file("2026_10_02", "HEAD", _quiet, {"note": "a test"})
    filed = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    run = filed["run_record_mp_approx_l14_2026_10_02"]
    assert filed["status"] == "RUN" and run["note"] == "a test" and run["reading"] == rd["reading"] and run["terminal"] == "STOP_FOR_REVIEW"
    assert run["primary"] == "TW-1x-b0" and set(run["contrasts"]) == set(F.CONTRASTS) and set(run["training_rows"]) == set(tr)
    assert run["train_split_rows_scored"] == n_r and run["dev_rows_read"] is False and run["stored_metrics_checked"] is False
    assert set(run["hop_contrasts"]) == {"hop=1", "hop=2", "hop=3"} and run["b0_view"] == pytest.approx(b0v["all"])
    with pytest.raises(SystemExit, match="exists"):
        F.stage_file("2026_10_02", "HEAD", _quiet)
    assert not (stops / "hard_stops.json").exists()


def test_the_stages_refuse_the_wrong_machine_and_wrong_arguments(stops):
    for argv in (["--stage", "fit", "--view", "b0", "--fit", "TW-1x", "--k", "0"], ["--stage", "repeat"], ["--stage", "read"],
                 ["--stage", "doc", "--host"], ["--stage", "file", "--host"], ["--stage", "fit", "--host"],
                 ["--stage", "fit", "--host", "--fit", "TW-4x", "--k", "0"], ["--stage", "fit", "--host", "--view", "b0", "--k", "0"],
                 ["--stage", "fit", "--host", "--view", "full", "--fit", "TW-1x"],
                 ["--stage", "fit", "--host", "--view", "b1", "--fit", "TW-1x", "--k", "0"],
                 ["--stage", "fit", "--host", "--view", "b0", "--fit", "TW-2x", "--k", "0"],
                 ["--stage", "fit", "--host", "--view", "b0", "--fit", "TW-1x", "--k", "3"],
                 ["--stage", "repeat", "--host", "--view", "b0"], ["--stage", "repeat", "--host", "--fit", "TW-1x"],
                 ["--stage", "read", "--host", "--k", "0"], ["--stage", "doc", "--date", "2026_10_02"],
                 ["--stage", "read", "--host", "--commit", "abc"], ["--stage", "file"],
                 ["--stage", "score", "--host"], ["--stage", "check", "--host"]):
        with pytest.raises(SystemExit):
            F.main(argv)
    assert not (stops / "hard_stops.json").exists()
    assert math.isinf(F.BETAS[-1])
