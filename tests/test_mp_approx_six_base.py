"""configs/mp_approx_six_base.yaml#tests, work 1: the pins, the twin as the frozen make_model builds it, the ceiling and
the output paths. Nothing here fits a model or reads the CRAG package."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_six_base as SB  # noqa: E402


def test_declaration_parses_and_pins_hold():
    decl = SB.load_declaration()
    assert decl["work_1_twin_fits"]["parameters"] == SB.PARAMETERS
    assert SB.pin_problems(decl) == []


def test_pin_problems_names_a_changed_file(tmp_path):
    decl = SB.load_declaration()
    rel = "scripts/universal_v2_six.py"
    decl["inputs"]["frozen_code_lf"] = {rel: "0" * 64}
    decl["inputs"]["gnn_checkpoints"] = {}
    decl["inputs"]["caches"] = {}
    decl["inputs"]["contract"] = {}
    assert SB.pin_problems(decl) == [rel]


def test_twin_is_the_frozen_arm_with_the_pinned_parameters():
    cfg, cfg_m3b, _ = SB.V2.load_configs()
    inputs = SB.V2.model_inputs(cfg, cfg_m3b)
    model = SB.build_twin(inputs, None)
    assert type(model).__name__ == "UMLPv2Mix"
    assert SB.parameter_count(model) == 330955
    assert inputs["n_scalars"] == 129


def test_keys_and_paths_are_this_files():
    assert SB.fit_key(1) == "u_mlp_v2_mix__H128__six__s1"
    assert SB.FITS == ROOT / "outputs" / "mp_approx_six_base" / "fits"
    assert "universal_v2" not in SB.FITS.as_posix()
    assert SB.fit_key(0) != SB.U6.fit_key(0)


def _record(d: Path, seed: int, seconds: float) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{SB.fit_key(seed)}.json").write_text(json.dumps({"seconds": seconds}), encoding="utf-8")


def test_ceiling_projects_fifteen_hours_before_any_fit(tmp_path):
    g = SB.ceiling_guard(0, tmp_path)
    assert g["fit_hours_spent"] == 0 and g["projected_fit_hours"] == 15.0 and g["within_ceiling"]


def test_ceiling_refuses_a_fit_that_would_breach(tmp_path):
    _record(tmp_path, 0, 20 * 3600)
    _record(tmp_path, 1, 14 * 3600)
    with pytest.raises(SystemExit):
        SB.ceiling_guard(2, tmp_path)   # 34 h spent + 20 h projected > 45
    assert json.loads((tmp_path / "ceiling_breach.json").read_text(encoding="utf-8"))["within_ceiling"] is False


def test_ceiling_allows_within(tmp_path):
    _record(tmp_path, 0, 10 * 3600)
    g = SB.ceiling_guard(1, tmp_path)
    assert g["fit_hours_spent"] == 10 and g["projected_fit_hours"] == 10 and g["within_ceiling"]


def test_an_existing_record_is_not_repeated(tmp_path, monkeypatch):
    monkeypatch.setattr(SB, "FITS", tmp_path)
    (tmp_path / f"{SB.fit_key(0)}.json").write_text(json.dumps({"seconds": 1.0, "marker": 7}), encoding="utf-8")
    monkeypatch.setattr(SB, "verify_pins", lambda decl: (_ for _ in ()).throw(AssertionError("must not run")))
    assert SB.stage_fit(0, log=lambda s: None)["marker"] == 7


def test_fit_refuses_another_thread_count():
    with pytest.raises(SystemExit):
        SB.main(["--stage", "fit", "--seed", "0", "--threads", "4"])


def test_module_shas_skips_a_module_whose_file_is_not_absolute(monkeypatch):
    # amendment 1: torch.ops and torch.classes carry __file__ = "_ops.py" and "_classes.py", which resolve under the
    # repository and name no file; seed 0's first process raised FileNotFoundError there while building its record
    import torch

    _ = torch.ops.aten
    assert any(isinstance(getattr(m, "__file__", None), str) and getattr(m, "__file__") == "_ops.py" for m in list(sys.modules.values()))
    monkeypatch.chdir(ROOT)
    shas = SB.module_shas()
    assert "_ops.py" not in shas and "_classes.py" not in shas
    assert "scripts/mp_approx_six_base.py" in shas and all((ROOT / rel).is_file() for rel in shas)
