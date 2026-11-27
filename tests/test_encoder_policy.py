"""The defined-encoder policy (docs/ING1_INGESTION_AND_STAGE_REGISTRY.md, amendment 2): no module reads the older encoder's cached lists or the frozen source unless it is listed with a reason."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _policy():
    spec = importlib.util.spec_from_file_location("encoder_policy", ROOT / "scripts" / "ingest" / "encoder_policy.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_defined_encoder_is_registered():
    pol = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))["encoder_policy"]
    assert pol["defined"] == "e3v"
    assert pol["comparison"] == "frozen"


def test_no_unlisted_module_reads_the_older_encoder():
    mod = _policy()
    listed = json.loads((ROOT / "configs" / "legacy_encoder_modules.json").read_text(encoding="utf-8"))["modules"]
    found = mod.scan()
    assert [m for m in found if m not in listed] == []
    assert [m for m in listed if m not in found] == []
    assert all(v.get("reason") for v in listed.values())


def test_the_amended_walk_takes_its_seeds_from_the_defined_encoder():
    src = (ROOT / "outputs" / "mp_unified" / "zjnt2.py").read_text(encoding="utf-8")
    assert 'sl["dense5"]' not in src.replace('sl["dense5"][qi]).tolist()', "")        # the older encoder's lists appear only in the information-only count
    assert "flat_topk(ctx.rows" in src
