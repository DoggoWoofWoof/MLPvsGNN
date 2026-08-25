from pathlib import Path

from mp_retrieval.compute_credentials import (
    load_modal_pool,
    select_modal_pool,
    should_rotate,
)


def test_missing_compute_config_uses_ambient_credentials(tmp_path: Path) -> None:
    pool = load_modal_pool(tmp_path / "missing.yaml")
    assert len(pool) == 1
    assert pool[0].name == "ambient"
    assert pool[0].environment() == {}


def test_compute_config_is_selected_without_exposing_values(tmp_path: Path) -> None:
    config = tmp_path / "compute.yaml"
    config.write_text(
        "modal:\n"
        "  - name: first\n"
        "    token_id: example-id\n"
        "    token_secret: example-secret\n"
        "  - name: second\n"
        "    profile: lab\n",
        encoding="utf-8",
    )
    pool = load_modal_pool(config)
    assert select_modal_pool(pool, "second")[0].environment() == {"MODAL_PROFILE": "lab"}
    assert select_modal_pool(pool, "0")[0].name == "first"


def test_rotation_only_triggers_for_registered_failure_signals() -> None:
    assert should_rotate("Workspace spend limit reached")
    assert should_rotate("HTTP 429 rate limit")
    assert not should_rotate("Python assertion failed")
