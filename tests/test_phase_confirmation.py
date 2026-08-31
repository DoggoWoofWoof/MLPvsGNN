import json
from pathlib import Path

import pytest

from scripts.run_phase_confirmation import _screen_seed_zero


def test_screen_seed_zero_reuse_rejects_test_access(tmp_path: Path) -> None:
    path = tmp_path / "screen.json"
    path.write_text(
        json.dumps(
            {
                "status": "PHASE_SCREEN_VALIDATION_ONLY_COMPLETE",
                "dataset": "toy",
                "axis": "feature_mask",
                "rate": 0.5,
                "data_fingerprint_sha256": "abc",
                "screen_contract": {"test_metrics_computed": True, "training_seed": 0},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reuse contract"):
        _screen_seed_zero(
            path,
            dataset="toy",
            axis="feature_mask",
            rate=0.5,
            fingerprint="abc",
        )
