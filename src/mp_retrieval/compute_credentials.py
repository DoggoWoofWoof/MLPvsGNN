"""Private compute-account loading for cloud launchers.

Credential files are local and git-ignored. The loader is intentionally small:
secrets are exported through SDK environment variables and never placed on a
command line, in a run manifest, or in a tracked file.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import yaml


ROTATE_SIGNALS = (
    "quota",
    "credit",
    "insufficient",
    "exhausted",
    "payment required",
    "rate limit",
    "unauthorized",
    "forbidden",
    "billing",
    "spend limit",
    "resourceexhausted",
)


@dataclass(frozen=True)
class ModalCredential:
    name: str
    token_id: str | None = None
    token_secret: str | None = None
    profile: str | None = None

    def environment(self) -> dict[str, str]:
        values = {
            "MODAL_TOKEN_ID": self.token_id,
            "MODAL_TOKEN_SECRET": self.token_secret,
            "MODAL_PROFILE": self.profile,
        }
        return {key: str(value) for key, value in values.items() if value}


def load_modal_pool(path: Path | None = None) -> list[ModalCredential]:
    configured = path or Path(
        os.environ.get("MP_RETRIEVAL_COMPUTE_CONFIG", "configs/compute.local.yaml")
    )
    if not configured.exists():
        return [ModalCredential(name="ambient")]
    payload: dict[str, Any] = yaml.safe_load(configured.read_text(encoding="utf-8")) or {}
    entries = payload.get("modal", []) or []
    return [
        ModalCredential(
            name=str(entry.get("name", f"modal-{idx}")),
            token_id=entry.get("token_id"),
            token_secret=entry.get("token_secret"),
            profile=entry.get("profile"),
        )
        for idx, entry in enumerate(entries)
    ] or [ModalCredential(name="ambient")]


def select_modal_pool(
    pool: list[ModalCredential], account: str | None
) -> list[ModalCredential]:
    if account is None:
        return pool
    if account.isdigit() and int(account) < len(pool):
        return [pool[int(account)]]
    selected = [credential for credential in pool if credential.name == account]
    if not selected:
        raise ValueError(f"Unknown Modal account {account!r}")
    return selected


def should_rotate(output: str) -> bool:
    lowered = output.lower()
    return any(signal in lowered for signal in ROTATE_SIGNALS)
