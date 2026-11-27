"""The defined-encoder policy check (docs/ING1_INGESTION_AND_STAGE_REGISTRY.md, amendment 2).

The defined encoder of the main line is `configs/datasets.json` -> `encoder_policy.defined` (e3v). The older encoder's cached first-stage lists (`retrieval_cache/dense_top1000.npz`,
`splade_top1000.npz`, read through `zgp1.dev_slice`) and the `"frozen"` source of the pool engine are legacy: a module that reads them is listed in `configs/legacy_encoder_modules.json`
with the reason, and its numbers are labelled older-encoder reads. A module that is not listed must not read them: this check fails on it, and the test `tests/test_encoder_policy.py`
runs it, so that a new module cannot take a seed or a first-stage list from the older encoder unnoticed.

    python -E -P -B scripts/ingest/encoder_policy.py check     # exit 1 on an unlisted module or a listed module that no longer has a marker
    python -E -P -B scripts/ingest/encoder_policy.py scan      # prints every module with a marker (to build or refresh the list)

Read only: it reads source files and writes nothing."""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LIST = ROOT / "configs" / "legacy_encoder_modules.json"
DATASETS = ROOT / "configs" / "datasets.json"
SCAN_DIRS = ("outputs/mp_unified", "outputs/host_ops", "scripts", "src")
MARKERS = {
    "cached_lists": re.compile(r"(dense_top1000|splade_top1000)"),
    "frozen_source": re.compile(r"""(first\s*=\s*["']frozen["']|load_source\([^)]*["']frozen["']|["']frozen["']\s*\)\s*#\s*source)"""),
    "dev_slice": re.compile(r"\bdev_slice\("),
}


def scan():
    out = {}
    for base in SCAN_DIRS:
        for p in sorted((ROOT / base).rglob("*.py")):
            if p.resolve() == Path(__file__).resolve() or "__pycache__" in p.parts:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
            hit = sorted(k for k, rx in MARKERS.items() if rx.search(txt))
            if hit:
                out[p.relative_to(ROOT).as_posix()] = hit
    return out


def check():
    pol = json.loads(DATASETS.read_text(encoding="utf-8")).get("encoder_policy")
    if not pol or not pol.get("defined"):
        print("configs/datasets.json has no encoder_policy.defined")
        return 1
    listed = json.loads(LIST.read_text(encoding="utf-8"))["modules"]
    found = scan()
    bad = [m for m in found if m not in listed]
    stale = [m for m in listed if m not in found]
    no_reason = [m for m, v in listed.items() if not v.get("reason")]
    for m in bad:
        print(f"UNLISTED: {m} reads the older encoder's lists or the frozen source ({found[m]}); use the defined encoder ({pol['defined']}) or list it with a reason")
    for m in stale:
        print(f"STALE: {m} is listed but has no marker any more")
    for m in no_reason:
        print(f"NO REASON: {m}")
    print(f"encoder policy: defined {pol['defined']}; {len(found)} modules with a marker, {len(listed)} listed, {len(bad)} unlisted, {len(stale)} stale")
    return 1 if (bad or stale or no_reason) else 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    if cmd == "scan":
        for m, h in scan().items():
            print(m, h)
        sys.exit(0)
    sys.exit(check())
