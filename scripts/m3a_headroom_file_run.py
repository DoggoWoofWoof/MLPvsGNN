"""File the run record of the headroom protocol into configs/m3a_headroom.yaml.

Run once, after all six datasets have been produced by scripts/m3a_headroom.py:
flips status DECLARED_NOT_RUN -> RUN and appends a run_record block carrying
the sha256 of every output file, the served freeze, the git commit and runner
sha the outputs report, and the wall seconds. The block is appended as text so
the declared body above it is untouched byte for byte.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "m3a_headroom.yaml"
OUT_DIR = ROOT / "outputs" / "m3a" / "headroom"
ORDER = ["metaqa", "squad", "musique", "hotpotqa", "2wiki", "webqsp"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    text = CONFIG.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    if cfg["status"] != "DECLARED_NOT_RUN":
        raise SystemExit(f"status is {cfg['status']}; the run record is filed once")
    missing = [n for n in ORDER if not (OUT_DIR / f"{n}.json").exists()]
    if missing:
        raise SystemExit(f"outputs missing for {missing}")
    lines = ["", "run_record:", "  status: RUN", f"  filed_utc: \"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\"", "  datasets:"]
    freezes, commits, runners = set(), set(), set()
    for name in ORDER:
        d = json.loads((OUT_DIR / f"{name}.json").read_text(encoding="utf-8"))
        p = d["provenance"]
        freezes.add(p["freeze_RECORD_SHA256"]); commits.add(p["git_commit"]); runners.add(p["runner_sha256"])
        js, npz = OUT_DIR / f"{name}.json", OUT_DIR / f"{name}_per_query.npz"
        lines += [
            f"    {name}:",
            f"      eval_split: {d['eval_split']}",
            f"      queries_scored: {d['populations']['eval']['queries']}",
            f"      zero_gold_excluded: {d['populations']['eval']['zero_gold_excluded']}",
            f"      rows: {len(d['rows'])}",
            f"      depth_rows: {len(d['R4_depth_curve'])}",
            f"      utc: \"{p['utc']}\"",
            f"      wall_seconds: {p['seconds_total']}",
            f"      json: {{path: outputs/m3a/headroom/{name}.json, sha256: {sha(js)}}}",
            f"      npz: {{path: outputs/m3a/headroom/{name}_per_query.npz, sha256: {sha(npz)}}}",
        ]
    combined = OUT_DIR / "HEADROOM.json"
    lines += [
        f"  combined: {{path: outputs/m3a/headroom/HEADROOM.json, sha256: {sha(combined)}}}",
        f"  freeze_RECORD_SHA256_seen: [{', '.join(sorted(freezes))}]",
        f"  git_commit_seen: [{', '.join(sorted(commits))}]",
        f"  runner_sha256_seen: [{', '.join(sorted(runners))}]",
        "  gpu_seconds: 0",
        "  report: docs/M3A_HEADROOM_REPORT.md (rendered by scripts/m3a_headroom_report.py from these files)",
        "  outputs_are_gitignored: true -- the sha256s above are the committed record of the numbers",
        "",
    ]
    new = text.replace("status: DECLARED_NOT_RUN\n", "status: RUN\nstatus_at_declaration: DECLARED_NOT_RUN\n", 1)
    if new == text:
        raise SystemExit("status line not found")
    CONFIG.write_text(new.rstrip("\n") + "\n" + "\n".join(lines), encoding="utf-8", newline="\n")
    yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    print("filed run_record; status RUN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
