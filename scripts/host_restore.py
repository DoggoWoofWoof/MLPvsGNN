"""Rebuild the GPU host's data on a new host from the hub (docs/HOST_RECOVERY.md). Systems only: it moves and verifies
bytes, and every step is an rx job driven from the laptop, with the hub tokens chosen by name and never sent to the host.

The order leaves every file at its newest archived version:
  1. the mirror workspace: the tag mirror-spared (KK9895/mpr-host-archive, the 97 files the first mirror took from
     another project's host workspace), then the six datasets' downloads from the two mpr-six-mirror repos with
     scripts/host_mirror_hf.py (urls on the laptop, download and verify on the host); its survey record
     (outputs/host_mirror_six/host_sources.json) goes along so the links skip exactly the files mirror-spared restores;
  2. every workspace tag of KK9895/mpr-host-archive and JGY9895/mpr-host-archive (the anchors, the o-* waves and the
     o-inc-* increments that outputs/host_ops/host_sync.py writes), newest manifest first, into the workspace without
     --overwrite: a file an older tag holds at other bytes is a conflict and stays at the newer version.
The tag t0 (the archive tool's test files) is skipped. Progress goes to outputs/host_archive/restore_progress.json, so a
rerun resumes after the last finished step (--redo starts over).

  python scripts/host_restore.py plan                       # list the steps; nothing runs
  python scripts/host_restore.py run [--only mirror|ws] [--redo]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import host_archive_hf as HA  # noqa: E402

ARCHIVES = (("KK9895/mpr-host-archive", "mpr"), ("JGY9895/mpr-host-archive", "mpr-archive2"))
MIRROR_GROUPS = ((("squad", "musique"), "Swastik9895/mpr-six-mirror", "freebase"),
                 (("metaqa",), "Swastik9895/mpr-six-mirror", "freebase"),
                 (("webqsp",), "Swastik9895/mpr-six-mirror", "freebase"),
                 (("hotpotqa",), "KK9895/mpr-six-mirror", "mpr"),
                 (("2wiki",), "KK9895/mpr-six-mirror", "mpr"))
SKIP_TAGS = {"t0"}
PROGRESS = HA.OUT / "restore_progress.json"
MIRROR_OUT = ROOT / "outputs" / "host_mirror_six"


def manifests(repo: str, token_name: str) -> list[dict]:
    """Every manifests/<tag>.json of an archive repo: tag, prefix, created_utc, files, bytes."""
    from huggingface_hub import hf_hub_download
    api = HA.hf_api(token_name)
    tok = HA.stored_token(token_name)
    out = []
    for e in api.list_repo_tree(repo, repo_type="dataset", path_in_repo="manifests"):
        if not e.path.endswith(".json"):
            continue
        p = hf_hub_download(repo, e.path, repo_type="dataset", token=tok)
        m = json.loads(Path(p).read_text(encoding="utf-8"))
        out.append({"repo": repo, "token_name": token_name, "tag": m["tag"], "prefix": m.get("prefix") or HA.REPO_PREFIX,
                    "created_utc": m.get("created_utc") or "", "files": len(m["files"]), "bytes": m["bytes"]})
    return out


def order_steps(tags: list[dict]) -> list[dict]:
    """mirror-spared, the five mirror downloads, then the workspace tags newest first."""
    steps = []
    spared = [t for t in tags if t["prefix"] == "mirror/" and t["tag"] not in SKIP_TAGS]
    for t in sorted(spared, key=lambda t: t["created_utc"]):
        steps.append({"key": f"tag:{t['repo']}:{t['tag']}", "kind": "tag", "dest": "../mirror", **t})
    for ds, repo, tname in MIRROR_GROUPS:
        steps.append({"key": "mirror:" + "_".join(ds), "kind": "mirror", "datasets": list(ds), "repo": repo,
                      "token_name": tname})
    ws = [t for t in tags if t["prefix"] == "ws/" and t["tag"] not in SKIP_TAGS]
    for t in sorted(ws, key=lambda t: t["created_utc"], reverse=True):
        steps.append({"key": f"tag:{t['repo']}:{t['tag']}", "kind": "tag", "dest": ".", **t})
    return steps


def rx(*args: str) -> str:
    p = subprocess.run([sys.executable, "tools/rx/rx.py", *args], cwd=ROOT, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        raise SystemExit(f"rx {' '.join(args[:3])} failed (rc={p.returncode}): {out[-800:]}")
    return out


def mirror_step(s: dict) -> str:
    ds, tag = s["datasets"], "_".join(s["datasets"])
    subprocess.run([sys.executable, "-u", "scripts/host_mirror_hf.py", "urls", "--datasets", *ds, "--repo", s["repo"],
                    "--token-name", s["token_name"]], cwd=ROOT, check=True)
    try:
        out = rx("run", "--name", f"mirror-dl-{tag.replace('_', '-')}", "--cpus", "4", "--mem", "8", "--shell", "cmd", "-w",
                 "--inputs", "outputs/host_mirror_six/manifest.json", "--inputs", "outputs/host_mirror_six/_urls.json",
                 "--inputs", "outputs/host_mirror_six/host_sources.json", "--",
                 f"python -u scripts/host_mirror_hf.py download --datasets {' '.join(ds)} --workers 8 && "
                 f"python -u scripts/host_mirror_hf.py verify --datasets {' '.join(ds)}")
    finally:
        (MIRROR_OUT / "_urls.json").unlink(missing_ok=True)                                # short-lived links stay nowhere
    print(out[-1200:], flush=True)
    rx("fetch", "--glob", f"outputs/host_mirror_six/download_{tag}.json", "--glob",
       f"outputs/host_mirror_six/verify_{tag}.json", "--overwrite")
    ver = json.loads((MIRROR_OUT / f"verify_{tag}.json").read_text(encoding="utf-8"))
    if ver.get("status") != "VERIFIED":
        raise SystemExit(f"mirror {tag}: verify says {ver.get('status')} (missing {len(ver.get('missing') or [])}, "
                         f"hash mismatch {len(ver.get('hash_mismatch') or [])}); stopping")
    return "VERIFIED"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("plan", "run"))
    ap.add_argument("--only", choices=("mirror", "ws"))
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    tags = [m for repo, tname in ARCHIVES for m in manifests(repo, tname)]
    steps = order_steps(tags)
    if a.only == "mirror":
        steps = [s for s in steps if s["kind"] == "mirror" or s.get("dest") == "../mirror"]
    elif a.only == "ws":
        steps = [s for s in steps if s.get("dest") == "."]
    done = {} if a.redo or not PROGRESS.exists() else HA.read_json(PROGRESS).get("done", {})
    for i, s in enumerate(steps, 1):
        what = (f"{s['repo']} {s['tag']} -> {s['dest']} ({s['files']} files, {s['bytes'] / 1e9:.2f} GB, {s['created_utc']})"
                if s["kind"] == "tag" else f"mirror {'+'.join(s['datasets'])} from {s['repo']}")
        print(f"{i:3d}. {'[done] ' if s['key'] in done else ''}{what}", flush=True)
    if a.mode == "plan":
        return
    for s in steps:
        if s["key"] in done:
            continue
        if s["kind"] == "tag":
            rec = HA.stage_drive_restore(s["repo"], s["tag"], [], s["token_name"], s["dest"], a.workers, 1, 1.0, "mpr-cpu",
                                         False)
            failed = [r["rel"] for r in rec["files"] if r["how"] not in ("download", "already_present", "conflict")]
            if failed:
                raise SystemExit(f"{s['tag']}: {len(failed)} files failed (first {failed[:3]}); stopping (a rerun resumes here)")
            n_conf = len(rec.get("conflicts") or [])
            status = f"{len(rec['files']) - n_conf} verified, {n_conf} left at a newer tag's version"
        else:
            status = mirror_step(s)
        done[s["key"]] = {"utc": HA.utc(), "status": status}
        HA.atomic_json(PROGRESS, {"done": done})
    print(f"restore: {len(done)} steps done", flush=True)


if __name__ == "__main__":
    main()
