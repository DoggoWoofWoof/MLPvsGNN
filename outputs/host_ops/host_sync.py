"""Keep the laptop and the hub in step with the GPU host, so that the experiments can restart on another host.

Each cycle (--every-min, default 60):
  ops       the session scratchpad's small files (the feeder, its queue and state, the jobs files, the wave scripts, the
            logs) are copied to outputs/host_ops/scratchpad/, where a Temp clean-up cannot reach them. A file is copied
            only if the scratchpad's copy is newer, so a feeder restarted from outputs/host_ops/scratchpad/ (after a host
            loss) is never overwritten by the old session's files; --scratch PATH reads another folder, --scratch none
            turns the copy off
  status    the host's job table -> outputs/host_job_logs/status_latest.json; each finished job's log (its last --log-kb KB)
            -> outputs/host_job_logs/<id>.log, once (at most --logs-per-cycle a cycle, newest first)
  results   result-like files under the host's outputs/ and logs/ (json md txt log yaml yml csv tsv jsonl up to --file-mb,
            models pt ckpt pth safetensors up to --model-mb) that the laptop lacks or holds at another size. A running
            job's files and anything modified in the last 30 min wait for a later cycle; a delete list's files and
            outputs/host_archive/ are never fetched. rx's own fetch, by exact path: a local file rx did not write is left
            alone and reported once. Filed stages' files (ROUTED: the levels, six-base, mq design, the universal-v2 pilot)
            go under C:/Users/Swastik/Desktop/MPR_Host_Archive/host/, never into the repo's outputs/. At most --cycle-gb a
            cycle; models stop at --floor-gb free, everything at --hard-floor-gb.
Every --archive-every-h hours (default 6), and on the first cycle:
  coverage  each file under the host's outputs/ and logs/ is classed: hub (a VERIFIED tag's plan holds it at the same size
            and mtime), records (outputs/host_archive, outputs/host_cleanup), delete_list, in_flux (a running job's, or
            modified in the last 30 min) or uncovered. With --mirror (first cycle) the mirror workspace is checked file by
            file against the three repos that hold it -> outputs/host_ops/coverage_latest.json
  archive   the uncovered files -> JGY9895/mpr-host-archive as tag o-inc-<stamp> (scripts/host_archive_hf.py's drive:
            signed links, no credential on the host), unless an archive or restore job is active; then the ops folder ->
            the same repo at ops/host_ops.tar.gz, uploaded from the laptop.
Tokens are chosen by name (mpr-archive2 = JGY9895, mpr = KK9895, freebase = Swastik9895); their values go to
huggingface_hub only and are never printed or written. Systems only: nothing is deleted anywhere and no host file changes.

  python outputs/host_ops/host_sync.py                     # the loop (log: outputs/host_ops/host_sync.log)
  python outputs/host_ops/host_sync.py --once --dry        # one cycle, report only
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path("C:/Users/Swastik/Desktop/message-passing-retrieval")
OPS = ROOT / "outputs" / "host_ops"
LOGS = ROOT / "outputs" / "host_job_logs"
ARCH = ROOT / "outputs" / "host_archive"
STATE = OPS / "sync_state.json"
SCRATCH = Path("C:/Users/Swastik/AppData/Local/Temp/claude/C--Users-Swastik-Desktop-message-passing-retrieval/"
               "57907381-9ef9-442c-8627-a25d24264fe8/scratchpad")
sys.path.insert(0, str(ROOT / "tools" / "rx"))
sys.path.insert(0, str(ROOT / "scripts"))
import rx as RX  # noqa: E402
import host_archive_hf as HA  # noqa: E402

REPO, TOKEN = "JGY9895/mpr-host-archive", "mpr-archive2"
MIRROR_REPOS = (("Swastik9895/mpr-six-mirror", "freebase", "final_canonical/"),
                ("KK9895/mpr-six-mirror", "mpr", "final_canonical/"),
                ("KK9895/mpr-host-archive", "mpr", "mirror/CRAG/data/final_canonical/"))
MIRROR_PREFIX = "CRAG/data/final_canonical/"
OPS_EXT = {".py", ".txt", ".json", ".log", ".yaml", ".yml", ".sh", ".md", ".b64", ".sha", ".csv", ".ps1", ".toml",
           ".tsv", ".jsonl", ".html"}
RES_EXT = {".json", ".md", ".txt", ".log", ".yaml", ".yml", ".csv", ".tsv", ".jsonl"}
MOD_EXT = {".pt", ".ckpt", ".pth", ".safetensors"}
ACTIVE = ("running", "queued", "waiting", "starting", "launching")
FINISHED = ("done", "failed", "cancelled", "lost", "error")
RECORDS = ("outputs/host_archive/", "outputs/host_cleanup/")
# Filed stages whose laptop outputs stay untouched (levels, six-base, mq design, the universal-v2 pilot): their host files
# are fetched under HOSTCOPY instead of the repo's outputs/.
ROUTED = ("outputs/mp_approx_l", "outputs/mp_approx_six_base", "outputs/mp_approx_mq_design", "outputs/universal_v2",
          "outputs/graph_context_pilot", "outputs/pilot3")
HOSTCOPY = Path("C:/Users/Swastik/Desktop/MPR_Host_Archive/host")
QUIET_S = 1800


def log(msg: str) -> None:
    print(f"{time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def jdump(obj) -> bytes:
    return json.dumps(obj, indent=1, default=str).encode("utf-8")


def remote():
    proj = RX.Project(RX.find_project(str(ROOT)))
    return proj, RX.Remote(RX.load_host(proj.host))


def job_name(jid: str) -> str | None:
    m = re.match(r"^\d{6}-\d{6}-(.+)-[0-9a-f]{4}$", jid or "")
    return m.group(1) if m else None


# ── ops ──────────────────────────────────────────────────────────────────────


def ops_mirror() -> tuple[int, int]:
    n = b = 0
    dst = OPS / "scratchpad"
    if SCRATCH is None or not SCRATCH.is_dir() or SCRATCH.resolve() == dst.resolve():
        return 0, 0
    for p in SCRATCH.rglob("*"):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix.lower() not in OPS_EXT or p.name.endswith(".tmp"):
            continue
        st = p.stat()
        if st.st_size > 5_000_000:
            continue
        q = dst / p.relative_to(SCRATCH)
        try:
            if q.stat().st_mtime >= st.st_mtime:              # newer-only: never older bytes over a live copy
                continue
        except FileNotFoundError:
            pass
        q.parent.mkdir(parents=True, exist_ok=True)
        tmp = q.with_name(q.name + ".tmp")
        shutil.copy2(p, tmp)
        os.replace(tmp, q)
        n += 1
        b += st.st_size
    return n, b


# ── status and logs ──────────────────────────────────────────────────────────


def job_table(proj, r) -> list[dict]:
    return r.call({"op": "status", "project": proj.name, "limit": 500}, retry_for=300)["jobs"]


def active_names(jobs: list[dict]) -> list[str]:
    return sorted({job_name(j.get("id")) for j in jobs if j.get("state") in ACTIVE} - {None})


def save_logs(jobs: list[dict], per_cycle: int, tail_kb: int, dry: bool) -> tuple[int, int]:
    have = {p.stem for p in LOGS.glob("*.log")}
    todo = sorted((j["id"] for j in jobs if j.get("state") in FINISHED and j.get("id") not in have), reverse=True)
    n = 0
    for jid in todo[:per_cycle]:
        if dry:
            n += 1
            continue
        try:
            p = subprocess.run([sys.executable, "tools/rx/rx.py", "logs", jid, "--tail", str(tail_kb)], cwd=ROOT,
                               capture_output=True, timeout=600)
        except subprocess.TimeoutExpired:
            continue
        if p.returncode == 0:
            atomic_write(LOGS / f"{jid}.log", p.stdout)
            n += 1
    return n, len(todo)


# ── host listings ────────────────────────────────────────────────────────────


def listing(proj, r, ws: str, include: list[str]) -> list:
    res = r.call({"op": "manifest", "project": proj.name, "ws": ws, "include": include, "exclude": [], "hash": False},
                 retry_for=600)
    if res.get("truncated"):
        raise SystemExit(f"the host listing of {ws} was truncated")
    return res["files"]


def delete_lists() -> set[str]:
    out = set()
    for p in sorted((ROOT / "outputs" / "host_cleanup").glob("delete_*.json")):
        doc = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("files"), list) and "dirs" in doc:    # a list, not a run record
            out |= {row[0] for row in doc["files"] if isinstance(row, list) and row}
    return out


def in_flux(rel: str, mtime_ns: int, names: list[str], now_ns: int) -> bool:
    return now_ns - mtime_ns < QUIET_S * 1e9 or any(("/" + n) in rel for n in names)


def dest_root(rel: str) -> Path:
    return HOSTCOPY if rel.startswith(ROUTED) else ROOT


def on_laptop(rel: str, size: int) -> bool:
    for base in (ROOT, HOSTCOPY):
        try:
            if (base / rel).stat().st_size == size:
                return True
        except (FileNotFoundError, NotADirectoryError):
            pass
    return False


# ── results to the laptop ────────────────────────────────────────────────────


def results(proj, r, rows: list, names: list[str], a, state: dict) -> dict:
    dele = delete_lists()
    now = time.time_ns()
    known = {tuple(x) for x in state.get("conflicts", [])}
    cand = []
    for rel, size, mtime_ns, _ in rows:
        ext = Path(rel).suffix.lower()
        if rel.startswith("outputs/host_archive/") or rel in dele or (rel, size) in known:
            continue
        cap = a.file_mb if ext in RES_EXT else a.model_mb if ext in MOD_EXT else None
        if cap is None or size > cap * 1e6 or in_flux(rel, mtime_ns, names, now) or on_laptop(rel, size):
            continue
        cand.append((rel, size))
    cand.sort(key=lambda x: (Path(x[0]).suffix.lower() in MOD_EXT, x[1]))
    free = shutil.disk_usage(ROOT).free
    take, tot, held = [], 0, 0
    for rel, size in cand:
        floor = a.floor_gb if Path(rel).suffix.lower() in MOD_EXT else a.hard_floor_gb
        if tot + size > a.cycle_gb * 1e9 or free - tot - size < floor * 1e9:
            held += 1
            continue
        take.append(rel)
        tot += size
    out = {"candidates": len(cand), "candidate_gb": round(sum(s for _, s in cand) / 1e9, 3), "taken": len(take),
           "taken_gb": round(tot / 1e9, 3), "held_for_later": held, "laptop_free_gb": round(free / 1e9, 1)}
    if a.dry or not take:
        return out
    fetched = have = 0
    conflicts = []
    for base in (ROOT, HOSTCOPY):
        mine = [x for x in take if dest_root(x) == base]
        for i in range(0, len(mine), 300):
            chunk = mine[i:i + 300]
            res = r.call({"op": "manifest", "project": proj.name, "ws": "ws", "include": [glob.escape(x) for x in chunk],
                          "exclude": []}, retry_for=600)
            if base == HOSTCOPY:
                HOSTCOPY.mkdir(parents=True, exist_ok=True)
            f, h, c = RX.fetch_files(proj, r, res["files"], ws="ws", into=None if base == ROOT else str(base), quiet=True)
            fetched, have = fetched + f, have + h
            size_of = {x[0]: x[1] for x in res["files"]}
            conflicts += [[rel, size_of.get(rel)] for rel in c]
    if conflicts:
        state["conflicts"] = sorted({tuple(x) for x in state.get("conflicts", [])} | {tuple(x) for x in conflicts})
        atomic_write(OPS / "fetch_conflicts.json", jdump({"utc": HA.utc(), "conflicts": state["conflicts"]}))
    out.update(fetched=fetched, already_here=have, new_conflicts=len(conflicts))
    return out


# ── coverage and archive ─────────────────────────────────────────────────────


def hub_index() -> dict:
    """(src_ws, rel) -> (bytes, mtime_ns, repo, tag, sha256) for every file of every VERIFIED tag's plan."""
    idx = {}
    for c in sorted(ARCH.glob("commit_*.json")):
        com = json.loads(c.read_text(encoding="utf-8"))
        plan = ARCH / f"plan_{com.get('tag')}.json"
        if com.get("status") != "VERIFIED" or not plan.exists():
            continue
        pl = json.loads(plan.read_text(encoding="utf-8"))
        for f in pl["files"]:
            idx[(pl.get("src_ws") or "ws", f["rel"])] = (f["bytes"], f["mtime_ns"], com["repo"], com["tag"], f["sha256"])
    return idx


def mirror_coverage(proj, r) -> dict:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import get_stored_tokens
    man = json.loads((ROOT / "outputs" / "host_mirror_six" / "manifest.json").read_text(encoding="utf-8"))
    want = {f["rel"]: f for f in man["files"]}
    host = {rel: size for rel, size, _, _ in listing(proj, r, "mirror", ["**"])}
    toks = get_stored_tokens()
    held = defaultdict(list)
    for repo, tname, prefix in MIRROR_REPOS:
        api = HfApi(token=toks[tname])
        for e in api.list_repo_tree(repo, repo_type="dataset", recursive=True, path_in_repo=prefix.rstrip("/")):
            if e.__class__.__name__ != "RepoFile" or not e.path.startswith(prefix):
                continue
            lfs = getattr(e, "lfs", None)
            held[e.path[len(prefix):]].append((repo, e.size, lfs.sha256 if lfs else None, getattr(e, "blob_id", None)))
    cov, unc, by_repo = 0, [], defaultdict(lambda: [0, 0])
    for rel, f in want.items():
        hits = [repo for repo, size, sha, blob in held.get(rel, [])
                if size == f["bytes"] and (sha == f["sha256"] if sha else blob == f.get("git_sha1"))]
        if hits:
            cov += 1
            by_repo[hits[0]][0] += 1
            by_repo[hits[0]][1] += f["bytes"]
        else:
            unc.append([rel, f["bytes"]])
    extra = sorted([rel, s] for rel, s in host.items() if not (rel.startswith(MIRROR_PREFIX) and rel[len(MIRROR_PREFIX):] in want))
    missing = sorted(rel for rel in want if (MIRROR_PREFIX + rel) not in host)
    return {"manifest_files": len(want), "manifest_gb": round(man["bytes"] / 1e9, 3), "on_hub": cov,
            "by_repo": {k: {"files": n, "gb": round(b / 1e9, 3)} for k, (n, b) in by_repo.items()},
            "not_on_hub": unc, "host_files": len(host), "host_files_not_in_manifest": extra,
            "manifest_files_missing_on_host": missing}


def coverage(proj, r, rows: list, names: list[str], mirror: bool) -> dict:
    dele = delete_lists()
    hub = hub_index()
    now = time.time_ns()
    cls = defaultdict(lambda: [0, 0])
    bydir = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    unc, maybe = [], {}
    for rel, size, mtime_ns, _ in rows:
        h = hub.get(("ws", rel))
        if h and h[0] == size and h[1] == mtime_ns:
            k = "hub"
        elif h and h[0] == size and not rel.startswith(RECORDS) and rel not in dele and not in_flux(rel, mtime_ns, names, now):
            maybe[rel] = (size, h[4])               # same size, another mtime (a restored or touched file): sha256 decides
            continue
        elif rel.startswith(RECORDS):
            k = "records"
        elif rel in dele:
            k = "delete_list"
        elif in_flux(rel, mtime_ns, names, now):
            k = "in_flux"
        else:
            k = ("uncovered_changed" if h else "uncovered") + ("_on_laptop" if on_laptop(rel, size) else "")
            unc.append([rel, size])
        cls[k][0] += 1
        cls[k][1] += size
        d = "/".join(rel.split("/")[:2])
        bydir[d][k][0] += 1
        bydir[d][k][1] += size
    todo = sorted(maybe)
    for i in range(0, len(todo), 300):
        res = r.call({"op": "manifest", "project": proj.name, "ws": "ws", "include": [glob.escape(x) for x in todo[i:i + 300]],
                      "exclude": [], "hash": True}, retry_for=900)
        got = {x[0]: x[3] for x in res["files"]}
        for rel in todo[i:i + 300]:
            size, sha = maybe[rel]
            k = "hub" if got.get(rel) == sha else "uncovered_changed" + ("_on_laptop" if on_laptop(rel, size) else "")
            if k != "hub":
                unc.append([rel, size])
            cls[k][0] += 1
            cls[k][1] += size
            d = "/".join(rel.split("/")[:2])
            bydir[d][k][0] += 1
            bydir[d][k][1] += size
    out = {"utc": HA.utc(), "ws_files": len(rows), "ws_gb": round(sum(x[1] for x in rows) / 1e9, 3),
           "classes": {k: {"files": n, "gb": round(b / 1e9, 3)} for k, (n, b) in sorted(cls.items())},
           "by_dir": {d: {k: {"files": n, "gb": round(b / 1e9, 3)} for k, (n, b) in v.items()} for d, v in sorted(bydir.items())},
           "running_names": names, "uncovered": unc}
    if mirror:
        out["mirror"] = mirror_coverage(proj, r)
    return out


def archive_busy(jobs: list[dict]) -> bool:
    """An archive or restore job is active on the host, or the jgy_waves watcher has not logged its exit."""
    if any(j.get("state") in ACTIVE and (job_name(j.get("id")) or "").startswith(("archive-", "restore-")) for j in jobs):
        return True
    w = SCRATCH / "archive_waves.log"
    if w.is_file():
        s = w.read_text(encoding="utf-8", errors="replace")
        return s.rfind("jgy waves exit") < s.rfind("jgy waves (watcher)")
    return False


def ops_to_hub(stamp: str) -> int:
    from huggingface_hub import HfApi
    from huggingface_hub.utils import get_stored_tokens
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for p in sorted((OPS / "scratchpad").rglob("*")):
            if p.is_file() and not p.name.endswith(".tmp"):
                tf.add(p, arcname=p.relative_to(OPS).as_posix())
        for p in (OPS / "host_sync.py", OPS / "coverage_latest.json", OPS / "sync_state.json",
                  LOGS / "status_latest.json", ROOT / ".rx" / "jobs.json"):
            if p.is_file():
                tf.add(p, arcname=p.relative_to(ROOT).as_posix())
    data = buf.getvalue()
    api = HfApi(token=get_stored_tokens()[TOKEN])
    api.upload_file(path_or_fileobj=data, path_in_repo="ops/host_ops.tar.gz", repo_id=REPO, repo_type="dataset",
                    commit_message=f"ops snapshot {stamp}")
    return len(data)


# ── the loop ─────────────────────────────────────────────────────────────────


def cycle(a, state: dict, first: bool) -> None:
    t0 = time.time()
    n, b = ops_mirror()
    log(f"ops: {n} files copied ({b / 1e6:.1f} MB)")
    proj, r = remote()
    jobs = job_table(proj, r)
    names = active_names(jobs)
    if not a.dry:
        atomic_write(LOGS / "status_latest.json", jdump({"utc": HA.utc(), "jobs": jobs}))
    k, pending = save_logs(jobs, a.logs_per_cycle, a.log_kb, a.dry)
    log(f"status: {len(jobs)} jobs, {len(names)} active; logs saved {k} of {pending} pending")
    rows = listing(proj, r, "ws", ["outputs/**", "logs/**"])
    res = results(proj, r, rows, names, a, state)
    log("results: " + json.dumps(res))
    due = first or time.time() - state.get("last_archive", 0) >= a.archive_every_h * 3600
    if due and not a.no_archive:
        rows = listing(proj, r, "ws", ["outputs/**", "logs/**"])
        cov = coverage(proj, r, rows, names, mirror=a.mirror and first)
        atomic_write(OPS / "coverage_latest.json", jdump(cov))
        log("coverage: " + json.dumps(cov["classes"]))
        if "mirror" in cov:
            m = cov["mirror"]
            log(f"mirror: {m['on_hub']}/{m['manifest_files']} manifest files on the hub; {len(m['not_on_hub'])} not; "
                f"{len(m['host_files_not_in_manifest'])} host files outside the manifest; "
                f"{len(m['manifest_files_missing_on_host'])} missing on the host")
        unc = cov["uncovered"]
        if a.dry:
            log(f"archive (dry): {len(unc)} files, {sum(s for _, s in unc) / 1e9:.2f} GB would go to {REPO}")
        elif archive_busy(job_table(proj, r)):
            log("archive: an archive or restore job is active; next cycle")
        else:
            ok = True
            if unc:
                tag = "o-inc-" + time.strftime("%Y%m%d-%H%M")
                log(f"archive {tag}: {len(unc)} files, {sum(s for _, s in unc) / 1e9:.2f} GB -> {REPO}")
                try:
                    HA.stage_drive(REPO, tag, [glob.escape(x) for x, _ in unc], [], TOKEN, workers=6, cpus=1, mem=0.5,
                                   per_commit=200, env="mpr-cpu")
                    com = json.loads((ARCH / f"commit_{tag}.json").read_text(encoding="utf-8"))
                    ok = com.get("status") == "VERIFIED"
                    log(f"archive {tag}: {com.get('status')}, {com.get('archived')} files")
                except (SystemExit, Exception) as e:  # noqa: BLE001
                    ok = False
                    log(f"archive {tag} stopped: {type(e).__name__}: {str(e)[:300]}")
            try:
                nb = ops_to_hub(time.strftime("%Y-%m-%d %H:%M"))
                log(f"ops snapshot -> {REPO}/ops/host_ops.tar.gz ({nb / 1e6:.1f} MB)")
            except Exception as e:  # noqa: BLE001
                ok = False
                log(f"ops snapshot failed: {type(e).__name__}: {str(e)[:300]}")
            if ok:
                state["last_archive"] = time.time()
    state["last_cycle"] = time.time()
    if not a.dry:
        atomic_write(STATE, jdump(state))
    log(f"cycle done in {time.time() - t0:.0f} s")


def main() -> None:
    global SCRATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--every-min", type=float, default=60)
    ap.add_argument("--archive-every-h", type=float, default=6)
    ap.add_argument("--cycle-gb", type=float, default=2.0)
    ap.add_argument("--floor-gb", type=float, default=15.0)
    ap.add_argument("--hard-floor-gb", type=float, default=5.0)
    ap.add_argument("--file-mb", type=float, default=200)
    ap.add_argument("--model-mb", type=float, default=500)
    ap.add_argument("--logs-per-cycle", type=int, default=80)
    ap.add_argument("--log-kb", type=int, default=256)
    ap.add_argument("--max-h", type=float, default=24 * 14)
    ap.add_argument("--mirror", action="store_true", help="check the mirror workspace against the hub on the first cycle")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--scratch", default=str(SCRATCH),
                    help="folder whose ops files are copied to outputs/host_ops/scratchpad ('none': no copy)")
    a = ap.parse_args()
    SCRATCH = None if a.scratch.lower() == "none" else Path(a.scratch)
    state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
    t_start, first = time.time(), True
    while True:
        t0 = time.time()
        try:
            cycle(a, state, first)
        except (SystemExit, Exception) as e:  # noqa: BLE001
            log(f"cycle failed: {type(e).__name__}: {str(e)[:400]}")
        first = False
        if a.once or time.time() - t_start > a.max_h * 3600:
            break
        time.sleep(max(60.0, a.every_min * 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
