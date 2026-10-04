"""Auto clean-up of mpr's host workspace, after cataloguing it (systems code; no science).

    python outputs/host_ops/auto_cleanup.py hub-index                      # laptop: the archive's VERIFIED plans -> index
    python outputs/host_ops/auto_cleanup.py run                            # host: catalogue + dry run (removes nothing)
    python outputs/host_ops/auto_cleanup.py run --delete --loop            # host: catalogue + clean-up every --every-h

`run` works in mpr's rx workspace on the host and walks its outputs/ and logs/ (junctions and links are not followed).
Every file is classed as
  keep        text (.py .json .md .txt .log .yaml .csv ...), anything under 1 MB, and everything under KEEP_DIRS (the
              feeder and host ops, the archive and clean-up records, job logs): never removed
  hub         the HF archive holds it byte for byte: a VERIFIED tag's plan lists it at this size and mtime
              (outputs/host_ops/hub_index.json, made on the laptop by `hub-index`); restore: scripts/host_restore.py
  regen       a build or cache that committed code rebuilds from the mirror (host_sync.py's REGEN pattern)
  smoke       a smoke test's output (a folder whose name starts with 'smoke')
  unarchived  anything else (results and models the archive does not hold yet): never removed
and as referenced when a queued or running mpr job names it or a folder above it (the feeder's unsent and running items
in hostfeed_items.txt / hostfeed_state.json, and the specs of mpr's active jobs in rx's active/ folder), or when it sits
under a folder that a queued or running job's script family reads without naming it (PINS: the look caches, anchors,
carves and six-base for the qd/lean families, the S6 builds and caches for chainscore), and as warm when it was read or
written within the age limit (NTFS last access, which this drive keeps, and mtime) or cold.

A file is removed only if it is hub, regen or smoke, unreferenced, and cold for the limit that the drive's free space
sets: free >= --ideal-gb (200): smoke older than --smoke-h and other files untouched for --idle-days (14); free below
--ideal-gb: untouched for --cold-days (3); free below --floor-gb (100): untouched for --urgent-days (1). Within a cycle
the largest go first, and removal stops once free space reaches --ideal-gb (smoke and the idle limit still apply). A
file that is open elsewhere (WinError 32) is skipped. Folders left empty are removed.

Each cycle writes the catalogue (outputs/host_ops/host_catalogue.json and .md: every folder three levels down, its
bytes by class, cold, referenced and removable, and why it is kept), the record of what went
(outputs/host_cleanup/deleted_auto_<stamp>.json, with each file's size, mtime and archive tag, so it can be restored;
dry runs overwrite outputs/host_cleanup/dry_run_auto_latest.json) and one line in outputs/host_ops/auto_cleanup.log.
Stdlib only; no credential and no network.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

GB = 1e9
TEXT_EXT = {".py", ".json", ".jsonl", ".md", ".txt", ".log", ".yaml", ".yml", ".csv", ".tsv", ".toml", ".sh", ".ps1",
            ".html", ".sha", ".b64", ".cfg", ".ini"}
KEEP_DIRS = ("outputs/host_ops/", "outputs/host_cleanup/", "outputs/host_archive/", "outputs/host_job_logs/",
             "outputs/host_checks/")
MIN_BYTES = 1 << 20
# host_sync.py's REGEN: builds and caches that committed code rebuilds from the mirror
REGEN = re.compile(r"^outputs/mp_unified/(lean/cs\d+e?-[a-z0-9]+(-[a-z0-9]+)*\.npz|lean/cs\d+[a-z]*tmp_[^/]+/|"
                   r"look/[^/]+/[^/]+/chunks/|cache/)|/__pycache__/")
SMOKE = re.compile(r"(^|/)smoke[^/]*/")
PATHTOK = re.compile(r"(?:outputs|logs)/[A-Za-z0-9_./@=+-]+")
# Folders a script family reads without naming them on its command line: pinned while such a job is queued or running.
_ANCHORS = ("outputs/mp_approx_hotpot_anchor/host/", "outputs/mp_approx_kb_anchor/host/",
            "outputs/mp_approx_2wiki_anchor/host/", "outputs/mp_approx_musique_anchor/host/")
_BASES = ("outputs/mp_approx_l12/", "outputs/mp_approx_six_base/", "outputs/m3b/csr/", "outputs/mp_approx_l16_design/look/")
PINS = (
    (re.compile(r"qd_gnn\d*\.py|qd_six\.py|qd_tdiag\.py"), ("outputs/mp_unified/look/", "outputs/mp_unified/qd/") + _BASES),
    (re.compile(r"lean_(?:host|mlp|read|fast|time)\d*\.py|pair9\.py|qfam9\.py|colshift10\.py|floor10\.py"),
     ("outputs/mp_unified/look/", "outputs/mp_unified/lean/") + _ANCHORS + _BASES),
    (re.compile(r"look_x_six\.py|mp_approx_\w+\.py|universal_v2_\w+\.py"),
     ("outputs/mp_unified/look/", "outputs/universal_v2/") + _ANCHORS + _BASES),
    (re.compile(r"chainscore\d+\.py|chainpop\d+\.py|cs_cache\.py|cs_dev\.py|cs21_rmdiag\d*\.py|reltype\d+\.py"),
     ("outputs/mp_unified/lean/", "outputs/mp_unified/cache/")),
)
# Why each folder is on the host (the catalogue's 'why' column); a folder not listed is reported as uncatalogued.
WHY = {
    "outputs/mp_unified/lean": "S6 chain builds cs18-cs27 (regen) and lean-MLP / S6 results; builds feed queued parts",
    "outputs/mp_unified/look": "per-graph look caches (features, twin z) read by the qd_gnn / lean jobs (W4, l3)",
    "outputs/mp_unified/cache": "cs_cache tensors of the S6 builds (regen; bit-identical to the builds)",
    "outputs/mp_unified/qd": "QD-GNN / W4 results and saved models",
    "outputs/mp_unified/smoke27": "S6 part 11 smoke builds (scratch)",
    "outputs/mp_approx_hotpot_anchor/host": "hotpot anchor looks (L13-L15, lean AW reads); HF tag anc-hotpot",
    "outputs/mp_approx_kb_anchor/host": "metaqa / webqsp anchor looks; HF tag anc-kb",
    "outputs/mp_approx_2wiki_anchor/host": "2wiki anchors and k-sweep models; HF tag anc-2wiki",
    "outputs/mp_approx_musique_anchor/host": "musique anchors; HF tag anc-musique",
    "outputs/mp_approx_l12/carves": "L12 carves fit/x1-x7 (the carve definitions later levels read); HF o-mp_approx_l12",
    "outputs/mp_approx_l12/metaqa": "L12 metaqa units; HF o-mp_approx_l12",
    "outputs/mp_approx_six_base": "six-base per-dataset features (anchors and looks start here); HF o-mp_approx_six_base",
    "outputs/mp_approx_l16_design/look": "L16 design looks; HF o-mp_approx_l16_design",
    "outputs/m3b/csr": "M3B CSR graphs (closed stage)",
    "outputs/universal_v2/cache": "universal-v2 caches (closed track)",
}


def now_s() -> float:
    return time.time()


def atomic_write(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


# ── laptop: the archive index ───────────────────────────────────────────────


def hub_index(arch: Path, out: Path) -> int:
    """Every file of every VERIFIED tag's plan under the host workspace ('ws') -> [rel, bytes, mtime_ns, tag]."""
    rows = {}
    for c in sorted(arch.glob("commit_*.json")):
        com = json.loads(c.read_text(encoding="utf-8"))
        plan = arch / f"plan_{com.get('tag')}.json"
        if com.get("status") != "VERIFIED" or not plan.exists():
            continue
        pl = json.loads(plan.read_text(encoding="utf-8"))
        if (pl.get("src_ws") or "ws") != "ws":
            continue
        for f in pl["files"]:
            rows[f["rel"]] = [f["rel"], f["bytes"], f["mtime_ns"], com["tag"]]
    atomic_write(out, json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "files": sorted(rows.values())}))
    return len(rows)


# ── host: references ────────────────────────────────────────────────────────


def feeder_refs(ws: Path) -> tuple[set[str], list[str], list[str]]:
    """Path tokens, names and commands of the feeder's unsent and running items."""
    items = ws / "outputs/host_ops/hostfeed_items.txt"
    state = ws / "outputs/host_ops/hostfeed_state.json"
    if not items.is_file():
        return set(), ["no hostfeed_items.txt"], []
    st = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    sent, term, dropped = st.get("sent", {}), st.get("terminal", {}), st.get("dropped", {})
    refs, names, cmds = set(), [], []
    for line in items.read_text(encoding="utf-8").split("\n"):
        if not line.startswith("line|"):
            continue
        f = line.split("|")
        if len(f) < 8:
            continue
        name = f[3]
        if name in dropped:
            continue
        if name in sent and sent[name] in term:
            continue
        names.append(name)
        refs.update(PATHTOK.findall(f[6].replace(";", " ") + " " + f[7]))
        cmds.append(f[7])
    return refs, names, cmds


def active_refs(rx_root: Path) -> tuple[set[str], list[str], list[str]]:
    """Path tokens, ids and spec texts of mpr's active jobs (rx's active/mpr--<id>.json records point to each job's
    folder, whose spec.json holds the command)."""
    refs, ids, cmds = set(), [], []
    d = rx_root / "active"
    for p in sorted(d.glob("mpr--*.json")) if d.is_dir() else []:
        try:
            rec = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            txt = (Path(rec["jd"]) / "spec.json").read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError, KeyError):
            continue
        ids.append(p.stem.split("--", 1)[1])
        txt = txt.replace("\\\\", "/").replace("\\", "/")
        refs.update(PATHTOK.findall(txt))
        cmds.append(txt)
    return refs, ids, cmds


def referenced(rel: str, refs: set[str], prefixes: list[str]) -> bool:
    if rel in refs:
        return True
    return any(rel.startswith(p) for p in prefixes)


# ── host: the walk ──────────────────────────────────────────────────────────


def walk(ws: Path):
    for top in ("outputs", "logs"):
        stack = [ws / top]
        while stack:
            d = stack.pop()
            try:
                it = os.scandir(d)
            except OSError:
                continue
            with it:
                for e in it:
                    try:
                        if e.is_symlink() or (hasattr(e, "is_junction") and e.is_junction()):
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append(Path(e.path))
                            continue
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    rel = Path(e.path).relative_to(ws).as_posix()
                    yield rel, st


def classify(rel: str, st, hub: dict) -> tuple[str, str | None]:
    ext = os.path.splitext(rel)[1].lower()
    if rel.startswith(KEEP_DIRS) or ext in TEXT_EXT or st.st_size < MIN_BYTES:
        return "keep", None
    h = hub.get(rel)
    if h and h[0] == st.st_size and h[1] == st.st_mtime_ns:
        return "hub", h[2]
    if SMOKE.search(rel):
        return "smoke", None
    if REGEN.search(rel):
        return "regen", None
    return "unarchived", None


def folder(rel: str) -> str:
    parts = rel.split("/")
    return "/".join(parts[:min(3, len(parts) - 1)])


def why_for(d: str) -> str:
    for k in sorted(WHY, key=len, reverse=True):
        if d == k or d.startswith(k + "/") or k.startswith(d + "/"):
            return WHY[k]
    return "uncatalogued"


def cycle(a, ws: Path) -> dict:
    t0 = now_s()
    du0 = shutil.disk_usage(str(ws))
    free_gb = du0.free / GB
    hub_p = ws / "outputs/host_ops/hub_index.json"
    hub = {}
    if hub_p.is_file():
        for rel, b, m, tag in json.loads(hub_p.read_text(encoding="utf-8"))["files"]:
            hub[rel] = (b, m, tag)
    r1, pending, c1 = feeder_refs(ws)
    r2, active, c2 = active_refs(Path(a.rx_root))
    refs = {x.rstrip("/.") for x in r1 | r2}
    pins = sorted({d for rx_, ds in PINS if any(rx_.search(c) for c in c1 + c2) for d in ds})
    prefixes = [x + "/" for x in refs] + pins
    if free_gb < a.floor_gb:
        tier, limit_d = "urgent", a.urgent_days
    elif free_gb < a.ideal_gb:
        tier, limit_d = "low", a.cold_days
    else:
        tier, limit_d = "ok", a.idle_days
    now = now_s()
    rows = defaultdict(lambda: defaultdict(int))
    cands = []
    for rel, st in walk(ws):
        cls, tag = classify(rel, st, hub)
        idle_d = (now - max(st.st_atime, st.st_mtime)) / 86400
        ref = referenced(rel, refs, prefixes)
        d = folder(rel)
        r = rows[d]
        r["files"] += 1
        r["bytes"] += st.st_size
        r[cls] += st.st_size
        if ref:
            r["referenced"] += st.st_size
        if idle_d >= a.cold_days:
            r["cold"] += st.st_size
        if cls == "keep" or cls == "unarchived" or ref:
            continue
        ok = (idle_d * 24 >= a.smoke_h) if cls == "smoke" else (idle_d >= limit_d)
        if ok:
            r["removable"] += st.st_size
            cands.append((st.st_size, rel, st.st_mtime_ns, cls, tag, round(idle_d, 2)))
    cands.sort(reverse=True)
    removed, failed, skipped_full = [], [], 0
    free = du0.free
    for size, rel, mtime_ns, cls, tag, idle_d in cands:
        if free / GB >= a.ideal_gb and cls != "smoke" and idle_d < a.idle_days:
            skipped_full += 1
            continue
        if not a.delete:
            removed.append([rel, size, mtime_ns, cls, tag, idle_d])
            free += size
            continue
        p = ws / rel
        try:
            st = p.stat()
            if st.st_size != size or st.st_mtime_ns != mtime_ns:
                failed.append([rel, size, "changed since the walk"])
                continue
            os.remove(p)
        except OSError as e:
            failed.append([rel, size, f"{type(e).__name__}: {e}"])
            continue
        removed.append([rel, size, mtime_ns, cls, tag, idle_d])
        free += size
        d = p.parent
        while d != ws and str(d).startswith(str(ws / "outputs")) and d != ws / "outputs":
            try:
                d.rmdir()
            except OSError:
                break
            d = d.parent
    du1 = shutil.disk_usage(str(ws))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    rec = {"mode": "delete" if a.delete else "dry_run", "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "tier": tier, "age_limit_days": limit_d, "free_gb_before": round(du0.free / GB, 2),
           "free_gb_after": round(du1.free / GB, 2) if a.delete else round(free / GB, 2),
           "removed": len(removed), "removed_gb": round(sum(x[1] for x in removed) / GB, 3),
           "by_class_gb": {c: round(sum(x[1] for x in removed if x[3] == c) / GB, 3) for c in ("hub", "regen", "smoke")},
           "failed": failed, "left_for_space": skipped_full, "pending_items": len(pending), "active_jobs": active,
           "pinned": pins, "files": removed}
    if a.delete and removed:
        atomic_write(ws / "outputs/host_cleanup" / f"deleted_auto_{stamp}.json", json.dumps(rec, indent=1))
    elif not a.delete:
        atomic_write(ws / "outputs/host_cleanup/dry_run_auto_latest.json", json.dumps(rec, indent=1))
    cat = []
    for d, r in sorted(rows.items(), key=lambda kv: -kv[1]["bytes"]):
        cat.append({"folder": d, "files": r["files"], "gb": round(r["bytes"] / GB, 3),
                    **{k: round(r[k] / GB, 3) for k in ("hub", "regen", "smoke", "unarchived", "keep", "cold",
                                                        "referenced", "removable")},
                    "why": why_for(d)})
    tot = {k: round(sum(r[k] for r in rows.values()) / GB, 2) for k in
           ("bytes", "hub", "regen", "smoke", "unarchived", "keep", "cold", "referenced", "removable")}
    doc = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ws": str(ws), "drive_free_gb": rec["free_gb_after"],
           "tier": tier, "age_limit_days": limit_d, "total_gb": tot, "pending_items": len(pending),
           "active_jobs": active, "pinned": pins, "folders": cat}
    atomic_write(ws / "outputs/host_ops/host_catalogue.json", json.dumps(doc, indent=1))
    md = [f"# mpr host catalogue ({doc['utc']})", "",
          f"Drive free {rec['free_gb_before']} GB before this cycle, {rec['free_gb_after']} GB after "
          f"({'removed' if a.delete else 'dry run: would remove'} {rec['removed_gb']} GB in {rec['removed']} files); "
          f"tier {tier} (age limit {limit_d} d). Workspace {tot['bytes']} GB: on HF byte for byte {tot['hub']}, "
          f"rebuildable {tot['regen']}, smoke {tot['smoke']}, not yet archived {tot['unarchived']}, kept text/small "
          f"{tot['keep']}; cold {tot['cold']}, named by or pinned for queued or running jobs {tot['referenced']}.", "",
          f"Pinned while their job families are queued or running: {', '.join(pins) or 'none'}.", "",
          "| folder | GB | on HF | rebuildable | not archived | cold | in use | removable | why it is here |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for c in cat:
        if c["gb"] < 0.05:
            continue
        md.append(f"| {c['folder']} | {c['gb']:.2f} | {c['hub']:.2f} | {c['regen']:.2f} | {c['unarchived']:.2f} | "
                  f"{c['cold']:.2f} | {c['referenced']:.2f} | {c['removable']:.2f} | {c['why']} |")
    atomic_write(ws / "outputs/host_ops/host_catalogue.md", "\n".join(md) + "\n")
    line = (f"{rec['time']} {rec['mode']} tier {tier} free {rec['free_gb_before']} -> {rec['free_gb_after']} GB; "
            f"{'removed' if a.delete else 'would remove'} {rec['removed']} files {rec['removed_gb']} GB "
            f"{rec['by_class_gb']}; failed {len(failed)}; pending {len(pending)}, active {len(active)}; "
            f"{round(now_s() - t0, 1)} s")
    with open(ws / "outputs/host_ops/auto_cleanup.log", "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)
    return rec


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hub-index")
    h.add_argument("--archive", default="outputs/host_archive")
    h.add_argument("--out", default="outputs/host_ops/hub_index.json")
    r = sub.add_parser("run")
    r.add_argument("--delete", action="store_true", help="remove the eligible files (default: catalogue and dry run)")
    r.add_argument("--loop", action="store_true")
    r.add_argument("--every-h", type=float, default=3.0)
    r.add_argument("--ideal-gb", type=float, default=200.0)
    r.add_argument("--floor-gb", type=float, default=100.0)
    r.add_argument("--idle-days", type=float, default=14.0)
    r.add_argument("--cold-days", type=float, default=3.0)
    r.add_argument("--urgent-days", type=float, default=1.0)
    r.add_argument("--smoke-h", type=float, default=24.0)
    r.add_argument("--rx-root", default="C:/Users/Student2/rx")
    a = ap.parse_args(argv)
    if a.cmd == "hub-index":
        n = hub_index(Path(a.archive), Path(a.out))
        print(f"{n} archived files -> {a.out}")
        return 0
    ws = Path.cwd().resolve()
    if not (ws / "outputs").is_dir() or not str(ws).replace("\\", "/").lower().endswith("projects/mpr/ws"):
        raise SystemExit(f"{ws} is not mpr's rx workspace")
    while True:
        try:
            cycle(a, ws)
        except Exception as e:                       # a bad cycle must not end the loop
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} cycle failed: {type(e).__name__}: {e}", flush=True)
            if not a.loop:
                raise
        if not a.loop:
            return 0
        time.sleep(a.every_h * 3600)


if __name__ == "__main__":
    sys.exit(main())
