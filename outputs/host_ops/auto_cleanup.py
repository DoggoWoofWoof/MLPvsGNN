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
  regen       a build or cache that committed code rebuilds from the mirror (host_sync.py's REGEN pattern), and since
              17 October a stage's builds (STAGE_BUILD: its cache, look, chains, links, cols, units, space, res, kt or
              csr folder, and U1's graphs); a stage's fits, pools, screens and records are never builds
  smoke       a smoke test's output (a folder whose name starts with 'smoke')
  unarchived  anything else (results and models the archive does not hold yet): never removed
and as referenced when a queued or running mpr job names it or a folder above it (the feeder's unsent and running items
in hostfeed_items.txt / hostfeed_state.json, and the specs of mpr's active jobs in rx's active/ folder), or when it sits
under a folder that a queued or running job's script family reads without naming it (PINS: the look caches, anchors,
carves and six-base for the qd/lean families, the S6 builds and caches for chainscore), or under a folder written in
the source of a script such a job runs or imports, its imports followed (source_pins, 17 October), and as warm when it was read or
written within the age limit (NTFS last access, which this drive keeps, and mtime) or cold.

A release (outputs/host_ops/release/<name>.json, written by `release`; 19 October) gives up the family and source pins
on the files of its folders that existed when it was written (created and modified before it), optionally only on some
classes ('hub': only what HF holds byte for byte). A queued or running job that names a released file, or a folder above
it, in its command or inputs still protects it. Below --floor-gb a released file goes whatever its age. A stage's last
item releases the builds that only it read (P1g-c's rebuilt looks, once their link, chain and residual builds ended),
so a redo that needs more than the drive's free space fits.

A file is removed only if it is hub, regen or smoke, unreferenced, and cold for the limit that the drive's free space
sets: free >= --ideal-gb (200): smoke older than --smoke-h and other files untouched for --idle-days (14); free below
--ideal-gb: untouched for --cold-days (3); free below --floor-gb (100): untouched for --urgent-days (1); free below
--emergency-gb (50): any age. The loop's next cycle comes after --every-h (3) at the ideal, 30 min below it and 10 min
below the floor (9 Oct: the drive went from 71 GB free to 0 within one 3 h cycle). Within a cycle
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
GRACE_H = 48.0     # 19 Oct: a failed item and the items dropped behind it keep their folders this long (feeder_refs)
RELEASE_DIR = "outputs/host_ops/release"   # 19 Oct: the release records (releases, release)
FOLDER = re.compile(r"(?:outputs|logs)/(?:[A-Za-z0-9_.@=+-]+/)+")
# host_sync.py's REGEN: builds and caches that committed code rebuilds from the mirror
REGEN = re.compile(r"^outputs/mp_unified/(lean/cs\d+e?-[a-z0-9]+(-[a-z0-9]+)*\.npz|lean/cs\d+[a-z]*tmp_[^/]+/|"
                   r"look/[^/]+/[^/]+/chunks/|cache/)|/__pycache__/|"
                   # 9 Oct: step 4e's look chunks (look_step4e.py rebuilds them bit for bit; its caches are read instead)
                   r"^outputs/step4e/look/[^/]+/[^/]+/chunks/")
# 17 Oct (the user: "setup auto delete script for host items, dont wait for me"): a stage's builds, which its
# committed code makes again from the mirror and the stage's own inputs: caches, looks, chains, links, columns, units,
# spaces, residual arrays, KT columns, CSR graphs and U1's graphs. Fits, pools, screens and records are never builds.
STAGE_BUILD = re.compile(r"^outputs/[a-z0-9_]+/(?:cache|look|chains|links|space|res|cols|units|scores|kt|csr)/|"
                         r"^outputs/u1[a-z]/(?:webqsp|metaqa|musique|hotpotqa|2wiki|squad)/")
SMOKE = re.compile(r"(^|/)smoke[^/]*/")
PATHTOK = re.compile(r"(?:outputs|logs)/[A-Za-z0-9_./@=+-]+")
# A folder named in the source of a script that a queued or running job runs or imports (followed through its
# imports) is pinned, whether or not the command names it: "outputs/x/y", "outputs" / "x" / "y" and
# os.path.join("outputs", "x", "y") each pin outputs/x/y/ (outputs/x/ when only one part is written).
SCRIPT = re.compile(r"[A-Za-z0-9_./\\-]+\.py\b")
LIT_PATH = re.compile(r"[\"'](?:\./)?((?:outputs|logs)/[A-Za-z0-9_.@=+-]+(?:/[A-Za-z0-9_.@=+-]+)?)")
LIT_PARTS = re.compile(r"[\"'](outputs|logs)[\"']\s*[/,]\s*[\"']([A-Za-z0-9_.@=+-]+)[\"']"
                       r"(?:\s*[/,]\s*[\"']([A-Za-z0-9_.@=+-]+)[\"'])?")
IMPORT = re.compile(r"^\s*(?:from\s+([A-Za-z_][\w.]*)\s+import|import\s+([A-Za-z_][\w.]*(?:\s*,\s*[A-Za-z_][\w.]*)*))",
                    re.M)
MOD_DIRS = ("outputs/mp_unified", "scripts", "outputs/host_ops", "tools")
# Folders a script family reads without naming them on its command line: pinned while such a job is queued or running.
_ANCHORS = ("outputs/mp_approx_hotpot_anchor/host/", "outputs/mp_approx_kb_anchor/host/",
            "outputs/mp_approx_2wiki_anchor/host/", "outputs/mp_approx_musique_anchor/host/")
_BASES = ("outputs/mp_approx_l12/", "outputs/mp_approx_six_base/", "outputs/m3b/csr/", "outputs/mp_approx_l16_design/look/")
PINS = (
    (re.compile(r"qd_gnn\d*\.py|qd_six\.py|qd_tdiag\.py"), ("outputs/mp_unified/look/", "outputs/mp_unified/qd/") + _BASES),
    (re.compile(r"lean_(?:host|mlp|read|fast|time)\d*\.py|pair9\.py|qfam9\.py|colshift10\.py|floor10\.py"),
     ("outputs/mp_unified/look/", "outputs/mp_unified/lean/") + _ANCHORS + _BASES),
    # 9 Oct: look_b1u.py (a wrapper over look_b1 -> look_x_six) matched nothing here and the cleaner took
    # universal_v2/six/fits from under its queued looks; every look_*.py wrapper now pins, and items' inputs are scanned
    (re.compile(r"look_\w+\.py|mp_approx_\w+\.py|universal_v2_\w+\.py"),
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


def ended_badly(t) -> bool:
    """A terminal record ([status, rc, ...]) of a job that did not end done with rc 0."""
    return bool(t) and not (t[0] == "done" and (len(t) < 2 or t[1] in (0, None)))


def feeder_refs(ws: Path, active: set[str] | None = None,
                grace_h: float = GRACE_H) -> tuple[set[str], list[str], list[str]]:
    """Path tokens, names and commands of the feeder's unsent and running items. A sent item counts while rx holds its
    job as active (active: the ids of mpr's active jobs), or for 6 h after its launch; 17 Oct: 739 of 753 'pending'
    items were jobs that had ended without an end record in the feeder's state, and they pinned every folder.
    19 Oct: a failed item, and every item the feeder dropped behind it, count for grace_h after the failed job's
    launch. P1g-c's 14 link and rmatch builds failed at once (18 Oct 15:36), the feeder dropped the ~300 items behind
    them, nothing left named outputs/p1gb/, and the emergency tier took P1g-b's rebuilt looks and KT columns at 22:28;
    the grace keeps a failed stage's inputs until it is fixed and sent again."""
    items = ws / "outputs/host_ops/hostfeed_items.txt"
    state = ws / "outputs/host_ops/hostfeed_state.json"
    if not items.is_file():
        return set(), ["no hostfeed_items.txt"], []
    st = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
    sent, term, dropped = st.get("sent", {}), st.get("terminal", {}), st.get("dropped", {})
    lines = []
    deps_of = {}
    for line in items.read_text(encoding="utf-8").split("\n"):
        if not line.startswith("line|"):
            continue
        f = line.split("|")
        if len(f) < 8:
            continue
        lines.append(f)
        deps_of[f[3]] = [d for d in f[2].split(",") if d]
    memo: dict[str, bool] = {}

    def recent_failure(name: str, depth: int = 0) -> bool:
        """A failed job launched within grace_h is this dropped item's dep, or a dep of a dropped dep, ..."""
        if name in memo:
            return memo[name]
        memo[name] = False                           # a cycle in the deps finds nothing
        hit = False
        for d in deps_of.get(name, []):
            if d in sent and ended_badly(term.get(sent[d])) and launched_within(sent[d], grace_h):
                hit = True
            elif d in dropped and d not in sent and depth < 64 and recent_failure(d, depth + 1):
                hit = True
            if hit:
                break
        memo[name] = hit
        return hit

    refs, names, cmds = set(), [], []
    for f in lines:
        name = f[3]
        if name in dropped and name not in sent:
            if not recent_failure(name):
                continue
        elif name in sent and sent[name] in term:
            if not (ended_badly(term[sent[name]]) and launched_within(sent[name], grace_h)):
                continue
        if name in sent and active is not None and sent[name] not in active and not launched_within(sent[name], 6.0):
            continue
        names.append(name)
        refs.update(PATHTOK.findall(f[6].replace(";", " ") + " " + f[7]))
        cmds.append(f[6].replace(";", " ") + " " + f[7])   # inputs too: wrappers list the scripts they import there
    return refs, names, cmds


def launched_within(jid: str, hours: float) -> bool:
    """rx's job ids start with the launch's local time, YYMMDD-HHMMSS."""
    try:
        t0 = time.mktime(time.strptime(jid[:13], "%y%m%d-%H%M%S"))
    except (ValueError, OverflowError):
        return True
    return now_s() - t0 < hours * 3600


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


def source_pins(ws: Path, cmds: list[str]) -> tuple[list[str], int]:
    """Folders written in the source of every script the commands run or list as inputs, and of the modules those
    import (found beside the script or in MOD_DIRS), transitively. Returns (folder prefixes, scripts read)."""
    todo = {s.replace("\\", "/") for c in cmds for s in SCRIPT.findall(c)}
    seen, pins = set(), set()
    while todo:
        rel = todo.pop()
        if rel in seen:
            continue
        seen.add(rel)
        p = ws / rel
        try:
            src = p.read_text(encoding="utf-8", errors="replace")
            here = p.resolve().parent.relative_to(ws).as_posix()
        except (OSError, ValueError):
            continue
        for m in LIT_PATH.findall(src):
            pins.add(m.rstrip("/.") + "/")
        for top, a, b in LIT_PARTS.findall(src):
            pins.add(f"{top}/{a}/{b}/" if b else f"{top}/{a}/")
        for m1, m2 in IMPORT.findall(src):
            for name in (m1 or m2).split(","):
                parts = name.strip().split(".")
                for d in (here,) + MOD_DIRS:
                    for cand in (f"{d}/{'/'.join(parts)}.py", f"{d}/{parts[-1]}.py"):
                        if cand not in seen and (ws / cand).is_file():
                            todo.add(cand)
    return sorted(pins), len(seen)


def referenced(rel: str, refs: set[str], prefixes: list[str]) -> bool:
    if rel in refs:
        return True
    return any(rel.startswith(p) for p in prefixes)


# ── releases: folders whose family and source pins are given up ─────────────


def valid_folder(f) -> bool:
    """A folder under outputs/ or logs/ ('/' at the end), with no '.' or '..' part, not a kept folder nor above one."""
    return (isinstance(f, str) and bool(FOLDER.fullmatch(f)) and not {".", ".."} & set(f.split("/"))
            and not f.startswith(KEEP_DIRS) and not any(k.startswith(f) for k in KEEP_DIRS))


def releases(ws: Path) -> list[tuple[str, tuple[str, ...], frozenset | None, int]]:
    """The release records as (name, folders, classes or None for every class, before_ns). A malformed record, or a
    folder in one that `release` would refuse, releases nothing."""
    out = []
    d = ws / RELEASE_DIR
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            r = json.loads(p.read_text(encoding="utf-8"))
            fs = tuple(f for f in r["folders"] if valid_folder(f))
            cl = r.get("classes")
            out.append((p.stem, fs, frozenset(cl) if cl else None, int(r["before_ns"])))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return out


def born_ns(st) -> int:
    """The later of a file's creation (st_birthtime on Windows; the inode change elsewhere) and its modification: a file
    restored with its old mtime, or rebuilt, after a release is not released."""
    b = getattr(st, "st_birthtime_ns", None)
    return max(st.st_mtime_ns, b if b is not None else st.st_ctime_ns)


def released(rel: str, born: int, cls: str, rules) -> bool:
    return any(fs and born <= b and (c is None or cls in c) and rel.startswith(fs) for _, fs, c, b in rules)


def release(ws: Path, name: str, folders: list[str], classes: list[str] | None, why: str) -> Path:
    """Write outputs/host_ops/release/<name>.json (the module's doc says what a release does)."""
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", name or ""):
        raise SystemExit(f"--name {name!r}: lower-case letters, digits, '_', '.' and '-'")
    fs = [f.replace("\\", "/").strip().rstrip("/") + "/" for f in folders]
    bad = [f for f in fs if not valid_folder(f)]
    if not fs or bad:
        raise SystemExit(f"--folder {bad or folders}: folders under outputs/ or logs/, none of them kept or above a kept "
                         f"folder")
    cl = sorted(set(classes or ()))
    if set(cl) - {"hub", "regen", "smoke"}:
        raise SystemExit(f"--classes {cl}: hub, regen or smoke (keep and unarchived files are never removed)")
    if not (why or "").strip():
        raise SystemExit("--why: say why nothing queued reads these folders")
    p = ws / RELEASE_DIR / f"{name}.json"
    rec = {"name": name, "folders": fs, "classes": cl or None, "why": why.strip(),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "before_ns": time.time_ns()}
    atomic_write(p, json.dumps(rec, indent=1) + "\n")
    print(f"released {', '.join(fs)}{' (' + ', '.join(cl) + ' only)' if cl else ''} -> {p.relative_to(ws).as_posix()}",
          flush=True)
    return p


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
    if REGEN.search(rel) or STAGE_BUILD.search(rel):
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
    r2, active, c2 = active_refs(Path(a.rx_root))
    r1, pending, c1 = feeder_refs(ws, set(active), getattr(a, "grace_h", GRACE_H))
    refs = {x.rstrip("/.") for x in r1 | r2}
    pins = sorted({d for rx_, ds in PINS if any(rx_.search(c) for c in c1 + c2) for d in ds})
    src_pins, n_src = source_pins(ws, c1 + c2)
    pins = sorted(set(pins) | set(src_pins))
    named = [x + "/" for x in refs]
    rules = releases(ws)
    if free_gb < a.emergency_gb:   # 9 Oct: the drive filled within one 3 h cycle; below this, no age limit
        tier, limit_d = "emergency", 0.0
    elif free_gb < a.floor_gb:
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
        rls = bool(rules) and released(rel, born_ns(st), cls, rules)
        ref = referenced(rel, refs, named) or (not rls and any(rel.startswith(p) for p in pins))
        d = folder(rel)
        r = rows[d]
        r["files"] += 1
        r["bytes"] += st.st_size
        r[cls] += st.st_size
        if ref:
            r["referenced"] += st.st_size
        if rls:
            r["released"] += st.st_size
        if idle_d >= a.cold_days:
            r["cold"] += st.st_size
        if cls == "keep" or cls == "unarchived" or ref:
            continue
        lim = 0.0 if rls and tier in ("urgent", "emergency") else limit_d
        ok = (idle_d * 24 >= a.smoke_h) if cls == "smoke" else (idle_d >= lim)
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
           "pinned": pins, "scripts_read_for_pins": n_src,
           "released": {n: {"folders": list(fs), "classes": sorted(c) if c else None} for n, fs, c, _ in rules},
           "files": removed}
    if a.delete and removed:
        atomic_write(ws / "outputs/host_cleanup" / f"deleted_auto_{stamp}.json", json.dumps(rec, indent=1))
    elif not a.delete:
        atomic_write(ws / "outputs/host_cleanup/dry_run_auto_latest.json", json.dumps(rec, indent=1))
    cat = []
    for d, r in sorted(rows.items(), key=lambda kv: -kv[1]["bytes"]):
        cat.append({"folder": d, "files": r["files"], "gb": round(r["bytes"] / GB, 3),
                    **{k: round(r[k] / GB, 3) for k in ("hub", "regen", "smoke", "unarchived", "keep", "cold",
                                                        "referenced", "released", "removable")},
                    "why": why_for(d)})
    tot = {k: round(sum(r[k] for r in rows.values()) / GB, 2) for k in
           ("bytes", "hub", "regen", "smoke", "unarchived", "keep", "cold", "referenced", "removable")}
    doc = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ws": str(ws), "drive_free_gb": rec["free_gb_after"],
           "tier": tier, "age_limit_days": limit_d, "total_gb": tot, "pending_items": len(pending),
           "active_jobs": active, "pinned": pins, "released": rec["released"], "folders": cat}
    atomic_write(ws / "outputs/host_ops/host_catalogue.json", json.dumps(doc, indent=1))
    md = [f"# mpr host catalogue ({doc['utc']})", "",
          f"Drive free {rec['free_gb_before']} GB before this cycle, {rec['free_gb_after']} GB after "
          f"({'removed' if a.delete else 'dry run: would remove'} {rec['removed_gb']} GB in {rec['removed']} files); "
          f"tier {tier} (age limit {limit_d} d). Workspace {tot['bytes']} GB: on HF byte for byte {tot['hub']}, "
          f"rebuildable {tot['regen']}, smoke {tot['smoke']}, not yet archived {tot['unarchived']}, kept text/small "
          f"{tot['keep']}; cold {tot['cold']}, named by or pinned for queued or running jobs {tot['referenced']}.", "",
          f"Pinned while their job families are queued or running: {', '.join(pins) or 'none'}.", "",
          f"Released (pins given up; below the floor they go at any age): "
          f"{'; '.join(n + ': ' + ', '.join(fs) + (' (' + ', '.join(sorted(c)) + ')' if c else '') for n, fs, c, _ in rules) or 'none'}.",
          "",
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


def selftest() -> int:
    """feeder_refs on a toy feeder: which items keep their folders (19 Oct's grace for failed and dropped items)."""
    import tempfile
    t0 = now_s()
    ws = Path(tempfile.mkdtemp(prefix="auto_cleanup_"))
    try:
        (ws / "outputs/host_ops").mkdir(parents=True)
        new = time.strftime("%y%m%d-%H%M%S", time.localtime(t0 - 3600))
        old = time.strftime("%y%m%d-%H%M%S", time.localtime(t0 - 72 * 3600))
        items = {   # name: (deps, sent job id or None, terminal record or None)
            "a-pend": ("", None, None),
            "a-done": ("", f"{new}-a-done-0001", ["done", 0, 0.1, 5.0]),
            "a-fail-new": ("", f"{new}-a-fail-new-0002", ["failed", 1, 0.0, 3.0]),
            "a-fail-old": ("", f"{old}-a-fail-old-0003", ["failed", 1, 0.0, 3.0]),
            "b-drop1": ("a-fail-new", None, None),
            "b-drop2": ("b-drop1,a-done", None, None),
            "b-drop-old": ("a-fail-old", None, None),
            "b-drop-old2": ("b-drop-old", None, None),
            "c-run-old": ("", f"{old}-c-run-old-0004", None),
            "c-run-new": ("", f"{new}-c-run-new-0005", None),
            "c-run-active": ("", f"{old}-c-run-active-0006", None),
        }
        dropped = ("b-drop1", "b-drop2", "b-drop-old", "b-drop-old2")
        lines = [f"line|x|{deps}|{n}|1|1|scripts/{n}.py|python scripts/{n}.py --out outputs/{n}/look/x"
                 for n, (deps, _, _) in items.items()]
        (ws / "outputs/host_ops/hostfeed_items.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        st = {"sent": {n: j for n, (_, j, _) in items.items() if j},
              "terminal": {j: t for _, (_, j, t) in items.items() if j and t},
              "dropped": {n: "dep ended badly" for n in dropped}}
        (ws / "outputs/host_ops/hostfeed_state.json").write_text(json.dumps(st), encoding="utf-8")
        active = {items["c-run-active"][1]}
        refs, names, cmds = feeder_refs(ws, active)
        want = ["a-pend", "a-fail-new", "b-drop1", "b-drop2", "c-run-new", "c-run-active"]
        assert sorted(names) == sorted(want), sorted(names)
        assert "outputs/b-drop2/look/x" in refs and "outputs/b-drop-old/look/x" not in refs, sorted(refs)
        assert len(cmds) == len(want)
        # without the grace, as before 19 Oct: failed and dropped items keep nothing
        _, names0, _ = feeder_refs(ws, active, grace_h=0.0)
        assert sorted(names0) == sorted(["a-pend", "c-run-new", "c-run-active"]), sorted(names0)
        # a cycle in the deps of dropped items ends
        lines.append("line|x|b-loop2|b-loop1|1|1|scripts/l.py|python scripts/l.py")
        lines.append("line|x|b-loop1|b-loop2|1|1|scripts/l.py|python scripts/l.py")
        (ws / "outputs/host_ops/hostfeed_items.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        st["dropped"].update({"b-loop1": "x", "b-loop2": "x"})
        (ws / "outputs/host_ops/hostfeed_state.json").write_text(json.dumps(st), encoding="utf-8")
        _, names1, _ = feeder_refs(ws, active)
        assert sorted(names1) == sorted(want), sorted(names1)
        assert ended_badly(["failed", 1]) and ended_badly(["done", 2]) and ended_badly(["cancelled", 1])
        assert not ended_badly(["done", 0]) and not ended_badly(None) and not ended_badly(["done"])
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    # releases (19 Oct), through cycle's dry run on a toy workspace; the tiers are forced by the thresholds
    ws = Path(tempfile.mkdtemp(prefix="auto_cleanup_rel_"))
    rx = Path(tempfile.mkdtemp(prefix="auto_cleanup_rx_"))
    try:
        (ws / "outputs/mp_unified").mkdir(parents=True)
        (ws / "outputs/mp_unified/look_toy.py").write_text('LOOK = "outputs/p1gb/look"\n', encoding="utf-8")
        items = ws / "outputs/host_ops/hostfeed_items.txt"
        lines = ["line|x||p-look|1|1|outputs/mp_unified/look_toy.py|python outputs/mp_unified/look_toy.py --x",
                 "line|x||p-named|1|1|scripts/n.py|python scripts/n.py --in outputs/p1gb/look/a/named.npz"]
        atomic_write(items, "\n".join(lines) + "\n")
        fresh, h_l12 = "outputs/p1gb/look/a/c0.npz", "outputs/mp_approx_l12/carves/h.npz"
        old_files = ("outputs/p1gb/look/a/named.npz", "outputs/p1gb/look/b/c0.npz", h_l12,
                     "outputs/mp_approx_l12/cache/r.npz")
        blob = b"\0" * (MIN_BYTES + 1)
        two_days = time.time_ns() - 2 * 86400 * 10**9
        for f in (fresh,) + old_files:
            (ws / f).parent.mkdir(parents=True, exist_ok=True)
            (ws / f).write_bytes(blob)
            if f != fresh:
                os.utime(ws / f, ns=(two_days, two_days))
        hst = (ws / h_l12).stat()
        atomic_write(ws / "outputs/host_ops/hub_index.json",
                     json.dumps({"files": [[h_l12, hst.st_size, hst.st_mtime_ns, "o-toy"]]}))
        base = dict(delete=False, emergency_gb=0.0, floor_gb=1e12, ideal_gb=1e12, urgent_days=1.0, cold_days=3.0,
                    idle_days=14.0, smoke_h=24.0, rx_root=str(rx), grace_h=GRACE_H)

        def gone(**kw):
            rec = cycle(argparse.Namespace(**{**base, **kw}), ws)
            return sorted(x[0] for x in rec["files"])
        assert gone() == [], gone()                  # urgent: each file is pinned (family or source) or named
        release(ws, "toy-look-a", ["outputs/p1gb/look/a"], None, "toy: the look's readers ended")
        release(ws, "toy-l12", ["outputs/mp_approx_l12/"], ["hub"], "toy: a closed track's archived copy")
        time.sleep(0.1)                              # past the coarse clock that stamps a file's creation
        late = ws / "outputs/p1gb/look/a/late.npz"   # rebuilt, or restored with an old mtime, after the release
        late.write_bytes(blob)
        os.utime(late, ns=(10**9, 10**9))
        assert gone() == sorted([fresh, h_l12]), gone()   # a released file goes at any age below the floor
        assert gone(ideal_gb=0.0, floor_gb=0.0) == [], gone(ideal_gb=0.0, floor_gb=0.0)   # above it, the idle limit
        assert gone(floor_gb=1e12, emergency_gb=1e12) == sorted([fresh, h_l12])
        atomic_write(items, "")                      # nothing queued: the old and the released go, not the fresh
        assert gone() == sorted(set(old_files) | {fresh, late.relative_to(ws).as_posix()}), gone()
        cat = json.loads((ws / "outputs/host_ops/host_catalogue.json").read_text(encoding="utf-8"))
        assert set(cat["released"]) == {"toy-look-a", "toy-l12"} and cat["released"]["toy-l12"]["classes"] == ["hub"]
        for bad in (["outputs/"], ["outputs"], ["outputs/host_ops/x"], ["outputs/../x"], ["outputs/./x"], ["c:/x"],
                    ["x/outputs/y"], []):
            try:
                release(ws, "bad", bad, None, "x")
            except SystemExit:
                continue
            raise AssertionError(f"release took {bad}")
        for kw in ({"name": "Bad Name"}, {"classes": ["unarchived"]}, {"classes": ["keep"]}, {"why": " "}):
            try:
                release(ws, **{"name": "ok", "folders": ["outputs/x"], "classes": None, "why": "x", **kw})
            except SystemExit:
                continue
            raise AssertionError(f"release took {kw}")
        (ws / RELEASE_DIR / "broken.json").write_text("{", encoding="utf-8")
        (ws / RELEASE_DIR / "wide.json").write_text(json.dumps({"folders": ["outputs/"], "before_ns": time.time_ns()}),
                                                     encoding="utf-8")
        assert {n for n, fs, _, _ in releases(ws) if fs} == {"toy-look-a", "toy-l12"}, releases(ws)
    finally:
        shutil.rmtree(ws, ignore_errors=True)
        shutil.rmtree(rx, ignore_errors=True)
    print(f"auto_cleanup selftest: pending, running (active or launched within 6 h), failed within the grace and "
          f"dropped behind a failure within the grace (through a chain of drops) keep their folders; done items, old "
          f"failures and items dropped behind them do not; grace 0 is the rule before 19 Oct; a cycle of drops ends; "
          f"a release gives up the family and source pins on its folders' files from before it (its classes only), a "
          f"named file stays, below the floor a released file goes at any age and above it waits for the idle limit, a "
          f"file born after the release stays, and release refuses kept, wide or outside folders "
          f"({now_s() - t0:.2f}s)", flush=True)
    return 0


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
    r.add_argument("--emergency-gb", type=float, default=50.0,
                   help="below this free space, unreferenced hub/regen files go whatever their age")
    r.add_argument("--smoke-h", type=float, default=24.0)
    r.add_argument("--rx-root", default=str(Path.home() / "rx"))
    r.add_argument("--grace-h", type=float, default=GRACE_H,
                   help="a failed item and the items dropped behind it keep their folders this long after its launch")
    sub.add_parser("selftest")
    rl = sub.add_parser("release", help="give up the family and source pins on folders that nothing queued reads")
    rl.add_argument("--name", required=True, help="the record's name: outputs/host_ops/release/<name>.json")
    rl.add_argument("--folder", action="append", required=True, help="a folder under outputs/ or logs/ (repeat)")
    rl.add_argument("--classes", default="", help="comma list of hub, regen, smoke (default: every class)")
    rl.add_argument("--why", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "hub-index":
        n = hub_index(Path(a.archive), Path(a.out))
        print(f"{n} archived files -> {a.out}")
        return 0
    ws = Path.cwd().resolve()
    if not (ws / "outputs").is_dir() or not str(ws).replace("\\", "/").lower().endswith("projects/mpr/ws"):
        raise SystemExit(f"{ws} is not mpr's rx workspace")
    if a.cmd == "release":
        release(ws, a.name, a.folder, [c for c in a.classes.split(",") if c], a.why)
        return 0
    while True:
        try:
            cycle(a, ws)
        except Exception as e:                       # a bad cycle must not end the loop
            print(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} cycle failed: {type(e).__name__}: {e}", flush=True)
            if not a.loop:
                raise
        if not a.loop:
            return 0
        # the cadence follows the free space: every --every-h at the ideal, 30 min below it, 10 min below the floor
        free = shutil.disk_usage(str(ws)).free / GB
        time.sleep(3600 * (a.every_h if free >= a.ideal_gb else min(a.every_h, 0.5) if free >= a.floor_gb
                           else min(a.every_h, 1 / 6)))


if __name__ == "__main__":
    sys.exit(main())
