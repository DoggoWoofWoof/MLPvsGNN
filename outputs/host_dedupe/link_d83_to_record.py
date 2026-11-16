"""D83's byte-identical copies hard-linked to one file, the record's where there is one (16 Nov 23:5x; the user: "creating
multiple duplicates instead of managing things smartly"). Systems code; no science. Read-only unless --link.

    python outputs/host_dedupe/link_d83_to_record.py SURVEY.json            # dry run: what would be linked, by folder
    python outputs/host_dedupe/link_d83_to_record.py SURVEY.json --link
    python outputs/host_dedupe/link_d83_to_record.py --fresh --link          # survey the live arms and records first

Input: a dupe_survey.py JSON, workspace-relative or absolute paths (absolute ones under the workspace are made relative;
run from the workspace). For each group holding a file under a LIVE prefix (outputs/abl/D83F/, outputs/abl/D83E/,
outputs/b1v/: zd83b's B1 read carve, 17 Nov 00:3x):
  - dupe_survey lists one name per file (hard links count once), so every listed file with st_nlink > 1 has all
    its names read (fsutil hardlink list; their count must equal st_nlink);
  - the keeper is the file with a name under a RECORD prefix (outputs/pko/, PKO's record, first; then outputs/b1p/,
    B1P's b1w, which b1v reuses: the record's files stay the files they are, D83's page, section 4, "never touched"),
    else the oldest file (the first written; later copies take its name);
  - another file is relinked only if EVERY one of its names is under a LIVE prefix and not under an EXCLUDE prefix (of
    a carve whose record, <arm>/records/<ds>__<carve>.json, is not filed yet: the folders its chain still writes, look,
    links, typed, kt, lab, rmatch, zrc, its pools files) and is not in a cache part without record.json; it still has
    the survey's size and was not modified within --skip-recent-h (default 1). A file with a name anywhere else (the
    record, a closed stage) is left as it is: relinking only some of its names frees nothing. A chain's rmatch and
    zrc copies are often one file with two names; both are relinked, which frees it. Only a chain's own folders
    (CHAIN_DIRS: cache, chains, kt, pools, labels, links, typed, look; b1v's finished g1b and bench) are linked:
    cols, lists, lab, fits and reads belong to rows that may still write (or retry) in place;
  - keeper and file are hashed now and must both equal the survey's sha256; then, for each of the file's names,
    os.link(keeper, name.lnk.tmp), which must be the keeper's file, and os.replace(tmp, name), which must leave the
    name as the keeper's file. A name another process holds open is recorded busy and left as it is (the temporary
    name is removed); its file is then not freed ("partial").
Every path keeps its bytes; a file is freed once all its names are relinked. The keeper's mtime becomes each relinked
name's. Uses dedupe_links.py's sha256/same_file (imported, not edited).

Hazard (dedupe_links.py's): a linked file is ONE file. A writer that rewrites a linked path in place changes every name,
the record's included. The filed carves' cache, chains, kt, pools and label files have no writer left (the fits, LTM,
LAB and reads write their own folders); `dedupe_links.py verify --record R --hash` re-checks.

Record: outputs/host_dedupe/link_d83_<dry|link>_<stamp>.json (every member: action, bytes, keeper; totals; free space).
"""
import json
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "outputs" / "host_ops"))
import dedupe_links as DL  # noqa: E402

CARVES = {"outputs/abl/D83F/": {"squad": "fit", "hotpotqa": "fit", "2wiki": "fit", "musique": "s1fit", "webqsp": "fit"},
          "outputs/abl/D83E/": {d: "s1eval" for d in ("squad", "hotpotqa", "2wiki", "musique", "webqsp", "metaqa")},
          "outputs/b1v/": {d: "b1v" for d in ("2wiki", "hotpotqa", "musique")}}
LIVE = tuple(CARVES)
CHAIN_DIRS = ("cache", "chains", "kt", "pools", "labels", "links", "typed", "look",   # the chain's own folders
              "g1b", "bench")                                     # b1v: G1b's check and the bench, both done
RECORD = ("outputs/pko/", "outputs/b1p/")
CWD = Path.cwd().resolve().as_posix().rstrip("/") + "/"


def rel(p):
    p = p.replace("\\", "/")
    return p[len(CWD):] if p.lower().startswith(CWD.lower()) else p


def excludes():
    """Of a carve whose chain has not filed its record, every folder its chain still writes: look, links, typed, kt, lab,
    rmatch, zrc and its pools files. Its filed cache parts are linked (the chain builds only the parts without a
    record.json, zpko.cache_part; nothing else writes under a part), and so are label sets (zd83.write_label_sets
    compares a set already on disk and never rewrites it; a new one goes in through a temporary name and os.replace).
    A cache part without its record.json is skipped in the loop (unfiled_part)."""
    out = []
    for a, cs in CARVES.items():
        for ds, carve in cs.items():
            if Path(f"{a}records/{ds}__{carve}.json").is_file():
                continue
            out += [f"{a}{k}/{ds}/" for k in ("look", "links", "typed", "kt", "lab")]
            out += [f"{a}chains/{c}/{ds}/" for c in ("rmatch", "zrc")] + [f"{a}pools/{ds}__"]
    return tuple(dict.fromkeys(out))


def unfiled_part(p):
    """A cache-part file whose part has no record.json yet (a cache step may still be writing it)."""
    for a in CARVES:
        if p.startswith(a):
            r = p[len(a):].split("/")
            if len(r) > 4 and r[0] == "cache" and r[3].startswith("part_"):
                return not Path(a + "/".join(r[:4]) + "/record.json").is_file()
    return False


def names_of(p, s):
    """Every name of p's file, workspace-relative where under the workspace; [p] when it has one."""
    if s.st_nlink == 1:
        return [p]
    out = subprocess.run(["fsutil", "hardlink", "list", p], capture_output=True, text=True, timeout=120)
    drive = os.path.splitdrive(os.path.abspath(p))[0]
    got = sorted(rel(drive + ln.strip()) for ln in out.stdout.splitlines() if ln.strip())
    if len(got) != s.st_nlink:
        raise OSError(f"fsutil listed {len(got)} names, st_nlink {s.st_nlink}")
    return got


def why_not(n):
    if not n.startswith(LIVE):
        return "has a name outside the live arms"
    if n[len(next(a for a in LIVE if n.startswith(a))):].split("/")[0] not in CHAIN_DIRS:
        return "not a chain folder (a row may still write it)"
    if n.startswith(EXCLUDE):
        return "excluded (a chain still writes beside it)"
    if unfiled_part(n):
        return "cache part not filed"
    return None


EXCLUDE = excludes()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
recent_h = float(sys.argv[sys.argv.index("--skip-recent-h") + 1]) if "--skip-recent-h" in sys.argv else 1.0
if "--skip-recent-h" in sys.argv:
    args.remove(sys.argv[sys.argv.index("--skip-recent-h") + 1])
do_link = "--link" in sys.argv
if "--fresh" in sys.argv:   # survey the live arms and the records now (dupe_survey.py, imported), then plan or link
    import dupe_survey as DS
    args = [f"outputs/host_dedupe/survey_d83_{time.strftime('%Y%m%d-%H%M%S')}.json"]
    DS.main(["--min-mb", "1", "--skip-recent-min", "10", "--out", args[0], "--root", "D83F=outputs/abl/D83F",
             "--root", "D83E=outputs/abl/D83E", "--root", "pko=outputs/pko", "--root", "b1v=outputs/b1v",
             "--root", "b1p=outputs/b1p"])
print(f"{len(EXCLUDE)} excluded prefixes, from the carves without a filed record: "
      f"{[a + ds for a, cs in CARVES.items() for ds, c in cs.items() if not Path(f'{a}records/{ds}__{c}.json').is_file()]}")
survey = json.loads(Path(args[0]).read_text(encoding="utf-8"))
now = time.time()
free0 = shutil.disk_usage(".").free
rows, tot, by_folder = [], defaultdict(int), defaultdict(int)
for g in survey["groups"]:
    B = g["bytes"]
    files = {}                                   # file id -> (stat, every name)
    for p in (rel(f["path"]) for f in g["files"]):
        try:
            st = os.stat(p)
            files.setdefault((st.st_dev, st.st_ino), (st, names_of(p, st)))
        except (OSError, subprocess.SubprocessError) as e:
            rows.append({"path": p, "action": "skip", "why": f"stat or names: {e!r}"[:160], "bytes": B})
    if not any(n.startswith(LIVE) for _, ns in files.values() for n in ns):
        continue

    def rank(k):
        ns = files[k][1]
        for i, r in enumerate(RECORD):
            if any(n.startswith(r) for n in ns):
                return i, 0.0, ns[0]
        return len(RECORD), files[k][0].st_mtime, ns[0]

    kk = min(files, key=rank, default=None)
    if kk is None or files[kk][0].st_size != B:
        rows.append({"paths": [n for _, ns in files.values() for n in ns], "action": "skip",
                     "why": "no keeper at the survey's size"})
        continue
    kst, kns = files[kk]
    keeper = next((n for r in RECORD for n in kns if n.startswith(r)), kns[0])
    ksha = None
    for k, (st, ns) in files.items():
        if k == kk:
            continue
        why = next((w for w in map(why_not, ns) if w), None)
        if why is None and st.st_size != B:
            why = "size changed"
        if why is None and now - st.st_mtime < recent_h * 3600:
            why = f"modified within {recent_h:g} h"
        if why:
            rows.append({"paths": ns, "keeper": keeper, "action": "skip", "why": why, "bytes": B})
            tot["skip:" + why] += B
            continue
        if not do_link:
            rows.append({"paths": ns, "keeper": keeper, "action": "would_link", "bytes": B})
            tot["would_link"] += B
            by_folder["/".join(ns[0].split("/")[:6])] += B
            continue
        try:
            if ksha is None:
                ksha = DL.sha256(keeper)
            msha = DL.sha256(ns[0])
        except OSError as e:
            rows.append({"paths": ns, "keeper": keeper, "action": "skip", "why": f"hash: {e!r}"[:160], "bytes": B})
            continue
        if ksha != g["sha256"] or msha != g["sha256"]:
            rows.append({"paths": ns, "keeper": keeper, "action": "skip", "why": "sha256 differs from the survey",
                         "bytes": B})
            tot["skip:sha"] += B
            continue
        done = []
        for n in ns:
            tmp = n + ".lnk.tmp"
            try:
                os.link(keeper, tmp)
                assert DL.same_file(os.stat(tmp), os.stat(keeper)), "temporary link is not the keeper"
                os.replace(tmp, n)
                assert DL.same_file(os.stat(n), os.stat(keeper)), "name is not the keeper after replace"
                done.append(n)
            except (OSError, AssertionError) as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                rows.append({"path": n, "keeper": keeper, "action": "busy", "why": repr(e)[:160], "bytes": B})
        act = "linked" if len(done) == len(ns) else "partial"
        rows.append({"paths": done, "keeper": keeper, "action": act, "bytes": B, "sha256": g["sha256"]})
        tot[act] += B
        if act == "linked":
            by_folder["/".join(ns[0].split("/")[:6])] += B
free1 = shutil.disk_usage(".").free
stamp = time.strftime("%Y%m%d-%H%M%S")
out = Path("outputs/host_dedupe") / f"link_d83_{'link' if do_link else 'dry'}_{stamp}.json"
out.write_text(json.dumps({"survey": args[0], "skip_recent_h": recent_h, "exclude": EXCLUDE, "totals": dict(tot),
                           "free_before": free0, "free_after": free1, "rows": rows}, indent=1), encoding="utf-8")
for k, v in sorted(by_folder.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {v / 1e9:7.2f} GB  {k}")
for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
    print(f"{k}: {v / 1e9:.2f} GB")
print(f"free {free0 / 1e9:.2f} -> {free1 / 1e9:.2f} GB; record {out.as_posix()}")
