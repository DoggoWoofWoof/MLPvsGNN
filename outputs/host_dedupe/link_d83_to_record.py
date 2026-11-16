"""D83's byte-identical copies hard-linked to one file, the record's where there is one (16 Nov 23:5x; the user: "creating
multiple duplicates instead of managing things smartly"). Systems code; no science. Read-only unless --link.

    python outputs/host_dedupe/link_d83_to_record.py SURVEY.json            # dry run: what would be linked, by folder
    python outputs/host_dedupe/link_d83_to_record.py SURVEY.json --link
    python outputs/host_dedupe/link_d83_to_record.py --fresh --link          # survey D83F, D83E and pko first

Input: a dupe_survey.py JSON whose paths are workspace-relative (run from the workspace). For each group holding a
file under outputs/abl/D83F/ or outputs/abl/D83E/:
  - the keeper is the member under outputs/pko/ (the record's roots stay the files they are: D83's page, section 4,
    "never touched"), else the oldest member (the first written; later copies take its name);
  - a member is linked only if it is under outputs/abl/D83F/ or outputs/abl/D83E/, has one name (st_nlink 1), still
    has the survey's size, was not modified within --skip-recent-h (default 1), and is not under an EXCLUDE prefix:
    of a carve whose record (outputs/abl/<arm>/records/<ds>__<carve>.json) is not filed yet, the folders its chain
    still writes (look, links, typed, kt, lab, rmatch, zrc, its pools files) and any cache part without record.json;
  - keeper and member are hashed now and must both equal the survey's sha256; then os.link(keeper, member.lnk.tmp),
    which must be the keeper's file; then os.replace(tmp, member), which must leave member as the keeper's file. A
    member another process holds open is recorded busy and left as it is (the temporary name is removed).
Every path keeps its bytes; only the replaced member's own copy is freed. The keeper's mtime becomes each linked
member's. Uses dedupe_links.py's sha256/same_file (imported, not edited).

Hazard (dedupe_links.py's): a linked file is ONE file. A writer that rewrites a linked path in place changes every name,
the record's included. The filed carves' cache, chains, kt, pools and label files have no writer left (the fits, LTM,
LAB and reads write their own folders); `dedupe_links.py verify --record R --hash` re-checks.

Record: outputs/host_dedupe/link_d83_<dry|link>_<stamp>.json (every member: action, bytes, keeper; totals; free space).
"""
import json
import os
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "outputs" / "host_ops"))
import dedupe_links as DL  # noqa: E402

LIVE = ("outputs/abl/D83F/", "outputs/abl/D83E/")
RECORD = "outputs/pko/"
CARVES = {"D83F": {"squad": "fit", "hotpotqa": "fit", "2wiki": "fit", "musique": "s1fit", "webqsp": "fit"},
          "D83E": {d: "s1eval" for d in ("squad", "hotpotqa", "2wiki", "musique", "webqsp", "metaqa")}}


def excludes():
    """Of a carve whose chain has not filed its record, every folder its chain still writes: look, links, typed, kt, lab,
    rmatch, zrc and its pools files. Its filed cache parts are linked (the chain builds only the parts without a
    record.json, zpko.cache_part; nothing else writes under a part), and so are label sets (zd83.write_label_sets
    compares a set already on disk and never rewrites it; a new one goes in through a temporary name and os.replace).
    A cache part without its record.json is skipped in the loop (unfiled_part)."""
    out = []
    for arm, cs in CARVES.items():
        for ds, carve in cs.items():
            if (Path("outputs/abl") / arm / "records" / f"{ds}__{carve}.json").is_file():
                continue
            a = f"outputs/abl/{arm}/"
            out += [f"{a}{k}/{ds}/" for k in ("look", "links", "typed", "kt", "lab")]
            out += [f"{a}chains/{c}/{ds}/" for c in ("rmatch", "zrc")] + [f"{a}pools/{ds}__"]
    return tuple(dict.fromkeys(out))


def unfiled_part(p):
    """A cache-part file whose part has no record.json yet (a cache step may still be writing it)."""
    parts = p.split("/")
    if len(parts) > 6 and parts[3] == "cache" and parts[6].startswith("part_"):
        return not Path("/".join(parts[:7]) + "/record.json").is_file()
    return False


EXCLUDE = excludes()
args = [a for a in sys.argv[1:] if not a.startswith("--")]
recent_h = float(sys.argv[sys.argv.index("--skip-recent-h") + 1]) if "--skip-recent-h" in sys.argv else 1.0
if "--skip-recent-h" in sys.argv:
    args.remove(sys.argv[sys.argv.index("--skip-recent-h") + 1])
do_link = "--link" in sys.argv
if "--fresh" in sys.argv:   # survey D83F, D83E and the record now (dupe_survey.py, imported), then plan or link on it
    import dupe_survey as DS
    args = [f"outputs/host_dedupe/survey_d83_{time.strftime('%Y%m%d-%H%M%S')}.json"]
    DS.main(["--min-mb", "1", "--skip-recent-min", "10", "--out", args[0], "--root", "D83F=outputs/abl/D83F",
             "--root", "D83E=outputs/abl/D83E", "--root", "pko=outputs/pko"])
print(f"{len(EXCLUDE)} excluded prefixes (carves without a filed record): "
      f"{sorted({p.split('/')[2] + ':' + p.rstrip('/').split('/')[-1].split('__')[0] for p in EXCLUDE})}")
survey = json.loads(Path(args[0]).read_text(encoding="utf-8"))
now = time.time()
free0 = shutil.disk_usage(".").free
rows, tot, by_folder = [], defaultdict(int), defaultdict(int)
for g in survey["groups"]:
    files = [f["path"].replace("\\", "/") for f in g["files"]]
    if not any(p.startswith(LIVE) for p in files):
        continue
    st = {}
    for p in files:
        try:
            st[p] = os.stat(p)
        except OSError:
            pass
    rec = [p for p in files if p.startswith(RECORD) and p in st]
    keeper = rec[0] if rec else min((p for p in st), key=lambda p: (st[p].st_mtime, p), default=None)
    if keeper is None or st[keeper].st_size != g["bytes"]:
        for p in files:
            rows.append({"path": p, "action": "skip", "why": "no keeper at the survey's size"})
        continue
    ksha = None
    for p in files:
        if p == keeper:
            continue
        why = None
        if not p.startswith(LIVE):
            why = "not a D83 path"
        elif p.startswith(EXCLUDE):
            why = "excluded (a chain still writes beside it)"
        elif unfiled_part(p):
            why = "cache part not filed"
        elif p not in st:
            why = "gone"
        elif DL.same_file(st[p], st[keeper]):
            why = "already the keeper's file"
        elif st[p].st_nlink > 1:
            why = "has other names"
        elif st[p].st_size != g["bytes"]:
            why = "size changed"
        elif now - st[p].st_mtime < recent_h * 3600:
            why = f"modified within {recent_h:g} h"
        if why:
            rows.append({"path": p, "keeper": keeper, "action": "skip", "why": why, "bytes": g["bytes"]})
            tot["skip:" + why] += g["bytes"]
            continue
        if not do_link:
            rows.append({"path": p, "keeper": keeper, "action": "would_link", "bytes": g["bytes"]})
            tot["would_link"] += g["bytes"]
            by_folder["/".join(p.split("/")[:6])] += g["bytes"]
            continue
        try:
            if ksha is None:
                ksha = DL.sha256(keeper)
            msha = DL.sha256(p)
        except OSError as e:
            rows.append({"path": p, "keeper": keeper, "action": "skip", "why": f"hash: {e!r}"[:160], "bytes": g["bytes"]})
            continue
        if ksha != g["sha256"] or msha != g["sha256"]:
            rows.append({"path": p, "keeper": keeper, "action": "skip", "why": "sha256 differs from the survey",
                         "bytes": g["bytes"]})
            tot["skip:sha"] += g["bytes"]
            continue
        tmp = p + ".lnk.tmp"
        try:
            os.link(keeper, tmp)
            assert DL.same_file(os.stat(tmp), os.stat(keeper)), "temporary link is not the keeper"
            os.replace(tmp, p)
            assert DL.same_file(os.stat(p), os.stat(keeper)), "member is not the keeper after replace"
            rows.append({"path": p, "keeper": keeper, "action": "linked", "bytes": g["bytes"], "sha256": g["sha256"]})
            tot["linked"] += g["bytes"]
            by_folder["/".join(p.split("/")[:6])] += g["bytes"]
        except (OSError, AssertionError) as e:
            if os.path.exists(tmp):
                os.remove(tmp)
            rows.append({"path": p, "keeper": keeper, "action": "busy", "why": repr(e)[:160], "bytes": g["bytes"]})
            tot["busy"] += g["bytes"]
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
