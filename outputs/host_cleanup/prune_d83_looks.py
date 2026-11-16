"""D83's look chunks, pruned once their readers are filed (16 Nov 23:5x; the user: "i dont think you are utilizing the space
properly"; 17 Nov 00:1x: "remove whatever is not required"). The rule is zpko.prune_carve's (PKO section 6), applied to
D83's arms and to b1v (outputs/b1v, zd83b's B1 read carve on the same chain): a carve's look chunks are read only by
its cache parts, links, and (chained carves) rmatch, zrc and typed. The carve record (zpko.record_carve, written by the
chain's last step) names every one of those records and the look's own record_*/pools_* JSON, never the chunks, and the
gate re-hashes only what the record names. A re-run treats a look shard as filed by its record_/pools_ JSON
(zpko.look_outs), so a pruned carve is never rebuilt. The fits and reads read cache, links, kt and chains, not the look.

A carve is pruned only if: records/<ds>__<carve>.json exists with no problems; every input it names exists; every
cache part has a links part (and rmatch/zrc parts, typed record when chained); no pruned.json yet. Only
look/<ds>/<carve>/chunks/c*.npz go. pruned.json (zpko's fields, plus the removed files' bytes and links) is written
beside the shard records first and completed after.

    python outputs/host_cleanup/prune_d83_looks.py                 # dry run: every carve, pruned or refused and why
    python outputs/host_cleanup/prune_d83_looks.py --delete        # prune the carves that pass
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path.cwd().resolve()
ARMS = {"D83F": {"squad": "fit", "hotpotqa": "fit", "2wiki": "fit", "musique": "s1fit", "webqsp": "fit"},
        "D83E": {d: "s1eval" for d in ("squad", "hotpotqa", "2wiki", "musique", "webqsp", "metaqa")},
        "b1v": {d: "b1v" for d in ("2wiki", "hotpotqa", "musique")}}       # zd83b's carve: the same chain and record
BASE = {"D83F": ROOT / "outputs" / "abl" / "D83F", "D83E": ROOT / "outputs" / "abl" / "D83E", "b1v": ROOT / "outputs" / "b1v"}


def parts(d):
    return sorted(p.name for p in d.glob("part_*") if (p / "record.json").is_file()) if d.is_dir() else []


def check(arm, ds, carve):
    a = BASE[arm]
    why = []
    rp = a / "records" / f"{ds}__{carve}.json"
    if not rp.is_file():
        return None, ["no carve record (chain not filed)"]
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec.get("problems"):
        why.append(f"record problems {rec['problems'][:2]}")
    miss = [f for f in rec.get("inputs", {}) if not (a / f).exists() and not (ROOT / f).exists()]
    if miss:
        why.append(f"{len(miss)} record inputs missing, e.g. {miss[:2]}")
    cp = parts(a / "cache" / ds / carve)
    if not cp:
        why.append("no cache parts")
    for kind in ("links",) + (("rmatch", "zrc") if rec.get("chained") else ()):
        sub = a / ("chains/" + kind if kind in ("rmatch", "zrc") else kind) / ds / carve
        if parts(sub) != cp:
            why.append(f"{kind}: {len(parts(sub))} filed parts against {len(cp)} cache parts")
    if rec.get("chained") and not (a / "typed" / ds / carve / "record.json").is_file():
        why.append("typed record missing")
    d = a / "look" / ds / carve
    if (d / "pruned.json").exists():
        why.append("already pruned")
    return d, why


todo, tot = [], 0
for arm, carves in ARMS.items():
    for ds, carve in carves.items():
        d, why = check(arm, ds, carve)
        chunks = sorted((d / "chunks").glob("c*.npz")) if d is not None and (d / "chunks").is_dir() else []
        rows = [[p.relative_to(ROOT).as_posix(), p.stat().st_size, p.stat().st_nlink] for p in chunks]
        b = sum(r[1] for r in rows)
        one = sum(r[1] for r in rows if r[2] == 1)
        other = 0
        if d is not None and d.is_dir():
            other = sum(p.stat().st_size for p in d.rglob("*") if p.is_file() and p.parent.name != "chunks")
        state = "PRUNE" if not why and rows else ("nothing" if not why else "KEEP")
        print(f"{state:7s} {arm}/{ds}/{carve}: {len(rows)} chunks {b / 1e9:.2f} GB ({one / 1e9:.2f} on one name); "
              f"look records etc. {other / 1e9:.3f} GB kept" + (f"; {'; '.join(why)}" if why else ""))
        if state == "PRUNE":
            todo.append((arm, ds, carve, d, rows))
            tot += one
free0 = shutil.disk_usage(ROOT).free / 1e9
print(f"would free about {tot / 1e9:.2f} GB from {len(todo)} carves; free now {free0:.2f} GB")
if "--delete" not in sys.argv[1:]:
    print("dry run: nothing removed")
    sys.exit(0)
for arm, ds, carve, d, rows in todo:
    pj = d / "pruned.json"
    readers = [f"cache:{ds}/{carve}", f"links:{ds}/{carve}"]
    rec = json.loads((BASE[arm] / "records" / f"{ds}__{carve}.json").read_text(encoding="utf-8"))
    if rec.get("chained"):
        readers += [f"rmatch:{ds}/{carve}", f"zrc:{ds}/{carve}", f"typed:{ds}/{carve}"]
    doc = {"declared_in": "outputs/host_cleanup/prune_d83_looks.py (zpko.prune_carve's rule)", "stage": "D83", "arm": arm,
           "dataset": ds, "carve": carve, "removed": 0, "bytes": int(sum(r[1] for r in rows)), "readers": sorted(readers),
           "files": rows, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "complete": False}
    pj.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    n = 0
    for r in rows:
        try:
            (ROOT / r[0]).unlink()
            n += 1
        except OSError as e:
            print("failed", r[0], e)
    doc.update({"removed": n, "complete": n == len(rows)})
    pj.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"pruned {arm}/{ds}/{carve}: {n} of {len(rows)} chunks")
print(f"free {free0:.2f} -> {shutil.disk_usage(ROOT).free / 1e9:.2f} GB")
