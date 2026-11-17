"""D91's linked records that name D83's arm roots (17 Nov). zd91 hard-links D83's kt, d83kt, chains, pools, look, settings and
cache files into D91's arms. Some of them are records holding a root path, e.g. kt/<ds>/<carve>/record.json
cache_root = outputs/abl/D83F/cache. zp1c3.load_kt and zu2d.load_via check that root against the root they read on
(Path.resolve), so in the D91 world (outputs/pko -> outputs/abl/D91F) every zrckpn fit stopped there.

    python outputs/host_ops/d91_roots.py survey      (read only: each record naming another arm's root, by arm, folder and key)
    python outputs/host_ops/d91_roots.py kt [--dry]  (each arm's kt/<ds>/<carve>/record.json, still D83's hard link, is replaced
                                                      by D91's own file: D83's record with cache_root = outputs/abl/<arm>/cache
                                                      and a d91 block naming D83's record and its sha256. kt.npy, n.npy and
                                                      ids.json stay D83's links. load_kt still checks the arrays' shas and
                                                      the cache parts' shas. A record already D91's is skipped. The ledger
                                                      is outputs/abl/<arm>/d91_kt_roots.json.)
    python outputs/host_ops/d91_roots.py check       (read only: each kt record names its arm's cache root and its arrays' shas)

The other records the survey lists (look pools_*.json pools_file, pruned.json, pools manifests, settings) name D83's
paths only as notes. No loader compares them with the root it reads on (D91's lists and cols ran on them), so they
stay D83's links.
"""
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"D91F": "outputs/abl/D83F", "D91E": "outputs/abl/D83E", "D91B": "outputs/b1v"}
TOPS = ("kt", "d83kt", "chains", "pools", "look", "settings", "cache")


def walk_strings(obj, path=()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk_strings(v, path + (str(k),))
    elif isinstance(obj, list):
        for i, v in enumerate(obj[:50]):
            yield from walk_strings(v, path + ("[]",))
    elif isinstance(obj, str):
        yield path, obj


def survey():
    groups = defaultdict(list)
    for arm, src in ARMS.items():
        base = ROOT / "outputs" / "abl" / arm
        for top in TOPS:
            t = base / top
            if not t.exists():
                continue
            for dp, _dn, fn in os.walk(t):
                for f in fn:
                    if not f.endswith(".json"):
                        continue
                    p = Path(dp) / f
                    if p.stat().st_size > 5_000_000:
                        continue
                    try:
                        rec = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    for kp, v in walk_strings(rec):
                        if src in v.replace("\\", "/"):
                            rp = p.relative_to(base).as_posix()
                            groups[(arm, top, f, ".".join(kp))].append((rp, v, os.stat(p).st_nlink))
    out = []
    for (arm, top, f, key), rows in sorted(groups.items()):
        out.append({"arm": arm, "top": top, "file": f, "key": key, "n": len(rows), "example": rows[0][0], "value": rows[0][1],
                    "nlink_min": min(r[2] for r in rows)})
    print(json.dumps(out, indent=1))
    return out


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def kt_records(arm):
    base = ROOT / "outputs" / "abl" / arm / "kt"
    return sorted(base.glob("*/*/record.json"))


def kt(dry=False):
    done = {}
    for arm, src in ARMS.items():
        ledger, new_root = [], f"outputs/abl/{arm}/cache"
        recs = kt_records(arm)
        if not recs:
            raise SystemExit(f"d91_roots: no kt records under outputs/abl/{arm}/kt")
        for rp in recs:
            ds, carve = rp.parent.parent.name, rp.parent.name
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if (rec.get("d91") or {}).get("arm") == arm and rec.get("cache_root") == new_root:
                ledger.append({"kt": f"{ds}/{carve}", "skipped": "already D91's"})
                continue
            srp = ROOT / src / "kt" / ds / carve / "record.json"
            if rec.get("cache_root") != f"{src}/cache":
                raise SystemExit(f"d91_roots: {rp} names cache root {rec.get('cache_root')}, not {src}/cache: nothing changed")
            if not (srp.is_file() and os.path.samefile(rp, srp)):
                raise SystemExit(f"d91_roots: {rp} is not D83's record {srp} (not the same file): nothing changed")
            for k, f in (("kt", "kt.npy"), ("n", "n.npy")):
                a = rp.parent / f
                if not (os.path.samefile(a, srp.parent / f) and sha(a) == rec["arrays"][k]["sha256"]):
                    raise SystemExit(f"d91_roots: {a} is not D83's {k} array with its record's sha256")
            old_sha = sha(rp)
            new = dict(rec, cache_root=new_root,
                       d91={"stage": "D91", "arm": arm, "declared_in": "docs/D91_D83_WITHOUT_LABELS.md",
                            "from": srp.relative_to(ROOT).as_posix(), "from_sha256": old_sha,
                            "changed": {"cache_root": [rec["cache_root"], new_root]},
                            "why": "zp1c3.load_kt checks the record's cache_root against the root it reads on; D91's cache "
                                   "parts are D83's files (cache_parts still checked)",
                            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            ledger.append({"kt": f"{ds}/{carve}", "from": new["d91"]["from"], "from_sha256": old_sha})
            if dry:
                continue
            tmp = rp.with_name("record.json.d91tmp")
            tmp.write_text(json.dumps(new, indent=1), encoding="utf-8")
            os.replace(tmp, rp)                     # a new file under the name: D83's record (the other name) is untouched
            if sha(srp) != old_sha or os.path.samefile(rp, srp):
                raise SystemExit(f"d91_roots: {srp} changed or is still the same file as {rp}")
            ledger[-1]["sha256"] = sha(rp)
        done[arm] = ledger
        if not dry:
            lp = ROOT / "outputs" / "abl" / arm / "d91_kt_roots.json"
            lp.write_text(json.dumps({"stage": "D91", "arm": arm, "records": ledger,
                                      "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1), encoding="utf-8")
    print(json.dumps({a: [r["kt"] + (" (skipped)" if "skipped" in r else "") for r in l] for a, l in done.items()}, indent=1))
    print(("DRY: " if dry else "") + f"{sum(1 for l in done.values() for r in l if 'skipped' not in r)} kt records "
          + ("would be D91's own" if dry else "are now D91's own"))
    return 0


def check():
    bad = []
    for arm in ARMS:
        want = (ROOT / "outputs" / "abl" / arm / "cache").resolve()
        for rp in kt_records(arm):
            rec = json.loads(rp.read_text(encoding="utf-8"))
            if (ROOT / rec.get("cache_root", "")).resolve() != want or (rec.get("d91") or {}).get("arm") != arm:
                bad.append(rp.relative_to(ROOT).as_posix())
            for k, f in (("kt", "kt.npy"), ("n", "n.npy")):
                if sha(rp.parent / f) != rec["arrays"][k]["sha256"]:
                    bad.append((rp.parent / f).relative_to(ROOT).as_posix())
    print(json.dumps({"bad": bad, "arms": list(ARMS)}))
    if bad:
        raise SystemExit(f"d91_roots check: {len(bad)} problems")
    print("d91_roots check: OK")
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["survey"]:
        survey()
    elif a[:1] == ["kt"]:
        kt(dry="--dry" in a)
    elif a[:1] == ["check"]:
        check()
    else:
        raise SystemExit(__doc__)
