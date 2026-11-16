"""Hard-link the byte-identical copies that dupe_survey.py found (systems code; no science). Read-only unless `link`.

    python outputs/host_ops/dedupe_links.py plan   --survey S.json [--exclude PREFIX ...] [--skip-recent-h 24]
    python outputs/host_ops/dedupe_links.py link   --survey S.json [the same options]
    python outputs/host_ops/dedupe_links.py verify --record R.json [--hash]
    python outputs/host_ops/dedupe_links.py selftest

A survey group is one sha256 held by two or more separate files (dupe_survey.py counts existing hard links once). In
each group the members that are not excluded are linked to one of them, the keeper: the member with the most names
(st_nlink), then the newest mtime. A hard link shares one set of timestamps, so a member takes the keeper's mtime;
choosing the newest means no path looks older (colder) to the cleaner after the link than before. For each member:
  - re-stat: the survey's size, not modified within --skip-recent-h hours, not already the keeper's file; a member
    with other names (st_nlink > 1) is left alone, since replacing one of its names frees nothing;
  - keeper and member are hashed now, and both must equal the survey's sha256;
  - os.link(keeper, <member>.lnk.tmp), and the new name must be the keeper's file (the same file id); then
    os.replace(tmp, member), and the member path must now be the keeper's file. A file that another process holds
    open cannot be replaced on Windows: it is recorded as busy, the temporary name is removed and the member is
    untouched; a rerun picks it up.
Nothing is deleted but the temporary names this run made, and every path keeps the same bytes.

Excluded (left as separate files): paths under --exclude prefixes (relative to the survey root; the defaults are the
host's own ops folders and logs, and outputs/gnc/ and outputs/gnr/, which the parked GNC rows may still write), and
the J5S hold (outputs/j5s/.../J5-s1 and J5-s2, and every .../J5/reads folder, left in place until D80 is done).

Hazard: a hard-linked file is ONE file. A writer that rewrites a linked path in place (open(p, "wb"), np.save,
np.savez) changes every name of it. Link only closed folders that no queued or parked row writes. To rebuild a
linked folder, delete it first or build into a new one: deleting a name removes only that name. `verify` re-checks
that each linked member is still the keeper's file and, with --hash, that the keeper still hashes to the sha256.

Record: outputs/host_dedupe/dedupe_<plan|link>_<stamp>.json (every member with its action and bytes, totals, and the
drive's free space before and after).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

OUT = Path("outputs/host_dedupe")
EXCLUDE_DEFAULT = ("outputs/host_ops/", "outputs/host_archive/", "outputs/host_cleanup/", "outputs/host_dedupe/",
                   "outputs/host_job_logs/", "logs/", "outputs/gnc/", "outputs/gnr/")


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def same_file(a: os.stat_result, b: os.stat_result) -> bool:
    return a.st_dev == b.st_dev and a.st_ino == b.st_ino and a.st_ino != 0


def held(rel: str) -> bool:
    """The J5S hold (left in place until D80 is done)."""
    parts = rel.split("/")
    if "j5s" in parts and ("J5-s1" in parts or "J5-s2" in parts):
        return True
    return any(parts[i] == "J5" and parts[i + 1] == "reads" for i in range(len(parts) - 1))


def norm_root(p: str) -> str:
    """The survey root with '/' separators; a root that is a workspace's outputs/ folder becomes the workspace, so
    relative paths always start at the workspace (outputs/...) and the excludes and the hold match on either machine."""
    r = p.replace("\\", "/").rstrip("/") + "/"
    return r[:-len("outputs/")] if r.endswith("/outputs/") else r


def rel_of(path: str, roots: dict) -> str:
    p = path.replace("\\", "/")
    for r in roots.values():
        if p.lower().startswith(r.lower()):
            return p[len(r):]
    return p


def folder(rel: str, depth: int = 3) -> str:
    parts = rel.split("/")
    return "/".join(parts[:min(depth, len(parts) - 1)])


def plan_groups(survey: dict, excludes: tuple, skip_recent_h: float):
    """Per group: the eligible members (re-stat'ed now) and the reason each other member is left alone."""
    roots = {k: norm_root(v) for k, v in survey["roots"].items()}
    now = time.time()
    out = []
    for g in survey["groups"]:
        elig, left = [], []
        for f in g["files"]:
            p = f["path"]
            rel = rel_of(p, roots)
            why = None
            if held(rel):
                why = "J5S hold"
            elif any(rel.startswith(x) for x in excludes):
                why = "excluded prefix"
            else:
                try:
                    st = os.stat(p)
                except OSError as e:
                    why = f"stat: {e!r}"[:160]
                else:
                    if st.st_size != g["bytes"]:
                        why = f"size {st.st_size} is not the survey's {g['bytes']}"
                    elif now - st.st_mtime < skip_recent_h * 3600:
                        why = f"modified within {skip_recent_h:g} h"
            if why is None:
                elig.append({"path": p, "rel": rel, "st": st})
            else:
                left.append({"path": p, "rel": rel, "why": why})
        keeper = None
        if len(elig) >= 2:
            keeper = max(elig, key=lambda m: (m["st"].st_nlink, m["st"].st_mtime, m["path"]))
        out.append({"g": g, "elig": elig, "left": left, "keeper": keeper})
    return out


def run(survey_path: Path, excludes: tuple, skip_recent_h: float, do_link: bool, out_dir: Path = OUT) -> dict:
    t0 = time.time()
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    anchor = Path(next(iter(survey["roots"].values()))).anchor or "."
    free0 = shutil.disk_usage(anchor).free
    planned = plan_groups(survey, excludes, skip_recent_h)
    rows = []
    tot = defaultdict(int)
    nbytes = defaultdict(int)
    by_folder = defaultdict(int)
    for pg in planned:
        g, keeper = pg["g"], pg["keeper"]
        for m in pg["left"]:
            rows.append({"sha256": g["sha256"], "path": m["rel"], "bytes": g["bytes"], "action": "left", "why": m["why"]})
            tot["left"] += 1
            nbytes["left"] += g["bytes"]
        if keeper is None:
            for m in pg["elig"]:
                rows.append({"sha256": g["sha256"], "path": m["rel"], "bytes": g["bytes"], "action": "left",
                             "why": "no other eligible copy"})
                tot["left"] += 1
                nbytes["left"] += g["bytes"]
            continue
        kp = keeper["path"]
        rows.append({"sha256": g["sha256"], "path": keeper["rel"], "bytes": g["bytes"], "action": "keeper",
                     "nlink": keeper["st"].st_nlink})
        k_ok = None                                  # the keeper's hash is checked once, and only when needed
        for m in pg["elig"]:
            if m is keeper:
                continue
            row = {"sha256": g["sha256"], "path": m["rel"], "bytes": g["bytes"], "keeper": keeper["rel"]}
            st = m["st"]
            if same_file(st, keeper["st"]):
                row.update(action="already")
            elif st.st_nlink > 1:
                row.update(action="left", why=f"has {st.st_nlink - 1} other name(s); replacing one frees nothing")
            elif not do_link:
                row.update(action="would_link")
                by_folder[" = ".join(sorted({folder(keeper["rel"]), folder(m["rel"])}))] += g["bytes"]
            else:
                if k_ok is None:
                    k_ok = sha256(kp) == g["sha256"]
                why = None
                if not k_ok:
                    why = "the keeper no longer hashes to the survey's sha256"
                elif sha256(m["path"]) != g["sha256"]:
                    why = "the member no longer hashes to the survey's sha256"
                if why:
                    row.update(action="left", why=why)
                else:
                    tmp = m["path"] + ".lnk.tmp"
                    try:
                        if os.path.lexists(tmp):
                            os.unlink(tmp)
                        os.link(kp, tmp)
                        if not same_file(os.stat(tmp), os.stat(kp)):
                            raise OSError("the new name is not the keeper's file")
                        os.replace(tmp, m["path"])
                        if not same_file(os.stat(m["path"]), os.stat(kp)):
                            raise OSError("after the replace the member path is not the keeper's file")
                        row.update(action="linked")
                        by_folder[" = ".join(sorted({folder(keeper["rel"]), folder(m["rel"])}))] += g["bytes"]
                    except OSError as e:
                        try:
                            if os.path.lexists(tmp):
                                os.unlink(tmp)       # only the temporary name this run made
                        except OSError:
                            pass
                        row.update(action="busy", why=repr(e)[:200])
            tot[row["action"]] += 1
            nbytes[row["action"]] += g["bytes"] if row["action"] != "keeper" else 0
            rows.append(row)
    free1 = shutil.disk_usage(anchor).free
    mode = "link" if do_link else "plan"
    res = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "host": os.environ.get("COMPUTERNAME"),
           "mode": mode, "survey": str(survey_path), "survey_utc": survey.get("utc"),
           "roots": {k: norm_root(v) for k, v in survey["roots"].items()}, "excludes": list(excludes),
           "skip_recent_h": skip_recent_h, "groups": len(planned), "counts": dict(tot),
           "gb": {k: round(v / 1e9, 3) for k, v in nbytes.items()},
           "gb_by_folder_pair": {k: round(v / 1e9, 3) for k, v in sorted(by_folder.items(), key=lambda kv: -kv[1])},
           "drive_free_gb_before": round(free0 / 1e9, 2), "drive_free_gb_after": round(free1 / 1e9, 2),
           "seconds": round(time.time() - t0, 1), "files": rows}
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"dedupe_{mode}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    i = 1
    while p.exists():                                # never overwrite an earlier record
        p = out_dir / f"dedupe_{mode}_{time.strftime('%Y%m%d-%H%M%S')}_{i}.json"
        i += 1
    tmpj = p.with_name(p.name + ".tmp")
    tmpj.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmpj, p)
    key = "linked" if do_link else "would_link"
    log(f"{mode}: {len(planned)} groups; counts {dict(tot)}; {key} {nbytes[key] / 1e9:.2f} GB; drive free "
        f"{res['drive_free_gb_before']} -> {res['drive_free_gb_after']} GB; {res['seconds']} s -> {p}")
    for k, v in list(res["gb_by_folder_pair"].items())[:25]:
        log(f"  {v:7.3f} GB  {k}")
    res["record"] = str(p)
    return res


def verify(record: Path, do_hash: bool) -> int:
    rec = json.loads(record.read_text(encoding="utf-8"))
    root = next(iter(rec["roots"].values()))
    bad, n = [], 0
    hashed = {}
    for r in rec["files"]:
        if r["action"] not in ("linked", "already"):
            continue
        n += 1
        m, k = Path(root + r["path"]), Path(root + r["keeper"])
        try:
            if not same_file(os.stat(m), os.stat(k)):
                bad.append((r["path"], "no longer the keeper's file"))
                continue
        except OSError as e:
            bad.append((r["path"], f"stat: {e!r}"[:120]))
            continue
        if do_hash:
            if r["keeper"] not in hashed:
                hashed[r["keeper"]] = sha256(k)
            if hashed[r["keeper"]] != r["sha256"]:
                bad.append((r["path"], "the shared file's bytes changed (an in-place write?)"))
    log(f"verify {record}: {n} linked members, {len(bad)} problems" + (f" (hashed {len(hashed)} keepers)" if do_hash else ""))
    for p, why in bad[:50]:
        log(f"  {p}: {why}")
    return 1 if bad else 0


def selftest() -> int:
    """Fixture made by dupe_survey.py itself (the real survey shape), then plan, link and verify."""
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import dupe_survey
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "ws"
        mb = 2 ** 20
        data = os.urandom(3 * mb)
        other = os.urandom(3 * mb)
        files = {
            "outputs/a/cache/x.npy": data, "outputs/b/cache/x.npy": data, "outputs/c/pools/y.npy": data,
            "outputs/gnr/s0/cols/x.npy": data,                           # excluded prefix: stays a separate file
            "outputs/j5s/fits/m/w/J5-s1/x.npy": data,                    # the J5S hold
            "outputs/b1b/fits/arm/J5/reads/x.npy": data,                 # the J5S hold
            "outputs/d/z.npy": other, "outputs/e/z.npy": other,
        }
        for rel, b in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(b)
        old = time.time() - 3 * 86400
        for i, rel in enumerate(files):
            os.utime(root / rel, (old + i, old + i))
        os.link(root / "outputs/e/z.npy", root / "outputs/e/z_twin.npy")   # an existing link: counted once by the survey
        surv = Path(td) / "survey.json"
        dupe_survey.main(["--root", f"ws={root}", "--min-mb", "1", "--skip-recent-min", "0", "--out", str(surv)])
        s = json.loads(surv.read_text(encoding="utf-8"))
        assert s["duplicate_groups"] == 2, s["duplicate_groups"]
        cwd = os.getcwd()
        os.chdir(td)
        try:
            p = run(surv, EXCLUDE_DEFAULT, 24.0, False, Path(td) / "rec")
            assert p["counts"].get("would_link") == 3, p["counts"]       # b, c -> keeper a/b/c; d -> e (e has 2 names)
            assert p["counts"].get("left") == 3, p["counts"]             # gnr + two J5S-hold copies
            mt_before = {rel: os.stat(root / rel).st_mtime for rel in files}
            r = run(surv, EXCLUDE_DEFAULT, 24.0, True, Path(td) / "rec")
            assert r["counts"].get("linked") == 3, r["counts"]
            ka = [x for x in r["files"] if x["action"] == "keeper"
                  and not x["path"].startswith(("outputs/d/", "outputs/e/"))][0]["path"]
            assert ka == "outputs/c/pools/y.npy", ka                     # the newest mtime of the three
            for rel in ("outputs/a/cache/x.npy", "outputs/b/cache/x.npy"):
                assert same_file(os.stat(root / rel), os.stat(root / ka)), rel
                assert (root / rel).read_bytes() == data
                assert os.stat(root / rel).st_mtime >= mt_before[rel]     # never older
            for rel in ("outputs/gnr/s0/cols/x.npy", "outputs/j5s/fits/m/w/J5-s1/x.npy",
                        "outputs/b1b/fits/arm/J5/reads/x.npy"):
                assert not same_file(os.stat(root / rel), os.stat(root / ka)), rel
                assert os.stat(root / rel).st_mtime == mt_before[rel], rel
            kz = [x for x in r["files"] if x["action"] == "keeper"
                  and x["path"].startswith(("outputs/d/", "outputs/e/"))][0]["path"]
            assert kz in ("outputs/e/z.npy", "outputs/e/z_twin.npy"), kz  # the most names wins
            assert same_file(os.stat(root / "outputs/d/z.npy"), os.stat(root / "outputs/e/z_twin.npy"))
            assert not list(root.rglob("*.lnk.tmp"))
            rec = Path(r["record"])
            assert verify(rec, True) == 0
            with open(root / ka, "r+b") as f:                            # an in-place write changes every name
                f.write(b"\0")
            assert verify(rec, True) == 1
            again = run(surv, EXCLUDE_DEFAULT, 24.0, True, Path(td) / "rec")
            assert again["counts"].get("linked", 0) == 0, again["counts"]
        finally:
            os.chdir(cwd)
    log("selftest: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["plan", "link", "verify", "selftest"])
    ap.add_argument("--survey")
    ap.add_argument("--record")
    ap.add_argument("--exclude", action="append", default=[], help="a path prefix relative to the survey root")
    ap.add_argument("--no-default-excludes", action="store_true")
    ap.add_argument("--skip-recent-h", type=float, default=24.0)
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--out-dir", default=str(OUT))
    a = ap.parse_args(argv)
    if a.mode == "selftest":
        return selftest()
    if a.mode == "verify":
        return verify(Path(a.record), a.hash)
    ex = tuple(x.replace("\\", "/") for x in a.exclude) + (() if a.no_default_excludes else EXCLUDE_DEFAULT)
    res = run(Path(a.survey), ex, a.skip_recent_h, a.mode == "link", Path(a.out_dir))
    return 0 if res["counts"].get("busy", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
