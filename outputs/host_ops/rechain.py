"""Re-file the steps of a zabl cell that are tied to its cache, after the cache was rebuilt (SWP part C, 11 Nov 2026).

zabl's prune removes a cell's look and cache and keeps its rmatch, zrc, links, typed, kt and carve record. zabl's chain
then rebuilds the look and cache and skips every step whose records exist. Those records still name the old cache: the
links parts' step1_record_sha256 (ltwdiag.part_arrays: "ltw: ... is not the links of ...") and the KT record's
cache_parts (zp1c3.load_kt), so every list and read of the cell stops. The carve record and the gate only re-hash
files, so they pass. On 8 Nov swpc7-build-* rebuilt SNK/squad and RK/metaqa this way; their lists failed on 11 Nov.

`run` = zabl's chain (fills anything missing; a no-op on a built cell), then `check`. When the check finds steps that
are not the ones of the cache on disk, those step folders and the carve record move to
outputs/abl/<arm>/before_rechain/<utc stamp>/ (nothing is deleted), the chain runs again and re-files them on the cache
on disk (rmatch, zrc, links, typed and kt took 3 min for RK/metaqa on 8 Nov), and the check must then pass. zabl.py is
not edited (the S-world's and N-world's verify pin it); its gate runs after this, as its own row.

    python outputs/host_ops/rechain.py check --arm SNK --dataset squad      (read only; exit 1 when the cell is stale)
    python outputs/host_ops/rechain.py run --arm SNK --dataset squad [--lanes 4] [--no-build]
        (--no-build: refuse a cell whose cache is not on disk, for rows that ask for a re-file's memory only)
    python outputs/host_ops/rechain.py selftest
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ABL = ROOT / "outputs" / "abl"
CARVE = "s1eval"                                                   # zabl.CARVE
DERIVED = ("chains/rmatch", "chains/zrc", "links", "typed", "kt")  # zabl's chain steps after the cache, in its order
ASIDE = "before_rechain"
REFUSED = ("SNKF", "NKF")                                          # the worlds' training bases (outputs/pko of the S- and N-world)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def arm_dir(arm, abl=ABL):
    if not arm or arm in REFUSED or "/" in arm or "\\" in arm or arm.startswith("."):
        raise SystemExit(f"rechain: arm {arm!r} refused")
    b = (Path(abl) / arm).resolve()
    if not b.is_relative_to(Path(abl).resolve()) or not b.is_dir():
        raise SystemExit(f"rechain: {b} is not an arm folder under {abl}")
    return b


def check(arm, ds, abl=ABL):
    """(problems, present): why the cell's steps are not the ones of the cache on disk ([] when they are), and the
    derived step folders that hold records ({step: record count})."""
    b = arm_dir(arm, abl)
    cd = b / "cache" / ds / CARVE
    parts = sorted(p for p in cd.glob("part_*of*") if p.is_dir()) if cd.is_dir() else []
    if not parts:
        return [f"no cache parts under {cd}"], {}
    missing = [p.name for p in parts if not (p / "record.json").is_file()]
    if missing:
        return [f"cache parts without a record: {missing[:4]}"], {}
    crec = {p.name: rj(p / "record.json") for p in parts}
    csha = {p.name: sha(p / "record.json") for p in parts}
    newest = max(str(r.get("utc", "")) for r in crec.values())
    problems, present = [], {}
    for sub in DERIVED:
        d = b / sub / ds / CARVE
        recs = sorted(d.rglob("record*.json")) if d.is_dir() else []
        if recs:
            present[sub] = len(recs)
        old = [r for r in recs if str(rj(r).get("utc", "")) < newest]
        if old:
            problems.append(f"{sub}: {len(old)} of {len(recs)} records are older than the cache's newest ({newest})")
    for lp in sorted((b / "links" / ds / CARVE).glob("part_*/record.json")):
        n, h = lp.parent.name, rj(lp)
        if n not in csha or h.get("step1_record_sha256") != csha[n] or h.get("queries") != crec[n].get("queries") \
                or h.get("rows") != crec[n].get("rows"):
            problems.append(f"links/{n}: not the links of the cache part on disk (ltwdiag's check)")
    kr = b / "kt" / ds / CARVE / "record.json"
    if kr.is_file() and rj(kr).get("cache_parts") != csha:
        problems.append("kt: its cache_parts are not the cache's records on disk (zp1c3.load_kt's check)")
    rec = b / "records" / f"{ds}__{CARVE}.json"
    if rec.is_file():
        r = rj(rec)
        if r.get("problems"):
            problems.append(f"carve record names problems {list(r['problems'])[:3]}")
        bad = [f for f, s in r.get("inputs", {}).items() if not (b / f).is_file() or sha(b / f) != s]
        if bad:
            problems.append(f"carve record: {len(bad)} of its inputs are not the files on disk, e.g. {bad[:3]}")
    return problems, present


def move_aside(arm, ds, abl=ABL, stamp=None):
    """The cell's derived step folders and carve record moved (os.replace, same drive) under before_rechain/<stamp>/."""
    b = arm_dir(arm, abl)
    stamp = stamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = b / ASIDE / stamp
    if dest.exists():
        raise SystemExit(f"rechain: {dest} exists")
    moved = []
    for sub in DERIVED:
        src = b / sub / ds / CARVE
        if src.exists():
            to = dest / sub / ds / CARVE
            to.parent.mkdir(parents=True, exist_ok=True)
            os.replace(src, to)
            moved.append(f"{sub}/{ds}/{CARVE}")
    rec = b / "records" / f"{ds}__{CARVE}.json"
    if rec.exists():
        (dest / "records").mkdir(parents=True, exist_ok=True)
        os.replace(rec, dest / "records" / rec.name)
        moved.append(f"records/{rec.name}")
    return dest, moved


def zabl_chain(arm, ds, lanes):
    """zabl's chain as the build rows run it (same command, same working directory)."""
    cmd = [sys.executable, "outputs/host_ops/pylib_run.py", "outputs/mp_unified/zabl.py", "chain", "--arm", arm,
           "--dataset", ds, "--lanes", str(int(lanes)), "--host"]
    log("rechain: " + " ".join(cmd[1:]))
    rc = subprocess.run(cmd, cwd=str(ROOT), check=False).returncode
    if rc:
        raise SystemExit(f"rechain {arm}/{ds}: zabl chain exited {rc}")


def run(arm, ds, lanes=4, abl=ABL, chain=None, stamp=None, no_build=False):
    """no_build: the row asks for a check and re-file only (a few GB), so a cell whose cache is not on disk (pruned) is
    refused instead of rebuilt (a look and cache rebuild needs a build row's asks)."""
    chain = chain or (lambda: zabl_chain(arm, ds, lanes))
    b = arm_dir(arm, abl)
    if no_build:
        first = check(arm, ds, abl)[0]
        if first and first[0].startswith(("no cache parts", "cache parts without a record")):
            raise SystemExit(f"rechain {arm}/{ds}: {first[0]}; --no-build refuses to rebuild the look and cache")
    chain()
    problems, present = check(arm, ds, abl)
    if not problems:
        log(f"rechain {arm}/{ds}: every step is the one of the cache on disk ({present}); nothing moved")
        return 0
    for p in problems:
        log(f"rechain {arm}/{ds}: {p}")
    dest, moved = move_aside(arm, ds, abl, stamp)
    (dest / "rechain.json").write_text(json.dumps(
        {"arm": arm, "dataset": ds, "carve": CARVE, "problems": problems, "moved": moved, "present_before": present,
         "rechain_sha256": sha(__file__), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1),
        encoding="utf-8")
    log(f"rechain {arm}/{ds}: moved {moved} to {dest}")
    chain()
    after, present2 = check(arm, ds, abl)
    if after:
        raise SystemExit(f"rechain {arm}/{ds}: after the chain, still {after[:3]}")
    lost = [s for s in present if s not in present2]
    if lost or not (b / "records" / f"{ds}__{CARVE}.json").is_file():
        raise SystemExit(f"rechain {arm}/{ds}: the chain did not re-file {lost or 'the carve record'}")
    log(f"rechain {arm}/{ds}: re-filed {present2} and the carve record on the cache on disk")
    return 0


def selftest():
    def wj(p, obj):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj), encoding="utf-8")

    with tempfile.TemporaryDirectory() as t:
        abl = Path(t) / "abl"
        b = abl / "T"
        parts = ("part_0of2", "part_1of2")
        for i, n in enumerate(parts):
            wj(b / "cache" / "toy" / CARVE / n / "record.json", {"utc": "2026-11-08T14:0%d:00Z" % i, "queries": 3, "rows": 10 + i})
            wj(b / "links" / "toy" / CARVE / n / "record.json",
               {"step1_record_sha256": "old", "queries": 3, "rows": 10 + i, "utc": "2026-11-08T04:00:00Z"})
        wj(b / "kt" / "toy" / CARVE / "record.json", {"cache_parts": {n: "old" for n in parts}, "utc": "2026-11-08T04:01:00Z"})
        wj(b / "records" / f"toy__{CARVE}.json", {"inputs": {}, "problems": [], "utc": "2026-11-08T14:05:00Z"})
        old_links = (b / "links" / "toy" / CARVE / parts[0] / "record.json").read_bytes()
        calls = []

        def fake_chain():                       # files only what is missing, from the cache on disk (as zabl's chain does)
            calls.append(1)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            csha = {n: sha(b / "cache" / "toy" / CARVE / n / "record.json") for n in parts}
            for n in parts:
                lp = b / "links" / "toy" / CARVE / n / "record.json"
                if not lp.exists():
                    c = rj(b / "cache" / "toy" / CARVE / n / "record.json")
                    wj(lp, {"step1_record_sha256": csha[n], "queries": c["queries"], "rows": c["rows"], "utc": now})
            kp = b / "kt" / "toy" / CARVE / "record.json"
            if not kp.exists():
                wj(kp, {"cache_parts": csha, "utc": now})
            rp = b / "records" / f"toy__{CARVE}.json"
            if not rp.exists():
                ins = {p.relative_to(b).as_posix(): sha(p) for p in sorted(b.rglob("*.json"))
                       if ASIDE not in p.parts and p.parent.name != "records"}
                wj(rp, {"inputs": ins, "problems": [], "utc": now})

        probs, present = check("T", "toy", abl)
        assert any("links/part_0of2" in p for p in probs) and any(p.startswith("kt:") for p in probs), probs
        assert present == {"links": 2, "kt": 1}, present
        assert run("T", "toy", abl=abl, chain=fake_chain, stamp="S1") == 0 and len(calls) == 2
        assert check("T", "toy", abl) == ([], {"links": 2, "kt": 1}), check("T", "toy", abl)
        aside = b / ASIDE / "S1"
        assert (aside / "links" / "toy" / CARVE / parts[0] / "record.json").read_bytes() == old_links
        assert (aside / "records" / f"toy__{CARVE}.json").is_file() and (aside / "kt" / "toy" / CARVE / "record.json").is_file()
        assert rj(aside / "rechain.json")["moved"] == [f"links/toy/{CARVE}", f"kt/toy/{CARVE}", f"records/toy__{CARVE}.json"]
        assert run("T", "toy", abl=abl, chain=fake_chain, stamp="S2") == 0 and not (b / ASIDE / "S2").exists()
        # a carve record whose input changed is a problem; a missing cache is a problem; refused arms
        rp = b / "records" / f"toy__{CARVE}.json"
        r = rj(rp)
        r["inputs"][next(iter(r["inputs"]))] = "x"
        wj(rp, r)
        assert any(p.startswith("carve record:") for p in check("T", "toy", abl)[0])
        assert check("T", "none", abl)[0][0].startswith("no cache parts")
        try:
            run("T", "none", abl=abl, chain=lambda: calls.append("built"), no_build=True)
            raise AssertionError("--no-build ran the chain on a cell without a cache")
        except SystemExit as e:
            assert "--no-build refuses" in str(e) and "built" not in calls, e
        for bad in ("SNKF", "NKF", "../T", ""):
            try:
                arm_dir(bad, abl)
                raise AssertionError(bad)
            except SystemExit:
                pass
        # a chain that does not re-file what was moved fails the run
        wj(b / "links" / "toy" / CARVE / parts[0] / "record.json",
           {"step1_record_sha256": "old", "queries": 3, "rows": 10, "utc": "2026-11-08T04:00:00Z"})
        try:
            run("T", "toy", abl=abl, chain=lambda: None, stamp="S3")
            raise AssertionError("a chain that files nothing passed")
        except SystemExit as e:
            assert "still" in str(e) or "did not re-file" in str(e), e
    print("rechain selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "run"):
        p = sub.add_parser(name)
        p.add_argument("--arm", required=True)
        p.add_argument("--dataset", required=True)
        if name == "run":
            p.add_argument("--lanes", type=int, default=4)
            p.add_argument("--no-build", action="store_true", help="refuse a cell whose cache is not on disk")
    sub.add_parser("selftest")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "check":
        problems, present = check(a.arm, a.dataset)
        print(f"rechain check {a.arm}/{a.dataset}: steps {present}; " + ("OK" if not problems else f"{len(problems)} problems"))
        for p in problems:
            print("  ", p)
        return 1 if problems else 0
    return run(a.arm, a.dataset, a.lanes, no_build=a.no_build)


if __name__ == "__main__":
    sys.exit(main())
