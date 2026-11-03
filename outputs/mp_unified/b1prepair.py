"""B1P's regenerable files that the host's clean-up removed on 3 Nov 2026 (outputs/host_cleanup/deleted_auto_20261103-
193143.json: 240 files, every look chunk of B1P's six carves and 28 cache arrays) rebuilt with B1P's own steps into a
scratch root, checked against B1P's records, and only then copied back to exactly the paths the clean-up removed.

B1P's chain skips a look shard whose record exists and a cache part whose record.json exists, so the chain run in place
rebuilds nothing, and a step run in place would rewrite records that other records pin by sha256. Here B1P's steps
(zb1p.py step look|bases|cache with --host, in the chain's environment: BLAS, OpenMP, MKL and numba at 2 threads,
through pylib_run) run on --root outputs/b1m/repair/root, whose pools manifests are byte copies of B1P's (each checked
against the carve record's sha256; a manifest names B1P's own pools file, which is read in place).

    python outputs/mp_unified/b1prepair.py prep                                    # the scratch root: manifests, bases
    python outputs/mp_unified/b1prepair.py look --dataset D --variant V --index i/n
    python outputs/mp_unified/b1prepair.py cache --dataset D --variant V          # the carve's parts, in order
    python outputs/mp_unified/b1prepair.py verify                                 # outputs/b1m/repair/verify.json
    python outputs/mp_unified/b1prepair.py restore                                # outputs/b1m/repair/restore.json
    python outputs/mp_unified/b1prepair.py selftest

verify passes only when (a) every array of every rebuilt cache part of the six carves (not only the removed ones) has
the dtype, shape and file sha256 that B1P's part record gives, and each part's rows, chunks, ids, spans, bases and look
head (but the look records' own shas) are the record's; (b) every rebuilt look record and pools record agrees with
B1P's on every field but seconds, peak memory and time, ids.json and info.json are B1P's byte for byte, every chunk B1P
still holds has its rebuilt twin's sha256, and every removed chunk's twin has the size the clean-up recorded; (c) no
file under outputs/b1p changed after prep began. restore copies only the files the clean-up record lists, never over a
different existing file, each checked after the copy (a cache array against B1P's part record, a chunk against the
sha256 verify recorded). Writes outputs/b1m/repair only, and the restored files.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B1P = ROOT / "outputs" / "b1p"
OUT = ROOT / "outputs" / "b1m" / "repair"
SCR = OUT / "root"
CLEANUP = ROOT / "outputs" / "host_cleanup" / "deleted_auto_20261103-193143.json"
ZB1P = HERE / "zb1p.py"
PYLIB_RUN = ROOT / "outputs" / "host_ops" / "pylib_run.py"
SETTINGS = ("musique", "2wiki", "hotpotqa")
CARVE = {"w": "b1w", "t": "b1t"}
BASES = ("2wiki", "hotpotqa")
VOLATILE = ("seconds", "peak_rss_bytes", "utc")          # a run's time, length and memory; every other field must agree
PART_KEYS = ("dataset", "carve", "part", "nparts", "chunks", "queries", "query_range", "rows", "row_range", "ids",
             "xc_spans", "odd_code_rows", "bases", "numba", "blas_threads", "script_sha256", "module_sha256")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] b1prepair: {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def rel(p):
    return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


def counts(ds, carve, base=B1P):
    """(look shards, cache parts) of a carve as B1P filed it: from its look records' and part folders' names."""
    nl = {int(re.search(r"of(\d+)\.json$", p.name).group(1)) for p in (base / "look" / ds / carve).glob("record_*of*.json")}
    nc = {int(re.search(r"of(\d+)$", p.name).group(1)) for p in (base / "cache" / ds / carve).glob("part_*of*")}
    if len(nl) != 1 or len(nc) != 1:
        raise SystemExit(f"{ds}/{carve}: look shard counts {sorted(nl)}, part counts {sorted(nc)}; one of each expected")
    return nl.pop(), nc.pop()


def env():
    return dict(os.environ, PYTHONIOENCODING="utf-8", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2",
                MKL_NUM_THREADS="2", NUMBA_NUM_THREADS="2")                 # zb1p.chain's environment


def step(kind, ds, v, index=None):
    cmd = [sys.executable, str(PYLIB_RUN), str(ZB1P), "step", kind, "--dataset", ds, "--variant", v] + \
        (["--index", f"{index[0]}/{index[1]}"] if index else []) + ["--host", "--root", rel(SCR)]
    log(f"zb1p step {kind} {ds}/{CARVE[v]} {index or ''} on {rel(SCR)}")
    t = time.time()
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env()).returncode
    if rc:
        raise SystemExit(f"b1prepair: zb1p step {kind} {ds}/{CARVE[v]} {index or ''} exited {rc}")
    log(f"step {kind} {ds}/{CARVE[v]} {index or ''} done in {time.time() - t:.0f} s")


def copy_checked(src, dst, want):
    """src to dst through a temporary name; dst must end with sha256 `want`."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".b1prepair.tmp")
    shutil.copyfile(src, tmp)
    if sha_file(tmp) != want:
        tmp.unlink()
        raise SystemExit(f"b1prepair: the copy of {src} is not sha256 {want[:12]}")
    os.replace(tmp, dst)


def prep():
    started = time.time()
    if (OUT / "prep.json").exists():
        raise SystemExit(f"{rel(OUT / 'prep.json')} exists: the scratch root is set up")
    manifests = {}
    for ds in SETTINGS:
        for v, carve in CARVE.items():
            st = f"{ds}__{carve}"
            rec = read_json(B1P / "records" / f"{st}.json")
            src = B1P / "pools" / f"{st}.json"
            want = rec["inputs"][f"pools/{st}.json"]
            if sha_file(src) != want:
                raise SystemExit(f"{rel(src)} is not the manifest B1P's carve record names")
            man = read_json(src)
            if man["sha256"] != rec["inputs"][f"pools/{st}.npz"] or sha_file(ROOT / man["file"]) != man["sha256"]:
                raise SystemExit(f"{st}: the pools file is not the one B1P's carve record names")
            copy_checked(src, SCR / "pools" / f"{st}.json", want)
            manifests[st] = {"manifest_sha256": want, "pools_file": man["file"], "pools_sha256": man["sha256"],
                             "counts": counts(ds, carve)}
    step("bases", "musique", "w")              # zpko.copy_bases: lean_cache's two bases from step 1, sha-checked
    bases = {}
    for b in BASES:
        mine, theirs = sha_file(SCR / "cache" / f"basis_{b}.npz"), sha_file(B1P / "cache" / f"basis_{b}.npz")
        if mine != theirs:
            raise SystemExit(f"basis {b}: {mine[:12]} here, B1P's {theirs[:12]}")
        bases[b] = mine
    write_json(OUT / "prep.json", {"started": started, "utc": utc(), "root": rel(SCR), "manifests": manifests,
                                   "bases": bases, "cleanup_record": rel(CLEANUP),
                                   "cleanup_record_sha256": sha_file(CLEANUP), "script_sha256": sha_file(__file__)})
    log(f"scratch root {rel(SCR)} set up: {len(manifests)} manifests, bases {sorted(bases)}")
    return 0


def own_chunks(d, i, n):
    """Chunk files of shard i of n under look folder d (chunk c is shard c mod n's), finished or not."""
    if not (d / "chunks").exists():
        return []
    out = []
    for p in (d / "chunks").glob("c*.npz"):
        m = re.fullmatch(r"c(\d+)(\.tmp)?", p.stem)
        if m and int(m.group(1)) % n == i:
            out.append(p)
    return out


def look(ds, v, index):
    carve = CARVE[v]
    nl, _nc = counts(ds, carve)
    i, n = index
    if n != nl:
        raise SystemExit(f"{ds}/{carve}: B1P filed {nl} look shards, not {n}")
    if not (OUT / "prep.json").exists():
        raise SystemExit("prep first")
    d = SCR / "look" / ds / carve
    outs = [d / f"record_{i}of{n}.json", d / f"pools_{i}of{n}.json"]
    if all(p.exists() for p in outs):
        log(f"look {ds}/{carve} {i}/{n}: built already")
        return 0
    stale = own_chunks(d, i, n) + [p for p in outs if p.exists()]
    for p in stale:                              # an unfinished run's files of this shard (the scratch root's own)
        p.unlink()
    if stale:
        log(f"look {ds}/{carve} {i}/{n}: {len(stale)} files of an unfinished run removed")
    step("look", ds, v, index)
    return 0


def cache(ds, v):
    carve = CARVE[v]
    nl, nc = counts(ds, carve)
    d = SCR / "look" / ds / carve
    miss = [i for i in range(nl) if not all((d / f"{k}_{i}of{nl}.json").exists() for k in ("record", "pools"))]
    if miss:
        raise SystemExit(f"{ds}/{carve}: look shards {miss} of {nl} not built")
    for i in range(nc):
        part = SCR / "cache" / ds / carve / f"part_{i}of{nc}"
        if (part / "record.json").exists():
            log(f"cache {ds}/{carve} part {i}/{nc}: built already")
            continue
        if part.exists():
            shutil.rmtree(part)                  # an unfinished part (the scratch root's own)
        step("cache", ds, v, (i, nc))
    return 0


def differ(mine, theirs, skip=VOLATILE):
    keys = sorted(set(mine) | set(theirs))
    return [k for k in keys if k not in skip and mine.get(k) != theirs.get(k)]


def removed_files(rec):
    """The clean-up record's files under outputs/b1p: {path under outputs/b1p: recorded size}."""
    out = {}
    for f in rec["files"]:
        p = str(f[0]).replace("\\", "/")
        if not p.startswith("outputs/b1p/"):
            raise SystemExit(f"{p}: the clean-up record lists a file outside outputs/b1p")
        out[p[len("outputs/b1p/"):]] = int(f[1])
    return out


def verify():
    pj = read_json(OUT / "prep.json")
    if sha_file(CLEANUP) != pj["cleanup_record_sha256"]:
        raise SystemExit("the clean-up record changed after prep")
    removed = removed_files(read_json(CLEANUP))
    problems, cache_rep, look_rep, chunk_sha = [], {}, {}, {}
    for ds in SETTINGS:
        for v, carve in CARVE.items():
            nl, nc = counts(ds, carve)
            key = f"{ds}/{carve}"
            parts = {}
            for i in range(nc):
                pn = f"part_{i}of{nc}"
                mp, tp = SCR / "cache" / ds / carve / pn, B1P / "cache" / ds / carve / pn
                mine, theirs = read_json(mp / "record.json"), read_json(tp / "record.json")
                bad = [k for k in PART_KEYS if mine.get(k) != theirs.get(k)]
                if mine["arrays"] != theirs["arrays"]:
                    bad.append("arrays")
                head_m = {k: x for k, x in mine["look"].items() if k != "records"}
                head_t = {k: x for k, x in theirs["look"].items() if k != "records"}
                if head_m != head_t:
                    bad.append("look head")
                if (mine.get("check") or {}).get("verdict") != (theirs.get("check") or {}).get("verdict"):
                    bad.append("check verdict")
                files_bad = [k for k, a in theirs["arrays"].items() if sha_file(mp / f"{k}.npy") != a["sha256"]]
                if files_bad:
                    bad.append(f"file sha256 {files_bad}")
                parts[pn] = {"arrays": len(theirs["arrays"]), "differs": bad}
                problems += [f"{key} {pn}: {b}" for b in bad]
            cache_rep[key] = parts
            d_m, d_t = SCR / "look" / ds / carve, B1P / "look" / ds / carve
            recs = {}
            listed = set()
            for i in range(nl):
                for kind in ("record", "pools"):
                    nm = f"{kind}_{i}of{nl}.json"
                    mine, theirs = read_json(d_m / nm), read_json(d_t / nm)
                    bad = differ(mine, theirs)
                    recs[nm] = bad
                    problems += [f"{key} look {nm}: {k}" for k in bad]
                    if kind == "record":
                        listed |= set(theirs["chunks"])
            for nm in ("ids.json", "info.json"):
                if sha_file(d_m / nm) != sha_file(d_t / nm):
                    problems.append(f"{key} look {nm}: not B1P's bytes")
            n_chunks = int(read_json(d_t / f"record_0of{nl}.json")["n_chunks"])
            mine_chunks = {p.name for p in (d_m / "chunks").glob("c*.npz")}
            want_chunks = {f"c{c:05d}.npz" for c in range(n_chunks)}
            if listed != set(range(n_chunks)) or mine_chunks != want_chunks:
                problems.append(f"{key} look chunks: {len(mine_chunks)} rebuilt, {n_chunks} in B1P's records")
            same = gone_ok = 0
            for nm in sorted(want_chunks & mine_chunks):
                s = sha_file(d_m / "chunks" / nm)
                chunk_sha[f"look/{ds}/{carve}/chunks/{nm}"] = s
                there = d_t / "chunks" / nm
                if there.exists():
                    if sha_file(there) == s:
                        same += 1
                    else:
                        problems.append(f"{key} chunk {nm}: not B1P's bytes")
                else:
                    sub = f"look/{ds}/{carve}/chunks/{nm}"
                    if sub not in removed:
                        problems.append(f"{key} chunk {nm}: absent from B1P and not in the clean-up record")
                    elif (d_m / "chunks" / nm).stat().st_size != removed[sub]:
                        problems.append(f"{key} chunk {nm}: {(d_m / 'chunks' / nm).stat().st_size} bytes, the "
                                        f"clean-up recorded {removed[sub]}")
                    else:
                        gone_ok += 1
            look_rep[key] = {"records": recs, "chunks": n_chunks, "same_as_b1p": same, "removed_sizes_match": gone_ok}
    for sub in removed:
        if sub.startswith("look/") and sub not in chunk_sha:
            problems.append(f"{sub}: removed and not rebuilt")
        if sub.startswith("cache/") and not (SCR / sub).exists():
            problems.append(f"{sub}: removed and not rebuilt")
    after = [rel(p) for p in B1P.rglob("*") if p.is_file() and p.stat().st_mtime > float(pj["started"])]
    problems += [f"{p}: changed under outputs/b1p after prep" for p in after[:20]]
    rec = {"verdict": "PASS" if not problems else "FAIL", "problems": problems[:200], "n_problems": len(problems),
           "cache": cache_rep, "look": look_rep, "chunk_sha256": chunk_sha, "removed": len(removed),
           "b1p_changed_after_prep": after[:20], "utc": utc(), "script_sha256": sha_file(__file__)}
    write_json(OUT / "verify.json", rec)
    log(f"verify {rec['verdict']}: {len(problems)} problems; "
        f"chunks same as B1P's {sum(x['same_as_b1p'] for x in look_rep.values())}, removed chunks of the recorded "
        f"size {sum(x['removed_sizes_match'] for x in look_rep.values())}")
    for p in problems[:30]:
        log(f"  {p}")
    return 0 if not problems else 1


def expected_sha(sub, ver):
    """What a removed file must hash to: a cache array, B1P's part record; a chunk, verify's rebuilt twin."""
    if sub.startswith("cache/"):
        part = (B1P / sub).parent
        return read_json(part / "record.json")["arrays"][Path(sub).stem]["sha256"]
    return ver["chunk_sha256"][sub]


def restore_files(removed, ver, src_root, dst_root):
    done = {}
    for sub in sorted(removed):
        want = expected_sha(sub, ver)
        src, dst = src_root / sub, dst_root / sub
        if sha_file(src) != want:
            raise SystemExit(f"{sub}: the rebuilt file is not sha256 {want[:12]}")
        if dst.exists():
            if sha_file(dst) != want:
                raise SystemExit(f"{sub}: a different file is there; nothing written over it")
            done[sub] = {"sha256": want, "bytes": dst.stat().st_size, "state": "present"}
            continue
        copy_checked(src, dst, want)
        done[sub] = {"sha256": want, "bytes": dst.stat().st_size, "state": "restored"}
    return done


def restore():
    ver = read_json(OUT / "verify.json")
    if ver["verdict"] != "PASS":
        raise SystemExit("verify did not pass; nothing restored")
    pj = read_json(OUT / "prep.json")
    if sha_file(CLEANUP) != pj["cleanup_record_sha256"]:
        raise SystemExit("the clean-up record changed after prep")
    removed = removed_files(read_json(CLEANUP))
    done = restore_files(removed, ver, SCR, B1P)
    n = sum(1 for x in done.values() if x["state"] == "restored")
    write_json(OUT / "restore.json", {"files": done, "restored": n, "present": len(done) - n,
                                      "bytes": sum(x["bytes"] for x in done.values()), "utc": utc(),
                                      "verify_sha256": sha_file(OUT / "verify.json"), "script_sha256": sha_file(__file__)})
    log(f"restore: {n} files restored, {len(done) - n} present already, of {len(removed)} removed")
    return 0


def selftest():
    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        d = t / "look" / "musique" / "b1t"
        (d / "chunks").mkdir(parents=True)
        for nm in ("c00000.npz", "c00004.npz", "c00001.npz", "c00008.tmp.npz", "c00005.tmp.npz", "x.npz"):
            (d / "chunks" / nm).write_bytes(b"x")
        assert sorted(p.name for p in own_chunks(d, 0, 4)) == ["c00000.npz", "c00004.npz", "c00008.tmp.npz"]
        assert sorted(p.name for p in own_chunks(d, 1, 4)) == ["c00001.npz", "c00005.tmp.npz"]
        assert len(own_chunks(d, 0, 1)) == 5
        (d / "record_0of4.json").write_text("{}", encoding="utf-8")
        (t / "cache" / "musique" / "b1t" / "part_0of4").mkdir(parents=True)
        assert counts("musique", "b1t", t) == (4, 4)
        a = {"chunks": [0, 4], "seconds": 1.0, "utc": "x", "peak_rss_bytes": 5, "threads": {"torch": 4}}
        b = dict(a, seconds=2.0, utc="y", peak_rss_bytes=6)
        assert differ(a, b) == [] and differ(a, dict(b, chunks=[0])) == ["chunks"] and differ(a, {}) != []
        rec = {"files": [["outputs/b1p/look/musique/b1t/chunks/c00000.npz", 3, 0, "regen", None, 1.0],
                         ["outputs\\b1p\\cache\\musique\\b1t\\part_0of4\\node.npy", 3, 0, "regen", None, 1.0]]}
        assert removed_files(rec) == {"look/musique/b1t/chunks/c00000.npz": 3, "cache/musique/b1t/part_0of4/node.npy": 3}
        try:
            removed_files({"files": [["outputs/b1m/x.npy", 1, 0, "regen", None, 1.0]]})
            raise AssertionError("a file outside outputs/b1p accepted")
        except SystemExit:
            pass
        src, dst = t / "src", t / "dst"
        (src / "look" / "c").mkdir(parents=True)
        (src / "look" / "c" / "a.npz").write_bytes(b"abc")
        (src / "look" / "c" / "b.npz").write_bytes(b"def")
        (dst / "look" / "c").mkdir(parents=True)
        (dst / "look" / "c" / "b.npz").write_bytes(b"def")
        ver = {"chunk_sha256": {"look/c/a.npz": hashlib.sha256(b"abc").hexdigest(),
                                "look/c/b.npz": hashlib.sha256(b"def").hexdigest()}}
        done = restore_files({"look/c/a.npz": 3, "look/c/b.npz": 3}, ver, src, dst)
        assert done["look/c/a.npz"]["state"] == "restored" and done["look/c/b.npz"]["state"] == "present"
        assert (dst / "look" / "c" / "a.npz").read_bytes() == b"abc"
        (dst / "look" / "c" / "b.npz").write_bytes(b"zzz")
        try:
            restore_files({"look/c/b.npz": 3}, ver, src, dst)
            raise AssertionError("a different file written over")
        except SystemExit:
            pass
        assert (dst / "look" / "c" / "b.npz").read_bytes() == b"zzz"
        assert not list(dst.rglob("*.b1prepair.tmp"))
    log("selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("prep", "look", "cache", "verify", "restore", "selftest"))
    ap.add_argument("--dataset", choices=SETTINGS)
    ap.add_argument("--variant", choices=tuple(CARVE))
    ap.add_argument("--index")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "prep":
        return prep()
    if a.cmd in ("look", "cache") and not (a.dataset and a.variant):
        raise SystemExit(f"{a.cmd}: --dataset and --variant")
    if a.cmd == "look":
        if not a.index:
            raise SystemExit("look: --index i/n")
        i, n = (int(x) for x in a.index.split("/"))
        if not 0 <= i < n:
            raise SystemExit("--index i/n with 0 <= i < n")
        return look(a.dataset, a.variant, (i, n))
    if a.cmd == "cache":
        return cache(a.dataset, a.variant)
    if a.cmd == "verify":
        return verify()
    return restore()


if __name__ == "__main__":
    sys.exit(main())
