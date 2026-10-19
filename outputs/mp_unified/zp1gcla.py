"""P1g-c's look alias (docs/P1GC_LOOK_ALIAS.md): P1g-b's rebuilt looks read under the look-record sha256s P1g-b's caches
stamped, once every cache part rebuilt from them is P1g-b's part, array for array.

P1g-c rebuilds P1g-b's looks (177 shards) with P1g-b's own commands. A look record holds its run's seconds, time and
peak memory, so a rebuilt record's sha256 is not the one P1g-b's cache parts stamped, and every builder that reads a
look beside a cache (zlink, rmatch, zrc, zg3res, scripts/a7_s6_cols.py) stops on that ("built from another look").
This file checks the rebuilt looks and, where they pass, lets those builders read them under the stamped sha256s:

    verify  one P1g-b cache part built again (lean_cache.build_part, unchanged, at the part's own BLAS threads and
            without numba, as P1g-b's were) from the rebuilt look, in memory: each array's sha256, dtype and shape,
            and the part's ids, ranges, chunks, spans, bases and look head against P1g-b's stored part record
    alias   a carve's alias, ALIAS when every part is IDENTICAL against the look records on disk now, every saved
            original record (outputs/p1gc/p1gb_look_records, each checked against its stamped sha256) equals its
            rebuild outside seconds, utc and peak_rss_bytes, and every lost one holds the saved originals' values of
            the carve-independent fields and its carve's values of the carve fields; REFUSED otherwise
    run     a builder's command (zp1gb.py, zp1gc.py or scripts/a7_s6_cols.py) with lean_cache.look_chunks reading
            P1g-b's look root under the alias: when the carve's alias is ALIAS and its rebuilt sha256s are the ones on
            disk, the head carries the stamped sha256s; otherwise the head is the look's own and the builder stops as
            before

    python outputs/mp_unified/zp1gcla.py verify --dataset 2wiki --carve fit --part 0 --host
    python outputs/mp_unified/zp1gcla.py alias --dataset 2wiki --carve fit
    python outputs/mp_unified/zp1gcla.py run -- outputs/mp_unified/zp1gb.py build --what links --dataset 2wiki \\
        --carve fit --host
    python outputs/mp_unified/zp1gcla.py --selftest

`run` imports nothing heavy before the builder: lean_cache is patched by an import hook when the builder imports it,
so the builder's own thread settings stand.
"""
import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import importlib.abc  # noqa: E402
import importlib.machinery  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import runpy  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

DECLARED = "docs/P1GC_LOOK_ALIAS.md"
STAGE = "P1gcLA"
P1GB = ROOT / "outputs" / "p1gb"
LOOK, CACHE = P1GB / "look", P1GB / "cache"
OUT = ROOT / "outputs" / "p1gc" / "look_alias"
SAVED = ROOT / "outputs" / "p1gc" / "p1gb_look_records"
RUN_KEYS = ("seconds", "utc", "peak_rss_bytes")
# the fields every look record of look_x_six --full holds alike, whatever its carve, and those its carve's hold alike
GLOBAL_KEYS = ("look", "pair", "pair_keys", "functions", "metric_names", "full", "proj", "threads", "limit",
               "copied_from", "copied_from_sha256", "script_sha256", "placement")
CARVE_KEYS = ("dataset", "carve", "n_chunks", "chunk_queries", "carve_queries", "carve_ids_sha256", "columns",
              "column_blocks", "rule", "relations")
PART_KEYS = ("dataset", "carve", "part", "nparts", "chunks", "queries", "query_range", "rows", "row_range", "ids",
             "xc_spans", "bases", "odd_code_rows")
HEAD_KEYS = ("carve_ids_sha256", "carve_queries", "n_chunks", "chunk_queries", "columns", "column_blocks")
SCRIPTS = ("outputs/mp_unified/zp1gb.py", "outputs/mp_unified/zp1gc.py", "scripts/a7_s6_cols.py")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def sha_src(p):
    return sha_bytes(Path(p).read_bytes().replace(b"\r\n", b"\n"))


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def write_json(p, rec):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(json.dumps(rec, indent=1).encode("utf-8"))
    os.replace(tmp, p)


def rel(p):
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


# ── P1g-b's parts and the looks on disk (json only) ──────────────────────────


def stored_parts(ds, carve, cache_root=CACHE):
    """P1g-b's cache parts of ds/carve in order, with their records; they must tile the carve."""
    d = Path(cache_root) / ds / carve
    parts = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not parts:
        raise SystemExit(f"zp1gcla: {d}: no cache part")
    recs = [read_json(p / "record.json") for p in parts]
    q = 0
    for p, r in zip(parts, recs):
        if r["query_range"][0] != q or r["nparts"] != len(parts) or p.name != f"part_{r['part']}of{r['nparts']}":
            raise SystemExit(f"zp1gcla: {d}: the parts do not tile the carve")
        q = r["query_range"][1]
    if q != recs[0]["look"]["carve_queries"]:
        raise SystemExit(f"zp1gcla: {d}: the parts hold {q} of {recs[0]['look']['carve_queries']} questions")
    return parts, recs


def look_records(ds, carve, look_root=LOOK):
    """{record file name: sha256} of the look's records on disk now (lean_cache.look_chunks's head['records'])."""
    d = Path(look_root) / ds / carve
    return {r.name: sha_file(r) for r in sorted(d.glob("record*.json"))}


def part_file(ds, carve, name, out=OUT):
    return Path(out) / "parts" / ds / carve / f"{name}.json"


def alias_file(ds, carve, out=OUT):
    return Path(out) / ds / f"{carve}.json"


# ── verify: one part built again from the rebuilt look ───────────────────────


def open_real(ds, bases, host):
    import lean_cache as LC
    placement = None
    if host:
        import lean_host as LH
        placement = LH.substitute()
    stores = LC.open_stores(bases, CACHE)
    nodes, _freeze = LC.L3.open_nodes([ds])
    return stores, nodes[ds], placement


def verify(ds, carve, part, host=False, look_root=LOOK, cache_root=CACHE, out=OUT, opener=None):
    """P1g-b's part `part` of ds/carve built again from the look on disk, its arrays hashed in memory, against the
    stored part record. Writes the part's verification record; returns it."""
    t0 = time.time()
    parts, recs = stored_parts(ds, carve, cache_root)
    if not 0 <= int(part) < len(parts):
        raise SystemExit(f"zp1gcla: {ds}/{carve} has parts 0..{len(parts) - 1}, not {part}")
    p, r = parts[int(part)], recs[int(part)]
    if os.environ.get("OPENBLAS_NUM_THREADS") != r["blas_threads"]:
        raise SystemExit(f"zp1gcla: BLAS threads {os.environ.get('OPENBLAS_NUM_THREADS')}, the part was built at "
                         f"{r['blas_threads']}")
    import numpy as np
    import lean_cache as LC
    if (LC.L2.njit is not None) != bool(r["numba"]):
        raise SystemExit(f"zp1gcla: numba {'on' if LC.L2.njit is not None else 'off'}, the part was built with it "
                         f"{'on' if r['numba'] else 'off'}")
    bases = list(r["bases"])
    stores, nodes, placement = (opener or open_real)(ds, bases, host)
    for b in bases:
        if stores[b].rec != r["bases"][b]:
            raise SystemExit(f"zp1gcla: basis {b} is not the one P1g-b's part {p.name} was built with")
    head_now = look_records(ds, carve, look_root)

    def npy_sha(_d, _name, a):
        buf = io.BytesIO()
        np.save(buf, np.ascontiguousarray(a))
        return sha_bytes(buf.getvalue())

    tmp = Path(out) / "tmp" / f"{ds}-{carve}-{part}-{os.getpid()}"
    saved = LC.save_npy
    LC.save_npy = npy_sha
    try:
        new = LC.build_part(ds, carve, int(part), len(parts), bases, stores, nodes, 0, Path(look_root), tmp, placement)
    finally:
        LC.save_npy = saved
        shutil.rmtree(tmp, ignore_errors=True)
    diffs = [k for k in PART_KEYS if new.get(k) != r.get(k)]
    diffs += [f"look.{k}" for k in HEAD_KEYS if new["look"].get(k) != r["look"].get(k)]
    if sorted(new["look"]["records"]) != sorted(r["look"]["records"]):
        diffs.append("look.records (names)")
    if new["look"]["records"] != head_now:
        diffs.append("look.records (changed during the build)")
    if sorted(new["arrays"]) != sorted(r["arrays"]):
        diffs.append("arrays (names)")
    diffs += [f"arrays.{k}" for k in sorted(r["arrays"]) if new["arrays"].get(k) != r["arrays"][k]]
    verdict = "IDENTICAL" if not diffs else "DIFFERENT"
    rec = {"stage": STAGE, "declared_in": DECLARED, "what": "verify", "dataset": ds, "carve": carve, "part": p.name,
           "verdict": verdict, "differences": diffs, "stored_record_sha256": sha_file(p / "record.json"),
           "look_records": head_now, "stamped": r["look"]["records"], "arrays_compared": len(r["arrays"]),
           "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "numba": LC.L2.njit is not None,
           "look_root": rel(look_root), "cache_root": rel(cache_root),
           "placement": {k: v for k, v in (placement or {"where": "laptop"}).items() if k != "mirror_root"},
           "zp1gcla_sha256": sha_src(__file__), "lean_cache_sha256": LC.sha_src(LC.__file__),
           "seconds": round(time.time() - t0, 1), "peak_rss_bytes": LC.peak_rss(), "utc": utc()}
    write_json(part_file(ds, carve, p.name, out), rec)
    log(f"zp1gcla verify {ds}/{carve} {p.name}: {verdict}" + (f" ({', '.join(diffs[:6])})" if diffs else
                                                              f" ({len(r['arrays'])} arrays)")
        + f" ({rec['seconds']:.0f}s)")
    return rec


# ── alias: a carve's verdict ─────────────────────────────────────────────────


def strip(rec, keys=RUN_KEYS):
    return {k: v for k, v in rec.items() if k not in keys}


def saved_globals(saved=SAVED):
    """The carve-independent fields' values, from every saved original (they must agree)."""
    vals = {}
    for f in sorted(Path(saved).glob("*/*/record*.json")):
        r = read_json(f)
        for k in GLOBAL_KEYS:
            vals.setdefault(k, set()).add(json.dumps(r.get(k), sort_keys=True))
    return vals


def alias(ds, carve, look_root=LOOK, cache_root=CACHE, out=OUT, saved=SAVED):
    parts, recs = stored_parts(ds, carve, cache_root)
    stamped = recs[0]["look"]["records"]
    now = look_records(ds, carve, look_root)
    probs, records = [], {}
    if any(r["look"]["records"] != stamped for r in recs):
        probs.append("P1g-b's parts stamped different look records")
    if sorted(now) != sorted(stamped):
        probs.append(f"the look's records {sorted(set(now) ^ set(stamped))[:4]} are not the stamped ones")
    pf = {}
    for p, r in zip(parts, recs):
        f = part_file(ds, carve, p.name, out)
        if not f.exists():
            probs.append(f"{p.name} not verified")
            continue
        v = read_json(f)
        pf[p.name] = sha_file(f)
        if v.get("stage") != STAGE or v.get("declared_in") != DECLARED or v.get("what") != "verify":
            probs.append(f"{f.name} is not this stage's verification")
        elif v["verdict"] != "IDENTICAL":
            probs.append(f"{p.name} {v['verdict']} ({', '.join(v['differences'][:4])})")
        elif v["look_records"] != now:
            probs.append(f"{p.name} was verified against other look records than those on disk now")
        elif v["stored_record_sha256"] != sha_file(p / "record.json"):
            probs.append(f"{p.name}: its stored record changed after it was verified")
    gl = saved_globals(saved)
    split = sorted(k for k in GLOBAL_KEYS if len(gl.get(k, ())) != 1)
    if split:
        probs.append(f"the saved originals disagree on {split} (or there are none)")
    sj = Path(saved) / "saved.json"
    listed = read_json(sj) if sj.exists() else {"saved": {}, "lost": {}}
    rebuilt = {n: read_json(Path(look_root) / ds / carve / n) for n in sorted(now)}
    carve_vals = {k: {json.dumps(r.get(k), sort_keys=True) for r in rebuilt.values()} for k in CARVE_KEYS}
    split_c = sorted(k for k, v in carve_vals.items() if len(v) != 1)
    if split_c:
        probs.append(f"the rebuilt records disagree on the carve fields {split_c}")
    for n in sorted(stamped):
        key = f"{ds}/{carve}/{n}"
        f = Path(saved) / ds / carve / n
        e = {"rebuilt": now.get(n), "stamped": stamped[n]}
        if n not in rebuilt:
            records[n] = dict(e, original="lost", same=None)
            continue
        bad_g = [k for k in GLOBAL_KEYS if k not in split and {json.dumps(rebuilt[n].get(k), sort_keys=True)} != gl[k]]
        if bad_g:
            probs.append(f"{n}: {bad_g} differ from the saved originals'")
        if f.exists():
            if sha_file(f) != stamped[n] or listed["saved"].get(key) != stamped[n]:
                probs.append(f"{n}: the saved copy is not the stamped record")
                records[n] = dict(e, original="saved", same=False)
                continue
            same = strip(read_json(f)) == strip(rebuilt[n])
            if not same:
                diff = sorted(k for k in set(read_json(f)) | set(rebuilt[n]) if k not in RUN_KEYS
                              and read_json(f).get(k) != rebuilt[n].get(k))
                probs.append(f"{n}: the rebuild differs from the original outside {list(RUN_KEYS)} ({diff[:4]})")
            records[n] = dict(e, original="saved", same=same, identical_bytes=now.get(n) == stamped[n])
        else:
            if listed["lost"].get(key) != stamped[n]:
                probs.append(f"{n}: neither saved nor listed as lost")
            records[n] = dict(e, original="lost", same=None)
    verdict = "ALIAS" if not probs else "REFUSED"
    rec = {"stage": STAGE, "declared_in": DECLARED, "what": "alias", "dataset": ds, "carve": carve, "verdict": verdict,
           "problems": probs, "records": records, "parts": pf, "run_keys": list(RUN_KEYS),
           "global_keys": list(GLOBAL_KEYS), "carve_keys": list(CARVE_KEYS),
           "originals": {"saved": sum(e["original"] == "saved" for e in records.values()),
                         "lost": sum(e["original"] == "lost" for e in records.values())},
           "saved_list_sha256": sha_file(sj) if sj.exists() else None, "look_root": rel(look_root),
           "cache_root": rel(cache_root), "zp1gcla_sha256": sha_src(__file__), "utc": utc()}
    write_json(alias_file(ds, carve, out), rec)
    log(f"zp1gcla alias {ds}/{carve}: {verdict}, {len(pf)} of {len(parts)} parts verified, originals "
        f"{rec['originals']}" + (f" ({'; '.join(probs[:4])})" if probs else ""))
    return rec


# ── run: lean_cache.look_chunks under the alias ──────────────────────────────


USED = []


def aliased(orig, look=LOOK, out=OUT):
    """look_chunks reading P1g-b's look root under its carves' aliases."""

    def look_chunks(ds, carve, *a, **k):
        d, files, ids, head = orig(ds, carve, *a, **k)
        root = a[0] if a else k.get("root")
        if root is None or Path(root).resolve() != Path(look).resolve():
            return d, files, ids, head
        f = alias_file(ds, carve, out)
        if not f.exists():
            return d, files, ids, head
        al = read_json(f)
        if al.get("stage") != STAGE or al.get("declared_in") != DECLARED or al.get("verdict") != "ALIAS":
            return d, files, ids, head
        if {n: e["rebuilt"] for n, e in al["records"].items()} != head["records"]:
            log(f"zp1gcla: {ds}/{carve}: the look's records changed after its alias; read as they are")
            return d, files, ids, head
        USED.append({"dataset": ds, "carve": carve, "alias_sha256": sha_file(f)})
        log(f"zp1gcla: {ds}/{carve}: the rebuilt look read under P1g-b's stamped record sha256s ({rel(f)})")
        return d, files, ids, dict(head, records={n: e["stamped"] for n, e in al["records"].items()})

    look_chunks.zp1gcla_orig = orig
    return look_chunks


def patch(LC, look=LOOK, out=OUT):
    if not hasattr(LC.look_chunks, "zp1gcla_orig"):
        LC.look_chunks = aliased(LC.look_chunks, look, out)


@contextlib.contextmanager
def on_alias(look=LOOK, out=OUT):
    """In this process: lean_cache.look_chunks under the alias inside, restored after."""
    import lean_cache as LC
    saved = LC.look_chunks
    LC.look_chunks = aliased(getattr(saved, "zp1gcla_orig", saved), look, out)
    try:
        yield
    finally:
        LC.look_chunks = saved


class Hook(importlib.abc.MetaPathFinder):
    """Patches lean_cache when it is first imported, after its own code ran."""

    def __init__(self, look=LOOK, out=OUT):
        self.look, self.out, self.busy = look, out, False

    def find_spec(self, name, path, target=None):
        if name != "lean_cache" or self.busy:
            return None
        self.busy = True
        try:
            spec = importlib.machinery.PathFinder.find_spec(name, path)
        finally:
            self.busy = False
        if spec is None:
            return None
        inner, hook = spec.loader, self

        class Loader(importlib.abc.Loader):
            def create_module(self, spec):
                return inner.create_module(spec)

            def exec_module(self, module):
                inner.exec_module(module)
                patch(module, hook.look, hook.out)

        spec.loader = Loader()
        return spec


def install_hook(look=LOOK, out=OUT):
    if "lean_cache" in sys.modules:
        patch(sys.modules["lean_cache"], look, out)
    else:
        sys.meta_path.insert(0, Hook(look, out))


def run(argv):
    argv = list(argv)
    if argv[:1] == ["--"]:
        argv = argv[1:]
    if not argv:
        raise SystemExit("zp1gcla run -- SCRIPT.py ARGS...")
    script = Path(argv[0]).resolve()
    if rel(script) not in SCRIPTS:
        raise SystemExit(f"zp1gcla run: {rel(script)} is not one of {', '.join(SCRIPTS)}")
    install_hook()
    sys.argv = [str(argv[0])] + argv[1:]
    sys.path.insert(0, str(script.parent))
    log(f"zp1gcla run: {rel(script)} {' '.join(argv[1:])} with P1g-b's looks under the alias ({DECLARED})")
    runpy.run_path(str(script), run_name="__main__")
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def refused(fn, words, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit as e:
        if words not in str(e):
            raise AssertionError(f"stopped for another reason: {e}")
        return str(e)
    raise AssertionError(f"not stopped ({words})")


def selftest():
    import subprocess
    import tempfile
    t0 = time.time()
    os.environ["OPENBLAS_NUM_THREADS"] = os.environ.get("OPENBLAS_NUM_THREADS") or "2"
    import numpy as np
    import lean_cache as LC
    sys.path.insert(0, str(ROOT / "scripts"))
    import a7_s6_cols as BLD
    BT = os.environ["OPENBLAS_NUM_THREADS"]
    tmp = Path(tempfile.mkdtemp(prefix="zp1gcla_"))
    look, cache, out, saved = tmp / "look", tmp / "cache", tmp / "la", tmp / "saved"
    rng = np.random.default_rng(5)
    orig_build = LC.build_part
    try:
        # 1. toy looks (A7's: a KB name and a passage name) and cache parts stamped with their records
        for ds in ("metaqa", "2wiki"):
            BLD.toy_look(tmp, ds, "fit", rng, 40, nparts=2)
        for d in sorted((look).glob("*/*")):
            (one,) = sorted(d.glob("record*.json"))          # the toy's one record, made two shards' records
            rr = read_json(one)
            one.unlink()
            ch = list(rr["chunks"])
            for i, cs in enumerate((ch[:len(ch) // 2 + 1], ch[len(ch) // 2 + 1:])):
                write_json(d / f"record_{i}of2.json", dict(rr, chunks=cs, shard=[i, 2], queries=len(cs), relations="r1",
                                                           **{k: f"g-{k}" for k in GLOBAL_KEYS if k not in ("full", "limit")},
                                                           seconds=1.0, utc="t0",
                                                           peak_rss_bytes=1))
        head0 = {}
        for ds in ("metaqa", "2wiki"):
            head0[ds] = LC.look_chunks(ds, "fit", look)[3]
            for i, p in enumerate(sorted((cache / ds / "fit").glob("part_*"))):
                r = read_json(p / "record.json")
                n = np.load(p / "n.npy")
                r.update({"dataset": ds, "carve": "fit", "part": i, "nparts": 2, "look": head0[ds], "xc_spans": {},
                          "bases": {"b": {"npz_sha256": "x"}}, "odd_code_rows": {"b": 0}, "blas_threads": BT,
                          "numba": LC.L2.njit is not None,
                          "arrays": {"n": {"dtype": str(n.dtype), "shape": list(n.shape), "sha256": LC.sha_file(p / "n.npy")}}})
                write_json(p / "record.json", r)
        # the saved originals and the list, as the host's copy made them
        listed = {"saved": {}, "lost": {}}
        for ds in ("metaqa", "2wiki"):
            for n, s in head0[ds]["records"].items():
                if ds == "2wiki" and n.endswith("1of2.json"):
                    listed["lost"][f"{ds}/fit/{n}"] = s
                    continue
                f = saved / ds / "fit" / n
                f.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(look / ds / "fit" / n, f)
                listed["saved"][f"{ds}/fit/{n}"] = s
        write_json(saved / "saved.json", listed)
        parts_of = {ds: stored_parts(ds, "fit", cache) for ds in ("metaqa", "2wiki")}
        assert all(len(v[0]) == 2 for v in parts_of.values())

        # 2. the rebuild: every record rewritten with a new run's seconds, time and memory -> new sha256s
        def rebuild(ds, change=None):
            for r in sorted((look / ds / "fit").glob("record*.json")):
                rr = read_json(r)
                rr.update(seconds=rr["seconds"] + 7.5, utc="t1", peak_rss_bytes=rr["peak_rss_bytes"] + 3)
                if change:
                    change(r.name, rr)
                write_json(r, rr)
        for ds in ("metaqa", "2wiki"):
            rebuild(ds)
            now_ = look_records(ds, "fit", look)
            assert now_ != head0[ds]["records"] and sorted(now_) == sorted(head0[ds]["records"])
        # A7's check, a builder's, stops without the alias
        r0 = read_json(parts_of["2wiki"][0][0] / "record.json")
        assert r0["look"]["records"] != LC.look_chunks("2wiki", "fit", look)[3]["records"]

        # 3. verify: build_part replaced by one that rebuilds n from the look (through save_npy, so the in-memory hash
        #    is the one tested), and optionally changes an array
        flip = {}

        def fake_build(ds, carve, part, nparts, bases, stores, nodes, check_rows, root, out_root, placement):
            assert check_rows == 0 and Path(root) == look and bases == ["b"]
            d, files, ids, head = LC.look_chunks(ds, carve, root)
            st = read_json(cache / ds / carve / f"part_{part}of{nparts}" / "record.json")
            lo, hi = st["chunks"]
            n = np.concatenate([np.load(files[c])["q_pool_size"] for c in range(lo, hi)]).astype(np.int64)
            q0 = sum(int(np.load(files[c])["q_pool_size"].size) for c in range(lo))
            if flip.get((ds, part)):
                n = n.copy()
                n[0] += 1
            Path(out_root).mkdir(parents=True, exist_ok=True)
            sha = LC.save_npy(Path(out_root), "n", n)
            assert not list(Path(out_root).glob("*.npy")), "an array was written"
            return {"dataset": ds, "carve": carve, "part": part, "nparts": nparts, "chunks": [lo, hi],
                    "queries": int(n.size), "query_range": [q0, q0 + int(n.size)], "rows": st["rows"],
                    "row_range": st["row_range"], "ids": ids[q0:q0 + int(n.size)], "look": head, "xc_spans": {},
                    "bases": {"b": {"npz_sha256": "x"}}, "odd_code_rows": {"b": 0},
                    "arrays": {"n": {"dtype": str(n.dtype), "shape": list(n.shape), "sha256": sha}}}

        class St:
            rec = {"npz_sha256": "x"}

        opener = lambda ds, bases, host: ({"b": St()}, None, None)  # noqa: E731
        LC.build_part = fake_build
        kw = dict(look_root=look, cache_root=cache, out=out, opener=opener)
        for ds in ("metaqa", "2wiki"):
            for i in range(2):
                v = verify(ds, "fit", i, **kw)
                assert v["verdict"] == "IDENTICAL", v
        assert not any((out / "tmp").iterdir())
        # the toy parts' n.npy hold int64 n: the in-memory sha256 equals the stored file's
        flip[("metaqa", 1)] = True
        v = verify("metaqa", "fit", 1, **kw)
        assert v["verdict"] == "DIFFERENT" and v["differences"] == ["arrays.n"], v
        flip.clear()
        # the preconditions: BLAS threads, numba, the bases, the part
        os.environ["OPENBLAS_NUM_THREADS"] = "7"
        refused(verify, "BLAS threads", "metaqa", "fit", 0, **kw)
        os.environ["OPENBLAS_NUM_THREADS"] = BT
        St.rec = {"npz_sha256": "y"}
        refused(verify, "basis b", "metaqa", "fit", 0, **kw)
        St.rec = {"npz_sha256": "x"}
        refused(verify, "has parts", "metaqa", "fit", 2, **kw)
        nb = LC.L2.njit
        LC.L2.njit = (lambda *a, **k: None) if nb is None else None
        refused(verify, "numba", "metaqa", "fit", 0, **kw)
        LC.L2.njit = nb

        # 4. alias: metaqa DIFFERENT in part 1 -> REFUSED; again IDENTICAL -> ALIAS; 2wiki with one lost original
        akw = dict(look_root=look, cache_root=cache, out=out, saved=saved)
        a = alias("metaqa", "fit", **akw)
        assert a["verdict"] == "REFUSED" and any("DIFFERENT" in x for x in a["problems"]), a["problems"]
        verify("metaqa", "fit", 1, **kw)
        a = alias("metaqa", "fit", **akw)
        assert a["verdict"] == "ALIAS" and a["originals"] == {"saved": 2, "lost": 0}, a["problems"]
        assert all(e["same"] and not e["identical_bytes"] for e in a["records"].values())
        a = alias("2wiki", "fit", **akw)
        assert a["verdict"] == "ALIAS" and a["originals"] == {"saved": 1, "lost": 1}, a["problems"]
        # a rebuilt record differing from its saved original outside the run keys
        rebuild("metaqa", lambda n, rr: rr.update(queries=rr["queries"] + 1) if n.endswith("0of2.json") else None)
        for i in range(2):
            verify("metaqa", "fit", i, **kw)
        a = alias("metaqa", "fit", **akw)
        assert a["verdict"] == "REFUSED" and any("differs from the original" in x for x in a["problems"]), a["problems"]
        rebuild("metaqa", lambda n, rr: rr.update(queries=rr["queries"] - 1) if n.endswith("0of2.json") else None)
        # a lost one with another carve-independent value, or another carve value than its carve's
        rebuild("2wiki", lambda n, rr: rr.update(threads="other") if n.endswith("1of2.json") else None)
        for i in range(2):
            verify("2wiki", "fit", i, **kw)
        a = alias("2wiki", "fit", **akw)
        assert a["verdict"] == "REFUSED" and any("differ from the saved originals" in x for x in a["problems"])
        rebuild("2wiki", lambda n, rr: rr.update(threads="g-threads", relations="r2") if n.endswith("1of2.json") else None)
        for i in range(2):
            verify("2wiki", "fit", i, **kw)
        a = alias("2wiki", "fit", **akw)
        assert a["verdict"] == "REFUSED" and any("carve fields" in x for x in a["problems"]), a["problems"]
        rebuild("2wiki", lambda n, rr: rr.update(relations="r1") if n.endswith("1of2.json") else None)
        # a part verified against older records, a part missing, a saved copy that is not the stamped record
        a = alias("2wiki", "fit", **akw)
        assert a["verdict"] == "REFUSED" and any("other look records" in x for x in a["problems"]), a["problems"]
        for ds in ("metaqa", "2wiki"):
            for i in range(2):
                verify(ds, "fit", i, **kw)
        pf = part_file("2wiki", "fit", "part_1of2", out)
        keep = pf.read_bytes()
        pf.unlink()
        assert any("not verified" in x for x in alias("2wiki", "fit", **akw)["problems"])
        pf.write_bytes(keep)
        sf = saved / "2wiki" / "fit" / "record_0of2.json"
        keep = sf.read_bytes()
        sf.write_bytes(keep + b" ")
        assert any("saved copy" in x for x in alias("2wiki", "fit", **akw)["problems"])
        sf.write_bytes(keep)
        for ds in ("metaqa", "2wiki"):
            assert alias(ds, "fit", **akw)["verdict"] == "ALIAS"

        # 5. the alias in this process: A7's check passes inside, fails outside; another root, a changed record and a
        #    REFUSED alias read as they are
        with on_alias(look, out):
            h = LC.look_chunks("2wiki", "fit", look)[3]
            assert h["records"] == r0["look"]["records"] and USED[-1]["carve"] == "fit"
            other = tmp / "look2"
            shutil.copytree(look, other)
            assert LC.look_chunks("2wiki", "fit", other)[3]["records"] == look_records("2wiki", "fit", other)
        assert LC.look_chunks("2wiki", "fit", look)[3]["records"] != r0["look"]["records"]
        assert not hasattr(LC.look_chunks, "zp1gcla_orig")
        rf = look / "metaqa" / "fit" / "record_0of2.json"
        keep = rf.read_bytes()
        rf.write_bytes(keep + b"\n")
        with on_alias(look, out):
            assert LC.look_chunks("metaqa", "fit", look)[3]["records"] == look_records("metaqa", "fit", look)
        rf.write_bytes(keep)
        af = alias_file("2wiki", "fit", out)
        keep = af.read_bytes()
        al = read_json(af)
        al["verdict"] = "REFUSED"
        write_json(af, al)
        with on_alias(look, out):
            assert LC.look_chunks("2wiki", "fit", look)[3]["records"] == look_records("2wiki", "fit", look)
        af.write_bytes(keep)

        # 6. the import hook in a fresh process: lean_cache patched as it is imported, nothing heavy before
        code = ("import sys; sys.path.insert(0, %r); import zp1gcla as A; from pathlib import Path; "
                "assert 'numpy' not in sys.modules; A.install_hook(Path(%r), Path(%r)); import lean_cache as LC; "
                "import json; print(json.dumps(LC.look_chunks('2wiki', 'fit', Path(%r))[3]['records']))"
                % (str(HERE), str(look), str(out), str(look)))
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        assert res.returncode == 0, res.stderr[-2000:]
        assert json.loads(res.stdout.strip().splitlines()[-1]) == r0["look"]["records"]
        refused(run, "is not one of", ["--", "outputs/mp_unified/zlink.py", "build"])
    finally:
        LC.build_part = orig_build
        shutil.rmtree(tmp, ignore_errors=True)
    log(f"zp1gcla selftest: the rebuild's new record sha256s stop a builder's check and the alias lets it read the "
        f"stamped ones; verify hashes in memory, writes no array and finds a changed array, refuses other BLAS "
        f"threads, numba, bases or parts; alias is ALIAS only with every part IDENTICAL against the records on disk, "
        f"saved originals equal outside {list(RUN_KEYS)}, lost ones holding the global and carve values; the alias "
        f"is restored after, ignores other roots, changed records and refused aliases; the import hook patches "
        f"lean_cache in a fresh process; run takes only the three builders ({time.time() - t0:.1f}s)")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["run"]:
        return run(argv[1:])
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", choices=("verify", "alias"))
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--part", type=int)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not (a.dataset and a.carve):
        raise SystemExit("zp1gcla: --dataset and --carve")
    if a.cmd == "verify":
        if a.part is None:
            raise SystemExit("zp1gcla verify: --part")
        parts, recs = stored_parts(a.dataset, a.carve)
        if 0 <= a.part < len(recs):
            os.environ["OPENBLAS_NUM_THREADS"] = recs[a.part]["blas_threads"]   # before numpy: the part's own threads
        verify(a.dataset, a.carve, a.part, a.host)
        return 0
    if a.cmd == "alias":
        return 0 if alias(a.dataset, a.carve)["verdict"] == "ALIAS" else 1
    ap.error("verify, alias, run or --selftest")


if __name__ == "__main__":
    sys.exit(main())
