"""Build cache for the chainscore line (systems code; untracked, LF; no science). Every run of a part repeats the same
work on the same builds before it trains: chainscore19's prep (the training labels), chainscore21's lb_offsets (from the
raw counts), chainscore22's normalise (the population map; up to two minutes a run under pq) and chainscore19's
input_stats. Here that work runs once per (build, kind, map) and is stored as a delta beside the build: only the arrays
the pipeline adds or rewrites, as uncompressed .npy files that a run memory-maps (runs on one host share the pages).
The build itself is loaded as before (CS.load: its npz is stored uncompressed, about 1.5 s per GB).
    entry   <root>/<stem>.<kind>.<map>/ (root: outputs/mp_unified/cache)
              one .npy per array the pipeline adds or rewrites: prep's on, gflag and train_rows (kind train); the map's
              XN, FI, FS and FR (a map other than none)
              lb.npy        lb's offsets, counted from the raw build
              stats.npz     input_stats of the mapped build (kind train), as a run takes them from its first build
              entry.json    the source's sha256, size and mtime, the kind, the map and its info, prep's scalars, the
                            sha256 of every stored array, of lb and of the stats, the code's sha256s
    kinds   train   CS.load, C19.prep, C21.lb_offsets, C22.normalise (a training build: chainscore22's train order)
            plain   CS.load, C21.lb_offsets, C22.normalise (the select carve and the reads)
A loaded entry is the pipeline's output array for array (dtype, shape, memory order and bytes): check runs the
pipeline fresh and compares, and a run's R0 (a cached arm against the same arm run without the cache) checks it end
to end. An entry made from another source (sha256), map or code is refused.
    python outputs/mp_unified/cs_cache.py build --kind train --map pq --src outputs/mp_unified/lean/cs19-mq-fit-id.npz
    python outputs/mp_unified/cs_cache.py check --kind train --map pq --src outputs/mp_unified/lean/cs19-mq-fit-id.npz
    python outputs/mp_unified/cs_cache.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore22 as C22  # noqa: E402

ROOT = HERE / "cache"
KINDS = ("train", "plain")
PREP_KEYS = ("on", "gflag", "train_rows")
MAP_KEYS = ("XN", "FI", "FS", "FR")
MMAP_MIN = 1 << 20      # arrays of at least this many bytes are memory-mapped; smaller ones are read in
_SHA = {}


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def sha(p):
    """sha256 of a file, kept per (path, size, mtime) for the life of the process."""
    p = Path(p)
    s = p.stat()
    k = (str(p.resolve()), s.st_size, s.st_mtime_ns)
    if k not in _SHA:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        _SHA[k] = h.hexdigest()
    return _SHA[k]


def arr_sha(x):
    x = np.asarray(x)
    h = hashlib.sha256(f"{x.dtype.str}|{x.shape}|{int(np.isfortran(x))}|".encode())
    h.update(np.ascontiguousarray(x).view(np.uint8).reshape(-1).data if x.size else b"")
    return h.hexdigest()


def code_shas():
    return {"cs_cache": sha(__file__), "chainscore18": sha(CS.__file__), "chainscore19": sha(C19.__file__),
            "chainscore21": sha(C21.__file__), "chainscore22": sha(C22.__file__)}


def entry_dir(src, kind, norm, root=ROOT):
    if kind not in KINDS:
        raise SystemExit(f"kind is one of {KINDS}, not {kind}")
    if norm not in C22.NORMS:
        raise SystemExit(f"map is one of {C22.NORMS}, not {norm}")
    return Path(root) / f"{Path(src).stem}.{kind}.{norm}"


def pipeline(src, kind, norm):
    """The runs' own statements (chainscore22's train_cmd order): (d, lb, map info)."""
    d = CS.load(src)
    if kind == "train":
        d = C19.prep(d)
    lb = C21.lb_offsets(d)
    info = C22.normalise(d, norm)
    return d, lb, info


def stored_keys(kind, norm):
    return (PREP_KEYS if kind == "train" else ()) + (MAP_KEYS if norm != "none" else ())


def build(src, kind, norm, root=ROOT):
    """Run the pipeline once and store its delta. An existing entry for this source, kind, map and code is kept."""
    t0 = time.time()
    out = entry_dir(src, kind, norm, root)
    if out.exists():
        try:
            load(src, kind, norm, root)
            log(f"{out.name}: present and current; kept")
            return json.loads((out / "entry.json").read_text(encoding="utf-8"))
        except SystemExit as e:
            raise SystemExit(f"{out}: an entry that does not load ({e}); move it aside first") from None
    d, lb, info = pipeline(src, kind, norm)
    tmp = out.with_name(out.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    arrays = {}
    for k in stored_keys(kind, norm):
        x = d[k]
        np.save(tmp / f"{k}.npy", x, allow_pickle=False)
        arrays[k] = {"dtype": x.dtype.str, "shape": list(x.shape), "fortran": bool(np.isfortran(x)),
                     "sha256": arr_sha(x)}
    np.save(tmp / "lb.npy", lb, allow_pickle=False)
    st = None
    if kind == "train":
        st = C19.input_stats(d)
        np.savez(tmp / "stats.npz", **st)
    s = Path(src).stat()
    ent = {"look": "cs_cache", "source": str(src), "source_sha256": sha(src), "source_size": s.st_size,
           "kind": kind, "norm": norm, "map_info": info, "scalars": {"n_regimes": d.get("n_regimes")},
           "arrays": arrays, "lb_sha256": arr_sha(lb),
           "stats_sha256": None if st is None else {k: arr_sha(v) for k, v in st.items()},
           "code": code_shas(), "seconds": None}
    ent["seconds"] = round(time.time() - t0, 1)
    (tmp / "entry.json").write_text(json.dumps(ent, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, out)
    log(f"{out.name}: stored {sorted(arrays)} + lb{' + stats' if st is not None else ''} in {ent['seconds']}s "
        f"({sum(f.stat().st_size for f in out.iterdir()) / 1e9:.3f} GB)")
    return ent


def _np_load(p, nbytes):
    return np.load(p, mmap_mode="r" if nbytes >= MMAP_MIN else None, allow_pickle=False)


def load(src, kind, norm, root=ROOT, verify=False):
    """(d, lb, map info, stats or None): the build as CS.load gives it, with the stored arrays in place of the
    pipeline's. verify: re-hash every stored array (check does; a run need not)."""
    ed = entry_dir(src, kind, norm, root)
    jp = ed / "entry.json"
    if not jp.exists():
        raise SystemExit(f"{ed}: no cache entry (build it first: cs_cache.py build --kind {kind} --map {norm} "
                         f"--src {src})")
    ent = json.loads(jp.read_text(encoding="utf-8"))
    if ent.get("look") != "cs_cache" or ent.get("kind") != kind or ent.get("norm") != norm:
        raise SystemExit(f"{ed}: not this kind and map")
    if ent["source_sha256"] != sha(src):
        raise SystemExit(f"{ed}: made from another {Path(src).name} (sha256 {ent['source_sha256'][:12]})")
    cur = code_shas()
    bad = [k for k in cur if k != "cs_cache" and ent["code"].get(k) != cur[k]]
    if bad:
        raise SystemExit(f"{ed}: made by other code ({bad})")
    d = CS.load(src)
    for k, spec in ent["arrays"].items():
        x = _np_load(ed / f"{k}.npy", int(np.prod(spec["shape"])) * np.dtype(spec["dtype"]).itemsize)
        if x.dtype.str != spec["dtype"] or list(x.shape) != spec["shape"] or bool(np.isfortran(x)) != spec["fortran"]:
            raise SystemExit(f"{ed}: {k} is not the stored array")
        if verify and arr_sha(x) != spec["sha256"]:
            raise SystemExit(f"{ed}: {k}'s bytes are not the stored ones")
        d[k] = x
    for k, v in (ent.get("scalars") or {}).items():
        if v is not None:
            d[k] = v
    lb = np.load(ed / "lb.npy", allow_pickle=False)
    if verify and arr_sha(lb) != ent["lb_sha256"]:
        raise SystemExit(f"{ed}: lb's bytes are not the stored ones")
    st = None
    if (ed / "stats.npz").exists():
        with np.load(ed / "stats.npz", allow_pickle=False) as z:
            st = {k: z[k] for k in z.files}
        if verify and {k: arr_sha(v) for k, v in st.items()} != ent["stats_sha256"]:
            raise SystemExit(f"{ed}: the stats are not the stored ones")
    return d, lb, ent["map_info"], st


def compare(src, kind, norm, root=ROOT):
    """The pipeline run fresh against the entry: every array of the dict (dtype, shape, order, bytes), lb and the
    stats. Returns the list of differences (empty: identical)."""
    d0, lb0, info0, = pipeline(src, kind, norm)
    d1, lb1, info1, st1 = load(src, kind, norm, root, verify=True)
    diff = []
    if sorted(d0) != sorted(d1):
        diff.append(f"keys {sorted(set(d0) ^ set(d1))}")
    for k in sorted(set(d0) & set(d1)):
        a, b = d0[k], d1[k]
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or arr_sha(a) != arr_sha(b):
                diff.append(k)
        elif a != b:
            diff.append(k)
    if arr_sha(lb0) != arr_sha(lb1):
        diff.append("lb")
    i0 = json.loads(json.dumps({k: v for k, v in info0.items() if k != "seconds"}, default=str))   # as stored
    if i0 != {k: v for k, v in info1.items() if k != "seconds"}:
        diff.append("map info")
    if kind == "train":
        st0 = C19.input_stats(d0)
        if st1 is None or any(arr_sha(st0[k]) != arr_sha(st1[k]) for k in st0):
            diff.append("stats")
    return diff


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    # arr_sha separates dtype, shape and order, and equal bytes give equal hashes
    x = rng.normal(size=(5, 4)).astype(np.float32)
    assert arr_sha(x) == arr_sha(x.copy()) and arr_sha(x) != arr_sha(x.astype(np.float64))
    assert arr_sha(x) != arr_sha(x.reshape(4, 5)) and arr_sha(x) != arr_sha(np.asfortranarray(x))
    assert arr_sha(np.zeros(0, np.int32)) != arr_sha(np.zeros(0, np.int64))
    # a toy build through build / load / compare under every map (chainscore22's toy build plus the keys prep and
    # lb_offsets read)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        d = C22.toy_build(rng, rows=30, nodes=40, chains=20, regimes=3)
        P = int(d["rowoff"][-1])
        N = d["n"].size
        d["FI"][:, CS.LOGC] = 0.0
        hop = d["FI"][:, C21.HOPC]
        lev = rng.integers(0, 2, size=P)
        row = np.repeat(np.arange(N), np.diff(d["rowoff"]))
        kl = row * 2 + lev
        d["FI"][:, CS.LOGC] = np.log(np.bincount(kl, minlength=N * 2)[kl])
        assert np.all(hop.sum(1) == 1)
        d.update({"level_names": np.array(["none", "dir"]), "p_lev": lev.astype(np.int8),
                  "xn_off": np.r_[0, np.cumsum(d["n"])].astype(np.int64), "gold_off": np.arange(0, N + 1),
                  "gold": rng.integers(0, 40, size=N).astype(np.int32),
                  "seed0": rng.random(int(d["n"].sum())) < 0.1, "meta": np.array(json.dumps({"toy": True}))})
        src = td / "toy.npz"
        np.savez(src, **d)
        prep0 = C19.prep

        def prep_toy(d_):       # the toy has no chain labels; prep's node-gold half and its scalars
            d_["on"] = np.zeros(int(d_["rowoff"][-1]), bool)
            d_["gflag"] = np.zeros(int(d_["xn_off"][-1]), bool)
            d_["train_rows"] = np.arange(d_["n"].size)
            d_["n_regimes"] = int(d_["FR"].shape[0])
            return d_

        C19.prep = prep_toy
        stats0 = C19.input_stats
        C19.input_stats = lambda d_: {"mn": np.asarray(d_["XN"], np.float64).mean(0), "sn": np.ones(3),
                                      "mp": np.zeros(2), "sp": np.ones(2)}
        try:
            for kind in KINDS:
                for norm in C22.NORMS:
                    build(src, kind, norm, td)
                    assert compare(src, kind, norm, td) == [], (kind, norm)
                    d1, lb1, info1, st1 = load(src, kind, norm, td)
                    assert (st1 is not None) == (kind == "train")
                    if norm != "none":      # stored mapped columns are memory-mapped when large, read-only
                        assert not d1["XN"].flags.writeable or d1["XN"].nbytes < MMAP_MIN
            # a changed source is refused
            d["XN"][0, 0] += 1
            np.savez(src, **d)
            try:
                load(src, "plain", "pq", td)
                raise AssertionError("a changed source loaded")
            except SystemExit as e:
                assert "another" in str(e)
        finally:
            C19.prep, C19.input_stats = prep0, stats0
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    for c in ("build", "check"):
        s = sub.add_parser(c)
        s.add_argument("--kind", required=True, choices=KINDS)
        s.add_argument("--map", required=True, choices=C22.NORMS)
        s.add_argument("--src", required=True)
        s.add_argument("--root", default=str(ROOT))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "build":
        build(a.src, a.kind, a.map, a.root)
        return 0
    if a.cmd == "check":
        diff = compare(a.src, a.kind, a.map, a.root)
        log(f"{Path(a.src).name} {a.kind} {a.map}: {'IDENTICAL' if not diff else 'DIFFERS ' + str(diff)}")
        return 0 if not diff else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
