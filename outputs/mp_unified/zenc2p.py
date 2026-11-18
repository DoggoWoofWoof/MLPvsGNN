"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md section 12.10): the arm's bases, EM types and PKO pools for the small worlds.

    python outputs/mp_unified/zenc2p.py bases --enc mpnet --graph 2wiki [--torch-threads 8] --threads 2   (plain python)
    python outputs/mp_unified/zenc2w.py run --enc mpnet --tag T -- python outputs/host_ops/pylib_run.py \\
        outputs/mp_unified/zenc2p.py types --enc mpnet --ds squad --host --threads 2
    python outputs/mp_unified/zenc2w.py run --enc mpnet --tag T -- python outputs/host_ops/pylib_run.py \\
        outputs/mp_unified/zenc2p.py pools --enc mpnet --ds squad --carve fit --shard 0 2 [--limit Q] --host --threads 2
    python outputs/mp_unified/zenc2w.py run --enc mpnet --tag T -- python outputs/host_ops/pylib_run.py \\
        outputs/mp_unified/zenc2p.py merge --enc mpnet --ds squad --carve fit --parts 2 --host --threads 2
    python outputs/mp_unified/zenc2p.py selftest

bases   lean_mlp3.fit_basis (the record's function, two BLAS threads, seed 0, 60,000 rows) on the arm's vectors of the
        rows it draws from the graph's nodes, zero-padded to 1536: the arm's own encode of the graph where it exists
        (HotpotQA), else those rows' texts encoded for this (2Wiki, on the CPU). Written as lean_cache writes a basis
        (m, V, w; a record with its sha256) under outputs/enc/e2/bases/<enc>/; zpko.copy_bases takes them from there in
        an arm's world (the hook).
types   zu2m_a_types.types_stage on the arm's vectors (the hook's swaps), into outputs/enc/e2/types/<enc>/.
pools   zpko.pools_pass on one part of a carve, in the arm's world (its outputs/pko), with the hook's swaps in force:
        the arm's vectors, dense lists and K; its knn store; its EM types; B1P's union (zenc1b2's view, section 11.2).
merge   zpko.merge_carve: the parts joined into the world's outputs/pko/pools, its KT rows and its manifest.

zpko runs as a module here, not as the main script: the hook patches modules as they load, and a main script is not
one. Every step checks that the swaps it needs are in force before it writes, and records what it ran on.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parents[1] / "outputs" / "host_ops"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import zenc as E  # noqa: E402  (first: sets the BLAS threads from --threads before numpy loads)

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

ROOT = HERE.parents[1]
DECLARED = "docs/ENC_ENCODER_COMPARISON.md section 12.10"
OUT = ROOT / "outputs" / "enc" / "e2"
SMALL = ("squad", "musique")
GRAPHS = ("2wiki", "hotpotqa")
CARVES = (("squad", "fit"), ("squad", "s1eval"), ("musique", "s1fit"), ("musique", "s1eval"))
PARTS = {("squad", "fit"): 2, ("squad", "s1eval"): 4, ("musique", "s1fit"): 2, ("musique", "s1eval"): 1}   # PKO's filed parts
FIT_NODES, SEED, DIM = 60000, 0, 1536


def log(msg):
    print(f"[zenc2p {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zenc2p: {msg}")


def rel(p):
    try:
        return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def bases_dir(enc):
    return OUT / "bases" / enc


def types_dir(enc):
    return OUT / "types" / enc


def code_shas(*paths):
    return {"code_sha256": {rel(p): E.sha256(p) for p in (Path(__file__),) + tuple(paths)}}


# ── bases ────────────────────────────────────────────────────────────────────

def draw(n_rows, n_fit=FIT_NODES, seed=SEED):
    """fit_basis's rows: the same draw, so the arm's basis is fit on the record's rows."""
    return np.sort(np.random.default_rng(seed).choice(n_rows, size=min(n_fit, n_rows), replace=False))


class Rows:
    """A node reader over the drawn rows only (fit_basis reads nothing else)."""

    def __init__(self, n_rows, rows, vecs):
        self.n_rows, self.rows, self.vecs = int(n_rows), np.asarray(rows, np.int64), vecs

    def read(self, r):
        i = np.searchsorted(self.rows, np.asarray(r, np.int64))
        if (i >= self.rows.size).any() or not np.array_equal(self.rows[np.minimum(i, self.rows.size - 1)], r):
            stop("fit_basis read a row outside the draw")
        return self.vecs[i]


def pad(v, dim_real):
    if v.shape[1] != dim_real or v.dtype != np.float16:
        stop(f"vectors {v.shape} {v.dtype}, want (n, {dim_real}) float16")
    out = np.zeros((v.shape[0], DIM), np.float16)
    out[:, :dim_real] = v
    return out


def arm_rows(enc, g, rows):
    """The arm's own float16 rows at the drawn positions (its encode of the whole graph), shard by shard."""
    root = E.served_root()
    paths = E.vec_shards(root, enc, g, "docs")
    out = np.empty((rows.size, int(E.ENCODERS[enc]["dim"])), np.float16)
    for s in np.unique(rows // E.SHARD):
        a = np.load(paths[int(s)], mmap_mode="r")
        m = rows // E.SHARD == s
        out[m] = a[rows[m] - s * E.SHARD]
    return out, {"source": "the arm's encode of the graph", "shards": len(paths),
                 "manifest_sha256": E.sha256(E.vec_dir(root, enc, g, "docs") / "manifest.json")}


def encoded_rows(enc, g, rows, torch_threads):
    import torch
    torch.set_num_threads(int(torch_threads))
    root = E.served_root()
    t = time.time()
    texts = E.doc_texts(root, g, rows)
    if len(texts) != rows.size:
        stop(f"{g}: {len(texts)} texts for {rows.size} rows")
    m = E.load_model(enc, "cpu", E.pinned(enc))
    if m.default_prompt_name:
        stop(f"{enc}: a default prompt would prefix the documents")
    v, st = E.encode_texts(m, enc, [texts[int(r)] for r in rows], "docs")
    bad = E.damaged(v)
    if bad:
        stop(f"{enc} {g}: {bad} damaged rows")
    return v, {"source": "the drawn rows' texts, encoded for this", "device": "cpu", "torch_threads": int(torch_threads),
               "revision": E.pinned(enc), "encode_seconds": round(time.time() - t, 1), **st}


def lean_mlp3():
    """lean_mlp3 needs numba. Run as a plain script (not under pylib_run, which loads numpy before zenc sets the BLAS
    threads): the host's numba is the unpacked pushed zip, put on the path here, after numpy has loaded."""
    try:
        import numba  # noqa: F401
    except ImportError:
        d = ROOT / "outputs" / "host_ops" / "pylib" / "cp313"
        if not (d / "unpacked_from.sha256").is_file():
            stop(f"no numba and no unpacked pylib at {rel(d)}")
        sys.path.insert(0, str(d))
        import numba  # noqa: F401
    import lean_mlp3 as L3
    return L3


def blas_threads():
    """The BLAS pools numpy loaded with (threadpoolctl, if present): each must be 2."""
    try:
        from threadpoolctl import threadpool_info
    except ImportError:
        return None
    got = {f"{x.get('internal_api')}:{Path(x.get('filepath', '')).name}": x.get("num_threads") for x in threadpool_info()
           if x.get("user_api") == "blas"}
    if any(v != 2 for v in got.values()):
        stop(f"the BLAS pools run {got} threads, not 2: numpy loaded before zenc set them")
    return got


def cmd_bases(a):
    enc, g = a.enc, a.graph
    if enc not in E.NEW or g not in GRAPHS:
        stop(f"bases runs on {E.NEW} and {GRAPHS}")
    if os.environ.get("OPENBLAS_NUM_THREADS") != "2":
        stop("bases are fit at two BLAS threads (lean_cache's rule): pass --threads 2")
    blas = blas_threads()
    L3 = lean_mlp3()
    out = bases_dir(enc)
    npz, js = out / f"basis_{g}.npz", out / f"basis_{g}.json"
    if js.exists():
        stop(f"{rel(js)} exists: a basis is fit once")
    n_rows = int(E.manifest(E.served_root(), g, "docs")["n_rows"])
    rows = draw(n_rows)
    have = E.vec_dir(E.served_root(), enc, g, "docs") / "manifest.json"
    v, src = arm_rows(enc, g, rows) if have.is_file() else encoded_rows(enc, g, rows, a.torch_threads)
    dim_real = int(E.ENCODERS[enc]["dim"])
    nodes = Rows(n_rows, rows, pad(v, dim_real))
    basis = L3.fit_basis(nodes, FIT_NODES)
    if basis["rows"] != rows.size:
        stop(f"fit_basis drew {basis['rows']} rows, the draw has {rows.size}")
    out.mkdir(parents=True, exist_ok=True)
    tmp = npz.with_name(npz.stem + ".tmp.npz")
    np.savez(tmp, m=basis["m"], V=basis["V"], w=basis["w"])
    os.replace(tmp, npz)
    rec = {"graph": g, "fit_nodes": FIT_NODES, "rows": basis["rows"], "explained": basis["explained"],
           "seconds": basis["seconds"], "freeze_RECORD_SHA256": E.RECORD_SHA, "npz": npz.name,
           "npz_sha256": E.sha256(npz), "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "blas_pools": blas,
           "stage": "ENC E2: the arm's basis", "declared_in": DECLARED, "enc": enc, "dim_real": dim_real, "dim": DIM,
           "draw": {"n_rows": n_rows, "seed": SEED, "rows_sha256": E.hashlib.sha256(rows.tobytes()).hexdigest()},
           "vectors": src, "lean_mlp3_sha256": E.sha256(L3.__file__), **code_shas(),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    E.wjson(js, rec)
    log(f"{enc} basis {g}: {basis['rows']} rows, {basis['explained']:.3f} of the variance; {rel(npz)}")
    return 0


# ── in an arm's world ────────────────────────────────────────────────────────

def spec_here(enc):
    sp = os.environ.get("E2HOOK")
    if not sp:
        stop("not in an E2 world: run through zenc2w.py run")
    spec = json.loads(Path(sp).read_text(encoding="utf-8"))
    if spec.get("enc") != enc or not spec.get("swap"):
        stop(f"the world's arm is {spec.get('enc')}, not {enc}, or it swaps nothing")
    return spec


def modules(host):
    if host:
        import zpkoh
        zpkoh.apply_mirror()                             # S6.host_mode + step 4's package root (PKOH's, unchanged)
    import zpko as PK
    import zu2m_a_types as AT
    return PK, AT


def need(ok, what):
    if not ok:
        stop(f"the swap of {what} is not in force")


def check_swaps(spec, PK, AT, opened=()):
    """The hook's patches, in this process: zpko and the types module always; after a step, the modules it opened
    ('canonical': the package loader; 'store': M3B's knn store; 'union': step 4e's union)."""
    need(Path(AT.OUT) == Path(spec["types_dir"]), "the EM types folder")
    need(getattr(PK.u2s_record, "_e2", False), "U2s's EM-types sha")
    if "canonical" in opened:
        can = sys.modules.get("canonical")
        need(can is not None and getattr(can.Embeddings, "_e2_swapped", False), "the package loader")
    if "store" in opened:
        mp = sys.modules.get("mp_retrieval.m3b_pools")
        need(mp is not None and getattr(mp.load_or_build_store, "_e2", False), "the knn store")
    if "union" in opened:
        e4 = sys.modules.get("step4e_paper_pools")
        need(e4 is not None and getattr(e4.union_pools, "_e2", False), "step 4e's union")


def cmd_types(a):
    spec = spec_here(a.enc)
    if a.ds not in SMALL:
        stop(f"types runs on {SMALL}")
    PK, AT = modules(a.host)
    check_swaps(spec, PK, AT)
    if (Path(AT.OUT) / f"{a.ds}.json").exists():
        stop(f"{rel(Path(AT.OUT) / (a.ds + '.json'))} exists: an arm's types are fit once")
    t = time.time()
    AT.types_stage(a.ds, threads=int(a.threads))
    check_swaps(spec, PK, AT, opened=("canonical",))
    arec = json.loads((Path(AT.OUT) / f"{a.ds}.json").read_text(encoding="utf-8"))
    E.wjson(Path(AT.OUT) / f"{a.ds}.e2.json", {
        "stage": "ENC E2: EM types", "declared_in": DECLARED, "enc": a.enc, "dataset": a.ds,
        "types": {"file": rel(Path(AT.OUT) / f"{a.ds}.npz"), "sha256": arec["file_sha256"]},
        "seconds": round(time.time() - t, 1), **code_shas(PK.__file__, AT.__file__),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    log(f"{a.enc} {a.ds}: EM types in {time.time() - t:.0f} s")
    return 0


def cmd_pools(a):
    spec = spec_here(a.enc)
    key = (a.ds, a.carve)
    if key not in PARTS:
        stop(f"pools runs on {CARVES}")
    i, k = int(a.shard[0]), int(a.shard[1])
    if k != PARTS[key] or not 0 <= i < k:
        stop(f"{a.ds}/{a.carve} is cut into {PARTS[key]} parts (PKO's)")
    PK, AT = modules(a.host)
    check_swaps(spec, PK, AT)
    if not (Path(spec["types_dir"]) / f"{a.ds}.json").is_file():
        stop(f"no EM types for {a.enc} on {a.ds}: run zenc2p.py types first")
    R = PK.roots()
    t = time.time()
    out = PK.pools_pass(a.ds, a.carve, threads=int(a.threads), limit=int(a.limit or 0), shard=(i, k), R=R)
    check_swaps(spec, PK, AT, opened=("canonical", "store", "union"))
    rec = {"stage": "ENC E2: pools", "declared_in": DECLARED, "enc": a.enc, "dataset": a.ds, "carve": a.carve,
           "shard": [i, k], "limit": int(a.limit or 0), "result": out, "parts": rel(R.parts),
           "seconds": round(time.time() - t, 1), **code_shas(PK.__file__, AT.__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = Path(R.parts) / f"{PK.stem(a.ds, a.carve, (i, k))}.e2{'.limit' if a.limit else ''}.json"
    if not a.limit:
        E.wjson(p, rec)
    log(f"{a.enc} {a.ds}/{a.carve} {i}of{k}: {json.dumps(out)[:300]} in {time.time() - t:.0f} s")
    return 0


def cmd_merge(a):
    spec = spec_here(a.enc)
    key = (a.ds, a.carve)
    if key not in PARTS or int(a.parts) != PARTS[key]:
        stop(f"{a.ds}/{a.carve} is cut into {PARTS.get(key)} parts (PKO's)")
    PK, AT = modules(a.host)
    check_swaps(spec, PK, AT)
    R = PK.roots()
    t = time.time()
    PK.merge_carve(a.ds, a.carve, int(a.parts), R)
    log(f"{a.enc} {a.ds}/{a.carve}: merged {a.parts} parts in {time.time() - t:.0f} s")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────

def selftest():
    L3 = lean_mlp3()
    print("BLAS pools:", blas_threads())
    rng = np.random.default_rng(1)
    n = 5000
    full = rng.standard_normal((n, 768)).astype(np.float32)
    full = (full / np.linalg.norm(full, axis=1, keepdims=True)).astype(np.float16)
    rows = draw(n, 600)
    assert np.array_equal(rows, np.sort(np.random.default_rng(0).choice(n, size=600, replace=False)))

    class AllRows:                                       # the record's reader shape: every row, padded
        n_rows = n

        def read(self, r):
            return pad(full[np.asarray(r)], 768)

    b_all = L3.fit_basis(AllRows(), 600)
    b_rows = L3.fit_basis(Rows(n, rows, pad(full[rows], 768)), 600)
    for k in ("m", "V", "w"):
        assert np.array_equal(b_all[k], b_rows[k]), k        # the draw-only reader gives the record reader's basis
    assert b_all["rows"] == 600 and not b_all["V"][768:].any() and not b_all["m"][768:].any()
    try:
        Rows(n, rows, pad(full[rows], 768)).read(np.array([int(np.setdiff1d(np.arange(n), rows)[0])]))
        raise AssertionError("a row outside the draw must be refused")
    except SystemExit:
        pass
    try:
        pad(full[:3].astype(np.float32), 768)
        raise AssertionError("pad must refuse non-float16 rows")
    except SystemExit:
        pass
    tmp = Path(tempfile.mkdtemp(prefix="zenc2p_t_"))
    try:
        spec = {"enc": "mpnet", "swap": {"squad": {}}, "types_dir": str(tmp / "types")}
        (tmp / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        os.environ["E2HOOK"] = str(tmp / "spec.json")
        assert spec_here("mpnet")["enc"] == "mpnet"
        try:
            spec_here("qwen3e06")
            raise AssertionError("the world's arm must match")
        except SystemExit:
            pass
        from types import SimpleNamespace
        PKs = SimpleNamespace(u2s_record=lambda *x: None)
        try:
            check_swaps(spec, PKs, SimpleNamespace(OUT=tmp / "types"))
            raise AssertionError("an unpatched u2s_record must be refused")
        except SystemExit:
            pass
    finally:
        os.environ.pop("E2HOOK", None)
        shutil.rmtree(tmp, ignore_errors=True)
    print("zenc2p self-test: OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zenc2p.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bases")
    p.add_argument("--enc", required=True)
    p.add_argument("--graph", required=True)
    p.add_argument("--torch-threads", type=int, default=8)
    p.add_argument("--threads", type=int, default=2)
    for name in ("types", "pools", "merge"):
        p = sub.add_parser(name)
        p.add_argument("--enc", required=True)
        p.add_argument("--ds", required=True)
        p.add_argument("--host", action="store_true")
        p.add_argument("--threads", type=int, default=2)
        if name in ("pools", "merge"):
            p.add_argument("--carve", required=True)
        if name == "pools":
            p.add_argument("--shard", nargs=2, type=int, required=True)
            p.add_argument("--limit", type=int, default=0)
        if name == "merge":
            p.add_argument("--parts", type=int, required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--threads", type=int, default=2)
    a = ap.parse_args(argv)
    return {"bases": cmd_bases, "types": cmd_types, "pools": cmd_pools, "merge": cmd_merge}.get(
        a.cmd, lambda _a: selftest())(a)


if __name__ == "__main__":
    raise SystemExit(main())
