"""ENC E2 (docs/ENC_ENCODER_COMPARISON.md sections 12.9 and 12.10): the start-up hook of a process in an E2 world.

Python imports `sitecustomize` at start-up from the first folder on sys.path that holds one. zenc2w.py puts this folder
first on PYTHONPATH for the commands it runs in a world, and names a spec file in E2HOOK. The children of those commands
inherit both. With E2HOOK unset this file does nothing.

From the spec (written by zenc2w.py run):
- audit: every file the process opens is logged once per path to <audit>/<pid>.log ('R' read, 'W' write), with the
  process's argv on the first line.
- read guard: opening a path under a "forbid" prefix raises PermissionError. These are the record's files that the arm
  replaces: the dense vectors, dense lists and knn graph of the arm's datasets, the CSR knn stores, the bases, the EM
  types and step 4d's pools built from the frozen vectors.
- write guard: a write whose real path (junctions resolved) is outside the world's tree, the audit folder, the spec's
  other write_ok folders or the temp folder (unless the spec's temp_ok is false) raises PermissionError.
- swaps (arm worlds only: the spec's swap is not empty). A module named below is patched right after its code runs,
  whichever way it was loaded (an import or a spec_from_file_location load: both run SourceFileLoader.exec_module):
    canonical            Embeddings.__init__: the dense docs/queries folder of an arm dataset is the arm's padded
                         folder; Dataset.cache('dense') of an arm dataset: the arm's first-stage lists; Graph(ds, 'knn')
                         of an arm dataset: the arm's K (the package's dtypes, its manifest entry with the arm's count).
    m3b_pools            load_or_build_store(ds, 'knn', dir) of an arm dataset: built from the (swapped) knn graph,
                         never loaded from or saved to a CSR cache.
    zu2m_a_types         OUT: the arm's EM types folder (spec types_dir).
    zpko                 u2s_record(d) of an arm dataset: U2s's record with em_types_sha256 = the arm's types file's sha
                         (zenc1b2's swap); copy_bases(R): from the arm's bases folder (spec bases).
    step4e_paper_pools   union_pools of an arm dataset: B1P's form (B = base united with the seeds, I = the prepared
                         pools; step 4d's files hold the frozen vectors' pools), as zenc1b2 swapped it.
  A module run as the main script (runpy) is not patched: an arm world runs zpko's steps through zenc2p.py.
"""
import json
import os
import sys


def _norm(p):
    return os.path.normcase(os.path.abspath(p)).replace("\\", "/")


_SPEC_PATH = os.environ.get("E2HOOK")

if _SPEC_PATH:
    import importlib.machinery
    import tempfile

    with open(_SPEC_PATH, encoding="utf-8") as _f:
        SPEC = json.load(_f)
    _FORBID = tuple(_norm(p) for p in SPEC.get("forbid", ()))
    _WRITE_OK = tuple(_norm(os.path.realpath(p)) for p in SPEC.get("write_ok", ())) + (
        (_norm(os.path.realpath(tempfile.gettempdir())),) if SPEC.get("temp_ok", True) else ())
    _SKIP = tuple(_norm(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix} if p)
    _SERVED = _norm(SPEC["served"]) if SPEC.get("served") else None
    _SWAP = SPEC.get("swap", {})
    _BUSY = [False]
    _SEEN = set()
    _REAL = {}
    _FD = [None]
    if SPEC.get("audit"):
        os.makedirs(SPEC["audit"], exist_ok=True)
        _FD[0] = os.open(os.path.join(SPEC["audit"], f"{os.getpid()}.log"),
                         os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0))
        os.write(_FD[0], ("A\t" + json.dumps(sys.argv) + "\n").encode("utf-8"))

    def _real(n):
        r = _REAL.get(n)
        if r is None:
            try:
                r = _norm(os.path.realpath(n))
            except OSError:
                r = n
            _REAL[n] = r
        return r

    def _under(p, prefixes):
        return any(p == q or p.startswith(q + "/") for q in prefixes)

    def _note(tag, n):
        if _FD[0] is None or (tag, n) in _SEEN:
            return
        _SEEN.add((tag, n))
        os.write(_FD[0], f"{tag}\t{n}\n".encode("utf-8"))

    def _hook(event, args):
        if event != "open" or _BUSY[0]:
            return
        path, mode, flags = (tuple(args) + (None, None, None))[:3]
        if isinstance(path, bytes):
            path = os.fsdecode(path)
        if not isinstance(path, str):
            return
        _BUSY[0] = True
        try:
            n = _norm(path)
            if _under(n, _SKIP) or "/__pycache__/" in n:
                return
            if isinstance(mode, str):
                write = any(c in mode for c in "wax+")
            else:
                write = bool(int(flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND))
            r = _real(n)
            if _under(n, _FORBID) or _under(r, _FORBID):
                _note("X", r)
                raise PermissionError(f"E2 guard: {path} is a record file the arm's world must not read")
            if write and not _under(r, _WRITE_OK):
                _note("Y", r)
                raise PermissionError(f"E2 guard: {path} resolves to {r}, outside the world: no write")
            _note("W" if write else "R", r)
        finally:
            _BUSY[0] = False

    sys.addaudithook(_hook)

    def _patch_canonical(mod):
        here = _norm(os.path.dirname(mod.__file__))
        if _SERVED is not None and here != _SERVED:
            raise SystemExit(f"E2 hook: the loader came from {here}, not the served root {_SERVED}")
        E, D = mod.Embeddings, mod.Dataset
        if getattr(E, "_e2_swapped", False):
            return
        orig_init, orig_cache = E.__init__, D.cache
        want = {}
        for ds, s in _SWAP.items():
            for kind in ("docs", "queries"):
                want[_norm(os.path.join(here, ds, "embeddings", "dense", kind))] = s[kind]

        def init(self, d, model):
            alt = want.get(_norm(d)) if model == "dense" else None
            if alt is not None:
                _note("S", _norm(alt))
            orig_init(self, alt if alt is not None else d, model)

        def cache(self, model):
            s = _SWAP.get(self.name)
            if model != "dense" or s is None:
                return orig_cache(self, model)
            import numpy as np
            _note("S", _norm(s["cache"]))
            with np.load(s["cache"]) as z:
                ids, scores = np.asarray(z["ids"], np.int32), np.asarray(z["scores"], np.float16)
            if ids.shape != scores.shape or ids.shape[1] != 1000:
                raise SystemExit(f"E2 hook: {s['cache']}: lists {ids.shape} / {scores.shape}")
            return ids, scores

        G = mod.Graph
        orig_graph = G.__init__

        def graph_init(self, ds, family):
            s = _SWAP.get(getattr(ds, "name", None))
            if family != "knn" or s is None or not s.get("knn"):
                return orig_graph(self, ds, family)
            import numpy as np
            self.dataset, self.family = ds.name, family
            self.manifest = ds.graph_manifest
            fam = self.manifest["families"].get(family)
            if not fam or not fam.get("present"):
                raise ValueError("%s has no %r graph" % (ds.name, family))
            n = int(self.manifest["n_nodes"])
            _note("S", _norm(s["knn"]))
            with np.load(s["knn"]) as z:
                src, dst = np.asarray(z["src"], np.int32), np.asarray(z["dst"], np.int32)
                w = np.asarray(z["weight"], np.float32) if "weight" in z.files else None
            if src.shape != dst.shape or (w is not None and w.shape != src.shape) or (
                    src.size and (int(min(src.min(), dst.min())) < 0 or int(max(src.max(), dst.max())) >= n
                                  or bool((src == dst).any()))):
                raise SystemExit(f"E2 hook: {s['knn']} is not a knn family of {n} nodes without self-loops")
            self.src, self.dst, self.weight, self.rel = src, dst, w, None
            self.rel_vocab = fam.get("relation_vocabulary")
            self.n_nodes = n
            self.directed = bool(fam["directed"])
            self.provenance = dict(fam, n_edges=int(src.size), npz_sha256=None, file=s["knn"], e2_arm=SPEC.get("enc"))
            self.history = (self.manifest.get("history") or {}).get(family)
            self.source = "graph"
            self._csr = None

        E.__init__, D.cache, G.__init__ = init, cache, graph_init
        E._e2_swapped = True

    def _patch_store(mod):
        orig = mod.load_or_build_store
        if getattr(orig, "_e2", False):
            return

        def load_or_build_store(ds, family, cache_dir):
            if family != "knn" or getattr(ds, "name", None) not in _SWAP:
                return orig(ds, family, cache_dir)
            g = ds.graph("knn")
            if (getattr(g, "provenance", None) or {}).get("e2_arm") != SPEC.get("enc"):
                raise SystemExit(f"E2 hook: {ds.name}'s knn graph is not the arm's (the loader swap is not in force)")
            _note("S", f"store:{ds.name}:knn")
            return mod.FamilyStore.from_graph(g, "knn")          # never loaded from or saved to a CSR cache

        load_or_build_store._e2 = True
        mod.load_or_build_store = load_or_build_store

    def _patch_types(mod):
        from pathlib import Path
        mod.OUT = Path(SPEC["types_dir"])
        _note("S", "types:" + _norm(SPEC["types_dir"]))

    def _patch_zpko(mod):
        from pathlib import Path
        orig_rec, orig_bases = mod.u2s_record, mod.copy_bases
        if getattr(orig_rec, "_e2", False):
            return

        def u2s_record(name, v=None):
            rec, npz = orig_rec(name, v)
            if name not in _SWAP:
                return rec, npz
            tj = Path(SPEC["types_dir"]) / f"{name}.json"
            arec = json.loads(tj.read_text(encoding="utf-8"))
            _note("S", f"u2s_record:{name}:em_types")
            return dict(rec, em_types_sha256=arec["file_sha256"]), npz

        def copy_bases(R=None, src=None):
            _note("S", "bases:" + _norm(SPEC["bases"]))
            return orig_bases(R, src=Path(SPEC["bases"]))

        u2s_record._e2 = True
        mod.u2s_record, mod.copy_bases = u2s_record, copy_bases

    def _patch_union(mod):
        orig = mod.union_pools
        if getattr(orig, "_e2", False):
            return

        def union_pools(name, carve, pop, prep, bases):
            if name not in _SWAP:
                return orig(name, carve, pop, prep, bases)
            import numpy as np
            _note("S", f"union_pools:{name}:{carve}")
            bs = [np.union1d(np.asarray(b, np.int64), np.asarray(s, np.int64)) for b, s in zip(bases, prep.seeds)]
            return None, None, bs, [np.asarray(p, np.int64) for p in prep.pools], None

        union_pools._e2 = True
        mod.union_pools = union_pools

    _PATCH = {"canonical": _patch_canonical, "m3b_pools": _patch_store, "zu2m_a_types": _patch_types,
              "zpko": _patch_zpko, "step4e_paper_pools": _patch_union}

    if _SWAP:
        _orig_exec = importlib.machinery.SourceFileLoader.exec_module

        def _exec_module(self, module):
            _orig_exec(self, module)
            f = _PATCH.get(module.__name__.rsplit(".", 1)[-1])
            if f is not None:
                f(module)

        importlib.machinery.SourceFileLoader.exec_module = _exec_module
