"""U1c looks (docs/U1C_RETRAIN_ON_U.md): look_step4e's binding on U1d's universal graph with step 4h's chosen pools.

Two things differ from look_step4e:
    the graph   right after S6.pair_open, the dataset context's `structural` store becomes U1d's structural_U
                (outputs/u1d/<D>/graph_structural_u.npz, its sha256 from build.json, and the one step 4h's coverage and
                U1c's pools ran on). Every later reader takes it from there: m3b_compile.prepare's frozen construction,
                the compile's STRUCT columns and edges, the six pair's packing. squad has no graph and keeps its stores.
    the pools   scripts/u1c_host.py's files (outputs/u1c/pools/<D>), through step4e_host.replace_pools unchanged: base
                and seeds, the frozen pool (now GU's, by its sha256) and the filed expansion.
Everything else is look_x_six's. numba-free (the host's looks run without pylib).

Output: outputs/u1c/look/<dataset>/<carve>, and beside each shard's record<tag>.json a pools<tag>.json naming the pools
file, the manifest, the graph and the counts.

    python outputs/mp_unified/look_u1c.py --dataset musique --carve s1eval --host --full [--shard i/n]
"""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_step1 as LS1  # noqa: E402  (imports look_x_six, which sets the thread variables before numpy and torch load)

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4e_host as S4E  # noqa: E402

LX = LS1.LX
C4 = S4E.C4
CARVES = ("fit", "select", "s1sel", "s1fit", "s1eval")
OUT = ROOT / "outputs" / "u1c"
POOLS = OUT / "pools"
LOOK = OUT / "look"
U1D = ROOT / "outputs" / "u1d"
NOGRAPH = ("squad",)
_ORIG_OPEN = LX.S6.pair_open


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def u_store(name, n, fam_store):
    """U1d's structural_U as a FamilyStore (scripts/step4h_u_pools.py's u_store, numba-free)."""
    b = json.loads((U1D / name / "build.json").read_text(encoding="utf-8"))
    p = U1D / name / "graph_structural_u.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != b["npz_sha256"] or int(b["n_nodes"]) != n:
        raise SystemExit(f"{p}: not U1d's build ({b.get('variant')})")
    z = np.load(p)
    g = SimpleNamespace(n_nodes=n, src=z["src"], dst=z["dst"], rel=z["rel"], weight=None)
    return fam_store.from_graph(g, "structural"), {"variant": b["variant"], "npz_sha256": b["npz_sha256"],
                                                    "edges": int(z["src"].size)}


def install(name, carve, filed, manifest, stats):
    """S6.pair_open with the context's structural store replaced, then its m3b_compile.prepare wrapped: one population,
    this carve's, its pools replaced."""

    def pair_open(*args, **kwargs):
        op = _ORIG_OPEN(*args, **kwargs)
        ctx, ds = op.contexts[name], op.handles[name]
        if name not in NOGRAPH:
            if "structural" not in ctx.stores:
                raise SystemExit(f"{name}: the context has no structural family ({sorted(ctx.stores)})")
            from mp_retrieval.m3b_pools import FamilyStore   # step4h_u_pools.u_store's class
            st, info = u_store(name, int(ds.n_nodes), FamilyStore)
            if manifest["graph"].get("npz_sha256") != info["npz_sha256"]:
                raise SystemExit(f"{name}: U1c's pools ran on another structural_U than U1d's build")
            ctx.stores["structural"] = st
            stats["graph"] = info
        else:
            stats["graph"] = {"note": "no graph: the stores unchanged"}
        stores_obj = ctx.stores
        m3c = op.m3b_compile
        orig = m3c.prepare

        def prepare(ds_, pops, construction, cfg_h, stores, m3a, m3b_contract):
            if "rows" in stats:
                raise SystemExit("m3b_compile.prepare ran twice in one look")
            if stores is not stores_obj:
                raise SystemExit("prepare: not the context's (patched) stores")
            out = orig(ds_, pops, construction, cfg_h, stores, m3a, m3b_contract)
            if len(out) != 1 or out[0].pop.dataset != name or out[0].pop.kind != carve:
                raise SystemExit(f"prepare built {[(p.pop.dataset, p.pop.kind) for p in out]}, not {name}/{carve}")
            t = time.time()
            stats.update(S4E.replace_pools(out[0], filed, m3c, construction, cfg_h, m3a, m3b_contract))
            stats["seconds"] = round(time.time() - t, 1)
            S4E.log(f"{name}/{carve}: U1c's pools on {stats['graph']}: {stats}")
            return out

        m3c.prepare = prepare
        return op

    LX.S6.pair_open = pair_open


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    name = argv[argv.index("--dataset") + 1]
    carve = argv[argv.index("--carve") + 1]
    if carve not in CARVES:
        raise SystemExit(f"--carve {carve}: one of {CARVES}")
    if "--out" in argv or "--limit" in argv:
        raise SystemExit("U1c's looks write outputs/u1c/look and read every question of the carve")
    if "--full" not in argv:
        raise SystemExit("the cache reads --full's arrays: run it with --full")
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    filed, ent, manifest_sha = C4.load_filed(name, carve, POOLS / name)
    manifest = json.loads((POOLS / name / "pools.json").read_text(encoding="utf-8"))
    want_rows = int(S4E.expected_rows(filed).sum())
    LS1.install(LS1.carves(), name)
    stats = {}
    install(name, carve, filed, manifest, stats)
    out = LOOK / name / carve
    rc = LX.main(argv + ["--out", str(out)])
    if "rows" not in stats:
        raise SystemExit("m3b_compile.prepare never ran")
    if (stats["questions"], stats["rows"]) != (ent["questions"], want_rows):
        raise SystemExit(f"{name}/{carve}: {stats} is not the file's {ent['questions']} questions and {want_rows} rows")
    sh = LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    rec = {"look": "look_u1c", "declared_in": "docs/U1C_RETRAIN_ON_U.md", "dataset": name, "carve": carve,
           "shard": list(sh) if sh else None, "pools_file": ent["file"], "pools_sha256": ent["sha256"],
           "manifest_sha256": manifest_sha, "config": manifest["config"], **stats,
           "script_sha256": {"look_u1c": sha_src(__file__), "look_step1": sha_src(LS1.__file__),
                             "look_x_six": sha_src(LX.__file__), "step4e_host": sha_src(S4E.__file__),
                             "step4c_walk_pools": sha_src(C4.__file__)},
           "utc": S4E.utc()}
    S4E.write_json(out / f"pools{tag}.json", rec)
    return rc


if __name__ == "__main__":
    main()
