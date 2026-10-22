"""U2d looks (docs/U2D_REFIT_ON_U2E.md, section 3): look_x_six's pass with U2d's pools of one build
(outputs/u2d/<build>/pools/<D>, outputs/mp_unified/zu2d_pools.py) in place of the frozen ones.

Two things differ from look_p1gb:
    the graph   new build: right after S6.pair_open, the dataset context's `structural` store becomes W2_S
                (zu2c.graph_csr and csr_store, unchanged), as look_u1c swaps U1d's graph. Every later reader takes it
                from there: m3b_compile.prepare's frozen pool, the compile's STRUCT columns and edges, the six pair's
                packing. The graph's description must be the one the pools manifest names (the one U2e's pick ran on).
                squad has no structural family and keeps its stores; metaqa and webqsp keep their KB. base build:
                today's stores, unchanged.
    the pools   each question's pool becomes B plus the filed expansion, through p1gb_pools.replace_pools, unchanged:
                base and seeds, the frozen pool on the build's graph (by its sha256) and the expansion's length.
The carves are step 1's (look_step1's bindings for s1fit and s1eval; look_x_six's own for fit). Everything else is
look_x_six's.

Placement: --host is look_x_six's (the host mirror). --modal reads the package mirror at /root/mirror (zu2d_modal.py
builds it from U2b's and U2c's volumes): V2.load_configs is wrapped in memory as host_mode wraps it, open_package's freeze
check still runs, and the records name the placement "modal".

Output: outputs/u2d/<build>/look/<dataset>/<carve>, and beside each shard's record<tag>.json a pools<tag>.json naming
the pools file, the manifest, the graph and the counts. --dev Q runs look_x_six's --limit Q into a temporary directory,
every check, and removes it: nothing is written.

    python outputs/mp_unified/look_u2d.py --build base|new --dataset D --carve C --full [--shard i/n] [--host|--modal]
    python outputs/mp_unified/look_u2d.py --build base|new --dataset D --carve C --full --modal --dev 8
"""
import hashlib
import os
import platform
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_step1 as LS1  # noqa: E402  (imports look_x_six, which sets the thread variables before numpy and torch load)

LX = LS1.LX
import numpy as np  # noqa: E402

import zu2d_pools as ZP  # noqa: E402  (U2d's roots, pools files and constructions; numba-free at import)

PGB, U2C, C4S = ZP.PGB, ZP.U2C, ZP.C4S
DECLARED = ZP.DECLARED
CARVES = ("fit", "s1fit", "s1eval")           # P1c2's eleven carves (zu2d_pools: PC2.carves_for)
MODAL_MIRROR = Path("/root/mirror")
_ORIG_OPEN = LX.S6.pair_open


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def graph_of(build, name, n):
    """(store or None, description): W2_S's store on the new build where the dataset has a passage graph, else None;
    the description is the one zu2d_pools records for the build (zu2d_pools.open_carve)."""
    if build == "base":
        return None, {"graph": "today's (the cached stores)"}
    g = U2C.graph_csr(name, "chosen", n)
    if g is None:
        return None, {"graph": "cached (the KB)"}
    ip, col, desc = g
    return U2C.csr_store(n, ip, col), desc


def modal_mode(log=print):
    """host_mode's wrap with the container's mirror: substrate.package_root becomes /root/mirror in memory."""
    root = MODAL_MIRROR
    served = root / "data" / "final_canonical"
    if not (served / "CANONICAL_FREEZE.json").exists():
        raise SystemExit(f"--modal: {served} holds no CANONICAL_FREEZE.json")
    original = LX.V2.load_configs

    def load_configs_on_the_mirror(*args, **kwargs):
        cfg, cfg_m3b, cfg_h = original(*args, **kwargs)
        cfg_m3b["substrate"]["package_root"] = str(root)    # in memory only
        return cfg, cfg_m3b, cfg_h

    LX.V2.load_configs = load_configs_on_the_mirror
    placement = {"where": "modal", "node": platform.node(), "mirror_root": str(root),
                 "freeze_sha256": hashlib.sha256((served / "CANONICAL_FREEZE.json").read_bytes()).hexdigest()}
    LX.S6.PLACEMENT.clear()
    LX.S6.PLACEMENT.update(placement)
    log(f"modal: the mirror at {root} in place of the package, in memory")
    return LX.S6.PLACEMENT


def install(build, name, carve, filed, ent, stats):
    """S6.pair_open with the context's structural store replaced on the new build, then its m3b_compile.prepare
    wrapped: one population, this carve's, on the context's stores, its pools replaced by U2d's."""

    def pair_open(*args, **kwargs):
        op = _ORIG_OPEN(*args, **kwargs)
        ctx, ds = op.contexts[name], op.handles[name]
        st, desc = graph_of(build, name, int(ds.n_nodes))
        if desc != ent["graph_detail"]:
            raise SystemExit(f"{name}: the graph {desc} is not the one the pools manifest names")
        swapped = False
        if st is not None and "structural" in ctx.stores:
            ctx.stores["structural"] = st
            swapped = True
        elif st is not None and name != "squad":
            raise SystemExit(f"{name}: the context has no structural family ({sorted(ctx.stores)})")
        del st
        stats["graph"] = {"detail": desc, "swapped": swapped}
        stores_obj = ctx.stores
        m3c = op.m3b_compile
        orig = m3c.prepare

        def prepare(ds_, pops, construction, cfg_h, stores, m3a, m3b_contract):
            if "questions" in stats:
                raise SystemExit("m3b_compile.prepare ran twice in one look")
            if stores is not stores_obj:
                raise SystemExit("prepare: not the context's stores")
            out = orig(ds_, pops, construction, cfg_h, stores, m3a, m3b_contract)
            if len(out) != 1 or out[0].pop.dataset != name or out[0].pop.kind != carve:
                raise SystemExit(f"prepare built {[(p.pop.dataset, p.pop.kind) for p in out]}, not {name}/{carve}")
            t = time.time()
            stats.update(PGB.replace_pools(out[0], filed, m3c, construction, cfg_h, m3a, m3b_contract))
            stats["seconds"] = round(time.time() - t, 1)
            big = max((int(np.asarray(pl).size) for pl in out[0].pools), default=0)
            if big > np.iinfo(np.int16).max:
                raise SystemExit(f"{name}/{carve}: a pool of {big} nodes; the look's edge arrays are int16")
            stats["largest_pool"] = big
            stats["ids"] = [str(q) for q in out[0].pop.ids]
            C4S.log(f"{name}/{carve}: U2d's {build} pools on {stats['graph']}: "
                    f"{ {k: stats[k] for k in ('questions', 'changed', 'short', 'seconds')} }")
            return out

        m3c.prepare = prepare
        return op

    LX.S6.pair_open = pair_open


def expected_counts(filed, ids):
    """The filed changed and short counts over the given questions (a --dev look's sample)."""
    at = {str(q): k for k, q in enumerate(filed["ids"].tolist())}
    k = np.asarray([at[q] for q in ids], dtype=np.int64)
    return (int(k.size), int(filed["changed"][k].astype(bool).sum()),
            int((filed["fill_k"][k] < filed["fill_want"][k]).sum()))


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)

    def take(flag, cast=str):
        if flag not in argv:
            return None
        i = argv.index(flag)
        v = cast(argv[i + 1])
        del argv[i:i + 2]
        return v

    build = take("--build")
    dev = take("--dev", int)
    on_modal = "--modal" in argv
    if on_modal:
        argv.remove("--modal")
        if "--host" in argv:
            raise SystemExit("--modal and --host: one placement")
    if build not in ZP.BUILDS:
        raise SystemExit(f"--build: one of {ZP.BUILDS}")
    name = argv[argv.index("--dataset") + 1]
    carve = argv[argv.index("--carve") + 1]
    if carve not in CARVES:
        raise SystemExit(f"--carve {carve}: one of {CARVES}")
    if "--out" in argv or "--limit" in argv:
        raise SystemExit("U2d's looks write outputs/u2d/<build>/look; --dev Q is the development run")
    if "--full" not in argv:
        raise SystemExit("the cache reads --full's arrays: run it with --full")
    if dev is not None and (dev < 1 or "--shard" in argv):
        raise SystemExit("--dev Q: Q >= 1, on the whole carve (no --shard)")
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    filed, ent, manifest_sha = ZP.load_filed(build, name, carve)
    LS1.install(LS1.carves(), name)
    stats = {}
    install(build, name, carve, filed, ent, stats)
    if on_modal:
        LX.S6.host_mode = lambda decl, log=print: modal_mode(log)
        argv.append("--host")
    R = ZP.roots(build)
    tmp = None
    if dev is not None:
        tmp = Path(tempfile.mkdtemp(prefix=f"u2d_look_{build}_{name}_{carve}_"))
        out = tmp / "look"
        argv += ["--limit", str(dev)]
    else:
        out = R["look"] / name / carve
    try:
        rc = LX.main(argv + ["--out", str(out)])
        if "questions" not in stats:
            raise SystemExit("m3b_compile.prepare never ran")
        if dev is not None:
            want = expected_counts(filed, stats["ids"])
            got = (stats["questions"], stats["changed"], stats["short"])
            if got != want:
                raise SystemExit(f"{name}/{carve} (--dev {dev}): {got} questions, changed, short; the file's {want}")
            chunks = sorted(p.name for p in (out / "chunks").glob("c*.npz"))
            C4S.log(f"u2d look {build} {name}/{carve} development run: {stats['questions']} questions, "
                    f"{len(chunks)} chunks, every check held; nothing kept")
            return rc
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    if (stats["questions"], stats["changed"], stats["short"]) != (ent["questions"], ent["changed"], ent["short"]):
        raise SystemExit(f"{name}/{carve}: {stats['questions']} questions, {stats['changed']} changed, "
                         f"{stats['short']} short, not the manifest's {ent['questions']}, {ent['changed']}, "
                         f"{ent['short']}")
    sh = LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    rec = {"look": "look_u2d", "declared_in": DECLARED, "build": build, "dataset": name, "carve": carve,
           "shard": list(sh) if sh else None, "pools_file": ent["file"], "pools_sha256": ent["sha256"],
           "manifest_sha256": manifest_sha, "construction": ent.get("construction"),
           **{k: v for k, v in stats.items() if k != "ids"},
           "placement": dict(LX.S6.PLACEMENT) if on_modal else None,
           "script_sha256": {"look_u2d": sha_src(__file__), "look_step1": sha_src(LS1.__file__),
                             "look_x_six": sha_src(LX.__file__), "p1gb_pools": sha_src(PGB.__file__),
                             "zu2d_pools": sha_src(ZP.__file__), "zu2c": sha_src(U2C.__file__)},
           "threads": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
           "utc": C4S.utc()}
    C4S.write_json(out / f"pools{tag}.json", rec)
    return rc


if __name__ == "__main__":
    sys.exit(main())
