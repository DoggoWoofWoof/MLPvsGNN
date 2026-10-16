"""P1g-b looks (docs/P1GB_POOLS_REFIT.md): P1c's binding (outputs/mp_unified/look_p1c.py, imported unchanged) with
P1g-b's pools, P1g2's chosen rule (scripts/p1gb_pools.py). After m3b_compile.prepare builds the carve's frozen pools,
each question's pool is replaced by its P1g-b pool: base and seeds, then the filed expansion (fill_f, then the extra
set's nodes outside it) with p1gb_pools.replace_pools, the pools file checked against its dataset's manifest
(outputs/p1gb/pools/<D>/pools.json). The replacement stops on a missing question, a base and seed set or frozen pool that
is not the filed one, or an expansion that is not the filed length, repeats a node or touches base and seeds.
Everything else is look_x_six's: the six pair's scores, the compiled columns, --full's edges and projections, the chunk
layout, the records. The carves are step 1's (look_step1's bindings for s1fit and s1eval; look_x_six's own for fit).

Output: outputs/p1gb/look/<dataset>/<carve>, and beside each shard's record<tag>.json a pools<tag>.json naming the pools
file and manifest it ran on and the counts of changed and short pools.

    python outputs/mp_unified/look_p1gb.py --dataset musique --carve s1eval --host --full [--shard i/n]
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_p1c as LP  # noqa: E402  (imports look_step1 and look_x_six, which set the thread variables first)

LS1, LX, S4C = LP.LS1, LP.LX, LP.S4C
import p1gb_pools as PGB  # noqa: E402  (scripts/ is on the path through look_p1c; numba-free at import)

DECLARED = PGB.DECLARED
LOOK = PGB.LOOK
CARVES = LP.CARVES                            # the screen's carves (scripts/p1c_pools.py SCREEN)
_ORIG_OPEN = LX.S6.pair_open


def install(name, carve, filed, stats):
    """S6.pair_open, then its m3b_compile.prepare wrapped: one population, this carve's, its pools replaced by P1g-b's."""

    def pair_open(*args, **kwargs):
        op = _ORIG_OPEN(*args, **kwargs)
        m3c = op.m3b_compile
        orig = m3c.prepare

        def prepare(ds, pops, construction, cfg_h, stores, m3a, m3b_contract):
            if stats:
                raise SystemExit("m3b_compile.prepare ran twice in one look")
            out = orig(ds, pops, construction, cfg_h, stores, m3a, m3b_contract)
            if len(out) != 1 or out[0].pop.dataset != name or out[0].pop.kind != carve:
                raise SystemExit(f"prepare built {[(p.pop.dataset, p.pop.kind) for p in out]}, not {name}/{carve}")
            t = time.time()
            stats.update(PGB.replace_pools(out[0], filed, m3c, construction, cfg_h, m3a, m3b_contract))
            stats["seconds"] = round(time.time() - t, 1)
            S4C.log(f"{name}/{carve}: P1g-b's pools in place of the frozen ones: {stats}")
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
        raise SystemExit("P1g-b's looks write outputs/p1gb/look and read every question of the carve")
    if "--full" not in argv:
        raise SystemExit("the cache reads --full's arrays: run it with --full")
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    filed, ent, manifest_sha = PGB.load_filed(name, carve)
    LS1.install(LS1.carves(), name)
    stats = {}
    install(name, carve, filed, stats)
    out = LOOK / name / carve
    rc = LX.main(argv + ["--out", str(out)])
    if not stats:
        raise SystemExit("m3b_compile.prepare never ran")
    if (stats["questions"], stats["changed"], stats["short"]) != (ent["questions"], ent["changed"], ent["short"]):
        raise SystemExit(f"{name}/{carve}: {stats} is not the manifest's {ent['questions']} questions, "
                         f"{ent['changed']} changed, {ent['short']} short")
    sh = LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    rec = {"look": "look_p1gb", "declared_in": DECLARED, "dataset": name, "carve": carve,
           "shard": list(sh) if sh else None, "pools_file": ent["file"], "pools_sha256": ent["sha256"],
           "manifest_sha256": manifest_sha, "construction": ent.get("construction"), **stats,
           "script_sha256": {"look_p1gb": LP.sha_src(__file__), "look_p1c": LP.sha_src(LP.__file__),
                             "look_step1": LP.sha_src(LS1.__file__), "look_x_six": LP.sha_src(LX.__file__),
                             "p1gb_pools": LP.sha_src(PGB.__file__), "step4c_walk_pools": LP.sha_src(S4C.__file__)},
           "utc": S4C.utc()}
    S4C.write_json(out / f"pools{tag}.json", rec)
    return rc


if __name__ == "__main__":
    main()
