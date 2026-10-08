"""Step 4e looks (docs/STEP4E_PAPER_POOLS.md section 5): look_step4c's binding with step 4e's chosen pools. After
m3b_compile.prepare builds the carve's frozen pools, each question's pool is replaced by its chosen pool: base and seeds
and the filed expansion (scripts/step4e_host.py replace_pools, the pools file checked against the manifest's sha256).
The replacement stops on a missing question, base and seeds or a frozen pool that is not the filed one, or an expansion
that repeats a node, touches base and seeds or loses a node of the frozen pool. Everything else is look_x_six's: the six
pair's scores, the compiled columns, --full's edges and projections, the chunk layout, the records. The carves are step
1's (look_step1's bindings for s1sel, s1fit and s1eval; look_x_six's own for fit and select).

Output: outputs/step4e/look/<dataset>/<carve>, and beside each shard's record<tag>.json a pools<tag>.json naming the
pools file and manifest it ran on, its question count and its row total.

    python outputs/mp_unified/look_step4e.py --dataset musique --carve s1eval --host --full [--shard i/n]
"""
import hashlib
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import look_step1 as LS1  # noqa: E402  (imports look_x_six, which sets the thread variables before numpy and torch load)

if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))
import step4e_host as S4E  # noqa: E402

LX = LS1.LX
CARVES = ("fit", "select", "s1sel", "s1fit", "s1eval")
_ORIG_OPEN = LX.S6.pair_open


def sha_src(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def install(name, carve, filed, stats):
    """S6.pair_open, then its m3b_compile.prepare wrapped: one population, this carve's, its pools replaced."""

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
            stats.update(S4E.replace_pools(out[0], filed, m3c, construction, cfg_h, m3a, m3b_contract))
            stats["seconds"] = round(time.time() - t, 1)
            S4E.log(f"{name}/{carve}: the chosen pools in place of the frozen ones: {stats}")
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
        raise SystemExit("the step's looks write outputs/step4e/look and read every question of the carve")
    if "--full" not in argv:
        raise SystemExit("the cache reads --full's arrays: run it with --full")
    shard = argv[argv.index("--shard") + 1] if "--shard" in argv else None
    filed, ent, manifest_sha = S4E.load_filed(name, carve)
    want_rows = int(S4E.expected_rows(filed).sum())
    LS1.install(LS1.carves(), name)
    stats = {}
    install(name, carve, filed, stats)
    out = S4E.LOOK / name / carve
    rc = LX.main(argv + ["--out", str(out)])
    if not stats:
        raise SystemExit("m3b_compile.prepare never ran")
    if (stats["questions"], stats["rows"]) != (ent["questions"], want_rows):
        raise SystemExit(f"{name}/{carve}: {stats} is not the file's {ent['questions']} questions and {want_rows} rows")
    sh = LX.L8.parse_shard(shard)
    tag = f"_{sh[0]}of{sh[1]}" if sh else ""
    rec = {"look": "look_step4e", "declared_in": "docs/STEP4E_PAPER_POOLS.md", "dataset": name, "carve": carve,
           "shard": list(sh) if sh else None, "pools_file": ent["file"], "pools_sha256": ent["sha256"],
           "manifest_sha256": manifest_sha, **stats,
           "script_sha256": {"look_step4e": sha_src(__file__), "look_step1": sha_src(LS1.__file__),
                             "look_x_six": sha_src(LX.__file__), "step4e_host": sha_src(S4E.__file__),
                             "step4c_walk_pools": sha_src(S4E.C4.__file__)},
           "utc": S4E.utc()}
    S4E.write_json(out / f"pools{tag}.json", rec)
    return rc


if __name__ == "__main__":
    main()
