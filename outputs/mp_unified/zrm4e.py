"""Step 4e's screen (docs/STEP4E_PAPER_POOLS.md section 5): zrm, unchanged, refit on the chosen pools. Every command is
zrm.py's or rmatch.py's with step 4e's roots in place of today's:
    caches  outputs/step4e/cache  (lean_cache on outputs/step4e/look; --cache-root, required on train and read)
    chains  outputs/step4e/chains (rmatch's chain build on those looks and caches; rmatch.ChainCarveBase reads them)
The model, its arm name (zrm), settings, seeds and batches are zrm's. Two fits, L-musique (scr-zrm-e) and L-hotpotqa
(scr-zrm-e-hp), each read on the six s1eval carves of the chosen pools and compared with zrm's screen fit of its split on
today's pools (scr-zrm, scr-zrm-hp: the same questions and gold totals; R@5 over the question's golds). The pair and
its re-call under the seed null are zlink.py's (relz's, decided against zrm's fit of each split), stamped as step 4e's.

    python outputs/mp_unified/zrm4e.py build --dataset metaqa --carve fit --host
    python outputs/mp_unified/zrm4e.py train --split L-musique --name scr-zrm-e --arm zrm --device cuda --host \\
        --cache-root outputs/step4e/cache
    python outputs/mp_unified/zrm4e.py read --name scr-zrm-e --device cuda --host --cache-root outputs/step4e/cache
    python outputs/mp_unified/zrm4e.py compare --new outputs/screen/fits/scr-zrm-e \\
        --base outputs/screen/fits/scr-zrm,outputs/step1/fits/L-musique --out outputs/screen/scr-zrm-e
    python outputs/mp_unified/zrm4e.py pair --screens outputs/screen/scr-zrm-e.json,outputs/screen/scr-zrm-e-hp.json \\
        --out outputs/screen/scr-zrm-e-pair
    python outputs/mp_unified/zrm4e.py recall --null N1,N2,N3,N4 --pair outputs/screen/scr-zrm-e-pair.json \\
        --out outputs/screen/scr-zrm-e-pair-recall
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zrm as ZRM  # noqa: E402  (sets the cuBLAS workspace before torch loads)
import zlink as ZL  # noqa: E402

RM, LC, log = ZRM.RM, ZRM.LC, ZRM.log
Z = ZL.Z
CACHE = ROOT / "outputs" / "step4e" / "cache"
LOOK = ROOT / "outputs" / "step4e" / "look"
CHAINS = ROOT / "outputs" / "step4e" / "chains"
_BUILT = RM.built_records


def on_chains():
    """rmatch's chain carve and its records read step 4e's chains."""
    RM.ChainCarveBase.CH_ROOT = CHAINS
    RM.built_records = lambda root=CHAINS: _BUILT(root)


def cache_root_of(argv):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--cache-root")
    k, _ = ap.parse_known_args(argv)
    if not k.cache_root or Path(k.cache_root).resolve() != CACHE.resolve():
        raise SystemExit(f"zrm4e: train and read take --cache-root {CACHE.relative_to(ROOT).as_posix()}")


def restamp(out):
    js = Path(out).with_suffix(".json")
    if js.exists():
        rec = json.loads(js.read_text(encoding="utf-8"))
        rec.update({"step": "4e", "declared_in": "docs/STEP4E_PAPER_POOLS.md",
                    "decided_against": "zrm's fit of each split", "new_pools": "outputs/step4e (A3, k = 2)",
                    "base_pools": "today's (step 1's frozen pools)", "zrm4e_sha256": LC.sha_src(__file__)})
        rec.pop("round", None)
        LC.write_json(js, rec)
    md = Path(out).with_suffix(".md")
    if md.exists():
        t = md.read_text(encoding="utf-8")
        t = (f"Step 4e (docs/STEP4E_PAPER_POOLS.md): zrm on the chosen pools against zrm on today's pools.\n\n"
             + t.replace("| zrm R@5 | zlk R@5 |", "| zrm R@5 (today's pools) | zrm R@5 (chosen pools) |")
             .replace("section 2 and the twenty-second round", "section 5 of docs/STEP4E_PAPER_POOLS.md")
             .replace("docs/FULL_ROUND22.md", "a full run in its own file"))
        md.write_text(t, encoding="utf-8")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    cmd = argv[0] if argv else None
    if cmd == "build":
        bp = argparse.ArgumentParser()
        bp.add_argument("cmd")
        bp.add_argument("--dataset", required=True)
        bp.add_argument("--carve", required=True)
        bp.add_argument("--host", action="store_true")
        b = bp.parse_args(argv)
        RM.build(b.dataset, b.carve, CHAINS, cache_root=CACHE, look_root=LOOK,
                 placement={"where": "host" if b.host else "laptop"})
        return 0
    if cmd in ("train", "read"):
        cache_root_of(argv)
        on_chains()
        return ZRM.main(argv)
    if cmd == "compare":
        return ZRM.main(argv)
    if cmd in ("pair", "recall"):
        gp = argparse.ArgumentParser()
        gp.add_argument("cmd")
        gp.add_argument("--screens", default="")
        gp.add_argument("--null", default="")
        gp.add_argument("--pair")
        gp.add_argument("--out", required=True)
        g = gp.parse_args(argv)
        saved = ZL.ARM
        ZL.ARM = "zrm"
        try:
            if cmd == "pair":
                with ZL.on_zrm():
                    Z.pair([x for x in g.screens.split(",") if x], g.out)
            else:
                null = [x for x in g.null.split(",") if x]
                if len(null) != 4 or not g.pair:
                    gp.error("recall: --pair and the four --null files")
                with ZL.on_zrm():
                    Z.recall(null, g.pair, g.out)
        except SystemExit as e:
            log(f"zrm4e {cmd}: {e}")
            return 2
        finally:
            ZL.ARM = saved
        restamp(g.out)
        return 0
    raise SystemExit("zrm4e: build, train, read, compare, pair or recall")


if __name__ == "__main__":
    sys.exit(main())
