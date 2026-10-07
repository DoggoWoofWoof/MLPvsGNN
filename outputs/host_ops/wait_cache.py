"""Wait until every part of the named lean caches is written (systems code; no science).

    python outputs/host_ops/wait_cache.py --carves metaqa:select,metaqa:s1eval,... [--root outputs/step1/cache]
        [--timeout-h 10] [--poll 60]

A feeder item that depends on a cache job is dropped for good when that job fails, even on a transient Windows file
race that a rerun fixes: lc1d-metaqa-s1eval-5 and -6 (7 Oct 05:36) failed on "[WinError 32]" in os.replace, and every
step-1 and step-2 read, check and grade was dropped with them. Reads that depend on this gate wait for the parts
themselves instead: a failed shard is requeued under a new name, and the gate sees its part when the rerun writes it.

A carve is ready when lean_cache.part_dirs would accept it: its part_KofN directories are all present (one N), each
has its record.json (lean_cache writes it after every array), and the records tile the carve's queries. The arrays'
sha256 are left to the readers, which check them on load. Exits 0 when every carve is ready and 2 at the time limit.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def carve_state(root, ds, carve):
    """'ready', or what is missing."""
    d = Path(root) / ds / carve
    parts = sorted(d.glob("part_*of*"), key=lambda p: int(p.name.split("_")[1].split("of")[0]))
    if not parts:
        return "no part yet"
    nparts = {int(p.name.split("of")[1]) for p in parts}
    if len(nparts) != 1:
        return f"parts of different splits: {[p.name for p in parts]}"
    n = nparts.pop()
    if len(parts) != n:
        return f"{len(parts)} of {n} parts started"
    recs = []
    for p in parts:
        f = p / "record.json"
        if not f.is_file():
            return f"{p.name} has no record yet"
        try:
            recs.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            return f"{p.name}: record unreadable ({type(e).__name__})"
    q = 0
    for p, r in zip(parts, recs):
        if r["query_range"][0] != q:
            return f"{p.name}: query range {r['query_range']} does not follow {q}"
        q = r["query_range"][1]
    want = recs[0]["look"]["carve_queries"]
    if q != want:
        return f"the parts hold {q} of {want} queries"
    return "ready"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--carves", required=True, help="dataset:carve,dataset:carve,...")
    ap.add_argument("--root", default=str(ROOT / "outputs" / "step1" / "cache"))
    ap.add_argument("--timeout-h", type=float, default=10.0)
    ap.add_argument("--poll", type=float, default=60.0)
    a = ap.parse_args(argv)
    want = [tuple(x.split(":", 1)) for x in a.carves.split(",") if x]
    if not want or any(len(w) != 2 for w in want):
        ap.error("--carves takes dataset:carve pairs")
    t0, last = time.time(), None
    while True:
        st = {f"{ds}/{c}": carve_state(a.root, ds, c) for ds, c in want}
        todo = {k: v for k, v in st.items() if v != "ready"}
        if todo != last:
            stamp = time.strftime("%H:%M:%S")
            print(f"[{stamp}] {len(st) - len(todo)} of {len(st)} carves ready"
                  + ("" if not todo else "; waiting on " + "; ".join(f"{k}: {v}" for k, v in sorted(todo.items()))),
                  flush=True)
            last = todo
        if not todo:
            return 0
        if time.time() - t0 > a.timeout_h * 3600:
            print(f"time limit ({a.timeout_h:g} h) reached", flush=True)
            return 2
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
