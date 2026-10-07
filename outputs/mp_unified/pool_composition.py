"""Pool composition per carve (docs/SCREENS.md, a diagnosis): rows per question, the share of a pool's rows that
retrieval ranked (rrf > 0), golds per question, and the share of golds that retrieval ranked. Reads the cache's arrays
memory-mapped; no features are computed and no model is read.

    python outputs/mp_unified/pool_composition.py --carves metaqa=fit,squad=fit,hotpotqa=fit,2wiki=fit \\
        --out outputs/screen/pool_composition
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

import lean_cache as LC  # noqa: E402
import lean_mlp as LM  # noqa: E402


def carve_stats(ds, cv, cache_root):
    parts, recs = LC.part_dirs(ds, cv, cache_root)
    per_q = {"rows": [], "ranked_share": [], "golds": [], "golds_ranked_share": [], "ranked_rows": []}
    for p, r in zip(parts, recs):
        c_rrf = r["xc_spans"]["rank"][0] + LM.SPLIT["rank"].index("rrf")
        rrf = np.asarray(LC.load_array(p, "xc", r, False)[:, c_rrf], np.float32)
        gold = np.asarray(LC.load_array(p, "gold", r, False), bool)
        n = np.asarray(LC.load_array(p, "n", r, False), np.int64)
        off = np.concatenate([[0], np.cumsum(n)])
        ranked = np.isfinite(rrf) & (rrf > 0)
        for q in range(n.size):
            a, e = int(off[q]), int(off[q + 1])
            rk, g = ranked[a:e], gold[a:e]
            per_q["rows"].append(e - a)
            per_q["ranked_rows"].append(int(rk.sum()))
            per_q["ranked_share"].append(float(rk.mean()) if e > a else 0.0)
            per_q["golds"].append(int(g.sum()))
            if g.any():
                per_q["golds_ranked_share"].append(float(rk[g].mean()))
    out = {"questions": len(per_q["rows"]), "questions_with_gold_in_pool": len(per_q["golds_ranked_share"])}
    for k, v in per_q.items():
        v = np.asarray(v, np.float64)
        out[k] = {"mean": round(float(v.mean()), 4), "p10": round(float(np.percentile(v, 10)), 4),
                  "median": round(float(np.median(v)), 4), "p90": round(float(np.percentile(v, 90)), 4)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--carves", required=True, help="DS=CARVE,...")
    ap.add_argument("--cache-root")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    cache_root = Path(a.cache_root) if a.cache_root else LC.OUT
    rec = {"cache_root": str(cache_root), "carves": {}, "script_sha256": LC.sha_src(__file__),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    md = ["# Pool composition (a diagnosis)", "", "Per question, then averaged over questions: rows in the pool, the share "
          "of rows that retrieval ranked (rrf > 0), golds in the pool, and the share of those golds that retrieval ranked.",
          "", "| carve | questions | rows (median) | ranked rows (median) | ranked share | golds in pool | golds ranked share |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for ds, cv in LM.parse_sets(a.carves):
        t = time.time()
        st = carve_stats(ds, cv, cache_root)
        rec["carves"][f"{ds}={cv}"] = st
        md.append(f"| {ds}={cv} | {st['questions']} | {st['rows']['mean']:.0f} ({st['rows']['median']:.0f}) | "
                  f"{st['ranked_rows']['mean']:.1f} ({st['ranked_rows']['median']:.0f}) | {st['ranked_share']['mean']:.3f} | "
                  f"{st['golds']['mean']:.2f} | {st['golds_ranked_share']['mean']:.3f} |")
        LC.log(f"  {ds}={cv}: {json.dumps({k: st[k]['mean'] for k in ('rows', 'ranked_rows', 'ranked_share', 'golds', 'golds_ranked_share')})} "
               f"({time.time() - t:.0f}s)")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
