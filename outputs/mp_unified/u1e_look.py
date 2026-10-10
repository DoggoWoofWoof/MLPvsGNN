"""U1e design look (a look, not a result; nothing here is chosen or filed as a number of a stage).

U1d's chosen links (V2) recover 0.57-0.68 of 2wiki's and hotpotqa's hyperlinks at 0.23-0.32 precision; its most-linked
titles are one-word or generic names, and on webqsp the mention links reach mean degree 246 beside the KB's 6.4
(docs/U1D_PRECISE_LINKS.md). This look reads U1d's V2 link shards as they are and asks how a link's precision moves with
its target's in-degree and with its source's out-degree, before any damping rule is declared:

  in-degree buckets   per log2 bucket of the target's V2 in-degree: links, links that are hyperlinks, precision
  target caps         P, R, F1 when a target keeps no V2 link above an in-degree cap (absolute and as a share of n)
  source caps         P, R, F1 when each source keeps its m links to the lowest in-degree targets
  all six             the V2 in-degree distribution on every dataset (how a cap would bite where no hyperlinks exist)

    python outputs/mp_unified/u1e_look.py --dataset 2wiki --host     -> outputs/u1e_look/<dataset>.json

No question, gold, pool or model is read. U1d's code and records are unchanged.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import u1b  # noqa: E402
import u1d  # noqa: E402

OUT = u1d.ROOT / "outputs" / "u1e_look"
ABS_CAPS = (16, 64, 256, 1024, 4096, 16384, 65536)
REL_CAPS = (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3)
SRC_CAPS = (2, 4, 8, 16, 32, 64)


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def f1(both, rule, ref):
    p, r = both / max(rule, 1), both / max(ref, 1)
    return {"rule": int(rule), "both": int(both), "precision": round(p, 4), "recall": round(r, 4),
            "f1": round(2 * p * r / max(p + r, 1e-12), 4)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=u1b.DATASETS)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    sc = json.loads((u1d.OUT / a.dataset / "score.json").read_text(encoding="utf-8"))
    a.shards = int(sc["shards"])
    s, t, n, _recs = u1d.load_variant(a, "V2")
    log(f"{a.dataset}: V2 {s.size} links on {n} nodes ({time.time() - t0:.0f} s)")
    ind = np.bincount(t, minlength=n)
    outd = np.bincount(s, minlength=n)
    tin = ind[t]
    rec = {"dataset": a.dataset, "n_nodes": int(n), "links": int(s.size), "look": "design look, not a result",
           "u1e_look_sha256": u1b.sha_src(__file__), "u1d_sha256": u1b.sha_src(u1d.__file__)}
    q = np.percentile(ind[ind > 0], [50, 90, 99, 99.9]) if s.size else np.zeros(4)
    rec["in_degree"] = {"targets": int((ind > 0).sum()), "p50": float(q[0]), "p90": float(q[1]), "p99": float(q[2]),
                        "p99.9": float(q[3]), "max": int(ind.max()) if s.size else 0}
    rec["out_degree"] = {"sources": int((outd > 0).sum()),
                         "p50": float(np.percentile(outd[outd > 0], 50)) if s.size else 0.0,
                         "p99": float(np.percentile(outd[outd > 0], 99)) if s.size else 0.0}
    # share of links landing on targets above each cap (every dataset)
    rec["links_above_abs_cap"] = {str(c): round(float((tin > c).mean()), 4) for c in ABS_CAPS}
    rec["links_above_rel_cap"] = {str(c): round(float((tin > c * n).mean()), 4) for c in REL_CAPS}
    if a.dataset in u1b.HYPERLINK:
        served, _freeze = u1b.package_root(a.host)
        ps, pd_, _rel, _fam = u1b.package_structural(served, a.dataset)
        ref = u1b.pair_keys(n, ps, pd_)
        key = s.astype(np.int64) * n + t.astype(np.int64)
        hit = np.isin(key, ref, assume_unique=True)
        R = ref.size
        rec["reference"] = int(R)
        rec["all"] = f1(int(hit.sum()), s.size, R)
        b = np.floor(np.log2(np.maximum(tin, 1))).astype(np.int64)
        rec["by_in_degree_log2"] = []
        for k in np.unique(b):
            m = b == k
            rec["by_in_degree_log2"].append({"bucket": f"[{2 ** k}, {2 ** (k + 1)})", "links": int(m.sum()),
                                             "hyperlinks": int(hit[m].sum()),
                                             "precision": round(float(hit[m].mean()), 4)})
        rec["target_cap_abs"] = {str(c): f1(int(hit[tin <= c].sum()), int((tin <= c).sum()), R) for c in ABS_CAPS}
        rec["target_cap_rel"] = {str(c): f1(int(hit[tin <= c * n].sum()), int((tin <= c * n).sum()), R)
                                 for c in REL_CAPS}
        # each source keeps its m links to the lowest in-degree targets (ties: lower target id)
        order = np.lexsort((t, tin, s))
        ss = s[order]
        start = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
        rank = np.arange(ss.size) - np.repeat(start, np.diff(np.r_[start, ss.size]))
        hs = hit[order]
        rec["source_cap"] = {str(m): f1(int(hs[rank < m].sum()), int((rank < m).sum()), R) for m in SRC_CAPS}
        # reference in-degree of the same targets, for scale
        rind = np.bincount(pd_.astype(np.int64), minlength=n)
        rec["reference_in_degree"] = {"p50": float(np.percentile(rind[rind > 0], 50)),
                                      "p99": float(np.percentile(rind[rind > 0], 99)), "max": int(rind.max())}
    rec["secs"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    u1d.write_json(OUT / f"{a.dataset}.json", rec)
    log(f"{a.dataset}: done ({rec['secs']} s) " + json.dumps({k: rec[k] for k in ("all",) if k in rec}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
