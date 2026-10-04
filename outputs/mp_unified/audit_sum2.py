"""audit_sum.py (unchanged numbers and bootstrap) with --partial: a carve whose shards are not all finished is read on the
chunks its FINISHED shards list (record_*.json 'chunks'), so a dataset's audit can be read before every shard has run.
The output records, per carve, which shards and how many of the carve's chunks were read; a partial read is a
subsample of rows (the shard split is L8.shard_chunks's), never mixed with a complete one under one name.
Untracked design look, written new because audit_sum.py's job ran.

    python outputs/mp_unified/audit_sum2.py --partial [--root outputs/mp_unified/look] [--out outputs/mp_unified/audit_sum_2.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KEY = ("recall@5", "full_coverage@5", "hit@1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "look"))
    ap.add_argument("--out", default=str(HERE / "audit_sum_2.json"))
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    res = {}
    for carve_dir in sorted(Path(a.root).glob("*/*")):
        recs = sorted(carve_dir.glob("record*.json"))
        if not recs:
            continue
        rl = [json.loads(r.read_text(encoding="utf-8")) for r in recs]
        r0 = rl[0]
        if not r0.get("audit", {}).get("chunks"):
            continue
        shards = {tuple(r.get("shard") or (0, 1)) for r in rl}
        n_sh = next(iter(shards))[1]
        complete = {s[0] for s in shards} == set(range(n_sh))
        if not complete and not a.partial:
            print(f"{carve_dir}: shards incomplete {sorted(shards)}; skipped")
            continue
        keep = set()
        for r in rl:
            keep.update(int(c) for c in r["chunks"])
        names = r0["metric_names"]
        ki = [names.index(k) for k in KEY]
        pert = [p[0] if isinstance(p, (list, tuple)) else p for p in r0["audit"]["perturbations"]]
        models = r0["audit"]["models"]
        seed = int(r0["audit"]["seed"])
        f_tw, f_gn = r0["functions"].index(f"twin{seed}"), r0["functions"].index(f"gnn{seed}")
        base, aud, used = [], [], 0
        for ch in sorted((carve_dir / "chunks").glob("c*.npz")):
            if ch.name.endswith(".tmp.npz"):
                continue
            ci = int(ch.stem[1:])
            if ci not in keep:
                continue
            with np.load(ch) as z:
                if "q_audit" not in z.files:
                    continue
                qm = z["q_metrics"]
                base.append(np.stack([qm[:, f_tw][:, ki], qm[:, f_gn][:, ki]], 1))       # (rows, 2, 3)
                aud.append(z["q_audit"][:, :, :, ki])                                  # (rows, P, 2, 3)
                used += 1
        if not base:
            continue
        base, aud = np.concatenate(base), np.concatenate(aud)
        n = base.shape[0]
        W = np.random.default_rng(20261004).poisson(1.0, (1000, n)).astype(np.float64)
        ds, carve = carve_dir.parent.name, carve_dir.name
        out = {"rows": n, "complete": complete, "shards_read": sorted(list(s) for s in shards),
               "audited_chunks_read": used, "carve_chunks": r0.get("n_chunks"),
               "base": {m: base[:, j].mean(0).round(4).tolist() for j, m in enumerate(models)},
               "gap_gnn_minus_twin": (base[:, 1] - base[:, 0]).mean(0).round(4).tolist(), "perturbations": {}}
        for pi, pn in enumerate(pert):
            e = {}
            for j, m in enumerate(models):
                d = aud[:, pi, j] - base[:, j]                                             # (rows, 3)
                bs = (W @ d) / W.sum(1, keepdims=True)
                e[m] = {k: [round(float(d[:, c].mean()), 4), [round(float(np.quantile(bs[:, c], 0.025)), 4),
                                                               round(float(np.quantile(bs[:, c], 0.975)), 4)]]
                        for c, k in enumerate(KEY)}
            dd = (aud[:, pi, 1] - base[:, 1]) - (aud[:, pi, 0] - base[:, 0])
            bs = (W @ dd) / W.sum(1, keepdims=True)
            e["gnn_minus_twin_drop"] = {k: [round(float(dd[:, c].mean()), 4), [round(float(np.quantile(bs[:, c], 0.025)), 4),
                                                                               round(float(np.quantile(bs[:, c], 0.975)), 4)]]
                                        for c, k in enumerate(KEY)}
            out["perturbations"][pn] = e
        key = f"{ds}/{carve}" + ("" if complete else "/partial")
        res[key] = out
        print(f"\n{key}: {n} audited rows ({used} chunks; shards {sorted(shards)}); base twin {out['base'][models[0]]} "
              f"gnn {out['base'][models[1]]}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
