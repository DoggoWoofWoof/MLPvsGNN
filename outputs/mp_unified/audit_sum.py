"""Summarise look_x_six's frozen-model audit (untracked design look): per dataset, carve and perturbation, the change in
recall@5, full_coverage@5 and hit@1 of the twin and the GNN (audit seed's pair) when a column block is permuted within
each row, the node embeddings are permuted, or an edge family is dropped, with a row bootstrap (1000 Poisson draws).
Runs where the chunks are (the host); reads only finished shards (every record present).

    python outputs/mp_unified/audit_sum.py [--root outputs/mp_unified/look] [--out outputs/mp_unified/audit_sum.json]
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
    ap.add_argument("--out", default=str(HERE / "audit_sum.json"))
    a = ap.parse_args()
    res = {}
    for carve_dir in sorted(Path(a.root).glob("*/*")):
        recs = sorted(carve_dir.glob("record*.json"))
        if not recs:
            continue
        r0 = json.loads(recs[0].read_text(encoding="utf-8"))
        if not r0.get("audit", {}).get("chunks"):
            continue
        shards = {tuple(json.loads(r.read_text(encoding="utf-8")).get("shard") or (0, 1)) for r in recs}
        n_sh = next(iter(shards))[1]
        if {s[0] for s in shards} != set(range(n_sh)):
            print(f"{carve_dir}: shards incomplete {sorted(shards)}; skipped")
            continue
        names = r0["metric_names"]
        ki = [names.index(k) for k in KEY]
        pert = [p[0] if isinstance(p, (list, tuple)) else p for p in r0["audit"]["perturbations"]]
        models = r0["audit"]["models"]
        seed = int(r0["audit"]["seed"])
        f_tw, f_gn = r0["functions"].index(f"twin{seed}"), r0["functions"].index(f"gnn{seed}")
        base, aud = [], []
        for ch in sorted((carve_dir / "chunks").glob("c*.npz")):
            with np.load(ch) as z:
                if "q_audit" not in z.files:
                    continue
                qm = z["q_metrics"]
                base.append(np.stack([qm[:, f_tw][:, ki], qm[:, f_gn][:, ki]], 1))       # (rows, 2, 3)
                aud.append(z["q_audit"][:, :, :, ki])                                  # (rows, P, 2, 3)
        if not base:
            continue
        base, aud = np.concatenate(base), np.concatenate(aud)
        n = base.shape[0]
        W = np.random.default_rng(20261004).poisson(1.0, (1000, n)).astype(np.float64)
        ds, carve = carve_dir.parent.name, carve_dir.name
        out = {"rows": n, "base": {m: base[:, j].mean(0).round(4).tolist() for j, m in enumerate(models)},
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
        res[f"{ds}/{carve}"] = out
        print(f"\n{ds}/{carve}: {n} audited rows; base twin {out['base'][models[0]]} gnn {out['base'][models[1]]}")
        print(f"  {'perturbation':22s} {'twin R@5':>16s} {'gnn R@5':>16s} {'twin hit@1':>16s} {'gnn hit@1':>16s}")
        for pn, e in out["perturbations"].items():
            def c(m, k):
                v, ci = e[m][k]
                return f"{100 * v:+6.2f}{'*' if ci[0] > 0 or ci[1] < 0 else ' '}"
            print(f"  {pn:22s} {c(models[0], 'recall@5'):>16s} {c(models[1], 'recall@5'):>16s} {c(models[0], 'hit@1'):>16s} {c(models[1], 'hit@1'):>16s}")
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
