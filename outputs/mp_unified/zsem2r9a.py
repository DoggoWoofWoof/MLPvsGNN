"""SEM2 R9 part A9, the reads declared in section 17.4 that the report of zsem2r9.py does not print (docs/SEM2_EDGE_OPERATORS.md, 17.11).

Read-only on outputs/sem2/r9/<dataset>/cap.json and cap.npz (the records zsem2r9.py wrote; their code shas and npz shas are checked by zsem2r9.load_rec).
Nothing is solved, nothing is fitted. Writes outputs/sem2/r9/report_cap_beside.json and .md.

For every dataset with a cap record and every scorer W in (W0, W1):
  1. Per rung: the share of the scorer's instances proven optimal, and the share pooled over both scorers (17.4: "a rung's instances"); the cells that the
     pooled reading would not read although the per-scorer reading did (the report of zsem2r9.py reads per scorer and rung).
  2. Recall and completion gains, rung against C1, with the completion read where the rung is read; the share of the recall gain the completion gain is.
  3. The paired completion on the questions proven at BOTH rungs (a selected subset, counted: it is not a call): C2 - C1 and C3 - C1 in points, paired interval, n.
  4. W0 - W1 at each rung on the questions proven in both (17.7 XC3).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import zsem2r9 as R9  # noqa: E402

Z = R9.Z
RUNGS = R9.RUNGS
SC9 = R9.SC9
OPT = R9.OPT_SHARE


def rung_name(ri):
    return f"{RUNGS[ri][0]}/{RUNGS[ri][1]}"


def read_ds(ds):
    r, z = R9.load_rec(ds, "cap")
    if r is None:
        return None
    proven = z["proven"].astype(bool)          # [Q, scorer, rung]
    comp = z["comp"].astype(float)
    cand = z["cand"].astype(float)
    out = {"slice": int(comp.shape[0]), "rungs": {}, "paired": {}, "w0_minus_w1": {}}
    for ri in range(len(RUNGS)):
        pooled = float(proven[:, :, ri].mean())
        row = {"pooled_proven_share": round(pooled, 4), "pooled_reads": bool(pooled >= OPT), "unproven_instances": int((~proven[:, :, ri]).sum()), "per_scorer": {}}
        for si, s in enumerate(SC9):
            sh = float(proven[:, si, ri].mean())
            row["per_scorer"][s] = {"proven_share": round(sh, 4), "per_scorer_reads": bool(sh >= OPT), "flips_under_pooled": bool(sh >= OPT and pooled < OPT)}
        out["rungs"][rung_name(ri)] = row
    for si, s in enumerate(SC9):
        out["paired"][s] = {}
        for ri in (1, 2):
            both = proven[:, si, 0] & proven[:, si, ri]
            n = int(both.sum())
            if n > 1:
                c = Z.paired(comp[both, si, ri], comp[both, si, 0])
            else:
                c = {"diff": None, "lo": None, "hi": None, "call": "NA"}
            rg = float(cand[:, si, ri].mean() - cand[:, si, 0].mean()) * 100
            out["paired"][s][rung_name(ri)] = {"n_proven_at_both": n, "of": int(comp.shape[0]), "completion_minus_C1": c,
                                               "completion_C1_on_subset_pct": round(100 * float(comp[both, si, 0].mean()), 2) if n else None,
                                               "completion_rung_on_subset_pct": round(100 * float(comp[both, si, ri].mean()), 2) if n else None,
                                               "recall_gain_points_all_questions": round(rg, 2)}
    for ri in range(len(RUNGS)):
        both = proven[:, 0, ri] & proven[:, 1, ri]
        n = int(both.sum())
        c = Z.paired(comp[both, 0, ri], comp[both, 1, ri]) if n > 1 else {"diff": None, "lo": None, "hi": None, "call": "NA"}
        out["w0_minus_w1"][rung_name(ri)] = {"n_proven_in_both": n, "completion_W0_minus_W1": c,
                                             "recall_W0_minus_W1_points": round(100 * float(cand[:, 0, ri].mean() - cand[:, 1, ri].mean()), 2)}
    return out


def main():
    rep = {"declared_in": R9.DECLARED, "code_sha256": R9.code_shas(), "datasets": {}, "missing": []}
    for ds in R9.FIVE:
        o = read_ds(ds)
        if o is None:
            rep["missing"].append(ds)
        else:
            rep["datasets"][ds] = o
    Z.wjson(R9.OUT / "report_cap_beside.json", rep)
    L = ["# SEM2 R9 part A9: the reads beside the ladder (17.11)", "", "Read-only on the cap records; nothing solved; the paired reads are on a selected subset (the questions both programs were proven on) and decide nothing.", "",
         "## Proven shares per rung, pooled over both scorers (the declared reading) and per scorer", "",
         "| dataset | rung | unproven instances | pooled share | pooled reads | W0 share | W1 share | cells that flip |", "|---|---|---:|---:|---|---:|---:|---|"]
    for ds, o in rep["datasets"].items():
        for rn, rw in o["rungs"].items():
            flips = ", ".join(s for s, v in rw["per_scorer"].items() if v["flips_under_pooled"]) or "-"
            L.append(f"| {ds} | {rn} | {rw['unproven_instances']} | {rw['pooled_proven_share']:.4f} | {rw['pooled_reads']} | {rw['per_scorer']['W0']['proven_share']:.4f} | {rw['per_scorer']['W1']['proven_share']:.4f} | {flips} |")
    L += ["", "## Completion against the rung 100/50, on the questions proven at both (n of the slice)", "",
          "| dataset | scorer | rung | n | C1 on the subset % | rung on the subset % | difference (points) | interval | call | recall gain, all questions (points) |", "|---|---|---|---:|---:|---:|---:|---|---|---:|"]
    for ds, o in rep["datasets"].items():
        for s in SC9:
            for rn, pr in o["paired"][s].items():
                c = pr["completion_minus_C1"]
                L.append(f"| {ds} | {s} | {rn} | {pr['n_proven_at_both']} of {pr['of']} | {pr['completion_C1_on_subset_pct']} | {pr['completion_rung_on_subset_pct']} | {c['diff']} | "
                         f"[{c['lo']}, {c['hi']}] | {c['call']} | {pr['recall_gain_points_all_questions']} |")
    L += ["", "## W0 minus W1 at each rung, on the questions proven in both", "", "| dataset | rung | n | completion W0 - W1 (points) | interval | call | recall W0 - W1 (points) |", "|---|---|---:|---:|---|---|---:|"]
    for ds, o in rep["datasets"].items():
        for rn, w in o["w0_minus_w1"].items():
            c = w["completion_W0_minus_W1"]
            L.append(f"| {ds} | {rn} | {w['n_proven_in_both']} | {c['diff']} | [{c['lo']}, {c['hi']}] | {c['call']} | {w['recall_W0_minus_W1_points']} |")
    L += ["", f"Missing: {', '.join(rep['missing']) if rep['missing'] else 'none'}."]
    (R9.OUT / "report_cap_beside.md").write_text("\n".join(L) + "\n", encoding="utf-8", newline="\n")
    print("beside report written", sorted(rep["datasets"]), "missing", rep["missing"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
