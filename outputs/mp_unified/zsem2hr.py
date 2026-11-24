"""SEM2 R1h report under amendment 8.9 (docs/SEM2_EDGE_OPERATORS.md; declared 24 Nov 2026 before any H1/H2 number).

    python outputs/mp_unified/zsem2hr.py [--threads 1]

Each hub method (HB: H1; the adjustment: H2) is carried into R2 if, on e3v and MRR_T, it is ABOVE on MetaQA and BELOW
on none of the seven datasets. 8.6's rule (ABOVE on 4 of 7, BELOW on none) is shown beside, superseded. Reads only
outputs/sem2/r1h/<ds>/<arm>.json; zsem2h is imported and never edited.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2h as H  # noqa: E402  (imports zsem2, whose zenc reads --threads from sys.argv)
import json  # noqa: E402
import time  # noqa: E402

Z = H.Z
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 8.9"
KEYS = {"HB": "H1 hb - rotation", "adjustment": "H2 rotation_la - rotation", "HB + adjustment": "hb_la - rotation"}


def judge(rows, key):
    calls = {d: rows[(d, "e3v")]["calls_mrr_t"][key]["call"] for d in Z.SEVEN if (d, "e3v") in rows}
    missing = [d for d in Z.SEVEN if d not in calls]
    below = [d for d, v in calls.items() if v == "BELOW"]
    meta = calls.get("metaqa")
    if below:
        v = "NOT_CARRIED"
    elif missing:
        v = "INCOMPLETE"
    else:
        v = "CARRIED" if meta == "ABOVE" else "NOT_CARRIED"
    above = sum(c == "ABOVE" for c in calls.values())
    old = "INCOMPLETE" if missing else ("CARRIED" if above >= 4 and not below else "NOT_CARRIED")
    md = rows[("metaqa", "e3v")]["calls_mrr_t"][key]["diff"] if ("metaqa", "e3v") in rows else None
    return {"verdict": v, "metaqa": meta, "metaqa_diff": md, "below": below, "missing": missing, "calls": calls,
            "rule_8_6_superseded": {"verdict": old, "above": above}}


def main():
    rows = {}
    for d in Z.SEVEN:
        for arm in Z.ARMS:
            p = H.OUT / d / f"{arm}.json"
            if p.is_file():
                r = json.loads(p.read_text(encoding="utf-8"))
                if r.get("stage") != "SEM2 R1h":
                    Z.stop(f"{p} is not an R1h record")
                rows[(d, arm)] = r
    carry = {k: judge(rows, key) for k, key in KEYS.items()}
    hb, la, both = carry["HB"]["verdict"], carry["adjustment"]["verdict"], carry["HB + adjustment"]
    if "INCOMPLETE" in (hb, la):
        r2 = "INCOMPLETE"
    elif hb == "CARRIED" and la == "CARRIED":
        if both["verdict"] == "CARRIED":
            r2 = "hb_la"
        else:
            r2 = "hb" if carry["HB"]["metaqa_diff"] >= carry["adjustment"]["metaqa_diff"] else "rotation_la"
    elif hb == "CARRIED":
        r2 = "hb"
    elif la == "CARRIED":
        r2 = "rotation_la"
    else:
        r2 = "none: the next hub step is declared from the per-label reads"
    rep = {"stage": "SEM2 R1h report (8.9)", "declared_in": DECLARED, "carry": carry, "r2_form": r2,
           "H0": {f"{d} {arm}": H.CALLS[r["calls_mrr_t"]["H0 rotation - cosine"]["call"]] for (d, arm), r in rows.items()},
           "rows": {f"{d} {arm}": {"calls_mrr_t": r["calls_mrr_t"], "beside_pooled": r["beside_pooled"],
                                   "mrr_t": {k: v["mrr_t"] for k, v in r["read"].items()},
                                   "pooled": {k: v["pooled"].get("mrr") for k, v in r["read"].items()},
                                   "per_group_mrr_t": r["per_group_mrr_t"], "gates": r["gates"],
                                   "sha256": Z.sha_file(H.OUT / d / f"{arm}.json")} for (d, arm), r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": Z.sha_file(Path(__file__)),
           "zsem2h_sha256": Z.sha_file(Path(H.__file__))}
    Z.wjson(H.OUT / "report.json", rep)
    f = lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"  # noqa: E731
    L = ["# SEM2 R1h: hubs (report under amendment 8.9)", "",
         f"Declared in {DECLARED}. Written by zsem2hr.py at {rep['utc']}. MRR_T in points (each read group and target "
         "once); intervals are paired 95% bootstraps over those groups. Pooled MRR beside.", "",
         "| dataset | arm | cosine | rotation | rotation+adj | HB | HB+adj | popularity | H1 (HB) | H2 (adj) | HB+adj vs R1 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for (d, arm), r in rows.items():
        m, c = rep["rows"][f"{d} {arm}"]["mrr_t"], r["calls_mrr_t"]
        L.append(f"| {d} | {arm} | {m['cosine']} | {m['rotation']} | {m['rotation_la']} | {m['hb']} | {m['hb_la']} | "
                 f"{m['popularity']} | {f(c[KEYS['HB']])} | {f(c[KEYS['adjustment']])} | {f(c[KEYS['HB + adjustment']])} |")
    L += ["", "| dataset | arm | pooled: rotation | rotation+adj | HB | HB+adj |", "| --- | --- | --- | --- | --- | --- |"]
    for (d, arm), r in rows.items():
        p = rep["rows"][f"{d} {arm}"]["pooled"]
        L.append(f"| {d} | {arm} | {p['rotation']} | {p['rotation_la']} | {p['hb']} | {p['hb_la']} |")
    L += ["", "**Carry (8.9: ABOVE on MetaQA, BELOW on none; e3v, MRR_T):**"]
    for k, v in carry.items():
        L.append(f"- {k}: {v['verdict']} (MetaQA {v['metaqa']}, BELOW on {v['below'] or 'none'}"
                 f"{', missing ' + ', '.join(v['missing']) if v['missing'] else ''}); 8.6's rule, superseded: "
                 f"{v['rule_8_6_superseded']['verdict']} ({v['rule_8_6_superseded']['above']} ABOVE)")
    L += ["", f"**R2 form:** {r2}", ""]
    (H.OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    Z.log(f"report (8.9): {r2}; " + "; ".join(f"{k} {v['verdict']}" for k, v in carry.items()))
    return 0


if __name__ == "__main__":
    if "--threads" in sys.argv:
        i = sys.argv.index("--threads")
        del sys.argv[i:i + 2]
    sys.exit(main())
