"""SEM2 R1k split (docs/SEM2_EDGE_OPERATORS.md section 8.12; post hoc, decides nothing): 8.11's ceiling gain C over
the test edges whose target is a training node (seen) and the rest (test or dev nodes: unseen, degree 0 by the split).

    python outputs/mp_unified/zsem2hk2.py split --ds D [--arm e3v] [--host] [--threads 1]
    python outputs/mp_unified/zsem2hk2.py report

Reads outputs/sem2/r1k/<ds>/<arm>.npz (sha checked against its record) and the graph (for deg_L and the split); no
ranking is recomputed. Gate G-A: the "all" subset reproduces the record's C exactly. Writes outputs/sem2/r1k/split/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2h as H  # noqa: E402  (imports zsem2, whose zenc reads --threads from sys.argv)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

Z = H.Z
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 8.12"
K = Z.ROOT / "outputs" / "sem2" / "r1k"
OUT = K / "split"
log, stop = Z.log, Z.stop


def subsets(split, deg, z):
    lab, d = z["test_lab"], z["test_d"]
    dt = deg.of(lab, d)
    seen = split[d] == Z.TRAIN
    return {"all": np.ones(d.size, bool), "seen target (train node)": seen, "unseen target (test or dev node)": ~seen,
            "seen, degree > 0": seen & (dt > 0), "seen, degree 0": seen & (dt == 0)}


def read(rr, rc, keys, m, nk):
    if not m.any():
        return None
    u = np.unique(keys[m]).size
    return {"edges": int(m.sum()), "targets": int(u), "target_share_pct": round(100 * u / nk, 2),
            "rotation": H.mrr_t(rr[m], keys[m]), "ceiling": H.mrr_t(rc[m], keys[m]), "C": H.paired_t(rc[m], rr[m], keys[m])}


def cmd_split(a):
    t0 = time.time()
    rec = json.loads((K / a.ds / f"{a.arm}.json").read_text(encoding="utf-8"))
    pz = K / a.ds / f"{a.arm}.npz"
    if Z.sha_file(pz) != rec["npz_sha256"]:
        stop(f"{pz}: sha256 is not its record's")
    z = np.load(pz)
    G = (Z.load_mhr(a.arm) if a.ds == "multihoprag" else Z.load_six(a.ds, a.arm, a.host))[0]
    split = Z.node_split(G.name, G.n)
    deg = H.Deg(G, split)
    rr, rc = z["rank_rotation"], z["rank_rotation_ceiling"]
    keys = H.gkeys(z["test_grp"], z["test_d"])
    nk = np.unique(keys).size
    sub = subsets(split, deg, z)
    out = {k: read(rr, rc, keys, m, nk) for k, m in sub.items()}
    ga = out["all"]["C"] == rec["C"]
    if not ga:
        stop(f"gate G-A: {out['all']['C']} is not the record's {rec['C']}")
    names = rec["read_groups"]
    per_group = {}
    for g, nm in enumerate(names):
        mg = z["test_grp"] == g
        if mg.any():
            per_group[nm] = {k: read(rr, rc, keys, mg & sub[k], np.unique(keys[mg]).size)
                             for k in ("seen target (train node)", "unseen target (test or dev node)")}
    res = {"stage": "SEM2 R1k split", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "gates": {"G-A": "PASS"},
           "subsets": out, "per_group": per_group, "r1k_record_sha256": Z.sha_file(K / a.ds / f"{a.arm}.json"),
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "script_sha256": Z.sha_file(Path(__file__))}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / f"{a.ds}_{a.arm}.json", res)
    log(f"{a.ds} {a.arm}: " + "; ".join(f"{k}: C {v['C']['diff']:+.2f} over {v['target_share_pct']}% of targets"
                                        for k, v in out.items() if v) + f" ({res['seconds']} s)")
    return 0


def cmd_report(a):
    rows = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT.glob("*_e3v.json"))}
    f = lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]"  # noqa: E731
    L = ["# SEM2 R1k split: the ceiling over seen and unseen targets (post hoc)", "",
         f"Declared in {DECLARED}; decides nothing (8.11's verdict stands). MRR_T in points; C = ceiling - rotation.", "",
         "| dataset | subset | targets % | rotation | ceiling | C |", "| --- | --- | --- | --- | --- | --- |"]
    for nm, r in rows.items():
        for k, v in r["subsets"].items():
            if v:
                L.append(f"| {r['dataset']} | {k} | {v['target_share_pct']} | {v['rotation']} | {v['ceiling']} | {f(v['C'])} |")
    rep = {"stage": "SEM2 R1k split report", "declared_in": DECLARED,
           "rows": {nm: r["subsets"] for nm, r in rows.items()},
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": Z.sha_file(Path(__file__))}
    Z.wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {len(rows)} rows")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("split", "report"))
    p.add_argument("--ds", choices=Z.SEVEN)
    p.add_argument("--arm", choices=Z.ARMS, default="e3v")
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args(argv)
    if a.cmd == "report":
        return cmd_report(a)
    if not a.ds:
        stop("split needs --ds")
    return cmd_split(a)


if __name__ == "__main__":
    sys.exit(main())
