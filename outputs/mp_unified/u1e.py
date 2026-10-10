"""U1e: damped links, a cap c on a mention link target's in-degree (docs/U1E_DAMPED_LINKS.md).

    python outputs/mp_unified/u1e.py build --dataset D --cap c [--host]   -> outputs/u1e/c<c>/<D>/graph_structural_u.npz, build.json
    python outputs/host_ops/pylib_run.py outputs/mp_unified/u1e.py coverage --dataset D --cap c --threads 5 --host
                                                                          -> outputs/u1e/c<c>/coverage_<D>.json
    python outputs/mp_unified/u1e.py choose                               -> outputs/u1e/choice.json, report.md
    python outputs/mp_unified/u1e.py --selftest

GU_c is U1d's structural_U without the mention links whose target has more than c mention links in (counted over the
mention links only); every KB triple is kept. Where no link is removed GU_c is GU: the build writes no npz and the
choice reads step 4h's filed GU record. Coverage is step 4h's coverage_stage, unchanged, with its U1D root and OUT at
outputs/u1e/c<c>.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import u1b  # noqa: E402

OUT = ROOT / "outputs" / "u1e"
U1D = ROOT / "outputs" / "u1d"
S4H = ROOT / "outputs" / "step4h"
CAPS = (1024, 4096, 16384)
READS = [(d, "s1sel") for d in ("metaqa", "musique", "squad", "2wiki", "hotpotqa")] + [("webqsp", "s1eval")]
SIX = ("metaqa", "musique", "squad", "2wiki", "hotpotqa", "webqsp")
CONFIG = "W1"              # step 4h's choice (outputs/step4h/choice.json)
TIE = 0.002


def log(m):
    print(time.strftime("[%H:%M:%S] ") + m, flush=True)


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    tmp.replace(p)


def damp(src, dst, rel, mention_id, n, c):
    """Keep every non-mention edge and every mention edge whose target has at most c mention edges in."""
    m = rel == mention_id
    ind = np.bincount(dst[m].astype(np.int64), minlength=n)
    keep = ~m | (ind[dst] <= c)
    return keep, m


def cmd_build(a):
    b = json.loads((U1D / a.dataset / "build.json").read_text(encoding="utf-8"))
    p = U1D / a.dataset / "graph_structural_u.npz"
    if sha_file(p) != b["npz_sha256"]:
        raise SystemExit(f"{p}: not U1d's build")
    z = np.load(p)
    src, dst, rel = z["src"], z["dst"], z["rel"]
    n, mid = int(b["n_nodes"]), int(b["mention_rel_id"])
    keep, m = damp(src, dst, rel, mid, n, a.cap)
    if int((~m).sum()) != int(b["kb_triples_kept"]):
        raise SystemExit(f"{a.dataset}: {int((~m).sum())} non-mention edges, U1d kept {b['kb_triples_kept']} triples")
    d = OUT / f"c{a.cap}" / a.dataset
    rec = {"declared_in": "docs/U1E_DAMPED_LINKS.md", "dataset": a.dataset, "cap": a.cap, "n_nodes": n,
           "mention_rel_id": mid, "variant": f"{b['variant']}-cap{a.cap}", "u1d_npz_sha256": b["npz_sha256"],
           "mention_edges": int(m.sum()), "mention_kept": int((keep & m).sum()), "kb_triples_kept": int((~m).sum()),
           "u1e_sha256": u1b.sha_src(__file__)}
    rec["removed_share"] = round(1 - rec["mention_kept"] / max(rec["mention_edges"], 1), 4)
    if keep.all():
        rec["same_as_gu"] = True
        write_json(d / "build.json", rec)
        log(f"{a.dataset} c={a.cap}: no link removed; GU_c is GU")
        return 0
    S_, D_, R_ = src[keep], dst[keep], rel[keep]      # U1d's order (src, dst) is kept
    tmp = d / "graph_structural_u.tmp.npz"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, src=S_, dst=D_, rel=R_)
    tmp.replace(d / "graph_structural_u.npz")
    rec["npz_sha256"] = sha_file(d / "graph_structural_u.npz")
    rec["edges"] = int(S_.size)
    rec["structural_u"] = u1b.degree_stats(n, S_.astype(np.int64), D_.astype(np.int64))[0]
    if a.dataset in u1b.HYPERLINK:
        served, _freeze = u1b.package_root(a.host)
        ps, pd_, _rel, _fam = u1b.package_structural(served, a.dataset)
        ent = u1b.pr(u1b.pair_keys(n, S_.astype(np.int64), D_.astype(np.int64)), u1b.pair_keys(n, ps, pd_))
        P, R = ent["precision"], ent["recall"]
        ent["f1"] = round(2 * P * R / max(P + R, 1e-12), 4)
        rec["directed"] = ent
    write_json(d / "build.json", rec)
    log(f"{a.dataset} c={a.cap}: kept {rec['mention_kept']} of {rec['mention_edges']} mention links "
        + json.dumps(rec.get("directed", {})))
    return 0


def cmd_coverage(a):
    for p in (ROOT / "scripts", ROOT / "src", ROOT / "outputs" / "step4f"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import step4h_u_pools as H
    d = OUT / f"c{a.cap}"
    b = json.loads((d / a.dataset / "build.json").read_text(encoding="utf-8"))
    if b.get("same_as_gu"):
        raise SystemExit(f"{a.dataset} c={a.cap}: GU_c is GU; the choice reads step 4h's record")
    H.U1D, H.OUT = d, d
    if a.host:
        import run_one
        run_one.on_the_mirror(SimpleNamespace(E=H.E))
    rc = H.coverage_stage([a.dataset], a.threads, ["GU", "G0"], a.host)
    f = d / f"coverage_{a.dataset}.json"
    rec = json.loads(f.read_text(encoding="utf-8"))
    if rec["GU"].get("npz_sha256") != b["npz_sha256"]:
        raise SystemExit("coverage ran on another graph")
    rec["u1e"] = {"cap": a.cap, "u1e_sha256": u1b.sha_src(__file__), "build": b["npz_sha256"]}
    write_json(f, rec)
    return rc


def w1_of(d, carve, cap):
    """(ALL, pool, step 4e pool) of config W1 on GU_c for one read; step 4h's GU where GU_c is GU or c is None."""
    src = "step4h"
    f = S4H / f"coverage_{d}.json"
    if cap is not None:
        b = OUT / f"c{cap}" / d / "build.json"
        if d != "squad":
            if not b.is_file():
                raise SystemExit(f"{b}: not built")
            if not json.loads(b.read_text(encoding="utf-8")).get("same_as_gu"):
                f, src = OUT / f"c{cap}" / f"coverage_{d}.json", f"u1e c{cap}"
    ent = json.loads(f.read_text(encoding="utf-8"))["carves"][carve]
    g = ent["graphs"].get("GU") or ent["graphs"]["G0"]       # squad: no graph, GU is G0
    v = g[CONFIG]
    return v["ALL"], v["pool"], ent["step4e"]["pool"], src


def cmd_choose(_a):
    cands = list(CAPS) + [None]
    rows = {}
    for c in cands:
        reads = {f"{d}/{carve}": w1_of(d, carve, c) for d, carve in READS}
        rows["inf" if c is None else str(c)] = {
            "mean_ALL": round(float(np.mean([r[0] for r in reads.values()])), 4),
            "size_ratio_to_4e": round(float(np.mean([r[1] / r[2] for r in reads.values()])), 3),
            "reads": {k: {"ALL": r[0], "pool": r[1], "from": r[3]} for k, r in reads.items()},
            "s1eval": {d: w1_of(d, "s1eval", c)[0] for d in SIX}}
    ok = {k: r for k, r in rows.items() if r["size_ratio_to_4e"] <= 1.0}
    best = max(r["mean_ALL"] for r in ok.values())
    order = [str(c) for c in CAPS] + ["inf"]                  # smaller c first
    chosen = next(k for k in order if k in ok and ok[k]["mean_ALL"] >= best - TIE)
    f1 = {}
    for c in CAPS:
        f1[str(c)] = {d: json.loads((OUT / f"c{c}" / d / "build.json").read_text(encoding="utf-8")).get("directed", {}).get("f1")
                      for d in u1b.HYPERLINK}
    sc = {d: json.loads((U1D / d / "score.json").read_text(encoding="utf-8")) for d in u1b.HYPERLINK}
    f1["inf"] = {d: sc[d]["variants"]["V2"]["directed"]["f1"] for d in u1b.HYPERLINK}
    out = {"rule": f"highest mean ALL of {CONFIG} over step 4h's six reads (five s1sel, webqsp s1eval) among caps whose "
                   f"mean pool size is at most step 4e's; within {TIE} the smaller c",
           "configs": rows, "chosen": chosen, "hyperlink_f1_reported_not_chosen": f1, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(OUT / "choice.json", out)
    md = ["# U1e: damped links", "", f"Chosen: **c = {chosen}**", "",
          f"| c | mean ALL ({CONFIG}, choice reads) | pool / step 4e | " + " | ".join(f"{d} s1eval" for d in SIX)
          + " | F1 2wiki | F1 hotpotqa |", "| --- " * (5 + len(SIX)) + "|"]
    for k, r in rows.items():
        md.append(f"| {k} | {r['mean_ALL']} | {r['size_ratio_to_4e']} | " + " | ".join(str(r["s1eval"][d]) for d in SIX)
                  + f" | {f1[k]['2wiki']} | {f1[k]['hotpotqa']} |")
    (OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


def selftest():
    n = 6
    src = np.array([0, 1, 2, 3, 4, 0, 1, 5], np.int32)
    dst = np.array([5, 5, 5, 1, 1, 2, 3, 4], np.int32)
    rel = np.array([9, 9, 9, 9, 9, 3, 9, 4], np.int16)          # 9: mentions; 3, 4: KB triples
    keep, m = damp(src, dst, rel, 9, n, 2)
    # mention in-degrees: node 5 has 3, node 1 has 2, node 3 has 1
    assert keep.tolist() == [False, False, False, True, True, True, True, True], keep
    assert m.sum() == 6
    keep, _ = damp(src, dst, rel, 9, n, 3)
    assert keep.all()
    keep, _ = damp(src, dst, np.zeros_like(rel), 0, n, 1)      # passage graphs: every edge is a mention
    assert keep.tolist() == [False, False, False, False, False, True, True, True], keep
    print("u1e selftest: cap on mention in-degree, KB triples kept, passage graphs: ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--selftest" in argv:
        return selftest()
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("build", "coverage", "choose"))
    ap.add_argument("--dataset", choices=u1b.DATASETS)
    ap.add_argument("--cap", type=int, choices=CAPS)
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    return {"build": cmd_build, "coverage": cmd_coverage, "choose": cmd_choose}[a.stage](a)


if __name__ == "__main__":
    sys.exit(main())
