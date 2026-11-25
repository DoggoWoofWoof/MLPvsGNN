"""PL2 report (docs/PL2_COVERAGE_TARGET.md section 10): the tables of outputs/pl2/report.md, read from the filed records only.

    python outputs/mp_unified/zpl2r.py [--out outputs/pl2/report.md]

Reads outputs/pl2/<dataset>/{select.json, confirm_frozen.json, confirm_e3v.json}, outputs/pl2/call.json and the filed reachability blocks of
outputs/m3a/headroom/<dataset>.json. A dataset without a select.json is listed as NOT READ. Nothing here chooses a cell or computes a pool:
every number is one a record holds. Writes outputs/pl2/report.md and outputs/pl2/report.json.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PL2 = ROOT / "outputs" / "pl2"
ORDER = ("multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
BOUND = 2500


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def f4(x):
    return "-" if x is None else f"{x:.4f}"


def f1(x):
    return "-" if x is None else f"{x:,.1f}"


def pts(x):
    return "-" if x is None else f"{100 * x:+.2f}"


def cell(c):
    return f"`{c['name']}` ({f4(c['all'])} at {f1(c['nodes'])}{'' if c.get('reached', True) else ', SHORT'})"


def reach(ds):
    """The filed reach ceilings (M3A's populations): share of questions missing a gold in the union, and the golds' reach in 1, 2, 3 hops."""
    if ds == "multihoprag":
        return None
    p = ROOT / "outputs" / "m3a" / "headroom" / f"{ds}.json"
    if not p.is_file():
        return None
    r = rj(p)["reachability"]
    out = {}
    for rg in ("STRUCT", "NER", "KNN", "FULL"):
        v = r.get(rg)
        if not v:
            continue
        b = v["buckets"]
        out[rg] = {"questions": v["queries_scanned"], "with_missing": v["queries_with_missing_gold"],
                   "w1": b["missing_golds_reachable_within_1_fraction"], "w2": b["missing_golds_reachable_within_2_fraction"],
                   "w3": b["missing_golds_reachable_within_3_fraction"], "all3": b["queries_with_all_missing_golds_within_3"]}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PL2 / "report.md"))
    a = ap.parse_args(argv)
    rep, md = {}, []
    sel, conf = {}, {}
    for ds in ORDER:
        sp = PL2 / ds / "select.json"
        if sp.is_file():
            sel[ds] = rj(sp)
            conf[ds] = {fs: rj(PL2 / ds / f"confirm_{fs}.json") for fs in ("frozen", "e3v") if (PL2 / ds / f"confirm_{fs}.json").is_file()}
    md.append("### Dev reading (the rule's own numbers, dev carve, frozen lists)\n")
    md.append("| dataset | dev q | R95 retrieval-only | R95 with links | R98 retrieval-only | R98 with links | OLD (own construction) |")
    md.append("|---|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            md.append(f"| {ds} | NOT READ | | | | | |")
            continue
        s = sel[ds]
        p = s["picks"]
        md.append(f"| {ds} | {s['questions']} | {cell(p['R95_retrieval'])} | {cell(p['R95_graph'])} | {cell(p['R98_retrieval'])} | {cell(p['R98_graph'])} | "
                  f"`{s['old']['name']}` ({f4(s['old']['all'])} at {f1(s['old']['nodes'])}) |")
        rep[ds] = {"dev_questions": s["questions"], "picks": {k: {x: v[x] for x in ("name", "all", "any", "nodes", "reached")} for k, v in p.items()},
                   "old": {x: s["old"][x] for x in ("name", "all", "nodes")}}
    md.append("")
    md.append("### The chosen cell (R95 over all cells) and its confirmation on eval\n")
    md.append("| dataset | chosen cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v | OLD eval ALL frozen | OLD eval ALL e3v |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        c = sel[ds]["picks"]["R95_all"]
        row = [f"| {ds} | `{c['name']}`{'' if c['reached'] else ' (SHORT)'} | {f4(c['all'])} | {f1(c['nodes'])}"]
        for fs in ("frozen", "e3v"):
            cf = conf[ds].get(fs)
            n = cf["named"].get("R95_all") if cf else None
            row.append(f"{f4(n['eval_all'])} ({f1(n['eval_nodes'])}){'' if n['confirmed'] else ' SHORT'}" if n else "not read")
        for fs in ("frozen", "e3v"):
            cf = conf[ds].get(fs)
            row.append(f4(cf["old"]["eval_all"]) if cf and cf.get("old") else "not read")
        md.append(" | ".join(row) + " |")
    md.append("")
    md.append("### The same at 0.98 (R98 over all cells) on eval\n")
    md.append("| dataset | chosen cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |")
    md.append("|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        c = sel[ds]["picks"]["R98_all"]
        row = [f"| {ds} | `{c['name']}`{'' if c['reached'] else ' (SHORT)'} | {f4(c['all'])} | {f1(c['nodes'])}"]
        for fs in ("frozen", "e3v"):
            cf = conf[ds].get(fs)
            n = cf["named"].get("R98_all") if cf else None
            row.append(f"{f4(n['eval_all'])} ({f1(n['eval_nodes'])}){'' if n['confirmed'] else ' SHORT'}" if n else "not read")
        md.append(" | ".join(row) + " |")
    md.append("")
    md.append("### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here), retrieval-only and with links\n")
    md.append("| dataset | retrieval-only 0.95 | with links 0.95 | retrieval-only 0.98 | with links 0.98 |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        tb = sel[ds]["table"]
        cs = []
        for fl in (0.95, 0.98):
            for part in ("retrieval", "graph"):
                rows = [x for x in tb if (x["family"] in ("R0", "R1")) == (part == "retrieval") and x["all"] >= fl]
                if rows:
                    m = min(rows, key=lambda x: (x["nodes"], -x["all"], x["name"]))
                    cs.append(f"`{m['name']}` {f4(m['all'])} at {f1(m['nodes'])}")
                else:
                    bb = max((x for x in tb if (x["family"] in ("R0", "R1")) == (part == "retrieval")), key=lambda x: (x["all"], -x["nodes"]))
                    cs.append(f"none on the grid (best {f4(bb['all'])})")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
    md.append("")
    md.append("### The highest ALL on dev (any cell; the bound is 2,500 mean nodes), retrieval-only and with links\n")
    md.append("| dataset | retrieval-only, within the bound | retrieval-only, any size | with links, within the bound | with links, any size |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        fr = sel[ds]["frontier"]

        def best(rows, bound):
            r = [x for x in rows if bound is None or x["nodes"] <= bound]
            return max(r, key=lambda x: (x["all"], -x["nodes"])) if r else None
        cs = []
        for part in ("retrieval", "graph"):
            for bd in (BOUND, None):
                b = best(fr[part], bd)
                cs.append(f"`{b['name']}` {f4(b['all'])} at {f1(b['nodes'])}" if b else "-")
        md.append(f"| {ds} | " + " | ".join(cs) + " |")
    md.append("")
    md.append("### What the links bought (dev frontier of the graph cells minus the retrieval cells, at equal mean nodes; ALL)\n")
    grid_n = (25, 50, 75, 100, 150, 200, 300, 500, 1000, 2000, 2500)
    md.append("| dataset | " + " | ".join(str(n) for n in grid_n) + " |")
    md.append("|---|" + "---|" * len(grid_n))
    for ds in ORDER:
        if ds not in sel:
            continue
        b = {x["nodes"]: x for x in sel[ds]["bought"]}
        cells_ = []
        for n in grid_n:
            x = b.get(n)
            if not x:
                cells_.append("-")
                continue
            cells_.append(f"{f4(x['retrieval'])} / {f4(x['graph'])} ({pts(x['links_bought'])})" if x["graph"] is not None else f"{f4(x['retrieval'])} / none")
        md.append(f"| {ds} | " + " | ".join(cells_) + " |")
        rep.setdefault(ds, {})["bought"] = sel[ds]["bought"]
    md.append("")
    md.append("(each entry: retrieval-only frontier / graph frontier (links bought in points); `none` = no graph cell is that small)\n")
    md.append("### The ceiling the filed reach blocks set (M3A's populations: the share of questions missing a gold from the union pool, and where the missing golds sit)\n")
    md.append("| dataset | questions | with a gold missing | STRUCT within 1/2/3 hops | NER within 1/2/3 | KNN within 1/2/3 | FULL within 1/2/3 | questions whose missing golds all lie within 3 (FULL) |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        r = reach(ds)
        if not r:
            continue
        g = lambda k: f"{r[k]['w1']:.3f} / {r[k]['w2']:.3f} / {r[k]['w3']:.3f}"  # noqa: E731
        md.append(f"| {ds} | {r['FULL']['questions']} | {r['FULL']['with_missing']} | {g('STRUCT')} | {g('NER')} | {g('KNN')} | {g('FULL')} | {r['FULL']['all3']} of {r['FULL']['with_missing']} |")
        rep.setdefault(ds, {})["reach"] = r
    md.append("")
    cp = PL2 / "call.json"
    if cp.is_file():
        c = rj(cp)
        md.append(f"### The call: {c['call']} (read {len(c['datasets_read'])} of 7: {', '.join(c['datasets_read'])})\n")
        md.append("| dataset | status | NEW | NEW ALL (nodes) | OLD | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |")
        md.append("|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                md.append(f"| {ds} | NOT READ | | | | | | |")
                continue
            md.append(f"| {ds} | {v['status']} | `{v['new']}` | {f4(v['new_all'])} ({f1(v['new_nodes'])}) | `{v['old']}` | {f4(v['old_all'])} ({f1(v['old_nodes'])}) | "
                      f"{pts(v['all_diff'])} [{pts(v['all_diff_ci'][0])}, {pts(v['all_diff_ci'][1])}] | {v['nodes_ratio']:.3f} [{v['nodes_ratio_ci'][0]:.3f}, {v['nodes_ratio_ci'][1]:.3f}] |")
        rep["call"] = {"call": c["call"], "datasets_read": c["datasets_read"], "statuses": {k: v["status"] for k, v in c["datasets"].items()}}
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (PL2 / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
