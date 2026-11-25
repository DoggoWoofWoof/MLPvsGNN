"""PL3 report (docs/PL3_COVERAGE_ROUND2.md section 14): the tables of outputs/pl3/report.md, read from the filed records only, and diagnostic D1.

    python outputs/mp_unified/zpl3r.py report [--out outputs/pl3/report.md]
    python outputs/mp_unified/zpl3r.py miss --ds D [D ...] [--n 200] [--all]       # D1 (docs section 10): labels are read; post-hoc, decides nothing

`report` reads outputs/pl3/<dataset>/{select.json, confirm_frozen.json, confirm_e3v.json}, outputs/pl3/call.json, outputs/pl3/{gr_*.json, selftest.json},
outputs/pl3/<dataset>/miss.json and the filed reachability blocks of outputs/m3a/headroom/<dataset>.json. A dataset without a select.json is listed as
NOT READ. Nothing in `report` chooses a cell or computes a pool: every number is one a record holds. Writes outputs/pl3/report.md and report.json.
`miss` classifies, for a dataset whose chosen cell (R95 over all cells) is SHORT, each gold missing from that cell's pool on the first n dev questions:
  R  in the cell's own walk but not in the pool (ranked past M),
  C  not in the cell's own walk but in the uncapped three-hop FULL reach of its seeds (a cap, the visited cap or the order dropped it),
  U  in neither (the reach ceiling); of which `U_in_lists`: in the first stage's top-1000 dense or SPLADE list (a deeper base would hold it).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PL3 = ROOT / "outputs" / "pl3"
PL2 = ROOT / "outputs" / "pl2"
ORDER = ("multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
BOUND = 2500
TAGS = ("off", "cs", "vr", "csvr", "fs", "seeds", "brg")
TAGNAME = {"off": "deeper base", "cs": "similarity caps", "vr": "rank cap", "csvr": "both", "fs": "FS regime", "seeds": "seeds 10/20", "brg": "bridge"}


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


def short_cell(c):
    return f"{f4(c['all'])} at {f1(c['nodes'])}{'' if c.get('reached', True) else ' SHORT'}"


def reach(ds):
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
        out[rg] = {"questions": v["queries_scanned"], "with_missing": v["queries_with_missing_gold"], "w1": b["missing_golds_reachable_within_1_fraction"],
                   "w2": b["missing_golds_reachable_within_2_fraction"], "w3": b["missing_golds_reachable_within_3_fraction"],
                   "all3": b["queries_with_all_missing_golds_within_3"]}
    return out


def smallest_at(table, floor, bound=None):
    """The smallest cell with dev ALL >= floor (any size unless a bound is given); None when no cell reaches it. Its best otherwise."""
    rows = [x for x in table if bound is None or x["nodes"] <= bound]
    hit = [x for x in rows if x["all"] >= floor]
    if hit:
        m = min(hit, key=lambda x: (x["nodes"], -x["all"], x["name"]))
        return m, True
    return max(rows, key=lambda x: (x["all"], -x["nodes"], x["name"])), False


def best_all(table, bound):
    rows = [x for x in table if bound is None or x["nodes"] <= bound]
    return max(rows, key=lambda x: (x["all"], -x["nodes"], x["name"])) if rows else None


def cmd_report(a):
    rep, md = {}, []
    sel, conf = {}, {}
    for ds in ORDER:
        sp = PL3 / ds / "select.json"
        if sp.is_file():
            sel[ds] = rj(sp)
            conf[ds] = {fs: rj(PL3 / ds / f"confirm_{fs}.json") for fs in ("frozen", "e3v") if (PL3 / ds / f"confirm_{fs}.json").is_file()}
    md.append("### Gates and cost\n")
    md.append("| dataset | G-R (PL2's 804 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |")
    md.append("|---|---|---|")
    for ds in ORDER:
        gp, hp = PL3 / f"gr_{ds}.json", PL3 / f"gr_{ds}_host.json"
        g = rj(gp) if gp.is_file() else (rj(hp) if hp.is_file() else None)
        gtxt = (f"{g['verdict']} ({g['questions']} q x {g['cells']} cells = {g['elements']:,} elements, mismatches {g['present_mismatches']}/{g['size_mismatches']})"
                if g else "not run")
        if ds in sel:
            hs = sel[ds]["grid_heads"]
            ctxt = f"{len(hs)}, {sum(h['seconds'] for h in hs):,.0f} s, {max(h['peak_gb'] for h in hs)} GB"
        else:
            ctxt = "NOT READ"
        md.append(f"| {ds} | {gtxt} | {ctxt} |")
    sp = PL3 / "selftest.json"
    if sp.is_file():
        s = rj(sp)
        md.append(f"\nSelf-test: {s['verdict']} (cells {s['cells']['verdict']}, G-F {s['G-F']['verdict']}, G-D {s['G-D']['verdict']}; engine "
                  f"{', '.join(e['dataset'] + ' ' + e['verdict'] + ' ' + str(e['checks']) for e in s['engine'])}; G-X3 {', '.join(x['verdict'] for x in s['G-X3'])}); "
                  f"code-path sha256 `{s['code_path_sha256'][:8]}...`.\n")
    md.append("### Dev reading (the rule's own numbers, dev carve, frozen lists): PL2's chosen cell against the union grid's\n")
    md.append("| dataset | dev q | PL2 R95 | PL3 R95 (all cells) | PL2 R98 | PL3 R98 | PL2's pick re-read = its filed pick |")
    md.append("|---|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            md.append(f"| {ds} | NOT READ | | | | | |")
            continue
        s, p = sel[ds], sel[ds]["picks"]
        md.append(f"| {ds} | {s['questions']} | {cell(p['R95_pl2'])} | {cell(p['R95_all'])} | {cell(p['R98_pl2'])} | {cell(p['R98_all'])} | {s['pl2_pick_equals_filed']} |")
        rep[ds] = {"dev_questions": s["questions"], "picks": {k: {x: v[x] for x in ("name", "all", "any", "nodes", "reached")} for k, v in p.items() if k.count("_") == 1 and "only" not in k}}
    md.append("")
    for floor in ("95", "98"):
        md.append(f"### The chosen cell at 0.{floor} (R{floor}) and its confirmation on eval: PL2's against the union grid's\n")
        md.append("| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |")
        md.append("|---|---|---|---|---|---|")
        for ds in ORDER:
            if ds not in sel:
                continue
            for rd, lab in (("pl2", "PL2"), ("all", "PL3")):
                c = sel[ds]["picks"][f"R{floor}_{rd}"]
                row = [f"| {ds} {lab} | `{c['name']}`{'' if c['reached'] else ' (SHORT)'} | {f4(c['all'])} | {f1(c['nodes'])}"]
                for fs in ("frozen", "e3v"):
                    cf = conf[ds].get(fs)
                    n = cf["named"].get(f"R{floor}_{rd}") if cf else None
                    row.append(f"{f4(n['eval_all'])} ({f1(n['eval_nodes'])}){'' if n['confirmed'] else ' SHORT'}" if n else "not read")
                md.append(" | ".join(row) + " |")
        md.append("")
    md.append("### The filed own construction on eval (PL2's OLD cell, re-read by this engine)\n")
    md.append("| dataset | OLD | eval ALL frozen (nodes) | eval ALL e3v (nodes) |")
    md.append("|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        row = [f"| {ds} | `{sel[ds]['old']}`"]
        for fs in ("frozen", "e3v"):
            cf = conf[ds].get(fs)
            row.append(f"{f4(cf['old']['eval_all'])} ({f1(cf['old']['eval_nodes'])})" if cf else "not read")
        md.append(" | ".join(row) + " |")
    md.append("")
    md.append("### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL2's 804 cells / the union grid\n")
    md.append("| dataset | 0.95: PL2 | 0.95: PL3 | 0.98: PL2 | 0.98: PL3 |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        tb = sel[ds]["table"]
        pl2 = [x for x in tb if x["family"] == "pl2"]
        cs = []
        for fl in (0.95, 0.98):
            for rows in (pl2, tb):
                m, ok = smallest_at(rows, fl)
                cs.append(f"`{m['name']}` {f4(m['all'])} at {f1(m['nodes'])}" if ok else f"none (best {f4(m['all'])} at {f1(m['nodes'])})")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
        rep.setdefault(ds, {})["smallest_any_size"] = {"pl2_95": cs[0], "all_95": cs[1], "pl2_98": cs[2], "all_98": cs[3]}
    md.append("")
    md.append("### The highest ALL on dev (the bound is 2,500 mean nodes): PL2's 804 cells / the union grid, within the bound and at any size\n")
    md.append("| dataset | PL2, within the bound | PL3, within the bound | PL2, any size | PL3, any size |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        tb = sel[ds]["table"]
        pl2 = [x for x in tb if x["family"] == "pl2"]
        cs = []
        for bd in (BOUND, None):
            for rows in (pl2, tb):
                b = best_all(rows, bd)
                cs.append(f"`{b['name']}` {f4(b['all'])} at {f1(b['nodes'])}")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
    md.append("")
    md.append("### What PL3's cells bought (dev frontier of the union grid minus PL2's frontier, at equal mean nodes; ALL)\n")
    grid_n = (25, 50, 75, 100, 150, 200, 300, 400, 500, 750, 1000, 1500, 2000, 2500)
    md.append("| dataset | " + " | ".join(str(n) for n in grid_n) + " |")
    md.append("|---|" + "---|" * len(grid_n))
    for ds in ORDER:
        if ds not in sel:
            continue
        b = {x["nodes"]: x for x in sel[ds]["bought"]}
        cells_ = []
        for n in grid_n:
            x = b.get(n)
            cells_.append("-" if not x or x["all"] is None else f"{f4(x['pl2'])} / {f4(x['all'])} ({pts(x['gain'])})")
        md.append(f"| {ds} | " + " | ".join(cells_) + " |")
        rep.setdefault(ds, {})["bought"] = sel[ds]["bought"]
    md.append("\n(each entry: PL2's frontier / the union grid's frontier (points bought); `-` = no cell is that small)\n")
    md.append("### What each family bought: R95 of PL2's cells plus that family's cells (dev), and the best ALL inside the bound among that family's cells alone\n")
    md.append("| dataset | PL2 R95 | " + " | ".join(TAGNAME[t] for t in TAGS) + " |")
    md.append("|---|---|" + "---|" * len(TAGS))
    for ds in ORDER:
        if ds not in sel:
            continue
        p = sel[ds]["picks"]
        row = [f"| {ds} | {short_cell(p['R95_pl2'])}"]
        for t in TAGS:
            c = p.get(f"R95_pl2+{t}")
            o = best_all([x for x in sel[ds]["table"] if x["family"] == "new" and t in x["tags"]], BOUND)
            if not c:
                row.append("-")
                continue
            same = c["name"] == p["R95_pl2"]["name"]
            row.append(("= PL2" if same else short_cell(c)) + (f"; alone best {f4(o['all'])} at {f1(o['nodes'])}" if o else ""))
        md.append(" | ".join(row) + " |")
    md.append("")
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
    cp = PL3 / "call.json"
    if cp.is_file():
        c = rj(cp)
        md.append(f"### The call: {c['call']} (read {len(c['datasets_read'])} of 7: {', '.join(c['datasets_read'])})\n")
        md.append("| dataset | status | NEW (PL3 R95) | NEW ALL (nodes) | OLD (PL2 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |")
        md.append("|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                md.append(f"| {ds} | NOT READ | | | | | | |")
                continue
            md.append(f"| {ds} | {v['status']}{' (same cell)' if v['same_cell'] else ''} | `{v['new']}` | {f4(v['new_all'])} ({f1(v['new_nodes'])}) | `{v['old']}` | "
                      f"{f4(v['old_all'])} ({f1(v['old_nodes'])}) | {pts(v['all_diff'])} [{pts(v['all_diff_ci'][0])}, {pts(v['all_diff_ci'][1])}] | "
                      f"{v['nodes_ratio']:.3f} [{v['nodes_ratio_ci'][0]:.3f}, {v['nodes_ratio_ci'][1]:.3f}] |")
        md.append("\nBeside the call (decide nothing): R98 and the `new`-only pick against PL2's R95.\n")
        md.append("| dataset | R98 status | R98 NEW ALL (nodes) vs PL2 | new-only status | new-only cell | ALL (nodes) |")
        md.append("|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                continue
            r98, no = v["r98"], v["new_only"]
            md.append(f"| {ds} | {r98['status']} | {f4(r98['new_all'])} ({f1(r98['new_nodes'])}) vs {f4(r98['old_all'])} ({f1(r98['old_nodes'])}) | {no['status']} | "
                      f"`{no['new']}` | {f4(no['new_all'])} ({f1(no['new_nodes'])}) |")
        rep["call"] = {"call": c["call"], "datasets_read": c["datasets_read"], "statuses": {k: v["status"] for k, v in c["datasets"].items()}}
        md.append("")
    mrows = [(ds, rj(PL3 / ds / "miss.json")) for ds in ORDER if (PL3 / ds / "miss.json").is_file()]
    if mrows:
        md.append("### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; the first n dev questions)\n")
        md.append("| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, m in mrows:
            t = m["totals"]
            md.append(f"| {ds} | `{m['cell']}` | {m['questions']} | {t['q_missing']} ({t['q_missing'] / m['questions']:.3f}) | {t['golds_missing']} | {t['golds_missing'] / max(1, t['q_missing']):.2f} | "
                      f"{t['R']} (q {t['q_R']}) | {t['C']} (q {t['q_C']}) | {t['U']} (q {t['q_U']}) | {t['U_in_lists']} | {f1(m['nodes_mean'])} |")
        md.append("")
        rep["D1"] = {ds: m["totals"] for ds, m in mrows}
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (PL3 / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


def cmd_miss(a):
    for _p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import numpy as np
    import zpl3g as Z
    P2 = Z.P2
    P2.HOST = False
    op = P2.opened()
    M3A = op.m3a
    unc = {"hops": 3, "per_seed_cap": 10 ** 9, "per_frontier_cap": 10 ** 9, "visited_cap": 10 ** 12}
    for ds in a.ds:
        sel = P2.rjson(Z.PL3 / ds / "select.json")
        pk = sel["picks"]["R95_all"]
        name = pk["name"]
        if pk["reached"] and not a.all:
            Z.log(f"{ds}: R95 reaches 0.95 ({name}); D1 is for the SHORT datasets (use --all to run anyway)")
            continue
        p = Z.parse3(name)
        if p[0] not in ("G", "K"):
            Z.log(f"{ds}: the chosen cell {name} has no walk; nothing to classify")
            continue
        src = P2.load_source(ds, "dev", "frozen")
        plan = Z.Plan3([name])
        n = min(a.n, src.nq)
        regime, stg, mods = p[3]
        fam_all = [src.fams[f] for f in P2.REGIMES["FULL"]]
        tot = {k: 0 for k in ("q_missing", "golds_missing", "R", "C", "U", "U_in_lists", "q_R", "q_C", "q_U")}
        nodes = []
        t0 = Z.time.time()
        for q, st in Z.iter_states3(op, src, plan, 0, n):
            wk = st.W[p[3]]
            exp = wk.exp
            lst = st.L[p[1]]
            base = lst.L[:p[2]]
            if p[0] == "G":
                pool = np.union1d(base, exp)
            else:
                ordered = exp[wk.ro[p[5]]] if exp.size else exp
                cand = ordered[lst.pos(ordered) >= p[2]] if exp.size else exp
                pool = np.union1d(base, cand[:p[4]])
            nodes.append(int(pool.size))
            g = src.golds[q]
            miss = g[~np.isin(g, pool)]
            if miss.size == 0:
                continue
            d = src.dense[q].astype(np.int64)
            s = src.splade[q].astype(np.int64)
            seeds5 = M3A.stable_union_rows(d[None, :5], s[None, :5])[0]
            f1_ = M3A.rrf_fused_rows(d[None, :], s[None, :], P2.RRF_CONST)[0]
            sd = np.asarray(f1_[:10] if "s10" in mods else (f1_[:20] if "s20" in mods else seeds5), dtype=np.int64)
            reach_ = np.union1d(sd, Z.expand3(sd, fam_all, fam_all, unc, False, False, None))
            in_exp = np.isin(miss, exp) if exp.size else np.zeros(miss.size, bool)
            in_reach = np.isin(miss, reach_)
            r_ = in_exp
            c_ = (~in_exp) & in_reach
            u_ = (~in_exp) & (~in_reach)
            lists = np.union1d(d[:1000], s[:1000])
            tot["q_missing"] += 1
            tot["golds_missing"] += int(miss.size)
            tot["R"] += int(r_.sum())
            tot["C"] += int(c_.sum())
            tot["U"] += int(u_.sum())
            tot["U_in_lists"] += int(np.isin(miss[u_], lists).sum())
            tot["q_R"] += int(r_.any())
            tot["q_C"] += int(c_.any())
            tot["q_U"] += int(u_.any())
        out = {"stage": "PL3 D1 (post-hoc diagnostic, labels read, decides nothing)", "declared_in": Z.DOC, "dataset": ds, "cell": name, "questions": n,
               "carve": "dev", "first": "frozen", "nodes_mean": float(np.mean(nodes)), "bound": BOUND, "totals": tot,
               "uncapped_setting": "3 hops, no per-seed, per-frontier or visited cap, FULL families, the cell's own seeds", "seconds": round(Z.time.time() - t0, 1),
               "code_path_sha256": Z.code_path_sha256(), "utc": Z.utc()}
        Z.wjson(Z.PL3 / ds / "miss.json", out)
        Z.log(f"D1 {ds} {name}: {n} q, {tot['q_missing']} miss a gold ({tot['golds_missing']} golds): R {tot['R']}, C {tot['C']}, U {tot['U']} (in lists {tot['U_in_lists']}) "
              f"at {np.mean(nodes):.0f} nodes")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(PL3 / "report.md"))
    p = sub.add_parser("miss")
    p.add_argument("--ds", nargs="+", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--all", action="store_true")
    a = ap.parse_args(argv)
    return {"report": cmd_report, "miss": cmd_miss}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
