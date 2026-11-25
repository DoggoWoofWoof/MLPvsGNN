"""PL4 report (docs/PL4_COVERAGE_ROUND3.md sections 10 and 14): the tables of outputs/pl4/report.md, read from the filed records only, and diagnostics D1 and D2.

    python outputs/mp_unified/zpl4r.py report [--out outputs/pl4/report.md]
    python outputs/mp_unified/zpl4r.py miss --ds D [D ...] [--n 300] [--all]       # D1 + D2 (docs section 10): labels are read; post-hoc, decides nothing

`report` reads outputs/pl4/<dataset>/{select.json, confirm_frozen.json, confirm_e3v.json, miss.json}, outputs/pl4/call.json, outputs/pl4/{gr_*.json, selftest.json}
and the filed reachability blocks of outputs/m3a/headroom/<dataset>.json. A dataset without a select.json is listed as NOT READ. Nothing in `report` chooses a cell
or computes a pool: every number is one a record holds. Writes outputs/pl4/report.md and report.json.
`miss` classifies, for a dataset whose chosen cell (R95 of the reading `all`) is SHORT, each gold missing from that cell's pool on n dev questions evenly spaced over the carve:
  D1  R  in the cell's own walk but not in the pool (ranked past M),
      C  not in the cell's own walk but in the uncapped three-hop FULL reach of its seeds,
      U  in neither (the reach ceiling); of which `U_in_lists`: in the first stage's top-1000 dense or SPLADE list.
  D2  for the C golds, from a traced walk of the cell (the trace is checked against the engine's own walk):
      cap       a neighbour of a hop's frontier that the per-seed / per-frontier cap dropped,
      budget    in the hop's capped entries but not kept (the visited cap, the hop's share, or the order),
      upstream  neither: no node the walk kept at any hop has it as a neighbour (a dropped node's neighbour, or more hops than the cell walks).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PL4 = ROOT / "outputs" / "pl4"
ORDER = ("multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp")
BOUND = 2500
BUDGETS = (500, 1000, 1500, 2000, 2500)
TAGS = ("sp", "rk", "eq", "rr", "seeds")
TAGNAME = {"sp": "support order", "rk": "row-rank order", "eq": "equal hop shares", "rr": "families take turns", "seeds": "seeds 10/20 with a new option"}


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
    """The smallest cell with dev ALL >= floor (any size unless a bound is given); its best otherwise."""
    rows = [x for x in table if bound is None or x["nodes"] <= bound]
    hit = [x for x in rows if x["all"] >= floor]
    if hit:
        return min(hit, key=lambda x: (x["nodes"], -x["all"], x["name"])), True
    return max(rows, key=lambda x: (x["all"], -x["nodes"], x["name"])), False


def best_all(table, bound):
    rows = [x for x in table if bound is None or x["nodes"] <= bound]
    return max(rows, key=lambda x: (x["all"], -x["nodes"], x["name"])) if rows else None


def cmd_report(a):
    rep, md = {}, []
    sel, conf = {}, {}
    for ds in ORDER:
        sp = PL4 / ds / "select.json"
        if sp.is_file():
            sel[ds] = rj(sp)
            conf[ds] = {fs: rj(PL4 / ds / f"confirm_{fs}.json") for fs in ("frozen", "e3v") if (PL4 / ds / f"confirm_{fs}.json").is_file()}
    md.append("### Gates and cost\n")
    md.append("| dataset | G-R (PL2's 804 + PL3's 1,470 cells rebuilt, first 400 dev questions) | dev grid: parts, seconds (sum), peak GB (max) |")
    md.append("|---|---|---|")
    for ds in ORDER:
        gp, hp = PL4 / f"gr_{ds}.json", PL4 / f"gr_{ds}_host.json"
        g = rj(gp) if gp.is_file() else (rj(hp) if hp.is_file() else None)
        gtxt = (f"{g['verdict']} ({g['questions']} q x {g['cells']} cells = {g['elements']:,} elements, mismatches {g['present_mismatches']}/{g['size_mismatches']})"
                if g else "not run")
        if ds in sel:
            hs = sel[ds]["grid_heads"]
            ctxt = f"{len(hs)}, {sum(h['seconds'] for h in hs):,.0f} s, {max(h['peak_gb'] for h in hs)} GB"
        else:
            ctxt = "NOT READ"
        md.append(f"| {ds} | {gtxt} | {ctxt} |")
    sp = PL4 / "selftest.json"
    if sp.is_file():
        s = rj(sp)
        md.append(f"\nSelf-test: {s['verdict']} (cells {s['cells']['verdict']}, G-F {s['G-F']['verdict']}, G-D {s['G-D']['verdict']}; engine "
                  f"{', '.join(e['dataset'] + ' ' + e['verdict'] + ' ' + str(e['checks']) for e in s['engine'])}; G-X4 {', '.join(x['verdict'] for x in s['G-X4'])}); "
                  f"code-path sha256 `{s['code_path_sha256'][:8]}...`.\n")
    md.append("### Dev reading (the rule's own numbers, dev carve, frozen lists): PL3's chosen cell (the incumbent) against the grid of all 4,206 cells\n")
    md.append("| dataset | dev q | PL3 R95 (PL2 U PL3 cells) | R95 (all cells) | PL3 R98 | R98 (all cells) | R95 of the new cells only | PL3's pick re-read = its filed pick |")
    md.append("|---|---|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            md.append(f"| {ds} | NOT READ | | | | | | |")
            continue
        s, p = sel[ds], sel[ds]["picks"]
        md.append(f"| {ds} | {s['questions']} | {cell(p['R95_pl3'])} | {cell(p['R95_all'])} | {cell(p['R98_pl3'])} | {cell(p['R98_all'])} | {cell(p['R95_new'])} | {s['pl3_pick_equals_filed']} |")
        rep[ds] = {"dev_questions": s["questions"], "picks": {k: {x: v[x] for x in ("name", "all", "any", "nodes", "reached")} for k, v in p.items() if k.count("_") == 1}}
    md.append("")
    for floor in ("95", "98"):
        md.append(f"### The chosen cell at 0.{floor} (R{floor}) and its confirmation on eval: PL3's against the grid of all cells (and the new cells alone)\n")
        md.append("| dataset | cell | dev ALL | dev nodes | eval ALL frozen (nodes) | eval ALL e3v (nodes) |")
        md.append("|---|---|---|---|---|---|")
        for ds in ORDER:
            if ds not in sel:
                continue
            for rd, lab in (("pl3", "PL3"), ("all", "all"), ("new", "new only")):
                c = sel[ds]["picks"].get(f"R{floor}_{rd}")
                if not c:
                    continue
                row = [f"| {ds} {lab} | `{c['name']}`{'' if c['reached'] else ' (SHORT)'} | {f4(c['all'])} | {f1(c['nodes'])}"]
                for fs in ("frozen", "e3v"):
                    cf = conf[ds].get(fs)
                    n = cf["named"].get(f"R{floor}_{rd}") if cf else None
                    row.append(f"{f4(n['eval_all'])} ({f1(n['eval_nodes'])}){'' if n['confirmed'] else ' SHORT'}{' OVER_BOUND' if n.get('over_bound') else ''}" if n else "not read")
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
    md.append("### The smallest cell at 0.95 and at 0.98 on dev, at any size (the bound is not applied here): PL3's grid / all cells\n")
    md.append("| dataset | 0.95: PL3's grid | 0.95: all cells | 0.98: PL3's grid | 0.98: all cells |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        tb = sel[ds]["table"]
        old = [x for x in tb if x["family"] in ("pl2", "pl3")]
        cs = []
        for fl in (0.95, 0.98):
            for rows in (old, tb):
                m, ok = smallest_at(rows, fl)
                cs.append(f"`{m['name']}` {f4(m['all'])} at {f1(m['nodes'])}" if ok else f"none (best {f4(m['all'])} at {f1(m['nodes'])})")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
        rep.setdefault(ds, {})["smallest_any_size"] = {"pl3_95": cs[0], "all_95": cs[1], "pl3_98": cs[2], "all_98": cs[3]}
    md.append("")
    md.append("### The highest ALL on dev (the bound is 2,500 mean nodes): PL3's grid / all cells, within the bound and at any size\n")
    md.append("| dataset | PL3's grid, within the bound | all cells, within the bound | PL3's grid, any size | all cells, any size |")
    md.append("|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        tb = sel[ds]["table"]
        old = [x for x in tb if x["family"] in ("pl2", "pl3")]
        cs = []
        for bd in (BOUND, None):
            for rows in (old, tb):
                b = best_all(rows, bd)
                cs.append(f"`{b['name']}` {f4(b['all'])} at {f1(b['nodes'])}")
        md.append(f"| {ds} | {cs[0]} | {cs[1]} | {cs[2]} | {cs[3]} |")
    md.append("")
    md.append("### What the PL4 cells bought (dev frontier of all cells minus PL3's frontier, at equal mean nodes; ALL)\n")
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
            cells_.append("-" if not x or x["all"] is None else f"{f4(x['pl3'])} / {f4(x['all'])} ({pts(x['gain'])})")
        md.append(f"| {ds} | " + " | ".join(cells_) + " |")
        rep.setdefault(ds, {})["bought"] = sel[ds]["bought"]
    md.append("\n(each entry: PL3's frontier / the frontier of all cells (points bought); `-` = no cell is that small)\n")
    md.append("### What each option bought: R95 of PL3's cells plus that option's cells (dev), and the best ALL inside the bound among that option's cells alone\n")
    md.append("| dataset | PL3 R95 | " + " | ".join(TAGNAME[t] for t in TAGS) + " |")
    md.append("|---|---|" + "---|" * len(TAGS))
    for ds in ORDER:
        if ds not in sel:
            continue
        p = sel[ds]["picks"]
        row = [f"| {ds} | {short_cell(p['R95_pl3'])}"]
        for t in TAGS:
            c = p.get(f"R95_pl3+{t}")
            o = best_all([x for x in sel[ds]["table"] if x["family"] == "pl4" and t in x["tags"]], BOUND)
            if not c:
                row.append("-")
                continue
            same = c["name"] == p["R95_pl3"]["name"]
            row.append(("= PL3" if same else short_cell(c)) + (f"; alone best {f4(o['all'])} at {f1(o['nodes'])}" if o else ""))
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
    cp = PL4 / "call.json"
    if cp.is_file():
        c = rj(cp)
        md.append(f"### The call, the old rule as PL3 states it (R95 of all cells against PL3's R95): {c['call']} (read {len(c['datasets_read'])} of 7: {', '.join(c['datasets_read'])}; "
                  f"BELOW {c['counts']['old']['BELOW']}, ABOVE {c['counts']['old']['ABOVE']}, LEVEL {c['counts']['old']['LEVEL']})\n")
        md.append("| dataset | status | NEW (R95, all cells) | NEW ALL (nodes) | OLD (PL3 R95) | OLD ALL (nodes) | ALL diff, 95% CI | nodes ratio, 95% CI |")
        md.append("|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                md.append(f"| {ds} | NOT READ | | | | | | |")
                continue
            md.append(f"| {ds} | {v['status']}{' (same cell)' if v['same_cell'] else ''} | `{v['new']}` | {f4(v['new_all'])} ({f1(v['new_nodes'])}) | `{v['old']}` | "
                      f"{f4(v['old_all'])} ({f1(v['old_nodes'])}) | {pts(v['all_diff'])} [{pts(v['all_diff_ci'][0])}, {pts(v['all_diff_ci'][1])}] | "
                      f"{v['nodes_ratio']:.3f} [{v['nodes_ratio_ci'][0]:.3f}, {v['nodes_ratio_ci'][1]:.3f}] |")
        md.append("\nBeside the call (decide nothing): R98 and the new-cells-only pick against PL3's R95.\n")
        md.append("| dataset | R98 status | R98 NEW ALL (nodes) vs PL3 | new-only status | new-only cell | ALL (nodes) |")
        md.append("|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                continue
            r98, no = v["r98"], v["new_only"]
            md.append(f"| {ds} | {r98['status']} | {f4(r98['new_all'])} ({f1(r98['new_nodes'])}) vs {f4(r98['old_all'])} ({f1(r98['old_nodes'])}) | {no['status']} | "
                      f"`{no['new']}` | {f4(no['new_all'])} ({f1(no['new_nodes'])}) |")
        md.append("")
        md.append(f"### The call, the frontier reading (matched node counts, proposed on the page before any number): {c['call_frontier']} "
                  f"(BELOW {c['counts']['frontier']['BELOW']}, ABOVE {c['counts']['frontier']['ABOVE']}, LEVEL {c['counts']['frontier']['LEVEL']})\n")
        md.append("Per dataset and budget N: the cell of all cells with the highest dev ALL at most N dev nodes against the same for PL3's grid; eval, frozen lists; paired bootstrap of the ALL difference.\n")
        md.append("| dataset | frontier status | N | NEW cell | OLD cell | NEW ALL (nodes) | OLD ALL (nodes) | ALL diff, 95% CI | status |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for ds in ORDER:
            v = c["datasets"].get(ds)
            if not v or v["status"] == "NOT_READ":
                md.append(f"| {ds} | NOT READ | | | | | | | |")
                continue
            for n in BUDGETS:
                fr = v["frontier"].get(str(n))
                if not fr or fr["status"] == "NONE":
                    md.append(f"| {ds} | {v['frontier_status']} | {n} | none | | | | | |")
                    continue
                md.append(f"| {ds} | {v['frontier_status']} | {n} | `{fr['new']}`{' (same cell)' if fr['same_cell'] else ''} | `{fr['old']}` | {f4(fr['new_all'])} ({f1(fr['new_nodes'])}) | "
                          f"{f4(fr['old_all'])} ({f1(fr['old_nodes'])}) | {pts(fr['all_diff'])} [{pts(fr['all_diff_ci'][0])}, {pts(fr['all_diff_ci'][1])}] | {fr['status']} |")
        md.append("")
        rep["call"] = {"call": c["call"], "call_frontier": c["call_frontier"], "datasets_read": c["datasets_read"], "counts": c["counts"],
                       "statuses": {k: v["status"] for k, v in c["datasets"].items()}, "frontier_statuses": {k: v.get("frontier_status") for k, v in c["datasets"].items()}}
    md.append("### The 2,500 bound on e3v lists (selection reads the frozen dev lists; this is the same cell re-run on the eval carve's e3v lists)\n")
    md.append("| dataset | cell (R95, all cells) | nodes: dev frozen | eval frozen | eval e3v | e3v over the bound |")
    md.append("|---|---|---|---|---|---|")
    for ds in ORDER:
        if ds not in sel:
            continue
        c = sel[ds]["picks"]["R95_all"]
        nf = conf[ds].get("frozen", {}).get("named", {}).get("R95_all")
        ne = conf[ds].get("e3v", {}).get("named", {}).get("R95_all")
        md.append(f"| {ds} | `{c['name']}` | {f1(c['nodes'])} | {f1(nf['eval_nodes']) if nf else 'not read'} | {f1(ne['eval_nodes']) if ne else 'not read'} | "
                  f"{('OVER_BOUND' if ne['over_bound'] else 'within') if ne else '-'} |")
    md.append("")
    mrows = [(ds, rj(PL4 / ds / "miss.json")) for ds in ORDER if (PL4 / ds / "miss.json").is_file()]
    if mrows:
        md.append("### D1: what remains unreached at the chosen cell of a SHORT dataset (post-hoc; labels read; n dev questions evenly spaced over the carve)\n")
        md.append("| dataset | chosen cell | n | questions missing a gold | missing golds | per question | R (ranked out) | C (cut by the caps) | U (unreached) | of U: in the top-1000 lists | nodes |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for ds, m in mrows:
            t = m["totals"]
            md.append(f"| {ds} | `{m['cell']}` | {m['questions']} | {t['q_missing']} ({t['q_missing'] / m['questions']:.3f}) | {t['golds_missing']} | {t['golds_missing'] / max(1, t['q_missing']):.2f} | "
                      f"{t['R']} (q {t['q_R']}) | {t['C']} (q {t['q_C']}) | {t['U']} (q {t['q_U']}) | {t['U_in_lists']} | {f1(m['nodes_mean'])} |")
        md.append("")
        md.append("### D2: which stage cut the C golds (post-hoc; labels read; the cell's walk traced hop by hop)\n")
        md.append("| dataset | cell | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for ds, m in mrows:
            d2 = m.get("D2")
            if not d2:
                continue
            h = d2["by_stage_hop"]
            md.append(f"| {ds} | `{m['cell']}` | {m['totals']['C']} | {h.get('cap1', 0)} | {h.get('cap2', 0)} | {h.get('cap3', 0)} | {h.get('budget2', 0)} | {h.get('budget3', 0)} | "
                      f"{h.get('upstream', 0)} | {d2['trace_matches_engine']} |")
        md.append("")
        rep["D1"] = {ds: m["totals"] for ds, m in mrows}
        rep["D2"] = {ds: m.get("D2") for ds, m in mrows}
    Path(a.out).write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    (PL4 / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8", newline="\n")
    print("\n".join(md))
    return 0


def uncapped_neighbours(np, nodes, families):
    """The distinct neighbours of `nodes` in `families`, rows read whole (no cap, no window)."""
    out = []
    for fam in families:
        starts = fam.indptr[nodes]
        lens = fam.indptr[nodes + 1] - starts
        total = int(lens.sum())
        if total == 0:
            continue
        off = np.cumsum(lens) - lens
        within = np.arange(total, dtype=np.int64) - np.repeat(off, lens)
        out.append(fam.col[np.repeat(starts, lens) + within].astype(np.int64, copy=False))
    return np.unique(np.concatenate(out)) if out else np.empty(0, dtype=np.int64)


def trace_walk(np, Z4, seeds, fams1, fams2, setting, mods, scorer):
    """expand4's loop, recording at every hop its frontier, its family list, the capped entries and the nodes it kept (the trace is checked against expand4's walk)."""
    first = Z4.m3b_pools._first_occurrence_unique
    cs = "cs" in mods
    key = "vr" if "vr" in mods else ("sp" if "sp" in mods else ("rk" if "rk" in mods else ""))
    rr, eq = "rr" in mods, "eq" in mods
    cap = int(setting["per_seed_cap"])
    hop1 = first(Z4.capped_parts(seeds, fams1, cap, scorer, cs)[0])
    hops = [{"h": 1, "frontier": seeds, "fams": fams1, "capped": np.unique(Z4.capped_parts(seeds, fams1, cap, scorer, cs)[0]), "kept": hop1}]
    if int(setting["hops"]) == 1:
        return hops
    cap2, limit = int(setting["per_frontier_cap"]), int(setting["visited_cap"])
    visited = np.union1d(seeds, hop1)
    frontier = hop1
    for h in range(2, int(setting["hops"]) + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = -(-rem // (int(setting["hops"]) - h + 1)) if eq else rem
        cols, grp, fam, rank = Z4.capped_parts(frontier, fams2, cap2, scorer, cs)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fresh = Z4.order_fresh(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))[:budget]
        hops.append({"h": h, "frontier": frontier, "fams": fams2, "capped": np.unique(cols[k]), "kept": fresh})
        if fresh.size == 0:
            break
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return hops


def cmd_miss(a):
    for _p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import numpy as np
    import zpl4g as Z4
    Z3, P2 = Z4.Z3, Z4.P2
    P2.HOST = bool(a.host)
    op = P2.opened()
    M3A = op.m3a
    unc = {"hops": 3, "per_seed_cap": 10 ** 9, "per_frontier_cap": 10 ** 9, "visited_cap": 10 ** 12}
    for ds in a.ds:
        sel = Z4.rjson(PL4 / ds / "select.json")
        pk = sel["picks"]["R95_all"]
        name = pk["name"]
        if pk["reached"] and not a.all:
            Z4.log(f"{ds}: R95 reaches 0.95 ({name}); D1 is for the SHORT datasets (use --all to run anyway)")
            continue
        p = Z4.parse4(name)
        if p[0] not in ("G", "K"):
            Z4.log(f"{ds}: the chosen cell {name} has no walk; nothing to classify")
            continue
        src = P2.load_source(ds, "dev", "frozen")
        plan = Z4.Plan4([name])
        n = min(a.n, src.nq)
        regime, stg, mods = p[3]
        h1, h2 = Z4.REG3[regime]
        fam_all = [src.fams[f] for f in P2.REGIMES["FULL"]]
        tot = {k: 0 for k in ("q_missing", "golds_missing", "R", "C", "U", "U_in_lists", "q_R", "q_C", "q_U")}
        stage = {}
        mismatch = 0
        nodes = []
        t0 = Z4.time.time()
        idx = np.unique(np.linspace(0, src.nq - 1, n).astype(np.int64))      # evenly spaced over the carve (a prefix is not representative: MetaQA's is ordered by hop count)

        def sample():
            for q0 in idx:
                yield from Z4.iter_states4(op, src, plan, int(q0), int(q0) + 1)
        for q, st in sample():
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
            reach_ = np.union1d(sd, Z3.expand3(sd, fam_all, fam_all, unc, False, False, None))
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
            if c_.any():
                scorer = Z3.Scorer(src.dvec, {"q": np.asarray(src.qvec(q, q + 1)[0], dtype=np.float64)}, src.n)
                trace = trace_walk(np, Z4, sd, [src.fams[f] for f in h1], [src.fams[f] for f in h2], P2.SETTINGS[stg], mods, scorer)
                scorer.release()
                if not np.array_equal(np.concatenate([t["kept"] for t in trace]), exp):
                    mismatch += 1
                for gold in miss[c_]:
                    gold = int(gold)
                    budget_h = next((t["h"] for t in trace if t["h"] > 1 and gold in t["capped"] and gold not in t["kept"]), None)
                    cap_h = None
                    if budget_h is None:
                        for t in trace:
                            if gold not in t["capped"] and gold in uncapped_neighbours(np, t["frontier"], t["fams"]):
                                cap_h = t["h"]
                                break
                    key = f"budget{budget_h}" if budget_h else (f"cap{cap_h}" if cap_h else "upstream")
                    stage[key] = stage.get(key, 0) + 1
        out = {"stage": "PL4 D1 + D2 (post-hoc diagnostic, labels read, decides nothing)", "declared_in": Z4.DOC, "dataset": ds, "cell": name, "questions": int(idx.size),
               "carve": "dev", "first": "frozen", "nodes_mean": float(np.mean(nodes)), "bound": BOUND, "totals": tot,
               "D2": {"by_stage_hop": stage, "c_golds": tot["C"], "trace_matches_engine": mismatch == 0, "trace_mismatch_questions": mismatch},
               "uncapped_setting": "3 hops, no per-seed, per-frontier or visited cap, FULL families, the cell's own seeds", "seconds": round(Z4.time.time() - t0, 1),
               "code_path_sha256": Z4.code_path_sha256(), "utc": Z4.utc()}
        Z4.wjson(PL4 / ds / "miss.json", out)
        Z4.log(f"D1 {ds} {name}: {idx.size} q, {tot['q_missing']} miss a gold ({tot['golds_missing']} golds): R {tot['R']}, C {tot['C']}, U {tot['U']} (in lists {tot['U_in_lists']}) "
               f"at {np.mean(nodes):.0f} nodes; D2 {stage} (trace = engine: {mismatch == 0})")
    return 0


def cmd_confirm_par(a):
    """zpl4g.py confirm, with the eval questions cut into K slices run in K processes (the same pools, the same arrays, the same records: the slices are
    concatenated before anything is graded into a record). Used where the single-process confirm would take hours (SQuAD's 11,873 eval questions).
    --verify re-computes and compares the arrays with the existing confirm grid instead of writing."""
    for _p in (ROOT / "scripts", ROOT / "src", HERE):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    import subprocess
    import numpy as np
    import zpl4g as Z4
    P2 = Z4.P2
    P2.HOST = bool(a.host)
    Z4.gates_passed4(a.ds)
    sel = Z4.rjson(PL4 / a.ds / "select.json")
    names = Z4.confirm_names4(sel)
    plan = Z4.Plan4(names)
    src = P2.load_source(a.ds, "eval", a.first)
    nq, k = src.nq, int(a.slices)
    tmp = PL4 / a.ds / f"_confirm_par_{a.first}"
    if a.worker is not None:
        i = int(a.worker)
        op = P2.opened()
        if plan.need_vec:
            src.dvec(np.asarray([0]))
        lo, hi = i * nq // k, (i + 1) * nq // k
        Z4.log(f"{a.ds}/eval/{a.first}: slice {i} of {k}, rows {lo}..{hi}")
        P, S = Z4.evaluate4(op, src, plan, lo, hi)
        tmp.mkdir(parents=True, exist_ok=True)
        np.savez(tmp / f"slice{i}of{k}.npz", P=P, S=S, lo=lo, hi=hi)
        return 0
    t0 = Z4.time.time()
    tmp.mkdir(parents=True, exist_ok=True)
    procs = []
    for i in range(k):
        cmd = [sys.executable, str(Path(__file__).resolve()), "confirm-par", "--ds", a.ds, "--first", a.first, "--slices", str(k), "--worker", str(i)]
        if a.host:
            cmd.append("--host")
        procs.append(subprocess.Popen(cmd, stdout=open(tmp / f"slice{i}.log", "w"), stderr=subprocess.STDOUT))
    rcs = [p.wait() for p in procs]
    if any(rcs):
        Z4.stop(f"{a.ds}/{a.first}: a slice failed ({rcs}); see {tmp}")
    P = np.concatenate([np.load(tmp / f"slice{i}of{k}.npz")["P"] for i in range(k)], axis=0)
    S = np.concatenate([np.load(tmp / f"slice{i}of{k}.npz")["S"] for i in range(k)], axis=0)
    base = Z4.grid_path4(a.ds, "eval", a.first, "confirm")
    if a.verify:
        g0 = Z4.load_grid4(base)
        same = bool(np.array_equal(g0["present"], P) and np.array_equal(g0["size"], S) and list(g0["names"]) == list(plan.names))
        Z4.log(f"{a.ds}/{a.first}: sliced run equals the filed confirm grid: {same}")
        return 0 if same else 3
    Z4.save_grid4(base, src, plan, P, S, 0, src.nq, Z4.time.time() - t0, stage="PL4 confirm grid")
    g = {"names": plan.names, "present": P, "size": S, "gold_counts": src.gold_counts}
    rows = {r["pool"]: r for r in Z4.rows_of_grid4(g, a.ds, "eval")}
    res = {"stage": "PL4 confirm", "declared_in": Z4.DOC, "dataset": a.ds, "first": a.first, "questions": src.nq, "utc": Z4.utc(), "slices": k, "cells": {}}
    for n in plan.names:
        r = rows[n]
        kk = int(round(r["all_gold_at_pool"] * src.nq))
        res["cells"][n] = {"all": r["all_gold_at_pool"], "all_ci": P2.wilson(kk, src.nq), "any": r["any_gold_at_pool"], "coverage": r["coverage_micro"],
                           "nodes": r["candidates_mean"], "p95": r["candidates_p95"], "max": r["candidates_max"], "over_bound": bool(r["candidates_mean"] > BOUND)}
    res["named"] = {}
    for key, c in sel["picks"].items():
        if key.split("_", 1)[1] in Z4.READINGS and c["name"] in res["cells"]:
            e = res["cells"][c["name"]]
            res["named"][key] = {"name": c["name"], "dev_all": c["all"], "dev_nodes": c["nodes"], "eval_all": e["all"], "eval_nodes": e["nodes"], "floor": c["floor"],
                                 "confirmed": bool(e["all"] >= c["floor"]), "short_by": max(0.0, c["floor"] - e["all"]), "dev_to_eval": e["all"] - c["all"],
                                 "over_bound": e["over_bound"]}
    e = res["cells"][sel["old"]]
    res["old"] = {"name": sel["old"], "eval_all": e["all"], "eval_nodes": e["nodes"]}
    Z4.wjson(PL4 / a.ds / f"confirm_{a.first}.json", res)
    for key, v in res["named"].items():
        Z4.log(f"{a.ds}/{a.first} {key}: {v['name']} eval ALL {v['eval_all']:.4f} at {v['eval_nodes']:.1f} nodes "
               f"({'CONFIRMED' if v['confirmed'] else 'SHORT by %.2f pts' % (100 * v['short_by'])}; dev {v['dev_all']:.4f})" + ("  OVER_BOUND" if v["over_bound"] else ""))
    for f in tmp.glob("*"):
        f.unlink()
    tmp.rmdir()
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("confirm-par")
    p.add_argument("--ds", required=True)
    p.add_argument("--first", default="frozen", choices=("frozen", "e3v"))
    p.add_argument("--slices", type=int, default=6)
    p.add_argument("--worker", default=None)
    p.add_argument("--verify", action="store_true")
    p.add_argument("--host", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--out", default=str(PL4 / "report.md"))
    p = sub.add_parser("miss")
    p.add_argument("--ds", nargs="+", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--all", action="store_true")
    p.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    return {"report": cmd_report, "miss": cmd_miss, "confirm-par": cmd_confirm_par}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
