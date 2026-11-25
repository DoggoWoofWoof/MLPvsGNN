"""Files section 14 (results) of docs/PL7_COVERAGE_RESTRICTION.md from the filed records. Re-run to re-file (it replaces any earlier section 14).
Every number is read from outputs/pl7/ and outputs/pl5/ records (call.json, <d>/select.json, <d>/miss.json, <d>/grid_*.json heads, selftest.json); nothing is chosen here.
The declaration's section 12 (item 1) says the results are filed in section 13; section 13 of the page is the declaration's own 'Disclosed', so the results are section 14.
Usage: python outputs/pl7/filing/build_section14.py [--page PATH]   (PATH defaults to the real page; use a scratch path to look at a draft)"""
import glob
import json
import sys
from pathlib import Path

SCR = Path(__file__).resolve().parent
ROOT = SCR.parents[2]
PL7 = ROOT / "outputs" / "pl7"
PL6 = ROOT / "outputs" / "pl6"
PL5 = ROOT / "outputs" / "pl5"
PAGE = ROOT / "docs" / "PL7_COVERAGE_RESTRICTION.md"
if "--page" in sys.argv:
    PAGE = Path(sys.argv[sys.argv.index("--page") + 1])
ORDER = ["multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp"]
LABEL = {"multihoprag": "MultiHop-RAG", "squad": "SQuAD", "hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue", "metaqa": "MetaQA", "webqsp": "WebQSP"}
LADDER = ["musique", "webqsp", "metaqa"]
INSIDE = ["multihoprag", "squad", "hotpotqa", "2wiki"]


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def f4(x):
    return f"{x:.4f}"


def pts(x):
    return f"{100 * x:+.2f}"


def ci(c):
    return f"[{100 * c[0]:+.2f}, {100 * c[1]:+.2f}]"


def n1(x):
    return f"{x:,.1f}"


def n0(x):
    return f"{x:,.0f}"


def mult(x):
    return f"{x:.2f}"


def yn(b):
    return "yes" if b else "no"


C = rj(PL7 / "call.json")
READ = [d for d in ORDER if C["datasets"][d]["status"] != "NOT_READ"]
NR = len(READ)
COMPLETE = NR == 7


def D(ds):
    return C["datasets"][ds]


def P(ds, k):
    return D(ds)["per_ceiling"][str(k)]


def sel(ds):
    return rj(PL7 / ds / "select.json")


def share(p):
    return "n/a" if p["share_of_graph"] is None else f"{100 * p['share_of_graph']:.2f}%"


def st(v):
    return f"{v['status']} (ALL-only {v['all_status']})"


def cmp_cell(v):
    return f"{pts(v['all_diff'])} {ci(v['all_diff_ci'])}; ratio {v['nodes_ratio']:.3f} [{v['nodes_ratio_ci'][0]:.3f}, {v['nodes_ratio_ci'][1]:.3f}]; {st(v)}"


# ── the numbers the hand-written text quotes, asserted (a re-filing that moves one stops the build) ──────────────────────────────────────────
def _guard():
    assert C["call"] == "NOT_CARRIED" and C["counts"] == {"BELOW": 0, "ABOVE": 0, "LEVEL": 7} and COMPLETE, (C["call"], C["counts"])
    assert C["code_path_equal_everywhere"] is True
    assert C["universality_check"]["unchanged_on_all_of_them"] is True and sorted(C["universality_check"]["datasets_that_met_the_target_inside_the_bound"]) == sorted(INSIDE)
    for ds in ORDER:
        v = P(ds, 2500)["vs_bounded"]
        assert v["same_cell"] and v["all_diff"] == 0.0 and v["nodes_ratio"] == 1.0 and v["status"] == "LEVEL", ds
        assert P(ds, 2500)["at_floor"] is False and P(ds, 2500)["vs_without_pl7_cells"]["same_cell"], ds
    # the three SHORT datasets at 2,500 (eval) and their ladder ends
    assert round(P("musique", 2500)["eval_all"], 4) == 0.8846 and round(P("webqsp", 2500)["eval_all"], 4) == 0.8789 and round(P("metaqa", 2500)["eval_all"], 4) == 0.8968
    assert D("musique")["ceilings"] == [2500, 3500, 5000, 7000, 10000, 14000] and D("webqsp")["ceilings"] == [2500, 3500, 5000, 7000, 10000, 14000, 20000, 28000] and D("metaqa")["ceilings"] == [2500, 3500]
    for ds in LADDER:
        assert D(ds)["knee_ceiling"] is None and not any(p["eval_reached"] for p in D(ds)["per_ceiling"].values()), ds
        assert D(ds)["end_reaches_eval"] is False and D(ds)["universal_same_as_bounded"] is None, ds
    for ds in INSIDE:
        assert D(ds)["knee_ceiling"] == 2500 and P(ds, 2500)["eval_reached"] is True and D(ds)["universal_same_as_bounded"] is True, ds
    # what the PL7 cells added (the only ceilings at which R95_all differs from R95_old)
    diff = {ds: [k for k in D(ds)["ceilings"] if not P(ds, k)["vs_without_pl7_cells"]["same_cell"]] for ds in ORDER}
    assert diff == {"multihoprag": [], "squad": [], "hotpotqa": [], "2wiki": [], "musique": [3500, 5000], "metaqa": [], "webqsp": [3500]}, diff
    w = P("webqsp", 3500)["vs_without_pl7_cells"]
    assert w["status"] == "ABOVE" and w["all_diff_ci"][0] > 0 and w["nodes_ratio"] < 1
    for ds in ("musique", "metaqa", "webqsp"):
        for k in D(ds)["ceilings"][1:]:
            v = P(ds, k)["vs_bounded"]
            assert v["status"] == "LEVEL" and v["all_status"] == "ABOVE", (ds, k)
    won = {ds: [k for k in D(ds)["ceilings"] if P(ds, k)["pick"] in {x["name"] for x in sel(ds)["table_pl7"]}] for ds in LADDER}
    assert won == {"musique": [3500, 5000], "webqsp": [3500], "metaqa": []}, won
    assert not any("+top" in P(ds, k)["pick"] for ds in LADDER for k in won[ds]), "a PL7 hard-cut cell is a pick: the text of 14.5 and 14.8 says none is"
    assert P("musique", 5000)["eval_all"] < P("musique", 3500)["eval_all"]
    assert P("webqsp", 10000)["pick"] == P("webqsp", 7000)["pick"] and P("webqsp", 20000)["pick"] == P("webqsp", 14000)["pick"]
    assert D("webqsp")["per_ceiling"]["2500"]["vs_u"]["status"] == "ABOVE" and D("webqsp")["per_ceiling"]["2500"]["vs_u"]["all_status"] == "BELOW"
    assert D("webqsp")["u_eval_all"] < 0.95 and D("musique")["u_eval_all"] >= 0.95 and D("metaqa")["u_eval_all"] >= 0.95


_guard()


def new_cells_within(ds, k):
    s = sel(ds)
    c = [x for x in s["table_pl7"] if x["nodes"] <= k]
    return len(c), (max(c, key=lambda x: (x["all"], -x["nodes"])) if c else None)


def gates_text():
    out = []
    t = rj(PL7 / "selftest.json")
    heads = {ds: sorted(glob.glob(str(PL7 / ds / "grid_dev_frozen.part*of3.json"))) for ds in LADDER}
    shas = {ds: {rj(f)["cells_sha256"] for f in fs} for ds, fs in heads.items()}
    assert all(len(v) == 1 for v in shas.values()) and all(len(heads[ds]) == 3 for ds in LADDER)
    codes = {sel(ds)["code_path_sha256"] for ds in ORDER} | {rj(f)["code_path_sha256"] for fs in heads.values() for f in fs} | {t["code_path_sha256"]}
    assert len(codes) == 1, codes
    page = PAGE.read_bytes().decode("utf-8")
    for ds in LADDER:
        assert next(iter(shas[ds])) in page, f"{ds}: the grid heads' cell-list sha256 is not the one the declaration states"
    g = {ds: sel(ds)["g_c"] for ds in LADDER}
    assert all(v["verdict"] == "PASS" and v["differing_cells"] == 0 for v in g.values())
    out.append(f"The engine `zpl7g.py` and its code-path hash `{next(iter(codes))[:12]}` are the declaration commit's (24f4892e): the same hash is in the self-test, in all {len(ORDER)} select files and in every grid head (`code_path_equal_everywhere` true; G-U PASS). "
               f"The cell-list sha256 of every dev grid head equals the one the declaration states (section 5): MuSiQue `{next(iter(shas['musique']))[:12]}`, WebQSP `{next(iter(shas['webqsp']))[:12]}`, MetaQA `{next(iter(shas['metaqa']))[:12]}`. "
               f"The self-test (`selftest.json`) is the declaration's own: {t['verdict']} in {t['seconds']} s.")
    out.append("**G-C (the controls, element for element on every dev question)** PASSED on all three: " + "; ".join(
        f"{LABEL[ds]} {g[ds]['cells_compared']} control cells x {g[ds]['questions']:,} questions ({g[ds]['elements']:,} elements), {g[ds]['differing_cells']} differ" for ds in LADDER) + ".")
    rows = []
    for ds in LADDER:
        dv = sorted(glob.glob(str(PL7 / ds / "grid_dev_frozen.part*of3.json")))
        ev = sorted(glob.glob(str(PL7 / ds / "grid_eval_frozen_confirm.part*of3.json")))
        assert len(ev) == 3, ds
        hd, he = [rj(f) for f in dv], [rj(f) for f in ev]
        assert all(h["host"] for h in hd + he)
        rows.append((ds, hd, he))
    est = {"musique": "about 9,500 s per part", "webqsp": "about 9,000 s per part", "metaqa": "under 1 h"}
    lines = ["| dataset | dev grid: cells | seconds per part (sum) | peak GB per part | declared estimate (section 11) | eval confirmation: cells built | seconds per part (sum) | peak GB |", "|---|---|---|---|---|---|---|---|"]
    for ds, hd, he in rows:
        lines.append(f"| {LABEL[ds]} | {hd[0]['cells']} | {', '.join(n0(h['seconds']) for h in hd)} ({n0(sum(h['seconds'] for h in hd))}) | {max(h['peak_gb'] for h in hd):.2f} | {est[ds]} | {he[0]['cells']} | "
                     f"{', '.join(n0(h['seconds']) for h in he)} ({n0(sum(h['seconds'] for h in he))}) | {max(h['peak_gb'] for h in he):.2f} |")
    out.append("Cost (host rows, three parts per dataset; the grids are `outputs/pl7/<d>/grid_dev_frozen.part*of3` and `grid_eval_frozen_confirm.part*of3`; `select` and `call` ran on the laptop in seconds):\n\n" + "\n".join(lines))
    return "\n\n".join(out)


def call_rows():
    r = []
    for ds in ORDER:
        d, p = D(ds), P(ds, 2500)
        v = p["vs_bounded"]
        r.append(f"| {LABEL[ds]} | `{p['pick']}` | {yn(v['same_cell'])} | {f4(p['eval_all'])} [{f4(p['eval_all_wilson'][0])}, {f4(p['eval_all_wilson'][1])}] | {n1(p['eval_nodes'])} / {n0(p['eval_p95'])} / {n0(p['eval_max'])} | "
                 f"{mult(p['nodes_over_2500'])} | {share(p)} | {yn(p['eval_reached'])} | {yn(p['at_floor'])} | {pts(v['all_diff'])} {ci(v['all_diff_ci'])} | {v['nodes_ratio']:.3f} | {v['status']} | `{p['r98']['pick']}` {f4(p['r98']['eval_all'])} at {n1(p['r98']['eval_nodes'])} |")
    return "\n".join(r)


def ladder_table(ds):
    d = D(ds)
    rows = ["| K | R95_all(K), the pick | dev ALL at dev nodes | eval ALL [Wilson 95%] at mean nodes (p95 / max) | x 2,500 | share of graph | reaches 0.95 on eval | step from the row above: eval ALL (points), nodes ratio | vs PL5's bounded pick: ALL diff (points) [95% CI], nodes ratio, status | vs PL6's U95 pick: ALL diff (points) [95% CI], nodes ratio, status | vs R95_old(K) (what the PL7 cells added) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    prev = None
    for k in d["ceilings"]:
        p = P(ds, k)
        step = "-" if prev is None else f"{pts(p['eval_all'] - prev['eval_all'])}, {p['eval_nodes'] / prev['eval_nodes']:.2f}"
        u, b, w = p["vs_u"], p["vs_bounded"], p["vs_without_pl7_cells"]
        wtxt = "same cell" if w["same_cell"] else f"`{w['old']}` {f4(w['old_all'])} at {n1(w['old_nodes'])}: {pts(w['all_diff'])} {ci(w['all_diff_ci'])}, ratio {w['nodes_ratio']:.3f}, {st(w)}"
        rows.append(f"| {k:,} | `{p['pick']}` | {f4(p['dev_all'])} at {n1(p['dev_nodes'])} | {f4(p['eval_all'])} [{f4(p['eval_all_wilson'][0])}, {f4(p['eval_all_wilson'][1])}] at {n1(p['eval_nodes'])} ({n0(p['eval_p95'])} / {n0(p['eval_max'])}) | "
                    f"{mult(p['nodes_over_2500'])} | {share(p)} | {yn(p['eval_reached'])}{' (at the floor)' if p['at_floor'] else ''} | {step} | {pts(b['all_diff'])} {ci(b['all_diff_ci'])}, {b['nodes_ratio']:.3f}, {st(b)} | "
                    f"{pts(u['all_diff'])} {ci(u['all_diff_ci'])}, {u['nodes_ratio']:.3f}, {st(u)} | {wtxt} |")
        prev = p
    ug = f"| PL6's U95 pick | `{d['u_pick']}` | - | {f4(d['u_eval_all'])} at {n1(d['u_eval_nodes'])} (dev nodes {n1(d['u_dev_nodes'])}) | {mult(d['u_eval_nodes'] / 2500)} | - | {yn(d['u_eval_all'] >= 0.95)} | - | - | - | - |"
    rows.append(ug)
    return "\n".join(rows)


def knee_rows():
    r = []
    for ds in ORDER:
        d, p = D(ds), P(ds, 2500)
        if d["knee_ceiling"] is not None:
            r.append(f"| {LABEL[ds]} | {d['knee_ceiling']:,} | {n1(d['knee_eval_nodes'])} | {d['knee_over_2500']:.3f} | none: the ladder starts at 2,500 | the dataset reaches 0.95 on eval inside 2,500 (`{p['pick']}` {f4(p['eval_all'])}); R7(2,500) is PL6's pick and PL5's bounded pick |")
        else:
            last = P(ds, d["ceilings"][-1])
            r.append(f"| {LABEL[ds]} | none on the ladder | - | - | each step is in the ladder table: ending at 2,500 costs {pts(d['end_all_cost_vs_u'])} {ci(d['end_all_cost_ci'])} points of ALL against PL6's U95 pick (nodes ratio {d['end_nodes_ratio_vs_u']:.3f}) | "
                     f"the last ceiling read ({d['ceilings'][-1]:,}) reads {f4(last['eval_all'])} at {n1(last['eval_nodes'])} nodes; PL6's U95 pick {f4(d['u_eval_all'])} at {n1(d['u_eval_nodes'])} ({mult(d['u_eval_nodes'] / 2500)} x 2,500) "
                     f"{'reaches 0.95 on eval' if d['u_eval_all'] >= 0.95 else 'does not reach 0.95 on eval either'} |")
    return "\n".join(r)


def added_rows():
    r = []
    for ds in LADDER:
        s = sel(ds)
        n, b = new_cells_within(ds, 2500)
        inc = s["incumbent"]
        ks = [k for k in D(ds)["ceilings"] if not P(ds, k)["vs_without_pl7_cells"]["same_cell"]]
        txt = "; ".join(f"K = {k:,}: {pts(P(ds, k)['vs_without_pl7_cells']['all_diff'])} {ci(P(ds, k)['vs_without_pl7_cells']['all_diff_ci'])}, nodes ratio {P(ds, k)['vs_without_pl7_cells']['nodes_ratio']:.3f}, {P(ds, k)['vs_without_pl7_cells']['status']} (ALL-only {P(ds, k)['vs_without_pl7_cells']['all_status']})" for k in ks) or "none"
        r.append(f"| {LABEL[ds]} | {s['new_cells']} | {n} | `{b['name']}` {f4(b['all'])} at {n1(b['nodes'])} | `{inc['name']}` {f4(inc['all'])} at {n1(inc['nodes'])} | {pts(b['all'] - inc['all'])} | {txt} |")
    return "\n".join(r)


def miss_rows():
    d1, d2 = [], []
    for ds in LADDER:
        m = rj(PL7 / ds / "miss.json")
        m5 = rj(PL5 / ds / "miss.json")
        t = m["totals"]
        same = (m["cell"] == m5["cell"] and m["totals"] == m5["totals"] and m["D2"]["by_stage_hop"] == m5["D2"]["by_stage_hop"] and m["questions"] == m5["questions"])
        assert m["cell"] == P(ds, 2500)["pick"], ds
        d1.append(f"| {LABEL[ds]} | `{m['cell']}` | {t['q_missing']} of {m['questions']} ({t['q_missing'] / m['questions']:.3f}) | {t['golds_missing']} ({t['golds_missing'] / t['q_missing']:.2f} per such question) | {t['R']} | {t['C']} | {t['U']} | {n0(m['nodes_mean'])} | {yn(same)} |")
        by = m["D2"]["by_stage_hop"]
        g = lambda k: by.get(k, 0)
        d2.append(f"| {LABEL[ds]} | {m['D2']['c_golds']} | {g('cap1')} | {g('cap2')} | {g('cap3')} | {g('budget2')} | {g('budget3')} | {g('upstream')} | {yn(m['D2']['trace_matches_engine'])} |")
    return "\n".join(d1), "\n".join(d2)


def build():
    page = PAGE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    cut = page.find("\n## 14. Results")
    head = (page[:cut] if cut >= 0 else page).rstrip("\n") + "\n"
    unread = [LABEL[d] for d in ORDER if d not in READ]
    title = f"## 14. Results (26 Nov 2026; {NR} of 7 datasets read" + ("" if COMPLETE else f"; {', '.join(unread)} not yet read: the call is INCOMPLETE") + ")"
    d1rows, d2rows = miss_rows()
    counts = C["counts"]
    mu, wq, mq = D("musique"), D("webqsp"), D("metaqa")
    nm, bm = new_cells_within("musique", 2500)
    nw, bw = new_cells_within("webqsp", 2500)
    na, ba = new_cells_within("metaqa", 2500)
    inc = {ds: sel(ds)["incumbent"] for ds in LADDER}
    pm35, pm50 = P("musique", 3500), P("musique", 5000)
    pw35 = P("webqsp", 3500)
    ww = pw35["vs_without_pl7_cells"]
    mw = {k: P("musique", k)["vs_without_pl7_cells"] for k in (3500, 5000)}
    hc = {ds: [rj(f) for f in sorted(glob.glob(str(PL7 / ds / "grid_dev_frozen.part*of3.json")))] for ds in LADDER}
    text = f"""
{title}

Everything here is read by the rules of sections 4 to 8 from the filed records (`outputs/pl7/<d>/{{select,miss}}.json`, `outputs/pl7/call.json`, the grid heads, `outputs/pl7/selftest.json`; PL5's `outputs/pl5/<d>/miss.json` for the equality check of 14.6) by `outputs/mp_unified/zpl7g.py`
(`select`, `confirm`, `call`) and `outputs/mp_unified/zpl5r.py miss --pl7`. No PL5 or PL6 record is changed. Nothing was changed in the rule, the cells or the engine after a number was read. The declaration's section 12 (item 1) says the results are filed in section 13; section 13 of this page is the
declaration's own disclosures, so the results are section 14.

### 14.1 Gates and cost

{gates_text()}

### 14.2 The call at K = 2,500 (read {NR} of 7: {C['call']})

R7(2,500) = R95_all(2,500) of the reading `all` over PL2 to PL6's cells and the PL7 cells, against PL5's bounded pick (what PL2 to PL5 carried); eval, frozen lists, paired bootstrap, one generator per dataset, ceiling and comparison (section 6). `at floor` is the floor caveat of section 6 (my definition): a pick that reaches 0.95 on eval
by its point estimate while its Wilson lower end is below 0.95.

| dataset | R7(2,500), the pick | the same cell as PL5's bounded pick | eval ALL [Wilson 95%] | mean / p95 / max nodes per question | x 2,500 | share of graph | reaches 0.95 on eval | at floor | ALL difference vs the bounded pick (points) [95% CI] | nodes ratio | status | R98(2,500): eval ALL at mean nodes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
{call_rows()}

- **The call: {C['call']}** (BELOW {counts['BELOW']}, ABOVE {counts['ABOVE']}, LEVEL {counts['LEVEL']}; read {NR} of 7). CARRY_PL7 needs all seven read, no BELOW against the bounded pick at 2,500, at least one ABOVE, and the universality check; with no ABOVE the call is NOT_CARRIED and PL5's bounded rule stays what PL2 to PL5 used (section 6).
- **R7(2,500) is PL5's bounded pick on all seven**: the same cell, the ALL difference exactly 0 and the nodes ratio exactly 1, so no dataset can read ABOVE or BELOW. The PL7 cells (507 on MuSiQue, 819 on WebQSP, 117 on MetaQA) are not the dev pick at 2,500 on any of them (14.5).
- **The universality check** (on every dataset whose bounded pick reached 0.95 on dev, R7(2,500) is that cell): {', '.join(LABEL[d] for d in INSIDE)}; unchanged on all of them: {yn(C['universality_check']['unchanged_on_all_of_them'])}. It holds by construction (section 6) and is a check of the code, not a finding.
- **Eval, frozen lists: four datasets reach 0.95 inside 2,500** (MultiHop-RAG {f4(P('multihoprag', 2500)['eval_all'])}, SQuAD {f4(P('squad', 2500)['eval_all'])}, HotpotQA {f4(P('hotpotqa', 2500)['eval_all'])}, 2Wiki {f4(P('2wiki', 2500)['eval_all'])}), at {mult(P('multihoprag', 2500)['nodes_over_2500'])}, {mult(P('squad', 2500)['nodes_over_2500'])}, {mult(P('hotpotqa', 2500)['nodes_over_2500'])}
  and {mult(P('2wiki', 2500)['nodes_over_2500'])} times 2,500 nodes; none is flagged at the floor. **Three do not**: MuSiQue {f4(P('musique', 2500)['eval_all'])}, MetaQA {f4(P('metaqa', 2500)['eval_all'])}, WebQSP {f4(P('webqsp', 2500)['eval_all'])}, short by {100 * (0.95 - P('musique', 2500)['eval_all']):.2f}, {100 * (0.95 - P('metaqa', 2500)['eval_all']):.2f} and {100 * (0.95 - P('webqsp', 2500)['eval_all']):.2f} points; their Wilson upper ends ({f4(P('musique', 2500)['eval_all_wilson'][1])}, {f4(P('metaqa', 2500)['eval_all_wilson'][1])}, {f4(P('webqsp', 2500)['eval_all_wilson'][1])}) are below 0.95.
  The same numbers as PL5's section 14.3, as they must be.

### 14.3 The restriction ladder: what each step down costs, per dataset

Eval, frozen lists; the cell of each ceiling was chosen on dev (rule R7(K)) and read once on eval. The three comparisons are paired over the same questions. **Read the status column as the rule gives it, and the ALL-only status beside it**: the `vs PL6's U95 pick` status reads ABOVE where the pick uses fewer nodes (clause ABOVE 2:
nodes-ratio upper end below 1) even when ALL is lower; the ALL difference beside it says which. Nothing above 2,500 is carried, whatever it reads (section 6).

{chr(10).join(f'**{LABEL[ds]}** ({D(ds)["n_nodes"]:,} nodes in the graph; {D(ds)["eval_questions"]:,} eval questions)' + chr(10) + chr(10) + ladder_table(ds) + chr(10) for ds in ('musique', 'webqsp', 'metaqa'))}
The other four datasets are read at 2,500 only (14.2): their U95 picks sit at {mult(min(P(d, 2500)['nodes_over_2500'] for d in INSIDE))} to {mult(max(P(d, 2500)['nodes_over_2500'] for d in INSIDE))} times 2,500.

### 14.4 The knee of each dataset

The smallest ceiling of the dataset's ladder whose R95_all pick reaches 0.95 on eval (point estimate), its eval nodes as a multiple of 2,500, and what the step below costs in ALL (section 6).

| dataset | knee ceiling | eval nodes at the knee | x 2,500 | what the step below costs in ALL | what the ladder says |
|---|---|---|---|---|---|
{knee_rows()}

**Three datasets have no knee on their ladders** (MuSiQue to 14,000, WebQSP to 28,000, MetaQA to 3,500). PL6's U95 pick is the known reach: MuSiQue {f4(mu['u_eval_all'])} at {n1(mu['u_eval_nodes'])} nodes ({mult(mu['u_eval_nodes'] / 2500)} times 2,500), MetaQA {f4(mq['u_eval_all'])} at {n1(mq['u_eval_nodes'])} ({mult(mq['u_eval_nodes'] / 2500)} times); WebQSP does not reach
0.95 on eval at any size read ({f4(wq['u_eval_all'])} at {n1(wq['u_eval_nodes'])} nodes, {mult(wq['u_eval_nodes'] / 2500)} times 2,500). Between the last ceiling and PL6's pick the ladder reads nothing (the next step of L lies above PL6's dev nodes, section 4), so where the knee lies inside that stretch is not read here.

### 14.5 What the PL7 cells added (the question of section 9)

Section 9 asked whether a wide walk with a hard cut keeps more golds per node than PL5's narrow bounded pick at 2,500. Dev, frozen lists: the best new cell inside 2,500 dev nodes against PL5's bounded pick, and the ceilings at which R95_all differs from R95_old (eval, paired):

| dataset | new cells | new cells inside 2,500 dev nodes | best new cell inside 2,500 (dev ALL at dev nodes) | PL5's bounded pick (dev) | difference (points, dev) | ceilings at which the PL7 cells changed the pick (eval, against R95_old(K)) |
|---|---|---|---|---|---|---|
{added_rows()}

**At 2,500 the answer is no, on all three: the best new cell inside the ceiling reads {f4(bm['all'])} on MuSiQue, {f4(bw['all'])} on WebQSP and {f4(ba['all'])} on MetaQA against {f4(inc['musique']['all'])}, {f4(inc['webqsp']['all'])} and {f4(inc['metaqa']['all'])}** ({100 * (bm['all'] - inc['musique']['all']):+.1f}, {100 * (bw['all'] - inc['webqsp']['all']):+.1f} and {100 * (ba['all'] - inc['metaqa']['all']):+.1f} points on dev).
Above 2,500 the cells added something at three of the thirteen ceilings read above 2,500 (sixteen on the three ladders, three of them at 2,500): MuSiQue at 3,500 ({pts(mw[3500]['all_diff'])} points {ci(mw[3500]['all_diff_ci'])}, nodes ratio {mw[3500]['nodes_ratio']:.3f}, {mw[3500]['status']}) and 5,000 ({pts(mw[5000]['all_diff'])} {ci(mw[5000]['all_diff_ci'])}, {mw[5000]['nodes_ratio']:.3f}, {mw[5000]['status']}), and WebQSP at 3,500
(`{pw35['pick']}`, a budget-following walk at V = 2,500: {f4(pw35['eval_all'])} at {n1(pw35['eval_nodes'])} nodes against `{ww['old']}` {f4(ww['old_all'])} at {n1(ww['old_nodes'])}; {pts(ww['all_diff'])} points {ci(ww['all_diff_ci'])}, nodes ratio {ww['nodes_ratio']:.3f}, {ww['status']}). At no other ceiling did a PL7 cell win on dev. **No hard-cut cell (`+top{{K - b}}`) is the pick at any ceiling on any dataset**: the three PL7 cells that won are union cells of the V = 2,500 walks (`base+walk` at V = 2,500), whose pools are not cut to the ceiling (their maxima are {n0(pm35['eval_max'])}, {n0(pm50['eval_max'])} and {n0(pw35['eval_max'])} nodes on eval).

### 14.6 Still short at 2,500: D1 and D2 (post-hoc, labels read; 300 dev questions evenly spaced over the carve; `zpl5r.py miss --pl7`)

The diagnostic of PL5's section 14.6 at R95_all(2,500) of the three SHORT datasets, which is PL5's chosen cell on each (14.2). **D1** (each gold missing from the pool: R ranked past M, C cut by the caps, the window or the order but inside the uncapped three-hop reach of its seeds, U beyond it):

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (mean, dev) | equals PL5's `miss.json` (cell, totals, stages) |
|---|---|---|---|---|---|---|---|---|
{d1rows}

**D2** (which stage cut each C gold, from a traced walk of the cell):

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|
{d2rows}

**Read:** the records equal PL5's exactly (same cell, same 300 questions, a deterministic walk), so PL5's reading of them stands unchanged: the golds the 2,500-node picks miss are cut by a per-seed or per-frontier cap or by the visited budget, not ranked out and (but for two on MetaQA) not unreachable. What this round adds is on dev ALL (14.5): a wider walk cut to 2,500 nodes did not keep more golds than the narrow bounded walk. **How the ceiling is cut is a separate question from which golds the cut keeps**: the cut ranks the walk's nodes by cosine to the question only (section 5), and a gold that arrives through a low-cosine bridge is the first to go; I did not test that on this page.

### 14.7 What the call means, and what it leaves unread

- **NOT_CARRIED, {NR} of 7 read, is a statement about the 2,500 pick, not about the curve.** With one rule on all seven, the node count restricted back to 2,500 reads PL5's bounded pick on every dataset; nothing in the PL7 grid improves on it there. The four datasets that reach 0.95 reach it at {mult(min(P(d, 2500)['nodes_over_2500'] for d in INSIDE))} to {mult(max(P(d, 2500)['nodes_over_2500'] for d in INSIDE))} times 2,500;
  the other three are SHORT at 2,500 and stay so.
- **The price curve is the finding** (section 9 said it would be), eval, frozen lists: MuSiQue {f4(P('musique', 2500)['eval_all'])} at {n0(P('musique', 2500)['eval_nodes'])} nodes, {f4(pm35['eval_all'])} at {n0(pm35['eval_nodes'])} ({mult(pm35['nodes_over_2500'])} times 2,500), {f4(P('musique', 7000)['eval_all'])} at {n0(P('musique', 7000)['eval_nodes'])} ({mult(P('musique', 7000)['nodes_over_2500'])}), {f4(P('musique', 10000)['eval_all'])} at {n0(P('musique', 10000)['eval_nodes'])} ({mult(P('musique', 10000)['nodes_over_2500'])}), {f4(P('musique', 14000)['eval_all'])} at {n0(P('musique', 14000)['eval_nodes'])} ({mult(P('musique', 14000)['nodes_over_2500'])}),
  PL6's {f4(mu['u_eval_all'])} at {n0(mu['u_eval_nodes'])} ({mult(mu['u_eval_nodes'] / 2500)}); WebQSP {f4(P('webqsp', 2500)['eval_all'])} at {n0(P('webqsp', 2500)['eval_nodes'])}, {f4(pw35['eval_all'])} at {n0(pw35['eval_nodes'])} ({mult(pw35['nodes_over_2500'])}), {f4(P('webqsp', 5000)['eval_all'])} at {n0(P('webqsp', 5000)['eval_nodes'])} ({mult(P('webqsp', 5000)['nodes_over_2500'])}), {f4(P('webqsp', 7000)['eval_all'])} at {n0(P('webqsp', 7000)['eval_nodes'])} ({mult(P('webqsp', 7000)['nodes_over_2500'])}), {f4(P('webqsp', 14000)['eval_all'])} at {n0(P('webqsp', 14000)['eval_nodes'])} ({mult(P('webqsp', 14000)['nodes_over_2500'])}), {f4(P('webqsp', 28000)['eval_all'])} at {n0(P('webqsp', 28000)['eval_nodes'])} ({mult(P('webqsp', 28000)['nodes_over_2500'])}),
  PL6's {f4(wq['u_eval_all'])} at {n0(wq['u_eval_nodes'])} ({mult(wq['u_eval_nodes'] / 2500)}), never 0.95; MetaQA {f4(P('metaqa', 2500)['eval_all'])} at {n0(P('metaqa', 2500)['eval_nodes'])}, {f4(P('metaqa', 3500)['eval_all'])} at {n0(P('metaqa', 3500)['eval_nodes'])} ({mult(P('metaqa', 3500)['nodes_over_2500'])}), PL6's {f4(mq['u_eval_all'])} at {n0(mq['u_eval_nodes'])} ({mult(mq['u_eval_nodes'] / 2500)}).
  **To the user's weighing, on the current links: PL6's pick reaches 0.95 on MetaQA at {mult(mq['u_eval_nodes'] / 2500)} times 2,500 nodes (the ladder's {mult(P('metaqa', 3500)['nodes_over_2500'])} times does not) and on MuSiQue at {mult(mu['u_eval_nodes'] / 2500)} times (the ladder's {mult(P('musique', 14000)['nodes_over_2500'])} times does not); on WebQSP no size read reaches it.** The curve has no cliff: the largest rise from one ceiling's pick to the next is {pts(max(P(ds, k)['eval_all'] - P(ds, kp)['eval_all'] for ds in LADDER for kp, k in zip(D(ds)['ceilings'], D(ds)['ceilings'][1:])))} points of eval ALL, for about 1.5 to 1.7 times the nodes (section 1 fact 1).
- **Every step above 2,500 buys ALL for nodes, and the rule's status never calls it ABOVE**: at every ceiling above 2,500 on the three ladders the status against the bounded pick is LEVEL while the ALL-only status is ABOVE (more ALL for more nodes: the price clause does not fire). A reader who wants the ALL gain at {mult(pw35['nodes_over_2500'])} times 2,500 on WebQSP ({pts(pw35['vs_bounded']['all_diff'])} points {ci(pw35['vs_bounded']['all_diff_ci'])}) or
  at {mult(pm35['nodes_over_2500'])} times on MuSiQue ({pts(pm35['vs_bounded']['all_diff'])} {ci(pm35['vs_bounded']['all_diff_ci'])}) is reading the ALL-only column; the decision is the user's.
- **What it leaves unread.** (a) Every number is on the current links and is provisional (section 10); the link set is not final, and PL8, the one pool re-run on it, is declared only after it is. (b) The e3v lists are not read here (this page declares none). (c) MetaQA's ladder stops at 3,500 (the steps of L below PL6's dev nodes), so its knee between 3,500 and PL6's {n0(mq['u_eval_nodes'])} nodes
  is not read. (d) The cut ranks by cosine only; a parent-conditioned rank, a hop-weighted schedule and label-ordered caps are named in PL5 section 12 and are not in this round.

### 14.8 Surprises and contradictions with filed numbers

- **WebQSP's PL6-comparison status at 2,500 reads ABOVE although ALL is {pts(P('webqsp', 2500)['vs_u']['all_diff'])} points** {ci(P('webqsp', 2500)['vs_u']['all_diff_ci'])}: neither pick reaches 0.95 on eval, and the nodes ratio {P('webqsp', 2500)['vs_u']['nodes_ratio']:.3f} has its upper end below 1 (clause ABOVE 2). It is a price statement; the ALL-only status is BELOW. The same comparison reads BELOW on MuSiQue and MetaQA, where PL6's pick reaches 0.95 and the 2,500 pick does not (clause BELOW 1).
- **Eval ALL is not monotone in the ceiling**: MuSiQue's pick at 5,000 reads {f4(pm50['eval_all'])} at {n0(pm50['eval_nodes'])} nodes against {f4(pm35['eval_all'])} at {n0(pm35['eval_nodes'])} nodes for the 3,500 ceiling (dev {f4(pm50['dev_all'])} against {f4(pm35['dev_all'])}): the dev frontier of all cells has no step between 3,518 and 5,783 nodes (section 1 fact 3, with the PL7 cells in), so the step buys nothing, and the dev gain of {100 * (pm50['dev_all'] - pm35['dev_all']):+.2f} points reads {pts(pm50['eval_all'] - pm35['eval_all'])} on eval (point estimates; no interval is filed for this pair).
  WebQSP's 10,000 and 20,000 picks are the cells of its 7,000 and 14,000 for the same reason.
- **A ceiling on the mean is not a ceiling on a question**: MetaQA's 3,500 pick has a mean of {n0(P('metaqa', 3500)['eval_nodes'])} nodes and a maximum of {n0(P('metaqa', 3500)['eval_max'])}; only the hard-cut cells have a pool at most K by construction, and no hard-cut cell of PL7 is a pick at any ceiling (14.5).
- **The declared host-cost estimate was too high for WebQSP, outside the stated factor of 1.5**: {n0(sum(h['seconds'] for h in hc['webqsp']) / 3)} s per part measured against about 9,000 declared (a factor of {9000 / (sum(h['seconds'] for h in hc['webqsp']) / 3):.1f}); MuSiQue {n0(sum(h['seconds'] for h in hc['musique']) / 3)} s per part against about 9,500; MetaQA {n0(sum(h['seconds'] for h in hc['metaqa']) / 3)} s per part against under an hour. The estimate came from PL6's seconds (section 11); it changes no number.
- **The declaration's pointer to section 13 for the results was wrong** (see the introduction of this section); nothing else in the declaration is corrected here.

### 14.9 The next changes (names only; none is adopted)

Declared separately, before any number, when the user says the link set is final: **PL8**, one pool re-run on it, in two stages (a dev-question subset to diagnose which golds the capped pools still miss, then the full dev and the one confirmation). From PL5 section 12, unchanged by these results: a parent-conditioned rank (a node's cosine plus its best parent's), a hop-weighted schedule,
label-ordered caps once a UL2 layout exists on every dataset. The link-side stage (section 10) comes first.

### 14.10 Disclosed and undone

- **`zpl5r.py` gained the `--pl7` flag of `miss`** after the call was read and before D1 and D2 were run (it reads `outputs/pl7/<d>/select.json` at `picks["2500"]["R95_all"]` and writes `outputs/pl7/<d>/miss.json`); the PL5 and PL6 paths of the script are unchanged. D1 and D2 ran at the three SHORT datasets' 2,500 picks, as section 7 declared, and equal PL5's records (14.6).
- **`select` ran on all seven datasets (26 Nov 03:04 to 03:05 IST) before the confirmation rows were issued**; `confirm` needed four cells that PL5's and PL6's confirmation grids do not hold (MetaQA 1, MuSiQue 2, WebQSP 1), the other eval columns were read from them. `call` ran once, after the nine confirmation parts arrived.
- **Every ceiling above 2,500 is a reading on the way down**, filed with its interval and never carried (section 6); the floor flag `at_floor` is my definition and flagged no pick on any dataset, so it changed no reading.
- **{sel('musique')['new_cells']} (MuSiQue), {sel('webqsp')['new_cells']} (WebQSP) and {sel('metaqa')['new_cells']} (MetaQA) new cells are added to the 6,400 or so per dataset in the selection** ({sum(sel(ds)['new_cells'] for ds in LADDER):,} in all; section 13 of the declaration said about 1,750, an estimate made before the lists were counted): a winner's dev ALL is optimistic, so every call is a paired interval on eval; the dev numbers of 14.3 and 14.5 are the selection's, not a result.
- **The pools are on the current links and provisional** (section 10); the four datasets inside 2,500 have no PL7 cells and no host row (section 5), their eval record is PL5's confirmation of the same cell.
- The grid arrays (`*.npz`) are not committed (as PL2's to PL6's); the heads, selects, diagnostics, the call and the builder are. The laptop select chain is `outputs/pl7/run_select.sh`.
"""
    PAGE.write_bytes((head + text.rstrip("\n") + "\n").encode("utf-8"))
    print(f"{len(text.splitlines())} lines; {NR} read; {C['call']} -> {PAGE}")


if __name__ == "__main__":
    build()
