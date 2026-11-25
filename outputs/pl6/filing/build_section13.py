"""Files section 13 (results) of docs/PL6_COVERAGE_ROUND5.md from the filed records. Re-run to re-file when more datasets are read (it replaces any earlier section 13).
Every number is read from outputs/pl6/ and outputs/pl5/ records (call.json, <d>/select.json, <d>/select_r<k>.json, <d>/miss.json, gr_*.json, grid heads, time_*.json); nothing is chosen here.
Section 12 of the page is the declaration's own 'Disclosed'; the results are section 13 (the declaration's section 11 says 'section 12 ... with the tables': the number was taken by the disclosures).
Usage: python outputs/pl6/filing/build_section13.py [--page PATH]   (PATH defaults to the real page; use a scratch path to look at a draft)"""
import glob
import json
import sys
from pathlib import Path

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
ROOT = SCR.parents[2]
PL6 = ROOT / "outputs" / "pl6"
PL5 = ROOT / "outputs" / "pl5"
PAGE = ROOT / "docs" / "PL6_COVERAGE_ROUND5.md"
if "--page" in sys.argv:
    PAGE = Path(sys.argv[sys.argv.index("--page") + 1])
ORDER = ["multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp"]
LABEL = {"multihoprag": "MultiHop-RAG", "squad": "SQuAD", "hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue", "metaqa": "MetaQA", "webqsp": "WebQSP"}
ANY6 = ["3000", "4000", "5000", "6000", "7000", "10000", "15000", "20000", "30000", "40000", "60000", "80000"]


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def pts(x):
    return f"{100 * x:+.2f}"


def yn(b):
    return "yes" if b else "no"


C = rj(PL6 / "call.json")
READ = [d for d in ORDER if C["datasets"][d]["status"] != "NOT_READ"]
NR = len(READ)
COMPLETE = NR == 7


def D(ds):
    return C["datasets"][ds]


def sel(ds):
    return rj(PL6 / ds / "select.json")


def graph_nodes(ds):
    """The graph's node count where a record holds it (PL6's grid heads, the time probes, the self-test's per-dataset checks); None where none does (call.json's n_nodes is empty at rung 0)."""
    fs = sorted(glob.glob(str(PL6 / ds / "grid_dev_frozen_r*.part*of3.json")))
    if fs:
        return rj(fs[0])["n_nodes"]
    fs = sorted(glob.glob(str(PL6 / f"time_{ds}_r*.json")))
    if fs:
        return rj(fs[0])["n_nodes"]
    st = rj(PL6 / "selftest.json")

    def walk(o):
        if isinstance(o, dict):
            if o.get("dataset") == ds and "n_nodes" in o:
                return o["n_nodes"]
            for v in o.values():
                r = walk(v)
                if r:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = walk(v)
                if r:
                    return r
        return None
    return walk(st)


def has_conf(ds):
    return (PL6 / ds / "confirm_frozen.json").is_file()


# ---------------------------------------------------------------- 13.1 gates and cost
def gates_text():
    st = rj(PL6 / "selftest.json")
    lap = [d for d in ORDER if (PL6 / f"gr_{d}.json").is_file()]
    host = [d for d in ORDER if (PL6 / f"gr_{d}_host.json").is_file()]
    bad = 0
    for d in lap:
        g = rj(PL6 / f"gr_{d}.json")
        bad += g["present_mismatches"] + g["size_mismatches"] + (g["verdict"] != "PASS")
    for d in host:
        g = rj(PL6 / f"gr_{d}_host.json")
        bad += g["present_mismatches"] + g["size_mismatches"] + (g["verdict"] != "PASS")
    assert bad == 0, "a G-R record is not PASS"
    s0 = sel(READ[0])
    gr_l = ", ".join(LABEL[d] for d in lap) or "none"
    gr_h = ", ".join(LABEL[d] for d in host) or "none"
    cost = []
    for d in ORDER:
        for r in (1, 2, 3, 4):
            fs = sorted(glob.glob(str(PL6 / d / f"grid_dev_frozen_r{r}.part*of3.json")))
            if len(fs) != 3:
                continue
            hs = [rj(f) for f in fs]
            where = "host" if hs[0].get("host") else "laptop"
            cost.append(f"{LABEL[d]} rung {r} {sum(h['seconds'] for h in hs):,.0f} s, {max(h['peak_gb'] for h in hs):.2f} GB peak ({where})")
    probes = []
    for d in ("musique", "webqsp"):
        for r in (1, 2, 3, 4):
            p = PL6 / f"time_{d}_r{r}.json"
            if p.is_file():
                t = rj(p)
                probes.append(f"{LABEL[d]} rung {r}: {t['seconds_per_question']:.1f} s/q, {t['nodes_scored_per_question']:,.0f} nodes scored")
    return (f"The engine `zpl6g.py` is the declaration commit's (744811e6): script sha256 starts `{st['script_sha256'][:12]}`, the code-path hash `{s0['code_path_sha256'][:12]}` is equal in every select head "
            f"and `call.json` (`code_path_equal_everywhere` {str(C['code_path_equal_everywhere']).lower()}; G-U PASS). **G-P** and the engine self-test PASSED before the commit (`selftest.json`). **G-R** (PL5's 1,872 cells on the first 100 dev questions and PL4's 1,932 on the first 40, "
            f"rebuilt by this engine with the ladder's settings added, compared element for element) PASSED with 0 present and 0 size mismatches on: laptop {gr_l}; host (MuSiQue's, read for every host rung) {gr_h}. "
            f"Cost of the ladder's dev grids (sum of the three parts): {'; '.join(cost) or 'no rung was run'}. The two-question probes filed before the runs: {'; '.join(probes)}.")


# ---------------------------------------------------------------- 13.2 the ladder
def ladder_rows():
    rows = []
    for d in ORDER:
        s = sel(d)
        read = s["rung"]
        best = " -> ".join(f"{b:.4f}" for b in s["best_by_rung"])
        u, bnd = s["picks"]["R95_all"], s["picks"]["R95_bounded"]
        n = s.get("n_nodes") or graph_nodes(d)
        rows.append(f"| {LABEL[d]} | {s['ladder']['state']} | {read} | {best} | `{u['name']}` {u['all']:.4f} at {u['nodes']:,.1f}{'' if u['reached'] else ' (SHORT)'} | "
                    f"`{bnd['name']}` {bnd['all']:.4f} at {bnd['nodes']:,.1f}{'' if bnd['reached'] else ' (SHORT)'} | {format(n, ',') if n else 'not filed here'} |")
    return "\n".join(rows)


# ---------------------------------------------------------------- 13.3 the call and the price
def clause_rows():
    rows = []
    for d in ORDER:
        v = D(d)
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[d]} | not read | | | | | **NOT READ** | | |")
            continue
        lo, hi = v["all_diff_ci"]
        rlo, rhi = v["nodes_ratio_ci"]
        b1 = (not v["new_reaches"]) and v["old_reaches"]
        b2 = (hi < 0) and (rlo >= 1)
        a1 = v["new_reaches"] and not v["old_reaches"]
        a2 = (rhi < 1) and not (b1 or b2)
        a3 = (lo > 0) and (rhi <= 1)
        st = "BELOW" if (b1 or b2) else ("ABOVE" if (a1 or a2 or a3) else "LEVEL")
        assert st == v["status"], (d, st, v["status"])
        same = " (the same cell)" if v["same_cell"] else ""
        rows.append(f"| {LABEL[d]} | `{v['new']}` {v['new_all']:.4f} at {v['new_nodes']:,.1f} | `{v['old']}` {v['old_all']:.4f} at {v['old_nodes']:,.1f} | {yn(v['new_reaches'])} / {yn(v['old_reaches'])} | "
                    f"{pts(v['all_diff'])} [{pts(lo)}, {pts(hi)}] | {v['nodes_ratio']:.4f} [{rlo:.4f}, {rhi:.4f}] | **{st}**{same} | {v['all_status']} | "
                    f"{yn(b1)} / {yn(b2)}; {yn(a1)} / {yn(a2)} / {yn(a3)} |")
    return "\n".join(rows)


def price_rows():
    rows = []
    for d in ORDER:
        v = D(d)
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[d]} | not read | | | | | |")
            continue
        n = graph_nodes(d)
        extra = v["new_nodes"] - v["old_nodes"]
        share = f"{100 * v['new_nodes'] / n:.2f}% ({n:,})" if n else "n/a"
        ratio = f"{v['new_nodes'] / v['old_nodes']:.2f}x" if v["old_nodes"] else "n/a"
        rows.append(f"| {LABEL[d]} | {v['new_nodes']:,.1f} | {v['old_nodes']:,.1f} | {extra:+,.1f} ({ratio}) | {v['new_p95']:,.0f} | {v['new_max']:,.0f} | {share} |")
    return "\n".join(rows)


# ---------------------------------------------------------------- 13.4 what the ladder bought
def ladder_cmp_rows():
    rows = []
    for d in ORDER:
        v = D(d)
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[d]} | not read | | | | |")
            continue
        for tag, key in (("U95", "ladder_cmp"), ("U98", "r98")):
            x = v[key]
            lo, hi = x["all_diff_ci"]
            rlo, rhi = x["nodes_ratio_ci"]
            rows.append(f"| {LABEL[d]} | {tag} | `{x['new']}` {x['new_all']:.4f} at {x['new_nodes']:,.1f} | `{x['old']}` {x['old_all']:.4f} at {x['old_nodes']:,.1f} | "
                        f"{pts(x['all_diff'])} [{pts(lo)}, {pts(hi)}] | {x['nodes_ratio']:.4f} [{rlo:.4f}, {rhi:.4f}] | {x['status']} ({x['all_status']}){' same cell' if x['same_cell'] else ''} |")
    return "\n".join(rows)


# ---------------------------------------------------------------- 13.5 the frontier above the bound
def frontier_rows():
    rows = []
    for d in ORDER:
        v = D(d)
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[d]} | not read |" + " |" * len(ANY6) + " **NOT READ** |")
            continue
        cells = []
        for b in ANY6:
            f = v["frontier"][b]
            if f["status"] == "NONE":
                cells.append("no cell")
                continue
            lo, hi = f["all_diff_ci"]
            cells.append(f"{pts(f['all_diff'])} [{pts(lo)}, {pts(hi)}] {f['status']}" + (" =" if f["same_cell"] else ""))
        rows.append(f"| {LABEL[d]} | " + " | ".join(cells) + f" | **{v['frontier_status']}** |")
    return "\n".join(rows)


def frontier_detail():
    lines = []
    for d in ORDER:
        if D(d)["status"] == "NOT_READ":
            continue
        for b in ANY6:
            f = D(d)["frontier"][b]
            if f["status"] == "NONE" or f["same_cell"]:
                continue
            lines.append(f"| {LABEL[d]} | {int(b):,} | ladder `{f['new']}`; rung 0 `{f['old']}` | {f['new_dev_all']:.4f} at {f['new_dev_nodes']:,.1f}; {f['old_dev_all']:.4f} at {f['old_dev_nodes']:,.1f} | "
                         f"{f['new_all']:.4f} at {f['new_nodes']:,.1f}; {f['old_all']:.4f} at {f['old_nodes']:,.1f} | {pts(f['all_diff'])} {f['status']} |")
    return "\n".join(lines) or "| (the ladder and rung 0 name the same cell at every budget of every dataset read) | | | | | |"


# ---------------------------------------------------------------- D1 / D2 at the ladder's pick
def miss_rows():
    d1, d2 = [], []
    for d in ORDER:
        p = PL6 / d / "miss.json"
        if not p.is_file():
            continue
        m = rj(p)
        t = m["totals"]
        st = m["D2"]["by_stage_hop"]
        d1.append(f"| {LABEL[d]} | `{m['cell']}` | {t['q_missing']} of {m['questions']} ({t['q_missing'] / m['questions']:.3f}) | {t['golds_missing']} ({t['golds_missing'] / max(1, t['q_missing']):.2f} per such question) | "
                  f"{t['R']} | {t['C']} | {t['U']} | {m['nodes_mean']:,.1f} |")
        d2.append(f"| {LABEL[d]} | {m['D2']['c_golds']} | {st.get('cap1', 0)} | {st.get('cap2', 0)} | {st.get('cap3', 0)} | {st.get('budget1', 0)} | {st.get('budget2', 0)} | {st.get('budget3', 0)} | "
                  f"{st.get('upstream', 0)} | {yn(m['D2']['trace_matches_engine'])} |")
    return "\n".join(d1), "\n".join(d2)


# ---------------------------------------------------------------- the page
def build(text_hook=None):
    page = PAGE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    cut = page.find("\n## 13. Results")
    head = (page[:cut] if cut >= 0 else page).rstrip("\n") + "\n"
    counts = C["counts"]
    below = [d for d in READ if D(d)["status"] == "BELOW"]
    above = [d for d in READ if D(d)["status"] == "ABOVE"]
    unread = [LABEL[d] for d in ORDER if d not in READ]
    meet = [d for d in READ if D(d)["new_all"] >= 0.95]
    not_meet = [d for d in READ if D(d)["new_all"] < 0.95]
    d1rows, d2rows = miss_rows()
    title = f"## 13. Results (26 Nov 2026; {NR} of 7 datasets read" + ("" if COMPLETE else f"; {', '.join(unread)} not yet read: the call is INCOMPLETE") + ")"
    ft = text_hook(locals()) if text_hook else {}
    uc = C["universality_check"]
    text = f"""
{title}

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl6/<d>/{{select,select_r<k>,confirm_frozen,miss}}.json`, `outputs/pl6/call.json`, `outputs/pl6/gr_*.json`, the grid heads) by
`outputs/mp_unified/zpl6g.py` (`select`, `ladder`, `confirm`, `call`) and `zpl5r.py miss`. PL5's pages and records are not changed; PL5's selects are this page's rung 0. Nothing was changed in the rule, the cells or the engine after a number was read.
Section 12 of this page is the declaration's own disclosures; this results section is therefore 13 (section 11 of the declaration says 12).
{ft.get('head', '')}
### 13.1 Gates and cost

{gates_text()}

### 13.2 The ladder: what the rule asked for, per dataset

The stop rule reads `best[r]` = the best dev ALL over rungs 0 to r: REACHED when it is 0.95 or more, STOPPED_RISING when the last rung raised it by less than 0.005, GRAPH_END when V has met the graph, LADDER_END after rung 4, otherwise the next rung is asked for.

| dataset | ladder state | rungs read | best dev ALL, rung 0 -> last rung | U95 of all cells (dev ALL at dev nodes) | PL5's bounded pick (dev ALL at dev nodes) | graph nodes |
|---|---|---|---|---|---|---|
{ladder_rows()}
{ft.get('ladder', '')}

### 13.3 The call (read {NR} of 7: {C['call']})

NEW = U95 of the reading `all` (rule U, no node bound), OLD = PL5's bounded pick (R95 of PL5's reading `all` inside 2,500, what PL2 to PL5 carried); eval, frozen lists, paired bootstrap, one generator per dataset. Clauses as section 5:
BELOW (1) NEW misses where OLD reaches / (2) ALL difference upper end < 0 with ratio lower end >= 1; ABOVE (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (not BELOW) / (3) ALL difference lower end > 0 with ratio upper end <= 1. The ALL-only status ignores nodes.

| dataset | NEW (U95, eval ALL at mean nodes) | OLD (PL5 bounded pick, eval ALL at mean nodes) | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | status | ALL-only status | BELOW 1 / 2; ABOVE 1 / 2 / 3 |
|---|---|---|---|---|---|---|---|---|
{clause_rows()}

- **BELOW: {len(below)}.** ABOVE: {len(above)}{' (' + ', '.join(LABEL[d] for d in above) + ')' if above else ''}; LEVEL: {NR - len(below) - len(above)} of the {NR} read.
- **The universality check** (on every dataset whose bounded pick reached 0.95 on dev, U95 is that same cell): datasets {', '.join(LABEL[d] for d in uc['datasets_that_met_the_target_inside_the_bound']) or 'none'}; unchanged on all of them: **{yn(uc['unchanged_on_all_of_them'])}** (an identity by construction, recorded as a code check).
- **The call: {C['call']}** (BELOW {counts['old']['BELOW']}, ABOVE {counts['old']['ABOVE']}, LEVEL {counts['old']['LEVEL']}; read {NR} of 7). CARRY_PL6 needs all seven read, the universality check, no BELOW, and at least one ABOVE{'' if COMPLETE else ': it cannot be called until ' + ', '.join(unread) + (' is' if len(unread) == 1 else ' are') + ' read'}.
{ft.get('call_note', '')}

**Which datasets meet the target on eval, frozen lists (ALL >= 0.95), of the {NR} read: {len(meet)}**: {', '.join(LABEL[d] for d in meet) or 'none'}. **They do not:** {', '.join(LABEL[d] + ' (' + format(D(d)['new_all'], '.4f') + ')' for d in not_meet) or 'none'}.

**The price, filed beside the call and deciding nothing** (eval, frozen lists; nodes per question of the pool the rule hands to the next stage):

| dataset | NEW mean nodes | OLD mean nodes | extra nodes per question (ratio) | NEW 95th percentile | NEW maximum | NEW mean as a share of the graph |
|---|---|---|---|---|---|---|
{price_rows()}
{ft.get('price', '')}

### 13.4 What the ladder bought beside the call (U95 and U98 of all cells against rule U over rung 0's 6,078 cells; eval; decides nothing)

| dataset | floor | ladder + rung 0 (rule U, eval) | rung 0 alone (rule U, eval) | ALL difference, 95% CI | nodes ratio, 95% CI | status (ALL-only) |
|---|---|---|---|---|---|---|
{ladder_cmp_rows()}
{ft.get('bought', '')}

### 13.5 The frontier above the bound (matched node counts; eval ALL difference of the ladder's cells against rung 0's, points; decides nothing)

A dataset is BELOW if any budget is BELOW, ABOVE if at least two budgets are ABOVE and none BELOW, else LEVEL (ALL-only status; `=` marks the same cell on both sides):

| dataset | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 | 10,000 | 15,000 | 20,000 | 30,000 | 40,000 | 60,000 | 80,000 | frontier status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
{frontier_rows()}

- **Frontier counts:** BELOW {counts['frontier']['BELOW']}, ABOVE {counts['frontier']['ABOVE']}, LEVEL {counts['frontier']['LEVEL']} (read {NR} of 7).

Where the ladder's cell and rung 0's differ (dev ALL at dev nodes; eval ALL at eval nodes):

| dataset | budget | cells | dev | eval | difference |
|---|---|---|---|---|---|
{frontier_detail()}
{ft.get('frontier_note', '')}

### 13.6 What is still short, in numbers (diagnostics D1 and D2, post-hoc, labels read, 300 dev questions evenly spaced over the carve; decides nothing)
{ft.get('short_intro', '')}
| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes |
|---|---|---|---|---|---|---|---|
{d1rows}

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 1 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|---|
{d2rows}
{ft.get('short', '')}

### 13.7 What the call means, and what it leaves unread
{ft.get('meaning', '')}

### 13.8 The next changes (names only; none is adopted)
{ft.get('next', '')}

### 13.9 Surprises and contradictions with filed numbers
{ft.get('surprises', '')}

### 13.10 Disclosed and undone
{ft.get('disclosed', '')}
"""
    out = head + text
    PAGE.write_bytes(out.replace("\r\n", "\n").encode("utf-8"))
    print(len(out.splitlines()), "lines;", NR, "read;", C["call"], "->", PAGE)


if __name__ == "__main__":
    hook = None
    try:
        import pl6_text as TX  # noqa: E402  (the hand-written sentences, added after the numbers are read)
        hook = TX.hook
    except ImportError:
        pass
    build(hook)
