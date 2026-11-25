"""Files section 14 (results) of docs/PL5_COVERAGE_ROUND4.md from the filed records. Re-run to re-file when more datasets are read (it replaces any earlier section 14).
Every number is read from outputs/pl5/ records (call.json, <d>/select.json, <d>/confirm_*.json, <d>/miss.json, gr_*.json, report.md); nothing is chosen here.
Usage: python outputs/pl5/filing/build_section14.py [--page PATH]   (PATH defaults to the real page; use a scratch path to look at a draft)"""
import glob
import json
import re
import sys
from pathlib import Path

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
import pl5_tables as T  # noqa: E402

ROOT = T.ROOT
PL5 = T.PL5
PAGE = ROOT / "docs" / "PL5_COVERAGE_ROUND4.md"
if "--page" in sys.argv:
    PAGE = Path(sys.argv[sys.argv.index("--page") + 1])
ORDER, LABEL = T.ORDER, T.LABEL
C = T.call()
pts, yn = T.pts, T.yn
READ = [d for d in ORDER if C["datasets"][d]["status"] != "NOT_READ"]
NR = len(READ)
SHORTS = [d for d in READ if T.sel(d)["short"]["95"]]
COMPLETE = NR == 7


def D(ds):
    return C["datasets"][ds]


def has(ds):
    return ds in READ


def pick(ds, key):
    return T.sel(ds)["picks"][key]


def f4(x):
    return f"{x:.4f}"


def f1(x):
    return f"{x:,.1f}"


# ---------------------------------------------------------------- 14.1 gates and cost
def gates_text():
    laptop = [d for d in ("multihoprag", "squad", "musique", "metaqa", "webqsp") if (PL5 / f"gr_{d}.json").is_file()]
    host = [d for d in ("musique", "hotpotqa", "2wiki") if (PL5 / f"gr_{d}_host.json").is_file()]
    gr_l = ", ".join(LABEL[d] for d in laptop)
    gr_h = ", ".join(LABEL[d] for d in host)
    bad = 0
    el = None
    for d in laptop:
        g = T.rj(PL5 / f"gr_{d}.json")
        bad += g["present_mismatches"] + g["size_mismatches"] + (g["verdict"] != "PASS")
        if d == "musique":
            el = g["elements"]
    for d in host:
        g = T.rj(PL5 / f"gr_{d}_host.json")
        bad += g["present_mismatches"] + g["size_mismatches"] + (g["verdict"] != "PASS")
        el = el or g["elements"]
    assert bad == 0, "a G-R record is not PASS"
    st = T.rj(PL5 / "selftest.json")
    s0 = T.sel(READ[0])
    cost = []
    for d in ORDER:
        fs = sorted(glob.glob(str(PL5 / d / "grid_dev_frozen.part*of3.json")))
        if len(fs) != 3:
            cost.append(f"{LABEL[d]}: not read")
            continue
        hs = [T.rj(f) for f in fs]
        where = "host" if hs[0].get("host") else "laptop"
        cost.append(f"{LABEL[d]} {sum(h['seconds'] for h in hs):,.0f} s, {max(h['peak_gb'] for h in hs):.2f} GB peak ({where})")
    cf = []
    for d in ORDER:
        p = PL5 / d / "grid_eval_frozen_confirm.json"
        if p.is_file():
            cf.append(f"{LABEL[d]} {T.rj(p)['seconds']:,.0f} s")
    return (f"The engine `zpl5g.py` is the declaration commit's (3168891e): script sha256 starts `{st['script_sha256'][:12]}`, the code-path hash `{s0['code_path_sha256'][:12]}` is equal "
            f"in every select head and `call.json` (`code_path_equal_everywhere` {str(C['code_path_equal_everywhere']).lower()}; G-U PASS), the cell list `{s0['cells_sha256_pl5'][:12]}` (1,872 cells; "
            f"6,078 with PL2's, PL3's and PL4's). **G-R** (PL2's 804, PL3's 1,470 and PL4's 1,932 cells rebuilt by this engine on the first 400 dev questions and PL4's on the first 150, {el:,} elements per dataset, "
            f"224 questions on MultiHop-RAG) PASSED with 0 present and 0 size mismatches on {gr_l} (laptop)" + (f" and on {gr_h} (host rows: `gr_<d>_host.json`)" if gr_h else "") + ". "
            f"The engine self-test PASSED (`selftest.json`: cell list, G-F, G-D, the engine against a brute-force pool builder, G-X5 the budget-following walk against a per-node loop). "
            f"Every select re-read PL4's own pick from PL2's, PL3's and PL4's cells and got PL4's filed pick (`prev_pick_equals_pl4_filed` "
            f"{str(all(T.sel(d)['prev_pick_equals_pl4_filed'] for d in READ)).lower()} on the {NR} read). "
            f"Cost of the dev grids (sum of the three parts): {'; '.join(cost)}. Frozen confirmations: {', '.join(cf)}.")


# ---------------------------------------------------------------- per-dataset paragraphs
def para(ds):
    v = D(ds)
    if not has(ds):
        return f"- **{LABEL[ds]}**: not read in this filing."
    s = T.sel(ds)
    rn, ro, rp = pick(ds, "R95_all"), pick(ds, "R95_prev"), pick(ds, "R95_new")
    short = " SHORT on dev." if s["short"]["95"] else " reaches 0.95 on dev."
    lo, hi = v["all_diff_ci"]
    rlo, rhi = v["nodes_ratio_ci"]
    if v["same_cell"]:
        eq = (f"R95 of all cells is PL4's own cell, `{rn['name']}` (dev {rn['all']:.4f} at {rn['nodes']:,.1f}; eval frozen {v['new_all']:.4f} at {v['new_nodes']:,.1f}); no PL5 cell is a smaller or higher-ALL R95 pick. "
              f"The best of the PL5 cells alone is `{rp['name']}` (dev {rp['all']:.4f} at {rp['nodes']:,.1f}).")
    else:
        eq = (f"R95 of all cells: `{rn['name']}` (dev {rn['all']:.4f} at {rn['nodes']:,.1f}) against PL4's `{ro['name']}` (dev {ro['all']:.4f} at {ro['nodes']:,.1f}); "
              f"on eval frozen {v['new_all']:.4f} at {v['new_nodes']:,.1f} against {v['old_all']:.4f} at {v['old_nodes']:,.1f}: ALL {pts(v['all_diff'])} points [{pts(lo)}, {pts(hi)}], "
              f"nodes ratio {v['nodes_ratio']:.4f} [{rlo:.4f}, {rhi:.4f}], status **{v['status']}**.")
    r98 = v["r98"]
    r98s = (f"R98: `{r98['new']}` {r98['new_all']:.4f} at {r98['new_nodes']:,.1f} on eval ({'the same cell as PL4' if r98['same_cell'] else 'PL4: `' + r98['old'] + '` ' + format(r98['old_all'], '.4f') + ' at ' + format(r98['old_nodes'], ',.1f')}), "
            f"{'reaches' if r98['new_all'] >= 0.98 else 'does not reach'} 0.98 on eval, status {r98['status']}.")
    fr = v["frontier"]
    gains = [f"{b}: {pts(fr[b]['all_diff'])}" for b in T.BUDGETS if fr[b]["status"] != "NONE" and not fr[b]["same_cell"]]
    frs = (f"Frontier reading: **{v['frontier_status']}**" + (f" (budgets with a different cell, eval ALL difference in points: {'; '.join(gains)})" if gains else " (the same cell at every budget)") + ".")
    p95 = v["price"]["R95any"]
    price = (f"At any size, R95 is `{p95['new']}` {p95['new_all']:.4f} at {p95['new_nodes']:,.1f} nodes on eval ({'PL4 picks the same cell' if p95['same_cell'] else 'PL4: `' + p95['old'] + '` ' + format(p95['old_all'], '.4f') + ' at ' + format(p95['old_nodes'], ',.1f')}"
             f"{'' if p95['new_dev_reaches'] else '; SHORT on dev'}), status {p95['status']}.")
    return f"- **{LABEL[ds]}** ({s['questions']:,} dev questions;{short}) {eq} {r98s} {frs} {price}"


SPECIFIC = {}


def specific(ds):
    return SPECIFIC.get(ds, lambda: "")() if has(ds) else ""


# ---------------------------------------------------------------- D1 / D2 table
def miss_rows():
    d1, d2 = [], []
    for d in ORDER:
        p = PL5 / d / "miss.json"
        if not p.is_file():
            continue
        m = T.rj(p)
        t = m["totals"]
        st = m["D2"]["by_stage_hop"]
        d1.append(f"| {LABEL[d]} | `{m['cell']}` | {t['q_missing']} of {m['questions']} ({t['q_missing'] / m['questions']:.3f}) | {t['golds_missing']} ({t['golds_missing'] / max(1, t['q_missing']):.2f} per such question) | "
                  f"{t['R']} | {t['C']} | {t['U']} | {m['nodes_mean']:,.1f} |")
        d2.append(f"| {LABEL[d]} | {m['D2']['c_golds']} | {st.get('cap1', 0)} | {st.get('cap2', 0)} | {st.get('cap3', 0)} | {st.get('budget2', 0)} | {st.get('budget3', 0)} | {st.get('upstream', 0)} | "
                  f"{yn(m['D2']['trace_matches_engine'])} |")
    return "\n".join(d1), "\n".join(d2)


# ---------------------------------------------------------------- the page
def build(text_hook=None):
    page = PAGE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    cut = page.find("\n## 14. Results")
    head = (page[:cut] if cut >= 0 else page).rstrip("\n") + "\n"
    rep = (PL5 / "report.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    rep = re.sub(r"^### ", "#### ", rep, flags=re.M)
    counts = C["counts"]
    meet_frozen = [d for d in READ if D(d)["new_all"] >= 0.95]
    not_meet = [d for d in READ if D(d)["new_all"] < 0.95]
    d1rows, d2rows = miss_rows()
    above = [d for d in READ if D(d)["status"] == "ABOVE"]
    below = [d for d in READ if D(d)["status"] == "BELOW"]
    unread = [LABEL[d] for d in ORDER if d not in READ]
    over = [LABEL[d] for d in READ if (T.conf(d, "e3v") and T.conf(d, "e3v")["cells"][D(d)["new"]]["over_bound"])]
    title = (f"## 14. Results (26 Nov 2026; {NR} of 7 datasets read" + ("" if COMPLETE else f"; {', '.join(unread)} not yet read: the calls are INCOMPLETE") + ")")
    paras = "\n".join(para(d) + ("\n" + specific(d) if specific(d) else "") for d in ORDER)
    hot = "" if COMPLETE else ("\n**Not yet read: " + ", ".join(unread) + ".** " + ("HotpotQA's and 2Wiki's dev grids, selects and confirmations are host rows (`outputs/pl5/feed_rows.txt`); " if any(d in ("HotpotQA", "2Wiki") for d in unread) else "")
                               + "the calls are INCOMPLETE until all seven are read, and this section is re-filed when they return (as PL4's was at c25eddf and 18cccdc1).\n")
    call_words = (f"The old rule is {C['call']} and the frontier reading is {C['call_frontier']} (read {NR} of 7)")
    ft = text_hook(locals()) if text_hook else {}
    text = f"""
{title}

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl5/<d>/{{select,confirm_frozen,confirm_e3v,miss,place}}.json`, `outputs/pl5/call.json`, `outputs/pl5/gr_*.json`) by
`outputs/mp_unified/zpl5g.py call` and `outputs/mp_unified/zpl5r.py report` / `miss`; the tables of section 14.10 are pasted unchanged from `outputs/pl5/report.md`. The laptop datasets (MultiHop-RAG, SQuAD,
MuSiQue, MetaQA, WebQSP) ran on the laptop (dev grid, select, frozen confirmation; SQuAD's frozen confirmation in four question slices through `zpl5r.py confirm-par`, as PL4's); HotpotQA and 2Wiki (their dev grids, frozen
and e3v confirmations) are host rows. The e3v confirmation ran for MultiHop-RAG, HotpotQA and 2Wiki only, as the declaration says (section 11); none was declared for SQuAD, MuSiQue, MetaQA or WebQSP. Nothing was changed in the rule, the cells or the engine after a number was read.
{hot}
### 14.1 Gates and cost

{gates_text()}

### 14.2 The calls (read {NR} of 7: old rule {C['call']}; frontier reading {C['call_frontier']})

**The old rule, as PL3 and PL4 state it** (unchanged; section 6): per dataset on eval, frozen lists, NEW = R95 of the reading `all` (all 6,078 cells), OLD = PL4's own chosen cell, paired bootstrap of the ALL difference and of the
ratio of mean nodes. Clause by clause:

| dataset | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | BELOW: (1) NEW misses where OLD reaches / (2) ALL diff upper end < 0 with ratio lower end >= 1 | ABOVE: (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (ALL not BELOW) / (3) ALL diff lower end > 0 with ratio upper end <= 1 | status | chosen cell inside 2,500 (eval nodes) |
|---|---|---|---|---|---|---|---|
{T.clause_rows(C)}

- **BELOW: {len(below)}.** ABOVE: {len(above)}{' (' + ', '.join(LABEL[d] for d in above) + ')' if above else ''}; LEVEL: {NR - len(below) - len(above)} of the {NR} read.
- **The old call: {C['call']}** (BELOW {counts['old']['BELOW']}, ABOVE {counts['old']['ABOVE']}, LEVEL {counts['old']['LEVEL']}; read {NR} of 7). CARRY_PL5 needs all seven read, no BELOW, at least one ABOVE and every chosen cell inside the 2,500 bound{'' if COMPLETE else ': it cannot be called until ' + ', '.join(unread) + (' is' if len(unread) == 1 else ' are') + ' read'}.
{ft.get('above_note', '')}

**The frontier reading** (section 6; filed beside the old one, never in its place): per budget N, the highest dev ALL among cells of at most N dev nodes, all cells against PL4's cells; paired
bootstrap of the eval ALL difference (points; `=` marks the same cell on both sides, difference exactly 0):

| dataset | 500 | 1,000 | 1,500 | 2,000 | 2,500 | frontier status |
|---|---|---|---|---|---|---|
{T.frontier_rows(C)}

- **Frontier call: {C['call_frontier']}** (BELOW {counts['frontier']['BELOW']}, ABOVE {counts['frontier']['ABOVE']}, LEVEL {counts['frontier']['LEVEL']}; read {NR} of 7).{ft.get('frontier_note', '')}
- **The two calls are filed under their own names and this page does not choose between them** (section 6): which one governs is the user's decision. {call_words}.

The cells behind each frontier budget (dev ALL at dev nodes; eval ALL at eval nodes; the eval difference in points):

| dataset | budget | cells | dev | eval | difference |
|---|---|---|---|---|---|
{T.frontier_detail(C)}

**R98, beside the calls** (the same pairing at the 0.98 floor; decides nothing):

| dataset | NEW (R98 of all cells), eval | OLD (PL4's R98), eval | ALL difference, 95% CI | nodes ratio, 95% CI | status by the rule |
|---|---|---|---|---|---|
{T.r98_rows(C)}

### 14.3 What each dataset says (dev = selection carve, frozen lists; eval = confirmation)

The construction each dataset would get from R95 of the reading `all` (NEW), and PL4's (OLD), eval frozen ALL at mean nodes, and the e3v re-run of NEW (flagged OVER_BOUND above 2,500 nodes):

| dataset | NEW (eval frozen ALL at nodes) | OLD = PL4's (eval frozen ALL at nodes) | NEW meets 0.95 on eval, frozen | NEW on e3v (ALL at nodes) |
|---|---|---|---|---|
{T.cell_rows(C)}

**Which datasets meet the target on eval, frozen lists (ALL >= 0.95), of the {NR} read: {len(meet_frozen)}**: {', '.join(LABEL[d] for d in meet_frozen) or 'none'}. **They do not:** {', '.join(LABEL[d] + ' (' + format(D(d)['new_all'], '.4f') + ')' for d in not_meet) or 'none'}.
{('The e3v re-run of a chosen cell exceeds the 2,500 bound on: ' + ', '.join(over) + '.') if over else 'No chosen cell exceeds the bound on the e3v lists where those were read.'} The bound is read on the selection carve's frozen lists, as declared (section 6).

{paras}

### 14.4 What each option bought
{ft.get('bought', '')}

### 14.5 The price of the floor at any size (beside the calls; decides nothing)

A dataset that stays SHORT inside the bound may not stay short without it; this table says what the 0.95 floor and the 0.98 floor cost when the bound is lifted, and whether the PL5 cells make that cheaper (PL6 reads the same question with a ladder of visited budgets). Eval, frozen lists;
NEW = the smallest cell (dev) at the floor over all cells, OLD = the same over PL2-PL4's cells; `SHORT on dev` marks a floor that no cell reaches on dev (then the cell is the highest-ALL cell).

| dataset | floor | NEW (R at any size, all cells), eval | OLD (PL2-PL4's cells), eval | ALL difference, 95% CI | nodes ratio, 95% CI | status |
|---|---|---|---|---|---|---|
{T.price_rows(C)}

The any-size frontier (the highest dev ALL at most N dev nodes, N above the bound; eval ALL difference of all cells against PL2-PL4's, points):

| dataset | 3,000 | 4,000 | 5,000 | 6,000 | 7,000 | any-size frontier status |
|---|---|---|---|---|---|---|
{T.frontier_any_rows(C)}
{ft.get('price', '')}

### 14.6 What is still short, in numbers (diagnostics D1 and D2, post-hoc, labels read; 300 dev questions evenly spaced over the carve)

Only the SHORT datasets are diagnosed (the chosen R95 cell of the reading `all`; budget-following walks are traced by `zpl5g.ref_walk5`, the stage "cap" of a `bf` walk being the scoring window). **D1** (each gold missing from the pool: R ranked past M, C cut by the caps, the window or the order but inside the uncapped three-hop FULL reach of its seeds, U beyond it):

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (bound 2,500) |
|---|---|---|---|---|---|---|---|
{d1rows}

**D2** (which stage cut each C gold, from a traced walk of the cell; the trace equals the engine's walk on every question):

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|
{d2rows}
{ft.get('short', '')}

### 14.7 Beside the calls: the e3v lists (post-hoc; decides nothing)

NEW (R95 of all cells) and OLD (PL4's) re-run on the eval carve's e3v lists. The mean nodes are flagged OVER_BOUND when above 2,500 (the bound is read on the frozen lists; the flag changes neither call):

| dataset | NEW, e3v (ALL at nodes) | OLD = PL4's, e3v (ALL at nodes) | ALL difference | NEW meets 0.95 |
|---|---|---|---|---|
{T.e3v_rows(C)}

### 14.8 What the calls mean, and what they leave unread
{ft.get('meaning', '')}

### 14.9 The next changes (names only; none is adopted, none is in this round's grid; each is declared on its own page before any number)
{ft.get('next', '')}

### 14.10 The tables (the report, as of {NR} of 7 read)

{rep.strip(chr(10))}

### 14.11 Surprises and contradictions with filed numbers
{ft.get('surprises', '')}

### 14.12 Disclosed and undone
{ft.get('disclosed', '')}
"""
    out = head + text
    PAGE.write_bytes(out.replace("\r\n", "\n").encode("utf-8"))
    print(len(out.splitlines()), "lines;", NR, "read;", C["call"], C["call_frontier"], "->", PAGE)


if __name__ == "__main__":
    hook = None
    try:
        import pl5_text as TX  # noqa: E402  (the hand-written sentences, added after the numbers are read)
        hook = TX.hook
    except ImportError:
        pass
    build(hook)
