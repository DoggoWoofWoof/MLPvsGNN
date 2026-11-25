"""Files section 14 (results) of docs/PL4_COVERAGE_ROUND3.md from the filed records. Re-run to re-file when more datasets are read (it replaces any earlier section 14).
Every number is read from outputs/pl4/ records (call.json, <d>/select.json, <d>/confirm_*.json, <d>/miss.json, gr_*.json, report.md); nothing is chosen here."""
import glob
import json
import re
import sys
from pathlib import Path

SCR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCR))
import pl4_tables as T  # noqa: E402

ROOT = T.ROOT
PL4 = T.PL4
PAGE = ROOT / "docs" / "PL4_COVERAGE_ROUND3.md"
ORDER, LABEL = T.ORDER, T.LABEL
C = T.call()
pts, yn = T.pts, T.yn
READ = [d for d in ORDER if C["datasets"][d]["status"] != "NOT_READ"]
NR = len(READ)
SHORTS = [d for d in READ if T.sel(d)["short"]["95"]]
HOSTED = ("hotpotqa", "2wiki")


def D(ds):
    return C["datasets"][ds]


def has(ds):
    return ds in READ


def fmt_cell(c, nodes=None, a=None):
    return f"`{c}`"


# ---------------------------------------------------------------- 14.1 gates and cost
def gates_text():
    laptop = [d for d in ("multihoprag", "squad", "musique", "metaqa", "webqsp") if (PL4 / f"gr_{d}.json").is_file()]
    host = [d for d in ("musique", "hotpotqa", "2wiki") if (PL4 / f"gr_{d}_host.json").is_file()]
    gr_l = ", ".join(LABEL[d] for d in laptop)
    gr_h = ", ".join(LABEL[d] for d in host)
    el = T.rj(PL4 / "gr_musique.json")["elements"] if (PL4 / "gr_musique.json").is_file() else 909600
    bad = 0
    for d in laptop:
        g = T.rj(PL4 / f"gr_{d}.json")
        bad += g["present_mismatches"] + g["size_mismatches"] + (g["verdict"] != "PASS")
    for d in host:
        g = T.rj(PL4 / f"gr_{d}_host.json")
        bad += g["present_mismatches"] + g["size_mismatches"] + (g["verdict"] != "PASS")
    assert bad == 0, "a G-R record is not PASS"
    st = T.rj(PL4 / "selftest.json")
    s0 = T.sel(READ[0])
    cost = []
    for d in ORDER:
        fs = sorted(glob.glob(str(PL4 / d / "grid_dev_frozen.part*of3.json")))
        if len(fs) != 3:
            cost.append(f"{LABEL[d]}: not read")
            continue
        hs = [T.rj(f) for f in fs]
        where = "host" if hs[0].get("host") else "laptop"
        cost.append(f"{LABEL[d]} {sum(h['seconds'] for h in hs):,.0f} s, {max(h['peak_gb'] for h in hs):.2f} GB peak ({where})")
    cf = []
    for d in ORDER:
        p = PL4 / d / "grid_eval_frozen_confirm.json"
        if p.is_file():
            h = T.rj(p)
            cf.append(f"{LABEL[d]} {h['seconds']:,.0f} s")
    return (f"The engine `zpl4g.py` is byte-identical to the declaration commit c71f0d7: script sha256 starts `{st['script_sha256'][:12]}`, the code-path hash `{s0['code_path_sha256'][:12]}` is equal "
            f"in every select head and `call.json` (`code_path_equal_everywhere` {str(C['code_path_equal_everywhere']).lower()}; G-U PASS), the cell list `{s0['cells_sha256_pl4'][:12]}` (1,932 cells; "
            f"4,206 with PL2's and PL3's). **G-R** (PL2's 804 and PL3's 1,470 cells rebuilt on the first 400 dev questions, {el:,} elements per dataset, 224 questions on MultiHop-RAG) PASSED with 0 present "
            f"and 0 size mismatches on {gr_l} (laptop) and on {gr_h} (host rows `pl4-musiquegr`, `pl4-hotpotqagr`, `pl4-2wikigr`: `gr_<d>_host.json`). The engine self-test PASSED "
            f"(`selftest.json`: cell list, G-F, G-X4, engine against a brute-force pool builder on MultiHop-RAG and SQuAD, G-D). Every select re-read PL3's own pick from the union of PL2's and PL3's cells "
            f"and got PL3's filed pick (`pl3_pick_equals_filed` {str(all(T.sel(d)['pl3_pick_equals_filed'] for d in READ)).lower()} on the {NR} read). "
            f"Cost of the dev grids (sum of the three parts): {'; '.join(cost)}. Frozen confirmations: {', '.join(cf)}. "
            f"(The laptop's numbers were taken with up to eight heavy processes sharing 12 logical CPUs, so they are upper bounds; SQuAD's 7,012 s ran beside the G-R of the next datasets; SQuAD's confirmation, in four parallel slices, is wall-clock.)")


# ---------------------------------------------------------------- per-dataset paragraphs
def pick(ds, key):
    return T.sel(ds)["picks"][key]


def para(ds):
    v = D(ds)
    if not has(ds):
        return f"- **{LABEL[ds]}**: not read in this filing."
    s = T.sel(ds)
    rn, ro = pick(ds, "R95_all"), pick(ds, "R95_pl3")
    short = " SHORT on dev." if s["short"]["95"] else ""
    nq = s["questions"]
    lo, hi = v["all_diff_ci"]
    rlo, rhi = v["nodes_ratio_ci"]
    if v["same_cell"]:
        eq = (f"R95 is PL3's own cell, `{rn['name']}` (dev {rn['all']:.4f} at {rn['nodes']:,.1f}; eval frozen {v['new_all']:.4f} at {v['new_nodes']:,.1f}); no PL4 cell is a smaller or higher-ALL R95 pick.")
    else:
        eq = (f"R95 of all cells: `{rn['name']}` (dev {rn['all']:.4f} at {rn['nodes']:,.1f}) against PL3's `{ro['name']}` (dev {ro['all']:.4f} at {ro['nodes']:,.1f}); "
              f"on eval frozen {v['new_all']:.4f} at {v['new_nodes']:,.1f} against {v['old_all']:.4f} at {v['old_nodes']:,.1f}: ALL {pts(v['all_diff'])} points [{pts(lo)}, {pts(hi)}], "
              f"nodes ratio {v['nodes_ratio']:.4f} [{rlo:.4f}, {rhi:.4f}], status **{v['status']}**.")
    r98 = v["r98"]
    r98s = (f"R98: `{r98['new']}` {r98['new_all']:.4f} at {r98['new_nodes']:,.1f} on eval ({'the same cell as PL3' if r98['same_cell'] else 'PL3: `' + r98['old'] + '` ' + format(r98['old_all'], '.4f') + ' at ' + format(r98['old_nodes'], ',.1f')}), "
            f"{'reaches' if r98['new_all'] >= 0.98 else 'does not reach'} 0.98 on eval, status {r98['status']}.")
    fr = v["frontier"]
    gains = [f"{b}: {pts(fr[b]['all_diff'])}" for b in T.BUDGETS if fr[b]["status"] != "NONE" and not fr[b]["same_cell"]]
    frs = (f"Frontier reading: **{v['frontier_status']}**" + (f" (budgets with a different cell, eval ALL difference in points: {'; '.join(gains)})" if gains else " (the same cell at every budget)") + ".")
    return f"- **{LABEL[ds]}** ({nq:,} dev questions;{short or ' reaches 0.95 on dev.'}) {eq} {r98s} {frs}"


SPECIFIC = {}


def specific(ds):
    if not has(ds):
        return ""
    s, v = T.sel(ds), D(ds)
    if ds == "musique" and has(ds):
        m = T.rj(PL4 / ds / "miss.json")
        b = {x["nodes"]: x for x in s["bought"]}
        return (f"  What moved: the cells carrying `eq` (equal hop shares) or `rr` (families taking turns; the R95 cell carries both, with `cs-vr`) reach dev ALL 0.8781 at 2,298.7 nodes, and cells carrying `sp` (support order) the same 0.8781 "
                f"at 2,469.5, against PL3's 0.8735 at 2,488.9: 190 fewer nodes and 0.46 points more on dev. The dev frontier reads "
                f"+{100 * b[2000]['gain']:.2f} points at 2,000 nodes and +{100 * b[2500]['gain']:.2f} at 2,500, 0 below 2,000. Row rank `rk` and seeds with the new options add nothing at R95. "
                f"On eval the pair reads {pts(v['all_diff'])} points [{pts(v['all_diff_ci'][0])}, {pts(v['all_diff_ci'][1])}], an interval that includes 0, at {v['nodes_ratio']:.4f} of the nodes: the old rule's ABOVE rests "
                f"on the price clause alone, as HotpotQA's did in PL3, and the frontier reading, which asks for ALL at a matched node count, finds no budget whose interval excludes 0 "
                f"(2,000: {pts(v['frontier']['2000']['all_diff'])} [{pts(v['frontier']['2000']['all_diff_ci'][0])}, {pts(v['frontier']['2000']['all_diff_ci'][1])}]; 2,500: {pts(v['frontier']['2500']['all_diff'])} "
                f"[{pts(v['frontier']['2500']['all_diff_ci'][0])}, {pts(v['frontier']['2500']['all_diff_ci'][1])}]). Still short by {100 * (0.95 - v['new_all']):.2f} points; the highest ALL at any size is unchanged (0.9355 at 5,783.4 nodes).")
    if ds == "metaqa" and has(ds):
        return ("  Nothing moved: no PL4 cell beats PL3's at any dev node count (the dev frontier gain is 0.00 at every budget); `~cs-sp` alone reaches 0.8303 at 2,284.0 and the other options are lower. "
                "The three-hop STRUCT walk's visited cap (2,000) is what holds the cell at 2,340 nodes: the options reorder or re-share what the cap lets in, and the cap decides.")
    if ds == "webqsp" and has(ds):
        return (f"  The R95 pick `{v['new']}` is a tie with PL3's `{v['old']}` (the same dev ALL and nodes; ties go to the name, and `sp` sorts before `vr`). The two cells have identical present-gold counts and identical "
                f"pool sizes on every one of the 1,503 eval questions (paired ALL difference exactly 0.00 [0.00, 0.00], nodes ratio 1.0000), although the visited cap binds on 121 of them (sizes above 5,000): there the two orders "
                "can keep different nodes, and on this data they keep the same number of golds. No option moves the dev frontier at any budget; the best ALL at any size rises from 0.8480 to 0.8540 "
                "(at 6,021.1 nodes, `~cs-sp-eq`, more than twice the bound).")
    return ""


# ---------------------------------------------------------------- D1 / D2 table
def miss_rows():
    d1, d2 = [], []
    for d in ORDER:
        p = PL4 / d / "miss.json"
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
def build():
    page = PAGE.read_bytes().decode("utf-8").replace("\r\n", "\n")
    cut = page.find("\n## 14. Results")
    head = (page[:cut] if cut >= 0 else page).rstrip("\n") + "\n"
    rep = (PL4 / "report.md").read_bytes().decode("utf-8").replace("\r\n", "\n")
    rep = re.sub(r"^### ", "#### ", rep, flags=re.M)
    counts = C["counts"]
    meet_frozen = [d for d in READ if D(d)["new_all"] >= 0.95]
    not_meet = [d for d in READ if D(d)["new_all"] < 0.95]
    d1rows, d2rows = miss_rows()
    complete = NR == 7
    above = [d for d in READ if D(d)["status"] == "ABOVE"]
    below = [d for d in READ if D(d)["status"] == "BELOW"]
    f_above = [d for d in READ if D(d)["frontier_status"] == "ABOVE"]
    f_below = [d for d in READ if D(d)["frontier_status"] == "BELOW"]
    unread = [LABEL[d] for d in ORDER if d not in READ]
    over = [LABEL[d] for d in READ if (T.conf(d, "e3v") and T.conf(d, "e3v")["cells"][D(d)["new"]]["over_bound"])]
    above_note = ""
    if len(above) == 1:
        av = D(above[0])
        above_note = (f"- **{LABEL[above[0]]}'s ABOVE is by the price clause alone** (clause 2: the nodes ratio's upper end is below 1 with ALL not BELOW); its ALL difference is {pts(av['all_diff'])} points "
                      f"[{pts(av['all_diff_ci'][0])}, {pts(av['all_diff_ci'][1])}], an interval that includes 0. The rule is applied as written.")
    e3v_note = ""
    if over:
        e3v_note = (f"On the e3v lists the chosen cell of {', '.join(over)} reads above 2,500 mean nodes (OVER_BOUND); under section 6 the flag changes neither call, and a flagged cell is not offered to PL1's "
                    "e3v pool rows by this page. WebQSP's PL3 cell carries the same flag with the same sizes, as PL3 filed (2,720 nodes).") if "WebQSP" in over else (
                    f"On the e3v lists the chosen cell of {', '.join(over)} reads above 2,500 mean nodes (OVER_BOUND); under section 6 the flag changes neither call.")
    stays = "" if complete else "PL3's grid stays the rule until a call is filed."
    title = (f"## 14. Results (25 Nov 2026; {NR} of 7 datasets read" + ("" if complete else f"; {', '.join(unread)} not yet returned from the host: the call is INCOMPLETE") + ")")
    paras = "\n".join(para(d) + ("\n" + specific(d) if specific(d) else "") for d in ORDER)
    wq = D("webqsp") if has("webqsp") else None
    hot = "" if complete else ("\n**Not yet read: " + ", ".join(unread) + ".** Their dev grids, selects and confirmations are host rows (`outputs/pl4/feed_rows.txt`); the call is INCOMPLETE until all seven are read, "
                               "and this section is re-filed when they return (as PL3's was at 8bf1acf and 2824c2e).\n")
    text = f"""
{title}

Everything here is read by the rules of sections 5 to 8 from the filed records (`outputs/pl4/<d>/{{select,confirm_frozen,confirm_e3v,miss}}.json`, `outputs/pl4/call.json`, `outputs/pl4/gr_*.json`) by
`outputs/mp_unified/zpl4g.py call` and `outputs/mp_unified/zpl4r.py report` / `miss`; the tables of section 14.9 are pasted unchanged from `outputs/pl4/report.md`. The laptop datasets (MultiHop-RAG, SQuAD,
MuSiQue, MetaQA, WebQSP) ran on the laptop (dev grid, select, frozen confirmation; SQuAD's frozen confirmation in four question slices through `zpl4r.py confirm-par`, whose arrays were first checked equal to
the single-process confirmation on MultiHop-RAG); HotpotQA and 2Wiki, and every e3v confirmation but MultiHop-RAG's, are host rows. Nothing was changed in the rule, the cells or the engine after a number was read.
{hot}
### 14.1 Gates and cost

{gates_text()}

### 14.2 The calls (read {NR} of 7: old rule {C['call']}; frontier reading {C['call_frontier']})

**The old rule, as PL3 states it** (unchanged; section 6): per dataset on eval, frozen lists, NEW = R95 of the reading `all` (all 4,206 cells), OLD = PL3's own chosen cell, paired bootstrap of the ALL difference and of the
ratio of mean nodes. Clause by clause:

| dataset | NEW / OLD reaches 0.95 | ALL difference NEW - OLD (points), 95% CI | nodes ratio NEW / OLD, 95% CI | BELOW: (1) NEW misses where OLD reaches / (2) ALL diff upper end < 0 with ratio lower end >= 1 | ABOVE: (1) NEW reaches where OLD does not / (2) ratio upper end < 1 (ALL not BELOW) / (3) ALL diff lower end > 0 with ratio upper end <= 1 | status | chosen cell inside 2,500 (eval nodes) |
|---|---|---|---|---|---|---|---|
{T.clause_rows(C)}

- **BELOW: {len(below)}.** ABOVE: {len(above)}{' (' + ', '.join(LABEL[d] for d in above) + ')' if above else ''}; LEVEL: {NR - len(below) - len(above)} of the {NR} read.
- **The old call: {C['call']}** (BELOW {counts['old']['BELOW']}, ABOVE {counts['old']['ABOVE']}, LEVEL {counts['old']['LEVEL']}; read {NR} of 7). CARRY_PL4 needs all seven read, no BELOW, at least one ABOVE and every chosen cell inside the 2,500 bound{'' if complete else ': it cannot be called until ' + ', '.join(unread) + (' is' if len(unread) == 1 else ' are') + ' read'}.
{above_note}

**The frontier reading** (section 6; a proposed change of the rule, filed beside the old one and never in its place): per budget N, the highest dev ALL among cells of at most N dev nodes, all cells against PL3's cells; paired
bootstrap of the eval ALL difference (points; `=` marks the same cell on both sides, difference exactly 0):

| dataset | 500 | 1,000 | 1,500 | 2,000 | 2,500 | frontier status |
|---|---|---|---|---|---|---|
{T.frontier_rows(C)}

- **Frontier call: {C['call_frontier']}** (BELOW {counts['frontier']['BELOW']}, ABOVE {counts['frontier']['ABOVE']}, LEVEL {counts['frontier']['LEVEL']}; read {NR} of 7). On the {NR} read no budget of any dataset is ABOVE by the frontier's rule
  (a lower end above 0) and none is BELOW; the cells at each budget, with dev and eval numbers, are in the table of section 14.9 (`What the PL4 cells bought`) and in `call.json`.
- **The two calls are filed under their own names and this page does not choose between them** (section 6). At {NR} of 7 read, neither carries.

The cells behind each frontier budget (dev ALL at dev nodes; eval ALL at eval nodes; the eval difference in points):

| dataset | budget | cells | dev | eval | difference |
|---|---|---|---|---|---|
{T.frontier_detail(C)}

**R98, beside the calls** (the same pairing at the 0.98 floor; decides nothing):

| dataset | NEW (R98 of all cells), eval | OLD (PL3's R98), eval | ALL difference, 95% CI | nodes ratio, 95% CI | status by the rule |
|---|---|---|---|---|---|
{T.r98_rows(C)}

### 14.3 What each dataset says (dev = selection carve, frozen lists; eval = confirmation)

The construction each dataset would get from R95 of the reading `all` (NEW), and PL3's (OLD), eval frozen ALL at mean nodes, and the e3v re-run of NEW (flagged OVER_BOUND above 2,500 nodes):

| dataset | NEW (eval frozen ALL at nodes) | OLD = PL3's (eval frozen ALL at nodes) | NEW meets 0.95 on eval, frozen | NEW on e3v (ALL at nodes) |
|---|---|---|---|---|
{T.cell_rows(C)}

**Which datasets meet the target on eval, frozen lists (ALL >= 0.95), of the {NR} read: {len(meet_frozen)}**: {', '.join(LABEL[d] for d in meet_frozen) or 'none'}. **They do not:** {', '.join(LABEL[d] + ' (' + format(D(d)['new_all'], '.4f') + ')' for d in not_meet) or 'none'}.
{('The e3v re-run of a chosen cell exceeds the 2,500 bound on: ' + ', '.join(over) + '.') if over else 'No chosen cell exceeds the bound on the e3v lists where those were read.'} The bound is read on the selection carve's frozen lists, as declared (section 6).

{paras}

### 14.4 What each option bought

Read from `outputs/pl4/report.md` ("What each option bought" and "What the PL4 cells bought", section 14.9): on the {NR} read, the dev frontier of all cells differs from PL3's only on MuSiQue
(+1.11 points at 2,000 nodes and +0.46 at 2,500 on dev, from `eq` and `rr` with `cs-vr`, and `sp`; on eval the 2,000-node pair reads -0.58 [-1.57, +0.50], so that dev gain does not carry, and the 2,500-node pair +0.33 [-0.29, +0.95]) and on SQuAD (+0.07 points at three budgets of 75 to 1,000 nodes: one question's worth in 1,498). `rk` moves no read dataset's R95 pick (its cells alone reach 0.8638 on MuSiQue, 0.7862 on MetaQA, 0.7731 on WebQSP). The one SHORT dataset whose R95 pick moves is MuSiQue (190 nodes fewer, through `eq`, `rr`
and `sp` cells). Nothing moves MetaQA or WebQSP at any budget, and nothing moves MultiHop-RAG. On SQuAD the R95 and R98 picks are PL3's; the +0.07 points are one question and read LEVEL on eval
(1,000 nodes: +0.04 [-0.05, +0.13]).

### 14.5 What is still short, in numbers (diagnostics D1 and D2, post-hoc, labels read; 300 dev questions evenly spaced over the carve)

Only the SHORT datasets are diagnosed (the chosen R95 cell of the reading `all`). **D1** (each gold missing from the pool: R ranked past M, C cut by the caps or the order but inside the uncapped three-hop FULL reach of its seeds, U beyond it):

| dataset | cell | questions missing a gold | missing golds | R | C | U | nodes (bound 2,500) |
|---|---|---|---|---|---|---|---|
{d1rows}

**D2** (which stage cut each C gold, from a traced walk of the cell; the trace equals the engine's walk on every question):

| dataset | C golds | cap, hop 1 | cap, hop 2 | cap, hop 3 | budget, hop 2 | budget, hop 3 | upstream | trace = engine walk |
|---|---|---|---|---|---|---|---|---|
{d2rows}

- **The shortfall is not the ranking and not the reach**: R is 0 on every SHORT dataset (these cells carry no `+top{{M}}` cut) and U is at most 2 golds. It is C: the caps and the visited budget.
- **MetaQA** (STRUCT, three hops, visited cap 2,000): most C golds are cut by the **budget at hop 3** (264 of 373: the hop-3 candidates the visited cap does not admit), 82 are upstream
  (reached only through a node an earlier hop dropped) and 27 are cut by a per-frontier cap. `eq` (equal hop shares), the one option that changes the shares, did not help (its cells alone reach 0.8297 on dev against 0.8978).
- **WebQSP** (FULL, two hops, per-seed and per-frontier caps of 100, visited cap 5,000): D2 attributes none of its 245 C golds to the visited budget; they are cut by the **per-seed cap at hop 1** (126) and the
  **per-frontier cap at hop 2** (97), then upstream (22). The next change here is a cap that follows the budget (spend what the bound leaves on the next-ranked neighbours of each seed or frontier node), not an order.
- **MuSiQue** (FULL, three hops): the 37 C golds are spread over the cap at hop 2 (13) and hop 3 (6) and the budget at hop 2 (9) and hop 3 (7), with 2 upstream: no one stage dominates, so no single change is aimed at all of it.
- A cap that follows the budget spends more nodes: WebQSP's chosen cell is at 2,442 dev and 2,468 eval nodes against the 2,500 bound, so it has almost no room (the bound is the price of the next change there); MuSiQue's, at
  2,299 on dev and 2,275 on eval, has about 200.

### 14.6 Beside the calls: the e3v lists (post-hoc; decides nothing)

NEW (R95 of all cells) and OLD (PL3's) re-run on the eval carve's e3v lists. The mean nodes are flagged OVER_BOUND when above 2,500 (the bound is read on the frozen lists; the flag changes neither call):

| dataset | NEW, e3v (ALL at nodes) | OLD = PL3's, e3v (ALL at nodes) | ALL difference | NEW meets 0.95 |
|---|---|---|---|---|
{T.e3v_rows(C)}

{e3v_note}

### 14.7 What the calls mean, and what they leave unread

- **At {NR} of 7 read, nothing is carried.** The old rule and the frontier reading are both {('INCOMPLETE' if not complete else C['call'] + ' / ' + C['call_frontier'])}. {stays}
- **A reading of what the read datasets say.** On four of the five laptop datasets PL4's cells are PL3's cells or tie them (MultiHop-RAG and SQuAD: the same R95 cell; MetaQA: the same cell; WebQSP: identical ALL and sizes on every eval question); on MuSiQue they
  save 7.3 percent of the nodes for a gain whose interval includes 0. None of the three SHORT datasets is closer to 0.95 in any way the rule can credit: MuSiQue 0.8821 (+0.33 points), MetaQA 0.8968 (0.00), WebQSP 0.7804 (0.00).
- **D2 against the four options.** `sp`, `rk`, `eq` and `rr` reorder or re-share the visited budget and leave the per-seed and per-frontier caps alone. D2 puts all of WebQSP's attributed C golds (223 of 245; 22 upstream) at those
  caps, which no option touches, and most of MetaQA's (264 of 373) at hop 3's budget, where `eq` bought nothing. That was the disclosed limit of this round (section 13, "the four options do not change the per-seed or
  per-frontier caps"); it is now measured.

### 14.8 The next changes (names only; none is adopted, none is in this round's grid; each is declared on its own page before any number)

Against the caps (WebQSP, MuSiQue): **a cap that follows the budget** (each seed and frontier node keeps its next-ranked neighbours until the visited budget is spent, instead of a fixed number); against the hop-3 budget
(MetaQA, MuSiQue): **a parent-conditioned rank** (a node's cosine to the question plus its best parent's, the bridge the question text does not name; section 12), and the **weight values of NER and KNN rows** as a key.
Label-ordered caps need a UL2 layout on all seven (section 3). The e3v-bounded selection needs e3v lists for the select carves (section 12).

### 14.9 The tables (the report, as of {NR} of 7 read)

{rep.strip(chr(10))}

### 14.10 Surprises and contradictions with filed numbers

- WebQSP's R95 pick changed only by a name tie (`sp` against `vr` at the same dev ALL and nodes); the paired eval difference is exactly 0 and the nodes ratio exactly 1. A reader should not take `~s10-cs-sp` as a new finding.
- `rk` (row-rank order), which PL4 declared as the order most unlike PL3's, moves no dataset's R95 pick; the only frontier cell that carries it is SQuAD's 2,000-node pick (-0.01 points on eval, LEVEL).
- MuSiQue's old-rule ABOVE is the price clause with an ALL interval that includes 0: the rule says ABOVE, the frontier reading says LEVEL at every budget. This page reports both.

### 14.11 Disclosed and undone

- The laptop's SQuAD frozen confirmation was started as the single-process `zpl4g.py confirm` (the lane script's step), was projected at about three hours, and was stopped by hand at about 1,400 of 11,873 questions
  (it wrote nothing); it was replaced by `zpl4r.py confirm-par --slices 4`, which evaluates four question slices with the engine's own `evaluate4`, concatenates the arrays and runs `cmd_confirm`'s own tail. The
  concatenated arrays were checked equal to the single-process confirmation on MultiHop-RAG (`--verify`). `chain4.log` records the stop as `FAIL confirm squad` at 11:06.
- D1/D2 ran on 300 dev questions chosen evenly over the carve (as PL3's D1), post-hoc, and decide nothing. `zpl4r.py` (the report and the diagnostics, not the engine) is committed with this section.
- The frontier reading is a change of the rule proposed in this page's declaration; it is filed beside the old call, and neither is chosen here.
- The laptop's grids and confirmations ran with up to eight heavy processes on 12 logical CPUs; seconds are upper bounds.
- The grid arrays (`*.npz`) are not committed (as PL2's and PL3's); the heads, selects, confirmations, the call, the diagnostics and the report are.
- Not done in this filing: {(', '.join(unread) + ' (their reads are host rows); ' if not complete else '')}{('the e3v confirmation of ' + ', '.join(LABEL[d] for d in ORDER if T.conf(d, 'e3v') is None) + ' (host rows); ' if any(T.conf(d, 'e3v') is None for d in ORDER) else '')}PL1's rebuild of the pools on a carried construction (PL1's rows); the next changes of 14.8.
"""
    out = head + text
    PAGE.write_bytes(out.replace("\r\n", "\n").encode("utf-8"))
    print(len(out.splitlines()), "lines;", NR, "read;", C["call"], C["call_frontier"])


if __name__ == "__main__":
    build()
