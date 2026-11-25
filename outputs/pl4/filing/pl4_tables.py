"""Tables of PL4 section 14 from the filed records (outputs/pl4/call.json, <d>/select.json, <d>/confirm_*.json, <d>/miss.json). Imported by the page builder.
Nothing here chooses or re-reads a number: it formats what the engine and zpl4r.py filed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PL4 = ROOT / "outputs" / "pl4"
ORDER = ["multihoprag", "squad", "hotpotqa", "2wiki", "musique", "metaqa", "webqsp"]
LABEL = {"multihoprag": "MultiHop-RAG", "squad": "SQuAD", "hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue", "metaqa": "MetaQA", "webqsp": "WebQSP"}
BUDGETS = ["500", "1000", "1500", "2000", "2500"]


def rj(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def pts(x):
    return f"{100 * x:+.2f}"


def yn(b):
    return "yes" if b else "no"


def call():
    return rj(PL4 / "call.json")


def sel(ds):
    return rj(PL4 / ds / "select.json")


def conf(ds, first):
    p = PL4 / ds / f"confirm_{first}.json"
    return rj(p) if p.is_file() else None


def clause_rows(C):
    rows = []
    for ds in ORDER:
        v = C["datasets"][ds]
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[ds]} | not read | | | | | **NOT READ** | |")
            continue
        lo, hi = v["all_diff_ci"]
        rlo, rhi = v["nodes_ratio_ci"]
        b1 = (not v["new_reaches"]) and v["old_reaches"]
        b2 = (hi < 0) and (rlo >= 1)
        a1 = v["new_reaches"] and not v["old_reaches"]
        a2 = rhi < 1
        a3 = (lo > 0) and (rhi <= 1)
        st = "BELOW" if (b1 or b2) else ("ABOVE" if (a1 or a2 or a3) else "LEVEL")
        assert st == v["status"], (ds, st, v["status"])
        same = " (the same cell)" if v["same_cell"] else ""
        rows.append(f"| {LABEL[ds]} | {yn(v['new_reaches'])} / {yn(v['old_reaches'])} | {pts(v['all_diff'])} [{pts(lo)}, {pts(hi)}] | {v['nodes_ratio']:.4f} [{rlo:.4f}, {rhi:.4f}] | "
                    f"{yn(b1)} / {yn(b2)} | {yn(a1)} / {yn(a2)} / {yn(a3)} | **{st}**{same} | {yn(v['within_bound'])} ({v['new_nodes']:,.1f}) |")
    return "\n".join(rows)


def cell_rows(C):
    """The construction each dataset gets (R95 of `all`) against PL3's, eval frozen and e3v."""
    rows = []
    for ds in ORDER:
        v = C["datasets"][ds]
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[ds]} | not read | | | |")
            continue
        cf, ce = conf(ds, "frozen"), conf(ds, "e3v")
        e3 = "not read"
        if ce is not None:
            n = ce["cells"][v["new"]]
            e3 = f"{n['all']:.4f} at {n['nodes']:,.1f}" + (" **OVER_BOUND**" if n["over_bound"] else "")
        rows.append(f"| {LABEL[ds]} | `{v['new']}` {v['new_all']:.4f} at {v['new_nodes']:,.1f} | `{v['old']}` {v['old_all']:.4f} at {v['old_nodes']:,.1f} | "
                    f"{'yes' if v['new_all'] >= 0.95 else 'no (short by %.2f points)' % (100 * (0.95 - v['new_all']))} | {e3} |")
    return "\n".join(rows)


def frontier_rows(C):
    rows = []
    for ds in ORDER:
        v = C["datasets"][ds]
        if v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[ds]} | not read | | | | | **NOT READ** |")
            continue
        fr = v["frontier"]
        cells = []
        for b in BUDGETS:
            f = fr[b]
            if f["status"] == "NONE":
                cells.append("no cell")
                continue
            lo, hi = f["all_diff_ci"]
            cells.append(f"{pts(f['all_diff'])} [{pts(lo)}, {pts(hi)}] {f['status']}" + (" =" if f["same_cell"] else ""))
        rows.append(f"| {LABEL[ds]} | " + " | ".join(cells) + f" | **{v['frontier_status']}** |")
    return "\n".join(rows)


def frontier_detail(C):
    """Per dataset and budget: the two cells (name, dev ALL at dev nodes -> eval ALL at eval nodes)."""
    lines = []
    for ds in ORDER:
        if C["datasets"][ds]["status"] == "NOT_READ":
            continue
        fr = C["datasets"][ds]["frontier"]
        for b in BUDGETS:
            f = fr[b]
            if f["status"] == "NONE":
                continue
            if f["same_cell"]:
                lines.append(f"| {LABEL[ds]} | {b} | `{f['new']}` (both) | {f['new_dev_all']:.4f} at {f['new_dev_nodes']:,.1f} | {f['new_all']:.4f} at {f['new_nodes']:,.1f} | same cell |")
            else:
                lines.append(f"| {LABEL[ds]} | {b} | NEW `{f['new']}`; PL3 `{f['old']}` | {f['new_dev_all']:.4f} at {f['new_dev_nodes']:,.1f}; {f['old_dev_all']:.4f} at {f['old_dev_nodes']:,.1f} | "
                             f"{f['new_all']:.4f} at {f['new_nodes']:,.1f}; {f['old_all']:.4f} at {f['old_nodes']:,.1f} | {pts(f['all_diff'])} {f['status']} |")
    return "\n".join(lines)


def r98_rows(C):
    rows = []
    for ds in ORDER:
        if C["datasets"][ds]["status"] == "NOT_READ":
            rows.append(f"| {LABEL[ds]} | not read | | | | |")
            continue
        r = C["datasets"][ds]["r98"]
        lo, hi = r["all_diff_ci"]
        rlo, rhi = r["nodes_ratio_ci"]
        rows.append(f"| {LABEL[ds]} | `{r['new']}` {r['new_all']:.4f} at {r['new_nodes']:,.1f} | `{r['old']}` {r['old_all']:.4f} at {r['old_nodes']:,.1f} | {pts(r['all_diff'])} [{pts(lo)}, {pts(hi)}] | "
                    f"{r['nodes_ratio']:.4f} [{rlo:.4f}, {rhi:.4f}] | {r['status']} |")
    return "\n".join(rows)


def e3v_rows(C):
    rows = []
    for ds in ORDER:
        ce = conf(ds, "e3v")
        v = C["datasets"][ds]
        if ce is None or v["status"] == "NOT_READ":
            rows.append(f"| {LABEL[ds]} | not read | | | |")
            continue
        n, o = ce["cells"][v["new"]], ce["cells"][v["old"]]
        rows.append(f"| {LABEL[ds]} | {n['all']:.4f} at {n['nodes']:,.1f}{' **OVER_BOUND**' if n['over_bound'] else ''} | {o['all']:.4f} at {o['nodes']:,.1f}{' **OVER_BOUND**' if o['over_bound'] else ''} | "
                    f"{pts(n['all'] - o['all'])} | {'yes' if n['all'] >= 0.95 else 'no'} |")
    return "\n".join(rows)


if __name__ == "__main__":
    C = call()
    print(C["call"], C["call_frontier"], C["counts"])
