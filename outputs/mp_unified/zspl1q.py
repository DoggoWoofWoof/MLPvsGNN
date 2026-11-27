"""SPL1 amendment A13 (docs/SPL1_SPARSE_LEG_SCREEN.md section 8.14): the pool coverage of the filed pool cells with the lever l5's SPLADE list in place of the record's,
the read that A11's D2 names when the list agreement is below 99 percent of the questions, and the rule D4.

It runs zspl1p.py (A8 to A10) unedited, in this process, with two module values changed: the folder that holds the leg's list (SPL) and the folder the reads are
written to (OUT), and the page reference (DOC). Everything else is the filed engine: the frozen dense list, graph, vectors, eval carve, cells, gate G-P1, paired
bootstrap and per-cell calls (A8: BETTER, LEVEL at -1.0 point, WORSE, MORE_NODES). In zspl1p's files and keys the label `c3` stands for the list read here (l5).

  selftest                          toy checks (the links, the module values, D4)
  prep                              hard-links outputs/spl1/lev/l5/D/top1000.npz as outputs/spl1/lev5view/c3/D/top1000.npz for the three datasets
  pool --ds D [--part I --of K]     zspl1p's `pool`, reading the l5 list, writing to outputs/spl1/pool_l5
  merge --ds D --of K               zspl1p's `merge`
  read                              D4 over the three datasets -> outputs/spl1/pool_l5/call.json and a table on stdout

Usage (laptop): python -E -P -B outputs/mp_unified/zspl1q.py pool --ds multihoprag
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
import zspl1p as Pm  # noqa: E402

DOC = "docs/SPL1_SPARSE_LEG_SCREEN.md section 8.14 (A13)"
ARM = "l5"
ROOT = Pm.P2.ROOT
SRC = ROOT / "outputs" / "spl1" / "lev" / ARM
VIEW = ROOT / "outputs" / "spl1" / "lev5view"
OUT = ROOT / "outputs" / "spl1" / "pool_l5"
DATASETS = Pm.DATASETS
DECIDING = ("frontier_2500", "R95_bounded")
POINT_MIN = -0.5
ok_calls = ("BETTER", "LEVEL")


def link_lists(src_root, view_root, datasets):
    """Hard links (a copy where the volume refuses) of src_root/D/top1000.npz at view_root/c3/D/top1000.npz; returns the paths made."""
    made = []
    for ds in datasets:
        s = Path(src_root) / ds / "top1000.npz"
        if not s.is_file():
            raise SystemExit(f"zspl1q: {s} is not here (fetch outputs/spl1/lev/{ARM}/{ds}/top1000.npz first)")
        d = Path(view_root) / "c3" / ds / "top1000.npz"
        d.parent.mkdir(parents=True, exist_ok=True)
        if d.exists():
            if d.stat().st_size != s.stat().st_size:
                raise SystemExit(f"zspl1q: {d} exists and differs from {s}")
        else:
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)
        made.append(d)
    return made


def d4(deciding):
    """D4 of section 8.14. deciding: list of (dataset, role, call, point difference). LEVER_POOL_OK if every one is BETTER or LEVEL (A8's calls) and its point
    difference is at least -0.5; else LEVER_POOL_FLAG."""
    bad = [(ds, role, call, round(d, 3)) for ds, role, call, d in deciding if call not in ok_calls or d < POINT_MIN]
    return ("LEVER_POOL_OK" if not bad else "LEVER_POOL_FLAG"), bad


def cmd_read():
    rows, deciding = [], []
    for ds in DATASETS:
        p = OUT / f"{ds}.json"
        if not p.is_file():
            raise SystemExit(f"zspl1q: {p} is not here: the merge of {ds} has not run")
        r = json.loads(p.read_text(encoding="utf-8"))
        if not r.get("g_p1", {}).get("pass"):
            raise SystemExit(f"zspl1q: {ds}: G-P1 did not pass")
        for role, name in r["roles"].items():
            c = r["cells"][name]
            rows.append((ds, role, c))
            if role in DECIDING:
                deciding.append((ds, role, c["call"], c["diff_pt"]))
    dec, bad = d4(deciding)
    lines = ["| dataset | cell role | record all-gold | l5 all-gold | diff pt [95% interval] | call | nodes record -> l5 |", "|---|---|---|---|---|---|---|"]
    for ds, role, c in rows:
        mark = " **(deciding)**" if role in DECIDING else ""
        lines.append(f"| {ds} | {role}{mark} | {c['c0_all']:.4f} | {c['c3_all']:.4f} | {c['diff_pt']:+.2f} [{c['lo_pt']:+.2f}, {c['hi_pt']:+.2f}] | {c['call']} | "
                     f"{c['c0_nodes']:.1f} -> {c['c3_nodes']:.1f} |")
    out = {"stage": "SPL1 A13 pool read", "declared_in": DOC, "arm": ARM, "decision": dec, "flagged": bad, "deciding": [list(x) for x in deciding]}
    Pm.P2.wjson(OUT / "call.json", out)
    print("\n".join(lines))
    print(f"\nD4: {dec}" + (f"; flagged: {bad}" if bad else ""))
    return 0


def selftest():
    bad = 0

    def chk(name, ok):
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        bad += 0 if ok else 1

    import numpy as np
    chk("the module values are set before zspl1p runs", (Pm.SPL, Pm.OUT, Pm.DOC) == (VIEW, OUT, DOC))
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        for ds in DATASETS:
            (t / "src" / ds).mkdir(parents=True)
            np.savez(t / "src" / ds / "top1000.npz", ids=np.arange(6).reshape(2, 3), rows=np.arange(2))
        made = link_lists(t / "src", t / "view", DATASETS)
        same = all(np.array_equal(np.load(m)["ids"], np.load(t / "src" / m.parent.name / "top1000.npz")["ids"]) for m in made)
        chk("the links read the list's own arrays, at SPL/c3/D/top1000.npz", same and all(m.parts[-3] == "c3" for m in made))
        made2 = link_lists(t / "src", t / "view", DATASETS)
        chk("running the links again changes nothing", made2 == made)
    cells = [("a", "frontier_2500", "LEVEL", -0.3), ("a", "R95_bounded", "BETTER", 0.2), ("b", "frontier_2500", "LEVEL", 0.0)]
    chk("D4: all LEVEL or BETTER and none below -0.5 -> OK", d4(cells)[0] == "LEVER_POOL_OK")
    chk("D4: a point difference of -0.6 flags", d4(cells + [("c", "R95_bounded", "LEVEL", -0.6)])[0] == "LEVER_POOL_FLAG")
    chk("D4: a WORSE call flags", d4(cells + [("c", "R95_bounded", "WORSE", -1.4)])[0] == "LEVER_POOL_FLAG")
    chk("D4: MORE_NODES flags", d4(cells + [("c", "R95_bounded", "MORE_NODES", 0.1)])[0] == "LEVER_POOL_FLAG")
    if bad:
        raise SystemExit(f"zspl1q: selftest FAILED ({bad})")
    print("zspl1q selftest PASS")
    return 0


def main():
    Pm.SPL, Pm.OUT, Pm.DOC = VIEW, OUT, DOC
    OUT.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "selftest":
        return selftest()
    if cmd == "prep":
        for m in link_lists(SRC, VIEW, DATASETS):
            print("linked", m.relative_to(ROOT).as_posix())
        return 0
    if cmd == "read":
        return cmd_read()
    if cmd in ("pool", "merge"):
        return Pm.main()                                           # argparse reads sys.argv: `pool --ds D ...` as for zspl1p
    raise SystemExit("usage: zspl1q.py selftest | prep | pool --ds D [--part I --of K] | merge --ds D --of K | read")


if __name__ == "__main__":
    sys.exit(main())
