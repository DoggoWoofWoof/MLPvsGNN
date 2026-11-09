"""SWP Part E (docs/SWP_S_TO_R_SWAP.md section 5f): the queue rows of Part E, made from Part C's last working rows.

Part E runs Part C's steps with arm NKF in place of SNKF and the N-world (nworld.py) in place of the S-world (sworld.py). This file reads the queue,
takes every Part C row (name swpc*), keeps for each step the row of its last generation and retry (swpcN- prefix, -rN suffix: the one that carries
the fixes of 8 Nov, e.g. the GPU shares after the out-of-memory retries), and writes the same step as a swpe-* row:
  - sworld.py -> nworld.py, zswpc.py -> zswpe.py, zswpt.py -> zswpe_t.py, zswpr.py -> zswpe_r.py, swpc_collect.py -> swpe_collect.py,
    outputs/abl/SNKF -> outputs/abl/NKF (in names, commands and inputs);
  - the new files are added to the inputs of every row (the feeder hashes them at send time);
  - dropped: the stamp rows (zswpt.py writes the stand-in read stamp itself since a6efadc, so a fresh fit needs none);
  - shared, not copied: the builds and gates of the read arms (zabl.py chain / gate, e.g. SNK on SQuAD, RK on MetaQA);
  - re-checked instead: a read arm built by zabl.py (SNK, NK, RK on SQuAD, MetaQA, WebQSP) loses its look and cache to its prune row once
    its own reads are filed (zabl.prune; on 8 Nov SNK/squad and RK/metaqa had to be rebuilt for Part C, swpc7-build-*). Each such cell gets
    swpe-build-ARM-DS (zabl's chain, which skips every filed step: seconds when nothing was pruned, the look and cache again when it was)
    and swpe-gate-ARM-DS, after Part C's collect row and after the cell's prune row, so no prune can come between it and Part E's reads;
    Part E's lists, columns and reads of the cell wait on that gate in place of the old build and gate rows;
  - every NKF chain waits on the bases row (zabl's chain copies the bases itself when they are missing: two copies at once would race);
  - the rows are written in a stable topological order (a row after the rows it waits on), checked for cycles, and every train-mode row is
    checked to come before the flip to read mode (swpe-mode-read) and every read-mode row after it.

    python outputs/host_ops/swpe_rows.py QUEUE_FILE OUT_FILE      # writes the rows (to be inserted by hand before a stuck head)
    python outputs/host_ops/swpe_rows.py selftest
"""
import re
import sys
from pathlib import Path

SWAP = (("outputs/host_ops/sworld.py", "outputs/host_ops/nworld.py"), ("outputs/mp_unified/zswpc.py", "outputs/mp_unified/zswpe.py"),
        ("outputs/mp_unified/zswpt.py", "outputs/mp_unified/zswpe_t.py"), ("outputs/mp_unified/zswpr.py", "outputs/mp_unified/zswpe_r.py"),
        ("outputs/host_ops/swpc_collect.py", "outputs/host_ops/swpe_collect.py"), ("outputs/abl/SNKF", "outputs/abl/NKF"))
NEW_FILES = ("outputs/host_ops/nworld.py", "outputs/mp_unified/zswpe.py", "outputs/mp_unified/zswpe_t.py", "outputs/mp_unified/zswpe_r.py",
             "outputs/host_ops/swpe_collect.py")
OLD_TOKENS = ("sworld.py", "zswpc.py", "zswpt.py", "zswpr.py", "swpc_collect.py", "SNKF")
NAME = re.compile(r"swpc(\d*)-(.+?)(?:-r(\d+))?")
CELL = re.compile(r"zswpe_r\.py (?:lists|cols|read) --arm (\S+)(?: --name \S+)? --dataset (\S+)")
FLIP = "swpe-mode-read"
BASES = "swpe-NKF-bases"
RUN = "python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py"


def ablreads(rows, arm, ds):
    """ABL's own read rows of one arm on one dataset (zabl.py read --arm A --name FIT --datasets D): zabl.prune removes a look and cache only
    when these reads are filed, so a cell without them is never pruned (its prune row ends rc 1)."""
    pat = re.compile(rf"zabl\.py read --arm {re.escape(arm)} --name \S+ --datasets {re.escape(ds)}(?:\s|$)")
    return [f for f in rows if pat.search(f[7])]


def zabl_rows(rows, verb, arm, ds):
    """The queue's zabl.py rows of one verb (chain, gate, prune) for one arm and one dataset alone, in file order."""
    flag = "--dataset" if verb == "chain" else "--datasets"
    pat = re.compile(rf"zabl\.py {verb} --arm {re.escape(arm)} {flag} {re.escape(ds)}(?:\s|$)")
    return [f for f in rows if pat.search(f[7])]


def parse(text):
    rows = []
    for ln in text.splitlines():
        ln = ln.rstrip("\r")
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("|")
        if f[0] != "line" or len(f) not in (8, 9):
            raise SystemExit(f"swpe_rows: not a queue row: {ln[:120]}")
        rows.append(f)
    return rows


def key_of(name):
    """(step, generation, retry) of a Part C row name, None for any other row."""
    m = NAME.fullmatch(name)
    if not m:
        return None
    return m.group(2), int(m.group(1) or 1), int(m.group(3) or 1)


def kind_of(f):
    cmd = f[7]
    if "stamp --name" in cmd:
        return "drop"
    if "zabl.py chain" in cmd or "zabl.py gate" in cmd:
        return "shared"
    return "map"


def swap(s):
    for a, b in SWAP:
        s = s.replace(a, b)
    return s


def make(rows):
    part_c = [f for f in rows if key_of(f[3])]
    best = {}
    for f in part_c:
        step, gen, rt = key_of(f[3])
        if kind_of(f) != "map":
            continue
        if step not in best or (gen, rt) >= best[step][0]:
            best[step] = ((gen, rt), f)
    kind = {f[3]: kind_of(f) for f in part_c}
    new_name = {f[3]: "swpe-" + key_of(f[3])[0].replace("SNKF", "NKF") for f in part_c if kind[f[3]] == "map"}
    trains = {}                                                   # fit name -> the Part E row that trains it
    for _rank, f in best.values():
        m = re.search(r"train --name (\S+)", f[7])
        if m:
            trains[m.group(1)] = new_name[f[3]]
    stamp_of = {}                                                 # a dropped stamp row -> the fit it marked as trained
    for f in part_c:
        m = re.search(r"stamp --name (\S+)", f[7])
        if kind[f[3]] == "drop" and m:
            stamp_of[f[3]] = m.group(1)
    out = []
    for step, (_rank, f) in sorted(best.items(), key=lambda kv: rows.index(kv[1][1])):
        deps = []
        for d in (x for x in f[2].split(",") if x):
            if d in new_name:
                deps.append(new_name[d])
            elif kind.get(d) == "drop":
                if d not in stamp_of or stamp_of[d] not in trains:
                    raise SystemExit(f"swpe_rows: {f[3]} waits on the stamp row {d}, whose fit no Part E row trains")
                deps.append(trains[stamp_of[d]])                  # the stamp stood for "that fit is trained": wait on its fit
            else:
                deps.append(d)                                    # a shared read-arm build, or a row outside Part C
        deps = list(dict.fromkeys(deps))
        inputs = [swap(x) for x in f[6].split(";") if x]
        inputs += [x for x in NEW_FILES if x not in inputs]
        g = f[8:9]
        out.append(["line", f[1], ",".join(deps), new_name[f[3]], f[4], f[5], ";".join(inputs), swap(f[7])] + g)
    names = {f[3] for f in out}
    for f in out:                                                 # every NKF chain after the bases row
        if "zswpe.py chain --dataset" in f[7] and BASES in names and BASES not in f[2].split(","):
            f[2] = ",".join([BASES] + [d for d in f[2].split(",") if d])
    collect_c = [best[s][1][3] for s in best if "swpc_collect.py" in best[s][1][7]]
    if len(collect_c) != 1:
        raise SystemExit(f"swpe_rows: Part C has {len(collect_c)} collect rows, not one")
    out = rebuilds(out, rows, collect_c[0])
    return ["|".join(f) for f in order(out)]


def rebuilds(out, rows, collect_c):
    """swpe-build-ARM-DS and swpe-gate-ARM-DS for every read cell zabl.py builds (and prunes); Part E's rows of the cell wait on the gate."""
    by_name = {f[3]: f for f in rows}
    cells = []
    for f in out:
        m = CELL.search(f[7])
        if m and m.groups() not in cells and zabl_rows(rows, "chain", *m.groups()):
            cells.append(m.groups())
    new = {}                                                      # cell -> its build and gate rows, written before the cell's first row
    for arm, ds in cells:
        chain = zabl_rows(rows, "chain", arm, ds)[-1]
        gates = zabl_rows(rows, "gate", arm, ds)
        prunes = zabl_rows(rows, "prune", arm, ds)
        if not ablreads(rows, arm, ds):                           # zabl.prune needs ABL's own reads of the arm: without them it removes
            prunes = []                                           # nothing and ends rc 1, and a failed row drops what waits on it
        lanes = re.search(r"--lanes (\d+)", chain[7])
        build, gate = f"swpe-build-{arm}-{ds}", f"swpe-gate-{arm}-{ds}"
        g = gates[-1] if gates else None
        new[(arm, ds)] = [
            ["line", chain[1], ",".join([collect_c] + [p[3] for p in prunes[-1:]]), build, chain[4], chain[5], chain[6],
             f"python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- {RUN} chain --arm {arm} --dataset {ds} "
             f"--lanes {lanes.group(1) if lanes else 3} --host"],
            ["line", chain[1], build, gate, g[4] if g else "2", g[5] if g else "8", g[6] if g else chain[6],
             f"{RUN} gate --arm {arm} --datasets {ds} --host"]]
        own = {x[3] for x in zabl_rows(rows, "chain", arm, ds) + gates}  # the old build and gate rows of the cell
        for f in out:
            m = CELL.search(f[7])
            if m and m.groups() == (arm, ds):
                deps = [d for d in f[2].split(",") if d and d not in own]
                f[2] = ",".join(dict.fromkeys(deps + [gate]))
    assert all(r[3] not in by_name for rs in new.values() for r in rs)
    res = []
    for f in out:
        m = CELL.search(f[7])
        if m and m.groups() in new:
            res += new.pop(m.groups())
        res.append(f)
    return res


def order(out):
    """A stable topological order of the Part E rows: the given order, except that a row the next one waits on comes just before it; a cycle
    stops."""
    by = {f[3]: f for f in out}
    need = {f[3]: [d for d in f[2].split(",") if d in by] for f in out}
    state, res = {}, []

    def visit(n, path):
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise SystemExit(f"swpe_rows: a dependency cycle: {' -> '.join(path + [n])}")
        state[n] = 1
        for d in need[n]:
            visit(d, path + [n])
        state[n] = 2
        res.append(by[n])

    for f in out:
        visit(f[3], [])
    return res


def check(out, rows):
    names = [r.split("|")[3] for r in out]
    assert len(set(names)) == len(names), "duplicate swpe names"
    known = set(names) | {f[3] for f in rows}
    for r in out:
        f = r.split("|")
        assert all(t not in f[7] for t in OLD_TOKENS), f"a Part C token is left in {f[3]}: {f[7][:160]}"
        assert all(t not in x for x in f[6].split(";") for t in OLD_TOKENS if t != "SNKF" and t not in ("sworld.py", "zswpt.py", "zswpr.py", "zswpc.py",
                                                                                                     "swpc_collect.py")), f[3]
        assert "outputs/abl/SNKF" not in f[6], f"{f[3]} still pins SNKF"
        unknown = [d for d in f[2].split(",") if d and d not in known]
        assert not unknown, f"{f[3]}: unknown deps {unknown}"
        mapped = {x[3] for x in rows if key_of(x[3]) and kind_of(x) == "map"}
        waits_c = [d for d in f[2].split(",") if d in mapped]
        assert not waits_c or (f[3].startswith("swpe-build-") and len(waits_c) == 1 and "swpc_collect.py" in
                               next(x for x in rows if x[3] == waits_c[0])[7]), f"{f[3]} still waits on a Part C row of its own steps: {waits_c}"
    fs = [r.split("|") for r in out]
    assert [f[3] for f in order([list(f) for f in fs])] == names, "the rows are not in a topological order"
    up = {f[3]: {d for d in f[2].split(",") if d in set(names)} for f in fs}

    def above(n, seen=None):
        seen = set() if seen is None else seen
        for d in up[n]:
            if d not in seen:
                seen.add(d)
                above(d, seen)
        return seen

    if FLIP in up:
        for f in fs:
            m = re.search(r"world\.py run --mode (\w+)", f[7])
            if m and m.group(1) == "train":
                assert FLIP not in above(f[3]), f"{f[3]} trains after the flip to read mode"
            if m and m.group(1) == "read":
                assert FLIP in above(f[3]), f"{f[3]} reads without waiting for the flip to read mode"
    for f in fs:
        m = CELL.search(f[7])
        if m and any(r[3] == f"swpe-gate-{m.group(1)}-{m.group(2)}" for r in fs):
            assert f"swpe-gate-{m.group(1)}-{m.group(2)}" in above(f[3]), f"{f[3]} does not wait on its cell's rebuild"
    return True


def main(argv):
    if argv[:1] == ["selftest"]:
        return selftest()
    if len(argv) != 2:
        raise SystemExit(__doc__)
    rows = parse(Path(argv[0]).read_text(encoding="utf-8"))
    out = make(rows)
    check(out, rows)
    head = ["# SWP Part E (docs/SWP_S_TO_R_SWAP.md section 5f): the pair trained with the slot empty (arm NKF, the N-world); made by "
            "outputs/host_ops/swpe_rows.py from Part C's last working rows"]
    Path(argv[1]).write_text("\n".join(head + out) + "\n", encoding="utf-8", newline="\n")
    gpu = sum(1 for r in out if r.split("|")[8:9] and r.split("|")[8].startswith("gpu"))
    print(f"swpe_rows: {len(out)} rows ({gpu} on the GPU) -> {argv[1]}")
    return 0


def selftest():
    q = "\n".join([
        "line|x||swpc-SNKF-squad|6|28|outputs/mp_unified/zswpc.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zswpc.py chain --dataset squad --host",
        "line|x||swpc-SNKF-bases|1|2|outputs/mp_unified/zswpc.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zswpc.py bases --host",
        "line|x|swpc-SNKF-squad|swpc-world-make|1|2|outputs/host_ops/sworld.py;outputs/abl/SNKF|python outputs/host_ops/sworld.py make",
        "line|x|swpc-world-make|swpc-fit-zspk-s0|1.1|7|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpt.py train --name scr-ml1-zspk-s0|gpu:0.29",
        "line|x|swpc-world-make|swpc4-stamp-zspk-s0|1|3|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpt.py stamp --name scr-ml1-zspk-s0",
        "line|x|swpc-fit-zspk-s0,swpc4-stamp-zspk-s0|swpc3-fit-zg6r-s0-r4|1.1|13|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python x --frac 0.84 -- outputs/mp_unified/zswpt.py train --name scr-ml1-zg6r-s0|gpu:0.85",
        "line|x|swpc-fit-zspk-s0,swpc4-stamp-zspk-s0|swpc3-fit-zg6r-s0-r3|1.1|13|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python x --frac 0.72 -- outputs/mp_unified/zswpt.py train --name scr-ml1-zg6r-s0|gpu:0.75",
        "line|x|swpc-world-make|swpc2-fit-zspk-s0-hp|1.1|7|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpt.py train --name scr-ml1-zspk-s0-hp|gpu:0.5",
        "line|x|swpc-world-make|swpc4-stamp-zspk-s0-hp|1|3|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpt.py stamp --name scr-ml1-zspk-s0-hp",
        "line|x|swpc4-stamp-zspk-s0-hp|swpc4-fit-zg6r-s0-hp|1.1|13|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpt.py train --name scr-ml1-zg6r-s0-hp|gpu:0.85",
        "line|x||swpc7-build-SNK-squad|8|38|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py chain --arm SNK --dataset squad --lanes 4 --host",
        "line|x|swpc7-build-SNK-squad|swpc7-gate-SNK-squad|2|8|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py gate --arm SNK --datasets squad --host",
        "line|x|abl-x|swp6-SNK-prune-squad|1|2|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py prune --arm SNK --datasets squad",
        "line|x||swp-SNK-read-squad-zg6r-s0|1|6|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py read --arm SNK --name scr-ml1-zg6r-s0 --datasets squad",
        "line|x||swp3-NK-squad|12|42|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py chain --arm NK --dataset squad --lanes 6 --host",
        "line|x|swp3-NK-squad|swp5-NK-squad-prune|1|2|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py prune --arm NK --datasets squad",
        "line|x|swpc-world-make|swpc5-NK-squad-lists|2|9|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpr.py lists --arm NK --dataset squad --host",
        "line|x||abl-SNK-squadx|8|38|outputs/mp_unified/zabl.py|python outputs/host_ops/pylib_run.py outputs/mp_unified/zabl.py chain --arm SNK --dataset squadx --host",
        "line|x|swpc-world-make,swpc7-gate-SNK-squad|swpc7-SNK-squad-lists-r2|2|9|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode train -- python outputs/mp_unified/zswpr.py lists --arm SNK --dataset squad --host",
        "line|x|swpc3-fit-zg6r-s0-r4,swpc7-SNK-squad-lists-r2,swpc5-NK-squad-lists|swpc5-mode-read-r2|1|2|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py mode read",
        "line|x|swpc3-fit-zg6r-s0-r4,swpc5-mode-read-r2,swpc7-gate-SNK-squad,abl2-other|swpc5-rd-SNK-squad-gnx-s0-r2|1.1|6|outputs/host_ops/sworld.py|python outputs/host_ops/sworld.py run --mode read -- python outputs/mp_unified/zswpr.py read --arm SNK --name scr-gnx-nx-s0 --dataset squad|gpu:0.29",
        "line|x|swpc5-rd-SNK-squad-gnx-s0-r2|swpc8-collect|1|2|outputs/host_ops/swpc_collect.py|python outputs/host_ops/swpc_collect.py",
        "line|x||abl2-other|1|1|x.py|python x.py",
    ])
    rows = parse(q)
    out = make(rows)
    check(out, rows)
    by = {r.split("|")[3]: r.split("|") for r in out}
    assert set(by) == {"swpe-NKF-bases", "swpe-NKF-squad", "swpe-world-make", "swpe-fit-zspk-s0", "swpe-fit-zg6r-s0", "swpe-rd-SNK-squad-gnx-s0",
                       "swpe-collect", "swpe-fit-zspk-s0-hp", "swpe-fit-zg6r-s0-hp", "swpe-SNK-squad-lists", "swpe-mode-read",
                       "swpe-build-SNK-squad", "swpe-gate-SNK-squad", "swpe-NK-squad-lists", "swpe-build-NK-squad", "swpe-gate-NK-squad"}, sorted(by)
    assert "--frac 0.84" in by["swpe-fit-zg6r-s0"][7] and by["swpe-fit-zg6r-s0"][8] == "gpu:0.85"      # the last retry, not the last line
    assert by["swpe-fit-zg6r-s0"][2] == "swpe-fit-zspk-s0"                                               # the stamp dependency becomes its fit's
    assert by["swpe-fit-zg6r-s0-hp"][2] == "swpe-fit-zspk-s0-hp"                                         # a row that waited on the stamp alone
    assert by["swpe-NKF-squad"][2] == "swpe-NKF-bases"                                                    # the chain waits on the bases copy
    assert by["swpe-build-SNK-squad"][2] == "swpc8-collect,swp6-SNK-prune-squad"                         # after Part C and after the prune
    assert by["swpe-build-NK-squad"][2] == "swpc8-collect"                       # NK has no ABL reads: its prune ends rc 1, never waited on
    assert by["swpe-NK-squad-lists"][2] == "swpe-world-make,swpe-gate-NK-squad"
    assert "zabl.py chain --arm SNK --dataset squad --lanes 4 --host" in by["swpe-build-SNK-squad"][7] and by["swpe-build-SNK-squad"][4:6] == ["8", "38"]
    assert by["swpe-gate-SNK-squad"][2] == "swpe-build-SNK-squad" and by["swpe-gate-SNK-squad"][4:6] == ["2", "8"]
    assert by["swpe-SNK-squad-lists"][2] == "swpe-world-make,swpe-gate-SNK-squad"                        # old gate replaced by the rebuild's
    assert by["swpe-rd-SNK-squad-gnx-s0"][2] == "swpe-fit-zg6r-s0,swpe-mode-read,abl2-other,swpe-gate-SNK-squad"   # outside rows kept
    assert "nworld.py run --mode read" in by["swpe-rd-SNK-squad-gnx-s0"][7] and "zswpe_r.py read" in by["swpe-rd-SNK-squad-gnx-s0"][7]
    assert by["swpe-world-make"][6].split(";")[:2] == ["outputs/host_ops/nworld.py", "outputs/abl/NKF"]
    assert "zswpe.py chain --dataset squad" in by["swpe-NKF-squad"][7] and by["swpe-collect"][7] == "python outputs/host_ops/swpe_collect.py"
    assert all(set(NEW_FILES) <= set(r[6].split(";")) for n, r in by.items() if not n.startswith(("swpe-build-", "swpe-gate-")))
    names = [r.split("|")[3] for r in out]
    assert names.index("swpe-NKF-bases") < names.index("swpe-NKF-squad") < names.index("swpe-world-make")
    assert names.index("swpe-build-SNK-squad") < names.index("swpe-gate-SNK-squad") < names.index("swpe-SNK-squad-lists")
    assert names.index("swpe-build-NK-squad") < names.index("swpe-gate-NK-squad") < names.index("swpe-NK-squad-lists")
    base = [r.split("|") for r in out]                            # the checks stop a train row after the flip, a read row before it, a cycle
    for (name, dep), why in ((("swpe-fit-zg6r-s0-hp", "swpe-mode-read"), "trains after the flip"),
                             (("swpe-rd-SNK-squad-gnx-s0", ""), "reads without waiting"),
                             (("swpe-world-make", "swpe-fit-zspk-s0"), "cycle")):
        fs = [list(f) for f in base]
        for f in fs:
            if f[3] == name:
                f[2] = ",".join([dep] + f[2].split(",")) if dep else "swpe-fit-zg6r-s0,swpe-gate-SNK-squad"
        stopped = None
        try:
            check(["|".join(f) for f in order(fs)], rows)
        except (AssertionError, SystemExit) as e:
            stopped = str(e)
        assert stopped and why in stopped, (why, stopped)
    print("swpe_rows self-test: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
