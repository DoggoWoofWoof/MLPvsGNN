"""ING1 phase A (docs/ING1_INGESTION_AND_STAGE_REGISTRY.md): read-only registry tools.

    python -I scripts/ingest/ingest.py check  [--ds NAME | --package DIR] [--deep]
    python -I scripts/ingest/ingest.py status [--json] [--host]
    python -I scripts/ingest/ingest.py plan   NAME

Reads configs/datasets.json and configs/rule_sets.json. Writes nothing and runs nothing but, with --host, one `rx exec` of
`status --json` on the host. No path is written in the code: the workspace is found from this file's own location.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

WS = Path(__file__).resolve().parents[2]
CFG = WS / "configs"


def load():
    ds = json.loads((CFG / "datasets.json").read_text(encoding="utf-8"))
    rs = json.loads((CFG / "rule_sets.json").read_text(encoding="utf-8"))
    return ds, rs, rs[rs["current"]]["stages"]


def candidates(cfg, root):
    out = []
    for c in cfg["roots"][root]:
        if c.startswith("$"):
            c = os.environ.get(c[1:], "")
            if not c:
                continue
        p = Path(c).expanduser()
        out.append(p if p.is_absolute() else (WS / p).resolve())
    return out


def package_dir(cfg, d):
    for base in candidates(cfg, d["root"]):
        p = base / d["dir"]
        if (p / "DATASET.json").is_file():
            return p
    return None


def sha(p, lf=False):
    b = Path(p).read_bytes()
    return hashlib.sha256(b.replace(b"\r\n", b"\n") if lf else b).hexdigest()


def rec_counts(rec):
    """Counts in the record, in either dialect (the original six, or the MultiHop-RAG / STaRK packages)."""
    n = rec.get("n_nodes")
    q = rec.get("n_queries")
    split = rec.get("queries_by_split")
    return (int(n) if n is not None else None, int(q) if q is not None else None, split if isinstance(split, dict) else None)


def count_lines(p):
    n = 0
    with open(p, "rb") as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                return n
            n += b.count(b"\n")


def check_package(pkg, deep=False, force_deep=False):
    """Gates that read only the package's own files. Each gate: (name, PASS | FAIL | N/A, detail)."""
    pkg = Path(pkg)
    g = []

    def add(name, ok, detail=""):
        g.append((name, "PASS" if ok is True else ("FAIL" if ok is False else "N/A"), detail))

    try:
        rec = json.loads((pkg / "DATASET.json").read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        return [("C1 record", "FAIL", f"DATASET.json unreadable: {e}")]
    add("C1 record", bool(rec.get("RECORD")), str(rec.get("RECORD")))
    n, q, split = rec_counts(rec)
    add("C2 counts in the record", n is not None and q is not None, f"nodes {n}, queries {q}")
    nodes = pkg / "nodes.jsonl"
    if nodes.is_file():
        add("C3 nodes file", True, f"{nodes.stat().st_size:,} bytes")
        if deep and n is not None:
            k = count_lines(nodes)
            add("C4 node lines equal the record", k == n, f"{k:,} lines, record {n:,}")
    else:
        add("C3 nodes file", None, "no nodes.jsonl at the package root (the record names its own layout)")
    qdir = pkg / "queries"
    if qdir.is_dir() and split:
        got = {}
        for sp in split:
            f = qdir / f"{sp}.jsonl"
            if f.is_file():
                got[sp] = count_lines(f) if deep else None
        add("C5 split files named by the record exist", set(got) == set(split), f"{sorted(got)} of {sorted(split)}")
        if deep:
            bad = {sp: (got[sp], int(split[sp])) for sp in got if got[sp] != int(split[sp])}
            add("C6 split line counts equal the record", not bad, str(bad) if bad else "all equal")
            gold, info, wanted = {}, [], set()
            for sp in got:
                n_q = withheld = empty = rep = 0
                with open(qdir / f"{sp}.jsonl", encoding="utf-8") as fh:
                    for ln in fh:
                        r = json.loads(ln)
                        gs = r.get("gold_node_ids")
                        n_q += 1
                        if gs is None:
                            continue
                        if len(gs) == 0:
                            empty += 1
                        if len(set(gs)) != len(gs):
                            rep += 1
                        wanted.update(gs)
                if n_q and empty == n_q:
                    withheld = 1
                info.append(f"{sp}: {'withheld (no answers)' if withheld else f'{empty} without a node answer'}, {rep} repeat a node")
            add("C7i gold lists recorded (not a gate: a split without answers, answers that are not nodes and repeats are properties of the data)",
                None, "; ".join(info))
            if nodes.is_file() and wanted:
                import re
                pat = re.compile(r'"node_id": "([^"]+)"')
                with open(nodes, encoding="utf-8") as fh:
                    for ln in fh:
                        m = pat.search(ln, 0, 4000)
                        if m is None:
                            m = pat.search(json.dumps(json.loads(ln)))
                        if m is not None:
                            wanted.discard(m.group(1))
                            if not wanted:
                                break
                add("C7 every gold id is a node id", not wanted, f"{len(wanted)} unresolved" if wanted else "all resolve")
    else:
        add("C5 split files", None, "no queries/ folder or no queries_by_split in the record")
    if not list(pkg.glob("embeddings/*/*/manifest.json")):
        add("C8 embedding manifests", None, "none in this package folder on this machine (the e3v arm lives under outputs/enc/e3v/<ds>/)")
    for kind in ("docs", "queries"):
        for man in sorted(pkg.glob(f"embeddings/*/{kind}/manifest.json")):
            try:
                m = json.loads(man.read_text(encoding="utf-8"))
                want = n if kind == "docs" else q
                rows = int(m.get("n_rows", -1))
                arm = man.parts[-3]
                add(f"C8 {arm} {kind} manifest rows", want is None or rows == want or rows == 0, f"{rows:,} rows, record {want}")
            except Exception as e:  # noqa: BLE001
                add(f"C8 {man.parts[-3]} {kind} manifest", False, str(e))
    edges = pkg / "graph" / "s_edges.npz"
    if not edges.is_file():
        fam = sorted(x.name for x in (pkg / "graph").glob("*.npz")) if (pkg / "graph").is_dir() else []
        add("C9 S edges in range", None, ("no graph/s_edges.npz; this package keeps its edges in graph/" + ", graph/".join(fam) +
                                         " (read by zworld, ING1 phase B)") if fam else "no graph folder: this package has no edge file")
    if edges.is_file() and n is not None:
        try:
            import numpy as np
            z = np.load(edges)
            src, dst = z["src"], z["dst"]
            add("C9 S edges in range", int(src.min()) >= 0 and int(dst.min()) >= 0 and int(max(src.max(), dst.max())) < n,
                f"{len(src):,} edges")
        except Exception as e:  # noqa: BLE001
            add("C9 S edges in range", None, f"not read here: {e}")
    freeze = None
    for cand in (pkg.parent / "PACKAGE_FREEZE.json", pkg.parent / "CANONICAL_FREEZE.json", pkg / "PACKAGE_FREEZE.json"):
        if cand.is_file():
            freeze = cand
            break
    if freeze is None:
        add("C10 freeze file names this record", None, "no freeze file beside the package")
    else:
        text = freeze.read_text(encoding="utf-8")
        hit = any(h in text for h in (sha(pkg / "DATASET.json"), sha(pkg / "DATASET.json", True)))
        add("C10 freeze file names this record", True if hit else None,
            f"{freeze.name}: sha found" if hit else f"{freeze.name}: the record's sha is not written in it (this freeze format does not carry it)")
    return g


def stage_cells(cfg, stages, d):
    """One dict of stage -> (state, note) for a dataset on THIS machine."""
    pkg = package_dir(cfg, d)
    rec = json.loads((pkg / "DATASET.json").read_text(encoding="utf-8")) if pkg else None
    n = rec_counts(rec)[0] if rec else None
    cells = {}
    cells["ingest"] = ("PRESENT", f"{n:,} nodes") if pkg else ("MISSING", "no package on this machine")
    kb = d["name"][len("stark_"):] if d["name"].startswith("stark_") else d["name"]
    for name, st in stages.items():
        if name == "ingest":
            continue
        arts = st.get("artifacts") or []
        if not arts:
            cells[name] = ("OPEN" if st["status"] == "OPEN" else "UNREGISTERED", "no record registered")
            continue
        found, notes, state = [], [], None
        for a in arts:
            if a.get("only") and d["name"] not in a["only"]:
                continue
            p = WS / a["template"].format(ds=d["name"], kb=kb)
            if not p.is_file():
                continue
            found.append(a["template"].split("/")[1] if a["template"].startswith("outputs/") else a["template"])
            try:
                r = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                r = {}
            if a.get("rows_equal") == "n_nodes" and n is not None and "n_rows" in r:
                notes.append("rows ok" if int(r["n_rows"]) == n else f"rows {r['n_rows']} != {n}")
            pg = r.get("page") or r.get("declared_in")
            if pg and st["status"] == "FINAL":
                state = "DONE" if any(Path(x).name in str(pg) for x in st["pages"]) else "STALE"
        if not found:
            applicable = any(not a.get("only") or d["name"] in a["only"] for a in arts)
            if st["status"] == "OPEN":
                cells[name] = ("OPEN", "no record found")
            elif applicable:
                cells[name] = ("MISSING", "no record at the registered path")
            else:
                cells[name] = ("UNREGISTERED", "no record registered for this dataset")
        else:
            cells[name] = ((state or "PRESENT") if st["status"] == "FINAL" else "OPEN", ", ".join(found + notes))
    return cells


def host_cells():
    """The host's cells for every dataset (one rx exec of `status --json`), or None."""
    r = subprocess.run([sys.executable, str(WS / "tools" / "rx" / "rx.py"), "exec", "--env", "mpr-cpu", "--timeout", "120", "--",
                        "python", "-I", "scripts/ingest/ingest.py", "status", "--json"], capture_output=True, text=True, timeout=300)
    line = [x for x in r.stdout.splitlines() if x.startswith("{")]
    if not line:
        print("host: no answer:", (r.stdout + r.stderr)[-300:])
        return None
    return json.loads(line[-1])


def cmd_status(a):
    cfg, rs, stages = load()
    names = list(stages)
    local = {d["name"]: stage_cells(cfg, stages, d) for d in cfg["datasets"]}
    if a.json:
        print(json.dumps(local))
        return 0
    host = host_cells() if a.host else None
    print(f"rule set {rs['current']}; stages: " + ", ".join(f"{k} [{v['status']}]" for k, v in stages.items()))
    print(f"cells: PRESENT = a record is there, DONE = its page is the pinned rule's, STALE = it names another page, MISSING = nothing at the "
          f"registered path, UNREGISTERED = no record path is registered, OPEN = the stage has no final rule, N/A\n")
    w = max(len(d["name"]) for d in cfg["datasets"]) + 2
    print("".ljust(w) + "".join(n[:14].ljust(16) for n in names))
    for d in cfg["datasets"]:
        row = d["name"].ljust(w)
        for n in names:
            st, note = local[d["name"]][n]
            if host and d["name"] in host:
                hs = host[d["name"]][n][0]
                tag = st if hs == st else f"{st[:7]}|H:{hs[:7]}"
                row += tag[:15].ljust(16)
            else:
                row += st[:15].ljust(16)
        print(row)
    if host:
        print("\nL = this machine; H = the host (shown only where it differs).")
    return 0


def cmd_check(a):
    cfg, _, _ = load()
    todo = []
    if a.package:
        todo.append((Path(a.package).name, Path(a.package)))
    else:
        for d in cfg["datasets"]:
            if a.ds and d["name"] != a.ds:
                continue
            p = package_dir(cfg, d)
            todo.append((d["name"], p))
    bad = 0
    for name, p in todo:
        if p is None:
            print(f"{name}: no package on this machine")
            continue
        res = check_package(p, deep=a.deep)
        f = sum(1 for _, s, _ in res if s == "FAIL")
        bad += f
        print(f"{name}: {sum(1 for _, s, _ in res if s == 'PASS')} pass, {f} fail, {sum(1 for _, s, _ in res if s == 'N/A')} n/a")
        for gname, s, det in res:
            if s != "PASS" or a.verbose:
                print(f"    {s:4} {gname}: {det}")
    return 1 if bad else 0


def cmd_plan(a):
    cfg, _, stages = load()
    d = next((x for x in cfg["datasets"] if x["name"] == a.name), None)
    if d is None:
        print(f"{a.name} is not in configs/datasets.json: add its row (name, kind, root, dir, record, s_source, adapter, arms) first, then its package "
              "must pass `check`. No stage command is registered for an unknown dataset.")
        return 2
    cells = stage_cells(cfg, stages, d)
    if a.host:
        h = host_cells()
        for n, (st, note) in ((h or {}).get(d["name"], {})).items():
            if st in ("PRESENT", "DONE") and cells.get(n, ("", ""))[0] not in ("PRESENT", "DONE"):
                cells[n] = (st, "on the host: " + note)
    print(f"{d['name']} ({d['kind']}, S: {d['s_source']}), adapter: {d['adapter']}")
    for n, (st, note) in cells.items():
        print(f"  {n:18} {st:12} {note}")
    cmds = d.get("commands")
    if not cmds:
        print("\nno stage commands are registered for this dataset: its stage drivers are not parametrised by the registry yet "
              "(ING1 phase B/C).")
        return 0
    print("\nrows for the stages that are not DONE or PRESENT:")
    for n, row in cmds.items():
        if cells.get(n, ("", ""))[0] not in ("DONE", "PRESENT"):
            print("  " + row)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--ds")
    c.add_argument("--package")
    c.add_argument("--deep", action="store_true")
    c.add_argument("--verbose", action="store_true")
    s = sub.add_parser("status")
    s.add_argument("--json", action="store_true")
    s.add_argument("--host", action="store_true")
    p = sub.add_parser("plan")
    p.add_argument("name")
    p.add_argument("--host", action="store_true")
    a = ap.parse_args(argv)
    return {"check": cmd_check, "status": cmd_status, "plan": cmd_plan}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
