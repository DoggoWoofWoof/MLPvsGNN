"""Design look (untracked; not a result and not filed): a label-free SEMB switch, read offline from filed rows.

lean_read9 reads each saved lean model twice on the same rows, with SEMB on (N) and off (N~-SEMB). qfam9 gives each
row's familiarity to the models' training queries, plus thresholds tau_p from the training graphs' select carves. This
look pairs them row by row: the switched read takes the on path's row where fam >= tau_p and the off path's row
elsewhere. Nothing is fitted or scored here. Every number is a mean over rows the jobs filed.

pairs      every N with an N~OFF partner in the --reads files (OFF = --off, default -SEMB), plus --pair ON|OFF for
           partners filed under other names or in different files.
alignment  per carve, the rrf@, twin0@ and gnn0@ rows must be equal across the --reads files, and fam@ must have their
           length. The carve's row and chunk counts in the jobs' JSONs must agree, and each rows file must hash to the
           sha256 its JSON records.
reads      per pair, carve and tau_p, the switched read's mean plus paired bootstrap differences (lean_mlp.boot_diff,
           the reads' own) against rrf alone, the on path, the off path and the twin. Also the share of rows switched
           off, abs (R@5 over the GNN's R@5) and rho ((x - twin) / (GNN - twin), on R@5).
bound      per pair and carve, the better of on and off by mean R@5. It is picked with the carve's labels, so it bounds
           any per-graph switch and is never a result.

    python outputs/mp_unified/pair9.py --reads outputs/mp_unified/lean/r9-2w-pick.json \\
        --fam outputs/mp_unified/lean/fam9-2w.json --out outputs/mp_unified/lean/pair9-2w-pick.json
    python outputs/mp_unified/pair9.py --selftest
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import lean_mlp as LM  # noqa: E402

REFS = ("rrf", "twin0", "gnn0")
SEP = "~"


def log(msg):
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_read(p):
    """(the job's JSON, {(name, carve key): rows}) for one lean_read9 / lean_mlp8 --out JSON and its rows file."""
    p = Path(p)
    res = json.loads(p.read_text(encoding="utf-8"))
    rf = p.parent / res["rows_file"]["path"]
    if sha(rf) != res["rows_file"]["sha256"]:
        raise SystemExit(f"{rf}: sha256 is not the one {p.name} records")
    out = {}
    with np.load(rf) as z:
        for k in z.files:
            name, key = k.rsplit("@", 1)
            out[(name, key)] = z[k].astype(np.float64)
    return res, out


def merge(reads):
    """All files' rows in one map. A reference row set (rrf, twin0, gnn0) filed by two jobs must be equal; a model name
    filed twice on one carve is refused."""
    rows, carves = {}, {}
    for p, (res, arr) in reads.items():
        for (name, key), v in arr.items():
            if (name, key) in rows:
                if name not in REFS:
                    raise SystemExit(f"{name}@{key} is filed by two --reads files")
                if not np.array_equal(rows[(name, key)], v):
                    raise SystemExit(f"{name}@{key} differs between --reads files: their rows are not aligned")
                continue
            rows[(name, key)] = v
        for key, inf in res["read_rows"].items():
            c = {"rows": inf["rows"], "chunks": inf.get("chunks")}
            if key in carves and carves[key] != c:
                raise SystemExit(f"{key}: the --reads files read different rows ({carves[key]} against {c})")
            carves[key] = c
    return rows, carves


def load_fam(p):
    p = Path(p)
    res = json.loads(p.read_text(encoding="utf-8"))
    ff = p.parent / res["fam_file"]["path"]
    if sha(ff) != res["fam_file"]["sha256"]:
        raise SystemExit(f"{ff}: sha256 is not the one {p.name} records")
    with np.load(ff) as z:
        fam = {k.split("@", 1)[1]: z[k].astype(np.float64) for k in z.files}
    return res, fam


def fam_carve(fres, key):
    for role in ("read", "held"):
        if f"{role} {key}" in fres["carves"]:
            return fres["carves"][f"{role} {key}"]
    return None


def find_pairs(rows, off_label, extra):
    names = {n for n, _ in rows if n not in REFS}
    pairs = []
    for n in sorted(names):
        if SEP in n and n.split(SEP, 1)[1] == off_label and n.split(SEP, 1)[0] in names:
            pairs.append((n.split(SEP, 1)[0], n))
    for spec in [s for s in extra.split(";") if s.strip()] if extra else []:
        on, off = spec.split("|")
        if on not in names or off not in names:
            raise SystemExit(f"--pair {spec}: {sorted({on, off} - names)} not in the --reads files")
        pairs.append((on, off))
    if not pairs:
        raise SystemExit(f"no pairs: no name has an {SEP}{off_label} partner and no --pair was given")
    return pairs


def rel(x, twin, gnn):
    """abs and rho on R@5 (column 0)."""
    d = gnn[0] - twin[0]
    return {"abs": float(x[0] / gnn[0]) if gnn[0] > 0 else None,
            "rho": float((x[0] - twin[0]) / d) if abs(d) > 1e-12 else None}


def run(a):
    reads = {p: load_read(p) for p in [s for s in a.reads.split(",") if s]}
    rows, carves = merge(reads)
    fres, fam = load_fam(a.fam)
    taus = {t: v for t, v in fres["tau"].items() if not a.taus or t in a.taus.split(",")}
    pairs = find_pairs(rows, a.off, a.pair)
    rng = np.random.default_rng(20261004)
    out = {"look": "pair9", "args": vars(a), "tau": taus, "pairs": [list(p) for p in pairs], "carves": {}, "reads": {},
           "bound": {}, "inputs": {str(p): sha(p) for p in list(reads) + [a.fam]},
           "script_sha256": sha(__file__)}
    keys = sorted({k for _, k in rows})
    for key in keys:
        if key not in fam:
            log(f"{key}: no fam@ rows in {a.fam}; skipped")
            continue
        refs = {nm: rows.get((nm, key)) for nm in REFS}
        if any(v is None for v in refs.values()):
            raise SystemExit(f"{key}: a reference row set is missing ({[n for n, v in refs.items() if v is None]})")
        f = fam[key]
        n = refs["rrf"].shape[0]
        if f.shape[0] != n:
            raise SystemExit(f"{key}: fam@ has {f.shape[0]} rows, the reads {n}: not the same rows")
        fc = fam_carve(fres, key)
        if fc is None or fc["rows"] != carves[key]["rows"] or (carves[key]["chunks"] is not None
                                                                and fc["chunks_read"] != carves[key]["chunks"]):
            raise SystemExit(f"{key}: the familiarity job read other rows ({fc} against {carves[key]})")
        means = {nm: v.mean(0) for nm, v in refs.items()}
        out["carves"][key] = {"rows": n, "refs": {nm: [float(x) for x in m] for nm, m in means.items()},
                              "below": {t: float((f < v).mean()) for t, v in taus.items()}}
        log(f"{key}: {n} rows, rrf {np.round(means['rrf'], 4).tolist()} twin0 {np.round(means['twin0'], 4).tolist()} "
            f"gnn0 {np.round(means['gnn0'], 4).tolist()}; off share {out['carves'][key]['below']}")
        for on, off in pairs:
            if (on, key) not in rows or (off, key) not in rows:
                continue
            ron, roff = rows[(on, key)], rows[(off, key)]
            mon, moff = ron.mean(0), roff.mean(0)
            pick = "on" if mon[0] >= moff[0] else "off"
            out["bound"][f"{on}@{key}"] = {"picks": pick, "mean": [float(x) for x in (mon if pick == "on" else moff)],
                                           "on": [float(x) for x in mon], "off": [float(x) for x in moff],
                                           "on_minus_off": LM.boot_diff(ron, roff, rng)}
            for t, tau in taus.items():
                use = f >= tau
                sw = np.where(use[:, None], ron, roff)
                m = sw.mean(0)
                r = {"mean": [float(x) for x in m], "off_share": float(1.0 - use.mean()),
                     "minus_rrf": LM.boot_diff(sw, refs["rrf"], rng), "minus_on": LM.boot_diff(sw, ron, rng),
                     "minus_off": LM.boot_diff(sw, roff, rng), "minus_twin0": LM.boot_diff(sw, refs["twin0"], rng)}
                r |= rel(m, means["twin0"], means["gnn0"])
                out["reads"][f"{on}|{t}@{key}"] = r
            log(f"  {on}: on {np.round(mon, 4).tolist()} off {np.round(moff, 4).tolist()} "
                + " ".join(f"{t} {100 * out['reads'][f'{on}|{t}@{key}']['minus_rrf'][0][0]:+.2f}" for t in taus)
                + " (R@5 minus rrf)")
    if a.out:
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(out, indent=1), encoding="utf-8")
        os.replace(tmp, p)
        log(f"wrote {p}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reads", help="comma-separated lean_read9 / lean_mlp8 --out JSONs (their rows files beside them)")
    ap.add_argument("--fam", help="a qfam9 --out JSON (its fam file beside it)")
    ap.add_argument("--off", default="-SEMB", help="the off path's label after ~ in lean_read9's variant names")
    ap.add_argument("--pair", default="", help="extra ON|OFF pairs, ;-separated")
    ap.add_argument("--taus", default="", help="comma-separated tau names to use (default: all of the fam JSON's)")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.reads or not a.fam:
        raise SystemExit("--reads and --fam are required")
    run(a)
    return 0


# ── selftest ─────────────────────────────────────────────────────────────────


def _write_read(d, name, store, read_rows):
    d = Path(d)
    rf = d / f"{name}.rows.npz"
    np.savez_compressed(rf, **{k: v.astype(np.float32) for k, v in store.items()})
    js = {"read_rows": read_rows, "rows_file": {"path": rf.name, "keys": len(store), "sha256": sha(rf)}}
    (d / f"{name}.json").write_text(json.dumps(js), encoding="utf-8")
    return d / f"{name}.json"


def _write_fam(d, fams, tau, carves):
    d = Path(d)
    ff = d / "f.fam.npz"
    np.savez_compressed(ff, **{f"fam@{k}": v.astype(np.float32) for k, v in fams.items()})
    js = {"tau": tau, "carves": carves, "fam_file": {"path": ff.name, "sha256": sha(ff)}}
    (d / "f.json").write_text(json.dumps(js), encoding="utf-8")
    return d / "f.json"


def selftest():
    import tempfile
    from types import SimpleNamespace
    rng = np.random.default_rng(0)
    n = 40

    def met(k):
        r = np.zeros((n, 3))
        r[:, 0] = rng.integers(0, 3, n) / 2
        r[:, 1] = (r[:, 0] == 1).astype(float)
        r[:, 2] = rng.integers(0, 2, n)
        return r if k is None else np.full((n, 3), k)

    refs = {"rrf": met(None), "twin0": met(None), "gnn0": met(None)}
    on, off = met(None), met(None)
    fam = rng.uniform(0, 1, n).astype(np.float32).astype(np.float64)
    with tempfile.TemporaryDirectory() as td:
        rr = {"g=x1": {"rows": n + 5, "chunks": 2}}
        st = {f"{k}@g=x1": v for k, v in refs.items()} | {"m/s0@best@g=x1": on, "m/s0@best~-SEMB@g=x1": off}
        p1 = _write_read(td, "a", st, rr)
        # a second job on the same carve: equal reference rows, another model
        p2 = _write_read(td, "b", {f"{k}@g=x1": v for k, v in refs.items()} | {"q/s0@best@g=x1": off}, rr)
        fj = _write_fam(td, {"g=x1": fam}, {"p50": 0.5, "p10": 0.1}, {"read g=x1": {"rows": n + 5, "chunks_read": 2}})
        a = SimpleNamespace(reads=f"{p1},{p2}", fam=str(fj), off="-SEMB", pair="q/s0@best|m/s0@best", taus="", out=None)
        res = run(a)
        # 1. pairs: the ~-SEMB partner found by name, the --pair one added
        assert res["pairs"] == [["m/s0@best", "m/s0@best~-SEMB"], ["q/s0@best", "m/s0@best"]], res["pairs"]
        # 2. the switched read is the on row where fam >= tau and the off row elsewhere; shares and means exact
        f32 = fam.astype(np.float32).astype(np.float64)
        for t, tau in (("p50", 0.5), ("p10", 0.1)):
            sw = np.where((f32 >= tau)[:, None], on, off)
            r = res["reads"][f"m/s0@best|{t}@g=x1"]
            assert np.allclose(r["mean"], sw.mean(0)) and abs(r["off_share"] - float((f32 < tau).mean())) < 1e-12
            assert abs(r["minus_on"][0][0] - (sw - on).mean(0)[0]) < 1e-12
            assert abs(r["rho"] - (sw.mean(0)[0] - refs["twin0"].mean(0)[0]) / (refs["gnn0"].mean(0)[0] - refs["twin0"].mean(0)[0])) < 1e-9
            sw2 = np.where((f32 >= tau)[:, None], off, on)
            assert np.allclose(res["reads"][f"q/s0@best|{t}@g=x1"]["mean"], sw2.mean(0))
        # 3. the bound picks the better path by R@5 and says which
        b = res["bound"]["m/s0@best@g=x1"]
        assert b["picks"] == ("on" if on.mean(0)[0] >= off.mean(0)[0] else "off")
        # 4. refusals: unaligned reference rows, a model filed twice, fam of another length, a hash mismatch,
        #    another row count in the familiarity job
        bad = {f"{k}@g=x1": v for k, v in refs.items()}
        bad["rrf@g=x1"] = met(None) if not np.array_equal(met(None), refs["rrf"]) else met(1.0)
        p3 = _write_read(td, "c", bad, rr)
        p4 = _write_read(td, "d", {f"{k}@g=x1": v for k, v in refs.items()} | {"m/s0@best@g=x1": on}, rr)
        fj5 = _write_fam(td, {"g=x1": fam[:-1]}, {"p50": 0.5}, {"read g=x1": {"rows": n + 5, "chunks_read": 2}})
        for case in ({"reads": f"{p1},{p3}"}, {"reads": f"{p1},{p4}"}, {"fam": str(fj5)}):
            try:
                run(SimpleNamespace(**{**vars(a), "pair": "", **case}))
                raise AssertionError(f"not refused: {case}")
            except SystemExit:
                pass
        fj6 = _write_fam(td, {"g=x1": fam}, {"p50": 0.5}, {"read g=x1": {"rows": n + 6, "chunks_read": 2}})
        try:
            run(SimpleNamespace(**{**vars(a), "pair": "", "fam": str(fj6)}))
            raise AssertionError("row count")
        except SystemExit:
            pass
        rf = Path(td) / "a.rows.npz"
        rf.write_bytes(rf.read_bytes() + b"x")
        try:
            run(SimpleNamespace(**{**vars(a), "pair": "", "reads": str(p1), "fam": str(fj)}))
            raise AssertionError("hash")
        except SystemExit:
            pass
    print("selftest: the switched read takes the on row where fam >= tau and the off row elsewhere; pairs come from "
          "~-SEMB names and --pair; the bound names its pick; unaligned reference rows, a model filed twice, fam of "
          "another length or row count, and a rows file that does not hash to its record are refused. all checks passed")


if __name__ == "__main__":
    main()
