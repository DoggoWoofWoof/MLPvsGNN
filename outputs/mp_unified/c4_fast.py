"""C4 (docs/C4_GALLOP_EDGES.md): the pool's typed edges by a galloping merge, in every family's compile alike.

fast_features' typed-edge kernel finds a pool node's in-pool neighbours by walking its whole stored row and probing the
global -> local lookup for every entry, hubs included (about 84,600 entries for a 104-node 2wiki pool, which keeps 256
pairs). edges_fast.typed_edges_gallop merges a row longer than SCAN_FACTOR x the pool size with the (ascending) pool
instead: the same arrays in the same order (its selftest compares every output on 12,000 random cases). C4 serves both
compiles with it, the MLP's (c2_fast.compile_mlp) and the six GNN's (FastCompiler's full compile), and checks on every
question that the compiled scalars, the edges and the scores equal C3's form bit for bit.

    python outputs/mp_unified/c4_fast.py check --dataset 2wiki --queries 30
    python outputs/mp_unified/c4_fast.py run --dataset 2wiki
    python outputs/mp_unified/c4_fast.py report
"""
import os
import sys
from pathlib import Path

THREADS = 1
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS",
           "LEAN_TIME_THREADS"):
    os.environ[_v] = str(THREADS)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import c2_fast as C2  # noqa: E402
import c3_fast as C3  # noqa: E402
import edges_fast as EF  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

C1 = C2.C1
ROOT = HERE.parents[1]
OUT = ROOT / "outputs" / "c4"
SCAN_FACTOR = 4
PATHS = ("zrc4", "zsp4", "gnn4", "zrc3", "zsp3", "gnn3")
NEW = {"zrc4": "zrc3", "zsp4": "zsp3", "gnn4": "gnn3"}


class GallopK:
    """A kernel table with typed_edges by the galloping merge; every other kernel is the table's own."""

    def __init__(self, base):
        self._base = base

    def __getattr__(self, name):
        return getattr(self._base, name)

    def typed_edges(self, tindptr, tcol, trel, tdir, pool, lookup, cap):
        return EF.typed_edges_gallop(tindptr, tcol, trel, tdir, pool, lookup, cap, SCAN_FACTOR * int(pool.shape[0]))


class Kernels:
    def __init__(self, fc):
        self.fc = fc
        self.base = (fc.K, fc._K_small)
        self.gallop = (GallopK(fc.K), GallopK(fc._K_small))

    def use(self, gallop):
        self.fc.K, self.fc._K_small = self.gallop if gallop else self.base


def run_path4(su, fa, ks, k, i, T):
    ks.use(k in NEW)
    try:
        return C3.run_path3(su, fa, NEW.get(k, k), i, T)
    finally:
        ks.use(False)


def comp_of(out, k):
    return out[k][1]["comp"]


def check_c4(su, i, out):
    """Untimed. Each new path against its C3 form on the same question: the compiled scalars and every family's edges
    bit for bit, the scores exactly; the six GNN against the look's stored scores (C1's check)."""
    c = {}
    for new, old in NEW.items():
        a, b = comp_of(out, new), comp_of(out, old)
        ok = np.array_equal(a.scalars, b.scalars, equal_nan=True) and a.edges.keys() == b.edges.keys()
        ok = ok and all(x.dtype == y.dtype and np.array_equal(x, y, equal_nan=True)
                        for f in a.edges for x, y in zip(a.edges[f], b.edges[f]))
        c[f"{new}_compile_bits"] = bool(ok)
        c[f"{new}_scores_equal"] = bool(torch.equal(out[new][0], out[old][0]))
    ref = su.ref[i]
    g = out["gnn4"][0].numpy()
    c["gnn4_vs_look"] = float(np.abs(g - ref["gnn0"].astype(np.float32)).max()) if g.size else 0.0
    c["gnn4_top5_look"] = bool(np.array_equal(C1.top5(g), C1.top5(ref["gnn0"].astype(np.float32))))
    c["pool"] = all(bool(np.array_equal(out[k][1]["pool"], ref["pool"])) for k in PATHS)
    return c


GOOD = ("zrc4_compile_bits", "zrc4_scores_equal", "zsp4_compile_bits", "zsp4_scores_equal", "gnn4_compile_bits",
        "gnn4_scores_equal", "gnn4_top5_look", "pool")


def cold(T, k):
    return sum(v for s_, v in T[k].items() if s_ != "warm_forward")


def run(a):
    import gc
    import time
    t_start = time.time()
    pinned = C1.pin() if not a.no_pin else {"pinned": False, "why": "--no-pin"}
    torch.set_num_threads(1)
    if not a.no_pin and not pinned.get("pinned"):
        raise SystemExit("the process could not pin itself; nothing is timed unpinned")
    if a.dataset in C1.TYPED:
        raise SystemExit(f"{a.dataset}: typed graphs are not in this stage")
    su = C1.Setup(a.dataset, a.queries)
    fa = C3.Fast(su)
    ks = Kernels(su.fc)
    nq = len(su.prep.pools) - 1
    t = time.time()
    for k in PATHS:
        run_path4(su, fa, ks, k, nq, {})
    su.index["jit_question_s"] = time.time() - t
    C1.log(f"{a.dataset}: {nq} measured questions, index {su.index}")
    recs, checks = [], []
    gc.collect()
    gc.disable()
    try:
        for i in range(nq):
            order = PATHS[i % len(PATHS):] + PATHS[:i % len(PATHS)]
            T, out = {"order": list(order)}, {}
            for k in order:
                T[k] = {}
                out[k] = run_path4(su, fa, ks, k, i, T[k])
            T["n"] = int(out["gnn4"][1]["pool"].size)
            recs.append(T)
            checks.append(check_c4(su, i, out))
            del out
            if (i + 1) % 25 == 0:
                gc.collect()
                C1.log(f"  {i + 1}/{nq}")
    finally:
        gc.enable()
    summ = {"questions": len(checks)}
    for key in GOOD:
        summ[f"{key}_ok"] = int(sum(c[key] for c in checks))
    summ["gnn4_vs_look_max"] = max(c["gnn4_vs_look"] for c in checks)
    rec = {"dataset": a.dataset, "carve": C1.CARVE, "queries": nq, "declared_in": "docs/C4_GALLOP_EDGES.md",
           "scan_factor": SCAN_FACTOR, "threads": 1, "pin": pinned, "index": su.index, "paths": list(PATHS),
           "per_question": recs, "checks": checks, "check_summary": summ,
           "fits": {k: {"dir": str(C1.FITS[k].relative_to(ROOT)).replace("\\", "/"),
                        "models_sha256": C1.sha_file(C1.FITS[k] / "models.pt"), "candidate": C1.CAND} for k in C1.FITS},
           "peak_rss_bytes": C1.LC.peak_rss(), "script_sha256": C1.sha_file(__file__),
           "c3_script_sha256": C1.sha_file(C3.__file__), "edges_fast_sha256": C1.sha_file(EF.__file__),
           "numpy": np.__version__, "torch": torch.__version__,
           "seconds": round(time.time() - t_start, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out = Path(a.out or OUT / f"{a.dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    C1.LC.write_json(out, rec)
    C1.log(f"{a.dataset}: done in {rec['seconds']}s; checks {summ}; -> {out}")
    return 0


def report():
    import json
    res = {}
    for f in sorted(OUT.glob("*.json")):
        if f.name.startswith("report"):
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        P = r["per_question"]
        good = [j for j, c in enumerate(r["checks"]) if all(c[k] for k in GOOD)]
        d = {"questions": len(P), "kept": len(good), "stopped": 1 - len(good) / max(len(P), 1) > 0.02}
        rng = np.random.default_rng(0)
        col = lambda k: np.asarray([cold(P[j], k) for j in good])  # noqa: E731
        for k in PATHS:
            x = col(k) * 1e3
            d[f"{k}_cold_ms"] = {str(p): float(np.percentile(x, p)) for p in (50, 95, 99)}
        for num, den in (("gnn4", "zrc4"), ("gnn4", "zsp4"), ("gnn3", "zrc3"), ("zrc3", "zrc4"), ("zsp3", "zsp4"),
                         ("gnn3", "gnn4")):
            d[f"cold_{num}_over_{den}"] = C2.ratio_ci(col(num), col(den), rng)
        d["compile_p50_ms"] = {k: float(np.percentile([P[j][k]["compile"] for j in good], 50)) * 1e3 for k in PATHS}
        d["checks"] = r["check_summary"]
        res[r["dataset"]] = d
    C1.LC.write_json(OUT / "report.json", res)
    f = lambda r_: f"{r_['ratio']:.2f} [{r_['ci'][0]:.2f}, {r_['ci'][1]:.2f}]"  # noqa: E731
    lines = ["Cold, batch 1, total per question, p50 ms:", "",
             "| dataset | zrc4 | zsp4 | gnn4 | **gnn4 / zrc4** | gnn4 / zsp4 | gnn3 / zrc3 (C3, same run) | zrc3 / zrc4 | "
             "gnn3 / gnn4 | compile p50 zrc3 -> zrc4 | gnn3 -> gnn4 |",
             "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, d in res.items():
        m = lambda k: f"{d[f'{k}_cold_ms']['50']:.1f}"  # noqa: E731
        cp = d["compile_p50_ms"]
        lines.append(f"| {ds} | {m('zrc4')} | {m('zsp4')} | {m('gnn4')} | **{f(d['cold_gnn4_over_zrc4'])}** | "
                     f"{f(d['cold_gnn4_over_zsp4'])} | {f(d['cold_gnn3_over_zrc3'])} | {f(d['cold_zrc3_over_zrc4'])} | "
                     f"{f(d['cold_gnn3_over_gnn4'])} | {cp['zrc3']:.2f} -> {cp['zrc4']:.2f} | "
                     f"{cp['gnn3']:.2f} -> {cp['gnn4']:.2f} |")
    lines += ["", "Checks:", ""] + [f"- {ds}: kept {d['kept']}/{d['questions']}; {d['checks']}" for ds, d in res.items()]
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


def check(a):
    C1.pin(2)
    torch.set_num_threads(1)
    su = C1.Setup(a.dataset, a.queries)
    fa = C3.Fast(su)
    ks = Kernels(su.fc)
    nq = len(su.prep.pools) - 1
    bad, comp = [], {k: [] for k in PATHS}
    for i in [nq] + list(range(nq)):
        out, T = {}, {}
        for k in PATHS:
            T[k] = {}
            out[k] = run_path4(su, fa, ks, k, i, T[k])
        c = check_c4(su, i, out)
        if not all(c[k] for k in GOOD):
            bad.append((i, {k: c[k] for k in GOOD if not c[k]}))
        if i != nq:
            for k in PATHS:
                comp[k].append((T[k]["compile"], cold(T, k)))
    C1.log(f"c4 check {a.dataset}: {nq} questions, {len(bad)} failing {bad[:3]}")
    for k, v in comp.items():
        v = np.asarray(v)
        C1.log(f"  {k}: compile p50 {np.median(v[:, 0]) * 1e3:.2f} ms, cold p50 {np.median(v[:, 1]) * 1e3:.2f} ms")
    return 0 if not bad else 1


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--dataset")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out")
    ap.add_argument("--no-pin", action="store_true")
    a = ap.parse_args(argv)
    return {"run": run, "check": check, "report": lambda _a: report()}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
