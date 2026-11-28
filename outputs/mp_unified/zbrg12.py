"""BRG12 (docs/BRG12_BUDGET_HEADROOM.md): the visited budget of the budget-following walk raised toward the 2,500-node line, screen on WebQSP, Prime and MAG.

Imports zbrg9.py (frozen; not edited) for the loader, the gates G-S0 and G-S1, the per-chunk resumable screen, the paired aggregate and the read; none of zpl5g.py, zpl4g.py, zbrg1.py is edited. The engine's
budget-following walk (zpl5g.expand5) stops after `visited_cap` visited nodes; the cell `dense_top200+STRUCT:h3_bf_v2000~cs-fu` of WebQSP, Prime and MAG ends at 2,043 to 2,099 mean nodes, 16 to 18 percent under the line.
Here zpl5g.expand5 is replaced, for the duration of one variant's pass over a chunk and restored after, by a wrapper that calls it with `visited_cap` set to the variant's budget (a setting that is not
budget-following is passed through unchanged). The order key, the window, the families and the hop shares are the cell's own. Nothing is fitted; screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg12.py selftest
    python -E -P -B outputs/mp_unified/zbrg12.py screen --ds webqsp | --kb prime|mag [--chunks N] [--smoke M] [--host]     # N = 0: every chunk; default 3
    python -E -P -B outputs/mp_unified/zbrg12.py read --ds webqsp | --kb prime|mag                                          # outputs/brg12/read_<set>.json

Run from the workspace root.
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import zbrg1 as B  # noqa: E402
import zbrg9 as Z9  # noqa: E402
import zpl3g as Z3  # noqa: E402
import zpl5g as Z5  # noqa: E402

DOC = "docs/BRG12_BUDGET_HEADROOM.md"
OUT = ROOT / "outputs" / "brg12"
BF_SETS = ("webqsp",)
BF_KBS = ("prime", "mag")
# variant = (name, visited budget, 0): `eng` is the engine unpatched, `same` the wrapper with the cell's own budget (0 = leave it as it is).
VARS12 = (("eng", None, None), ("same", 0, 0), ("v22", 2200, 0), ("v23", 2300, 0), ("v24", 2400, 0), ("v25", 2500, 0))


class Budget:
    """Context manager: zpl5g.expand5 is the wrapper with the visited budget `cap` for the duration of the block and the engine's function is put back on exit, also after an error."""

    def __init__(self, cap):
        self.cap, self.old = int(cap), None

    def __enter__(self):
        self.old = Z5.expand5
        old, cap = self.old, self.cap

        def wrapped(seeds, fams1, fams2, setting, mods, scorer):
            if cap and setting.get("bf"):
                setting = dict(setting)
                setting["visited_cap"] = cap
            return old(seeds, fams1, fams2, setting, mods, scorer)

        Z5.expand5 = wrapped
        return self

    def __exit__(self, *_exc):
        Z5.expand5 = self.old
        return False


def eval_range(op, d, plan, lo, hi, limit=None):
    """BRG9's per-question record (golds, then every variant's golds in the pool and pool size, graded by zpl3g.grade3) with a visited budget in place of the length rule."""
    res = {}
    for v, (nm, cap, _z) in enumerate(Z9.VARS):
        rows = []
        if nm == "eng":
            it = B.states(op, d, plan, lo, hi, limit)
            for q, st, _seeds, _qv in it:
                P = np.zeros(1, dtype=np.int16)
                S = np.zeros(1, dtype=np.int32)
                Z3.grade3(plan, st, d.golds[q], P, S)
                rows.append((q, int(np.unique(d.golds[q]).size), int(P[0]), int(S[0])))
        else:
            with Budget(cap):
                for q, st, _seeds, _qv in B.states(op, d, plan, lo, hi, limit):
                    P = np.zeros(1, dtype=np.int16)
                    S = np.zeros(1, dtype=np.int32)
                    Z3.grade3(plan, st, d.golds[q], P, S)
                    rows.append((q, int(np.unique(d.golds[q]).size), int(P[0]), int(S[0])))
        res[v] = rows
    nq = len(res[0])
    Q = np.zeros((nq, len(Z9.QCOLS)), dtype=np.int64)
    for v in range(len(Z9.VN)):
        if len(res[v]) != nq or any(res[v][i][0] != res[0][i][0] for i in range(nq)):
            raise RuntimeError(f"variant {Z9.VN[v]} did not give the engine's questions")
        for i, r in enumerate(res[v]):
            if v == 0:
                Q[i, Z9.QI["q"]], Q[i, Z9.QI["ng"]] = r[0], r[1]
            Q[i, Z9.QI[f"p{v}"]], Q[i, Z9.QI[f"s{v}"]] = r[2], r[3]
    return Q


def use_vars():
    """Rebind BRG9's module-level variant list, column layout, output folder, page and row reader for this process."""
    Z9.VARS = VARS12
    Z9.VN = tuple(v[0] for v in VARS12)
    Z9.QCOLS = ("q", "ng") + tuple(f"p{i}" for i in range(len(VARS12))) + tuple(f"s{i}" for i in range(len(VARS12)))
    Z9.QI = {c: i for i, c in enumerate(Z9.QCOLS)}
    Z9.OUT = OUT
    Z9.DOC = DOC
    Z9.eval_range = eval_range


def cmd_selftest(_a):
    rc = Z9.cmd_selftest(_a)
    assert rc == 0
    use_vars()
    assert Z9.VN == ("eng", "same", "v22", "v23", "v24", "v25") and len(Z9.QCOLS) == 2 + 2 * len(Z9.VN) and Z9.OUT == OUT
    rng = np.random.default_rng(29)
    n, dim = 700, 8
    V = rng.normal(size=(n, dim))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    fams = B._toy(rng, n, deg=40)
    ok, differs = 0, 0
    for trial in range(30):
        q = rng.normal(size=dim)
        q /= np.linalg.norm(q)
        seeds = rng.choice(n, size=int(rng.integers(2, 9)), replace=False)
        for sname in ("h2_bf_v2000", "h3_bf_v2000"):
            st = dict(Z5.SET5[sname])
            st["visited_cap"] = int(rng.integers(40, 160))
            st["window"] = 10
            for mods in (("cs", "vr"), ("cs", "fu"), ("cs", "vr", "rr")):
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e0 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
                with Budget(0):
                    e1 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
                assert np.array_equal(e0, e1), "the wrapper with the cell's own budget differs from the engine"
                with Budget(st["visited_cap"]):
                    e2 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
                assert np.array_equal(e0, e2), "the wrapper with budget = cap differs from the engine"
                for extra in (10, 30, 60):
                    cap2 = st["visited_cap"] + extra
                    with Budget(cap2):
                        e3 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
                    assert np.unique(e3).size == e3.size and not np.isin(e3, seeds).any(), "a node is visited twice"
                    assert e3.size + np.unique(seeds).size <= cap2, "the raised visited budget is exceeded"
                    differs += int(not np.array_equal(e0, e3))
                assert e0.size + np.unique(seeds).size <= st["visited_cap"], "the engine's own budget is exceeded"
                sc.release()
                ok += 1
    assert differs > 0, "no raised budget differs from the engine's on any toy walk: the patch would be inert"
    o5 = Z5.expand5
    with Budget(2400):
        assert Z5.expand5 is not o5
    assert Z5.expand5 is o5, "the engine's expansion is not restored"
    try:
        with Budget(2400):
            raise ValueError("x")
    except ValueError:
        pass
    assert Z5.expand5 is o5, "not restored after an error"
    # a non-budget-following setting is passed through unchanged
    st = dict(Z5.SET5["h3_c25_v2000"])
    st["per_seed_cap"] = st["per_frontier_cap"] = 8
    st["visited_cap"] = 80
    sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
    a0 = Z5.expand5(seeds, [fams[0]], fams, st, ("cs", "vr"), sc)
    with Budget(2400):
        a1 = Z5.expand5(seeds, [fams[0]], fams, st, ("cs", "vr"), sc)
    assert np.array_equal(a0, a1)
    sc.release()
    print(f"zbrg12 selftest: BRG9's selftest passes; on {ok} toy budget-following walks (two and three hops, three keys) the wrapper with the cell's own budget (0 and cap) equals zpl5g.expand5 element for element, "
          f"a raised budget keeps distinct fresh nodes inside the raised budget, {differs} raised-budget runs differ from the engine's; the engine's function is restored after a block and after an error; "
          f"a non-budget-following setting is the engine's own: PASS")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for c in ("screen", "read"):
        p = sub.add_parser(c)
        g = p.add_mutually_exclusive_group(required=True)
        g.add_argument("--ds", choices=BF_SETS)
        g.add_argument("--kb", choices=BF_KBS)
        p.add_argument("--host", action="store_true")
        if c == "screen":
            p.add_argument("--smoke", type=int, default=0)
            p.add_argument("--chunks", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return cmd_selftest(a)
    use_vars()
    return Z9.cmd_screen(a) if a.cmd == "screen" else Z9.cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
