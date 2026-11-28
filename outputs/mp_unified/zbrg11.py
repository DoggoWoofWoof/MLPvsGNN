"""BRG11 (docs/BRG11_HOP_BUDGET_SCHEDULE.md): how the visited budget of a budget-following walk is split over the hops, screen on the five budget-following sets.

Imports zbrg9.py (frozen; not edited) for the loader, the gates G-S0 and G-S1, the per-chunk resumable screen, the paired aggregate and the read; none of zpl5g.py, zpl4g.py, zbrg1.py is edited. The engine's
budget-following walk (zpl5g.expand5) keeps at hop h at most ceil(rem / (H - h + 1)) fresh nodes, rem = the visited budget minus the nodes visited so far: an equal share of what is left. Here zpl5g.expand5
is replaced, for the duration of one variant's pass over a chunk and restored after, by a copy whose share of hop h is ceil(rem * w_h / (w_h + ... + w_H)) with w_k = r^(k-1): r = 1 is the engine's own rule,
r > 1 gives the later hops more, r < 1 the earlier hops. Weights are integers (no float), so r = 1 equals the engine element for element. The order key, the window, the families and the visited budget are
the cell's own. Nothing is fitted; screen only; nothing is adopted.

    python -E -P -B outputs/mp_unified/zbrg11.py selftest
    python -E -P -B outputs/mp_unified/zbrg11.py screen --ds musique|webqsp | --kb prime|mag|amazon [--chunks N] [--smoke M] [--host]     # N = 0: every chunk; default 3
    python -E -P -B outputs/mp_unified/zbrg11.py read --ds D | --kb K                                                                    # outputs/brg11/read_<set>.json

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
import zpl4g as Z4  # noqa: E402
import zpl5g as Z5  # noqa: E402

DOC = "docs/BRG11_HOP_BUDGET_SCHEDULE.md"
OUT = ROOT / "outputs" / "brg11"
BF_SETS = ("musique", "webqsp")
BF_KBS = ("prime", "mag", "amazon")
# variant = (name, num, den): the hop weights are (num/den)^(k-1). `eng` is the engine unpatched, `same` the copy with r = 1.
VARS11 = (("eng", None, None), ("same", 1, 1), ("f2", 1, 2), ("b2", 2, 1), ("b4", 4, 1), ("b8", 8, 1))


def weights(hops, num, den):
    """The integer hop weights proportional to (num/den)^(k-1), k = 1..hops."""
    return [num ** k * den ** (hops - 1 - k) for k in range(hops)]


def hop_budget(rem, h, w):
    """The share of hop h (1-based): ceil(rem * w_h / (w_h + ... + w_H))."""
    return -(-(rem * w[h - 1]) // sum(w[h - 1:]))


def expand5s(seeds, fams1, fams2, setting, mods, scorer, num, den):
    """zpl5g.expand5 with the hop shares of weights (num/den)^(h-1); the same candidate lists, order key, window and visited budget. A setting that is not budget-following is the engine's own."""
    if not setting.get("bf"):
        return Z4.expand4(seeds, fams1, fams2, setting, mods, scorer)
    key = next(m for m in Z5.KEYS5 if m in mods)
    rr = "rr" in mods
    hops, limit, win = int(setting["hops"]), int(setting["visited_cap"]), int(setting["window"])
    w = weights(hops, num, den)
    seeds = np.asarray(seeds, dtype=np.int64)
    visited = np.unique(seeds)
    frontier = seeds
    out = []
    for h in range(1, hops + 1):
        rem = limit - int(visited.size)
        if rem <= 0 or frontier.size == 0:
            break
        budget = hop_budget(rem, h, w)
        cols, grp, fam, rank = Z5.capped_parts5(frontier, fams1 if h == 1 else fams2, win, scorer)
        if cols.size == 0:
            break
        k = np.isin(cols, visited, invert=True)
        fresh = Z5.order_fresh5(cols[k], grp[k], fam[k], rank[k], key, rr, scorer, int(frontier.size))[:budget]
        if fresh.size == 0:
            break
        out.append(fresh)
        visited = np.union1d(visited, fresh)
        frontier = fresh
    return np.concatenate(out) if out else np.empty(0, dtype=np.int64)


class Sched:
    """Context manager: zpl5g.expand5 is the schedule version for the duration of the block and the engine's function is put back on exit, also after an error."""

    def __init__(self, num, den):
        self.num, self.den, self.old = num, den, None

    def __enter__(self):
        self.old = Z5.expand5
        n, d = self.num, self.den
        Z5.expand5 = lambda seeds, f1, f2, setting, mods, scorer: expand5s(seeds, f1, f2, setting, mods, scorer, n, d)
        return self

    def __exit__(self, *_exc):
        Z5.expand5 = self.old
        return False


def eval_range(op, d, plan, lo, hi, limit=None):
    """BRG9's per-question record (golds, then every variant's golds in the pool and pool size, graded by zpl3g.grade3) with a hop-share schedule in place of the length rule."""
    res = {}
    for v, (nm, num, den) in enumerate(Z9.VARS):
        rows = []
        if nm == "eng":
            it = B.states(op, d, plan, lo, hi, limit)
            for q, st, _seeds, _qv in it:
                P = np.zeros(1, dtype=np.int16)
                S = np.zeros(1, dtype=np.int32)
                Z3.grade3(plan, st, d.golds[q], P, S)
                rows.append((q, int(np.unique(d.golds[q]).size), int(P[0]), int(S[0])))
        else:
            with Sched(num, den):
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
    Z9.VARS = VARS11
    Z9.VN = tuple(v[0] for v in VARS11)
    Z9.QCOLS = ("q", "ng") + tuple(f"p{i}" for i in range(len(VARS11))) + tuple(f"s{i}" for i in range(len(VARS11)))
    Z9.QI = {c: i for i, c in enumerate(Z9.QCOLS)}
    Z9.OUT = OUT
    Z9.DOC = DOC
    Z9.eval_range = eval_range


def cmd_selftest(_a):
    rc = Z9.cmd_selftest(_a)
    assert rc == 0
    # the arithmetic of the shares
    assert weights(3, 1, 1) == [1, 1, 1] and weights(3, 2, 1) == [1, 2, 4] and weights(3, 1, 2) == [4, 2, 1] and weights(2, 8, 1) == [1, 8]
    for hops in (2, 3, 4):
        for rem in (1, 7, 100, 1800):
            for h in range(1, hops + 1):
                assert hop_budget(rem, h, weights(hops, 1, 1)) == -(-rem // (hops - h + 1)), "r = 1 is not the engine's share"
    assert hop_budget(1800, 1, [1, 2, 4]) == 258 and hop_budget(1800, 3, [1, 2, 4]) == 1800 and hop_budget(1800, 1, [4, 2, 1]) == 1029
    use_vars()
    assert Z9.VN == ("eng", "same", "f2", "b2", "b4", "b8") and len(Z9.QCOLS) == 2 + 2 * len(Z9.VN) and Z9.OUT == OUT
    rng = np.random.default_rng(23)
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
            for mods in (("cs", "vr"), ("cs", "vr", "rr")):
                sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
                e0 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
                e1 = expand5s(seeds, [fams[0]], fams, st, mods, sc, 1, 1)
                assert np.array_equal(e0, e1), ("r = 1 differs from the engine's expansion", trial, sname, mods)
                with Sched(1, 1):
                    e2 = Z5.expand5(seeds, [fams[0]], fams, st, mods, sc)
                assert np.array_equal(e0, e2)
                for num, den in ((1, 2), (2, 1), (4, 1), (8, 1)):
                    e3 = expand5s(seeds, [fams[0]], fams, st, mods, sc, num, den)
                    assert np.unique(e3).size == e3.size and not np.isin(e3, seeds).any(), "a node is visited twice"
                    assert e3.size + np.unique(seeds).size <= st["visited_cap"], "the visited budget is exceeded"
                    differs += int(not np.array_equal(e0, e3))
                sc.release()
                ok += 1
    assert differs > 0, "no schedule differs from the engine's on any toy walk: the patch would be inert"
    o5 = Z5.expand5
    with Sched(2, 1):
        assert Z5.expand5 is not o5
    assert Z5.expand5 is o5, "the engine's expansion is not restored"
    try:
        with Sched(2, 1):
            raise ValueError("x")
    except ValueError:
        pass
    assert Z5.expand5 is o5, "not restored after an error"
    # a non-budget-following setting is the engine's own
    st = dict(Z5.SET5["h3_c25_v2000"])
    st["per_seed_cap"] = st["per_frontier_cap"] = 8
    st["visited_cap"] = 80
    sc = Z3.Scorer(lambda x: V[np.asarray(x)], {"q": q}, n)
    assert np.array_equal(Z5.expand5(seeds, [fams[0]], fams, st, ("cs", "vr"), sc), expand5s(seeds, [fams[0]], fams, st, ("cs", "vr"), sc, 4, 1))
    sc.release()
    print(f"zbrg11 selftest: BRG9's selftest passes; the integer hop shares equal the engine's ceil(rem / (H - h + 1)) at r = 1 and the hand-made values at r = 2 and 1/2; on {ok} toy budget-following walks "
          f"(two and three hops, two keys) r = 1 equals zpl5g.expand5 element for element, every schedule keeps distinct fresh nodes inside the visited budget, {differs} schedule runs differ from the engine's; "
          f"the engine's function is restored after a block and after an error; a non-budget-following setting is the engine's own: PASS")
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
