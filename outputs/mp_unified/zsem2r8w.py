"""SEM2 R8, Part B: the exact solver of Pi_score (docs/SEM2_EDGE_OPERATORS.md 16.4) and the worker loop that runs it in its own process.
numpy and scipy only, so a worker starts in a moment. zsem2r8b imports solve_score from here (its self-test checks it against exhaustive search) and
starts workers with `python zsem2r8w.py` (pickled tasks on stdin, pickled results on stdout)."""
import os
import pickle
import sys

import numpy as np

TIME_LIMIT = 60.0


def solve_score(w, free, par_ptr, par_flat, budget, time_limit=TIME_LIMIT):
    """Pi_score: maximise sum w x subject to the closure and sum x <= budget. HiGHS branch and bound (scipy.optimize.milp), relative gap 0.
    A node with free[u] False needs one of its parents (par_flat[par_ptr[u]:par_ptr[u+1]], variable indices) in the solution; free[u] True needs none.
    Returns (x as 0/1 int8, 'optimal' | 'not_optimal', objective)."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import coo_matrix
    w = np.asarray(w, float)
    n = w.size
    cnt = np.diff(par_ptr)
    need = ~np.asarray(free, bool)
    ubd = np.where(need & (cnt == 0), 0.0, 1.0)                    # a node that needs a parent and has none can never be added
    rows_u = np.flatnonzero(need & (cnt > 0))                      # one closure row per such node: x_u - sum x_parents <= 0
    r = rows_u.size
    rid = np.full(n, -1, np.int64)
    rid[rows_u] = np.arange(r)
    owner = np.repeat(np.arange(n), cnt)                           # the node each parent entry belongs to
    pm = rid[owner] >= 0
    rows = np.concatenate([np.arange(r), rid[owner[pm]], np.full(n, r)])
    cols = np.concatenate([rows_u, par_flat[pm], np.arange(n)])
    vals = np.concatenate([np.ones(r), -np.ones(int(pm.sum())), np.ones(n)])
    ub = np.concatenate([np.zeros(r), [float(budget)]])
    A = coo_matrix((vals, (rows, cols)), shape=(r + 1, n)).tocsr()
    res = milp(-w, constraints=[LinearConstraint(A, -np.inf, ub)], bounds=Bounds(np.zeros(n), ubd),
               integrality=np.ones(n), options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if res.x is None:
        return np.zeros(n, np.int8), "not_optimal", 0.0
    x = np.rint(res.x).astype(np.int8)
    return x, ("optimal" if res.status == 0 else "not_optimal"), float(w @ x)


def feasible(x, free, par_ptr, par_flat, budget):
    """The closure and the budget, read off the solution."""
    if int(x.sum()) > budget:
        return False
    for u in np.flatnonzero(x):
        if free[u]:
            continue
        ps = par_flat[par_ptr[u]:par_ptr[u + 1]]
        if not (ps.size and x[ps].any()):
            return False
    return True


def solve_checked(w, free, par_ptr, par_flat, budget, time_limit=TIME_LIMIT):
    x, st, obj = solve_score(w, free, par_ptr, par_flat, budget, time_limit)
    return x, st, obj, bool(feasible(x, free, par_ptr, par_flat, budget))


def serve():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr                                        # nothing but results may reach the pipe
    while True:
        try:
            task = pickle.load(inp)
        except EOFError:
            break
        if task is None:
            break
        pickle.dump(solve_checked(*task), out, protocol=4)
        out.flush()


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    serve()
