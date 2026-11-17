"""HR1 (docs/HR1_HOPRETRIEVER_ROUTE1.md): HopRetriever's eq. 5 from vectors we already have, scouted on D83F's training
carves of the four passage datasets. Parameter-free; trains nothing; writes only under outputs/hr1.

    python outputs/mp_unified/zhr1.py selftest            (laptop: synthetic stores, vectors and pools vs brute force)
    python outputs/mp_unified/zhr1.py selftest --host     (host: G1 and G2 on squad's first cache part; no measure)
    python outputs/mp_unified/zhr1.py cols --ds squad --host [--budget-gb 4]
    python outputs/mp_unified/zhr1.py report

cols: per cache row, s_u (the plain cosine), and per family f (ment = D83's R.mention store, co = its R.name_comention +
R.title_comention store) s_m^f = the cosine of the question with j's entity vector (the sum of the unit vectors of j's
package-level neighbours; NaN when j has none) and deg_f. report: section 4's measures and section 6's calls.
"""
import os
import sys

_T = os.environ.get("HR1_THREADS", "4")
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, _T)

import argparse  # noqa: E402
import ctypes  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "scripts", ROOT / "src", HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

DECLARED = "docs/HR1_HOPRETRIEVER_ROUTE1.md"
STAGE = "HR1"
OUT = ROOT / "outputs" / "hr1"
SRC_ARM = "D83F"
CACHE = ROOT / "outputs" / "abl" / SRC_ARM / "cache"
CARVES = {"squad": "fit", "musique": "s1fit", "hotpotqa": "fit", "2wiki": "fit"}
FAMS = ("ment", "co")            # zd91.Stores: 0 = the slot (R.mention), 1 = NER (R.name_comention + R.title_comention)
DIM = 1536
TOL_G2 = 2e-3
BAR = 0.55
TOP = 5
CHUNK = 262144                   # node rows per gather
ROWDOT = 131072                  # cache rows per scoring chunk
EMAX = 60_000_000                # neighbour entries per entity group


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def rel(p):
    return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)


def save_npy(d, k, x):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / f"{k}.tmp.npy"
    np.save(tmp, x)
    os.replace(tmp, d / f"{k}.npy")
    return sha_file(d / f"{k}.npy")


def peak_gb():
    """Peak working set and peak private commit of this process, GB (Windows; None elsewhere)."""
    try:
        import ctypes.wintypes as W

        class PMC(ctypes.Structure):
            _fields_ = [("cb", W.DWORD), ("PageFaultCount", W.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        k32.GetCurrentProcess.restype = W.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [W.HANDLE, ctypes.POINTER(PMC), W.DWORD]
        c = PMC()
        c.cb = ctypes.sizeof(c)
        if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return {"peak_working_set_gb": round(c.PeakWorkingSetSize / 2 ** 30, 2),
                "peak_commit_gb": round(c.PeakPagefileUsage / 2 ** 30, 2)}
    except Exception:
        return None


def unit(a):
    a = np.asarray(a, np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


# ── the entity vectors and the per-row columns ───────────────────────────────


def neighbours(indptr, col, rows):
    """(pos, nbr): every stored neighbour of rows[pos], each once, the row itself never."""
    rows = np.asarray(rows, np.int64)
    ip = np.asarray(indptr, np.int64)
    a, b = ip[rows], ip[rows + 1]
    cnt = b - a
    tot = int(cnt.sum())
    if tot == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pos = np.repeat(np.arange(rows.size, dtype=np.int64), cnt)
    first = np.concatenate(([0], np.cumsum(cnt)[:-1]))
    idx = np.arange(tot, dtype=np.int64) + np.repeat(a - first, cnt)
    nb = np.asarray(col)[idx].astype(np.int64)
    keep = nb != rows[pos]
    pos, nb = pos[keep], nb[keep]
    key = np.unique(pos * (int(nb.max(initial=0)) + 1) + nb) if pos.size else pos
    if key.size != pos.size:
        w = int(nb.max(initial=0)) + 1
        pos, nb = key // w, key % w
    return pos, nb


def groups_of(U, stores, g_max, e_max=EMAX):
    """Consecutive slices of U holding at most g_max entity rows and e_max stored neighbour entries."""
    deg = np.zeros(U.size, np.int64)
    for ip, _c in stores.values():
        ip = np.asarray(ip, np.int64)
        deg += ip[U + 1] - ip[U]
    out, g0, acc = [], 0, 0
    for i in range(U.size):
        if i > g0 and (i - g0 >= g_max or acc + deg[i] > e_max):
            out.append((g0, i))
            g0, acc = i, 0
        acc += int(deg[i])
    if U.size:
        out.append((g0, U.size))
    return out


def rowdot(A, ai, Q, qi):
    out = np.empty(ai.size, np.float32)
    for a in range(0, ai.size, ROWDOT):
        b = min(ai.size, a + ROWDOT)
        out[a:b] = np.einsum("ij,ij->i", A[ai[a:b]], Q[qi[a:b]])
    return out


def entity_columns(U, uidx, qi, Qhat, stores, X, g_max, chunk=CHUNK, e_max=EMAX):
    """Per cache row r (node U[uidx[r]], question qi[r]): s_u and, per family, s_m (NaN when undefined) and deg."""
    from scipy import sparse
    R = uidx.size
    s_u = np.full(R, np.nan, np.float32)
    s_m = {f: np.full(R, np.nan, np.float32) for f in stores}
    deg = {f: np.zeros(R, np.int32) for f in stores}
    order = np.argsort(uidx, kind="stable")
    su = uidx[order]
    gs = groups_of(U, stores, g_max, e_max)
    t0 = time.time()
    for k, (g0, g1) in enumerate(gs):
        rows_g = U[g0:g1]
        G = g1 - g0
        nbs = {f: neighbours(ip, col, rows_g) for f, (ip, col) in stores.items()}
        need = np.unique(np.concatenate([rows_g] + [nb for _p, nb in nbs.values()]))
        S, dg = {}, {}
        for f, (pos, nb) in nbs.items():
            dg[f] = np.bincount(pos, minlength=G).astype(np.int32)
            S[f] = sparse.csc_matrix((np.ones(pos.size, np.float32), (pos, np.searchsorted(need, nb))),
                                     shape=(G, need.size))
        M = {f: np.zeros((G, DIM), np.float32) for f in stores}
        XU = np.zeros((G, DIM), np.float32)
        at = np.searchsorted(need, rows_g)
        for a in range(0, need.size, chunk):
            b = min(need.size, a + chunk)
            Xn = unit(X.read(need[a:b]))
            for f in stores:
                sub = S[f][:, a:b]
                if sub.nnz:
                    M[f] += np.asarray(sub @ Xn, np.float32)
            m = (at >= a) & (at < b)
            XU[m] = Xn[at[m] - a]
            del Xn
        lo, hi = np.searchsorted(su, g0), np.searchsorted(su, g1)
        rr = order[lo:hi]
        loc = uidx[rr] - g0
        s_u[rr] = rowdot(XU, loc, Qhat, qi[rr])
        for f in stores:
            nrm = np.linalg.norm(M[f], axis=1)
            ok = (dg[f] > 0) & (nrm > 0)
            Mn = M[f] / np.maximum(nrm, 1e-30)[:, None]
            v = rowdot(Mn, loc, Qhat, qi[rr])
            v[~ok[loc]] = np.nan
            s_m[f][rr] = v
            deg[f][rr] = dg[f][loc]
            del Mn
        del M, XU, S
        if (k + 1) % max(1, len(gs) // 20) == 0 or k + 1 == len(gs):
            log(f"  group {k + 1}/{len(gs)}: {G} entity rows, {need.size} node rows read ({time.time() - t0:.0f}s)")
    return s_u, s_m, deg


# ── one dataset's columns (host) ──────────────────────────────────────────────


def load_carve(ds, carve, cache=CACHE, first_part_only=False):
    import lean_cache as LC
    parts, recs = LC.part_dirs(ds, carve, cache)
    if first_part_only:
        parts, recs = parts[:1], recs[:1]
    acc = {k: [] for k in ("node", "gold", "dc", "n", "gt", "q", "twin0")}
    for p, r in zip(parts, recs):
        c = int(r["xc_spans"]["dense_cos"][0])
        A = {k: np.load(p / f"{k}.npy", mmap_mode="r") for k in ("n", "gold_total", "q_emb", "node", "gold", "xc", "score2")}
        if int(np.asarray(A["n"]).sum()) != A["node"].shape[0] or A["xc"].shape[0] != A["node"].shape[0]:
            raise SystemExit(f"HR1 G2: {rel(p)}: rows do not match n")
        acc["node"].append(np.asarray(A["node"], np.int64))
        acc["gold"].append(np.asarray(A["gold"], bool))
        acc["dc"].append(np.asarray(A["xc"][:, c], np.float32))
        acc["n"].append(np.asarray(A["n"], np.int64))
        acc["gt"].append(np.asarray(A["gold_total"], np.int64))
        acc["q"].append(np.asarray(A["q_emb"], np.float32))
        acc["twin0"].append(np.asarray(A["score2"][:, 0], np.float32))
    out = {k: np.concatenate(v) for k, v in acc.items()}
    out["parts"] = {p.name: sha_file(p / "record.json") for p in parts}
    return out


def open_world(ds, host):
    """D83's two stores of ds (zd91.Stores, gated against D83's key sets) and the package's node rows."""
    import zb1p as B
    B.mods(host)                                              # on the host: the mirror first, so open_nodes reads it
    import lean_mlp3 as L3
    import zd91
    st = zd91.Stores(SRC_ARM, ds, host)
    stores = {"ment": st.get(0), "co": st.get(1)}
    nodes, rsha = L3.open_nodes([ds])
    X = nodes[ds]
    for f, s in stores.items():
        if int(s.n_nodes) != int(X.n_rows):
            raise SystemExit(f"HR1: {ds} store {f} holds {s.n_nodes} nodes, the package {X.n_rows}")
    return {f: (s.indptr, s.col) for f, s in stores.items()}, X, rsha, st.info


def g2(s_u, dc):
    d = np.abs(s_u.astype(np.float64) - dc.astype(np.float64))
    bad = ~np.isfinite(d)
    fin = d[~bad]
    mx = float(fin.max()) if fin.size else 0.0
    return {"rows": int(d.size), "max_abs_diff": round(mx, 6), "not_finite": int(bad.sum()),
            "ok": bool(mx <= TOL_G2 and not bad.any())}


def cols(ds, host, budget_gb):
    t = time.time()
    carve = CARVES[ds]
    C = load_carve(ds, carve)
    stores, X, rsha, sinfo = open_world(ds, host)
    U, uidx = np.unique(C["node"], return_inverse=True)
    uidx = uidx.astype(np.int64).ravel()
    qi = np.repeat(np.arange(C["n"].size, dtype=np.int64), C["n"])
    Qhat = unit(C["q"])
    g_max = max(1024, int(budget_gb * 2 ** 30 // ((1 + len(FAMS)) * DIM * 4)))
    log(f"{ds}/{carve}: {C['n'].size} questions, {uidx.size} rows, {U.size} pool nodes, groups of {g_max}")
    s_u, s_m, deg = entity_columns(U, uidx, qi, Qhat, stores, X, g_max)
    gate = g2(s_u, C["dc"])
    if not gate["ok"]:
        raise SystemExit(f"HR1 G2: {ds}: s_u is not the cache's dense_cos: {gate}")
    od = OUT / "cols" / ds
    shas = {"s_u": save_npy(od, "s_u", s_u)}
    for f in FAMS:
        shas[f"s_m_{f}"] = save_npy(od, f"s_m_{f}", s_m[f])
        shas[f"deg_{f}"] = save_npy(od, f"deg_{f}", deg[f])
    rec = {"stage": STAGE, "declared": DECLARED, "dataset": ds, "carve": carve, "src_arm": SRC_ARM,
           "cache": rel(CACHE / ds / carve), "cache_parts_record_sha256": C["parts"], "rows": int(uidx.size),
           "queries": int(C["n"].size), "pool_nodes": int(U.size), "node_record_sha256": rsha, "stores": sinfo,
           "g2": gate, "arrays_sha256": shas, "code_sha256": sha_file(Path(__file__)), "budget_gb": budget_gb,
           "seconds": round(time.time() - t, 1), "memory": peak_gb(), "written_utc": utc()}
    write_json(od / "record.json", rec)
    log(f"{ds}: written {rel(od)}; G2 {gate}; memory {rec['memory']}; {rec['seconds']}s")
    return 0


# ── the measures ──────────────────────────────────────────────────────────────


def sorted_pos(score):
    """Pool positions in score order: descending, ties by pool position."""
    m = score.size
    return np.lexsort((np.arange(m), -np.asarray(score, np.float64)))


def matched_pairs(gold_o):
    """Positions (in the order) of each gold and of its nearest non-gold above and below (-1 when none)."""
    gp = np.flatnonzero(gold_o)
    npos = np.flatnonzero(~gold_o)
    k = np.searchsorted(npos, gp)
    if npos.size == 0:
        return gp, np.full(gp.size, -1), np.full(gp.size, -1)
    above = np.where(k > 0, npos[np.maximum(k - 1, 0)], -1)
    below = np.where(k < npos.size, npos[np.minimum(k, npos.size - 1)], -1)
    return gp, above, below


def auc(score, gold):
    from scipy.stats import rankdata
    ng, nn = int(gold.sum()), int((~gold).sum())
    if ng == 0 or nn == 0:
        return np.nan
    r = rankdata(np.asarray(score, np.float64), method="average")
    return float((r[gold].sum() - ng * (ng + 1) / 2) / (ng * nn))


def r_at(score, gold, gt, k=TOP):
    return float(gold[sorted_pos(score)[:k]].sum()) / gt if gt > 0 else np.nan


def fuse(s_u, s_m):
    """eq. 5 with W_k = W_v = I and h = q: softmax(s_m, s_u) weights; s_u where s_m is undefined."""
    s_u = np.asarray(s_u, np.float64)
    s_m = np.asarray(s_m, np.float64)
    w_m = 1.0 / (1.0 + np.exp(s_u - s_m))
    f = w_m * s_m + (1.0 - w_m) * s_u
    return np.where(np.isnan(s_m), s_u, f)


def measure(gold, n, gt, s_u, s_m, deg, twin0):
    """Section 4 on one carve. s_m and deg are per family."""
    off = np.concatenate(([0], np.cumsum(n)))
    elig = []
    pr = {f: {"q": [], "w": [], "d": [], "miss": []} for f in s_m}
    cov = {f: np.zeros(4, np.int64) for f in s_m}         # rows, rows defined, golds, golds defined
    dgs = {f: ([], []) for f in s_m}
    aucs = {k: [] for k in ("s_u", "twin0", *[f"F_{f}" for f in s_m])}
    r5 = {k: [] for k in ("s_u", *[f"F_{f}" for f in s_m])}
    for q in range(n.size):
        a, b = off[q], off[q + 1]
        g = gold[a:b]
        if not g.any() or g.all():
            continue
        e = len(elig)
        elig.append(q)
        su = s_u[a:b]
        o = sorted_pos(su)
        go = g[o]
        gp, ab, be = matched_pairs(go)
        aucs["s_u"].append(auc(su, g))
        aucs["twin0"].append(auc(twin0[a:b], g))
        r5["s_u"].append(r_at(su, g, gt[q]))
        for f in s_m:
            sm, dd = s_m[f][a:b], deg[f][a:b]
            ok = ~np.isnan(sm)
            cov[f] += [b - a, int(ok.sum()), int(g.sum()), int((ok & g).sum())]
            dgs[f][0].append(dd[g])
            dgs[f][1].append(dd[~g])
            smo, oko, ddo = sm[o], ok[o], dd[o]
            for v in (ab, be):
                m = (v >= 0)
                gg, vv = gp[m], v[m]
                c = oko[gg] & oko[vv]
                gg, vv = gg[c], vv[c]
                w = np.where(smo[gg] > smo[vv], 1.0, np.where(smo[gg] == smo[vv], 0.5, 0.0))
                pr[f]["q"].append(np.full(gg.size, e))
                pr[f]["w"].append(w)
                pr[f]["d"].append(ddo[gg])
                pr[f]["miss"].append(gg >= TOP)
            F = fuse(su, sm)
            aucs[f"F_{f}"].append(auc(F, g))
            r5[f"F_{f}"].append(r_at(F, g, gt[q]))
    E = len(elig)
    out = {"eligible": E, "questions": int(n.size), "fam": {}}
    for f in s_m:
        P = {k: np.concatenate(v) if v else np.zeros(0) for k, v in pr[f].items()}
        qn = P["q"].astype(np.int64)
        cnt = np.bincount(qn, minlength=E)
        A_q = np.bincount(qn, weights=P["w"], minlength=E)[cnt > 0] / cnt[cnt > 0]
        mm = P["miss"].astype(bool)
        cm = np.bincount(qn[mm], minlength=E)
        Am_q = np.bincount(qn[mm], weights=P["w"][mm], minlength=E)[cm > 0] / cm[cm > 0]
        terc = {}
        if P["d"].size:
            c1, c2 = np.quantile(P["d"], [1 / 3, 2 / 3])
            tb = np.where(P["d"] <= c1, 0, np.where(P["d"] <= c2, 1, 2))
            for t in range(3):
                mt = tb == t
                ct = np.bincount(qn[mt], minlength=E)
                at = np.bincount(qn[mt], weights=P["w"][mt], minlength=E)[ct > 0] / ct[ct > 0]
                terc[str(t)] = {"cut": [float(c1), float(c2)], "questions": int((ct > 0).sum()), "pairs": int(mt.sum()),
                                "A": round(float(at.mean()), 5) if at.size else None}
        gd = np.concatenate(dgs[f][0]) if dgs[f][0] else np.zeros(0)
        nd = np.concatenate(dgs[f][1]) if dgs[f][1] else np.zeros(0)
        out["fam"][f] = {"A_q": A_q, "A_miss_q": Am_q, "counted": int((cnt > 0).sum()), "pairs": int(cnt.sum()),
                         "counted_miss": int((cm > 0).sum()), "terciles": terc,
                         "coverage": {"rows": round(cov[f][1] / max(cov[f][0], 1), 5),
                                      "golds": round(cov[f][3] / max(cov[f][2], 1), 5)},
                         "median_deg": {"gold": float(np.median(gd)) if gd.size else None,
                                        "non_gold": float(np.median(nd)) if nd.size else None}}
    out["auc"] = {k: np.asarray(v, np.float64) for k, v in aucs.items()}
    out["r5"] = {k: np.asarray(v, np.float64) for k, v in r5.items()}
    return out


def call_of(A_q):
    import zd83rep as R
    if A_q.size == 0:
        return {"diff": None, "ci": None, "call": "EMPTY"}
    return R.interval(A_q, np.full(A_q.size, 0.5))


def decide(cells):
    """cells: {ds: {"call", "A", "counted", "eligible"}} of one family over the four datasets."""
    if set(cells) != set(CARVES):
        return {"verdict": "INCOMPLETE", "carry": [], "below": []}
    carry = sorted(d for d, c in cells.items()
                   if c["call"] == "ABOVE" and c["A"] is not None and c["A"] >= BAR and 2 * c["counted"] >= c["eligible"])
    below = sorted(d for d, c in cells.items() if c["call"] == "BELOW")
    return {"verdict": "CARRIED" if len(carry) >= 3 and not below else "NO_SIGNAL", "carry": carry, "below": below}


def report():
    import zd83rep as R
    res, cells = {}, {f: {} for f in FAMS}
    for ds, carve in CARVES.items():
        od = OUT / "cols" / ds
        if not (od / "record.json").exists():
            log(f"report: {ds} has no columns yet")
            continue
        rec = json.loads((od / "record.json").read_text(encoding="utf-8"))
        C = load_carve(ds, carve)
        if C["parts"] != rec["cache_parts_record_sha256"]:
            raise SystemExit(f"HR1 G3: {ds}: the cache parts are not the ones the columns name")
        cols_ = {}
        for k, s in rec["arrays_sha256"].items():
            if sha_file(od / f"{k}.npy") != s:
                raise SystemExit(f"HR1 G3: {ds}/{k}.npy is not the file its record names")
            cols_[k] = np.load(od / f"{k}.npy")
            if cols_[k].size != C["node"].size:
                raise SystemExit(f"HR1 G3: {ds}/{k}.npy holds {cols_[k].size} rows, the cache {C['node'].size}")
        m = measure(C["gold"], C["n"], C["gt"], cols_["s_u"], {f: cols_[f"s_m_{f}"] for f in FAMS},
                    {f: cols_[f"deg_{f}"] for f in FAMS}, C["twin0"])
        row = {"carve": carve, "eligible": m["eligible"], "questions": m["questions"], "g2": rec["g2"], "fam": {}}
        for f in FAMS:
            fm = m["fam"][f]
            c = call_of(fm["A_q"])
            cm = call_of(fm["A_miss_q"])
            A = round(float(fm["A_q"].mean()), 5) if fm["A_q"].size else None
            row["fam"][f] = {"A": A, "A_call": c, "A_miss": round(float(fm["A_miss_q"].mean()), 5) if fm["A_miss_q"].size else None,
                             "A_miss_call": cm, **{k: fm[k] for k in ("counted", "pairs", "counted_miss", "terciles",
                                                                       "coverage", "median_deg")}}
            cells[f][ds] = {"call": c["call"], "A": A, "counted": fm["counted"], "eligible": m["eligible"]}
        au, r5 = m["auc"], m["r5"]
        row["auc"] = {k: round(float(np.nanmean(v)), 5) for k, v in au.items()}
        row["r5"] = {k: round(float(np.nanmean(v)), 5) for k, v in r5.items()}
        for f in FAMS:
            ok = ~np.isnan(au[f"F_{f}"]) & ~np.isnan(au["s_u"])
            row["auc"][f"F_{f}-s_u"] = R.interval(au[f"F_{f}"][ok], au["s_u"][ok])
            ok = ~np.isnan(r5[f"F_{f}"]) & ~np.isnan(r5["s_u"])
            row["r5"][f"F_{f}-s_u"] = R.interval(r5[f"F_{f}"][ok], r5["s_u"][ok])
        res[ds] = row
        log(f"report: {ds} measured")
    verdicts = {f: decide(cells[f]) for f in FAMS}
    out = {"stage": STAGE, "declared": DECLARED, "datasets": res, "cells": cells, "verdicts": verdicts, "bar": BAR,
           "code_sha256": sha_file(Path(__file__)), "written_utc": utc()}
    write_json(OUT / "report.json", out)
    (OUT / "report.md").write_text(render(out), encoding="utf-8")
    log(f"report: {verdicts}")
    return 0


def render(o):
    L = [f"# HR1 report ({o['written_utc']})", "", f"Declared in `{o['declared']}`. Verdicts: " +
         "; ".join(f"{f} {v['verdict']} (carry {v['carry']}, below {v['below']})" for f, v in o["verdicts"].items()), ""]
    L += ["| dataset | family | A | interval (A - 0.5) | call | counted / eligible | A_miss | golds covered | median deg gold / non-gold |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds, r in o["datasets"].items():
        for f, x in r["fam"].items():
            c = x["A_call"]
            L.append(f"| {ds} | {f} | {x['A']} | {c['ci']} | {c['call']} | {x['counted']} / {r['eligible']} | {x['A_miss']} | "
                     f"{x['coverage']['golds']} | {x['median_deg']['gold']} / {x['median_deg']['non_gold']} |")
    L += ["", "| dataset | AUC s_u | AUC twin0 | AUC F_ment | AUC F_co | R@5 s_u | R@5 F_ment | R@5 F_co |", "|---|---|---|---|---|---|---|---|"]
    for ds, r in o["datasets"].items():
        a, q = r["auc"], r["r5"]
        L.append(f"| {ds} | {a['s_u']} | {a['twin0']} | {a['F_ment']} ({a['F_ment-s_u']['call']}) | {a['F_co']} ({a['F_co-s_u']['call']}) | "
                 f"{q['s_u']} | {q['F_ment']} ({q['F_ment-s_u']['call']}) | {q['F_co']} ({q['F_co-s_u']['call']}) |")
    return "\n".join(L) + "\n"


# ── self-tests ───────────────────────────────────────────────────────────────


class _Nodes:
    def __init__(self, X):
        self.X, self.n_rows = X, X.shape[0]

    def read(self, rows, dtype=np.float32):
        return self.X[np.asarray(rows, np.int64)].astype(dtype)


def _store(n, pairs):
    a, b = pairs[:, 0], pairs[:, 1]
    h, t = np.concatenate((a, b)), np.concatenate((b, a))
    o = np.lexsort((t, h))
    h, t = h[o], t[o]
    ip = np.zeros(n + 1, np.int64)
    np.add.at(ip, h + 1, 1)
    return np.cumsum(ip), t.astype(np.int32)


def selftest():
    rng = np.random.default_rng(7)
    n = 400
    X = rng.standard_normal((n, DIM)).astype(np.float16)
    X[5] = 0                                                    # a zero row: its unit vector is zero
    stores = {}
    for f, m in (("ment", 900), ("co", 2500)):
        p = rng.integers(0, n - 40, size=(m, 2))                # nodes n-40.. are isolated
        p = p[p[:, 0] != p[:, 1]]
        p = np.unique(np.sort(p, axis=1), axis=0)
        stores[f] = _store(n, p)
    nq = 60
    sizes = rng.integers(0, 35, nq)
    sizes[3] = 0
    node = np.concatenate([rng.choice(n, s, replace=False) for s in sizes]).astype(np.int64)
    gold = rng.random(node.size) < 0.15
    off = np.concatenate(([0], np.cumsum(sizes)))
    gold[off[7]:off[8]] = True                                  # an all-gold pool
    Q = rng.standard_normal((nq, DIM)).astype(np.float16).astype(np.float32)
    Qhat = unit(Q)
    U, uidx = np.unique(node, return_inverse=True)
    uidx = uidx.astype(np.int64).ravel()
    qi = np.repeat(np.arange(nq), sizes)
    Xu = unit(X.astype(np.float32))
    # brute force
    bf_u = np.array([Xu[node[r]] @ Qhat[qi[r]] for r in range(node.size)], np.float32)
    bf_m, bf_d = {}, {}
    for f, (ip, col) in stores.items():
        sm = np.full(node.size, np.nan, np.float32)
        dd = np.zeros(node.size, np.int32)
        for r in range(node.size):
            j = node[r]
            nb = sorted(set(int(c) for c in col[ip[j]:ip[j + 1]]) - {int(j)})
            dd[r] = len(nb)
            if nb:
                s = Xu[nb].sum(0)
                if np.linalg.norm(s) > 0:
                    sm[r] = (s / np.linalg.norm(s)) @ Qhat[qi[r]]
        bf_m[f], bf_d[f] = sm, dd
    for g_max, chunk, emax in ((10 ** 6, 10 ** 6, EMAX), (7, 13, 50), (1, 1, 1)):
        s_u, s_m, deg = entity_columns(U, uidx, qi, Qhat, stores, _Nodes(X), g_max, chunk, emax)
        assert np.allclose(s_u, bf_u, atol=1e-5), "s_u"
        for f in stores:
            assert np.array_equal(np.isnan(s_m[f]), np.isnan(bf_m[f])), f"s_m nan {f}"
            ok = ~np.isnan(bf_m[f])
            assert np.allclose(s_m[f][ok], bf_m[f][ok], atol=1e-5), f"s_m {f}"
            assert np.array_equal(deg[f], bf_d[f]), f"deg {f}"
    # neighbours drops self and duplicates
    ip = np.array([0, 3, 4], np.int64)
    col = np.array([0, 1, 1, 0], np.int32)
    pos, nb = neighbours(ip, col, np.array([0, 1]))
    assert pos.tolist() == [0, 1] and nb.tolist() == [1, 0], (pos, nb)
    # matched pairs against a scan
    for _ in range(3000):
        m = int(rng.integers(1, 12))
        go = rng.random(m) < 0.4
        gp, ab, be = matched_pairs(go)
        for i, p in enumerate(gp):
            up = [x for x in range(p - 1, -1, -1) if not go[x]]
            dn = [x for x in range(p + 1, m) if not go[x]]
            assert ab[i] == (up[0] if up else -1) and be[i] == (dn[0] if dn else -1)
    # AUC against pairwise counting, with ties
    for _ in range(500):
        m = int(rng.integers(2, 20))
        s = rng.integers(0, 4, m).astype(float)
        g = rng.random(m) < 0.4
        if g.all() or not g.any():
            continue
        w = [1.0 if s[i] > s[j] else (0.5 if s[i] == s[j] else 0.0) for i in np.flatnonzero(g) for j in np.flatnonzero(~g)]
        assert abs(auc(s, g) - np.mean(w)) < 1e-12
    # R@5 ties by pool position
    s = np.array([1.0, 2.0, 2.0, 0.5, 2.0, 2.0, 2.0])
    g = np.array([1, 0, 0, 0, 0, 0, 1], bool)
    assert sorted_pos(s)[:5].tolist() == [1, 2, 4, 5, 6] and r_at(s, g, 2) == 0.5 and r_at(s, g, 2, k=4) == 0.0
    # fuse: eq. 5 with identity maps, and the fallback
    su, sm = np.array([0.2, 0.5]), np.array([0.6, np.nan])
    w = np.exp(0.6) / (np.exp(0.6) + np.exp(0.2))
    assert np.allclose(fuse(su, sm), [w * 0.6 + (1 - w) * 0.2, 0.5])
    # measure against a direct computation of A on the synthetic carve
    twin0 = rng.standard_normal(node.size).astype(np.float32)
    gt = np.array([max(1, int(gold[off[q]:off[q + 1]].sum()) + int(rng.integers(0, 2))) for q in range(nq)])
    M = measure(gold, sizes.astype(np.int64), gt, bf_u, bf_m, bf_d, twin0)
    for f in stores:
        vals = []
        for q in range(nq):
            g = gold[off[q]:off[q + 1]]
            if not g.any() or g.all():
                continue
            o = sorted_pos(bf_u[off[q]:off[q + 1]])
            go, smo = g[o], bf_m[f][off[q]:off[q + 1]][o]
            ws = []
            for p in np.flatnonzero(go):
                for x in [next((x for x in range(p - 1, -1, -1) if not go[x]), None),
                          next((x for x in range(p + 1, go.size) if not go[x]), None)]:
                    if x is None or np.isnan(smo[p]) or np.isnan(smo[x]):
                        continue
                    ws.append(1.0 if smo[p] > smo[x] else (0.5 if smo[p] == smo[x] else 0.0))
            if ws:
                vals.append(np.mean(ws))
        assert np.allclose(np.sort(M["fam"][f]["A_q"]), np.sort(vals)), f"A {f}"
        assert M["fam"][f]["counted"] == len(vals)
    assert M["eligible"] == sum(1 for q in range(nq) if gold[off[q]:off[q + 1]].any() and not gold[off[q]:off[q + 1]].all())
    # the rule
    base = {d: {"call": "ABOVE", "A": 0.6, "counted": 10, "eligible": 12} for d in CARVES}
    assert decide(base)["verdict"] == "CARRIED"
    b2 = dict(base, squad={"call": "BELOW", "A": 0.4, "counted": 10, "eligible": 12})
    assert decide(b2)["verdict"] == "NO_SIGNAL" and decide(b2)["below"] == ["squad"]
    b3 = dict(base, squad={"call": "LEVEL", "A": 0.51, "counted": 10, "eligible": 12})
    assert decide(b3)["verdict"] == "CARRIED"
    b4 = dict(b3, musique={"call": "ABOVE", "A": 0.54, "counted": 10, "eligible": 12})
    assert decide(b4)["verdict"] == "NO_SIGNAL"
    b5 = dict(b3, musique={"call": "ABOVE", "A": 0.7, "counted": 5, "eligible": 12})
    assert decide(b5)["verdict"] == "NO_SIGNAL"
    assert decide({d: base[d] for d in list(CARVES)[:3]})["verdict"] == "INCOMPLETE"
    # G2 refuses a perturbed column
    assert g2(bf_u, bf_u)["ok"] and not g2(bf_u, bf_u + 3e-3)["ok"] and not g2(bf_u, np.full_like(bf_u, np.nan))["ok"]
    # the interval helper is zd83rep's
    c = call_of(np.array([0.6, 0.7, 0.65, 0.8]))
    assert c["call"] == "ABOVE", c
    print("HR1 selftest: OK (entity sums, s_u, s_m, deg over three grouping sizes; pairs; AUC; R@5; fuse; A; rule; G2)")
    return 0


def selftest_host():
    """G1 (inside zd91.Stores) and G2 on squad's first cache part; no measure is computed or printed."""
    ds = "squad"
    C = load_carve(ds, CARVES[ds], first_part_only=True)
    stores, X, rsha, sinfo = open_world(ds, True)
    U, uidx = np.unique(C["node"], return_inverse=True)
    uidx = uidx.astype(np.int64).ravel()
    qi = np.repeat(np.arange(C["n"].size, dtype=np.int64), C["n"])
    s_u, _s_m, _deg = entity_columns(U, uidx, qi, unit(C["q"]), {}, X, 10 ** 6)
    gate = g2(s_u, C["dc"])
    print(f"HR1 host selftest: G1 {sinfo.get('g2d_stores')} ({sinfo.get('slot_pairs')} R.mention pairs, "
          f"{sinfo.get('ner_pairs')} R.co pairs); G2 {gate}; node record {rsha[:12]}; memory {peak_gb()}")
    return 0 if gate["ok"] and sinfo.get("g2d_stores") == "EQUAL" else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "cols", "report"])
    ap.add_argument("--ds", choices=sorted(CARVES))
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--budget-gb", type=float, default=4.0)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest_host() if a.host else selftest()
    if a.cmd == "cols":
        if not a.ds:
            raise SystemExit("cols: --ds is required")
        return cols(a.ds, a.host, a.budget_gb)
    return report()


if __name__ == "__main__":
    sys.exit(main())
