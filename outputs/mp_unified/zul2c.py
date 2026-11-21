"""UL2 section 12.6: carrying by budget (docs/UL2_LAYOUT_ON_E3V.md, section 12.6; the user on section 12's BASE_OVER_BOUND, 21 Nov:
"what? somehow make it carry"). The base takes F0T on the node's own text (a passage's body, a KB entity's name: UL1's filed F0T); a
form that would take a dataset's union over 1.30 x |REC| keeps its highest-ranked pairs, in its build's own score order, up to the room
left there; (a) and (b) of section 12.3 are read on what it keeps.

    python outputs/mp_unified/zul2c.py selftest
    python outputs/mp_unified/zul2c.py ntrank --ds D [--host]      # NTc's groups and NT1's scores (the written-name forms' orders)
    python outputs/mp_unified/zul2c.py grade --ds D [--host]       # each form's ranked pairs beyond B0, the slots, the sizes
    python outputs/mp_unified/zul2c.py report                      # the chain of section 12.6.3

Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit or record changes. Labels (gold positions, MuSiQue's
decompositions) are the grading only; no rule reads one. zul2u, zul2, zul1, zmhb, zap1 and zenc are imported, never edited; the tables
and written names are zul2's (passage datasets) and zul2u's (KBs), read by their build identities. Outputs only under outputs/ul2c/.
Arrays (npz) stay uncommitted.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zul2u as L2U  # noqa: E402  (imports zul2 -> zmhb -> zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zap1 as Z  # noqa: E402
import zmhb as MB  # noqa: E402
import zul1 as UL  # noqa: E402

L2 = L2U.L2
E = L2.E
DECLARED = "docs/UL2_LAYOUT_ON_E3V.md, section 12.6"
OUT = ROOT / "outputs" / "ul2c"
SIX, PASSAGE, KB = UL.SIX, UL.PASSAGE, UL.KB
VERSIONS = UL.DOMAINS
ORDER = L2U.ORDER                                  # H5, HM, HK1, NTc, NT1, HK2, H10
LABEL = L2U.LABEL
P_STAR = L2U.P_STAR                                # 4,997.1 (section 7, filed in 61fca78)
BOUND_NUM, BOUND_DEN = 130, 100                    # a union U fits when 100 U <= 130 |REC| (1.30, read in integers)
JUDGED = L2U.JUDGED
DECIDE, BESIDE = L2U.DECIDE, L2U.BESIDE
NT1_BLOCK = 1 << 16                                # NT1's pairs scored this many at a time (two rows each, float32)
SCORE = {"H5": "R.hub table score: float32 dot of the two stored e3v rows (tab_s; ov_s on flagged rows), a pair's highest",
         "H10": "as H5", "HM": "as H5 (H10's pairs picked from both ends)",
         "HK1": "the hub key's score (hk_s1), a pair's highest", "HK2": "the hub key's scores (hk_s1, hk_s2), a pair's highest",
         "NTc": "no score in its build: the group size (a key's namers + writers, at most 25), smallest first; stored as minus the size",
         "NT1": "its build's score: the namer-writer float32 dot of the stored e3v rows (recomputed by ntrank)"}
log = MB.log
assert BOUND_NUM / BOUND_DEN == L2U.BOUND


def stop(msg):
    raise SystemExit(f"UL2c: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes():
    return {"zul2c_sha256": script_sha256(), **L2U.codes()}


NR_FUNCS = ("ranked", "nt_groups", "Rows", "nt_filed", "cmd_ntrank")
G_FUNCS = ("ranked", "derive_scored", "table_forms", "e3v_table", "nt_filed", "load_ntrank", "rank_in", "build_ranks", "assemble",
           "union_count", "cmd_grade")


def build_sha(funcs):
    """A build's identity: the source of the functions it runs here, the identities of the zul2 and zul2u builds it reads, and the
    constants it reads (a change to the report alone does not stale a grade)."""
    h = hashlib.sha256()
    for f in funcs:
        h.update(inspect.getsource(globals()[f]).replace("\r\n", "\n").encode("utf-8"))
    h.update(json.dumps([L2.build_sha(L2.H_FUNCS), L2.build_sha(L2.NT_FUNCS), L2U.build_sha(L2U.H_FUNCS), L2U.build_sha(L2U.NT_FUNCS),
                         L2U.build_sha(L2U.KT_FUNCS), list(ORDER), L2.K_H, L2.EPS, L2.HUB_LO, UL.codes()], sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return Path(p).name


def peak_gb():
    """This process's peak working set and peak commit (GB): the next rows' memory asks are read from these (rx reads the wrapper only)."""
    try:
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k32, ps = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            if ps.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return {"peak_working_set_gb": round(c.PeakWorkingSetSize / 2 ** 30, 2), "peak_commit_gb": round(c.PeakPagefileUsage / 2 ** 30, 2)}
            return {"error": "GetProcessMemoryInfo failed"}
        import resource
        return {"peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 2)}
    except Exception as e:                                       # a record without its peak still stands
        return {"error": repr(e)[:200]}


# ── ranked lists: each form's pairs in its build's own score order ───────────


def ranked(a, b, s, n):
    """The unordered pairs of (a, b) ranked: score s descending (a pair seen more than once keeps its highest), the lower pair key
    (lo * n + hi) first on ties. Returns (keys, scores) in rank order."""
    a, b, s = np.asarray(a, np.int64), np.asarray(b, np.int64), np.asarray(s, np.float32)
    m = a != b
    k = np.minimum(a[m], b[m]) * n + np.maximum(a[m], b[m])
    s = s[m]
    if not np.isfinite(s).all():
        stop("a ranked pair without a finite score")
    o = np.lexsort((-s, k))
    k, s = k[o], s[o]
    first = np.r_[True, k[1:] != k[:-1]] if k.size else np.zeros(0, bool)
    k, s = k[first], s[first]
    o = np.lexsort((k, -s))
    return k[o], s[o]


def derive_scored(tab, n, k):
    """zul2.derive (variant 0, no S filter) with each directed pick's score: the table's tab_s, or the exact list's ov_s on a flagged row."""
    rows = np.arange(n)
    picked, _kth = L2.kth_picks(tab["tab_b"], tab["tab_s"], k)
    m = tab["ov_v"] == 0
    oa, ob, os_ = tab["ov_a"][m].astype(np.int64), tab["ov_b"][m].astype(np.int64), tab["ov_s"][m]
    picked[np.unique(oa)] = False
    a = np.broadcast_to(rows[:, None], picked.shape)[picked]
    b = tab["tab_b"][picked].astype(np.int64)
    s = tab["tab_s"][picked].astype(np.float32)
    if oa.size:
        st = np.r_[0, np.flatnonzero(oa[1:] != oa[:-1]) + 1]
        en = np.r_[st[1:], oa.size]
        A, B, S = [a], [b], [s]
        for s0, e0 in zip(st.tolist(), en.tolist()):
            ss = os_[s0:e0]
            kk = ss[k - 1] if ss.size >= k else -np.inf
            mm = ss >= kk - L2.EPS
            A.append(oa[s0:e0][mm])
            B.append(ob[s0:e0][mm])
            S.append(ss[mm].astype(np.float32))
        a, b, s = np.concatenate(A), np.concatenate(B), np.concatenate(S)
    return a, b, s


def table_forms(tab, n):
    """H5, H10, HM, HK1 and HK2 from e3v's R.hub table, each ranked by the table's own scores. Gate G-R: each list holds exactly the pairs
    zul2 derives (h_pairs, mutual_pairs, hk_pairs)."""
    out, ref = {}, {}
    a5, b5, s5 = derive_scored(tab, n, L2.K_H)
    out["H5"], ref["H5"] = ranked(a5, b5, s5, n), L2.h_pairs(tab, n, L2.K_H)
    del a5, b5, s5
    a, b, s = derive_scored(tab, n, 10)
    out["H10"], ref["H10"] = ranked(a, b, s, n), L2.h_pairs(tab, n, 10)
    hm = L2.mutual_pairs(tab, n, 10)
    m = UL.member(np.minimum(a, b) * n + np.maximum(a, b), hm)
    out["HM"], ref["HM"] = ranked(a[m], b[m], s[m], n), hm
    del a, b, s, m
    ha, hb1 = tab["hk_a"].astype(np.int64), tab["hk_b1"].astype(np.int64)
    m2 = tab["hk_b2"] >= 0
    out["HK1"], ref["HK1"] = ranked(ha, hb1, tab["hk_s1"], n), L2.hk_pairs(tab, n, 1)
    out["HK2"] = ranked(np.r_[ha, ha[m2]], np.r_[hb1, tab["hk_b2"][m2].astype(np.int64)], np.r_[tab["hk_s1"], tab["hk_s2"][m2]], n)
    ref["HK2"] = L2.hk_pairs(tab, n, 2)
    gate = {f: {"pairs": int(ref[f].size), "verdict": "PASS" if np.array_equal(np.sort(out[f][0]), ref[f]) else "FAIL"} for f in out}
    return out, gate


def e3v_table(d, n):
    """e3v's R.hub table: zul2's on the passage datasets (after gate G-H on frozen's, read from its record), zul2u's on the KBs."""
    gh = None
    if d in PASSAGE:
        if d in L2.GH_SETS:
            p, j = L2.OUT / d / "h_frozen.npz", L2.OUT / d / "h_frozen.json"
            rf = json.loads(j.read_text(encoding="utf-8")) if j.is_file() else {}
            if (rf.get("build_sha256") != L2.build_sha(L2.H_FUNCS) or not p.is_file() or Z.sha_file(p) != rf.get("npz_sha256")
                    or rf.get("gate_GH", {}).get("verdict") != "PASS"):
                stop(f"gate G-H has not passed on {d} (or frozen's table is not its record's): no e3v number is read")
            gh = "PASS"
        tab, r = L2.load_table(d, "e3v", n)
    else:
        tab, r = L2U.load_kb_table(d, "e3v", n)
    return tab, {"gate_GH": gh, **{k: r[k] for k in ("npz_sha256", "sizes", "flagged_rows", "build_sha256", "exclusion_pairs") if k in r}}


def nt_filed(d, n):
    """The filed written-name forms (NTc, NT1): zul2's on the passage datasets, zul2u's (rule text) on the KBs; and their record."""
    if d in KB:
        return L2U.kb_nt(d, n), rel(L2U.OUT / d / "nt_e3v.json")
    p, j = L2.OUT / d / "nt_e3v.npz", L2.OUT / d / "nt_e3v.json"
    if not p.is_file() or not j.is_file():
        stop(f"{d}: zul2's written-name forms (outputs/ul2/{d}/nt_e3v.npz) are not built")
    r = json.loads(j.read_text(encoding="utf-8"))
    if r["build_sha256"] != L2.build_sha(L2.NT_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n:
        stop(f"{d}: nt_e3v.npz is not zul2's build, or not its record's")
    with np.load(p) as z:
        return (z["ntc"], z["nt1"]), rel(j)


def nt_groups(names, wy, wk, n, cap=L2.HUB_LO):
    """NTc's pairs with the group each comes from (a key's namers plus writers, at most cap: the count NTc's cap reads), a pair from
    several keys keeping its smallest; zul2.nt_forms's NTc loop with the size kept. Ranked smallest group first (score = minus the size)."""
    nk = names.df.size
    wptr, wcol = MB.csr(wk, wy, nk)
    wdf = np.diff(wptr)
    tot = names.df + wdf
    ca, cb, cs = [], [], []
    small = np.flatnonzero((wdf > 0) & (names.df > 0) & (tot <= cap))
    for kk in small.tolist():
        P = names.key_node[names.key_ptr[kk]:names.key_ptr[kk + 1]].astype(np.int64)
        W = wcol[wptr[kk]:wptr[kk + 1]].astype(np.int64)
        ca.append(np.repeat(P, W.size))
        cb.append(np.tile(W, P.size))
        cs.append(np.full(P.size * W.size, -float(tot[kk]), np.float32))
    if not ca:
        return np.zeros(0, np.int64), np.zeros(0, np.float32), int(small.size)
    k, s = ranked(np.concatenate(ca), np.concatenate(cb), np.concatenate(cs), n)
    return k, s, int(small.size)


class Rows:
    """The stored e3v document rows through memory maps, a few at a time (NT1's scores need two rows per pair, never every row)."""

    def __init__(self, maps):
        self.maps = maps
        self.starts = np.cumsum([0] + [int(m.shape[0]) for m in maps])
        self.dim = int(maps[0].shape[1])

    @classmethod
    def of(cls, served, d, n):
        r = cls([np.load(p, mmap_mode="r") for p in E.vec_shards(served, "e3v", d, "docs")])
        if int(r.starts[-1]) != n:
            stop(f"e3v {d}: {int(r.starts[-1])} vector rows for {n} nodes")
        return r

    def take(self, idx):
        idx = np.asarray(idx, np.int64)
        out = np.empty((idx.size, self.dim), np.float32)
        sh = np.searchsorted(self.starts, idx, side="right") - 1
        for j in np.unique(sh).tolist():
            w = np.flatnonzero(sh == j)
            loc = idx[w] - self.starts[j]
            o = np.argsort(loc, kind="stable")
            out[w[o]] = np.asarray(self.maps[j][loc[o]], np.float32)
        return out


# ── ntrank: the written-name forms' orders ───────────────────────────────────


def cmd_ntrank(a):
    d, t0 = a.ds, time.time()
    served, freeze, n, nsha = L2.dataset_head(a)
    if d in KB:
        _f0tu, kt = L2U.load_f0tu(d, n)
        names, vocab, passes = L2U.names_u(served, d, n, nsha, kt)
        text = "ABL-U's rule text (section 12.1)"
    else:
        names, vocab, passes = L2.names_of(served, d, n, nsha)
        text = "the passage's own text"
    (ntc, nt1), nrec = nt_filed(d, n)
    wy, wk = L2.written_entries(passes, vocab, names, n)
    del vocab, passes
    log(f"{d}: {wy.size:,} written entries ({time.time() - t0:.0f} s)")
    ck, cs, nkeys = nt_groups(names, wy, wk, n)
    del names, wy, wk
    gn = {"ntc_pairs": int(ntc.size), "ntc_recomputed": int(ck.size), "ntc_keys": nkeys, "ntc_same": bool(np.array_equal(np.sort(ck), ntc))}
    rows = Rows.of(served, d, n)
    a1, b1 = nt1 // n, nt1 % n
    s1 = np.empty(nt1.size, np.float32)
    for i in range(0, nt1.size, NT1_BLOCK):
        s1[i:i + NT1_BLOCK] = (rows.take(a1[i:i + NT1_BLOCK]) * rows.take(b1[i:i + NT1_BLOCK])).sum(1)
    k1, s1r = ranked(a1, b1, s1, n)
    del a1, b1, s1
    gn.update(nt1_pairs=int(nt1.size), nt1_scored=int(k1.size), nt1_same=bool(np.array_equal(np.sort(k1), nt1)))
    gn["verdict"] = "PASS" if gn["ntc_same"] and gn["nt1_same"] else "FAIL"
    log(f"{d}: gate G-N {gn['verdict']} ({gn}) ({time.time() - t0:.0f} s)")
    if gn["verdict"] != "PASS":
        stop(f"gate G-N: {d}: the recomputed NTc or the scored NT1 is not the filed form")
    sha = L2.save_npz(OUT / d / "ntrank.npz", ntc_keys=ck, ntc_score=cs, nt1_keys=k1, nt1_score=s1r)
    grp = -cs
    rec = {"dataset": d, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze, "text": text,
           "nt_record": nrec, "vectors": {"enc": "e3v", "shards": len(rows.maps), "dim": rows.dim,
                                          "manifest_sha256": E.sha256(E.vec_dir(served, "e3v", d, "docs") / "manifest.json")},
           "gate_GN": gn, "ntc_group_sizes": {str(int(g)): int((grp == g).sum()) for g in np.unique(grp)} if grp.size else {},
           "nt1_score": {"max": float(s1r[0]), "min": float(s1r[-1])} if s1r.size else None, "order": {f: SCORE[f] for f in ("NTc", "NT1")},
           "npz_sha256": sha, "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "memory": peak_gb(), "build_sha256": build_sha(NR_FUNCS), **codes()}
    L2.wjson(OUT / d / "ntrank.json", rec)
    log(f"{d}: NTc {ck.size:,} pairs in {nkeys:,} groups, NT1 {k1.size:,} scored; written {rel(OUT / d / 'ntrank.json')} ({rec['seconds']} s)")
    return 0


def load_ntrank(d, n):
    p, j = OUT / d / "ntrank.npz", OUT / d / "ntrank.json"
    if not p.is_file() or not j.is_file():
        stop(f"{d}: the written-name orders are not built (ntrank first)")
    r = json.loads(j.read_text(encoding="utf-8"))
    if (r["build_sha256"] != build_sha(NR_FUNCS) or Z.sha_file(p) != r["npz_sha256"] or int(r["nodes"]) != n
            or r["gate_GN"]["verdict"] != "PASS"):
        stop(f"{d}: ntrank.npz is not this code's build, or not its record's")
    (ntc, nt1), _ = nt_filed(d, n)
    with np.load(p) as z:
        out = {"NTc": (z["ntc_keys"], z["ntc_score"]), "NT1": (z["nt1_keys"], z["nt1_score"])}
    if not (np.array_equal(np.sort(out["NTc"][0]), ntc) and np.array_equal(np.sort(out["NT1"][0]), nt1)):
        stop(f"gate G-N: {d}: ntrank's lists are not the filed NTc and NT1")
    return out, {"npz_sha256": r["npz_sha256"], "gate_GN": r["gate_GN"]}


# ── grade: the pairs beyond B0 with their ranks; the slots; the sizes ─────────


def rank_in(keys, x):
    """For each of x: its position in the ranked list keys (-1 when absent)."""
    x = np.asarray(x, np.int64)
    if keys.size == 0 or x.size == 0:
        return np.full(x.size, -1, np.int32)
    o = np.argsort(keys, kind="stable")
    ks = keys[o]
    i = np.searchsorted(ks, x)
    i[i == ks.size] = ks.size - 1
    return np.where(ks[i] == x, o[i], -1).astype(np.int32)


def build_ranks(forms, in_base, extra):
    """allF: every form's pairs beyond the base (sorted); per form, each one's rank in the form's list (-1: not in it), the form's scores in
    its own rank order (the score at any rank r is fscore[r]); and the ranks of each array of extra (the slots)."""
    parts = [forms[f][0][~in_base(forms[f][0])] for f in ORDER]
    allF = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
    del parts
    sizes = [allF.size] + [x.size for x in extra]
    cuts = np.cumsum(sizes)[:-1]
    X = np.concatenate([allF] + list(extra))
    A, X_r = {}, {}
    for f in ORDER:
        k, s = forms[f]
        r = rank_in(k, X)
        A[f"rank__{f}"] = r[:allF.size]
        A[f"fscore__{f}"] = np.asarray(s, np.float32)
        X_r[f] = np.split(r, cuts)[1:]
    return allF, A, X_r


def assemble(forms, in_base, Skeys, vers, cells, cover):
    """The arrays one dataset's chain reads: the ranks and scores over allF, S membership (SRK), and each cell's question index, slot
    ranks and B0 cover per version. cells: {(carve, kind): (qi, sk, nq)}; cover(sk, version) -> bool."""
    keys = list(cells)
    allF, A, X_r = build_ranks(forms, in_base, [cells[c][1] for c in keys])
    A["inS"] = UL.member(allF, Skeys) if "SRK" in vers else np.zeros(0, bool)
    for j, (cv, kind) in enumerate(keys):
        qi, sk, nq = cells[(cv, kind)]
        pre = f"{cv}__{kind}__"
        A[pre + "qi"], A[pre + "nq"] = np.asarray(qi, np.int64), np.asarray(nq, np.int64)
        for f in ORDER:
            A[pre + "srank__" + f] = X_r[f][j]
        for v in vers:
            A[f"{pre}{v}__B0"] = cover(sk, v)
    return allF, A


def union_count(parts):
    """|p1 | p2 | ...| for sorted unique key arrays, counted part by part (no union is built): largest first is cheapest."""
    tot = 0
    for i, p in enumerate(parts):
        m = np.ones(p.size, bool)
        for q in parts[:i]:
            m &= ~UL.member(p, q)
        tot += int(m.sum())
    return tot


def cmd_grade(a):
    d, t0 = a.ds, time.time()
    served, freeze, n, nsha = L2.dataset_head(a)
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)
    del F["N_sp"], F["CN_all"]
    Ke, krec = L2.knn_keys(served, d, n, nsha)
    tab, trec = e3v_table(d, n)
    forms, g_r = table_forms(tab, n)
    del tab
    if any(v["verdict"] != "PASS" for v in g_r.values()):
        stop(f"gate G-R: {d}: {g_r}")
    nt, nrec = load_ntrank(d, n)
    forms.update(nt)
    log(f"{d}: ranked " + ", ".join(f"{f} {forms[f][0].size:,}" for f in ORDER) + f" ({time.time() - t0:.0f} s)")

    def in_base(x):                                              # B0 on RK: W2_S | F0T (UL1's filed) | K_e3v
        return UL.member(x, F["W2_S"]) | UL.member(x, F["F0T"]) | UL.member(x, Ke)

    def in_rec(x):
        return UL.member(x, F["W2_S"]) | UL.member(x, F["F0T"]) | UL.member(x, F["K"])

    S = F.get("S", np.zeros(0, np.int64))
    cells, ul1, nqs = {}, json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8")), {}
    for cv in UL.CARVES[d]:
        ids, kinds, _hops, qsha = L2.slot_kinds(served, d, n, cv)
        for kind, (qi, sk) in kinds.items():
            if sk.size:
                cells[(cv, kind)] = (qi, sk, len(ids))
                nqs[(cv, kind)] = qsha
    vers = VERSIONS[d]
    allF, A = assemble(forms, in_base, S, vers, cells,
                       lambda sk, v: in_base(sk) | (UL.member(sk, S) if v == "SRK" else False))
    form_pairs = {f: int(forms[f][0].size) for f in ORDER}
    srange = {f: ({"first": float(forms[f][1][0]), "last": float(forms[f][1][-1])} if forms[f][1].size else None) for f in ORDER}
    del forms
    new_over = {f: int((A[f"rank__{f}"] >= 0).sum()) for f in ORDER}
    sizes = {"REC": {}, "B0": {}}
    for v in vers:
        Sv = [S] if v == "SRK" else []
        sizes["REC"][v] = union_count([F["W2_S"]] + Sv + [F["F0T"], F["K"]])
        sizes["B0"][v] = union_count([F["W2_S"]] + Sv + [F["F0T"], Ke])
    log(f"{d}: {allF.size:,} pairs beyond B0; sizes {sizes} ({time.time() - t0:.0f} s)")
    against = {}
    for name, p, key in (("zul2u", L2U.OUT / d / "grade.json", "N"), ("zul2", L2.OUT / d / "grade.json", "L2e" if d in KB else None)):
        if p.is_file():
            zg = json.loads(p.read_text(encoding="utf-8"))
            for v in vers:
                if zg["sizes"]["REC"][v] != sizes["REC"][v] or (key and zg["sizes"][key][v] != sizes["B0"][v]):
                    stop(f"{d}/{v}: REC or B0 is not {name}'s filed REC or {key}")
            against[name] = {"grade_sha256": Z.sha_file(p), "REC": "same", **({key: "same as B0"} if key else {})}
    res = {"dataset": d, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze, "kb": d in KB,
           "versions": list(vers), "inputs": shas, "knn_e3v": krec, "table": trec, "ntrank": nrec, "gate_GR": g_r,
           "order": SCORE, "form_pairs": form_pairs, "score_range": srange, "pairs_beyond_B0": {"any": int(allF.size), **new_over},
           "in_S": int(A["inS"].sum()), "sizes": sizes,
           "ratio_to_REC": {v: round(sizes["B0"][v] / sizes["REC"][v], 4) for v in vers},
           "room": {v: BOUND_NUM * sizes["REC"][v] // BOUND_DEN - sizes["B0"][v] for v in vers}, "against": against, "cells": {}}
    for (cv, kind), (qi, sk, nq) in cells.items():
        pre = f"{cv}__{kind}__"
        rc = {"questions": nq, "slots": int(sk.size), "questions_npz_sha256": nqs[(cv, kind)]}
        for v in vers:
            rec_cov = in_rec(sk) | (UL.member(sk, S) if v == "SRK" else False)
            want = ul1["carves"][cv][v][kind]["forms"]["F0T"]["linked"]
            if int(rec_cov.sum()) != int(want):
                stop(f"gate G-U: {d}/{cv}/{v}/{kind}: REC links {int(rec_cov.sum())}, UL1 filed {want}")
            rc[v] = {"linked": {"REC": int(rec_cov.sum()), "B0": int(A[f"{pre}{v}__B0"].sum())},
                     "B0-REC": L2.diff_row(A[f"{pre}{v}__B0"], rec_cov, qi, nq)}
        res["cells"].setdefault(cv, {})[kind] = rc
    res["ranks_npz_sha256"] = L2.save_npz(OUT / d / "ranks.npz", allF=allF, **A)
    res.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), memory=peak_gb(),
               build_sha256=build_sha(G_FUNCS), **codes())
    L2.wjson(OUT / d / "grade.json", res)
    log(f"{d}: graded in {res['seconds']} s ({res['memory']}); written {rel(OUT / d / 'grade.json')}")
    return 0


# ── the chain (section 12.6.3) ───────────────────────────────────────────────


class State:
    """One dataset in the chain: the running union's size per version, which pairs beyond B0 it holds, and the carried forms' cuts."""

    def __init__(self, d, g, z):
        self.d, self.vers = d, list(g["versions"])
        self.rank = {f: z[f"rank__{f}"] for f in ORDER}
        self.fscore = {f: z[f"fscore__{f}"] for f in ORDER}
        self.inS = z["inS"]
        self.inU = np.zeros(self.rank[ORDER[0]].size, bool)
        self.rec = {v: int(g["sizes"]["REC"][v]) for v in self.vers}
        self.size = {v: int(g["sizes"]["B0"][v]) for v in self.vers}
        self.base = dict(self.size)
        self.cap = {v: BOUND_NUM * self.rec[v] // BOUND_DEN for v in self.vers}
        self.fsize = {f: int(g["form_pairs"][f]) for f in ORDER}
        self.cut, self.cells = {}, {}
        for cv, kinds in g["cells"].items():
            for kind in kinds:
                pre = f"{cv}__{kind}__"
                self.cells[(cv, kind)] = {"qi": z[pre + "qi"], "nq": int(z[pre + "nq"]), "srank": {f: z[pre + "srank__" + f] for f in ORDER},
                                          "cov": {v: z[f"{pre}{v}__B0"] for v in self.vers}}


def cut_of(st, f, budget=True):
    """Form f on one dataset over the running union: its candidates (pairs beyond the union) in rank order, kept up to the longest prefix
    whose new pairs fit every version's room (budget) or all of them; the rank cut (the kept part is every pair of f ranked before it)."""
    r = st.rank[f]
    cand = np.flatnonzero((r >= 0) & ~st.inU)
    cand = cand[np.argsort(r[cand], kind="stable")]
    t, cums = cand.size, {}
    for v in st.vers:
        cost = np.ones(cand.size, np.int64) if v == "RK" else (~st.inS[cand]).astype(np.int64)
        cums[v] = np.cumsum(cost)
        if budget:
            t = min(t, int(np.searchsorted(cums[v], st.cap[v] - st.size[v], side="right")))
    add = {v: int(cums[v][t - 1]) if t else 0 for v in st.vers}
    cutrank = int(r[cand[t]]) if t < cand.size else st.fsize[f]
    sc, sg = st.fscore[f], (-1.0 if f == "NTc" else 1.0)                    # NTc's stored score is minus its group size
    info = {"new_pairs": int(cand.size), "kept_new_pairs": int(t), "kept_rank": cutrank, "of": st.fsize[f],
            "cut": bool(t < cand.size), "score": "group size" if f == "NTc" else "score",
            "last_kept": float(sg * sc[r[cand[t - 1]]]) if t else None,
            "first_dropped": float(sg * sc[r[cand[t]]]) if t < cand.size else None}
    return cand[:t], cutrank, add, info


def read_cells(st, f, cutrank, add, carve):
    out = []
    for kind in JUDGED[st.d]:
        c = st.cells.get((carve, kind))
        if c is None:
            continue
        sr = c["srank"]
        held = np.zeros(sr[f].size, bool)
        for g, cr in st.cut.items():
            held |= (sr[g] >= 0) & (sr[g] < cr)
        mine = (sr[f] >= 0) & (sr[f] < cutrank)
        for v in st.vers:
            covc = c["cov"][v] | held
            cf = covc | mine
            new, ad, slots = int((cf & ~covc).sum()), add[v], int(cf.size)
            out.append({"dataset": st.d, "version": v, "kind": kind, "carve": carve, "slots": slots, "new": new, "pairs_added": ad,
                        "price": round(ad / new, 1) if new else None,
                        "scale_free": round((ad / st.rec[v]) / (new / slots), 3) if new else None,
                        "gain": L2.diff_row(cf, covc, c["qi"], c["nq"])})
    return out


def step(D, f, budget=True):
    per, cells = {}, []
    for d, st in D.items():
        kept, cutrank, add, info = cut_of(st, f, budget)
        size = {v: {"pairs_added": add[v], "union": st.size[v] + add[v], "ratio_to_REC": round((st.size[v] + add[v]) / st.rec[v], 4),
                    "fits": BOUND_DEN * (st.size[v] + add[v]) <= BOUND_NUM * st.rec[v]} for v in st.vers}
        per[d] = {"cut": info, "size": size, "_kept": kept, "_cutrank": cutrank, "_add": add}
        for carve in (DECIDE, BESIDE):
            cells += read_cells(st, f, cutrank, add, carve)
    return per, cells


def public(per):
    return {d: {k: v for k, v in p.items() if not k.startswith("_")} for d, p in per.items()}


def chain(D):
    """Section 12.6.3: the forms in ORDER over the running union, each cut to its budget on every dataset; carried when (a) ABOVE in a
    judged s1eval cell and BELOW in none and (b) every judged s1eval cell's price at most P* where the form adds slots, both on what it
    keeps. (c) holds by construction (checked). A form not carried takes no room."""
    carried, steps = [], []
    for f in ORDER:
        per, cells = step(D, f)
        bad = [f"{d}/{v}" for d, p in per.items() for v, s in p["size"].items() if p["_add"][v] and not s["fits"]]
        if bad:
            stop(f"condition (c) does not hold by construction for {f}: {bad}")
        unc, ucells = step(D, f, budget=False)
        dec = [x for x in cells if x["carve"] == DECIDE]
        above = any(x["gain"]["call"] == "ABOVE" for x in dec)
        below = any(x["gain"]["call"] == "BELOW" for x in dec)
        dear = [{"cell": f"{x['dataset']}/{x['version']}/{x['kind']}", "price": x["price"], "over_P_star": round(x["price"] - P_STAR, 1)}
                for x in dec if x["new"] and x["price"] > P_STAR]
        ok = above and not below and not dear
        steps.append({"form": f, "label": LABEL[f], "order": SCORE[f], "over": list(carried), "carried": bool(ok),
                      "conditions": {"a_gain": bool(above and not below), "b_price": not dear, "c_size": "by construction"},
                      "price_over_P_star": dear, "datasets": public(per), "cells": dec,
                      "beside": {"s1sel": [x for x in cells if x["carve"] == BESIDE], "uncut": {
                          "datasets": public(unc), "cells": [x for x in ucells if x["carve"] == DECIDE]}}})
        if ok:
            carried.append(f)
            for d, st in D.items():
                p = per[d]
                st.inU[p["_kept"]] = True
                for v in st.vers:
                    st.size[v] += p["_add"][v]
                st.cut[f] = p["_cutrank"]
    return carried, steps


def load_all():
    D, G, bad = {}, {}, []
    for d in SIX:
        j, p = OUT / d / "grade.json", OUT / d / "ranks.npz"
        if not j.is_file() or not p.is_file():
            bad.append(d)
            continue
        g = json.loads(j.read_text(encoding="utf-8"))
        if g.get("build_sha256") != build_sha(G_FUNCS) or Z.sha_file(p) != g["ranks_npz_sha256"]:
            bad.append(d)
            continue
        with np.load(p) as z:
            D[d] = State(d, g, {k: z[k] for k in z.files if k != "allF"})
        G[d] = g
    return D, G, bad


def write_carried(D, carried):
    """Each dataset's kept pairs beyond B0 and, per pair, the carried forms whose kept part holds it (bit i: carried[i]), for the labels."""
    out = {}
    for d, st in D.items():
        with np.load(OUT / d / "ranks.npz") as z:
            allF = z["allF"]
        idx = np.flatnonzero(st.inU)
        code = np.zeros(idx.size, np.uint8)
        for i, f in enumerate(carried):
            r = st.rank[f][idx]
            code |= ((r >= 0) & (r < st.cut[f])).astype(np.uint8) << np.uint8(i)
        out[d] = L2.save_npz(OUT / d / "carried.npz", keys=allF[idx], code=code, forms=np.asarray(carried, dtype="U8"))
    return out


def cmd_report(_a):
    D, G, bad = load_all()
    if bad:
        L2.wjson(OUT / "report.json", {"declared_in": DECLARED, "verdict": "REFUSED", "missing_or_other_code": bad, **codes()})
        log(f"REFUSED: no grade of this code for {bad}")
        return 1
    mp = L2.OUT / "musique" / "grade.json"
    mg = json.loads(mp.read_text(encoding="utf-8"))
    if mg["part_b_chain"]["P_star"] != P_STAR:
        stop(f"P* {P_STAR} is not section 7's filed {mg['part_b_chain']['P_star']}")
    over = [{"dataset": d, "version": v, "ratio": round(st.base[v] / st.rec[v], 4)} for d, st in D.items() for v in st.vers
            if BOUND_DEN * st.base[v] > BOUND_NUM * st.rec[v]]
    carried, steps = chain(D)
    sha = write_carried(D, carried)
    layout = {d: {"forms": {f: next(s["datasets"][d]["cut"] for s in steps if s["form"] == f) for f in carried},
                  "union": dict(st.size), "ratio_to_REC": {v: round(st.size[v] / st.rec[v], 4) for v in st.vers},
                  "carried_npz_sha256": sha[d]} for d, st in D.items()}
    s12 = L2U.OUT / "report.json"
    sec12 = None
    if s12.is_file():
        r12 = json.loads(s12.read_text(encoding="utf-8"))
        sec12 = {"report_sha256": Z.sha_file(s12), "U": {k: r12.get("U", {}).get(k) for k in ("verdict", "forms")},
                 "U_name": {k: r12.get("beside", {}).get("U_name", {}).get(k) for k in ("verdict", "forms")}}
    rec = {"declared_in": DECLARED, "datasets": list(D), "pending": ["multihop-rag (joins when MHR1 files its package)"], "P_star": P_STAR,
           "bound": f"{BOUND_NUM}/{BOUND_DEN}", "boot": L2.BOOT, "seed": L2.SEED, "order": list(ORDER), "scores": SCORE,
           "verdict": "CARRIED" if carried else "NONE_CARRIED", "forms": carried, "base_over_bound": over, "layout": layout,
           "steps": steps, "beside": {"section_12": sec12, "musique_only_chain_as_filed": {
               "carried": mg["part_b_chain"]["carried"], "zul2_grade_sha256": Z.sha_file(mp)}},
           "bases": {d: {"sizes": g["sizes"], "ratio_to_REC": g["ratio_to_REC"]} for d, g in G.items()},
           "grades_sha256": {d: Z.sha_file(OUT / d / "grade.json") for d in G}, **codes()}
    L2.wjson(OUT / "report.json", rec)
    (OUT / "report.md").write_text(render(rec), encoding="utf-8", newline="\n")
    log(f"section 12.6: {rec['verdict']} {carried}; base over the bound: {over}")
    return 0


def _g(c):
    g = c["gain"]
    return f"{g['diff']:+.2f} [{g['lo']:+.2f}, {g['hi']:+.2f}] {g['call']}"


def render(rec):
    L = [f"# UL2 section 12.6: carrying by budget\n", f"Verdict: **{rec['verdict']}** {rec['forms']}. P* = {rec['P_star']}; bound "
         f"{rec['bound']} x |REC|; {rec['boot']} resamples, seed {rec['seed']}.\n"]
    if rec["base_over_bound"]:
        L.append(f"Base over the bound (the forms keep nothing there): {rec['base_over_bound']}\n")
    L.append("| form | carried | (a) | (b) | kept new pairs / new pairs, per dataset |\n|---|---|---|---|---|")
    for s in rec["steps"]:
        kept = "; ".join(f"{d} {p['cut']['kept_new_pairs']:,}/{p['cut']['new_pairs']:,}" for d, p in s["datasets"].items())
        L.append(f"| {s['form']} | {s['carried']} | {s['conditions']['a_gain']} | {s['conditions']['b_price']} | {kept} |")
    L.append("\n## Judged cells (s1eval), on what each form keeps\n\n| form | cell | new | pairs added | price | gain (points) |"
             "\n|---|---|---|---|---|---|")
    for s in rec["steps"]:
        for c in s["cells"]:
            L.append(f"| {s['form']} | {c['dataset']}/{c['version']}/{c['kind']} | {c['new']} | {c['pairs_added']:,} | {c['price']} | {_g(c)} |")
    L.append("\n## Layout\n")
    for d, x in rec["layout"].items():
        L.append(f"- {d}: union {x['union']}, ratio to REC {x['ratio_to_REC']}; " + "; ".join(
            f"{f} kept rank {c['kept_rank']:,} of {c['of']:,} (last kept {c['score']} {c['last_kept']})" for f, c in x["forms"].items()))
    return "\n".join(L) + "\n"


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    # ranked: highest score per pair, ties to the lower key
    n = 40
    a, b = rng.integers(0, n, 300), rng.integers(0, n, 300)
    s = rng.choice(np.linspace(0, 1, 7).astype(np.float32), 300)
    k, sc = ranked(a, b, s, n)
    best = {}
    for x, y, z in zip(a.tolist(), b.tolist(), s.tolist()):
        if x != y:
            kk = min(x, y) * n + max(x, y)
            best[kk] = max(best.get(kk, -9.0), z)
    want = sorted(best, key=lambda kk: (-best[kk], kk))
    assert k.tolist() == want and np.allclose(sc, [best[kk] for kk in want])
    # the table's forms with their scores against a direct reading (ties, flagged rows, hub keys); gate G-R
    n, R = 60, 6
    tb, ts = np.full((n, R), -1, np.int32), np.full((n, R), -np.inf, np.float32)
    for i in range(n):
        m = int(rng.integers(0, R + 1))
        tb[i, :m] = rng.choice(np.delete(np.arange(n), i), m, replace=False)
        ts[i, :m] = np.sort(rng.choice(np.linspace(0, 1, 9), m))[::-1]
    oa, ob, os_ = [], [], []
    for i in (3, 9, 31):
        c = rng.choice(np.delete(np.arange(n), i), 14, replace=False)
        oa += [i] * c.size
        ob += c.tolist()
        os_ += np.sort(rng.choice(np.linspace(0, 1, 9), c.size))[::-1].tolist()
    ha = rng.integers(0, n, 80)
    hb1, hb2 = (ha + rng.integers(1, n, 80)) % n, np.where(rng.random(80) < 0.6, (ha + rng.integers(1, n, 80)) % n, -1)
    hs1 = rng.choice(np.linspace(0, 1, 9), 80).astype(np.float32)
    tab = {"tab_b": tb, "tab_s": ts, "tmax": np.full(n, -np.inf, np.float32), "ov_a": np.asarray(oa, np.int32),
           "ov_v": np.zeros(len(oa), np.int8), "ov_b": np.asarray(ob, np.int32), "ov_s": np.asarray(os_, np.float32),
           "hk_a": ha.astype(np.int32), "hk_b1": hb1.astype(np.int32), "hk_s1": hs1, "hk_b2": hb2.astype(np.int32),
           "hk_s2": np.where(hb2 >= 0, hs1 - 0.05, -np.inf).astype(np.float32)}
    forms, gate = table_forms(tab, n)
    assert all(v["verdict"] == "PASS" for v in gate.values()), gate

    def direct(k_):
        picks = {}
        for i in range(n):
            if i in (3, 9, 31):
                ent = [(ob[j], os_[j]) for j in range(len(oa)) if oa[j] == i]
            else:
                ent = [(int(tb[i, j]), float(ts[i, j])) for j in range(R) if tb[i, j] >= 0]
            kth = ent[k_ - 1][1] if len(ent) >= k_ else -np.inf
            for y, z in ent:
                if z >= np.float32(kth) - L2.EPS:
                    picks[(i, y)] = max(picks.get((i, y), -9.0), np.float32(z))
        return picks

    def rank_dict(dct):
        best_ = {}
        for (x, y), z in dct.items():
            kk = min(x, y) * n + max(x, y)
            best_[kk] = max(best_.get(kk, -9.0), float(z))
        o = sorted(best_, key=lambda kk: (-best_[kk], kk))
        return o, [best_[kk] for kk in o]
    for f, k_ in (("H5", 5), ("H10", 10)):
        o, z = rank_dict(direct(k_))
        assert forms[f][0].tolist() == o and np.allclose(forms[f][1], z), f
    p10 = direct(10)
    o, z = rank_dict({(x, y): v for (x, y), v in p10.items() if (y, x) in p10})
    assert forms["HM"][0].tolist() == o and np.allclose(forms["HM"][1], z)
    hk1 = {}
    for x, y, z in zip(ha.tolist(), hb1.tolist(), hs1.tolist()):
        hk1[(x, y)] = max(hk1.get((x, y), -9.0), z)
    o, z = rank_dict(hk1)
    assert forms["HK1"][0].tolist() == o and np.allclose(forms["HK1"][1], z)
    hk2 = dict(hk1)
    for x, y, z in zip(ha.tolist(), hb2.tolist(), tab["hk_s2"].tolist()):
        if y >= 0:
            hk2[(x, y)] = max(hk2.get((x, y), -9.0), z)
    o, z = rank_dict(hk2)
    assert forms["HK2"][0].tolist() == o and np.allclose(forms["HK2"][1], z)
    # NTc's groups against zul2.nt_forms's NTc and a direct reading; NT1's rows through the maps
    n, nk = 40, 25
    ent = np.unique(rng.integers(0, n, 90) * nk + rng.integers(0, nk, 90))
    names = MB.Names((ent // nk).astype(np.int64), (ent % nk).astype(np.int64), nk, n)
    wr = np.unique(rng.integers(0, n, 70) * nk + rng.integers(0, nk, 70))
    wy, wk = (wr // nk).astype(np.int64), (wr % nk).astype(np.int64)
    V = rng.standard_normal((n, 8)).astype(np.float16)
    ntc_ref, nt1_ref, _st = L2.nt_forms(names, wy, wk, V, np.zeros(n + 1, np.int64), np.zeros(0, np.int32), n, cap=8, dfmax=100)
    ck, cs, nkeys = nt_groups(names, wy, wk, n, cap=8)
    assert np.array_equal(np.sort(ck), ntc_ref) and ck.size > 10 and nt1_ref.size > 0, (ck.size, ntc_ref.size)
    grp = {}
    for kk in range(nk):
        P = sorted(set((ent // nk)[(ent % nk) == kk].tolist()))
        W = sorted(set(wy[wk == kk].tolist()))
        if P and W and len(P) + len(W) <= 8:
            for x in P:
                for y in W:
                    if x != y:
                        q = min(x, y) * n + max(x, y)
                        grp[q] = min(grp.get(q, 99), len(P) + len(W))
    want = sorted(grp, key=lambda q: (grp[q], q))
    assert ck.tolist() == want and (-cs).tolist() == [grp[q] for q in want] and nkeys > 0
    rows = Rows([V[:13], V[13:29], V[29:]])
    idx = rng.integers(0, n, 50)
    assert np.array_equal(rows.take(idx), V[idx].astype(np.float32))
    # union_count against np.union1d
    P3 = [np.unique(rng.integers(0, 500, m)) for m in (300, 120, 40, 0, 90)]
    assert union_count(P3) == L2.union_of(P3).size
    # the chain against a direct set reading: cuts that bind (MuSiQue, SQuAD), a form failing (b) (MetaQA), S taking no room in SRK,
    # a base over the bound (WebQSP: nothing kept there), a form with nothing new
    BIG = 10_000_000

    def slot(q, o):
        return (2 * q + o) * BIG + (2 * q + o + 1)

    def junk(lo, m):
        return (np.arange(lo, lo + m, dtype=np.int64) + 5_000_000) * BIG + 9_999_999
    nq, qs = 200, np.arange(200, dtype=np.int64)
    gp, ch, kq = slot(qs, 0), slot(qs, 600), slot(qs[:100], 1_000)
    Smq = junk(70_000, 200)
    spec = {
        "musique": {"vers": ("RK",), "rec": {"RK": 4000}, "base": np.union1d(gp[:40], junk(0, 4000)), "S": None,
                    "cells": {("s1eval", "GP"): gp, ("s1eval", "CH"): ch, ("s1sel", "GP"): gp[::-1].copy()},
                    "forms": {"H5": [gp[40:60], junk(10_000, 600)], "HM": [junk(11_000, 3000), gp[60:62]],
                              "HK1": [ch[:30], junk(15_000, 1000)], "NTc": [gp[100:130], junk(17_000, 50)],
                              "NT1": [gp[130:140]], "HK2": [junk(0, 300)], "H10": [gp[40:60], junk(10_000, 600)]}},
        "squad": {"vers": ("RK",), "rec": {"RK": 1000}, "base": junk(30_000, 1000), "S": None, "cells": {},
                  "forms": {"H5": [junk(32_000, 500)], "NT1": [junk(34_000, 50)]}},
        "metaqa": {"vers": ("SRK",), "rec": {"SRK": 10_000}, "base": np.union1d(kq[:10], junk(40_000, 750)), "S": Smq,
                   "cells": {("s1eval", "GP"): kq},
                   "forms": {"NTc": [junk(50_000, 6000), kq[10:11]], "NT1": [Smq[:50], kq[11:31], junk(60_000, 100)]}},
        "webqsp": {"vers": ("SRK",), "rec": {"SRK": 100}, "base": junk(80_000, 200), "S": np.zeros(0, np.int64),
                   "cells": {("s1eval", "GP"): kq[:40]}, "forms": {"H5": [kq[:40]], "NT1": [kq[:20]]}}}
    D = {}
    for d, sp in spec.items():
        fm = {}
        for f in ORDER:
            parts = sp["forms"].get(f, [])
            kk = np.concatenate(parts) if parts else np.zeros(0, np.int64)
            fm[f] = (kk, -np.arange(kk.size, dtype=np.float32))              # the list as given is its rank order
        b0 = sp["base"]
        Sk = sp["S"] if sp["S"] is not None else np.zeros(0, np.int64)
        cells = {c: (np.arange(sk.size, dtype=np.int64) % (nq if d != "metaqa" and d != "webqsp" else 100), sk,
                     nq if d not in ("metaqa", "webqsp") else 100) for c, sk in sp["cells"].items()}
        allF, A = assemble(fm, lambda x: UL.member(x, b0), Sk, sp["vers"], cells,
                           lambda sk, v: UL.member(sk, b0) | (UL.member(sk, Sk) if v == "SRK" else False))
        b0v = {v: int(b0.size) + (int((~UL.member(Sk, b0)).sum()) if v == "SRK" else 0) for v in sp["vers"]}
        g = {"versions": list(sp["vers"]), "sizes": {"REC": dict(sp["rec"]), "B0": b0v}, "form_pairs": {f: int(fm[f][0].size) for f in ORDER},
             "cells": {}}
        for cv, kind in cells:
            g["cells"].setdefault(cv, {})[kind] = {}
        D[d] = State(d, g, A)
        sp["lists"] = {f: fm[f][0].tolist() for f in ORDER}
        sp["b0v"] = b0v
    carried, steps = chain(D)
    # the direct reading: walk each form's list, stop at the first pair that does not fit some version
    run = {d: {v: set(sp["base"].tolist()) | (set(sp["S"].tolist()) if v == "SRK" else set()) for v in sp["vers"]} for d, sp in spec.items()}
    got = {s_["form"]: s_ for s_ in steps}
    want_carried = []
    for f in ORDER:
        kept, add = {}, {}
        for d, sp in spec.items():
            cap = {v: 130 * sp["rec"][v] // 100 for v in sp["vers"]}
            ad, kp = {v: 0 for v in sp["vers"]}, set()
            for q in sp["lists"][f]:
                cost = {v: int(q not in run[d][v] and q not in kp) for v in sp["vers"]}
                if any(len(run[d][v]) + ad[v] + cost[v] > cap[v] for v in sp["vers"]):
                    break
                for v in sp["vers"]:
                    ad[v] += cost[v]
                kp.add(q)
            kept[d], add[d] = kp, ad
            for v in sp["vers"]:
                assert got[f]["datasets"][d]["size"][v]["pairs_added"] == ad[v], (f, d, v)
        ok_a, below, dear = False, False, False
        for c in got[f]["cells"]:
            d, v = c["dataset"], c["version"]
            sk = spec[d]["cells"][("s1eval", c["kind"])]
            covc = np.array([q in run[d][v] for q in sk.tolist()])
            cf = covc | np.array([q in kept[d] for q in sk.tolist()])
            new = int((cf & ~covc).sum())
            assert c["new"] == new and c["pairs_added"] == add[d][v], (f, c, new)
            ok_a |= c["gain"]["call"] == "ABOVE"
            below |= c["gain"]["call"] == "BELOW"
            dear |= bool(new and round(add[d][v] / new, 1) > P_STAR)
        if ok_a and not below and not dear:
            want_carried.append(f)
            for d, sp in spec.items():
                for v in sp["vers"]:
                    run[d][v] |= kept[d]
    assert carried == want_carried, (carried, want_carried)
    assert carried == ["H5", "HK1", "NT1"], carried
    assert got["H5"]["datasets"]["squad"]["cut"]["cut"] and got["H5"]["datasets"]["squad"]["cut"]["kept_new_pairs"] == 300
    assert got["HM"]["datasets"]["musique"]["cut"]["cut"] and not got["HM"]["conditions"]["a_gain"]     # its slots fall past the cut
    assert got["HK1"]["datasets"]["musique"]["cut"]["kept_new_pairs"] == 540                         # the room left after H5
    assert got["NTc"]["price_over_P_star"][0]["cell"] == "metaqa/SRK/GP" and not got["NTc"]["carried"]
    assert got["NTc"]["datasets"]["musique"]["cut"]["kept_new_pairs"] == 0
    assert got["NT1"]["datasets"]["metaqa"]["size"]["SRK"]["pairs_added"] == 120                     # S's 50 pairs take no room
    assert got["NT1"]["datasets"]["webqsp"]["cut"]["kept_new_pairs"] == 0                            # base over the bound
    assert got["HK2"]["datasets"]["musique"]["cut"]["new_pairs"] == 0 and not got["HK2"]["carried"]
    assert got["H10"]["datasets"]["musique"]["cut"]["new_pairs"] == 0
    assert got["H5"]["beside"]["uncut"]["datasets"]["squad"]["size"]["RK"]["pairs_added"] == 500
    for d, st in D.items():
        for v in st.vers:
            want_u = len(run[d][v])
            assert st.size[v] == want_u, (d, v, st.size[v], want_u)
            assert st.size[v] * BOUND_DEN <= BOUND_NUM * st.rec[v] or st.size[v] == st.base[v], (d, v)
    pk = peak_gb()
    assert "error" not in pk, pk
    assert len({build_sha(NR_FUNCS), build_sha(G_FUNCS)}) == 2
    print(f"zul2c self-test: OK ({pk})")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="zul2c.py")
    ap.add_argument("cmd", choices=["selftest", "ntrank", "grade", "report"])
    ap.add_argument("--ds", choices=SIX)
    ap.add_argument("--host", action="store_true")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    if a.cmd in ("ntrank", "grade") and not a.ds:
        stop("--ds is needed")
    return {"selftest": lambda _a: selftest(), "ntrank": cmd_ntrank, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
