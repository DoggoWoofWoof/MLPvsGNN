"""GP2 (docs/GP2_UNIVERSAL_LINK_SCORE.md): the best universal link-keeping rule, as one exact constrained program over question windows (rank of the tail in
the question's own dense / SPLADE list), text windows (question tokens around the tail's mention in the head's text) and per-hop caps (the step is part of the
key). It reads GP1's link rows (outputs/gp1/<d>/work/*.npz) and adds columns from the package's stored first-stage lists and node texts; no walker, no host
row, no GPU, nothing trains, no model reads anything. zgp1.py and zgp1p.py are imported, never edited. s1eval is never opened.
Outputs only under outputs/gp2/ (aggregated records are committed; the link-level column files are not).

    python outputs/mp_unified/zgp2.py selftest
    python outputs/mp_unified/zgp2.py cols  --part q|r [--ds D ...] [--text]     # dr, sr (and with --text: tw) per link row
    python outputs/mp_unified/zgp2.py diag  --part q|r                           # needed-versus-control tables, stratified AUC (diagnostic)
    python outputs/mp_unified/zgp2.py solve --phase 1|2                          # the quarter: L, U, M, T on every space of the phase
    python outputs/mp_unified/zgp2.py final                                      # forward triples, the final space, the beside runs, the rule record
    python outputs/mp_unified/zgp2.py grade                                      # the rest of dev, once
    python outputs/mp_unified/zgp2.py report
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zgp1 as G1  # noqa: E402  (imports zul2h first: the BLAS threads before numpy loads)
import zgp1p as GP  # noqa: E402
import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402

DECLARED = "docs/GP2_UNIVERSAL_LINK_SCORE.md"
OUT = ROOT / "outputs" / "gp2"
PIN = {"zgp1": "0a9c6bc0026f71849fc4cacee333f9665390ce2bb3616ad9e338d7abe9df2c14", "zgp1p": "062986dcc0b6f5f5020c3094523228976f86839563dcf8e9d3fe851e80730839"}
CELLS = GP.CELLS
RHO, RHOS, CAPS_P, SIZES = GP.RHO, GP.RHOS, GP.CAPS_P, GP.SIZES
WIN = 80                                                   # characters either side of the mention (B1)
RANK_NAMES = (">1000", "301-1000", "101-300", "31-100", "11-30", "4-10", "1-3")
ATTRS = {"dr": 7, "sr": 7, "ur": 7, "pr": 7, "vo": 3, "tw": 4}
NAMES = {"dr": RANK_NAMES, "sr": RANK_NAMES, "ur": RANK_NAMES, "pr": RANK_NAMES, "vo": ("1", "2", "3+"), "tw": ("no mention", "0", "1", "2+")}
POOL_A = ("dr", "sr", "ur", "pr", "vo")
POOL_B = POOL_A + ("tw",)
PHASE1 = [()] + [(a,) for a in POOL_A] + list(itertools.combinations(POOL_A, 2))
PHASE2 = [("tw",)] + [("tw", a) for a in POOL_A]
GRPN = ("S", "R", "K")
G_Q1_ROWS, G_W1_ROWS = 2000, 1000
EMPTY = np.zeros(0, np.int64)
log = GP.log


def stop(msg):
    raise SystemExit(f"[zgp2] STOP: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def sha_of(p):
    return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def gate_pins():
    got = {"zgp1": G1.script_sha256(), "zgp1p": GP.script_sha256()}
    if got != PIN:
        stop(f"G-C3: zgp1 / zgp1p are not the pinned files: {got}")


def sname(attrs):
    return "S0" if not attrs else "+".join(attrs)


# ── bins ─────────────────────────────────────────────────────────────────────


def rank_bin(r):
    """7 bins, 0 worst: absent (0) or beyond 1000, 301-1000, 101-300, 31-100, 11-30, 4-10, 1-3."""
    r = np.asarray(r, np.int64)
    b = np.zeros(r.shape, np.int8)
    for i, hi in enumerate((1000, 300, 100, 30, 10, 3)):
        b = np.where((r >= 1) & (r <= hi), i + 1, b)
    return b.astype(np.int8)


def rank_in(lst, c):
    """Position (1-based, the first occurrence) of each c in the list; 0 where absent."""
    lst = np.asarray(lst, np.int64)
    c = np.asarray(c, np.int64)
    o = np.argsort(lst, kind="stable")
    ls = lst[o]
    i = np.minimum(np.searchsorted(ls, c, side="left"), ls.size - 1)
    hit = ls[i] == c
    return np.where(hit, o[i] + 1, 0).astype(np.int64)


def bin_columns(T, X):
    """The declared attributes' bins (int8) of a cell's rows: dr, sr, ur (X), pr and vo (GP1's rank and votes), tw (X, when present)."""
    dr, sr = X["dr"].astype(np.int64), X["sr"].astype(np.int64)
    ur = np.where((dr > 0) & (sr > 0), np.minimum(dr, sr), np.where(dr > 0, dr, sr))
    B = {"dr": rank_bin(dr), "sr": rank_bin(sr), "ur": rank_bin(ur), "pr": rank_bin(T["rank"]),
         "vo": np.clip(T["votes"].astype(np.int64) - 1, 0, 2).astype(np.int8)}
    if "tw" in X:
        B["tw"] = X["tw"].astype(np.int8)
    return B


# ── the key space ────────────────────────────────────────────────────────────


class Space2:
    """Keys (u, o): u = (group, step) (nine unordered groups; three when the steps are tied) and o = the chosen ordered attributes' bins (0 worst).
    Provides what zgp1p's solvers read: K, grp (0 = S), mono_pairs(). nbs overrides the declared bin counts (self-test only)."""

    def __init__(self, attrs, by_step=True, nbs=None):
        self.attrs, self.by_step = list(attrs), by_step
        self.nbs = list(nbs) if nbs is not None else [ATTRS[a] for a in attrs]
        self.ng = 9 if by_step else 3
        self.nk = int(np.prod(self.nbs)) if self.attrs else 1
        self.K = self.ng * self.nk
        ks = np.arange(self.K)
        self.u = ks // self.nk
        self.grp = self.u // 3 if by_step else self.u
        rest = ks % self.nk
        bins = []
        for nb in reversed(self.nbs):
            bins.append(rest % nb)
            rest = rest // nb
        self.bins = list(reversed(bins))
        stride, s = [], 1
        for nb in reversed(self.nbs):
            stride.append(s)
            s *= nb
        self.stride = list(reversed(stride))

    def key_of(self, grp, step, B):
        g = np.asarray(grp, np.int64)
        u = g * 3 + (np.asarray(step, np.int64) - 1) if self.by_step else g
        o = np.zeros(g.size, np.int64)
        for at, nb in zip(self.attrs, self.nbs):
            o = o * nb + B[at].astype(np.int64)
        return u * self.nk + o

    def mono_pairs(self):
        out = []
        for i, nb in enumerate(self.nbs):
            ks = np.flatnonzero(self.bins[i] + 1 < nb)
            out += list(zip(ks.tolist(), (ks + self.stride[i]).tolist()))
        return out

    def score(self):
        return np.sum(self.bins, axis=0).astype(np.int64) if self.attrs else np.zeros(self.K, np.int64)

    def ulabel(self, u):
        if self.by_step:
            return f"{GRPN[u // 3]}@{u % 3 + 1}"
        return GRPN[u]

    def label(self, k):
        a = "|".join(f"{at}={NAMES[at][int(self.bins[i][k])]}" if at in NAMES else f"{at}={int(self.bins[i][k])}" for i, at in enumerate(self.attrs))
        return self.ulabel(int(self.u[k])) + ("|" + a if a else "")


# ── the exact rank-sum solver (T) ────────────────────────────────────────────


def solve_ranksum(r, p, sp, rho, protect=True):
    """Exact over the rank-sum rules: score = the sum of the row's bin indexes; keep when score >= theta_u, one integer theta per non-S group u (S kept
    entire when protected). Enumerates every threshold vector (two halves of the groups, blocks), the smallest feasible max price, then the fewest keys."""
    J, K = r.shape
    score = sp.score()
    smax = int(score.max()) if K else 0
    th = np.arange(smax + 2)
    gids = [u for u in range(sp.ng) if not (protect and (u // 3 if sp.by_step else u) == 0)]
    base = (sp.grp == 0) if protect else np.zeros(K, bool)
    br, bp, bn = r[:, base].sum(1), p[:, base].sum(1), int(base.sum())
    tabs = []
    for u in gids:
        m = sp.u == u
        keep = [m & (score >= t) for t in th]
        tabs.append((np.array([r[:, k].sum(1) for k in keep]), np.array([p[:, k].sum(1) for k in keep]), np.array([int(k.sum()) for k in keep])))

    def combine(tl):
        R, P, N = np.zeros((1, J)), np.zeros((1, J)), np.zeros(1, np.int64)
        for Rg, Pg, Ng in tl:
            R = (R[:, None, :] + Rg[None, :, :]).reshape(-1, J)
            P = (P[:, None, :] + Pg[None, :, :]).reshape(-1, J)
            N = (N[:, None] + Ng[None, :]).reshape(-1)
        return R, P, N

    h = len(tabs) // 2
    RA, PA, NA = combine(tabs[:h])
    RB, PB, NB = combine(tabs[h:])
    blk = max(1, 4_000_000 // max(1, RB.shape[0] * J))
    best = None
    for s in range(0, RA.shape[0], blk):
        Rt = RA[s:s + blk, None, :] + RB[None, :, :] + br
        ok = (Rt >= rho - 1e-12).all(axis=2)
        if not ok.any():
            continue
        Pt = (PA[s:s + blk, None, :] + PB[None, :, :] + bp).max(axis=2)
        Nt = NA[s:s + blk, None] + NB[None, :] + bn
        val = np.where(ok, Pt, np.inf)
        mv = float(val.min())
        cand = np.argwhere(val <= mv + 1e-12)
        nk = Nt[cand[:, 0], cand[:, 1]]
        j = int(np.argmin(nk))
        ia, ib = int(cand[j, 0]) + s, int(cand[j, 1])
        if best is None or mv < best[0] - 1e-12 or (abs(mv - best[0]) <= 1e-12 and int(nk[j]) < best[1]):
            best = (mv, int(nk[j]), ia, ib)
    if best is None:
        return {"status": "INFEASIBLE", "tau": None, "x": None, "thresholds": None, "kept": None}
    sizes = [th.size] * len(tabs)
    ia = np.unravel_index(best[2], sizes[:h]) if h else ()
    ib = np.unravel_index(best[3], sizes[h:]) if len(tabs) - h else ()
    idx = [int(x) for x in ia] + [int(x) for x in ib]
    x = base.copy()
    thr = {}
    for u, t in zip(gids, idx):
        x = x | ((sp.u == u) & (score >= t))
        thr[sp.ulabel(u)] = int(t)
    return {"status": "OPTIMAL_ON_RANKSUM", "tau": float(best[0]), "x": x.astype(float), "thresholds": thr, "kept": int(x.sum())}


def is_upper_set(sp, x):
    return all(x[k] <= x[k2] + 1e-9 for k, k2 in sp.mono_pairs())


# ── package lists, dev carve ─────────────────────────────────────────────────


def npz_member(path, name):
    """A read-only array of a .npz member: memory-mapped when the member is stored uncompressed, else loaded whole."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path) as z:
            return np.asarray(z[name])
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        n, m = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
        f.seek(info.header_offset + 30 + n + m)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f))
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def dev_lists(d):
    """The dev population's lists: nq, question texts, the dense and SPLADE lists by dev index (full lists), the top fives (the walk's seeds' sources)."""
    if d == "multihoprag":
        import zul2m as M2
        D = G1.P.load_mhr()
        sl = G1.dev_slice(d, D, False)
        root, _nodes, _n, _sha = M2.head()
        queries = M2.M.queries_of(M2.M.mhr_root())
        pos = {str(x["query_id"]): j for j, x in enumerate(queries)}
        s0 = ROOT / "outputs" / "pl1" / "multihoprag" / "RK" / "s0"
        rec = json.loads((s0 / "first.json").read_text(encoding="utf-8"))
        pe = s0 / "first_e3v.npz"
        if G1.E.sha256(pe) != rec["files"]["first_e3v"]["sha256"]:
            stop("G-Q0: S0a's dense list is not the one first.json names")
        spp = root / "multihoprag" / "retrieval_cache" / "splade_top1000.npz"
        if G1.E.sha256(spp) != rec["sources"]["splade_top1000_sha256"]:
            stop("G-Q0: the package's SPLADE list is not the one first.json names")
        dense, spl = npz_member(pe, "ids"), npz_member(spp, "ids")
        rows = np.array([pos[str(i)] for i in sl["ids"]], np.int64)
        return {"nq": len(sl["ids"]), "qtext": sl["qtext"], "dense": lambda qi: np.asarray(dense[rows[qi]], np.int64),
                "spl": lambda qi: np.asarray(spl[rows[qi]], np.int64), "dense5": np.asarray(sl["dense5"], np.int64), "spl5": np.asarray(sl["spl5"], np.int64)}
    root = G1.E.served_root()
    if d == "webqsp":
        with np.load(ROOT / "outputs" / "pl2" / "webqsp" / "pop_dev.npz") as z:
            ids, rows, flat, off = [str(x) for x in z["ids"]], z["idx"].astype(np.int64), z["flat"].astype(np.int64), z["off"]
        golds = [np.unique(flat[off[i]:off[i + 1]]) for i in range(len(ids))]
    else:
        ids, golds, _q = G1.UL.carve_questions(d, G1.C.BESIDE)
        qids = json.loads((root / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
        qmap = {q: j for j, q in enumerate(qids)}
        rows = np.asarray([qmap[q] for q in ids], np.int64)
    keep = [q for q in range(len(golds)) if np.asarray(golds[q]).size]
    rk = rows[keep]
    import zenc
    qt = zenc.query_texts(root, d, rk.tolist())
    rc = root / d / "retrieval_cache"
    dense, spl = npz_member(rc / "dense_top1000.npz", "ids"), npz_member(rc / "splade_top1000.npz", "ids")
    top = G1.H.TOP
    return {"nq": len(keep), "qtext": [str(qt[int(r)]) for r in rk], "dense": lambda qi: np.asarray(dense[rk[qi]], np.int64),
            "spl": lambda qi: np.asarray(spl[rk[qi]], np.int64), "dense5": np.asarray(dense[rk][:, :top], np.int64),
            "spl5": np.asarray(spl[rk][:, :top], np.int64)}


def colfile(d, v, part):
    return OUT / d / f"cols_{part}_{v}.npz"


def qac_sha(T):
    return hashlib.sha256(np.asarray(T["q"], np.int32).tobytes() + np.asarray(T["a"], np.int32).tobytes() + np.asarray(T["c"], np.int32).tobytes()).hexdigest()


def read_cols(d, v, part, T):
    p = colfile(d, v, part)
    if not p.is_file():
        stop(f"{p.name} is missing: run `zgp2.py cols --part {part}`")
    with np.load(p) as z:
        X = {k: z[k] for k in z.files if k in ("dr", "sr", "tw")}
        h = str(z["qac_sha"])
    if h != qac_sha(T):
        stop(f"G-Q3: {p.name} is not aligned with the chunk rows of {d}/{v}/{part}")
    return X


def cmd_cols(a):
    gate_pins()
    part, t0 = a.part, time.time()
    dss = a.ds or [x for x in GP.DATASETS]
    for d in dss:
        cells = [(dd, v) for dd, v in CELLS if dd == d]
        recs = {}
        L = None
        gates = {}
        for _dd, v in cells:
            T = GP.load_cell(d, v, part, cols=("q", "a", "c", "step", "exact"), gold=False)
            if T is None:
                log(f"{d}/{v}/{part}: no extract here yet: skipped")
                continue
            if "not_read" in T:
                log(f"{d}/{v}/{part}: NOT_READ ({T['not_read']})")
                recs[v] = {"status": "NOT_READ", "why": T["not_read"]}
                continue
            if L is None:
                L = dev_lists(d)
                rec0, _s = GP.read_extract(d, part)
                nq_rec = rec0["gates"]["G-G1"]["questions"]
                if L["nq"] != nq_rec:
                    stop(f"G-Q0: {d}: {L['nq']} dev questions rebuilt, the extract record says {nq_rec}")
            q, aa, cc, st = T["q"].astype(np.int64), T["a"].astype(np.int64), T["c"].astype(np.int64), T["step"]
            if st.size and (int(st.min()) < 1 or int(st.max()) > 3):
                stop(f"{d}/{v}: step outside 1..3")
            dr, sr = np.zeros(q.size, np.int32), np.zeros(q.size, np.int32)
            o = np.argsort(q, kind="stable")
            qs = q[o]
            b = np.flatnonzero(np.r_[True, qs[1:] != qs[:-1], True]) if qs.size else np.zeros(1, np.int64)
            bad0 = 0
            for s, e in zip(b[:-1], b[1:]):
                qi = int(qs[s])
                idx = o[s:e]
                ld, ls = L["dense"](qi), L["spl"](qi)
                dr[idx] = rank_in(ld, cc[idx])
                sr[idx] = rank_in(ls, cc[idx])
                seeds = set(L["dense5"][qi].tolist()) | set(L["spl5"][qi].tolist())
                s1 = idx[st[idx] == 1]
                if s1.size and not set(aa[s1].tolist()) <= seeds:
                    bad0 += 1
                if not np.array_equal(ld[:L["dense5"].shape[1]], L["dense5"][qi]) or not np.array_equal(ls[:L["spl5"].shape[1]], L["spl5"][qi]):
                    bad0 += 1
            if bad0:
                stop(f"G-Q0: {d}/{v}: {bad0} questions whose step-1 sources or top fives are not the rebuilt seeds")
            rng = np.random.default_rng(20261126)
            pick = rng.choice(q.size, size=min(G_Q1_ROWS, q.size), replace=False) if q.size else EMPTY
            bad1 = 0
            for i in pick.tolist():
                ld, ls = L["dense"](int(q[i])), L["spl"](int(q[i]))
                nd, ns = np.flatnonzero(ld == cc[i]), np.flatnonzero(ls == cc[i])
                if int(dr[i]) != (int(nd[0]) + 1 if nd.size else 0) or int(sr[i]) != (int(ns[0]) + 1 if ns.size else 0):
                    bad1 += 1
            if bad1:
                stop(f"G-Q1: {d}/{v}: {bad1} of {pick.size} rows differ from a naive scan")
            p = colfile(d, v, part)
            p.parent.mkdir(parents=True, exist_ok=True)
            keep_tw = {}
            if p.is_file():
                with np.load(p) as z:
                    if "tw" in z.files and str(z["qac_sha"]) == qac_sha(T):
                        keep_tw = {"tw": z["tw"]}
            np.savez_compressed(p, dr=dr, sr=sr, qac_sha=np.array(qac_sha(T)), **keep_tw)
            recs[v] = {"rows": int(q.size), "G-Q0": "PASS", "G-Q1": f"PASS ({pick.size} rows)", "in_dense_list": round(float((dr > 0).mean()), 4),
                       "in_splade_list": round(float((sr > 0).mean()), 4), "file": GP.rel(p)}
            log(f"{d}/{v}/{part}: {q.size} rows, G-Q0 and G-Q1 pass ({time.time() - t0:.0f} s)")
        if recs:
            GP.wjson(OUT / d / f"gates_{part}.json", {"dataset": d, "part": part, "declared_in": DECLARED, "cells": recs, "zgp2_sha256": script_sha256(),
                                                       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if a.text:
        cmd_text(a)
    return 0


# ── text windows (B1) ────────────────────────────────────────────────────────


def tw_bin(title, text, qtok, found=None):
    """0 no mention; 1 + min(2, n) with n the question tokens (content tokens, no IDF) in the window of WIN characters either side of the first mention of the
    title, the title's own tokens removed. found = (level, first offset) when already known."""
    lo, _co, fo, _ld = G1.mention_stats(title, text) if found is None else (found[0], 0, found[1], False)
    if not lo:
        return 0
    win = text[max(0, fo - WIN):fo + len(title) + WIN]
    n = len((G1.content_tokens(win) - G1.content_tokens(title)) & qtok)
    return 1 + min(2, n)


def tw_independent(title, text, qtext):
    """G-W1's second implementation for a title that occurs exactly as written: regular-expression word search, a re-tokenised window."""
    pat = re.compile(r"(?<![^\W_])" + re.escape(title) + r"(?![^\W_])")
    m = pat.search(text)
    if not m:
        return None
    win = text[max(0, m.start() - WIN):m.start() + len(title) + WIN].lower()
    toks = {t for t in re.split(r"[^\w]+", win) if t and t not in G1.STOP_WORDS}
    tt = {t for t in re.split(r"[^\w]+", title.lower()) if t and t not in G1.STOP_WORDS}
    qq = {t for t in re.split(r"[^\w]+", qtext.lower()) if t and t not in G1.STOP_WORDS}
    return 1 + min(2, len((toks - tt) & qq))


def cmd_text(a):
    gate_pins()
    part, t0 = a.part, time.time()
    for d in (a.ds or list(GP.DATASETS)):
        cells = [(dd, v) for dd, v in CELLS if dd == d]
        todo = []
        for _dd, v in cells:
            T = GP.load_cell(d, v, part, cols=("q", "a", "c", "step", "exact"), gold=False)
            if T is None or "not_read" in T:
                continue
            todo.append((v, T))
        if not todo:
            continue
        if d == "multihoprag":
            import zul2m as M2
            _root, nodes, _n, _s = M2.head()
            store = G1.NodeStore(records=nodes)
            D = G1.P.load_mhr()
            qtext = G1.dev_slice(d, D, False)["qtext"]
        else:
            store = G1.NodeStore(G1.E.served_root() / d / "nodes.jsonl", None)
            qtext = dev_lists(d)["qtext"]
        qtok_cache = {}
        gates = json.loads((OUT / d / f"gates_{part}.json").read_text(encoding="utf-8")) if (OUT / d / f"gates_{part}.json").is_file() else {"cells": {}}
        for v, T in todo:
            q, aa, cc, ex = T["q"].astype(np.int64), T["a"].astype(np.int64), T["c"].astype(np.int64), T["exact"]
            tw = np.zeros(q.size, np.uint8)
            ms_cache = {}
            for i in np.flatnonzero(ex > 0).tolist():
                qi = int(q[i])
                qt = qtok_cache.get(qi)
                if qt is None:
                    qt = qtok_cache[qi] = G1.content_tokens(qtext[qi])
                key = int(aa[i]) * (1 << 31) + int(cc[i])
                f = ms_cache.get(key)
                title, text = store.get(cc[i])[0], store.get(aa[i])[1]
                if f is None:
                    lo, _co, fo, _ld = G1.mention_stats(title, text)
                    f = ms_cache[key] = (lo, fo)
                if f[0] != int(ex[i]):
                    stop(f"G-W1: {d}/{v}: row {i}: the mention level {f[0]} differs from the stored exact {int(ex[i])}")
                tw[i] = tw_bin(title, text, qt, f)
            rng = np.random.default_rng(20261126)
            ex3 = np.flatnonzero(ex == 3)
            pick = rng.choice(ex3, size=min(G_W1_ROWS, ex3.size), replace=False) if ex3.size else EMPTY
            bad = 0
            for i in pick.tolist():
                ref = tw_independent(store.get(cc[i])[0], store.get(aa[i])[1], qtext[int(q[i])])
                if ref is None or int(tw[i]) != ref:
                    bad += 1
            if bad:
                stop(f"G-W1: {d}/{v}: {bad} of {pick.size} rows differ from the independent implementation")
            p = colfile(d, v, part)
            with np.load(p) as z:
                if str(z["qac_sha"]) != qac_sha(T):
                    stop(f"G-Q3: {p.name}")
                dr, sr = z["dr"], z["sr"]
            np.savez_compressed(p, dr=dr, sr=sr, tw=tw, qac_sha=np.array(qac_sha(T)))
            gates.setdefault("cells", {}).setdefault(v, {}).update({"G-W1": f"PASS ({pick.size} rows)", "tw_bins": np.bincount(tw, minlength=4).tolist(),
                                                                      "mention_rows": int((ex > 0).sum()), "mention_pairs": len(ms_cache)})
            log(f"{d}/{v}/{part}: tw for {int((ex > 0).sum())} mention rows, G-W1 passes ({time.time() - t0:.0f} s)")
        GP.wjson(OUT / d / f"gates_{part}.json", gates)
    return 0


# ── tables ───────────────────────────────────────────────────────────────────

COLS_BASE = ("q", "a", "c", "step", "kind", "w", "grp", "rank", "votes", "exact")


def cell_tables(part, need_tw, gold=False, only=None):
    """Per judged cell (key 'd/v'): the compact table of the program: grp, step, kind, w, q, a, bins of the declared attributes (and the raw T when gold)."""
    out = {}
    for d, v in CELLS:
        key = f"{d}/{v}"
        if only and key not in only:
            continue
        T = GP.load_cell(d, v, part, cols=COLS_BASE, gold=gold)
        if T is None:
            stop(f"{key}: no {part} extract here: the program needs all nine cells")
        if "not_read" in T:
            out[key] = {"not_read": T["not_read"]}
            continue
        X = read_cols(d, v, part, T)
        if need_tw and "tw" not in X:
            stop(f"{key}: tw is not computed ({colfile(d, v, part).name}): run `cols --part {part} --text`")
        B = bin_columns(T, X)
        t = {"grp": T["grp"], "step": T["step"], "kind": T["kind"], "w": T["w"].astype(np.float64), "q": T["q"], "a": T["a"], "B": B, "stats": T["stats"],
             "extract_sha": T["extract_sha"]}
        if gold:
            t["T"] = T
        out[key] = t
    return out


def gate_c4(tabs):
    """G-C4: each cell's needed weight and row counts equal the extract record's / GP1's screen record's (quarter)."""
    res = {}
    sc = json.loads((GP.OUT / "screen_q.json").read_text(encoding="utf-8")) if (GP.OUT / "screen_q.json").is_file() else None
    for key, t in tabs.items():
        if "not_read" in t:
            continue
        nd = t["kind"] == 1
        nw = float(t["w"][nd].sum())
        row = {"needed_units": int(nd.sum()), "needed_weight": round(nw, 4), "rows": int(nd.size)}
        want = t["stats"]["units"]
        if int(nd.sum()) != int(want):
            stop(f"G-C4: {key}: {int(nd.sum())} needed rows, the extract record says {want} units")
        if sc is not None and sc["cells"].get(key, {}).get("status") == "read":
            w0 = float(sc["cells"][key]["needed_weight"])
            row["screen_q_needed_weight"] = w0
            if abs(w0 - nw) > 1e-3 * max(1.0, w0):
                stop(f"G-C4: {key}: needed weight {nw} against GP1's screen {w0}")
        res[key] = row
    return res


# ── solving a space ──────────────────────────────────────────────────────────


def columns_of(sp, tabs):
    cc = {k: (sp.key_of(t["grp"], t["step"], t["B"]), t["w"], t["kind"]) for k, t in tabs.items() if "not_read" not in t}
    return GP.columns(sp, cc)


def solve_space(tabs, attrs, by_step=True, full=False):
    t0 = time.time()
    sp = Space2(attrs, by_step)
    names, r, p, mm, ee = columns_of(sp, tabs)
    L = GP.solve_lp(r, p, sp, RHO, True, mono=False)
    U = GP.solve_milp(r, p, sp, RHO, True, False)
    M = GP.solve_milp(r, p, sp, RHO, True, True)
    T = solve_ranksum(r, p, sp, RHO, True)
    chain = [L["tau"], U["tau"], M["tau"], T["tau"]]
    exact = U["status"] == "OPTIMAL" and M["status"] == "OPTIMAL"
    if exact and None not in chain and not all(chain[i] <= chain[i + 1] + 1e-6 for i in range(3)):
        stop(f"G-C1: {sname(attrs)}: the chain tau_L <= tau_U <= tau_M <= tau_T fails: {chain}")
    if M["x"] is not None:
        if not GP.check_solution(r, p, sp, M["x"], RHO, True, True) or not is_upper_set(sp, M["x"]):
            stop(f"G-C1: {sname(attrs)}: the M solution breaks a constraint")
    blk = {"attrs": list(attrs), "by_step": by_step, "keys": sp.K, "cells": names, "chain_L_U_M_T": chain, "chain_checked": exact,
           "L": {"status": L["status"], "tau": L["tau"], "duals_reach_by_cell": dict(zip(names, L["duals"])) if L.get("duals") else None},
           "U": {"status": U["status"], "tau": U["tau"]},
           "M": {"status": M["status"], "tau": M["tau"], "gap": M["gap"], "bound": M["bound"], "kept": int(M["x"].sum()) if M["x"] is not None else None},
           "T": {"status": T["status"], "tau": T["tau"], "thresholds": T["thresholds"], "kept": T["kept"]},
           "seconds": round(time.time() - t0, 1)}
    if M["x"] is not None:
        x = M["x"]
        blk["M"]["kept_keys"] = np.flatnonzero(x > 0.5).tolist()
        blk["M"]["per_cell_on_Q"] = {c: {"reach": round(float(r[j] @ x), 4), "price": round(float(p[j] @ x), 4)} for j, c in enumerate(names)}
        blk["seen"] = ((r.sum(0) + p.sum(0)) > 0).astype(int).tolist()
    if T["x"] is not None:
        blk["T"]["per_cell_on_Q"] = {c: {"reach": round(float(r[j] @ T["x"]), 4), "price": round(float(p[j] @ T["x"]), 4)} for j, c in enumerate(names)}
    return (blk, (sp, names, r, p, mm, ee, M, L)) if full else blk


def provenance(part, tabs):
    return {"part": part, "declared_in": DECLARED, "zgp2_sha256": script_sha256(), "zgp1_sha256": G1.script_sha256(), "zgp1p_sha256": GP.script_sha256(),
            "extract_sha256": {k: t.get("extract_sha") for k, t in tabs.items()},
            "cols_sha256": {k: hashlib.sha256(colfile(*k.split("/"), part).read_bytes()).hexdigest() for k in tabs if "not_read" not in tabs[k]}}


def cmd_solve(a):
    gate_pins()
    t0 = time.time()
    need_tw = a.phase == 2
    tabs = cell_tables("q", need_tw)
    nr = [k for k, t in tabs.items() if "not_read" in t]
    if nr:
        stop(f"the program needs all nine cells on the quarter; not read: {nr}")
    c4 = gate_c4(tabs)
    rec = provenance("q", tabs)
    rec.update({"phase": a.phase, "G-C4": c4, "rho": RHO, "spaces": {}})
    fam = PHASE1 if a.phase == 1 else PHASE2
    for attrs in fam:
        blk = solve_space(tabs, attrs)
        rec["spaces"][sname(attrs)] = blk
        log(f"phase {a.phase} {sname(attrs)}: tau L {blk['chain_L_U_M_T'][0]:.4f} U {blk['chain_L_U_M_T'][1]:.4f} M {blk['chain_L_U_M_T'][2]:.4f} "
            f"({blk['M']['status']}, {blk['M']['kept']} keys) T {blk['chain_L_U_M_T'][3]:.4f} ({time.time() - t0:.0f} s)")
        GP.wjson(OUT / f"rule_q_phase{a.phase}.json", rec)
    rec["seconds"] = round(time.time() - t0, 1)
    GP.wjson(OUT / f"rule_q_phase{a.phase}.json", rec)
    return 0


def best_of(spaces):
    """The smallest tau_M (M with a solution); ties: fewer kept keys, fewer attributes, declaration order (the dict order)."""
    best = None
    for i, (n, b) in enumerate(spaces.items()):
        if b["M"]["tau"] is None:
            continue
        key = (round(b["M"]["tau"], 9), b["M"]["kept"], len(b["attrs"]), i)
        if best is None or key < best[0]:
            best = (key, n)
    return None if best is None else best[1]


def cmd_final(a):
    gate_pins()
    t0 = time.time()
    p1 = json.loads((OUT / "rule_q_phase1.json").read_text(encoding="utf-8"))
    p2 = json.loads((OUT / "rule_q_phase2.json").read_text(encoding="utf-8")) if (OUT / "rule_q_phase2.json").is_file() else None
    have_tw = p2 is not None
    tabs = cell_tables("q", have_tw)
    c4 = gate_c4(tabs)
    spaces = dict(p1["spaces"])
    if have_tw:
        spaces.update(p2["spaces"])
    pool = POOL_B if have_tw else POOL_A
    pairs = {n: b for n, b in spaces.items() if len(b["attrs"]) == 2}
    bp = best_of(pairs)
    triples = {}
    if bp is not None:
        base = spaces[bp]["attrs"]
        for x in pool:
            if x in base:
                continue
            attrs = tuple(base) + (x,)
            blk = solve_space(tabs, attrs)
            triples[sname(attrs)] = blk
            log(f"triple {sname(attrs)}: tau M {blk['chain_L_U_M_T'][2]:.4f} ({blk['M']['status']}, {blk['M']['kept']} keys) ({time.time() - t0:.0f} s)")
    spaces.update(triples)
    fin = best_of(spaces)
    p1best = best_of(p1["spaces"])
    attrs = tuple(spaces[fin]["attrs"])
    blk, (sp, names, r, p, mm, ee, M, L) = solve_space(tabs, attrs, True, full=True)
    x = M["x"]
    rec = provenance("q", tabs)
    rec.update({"phase": "final", "G-C4": c4, "rho": RHO, "family_tau_M": {n: {"attrs": b["attrs"], "keys": b["keys"], "tau_L": b["L"]["tau"], "tau_U": b["U"]["tau"], "tau_M": b["M"]["tau"],
                                                                                 "M_status": b["M"]["status"], "kept": b["M"]["kept"], "tau_T": b["T"]["tau"]} for n, b in spaces.items()},
                "phase1_sha256": hashlib.sha256((OUT / "rule_q_phase1.json").read_bytes()).hexdigest(),
                "phase2_sha256": hashlib.sha256((OUT / "rule_q_phase2.json").read_bytes()).hexdigest() if have_tw else None,
                "final_space": sname(attrs), "phase1_best": p1best, "forward_triple_base": bp, "tw_computed": have_tw})
    rec["rule"] = {"space": sname(attrs), "attrs": list(attrs), "by_step": True, "keep": x.astype(int).tolist(), "seen": blk["seen"], "M": blk["M"], "T": blk["T"],
                   "L": blk["L"], "U": blk["U"], "kept_keys": [{"key": int(k), "label": sp.label(k), "reach_weight_pooled": round(float(r[:, k].sum()), 4),
                                                                "price_weight_pooled": round(float(p[:, k].sum()), 4)} for k in np.flatnonzero(x > 0.5)]}
    for nm, which in (("rule_phase1_best", p1best), ("rule_S0", "S0")):
        b = spaces[which]
        s2 = Space2(b["attrs"], True)
        xx = np.zeros(s2.K)
        xx[b["M"]["kept_keys"]] = 1.0
        rec[nm] = {"space": which, "attrs": b["attrs"], "by_step": True, "keep": xx.astype(int).tolist(), "seen": b["seen"], "M_tau": b["M"]["tau"]}
    be = {"S_free": {}, "rho": {}, "size": {}, "own": {}, "frontier": {}, "tied_steps": {}, "gp1_reference_tau": 0.99997}
    sf = GP.solve_milp(r, p, sp, RHO, False, True)
    be["S_free"] = {"status": sf["status"], "tau": sf["tau"], "cost_of_A1": (M["tau"] - sf["tau"]) if sf["tau"] is not None else None}
    for rho in RHOS:
        sr_ = GP.solve_milp(r, p, sp, rho, True, True)
        be["rho"][str(rho)] = {"status": sr_["status"], "tau": sr_["tau"]}
    for rho in (0.94, 0.96):
        be["rho"][f"L_{rho}"] = {"tau": GP.solve_lp(r, p, sp, rho, True, mono=False)["tau"]}
    for B in SIZES:
        sz = GP.solve_milp(r, p, sp, RHO, True, True, size_cap=B)
        be["size"][str(B)] = {"status": sz["status"], "tau": sz["tau"]}
    own = {}
    for j, c in enumerate(names):
        oj = GP.solve_milp(r[j:j + 1], p[j:j + 1], sp, RHO, True, True)
        own[c] = {"status": oj["status"], "tau": oj["tau"]}
    be["own"] = own
    ts = [v["tau"] for v in own.values() if v["tau"] is not None]
    be["cost_of_universality"] = (M["tau"] - max(ts)) if ts else None
    for cap in CAPS_P:
        be["frontier"][str(cap)] = GP.solve_frontier(r, p, sp, cap, True, True)
    tied = solve_space(tabs, attrs, by_step=False)
    be["tied_steps"] = {"tau_M": tied["M"]["tau"], "status": tied["M"]["status"], "kept": tied["M"]["kept"], "gain_of_per_hop": (tied["M"]["tau"] - M["tau"]) if tied["M"]["tau"] is not None else None}
    s0t = solve_space(tabs, (), by_step=False)
    be["groups_only_tied_steps_tau_M"] = s0t["M"]["tau"]
    rec["beside"] = be
    # the per-step split of the cell tables (where the weight sits)
    split = {}
    for k, t in tabs.items():
        nd = t["kind"] == 1
        wp = np.where(nd, 1.0, t["w"])
        split[k] = {"needed_weight_by_step": [round(float(t["w"][nd & (t["step"] == s)].sum() / max(1e-12, t["w"][nd].sum())), 4) for s in (1, 2, 3)],
                    "price_weight_by_step": [round(float(wp[t["step"] == s].sum() / wp.sum()), 4) for s in (1, 2, 3)],
                    "price_weight_by_grp_S_R_K": [round(float(wp[t["grp"] == g].sum() / wp.sum()), 4) for g in (0, 1, 2)]}
    rec["split_by_step"] = split
    rec["seconds"] = round(time.time() - t0, 1)
    GP.wjson(OUT / "rule_q.json", rec)
    log(f"final space {sname(attrs)}: tau_M {M['tau']:.4f}; phase 1 best {p1best}; rule record {GP.rel(OUT / 'rule_q.json')} ({time.time() - t0:.0f} s)")
    return 0


# ── diagnostics ──────────────────────────────────────────────────────────────


def cmd_diag(a):
    gate_pins()
    part, t0 = a.part, time.time()
    rec = {"part": part, "declared_in": DECLARED, "zgp2_sha256": script_sha256(), "cells": {}}
    def _has_tw(d, v):
        p = colfile(d, v, part)
        if not p.is_file():
            return False
        with np.load(p) as z:
            return "tw" in z.files
    have_tw = all(_has_tw(d, v) for d, v in CELLS)
    tabs = cell_tables(part, have_tw)
    for key, t in tabs.items():
        if "not_read" in t:
            rec["cells"][key] = {"status": "NOT_READ", "why": t["not_read"]}
            continue
        nd = t["kind"] == 1
        wn, wc = t["w"][nd], t["w"][~nd]
        nqm = int(t["q"].max()) + 1 if t["q"].size else 1
        row = {}
        for at, B in t["B"].items():
            nb = ATTRS[at]
            sn = np.bincount(B[nd], weights=wn, minlength=nb) / max(1e-12, wn.sum())
            sc = np.bincount(B[~nd], weights=wc, minlength=nb) / max(1e-12, wc.sum())
            A, Bq = GP.strata_parts(t["q"].astype(np.int64), t["a"].astype(np.int64), t["step"], t["kind"], t["w"], B, nb, nqm)
            ci = GP.auc_ci(A, Bq)
            row[at] = {"bins": list(NAMES[at]), "needed_share": [round(float(x), 4) for x in sn], "control_share": [round(float(x), 4) for x in sc], "auc": ci}
        rec["cells"][key] = row
        log(f"diag {key} ({time.time() - t0:.0f} s)")
    GP.wjson(OUT / f"diag_{part}.json", rec)
    return 0


# ── grading on the rest of dev ───────────────────────────────────────────────


def cmd_grade(a):
    gate_pins()
    t0 = time.time()
    rule = json.loads((OUT / "rule_q.json").read_text(encoding="utf-8"))
    have_tw = bool(rule.get("tw_computed"))
    res = {"part": "r", "declared_in": DECLARED, "rule_sha256": hashlib.sha256((OUT / "rule_q.json").read_bytes()).hexdigest(), "zgp2_sha256": script_sha256(), "rules": {}}
    cells_out = {w: {} for w in ("rule", "rule_phase1_best", "rule_S0")}
    for d, v in CELLS:
        key = f"{d}/{v}"
        tabs = cell_tables("r", have_tw, gold=True, only=[key])
        t = tabs[key]
        if "not_read" in t:
            for w in cells_out:
                cells_out[w][key] = {"status": "NOT_READ", "why": t["not_read"]}
            continue
        T = t["T"]
        nd = t["kind"] == 1
        if t["kind"].size == 0 or float(t["w"][nd].sum()) <= 0:
            for w in cells_out:
                cells_out[w][key] = {"status": "NO_REACHABLE_GOLD"}
            continue
        for w in cells_out:
            rr = rule[w]
            sp = Space2(rr["attrs"], rr.get("by_step", True))
            x = np.array(rr["keep"], float)
            kk = sp.key_of(t["grp"], t["step"], t["B"])
            keep = x[kk] > 0.5
            den = float(t["w"][nd].sum())
            cand = float(nd.sum() + t["w"][~nd].sum())
            reach = float(t["w"][nd & keep].sum()) / den
            price = float(((nd & keep).sum() + t["w"][(~nd) & keep].sum()) / cand)
            k_, t_, per = GP.path_retention(T, keep)
            seen = np.asarray(rr["seen"])
            g = {"status": "read", "reach": round(reach, 4), "price": round(price, 4), "path_retention": round(k_ / t_, 4) if t_ else float("nan"), "golds_with_units": t_,
                 "path_by_distance": {str(dd): [int(a_), int(b_)] for dd, (a_, b_) in per.items()}, "needed_weight": round(den, 4),
                 "needed_weight_in_keys_unseen_on_Q": round(float(t["w"][nd & (seen[kk] == 0)].sum()) / den, 4)}
            g["SCREEN_IN"] = bool(g["reach"] >= GP.SCREEN_REACH and g["price"] <= GP.SCREEN_PRICE and g["path_retention"] >= GP.SCREEN_PATH)
            cells_out[w][key] = g
            log(f"{w} {key}: reach {g['reach']} price {g['price']} path {g['path_retention']} SCREEN_IN {g['SCREEN_IN']} ({time.time() - t0:.0f} s)")
        del tabs, t, T
    for w, cs in cells_out.items():
        read = [c for c in cs.values() if c["status"] == "read"]
        res["rules"][w] = {"space": rule[w]["space"], "attrs": rule[w]["attrs"], "cells": cs, "cells_read": len(read), "screen_in": int(sum(c["SCREEN_IN"] for c in read)),
                           "call": "UNIVERSAL_RULE" if (read and len(read) == len(CELLS) and all(c["SCREEN_IN"] for c in read)) else "NOT_UNIVERSAL"}
    GP.wjson(OUT / "grade_r.json", res)
    log(f"grade: {res['rules']['rule']['call']} ({res['rules']['rule']['screen_in']}/{res['rules']['rule']['cells_read']} cells SCREEN_IN)")
    return 0


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(a):
    L = []

    def load(n):
        p = OUT / n
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    ru, gr, dq = load("rule_q.json"), load("grade_r.json"), load("diag_q.json")
    for ph in (1, 2):
        pr = load(f"rule_q_phase{ph}.json")
        if pr:
            L.append(f"### Phase {ph} (quarter): tau by space")
            L.append("| space | keys | tau L | tau U | tau M | M status | kept | tau T |")
            L.append("|---|---|---|---|---|---|---|---|")
            for n, b in pr["spaces"].items():
                c = b["chain_L_U_M_T"]
                L.append(f"| {n} | {b['keys']} | {c[0]:.4f} | {c[1]:.4f} | {c[2]:.4f} | {b['M']['status']} | {b['M']['kept']} | {c[3]:.4f} |")
    if ru:
        L.append(f"\n### Final: space {ru['final_space']} (phase 1 best {ru['phase1_best']}, forward-triple base {ru['forward_triple_base']})")
        for n, b in ru["family_tau_M"].items():
            if len(b["attrs"]) == 3:
                L.append(f"- triple {n}: tau M {b['tau_M']:.4f} ({b['M_status']}, {b['kept']} keys)")
        be = ru["beside"]
        L.append(f"- rule: tau M {ru['rule']['M']['tau']:.4f}, T {ru['rule']['T']['tau']:.4f}, kept {ru['rule']['M']['kept']} keys; GP1 reference {be['gp1_reference_tau']}")
        L.append(f"- S free {be['S_free']}; rho {be['rho']}; size {be['size']}; cost of universality {be['cost_of_universality']}")
        L.append(f"- own optima {be['own']}")
        L.append(f"- frontier {be['frontier']}")
        L.append(f"- tied steps {be['tied_steps']}; groups only, tied {be['groups_only_tied_steps_tau_M']}")
        L.append(f"- duals (reach, by cell) {ru['rule']['L'].get('duals_reach_by_cell')}")
        L.append("- kept keys:")
        for k in ru["rule"]["kept_keys"]:
            L.append(f"  - {k['label']} (reach weight {k['reach_weight_pooled']}, price weight {k['price_weight_pooled']})")
        L.append("- per cell on the quarter: " + "; ".join(f"{c}: reach {v['reach']} price {v['price']}" for c, v in ru["rule"]["M"]["per_cell_on_Q"].items()))
        L.append("- split by step: " + "; ".join(f"{c}: needed {v['needed_weight_by_step']} price {v['price_weight_by_step']} S/R/K {v['price_weight_by_grp_S_R_K']}" for c, v in ru["split_by_step"].items()))
    if dq:
        L.append("\n### Diagnostic tables (quarter): needed share / control share per bin, stratified AUC")
        for key, row in dq["cells"].items():
            if row.get("status") == "NOT_READ":
                continue
            L.append(f"\n**{key}**")
            for at, t in row.items():
                auc = t["auc"]
                a_ = "na" if auc is None else f"{auc['auc']:.3f} [{auc['lo']:.3f},{auc['hi']:.3f}]"
                L.append(f"  - {at}: needed " + " / ".join(f"{b} {s:.2f}" for b, s in zip(t["bins"], t["needed_share"])) + "; control " + " / ".join(f"{s:.2f}" for s in t["control_share"]) + f"; AUC {a_}")
    if gr:
        L.append("\n### Grade (rest of dev)")
        for w, g in gr["rules"].items():
            L.append(f"- {w} ({g['space']}): {g['call']} ({g['screen_in']}/{g['cells_read']} cells SCREEN_IN)")
            for k, c in g["cells"].items():
                if c["status"] == "read":
                    L.append(f"  - {k}: reach {c['reach']}, price {c['price']}, path retention {c['path_retention']} ({c['golds_with_units']} golds), unseen-key needed weight {c['needed_weight_in_keys_unseen_on_Q']} -> {c['SCREEN_IN']}")
    txt = "\n".join(L)
    (OUT / "report.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def brute_m(r, p, sp, rho, protect):
    """Reference for M: every subset of the non-S keys that is an upper set, the smallest feasible max price."""
    K = sp.K
    free = [k for k in range(K) if not (protect and sp.grp[k] == 0)]
    pairs = sp.mono_pairs()
    best = None
    for m in range(1 << len(free)):
        x = np.zeros(K)
        if protect:
            x[sp.grp == 0] = 1.0
        for i, k in enumerate(free):
            if (m >> i) & 1:
                x[k] = 1.0
        if not all(x[k] <= x[k2] for k, k2 in pairs):
            continue
        if bool(((r @ x) >= rho - 1e-9).all()):
            val = float((p @ x).max())
            if best is None or val < best - 1e-12:
                best = val
    return best


def brute_t(r, p, sp, rho, protect):
    score = sp.score()
    gids = [u for u in range(sp.ng) if not (protect and (u // 3 if sp.by_step else u) == 0)]
    smax = int(score.max())
    best = None
    for ths in itertools.product(range(smax + 2), repeat=len(gids)):
        x = (sp.grp == 0).astype(float) if protect else np.zeros(sp.K)
        for u, t in zip(gids, ths):
            x = np.maximum(x, ((sp.u == u) & (score >= t)).astype(float))
        if bool(((r @ x) >= rho - 1e-9).all()):
            val = float((p @ x).max())
            if best is None or val < best - 1e-12:
                best = val
    return best


def selftest():
    import tempfile
    rng = np.random.default_rng(7)
    # rank bins and lookups
    r = np.array([0, 1, 3, 4, 10, 11, 30, 31, 100, 101, 300, 301, 1000, 1001])
    assert rank_bin(r).tolist() == [0, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0], rank_bin(r).tolist()
    for _ in range(50):
        lst = rng.permutation(60)[:40]
        c = rng.integers(0, 60, 200)
        got = rank_in(lst, c)
        want = np.array([(int(np.flatnonzero(lst == x)[0]) + 1) if (lst == x).any() else 0 for x in c])
        assert np.array_equal(got, want)
    dup = np.array([5, 7, 5, 9])
    assert rank_in(dup, np.array([5, 9, 1])).tolist() == [1, 4, 0]
    # npz members: stored (mmap) and compressed
    with tempfile.TemporaryDirectory() as td:
        a = rng.integers(0, 1000, (50, 20)).astype(np.int32)
        np.savez(Path(td) / "s.npz", ids=a, scores=a.astype(np.float16))
        np.savez_compressed(Path(td) / "c.npz", ids=a)
        m1, m2 = npz_member(Path(td) / "s.npz", "ids"), npz_member(Path(td) / "c.npz", "ids")
        assert isinstance(m1, np.memmap) and np.array_equal(m1, a) and np.array_equal(m2, a) and m1.dtype == np.int32
        assert np.array_equal(npz_member(Path(td) / "s.npz", "scores"), a.astype(np.float16))
        del m1
    # space layout
    sp = Space2(["dr", "vo"], True)
    assert sp.K == 9 * 21 and len(sp.mono_pairs()) == 9 * (6 * 3 + 7 * 2)
    B = {"dr": np.array([0, 6, 3], np.int8), "vo": np.array([2, 0, 1], np.int8)}
    k = sp.key_of(np.array([0, 1, 2]), np.array([1, 2, 3]), B)
    assert [sp.label(int(x)) for x in k] == ["S@1|dr=>1000|vo=3+", "R@2|dr=1-3|vo=1", "K@3|dr=31-100|vo=2"], [sp.label(int(x)) for x in k]
    sp3 = Space2([], False)
    assert sp3.K == 3 and sp3.mono_pairs() == [] and sp3.key_of(np.array([2]), np.array([3]), {}).tolist() == [2]
    # the exact solvers against brute force on toy programs
    for trial in range(40):
        by_step = bool(trial % 2)
        nbs = [2, 2] if (trial % 4) < 2 else [3]
        sp = Space2(["a", "b"][:len(nbs)], by_step, nbs=nbs)
        if sp.K - int((sp.grp == 0).sum()) > 14:
            sp = Space2(["a"], False, nbs=[3])
        J = 3
        r = rng.random((J, sp.K)) ** 2
        p = rng.random((J, sp.K)) ** 2
        r, p = r / r.sum(1, keepdims=True), p / p.sum(1, keepdims=True)
        rho = float(rng.choice([0.5, 0.7, 0.9]))
        bm, bt = brute_m(r, p, sp, rho, True), brute_t(r, p, sp, rho, True)
        M = GP.solve_milp(r, p, sp, rho, True, True)
        T = solve_ranksum(r, p, sp, rho, True)
        U = GP.solve_milp(r, p, sp, rho, True, False)
        Lp = GP.solve_lp(r, p, sp, rho, True, mono=False)
        assert (bm is None) == (M["tau"] is None), (trial, bm, M["tau"])
        assert (bt is None) == (T["tau"] is None), (trial, bt, T["tau"])
        if bm is not None:
            assert abs(M["tau"] - bm) < 1e-6, (trial, M["tau"], bm)
            assert is_upper_set(sp, M["x"])
        if bt is not None:
            assert abs(T["tau"] - bt) < 1e-9, (trial, T["tau"], bt)
            assert bm is not None and M["tau"] <= T["tau"] + 1e-6
        if bm is not None and U["tau"] is not None and Lp["tau"] is not None:
            assert Lp["tau"] <= U["tau"] + 1e-6 <= M["tau"] + 2e-6
    # windows: both implementations agree on strings
    for title, text, q in [("Paris", "He lived in Paris, the capital of France, and wrote about wine there.", "which wine did he write about in france"),
                           ("Ann Lee", "x " * 100 + "Ann Lee won the prize for physics in 1999. " + "y " * 100, "who won the physics prize"),
                           ("Rome", "Rome is old.", "what is old")]:
        qtok = G1.content_tokens(q)
        assert tw_bin(title, text, qtok) == tw_independent(title, text, q), (title, tw_bin(title, text, qtok), tw_independent(title, text, q))
    assert tw_bin("Zed", "nothing here", {"nothing"}) == 0
    print("zgp2 selftest: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    c = sub.add_parser("cols")
    c.add_argument("--part", required=True, choices=["q", "r"])
    c.add_argument("--ds", nargs="*")
    c.add_argument("--text", action="store_true")
    t = sub.add_parser("text")
    t.add_argument("--part", required=True, choices=["q", "r"])
    t.add_argument("--ds", nargs="*")
    dg = sub.add_parser("diag")
    dg.add_argument("--part", required=True, choices=["q", "r"])
    s = sub.add_parser("solve")
    s.add_argument("--phase", type=int, required=True, choices=[1, 2])
    sub.add_parser("final")
    sub.add_parser("grade")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": lambda: selftest(), "cols": lambda: cmd_cols(a), "text": lambda: cmd_text(a), "diag": lambda: cmd_diag(a), "solve": lambda: cmd_solve(a),
            "final": lambda: cmd_final(a), "grade": lambda: cmd_grade(a), "report": lambda: cmd_report(a)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
