"""GP2 amendment A3 (docs/GP2_UNIVERSAL_LINK_SCORE.md section 14): the text-window ladder (tw40, tw160, tw320), the co-mention distance (cd), the question-tail score windows
(qn, qs), the per-question caps C2, the node bound, and the rest-of-dev grade of the declared set of rules. It imports zgp2.py (unchanged, its sha is in every record) and
through it zgp1.py / zgp1p.py, read-only. No walker, no host row, no GPU, nothing trains, s1eval is never opened. Outputs only under outputs/gp2/.

    python outputs/mp_unified/zgp2a3.py selftest
    python outputs/mp_unified/zgp2a3.py text2 --part q|r [--ds D ...]      # tw40, tw160, tw320, cd per link row (G-W2)
    python outputs/mp_unified/zgp2a3.py score --part q|r [--ds D ...]      # qn, qs per link row (G-Q2)
    python outputs/mp_unified/zgp2a3.py solve5                             # the quarter: phase 5 (30 label spaces), L U M
    python outputs/mp_unified/zgp2a3.py cap                                # the quarter: C2, six scores, 286 triples each
    python outputs/mp_unified/zgp2a3.py a3                                 # the final rule, the rule set, the node bound on the quarter
    python outputs/mp_unified/zgp2a3.py grade                              # the rest of dev, once, the declared set (item 7)
    python outputs/mp_unified/zgp2a3.py report
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zgp2 as Z  # noqa: E402

G1, GP = Z.G1, Z.GP
OUT, ROOT, DECLARED = Z.OUT, Z.ROOT, Z.DECLARED
log, stop = Z.log, Z.stop

WIDTHS = {"tw40": 40, "tw160": 160, "tw320": 320}
NEW = ("tw40", "tw160", "tw320", "cd", "qn", "qs")
CD_NAMES = ("no mention", "no question token", ">320", "161-320", "81-160", "41-80", "<=40")
ZB_NAMES = ("not in list", "z<0.1", "0.1-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "z>=0.8")
ZB_EDGES = (0.1, 0.2, 0.4, 0.6, 0.8)
Z.ATTRS.update({"tw40": 4, "tw160": 4, "tw320": 4, "cd": 7, "qn": 7, "qs": 7})
Z.NAMES.update({"tw40": Z.NAMES["tw"], "tw160": Z.NAMES["tw"], "tw320": Z.NAMES["tw"], "cd": CD_NAMES, "qn": ZB_NAMES, "qs": ZB_NAMES})
OLD_POOL = ("dr", "sr", "ur", "tw", "qov", "dir")
PHASE5 = [t for x in NEW for t in ((x,), (x, "pr"), (x, "vo"), (x, "pr", "vo"))] + [("pr", "vo", x) for x in OLD_POOL]
GRID = (10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000, 300000, float("inf"))
SCORES = {"K1": ("ur", "dot"), "K2": ("-rank", "dot"), "K3": ("ur", "-rank", "votes", "dot"), "K4": ("dot",), "K5": ("tw", "ur", "-rank", "dot"), "K6": ("cd", "ur", "-rank", "dot")}
NODE_TARGET = 2500
COLS_A3 = Z.COLS_BASE + ("dot",)
EXTRA = ("dr", "sr", "tw") + NEW


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def prov(part, tabs):
    d = Z.provenance(part, tabs)
    d["zgp2a3_sha256"] = script_sha256()
    return d


# ── the new columns ──────────────────────────────────────────────────────────


def cd_bin(dist):
    """None (no question token in the head's text) -> 1; > 320 -> 2; 161-320 -> 3; 81-160 -> 4; 41-80 -> 5; <= 40 -> 6."""
    if dist is None:
        return 1
    return 2 if dist > 320 else 3 if dist > 160 else 4 if dist > 80 else 5 if dist > 40 else 6


def head_index(text):
    """Lower-cased \\w+ tokens of the head's text -> array of (start, end) in the original string."""
    d = {}
    for m in re.finditer(r"\w+", text):
        d.setdefault(m.group().lower(), []).append((m.start(), m.end()))
    return {k: np.asarray(v, np.int64) for k, v in d.items()}


def cd_of(hidx, qset, fo, tlen):
    """Characters from the span [fo, fo + tlen) to the nearest occurrence of a token of qset in the head's text; None when no token of qset occurs."""
    best, end_m = None, fo + tlen
    for tok in qset:
        arr = hidx.get(tok)
        if arr is None:
            continue
        s, e = arr[:, 0], arr[:, 1]
        m = int(np.where(s >= end_m, s - end_m, np.where(e <= fo, fo - e, 0)).min())
        if best is None or m < best:
            best = m
    return best


def tw_w(title, text, qtok, fo, w):
    win = text[max(0, fo - w):fo + len(title) + w]
    return 1 + min(2, len((G1.content_tokens(win) - G1.content_tokens(title)) & qtok))


def tw_independent_w(title, text, qtext, w):
    pat = re.compile(r"(?<![^\W_])" + re.escape(title) + r"(?![^\W_])")
    m = pat.search(text)
    if not m:
        return None
    win = text[max(0, m.start() - w):m.start() + len(title) + w].lower()
    toks = {t for t in re.split(r"[^\w]+", win) if t and t not in G1.STOP_WORDS}
    tt = {t for t in re.split(r"[^\w]+", title.lower()) if t and t not in G1.STOP_WORDS}
    qq = {t for t in re.split(r"[^\w]+", qtext.lower()) if t and t not in G1.STOP_WORDS}
    return 1 + min(2, len((toks - tt) & qq))


def cd_independent(title, text, qtext):
    """G-W2's second implementation of cd for a title that occurs as written: a boundary search per question token."""
    pat = re.compile(r"(?<![^\W_])" + re.escape(title) + r"(?![^\W_])")
    m = pat.search(text)
    if not m:
        return None
    fo, end_m = m.start(), m.start() + len(title)
    tt = {t for t in re.split(r"[^\w]+", title.lower()) if t and t not in G1.STOP_WORDS}
    qq = {t for t in re.split(r"[^\w]+", qtext.lower()) if t and t not in G1.STOP_WORDS} - tt
    best = None
    for tok in qq:
        for mm in re.finditer(r"(?<!\w)" + re.escape(tok) + r"(?!\w)", text, re.I):
            s, e = mm.start(), mm.end()
            d = s - end_m if s >= end_m else fo - e if e <= fo else 0
            if best is None or d < best:
                best = d
    return cd_bin(best)


def save_extra(p, T, **arrs):
    """Rewrite a cols_*.npz with extra arrays, keeping what is there; the (q, a, c) hash must match."""
    with np.load(p) as z:
        if str(z["qac_sha"]) != Z.qac_sha(T):
            stop(f"G-Q3: {p.name}")
        cur = {k: z[k] for k in z.files if k != "qac_sha"}
    cur.update(arrs)
    np.savez_compressed(p, qac_sha=np.array(Z.qac_sha(T)), **cur)


def run_each(a, cmd):
    dss = a.ds or list(GP.DATASETS)
    if len(dss) <= 1:
        return False
    import subprocess
    for d in dss:
        rc = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), cmd, "--part", a.part, "--ds", d]).returncode
        if rc:
            stop(f"{cmd} {d} exited with {rc}")
    return True


def cmd_text2(a):
    Z.gate_pins()
    if run_each(a, "text2"):
        return 0
    part, t0 = a.part, time.time()
    for d in (a.ds or list(GP.DATASETS)):
        todo = []
        for _dd, v in [(dd, vv) for dd, vv in Z.CELLS if dd == d]:
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
            qtext = Z.dev_lists(d)["qtext"]
        qtok_cache = {}
        gates = json.loads((OUT / d / f"gates_{part}.json").read_text(encoding="utf-8")) if (OUT / d / f"gates_{part}.json").is_file() else {"cells": {}}
        for v, T in todo:
            q, aa, cc, ex = T["q"].astype(np.int64), T["a"].astype(np.int64), T["c"].astype(np.int64), T["exact"]
            out = {k: np.zeros(q.size, np.uint8) for k in ("tw40", "tw160", "tw320", "cd")}
            rows = np.flatnonzero(ex > 0)
            rows = rows[np.argsort(aa[rows], kind="stable")]
            ms_cache, last_a, hidx = {}, None, None
            for i in rows.tolist():
                qi = int(q[i])
                qt = qtok_cache.get(qi)
                if qt is None:
                    qt = qtok_cache[qi] = G1.content_tokens(qtext[qi])
                title, text = store.get(cc[i])[0], store.get(aa[i])[1]
                key = int(aa[i]) * (1 << 31) + int(cc[i])
                f = ms_cache.get(key)
                if f is None:
                    lo, _co, fo, _ld = G1.mention_stats(title, text)
                    f = ms_cache[key] = (lo, fo)
                if f[0] != int(ex[i]):
                    stop(f"G-W2: {d}/{v}: row {i}: the mention level {f[0]} differs from the stored exact {int(ex[i])}")
                for nm, w in WIDTHS.items():
                    out[nm][i] = tw_w(title, text, qt, f[1], w)
                if last_a != int(aa[i]):
                    last_a, hidx = int(aa[i]), head_index(text)
                qset = qt - G1.content_tokens(title)
                out["cd"][i] = cd_bin(cd_of(hidx, qset, f[1], len(title)))
            rng = np.random.default_rng(20261126)
            ex3 = np.flatnonzero(ex == 3)
            pick = rng.choice(ex3, size=min(Z.G_W1_ROWS, ex3.size), replace=False) if ex3.size else Z.EMPTY
            bad = 0
            for i in pick.tolist():
                title, text, qx = store.get(cc[i])[0], store.get(aa[i])[1], qtext[int(q[i])]
                for nm, w in WIDTHS.items():
                    if tw_independent_w(title, text, qx, w) != int(out[nm][i]):
                        bad += 1
                if cd_independent(title, text, qx) != int(out["cd"][i]):
                    bad += 1
            if bad:
                stop(f"G-W2: {d}/{v}: {bad} disagreements with the independent implementations on {pick.size} rows")
            save_extra(Z.colfile(d, v, part), T, **out)
            gates.setdefault("cells", {}).setdefault(v, {}).update({"G-W2": f"PASS ({pick.size} rows)", "cd_bins": np.bincount(out["cd"], minlength=7).tolist(),
                                                                      "tw160_bins": np.bincount(out["tw160"], minlength=4).tolist()})
            log(f"{d}/{v}/{part}: ladder and cd for {rows.size} mention rows, G-W2 passes ({time.time() - t0:.0f} s)")
        GP.wjson(OUT / d / f"gates_{part}.json", gates)
    return 0


def dev_scores(d):
    """The stored scores of the dev population's dense and SPLADE lists by dev index, rows mapped exactly as zgp2.dev_lists maps the ids."""
    if d == "multihoprag":
        import zul2m as M2
        D = G1.P.load_mhr()
        sl = G1.dev_slice(d, D, False)
        root, _nodes, _n, _sha = M2.head()
        queries = M2.M.queries_of(M2.M.mhr_root())
        pos = {str(x["query_id"]): j for j, x in enumerate(queries)}
        s0 = ROOT / "outputs" / "pl1" / "multihoprag" / "RK" / "s0"
        pe = s0 / "first_e3v.npz"
        spp = root / "multihoprag" / "retrieval_cache" / "splade_top1000.npz"
        dense, spl = Z.npz_member(pe, "scores"), Z.npz_member(spp, "scores")
        rows = np.array([pos[str(i)] for i in sl["ids"]], np.int64)
        return {"dense": lambda qi: np.asarray(dense[rows[qi]], np.float32), "spl": lambda qi: np.asarray(spl[rows[qi]], np.float32)}
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
    rc = root / d / "retrieval_cache"
    dense, spl = Z.npz_member(rc / "dense_top1000.npz", "scores"), Z.npz_member(rc / "splade_top1000.npz", "scores")
    return {"dense": lambda qi: np.asarray(dense[rk[qi]], np.float32), "spl": lambda qi: np.asarray(spl[rk[qi]], np.float32)}


def zbin(s, first, last):
    """z = (s - last) / (first - last) (1 when first == last), then 1 + the number of edges at or below z (1..6)."""
    z = 1.0 if first == last else (float(s) - float(last)) / (float(first) - float(last))
    return 1 + sum(1 for e in ZB_EDGES if z >= e)


def score_bins(rank, scores):
    """Bins (int8, 0 = not in the list) of the rows whose 1-based position in one question's list is `rank` (0 = absent); scores = that list's stored scores."""
    out = np.zeros(rank.size, np.int8)
    if not scores.size:
        return out
    first, last = float(scores[0]), float(scores[-1])
    for j in np.flatnonzero(rank > 0).tolist():
        out[j] = zbin(scores[int(rank[j]) - 1], first, last)
    return out


def cmd_score(a):
    Z.gate_pins()
    if run_each(a, "score"):
        return 0
    part, t0 = a.part, time.time()
    for d in (a.ds or list(GP.DATASETS)):
        S = None
        gates = json.loads((OUT / d / f"gates_{part}.json").read_text(encoding="utf-8")) if (OUT / d / f"gates_{part}.json").is_file() else {"cells": {}}
        for _dd, v in [(dd, vv) for dd, vv in Z.CELLS if dd == d]:
            T = GP.load_cell(d, v, part, cols=("q", "a", "c", "step", "exact"), gold=False)
            if T is None or "not_read" in T:
                continue
            p = Z.colfile(d, v, part)
            with np.load(p) as zf:
                if str(zf["qac_sha"]) != Z.qac_sha(T):
                    stop(f"G-Q3: {p.name}")
                dr, sr = zf["dr"].astype(np.int64), zf["sr"].astype(np.int64)
            if S is None:
                S = dev_scores(d)
                L = Z.dev_lists(d)
            q, cc = T["q"].astype(np.int64), T["c"].astype(np.int64)
            qn, qs = np.zeros(q.size, np.int8), np.zeros(q.size, np.int8)
            o = np.argsort(q, kind="stable")
            qsrt = q[o]
            b = np.flatnonzero(np.r_[True, qsrt[1:] != qsrt[:-1], True]) if qsrt.size else np.zeros(1, np.int64)
            for s_, e_ in zip(b[:-1], b[1:]):
                qi, idx = int(qsrt[s_]), o[s_:e_]
                qn[idx] = score_bins(dr[idx], S["dense"](qi))
                qs[idx] = score_bins(sr[idx], S["spl"](qi))
            rng = np.random.default_rng(20261126)
            pick = rng.choice(q.size, size=min(Z.G_Q1_ROWS, q.size), replace=False) if q.size else Z.EMPTY
            bad = 0
            for i in pick.tolist():
                for ids_, sc_, got in ((L["dense"](int(q[i])), S["dense"](int(q[i])), qn[i]), (L["spl"](int(q[i])), S["spl"](int(q[i])), qs[i])):
                    nz = np.flatnonzero(ids_ == cc[i])
                    want = 0 if not nz.size else zbin(sc_[int(nz[0])], sc_[0], sc_[-1])
                    if int(got) != want:
                        bad += 1
            if bad:
                stop(f"G-Q2: {d}/{v}: {bad} disagreements with a naive lookup on {pick.size} rows")
            save_extra(p, T, qn=qn, qs=qs)
            gates.setdefault("cells", {}).setdefault(v, {}).update({"G-Q2": f"PASS ({pick.size} rows)", "qn_bins": np.bincount(qn, minlength=7).tolist(), "qs_bins": np.bincount(qs, minlength=7).tolist()})
            log(f"{d}/{v}/{part}: qn and qs for {q.size} rows, G-Q2 passes ({time.time() - t0:.0f} s)")
        GP.wjson(OUT / d / f"gates_{part}.json", gates)
    return 0


# ── the tables of the program ────────────────────────────────────────────────


def tables(part, only=None, gold=False):
    """zgp2.cell_tables with the new columns: B holds every declared bin, plus rank, votes and dot for C2."""
    out = {}
    for d, v in Z.CELLS:
        key = f"{d}/{v}"
        if only and key not in only:
            continue
        T = GP.load_cell(d, v, part, cols=COLS_A3, gold=gold)
        if T is None:
            stop(f"{key}: no {part} extract here: the program needs all nine cells")
        if "not_read" in T:
            out[key] = {"not_read": T["not_read"]}
            continue
        p = Z.colfile(d, v, part)
        if not p.is_file():
            stop(f"{p.name} is missing")
        with np.load(p) as zf:
            if str(zf["qac_sha"]) != Z.qac_sha(T):
                stop(f"G-Q3: {p.name} is not aligned with the chunk rows of {key}")
            miss = [k for k in EXTRA if k not in zf.files]
            if miss:
                stop(f"{key}: {p.name} lacks {miss}: run `cols --text`, `text2`, `score`")
            X = {k: zf[k] for k in EXTRA}
        B = Z.bin_columns(T, X)
        for k in NEW:
            B[k] = X[k].astype(np.int8)
        lab = Z.lab_of(T["prof"], v)
        gmap = np.where(lab == 0, 0, np.where(lab <= 3, 1, 2))
        if not np.array_equal(gmap, T["grp"].astype(gmap.dtype)):
            stop(f"G-L1: {key}")
        t = {"grp": T["grp"], "lab": lab, "step": T["step"], "kind": T["kind"], "w": T["w"].astype(np.float64), "q": T["q"], "a": T["a"], "B": B, "stats": T["stats"],
             "extract_sha": T["extract_sha"], "rank": T["rank"].astype(np.float64), "votes": T["votes"].astype(np.float64), "dot": T["dot"].astype(np.float64)}
        if gold:
            t["T"] = T
        out[key] = t
    return out


# ── phase 5 ──────────────────────────────────────────────────────────────────


def cmd_solve5(a):
    Z.gate_pins()
    t0 = time.time()
    tabs = tables("q")
    nr = [k for k, t in tabs.items() if "not_read" in t]
    if nr:
        stop(f"the program needs all nine cells on the quarter; not read: {nr}")
    c4 = Z.gate_c4(tabs)
    rec = prov("q", tabs)
    rec.update({"phase": 5, "dim": "lab", "G-C4": c4, "rho": Z.RHO, "spaces": {}})
    f4 = lambda x: "  n/a " if x is None else f"{x:.4f}"  # noqa: E731
    for attrs in PHASE5:
        blk = Z.solve_space(tabs, attrs, dim="lab")
        rec["spaces"][Z.sname(attrs)] = blk
        c = blk["chain_L_U_M_T"]
        log(f"phase 5 lab {Z.sname(attrs)}: tau L {f4(c[0])} U {f4(c[1])} M {f4(c[2])} ({blk['M']['status']}, {blk['M']['kept']} keys) ({time.time() - t0:.0f} s)")
        GP.wjson(OUT / "rule_q_phase5.json", rec)
    rec["seconds"] = round(time.time() - t0, 1)
    GP.wjson(OUT / "rule_q_phase5.json", rec)
    return 0


# ── C2: per-question caps on the link-ranked fan-out ─────────────────────────


def score_keys(t, name):
    out = []
    for k in SCORES[name]:
        if k == "dot":
            out.append(t["dot"])
        elif k == "-rank":
            out.append(-t["rank"])
        elif k == "votes":
            out.append(t["votes"])
        else:
            out.append(t["B"][k].astype(np.float64))
    return out


def cap_structure(t, keys):
    """Per (step, question) the non-S rows sorted best-first by `keys` (larger better), cut into tie groups: each group's candidate count T (needed 1, control its weight), the needed
    weight in it, and the count of candidates of the same (step, question) strictly ahead of it."""
    nd = t["kind"] == 1
    cnt = np.where(nd, 1.0, t["w"])
    idx = np.flatnonzero(t["grp"] != 0)
    q, st = t["q"][idx].astype(np.int64), t["step"][idx].astype(np.int64)
    ks = [np.asarray(k, np.float64)[idx] for k in keys]
    order = np.lexsort(tuple(-k for k in reversed(ks)) + (q, st))
    idx, q, st, ks = idx[order], q[order], st[order], [k[order] for k in ks]
    n = idx.size
    st_info = {"idx": idx, "nd": nd, "cnt": cnt, "w": t["w"], "grp": t["grp"], "step": t["step"], "q": t["q"]}
    if n == 0:
        st_info.update({"gid": np.zeros(0, np.int64), "T": np.zeros(0), "needw": np.zeros(0), "ahead": np.zeros(0), "gstep": np.zeros(0, np.int64)})
        return st_info
    newq = np.ones(n, bool)
    newq[1:] = (q[1:] != q[:-1]) | (st[1:] != st[:-1])
    tie = newq.copy()
    for k in ks:
        tie[1:] |= k[1:] != k[:-1]
    gid = np.cumsum(tie) - 1
    G = int(gid[-1]) + 1
    first = np.flatnonzero(tie)
    c = cnt[idx]
    T = np.bincount(gid, weights=c, minlength=G)
    needw = np.bincount(gid, weights=np.where(nd[idx], t["w"][idx], 0.0), minlength=G)
    cumT = np.cumsum(T) - T
    base = np.maximum.accumulate(np.where(newq[first], cumT, -1.0))
    st_info.update({"gid": gid, "T": T, "needw": needw, "ahead": cumT - base, "gstep": st[first]})
    return st_info


def frac(S_, W):
    """Fraction of each tie group kept under a cap of W candidates per (step, question): the expectation over a random order inside the tie."""
    if not np.isfinite(W):
        return np.ones(S_["T"].size)
    T = np.maximum(S_["T"], 1e-300)
    return np.clip((W - S_["ahead"]) / T, 0.0, 1.0)


def cell_norms(t):
    nd = t["kind"] == 1
    cnt = np.where(nd, 1.0, t["w"])
    return float(t["w"][nd].sum()), float(cnt.sum())


def cap_curves(t, S_, grid=GRID):
    """reach[h][g], price[h][g] of the cell for a cap grid value g at hop h+1 (S protected: kept whole at every hop)."""
    den, cand = cell_norms(t)
    nd, cnt = S_["nd"], S_["cnt"]
    reach, price = np.zeros((3, len(grid))), np.zeros((3, len(grid)))
    for h in (1, 2, 3):
        s = (S_["grp"] == 0) & (S_["step"] == h)
        r0, p0 = float(S_["w"][s & nd].sum()) / den, float(cnt[s].sum()) / cand
        m = S_["gstep"] == h
        T, nw, ah = S_["T"][m], S_["needw"][m], S_["ahead"][m]
        for gi, W in enumerate(grid):
            f = np.ones(T.size) if not np.isfinite(W) else np.clip((W - ah) / np.maximum(T, 1e-300), 0.0, 1.0)
            reach[h - 1, gi] = r0 + float((f * nw).sum()) / den
            price[h - 1, gi] = p0 + float((f * T).sum()) / cand
    return reach, price


def cap_keep(t, S_, Wt):
    """Per-row kept fraction (1 for S, the group's fraction for the rest) under per-hop caps Wt = (W1, W2, W3)."""
    f = np.ones(t["kind"].size)
    if S_["idx"].size:
        fg = np.empty(S_["T"].size)
        for h in (1, 2, 3):
            m = S_["gstep"] == h
            W = Wt[h - 1]
            fg[m] = 1.0 if not np.isfinite(W) else np.clip((W - S_["ahead"][m]) / np.maximum(S_["T"][m], 1e-300), 0.0, 1.0)
        f[S_["idx"]] = fg[S_["gid"]]
    return f


def c2_program(curves, rho=Z.RHO):
    """Over the 286 non-increasing triples (W1 >= W2 >= W3): the smallest worst-cell price with every cell's reach >= rho. curves[cell] = (reach, price); returns the best or None."""
    G, cells = len(GRID), list(curves)
    best = None
    for g1 in range(G):
        for g2 in range(g1 + 1):
            for g3 in range(g2 + 1):
                rs = [sum(curves[c][0][h, g] for h, g in enumerate((g1, g2, g3))) for c in cells]
                if min(rs) < rho - 1e-9:
                    continue
                ps = [sum(curves[c][1][h, g] for h, g in enumerate((g1, g2, g3))) for c in cells]
                key = (round(max(ps), 9), g1 + g2 + g3, g1, g2, g3)
                if best is None or key < best[0]:
                    best = (key, (g1, g2, g3), dict(zip(cells, zip(rs, ps))))
    if best is None:
        return None
    _k, gt, per = best
    return {"W": [GRID[g] if np.isfinite(GRID[g]) else "inf" for g in gt], "tau": best[0][0], "per_cell": {c: {"reach": round(r, 4), "price": round(p, 4)} for c, (r, p) in per.items()}}


def required_caps(curves, rho=Z.RHO):
    """Per cell alone: the smallest uniform cap (the same value at all three hops) whose reach is >= rho; None when even no cap does not."""
    out = {}
    for c, (r, p) in curves.items():
        out[c] = None
        for g in range(len(GRID)):
            if float(r[:, g].sum()) >= rho - 1e-9:
                out[c] = GRID[g] if np.isfinite(GRID[g]) else "inf"
                break
    return out


def node_bound(t, price, S_prot_per_q=None):
    """Per cell: questions, candidate links per question, kept links per question at `price`, the price that NODE_TARGET links per question would be, S links per question."""
    den, cand = cell_norms(t)
    nq = int(np.unique(t["q"]).size)
    nd = t["kind"] == 1
    cnt = np.where(nd, 1.0, t["w"])
    s_links = float(cnt[t["grp"] == 0].sum())
    return {"questions": nq, "candidate_links_per_q": round(cand / nq, 1), "kept_links_per_q": round(price * cand / nq, 1), "price_at_2500_links_per_q": round(min(1.0, NODE_TARGET * nq / cand), 5),
            "S_links_per_q": round(s_links / nq, 1)}


def cmd_cap(a):
    Z.gate_pins()
    t0 = time.time()
    rec = {"part": "q", "declared_in": DECLARED, "zgp2_sha256": Z.script_sha256(), "zgp2a3_sha256": script_sha256(), "rho": Z.RHO, "grid": [g if np.isfinite(g) else "inf" for g in GRID],
           "scores": {k: list(v) for k, v in SCORES.items()}, "curves": {k: {} for k in SCORES}, "nodes": {}, "gates": {}}
    curves = {k: {} for k in SCORES}
    for d, v in Z.CELLS:
        key = f"{d}/{v}"
        t = tables("q", only=[key])[key]
        if "not_read" in t:
            stop(f"{key}: not read on the quarter")
        den, cand = cell_norms(t)
        nd = t["kind"] == 1
        # G-C5 on the cell: no cap keeps everything, a cap of 0 keeps exactly S
        for k in SCORES:
            S_ = cap_structure(t, score_keys(t, k))
            r, p = cap_curves(t, S_)
            if abs(r[:, -1].sum() - 1.0) > 1e-6 or abs(p[:, -1].sum() - 1.0) > 1e-6:
                stop(f"G-C5: {key}/{k}: no cap does not keep everything (reach {r[:, -1].sum()}, price {p[:, -1].sum()})")
            f0 = cap_keep(t, S_, (0.0, 0.0, 0.0))
            if not np.array_equal(f0 > 0.5, t["grp"] == 0) or abs(f0[t["grp"] != 0].sum()) > 1e-9:
                stop(f"G-C5: {key}/{k}: a cap of 0 does not keep exactly S")
            curves[k][key] = (r, p)
            rec["curves"][k][key] = {"reach": np.round(r, 5).tolist(), "price": np.round(p, 5).tolist()}
        rec["nodes"][key] = node_bound(t, 1.0)
        rec["gates"][key] = "G-C5 PASS"
        log(f"cap {key}: {len(SCORES)} scores ({time.time() - t0:.0f} s)")
        del t
    rec["by_score"] = {}
    for k in SCORES:
        prog = c2_program(curves[k])
        rec["by_score"][k] = {"program": prog, "required_uniform_cap_by_cell": required_caps(curves[k])}
        log(f"C2 {k}: " + ("INFEASIBLE" if prog is None else f"W {prog['W']} tau {prog['tau']:.4f}"))
    ok = {k: v["program"] for k, v in rec["by_score"].items() if v["program"] is not None}
    best = min(ok, key=lambda k: (ok[k]["tau"], sum(GRID.index(float("inf") if w == "inf" else w) for w in ok[k]["W"]), k)) if ok else None
    rec["best_score"] = best
    if best is not None:
        prog = rec["by_score"][best]["program"]
        rec["rule"] = {"kind": "c2", "score": best, "keys": list(SCORES[best]), "W": prog["W"], "tau": prog["tau"], "per_cell_on_Q": prog["per_cell"]}
        # the node bound at the chosen caps, per cell
        nb = {}
        for key, pc in prog["per_cell"].items():
            c = rec["nodes"][key]
            nb[key] = {"price": pc["price"], "kept_links_per_q": round(pc["price"] * c["candidate_links_per_q"], 1), "candidate_links_per_q": c["candidate_links_per_q"],
                       "price_at_2500_links_per_q": c["price_at_2500_links_per_q"], "S_links_per_q": c["S_links_per_q"]}
        rec["rule"]["node_bound"] = nb
    rec["seconds"] = round(time.time() - t0, 1)
    GP.wjson(OUT / "rule_q_c2.json", rec)
    return 0


# ── the final rule, the rule set, the node bound on the quarter ──────────────


def grid_w(x):
    return float("inf") if x == "inf" else float(x)


def keep_of(rr, t, S_cache=None):
    """The per-row kept fraction of a rule record (a key-space rule: 0/1; C2: the expectation)."""
    if rr.get("kind") == "c2":
        S_ = cap_structure(t, score_keys(t, rr["score"]))
        return cap_keep(t, S_, tuple(grid_w(w) for w in rr["W"])), True
    sp = Z.Space2(rr["attrs"], rr.get("by_step", True), dim=rr.get("dim", "grp"))
    x = np.array(rr["keep"], float)
    kk = sp.key_of(t["lab"] if sp.dim == "lab" else t["grp"], t["step"], t["B"])
    return (x[kk] > 0.5).astype(float), False


def eval_rule(t, rr):
    f, frac_ = keep_of(rr, t)
    den, cand = cell_norms(t)
    nd = t["kind"] == 1
    cnt = np.where(nd, 1.0, t["w"])
    reach = float((f * np.where(nd, t["w"], 0.0)).sum()) / den
    price = float((f * cnt).sum()) / cand
    return reach, price, f


def rule_from_space(blk, dim="lab"):
    sp = Z.Space2(blk["attrs"], blk.get("by_step", True), dim=blk.get("dim", dim))
    x = np.zeros(sp.K)
    x[blk["M"]["kept_keys"]] = 1.0
    return {"space": Z.sname(blk["attrs"]), "attrs": list(blk["attrs"]), "dim": blk.get("dim", dim), "by_step": blk.get("by_step", True), "keep": x.astype(int).tolist(), "seen": blk["seen"],
            "M_tau": blk["M"]["tau"]}


def best_space(spaces):
    return Z.best_of(spaces)


def load(n):
    p = OUT / n
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def cmd_a3(a):
    Z.gate_pins()
    t0 = time.time()
    p1, p2, p5 = load("rule_q_phase1.json"), load("rule_q_phase2.json"), load("rule_q_phase5.json")
    ru, a2, c2 = load("rule_q.json"), load("rule_q_a2.json"), load("rule_q_c2.json")
    if not (p1 and p2 and p5 and ru and a2 and c2):
        stop("phases 1, 2, 5, the final, A2 and C2 records are all needed")
    rules = {"rule_s11_final": {**ru["rule"], "kind": "keys"}}
    p1n, p2n, p5n = best_space(p1["spaces"]), best_space(p2["spaces"]), best_space(p5["spaces"])
    rules["rule_A"] = {**rule_from_space({**p1["spaces"][p1n], "dim": "grp"}, "grp"), "kind": "keys", "family": "A (phase 1 best)"}
    rules["rule_B1"] = {**rule_from_space({**p2["spaces"][p2n], "dim": "grp"}, "grp"), "kind": "keys", "family": "B1 (phase 2 best)"}
    rules["rule_C"] = {**rule_from_space({**p1["spaces"]["S0"], "dim": "grp"}, "grp"), "kind": "keys", "family": "C (groups x hop)"}
    if a2.get("rule"):
        rules["rule_A2"] = {**a2["rule"], "kind": "keys", "family": "A2 (labels)"}
    rules["rule_P5"] = {**rule_from_space({**p5["spaces"][p5n], "dim": "lab"}, "lab"), "kind": "keys", "family": "phase 5 best"}
    if c2.get("rule"):
        rules["rule_C2"] = {**c2["rule"], "family": "C2 (per-question caps)"}
    # the final rule: the smallest worst-cell price over every key space of sections 4, 12 (A2's record holds both), 14 and every C2 score;
    # ties: fewer kept keys, fewer attributes, declaration order (section 12, section 14, C2)
    gb = a2["global_best"]
    a2_rule = "rule_A2" if a2.get("rule") else "rule_s11_final"
    cand = [(round(gb["tau_M"], 9), a2["all_spaces_tau_M"][gb["label"]]["kept"], len(gb["attrs"]), 0, a2_rule, gb["label"], gb["tau_M"])]
    m5 = p5["spaces"][p5n]["M"]
    if m5["tau"] is not None:
        cand.append((round(m5["tau"], 9), m5["kept"], len(p5["spaces"][p5n]["attrs"]), 1, "rule_P5", f"lab:{p5n}", m5["tau"]))
    if c2.get("rule"):
        cand.append((round(c2["rule"]["tau"], 9), 10 ** 9, 0, 2, "rule_C2", f"C2 {c2['rule']['score']}", c2["rule"]["tau"]))
    best = min(cand, key=lambda x: x[:4])
    final_key, label, tau = best[4], best[5], best[6]
    kind = "c2" if final_key == "rule_C2" else "keys"
    tabs = tables("q")
    c4 = Z.gate_c4(tabs)
    rec = prov("q", tabs)
    rec.update({"phase": "a3", "G-C4": c4, "rho": Z.RHO, "final": {"kind": kind, "label": label, "tau": tau, "rule_key": final_key}, "phase5_best": {"space": p5n, "tau_M": p5["spaces"][p5n]["M"]["tau"],
                "kept": p5["spaces"][p5n]["M"]["kept"]}, "phase1_best": p1n, "phase2_best": p2n, "c2_best_score": c2.get("best_score"), "section_13_best": a2["global_best"],
                "phase5_sha256": hashlib.sha256((OUT / "rule_q_phase5.json").read_bytes()).hexdigest(), "c2_sha256": hashlib.sha256((OUT / "rule_q_c2.json").read_bytes()).hexdigest(),
                "rules": rules, "on_Q": {}})
    log(f"final: {label} tau {tau:.4f} (rule {final_key}); phase 5 best {p5n} {p5['spaces'][p5n]['M']['tau']:.4f}; C2 best {c2.get('best_score')}")
    worst, lowest = {}, {}
    for key, t in tabs.items():
        rec["on_Q"][key] = {}
        for rn, rr in rules.items():
            reach, price, _f = eval_rule(t, rr)
            nb = node_bound(t, price)
            rec["on_Q"][key][rn] = {"reach": round(reach, 4), "price": round(price, 4), **nb}
            worst[rn] = max(worst.get(rn, 0.0), price)
            lowest[rn] = min(lowest.get(rn, 1.0), reach)
    # G-C6: every rule, evaluated link by link here, gives the program's own number on the quarter (worst-cell price = tau, every reach >= rho)
    tau_of = {"rule_s11_final": ru["rule"]["M"]["tau"], "rule_A": p1["spaces"][p1n]["M"]["tau"], "rule_B1": p2["spaces"][p2n]["M"]["tau"], "rule_C": p1["spaces"]["S0"]["M"]["tau"],
              "rule_P5": m5["tau"], "rule_C2": c2["rule"]["tau"] if c2.get("rule") else None}
    if a2.get("rule"):
        tau_of["rule_A2"] = a2["rule"]["M"]["tau"]
    g6 = {}
    for rn, tt in tau_of.items():
        if tt is None:
            continue
        tol = 1e-6 if rn == "rule_C2" else 1e-4
        g6[rn] = {"tau": tt, "worst_price_here": round(worst[rn], 6), "lowest_reach_here": round(lowest[rn], 6)}
        if abs(worst[rn] - tt) > tol or lowest[rn] < Z.RHO - tol:
            stop(f"G-C6: {rn}: worst-cell price {worst[rn]:.6f} against the program's {tt:.6f}, lowest reach {lowest[rn]:.6f}")
    rec["G-C6"] = g6
    # beside runs and the class split for the final when it is a key space of phase 5 (the others have theirs)
    if kind == "keys" and final_key == "rule_P5":
        blk, (sp, names, r, p, mm, ee, M, L) = Z.solve_space(tabs, tuple(rules["rule_P5"]["attrs"]), True, full=True, dim="lab")
        rec["beside_P5"] = Z.beside_runs(tabs, tuple(rules["rule_P5"]["attrs"]), "lab", sp, names, r, p, M)
        rec["class_split_P5"] = Z.class_split(tabs, [("lab:" + rules["rule_P5"]["space"], tuple(rules["rule_P5"]["attrs"]), "lab")])
    rec["seconds"] = round(time.time() - t0, 1)
    GP.wjson(OUT / "rule_q_a3.json", rec)
    log(f"a3 record {GP.rel(OUT / 'rule_q_a3.json')} ({time.time() - t0:.0f} s)")
    return 0


# ── the rest of dev, once ────────────────────────────────────────────────────


def cmd_grade(a):
    Z.gate_pins()
    t0 = time.time()
    a3 = load("rule_q_a3.json")
    if not a3:
        stop("rule_q_a3.json is missing: the rule set is declared by it")
    rules = a3["rules"]
    res = {"part": "r", "declared_in": DECLARED, "zgp2_sha256": Z.script_sha256(), "zgp2a3_sha256": script_sha256(), "a3_sha256": hashlib.sha256((OUT / "rule_q_a3.json").read_bytes()).hexdigest(),
           "final_rule": a3["final"]["rule_key"], "rules": {}}
    cells_out = {w: {} for w in rules}
    for d, v in Z.CELLS:
        key = f"{d}/{v}"
        t = tables("r", only=[key], gold=True)[key]
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
        for w, rr in rules.items():
            reach, price, f = eval_rule(t, rr)
            if rr.get("kind") == "c2":
                keep = np.random.default_rng(20261126).random(f.size) < f
            else:
                keep = f > 0.5
            k_, t_, per = GP.path_retention(T, keep)
            seen = np.asarray(rr["seen"]) if rr.get("seen") is not None else None
            g = {"status": "read", "reach": round(reach, 4), "price": round(price, 4), "path_retention": round(k_ / t_, 4) if t_ else float("nan"), "golds_with_units": t_,
                 "path_by_distance": {str(dd): [int(a_), int(b_)] for dd, (a_, b_) in per.items()}, "needed_weight": round(float(t["w"][nd].sum()), 4), **node_bound(t, price)}
            if seen is not None and rr.get("kind") != "c2":
                sp = Z.Space2(rr["attrs"], rr.get("by_step", True), dim=rr.get("dim", "grp"))
                kk = sp.key_of(t["lab"] if sp.dim == "lab" else t["grp"], t["step"], t["B"])
                g["needed_weight_in_keys_unseen_on_Q"] = round(float(t["w"][nd & (seen[kk] == 0)].sum()) / float(t["w"][nd].sum()), 4)
            g["SCREEN_IN"] = bool(g["reach"] >= GP.SCREEN_REACH and g["price"] <= GP.SCREEN_PRICE and g["path_retention"] >= GP.SCREEN_PATH)
            cells_out[w][key] = g
            log(f"{w} {key}: reach {g['reach']} price {g['price']} path {g['path_retention']} kept/q {g['kept_links_per_q']} SCREEN_IN {g['SCREEN_IN']} ({time.time() - t0:.0f} s)")
        del t, T
    for w, cs in cells_out.items():
        read = [c for c in cs.values() if c["status"] == "read"]
        res["rules"][w] = {"family": rules[w].get("family"), "space": rules[w].get("space", rules[w].get("score")), "cells": cs, "cells_read": len(read), "screen_in": int(sum(c["SCREEN_IN"] for c in read)),
                           "call": "UNIVERSAL_RULE" if (read and len(read) == len(Z.CELLS) and all(c["SCREEN_IN"] for c in read)) else "NOT_UNIVERSAL"}
    GP.wjson(OUT / "grade_r_a3.json", res)
    fr = res["final_rule"]
    log(f"grade: final {fr}: {res['rules'][fr]['call']} ({res['rules'][fr]['screen_in']}/{res['rules'][fr]['cells_read']} cells SCREEN_IN) ({time.time() - t0:.0f} s)")
    return 0


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(a):
    L = []
    fx = lambda x: "n/a" if x is None else f"{x:.4f}"  # noqa: E731
    p5 = load("rule_q_phase5.json")
    if p5:
        L.append("### Phase 5 (quarter, dimension lab): tau by space")
        L.append("| space | keys | tau L | tau U | tau M | M status | kept |")
        L.append("|---|---|---|---|---|---|---|")
        for n, b in p5["spaces"].items():
            c = b["chain_L_U_M_T"]
            L.append(f"| {n} | {b['keys']} | {fx(c[0])} | {fx(c[1])} | {fx(c[2])} | {b['M']['status']} | {b['M']['kept']} |")
    c2 = load("rule_q_c2.json")
    if c2:
        L.append("\n### C2 (quarter): per score, the best universal caps (W1 >= W2 >= W3)")
        for k, v in c2["by_score"].items():
            pg = v["program"]
            L.append(f"- {k} {c2['scores'][k]}: " + ("INFEASIBLE" if pg is None else f"W {pg['W']}, worst-cell price {pg['tau']:.4f}; per cell {pg['per_cell']}") + f"; uniform cap each cell alone needs: {v['required_uniform_cap_by_cell']}")
        if c2.get("rule"):
            L.append(f"- node bound at the chosen caps: {json.dumps(c2['rule']['node_bound'])}")
    a3 = load("rule_q_a3.json")
    if a3:
        L.append(f"\n### A3 final: {a3['final']['label']} tau {a3['final']['tau']:.4f} (rule {a3['final']['rule_key']}); section 13's best {a3['section_13_best']['label']} {a3['section_13_best']['tau_M']:.4f}")
        for key, rs in a3["on_Q"].items():
            L.append(f"\n**{key}** (quarter, per rule: reach / price / kept links per question / candidates per question / price at 2,500 / S links per question)")
            for rn, g in rs.items():
                L.append(f"  - {rn}: {g['reach']} / {g['price']} / {g['kept_links_per_q']} / {g['candidate_links_per_q']} / {g['price_at_2500_links_per_q']} / {g['S_links_per_q']}")
    gr = load("grade_r_a3.json")
    if gr:
        L.append("\n### Grade (rest of dev)")
        for w, g in gr["rules"].items():
            L.append(f"- {w} ({g['family']}; {g['space']}): {g['call']} ({g['screen_in']}/{g['cells_read']} cells SCREEN_IN)")
            for k, c in g["cells"].items():
                if c["status"] == "read":
                    L.append(f"  - {k}: reach {c['reach']}, price {c['price']}, path retention {c['path_retention']} ({c['golds_with_units']} golds), kept links per question {c['kept_links_per_q']} of {c['candidate_links_per_q']} (price at 2,500: {c['price_at_2500_links_per_q']}; S per question {c['S_links_per_q']}) -> {c['SCREEN_IN']}")
    txt = "\n".join(L)
    (OUT / "report_a3.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def brute_cap(rows, W):
    """Reference for the C2 cap on one (step, question): rows = [(key tuple, count (int), needed weight or None)]; enumerates every order of the candidate units inside each tie group and averages
    the kept units and the kept needed weight (a unit of a row is kept when its position, best-first, is <= W)."""
    rows = sorted(rows, key=lambda r: tuple(-x for x in r[0]))
    groups = []
    for r in rows:
        if groups and groups[-1][0][0] == r[0]:
            groups[-1].append(r)
        else:
            groups.append([r])
    ahead, kept_cnt, kept_need = 0, 0.0, 0.0
    for g in groups:
        units = [(i, r[2]) for i, r in enumerate(g) for _ in range(r[1])]
        n = len(units)
        orders = list(itertools.permutations(range(n)))
        kc, kn = 0.0, 0.0
        for o in orders:
            for pos, u in enumerate(o):
                if ahead + pos + 1 <= W:
                    kc += 1
                    kn += units[u][1] or 0.0
        kept_cnt += kc / len(orders)
        kept_need += kn / len(orders)
        ahead += n
    return kept_cnt, kept_need


def selftest():
    rng = np.random.default_rng(11)
    # bins
    assert cd_bin(None) == 1 and cd_bin(400) == 2 and cd_bin(321) == 2 and cd_bin(320) == 3 and cd_bin(161) == 3 and cd_bin(160) == 4 and cd_bin(81) == 4 and cd_bin(80) == 5 and cd_bin(41) == 5 and cd_bin(40) == 6 and cd_bin(0) == 6
    assert zbin(1.0, 1.0, 0.0) == 6 and zbin(0.0, 1.0, 0.0) == 1 and zbin(0.1, 1.0, 0.0) == 2 and zbin(0.39, 1.0, 0.0) == 3 and zbin(0.4, 1.0, 0.0) == 4 and zbin(5.0, 2.0, 2.0) == 6
    assert score_bins(np.array([0, 1, 3]), np.array([1.0, 0.5, 0.0], np.float32)).tolist() == [0, 6, 1]
    # co-mention distance: the two implementations agree on strings
    for title, text, q in [("Paris", "He lived in Paris, the capital of France, and wrote about wine there.", "which wine did he write about in france"),
                           ("Ann Lee", "x " * 100 + "Ann Lee won the prize for physics in 1999. " + "y " * 100, "who won the physics prize"),
                           ("Rome", "Rome is old.", "what is old"), ("Rome", "wine " * 60 + "Rome is old.", "wine in Rome")]:
        qtok = G1.content_tokens(q)
        m = re.search(r"(?<![^\W_])" + re.escape(title) + r"(?![^\W_])", text)
        fo = m.start()
        got = cd_bin(cd_of(head_index(text), qtok - G1.content_tokens(title), fo, len(title)))
        assert got == cd_independent(title, text, q), (title, got, cd_independent(title, text, q))
        for nm, w in WIDTHS.items():
            assert tw_w(title, text, qtok, fo, w) == tw_independent_w(title, text, q, w), (title, nm)
    # C2 against exhaustive enumeration of the tie orders on toy (step, question) groups
    for trial in range(40):
        n = int(rng.integers(2, 5))
        keys = rng.integers(0, 3, (n, 2))
        cnt = rng.integers(1, 3, n)
        while cnt.sum() > 6:
            cnt[int(np.argmax(cnt))] -= 1
        cnt = np.maximum(cnt, 1)
        nd = rng.random(n) < 0.5
        nd_cnt = np.where(nd, 1, cnt)
        w = np.where(nd, rng.random(n) + 0.1, 1.0)
        t = {"kind": nd.astype(int), "w": np.where(nd, w, cnt.astype(float)), "grp": np.ones(n, int), "step": np.full(n, 1), "q": np.zeros(n, int)}
        S_ = cap_structure(t, [keys[:, 0].astype(float), keys[:, 1].astype(float)])
        for W in (0, 1, 2, 3, 5, 100):
            f = np.clip((W - S_["ahead"]) / np.maximum(S_["T"], 1e-300), 0, 1)
            got_cnt, got_need = float((f * S_["T"]).sum()), float((f * S_["needw"]).sum())
            want_cnt, want_need = brute_cap([(tuple(keys[i]), int(nd_cnt[i]), float(w[i]) if nd[i] else None) for i in range(n)], W)
            assert abs(got_cnt - want_cnt) < 1e-9 and abs(got_need - want_need) < 1e-9, (trial, W, got_cnt, want_cnt, got_need, want_need)
    # S is kept whole and not counted; two questions and two steps do not mix
    t = {"kind": np.array([1, 0, 0, 1, 0, 1]), "w": np.array([1.0, 3.0, 4.0, 2.0, 5.0, 1.0]), "grp": np.array([0, 1, 1, 1, 2, 0]), "step": np.array([1, 1, 1, 1, 2, 2]), "q": np.array([0, 0, 0, 1, 1, 1])}
    S_ = cap_structure(t, [np.array([0.0, 2.0, 1.0, 0.0, 0.0, 0.0])])
    f = cap_keep(t, S_, (3.0, 3.0, 3.0))
    assert f[0] == 1 and f[5] == 1 and abs(f[1] - 1.0) < 1e-12 and abs(f[2] - 0.0) < 1e-12, f      # q0 step 1: the control of weight 3 (key 2) is first, 3 <= 3 is kept; the weight-4 control is behind it
    f = cap_keep(t, S_, (5.0, 5.0, 3.0))
    assert abs(f[2] - 0.5) < 1e-12 and abs(f[3] - 1.0) < 1e-12 and abs(f[4] - 1.0) < 1e-12, f      # (5 - 3) / 4 of the second control; q1 step 1 is its own block; step 2 has a cap of 5 for weight 5
    f = cap_keep(t, S_, (0.0, 0.0, 0.0))
    assert f[0] == 1 and f[5] == 1 and f[1:5].sum() == 0, f                                          # a cap of 0 keeps exactly S
    f = cap_keep(t, S_, (float("inf"),) * 3)
    assert (f == 1).all(), f                                                                         # no cap keeps everything
    # cap_curves equals the direct evaluation of cap_keep for the same cap at every hop, at each grid value
    grid = (0, 1, 3, 4, 5, 8, float("inf"))
    rc, pc = cap_curves(t, S_, grid)
    den, cand = cell_norms(t)
    nd_ = t["kind"] == 1
    for gi, W in enumerate(grid):
        f = cap_keep(t, S_, (float(W),) * 3)
        assert abs(float((f * np.where(nd_, t["w"], 0.0)).sum()) / den - rc[:, gi].sum()) < 1e-12 and abs(float((f * np.where(nd_, 1.0, t["w"])).sum()) / cand - pc[:, gi].sum()) < 1e-12, (gi, W)
    assert abs(rc[:, -1].sum() - 1.0) < 1e-12 and abs(pc[:, -1].sum() - 1.0) < 1e-12
    # the program: reach and price are the same line here, so the cheapest triple with a sum of values >= 0.95 is (no cap, no cap, 300,000); the order W1 >= W2 >= W3 holds
    line = np.tile(np.linspace(0.1, 1.0, len(GRID)), (3, 1)) / 3
    pr = c2_program({"c": (line, line)}, 0.95)
    assert pr is not None
    gi = [GRID.index(float("inf") if w == "inf" else w) for w in pr["W"]]
    assert gi == [10, 10, 9] and abs(pr["tau"] - 0.97) < 1e-9, (gi, pr["tau"])
    assert c2_program({"c": (line * 0.5, line)}, 0.95) is None                                       # infeasible when no triple reaches rho
    assert required_caps({"c": (line, line)}, 0.95)["c"] == "inf"
    print("zgp2a3 selftest: PASS")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for n in ("text2", "score"):
        c = sub.add_parser(n)
        c.add_argument("--part", required=True, choices=["q", "r"])
        c.add_argument("--ds", nargs="*")
    for n in ("solve5", "cap", "a3", "grade", "report"):
        sub.add_parser(n)
    a = ap.parse_args(argv)
    return {"selftest": lambda: selftest(), "text2": lambda: cmd_text2(a), "score": lambda: cmd_score(a), "solve5": lambda: cmd_solve5(a), "cap": lambda: cmd_cap(a), "a3": lambda: cmd_a3(a),
            "grade": lambda: cmd_grade(a), "report": lambda: cmd_report(a)}[a.cmd]()


if __name__ == "__main__":
    sys.exit(main())
