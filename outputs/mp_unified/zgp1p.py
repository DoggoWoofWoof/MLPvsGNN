"""GP1's program (docs/GP1_GOLD_LINK_PATTERNS.md sections 6, 7, 9; amendments in section 13 and 14): reads the link-level work files zgp1.py wrote and
    screen   the stratified AUC of every attribute in every judged cell (needed against control within a (question, source, step) stratum), the data-derived dot cuts,
             the votes guard, the calls, the attribute choice K3 / K4 and the directions, and the per-cell tables of the pattern (needed share against control share);
    solve    the constrained program on the quarter's columns: L (LP relaxation), U (integer, no monotone), M (integer, monotone; the call), T (threshold
             enumeration), the baseline K0, K4, S free, rho 0.90 / 0.99, size caps, own optima, the frontier; writes the rule record;
    grade    the committed rule on the rest of dev (reach, price, gold path retention) and the call;
    report   the tables as text.
Nothing here trains or reads a model. Pure numpy and scipy (HiGHS); the laptop runs it. Outputs under outputs/gp1/ only; s1eval is never opened.

    python outputs/mp_unified/zgp1p.py selftest
    python outputs/mp_unified/zgp1p.py screen | solve | grade | report
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import hashlib  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import zap4 as AP  # noqa: E402
from scipy.optimize import Bounds, LinearConstraint, linprog, milp  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

OUT = ROOT / "outputs" / "gp1"
DECLARED = "docs/GP1_GOLD_LINK_PATTERNS.md"
CELLS = (("multihoprag", "RK"), ("musique", "RK"), ("squad", "RK"), ("metaqa", "SRK"), ("webqsp", "SRK"), ("hotpotqa", "RK"), ("hotpotqa", "SRK"),
         ("2wiki", "RK"), ("2wiki", "SRK"))
DATASETS = ("multihoprag", "musique", "squad", "metaqa", "webqsp", "hotpotqa", "2wiki")
CAND = ("dot", "rank", "votes", "dir", "exact", "ntok", "caps", "lead", "off", "cnt", "qov")             # A2 .. A12
AID = {a: f"A{i + 2}" for i, a in enumerate(CAND)}
INFO = {"dot": (3, None), "rank": (3, None), "votes": (3, None), "dir": (3, None), "exact": (3, None), "ntok": (3, None), "caps": (3, None),
        "lead": (3, 2), "off": (4, 3), "cnt": (4, 3), "qov": (3, None)}                                   # (bins in all, the na bin)
BIN_NAMES = {"dot": ("low", "mid", "high"), "rank": ("1-3", "4-30", ">30"), "votes": ("1", "2", "3+"), "dir": ("none", "one-way", "mutual"),
             "exact": ("none", "folded/alias", "exact"), "ntok": ("1", "2", "3+"), "caps": ("lower", "mixed", "all caps"), "lead": ("no", "yes", "na"),
             "off": ("<=.10", ".10-.50", ">.50", "na"), "cnt": ("1", "2", "3+", "na"), "qov": ("0", "1", "2+")}
GRP = ("S", "R", "K")
RHO, RHOS = 0.95, (0.90, 0.99)
CAPS_P = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80)
SIZES = (12, 24, 48)
SCREEN_REACH, SCREEN_PRICE, SCREEN_PATH = 0.92, 0.50, 0.90
SEP_MIN, WEAK_MIN = 0.10, 0.05
VOTES_RHO = 0.70
MILP_LIMIT = 600
EMPTY = np.zeros(0, np.int64)
SUFFIX = ""                    # "_limit" under --smoke: the 5-question smoke files, code paths only, no number is read


def stop(msg):
    raise SystemExit(f"[zgp1p] STOP: {msg}")


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def wjson(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)), encoding="utf-8")
    tmp.replace(p)


# ── loading ──────────────────────────────────────────────────────────────────


def read_extract(d, part):
    """The extract record of a dataset and part: the one file, or the shards' files merged (every shard of the same N must be present; the chunks sorted by
    their question offset, the counts added, the capped flag recomputed from the totals). (None, None) when absent or when a shard is missing."""
    p = OUT / d / f"extract_{part}{SUFFIX}.json"
    if p.is_file():
        return json.loads(p.read_text(encoding="utf-8")), hashlib.sha256(p.read_bytes()).hexdigest()
    import re as _re
    files = sorted((OUT / d).glob(f"extract_{part}_s*of*{SUFFIX}.json")) if (OUT / d).is_dir() else []
    files = [f for f in files if _re.fullmatch(rf"extract_{part}_s\d+of\d+{SUFFIX}\.json", f.name)]
    if not files:
        return None, None
    recs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    ns = {r["shard"][1] for r in recs}
    got = sorted(r["shard"][0] for r in recs)
    if len(ns) != 1 or got != list(range(next(iter(ns)))):
        return None, None
    if any(r.get("status") != "read" for r in recs):
        return recs[0], hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    out = dict(recs[0])
    out["versions"] = {}
    for v in recs[0]["versions"]:
        stats = None
        chunks = []
        for r in recs:
            vr = r["versions"][v]
            chunks += vr["chunks"]
            if stats is None:
                stats = {k: (list(x) if isinstance(x, list) else x) for k, x in vr["stats"].items()}
            else:
                for k, x in vr["stats"].items():
                    stats[k] = [a + b for a, b in zip(stats[k], x)] if isinstance(x, list) else stats[k] + x
        chunks.sort(key=lambda c: c.rsplit("_", 1)[1])
        stats["found"] = stats["dist"][0]
        stats["reachable_missed"] = int(sum(stats["dist"][1:4]))
        stats["unreached"] = stats["dist"][4]
        out["versions"][v] = {"stats": stats, "chunks": chunks}
        if stats["reachable_missed"] and stats["capped_golds"] / stats["reachable_missed"] > 0.05:
            out["versions"][v]["flag"] = "CAPPED"
    out["shards"] = len(recs)
    return out, hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()

COLS = ("q", "a", "c", "step", "kind", "w", "dot", "rank", "votes", "tdeg", "prof", "grp", "rel", "dir", "exact", "ntok", "caps", "lead", "off", "cnt", "qov")


def load_cell(d, v, part, cols=COLS, gold=True):
    """The cell's link table of one part: the columns concatenated over the chunks, with the gold arrays and the gold-to-unit incidences (global indices).
    None when the extract is absent or NOT_READ; {'not_read': why} when flagged."""
    rec, esha = read_extract(d, part)
    if rec is None:
        return None
    if rec.get("status") == "NOT_READ":
        return {"not_read": rec["why"]}
    vr = rec["versions"].get(v)
    if vr is None:
        return None
    if vr.get("flag") == "CAPPED":
        return {"not_read": "CAPPED: more than 5 percent of the reachable golds hit the per-gold cap"}
    T = {k: [] for k in cols}
    G = {k: [] for k in ("gq", "gnode", "gdist", "gcapped", "gunits", "inc_g", "inc_row")}
    goff = roff = 0
    for c in vr["chunks"]:
        z = np.load(ROOT / c)
        for k in cols:
            T[k].append(z[k])
        if gold:
            G["gq"].append(z["gq"])
            G["gnode"].append(z["gnode"])
            G["gdist"].append(z["gdist"])
            G["gcapped"].append(z["gcapped"])
            G["gunits"].append(z["gunits"])
            G["inc_g"].append(z["inc_g"].astype(np.int64) + goff)
            G["inc_row"].append(z["inc_row"].astype(np.int64) + roff)
        goff += int(z["gq"].size)
        roff += int(z["a"].size)
    out = {k: (np.concatenate(x) if x else np.zeros(0)) for k, x in T.items()}
    if gold:
        out["G"] = {k: (np.concatenate(x) if x else np.zeros(0, np.int64)) for k, x in G.items()}
    out["stats"] = vr["stats"]
    out["extract_sha"] = esha
    return out


def cells_present(part):
    out = []
    for d, v in CELLS:
        rec, _sha = read_extract(d, part)
        if rec is not None and (rec.get("status") == "NOT_READ" or v in rec.get("versions", {})):
            out.append((d, v))
    return out


# ── binning ──────────────────────────────────────────────────────────────────


def wquantiles(v, w, qs):
    o = np.argsort(v, kind="stable")
    v, cw = v[o], np.cumsum(w[o])
    return [float(v[min(int(np.searchsorted(cw, q * cw[-1], side="left")), v.size - 1)]) for q in qs]


def binned(T, cuts):
    """Every attribute's bin (int8) by the declared bins; dot by the two cut points; grp is T['grp'] (0 S, 1 R, 2 K)."""
    dot = T["dot"].astype(np.float64)
    off = T["off"].astype(np.float64)
    b = {"dot": (dot > cuts[0]).astype(np.int8) + (dot > cuts[1]).astype(np.int8),
         "rank": np.where(T["rank"] <= 3, 0, np.where(T["rank"] <= 30, 1, 2)).astype(np.int8),
         "votes": np.clip(T["votes"].astype(np.int64) - 1, 0, 2).astype(np.int8),
         "dir": np.where(T["dir"] == 0, 0, np.where(T["dir"] == 3, 2, 1)).astype(np.int8),
         "exact": np.where(T["exact"] == 0, 0, np.where(T["exact"] == 3, 2, 1)).astype(np.int8),
         "ntok": (np.clip(T["ntok"].astype(np.int64), 1, 3) - 1).astype(np.int8),
         "caps": np.clip(T["caps"].astype(np.int64), 0, 2).astype(np.int8),
         "lead": np.clip(T["lead"].astype(np.int64), 0, 2).astype(np.int8),
         "off": np.where(np.isnan(off), 3, np.where(off <= 0.10, 0, np.where(off <= 0.50, 1, 2))).astype(np.int8),
         "cnt": np.where(T["cnt"] == 0, 3, np.clip(T["cnt"].astype(np.int64), 1, 3) - 1).astype(np.int8),
         "qov": np.clip(T["qov"].astype(np.int64), 0, 2).astype(np.int8)}
    return b


# ── the screen ───────────────────────────────────────────────────────────────


def strata_parts(q, a, step, kind, w, b, nb, nqm):
    """Per question (A_q, B_q) of the stratified AUC: within each (question, source, step) stratum holding both needed and control links, the weighted
    probability that a needed link's bin exceeds a control's (ties half); strata weighted by their needed weight. AUC = sum A / sum B."""
    key = (q.astype(np.int64) * (1 << 31) + a.astype(np.int64)) * 4 + step.astype(np.int64)
    uk, inv = np.unique(key, return_inverse=True)
    ns = uk.size
    nd = kind == 1
    wn = np.bincount(inv[nd] * nb + b[nd], weights=w[nd], minlength=ns * nb).reshape(ns, nb)
    wc = np.bincount(inv[~nd] * nb + b[~nd], weights=w[~nd], minlength=ns * nb).reshape(ns, nb)
    cum = np.cumsum(wc, axis=1) - wc
    num = (wn * (cum + 0.5 * wc)).sum(1)
    tn, tc = wn.sum(1), wc.sum(1)
    ok = (tn > 0) & (tc > 0)
    auc = np.where(ok, num / np.where(ok, tn * tc, 1.0), 0.0)
    om = np.where(ok, tn, 0.0)
    sq = (uk // 4) // (1 << 31)
    return (np.bincount(sq, weights=om * auc, minlength=nqm), np.bincount(sq, weights=om, minlength=nqm))


def auc_ci(A, B):
    keep = B > 0
    if not keep.any():
        return None
    r = AP.boot(lambda s: s["A"] / np.where(s["B"] > 0, s["B"], np.nan), {"A": A[keep], "B": B[keep]})
    return None if r is None else {"auc": r["pt"], "lo": r["ci"][0], "hi": r["ci"][1], "questions": int(keep.sum())}


def attr_calls(per_cell):
    """per_cell: {cell: {'auc', 'lo', 'hi'} or None} -> (call, score, sign, coverage notes)."""
    rows = {c: r for c, r in per_cell.items() if r is not None}
    if not rows or len(rows) < len(per_cell):
        return "NO_SIGNAL", 0.0, 0, "undefined in some cell"
    dev = {c: r["auc"] - 0.5 for c, r in rows.items()}
    signs = {c: (1 if r["lo"] > 0.5 else (-1 if r["hi"] < 0.5 else 0)) for c, r in rows.items()}
    if 1 in signs.values() and -1 in signs.values():
        return "SIGN_SPLIT", 0.0, 0, ""
    mind = min(abs(x) for x in dev.values())
    allsig = all(s != 0 for s in signs.values()) and len(set(signs.values())) == 1
    if allsig and mind >= SEP_MIN:
        return "SEPARATES", mind, signs[next(iter(signs))], ""
    mean = float(np.mean([abs(x) for x in dev.values()]))
    return ("WEAK" if mean >= WEAK_MIN else "NO_SIGNAL"), mind, int(np.sign(np.mean(list(dev.values())))), ""


def cmd_screen(a):
    part, t0 = "q", time.time()
    cells = cells_present(part)
    if not cells:
        stop("no extract_q.json under outputs/gp1")
    log("pass 1: the dot cuts over the pooled control links, each dataset weighted equally")
    vals, wts = [], []
    ds_seen = sorted({d for d, _v in cells})
    for d in ds_seen:
        vv, ww = [], []
        for dd, v in cells:
            if dd != d:
                continue
            T = load_cell(d, v, part, cols=("kind", "w", "dot"), gold=False)
            if T is None or "not_read" in T:
                continue
            m = T["kind"] == 0
            vv.append(T["dot"][m].astype(np.float64))
            ww.append(T["w"][m].astype(np.float64))
        if vv:
            v_, w_ = np.concatenate(vv), np.concatenate(ww)
            vals.append(v_)
            wts.append(w_ / w_.sum())
    cuts = wquantiles(np.concatenate(vals), np.concatenate(wts) / len(wts), (1 / 3, 2 / 3))
    log(f"dot cuts {cuts} over {len(vals)} datasets")
    res = {"part": part, "declared_in": DECLARED, "dot_cuts": cuts, "dot_cut_datasets": ds_seen, "cells": {}, "attrs": {}, "votes_guard": {}}
    percell = {}
    guard_arrays = {}
    for d, v in cells:
        T = load_cell(d, v, part)
        key = f"{d}/{v}"
        if T is None:
            continue
        if "not_read" in T:
            res["cells"][key] = {"status": "NOT_READ", "why": T["not_read"]}
            continue
        G = T["G"]
        nqm = int(T["q"].max()) + 1 if T["q"].size else 1
        B = binned(T, cuts)
        nd = T["kind"] == 1
        dist = np.bincount(G["gdist"].astype(np.int64), minlength=5).tolist() if G["gdist"].size else [0] * 5
        reach_n = float(T["w"][nd].sum())
        cell = {"status": "read", "gold_distance_found_1_2_3_unreached": dist, "needed_weight": round(reach_n, 4), "needed_units": int(nd.sum()),
                "controls_sampled": int((~nd).sum()), "controls_exact_weight": round(float(T["w"][~nd].sum()), 2),
                "needed_by_step": np.bincount(T["step"][nd].astype(np.int64), minlength=4)[1:].tolist(),
                "needed_by_grp": np.bincount(T["grp"][nd].astype(np.int64), minlength=3).tolist(),
                "needed_weight_by_grp": [round(float(x), 4) for x in np.bincount(T["grp"][nd].astype(np.int64), weights=T["w"][nd], minlength=3)],
                "control_weight_by_grp": [round(float(x), 2) for x in np.bincount(T["grp"][~nd].astype(np.int64), weights=T["w"][~nd], minlength=3)],
                "capped_golds": int(G["gcapped"].sum()), "stats": T["stats"], "tables": {}, "auc": {}}
        prof = T["prof"].astype(np.int64)
        cell["needed_by_profile"] = {str(int(k)): int((nd & (prof == k)).sum()) for k in np.unique(prof[nd])} if nd.any() else {}
        if reach_n <= 0:
            cell["status"] = "NO_REACHABLE_GOLD"
            res["cells"][key] = cell
            continue
        for at in CAND:
            nb, na = INFO[at]
            b = B[at].astype(np.int64)
            tab_n = np.bincount(b[nd], weights=T["w"][nd], minlength=nb)
            tab_c = np.bincount(b[~nd], weights=T["w"][~nd], minlength=nb)
            cell["tables"][at] = {"bins": BIN_NAMES[at], "needed_share": [round(float(x) / max(tab_n.sum(), 1e-12), 4) for x in tab_n],
                                  "control_share": [round(float(x) / max(tab_c.sum(), 1e-12), 4) for x in tab_c],
                                  "needed_units": np.bincount(b[nd], minlength=nb).tolist()}
            m = np.ones(b.size, bool) if na is None else (b != na)
            nbo = nb if na is None else nb - 1
            A_, B_ = strata_parts(T["q"][m], T["a"][m], T["step"][m], T["kind"][m], T["w"][m], b[m], nbo, nqm)
            percell.setdefault(at, {})[key] = (A_, B_)
            cell["auc"][at] = auc_ci(A_, B_)
            cell["tables"][at]["needed_weight_defined"] = round(float(T["w"][nd & m].sum()) / max(reach_n, 1e-12), 4)
        if v == "SRK" or d in ("metaqa", "webqsp"):
            rid = T["rel"][nd & (T["rel"] >= 0)]
            if rid.size:
                wr = np.bincount(T["rel"][nd & (T["rel"] >= 0)].astype(np.int64), weights=T["w"][nd & (T["rel"] >= 0)])
                top = np.argsort(-wr)[:10]
                allr = T["rel"][(T["rel"] >= 0)].astype(np.int64)
                wc_ = np.bincount(allr[~nd[T["rel"] >= 0]], weights=T["w"][(T["rel"] >= 0)][~nd[T["rel"] >= 0]], minlength=wr.size) if allr.size else np.zeros(wr.size)
                cell["relations_top"] = [{"rel": int(i), "needed_weight": round(float(wr[i]), 4),
                                          "control_weight": round(float(wc_[i]) if i < wc_.size else 0.0, 2)} for i in top if wr[i] > 0]
        guard_arrays.setdefault(d, []).append((T["votes"], T["tdeg"]))
        res["cells"][key] = cell
        log(f"{key}: screened ({time.time() - t0:.0f} s)")
    # votes guard
    drop_votes = False
    for d, lst in guard_arrays.items():
        vo = np.concatenate([x[0] for x in lst]).astype(np.float64)
        td = np.concatenate([x[1] for x in lst]).astype(np.float64)
        if vo.size > 200_000:
            sel = np.argsort(np.random.default_rng(0).random(vo.size))[:200_000]
            vo, td = vo[sel], td[sel]
        rho = float(spearmanr(vo, td).statistic) if vo.size > 2 and vo.std() > 0 and td.std() > 0 else float("nan")
        res["votes_guard"][d] = round(rho, 4)
        if np.isfinite(rho) and rho > VOTES_RHO:
            drop_votes = True
    res["votes_dropped"] = bool(drop_votes)
    read_cells = [k for k, c in res["cells"].items() if c["status"] == "read"]
    pooled = {}
    for at in CAND:
        pc = {k: res["cells"][k]["auc"].get(at) for k in read_cells}
        call, score, sign, note = attr_calls(pc)
        As = sum(percell[at][k][0].sum() for k in read_cells if k in percell.get(at, {}))
        Bs = sum(percell[at][k][1].sum() for k in read_cells if k in percell.get(at, {}))
        pooled[at] = float(As / Bs) if Bs > 0 else float("nan")
        direction = 1 if (np.isfinite(pooled[at]) and pooled[at] >= 0.5) else -1
        res["attrs"][at] = {"id": AID[at], "call": call, "score": round(float(score), 4), "direction": direction, "pooled_auc": round(pooled[at], 4),
                            "per_cell": pc, "note": note, "in_program": not (at == "votes" and drop_votes)}
    elig = [at for at in CAND if res["attrs"][at]["call"] != "SIGN_SPLIT" and res["attrs"][at]["in_program"] and res["attrs"][at]["score"] > 0]
    order = sorted(elig, key=lambda at: (-res["attrs"][at]["score"], CAND.index(at)))
    res["ranking"] = order
    res["K3"], res["K4"], res["K0"] = order[:3], order[:4], ["dot"]
    res["cells_read"] = read_cells
    res["cells_not_read"] = [k for k, c in res["cells"].items() if c["status"] != "read"]
    res["seconds"] = round(time.time() - t0, 1)
    res["zgp1p_sha256"] = script_sha256()
    wjson(OUT / f"screen_q{SUFFIX}.json", res)
    log(f"screen: K3 {res['K3']}  K4 {res['K4']}  votes dropped {drop_votes}  ({rel(OUT / f'screen_q{SUFFIX}.json')})")
    return 0


# ── keys, columns ────────────────────────────────────────────────────────────


class Space:
    """grp x the chosen attributes' bins: key ids, the monotone pairs, the threshold structure. dirs[i] = +1: keep the larger bins first."""

    def __init__(self, attrs, dirs):
        self.attrs, self.dirs = list(attrs), list(dirs)
        self.nbs = [INFO[a][0] for a in attrs]
        self.nas = [INFO[a][1] for a in attrs]
        self.nord = [nb - (0 if na is None else 1) for nb, na in zip(self.nbs, self.nas)]
        self.K = 3 * int(np.prod(self.nbs)) if attrs else 3
        ks = np.arange(self.K)
        self.grp = ks // (self.K // 3)
        rest = ks % (self.K // 3)
        self.bins = []
        for nb in reversed(self.nbs):
            self.bins.append(rest % nb)
            rest = rest // nb
        self.bins = list(reversed(self.bins))
        self.stride = []
        s = 1
        for nb in reversed(self.nbs):
            self.stride.append(s)
            s *= nb
        self.stride = list(reversed(self.stride))

    def key_of(self, T, B):
        k = T["grp"].astype(np.int64)
        for at, nb in zip(self.attrs, self.nbs):
            k = k * nb + B[at].astype(np.int64)
        return k

    def label(self, k):
        g = GRP[self.grp[k]]
        return g + "|" + "|".join(f"{at}={BIN_NAMES[at][int(self.bins[i][k])]}" for i, at in enumerate(self.attrs))

    def mono_pairs(self):
        """(k, k'): the rule keeping k keeps k', the key one bin better in the attribute's direction, the others equal; the na bin is outside the order."""
        out = []
        for k in range(self.K):
            for i in range(len(self.attrs)):
                b = int(self.bins[i][k])
                if self.nas[i] is not None and b == self.nas[i]:
                    continue
                b2 = b + self.dirs[i]
                if 0 <= b2 < self.nord[i]:
                    out.append((k, k + (b2 - b) * self.stride[i]))
        return out

    def threshold_masks(self, g):
        """For group g: every threshold triple t (t_i = how many ordered bins, from the best end, are kept) -> (t, boolean mask over the K keys)."""
        out = []
        rngs = [range(n + 1) for n in self.nord]
        for t in itertools.product(*rngs):
            m = self.grp == g
            for i, ti in enumerate(t):
                b = self.bins[i]
                if self.nas[i] is not None:
                    keep = np.where(b == self.nas[i], ti == self.nord[i], False) | ((b != self.nas[i]) & self._best(i, b, ti))
                else:
                    keep = self._best(i, b, ti)
                m = m & keep
            out.append((t, m))
        return out

    def _best(self, i, b, ti):
        return (b >= self.nord[i] - ti) if self.dirs[i] > 0 else (b < ti)


def columns(space, cells_cols):
    """cells_cols: {cell: (key array, w, kind)} -> r (J, K) gold-normalised reach, p (J, K) share of candidate weight, m and e raw, the cell list."""
    names, r, p, mm, ee = [], [], [], [], []
    for c, (key, w, kind) in cells_cols.items():
        nd = kind == 1
        rr = np.bincount(key[nd], weights=w[nd], minlength=space.K)
        m = np.bincount(key[nd], minlength=space.K).astype(np.float64)
        e = np.bincount(key[~nd], weights=w[~nd], minlength=space.K)
        if rr.sum() <= 0:
            continue
        names.append(c)
        r.append(rr / rr.sum())
        p.append((m + e) / (m + e).sum())
        mm.append(m)
        ee.append(e)
    return names, np.array(r), np.array(p), np.array(mm), np.array(ee)


# ── solvers ──────────────────────────────────────────────────────────────────


def _rows(r, p, rho, pairs, size_cap, K):
    J = r.shape[0]
    rows, lo, hi = [], [], []
    rows.append(np.hstack([r, np.zeros((J, 1))]))
    lo += [rho] * J
    hi += [np.inf] * J
    rows.append(np.hstack([p, -np.ones((J, 1))]))
    lo += [-np.inf] * J
    hi += [0.0] * J
    if pairs:
        M = np.zeros((len(pairs), K + 1))
        for i, (k, k2) in enumerate(pairs):
            M[i, k] = 1.0
            M[i, k2] = -1.0
        rows.append(M)
        lo += [-np.inf] * len(pairs)
        hi += [0.0] * len(pairs)
    if size_cap is not None:
        s = np.zeros((1, K + 1))
        s[0, :K] = 1.0
        rows.append(s)
        lo.append(-np.inf)
        hi.append(float(size_cap))
    return np.vstack(rows), np.array(lo), np.array(hi)


def solve_milp(r, p, space, rho, protect, mono, size_cap=None, relax=False, second=True):
    """min tau (second stage: then the fewest keys kept at that tau) over x in {0,1}^K (or [0,1]^K when relax). Returns dict(status, tau, x, gap, bound)."""
    J, K = r.shape
    pairs = space.mono_pairs() if mono else []
    A, lo, hi = _rows(r, p, rho, pairs, size_cap, K)
    lb = np.zeros(K + 1)
    ub = np.ones(K + 1)
    if protect:
        lb[:K][space.grp == 0] = 1.0
    c = np.zeros(K + 1)
    c[K] = 1.0
    integ = np.zeros(K + 1) if relax else np.r_[np.ones(K), 0.0]
    res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integ, bounds=Bounds(lb, ub), options={"time_limit": MILP_LIMIT, "mip_rel_gap": 0.0})
    if res.x is None:
        return {"status": "INFEASIBLE" if res.status == 2 else f"NO_SOLUTION({res.status})", "tau": None, "x": None, "gap": None, "bound": None}
    out = {"status": {0: "OPTIMAL", 1: "NOT_PROVEN"}.get(res.status, f"STATUS_{res.status}"), "tau": float(res.x[K]), "gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
           "bound": float(getattr(res, "mip_dual_bound", res.x[K]) if getattr(res, "mip_dual_bound", None) is not None else res.x[K]), "x": res.x[:K].copy()}
    if relax:
        out["dual_reach"] = None
        return out
    if second and out["status"] == "OPTIMAL":
        hi2 = hi.copy()
        A2 = np.vstack([A, np.r_[np.zeros(K), 1.0][None, :]])
        lo2 = np.r_[lo, -np.inf]
        hi2 = np.r_[hi, out["tau"] + 1e-9]
        c2 = np.r_[np.ones(K), 0.0]
        r2 = milp(c2, constraints=LinearConstraint(A2, lo2, hi2), integrality=integ, bounds=Bounds(lb, ub), options={"time_limit": MILP_LIMIT, "mip_rel_gap": 0.0})
        if r2.x is not None and r2.status == 0:
            out["x"] = np.round(r2.x[:K]).copy()
            out["tau"] = float(max(0.0, ((p @ out["x"]).max())))
            out["tau_stage1"] = float(res.x[K])
    else:
        out["x"] = np.round(out["x"]) if not relax else out["x"]
    return out


def solve_lp(r, p, space, rho, protect, mono=False):
    """The relaxation by linprog (HiGHS) with the duals of the reach rows: (tau, x, duals)."""
    J, K = r.shape
    pairs = space.mono_pairs() if mono else []
    A, lo, hi = _rows(r, p, rho, pairs, None, K)
    Aub, bub = [], []
    for row, l, h in zip(A, lo, hi):
        if np.isfinite(h):
            Aub.append(row)
            bub.append(h)
        if np.isfinite(l):
            Aub.append(-row)
            bub.append(-l)
    bounds = [(1.0 if (protect and space.grp[k] == 0) else 0.0, 1.0) for k in range(K)] + [(0.0, 1.0)]
    c = np.zeros(K + 1)
    c[K] = 1.0
    res = linprog(c, A_ub=np.array(Aub), b_ub=np.array(bub), bounds=bounds, method="highs")
    if res.status != 0:
        return {"status": "INFEASIBLE" if res.status == 2 else f"STATUS_{res.status}", "tau": None, "x": None, "duals": None}
    marg = np.asarray(res.ineqlin.marginals)
    # the reach rows are the rows with a finite lower bound: they sit at odd positions after their upper bound row; read them by index
    idx, duals = 0, {}
    for i, (l, h) in enumerate(zip(lo, hi)):
        if np.isfinite(h):
            idx += 1
        if np.isfinite(l):
            if i < J:
                duals[i] = -float(marg[idx])
            idx += 1
    return {"status": "OPTIMAL", "tau": float(res.x[K]), "x": res.x[:K].copy(), "duals": [duals[i] for i in range(J)]}


def solve_threshold(r, p, space, rho, protect):
    """Exact over the threshold rules: one threshold triple per group R and K (and S when not protected); S kept entire when protected. Returns the
    smallest feasible max price, the keys kept (fewest on ties), the thresholds."""
    J, K = r.shape
    groups = [1, 2] if protect else [0, 1, 2]
    masks = {g: space.threshold_masks(g) for g in groups}
    base = np.zeros(K, bool)
    if protect:
        base = space.grp == 0
    br, bp = r[:, base].sum(1), p[:, base].sum(1)
    rg = {g: np.array([r[:, m].sum(1) for _t, m in masks[g]]) for g in groups}
    pg = {g: np.array([p[:, m].sum(1) for _t, m in masks[g]]) for g in groups}
    ng = {g: np.array([int(m.sum()) for _t, m in masks[g]]) for g in groups}
    best = None
    for combo in itertools.product(*[range(len(masks[g])) for g in groups]):
        rr = br.copy()
        pp = bp.copy()
        nk = int(base.sum())
        for g, ci in zip(groups, combo):
            rr = rr + rg[g][ci]
            pp = pp + pg[g][ci]
            nk += int(ng[g][ci])
        if bool((rr >= rho - 1e-12).all()):
            val = float(pp.max())
            if best is None or val < best[0] - 1e-12 or (abs(val - best[0]) <= 1e-12 and nk < best[1]):
                best = (val, nk, combo)
    if best is None:
        return {"status": "INFEASIBLE", "tau": None, "x": None, "thresholds": None}
    x = base.copy()
    th = {}
    for g, ci in zip(groups, best[2]):
        x = x | masks[g][ci][1]
        th[GRP[g]] = list(masks[g][ci][0])
    return {"status": "OPTIMAL_ON_THRESHOLDS", "tau": float(best[0]), "x": x.astype(float), "thresholds": th, "kept": int(x.sum())}


def solve_frontier(r, p, space, cap, protect, mono):
    """The best reach the worst cell can have at price caps: max z s.t. reach_j >= z, price_j <= cap (MILP)."""
    J, K = r.shape
    pairs = space.mono_pairs() if mono else []
    rows = [np.hstack([r, -np.ones((J, 1))]), np.hstack([p, np.zeros((J, 1))])]
    lo = [np.zeros(J), np.full(J, -np.inf)]
    hi = [np.full(J, np.inf), np.full(J, cap)]
    if pairs:
        M = np.zeros((len(pairs), K + 1))
        for i, (k, k2) in enumerate(pairs):
            M[i, k] = 1.0
            M[i, k2] = -1.0
        rows.append(M)
        lo.append(np.full(len(pairs), -np.inf))
        hi.append(np.zeros(len(pairs)))
    lb, ub = np.zeros(K + 1), np.ones(K + 1)
    if protect:
        lb[:K][space.grp == 0] = 1.0
    c = np.zeros(K + 1)
    c[K] = -1.0
    res = milp(c, constraints=LinearConstraint(np.vstack(rows), np.concatenate(lo), np.concatenate(hi)), integrality=np.r_[np.ones(K), 0.0],
               bounds=Bounds(lb, ub), options={"time_limit": MILP_LIMIT, "mip_rel_gap": 0.0})
    if res.x is None:
        return {"status": "INFEASIBLE", "reach": None}
    return {"status": "OPTIMAL" if res.status == 0 else "NOT_PROVEN", "reach": float(res.x[K])}


def greedy_old(r, p, rho):
    """UL3's order, not exact: keys by pooled precision (needed share of the pooled candidates) until every cell holds rho of its reach."""
    J, K = r.shape
    pr = r.sum(0) / np.maximum(r.sum(0) + p.sum(0), 1e-12)
    order = np.argsort(-pr, kind="stable")
    x = np.zeros(K)
    for k in order:
        if bool(((r @ x) >= rho - 1e-12).all()):
            break
        if r[:, k].sum() + p[:, k].sum() > 0:
            x[k] = 1.0
    ok = bool(((r @ x) >= rho - 1e-12).all())
    return {"status": "FEASIBLE" if ok else "INFEASIBLE", "tau": float((p @ x).max()) if ok else None, "x": x}


def check_solution(r, p, space, x, rho, protect, mono):
    """The program's constraints on a candidate x: reach, S, monotone."""
    ok = bool(((r @ x) >= rho - 2e-6).all())     # HiGHS accepts a primal violation up to its feasibility tolerance
    if protect:
        ok &= bool((x[space.grp == 0] > 0.5).all())
    if mono:
        ok &= all(x[k] <= x[k2] + 1e-9 for k, k2 in space.mono_pairs())
    return ok


# ── the program on the quarter ───────────────────────────────────────────────


def cells_columns(cellnames, part, cuts):
    out = {}
    for key in cellnames:
        d, v = key.split("/")
        T = load_cell(d, v, part, cols=("grp", "kind", "w", "q", "a", "c", "step") + CAND, gold=False)
        out[key] = T
    return out


def cmd_solve(a):
    t0 = time.time()
    sc = json.loads((OUT / f"screen_q{SUFFIX}.json").read_text(encoding="utf-8"))
    cuts = sc["dot_cuts"]
    names = sc["cells_read"]
    tabs = {}
    for key in names:
        d, v = key.split("/")
        tabs[key] = load_cell(d, v, "q", cols=("grp", "kind", "w", "dot", "rank", "votes", "dir", "exact", "ntok", "caps", "lead", "off", "cnt", "qov"), gold=False)
    bins = {k: binned(t, cuts) for k, t in tabs.items()}
    rec = {"part": "q", "declared_in": DECLARED, "screen_sha256": hashlib.sha256((OUT / f"screen_q{SUFFIX}.json").read_bytes()).hexdigest(), "dot_cuts": cuts,
           "cells": names, "spaces": {}, "zgp1p_sha256": script_sha256()}
    spaces = {"K3": sc["K3"], "K4": sc["K4"], "K0": sc["K0"]}
    for sname, attrs in spaces.items():
        dirs = [sc["attrs"][at]["direction"] for at in attrs]
        sp = Space(attrs, dirs)
        cc = {k: (sp.key_of(tabs[k], bins[k]), tabs[k]["w"].astype(np.float64), tabs[k]["kind"]) for k in names}
        cnames, r, p, mm, ee = columns(sp, cc)
        block = {"attrs": attrs, "attr_ids": [AID[x] for x in attrs], "directions": dirs, "keys": sp.K, "cells": cnames}
        main = solve_milp(r, p, sp, RHO, True, True)
        U = solve_milp(r, p, sp, RHO, True, False)
        L = solve_lp(r, p, sp, RHO, True, mono=False)
        T = solve_threshold(r, p, sp, RHO, True)
        chain = [x["tau"] for x in (L, U, main, T)]
        block["chain_tau_L_U_M_T"] = chain
        if None not in chain and not all(chain[i] <= chain[i + 1] + 1e-6 for i in range(3)):
            stop(f"G-G6: {sname}: the chain tau_L <= tau_U <= tau_M <= tau_T fails: {chain}")
        if main["x"] is not None and not check_solution(r, p, sp, main["x"], RHO, True, True):
            stop(f"G-G6: {sname}: the M solution breaks a constraint")
        block["M"] = {"status": main["status"], "tau": main["tau"], "gap": main["gap"], "bound": main["bound"], "kept": int(main["x"].sum()) if main["x"] is not None else None}
        block["U"] = {"status": U["status"], "tau": U["tau"]}
        block["L"] = {"status": L["status"], "tau": L["tau"], "duals_reach_by_cell": dict(zip(cnames, L["duals"])) if L["duals"] else None}
        block["T"] = {"status": T["status"], "tau": T["tau"], "thresholds": T.get("thresholds"), "kept": T.get("kept")}
        gr = greedy_old(r, p, RHO)
        block["greedy_old_form"] = {"status": gr["status"], "tau": gr["tau"], "kept": int(gr["x"].sum())}
        block["seen"] = ((r.sum(0) + p.sum(0)) > 0).astype(int).tolist()
        if main["x"] is not None:
            x = main["x"]
            block["M"]["per_cell_on_Q"] = {c: {"reach": round(float(r[j] @ x), 4), "price": round(float(p[j] @ x), 4)} for j, c in enumerate(cnames)}
            block["M"]["kept_keys"] = [{"key": int(k), "label": sp.label(k), "reach_weight_pooled": round(float(r[:, k].sum()), 4),
                                        "price_weight_pooled": round(float(p[:, k].sum()), 4)} for k in np.flatnonzero(x > 0.5)]
            block["M"]["x"] = x.astype(int).tolist()
        if T["x"] is not None:
            xt = T["x"]
            block["T"]["per_cell_on_Q"] = {c: {"reach": round(float(r[j] @ xt), 4), "price": round(float(p[j] @ xt), 4)} for j, c in enumerate(cnames)}
        # beside, on K3 (the others get the call's program only)
        if sname == "K3":
            beside = {"S_free": {}, "rho": {}, "size": {}, "own": {}, "frontier": {}, "no_monotone_S_free": None}
            sf = solve_milp(r, p, sp, RHO, False, True)
            beside["S_free"] = {"status": sf["status"], "tau": sf["tau"], "cost_of_A1": (main["tau"] - sf["tau"]) if (main["tau"] is not None and sf["tau"] is not None) else None}
            sfl = solve_lp(r, p, sp, RHO, False, mono=False)
            beside["S_free_L"] = {"tau": sfl["tau"]}
            for rho in RHOS:
                sr = solve_milp(r, p, sp, rho, True, True)
                beside["rho"][str(rho)] = {"status": sr["status"], "tau": sr["tau"]}
            for rho in (0.94, 0.96):
                sl = solve_lp(r, p, sp, rho, True, mono=False)
                beside["rho"][f"L_{rho}"] = {"tau": sl["tau"]}
            for B in SIZES:
                sz = solve_milp(r, p, sp, RHO, True, True, size_cap=B)
                beside["size"][str(B)] = {"status": sz["status"], "tau": sz["tau"]}
            own = {}
            for j, c in enumerate(cnames):
                oj = solve_milp(r[j:j + 1], p[j:j + 1], sp, RHO, True, True)
                own[c] = {"status": oj["status"], "tau": oj["tau"]}
            beside["own"] = own
            ts = [v["tau"] for v in own.values() if v["tau"] is not None]
            beside["cost_of_universality"] = (main["tau"] - max(ts)) if (ts and main["tau"] is not None) else None
            for cap in CAPS_P:
                fr = solve_frontier(r, p, sp, cap, True, True)
                beside["frontier"][str(cap)] = fr
            block["beside"] = beside
        block["total_per_cell"] = {c: {"needed_weight": round(float(tabs[c]["w"][tabs[c]["kind"] == 1].sum()), 4),
                                       "candidate_weight": round(float(mm[j].sum() + ee[j].sum()), 2)} for j, c in enumerate(cnames)}
        rec["spaces"][sname] = block
        log(f"{sname}: tau L {chain[0]} U {chain[1]} M {chain[2]} T {chain[3]} ({time.time() - t0:.0f} s)")
    k3, k0 = rec["spaces"]["K3"], rec["spaces"]["K0"]
    rec["improvement_over_K0_tau"] = (k0["M"]["tau"] - k3["M"]["tau"]) if (k0["M"]["tau"] is not None and k3["M"]["tau"] is not None) else None
    rec["rule"] = {"space": "K3", "attrs": k3["attrs"], "directions": k3["directions"], "dot_cuts": cuts, "keep": k3["M"].get("x"),
                   "keys_kept": k3["M"].get("kept_keys"), "kept_keys_count": k3["M"].get("kept"), "seen": k3["seen"], "empty_keys_default": "removed (second stage minimises the keys kept)"}
    rec["rule_K0"] = {"space": "K0", "attrs": k0["attrs"], "directions": k0["directions"], "dot_cuts": cuts, "keep": k0["M"].get("x"), "seen": k0["seen"]}
    rec["seconds"] = round(time.time() - t0, 1)
    wjson(OUT / f"rule_q{SUFFIX}.json", rec)
    log(f"rule record: {rel(OUT / f'rule_q{SUFFIX}.json')}")
    return 0


# ── grading on the rest of dev ───────────────────────────────────────────────


def path_retention(T, keep_row):
    """The share of the reachable missed golds that keep at least one complete shortest path (every link kept), step by step; the golds' unit rows are
    the incidence arrays. Returns (kept golds, golds with units, per-distance (kept, total))."""
    G = T["G"]
    if G["inc_g"].size == 0:
        return 0, 0, {}
    o = np.argsort(G["inc_g"], kind="stable")
    ig, ir = G["inc_g"][o], G["inc_row"][o]
    bounds = np.flatnonzero(np.r_[True, ig[1:] != ig[:-1], True])
    kept = tot = 0
    per = {}
    for s, e in zip(bounds[:-1], bounds[1:]):
        g = int(ig[s])
        rows = ir[s:e]
        d = int(G["gdist"][g])
        st, aa, cc, kp = T["step"][rows], T["a"][rows], T["c"][rows], keep_row[rows]
        ok = False
        c1 = set(cc[(st == 1) & kp].tolist())
        if d == 1:
            ok = bool(c1)
        else:
            m2 = (st == 2) & kp & np.isin(aa, list(c1)) if c1 else np.zeros(st.size, bool)
            c2 = set(cc[m2].tolist())
            if d == 2:
                ok = bool(c2)
            elif c2:
                ok = bool(((st == 3) & kp & np.isin(aa, list(c2))).any())
        tot += 1
        kept += int(ok)
        k_, t_ = per.get(d, (0, 0))
        per[d] = (k_ + int(ok), t_ + 1)
    return kept, tot, per


def grade_cell_rule(T, cuts, space, x, seen=None):
    B = binned(T, cuts)
    key = space.key_of(T, B)
    keep = x[key] > 0.5
    nd = T["kind"] == 1
    reach_den = float(T["w"][nd].sum())
    cand_den = float(nd.sum() + T["w"][~nd].sum())
    reach = float(T["w"][nd & keep].sum()) / reach_den if reach_den > 0 else float("nan")
    price = float(((nd & keep).sum() + T["w"][(~nd) & keep].sum()) / cand_den) if cand_den > 0 else float("nan")
    k_, t_, per = path_retention(T, keep)
    return {"reach": round(reach, 4), "price": round(price, 4), "path_retention": round(k_ / t_, 4) if t_ else float("nan"), "golds_with_units": t_,
            "path_by_distance": {str(d): [int(a_), int(b_)] for d, (a_, b_) in per.items()}, "needed_weight": round(reach_den, 4),
            "needed_weight_in_keys_unseen_on_Q": (round(float(T["w"][nd & (np.asarray(seen)[key] == 0)].sum()) / reach_den, 4) if seen is not None and reach_den > 0 else None)}


def cmd_grade(a):
    t0 = time.time()
    rule = json.loads((OUT / f"rule_q{SUFFIX}.json").read_text(encoding="utf-8"))
    sc = json.loads((OUT / f"screen_q{SUFFIX}.json").read_text(encoding="utf-8"))
    cuts = rule["dot_cuts"]
    res = {"part": "r", "declared_in": DECLARED, "rule_sha256": hashlib.sha256((OUT / f"rule_q{SUFFIX}.json").read_bytes()).hexdigest(), "cells": {}, "zgp1p_sha256": script_sha256()}
    for which in ("rule", "rule_K0"):
        rr = rule[which]
        if rr.get("keep") is None:
            res[which] = {"attrs": rr["attrs"], "cells": {}, "cells_read": 0, "screen_in": 0, "call": "NO_RULE_INFEASIBLE_ON_Q"}
            continue
        sp = Space(rr["attrs"], rr["directions"])
        x = np.array(rr["keep"], float)
        res[which] = {"attrs": rr["attrs"], "cells": {}}
        for d, v in CELLS:
            T = load_cell(d, v, "q" if SUFFIX else "r", cols=("grp", "kind", "w", "q", "a", "c", "step", "dot", "rank", "votes", "dir", "exact", "ntok", "caps", "lead", "off", "cnt", "qov"))
            key = f"{d}/{v}"
            if T is None:
                continue
            if "not_read" in T:
                res[which]["cells"][key] = {"status": "NOT_READ", "why": T["not_read"]}
                continue
            if T["kind"].size == 0 or float(T["w"][T["kind"] == 1].sum()) <= 0:
                res[which]["cells"][key] = {"status": "NO_REACHABLE_GOLD"}
                continue
            g = grade_cell_rule(T, cuts, sp, x, rr.get("seen"))
            g["status"] = "read"
            g["SCREEN_IN"] = bool(g["reach"] >= SCREEN_REACH and g["price"] <= SCREEN_PRICE and g["path_retention"] >= SCREEN_PATH)
            res[which]["cells"][key] = g
            log(f"{which} {key}: reach {g['reach']} price {g['price']} path {g['path_retention']} SCREEN_IN {g['SCREEN_IN']} ({time.time() - t0:.0f} s)")
        read = [c for c in res[which]["cells"].values() if c["status"] == "read"]
        res[which]["cells_read"] = len(read)
        res[which]["screen_in"] = int(sum(c["SCREEN_IN"] for c in read))
        res[which]["call"] = "UNIVERSAL_RULE" if (read and all(c["SCREEN_IN"] for c in read)) else "NOT_UNIVERSAL"
    wjson(OUT / f"grade_r{SUFFIX}.json", res)
    log(f"grade: rule {res['rule']['call']} ({res['rule']['screen_in']}/{res['rule']['cells_read']}); K0 {res['rule_K0']['call']}")
    return 0


# ── report ───────────────────────────────────────────────────────────────────


def cmd_report(a):
    sc = json.loads((OUT / f"screen_q{SUFFIX}.json").read_text(encoding="utf-8")) if (OUT / f"screen_q{SUFFIX}.json").is_file() else None
    ru = json.loads((OUT / f"rule_q{SUFFIX}.json").read_text(encoding="utf-8")) if (OUT / f"rule_q{SUFFIX}.json").is_file() else None
    gr = json.loads((OUT / f"grade_r{SUFFIX}.json").read_text(encoding="utf-8")) if (OUT / f"grade_r{SUFFIX}.json").is_file() else None
    L = []
    if sc:
        L.append(f"### Screen (quarter): {len(sc['cells_read'])} cells read, not read: {sc['cells_not_read']}")
        L.append(f"dot cuts {sc['dot_cuts']}; votes guard {sc['votes_guard']} (dropped: {sc['votes_dropped']})")
        L.append("")
        L.append("| attribute | call | score | direction | pooled AUC | " + " | ".join(sc["cells_read"]) + " |")
        L.append("|---|---|---|---|---|" + "---|" * len(sc["cells_read"]))
        for at in CAND:
            x = sc["attrs"][at]
            cs = []
            for k in sc["cells_read"]:
                r = x["per_cell"].get(k)
                cs.append("na" if r is None else f"{r['auc']:.3f} [{r['lo']:.3f},{r['hi']:.3f}]")
            L.append(f"| {x['id']} {at} | {x['call']} | {x['score']:.3f} | {x['direction']:+d} | {x['pooled_auc']:.3f} | " + " | ".join(cs) + " |")
        L.append(f"\nK3 = {sc['K3']}, K4 = {sc['K4']}, ranking {sc['ranking']}")
        for k in sc["cells_read"]:
            c = sc["cells"][k]
            L.append(f"\n**{k}**: gold distance (found, 1, 2, 3, unreached) {c['gold_distance_found_1_2_3_unreached']}; needed units {c['needed_units']} (weight {c['needed_weight']}), "
                     f"by step {c['needed_by_step']}, by group S/R/K {c['needed_by_grp']}; controls {c['controls_sampled']} sampled")
            for at in CAND:
                t = c["tables"][at]
                L.append(f"  - {at}: needed " + " / ".join(f"{b} {s:.2f}" for b, s in zip(t["bins"], t["needed_share"])) + "; control " + " / ".join(f"{s:.2f}" for s in t["control_share"]))
    if ru:
        L.append("\n### Program (quarter)")
        for s in ("K3", "K0", "K4"):
            b = ru["spaces"][s]
            L.append(f"- {s} {b['attrs']}: tau L {b['L']['tau']} <= U {b['U']['tau']} <= M {b['M']['tau']} ({b['M']['status']}, kept {b['M']['kept']}) <= T {b['T']['tau']}; greedy (old form) {b['greedy_old_form']['tau']}")
        be = ru["spaces"]["K3"].get("beside", {})
        L.append(f"- S free: {be.get('S_free')}; rho: {be.get('rho')}; size: {be.get('size')}; cost of universality {be.get('cost_of_universality')}")
        L.append(f"- own optima: {be.get('own')}")
        L.append(f"- frontier: {be.get('frontier')}")
        L.append(f"- improvement over K0 (tau points): {ru['improvement_over_K0_tau']}")
    if gr:
        L.append("\n### Grade (rest of dev)")
        for w in ("rule", "rule_K0"):
            L.append(f"- {w}: {gr[w]['call']} ({gr[w]['screen_in']}/{gr[w]['cells_read']} cells SCREEN_IN)")
            for k, c in gr[w]["cells"].items():
                if c["status"] == "read":
                    L.append(f"  - {k}: reach {c['reach']}, price {c['price']}, path retention {c['path_retention']} ({c['golds_with_units']} golds) -> {c['SCREEN_IN']}")
    txt = "\n".join(L)
    (OUT / f"report{SUFFIX}.md").write_text(txt, encoding="utf-8")
    print(txt)
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def brute_auc(q, a, step, kind, w, b):
    """Reference: strata by explicit loops."""
    strata = {}
    for i in range(q.size):
        strata.setdefault((int(q[i]), int(a[i]), int(step[i])), []).append(i)
    A, Bw = {}, {}
    for (qq, _a, _s), idx in strata.items():
        nd = [i for i in idx if kind[i] == 1]
        cd = [i for i in idx if kind[i] == 0]
        if not nd or not cd:
            continue
        num = den = 0.0
        for i in nd:
            for j in cd:
                s = 1.0 if b[i] > b[j] else (0.5 if b[i] == b[j] else 0.0)
                num += w[i] * w[j] * s
                den += w[i] * w[j]
        om = sum(w[i] for i in nd)
        A[qq] = A.get(qq, 0.0) + om * num / den
        Bw[qq] = Bw.get(qq, 0.0) + om
    return A, Bw


def toy_program(rng, K=8, J=3):
    r = rng.random((J, K)) ** 2
    r = r / r.sum(1, keepdims=True)
    p = rng.random((J, K)) + 0.05
    p = p / p.sum(1, keepdims=True)
    return r, p


def brute_min_tau(r, p, space, rho, protect, mono):
    K = r.shape[1]
    best = None
    pairs = space.mono_pairs() if mono else []
    for bits in itertools.product((0.0, 1.0), repeat=K):
        x = np.array(bits)
        if protect and not bool((x[space.grp == 0] > 0.5).all()):
            continue
        if not bool(((r @ x) >= rho - 1e-9).all()):
            continue
        if any(x[k] > x[k2] + 1e-9 for k, k2 in pairs):
            continue
        t = float((p @ x).max())
        if best is None or t < best:
            best = t
    return best


def selftest():
    rng = np.random.default_rng(1)
    # AUC strata equal an explicit loop
    for trial in range(30):
        m = int(rng.integers(20, 80))
        q = rng.integers(0, 4, m)
        a = rng.integers(0, 3, m)
        step = rng.integers(1, 4, m)
        kind = rng.integers(0, 2, m)
        w = rng.random(m) + 0.1
        b = rng.integers(0, 3, m)
        A_, B_ = strata_parts(q, a, step, kind, w, b, 3, 4)
        RA, RB = brute_auc(q, a, step, kind, w, b)
        for qq in range(4):
            assert abs(A_[qq] - RA.get(qq, 0.0)) < 1e-9 and abs(B_[qq] - RB.get(qq, 0.0)) < 1e-9, (trial, qq)
    # the quantile and the bins
    assert wquantiles(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), np.ones(6), (1 / 3, 2 / 3)) == [2.0, 4.0]
    T = {"dot": np.array([0.1, 0.5, 0.9]), "rank": np.array([1, 4, 31]), "votes": np.array([1, 2, 9]), "dir": np.array([0, 1, 3]), "exact": np.array([0, 2, 3]),
         "ntok": np.array([1, 2, 3]), "caps": np.array([0, 1, 2]), "lead": np.array([0, 1, 2]), "off": np.array([np.nan, 0.05, 0.7], np.float32),
         "cnt": np.array([0, 1, 5]), "qov": np.array([0, 1, 4])}
    B = binned(T, [0.3, 0.7])
    assert B["dot"].tolist() == [0, 1, 2] and B["rank"].tolist() == [0, 1, 2] and B["votes"].tolist() == [0, 1, 2] and B["dir"].tolist() == [0, 1, 2]
    assert B["off"].tolist() == [3, 0, 2] and B["cnt"].tolist() == [3, 0, 2] and B["lead"].tolist() == [0, 1, 2] and B["qov"].tolist() == [0, 1, 2]
    # solvers: the integer program equals all-subsets brute force (with and without the monotone constraint, S protected or free); the chain; a wrong objective is caught
    sp = Space(["dot"], [1])          # grp x dot: 9 keys
    for trial in range(40):
        r, p = toy_program(rng, sp.K, 3)
        for protect in (True, False):
            for mono in (True, False):
                for rho in (0.7, 0.9):
                    bf = brute_min_tau(r, p, sp, rho, protect, mono)
                    m = solve_milp(r, p, sp, rho, protect, mono, second=False)
                    if bf is None:
                        assert m["tau"] is None or m["status"] != "OPTIMAL", (trial, protect, mono, rho)
                    else:
                        assert m["status"] == "OPTIMAL" and abs(m["tau"] - bf) < 1e-7, (trial, protect, mono, rho, m["tau"], bf)
                        m2 = solve_milp(r, p, sp, rho, protect, mono)
                        assert abs(m2["tau"] - bf) < 1e-7 and check_solution(r, p, sp, m2["x"], rho, protect, mono)
        L = solve_lp(r, p, sp, 0.9, True, mono=False)
        U = solve_milp(r, p, sp, 0.9, True, False, second=False)
        M = solve_milp(r, p, sp, 0.9, True, True, second=False)
        T_ = solve_threshold(r, p, sp, 0.9, True)
        if None not in (L["tau"], U["tau"], M["tau"], T_["tau"]):
            assert L["tau"] <= U["tau"] + 1e-7 <= M["tau"] + 2e-7 <= T_["tau"] + 3e-7, (L["tau"], U["tau"], M["tau"], T_["tau"])
            # the threshold rules are exactly the brute-force best over the threshold space
            best = None
            ms = {g: sp.threshold_masks(g) for g in (1, 2)}
            for (t1, m1), (t2, m2) in itertools.product(ms[1], ms[2]):
                x = (sp.grp == 0) | m1 | m2
                x = x.astype(float)
                if bool(((r @ x) >= 0.9 - 1e-9).all()):
                    v = float((p @ x).max())
                    best = v if best is None or v < best else best
            assert best is not None and abs(best - T_["tau"]) < 1e-7, (best, T_["tau"])
    # a deliberately wrong objective (maximise tau) is caught by the same brute-force check
    r, p = toy_program(rng, sp.K, 3)
    bf = brute_min_tau(r, p, sp, 0.7, True, True)
    A, lo, hi = _rows(r, p, 0.7, sp.mono_pairs(), None, sp.K)
    lb = np.zeros(sp.K + 1)
    lb[:sp.K][sp.grp == 0] = 1.0
    wrong = milp(-np.r_[np.zeros(sp.K), 1.0], constraints=LinearConstraint(A, lo, hi), integrality=np.r_[np.ones(sp.K), 0.0], bounds=Bounds(lb, np.ones(sp.K + 1)))
    assert wrong.x is not None and abs(wrong.x[sp.K] - bf) > 1e-6
    # the frontier: at a cap equal to the program's tau the best worst-cell reach is at least rho
    r, p = toy_program(rng, sp.K, 3)
    m = solve_milp(r, p, sp, 0.8, True, True, second=False)
    if m["tau"] is not None:
        fr = solve_frontier(r, p, sp, m["tau"] + 1e-9, True, True)
        assert fr["reach"] is not None and fr["reach"] >= 0.8 - 1e-7
    # path retention on a toy: gold 0 (distance 2) keeps its path only when both links are kept; gold 1 (distance 1) when its link is
    T = {"G": {"inc_g": np.array([0, 0, 0, 1]), "inc_row": np.array([0, 1, 2, 3]), "gdist": np.array([2, 1])}, "step": np.array([1, 1, 2, 1]),
         "a": np.array([10, 11, 5, 12]), "c": np.array([5, 6, 99, 98])}
    assert path_retention(T, np.array([True, False, True, True]))[:2] == (2, 2)
    assert path_retention(T, np.array([False, True, True, True]))[:2] == (1, 2)       # the step-2 link starts at 5, only the first step-1 link ends there
    assert path_retention(T, np.array([True, True, False, False]))[:2] == (0, 2)
    # votes guard and calls
    pc = {"x/RK": {"auc": 0.8, "lo": 0.7, "hi": 0.9}, "y/RK": {"auc": 0.75, "lo": 0.7, "hi": 0.8}}
    assert attr_calls(pc)[0] == "SEPARATES"
    pc = {"x/RK": {"auc": 0.8, "lo": 0.7, "hi": 0.9}, "y/RK": {"auc": 0.3, "lo": 0.2, "hi": 0.4}}
    assert attr_calls(pc)[0] == "SIGN_SPLIT"
    pc = {"x/RK": {"auc": 0.56, "lo": 0.51, "hi": 0.6}, "y/RK": {"auc": 0.55, "lo": 0.5, "hi": 0.6}}
    assert attr_calls(pc)[0] == "WEAK"
    log("selftest PASS: strata AUC equals the explicit loop; bins; the integer program equals all-subsets brute force (S protected or free, with and without "
        "the monotone constraint); the threshold enumeration equals its brute force; the chain L <= U <= M <= T; a wrong objective is caught; the frontier; "
        "path retention; the calls")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("selftest", "screen", "solve", "grade", "report"))
    ap.add_argument("--smoke", action="store_true", help="read the 5-question smoke files (code paths only)")
    a = ap.parse_args(argv)
    global SUFFIX
    SUFFIX = "_limit" if a.smoke else ""
    return {"selftest": lambda a: selftest(), "screen": cmd_screen, "solve": cmd_solve, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
