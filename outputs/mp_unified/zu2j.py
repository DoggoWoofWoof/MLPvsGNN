"""U2j (docs/U2J_LINK_PRECISION.md): link precision, step 1 of the hyperlink-gap plan. Laptop only; nothing trains.

    python outputs/mp_unified/zu2j.py --selftest
    python outputs/mp_unified/zu2j.py grade --dataset D [--carve C]   -> outputs/u2j/<D>/grade_<C>.json
    python outputs/mp_unified/zu2j.py beside --dataset D               -> outputs/u2j/<D>/beside.json
    python outputs/mp_unified/zu2j.py choose                           -> outputs/u2j/choice.json
    python outputs/mp_unified/zu2j.py report                           -> outputs/u2j/report.md

W2_S's links stay as they are. Each order R1..R4 sorts the rows the frozen I reads by rule-based evidence of each link,
ties to node id; the frozen I then takes its first 25 per seed from the reordered rows. The evidence is U2i's pass's
(the sentence of a link's first mention, its offset, its count); the links rebuilt from U2i's hits must equal U2i's
links.npz exactly, or nothing is graded. No question, gold or hyperlink enters an order.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402
import zu2i  # noqa: E402

DECLARED = "docs/U2J_LINK_PRECISION.md"
OUT = ROOT / "outputs" / "u2j"
U2I = zu2i.OUT
U2B = zu2i.U2B
ORDERS = ("R1", "R2", "R3", "R4")          # declared order
SCORED = ("musique", "hotpotqa", "2wiki")  # the choice's datasets (squad has no I)
TIE = 0.002
EPS = 1e-9
BRIDGE_R = (5, 25)
FEATURES = ("lead", "pos", "count", "rare", "caps", "capmid", "tokens", "alias")
BLOCK = 8_000_000


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def codes():
    return {"zu2j_sha256": u1b.sha_src(__file__), "zu2i_sha256": u1b.sha_src(zu2i.__file__),
            "zu2b_sha256": u1b.sha_src(zu2b.__file__), "u1b_sha256": u1b.sha_src(u1b.__file__)}


def rd(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ── the evidence of a link (section 2) ──────────────────────────────────────


def _expand(g_ptr, g_nodes, src, grp):
    """zu2i.expand with each pair's position in the input: (s, t, j)."""
    cnt = (g_ptr[grp + 1] - g_ptr[grp]).astype(np.int64)
    j = np.repeat(np.arange(src.size, dtype=np.int64), cnt)
    s = src.astype(np.int64)[j]
    starts = g_ptr[grp][j]
    off = np.arange(cnt.sum(), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    t = g_nodes[starts + off].astype(np.int64)
    keep = s != t
    return s[keep], t[keep], j[keep]


def _chunk_links(src, k, E, n):
    """One chunk's W2 links (zu2i.w2_links's rule) as sorted unique keys src * n + dst, with the hit behind each;
    and how many repeated keys were dropped (a repeat is the same surface matched twice: the same evidence)."""
    m = E.is_title[k] & ~E.common[E.lowid[k]]
    idx = np.flatnonzero(m)
    s, t, j = _expand(E.g_ptr, E.g_nodes, src[idx], E.tgroup[k[idx]])
    key = s * n + t
    o = np.argsort(key, kind="stable")
    key, j = key[o], idx[j[o]]
    u = np.ones(key.size, bool)
    u[1:] = key[1:] != key[:-1]
    return key[u], j[u], int((~u).sum())


def evidence_tables(d):
    """U2i's evidence for d, every file checked against pass.json; REFUSED unless U2i's gate 1 passed."""
    pas = rd(U2I / d / "pass.json")
    g1 = rd(U2I / d / "gate1.json")
    if g1["status"] != "PASS" or pas["gate1"] != "PASS":
        raise SystemExit(f"{d}: U2i's gate 1 is {g1['status']}; REFUSED")
    for f, k in (("links.npz", "links_npz_sha256"), ("evidence.npz", "evidence_npz_sha256")):
        if zu2i.sha_file(U2I / d / f) != pas[k]:
            raise SystemExit(f"{d}: {f} is not the pass's")
    with np.load(U2I / d / "evidence.npz") as z:
        ev = {k: z[k] for k in z.files}
    group = ev["group"]
    order = np.argsort(group, kind="stable")
    E = SimpleNamespace(is_title=ev["surf_is_title"], lowid=ev["surf_lowid"], tgroup=ev["surf_tgroup"],
                        common=ev["low_low"] > ev["low_cap"], cap=ev["low_cap"], low=ev["low_low"], group=group,
                        text_len=ev["text_len"], g_nodes=order,
                        g_ptr=np.searchsorted(group[order], np.arange(int(group.max()) + 2)))
    return pas, E


def link_evidence(d, fields):
    """Per directed W2 link, in links.npz's order (by source, then target): the named fields of the hit that made it,
    from the pass's chunk files (each checked by sha256). The rebuilt links must equal links.npz's, else REFUSED."""
    pas, E = evidence_tables(d)
    n = E.group.size
    with np.load(U2I / d / "links.npz") as z:
        ws, wt = z["w2_src"], z["w2_dst"]
    out = {f: None for f in fields}
    filled = dup = 0
    for name in sorted(pas["hit_chunks"]):
        p = U2I / d / "work" / name
        if zu2i.sha_file(p) != pas["hit_chunks"][name]:
            raise SystemExit(f"{p}: not the pass's chunk; REFUSED")
        with np.load(p) as z:
            if str(z["table_sha"]) != pas["table_sha256"]:
                raise SystemExit(f"{p}: another table; REFUSED")
            key, j, dd = _chunk_links(z["src"], z["k"], E, n)
            lo, hi = filled, filled + key.size
            if hi > ws.size or not np.array_equal(key, ws[lo:hi].astype(np.int64) * n + wt[lo:hi]):
                raise SystemExit(f"{d}: {name}'s links are not links.npz's; REFUSED")
            for f in fields:
                v = z[f][j]
                if out[f] is None:
                    out[f] = np.empty(ws.size, v.dtype)
                out[f][lo:hi] = v
            filled, dup = hi, dup + dd
    if filled != ws.size:
        raise SystemExit(f"{d}: {filled:,} links rebuilt, links.npz holds {ws.size:,}; REFUSED")
    return ws, wt, out, E, {"duplicate_hits": dup, "w2_links": int(ws.size)}


def _member(keys, q):
    if keys.size == 0:
        return np.zeros(q.size, bool), np.zeros(q.size, np.int64)
    p = np.minimum(np.searchsorted(keys, q), keys.size - 1)
    return keys[p] == q, p


def row_features(ip, col, rows, w2key, lead, sibkey, indeg, n):
    """Every entry of the given rows (ascending): its position in col, row rank, neighbour, class, lead, in-degree."""
    deg = (ip[rows + 1] - ip[rows]).astype(np.int64)
    r = np.repeat(np.arange(rows.size, dtype=np.int64), deg)
    pos = np.repeat(ip[rows], deg) + (np.arange(int(deg.sum()), dtype=np.int64) - np.repeat(np.cumsum(deg) - deg, deg))
    u = rows[r]
    v = col[pos].astype(np.int64)
    cls = np.empty(v.size, np.int8)
    ld = np.empty(v.size, bool)
    for a in range(0, v.size, BLOCK):
        b = slice(a, a + BLOCK)
        out, po = _member(w2key, u[b] * n + v[b])
        inn, pi = _member(w2key, v[b] * n + u[b])
        sib, _ = _member(sibkey, u[b] * n + v[b])
        if not (out | inn | sib).all():
            raise SystemExit("a W2_S neighbour that is neither a mention link nor a sibling; REFUSED")
        cls[b] = np.where(out & inn, 0, np.where(out, 1, np.where(inn, 2, 3)))
        ld[b] = (out & lead[po]) | (inn & lead[pi])
    return pos, r, v, cls, ld, indeg[v]


# ── the orders (section 3) ──────────────────────────────────────────────────


def order_key(name, cls, ld, rare):
    """The within-row key of an order (smaller first); ties go to node id. A mutual pair is lead if either mention is."""
    nl = (~ld).astype(np.int64)
    c = cls.astype(np.int64)
    if name == "R0":
        return np.zeros(cls.size, np.int64)
    if name == "R1":
        return c
    if name == "R2":
        return c * 2 + nl
    if name == "R3":
        return rare.astype(np.int64)
    if name == "R4":
        return ((c * 2 + nl) << 32) | rare.astype(np.int64)
    raise KeyError(name)


def reorder(col, pos, r, v, key):
    """col with the given rows' entries sorted by (row, key, node id); other rows unchanged."""
    o = np.lexsort((v, key, r))
    new = col.copy()
    new[pos] = v[o].astype(col.dtype)
    return new


# ── the grading (section 4) ─────────────────────────────────────────────────


def frozen_I(st, others, fams, seeds, base, golds, setting):
    """zu2i.cmd_grade's loop: the frozen construction over the given structural graph."""
    import m3b_pools
    fam_objs = [st if f == "structural" else others[f] for f in fams]
    al = rc = size = 0.0
    for j in range(len(golds)):
        sd = np.asarray(seeds[j], np.int64)
        exp = m3b_pools.expand_hops(sd, fam_objs, setting)
        pool = np.union1d(np.union1d(np.asarray(base[j], np.int64), exp), sd)
        hit = np.isin(golds[j], pool)
        al += hit.all()
        rc += hit.mean()
        size += pool.size
    nq = len(golds)
    return {"ALL": round(al / nq, 4), "recall": round(rc / nq, 4), "pool": round(size / nq, 1)}


def bridge(ip, col, B, golds, rs=BRIDGE_R):
    """Over the (question, gold) pairs whose gold is outside B: the share within the first r of a gold-in-B's row."""
    hit = dict.fromkeys(rs, 0)
    pairs = with_in = 0
    for b, g in zip(B, golds):
        g = np.asarray(g, np.int64)
        inb = np.isin(g, b)
        gin, gout = g[inb], g[~inb]
        pairs += gout.size
        if not gout.size or not gin.size:
            continue
        with_in += gout.size
        for r in rs:
            near = np.concatenate([col[ip[h]:min(ip[h] + r, ip[h + 1])] for h in gin]).astype(np.int64)
            hit[r] += int(np.isin(gout, near).sum())
    return {"pairs": pairs, "pairs_with_gold_in_B": with_in,
            **{f"at{r}": (round(hit[r] / pairs, 4) if pairs else None) for r in rs}}


def cmd_grade(a):
    d, carve = a.dataset, a.carve
    t0 = time.time()
    qrec = rd(U2B / d / "questions.json")
    qp = U2B / d / f"questions_{carve}.npz"
    if zu2i.sha_file(qp) != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"{d}/{carve}: the questions file is not its record's")
    q = np.load(qp)
    base = zu2b.unragged(q["base_ptr"], q["base_col"])
    seeds = zu2b.unragged(q["seeds_ptr"], q["seeds_col"])
    golds = zu2b.unragged(q["gold_ptr"], q["gold_col"])
    con = qrec["construction"]
    fams = [str(f) for f in q["fam_order"]]
    others = {f: SimpleNamespace(indptr=q[f"{f}_indptr"], col=q[f"{f}_col"]) for f in fams if f != "structural"}
    B = [np.union1d(b, s) for b, s in zip(base, seeds)]
    ws, wt, ev, E, ecount = link_evidence(d, ("sent",))
    n = E.group.size
    lead = ev["sent"] == 0
    indeg = np.bincount(wt, minlength=n).astype(np.int64)
    w2key = ws.astype(np.int64) * n + wt
    del ws, wt, ev
    with np.load(U2I / d / "links.npz") as z:
        ss, sd_ = z["sib_src"].astype(np.int64), z["sib_dst"].astype(np.int64)
    sibkey = np.unique(np.concatenate([ss * n + sd_, sd_ * n + ss]))
    G = zu2i.graphs_of(d, ["W2_S", "PKG"])
    ip, col = G["W2_S"]
    if ip.size - 1 != n:
        raise SystemExit(f"{d}: W2_S has {ip.size - 1:,} nodes, the evidence {n:,}")
    hops = int(con["setting"]["hops"]) if con["setting"] else 0
    if hops >= 2:
        rows = np.arange(n, dtype=np.int64)
    else:
        gin = [np.asarray(g, np.int64)[np.isin(g, b)] for g, b in zip(golds, B)]
        rows = np.unique(np.concatenate([np.concatenate(seeds).astype(np.int64)] + gin))
    pos, r, v, cls, ld, rare = row_features(ip, col, rows, w2key, lead, sibkey, indeg, n)
    del w2key, sibkey
    t1 = time.time()
    log(f"{d}: {rows.size:,} rows, {v.size:,} entries reordered ({t1 - t0:.0f} s)")
    res = {}
    for name in ("R0",) + ORDERS:
        new = reorder(col, pos, r, v, order_key(name, cls, ld, rare))
        if name == "R0" and not np.array_equal(new, col):
            raise SystemExit(f"{d}: W2_S's rows are not in node-id order; REFUSED")
        ent = {"I": None}
        if con["regime"] != "RETRIEVAL":
            ent["I"] = frozen_I(SimpleNamespace(indptr=ip, col=new), others, fams, seeds, base, golds, con["setting"])
        ent["bridge"] = bridge(ip, new, B, golds)
        res[name] = ent
        log(f"{d}: {name} {ent}")
        del new
    pi, pc = G["PKG"]
    pk = {"I": None if con["regime"] == "RETRIEVAL" else
          frozen_I(SimpleNamespace(indptr=pi, col=pc), others, fams, seeds, base, golds, con["setting"]),
          "bridge": bridge(pi, pc, B, golds)}
    gap = {}
    for name in ORDERS:
        w, m = res["R0"], res[name]
        gap[name] = {"I_ALL": zu2i.share(m["I"]["ALL"], w["I"]["ALL"], pk["I"]["ALL"]) if m["I"] else None,
                     "bridge25": (zu2i.share(m["bridge"]["at25"], w["bridge"]["at25"], pk["bridge"]["at25"])
                                  if m["bridge"]["at25"] is not None else None)}
    cl = np.bincount(cls, minlength=4)
    out = {"dataset": d, "carve": carve, "declared_in": DECLARED, "questions": len(golds),
           "questions_npz_sha256": qrec["carves"][carve]["npz_sha256"], "construction": con,
           "rows_reordered": int(rows.size), "entries_reordered": int(v.size),
           "entry_classes": {"mutual": int(cl[0]), "out": int(cl[1]), "in": int(cl[2]), "sibling": int(cl[3])},
           "lead_entries": int(ld.sum()), **ecount, "w2s_mean_degree": round(col.size / n, 2),
           "w2s_max_degree": int(np.diff(ip).max()), "orders": res, "PKG": pk, "gap_share": gap,
           "secs_features": round(t1 - t0, 1), "secs_orders": round(time.time() - t1, 1), **codes()}
    zu2b.write_json(OUT / d / f"grade_{carve}.json", out)
    log(f"{d}/{carve}: graded ({time.time() - t0:.0f} s)")
    return 0


# ── beside (section 5) ──────────────────────────────────────────────────────


def auc(x, y):
    """P(a positive's x > a negative's x) plus half the ties, by mid-ranks; None if a class or x is constant."""
    npos = int(y.sum())
    nneg = int(y.size - npos)
    if not npos or not nneg:
        return None
    uq, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    if uq.size < 2:
        return None
    mid = np.cumsum(cnt) - (cnt - 1) / 2.0
    rpos = float(mid[inv[y]].sum())
    return round((rpos - npos * (npos + 1) / 2.0) / (npos * nneg), 4)


def cmd_beside(a):
    d = a.dataset
    t0 = time.time()
    ws, wt, ev, E, ecount = link_evidence(d, ("sent", "first", "nocc", "capmid", "k"))
    n = E.group.size
    pi, pc = zu2i.graphs_of(d, ["PKG"])["PKG"]
    pkey = np.repeat(np.arange(n, dtype=np.int64), np.diff(pi)) * n + pc
    if pkey.size > 1 and not (np.diff(pkey) > 0).all():
        raise SystemExit(f"{d}: PKG's rows are not sorted")
    y = np.empty(ws.size, bool)
    for s in range(0, ws.size, BLOCK):
        b = slice(s, s + BLOCK)
        y[b] = _member(pkey, ws[b].astype(np.int64) * n + wt[b])[0]
    del pkey, pc
    served, _f = u1b.package_root(False)
    ntok = np.zeros(n, np.int16)
    for i, title, _x in zu2i.stream(served, d):
        ntok[i] = len(zu2b.norm_title(title, zu2i.RULE).strip().split())
    indeg = np.bincount(wt, minlength=n)
    def caps():
        lid = E.lowid[ev["k"]]
        return (E.cap[lid] / (E.cap[lid] + E.low[lid] + 1.0)).astype(np.float32)

    make = {"lead": lambda: ev["sent"] == 0,
            "pos": lambda: (ev["first"] / np.maximum(E.text_len[ws], 1)).astype(np.float32),
            "count": lambda: ev["nocc"],
            "rare": lambda: -indeg[wt],
            "caps": caps,
            "capmid": lambda: ev["capmid"],
            "tokens": lambda: ntok[wt],
            "alias": lambda: ~E.is_title[ev["k"]]}
    res = {}
    for f in FEATURES:
        x = make[f]()
        e = {"auc": auc(x, y)}
        if x.dtype == bool:
            e["hyperlinked_when_1"] = round(float(y[x].mean()), 4) if x.any() else None
            e["hyperlinked_when_0"] = round(float(y[~x].mean()), 4) if (~x).any() else None
            e["share_1"] = round(float(x.mean()), 4)
        res[f] = e
        log(f"{d}: {f} {e}")
        del x
    out = {"dataset": d, "declared_in": DECLARED, "decides": "nothing (section 5)", **ecount,
           "hyperlinked_share": round(float(y.mean()), 4), "features": res, "secs": round(time.time() - t0, 1),
           **codes()}
    zu2b.write_json(OUT / d / "beside.json", out)
    return 0


# ── the choice (section 6) and the report ───────────────────────────────────


def choose(grades):
    """grades: {dataset: grade record or None}; a refusal or a missing grade arrives as None."""
    missing = [d for d in SCORED if grades.get(d) is None
               or any(grades[d]["orders"].get(o, {}).get("I") is None for o in ("R0",) + ORDERS)]
    if missing:
        return {"outcome": "REFUSED", "missing": missing}
    score = {o: round(float(np.mean([grades[d]["orders"][o]["I"]["ALL"] for d in SCORED])), 4)
             for o in ("R0",) + ORDERS}
    top = max(score[o] for o in ORDERS)
    best = next(o for o in ORDERS if score[o] >= top - TIE - EPS)
    falls = [f"{d} {grades[d]['orders'][best]['I']['ALL']} < {grades[d]['orders']['R0']['I']['ALL']}"
             for d in SCORED
             if grades[d]["orders"][best]["I"]["ALL"] < grades[d]["orders"]["R0"]["I"]["ALL"] - TIE - EPS]
    ok = score[best] >= score["R0"] + TIE - EPS and not falls
    return {"outcome": "ADOPT_FOR_POOLS" if ok else "NO_GAIN", "best": best, "score": score, "falls": falls}


def cmd_choose(_a):
    grades = {}
    for d in SCORED:
        p = OUT / d / "grade_s1sel.json"
        g1 = U2I / d / "gate1.json"
        ok = g1.exists() and rd(g1)["status"] == "PASS"
        grades[d] = rd(p) if (p.exists() and ok) else None
    res = {"declared_in": DECLARED, "carve": "s1sel", **choose(grades), **codes()}
    zu2b.write_json(OUT / "choice.json", res)
    print(json.dumps({k: res[k] for k in ("outcome", "best", "score", "falls", "missing") if k in res}, indent=1))
    return 0


def cmd_report(_a):
    ch = rd(OUT / "choice.json")
    lines = ["# U2j report (docs/U2J_LINK_PRECISION.md)", "", f"Outcome: **{ch['outcome']}**"
             + (f", best {ch['best']}" if "best" in ch else ""), ""]
    if "score" in ch:
        lines += ["Score (mean I's ALL, " + ", ".join(SCORED) + "): "
                  + ", ".join(f"{o} {s}" for o, s in ch["score"].items()), ""]
    for d in zu2i.PASSAGE:
        for gp in sorted(OUT.glob(f"{d}/grade_*.json")):
            g = rd(gp)
            lines += [f"## {d}, {g['carve']} ({g['questions']} questions)", "",
                      f"{g['rows_reordered']:,} rows, {g['entries_reordered']:,} entries; classes {g['entry_classes']};"
                      f" lead {g['lead_entries']:,}; W2_S mean degree {g['w2s_mean_degree']}.", "",
                      "| order | I's ALL | recall | pool | bridge@5 | bridge@25 | gap share (I, bridge@25) |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for o, e in list(g["orders"].items()) + [("PKG", g["PKG"])]:
                i = e["I"] or {}
                gs = g["gap_share"].get(o, {})
                lines.append(f"| {o} | {i.get('ALL')} | {i.get('recall')} | {i.get('pool')} | {e['bridge']['at5']} | "
                             f"{e['bridge']['at25']} | {gs.get('I_ALL')}, {gs.get('bridge25')} |")
            lines.append("")
        bp = OUT / d / "beside.json"
        if bp.exists():
            b = rd(bp)
            lines += [f"Beside ({d}): {b['w2_links']:,} W2 links, {b['hyperlinked_share']} hyperlinked. AUC: "
                      + ", ".join(f"{f} {e['auc']}" for f, e in b["features"].items()), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


# ── selftest ────────────────────────────────────────────────────────────────


def _selftest(_a=None):
    # _expand equals zu2i.expand
    S = SimpleNamespace(g_ptr=np.array([0, 2, 3, 6]), g_nodes=np.array([4, 1, 0, 2, 3, 5]))
    src, grp = np.array([1, 0, 3]), np.array([0, 2, 1])
    s, t, j = _expand(S.g_ptr, S.g_nodes, src, grp)
    s2, t2 = zu2i.expand(S, src, grp)
    assert np.array_equal(s, s2) and np.array_equal(t, t2) and np.array_equal(src[j], s)
    # _chunk_links equals zu2i.w2_links on a toy table (surfaces 0..3: titles of groups 0, 1, 2 and one alias)
    group = np.array([0, 1, 2, 2, 1])
    order = np.argsort(group, kind="stable")
    E = SimpleNamespace(is_title=np.array([True, True, True, False]), lowid=np.array([0, 1, 2, 3]),
                        tgroup=np.array([0, 1, 2, -1]), common=np.array([False, False, True, False]), group=group,
                        g_nodes=order, g_ptr=np.searchsorted(group[order], np.arange(4)))
    hsrc, hk = np.array([0, 0, 1, 3, 3, 4]), np.array([1, 3, 0, 1, 2, 0])
    key, jj, dup = _chunk_links(hsrc, hk, E, 5)
    Sx = SimpleNamespace(is_title=E.is_title, lowid=E.lowid, tgroup=E.tgroup, group=group, g_ptr=E.g_ptr,
                         g_nodes=E.g_nodes)
    a2, b2 = zu2i.w2_links(Sx, {"src": hsrc, "k": hk}, E.common)
    assert np.array_equal(key, a2 * 5 + b2) and dup == 0
    assert key.tolist() == [1, 4, 5, 16, 19, 20]
    assert hk[jj].tolist() == [1, 1, 0, 1, 1, 0] and hsrc[jj].tolist() == [0, 0, 1, 3, 3, 4]
    key2, _j2, dup2 = _chunk_links(np.r_[hsrc, 0], np.r_[hk, 1], E, 5)
    assert np.array_equal(key2, key) and dup2 == 2
    # the orders on row 0 of a toy graph (W2 links with lead flags; one sibling pair 0-5)
    n = 7
    w2 = [(0, 1, 0), (0, 3, 1), (0, 4, 0), (0, 6, 1), (2, 0, 1), (4, 0, 1), (5, 3, 0)]
    ws = np.array([x[0] for x in w2])
    wt = np.array([x[1] for x in w2])
    lead = np.array([x[2] == 1 for x in w2])
    o = np.argsort(ws * n + wt)
    ws, wt, lead = ws[o], wt[o], lead[o]
    ip, col = zu2b.undirected_csr(n, np.concatenate([ws, [0]]), np.concatenate([wt, [5]]))
    sibkey = np.unique([0 * n + 5, 5 * n + 0])
    indeg = np.bincount(wt, minlength=n)
    rows = np.array([0, 3])
    pos, r, v, cls, ld, rare = row_features(ip, col, rows, ws * n + wt, lead, sibkey, indeg, n)
    want = {"R0": [1, 2, 3, 4, 5, 6], "R1": [4, 1, 3, 6, 2, 5], "R2": [4, 3, 6, 1, 2, 5], "R3": [2, 5, 1, 4, 6, 3],
            "R4": [4, 6, 3, 1, 2, 5]}
    for name, w in want.items():
        new = reorder(col, pos, r, v, order_key(name, cls, ld, rare))
        assert new[ip[0]:ip[1]].tolist() == w, (name, new[ip[0]:ip[1]].tolist())
        assert np.array_equal(new[ip[1]:ip[3]], col[ip[1]:ip[3]]) and np.array_equal(new[ip[4]:], col[ip[4]:])
    # row 3: neighbours 0 (in: 0->3, lead) and 5 (in: 5->3, not lead): R2 puts 0 first; R3 by in-degree (0: 2, 5: 0)
    assert reorder(col, pos, r, v, order_key("R2", cls, ld, rare))[ip[3]:ip[4]].tolist() == [0, 5]
    assert reorder(col, pos, r, v, order_key("R3", cls, ld, rare))[ip[3]:ip[4]].tolist() == [5, 0]
    # a neighbour that is neither a mention nor a sibling is refused
    try:
        row_features(ip, col, rows, ws * n + wt, lead, np.zeros(0, np.int64), indeg, n)
        raise AssertionError("no refusal")
    except SystemExit:
        pass
    # the bridge
    bip, bcol = np.array([0, 3, 3, 4]), np.array([2, 1, 0, 1])
    b = bridge(bip, bcol, [np.array([0]), np.array([1])], [np.array([0, 1]), np.array([0, 2])], rs=(1, 2))
    assert b == {"pairs": 3, "pairs_with_gold_in_B": 1, "at1": 0.0, "at2": round(1 / 3, 4)}, b
    # the AUC
    assert auc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1], bool)) == 1.0
    assert auc(np.array([1, 1]), np.array([0, 1], bool)) is None
    assert auc(np.array([1, 1, 2]), np.array([0, 1, 1], bool)) == 0.75
    # the choice

    def g(vals):
        return {"orders": {o: {"I": {"ALL": x}} for o, x in vals.items()}}
    base = {"R0": 0.8, "R1": 0.8, "R2": 0.8, "R3": 0.8, "R4": 0.8}
    assert choose({d: g(base) for d in SCORED})["outcome"] == "NO_GAIN"
    up = dict(base, R2=0.81, R4=0.811)
    c = choose({d: g(up) for d in SCORED})
    assert c["outcome"] == "ADOPT_FOR_POOLS" and c["best"] == "R2", c
    c = choose({"musique": g(dict(up, R2=0.79)), "hotpotqa": g(dict(up, R2=0.83)), "2wiki": g(up)})
    assert c["outcome"] == "NO_GAIN" and c["falls"], c
    assert choose({"musique": g(up), "hotpotqa": None, "2wiki": g(up)})["outcome"] == "REFUSED"
    edge = dict(base, R1=0.802)
    assert choose({d: g(edge) for d in SCORED})["outcome"] == "ADOPT_FOR_POOLS"
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset")
    ap.add_argument("--carve", default="s1sel")
    a = ap.parse_args(argv)
    if a.selftest or a.cmd == "selftest":
        return _selftest(a)
    return {"grade": cmd_grade, "beside": cmd_beside, "choose": cmd_choose, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
