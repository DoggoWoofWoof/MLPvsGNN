"""U2b (docs/U2B_LINK_RULE.md): the link rule U2a's causes call for, one rule for all six, chosen on s1sel by reach.

    python outputs/mp_unified/zu2b.py selftest
    python outputs/mp_unified/zu2b.py check-u1d            -> outputs/u2b/check_u1d_musique.json  (gate 1)
    python outputs/mp_unified/zu2b.py questions --dataset D -> outputs/u2b/<D>/questions_<carve>.npz (laptop)
    modal run outputs/mp_unified/zu2b_modal.py --dataset D --stage S                     (the corpus stages)
    python outputs/mp_unified/zu2b.py choose                -> outputs/u2b/choice.json
    python outputs/mp_unified/zu2b.py report                -> outputs/u2b/report.md

U1d's rule (outputs/mp_unified/u1d.py) is the starting point, and its functions are imported unchanged. This file adds
four switches, each applied the same way on every dataset:
- boundary "look": a surface occurs where the characters just outside it are not word characters. u1d's test copies
  re's \\b, which asks for a word character next to a surface edge that is itself punctuation, so a title ending in
  "Jr.", "F.C.", "Inc." or "!" matches almost nowhere;
- underscore: a title's underscores are read as spaces (a title stored in its URL form);
- prefixes (kind 3): the first j tokens of a multi-token title (parenthetical and comma tail removed), for each j
  whose last token starts with a capital;
- descriptive sources: a node whose text is its own title (an entity name) names nothing and is not read.
With every switch off and u1d's variants, the code reproduces u1d's filed musique edges exactly (gate 1).
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import u1b  # noqa: E402
import u1d  # noqa: E402

OUT = ROOT / "outputs" / "u2b"
DATASETS = u1b.DATASETS
PASSAGE = ("musique", "squad", "hotpotqa", "2wiki")
MIN_LEN = u1d.MIN_LEN
AMBIG = u1d.AMBIG
PAREN = u1d.PAREN
TRAIL = " ,:;-"
DAMP = 1024                                   # U1e's damping: a target with more mention links in is dropped
GRID = (25, 50, 100, 250, 500, 1000, 2000, 5000)
CHOICE_GRID = (100, 250, 500, 1000, 2000)
TIE = 0.002


@dataclass(frozen=True)
class Rule:
    boundary: str = "u1d"                     # "u1d" (u1b.bounded, re's \b) or "look"
    underscore: bool = False
    prefixes: bool = False
    descriptive: bool = False


U1D_RULE = Rule()
NEW_RULE = Rule("look", True, True, True)
# U2b's variants: (kinds, exact-first[, gate]). W2 is u1d's V2 under the new switches; the rest add aliases; a "g"
# variant also links a common form where the text writes it capitalized, mid-sentence ("capmid"). Simpler first.
VARIANTS = {"W2": ((0,), False), "W5e": ((0, 1), True), "W6e": ((0, 1, 2), True), "W3e": ((0, 1, 3), True),
            "W7e": ((0, 1, 2, 3), True), "W7": ((0, 1, 2, 3), False), "W3eg": ((0, 1, 3), True, "capmid"),
            "W7eg": ((0, 1, 2, 3), True, "capmid")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def save_npz(p, **arrs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez(tmp, **arrs)
    os.replace(tmp, p)
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ── the rule ────────────────────────────────────────────────────────────────


def norm_title(t, rule):
    return t.replace("_", " ") if rule.underscore else t


def is_descriptive(title, text):
    """False for a node whose text is its own title (a KB entity name): it names nothing."""
    return (text or "").strip() != (title or "").strip()


def forms_of(t, rule):
    """u1d.Surfaces' forms of a stripped title, then the prefixes (kind 3) when the rule has them."""
    forms = [(t, 0)]
    a = PAREN.sub("", t).strip()
    if a != t:
        forms.append((a, 1))
    if "," in t:
        b = t.split(",")[0].strip()
        if b != t:
            forms.append((b, 1))
    toks = PAREN.sub("", t).split()
    if len(toks) >= 2 and toks[-1][:1].isupper():
        forms.append((toks[-1], 2))
    if rule.prefixes:
        base = PAREN.sub("", t).split(",")[0].split()
        for j in range(1, len(base)):
            if base[j - 1][:1].isupper():
                forms.append((" ".join(base[:j]).rstrip(TRAIL), 3))
    return forms


def bounded_look(tl, low):
    return next(occ_look(tl, low), None) is not None


def occ_look(tl, low):
    n, m = len(low), len(tl)
    isw = u1b._isw
    p = tl.find(low)
    while p >= 0:
        if (p == 0 or not isw(tl[p - 1])) and (p + n >= m or not isw(tl[p + n])):
            yield p
        p = tl.find(low, p + 1)


class Table(u1b.TitleTable):
    """u1b's title index (its run tuples are a superset matcher for either boundary), with the rule's bounded test."""

    def __init__(self, surfs, rule):
        super().__init__(surfs)
        self.rule = rule

    def match(self, i, text):
        if self.rule.boundary == "u1d":
            return u1b.TitleTable.match(self, i, text)
        tl = text.lower()
        runs = u1b.WORD.findall(tl)
        if not runs:
            return []
        nr = len(runs)
        cands = set()
        first, idx = self.first, self.idx
        for k, r in enumerate(runs):
            ls = first.get(r)
            if ls is None:
                continue
            for L in ls:
                if k + L > nr:
                    break
                js = idx.get(tuple(runs[k:k + L]))
                if js:
                    cands.update(js)
        if not cands:
            return []
        toks = set(runs)
        key, low = self.key, self.low
        return sorted(j for j in cands if j != i and key[j] in toks and bounded_look(tl, low[j]))


class Surfaces(u1d.Surfaces):
    """u1d.Surfaces with the rule's title reading and kind-3 forms; with U1D_RULE it builds u1d's object exactly."""

    def __init__(self, titles, rule):
        self.rule = rule
        own = {}
        nt = [norm_title(t, rule).strip() for t in titles]
        for j, t in enumerate(nt):
            seen = set()
            for sf, kd in forms_of(t, rule):
                if len(sf) < MIN_LEN or (sf, kd) in seen:
                    continue
                seen.add((sf, kd))
                own.setdefault(sf, []).append((j, kd))
        self.surf = sorted(own)
        ptr = [0]
        nodes, kinds = [], []
        for sf in self.surf:
            for j, kd in own[sf]:
                nodes.append(j)
                kinds.append(kd)
            ptr.append(len(nodes))
        del own
        self.ptr = np.asarray(ptr, np.int64)
        self.nodes = np.asarray(nodes, np.int64)
        self.kinds = np.asarray(kinds, np.int8)
        lows = {}
        self.lowid = np.asarray([lows.setdefault(s.lower(), len(lows)) for s in self.surf], np.int64)
        self.n_low = len(lows)
        g = {}
        self.group = np.asarray([g.setdefault(t, len(g)) for t in nt], np.int64)
        order = np.argsort(self.group, kind="stable")
        self.g_nodes = order
        self.g_ptr = np.searchsorted(self.group[order], np.arange(len(g) + 1))
        self.table = Table(self.surf, rule)


def occurrences(tl, low, rule):
    return u1d.occurrences(tl, low) if rule.boundary == "u1d" else occ_look(tl, low)


def scan(S, text, rule):
    """u1d.scan with the rule's boundary: [(surface k, case_ok)], {lowid: [cap, lower]} mid-sentence counts, and the
    surfaces written exactly as titled, capitalized, somewhere mid-sentence (empty under u1d's boundary)."""
    if rule.boundary == "u1d":
        return (*u1d.scan(S, text), frozenset())
    tl = text.lower()
    same = len(tl) == len(text)
    out, stat = [], {}
    capmid = set()
    done_low = set()
    for k in S.table.match(-1, text):
        sf = S.surf[k]
        low = S.table.low[k]
        lid = int(S.lowid[k])
        ok = False
        count = lid not in done_low
        done_low.add(lid)
        for p in occ_look(tl, low):
            if not same:
                ok = True
                break
            sl = text[p:p + len(low)]
            if sl[1:] == sf[1:] and sl[0].lower() == sf[0].lower():
                ok = True
            mid = not u1d.sent_start(text, p)
            if mid and sl == sf and sf[:1].isupper():
                capmid.add(k)
            if count and mid:
                c = stat.setdefault(lid, [0, 0])
                if sl[0].isupper():
                    c[0] += 1
                elif sl == low:
                    c[1] += 1
        out.append((k, ok))
    return out, stat, capmid


def choose_targets(S, i, hits, common, need, variants, capmid=frozenset()):
    """u1d.choose_targets over the given variants; a "capmid" variant keeps a common form written capitalized,
    mid-sentence."""
    res = {v: [] for v in variants}
    for k, ok in hits:
        if not ok:
            continue
        com = bool(common[S.lowid[k]])
        if com and k not in capmid:
            continue
        nodes, kinds = S.owners(k)
        keep = nodes != i
        nodes, kinds = nodes[keep], kinds[keep]
        if nodes.size == 0:
            continue
        for v, spec in variants.items():
            vk, exact_first = spec[0], spec[1]
            if com and spec[2:] != ("capmid",):
                continue
            m = np.isin(kinds, vk)
            if exact_first and (kinds[m] == 0).any():
                m &= kinds == 0
            if not m.any():
                continue
            gs = np.unique(S.group[nodes[m]])
            if gs.size > AMBIG:
                continue
            if gs.size == 1:
                res[v].append(int(gs[0]))
            else:
                need.append((i, v, tuple(gs.tolist())))
    return res


def stat_pass(S, texts, titles_lo, rule, lo=0):
    """The capitalization counts over texts (positions lo..), skipping non-descriptive nodes under the rule."""
    cap = np.zeros(S.n_low, np.int64)
    low = np.zeros(S.n_low, np.int64)
    for off, text in enumerate(texts):
        if rule.descriptive and not is_descriptive(titles_lo[off], text):
            continue
        _h, st, _c = scan(S, text, rule)
        for lid, (c, l_) in st.items():
            cap[lid] += c
            low[lid] += l_
    return cap, low


def link_pass(S, texts, titles_lo, rule, common, dense_get, variants, lo=0, chunk=u1d.CHUNK, cache_cap=200000,
              progress=None):
    """u1d.cmd_link's loop: directed (src, dst) per variant, with the context choice by the frozen dense vectors."""
    edges = {v: ([], []) for v in variants}
    stats = {v: {"surface_links": 0, "ambiguous": 0} for v in variants}
    gcache = {}

    def group_vecs(g):
        v = gcache.get(g)
        if v is None:
            if len(gcache) >= cache_cap:
                gcache.clear()
            v = gcache[g] = dense_get(S.group_nodes(g))
        return v

    for c0 in range(0, len(texts), chunk):
        chosen, need = {}, []
        for off in range(c0, min(c0 + chunk, len(texts))):
            i = lo + off
            if rule.descriptive and not is_descriptive(titles_lo[off], texts[off]):
                chosen[i] = {v: [] for v in variants}
                continue
            hits, _st, cm = scan(S, texts[off], rule)
            chosen[i] = choose_targets(S, i, hits, common, need, variants, cm)
        if need:
            src = np.asarray(sorted({x[0] for x in need}), np.int64)
            sv = dict(zip(src.tolist(), dense_get(src)))
            for i, v, gs in need:
                best, bg = -9.0, -1
                for g in gs:
                    s = float((group_vecs(g) @ sv[i]).max())
                    if s > best:
                        best, bg = s, g
                chosen[i][v].append(bg)
                stats[v]["ambiguous"] += 1
        for i, per in chosen.items():
            for v, gl in per.items():
                stats[v]["surface_links"] += len(gl)
                tg = set()
                for g in gl:
                    tg.update(S.group_nodes(g).tolist())
                tg.discard(i)
                for j in sorted(tg):
                    edges[v][0].append(i)
                    edges[v][1].append(j)
        if progress is not None:
            progress(min(c0 + chunk, len(texts)), {v: len(e[0]) for v, e in edges.items()})
    return ({v: (np.asarray(s_, np.int32), np.asarray(d_, np.int32)) for v, (s_, d_) in edges.items()}, stats)


# ── the graphs a variant gives ──────────────────────────────────────────────


def damp(src, dst, n, c=DAMP):
    """U1e: drop the mention links whose target has more than c mention links in."""
    ind = np.bincount(dst, minlength=n)
    keep = ind[dst] <= c
    return src[keep], dst[keep]


def sibling_pairs(group, descriptive):
    """Every ordered pair of distinct descriptive nodes that share a title group."""
    g = np.where(descriptive, group, -1)
    o = np.argsort(g, kind="stable")
    gs = g[o]
    src, dst = [], []
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
    ends = np.r_[starts[1:], gs.size]
    for a, b in zip(starts, ends):
        if gs[a] < 0 or b - a < 2:
            continue
        mem = o[a:b]
        ii, jj = np.meshgrid(mem, mem, indexing="ij")
        m = ii != jj
        src.append(ii[m])
        dst.append(jj[m])
    if not src:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(src).astype(np.int64), np.concatenate(dst).astype(np.int64)


def undirected_csr(n, src, dst):
    """Undirected neighbours, each row sorted by node id (m3b_pools' FamilyStore order)."""
    a = np.concatenate([src, dst]).astype(np.int64)
    b = np.concatenate([dst, src]).astype(np.int64)
    k = np.unique(a * n + b)
    a, b = k // n, k % n
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(a, minlength=n), out=indptr[1:])
    return indptr, b.astype(np.int32)


# ── grading: reach of B and its one-link neighbours by cos(q, v) ────────────


def x_of(indptr, col, B):
    """B's one-link neighbours outside B, ascending."""
    if not B.size:
        return np.zeros(0, np.int64)
    nb = np.unique(np.concatenate([col[indptr[b]:indptr[b + 1]] for b in B]).astype(np.int64))
    return nb[~np.isin(nb, B)]


def reach_curves(graphs, B_list, golds, qemb, dense_get, grid=GRID, chunk=65536):
    """Per graph, the pool B ∪ X(k), X = B's one-link neighbours outside B ranked by cos(q, v), ties by node id.
    A question's scores are computed once, over the union of the graphs' X, so every graph ranks by the same numbers.
    ALL and recall are per-question means (recall as the frozen I's tally takes it); rows are |B| + min(k, |X|)."""
    names = list(graphs)
    nq = len(B_list)
    acc = {g: {"all": dict.fromkeys(grid, 0), "rec": dict.fromkeys(grid, 0.0), "rows": dict.fromkeys(grid, 0),
               "sat_all": 0, "sat_rec": 0.0, "x": 0} for g in names}
    b_rows = b_all = 0
    b_rec = 0.0
    n_g = 0
    for j in range(nq):
        B = np.unique(np.asarray(B_list[j], np.int64))
        g = np.unique(np.asarray(golds[j], np.int64))
        n_g += g.size
        b_rows += B.size
        inB = np.isin(g, B)
        b_all += int(inB.all())
        b_rec += float(inB.mean())
        xs = {name: x_of(graphs[name][0], graphs[name][1], B) for name in names}
        want = [name for name in names if np.isin(g[~inB], xs[name]).any()]
        if want:
            U = np.unique(np.concatenate([xs[name] for name in want]))
            q = np.asarray(qemb[j], np.float32)
            sc = np.empty(U.size, np.float32)
            for c0 in range(0, U.size, chunk):
                sc[c0:c0 + chunk] = dense_get(U[c0:c0 + chunk]) @ q
        for name in names:
            X, a = xs[name], acc[name]
            ing = np.isin(g, X)
            a["sat_all"] += int((inB | ing).all())
            a["sat_rec"] += float((inB | ing).mean())
            a["x"] += X.size
            gx = g[~inB & ing]
            if gx.size:
                sx = sc[np.searchsorted(U, X)]
                sg = sc[np.searchsorted(U, gx)]
                gpos = np.asarray([int(((sx > s_) | ((sx == s_) & (X < x_))).sum()) for x_, s_ in zip(gx, sg)])
            else:
                gpos = np.zeros(0, np.int64)
            base = int(inB.sum())
            for k in grid:
                got = base + int((gpos < k).sum())
                a["all"][k] += int(got == g.size)
                a["rec"][k] += got / g.size
                a["rows"][k] += B.size + min(k, X.size)
    out = {}
    for name in names:
        a = acc[name]
        out[name] = {"questions": nq, "golds": n_g,
                     "B": {"ALL": round(b_all / nq, 4), "recall": round(b_rec / nq, 4), "rows": round(b_rows / nq, 1)},
                     "curve": {str(k): {"ALL": round(a["all"][k] / nq, 4), "recall": round(a["rec"][k] / nq, 4),
                                        "rows": round(a["rows"][k] / nq, 1)} for k in grid},
                     "all_neighbours": {"ALL": round(a["sat_all"] / nq, 4), "recall": round(a["sat_rec"] / nq, 4),
                                        "rows": round((b_rows + a["x"]) / nq, 1)}}
    return out


def reach_curve(indptr, col, B_list, golds, qemb, dense_get, grid=GRID):
    return reach_curves({"g": (indptr, col)}, B_list, golds, qemb, dense_get, grid)["g"]


def choice_score(curve):
    return float(np.mean([curve[str(k)]["ALL"] for k in CHOICE_GRID]))


# ── the questions' side (laptop): B, golds, the query vector, the frozen I's other families ─────────────────


def ragged(rows):
    rows = [np.asarray(r, np.int64) for r in rows]
    ptr = np.zeros(len(rows) + 1, np.int64)
    np.cumsum([r.size for r in rows], out=ptr[1:])
    col = np.concatenate(rows).astype(np.int64) if rows and ptr[-1] else np.zeros(0, np.int64)
    return ptr, col


def unragged(ptr, col):
    return [col[ptr[j]:ptr[j + 1]] for j in range(ptr.size - 1)]


def cmd_questions(a):
    """Per carve: base rows, seeds, golds and the query vectors, sliced as m3b_compile.prepare slices them."""
    import step4_pool_reach as S4
    import step4c_walk_pools as C4
    t0 = time.time()
    name = a.dataset
    op = S4.Opened()
    mc, mp = op.m3b_compile, S4.m3b_pools
    ds = op.canonical.Dataset(name, root=str(op.served))
    construction = op.construction[name]
    pops = [(c, p) for c, p in C4.populations(op, S4, ds, name) if c in ("s1sel", "s1eval")]
    all_idx = np.concatenate([p.idx for _c, p in pops])
    sliced = {}
    for model in ("dense", "splade"):
        ids, _scores = ds.cache(model)
        sliced[model] = np.asarray(ids[all_idx], np.int64)
        del ids, _scores
    qemb = np.asarray(ds.embeddings("dense", "queries").read(all_idx), dtype=np.float32)
    constant = int(op.cfg_h["retrieval_pools"]["equal_rrf"]["constant"])
    fams = mc.regime_families(op.cfg_h, construction["regime"]) if construction["regime"] != "RETRIEVAL" else []
    def csr(f):
        z = np.load(mc.CSR_CACHE / f"{name}_{f}.npz")
        return z["indptr"], z["col"]

    others = {f: csr(f) for f in fams if f != "structural"}
    rec = {"declared_in": "docs/U2B_LINK_RULE.md", "dataset": name, "construction": construction, "families": fams,
           "freeze_RECORD_SHA256": op.freeze["RECORD_SHA256"], "zu2b_sha256": u1b.sha_src(__file__), "carves": {}}
    start = 0
    for c, pop in pops:
        sl = slice(start, start + pop.idx.size)
        start += pop.idx.size
        d_ids, s_ids = sliced["dense"][sl], sliced["splade"][sl]
        seeds = [mp.seeds_of(d, s) for d, s in zip(d_ids, s_ids)]
        bases = mc.base_rows(construction, d_ids, s_ids, op.m3a, op.m3b_contract, constant)
        golds = [np.unique(np.asarray(g, np.int64)) for g in pop.golds]
        arrs = {}
        for key, rows in (("base", bases), ("seeds", seeds), ("gold", golds)):
            arrs[f"{key}_ptr"], arrs[f"{key}_col"] = ragged(rows)
        arrs["qemb"] = qemb[sl]
        arrs["fam_order"] = np.asarray([f for f in fams], dtype="U16")
        for f, (ip, cl) in others.items():
            arrs[f"{f}_indptr"], arrs[f"{f}_col"] = ip, cl
        p = OUT / name / f"questions_{c}.npz"
        sha = save_npz(p, **arrs)
        write_json(OUT / name / f"questions_{c}.json", {"ids": list(pop.ids)})
        rec["carves"][c] = {"questions": len(golds), "golds": int(sum(g.size for g in golds)), "npz_sha256": sha,
                            "ids_sha256": S4.digest(list(pop.ids))}
        log(f"{name}/{c}: {len(golds)} questions ({time.time() - t0:.0f} s)")
    ip, cl = csr("structural")
    rec["pkg_structural_npz_sha256"] = save_npz(OUT / name / "pkg_structural.npz", indptr=ip, col=cl)
    write_json(OUT / name / "questions.json", rec)
    return 0


# ── gate 1: u1d's musique edges, reproduced with every switch off ───────────


def cmd_check_u1d(a):
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    titles, texts, n, nodes_sha = u1b.read_nodes(served, "musique")
    S0 = u1d.Surfaces(titles)
    S1 = Surfaces(titles, U1D_RULE)
    same_s = (S0.surf == S1.surf and all(np.array_equal(getattr(S0, f), getattr(S1, f))
                                         for f in ("ptr", "nodes", "kinds", "lowid", "group", "g_nodes", "g_ptr")))
    log(f"surfaces equal: {same_s} ({time.time() - t0:.0f} s)")
    cs = np.load(ROOT / "outputs" / "u1d" / "musique" / "capstat.npz")
    cap, low = stat_pass(S1, texts, titles, U1D_RULE)
    same_c = bool(np.array_equal(cap, cs["cap"]) and np.array_equal(low, cs["low"]))
    log(f"capstat equal: {same_c} ({time.time() - t0:.0f} s)")
    D = u1d.Dense(served, "musique", n)
    edges, _stats = link_pass(S1, texts, titles, U1D_RULE, cs["low"] > cs["cap"], D.get, u1d.VARIANTS)
    z = np.load(ROOT / "outputs" / "u1d" / "musique" / "link" / "s0of1.npz")
    same_e = {v: bool(np.array_equal(edges[v][0], z[f"{v}_src"]) and np.array_equal(edges[v][1], z[f"{v}_dst"]))
              for v in u1d.VARIANTS}
    rec = {"declared_in": "docs/U2B_LINK_RULE.md", "gate": 1, "surfaces_equal": same_s, "capstat_equal": same_c,
           "edges_equal": same_e, "nodes_sha256": nodes_sha, "zu2b_sha256": u1b.sha_src(__file__),
           "u1d_sha256": u1b.sha_src(u1d.__file__), "secs": round(time.time() - t0, 1),
           "status": "PASS" if same_s and same_c and all(same_e.values()) else "FAIL"}
    write_json(OUT / "check_u1d_musique.json", rec)
    log(json.dumps(rec))
    return 0 if rec["status"] == "PASS" else 1


# ── choice and report (laptop, over the grade records fetched to outputs/u2b/<D>/grade/) ────────────────────


FORMS = ("M", "D", "S", "DS")       # mention links; damped (U1e); with the sibling links; both. Simpler first.
REFS = ("PKG", "V2_M", "V2_D")      # the package's own structural family; u1d's V2 rebuilt; V2 damped
FILED_I = {"PKG": "G0", "V2_M": "GU"}


def candidates():
    return [f"{v}_{f}" for v in VARIANTS for f in FORMS]


def load_grades(carve):
    res = {}
    for d in DATASETS:
        for p in sorted((OUT / d / "grade").glob(f"{carve}_*.json")):
            r = json.loads(p.read_text(encoding="utf-8"))
            for g, e in r["graphs"].items():
                if g in res.setdefault(d, {}) and res[d][g]["npz_sha256"] != e["npz_sha256"]:
                    raise SystemExit(f"{d} {g}: two grade records name different graphs")
                res[d][g] = e
    return res


def gate2(grades, carve):
    """The frozen I over the package's family and over u1d's rebuilt V2 must be step 4h's filed G0 and GU."""
    out = {}
    for d in ("musique", "hotpotqa", "2wiki"):
        h = json.loads((ROOT / "outputs" / "step4h" / f"coverage_{d}.json").read_text(encoding="utf-8"))
        for g, f in FILED_I.items():
            got = grades.get(d, {}).get(g, {}).get("I")
            want = h["carves"][carve]["graphs"][f]["I"]
            out[f"{d}/{g}"] = {"got": got, "filed": want, "equal": got == want}
    return out


def cmd_choose(_a):
    sel = load_grades("s1sel")
    g2 = gate2(sel, "s1sel")
    missing = [(d, c) for d in PASSAGE for c in candidates() + list(REFS) if c not in sel.get(d, {})]
    score = {c: round(float(np.mean([choice_score(sel[d][c]["curve"]) for d in PASSAGE])), 4)
             for c in candidates() + list(REFS) if all(c in sel.get(d, {}) for d in PASSAGE)}
    cands = [c for c in candidates() if c in score]
    best = max(score[c] for c in cands) if cands else None
    chosen = next((c for c in cands if score[c] >= best - TIE), None)
    ok = all(v["equal"] for v in g2.values()) and not missing
    rec = {"declared_in": "docs/U2B_LINK_RULE.md", "carve": "s1sel", "criterion": "mean over the four passage datasets"
           f" of the mean ALL of B ∪ X(k) over k in {list(CHOICE_GRID)}; within {TIE} of the best, the earlier",
           "gate2": g2, "missing": missing, "score": score, "best": best, "chosen": chosen,
           "per_dataset": {d: {c: round(choice_score(sel[d][c]["curve"]), 4) for c in sel.get(d, {})} for d in sel},
           "zu2b_sha256": u1b.sha_src(__file__), "status": ("CHOSEN" if ok and chosen else "REFUSED")}
    write_json(OUT / "choice.json", rec)
    log(f"{rec['status']} {chosen} (score {score.get(chosen)}; V2_M {score.get('V2_M')}, PKG {score.get('PKG')})")
    return 0


def cmd_report(_a):
    ch = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))
    lines = [f"# U2b: {ch['status']} {ch['chosen']}", "", f"Criterion: {ch['criterion']}.", "",
             "## Gate 2 (the frozen I, s1sel)", "", "| graph | got | filed | equal |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['got']} | {v['filed']} | {v['equal']} |" for k, v in ch["gate2"].items()]
    lines += ["", "## Choice score (s1sel)", "", "| graph | " + " | ".join(PASSAGE) + " | mean |",
              "|---|" + "---|" * (len(PASSAGE) + 1)]
    for c, s in sorted(ch["score"].items(), key=lambda x: -x[1]):
        lines.append(f"| {c} | " + " | ".join(f"{ch['per_dataset'][d].get(c, float('nan')):.4f}" for d in PASSAGE)
                     + f" | {s:.4f} |")
    for carve in ("s1sel", "s1eval"):
        gr = load_grades(carve)
        for d in DATASETS:
            if d not in gr:
                continue
            show = [g for g in (ch["chosen"], "V2_M", "V2_D", "PKG") if g in gr[d]]
            lines += ["", f"## {d} {carve}: ALL / recall / rows of B ∪ X(k)", "",
                      "| k | " + " | ".join(show) + " |", "|---|" + "---|" * len(show)]
            for k in ["B"] + [str(k) for k in GRID] + ["all", "I"]:
                cells = []
                for g in show:
                    e = gr[d][g]
                    v = e["B"] if k == "B" else e["all_neighbours"] if k == "all" else e["I"] if k == "I" else \
                        e["curve"][k]
                    cells.append("-" if v is None else
                                 f"{v['ALL']:.4f} / {v['recall']:.4f} / {v.get('rows', v.get('pool'))}")
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log(f"wrote {OUT / 'report.md'}")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def _selftest_reach():
    """reach_curves against a full sort per graph, on integer vectors (exact dot products) with repeated scores."""
    rng = np.random.default_rng(0)
    n, d = 80, 8
    V = rng.integers(-2, 3, (n, d)).astype(np.float32)
    V[40:50] = V[30]                                                            # ties broken by node id
    graphs = {}
    for name, m in (("a", 150), ("b", 400)):
        ip, col = undirected_csr(n, rng.integers(0, n, m), rng.integers(0, n, m))
        graphs[name] = (ip, col)
    B_list = [rng.choice(n, rng.integers(0, 6), replace=False) for _ in range(40)]
    golds = [rng.choice(n, rng.integers(1, 4), replace=False) for _ in range(40)]
    Q = rng.integers(-2, 3, (40, d)).astype(np.float32)
    grid = (1, 2, 5, 10, 1000)
    got = reach_curves(graphs, B_list, golds, Q, lambda r: V[np.asarray(r)], grid, chunk=7)
    for name, (ip, col) in graphs.items():
        allk, rec = dict.fromkeys(grid, 0), dict.fromkeys(grid, 0.0)
        for B, g, q in zip(B_list, golds, Q):
            B, g = np.unique(B), np.unique(g)
            X = x_of(ip, col, B)
            order = X[np.lexsort((X, -(V[X] @ q)))] if X.size else X
            for k in grid:
                pool = set(B.tolist()) | set(order[:k].tolist())
                h = np.asarray([x in pool for x in g.tolist()])
                allk[k] += int(h.all())
                rec[k] += float(h.mean())
        for k in grid:
            assert got[name]["curve"][str(k)]["ALL"] == round(allk[k] / 40, 4), (name, k)
            assert got[name]["curve"][str(k)]["recall"] == round(rec[k] / 40, 4), (name, k)


def cmd_selftest(_a):
    titles = ["Leeds United F.C.", "Leeds", "Louis Gossett Jr.", "New_York_City", "Siouxsie Sioux",
              "Siouxsie and the Banshees", "Kismet", "Croatia national football team", "Croatia", "Dio (band)"]
    S = Surfaces(titles, NEW_RULE)
    sid = {s: k for k, s in enumerate(S.surf)}
    own = lambda s: sorted(zip(*[x.tolist() for x in S.owners(sid[s])]))  # noqa: E731
    assert own("Leeds United") == [(0, 3)], own("Leeds United")
    assert own("Leeds") == [(0, 3), (1, 0)], own("Leeds")
    assert own("New York City") == [(3, 0)] and "New_York_City" not in sid
    assert own("New York") == [(3, 3)]
    assert own("Siouxsie") == [(4, 3), (5, 3)]
    assert "Siouxsie and" not in sid and "Croatia national" not in sid           # a prefix ends on a capital
    assert own("Croatia") == [(7, 3), (8, 0)]
    assert "Dio" not in sid                                                      # under 4 characters
    S0 = Surfaces(titles, U1D_RULE)
    t = "He played for Leeds United F.C. and met Louis Gossett Jr.. Then New York City."
    h_new = {S.surf[k] for k, ok in scan(S, t, NEW_RULE)[0] if ok}
    h_old = {S0.surf[k] for k, ok in scan(S0, t, U1D_RULE)[0] if ok}
    assert {"Leeds United F.C.", "Louis Gossett Jr.", "New York City", "Leeds United"} <= h_new, h_new
    assert "Leeds United F.C." not in h_old and "Louis Gossett Jr." not in h_old, h_old
    assert not is_descriptive("Kismet", "Kismet") and is_descriptive("Kismet", "Kismet is a film.")
    cap, low = stat_pass(S, ["Kismet", "They said Leeds United won."], ["Kismet", "x"], NEW_RULE)
    assert cap.sum() == 2 and low.sum() == 0, (cap.sum(), low.sum())             # "leeds united" and "leeds"
    t2, x2 = ["Leeds", "Leeds Rhinos"], ["Leeds is a city.", "Leeds Rhinos"]
    S2 = Surfaces(t2, NEW_RULE)
    no_gate = np.zeros(S2.n_low, bool)
    e_new, _ = link_pass(S2, x2, t2, NEW_RULE, no_gate, None, {"W2": ((0,), False)})
    e_all, _ = link_pass(S2, x2, t2, Rule("look", True, True, False), no_gate, None, {"W2": ((0,), False)})
    assert e_new["W2"][0].size == 0, e_new                                        # the name-only node is not read
    assert list(zip(e_all["W2"][0].tolist(), e_all["W2"][1].tolist())) == [(1, 0)], e_all
    t3, x3 = ["Apple", "Orchard"], ["Apple is a company.", "In the orchard an apple fell, and Apple sued."]
    S3 = Surfaces(t3, NEW_RULE)
    com = np.zeros(S3.n_low, bool)
    com[S3.lowid[S3.surf.index("Apple")]] = True
    e3, _ = link_pass(S3, x3, t3, NEW_RULE, com, None, {"W2": ((0,), False), "G": ((0,), False, "capmid")})
    assert e3["W2"][0].size == 0 and list(zip(e3["G"][0].tolist(), e3["G"][1].tolist())) == [(1, 0)], e3
    src, dst = sibling_pairs(np.array([0, 0, 1, 2, 2, 2]), np.array([True, True, True, True, True, False]))
    assert sorted(zip(src.tolist(), dst.tolist())) == [(0, 1), (1, 0), (3, 4), (4, 3)]
    s_, d_ = damp(np.array([0, 1, 2, 3]), np.array([5, 5, 5, 6]), 7, c=2)
    assert s_.tolist() == [3] and d_.tolist() == [6]
    ip, col = undirected_csr(4, np.array([0, 2]), np.array([1, 1]))
    assert col[ip[1]:ip[2]].tolist() == [0, 2]
    _selftest_reach()
    log("selftest passed")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    sub.add_parser("check-u1d")
    p = sub.add_parser("questions")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    sub.add_parser("choose")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": cmd_selftest, "check-u1d": cmd_check_u1d, "questions": cmd_questions, "choose": cmd_choose,
            "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
