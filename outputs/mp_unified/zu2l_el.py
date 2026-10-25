"""U2l: one entity-link family (EL) on all six datasets, graded as U2b graded graphs (docs/U2L_ENTITY_MATCH.md).

    python outputs/mp_unified/zu2l_el.py build --dataset D
    python outputs/mp_unified/zu2l_el.py gate3                       # musique: family-built X and I against zu2b's
    python outputs/mp_unified/zu2l_el.py grade --dataset D [--carve C]
    python outputs/mp_unified/zu2l_el.py choose
    python outputs/mp_unified/zu2l_el.py report
    python outputs/mp_unified/zu2l_el.py qlinks --dataset D [--carve C]   # beside, after the choice
    python outputs/mp_unified/zu2l_el.py --selftest

EL has two kinds. IS: a node mentions a title group's surface (U2b's scan, W2's gates); on a passage dataset these are
U2i's W2 links plus the aliases U2i's filed choice names. CO: two nodes mention one entity, kept at NER's df cap
[2, 25] and weighted by the sum of 1/df: CO_t over the title groups, CO_s the package's NER edges. On a KB a node's
name is read as its text (U2b's descriptive switch off); webqsp's CVT mediators neither mention nor own.

Outputs only under outputs/u2l/el/. Arrays (npz) stay uncommitted. Nothing trains; no filed graph or record changes.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import u1b  # noqa: E402
import u1d  # noqa: E402
import zu2b  # noqa: E402
import zu2i  # noqa: E402

DECLARED = "docs/U2L_ENTITY_MATCH.md"
OUT = ROOT / "outputs" / "u2l" / "el"
U2B = ROOT / "outputs" / "u2b"
U2I = ROOT / "outputs" / "u2i"
SIX = ("musique", "squad", "hotpotqa", "2wiki", "metaqa", "webqsp")
PASSAGE = zu2b.PASSAGE
KB = u1b.KB
FIVE = ("musique", "squad", "hotpotqa", "2wiki", "metaqa")
CARVE = {d: ("s1eval" if d == "webqsp" else "s1sel") for d in SIX}
DF = (2, 25)                                      # NER's cap, applied to the title groups (CO_t)
KB_RULE = zu2b.Rule("look", True, True, False)    # U2b's NEW_RULE with the descriptive switch off
W2 = {"W2": ((0,), False)}
CVT = "CVT_MEDIATOR"
FORMS = {"E1": ("IS",), "E2": ("IS", "CO_t"), "E3": ("IS", "CO_s"), "E4": ("IS", "CO_t", "CO_s")}
UNIONS = {"alone": (), "R": ("R",), "R_kNN": ("R", "kNN")}
CANDS = {f"{f}+{u}": FORMS[f] + UNIONS[u] for f in FORMS for u in UNIONS}
REFS = {"PKG": ("PKG",), "W2_S": ("W2_S",), "PKG_NER_kNN": ("PKG", "CO_s", "kNN"),
        "W2_S_NER_kNN": ("W2_S", "CO_s", "kNN")}
GRAPHS = {**CANDS, **REFS}
I_OF = {**{f: FORMS[f] + ("R",) for f in FORMS}, "PKG": ("PKG",), "W2_S": ("W2_S",)}
PICK = "R_kNN"                                    # the choice reads each form's three-family union
TIE = zu2b.TIE
BLOCK = 262144


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False), encoding="utf-8")
    tmp.replace(p)


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def codes():
    return {"script_sha256": script_sha256(), "zu2b_sha256": zu2b_sha(), "zu2i_sha256": zu2i_sha()}


def zu2b_sha():
    return hashlib.sha256((HERE / "zu2b.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def zu2i_sha():
    return hashlib.sha256((HERE / "zu2i.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── EL's kinds ───────────────────────────────────────────────────────────────


def co_mentions(src, grp, n, lo=DF[0], hi=DF[1]):
    """CO_t: every pair of distinct nodes that both mention one group E with df(E) in [lo, hi] (df = the distinct
    nodes mentioning E), as (a, b, weight) with a < b, weight the sum over shared E of 1/df(E); pairs ascending."""
    src = np.asarray(src, np.int64)
    grp = np.asarray(grp, np.int64)
    if src.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    k = np.unique(grp * n + src)                 # one (E, u) per mention, sorted by E then u
    e, u = k // n, k % n
    del k
    starts = np.flatnonzero(np.r_[True, e[1:] != e[:-1]])
    sizes = np.diff(np.r_[starts, e.size])
    keys, wts = [], []
    for c in range(max(lo, 2), hi + 1):
        s = starts[sizes == c]
        if s.size == 0:
            continue
        mem = u[s[:, None] + np.arange(c)[None, :]]          # (groups of size c) x c, each row ascending
        ia, ib = np.triu_indices(c, 1)
        a, b = mem[:, ia].ravel(), mem[:, ib].ravel()
        keys.append(a * n + b)
        wts.append(np.full(a.size, 1.0 / c))
    del e, u
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.float64)
    keys, wts = np.concatenate(keys), np.concatenate(wts)
    uk, inv = np.unique(keys, return_inverse=True)
    w = np.bincount(inv, weights=wts, minlength=uk.size)
    return uk // n, uk % n, w


def kb_texts(served, d):
    """A KB's titles and texts for EL: the name as both, blanked for webqsp's CVT mediators (they neither mention nor
    own); the count of CVTs is checked against the frozen node-kind file."""
    titles, _none, n, sha = u1b.read_nodes(served, d, 0, 0)
    texts = list(titles)                          # a KB node's name is read as its text (section 2)
    n_cvt = 0
    if d == "webqsp":
        kinds = []
        with open(served / d / "nodes.jsonl", "rb") as f:
            for raw in f:
                kinds.append(json.loads(raw).get("node_kind"))
        fz = json.loads((served / d / "V1_NODE_KIND_FROZEN.json").read_text(encoding="utf-8"))
        n_cvt = sum(1 for k in kinds if k == CVT)
        if n_cvt != int(fz["counts"][CVT]) or len(kinds) != int(fz["nodes"]):
            raise SystemExit(f"webqsp: {n_cvt} CVT mediators in nodes.jsonl, the frozen file names {fz['counts'][CVT]}")
        for i, k in enumerate(kinds):
            if k == CVT:
                titles[i] = ""
                texts[i] = ""
        del kinds
    return titles, texts, n, sha, n_cvt


def kb_links(titles, texts, dense_get):
    """IS on a KB: zu2b's stat and link passes, unchanged, under KB_RULE and variant W2; then a node's match to its
    own group links nothing (the passes drop only the node itself, not the other nodes of its name)."""
    S = zu2b.Surfaces(titles, KB_RULE)
    cap, low = zu2b.stat_pass(S, texts, titles, KB_RULE)
    common = low > cap
    edges, stats = zu2b.link_pass(S, texts, titles, KB_RULE, common, dense_get, W2)
    s_, t_ = edges["W2"][0].astype(np.int64), edges["W2"][1].astype(np.int64)
    own = S.group[s_] == S.group[t_]
    return s_[~own], t_[~own], S.group, {"surfaces": len(S.surf), "common_forms": int(common.sum()),
                                         "own_group_links_dropped": int(own.sum()), **stats["W2"]}


def u2i_cue():
    """The aliases U2i's filed choice names: (candidate or None, the choice's outcome)."""
    p = U2I / "choice.json"
    if not p.exists():
        raise SystemExit("outputs/u2i/choice.json does not exist yet: U2l builds no passage dataset before it")
    ch = json.loads(p.read_text(encoding="utf-8"))
    if ch["outcome"] == "ADOPT_FOR_POOLS":
        return ch["best"], ch["outcome"]
    if ch["outcome"] == "NO_GAIN":
        return None, ch["outcome"]
    raise SystemExit(f"U2i's choice is {ch['outcome']}: REFUSED")


def u2i_inputs(d):
    pas = json.loads((U2I / d / "pass.json").read_text(encoding="utf-8"))
    g1 = json.loads((U2I / d / "gate1.json").read_text(encoding="utf-8"))
    ok = g1["status"] == "PASS" and sha_file(U2I / d / "links.npz") == pas["links_npz_sha256"] \
        and sha_file(U2I / d / "evidence.npz") == pas["evidence_npz_sha256"]
    return pas, g1, ok


def cmd_build(a):
    d = a.dataset
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    od = OUT / d
    od.mkdir(parents=True, exist_ok=True)
    rec = {"dataset": d, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze}
    if d in PASSAGE:
        cue, outcome = u2i_cue()
        pas, g1, ok = u2i_inputs(d)
        if not ok:
            write_json(od / "build.json", {**rec, "gate1": "REFUSED", "why": "U2i's pass is not PASS or its files "
                                           "are not its record's", **codes()})
            raise SystemExit(f"{d}: gate 1 REFUSED")
        with np.load(U2I / d / "links.npz") as z:
            s_, t_ = [z["w2_src"].astype(np.int64)], [z["w2_dst"].astype(np.int64)]
            if cue is not None:
                s_.append(z[f"{cue}_src"].astype(np.int64))
                t_.append(z[f"{cue}_dst"].astype(np.int64))
            r_s, r_d = z["sib_src"].astype(np.int64), z["sib_dst"].astype(np.int64)
        with np.load(U2I / d / "evidence.npz") as z:
            group = z["group"].astype(np.int64)
        n = group.size
        is_s, is_d = np.concatenate(s_), np.concatenate(t_)
        # gate 1: E1 with the sibling links is U2i's graph for the chosen cue set, array for array
        ip, col = zu2b.undirected_csr(n, np.concatenate([is_s, r_s]), np.concatenate([is_d, r_d]))
        want = zu2i.graphs_of(d, [cue if cue is not None else "W2_S"])[cue if cue is not None else "W2_S"]
        eq = bool(np.array_equal(ip, want[0]) and np.array_equal(col, want[1]))
        del ip, col, want
        rec.update({"u2i_choice": outcome, "u2i_cue": cue, "u2i_links_npz_sha256": pas["links_npz_sha256"],
                    "gate1": "PASS" if eq else "FAIL"})
        if not eq:
            write_json(od / "build.json", {**rec, **codes()})
            raise SystemExit(f"{d}: gate 1 FAIL (E1 with the siblings is not U2i's graph)")
        kb = {}
    else:
        titles, texts, n, nsha, n_cvt = kb_texts(served, d)
        D = u1d.Dense(served, d, n)
        is_s, is_d, group, kb = kb_links(titles, texts, D.get)
        del titles, texts
        r_s = r_d = np.zeros(0, np.int64)
        rec.update({"nodes_sha256": nsha, "cvt_mediators": n_cvt, "gate1": "NOT_APPLICABLE (a KB: U2i has no pass)",
                    "dense_manifest_sha256": D.man_sha})
    t1 = time.time()
    cs, cd, cw = co_mentions(is_s, group[is_d], n)
    t2 = time.time()
    sha = zu2b.save_npz(od / "el.npz", is_src=is_s.astype(np.int32), is_dst=is_d.astype(np.int32),
                        cot_src=cs.astype(np.int32), cot_dst=cd.astype(np.int32), cot_w=cw.astype(np.float32),
                        r_src=r_s.astype(np.int32), r_dst=r_d.astype(np.int32))
    m = np.unique(np.asarray(group[is_d], np.int64) * n + is_s) if is_s.size else np.zeros(0, np.int64)
    gdf = np.bincount(m // n) if m.size else np.zeros(0, np.int64)
    rec.update({"nodes": int(n), "is_links": int(is_s.size), "is_sources": int(np.unique(is_s).size),
                "mentioned_groups": int((gdf > 0).sum()), "groups_df_2_25": int(((gdf >= 2) & (gdf <= 25)).sum()),
                "co_t_pairs": int(cs.size), "r_links": int(r_s.size), "kb_scan": kb, "el_npz_sha256": sha,
                "secs": {"is": round(t1 - t0, 1), "co_t": round(t2 - t1, 1)}, **codes()})
    write_json(od / "build.json", rec)
    log(f"{d}: built EL: {is_s.size:,} IS links, {cs.size:,} CO_t pairs ({time.time() - t0:.0f} s)")
    return 0


# ── the families as CSRs ────────────────────────────────────────────────────


def pkg_family(served, d, fam):
    gm = json.loads((served / d / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))
    f = gm["families"][fam]
    p = served / d / "graph" / Path(f["file"]).name
    if sha_file(p) != f["npz_sha256"]:
        raise SystemExit(f"{d}: {p.name} is not GRAPH_MANIFEST's (gate 2)")
    with np.load(p) as z:
        return z["src"].astype(np.int64), z["dst"].astype(np.int64), f["npz_sha256"]


def families(served, d, n):
    """{family: (indptr, col)} over IS, CO_t, CO_s, R, kNN, PKG and W2_S, each undirected and collapsed."""
    b = json.loads((OUT / d / "build.json").read_text(encoding="utf-8"))
    if sha_file(OUT / d / "el.npz") != b["el_npz_sha256"]:
        raise SystemExit(f"{d}: el.npz is not its build record's")
    F, shas = {}, {"el_npz_sha256": b["el_npz_sha256"]}
    with np.load(OUT / d / "el.npz") as z:
        F["IS"] = zu2b.undirected_csr(n, z["is_src"], z["is_dst"])
        F["CO_t"] = zu2b.undirected_csr(n, z["cot_src"], z["cot_dst"])
        r = (z["r_src"], z["r_dst"])
    qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
    p = U2B / d / "pkg_structural.npz"
    if sha_file(p) != qrec["pkg_structural_npz_sha256"]:
        raise SystemExit(f"{d}: pkg_structural.npz is not its record's")
    with np.load(p) as z:
        F["PKG"] = (z["indptr"].astype(np.int64), z["col"])
    shas["pkg_structural_npz_sha256"] = qrec["pkg_structural_npz_sha256"]
    F["R"] = zu2b.undirected_csr(n, *r) if d in PASSAGE else F["PKG"]
    for fam, key in (("ner", "CO_s"), ("knn", "kNN")):
        s_, t_, sh = pkg_family(served, d, fam)
        F[key] = zu2b.undirected_csr(n, s_, t_)
        shas[f"{fam}_npz_sha256"] = sh
        del s_, t_
    if d in PASSAGE:
        ref, _e = zu2i.ref_graph(d)
        with np.load(ref) as z:
            F["W2_S"] = (z["indptr"].astype(np.int64), z["col"])
    else:
        F["W2_S"] = F["PKG"]                     # U2b's descriptive switch leaves a KB its triples
    for k, (ip, col) in F.items():
        if ip.size != n + 1:
            raise SystemExit(f"{d}: family {k} has {ip.size - 1} rows, the package {n}")
    return F, shas


def union_csr(F, comps, n):
    """The union graph of the named families, in full (for gate 3)."""
    parts_s, parts_d = [], []
    for c in dict.fromkeys(comps):
        ip, col = F[c]
        parts_s.append(np.repeat(np.arange(n, dtype=np.int64), np.diff(ip)))
        parts_d.append(col.astype(np.int64))
    return zu2b.undirected_csr(n, np.concatenate(parts_s), np.concatenate(parts_d))


def capped_union(F, comps, n, C, block=BLOCK):
    """Each row of the union graph cut to its first C neighbours (node-id order). A family's row is sorted, so the
    union's first C are among each family's first C: capped_multi with a cap <= C reads this exactly as the full union."""
    comps = list(dict.fromkeys(comps))
    if len(comps) == 1:
        return F[comps[0]]
    rows_ip = [np.zeros(1, np.int64)]
    cols = []
    tot = 0
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        keys = []
        for c in comps:
            ip, col = F[c]
            s = ip[lo:hi]
            ln = np.minimum(ip[lo + 1:hi + 1] - s, C)
            t = int(ln.sum())
            if t == 0:
                continue
            off = np.repeat(np.cumsum(ln) - ln, ln)
            idx = np.repeat(s, ln) + (np.arange(t, dtype=np.int64) - off)
            r = np.repeat(np.arange(lo, hi, dtype=np.int64), ln)
            keys.append(r * n + col[idx].astype(np.int64))
        if keys:
            k = np.unique(np.concatenate(keys))
            a, b = k // n, k % n
            cnt = np.bincount(a - lo, minlength=hi - lo)
            st = np.cumsum(cnt) - cnt
            rank = np.arange(k.size) - np.repeat(st, cnt)
            keep = rank < C
            a, b = a[keep], b[keep]
            cnt = np.bincount(a - lo, minlength=hi - lo)
        else:
            b = np.zeros(0, np.int64)
            cnt = np.zeros(hi - lo, np.int64)
        cols.append(b.astype(np.int32))
        rows_ip.append(tot + np.cumsum(cnt))
        tot += int(cnt.sum())
    return np.concatenate(rows_ip), np.concatenate(cols) if cols else np.zeros(0, np.int32)


# ── grading: U2b's section 3, each union's X from its families ──────────────


def reach_unions(F, unions, B_list, golds, qemb, dense_get, grid=zu2b.GRID, chunk=65536):
    """zu2b.reach_curves over the union graphs named in unions ({name: families}), with a union's X built as the
    union of its families' X (B's neighbours in a union of graphs are the union of B's neighbours in each). The rest
    is zu2b.reach_curves's code, line for line."""
    names = list(unions)
    need = sorted({c for u in unions.values() for c in u})
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
        xc = {c: zu2b.x_of(F[c][0], F[c][1], B) for c in need}
        xs = {}
        for name in names:
            cs = list(dict.fromkeys(unions[name]))
            xs[name] = xc[cs[0]] if len(cs) == 1 else np.unique(np.concatenate([xc[c] for c in cs]))
        del xc
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


def questions(d, carve):
    qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
    qp = U2B / d / f"questions_{carve}.npz"
    if sha_file(qp) != qrec["carves"][carve]["npz_sha256"]:
        raise SystemExit(f"{d}/{carve}: the questions file is not its record's")
    q = np.load(qp)
    base = zu2b.unragged(q["base_ptr"], q["base_col"])
    seeds = zu2b.unragged(q["seeds_ptr"], q["seeds_col"])
    golds = zu2b.unragged(q["gold_ptr"], q["gold_col"])
    B = [np.union1d(b, s) for b, s in zip(base, seeds)]
    return qrec, q, base, seeds, golds, B


def frozen_I(struct, q, qrec, base, seeds, golds):
    """zu2i's frozen I with struct in the structural family's place and the other families unchanged."""
    import m3b_pools
    con = qrec["construction"]
    if con["regime"] == "RETRIEVAL":
        return None
    fams = [str(f) for f in q["fam_order"]]
    others = {f: SimpleNamespace(indptr=q[f"{f}_indptr"], col=q[f"{f}_col"]) for f in fams if f != "structural"}
    st = SimpleNamespace(indptr=struct[0], col=struct[1])
    fam_objs = [st if f == "structural" else others[f] for f in fams]
    al = rc = size = 0.0
    for j in range(len(golds)):
        sd = np.asarray(seeds[j], np.int64)
        exp = m3b_pools.expand_hops(sd, fam_objs, con["setting"])
        pool = np.union1d(np.union1d(np.asarray(base[j], np.int64), exp), sd)
        hit = np.isin(golds[j], pool)
        al += hit.all()
        rc += hit.mean()
        size += pool.size
    nq = len(golds)
    return {"ALL": round(al / nq, 4), "recall": round(rc / nq, 4), "pool": round(size / nq, 1)}


def gate2(d, carve, qsha, got):
    """U2b's gate 2: the frozen I over PKG is U2b's filed I for PKG (every filed grade record of this carve)."""
    filed = []
    for p in sorted((U2B / d / "grade").glob(f"{carve}_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if "PKG" in r["graphs"]:
            filed.append({"record": p.name, "I": r["graphs"]["PKG"]["I"],
                          "same_questions": r.get("questions_npz_sha256") == qsha})
    ok = bool(filed) and all(f["same_questions"] and f["I"] == got for f in filed)
    return {"status": "PASS" if ok else "FAIL", "I_PKG": got, "filed": filed}


def i_cap(qrec):
    s = qrec["construction"].get("setting") or {}
    return max(int(s.get("per_seed_cap", 0) or 0), int(s.get("per_frontier_cap", 0) or 0))


def cmd_grade(a, check_full=False):
    d = a.dataset
    carve = a.carve or CARVE[d]
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    if d != "musique" and not check_full:
        g3 = OUT / "gate3.json"
        if not g3.exists() or json.loads(g3.read_text(encoding="utf-8"))["status"] != "PASS":
            raise SystemExit("gate 3 has not passed on musique: nothing else is graded")
    qrec, q, base, seeds, golds, B = questions(d, carve)
    n = int(json.loads((OUT / d / "build.json").read_text(encoding="utf-8"))["nodes"])
    F, shas = families(served, d, n)
    D = u1d.Dense(served, d, n)
    t1 = time.time()
    res = reach_unions(F, GRAPHS, B, golds, q["qemb"], D.get)
    for g in res:
        res[g]["choice_score"] = round(zu2b.choice_score(res[g]["curve"]), 4)
        res[g]["families"] = list(GRAPHS[g])
    t2 = time.time()
    C = i_cap(qrec)
    I = {}
    for name, comps in I_OF.items():
        st = capped_union(F, comps, n, C) if C > 0 else union_csr(F, comps, n)
        I[name] = frozen_I(st, q, qrec, base, seeds, golds)
        del st
    t3 = time.time()
    g2 = gate2(d, carve, qrec["carves"][carve]["npz_sha256"], I["PKG"])
    out = {"dataset": d, "carve": carve, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "gate2": g2,
           "questions": len(golds), "questions_npz_sha256": qrec["carves"][carve]["npz_sha256"],
           "dense_manifest_sha256": D.man_sha, **shas, "family_neighbours": {k: int(v[1].size) for k, v in F.items()},
           "graphs": res, "frozen_I": I, "frozen_I_cap": C,
           "secs": {"load": round(t1 - t0, 1), "reach": round(t2 - t1, 1), "I": round(t3 - t2, 1)}, **codes()}
    if check_full:
        return out, F, B, golds, q, D, qrec, base, seeds, n
    write_json(OUT / d / f"grade_{carve}.json", out)
    log(f"{d}/{carve}: graded {len(res)} graphs, gate 2 {g2['status']} ({time.time() - t0:.0f} s)")
    return 0 if g2["status"] == "PASS" else 1


def cmd_gate3(_a):
    """musique: the family-built X equals zu2b.reach_curves over the 16 union graphs built in full, graded together;
    the I over each capped union equals the I over the full union."""
    a = SimpleNamespace(dataset="musique", carve="s1sel")
    out, F, B, golds, q, D, qrec, base, seeds, n = cmd_grade(a, check_full=True)
    G = {name: union_csr(F, comps, n) for name, comps in GRAPHS.items()}
    ref = zu2b.reach_curves(G, B, golds, q["qemb"], D.get)
    bad = [g for g in GRAPHS if any(ref[g][k] != out["graphs"][g][k] for k in ref[g])]
    C = i_cap(qrec)
    bad_i = []
    for name, comps in I_OF.items():
        full = frozen_I(union_csr(F, comps, n), q, qrec, base, seeds, golds)
        if full != out["frozen_I"][name]:
            bad_i.append(name)
    status = "PASS" if not bad and not bad_i and out["gate2"]["status"] == "PASS" else "FAIL"
    write_json(OUT / "gate3.json", {"dataset": "musique", "carve": "s1sel", "declared_in": DECLARED, "status": status,
                                    "reach_mismatch": bad, "I_mismatch": bad_i, "frozen_I_cap": C, **codes()})
    log(f"gate 3 {status} (reach mismatches {bad}, I mismatches {bad_i})")
    if status == "PASS":
        write_json(OUT / "musique" / "grade_s1sel.json", out)
        log("musique/s1sel: graded (the gate's run)")
    return 0 if status == "PASS" else 1


# ── the choice (section 6) and the report ───────────────────────────────────


def choose(grades, gates_ok):
    missing = [d for d in FIVE if grades.get(d) is None or any(g not in grades[d]["graphs"] for g in GRAPHS)]
    if missing or not gates_ok:
        return {"outcome": "REFUSED", "missing": missing, "gates_ok": gates_ok}
    score = {g: round(float(np.mean([grades[d]["graphs"][g]["choice_score"] for d in FIVE])), 4) for g in GRAPHS}
    picks = [f"{f}+{PICK}" for f in FORMS]
    top = max(score[c] for c in picks)
    best = next(c for c in picks if score[c] >= top - TIE)
    per_d = {d: max(picks, key=lambda c: (grades[d]["graphs"][c]["choice_score"], -picks.index(c))) for d in FIVE}
    return {"outcome": f"CHOSEN {best.split('+')[0]}", "form": best.split("+")[0], "graph": best, "score": score,
            "per_dataset_best_upper_bound": per_d}


def cmd_choose(_a):
    grades, gates = {}, []
    g3 = OUT / "gate3.json"
    gates.append(g3.exists() and json.loads(g3.read_text(encoding="utf-8"))["status"] == "PASS")
    for d in SIX:
        b = OUT / d / "build.json"
        p = OUT / d / f"grade_{CARVE[d]}.json"
        ok = b.exists() and json.loads(b.read_text(encoding="utf-8"))["gate1"] in ("PASS",
                                                                                  "NOT_APPLICABLE (a KB: U2i has no pass)")
        r = json.loads(p.read_text(encoding="utf-8")) if (p.exists() and ok) else None
        grades[d] = r if (r is not None and r["gate2"]["status"] == "PASS") else None
    res = {"declared_in": DECLARED, "carve": "s1sel; webqsp s1eval beside", **choose(grades, all(gates)), **codes()}
    if grades.get("webqsp"):
        res["webqsp_s1eval_beside"] = {g: grades["webqsp"]["graphs"][g]["choice_score"] for g in GRAPHS}
    write_json(OUT / "choice.json", res)
    print(json.dumps({k: res[k] for k in ("outcome", "graph") if k in res}, indent=1))
    return 0


def cmd_report(_a):
    ch = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))
    lines = [f"# U2l: EL graded as U2b graded graphs ({ch['outcome']})", "",
             f"Declared in {DECLARED}. Development numbers (s1sel; webqsp's s1eval beside).", "",
             "| graph | " + " | ".join(SIX) + " | mean of five |", "| --- |" + " ---: |" * 7]
    for g in GRAPHS:
        row = []
        for d in SIX:
            p = OUT / d / f"grade_{CARVE[d]}.json"
            row.append(str(json.loads(p.read_text(encoding="utf-8"))["graphs"][g]["choice_score"]) if p.exists()
                       else "")
        lines.append(f"| {g} | " + " | ".join(row) + f" | {ch.get('score', {}).get(g, '')} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log("report written")
    return 0


# ── the question's links (section 5, beside; after the choice, on the chosen form only) ─────────────────────


def question_texts(served, d, carve):
    """The carve's questions in U2b's id order, from the train split only: the text the encoder embedded (CRAG's
    bridge: metaqa's question_plain, without the topic entity's brackets; elsewhere question)."""
    ids = json.loads((U2B / d / f"questions_{carve}.json").read_text(encoding="utf-8"))["ids"]
    qrec = json.loads((U2B / d / "questions.json").read_text(encoding="utf-8"))
    if hashlib.sha256(",".join(ids).encode("utf-8")).hexdigest() != qrec["carves"][carve]["ids_sha256"]:
        raise SystemExit(f"{d}/{carve}: the ids are not their record's")
    field = "question_plain" if d == "metaqa" else "question"
    want, got = set(ids), {}
    with open(served / d / "queries" / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["query_id"] in want:
                got[r["query_id"]] = r[field]
    miss = [i for i in ids if i not in got]
    if miss:
        raise SystemExit(f"{d}/{carve}: {len(miss)} questions are not in train.jsonl")
    return [got[i] for i in ids], field


def passage_q(served, d, texts):
    """Q(q) on a passage dataset: U2i's surfaces (its table, sha checked) under W2's gates, with the corpus's
    capitalization statistic (U2i's chunk counts)."""
    cue, _outcome = u2i_cue()
    if cue is not None:
        raise SystemExit("U2i adopted an alias cue: reading its per-cue owners for the question is not coded")
    pas = json.loads((U2I / d / "pass.json").read_text(encoding="utf-8"))
    with np.load(U2I / d / "aliases.npz") as z:
        al = [str(x) for x in z["alias"]]
    titles, _none, n, _sha = u1b.read_nodes(served, d, 0, 0)
    S = zu2i.Table(titles, {a: {} for a in al})
    del titles, al
    if S.sha != pas["table_sha256"]:
        raise SystemExit(f"{d}: the rebuilt table is not U2i's")
    cap, low = np.zeros(S.n_low, np.int64), np.zeros(S.n_low, np.int64)
    files = sorted((U2I / d / "work").glob("chunk_*.npz"))
    if len(files) != -(-n // zu2i.CHUNK):
        raise SystemExit(f"{d}: {len(files)} of U2i's chunks, {-(-n // zu2i.CHUNK)} wanted")
    for p in files:
        with np.load(p) as z:
            if str(z["table_sha"]) != S.sha:
                raise SystemExit(f"{p} was scanned with another table")
            np.add.at(cap, z["lid"], z["cap"])
            np.add.at(low, z["lid"], z["low"])
    common = low > cap
    Q = []
    for t in texts:
        hits, _st, _cm = zu2i.scan_ev(S, t)
        gs = sorted({int(S.tgroup[k]) for k, ok, _p, _n in hits if ok and S.is_title[k] and not common[S.lowid[k]]})
        Q.append(np.unique(np.concatenate([S.group_nodes(g) for g in gs])) if gs else np.zeros(0, np.int64))
    return Q


def kb_q(served, d, texts, qemb, dense_get):
    """Q(q) on a KB: section 2's surfaces (names, CVTs blanked) under W2's gates, the names' capitalization statistic,
    and the context choice by the question's vector."""
    titles, names, _n, _sha, _cvt = kb_texts(served, d)
    S = zu2b.Surfaces(titles, KB_RULE)
    cap, low = zu2b.stat_pass(S, names, titles, KB_RULE)
    del names
    common = low > cap
    Q = []
    for j, t in enumerate(texts):
        hits, _st, cm = zu2b.scan(S, t, KB_RULE)
        need = []
        gs = list(zu2b.choose_targets(S, -1, hits, common, need, W2, cm)["W2"])
        q = np.asarray(qemb[j], np.float32)
        for _i, _v, cand in need:
            best, bg = -9.0, -1
            for g in cand:
                s_ = float((dense_get(S.group_nodes(g)) @ q).max())
                if s_ > best:
                    best, bg = s_, g
            gs.append(bg)
        Q.append(np.unique(np.concatenate([S.group_nodes(g) for g in gs])) if gs else np.zeros(0, np.int64))
    return Q


def cmd_qlinks(a):
    d = a.dataset
    carve = a.carve or CARVE[d]
    ch = json.loads((OUT / "choice.json").read_text(encoding="utf-8"))
    if not ch["outcome"].startswith("CHOSEN"):
        raise SystemExit(f"U2l's choice is {ch['outcome']}: no form to read the question's links on")
    g = ch["graph"]
    t0 = time.time()
    served, freeze = u1b.package_root(False)
    qrec, q, base, seeds, golds, B = questions(d, carve)
    texts, field = question_texts(served, d, carve)
    n = int(json.loads((OUT / d / "build.json").read_text(encoding="utf-8"))["nodes"])
    D = u1d.Dense(served, d, n)
    Q = kb_q(served, d, texts, q["qemb"], D.get) if d in KB else passage_q(served, d, texts)
    t1 = time.time()
    F, shas = families(served, d, n)
    BQ = [np.union1d(b, x) for b, x in zip(B, Q)]
    res = reach_unions(F, {g: GRAPHS[g]}, BQ, golds, q["qemb"], D.get)[g]
    res["choice_score"] = round(zu2b.choice_score(res["curve"]), 4)
    sizes = np.asarray([x.size for x in Q])
    out = {"dataset": d, "carve": carve, "declared_in": DECLARED, "freeze_RECORD_SHA256": freeze, "graph": g,
           "question_field": field, "questions": len(golds), **shas,
           "Q": {"mean": round(float(sizes.mean()), 2), "median": float(np.median(sizes)),
                 "nonempty": round(float((sizes > 0).mean()), 4), "max": int(sizes.max())},
           "reach_B_union_Q": res, "secs": {"Q": round(t1 - t0, 1), "reach": round(time.time() - t1, 1)},
           "decides": "nothing (section 5, beside)", **codes()}
    write_json(OUT / d / f"qlinks_{carve}.json", out)
    log(f"{d}/{carve}: the question's links on {g} ({time.time() - t0:.0f} s)")
    return 0


# ── the selftest (gate: section 7, item 1) ──────────────────────────────────


def _selftest(_a=None):
    rng = np.random.default_rng(0)
    # CO_t against a brute-force count
    n, ng = 40, 12
    src = rng.integers(0, n, 300)
    grp = rng.integers(0, ng, 300)
    a_, b_, w_ = co_mentions(src, grp, n, 2, 5)
    men = {}
    for u, e in zip(src.tolist(), grp.tolist()):
        men.setdefault(e, set()).add(u)
    bf = {}
    for e, us in men.items():
        if 2 <= len(us) <= 5:
            for x in us:
                for y in us:
                    if x < y:
                        bf[(x, y)] = bf.get((x, y), 0.0) + 1.0 / len(us)
    got = {(int(x), int(y)): float(w) for x, y, w in zip(a_, b_, w_)}
    assert got.keys() == bf.keys() and all(abs(got[k] - bf[k]) < 1e-12 for k in bf), "CO_t"
    # the KB scan: a name holding another name links to it; its own group links nothing; a blanked CVT neither
    titles = ["Harry Potter", "Harry Potter and the Goblet of Fire", "", "Goblet", "Goblet"]
    texts = list(titles)
    vec = rng.standard_normal((len(titles), 8)).astype(np.float32)
    s_, t_, gr, _st = kb_links(titles, texts, lambda ids: vec[np.asarray(ids)])
    pairs = sorted(zip(s_.tolist(), t_.tolist()))
    assert (1, 0) in pairs and (1, 3) in pairs and (1, 4) in pairs, pairs
    assert all(s != t for s, t in pairs) and all(2 not in p for p in pairs), pairs
    assert (3, 4) not in pairs and (4, 3) not in pairs and _st["own_group_links_dropped"] == 2, (pairs, _st)
    # the same texts under zu2b's own passes give the same links, less the own-group ones
    S = zu2b.Surfaces(titles, KB_RULE)
    cap, low = zu2b.stat_pass(S, texts, titles, KB_RULE)
    e2, _ = zu2b.link_pass(S, texts, titles, KB_RULE, low > cap, lambda ids: vec[np.asarray(ids)], W2)
    raw = sorted(zip(e2["W2"][0].tolist(), e2["W2"][1].tolist()))
    assert [p for p in raw if S.group[p[0]] != S.group[p[1]]] == pairs and (3, 4) in raw, raw
    # capped_union = the full union cut to its first C per row; reach_unions = zu2b.reach_curves on the unions
    n = 300
    F = {}
    for k in ("A", "B", "C"):
        m = int(rng.integers(200, 900))
        F[k] = zu2b.undirected_csr(n, rng.integers(0, n, m), rng.integers(0, n, m))
    for comps in (("A",), ("A", "B"), ("A", "B", "C")):
        ip, col = union_csr(F, comps, n)
        for C in (1, 3, 25):
            cip, ccol = capped_union(F, comps, n, C, block=37)
            for r in range(n):
                assert np.array_equal(ccol[cip[r]:cip[r + 1]][:C], col[ip[r]:ip[r + 1]][:C]), (comps, C, r)
    unions = {"a": ("A",), "ab": ("A", "B"), "abc": ("A", "B", "C"), "bc": ("B", "C")}
    Bl = [np.unique(rng.integers(0, n, int(rng.integers(1, 9)))) for _ in range(40)]
    gl = [np.unique(rng.integers(0, n, int(rng.integers(1, 4)))) for _ in range(40)]
    qe = rng.standard_normal((40, 8)).astype(np.float32)
    dv = rng.standard_normal((n, 8)).astype(np.float32)
    get = (lambda ids: dv[np.asarray(ids)])
    mine = reach_unions(F, unions, Bl, gl, qe, get)
    ref = zu2b.reach_curves({u: union_csr(F, c, n) for u, c in unions.items()}, Bl, gl, qe, get)
    assert mine == ref, "reach_unions"
    print("selftest OK")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    ap.add_argument("--dataset")
    ap.add_argument("--carve")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest(a)
    fn = {"build": cmd_build, "gate3": cmd_gate3, "grade": cmd_grade, "choose": cmd_choose,
          "report": cmd_report, "qlinks": cmd_qlinks}.get(a.cmd)
    if fn is None:
        raise SystemExit("build | gate3 | grade | choose | report | qlinks | --selftest")
    if a.cmd in ("build", "grade", "qlinks") and a.dataset not in SIX:
        raise SystemExit(f"--dataset must be one of {SIX}")
    return fn(a)


if __name__ == "__main__":
    sys.exit(main())
