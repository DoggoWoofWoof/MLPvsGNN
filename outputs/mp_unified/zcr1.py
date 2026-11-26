"""CR1 (docs/CR1_CHAIN_RECREATION.md): how far the gold slots are from linked on the e3v layout B0, and which order finds the next hop.
Gold-anchored and pool-free; no model reads anything, nothing trains, no LLM, no spaCy, no GPU. Gold positions and MuSiQue's decompositions are the
grading only. zul2p, zul2c, zul2, zul1, zap4 and zenc are imported, never edited. Outputs only under outputs/cr1/. Arrays are never written.

    python outputs/mp_unified/zcr1.py selftest
    python outputs/mp_unified/zcr1.py ledger --ds D [--host] [--threads 1] [--limit N]    # D's ledger (squad: its n/a record)
    python outputs/mp_unified/zcr1.py report                                              # the tables, the calls A and B, the expectations X1-X8
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
sys.dont_write_bytecode = True

import zul2p as P  # noqa: E402  (first: imports zul2c -> zul2u -> zul2 -> zmhb -> zenc, so the BLAS threads from --threads are set before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import zipfile  # noqa: E402

import numpy as np  # noqa: E402
import zenc as E  # noqa: E402

C, L2, UL, Z, AP, MB = P.C, P.L2, P.UL, P.Z, P.AP, P.MB
DECLARED = "docs/CR1_CHAIN_RECREATION.md"
OUT = ROOT / "outputs" / "cr1"
DS7 = P.DS7
KS = (10, 30, 100)                         # rank cuts read (30 is GP1's)
CALL_K = 30
ORDERS = ("pd", "fq", "rq")                # the pair's own dot, the flat question cosine, the residual question cosine
BLOCK = 1 << 14                            # neighbour rows scored per block
MIN_T = P.MIN_T
log = MB.log
EMPTY = np.zeros(0, np.int64)


def stop(msg):
    raise SystemExit(f"[zcr1] STOP: {msg}")


def rel(p):
    return P.rel(p)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=False, default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n", encoding="utf-8")
    os.replace(tmp, p)


# ── the neighbour lists of the anchors ───────────────────────────────────────


def b0_keys_at(F, srk, am, n):
    """B0's pairs (sorted unique keys) with an end in the anchor mask am: S (SRK only), W2_S, F0T and the e3v kNN."""
    parts = [P.incident(F[x], am, n) for x in ("W2_S", "F0T", "Ke")]
    if srk:
        parts.append(P.incident(F["S"], am, n))
    return np.unique(np.concatenate(parts)) if parts else EMPTY


class Nbrs:
    """The neighbour lists of the anchor nodes from their incident pairs: one sorted unique array per anchor with at least one pair."""

    def __init__(self, keys, am, n):
        lo, hi = keys // n, keys % n
        m1, m2 = am[lo], am[hi]
        src = np.concatenate([lo[m1], hi[m2]])
        dst = np.concatenate([hi[m1], lo[m2]])
        o = np.lexsort((dst, src))
        src, dst = src[o], dst[o]
        self.nodes, start = np.unique(src, return_index=True)
        self.ptr = np.append(start, src.size).astype(np.int64)
        self.dst = dst
        self.n = n

    def of(self, x):
        i = int(np.searchsorted(self.nodes, x))
        if i < self.nodes.size and self.nodes[i] == x:
            return self.dst[self.ptr[i]:self.ptr[i + 1]]
        return EMPTY

    def deg(self, nodes):
        i = np.minimum(np.searchsorted(self.nodes, nodes), max(self.nodes.size - 1, 0))
        ok = (self.nodes[i] == nodes) if self.nodes.size else np.zeros(nodes.size, bool)
        return np.where(ok, self.ptr[i + 1] - self.ptr[i], 0)


def has(arr, x):
    i = int(np.searchsorted(arr, x))
    return i < arr.size and int(arr[i]) == int(x)


# ── Part A: distance classes ────────────────────────────────────────────────


def classify(nb, a, b, golds_of=None):
    """Per slot (a, b): cls 0 = D (linked), 1 = T (a node is linked to both), 2 = N; m = the shared nodes' count (T only), via = a shared node is
    a gold of the slot's question (golds_of[i]). a and b are node arrays of the slots."""
    k = a.size
    cls = np.full(k, 2, np.int8)
    m = np.zeros(k, np.int64)
    via = np.zeros(k, bool)
    for i in range(k):
        na, nbb = nb.of(int(a[i])), nb.of(int(b[i]))
        if has(na, b[i]):
            cls[i] = 0
            continue
        if na.size and nbb.size:
            com = np.intersect1d(na, nbb, assume_unique=True)
            com = com[(com != a[i]) & (com != b[i])]
            if com.size:
                cls[i], m[i] = 1, com.size
                if golds_of is not None:
                    via[i] = bool(np.isin(com, golds_of[i]).any())
    return cls, m, via


def control_slots(qi, a, b):
    """The control of each slot: (a, b') with b' an end of the next question's slot (in order of the questions with a slot, cyclic): the same
    position modulo that question's slot count. Returns (index of the kept slots, the control's b); a control with b' in (a, b) is dropped."""
    qs = np.unique(qi)
    if qs.size < 2:
        return EMPTY, EMPTY
    start = np.searchsorted(qi, qs)
    cnt = np.diff(np.append(start, qi.size))
    pos = np.searchsorted(qs, qi)
    nxt = (pos + 1) % qs.size
    first_in_q = np.arange(qi.size) - start[pos]
    j = start[nxt] + (first_in_q % cnt[nxt])
    bp = b[j]
    keep = (bp != a) & (bp != b) & (qs[nxt] != qi)
    return np.flatnonzero(keep), bp[keep]


def boot_share(num, den):
    r = AP.boot(lambda s: AP.ratio(s["x"], s["n"]), {"x": num, "n": den})
    return None if r is None else {"pt": round(100 * r["pt"], 3), "ci": [round(100 * r["ci"][0], 3), round(100 * r["ci"][1], 3)]}


def boot_diff(x, y, n):
    r = AP.boot(lambda s: AP.ratio(s["x"] - s["y"], s["n"]), {"x": x, "y": y, "n": n})
    if r is None:
        return None
    out = {"pt": round(100 * r["pt"], 3), "ci": [round(100 * r["ci"][0], 3), round(100 * r["ci"][1], 3)]}
    out["call"] = AP.call(r)
    return out


def part_a(nq, qi, a, b, n, nb, golds, hop=None, with_control=True):
    """Part A of one cell: D / T / N shares of the slots, CC1 and CC2 of the questions (all and by hop), |M|, via gold, and the control."""
    gl = [golds[int(q)] for q in qi]
    cls, m, via = classify(nb, a, b, gl)
    qs = np.flatnonzero(np.bincount(qi, minlength=nq))
    ns = np.bincount(qi, minlength=nq)[qs].astype(float)
    cnt = lambda c: np.bincount(qi, weights=(cls == c).astype(float), minlength=nq)[qs]       # noqa: E731
    nD, nT, nN = cnt(0), cnt(1), cnt(2)
    cc1, cc2 = (nD == ns).astype(float), (nN == 0).astype(float)
    one = np.ones(qs.size)
    rec = {"questions": int(qs.size), "slots": int(qi.size),
           "D": {"n": int(nD.sum()), "share": boot_share(nD, ns)}, "T": {"n": int(nT.sum()), "share": boot_share(nT, ns)},
           "N": {"n": int(nN.sum()), "share": boot_share(nN, ns)},
           "CC1": {"n": int(cc1.sum()), "share": boot_share(cc1, one)}, "CC2": {"n": int(cc2.sum()), "share": boot_share(cc2, one)}}
    tm = m[cls == 1]
    rec["M"] = {"t_slots": int(tm.size), "median": float(np.median(tm)) if tm.size else None, "p90": float(np.percentile(tm, 90)) if tm.size else None,
                "share_ge_10": round(float((tm >= 10).mean()), 4) if tm.size else None, "via_gold_share_of_T": round(float(via[cls == 1].mean()), 4) if tm.size else None}
    if hop is not None:
        rec["by_hop"] = {}
        for h in P.HOPS:
            sel = hop[qs] == h
            if sel.any():
                rec["by_hop"][str(h)] = {"questions": int(sel.sum()), "CC1": boot_share(cc1[sel], one[sel]), "CC2": boot_share(cc2[sel], one[sel])}
    if with_control:
        keep, bp = control_slots(qi, a, b)
        rec["control"] = {"slots": int(keep.size), "dropped": int(qi.size - keep.size)}
        if keep.size:
            qc, ac = qi[keep], a[keep]
            clsc, _mc, _vc = classify(nb, ac, bp, None)
            qsc = np.flatnonzero(np.bincount(qc, minlength=nq))
            nsc = np.bincount(qc, minlength=nq)[qsc].astype(float)
            cc = lambda c: np.bincount(qc, weights=(clsc == c).astype(float), minlength=nq)[qsc]   # noqa: E731
            cD, cT, cN = cc(0), cc(1), cc(2)
            rec["control"].update({"D": boot_share(cD, nsc), "T": boot_share(cT, nsc), "N": boot_share(cN, nsc)})
            c1, c2 = (cD == nsc).astype(float), (cN == 0).astype(float)
            rec["control"]["CC1"], rec["control"]["CC2"] = boot_share(c1, np.ones(qsc.size)), boot_share(c2, np.ones(qsc.size))
            both = np.intersect1d(qs, qsc)
            if both.size:
                ia, ib = np.searchsorted(qs, both), np.searchsorted(qsc, both)
                rec["CC2_minus_control"] = boot_diff(cc2[ia], c2[ib], np.ones(both.size))
                rec["CC2_minus_control"]["questions"] = int(both.size)
    return rec, cls


# ── Part B: the order of the next hop ───────────────────────────────────────


def source_target(fa, fb, a, b):
    """The source is the end with the larger question cosine (ties: the lower node id)."""
    sw = (fb > fa) | ((fb == fa) & (b < a))
    return np.where(sw, b, a), np.where(sw, a, b)


def neighbour_scores(rows, q, s, ns):
    """The three scores of every neighbour of s: pd = row(s) . row(x), fq = q . row(x), rq = fq - (q . sh)(sh . row(x)), sh = row(s) / |row(s)|."""
    rs = rows.take([s])[0]
    nr = float(np.linalg.norm(rs))
    sh = rs / nr if nr > 0 else rs
    c = np.float32(q @ sh)
    pd = np.empty(ns.size, np.float32)
    fq = np.empty(ns.size, np.float32)
    rq = np.empty(ns.size, np.float32)
    for lo in range(0, ns.size, BLOCK):
        R = rows.take(ns[lo:lo + BLOCK])
        pd[lo:lo + BLOCK] = R @ rs
        f = R @ q
        fq[lo:lo + BLOCK] = f
        rq[lo:lo + BLOCK] = f - c * (R @ sh)
    return pd, fq, rq


def ranks_of(scores, ns, t):
    """1 + the number of neighbours with a strictly larger score, per order (ties share the rank)."""
    i = int(np.searchsorted(ns, t))
    if i >= ns.size or ns[i] != t:
        stop(f"the target {t} is not a neighbour of its source (a D slot)")
    return [int(1 + (sc > sc[i]).sum()) for sc in scores]


def part_b_question(rows, q, a, b, nb):
    """One question's D slots: per slot (source, target, |N(s)|, the three ranks). The scores of a source are computed once."""
    ends = np.unique(np.concatenate([a, b]))
    fe = rows.take(ends) @ q
    fa, fb = fe[np.searchsorted(ends, a)], fe[np.searchsorted(ends, b)]
    s, t = source_target(fa, fb, a, b)
    cache, out = {}, []
    for i in range(a.size):
        si = int(s[i])
        ns = nb.of(si)
        if si not in cache:
            cache[si] = neighbour_scores(rows, q, si, ns)
        out.append((si, int(t[i]), int(ns.size), *ranks_of(cache[si], ns, int(t[i]))))
    return out


def part_b(nq, qi, a, b, cls, nb, rows, qrow, qrows):
    """Part B of one cell: the target's rank among the source's neighbours, over the D slots, by the three orders."""
    dsl = np.flatnonzero(cls == 0)
    if dsl.size == 0:
        return {"d_slots": 0}, None
    qd, ad, bd = qi[dsl], a[dsl], b[dsl]
    res = np.zeros((dsl.size, 5), np.int64)                     # |N(s)|, rank pd, rank fq, rank rq, source
    start = np.flatnonzero(np.r_[True, qd[1:] != qd[:-1]])
    end = np.append(start[1:], qd.size)
    for lo, hi in zip(start, end):
        q = qrows.take([qrow[int(qd[lo])]])[0]
        for k, (s, _t, dg, r1, r2, r3) in enumerate(part_b_question(rows, q, ad[lo:hi], bd[lo:hi], nb)):
            res[lo + k] = (dg, r1, r2, r3, s)
    qs = np.flatnonzero(np.bincount(qd, minlength=nq))
    nsl = np.bincount(qd, minlength=nq)[qs].astype(float)
    deg = res[:, 0].astype(float)
    rec = {"d_slots": int(dsl.size), "questions": int(qs.size), "sources": int(np.unique(np.stack([qd, res[:, 4]]), axis=1).shape[1]),
           "degree_median": float(np.median(deg)), "degree_p90": float(np.percentile(deg, 90)), "orders": {}}
    hits = {}
    for oi, o in enumerate(ORDERS):
        r = res[:, 1 + oi]
        ro = {"median_rank_share": round(float(np.median(r / deg)), 5)}
        for K in KS:
            h = (r <= K).astype(float)
            hq = np.bincount(qd, weights=h, minlength=nq)[qs]
            hits[(o, K)] = hq
            ch = np.bincount(qd, weights=np.minimum(1.0, K / deg), minlength=nq)[qs]
            ro[f"top{K}"] = {"hits": int(h.sum()), "share": boot_share(hq, nsl), "chance": round(100 * float(ch.sum() / nsl.sum()), 3)}
        rec["orders"][o] = ro
    rec["paired"] = {}
    for K in KS:
        rec["paired"][f"top{K}"] = {"rq_minus_fq": boot_diff(hits[("rq", K)], hits[("fq", K)], nsl),
                                    "fq_minus_pd": boot_diff(hits[("fq", K)], hits[("pd", K)], nsl),
                                    "rq_minus_pd": boot_diff(hits[("rq", K)], hits[("pd", K)], nsl)}
    return rec, res


# ── data ─────────────────────────────────────────────────────────────────────


def carve_inputs(D, d, cv, host, served):
    """(golds per carve question, the question rows in the e3v 'queries' vectors) of one carve."""
    if d == "multihoprag":
        import zul2m as M2
        root, nodes, _n, _sha = M2.head()
        golds, split, _types = M2.golds_splits(root, nodes)
        idx = np.flatnonzero(split == M2.CARVE_SPLIT[cv])
        return [golds[j] for j in idx], idx.astype(np.int64)
    ids, golds, _sha = UL.carve_questions(d, cv)
    qids = json.loads((served / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {str(q): j for j, q in enumerate(qids)}
    miss = [x for x in ids if str(x) not in qmap]
    if miss:
        stop(f"G-C5: {d}/{cv}: {len(miss)} carve ids are not in the queries' ids (e.g. {miss[:3]})")
    return golds, np.asarray([qmap[str(x)] for x in ids], np.int64)


def query_rows(served, d):
    return C.Rows([np.load(p, mmap_mode="r") for p in E.vec_shards(served, "e3v", d, "queries")])


KB_ANSWER_GOLDS = ("metaqa", "webqsp")                                            # amendment A1: the golds are answer entities, not passages


def npz_member(path, name):
    """A read-only array of a .npz member: memory-mapped when the member is stored uncompressed, else loaded whole (the first-stage dense lists are stored)."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
    if info.compress_type != zipfile.ZIP_STORED:
        with np.load(path) as z:
            return np.asarray(z[name])
    with open(path, "rb") as f:
        f.seek(info.header_offset)
        h = f.read(30)
        k, m = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
        f.seek(info.header_offset + 30 + k + m)
        ver = np.lib.format.read_magic(f)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0) else np.lib.format.read_array_header_2_0(f))
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def gate_rows_aligned(rows, qrows, qrow, golds, n, seed=0, top=None):
    """G-C5: over the first 200 questions the question cosine of the reference nodes is above that of 1,000 fixed random nodes in at least 0.9 of the comparisons;
    the question rows and a sample of node rows have norm 1 within 1e-3. Reference nodes: the golds; amendment A1, where the golds are answer entities
    (`top` given: the stored first-stage dense list's ids by question row), the five first-ranked dense results (the golds' share is recorded beside)."""
    rng = np.random.default_rng(seed)
    rnd = np.sort(rng.choice(n, size=min(1000, n), replace=False))
    rr = rows.take(rnd)
    nrm = np.linalg.norm(rr[:200], axis=1)
    if np.abs(nrm - 1).max() > 1e-3:
        stop(f"G-C5: node rows are not unit norm (max deviation {np.abs(nrm - 1).max():.4f})")
    win = tot = tw = tt = 0
    for j in range(min(200, len(golds))):
        q = qrows.take([qrow[j]])[0]
        if abs(float(np.linalg.norm(q)) - 1) > 1e-3:
            stop(f"G-C5: question row {j} is not unit norm ({np.linalg.norm(q):.4f})")
        fr = rr @ q
        g = np.asarray(golds[j], np.int64)
        if g.size:
            fg = rows.take(g) @ q
            win += int((fg[:, None] > fr[None, :]).sum())
            tot += int(fg.size * fr.size)
        if top is not None:
            t5 = np.asarray(top[int(qrow[j])][:5], np.int64)
            f5 = rows.take(t5) @ q
            tw += int((f5[:, None] > fr[None, :]).sum())
            tt += int(f5.size * fr.size)
    share = win / max(tot, 1)
    if top is None:
        if share < 0.9:
            stop(f"G-C5: the golds' question cosine is above random nodes' in {share:.3f} of the comparisons, below 0.9 (the question rows are not aligned)")
        return round(share, 4)
    s5 = tw / max(tt, 1)
    if s5 < 0.9:
        stop(f"G-C5: the dense top five's question cosine is above random nodes' in {s5:.3f} of the comparisons, below 0.9 (the question rows are not aligned)")
    return {"basis": "dense_top5", "share": round(s5, 4), "golds_share": round(share, 4)}


def gate_ranks_loop(rows, qrows, qrow, qi, a, b, cls, nb, k=1000, max_deg=2000):
    """G-C6: on up to k D slots (sources of at most max_deg neighbours) the blocked ranks equal a per-neighbour loop's, exactly."""
    dsl = np.flatnonzero(cls == 0)[:k * 3]
    done = bad = 0
    for i in dsl:
        if done >= k:
            break
        q = qrows.take([qrow[int(qi[i])]])[0]
        s, t = (int(a[i]), int(b[i]))
        fe = rows.take([s, t]) @ q
        s, t = source_target(fe[0], fe[1], np.int64(s), np.int64(t))
        s, t = int(s), int(t)
        ns = nb.of(s)
        if ns.size > max_deg:
            continue
        blocked = ranks_of(neighbour_scores(rows, q, s, ns), ns, t)
        rs = rows.take([s])[0]
        sh = rs / np.linalg.norm(rs)
        c = np.float32(q @ sh)
        sc = [[], [], []]
        for x in ns.tolist():
            rx = rows.take([x])[0]
            f = np.float32(rx @ q)
            sc[0].append(np.float32(rx @ rs))
            sc[1].append(f)
            sc[2].append(np.float32(f - c * np.float32(rx @ sh)))
        loop = ranks_of([np.asarray(z, np.float32) for z in sc], ns, t)
        done += 1
        bad += blocked != loop
    if bad:
        stop(f"G-C6: {bad} of {done} slots' ranks differ between the blocked and the per-neighbour computation")
    return done


# ── the ledger ───────────────────────────────────────────────────────────────


def ul2p_cell(d, v, cv, kind):
    p = P.OUT / d / "ledger.json"
    if not p.is_file():
        stop(f"G-C1: {rel(p)} is not here")
    c = json.loads(p.read_text(encoding="utf-8"))["versions"][v]["cells"][cv][kind]
    return int(c["units"]["any"]["covered"]), int(c["slots"]), int(c["questions"])


def cmd_ledger(a):
    d, t0 = a.ds, time.time()
    out = OUT / d / ("ledger_limit.json" if a.limit else "ledger.json")
    if d == "squad":
        wjson(out, {"dataset": d, "declared_in": DECLARED, "status": "n/a", "reason": "one gold per question: no gold pair, so no slot (section 3)"})
        log("squad: n/a record written")
        return 0
    D = P.load_mhr() if d == "multihoprag" else P.load_host(d, a.host)
    P.gates(D, D.g)                                                                    # G-C7: B0's size and the cells are the grade's
    if d == "multihoprag":
        import zul2m as M2
        served = M2.head()[0]
    else:
        served = L2.dataset_head(types.SimpleNamespace(ds=d, host=a.host))[0]
    rows = C.Rows.of(served, d, D.n)
    qrows = query_rows(served, d)
    am = P.anchor_mask(D)
    rec = {"dataset": d, "declared_in": DECLARED, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "sources": D.src, "limit": a.limit or None,
           "zcr1_sha256": script_sha256(), "zul2p_sha256": P.script_sha256(), "versions": {}, "gates": {"G-C1": {}, "G-C2": {}, "G-C5": {}, "G-C6": {}}}
    carve = {}
    top = npz_member(served / d / "retrieval_cache" / "dense_top1000.npz", "ids") if d in KB_ANSWER_GOLDS else None
    for cv in sorted({c for c, _k in D.cells}):
        golds, qrow = carve_inputs(D, d, cv, a.host, served)
        carve[cv] = (golds, qrow)
        rec["gates"]["G-C5"][cv] = gate_rows_aligned(rows, qrows, qrow, golds, D.n, top=top)
    log(f"{d}: gates G-P pass, question rows aligned {rec['gates']['G-C5']} ({time.time() - t0:.0f} s)")
    for v in D.vers:
        keys = b0_keys_at(D.F, v == "SRK", am, D.n)
        nb = Nbrs(keys, am, D.n)
        an = types.SimpleNamespace(nodes=np.flatnonzero(am), n=D.n)
        if not np.array_equal(P.node_deg(an, keys), nb.deg(an.nodes).astype(float)):
            stop(f"G-C2: {d}/{v}: the neighbour lists' degrees are not the incident pairs' degrees")
        rec["gates"]["G-C2"][v] = {"anchors": int(an.nodes.size), "pairs_at_anchors": int(keys.size)}
        rec["versions"][v] = {}
        for (cv, kind), cell in sorted(D.cells.items()):
            golds, qrow = carve[cv]
            qi, sk, nq = cell["qi"], cell["sk"], cell["nq"]
            if a.limit:
                m = qi < a.limit
                qi, sk = qi[m], sk[m]
            slots_a, slots_b = sk // D.n, sk % D.n
            ra, cls = part_a(nq, qi, slots_a, slots_b, D.n, nb, golds, cell["hop"] if (d == "musique" and kind == "CH") else None)
            if not a.limit:
                cov, nsl, nqq = ul2p_cell(d, v, cv, kind)
                if int(ra["D"]["n"]) != cov or int(qi.size) != nsl or int(ra["questions"]) != nqq:
                    stop(f"G-C1: {d}/{v}/{cv}/{kind}: D {ra['D']['n']}, slots {qi.size}, questions {ra['questions']}; UL2P's any-link covered {cov}, slots {nsl}, questions {nqq}")
                rec["gates"]["G-C1"][f"{v}/{cv}/{kind}"] = {"D": cov, "slots": nsl}
            ga = int(np.flatnonzero(cls == 0).size)
            rb, _res = part_b(nq, qi, slots_a, slots_b, cls, nb, rows, qrow, qrows)
            if cls.size and ga and (cv, kind) == (sorted({c for c, _k in D.cells})[-1], sorted({k for _c, k in D.cells})[0]) and v == D.vers[0]:
                rec["gates"]["G-C6"][f"{v}/{cv}/{kind}"] = gate_ranks_loop(rows, qrows, qrow, qi, slots_a, slots_b, cls, nb)
            rec["versions"][v][f"{cv}/{kind}"] = {"A": ra, "B": rb}
            log(f"{d}/{v}/{cv}/{kind}: slots {qi.size}, D {ra['D']['share']['pt']}, T {ra['T']['share']['pt']}, N {ra['N']['share']['pt']}, "
                f"CC2 {ra['CC2']['share']['pt']}, D slots ranked {rb['d_slots']} ({time.time() - t0:.0f} s)")
            wjson(out, {**rec, "seconds": round(time.time() - t0, 1), "partial": True})
    rec["seconds"] = round(time.time() - t0, 1)
    rec["memory"] = {"peak_working_set_gb": C.peak_gb() if hasattr(C, "peak_gb") else None}
    wjson(out, rec)
    log(f"{d}: {rel(out)} written ({rec['seconds']} s)")
    return 0


# ── the report: tables, the calls A and B, the expectations ──────────────────


def pc(x):
    return "-" if x is None else f"{x['pt']:.2f} [{x['ci'][0]:.2f},{x['ci'][1]:.2f}]"


def cmd_report(_a):
    recs = {}
    for d in DS7:
        p = OUT / d / "ledger.json"
        if p.is_file():
            recs[d] = json.loads(p.read_text(encoding="utf-8"))
    L = ["# CR1 report (generated by zcr1.py report; numbers from outputs/cr1/<d>/ledger.json)", ""]
    L += ["## Part A: distance of the slots on B0 (percent of slots; CC = percent of questions)", "",
          "| dataset | ver | carve/kind | slots | D | T | N | CC1 | CC2 | control D+T | median M | M>=10 | T via gold | CC2 minus control |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in recs.items():
        for v, cells in r.get("versions", {}).items():
            for ck, c in cells.items():
                A = c["A"]
                ct = A.get("control", {})
                dt = (ct["D"]["pt"] + ct["T"]["pt"]) if ct.get("D") and ct.get("T") else None
                L.append(f"| {d} | {v} | {ck} | {A['slots']} | {pc(A['D']['share'])} | {pc(A['T']['share'])} | {pc(A['N']['share'])} | {pc(A['CC1']['share'])} | {pc(A['CC2']['share'])} | "
                         f"{'-' if dt is None else f'{dt:.2f}'} | {A['M']['median']} | {A['M']['share_ge_10']} | {A['M']['via_gold_share_of_T']} | {pc(A.get('CC2_minus_control'))} |")
    hops = [(d, v, ck, c["A"]["by_hop"]) for d, r in recs.items() for v, cells in r.get("versions", {}).items() for ck, c in cells.items() if c["A"].get("by_hop")]
    if hops:                                                                          # amendment A2: the ledger's by-hop reads, printed
        L += ["", "## Part A by hop (MuSiQue CH; percent of the hop count's questions)", "", "| dataset | ver | carve/kind | hops | questions | CC1 | CC2 |", "|---|---|---|---|---|---|---|"]
        for d, v, ck, bh in hops:
            for h in sorted(bh, key=int):
                L.append(f"| {d} | {v} | {ck} | {h} | {bh[h]['questions']} | {pc(bh[h]['CC1'])} | {pc(bh[h]['CC2'])} |")
    L += ["", "## Part B: the target's rank among the source's neighbours (D slots; percent with rank <= K; chance in parentheses)", "",
          "| dataset | ver | carve/kind | D slots | median degree | pd@30 | fq@30 | rq@30 | rq-fq@30 | fq-pd@30 | rq-fq@10 | rq-fq@100 | median rank share pd / fq / rq |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d, r in recs.items():
        for v, cells in r.get("versions", {}).items():
            for ck, c in cells.items():
                B = c["B"]
                if not B.get("d_slots"):
                    L.append(f"| {d} | {v} | {ck} | 0 | - | - | - | - | - | - | - | - | - |")
                    continue
                o = B["orders"]
                pr = B["paired"]
                fm = lambda x: "-" if x is None else f"{x['pt']:+.2f} [{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}] {x['call']}"      # noqa: E731
                L.append(f"| {d} | {v} | {ck} | {B['d_slots']} | {B['degree_median']:.0f} | {pc(o['pd']['top30']['share'])} ({o['pd']['top30']['chance']}) | {pc(o['fq']['top30']['share'])} | "
                         f"{pc(o['rq']['top30']['share'])} | {fm(pr['top30']['rq_minus_fq'])} | {fm(pr['top30']['fq_minus_pd'])} | {fm(pr['top10']['rq_minus_fq'])} | {fm(pr['top100']['rq_minus_fq'])} | "
                         f"{o['pd']['median_rank_share']} / {o['fq']['median_rank_share']} / {o['rq']['median_rank_share']} |")
    calls = calls_of(recs)
    L += ["", "## Calls", ""] + [f"- {k}: {v}" for k, v in calls.items()]
    (OUT).mkdir(parents=True, exist_ok=True)
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    wjson(OUT / "report.json", {"declared_in": DECLARED, "datasets": sorted(recs), "calls": calls, "zcr1_sha256": script_sha256()})
    print("\n".join(L))
    return 0


def decide_cells(recs):
    """The decide carve's cells per dataset and version: [(dataset, version, cell key, record)], one per dataset (the first version, the first kind that has slots)
    for the dataset-level calls, plus MuSiQue's CH read separately."""
    out = {}
    for d, r in recs.items():
        for v, cells in r.get("versions", {}).items():
            for ck, c in cells.items():
                if ck.startswith("s1eval/"):
                    out.setdefault(d, []).append((v, ck, c))
    return out


def calls_of(recs):
    cells = decide_cells(recs)
    calls = {}
    gp = {d: [(v, ck, c) for v, ck, c in lst if ck.endswith("/GP")] for d, lst in cells.items()}
    # Part B: per dataset the first version's GP cell (MuSiQue: CH, the oracle's own slots)
    pick = {}
    for d, lst in cells.items():
        kinds = [x for x in lst if x[1].endswith("/CH")] if d == "musique" else gp[d]
        if kinds:
            pick[d] = kinds[0]
    for name in ("rq_minus_fq", "fq_minus_pd"):
        res = {}
        for d, (v, ck, c) in pick.items():
            B = c["B"]
            if B.get("d_slots", 0) < MIN_T:
                res[d] = "NOT_SCORABLE"
                continue
            res[d] = B["paired"][f"top{CALL_K}"][name]["call"]
        ab = sum(1 for x in res.values() if x == "ABOVE")
        be = sum(1 for x in res.values() if x == "BELOW")
        calls[f"B {name} @ {CALL_K}"] = (f"{'UNIVERSAL' if ab >= 5 and be == 0 else 'NOT_UNIVERSAL'} (ABOVE on {ab}, BELOW on {be} of {len(res)}); " + ", ".join(f"{d} {x}" for d, x in res.items()))
    ca = {}
    for d, lst in cells.items():
        for v, ck, c in lst:
            x = c["A"].get("CC2_minus_control")
            if x:
                ca[f"{d}/{v}/{ck}"] = f"{x['call']} ({x['pt']:+.2f} points)"
    calls["A CHAIN_WITHIN_TWO (CC2 minus control)"] = "; ".join(f"{k} {v}" for k, v in ca.items()) or "no cell"
    return calls


# ── self-test (G-C3) ─────────────────────────────────────────────────────────


class ToyRows:
    def __init__(self, x):
        self.x = x.astype(np.float32)

    def take(self, idx):
        return self.x[np.asarray(idx, np.int64)].copy()


def selftest():
    rng = np.random.default_rng(7)
    checks = 0
    for trial in range(30):
        n = int(rng.integers(12, 70))
        p = float(rng.uniform(0.02, 0.25))
        A = np.triu(rng.random((n, n)) < p, 1)
        A = A | A.T
        if trial % 5 == 0:                                                         # a hub
            A[0, :] = A[:, 0] = True
            A[0, 0] = False
        iu = np.argwhere(np.triu(A, 1))
        keys = np.unique(iu[:, 0] * n + iu[:, 1]).astype(np.int64)
        am = np.ones(n, bool) if trial % 2 else rng.random(n) < 0.6
        if am.sum() < 2:
            am[:2] = True
        keys_at = P.incident(keys, am, n)
        nb = Nbrs(keys_at, am, n)
        for x in np.flatnonzero(am):                                                  # the lists equal the adjacency rows of the anchors
            want = np.flatnonzero(A[x])
            got = nb.of(int(x))
            if not np.array_equal(got, want):
                stop(f"G-C3: trial {trial}: neighbour list of {x} differs")
        anch = np.flatnonzero(am)
        k = 40
        a = rng.choice(anch, k)
        b = rng.choice(anch, k)
        ok = a != b
        a, b = a[ok], b[ok]
        gl = [rng.choice(n, size=3, replace=False) for _ in range(a.size)]
        cls, m, via = classify(nb, a, b, gl)
        for i in range(a.size):
            ia, ib = int(a[i]), int(b[i])
            if A[ia, ib]:
                want, wm = 0, 0
            else:
                com = np.flatnonzero(A[ia] & A[ib])
                com = com[(com != ia) & (com != ib)]
                want, wm = (1, com.size) if com.size else (2, 0)
            if int(cls[i]) != want or int(m[i]) != wm:
                stop(f"G-C3: trial {trial}: slot ({ia},{ib}) class {cls[i]} m {m[i]}, brute force {want} {wm}")
            if want == 1:
                com = np.flatnonzero(A[ia] & A[ib])
                com = com[(com != ia) & (com != ib)]
                if bool(via[i]) != bool(np.isin(com, gl[i]).any()):
                    stop(f"G-C3: trial {trial}: via-gold differs for ({ia},{ib})")
            checks += 1
        # the ranks against a brute force
        dim = 8
        X = rng.standard_normal((n, dim))
        X /= np.linalg.norm(X, axis=1, keepdims=True)
        if trial % 4 == 0:
            X[3] = X[5]                                                                # a tie
        rows = ToyRows(X)
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        for i in np.flatnonzero(cls == 0)[:10]:
            s, t = source_target(rows.take([int(a[i])])[0] @ q, rows.take([int(b[i])])[0] @ q, a[i], b[i])
            s, t = int(s), int(t)
            ns = nb.of(s)
            got = ranks_of(neighbour_scores(rows, q, s, ns), ns, t)
            Xs = X.astype(np.float32)
            sh = Xs[s] / np.linalg.norm(Xs[s])
            c = np.float32(q @ sh)
            pdv = [np.float32(Xs[x] @ Xs[s]) for x in ns]
            fqv = [np.float32(Xs[x] @ q) for x in ns]
            rqv = [np.float32(f - c * np.float32(Xs[x] @ sh)) for f, x in zip(fqv, ns)]
            want = []
            for sc in (pdv, fqv, rqv):
                ti = int(np.searchsorted(ns, t))
                want.append(1 + sum(1 for z in sc if z > sc[ti]))
            if got != want:
                stop(f"G-C3: trial {trial}: ranks {got}, brute force {want}")
            if not (np.dot(q, Xs[t]) <= 1.0001):
                stop("G-C3: a cosine above 1")
            checks += 1
        # the control: no self pair, ends are anchors, drops counted
        qi = np.sort(rng.integers(0, 6, a.size))
        keep, bp = control_slots(qi, a, b)
        if keep.size and (bp == a[keep]).any():
            stop("G-C4: a control slot is a self pair")
        checks += 1
    # a chain: A - m - B, no direct link; and the two ends of a slot that share a gold neighbour
    n = 6
    keys = np.array([0 * n + 1, 1 * n + 2], np.int64)
    nb = Nbrs(keys, np.ones(n, bool), n)
    cls, m, via = classify(nb, np.array([0, 0, 3]), np.array([2, 1, 4]), [np.array([1]), np.array([5]), np.array([5])])
    if cls.tolist() != [1, 0, 2] or m.tolist() != [1, 0, 0] or via.tolist() != [True, False, False]:
        stop(f"G-C3: the chain case gives {cls.tolist()} {m.tolist()} {via.tolist()}")
    checks += 1
    # G-C5 (amendment A1): answer-entity golds are far from the question, the dense top five is near; the memory-mapped member equals the array
    nn, dim = 1200, 16
    X = rng.standard_normal((nn, dim))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = X[:300] + 0.05 * rng.standard_normal((300, dim))
    Q = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float32)
    toprows = np.argsort(-(Q @ X.astype(np.float32).T), axis=1)[:, :20].astype(np.int32)
    far = [np.argsort(Q[j] @ X.astype(np.float32).T)[:2] for j in range(300)]         # the two farthest nodes stand for unaligned answer entities
    qrow = np.arange(300)
    if gate_rows_aligned(ToyRows(X), ToyRows(Q), qrow, [np.asarray([j]) for j in range(300)], nn) < 0.9:
        stop("G-C3: the gate rejects aligned passage golds")
    try:
        gate_rows_aligned(ToyRows(X), ToyRows(Q), qrow, far, nn)
        stop("G-C3: the gate accepts unaligned golds without a dense list")
    except SystemExit as e:
        if "golds' question cosine" not in str(e):
            raise
    res = gate_rows_aligned(ToyRows(X), ToyRows(Q), qrow, far, nn, top=toprows)
    if not (isinstance(res, dict) and res["basis"] == "dense_top5" and res["share"] >= 0.9 and res["golds_share"] < 0.5):
        stop(f"G-C3: the dense-list gate gives {res}")
    shuf = toprows[::-1].copy()                                                       # rows of another question: misaligned, must stop
    try:
        gate_rows_aligned(ToyRows(X), ToyRows(Q), qrow, far, nn, top=shuf)
        stop("G-C3: the dense-list gate accepts a misaligned list")
    except SystemExit as e:
        if "dense top five" not in str(e):
            raise
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        for name, fn in (("stored.npz", np.savez), ("packed.npz", np.savez_compressed)):
            pth = Path(td) / name
            fn(pth, ids=toprows, scores=toprows.astype(np.float32))
            if not np.array_equal(np.asarray(npz_member(pth, "ids")), toprows):
                stop(f"G-C3: npz_member of {name} differs")
    checks += 6
    print(f"selftest PASS ({checks} checks: classes, shared neighbours, via gold, the three ranks with ties, the control, the alignment gate on both bases, npz members)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("ledger")
    s.add_argument("--ds", required=True, choices=DS7)
    s.add_argument("--host", action="store_true")
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--limit", type=int, default=0, help="smoke: the first N questions of each cell (no number is read)")
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": lambda _a: selftest(), "ledger": cmd_ledger, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
