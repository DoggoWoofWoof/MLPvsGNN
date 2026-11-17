"""ENC E1a (docs/ENC_ENCODER_COMPARISON.md, section 9): each encoder's K family at UL1's oracle pairs. Graph level, CPU only.

    python outputs/mp_unified/zenc1.py selftest
    python outputs/mp_unified/zenc1.py nn --enc frozen --ds musique [--host] [--threads 8]   # text groups (cached), endpoint top 5; G1a on frozen
    python outputs/mp_unified/zenc1.py grade --ds musique [--host]                         # G2a, the shares, the paired differences
    python outputs/mp_unified/zenc1.py report

The K rule is the record's (GRAPH_MANIFEST's source_manifest; CRAG/scratchpad/knn_parity_check.py): exact fp32 inner products of
L2-renormalised float32 copies of the stored float16 rows, over distinct source rows; the top 4 including the row itself; self
dropped (all 4 kept when self is not among them); each position inherits its source row's neighbours, mapped to the lowest position
of each neighbour; undirected. The package carries no pointer index, so a source row is stood in for by a text group (positions
with byte-identical `text`), the group's vector being its lowest position's row; G1a checks the stand-in on the frozen arm.

E1a computes the neighbour lists only for the groups of the slot ends (the golds of questions with a slot), top 5 by score and then
group, the 5th kept to mark ties. A slot (a, b) is linked by an arm's K iff b is in out(a) or a is in out(b), out(x) being the lowest
positions of x's group's neighbours; both ends are slot ends, so the membership is exact for every slot.

Nothing trains, no model is read, no filed graph, pool, fit or record changes. Outputs only under outputs/enc/e1/ (arrays uncommitted).
"""
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zenc as E  # noqa: E402  (sets the BLAS threads from --threads, default 8, before numpy loads)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import u1b  # noqa: E402
import zap1 as Z  # noqa: E402
import zu2l_el as EL  # noqa: E402
import zul1 as UL  # noqa: E402

DECLARED = "docs/ENC_ENCODER_COMPARISON.md (section 9)"
ROOT = HERE.parents[1]
OUT1 = ROOT / "outputs" / "enc" / "e1"
ARMS = ("frozen",) + tuple(E.NEW)
SETS = ("musique", "hotpotqa", "2wiki", "metaqa", "webqsp")     # SQuAD has one gold a question: no slot
DECIDE = ("2wiki", "hotpotqa", "musique")
CARVE = "s1eval"
SEARCH_K = 4                                                   # the record's search_k (self included)
TOPN = 5                                                       # one more, to mark ties at the 4th place
TIE = 1e-3                                                     # the record's tie_tolerance
G1A_MIN = 0.995
G2A_TOL = 0.01
BLOCK = 2048                                                   # endpoint rows per GEMM block
BIG = np.iinfo(np.int64).max


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"ZENC1: {msg}")


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def code_shas():
    return {"script_sha256": script_sha256(), "zenc_sha256": hashlib.sha256(Path(E.__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "zul1_sha256": UL.script_sha256(), "zap1_sha256": Z.script_sha256()}


def save_npz(p, **arrs):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrs)
    os.replace(tmp, p)
    return E.sha256(p)


def package(host):
    served, freeze = u1b.package_root(host)
    if freeze != E.RECORD_SHA:
        stop(f"the package is frozen as {freeze}, not {E.RECORD_SHA[:8]}...")
    return served, freeze


def dataset_nodes(served, d):
    man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
    return int(man["nodes"]["n"]), man["nodes"]["sha256"]


# ── text groups (the stand-in for the record's source rows) ─────────────────


def group_digests(dig):
    """(pos2g, g_first) from per-position 16-byte digests: a group = one digest; group ids ranked by their lowest position."""
    v = np.ascontiguousarray(dig, np.uint8).view(np.dtype((np.void, dig.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(order.size, np.int64)
    rank[order] = np.arange(order.size, dtype=np.int64)
    return rank[np.asarray(inv).ravel()], first[order].astype(np.int64)


def text_digest(text):
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def text_groups(served, d, n, nsha):
    """Cached at outputs/enc/e1/groups/<d>.npz; built from nodes.jsonl, whose sha256 must be DATASET.json's."""
    p = OUT1 / "groups" / f"{d}.npz"
    if p.is_file():
        with np.load(p) as z:
            if str(z["nodes_sha256"]) == nsha and int(z["n"]) == n:
                return z["pos2g"], z["g_first"], E.sha256(p)
    t0, h, buf, m = time.time(), hashlib.sha256(), bytearray(), 0
    with open(served / d / "nodes.jsonl", "rb") as f:
        for raw in f:
            h.update(raw)
            buf += text_digest(json.loads(raw)["text"])
            m += 1
    if m != n or h.hexdigest() != nsha:
        stop(f"{d}: nodes.jsonl has {m} lines and sha256 {h.hexdigest()[:12]}, DATASET.json {n} and {nsha[:12]}")
    pos2g, g_first = group_digests(np.frombuffer(bytes(buf), np.uint8).reshape(n, 16))
    sha = save_npz(p, n=np.int64(n), nodes_sha256=np.asarray(nsha), pos2g=pos2g, g_first=g_first)
    log(f"{d}: {g_first.size:,} text groups over {n:,} positions ({time.time() - t0:.0f} s)")
    return pos2g, g_first, sha


# ── the questions and their slots (UL1's, unchanged) ────────────────────────


def slots(served, d, n):
    """UL1's s1eval carve: ids, {kind: (qidx, keys)} (GP; MuSiQue's CH too), the hop counts (MuSiQue) and the questions file's sha256."""
    ids, golds, qsha = UL.carve_questions(d, CARVE)
    kinds = {"GP": UL.gp_slots(golds, n, UL.KB_GOLDS if d in UL.KB else None)}
    hops = None
    if d == "musique":
        recs = UL.musique_queries(served, ids)
        posmap = UL.node_positions(served, d)
        if len(posmap) != n:
            stop(f"musique: {len(posmap)} distinct node ids for {n} nodes")
        qi, keys, hops = UL.chain_slots(recs, ids, golds, posmap, n)
        kinds["CH"] = (qi, keys)
    return ids, kinds, hops, qsha


def slot_ends(kinds, n):
    ks = [k for _, k in kinds.values() if k.size]
    if not ks:
        return np.zeros(0, np.int64)
    k = np.concatenate(ks)
    return np.unique(np.concatenate([k // n, k % n]))


# ── the neighbour lists ──────────────────────────────────────────────────────


def shard_offsets(paths):
    return np.concatenate([[0], np.cumsum([np.load(p, mmap_mode="r").shape[0] for p in paths])]).astype(np.int64)


def unit(X):
    X = np.asarray(X, np.float32)
    nr = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.isfinite(X).all() or (nr == 0).any():
        stop("a damaged vector row (non-finite or zero)")
    X /= nr
    return X


def gather(paths, offs, rows):
    rows = np.asarray(rows, np.int64)
    s_of = np.searchsorted(offs, rows, "right") - 1
    out = None
    for s in np.unique(s_of):
        a = np.load(paths[s], mmap_mode="r")
        if out is None:
            out = np.empty((rows.size, a.shape[1]), np.float32)
        m = s_of == s
        out[m] = a[rows[m] - offs[s]]
    return out


def row_top(S, gid, k):
    """Per row of S (rows x cols), the top k columns by score descending, then group ascending: (scores, groups), padded."""
    nr, nc = S.shape
    ts, tg = np.full((nr, k), -np.inf, np.float32), np.full((nr, k), BIG, np.int64)
    kk = min(k, nc)
    if kk == 0:
        return ts, tg
    part = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    ps = np.take_along_axis(S, part, 1)
    pg = gid[part]
    thr = ps.min(1)
    for r in np.flatnonzero((S >= thr[:, None]).sum(1) > kk):        # exact ties at the boundary: the lower groups win
        c = np.flatnonzero(S[r] >= thr[r])
        o = np.lexsort((gid[c], -S[r, c]))[:kk]
        ps[r], pg[r] = S[r, c[o]], gid[c[o]]
    o = np.lexsort((pg, -ps), axis=1)
    ts[:, :kk], tg[:, :kk] = np.take_along_axis(ps, o, 1), np.take_along_axis(pg, o, 1)
    return ts, tg


def nn_top(paths, offs, pos2g, g_first, eg, topn=TOPN, block=BLOCK, every=25):
    """For each endpoint group eg, its top `topn` groups over every group's vector (its lowest position's row), exact float32."""
    m = eg.size
    Xe = unit(gather(paths, offs, g_first[eg]))
    bs, bg = np.full((m, topn), -np.inf, np.float32), np.full((m, topn), BIG, np.int64)
    rep = np.zeros(int(offs[-1]), bool)
    rep[g_first] = True
    t0 = time.time()
    for s, p in enumerate(paths):
        lo, hi = int(offs[s]), int(offs[s + 1])
        r = np.flatnonzero(rep[lo:hi])
        if r.size:
            X = unit(np.load(p, mmap_mode="r")[r])
            gid = pos2g[lo + r]
            for b0 in range(0, m, block):
                b1 = min(m, b0 + block)
                ts, tg = row_top(Xe[b0:b1] @ X.T, gid, topn)
                cs, cg = np.concatenate([bs[b0:b1], ts], 1), np.concatenate([bg[b0:b1], tg], 1)
                o = np.lexsort((cg, -cs), axis=1)[:, :topn]
                bs[b0:b1], bg[b0:b1] = np.take_along_axis(cs, o, 1), np.take_along_axis(cg, o, 1)
        if every and (s + 1) % every == 0:
            log(f"  shard {s + 1}/{len(paths)} ({time.time() - t0:.0f} s)")
    return bg, bs


def out_positions(eg, nbr, g_first):
    """(m, 4): the lowest positions of each endpoint group's neighbours among its top 4, -1 where the entry is the group itself or padding."""
    top = nbr[:, :SEARCH_K]
    ok = (top != eg[:, None]) & (top < g_first.size)
    return np.where(ok, g_first[np.where(ok, top, 0)], -1)


def tie_rows(sc):
    return (sc[:, SEARCH_K - 1] - sc[:, SEARCH_K]) <= TIE


def end_rows(eg, pos2g, x):
    i = np.searchsorted(eg, pos2g[x])
    i[i == eg.size] = 0
    if eg.size == 0 or not (eg[i] == pos2g[x]).all():
        stop("a slot end has no neighbour row")
    return i


def k_member(keys, n, eg, outp, pos2g):
    """Which slots (keys a*n+b, a<b) the K rebuilt from these neighbour rows links: b in out(a) or a in out(b)."""
    keys = np.asarray(keys, np.int64)
    a, b = keys // n, keys % n
    ia, ib = end_rows(eg, pos2g, a), end_rows(eg, pos2g, b)
    return (outp[ia] == b[:, None]).any(1) | (outp[ib] == a[:, None]).any(1)


# ── G1a: the rebuild reproduces the record's K at the slot ends (frozen arm) ─


def g1a_core(eg, outp, tie, pos2g, ends, kinds, K, n):
    ie = end_rows(eg, pos2g, ends)
    P = outp[ie]
    v = P >= 0
    x = np.broadcast_to(ends[:, None], P.shape)
    keys = (np.minimum(x, P) * n + np.maximum(x, P))[v]
    hit = UL.member(keys, K)
    trow = np.broadcast_to(tie[ie][:, None], P.shape)[v]
    res = {"ends": int(ends.size), "out_pairs": int(keys.size), "in_record": int(hit.sum()),
           "missed_at_tie_rows": int((~hit & trow).sum()), "missed_elsewhere": int((~hit & ~trow).sum()),
           "precision": round(float(hit.mean()), 6) if keys.size else None,
           "precision_with_tie_churn": round(float((hit | trow).mean()), 6) if keys.size else None, "slots": {}}
    ok = keys.size == 0 or (hit | trow).mean() >= G1A_MIN
    for kind, (qidx, sk) in kinds.items():
        if not sk.size:
            continue
        a, b = sk // n, sk % n
        reb, rec = k_member(sk, n, eg, outp, pos2g), UL.member(sk, K)
        agree = reb == rec
        t = tie[end_rows(eg, pos2g, a)] | tie[end_rows(eg, pos2g, b)]
        same = pos2g[a] == pos2g[b]
        adj = float((agree | t).mean())
        res["slots"][kind] = {"slots": int(sk.size), "agree": int(agree.sum()), "rebuilt_only": int((reb & ~rec).sum()),
                              "record_only": int((rec & ~reb).sum()), "disagree_at_tie_rows": int((~agree & t).sum()),
                              "same_text_slots": int(same.sum()), "agreement": round(float(agree.mean()), 6),
                              "agreement_with_tie_churn": round(adj, 6), "linked_rebuilt": int(reb.sum()), "linked_record": int(rec.sum())}
        ok = ok and adj >= G1A_MIN
    res["verdict"] = "PASS" if ok else "FAIL"
    return res


# ── nn ───────────────────────────────────────────────────────────────────────


def cmd_nn(a):
    d, enc, t0 = a.ds, a.enc, time.time()
    if d not in SETS or enc not in ARMS:
        stop(f"E1a reads {SETS} and {ARMS}")
    served, freeze = package(a.host)
    n, nsha = dataset_nodes(served, d)
    pos2g, g_first, gsha = text_groups(served, d, n, nsha)
    ids, kinds, _hops, qsha = slots(served, d, n)
    ends = slot_ends(kinds, n)
    eg = np.unique(pos2g[ends])
    paths = E.vec_shards(served, enc, d, "docs")                       # G3a: every shard present, the arm's manifest rows = the record's
    offs = shard_offsets(paths)
    if int(offs[-1]) != n:
        stop(f"{enc} {d}: {int(offs[-1])} vector rows for {n} nodes")
    vec = {"record_sha256": E.RECORD_SHA} if enc == "frozen" else {
        "manifest_sha256": E.sha256(E.vec_dir(served, enc, d, "docs") / "manifest.json"), "revision": E.pinned(enc)}
    log(f"{d} {enc}: {len(ids):,} questions, " + ", ".join(f"{k} {v[1].size:,} slots" for k, v in kinds.items())
        + f", {ends.size:,} ends in {eg.size:,} groups; {len(paths)} shards")
    t1 = time.time()
    nbr, sc = nn_top(paths, offs, pos2g, g_first, eg)
    secs = round(time.time() - t1, 1)
    base = OUT1 / "nn" / enc
    sha = save_npz(base / f"{d}.npz", eg=eg, nbr=nbr, sc=sc)
    rec = {"dataset": d, "enc": enc, "declared_in": DECLARED, "nodes": n, "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze,
           "groups": int(g_first.size), "groups_sha256": gsha, "questions_npz_sha256": qsha, "questions": len(ids),
           "slots": {k: int(v[1].size) for k, v in kinds.items()}, "ends": int(ends.size), "end_groups": int(eg.size),
           "tie_rows": int(tie_rows(sc).sum()), "vectors": vec, "threads": E.THREADS, "search_seconds": secs,
           "seconds": round(time.time() - t0, 1), "npz_sha256": sha, **code_shas()}
    if enc == "frozen":
        s_, t_, ksha = EL.pkg_family(served, d, "knn")
        K = UL.pair_keys(s_, t_, n)
        del s_, t_
        g = g1a_core(eg, out_positions(eg, nbr, g_first), tie_rows(sc), pos2g, ends, kinds, K, n)
        gm = json.loads((served / d / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))["families"]["knn"]
        sm = (gm.get("source") or {}).get("source_manifest") or {}
        g.update({"dataset": d, "knn_npz_sha256": ksha, "record_pairs": int(K.size), "text_groups": int(g_first.size),
                  "manifest_distinct_source_rows": sm.get("n_distinct_source_rows"), "nn_npz_sha256": sha, **code_shas()})
        E.wjson(OUT1 / "g1a" / f"{d}.json", g)
        rec["g1a"] = g["verdict"]
        log(f"{d}: G1a {g['verdict']}: precision {g['precision']} ({g['precision_with_tie_churn']} with tie churn); "
            + ", ".join(f"{k} agreement {v['agreement']} ({v['agreement_with_tie_churn']})" for k, v in g["slots"].items()))
    E.wjson(base / f"{d}.json", rec)
    log(f"{d} {enc}: written {base / (d + '.npz')} ({rec['seconds']} s)")
    return 0 if rec.get("g1a", "PASS") == "PASS" else 3


# ── grade ────────────────────────────────────────────────────────────────────


def csr_upper_member(indptr, col, sk, n):
    """Which slots (a<b) are pairs of an undirected CSR, read as zul1.csr_keys reads it: the entry b in row a."""
    out = np.zeros(sk.size, bool)
    for j, k in enumerate(np.asarray(sk, np.int64).tolist()):
        a, b = divmod(k, n)
        out[j] = bool((np.asarray(col[indptr[a]:indptr[a + 1]]) == b).any())
    return out


def family_members(served, d, n, nsha, freeze, sk):
    """zul1.families' S, W2_S, CN_all, F0T and the record's K, as membership of the slot keys sk only (one family in memory at a time)."""
    sh, F = {}, {}
    s_, t_, sh["knn_npz_sha256"] = EL.pkg_family(served, d, "knn")
    F["K_record"] = UL.member(sk, UL.pair_keys(s_, t_, n))
    del s_, t_
    if d in UL.HAS_S:
        slab, sh["structural_npz_sha256"] = UL.s_labels(served, d, n)
        F["S"] = UL.member(sk, np.unique(slab[0]))
        del slab
    p, sh["w2_s_npz_sha256"] = UL.rule_part(d, n, nsha, freeze)
    with np.load(p) as z:
        ip = z["indptr"]
        if ip.size != n + 1:
            stop(f"{d}: W2_S has {ip.size - 1} rows, the package {n}")
        F["W2_S"] = csr_upper_member(ip, z["col"], sk, n)
    UL.names_record(d)
    b = json.loads((UL.OUT / d / "build.json").read_text(encoding="utf-8"))
    if b["script_sha256"] != UL.script_sha256() or Z.sha_file(UL.OUT / d / "f0t.npz") != b["f0t_npz_sha256"] or b["nodes"] != n:
        stop(f"{d}: UL1's build is not zul1's, or f0t.npz is not its record's")
    with np.load(UL.U2R / d / "names.npz") as z:
        cn = UL.pair_keys(z["cn_all_src"], z["cn_all_dst"], n)
    with np.load(UL.OUT / d / "f0t.npz") as z:
        lt = UL.pair_keys(z["lt_src"], z["lt_dst"], n)
    if UL.union_size(cn, lt) != b["f0t_pairs"]:
        stop(f"{d}: F0T has {UL.union_size(cn, lt)} pairs, UL1's build {b['f0t_pairs']}")
    F["CN_all"] = UL.member(sk, cn)
    F["F0T"] = F["CN_all"] | UL.member(sk, lt)
    del cn, lt
    sh["f0t_npz_sha256"], sh["names_npz_sha256"] = b["f0t_npz_sha256"], b["names_npz_sha256"]
    return F, sh


def load_nn(enc, d, gsha, eg):
    rp, p = OUT1 / "nn" / enc / f"{d}.json", OUT1 / "nn" / enc / f"{d}.npz"
    if not rp.is_file():
        return None
    rec = json.loads(rp.read_text(encoding="utf-8"))
    if rec["script_sha256"] != script_sha256() or rec["groups_sha256"] != gsha or E.sha256(p) != rec["npz_sha256"]:
        stop(f"{enc} {d}: the neighbour rows are not this code's, these groups' or their record's")
    with np.load(p) as z:
        if not np.array_equal(z["eg"], eg):
            stop(f"{enc} {d}: the neighbour rows' endpoint groups differ from the slots'")
        return z["nbr"], z["sc"], rec


def connected(cov, qidx, hops, nq):
    """Per question with a chain slot: all its chain slots linked."""
    has = np.bincount(qidx, minlength=nq) > 0
    miss = np.bincount(qidx, weights=(~cov).astype(float), minlength=nq)
    return has, has & (miss == 0)


def cmd_grade(a):
    d, t0 = a.ds, time.time()
    served, freeze = package(a.host)
    n, nsha = dataset_nodes(served, d)
    g = OUT1 / "g1a" / f"{d}.json"
    if not g.is_file() or json.loads(g.read_text(encoding="utf-8"))["verdict"] != "PASS":
        stop(f"{d}: G1a has not passed")
    pos2g, g_first, gsha = text_groups(served, d, n, nsha)
    ids, kinds, hops, qsha = slots(served, d, n)
    nq = len(ids)
    eg = np.unique(pos2g[slot_ends(kinds, n)])
    allk = np.unique(np.concatenate([k for _, k in kinds.values()]))
    F, fsh = family_members(served, d, n, nsha, freeze, allk)
    log(f"{d}: families read for {allk.size:,} slots ({time.time() - t0:.0f} s)")
    K, nnrec = {"record": F["K_record"]}, {}
    for enc in ARMS:
        r = load_nn(enc, d, gsha, eg)
        if r is None:
            if enc == "frozen":
                stop(f"{d}: no frozen neighbour rows")
            continue
        K[enc] = k_member(allk, n, eg, out_positions(eg, r[0], g_first), pos2g)
        nnrec[enc] = {"npz_sha256": r[2]["npz_sha256"], "vectors": r[2]["vectors"], "tie_rows": r[2]["tie_rows"]}
    arms = [e for e in ARMS if e in K]
    ul1 = json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8"))["carves"][CARVE]
    res, g2a = {}, []
    for kind, (qidx, sk) in kinds.items():
        if not sk.size:
            continue
        j = np.searchsorted(allk, sk)
        res[kind] = {"slots": int(sk.size), "domains": {}}
        for dom in UL.DOMAINS[d]:
            sr = F["W2_S"][j] | F["F0T"][j]
            if dom == "SRK":
                sr = sr | F["S"][j]
            rows = {"S_R": Z.pct(sr.mean())}
            kf = K["frozen"][j]
            for arm in ("record",) + tuple(arms):
                k = K[arm][j]
                u = sr | k
                rows[arm] = {"K_alone": Z.pct(k.mean()), "union": Z.pct(u.mean()), "K_only": Z.pct((k & ~sr).mean())}
                if arm != "frozen":
                    rows[arm]["vs_frozen"] = {"K_alone": UL.grade_diff(k, kf, qidx, nq), "union": UL.grade_diff(u, sr | kf, qidx, nq)}
                if kind == "CH":
                    has, cc = connected(u, qidx, hops, nq)
                    _h, ccf = connected(sr | kf, qidx, hops, nq)
                    q = np.flatnonzero(has)
                    rows[arm]["connected"] = {str(h): Z.pct(cc[has & (hops == h)].mean()) for h in (2, 3, 4) if (has & (hops == h)).any()}
                    rows[arm]["connected"]["all"] = Z.pct(cc[has].mean())
                    if arm != "frozen":
                        rows[arm]["vs_frozen"]["connected"] = UL.grade_diff(cc[q], ccf[q], np.arange(q.size), q.size)
            want = ul1[dom][kind]["forms"]["F0T"]["share"]
            g2a.append({"kind": kind, "domain": dom, "ul1_F0T_share": want, "record_union": rows["record"]["union"],
                        "ok": abs(want - rows["record"]["union"]) <= G2A_TOL})
            res[kind]["domains"][dom] = rows
    if not all(x["ok"] for x in g2a):
        E.wjson(OUT1 / "grade" / f"{d}.FAILED.json", {"g2a": g2a})
        stop(f"{d}: G2a failed (the record's union is not UL1's F0T share): {g2a}")
    out = {"dataset": d, "declared_in": DECLARED, "carve": CARVE, "questions": nq, "questions_npz_sha256": qsha, "nodes": n,
           "nodes_sha256": nsha, "freeze_RECORD_SHA256": freeze, "groups_sha256": gsha, "arms": arms, "nn": nnrec, "families": fsh,
           "g2a": g2a, "kinds": res, "seconds": round(time.time() - t0, 1), **code_shas()}
    E.wjson(OUT1 / "grade" / f"{d}.json", out)
    log(f"{d}: graded in {out['seconds']} s; written {OUT1 / 'grade' / (d + '.json')}")
    return 0


# ── the report (section 9.6) ─────────────────────────────────────────────────


def decide(G):
    """Y1, Y2, Y3 per new arm over the filed grades {dataset: grade}."""
    out = {}
    for arm in E.NEW:
        if not all(arm in G[d]["arms"] for d in DECIDE if d in G) or not all(d in G for d in DECIDE):
            out[arm] = {"read": False}
            continue
        y1 = {d: G[d]["kinds"]["GP"]["domains"][UL.DOMAINS[d][0]][arm]["vs_frozen"]["K_alone"]["call"] for d in DECIDE}
        y2 = {}
        for d in DECIDE:
            for dom in UL.DOMAINS[d]:
                v = G[d]["kinds"]["GP"]["domains"][dom][arm]["vs_frozen"]
                y2[f"{d} {dom}"] = abs(v["union"]["diff"]) < abs(v["K_alone"]["diff"])
        v = G["musique"]["kinds"]["CH"]["domains"]["RK"][arm]["vs_frozen"]
        out[arm] = {"read": True, "Y1": "HELD" if all(c == "BELOW" for c in y1.values()) else "NOT_HELD", "Y1_calls": y1,
                    "Y2": "HELD" if all(y2.values()) else "NOT_HELD", "Y2_cells": y2,
                    "Y3": "HELD" if abs(v["union"]["diff"]) < abs(v["K_alone"]["diff"]) else "NOT_HELD"}
    return out


def render(rec):
    L = ["# ENC E1a: each encoder's K at UL1's oracle pairs (s1eval)", "", f"Declared in {DECLARED}. Shares in % of slots; "
         "differences in points against the frozen arm rebuilt by the same code, with 95% intervals.", ""]
    for d in SETS:
        G = rec["grades"].get(d)
        if G is None:
            L += [f"## {d}: not graded", ""]
            continue
        L += [f"## {d} ({G['questions']:,} questions)", ""]
        for kind, kv in G["kinds"].items():
            for dom, rows in kv["domains"].items():
                L += [f"**{kind}, {dom}** ({kv['slots']:,} slots; S and R alone {rows['S_R']}%)", "",
                      "| arm | K alone | union | K only | K alone - frozen | union - frozen |", "|---|---|---|---|---|---|"]
                for arm in ("record",) + tuple(G["arms"]):
                    r = rows[arm]
                    vf = r.get("vs_frozen")
                    cell = (lambda c: f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}")
                    L.append(f"| {arm} | {r['K_alone']} | {r['union']} | {r['K_only']} | "
                             f"{cell(vf['K_alone']) if vf else ''} | {cell(vf['union']) if vf else ''} |")
                L.append("")
    L += ["## Expectations (section 9.6)", ""]
    for arm, v in rec["decision"].items():
        L.append(f"- **{arm}**: " + ("not read" if not v["read"] else f"Y1 {v['Y1']} ({v['Y1_calls']}), Y2 {v['Y2']}, Y3 {v['Y3']}"))
    L += ["", "D97 (open; default: every arm goes on to E1b and E2; an exclusion is the user's call).", ""]
    return "\n".join(L)


def cmd_report(_a):
    G = {}
    for d in SETS:
        p = OUT1 / "grade" / f"{d}.json"
        if p.is_file():
            G[d] = json.loads(p.read_text(encoding="utf-8"))
    rec = {"declared_in": DECLARED, "grades": G, "decision": decide(G), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           **code_shas()}
    E.wjson(OUT1 / "report.json", rec)
    (OUT1 / "report.md").write_text(render(rec), encoding="utf-8")
    log(f"report: {', '.join(f'{k} {v}' for k, v in rec['decision'].items())}")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def lifted_record_k(V, pos2g, g_first):
    """The record's rule by brute force over all groups (float64 here, exact for the test's vectors): sorted unique pair keys."""
    n, G = pos2g.size, g_first.size
    X = V[g_first].astype(np.float64)
    S = X @ X.T
    nbr = np.empty((G, SEARCH_K), np.int64)
    for g in range(G):
        o = np.lexsort((np.arange(G), -S[g]))[:SEARCH_K]
        nbr[g] = o
    I, J = [], []
    for x in range(n):
        for v in nbr[pos2g[x]]:
            if v != pos2g[x]:
                I.append(x)
                J.append(g_first[v])
    return UL.pair_keys(np.asarray(I), np.asarray(J), n), nbr, S


def selftest():
    rng = np.random.default_rng(0)
    # groups
    texts = [f"t{int(x)}" for x in rng.integers(0, 40, 120)]
    p2g, gf = group_digests(np.frombuffer(b"".join(text_digest(t) for t in texts), np.uint8).reshape(len(texts), 16))
    seen = {}
    ref = np.asarray([seen.setdefault(t, len(seen)) for t in texts])
    assert (p2g == ref).all() and (gf == np.asarray([texts.index(t) for t in seen])).all(), "groups"
    # a corpus with exact scores: unit vectors with entries in {0, +-0.5} (four non-zeros), so every dot product is exact
    G, dim = 90, 16
    W = np.zeros((G, dim), np.float32)
    for g in range(G):
        W[g, rng.choice(dim, 4, replace=False)] = rng.choice([-0.5, 0.5], 4)
    W[61:66] = W[60]                     # six identical groups: the higher ones lose self from their top 4 (ties go to lower groups)
    W[17] = W[5]
    n = 260
    gseq = np.concatenate([np.arange(G), rng.integers(0, G, n - G)])
    gtxt = [f"g{int(g)}" for g in gseq]
    pos2g, g_first = group_digests(np.frombuffer(b"".join(text_digest(t) for t in gtxt), np.uint8).reshape(n, 16))
    V = np.empty((n, dim), np.float32)
    for x in range(n):
        V[x] = W[int(gtxt[x][1:])]
    with tempfile.TemporaryDirectory() as td:
        paths = []
        for s in range(0, n, 37):
            p = Path(td) / f"shard_{s // 37:05d}.npy"
            np.save(p, V[s:s + 37].astype(np.float16))
            paths.append(p)
        offs = shard_offsets(paths)
        ends = np.unique(np.concatenate([rng.choice(n, 70, replace=False), np.flatnonzero(np.isin(pos2g, pos2g[np.isin(gseq, [60, 63, 64, 65, 5, 17])]))]))
        eg = np.unique(pos2g[ends])
        nbr, sc = nn_top(paths, offs, pos2g, g_first, eg, block=7, every=0)
    Kfull, bnbr, S = lifted_record_k(V, pos2g, g_first)
    for i, g in enumerate(eg):
        o = np.lexsort((np.arange(g_first.size), -S[g]))[:TOPN]
        assert (nbr[i] == o).all() and np.allclose(sc[i], S[g, o]), f"top {TOPN} of group {g}"
    outp = out_positions(eg, nbr, g_first)
    assert ((outp >= 0).sum(1) == SEARCH_K).any(), "a group with self outside its top 4 keeps all four"
    a, b = np.triu_indices(ends.size, 1)
    sk = np.unique(np.minimum(ends[a], ends[b]) * n + np.maximum(ends[a], ends[b]))
    sk = sk[sk // n != sk % n]
    assert (k_member(sk, n, eg, outp, pos2g) == UL.member(sk, Kfull)).all(), "slot membership = the record's lifted K"
    # G1a: the full K passes; a K without one out-pair of a non-tie row fails
    tie = tie_rows(sc)
    kinds = {"GP": (np.zeros(sk.size, np.int64), sk)}
    g = g1a_core(eg, outp, tie, pos2g, ends, kinds, Kfull, n)
    assert g["verdict"] == "PASS" and g["precision"] == 1.0 and g["slots"]["GP"]["agreement"] == 1.0, g
    nt = np.flatnonzero(~tie & ((outp >= 0).sum(1) > 0))
    assert nt.size, "a non-tie row"
    r0 = nt[0]
    x = ends[np.flatnonzero(pos2g[ends] == eg[r0])[0]]
    y = int(outp[r0][outp[r0] >= 0][0])
    drop = min(x, y) * n + max(x, y)
    g = g1a_core(eg, outp, tie, pos2g, ends, kinds, Kfull[Kfull != drop], n)
    assert g["missed_elsewhere"] >= 1 and g["precision"] < 1.0, g
    # the CSR read
    m = 50
    A = np.triu(rng.random((m, m)) < 0.1, 1)
    A = A | A.T
    ip = np.concatenate([[0], np.cumsum(A.sum(1))]).astype(np.int64)
    col = np.concatenate([np.flatnonzero(r) for r in A]).astype(np.int64)
    keys = UL.csr_keys(ip, col, m)
    xy = rng.integers(0, m, (2, 300))
    q = np.unique(xy.min(0) * m + xy.max(0))
    q = q[q // m != q % m]
    assert (csr_upper_member(ip, col, q, m) == UL.member(q, keys)).all(), "the CSR read = zul1.csr_keys"
    # the expectations
    cell = lambda d, lo, hi: {"diff": d, "lo": lo, "hi": hi, "call": Z.call(lo, hi)}
    mk = lambda kd, ud: {"vs_frozen": {"K_alone": cell(kd, kd - 1, kd + 1), "union": cell(ud, ud - 1, ud + 1)}}
    Gs = {}
    for d in DECIDE:
        dom = {dm: {arm: mk(-5.0, -2.0) for arm in E.NEW} for dm in UL.DOMAINS[d]}
        Gs[d] = {"arms": list(ARMS), "kinds": {"GP": {"domains": dom}}}
    Gs["musique"]["kinds"]["CH"] = {"domains": {"RK": {arm: mk(-4.0, -1.0) for arm in E.NEW}}}
    dec = decide(Gs)
    assert all(v["Y1"] == "HELD" and v["Y2"] == "HELD" and v["Y3"] == "HELD" for v in dec.values()), dec
    Gs["2wiki"]["kinds"]["GP"]["domains"]["SRK"]["mpnet"] = mk(-5.0, -6.0)
    assert decide(Gs)["mpnet"]["Y2"] == "NOT_HELD"
    print("zenc1 selftest: PASS (groups, top 5 against brute force with exact ties, self outside the top 4, slot membership against the "
          "record's lifted K, G1a, the CSR read, the expectations)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for name in ("nn", "grade"):
        p = sub.add_parser(name)
        p.add_argument("--ds", required=True, choices=SETS)
        p.add_argument("--host", action="store_true")
        p.add_argument("--threads", default="8")
        if name == "nn":
            p.add_argument("--enc", required=True, choices=ARMS)
    sub.add_parser("report")
    a = ap.parse_args(argv)
    return {"selftest": lambda _a: selftest(), "nn": cmd_nn, "grade": cmd_grade, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
