"""GP3's slot reads (amendment A4, docs/GP3_NODE_COUNT_RULE.md): a BESIDE read of the links the oracle result says the models use. It decides nothing: no call, no gate pass
condition, no change to any program, the final rule or the link set. Slots are GP (every unordered pair of distinct golds of a question; the KBs read 2 to 10 golds) and CH (MuSiQue's
ORC chain pairs); both ends of a slot are gold nodes; a slot is LINKED in a cell version when its two ends are joined in that version's any-link graph.

    python outputs/mp_unified/zgp3p.py selftest                       # slots, linked flags, induced pairs and pools against brute force on toy graphs (no real data)
    python outputs/mp_unified/zgp3p.py slots --ds D [--host]          # outputs/gp3/<d>/slots.npz: the slots of the dev slice and their linked flag per version (no walk)
    python outputs/mp_unified/zgp3p.py precision --ds D --part q|r [--sample 40] [--host] [--threads 1]
        # needs outputs/gp3/rule_q.json (the final rule): re-walks the first SAMPLE questions of the part's declared list at the final triple and the cell's own triple of the final
        # score (restricted walker, zgp3r.py), materialises Pool(q), counts the distinct unordered any-link pairs with both ends in the pool, and the linked slots in the pool.
Recall reads (slots kept at every (score, triple)) need no walk and are made by zgp3s.py from the stored gold coverage and slots.npz.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zgp3r as R  # noqa: E402  (imports zgp3 -> zgp1 first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

G3 = R.G3
G1, H, UL, L2, E = G3.G1, G3.H, G3.G1.UL, G3.L2, G3.G1.E
CELLS, OUT, EMPTY = G3.CELLS, G3.OUT, G3.EMPTY
log, rel, stop = G3.log, G3.rel, G3.stop
WORK_CAP = 40_000_000                                      # A4: neighbour entries of a pool's nodes read per question; over it the pair count is "not read"
SAMPLE = 40                                                # A4: the first questions of the part's declared list
KIND = {"GP": 0, "CH": 1}


def script_sha256():
    return R.hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── slots ────────────────────────────────────────────────────────────────────


def gold_pairs(golds, lim):
    """GP as (question position, index of the first gold, index of the second): every pair of golds of a question in zul1.gp_slots' order; lim = (lo, hi) gold counts read (the KBs)."""
    q, ia, ib = [], [], []
    for j, g in enumerate(golds):
        m = int(np.asarray(g).size)
        if m < 2 or (lim and not (lim[0] <= m <= lim[1])):
            continue
        a, b = np.triu_indices(m, 1)
        q.append(np.full(a.size, j, np.int64))
        ia.append(a.astype(np.int64))
        ib.append(b.astype(np.int64))
    if not q:
        return EMPTY, EMPTY, EMPTY
    return np.concatenate(q), np.concatenate(ia), np.concatenate(ib)


def key_to_indices(golds, qi, keys, n):
    """The gold indices (within the question's golds) of the two ends of each slot key a * n + b."""
    ia = np.zeros(qi.size, np.int64)
    ib = np.zeros(qi.size, np.int64)
    for t in range(qi.size):
        g = np.asarray(golds[int(qi[t])]).tolist()
        a, b = int(keys[t]) // n, int(keys[t]) % n
        if a not in g or b not in g:
            stop(f"slot {t}: an end of key {keys[t]} is not one of the question's golds")
        ia[t], ib[t] = g.index(a), g.index(b)
    return ia, ib


def degree(G, x):
    return sum(int(ip[x + 1] - ip[x]) for ip, _ix in G.hv)


def is_linked(G, a, b):
    """The nodes a and b are joined in the any-link graph: the lower-degree end's neighbour rows (all halves) are scanned for the other."""
    if a == b:
        return False
    u, v = (a, b) if degree(G, a) <= degree(G, b) else (b, a)
    return any(bool(np.any(ix[ip[u]:ip[u + 1]] == v)) for ip, ix in G.hv)


def induced_pairs(G, pool, n, work_cap=WORK_CAP):
    """The distinct unordered any-link pairs (key a * n + b, a < b) with both ends in `pool` (sorted unique), or None when the pool's nodes carry more than `work_cap` neighbour entries."""
    pool = np.asarray(pool, np.int64)
    if pool.size < 2:
        return np.zeros(0, np.int64)
    tot = sum(int((ip[pool + 1] - ip[pool]).sum()) for ip, _ix in G.hv)
    if tot > work_cap:
        return None
    s, x, thin = G.gather(pool, limit=work_cap + 1)
    if thin:
        return None
    i = np.minimum(np.searchsorted(pool, x), pool.size - 1)
    inp = pool[i] == x
    s, x = s[inp], x[inp]
    m = s < x
    return np.unique(s[m] * n + x[m])


def slots_of(d, ids, golds, n, served):
    """All slots of the dev slice: arrays q, ia, ib, kind (GP 0, CH 1) and a note for CH."""
    lim = UL.KB_GOLDS if d in UL.KB else None
    q, ia, ib = gold_pairs(golds, lim)
    kind = np.zeros(q.size, np.int8)
    note = {"GP": f"{q.size} slots on {int(np.unique(q).size)} questions"}
    if d == "musique":
        try:
            recs = UL.musique_queries(served, ids)
            cq, ck, _hops = UL.chain_slots(recs, ids, golds, UL.node_positions(served, d), n)
            cia, cib = key_to_indices(golds, cq, ck, n)
            q, ia, ib, kind = np.r_[q, cq], np.r_[ia, cia], np.r_[ib, cib], np.r_[kind, np.ones(cq.size, np.int8)]
            note["CH"] = f"{cq.size} slots on {int(np.unique(cq).size)} questions"
        except SystemExit as err:
            note["CH"] = f"not read: {err}"
    return q, ia, ib, kind, note


def cmd_slots(a):
    t0 = time.time()
    d = a.ds
    D, gt, sl, L, nq = G3.setup(a)
    served = E.served_root()
    q, ia, ib, kind, note = slots_of(d, sl["ids"], sl["golds"], D.n, served)
    uq, uk = UL.gp_slots([np.asarray(g, np.int64) for g in sl["golds"]], D.n, UL.KB_GOLDS if d in UL.KB else None)                  # G3-10a: GP equals UL2P's own builder
    mine = np.asarray([int(np.asarray(sl["golds"][int(j)])[int(x)]) * D.n + int(np.asarray(sl["golds"][int(j)])[int(y)]) for j, x, y in zip(q[kind == 0], ia[kind == 0], ib[kind == 0])], np.int64)
    if not (np.array_equal(uq, q[kind == 0]) and np.array_equal(uk, mine)):
        stop(f"G3-10a: {d}: the GP slots differ from zul1.gp_slots ({q[kind == 0].size} against {uq.size})")
    na = np.asarray([int(np.asarray(sl["golds"][int(j)])[int(x)]) for j, x in zip(q, ia)], np.int64)
    nb = np.asarray([int(np.asarray(sl["golds"][int(j)])[int(x)]) for j, x in zip(q, ib)], np.int64)
    out = {"q": q.astype(np.int32), "ia": ia.astype(np.int16), "ib": ib.astype(np.int16), "kind": kind, "nq": np.asarray(nq), "na": na.astype(np.int64), "nb": nb.astype(np.int64)}
    rec = {"dataset": d, "declared_in": G3.DECLARED, "amendment": "A4", "slots": int(q.size), "notes": note, "versions": {}, "nq": nq}
    for _dd, v in [(dd, vv) for dd, vv in CELLS if dd == d]:
        prov = H.RealProv(D)
        reg = H.CompReg(prov, v, D.n)
        names = (["S"] if v == "SRK" else []) + ["W", "T0", "K"]
        G = G1.Graph([reg.get(x) for x in names], D.n)
        lk = np.asarray([is_linked(G, int(x), int(y)) for x, y in zip(na, nb)], bool)
        out[f"linked_{v}"] = lk
        rec["versions"][v] = {k: {"slots": int((kind == c).sum()), "linked": int(lk[kind == c].sum())} for k, c in KIND.items()}
        log(f"{d}/{v}: slots {rec['versions'][v]} ({time.time() - t0:.0f} s)")
    (OUT / d).mkdir(parents=True, exist_ok=True)
    tmp = OUT / d / "slots.tmp.npz"
    np.savez_compressed(tmp, **out)
    tmp.replace(OUT / d / "slots.npz")
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zgp3p_sha256=script_sha256(), zgp3_sha256=G3.script_sha256())
    L2.wjson(OUT / d / "slots.json", rec)
    log(f"{d}: {rel(OUT / d / 'slots.npz')} and slots.json ({rec['seconds']} s)")
    return 0


# ── pools ────────────────────────────────────────────────────────────────────


def pool_of(res, triple, ladder=G3.LADDER):
    """Pool(q) at one triple from a restricted walk run with keep_nodes: the free nodes (base, seeds), the S-tails and the first N_h others of each hop."""
    i1, i2, i3 = triple
    kn = res["kept"]
    s1, t1 = kn["h1"]
    s2, t2 = kn["h2"][i1]
    s3, t3 = kn["h3"][(i1, i2)]
    return np.unique(np.concatenate([res["free"], s1, t1[:ladder[i1]], s2, t2[:ladder[i2]], s3, t3[:ladder[i3]]]).astype(np.int64))


def cmd_precision(a):
    import zgp3s as S
    t0, d = time.time(), a.ds
    rule = Path(a.rule) if a.rule else OUT / "rule_q.json"
    q = json.loads(rule.read_text(encoding="utf-8"))
    fin = q["final"]
    if not fin:
        stop("the quarter has no final rule")
    K = fin["score"]
    k = G3.SCORES.index(K)
    D, gt, sl, L, nq = G3.setup(a)
    want = S.want_questions(d, nq, a.part)[:a.sample]
    sz = np.load(OUT / d / "slots.npz")
    sq_all, kind_all = sz["q"], sz["kind"]
    rec = {"dataset": d, "declared_in": G3.DECLARED, "amendment": "A4", "part": a.part, "score": K, "rule_file": str(rule), "sample": [int(x) for x in want], "work_cap": WORK_CAP, "cells": {},
           "zgp3p_sha256": script_sha256(), "zgp3r_sha256": R.script_sha256(), "zgp3_sha256": G3.script_sha256()}
    for _dd, v in [(dd, vv) for dd, vv in CELLS if dd == d]:
        name = f"{d}/{v}"
        own = q["scores"][K]["own"].get(name)
        tr = {"floor": (0, 0, 0), "final": tuple(fin["triple"])}
        if own:
            tr["own_quarter"] = tuple(own["triple"])
        triples = sorted(set(tr.values()))
        ctx, _reg = G3.real_context(d, v, D, a.host)
        stored, _sh = S.load_cell_chunks(d, v, a.part)
        if stored is None:
            stop(f"{name}: no {a.part} chunks to check the pools against (G3-10)")
        if stored.get("restricted"):
            stop(f"{name}: restricted chunks hold no per-gold coverage for the whole grid: the pools cannot be checked (G3-10); read the part with full chunks")
        ev = S.triple_eval(stored, k, triples)
        goff = np.searchsorted(stored["gq"], np.arange(stored["nfree"].size))
        row_of = {int(x): i for i, x in enumerate(stored["qidx"].tolist())}
        lk = sz[f"linked_{v}"]
        agg = {t: {"questions": 0, "over_work_cap": 0, "pool_nodes": 0, "g3_10_checked": 0,
                   **{f"{nm}_{f}": 0 for nm in ("GP", "CH") for f in ("questions", "slots", "linked", "kept_any", "kept_linked", "pairs")}, "macro_GP": [], "macro_CH": []} for t in triples}
        for qi in want.tolist():
            seeds, golds = sl["seeds"][qi], np.asarray(sl["golds"][qi], np.int64)
            ld, ls = L["dense"](qi), L["spl"](qi)
            ctx.set_question(ld, ls, G3.base_lists(ld, ls, G3.BASE_B[d]))
            res = R.run_question_r(ctx, seeds, golds, int(qi), K, triples, keep_nodes=True)
            r_ = row_of[qi]
            rows = np.flatnonzero(sq_all == qi)
            ia, ib, kd, lkq = sz["ia"][rows].astype(np.int64), sz["ib"][rows].astype(np.int64), kind_all[rows], lk[rows]
            exact = not bool(stored["capped"][r_].any()) and not bool(np.asarray(res["diag"]["capped"]).any())
            for j, t in enumerate(triples):
                pool = pool_of(res, t)
                size = int(res["nfree"]) + int(res["sz1"][0, t[0]]) + int(res["sz2"][0, t[0], t[1]]) + int(res["sz3"][0, t[0], t[1], t[2]])
                if pool.size != size:
                    stop(f"G3-10: {name} q{qi} {t}: the materialised pool has {pool.size} nodes, the sizes say {size}")
                inpool = np.isin(golds, pool)
                A = agg[t]
                if exact:                                             # a capped hop may keep different candidates in the restricted and the full walk: compared only where neither was capped
                    if not np.array_equal(ev["cov"][goff[r_]:goff[r_] + golds.size, j], inpool) or int(ev["size"][r_, j]) != pool.size:
                        stop(f"G3-10 FAILED: {name} q{qi} {t}: the golds in the pool or its size differ from the stored chunks")
                    A["g3_10_checked"] += 1
                pairs = induced_pairs(ctx.G, pool, D.n)
                A["questions"] += 1
                A["pool_nodes"] += int(pool.size)
                if pairs is None:
                    A["over_work_cap"] += 1
                    continue
                both = inpool[ia] & inpool[ib]
                for nm, c in (("GP", 0), ("CH", 1)):
                    m = kd == c
                    if not m.any():
                        continue
                    A[f"{nm}_questions"] += 1
                    A[f"{nm}_slots"] += int(m.sum())
                    A[f"{nm}_linked"] += int(lkq[m].sum())
                    A[f"{nm}_kept_any"] += int(both[m].sum())
                    kl = int((both[m] & lkq[m]).sum())
                    A[f"{nm}_kept_linked"] += kl
                    A[f"{nm}_pairs"] += int(pairs.size)
                    A[f"macro_{nm}"].append(kl / max(int(pairs.size), 1))
        cell = {}
        for nm_t, t in tr.items():
            A = agg[t]
            c = {"triple": list(t), "N": [G3.LADDER[x] for x in t], "questions": A["questions"], "over_work_cap": A["over_work_cap"], "pool_nodes": A["pool_nodes"], "g3_10_checked": A["g3_10_checked"]}
            for nm in ("GP", "CH"):
                if A[f"{nm}_questions"]:
                    c[nm] = {f: A[f"{nm}_{f}"] for f in ("questions", "slots", "linked", "kept_any", "kept_linked", "pairs")}
                    c[nm]["precision_micro"] = A[f"{nm}_kept_linked"] / A[f"{nm}_pairs"] if A[f"{nm}_pairs"] else None
                    c[nm]["precision_macro"] = float(np.mean(A[f"macro_{nm}"]))
            cell[nm_t] = c
            log(f"{name} {nm_t} {t}: {A['questions'] - A['over_work_cap']}/{A['questions']} questions read, G3-10 checked {A['g3_10_checked']}, "
                + ", ".join(f"{nm} kept {A[f'{nm}_kept_linked']} of {A[f'{nm}_slots']} slots over {A[f'{nm}_pairs']} pairs" for nm in ("GP", "CH") if A[f"{nm}_questions"]) + f" ({time.time() - t0:.0f} s)")
        rec["cells"][name] = cell
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    fn = Path(a.out) if a.out else OUT / d / f"precision_{a.part}.json"
    L2.wjson(fn, rec)
    log(f"{d}: {fn} ({rec['seconds']} s)")
    return 0


# ── self-test ────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(23)
    ladder = (0, 1, 3, 6)
    NL = len(ladder)
    all_tr = [(i, j, k_) for i in range(NL) for j in range(i + 1) for k_ in range(j + 1)]
    trials = nlinks = npairs = npools = 0
    for trial in range(70):
        n = int(rng.integers(14, 46))
        keys = G1.toy_graph(rng, n, int(rng.integers(2 * n, 5 * n)))
        if keys.size < 8:
            continue
        comp = H.build_comp(keys, n)
        G = G1.Graph([comp], n)
        adj = {i: set() for i in range(n)}
        for kk in keys.tolist():
            adj[kk // n].add(kk % n)
            adj[kk % n].add(kk // n)
        for _ in range(60):                                                    # is_linked against the adjacency sets
            x, y = (int(v) for v in rng.integers(0, n, 2))
            assert is_linked(G, x, y) == (y in adj[x] and x != y), (trial, x, y)
            nlinks += 1
        for _ in range(8):                                                     # induced pairs against a loop
            pool = np.unique(rng.integers(0, n, int(rng.integers(2, n))))
            want = sorted(min(x, y) * n + max(x, y) for x in pool.tolist() for y in adj[x] if y in set(pool.tolist()) and x != y)
            got = induced_pairs(G, pool, n)
            assert got.tolist() == sorted(set(want)), (trial, got.tolist(), want)
            tot_ = sum(int(len(adj[x])) for x in pool.tolist())
            assert (induced_pairs(G, pool, n, work_cap=0) is None) == (pool.size >= 2 and tot_ > 0)             # the work cap refuses a pool whose nodes carry entries
            npairs += 1
        W = H.Walker(n)
        lev = np.full(n, G1.DEFAULT_LEV, np.int8)
        seeds = np.unique(rng.integers(0, n, int(rng.integers(1, 4))))
        golds = np.unique(rng.integers(0, n, int(rng.integers(1, 6))))
        mod = int(rng.choice([4, 16, 997]))
        dotf = lambda kk_, mod=mod: (((np.minimum(kk_ // n, kk_ % n) * 7919 + np.maximum(kk_ // n, kk_ % n) * 104729) % mod).astype(np.float32) / np.float32(mod))  # noqa: E731
        s_mod = int(rng.choice([1000003, 5, 2]))
        isS = lambda a_, c_, s_mod=s_mod: ((np.minimum(a_, c_) * 31 + np.maximum(a_, c_) * 17) % s_mod == 0)  # noqa: E731
        ubin = lambda c_: ((np.asarray(c_) * 5 + 3) % 7).astype(np.int8)  # noqa: E731
        ctx = G3.Ctx(G, W, lev, n, dotf, isS, ladder=ladder, ubin=ubin, seedw=lambda s_: (np.asarray(s_) * 13 + 5) % 9 + 1)
        ctx.set_question(None, None, np.unique(rng.integers(0, n, int(rng.integers(1, 6)))))
        qseed = int(rng.integers(0, 1000))
        for K in G3.SCORES:
            pick = [all_tr[i] for i in rng.choice(len(all_tr), 4, replace=False)] + [(0, 0, 0)]
            res = R.run_question_r(ctx, seeds, golds, qseed, K, pick, keep_nodes=True)
            rd = R.read_triples(res, sorted(set(pick)))
            for t in sorted(set(pick)):
                pool = pool_of(res, t, ladder)
                assert pool.size == rd[t][0], (trial, K, t, pool.size, rd[t][0])                                # the materialised pool has the stored size
                assert np.array_equal(np.isin(golds, pool), rd[t][1]), (trial, K, t)                            # and holds exactly the golds the places say
                npools += 1
        trials += 1
    assert trials > 40 and nlinks > 2000 and npools > 1000, (trials, nlinks, npools)
    gp = gold_pairs([np.array([5, 7, 9]), np.array([4]), np.array([1, 2])], None)
    assert [x.tolist() for x in gp] == [[0, 0, 0, 2], [0, 0, 1, 0], [1, 2, 2, 1]], gp
    assert gold_pairs([np.arange(12)], (2, 10))[0].size == 0 and gold_pairs([np.arange(3)], (2, 10))[0].size == 3
    log(f"selftest PASS (zgp3p): is_linked equals the adjacency on {nlinks} pairs, induced pairs equal a loop on {npairs} node sets, the materialised pool equals the stored size and the stored gold places on {npools} (score, triple) pools of {trials} toy graphs")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("slots")
    s.add_argument("--ds", required=True)
    s.add_argument("--host", action="store_true")
    s.add_argument("--threads", type=int, default=1)
    p = sub.add_parser("precision")
    p.add_argument("--ds", required=True)
    p.add_argument("--part", required=True)
    p.add_argument("--sample", type=int, default=SAMPLE)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--cap", type=int, default=0)
    p.add_argument("--rule", default="")
    p.add_argument("--out", default="")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if getattr(a, "cap", 0):
        G3.GATHER_CAP = a.cap
    return {"slots": cmd_slots, "precision": cmd_precision}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
