"""HFO: the hub family H against the gold links (UL1's oracle slots), before any training. docs/HFO_HUB_FAMILY_ORACLE_CHECK.md.

    python outputs/mp_unified/zhfo.py run --dataset D [--host] [--threads 8] [--limit Q] [--budget N] [--out DIR]
    python outputs/mp_unified/zhfo.py report                                       # the three -> outputs/hfo/report.*
    python outputs/mp_unified/zhfo.py selftest

H_k: each passage's k best new partners (by the frozen rows' inner product) among the passages sharing a name key of df 2..100,000
with it (MHB's C(a)), its union neighbours removed; label R.hub. D_k, the control: each passage's k best new plain neighbours.
Graded only at the gold slot ends. Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit or record
changes. zmhb, zul1, zap1, zenc, u1b and zu2r are imported, never edited. Gold positions and MuSiQue's decompositions are the
grading only; no form reads one.
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

import zmhb as MB  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import u1b  # noqa: E402
import zap1 as Z1  # noqa: E402
import zenc as E  # noqa: E402
import zu2r as U  # noqa: E402
import zul1 as UL  # noqa: E402

DECLARED = "docs/HFO_HUB_FAMILY_ORACLE_CHECK.md"
OUT = ROOT / "outputs" / "hfo"
SETS = MB.SETS
DOMAIN = MB.DOMAIN
KS = (3, 5, 10)
EPS = MB.EPS
BUDGET_N = 20_000
G4_N = 64
G4_SLACK, G4_REL = 2, 1e-3
CALL = {"carve": "s1eval", "dataset": "musique", "kind": "CH", "gain": 0.02, "min_unlinked": 200,
        "not_below": [("musique", "GP"), ("2wiki", "GP"), ("hotpotqa", "GP")]}
log = MB.log
wjson = MB.wjson


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


# ── the union's neighbours and C(a) membership ──────────────────────────────


def union_nbrs(union, n, nodes, chunk=50_000_000):
    """{node: sorted unique union neighbours} for the given nodes. Keys are lo*n+hi (lo < hi): a node's higher neighbours are one
    searchsorted range; its lower ones come from one pass over the keys."""
    nodes = np.unique(np.asarray(nodes, np.int64))
    lo = np.searchsorted(union, nodes * n)
    hi = np.searchsorted(union, nodes * n + n)
    hs, hd = [], []
    for s in range(0, union.size, chunk):
        u = union[s:s + chunk]
        b = u % n
        m = UL.member(b, nodes)
        if m.any():
            hs.append(b[m])
            hd.append(u[m] // n)
    hs = np.concatenate(hs) if hs else np.zeros(0, np.int64)
    hd = np.concatenate(hd) if hd else np.zeros(0, np.int64)
    o = np.argsort(hs, kind="stable")
    hs, hd = hs[o], hd[o]
    st, en = np.searchsorted(hs, nodes), np.searchsorted(hs, nodes, "right")
    return {int(x): np.unique(np.concatenate([union[lo[i]:hi[i]] % n, hd[st[i]:en[i]]])) for i, x in enumerate(nodes.tolist())}


def in_cand(names, a, cs, dfmax=MB.DFMAX):
    """Which of cs are in C(a): they share with a a key of df 2..dfmax."""
    k = names.keys(a)
    d = names.df[k]
    k = np.sort(k[(d >= 2) & (d <= dfmax)])
    out = np.zeros(len(cs), bool)
    if k.size:
        for i, c in enumerate(np.asarray(cs, np.int64).tolist()):
            kc = names.keys(c)
            out[i] = bool(kc.size) and bool(UL.member(kc, k).any())
    return out


def einrow(q, V):
    """q . each row of V, in the einsum form zmhb.ranks uses for its thresholds."""
    V = np.asarray(V, np.float32)
    return np.einsum("ij,ij->i", np.broadcast_to(np.asarray(q, np.float32), V.shape), V)


# ── the ranks among a's new partners ────────────────────────────────────────


def super_blocks(cases, names, cap):
    """Cases grouped by anchor row, cut so each block's candidate entries stay under cap (an upper bound: the df sum of its keys)."""
    by_q = defaultdict(list)
    for c in cases:
        by_q[c[0]].append(c)
    blocks, block, tot = [], [], 0
    for q in sorted(by_q):
        k = names.keys(by_q[q][0][1])
        dd = names.df[k]
        e = int(dd[(dd >= 2) & (dd <= MB.DFMAX)].sum())
        if block and tot + e > cap:
            blocks.append(block)
            block, tot = [], 0
        block += by_q[q]
        tot += e
    if block:
        blocks.append(block)
    return blocks


def new_ranks(Q, aidx, shards, offs, cases, names, NU, cap=MB.MAX_C // 2):
    """{(q, a, b): (rank_D, rank_H)}: b's rank among a's new plain partners (all but a and its union neighbours) and among its new hub
    partners (C(a) without its union neighbours; 0 when b is not in C(a)). zmhb.ranks' exact pass with the anchor rows as the
    queries, then a itself and its union neighbours scoring above b removed. G2 on every case."""
    out, nbc = {}, {}
    for bi, blk in enumerate(super_blocks(cases, names, cap)):
        tb = time.time()
        rd, rh, _cz, _ab, ic = MB.ranks(Q, None, shards, offs, blk, names)
        for i, (q, a, b) in enumerate(blk):
            if a not in nbc:
                nb = NU[a]
                e = einrow(Q[q], MB.gather_rows(shards, offs, nb)) if nb.size else np.zeros(0, np.float32)
                nbc[a] = (nb, e, in_cand(names, a, nb), float(einrow(Q[q], Q[[q]])[0]))
            nb, e, inc, self_s = nbc[a]
            t = float(einrow(Q[q], Q[[aidx[b]]])[0])
            if nb.size and UL.member(np.asarray([b], np.int64), nb)[0]:
                raise SystemExit(f"gate G2: case {(q, a, b)}: b is a union neighbour of a, yet the slot is unlinked")
            ab_ = e > t + EPS
            sa = int(self_s > t + EPS)
            r_d = rd[(q, b)] - sa - int(ab_.sum())
            r_h = int(rh[i]) - int((ab_ & inc).sum()) if ic[i] else 0
            if r_d < 1 or (ic[i] and r_h < 1) or rd[(q, b)] < 1:
                raise SystemExit(f"gate G2: case {(q, a, b)}: raw {rd[(q, b)]}/{int(rh[i])}, removed {sa}+{int(ab_.sum())}, ranks {r_d}/{r_h}")
            out[(q, a, b)] = (int(r_d), int(r_h))
        log(f"  block {bi + 1}: {len(blk):,} cases {time.time() - tb:.0f} s")
    return out


def direct_ranks(Q, aidx, shards, offs, cases, names, NU):
    """G4's recomputation: one scan per shard with explicit masks (a and its union neighbours out; C(a) minus them in)."""
    qs = sorted({c[0] for c in cases})
    loc = {q: i for i, q in enumerate(qs)}
    Qs = np.asarray(Q[qs], np.float32)
    t = {c: float(einrow(Q[c[0]], Q[[aidx[c[2]]]])[0]) for c in cases}
    cand = {c[1]: names.cand(c[1])[0] for c in cases}
    rD = {c: 1 for c in cases}
    rH = {c: 0 for c in cases}
    for s, p in enumerate(shards):
        D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        lo, hi = int(offs[s]), int(offs[s]) + D.shape[0]
        S = Qs @ D.T
        del D
        for c in cases:
            q, a, _b = c
            above = S[loc[q]] > t[c] + EPS

            def inside(x):
                x = np.asarray(x, np.int64)
                return x[(x >= lo) & (x < hi)] - lo
            m_all = np.ones(hi - lo, bool)
            m_all[inside(np.concatenate([[a], NU[a]]))] = False
            m_hub = np.zeros(hi - lo, bool)
            m_hub[inside(cand[a])] = True
            m_hub[inside(NU[a])] = False
            rD[c] += int((above & m_all).sum())
            rH[c] += int((above & m_hub).sum())
    for c in cases:
        cc = cand[c[1]]
        j = np.searchsorted(cc, c[2])
        rH[c] = rH[c] + 1 if (j < cc.size and cc[j] == c[2]) else 0
    return {c: (rD[c], rH[c]) for c in cases}


def g4(main, direct):
    bad = []
    for c, (d0, h0) in direct.items():
        d1, h1 = main[c]
        ok = abs(d1 - d0) <= G4_SLACK + G4_REL * d0 and abs(h1 - h0) <= G4_SLACK + G4_REL * h0 and (h0 == 0) == (h1 == 0)
        ok = ok and all((d1 <= k) == (d0 <= k) and (0 < h1 <= k) == (0 < h0 <= k) for k in KS)
        if not ok:
            bad.append({"case": list(c), "main": [d1, h1], "direct": [d0, h0]})
    return {"cases": len(direct), "disagree": len(bad), "examples": bad[:5],
            "max_abs_rank_diff": max([abs(main[c][0] - v[0]) for c, v in direct.items()] + [abs(main[c][1] - v[1]) for c, v in direct.items()] + [0])}


def picks(r, k):
    """(D_k links, H_k links) for one end's ranks."""
    return r[0] <= k, 0 < r[1] <= k


def budget(names, NU, nodes, n):
    """Per k: the mean new partners per passage (H: min(k, |C(a) \\ N_U(a)|); D: min(k, n - 1 - |N_U(a)|)) over the sampled nodes."""
    hav, dav = [], []
    for x in np.asarray(nodes, np.int64).tolist():
        c, _ = names.cand(x)
        nb = NU[x]
        hav.append(c.size - int(UL.member(nb, c).sum()) if c.size else 0)
        dav.append(n - 1 - nb.size)
    hav, dav = np.asarray(hav, np.int64), np.asarray(dav, np.int64)
    out = {}
    for k in KS:
        h, d = np.minimum(hav, k).mean(), np.minimum(dav, k).mean()
        out[str(k)] = {"H_per_passage": round(float(h), 4), "D_per_passage": round(float(d), 4), "H_directed_est": int(round(h * n)),
                       "D_directed_est": int(round(d * n)), "H_full_share": round(float((hav >= k).mean()), 4)}
    return {"sample": int(len(hav)), "C_minus_U": {"median": int(np.median(hav)), "p10": int(np.percentile(hav, 10))}, "k": out}


# ── one dataset ─────────────────────────────────────────────────────────────


def ci(t):
    return [round(float(x), 4) for x in t]


def grade_kind(qi, keys, linked, rk, n, nq, hops=None):
    """One kind's slots: U, U | H_k, U | D_k, their gains over U and the diff, each with its interval over questions."""
    res = {"slots": int(keys.size), "linked_U": int(linked.sum()), "share_U": MB.share(int(linked.sum()), keys.size),
           "unlinked": int((~linked).sum()), "k": {}}
    lu = linked.astype(float)
    for k in KS:
        h = linked.copy()
        d = linked.copy()
        for j in np.flatnonzero(~linked):
            x, y = int(keys[j] // n), int(keys[j] % n)
            r1, r2 = rk[(x, y)], rk[(y, x)]
            d1, h1 = picks(r1, k)
            d2, h2 = picks(r2, k)
            d[j], h[j] = d1 or d2, h1 or h2
        gh, gd, df = Z1.boot_diff(h, lu, qi, nq), Z1.boot_diff(d, lu, qi, nq), Z1.boot_diff(h, d, qi, nq)
        row = {"H_new": int(h.sum() - linked.sum()), "D_new": int(d.sum() - linked.sum()), "share_UH": MB.share(int(h.sum()), keys.size),
               "share_UD": MB.share(int(d.sum()), keys.size), "gain_H": ci(gh), "gain_D": ci(gd), "diff": ci(df), "call": Z1.call(df[1], df[2])}
        if hops is not None:
            hq = hops[qi]
            row["by_hops"] = {str(v): {"slots": int((hq == v).sum()), "linked_U": int(linked[hq == v].sum()),
                                       "H_new": int((h & ~linked)[hq == v].sum()), "D_new": int((d & ~linked)[hq == v].sum())}
                              for v in sorted(set(hq.tolist()))}
        res["k"][str(k)] = row
    return res


def cmd_run(a):
    d, t0 = a.dataset, time.time()
    out = Path(a.out) if a.out else OUT / d
    served, freeze = u1b.package_root(a.host)
    man = json.loads((served / d / "DATASET.json").read_text(encoding="utf-8"))
    n, nsha = int(man["nodes"]["n"]), man["nodes"]["sha256"]
    F, shas, _slab = UL.families(served, d, n, nsha, freeze)                        # G1: zul1's own gates
    dom = DOMAIN[d]
    fams = ["W2_S", "F0T", "K"] + (["S"] if dom == "SRK" else [])
    union = F["W2_S"]
    for f in fams[1:]:
        union = np.union1d(union, F[f])
    for f in [x for x in F if x not in fams]:
        del F[f]
    log(f"{d}: union {dom} {union.size:,} pairs")
    grade = json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8"))
    passes, n2, nsha2, _cvt = U.corpus(served, d)
    if (n2, nsha2) != (n, nsha):
        raise SystemExit(f"{d}: the corpus is not DATASET.json's")
    s, g, vocab, _usual = MB.build_entries_keep(passes, n)
    nkeys = len(vocab)
    del vocab
    b_rec = json.loads((UL.OUT / d / "build.json").read_text(encoding="utf-8"))
    if nkeys != b_rec["keys"]:
        raise SystemExit(f"gate G1: {d}: {nkeys:,} keys, UL1's build {b_rec['keys']:,}")
    names = MB.Names(s, g, nkeys, n)
    del s, g
    log(f"{d}: {names.node_key.size:,} name entries, {nkeys:,} keys ({time.time() - t0:.0f} s)")
    shards = E.vec_shards(served, "frozen", d, "docs")
    offs = np.concatenate([[0], np.cumsum([np.load(p, mmap_mode="r").shape[0] for p in shards])]).astype(np.int64)
    if offs[-1] != n:
        raise SystemExit(f"{d}: {offs[-1]} vector rows for {n} nodes")
    posmap = UL.node_positions(served, d) if d == "musique" else None
    rng = np.random.default_rng(0)
    bn = min(n, a.budget)
    bnodes = np.sort(rng.choice(n, size=bn, replace=False)) if bn < n else np.arange(n, dtype=np.int64)
    carves = {}
    for cv in UL.CARVES[d]:
        ids, golds, qsha = UL.carve_questions(d, cv)
        if a.limit:
            ids, golds = ids[:a.limit], golds[:a.limit]
        nq = len(ids)
        kinds = {"GP": UL.gp_slots(golds, n)}
        hops = None
        if d == "musique":
            recs = UL.musique_queries(served, ids)
            qi, keys, hops = UL.chain_slots(recs, ids, golds, posmap, n)
            kinds["CH"] = (qi, keys)
        linked = {k: UL.member(v[1], union) for k, v in kinds.items()}
        if not a.limit:
            for k in kinds:
                want = grade["carves"][cv][dom][k]["forms"]["F0T"]["linked"]
                if int(linked[k].sum()) != want:
                    raise SystemExit(f"gate G3: {d}/{cv}/{k}: the union links {int(linked[k].sum())}, UL1 filed {want}")
        ends = sorted({(int(key // n), int(key % n)) for k, v in kinds.items() for key, ln in zip(v[1].tolist(), linked[k]) if not ln})
        anchors = np.asarray(sorted({x for e in ends for x in e}), np.int64)
        aidx = {int(x): i for i, x in enumerate(anchors.tolist())}
        cases = sorted({(aidx[x], x, y) for e in ends for x, y in (e, e[::-1])})
        NU = union_nbrs(union, n, anchors)
        Q = MB.gather_rows(shards, offs, anchors)
        log(f"{d}/{cv}: {nq} questions; unlinked {', '.join(f'{k} {int((~v).sum())}' for k, v in linked.items())}; {anchors.size:,} anchors, "
            f"{len(cases):,} cases, {sum(x.size for x in NU.values()):,} union neighbours")
        main = new_ranks(Q, aidx, shards, offs, cases, names, NU)
        rk = {(x, y): main[(aidx[x], x, y)] for _q, x, y in cases}
        samp = [cases[i] for i in np.sort(np.random.default_rng(0).choice(len(cases), size=min(G4_N, len(cases)), replace=False))] if cases else []
        gate4 = g4(main, direct_ranks(Q, aidx, shards, offs, samp, names, NU))
        if gate4["disagree"]:
            raise SystemExit(f"gate G4: {d}/{cv}: {gate4['disagree']} of {gate4['cases']} cases disagree: {gate4['examples']}")
        log(f"{d}/{cv}: G4 PASS ({gate4['cases']} cases, max rank diff {gate4['max_abs_rank_diff']}) {time.time() - t0:.0f} s")
        cres = {"questions": nq, "questions_npz_sha256": qsha, "anchors": int(anchors.size), "cases": len(cases), "g4": gate4, "kinds": {}}
        for k, (qi, keys) in kinds.items():
            r = grade_kind(qi, keys, linked[k], rk, n, nq, hops if k == "CH" else None)
            r["alone"] = {f: int(UL.member(keys, F[f]).sum()) for f in fams}
            cres["kinds"][k] = r
        carves[cv] = cres
        del Q, NU, main
    NUb = union_nbrs(union, n, bnodes)
    bud = budget(names, NUb, bnodes, n)
    log(f"{d}: budget over {bud['sample']:,} passages: " + ", ".join(f"k={k} H {v['H_per_passage']} D {v['D_per_passage']}" for k, v in bud["k"].items()))
    res = {"dataset": d, "declared_in": DECLARED, "domain": dom, "dev_limit": int(a.limit or 0), "nodes": n, "union_pairs": int(union.size),
           "families": {f: int(F[f].size) for f in fams}, "keys": nkeys, "entries": int(names.node_key.size), "dfmax": MB.DFMAX, "ks": list(KS),
           "budget": bud, "carves": carves, "inputs": shas,
           "code": {"zhfo_sha256": script_sha256(), "zmhb_sha256": MB.script_sha256(), **UL.codes()}}
    res["seconds"] = round(time.time() - t0, 1)
    wjson(out / "result.json", res)
    log(f"{d}: written {out / 'result.json'} ({res['seconds']} s)")
    return 0


# ── the report: section 6's call ────────────────────────────────────────────


def call_of(R):
    cv = CALL["carve"]
    per_k = {}
    for k in map(str, KS):
        ch = R[CALL["dataset"]]["carves"][cv]["kinds"][CALL["kind"]]
        row = ch["k"][k]
        p1 = row["call"] == "ABOVE" and ch["unlinked"] >= CALL["min_unlinked"]
        p2 = row["gain_H"][0] >= CALL["gain"]
        others = {f"{d}/{kd}": R[d]["carves"][cv]["kinds"][kd]["k"][k]["call"] for d, kd in CALL["not_below"]}
        p3 = all(v != "BELOW" for v in others.values())
        per_k[k] = {"P1": p1, "P2": p2, "P3": p3, "others": others, "diff": row["diff"], "gain_H": row["gain_H"], "gain_D": row["gain_D"]}
    adopt = [k for k in map(str, KS) if per_k[k]["P1"] and per_k[k]["P2"] and per_k[k]["P3"]]
    if adopt:
        out, kk = "ADOPT_H", adopt[0]
    elif any(per_k[k]["P1"] and per_k[k]["P2"] for k in per_k):
        out, kk = "SPLIT", None
    else:
        out, kk = "NO_H", None
    nxt = {"ADOPT_H": "the transfer step (section 7), declared in its own section before it runs",
           "SPLIT": "the mixed form (half the budget from each), graded the same way before any training",
           "NO_H": ("a larger K, graded the same way" if any(per_k[k]["gain_D"][0] >= CALL["gain"] for k in per_k)
                    else "static link growth closes at these budgets")}[out]
    return {"outcome": out, "k": kk, "per_k": per_k, "next": nxt}


def cmd_report(_a):
    R = {}
    for d in SETS:
        p = OUT / d / "result.json"
        if not p.exists():
            raise SystemExit(f"{p} is missing")
        R[d] = json.loads(p.read_text(encoding="utf-8"))
        if R[d]["dev_limit"]:
            raise SystemExit(f"{d}: a dev result (--limit {R[d]['dev_limit']}) is not read")
    rep = {"declared_in": DECLARED, "call": call_of(R), "results": R}
    wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text(render(rep), encoding="utf-8")
    log(f"call: {rep['call']['outcome']} (k {rep['call']['k']}); written {OUT / 'report.md'}")
    return 0


def pc(x):
    return "-" if x is None else f"{100 * x:.1f}"


def render(rep):
    c = rep["call"]
    L = ["# HFO: the hub family H against the gold links", "", f"Declared in {rep['declared_in']}. Call (s1eval): **{c['outcome']}**"
         + (f" at k = {c['k']}" if c["k"] else "") + f". Next: {c['next']}.", ""]
    L += ["| k | P1 diff on MuSiQue CH | P2 gain H | P3 others | gain D |", "|---|---|---|---|---|"]
    for k, v in c["per_k"].items():
        L.append(f"| {k} | {v['diff']} {'ABOVE' if v['P1'] else '-'} | {v['gain_H'][0]} {'ok' if v['P2'] else '-'} | "
                 f"{', '.join(f'{x} {y}' for x, y in v['others'].items())} | {v['gain_D'][0]} |")
    for d, r in rep["results"].items():
        L += ["", f"## {d} ({r['domain']}, union {r['union_pairs']:,} pairs)", ""]
        L.append("Budget (new partners per passage): " + "; ".join(f"k={k}: H {v['H_per_passage']}, D {v['D_per_passage']}" for k, v in r["budget"]["k"].items()))
        for cv, cr in r["carves"].items():
            for kd, kr in cr["kinds"].items():
                L += ["", f"**{cv} {kd}**: {kr['slots']:,} slots, union links {kr['linked_U']:,} ({pc(kr['share_U'])}%); alone "
                      + ", ".join(f"{f} {v:,}" for f, v in kr["alone"].items()), "",
                      "| k | U+H % | U+D % | H new | D new | diff (pts) | call |", "|---|---|---|---|---|---|---|"]
                for k, v in kr["k"].items():
                    L.append(f"| {k} | {pc(v['share_UH'])} | {pc(v['share_UD'])} | {v['H_new']} | {v['D_new']} | "
                             f"{100 * v['diff'][0]:+.2f} [{100 * v['diff'][1]:+.2f}, {100 * v['diff'][2]:+.2f}] | {v['call']} |")
    return "\n".join(L) + "\n"


# ── selftest ────────────────────────────────────────────────────────────────


def selftest():
    import tempfile
    rng = np.random.default_rng(1)
    n, dim, nkeys = 400, 16, 60
    s = rng.integers(0, n, 2400)
    g = rng.integers(0, nkeys, 2400)
    pr = np.unique(s * nkeys + g)
    s, g = pr // nkeys, pr % nkeys
    names = MB.Names(s, g, nkeys, n)
    a_, b_ = rng.integers(0, n, (2, 900))
    m = a_ != b_
    union = np.unique(np.minimum(a_[m], b_[m]) * n + np.maximum(a_[m], b_[m]))
    pairs = {(int(k // n), int(k % n)) for k in union.tolist()}
    nodes = rng.choice(n, 50, replace=False)
    NU = union_nbrs(union, n, nodes, chunk=97)
    for x in nodes.tolist():
        assert NU[x].tolist() == sorted({q for p, q in pairs if p == x} | {p for p, q in pairs if q == x}), x
    for x in nodes[:10].tolist():
        c = set(names.cand(x)[0].tolist())
        assert in_cand(names, x, NU[x]).tolist() == [int(y) in c for y in NU[x].tolist()], x
    with tempfile.TemporaryDirectory() as td:
        V = rng.standard_normal((n, dim)).astype(np.float16)
        paths = []
        for i, (lo, hi) in enumerate(((0, 130), (130, 260), (260, n))):
            p = Path(td) / f"shard_{i:05d}.npy"
            np.save(p, V[lo:hi])
            paths.append(p)
        offs = np.asarray([0, 130, 260, n], np.int64)
        slots = [(int(x), int(y)) for x, y in rng.integers(0, n, (80, 2)) if x != y]
        slots = sorted({(min(x, y), max(x, y)) for x, y in slots} - pairs)
        anchors = np.asarray(sorted({v for e in slots for v in e}), np.int64)
        aidx = {int(x): i for i, x in enumerate(anchors.tolist())}
        cases = sorted({(aidx[x], x, y) for e in slots for x, y in (e, e[::-1])})
        NU = union_nbrs(union, n, anchors)
        Q = MB.gather_rows(paths, offs, anchors)
        old = MB.BLOCK_Q
        MB.BLOCK_Q = 5
        try:
            got = new_ranks(Q, aidx, paths, offs, cases, names, NU, cap=400)
        finally:
            MB.BLOCK_Q = old
        direct = direct_ranks(Q, aidx, paths, offs, cases, names, NU)
        assert got == direct, [(c, got[c], direct[c]) for c in cases if got[c] != direct[c]][:3]
        Vf = V.astype(np.float32)
        for q, a, b in cases:                                                          # brute force with Python sets
            sc = Vf @ Q[q]
            t = float(einrow(Q[q], Q[[aidx[b]]])[0])
            nu = set(NU[a].tolist())
            plain = [c for c in range(n) if c != a and c not in nu and sc[c] > t + EPS]
            cset = set(names.cand(a)[0].tolist())
            hub = [c for c in cset - nu if sc[c] > t + EPS]
            assert got[(q, a, b)] == (1 + len(plain), (1 + len(hub)) if b in cset else 0), (q, a, b)
        assert g4(got, direct)["disagree"] == 0
        rk = {(x, y): got[(aidx[x], x, y)] for _q, x, y in cases}
        keys = np.asarray([x * n + y for x, y in slots], np.int64)
        qi = np.arange(keys.size, dtype=np.int64) // 2
        r = grade_kind(qi, keys, np.zeros(keys.size, bool), rk, n, int(qi.max()) + 1, hops=np.full(int(qi.max()) + 1, 2, np.int64))
        for k in KS:
            want_h = sum((0 < rk[(x, y)][1] <= k) or (0 < rk[(y, x)][1] <= k) for x, y in slots)
            want_d = sum((rk[(x, y)][0] <= k) or (rk[(y, x)][0] <= k) for x, y in slots)
            assert r["k"][str(k)]["H_new"] == want_h and r["k"][str(k)]["D_new"] == want_d, k
            assert r["k"][str(k)]["by_hops"]["2"]["H_new"] == want_h
        bud = budget(names, union_nbrs(union, n, np.arange(n)), np.arange(n), n)
        NUall = union_nbrs(union, n, np.arange(n))
        for k in KS:
            hk = np.mean([min(k, len(set(names.cand(x)[0].tolist()) - set(NUall[x].tolist()))) for x in range(n)])
            assert abs(bud["k"][str(k)]["H_per_passage"] - round(float(hk), 4)) < 1e-9, k
    log(f"selftest: union neighbours, C(a) membership, both new-partner ranks (block pass = direct masks = brute force on {len(cases)} cases), "
        "the slot picks and the budget all agree")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--dataset", required=True, choices=SETS)
    r.add_argument("--host", action="store_true")
    r.add_argument("--threads", default="8")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--budget", type=int, default=BUDGET_N)
    r.add_argument("--out", default=None)
    sp.add_parser("report")
    t = sp.add_parser("selftest")
    t.add_argument("--threads", default="8")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "report":
        return cmd_report(a)
    return selftest()


if __name__ == "__main__":
    sys.exit(main())
