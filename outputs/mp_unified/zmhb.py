"""MHB: the multi-hop bridges our links miss: hub names as entity nodes with the question choosing (Part A, the call), MuSiQue's
unwritten bridges (Part B) and how the datasets build their chains and triplets (Part C). docs/MHB_MULTIHOP_BRIDGES.md.

    python outputs/mp_unified/zmhb.py run --dataset D [--host] [--threads 8] [--out DIR]   # musique, 2wiki or hotpotqa
    python outputs/mp_unified/zmhb.py report                                               # the three -> outputs/mhb/report.*
    python outputs/mp_unified/zmhb.py selftest

Graph level only: nothing trains, no model reads anything, no filed graph, pool, fit or record changes. zul1, zap1, zap1b, zu2r and
zenc are imported, never edited. Labels (gold pairs, decompositions, evidence triplets, types) are the grading and the census only.
"""
import os
import sys

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from array import array  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (HERE, ROOT / "scripts", ROOT / "src", ROOT / "src" / "mp_retrieval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import zenc as E  # noqa: E402  (sets the BLAS threads from --threads before numpy loads)
import numpy as np  # noqa: E402
import oraclepairs as O  # noqa: E402
import u1b  # noqa: E402
import zap1b as AP  # noqa: E402
import zu2r as U  # noqa: E402
import zul1 as UL  # noqa: E402

sys.dont_write_bytecode = True
DECLARED = "docs/MHB_MULTIHOP_BRIDGES.md"
OUT = ROOT / "outputs" / "mhb"
SETS = ("musique", "2wiki", "hotpotqa")
DOMAIN = {"musique": "RK", "2wiki": "SRK", "hotpotqa": "SRK"}
DFMAX = 100_000
KS = (5, 10, 20)
FOUND = 5
CALL = {"dataset": "musique", "carve": "s1eval", "kind": "CH", "k": 10, "share": 0.10, "min_slots": 200}
G4_TOL = 1e-3
G4_TOP = 10
EPS = 1e-6                              # a score counts as above only by more than this (GEMM blocks round differently)
BLOCK_Q = 512
MAX_C = 250_000_000                     # candidate entries held at once (int32), questions split into super-blocks to stay under it
WORD3 = re.compile(r"[a-z0-9]{3,}")
TOPREL = 25


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def wjson(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)


# ── the names: F0T's entries, with the vocabulary and the capitalization rule kept ─


def build_entries_keep(passes, n):
    """zul1.build_entries, step for step, also returning the vocabulary and the corpus's `usual` rule (Part B looks keys up)."""
    cap, low = {}, {}
    for _t, x in passes():
        if x is not None:
            U.count_stat(U.tokens(x), cap, low)

    def usual(k):
        return not (low.get(k, 0) > cap.get(k, 0))

    vocab, src, grp, tsrc, tgrp = {}, array("i"), array("i"), array("i"), array("i")
    seen = 0
    for i, (t, x) in enumerate(passes()):
        seen += 1
        if x is None:
            continue
        names = U.names_of(U.tokens(x), usual)
        for k in names:
            src.append(i)
            grp.append(vocab.setdefault(k, len(vocab)))
        for k in set(AP.runs_of(U.tokens(AP.title_stripped(t)), usual)) - names:
            tsrc.append(i)
            tgrp.append(vocab.setdefault(k, len(vocab)))
    if seen != n:
        raise SystemExit(f"the second pass read {seen} nodes, expected {n}")
    s = np.concatenate([np.asarray(src, np.int64), np.asarray(tsrc, np.int64)])
    g = np.concatenate([np.asarray(grp, np.int64), np.asarray(tgrp, np.int64)])
    return s, g, vocab, usual


def csr(rows, cols, nrows):
    """(indptr, col) with each row's cols ascending."""
    o = np.lexsort((cols, rows))
    ptr = np.zeros(nrows + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=nrows), out=ptr[1:])
    return ptr, cols[o].astype(np.int32)


class Names:
    def __init__(self, s, g, nkeys, n):
        self.df = np.bincount(g, minlength=nkeys).astype(np.int64)
        self.node_ptr, self.node_key = csr(s, g, n)          # passage -> keys
        self.key_ptr, self.key_node = csr(g, s, nkeys)       # key -> passages

    def keys(self, x):
        return self.node_key[self.node_ptr[x]:self.node_ptr[x + 1]]

    def cand(self, x, dfmax=DFMAX):
        """C(x): the passages sharing a key of df 2..dfmax with x, without x; and the count of x's keys above dfmax."""
        k = self.keys(x)
        d = self.df[k]
        walk = k[(d >= 2) & (d <= dfmax)]
        parts = [self.key_node[self.key_ptr[j]:self.key_ptr[j + 1]] for j in walk.tolist()]
        c = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int32)
        c = c[c != x]
        return c, int((d > dfmax).sum())

    def shared(self, a, b):
        """The keys a and b share, and their dfs."""
        k = np.intersect1d(self.keys(a), self.keys(b), assume_unique=True)
        return k, self.df[k]


# ── the question records and the directed chain slots ───────────────────────


def query_records(served, d, ids):
    need, out = set(ids), {}
    for f in sorted((served / d / "queries").glob("*.jsonl")):
        with open(f, "rb") as h:
            for raw in h:
                r = json.loads(raw)
                q = str(r.get("query_id"))
                if q in need:
                    if q in out:
                        raise SystemExit(f"{d}: query {q} is in two query files")
                    out[q] = r
    miss = need - set(out)
    if miss:
        raise SystemExit(f"{d}: {len(miss)} carve ids have no query record")
    return out


def relation(sub_q):
    s = str(sub_q)
    return " ".join(s.split(">>", 1)[1].split()).lower() if ">>" in s else "(free text)"


def chain_directed(recs, ids, golds, posmap, n):
    """One row per (question, distinct pair) as zul1.chain_slots, kept directed: (qi, key, earlier, later, bridge answer, earlier relation,
    hop count). The first dependency (later hop k, earlier hop x) met in k then x order names the direction, as AP1 did."""
    out = []
    for j, (qid, _g) in enumerate(zip(ids, golds)):
        r = recs[qid]
        ctx, dec = UL.field(r, "context_node_ids"), UL.field(r, "question_decomposition")
        pos = [posmap[ctx[int(sub["paragraph_support_idx"])]] for sub in dec]
        seen = {}
        for k, sub in enumerate(dec):
            for x in sorted({int(m) - 1 for m in O.REF_RE.findall(sub["question"])}):
                if 0 <= x < k and pos[k] != pos[x]:
                    key = min(pos[k], pos[x]) * n + max(pos[k], pos[x])
                    seen.setdefault(key, (pos[x], pos[k], str(dec[x]["answer"]), relation(dec[x]["question"])))
        for key in sorted(seen):
            out.append((j, key, *seen[key], len(dec)))
    return out


# ── Part A's ranks: one exact pass over the document vectors per super-block of questions ─


def ranks(Q, qrow, shards, offs, cases, names, bar=None):
    """For each unique (question row, node) of the cases: rank_dense (1 + passages scoring above it). For each case (q, anchor, target):
    rank_hub (1 + passages in C(anchor) scoring above the target; 0 when the target is not in C(anchor)). Exact float32 inner products
    over float16 rows, ties counted as not above."""
    nc = len(cases)
    need = sorted({(q, x) for q, a, b in cases for x in (a, b)})
    nidx = {p: i for i, p in enumerate(need)}
    nq_, nn_ = np.asarray([p[0] for p in need], np.int64), np.asarray([p[1] for p in need], np.int64)
    V = np.asarray(gather_rows(shards, offs, nn_), np.float32)
    thr = np.einsum("ij,ij->i", Q[nq_], V)
    del V
    rd = np.ones(len(need), np.int64)
    rh = np.zeros(nc, np.int64)
    csize = np.zeros(nc, np.int64)
    above = np.zeros(nc, np.int64)
    in_c = np.zeros(nc, bool)
    by_anchor = defaultdict(list)
    for i, (q, a, b) in enumerate(cases):
        by_anchor[a].append(i)
    anchors = sorted(by_anchor)
    cand = {}
    for a in anchors:
        c, ab = names.cand(a)
        cand[a] = c
        for i in by_anchor[a]:
            csize[i], above[i] = c.size, ab
            b = cases[i][2]
            j = np.searchsorted(c, b)
            in_c[i] = j < c.size and c[j] == b
    tot = sum(c.size for c in cand.values())
    if tot > MAX_C:
        raise SystemExit(f"{tot:,} candidate entries held at once; split the questions (MAX_C {MAX_C:,})")
    t_case = np.asarray([thr[nidx[(q, b)]] for q, a, b in cases], np.float32)
    qs = sorted({q for q, _a, _b in cases})
    qloc = {q: i for i, q in enumerate(qs)}
    Qb = Q[qs]
    need_loc = np.asarray([qloc[q] for q in nq_], np.int64)
    case_loc = np.asarray([qloc[q] for q, _a, _b in cases], np.int64)
    hub_cases = np.flatnonzero(in_c)
    for s, p in enumerate(shards):
        D = np.asarray(np.load(p, mmap_mode="r"), np.float32)
        lo, hi = offs[s], offs[s] + D.shape[0]
        for q0 in range(0, len(qs), BLOCK_Q):
            q1 = min(len(qs), q0 + BLOCK_Q)
            S = Qb[q0:q1] @ D.T
            m = (need_loc >= q0) & (need_loc < q1)
            if m.any():
                ii = np.flatnonzero(m)
                rd[ii] += (S[need_loc[ii] - q0] > thr[ii, None] + EPS).sum(1)
            for i in hub_cases[(case_loc[hub_cases] >= q0) & (case_loc[hub_cases] < q1)]:
                c = cand[cases[i][1]]
                j0, j1 = np.searchsorted(c, lo), np.searchsorted(c, hi)
                if j1 > j0:
                    rh[i] += int((S[case_loc[i] - q0, c[j0:j1] - lo] > t_case[i] + EPS).sum())
        if bar:
            bar(s)
    rh[in_c] += 1
    return {p: int(rd[i]) for p, i in nidx.items()}, rh, csize, above, in_c


def gather_rows(paths, offs, rows):
    rows = np.asarray(rows, np.int64)
    s_of = np.searchsorted(offs, rows, "right") - 1
    out = None
    for s in np.unique(s_of):
        a = np.load(paths[s], mmap_mode="r")
        if out is None:
            out = np.empty((rows.size, a.shape[1]), np.float32)
        m = s_of == s
        out[m] = a[rows[m] - offs[s]]
    return out if out is not None else np.zeros((0, 1), np.float32)


# ── Part B: the class of an unwritten bridge ────────────────────────────────


def words3(s):
    return set(WORD3.findall(str(s).lower()))


def bridge_class(ans, early_keys, late_keys, late_text, late_title, key_id, df, usual):
    """Section 4's classes, first that fits. early_keys/late_keys: sets of key strings."""
    k = U.norm_key(str(ans).split())
    if k in early_keys and k in late_keys:
        return ("B2" if 2 <= df.get(k, 0) <= 25 else "B1"), k
    if not U.names_of(U.tokens(str(ans)), usual):
        return ("B3w" if str(ans).lower() in late_text.lower() else "B3u"), k
    if str(ans).lower() in late_text.lower():
        return "B4", k
    aw, tw = words3(ans), words3(AP.title_stripped(late_title))
    ts, al = AP.title_stripped(late_title).lower(), str(ans).lower()
    if (aw & tw) or (len(ts) >= 3 and (ts in al or al in ts)):
        return "B5", k
    if any(aw & words3(x) for x in late_keys):
        return "B6", k
    return "B7", k


# ── one dataset ─────────────────────────────────────────────────────────────


def node_text(served, d, want):
    out = {}
    with open(served / d / "nodes.jsonl", "rb") as f:
        for i, raw in enumerate(f):
            if i in want:
                r = json.loads(raw)
                out[i] = (u1b.title_of(r), r.get("text") or "")
    return out


def share(x, n):
    return round(float(x) / n, 4) if n else None


def band(dfv):
    return "26-100" if dfv <= 100 else "101-1k" if dfv <= 1000 else "1k-10k" if dfv <= 10_000 else "10k-100k" if dfv <= DFMAX else ">100k"


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
    log(f"{d}: union {dom} {union.size:,} pairs ({', '.join(f'{f} {F[f].size:,}' for f in fams)})")
    grade = json.loads((UL.OUT / d / "grade.json").read_text(encoding="utf-8"))
    # names
    passes, n2, nsha2, _cvt = U.corpus(served, d)
    if (n2, nsha2) != (n, nsha):
        raise SystemExit(f"{d}: the corpus is not DATASET.json's")
    s, g, vocab, usual = build_entries_keep(passes, n)
    b_rec = json.loads((UL.OUT / d / "build.json").read_text(encoding="utf-8"))
    if len(vocab) != b_rec["keys"]:
        raise SystemExit(f"gate G1: {d}: {len(vocab):,} keys, UL1's build {b_rec['keys']:,}")
    names = Names(s, g, len(vocab), n)
    del s, g
    log(f"{d}: {names.node_key.size:,} entries, {len(vocab):,} keys; df > 25: {(names.df > 25).sum():,}, df > {DFMAX:,}: {(names.df > DFMAX).sum():,}")
    # vectors
    shards = E.vec_shards(served, "frozen", d, "docs")
    offs = np.concatenate([[0], np.cumsum([np.load(p, mmap_mode="r").shape[0] for p in shards])]).astype(np.int64)
    if offs[-1] != n:
        raise SystemExit(f"{d}: {offs[-1]} vector rows for {n} nodes")
    qpaths = E.vec_shards(served, "frozen", d, "queries")
    qoffs = np.concatenate([[0], np.cumsum([np.load(p, mmap_mode="r").shape[0] for p in qpaths])]).astype(np.int64)
    qids_all = json.loads((served / d / "queries" / "query_ids.json").read_text(encoding="utf-8"))
    qmap = {str(q): j for j, q in enumerate(qids_all)}
    with np.load(served / d / "retrieval_cache" / "dense_top1000.npz") as z:
        rec_ids, rec_sc = z["ids"][:, :G4_TOP].astype(np.int64), z["scores"][:, :G4_TOP].astype(np.float32)
    posmap = UL.node_positions(served, d) if d == "musique" else None
    res = {"dataset": d, "declared_in": DECLARED, "domain": dom, "families": {f: int(F[f].size) for f in fams}, "union_pairs": int(union.size),
           "keys": len(vocab), "entries": int(names.node_key.size), "dfmax": DFMAX, "carves": {}, "code": {"zmhb_sha256": script_sha256(), **UL.codes()},
           "inputs": shas}
    for cv in UL.CARVES[d]:
        ids, golds, qsha = UL.carve_questions(d, cv)
        recs = query_records(served, d, ids)
        kinds = {"GP": UL.gp_slots(golds, n)}
        chd = None
        if d == "musique":
            chd = chain_directed(recs, ids, golds, posmap, n)
            qi, keys, _h = UL.chain_slots(recs, ids, golds, posmap, n)
            if not (np.array_equal(qi, np.asarray([r[0] for r in chd], np.int64)) and np.array_equal(keys, np.asarray([r[1] for r in chd], np.int64))):
                raise SystemExit(f"gate G2: {d}/{cv}: the directed chain slots are not zul1's")
            kinds["CH"] = (qi, keys)
        linked = {k: UL.member(v[1], union) for k, v in kinds.items()}
        for k in kinds:
            want = grade["carves"][cv][dom][k]["forms"]["F0T"]["linked"]
            if int(linked[k].sum()) != want:
                raise SystemExit(f"gate G3: {d}/{cv}/{k}: the union links {int(linked[k].sum())}, UL1 filed {want}")
        # Part A cases: every unlinked slot, from both ends
        slots_un = {k: [(int(q), int(key // n), int(key % n)) for q, key, ln in zip(v[0], v[1], linked[k]) if not ln] for k, v in kinds.items()}
        allc = sorted({(q, a_, b_) for k in slots_un for q, x, y in slots_un[k] for a_, b_ in ((x, y), (y, x))})
        qrows = np.asarray([qmap[str(i)] for i in ids], np.int64)
        Q = gather_rows(qpaths, qoffs, qrows)
        # G4: the record's dense scores of each case question's top ids
        cq = sorted({q for q, _a, _b in allc})
        top, tsc = rec_ids[qrows[cq]], rec_sc[qrows[cq]]
        mine = np.einsum("ij,ikj->ik", Q[cq], gather_rows(shards, offs, top.ravel()).reshape(top.shape[0], top.shape[1], -1))
        g4 = float(np.abs(mine - tsc).max()) if top.size else 0.0
        if g4 > G4_TOL:
            raise SystemExit(f"gate G4: {d}/{cv}: dense scores differ from the record's by {g4:.2e}")
        log(f"{d}/{cv}: {len(ids)} questions; unlinked {', '.join(f'{k} {len(v)}' for k, v in slots_un.items())}; {len(allc):,} cases; G4 {g4:.1e}")
        # super-blocks of questions under MAX_C candidate entries
        rdense, rhub, csz, abv, inc = {}, {}, {}, {}, {}
        cs_by_q = defaultdict(list)
        for c in allc:
            cs_by_q[c[0]].append(c)
        est = {}
        block, tot = [], 0
        blocks = []
        for q in cq:
            e = sum(est.setdefault(x, int(names.df[names.keys(x)][(names.df[names.keys(x)] <= DFMAX)].sum())) for x in {c[1] for c in cs_by_q[q]})
            if block and tot + e > MAX_C // 2:
                blocks.append(block)
                block, tot = [], 0
            block += cs_by_q[q]
            tot += e
        if block:
            blocks.append(block)
        for bi, blk in enumerate(blocks):
            tb = time.time()
            rd, rh, cz, ab, ic = ranks(Q, None, shards, offs, blk, names)
            rdense.update(rd)
            for i, c in enumerate(blk):
                rhub[c], csz[c], abv[c], inc[c] = int(rh[i]), int(cz[i]), int(ab[i]), bool(ic[i])
            log(f"{d}/{cv}: super-block {bi + 1}/{len(blocks)} ({len(blk):,} cases) {time.time() - tb:.0f} s")
        cres = {"questions": len(ids), "questions_npz_sha256": qsha, "g4_max_abs": g4, "slots": {k: int(v[1].size) for k, v in kinds.items()},
                "linked": {k: int(v.sum()) for k, v in linked.items()}, "partA": {}}
        for k, sl in slots_un.items():
            cres["partA"][k] = part_a(sl, rdense, rhub, csz, abv, inc, names)
        if chd is not None:
            cres["partB"] = part_b(served, d, chd, linked["CH"], names, vocab, usual, rdense, rhub)
        cres["partC"] = part_c(d, recs, ids, kinds, F, fams, union, chd, served, golds, n)
        res["carves"][cv] = cres
    res["seconds"] = round(time.time() - t0, 1)
    wjson(out / "result.json", res)
    log(f"{d}: written {out / 'result.json'} ({res['seconds']} s)")
    return 0


def part_a(sl, rdense, rhub, csz, abv, inc, names):
    """Section 3's read over one kind's unlinked slots."""
    nsl = len(sl)
    shared_any, bands, only_above = 0, Counter(), 0
    hub = {k: 0 for k in KS}
    new = {k: 0 for k in KS}
    hub_f = {k: 0 for k in KS}
    new_f = {k: 0 for k in KS}
    found_slots = 0
    sizes = []
    for q, x, y in sl:
        kk, dd = names.shared(x, y)
        if kk.size:
            shared_any += 1
            bands[band(int(dd.min()))] += 1
            if (dd > DFMAX).all():
                only_above += 1
        ends = [(x, y), (y, x)]
        f_any = False
        for k in KS:
            h = n_ = hf = nf = False
            for a_, b_ in ends:
                r = rhub[(q, a_, b_)]
                ok = 0 < r <= k
                nw = ok and rdense[(q, b_)] > k
                fd = rdense[(q, a_)] <= FOUND
                h |= ok
                n_ |= nw
                hf |= ok and fd
                nf |= nw and fd
            hub[k] += h
            new[k] += n_
            hub_f[k] += hf
            new_f[k] += nf
        for a_, _b in ends:
            f_any |= rdense[(q, a_)] <= FOUND
            sizes.append(csz[(q, a_, _b)])
        found_slots += f_any
    sizes = np.asarray(sizes, np.int64) if sizes else np.zeros(1, np.int64)
    return {"unlinked_slots": nsl, "share_any_key": share(shared_any, nsl), "min_df_band": dict(bands), "only_above_dfmax": share(only_above, nsl),
            "HUB": {k: share(v, nsl) for k, v in hub.items()}, "NEW": {k: share(v, nsl) for k, v in new.items()},
            "slots_with_a_found_end": found_slots, "HUB_found": {k: share(v, nsl) for k, v in hub_f.items()},
            "NEW_found": {k: share(v, nsl) for k, v in new_f.items()},
            "C_size": {"median": int(np.median(sizes)), "p90": int(np.percentile(sizes, 90)), "max": int(sizes.max())}}


def part_b(served, d, chd, linked_ch, names, vocab, usual, rdense, rhub):
    inv = None
    need = {r[2] for r in chd} | {r[3] for r in chd}
    tx = node_text(served, d, need)
    df = {}

    def kset(x):
        nonlocal inv
        if inv is None:
            inv = np.empty(len(vocab), object)
            for kk, v in vocab.items():
                inv[v] = kk
        return {inv[j] for j in names.keys(x).tolist()}

    tab = {"unlinked": Counter(), "all": Counter()}
    hub10 = Counter()
    early_written = Counter()
    ex = defaultdict(list)
    for (q, _key, e, l_, ans, _rel, _h), ln in zip(chd, linked_ch):
        ek, lk = kset(e), kset(l_)
        kk = U.norm_key(str(ans).split())
        if kk in vocab:
            df[kk] = int(names.df[vocab[kk]])
        cls, _k = bridge_class(ans, ek, lk, tx[l_][1], tx[l_][0], vocab, df, usual)
        tab["all"][cls] += 1
        if not ln:
            tab["unlinked"][cls] += 1
            r = rhub.get((q, e, l_), 0)
            hub10[cls] += 0 < r <= 10
            early_written[cls] += str(ans).lower() in tx[e][1].lower()
            if len(ex[cls]) < 6:
                ex[cls].append({"ans": ans, "earlier": tx[e][0], "later": tx[l_][0]})
    if tab["unlinked"]["B2"]:
        raise SystemExit(f"gate G5: {d}: {tab['unlinked']['B2']} unlinked chain slots share the bridge key at df <= 25")
    nu = sum(tab["unlinked"].values())
    return {"unlinked": dict(tab["unlinked"]), "all": dict(tab["all"]), "unlinked_share": {c: share(v, nu) for c, v in tab["unlinked"].items()},
            "hub10_from_earlier": {c: share(hub10[c], tab["unlinked"][c]) for c in tab["unlinked"]},
            "ans_written_in_earlier": {c: share(early_written[c], tab["unlinked"][c]) for c in tab["unlinked"]}, "examples": dict(ex)}


def fam_rows(keys, F, fams, union):
    out = {f: UL.member(keys, F[f]) for f in fams}
    out["union"] = UL.member(keys, union)
    return out


def tally(groups, fr):
    """{group: {n, family: share}} over slot rows."""
    out = {}
    for gname, idx in groups.items():
        idx = np.asarray(idx, np.int64)
        out[gname] = {"n": int(idx.size), **{f: share(v[idx].sum(), idx.size) for f, v in fr.items()}}
    return out


def part_c(d, recs, ids, kinds, F, fams, union, chd, served, golds, n):
    qi, keys = kinds["GP"]
    fr = fam_rows(keys, F, fams, union)
    out = {}
    if d in ("2wiki", "hotpotqa"):
        typ = [str(recs[ids[int(q)]].get("type") or "unknown") for q in qi]
        out["GP_by_type"] = tally({t: [i for i, x in enumerate(typ) if x == t] for t in sorted(set(typ))}, fr)
    if d == "2wiki":
        rel = []
        for q in qi:
            ev = UL.field(recs[ids[int(q)]], "evidences") if recs[ids[int(q)]].get("evidences") else []
            rel.append(str(ev[0][1]) if ev else "(none)")
        top = [r for r, _c in Counter(rel).most_common(TOPREL)]
        out["GP_by_first_relation"] = tally({r: [i for i, x in enumerate(rel) if x == r] for r in top}, fr)
        out["bridge_object"] = bridge_object(served, d, recs, ids, golds)
    if d == "musique" and chd is not None:
        ck = np.asarray([r[1] for r in chd], np.int64)
        cfr = fam_rows(ck, F, fams, union)
        hops = [r[6] for r in chd]
        rels = [r[5] for r in chd]
        out["CH_by_hops"] = tally({str(h): [i for i, x in enumerate(hops) if x == h] for h in sorted(set(hops))}, cfr)
        top = [r for r, _c in Counter(rels).most_common(TOPREL)]
        out["CH_by_relation"] = tally({r: [i for i, x in enumerate(rels) if x == r] for r in top}, cfr)
    return out


def bridge_object(served, d, recs, ids, golds):
    """2Wiki compositional and inference questions: the first triplet's object (the bridge). Is it written in the subject's gold passage,
    and is a gold passage titled as it (stripped title, case-insensitive)."""
    want = {int(x) for g in golds for x in g.tolist()}
    tx = node_text(served, d, want)
    cnt = Counter()
    for qid, g in zip(ids, golds):
        r = recs[qid]
        if str(r.get("type")) not in ("compositional", "inference"):
            continue
        ev = UL.field(r, "evidences") if r.get("evidences") else []
        if not ev:
            cnt["no_evidence"] += 1
            continue
        s_, _r, o_ = (str(x) for x in ev[0][:3])
        gl = [int(x) for x in g.tolist()]
        st = {x: AP.title_stripped(tx[x][0]).lower() for x in gl}
        subj = [x for x in gl if st[x] == AP.title_stripped(s_).lower() or tx[x][0].lower() == s_.lower()]
        cnt["questions"] += 1
        if not subj:
            cnt["no_subject_passage"] += 1
            continue
        cnt["written_in_subject"] += any(o_.lower() in tx[x][1].lower() for x in subj)
        cnt["gold_titled_object"] += any(st[x] == AP.title_stripped(o_).lower() for x in gl)
    nq = cnt["questions"]
    return {**dict(cnt), "written_share": share(cnt["written_in_subject"], nq - cnt["no_subject_passage"]),
            "titled_share": share(cnt["gold_titled_object"], nq - cnt["no_subject_passage"])}


# ── the report and the call ─────────────────────────────────────────────────


def cmd_report(_a):
    R = {}
    for d in SETS:
        p = OUT / d / "result.json"
        if not p.is_file():
            raise SystemExit(f"no {p}")
        R[d] = json.loads(p.read_text(encoding="utf-8"))
    c = CALL
    a = R[c["dataset"]]["carves"][c["carve"]]["partA"][c["kind"]]
    nsl = a["unlinked_slots"]
    newk = a["NEW"][c["k"]] if c["k"] in a["NEW"] else a["NEW"][str(c["k"])]
    if nsl < c["min_slots"]:
        verdict = "INCOMPLETE"
    elif newk >= c["share"]:
        verdict = "SIGNAL"
    else:
        verdict = "NO_SIGNAL (" + ("RARE" if (a["share_any_key"] or 0) < c["share"] else "NOT_SELECTIVE") + ")"
    rep = {"declared_in": DECLARED, "call": {**c, "unlinked_slots": nsl, "NEW@k": newk, "verdict": verdict}, "results": R}
    wjson(OUT / "report.json", rep)
    (OUT / "report.md").write_text(render(rep), encoding="utf-8")
    log(f"MHB: {verdict} (NEW@10 {newk} over {nsl} unlinked chain slots)")
    return 0


def render(rep):
    c = rep["call"]
    L = [f"# MHB: the multi-hop bridges our links miss ({rep['declared_in']})", "",
         f"## Call on {c['dataset']}/{c['carve']} {c['kind']}: {c['verdict']}", "",
         f"NEW@{c['k']} = {c['NEW@k']} over {c['unlinked_slots']} unlinked slots (rule: SIGNAL iff >= {c['share']} with >= {c['min_slots']} slots).", "",
         "## Part A: hub names as entity nodes, the question choosing", "",
         "| dataset | carve | kind | unlinked | any key | only > DFMAX | HUB@5 | HUB@10 | HUB@20 | NEW@5 | NEW@10 | NEW@20 | NEW@10 found | C median | C p90 |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for d, r in rep["results"].items():
        for cv, x in r["carves"].items():
            for k, a in x["partA"].items():
                g = lambda m, kk: m.get(kk, m.get(str(kk)))  # noqa: E731
                L.append(f"| {d} | {cv} | {k} | {a['unlinked_slots']} | {a['share_any_key']} | {a['only_above_dfmax']} | "
                         + " | ".join(str(g(a['HUB'], kk)) for kk in KS) + " | " + " | ".join(str(g(a['NEW'], kk)) for kk in KS)
                         + f" | {g(a['NEW_found'], 10)} | {a['C_size']['median']} | {a['C_size']['p90']} |")
    L += ["", "## Part B: MuSiQue's unwritten bridges (unlinked chain slots)", ""]
    for cv, x in rep["results"]["musique"]["carves"].items():
        b = x.get("partB")
        if not b:
            continue
        L += [f"### {cv}", "", "| class | unlinked | share | all slots | HUB@10 from the earlier end | ans written in the earlier passage |", "|---|---:|---:|---:|---:|---:|"]
        for cl in sorted(set(b["all"]) | set(b["unlinked"])):
            L.append(f"| {cl} | {b['unlinked'].get(cl, 0)} | {b['unlinked_share'].get(cl)} | {b['all'].get(cl, 0)} | {b['hub10_from_earlier'].get(cl)} | "
                     f"{b['ans_written_in_earlier'].get(cl)} |")
        L.append("")
    L += ["## Part C: how the chains and triplets are built (shares of slots each family links)", ""]
    for d, r in rep["results"].items():
        for cv, x in r["carves"].items():
            for tname, t in x["partC"].items():
                if tname == "bridge_object":
                    L += [f"- {d}/{cv} bridge object: {json.dumps(t)}", ""]
                    continue
                fams = [f for f in next(iter(t.values())).keys() if f != "n"]
                L += [f"### {d}/{cv} {tname}", "", "| group | n | " + " | ".join(fams) + " |", "|---|---:|" + "---:|" * len(fams)]
                for gname, row in t.items():
                    L.append(f"| {gname} | {row['n']} | " + " | ".join(str(row[f]) for f in fams) + " |")
                L.append("")
    return "\n".join(L) + "\n"


# ── self-test (toy, no package) ─────────────────────────────────────────────


def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    n, dim, nkeys = 300, 16, 40
    s = rng.integers(0, n, 1500)
    g = rng.integers(0, nkeys, 1500)
    pr = np.unique(s * nkeys + g)
    s, g = pr // nkeys, pr % nkeys
    names = Names(s, g, nkeys, n)
    for x in (0, 5, 77):
        c, _ab = names.cand(x)
        brute = sorted({int(p) for k in names.keys(x).tolist() for p in s[g == k].tolist() if names.df[k] >= 2} - {x})
        assert c.tolist() == brute, x
    with tempfile.TemporaryDirectory() as td:
        V = rng.standard_normal((n, dim)).astype(np.float16)
        paths = []
        for i, (lo, hi) in enumerate(((0, 120), (120, 250), (250, n))):
            p = Path(td) / f"shard_{i:05d}.npy"
            np.save(p, V[lo:hi])
            paths.append(p)
        offs = np.asarray([0, 120, 250, n], np.int64)
        Q = rng.standard_normal((7, dim)).astype(np.float32)
        cases = [(q, int(a_), int(b_)) for q in range(7) for a_, b_ in rng.integers(0, n, (4, 2)) if a_ != b_]
        global BLOCK_Q
        old, BLOCK_Q = BLOCK_Q, 3
        try:
            rd, rh, cz, _ab, ic = ranks(Q, None, paths, offs, cases, names)
        finally:
            BLOCK_Q = old
        Vf = V.astype(np.float32)
        S = Q @ Vf.T
        for (q, x), r in rd.items():
            assert r == 1 + int((S[q] > S[q, x] + EPS).sum()), (q, x)
        for i, (q, a_, b_) in enumerate(cases):
            c, _ = names.cand(a_)
            assert cz[i] == c.size
            if b_ in set(c.tolist()):
                assert ic[i] and rh[i] == 1 + int((S[q, c] > S[q, b_] + EPS).sum()), i
            else:
                assert not ic[i] and rh[i] == 0, i
    rec = {"q1": {"context_node_ids": ["n0", "n1", "n2"], "question_decomposition": [
        {"answer": "Steve Hillage", "paragraph_support_idx": 2, "question": "Green >> performer"},
        {"answer": "Miquette Giraudy", "paragraph_support_idx": 0, "question": "#1 >> spouse"}]}}
    ch = chain_directed(rec, ["q1"], [np.asarray([0, 2])], {"n0": 0, "n1": 1, "n2": 2}, 10)
    assert ch == [(0, 0 * 10 + 2, 2, 0, "Steve Hillage", "performer", 2)], ch
    usual = lambda k: True  # noqa: E731
    assert bridge_class("1990", set(), set(), "born in 1990", "X", {}, {}, usual)[0] == "B3w"
    assert bridge_class("Steve Hillage", {"steve hillage"}, {"steve hillage"}, "", "Y", {}, {"steve hillage": 40}, usual)[0] == "B1"
    assert bridge_class("Steve Hillage", {"steve hillage"}, {"steve hillage"}, "", "Y", {}, {"steve hillage": 4}, usual)[0] == "B2"
    assert bridge_class("Steve Hillage", {"steve hillage"}, set(), "He married Steve Hillage", "Y", {}, {}, usual)[0] == "B4"
    assert bridge_class("Steve Hillage", {"steve hillage"}, set(), "He married", "Hillage (band)", {}, {}, usual)[0] == "B5"
    assert bridge_class("Steve Hillage", {"steve hillage"}, {"hillage records"}, "He married", "Z", {}, {}, usual)[0] == "B6"
    assert bridge_class("Steve Hillage", {"steve hillage"}, set(), "He married", "Z", {}, {}, usual)[0] == "B7"
    log("selftest: the hop candidates, both ranks, the directed chain slot and the bridge classes all equal brute force")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--dataset", required=True, choices=SETS)
    r.add_argument("--host", action="store_true")
    r.add_argument("--threads", default="8")
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
