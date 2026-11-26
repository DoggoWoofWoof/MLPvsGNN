"""GP3's restricted walker (draft for amendment A4, docs/GP3_NODE_COUNT_RULE.md; NOT run on any real row before A4 is committed). The full grid of zgp3.py walks, per question, 7 scores x 45
(N_1, N_2) hop-3 layers over one hop-3 table built from the union of every kept set; for a heavy cell that is too costly for the rest of dev (about 1,125 questions). The rest is read at
the quarter's final rule only, so this file walks ONE score and the few triples the grade reads (the final triple, the cell's quarter optimum, the three P2 triples and the floor), with the
same layers, keys and arithmetic as zgp3.py (it imports zgp3 read-only): on a question where no hop trips the gather cap the sizes and the gold bins at those triples are EQUAL to the full grid's
(the self-test and the equality command check it); under the cap the tables differ (they are built from the kept sets of one score), so a capped question is recorded as capped and read as the walk with its cap.

    python outputs/mp_unified/zgp3r.py selftest                                         # restricted == full grid on toy graphs (no real data)
    python outputs/mp_unified/zgp3r.py equal --ds D [--n 8] [--host]                    # restricted == full grid on the first quarter questions of a real cell, no cap trip
    python outputs/mp_unified/zgp3r.py extract --ds D --part r --score K --triples i-j-k,i-j-k [--cap N] [--shard I/N] [--host] [--threads 1]
The chunks are written like zgp3.py's (same names, same arrays with ONE score on the score axis) plus `rscore`, `rtriples`, `zrsha`; their `zsha` is zgp3.py's sha256, so G3-7's one-sha rule reads zgp3.py's.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zgp3 as G3  # noqa: E402  (imports zgp1 first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

G1, H, L2 = G3.G1, G3.H, G3.L2
LADDER, SCORES, CELLS, OUT, EMPTY = G3.LADDER, G3.SCORES, G3.CELLS, G3.OUT, G3.EMPTY
log, rel, stop = G3.log, G3.rel, G3.stop


def script_sha256():
    return hashlib.sha256(Path(__file__).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def parse_triples(s, nl=len(LADDER)):
    """'i-j-k,i-j-k' (ladder indices with i >= j >= k) -> sorted tuples; the floor (0, 0, 0) is always read (G3-4)."""
    out = {(0, 0, 0)}
    for x in [t for t in s.split(",") if t.strip()]:
        i, j, k = (int(v) for v in x.split("-"))
        if not (nl > i >= j >= k >= 0):
            stop(f"triple {x} is not a triple of the ladder")
        out.add((i, j, k))
    return sorted(out)


def run_question_r(ctx, seeds, golds, qseed, K, triples, keep_nodes=False):
    """zgp3.run_question for ONE score K and the given triples (ladder indices). Returns the same arrays with the score axis of length 1; entries of triples that are not read stay -1 (sizes)
    and NL (bins). Nothing is computed beyond what the triples need: hop 1 to the largest N_1 read, hop 2 once per distinct N_1, hop 3 once per distinct (N_1, N_2).
    keep_nodes (A4, the precision read): also return, per layer, the S-tails and the kept others in order, so a pool can be materialised (zgp3p.py)."""
    ladder = ctx.ladder
    NL = len(ladder)
    n, G, W, lev = ctx.n, ctx.G, ctx.W, ctx.lev
    seeds = np.asarray(seeds, np.int64)
    golds = np.asarray(golds, np.int64)
    triples = sorted({tuple(int(x) for x in t) for t in triples})
    need1 = sorted({t[0] for t in triples})
    need2 = {i1: sorted({t[1] for t in triples if t[0] == i1}) for i1 in need1}
    need3 = {(i1, i2): sorted({t[2] for t in triples if t[0] == i1 and t[1] == i2}) for i1 in need1 for i2 in need2[i1]}
    free = np.union1d(ctx.base, seeds).astype(np.int64)
    L1, L2_ = G1.levels(W, G, lev, seeds)
    try:
        ng = golds.size
        glev = np.zeros(ng, np.int8)
        for j in range(ng):
            g = int(golds[j])
            glev[j] = int(lev[g]) if lev[g] < G1.DEFAULT_LEV else G1.gold_distance(G, lev, g)
        gfree = np.isin(golds, free)
        sz1 = np.full((1, NL), -1, np.int32)
        sz2 = np.full((1, NL, NL), -1, np.int32)
        sz3 = np.full((1, NL, NL, NL), -1, np.int32)
        gb1 = np.full((1, ng), NL, np.int8)
        gb2 = np.full((1, NL, ng), NL, np.int8)
        gb3 = np.full((1, NL, NL, ng), NL, np.int8)
        diag = {"seeds": int(seeds.size), "free": int(free.size), "L1": int(L1.size), "L2": int(L2_.size), "links": [0, 0, 0], "capped": [False, False, False], "ncand": [0, 0, 0], "raw": [0, 0, 0]}
        pp = K == "PP"
        kn = {"h1": None, "h2": {}, "h3": {}}
        # hop 1
        tab1, cap = G3.hop_table(ctx, seeds, 1)
        diag["links"][0], diag["capped"][0], diag["raw"][0] = int(tab1.c.size), cap, ctx.raw_last
        lw1 = G3.link_weights(ctx, tab1, np.unique(seeds), ctx.seed_mass(np.unique(seeds))) if pp else None
        r = G3.layer(ctx, tab1, np.ones(tab1.c.size, bool), K, ladder[need1[-1]], qseed, golds, free, lw=lw1)
        diag["ncand"][0] = r["ncand"]
        kn["h1"] = (r["s_nodes"], r["tops"])
        for i1 in need1:
            sz1[0, i1] = G3.size_at(r, ladder[i1])
        gb1[0] = G3.bins_of(r["gpos"], ladder)
        kept1 = (np.concatenate([r["s_nodes"], r["tops"]]), np.concatenate([np.zeros(r["s_nodes"].size, np.int32), np.arange(1, r["tops"].size + 1, dtype=np.int32)]),
                 np.concatenate([r["s_mass"], r["tops_mass"]]))
        # hop 2
        tab2, cap = G3.hop_table(ctx, kept1[0], 2)
        diag["links"][1], diag["capped"][1], diag["raw"][1] = int(tab2.c.size), cap, ctx.raw_last
        p2 = G3.posmap(ctx, tab2.a, kept1[0], kept1[1])
        lw2 = G3.link_weights(ctx, tab2, kept1[0], kept1[2]) if pp else None
        kept2 = {}
        for i1 in need1:
            r = G3.layer(ctx, tab2, p2 <= ladder[i1], K, ladder[need2[i1][-1]], qseed, golds, free, lw=lw2)       # only the first N_2 (largest read) kept: the order is a prefix
            diag["ncand"][1] = max(diag["ncand"][1], r["ncand"])
            for i2 in need2[i1]:
                sz2[0, i1, i2] = G3.size_at(r, ladder[i2])
            gb2[0, i1] = G3.bins_of(r["gpos"], ladder)
            kn["h2"][i1] = (r["s_nodes"], r["tops"])
            kept2[i1] = (np.concatenate([r["s_nodes"], r["tops"]]), np.concatenate([np.zeros(r["s_nodes"].size, np.int32), np.arange(1, r["tops"].size + 1, dtype=np.int32)]),
                         np.concatenate([r["s_mass"], r["tops_mass"]]))
        # hop 3
        tab3, cap = G3.hop_table(ctx, np.concatenate([v_[0] for v_ in kept2.values()]), 3)
        diag["links"][2], diag["capped"][2], diag["raw"][2] = int(tab3.c.size), cap, ctx.raw_last
        for i1 in need1:
            nodes, pos, mass = kept2[i1]
            p3 = G3.posmap(ctx, tab3.a, nodes, pos)
            keep3 = p3 <= ladder[i1]
            sub = tab3.subset(keep3)
            ps = p3[keep3]
            lw3 = G3.link_weights(ctx, tab3, nodes, mass)[keep3] if pp else None
            for i2 in need2[i1]:
                r = G3.layer(ctx, sub, ps <= ladder[i2], K, ladder[need3[(i1, i2)][-1]], qseed, golds, free, lw=lw3)
                diag["ncand"][2] = max(diag["ncand"][2], r["ncand"])
                for i3 in need3[(i1, i2)]:
                    sz3[0, i1, i2, i3] = G3.size_at(r, ladder[i3])
                gb3[0, i1, i2] = G3.bins_of(r["gpos"], ladder)
                kn["h3"][(i1, i2)] = (r["s_nodes"], r["tops"])
    finally:
        G1.reset(W, lev, seeds, L1, L2_)
    out = {"nfree": int(free.size), "glev": glev, "gfree": gfree, "sz1": sz1, "sz2": sz2, "sz3": sz3, "gb1": gb1, "gb2": gb2, "gb3": gb3, "diag": diag}
    if keep_nodes:
        out["free"], out["kept"] = free, kn
    return out


def read_triples(res, triples):
    """Per triple: (pool size, per gold covered) from a result with the score axis of length 1 (the restricted walker's) read at index 0."""
    out = {}
    for (i1, i2, i3) in triples:
        size = res["nfree"] + int(res["sz1"][0, i1]) + int(res["sz2"][0, i1, i2]) + int(res["sz3"][0, i1, i2, i3])
        cov = np.zeros(res["glev"].size, bool)
        for j in range(cov.size):
            lv = int(res["glev"][j])
            cov[j] = bool(res["gfree"][j]) or lv == 0 or (lv == 1 and res["gb1"][0, j] <= i1) or (lv == 2 and res["gb2"][0, i1, j] <= i2) or (lv == 3 and res["gb3"][0, i1, i2, j] <= i3)
        out[(i1, i2, i3)] = (size, cov)
    return out


def same_as_full(res_r, res_f, k, triples):
    """The restricted result equals the full grid's at index k of the score axis on every triple read (sizes and per-gold coverage)."""
    sub = {key: (res_f[key][k:k + 1] if key in ("sz1", "sz2", "sz3", "gb1", "gb2", "gb3") else res_f[key]) for key in res_f if key != "diag"}
    a, b = read_triples(res_r, triples), read_triples(sub, triples)
    return all(a[t][0] == b[t][0] and np.array_equal(a[t][1], b[t][1]) for t in triples)


def selftest():
    rng = np.random.default_rng(11)
    ladder = (0, 1, 3, 6)
    NL = len(ladder)
    all_tr = [(i, j, k_) for i in range(NL) for j in range(i + 1) for k_ in range(j + 1)]
    trials = checks = 0
    for trial in range(70):
        n = int(rng.integers(14, 46))
        keys = G1.toy_graph(rng, n, int(rng.integers(2 * n, 5 * n)))
        if keys.size < 8:
            continue
        comp = H.build_comp(keys, n)
        G = G1.Graph([comp], n)
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
        full = G3.run_question(ctx, seeds, golds, qseed)
        for k, K in enumerate(SCORES):
            for _ in range(3):
                pick = [all_tr[i] for i in rng.choice(len(all_tr), int(rng.integers(1, 6)), replace=False)] + [(0, 0, 0)]
                rr = run_question_r(ctx, seeds, golds, qseed, K, pick)
                assert same_as_full(rr, full, k, sorted(set(pick))), (trial, K, pick)
                checks += len(set(pick))
                r0 = run_question_r(ctx, seeds, EMPTY, qseed, K, pick)             # G3-5: the sizes do not depend on the probes
                for key_ in ("sz1", "sz2", "sz3"):
                    assert np.array_equal(rr[key_], r0[key_]), key_
        assert bool((lev == G1.DEFAULT_LEV).all()) and not bool(W.vis.any())
        trials += 1
    assert trials > 40 and checks > 1500, (trials, checks)
    assert parse_triples("3-2-1,0-0-0") == [(0, 0, 0), (3, 2, 1)]
    log(f"selftest PASS (zgp3r): the restricted walk equals the full grid on {trials} toy graphs x {len(SCORES)} scores x 3 random sets of triples = {checks} triples (sizes and per-gold coverage); no probe changes a size")
    return 0


# ── real data ────────────────────────────────────────────────────────────────


def cmd_equal(a):
    t0 = time.time()
    D, gt, sl, L, nq = G3.setup(a)
    sel = G3.part_indices(nq, "q")[:a.n]
    rng = np.random.default_rng(5)
    all_tr = [(i, j, k_) for i in range(len(LADDER)) for j in range(i + 1) for k_ in range(j + 1) if i <= 7]
    nchk = nskip = 0
    for _dd, v in [(dd, vv) for dd, vv in CELLS if dd == a.ds]:
        ctx, _reg = G3.real_context(a.ds, v, D, a.host)
        for qi in sel.tolist():
            ld, ls = L["dense"](qi), L["spl"](qi)
            ctx.set_question(ld, ls, G3.base_lists(ld, ls, G3.BASE_B[a.ds]))
            full = G3.run_question(ctx, sl["seeds"][qi], sl["golds"][qi], int(qi))
            if any(full["diag"]["capped"]):
                nskip += 1
                log(f"{a.ds}/{v} q{qi}: a hop tripped the cap in the full grid: not compared")
                continue
            for K in a.scores.split(","):
                k = SCORES.index(K)
                pick = [all_tr[i] for i in rng.choice(len(all_tr), 10, replace=False)] + [(0, 0, 0), (7, 7, 7)]
                t1 = time.time()
                rr = run_question_r(ctx, sl["seeds"][qi], sl["golds"][qi], int(qi), K, pick)
                if any(rr["diag"]["capped"]):
                    stop(f"{a.ds}/{v} q{qi}: the restricted walk tripped the cap where the full grid did not")
                if not same_as_full(rr, full, k, sorted(set(pick))):
                    stop(f"EQUALITY FAILED {a.ds}/{v} q{qi} {K} {sorted(set(pick))}")
                nchk += 1
                log(f"{a.ds}/{v} q{qi} {K}: restricted == full on {len(set(pick))} triples ({time.time() - t1:.1f} s restricted)")
    log(f"equal PASS: {nchk} (question, score) restricted walks equal the full grid, {nskip} questions skipped for a cap trip ({time.time() - t0:.0f} s)")
    return 0


def cmd_extract(a):
    d, t0 = a.ds, time.time()
    if a.part not in ("q", "r"):
        stop("--part is q or r")
    K = a.score
    if K not in SCORES:
        stop(f"--score {K}")
    triples = parse_triples(a.triples)
    D, gt, sl, L, nq = G3.setup(a)
    sel = G3.part_indices(nq, a.part)
    if a.limit:
        sel = sel[:a.limit]
    shard_i, shard_n = (int(x) for x in a.shard.split("/"))
    if not (shard_n >= 1 and 0 <= shard_i < shard_n):
        stop(f"--shard {a.shard}")
    mine = [c0 for c0 in range(0, sel.size, G3.CHUNK_Q) if (c0 // G3.CHUNK_Q) % shard_n == shard_i]
    tag = "" if shard_n == 1 else f"_s{shard_i}of{shard_n}"
    OUT_D = OUT / d
    wd = OUT_D / ("work_limit" if a.limit else "work")
    wd.mkdir(parents=True, exist_ok=True)
    zsha, zrsha = G3.script_sha256(), script_sha256()
    tr_arr = np.asarray(triples, np.int16)
    log(f"{d}: {nq} dev questions with a gold, part {a.part}: {sel.size}; RESTRICTED walk, score {K}, triples {triples} ({time.time() - t0:.0f} s)")
    rec = {"dataset": d, "declared_in": G3.DECLARED, "part": a.part, "status": "read", "nodes": D.n, "nodes_sha256": D.nsha, "shard": [shard_i, shard_n], "nq": nq,
           "gates": {"UL2P": gt, "G3-1": "PASS"}, "versions": {}, "restricted": {"score": K, "triples": [list(t) for t in triples]},
           "constants": {"LADDER": list(LADDER), "SCORES": list(SCORES), "BASE_B": G3.BASE_B[d], "GATHER_CAP": G3.GATHER_CAP, "RKCAP": G3.RKCAP, "VTCAP": G3.VTCAP, "CHUNK_Q": G3.CHUNK_Q,
                         "QUARTER_M": G3.QUARTER_M}}
    for _dd, v in [(dd, vv) for dd, vv in CELLS if dd == d]:
        srk = v == "SRK"
        ctx, _reg = G3.real_context(d, v, D, a.host)
        stats = {"questions": 0, "golds": 0, "free_golds": 0, "levels": [0, 0, 0, 0, 0], "capped_questions": 0}
        chunks = []
        for c0 in mine:
            qs = sel[c0:c0 + G3.CHUNK_Q]
            cp = wd / f"{v}_{a.part}_{c0:06d}.npz"
            if cp.is_file():
                with np.load(cp) as z:
                    same = str(z["zsha"]) == zsha and "zrsha" in z.files and str(z["zrsha"]) == zrsha and str(z["rscore"]) == K and np.array_equal(z["rtriples"], tr_arr)
                    if same:
                        cs = json.loads(str(z["cstats"]))
                if same:
                    chunks.append(cp)
                    for k_, x in cs.items():
                        stats[k_] = [p + q for p, q in zip(stats[k_], x)] if k_ == "levels" else stats[k_] + x
                    continue
                log(f"{rel(cp)} was written by another version or another set of triples: recomputed")
            res = []
            for qi in qs.tolist():
                ld, ls = L["dense"](qi), L["spl"](qi)
                ctx.set_question(ld, ls, G3.base_lists(ld, ls, G3.BASE_B[d]))
                r = run_question_r(ctx, sl["seeds"][qi], sl["golds"][qi], int(qi), K, triples)
                if not srk and not (int(r["sz1"][0, 0]) == 0 and int(r["sz2"][0, 0, 0]) == 0 and int(r["sz3"][0, 0, 0, 0]) == 0):
                    stop(f"G3-4: {d}/{v}: question {qi}: an RK cell keeps nodes at N = 0")
                if len(res) < 2:                                                    # G3-5: the sizes do not depend on the probes
                    r0 = run_question_r(ctx, sl["seeds"][qi], EMPTY, int(qi), K, triples)
                    for k_ in ("sz1", "sz2", "sz3"):
                        if not np.array_equal(r[k_], r0[k_]):
                            stop(f"G3-5: {d}/{v}: question {qi}: {k_} depends on the probes")
                res.append(r)
            pk = G3.pack_chunk(res, qs)
            cs = {"questions": int(len(res)), "golds": int(pk["glev"].size), "free_golds": int(pk["gfree"].sum()),
                  "levels": [int((pk["glev"] == j).sum()) for j in range(5)], "capped_questions": int(pk["capped"].any(axis=1).sum())}
            tmp = cp.with_name(cp.name + ".tmp.npz")
            np.savez_compressed(tmp, nq=np.asarray(nq), zsha=np.array(zsha), zrsha=np.array(zrsha), rscore=np.array(K), rtriples=tr_arr, cstats=np.array(json.dumps(cs)), **pk)
            tmp.replace(cp)
            for k_, x in cs.items():
                stats[k_] = [p + q for p, q in zip(stats[k_], x)] if k_ == "levels" else stats[k_] + x
            chunks.append(cp)
            log(f"{d}/{v}/{a.part}: {min(c0 + G3.CHUNK_Q, sel.size)}/{sel.size} questions ({time.time() - t0:.0f} s, {G1.C.peak_gb()} GB)")
        rec["versions"][v] = {"stats": stats, "chunks": [rel(x) for x in chunks]}
    rec["memory"] = G1.C.peak_gb()
    rec.update(seconds=round(time.time() - t0, 1), utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), zgp3_sha256=zsha, zgp3r_sha256=zrsha, zgp2_sha256=G3.Z.script_sha256(),
               zgp1_sha256=G1.script_sha256())
    fn = OUT_D / f"extract_{a.part}{tag}{'_limit' if a.limit else ''}.json"
    L2.wjson(fn, rec)
    log(f"{d}: {rel(fn)} ({rec['seconds']} s)")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    e = sub.add_parser("equal")
    e.add_argument("--ds", required=True)
    e.add_argument("--n", type=int, default=8)
    e.add_argument("--scores", default="K2,PP")
    e.add_argument("--host", action="store_true")
    e.add_argument("--threads", type=int, default=1)
    e.add_argument("--cap", type=int, default=0)
    x = sub.add_parser("extract")
    x.add_argument("--ds", required=True)
    x.add_argument("--part", required=True)
    x.add_argument("--score", required=True)
    x.add_argument("--triples", required=True)
    x.add_argument("--host", action="store_true")
    x.add_argument("--threads", type=int, default=1)
    x.add_argument("--limit", type=int, default=0)
    x.add_argument("--shard", default="0/1")
    x.add_argument("--cap", type=int, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        return selftest()
    if a.cap:
        G3.GATHER_CAP = a.cap
    return {"equal": cmd_equal, "extract": cmd_extract}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
