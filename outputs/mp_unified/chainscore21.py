"""Design look (untracked; not a result and not filed): S6 part 5 of the transfer plan, option B: passage graphs typed
by what their edges connect, so that one model trains on KBs and passages alike. Part 4 (chainscore20) put a GNN in
the node scorer's place; its decision names the f this file carries (em-gr's MLP f, or en-gr's or ena-gr's GNN).
Part 3's hop-count diagnostic and its read-time counterfactual (cs19_lb_diag, 17:45 on 4 Oct: offsetting the chain
prior by log C_level,length instead of log C_level, at read time only, moved webqsp selectf +6.53 [+3.20, +10.01] R@5
and metaqa select -1.50 [-2.55, -0.49]) give this file's second question: the length offset, trained.

Pseudo-relations. Passage graphs carry no relation ids. Their structural edges are typed by what they connect, as
chainpop17's tt levels type a KB's relations: each distinct structural triple (h, t) of the carve's union graph gets
the descriptor [n(p_h), n(p_t)] / sqrt 2 (p a node's stored projection), and spherical k-means (chainscore19's
skmeans, K_FINE clusters, fitted on a seeded sample of at most FIT_SAMPLE triples, then every triple to its nearest
centroid) makes the clusters the graph's pseudo-relations. Index time, no gold, no names; the read carve's own union
graph, as a KB build's levels are its own carve's.

A pbuild writes typed copies of the passage look's chunks into a temporary root inside the output directory: the
same arrays (x cut after its rrf column, the only one the builds read), plus e_rel, each message's relation slots
(two, -1 empty) in the KB looks' sense: the cluster of its forward triple (u, v) if it carries one, and of its
backward triple (v, u) if it carries one, sorted, a repeated cluster once. The records say typed, with K_FINE
relations. chainscore19's build and chainscore20's companion then run unchanged on that root (the build's levels
are chainpop17's hierarchy over the pseudo-relations: none < dir < tt<k> < exact), and the root is removed. A
sidecar JSON keeps the clustering (sample, seed, cluster sizes, centroid hash) beside the build. The five passage
builds ran on this file's pbuild draft (PBUILD_SHA); pbuild is unchanged since, and training refuses a passage build
whose sidecar names another script or another build.

The model is chainscore20's with f = its carried arm (--f), unchanged. This file changes what it trains on, the
prior's offset (lb) and, on passages, the read's rule.
    lb      p0 = softmax(g - log C_level,length) in training and in every read, C_level,length the row's candidate
            chains at that level with that length; without lb, chainscore19's log C_level
Arms. One seed; training as chainscore19 (Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of
64, one thread, deterministic), the epoch chosen by R@5 on the select carve:
    b, b-lb     metaqa's fit carve and its four transforms (chainscore19's -gr builds), as part 4; select metaqa's
                select carve. b is part 4's carried arm (R0)
    mp, mp-lb   b's five builds and 2wiki's typed fit carve (regimes 1, 0.25, 0.05, 0.01, 0.002 and 0, as metaqa's):
                each example's dataset uniform between metaqa and 2wiki, then chainscore19's draw within it (the build
                uniform among metaqa's five, the row, the regime); select metaqa's select carve
    p, p-lb     2wiki's typed fit carve alone; select 2wiki's typed select carve, by the passage rule
Inputs are standardised by the first build's moments (metaqa's id fit; 2wiki's for p).
Reads. The KB rule (chainscore19's: the non-seed nodes by s_T, ties by rrf, the seeds last): metaqa x1f, webqsp
selectf (305 rows, the twin valid) and webqsp selectf + fit (1,507 rows read as one population), from chainscore19's
builds. The passage rule (the bucket-0 seeds first, in rrf order, then the other nodes by s_T, ties by rrf): 2wiki x1
(in-domain for mp and p), hotpotqa x1 and musique x1f (zero-shot for every arm), from this file's builds, each typed
on its own carve's union graph. Passage references, per row: rrf (seeds rankable), s0+rrf (the bucket-0 seeds first,
then rrf), s0+walk (the bucket-0 seeds first, then the untyped walk: the none level's chains weighted by their
whole-carve rd posterior, as none/rd), the twin (twin0) and the six GNN (gnn0, from the look's stored metrics, checked
row by row against the build's twin0).

Verdicts, fixed at 18:33 on 4 Oct before any arm was trained or read on real data (the builds report only their sizes;
the passage references were run once on the smoke build's 452 2wiki fit rows, a training carve no verdict reads).
Paired row bootstrap (BOOT 1000) on R@5:
    Q1  (primary) webqsp selectf + fit: mp minus b, and mp-lb minus b-lb: ABOVE / AT / BELOW (do typed passages in
        training add to a KB's zero-shot read?)
    Q2  each read: b-lb minus b, mp-lb minus mp and p-lb minus p: HELPS / HURTS / SAME (the trained length offset)
    Q3  webqsp selectf + fit, webqsp selectf and metaqa x1f: p and p-lb minus none/rd (the untyped walk), on all rows
        and on webqsp selectf + fit by seed-gold distance D (1 and 2); and p minus b, p-lb minus b-lb: ABOVE / AT /
        BELOW (passages alone, read on KBs)
    Q4  each passage read: every arm minus s0+rrf and minus s0+walk: ABOVE / AT / BELOW, with its share of the twin's
        lead over s0+rrf (HIGH if the interval lies at or above 0.5, LOW if below 0.25, else MID); mp minus b and p
        minus b (and their lb pairs): ABOVE / AT / BELOW
    Q5  metaqa x1f: mp minus b, and mp-lb minus b-lb: ABOVE / AT / BELOW (what mixing in passages costs in-domain)
    R0  b's per-row reads equal chainscore20's carried arm's (|diff| < 1e-6) on metaqa, webqsp and webqsp_sf:
        REPRODUCES / DIFFERS
Decision. (1) lb carries into later parts if b-lb minus b is HELPS on webqsp selectf + fit and not HURTS on metaqa
x1f; otherwise the offset stays log C_level. (2) Q1 is read on the pair with the carried offset. If it is ABOVE, part
6 trains one model on metaqa and the three passage fit carves and reads each passage graph held out in turn (LOGO). If
it is not ABOVE and p (with the carried offset) is ABOVE none/rd on webqsp selectf + fit (Q3), part 6 takes the
passage-only route over the three passage graphs. Otherwise endpoint-text pseudo-relations carry nothing a KB's
zero-shot read can use, and the result goes to Swastik before part 6. (3) If R0 DIFFERS, nothing is read until its
cause is found. Train-split rows throughout: a look, not a result.

    python outputs/mp_unified/chainscore21.py pbuild --ds 2wiki --carve fit --hops 2 --ks 4,16,64 \
        --regimes 1,0.25,0.05,0.01,0.002,0 --out outputs/mp_unified/lean/cs21-2w-fit.npz \
        --edges-out outputs/mp_unified/lean/cs21e-2w-fit.npz
    python outputs/mp_unified/chainscore21.py train --arm mp-lb --f em-gr \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz \
        (and sp4, mg3, rf) --select outputs/mp_unified/lean/cs19-mq-select.npz \
        --pfit outputs/mp_unified/lean/cs21-2w-fit.npz \
        --read metaqa=outputs/mp_unified/lean/cs19-mq-x1f.npz --read webqsp=outputs/mp_unified/lean/cs19-wq.npz \
        --read webqsp_sf=outputs/mp_unified/lean/cs19-wq-sf.npz --pread 2wiki=outputs/mp_unified/lean/cs21-2w-x1.npz \
        --pread hotpotqa=outputs/mp_unified/lean/cs21-hp-x1.npz \
        --pread musique=outputs/mp_unified/lean/cs21-mu-x1f.npz \
        (a GNN f: --edges NAME=COMPANION for every build it loads) --out outputs/mp_unified/lean/cs21-mp-lb.json \
        --rows-out outputs/mp_unified/lean/cs21-mp-lb.rows.npz --state-out outputs/mp_unified/lean/cs21-mp-lb.pt
    python outputs/mp_unified/chainscore21.py grade --arm b=outputs/mp_unified/lean/cs21-b.json (each arm) \
        --cs20 em-gr=outputs/mp_unified/lean/cs20-em-gr.json --read metaqa=... (the three) --pread 2wiki=... \
        (the three) --out outputs/mp_unified/lean/cs21.json
    python outputs/mp_unified/chainscore21.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count: the k-means and the builds' levels depend on it

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import reltype11 as RT  # noqa: E402

LOOK = RT.LOOK
SEED = C19.SEED
K_FINE = 256
FIT_SAMPLE = 100_000
K_SLOTS = 2
SHIFT = RT.SHIFT
PKEYS = ("q_pool_size", "q_edges", "q_emb", "q_seed_local", "q_seed_bucket", "q_gold_total", "q_metrics", "pool",
         "proj", "is_gold", "x", "e_u", "e_v", "e_fam", "e_fwd", "e_bwd")
PBUILD_SHA = "d834786b2a0cd7d0746189dfedd93a355d16d6d790c6205424a3b5275b2df8fb"   # the draft the five builds ran on
BOOT, BATCH = C19.BOOT, C19.BATCH
ARMS = ("b", "b-lb", "mp", "mp-lb", "p", "p-lb")
FS = ("em-gr", "en-gr", "ena-gr")
KB_READS = ("metaqa", "webqsp", "webqsp_sf")
P_READS = ("2wiki", "hotpotqa", "musique")
HOPC = [CS.FI.index(f"hop{k}") for k in (1, 2, 3)]
log, unit, sha = RT.log, RT.unit, CS.sha


# ── passage looks: pseudo-relations from endpoint text ───────────────────────────────────────────────────────────


def passage_paths(ds, cv, root, limit=None):
    """A passage look's listed chunks (the first `limit`) and its records."""
    d = Path(root) / ds / cv
    recs = sorted(d.glob("record*.json"))
    if not recs:
        raise SystemExit(f"{d}: no look record")
    listed, out = set(), []
    for r_ in recs:
        rr = json.loads(r_.read_text(encoding="utf-8"))
        if not rr.get("full"):
            raise SystemExit(f"{r_}: not a --full look (it has no edges)")
        if (rr.get("relations") or {}).get("typed"):
            raise SystemExit(f"{r_}: a typed (KB) look; chainscore19 reads it as it is")
        if rr.get("proj") != {"dim": RT.PROJ_DIM, "seed": RT.PROJ_SEED}:
            raise SystemExit(f"{r_}: projection {rr.get('proj')}")
        listed |= set(rr["chunks"])
        out.append((r_.name, rr))
    paths = [p for p in sorted((d / "chunks").glob("c*.npz"))
             if RT.CHUNK_RE.fullmatch(p.name) and int(RT.CHUNK_RE.fullmatch(p.name).group(1)) in listed]
    if not paths:
        raise SystemExit(f"{d}: no listed chunk is here")
    return (paths[:limit] if limit else paths), out


def struct_triples(c, ea, eb):
    """A row's structural messages: their index within the row's messages, endpoints, direction flags."""
    s = np.flatnonzero(c["e_fam"][ea:eb] == 0)
    u = c["e_u"][ea:eb][s].astype(np.int64)
    v = c["e_v"][ea:eb][s].astype(np.int64)
    fw = c["e_fwd"][ea:eb][s] == 1
    bw = c["e_bwd"][ea:eb][s] == 1
    return s, u, v, fw, bw


def gkey(pool, h, t):
    return (pool[h] << SHIFT) | pool[t]


def union_triples(paths):
    """Distinct global structural triples (h, t) over the chunks, and the node table (global id, projection)."""
    keys, nk, npj = [], [], []
    gap = 0.0
    for p in paths:
        with np.load(p) as zf:
            c = {k: zf[k] for k in ("q_pool_size", "q_edges", "pool", "proj", "e_u", "e_v", "e_fam", "e_fwd",
                                    "e_bwd")}
        n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
        e_off = np.r_[0, np.cumsum(c["q_edges"])]
        pool_all = c["pool"].astype(np.int64)
        if pool_all.size and int(pool_all.max()) >= (1 << SHIFT - 1):
            raise SystemExit(f"{p}: a node id does not fit {SHIFT - 1} bits")
        k_n, p_n, g = RT.dedupe(pool_all, c["proj"], gap=True)
        gap = max(gap, g)
        nk.append(k_n)
        npj.append(p_n)
        for i in range(c["q_pool_size"].size):
            a, b = int(n_off[i]), int(n_off[i + 1])
            ea, eb = int(e_off[i]), int(e_off[i + 1])
            pool = pool_all[a:b]
            _s, u, v, fw, bw = struct_triples(c, ea, eb)
            keys.append(np.unique(np.r_[gkey(pool, u[fw], v[fw]), gkey(pool, v[bw], u[bw])]))
    keys = np.unique(np.concatenate(keys)) if keys else np.zeros(0, np.int64)
    nodes, proj, g = RT.dedupe(np.concatenate(nk), np.concatenate(npj), gap=True)
    return keys, nodes, proj, max(gap, g)


def descriptors(keys, nodes, Pn, lo=0, hi=None):
    hi = keys.size if hi is None else hi
    k = keys[lo:hi]
    H, T = k >> SHIFT, k & ((1 << SHIFT) - 1)
    Hi, Ti = np.searchsorted(nodes, H), np.searchsorted(nodes, T)
    if not (np.array_equal(nodes[Hi], H) and np.array_equal(nodes[Ti], T)):
        raise SystemExit("a triple's endpoint is not in the node table")
    return (np.c_[Pn[Hi], Pn[Ti]] / math.sqrt(2)).astype(np.float32)


def fit_types(keys, nodes, proj, K, seed=SEED + 21, sample=FIT_SAMPLE):
    """Spherical k-means centroids on a seeded sample of the triples' descriptors; every triple's cluster."""
    Pn = unit(proj.astype(np.float32))
    n = keys.size
    K = int(min(K, n))
    idx = np.sort(np.random.default_rng(seed).choice(n, size=min(sample, n), replace=False))
    Xs = descriptors(keys[idx], nodes, Pn)
    lab_s = C19.skmeans(Xs, K, seed)
    C = np.zeros((K, Xs.shape[1]), np.float32)
    np.add.at(C, lab_s, Xs)
    empty = ~np.any(C != 0, axis=1)
    C = unit(C)
    lab = np.empty(n, np.int64)
    for a in range(0, n, 1 << 18):
        S = descriptors(keys, nodes, Pn, a, min(n, a + (1 << 18))) @ C.T
        S[:, empty] = -np.inf
        lab[a:a + S.shape[0]] = np.argmax(S, 1)
    return lab, C, {"K": K, "triples": int(n), "sample": int(idx.size), "seed": int(seed),
                    "empty": int(empty.sum()), "sizes": np.bincount(lab, minlength=K).tolist(),
                    "centroids_sha256": hashlib.sha256(np.ascontiguousarray(C).tobytes()).hexdigest()}


def message_slots(c, ea, eb, pool, keys, lab):
    """The row's e_rel: per message K_SLOTS relation slots, the clusters of its forward and backward triples."""
    E = np.full((eb - ea, K_SLOTS), -1, np.int16)
    s, u, v, fw, bw = struct_triples(c, ea, eb)
    if not s.size:
        return E
    cf = np.full(s.size, -1, np.int64)
    cb = np.full(s.size, -1, np.int64)
    for msk, out, kk in ((fw, cf, gkey(pool, u, v)), (bw, cb, gkey(pool, v, u))):
        j = np.searchsorted(keys, kk[msk])
        if not np.array_equal(keys[np.minimum(j, keys.size - 1)], kk[msk]):
            raise SystemExit("a row's triple is not in the union graph")
        out[msk] = lab[j]
    lo = np.where((cf >= 0) & (cb >= 0), np.minimum(cf, cb), np.maximum(cf, cb))
    hi = np.where((cf >= 0) & (cb >= 0) & (cf != cb), np.maximum(cf, cb), -1)
    E[s, 0] = lo
    E[s, 1] = hi
    return E


def typed_root(ds, carve, src_root, dst_root, K, limit=None):
    """Typed copies of a passage look's carves (joined by +, typed as one union graph) under dst_root/ds/carve."""
    carves = [x for x in str(carve).split("+") if x]
    allp, per = [], []
    for cv in carves:
        ps, recs = passage_paths(ds, cv, src_root, limit)
        allp += ps
        per.append((cv, ps, recs))
    t0 = time.time()
    keys, nodes, proj, gap = union_triples(allp)
    lab, C, info = fit_types(keys, nodes, proj, K)
    info.update({"carves": carves, "chunks": len(allp), "nodes": int(nodes.size), "proj_gap": gap,
                 "seconds": round(time.time() - t0, 1)})
    log(f"{ds}={carve}: {keys.size} distinct structural triples over {nodes.size} nodes, {info['K']} clusters "
        f"(sizes {min(info['sizes'])} to {max(info['sizes'])}, {info['empty']} empty) in {info['seconds']}s")
    for cv, ps, recs in per:
        d = Path(dst_root) / ds / cv
        (d / "chunks").mkdir(parents=True, exist_ok=True)
        rrf_col = None
        for name, rr in recs:
            cols = list(rr["columns"])
            rc = cols.index("rrf")
            if rrf_col is not None and rc != rrf_col:
                raise SystemExit(f"{ds}/{cv}: records disagree on the rrf column")
            rrf_col = rc
            r2 = dict(rr)
            r2["columns"] = cols[:rc + 1]
            r2["relations"] = {"typed": True, "offset": 0, "n_relations": info["K"], "k_rel": K_SLOTS,
                               "passage_types": {k: info[k] for k in ("K", "sample", "seed", "centroids_sha256")}}
            (d / name).write_text(json.dumps(r2), encoding="utf-8")
        for p in ps:
            with np.load(p) as zf:
                c = {k: zf[k] for k in PKEYS}
            c["x"] = np.ascontiguousarray(c["x"][:, :rrf_col + 1])
            n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
            e_off = np.r_[0, np.cumsum(c["q_edges"])]
            pool_all = c["pool"].astype(np.int64)
            c["e_rel"] = np.concatenate(
                [message_slots(c, int(e_off[i]), int(e_off[i + 1]), pool_all[int(n_off[i]):int(n_off[i + 1])], keys,
                               lab) for i in range(c["q_pool_size"].size)]
                + [np.zeros((0, K_SLOTS), np.int16)])
            np.savez(d / "chunks" / p.name, **c)
    return info, keys, lab, C


def pbuild_cmd(a):
    t0 = time.time()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="cs21tmp_", dir=out.parent))
    try:
        info, _k, _l, _c = typed_root(a.ds, a.carve, Path(a.src_root) if a.src_root else LOOK, tmp / "look",
                                      a.k_fine, a.limit)
        (tmp / "norel").mkdir()
        ba = SimpleNamespace(ds=a.ds, carve=a.carve, hops=a.hops, ks=a.ks, regimes=a.regimes, transform="id",
                             rows=a.rows, max_chains=a.max_chains, limit=a.limit, out=a.out)
        C19.build19(ba, root=tmp / "look", rel_dir=tmp / "norel")
        if a.edges_out:
            C20.edges_cmd(SimpleNamespace(base=a.out, out=a.edges_out, no_node_check=False), root=tmp / "look")
        side = {"look": "chainscore21-passage", "script_sha256": sha(__file__), "chainscore19_sha256": sha(C19.__file__),
                "chainscore20_sha256": sha(C20.__file__), "args": dict(vars(a)), "types": info,
                "build_sha256": sha(a.out), "edges_sha256": sha(a.edges_out) if a.edges_out else None,
                "seconds": round(time.time() - t0, 1)}
        sp_ = out.with_name(out.stem + ".passage.json")
        tmpj = sp_.with_name(sp_.name + ".tmp")
        tmpj.write_text(json.dumps(side, indent=1), encoding="utf-8")
        os.replace(tmpj, sp_)
        log(f"pbuild done in {time.time() - t0:.0f}s -> {out}" + (f", {a.edges_out}" if a.edges_out else ""))
        return side
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def check_passage(p, build_sha, edges_sha=None):
    """A passage build's sidecar: written by this file's pbuild (its draft or this file) for this build and, for a
    GNN f, this companion."""
    sp_ = Path(p).with_name(Path(p).stem + ".passage.json")
    if not sp_.exists():
        raise SystemExit(f"{p}: no sidecar {sp_.name}; not a chainscore21 passage build")
    side = json.loads(sp_.read_text(encoding="utf-8"))
    if side.get("look") != "chainscore21-passage" or side.get("script_sha256") not in (PBUILD_SHA, sha(__file__)):
        raise SystemExit(f"{sp_}: written by script {str(side.get('script_sha256'))[:12]}, not this file's pbuild")
    if side.get("build_sha256") != build_sha:
        raise SystemExit(f"{sp_}: its build hash is not {p}'s")
    if edges_sha is not None and side.get("edges_sha256") != edges_sha:
        raise SystemExit(f"{sp_}: its companion hash is not the given companion's")
    return side


# ── part 5: one model on KBs and typed passages; the trained length offset; the passage rule ─────────────────────


def arm_spec(arm, f):
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm}")
    if f not in FS:
        raise SystemExit(f"--f is one of {FS} (chainscore20's carried arm), not {f}")
    return dict(C20.arm_spec(f), data=arm.split("-")[0], lb=arm.endswith("-lb"), f_arm=f)


def lb_offsets(d):
    """Per chain, log C_level,length: the row's candidate chains at its level with its length. Refuses a build whose
    log C_level, counted the same way, is not its stored one."""
    ro = d["rowoff"].astype(np.int64)
    N, P = ro.size - 1, int(ro[-1])
    L = int(d["level_names"].size)
    hop = d["FI"][:, HOPC]
    if P and not np.array_equal(hop.sum(1), np.ones(P, hop.dtype)):
        raise SystemExit("a chain without exactly one length in 1..3")
    row = np.repeat(np.arange(N, dtype=np.int64), np.diff(ro))
    kl = row * L + d["p_lev"].astype(np.int64)
    kll = kl * 3 + np.argmax(hop, 1)
    cl = np.bincount(kl, minlength=N * L)[kl]
    cll = np.bincount(kll, minlength=N * L * 3)[kll]
    if P and float(np.abs(np.log(cl) - d["FI"][:, CS.LOGC].astype(np.float64)).max()) > 1e-5:
        raise SystemExit("log C_level counted from the build is not its stored one")
    return np.log(cll).astype(np.float32)


def draw21(rng, builds, groups, count):
    """count examples (build, row, regime): the group (a dataset's builds) uniform among groups, then chainscore19's
    draw within it. One group is chainscore19's draw itself."""
    if len(groups) == 1:
        return C19.draw(rng, builds, groups[0], count)
    t = rng.integers(len(groups), size=count)
    out = [C19.draw(rng, builds, g, int((t == k).sum())) for k, g in enumerate(groups) if (t == k).any()]
    items = np.concatenate(out)
    return items[rng.permutation(len(items))]


def make_batch21(items, builds, comps, LB, st, pairs=True, train=True, gnn=False, att=False):
    """chainscore20's batch; with LB (per build, lb_offsets) the prior's offset is log C_level,length."""
    bt = C20.make_batch20(items, builds, comps, st, pairs=pairs, train=train, gnn=gnn, att=att)
    if LB is None or not pairs:
        return bt
    import torch
    lc = bt["lc"].numpy().copy()
    for j, (bi, i, _ri) in enumerate(np.asarray(items, np.int64).reshape(-1, 3)):
        d = builds[bi]
        p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
        lc[j, :p1 - p0] = LB[bi][p0:p1]
    bt["lc"] = torch.from_numpy(lc)
    return bt


def rank_row_p(d, i, s):
    """The passage rule: the row's bucket-0 seeds first, in rrf order, then its other nodes by s, ties by rrf."""
    a, n = int(d["xn_off"][i]), int(d["n"][i])
    ns = np.flatnonzero(~d["seed0"][a:a + n])
    rank = d["rank"][d["rank_off"][i]:d["rank_off"][i + 1]].astype(np.int64)
    srr = d["seeds_rr"][d["seeds_off"][i]:d["seeds_off"][i + 1]].astype(np.int64)
    top = np.r_[srr, ns[np.lexsort((rank[ns], -s[ns]))]][:5]
    gold = np.zeros(n, bool)
    gold[d["gold"][d["gold_off"][i]:d["gold_off"][i + 1]]] = True
    return CP.metrics(top, gold, int(d["gt"][i]))


def read21(model, d, c, lb, st, rule, ri=0, batch=BATCH):
    """chainscore20's read20 with lb's offset (lb: the build's lb_offsets, or None) and the rule's ranking."""
    import torch
    rank = {"kb": C19.rank_row, "passage": rank_row_p}[rule]
    N = d["n"].size
    out = np.zeros((N, 3))
    L = d["level_names"].size
    ml = np.zeros(L)
    plev = d["p_lev"].astype(np.int64)
    gnn = model.gnn is not None
    with torch.no_grad():
        for b0 in range(0, N, batch):
            rows = np.arange(b0, min(N, b0 + batch))
            items = np.c_[np.zeros(rows.size, np.int64), rows, np.full(rows.size, ri)]
            bt = make_batch21(items, [d], [c], None if lb is None else [lb], st, pairs=model.g is not None,
                              train=False, gnn=gnn, att=gnn and model.att)
            s, lp = C20.forward20(model, bt)
            s = s.double().numpy()
            Pm = None if lp is None else np.exp(lp.double().numpy())
            for j, i in enumerate(rows):
                out[i] = rank(d, i, s[j, :int(d["n"][i])])
                if Pm is not None:
                    p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
                    if p1 > p0:
                        ml += np.bincount(plev[p0:p1], weights=Pm[j, :p1 - p0], minlength=L)
    return out, ml / max(N, 1)


def train21(arm, f, builds, comps, LB, groups, Sd, Sc, Slb, srule, epochs, per_epoch, batch, seed, threads=1):
    """chainscore20's train20 with the grouped draw, lb's offset (LB per build, or None) and the select read's rule."""
    import torch
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    spec = arm_spec(arm, f)
    st = C19.input_stats(builds[0])
    model = C20.make_model(f, seed)
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    hist, best, best_state = [], None, None
    for ep in range(epochs):
        t1 = time.time()
        model.train()
        items = draw21(rng, builds, groups, per_epoch)
        tot, tn, tc, nb = 0.0, 0.0, 0.0, 0
        for b0 in range(0, len(items), batch):
            bt = make_batch21(items[b0:b0 + batch], builds, comps, LB, st, pairs=spec["g"], gnn=spec["gnn"],
                              att=spec["att"])
            s, lp = C20.forward20(model, bt)
            loss, ln, lc = C19.objective(s, lp, bt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            tn += ln
            tc += lc
            nb += 1
        model.eval()
        x, _ = read21(model, Sd, Sc, Slb, st, srule)
        sr5 = float(x[:, 0].mean())
        hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                     "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                     "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(), **C20.coupling20(model),
                     "seconds": round(time.time() - t1, 1)})
        log(f"    {arm} ({f}) ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
            f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {C20.coupling20(model)} {time.time() - t1:.0f}s")
        if best is None or sr5 > best[1]:
            best = (ep, sr5)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "coupling": C20.coupling20(model), "spec": spec}


def train_cmd(a):
    t0 = time.time()
    spec = arm_spec(a.arm, a.f)
    data = spec["data"]
    res = {"look": "chainscore21", "arm": a.arm, "f": a.f, "args": dict(vars(a)), "script_sha256": sha(__file__),
           "chainscore20_sha256": sha(C20.__file__), "chainscore19_sha256": sha(C19.__file__),
           "chainscore18_sha256": sha(CS.__file__), "chainpop17_sha256": sha(CP.__file__), "inputs": {},
           "inputs_sha256": {}, "passage_types": {}, "reads": {}, "rows_out": a.rows_out}
    need = {"b": ("fit", "aug", "select"), "mp": ("fit", "aug", "select", "pfit"), "p": ("pfit", "pselect")}[data]
    given = {"fit": a.fit, "aug": a.aug or None, "select": a.select, "pfit": a.pfit, "pselect": a.pselect}
    for k, v in given.items():
        if (k in need) != (v is not None):
            raise SystemExit(f"arm {a.arm} takes {', '.join('--' + x for x in need)} and no other training input")
    aug = dict(x.split("=", 1) for x in a.aug)
    if "aug" in need and sorted(aug) != sorted(C19.TRANSFORMS[1:]):
        raise SystemExit(f"arm {a.arm} takes --aug for each of {C19.TRANSFORMS[1:]}")
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_READS):
        raise SystemExit(f"every arm reads {KB_READS} (--read) and {P_READS} (--pread)")
    train_in = ([("fit", a.fit, False)] + [(t, aug[t], False) for t in C19.TRANSFORMS[1:]] if data != "p" else []) \
        + ([("pfit", a.pfit, True)] if data != "b" else [])
    sel = ("pselect", a.pselect, True) if data == "p" else ("select", a.select, False)
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    loads = train_in + [sel] + reads
    path = {n: p for n, p, _ in loads}
    edges = dict(x.split("=", 1) for x in a.edges)
    if spec["gnn"] and sorted(edges) != sorted(path):
        raise SystemExit(f"a GNN f takes --edges for each of {sorted(path)}; given {sorted(edges)}")
    if not spec["gnn"] and edges:
        raise SystemExit("the MLP f takes no --edges")
    for n, p, is_p in loads:
        res["inputs_sha256"][p] = sha(p)
        if spec["gnn"]:
            res["inputs_sha256"][edges[n]] = sha(edges[n])
        if is_p:
            side = check_passage(p, res["inputs_sha256"][p], res["inputs_sha256"][edges[n]] if spec["gnn"] else None)
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}

    def comp(n, d):
        if not spec["gnn"]:
            return None
        c = C20.load_comp(edges[n], d, res["inputs_sha256"][path[n]])
        if not spec["att"]:     # the untyped GNN reads the messages only
            for k in ("slots", "p_slot", "i_e", "i_s"):
                c.pop(k)
        return c

    builds, comps, LB = [], [], []
    for n, p, _is_p in train_in:
        d = C19.prep(CS.load(p))
        tf = d["meta"]["transform"]["name"]
        if tf != ("id" if n in ("fit", "pfit") else n):
            raise SystemExit(f"{p} holds transform {tf}, not {'id' if n in ('fit', 'pfit') else n}")
        res["inputs"][n] = d["meta"]
        builds.append(d)
        comps.append(comp(n, d))
        LB.append(lb_offsets(d) if spec["lb"] else None)
    groups = {"b": [list(range(5))], "mp": [list(range(5)), [5]], "p": [[0]]}[data]
    Sd = CS.load(sel[1])
    Sc = comp(sel[0], Sd)
    Slb = lb_offsets(Sd) if spec["lb"] else None
    res["inputs"][sel[0]] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p, _q), b in zip(train_in, builds)}
    res["groups"] = groups
    if spec["gnn"]:
        res["edges"] = {n: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                        for (n, _p, _q), c in zip(train_in, comps)}
    model, info = train21(a.arm, a.f, builds, comps, LB if spec["lb"] else None, groups, Sd, Sc, Slb,
                          "passage" if data == "p" else "kb", a.epochs, a.per_epoch, a.batch, SEED)
    del builds, comps, LB, Sd, Sc, Slb
    res.update({"history": info["history"], "best_epoch": info["best_epoch"], "best_select_r5": info["best_select_r5"],
                "coupling": info["coupling"], "spec": info["spec"]})
    rows_out = {}
    for n, p, is_p in reads:
        t1 = time.time()
        d = CS.load(p)
        c = comp(n, d)
        rule = "passage" if is_p else "kb"
        xr, ml = read21(model, d, c, lb_offsets(d) if spec["lb"] else None, info["stats"], rule, batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml if model.g is not None else None, d["level_names"]),
                               rule=rule)
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} ({a.f}) on {n} ({rule} rule): {res['reads'][n]['mean']} "
            f"mass {res['reads'][n]['mass_by_level']} ({time.time() - t1:.0f}s)")
        del d, c
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out:
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": model.state_dict(), "stats": info["stats"], "spec": info["spec"], "arm": a.arm,
                    "f": a.f}, sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def passage_refs(d):
    """Per row (N x 3) of a passage build: rrf (seeds rankable), s0+rrf and s0+walk (the bucket-0 seeds first, then
    rrf, or the untyped walk's mass: the none level's chains weighted by their whole-carve rd posterior, ties by rrf,
    the rest by rrf), and the twin."""
    ro = d["rowoff"].astype(np.int64)
    N = ro.size - 1
    allow = d["p_lev"].astype(np.int64) == list(d["level_names"]).index("none")
    w = d["post1"][:, CP.FITS.index("rd")]
    no, mass, node = d["node_off"], d["mass"], d["node"]
    s0r, s0w = np.zeros((N, 3)), np.zeros((N, 3))
    for i in range(N):
        n = int(d["n"][i])
        gold = np.zeros(n, bool)
        gold[d["gold"][d["gold_off"][i]:d["gold_off"][i + 1]]] = True
        gt = int(d["gt"][i])
        rank = d["rank"][d["rank_off"][i]:d["rank_off"][i + 1]].astype(np.int64)
        ons = d["order_ns"][d["order_off"][i]:d["order_off"][i + 1]].astype(np.int64)
        srr = d["seeds_rr"][d["seeds_off"][i]:d["seeds_off"][i + 1]].astype(np.int64)
        s0r[i] = CP.metrics(np.r_[srr, ons][:5], gold, gt)
        top = ons[:5]
        idx = np.flatnonzero(allow[ro[i]:ro[i + 1]]) + ro[i]
        if idx.size:
            wi = np.nan_to_num(w[idx].astype(np.float64))
            if wi.sum() > 0:
                wi = wi / wi.sum()
                lens = no[idx + 1] - no[idx]
                sel = np.concatenate([np.arange(no[k], no[k + 1]) for k in idx])
                s = np.bincount(node[sel].astype(np.int64), weights=np.repeat(wi, lens) * mass[sel], minlength=n)
                a0 = int(d["xn_off"][i])
                sup = np.flatnonzero((s > 0) & ~d["seed0"][a0:a0 + n])      # the seeds are already first
                top = CP.top5(sup, s[sup], rank, ons, np.zeros(0, np.int64))
        s0w[i] = CP.metrics(np.r_[srr, top][:5], gold, gt)
    return {"rrf": np.asarray(d["rrf"], np.float64), "s0+rrf": s0r, "s0+walk": s0w,
            "twin0": np.asarray(d["twin0"], np.float64)}


def look_fn(d, fn, root=LOOK):
    """A passage build's rows' stored metrics (R@5, FC@5, hit@1) of the look's function fn, after checking that the
    look's twin0 is the build's, row by row."""
    m = d["meta"]
    ds, limit = m["args"]["ds"], m["args"].get("limit")
    per = []
    for cv in m["carves"]:
        paths, recs = passage_paths(ds, cv, root, limit)
        _rc, m_idx = CP.read_record(ds, cv, root)
        fns = {tuple(rr["functions"]) for _n, rr in recs}
        if len(fns) != 1 or fn not in next(iter(fns)):
            raise SystemExit(f"{ds}/{cv}: the look's records do not all store {fn}")
        fi = list(next(iter(fns))).index(fn)
        tw, fx = [], []
        for p in paths:
            with np.load(p) as zf:
                qm = zf["q_metrics"]
            tw.append(qm[:, 0][:, m_idx])
            fx.append(qm[:, fi][:, m_idx])
        per.append((np.concatenate(tw).astype(np.float64), np.concatenate(fx).astype(np.float64)))
    ci, ri = d["carve_ix"].astype(np.int64), d["row_ix"].astype(np.int64)
    if ci.size and (int(ci.max()) >= len(per) or any(int(ri[ci == k].max(initial=-1)) >= per[k][0].shape[0]
                                                      for k in range(len(per)))):
        raise SystemExit(f"{ds}: the build has rows the look does not")
    TW = np.stack([per[k][0][r] for k, r in zip(ci, ri)]) if ci.size else np.zeros((0, 3))
    FX = np.stack([per[k][1][r] for k, r in zip(ci, ri)]) if ci.size else np.zeros((0, 3))
    if not np.array_equal(TW, np.asarray(d["twin0"], np.float64)):
        raise SystemExit(f"{ds}: the look's twin0 is not the build's, row by row")
    return FX


def grade_cmd(a):
    t0 = time.time()
    arms, rows = {}, {}
    for x in a.arm:
        nm_, p_ = x.split("=", 1)
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore21" or js.get("arm") != nm_:
            raise SystemExit(f"{p_} is not chainscore21's arm {nm_}")
        with np.load(js["rows_out"]) as z:
            rows[nm_] = {k: z[k] for k in z.files}
        arms[nm_] = js
    if sorted(arms) != sorted(ARMS):
        raise SystemExit(f"the grade takes every arm, {ARMS}; given {sorted(arms)}")
    fs = {js["f"] for js in arms.values()}
    if len(fs) != 1:
        raise SystemExit(f"the arms carry different f: {sorted(fs)}")
    f = next(iter(fs))
    if a.cs20:
        nm_, p_ = a.cs20.split("=", 1)
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore20" or js.get("arm") != nm_ or nm_ != f:
            raise SystemExit(f"{p_}: R0 reads b against chainscore20's {f}, not {js.get('arm')}")
        with np.load(js["rows_out"]) as z:
            rows["cs20"] = {k: z[k] for k in z.files}
    refs, Ds, rule = {}, {}, {}
    for flag, xs in (("kb", a.read), ("passage", a.pread)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            h = sha(p_)
            for k, js in arms.items():
                if js["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"arm {k} read another {nm_} than {p_}")
            d = CS.load(p_)
            if flag == "kb":
                refs[nm_] = CS.reference_arms(d)
            else:
                refs[nm_] = passage_refs(d)
                refs[nm_]["gnn0"] = look_fn(d, "gnn0", Path(a.look_root) if a.look_root else LOOK)
            Ds[nm_] = d["D"]
            rule[nm_] = flag
            del d
    if sorted(k for k in rule if rule[k] == "kb") != sorted(KB_READS) or \
            sorted(k for k in rule if rule[k] == "passage") != sorted(P_READS):
        raise SystemExit(f"the grade reads {KB_READS} (--read) and {P_READS} (--pread)")
    res = {"look": "chainscore21", "f": f, "args": dict(vars(a)), "script_sha256": sha(__file__),
           "arms": {k: {"best_epoch": v["best_epoch"], "best_select_r5": v["best_select_r5"],
                        "coupling": v.get("coupling"), "reads": v["reads"], "train_rows": v.get("train_rows")}
                    for k, v in arms.items()},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()} for nm_, r in refs.items()},
           "rows": {nm_: int(r["twin0"].shape[0]) for nm_, r in refs.items()}, "verdicts": {}}
    hs, abv = ("HELPS", "HURTS", "SAME"), ("ABOVE", "BELOW", "AT")
    V = {k: {} for k in ("Q1", "Q2", "Q3", "Q4", "Q5", "R0")}

    def r5(arm, nm_):
        x = rows.get(arm, {}).get(nm_)
        return None if x is None else x[:, 0].astype(np.float64)

    def diff(x_, y_, nm_, idx, labels_, m=None):
        X_, Y_ = r5(x_, nm_), (r5(y_, nm_) if isinstance(y_, str) else y_)
        if X_ is None or Y_ is None:
            return None
        dd = X_ - Y_
        if m is not None:
            dd = dd[m]
            idx = np.random.default_rng(SEED).integers(0, dd.size, size=(BOOT, dd.size))
        ci = CP.boot_mean(dd, idx)
        out = {"diff": ci, "verdict": C19.ci_label(ci, labels_)}
        if m is not None:
            out["rows"] = int(dd.size)
        return out

    for nm_ in refs:
        N = refs[nm_]["twin0"].shape[0]
        for k, r in rows.items():
            if nm_ in r and r[nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {r[nm_].shape[0]} rows, the build {N}")
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        for x_ in ("b", "mp", "p"):
            V["Q2"].setdefault(nm_, {})[f"{x_}-lb - {x_}"] = diff(f"{x_}-lb", x_, nm_, idx, hs)
        if nm_ == "webqsp_sf":
            for x_, y_ in (("mp", "b"), ("mp-lb", "b-lb")):
                V["Q1"][f"{x_} - {y_}"] = diff(x_, y_, nm_, idx, abv)
        if nm_ == "metaqa":
            for x_, y_ in (("mp", "b"), ("mp-lb", "b-lb")):
                V["Q5"][f"{x_} - {y_}"] = diff(x_, y_, nm_, idx, abv)
        if rule[nm_] == "kb":
            walk = refs[nm_]["none/rd"][:, 0]
            for x_, y_ in (("p", "b"), ("p-lb", "b-lb")):
                e = {"minus_none_rd": diff(x_, walk, nm_, idx, abv), f"minus_{y_}": diff(x_, y_, nm_, idx, abv)}
                if nm_ == "webqsp_sf":
                    for dd in (1, 2):
                        m = Ds[nm_] == dd
                        if m.any():
                            e[f"minus_none_rd_D{dd}"] = diff(x_, walk, nm_, idx, abv, m)
                V["Q3"].setdefault(nm_, {})[x_] = e
            if "cs20" in rows and nm_ in rows["cs20"] and nm_ in rows["b"]:
                dif = float(np.abs(rows["b"][nm_].astype(np.float64) - rows["cs20"][nm_].astype(np.float64)).max())
                V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
        else:
            s0r, s0w, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["s0+walk"][:, 0], refs[nm_]["twin0"][:, 0]
            e = {}
            for x_ in ARMS:
                X_ = r5(x_, nm_)
                if X_ is None:
                    continue
                sh = CP.share(X_, s0r, tw, idx)
                e[x_] = {"minus_s0+rrf": diff(x_, s0r, nm_, idx, abv), "minus_s0+walk": diff(x_, s0w, nm_, idx, abv),
                         "share": sh, "share_verdict": "HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID"}
            for x_, y_ in (("mp", "b"), ("p", "b"), ("mp-lb", "b-lb"), ("p-lb", "b-lb")):
                e[f"{x_} - {y_}"] = diff(x_, y_, nm_, idx, abv)
            e["refs_minus_s0+rrf"] = {k: CP.boot_mean(refs[nm_][k][:, 0] - s0r, idx)
                                      for k in ("rrf", "s0+walk", "twin0", "gnn0")}
            V["Q4"][nm_] = e
    res["verdicts"] = V
    res["decision"] = decide(V)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"verdicts {json.dumps(V)}")
    log(f"decision {res['decision']}")
    return res


def decide(V):
    """The declared rule, from R0, Q2, Q1 and Q3."""
    r0 = [(V["R0"].get(k) or {}).get("verdict") for k in KB_READS]
    if None in r0:
        return {"carry_lb": None, "next": None, "why": "R0 incomplete (no --cs20)", "report": False}
    if "DIFFERS" in r0:
        return {"carry_lb": None, "next": "stop", "why": "R0 DIFFERS: nothing is read until its cause is found",
                "report": True}

    def q2(nm_):
        return ((V["Q2"].get(nm_) or {}).get("b-lb - b") or {}).get("verdict")

    if None in (q2("webqsp_sf"), q2("metaqa")):
        return {"carry_lb": None, "next": None, "why": "Q2 incomplete", "report": False}
    lb = q2("webqsp_sf") == "HELPS" and q2("metaqa") != "HURTS"
    sfx = "-lb" if lb else ""
    q1 = (V["Q1"].get(f"mp{sfx} - b{sfx}") or {}).get("verdict")
    q3 = (((V["Q3"].get("webqsp_sf") or {}).get(f"p{sfx}") or {}).get("minus_none_rd") or {}).get("verdict")
    if q1 == "ABOVE":
        return {"carry_lb": lb, "next": "part 6: metaqa and the three passage fit carves in training, each passage "
                                        "graph held out in turn (LOGO)", "why": f"Q1 mp{sfx} - b{sfx} ABOVE",
                "report": False}
    if q3 == "ABOVE":
        return {"carry_lb": lb, "next": "part 6: the passage-only route over the three passage graphs",
                "why": f"Q1 mp{sfx} - b{sfx} {q1}; Q3 p{sfx} - none/rd ABOVE", "report": False}
    return {"carry_lb": lb, "next": "report to Swastik before part 6",
            "why": f"Q1 mp{sfx} - b{sfx} {q1}; Q3 p{sfx} - none/rd {q3}", "report": True}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def untyped_toy(root, rng, carve="read", rows=48):
    """chainpop17's toy look with its relations removed and a second stored function (gnn0: the twin's metrics of
    the chunk's rows in reverse order): a passage look in form, at root/ptoy/carve."""
    P = RT.proj_matrix()
    src_root = Path(root) / f"_src_{carve}"
    CP.toy_world(src_root, rng, P, rows=rows, chunk=8)
    src = src_root / "toy" / "read"
    dst = Path(root) / "ptoy" / carve
    (dst / "chunks").mkdir(parents=True)
    for p in sorted((src / "chunks").glob("c*.npz")):
        with np.load(p) as zf:
            c = {k: zf[k] for k in zf.files if k != "e_rel"}
        c["q_metrics"] = np.concatenate([c["q_metrics"], c["q_metrics"][::-1]], axis=1)
        np.savez(dst / "chunks" / p.name, **c)
    for r_ in sorted(src.glob("record*.json")):
        rr = json.loads(r_.read_text(encoding="utf-8"))
        rr["relations"] = {"typed": False, "offset": -1, "n_relations": 0, "k_rel": 4}
        rr["functions"] = ["twin0", "gnn0"]
        (dst / r_.name).write_text(json.dumps(rr), encoding="utf-8")
    return dst


def selftest():
    import torch
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    strip = lambda h: [{k: v for k, v in e.items() if k != "seconds"} for e in h]  # noqa: E731
    rng = np.random.default_rng(7)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # 1. the passage builds
        root = td / "look"
        src = untyped_toy(root, rng)
        out = td / "typed"
        info, keys, lab, C = typed_root("ptoy", "read", root, out, 4)
        check("k-means: K clusters, none empty", info["K"] == 4 and info["empty"] == 0)
        same, slots_ok, nonstruct_ok, xcut = True, True, True, True
        _k2, nodes, proj, _g = union_triples(sorted((src / "chunks").glob("c*.npz")))
        Pn = unit(proj.astype(np.float32))
        brute = np.argmax(descriptors(keys, nodes, Pn) @ C.T, 1)
        check("every triple's cluster is its nearest centroid", np.array_equal(brute, lab))
        for p in sorted((src / "chunks").glob("c*.npz")):
            with np.load(p) as a_, np.load(out / "ptoy" / "read" / "chunks" / p.name) as b_:
                for k in PKEYS:
                    if k == "x":
                        xcut &= np.array_equal(a_["x"][:, :4], b_["x"])
                    else:
                        same &= np.array_equal(a_[k], b_[k])
                c = {k: a_[k] for k in a_.files}
                E = b_["e_rel"]
            n_off = np.r_[0, np.cumsum(c["q_pool_size"])]
            e_off = np.r_[0, np.cumsum(c["q_edges"])]
            for i in range(c["q_pool_size"].size):
                pool = c["pool"][n_off[i]:n_off[i + 1]].astype(np.int64)
                for j in range(int(e_off[i]), int(e_off[i + 1])):
                    u, v = int(c["e_u"][j]), int(c["e_v"][j])
                    want = set()
                    if c["e_fam"][j] == 0:
                        if c["e_fwd"][j] == 1:
                            want.add(int(lab[np.searchsorted(keys, (pool[u] << SHIFT) | pool[v])]))
                        if c["e_bwd"][j] == 1:
                            want.add(int(lab[np.searchsorted(keys, (pool[v] << SHIFT) | pool[u])]))
                        got = [int(x) for x in E[j] if x >= 0]
                        slots_ok &= sorted(want) == got
                    else:
                        nonstruct_ok &= bool((E[j] == -1).all())
        check("typed copies keep every array (x cut after rrf)", same and xcut)
        check("each structural message's slots are its triples' clusters", slots_ok)
        check("non-structural messages carry no slot", nonstruct_ok)
        lean = td / "lean"
        lean.mkdir()
        a = SimpleNamespace(ds="ptoy", carve="read", hops=2, ks="2", regimes="1,0", rows="all",
                            max_chains=CP.MAX_CHAINS, limit=None, k_fine=4, src_root=str(root),
                            out=str(lean / "b.npz"), edges_out=str(lean / "e.npz"))
        side = pbuild_cmd(a)
        d = CS.load(lean / "b.npz")
        check("the build reads 4 pseudo-relations and its levels", side["types"]["K"] == 4
              and list(d["level_names"]) == ["none", "dir", "tt2", "exact"])
        check("the companion was checked against the build", (lean / "e.npz").exists())
        check("the temporary root is removed", not any(p.name.startswith("cs21tmp_") for p in lean.iterdir()))

        # 2. toy KB builds and companions (as chainscore20's selftest makes them) and toy passage builds
        P = RT.proj_matrix()
        roots = {}
        for nm_, sd, rws in (("train", 11, 96), ("select", 12, 48), ("read", 13, 64)):
            r_ = td / "kb" / nm_
            reld = td / "kb" / f"{nm_}_rel"
            reld.mkdir(parents=True)
            _truth, names = CP.toy_world(r_, np.random.default_rng(sd), P, rows=rws)
            np.save(reld / "toy_rel_embeddings.npy", names)
            roots[nm_] = (r_, reld)
        K_ = td / "builds"
        K_.mkdir()
        for nm_, regs, tfn, rws, hops in (("train", "1,0.25,0", "id", "all", 2), ("select", "1", "id", "all", 2),
                                          ("read", "1", "id", "all", 3), ("train", "1,0", "al4", "even", 2),
                                          ("train", "1,0", "sp4", "odd", 2), ("train", "1,0", "mg3", "all", 3),
                                          ("train", "1,0", "rf", "even", 3)):
            r_, reld = roots[nm_]
            name = nm_ if tfn == "id" else f"train-{tfn}"
            C19.build19(SimpleNamespace(ds="toy", carve="read", hops=hops, ks="2", limit=None,
                                        max_chains=CP.MAX_CHAINS, regimes=regs, transform=tfn, rows=rws,
                                        out=str(K_ / f"{name}.npz")), root=r_, rel_dir=reld)
            C20.edges_cmd(SimpleNamespace(base=str(K_ / f"{name}.npz"), out=str(K_ / f"{name}.e.npz"),
                                          no_node_check=False), root=r_)
        proot = td / "plook"
        for carve, sd, rws in (("fit", 21, 64), ("select", 22, 32), ("read", 23, 40)):
            untyped_toy(proot, np.random.default_rng(sd), carve, rws)
        for carve, regs in (("fit", "1,0.25,0"), ("select", "1"), ("read", "1")):
            pbuild_cmd(SimpleNamespace(ds="ptoy", carve=carve, hops=2, ks="2", regimes=regs, rows="all",
                                       max_chains=CP.MAX_CHAINS, limit=None, k_fine=4, src_root=str(proot),
                                       out=str(K_ / f"p{carve}.npz"), edges_out=str(K_ / f"p{carve}.e.npz")))

        # 3. lb's offsets against a count, pair by pair
        worst = 0.0
        for name in ("train", "train-rf", "read", "pfit"):
            d = CS.load(K_ / f"{name}.npz")
            got = lb_offsets(d)
            ro = d["rowoff"]
            ln = np.argmax(d["FI"][:, HOPC], 1)
            for i in range(ro.size - 1):
                q = np.arange(int(ro[i]), int(ro[i + 1]))
                for p_ in q:
                    k = int(((d["p_lev"][q] == d["p_lev"][p_]) & (ln[q] == ln[p_])).sum())
                    worst = max(worst, abs(float(got[p_]) - math.log(k)))
        check(f"lb_offsets is log C_level,length, pair by pair (max |diff| {worst:.1e})", worst < 1e-6)
        bad = CS.load(K_ / "read.npz")
        bad["FI"] = bad["FI"].copy()
        bad["FI"][0, CS.LOGC] += 0.5
        try:
            lb_offsets(bad)
            check("lb_offsets refuses a build whose log C_level it cannot recount", False)
        except SystemExit:
            check("lb_offsets refuses a build whose log C_level it cannot recount", True)

        # 4. the grouped draw
        tr = C19.prep(CS.load(K_ / "train.npz"))
        aug = [C19.prep(CS.load(K_ / f"train-{t}.npz")) for t in C19.TRANSFORMS[1:]]
        pf = C19.prep(CS.load(K_ / "pfit.npz"))
        bl = [tr] + aug + [pf]
        x1 = draw21(np.random.default_rng(3), bl, [list(range(5))], 500)
        x2 = C19.draw(np.random.default_rng(3), bl, list(range(5)), 500)
        check("one group is chainscore19's draw", np.array_equal(x1, x2))
        x3 = draw21(np.random.default_rng(3), bl, [list(range(5)), [5]], 6000)
        sh5 = float((x3[:, 0] == 5).mean())
        valid = all(np.isin(x3[x3[:, 0] == b, 1], bl[b]["train_rows"]).all()
                    and (x3[x3[:, 0] == b, 2] < bl[b]["n_regimes"]).all() for b in range(6))
        check(f"two groups: 2wiki's share {sh5:.3f} (half), rows and regimes valid", abs(sh5 - 0.5) < 0.03 and valid)

        # 5. the batch: without LB chainscore20's; with LB only the offset changes, to lb_offsets at the pairs
        st = C19.input_stats(tr)
        LBs = [lb_offsets(b) for b in bl]
        it = x3[:24]
        b0 = C20.make_batch20(it, bl, [None] * 6, st, pairs=True, train=True)
        b1 = make_batch21(it, bl, [None] * 6, None, st, pairs=True, train=True)
        b2 = make_batch21(it, bl, [None] * 6, LBs, st, pairs=True, train=True)
        tens = [k for k in b0 if isinstance(b0[k], torch.Tensor)]
        check("without LB the batch is chainscore20's", all(torch.equal(b0[k], b1[k]) for k in tens))
        lc_ok = all(torch.equal(b0[k], b2[k]) for k in tens if k != "lc")
        for j, (bi, i, _ri) in enumerate(it):
            p0, p1 = int(bl[bi]["rowoff"][i]), int(bl[bi]["rowoff"][i + 1])
            lc_ok &= np.array_equal(b2["lc"][j, :p1 - p0].numpy(), LBs[bi][p0:p1])
            lc_ok &= bool((b2["lc"][j, p1 - p0:] == 0).all())
        check("with LB only lc changes, to lb_offsets at the pairs", lc_ok)

        # 6. the passage rule and the references against brute force; gnn0 aligned and checked
        pr = CS.load(K_ / "pread.npz")
        rr = np.random.default_rng(5)
        good = True
        for i in range(pr["n"].size):
            n = int(pr["n"][i])
            s = rr.normal(size=n)
            s[rr.random(n) < 0.3] = 0.0      # ties, broken by rrf
            a0 = int(pr["xn_off"][i])
            is_s = pr["seed0"][a0:a0 + n]
            rank = pr["rank"][pr["rank_off"][i]:pr["rank_off"][i + 1]]
            keyf = sorted(range(n), key=lambda v: (0, int(rank[v])) if is_s[v] else (1, -float(s[v]), int(rank[v])))
            gold = np.zeros(n, bool)
            gold[pr["gold"][pr["gold_off"][i]:pr["gold_off"][i + 1]]] = True
            good &= rank_row_p(pr, i, s) == CP.metrics(np.asarray(keyf[:5]), gold, int(pr["gt"][i]))
        check("the passage rule: seeds first by rrf, then s, ties by rrf", good)
        R = passage_refs(pr)
        li = list(pr["level_names"]).index("none")
        w = pr["post1"][:, CP.FITS.index("rd")]
        good = True
        for i in range(pr["n"].size):
            n = int(pr["n"][i])
            s = np.zeros(n)
            q = [p_ for p_ in range(int(pr["rowoff"][i]), int(pr["rowoff"][i + 1])) if pr["p_lev"][p_] == li]
            tw_ = float(np.nan_to_num(w[np.asarray(q, np.int64)].astype(np.float64)).sum())
            for p_ in q:
                for t_ in range(int(pr["node_off"][p_]), int(pr["node_off"][p_ + 1])):
                    if tw_ > 0:
                        s[pr["node"][t_]] += float(np.nan_to_num(w[p_])) / tw_ * float(pr["mass"][t_])
            a0 = int(pr["xn_off"][i])
            is_s = pr["seed0"][a0:a0 + n]
            rank = pr["rank"][pr["rank_off"][i]:pr["rank_off"][i + 1]]
            keyf = sorted(range(n), key=lambda v: (0, int(rank[v])) if is_s[v] else (1, -float(s[v]), int(rank[v])))
            keyr = sorted(range(n), key=lambda v: (0 if is_s[v] else 1, int(rank[v])))
            gold = np.zeros(n, bool)
            gold[pr["gold"][pr["gold_off"][i]:pr["gold_off"][i + 1]]] = True
            good &= np.allclose(R["s0+walk"][i], CP.metrics(np.asarray(keyf[:5]), gold, int(pr["gt"][i])))
            good &= np.allclose(R["s0+rrf"][i], CP.metrics(np.asarray(keyr[:5]), gold, int(pr["gt"][i])))
        check("s0+rrf and s0+walk against brute force", good)
        g0 = look_fn(pr, "gnn0", proot)
        _rc, m_idx = CP.read_record("ptoy", "read", proot)
        want = []
        for p_ in passage_paths("ptoy", "read", proot)[0]:
            with np.load(p_) as zf:
                want.append(zf["q_metrics"][:, 1][:, m_idx])
        check("gnn0: the look's stored metrics of each build row", np.array_equal(
            g0, np.concatenate(want).astype(np.float64)[pr["row_ix"]]))
        pr2 = dict(pr)
        pr2["twin0"] = np.asarray(pr["twin0"])[::-1].copy()
        try:
            look_fn(pr2, "gnn0", proot)
            check("look_fn refuses a build whose rows are not the look's", bool(np.array_equal(pr2["twin0"],
                                                                                              pr["twin0"])))
        except SystemExit:
            check("look_fn refuses a build whose rows are not the look's", True)

        # 7. b runs chainscore20's loop step for step (the MLP f and the untyped GNN f)
        sel = CS.load(K_ / "select.npz")
        rd = CS.load(K_ / "read.npz")
        comps = [C20.load_comp(K_ / "train.e.npz", tr, sha(K_ / "train.npz"))] + [
            C20.load_comp(K_ / f"train-{t}.e.npz", d_, sha(K_ / f"train-{t}.npz"))
            for t, d_ in zip(C19.TRANSFORMS[1:], aug)]
        sc = C20.load_comp(K_ / "select.e.npz", sel, sha(K_ / "select.npz"))
        rc = C20.load_comp(K_ / "read.e.npz", rd, sha(K_ / "read.npz"))
        for f in ("em-gr", "en-gr"):
            gn = C20.arm_spec(f)["gnn"]
            cps, scc, rcc = (comps, sc, rc) if gn else ([None] * 5, None, None)
            m20, i20 = C20.train20(f, [tr] + aug, cps, sel, scc, 2, 192, 16, SEED)
            x20, _ = C20.read20(m20, rd, rcc, i20["stats"])
            m21, i21 = train21("b", f, [tr] + aug, cps, None, [list(range(5))], sel, scc, None, "kb", 2, 192, 16,
                               SEED)
            x21, _ = read21(m21, rd, rcc, None, i21["stats"], "kb")
            check(f"b with {f} is chainscore20's train20 and read20 exactly (read {np.round(x21.mean(0), 4).tolist()})",
                  strip(i20["history"]) == strip(i21["history"]) and np.array_equal(x20, x21))

        # 8. the lb, mixed and passage arms: deterministic, and the loss falls
        ps_ = CS.load(K_ / "pselect.npz")
        pfc = C20.load_comp(K_ / "pfit.e.npz", pf, sha(K_ / "pfit.npz"))
        psc = C20.load_comp(K_ / "pselect.e.npz", ps_, sha(K_ / "pselect.npz"))
        for arm, f in (("mp-lb", "em-gr"), ("p", "en-gr"), ("p-lb", "ena-gr"), ("mp", "ena-gr")):
            spec = arm_spec(arm, f)
            gn = spec["gnn"]
            if spec["data"] == "p":
                B_, Cp, G_, Sd, Sc_ = [pf], [pfc if gn else None], [[0]], ps_, (psc if gn else None)
            else:
                B_, Cp = [tr] + aug + [pf], (comps + [pfc] if gn else [None] * 6)
                G_, Sd, Sc_ = [list(range(5)), [5]], sel, (sc if gn else None)
            LB_ = [lb_offsets(b) for b in B_] if spec["lb"] else None
            Slb = lb_offsets(Sd) if spec["lb"] else None
            rule = "passage" if spec["data"] == "p" else "kb"
            m1, i1 = train21(arm, f, B_, Cp, LB_, G_, Sd, Sc_, Slb, rule, 3, 192, 16, SEED)
            m2, i2 = train21(arm, f, B_, Cp, LB_, G_, Sd, Sc_, Slb, rule, 3, 192, 16, SEED)
            prc = C20.load_comp(K_ / "pread.e.npz", pr, sha(K_ / "pread.npz")) if gn else None
            y1, _ = read21(m1, pr, prc, lb_offsets(pr) if spec["lb"] else None, i1["stats"], "passage")
            y2, _ = read21(m2, pr, prc, lb_offsets(pr) if spec["lb"] else None, i2["stats"], "passage")
            h_ = i1["history"]
            check(f"{arm} with {f}: deterministic, loss {h_[0]['loss']} -> {h_[-1]['loss']}, passage read "
                  f"{np.round(y1.mean(0), 4).tolist()}", strip(h_) == strip(i2["history"]) and np.array_equal(y1, y2)
                  and h_[-1]["loss"] < h_[0]["loss"])

        # 9. a passage build whose sidecar is another build's is refused
        try:
            check_passage(K_ / "pread.npz", sha(K_ / "pfit.npz"))
            check("check_passage refuses another build's sidecar", False)
        except SystemExit:
            check("check_passage refuses another build's sidecar", True)

        # 10. the commands end to end: chainscore20's em-gr (R0's reference), the six arms, the grade
        L_ = td / "out"
        L_.mkdir()
        augf = []
        for t in C19.TRANSFORMS[1:]:
            augf += ["--aug", f"{t}={K_}/train-{t}.npz"]
        kbr = []
        for n in KB_READS:
            kbr += ["--read", f"{n}={K_}/read.npz"]
        pr_ = []
        for n in P_READS:
            pr_ += ["--pread", f"{n}={K_}/pread.npz"]
        ep = ["--epochs", "2", "--per-epoch", "128", "--batch", "16"]
        assert C20.main(["train", "--arm", "em-gr", "--fit", f"{K_}/train.npz", *augf, "--select",
                         f"{K_}/select.npz", *kbr, *ep, "--out", f"{L_}/cs20-em-gr.json",
                         "--rows-out", f"{L_}/cs20-em-gr.rows.npz"]) == 0
        for arm in ARMS:
            dat = arm.split("-")[0]
            tin = (["--fit", f"{K_}/train.npz", *augf, "--select", f"{K_}/select.npz"] if dat != "p" else []) + (
                ["--pfit", f"{K_}/pfit.npz"] if dat != "b" else []) + (
                ["--pselect", f"{K_}/pselect.npz"] if dat == "p" else [])
            assert main(["train", "--arm", arm, "--f", "em-gr", *tin, *kbr, *pr_, *ep, "--out", f"{L_}/cs21-{arm}.json",
                         "--rows-out", f"{L_}/cs21-{arm}.rows.npz", "--state-out", f"{L_}/cs21-{arm}.pt"]) == 0
        ed = [f"fit={K_}/train.e.npz", f"select={K_}/select.e.npz", f"pfit={K_}/pfit.e.npz"] + [
            f"{t}={K_}/train-{t}.e.npz" for t in C19.TRANSFORMS[1:]] + [
            f"{n}={K_}/read.e.npz" for n in KB_READS] + [f"{n}={K_}/pread.e.npz" for n in P_READS]
        eda = []
        for e_ in ed:
            eda += ["--edges", e_]
        assert main(["train", "--arm", "mp", "--f", "en-gr", "--fit", f"{K_}/train.npz", *augf, "--select",
                     f"{K_}/select.npz", "--pfit", f"{K_}/pfit.npz", *kbr, *pr_, *eda, *ep, "--out",
                     f"{L_}/cs21-mp-en.json", "--rows-out", f"{L_}/cs21-mp-en.rows.npz"]) == 0
        try:
            main(["train", "--arm", "p", "--f", "em-gr", "--fit", f"{K_}/train.npz", "--pfit", f"{K_}/pfit.npz",
                  "--pselect", f"{K_}/pselect.npz", *kbr, *pr_, *ep, "--out", f"{L_}/x.json", "--rows-out",
                  f"{L_}/x.rows.npz"])
            check("an arm refuses a training input it does not take", False)
        except SystemExit:
            check("an arm refuses a training input it does not take", True)
        g = ["grade"]
        for arm in ARMS:
            g += ["--arm", f"{arm}={L_}/cs21-{arm}.json"]
        g += ["--cs20", f"em-gr={L_}/cs20-em-gr.json", *kbr, *pr_, "--look-root", str(proot)]
        assert main(g + ["--out", f"{L_}/cs21.json"]) == 0
        res = json.loads((L_ / "cs21.json").read_text(encoding="utf-8"))
        V = res["verdicts"]
        check(f"the grade: R0 {[(k, v['verdict']) for k, v in V['R0'].items()]}",
              all(V["R0"][k]["verdict"] == "REPRODUCES" for k in KB_READS))
        check("the grade fills every verdict", set(V["Q1"]) == {"mp - b", "mp-lb - b-lb"}
              and set(V["Q5"]) == set(V["Q1"])
              and set(V["Q2"]) == set(KB_READS) | set(P_READS) and set(V["Q3"]) == set(KB_READS)
              and set(V["Q4"]) == set(P_READS) and all(set(ARMS) <= set(V["Q4"][n]) for n in P_READS)
              and "minus_none_rd_D1" in V["Q3"]["webqsp_sf"]["p"])
        print(f"  toy decision: {res['decision']}")
    print("selftest", "passed" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    b = sub.add_parser("pbuild")
    b.add_argument("--ds", required=True)
    b.add_argument("--carve", required=True, help="a carve, or carves joined by + (one union graph and population)")
    b.add_argument("--hops", type=int, default=2)
    b.add_argument("--ks", default="4,16,64")
    b.add_argument("--regimes", default="1")
    b.add_argument("--rows", default="all", choices=("all", "even", "odd"))
    b.add_argument("--max-chains", type=int, default=CP.MAX_CHAINS)
    b.add_argument("--k-fine", type=int, default=K_FINE)
    b.add_argument("--limit", type=int, default=None)
    b.add_argument("--src-root", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--edges-out", default=None)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--f", required=True, choices=FS, help="chainscore20's carried arm")
    t.add_argument("--fit", default=None, help="metaqa's id fit build (b, mp)")
    t.add_argument("--aug", action="append", default=[], help="NAME=BUILD for each of metaqa's transforms (b, mp)")
    t.add_argument("--select", default=None, help="metaqa's select build (b, mp)")
    t.add_argument("--pfit", default=None, help="2wiki's typed fit build (mp, p)")
    t.add_argument("--pselect", default=None, help="2wiki's typed select build (p)")
    t.add_argument("--read", action="append", default=[], help=f"NAME=BUILD for each of {KB_READS}")
    t.add_argument("--pread", action="append", default=[], help=f"NAME=BUILD for each of {P_READS}")
    t.add_argument("--edges", action="append", default=[], help="NAME=COMPANION for every build a GNN f loads")
    t.add_argument("--epochs", type=int, default=C19.EPOCHS)
    t.add_argument("--per-epoch", type=int, default=C19.PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    g = sub.add_parser("grade")
    g.add_argument("--arm", action="append", required=True)
    g.add_argument("--cs20", default=None, help="NAME=JSON: chainscore20's carried arm (R0)")
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--pread", action="append", required=True)
    g.add_argument("--look-root", default=None)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "pbuild":
        pbuild_cmd(a)
        return 0
    if a.cmd in ("train", "grade"):
        res = train_cmd(a) if a.cmd == "train" else grade_cmd(a)
        p = Path(a.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                       encoding="utf-8")
        os.replace(tmp, p)
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
