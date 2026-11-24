"""SEM2 R2: the one-hop traversal in the first stage (docs/SEM2_EDGE_OPERATORS.md section 9, declared 24 Nov 2026
before any R2 number).

    python outputs/mp_unified/zsem2t.py selftest [--threads 2]
    python outputs/mp_unified/zsem2t.py sel --ds D --arm e3v|frozen [--host] [--threads 8]
    python outputs/mp_unified/zsem2t.py read --ds D --arm e3v|frozen [--host] [--threads 8]
    python outputs/mp_unified/zsem2t.py report

R1's saved rotation, as R1 trained it: no refit and no question-side fit. Each question's seeds are the top 5 of the
dense list v and the top 5 of SPLADE. Every operator moves each seed (unit length). The attention alpha_r(q, s) is a
softmax, at R1's temperature, of the question's residual beyond the seed against each moved seed; no weight is learned.
Hop 1 scores node d by max over seeds of t(q, s) . x_d, t = sum_r alpha_r m_r(s) + lambda x_q, the seeds left out; the
list is the seeds (hop 0), then the hop-1 nodes. lambda is chosen on the selection carves, leave-one-dataset-out. The
read sets RRF(v, trav, SPLADE) against RRF(v, SPLADE) and against trav0, the same traversal with every operator at its
start (the cosine). zsem2, zsem2h,
zsem1, zul1, zmhr, zenc, zenc1 and zenc3v are imported and never edited. Outputs only under outputs/sem2/r2/.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import zsem2h as H  # noqa: E402
import zsem1 as S1  # noqa: E402
import zap1 as ZA  # noqa: E402

E = Z.E
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 9"
OUT = Z.ROOT / "outputs" / "sem2" / "r2"
SEEDS = 5                                     # per list: the top 5 of v and the top 5 of SPLADE (P5b's seeds)
LAMBDAS = (0.0, 0.25, 0.5, 1.0, 2.0)
TAU = Z.CFG["tau"]                            # R1's temperature
SEL = ("multihoprag", "squad", "musique", "metaqa", "hotpotqa", "2wiki")   # WebQSP has no selection carve
SETS = {"e3v": Z.SEVEN, "frozen": Z.SMALL}
SELS = {"e3v": SEL, "frozen": Z.SMALL}
QB, NB = 128, 1 << 15                         # questions per batch, nodes per block
ROOM = 1.0                                    # the ceiling C in R@5 points (9.6)
KINDS = ("trav", "trav0", "oracle")


def log(msg):
    print(f"[zsem2t {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stop(msg):
    raise SystemExit(f"zsem2t: STOP: {msg}")


def script_sha():
    return Z.sha_file(Path(__file__))


# ── questions, golds and the first-stage lists ──────────────────────────────


def questions(ds, arm, carve, host):
    """carve 'sel' (s1sel; MultiHop-RAG: its dev split) or 'read' (s1eval; MultiHop-RAG: its test split)."""
    import zenc3v  # noqa: F401  (registers e3v in zenc's ENCODERS)
    info = {}
    if ds == "multihoprag":
        import zmhr as M
        served = M.mhr_root()
        M.point_zenc()
        qs, nodes = M.queries_of(served), M.nodes_of(served)
        pos = {x["node_id"]: i for i, x in enumerate(nodes)}
        nby = json.loads((served / ds / "DATASET.json").read_text(encoding="utf-8"))["queries_by_split"]
        split = np.asarray(sum(([s] * int(nby[s]) for s in M.SPLITS), []))
        if split.size != len(qs):
            stop("MultiHop-RAG: the split counts do not cover the questions")
        idx = np.flatnonzero(split == ("dev" if carve == "sel" else "test"))
        ids = [qs[i]["query_id"] for i in idx]
        golds = [sorted({pos[g] for g in qs[i]["gold_node_ids"]}) for i in idx]
        flat, off = S1.flatten(golds)
        info = {"split": "dev" if carve == "sel" else "test", "dataset_json_sha256": E.sha256(served / ds / "DATASET.json")}
    else:
        import zenc1 as E1
        import zul1 as UL
        served, _freeze = E1.package(host)
        if carve == "sel":
            ids, golds, csha = UL.carve_questions(ds, "s1sel")
            flat, off = S1.flatten(golds)
            info = {"carve": "s1sel", "npz_sha256": csha}
        else:
            s1 = E.s1eval(ds)
            ids, gt = s1["ids"], s1["gold_total"]
            if np.any(gt <= 0):
                stop(f"{ds}: an s1eval question has no gold")
            flat, off = E.gold_sets(served, ds, ids, gt)
            info = {"carve": "s1eval", "path": s1["path"], "sha256": s1["sha256"]}
    rows, Q = S1.query_rows(served, arm, ds, ids)
    spl = S1.splade(served, ds, rows)
    paths = E.vec_shards(served, arm, ds, "docs")
    info.update({"questions": len(ids), "with_gold": int((np.diff(off) > 0).sum())})
    return {"ids": list(ids), "rows": rows, "Q": Q, "flat": flat, "off": off, "spl": spl, "paths": paths, "served": served,
            "info": info}


def vlist(qd, n):
    """zenc's exact dense top K; a corpus shorter than K (MultiHop-RAG, 609) pads with -1."""
    v_top, _s, _p = E.search(qd["Q"], qd["paths"], E.K)
    v_top[(v_top < 0) | (v_top >= n)] = -1
    return v_top


def seeds_of(v_top, spl, k=SEEDS):
    """The top k of v, then the top k of SPLADE not already in; -1 pads to 2k."""
    out = np.full((v_top.shape[0], 2 * k), -1, np.int64)
    for q in range(v_top.shape[0]):
        got = []
        for g in list(v_top[q, :k]) + list(spl[q, :k]):
            g = int(g)
            if g >= 0 and g not in got:
                got.append(g)
        out[q, :len(got)] = got
    return out


# ── the operators (R1's, gate G-R as section 8.10) ──────────────────────────


def operators(torch, G, X, ds, arm, gates):
    cfg = dict(Z.CFG)
    r1, z = H.load_r1(ds, arm)
    if r1 is None:
        stop(f"R1's {ds} {arm} record or npz is missing")
    split = Z.node_split(G.name, G.n)
    labs, ops, op_of, groups = Z.plan(G, split, cfg)
    names = [g.name for g in groups]
    rng = np.random.default_rng(Z.SEED)
    test = Z.take(groups, split, Z.is_test, cfg["n_test"], rng)
    Z.take(groups, split, Z.is_dev, cfg["n_dev"], rng)               # the dev draw, so the rng is R1's after it
    test["nR"] = G.nR
    same = {k: bool(np.array_equal(test[k], z[f"test_{k}"])) for k in ("s", "d", "lab", "grp")}
    same["read groups"] = names == list(r1["read_groups"])
    same["operators"] = [o["name"] for o in ops] == list(r1["operators"])
    if not all(same.values()):
        stop(f"gate G-R: R1's samples are not reproduced: {same}")
    D = X.D
    Dp = D + cfg["extra"]
    if r1["fits"]["rotation"]["D_prime"] != Dp:
        stop("R1's rotation D' differs")
    m1 = Z.make_model(torch, D, Dp, len(ops), "rotation")
    m1.load_state_dict({k: torch.from_numpy(np.asarray(z[f"rotation.{k}"])) for k in m1.state_dict()})
    m1.eval()
    r_rot, _ = Z.rank_read(torch, m1, X, test, Z.exclusions(G, test), op_of)
    agree = float((r_rot == z["rank_rotation"]).mean())
    dm = abs(Z.measures(r_rot)["mrr"] - Z.measures(z["rank_rotation"])["mrr"])
    gates["G-R"] = {"samples": same, "rank_agreement": round(agree, 6), "mrr_diff": round(dm, 4),
                    "verdict": "PASS" if agree >= 0.999 and dm <= 0.01 else "FAIL"}
    if gates["G-R"]["verdict"] != "PASS":
        stop(f"gate G-R: {gates['G-R']}")
    return m1, ops, op_of, {"r1_record_sha256": Z.sha_file(H.R1 / ds / f"{arm}.json"), "r1_npz_sha256": r1["npz_sha256"]}


def mapped_inv(torch, m1, X, block=1 << 16):
    """1 / |P v_d| for every node (x_d = P v_d / |P v_d|)."""
    out = np.empty(X.n, np.float32)
    P = m1.P.detach()
    with torch.no_grad():
        for b0 in range(0, X.n, block):
            b1 = min(X.n, b0 + block)
            x = torch.from_numpy(X.rows(slice(b0, b1))) @ P.T
            out[b0:b1] = (1.0 / x.norm(dim=1).clamp_min(1e-12)).numpy()
    return out


# ── the traversal (9.3) ─────────────────────────────────────────────────────


def plan_batch(torch, m1, dirs, X, Qraw, seeds, quads=None):
    """Raw-space vectors for one batch. u (b, S, D): each seed's alpha mix moved back through P; uq (b, D): x_q through
    P; u0 / u0q: the seed and the question themselves (trav0); u_or (b, S, D): the oracle's mix (one-hot on the chosen
    operator where a seed has one); alpha (b, S, R)."""
    b, S = seeds.shape
    valid = seeds >= 0
    sd = np.where(valid, seeds, 0)
    q = Qraw / np.linalg.norm(Qraw, axis=1, keepdims=True)
    vs = X.rows(sd.reshape(-1))
    R = len(dirs)
    with torch.no_grad():
        P = m1.P.detach()
        xq = m1.embed(torch.from_numpy(q))
        xs = m1.embed(torch.from_numpy(vs))
        Ms = []
        for r in range(R):
            y = m1.query(xs, torch.full((xs.shape[0],), r, dtype=torch.long),
                         torch.full((xs.shape[0],), bool(dirs[r]), dtype=torch.bool))
            Ms.append(y / y.norm(dim=1, keepdim=True).clamp_min(1e-12))
        M3 = torch.stack(Ms, 1).view(b, S, R, -1)
        xs3 = xs.view(b, S, -1)
        rho = xq[:, None, :] - (xq[:, None, :] * xs3).sum(-1, keepdim=True) * xs3
        rn = rho.norm(dim=-1, keepdim=True)
        rhat = rho / rn.clamp_min(1e-12)
        logits = (M3 * rhat[:, :, None, :]).sum(-1) / TAU
        logits = torch.where(rn < 1e-6, torch.zeros_like(logits), logits)
        alpha = torch.softmax(logits, -1)
        mix = (alpha[..., None] * M3).sum(2)
        u = mix @ P
        uq = xq @ P
        u_or, picked = None, None
        or_q = or_s = torch.zeros(0, dtype=torch.long)
        if quads is not None:
            mix_or = mix.clone()
            picked = np.full((b, S), -1, np.int64)
            if quads[0].size:
                qi, sj, r, g = quads
                xg = m1.embed(torch.from_numpy(X.rows(g)))
                val = (M3[torch.from_numpy(qi), torch.from_numpy(sj), torch.from_numpy(r)] * xg).sum(-1).numpy()
                o = np.lexsort((r, -val, sj, qi))                   # per (question, seed): the highest score, then op id
                qi, sj, r = qi[o], sj[o], r[o]
                first = np.r_[True, (qi[1:] != qi[:-1]) | (sj[1:] != sj[:-1])]
                qi, sj, r = qi[first], sj[first], r[first]
                picked[qi, sj] = r
                or_q, or_s = torch.from_numpy(qi), torch.from_numpy(sj)
                mix_or[torch.from_numpy(qi), torch.from_numpy(sj)] = M3[torch.from_numpy(qi), torch.from_numpy(sj),
                                                                        torch.from_numpy(r)]
            u_or = mix_or @ P
    return {"u": u, "uq": uq, "u0": torch.from_numpy(vs).view(b, S, -1), "u0q": torch.from_numpy(q.astype(np.float32)),
            "u_or": u_or, "or_q": or_q, "or_s": or_s, "alpha": alpha.numpy(), "valid": torch.from_numpy(valid),
            "picked": picked}


def oracle_quads(G, op_of, seeds, flat, off, q0):
    """(question in batch, seed slot, operator, gold) for every record edge from a seed to one of its question's golds
    (R labels with an operator, either direction; S labels from the seed as stored). Sorted unique."""
    n = G.n
    qi, sj, s, g = [], [], [], []
    for i in range(seeds.shape[0]):
        gs = flat[off[q0 + i]:off[q0 + i + 1]]
        for j in range(seeds.shape[1]):
            if seeds[i, j] < 0 or gs.size == 0:
                continue
            gg = gs[gs != seeds[i, j]]
            qi += [i] * gg.size
            sj += [j] * gg.size
            s += [int(seeds[i, j])] * gg.size
            g += gg.tolist()
    qi, sj, s, g = (np.asarray(x, np.int64) for x in (qi, sj, s, g))
    out = []
    if qi.size:
        k = np.minimum(s, g) * n + np.maximum(s, g)
        for L in range(G.nR):
            if op_of[L] >= 0:
                m = Z.member(k, G.r_keys[L])
                out.append(np.stack([qi[m], sj[m], np.full(int(m.sum()), op_of[L]), g[m]]))
        if G.s_key.size:
            base = (s * n + g) * G.RS
            lo, hi = np.searchsorted(G.s_key, base, "left"), np.searchsorted(G.s_key, base + G.RS, "left")
            pos = Z.expand(lo, hi)
            rep = np.repeat(np.arange(qi.size), hi - lo)
            out.append(np.stack([qi[rep], sj[rep], op_of[G.nR + G.s_rel[pos]], g[rep]]))
    if not out:
        return tuple(np.zeros(0, np.int64) for _ in range(4))
    A = np.unique(np.concatenate(out, 1).T, axis=0)
    if A.size and (A[:, 2] < 0).any():
        stop("an S edge's label has no operator")
    return tuple(A[:, c].astype(np.int64) for c in range(4)) if A.size else tuple(np.zeros(0, np.int64) for _ in range(4))


def lists_batch(torch, X, inv_m, specs, pb, seeds, k, nb=NB):
    """Top k node ids (ties to the lower id) of each spec (kind, lambda) for one batch, seeds left out: (L, b, k)."""
    b, S = seeds.shape
    L = len(specs)
    best_s = torch.full((L, b, k), float("-inf"))
    best_i = torch.full((L, b, k), np.iinfo(np.int64).max, dtype=torch.long)
    need = {kd for kd, _ in specs}
    valid = pb["valid"]
    qi, sj = np.nonzero(seeds >= 0)
    sn = seeds[qi, sj]
    o = np.argsort(sn, kind="stable")
    qi, sn = qi[o], sn[o]

    def amax(U, Vt):
        Sc = (U.reshape(b * S, -1) @ Vt).view(b, S, -1)
        Sc = Sc.masked_fill(~valid[:, :, None], float("-inf"))
        return Sc.max(1).values

    with torch.no_grad():
        for b0 in range(0, X.n, nb):
            b1 = min(X.n, b0 + nb)
            Vt = torch.from_numpy(X.rows(slice(b0, b1))).T.contiguous()
            got = {}
            if need & {"trav", "oracle"}:
                im = torch.from_numpy(inv_m[b0:b1])
                got["B"] = (pb["uq"] @ Vt) * im
                Su = (pb["u"].reshape(b * S, -1) @ Vt).view(b, S, -1).masked_fill(~valid[:, :, None], float("-inf"))
                got["trav"] = Su.max(1).values * im
                if "oracle" in need:
                    if pb["or_q"].numel():           # only the seeds whose alpha the oracle changes are scored again
                        Su[pb["or_q"], pb["or_s"]] = pb["u_or"][pb["or_q"], pb["or_s"]] @ Vt
                    got["oracle"] = Su.max(1).values * im
            if "trav0" in need:
                got["B0"] = pb["u0q"] @ Vt
                got["trav0"] = amax(pb["u0"], Vt)
            lo, hi = np.searchsorted(sn, [b0, b1])
            ex_q, ex_n = torch.from_numpy(qi[lo:hi]), torch.from_numpy(sn[lo:hi] - b0)
            ids = torch.arange(b0, b1, dtype=torch.long)
            for li, (kd, lam) in enumerate(specs):
                sc = got[kd] + lam * got["B0" if kd == "trav0" else "B"]
                if ex_q.numel():
                    sc[ex_q, ex_n] = float("-inf")
                cs = torch.cat([best_s[li], sc], 1)
                ci = torch.cat([best_i[li], ids[None, :].expand(b, -1)], 1)
                top = torch.topk(cs, min(k, cs.shape[1]), dim=1)
                best_s[li] = top.values
                best_i[li] = torch.gather(ci, 1, top.indices)
    s_np, i_np = best_s.numpy(), best_i.numpy()
    out = np.empty((L, b, k), np.int64)
    for li in range(L):
        o = np.lexsort((i_np[li], -s_np[li]), axis=1)
        ids_s = np.take_along_axis(i_np[li], o, 1)
        ids_s[~np.isfinite(np.take_along_axis(s_np[li], o, 1))] = -1      # fewer than k nodes (MultiHop-RAG: 609)
        out[li] = ids_s
    return out


def traverse(torch, m1, dirs, X, inv_m, Qn, seeds, specs, k, G=None, op_of=None, flat=None, off=None, qb=QB, nb=NB):
    """All batches: lists (L, nq, k), the mean alpha per operator, the oracle's per-seed picks."""
    nq = Qn.shape[0]
    out = np.empty((len(specs), nq, k), np.int64)
    a_sum = np.zeros(len(dirs))
    a_cnt = 0
    picked = np.full(seeds.shape, -1, np.int64)
    orc = any(kd == "oracle" for kd, _ in specs)
    t0 = time.time()
    for q0 in range(0, nq, qb):
        q1 = min(nq, q0 + qb)
        sd = seeds[q0:q1]
        quads = oracle_quads(G, op_of, sd, flat, off, q0) if orc else None
        pb = plan_batch(torch, m1, dirs, X, Qn[q0:q1], sd, quads)
        out[:, q0:q1] = lists_batch(torch, X, inv_m, specs, pb, sd, k, nb)
        v = sd >= 0
        a_sum += pb["alpha"][v].sum(0)
        a_cnt += int(v.sum())
        if orc:
            picked[q0:q1] = pb["picked"]
        if q0 == 0 or (q0 // qb) % 10 == 0:
            log(f"  traversal {q1:,}/{nq:,} questions ({time.time() - t0:.0f} s)")
    return out, a_sum / max(a_cnt, 1), picked


def with_seeds(lists, seeds):
    """9.3: each list is its question's seeds (hop 0, in seeds_of's order), then the hop-1 nodes in score order."""
    L, nq, k = lists.shape
    out = np.full_like(lists, -1)
    for q in range(nq):
        s = seeds[q][seeds[q] >= 0]
        out[:, q, :s.size] = s
        out[:, q, s.size:] = lists[:, q, :k - s.size]
    return out


def fused(v_top, L, spl, k=20):
    return S1.rrf((v_top, L, spl), k)


def recall(top, flat, off):
    return S1.per_q(top, flat, off)


# ── loading a dataset ────────────────────────────────────────────────────────


def load(a, torch, gates):
    t0 = time.time()
    if a.ds == "multihoprag":
        G, V, inputs, ge = Z.load_mhr(a.arm)
    else:
        G, V, inputs, ge = Z.load_six(a.ds, a.arm, a.host)
    gates["G-E"] = {"families": ge, "verdict": "PASS"}
    X = Z.Vecs(V)
    log(f"{a.ds} {a.arm}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    m1, ops, op_of, r1rec = operators(torch, G, X, a.ds, a.arm, gates)
    log(f"{a.ds} {a.arm}: G-R PASS; {len(ops)} operators ({time.time() - t0:.0f} s)")
    inv_m = mapped_inv(torch, m1, X)
    dirs = [bool(o["directed"]) for o in ops]
    return G, X, m1, ops, op_of, dirs, inv_m, {**inputs, **r1rec}


def check_seeds_out(lists, seeds):
    """Gate G-X: no seed among its own question's hop-1 nodes."""
    for li in range(lists.shape[0]):
        for q in range(seeds.shape[0]):
            s = seeds[q][seeds[q] >= 0]
            if np.isin(lists[li, q], s).any():
                return False
    return True


# ── sel (9.4) ────────────────────────────────────────────────────────────────


def cmd_sel(a):
    torch = Z.torch_()
    t0 = time.time()
    if a.ds not in SELS[a.arm]:
        stop(f"{a.ds} has no selection row for {a.arm}")
    gates = {}
    G, X, m1, ops, op_of, dirs, inv_m, inputs = load(a, torch, gates)
    qd = questions(a.ds, a.arm, "sel", a.host)
    v_top = vlist(qd, X.n)
    seeds = seeds_of(v_top, qd["spl"])
    specs = [(kd, lam) for kd in ("trav", "trav0") for lam in LAMBDAS]
    hop1, amean, _ = traverse(torch, m1, dirs, X, inv_m, qd["Q"], seeds, specs, E.K)
    gates["G-X"] = "PASS" if check_seeds_out(hop1, seeds) else "FAIL"
    if gates["G-X"] != "PASS":
        stop("gate G-X: a seed is in its own list")
    lists = with_seeds(hop1, seeds)
    del hop1
    flat, off = qd["flat"], qd["off"]
    res = {kd: {} for kd in ("trav", "trav0")}
    for li, (kd, lam) in enumerate(specs):
        res[kd][str(lam)] = round(S1.mean(recall(fused(v_top, lists[li], qd["spl"], 5), flat, off)), 6)
    base = S1.rrf((v_top, qd["spl"]))
    ctrl = S1.rrf((v_top, v_top, qd["spl"]))
    rec = {"stage": "SEM2 R2 selection", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "carve": qd["info"],
           "gates": gates, "base5": round(S1.mean(recall(base, flat, off)), 6),
           "ctrl5": round(S1.mean(recall(ctrl, flat, off)), 6), "fused5": res, "lambdas": list(LAMBDAS),
           "seeds_mean": round(float((seeds >= 0).sum(1).mean()), 3),
           "alpha_mean": {o["name"]: round(float(x), 4) for o, x in zip(ops, amean)}, "inputs": inputs,
           "threads": E.THREADS, "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": script_sha()}
    p = OUT / "sel" / a.arm / f"{a.ds}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    Z.wjson(p, rec)
    log(f"{a.ds} {a.arm}: selection written: base {rec['base5']:.4f}, trav {res['trav']}, trav0 {res['trav0']} "
        f"({rec['seconds']} s)")
    return 0


def choose(arm, ds, sha):
    """9.4: per kind, the lambda with the highest mean selection R@5 of RRF(v, list, SPLADE) over the other selection
    datasets (each counts once); ties to the smaller lambda."""
    others = [d for d in SELS[arm] if d != ds]
    recs = {}
    for d in others:
        p = OUT / "sel" / arm / f"{d}.json"
        if not p.is_file():
            stop(f"{arm}: no selection row for {d}")
        r = json.loads(p.read_text(encoding="utf-8"))
        if r["script_sha256"] != sha:
            stop(f"{arm} {d}: the selection was made by other code")
        recs[d] = r
    out = {}
    for kd in ("trav", "trav0"):
        means = {lam: sum(recs[d]["fused5"][kd][str(lam)] for d in others) / len(others) for lam in LAMBDAS}
        best = sorted(LAMBDAS, key=lambda x: (-means[x], x))[0]
        out[kd] = {"lambda": best, "means": {str(k): round(v, 6) for k, v in means.items()}, "over": others}
    return out


# ── read (9.5) ───────────────────────────────────────────────────────────────


def gate_base(a, qd, base, flat, off):
    """Gate G2 (SEM1's): the base RRF(v, SPLADE) is the record's."""
    if a.ds == "multihoprag":
        import zmhr as M
        z = np.load(M.OUT / "first" / "lists.npz")
        qs = M.queries_of(qd["served"])
        allid = [q["query_id"] for q in qs]
        pos = {q: i for i, q in enumerate(allid)}
        ref = z[f"{a.arm}_splade"][[pos[q] for q in qd["ids"]], :5].astype(np.int64)
        same = float(np.mean([set(x.tolist()) == set(y.tolist()) for x, y in zip(base[:, :5], ref)]))
        g = {"rows_equal_share": round(same, 6), "reference": "outputs/mhr1/first/lists.npz"}
        ok = same >= S1.G2_SHARE
    else:
        e0 = np.load(E.OUT / "e0" / a.arm / f"{a.ds}.npz", allow_pickle=False)
        if [str(x) for x in e0["ids"]] != list(qd["ids"]):
            stop(f"gate G2: {a.arm} {a.ds}: E0's ids are not s1eval's, in order")
        hb = E.hits(base[:, :5], flat, off)
        same = float(np.mean(hb == e0["h_rrf5"].astype(np.int64)))
        e0r = json.loads((E.OUT / "e0" / a.arm / f"{a.ds}.json").read_text(encoding="utf-8"))["recall"]["rrf5"]
        b5 = S1.mean(recall(base[:, :5], flat, off))
        g = {"hits_equal_share": round(same, 6), "base5": round(b5, 6), "e0_rrf5": e0r}
        ok = same >= S1.G2_SHARE and abs(b5 - e0r) <= S1.G2_TOL
    g["verdict"] = "PASS" if ok else "FAIL"
    return g


def cmd_read(a):
    torch = Z.torch_()
    t0 = time.time()
    if a.ds not in SETS[a.arm]:
        stop(f"{a.ds} is not read for {a.arm}")
    sha = script_sha()
    if a.lam is not None:
        ch = {kd: {"lambda": a.lam, "smoke": True} for kd in ("trav", "trav0")}
    else:
        ch = choose(a.arm, a.ds, sha)
    lam, lam0 = ch["trav"]["lambda"], ch["trav0"]["lambda"]
    log(f"{a.ds} {a.arm}: chosen lambda trav {lam}, trav0 {lam0}")
    gates = {}
    G, X, m1, ops, op_of, dirs, inv_m, inputs = load(a, torch, gates)
    qd = questions(a.ds, a.arm, "sel" if a.smoke_sel else "read", a.host)
    flat, off = qd["flat"], qd["off"]
    v_top = vlist(qd, X.n)
    base = S1.rrf((v_top, qd["spl"]), 20)
    gates["G2"] = gate_base(a, qd, base, flat, off) if not (a.smoke_sel and a.ds != "multihoprag") else {"verdict": "PASS", "smoke": "no E0 list for s1sel"}
    if gates["G2"]["verdict"] != "PASS":
        stop(f"gate G2: {gates['G2']}")
    seeds = seeds_of(v_top, qd["spl"])
    specs = [("trav", lam), ("trav0", lam0), ("oracle", lam)]
    hop1, amean, picked = traverse(torch, m1, dirs, X, inv_m, qd["Q"], seeds, specs, E.K, G, op_of, flat, off)
    gates["G-X"] = "PASS" if check_seeds_out(hop1, seeds) else "FAIL"
    if gates["G-X"] != "PASS":
        stop("gate G-X: a seed is in its own list")
    lists = with_seeds(hop1, seeds)
    R_hop1 = {f"{kd}_hop1_20": recall(hop1[i][:, :20], flat, off) for i, kd in enumerate(KINDS)}
    del hop1
    spl = qd["spl"]
    F = {kd: fused(v_top, lists[i], spl, 20) for i, (kd, _l) in enumerate(specs)}
    ctrl = S1.rrf((v_top, v_top, spl), 20)
    R = {"base5": recall(base[:, :5], flat, off), "base20": recall(base, flat, off),
         "ctrl5": recall(ctrl[:, :5], flat, off), "ctrl20": recall(ctrl, flat, off),
         "v5": recall(v_top[:, :5], flat, off), "v1000": recall(v_top, flat, off)}
    for kd in KINDS:
        R[f"{kd}_fused5"] = recall(F[kd][:, :5], flat, off)
        R[f"{kd}_fused20"] = recall(F[kd], flat, off)
        R[f"{kd}_alone5"] = recall(lists[KINDS.index(kd)][:, :5], flat, off)
        R[f"{kd}_alone1000"] = recall(lists[KINDS.index(kd)], flat, off)
    R.update(R_hop1)
    reach = (picked >= 0).any(1)
    d = OUT / "read" / a.arm
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / f"{a.ds}.npz", ids=np.asarray(qd["ids"]), flat=flat, off=off, seeds=seeds.astype(np.int32),
             picked=picked.astype(np.int16), **{f"top5_{kd}": F[kd][:, :5].astype(np.int32) for kd in KINDS},
             base5=base[:, :5].astype(np.int32), **{f"r_{k}": v for k, v in R.items()})
    rec = {"stage": "SEM2 R2 read", "declared_in": DECLARED, "dataset": a.ds, "arm": a.arm, "carve": qd["info"],
           "chosen": ch, "gates": gates, "recall": {k: round(S1.mean(v), 6) for k, v in R.items()},
           "oracle_reach": {"questions": int(reach.sum()), "share": round(float(reach.mean()), 6),
                            "seeds_with_a_pick": int((picked >= 0).sum())},
           "seeds_mean": round(float((seeds >= 0).sum(1).mean()), 3),
           "alpha_mean": {o["name"]: round(float(x), 4) for o, x in zip(ops, amean)},
           "operators": [o["name"] for o in ops], "inputs": inputs, "threads": E.THREADS,
           "npz_sha256": Z.sha_file(d / f"{a.ds}.npz"), "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": sha}
    Z.wjson(d / f"{a.ds}.json", rec)
    log(f"{a.ds} {a.arm}: read written {json.dumps(rec['recall'])} ({rec['seconds']} s)")
    return 0


# ── report (9.6) ─────────────────────────────────────────────────────────────


def cmd_report(_a):
    sha = script_sha()
    rows = {}
    for arm in SETS:
        for ds in SETS[arm]:
            p = OUT / "read" / arm / f"{ds}.json"
            if p.is_file():
                r = json.loads(p.read_text(encoding="utf-8"))
                if r["script_sha256"] != sha:
                    stop(f"{arm} {ds}: read by other code")
                z = np.load(OUT / "read" / arm / f"{ds}.npz")
                if Z.sha_file(OUT / "read" / arm / f"{ds}.npz") != r["npz_sha256"]:
                    stop(f"{arm} {ds}: npz sha256 is not its record's")
                rows[(arm, ds)] = (r, z)
    calls = {}
    for (arm, ds), (r, z) in rows.items():
        c = {"T1 trav - base (R@5)": S1.boot(z["r_trav_fused5"], z["r_base5"]),
             "T2 trav - trav0 (R@5)": S1.boot(z["r_trav_fused5"], z["r_trav0_fused5"]),
             "beside": {"trav - SEM1 control (R@5)": S1.boot(z["r_trav_fused5"], z["r_ctrl5"]),
                        "trav0 - base (R@5)": S1.boot(z["r_trav0_fused5"], z["r_base5"]),
                        "T1 at R@20": S1.boot(z["r_trav_fused20"], z["r_base20"]),
                        "T2 at R@20": S1.boot(z["r_trav_fused20"], z["r_trav0_fused20"]),
                        "trav alone - v alone (R@1000)": S1.boot(z["r_trav_alone1000"], z["r_v1000"])},
             "C oracle - trav (R@5)": S1.boot(z["r_oracle_fused5"], z["r_trav_fused5"]),
             "oracle_reach": r["oracle_reach"], "chosen": {k: v["lambda"] for k, v in r["chosen"].items()},
             "recall": r["recall"]}
        c["ADDS"] = c["T1 trav - base (R@5)"]["call"] == "ABOVE" and c["T2 trav - trav0 (R@5)"]["call"] == "ABOVE"
        c["ROOM"] = c["C oracle - trav (R@5)"]["diff"] >= ROOM
        calls[f"{arm}/{ds}"] = c
    e3 = {ds: calls.get(f"e3v/{ds}") for ds in Z.SEVEN}
    have = [ds for ds, c in e3.items() if c]
    if len(have) < len(Z.SEVEN):
        verdict = "INCOMPLETE"
    elif all(e3[ds]["ADDS"] for ds in Z.SEVEN):
        verdict = "HOLDS"
    elif any(e3[ds]["ADDS"] for ds in Z.SEVEN) and not any(e3[ds]["T1 trav - base (R@5)"]["call"] == "BELOW"
                                                           for ds in Z.SEVEN):
        verdict = "PARTIAL"
    else:
        verdict = "FAILS"
    room = [ds for ds in have if e3[ds]["ROOM"]]
    x = {"X12 T1 ABOVE on MuSiQue": None if not e3["musique"] else e3["musique"]["T1 trav - base (R@5)"]["call"] == "ABOVE",
         "X13 T1 not ABOVE on SQuAD": None if not e3["squad"] else e3["squad"]["T1 trav - base (R@5)"]["call"] != "ABOVE",
         "X14 ROOM on MetaQA": None if not e3["metaqa"] else e3["metaqa"]["ROOM"]}
    rep = {"stage": "SEM2 R2 report", "declared_in": DECLARED, "verdict_e3v": verdict, "room_e3v": room,
           "expectations": x, "calls": calls, "script_sha256": sha,
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Z.wjson(OUT / "report.json", rep)
    lines = ["# SEM2 R2 report", "", f"Declared in {DECLARED}. e3v verdict: **{verdict}**; ROOM (C >= {ROOM} pt): "
             f"{', '.join(room) or 'none'}.", "",
             "| arm/dataset | T1 trav - base | T2 trav - trav0 | C oracle - trav | reach | lambda trav/trav0 | ADDS |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for k, c in calls.items():
        lines.append(f"| {k} | {S1.fmt(c['T1 trav - base (R@5)'])} | {S1.fmt(c['T2 trav - trav0 (R@5)'])} | "
                     f"{S1.fmt(c['C oracle - trav (R@5)'])} | {c['oracle_reach']['share']:.3f} | "
                     f"{c['chosen']['trav']}/{c['chosen']['trav0']} | {c['ADDS']} |")
    lines += ["", "Expectations: " + "; ".join(f"{k}: {v}" for k, v in x.items()), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"report: e3v {verdict}; ROOM {room}")
    return 0


# ── self-test (9.7) ──────────────────────────────────────────────────────────


def brute(m1, X, inv_m, pb, seeds, kd, lam, k):
    """One list per question by a full float64 sort over every node (ties to the lower id), seeds left out."""
    V = X.rows(slice(0, X.n)).astype(np.float64)
    out = np.empty((seeds.shape[0], k), np.int64)
    for q in range(seeds.shape[0]):
        ok = seeds[q] >= 0
        if kd == "trav0":
            U, uq, im = pb["u0"][q].numpy()[ok], pb["u0q"][q].numpy(), np.ones(X.n)
        else:
            U = (pb["u"] if kd == "trav" else pb["u_or"])[q].numpy()[ok]
            uq, im = pb["uq"][q].numpy(), inv_m.astype(np.float64)
        sc = (U.astype(np.float64) @ V.T).max(0) * im + lam * (uq.astype(np.float64) @ V.T) * im
        sc[seeds[q][ok]] = -np.inf
        o = np.lexsort((np.arange(X.n), -sc))
        out[q] = o[:k]
    return out


def selftest(a):
    torch = Z.torch_()
    rng = np.random.default_rng(0)
    n, D, Dp, nq, k = 2000, 32, 40, 120, 50
    dirs = [False, False, True, True]
    m1 = Z.make_model(torch, D, Dp, 4, "rotation")
    with torch.no_grad():
        th = rng.uniform(2.0, 3.0, (4, Dp // 2)) * rng.choice([-1.0, 1.0], (4, Dp // 2))   # far from the seed
        m1.theta.copy_(torch.from_numpy(th.astype(np.float32)))
        m1.w.copy_(torch.from_numpy(rng.uniform(0.5, 1.5, (4, Dp // 2)).astype(np.float32)))
    m1.eval()
    V = rng.standard_normal((n, D)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    src = rng.choice(n // 2, nq, replace=False)
    dst = n // 2 + rng.choice(n // 2, nq, replace=False)
    rel = rng.integers(2, 4, nq)
    Q = np.empty((nq, D), np.float32)
    with torch.no_grad():
        xs = m1.embed(torch.from_numpy(V[src]))
        for i in range(nq):
            y = m1.query(xs[i:i + 1], torch.tensor([int(rel[i])]), torch.tensor([True]))[0]
            y = y / y.norm()
            t = y[:D].numpy() + 0.05 * rng.standard_normal(D).astype(np.float32)
            V[dst[i]] = t / np.linalg.norm(t)
            res = (y - (y @ xs[i]) * xs[i])[:D].numpy()
            qv = V[src[i]] + 0.3 * res / max(np.linalg.norm(res), 1e-9)
            Q[i] = qv / np.linalg.norm(qv)
    X = Z.Vecs(V.astype(np.float16))
    inv_m = mapped_inv(torch, m1, X)
    flat, off = S1.flatten([[int(s), int(d)] for s, d in zip(src, dst)])
    v_top, _ = S1.search_mem(Q, X.rows(slice(0, n)), k)
    spl = np.stack([rng.permutation(n)[:k] for _ in range(nq)]).astype(np.int64)
    seeds = seeds_of(v_top, spl)
    keysA = np.unique(np.sort(rng.integers(0, n, (300, 2)), 1) @ np.array([n, 1]))
    keysA = keysA[keysA // n != keysA % n]
    skey = np.unique((src * n + dst) * 2 + (rel - 2))
    G = Z.Graph("toy", n, [("R.a", keysA), ("R.b", np.zeros(0, np.int64))], skey, ("x", "y"))
    op_of = np.asarray([0, 1, 2, 3], np.int64)
    gates = {}
    specs = [("trav", 0.0), ("trav", 0.5), ("trav0", 0.0), ("trav0", 0.5), ("oracle", 0.0)]
    lists, amean, picked = traverse(torch, m1, dirs, X, inv_m, Q, seeds, specs, k, G, op_of, flat, off, qb=32, nb=300)
    quads = oracle_quads(G, op_of, seeds, flat, off, 0)
    pb = plan_batch(torch, m1, dirs, X, Q, seeds, quads)
    agree = []
    for li, (kd, lam) in enumerate(specs):
        bl = brute(m1, X, inv_m, pb, seeds, kd, lam, k)
        agree.append(float(np.mean([np.array_equal(x, y) for x, y in zip(lists[li], bl)])))
    gates["G-T"] = {"agreement": agree, "verdict": "PASS" if min(agree) >= 0.99 else "FAIL"}
    gates["G-X"] = "PASS" if check_seeds_out(lists, seeds) else "FAIL"
    comp = with_seeds(lists, seeds)
    ns = (seeds >= 0).sum(1)
    okc = all(np.array_equal(comp[li, q, :ns[q]], seeds[q][seeds[q] >= 0]) and
              np.array_equal(comp[li, q, ns[q]:], lists[li, q, :k - ns[q]]) for li in range(len(specs)) for q in range(nq))
    gates["G-H0"] = "PASS" if okc else "FAIL"
    # G-0: every operator at its start (D' = D, theta 0, w 1): the traversal is trav0
    m0 = Z.make_model(torch, D, D, 4, "rotation")
    m0.eval()
    inv0 = mapped_inv(torch, m0, X)
    l0, _, _ = traverse(torch, m0, dirs, X, inv0, Q, seeds, [("trav", 0.5), ("trav0", 0.5)], k, qb=32, nb=300)
    g0 = float(np.mean([np.array_equal(x[:20], y[:20]) for x, y in zip(l0[0], l0[1])]))
    gates["G-0"] = {"top20_equal_share": round(g0, 4), "verdict": "PASS" if g0 >= 0.99 else "FAIL"}
    hit = {f"{kd}@{lam}": float(np.mean([dst[i] in lists[li, i, :10] for i in range(nq)])) for li, (kd, lam) in
           enumerate(specs)}
    gates["planted"] = {"tail_in_top10": hit, "oracle_seeds": int((picked >= 0).sum()),
                        "tail_is_a_seed": int(sum(dst[i] in seeds[i] for i in range(nq))),
                        "verdict": "PASS" if hit["trav@0.0"] >= 0.8 and hit["trav@0.0"] > hit["trav0@0.0"]
                        and hit["oracle@0.0"] >= hit["trav@0.0"] else "FAIL"}
    log(f"selftest: {json.dumps(gates)}; alpha mean {np.round(amean, 3).tolist()}")
    bad = [g for g, v in gates.items() if (v if isinstance(v, str) else v["verdict"]) != "PASS"]
    if bad:
        stop(f"self-test failed: {bad}")
    log("selftest PASS")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "sel", "read", "report"))
    p.add_argument("--ds", choices=Z.SEVEN)
    p.add_argument("--arm", choices=Z.ARMS)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--out", help="smoke runs only: write under this folder instead of outputs/sem2/r2")
    p.add_argument("--lam", type=float, help="smoke reads only (needs --out): both lambdas fixed, no selection")
    p.add_argument("--smoke-sel", action="store_true", help="smoke reads only (needs --out): read the selection carve")
    a = p.parse_args(argv)
    global OUT
    if (a.lam is not None or a.smoke_sel) and not a.out:
        stop("--lam is for smoke reads under --out only")
    if a.out:
        OUT = Path(a.out)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not (a.ds and a.arm):
        stop(f"{a.cmd} needs --ds and --arm")
    return cmd_sel(a) if a.cmd == "sel" else cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
