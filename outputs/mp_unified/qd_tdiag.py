"""Design look (untracked; not a result and not filed): why a QD residual fitted on one graph harms another graph,
read from saved models with no refit.

Loads QD models that qd_gnn2.run saved (<out>_models/a<ai>_s<sd>.pt: arm, spec, state_dict, train), loads every graph
in the read role through qd_gnn2.load_all (x1, whose half B is read; webqsp its whole select carve, qd_gnn2.WHOLE), and
reads each model on each graph under read-time counterfactuals:
    base     the model as fitted (checked against the run's own stored ID/np per-row metrics where --runs names them)
    a<x>     s = z + x (s0 - z), x in ALPHAS: the residual scaled (a0 is the twin exactly)
    rstd     s = z + c (r - mean r) / std r on each row, c the model's mean per-row std of r on its training graph's rows:
             the residual's spread matched to what it had where it was fitted (label-free)
    g1       gamma forced to 1 (sigmoid(gamma_raw) -> 1: the mean over active in-neighbours)
    cap<K>   every node keeps at most K in-edges (the K of largest w, ties by edge position) before the forward; cap0 is
             the seeds' own state with no edge
    NR       (label arms) every label 'other', as qd_gnn2 reads it
and records per model and graph the learned gamma and, as row means: the reached share (nodes within L edges of a seed
along u -> v, as the layers pass), in-degree quantiles over all families, mean |r| on reached and on unreached nodes,
corr(r, log1p in-degree) over reached nodes, lift = mean r on golds - mean r on the rest, and ratio = std(r) / std(z).
Only passage-graph-trained models are read: a KB's relation ranks come from its training rows, which a read-role load
does not rank over. Nothing is fitted or selected; nothing is written outside --out.

    python outputs/mp_unified/qd_tdiag.py --models outputs/mp_unified/qd/q3-2w-both_models/a0_s0.pt \
        --runs outputs/mp_unified/qd/q3-2w-both.json --graphs 2wiki,hotpotqa,musique,metaqa,webqsp \
        --out outputs/mp_unified/qd/tdiag_2w.json
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qd_gnn2 as Q2  # noqa: E402  (imports qd_six and qd_gnn, which sets the BLAS thread counts before numpy loads)

import numpy as np  # noqa: E402
import torch  # noqa: E402

QG, Q6 = Q2.QG, Q2.Q6
log, sha = QG.log, QG.sha
ALPHAS = (0.0, 0.25, 0.5, 0.75)
CAPS = (0, 4, 16)
PASSAGE = ("2wiki", "hotpotqa", "musique")
EDGE_KEYS = ("u", "v", "fam", "fwd", "bwd", "w")
STRUCT_KEYS = ("w2_f", "w2_b", "w1_f", "w1_b")


def setup():
    """qd_gnn2.run's preamble: the six-pair 2wiki looks, anchor_univ's pins and binding, the frontier extension."""
    Q6.bind()
    sys.path.insert(0, str(QG.AU_DIR))
    sys.path.insert(0, str(QG.KB_DIR))
    import anchor_univ as AU
    if sha(AU.__file__) != QG.PINS["anchor_univ"]:
        raise SystemExit("anchor_univ.py is not the pinned file")
    AU.check_pins()
    AU.AC.bind()
    for mod, key in ((AU.AW11, "anchor_walk11"), (AU.A16, "l16_look_analyze")):
        if sha(mod.__file__) != QG.PINS[key]:
            raise SystemExit(f"{mod.__file__} is not the pinned file")
    QG.bind_frontier(AU.AW11)
    return AU


# ── per-row helpers ──────────────────────────────────────────────────────────


def reach(q, L):
    r = np.zeros(int(q["n"]), dtype=bool)
    r[q["seeds"][q["seeds"] >= 0]] = True
    for _ in range(L):
        nxt = r.copy()
        nxt[q["v"][r[q["u"]]]] = True
        r = nxt
    return r


def cap_row(q, tok, K):
    """q with every node's in-edges cut to the K of largest w (ties: the earlier edge), and its tokens alike."""
    v, w = q["v"].astype(np.int64), q["w"].astype(np.float64)
    order = np.lexsort((np.arange(v.size), -w, v))
    vs = v[order]
    if vs.size:
        start = np.r_[0, np.flatnonzero(vs[1:] != vs[:-1]) + 1]
        rank = np.arange(vs.size) - np.repeat(start, np.diff(np.r_[start, vs.size]))
    else:
        rank = np.zeros(0, dtype=np.int64)
    keep = np.zeros(v.size, dtype=bool)
    keep[order[rank < K]] = True
    nq = dict(q)
    for k in EDGE_KEYS:
        nq[k] = q[k][keep]
    sm = q["fam"] == 0
    for k in STRUCT_KEYS:
        if k in q:
            nq[k] = q[k][keep[sm]]
    return nq, (tok[0][keep], tok[1][keep])


def row_stats(s, z, q, L):
    r = (s - z).astype(np.float64)
    deg = np.bincount(q["v"], minlength=int(q["n"])).astype(np.float64)
    rc = reach(q, L)
    g = q["gold"] > 0
    out = {"reached": float(rc.mean()), "deg50": float(np.median(deg)), "deg90": float(np.quantile(deg, 0.9)),
           "deg_max": float(deg.max()) if deg.size else 0.0, "edges": int(q["u"].size), "n": int(q["n"]),
           "abs_r_reached": float(np.abs(r[rc]).mean()) if rc.any() else None,
           "abs_r_unreached": float(np.abs(r[~rc]).mean()) if (~rc).any() else None,
           "ratio": float(r.std() / max(float(z.std()), 1e-6)),
           "lift": float(r[g].mean() - r[~g].mean()) if g.any() and (~g).any() else None,
           "corr_r_logdeg": None}
    if rc.sum() > 2:
        a, b = r[rc], np.log1p(deg[rc])
        if a.std() > 1e-9 and b.std() > 1e-9:
            out["corr_r_logdeg"] = float(np.corrcoef(a, b)[0, 1])
    return out


def mean_stats(rows):
    keys = rows[0].keys()
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None]
        out[k] = round(float(np.mean(vals)), 4) if vals else None
    return out


# ── the reads ────────────────────────────────────────────────────────────────


def scored(arm, model, g, rows):
    return Q2.score_rows(arm, model, g, rows)


def record(R, SZ, rows, Q, metrics_of):
    m = Q2.metrics_at(SZ, rows, Q, None, metrics_of)
    rec = R.record(m, by_type=False)
    return Q2.with_abs(rec, R.base()), m


def run(a):
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    t0 = time.time()
    AU = setup()
    A16, AG = AU.A16, AU.AG
    graphs = a.graphs.split(",")
    for g in graphs:
        if g not in Q2.READ:
            raise SystemExit(f"--graphs {g}: one of {list(Q2.READ)}")
    cks = []
    for p in a.models.split(","):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        sp = ck["spec"]
        if sp.get("family") != "qd" or sp.get("w3") or sp.get("w4"):
            raise SystemExit(f"{p}: not a QD arm")
        tg = list(ck["train"])
        if any(t not in PASSAGE for t in tg):
            raise SystemExit(f"{p}: trained on {tg}; only passage-graph-trained models are read here")
        if any(t not in graphs for t in tg):
            raise SystemExit(f"{p}: its training graphs {tg} must be among --graphs (the in-domain reference)")
        cks.append((p, ck))
    max_len = max(int(ck["spec"]["max_len"]) for _p, ck in cks)
    Q, part, G = Q2.load_all({}, graphs, max_len, t0, AU)
    TOK = [QG.edge_labels(q) for q in Q]
    z_of = [A16.zscore(q["score"][:, 0]) for q in Q]
    B = {g: (list(part[(g, Q2.READ[g])]) if g in Q2.WHOLE else part[(g, Q2.READ[g])][1::2]) for g in graphs}
    RB = {g: AG.Reader(Q, B[g], QG.READ_SEED) for g in graphs}
    ref = {}
    for rp in [x for x in (a.runs or "").split(",") if x]:
        rj = json.loads(Path(rp).read_text(encoding="utf-8"))
        npz = np.load(Path(rp).with_suffix(".npz"))
        ref[str(Path(rp).with_suffix(""))] = (rj, {k: npz[k] for k in npz.files})
    res = {"look": "qd_tdiag", "script_sha256": sha(__file__), "pins": {**QG.PINS, "qd_gnn2": sha(Q2.__file__)},
           "graphs": {g: {"rows": len(B[g]), "B": RB[g].base(), "info": G[g]["info"]} for g in graphs},
           "max_len": max_len, "alphas": list(ALPHAS), "caps": list(CAPS), "threads": a.threads, "models": {}}
    out_path = Path(a.out)

    def save():
        out_path.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")

    for p, ck in cks:
        t1 = time.time()
        sp, name = ck["spec"], ck["arm"]
        n_id = sp["K"] if sp["kind"] == "ID" else 0
        arm = QG.QDArm(sp, Q, TOK, G, z_of, A16, n_id)
        model = arm.make()
        model.load_state_dict(ck["state_dict"])
        model.eval()
        L = int(sp["L"])
        gamma = float(torch.sigmoid(model.gamma_raw).item())
        g1 = copy.deepcopy(model)
        with torch.no_grad():
            g1.gamma_raw.fill_(40.0)
        tg = list(ck["train"])
        ent = {"arm": name, "spec": {k: v for k, v in sp.items() if k != "seeds"}, "train": ck["train"], "gamma": round(gamma, 4),
               "margin": str(ck.get("margin")), "by_graph": {}}
        # the run's stored per-row ID/np reads, to check the load (model file a<ai>_s<sd>.pt of run <stem>_models)
        mdir = Path(p).parent
        stem = str(mdir.parent / mdir.name[:-len("_models")]) if mdir.name.endswith("_models") else None
        ai_sd = Path(p).stem[1:].split("_s")
        own = ref.get(stem)
        base_SZ = {}
        for g in graphs:
            arm.prepare(model, g)
            SZ = scored(arm, model, g, B[g])
            base_SZ[g] = SZ
        # c: the residual's mean per-row std on the training graph(s)
        c = float(np.mean([float(np.std(s - z)) for g in tg for (s, z) in base_SZ[g]]))
        ent["c_train_std"] = round(c, 4)
        for g in graphs:
            t2 = time.time()
            R = RB[g]
            reads, chk = {}, None
            rows = B[g]
            SZ = base_SZ[g]
            rec, m0 = record(R, SZ, rows, Q, A16.metrics_of)
            reads["base"] = rec
            if own is not None:
                key = f"{ai_sd[0]}_{ai_sd[1]}_{g}_ID_np"
                if key in own[1]:
                    d = np.abs(own[1][key] - m0)
                    chk = {"key": key, "max_abs_diff": float(d.max()), "rows_differ": int((d.max(1) > 1e-9).sum())}
            for x in ALPHAS:
                reads[f"a{x}"] = record(R, [(z + x * (s - z), z) for s, z in SZ], rows, Q, A16.metrics_of)[0]
            rs = []
            for s, z in SZ:
                r = s - z
                sd_ = float(r.std())
                rs.append((z + (c * (r - r.mean()) / sd_ if sd_ > 1e-9 else 0.0 * r), z))
            reads["rstd"] = record(R, rs, rows, Q, A16.metrics_of)[0]
            arm.prepare(g1, g)
            reads["g1"] = record(R, scored(arm, g1, g, rows), rows, Q, A16.metrics_of)[0]
            arm.prepare(model, g)
            q_save, t_save = arm.Q, arm.TOK
            for K in CAPS:
                Qc, Tc = list(Q), list(TOK)
                for i in rows:
                    Qc[i], Tc[i] = cap_row(Q[i], TOK[i], K)
                arm.Q, arm.TOK = Qc, Tc
                try:
                    reads[f"cap{K}"] = record(R, scored(arm, model, g, rows), rows, Qc, A16.metrics_of)[0]
                finally:
                    arm.Q, arm.TOK = q_save, t_save
                del Qc, Tc
            if sp["kind"] != "T0":
                arm.nr = True
                reads["NR"] = record(R, scored(arm, model, g, rows), rows, Q, A16.metrics_of)[0]
                arm.nr = False
            st = mean_stats([row_stats(s, z, Q[i], L) for (s, z), i in zip(SZ, rows)])
            ent["by_graph"][g] = {"role": "train" if g in tg else "read", "reads": reads, "stats": st, "check": chk,
                                  "seconds": round(time.time() - t2, 1)}
            log(f"{name} [{','.join(tg)}] on {g}: gamma {gamma:.3f} " + "; ".join(
                f"{k} rho {v['rho (R@5, FC@5, hit@1)']} abs {v['abs (R@5, FC@5, hit@1)']}" for k, v in reads.items())
                + f"; stats {st}; check {chk}")
        ent["seconds"] = round(time.time() - t1, 1)
        res["models"][p] = ent
        save()
    res["seconds"] = round(time.time() - t0, 1)
    save()
    log(f"done in {res['seconds']}s")


# ── selftest ─────────────────────────────────────────────────────────────────


def selftest():
    rng = np.random.default_rng(0)
    Q = QG.synthetic_rows(6, rng, 8)
    for q in Q:
        tok = QG.edge_labels(q)
        for K in (0, 1, 3):
            nq, nt = cap_row(q, tok, K)
            deg = np.bincount(nq["v"], minlength=q["n"])
            want = np.minimum(np.bincount(q["v"], minlength=q["n"]), K)
            assert np.array_equal(deg, want), (K, deg, want)
            assert nt[0].size == nq["u"].size == nq["w"].size
            sm = nq["fam"] == 0
            assert nq["w2_f"].size == int(sm.sum())
            # every kept edge's in-edge rank by w is below K
            for vv in np.unique(nq["v"]):
                kept = np.sort(nq["w"][nq["v"] == vv])[::-1]
                allw = np.sort(q["w"][q["v"] == vv])[::-1]
                assert np.allclose(kept, allw[:kept.size])
        r = reach(q, 0)
        assert r.sum() == np.unique(q["seeds"][q["seeds"] >= 0]).size
        r2 = reach(q, 2)
        assert r2.sum() >= reach(q, 1).sum() >= r.sum()
        z = rng.standard_normal(q["n"])
        st = row_stats(z + 1.0 * q["gold"], z, q, 2)
        assert abs(st["lift"] - 1.0) < 1e-12, st
    print("selftest: cap_row keeps the top-K in-edges by w with tokens and structural tables aligned; reach and stats ok")


def main(argv=None):
    if "--selftest" in (argv or sys.argv):
        selftest()
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--graphs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
