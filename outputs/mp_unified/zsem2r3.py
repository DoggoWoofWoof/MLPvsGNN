"""SEM2 R3(c): an operator for a relation never seen, from its name alone (docs/SEM2_EDGE_OPERATORS.md section 11).

    python outputs/mp_unified/zsem2r3.py selftest
    python outputs/mp_unified/zsem2r3.py names  --ds metaqa|webqsp --host            (encode the relation names: e3v, host)
    python outputs/mp_unified/zsem2r3.py fit    --ds metaqa --fold 0|1|2 --host --threads 8
    python outputs/mp_unified/zsem2r3.py fit    --ds webqsp --fold 0 --host --threads 8
    python outputs/mp_unified/zsem2r3.py report                                     (laptop: the calls, from the fit records)

A wrapper over zsem2.py (R1): it imports its graph, split, sampler, filtered rank and measures and edits none of them.
- names: each relation's name string (11.4) through the e3v encoder as a document, G-N against 64 stored node vectors.
- fit: relations in train / dev / test by 11.5; one shared map P and a small net g(name vector) -> D'/2 angles and D'/2 log plane weights;
  the last layer starts at zero (every score = cosine). Trained on train relations' train edges, stopped on dev relations' dev edges.
  Read on the test relations' test edges (up to 200 each): cosine, the name net, the mean-name operator, the shuffled-name operator.
  A checkpoint at every epoch and after every chunk of ranks: a kill costs at most one epoch or one chunk.
- report: the calls of 11.7 (a bootstrap over relations).
Outputs: outputs/sem2/r3/<ds>/f<fold>.json and .npz, outputs/sem2/r3/names/<ds>.npz, outputs/sem2/r3/report.json and report.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 11"
OUT = Z.ROOT / "outputs" / "sem2" / "r3"
ROWS = ("metaqa", "webqsp")
FOLDS = {"metaqa": 3, "webqsp": 1}
CFGR = dict(Z.CFG, hidden=256, n_read=200, chunk=4000)
BOOT, SEED = Z.BOOT, Z.SEED
SCORERS = ("cosine", "name", "mean", "shuffled")
G_N_COS = 0.999


def script_sha():
    return Z.sha_file(Path(__file__))


# ── relations: strings, roles ───────────────────────────────────────────────


def name_string(label):
    """11.4: the label's name without its family prefix, lower case, '_' and '.' read as spaces."""
    s = label.split(".", 1)[1] if label[:2] in ("S.", "R.") else label
    return " ".join(s.lower().replace("_", " ").replace(".", " ").split())


def u_of(ds, label):
    return int.from_bytes(hashlib.sha256(f"sem2r3|{ds}|{label}".encode()).digest()[:8], "big") / 2 ** 64


def assign(ds, eligible, fold):
    """11.5: the eligible S labels (an operator of their own) -> test / dev / train."""
    u = {n: u_of(ds, n) for n in eligible}
    role = {}
    if ds in ("webqsp", "toy"):
        if fold != 0:
            stop(f"{ds} has one fold")
        for n in eligible:
            role[n] = "test" if u[n] < 0.20 else ("dev" if u[n] < 0.30 else "train")
    elif ds == "metaqa":
        order = sorted(eligible, key=lambda n: (u[n], n))
        test = [n for i, n in enumerate(order) if i % FOLDS[ds] == fold]
        rest = [n for n in order if n not in test]
        if not test or len(rest) < 2:
            stop(f"{ds} fold {fold}: {len(test)} test, {len(rest)} other relations")
        dev = min(rest, key=lambda n: (u[n], n))
        for n in eligible:
            role[n] = "test" if n in test else ("dev" if n == dev else "train")
    else:
        stop(f"no hold-out rule for {ds}")
    return role


def own_labels(labs):
    """The labels with an operator of their own (7.2): [(L, name)] in label order. S.rare's relations are not among them."""
    return [(L, v["name"]) for L, v in sorted(labs.items()) if v.get("op") == v["name"]]


def make_group(G, L, name):
    if L < G.nR:
        return Z.Group(G, name, lab=L)
    return Z.Group(G, name, gi=np.flatnonzero(G.s_rel == L - G.nR))


# ── the model ───────────────────────────────────────────────────────────────


def make_net(torch, D, Dp, Dn, hid):
    nn = torch.nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            P = torch.zeros(Dp, D)
            P[:D, :D] = torch.eye(D)
            if Dp > D:
                g = torch.Generator().manual_seed(Z.SEED)
                P[D:] = Z.EXTRA_INIT * torch.randn(Dp - D, D, generator=g)
            self.P = nn.Parameter(P)
            torch.manual_seed(Z.SEED)
            self.g = nn.Sequential(nn.Linear(Dn, hid), nn.Tanh(), nn.Linear(hid, Dp))     # D'/2 angles, D'/2 log plane weights
            nn.init.zeros_(self.g[2].weight)
            nn.init.zeros_(self.g[2].bias)
            self.H = Dp // 2

        def embed(self, v):
            x = v @ self.P.T
            return x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)

        def query(self, x, op, directed, table):
            m = x.shape[0]
            u, inv = torch.unique(op, return_inverse=True)
            out = self.g(table[u])[inv]
            th, lw = out[:, :self.H], out[:, self.H:]
            c, s = torch.cos(th), torch.sin(th)
            x2 = x.view(m, self.H, 2)
            x0, x1 = x2[..., 0], x2[..., 1]
            rot = torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).view(m, 2 * self.H)
            sym = (x2 * torch.exp(lw).unsqueeze(-1)).view(m, 2 * self.H)
            return torch.where(directed[:, None], rot, sym)

    if Dp % 2:
        stop(f"D' = {Dp} is odd")
    return Net()


class NameScorer:
    """zsem2.rank_read's scorer: embed and query, the operator of an edge's relation from the name-table row of op."""

    def __init__(self, model, table):
        self.model, self.table = model, table

    def embed(self, v):
        return self.model.embed(v)

    def query(self, x, op, directed):
        return self.model.query(x, op, directed, self.table)


def derange(idx, seed):
    rng = np.random.default_rng(seed)
    idx = np.asarray(idx)
    if idx.size < 2:
        return idx.copy()
    while True:
        p = rng.permutation(idx.size)
        if not (p == np.arange(idx.size)).any():
            return idx[p]


# ── ranks in chunks, with a checkpoint per chunk ────────────────────────────


def subsample(smp, idx):
    out = {k: smp[k][idx] for k in ("s", "d", "lab", "grp")}
    out["nR"] = smp["nR"]
    return out


def chunked(torch, G, X, sc, smp, op_of, ckdir, tag, chunk, key):
    n = smp["s"].size
    parts, t0 = [], time.time()
    for i, lo in enumerate(range(0, n, chunk)):
        p = ckdir / f"rank_{key}_{tag}_{i:04d}.npy" if ckdir else None
        if p is not None and p.is_file():
            parts.append(np.load(p))
            continue
        sub = {k: smp[k][lo:lo + chunk] for k in ("s", "d", "lab", "grp")}
        sub["nR"] = smp["nR"]
        r, _ = Z.rank_read(torch, sc, X, sub, Z.exclusions(G, sub), op_of)
        parts.append(r)
        if p is not None:
            tmp = p.with_name(p.name + ".tmp.npy")
            np.save(tmp, r)
            os.replace(tmp, p)
        log(f"  {tag}: {min(lo + chunk, n):,}/{n:,} edges ({time.time() - t0:.0f} s)")
    return np.concatenate(parts) if parts else np.zeros(0)


# ── the fit and the read of one fold ────────────────────────────────────────


def run_fold(torch, G, X, vec_of, cfg, ds, fold, ckdir, gates):
    t0 = time.time()
    split = Z.node_split(G.name, G.n)
    labs, _ops, _op0, _g = Z.plan(G, split, cfg)
    own = own_labels(labs)
    table_names = [nm for _L, nm in own]
    missing = [nm for nm in table_names if nm not in vec_of]
    if missing:
        stop(f"no name vector for {missing[:5]} ({len(missing)})")
    Dn = len(next(iter(vec_of.values())))
    N = torch.from_numpy(np.stack([vec_of[nm] for nm in table_names]).astype(np.float32))
    op_of = np.full(G.nR + G.RS, -1, np.int64)
    for k, (L, _nm) in enumerate(own):
        op_of[L] = k
    eligible = [nm for L, nm in own if L >= G.nR]
    role = assign(ds, eligible, fold)
    kind = {nm: role.get(nm, "train") for _L, nm in own}
    tr_g = [make_group(G, L, nm) for L, nm in own if kind[nm] == "train"]
    dv_g = [make_group(G, L, nm) for L, nm in own if kind[nm] == "dev"]
    te_g = [make_group(G, L, nm) for L, nm in own if kind[nm] == "test"]
    log(f"{ds} fold {fold}: {len(own)} relations: train {len(tr_g)}, dev {len(dv_g)} {[g.name for g in dv_g][:3]}, "
        f"test {len(te_g)} {[g.name for g in te_g][:4]}")
    if not (tr_g and dv_g and te_g):
        stop("a group of relations is empty")
    tr_names = {g.name for g in tr_g}
    tr_strings = {name_string(nm) for nm in tr_names}
    clash = sorted(g.name for g in dv_g + te_g if name_string(g.name) in tr_strings)
    gs = {"dev and test name strings are not train name strings": not clash}
    if clash:
        stop(f"gate G-S: name strings of held-out relations equal a train relation's: {clash[:5]}")
    train_nodes = np.flatnonzero(split == Z.TRAIN)
    rng = np.random.default_rng(SEED)
    dev = Z.take(dv_g, split, Z.is_dev, cfg["n_dev"], rng)
    test = Z.take(te_g, split, Z.is_test, cfg["n_read"], rng)
    for smp in (dev, test):
        smp["nR"] = G.nR
    gs["test sources are test nodes"] = bool((split[test["s"]] == Z.TEST).all())
    gs["dev sources are dev nodes, targets not test"] = bool(((split[dev["s"]] == Z.DEV) & (split[dev["d"]] != Z.TEST)).all())
    dev_ex = Z.exclusions(G, dev)
    log(f"{ds} fold {fold}: dev {dev['s'].size:,} edges, test {test['s'].size:,} edges ({time.time() - t0:.0f} s)")
    cos = Z.Cosine()
    cos_dev = Z.rank_read(torch, cos, X, dev, dev_ex, op_of)
    D = X.D
    Dp = D + cfg["extra"]
    model = make_net(torch, D, Dp, Dn, cfg["hidden"])
    sc = NameScorer(model, N)
    r0, t0s = Z.rank_read(torch, sc, X, dev, dev_ex, op_of)
    g0 = {"max_score_diff": float(np.abs(t0s - cos_dev[1]).max()), "mrr_diff": float(abs((1.0 / r0).mean() - (1.0 / cos_dev[0]).mean()))}
    g0["verdict"] = "PASS" if g0["max_score_diff"] <= Z.G0_SCORE and g0["mrr_diff"] <= Z.G0_MRR else "FAIL"
    # 11.9 reads G-0 on the test relations: the same check on up to 300 test edges (evenly spaced), before any step
    sel = np.unique(np.linspace(0, test["s"].size - 1, min(300, test["s"].size)).astype(np.int64))
    tsub = subsample(test, sel)
    tex = Z.exclusions(G, tsub)
    rc, _ = Z.rank_read(torch, cos, X, tsub, tex, op_of)
    rn, _ = Z.rank_read(torch, sc, X, tsub, tex, op_of)
    same = float((np.abs(rc - rn) < 1e-9).mean())
    g0["test_edges_checked"] = int(sel.size)
    g0["test_rank_equal_share"] = round(same, 4)
    if same < 0.99:
        g0["verdict"] = "FAIL"
    gates["G-0"] = g0
    if g0["verdict"] != "PASS":
        stop(f"gate G-0: {g0}")
    key = hashlib.sha256(json.dumps({"ds": ds, "fold": fold, "cfg": cfg, "roles": sorted(kind.items()), "code": script_sha(),
                                     "n": int(G.n), "zsem2": Z.sha_file(Path(Z.__file__))}, sort_keys=True).encode()).hexdigest()[:16]
    ck = (ckdir / "state.pt") if ckdir else None
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    best = {"epoch": 0, "dev_mrr": Z.measures(r0)["mrr"], "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
    hist = [{"epoch": 0, "dev_mrr": best["dev_mrr"]}]
    since, ep0, resumed = 0, 1, None
    if ck is not None and ck.is_file():
        try:
            st = torch.load(ck, weights_only=False)
            if st.get("key") == key:
                model.load_state_dict(st["model"])
                opt.load_state_dict(st["opt"])
                best, hist, since, ep0, resumed = st["best"], st["hist"], st["since"], st["epoch"] + 1, st["epoch"]
                log(f"{ds} fold {fold}: resumed after epoch {resumed}")
        except Exception as ex:  # noqa: BLE001
            log(f"{ds} fold {fold}: checkpoint not read ({ex}); the fit starts at epoch 1")
    share = cfg["epoch"] // max(len(tr_g), 1)
    gs_ok = True
    done = resumed is not None and since >= cfg["patience"]
    for ep in range(ep0, cfg["max_epochs"] + 1):
        if done:
            break
        te = time.time()
        er = np.random.default_rng(ep)
        tr = Z.take(tr_g, split, Z.is_train, share, er)
        m = tr["s"].size
        gs_ok &= bool(((split[tr["s"]] == Z.TRAIN) & (split[tr["d"]] == Z.TRAIN)).all())
        gs_ok &= bool(np.isin(tr["lab"], [L for L, nm in own if nm in tr_names]).all())
        perm = er.permutation(m)
        tot, nb = 0.0, 0
        model.train()
        for b0 in range(0, m, cfg["batch"]):
            I = perm[b0:b0 + cfg["batch"]]
            s, d, lab = tr["s"][I], tr["d"][I], tr["lab"][I]
            neg = train_nodes[er.integers(0, train_nodes.size, cfg["neg"])]
            cand = np.concatenate([d, neg])
            mask = Z.true_mask(G, s, lab, cand)
            mask[np.arange(I.size), np.arange(I.size)] = False
            xs = model.embed(torch.from_numpy(X.rows(s)))
            xc = model.embed(torch.from_numpy(X.rows(cand)))
            q = model.query(xs, torch.from_numpy(op_of[lab]), torch.from_numpy(lab >= G.nR), N)
            logits = (q @ xc.T) / cfg["tau"]
            logits = logits.masked_fill(torch.from_numpy(mask), float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(I.size))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        model.eval()
        r, _ = Z.rank_read(torch, sc, X, dev, dev_ex, op_of)
        mrr = Z.measures(r)["mrr"]
        hist.append({"epoch": ep, "edges": int(m), "loss": round(tot / max(nb, 1), 4), "dev_mrr": mrr, "seconds": round(time.time() - te, 1)})
        log(f"{ds} fold {fold}: epoch {ep}: {m:,} edges, loss {tot / max(nb, 1):.4f}, dev-relation MRR {mrr} ({time.time() - te:.0f} s)")
        if mrr > best["dev_mrr"]:
            best = {"epoch": ep, "dev_mrr": mrr, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            since = 0
        else:
            since += 1
        if ck is not None:
            ckdir.mkdir(parents=True, exist_ok=True)
            tmp = ck.with_name("state.pt.tmp")
            torch.save({"key": key, "model": model.state_dict(), "opt": opt.state_dict(), "best": best, "hist": hist, "since": since,
                        "epoch": ep}, tmp)
            os.replace(tmp, ck)
        if since >= cfg["patience"]:
            break
    gates["G-S training"] = "PASS" if gs_ok else "FAIL"
    gates["G-S names"] = "PASS" if gs["dev and test name strings are not train name strings"] else "FAIL"
    gates["G-S samples"] = "PASS" if gs["test sources are test nodes"] and gs["dev sources are dev nodes, targets not test"] else "FAIL"
    for k in ("G-S training", "G-S names", "G-S samples"):
        if gates[k] != "PASS":
            stop(f"gate {k}")
    model.load_state_dict(best["state"])
    model.eval()
    # the read: cosine, the name net, the mean-name operator, the shuffled name
    tr_idx = [k for k, (_L, nm) in enumerate(own) if kind[nm] == "train"]
    te_idx = [k for k, (_L, nm) in enumerate(own) if kind[nm] == "test"]
    Nmean = N.clone()
    Nmean[:] = N[tr_idx].mean(0, keepdim=True)
    Nshuf = N.clone()
    Nshuf[te_idx] = N[derange(te_idx, SEED)]
    scs = {"cosine": cos, "name": NameScorer(model, N), "mean": NameScorer(model, Nmean), "shuffled": NameScorer(model, Nshuf)}
    ranks = {}
    for tag in SCORERS:
        ranks[tag] = chunked(torch, G, X, scs[tag], test, op_of, ckdir, tag, cfg["chunk"], key)
        log(f"{ds} fold {fold}: {tag} test {Z.measures(ranks[tag])} ({time.time() - t0:.0f} s)")
    names = [g.name for g in te_g]
    per = {}
    for g, nm in enumerate(names):
        m = test["grp"] == g
        per[nm] = {"edges": int(m.sum()), **{f"mrr_{t}": round(100 * float((1.0 / ranks[t][m]).mean()), 3) for t in SCORERS if m.any()}}
        per[nm].update({f"h10_{t}": round(100 * float((ranks[t][m] <= 10).mean()), 3) for t in SCORERS if m.any()})
        per[nm].update({f"h100_{t}": round(100 * float((ranks[t][m] <= 100).mean()), 3) for t in SCORERS if m.any()})
    rec = {"relations": {"train": sorted(tr_names), "dev": [g.name for g in dv_g], "test": names},
           "n_train_relations": len(tr_g), "fit": {"history": hist, "best_epoch": best["epoch"], "best_dev_mrr": best["dev_mrr"],
                                                   "last_epoch": hist[-1]["epoch"],
                                                   "ended_by": "patience" if since >= cfg["patience"] else "max_epochs",
                                                   "best_is_last_epoch": bool(best["epoch"] == hist[-1]["epoch"]),
                                                   "resumed_after_epoch": resumed, "D_prime": Dp, "hidden": cfg["hidden"]},
           "per_relation": per, "test_edges": int(test["s"].size), "dev_edges": int(dev["s"].size),
           "pooled": {t: Z.measures(ranks[t]) for t in SCORERS}}
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              "test_names": np.asarray(names), **{f"rank_{t}": ranks[t] for t in SCORERS},
              **{f"model.{k}": v.detach().numpy() for k, v in model.state_dict().items()}}
    return rec, arrays


# ── names (11.4) ────────────────────────────────────────────────────────────


def encode_names(strings, dev):
    import zenc3v as V
    vs = []
    for e in V.PARTS:
        rev = E.pinned(e)
        m = E.load_model(e, dev, rev)
        if m.default_prompt_name:
            stop(f"{e}: a default prompt would prefix the documents")
        v, _st = E.encode_texts(m, e, strings, "docs")
        if E.damaged(v):
            stop(f"{e}: damaged rows")
        vs.append(v)
        del m
    X = V.fuse(vs)
    return np.asarray(X, np.float32)


def cmd_names(a):
    import u1b
    import zul2 as U2
    t0 = time.time()
    torch = Z.torch_()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    served, _freeze = u1b.package_root(a.host)
    man = json.loads((served / a.ds / "DATASET.json").read_text(encoding="utf-8"))
    n = int(man["nodes"]["n"])
    f = json.loads((served / a.ds / "graph" / "GRAPH_MANIFEST.json").read_text(encoding="utf-8"))["families"]["structural"]
    vocab = [str(v) for v in (f.get("relation_vocabulary") or [])]
    labels = ["R.mention", "R.name_comention", "R.title_comention"] + ["S." + v for v in vocab]
    strings = [name_string(x) for x in labels]
    log(f"{a.ds}: {len(labels)} labels, device {dev} ({strings[:2]} ... {strings[-1]!r})")
    Xn = encode_names(strings, dev)
    Xn /= np.linalg.norm(Xn, axis=1, keepdims=True)
    # G-N: the same function reproduces stored node vectors from their texts
    V, _vrec = U2.load_rows(served, "e3v", a.ds, n)
    rows = np.arange(64)
    texts = E.doc_texts(E.served_root(), a.ds, rows)
    Xt = encode_names([texts[int(r)] for r in rows], dev)
    Xs = np.asarray(V[rows], np.float32)
    cs = (Xt * Xs).sum(1) / (np.linalg.norm(Xt, axis=1) * np.linalg.norm(Xs, axis=1))
    gn = {"nodes": 64, "min_cosine": round(float(cs.min()), 6), "device": dev, "distinct_strings": len(set(strings)), "labels": len(labels),
          "verdict": "PASS" if float(cs.min()) >= G_N_COS else "FAIL"}
    log(f"{a.ds}: G-N {gn}")
    if gn["verdict"] != "PASS":
        stop(f"gate G-N: {gn}")
    (OUT / "names").mkdir(parents=True, exist_ok=True)
    p = OUT / "names" / f"{a.ds}.npz"
    np.savez(p, labels=np.asarray(labels), strings=np.asarray(strings), vectors=Xn.astype(np.float32))
    Z.wjson(OUT / "names" / f"{a.ds}.json", {"stage": "SEM2 R3 names", "declared_in": DECLARED, "dataset": a.ds, "labels": len(labels),
                                            "G-N": gn, "npz_sha256": Z.sha_file(p), "script_sha256": script_sha(),
                                            "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    return 0


def load_names(ds):
    p = OUT / "names" / f"{ds}.npz"
    if not p.is_file():
        stop(f"no names file for {ds}: run `names` first")
    rec = json.loads((OUT / "names" / f"{ds}.json").read_text(encoding="utf-8"))
    if Z.sha_file(p) != rec["npz_sha256"]:
        stop("names npz sha256 is not its record's")
    z = np.load(p)
    return {str(k): v for k, v in zip(z["labels"], z["vectors"])}, rec


def cmd_fit(a):
    torch = Z.torch_()
    t0 = time.time()
    if a.ds not in ROWS or not 0 <= a.fold < FOLDS[a.ds]:
        stop(f"{a.ds} fold {a.fold} is not a row")
    vec_of, nrec = load_names(a.ds)
    G, V, inputs, ge = Z.load_six(a.ds, "e3v", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}, "G-N": nrec["G-N"]}
    X = Z.Vecs(V)
    log(f"{a.ds}: {G.n:,} nodes, rows {V.shape} ({time.time() - t0:.0f} s)")
    cfg = dict(CFGR, **Z.SMOKE, n_read=20, chunk=500) if a.smoke else CFGR
    od = (OUT / "smoke" / a.ds) if a.smoke else (OUT / a.ds)
    ck = od / "ckpt" / f"f{a.fold}"
    rec, arrays = run_fold(torch, G, X, vec_of, cfg, a.ds, a.fold, ck, gates)
    rec = {"stage": "SEM2 R3(c) fit and read" + (" (SMOKE: a code check, not a read)" if a.smoke else ""), "declared_in": DECLARED,
           "dataset": a.ds, "arm": "e3v", "fold": a.fold, "smoke": bool(a.smoke),
           "config": cfg, "gates": gates, **rec, "inputs": {**inputs, "names_npz_sha256": nrec["npz_sha256"]}, "threads": E.THREADS,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": script_sha()}
    od.mkdir(parents=True, exist_ok=True)
    np.savez(od / f"f{a.fold}.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / f"f{a.fold}.npz")
    Z.wjson(od / f"f{a.fold}.json", rec)
    for p in sorted(ck.glob("*")):          # clean-up: the checkpoint folder is this row's own temporary output
        p.unlink()
    ck.rmdir()
    log(f"{a.ds} fold {a.fold}: written {od / f'f{a.fold}.json'} ({rec['seconds']} s); pooled {rec['pooled']}")
    return 0


# ── the calls (11.7) ────────────────────────────────────────────────────────


def calls_from(per):
    """per: {relation: {mrr_cosine, mrr_name, mrr_mean, mrr_shuffled, edges}} -> the dataset's calls (bootstrap over relations)."""
    import zap1 as ZA
    names = sorted(r for r, v in per.items() if v["edges"] > 0)
    arr = {t: np.asarray([per[r][f"mrr_{t}"] for r in names], np.float64) / 100.0 for t in SCORERS}
    n = len(names)

    def boot(x, y):
        dl, lo, hi = ZA.boot_diff(x, y, np.arange(n), n, BOOT, SEED)
        return {"diff": ZA.pct(dl), "lo": ZA.pct(lo), "hi": ZA.pct(hi), "call": ZA.call(lo, hi)}
    d = boot(arr["name"], arr["cosine"])
    e = boot(arr["name"], arr["mean"])
    out = {"relations": n, "Delta name - cosine": d, "Gain from the name (name - mean name)": e,
           "beside": {"mean name - cosine": boot(arr["mean"], arr["cosine"]), "name - shuffled name": boot(arr["name"], arr["shuffled"]),
                      "shuffled name - cosine": boot(arr["shuffled"], arr["cosine"]),
                      "relations with d_r > 0": int((arr["name"] > arr["cosine"]).sum()),
                      "mean MRR (points)": {t: round(100 * float(arr[t].mean()), 3) for t in SCORERS}}}
    if d["call"] == "ABOVE":
        out["call"] = "TRANSFERS" if e["call"] == "ABOVE" else "GENERIC"
    else:
        out["call"] = "LOSES" if d["call"] == "BELOW" else "NO"
    return out


def fmt(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def cmd_report(_a):
    sha = script_sha()
    calls, per_ds, recs = {}, {}, {}
    for ds in ROWS:
        per = {}
        for f in range(FOLDS[ds]):
            p = OUT / ds / f"f{f}.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["script_sha256"] != sha:
                stop(f"{ds} fold {f}: fitted by other code")
            if Z.sha_file(OUT / ds / f"f{f}.npz") != r["npz_sha256"]:
                stop(f"{ds} fold {f}: npz sha256 is not its record's")
            per.update(r["per_relation"])
            recs.setdefault(ds, []).append(r)
        if len(recs.get(ds, [])) == FOLDS[ds]:
            calls[ds] = calls_from(per)
        per_ds[ds] = per
    if len(calls) < len(ROWS):
        verdict = "INCOMPLETE"
    else:
        n_t = sum(c["call"] == "TRANSFERS" for c in calls.values())
        verdict = "HOLDS" if n_t == len(ROWS) else ("PARTIAL" if n_t >= 1 else "FAILS")
    x = {}
    if "webqsp" in calls:
        w = calls["webqsp"]
        x = {"X18 WebQSP Delta ABOVE": w["Delta name - cosine"]["call"] == "ABOVE",
             "X19 WebQSP gain from the name ABOVE": w["Gain from the name (name - mean name)"]["call"] == "ABOVE",
             "X21 WebQSP shuffled name below the name net": w["beside"]["name - shuffled name"]["call"] == "ABOVE"}
    if "metaqa" in calls:
        x["X20 MetaQA Delta not ABOVE"] = calls["metaqa"]["Delta name - cosine"]["call"] != "ABOVE"
    rep = {"stage": "SEM2 R3(c) report", "declared_in": DECLARED, "verdict": verdict, "calls": calls, "expectations": x,
           "relations": {ds: {k: v for k, v in per.items()} for ds, per in per_ds.items()},
           "fits": {ds: [{"fold": r["fold"], "best_epoch": r["fit"]["best_epoch"], "best_dev_mrr": r["fit"]["best_dev_mrr"],
                          "train": r["n_train_relations"], "test": len(r["relations"]["test"]), "gates": r["gates"]} for r in rs]
                    for ds, rs in recs.items()},
           "script_sha256": sha, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / "report.json", rep)
    lines = ["# SEM2 R3(c) report", "", f"Declared in {DECLARED}. Verdict: **{verdict}**.", "",
             "| dataset | test relations | Delta = name - cosine | Gain from the name | call | d_r > 0 | mean MRR cosine / mean name / name / shuffled |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for ds, c in calls.items():
        mm = c["beside"]["mean MRR (points)"]
        lines.append(f"| {ds} | {c['relations']} | {fmt(c['Delta name - cosine'])} | {fmt(c['Gain from the name (name - mean name)'])} | "
                     f"{c['call']} | {c['beside']['relations with d_r > 0']} | {mm['cosine']:.2f} / {mm['mean']:.2f} / {mm['name']:.2f} / {mm['shuffled']:.2f} |")
    lines += ["", "Expectations: " + "; ".join(f"{k}: {v}" for k, v in x.items()), ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"report: {verdict}; {json.dumps({ds: c['call'] for ds, c in calls.items()})}")
    return 0


# ── self-test (11.9): planted names ─────────────────────────────────────────


def selftest(_a):
    torch = Z.torch_()
    rng = np.random.default_rng(0)
    nf, per_f, m = 8, 5, 120                                   # 8 families of 5 relations, m edges each
    D, Dn, nrel = 32, 16, nf * per_f
    n = nrel * 2 * m + 1200
    V = rng.standard_normal((n, D))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    cent = unit(rng.standard_normal((nf, Dn)))
    th_f = rng.uniform(1.0, 2.0, (nf, D // 2)) * rng.choice([-1.0, 1.0], (nf, D // 2))
    vocab, vec_of, S = [], {}, []
    for r in range(nrel):
        f = r // per_f
        src = np.arange(r * 2 * m, r * 2 * m + m)
        dst = src + m
        V[dst] = unit(Z.rotate_np(V[src], th_f[f]) + 0.2 * rng.standard_normal((m, D)) / np.sqrt(D))
        vocab.append(f"fam{f}_rel{r % per_f}")
        vec_of["S." + vocab[-1]] = unit((cent[f] + 0.05 * rng.standard_normal(Dn))[None])[0].astype(np.float32)
        S.append((src.astype(np.int64) * n + dst) * nrel + r)
    a0 = nrel * 2 * m
    aa, bb = np.arange(a0, a0 + 500), np.arange(a0 + 500, a0 + 1000)
    w = np.repeat(rng.uniform(-1, 1, D // 2), 2)
    V[bb] = unit(V[aa] * w + 0.2 * rng.standard_normal((500, D)) / np.sqrt(D))
    vec_of["R.toy"] = unit(rng.standard_normal((1, Dn)))[0].astype(np.float32)
    rk = np.unique(np.minimum(aa, bb).astype(np.int64) * n + np.maximum(aa, bb))
    Sk = np.unique(np.concatenate(S))
    G = Z.Graph("toy", n, [("R.toy", rk)], Sk, vocab)
    X = Z.Vecs(V.astype(np.float32).astype(np.float16))
    cfg = dict(CFGR, batch=256, neg=512, epoch=30_000, max_epochs=30, patience=5, lr=0.02, extra=8, hidden=32, min_op=30, n_dev=100,
               n_read=100, chunk=200)
    gates = {}
    rec, _arr = run_fold(torch, G, X, vec_of, cfg, "toy", 0, None, gates)
    c = calls_from(rec["per_relation"])
    d, e = c["Delta name - cosine"], c["Gain from the name (name - mean name)"]
    log(f"self-test: {len(rec['relations']['train'])} train, {len(rec['relations']['dev'])} dev, {len(rec['relations']['test'])} test relations; "
        f"mean MRR {c['beside']['mean MRR (points)']}; Delta {fmt(d)}; gain {fmt(e)}; shuffled {fmt(c['beside']['name - shuffled name'])}; "
        f"gates {gates}")
    mm = c["beside"]["mean MRR (points)"]
    ok = (d["call"] == "ABOVE" and d["diff"] > 10 and mm["name"] > mm["mean"] + 5 and mm["name"] > mm["shuffled"] + 5
          and all(v == "PASS" or (isinstance(v, dict) and v.get("verdict") == "PASS") for v in gates.values()))
    assert u_of("toy", "S.a") == u_of("toy", "S.a") and name_string("S.directed_by") == "directed by" and name_string("R.name_comention") == "name comention"
    assert name_string("S.award honor award winner") == "award honor award winner"
    role = assign("metaqa", [f"S.r{i}" for i in range(7)], 0)
    assert sum(v == "test" for v in role.values()) == 3 and sum(v == "dev" for v in role.values()) == 1
    allt = [n_ for f in range(3) for n_, v in assign("metaqa", [f"S.r{i}" for i in range(7)], f).items() if v == "test"]
    assert sorted(allt) == sorted(f"S.r{i}" for i in range(7)), allt
    dd = derange([3, 5, 9, 11], 0)
    assert sorted(dd.tolist()) == [3, 5, 9, 11] and not (dd == np.asarray([3, 5, 9, 11])).any()
    if not ok:
        stop("self-test FAILED")
    log("self-test PASS: planted family operators transfer to held-out relations through their names; G-0, G-S hold")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "names", "fit", "report"))
    p.add_argument("--ds", choices=ROWS)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="a one-epoch code check on a few edges; written under outputs/sem2/r3/smoke/, read by nothing")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if not a.ds:
        stop(f"{a.cmd} needs --ds")
    return cmd_names(a) if a.cmd == "names" else cmd_fit(a)


if __name__ == "__main__":
    sys.exit(main())
