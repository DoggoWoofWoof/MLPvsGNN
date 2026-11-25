"""SEM2 R6: do link operators cross between passages and a KB? (docs/SEM2_EDGE_OPERATORS.md section 14)

    python outputs/mp_unified/zsem2r6.py selftest
    python outputs/mp_unified/zsem2r6.py names --host                          (R3's names command on HotpotQA: the string "hyperlink")
    python outputs/mp_unified/zsem2r6.py fit   --net p --host --threads 8      (N_P: a name net on HotpotQA's four labels)
    python outputs/mp_unified/zsem2r6.py read  --net kb|p --ds D --host --threads 8
    python outputs/mp_unified/zsem2r6.py report                                (laptop: the calls, from the read records)

A wrapper over zsem2.py (R1) and zsem2r3.py (R3): it imports their graph, split, sampler, filtered rank, net and calls and edits none.
- N_KB is R3's stored WebQSP net (outputs/sem2/r3/webqsp/f0.npz, best epoch; pinned by R3's record), never refit.
- N_P is fitted here on one passage corpus (HotpotQA): its four labels, all train; the stop is on dev edges of those labels (R1's rule).
- read: a stored net applied to every unit of a target corpus (14.4): cosine, name, mean name, shuffled name on up to 200 test edges
  per unit. No target is fitted on.
- report: the four directions' calls and the bridge verdict (14.5).
Outputs: outputs/sem2/r6/nets/p_hotpotqa.json/.npz, outputs/sem2/r6/reads/<net>_<dataset>.json/.npz, report.json/.md.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import zsem2 as Z  # noqa: E402  (imports zenc first: the BLAS threads from --threads before numpy loads)
import zsem2r3 as R3  # noqa: E402
import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

E = Z.E
log, stop = Z.log, Z.stop
DECLARED = "docs/SEM2_EDGE_OPERATORS.md section 14"
OUT = Z.ROOT / "outputs" / "sem2" / "r6"
PASSAGE = ("multihoprag", "squad", "musique", "hotpotqa", "2wiki")
KB = ("metaqa", "webqsp")
SCORERS = R3.SCORERS
CFG6 = dict(R3.CFGR)
MIN_UNIT = 20
SEED, BOOT = Z.SEED, Z.BOOT
KB_REC, KB_NPZ = R3.OUT / "webqsp" / "f0.json", R3.OUT / "webqsp" / "f0.npz"
P_EXPECT = ("R.mention", "R.name_comention", "R.title_comention", "S.hyperlink")


def script_sha():
    return Z.sha_file(Path(__file__))


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


# ── names ───────────────────────────────────────────────────────────────────


NAMES_ON = ("hotpotqa", "2wiki")


def name_pool(datasets=("webqsp", "metaqa", "hotpotqa", "2wiki")):
    """label -> name vector from R3's names files, the first file that has the label winning (WebQSP's, which N_KB was fitted on).
    A label in two files must have the same vector: the lowest cosine between two copies is recorded and must be >= 0.999."""
    pool, recs, low = {}, {}, 1.0
    for ds in datasets:
        if not (R3.OUT / "names" / f"{ds}.npz").is_file():
            continue
        vec, rec = R3.load_names(ds)
        recs[ds] = rec
        for k, v in vec.items():
            if k in pool:
                c = float(np.dot(pool[k], v) / (np.linalg.norm(pool[k]) * np.linalg.norm(v)))
                low = min(low, c)
                if c < R3.G_N_COS:
                    stop(f"name vector of {k} differs between names files (cosine {c:.6f})")
            else:
                pool[k] = np.asarray(v, np.float32)
    pool_low[0] = low
    return pool, recs


pool_low = [1.0]


# ── a stored net ────────────────────────────────────────────────────────────


class Net:
    def __init__(self, kind, model, train_names, mean_vec, sha, rec):
        self.kind, self.model, self.train_names, self.mean_vec, self.sha, self.rec = kind, model, list(train_names), mean_vec, sha, rec
        self.seen = {R3.name_string(n) for n in self.train_names}


def model_from_arrays(torch, arrays, hid=None):
    P = np.asarray(arrays["model.P"])
    Dp, D = P.shape
    w0 = np.asarray(arrays["model.g.0.weight"])
    Dn = w0.shape[1]
    model = R3.make_net(torch, D, Dp, Dn, w0.shape[0] if hid is None else hid)
    sd = {k[len("model."):]: torch.from_numpy(np.asarray(v).copy()) for k, v in arrays.items() if k.startswith("model.")}
    model.load_state_dict(sd)
    model.eval()
    return model


def mean_of(pool, names):
    miss = [n for n in names if n not in pool]
    if miss:
        stop(f"no name vector for {miss[:5]} ({len(miss)}) among the net's train names")
    return np.mean(np.stack([pool[n] for n in names]), axis=0).astype(np.float32)


def load_net(torch, kind, pool, smoke=False):
    if kind == "kb":
        rec = json.loads(KB_REC.read_text(encoding="utf-8"))
        if rec["script_sha256"] != R3.script_sha():
            stop("R3's WebQSP record was fitted by other code than the zsem2r3.py on disk")
        if Z.sha_file(KB_NPZ) != rec["npz_sha256"]:
            stop("outputs/sem2/r3/webqsp/f0.npz is not its record's")
        if rec["smoke"] or rec["fold"] != 0:
            stop("not R3's WebQSP fold 0")
        arrays = dict(np.load(KB_NPZ))
        model = model_from_arrays(torch, arrays)
        names = rec["relations"]["train"]
        return Net("kb", model, names, mean_of(pool, names), rec["npz_sha256"], rec)
    base = OUT / ("smoke" if smoke else "") / "nets"
    jp, zp = base / "p_hotpotqa.json", base / "p_hotpotqa.npz"
    if not jp.is_file():
        stop(f"no N_P: run `fit --net p` first ({jp})")
    rec = json.loads(jp.read_text(encoding="utf-8"))
    if rec["script_sha256"] != script_sha():
        stop("N_P was fitted by other code than the zsem2r6.py on disk")
    if Z.sha_file(zp) != rec["npz_sha256"]:
        stop("N_P's npz is not its record's")
    if rec["gates"]["G-0"]["verdict"] != "PASS":
        stop("N_P's G-0 did not pass")
    model = model_from_arrays(torch, dict(np.load(zp)))
    return Net("p", model, rec["train_names"], mean_of(pool, rec["train_names"]), rec["npz_sha256"], rec)


# ── units (14.4) ────────────────────────────────────────────────────────────


def unit_plan(ds, own, G):
    """[(L, name, kind, main)] for the target's own labels that are units; kind S/R; main = in the direction's main call."""
    s_names = [nm for L, nm in own if L >= G.nR]
    out = []
    if ds == "webqsp":
        role = R3.assign("webqsp", s_names, 0)
        for L, nm in own:
            if L < G.nR:
                out.append((L, nm, "R", False))
            elif role[nm] == "test":
                out.append((L, nm, "S", True))
    elif ds in PASSAGE:
        out = [(L, nm, "S" if L >= G.nR else "R", True) for L, nm in own]
    else:                                   # metaqa, and the toy graphs of the self-test
        out = [(L, nm, "S" if L >= G.nR else "R", L >= G.nR) for L, nm in own]
    return out


def read_core(torch, G, X, vec_of, net, ds, cfg, ckdir, gates, key_extra=""):
    """One net on one target: the per-unit measures of the four scorers on up to n_read test edges per unit."""
    t0 = time.time()
    split = Z.node_split(G.name, G.n)
    labs, _ops, _op0, _g = Z.plan(G, split, cfg)
    own = R3.own_labels(labs)
    miss = [nm for _L, nm in own if nm not in vec_of]
    if miss:
        stop(f"no name vector for {miss[:5]} ({len(miss)}): run `names` for the dataset that has the label")
    Nt = torch.from_numpy(np.stack([vec_of[nm] for _L, nm in own]).astype(np.float32))
    op_of = np.full(G.nR + G.RS, -1, np.int64)
    for k, (L, _nm) in enumerate(own):
        op_of[L] = k
    units = unit_plan(ds, own, G)
    if not units:
        stop(f"{ds}: no unit")
    groups = [R3.make_group(G, L, nm) for L, nm, _k, _m in units]
    rng = np.random.default_rng(SEED)
    test = Z.take(groups, split, Z.is_test, cfg["n_read"], rng)
    test["nR"] = G.nR
    gates["G-U"] = "PASS" if (bool((split[test["s"]] == Z.TEST).all()) and bool((test["lab"] == np.asarray([units[g][0] for g in test["grp"]])).all())) else "FAIL"
    if gates["G-U"] != "PASS":
        stop("gate G-U")
    log(f"{ds}: {len(units)} units ({sum(1 for u in units if u[3])} main), {test['s'].size:,} test edges ({time.time() - t0:.0f} s)")
    seen = [R3.name_string(nm) in net.seen for _L, nm, _k, _m in units]
    gates["G-S"] = {"units": len(units), "seen_names": int(sum(seen)),
                    "seen_labels": [u[1] for u, s in zip(units, seen) if s][:20]}
    row = {nm: k for k, (_L, nm) in enumerate(own)}
    ur = [row[u[1]] for u in units]
    Nmean = Nt.clone()
    Nmean[:] = torch.from_numpy(net.mean_vec)[None, :]
    Nshuf = Nt.clone()
    Nshuf[ur] = Nt[R3.derange(ur, SEED)]
    cos = Z.Cosine()
    scs = {"cosine": cos, "name": R3.NameScorer(net.model, Nt), "mean": R3.NameScorer(net.model, Nmean),
           "shuffled": R3.NameScorer(net.model, Nshuf)}
    key = hashlib.sha256(json.dumps({"net": net.kind, "net_sha": net.sha, "ds": ds, "cfg": cfg, "code": script_sha(), "n": int(G.n),
                                     "extra": key_extra}, sort_keys=True).encode()).hexdigest()[:16]
    ranks = {}
    if ckdir is not None:
        ckdir.mkdir(parents=True, exist_ok=True)
    for tag in SCORERS:
        ranks[tag] = R3.chunked(torch, G, X, scs[tag], test, op_of, ckdir, tag, cfg["chunk"], key)
        log(f"{ds}: {net.kind} {tag} test {Z.measures(ranks[tag])} ({time.time() - t0:.0f} s)")
    per = {}
    for g, (_L, nm, kind, main) in enumerate(units):
        m = test["grp"] == g
        e = {"label": nm, "kind": kind, "main": bool(main), "seen": bool(seen[g]), "edges": int(m.sum())}
        if m.any():
            for t in SCORERS:
                e[f"mrr_{t}"] = round(100 * float((1.0 / ranks[t][m]).mean()), 3)
                e[f"h10_{t}"] = round(100 * float((ranks[t][m] <= 10).mean()), 3)
                e[f"h100_{t}"] = round(100 * float((ranks[t][m] <= 100).mean()), 3)
        per[f"{ds}|{nm}"] = e
    arrays = {"test_s": test["s"], "test_d": test["d"], "test_lab": test["lab"], "test_grp": test["grp"],
              "unit_names": np.asarray([u[1] for u in units]), **{f"rank_{t}": ranks[t] for t in SCORERS}}
    rec = {"units": per, "test_edges": int(test["s"].size), "pooled": {t: Z.measures(ranks[t]) for t in SCORERS}}
    return rec, arrays


def gate_loader(torch, G, X, vec_of, net, cfg, gates):
    """G-L: the stored WebQSP net, reloaded, reproduces R3's recorded name and mean-name ranks on up to 300 stored test edges."""
    split = Z.node_split(G.name, G.n)
    labs, _ops, _op0, _g = Z.plan(G, split, cfg)
    own = R3.own_labels(labs)
    Nt = torch.from_numpy(np.stack([vec_of[nm] for _L, nm in own]).astype(np.float32))
    op_of = np.full(G.nR + G.RS, -1, np.int64)
    for k, (L, _nm) in enumerate(own):
        op_of[L] = k
    z = np.load(KB_NPZ)
    test = {"s": z["test_s"], "d": z["test_d"], "lab": z["test_lab"], "grp": z["test_grp"], "nR": G.nR}
    sel = np.unique(np.linspace(0, test["s"].size - 1, min(300, test["s"].size)).astype(np.int64))
    sub = R3.subsample(test, sel)
    ex = Z.exclusions(G, sub)
    Nmean = Nt.clone()
    Nmean[:] = torch.from_numpy(net.mean_vec)[None, :]
    out = {"edges_checked": int(sel.size)}
    ok = True
    for tag, tab in (("name", Nt), ("mean", Nmean)):
        r, _t = Z.rank_read(torch, R3.NameScorer(net.model, tab), X, sub, ex, op_of)
        rec = z[f"rank_{tag}"][sel]
        near = float((np.abs(r - rec) <= np.maximum(1.0, 0.001 * rec)).mean())
        out[f"{tag}_rank_near_share"] = round(near, 4)
        out[f"{tag}_rank_equal_share"] = round(float((np.abs(r - rec) < 1e-9).mean()), 4)
        ok &= near >= 0.99
    out["verdict"] = "PASS" if ok else "FAIL"
    gates["G-L"] = out
    if not ok:
        stop(f"gate G-L: the reloaded net does not reproduce R3's ranks: {out}")


# ── the fit of N_P (14.3) ───────────────────────────────────────────────────


def fit_p(torch, G, X, vec_of, cfg, ckdir, gates, expect=P_EXPECT):
    t0 = time.time()
    split = Z.node_split(G.name, G.n)
    labs, _ops, _op0, _g = Z.plan(G, split, cfg)
    own = R3.own_labels(labs)
    if expect is not None and sorted(nm for _L, nm in own) != sorted(expect):
        stop(f"{G.name}: the labels with an operator are {[nm for _L, nm in own]}, not {list(expect)}")
    names = [nm for _L, nm in own]
    N = torch.from_numpy(np.stack([vec_of[nm] for nm in names]).astype(np.float32))
    Dn = N.shape[1]
    op_of = np.full(G.nR + G.RS, -1, np.int64)
    for k, (L, _nm) in enumerate(own):
        op_of[L] = k
    groups = [R3.make_group(G, L, nm) for L, nm in own]
    train_nodes = np.flatnonzero(split == Z.TRAIN)
    rng = np.random.default_rng(SEED)
    dev = Z.take(groups, split, Z.is_dev, cfg["n_dev"], rng)
    dev["nR"] = G.nR
    gates["G-U dev"] = "PASS" if bool(((split[dev["s"]] == Z.DEV) & (split[dev["d"]] != Z.TEST)).all()) else "FAIL"
    if gates["G-U dev"] != "PASS":
        stop("gate G-U (dev)")
    dev_ex = Z.exclusions(G, dev)
    log(f"{G.name}: N_P on {names}: dev {dev['s'].size:,} edges ({time.time() - t0:.0f} s)")
    cos = Z.Cosine()
    cos_dev = Z.rank_read(torch, cos, X, dev, dev_ex, op_of)
    D = X.D
    Dp = D + cfg["extra"]
    model = R3.make_net(torch, D, Dp, Dn, cfg["hidden"])
    sc = R3.NameScorer(model, N)
    r0, t0s = Z.rank_read(torch, sc, X, dev, dev_ex, op_of)
    g0 = {"max_score_diff": float(np.abs(t0s - cos_dev[1]).max()), "mrr_diff": float(abs((1.0 / r0).mean() - (1.0 / cos_dev[0]).mean()))}
    g0["verdict"] = "PASS" if g0["max_score_diff"] <= Z.G0_SCORE and g0["mrr_diff"] <= Z.G0_MRR else "FAIL"
    gates["G-0"] = g0
    if g0["verdict"] != "PASS":
        stop(f"gate G-0: {g0}")
    key = hashlib.sha256(json.dumps({"names": names, "cfg": cfg, "code": script_sha(), "n": int(G.n), "zsem2": Z.sha_file(Path(Z.__file__))},
                                    sort_keys=True).encode()).hexdigest()[:16]
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
                log(f"{G.name}: resumed after epoch {resumed}")
        except Exception as ex:  # noqa: BLE001
            log(f"{G.name}: checkpoint not read ({ex}); the fit starts at epoch 1")
    share = cfg["epoch"] // max(len(groups), 1)
    done = resumed is not None and since >= cfg["patience"]
    for ep in range(ep0, cfg["max_epochs"] + 1):
        if done:
            break
        te = time.time()
        er = np.random.default_rng(ep)
        tr = Z.take(groups, split, Z.is_train, share, er)
        m = tr["s"].size
        if not bool(((split[tr["s"]] == Z.TRAIN) & (split[tr["d"]] == Z.TRAIN)).all()):
            stop("a training edge is not a train-train edge")
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
        log(f"{G.name}: epoch {ep}: {m:,} edges, loss {tot / max(nb, 1):.4f}, dev MRR {mrr} ({time.time() - te:.0f} s)")
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
    model.load_state_dict(best["state"])
    model.eval()
    rec = {"train_names": names, "fit": {"history": hist, "best_epoch": best["epoch"], "best_dev_mrr": best["dev_mrr"],
                                         "cosine_dev_mrr": Z.measures(cos_dev[0])["mrr"], "last_epoch": hist[-1]["epoch"],
                                         "ended_by": "patience" if since >= cfg["patience"] else "max_epochs",
                                         "best_is_last_epoch": bool(best["epoch"] == hist[-1]["epoch"]),
                                         "resumed_after_epoch": resumed, "D_prime": Dp, "hidden": cfg["hidden"]},
           "dev_edges": int(dev["s"].size)}
    arrays = {f"model.{k}": v.detach().numpy() for k, v in model.state_dict().items()}
    return rec, arrays


# ── commands ────────────────────────────────────────────────────────────────


def cmd_names(a):
    """R3's names command on the two corpora that carry a structural label (14.3): HotpotQA's string 'hyperlink'; 2Wiki's, read too."""
    for ds in ([a.ds] if a.ds else NAMES_ON):
        R3.cmd_names(argparse.Namespace(ds=ds, host=a.host))
    return 0


def load_target(ds, host):
    G, V, inputs, ge = Z.load_mhr("e3v") if ds == "multihoprag" else Z.load_six(ds, "e3v", host)
    return G, V, inputs, ge


def cmd_fit(a):
    torch = Z.torch_()
    t0 = time.time()
    if a.net != "p":
        stop("only --net p is fitted here (N_KB is R3's stored net)")
    pool, nrecs = name_pool()
    if "hotpotqa" not in nrecs:
        stop("no names file for hotpotqa: run `names` first")
    G, V, inputs, ge = load_target("hotpotqa", a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"}, "G-N": nrecs["hotpotqa"]["G-N"]}
    X = Z.Vecs(V)
    cfg = dict(CFG6, **Z.SMOKE, n_read=20, chunk=500) if a.smoke else CFG6
    od = OUT / ("smoke" if a.smoke else "") / "nets"
    ck = od / "ckpt"
    rec, arrays = fit_p(torch, G, X, pool, cfg, ck, gates)
    rec = {"stage": "SEM2 R6 N_P fit" + (" (SMOKE: a code check, not a read)" if a.smoke else ""), "declared_in": DECLARED,
           "dataset": "hotpotqa", "arm": "e3v", "smoke": bool(a.smoke), "config": cfg, "gates": gates, **rec,
           "inputs": {**inputs, "names_npz_sha256": nrecs["hotpotqa"]["npz_sha256"]}, "threads": E.THREADS,
           "seconds": round(time.time() - t0, 1), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script_sha256": script_sha()}
    atomic_npz(od / "p_hotpotqa.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / "p_hotpotqa.npz")
    Z.wjson(od / "p_hotpotqa.json", rec)
    if ck.is_dir():
        for p in sorted(ck.glob("*")):
            p.unlink()
        ck.rmdir()
    log(f"N_P written {od / 'p_hotpotqa.json'} ({rec['seconds']} s); best epoch {rec['fit']['best_epoch']}, dev MRR {rec['fit']['best_dev_mrr']} "
        f"(cosine {rec['fit']['cosine_dev_mrr']})")
    return 0


def cmd_read(a):
    torch = Z.torch_()
    t0 = time.time()
    if a.ds not in Z.SEVEN or a.net not in ("kb", "p"):
        stop("read needs --net kb|p and --ds one of the seven")
    pool, nrecs = name_pool()
    net = load_net(torch, a.net, pool, smoke=a.smoke)
    G, V, inputs, ge = load_target(a.ds, a.host)
    gates = {"G-E": {"families": ge, "verdict": "PASS"},
             "G-N": {k: {"min_cosine": r["G-N"]["min_cosine"], "verdict": r["G-N"]["verdict"]} for k, r in nrecs.items()}}
    if any(v["verdict"] != "PASS" for v in gates["G-N"].values()):
        stop("a names file's G-N is not PASS")
    X = Z.Vecs(V)
    cfg = dict(CFG6, **Z.SMOKE, n_read=20, chunk=500) if a.smoke else CFG6
    od = OUT / ("smoke" if a.smoke else "") / "reads"
    base = {"stage": "SEM2 R6 read" + (" (SMOKE: a code check, not a read)" if a.smoke else ""), "declared_in": DECLARED, "net": a.net,
            "net_sha256": net.sha, "dataset": a.ds, "arm": "e3v", "smoke": bool(a.smoke), "config": cfg}
    tail = {"inputs": {**inputs, "names_npz_sha256": {k: r["npz_sha256"] for k, r in nrecs.items()}}, "threads": E.THREADS,
            "script_sha256": script_sha()}
    if a.net == "kb" and a.ds == "webqsp":
        vec_of = pool
        gate_loader(torch, G, X, vec_of, net, cfg, gates)
        rec = {**base, "gate_only": True, "gates": gates, **tail, "seconds": round(time.time() - t0, 1),
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        Z.wjson(od / f"{a.net}_{a.ds}.json", rec)
        log(f"G-L {gates['G-L']}")
        return 0
    ck = od / "ckpt" / f"{a.net}_{a.ds}"
    rec, arrays = read_core(torch, G, X, pool, net, a.ds, cfg, ck, gates)
    rec = {**base, "gate_only": False, "gates": gates, **rec, **tail, "seconds": round(time.time() - t0, 1),
           "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    atomic_npz(od / f"{a.net}_{a.ds}.npz", **arrays)
    rec["npz_sha256"] = Z.sha_file(od / f"{a.net}_{a.ds}.npz")
    Z.wjson(od / f"{a.net}_{a.ds}.json", rec)
    if ck.is_dir():
        for p in sorted(ck.glob("*")):
            p.unlink()
        ck.rmdir()
    log(f"{a.net} on {a.ds}: written {od / f'{a.net}_{a.ds}.json'} ({rec['seconds']} s); pooled {rec['pooled']}")
    return 0


# ── the calls (14.5) ────────────────────────────────────────────────────────


def fmt(c):
    return f"{c['diff']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}] {c['call']}"


def pooled_call(units, min_unit=MIN_UNIT):
    """units: {key: per-unit entry}; the calls of R3's form over the units with at least min_unit (14.4: 20) edges."""
    keep = {k: v for k, v in units.items() if v["edges"] >= min_unit}
    out = {"units": len(keep), "dropped_under_min": sorted(set(units) - set(keep))}
    if len(keep) >= 3:
        out.update(R3.calls_from(keep))
    elif keep:
        out["mean MRR (points)"] = {t: round(float(np.mean([v[f"mrr_{t}"] for v in keep.values()])), 3) for t in SCORERS}
        out["call"] = "TOO_FEW_UNITS"
    else:
        out["call"] = "NO_UNITS"
    return out


DIRECTIONS = (
    ("1 KB -> KB (WebQSP net on MetaQA's S relations)", "kb", ("metaqa",), "S"),
    ("2 KB -> passage (WebQSP net on the 15 passage units)", "kb", PASSAGE, "all"),
    ("3 passage -> KB (HotpotQA net on WebQSP's 163 and MetaQA's 7)", "p", KB, "S"),
    ("4 passage -> passage, control (HotpotQA net on the other four corpora)", "p", tuple(d for d in PASSAGE if d != "hotpotqa"), "all"),
)


def units_of(reads, net, dsets, which):
    out = {}
    for ds in dsets:
        r = reads.get((net, ds))
        if r is None:
            return None
        for k, v in r["units"].items():
            if v["main"] and (which == "all" or v["kind"] == which):
                out[k] = v
    return out


def report_core(reads, nets, gate_l):
    """reads: {(net, ds): record}; nets: {'p': fit record or None}; gate_l: the G-L record or None."""
    res = {"directions": {}, "beside": {}, "missing": []}
    if gate_l is None or gate_l.get("verdict") != "PASS":
        res["missing"].append("G-L (the loader gate on WebQSP)")
    if nets.get("p") is None:
        res["missing"].append("the N_P fit")
    for name, net, dsets, which in DIRECTIONS:
        u = units_of(reads, net, dsets, which)
        if u is None:
            res["directions"][name] = {"call": "INCOMPLETE", "missing": [f"{net} on {d}" for d in dsets if (net, d) not in reads]}
            continue
        res["directions"][name] = pooled_call(u)
    d1, d2, d3, d4 = (res["directions"][n[0]] for n in DIRECTIONS)
    ctl = d4.get("Delta name - cosine", {}).get("call")
    if res["missing"] or d2.get("call") == "INCOMPLETE" or d3.get("call") == "INCOMPLETE":
        res["verdict"] = "INCOMPLETE"
    else:
        a2 = d2["Delta name - cosine"]["call"] == "ABOVE"
        a3 = d3["Delta name - cosine"]["call"] == "ABOVE"
        res["verdict"] = "BRIDGES" if a2 and a3 else ("ONE_WAY (KB -> passage)" if a2 else ("ONE_WAY (passage -> KB)" if a3 else "NOT_BRIDGED"))
    res["control_direction_4_delta"] = ctl
    # beside: per dataset, per kind
    for net in ("kb", "p"):
        for ds in Z.SEVEN:
            r = reads.get((net, ds))
            if r is None or r.get("gate_only"):
                continue
            main = {k: v for k, v in r["units"].items() if v["main"]}
            if main:
                res["beside"][f"{net} on {ds} (main units)"] = pooled_call(main)
            other = {k: v for k, v in r["units"].items() if not v["main"]}
            if other:
                res["beside"][f"{net} on {ds} (beside units: R labels)"] = pooled_call(other)
    r2 = [(k, v) for ds in PASSAGE for k, v in (reads.get(("kb", ds)) or {"units": {}})["units"].items()]
    hyper = {k: v for k, v in r2 if v["label"] == "S.hyperlink"}
    rlab = {k: v for k, v in r2 if v["kind"] == "R"}
    if hyper:           # two units cannot carry an interval: the per-unit MRRs are the read (14.6 X23)
        res["hyperlink_units"] = {k: {"edges": v["edges"], **{t: v.get(f"mrr_{t}") for t in SCORERS}} for k, v in hyper.items()}
    if rlab:
        res["beside"]["kb on R-label units (names seen)"] = pooled_call(rlab)
    # the expectations that can be read from the calls (14.6)
    def call_of(name):
        return res["directions"].get(name, {}).get("Delta name - cosine", {}).get("call")
    n1, n2, n3, n4 = (d[0] for d in DIRECTIONS)
    x = {"X22 KB -> KB Delta not ABOVE": None if call_of(n1) is None else call_of(n1) != "ABOVE",
         "X23 KB -> passage Delta not ABOVE": None if call_of(n2) is None else call_of(n2) != "ABOVE",
         "X23 hyperlink units: name no better than cosine on both": (None if not hyper else all(v["mrr_name"] <= v["mrr_cosine"] for v in hyper.values())),
         "X24 passage -> KB Delta not ABOVE": None if call_of(n3) is None else call_of(n3) != "ABOVE",
         "X25 passage -> passage Delta ABOVE": None if call_of(n4) is None else call_of(n4) == "ABOVE"}
    rp = (reads.get(("p", "hotpotqa")) or {}).get("units")
    if rp:
        pc = pooled_call({k: v for k, v in rp.items() if v["main"]})
        res["beside"]["p on hotpotqa, in sample (X26)"] = pc
        x["X26 N_P in sample Delta ABOVE"] = pc.get("Delta name - cosine", {}).get("call") == "ABOVE"
    else:
        x["X26 N_P in sample Delta ABOVE"] = None
    res["expectations"] = x
    return res


def cmd_report(_a):
    sha = script_sha()
    reads, bad = {}, []
    rd = OUT / "reads"
    for net in ("kb", "p"):
        for ds in Z.SEVEN:
            p = rd / f"{net}_{ds}.json"
            if not p.is_file():
                continue
            r = json.loads(p.read_text(encoding="utf-8"))
            if r["smoke"] or r["script_sha256"] != sha:
                bad.append(p.name)
                continue
            if not r.get("gate_only"):
                if Z.sha_file(rd / f"{net}_{ds}.npz") != r["npz_sha256"]:
                    bad.append(p.name + " (npz)")
                    continue
            if any(isinstance(v, dict) and v.get("verdict") == "FAIL" for v in r["gates"].values()):
                bad.append(p.name + " (gate)")
                continue
            reads[(net, ds)] = r
    if bad:
        stop(f"records fitted or read by other code, smoke records, or failed gates: {bad}")
    gl = (reads.get(("kb", "webqsp")) or {}).get("gates", {}).get("G-L")
    nets = {"p": None}
    pj = OUT / "nets" / "p_hotpotqa.json"
    if pj.is_file():
        nets["p"] = json.loads(pj.read_text(encoding="utf-8"))
        if nets["p"]["script_sha256"] != sha or nets["p"]["smoke"]:
            stop("N_P's record is a smoke record or was fitted by other code")
    res = report_core(reads, nets, gl)
    res = {"declared_in": DECLARED, "verdict": res["verdict"], **{k: v for k, v in res.items() if k != "verdict"},
           "records": {f"{n}_{d}": {"npz_sha256": r.get("npz_sha256"), "utc": r["utc"]} for (n, d), r in sorted(reads.items())},
           "script_sha256": sha, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUT.mkdir(parents=True, exist_ok=True)
    Z.wjson(OUT / "report.json", res)
    L = ["# SEM2 R6 report (docs/SEM2_EDGE_OPERATORS.md section 14)", "", f"Verdict: **{res['verdict']}**. Missing: {res['missing'] or 'none'}.",
         f"Control, direction 4 Delta: {res['control_direction_4_delta']}.", "",
         "| direction | units | Delta name - cosine | Gain from the name | call | mean MRR cosine / mean name / name / shuffled |", "|---|---|---|---|---|---|"]
    for k, v in res["directions"].items():
        if "Delta name - cosine" in v:
            mm = v["beside"]["mean MRR (points)"]
            L.append(f"| {k} | {v['units']} | {fmt(v['Delta name - cosine'])} | {fmt(v['Gain from the name (name - mean name)'])} | {v['call']} | "
                     f"{mm['cosine']:.2f} / {mm['mean']:.2f} / {mm['name']:.2f} / {mm['shuffled']:.2f} |")
        else:
            L.append(f"| {k} | {v.get('units', 0)} | - | - | {v['call']} {v.get('missing', '')} | - |")
    L += ["", "## Beside (per dataset and per kind; deciding nothing)", "", "| read | units | Delta | Gain | call | mean MRR c / m / n / s |", "|---|---|---|---|---|---|"]
    for k, v in res["beside"].items():
        if "Delta name - cosine" in v:
            mm = v["beside"]["mean MRR (points)"]
            L.append(f"| {k} | {v['units']} | {fmt(v['Delta name - cosine'])} | {fmt(v['Gain from the name (name - mean name)'])} | {v['call']} | "
                     f"{mm['cosine']:.2f} / {mm['mean']:.2f} / {mm['name']:.2f} / {mm['shuffled']:.2f} |")
        else:
            mm = v.get("mean MRR (points)")
            L.append(f"| {k} | {v.get('units', 0)} | - | - | {v['call']} | " + (" / ".join(f"{mm[t]:.2f}" for t in SCORERS) if mm else "-") + " |")
    if res.get("hyperlink_units"):
        L += ["", "## S.hyperlink units under N_KB (the only passage-side name it has not seen; MRR points)", "",
              "| unit | edges | cosine | name | mean name | shuffled |", "|---|---|---|---|---|---|"]
        for k, v in res["hyperlink_units"].items():
            L.append(f"| {k} | {v['edges']} | " + " | ".join(f"{v[t]:.2f}" for t in SCORERS) + " |")
    L += ["", "## Expectations (14.6; None = not yet readable)", ""] + [f"- {k}: {v}" for k, v in res["expectations"].items()]
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    log(f"report: {res['verdict']} (missing {res['missing']})")
    for k, v in res["directions"].items():
        log(f"  {k}: {v.get('call')} {fmt(v['Delta name - cosine']) if 'Delta name - cosine' in v else ''}")
    return 0


# ── self-test ───────────────────────────────────────────────────────────────


def toy_graph(seed, th_f, names_of, cent, nf=8, per_f=5, m=120, D=32, Dn=16):
    """A toy KB: nf families of per_f relations, each family's edges rotate a source by th_f[f]; relation names from names_of(f, k)."""
    rng = np.random.default_rng(seed)
    nrel = nf * per_f
    n = nrel * 2 * m + 1200
    V = rng.standard_normal((n, D))
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731
    vocab, vec_of, S = [], {}, []
    for r in range(nrel):
        f = r // per_f
        src = np.arange(r * 2 * m, r * 2 * m + m)
        dst = src + m
        V[dst] = unit(Z.rotate_np(V[src], th_f[f]) + 0.2 * rng.standard_normal((m, D)) / np.sqrt(D))
        vocab.append(names_of(f, r % per_f))
        vec_of["S." + vocab[-1]] = unit((cent[f] + 0.05 * rng.standard_normal(Dn))[None])[0].astype(np.float32)
        S.append((src.astype(np.int64) * n + dst) * nrel + r)
    a0 = nrel * 2 * m
    aa, bb = np.arange(a0, a0 + 500), np.arange(a0 + 500, a0 + 1000)
    w = np.repeat(rng.uniform(-1, 1, D // 2), 2)
    V[bb] = unit(V[aa] * w + 0.2 * rng.standard_normal((500, D)) / np.sqrt(D))
    vec_of["R.toy"] = unit(np.random.default_rng(99).standard_normal((1, Dn)))[0].astype(np.float32)
    rk = np.unique(np.minimum(aa, bb).astype(np.int64) * n + np.maximum(aa, bb))
    Sk = np.unique(np.concatenate(S))
    G = Z.Graph("toy", n, [("R.toy", rk)], Sk, vocab)
    X = Z.Vecs(V.astype(np.float32).astype(np.float16))
    return G, X, vec_of


def selftest(_a):
    torch = Z.torch_()
    nf, per_f, D, Dn = 8, 5, 32, 16
    rng = np.random.default_rng(0)
    cent = np.stack([v / np.linalg.norm(v) for v in rng.standard_normal((nf, Dn))])
    th_f = rng.uniform(1.0, 2.0, (nf, D // 2)) * rng.choice([-1.0, 1.0], (nf, D // 2))
    th_other = rng.uniform(1.0, 2.0, (nf, D // 2)) * rng.choice([-1.0, 1.0], (nf, D // 2))
    cfg = dict(CFG6, batch=256, neg=512, epoch=30_000, max_epochs=30, patience=5, lr=0.02, extra=8, hidden=32, min_op=30, n_dev=100,
               n_read=100, chunk=200)
    # 1. the toy fit (R3's own fold), its stored arrays reloaded: the reloaded net reproduces the fit's recorded ranks (G-L)
    G1, X1, v1 = toy_graph(1, th_f, lambda f, k: f"fam{f}_rel{k}", cent)
    gates1 = {}
    rec1, arr1 = R3.run_fold(torch, G1, X1, v1, cfg, "toy", 0, None, gates1)
    model = model_from_arrays(torch, arr1, hid=cfg["hidden"])
    assert all(np.allclose(model.state_dict()[k].numpy(), arr1[f"model.{k}"]) for k in model.state_dict())
    split = Z.node_split(G1.name, G1.n)
    labs, _o, _o0, _g = Z.plan(G1, split, cfg)
    own = R3.own_labels(labs)
    Nt = torch.from_numpy(np.stack([v1[nm] for _L, nm in own]).astype(np.float32))
    op_of = np.full(G1.nR + G1.RS, -1, np.int64)
    for k, (L, _nm) in enumerate(own):
        op_of[L] = k
    test = {"s": arr1["test_s"], "d": arr1["test_d"], "lab": arr1["test_lab"], "grp": arr1["test_grp"], "nR": G1.nR}
    r, _t = Z.rank_read(torch, R3.NameScorer(model, Nt), X1, test, Z.exclusions(G1, test), op_of)
    near = float((np.abs(r - arr1["rank_name"]) <= np.maximum(1.0, 0.001 * arr1["rank_name"])).mean())
    log(f"self-test G-L: reloaded net reproduces {near:.4f} of {r.size} recorded name ranks")
    assert near >= 0.99
    net = Net("kb", model, rec1["relations"]["train"], mean_of(v1, rec1["relations"]["train"]), "selftest", rec1)
    # 2. the same planted operators on a second graph with NEW relation names of the same families: the name net beats the cosine
    G2, X2, v2 = toy_graph(2, th_f, lambda f, k: f"fam{f}_new{k}", cent)
    vec2 = {**v1, **v2}
    g2 = {}
    import tempfile
    with tempfile.TemporaryDirectory() as td:        # a read with a checkpoint folder, then again from it: the same ranks
        ck2 = Path(td) / "ck"
        rd2, a2 = read_core(torch, G2, X2, vec2, net, "toy2", cfg, ck2, g2)
        n_ck = len(list(ck2.glob("rank_*.npy")))
        rd2b, a2b = read_core(torch, G2, X2, vec2, net, "toy2", cfg, ck2, {})
        assert n_ck > 0 and all(np.array_equal(a2[f"rank_{t}"], a2b[f"rank_{t}"]) for t in SCORERS)
        log(f"self-test: {n_ck} rank chunks checkpointed; the second read took them all and agrees")
    c2 = pooled_call({k: v for k, v in rd2["units"].items() if v["main"]}, min_unit=5)     # toy units have ~14 test edges, not 20
    log(f"self-test toy2 (same planted operators): Delta {fmt(c2['Delta name - cosine'])}; gain {fmt(c2['Gain from the name (name - mean name)'])}; "
        f"call {c2['call']}; means {c2['beside']['mean MRR (points)']}; gates {g2}")
    assert c2["Delta name - cosine"]["call"] == "ABOVE" and c2["Delta name - cosine"]["diff"] > 10, c2
    # 3. a DIFFERENT planted operator under the same names: the net does not transfer (its operator is the first graph's)
    G3, X3, v3 = toy_graph(3, th_other, lambda f, k: f"fam{f}_new{k}", cent)
    g3 = {}
    rd3, _a3 = read_core(torch, G3, X3, {**v1, **v3}, net, "toy3", cfg, None, g3)
    c3 = pooled_call({k: v for k, v in rd3["units"].items() if v["main"]}, min_unit=5)
    log(f"self-test toy3 (other planted operators): Delta {fmt(c3['Delta name - cosine'])}; call {c3['call']}; means {c3['beside']['mean MRR (points)']}")
    # the declared call is an interval call with no size margin (9.6): at the floor it can read ABOVE by a hundredth of a point, so the
    # test is on the size: the name net is at the floor (under one MRR point) and gains under one point over the cosine
    assert c3["beside"]["mean MRR (points)"]["name"] < 1.0 and c3["Delta name - cosine"]["diff"] < 1.0, c3
    # 4. N_P-style fit (every label train, stop on dev edges) on the first toy: G-0 holds, the dev MRR rises above the cosine,
    #    the arrays reload and a read of the same graph reproduces a gain
    gp = {}
    recp, arrp = fit_p(torch, G1, X1, v1, cfg, None, gp, expect=None)
    assert gp["G-0"]["verdict"] == "PASS" and recp["fit"]["best_dev_mrr"] > recp["fit"]["cosine_dev_mrr"] + 5, (gp, recp["fit"])
    netp = Net("p", model_from_arrays(torch, arrp, hid=cfg["hidden"]), recp["train_names"], mean_of(v1, recp["train_names"]), "selftest-p", recp)
    gq = {}
    rdp, _ap = read_core(torch, G1, X1, v1, netp, "toy", cfg, None, gq)
    cp = pooled_call({k: v for k, v in rdp["units"].items() if v["main"]}, min_unit=5)
    log(f"self-test N_P: dev MRR {recp['fit']['best_dev_mrr']} vs cosine {recp['fit']['cosine_dev_mrr']}; read Delta {fmt(cp['Delta name - cosine'])}; "
        f"seen names {gq['G-S']['seen_names']} of {gq['G-S']['units']}")
    assert cp["Delta name - cosine"]["call"] == "ABOVE" and gq["G-S"]["seen_names"] == gq["G-S"]["units"]
    # 5. name strings and the derangement are as R3's
    assert R3.name_string("S.hyperlink") == "hyperlink" and R3.name_string("R.title_comention") == "title comention"
    # 6. the report's rule on stand-in records
    def unit(label, kind, main, c, n, m):
        e = {"label": label, "kind": kind, "main": main, "seen": False, "edges": 100}
        for t, v in (("cosine", c), ("name", n), ("mean", m), ("shuffled", m)):
            e.update({f"mrr_{t}": v, f"h10_{t}": v, f"h100_{t}": v})
        return e

    def make(d2, d3):
        reads = {}
        for ds in PASSAGE:
            reads[("kb", ds)] = {"gate_only": False, "units": {f"{ds}|R{i}": unit(f"R{i}", "R", True, 5.0, 5.0 + d2 + (i - 1) * 0.1, 5.0) for i in range(3)}}
            reads[("p", ds)] = {"gate_only": False, "units": {f"{ds}|R{i}": unit(f"R{i}", "R", True, 5.0, 6.0 + (i - 1) * 0.1, 5.0) for i in range(3)}}
        reads[("kb", "metaqa")] = {"gate_only": False, "units": {f"metaqa|S{i}": unit(f"S{i}", "S", True, 1.0, 1.0, 1.0) for i in range(7)}}
        for ds in KB:
            reads[("p", ds)] = {"gate_only": False, "units": {f"{ds}|S{i}": unit(f"S{i}", "S", True, 2.0, 2.0 + d3 + (i % 4 - 1.5) * 0.05, 2.0) for i in range(12)}}
        reads[("kb", "webqsp")] = {"gate_only": True, "gates": {}}
        return reads

    ok_gate, ok_net = {"verdict": "PASS"}, {"p": {"fit": {}}}
    assert report_core(make(3.0, 3.0), ok_net, ok_gate)["verdict"] == "BRIDGES"
    assert report_core(make(3.0, 0.0), ok_net, ok_gate)["verdict"] == "ONE_WAY (KB -> passage)"
    assert report_core(make(0.0, 3.0), ok_net, ok_gate)["verdict"] == "ONE_WAY (passage -> KB)"
    assert report_core(make(0.0, 0.0), ok_net, ok_gate)["verdict"] == "NOT_BRIDGED"
    rr = make(3.0, 3.0)
    del rr[("p", "2wiki")]
    assert report_core(rr, ok_net, ok_gate)["directions"][DIRECTIONS[3][0]]["call"] == "INCOMPLETE"
    assert report_core(make(3.0, 3.0), ok_net, None)["verdict"] == "INCOMPLETE"
    log("self-test PASS: a stored net reloads and reproduces the fit's own ranks; the same planted operators transfer to new names, "
        "a different planted operator does not; N_P's fit holds G-0 and learns; the report's rule gives BRIDGES / ONE_WAY / NOT_BRIDGED / INCOMPLETE")
    return 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("selftest", "names", "fit", "read", "report"))
    p.add_argument("--net", choices=("kb", "p"))
    p.add_argument("--ds", choices=Z.SEVEN)
    p.add_argument("--host", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--smoke", action="store_true", help="a one-epoch fit or a few edges per unit; written under outputs/sem2/r6/smoke/, read by nothing")
    a = p.parse_args(argv)
    if a.cmd == "selftest":
        return selftest(a)
    if a.cmd == "report":
        return cmd_report(a)
    if a.cmd == "names":
        return cmd_names(a)
    if a.cmd == "fit":
        if not a.net:
            stop("fit needs --net")
        return cmd_fit(a)
    return cmd_read(a)


if __name__ == "__main__":
    sys.exit(main())
