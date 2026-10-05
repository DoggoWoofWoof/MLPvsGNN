"""Design look (untracked; not a result and not filed): S6 part 12, the zero-shot gap against directly trained
counterparts. Swastik (5 Oct): the MLP and the GNN use the same features in two ways (the MLP scores them as static
inputs and never passes a message; the GNN passes messages over them), so both should improve, each within its own
constraints; 'rn the problem is generalizability so we need to see how we can take and influence the features and
weights to their directly trained counterparts and isolate what exactly is the issue, use all the info we have, shapley
etc'. Parts 1 to 11 trained on metaqa (and its transforms, and passage graphs) and read webqsp zero-shot; none asked
what a model of the same class reaches when webqsp itself trains it, or where the zero-shot model differs from that
one. This part does, for the MLP f and both GNN fs, as a diagnostic.

Diagnostic oracles, never zero-shot models. The counterparts train on webqsp's fit carve (fitf), so no number of
theirs is a transfer number and none enters a zero-shot comparison as a model; they are references, like twin0.
webqsp's train_holdout and test are not touched: the counterparts read webqsp selectf only (never trained on), and the
fit carve's rows (1,202 in cs19-wq-sf.npz, carve 'fitf') are split once by a seeded permutation (SEED + 28) into fit-A
(85%, the training rows) and fit-B (15%, the epoch choice).
Arms (f, training graphs), lb and part 6's carried map (pq) on every arm, three seeds each:
    c-mlp   the MLP f (chainscore20's em-gr) on webqsp fit-A
    c-en    the untyped GNN f (en-gr) on webqsp fit-A
    c-ena   the posterior-typed GNN f (ena-gr) on webqsp fit-A
    p-mlp   the MLP f on metaqa's k5 mix (fit id and its four transforms, one group) and webqsp fit-A (a second group;
            chainscore21's draw: the group uniform per example)
    p-en    the untyped GNN f on the same pooled mix
The zero-shot models compared (not retrained; read from their saved weights, checked by state sha256): part 8's k5
(cs24-k5-s0..2, the MLP f), part 9's n-b-lb-pq (cs25-n-b-lb-pq-s0..2, en-gr) and e-b-lb-pq (cs25-e-b-lb-pq-s0..2,
ena-gr), each paired with the counterpart of its f, seed k with seed k.
Training as part 9 (cs_dev's settings on the host GPU: deterministic algorithms, TF32 off; Adam 1e-3, weight decay
1e-4, 12 epochs of 5,960 examples in batches of 64; seeds SEED + 0, 1, 2), but for the epoch choice: R@5 on webqsp
fit-B (c-*), or the mean of metaqa's select carve and webqsp fit-B (p-*). Inputs from cs_cache: webqsp selectf + fitf
under pq (kind train: prep, lb, the map over its whole population, and its input stats, which standardise the c-*
arms; the p-* arms take metaqa fit id's, as every zero-shot run did). The GNN fs read the rows' messages from part 4's
companions (cs20e-*, checked against each build's sha256). Reads in the run: the KB rule on metaqa x1f and webqsp
selectf.
Diagnostics (diag, per f; every read on webqsp selectf, on one device, inside one job). Components of a model: f (the
node scorer: the MLP, or the GNN but its attention parameters), g (the chain scorer), cpl (the EM coupling: beta, gamma,
eps, and ena's attention theta and u) and st (the input stats that standardise every input). For seed k:
    E2 component Shapley: the 16 models taking each subset of components from the counterpart and the rest from the
       zero-shot model (the empty set is the zero-shot model, the full set the counterpart); exact Shapley values of
       the four components over R@5 per row; by efficiency they sum to the counterpart's lead
    E3 feature-group Shapley, for the zero-shot model and for the counterpart: node groups text (cq, cs, rr), dist (d1,
       d2, d3, dinf), struct (deg, nseed1, qnb) and walk (w1, w2, w3), with the pair inputs intact; pair groups text
       (cos_e, cos_nm, has_nm, cos_rd, cos_s), shape (hop1-3, fwd, k_none, k_dir, k_tt, k_exact, logC, logN), evidence
       (reach, ment, mmax, minrank, avail, thick) and pop (mu_*, lr_*, lpi_*, ent_*), with the node inputs intact; a
       group left out is set to 0 after standardisation (its training mean under that model's stats); exact Shapley
       over the four groups of each kind; and the counterpart's value minus the zero-shot model's per group
    E4 label-free re-standardisation: the zero-shot model (and the counterpart) with st computed on webqsp selectf
       itself (its mapped inputs; no label), minus the model as trained
    E6 the shift table: per node and pair input, standardised by the zero-shot model's stats, its mean and std and the
       AUC of gold against other pool nodes (chains: on, chainscore18's label, against the rest) on metaqa fit id and on
       webqsp selectf
    A  score agreement: per row, the Spearman correlation of the zero-shot model's and the counterpart's node scores
Grade (every arm x seed, the zero-shot runs' rows files, and the three diag outputs). Each row's R@5 averaged over its
three seeds; paired row bootstrap (BOOT 1000):
    E1 webqsp selectf: c-* minus the zero-shot model of its f (the counterpart's lead), p-* minus c-* (does pooling
       cost webqsp), and on metaqa x1f: p-* minus the zero-shot model (does pooling cost metaqa); c-* on metaqa x1f
       minus the zero-shot model there (the reverse transfer)
    E2 to E6 and A as above, averaged over seeds before the Shapley sums (they are linear in the values)
    S  every E1 difference per seed, with its sign
Decision, fixed at 10:10 on 5 Oct (this file's first write), before any part 12 run. Per f:
    CAPACITY  the counterpart's lead on webqsp selectf below 0.03 R@5 (or its CI includes 0): the class and its
              features cap webqsp; transfer is not the binding gap, and the next part improves the class in-domain
    SHARED    a lead of at least 0.03, p-* not BELOW c-* on webqsp by more than 0.03 and not BELOW the zero-shot model
              on metaqa by more than 0.02 (MLP f and en-gr; ena-gr takes en-gr's pooled reading): one function of
              these features serves both graphs, and the zero-shot gap is which function to use on an unseen graph:
              the next part tries label-free read-time adaptation and invariant inputs
    CONFLICT  a lead of at least 0.03 and pooling costs either graph beyond those margins: the graphs ask different
              functions of the same features; the next part conditions the components on the graph's population
              (typed messages for the GNN f, population-conditioned inputs for the MLP f)
    and, whatever the label, the component carrying the largest E2 share of the lead, E4 (HELPS / HURTS / SAME) and
    the E3 groups whose counterpart-minus-zero-shot CI excludes 0 name where the next part acts.
Train-split rows throughout (webqsp's fit and selectf carves are the train split's): a look, not a result.
    python outputs/mp_unified/cs_cache.py build --kind train --map pq --src outputs/mp_unified/lean/cs19-wq-sf.npz
    python outputs/mp_unified/chainscore28.py train --arm c-ena --seed 0 --device cuda \
        --map-from outputs/mp_unified/lean/cs22.json --cache outputs/mp_unified/cache \
        --wqfit outputs/mp_unified/lean/cs19-wq-sf.npz [--fit ...cs19-mq-fit-id.npz --aug al4=... (p-*)] \
        [--select outputs/mp_unified/lean/cs19-mq-select.npz (p-*)] --read metaqa=... --read webqsp=... \
        --edges wqfit=...cs20e-wq-sf.npz (GNN fs: every input's companion) --out X.json --rows-out X.rows.npz \
        --state-out X.pt
    python outputs/mp_unified/chainscore28.py diag --f ena-gr --z Z0.json --z Z1.json --z Z2.json --c C0.json ... \
        --read webqsp=...cs19-wq.npz --ref metaqa=...cs19-mq-fit-id.npz [--edges webqsp=...] --cache ... \
        --device cuda --out D.json
    python outputs/mp_unified/chainscore28.py grade --run X.json (every arm x seed) --z Z.json (every zero-shot run)
        --diag D.json (each f) --out outputs/mp_unified/lean/cs28.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore22 as C22  # noqa: E402
import chainscore24 as C24  # noqa: E402
import chainscore25 as C25  # noqa: E402
import cs_cache as CC  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
BATCH = C21.BATCH
SEEDS = (0, 1, 2)
TF4 = C19.TRANSFORMS[1:]
HOLD_B = 0.15
ARMS = {"c-mlp": ("em-gr", "wq"), "c-en": ("en-gr", "wq"), "c-ena": ("ena-gr", "wq"),
        "p-mlp": ("em-gr", "pool"), "p-en": ("en-gr", "pool")}
ZSTEM = {"em-gr": "cs24-k5", "en-gr": "cs25-n-b-lb-pq", "ena-gr": "cs25-e-b-lb-pq"}
CSTEM = {"em-gr": "c-mlp", "en-gr": "c-en", "ena-gr": "c-ena"}
PSTEM = {"em-gr": "p-mlp", "en-gr": "p-en", "ena-gr": "p-en"}
COMPS = ("f", "g", "cpl", "st")
NODE_GROUPS = {"text": ("cq", "cs", "rr"), "dist": ("d1", "d2", "d3", "dinf"), "struct": ("deg", "nseed1", "qnb"),
               "walk": ("w1", "w2", "w3")}
PAIR_GROUPS = {"text": ("cos_e", "cos_nm", "has_nm", "cos_rd", "cos_s"),
               "shape": ("hop1", "hop2", "hop3", "fwd", "k_none", "k_dir", "k_tt", "k_exact", "logC", "logN"),
               "evidence": ("reach", "ment", "mmax", "minrank", "avail", "thick"),
               "pop": ("mu_pop", "mu_hpop", "mu_nm1", "lr_pop", "lr_pop1", "lr_rd", "lr_hpop", "lr_nm1", "lr_nm",
                       "lpi_pop", "lpi_nm1", "ent_pop", "ent_nm1")}
CAP_LEAD, POOL_WQ, POOL_MQ = 0.03, 0.03, 0.02
SHIFT_SAMPLE = 400_000
log, sha = C21.log, CC.sha


def _check_groups():
    nf = sorted(x for g in NODE_GROUPS.values() for x in g)
    pf = sorted(x for g in PAIR_GROUPS.values() for x in g)
    if nf != sorted(C19.NODEF) or pf != sorted(C19.PAIRF):
        raise SystemExit("the feature groups do not cover NODEF / PAIRF exactly once")


_check_groups()
NCOLS = {k: [C19.NODEF.index(x) for x in v] for k, v in NODE_GROUPS.items()}
PCOLS = {k: [C19.PAIRF.index(x) for x in v] for k, v in PAIR_GROUPS.items()}


# ── the fit carve's split ───────────────────────────────────────────────────────────────────────────────────────


def fit_split(d):
    """(fit-A rows, fit-B rows) of a selectf + fitf build: its fitf rows, permuted once by SEED + 28."""
    carves = list(d["meta"]["carves"])
    if "fitf" not in carves:
        raise SystemExit(f"the webqsp training build has carves {carves}, no fitf")
    rows = np.flatnonzero(d["carve_ix"] == carves.index("fitf")).astype(np.int64)
    perm = rows[np.random.default_rng(SEED + 28).permutation(rows.size)]
    nb = int(round(HOLD_B * rows.size))
    return np.sort(perm[nb:]), np.sort(perm[:nb])


# ── selection and reading over a row subset ─────────────────────────────────────────────────────────────────────


def row_batches(d, c, lb, rows, st, spec, batch=BATCH):
    """make_batch21's read batches of the given rows (the KB rule's inputs), made once on the CPU."""
    out = []
    for b0 in range(0, rows.size, batch):
        r = rows[b0:b0 + batch]
        items = np.c_[np.zeros(r.size, np.int64), r, np.zeros(r.size, np.int64)]
        out.append((r, C21.make_batch21(items, [d], [c], None if lb is None else [lb], st, pairs=spec["g"],
                                        train=False, gnn=spec["gnn"], att=spec["att"])))
    return out


def forward_rows(model, d, bl, device, zn=(), zp=(), keep=False):
    """KB-rule (R@5, FC@5, hit@1) per row of the batches (indexed by the build's rows), with the node columns zn and
    the pair columns zp set to 0 after standardisation; with keep, each row's node scores too."""
    import torch
    out = np.full((d["n"].size, 3), np.nan)
    sc = {}
    with torch.no_grad():
        for rows, bt in bl:
            b = dict(bt)
            if zn:
                b["Xn"] = bt["Xn"].clone()
                b["Xn"][:, list(zn)] = 0.0
            if zp and bt.get("Xp") is not None:
                b["Xp"] = bt["Xp"].clone()
                b["Xp"][:, list(zp)] = 0.0
            if CD.placed(device):
                with torch.device(device):
                    s, _lp = C20.forward20(model, CD.to_dev(b, device))
            else:
                s, _lp = C20.forward20(model, b)
            s = s.double().cpu().numpy()
            for j, i in enumerate(rows):
                si = s[j, :int(d["n"][i])]
                out[i] = C19.rank_row(d, i, si)
                if keep:
                    sc[int(i)] = si.copy()
    return (out, sc) if keep else out


# ── training ────────────────────────────────────────────────────────────────────────────────────────────────────


def train28(arm, f, builds, comps, LB, groups, sels, st, epochs, per_epoch, batch, seed, device=None, threads=1):
    """chainscore25's train25 (cs_dev's train_dev statements) for any f, but for the epoch choice: the mean over the
    selection sets (each a build's rows, read by the KB rule) of R@5. st: the first build's input stats."""
    import torch
    block = CD.settings(device, threads)
    spec = C21.arm_spec("b-lb", f)
    if not (spec["g"] and spec["lb"]):
        raise SystemExit("this part trains f with g and lb")
    model = C20.make_model(f, seed)
    if CD.placed(device):
        CD._dp().model_to(model, device)
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    sb = [(sd, rows, row_batches(sd, sc_, slb, rows, st, spec)) for sd, sc_, slb, rows in sels]
    hist, best, best_state = [], None, None
    tb, nbt = 0.0, 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ep in range(epochs):
            t1 = time.time()
            model.train()
            items = C21.draw21(rng, builds, groups, per_epoch)
            tot, tn, tc, nb = 0.0, 0.0, 0.0, 0
            for b0 in range(0, len(items), batch):
                t2 = time.time()
                bt = C21.make_batch21(items[b0:b0 + batch], builds, comps, LB, st, pairs=spec["g"], gnn=spec["gnn"],
                                      att=spec["att"])
                if CD.placed(device):
                    bt = CD.to_dev(bt, device)
                    with torch.device(device):
                        s, lp = C20.forward20(model, bt)
                        loss, ln, lc = C19.objective(s, lp, bt)
                else:
                    s, lp = C20.forward20(model, bt)
                    loss, ln, lc = C19.objective(s, lp, bt)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss.detach())
                tn += ln
                tc += lc
                nb += 1
                tb += time.time() - t2
                nbt += 1
            model.eval()
            parts = [float(forward_rows(model, sd, bl, device)[rows, 0].mean()) for sd, rows, bl in sb]
            sr5 = float(np.mean(parts))
            hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                         "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                         "select_parts": [round(x, 5) for x in parts],
                         "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(),
                         **C20.coupling20(model), "seconds": round(time.time() - t1, 1)})
            log(f"    {arm} ({f}) ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
                f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {[round(x, 4) for x in parts]} {time.time() - t1:.0f}s")
            if best is None or sr5 > best[1]:
                best = (ep, sr5)
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    det = sorted({str(w.message)[:300] for w in caught if "deterministic" in str(w.message).lower()})
    model.load_state_dict(best_state)
    model.eval()
    return model, {"history": hist, "best_epoch": best[0], "best_select_r5": round(best[1], 5), "stats": st,
                   "coupling": C20.coupling20(model), "spec": spec, "placement": block,
                   "state_sha256": CD.state_sha(best_state), "seconds_per_batch": round(tb / max(nbt, 1), 4),
                   "deterministic_warnings": det}


def train_cmd(a):
    t0 = time.time()
    if a.arm not in ARMS:
        raise SystemExit(f"unknown arm {a.arm}; one of {list(ARMS)}")
    f, data = ARMS[a.arm]
    if a.map_from:
        minfo = C25.carried_map(a.map_from)
        norm = "pq"
    elif a.smoke:
        norm, minfo = a.map, {"map_given": a.map}
    else:
        raise SystemExit("give --map-from (part 6's grade; the runs) or --smoke --map (a local smoke)")
    dev = None if a.device == "cpu" else a.device
    spec = C21.arm_spec("b-lb", f)
    res = {"look": "chainscore28", "arm": a.arm, "seed": a.seed, "f": f, "data": data, "norm": norm,
           "device": a.device, "args": dict(vars(a)), "map_rule": minfo, "cache": a.cache, "smoke": bool(a.smoke),
           "state_out": a.state_out, "script_sha256": sha(__file__), "cs_cache_sha256": sha(CC.__file__),
           "chainscore25_sha256": sha(C25.__file__), "chainscore24_sha256": sha(C24.__file__),
           "chainscore22_sha256": sha(C22.__file__), "chainscore21_sha256": sha(C21.__file__),
           "chainscore20_sha256": sha(C20.__file__), "chainscore19_sha256": sha(C19.__file__),
           "chainscore18_sha256": sha(CS.__file__), "cs_dev_sha256": sha(CD.__file__), "inputs": {},
           "inputs_sha256": {}, "maps": {}, "edges": {}, "reads": {}, "rows_out": a.rows_out, "timing": {}}
    aug = dict(x.split("=", 1) for x in a.aug)
    if data == "pool":
        if sorted(aug) != sorted(TF4) or not a.fit or not a.select:
            raise SystemExit(f"a pooled arm takes --fit, --select and --aug for each of {TF4}")
    elif aug or a.fit or a.select:
        raise SystemExit("a counterpart arm trains on webqsp fit-A alone: no --fit, --aug or --select")
    if not a.wqfit:
        raise SystemExit("every arm takes --wqfit (webqsp selectf + fitf)")
    reads = [tuple(x.split("=", 1)) for x in a.read]
    if not a.smoke and sorted(n for n, _ in reads) != ["metaqa", "webqsp"]:
        raise SystemExit("every arm reads metaqa (x1f) and webqsp (selectf)")
    train_in = ([("fit", a.fit)] + [(t, aug[t]) for t in TF4] if data == "pool" else []) + [("wqfit", a.wqfit)]
    path = dict(train_in)
    if data == "pool":
        path["select"] = a.select
    path.update(dict(reads))
    edges = dict(x.split("=", 1) for x in a.edges)
    if spec["gnn"] and sorted(edges) != sorted(path):
        raise SystemExit(f"a GNN f takes --edges for each of {sorted(path)}; given {sorted(edges)}")
    if not spec["gnn"] and edges:
        raise SystemExit("the MLP f takes no --edges")
    t1 = time.time()
    for n, p in path.items():
        res["inputs_sha256"][p] = sha(p)
        if spec["gnn"]:
            res["inputs_sha256"][edges[n]] = sha(edges[n])
    res["timing"]["hash_s"] = round(time.time() - t1, 1)

    def load(p, kind):
        if a.cache:
            return C25.get(p, kind, norm, a.cache)
        d, lb, info = CC.pipeline(p, kind, norm)
        st = C19.input_stats(d) if kind == "train" else None
        return d, lb, info, st

    def comp(n, d):
        return C25.load_comp(edges[n], d, res["inputs_sha256"][path[n]], spec["att"]) if spec["gnn"] else None

    t1 = time.time()
    builds, comps, LB, st0 = [], [], [], None
    for n, p in train_in:
        d, lb, info, st = load(p, "train")
        tf = d["meta"]["transform"]["name"]
        if tf != (n if n in TF4 else "id"):
            raise SystemExit(f"{p} holds transform {tf}")
        if n == "wqfit":
            if d["meta"]["args"]["ds"] != "webqsp" and not a.smoke:
                raise SystemExit(f"{p} is not a webqsp build")
            if a.smoke and "fitf" not in list(d["meta"]["carves"]):
                rows = np.arange(d["n"].size, dtype=np.int64)       # a smoke on selectf: its rows stand in
                perm = rows[np.random.default_rng(SEED + 28).permutation(rows.size)]
                nb_ = int(round(HOLD_B * rows.size))
                fa, fb = np.sort(perm[nb_:]), np.sort(perm[:nb_])
            else:
                fa, fb = fit_split(d)
            tr0 = d["train_rows"]
            d["train_rows"] = np.intersect1d(tr0, fa)
            res["wq_split"] = {"fit_a": int(fa.size), "fit_b": int(fb.size), "train_rows_a": int(d["train_rows"].size),
                               "fit_b_sha256": CC.arr_sha(fb)}
            wq = (d, lb, fb)
        if not builds:
            st0 = st
        res["inputs"][n] = d["meta"]
        res["maps"][n] = info
        builds.append(d)
        comps.append(comp(n, d))
        LB.append(lb)
        log(f"  {a.arm} s{a.seed}: {n} ready ({'cache' if a.cache else 'pipeline'}; {info})")
    if st0 is None:
        raise SystemExit("no input stats for the first build")
    nkb = len(builds) - 1
    groups = [list(range(nkb)), [nkb]] if data == "pool" else [[0]]
    sels = []
    if data == "pool":
        Sd, Slb, sinfo, _ = load(a.select, "plain")
        sels.append((Sd, comp("select", Sd), Slb, np.arange(Sd["n"].size, dtype=np.int64)))
        res["maps"]["select"] = sinfo
        res["inputs"]["select"] = Sd["meta"]
    wd, wlb, fb = wq
    sels.append((wd, comps[-1], wlb, fb))
    res["groups"] = groups
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p), b in zip(train_in, builds)}
    if spec["gnn"]:
        res["edges"] = {n: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                        for (n, _p), c in zip(train_in, comps)}
    res["timing"]["load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    model, info = train28(a.arm, f, builds, comps, LB, groups, sels, st0, a.epochs, a.per_epoch, a.batch,
                          SEED + a.seed, device=dev)
    res["timing"]["train_s"] = round(time.time() - t1, 1)
    del builds, comps, LB, sels, wq, wd
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings")})
    t1 = time.time()
    rows_out = {}
    for n, p in reads:
        t2 = time.time()
        d, lb, rinfo, _ = load(p, "plain")
        c = comp(n, d)
        res["maps"][n] = rinfo
        xr, ml = CD.read_dev(model, d, c, lb, info["stats"], "kb", dev, batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule="kb",
                               seconds=round(time.time() - t2, 1))
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} s{a.seed} on {n} (kb rule): {res['reads'][n]['mean']} ({time.time() - t2:.0f}s)")
        del d, c
    res["timing"]["reads_s"] = round(time.time() - t1, 1)
    p = Path(a.rows_out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows_out)
    os.replace(tmp, p)
    if a.state_out:
        import torch
        sp_ = Path(a.state_out)
        torch.save({"state": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "stats": info["stats"],
                    "spec": info["spec"], "arm": a.arm, "seed": a.seed, "f": f, "norm": norm},
                   sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  {a.arm} s{a.seed}: state {res['state_sha256'][:16]}, {res['seconds_per_batch']} s/batch, timing "
        f"{res['timing']}, {res['seconds']}s")
    return res


# ── diagnostics ─────────────────────────────────────────────────────────────────────────────────────────────────


def comp_of(key):
    if key in ("beta_r", "gamma_r", "leps") or key.startswith("gnn.theta") or key.startswith("gnn.u"):
        return "cpl"
    if key.startswith("g."):
        return "g"
    if key.startswith("f.") or key.startswith("gnn."):
        return "f"
    raise SystemExit(f"no component for {key}")


def load_run(path, f):
    """(run json, saved weights): the weights checked against the run's state sha256, f and map pq."""
    import torch
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    sp_ = js.get("state_out") or (js.get("args") or {}).get("state_out")
    if not sp_:
        raise SystemExit(f"{path}: no saved weights")
    S = torch.load(sp_, map_location="cpu", weights_only=False)
    got = CD.state_sha(S["state"])
    if got != js["state_sha256"]:
        raise SystemExit(f"{sp_}: weights {got[:16]}, the run's {js['state_sha256'][:16]}")
    if js.get("f") != f or js.get("norm") != "pq":
        raise SystemExit(f"{path}: f {js.get('f')} map {js.get('norm')}, not {f} under pq")
    return js, S


def build_model(f, state, device):
    m = C20.make_model(f, 0)
    m.load_state_dict(state)
    if CD.placed(device):
        CD._dp().model_to(m, device)
    m.eval()
    return m


def shapley(V, n):
    """Exact Shapley values per row: V maps a player mask (bit i: player i present) to per-row values."""
    N = V[0].shape[0]
    phi = np.zeros((n, N))
    for i in range(n):
        for S in range(1 << n):
            if S >> i & 1:
                continue
            k = bin(S).count("1")
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            phi[i] += w * (V[S | 1 << i] - V[S])
    return phi


def boot_ci(x, rng):
    x = np.asarray(x, np.float64)
    idx = rng.integers(x.size, size=(BOOT, x.size))
    m = x[idx].mean(1)
    return [round(float(x.mean()), 5), round(float(np.quantile(m, 0.025)), 5), round(float(np.quantile(m, 0.975)), 5)]


def avg_rank(x):
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt)
    return (cum - (cnt - 1) / 2.0)[inv]


def auc(x, y):
    y = np.asarray(y, bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    r = avg_rank(np.asarray(x, np.float64))
    return round(float((r[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)), 4)


def spearman_rows(sz, sc):
    out = []
    for i, a in sz.items():
        b = sc[i]
        if a.size < 3:
            continue
        ra, rb = avg_rank(a), avg_rank(b)
        sa, sb_ = ra.std(), rb.std()
        if sa > 0 and sb_ > 0:
            out.append(float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb_)))
    return out


def shift_table(dref, dread, st, rng):
    """Per input, standardised by st: mean, std and AUC (gold pool nodes / on chains against the rest) on the
    reference build (metaqa fit id) and the read build (webqsp selectf)."""
    out = {"node": {}, "pair": {}}
    for tag, d in (("ref", dref), ("read", dread)):
        d = C19.prep(dict(d)) if "gflag" not in d else d
        Xn = d["XN"]
        keep = ~d["seed0"]
        ix = np.flatnonzero(keep)
        if ix.size > SHIFT_SAMPLE:
            ix = np.sort(rng.choice(ix, SHIFT_SAMPLE, replace=False))
        X = (Xn[ix].astype(np.float64) - st["mn"]) / st["sn"]
        y = d["gflag"][ix]
        for j, nm in enumerate(C19.NODEF):
            out["node"].setdefault(nm, {})[tag] = {"mean": round(float(X[:, j].mean()), 4),
                                                   "std": round(float(X[:, j].std()), 4), "auc": auc(X[:, j], y)}
        P = int(d["rowoff"][-1])
        pix = np.arange(P) if P <= SHIFT_SAMPLE else np.sort(rng.choice(P, SHIFT_SAMPLE, replace=False))
        Xp = (C19.pair_raw(d, 0, 0, P)[pix].astype(np.float64) - st["mp"]) / st["sp"]
        yp = d["on"][pix]
        for j, nm in enumerate(C19.PAIRF):
            out["pair"].setdefault(nm, {})[tag] = {"mean": round(float(Xp[:, j].mean()), 4),
                                                   "std": round(float(Xp[:, j].std()), 4), "auc": auc(Xp[:, j], yp)}
    for kind in ("node", "pair"):
        for nm, v in out[kind].items():
            r, q = v["ref"], v["read"]
            v["mean_shift"] = round(q["mean"] - r["mean"], 4)
            v["auc_flip"] = (r["auc"] is not None and q["auc"] is not None and (r["auc"] - 0.5) * (q["auc"] - 0.5) < 0)
    return out


def diag_cmd(a):
    import torch
    t0 = time.time()
    f = a.f
    if f not in ZSTEM:
        raise SystemExit(f"--f is one of {list(ZSTEM)}")
    if len(a.z) != len(a.c):
        raise SystemExit("one counterpart run per zero-shot run (seed k with seed k)")
    dev = None if a.device == "cpu" else a.device
    block = CD.settings(dev, a.threads)
    spec = C21.arm_spec("b-lb", f)
    norm = "pq"
    (rn, rp), = [tuple(x.split("=", 1)) for x in a.read]
    edges = dict(x.split("=", 1) for x in a.edges)
    if spec["gnn"] and sorted(edges) != [rn]:
        raise SystemExit(f"a GNN f takes --edges {rn}=...")
    res = {"look": "chainscore28-diag", "f": f, "read": rn, "device": a.device, "placement": block,
           "args": dict(vars(a)), "script_sha256": sha(__file__), "inputs_sha256": {rp: sha(rp)}, "pairs": [],
           "comps": list(COMPS), "node_groups": NODE_GROUPS, "pair_groups": PAIR_GROUPS}
    d, lb, rinfo, _ = (C25.get(rp, "plain", norm, a.cache) if a.cache else (*CC.pipeline(rp, "plain", norm), None))
    c = C25.load_comp(edges[rn], d, res["inputs_sha256"][rp], spec["att"]) if spec["gnn"] else None
    st_read = C19.input_stats(d)
    res["map"] = rinfo
    N = d["n"].size
    rows = np.arange(N, dtype=np.int64)
    V_comp = {m: [] for m in range(1 << len(COMPS))}
    V_node = {"z": {m: [] for m in range(16)}, "c": {m: [] for m in range(16)}}
    V_pair = {"z": {m: [] for m in range(16)}, "c": {m: [] for m in range(16)}}
    V_rest = {"z_stread": [], "c_stread": []}
    agree = []
    nk, pk = list(NODE_GROUPS), list(PAIR_GROUPS)
    for zp_, cp_ in zip(a.z, a.c):
        t1 = time.time()
        zj, ZS = load_run(zp_, f)
        cj, CSt = load_run(cp_, f)
        if int(zj["seed"]) != int(cj["seed"]):
            raise SystemExit(f"{zp_} seed {zj['seed']} paired with {cp_} seed {cj['seed']}")
        res["pairs"].append({"z": zp_, "c": cp_, "seed": int(zj["seed"]), "z_state": zj["state_sha256"],
                             "c_state": cj["state_sha256"]})
        bl = {"z": row_batches(d, c, lb, rows, ZS["stats"], spec), "c": row_batches(d, c, lb, rows, CSt["stats"], spec),
              "r": row_batches(d, c, lb, rows, st_read, spec)}
        zs, cs_ = ZS["state"], CSt["state"]
        if sorted(zs) != sorted(cs_):
            raise SystemExit("the zero-shot model and its counterpart differ in their parameters")
        for m in range(1 << len(COMPS)):
            take = {COMPS[i] for i in range(len(COMPS)) if m >> i & 1}
            state = {k: (cs_[k] if comp_of(k) in take else zs[k]) for k in zs}
            model = build_model(f, state, dev)
            x = forward_rows(model, d, bl["c" if "st" in take else "z"], dev)
            V_comp[m].append(x[:, 0])
            del model
        mz, mc = build_model(f, zs, dev), build_model(f, cs_, dev)
        xz, sz = forward_rows(mz, d, bl["z"], dev, keep=True)
        xc, scc = forward_rows(mc, d, bl["c"], dev, keep=True)
        if np.abs(xz[:, 0] - V_comp[0][-1]).max() > 1e-9 or np.abs(xc[:, 0] - V_comp[15][-1]).max() > 1e-9:
            raise SystemExit("the empty and full component sets are not the two models")
        agree.extend(spearman_rows(sz, scc))
        for who, mdl, key in (("z", mz, "z"), ("c", mc, "c")):
            for m in range(16):
                zn = [j for i, g in enumerate(nk) if not m >> i & 1 for j in NCOLS[g]]
                V_node[who][m].append(forward_rows(mdl, d, bl[key], dev, zn=zn)[:, 0])
                zpc = [j for i, g in enumerate(pk) if not m >> i & 1 for j in PCOLS[g]]
                V_pair[who][m].append(forward_rows(mdl, d, bl[key], dev, zp=zpc)[:, 0])
            V_rest[f"{who}_stread"].append(forward_rows(mdl, d, bl["r"], dev)[:, 0])
        log(f"  diag {f} seed {zj['seed']}: Z {xz[:, 0].mean():.4f} C {xc[:, 0].mean():.4f} "
            f"({time.time() - t1:.0f}s)")
        del mz, mc, bl
    rng = np.random.default_rng(SEED + 28)
    Vc = {m: np.mean(v, 0) for m, v in V_comp.items()}
    phi = shapley(Vc, len(COMPS))
    lead = Vc[15] - Vc[0]
    res["E2"] = {"z": boot_ci(Vc[0], rng), "c": boot_ci(Vc[15], rng), "lead": boot_ci(lead, rng),
                 "phi": {COMPS[i]: boot_ci(phi[i], rng) for i in range(len(COMPS))},
                 "share": {COMPS[i]: (round(float(phi[i].mean() / lead.mean()), 4) if abs(lead.mean()) > 1e-9
                                      else None) for i in range(len(COMPS))},
                 "values": {"+".join(COMPS[i] for i in range(len(COMPS)) if m >> i & 1) or "none":
                            round(float(Vc[m].mean()), 5) for m in Vc}}
    res["E3"] = {}
    for kind, V, names in (("node", V_node, nk), ("pair", V_pair, pk)):
        ph = {}
        for who in ("z", "c"):
            Vm = {m: np.mean(v, 0) for m, v in V[who].items()}
            ph[who] = shapley(Vm, 4)
            res["E3"].setdefault(kind, {})[who] = {
                "phi": {g: boot_ci(ph[who][i], rng) for i, g in enumerate(names)},
                "none": round(float(Vm[0].mean()), 5), "all": round(float(Vm[15].mean()), 5)}
        res["E3"][kind]["c_minus_z"] = {g: boot_ci(ph["c"][i] - ph["z"][i], rng) for i, g in enumerate(names)}
    zr, cr = np.mean(V_rest["z_stread"], 0), np.mean(V_rest["c_stread"], 0)
    e4z, e4c = boot_ci(zr - Vc[0], rng), boot_ci(cr - Vc[15], rng)
    res["E4"] = {"z": e4z, "z_label": verdict(e4z), "c": e4c, "c_label": verdict(e4c)}
    res["A"] = {"spearman_mean": round(float(np.mean(agree)), 4) if agree else None,
                "spearman_q": [round(float(q), 4) for q in np.quantile(agree, [0.1, 0.5, 0.9])] if agree else None,
                "rows": len(agree)}
    if a.ref:
        t1 = time.time()
        (_rfn, rfp), = [tuple(x.split("=", 1)) for x in a.ref]
        dref, _l, _i, _s = (C25.get(rfp, "train", norm, a.cache) if a.cache else
                            (*CC.pipeline(rfp, "train", norm), None))
        _zj, ZS0 = load_run(a.z[0], f)
        res["E6"] = shift_table(dref, d, ZS0["stats"], np.random.default_rng(SEED + 28))
        res["inputs_sha256"][rfp] = sha(rfp)
        res["E6_seconds"] = round(time.time() - t1, 1)
        del dref
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  diag {f}: E2 lead {res['E2']['lead']} shares {res['E2']['share']}, E4 {res['E4']['z_label']}, "
        f"{res['seconds']}s")
    return res


def verdict(ci, pos="HELPS", neg="HURTS", zero="SAME"):
    return pos if ci[1] > 0 else neg if ci[2] < 0 else zero


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def stem_seed(js):
    look = js.get("look")
    if look == "chainscore28":
        return js["arm"], int(js["seed"])
    if look == "chainscore24" and js.get("arm") == "k5":
        return "cs24-k5", int(js["seed"])
    if look == "chainscore25" and js.get("arm") in ("n-b-lb-pq", "e-b-lb-pq") and not js.get("via_dev"):
        return f"cs25-{js['arm']}", int(js["seed"])
    raise SystemExit(f"not a run this grade reads: {look} {js.get('arm')}")


def grade_cmd(a):
    t0 = time.time()
    runs, rows = {}, {}
    for p_ in list(a.run) + list(a.z):
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        st_, s = stem_seed(js)
        if js.get("norm") != "pq":
            raise SystemExit(f"{p_}: map {js.get('norm')}, not pq")
        runs[(st_, s)] = {"path": p_, "sha256": sha(p_), "state_sha256": js.get("state_sha256"),
                          "best_epoch": js.get("best_epoch")}
        R = np.load(js["rows_out"])
        rows[(st_, s)] = {k: R[k][:, 0].astype(np.float64) for k in ("metaqa", "webqsp")}
    need = [(x, s) for x in list(ARMS) + list(ZSTEM.values()) for s in SEEDS]
    miss = [k for k in need if k not in rows]
    if miss:
        raise SystemExit(f"missing runs: {miss}")
    rng = np.random.default_rng(SEED + 28)

    def mean_rows(stem, rd):
        return np.mean([rows[(stem, s)][rd] for s in SEEDS], 0)

    out = {"look": "chainscore28", "script_sha256": sha(__file__), "runs": {f"{k[0]}-s{k[1]}": v for k, v in runs.items()},
           "E1": {}, "S": {}, "diag": {}, "decision": {}}
    for f in ZSTEM:
        z, cst, pst = ZSTEM[f], CSTEM[f], PSTEM[f]
        e = {"z": {rd: boot_ci(mean_rows(z, rd), rng) for rd in ("metaqa", "webqsp")},
             "c": {rd: boot_ci(mean_rows(cst, rd), rng) for rd in ("metaqa", "webqsp")},
             "p": {rd: boot_ci(mean_rows(pst, rd), rng) for rd in ("metaqa", "webqsp")},
             "lead_wq": boot_ci(mean_rows(cst, "webqsp") - mean_rows(z, "webqsp"), rng),
             "pool_minus_c_wq": boot_ci(mean_rows(pst, "webqsp") - mean_rows(cst, "webqsp"), rng),
             "pool_minus_z_mq": boot_ci(mean_rows(pst, "metaqa") - mean_rows(z, "metaqa"), rng),
             "c_minus_z_mq": boot_ci(mean_rows(cst, "metaqa") - mean_rows(z, "metaqa"), rng),
             "pooled_reading_from": pst}
        out["E1"][f] = e
        out["S"][f] = {rd: {lab: [round(float((rows[(x, s)][rd] - rows[(y, s)][rd]).mean()), 5) for s in SEEDS]
                            for lab, x, y in (("c-z", cst, z), ("p-c", pst, cst), ("p-z", pst, z))}
                       for rd in ("metaqa", "webqsp")}
        lead = e["lead_wq"]
        if lead[0] < CAP_LEAD or lead[1] <= 0:
            lab = "CAPACITY"
        elif e["pool_minus_c_wq"][0] >= -POOL_WQ and e["pool_minus_z_mq"][0] >= -POOL_MQ:
            lab = "SHARED"
        else:
            lab = "CONFLICT"
        out["decision"][f] = {"label": lab}
    for p_ in a.diag:
        dj = json.loads(Path(p_).read_text(encoding="utf-8"))
        if dj.get("look") != "chainscore28-diag":
            raise SystemExit(f"{p_} is not a part 12 diag")
        f = dj["f"]
        out["diag"][f] = {"path": p_, "sha256": sha(p_), **{k: dj.get(k) for k in ("E2", "E3", "E4", "A", "E6")}}
        sh = dj["E2"]["share"]
        top = max((k for k in sh if sh[k] is not None), key=lambda k: sh[k], default=None)
        sig = {kind: [g for g, ci in dj["E3"][kind]["c_minus_z"].items() if ci[1] > 0 or ci[2] < 0]
               for kind in ("node", "pair")}
        out["decision"].setdefault(f, {}).update({"largest_component": top, "E4": dj["E4"]["z_label"],
                                                  "E3_groups": sig})
    miss_d = [f for f in ZSTEM if f not in out["diag"]]
    out["diag_missing"] = miss_d
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ── main ────────────────────────────────────────────────────────────────────────────────────────────────────────


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="cpu")
    t.add_argument("--map-from")
    t.add_argument("--map", default="pq")
    t.add_argument("--smoke", action="store_true")
    t.add_argument("--cache")
    t.add_argument("--wqfit")
    t.add_argument("--fit")
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select")
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--edges", action="append", default=[])
    t.add_argument("--epochs", type=int, default=12)
    t.add_argument("--per-epoch", type=int, default=5960)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    g = sub.add_parser("diag")
    g.add_argument("--f", required=True)
    g.add_argument("--z", action="append", required=True)
    g.add_argument("--c", action="append", required=True)
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--ref", action="append", default=[])
    g.add_argument("--edges", action="append", default=[])
    g.add_argument("--cache")
    g.add_argument("--device", default="cpu")
    g.add_argument("--threads", type=int, default=1)
    g.add_argument("--out", required=True)
    r = sub.add_parser("grade")
    r.add_argument("--run", action="append", required=True)
    r.add_argument("--z", action="append", required=True)
    r.add_argument("--diag", action="append", default=[])
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"train": train_cmd, "diag": diag_cmd, "grade": grade_cmd}[a.cmd](a)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    os.replace(tmp, p)
    log(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
