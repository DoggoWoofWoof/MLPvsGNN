"""Design look (untracked; not a result and not filed): S6 part 9 of the transfer plan, the GNN f under the carried
map. Part 4 (chainscore20) carried the posterior-typed GNN as the node scorer f: ena-gr, K1 ABOVE, webqsp selectf + fit
+3.5 [+1.8, +5.0] R@5 over the MLP f (em-gr), and K3 HELPS there, +4.0 [+2.8, +5.1] over the untyped GNN (en-gr).
Parts 5, 6 and 8 were declared on the MLP f before part 4's grade. Part 5 (chainscore21) carried the trained length
offset (lb); part 6 (chainscore22) carried the population map pq (N1 ABOVE, webqsp selectf + fit +3.9 [+2.8, +5.1]
over b-lb; metaqa not BELOW). This part puts part 4's GNN f into that configuration. Swastik (4 Oct): the scorer and
the chain model 'should be a part of the same model', and 'the scorer with the mlp can be a EM step refinement which
might be able to solve the problem of webqsp'. ena-gr is that model: its GNN reruns on each EM round's chain posterior.

Arms. f is chainscore20's GNN (two message layers over the row's graph, one channel per direction; no relation id or
name enters it) with chainscore19's chain scorer g, coupled by chainscore20's EM rounds, and lb on every arm:
    e-b-lb      the posterior-typed GNN (ena-gr), map none: part 5's b-lb with part 4's f
    e-b-lb-pq   the same with every population mapped by pq (the candidate)
    n-b-lb-pq   the untyped GNN (en-gr) under pq: part 4's K3 again, under the map
Controls, not retrained: part 6's MLP-f arms on the same device (the host GPU through cs_dev), three seeds each,
read from their rows files by sha256: m-b-lb (cs22-b-lb-s0..2, map none) and m-b-lb-pq (cs22-b-lb-pq-s0..2).
Training as part 6: metaqa's fit carve and its four transforms (chainscore19's -gr builds) with part 4's companions
(cs20e-*, the rows' messages and the chains' type slots; chainscore20's load_comp checks each against its build's
sha256), Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, the epoch chosen by R@5 on
metaqa's select carve, seeds SEED + 0, 1, 2 (the initial weights and the draw), on the host GPU with cs_dev's settings
(deterministic algorithms, TF32 off): no number here is set beside a CPU fit's (part 4's arms ran on the CPU). Every
input comes from cs_cache (prep, lb from the raw counts, the map, the first build's input stats; map none's entries
are made for this part, pq's are part 8's). The select carve's batches are made once, on the CPU, and moved to the
device at each epoch's read; the training statements are cs_dev's train_dev.
Reads. The KB rule on metaqa x1f, webqsp selectf and webqsp selectf + fit (webqsp is never trained on); the passage
rule on 2wiki x1 and hotpotqa x1 in the run. musique x1f is read by its own job from the run's saved weights (pread):
the smoke (cs25t-ena: cs_dev run21, one epoch of 640 examples; its numbers are not read) took 30 min for musique's
select carve alone on the shared GPU (1.89 GB of messages; x1f's are 5.68 GB) and 0.73 s a training batch (part 6's
MLP f: 0.08 to 0.09). Until every musique read is in, the grade marks musique NOT_READ; a second grade fills it. The
decision does not use musique. A pread of 2wiki x1 from a smoke's saved weights must equal that smoke's own 2wiki read
row for row (the check that a read from saved weights is the run's read).
Verdicts, fixed at 00:40 on 5 Oct (this file's first write), before any part 9 run. Per arm, each row's R@5 averaged
over its three seeds; paired row bootstrap (BOOT 1000):
    G1 (primary) webqsp selectf + fit: e-b-lb-pq minus m-b-lb-pq (part 6's carried model): ABOVE / AT / BELOW
    G1b webqsp selectf + fit: e-b-lb minus m-b-lb-pq and n-b-lb-pq minus m-b-lb-pq
    G2 webqsp selectf + fit: e-b-lb-pq minus e-b-lb (does the map carry with the GNN f)
    G3 metaqa x1f: G1's, G1b's and G2's differences
    G4 each KB read: e-b-lb minus m-b-lb (the GNN f over the MLP f without the map: part 4's K1 with lb, on the GPU)
       and e-b-lb-pq minus n-b-lb-pq (HELPS / HURTS / SAME: does the EM typing add under the map)
    G5 webqsp selectf + fit by seed-gold distance D (1, 2): G1's, G1b's and G2's differences
    G6 each KB read: every arm and both controls minus none/rd (the untyped walk)
    G7 each passage read (2wiki x1, hotpotqa x1; musique x1f once read): G1's and G2's differences, and every arm and
       both controls minus s0+rrf with its share of the twin's lead over s0+rrf (HIGH at or above 0.5, LOW below
       0.25, else MID)
    S  every difference above per seed (seed k's rows minus seed k's), with its sign
    R0 e-b-lb seed 0 through this file's loop and cache equals the same arm through cs_dev's train_dev on the
       pipeline's inputs (--via-dev, no cache) row for row (|diff| < 1e-6) on every read in the run, and weight for
       weight (state_sha256): REPRODUCES / DIFFERS. The pq entries are part 8's; its R0 (k5 seed 0 against part 6's
       b-lb-pq seed 0) checks them, and if it DIFFERS no pq arm here is read until its cause is found.
Decision. The candidates e-b-lb-pq, e-b-lb and n-b-lb-pq, each against m-b-lb-pq: one ABOVE on webqsp selectf + fit
and not BELOW on metaqa carries (the largest webqsp selectf + fit mean difference if several); if none, one AT on
webqsp selectf + fit and ABOVE on metaqa carries (the largest metaqa difference). A GNN arm that carries is the next
parts' model (f, map); otherwise part 6's MLP f under pq stays, at about an eighth of the GNN's cost a batch. G2
BELOW (the map costs the GNN f zero-shot) or no GNN arm carrying goes to Swastik. R0 DIFFERS: nothing is read until
its cause is found. Train-split rows throughout: a look, not a result.
    python outputs/mp_unified/cs_cache.py build --kind train --map none --src outputs/mp_unified/lean/cs19-mq-fit-id.npz
        (each fit build; --kind plain: the select carve and every read)
    python outputs/mp_unified/chainscore25.py train --arm e-b-lb-pq --seed 0 --device cuda \
        --map-from outputs/mp_unified/lean/cs22.json --cache outputs/mp_unified/cache \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz \
        (sp4, mg3, rf) --select outputs/mp_unified/lean/cs19-mq-select.npz --read metaqa=... --read webqsp=... \
        --read webqsp_sf=... --pread 2wiki=... --pread hotpotqa=... --edges fit=outputs/mp_unified/lean/cs20e-mq-fit-id.npz \
        (al4, sp4, mg3, rf, select, metaqa, webqsp, webqsp_sf: cs20e-*; 2wiki, hotpotqa: cs21e-*-x1) \
        --out X.json --rows-out X.rows.npz --state-out X.pt
    python outputs/mp_unified/chainscore25.py pread --run X.json --pread musique=outputs/mp_unified/lean/cs21-mu-x1f.npz \
        --edges musique=outputs/mp_unified/lean/cs21e-mu-x1f.npz --cache outputs/mp_unified/cache --device cuda \
        --out X.mu.json --rows-out X.mu.rows.npz
    python outputs/mp_unified/chainscore25.py grade --run X.json (every arm x seed) --r0 R.json \
        --control outputs/mp_unified/lean/cs22-b-lb-s0.json (s1, s2; and cs22-b-lb-pq-s0..2) [--mu-run X.mu.json ...] \
        --read ... --pread 2wiki=... --pread hotpotqa=... --pread musique=... --out outputs/mp_unified/lean/cs25.json
    python outputs/mp_unified/chainscore25.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore22 as C22  # noqa: E402
import chainscore24 as C24  # noqa: E402
import cs_cache as CC  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
BATCH = C21.BATCH
SEEDS = (0, 1, 2)
TF4 = C19.TRANSFORMS[1:]
KB_READS = C21.KB_READS
P_RUN = ("2wiki", "hotpotqa")           # read in the run
P_READS = P_RUN + ("musique",)          # musique by pread
ARMS = {"e-b-lb": ("ena-gr", "none"), "e-b-lb-pq": ("ena-gr", "pq"), "n-b-lb-pq": ("en-gr", "pq")}
CTLS = {"m-b-lb": ("b-lb", "none"), "m-b-lb-pq": ("b-lb-pq", "pq")}      # part 6's arm, its map
CANDS = ("e-b-lb-pq", "e-b-lb", "n-b-lb-pq")
ABV, HS = ("ABOVE", "BELOW", "AT"), ("HELPS", "HURTS", "SAME")
UNTYPED_POP = ("slots", "p_slot", "i_e", "i_s")     # chainscore21's comp: the untyped GNN reads the messages only
log, sha = C21.log, CC.sha


def get(p, kind, norm, cache):
    return C24.get(p, kind, norm, cache)


def carried_map(path):
    """Part 6's grade must carry pq: this part's arms are declared under it."""
    norm, info = C24.map_rule(path)
    if norm != "pq":
        raise SystemExit(f"part 6's grade ({path}) carries {norm}, not pq: this part's arms are declared under pq")
    return info


def load_comp(edges, d, base_sha, att):
    c = C20.load_comp(edges, d, base_sha)
    if not att:
        for k in UNTYPED_POP:
            c.pop(k)
    return c


# ── training: cs_dev's train_dev with the select carve's batches made once ─────────────────────────────────────


def select_batches(Sd, Sc, Slb, st, spec, batch=BATCH):
    """read_dev's batches of the select carve (the KB rule, with its companion), made once on the CPU."""
    N = Sd["n"].size
    out = []
    for b0 in range(0, N, batch):
        rows = np.arange(b0, min(N, b0 + batch))
        items = np.c_[np.zeros(rows.size, np.int64), rows, np.zeros(rows.size, np.int64)]
        out.append((rows, C21.make_batch21(items, [Sd], [Sc], [Slb], st, pairs=spec["g"], train=False,
                                           gnn=spec["gnn"], att=spec["att"])))
    return out


def read_select(model, Sd, sb, device):
    """read_dev's KB-rule ranking of the select carve from the batches made once (R@5, FC@5, hit@1 per row)."""
    import torch
    N = Sd["n"].size
    out = np.zeros((N, 3))
    with torch.no_grad():
        for rows, bt in sb:
            if CD.placed(device):
                with torch.device(device):
                    s, _lp = C20.forward20(model, CD.to_dev(bt, device))
            else:
                s, _lp = C20.forward20(model, bt)
            s = s.double().cpu().numpy()
            for j, i in enumerate(rows):
                out[i] = C19.rank_row(Sd, i, s[j, :int(Sd["n"][i])])
    return out


def train25(arm, f, builds, comps, LB, groups, Sd, Sc, Slb, st, epochs, per_epoch, batch, seed, device=None,
            threads=1):
    """cs_dev's train_dev (a GNN f with g and lb, the KB select rule) statement for statement, but for one change: the
    select carve's batches are made once. st is the first build's input stats (cs_cache's, or computed here as
    train_dev does)."""
    import torch
    block = CD.settings(device, threads)
    spec = C21.arm_spec("b-lb", f)
    if not (spec["gnn"] and spec["g"] and spec["lb"]):
        raise SystemExit("this part trains a GNN f with g and lb")
    st = st if st is not None else C19.input_stats(builds[0])
    model = C20.make_model(f, seed)
    if CD.placed(device):
        CD._dp().model_to(model, device)       # once, before the optimiser is built
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    sb = select_batches(Sd, Sc, Slb, st, spec)
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
            x = read_select(model, Sd, sb, device)
            sr5 = float(x[:, 0].mean())
            hist.append({"epoch": ep, "loss": round(tot / max(nb, 1), 5), "node_loss": round(tn / max(nb, 1), 5),
                         "chain_loss": round(tc / max(nb, 1), 5), "select_r5": round(sr5, 5),
                         "by_build": np.bincount(items[:, 0], minlength=len(builds)).tolist(),
                         **C20.coupling20(model), "seconds": round(time.time() - t1, 1)})
            log(f"    {arm} ({f}) ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
                f"{tc / max(nb, 1):.4f}) select R@5 {sr5:.4f} {C20.coupling20(model)} {time.time() - t1:.0f}s")
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


# ── train (one arm, one seed) ───────────────────────────────────────────────────────────────────────────────────


def train_cmd(a):
    t0 = time.time()
    if a.arm not in ARMS:
        raise SystemExit(f"unknown arm {a.arm}; one of {list(ARMS)}")
    f, norm = ARMS[a.arm]
    minfo = carried_map(a.map_from)
    if a.via_dev and (a.cache or a.arm != "e-b-lb"):
        raise SystemExit("--via-dev (R0) runs e-b-lb through cs_dev's train_dev on the pipeline's inputs: no cache")
    if not a.via_dev and not a.cache:
        raise SystemExit("the arms load their inputs from cs_cache (--cache)")
    dev = None if a.device == "cpu" else a.device
    spec = C21.arm_spec("b-lb", f)
    res = {"look": "chainscore25", "arm": a.arm, "seed": a.seed, "f": f, "norm": norm, "device": a.device,
           "args": dict(vars(a)), "map_rule": minfo, "via_dev": bool(a.via_dev), "cache": a.cache,
           "state_out": a.state_out, "script_sha256": sha(__file__), "cs_cache_sha256": sha(CC.__file__),
           "chainscore24_sha256": sha(C24.__file__), "chainscore22_sha256": sha(C22.__file__),
           "chainscore21_sha256": sha(C21.__file__), "chainscore20_sha256": sha(C20.__file__),
           "chainscore19_sha256": sha(C19.__file__), "chainscore18_sha256": sha(CS.__file__),
           "cs_dev_sha256": sha(CD.__file__), "inputs": {}, "inputs_sha256": {}, "passage_types": {}, "maps": {},
           "edges": {}, "reads": {}, "rows_out": a.rows_out, "timing": {}}
    aug = dict(x.split("=", 1) for x in a.aug)
    if sorted(aug) != sorted(TF4):
        raise SystemExit(f"every arm takes --aug for each of {TF4}")
    if not a.fit or not a.select:
        raise SystemExit("every arm takes --fit and --select")
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_RUN):
        raise SystemExit(f"every arm reads {KB_READS} (--read) and {P_RUN} (--pread; musique by pread)")
    train_in = [("fit", a.fit)] + [(t, aug[t]) for t in TF4]
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    path = dict(train_in)
    path["select"] = a.select
    path.update({n: p for n, p, _ in reads})
    edges = dict(x.split("=", 1) for x in a.edges)
    if sorted(edges) != sorted(path):
        raise SystemExit(f"a GNN f takes --edges for each of {sorted(path)}; given {sorted(edges)}")
    t1 = time.time()
    for n, p in path.items():
        res["inputs_sha256"][p] = sha(p)
        res["inputs_sha256"][edges[n]] = sha(edges[n])
    for n, p, is_p in reads:
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], res["inputs_sha256"][edges[n]])
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    res["timing"]["hash_s"] = round(time.time() - t1, 1)

    def load(p, kind):
        if a.via_dev:
            d, lb, info = CC.pipeline(p, kind, norm)
            return d, lb, info, None
        return get(p, kind, norm, a.cache)

    def comp(n, d):
        return load_comp(edges[n], d, res["inputs_sha256"][path[n]], spec["att"])

    t1 = time.time()
    builds, comps, LB, st0 = [], [], [], None
    for n, p in train_in:
        d, lb, info, st = load(p, "train")
        tf = d["meta"]["transform"]["name"]
        if tf != ("id" if n == "fit" else n):
            raise SystemExit(f"{p} holds transform {tf}, not {'id' if n == 'fit' else n}")
        if not builds:
            st0 = st
        res["inputs"][n] = d["meta"]
        res["maps"][n] = info
        builds.append(d)
        comps.append(comp(n, d))
        LB.append(lb)
        log(f"  {a.arm} s{a.seed}: {n} ready ({'pipeline' if a.via_dev else 'cache'}; {info})")
    groups = [list(range(len(builds)))]
    Sd, Slb, sinfo, _ = load(a.select, "plain")
    Sc = comp("select", Sd)
    res["maps"]["select"] = sinfo
    res["inputs"]["select"] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p), b in zip(train_in, builds)}
    res["groups"] = groups
    res["edges"] = {n: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                    for (n, _p), c in zip(train_in, comps)}
    res["timing"]["load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    if a.via_dev:
        model, info = CD.train_dev("b-lb", f, builds, comps, LB, groups, Sd, Sc, Slb, "kb", a.epochs, a.per_epoch,
                                   a.batch, SEED + a.seed, device=dev)
    else:
        model, info = train25(a.arm, f, builds, comps, LB, groups, Sd, Sc, Slb, st0, a.epochs, a.per_epoch, a.batch,
                              SEED + a.seed, device=dev)
    res["timing"]["train_s"] = round(time.time() - t1, 1)
    del builds, comps, LB, Sd, Sc, Slb
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings")})
    t1 = time.time()
    rows_out = {}
    for n, p, is_p in reads:
        t2 = time.time()
        d, lb, rinfo, _ = load(p, "plain")
        c = comp(n, d)
        res["maps"][n] = rinfo
        rule = "passage" if is_p else "kb"
        xr, ml = CD.read_dev(model, d, c, lb, info["stats"], rule, dev, batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule=rule, seconds=round(
            time.time() - t2, 1))
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} s{a.seed} on {n} ({rule} rule): {res['reads'][n]['mean']} ({time.time() - t2:.0f}s)")
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


# ── pread: one passage read from a run's saved weights ──────────────────────────────────────────────────────────


def pread_cmd(a):
    import torch
    t0 = time.time()
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    if run.get("look") != "chainscore25" or run.get("via_dev") or not run.get("state_out"):
        raise SystemExit(f"{a.run}: not a chainscore25 arm run with saved weights")
    if a.device != run["device"]:
        raise SystemExit(f"a read runs on its run's device ({run['device']}), not {a.device}")
    f, norm = ARMS[run["arm"]]
    n, p = a.pread.split("=", 1)
    ne, pe = a.edges.split("=", 1)
    if n != ne or n not in P_READS:
        raise SystemExit(f"pread reads one of {P_READS} with its own --edges")     # 2wiki or hotpotqa: a check
    dev = None if a.device == "cpu" else a.device
    block = CD.settings(dev, 1)
    S = torch.load(run["state_out"], map_location="cpu", weights_only=False)
    got = CD.state_sha(S["state"])
    if got != run["state_sha256"]:
        raise SystemExit(f"{run['state_out']}: weights {got[:16]}, the run's {run['state_sha256'][:16]}")
    if (S["arm"], int(S["seed"]), S["norm"]) != (run["arm"], int(run["seed"]), norm):
        raise SystemExit(f"{run['state_out']} holds {S['arm']} seed {S['seed']} ({S['norm']})")
    spec = C21.arm_spec("b-lb", f)
    model = C20.make_model(f, SEED + int(run["seed"]))
    model.load_state_dict(S["state"])
    if CD.placed(dev):
        CD._dp().model_to(model, dev)
    model.eval()
    sp, se = sha(p), sha(pe)
    side = C21.check_passage(p, sp, se)
    d, lb, rinfo, _ = get(p, "plain", norm, a.cache)
    c = load_comp(pe, d, sp, spec["att"])
    t1 = time.time()
    xr, ml = CD.read_dev(model, d, c, lb, S["stats"], "passage", dev, batch=a.read_batch)
    rows = {n: xr.astype(np.float32), f"{n}__D": d["D"]}
    q = Path(a.rows_out)
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_name(q.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows)
    os.replace(tmp, q)
    res = {"look": "chainscore25-pread", "run": a.run, "arm": run["arm"], "seed": int(run["seed"]), "f": f,
           "norm": norm, "device": a.device, "placement": block, "state_sha256": got, "read": n,
           "inputs_sha256": {p: sp, pe: se}, "passage_types": {n: {"script_sha256": side["script_sha256"],
                                                                   **side["types"]}},
           "maps": {n: rinfo}, "inputs": {n: d["meta"]}, "script_sha256": sha(__file__),
           "reads": {n: dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule="passage",
                             seconds=round(time.time() - t1, 1))}, "rows_out": a.rows_out}
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  {run['arm']} s{run['seed']} on {n} (passage rule): {res['reads'][n]['mean']} ({res['seconds']}s)")
    return res


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def grade_cmd(a):
    t0 = time.time()
    runs, rows = {}, {}
    for p_ in a.run:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore25" or js.get("via_dev"):
            raise SystemExit(f"{p_} is not a chainscore25 arm run")
        key = (js["arm"], int(js["seed"]))
        if key in runs:
            raise SystemExit(f"two runs of {key}")
        with np.load(js["rows_out"]) as z:
            rows[key] = {k: z[k] for k in z.files}
        runs[key] = js
    want = {(arm, s) for arm in ARMS for s in SEEDS}
    if set(runs) != want:
        raise SystemExit(f"the grade takes every arm x seed; missing {sorted(want - set(runs))}, "
                         f"extra {sorted(set(runs) - want)}")
    devs = {js["device"] for js in runs.values()}
    shas = {js["script_sha256"] for js in runs.values()}
    if len(devs) != 1 or len(shas) != 1:
        raise SystemExit(f"runs on several devices {devs} or scripts {[s[:12] for s in shas]}")
    dev = next(iter(devs))
    mu = {}
    for p_ in a.mu_run:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore25-pread" or js.get("read") in P_RUN:
            raise SystemExit(f"{p_} is not a chainscore25 pread of {sorted(set(P_READS) - set(P_RUN))}")
        key = (js["arm"], int(js["seed"]))
        if key not in runs or js["state_sha256"] != runs[key]["state_sha256"] or js["device"] != dev:
            raise SystemExit(f"{p_}: not a read of the graded run {key} on {dev}")
        if (key, js["read"]) in mu:
            raise SystemExit(f"two preads of {key} on {js['read']}")
        mu[(key, js["read"])] = js
        with np.load(js["rows_out"]) as z:
            rows[key].update({k: z[k] for k in z.files})
    ctl = {}
    for p_ in a.control:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        hit = [c for c, (arm6, _n) in CTLS.items() if js.get("look") == "chainscore22" and js.get("arm") == arm6]
        if not hit:
            raise SystemExit(f"{p_} is not one of part 6's {[v[0] for v in CTLS.values()]} runs")
        key = (hit[0], int(js["seed"]))
        if key in ctl:
            raise SystemExit(f"two controls {key}")
        if js.get("device") != dev:
            raise SystemExit(f"{p_} ran on {js.get('device')}, the runs on {dev}")
        if js.get("norm") != CTLS[hit[0]][1]:
            raise SystemExit(f"{p_} has map {js.get('norm')}, not {CTLS[hit[0]][1]}")
        ctl[key] = js
        with np.load(js["rows_out"]) as z:
            rows[key] = {k: z[k] for k in z.files}
    if set(ctl) != {(c, s) for c in CTLS for s in SEEDS}:
        raise SystemExit(f"the grade takes part 6's {list(CTLS)} x {SEEDS}")
    refs, Ds, rule, read_by = {}, {}, {}, {}
    for flag, xs in (("kb", a.read), ("passage", a.pread)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            h = sha(p_)
            keys = [k for k in runs if nm_ in rows[k]]
            if nm_ in P_RUN or nm_ in KB_READS:
                keys = list(runs)
            for k in keys:
                src = runs[k] if nm_ in KB_READS + P_RUN else mu[(k, nm_)]
                if src["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"run {k} read another {nm_} than {p_}")
            for k, js in ctl.items():
                if js["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"control {k} read another {nm_} than {p_}")
            read_by[nm_] = len(keys)
            d = CS.load(p_)
            refs[nm_] = CS.reference_arms(d) if flag == "kb" else C21.passage_refs(d)
            Ds[nm_] = d["D"]
            rule[nm_] = flag
            del d
    if sorted(k for k in rule if rule[k] == "kb") != sorted(KB_READS) or \
            sorted(k for k in rule if rule[k] == "passage") != sorted(P_READS):
        raise SystemExit(f"the grade reads {KB_READS} (--read) and {P_READS} (--pread)")
    graded = [nm_ for nm_ in refs if read_by[nm_] == len(runs)]
    not_read = [nm_ for nm_ in refs if nm_ not in graded]
    if any(read_by[nm_] not in (0, len(runs)) for nm_ in not_read):
        raise SystemExit(f"a read in by some runs only: {[(nm_, read_by[nm_]) for nm_ in not_read]}")
    ALL = list(ARMS) + list(CTLS)

    def r5(arm, nm_, s=None):
        if s is not None:
            return rows[(arm, s)][nm_][:, 0].astype(np.float64)
        return np.mean([rows[(arm, k)][nm_][:, 0].astype(np.float64) for k in SEEDS], 0)

    V = {k: {} for k in ("G1", "G1b", "G2", "G3", "G4", "G5", "G6", "G7", "S", "R0")}
    V["not_read"] = not_read

    def ci(dd, labels_, idx):
        c = CP.boot_mean(dd, idx)
        return {"diff": c, "verdict": C19.ci_label(c, labels_)}

    def seeds_of(x_, y_, nm_):
        out = []
        for s in SEEDS:
            dd = r5(x_, nm_, s) - (r5(y_, nm_, s) if isinstance(y_, str) else y_)
            out.append(round(float(dd.mean()), 5))
        return {"per_seed": out, "signs": [int(np.sign(v)) for v in out]}

    def pair(x_, y_, nm_, idx, labels_=ABV):
        Y = r5(y_, nm_) if isinstance(y_, str) else y_
        e = ci(r5(x_, nm_) - Y, labels_, idx)
        V["S"].setdefault(nm_, {})[f"{x_} - {y_ if isinstance(y_, str) else 'ref'}"] = seeds_of(x_, y_, nm_)
        return e

    G1P = ("e-b-lb-pq", "m-b-lb-pq")
    G1BP = (("e-b-lb", "m-b-lb-pq"), ("n-b-lb-pq", "m-b-lb-pq"))
    G2P = ("e-b-lb-pq", "e-b-lb")
    for nm_ in graded:
        N = refs[nm_]["twin0"].shape[0]
        for k, r in rows.items():
            if r[nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {r[nm_].shape[0]} rows, the build {N}")
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        if nm_ == "webqsp_sf":
            V["G1"] = pair(*G1P, nm_, idx)
            V["G1b"] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in G1BP}
            V["G2"] = pair(*G2P, nm_, idx)
            for dd in (1, 2):
                m = Ds[nm_] == dd
                if m.any():
                    ii = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                    V["G5"][f"D{dd}"] = {f"{x_} - {y_}": dict(ci((r5(x_, nm_) - r5(y_, nm_))[m], ABV, ii),
                                                              rows=int(m.sum()))
                                         for x_, y_ in (G1P,) + G1BP + (G2P,)}
        if nm_ == "metaqa":
            V["G3"] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in (G1P,) + G1BP + (G2P,)}
        if rule[nm_] == "kb":
            V["G4"][nm_] = {"e-b-lb - m-b-lb": pair("e-b-lb", "m-b-lb", nm_, idx),
                            "e-b-lb-pq - n-b-lb-pq": pair("e-b-lb-pq", "n-b-lb-pq", nm_, idx, HS)}
            walk = refs[nm_]["none/rd"][:, 0]
            V["G6"][nm_] = {f"{x_} - none/rd": pair(x_, walk, nm_, idx) for x_ in ALL}
        else:
            e = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in (G1P, G2P)}
            s0r, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["twin0"][:, 0]
            for x_ in ALL:
                X_ = r5(x_, nm_)
                sh = CP.share(X_, s0r, tw, idx)
                e[f"{x_} - s0+rrf"] = dict(ci(X_ - s0r, ABV, idx), share=sh,
                                           share_verdict="HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID")
            V["G7"][nm_] = e
    for p_ in a.r0 or []:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore25" or not js.get("via_dev") or (js["arm"], int(js["seed"])) != ("e-b-lb", 0):
            raise SystemExit(f"{p_} is not R0's run (e-b-lb seed 0 with --via-dev)")
        if js["device"] != dev:
            raise SystemExit(f"R0 ran on {js['device']}, the runs on {dev}")
        with np.load(js["rows_out"]) as z:
            r0 = {k: z[k] for k in z.files}
        for nm_ in KB_READS + P_RUN:
            dif = float(np.abs(rows[("e-b-lb", 0)][nm_].astype(np.float64) - r0[nm_].astype(np.float64)).max())
            V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
        same_w = js.get("state_sha256") == runs[("e-b-lb", 0)].get("state_sha256")
        V["R0"]["weights"] = {"state_sha256": [runs[("e-b-lb", 0)].get("state_sha256"), js.get("state_sha256")],
                              "verdict": "REPRODUCES" if same_w else "DIFFERS"}
        V["R0"]["run"] = p_
    res = {"look": "chainscore25", "args": dict(vars(a)), "script_sha256": sha(__file__), "run_script": shas.pop(),
           "device": dev, "graded": graded, "not_read": not_read,
           "runs": {f"{k[0]}#s{k[1]}": {"best_epoch": js["best_epoch"], "best_select_r5": js["best_select_r5"],
                                        "state_sha256": js.get("state_sha256"), "timing": js.get("timing"),
                                        "seconds_per_batch": js.get("seconds_per_batch"),
                                        "reads": {n: v["mean"] for n, v in js["reads"].items()}}
                    for k, js in sorted(runs.items())},
           "preads": {f"{k[0]}#s{k[1]}:{n}": js["reads"][n]["mean"] for (k, n), js in sorted(mu.items())},
           "means": {nm_: {arm: [round(float(np.mean([rows[(arm, s)][nm_][:, c].astype(np.float64).mean()
                                                       for s in SEEDS])), 5) for c in range(3)] for arm in ALL}
                     for nm_ in graded},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()
                          if k in ("rrf", "twin0", "none/rd", "s0+rrf", "s0+walk")} for nm_, r in refs.items()},
           "verdicts": V, "decision": decide(V)}
    res["seconds"] = round(time.time() - t0, 1)
    log(f"decision {res['decision']}")
    return res


def decide(V):
    r0 = [(V["R0"].get(k) or {}).get("verdict") for k in KB_READS + P_RUN + ("weights",)]
    if None in r0:
        return {"carry": None, "why": "R0 incomplete (no --r0 run)", "report": True}
    if "DIFFERS" in r0:
        return {"carry": None, "why": "R0 DIFFERS: nothing is read until its cause is found", "report": True}

    def w(x_):
        return V["G1"] if x_ == "e-b-lb-pq" else V["G1b"][f"{x_} - m-b-lb-pq"]

    def m(x_):
        return V["G3"][f"{x_} - m-b-lb-pq"]

    g2 = V["G2"]["verdict"]
    up = [(w(x_)["diff"][0], x_) for x_ in CANDS if w(x_)["verdict"] == "ABOVE" and m(x_)["verdict"] != "BELOW"]
    if up:
        c = max(up)[1]
        why = f"ABOVE m-b-lb-pq on webqsp selectf + fit and not BELOW on metaqa: {[x_ for _d, x_ in up]}"
    else:
        up = [(m(x_)["diff"][0], x_) for x_ in CANDS if w(x_)["verdict"] == "AT" and m(x_)["verdict"] == "ABOVE"]
        c = max(up)[1] if up else None
        why = (f"AT m-b-lb-pq on webqsp selectf + fit and ABOVE on metaqa: {[x_ for _d, x_ in up]}" if up else
               "no GNN arm ABOVE m-b-lb-pq on webqsp selectf + fit (or AT with metaqa ABOVE): part 6's MLP f under "
               "pq stays")
    return {"carry": c or "m-b-lb-pq", "gnn_carries": c is not None, "G2": g2, "why": why,
            "report": c is None or g2 == "BELOW"}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def selftest():
    import inspect
    import tempfile

    import torch
    # 1. the arm table: GNN f's with g and lb; the typed arm attends, the untyped one does not
    for arm, (f, norm) in ARMS.items():
        sp = C21.arm_spec("b-lb", f)
        assert sp["gnn"] and sp["g"] and sp["lb"] and norm in C22.NORMS, arm
        assert sp["att"] == (f == "ena-gr"), arm
    assert set(CANDS) == set(ARMS) and set(P_RUN) < set(P_READS) == set(C21.P_READS)
    assert {v[0] for v in CTLS.values()} <= set(C22.ARMS)
    # 2. saved weights hash as the run's: state_sha survives torch.save and torch.load
    for f in ("ena-gr", "en-gr"):
        model = C20.make_model(f, 3)
        st = {k: v.detach().clone() for k, v in model.state_dict().items()}
        h = CD.state_sha(st)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.pt"
            torch.save({"state": {k: v.detach().cpu() for k, v in st.items()}, "stats": {"m": np.zeros(3)}}, p)
            S = torch.load(p, map_location="cpu", weights_only=False)
        assert CD.state_sha(S["state"]) == h, f
        m2 = C20.make_model(f, 4)
        m2.load_state_dict(S["state"])
        assert CD.state_sha(m2.state_dict()) == h, f
    # 3. train25 keeps train_dev's statements: the same calls in the same order
    src_dev, src_25 = inspect.getsource(CD.train_dev), inspect.getsource(train25)
    order = ("C20.make_model(", "torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)",
             "np.random.default_rng(seed)", "C21.draw21(rng, builds, groups, per_epoch)", "C20.forward20(model, bt)",
             "C19.objective(s, lp, bt)", "opt.zero_grad()", "loss.backward()", "opt.step()",
             "if best is None or sr5 > best[1]:", "model.load_state_dict(best_state)")
    for s_ in order:
        assert s_ in src_dev and s_ in src_25, s_
    pos = [src_25.index(s_) for s_ in order]
    assert pos == sorted(pos)
    # 4. the carried map must be pq
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cs22.json"
        p.write_text(json.dumps({"look": "chainscore22", "decision": {"carry": "pq"}}), encoding="utf-8")
        assert carried_map(p)["part6_decision"]["carry"] == "pq"
        for carry in ("pz", "none", None):
            p.write_text(json.dumps({"look": "chainscore22", "decision": {"carry": carry}}), encoding="utf-8")
            try:
                carried_map(p)
                raise AssertionError(f"carry {carry} accepted")
            except SystemExit:
                pass
    # 5. decide

    def v(verdict, d=0.0):
        return {"verdict": verdict, "diff": [d, d - 0.01, d + 0.01]}

    def table(w, m):
        return {"R0": {k: {"verdict": "REPRODUCES"} for k in KB_READS + P_RUN + ("weights",)},
                "G1": v(*w["e-b-lb-pq"]), "G1b": {f"{x} - m-b-lb-pq": v(*w[x]) for x in ("e-b-lb", "n-b-lb-pq")},
                "G2": v("AT"), "G3": {f"{x} - m-b-lb-pq": v(*m[x]) for x in CANDS}}

    at = {x: ("AT",) for x in CANDS}
    V = table(at, at)
    assert decide(V)["carry"] == "m-b-lb-pq" and decide(V)["report"]
    V = table({**at, "e-b-lb-pq": ("ABOVE", 0.02), "e-b-lb": ("ABOVE", 0.03)}, at)
    assert decide(V)["carry"] == "e-b-lb" and not decide(V)["report"]
    V = table({**at, "e-b-lb-pq": ("ABOVE", 0.02), "e-b-lb": ("ABOVE", 0.03)}, {**at, "e-b-lb": ("BELOW", -0.02)})
    assert decide(V)["carry"] == "e-b-lb-pq"
    V = table(at, {**at, "n-b-lb-pq": ("ABOVE", 0.01)})
    assert decide(V)["carry"] == "n-b-lb-pq" and decide(V)["gnn_carries"]
    V["G2"] = v("BELOW", -0.02)
    assert decide(V)["report"]
    V["R0"]["weights"] = {"verdict": "DIFFERS"}
    assert decide(V)["carry"] is None
    del V["R0"]["weights"]
    assert decide(V)["carry"] is None
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=list(ARMS))
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--map-from", required=True)
    t.add_argument("--cache", default=None)
    t.add_argument("--via-dev", action="store_true", help="R0: cs_dev's train_dev on the pipeline's inputs, no cache")
    t.add_argument("--fit", default=None)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select", default=None)
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--pread", action="append", default=[])
    t.add_argument("--edges", action="append", default=[])
    t.add_argument("--epochs", type=int, default=C19.EPOCHS)
    t.add_argument("--per-epoch", type=int, default=C19.PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--device", default="cpu")
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    r = sub.add_parser("pread")
    r.add_argument("--run", required=True)
    r.add_argument("--pread", required=True)
    r.add_argument("--edges", required=True)
    r.add_argument("--cache", required=True)
    r.add_argument("--read-batch", type=int, default=BATCH)
    r.add_argument("--device", default="cpu")
    r.add_argument("--out", required=True)
    r.add_argument("--rows-out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--run", action="append", required=True)
    g.add_argument("--r0", action="append", default=[])
    g.add_argument("--control", action="append", required=True)
    g.add_argument("--mu-run", action="append", default=[])
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--pread", action="append", required=True)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("train", "pread", "grade"):
        res = {"train": train_cmd, "pread": pread_cmd, "grade": grade_cmd}[a.cmd](a)
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
