"""Design look (untracked; not a result and not filed): S6 part 8 of the transfer plan, more distributions in
training. Swastik (4 Oct, after the first read-time round): 'this can still be improved, more distributions should
work'; earlier the same day: 'train in multiple regimes getting from coarser to fine grained labels ... so that we can
do some sort of population estimation and better learning' and 'maybe some sort of KL divergence might help to map the
distributions'. Parts 6 (chainscore22) and 7 (chainscore23) map and relate the populations of one training mix; this
part varies the mix: how many graphs, and of which kind, the one model trains on, and whether a KL term pulls their
hidden distributions together. Part 5 (chainscore21) found one typed passage graph in training cost the KB zero-shot
read (Q1 BELOW, -3.3); round two of the read-time maps (cs21_rmdiag2, read 22:02) carried nothing.

Training graphs, in groups (chainscore21's draw21: the group uniform per example, then chainscore19's draw in it):
    KB          metaqa's fit carve (id) and, where the arm says so, its four granularity transforms (al4 sp4 mg3 rf;
                chainscore19's -gr builds): one group
    2wiki, hotpotqa, musique    each its typed passage fit carve (chainscore21's pbuild): a group each
Arms. f is part 5's MLP f (chainscore20's em-gr) with lb on every arm:
    k1                          KB id alone
    k5                          KB id and the four transforms (part 5's b-lb)
    k6                          k5 and 2wiki (part 5's mp-lb)
    k8                          k5 and 2wiki, hotpotqa and musique
    k8-lo-2w k8-lo-hp k8-lo-mu  k8 without one passage graph, which is then read zero-shot (leave one graph out)
    k6-kl01 k6-kl1 k8-kl01 k8-kl1   k6 / k8 with the KL term at lambda 0.1 / 1
The KL term (the kl arms). Per batch, for each group with at least MIN_ALIGN points in it, each unit of f's second
hidden pre-activation (over the pool nodes of that group's examples) and of g's (over their chains) is fitted by a
Gaussian (mean m, variance v + EPS_V). The term is the symmetric KL, 0.5 [v_a/v_b + v_b/v_a - 2 + (m_a - m_b)^2
(1/v_a + 1/v_b)], averaged over units and over pairs of groups, f's and g's added; lambda times it joins the loss. It
reads no label. Lambda 0 (every other arm) is cs_dev's train_dev statement for statement, with the select carve's
batches made once instead of every epoch (the same tensors).
Inputs. Every population (a build at one regime; a read is its whole carve) is mapped by part 6's carried map
(chainscore22's normalise: pz, pq or pqs), in training and in every read; if part 6 carried no map or its R0 DIFFERED,
by none (part 5's inputs). The map is read from part 6's grade (cs22.json) when a run starts; this rule was fixed
before part 6's grade existed. The first training build's stats then standardise, as parts 5 and 6. Each build's
prep, lb offsets, map and input stats come from cs_cache (one entry per build, kind and map, made once before the runs
and loaded by every run; cs_cache's check compares an entry to the pipeline run fresh, byte for byte).
Training as part 6: Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, the epoch chosen by
R@5 on metaqa's select carve (the KB rule), seeds SEED + 0, 1, 2 (the initial weights and the draw), every arm on the
host GPU with cs_dev's settings (deterministic algorithms, TF32 off): no number here is set beside a CPU fit's.
Reads as parts 5 and 6: the KB rule on metaqa x1f, webqsp selectf and webqsp selectf + fit (webqsp is never trained
on), the passage rule on 2wiki x1, hotpotqa x1 and musique x1f; each mapped by its own population under the arm's map.
Verdicts, fixed at 22:19 on 4 Oct (this file's first write), before part 6's grade and before any declared run (the
local smokes train two short epochs on stand-in inputs to catch crashes; no number of theirs is read). Per arm, each
row's R@5 averaged over its three seeds; paired row bootstrap (BOOT 1000):
    V1 (primary) webqsp selectf + fit: k8 minus k5: ABOVE / AT / BELOW
    V2 metaqa x1f: k8 minus k5
    V3 the curve, on every read: k5 - k1, k6 - k5, k8 - k6, k8 - k5
    V4 leave one graph out, for each passage graph X: on X's read, k8-lo-X minus k5 (the other passage graphs'
       transfer to X over KB-only training) and k8-lo-X minus k8 (the cost of holding X out); on webqsp selectf + fit
       and metaqa, k8-lo-X minus k8; on every passage read, every arm minus s0+rrf with its share of the twin's lead
       over s0+rrf (HIGH at or above 0.5, LOW below 0.25, else MID)
    V5 the KL term, on every read: k6-kl01 - k6, k6-kl1 - k6, k8-kl01 - k8, k8-kl1 - k8: HELPS / HURTS / SAME
    V6 webqsp selectf + fit by seed-gold distance D (1, 2): V1's and V5's differences
    V7 each KB read: every arm minus none/rd (the untyped walk)
    C  the candidates k6, k8, k6-kl01, k6-kl1, k8-kl01 and k8-kl1, each minus k5 on webqsp selectf + fit and on metaqa
    S  every difference above per seed, with its sign
    R0 k5 seed 0 reproduces part 6's arm under the same map (b-lb, or b-lb-<map>) seed 0 row for row (|diff| < 1e-6)
       on every read: REPRODUCES / DIFFERS (it checks the cache, the select-batch reuse and this file's loop at once)
Decision. A candidate whose C difference is ABOVE on webqsp selectf + fit and not BELOW on metaqa carries (the largest
webqsp selectf + fit mean if several): more distributions in training carry the zero-shot KB read, and the next parts
train on that mix. If none does, they do not (with this f and map), and the result goes to Swastik with parts 6 and 7.
R0 DIFFERS: nothing is read until its cause is found. Train-split rows throughout: a look, not a result.
    python outputs/mp_unified/chainscore24.py cache --map-from outputs/mp_unified/lean/cs22.json --kind train \
        --src outputs/mp_unified/lean/cs19-mq-fit-id.npz          (each training build; --kind plain: select, reads)
    python outputs/mp_unified/chainscore24.py train --arm k8 --seed 0 --device cuda \
        --map-from outputs/mp_unified/lean/cs22.json --cache outputs/mp_unified/cache \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz \
        (sp4, mg3, rf) --pfit 2wiki=outputs/mp_unified/lean/cs21-2w-fit.npz (hotpotqa, musique) \
        --select outputs/mp_unified/lean/cs19-mq-select.npz --read metaqa=... --read webqsp=... --read webqsp_sf=... \
        --pread 2wiki=... --pread hotpotqa=... --pread musique=... --out X.json --rows-out X.rows.npz
    python outputs/mp_unified/chainscore24.py grade --run X.json (every arm x seed) \
        --control outputs/mp_unified/lean/cs22-b-lb-s0.json (and b-lb-pz, -pq, -pqs: the one under the runs' map) \
        --read ... --pread ... --out outputs/mp_unified/lean/cs24.json
    python outputs/mp_unified/chainscore24.py --selftest
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
import cs_cache as CC  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
BATCH = C21.BATCH
F = "em-gr"
SEEDS = (0, 1, 2)
KB_READS, P_READS = C21.KB_READS, C21.P_READS
PDS = ("2wiki", "hotpotqa", "musique")
LO = {"2w": "2wiki", "hp": "hotpotqa", "mu": "musique"}
MIN_ALIGN, EPS_V = 32, 1e-3
TF4 = C19.TRANSFORMS[1:]


def _arm(kb, p, lam=0.0):
    return {"kb": tuple(kb), "p": tuple(p), "lam": float(lam)}


ARMS = {"k1": _arm((), ()), "k5": _arm(TF4, ()), "k6": _arm(TF4, ("2wiki",)), "k8": _arm(TF4, PDS),
        **{f"k8-lo-{s}": _arm(TF4, tuple(x for x in PDS if x != n)) for s, n in LO.items()},
        "k6-kl01": _arm(TF4, ("2wiki",), 0.1), "k6-kl1": _arm(TF4, ("2wiki",), 1.0),
        "k8-kl01": _arm(TF4, PDS, 0.1), "k8-kl1": _arm(TF4, PDS, 1.0)}
CANDS = ("k6", "k8", "k6-kl01", "k6-kl1", "k8-kl01", "k8-kl1")
KLP = (("k6-kl01", "k6"), ("k6-kl1", "k6"), ("k8-kl01", "k8"), ("k8-kl1", "k8"))
CURVE = (("k5", "k1"), ("k6", "k5"), ("k8", "k6"), ("k8", "k5"))
log = C21.log
ABV, HS = ("ABOVE", "BELOW", "AT"), ("HELPS", "HURTS", "SAME")


def sha(p):
    return CC.sha(p)


def map_rule(path):
    """Part 6's carried map, or none (no carry, an incomplete grade, R0 DIFFERS)."""
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if js.get("look") != "chainscore22":
        raise SystemExit(f"{path} is not part 6's grade")
    dec = js.get("decision") or {}
    c = dec.get("carry")
    norm = c if c in ("pz", "pq", "pqs") else "none"
    return norm, {"map_from": str(path), "map_from_sha256": sha(path), "part6_decision": dec}


def resolve_map(a):
    if (a.map is None) == (a.map_from is None):
        raise SystemExit("give --map-from (part 6's grade; the runs) or --map (a local smoke), not both")
    if a.map is not None:
        if a.map not in C22.NORMS:
            raise SystemExit(f"--map is one of {C22.NORMS}")
        return a.map, {"map_from": None, "map_given": a.map}
    return map_rule(a.map_from)


def get(p, kind, norm, cache):
    """(d, lb, map info, stats or None): from cs_cache's entry, or the pipeline's statements run here."""
    if cache:
        return CC.load(p, kind, norm, cache)
    d, lb, info = CC.pipeline(p, kind, norm)
    return d, lb, info, None


# ── the KL term ─────────────────────────────────────────────────────────────────────────────────────────────────


def kl_term(model, bt, gid):
    """Symmetric KL between the groups' per-unit Gaussian fits of f's and g's second hidden pre-activation: averaged
    over units and pairs of groups (groups with at least MIN_ALIGN points), f's and g's added. None if no pair."""
    import torch
    out = []
    for net, X, pos, L in ((model.f, bt["Xn"], bt.get("npos"), bt["Ln"]), (model.g, bt.get("Xp"), bt.get("ppos"),
                                                                            bt["Lk"])):
        if net is None or X is None or X.shape[0] == 0:
            continue
        H = net[:3](X)
        g = gid[torch.div(pos, L, rounding_mode="floor")]
        fits = []
        for k in range(int(gid.max()) + 1):
            m = g == k
            if int(m.sum()) < MIN_ALIGN:
                continue
            h = H[m]
            mu = h.mean(0)
            fits.append((mu, ((h - mu) ** 2).mean(0) + EPS_V))
        pairs = [(0.5 * (va / vb + vb / va - 2.0 + (ma - mb) ** 2 * (1.0 / va + 1.0 / vb))).mean()
                 for i, (ma, va) in enumerate(fits) for (mb, vb) in fits[i + 1:]]
        if pairs:
            out.append(torch.stack(pairs).mean())
    return torch.stack(out).sum() if out else None


# ── training: cs_dev's train_dev with the select batches made once and the KL term ─────────────────────────────


def select_batches(Sd, Slb, st, device, batch=BATCH):
    """read_dev's batches of the select carve (the KB rule; no companion), made once."""
    N = Sd["n"].size
    out = []
    for b0 in range(0, N, batch):
        rows = np.arange(b0, min(N, b0 + batch))
        items = np.c_[np.zeros(rows.size, np.int64), rows, np.zeros(rows.size, np.int64)]
        bt = C21.make_batch21(items, [Sd], [None], [Slb], st, pairs=True, train=False, gnn=False, att=False)
        out.append((rows, CD.to_dev(bt, device) if CD.placed(device) else bt))
    return out


def read_select(model, Sd, sb, device):
    """read_dev's KB-rule ranking over the select carve from the batches made once (R@5, FC@5, hit@1 per row)."""
    import torch
    N = Sd["n"].size
    out = np.zeros((N, 3))
    with torch.no_grad():
        for rows, bt in sb:
            if CD.placed(device):
                with torch.device(device):
                    s, _lp = C20.forward20(model, bt)
            else:
                s, _lp = C20.forward20(model, bt)
            s = s.double().cpu().numpy()
            for j, i in enumerate(rows):
                out[i] = C19.rank_row(Sd, i, s[j, :int(Sd["n"][i])])
    return out


def train24(arm, builds, LB, groups, gidb, Sd, Slb, st, epochs, per_epoch, batch, seed, lam, device=None,
            threads=1):
    """cs_dev's train_dev (MLP f, KB select rule) statement for statement, but for two changes: the select carve's
    batches are made once, and with lam > 0 lam times the KL term joins the loss. st is the first build's input stats
    (cs_cache's, or computed here as train_dev does)."""
    import torch
    block = CD.settings(device, threads)
    spec = C21.arm_spec("mp-lb" if len(groups) > 1 else "b-lb", F)
    if spec["gnn"] or not spec["g"]:
        raise SystemExit("this part trains the MLP f with g")
    st = st if st is not None else C19.input_stats(builds[0])
    model = C20.make_model(F, seed)
    if CD.placed(device):
        CD._dp().model_to(model, device)
    opt = torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)
    rng = np.random.default_rng(seed)
    sb = select_batches(Sd, Slb, st, device)
    gid_b = np.asarray(gidb, np.int64)
    hist, best, best_state = [], None, None
    tb, nbt = 0.0, 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for ep in range(epochs):
            t1 = time.time()
            model.train()
            items = C21.draw21(rng, builds, groups, per_epoch)
            tot, tn, tc, tk, nb = 0.0, 0.0, 0.0, 0.0, 0
            for b0 in range(0, len(items), batch):
                t2 = time.time()
                it = items[b0:b0 + batch]
                bt = C21.make_batch21(it, builds, [None] * len(builds), LB, st, pairs=spec["g"], gnn=spec["gnn"],
                                      att=spec["att"])
                gid = torch.from_numpy(gid_b[np.asarray(it, np.int64).reshape(-1, 3)[:, 0]]) if lam > 0 else None
                if CD.placed(device):
                    bt = CD.to_dev(bt, device)
                    with torch.device(device):
                        s, lp = C20.forward20(model, bt)
                        loss, ln, lc = C19.objective(s, lp, bt)
                        kl = kl_term(model, bt, gid.to(device)) if lam > 0 else None
                else:
                    s, lp = C20.forward20(model, bt)
                    loss, ln, lc = C19.objective(s, lp, bt)
                    kl = kl_term(model, bt, gid) if lam > 0 else None
                if kl is not None:
                    loss = loss + lam * kl
                    tk += float(kl.detach())
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
                         **C20.coupling20(model), "seconds": round(time.time() - t1, 1),
                         **({"kl": round(tk / max(nb, 1), 5)} if lam > 0 else {})})
            log(f"    {arm} ({F}) ep {ep}: loss {tot / max(nb, 1):.4f} (node {tn / max(nb, 1):.4f}, chain "
                f"{tc / max(nb, 1):.4f}{f', kl {tk / max(nb, 1):.4f}' if lam > 0 else ''}) select R@5 {sr5:.4f} "
                f"{C20.coupling20(model)} {time.time() - t1:.0f}s")
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
    A = ARMS[a.arm]
    norm, minfo = resolve_map(a)
    dev = None if a.device == "cpu" else a.device
    res = {"look": "chainscore24", "arm": a.arm, "seed": a.seed, "f": F, "norm": norm, "lam": A["lam"],
           "device": a.device, "args": dict(vars(a)), "map_rule": minfo, "via_dev": bool(a.via_dev),
           "cache": a.cache, "script_sha256": sha(__file__), "cs_cache_sha256": sha(CC.__file__),
           "chainscore22_sha256": sha(C22.__file__), "chainscore21_sha256": sha(C21.__file__),
           "chainscore20_sha256": sha(C20.__file__), "chainscore19_sha256": sha(C19.__file__),
           "chainscore18_sha256": sha(CS.__file__), "cs_dev_sha256": sha(CD.__file__), "inputs": {},
           "inputs_sha256": {}, "passage_types": {}, "maps": {}, "reads": {}, "rows_out": a.rows_out, "timing": {}}
    aug = dict(x.split("=", 1) for x in a.aug)
    pfit = dict(x.split("=", 1) for x in a.pfit)
    if sorted(aug) != sorted(A["kb"]):
        raise SystemExit(f"arm {a.arm} takes --aug for exactly {A['kb'] or 'none'}")
    if sorted(pfit) != sorted(A["p"]):
        raise SystemExit(f"arm {a.arm} takes --pfit for exactly {A['p'] or 'none'}")
    if not a.fit or not a.select:
        raise SystemExit("every arm takes --fit and --select")
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_READS):
        raise SystemExit(f"every arm reads {KB_READS} (--read) and {P_READS} (--pread)")
    if a.via_dev and (A["lam"] > 0 or a.cache):
        raise SystemExit("--via-dev (a check) runs cs_dev's train_dev itself: no KL term, no cache")
    train_in = [("fit", a.fit, False)] + [(t, aug[t], False) for t in TF4 if t in aug] + \
        [(n, pfit[n], True) for n in PDS if n in pfit]
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    t1 = time.time()
    for n, p, is_p in train_in + [("select", a.select, False)] + reads:
        res["inputs_sha256"][p] = sha(p)
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], None)
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    res["timing"]["hash_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    builds, LB, st0 = [], [], None
    for n, p, is_p in train_in:
        d, lb, info, st = get(p, "train", norm, a.cache)
        tf = d["meta"]["transform"]["name"]
        if tf != ("id" if n == "fit" or is_p else n):
            raise SystemExit(f"{p} holds transform {tf}, not {'id' if n == 'fit' or is_p else n}")
        if not builds:
            st0 = st
        res["inputs"][n] = d["meta"]
        res["maps"][n] = info
        LB.append(lb)
        builds.append(d)
        log(f"  {a.arm} s{a.seed}: {n} ready ({'cache' if a.cache else 'pipeline'}; {info})")
    nkb = 1 + len(A["kb"])
    groups = [list(range(nkb))] + [[nkb + j] for j in range(len(A["p"]))]
    gidb = [0] * nkb + [1 + j for j in range(len(A["p"]))]
    Sd, Slb, sinfo, _ = get(a.select, "plain", norm, a.cache)
    res["maps"]["select"] = sinfo
    res["inputs"]["select"] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p, _q), b in zip(train_in, builds)}
    res["groups"] = groups
    res["timing"]["load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    if a.via_dev:
        model, info = CD.train_dev("mp-lb" if len(groups) > 1 else "b-lb", F, builds, [None] * len(builds), LB,
                                   groups, Sd, None, Slb, "kb", a.epochs, a.per_epoch, a.batch, SEED + a.seed,
                                   device=dev)
    else:
        model, info = train24(a.arm, builds, LB, groups, gidb, Sd, Slb, st0, a.epochs, a.per_epoch, a.batch,
                              SEED + a.seed, A["lam"], device=dev)
    res["timing"]["train_s"] = round(time.time() - t1, 1)
    del builds, LB, Sd, Slb
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings")})
    t1 = time.time()
    rows_out = {}
    for n, p, is_p in reads:
        t2 = time.time()
        d, lb, rinfo, _ = get(p, "plain", norm, a.cache)
        res["maps"][n] = rinfo
        rule = "passage" if is_p else "kb"
        xr, ml = CD.read_dev(model, d, None, lb, info["stats"], rule, dev, batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule=rule)
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} s{a.seed} on {n} ({rule} rule): {res['reads'][n]['mean']} ({time.time() - t2:.0f}s)")
        del d
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
                    "spec": info["spec"], "arm": a.arm, "seed": a.seed, "norm": norm}, sp_.with_name(sp_.name + ".tmp"))
        os.replace(sp_.with_name(sp_.name + ".tmp"), sp_)
    res["seconds"] = round(time.time() - t0, 1)
    log(f"  {a.arm} s{a.seed}: state {res['state_sha256'][:16]}, timing {res['timing']}, {res['seconds']}s")
    return res


def cache_cmd(a):
    norm, minfo = resolve_map(a)
    ent = CC.build(a.src, a.kind, norm, a.root)
    return {"look": "chainscore24-cache", "src": a.src, "kind": a.kind, "norm": norm, "map_rule": minfo,
            "entry": {k: ent[k] for k in ("source_sha256", "arrays", "lb_sha256", "seconds") if k in ent}}


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def grade_cmd(a):
    t0 = time.time()
    runs, rows = {}, {}
    for p_ in a.run:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore24":
            raise SystemExit(f"{p_} is not a chainscore24 run")
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
    norms = {js["norm"] for js in runs.values()}
    if len(devs) != 1 or len(shas) != 1 or len(norms) != 1 or any(js.get("via_dev") for js in runs.values()):
        raise SystemExit(f"runs on several devices {devs}, scripts {[s[:12] for s in shas]} or maps {norms}, or a "
                         f"--via-dev check among them")
    dev, norm = next(iter(devs)), next(iter(norms))
    refs, Ds, rule = {}, {}, {}
    for flag, xs in (("kb", a.read), ("passage", a.pread)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            h = sha(p_)
            for k, js in runs.items():
                if js["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"run {k} read another {nm_} than {p_}")
            d = CS.load(p_)
            refs[nm_] = CS.reference_arms(d) if flag == "kb" else C21.passage_refs(d)
            Ds[nm_] = d["D"]
            rule[nm_] = flag
            del d
    if sorted(k for k in rule if rule[k] == "kb") != sorted(KB_READS) or \
            sorted(k for k in rule if rule[k] == "passage") != sorted(P_READS):
        raise SystemExit(f"the grade reads {KB_READS} (--read) and {P_READS} (--pread)")

    def r5(arm, nm_, s=None):
        if s is not None:
            return rows[(arm, s)][nm_][:, 0].astype(np.float64)
        return np.mean([rows[(arm, k)][nm_][:, 0].astype(np.float64) for k in SEEDS], 0)

    V = {k: {} for k in ("V1", "V2", "V3", "V4", "V5", "V6", "V7", "C", "S", "R0")}

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

    for nm_ in refs:
        N = refs[nm_]["twin0"].shape[0]
        for k, r in rows.items():
            if r[nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {r[nm_].shape[0]} rows, the build {N}")
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        if nm_ == "webqsp_sf":
            V["V1"] = pair("k8", "k5", nm_, idx)
            for dd in (1, 2):
                m = Ds[nm_] == dd
                if m.any():
                    ii = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                    V["V6"][f"D{dd}"] = {f"{x_} - {y_}": dict(ci((r5(x_, nm_) - r5(y_, nm_))[m],
                                                                 ABV if (x_, y_) == ("k8", "k5") else HS, ii),
                                                              rows=int(m.sum()))
                                         for x_, y_ in (("k8", "k5"),) + KLP}
        if nm_ == "metaqa":
            V["V2"] = pair("k8", "k5", nm_, idx)
        if nm_ in ("webqsp_sf", "metaqa"):
            V["C"][nm_] = {f"{x_} - k5": pair(x_, "k5", nm_, idx) for x_ in CANDS}
            V["V4"][nm_] = {f"k8-lo-{s} - k8": pair(f"k8-lo-{s}", "k8", nm_, idx) for s in LO}
        V["V3"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in CURVE}
        V["V5"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx, HS) for x_, y_ in KLP}
        if rule[nm_] == "kb":
            walk = refs[nm_]["none/rd"][:, 0]
            V["V7"][nm_] = {f"{x_} - none/rd": pair(x_, walk, nm_, idx) for x_ in ARMS}
        else:
            s_ = next(s for s, n in LO.items() if n == nm_)
            e = {f"k8-lo-{s_} - k5": pair(f"k8-lo-{s_}", "k5", nm_, idx),
                 f"k8-lo-{s_} - k8": pair(f"k8-lo-{s_}", "k8", nm_, idx)}
            s0r, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["twin0"][:, 0]
            for x_ in ARMS:
                X_ = r5(x_, nm_)
                sh = CP.share(X_, s0r, tw, idx)
                e[f"{x_} - s0+rrf"] = dict(ci(X_ - s0r, ABV, idx), share=sh,
                                           share_verdict="HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID")
            V["V4"][nm_] = e
    want_ctl = "b-lb" if norm == "none" else f"b-lb-{norm}"
    ctl = None
    for p_ in a.control or []:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") == "chainscore22" and js.get("arm") == want_ctl and int(js.get("seed", -1)) == 0:
            ctl = (p_, js)
    if ctl is not None:
        p_, js = ctl
        if js.get("device") != dev:
            raise SystemExit(f"R0 reads part 6's {want_ctl} on the runs' device ({dev}), not {js.get('device')}")
        with np.load(js["rows_out"]) as z:
            c22 = {k: z[k] for k in z.files}
        for nm_ in refs:
            dif = float(np.abs(rows[("k5", 0)][nm_].astype(np.float64) - c22[nm_].astype(np.float64)).max())
            V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
        V["R0"]["control"] = p_
        V["R0"]["state_sha256"] = [runs[("k5", 0)].get("state_sha256"), js.get("state_sha256")]
    res = {"look": "chainscore24", "args": dict(vars(a)), "script_sha256": sha(__file__), "run_script": shas.pop(),
           "device": dev, "norm": norm, "map_rule": runs[("k5", 0)].get("map_rule"),
           "runs": {f"{k[0]}#s{k[1]}": {"best_epoch": js["best_epoch"], "best_select_r5": js["best_select_r5"],
                                        "state_sha256": js.get("state_sha256"), "timing": js.get("timing"),
                                        "reads": {n: v["mean"] for n, v in js["reads"].items()}}
                    for k, js in sorted(runs.items())},
           "means": {nm_: {arm: [round(float(np.mean([rows[(arm, s)][nm_][:, c].astype(np.float64).mean()
                                                       for s in SEEDS])), 5) for c in range(3)] for arm in ARMS}
                     for nm_ in refs},
           "refs": {nm_: {k: [round(float(u), 5) for u in x.mean(0)] for k, x in r.items()
                          if k in ("rrf", "twin0", "none/rd", "s0+rrf", "s0+walk")} for nm_, r in refs.items()},
           "verdicts": V, "decision": decide(V)}
    res["seconds"] = round(time.time() - t0, 1)
    log(f"decision {res['decision']}")
    return res


def decide(V):
    r0 = [(V["R0"].get(k) or {}).get("verdict") for k in KB_READS + P_READS]
    if None in r0:
        return {"carry": None, "why": "R0 incomplete (no --control under the runs' map)", "report": True}
    if "DIFFERS" in r0:
        return {"carry": None, "why": "R0 DIFFERS: nothing is read until its cause is found", "report": True}
    ok = []
    for x_ in CANDS:
        w = (V["C"].get("webqsp_sf") or {}).get(f"{x_} - k5") or {}
        m = (V["C"].get("metaqa") or {}).get(f"{x_} - k5") or {}
        if w.get("verdict") == "ABOVE" and m.get("verdict") != "BELOW":
            ok.append((w["diff"][0], x_))
    if not ok:
        return {"carry": "none", "why": "no candidate is ABOVE k5 on webqsp selectf + fit without BELOW on metaqa",
                "report": True}
    return {"carry": max(ok)[1], "candidates": [x_ for _d, x_ in sorted(ok, reverse=True)], "report": False,
            "why": f"ABOVE on webqsp selectf + fit and not BELOW on metaqa: {[x_ for _d, x_ in ok]}"}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def selftest():
    import inspect
    import tempfile

    import torch
    # 1. the arm table
    assert set(CANDS) <= set(ARMS) and all(x in ARMS and y in ARMS for x, y in KLP + CURVE)
    assert ARMS["k5"]["kb"] == TF4 and ARMS["k1"]["kb"] == () and ARMS["k8"]["p"] == PDS
    for s, n in LO.items():
        assert n not in ARMS[f"k8-lo-{s}"]["p"] and len(ARMS[f"k8-lo-{s}"]["p"]) == 2
    assert {a_ for a_ in ARMS if ARMS[a_]["lam"] > 0} == {"k6-kl01", "k6-kl1", "k8-kl01", "k8-kl1"}
    # 2. the KL term: zero for one distribution split in two groups, symmetric, positive when they differ, skips
    #    small groups, its gradient reaches f's and g's first layers
    model = C20.make_model(F, 0)
    rng = np.random.default_rng(0)
    B, Ln, Lk = 4, 50, 30
    X = torch.from_numpy(rng.normal(size=(B * Ln, len(C19.NODEF))).astype(np.float32))
    Xp = torch.from_numpy(rng.normal(size=(B * Lk, len(C19.PAIRF))).astype(np.float32))
    bt = {"Xn": X, "npos": torch.arange(B * Ln), "Ln": Ln, "Xp": Xp, "ppos": torch.arange(B * Lk), "Lk": Lk}
    same = {**bt, "Xn": torch.cat([X[:2 * Ln], X[:2 * Ln]]), "Xp": torch.cat([Xp[:2 * Lk], Xp[:2 * Lk]])}
    g01 = torch.tensor([0, 0, 1, 1])
    assert abs(float(kl_term(model, same, g01))) < 1e-6
    sh = {**bt, "Xn": torch.cat([X[:2 * Ln], X[2 * Ln:] * 3 + 2]), "Xp": torch.cat([Xp[:2 * Lk], Xp[2 * Lk:] - 1])}
    k1 = kl_term(model, sh, g01)
    k2 = kl_term(model, sh, torch.tensor([1, 1, 0, 0]))
    assert float(k1) > 0.1 and abs(float(k1) - float(k2)) < 1e-5
    assert kl_term(model, bt, torch.tensor([0, 0, 0, 0])) is None     # one group: no pair
    tiny = {**bt, "npos": torch.arange(B * Ln), "ppos": torch.arange(B * Lk)}
    assert kl_term(model, tiny, torch.tensor([0, 0, 0, 1])) is not None   # 50 nodes, 30 chains per row: >= 32 / ok
    model.zero_grad()
    k1 = kl_term(model, sh, g01)
    k1.backward()
    assert float(model.f[0].weight.grad.abs().sum()) > 0 and float(model.g[0].weight.grad.abs().sum()) > 0
    assert model.f[4].weight.grad is None or float(model.f[4].weight.grad.abs().sum()) == 0
    # 3. the map rule reads part 6's carry
    with tempfile.TemporaryDirectory() as td:
        for carry, want in (("pq", "pq"), ("pz", "pz"), ("pqs", "pqs"), ("none", "none"), (None, "none")):
            p = Path(td) / "cs22.json"
            p.write_text(json.dumps({"look": "chainscore22", "decision": {"carry": carry}}), encoding="utf-8")
            assert map_rule(p)[0] == want
        p.write_text(json.dumps({"look": "chainscore21"}), encoding="utf-8")
        try:
            map_rule(p)
            raise AssertionError("another look accepted")
        except SystemExit:
            pass
    # 4. train24 keeps train_dev's statements (no KL term): the same calls in the same order
    src_dev, src_24 = inspect.getsource(CD.train_dev), inspect.getsource(train24)
    for s_ in ("C20.make_model(", "torch.optim.Adam(model.parameters(), lr=C19.LR, weight_decay=C19.WD)",
               "np.random.default_rng(seed)", "C21.draw21(rng, builds, groups, per_epoch)",
               "C19.objective(s, lp, bt)", "opt.zero_grad()", "loss.backward()", "opt.step()",
               "if best is None or sr5 > best[1]:", "model.load_state_dict(best_state)"):
        assert s_ in src_dev and s_ in src_24, s_
    # 5. decide
    V = {"R0": {k: {"verdict": "REPRODUCES"} for k in KB_READS + P_READS},
         "C": {"webqsp_sf": {f"{x} - k5": {"verdict": "AT", "diff": [0.0, -0.01, 0.01]} for x in CANDS},
               "metaqa": {f"{x} - k5": {"verdict": "AT", "diff": [0.0, -0.01, 0.01]} for x in CANDS}}}
    assert decide(V)["carry"] == "none"
    V["C"]["webqsp_sf"]["k8"] = {"verdict": "ABOVE", "diff": [0.02, 0.01, 0.03]}
    V["C"]["webqsp_sf"]["k6-kl1"] = {"verdict": "ABOVE", "diff": [0.03, 0.01, 0.05]}
    V["C"]["webqsp_sf"]["k8 - k5"] = V["C"]["webqsp_sf"].pop("k8")
    V["C"]["webqsp_sf"]["k6-kl1 - k5"] = V["C"]["webqsp_sf"].pop("k6-kl1")
    assert decide(V)["carry"] == "k6-kl1"
    V["C"]["metaqa"]["k6-kl1 - k5"] = {"verdict": "BELOW", "diff": [-0.02, -0.03, -0.01]}
    assert decide(V)["carry"] == "k8"
    V["R0"]["musique"] = {"verdict": "DIFFERS"}
    assert decide(V)["carry"] is None
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=list(ARMS))
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--map-from", default=None)
    t.add_argument("--map", default=None)
    t.add_argument("--cache", default=None)
    t.add_argument("--via-dev", action="store_true", help="a check: cs_dev's train_dev itself, no cache")
    t.add_argument("--fit", default=None)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--pfit", action="append", default=[])
    t.add_argument("--select", default=None)
    t.add_argument("--read", action="append", default=[])
    t.add_argument("--pread", action="append", default=[])
    t.add_argument("--epochs", type=int, default=C19.EPOCHS)
    t.add_argument("--per-epoch", type=int, default=C19.PER_EPOCH)
    t.add_argument("--batch", type=int, default=BATCH)
    t.add_argument("--read-batch", type=int, default=BATCH)
    t.add_argument("--device", default="cpu")
    t.add_argument("--out", required=True)
    t.add_argument("--rows-out", required=True)
    t.add_argument("--state-out")
    c = sub.add_parser("cache")
    c.add_argument("--map-from", default=None)
    c.add_argument("--map", default=None)
    c.add_argument("--kind", required=True, choices=CC.KINDS)
    c.add_argument("--src", required=True)
    c.add_argument("--root", default=str(CC.ROOT))
    c.add_argument("--out", default=None)
    g = sub.add_parser("grade")
    g.add_argument("--run", action="append", required=True)
    g.add_argument("--control", action="append", default=[], help="part 6's b-lb arms, seed 0 (R0)")
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--pread", action="append", required=True)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd in ("train", "grade", "cache"):
        res = {"train": train_cmd, "grade": grade_cmd, "cache": cache_cmd}[a.cmd](a)
        if a.out:
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
