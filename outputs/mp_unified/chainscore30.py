"""S6 part 14 (chainscore30.py): EM as the balancer, and message passing as going back to the neighbours.

Swastik's framing (5 Oct). The MLP and the GNN use the same features. Going back means summarising the neighbours'
features through message passing: the GNN learns that summary (and can send messages back along the edges); the MLP
scores static inputs and cannot, though it can be given static summaries (sg2's propagated inputs [x, P x, P^2 x]).
EM is not message passing and both models run it: the E-step estimates the chain posterior (which relation chain, at
which granularity, none / dir / typed / exact, explains the question, given the node scores), the M-step re-scores the
nodes with the mass of the chains that reach them. EM is meant to be the balancer of the richer forms (relations,
transformations, levels) against the output. Two limits of the EM so far: it runs a fixed T = 2 rounds per question,
and its balance (the coupling and each level's prior share) is learned once on metaqa and never re-estimated on a new
graph.

Classes (every one on part 9's k5 mix: metaqa fit + the four transforms, pq map, lb; seeds 0, 1, 2):
    mlp  chainscore24 k5          the MLP f: static node inputs, no summary
    sg2  chainscore29 sg2         the MLP f over static propagated inputs (K = 2 steps per direction)
    en   chainscore25 n-b-lb-pq   the untyped mean GNN (L = 2 layers; no coupling)
    rgu  chainscore29 rgu         the relational GNN without the posterior (it passes messages once)
    ena  chainscore25 e-b-lb-pq   the GNN with EM-typed attention (it reruns each EM round)
    rga  chainscore29 rga         RGCN bases over the posterior's soft types with GATv2 attention (reruns each round)

(A) EM rounds at read time (sweep): each saved zero-shot run of mlp, sg2, rgu, ena and rga (the parts 9 and 13 runs,
    on their GPU) read again at T in {0, 1, 2, 3, 4, 6, 10} on metaqa x1f, webqsp and webqsp selectf + fit
    (webqsp_sf). Nothing is retrained. Identity: at T = 2 each read must equal the run's own rows bit for bit; a run
    that fails is excluded from the grade and listed.
(B) EM over the population at read time (balance): the level mix re-estimated on the read graph's own unlabeled
    questions, as Saerens et al.'s prior-shift EM. The levels are taken coarse (none, dir, typed = every tt level,
    exact; the typed levels' number differs by graph). pi_s is the run's own metaqa x1f read's mass by coarse level
    (the source mix; unlabeled). Read r = 0 is the run itself (offsets 0; identity as in (A)); after read r, with m_r
    its mean posterior mass by coarse level, read r + 1 adds w = clip(log max(m_r, 1e-6) - log max(pi_s, 1e-6), -4, 4)
    to each chain's prior logit by its coarse level. Stop when max |m_r - m_{r-1}| < 1e-4 or at r = 10. Same runs and
    reads as (A), at the run's T = 2. A second, damped sequence (map) re-estimates the mix as a Dirichlet MAP with the
    source mix as its prior at the read's own weight: pi_{r+1} = (m_r + pi_s) / 2, w = log pi_{r+1} - log pi_s (same
    stop, same clip). The plain sequence (mle) is the primary read of (B); map is secondary.
    Before these rules were committed, local smokes of (B) were run (the k5 run, seed 0, on webqsp's 305 rows; part
    13's rgu smoke run on 64 rows; CPU): the plain sequence moved the mix toward the typed corner (typed 0.31 -> 0.97
    and 0.60 -> 0.995) and R@5 fell on both. That is the earlier finding that the likelihood does not identify the
    level; the damped sequence was added after it (its own smoke, run once to check the code, fell less), and the
    plain one kept as the primary read.
(C) training arms, through each family's own train command; this file changes one thing per arm:
    L  message-passing layers of a GNN class (chainscore20's NLAYER, and chainscore29's copy): L = 0 is the same model
       with message passing off (its input and output networks only). en, rgu: L in {0, 1, 3, 4}; ena, rga: L in
       {0, 1, 3} (the edge files hold chain slots for two steps: a third layer's attention has no posterior mass and
       is uniform).
    K  sg2's static propagation steps per direction, K in {0, 1, 3, 4} (K = 0: x alone).
    dir  messages (or static summaries) of one direction only: fwd drops the backward edges (no going back along an
       edge), bwd drops the forward ones. GNN classes and sg2.
    T  EM rounds in training, T in {0, 1, 3, 4}: mlp, sg2, rgu, ena, rga (en has no coupling).
    Devices: mlp, sg2, en and rgu on the CPU (one thread; their base arm, the family's run unchanged, is the CPU
    control, trained here); ena, rga and sg2g (sg2's K and dir arms again, on the GPU) on the GPU, controlled by their
    parts 9 and 13 runs (same GPU, same code path).

Grade (R@5 per question; seeds averaged per row; paired row bootstrap, BOOT resamples; ABOVE if the CI's lower end > 0,
BELOW if its upper end < 0, AT otherwise). webqsp_sf is the primary read, metaqa the guard. CARRY (part 13's rule):
ABOVE on webqsp_sf and not BELOW on metaqa, or AT on webqsp_sf and ABOVE on metaqa.
    A1 per class, D(T) = read at T minus read at 2; MORE_ROUNDS if some T > 2 carries. A2 D(0), D(1) labels.
    A3 [ena(T) - ena(2)] - [mlp(T) - mlp(2)] and [rga(T) - rga(2)] - [rgu(T) - rgu(2)], per T and read.
    B1 per class, final minus r = 0 and r = 1 minus r = 0 (mle and map); BALANCER carries for a class if mle's final
       carries (BALANCER_MAP, secondary: map's final).
    B2 [ena_fin - ena_0] - [mlp_fin - mlp_0] and [rga_fin - rga_0] - [rgu_fin - rgu_0] (mle and map).
    C1 per class and arm, the arm minus its control; CARRY as above; the class's choice per axis is the carried arm
       with the largest webqsp_sf mean (none carried: the base stays).
    C2 message passing needed: L0 against the control, per read (BELOW: needed on that read); the static summary
       needed: sg2 K0 against its control.
    C3 learned against static summaries, within a device: [g - g_L0] - [s - s_K0] for g in (en, rgu) with s = sg2
       (CPU) and g in (ena, rga) with s = sg2g (GPU).
    C4 going back along the edges: fwd against the control (BELOW: the backward messages are needed).
    2wiki and hotpotqa (passage reads, in every train run) are reported as secondary deltas, without a rule.
No webqsp label is used in training, selection or the read-time EM; webqsp train_holdout and test are not touched.
Smokes write under smoke30/ and are never graded.

Rules fixed at 11:20 on 5 Oct 2026, before any number of parts 12, 13 or 14 was looked at (the smokes above aside).
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import chainscore24 as C24  # noqa: E402
import chainscore25 as C25  # noqa: E402
import chainscore28 as C28  # noqa: E402
import chainscore29 as C29  # noqa: E402
import cs_cache as CC  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
SEEDS = (0, 1, 2)
KB_READS = C21.KB_READS                  # metaqa, webqsp, webqsp_sf
P_SECOND = ("2wiki", "hotpotqa")
PRIMARY, GUARD = "webqsp_sf", "metaqa"
T_READ = (0, 1, 2, 3, 4, 6, 10)
T_OWN = C19.T_EM                         # 2
L_OWN, K_OWN = C20.NLAYER, 2             # 2 layers; sg2's two steps per direction
R_MAX, TOL, FLOOR, WCLIP = 10, 1e-4, 1e-6, 4.0
COARSE = ("none", "dir", "typed", "exact")
# class: (family script, arm, f, device, {axis: values})
CLASSES = {"mlp": ("chainscore24", "k5", "em-gr", "cpu", {"T": (0, 1, 3, 4)}),
           "sg2": ("chainscore29", "sg2", "sg2", "cpu", {"K": (0, 1, 3, 4), "dir": ("fwd", "bwd"), "T": (0, 1, 3, 4)}),
           "en": ("chainscore25", "n-b-lb-pq", "en-gr", "cpu", {"L": (0, 1, 3, 4), "dir": ("fwd", "bwd")}),
           "rgu": ("chainscore29", "rgu", "rgu", "cpu", {"L": (0, 1, 3, 4), "dir": ("fwd", "bwd"), "T": (0, 1, 3, 4)}),
           "ena": ("chainscore25", "e-b-lb-pq", "ena-gr", "cuda", {"L": (0, 1, 3), "dir": ("fwd", "bwd"),
                                                                   "T": (0, 1, 3, 4)}),
           "rga": ("chainscore29", "rga", "rga", "cuda", {"L": (0, 1, 3), "dir": ("fwd", "bwd"), "T": (0, 1, 3, 4)}),
           "sg2g": ("chainscore29", "sg2", "sg2", "cuda", {"K": (0, 1, 3, 4), "dir": ("fwd", "bwd")})}
GPU_CONTROL = ("ena", "rga", "sg2g")     # controlled by the parts 9 / 13 runs
READ_CLASSES = ("mlp", "sg2", "rgu", "ena", "rga")
PAIRS_T = (("ena", "mlp"), ("rga", "rgu"))
PAIRS_LS = (("en", "sg2"), ("rgu", "sg2"), ("ena", "sg2g"), ("rga", "sg2g"))
MOD = {"chainscore24": C24, "chainscore25": C25, "chainscore29": C29}
OUT_PREFIX = "outputs/mp_unified/lean/cs30-"
SMOKE_PREFIX = "outputs/mp_unified/smoke30/"
DIRS = {"fwd": 1, "bwd": 0}              # the direction whose edges are dropped
log, sha = C21.log, CC.sha


def cls_of(js):
    """The read class of a family run json (look, arm, f), or None (sg2's runs read as sg2)."""
    for k in READ_CLASSES:
        look, arm, f = CLASSES[k][:3]
        if js.get("look") == look and js.get("arm") == arm and js.get("f") == f:
            return k
    return None


def label(ci):
    return "ABOVE" if ci[1] > 0 else "BELOW" if ci[2] < 0 else "AT"


def carry(lp, lg):
    return bool((lp == "ABOVE" and lg != "BELOW") or (lp == "AT" and lg == "ABOVE"))


def boot_ci(x, rng):
    return C28.boot_ci(x, rng)


def coarse_map(names):
    out = []
    for n in [str(x) for x in names]:
        if n in ("none", "dir", "exact"):
            out.append(COARSE.index(n))
        elif n.startswith("tt"):
            out.append(2)
        else:
            raise SystemExit(f"unknown level {n}")
    return np.asarray(out, np.int64)


def coarse_mass(ml, names):
    m = np.bincount(coarse_map(names), weights=np.asarray(ml, np.float64), minlength=len(COARSE))
    return m / max(m.sum(), 1e-12)


# ── patches (part 15 uses the runner and the batch wrapper too) ─────────────────────────────────────────────────


def flag(rest, name):
    if rest.count(name) != 1 or rest.index(name) + 1 >= len(rest):
        raise SystemExit(f"the family's arguments take {name} exactly once")
    return rest[rest.index(name) + 1]


def patch_threads(n):
    """Every family calls cs_dev's settings(device, threads) before it trains or reads: this sets the thread count."""
    orig = CD.settings

    def settings(device, threads):
        return orig(device, n)

    CD.settings = settings


def wrap_batch(post):
    """Every train and read batch of the families is made by chainscore21's make_batch21 (called through the module):
    post(bt, args) edits each batch after it is made."""
    orig = C21.make_batch21

    def make_batch21(*args, **kw):
        bt = orig(*args, **kw)
        post(bt, args)
        return bt

    C21.make_batch21 = make_batch21


def patch_T(look, T):
    """The model's EM rounds, set right after the family makes it (a plain attribute: no parameter changes)."""
    if look == "chainscore29":
        orig29 = C29.make_model29

        def make_model29(kind, seed):
            m = orig29(kind, seed)
            m.T = int(T)
            return m

        C29.make_model29 = make_model29
    else:
        orig20 = C20.make_model

        def make_model(arm, seed):
            m = orig20(arm, seed)
            m.T = int(T)
            return m

        C20.make_model = make_model


def patch_L(L):
    """Message-passing layers: chainscore20's NLAYER (its GNN, batch slots and marginals) and chainscore29's copy."""
    C20.NLAYER = int(L)
    C29.NLAYER = int(L)


def patch_K(K):
    """sg2's static propagation steps per direction: its input width and its sign_inputs (K = 2 is the original's
    statements in the original's order)."""
    import torch
    K = int(K)
    C29.NSIGN = 1 + 2 * K

    def sign_inputs(X, E):
        out = [X]
        for d in (0, 1):
            src, dst, _eix, cnt = E[d]
            inv = (1.0 / cnt.clamp_min(1.0))[:, None]
            Z = X
            for _ in range(K):
                Z = torch.zeros_like(Z).index_add(0, dst, Z[src]) * inv if src.numel() else torch.zeros_like(Z)
                out.append(Z)
        return torch.cat(out, 1)

    C29.sign_inputs = sign_inputs


def patch_dir(which):
    """Drop one direction's edges from every batch (a GNN skips a direction with no edge; sg2's summaries of it are
    zero)."""
    drop = DIRS[which]
    seen = {"batches": 0}

    def post(bt, _args):
        if "E" in bt:
            src, dst, eix, cnt = bt["E"][drop]
            bt["E"][drop] = (src[:0], dst[:0], eix[:0], cnt * 0)
            seen["batches"] += 1

    wrap_batch(post)
    return seen


BAL = {"W": None}


def patch_balance():
    """Read-time level offsets: each chain's prior logit gains BAL['W'][coarse level] (lp0 = log softmax(z - lc))."""
    def post(bt, args):
        W = BAL["W"]
        if W is None or bt.get("Xp") is None:
            return
        items, builds = np.asarray(args[0], np.int64).reshape(-1, 3), args[1]
        lc = bt["lc"]
        for j, (bi, i, _ri) in enumerate(items):
            d = builds[bi]
            p0, p1 = int(d["rowoff"][i]), int(d["rowoff"][i + 1])
            if p1 > p0:
                cm = coarse_map(d["level_names"])
                off = W[cm[d["p_lev"][p0:p1].astype(np.int64)]].astype(np.float32)
                lc[j, :p1 - p0] -= lc.new_tensor(off)

    wrap_batch(post)


def run_family(look, arm, device, rest, threads, tag, extra, smoke, prefix=OUT_PREFIX):
    """Check the family's train arguments, run its main, and add {tag: extra} to the json it wrote."""
    if not rest or rest[0] != "train":
        raise SystemExit("give the family's train command after --")
    if flag(rest, "--arm") != arm:
        raise SystemExit(f"this class trains {look}'s arm {arm}")
    if flag(rest, "--device") != device:
        raise SystemExit(f"this class trains on {device}")
    if int(flag(rest, "--seed")) not in SEEDS:
        raise SystemExit(f"seeds {SEEDS}")
    pre = SMOKE_PREFIX if smoke else prefix
    outs = [flag(rest, x) for x in ("--out", "--rows-out", "--state-out")]
    if not all(o.startswith(pre) for o in outs):
        raise SystemExit(f"--out, --rows-out and --state-out go under {pre}")
    if not smoke and any(x in rest for x in ("--smoke", "--epochs", "--per-epoch", "--read-limit", "--via-dev")):
        raise SystemExit("a run takes its family's defaults (no --smoke, --epochs, --per-epoch, --read-limit, --via-dev)")
    patch_threads(threads)
    t0 = time.time()
    rc = MOD[look].main(rest)
    if rc:
        raise SystemExit(f"{look} returned {rc}")
    p = Path(outs[0])
    js = json.loads(p.read_text(encoding="utf-8"))
    js[tag] = dict(extra, look=tag, threads=threads, script_sha256=sha(__file__), wrapper_seconds=round(
        time.time() - t0, 1))
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(js, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)),
                   encoding="utf-8")
    os.replace(tmp, p)
    return js


def train_cmd(a, rest):
    look, arm, f, device, axes = CLASSES[a.cls]
    if a.axis == "base":
        if a.cls in GPU_CONTROL:
            raise SystemExit(f"{a.cls}'s control is its parts 9 / 13 runs")
        value = None
    else:
        if a.axis not in axes:
            raise SystemExit(f"class {a.cls} has axes {sorted(axes)}")
        value = a.value if a.axis == "dir" else int(a.value)
        if value not in axes[a.axis]:
            raise SystemExit(f"class {a.cls}, axis {a.axis}: values {axes[a.axis]}")
    seen = None
    if a.axis == "T":
        patch_T(look, value)
    elif a.axis == "L":
        patch_L(value)
    elif a.axis == "K":
        patch_K(value)
    elif a.axis == "dir":
        seen = patch_dir(value)
    js = run_family(look, arm, device, rest, a.threads, "part14",
                    {"cls": a.cls, "axis": a.axis, "value": value, "arm14": arm_name(a.axis, value)}, a.smoke)
    T = int(js["coupling"]["T"])
    if T != (value if a.axis == "T" else T_OWN):
        raise SystemExit(f"the model ran T = {T}")
    if seen is not None and not seen["batches"]:
        raise SystemExit("no batch had its edges dropped")
    log(f"  part 14 {a.cls} {arm_name(a.axis, value)} s{js['seed']}: " + ", ".join(
        f"{n} {js['reads'][n]['mean']}" for n in js["reads"]))
    return 0


def arm_name(axis, value):
    return "base" if axis == "base" else f"{axis}{value}"


# ── (A) and (B): reads of a saved run ────────────────────────────────────────────────────────────────────────────


def load_run(a):
    import torch
    run = json.loads(Path(a.run).read_text(encoding="utf-8"))
    cls = cls_of(run)
    look = run.get("look")
    if a.smoke:
        if look not in MOD:
            raise SystemExit(f"{a.run}: not a family run")
    elif cls != a.cls or run.get("smoke") or run.get("via_dev") or "part14" in run or "part15" in run:
        raise SystemExit(f"{a.run}: not a zero-shot run of class {a.cls}")
    args = run["args"]
    device = a.device or run["device"]
    if device != run["device"] and not a.smoke:
        raise SystemExit(f"a read runs on its run's device ({run['device']})")
    dev = None if device == "cpu" else device
    block = CD.settings(dev, 1)
    S = torch.load(args["state_out"], map_location="cpu", weights_only=False)
    got = CD.state_sha(S["state"])
    if got != run["state_sha256"]:
        raise SystemExit(f"{args['state_out']}: weights {got[:16]}, the run's {run['state_sha256'][:16]}")
    seed = int(run["seed"])
    model = (C29.make_model29(run["f"], SEED + seed) if look == "chainscore29"
             else C20.make_model(run["f"], SEED + seed))
    model.load_state_dict(S["state"])
    if CD.placed(dev):
        CD._dp().model_to(model, dev)
    model.eval()
    if int(model.T) != T_OWN or int(run["coupling"]["T"]) != T_OWN:
        raise SystemExit(f"{a.run} ran T = {run['coupling']['T']}; these reads take T = {T_OWN} runs")
    own = None
    if Path(args["rows_out"]).exists():
        own = np.load(args["rows_out"])
    elif not a.smoke:
        raise SystemExit(f"{args['rows_out']}: the run's rows are needed for the identity check")
    reads = dict(x.split("=", 1) for x in args["read"])
    names = [n for n in KB_READS if n in reads and (not a.only or n in a.only)]
    if not a.smoke and names != list(KB_READS):
        raise SystemExit(f"these reads take {KB_READS}")
    return run, cls, look, dev, device, block, S, got, seed, model, own, names


def reader(a, run, look, dev, model, S, own):
    """A function n -> (d, read(lim) -> (rows, level mass)) over the run's KB reads, as the family reads them."""
    args = run["args"]
    reads = dict(x.split("=", 1) for x in args["read"])
    edges = dict(x.split("=", 1) for x in (args.get("edges") or []))
    cache = None if a.no_cache else args.get("cache")
    rb = int(args.get("read_batch") or C21.BATCH)

    def open_read(n):
        p = reads[n]
        d, lb, _rinfo, _ = C24.get(p, "plain", run["norm"], cache)
        c = None if look == "chainscore24" else C25.load_comp(edges[n], d, sha(p), model.att)
        lim = None
        if look == "chainscore29":
            if a.smoke and own is not None and n in own.files:
                lim = int(own[n].shape[0])
            if a.limit:
                lim = min(lim or a.limit, a.limit)
        elif a.limit:
            raise SystemExit("a limit reads chainscore29's runs only")

        def read():
            if look == "chainscore29":
                return C29.read29(model, d, c, lb, S["stats"], "kb", dev, batch=rb, limit=lim)
            return CD.read_dev(model, d, c, lb, S["stats"], "kb", dev, batch=rb)

        return d, read, sha(p)

    return open_read


def ident(res, n, x, own):
    if own is not None and n in own.files:
        ref = own[n][:x.shape[0]]
        res["identity"][n] = bool(np.array_equal(x, ref))
        res["identity_maxdiff"][n] = float(np.abs(x.astype(np.float64) - ref).max())


def save_rows(path, rows):
    q = Path(path)
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_name(q.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows)
    os.replace(tmp, q)


def sweep_cmd(a):
    t0 = time.time()
    run, cls, look, dev, device, block, S, got, seed, model, own, names = load_run(a)
    ts = sorted(set(a.T))
    if T_OWN not in ts:
        raise SystemExit("the sweep reads T = 2 (the identity)")
    res = {"look": "chainscore30-sweep", "cls": cls, "run": a.run, "arm": run["arm"], "f": run["f"], "seed": seed,
           "device": device, "placement": block, "state_sha256": got, "T": ts, "smoke": bool(a.smoke),
           "script_sha256": sha(__file__), "reads": {}, "identity": {}, "identity_maxdiff": {}, "seconds_read": {},
           "rows_out": a.rows_out, "inputs_sha256": {}}
    rows = {}
    open_read = reader(a, run, look, dev, model, S, own)
    for n in names:
        t1 = time.time()
        d, read, sp = open_read(n)
        res["inputs_sha256"][n] = sp
        for T in ts:
            model.T = T
            t2 = time.time()
            xr, ml = read()
            x = xr.astype(np.float32)
            rows[f"{n}__T{T}"] = x
            res["reads"][f"{n}__T{T}"] = C19.summarise(xr, d["D"][:xr.shape[0]], ml, d["level_names"])
            res["seconds_read"][f"{n}__T{T}"] = round(time.time() - t2, 1)
            if T == T_OWN:
                ident(res, n, x, own)
            log(f"  sweep {cls or run['arm']} s{seed} {n} T={T}: {res['reads'][f'{n}__T{T}']['mean']} "
                f"({time.time() - t2:.0f}s)")
        rows[f"{n}__D"] = d["D"][:rows[f"{n}__T{T_OWN}"].shape[0]]
        model.T = T_OWN
        log(f"  sweep {n}: {time.time() - t1:.0f}s; identity {res['identity'].get(n)}")
        del d
    save_rows(a.rows_out, rows)
    res["ok"] = bool(res["identity"]) and all(res["identity"].values())
    res["seconds"] = round(time.time() - t0, 1)
    return res


def balance_cmd(a):
    t0 = time.time()
    run, cls, look, dev, device, block, S, got, seed, model, own, names = load_run(a)
    src = run["reads"].get("metaqa", {}).get("mass_by_level")
    if not src:
        raise SystemExit(f"{a.run}: no metaqa level mass (the source mix)")
    pi_s = coarse_mass(list(src.values()), list(src.keys()))
    patch_balance()
    res = {"look": "chainscore30-balance", "cls": cls, "run": a.run, "arm": run["arm"], "f": run["f"], "seed": seed,
           "device": device, "placement": block, "state_sha256": got, "smoke": bool(a.smoke), "pi_s": pi_s.tolist(),
           "coarse": list(COARSE), "rule": {"R_MAX": R_MAX, "TOL": TOL, "FLOOR": FLOOR, "WCLIP": WCLIP},
           "script_sha256": sha(__file__), "reads": {}, "trace": {}, "identity": {}, "identity_maxdiff": {},
           "rows_out": a.rows_out, "inputs_sha256": {}}
    rows = {}
    open_read = reader(a, run, look, dev, model, S, own)

    def offsets(m, variant):
        pi = m if variant == "mle" else (m + pi_s) / 2.0
        return np.clip(np.log(np.maximum(pi, FLOOR)) - np.log(np.maximum(pi_s, FLOOR)), -WCLIP, WCLIP)

    for n in names:
        t1 = time.time()
        d, read, sp = open_read(n)
        res["inputs_sha256"][n] = sp
        BAL["W"] = None
        xr, ml = read()
        x0 = xr.astype(np.float32)
        m0 = coarse_mass(ml, d["level_names"])
        rows[f"{n}__R0"] = x0
        ident(res, n, x0, own)
        res["reads"][n], res["trace"][n] = {"R0": [round(float(v), 5) for v in xr.mean(0)]}, {}
        for variant, key in (("mle", "R"), ("map", "M")):
            trace = [{"r": 0, "w": None, "mass": [round(float(v), 6) for v in m0],
                      "mean": res["reads"][n]["R0"]}]
            prev, W = m0, offsets(m0, variant)
            for r in range(1, R_MAX + 1):
                BAL["W"] = W
                xr, ml = read()
                x = xr.astype(np.float32)
                m = coarse_mass(ml, d["level_names"])
                trace.append({"r": r, "w": [round(float(v), 5) for v in W], "mass": [round(float(v), 6) for v in m],
                              "mean": [round(float(v), 5) for v in xr.mean(0)]})
                if r == 1:
                    rows[f"{n}__{key}1"] = x
                conv = float(np.abs(m - prev).max()) < TOL
                if conv or r == R_MAX:
                    rows[f"{n}__{key}fin"] = x
                    res["reads"][n][variant] = {"fin": trace[-1]["mean"], "iterations": r, "converged": bool(conv)}
                    break
                prev, W = m, offsets(m, variant)
            BAL["W"] = None
            res["trace"][n][variant] = trace
            log(f"  balance {variant} {cls or run['arm']} s{seed} {n}: R0 {trace[0]['mean'][0]} -> "
                f"{trace[-1]['mean'][0]} in {len(trace) - 1} rounds, mass {trace[0]['mass']} -> {trace[-1]['mass']}")
        rows[f"{n}__D"] = d["D"][:x0.shape[0]]
        log(f"  balance {n}: {time.time() - t1:.0f}s; identity {res['identity'].get(n)}")
        del d
    save_rows(a.rows_out, rows)
    res["ok"] = bool(res["identity"]) and all(res["identity"].values())
    res["seconds"] = round(time.time() - t0, 1)
    return res


# ── grade ────────────────────────────────────────────────────────────────────────────────────────────────────────


def seed_rows(by, key):
    """R@5 on read `key`, averaged per row over the seeds given (dict seed -> npz)."""
    xs = [by[s][key][:, 0].astype(np.float64) for s in sorted(by)]
    if len({x.shape for x in xs}) != 1:
        raise SystemExit(f"{key}: the seeds' rows differ in length")
    return np.mean(xs, 0)


def delta(v, c, rng, ruled=True):
    ci = boot_ci(v - c, rng)
    return ci + ([label(ci)] if ruled else [])


def read_set(paths, look, res):
    out = {}
    for p in paths:
        js = json.loads(Path(p).read_text(encoding="utf-8"))
        if js.get("look") != look or js.get("smoke"):
            raise SystemExit(f"{p}: not a {look}")
        if not js.get("ok"):
            res["excluded"].append({look: p, "identity": js.get("identity"), "maxdiff": js.get("identity_maxdiff")})
            continue
        out.setdefault(js["cls"], {})[int(js["seed"])] = np.load(js["rows_out"])
    return out


def grade_cmd(a):
    rng = np.random.default_rng(SEED + 30)
    res = {"look": "chainscore30-grade", "script_sha256": sha(__file__), "T_read": list(T_READ), "primary": PRIMARY,
           "guard": GUARD, "excluded": [], "A": {}, "A3": {}, "B": {}, "B2": {}, "C": {}, "C3": {}, "decision": {}}
    sw = read_set(a.sweep, "chainscore30-sweep", res)
    vals = {}
    for cls, by in sorted(sw.items()):
        ent = {"seeds": sorted(by), "reads": {}}
        for n in KB_READS:
            base = seed_rows(by, f"{n}__T{T_OWN}")
            e = {"mean": {}, "delta": {}}
            for T in T_READ:
                v = seed_rows(by, f"{n}__T{T}")
                vals[(cls, n, T)] = v
                e["mean"][str(T)] = round(float(v.mean()), 5)
                if T != T_OWN:
                    e["delta"][str(T)] = delta(v, base, rng)
            ent["reads"][n] = e
        more = [T for T in T_READ if T > T_OWN and carry(ent["reads"][PRIMARY]["delta"][str(T)][3],
                                                       ent["reads"][GUARD]["delta"][str(T)][3])]
        ent["A1"] = {"MORE_ROUNDS": bool(more), "T": more}
        ent["A2"] = {str(T): {n: ent["reads"][n]["delta"][str(T)][3] for n in KB_READS} for T in (0, 1)}
        res["A"][cls] = ent
    for g, s in PAIRS_T:
        if g not in sw or s not in sw:
            res["A3"][f"{g}-{s}"] = "missing"
            continue
        res["A3"][f"{g}-{s}"] = {n: {str(T): delta(vals[(g, n, T)] - vals[(g, n, T_OWN)], vals[(s, n, T)]
                                                    - vals[(s, n, T_OWN)], rng)
                                     for T in T_READ if T != T_OWN} for n in KB_READS}
    bl = read_set(a.balance, "chainscore30-balance", res)
    bv = {}
    for cls, by in sorted(bl.items()):
        ent = {"seeds": sorted(by), "reads": {}}
        for n in KB_READS:
            r0 = seed_rows(by, f"{n}__R0")
            e = {"R0": round(float(r0.mean()), 5)}
            for variant, key in (("mle", "R"), ("map", "M")):
                r1, rf = seed_rows(by, f"{n}__{key}1"), seed_rows(by, f"{n}__{key}fin")
                bv[(cls, n, variant)] = (r0, rf)
                e[variant] = {"r1": round(float(r1.mean()), 5), "fin": round(float(rf.mean()), 5),
                              "d_fin": delta(rf, r0, rng), "d_r1": delta(r1, r0, rng)}
            ent["reads"][n] = e
        ent["BALANCER"] = carry(ent["reads"][PRIMARY]["mle"]["d_fin"][3], ent["reads"][GUARD]["mle"]["d_fin"][3])
        ent["BALANCER_MAP"] = carry(ent["reads"][PRIMARY]["map"]["d_fin"][3],
                                    ent["reads"][GUARD]["map"]["d_fin"][3])
        res["B"][cls] = ent
    for g, s in PAIRS_T:
        if all((k, PRIMARY, "mle") in bv for k in (g, s)):
            res["B2"][f"{g}-{s}"] = {v: {n: delta(bv[(g, n, v)][1] - bv[(g, n, v)][0],
                                                  bv[(s, n, v)][1] - bv[(s, n, v)][0], rng) for n in KB_READS}
                                     for v in ("mle", "map")}
    # (C)
    tr = {}
    for p in a.run:
        js = json.loads(Path(p).read_text(encoding="utf-8"))
        part = js.get("part14")
        if not part or js.get("smoke"):
            raise SystemExit(f"{p}: not a part 14 train run")
        if js["device"] != CLASSES[part["cls"]][3]:
            raise SystemExit(f"{p}: class {part['cls']} trains on {CLASSES[part['cls']][3]}")
        tr.setdefault(part["cls"], {}).setdefault(part["arm14"], {})[int(js["seed"])] = np.load(js["rows_out"])
    for p in a.control:
        js = json.loads(Path(p).read_text(encoding="utf-8"))
        cls = cls_of(js)
        cls = {"sg2": "sg2g"}.get(cls, cls)
        if cls not in GPU_CONTROL or js.get("smoke") or "part14" in js or js["device"] != "cuda":
            raise SystemExit(f"{p}: not a GPU control of {GPU_CONTROL}")
        tr.setdefault(cls, {}).setdefault("base", {})[int(js["seed"])] = np.load(js["rows_out"])
    cv = {}
    for cls, byA in sorted(tr.items()):
        if "base" not in byA:
            res["C"][cls] = {"error": "no control"}
            continue
        ctrl = byA["base"]
        ent = {"control_seeds": sorted(ctrl), "arms": {}}
        for arm, by in sorted(byA.items()):
            if arm == "base":
                continue
            seeds = sorted(set(by) & set(ctrl))
            if not seeds:
                ent["arms"][arm] = {"error": "no seed in common with the control"}
                continue
            A_, B_ = {s: by[s] for s in seeds}, {s: ctrl[s] for s in seeds}
            e = {"seeds": seeds, "reads": {}}
            for n in KB_READS + P_SECOND:
                if not all(n in A_[s].files and n in B_[s].files for s in seeds):
                    continue
                v, c = seed_rows(A_, n), seed_rows(B_, n)
                cv[(cls, arm, n)], cv[(cls, "base", n)] = v, c
                e["reads"][n] = {"arm": round(float(v.mean()), 5), "control": round(float(c.mean()), 5),
                                 "delta": delta(v, c, rng, n in KB_READS)}
            if PRIMARY in e["reads"] and GUARD in e["reads"]:
                e["CARRY"] = carry(e["reads"][PRIMARY]["delta"][3], e["reads"][GUARD]["delta"][3])
            ent["arms"][arm] = e
        choice = {}
        for ax in CLASSES[cls][4]:
            got = [(e["reads"][PRIMARY]["arm"], arm) for arm, e in ent["arms"].items()
                   if arm.startswith(ax) and e.get("CARRY")]
            choice[ax] = max(got)[1] if got else "base"
        ent["choice"] = choice
        for key, arm in (("C2_need_mp", "L0"), ("C2_need_static", "K0"), ("C4_need_back", "dirfwd"),
                         ("need_rounds", "T0")):
            if arm in ent["arms"] and "reads" in ent["arms"][arm]:
                ent[key] = {n: ent["arms"][arm]["reads"][n]["delta"][3] for n in KB_READS
                            if n in ent["arms"][arm]["reads"]}
        res["C"][cls] = ent
    for g, s in PAIRS_LS:
        e = {}
        for n in KB_READS:
            ks = [(g, "base", n), (g, "L0", n), (s, "base", n), (s, "K0", n)]
            if all(k in cv for k in ks):
                e[n] = delta(cv[ks[0]] - cv[ks[1]], cv[ks[2]] - cv[ks[3]], rng)
        res["C3"][f"{g}-{s}"] = e or "missing"
    for cls in CLASSES:
        res["decision"][cls] = {"more_rounds": res["A"].get(cls, {}).get("A1"),
                                "balancer": res["B"].get(cls, {}).get("BALANCER"),
                                "train_choice": res["C"].get(cls, {}).get("choice")}
    return res


# ── selftest ─────────────────────────────────────────────────────────────────────────────────────────────────────


def selftest():
    import torch
    for f in ("em-gr", "ena-gr"):
        o20 = C20.make_model
        patch_T("chainscore25", 4)
        assert C20.make_model(f, 7).T == 4
        C20.make_model = o20
        assert C20.make_model(f, 7).T == T_OWN
    o29 = C29.make_model29
    for k in ("rga", "rgu", "sg2"):
        patch_T("chainscore29", 0)
        assert C29.make_model29(k, 7).T == 0
        C29.make_model29 = o29
    # K = 2 reproduces sg2's own inputs; K = 0 is x alone
    o_sign, o_ns = C29.sign_inputs, C29.NSIGN
    g = torch.Generator().manual_seed(0)
    X = torch.randn(9, len(C19.NODEF), generator=g)
    E = {}
    for d in (0, 1):
        src = torch.randint(0, 9, (14,), generator=g)
        dst = torch.randint(0, 9, (14,), generator=g)
        E[d] = (src, dst, torch.arange(14), torch.bincount(dst, minlength=9).float())
    ref = o_sign(X, E)
    patch_K(2)
    assert C29.NSIGN == o_ns and torch.equal(C29.sign_inputs(X, E), ref)
    patch_K(0)
    assert C29.NSIGN == 1 and torch.equal(C29.sign_inputs(X, E), X)
    assert C29.make_model29("sg2", 3).f[0].in_features == len(C19.NODEF)
    C29.sign_inputs, C29.NSIGN = o_sign, o_ns
    # L: the GNNs are made with the layers asked
    oL = C20.NLAYER
    patch_L(0)
    assert len(C20.make_gnn(True).lin) == 0 and C29.make_model29("rgu", 3).gnn.V.shape[0] == 0
    h = C20.make_gnn(False)(X, E)
    assert h.shape == (9,)
    patch_L(3)
    assert len(C20.make_gnn(False).lin) == 3
    patch_L(oL)
    assert C20.NLAYER == C29.NLAYER == 2
    assert coarse_map(["none", "dir", "tt4", "tt16", "exact"]).tolist() == [0, 1, 2, 2, 3]
    assert np.allclose(coarse_mass([0.1, 0.1, 0.2, 0.2, 0.4], ["none", "dir", "tt4", "tt16", "exact"]),
                       [0.1, 0.1, 0.4, 0.4])
    rng = np.random.default_rng(0)
    assert label(boot_ci(np.ones(50), rng)) == "ABOVE" and label(boot_ci(-np.ones(50), rng)) == "BELOW"
    assert cls_of({"look": "chainscore25", "arm": "e-b-lb-pq", "f": "ena-gr"}) == "ena"
    assert cls_of({"look": "chainscore29", "arm": "sg2", "f": "sg2"}) == "sg2"
    assert carry("ABOVE", "AT") and carry("AT", "ABOVE") and not carry("ABOVE", "BELOW") and not carry("AT", "AT")
    print("selftest ok")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    rest = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train", help="a family's train command with one change (after --)")
    t.add_argument("--cls", required=True, choices=list(CLASSES))
    t.add_argument("--axis", required=True, choices=("base", "T", "L", "K", "dir"))
    t.add_argument("--value", default=None)
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--smoke", action="store_true")
    for name in ("sweep", "balance"):
        s = sub.add_parser(name)
        s.add_argument("--cls", required=True, choices=list(READ_CLASSES))
        s.add_argument("--run", required=True)
        if name == "sweep":
            s.add_argument("--T", type=int, action="append", default=None)
        s.add_argument("--device", default=None, help="a smoke's override")
        s.add_argument("--only", action="append", default=[], help="a smoke's reads")
        s.add_argument("--limit", type=int, default=None, help="a smoke's rows (chainscore29's runs)")
        s.add_argument("--no-cache", action="store_true", help="a smoke's: the pipeline, not the cache")
        s.add_argument("--smoke", action="store_true")
        s.add_argument("--out", required=True)
        s.add_argument("--rows-out", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--sweep", action="append", default=[])
    g.add_argument("--balance", action="append", default=[])
    g.add_argument("--run", action="append", default=[])
    g.add_argument("--control", action="append", default=[])
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if a.cmd == "train":
        return train_cmd(a, rest)
    if rest:
        raise SystemExit("only train takes arguments after --")
    if a.cmd in ("sweep", "balance"):
        if a.cmd == "sweep":
            a.T = a.T or list(T_READ)
        if (a.only or a.limit or a.no_cache or a.device) and not a.smoke:
            raise SystemExit("--only, --limit, --no-cache and --device are a smoke's")
        pre = SMOKE_PREFIX if a.smoke else OUT_PREFIX
        if not (a.out.startswith(pre) and a.rows_out.startswith(pre)):
            raise SystemExit(f"write under {pre}")
        res = sweep_cmd(a) if a.cmd == "sweep" else balance_cmd(a)
    elif a.cmd == "grade":
        res = grade_cmd(a)
    else:
        ap.print_help()
        return 2
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
