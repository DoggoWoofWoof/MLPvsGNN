"""Design look (untracked; not a result and not filed): S6 part 10 of the transfer plan, part 8's training mixes with
the GNN f. Part 8 (chainscore24) varies the training mix of the MLP f (em-gr) under part 6's carried map pq: metaqa's
fit carve and its four granularity transforms (k5), with 2wiki (k6), 2wiki and hotpotqa (k8-lo-mu) or all three
passage graphs (k8), and leaves one passage graph out. Part 9 (chainscore25) puts part 4's GNN f (ena-gr: the
posterior-typed GNN, rerun on each EM round's chain posterior; no relation id or name enters it) under pq, trained on
k5's mix. This part trains that GNN f on part 8's passage mixes. Swastik (4 Oct): 'more distributions should work',
and the scorer and the chain model 'should be a part of the same model'. Part 5 (chainscore21) found one typed passage
graph in training cost the MLP f's KB zero-shot read (Q1 BELOW, -3.3); the GNN f reads the row's graph, not a relation
id, and may not pay that. Written before part 8's grade and before any part 9 run.

Arms. f is ena-gr with chainscore19's chain scorer g, chainscore20's EM rounds and the trained length offset lb, every
population mapped by pq (part 6's carry, read from its grade; anything else is refused):
    g-k5    KB: metaqa's fit carve and its four transforms (al4 sp4 mg3 rf): part 9's e-b-lb-pq, not retrained here;
            its three seeds are part 9's runs (read from their rows files by sha256); seed 0 is rerun here (R0)
    g-k6w   g-k5's KB group and 2wiki's typed passage fit carve (part 8's k6 mix)
    g-k6h   g-k5's KB group and hotpotqa's (a new mix: k6w's partner for leaving one graph out)
    g-k7    g-k5's KB group, 2wiki's and hotpotqa's (part 8's k8-lo-mu mix)
Groups as part 8 (chainscore21's draw21): the KB builds one group, each passage graph its own; the group uniform per
example. musique is never a training graph here: the GNN f's musique batches are the cost (part 9's smoke read
musique's select carve, 1,534 rows, in 1,799 s on the shared GPU: about 75 s for 64 rows; a mix with musique would
draw about 1,490 musique rows an epoch, near 6 h a run). Every arm reads musique zero-shot.
Controls, not retrained, on the same device (the host GPU) and map: part 9's e-b-lb-pq (g-k5, seeds 0..2) with its
musique preads, and part 8's MLP-f runs on the same mixes: k5 (m-k5), k6 (m-k6) and k8-lo-mu (m-k7), seeds 0..2.
Training as parts 6 and 9: Adam 1e-3, weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, the epoch
chosen by R@5 on metaqa's select carve (the KB rule), seeds SEED + 0, 1, 2 (the initial weights and the draw), the
host GPU with cs_dev's settings (deterministic algorithms, TF32 off): no number here is set beside a CPU fit's. The
training statements are part 9's train25 (cs_dev's train_dev with the select carve's batches made once), called with
the arm's groups. Every input comes from cs_cache under pq (part 8 made the passage fit carves' entries, part 9 or 8
the rest); each passage build's sidecar must name its companion (chainscore21's check_passage), and each companion
must be its build's (chainscore20's load_comp).
Reads. The KB rule on metaqa x1f, webqsp selectf and webqsp selectf + fit (webqsp is never trained on); the passage
rule on 2wiki x1 and hotpotqa x1 in the run; musique x1f by its own job from the run's saved weights (pread). Until
every musique read is in, the grade marks musique NOT_READ and a second grade fills it; the decision does not use it.
Verdicts, fixed at 00:58 on 5 Oct (this file's first write), before part 8's grade and before any part 9 or part 10
run. Per arm, each row's R@5 averaged over its three seeds; paired row bootstrap (BOOT 1000):
    W1 (primary) webqsp selectf + fit: g-k6w, g-k6h and g-k7, each minus g-k5: ABOVE / AT / BELOW
    W2 metaqa x1f: W1's differences
    W3 the curve, on every read: g-k6w - g-k5, g-k6h - g-k5, g-k7 - g-k6w, g-k7 - g-k6h, g-k7 - g-k5
    W4 leave one graph out. On 2wiki x1: g-k6h - g-k5 (hotpotqa's transfer to 2wiki over KB-only training), g-k6h -
       g-k7 (the cost of holding 2wiki out) and g-k6w - g-k5 (2wiki in training). On hotpotqa x1 the same with the
       graphs swapped. On musique x1f (held out by every arm): each mix minus g-k5. On every passage read, every arm
       and control minus s0+rrf with its share of the twin's lead over s0+rrf (HIGH at or above 0.5, LOW below 0.25,
       else MID)
    W5 the GNN f over the MLP f on the same mix, on every read: g-k5 - m-k5, g-k6w - m-k6, g-k7 - m-k7
    W6 webqsp selectf + fit by seed-gold distance D (1, 2): W1's differences
    W7 each KB read: every arm and control minus none/rd (the untyped walk)
    S  every difference above per seed (seed k's rows minus seed k's), with its sign
    R0 g-k5 seed 0 through this file (its driver, cache and train25 with one group) equals part 9's e-b-lb-pq seed 0
       row for row (|diff| < 1e-6) on every read in the run, and weight for weight (state_sha256): REPRODUCES /
       DIFFERS
Decision. A mix (g-k6w, g-k6h, g-k7) ABOVE g-k5 on webqsp selectf + fit and not BELOW on metaqa carries with the GNN
f (the largest webqsp selectf + fit mean difference if several): the next parts train the GNN f on that mix. If none
does, KB-only training stays for the GNN f, and the result goes to Swastik with parts 8 and 9. W5 says whether the
GNN f pays part 5's passage cost (Q1) or not; it decides nothing here. R0 DIFFERS: nothing is read until its cause is
found. Train-split rows throughout: a look, not a result.
    python outputs/mp_unified/chainscore26.py train --arm g-k7 --seed 0 --device cuda \
        --map-from outputs/mp_unified/lean/cs22.json --cache outputs/mp_unified/cache \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz \
        (sp4, mg3, rf) --pfit 2wiki=outputs/mp_unified/lean/cs21-2w-fit.npz --pfit hotpotqa=... \
        --select outputs/mp_unified/lean/cs19-mq-select.npz --read metaqa=... --read webqsp=... --read webqsp_sf=... \
        --pread 2wiki=... --pread hotpotqa=... --edges fit=outputs/mp_unified/lean/cs20e-mq-fit-id.npz \
        (al4, sp4, mg3, rf, select, the KB reads: cs20e-*; 2wiki-fit, hotpotqa-fit: cs21e-*-fit; 2wiki, hotpotqa:
        cs21e-*-x1) --out X.json --rows-out X.rows.npz --state-out X.pt
    python outputs/mp_unified/chainscore26.py pread --run X.json \
        --pread musique=outputs/mp_unified/lean/cs21-mu-x1f.npz --edges musique=outputs/mp_unified/lean/cs21e-mu-x1f.npz \
        --cache outputs/mp_unified/cache --device cuda \
        --out X.mu.json --rows-out X.mu.rows.npz
    python outputs/mp_unified/chainscore26.py grade --run X.json (every mix x seed, and g-k5 seed 0) \
        --ref9 outputs/mp_unified/lean/cs25-e-b-lb-pq-s0.json (s1, s2) [--ref9-mu ...cs25-e-b-lb-pq-s0.mu.json ...] \
        --mlp outputs/mp_unified/lean/cs24-k5-s0.json (k5, k6, k8-lo-mu x s0..2) [--mu-run X.mu.json ...] \
        --read ... --pread 2wiki=... --pread hotpotqa=... --pread musique=... --out outputs/mp_unified/lean/cs26.json
    python outputs/mp_unified/chainscore26.py --selftest
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
import chainpop17 as CP  # noqa: E402
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
F, NORM = "ena-gr", "pq"
SEEDS = (0, 1, 2)
TF4 = C19.TRANSFORMS[1:]
KB_READS = C21.KB_READS
P_RUN = ("2wiki", "hotpotqa")           # read in the run
P_READS = P_RUN + ("musique",)          # musique by pread
PFIT = ("2wiki", "hotpotqa")            # the passage graphs a mix may train on (never musique: see the docstring)
ARMS = {"g-k5": (), "g-k6w": ("2wiki",), "g-k6h": ("hotpotqa",), "g-k7": ("2wiki", "hotpotqa")}
MIXES = ("g-k6w", "g-k6h", "g-k7")      # the candidates; g-k5 runs here only as R0 (seed 0)
REF9 = "e-b-lb-pq"                      # part 9's arm: g-k5's seeds
MLP8 = {"m-k5": "k5", "m-k6": "k6", "m-k7": "k8-lo-mu"}       # part 8's MLP-f arms on the same mixes
W5P = (("g-k5", "m-k5"), ("g-k6w", "m-k6"), ("g-k7", "m-k7"))
CURVE = (("g-k6w", "g-k5"), ("g-k6h", "g-k5"), ("g-k7", "g-k6w"), ("g-k7", "g-k6h"), ("g-k7", "g-k5"))
HELD = {"2wiki": ("g-k6h", "g-k6w"), "hotpotqa": ("g-k6w", "g-k6h")}     # read: (the mix holding it out, the one in)
ABV = ("ABOVE", "BELOW", "AT")
log, sha = C21.log, CC.sha


def groups_of(npfit, nkb=1 + len(TF4)):
    """The KB builds one group, each passage fit carve its own (part 8's)."""
    return [list(range(nkb))] + [[nkb + j] for j in range(npfit)]


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def load_rows(p):
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


# ── train (one arm, one seed) ───────────────────────────────────────────────────────────────────────────────────


def train_cmd(a):
    t0 = time.time()
    if a.arm not in ARMS:
        raise SystemExit(f"unknown arm {a.arm}; one of {list(ARMS)}")
    if a.arm == "g-k5" and a.seed != 0:
        raise SystemExit("g-k5 runs here only as R0 (seed 0); its seeds are part 9's e-b-lb-pq runs")
    minfo = C25.carried_map(a.map_from)
    if not a.cache:
        raise SystemExit("the arms load their inputs from cs_cache (--cache)")
    dev = None if a.device == "cpu" else a.device
    spec = C21.arm_spec("b-lb", F)
    res = {"look": "chainscore26", "arm": a.arm, "seed": a.seed, "f": F, "norm": NORM, "device": a.device,
           "args": dict(vars(a)), "map_rule": minfo, "cache": a.cache, "state_out": a.state_out,
           "script_sha256": sha(__file__), "chainscore25_sha256": sha(C25.__file__),
           "cs_cache_sha256": sha(CC.__file__),
           "chainscore24_sha256": sha(C24.__file__), "chainscore22_sha256": sha(C22.__file__),
           "chainscore21_sha256": sha(C21.__file__), "chainscore20_sha256": sha(C20.__file__),
           "chainscore19_sha256": sha(C19.__file__), "chainscore18_sha256": sha(CS.__file__),
           "cs_dev_sha256": sha(CD.__file__), "inputs": {}, "inputs_sha256": {}, "passage_types": {}, "maps": {},
           "edges": {}, "reads": {}, "rows_out": a.rows_out, "timing": {}}
    aug = dict(x.split("=", 1) for x in a.aug)
    pfit = dict(x.split("=", 1) for x in a.pfit)
    if sorted(aug) != sorted(TF4):
        raise SystemExit(f"every arm takes --aug for each of {TF4}")
    if sorted(pfit) != sorted(ARMS[a.arm]):
        raise SystemExit(f"arm {a.arm} takes --pfit for exactly {ARMS[a.arm] or 'none'}")
    if not a.fit or not a.select:
        raise SystemExit("every arm takes --fit and --select")
    kb = [tuple(x.split("=", 1)) for x in a.read]
    ps = [tuple(x.split("=", 1)) for x in a.pread]
    if sorted(n for n, _ in kb) != sorted(KB_READS) or sorted(n for n, _ in ps) != sorted(P_RUN):
        raise SystemExit(f"every arm reads {KB_READS} (--read) and {P_RUN} (--pread; musique by pread)")
    train_in = [("fit", a.fit, False)] + [(t, aug[t], False) for t in TF4] + \
        [(f"{n}-fit", pfit[n], True) for n in PFIT if n in pfit]
    reads = [(n, p, False) for n, p in kb] + [(n, p, True) for n, p in ps]
    path = {n: p for n, p, _ in train_in}
    path["select"] = a.select
    path.update({n: p for n, p, _ in reads})
    edges = dict(x.split("=", 1) for x in a.edges)
    if sorted(edges) != sorted(path):
        raise SystemExit(f"a GNN f takes --edges for each of {sorted(path)}; given {sorted(edges)}")
    t1 = time.time()
    for n, p in path.items():
        res["inputs_sha256"][p] = sha(p)
        res["inputs_sha256"][edges[n]] = sha(edges[n])
    for n, p, is_p in train_in + reads:
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], res["inputs_sha256"][edges[n]])
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    res["timing"]["hash_s"] = round(time.time() - t1, 1)

    def comp(n, d):
        return C25.load_comp(edges[n], d, res["inputs_sha256"][path[n]], spec["att"])

    t1 = time.time()
    builds, comps, LB, st0 = [], [], [], None
    for n, p, is_p in train_in:
        d, lb, info, st = C25.get(p, "train", NORM, a.cache)
        tf = d["meta"]["transform"]["name"]
        want = "id" if n == "fit" or is_p else n
        if tf != want:
            raise SystemExit(f"{p} holds transform {tf}, not {want}")
        if not builds:
            st0 = st
        res["inputs"][n] = d["meta"]
        res["maps"][n] = info
        builds.append(d)
        comps.append(comp(n, d))
        LB.append(lb)
        log(f"  {a.arm} s{a.seed}: {n} ready (cache; {info})")
    groups = groups_of(len(pfit))
    Sd, Slb, sinfo, _ = C25.get(a.select, "plain", NORM, a.cache)
    Sc = comp("select", Sd)
    res["maps"]["select"] = sinfo
    res["inputs"]["select"] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p, _q), b in zip(train_in, builds)}
    res["groups"] = groups
    res["edges"] = {n: {"meta_check": c["meta"]["check"], "steps": c["meta"]["steps"]}
                    for (n, _p, _q), c in zip(train_in, comps)}
    res["timing"]["load_s"] = round(time.time() - t1, 1)
    t1 = time.time()
    model, info = C25.train25(a.arm, F, builds, comps, LB, groups, Sd, Sc, Slb, st0, a.epochs, a.per_epoch, a.batch,
                              SEED + a.seed, device=dev)
    res["timing"]["train_s"] = round(time.time() - t1, 1)
    del builds, comps, LB, Sd, Sc, Slb
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings")})
    t1 = time.time()
    rows_out = {}
    for n, p, is_p in reads:
        t2 = time.time()
        d, lb, rinfo, _ = C25.get(p, "plain", NORM, a.cache)
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
                    "spec": info["spec"], "arm": a.arm, "seed": a.seed, "f": F, "norm": NORM, "groups": groups},
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
    run = load_json(a.run)
    if run.get("look") != "chainscore26" or not run.get("state_out"):
        raise SystemExit(f"{a.run}: not a chainscore26 run with saved weights")
    if a.device != run["device"]:
        raise SystemExit(f"a read runs on its run's device ({run['device']}), not {a.device}")
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
    if (S["arm"], int(S["seed"]), S["norm"], S["f"]) != (run["arm"], int(run["seed"]), NORM, F):
        raise SystemExit(f"{run['state_out']} holds {S['arm']} seed {S['seed']} ({S['f']}, {S['norm']})")
    spec = C21.arm_spec("b-lb", F)
    model = C20.make_model(F, SEED + int(run["seed"]))
    model.load_state_dict(S["state"])
    if CD.placed(dev):
        CD._dp().model_to(model, dev)
    model.eval()
    sp, se = sha(p), sha(pe)
    side = C21.check_passage(p, sp, se)
    d, lb, rinfo, _ = C25.get(p, "plain", NORM, a.cache)
    c = C25.load_comp(pe, d, sp, spec["att"])
    t1 = time.time()
    xr, ml = CD.read_dev(model, d, c, lb, S["stats"], "passage", dev, batch=a.read_batch)
    rows = {n: xr.astype(np.float32), f"{n}__D": d["D"]}
    q = Path(a.rows_out)
    q.parent.mkdir(parents=True, exist_ok=True)
    tmp = q.with_name(q.stem + ".tmp.npz")
    np.savez_compressed(tmp, **rows)
    os.replace(tmp, q)
    res = {"look": "chainscore26-pread", "run": a.run, "arm": run["arm"], "seed": int(run["seed"]), "f": F,
           "norm": NORM, "device": a.device, "placement": block, "state_sha256": got, "read": n,
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
    runs, rows, src = {}, {}, {}        # src[(arm, seed)]: the json whose inputs_sha256 names each read's file
    r0 = None
    for p_ in a.run:
        js = load_json(p_)
        if js.get("look") != "chainscore26":
            raise SystemExit(f"{p_} is not a chainscore26 run")
        key = (js["arm"], int(js["seed"]))
        if key == ("g-k5", 0):
            if r0 is not None:
                raise SystemExit("two R0 runs")
            r0 = (p_, js, load_rows(js["rows_out"]))
            continue
        if key in runs:
            raise SystemExit(f"two runs of {key}")
        runs[key] = js
        rows[key] = load_rows(js["rows_out"])
        src[key] = {"run": js}
    want = {(m, s) for m in MIXES for s in SEEDS}
    if set(runs) != want or r0 is None:
        raise SystemExit(f"the grade takes every mix x seed and g-k5 seed 0 (R0); missing {sorted(want - set(runs))}"
                         f"{'' if r0 else ' and R0'}, extra {sorted(set(runs) - want)}")
    devs = {js["device"] for js in runs.values()} | {r0[1]["device"]}
    shas = {js["script_sha256"] for js in runs.values()} | {r0[1]["script_sha256"]}
    if len(devs) != 1 or len(shas) != 1:
        raise SystemExit(f"runs on several devices {devs} or scripts {[s[:12] for s in shas]}")
    dev = next(iter(devs))
    for p_ in a.ref9:
        js = load_json(p_)
        if js.get("look") != "chainscore25" or js.get("via_dev") or js.get("arm") != REF9:
            raise SystemExit(f"{p_} is not part 9's {REF9} run")
        key = ("g-k5", int(js["seed"]))
        if key in rows:
            raise SystemExit(f"two part 9 runs of seed {key[1]}")
        if js["device"] != dev or js.get("norm") != NORM:
            raise SystemExit(f"{p_} ran on {js['device']} under {js.get('norm')}; the runs on {dev} under {NORM}")
        runs[key] = js
        rows[key] = load_rows(js["rows_out"])
        src[key] = {"run": js}
    for p_ in a.mlp:
        js = load_json(p_)
        hit = [m for m, k in MLP8.items() if js.get("look") == "chainscore24" and js.get("arm") == k]
        if not hit:
            raise SystemExit(f"{p_} is not one of part 8's {list(MLP8.values())} runs")
        key = (hit[0], int(js["seed"]))
        if key in rows:
            raise SystemExit(f"two runs of {key}")
        if js["device"] != dev or js.get("norm") != NORM or js.get("via_dev"):
            raise SystemExit(f"{p_} ran on {js['device']} under {js.get('norm')}; the runs on {dev} under {NORM}")
        rows[key] = load_rows(js["rows_out"])
        src[key] = {"run": js}
    have = set(rows)
    full = {(m, s) for m in ["g-k5", *MIXES, *MLP8] for s in SEEDS}
    if have != full:
        raise SystemExit(f"missing {sorted(full - have)} (part 9's --ref9, part 8's --mlp)")
    mu = {}
    for flag, xs in (("mu", a.mu_run), ("ref9", a.ref9_mu)):
        for p_ in xs:
            js = load_json(p_)
            look = "chainscore26-pread" if flag == "mu" else "chainscore25-pread"
            if js.get("look") != look or js.get("read") in P_RUN:
                raise SystemExit(f"{p_} is not a {look} of {sorted(set(P_READS) - set(P_RUN))}")
            key = (js["arm"] if flag == "mu" else "g-k5", int(js["seed"]))
            if flag == "ref9" and js["arm"] != REF9:
                raise SystemExit(f"{p_} reads part 9's {js['arm']}, not {REF9}")
            if key not in runs or js["state_sha256"] != runs[key]["state_sha256"] or js["device"] != dev:
                raise SystemExit(f"{p_}: not a read of the graded run {key} on {dev}")
            if (key, js["read"]) in mu:
                raise SystemExit(f"two preads of {key} on {js['read']}")
            mu[(key, js["read"])] = js
            rows[key].update(load_rows(js["rows_out"]))
            src[key][js["read"]] = js
    GNN = [(m, s) for m in ["g-k5", *MIXES] for s in SEEDS]
    refs, Ds, rule = {}, {}, {}
    for flag, xs in (("kb", a.read), ("passage", a.pread)):
        for x in xs:
            nm_, p_ = x.split("=", 1)
            h = sha(p_)
            for k in full:
                if nm_ not in rows[k]:
                    continue
                js = src[k].get(nm_, src[k]["run"])
                if js["inputs_sha256"].get(p_) != h:
                    raise SystemExit(f"{k} read another {nm_} than {p_}")
            d = CS.load(p_)
            refs[nm_] = CS.reference_arms(d) if flag == "kb" else C21.passage_refs(d)
            Ds[nm_] = d["D"]
            rule[nm_] = flag
            del d
    if sorted(k for k in rule if rule[k] == "kb") != sorted(KB_READS) or \
            sorted(k for k in rule if rule[k] == "passage") != sorted(P_READS):
        raise SystemExit(f"the grade reads {KB_READS} (--read) and {P_READS} (--pread)")
    graded, not_read = [], []
    for nm_ in refs:
        n_in = sum(nm_ in rows[k] for k in GNN)
        if n_in == len(GNN) and all(nm_ in rows[k] for k in full):
            graded.append(nm_)
        elif n_in == 0:
            not_read.append(nm_)
        else:
            raise SystemExit(f"{nm_} is read by {n_in} of the {len(GNN)} GNN runs: give every pread or none")
    ALL = ["g-k5", *MIXES, *MLP8]

    def r5(arm, nm_, s=None):
        if s is not None:
            return rows[(arm, s)][nm_][:, 0].astype(np.float64)
        return np.mean([rows[(arm, k)][nm_][:, 0].astype(np.float64) for k in SEEDS], 0)

    V = {k: {} for k in ("W1", "W2", "W3", "W4", "W5", "W6", "W7", "S", "R0")}
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

    for nm_ in graded:
        N = refs[nm_]["twin0"].shape[0]
        for k in full:
            if rows[k][nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {rows[k][nm_].shape[0]} rows, the build {N}")
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        if nm_ == "webqsp_sf":
            V["W1"] = {f"{m} - g-k5": pair(m, "g-k5", nm_, idx) for m in MIXES}
            for dd in (1, 2):
                msk = Ds[nm_] == dd
                if msk.any():
                    ii = np.random.default_rng(SEED).integers(0, int(msk.sum()), size=(BOOT, int(msk.sum())))
                    V["W6"][f"D{dd}"] = {f"{m} - g-k5": dict(ci((r5(m, nm_) - r5("g-k5", nm_))[msk], ABV, ii),
                                                            rows=int(msk.sum())) for m in MIXES}
        if nm_ == "metaqa":
            V["W2"] = {f"{m} - g-k5": pair(m, "g-k5", nm_, idx) for m in MIXES}
        V["W3"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in CURVE}
        V["W5"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in W5P}
        if rule[nm_] == "kb":
            walk = refs[nm_]["none/rd"][:, 0]
            V["W7"][nm_] = {f"{x_} - none/rd": pair(x_, walk, nm_, idx) for x_ in ALL}
        else:
            if nm_ in HELD:
                out_, in_ = HELD[nm_]
                e = {f"{out_} - g-k5": pair(out_, "g-k5", nm_, idx), f"{out_} - g-k7": pair(out_, "g-k7", nm_, idx),
                     f"{in_} - g-k5": pair(in_, "g-k5", nm_, idx)}
            else:
                e = {f"{m} - g-k5": pair(m, "g-k5", nm_, idx) for m in MIXES}
            s0r, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["twin0"][:, 0]
            for x_ in ALL:
                X_ = r5(x_, nm_)
                sh = CP.share(X_, s0r, tw, idx)
                e[f"{x_} - s0+rrf"] = dict(ci(X_ - s0r, ABV, idx), share=sh,
                                           share_verdict="HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID")
            V["W4"][nm_] = e
    p_, js, rr = r0
    for nm_ in KB_READS + P_RUN:
        dif = float(np.abs(rows[("g-k5", 0)][nm_].astype(np.float64) - rr[nm_].astype(np.float64)).max())
        V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
    same_w = js.get("state_sha256") == runs[("g-k5", 0)].get("state_sha256")
    V["R0"]["weights"] = {"state_sha256": [runs[("g-k5", 0)].get("state_sha256"), js.get("state_sha256")],
                          "verdict": "REPRODUCES" if same_w else "DIFFERS"}
    V["R0"]["run"] = p_
    res = {"look": "chainscore26", "args": dict(vars(a)), "script_sha256": sha(__file__), "run_script": shas.pop(),
           "device": dev, "norm": NORM, "graded": graded, "not_read": not_read,
           "runs": {f"{k[0]}#s{k[1]}": {"best_epoch": js_["best_epoch"], "best_select_r5": js_["best_select_r5"],
                                        "state_sha256": js_.get("state_sha256"), "timing": js_.get("timing"),
                                        "seconds_per_batch": js_.get("seconds_per_batch"),
                                        "reads": {n: v["mean"] for n, v in js_["reads"].items()}}
                    for k, js_ in sorted(runs.items())},
           "preads": {f"{k[0]}#s{k[1]}:{n}": js_["reads"][n]["mean"] for (k, n), js_ in sorted(mu.items())},
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
        return {"carry": None, "why": "R0 incomplete", "report": True}
    if "DIFFERS" in r0:
        return {"carry": None, "why": "R0 DIFFERS: nothing is read until its cause is found", "report": True}
    ok = []
    for m in MIXES:
        w = (V.get("W1") or {}).get(f"{m} - g-k5") or {}
        q = (V.get("W2") or {}).get(f"{m} - g-k5") or {}
        if w.get("verdict") == "ABOVE" and q.get("verdict") != "BELOW":
            ok.append((w["diff"][0], m))
    if not ok:
        return {"carry": "g-k5", "mix_carries": False, "report": True,
                "why": "no mix is ABOVE g-k5 on webqsp selectf + fit without BELOW on metaqa: KB-only training stays "
                       "for the GNN f"}
    return {"carry": max(ok)[1], "mix_carries": True, "candidates": [m for _d, m in sorted(ok, reverse=True)],
            "report": False,
            "why": f"ABOVE g-k5 on webqsp selectf + fit and not BELOW on metaqa: {[m for _d, m in ok]}"}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def selftest():
    import inspect
    # 1. the arm table: part 8's mixes, the GNN f with g and lb, never musique in training
    sp = C21.arm_spec("b-lb", F)
    assert sp["gnn"] and sp["att"] and sp["g"] and sp["lb"]
    assert NORM in C22.NORMS and set(MIXES) < set(ARMS) and ARMS["g-k5"] == ()
    assert all(set(v) <= set(PFIT) for v in ARMS.values()) and "musique" not in PFIT
    for g_, m_ in W5P:
        k = MLP8[m_]
        assert C24.ARMS[k]["kb"] == TF4 and C24.ARMS[k]["lam"] == 0.0, k
        assert sorted(C24.ARMS[k]["p"]) == sorted(ARMS[g_]), (g_, k)
    for nm_, (out_, in_) in HELD.items():
        assert nm_ not in ARMS[out_] and nm_ in ARMS[in_] and nm_ in ARMS["g-k7"]
    assert all(x in ARMS and y in ARMS for x, y in CURVE)
    # 2. the groups: the KB builds one group, each passage graph its own (part 8's)
    assert groups_of(0) == [[0, 1, 2, 3, 4]] and groups_of(2) == [[0, 1, 2, 3, 4], [5], [6]]
    # 3. train25 draws by the groups given (draw21) and is part 9's sha-pinned loop
    src = inspect.getsource(C25.train25)
    assert "C21.draw21(rng, builds, groups, per_epoch)" in src and "C21.arm_spec(\"b-lb\", f)" in src
    # 4. decide
    def v(verdict, d=0.0):
        return {"verdict": verdict, "diff": [d, d - 0.01, d + 0.01]}

    def table(w, m):
        return {"R0": {k: {"verdict": "REPRODUCES"} for k in KB_READS + P_RUN + ("weights",)},
                "W1": {f"{x} - g-k5": v(*w[x]) for x in MIXES}, "W2": {f"{x} - g-k5": v(*m[x]) for x in MIXES}}

    at = {x: ("AT",) for x in MIXES}
    V = table(at, at)
    assert decide(V)["carry"] == "g-k5" and decide(V)["report"] and not decide(V)["mix_carries"]
    V = table({**at, "g-k6w": ("ABOVE", 0.02), "g-k7": ("ABOVE", 0.03)}, at)
    assert decide(V)["carry"] == "g-k7" and not decide(V)["report"]
    V = table({**at, "g-k6w": ("ABOVE", 0.02), "g-k7": ("ABOVE", 0.03)}, {**at, "g-k7": ("BELOW", -0.02)})
    assert decide(V)["carry"] == "g-k6w"
    V["R0"]["weights"] = {"verdict": "DIFFERS"}
    assert decide(V)["carry"] is None
    del V["R0"]["weights"]
    assert decide(V)["carry"] is None
    # 5. the carried map must be pq (part 9's rule)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "cs22.json"
        for carry, ok in (("pq", True), ("pz", False), ("none", False)):
            p.write_text(json.dumps({"look": "chainscore22", "decision": {"carry": carry}}), encoding="utf-8")
            try:
                C25.carried_map(p)
                assert ok, carry
            except SystemExit:
                assert not ok, carry
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
    t.add_argument("--fit", default=None)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--pfit", action="append", default=[])
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
    g.add_argument("--ref9", action="append", required=True)
    g.add_argument("--ref9-mu", action="append", default=[])
    g.add_argument("--mlp", action="append", required=True)
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
