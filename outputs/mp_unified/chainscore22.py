"""Design look (untracked; not a result and not filed): S6 part 6 of the transfer plan, distributions in training.
Part 5 (chainscore21, read 19:55 on 4 Oct) found typed passages in training cost the KB zero-shot read (Q1 BELOW,
-3.3) and the trained length offset (lb) carries. The read-time map of its first round (cs21_rmdiag, 20:05) z-scored
each read by its own moments: webqsp selectf + fit +2.0, the passage reads +2.7 to +3.9, metaqa -3.5. Swastik (4 Oct):
'this can still be improved, more distributions should work'. Read-time maps change what a frozen model sees; this
part trains the model on populations that are each mapped by their own distribution, so that what it learns is
relative to its population and a new graph's population is read the same way. cs21_rmdiag2 (queued 21:06) reads the
richer read-time maps on part 5's frozen arms beside it.

Populations. A training population is a build at one population regime (its rows refit in groups of that regime's
size; chainscore18); a read, and the select carve, is its whole carve at its one regime. Each population's continuous
inputs are mapped by its own distribution, in training and in every read, and the stored stats (the first training
build's moments, after its map) then standardise as in part 5:
    none  part 5's inputs (the control)
    pz    each continuous column z-scored by the population's own mean and sd
    pq    each continuous column's population marginal mapped onto N(0, 1): Phi^-1 of its mid-rank CDF (rank-gauss)
    pqs   pq within strata (nodes: hop class d1/d2/d3/dinf/none x nseed1; chains: level kind x length), each
          stratum's points by their own marginal; a stratum with fewer than MIN_STRAT points, or a column constant
          within a stratum, keeps pq's value there
Node columns and the regime-free chain columns (FI, FS) are mapped per build, FR's per (build, regime), the lr columns
after chainscore18's floor. Discrete columns (d1 d2 d3 dinf nseed1; has_nm hop1 hop2 hop3 fwd k_none k_dir k_tt
k_exact), logN (the population's size, a descriptor of the population itself) and a column constant over the
population keep their values. lb's offset is computed from the raw counts before the map, so every arm's prior is
part 5's lb prior; the logC feature is mapped like any continuous column. Mapped node features are stored back as
float16, as the builds store them.
Arms. f is part 5's MLP f (chainscore20's em-gr, the arm part 5 carried; part 4's GNN arms are graded separately) with
lb on every arm. Three seeds each (SEED + 0, 1, 2: the initial weights and the draw); training as part 5 (Adam 1e-3,
weight decay 1e-4, 12 epochs of 5,960 examples in batches of 64, the epoch chosen by R@5 on the select carve), every
arm on one device: the host GPU through cs_dev (deterministic algorithms, TF32 off; cs_dev's two GPU repeats of part
5's b-lb, csdev-b-lb-g1 and -g2, were identical to the weight), so no number here is set beside a CPU fit's:
    b-lb, b-lb-pz, b-lb-pq, b-lb-pqs   metaqa's fit carve and its four transforms (chainscore19's -gr builds)
    mp-lb, mp-lb-pq                     the same and 2wiki's typed fit carve, the dataset uniform per example (part 5)
    p-lb, p-lb-pq                       2wiki's typed fit carve alone (its typed select carve, the passage rule)
Reads as part 5: the KB rule on metaqa x1f, webqsp selectf and webqsp selectf + fit; the passage rule on 2wiki x1,
hotpotqa x1 and musique x1f; every read mapped by its own population under the arm's map.
Verdicts, fixed at 21:29 on 4 Oct before any arm was trained or read on a real build (a smoke ran on stand-in builds).
Per arm, each row's R@5 averaged over its three seeds; paired row bootstrap (BOOT 1000):
    N1 (primary) webqsp selectf + fit: b-lb-pz, b-lb-pq and b-lb-pqs minus b-lb: ABOVE / AT / BELOW
    N2 metaqa x1f: the same differences
    N3 each passage read: the same differences, and p-lb-pq minus p-lb and mp-lb-pq minus mp-lb; every arm minus
       s0+rrf with its share of the twin's lead over s0+rrf (HIGH at or above 0.5, LOW below 0.25, else MID)
    N4 webqsp selectf + fit and metaqa x1f: mp-lb-pq minus b-lb-pq, mp-lb-pq minus mp-lb, and mp-lb minus b-lb (part
       5's Q1 again, three seeds): does a mapped mix with passages still cost the KB read
    N5 the three KB reads: p-lb-pq minus p-lb, and p-lb-pq and p-lb minus none/rd (the untyped walk)
    N6 webqsp selectf + fit by seed-gold distance D (1, 2): N1's differences
    S  every difference above per seed (seed k's rows minus seed k's), with its sign
    R0 b-lb seed 0 reproduces part 5's b-lb on the same device row for row (|diff| < 1e-6) on every read (on the
       GPU: csdev-b-lb-g1): REPRODUCES / DIFFERS
Decision. A map whose N1 is ABOVE and whose N2 is not BELOW carries into the next parts (by N1's mean if several);
with it, N4's mp-lb-pq minus b-lb-pq not BELOW means passages in training no longer cost the KB read. If none
carries, input maps in training do not carry the zero-shot read, and the result goes to Swastik with cs21_rmdiag2's
read-time maps. R0 DIFFERS: nothing is read until its cause is found. Train-split rows throughout: a look, not a result.
    python outputs/mp_unified/chainscore22.py train --arm b-lb-pq --seed 0 \
        --fit outputs/mp_unified/lean/cs19-mq-fit-id.npz --aug al4=outputs/mp_unified/lean/cs19-mq-fit-al4.npz \
        (and sp4, mg3, rf) --select outputs/mp_unified/lean/cs19-mq-select.npz \
        (mp: --pfit outputs/mp_unified/lean/cs21-2w-fit.npz; p: --pfit and --pselect only) \
        --read metaqa=outputs/mp_unified/lean/cs19-mq-x1f.npz --read webqsp=outputs/mp_unified/lean/cs19-wq.npz \
        --read webqsp_sf=outputs/mp_unified/lean/cs19-wq-sf.npz --pread 2wiki=outputs/mp_unified/lean/cs21-2w-x1.npz \
        --pread hotpotqa=outputs/mp_unified/lean/cs21-hp-x1.npz --pread musique=outputs/mp_unified/lean/cs21-mu-x1f.npz \
        --out outputs/mp_unified/lean/cs22-b-lb-pq-s0.json --rows-out outputs/mp_unified/lean/cs22-b-lb-pq-s0.rows.npz
    python outputs/mp_unified/chainscore22.py grade --run outputs/mp_unified/lean/cs22-b-lb-s0.json (every arm x seed)\
        --cs21 outputs/mp_unified/lean/cs21m-b-lb.json --read metaqa=... (the three) --pread 2wiki=... (the three) \
        --out outputs/mp_unified/lean/cs22.json
    python outputs/mp_unified/chainscore22.py --selftest
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
from scipy.special import ndtri  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainpop17 as CP  # noqa: E402
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import cs_dev as CD  # noqa: E402

SEED, BOOT = C19.SEED, C19.BOOT
BATCH = C21.BATCH
MIN_STRAT = 200
F = "em-gr"
NORMS = ("none", "pz", "pq", "pqs")
ARMS = ("b-lb", "b-lb-pz", "b-lb-pq", "b-lb-pqs", "mp-lb", "mp-lb-pq", "p-lb", "p-lb-pq")
SEEDS = (0, 1, 2)
KB_READS, P_READS = C21.KB_READS, C21.P_READS
NODEF = C19.NODEF
NODE_DISC = ("d1", "d2", "d3", "dinf", "nseed1")
PAIR_DISC = ("has_nm", "hop1", "hop2", "hop3", "fwd", "k_none", "k_dir", "k_tt", "k_exact")
NODE_STRAT = (("d1", "d2", "d3", "dinf"), ("nseed1",))
PAIR_STRAT = (("k_none", "k_dir", "k_tt", "k_exact"), ("hop1", "hop2", "hop3"))
NODE_MOVE = [j for j, c in enumerate(NODEF) if c not in NODE_DISC]
FI_MOVE = [j for j, c in enumerate(CS.FI) if c not in PAIR_DISC]
FR_MOVE = [j for j, c in enumerate(CS.FR) if c not in PAIR_DISC and c != "logN"]
FR_LR = [CS.FR.index("lr_" + f) for f in CP.FITS]
log, sha = C21.log, C21.sha
ABV, HS = ("ABOVE", "BELOW", "AT"), ("HELPS", "HURTS", "SAME")


def arm_parts(arm):
    """(chainscore21's arm, the map): b-lb-pq -> (b-lb, pq)."""
    if arm not in ARMS:
        raise SystemExit(f"unknown arm {arm}; one of {ARMS}")
    p = arm.split("-")
    return "-".join(p[:2]), (p[2] if len(p) == 3 else "none")


# ── the maps ────────────────────────────────────────────────────────────────────────────────────────────────────


def strata(X, names, groups):
    """A stratum id per row from its own discrete columns: per one-hot group the set column (or none), per single
    binary column its value."""
    sid = np.zeros(X.shape[0], np.int64)
    for g in groups:
        cols = [names.index(c) for c in g]
        if len(cols) == 1:
            v, k = (np.asarray(X[:, cols[0]], np.float64) > 0.5).astype(np.int64), 2
        else:
            B = np.asarray(X[:, cols], np.float64) > 0.5
            v, k = np.where(B.any(1), np.argmax(B, 1), len(cols)), len(cols) + 1
        sid = sid * k + v
    return sid


def rank_gauss(x):
    """Phi^-1 of the mid-rank CDF: ties share a value; strictly inside (0, 1)."""
    s = np.sort(x)
    u = (np.searchsorted(s, x, "left") + np.searchsorted(s, x, "right")) / (2.0 * x.size)
    return ndtri(u)


def u_f16(x16, sid=None):
    """rank_gauss's mid-rank CDF for a float16 column in O(n): each value's order-preserving 16-bit key (-0 as +0)
    counted per stratum (sid; one stratum if None). Returns (u per point, per-stratum counts)."""
    b = np.ascontiguousarray(x16).view(np.uint16).astype(np.int64)
    b[b == 0x8000] = 0
    k = np.where(b >= 0x8000, 0xFFFF - b, b + 0x8000)
    S = 1 if sid is None else int(sid.max()) + 1
    kk = k if sid is None else sid * 65536 + k
    C = np.bincount(kk, minlength=S * 65536).reshape(S, 65536)
    ns = C.sum(1)
    U = (2 * (np.cumsum(C, 1) - C) + C) / (2.0 * np.maximum(ns, 1)[:, None])
    return U.ravel()[kk], ns


def strat_ok(xd, sid):
    """Per stratum: at least MIN_STRAT points and a standard deviation of at least 1e-6."""
    n = np.bincount(sid).astype(np.float64)
    m = np.bincount(sid, weights=xd) / np.maximum(n, 1)
    v = np.bincount(sid, weights=(xd - m[sid]) ** 2) / np.maximum(n, 1)
    return (n >= MIN_STRAT) & (np.sqrt(v) >= 1e-6)


def map_col(x, norm, sid=None):
    """One population's column under norm, as float64. A column constant over the population is returned unchanged;
    under pqs a stratum below MIN_STRAT points, or constant within, keeps pq's value."""
    xd = x.astype(np.float64)
    if xd.size == 0 or float(xd.std()) < 1e-6:
        return xd
    if norm == "pz":
        return (xd - xd.mean()) / xd.std()
    f16 = x.dtype == np.float16
    y = ndtri(u_f16(x)[0]) if f16 else rank_gauss(xd)
    if norm == "pqs" and sid is not None:
        ok = strat_ok(xd, sid)
        if f16:
            m = ok[sid]
            y[m] = ndtri(u_f16(x, sid)[0][m])
        else:
            for k in np.flatnonzero(ok):
                m = sid == k
                y[m] = rank_gauss(xd[m])
    return y


def normalise(d, norm):
    """The build's populations mapped in place (node columns and FI, FS per build; FR per regime). Call after
    lb_offsets (it reads FI's raw logC)."""
    if norm not in NORMS:
        raise SystemExit(f"unknown map {norm}")
    if norm == "none":
        return {"norm": "none"}
    t0 = time.time()
    for k in ("XN", "FI", "FS", "FR"):
        if not d[k].flags.writeable:
            d[k] = np.array(d[k], copy=True)
    XN, FI, FS, FR = d["XN"], d["FI"], d["FS"], d["FR"]
    sid_n = strata(XN, NODEF, NODE_STRAT) if norm == "pqs" else None
    for j in NODE_MOVE:     # each column from its own values; the discrete (strata) columns are never written
        XN[:, j] = map_col(XN[:, j], norm, sid_n)
    sid_p = strata(FI, CS.FI, PAIR_STRAT) if norm == "pqs" else None
    for j in FI_MOVE:
        FI[:, j] = map_col(FI[:, j], norm, sid_p)
    FS[:] = map_col(FS, norm, sid_p)
    for r in range(FR.shape[0]):
        B = FR[r]
        B[:, FR_LR] = np.maximum(B[:, FR_LR], CS.LR_FLOOR)
        for j in FR_MOVE:
            B[:, j] = map_col(B[:, j], norm, sid_p)
    info = {"norm": norm, "nodes": int(XN.shape[0]), "pairs": int(FI.shape[0]), "regimes": int(FR.shape[0]),
            "seconds": round(time.time() - t0, 1)}
    if norm == "pqs":
        info["strata_nodes"] = {int(k): int(v) for k, v in zip(*np.unique(sid_n, return_counts=True))}
        info["strata_pairs"] = {int(k): int(v) for k, v in zip(*np.unique(sid_p, return_counts=True))}
    return info


# ── train (one arm, one seed) ───────────────────────────────────────────────────────────────────────────────────


def train_cmd(a):
    t0 = time.time()
    base, norm = arm_parts(a.arm)
    spec = C21.arm_spec(base, F)
    if not spec["lb"] or spec["gnn"]:
        raise SystemExit("every arm here is an lb arm with the MLP f")
    data = spec["data"]
    dev = None if a.device == "cpu" else a.device
    res = {"look": "chainscore22", "arm": a.arm, "base": base, "norm": norm, "seed": a.seed, "f": F,
           "device": a.device, "args": dict(vars(a)), "script_sha256": sha(__file__),
           "chainscore21_sha256": sha(C21.__file__), "chainscore20_sha256": sha(C20.__file__),
           "chainscore19_sha256": sha(C19.__file__), "chainscore18_sha256": sha(CS.__file__),
           "cs_dev_sha256": sha(CD.__file__), "inputs": {}, "inputs_sha256": {}, "passage_types": {}, "maps": {},
           "reads": {}, "rows_out": a.rows_out}
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
    for n, p, is_p in train_in + [sel] + reads:
        res["inputs_sha256"][p] = sha(p)
        if is_p:
            side = C21.check_passage(p, res["inputs_sha256"][p], None)
            res["passage_types"][n] = {"script_sha256": side["script_sha256"], **side["types"]}
    builds, LB = [], []
    for n, p, _is_p in train_in:
        d = C19.prep(CS.load(p))
        tf = d["meta"]["transform"]["name"]
        if tf != ("id" if n in ("fit", "pfit") else n):
            raise SystemExit(f"{p} holds transform {tf}, not {'id' if n in ('fit', 'pfit') else n}")
        res["inputs"][n] = d["meta"]
        LB.append(C21.lb_offsets(d))
        res["maps"][n] = normalise(d, norm)
        builds.append(d)
        log(f"  {a.arm} s{a.seed}: {n} loaded and mapped ({res['maps'][n]})")
    groups = {"b": [list(range(5))], "mp": [list(range(5)), [5]], "p": [[0]]}[data]
    Sd = CS.load(sel[1])
    Slb = C21.lb_offsets(Sd)
    res["maps"][sel[0]] = normalise(Sd, norm)
    res["inputs"][sel[0]] = Sd["meta"]
    res["train_rows"] = {n: int(b["train_rows"].size) for (n, _p, _q), b in zip(train_in, builds)}
    res["groups"] = groups
    model, info = CD.train_dev(base, F, builds, [None] * len(builds), LB, groups, Sd, None, Slb,
                               "passage" if data == "p" else "kb", a.epochs, a.per_epoch, a.batch, SEED + a.seed,
                               device=dev)
    del builds, LB, Sd, Slb
    res.update({k: info[k] for k in ("history", "best_epoch", "best_select_r5", "coupling", "spec", "placement",
                                     "state_sha256", "seconds_per_batch", "deterministic_warnings")})
    rows_out = {}
    for n, p, is_p in reads:
        t1 = time.time()
        d = CS.load(p)
        lb = C21.lb_offsets(d)
        res["maps"][n] = normalise(d, norm)
        rule = "passage" if is_p else "kb"
        xr, ml = CD.read_dev(model, d, None, lb, info["stats"], rule, dev, batch=a.read_batch)
        rows_out[n] = xr.astype(np.float32)
        rows_out[f"{n}__D"] = d["D"]
        res["reads"][n] = dict(C19.summarise(xr, d["D"], ml, d["level_names"]), rule=rule)
        res["inputs"][n] = d["meta"]
        log(f"  {a.arm} s{a.seed} on {n} ({rule} rule): {res['reads'][n]['mean']} ({time.time() - t1:.0f}s)")
        del d
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
    return res


# ── grade ───────────────────────────────────────────────────────────────────────────────────────────────────────


def grade_cmd(a):
    t0 = time.time()
    runs, rows = {}, {}
    for p_ in a.run:
        js = json.loads(Path(p_).read_text(encoding="utf-8"))
        if js.get("look") != "chainscore22":
            raise SystemExit(f"{p_} is not a chainscore22 run")
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

    V = {k: {} for k in ("N1", "N2", "N3", "N4", "N5", "N6", "S", "R0")}

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

    N1P = [(f"b-lb-{m}", "b-lb") for m in ("pz", "pq", "pqs")]
    for nm_ in refs:
        N = refs[nm_]["twin0"].shape[0]
        for k, r in rows.items():
            if r[nm_].shape[0] != N:
                raise SystemExit(f"{k}'s {nm_} read has {r[nm_].shape[0]} rows, the build {N}")
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        if nm_ == "webqsp_sf":
            V["N1"] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in N1P}
            for dd in (1, 2):
                m = Ds[nm_] == dd
                if m.any():
                    ii = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                    V["N6"][f"D{dd}"] = {f"{x_} - {y_}": dict(ci((r5(x_, nm_) - r5(y_, nm_))[m], ABV, ii),
                                                              rows=int(m.sum())) for x_, y_ in N1P}
        if nm_ == "metaqa":
            V["N2"] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in N1P}
        if nm_ in ("webqsp_sf", "metaqa"):
            V["N4"][nm_] = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in
                            (("mp-lb-pq", "b-lb-pq"), ("mp-lb-pq", "mp-lb"), ("mp-lb", "b-lb"))}
        if rule[nm_] == "kb":
            walk = refs[nm_]["none/rd"][:, 0]
            V["N5"][nm_] = {"p-lb-pq - p-lb": pair("p-lb-pq", "p-lb", nm_, idx),
                            "p-lb-pq - none/rd": pair("p-lb-pq", walk, nm_, idx),
                            "p-lb - none/rd": pair("p-lb", walk, nm_, idx)}
        else:
            s0r, tw = refs[nm_]["s0+rrf"][:, 0], refs[nm_]["twin0"][:, 0]
            e = {f"{x_} - {y_}": pair(x_, y_, nm_, idx) for x_, y_ in
                 N1P + [("p-lb-pq", "p-lb"), ("mp-lb-pq", "mp-lb")]}
            for x_ in ARMS:
                X_ = r5(x_, nm_)
                sh = CP.share(X_, s0r, tw, idx)
                e[f"{x_} - s0+rrf"] = dict(ci(X_ - s0r, ABV, idx), share=sh,
                                           share_verdict="HIGH" if sh[1] >= 0.5 else "LOW" if sh[2] < 0.25 else "MID")
            V["N3"][nm_] = e
    dev = next(iter(devs))
    if a.cs21:
        js = json.loads(Path(a.cs21).read_text(encoding="utf-8"))
        if js.get("look") not in ("chainscore21", "chainscore21 via cs_dev") or js.get("arm") != "b-lb" or \
                js.get("f") != F:
            raise SystemExit(f"{a.cs21}: R0 reads part 5's b-lb ({F})")
        if js.get("device", "cpu") != dev:
            raise SystemExit(f"R0 reads part 5's b-lb on the runs' device ({dev}), not {js.get('device', 'cpu')}")
        with np.load(js["rows_out"]) as z:
            c21 = {k: z[k] for k in z.files}
        for nm_ in refs:
            dif = float(np.abs(rows[("b-lb", 0)][nm_].astype(np.float64) - c21[nm_].astype(np.float64)).max())
            V["R0"][nm_] = {"max_abs_diff": dif, "verdict": "REPRODUCES" if dif < 1e-6 else "DIFFERS"}
    res = {"look": "chainscore22", "args": dict(vars(a)), "script_sha256": sha(__file__), "run_script": shas.pop(),
           "device": devs.pop(),
           "runs": {f"{k[0]}#s{k[1]}": {"best_epoch": js["best_epoch"], "best_select_r5": js["best_select_r5"],
                                        "state_sha256": js.get("state_sha256"),
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
        return {"carry": None, "why": "R0 incomplete (no --cs21)", "report": False}
    if "DIFFERS" in r0:
        return {"carry": None, "why": "R0 DIFFERS: nothing is read until its cause is found", "report": True}
    ok = []
    for m in ("pz", "pq", "pqs"):
        k = f"b-lb-{m} - b-lb"
        n1, n2 = V["N1"].get(k) or {}, V["N2"].get(k) or {}
        if n1.get("verdict") == "ABOVE" and n2.get("verdict") != "BELOW":
            ok.append((n1["diff"][0], m))
    if not ok:
        return {"carry": "none", "why": "no map is ABOVE on N1 without BELOW on N2", "report": True}
    best = max(ok)[1]
    n4 = ((V["N4"].get("webqsp_sf") or {}).get("mp-lb-pq - b-lb-pq") or {}).get("verdict")
    return {"carry": best, "candidates": [m for _d, m in sorted(ok, reverse=True)],
            "passages_no_longer_cost": None if best != "pq" else n4 != "BELOW", "report": False,
            "why": f"N1 ABOVE and N2 not BELOW for {[m for _d, m in ok]}"}


# ── selftest ────────────────────────────────────────────────────────────────────────────────────────────────────


def toy_build(rng, rows=40, nodes=60, chains=30, regimes=3):
    n = np.full(rows, nodes)
    XN = rng.normal(size=(rows * nodes, len(NODEF))).astype(np.float16)
    h = rng.integers(0, 5, size=rows * nodes)
    for j, c in enumerate(("d1", "d2", "d3", "dinf")):
        XN[:, NODEF.index(c)] = (h == j)
    XN[:, NODEF.index("nseed1")] = rng.random(rows * nodes) < 0.05
    P = rows * chains
    FI = (rng.normal(size=(P, len(CS.FI))) * 3 + 1).astype(np.float32)
    lv, hp = rng.integers(0, 4, size=P), rng.integers(0, 3, size=P)
    for j, c in enumerate(("k_none", "k_dir", "k_tt", "k_exact")):
        FI[:, CS.FI.index(c)] = (lv == j)
    for j, c in enumerate(("hop1", "hop2", "hop3")):
        FI[:, CS.FI.index(c)] = (hp == j)
    FI[:, CS.FI.index("has_nm")] = rng.random(P) < 0.5
    FI[:, CS.FI.index("fwd")] = rng.choice([0.0, 0.5, 1.0], size=P)
    FR = (rng.normal(size=(regimes, P, len(CS.FR))) * np.arange(1, regimes + 1)[:, None, None] - 2).astype(np.float32)
    FR[:, :, FR_LR[0]] = -60.0
    FR[:, :, CS.FR.index("logN")] = np.log([rows, rows / 4, 1])[:regimes, None]
    return {"n": n, "XN": XN, "FI": FI, "FR": FR, "FS": rng.normal(size=P).astype(np.float32),
            "rowoff": np.arange(0, P + 1, chains)}


def selftest():
    rng = np.random.default_rng(0)
    # 1. rank-gauss: ties share a value, order kept, about N(0, 1)
    x = np.r_[rng.normal(size=5000), np.zeros(300)]
    y = rank_gauss(x)
    assert np.unique(y[x == 0]).size == 1 and np.all(np.diff(y[np.argsort(x, kind="stable")]) >= 0)
    assert abs(y.mean()) < 0.02 and abs(y.std() - 1) < 0.05
    # 1b. the float16 path is the reference path (rank_gauss and a per-stratum loop) on ties, -0 and +0, small strata
    x16 = np.r_[rng.normal(size=20000) * 3, np.zeros(500), -np.zeros(400), np.full(300, 2.5)].astype(np.float16)
    x16 = x16[rng.permutation(x16.size)]
    sid = rng.choice(7, size=x16.size, p=[0.4, 0.3, 0.2, 0.08, 0.015, 0.004, 0.001])
    sid[x16 == 2.5] = 6     # a stratum constant within
    xd = x16.astype(np.float64)
    assert np.array_equal(ndtri(u_f16(x16)[0]), rank_gauss(xd))
    ref = rank_gauss(xd)
    for k in np.unique(sid):
        m = sid == k
        if int(m.sum()) >= MIN_STRAT and float(xd[m].std()) >= 1e-6:
            ref[m] = rank_gauss(xd[m])
    got = map_col(x16, "pqs", sid)
    assert np.allclose(got, ref, rtol=0, atol=1e-12) and np.array_equal(map_col(x16, "pq"), rank_gauss(xd))
    assert np.allclose(map_col(xd.astype(np.float32), "pqs", sid), ref, rtol=0, atol=1e-12)
    # 2. none leaves the build as it is; pz, pq, pqs map each population's continuous columns, keep the rest
    d0 = toy_build(rng)
    d = {k: np.array(v, copy=True) for k, v in d0.items()}
    assert normalise(d, "none") == {"norm": "none"} and all(np.array_equal(d[k], d0[k]) for k in d0)
    keep_n = [NODEF.index(c) for c in NODE_DISC]
    keep_i = [CS.FI.index(c) for c in PAIR_DISC]
    for norm in ("pz", "pq", "pqs"):
        d = {k: np.array(v, copy=True) for k, v in d0.items()}
        normalise(d, norm)
        assert np.array_equal(d["XN"][:, keep_n], d0["XN"][:, keep_n]) and np.array_equal(d["FI"][:, keep_i],
                                                                                       d0["FI"][:, keep_i])
        jl = CS.FR.index("logN")
        assert np.array_equal(d["FR"][:, :, jl], d0["FR"][:, :, jl])
        assert np.all(d["FR"][:, :, FR_LR[0]] == CS.LR_FLOOR)   # floored first, then constant: it stays
        for r in range(d["FR"].shape[0]):
            B = d["FR"][r][:, FR_MOVE].astype(np.float64)
            const = d0["FR"][r][:, FR_MOVE].std(0) < 1e-6
            if norm != "pqs":
                assert np.all(np.abs(B.mean(0)[~const]) < 0.05) and np.all(np.abs(B.std(0)[~const] - 1) < 0.1), norm
        Xn = d["XN"][:, NODE_MOVE].astype(np.float64)
        assert np.all(np.abs(Xn.mean(0)) < 0.05), norm
        if norm == "pqs":   # within each large stratum, about N(0, 1)
            sid = strata(d0["FI"], CS.FI, PAIR_STRAT)
            for k in np.unique(sid):
                m = sid == k
                if m.sum() >= MIN_STRAT:
                    v = d["FI"][m][:, FI_MOVE[0]].astype(np.float64)
                    assert abs(v.mean()) < 0.1 and abs(v.std() - 1) < 0.15, (k, v.mean(), v.std())
    # 3. lb's offset is read from the raw counts: normalise runs after lb_offsets in every path (train and read)
    import inspect
    src = inspect.getsource(train_cmd)
    for a_, b_ in (("LB.append(C21.lb_offsets(d))", "res[\"maps\"][n] = normalise(d, norm)\n        builds"),
                   ("Slb = C21.lb_offsets(Sd)", "normalise(Sd, norm)"), ("lb = C21.lb_offsets(d)",
                                                                         "res[\"maps\"][n] = normalise(d, norm)\n        rule")):
        assert src.index(a_) < src.index(b_), (a_, b_)
    # 4. arms
    assert arm_parts("b-lb") == ("b-lb", "none") and arm_parts("mp-lb-pq") == ("mp-lb", "pq")
    for arm in ARMS:
        base, norm = arm_parts(arm)
        assert C21.arm_spec(base, F)["lb"] and norm in NORMS
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--seed", type=int, required=True, choices=SEEDS)
    t.add_argument("--fit", default=None)
    t.add_argument("--aug", action="append", default=[])
    t.add_argument("--select", default=None)
    t.add_argument("--pfit", default=None)
    t.add_argument("--pselect", default=None)
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
    g = sub.add_parser("grade")
    g.add_argument("--run", action="append", required=True)
    g.add_argument("--cs21", default=None, help="part 5's b-lb arm JSON (R0)")
    g.add_argument("--read", action="append", required=True)
    g.add_argument("--pread", action="append", required=True)
    g.add_argument("--out", required=True)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
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
