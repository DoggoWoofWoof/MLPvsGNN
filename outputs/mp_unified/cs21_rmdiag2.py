"""Diagnostic (untracked; logged, not a part and not graded): read-time distribution mapping, second round, on
chainscore21's six trained MLP-f arms (cs21m-*). Swastik (4 Oct, after the first round): 'this can still be improved,
more distributions should work'. The first round (cs21_rmdiag) had two flaws. It mapped each read population (its
whole carve, one population) onto the training build's distribution pooled over all six population regimes (every
row together down to each row alone), so a whole-carve read's population features were pulled toward those of tiny
populations. And it moved one-hot columns as if they were continuous. Here:
    matched regime  the source distribution is the training build's regime whose population size is nearest the
                    read's: argmin over regimes of |median logN - logN_read| (logN the pair feature, log of the
                    population's rows; ties to the lower regime index). Label-free: it reads only population sizes
    type-aware      only continuous columns move. d1 d2 d3 dinf nseed1 (nodes) and has_nm hop1 hop2 hop3 fwd k_none
                    k_dir k_tt k_exact (pairs) keep their values, as does a column constant over either sample (logN)
Node features carry no regime: their source is the training build's nodes, as in the first round.
Variants (each read's rows as one population, regime 0, as the arm's own reads; then the stored stats standardise, as
in training):
    src    the stored stats: must reproduce the arm's stored reads
    zp ot  the first round's two variants, re-read by its own code
    zr     each moving column affinely mapped onto the matched source's mean and sd
    qr     each moving column's marginal mapped onto the matched source's by quantiles (the first round's QMap)
    qsr    qr within strata: nodes by hop class (d1, d2, d3, dinf, none) x nseed1, pairs by level kind (k_none, k_dir,
           k_tt, k_exact) x length (hop1..3); a stratum with fewer than MIN_STRAT points in either sample uses qr, and
           a column constant within the stratum (either sample) keeps its value there
    cr     CORAL over the moving columns: the read's joint second moments onto the matched source's,
           x -> mu_s + D_s R_s^(1/2) R_t^(-1/2) D_t^(-1) (x - mu_t), R the correlation matrix shrunk toward I by LAM
    hr     the inputs as src; inside f and g, every linear layer's output (each hidden unit's pre-activation, and the
           score itself) affinely mapped onto the matched source's mean and sd, layer by layer (each layer fitted on
           the read's inputs with the earlier layers already mapped): AdaBN's re-estimated statistics at every layer
    hsr    hr within the strata of qsr (rows of a stratum short of MIN_STRAT points in either sample use hr's maps)
Samples: at most SUB points per sample (seeded), as the first round; hr and hsr fit on HSUB of each.
Fixed at 21:06 on 4 Oct before any number on a real build: the primary is b-lb (part 5's carried pair) on webqsp_sf:
zr, qr, qsr, cr, hr and hsr minus src, paired row bootstrap (BOOT 1000) on R@5, HELPS / HURTS / SAME; every arm x read is
reported the same way, with seed-gold distance D1 and D2 on the KB reads. A variant that HELPS on the primary and does
not HURT b-lb's metaqa read is carried: its trained version (every training population's inputs mapped the same way
onto one reference, and every read's) becomes the next distribution part. If none does, matched marginal and joint
maps of these inputs carry nothing to the zero-shot read, and the per-read numbers go to Swastik.
    python outputs/mp_unified/cs21_rmdiag2.py --arm outputs/mp_unified/lean/cs21m-b-lb.json \
        --out outputs/mp_unified/lean/cs21-rmdiag2-b-lb.json
    python outputs/mp_unified/cs21_rmdiag2.py --selftest
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")      # chainscore19's count

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import chainscore18 as CS  # noqa: E402
import chainscore19 as C19  # noqa: E402
import chainscore20 as C20  # noqa: E402
import chainscore21 as C21  # noqa: E402
import cs21_rmdiag as R1  # noqa: E402

SUB = R1.SUB
HSUB = 500_000
MIN_STRAT = 200
LAM = 0.05
SEED, BOOT = C19.SEED, C19.BOOT
log, sha = C21.log, C21.sha
PAIR_RAW = R1.PAIR_RAW
NODEF, PAIRF = C19.NODEF, C19.PAIRF
NODE_DISC = ("d1", "d2", "d3", "dinf", "nseed1")
PAIR_DISC = ("has_nm", "hop1", "hop2", "hop3", "fwd", "k_none", "k_dir", "k_tt", "k_exact")
NODE_STRAT = (("d1", "d2", "d3", "dinf"), ("nseed1",))
PAIR_STRAT = (("k_none", "k_dir", "k_tt", "k_exact"), ("hop1", "hop2", "hop3"))
VARIANTS = ("src", "zp", "ot", "zr", "qr", "qsr", "cr", "hr", "hsr")
NEW = ("zr", "qr", "qsr", "cr", "hr", "hsr")
JN = PAIRF.index("logN")
HS = ("HELPS", "HURTS", "SAME")


def strata(X, names, groups):
    """A stratum id per row from its own discrete columns: per one-hot group the set column (or 'none'), per single
    binary column its value."""
    sid = np.zeros(X.shape[0], np.int64)
    for g in groups:
        cols = [names.index(c) for c in g]
        if len(cols) == 1:
            v, k = (X[:, cols[0]] > 0.5).astype(np.int64), 2
        else:
            B = X[:, cols] > 0.5
            v, k = np.where(B.any(1), np.argmax(B, 1), len(cols)), len(cols) + 1
        sid = sid * k + v
    return sid


def moving(tgt, src, names, disc):
    """Columns that move: continuous, and not constant over either sample."""
    st, ss = tgt.astype(np.float64).std(0), src.astype(np.float64).std(0)
    return np.asarray([j for j, c in enumerate(names) if c not in disc and st[j] >= 1e-6 and ss[j] >= 1e-6], np.int64)


class ZMap:
    def __init__(self, tgt, src, move):
        t, s = tgt[:, move].astype(np.float64), src[:, move].astype(np.float64)
        self.move, self.mt, self.st, self.ms, self.ss = move, t.mean(0), t.std(0), s.mean(0), s.std(0)

    def __call__(self, X):
        Y = np.array(X, np.float64)
        if self.move.size:
            Y[:, self.move] = (Y[:, self.move] - self.mt) / self.st * self.ss + self.ms
        return Y.astype(np.float32)


def qmap(tgt, src, move, width):
    return R1.QMap(tgt, src, np.setdiff1d(np.arange(width), move))


class SQMap:
    """qr within strata (sfun: X -> stratum ids); a stratum short of MIN_STRAT points in either sample uses base."""

    def __init__(self, tgt, src, move, sfun, base, width):
        self.sfun, self.base, self.maps = sfun, base, {}
        it, is_ = sfun(tgt), sfun(src)
        self.counts = {}
        for k in np.unique(it):
            mt, ms = it == k, is_ == k
            self.counts[int(k)] = [int(mt.sum()), int(ms.sum())]
            if mt.sum() >= MIN_STRAT and ms.sum() >= MIN_STRAT:
                T, S = tgt[mt], src[ms]
                st, ss = T[:, move].astype(np.float64).std(0), S[:, move].astype(np.float64).std(0)
                mv = move[(st >= 1e-6) & (ss >= 1e-6)]
                self.maps[int(k)] = qmap(T, S, mv, width)

    def __call__(self, X):
        Y = self.base(X)
        if not self.maps:
            return Y
        sid = self.sfun(X)
        for k, m in self.maps.items():
            ix = np.flatnonzero(sid == k)
            if ix.size:
                Y[ix] = m(X[ix])
        return Y


def msqrt(R, inv=False):
    w, V = np.linalg.eigh(R)
    w = np.clip(w, 1e-8, None)
    return (V * (w ** (-0.5 if inv else 0.5))) @ V.T


class CMap:
    def __init__(self, tgt, src, move):
        self.move = move
        t, s = tgt[:, move].astype(np.float64), src[:, move].astype(np.float64)
        self.mt, self.dt, self.ms, self.ds = t.mean(0), t.std(0), s.mean(0), s.std(0)
        k = move.size

        def corr(x, m, sd):
            z = (x - m) / sd
            R = z.T @ z / max(z.shape[0], 1)
            return (1.0 - LAM) * R + LAM * np.eye(k)

        self.A = msqrt(corr(t, self.mt, self.dt), inv=True) @ msqrt(corr(s, self.ms, self.ds)) if k else None

    def __call__(self, X):
        Y = np.array(X, np.float64)
        if self.move.size:
            z = (Y[:, self.move] - self.mt) / self.dt
            Y[:, self.move] = (z @ self.A) * self.ds + self.ms
        return Y.astype(np.float32)


def subsample(X, seed):
    if X.shape[0] <= HSUB:
        return X
    return X[np.sort(np.random.default_rng(seed).choice(X.shape[0], HSUB, replace=False))]


def fit_hidden(seq, S, T):
    """Per linear layer of seq, the per-unit (a, b) moving the target's outputs onto the source's mean and sd, layer by
    layer (the target's later layers see its mapped outputs). A unit constant on the target keeps (1, 0)."""
    import torch
    maps, hs, ht = [], S, T
    with torch.no_grad():
        for layer in seq:
            hs, ht = layer(hs), layer(ht)
            if isinstance(layer, torch.nn.Linear):
                ms, ss = hs.mean(0), hs.std(0, unbiased=False)
                mt, sd = ht.mean(0), ht.std(0, unbiased=False)
                ok = sd >= 1e-6
                a = torch.where(ok, ss / sd.clamp_min(1e-6), torch.ones_like(sd))
                b = torch.where(ok, ms - mt * a, torch.zeros_like(mt))
                maps.append((a, b))
                ht = ht * a + b
    return maps


def make_hwrap(seq, maps, smaps, sfun):
    """seq with each linear layer's output mapped by maps (rows of a stratum in smaps by that stratum's maps)."""
    import torch

    class HWrap(torch.nn.Module):
        def forward(self, X):
            sid = sfun(X) if smaps else None
            h, li = X, 0
            for layer in seq:
                h = layer(h)
                if isinstance(layer, torch.nn.Linear):
                    a, b = maps[li]
                    if smaps:
                        A, B = a.expand(h.shape[0], -1).clone(), b.expand(h.shape[0], -1).clone()
                        for k, mk in smaps.items():
                            ix = torch.from_numpy(np.flatnonzero(sid == k))
                            if ix.numel():
                                A[ix] = mk[li][0]
                                B[ix] = mk[li][1]
                        h = h * A + B
                    else:
                        h = h * a + b
                    li += 1
            return h

    return HWrap()


def std_strata(names, groups, m, s):
    """Strata of standardised inputs (the discrete columns recovered by undoing the standardisation)."""
    def f(X):
        return strata(X.detach().cpu().numpy().astype(np.float64) * s + m, names, groups)
    return f


def hidden_for(model, st, tgt_n, src_n, tgt_p, src_p):
    """hr's and hsr's wrapped f and g."""
    import torch
    if getattr(model, "gnn", None) is not None or model.f is None or model.g is None:
        raise SystemExit("hr and hsr need chainscore19's MLP f and g")

    def std(X, m, s):
        return torch.from_numpy(((X.astype(np.float32) - m) / s).astype(np.float32))

    mn, sn, mp, sp = (np.asarray(st[k], np.float32) for k in ("mn", "sn", "mp", "sp"))
    out, info = {}, {}
    for key, seq, S, T, names, groups, m, s, k0 in (
            ("f", model.f, src_n, tgt_n, NODEF, NODE_STRAT, mn, sn, 21),
            ("g", model.g, src_p, tgt_p, PAIRF, PAIR_STRAT, mp, sp, 23)):
        S, T = subsample(S, SEED + k0), subsample(T, SEED + k0 + 1)
        S_, T_ = std(S, m, s), std(T, m, s)
        base = fit_hidden(seq, S_, T_)
        ids_s, ids_t = strata(S, names, groups), strata(T, names, groups)
        smaps, counts = {}, {}
        for k in np.unique(ids_t):
            ms_, mt_ = ids_s == k, ids_t == k
            counts[int(k)] = [int(mt_.sum()), int(ms_.sum())]
            if mt_.sum() >= MIN_STRAT and ms_.sum() >= MIN_STRAT:
                smaps[int(k)] = fit_hidden(seq, S_[torch.from_numpy(np.flatnonzero(ms_))],
                                           T_[torch.from_numpy(np.flatnonzero(mt_))])
        sf = std_strata(names, groups, m.astype(np.float64), s.astype(np.float64))
        out[key] = (make_hwrap(seq, base, {}, None), make_hwrap(seq, base, smaps, sf))
        info[key] = {"strata_mapped": len(smaps), "counts": counts,
                     "score_affine": [round(float(base[-1][0][0]), 5), round(float(base[-1][1][0]), 5)]}
    return {"hr": (out["f"][0], out["g"][0]), "hsr": (out["f"][1], out["g"][1])}, info


def wrapped_read(model, d, lb, st, rule, fw, gw):
    f0, g0 = model.f, model.g
    model.f, model.g = fw, gw
    try:
        x, _ = C21.read21(model, d, None, lb, st, rule)
    finally:
        model.f, model.g = f0, g0
    return x


def regime_logn(d, R):
    P = int(d["rowoff"][-1])
    return [float(np.median(PAIR_RAW(d, r, 0, P)[:, JN])) for r in range(R)] if P else [0.0] * R


def match_regime(logn_src, logn_read):
    return int(np.argmin(np.abs(np.asarray(logn_src) - logn_read)))


def maps_for(tgt_n, src_n, tgt_p, src_p):
    """The four new variants' node and pair maps."""
    mn_ = moving(tgt_n, src_n, NODEF, NODE_DISC)
    mp_ = moving(tgt_p, src_p, PAIRF, PAIR_DISC)
    qn, qp = qmap(tgt_n, src_n, mn_, len(NODEF)), qmap(tgt_p, src_p, mp_, len(PAIRF))
    sqn = SQMap(tgt_n, src_n, mn_, lambda X: strata(X, NODEF, NODE_STRAT), qn, len(NODEF))
    sqp = SQMap(tgt_p, src_p, mp_, lambda X: strata(X, PAIRF, PAIR_STRAT), qp, len(PAIRF))
    out = {"zr": (ZMap(tgt_n, src_n, mn_), ZMap(tgt_p, src_p, mp_)), "qr": (qn, qp), "qsr": (sqn, sqp),
           "cr": (CMap(tgt_n, src_n, mn_), CMap(tgt_p, src_p, mp_))}
    info = {"moving_node": [NODEF[j] for j in mn_], "moving_pair": [PAIRF[j] for j in mp_],
            "strata_node": {"mapped": len(sqn.maps), "counts": sqn.counts},
            "strata_pair": {"mapped": len(sqp.maps), "counts": sqp.counts}}
    return out, info


def mapped_read(model, d, lb, st, rule, nmap, pmap):
    XN0 = d["XN"]
    d["XN"] = nmap(np.asarray(XN0, np.float32))
    C19.pair_raw = lambda dd, ri, p0, p1: pmap(PAIR_RAW(dd, ri, p0, p1))
    try:
        x, _ = C21.read21(model, d, None, lb, st, rule)
    finally:
        C19.pair_raw = PAIR_RAW
        d["XN"] = XN0
    return x


def run_arm(path, res, cache, only=None):
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    js = json.loads(Path(path).read_text(encoding="utf-8"))
    if js.get("look") != "chainscore21":
        raise SystemExit(f"{path} is not a chainscore21 arm")
    arm, f, A = js["arm"], js["f"], js["args"]
    ck = torch.load(A["state_out"], weights_only=False)
    if ck["arm"] != arm or ck["f"] != f:
        raise SystemExit(f"{A['state_out']} holds {ck['arm']} ({ck['f']}), not {arm} ({f})")
    model = C20.make_model(f, SEED)
    model.load_state_dict(ck["state"])
    model.eval()
    st, spec = ck["stats"], ck["spec"]
    src_path = A["pfit"] if spec["data"] == "p" else A["fit"]
    if src_path not in cache:
        t1 = time.time()
        d0 = C19.prep(CS.load(src_path))
        R = int(d0["FR"].shape[0])
        cache[src_path] = {"stats": C19.input_stats(d0), "nodes": R1.sample_nodes(d0, SEED + 1),
                           "pairs": R1.sample_pairs(d0, list(range(R)), SEED + 2), "regimes": R,
                           "logn": regime_logn(d0, R), "by_regime": {}, "d0": d0}
        log(f"source {src_path}: {R} regimes, median logN {[round(v, 3) for v in cache[src_path]['logn']]} "
            f"({time.time() - t1:.0f}s)")
    src = cache[src_path]
    gap = max(float(np.abs(np.asarray(st[k], np.float64) - np.asarray(src["stats"][k], np.float64)).max())
              for k in ("mn", "sn", "mp", "sp"))
    if gap > 1e-6:
        raise SystemExit(f"{arm}'s stored stats are not {src_path}'s ({gap})")
    with np.load(js["rows_out"]) as z:
        stored = {k: z[k] for k in z.files}
    out = {"f": f, "source": src_path, "source_sha256": js["inputs_sha256"].get(src_path),
           "source_regime_logn": src["logn"], "reads": {}}
    reads = [(x.split("=", 1)[0], x.split("=", 1)[1], "kb") for x in A["read"]] + \
            [(x.split("=", 1)[0], x.split("=", 1)[1], "passage") for x in A["pread"]]
    for nm_, p_, rule in reads:
        if only and nm_ not in only:
            continue
        t1 = time.time()
        d = CS.load(p_)
        lb = C21.lb_offsets(d) if spec["lb"] else None
        P = int(d["rowoff"][-1])
        logn_read = float(np.median(PAIR_RAW(d, 0, 0, P)[:, JN])) if P else 0.0
        r_ = match_regime(src["logn"], logn_read)
        if r_ not in src["by_regime"]:
            src["by_regime"][r_] = R1.sample_pairs(src["d0"], [r_], SEED + 10 + r_)
        tgt_n, tgt_p = R1.sample_nodes(d, SEED + 3), R1.sample_pairs(d, [0], SEED + 4)
        maps, info = maps_for(tgt_n, src["nodes"], tgt_p, src["by_regime"][r_])
        hmaps, info["hidden"] = hidden_for(model, st, tgt_n, src["nodes"], tgt_p, src["by_regime"][r_])
        # the first round's variants, by its own code
        mn, sn, mp, sp = R1.read_moments(d)
        kn, kp = np.flatnonzero(sn < 1e-6), np.flatnonzero(sp < 1e-6)
        st_zp = {"mn": mn.astype(np.float32), "sn": np.where(sn < 1e-6, 1.0, sn).astype(np.float32),
                 "mp": mp.astype(np.float32), "sp": np.where(sp < 1e-6, 1.0, sp).astype(np.float32)}
        for k_, keep in (("n", kn), ("p", kp)):
            st_zp["m" + k_][keep] = np.asarray(st["m" + k_])[keep]
            st_zp["s" + k_][keep] = np.asarray(st["s" + k_])[keep]
        x = {}
        x["src"], _ = C21.read21(model, d, None, lb, st, rule)
        x["zp"], _ = C21.read21(model, d, None, lb, st_zp, rule)
        x["ot"] = mapped_read(model, d, lb, st, rule, R1.QMap(tgt_n, src["nodes"], kn),
                              R1.QMap(tgt_p, src["pairs"], kp))
        for v in NEW:
            x[v] = mapped_read(model, d, lb, st, rule, *maps[v]) if v in maps else \
                wrapped_read(model, d, lb, st, rule, *hmaps[v])
        rep = float(np.abs(x["src"].astype(np.float32) - stored[nm_]).max())
        N = x["src"].shape[0]
        idx = np.random.default_rng(SEED).integers(0, N, size=(BOOT, N))
        e = {"rows": N, "rule": rule, "src_reproduces": rep == 0.0, "src_max_abs_diff": rep,
             "logn_read": round(logn_read, 4), "matched_regime": r_, "matched_logn": round(src["logn"][r_], 4),
             **info, "mean": {v: [round(float(u), 5) for u in x[v].mean(0)] for v in VARIANTS}}
        for v in VARIANTS[1:]:
            e[f"{v} - src"] = R1.bt_diff(x[v][:, 0], x["src"][:, 0], idx, HS)
            if rule == "kb":
                for dd in (1, 2):
                    m = d["D"] == dd
                    if m.any():
                        ii = np.random.default_rng(SEED).integers(0, int(m.sum()), size=(BOOT, int(m.sum())))
                        e[f"{v} - src D{dd}"] = dict(R1.bt_diff(x[v][m, 0], x["src"][m, 0], ii, HS),
                                                      rows=int(m.sum()))
        out["reads"][nm_] = e
        log(f"  {arm} on {nm_}: regime {r_} (logN {logn_read:.2f} vs {src['logn'][r_]:.2f}) src "
            f"{e['mean']['src'][0]:.4f} (reproduces {e['src_reproduces']}) "
            + " ".join(f"{v} {e['mean'][v][0]:.4f} {e[f'{v} - src']['verdict']}" for v in VARIANTS[1:])
            + f" ({time.time() - t1:.0f}s)")
        del d
    res["arms"][arm] = out


def decide(res):
    pr = res["arms"].get("b-lb", {}).get("reads", {})
    if "webqsp_sf" not in pr or "metaqa" not in pr:
        return None
    prim = {v: {"webqsp_sf": pr["webqsp_sf"][f"{v} - src"]["verdict"],
                "metaqa": pr["metaqa"][f"{v} - src"]["verdict"]} for v in NEW}
    return {"primary": prim, "carried": [v for v, r in prim.items()
                                         if r["webqsp_sf"] == "HELPS" and r["metaqa"] != "HURTS"]}


def selftest():
    rng = np.random.default_rng(0)
    n = 5000

    def nodes(k, shift=0.0):
        X = rng.normal(size=(k, len(NODEF))).astype(np.float32) + shift
        h = rng.integers(0, 4, size=k)
        for j, c in enumerate(("d1", "d2", "d3", "dinf")):
            X[:, NODEF.index(c)] = (h == j)
        X[:, NODEF.index("nseed1")] = rng.random(k) < 0.1
        return X

    def pairs(k, shift=0.0):
        X = (rng.normal(size=(k, len(PAIRF))) * 2 + shift).astype(np.float32)
        lv, hp = rng.integers(0, 4, size=k), rng.integers(0, 3, size=k)
        for j, c in enumerate(("k_none", "k_dir", "k_tt", "k_exact")):
            X[:, PAIRF.index(c)] = (lv == j)
        for j, c in enumerate(("hop1", "hop2", "hop3")):
            X[:, PAIRF.index(c)] = (hp == j)
        X[:, PAIRF.index("has_nm")] = rng.random(k) < 0.5
        X[:, PAIRF.index("fwd")] = rng.choice([0.0, 0.5, 1.0], size=k)
        X[:, JN] = 3.0
        return X

    Sn, Sp = nodes(n), pairs(n)
    # 1. each map of a sample onto itself is the identity (qr/qsr at the sample's own points)
    maps, info = maps_for(Sn, Sn, Sp, Sp)
    for v, (mn_, mp_) in maps.items():
        en = float(np.abs(mn_(Sn) - Sn).max())
        ep = float(np.abs(mp_(Sp) - Sp).max())
        assert en < 1e-4 and ep < 1e-4, (v, en, ep)
    assert "logN" not in info["moving_pair"] and "d1" not in info["moving_node"] and "cq" in info["moving_node"]
    # 2. discrete and constant columns never move; moved columns take the source's moments
    Tn, Tp = nodes(n, 1.5), pairs(n, -2.0)
    Tp[:, JN] = 5.0
    maps, info = maps_for(Tn, Sn, Tp, Sp)
    dn = [NODEF.index(c) for c in NODE_DISC]
    dp = [PAIRF.index(c) for c in PAIR_DISC] + [JN]
    for v, (mn_, mp_) in maps.items():
        Yn, Yp = mn_(Tn), mp_(Tp)
        assert np.array_equal(Yn[:, dn], Tn[:, dn]) and np.array_equal(Yp[:, dp], Tp[:, dp]), v
        mv = [PAIRF.index(c) for c in info["moving_pair"]]
        gap = float(np.abs(Yp[:, mv].mean(0) - Sp[:, mv].mean(0)).max())
        assert gap < 0.1, (v, gap)
    # 3. CORAL matches the source's correlation (up to the shrinkage)
    A = rng.normal(size=(len(PAIRF), len(PAIRF))) * 0.3
    Sc = Sp.copy()
    Sc[:, :] = (Sp @ np.eye(len(PAIRF)) + (Sp @ A) * 0.5).astype(np.float32)
    for c in PAIR_DISC:
        Sc[:, PAIRF.index(c)] = Sp[:, PAIRF.index(c)]
    Sc[:, JN] = 3.0
    cm = CMap(Tp, Sc, moving(Tp, Sc, PAIRF, PAIR_DISC))
    Y = cm(Tp)[:, cm.move].astype(np.float64)
    Rs = np.corrcoef(Sc[:, cm.move].astype(np.float64), rowvar=False)
    Ry = np.corrcoef(Y, rowvar=False)
    assert float(np.abs(Ry - Rs).max()) < 0.12, float(np.abs(Ry - Rs).max())
    # 4. strata: thin strata fall back to qr; ids follow the discrete columns
    Tq = Tn.copy()
    Tq[:, NODEF.index("nseed1")] = 0
    Tq[:3, NODEF.index("nseed1")] = 1
    sq = SQMap(Tq, Sn, moving(Tq, Sn, NODEF, NODE_DISC), lambda X: strata(X, NODEF, NODE_STRAT),
               qmap(Tq, Sn, moving(Tq, Sn, NODEF, NODE_DISC), len(NODEF)), len(NODEF))
    sid = strata(Tq, NODEF, NODE_STRAT)
    assert set(np.unique(sid).tolist()) == set(sq.counts) and all(sq.counts[k][0] < MIN_STRAT for k in sq.counts
                                                                  if k not in sq.maps)
    assert len(sq.maps) == 4, sq.counts
    Y = sq(Tq)
    assert np.array_equal(Y[:3], qmap(Tq, Sn, moving(Tq, Sn, NODEF, NODE_DISC), len(NODEF))(Tq[:3]))
    # 5. the matched regime: nearest median logN, ties to the lower index
    assert match_regime([8.7, 7.3, 5.7, 4.1, 2.5, 0.0], 7.32) == 1
    assert match_regime([6.1, 4.7, 3.1, 1.6, 0.0, 0.0], -1.0) == 4
    # 6. hidden maps: identity onto the same sample; otherwise each layer's (and the score's) moments are the source's;
    #    hsr with every stratum mapped keeps each stratum's score moments; a model's own forward is unchanged
    import torch
    model = C20.make_model("em-gr", SEED)
    model.eval()
    st = {"mn": Sn.mean(0), "sn": Sn.std(0) + 1e-3, "mp": Sp.mean(0), "sp": Sp.std(0) + 1e-3}
    st = {k: np.asarray(v, np.float32) for k, v in st.items()}
    hm, _ = hidden_for(model, st, Sn, Sn, Sp, Sp)
    Xs = torch.from_numpy(((Sp - st["mp"]) / st["sp"]).astype(np.float32))
    with torch.no_grad():
        for v in ("hr", "hsr"):
            assert float((hm[v][1](Xs) - model.g(Xs)).abs().max()) < 1e-4, v
        hm, hi = hidden_for(model, st, Tn, Sn, Tp, Sp)
        Xt = torch.from_numpy(((Tp - st["mp"]) / st["sp"]).astype(np.float32))
        ys, yt = model.g(Xs).squeeze(-1), hm["hr"][1](Xt).squeeze(-1)
        assert abs(float(ys.mean() - yt.mean())) < 1e-3 and abs(float(ys.std(unbiased=False) - yt.std(unbiased=False))) < 1e-3
        assert hi["g"]["strata_mapped"] == 12, hi["g"]
        ys_, yt_ = hm["hsr"][1](Xs).squeeze(-1), hm["hsr"][1](Xt).squeeze(-1)
        ss, tt = strata(Sp, PAIRF, PAIR_STRAT), strata(Tp, PAIRF, PAIR_STRAT)
        for k in np.unique(tt)[:3]:
            a_, b_ = model.g(Xs).squeeze(-1)[torch.from_numpy(ss == k)], yt_[torch.from_numpy(tt == k)]
            assert abs(float(a_.mean() - b_.mean())) < 1e-3, k
        del ys_
        f0 = model.f
        x0 = model.f(torch.from_numpy(((Sn - st["mn"]) / st["sn"]).astype(np.float32)))
        assert model.f is f0 and x0.shape[1] == 1
    print("selftest ok")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--out")
    ap.add_argument("--only", default=None, help="smoke only: a comma list of reads")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.arm or not a.out:
        ap.error("--arm and --out")
    t0 = time.time()
    res = {"look": "cs21_rmdiag2", "script_sha256": sha(__file__), "cs21_rmdiag_sha256": sha(R1.__file__),
           "chainscore21_sha256": sha(C21.__file__), "args": dict(vars(a)), "arms": {}}
    cache = {}
    for p_ in a.arm:
        run_arm(p_, res, cache, only=a.only.split(",") if a.only else None)
    res["decision"] = decide(res)
    res["seconds"] = round(time.time() - t0, 1)
    p = Path(a.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(res, indent=1), encoding="utf-8")
    os.replace(tmp, p)
    log(f"wrote {p} ({res['seconds']}s); decision {res['decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
