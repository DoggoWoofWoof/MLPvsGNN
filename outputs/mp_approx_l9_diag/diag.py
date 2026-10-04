"""Descriptive look at level 9's filed units and sidecar, run on the host. Not a result and not filed. It informs the
design of a later level, whose population leaves these 3,000 rows out and whose file discloses the look.

For the primary arm and four others it splits the distance to the NB-oracle into parts. The oracle adds 1e6 to the true
chain's reach set, taken from the bucket whose reach set has the larger Jaccard with the golds. The arm instead mixes
the reach sets of every type under p(type | q) and gates the twin's z-score softly with kappa and eta. Each variant
below swaps one of those parts. m_q is recovered from the stored scores by inverting s = z + kappa log(n_q m_q + eta)."""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path.cwd()
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l9 as L9  # noqa: E402

L8, L0 = L9.L8, L9.L0
OUT = ROOT / "outputs" / "mp_approx_l9_diag"
ARMS = ("NB-hyb", "NB-id", "NB-text", "STD-hyb", "STD-text")
BIG = 1e6
RET = L8.RETRIEVAL
DIRN = ("fwd", "bwd", "both")
VARIANTS = ("learned", "learned_hard", "argmax_hard", "argmax_seq_union_hard", "true_soft", "true_b0_hard",
            "true_union_hard", "oracle")


def tok_name(t: int) -> str:
    return f"{L8.REL_ORDER[t // 3]}:{DIRN[t % 3]}"


def seq_name(seq) -> str:
    return "null" if (not seq or seq[0] == "null") else " > ".join(tok_name(int(t)) for t in seq)


def metrics3(score: np.ndarray, g: np.ndarray, gt: int) -> list:
    r = L9.rank_metrics(score, g, gt)
    return [r[m] for m in RET]


def lexi(key: np.ndarray, z: np.ndarray) -> np.ndarray:
    """A score whose order ranks by key first (descending), then by z."""
    order = np.lexsort((-z, -key))
    s = np.empty(z.size)
    s[order] = -np.arange(z.size, dtype=np.float64)
    return s


def with_bonus(z: np.ndarray, rset: np.ndarray) -> np.ndarray:
    b = np.zeros(z.size)
    b[rset] = BIG
    return z + b


def summarise(x: np.ndarray, sel=None) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if sel is not None:
        x = x[sel]
    x = x[np.isfinite(x)]
    if not x.size:
        return {"n": 0}
    return {"n": int(x.size), "mean": float(x.mean()), "q10": float(np.quantile(x, 0.1)), "q50": float(np.median(x)),
            "q90": float(np.quantile(x, 0.9))}


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    vs = L9.views(L9.DATA)
    base = vs["std"]
    n = base.n_q
    qm = base.q_metrics
    ri = [L9.METRIC_NAMES.index(m) for m in RET]
    T = qm[:, [L9.FUNCS.index(f"twin{k}") for k in L9.SEEDS]][:, :, ri]
    G = qm[:, [L9.FUNCS.index(f"gnn{k}") for k in L9.SEEDS]][:, :, ri]
    W = L0.boot_weights(n)
    _den_out, dens, readable = L8.denominators(T, G, W)
    den_sum = {m: float(np.asarray(dens[m]).sum()) for m in RET}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    hop = np.asarray(base.q_hop)
    sets = {}
    for fam, v in vs.items():
        rows = []
        for q in range(n):
            r0, r1 = L8.chain_reach(v, q, chains[qt_names[q]])
            rstar, b = L8.r_star(v, q, chains[qt_names[q]])
            rows.append((r0, r1, rstar, b))
        sets[fam] = rows
    print(f"[{time.time() - t0:.0f}s] views and true-chain reach sets ready, {n} queries", flush=True)
    out = {"queries": n, "readable": readable, "den_sum": den_sum, "arms": {}}
    for arm in ARMS:
        fam = L9.ARM_SPEC[arm][0]
        v, rows = vs[fam], sets[fam]
        V = {name: np.full((n, len(L9.SEEDS), len(RET)), np.nan) for name in VARIANTS}
        pq = {key: np.full((n, len(L9.SEEDS)), np.nan) for key in (
            "mass_rstar", "mass_union", "sum_m_err", "nm_min", "argmax_ok", "argmax_bucket", "argmax_size",
            "argmax_has_gold", "argmax_null", "rstar_size", "rstar_has_gold", "oracle_bucket")}
        conf, stored_diff, units = Counter(), 0.0, {}
        for k in L9.SEEDS:
            for fold in range(L9.FOLDS):
                npz, js = L9.unit_paths(arm, k, fold)
                flog = json.loads(js.read_text(encoding="utf-8"))
                kappa, eta = float(flog["kappa"]), float(flog["eta"])
                units[f"k{k}_f{fold}"] = {"kappa": kappa, "eta": eta, "kept_round": flog["kept_round"]}
                with np.load(npz) as zf:
                    qs, score, ptr, amax, mets = zf["q"], zf["score"], zf["score_ptr"], zf["argmax"], zf["metrics"]
                for i, q in enumerate(qs):
                    q = int(q)
                    s = np.asarray(score[ptr[i]:ptr[i + 1]], dtype=np.float64)
                    z = v.z(q, k)
                    nq = z.size
                    g, gt = v.gold_local(q), int(v.q_gold_total[q])
                    nm = np.exp((s - z) / kappa) - eta
                    m = nm / nq
                    r0, r1, rstar, b = rows[q]
                    union = np.union1d(r0, r1)
                    pq["sum_m_err"][q, k] = abs(float(m.sum()) - 1.0)
                    pq["nm_min"][q, k] = float(nm.min())
                    pq["mass_rstar"][q, k] = float(m[rstar].sum())
                    pq["mass_union"][q, k] = float(m[union].sum())
                    pq["rstar_size"][q, k] = rstar.size
                    pq["rstar_has_gold"][q, k] = float(np.intersect1d(rstar, g).size > 0)
                    pq["oracle_bucket"][q, k] = b
                    V["learned"][q, k] = metrics3(s, g, gt)
                    stored_diff = max(stored_diff, float(np.abs(np.asarray(V["learned"][q, k]) - mets[i][ri]).max()))
                    V["learned_hard"][q, k] = metrics3(lexi(np.round(nm, 6), z), g, gt)
                    a = int(amax[i])
                    if a >= 0:
                        ra = v.reach(q, a)
                        bucket = a // L8.TB ** L8.MAX_L
                        seq = L8.token_sequence(a)
                        other = v.reach(q, L8.type_code(1 - bucket, seq))
                        pq["argmax_bucket"][q, k] = bucket
                        pq["argmax_size"][q, k] = ra.size
                        pq["argmax_has_gold"][q, k] = float(np.intersect1d(ra, g).size > 0)
                        pq["argmax_ok"][q, k] = float(tuple(seq) == truth[q])
                        pq["argmax_null"][q, k] = 0.0
                        if tuple(seq) != truth[q] and k == 0:
                            conf[(int(hop[q]), qt_names[q], seq_name(seq))] += 1
                    else:
                        ra = other = np.zeros(0, dtype=np.int64)
                        pq["argmax_ok"][q, k] = 0.0
                        pq["argmax_null"][q, k] = 1.0
                        if k == 0:
                            conf[(int(hop[q]), qt_names[q], "null or uniform")] += 1
                    V["argmax_hard"][q, k] = metrics3(with_bonus(z, ra), g, gt)
                    V["argmax_seq_union_hard"][q, k] = metrics3(with_bonus(z, np.union1d(ra, other)), g, gt)
                    ind = np.zeros(nq)
                    if rstar.size:
                        ind[rstar] = 1.0 / rstar.size
                    V["true_soft"][q, k] = metrics3(z + kappa * np.log(nq * ind + eta), g, gt)
                    V["true_b0_hard"][q, k] = metrics3(with_bonus(z, r0), g, gt)
                    V["true_union_hard"][q, k] = metrics3(with_bonus(z, union), g, gt)
                    V["oracle"][q, k] = metrics3(with_bonus(z, rstar), g, gt)
            print(f"[{time.time() - t0:.0f}s] {arm} seed {k} done", flush=True)
        rec = {"stored_metrics_max_abs_diff": stored_diff, "units": units, "variants": {}, "by_hop": {}}
        for name in VARIANTS:
            e = L8.read_arm(V[name], T, G, dens, readable, W)
            e.pop("_boot", None)
            rec["variants"][name] = {"rho_bar": e["rho_bar"], "rho": {m: e["rho"][m]["point"] for m in RET},
                                     "mean": {m: e["mean"][m]["arm"] for m in RET}}
        for h in (1, 2, 3):
            mask = hop == h
            _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
            rec["by_hop"][f"hop={h}"] = {}
            for name in VARIANTS:
                e = L8.read_arm(V[name], T, G, s_dens, s_read, W, mask)
                rec["by_hop"][f"hop={h}"][name] = {"rho_bar": e["rho_bar"]["point"], "rho": {m: e["rho"][m]["point"] for m in RET}}
        ok = pq["argmax_ok"] == 1.0
        gap = V["oracle"] - V["learned"]                       # (q, k, metric)
        share = {}
        for sel_name, sel in (("argmax_right", ok), ("argmax_wrong", ~ok)):
            share[sel_name] = {"pairs": int(sel.sum()),
                               "gap_rho_units": {m: float(np.where(sel, gap[:, :, i], 0.0).sum() / len(L9.SEEDS) / den_sum[m])
                                                 for i, m in enumerate(RET)},
                               "by_hop_pairs": {f"hop={h}": int((sel & (hop[:, None] == h)).sum()) for h in (1, 2, 3)},
                               "gap_rho_units_by_hop": {f"hop={h}": {m: float(np.where(sel & (hop[:, None] == h), gap[:, :, i], 0.0).sum()
                                                                            / len(L9.SEEDS) / den_sum[m]) for i, m in enumerate(RET)}
                                                        for h in (1, 2, 3)}}
        rec["gap_split_by_argmax"] = share
        rec["per_pair"] = {
            "sum_m_err": summarise(pq["sum_m_err"]), "nm_min": summarise(pq["nm_min"]),
            "mass_rstar_when_right": summarise(pq["mass_rstar"], ok), "mass_rstar_when_wrong": summarise(pq["mass_rstar"], ~ok),
            "mass_union_when_right": summarise(pq["mass_union"], ok), "mass_union_when_wrong": summarise(pq["mass_union"], ~ok),
            "argmax_null_share": float(np.nanmean(pq["argmax_null"])),
            "argmax_has_gold_when_wrong": summarise(pq["argmax_has_gold"], ~ok & (pq["argmax_null"] == 0)),
            "argmax_size_when_right": summarise(pq["argmax_size"], ok), "argmax_size_when_wrong": summarise(pq["argmax_size"], ~ok),
            "rstar_size": summarise(pq["rstar_size"]), "rstar_has_gold": summarise(pq["rstar_has_gold"]),
            "argmax_bucket_equals_oracle_when_right": summarise((pq["argmax_bucket"] == pq["oracle_bucket"]).astype(float), ok),
            "agreement_by_hop": {f"hop={h}": float(ok[hop == h].mean()) for h in (1, 2, 3)}}
        rec["mass_rstar_by_hop_when_right"] = {f"hop={h}": summarise(pq["mass_rstar"], ok & (hop[:, None] == h)) for h in (1, 2, 3)}
        rec["confusions_k0_top"] = [{"hop": h, "qtype": qt, "argmax": nm_, "queries": c} for (h, qt, nm_), c in conf.most_common(40)]
        per_qt = Counter()
        for q in range(n):
            per_qt[(int(hop[q]), qt_names[q])] += 1
        rec["qtype_wrong_share_k0"] = sorted(
            [{"hop": h, "qtype": qt, "queries": c, "wrong": int(sum(v_ for (h2, qt2, _nm), v_ in conf.items() if (h2, qt2) == (h, qt)))}
             for (h, qt), c in per_qt.items()], key=lambda d: (-d["wrong"] / d["queries"], d["qtype"]))
        out["arms"][arm] = rec
        np.savez_compressed(OUT / f"per_pair_{arm}.npz", **{f"V_{k_}": v_ for k_, v_ in V.items()}, **{f"pq_{k_}": v_ for k_, v_ in pq.items()})
        print(f"[{time.time() - t0:.0f}s] {arm}: " + ", ".join(f"{nm_} {rec['variants'][nm_]['rho_bar']['point']:.3f}" for nm_ in VARIANTS),
              flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    (OUT / "diag.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"done in {out['seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
