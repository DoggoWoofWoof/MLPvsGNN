"""Descriptive look, after level 14's read (99a2685), at which walk types level 14's typed-walk units pick on r when the
pick (the posterior mode, before tempering) is not the true chain. Not a result and not filed. scripts/mp_approx_l14_fit.py
and every module it imports are imported and not edited; level 14's units and sidecar are read in place and never written.

Level 14's ceiling_gap_b0 split 0.016 on right-chain pairs and 0.083 on wrong-chain pairs at 1x (0.055 of it on hop 3).
This look sorts the wrong-chain (row, seed) pairs of a unit by kind, using the row's qtype only to name the true chain (as
level 14's agreement anchor does), and gives each kind's share of the gap to the unit's oracle (level 10's gap_split sum,
over the same overall denominator, so the kinds sum to the wrong-chain part):

  null       the null type is the mode
  absent     the true chain has no walk in the unit's view of the row (the bucket-0 seed, or both seeds, cannot reach it)
  shorter    a walk with fewer tokens than the true chain (the 1-hop shortcut of a 3-hop question, say)
  longer     more tokens
  dir_only   the true chain's relations in the same order, a direction differs
  rel@i      same length, the relation differs at the positions i (0-based), e.g. rel@0, rel@1+2

For each kind: pairs, the gap share, and the mean Jaccard between the picked walk's reach set and the true chain's (a
benign confusion has a high Jaccard). Per qtype: rows, agreement, the wrong-chain gap, the top picked sequences.

    python outputs/mp_approx_l14_diag/diag_chain.py --host    # -> diag_chain.json
"""
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l14_fit as F14  # noqa: E402  (level 14's fit module, imported unchanged)

P, F10, L0, L8, L9 = F14.P, F14.F10, F14.L0, F14.L8, F14.L9
SEEDS, FUNCS, METRIC_NAMES, RETRIEVAL, HOPS = F14.SEEDS, F14.FUNCS, F14.METRIC_NAMES, F14.RETRIEVAL, F14.HOPS
OUT_JSON = HERE / "diag_chain.json"
# (unit view, fit, the unit's score read as the arm, the oracle its gap is taken to)
UNITS = (("b0", "TW-1x", "dsh", "NB-oracle-b0"), ("b0", "TW-4x", "dsh", "NB-oracle-b0"), ("full", "TW-1x", "b1d", "NB-oracle"))


def log(msg):
    print(f"[{L0.utc()}] {msg}", flush=True)


def seq_name(seq) -> str:
    if seq == ("null",):
        return "null"
    out = []
    for t in seq:
        r, d = divmod(int(t), 3)
        out.append(L8.REL_ORDER[r] + ("" if d == 0 else "^-1" if d == 1 else "^~"))
    return ">".join(out)


def kind(truth: tuple, seq: tuple, present: bool) -> str:
    if seq == ("null",):
        return "null"
    if not present:
        return "absent"
    if len(seq) < len(truth):
        return "shorter"
    if len(seq) > len(truth):
        return "longer"
    rel = [i for i in range(len(seq)) if seq[i] // 3 != truth[i] // 3]
    if not rel:
        return "dir_only"
    return "rel@" + "+".join(str(i) for i in rel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    P.HARD_STOP_DIR[0] = HERE   # a hard stop of this look lands here, never in level 14's outputs
    P.route_stops()
    decl = P.load_declaration()
    if args.host:
        L8.host_mode(decl, log)
    P.verify_inputs(decl)
    vs = L9.views(F14.DATA)
    base, nb = vs["std"], vs["nb"]   # a b0 unit's codes are nb codes of bucket 0, so nb.reach serves both views
    n = base.n_q
    ri = [METRIC_NAMES.index(m) for m in RETRIEVAL]
    qm = base.q_metrics
    T = qm[:, [FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, ri]
    G = qm[:, [FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, ri]
    hop = np.asarray(base.q_hop)
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    qts = base.meta["qtypes"]
    qt_names = [qts[i] for i in base.q_qtype]
    chains = {qt: L8.true_chain(qt) for qt in qts}
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    # the oracles, as level 14's read computes them
    Mo = {ref: np.zeros((n, len(SEEDS), len(RETRIEVAL))) for ref in ("NB-oracle", "NB-oracle-b0")}
    true_reach = {}
    for q in range(n):
        steps = chains[qt_names[q]]
        r0, r1 = L8.chain_reach(nb, q, steps)
        rs, _b = L8.r_star(nb, q, steps)
        true_reach[q] = (r0, r1, rs)
        g, gt = nb.gold_local(q), int(nb.q_gold_total[q])
        for ref, reach in (("NB-oracle", rs), ("NB-oracle-b0", r0)):
            bonus = np.zeros(int(nb.q_pool_size[q]))
            bonus[reach] = L8.ORACLE_BONUS
            for i, k in enumerate(SEEDS):
                rm = F14.rank_metrics(nb.z(q, k) + bonus, g, gt)
                Mo[ref][q, i] = [rm[m] for m in RETRIEVAL]
    log(f"oracles recomputed on {n} rows")
    out = {"look": "diag_chain", "after": "99a2685", "rows": n, "readable": readable, "units": {}}
    for view, fit, sc, ref in UNITS:
        uk = f"{view}/{fit}"
        Mx = np.zeros((n, len(SEEDS), len(RETRIEVAL)))
        A = np.zeros((n, len(SEEDS)), dtype=np.int64)
        for i, k in enumerate(SEEDS):
            npz, js = F14.unit_paths(view, fit, k)
            flog = L8.read_json(js)
            if L0.sha256_file(npz) != flog["arrays_sha256"]:
                raise SystemExit(f"{npz}: not the arrays its log records")
            with np.load(npz) as z:
                if not np.array_equal(z["q"], np.arange(n)):
                    raise SystemExit(f"{npz}: not r's rows, each once")
                Mx[:, i] = z[f"metrics_{sc}"][:, ri]
                A[:, i] = z["argmax"]
        diff = {m: Mo[ref][:, :, j] - Mx[:, :, j] for j, m in enumerate(RETRIEVAL)}
        den = {m: len(SEEDS) * float(dens[m].sum()) for m in RETRIEVAL}

        def gap(sel: np.ndarray) -> float:
            return float(np.mean([float((diff[m] * sel).sum()) / den[m] for m in readable]))

        kinds = np.empty((n, len(SEEDS)), dtype=object)
        jac = np.full((n, len(SEEDS)), np.nan)
        right = np.zeros((n, len(SEEDS)), dtype=bool)
        picked = np.empty((n, len(SEEDS)), dtype=object)
        for q in range(n):
            r0, r1, rs = true_reach[q]
            present = bool(r0.size) if view == "b0" else bool(r0.size or r1.size)
            tr = r0 if view == "b0" else rs
            for i in range(len(SEEDS)):
                code = int(A[q, i])
                seq = L8.token_sequence(code)
                picked[q, i] = seq_name(seq)
                if seq == truth[q]:
                    right[q, i] = True
                    kinds[q, i] = "right"
                    continue
                kinds[q, i] = kind(truth[q], seq, present)
                if code >= 0:
                    if view == "b0" and code // F14.TBL != 0:
                        raise SystemExit(f"{uk} row {q}: a b0 unit's mode outside bucket 0")
                    jac[q, i] = L8.jaccard(nb.reach(q, code), tr)
        res = {"arm": f"{fit}-{sc}", "oracle": ref, "right_pairs": int(right.sum()), "wrong_pairs": int((~right).sum()),
               "gap_right": gap(right), "gap_wrong": gap(~right), "kinds": {}, "by_hop": {}, "qtypes": {}}
        for h in HOPS:
            hm = (hop == h)[:, None]
            res["by_hop"][f"hop={h}"] = {"pairs": int(hm.sum() * len(SEEDS)), "right_share": float(right[hop == h].mean()),
                                         "gap_right": gap(right & hm), "gap_wrong": gap(~right & hm), "kinds": {}}
        for kd in sorted(set(kinds.ravel()) - {"right"}):
            sel = kinds == kd
            res["kinds"][kd] = {"pairs": int(sel.sum()), "gap": gap(sel), "jaccard_mean": float(np.nanmean(jac[sel])) if np.isfinite(jac[sel]).any() else None}
            for h in HOPS:
                hm = sel & (hop == h)[:, None]
                if hm.any():
                    res["by_hop"][f"hop={h}"]["kinds"][kd] = {"pairs": int(hm.sum()), "gap": gap(hm),
                                                              "jaccard_mean": float(np.nanmean(jac[hm])) if np.isfinite(jac[hm]).any() else None}
        by_qt = defaultdict(list)
        for q in range(n):
            by_qt[qt_names[q]].append(q)
        for qt, qs in sorted(by_qt.items(), key=lambda kv: -len(kv[1])):
            qs = np.asarray(qs)
            sel = np.zeros((n, len(SEEDS)), dtype=bool)
            sel[qs] = True
            wrong = sel & ~right
            top = Counter()
            top_gap = defaultdict(float)
            for q, i in zip(*np.nonzero(wrong)):
                top[(kinds[q, i], picked[q, i])] += 1
            for (kd, name), _c in top.items():
                s2 = wrong & (kinds == kd) & (picked == name)
                top_gap[(kd, name)] = gap(s2)
            res["qtypes"][qt] = {"hop": int(hop[qs[0]]), "rows": int(qs.size), "truth": seq_name(truth[qs[0]]),
                                 "right_share": float(right[qs].mean()), "gap_wrong": gap(wrong),
                                 "top_wrong": [{"kind": kd, "picked": name, "pairs": c, "gap": top_gap[(kd, name)]}
                                               for (kd, name), c in sorted(top.items(), key=lambda kv: -top_gap[kv[0]])[:6]]}
        out["units"][uk + "/" + sc] = res
        log(f"{uk} {sc} vs {ref}: right {res['right_pairs']} wrong {res['wrong_pairs']}; gap right {res['gap_right']:.4f} "
            f"wrong {res['gap_wrong']:.4f}")
        for kd, v in sorted(res["kinds"].items(), key=lambda kv: -kv[1]["gap"]):
            jm = "n/a" if v["jaccard_mean"] is None else f"{v['jaccard_mean']:.3f}"
            hb = " / ".join(f"{res['by_hop'][f'hop={h}']['kinds'].get(kd, {}).get('gap', 0.0):+.4f}" for h in HOPS)
            log(f"   {kd:10s} pairs {v['pairs']:6d}  gap {v['gap']:+.4f} (by hop {hb})  jaccard {jm}")
        for qt, v in sorted(res["qtypes"].items(), key=lambda kv: -kv[1]["gap_wrong"])[:12]:
            log(f"   {qt:40s} hop {v['hop']} rows {v['rows']:5d} right {v['right_share']:.3f} gap_wrong {v['gap_wrong']:+.4f}  true {v['truth']}")
            for t in v["top_wrong"][:3]:
                log(f"        {t['kind']:9s} {t['picked']:55s} pairs {t['pairs']:5d} gap {t['gap']:+.4f}")
    out.update({"diag_script_sha256": L0.sha256_file(Path(__file__)), "seconds": round(time.time() - t0, 1)})
    L8.write_json(OUT_JSON, out)
    log(f"wrote {OUT_JSON} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
