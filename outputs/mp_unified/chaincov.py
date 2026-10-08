"""What the chain entries' caps cost a typed graph's chain match (a diagnosis, docs/SCREENS.md; decides nothing).

rmatch's build (rmatch.question_entries) keeps, per row and seed bucket, the 64 heaviest entries by walk mass (the row
cap), and stops a walk after the hop level that takes its chain count above 20,000 (the chain cap). On webqsp the row
cap keeps 0.383 and 0.297 of the walk's mass (buckets 0 and 1), and the chain cap stops 593 and 1,078 of 1,503 walks
(docs/SCREENS.md, round twelve's builds). Mass is set by the graph's degrees, not by the question, so the chains a row
keeps need not be the ones the question asks for.

Per question of a typed carve, rmatch's walk is re-run as its build runs it (the same caps), and every entry before the
row cap is scored by the match at its start: w0(c) = (1/3) prod_k sigmoid(zcos_q(r_k)) (rmatch.ChainMatch with kappa 1
and every other match parameter 0; r_k the relation of the chain's k-th step, zcos as ChainMatch.zcos). Three entry
sets are compared:
    all       every entry (no row cap)
    mass      the build's: per row and bucket the 64 heaviest by mass (ties to the earlier chain)
    contrib   per row and bucket the 64 with the largest w0(c) m_c(v) (ties to the earlier chain)
For each set: R@5 of the pool's rows ranked by S0(v) = sum over the set's entries of w0 m (rows with S0 = 0 last, ties
in pool order), over the questions with a gold, out of each question's gold total as the reads count it; and the share
of S0 (all entries) each capped set keeps on gold rows and on the other reached rows. Golds reached by any chain are
counted too, for questions whose walk the chain cap stopped and the others. Nothing is trained or read; the labels are
only counted.

    python outputs/mp_unified/chaincov.py --dataset webqsp --carve s1eval --out outputs/diag/chaincov-webqsp-s1eval
    python outputs/mp_unified/chaincov.py --selftest
"""
import os
import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import rmatch as RM  # noqa: E402

LC, LM = RM.LC, RM.LM
SETS = ("all", "mass", "contrib")
GOLD_KEYS = ("q_pool_size", "q_gold_total", "q_emb", "is_gold")


def zcos_of(q, U, pool_rels):
    """ChainMatch.zcos for one question in float64: q's cosine with every relation, z-scored over the relations on
    its pool (0 when fewer than two, or none that vary)."""
    q = np.asarray(q, np.float64)
    q = q / max(float(np.linalg.norm(q)), 1e-12)
    cos = U @ q
    v = cos[pool_rels]
    if v.size < 2:
        return np.zeros_like(cos)
    var = float(((v - v.mean()) ** 2).mean())
    if var <= 1e-12:
        return np.zeros_like(cos)
    return (cos - v.mean()) / np.sqrt(var)


def keep_top(row, key, cid, k_row):
    """Per row the k_row entries with the largest key (ties to the earlier chain), as rmatch.question_entries keeps
    them by mass."""
    order = np.lexsort((cid, -key, row))
    rs = row[order]
    first = np.r_[True, rs[1:] != rs[:-1]]
    start = np.maximum.accumulate(np.where(first, np.arange(rs.size), 0))
    keep = np.zeros(row.size, bool)
    keep[order[np.arange(rs.size) - start < k_row]] = True
    return keep


def bucket_entries(n, g, seeds, zc_of_step, hops=RM.HOPS, cap=RM.MAX_CHAINS):
    """One bucket's entries before the row cap, as question_entries builds them: (row, chain id, mass renormalised over
    the chain's non-seed rows, w0 of the chain), and whether the chain cap stopped the walk."""
    levels, capped = RM.walk(n, g, seeds, hops, cap)
    J = sum(t.shape[0] for t, _o, _v, _m in levels)
    if not J:
        return None, bool(capped)
    zc = np.full((J, hops), -1, np.int64)
    at = 0
    for t, _o, _v, _m in levels:
        zc[at:at + t.shape[0], :t.shape[1]] = t
        at += t.shape[0]
    cid = np.repeat(np.arange(J), np.concatenate([np.diff(o) for _t, o, _v, _m in levels]))
    node = np.concatenate([v for _t, _o, v, _m in levels])
    mass = np.concatenate([m for _t, _o, _v, m in levels])
    seed = np.zeros(n, bool)
    seed[seeds] = True
    keep = ~seed[node]
    if not keep.any():
        return None, bool(capped)
    ck = cid[keep]
    mass = mass[keep] / np.bincount(ck, weights=mass[keep], minlength=J)[ck]
    valid = zc >= 0
    lw = np.where(valid, zc_of_step[np.maximum(zc, 0)], 0.0).sum(1) + np.log(1.0 / 3.0)
    return (node[keep], ck, mass, np.exp(lw)[ck]), bool(capped)


def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


def question_scores(n, hl, tl, sl, seeds_by_bucket, kz, zcos, k_row=RM.K_ROW):
    """S0 of every pool row under each entry set, and the chain-cap flag of each bucket's walk."""
    g = RM.typed_graph(n, hl, tl, sl, kz)
    step = log_sigmoid(np.repeat(zcos, 2))                      # z = 2 r + d: the step's relation is z // 2
    S = {s: np.zeros(n) for s in SETS}
    capped = []
    for seeds in seeds_by_bucket:
        e, cp = bucket_entries(n, g, seeds, step)
        capped.append(cp)
        if e is None:
            continue
        row, cid, mass, w0 = e
        contrib = w0 * mass
        S["all"] += np.bincount(row, weights=contrib, minlength=n)
        km = keep_top(row, mass, cid, k_row)
        S["mass"] += np.bincount(row[km], weights=contrib[km], minlength=n)
        kc = keep_top(row, contrib, cid, k_row)
        S["contrib"] += np.bincount(row[kc], weights=contrib[kc], minlength=n)
    return S, capped


def r_at_5(score, gold, gold_total):
    """Golds among the 5 rows with the largest score (rows with score 0 after every scored row, ties in pool order),
    over the question's gold total."""
    order = np.lexsort((np.arange(score.size), -score))
    return float(gold[order[:5]].sum()) / gold_total


def run(ds, carve, out, look_root=LM.LOOK, rel_dir=RM.REL_DIR, limit=None):
    t0 = time.time()
    d, files, _ids, head = LC.look_chunks(ds, carve, look_root)
    rel = RM.look_relations(d)
    n_rel, k_rel = int(rel["n_relations"]), int(rel["k_rel"])
    E = np.load(Path(rel_dir) / f"{ds}_rel_embeddings.npy").astype(np.float64)
    if E.shape[0] != n_rel:
        raise SystemExit(f"{ds}: {E.shape[0]} relation embeddings, the look has {n_rel} relations")
    U = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    acc = {"questions": 0, "with_gold": 0, "capped_b0": 0, "capped_any": 0}
    r5 = {s: [] for s in SETS}
    r5_cap = {s: [] for s in SETS}
    kept = {s: {"gold": [0.0, 0.0], "other": [0.0, 0.0]} for s in ("mass", "contrib")}
    reach = {"capped": [0, 0], "not_capped": [0, 0]}
    done = False
    for f in files:
        with np.load(f) as zf:
            c = {k: zf[k] for k in RM.KEYS + GOLD_KEYS}
        qps = c["q_pool_size"].astype(np.int64)
        eo = np.r_[0, np.cumsum(c["q_edges"].astype(np.int64))]
        ro = np.r_[0, np.cumsum(qps)]
        if c["e_rel"].shape != (c["e_u"].size, k_rel) or int(ro[-1]) != c["is_gold"].size:
            raise SystemExit(f"{f}: its edges or rows do not match its question arrays")
        for i in range(qps.size):
            n = int(qps[i])
            gold = c["is_gold"][ro[i]:ro[i + 1]].astype(bool)
            gt = int(c["q_gold_total"][i])
            hl, tl, sl = RM.row_triples(c, int(eo[i]), int(eo[i + 1]))
            sl0, bk = c["q_seed_local"][i], c["q_seed_bucket"][i]
            ok = sl0 >= 0
            seeds = [np.unique(sl0[ok & (bk == b)]).astype(np.int64) for b in range(RM.BUCKETS)]
            pool_rels = np.unique(sl[sl >= 0])
            zc = zcos_of(c["q_emb"][i], U, pool_rels)
            S, capped = question_scores(n, hl, tl, sl, seeds, 2 * n_rel, zc)
            acc["questions"] += 1
            acc["capped_b0"] += int(capped[0])
            acc["capped_any"] += int(any(capped))
            if gt > 0:
                acc["with_gold"] += 1
                for s in SETS:
                    v = r_at_5(S[s], gold, gt)
                    r5[s].append(v)
                    if any(capped):
                        r5_cap[s].append(v)
                key = "capped" if any(capped) else "not_capped"
                reach[key][0] += int((gold & (S["all"] > 0)).sum())
                reach[key][1] += int(gold.sum())
            on = S["all"] > 0
            for s in ("mass", "contrib"):
                for nm, m in (("gold", gold & on), ("other", ~gold & on)):
                    kept[s][nm][0] += float(S[s][m].sum())
                    kept[s][nm][1] += float(S["all"][m].sum())
            if limit and acc["questions"] >= limit:
                done = True
                break
        if done:
            break
    rec = {"dataset": ds, "carve": carve, "look": head["records"], "carve_ids_sha256": head["carve_ids_sha256"],
           "walk": {k: RM.SETTINGS[k] for k in ("hops", "max_chains", "k_row", "buckets")},
           "prior": "w0(c) = (1/3) prod_k sigmoid(zcos_q(r_k)): rmatch.ChainMatch at its start", **acc,
           "R@5_by_S0": {s: float(np.mean(r5[s])) if r5[s] else None for s in SETS},
           "R@5_by_S0_capped_questions": {s: float(np.mean(r5_cap[s])) if r5_cap[s] else None for s in SETS},
           "capped_questions_with_gold": len(r5_cap["all"]),
           "S0_kept_share": {s: {nm: (v[0] / v[1] if v[1] else None) for nm, v in kept[s].items()} for s in kept},
           "golds_reached": {k: {"reached": v[0], "golds_in_pool": v[1], "share": (v[0] / v[1] if v[1] else None)}
                             for k, v in reach.items()},
           "limit": limit, "script_sha256": LC.sha_src(__file__), "rmatch_sha256": LC.sha_src(RM.__file__),
           "seconds": round(time.time() - t0, 1)}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    LC.write_json(out.with_suffix(".json"), rec)
    md = [f"# Chain caps on {ds}/{carve} (a diagnosis; decides nothing)", "",
          f"{acc['questions']} questions, {acc['with_gold']} with a gold; the chain cap stopped bucket 0's walk on "
          f"{acc['capped_b0']} and some bucket's on {acc['capped_any']}.", "",
          "| entry set | R@5 by S0 | R@5 by S0, questions the chain cap stopped | S0 kept on gold rows | on other rows |",
          "| --- | ---: | ---: | ---: | ---: |"]
    for s in SETS:
        a, b = rec["R@5_by_S0"][s], rec["R@5_by_S0_capped_questions"][s]
        kg = rec["S0_kept_share"][s]["gold"] if s != "all" else 1.0
        ko = rec["S0_kept_share"][s]["other"] if s != "all" else 1.0
        cells = ["" if x is None else f"{x:.4f}" for x in (a, b)] + ["" if x is None else f"{x:.3f}" for x in (kg, ko)]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += ["", "| questions | golds in pool | reached by a chain | share |", "| --- | ---: | ---: | ---: |"]
    for k, v in rec["golds_reached"].items():
        sh = "" if v["share"] is None else f"{v['share']:.3f}"
        md.append(f"| {k.replace('_', ' ')} | {v['golds_in_pool']} | {v['reached']} | {sh} |")
    out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    return rec


def selftest():
    rng = np.random.default_rng(0)
    # keep_top is question_entries' rule when the key is the mass
    row = rng.integers(0, 6, 400)
    cid = np.arange(400)
    mass = rng.random(400)
    k = keep_top(row, mass, cid, 3)
    for r in range(6):
        m = row == r
        want = np.sort(cid[m][np.argsort(-mass[m], kind="stable")][:3])
        assert np.array_equal(np.sort(cid[m & k]), want)
    # one toy question: S0 under every set equals a brute force over rmatch's entries; the mass set is the build's
    n, n_rel = 30, 4
    hl, tl = rng.integers(0, n, 60), rng.integers(0, n, 60)
    sl = np.full((60, 2), -1, np.int64)
    sl[:, 0] = rng.integers(0, n_rel, 60)
    seeds = [np.array([0, 1]), np.array([2])]
    zc = rng.standard_normal(n_rel)
    S, capped = question_scores(n, hl, tl, sl, seeds, 2 * n_rel, zc, k_row=2)
    assert capped == [False, False]
    (row_b, z_b, b_b, m_b), _st = RM.question_entries(n, hl, tl, sl, seeds, 2 * n_rel, k_row=2)
    valid = z_b >= 0
    w0 = np.exp(np.where(valid, log_sigmoid(zc[np.maximum(z_b, 0) // 2]), 0.0).sum(1) + np.log(1 / 3))
    assert np.allclose(np.bincount(row_b, weights=w0 * m_b, minlength=n), S["mass"])
    assert (S["all"] >= S["mass"] - 1e-12).all() and (S["all"] >= S["contrib"] - 1e-12).all()
    assert S["contrib"].sum() >= S["mass"].sum() - 1e-12
    # the k_row large enough to keep everything: all three sets agree
    S2, _c = question_scores(n, hl, tl, sl, seeds, 2 * n_rel, zc, k_row=10 ** 6)
    assert np.allclose(S2["all"], S2["mass"]) and np.allclose(S2["all"], S2["contrib"])
    # r_at_5: unscored rows last, ties in pool order
    assert r_at_5(np.array([0, 0, 3, 0, 0, 0, 1.0]), np.array([1, 0, 1, 0, 0, 0, 1], bool), 3) == 1.0
    assert r_at_5(np.zeros(8), np.r_[np.zeros(7), 1].astype(bool), 1) == 0.0
    # zcos_of matches ChainMatch.zcos on one question
    U = rng.standard_normal((n_rel, 5))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    q = rng.standard_normal(5)
    z = zcos_of(q, U, np.array([0, 2, 3]))
    cos = U @ (q / np.linalg.norm(q))
    v = cos[[0, 2, 3]]
    assert np.allclose(z, (cos - v.mean()) / v.std())
    assert np.allclose(zcos_of(q, U, np.array([1])), 0.0)
    print("chaincov selftest: ok")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset")
    ap.add_argument("--carve", default="s1eval")
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.dataset not in RM.TYPED or not a.out:
        ap.error(f"--dataset one of {RM.TYPED}, and --out")
    run(a.dataset, a.carve, a.out, limit=a.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
