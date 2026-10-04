"""Descriptive look at level 12's TW-1x units, run on the host after level 12's run record (a43746a). Not a result and
not filed. It informs the design of level 13, whose population leaves level 12's 2,221 dev rows out and whose file
discloses the look.

Each TW-1x unit is refitted exactly as level 12 fitted it (scripts/mp_approx_l12_fit.py's deploy view, then level 11's
fit_and_capture and score_dsh, imported and called unchanged, at 4 threads with torch's deterministic algorithms), and
its arrays are compared with the filed unit's. The per-query type log-probabilities, which the filed unit does not keep,
are captured and saved, and the dev rows are scored again by level 11's score_dsh (beta, kappa and eta chosen together
on the select carve's rows by the mean of recall@5, full_coverage@5 and hit@1, level 11's ties) on a transformed
posterior. A transform acts on p(tau | q) before the tempering:
- dsh: none (it must equal the filed unit bit for bit),
- b0merge: each bucket-1 type's mass added to the bucket-0 type of the same token sequence (no gold read; deployable),
- b1drop: each bucket-1 type's mass moved to the null type (no gold read; deployable),
- bktfix: each sequence's mass moved to the bucket of the true chain's R* (a reference: R*'s bucket reads the golds),
- tebkt: each sequence's mass moved to the bucket holding the annotated topic entity, when it is a seed (a reference:
  the annotation is an input of neither the GNN nor the model),
- seqfix: the posterior restricted to the true chain's token sequence, both buckets, renormalised (a reference),
- seqfix_b0merge: seqfix, then b0merge (a reference),
- seqbktfix: all mass on (R*'s bucket, the true sequence) (a reference; about the NB-oracle under a soft gate).
A transform leaves a query's posterior as it is when the type it needs is not among the query's types.

The grid stage scores the select carve's rows and the dev rows under every (beta, kappa, eta) of level 11's grid with
the unit's own posterior (no transform), and keeps each row's three metrics and its posterior's statistics (the largest
probability, the null type's, the margin, the entropy, the argmax type's length and bucket). gridread then asks whether
choosing (beta, kappa, eta) per stratum of those statistics (level 11's rule within each stratum, on the select carve's
rows) closes the gap; a stratum rule that reads the true chain is a reference.

    python outputs/mp_approx_l12_diag/diag.py unit --k 0 --host    # the refitted TW-1x unit of seed k and its variants
    python outputs/mp_approx_l12_diag/diag.py read --host           # rho per variant, the gap by cell -> diag.json
    python outputs/mp_approx_l12_diag/diag.py grid --k 0 --host    # per-row metrics over the whole grid (after unit)
    python outputs/mp_approx_l12_diag/diag.py gridread --host       # stratified choices -> diag_grid.json
    python outputs/mp_approx_l12_diag/diag.py mix --k 0 --post b1drop --host   # count-weighted mixtures (after unit)
    python outputs/mp_approx_l12_diag/diag.py mixread --host        # -> diag_mix.json

The mix stage, added after read and gridread, weights each type's reach set by c_tau(v)^alpha (alpha 0.5 and 1; 0 is
the uniform reach set) under the unit's posterior (dsh) or b1drop's, with level 11's selection unchanged.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "4"

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path.cwd()
for _p in (ROOT / "src", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mp_approx_l12_fit as F12  # noqa: E402

F11, F10, L8, L9, L0, P = F12.F11, F12.F10, F12.L8, F12.L9, F12.L0, F12.P
OUT = ROOT / "outputs" / "mp_approx_l12_diag"
FIT = "TW-1x"
RET = F12.RETRIEVAL
RI = [F12.METRIC_NAMES.index(m) for m in RET]
SEEDS = F12.SEEDS
TBL = L8.TB ** L8.MAX_L                   # a type code's bucket is code // TBL, its token sequence code % TBL
TRANSFORMS = ("dsh", "b0merge", "b1drop", "bktfix", "tebkt", "seqfix", "seqfix_b0merge", "seqbktfix")
REFERENCE_TRANSFORMS = ("bktfix", "tebkt", "seqfix", "seqfix_b0merge", "seqbktfix")
PAIR_KEYS = ("p_true", "p_true_b0", "p_true_b1", "p_b1", "p_bstar", "p_null", "conf", "bstar", "rstar_size", "te_bucket",
             "seeds_b0", "seeds_b1", "true_types")
COMBOS = [(b, kp, e) for b in F11.BETAS for kp in F11.KAPPAS for e in F11.ETAS]   # level 11's grid, in its loop order
FEATURES = ("conf", "conf_t", "p_null", "margin", "entropy", "arg_len", "arg_bkt", "right")
MIN_STRATUM = 100                         # a stratum with fewer select-carve rows takes the global choice


# ── the transforms (pure: arrays in, log-probabilities out) ──────────────────


def to_log(p: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(p)


def move_to_bucket(codes: np.ndarray, p: np.ndarray, target: int) -> tuple[np.ndarray, bool]:
    """Each type outside bucket `target` gives its mass to the type of bucket `target` with the same token sequence,
    when the query has that type. p holds the query's types then the null type; codes are ascending."""
    codes = np.asarray(codes, dtype=np.int64)
    out = np.array(p, dtype=np.float64, copy=True)
    bk, seq = codes // TBL, codes % TBL
    src = np.flatnonzero(bk != target)
    if not src.size:
        return out, False
    want = target * TBL + seq[src]
    j = np.searchsorted(codes, want)
    ok = (j < codes.size)
    ok[ok] = codes[j[ok]] == want[ok]
    src, j = src[ok], j[ok]
    if not src.size:
        return out, False
    np.add.at(out, j, out[src])
    out[src] = 0.0
    return out, True


def drop_bucket(codes: np.ndarray, p: np.ndarray, bucket: int) -> tuple[np.ndarray, bool]:
    """Every type of `bucket` gives its mass to the null type (the last entry)."""
    codes = np.asarray(codes, dtype=np.int64)
    out = np.array(p, dtype=np.float64, copy=True)
    m = np.r_[codes // TBL == bucket, False]
    if not m.any():
        return out, False
    out[-1] += out[m].sum()
    out[m] = 0.0
    return out, True


def restrict_sequence(codes: np.ndarray, p: np.ndarray, seq_code: int) -> tuple[np.ndarray, bool]:
    """p restricted to the types whose token sequence is seq_code (both buckets) and renormalised; the null type gets
    none. Unchanged when the query has no such type or they hold no mass."""
    codes = np.asarray(codes, dtype=np.int64)
    m = np.r_[codes % TBL == seq_code, False]
    tot = float(np.asarray(p)[m].sum())
    if not m.any() or tot <= 0:
        return np.array(p, dtype=np.float64, copy=True), False
    out = np.where(m, p, 0.0) / tot
    return out, True


def point_mass(codes: np.ndarray, p: np.ndarray, code: int) -> tuple[np.ndarray, bool]:
    codes = np.asarray(codes, dtype=np.int64)
    j = int(np.searchsorted(codes, code))
    if j >= codes.size or int(codes[j]) != code:
        return np.array(p, dtype=np.float64, copy=True), False
    out = np.zeros(np.asarray(p).size)
    out[j] = 1.0
    return out, True


def transform(name: str, lp: np.ndarray, codes: np.ndarray, true_seq: int, bstar: int, te_bucket: int) -> tuple[np.ndarray, bool]:
    """(log p after the transform, whether it changed anything). dsh returns lp itself, so it is level 11's bit for bit."""
    if name == "dsh":
        return lp, False
    p = np.exp(np.asarray(lp, dtype=np.float64))
    if name == "b0merge":
        out, ch = move_to_bucket(codes, p, 0)
    elif name == "b1drop":
        out, ch = drop_bucket(codes, p, 1)
    elif name == "bktfix":
        out, ch = move_to_bucket(codes, p, bstar)
    elif name == "tebkt":
        out, ch = move_to_bucket(codes, p, te_bucket) if te_bucket in (0, 1) else (p, False)
    elif name == "seqfix":
        out, ch = restrict_sequence(codes, p, true_seq)
    elif name == "seqfix_b0merge":
        out, ch = restrict_sequence(codes, p, true_seq)
        out, ch2 = move_to_bucket(codes, out, 0)
        ch = ch or ch2
    elif name == "seqbktfix":
        out, ch = point_mass(codes, p, bstar * TBL + true_seq)
    else:
        raise ValueError(name)
    return (to_log(out) if ch else lp), ch


# ── the per-query facts the transforms need ──────────────────────────────────


def te_bucket_of(te: int, seeds: np.ndarray, buckets: np.ndarray) -> int:
    """-2: the annotated topic entity is not in the pool; -1: in the pool, not a seed; else its seed bucket."""
    if te < 0:
        return -2
    hit = np.flatnonzero(np.asarray(seeds) == te)
    return int(np.asarray(buckets)[hit[0]]) if hit.size else -1


def query_facts(data, q: int, chains: dict) -> dict:
    q = int(q)
    steps = chains[data.meta["qtypes"][int(data.q_qtype[q])]]
    rs, bstar = L8.r_star(data, q, steps)
    part, i = data._at(q)
    seeds, buckets = np.asarray(part.q_seed_local[i]), np.asarray(part.q_seed_bucket[i])
    return {"codes": np.asarray(data.t_code[data.type_rows(q)], dtype=np.int64),
            "true_seq": L8.type_code(0, L8.chain_tokens(steps)), "bstar": int(bstar), "rstar_size": int(rs.size),
            "te_bucket": te_bucket_of(int(part.q_te_local[i]), seeds, buckets),
            "seeds_b0": int((buckets == 0).sum()), "seeds_b1": int((buckets == 1).sum())}


def pair_facts(f: dict, lp: np.ndarray) -> dict:
    """The posterior's mass on the true sequence by bucket, on bucket 1, on R*'s bucket and on the null type."""
    p = np.exp(np.asarray(lp, dtype=np.float64))
    codes = f["codes"]
    bk, seq = codes // TBL, codes % TBL
    pt, is_true = p[:-1], seq == f["true_seq"]
    return {"p_true": float(pt[is_true].sum()), "p_true_b0": float(pt[is_true & (bk == 0)].sum()),
            "p_true_b1": float(pt[is_true & (bk == 1)].sum()), "p_b1": float(pt[bk == 1].sum()),
            "p_bstar": float(pt[bk == f["bstar"]].sum()), "p_null": float(p[-1]), "conf": float(p.max()),
            "bstar": f["bstar"], "rstar_size": f["rstar_size"], "te_bucket": f["te_bucket"], "seeds_b0": f["seeds_b0"],
            "seeds_b1": f["seeds_b1"], "true_types": int(is_true.sum())}


# ── a unit ───────────────────────────────────────────────────────────────────


def pack(lps: list) -> tuple[np.ndarray, np.ndarray]:
    ptr = np.r_[0, np.cumsum([int(np.asarray(x).size) for x in lps])].astype(np.int64)
    return (np.concatenate([np.asarray(x, dtype=np.float64) for x in lps]) if lps else np.zeros(0)), ptr


def unpack(flat: np.ndarray, ptr: np.ndarray) -> list:
    return [flat[int(ptr[i]):int(ptr[i + 1])] for i in range(ptr.size - 1)]


def setup(host: bool, log):
    L8.fit_process()
    decl = P.load_declaration()
    P.HARD_STOP_DIR[0] = OUT
    P.route_stops()
    if host:
        L8.host_mode(decl, log)
    P.verify_inputs(decl)
    view = F12.deploy_view(decl, FIT)
    fx = F10.make_fitter(view, L8.load_rel_emb(decl), F11.LEVEL10_FIT)
    return decl, fx, {qt: L8.true_chain(qt) for qt in view.meta["qtypes"]}


def equal_to_filed(arrays: dict, flog: dict, d: dict, k: int) -> dict:
    npz, js = F12.unit_paths(FIT, k)
    filed = F12.read_json(js)
    same = {}
    with np.load(npz) as zf:
        for key in zf.files:
            if key in ("metrics_dsh", "score_dsh"):
                continue
            same[key] = bool(key in arrays and np.array_equal(zf[key], arrays[key]))
        same["metrics_dsh"] = bool(np.array_equal(zf["metrics_dsh"], d["metrics"]))
        same["score_dsh"] = bool(np.array_equal(zf["score_dsh"], np.concatenate(d["scores"]) if d["scores"] else np.zeros(0)))
    same["kept_round"] = filed["kept_round"] == flog["kept_round"]
    same["dsh_choice"] = (filed["beta_dsh"], filed["kappa_dsh"], filed["eta_dsh"]) == (F11.beta_label(d["beta"]), d["kappa"], d["eta"])
    same["dens_cov_choice"] = all(filed[f"{x}_{sc}"] == flog[f"{x}_{sc}"] for x in ("kappa", "eta") for sc in F10.SCORES)
    return same


def stage_unit(k: int, host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    _decl, fx, chains = setup(host, log)
    d_out = OUT / "units"
    d_out.mkdir(parents=True, exist_ok=True)
    npz, js = d_out / f"k{k}.npz", d_out / f"k{k}.json"
    if js.exists():
        log(f"{js} exists")
        return
    data = fx.data
    rows = F12.rows_by_part(data)
    cnpz, cjs = d_out / f"k{k}.cap.npz", d_out / f"k{k}.cap.json"
    if cjs.exists():   # the refit's capture, kept so that a later step that fails does not cost a refit
        flog = json.loads(cjs.read_text(encoding="utf-8"))
        with np.load(cnpz) as zf:
            arrays = {key[2:]: zf[key] for key in zf.files if key.startswith("A_")}
            cap = {"inner_q": zf["inner_q"], "score_q": zf["score_q"], "lp_inner": unpack(zf["lp_inner"], zf["lp_inner_ptr"]),
                   "lp_score": unpack(zf["lp_score"], zf["lp_score_ptr"])}
        log(f"k{k}: the capture read back from {cnpz}")
    else:
        arrays, flog, cap = F11.fit_and_capture(fx, k, F12.FOLD, log)
        li, li_ptr = pack(cap["lp_inner"])
        ls, ls_ptr = pack(cap["lp_score"])
        tmp = cnpz.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, inner_q=cap["inner_q"], score_q=cap["score_q"], lp_inner=li, lp_inner_ptr=li_ptr, lp_score=ls,
                            lp_score_ptr=ls_ptr, **{f"A_{key}": val for key, val in arrays.items()})
        os.replace(tmp, cnpz)
        L8.write_json(cjs, {key: flog[key] for key in ("kept_round", "timing", *(f"{x}_{sc}" for x in ("kappa", "eta") for sc in F10.SCORES))})
        flog = json.loads(cjs.read_text(encoding="utf-8"))
    t1 = time.time()
    inner_f = [query_facts(data, q, chains) for q in cap["inner_q"]]
    score_f = [query_facts(data, q, chains) for q in cap["score_q"]]
    res, sel, changed = {}, {}, {}
    for name in TRANSFORMS:
        t2 = time.time()
        lpi, chi = zip(*[transform(name, lp, f["codes"], f["true_seq"], f["bstar"], f["te_bucket"]) for lp, f in zip(cap["lp_inner"], inner_f)])
        lps, chs = zip(*[transform(name, lp, f["codes"], f["true_seq"], f["bstar"], f["te_bucket"]) for lp, f in zip(cap["lp_score"], score_f)])
        cap_v = {"inner_q": cap["inner_q"], "score_q": cap["score_q"], "lp_inner": list(lpi), "lp_score": list(lps)}
        d = F11.score_dsh(fx, k, cap_v)
        if name == "dsh":
            same = equal_to_filed(arrays, flog, d, k)
        res[f"V_{name}"] = d["metrics"][:, RI]
        sel[name] = {"beta": F11.beta_label(d["beta"]), "kappa": d["kappa"], "eta": d["eta"], "inner_mean3": d["inner_mean3"],
                     "seconds": round(time.time() - t2, 1)}
        changed[name] = {"inner": int(sum(chi)), "scored": int(sum(chs))}
        log(f"   k{k} {name}: beta/kappa/eta {sel[name]['beta']}/{d['kappa']}/{d['eta']}, inner mean3 {d['inner_mean3']:.4f}, "
            f"changed {changed[name]}, {sel[name]['seconds']}s")
        del d, cap_v, lpi, lps
    facts = [pair_facts(f, lp) for f, lp in zip(score_f, cap["lp_score"])]
    for key in PAIR_KEYS:
        res[key] = np.asarray([x[key] for x in facts], dtype=np.float64)
    li, li_ptr = pack(cap["lp_inner"])
    ls, ls_ptr = pack(cap["lp_score"])
    res.update(q=np.asarray(cap["score_q"], dtype=np.int64), inner_q=np.asarray(cap["inner_q"], dtype=np.int64),
               argmax=np.asarray(arrays["argmax"], dtype=np.int64), lp_inner=li, lp_inner_ptr=li_ptr, lp_score=ls, lp_score_ptr=ls_ptr)
    tmp = npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, npz)
    info = {"fit": FIT, "k": k, "refit_equals_filed": same, "kept_round": flog["kept_round"], "rows_by_part": rows,
            "selection": sel, "changed": changed, "fit_seconds": flog["timing"]["seconds"], "variant_seconds": round(time.time() - t1, 1),
            "inner_queries": int(cap["inner_q"].size), "scored_queries": int(cap["score_q"].size), **L8.job_fields(t0)}
    L8.write_json(js, info)
    log(f"k{k}: refit equals filed {all(same.values())} {same if not all(same.values()) else ''}; {round(time.time() - t0)}s")


# ── the grid (per-row metrics under every beta, kappa and eta) ───────────────


def row_features(codes: np.ndarray, lp: np.ndarray, true_seq: int) -> dict:
    """The posterior's statistics a deployed model has (all but `right`, which reads the true chain): the largest
    probability over the types and the null type, the largest over the types, the null type's, the margin between the
    two largest, the entropy, and the argmax type's length and bucket (0 and -1 when the null type is the argmax)."""
    p = np.exp(np.asarray(lp, dtype=np.float64))
    top2 = np.sort(p)[::-1][:2]
    a = int(np.argmax(lp))
    null = a == codes.size
    code = -1 if null else int(codes[a])
    pos = p[p > 0]
    return {"conf": float(p.max()), "conf_t": float(p[:-1].max()) if codes.size else 0.0, "p_null": float(p[-1]),
            "margin": float(top2[0] - (top2[1] if top2.size > 1 else 0.0)), "entropy": float(-(pos * np.log(pos)).sum()),
            "arg_len": 0 if null else len(L8.token_sequence(code)), "arg_bkt": -1 if null else code // TBL,
            "right": float((not null) and code % TBL == true_seq)}


def grid_metrics(fx, k: int, qs, lps) -> np.ndarray:
    """[combo, row, metric]: level 11's dsh score under each (beta, kappa, eta), the expressions of its score_dsh."""
    data = fx.data
    base = [(data.gold_local(int(q)), int(data.q_gold_total[q]), data.z(int(q), k)) for q in qs]
    M = np.zeros((len(COMBOS), len(qs), len(RET)))
    ci = 0
    for beta in F11.BETAS:
        nms = [fx.mixture(int(q), F11.tempered(lp, beta)) for q, lp in zip(qs, lps)]
        for kappa in F11.KAPPAS:
            for eta in F11.ETAS:
                for j, (nm, (g, gt, z)) in enumerate(zip(nms, base)):
                    m = F12.rank_metrics(z + kappa * np.log(nm + eta), g, gt)
                    M[ci, j] = [m[r] for r in RET]
                ci += 1
        del nms
    return M


def inner_mean3(M: np.ndarray, rows: np.ndarray) -> dict:
    """level 11's grid value (L8.mean3: the mean over the metrics of the mean over the rows) per combo."""
    return {F11.grid_key(*c): float(np.mean([np.mean(np.ascontiguousarray(M[ci, rows, j])) for j in range(len(RET))]))
            for ci, c in enumerate(COMBOS)}


def stage_grid(k: int, host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    d_out = OUT / "units"
    gnpz, gjs = d_out / f"k{k}.grid.npz", d_out / f"k{k}.grid.json"
    if gjs.exists():
        log(f"{gjs} exists")
        return
    _decl, fx, chains = setup(host, log)
    data = fx.data
    with np.load(d_out / f"k{k}.cap.npz") as zf:
        parts = {"inner": (zf["inner_q"], unpack(zf["lp_inner"], zf["lp_inner_ptr"])),
                 "score": (zf["score_q"], unpack(zf["lp_score"], zf["lp_score_ptr"]))}
    res, seconds = {}, {}
    for part, (qs, lps) in parts.items():
        t1 = time.time()
        res[f"M_{part}"] = grid_metrics(fx, k, qs, lps)
        feats = []
        for q, lp in zip(qs, lps):
            f = query_facts(data, q, chains)
            feats.append(row_features(f["codes"], lp, f["true_seq"]))
        res[f"q_{part}"] = np.asarray(qs, dtype=np.int64)
        for key in FEATURES:
            res[f"{key}_{part}"] = np.asarray([f[key] for f in feats], dtype=np.float64)
        seconds[part] = round(time.time() - t1, 1)
        log(f"   k{k} grid {part}: {len(qs)} queries, {seconds[part]}s")
    # level 11's choice, recovered from the grid; gridread checks it, and its dev metrics, against the unit's dsh
    grid = inner_mean3(res["M_inner"], np.arange(res["M_inner"].shape[1]))
    choice = F11.choose(grid)
    tmp = gnpz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, gnpz)
    info = {"fit": FIT, "k": k, "combos": len(COMBOS), "global_choice": [F11.beta_label(choice[0]), choice[1], choice[2]],
            "global_inner_mean3": grid[F11.grid_key(*choice)], "seconds": seconds, **L8.job_fields(t0)}
    L8.write_json(gjs, info)
    log(f"k{k} grid: global choice {info['global_choice']}, inner mean3 {info['global_inner_mean3']:.4f}; {round(time.time() - t0)}s")


def grid_equals_dsh(k: int, g: dict) -> dict:
    """The grid's global choice is the unit's dsh choice, with its inner value, and scores the dev rows as it did."""
    gi = json.loads((OUT / "units" / f"k{k}.grid.json").read_text(encoding="utf-8"))
    unit = json.loads((OUT / "units" / f"k{k}.json").read_text(encoding="utf-8"))["selection"]["dsh"]
    b, kp, e = gi["global_choice"]
    ci = next(i for i, c in enumerate(COMBOS) if (F11.beta_label(c[0]), c[1], c[2]) == (b, kp, e))
    with np.load(OUT / "units" / f"k{k}.npz") as zf:
        same_metrics = bool(np.array_equal(g["M_score"][ci], zf["V_dsh"]))
    return {"choice": [b, kp, e] == [unit["beta"], unit["kappa"], unit["eta"]], "inner_mean3": gi["global_inner_mean3"] == unit["inner_mean3"],
            "dev_metrics": same_metrics}


def strata(rule: str, g: dict, part: str) -> np.ndarray:
    """The stratum of each row of `part` under `rule`; quantile cuts come from the select carve's rows."""
    x = lambda key, p=part: g[f"{key}_{p}"]   # noqa: E731
    if rule == "global":
        return np.zeros(x("conf").size, dtype=np.int64)
    if rule in ("conf3", "conf5", "margin3", "pnull3", "entropy3"):
        key = {"conf3": "conf", "conf5": "conf", "margin3": "margin", "pnull3": "p_null", "entropy3": "entropy"}[rule]
        nb = 5 if rule == "conf5" else 3
        cuts = np.quantile(g[f"{key}_inner"], [i / nb for i in range(1, nb)])
        return np.searchsorted(cuts, x(key), side="right").astype(np.int64)
    if rule == "len":
        return x("arg_len").astype(np.int64)
    if rule == "bkt":
        return (x("arg_bkt") + 1).astype(np.int64)
    if rule == "conf3_len":
        return strata("conf3", g, part) * 4 + strata("len", g, part)
    if rule == "conf3_bkt":
        return strata("conf3", g, part) * 3 + strata("bkt", g, part)
    if rule == "truth":
        return x("right").astype(np.int64)
    if rule == "truth_conf3":
        return strata("truth", g, part) * 3 + strata("conf3", g, part)
    raise ValueError(rule)


DEPLOYABLE_RULES = ("global", "conf3", "conf5", "margin3", "pnull3", "entropy3", "len", "bkt", "conf3_len", "conf3_bkt")
REFERENCE_RULES = ("truth", "truth_conf3")


def stratified(g: dict, rule: str) -> tuple[np.ndarray, dict]:
    """Level 11's choice within each stratum of the select carve's rows (MIN_STRATUM or more, else the global choice);
    each dev row takes its stratum's choice. Returns the dev rows' metrics and the choices."""
    M_in, M_sc = g["M_inner"], g["M_score"]
    s_in, s_sc = strata(rule, g, "inner"), strata(rule, g, "score")
    glob = COMBOS.index(F11.choose(inner_mean3(M_in, np.arange(M_in.shape[1]))))
    out = M_sc[glob].copy()
    picks = {}
    for s in np.unique(np.r_[s_in, s_sc]):
        rows = np.flatnonzero(s_in == s)
        ci = COMBOS.index(F11.choose(inner_mean3(M_in, rows))) if rows.size >= MIN_STRATUM else glob
        sel = s_sc == s
        out[sel] = M_sc[ci][sel]
        b, kp, e = COMBOS[ci]
        picks[str(int(s))] = {"inner_rows": int(rows.size), "dev_rows": int(sel.sum()), "beta": F11.beta_label(b), "kappa": kp, "eta": e,
                              "global": ci == glob}
    return out, picks


def dev_frame(host: bool, log):
    decl = P.load_declaration()
    P.HARD_STOP_DIR[0] = OUT
    P.route_stops()
    if host:
        L8.host_mode(decl, log)
    vs = L9.views(F12.DATA)
    base = vs["std"]
    qm = base.q_metrics
    T = qm[:, [F12.FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, RI]
    G = qm[:, [F12.FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, RI]
    W = L0.boot_weights(base.n_q)
    _den, dens, readable = L8.denominators(T, G, W)
    return vs, base, T, G, W, dens, readable, np.asarray(base.q_hop)


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    """P(a right row's statistic > a wrong row's), ties half (Mann-Whitney, by average ranks)."""
    if not pos.size or not neg.size:
        return None
    x = np.r_[pos, neg]
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size)
    ranks[order] = np.arange(1, x.size + 1)
    _u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def stage_gridread(host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    _vs, base, T, G, W, dens, readable, hop = dev_frame(host, log)
    n, S = base.n_q, len(SEEDS)
    gs, checks = [], {}
    for k in SEEDS:
        with np.load(OUT / "units" / f"k{k}.grid.npz") as zf:
            g = {key: zf[key] for key in zf.files}
        if not np.array_equal(g["q_score"], np.arange(n)):
            raise SystemExit(f"k{k}: the grid's dev rows are not the dev rows, each once")
        gs.append(g)
        checks[f"k{k}"] = grid_equals_dsh(k, g)
    rules, picks = {}, {}
    for rule in (*DEPLOYABLE_RULES, *REFERENCE_RULES):
        M = np.zeros((n, S, len(RET)))
        picks[rule] = {}
        for i, (k, g) in enumerate(zip(SEEDS, gs)):
            M[:, i], picks[rule][f"k{k}"] = stratified(g, rule)
        rules[rule] = M
    # the row-wise best combination (an upper bound of any choice within the family, a reference)
    M = np.zeros((n, S, len(RET)))
    for i, g in enumerate(gs):
        best = g["M_score"].mean(axis=2).argmax(axis=0)   # first of the largest
        M[:, i] = g["M_score"][best, np.arange(n)]
    rules["row_best"] = M
    arms = {r: F12.read_arm(M, T, G, dens, readable, W) for r, M in rules.items()}
    out = {"queries": n, "readable": readable, "min_stratum": MIN_STRATUM, "grid_checks": checks,
           "global_equals_dsh": None, "rules": {}, "by_hop": {}, "picks": picks}
    with np.load(OUT / "units" / "k0.npz") as zf:
        out["global_equals_dsh"] = bool(np.array_equal(rules["global"][:, 0], zf["V_dsh"]))
    for r, e in arms.items():
        out["rules"][r] = {"rho_bar": e["rho_bar"], "band": e["band"], "rho": {m: e["rho"][m]["point"] for m in RET},
                           "gap_to_gnn": {m: {"point": e["gap_to_gnn"][m]["point"], "flag": e["gap_to_gnn"][m]["flag"]} for m in RET},
                           "minus_global": {"point": e["rho_bar"]["point"] - arms["global"]["rho_bar"]["point"],
                                            "ci": L0.ci(e["_boot"] - arms["global"]["_boot"])},
                           "reference": r in (*REFERENCE_RULES, "row_best")}
    for h in (1, 2, 3):
        mask = hop == h
        _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
        out["by_hop"][f"hop={h}"] = {r: F12.read_arm(M, T, G, s_dens, s_read, W, mask)["rho_bar"]["point"] for r, M in rules.items()}
    # how well the deployable statistics separate right from wrong argmax chains on the dev rows
    sep = {}
    for key in ("conf", "margin", "p_null", "entropy"):
        xs = np.concatenate([g[f"{key}_score"] for g in gs])
        rt = np.concatenate([g["right_score"] for g in gs]).astype(bool)
        cuts = np.quantile(xs, [1 / 3, 2 / 3])
        b = np.searchsorted(cuts, xs, side="right")
        sep[key] = {"right_share_by_tercile": [float(rt[b == t].mean()) for t in range(3)],
                    "auc_right": auc(xs[rt], xs[~rt])}
    out["separation"] = sep
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT / "diag_grid.json", out)
    log("gridread: " + ", ".join(f"{r} {out['rules'][r]['rho_bar']['point']:.3f}" for r in out["rules"])
        + f"; global equals dsh {out['global_equals_dsh']}; {out['seconds']}s")


# ── the read ─────────────────────────────────────────────────────────────────


def stage_read(host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    decl = P.load_declaration()
    P.HARD_STOP_DIR[0] = OUT
    P.route_stops()
    if host:
        L8.host_mode(decl, log)
    vs = L9.views(F12.DATA)
    base, v = vs["std"], vs["nb"]
    n, S = base.n_q, len(SEEDS)
    qm = base.q_metrics
    T = qm[:, [F12.FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, RI]
    G = qm[:, [F12.FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, RI]
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    truth = [tuple(L8.chain_tokens(chains[qt])) for qt in qt_names]
    hop = np.asarray(base.q_hop)
    # oracles (references, the hard bonus on a node set): NB-oracle is level 12's; STD-oracle the true chain's reach in
    # the std family (backtracking walks kept) from its better bucket; ANY-oracle the best of the four reach sets (nb b0,
    # nb b1, std b0, std b1) by Jaccard with the in-pool golds, ties in that order
    Mo, Ms, Ma = np.zeros((n, S, 3)), np.zeros((n, S, 3)), np.zeros((n, S, 3))
    any_pick = np.zeros(n, dtype=np.int64)
    std_rs_size, nb_rs_size = np.zeros(n), np.zeros(n)

    def hard(q, rs):
        bonus = np.zeros(int(v.q_pool_size[q]))
        bonus[rs] = L8.ORACLE_BONUS
        g, gt = v.gold_local(q), int(v.q_gold_total[q])
        return [[F12.rank_metrics(v.z(q, k) + bonus, g, gt)[m] for m in RET] for k in SEEDS]

    for q in range(n):
        steps = chains[qt_names[q]]
        rs, _b = L8.r_star(v, q, steps)
        rss, _bs = L8.r_star(base, q, steps)
        cand = [*L8.chain_reach(v, q, steps), *L8.chain_reach(base, q, steps)]
        g = v.gold_local(q)
        jac = [L8.jaccard(c, g) if g.size else 0.0 for c in cand]
        best = int(np.argmax(jac))   # the first of the largest
        any_pick[q], std_rs_size[q], nb_rs_size[q] = best, rss.size, rs.size
        Mo[q], Ms[q], Ma[q] = hard(q, rs), hard(q, rss), hard(q, cand[best])
    V = {name: np.full((n, S, 3), np.nan) for name in TRANSFORMS}
    per = {key: np.full((n, S), np.nan) for key in PAIR_KEYS}
    argmax = np.full((n, S), -3, dtype=np.int64)
    filed_dsh = np.full((n, S, 3), np.nan)
    units = {}
    for i, k in enumerate(SEEDS):
        js = OUT / "units" / f"k{k}.json"
        units[f"k{k}"] = json.loads(js.read_text(encoding="utf-8"))
        with np.load(OUT / "units" / f"k{k}.npz") as zf:
            if not np.array_equal(zf["q"], np.arange(n)):
                raise SystemExit(f"k{k}: not the dev rows, each once")
            for name in TRANSFORMS:
                V[name][:, i] = zf[f"V_{name}"]
            for key in PAIR_KEYS:
                per[key][:, i] = zf[key]
            argmax[:, i] = zf["argmax"]
        with np.load(F12.unit_paths(FIT, k)[0]) as zf:
            filed_dsh[:, i] = zf["metrics_dsh"][:, RI]
    dsh_equal_filed = bool(np.array_equal(V["dsh"], filed_dsh))
    arms = {name: F12.read_arm(M, T, G, dens, readable, W) for name, M in V.items()}
    for name, M in (("NB-oracle", Mo), ("STD-oracle", Ms), ("ANY-oracle", Ma)):
        arms[name] = F12.read_arm(M, T, G, dens, readable, W)
    out = {"queries": n, "readable": readable, "dsh_equal_filed": dsh_equal_filed, "variants": {}, "by_hop": {}, "units": units}
    for name, e in arms.items():
        out["variants"][name] = {"rho_bar": e["rho_bar"], "band": e["band"], "rho": {m: e["rho"][m]["point"] for m in RET},
                                 "gap_to_gnn": {m: {"point": e["gap_to_gnn"][m]["point"], "flag": e["gap_to_gnn"][m]["flag"]} for m in RET},
                                 "minus_dsh": {"point": e["rho_bar"]["point"] - arms["dsh"]["rho_bar"]["point"],
                                               "ci": L0.ci(e["_boot"] - arms["dsh"]["_boot"])}}
    allM = {**V, "NB-oracle": Mo, "STD-oracle": Ms, "ANY-oracle": Ma}
    for h in (1, 2, 3):
        mask = hop == h
        _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
        out["by_hop"][f"hop={h}"] = {name: F12.read_arm(M, T, G, s_dens, s_read, W, mask)["rho_bar"]["point"] for name, M in allM.items()}
    right = np.asarray([[L8.token_sequence(int(argmax[q, i])) == truth[q] for i in range(S)] for q in range(n)], dtype=bool)
    out["gap_split"] = {name: F10.gap_split(Mo, M, dens, readable, right, hop) for name, M in V.items()}

    def share_of(Ma, Mb, sel):
        """sum over the selected (row, seed) pairs of M(a) - M(b), over the whole population's summed denominator (so
        the cells of a split add up), the mean over the readable metrics."""
        if not readable:
            return None
        return float(np.mean([float(((Ma[:, :, RET.index(m)] - Mb[:, :, RET.index(m)]) * sel).sum()) / (S * float(dens[m].sum()))
                              for m in readable]))

    hop2 = np.repeat(hop[:, None], S, 1)
    rs_empty = per["rstar_size"] == 0
    cats = {"rstar": {"empty": rs_empty, "b0": (~rs_empty) & (per["bstar"] == 0), "b1": (~rs_empty) & (per["bstar"] == 1)},
            "te": {"b0": per["te_bucket"] == 0, "b1": per["te_bucket"] == 1, "not_seed": per["te_bucket"] == -1,
                   "not_in_pool": per["te_bucket"] == -2},
            "chain": {"right": right, "wrong": ~right}}
    cells = {}
    for cat, groups in cats.items():
        cells[cat] = {}
        for h in (1, 2, 3):
            for g_name, g_sel in groups.items():
                sel = g_sel & (hop2 == h)
                if not sel.any():
                    continue
                cells[cat][f"hop={h}/{g_name}"] = {
                    "pairs": int(sel.sum()), "right_share": float(right[sel].mean()),
                    "gnn_minus_oracle": share_of(G, Mo, sel), "oracle_minus_dsh": share_of(Mo, V["dsh"], sel),
                    "gnn_minus_std_oracle": share_of(G, Ms, sel), "gnn_minus_any_oracle": share_of(G, Ma, sel),
                    "gnn_minus": {name: share_of(G, M, sel) for name, M in V.items()},
                    "p_true": float(per["p_true"][sel].mean()), "p_bstar": float(per["p_bstar"][sel].mean()),
                    "p_b1": float(per["p_b1"][sel].mean()), "conf": float(per["conf"][sel].mean())}
    out["cells"] = cells
    out["gnn_minus_dsh_total"] = share_of(G, V["dsh"], np.ones((n, S), dtype=bool))
    out["any_oracle_pick"] = {f"hop={h}": {nm: float((any_pick[hop == h] == j).mean()) for j, nm in
                                           enumerate(("nb_b0", "nb_b1", "std_b0", "std_b1"))} for h in (1, 2, 3)}
    out["rstar_size_mean"] = {f"hop={h}": {"nb": float(nb_rs_size[hop == h].mean()), "std": float(std_rs_size[hop == h].mean())}
                              for h in (1, 2, 3)}
    out["pairs"] = {"rstar_empty": float(rs_empty.mean()), "rstar_from_b1": float(((~rs_empty) & (per["bstar"] == 1)).mean()),
                    "te_bucket": {str(b): float((per["te_bucket"] == b).mean()) for b in (-2, -1, 0, 1)},
                    "argmax_right": float(right.mean()),
                    "argmax_bucket_is_bstar_right": float(((argmax // TBL) == per["bstar"])[right & ~rs_empty].mean()),
                    "p_true_right_mean": float(per["p_true"][right].mean()), "p_true_wrong_mean": float(per["p_true"][~right].mean()),
                    "seeds_b0_mean": float(per["seeds_b0"].mean()), "seeds_b1_mean": float(per["seeds_b1"].mean())}
    qtype_idx = np.asarray(base.q_qtype)
    per_qt = {}
    for qi, qt in enumerate(base.meta["qtypes"]):
        mq = np.repeat((qtype_idx == qi)[:, None], S, 1)
        if not mq.any():
            continue
        per_qt[qt] = {"hop": int(hop[qtype_idx == qi][0]), "pairs": int(mq.sum()), "right_share": float(right[mq].mean()),
                      "gnn_minus_dsh": share_of(G, V["dsh"], mq), "gnn_minus_oracle": share_of(G, Mo, mq),
                      "gnn_minus_std_oracle": share_of(G, Ms, mq), "gnn_minus_any_oracle": share_of(G, Ma, mq),
                      "rstar_empty": float(rs_empty[mq].mean()),
                      "gnn_minus": {name: share_of(G, M, mq) for name, M in V.items() if name != "dsh"}}
    out["per_qtype"] = dict(sorted(per_qt.items(), key=lambda x: -(x[1]["gnn_minus_dsh"] or 0)))
    out["seconds"] = round(time.time() - t0, 1)
    L8.write_json(OUT / "diag.json", out)
    log("read: " + ", ".join(f"{nm} {out['variants'][nm]['rho_bar']['point']:.3f}" for nm in out["variants"])
        + f"; dsh equal filed {dsh_equal_filed}; {out['seconds']}s")


# ── the count-weighted mixture (after read and gridread) ─────────────────────

MIX_ALPHAS = (0.5, 1.0)
MIX_POSTERIORS = ("dsh", "b1drop")


def mixture_alpha(fx, q: int, logp_q: np.ndarray, alpha: float) -> np.ndarray:
    """Level 8's mixture with each type's reach set weighted by its walk counts: n_q (sum_tau p(tau | q) c_tau(v)^alpha /
    sum_u c_tau(u)^alpha + p_null / n_q). alpha = 0 is Fitter.mixture's uniform reach set (checked bit for bit)."""
    node, count, tl = fx.data.entries(q)
    n = int(fx.data.q_pool_size[q])
    p = np.exp(logp_q)
    w = count.astype(np.float64) ** alpha
    norm = np.bincount(tl, weights=w, minlength=p.size - 1)
    m = np.bincount(node, weights=p[:-1][tl] * w / norm[tl], minlength=n) + p[-1] / n
    return n * m


def score_dsh_mix(fx, k: int, cap: dict, alpha: float) -> dict:
    """Level 11's score_dsh line for line, with mixture_alpha in place of Fitter.mixture."""
    data = fx.data
    inner_q, score_q = cap["inner_q"], cap["score_q"]
    base = [(data.gold_local(int(q)), int(data.q_gold_total[q]), data.z(int(q), k)) for q in inner_q]
    grid = {}
    for beta in F11.BETAS:
        nms = [mixture_alpha(fx, int(q), F11.tempered(lp, beta), alpha) for q, lp in zip(inner_q, cap["lp_inner"])]
        for kappa in F11.KAPPAS:
            for eta in F11.ETAS:
                grid[F11.grid_key(beta, kappa, eta)] = L8.mean3([F12.rank_metrics(z + kappa * np.log(nm + eta), g, gt)
                                                                for nm, (g, gt, z) in zip(nms, base)])
        del nms
    beta, kappa, eta = F11.choose(grid)
    metrics = []
    for q, lp in zip(score_q, cap["lp_score"]):
        q = int(q)
        s = data.z(q, k) + kappa * np.log(mixture_alpha(fx, q, F11.tempered(lp, beta), alpha) + eta)
        r = F12.rank_metrics(s, data.gold_local(q), int(data.q_gold_total[q]))
        metrics.append([r[m] for m in RET])
    return {"beta": beta, "kappa": kappa, "eta": eta, "inner_mean3": grid[F11.grid_key(beta, kappa, eta)],
            "metrics": np.asarray(metrics, dtype=np.float64).reshape(-1, len(RET))}


def stage_mix(k: int, post: str, host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    d_out = OUT / "units"
    npz, js = d_out / f"k{k}.mix_{post}.npz", d_out / f"k{k}.mix_{post}.json"
    if js.exists():
        log(f"{js} exists")
        return
    _decl, fx, _chains = setup(host, log)
    data = fx.data
    with np.load(d_out / f"k{k}.cap.npz") as zf:
        cap = {"inner_q": zf["inner_q"], "score_q": zf["score_q"], "lp_inner": unpack(zf["lp_inner"], zf["lp_inner_ptr"]),
               "lp_score": unpack(zf["lp_score"], zf["lp_score_ptr"])}
    if post == "b1drop":
        for key, qkey in (("lp_inner", "inner_q"), ("lp_score", "score_q")):
            cap[key] = [transform("b1drop", lp, np.asarray(data.t_code[data.type_rows(int(q))], dtype=np.int64), -1, -1, -2)[0]
                        for q, lp in zip(cap[qkey], cap[key])]
    same = all(np.array_equal(mixture_alpha(fx, int(q), F11.tempered(lp, beta), 0.0), fx.mixture(int(q), F11.tempered(lp, beta)))
               for beta in (1.0, float("inf")) for qkey, key in (("inner_q", "lp_inner"), ("score_q", "lp_score"))
               for q, lp in zip(cap[qkey], cap[key]))
    log(f"k{k} {post}: the count mixture at alpha 0 equals Fitter.mixture {same}")
    res, sel = {"q": np.asarray(cap["score_q"], dtype=np.int64)}, {}
    for alpha in MIX_ALPHAS:
        t1 = time.time()
        d = score_dsh_mix(fx, k, cap, alpha)
        res[f"V_a{alpha}"] = d["metrics"]
        sel[str(alpha)] = {"beta": F11.beta_label(d["beta"]), "kappa": d["kappa"], "eta": d["eta"], "inner_mean3": d["inner_mean3"],
                           "seconds": round(time.time() - t1, 1)}
        log(f"   k{k} {post} alpha {alpha}: beta/kappa/eta {sel[str(alpha)]['beta']}/{d['kappa']}/{d['eta']}, inner mean3 "
            f"{d['inner_mean3']:.4f}, {sel[str(alpha)]['seconds']}s")
    tmp = npz.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **res)
    os.replace(tmp, npz)
    L8.write_json(js, {"k": k, "posterior": post, "alpha0_equals_mixture": same, "selection": sel, **L8.job_fields(t0)})
    log(f"k{k} {post}: mix filed; {round(time.time() - t0)}s")


def share(G, M, dens, readable, sel) -> float | None:
    """stage_read's share_of: sum over the selected pairs of M(G) - M(arm), over the population's summed denominator."""
    if not readable:
        return None
    S = G.shape[1]
    return float(np.mean([float(((G[:, :, RET.index(m)] - M[:, :, RET.index(m)]) * sel).sum()) / (S * float(dens[m].sum()))
                          for m in readable]))


def stage_mixread(host: bool, log=L8.log_utc) -> None:
    t0 = time.time()
    decl = P.load_declaration()
    P.HARD_STOP_DIR[0] = OUT
    P.route_stops()
    if host:
        L8.host_mode(decl, log)
    vs = L9.views(F12.DATA)
    base, v = vs["std"], vs["nb"]
    n, S = base.n_q, len(SEEDS)
    qm = base.q_metrics
    T = qm[:, [F12.FUNCS.index(f"twin{k}") for k in SEEDS]][:, :, RI]
    G = qm[:, [F12.FUNCS.index(f"gnn{k}") for k in SEEDS]][:, :, RI]
    W = L0.boot_weights(n)
    _den, dens, readable = L8.denominators(T, G, W)
    chains = {qt: L8.true_chain(qt) for qt in base.meta["qtypes"]}
    qt_names = [base.meta["qtypes"][i] for i in base.q_qtype]
    hop = np.asarray(base.q_hop)
    V = {name: np.full((n, S, 3), np.nan) for name in ("dsh", "b1drop", "seqbktfix")}
    for post in MIX_POSTERIORS:
        for alpha in MIX_ALPHAS:
            V[f"{post}_a{alpha}"] = np.full((n, S, 3), np.nan)
    per = {key: np.full((n, S), np.nan) for key in ("seeds_b0", "seeds_b1", "rstar_size", "bstar", "te_bucket")}
    checks = {}
    for i, k in enumerate(SEEDS):
        with np.load(OUT / "units" / f"k{k}.npz") as zf:
            for name in ("dsh", "b1drop", "seqbktfix"):
                V[name][:, i] = zf[f"V_{name}"]
            for key in per:
                per[key][:, i] = zf[key]
        for post in MIX_POSTERIORS:
            info = json.loads((OUT / "units" / f"k{k}.mix_{post}.json").read_text(encoding="utf-8"))
            checks[f"k{k}/{post}"] = {"alpha0_equals_mixture": info["alpha0_equals_mixture"], "selection": info["selection"]}
            with np.load(OUT / "units" / f"k{k}.mix_{post}.npz") as zf:
                if not np.array_equal(zf["q"], np.arange(n)):
                    raise SystemExit(f"k{k} {post}: not the dev rows, each once")
                for alpha in MIX_ALPHAS:
                    V[f"{post}_a{alpha}"][:, i] = zf[f"V_a{alpha}"]
    Mo = np.zeros((n, S, 3))
    for q in range(n):
        rs, _b = L8.r_star(v, q, chains[qt_names[q]])
        bonus = np.zeros(int(v.q_pool_size[q]))
        bonus[rs] = L8.ORACLE_BONUS
        g, gt = v.gold_local(q), int(v.q_gold_total[q])
        Mo[q] = [[F12.rank_metrics(v.z(q, k) + bonus, g, gt)[m] for m in RET] for k in SEEDS]
    V["NB-oracle"] = Mo
    arms = {name: F12.read_arm(M, T, G, dens, readable, W) for name, M in V.items()}
    out = {"queries": n, "readable": readable, "checks": checks, "variants": {}, "by_hop": {}, "cells": {}, "per_qtype": {}}
    for name, e in arms.items():
        out["variants"][name] = {"rho_bar": e["rho_bar"], "rho": {m: e["rho"][m]["point"] for m in RET},
                                 "gap_to_gnn": {m: {"point": e["gap_to_gnn"][m]["point"], "flag": e["gap_to_gnn"][m]["flag"]} for m in RET},
                                 **{f"minus_{b}": {"point": e["rho_bar"]["point"] - arms[b]["rho_bar"]["point"],
                                                   "ci": L0.ci(e["_boot"] - arms[b]["_boot"])} for b in ("dsh", "b1drop")}}
    for h in (1, 2, 3):
        mask = hop == h
        _sd, s_dens, s_read = L8.denominators(T, G, W, mask)
        out["by_hop"][f"hop={h}"] = {name: F12.read_arm(M, T, G, s_dens, s_read, W, mask)["rho_bar"]["point"] for name, M in V.items()}
    hop2 = np.repeat(hop[:, None], S, 1)
    groups = {"b0_one_seed": per["seeds_b0"] == 1, "b0_two_seeds": per["seeds_b0"] >= 2,
              "rstar_b0_one": (per["rstar_size"] > 0) & (per["bstar"] == 0) & (per["seeds_b0"] == 1),
              "rstar_b0_two": (per["rstar_size"] > 0) & (per["bstar"] == 0) & (per["seeds_b0"] >= 2),
              "rstar_b1": (per["rstar_size"] > 0) & (per["bstar"] == 1), "rstar_empty": per["rstar_size"] == 0}
    for h in (1, 2, 3):
        for g_name, g_sel in groups.items():
            sel = g_sel & (hop2 == h)
            if sel.any():
                out["cells"][f"hop={h}/{g_name}"] = {"pairs": int(sel.sum()),
                                                     **{f"G-{name}": share(G, M, dens, readable, sel) for name, M in V.items()}}
    qidx = np.asarray(base.q_qtype)
    for qi, qt in enumerate(base.meta["qtypes"]):
        mq = np.repeat((qidx == qi)[:, None], S, 1)
        if mq.any():
            out["per_qtype"][qt] = {"pairs": int(mq.sum()), "b0_two_share": float((per["seeds_b0"][mq] >= 2).mean()),
                                    **{f"G-{name}": share(G, M, dens, readable, mq) for name, M in V.items()}}
    out["per_qtype"] = dict(sorted(out["per_qtype"].items(), key=lambda x: -(x[1]["G-dsh"] or 0)))
    out["seeds_b0_share"] = {f"hop={h}": float((per["seeds_b0"][hop == h, 0] >= 2).mean()) for h in (1, 2, 3)}
    out["seconds"] = round(time.time() - t0, 1)
    for e in arms.values():
        e.pop("_boot", None)
    L8.write_json(OUT / "diag_mix.json", out)
    log("mixread: " + ", ".join(f"{nm} {out['variants'][nm]['rho_bar']['point']:.3f}" for nm in out["variants"]) + f"; {out['seconds']}s")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("unit", "read", "grid", "gridread", "mix", "mixread"))
    ap.add_argument("--k", type=int, choices=tuple(SEEDS))
    ap.add_argument("--post", choices=MIX_POSTERIORS)
    ap.add_argument("--host", action="store_true")
    a = ap.parse_args()
    if a.stage in ("unit", "grid", "mix"):
        if a.k is None:
            ap.error(f"{a.stage} needs --k")
        if a.stage == "mix":
            if a.post is None:
                ap.error("mix needs --post")
            stage_mix(a.k, a.post, a.host)
        else:
            (stage_unit if a.stage == "unit" else stage_grid)(a.k, a.host)
    elif a.stage == "read":
        stage_read(a.host)
    elif a.stage == "mixread":
        stage_mixread(a.host)
    else:
        stage_gridread(a.host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
